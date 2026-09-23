"""TDD tests for the Phase 1 *shadow* adaptive router (``agent/adaptive_routing.py``).

Phase 1 contract being pinned here:

* Classification is deterministic, heuristic, bilingual (PT-BR + EN) and makes
  **zero LLM calls** — no network, no provider client, no model tool.
* Routing is a *suggestion* only: ``applied`` is always False and nothing in
  this feature may change the agent's effective provider/model.
* The whole feature is inert unless ``agent.adaptive_routing.enabled`` is true:
  no file, no log line, no DB write, no extra call — behavior is byte-identical
  to a build without the feature.
* Telemetry is bounded and enumerated: never user text, never prompt content.
* Fail-open: malformed/absent config must never raise and must never produce a
  half-empty route.

Written before the implementation on purpose (strict TDD): every test below is
expected to fail with ``ModuleNotFoundError``/``ImportError`` until
``agent/adaptive_routing.py`` exists.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import adaptive_routing as ar
from agent.adaptive_routing import (
    COMPLEXITY_CLASSES,
    MODES,
    TIER_ORDER,
    TaskSignals,
    build_escalation_chain,
    classify_task,
    load_adaptive_config,
    observe_shadow_route,
    record_shadow_decision,
    resolve_route,
    should_observe_first_turn,
)

SHADOW_FILENAME = "adaptive_routing_shadow.jsonl"


# ── helpers ──────────────────────────────────────────────────────────────────


def _tiers():
    """A fully-configured, realistic tier ladder (IDs come from config only)."""
    return {
        "local": [
            {"provider": "ollama", "model": "qwen3:4b", "reasoning_effort": "low"}
        ],
        "free": [
            {"provider": "openrouter", "model": "deepseek/deepseek-chat:free"}
        ],
        "workhorse": [
            {"provider": "anthropic", "model": "claude-sonnet-4.6"}
        ],
        "multimodal": [
            {"provider": "gemini", "model": "gemini-3-pro"}
        ],
        "premium": [
            {"provider": "openai", "model": "gpt-5.4"}
        ],
        "frontier": [
            {"provider": "openai", "model": "gpt-5.4-pro"}
        ],
    }


def _config(mode="balanced", max_escalations=2, tiers=None, enabled=True, shadow_mode=True):
    return {
        "enabled": enabled,
        "shadow_mode": shadow_mode,
        "mode": mode,
        "max_escalations": max_escalations,
        "tiers": _tiers() if tiers is None else tiers,
    }


def _home() -> Path:
    """The per-test sandboxed HERMES_HOME (see tests/conftest.py)."""
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    return home


def _write_config(adaptive: dict) -> Path:
    """Write a real ``config.yaml`` (JSON is valid YAML) into HERMES_HOME."""
    home = _home()
    path = home / "config.yaml"
    payload = {"agent": {"adaptive_routing": adaptive}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_shadow_lines(home) -> list:
    path = Path(home) / SHADOW_FILENAME
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _StubAgent:
    """Minimal agent surface the observer is allowed to read (nothing else)."""

    def __init__(self, *, provider="openrouter", model="test/model", platform="cli",
                 requested_provider="openrouter", requested_model=""):
        self.provider = provider
        self.model = model
        self.platform = platform
        self.session_id = "sess-shadow-1"
        self.requested_provider = requested_provider
        self.requested_model = requested_model
        self.switch_model = MagicMock()
        self.resolve_runtime_provider = MagicMock()


# ── 0. module shape ──────────────────────────────────────────────────────────


def test_public_constants_are_the_documented_enums():
    assert TIER_ORDER == (
        "local",
        "free",
        "workhorse",
        "multimodal",
        "premium",
        "frontier",
    )
    assert COMPLEXITY_CLASSES == ("TRIVIAL", "LOW", "NORMAL", "HIGH", "CRITICAL")
    assert MODES == ("economy", "balanced", "quality", "maximum")
    assert isinstance(ar.ROUTER_VERSION, str) and ar.ROUTER_VERSION


def test_module_has_no_llm_client_no_model_switch_and_no_fallback_wiring():
    """Static proof of constraints 2/3: pure stdlib, no switch_model, no fallback."""
    src = Path(ar.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "openai",
        "anthropic",
        "litellm",
        "requests",
        "httpx",
        "aiohttp",
        "urllib.request",
        "switch_model",
        "fallback_model",
        "_fallback_chain",
    ):
        assert forbidden not in src, f"adaptive_routing must not reference {forbidden!r}"


# ── 1. classification: TRIVIAL / LOW / NORMAL / HIGH / CRITICAL ──────────────


def test_trivial_short_command_is_trivial_and_routes_local():
    signals = classify_task(message="Renomeie essas variáveis")
    assert signals.complexity_class == "TRIVIAL"
    assert signals.score == 0
    assert signals.requires_vision is False
    assert "short_prompt" in signals.reason_codes

    decision = resolve_route(signals, config=_config())
    assert decision.tier == "local"
    assert (decision.provider, decision.model) == ("ollama", "qwen3:4b")
    assert decision.reasoning_effort == "low"
    assert decision.complexity_class == "TRIVIAL"
    assert decision.mode == "balanced"
    assert decision.pinned is False
    assert decision.applied is False
    assert decision.shadow is True


def test_critical_ptbr_payment_webhook_concurrency_request():
    message = (
        "O webhook de pagamento está com problema de concorrência e pode "
        "causar perda de dados de cliente em produção"
    )
    signals = classify_task(message=message)
    assert signals.complexity_class == "CRITICAL"
    assert signals.score == 4
    codes = set(signals.reason_codes)
    assert {"risk_payment", "risk_webhook", "risk_concurrency", "risk_data_loss"} <= codes

    decision = resolve_route(signals, config=_config())
    assert decision.tier == "premium"
    assert (decision.provider, decision.model) == ("openai", "gpt-5.4")


@pytest.mark.parametrize(
    "message",
    [
        "A migração vai tocar em billing e em dados de cliente",
        "Rode o backup antes do restore do banco de produção",
        "O login está quebrado e as permissões RBAC estão erradas",
        "Preciso garantir idempotência no webhook de cobrança",
        "Nunca exponha o segredo da API no log de produção",
    ],
)
def test_critical_signals_bilingual_risk_vocabulary(message):
    signals = classify_task(message=message)
    assert signals.complexity_class == "CRITICAL"
    assert signals.score == 4


@pytest.mark.parametrize(
    "message",
    [
        "Preciso desenhar a arquitetura distribuída desse serviço",
        "O bug é intermitente e já falhou três vezes",
        "There is a deadlock in the distributed worker pool",
        "Existe um vazamento de memória no worker de ingestão",
        "It works sometimes, cannot reproduce: intermittent failure",
    ],
)
def test_high_signals_bilingual_vocabulary(message):
    signals = classify_task(message=message)
    assert signals.complexity_class == "HIGH"
    assert signals.score == 3


def test_high_ptbr_architecture_and_intermittent_bug():
    message = (
        "Preciso desenhar a arquitetura distribuída desse serviço porque o "
        "bug é intermitente e não consigo reproduzir"
    )
    signals = classify_task(message=message)
    assert signals.complexity_class == "HIGH"
    assert signals.score == 3
    assert {"architecture", "intermittent"} <= set(signals.reason_codes)

    decision = resolve_route(signals, config=_config())
    assert decision.tier == "workhorse"
    assert (decision.provider, decision.model) == ("anthropic", "claude-sonnet-4.6")
    assert decision.reasoning_effort == "medium"


def test_prior_failure_marker_alone_is_high():
    signals = classify_task(message="Esse teste já falhou de novo, veja o log")
    assert signals.complexity_class == "HIGH"
    assert "prior_failure" in signals.reason_codes


def test_low_single_clear_verb_that_is_not_trivial():
    # Same verb family as TRIVIAL, but the session already has history, so the
    # "no history" clause of TRIVIAL does not hold → LOW.
    signals = classify_task(message="Resuma o conteúdo acima", history_len=4)
    assert signals.complexity_class == "LOW"
    assert signals.score == 1
    assert "has_history" in signals.reason_codes
    assert resolve_route(signals, config=_config()).tier == "local"


def test_normal_default_for_code_analysis_and_long_prompts():
    code_signals = classify_task(message="Revise o código do módulo de ingestão")
    assert code_signals.complexity_class == "NORMAL"
    assert code_signals.score == 2

    long_signals = classify_task(message="palavra " * 90)
    assert long_signals.complexity_class == "NORMAL"
    assert "long_prompt" in long_signals.reason_codes
    assert resolve_route(long_signals, config=_config()).tier == "workhorse"


def test_classification_is_deterministic_and_bilingual():
    ptbr = classify_task(message="Resuma este relatório")
    en = classify_task(message="Summarize this report")
    assert ptbr == classify_task(message="Resuma este relatório")
    assert ptbr.complexity_class == en.complexity_class
    assert ptbr.complexity_class == "TRIVIAL"


def test_reason_codes_are_bounded_enum_strings_and_never_echo_user_text():
    marker = "zqx-secret-token-42"
    signals = classify_task(
        message=f"O webhook de pagamento {marker} pode causar perda de dados"
    )
    assert signals.reason_codes
    for code in signals.reason_codes:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", code), code
        assert marker not in code
    assert len(set(signals.reason_codes)) == len(signals.reason_codes)
    # message text never leaks into the frozen signal record
    assert marker not in repr(signals)


def test_non_text_payload_is_classified_without_raising():
    signals = classify_task(
        message=[{"type": "text", "text": "Resuma"}, {"type": "image_url"}]
    )
    assert signals.complexity_class in COMPLEXITY_CLASSES
    assert "non_text_payload" in signals.reason_codes


# ── 2. vision ────────────────────────────────────────────────────────────────


def test_has_images_sets_requires_vision_and_selects_multimodal():
    signals = classify_task(message="Analise esse print da tela de checkout", has_images=True)
    assert signals.requires_vision is True
    assert "has_images" in signals.reason_codes

    decision = resolve_route(signals, config=_config())
    assert decision.tier == "multimodal"
    assert (decision.provider, decision.model) == ("gemini", "gemini-3-pro")


# ── 3. modes ─────────────────────────────────────────────────────────────────


def test_economy_mode_never_selects_premium_or_frontier_except_critical():
    normal = classify_task(message="Revise o código do módulo de ingestão")
    assert normal.complexity_class == "NORMAL"
    cheap = resolve_route(normal, config=_config(mode="economy"))
    assert cheap.tier not in ("premium", "frontier")
    assert cheap.mode == "economy"

    important = classify_task(
        message="O webhook de pagamento tem condição de corrida e perde dados"
    )
    assert important.complexity_class == "CRITICAL"
    allowed = resolve_route(important, config=_config(mode="economy"))
    assert allowed.tier == "premium"


def test_quality_mode_raises_the_floor_to_workhorse():
    trivial = classify_task(message="Renomeie essas variáveis")
    decision = resolve_route(trivial, config=_config(mode="quality"))
    assert decision.tier == "workhorse"
    assert (decision.provider, decision.model) == ("anthropic", "claude-sonnet-4.6")

    vision = classify_task(message="Analise esse print", has_images=True)
    vision_decision = resolve_route(vision, config=_config(mode="quality"))
    assert vision_decision.tier == "multimodal"


def test_maximum_mode_pushes_critical_and_high_to_frontier_and_normal_to_premium():
    critical = classify_task(
        message="A migração de billing em produção pode causar perda de dados"
    )
    assert resolve_route(critical, config=_config(mode="maximum")).tier == "frontier"

    high = classify_task(message="Preciso desenhar a arquitetura distribuída")
    assert resolve_route(high, config=_config(mode="maximum")).tier == "frontier"

    normal = classify_task(message="Revise o código do módulo de ingestão")
    assert resolve_route(normal, config=_config(mode="maximum")).tier == "premium"


def test_invalid_mode_argument_falls_back_to_balanced():
    signals = classify_task(message="Revise o código do módulo de ingestão")
    decision = resolve_route(signals, config=_config(mode="turbo-boost"))
    assert decision.mode == "balanced"
    assert decision.tier == "workhorse"


def test_shadow_flag_follows_config():
    signals = classify_task(message="Renomeie essas variáveis")
    assert resolve_route(signals, config=_config(shadow_mode=False)).shadow is False
    assert resolve_route(signals, config=_config(shadow_mode=True)).shadow is True


# ── 4. explicit pin ──────────────────────────────────────────────────────────


def test_explicit_pin_echoes_values_and_is_never_applied():
    signals = classify_task(message="Resuma o relatório")
    by_model = resolve_route(signals, config=_config(), explicit_model="gpt-5.4-pro")
    assert by_model.pinned is True
    assert by_model.applied is False
    assert by_model.shadow is False
    assert by_model.model == "gpt-5.4-pro"
    assert "explicit_pin" in by_model.reason_codes

    by_provider = resolve_route(signals, config=_config(), explicit_provider="anthropic")
    assert by_provider.pinned is True
    assert by_provider.provider == "anthropic"
    # A pinned route never suggests an escalation chain as authoritative.
    assert by_provider.escalation_chain == ()
    assert build_escalation_chain(by_provider, _config()) == []


def test_observer_reports_pinned_without_touching_the_agent():
    _write_config({**_config(enabled=True), "tiers": _tiers()})
    agent = _StubAgent(requested_model="gpt-5.4-pro")
    decision = observe_shadow_route(
        agent=agent, user_message="Resuma o relatório", conversation_history=[]
    )
    assert decision is not None
    assert decision.pinned is True
    assert decision.applied is False
    assert agent.model == "test/model"
    agent.switch_model.assert_not_called()
    agent.resolve_runtime_provider.assert_not_called()


# ── 5. escalation chain ──────────────────────────────────────────────────────


def _dedupe_tiers():
    """``multimodal`` duplicates the workhorse model to exercise dedupe."""
    tiers = _tiers()
    tiers["multimodal"] = [{"provider": "anthropic", "model": "claude-sonnet-4.6"}]
    return tiers


def test_escalation_chain_order_dedupe_truncation_and_exclusion():
    signals = classify_task(message="Renomeie essas variáveis")
    config = _config(max_escalations=4, tiers=_dedupe_tiers())
    decision = resolve_route(signals, config=config)

    assert decision.tier == "local"
    assert decision.escalation_chain == (
        ("openrouter", "deepseek/deepseek-chat:free"),  # free, ascending TIER_ORDER
        ("anthropic", "claude-sonnet-4.6"),            # workhorse
        # multimodal duplicates the workhorse model → deduped away
        ("openai", "gpt-5.4"),                          # premium
        ("openai", "gpt-5.4-pro"),                      # frontier
    )
    # selected route excluded
    assert ("ollama", "qwen3:4b") not in decision.escalation_chain

    truncated = resolve_route(
        signals, config=_config(max_escalations=2, tiers=_dedupe_tiers())
    )
    assert truncated.escalation_chain == (
        ("openrouter", "deepseek/deepseek-chat:free"),
        ("anthropic", "claude-sonnet-4.6"),
    )

    chain = build_escalation_chain(decision, config)
    assert [entry["provider"] for entry in chain] == [
        "openrouter",
        "anthropic",
        "openai",
        "openai",
    ]
    assert [entry["model"] for entry in chain] == [
        "deepseek/deepseek-chat:free",
        "claude-sonnet-4.6",
        "gpt-5.4",
        "gpt-5.4-pro",
    ]
    assert all("reasoning_effort" in entry for entry in chain)
    # dedupe held: the duplicated multimodal model never appears twice
    assert len(chain) == len({(e["provider"], e["model"]) for e in chain})


def test_escalation_chain_is_empty_for_the_top_tier():
    signals = classify_task(message="A migração em produção pode causar perda de dados")
    decision = resolve_route(signals, config=_config(mode="maximum"))
    assert decision.tier == "frontier"
    assert decision.escalation_chain == ()


# ── 6. privacy ───────────────────────────────────────────────────────────────


def test_privacy_local_only_forces_local_when_configured():
    signals = classify_task(message="Revise o código do módulo de ingestão")
    decision = resolve_route(signals, config=_config(), privacy="local_only")
    assert decision.tier == "local"
    assert (decision.provider, decision.model) == ("ollama", "qwen3:4b")
    assert decision.pinned is False
    assert decision.applied is False
    assert "privacy_local_only" in decision.reason_codes


def test_privacy_local_only_without_a_local_tier_is_a_safe_no_route():
    tiers = _tiers()
    tiers["local"] = []
    signals = classify_task(message="Revise o código do módulo de ingestão")
    decision = resolve_route(signals, config=_config(tiers=tiers), privacy="local_only")
    assert decision.provider == ""
    assert decision.model == ""
    assert decision.pinned is False
    assert decision.applied is False
    assert "privacy_local_only_unavailable" in decision.reason_codes


# ── 7. fail-open ─────────────────────────────────────────────────────────────


def test_empty_tiers_never_raise_and_never_return_a_half_empty_route():
    signals = classify_task(message="Renomeie essas variáveis")
    decision = resolve_route(signals, config=_config(tiers={}))
    assert decision.provider == ""
    assert decision.model == ""
    assert decision.pinned is False
    assert decision.applied is False
    assert decision.tier in TIER_ORDER or decision.tier == ""

    no_config_at_all = resolve_route(signals, config={})
    assert (no_config_at_all.provider, no_config_at_all.model) == ("", "")


def test_malformed_tier_entries_and_unknown_tier_names_are_ignored():
    tiers = {
        "local": [None, "junk", 42, {}, {"provider": "", "model": "x"},
                  {"provider": "ollama"}, {"provider": "ollama", "model": "qwen3:4b"}],
        "workhorse": "not-a-list",
        "bogus_tier": [{"provider": "nope", "model": "nope"}],
    }
    signals = classify_task(message="Renomeie essas variáveis")
    decision = resolve_route(signals, config=_config(tiers=tiers))
    assert (decision.provider, decision.model) == ("ollama", "qwen3:4b")
    assert decision.tier == "local"

    # workhorse is unconfigured (string) → fall through, never crash
    normal = classify_task(message="Revise o código do módulo de ingestão")
    fallback = resolve_route(normal, config=_config(tiers=tiers))
    assert fallback.provider and fallback.model
    assert fallback.tier in TIER_ORDER


def test_unconfigured_mapped_tier_falls_through_to_a_configured_tier():
    tiers = _tiers()
    tiers["workhorse"] = []
    normal = classify_task(message="Revise o código do módulo de ingestão")
    decision = resolve_route(normal, config=_config(tiers=tiers))
    assert decision.tier in TIER_ORDER
    assert decision.provider and decision.model
    assert decision.tier != "workhorse"
    assert "tier_fallback" in decision.reason_codes


def test_vision_without_a_multimodal_tier_still_routes_somewhere_real():
    tiers = _tiers()
    tiers["multimodal"] = []
    signals = classify_task(message="Analise esse print", has_images=True)
    decision = resolve_route(signals, config=_config(tiers=tiers))
    assert decision.provider and decision.model
    assert decision.tier in TIER_ORDER
    assert "tier_fallback" in decision.reason_codes


def test_none_and_garbage_config_arguments_never_raise():
    signals = classify_task(message="Renomeie essas variáveis")
    for bad in (None, "nonsense", 7, [], {"tiers": None}, {"tiers": {"local": None}}):
        decision = resolve_route(signals, config=bad)
        assert isinstance(decision.provider, str)
        assert isinstance(decision.model, str)
        assert decision.applied is False


# ── 8. config loading ────────────────────────────────────────────────────────


def test_load_adaptive_config_defaults_with_no_config_file():
    cfg = load_adaptive_config()
    assert cfg["enabled"] is False
    assert cfg["shadow_mode"] is True
    assert cfg["mode"] == "balanced"
    assert cfg["max_escalations"] == 2
    assert set(cfg["tiers"]) == set(TIER_ORDER)
    assert all(cfg["tiers"][tier] == () for tier in TIER_ORDER)


def test_load_adaptive_config_normalises_and_filters_malformed_values():
    cfg = load_adaptive_config(
        {
            "agent": {
                "adaptive_routing": {
                    "enabled": "yes-please",          # not a bool → default False
                    "shadow_mode": True,
                    "mode": "turbo",                  # invalid → balanced
                    "max_escalations": -3,            # invalid → 2
                    "tiers": {
                        "workhorse": [
                            {"provider": "anthropic", "model": "claude-sonnet-4.6"},
                            {"provider": "anthropic"},        # dropped: no model
                            "junk",                            # dropped: not a dict
                            {"provider": "x", "model": "y",
                             "reasoning_effort": "extreme"},   # effort → default
                        ],
                        "not_a_tier": [{"provider": "a", "model": "b"}],
                    },
                }
            }
        }
    )
    assert cfg["enabled"] is False
    assert cfg["shadow_mode"] is True
    assert cfg["mode"] == "balanced"
    assert cfg["max_escalations"] == 2
    assert "not_a_tier" not in cfg["tiers"]
    assert len(cfg["tiers"]["workhorse"]) == 2
    first, second = cfg["tiers"]["workhorse"]
    assert (first["provider"], first["model"]) == ("anthropic", "claude-sonnet-4.6")
    assert first["reasoning_effort"] == "medium"
    assert second["reasoning_effort"] == "medium"


@pytest.mark.parametrize("bad", [None, "nope", -1, True, 0.5, []])
def test_max_escalations_malformed_falls_back_to_two(bad):
    cfg = load_adaptive_config({"agent": {"adaptive_routing": {"max_escalations": bad}}})
    assert cfg["max_escalations"] == 2


def test_load_adaptive_config_reads_the_real_config_yaml():
    _write_config(
        {
            "enabled": True,
            "shadow_mode": False,
            "mode": "economy",
            "max_escalations": 1,
            "tiers": {"local": [{"provider": "ollama", "model": "qwen3:4b"}]},
        }
    )
    cfg = load_adaptive_config()
    assert cfg["enabled"] is True
    assert cfg["shadow_mode"] is False
    assert cfg["mode"] == "economy"
    assert cfg["max_escalations"] == 1
    assert cfg["tiers"]["local"][0]["model"] == "qwen3:4b"
    assert cfg["tiers"]["premium"] == ()

    # A route built from the real file respects the configured mode.
    normal = classify_task(message="Revise o código do módulo de ingestão")
    assert resolve_route(normal, config=cfg).mode == "economy"


def test_config_object_is_not_mutated_by_normalisation():
    raw = {
        "agent": {
            "adaptive_routing": {
                "mode": "quality",
                "tiers": {"local": [{"provider": "ollama", "model": "qwen3:4b"}]},
            }
        }
    }
    snapshot = json.dumps(raw)
    load_adaptive_config(raw)
    assert json.dumps(raw) == snapshot


# ── 9. first-turn gating ─────────────────────────────────────────────────────


def test_should_observe_first_turn_only_when_enabled_and_fresh():
    assert should_observe_first_turn(config=_config(enabled=False), has_history=False) is False
    assert should_observe_first_turn(config=_config(enabled=True), has_history=False) is True
    assert should_observe_first_turn(config=_config(enabled=True), has_history=True) is False
    assert should_observe_first_turn(config={}, has_history=False) is False
    assert (
        should_observe_first_turn(
            config=_config(enabled=True),
            has_history=False,
            explicit_provider="openai",
            explicit_model="gpt-5.4",
        )
        is True
    )


# ── 10. telemetry ────────────────────────────────────────────────────────────


def test_record_shadow_decision_writes_one_bounded_line_and_no_user_text(tmp_path):
    secret = "zqx9f3ausertext42"
    signals = classify_task(message=f"Resuma o relatório confidencial {secret}")
    decision = resolve_route(signals, config=_config())

    first = record_shadow_decision(
        decision,
        effective_provider="openrouter",
        effective_model="test/model",
        session_id="sess-1",
        platform="cli",
        home=tmp_path,
    )
    second = record_shadow_decision(decision, home=tmp_path)
    assert first and second
    assert Path(first) == Path(second) == Path(tmp_path) / SHADOW_FILENAME

    raw = (Path(tmp_path) / SHADOW_FILENAME).read_text(encoding="utf-8")
    assert secret not in raw
    lines = [line for line in raw.splitlines() if line.strip()]
    assert len(lines) == 2  # exactly one JSON line per call

    record = json.loads(lines[0])
    assert set(record) == {
        "ts",
        "session_id",
        "platform",
        "tier",
        "provider",
        "model",
        "mode",
        "complexity_class",
        "score",
        "requires_vision",
        "reason_codes",
        "escalation_chain",
        "effective_provider",
        "effective_model",
        "applied",
        "matched",
        "surface",
        "router_version",
    }
    assert isinstance(record["ts"], (int, float))
    assert record["tier"] == "local"
    assert record["provider"] == "ollama"
    assert record["model"] == "qwen3:4b"
    assert record["mode"] == "balanced"
    assert record["complexity_class"] == "TRIVIAL"
    assert record["score"] == 0
    assert record["requires_vision"] is False
    assert isinstance(record["reason_codes"], list)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in record["reason_codes"])
    assert record["escalation_chain"] == [
        "openrouter:deepseek/deepseek-chat:free",
        "anthropic:claude-sonnet-4.6",
    ]
    assert record["effective_provider"] == "openrouter"
    assert record["effective_model"] == "test/model"
    assert record["applied"] is False
    assert record["matched"] is False  # shadow choice != effective model
    assert record["router_version"] == ar.ROUTER_VERSION


def test_record_shadow_decision_matched_flag_is_true_on_agreement(tmp_path):
    signals = classify_task(message="Renomeie essas variáveis")
    decision = resolve_route(signals, config=_config())
    record_shadow_decision(
        decision, effective_provider="ollama", effective_model="qwen3:4b", home=tmp_path
    )
    record = _read_shadow_lines(tmp_path)[0]
    assert record["matched"] is True

    record_shadow_decision(
        decision, effective_provider="", effective_model="", home=tmp_path
    )
    assert _read_shadow_lines(tmp_path)[1]["matched"] is False


def test_record_shadow_decision_never_raises_and_returns_none_on_failure(tmp_path):
    signals = classify_task(message="Renomeie essas variáveis")
    decision = resolve_route(signals, config=_config())
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("nope", encoding="utf-8")
    assert record_shadow_decision(decision, home=blocker) is None
    # A garbage decision object still produces a bounded record, never a raise.
    assert isinstance(record_shadow_decision(object(), home=tmp_path), str)
    assert record_shadow_decision(None, home=tmp_path)
    assert record_shadow_decision(decision, effective_provider="p", home=tmp_path)


# ── 11. disabled is inert ────────────────────────────────────────────────────


def test_disabled_observer_returns_none_and_writes_nothing():
    _write_config(_config(enabled=False))
    # Warm the config layer first: ensure_hermes_home() scaffolding is a
    # pre-existing side effect of ANY config read, not of this feature. The
    # assertion below must therefore compare against a warmed baseline.
    assert ar.load_adaptive_config()["enabled"] is False
    home = _home()
    before = sorted(os.listdir(home))

    agent = _StubAgent()
    real_classify = ar.classify_task
    real_resolve = ar.resolve_route
    with patch.object(ar, "classify_task", MagicMock(wraps=real_classify)) as spy_classify, \
         patch.object(ar, "resolve_route", MagicMock(wraps=real_resolve)) as spy_resolve:
        result = observe_shadow_route(
            agent=agent, user_message="Renomeie essas variáveis", conversation_history=[]
        )

    assert result is None
    spy_classify.assert_not_called()
    spy_resolve.assert_not_called()
    assert SHADOW_FILENAME not in os.listdir(home)
    assert sorted(os.listdir(home)) == before
    agent.switch_model.assert_not_called()


def test_disabled_by_default_when_key_absent_from_config_yaml():
    _write_config({"shadow_mode": True, "mode": "balanced"})  # no "enabled" key
    agent = _StubAgent()
    assert (
        observe_shadow_route(
            agent=agent, user_message="Renomeie essas variáveis", conversation_history=[]
        )
        is None
    )
    assert not (Path(_home()) / SHADOW_FILENAME).exists()


def test_non_first_turn_is_not_observed_even_when_enabled():
    _write_config(_config(enabled=True))
    history = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "ok"}]
    assert (
        observe_shadow_route(
            agent=_StubAgent(), user_message="Renomeie essas variáveis",
            conversation_history=history,
        )
        is None
    )
    assert not (Path(_home()) / SHADOW_FILENAME).exists()


# ── 12. enabled observer: shadow only ────────────────────────────────────────


def test_enabled_observer_records_a_line_and_changes_nothing():
    _write_config(_config(enabled=True))
    agent = _StubAgent()

    decision = observe_shadow_route(
        agent=agent,
        user_message="Renomeie essas variáveis",
        conversation_history=[],
    )

    assert decision is not None
    assert decision.applied is False
    assert decision.tier == "local"
    # The agent's effective runtime is untouched.
    assert (agent.provider, agent.model) == ("openrouter", "test/model")
    agent.switch_model.assert_not_called()
    agent.resolve_runtime_provider.assert_not_called()

    records = _read_shadow_lines(_home())
    assert len(records) == 1
    assert records[0]["effective_provider"] == "openrouter"
    assert records[0]["effective_model"] == "test/model"
    assert records[0]["session_id"] == "sess-shadow-1"
    assert records[0]["platform"] == "cli"
    assert records[0]["matched"] is False


def test_observer_returns_none_on_a_broken_agent_and_never_raises():
    _write_config(_config(enabled=True))

    class _Exploding:
        def __getattr__(self, item):  # pragma: no cover - defensive
            raise RuntimeError("boom")

    assert (
        observe_shadow_route(
            agent=_Exploding(), user_message="Renomeie as variáveis", conversation_history=[]
        )
        is None
    )


def test_observer_with_malformed_config_tiers_still_records_safely():
    _write_config({"enabled": True, "tiers": {"workhorse": "junk", "local": [None]}})
    agent = _StubAgent()
    decision = observe_shadow_route(
        agent=agent, user_message="Renomeie essas variáveis", conversation_history=[]
    )
    assert decision is not None
    assert decision.provider == "" and decision.model == ""
    assert len(_read_shadow_lines(_home())) == 1


# ── 13. integration: build_turn_context observer call site ───────────────────


@pytest.fixture(autouse=True)
def _stub_aux_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main``.

    Mirrors the same stub in ``tests/agent/test_turn_context.py`` so the
    integration test here stays hermetic.
    """
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _build_with(agent, **overrides):
    from tests.agent.test_turn_context import _build

    return _build(agent, **overrides)


def _fake_agent():
    from tests.agent.test_turn_context import _FakeAgent

    return _FakeAgent()


def test_turn_context_with_routing_disabled_is_byte_identical_and_writes_no_file():
    _write_config(_config(enabled=False))
    home = _home()

    agent = _fake_agent()
    model_before, provider_before = agent.model, agent.provider
    real_classify = ar.classify_task
    with patch.object(ar, "classify_task", MagicMock(wraps=real_classify)) as spy_classify:
        _build_with(agent)

    assert (agent.model, agent.provider) == (model_before, provider_before)
    assert agent.model == "test/model" and agent.provider == "openrouter"
    assert SHADOW_FILENAME not in os.listdir(home)
    assert not (home / SHADOW_FILENAME).exists()
    spy_classify.assert_not_called()


def test_turn_context_with_routing_enabled_writes_shadow_line_and_keeps_model():
    _write_config(_config(enabled=True))

    agent = _fake_agent()
    model_before, provider_before = agent.model, agent.provider
    _build_with(agent)

    # SHADOW ONLY: the effective runtime is untouched.
    assert (agent.model, agent.provider) == (model_before, provider_before)
    assert agent.model == "test/model"
    assert agent.provider == "openrouter"

    records = _read_shadow_lines(_home())
    assert len(records) == 1
    assert records[0]["effective_provider"] == "openrouter"
    assert records[0]["effective_model"] == "test/model"
    assert records[0]["session_id"] == "sess-1"
    assert records[0]["platform"] == "cli"
    assert records[0]["tier"] == "local"  # "hello" is a TRIVIAL first turn
    assert records[0]["matched"] is False


# ── 14. constraint guards (added after the RED→GREEN cycle) ──────────────────


def test_observer_is_pure_and_never_touches_the_network(monkeypatch):
    """Zero LLM calls means zero network syscalls: break the socket module."""
    _write_config(_config(enabled=True))
    import socket

    def _boom(*_a, **_k):
        raise AssertionError("adaptive routing must never touch the network")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)

    agent = _StubAgent()
    decision = observe_shadow_route(
        agent=agent, user_message="Renomeie essas variáveis", conversation_history=[]
    )
    assert decision is not None
    assert decision.tier == "local"
    assert len(_read_shadow_lines(_home())) == 1


def test_no_hermes_env_var_can_enable_routing(monkeypatch):
    """Config lives in config.yaml: an env var must not flip the feature on."""
    _write_config(_config(enabled=False))
    for name in (
        "HERMES_ADAPTIVE_ROUTING",
        "HERMES_ADAPTIVE_ROUTING_ENABLED",
        "HERMES_SHADOW_ROUTING",
        "ADAPTIVE_ROUTING_ENABLED",
    ):
        monkeypatch.setenv(name, "1")

    agent = _StubAgent()
    assert (
        observe_shadow_route(
            agent=agent, user_message="Renomeie essas variáveis", conversation_history=[]
        )
        is None
    )
    assert not (Path(_home()) / SHADOW_FILENAME).exists()


def test_default_config_ships_the_adaptive_routing_block_disabled():
    """The shipped defaults exist (discoverable) but are off and empty."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    section = DEFAULT_CONFIG["agent"]["adaptive_routing"]
    assert section["enabled"] is False
    assert section["apply_routes"] is False
    assert section["shadow_mode"] is True
    assert section["mode"] == "balanced"
    assert section["max_escalations"] == 2
    assert set(section["tiers"]) == set(TIER_ORDER)
    # Tiers ship empty: real model IDs are install-specific, never hardcoded.
    assert all(section["tiers"][tier] == [] for tier in TIER_ORDER)


def test_escalation_chain_skips_multimodal_for_text_only_tasks():
    """A text-only task must never be told to escalate INTO the multimodal tier.

    The ladder orders ``multimodal`` right after ``workhorse``, so a naive
    "next configured tier" walk suggests a vision model for a pure-text task.
    Vision-only tiers are reachable only when the task actually needs vision.
    """
    config = _config(max_escalations=4, tiers=_tiers())
    multimodal_ids = {(e["provider"], e["model"]) for e in config["tiers"]["multimodal"]}
    assert multimodal_ids, "fixture must configure a multimodal tier"

    text_signals = TaskSignals(
        complexity_class="NORMAL",
        score=2,
        requires_vision=False,
        reason_codes=("code_intent",),
    )
    text_decision = resolve_route(text_signals, config=config)
    assert text_decision.tier == "workhorse"
    assert not (set(text_decision.escalation_chain) & multimodal_ids), (
        "text-only escalation chain leaked multimodal tier entries: "
        f"{text_decision.escalation_chain}"
    )

    vision_signals = TaskSignals(
        complexity_class="NORMAL",
        score=2,
        requires_vision=True,
        reason_codes=("has_images",),
    )
    vision_decision = resolve_route(vision_signals, config=config)
    assert vision_decision.tier == "multimodal"
