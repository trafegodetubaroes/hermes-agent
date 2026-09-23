"""Per-surface observation tests for HAIR adaptive routing.

Regression target: once ``apply_routes`` is live (``apply_routes: true`` +
``shadow_mode: false``), the shared prologue observer used to go inert for
*every* surface. CLI and gateway were fine — their pre-construction appliers
write the telemetry — but ``cron``, ``delegation``/``subagent``, ``tui`` and any
unmapped platform stopped appearing in ``adaptive_routing_shadow.jsonl`` at all,
which is exactly the evidence Phase 4 (evidence-driven optimisation) needs from
the surfaces that may NOT apply yet.

Contract pinned here:

* observation is per surface: a surface that owns its telemetry (it has an
  applier, or config authorises it to apply) is never also observed — the log
  must not gain a second line for the same first turn;
* every other surface keeps observing while application is live, recording a
  NON-applied decision annotated with the ``observe_only_surface`` reason code;
* with application off (shadow mode / ``apply_routes: false``) every surface
  observes, exactly as in Phase 1;
* the observer still changes nothing: no provider/model switch, no LLM call, no
  socket, no mutation of the agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import adaptive_routing as ar

SHADOW_FILENAME = ar.SHADOW_LOG_NAME

TIERS: dict = {
    "local": [
        {"provider": "custom:local-qwen", "model": "qwen3.5-4b-local", "reasoning_effort": "low"}
    ],
    "free": [
        {"provider": "openrouter", "model": "nex-agi/nex-n2.5-mini:free", "reasoning_effort": "low"}
    ],
    "workhorse": [
        {"provider": "deepseek", "model": "deepseek-v4-flash", "reasoning_effort": "medium"}
    ],
    "multimodal": [
        {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview", "reasoning_effort": "low"}
    ],
    "premium": [
        {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "high"}
    ],
    "frontier": [
        {"provider": "openai-codex", "model": "gpt-6-astra", "reasoning_effort": "high"}
    ],
}

#: A first turn that lands on the ``workhorse`` tier and is NOT the effective
#: route, so a record can never be a false "matched" positive.
NORMAL_MESSAGE = (
    "Quero entender com calma como funciona a contabilidade da empresa e quais passos "
    "devo seguir para organizar melhor o acompanhamento mensal das despesas fixas."
)

EFFECTIVE_PROVIDER = "openai-codex"
EFFECTIVE_MODEL = "gpt-5.6-sol"

#: The user's live configuration: application is ON, only cli+gateway may apply.
LIVE_SURFACES = {
    "cli": True,
    "gateway": True,
    "tui": False,
    "cron": False,
    "delegation": False,
}


def config(**overrides) -> dict:
    """The live install shape: application active, cli+gateway on the allowlist."""
    section: dict = {
        "enabled": True,
        "apply_routes": True,
        "shadow_mode": False,
        "mode": "balanced",
        "max_escalations": 2,
        "tiers": TIERS,
        "surfaces": dict(LIVE_SURFACES),
        "budget": {"premium": 10, "frontier": 5},
        "data_policy": {"local_only_classes": [], "forbidden_tiers": []},
    }
    section.update(overrides)
    return {"agent": {"adaptive_routing": section}}


def agent(platform: str) -> SimpleNamespace:
    return SimpleNamespace(
        provider=EFFECTIVE_PROVIDER,
        model=EFFECTIVE_MODEL,
        requested_provider=EFFECTIVE_PROVIDER,
        requested_model="",
        session_id="sess-surface-1",
        platform=platform,
        switch_model=MagicMock(),
        resolve_runtime_provider=MagicMock(),
    )


def home() -> Path:
    """The per-test sandboxed HERMES_HOME (see tests/conftest.py)."""
    from hermes_constants import get_hermes_home

    root = Path(get_hermes_home())
    root.mkdir(parents=True, exist_ok=True)
    return root


def records() -> list:
    path = home() / SHADOW_FILENAME
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def observe(platform: str, *, message: str = NORMAL_MESSAGE, cfg=None, history=None):
    return ar.observe_shadow_route(
        agent=agent(platform),
        user_message=message,
        conversation_history=[] if history is None else history,
        config=config() if cfg is None else cfg,
    )


# ── 1. platform → surface mapping ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("platform", "surface"),
    [
        ("cli", "cli"),
        ("gateway", "gateway"),
        ("tui", "tui"),
        ("cron", "cron"),
        # The delegation tool stamps child agents with ``platform="subagent"``.
        ("subagent", "delegation"),
        ("delegation", "delegation"),
        # A gateway agent carries the *channel* it serves, not the surface name.
        ("telegram", "gateway"),
        ("signal", "gateway"),
        ("discord", "gateway"),
        ("slack", "gateway"),
        # Graphical/other front ends: not a surface the config can name.
        ("desktop", ""),
        ("acp", ""),
        ("nonsense", ""),
        ("", ""),
    ],
)
def test_platform_maps_to_its_routing_surface(platform, surface):
    assert ar.surface_for_platform(platform) == surface


def test_platform_mapping_tolerates_case_spaces_and_junk():
    assert ar.surface_for_platform("  Telegram ") == "gateway"
    assert ar.surface_for_platform("CRON") == "cron"
    assert ar.surface_for_platform(None) == ""
    assert ar.surface_for_platform(123) == ""


def test_every_mapped_surface_is_a_configured_surface_name():
    for platform in ("cli", "gateway", "telegram", "tui", "cron", "subagent"):
        assert ar.surface_for_platform(platform) in ar.SURFACES


# ── 2. the cross-surface gate ────────────────────────────────────────────────


def test_should_observe_first_turn_is_decided_per_surface_while_application_is_live():
    cfg = config()
    # Surfaces that may not apply still need the observer: nothing else records
    # their first turn.
    for surface in ("cron", "delegation", "tui", ""):
        assert ar.should_observe_first_turn(
            config=cfg, has_history=False, surface=surface
        ) is True, surface
    # A surface that owns its telemetry must not be observed twice.
    for surface in ("cli", "gateway"):
        assert ar.should_observe_first_turn(
            config=cfg, has_history=False, surface=surface
        ) is False, surface
    # The first-turn and enabled gates still apply, per surface.
    assert (
        ar.should_observe_first_turn(config=cfg, has_history=True, surface="cron") is False
    )
    assert (
        ar.should_observe_first_turn(
            config=config(enabled=False), has_history=False, surface="cron"
        )
        is False
    )


def test_should_observe_first_turn_observes_every_surface_in_observe_mode():
    for overrides in ({"apply_routes": False}, {"shadow_mode": True}):
        cfg = config(**overrides)
        for surface in ar.SURFACES:
            assert ar.should_observe_first_turn(
                config=cfg, has_history=False, surface=surface
            ) is True, (overrides, surface)


def test_surface_allowed_to_apply_is_the_shared_fail_closed_allowlist():
    cfg = config()
    assert ar.surface_allowed_to_apply(cfg, "cli") is True
    assert ar.surface_allowed_to_apply(cfg, "gateway") is True
    for surface in ("tui", "cron", "delegation", "nonsense", "", None, 7):
        assert ar.surface_allowed_to_apply(cfg, surface) is False, surface
    # Non-boolean values keep the fail-closed default; unknown names are ignored.
    assert ar.surface_allowed_to_apply(config(surfaces={"cli": "yes", "cron": 1}), "cli") is True
    assert ar.surface_allowed_to_apply(config(surfaces={"cli": "yes", "cron": 1}), "cron") is False


def test_surface_owns_its_telemetry_only_for_appliers_and_allowlisted_surfaces():
    cfg = config()
    for surface in ("cli", "gateway"):
        assert ar.surface_owns_its_telemetry(cfg, surface) is True
    for surface in ("tui", "cron", "delegation", ""):
        assert ar.surface_owns_its_telemetry(cfg, surface) is False, surface
    # A surface the config authorises is expected to record its own route…
    assert ar.surface_owns_its_telemetry(config(surfaces={"cli": False, "cron": True}), "cron") is True
    # …and an applier surface keeps owning it even when the allowlist denies it,
    # because its applier records the denial itself (applied=false).
    assert ar.surface_owns_its_telemetry(config(surfaces={"cli": False}), "cli") is True


@pytest.mark.parametrize("surface", ["cli", "gateway", "tui", "cron", "delegation", "nonsense"])
def test_applier_denies_exactly_the_surfaces_the_allowlist_rejects(surface):
    """The applier and the observer must never disagree about "may apply"."""
    cfg = config()
    plan = ar.plan_route_application(
        message=NORMAL_MESSAGE,
        current_model=EFFECTIVE_MODEL,
        current_runtime={"provider": EFFECTIVE_PROVIDER},
        config=cfg,
        explicit_pin=False,
        has_history=False,
        session_is_new=True,
        surface=surface,
    )
    if ar.surface_allowed_to_apply(cfg, surface):
        assert plan.should_apply is True, surface
        assert plan.decision is not None
    else:
        assert plan.should_apply is False, surface
        assert plan.decision is not None
        assert "surface_not_allowed" in plan.decision.reason_codes


# ── 3. no duplicate record for a surface that applies ────────────────────────


@pytest.mark.parametrize("platform", ["cli", "gateway", "telegram", "signal", "whatsapp"])
def test_surface_that_applies_is_not_observed_while_application_is_live(platform):
    with patch.object(ar, "record_shadow_decision") as record:
        decision = observe(platform)

    assert decision is None
    record.assert_not_called()
    assert not (home() / SHADOW_FILENAME).exists()


def test_disallowed_applier_surface_is_still_owned_by_its_applier():
    """``surfaces.cli: false`` does not hand the CLI's telemetry to the observer.

    The CLI applier records the denied decision itself (``applied=false`` with
    ``surface_not_allowed``); observing here would log the same turn twice.
    """
    cfg = config(surfaces={"cli": False, "gateway": False})
    with patch.object(ar, "record_shadow_decision") as record:
        decision = observe("cli", cfg=cfg)

    assert decision is None
    record.assert_not_called()


# ── 4. surfaces that may not apply DO observe ────────────────────────────────


@pytest.mark.parametrize(
    ("platform", "surface"),
    [
        ("cron", "cron"),
        ("subagent", "delegation"),
        ("tui", "tui"),
        # Unmapped platforms are not on the allowlist either: they observe.
        ("desktop", ""),
        ("acp", ""),
    ],
)
def test_surface_that_may_not_apply_observes_while_application_is_live(platform, surface):
    agent_obj = agent(platform)
    decision = ar.observe_shadow_route(
        agent=agent_obj,
        user_message=NORMAL_MESSAGE,
        conversation_history=[],
        config=config(),
    )

    assert decision is not None
    assert decision.applied is False
    assert decision.tier == "workhorse"

    lines = records()
    assert len(lines) == 1
    record = lines[0]
    assert record["applied"] is False
    assert record["surface"] == surface
    assert record["platform"] == platform
    assert record["tier"] == "workhorse"
    assert record["provider"] == "deepseek"
    assert record["effective_provider"] == EFFECTIVE_PROVIDER
    assert record["effective_model"] == EFFECTIVE_MODEL
    assert record["matched"] is False
    assert record["reason_codes"][0] == "observe_only_surface"
    assert record["router_version"] == ar.ROUTER_VERSION

    # Observing changes nothing about the agent.
    assert (agent_obj.provider, agent_obj.model) == (EFFECTIVE_PROVIDER, EFFECTIVE_MODEL)
    agent_obj.switch_model.assert_not_called()
    agent_obj.resolve_runtime_provider.assert_not_called()


def test_observe_only_record_never_carries_user_text():
    secret = "zqx9f3ausertext42"
    decision = observe("cron", message=f"Resuma o relatório confidencial {secret}")

    assert decision is not None
    raw = (home() / SHADOW_FILENAME).read_text(encoding="utf-8")
    assert secret not in raw
    assert len(records()) == 1


def test_observing_surface_opens_no_socket(monkeypatch):
    import socket

    def _boom(*_a, **_k):
        raise AssertionError("adaptive routing must never touch the network")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)

    assert observe("cron") is not None
    assert len(records()) == 1


def test_observing_surface_still_respects_enabled_first_turn_and_failure_gates():
    # Disabled: inert, no file, no classification.
    real_classify = ar.classify_task
    with patch.object(ar, "classify_task", MagicMock(wraps=real_classify)) as spy:
        assert observe("cron", cfg=config(enabled=False)) is None
    spy.assert_not_called()
    assert not (home() / SHADOW_FILENAME).exists()

    # Not a first turn: never observed.
    assert observe("cron", history=[{"role": "user", "content": "oi"}]) is None
    assert not (home() / SHADOW_FILENAME).exists()

    # A broken agent still fails open, and never raises.
    class _Exploding:
        def __getattr__(self, item):  # pragma: no cover - defensive
            raise RuntimeError("boom")

    assert (
        ar.observe_shadow_route(
            agent=_Exploding(),
            user_message=NORMAL_MESSAGE,
            conversation_history=[],
            config=config(),
        )
        is None
    )


# ── 5. observe mode (application off) is unchanged ───────────────────────────


@pytest.mark.parametrize("platform", ["cli", "telegram", "cron", "subagent", "tui", "desktop"])
def test_every_surface_is_observed_while_application_is_off(platform):
    decision = observe(platform, cfg=config(apply_routes=False, shadow_mode=False))

    assert decision is not None
    assert decision.applied is False
    lines = records()
    assert len(lines) == 1
    # No "was not allowed to apply" annotation: nothing was applying at all.
    assert "observe_only_surface" not in lines[0]["reason_codes"]


# ── 6. integration: the shared prologue call site ────────────────────────────


@pytest.fixture(autouse=True)
def _stub_aux_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main``."""
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _write_live_config(adaptive: dict) -> None:
    path = home() / "config.yaml"
    path.write_text(json.dumps({"agent": {"adaptive_routing": adaptive}}), encoding="utf-8")


def _build_turn(agent_obj, **overrides):
    from tests.agent.test_turn_context import _build

    return _build(agent_obj, **overrides)


def _fake_agent(platform: str):
    from tests.agent.test_turn_context import _FakeAgent

    agent_obj = _FakeAgent()
    agent_obj.platform = platform
    return agent_obj


@pytest.mark.parametrize(("platform", "surface"), [("cron", "cron"), ("subagent", "delegation"), ("tui", "tui")])
def test_turn_context_observes_a_non_applying_surface_and_keeps_its_model(platform, surface):
    _write_live_config(config()["agent"]["adaptive_routing"])
    agent_obj = _fake_agent(platform)
    model_before, provider_before = agent_obj.model, agent_obj.provider
    assert ar.load_adaptive_config()["apply_routes"] is True

    _build_turn(agent_obj)

    assert (agent_obj.model, agent_obj.provider) == (model_before, provider_before)
    lines = records()
    assert len(lines) == 1
    assert lines[0]["surface"] == surface
    assert lines[0]["platform"] == platform
    assert lines[0]["applied"] is False
    assert lines[0]["effective_model"] == "test/model"


@pytest.mark.parametrize("platform", ["cli", "telegram"])
def test_turn_context_does_not_observe_a_surface_that_applies(platform):
    _write_live_config(config()["agent"]["adaptive_routing"])
    agent_obj = _fake_agent(platform)

    _build_turn(agent_obj)

    assert not (home() / SHADOW_FILENAME).exists()


def test_turn_context_writes_nothing_when_routing_is_disabled():
    _write_live_config(config(enabled=False)["agent"]["adaptive_routing"])
    # Warm the config layer first: ensure_hermes_home() scaffolding (SOUL.md,
    # skills/, …) is a pre-existing side effect of ANY config read, not of this
    # feature, so the snapshot below must be taken after it.
    assert ar.load_adaptive_config()["enabled"] is False
    agent_obj = _fake_agent("cron")
    before = sorted(os.listdir(home()))

    _build_turn(agent_obj)

    assert SHADOW_FILENAME not in os.listdir(home())
    assert sorted(os.listdir(home())) == before
