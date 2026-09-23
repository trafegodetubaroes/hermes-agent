"""Behavior tests for HAIR Phase 3 — controlled expansion.

Phase 3 keeps the router's Phase 2 contract (a sticky, first-turn-only decision
applied before agent construction) and adds the three gates that make expansion
safe, per surface:

* ``surfaces`` — allowlist of surfaces allowed to *apply* a route;
* ``budget`` — per-day cap of applied routes for a tier;
* ``data_policy`` — classes that must stay local, tiers never to be used.

Invariants under test: a denied gate never changes the surface's own route
(prompt cache and credentials stay untouched), a spent budget only ever walks
*down* the ladder, and telemetry stays bounded (no user text, enumerated fields).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent import adaptive_routing as ar

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

TRIVIAL_MESSAGE = "oi"
NORMAL_MESSAGE = (
    "Quero entender com calma como funciona a contabilidade da empresa e quais passos "
    "devo seguir para organizar melhor o acompanhamento mensal das despesas fixas e variaveis."
)
CRITICAL_MESSAGE = "Preciso pagar o boleto do cliente em producao sem perder nenhum dado."

ORIGINAL_MODEL = "gpt-5.6-sol"
ORIGINAL_PROVIDER = "openai-codex"


def config(**overrides) -> dict:
    section: dict = {
        "enabled": True,
        "apply_routes": True,
        "shadow_mode": False,
        "mode": "balanced",
        "max_escalations": 2,
        "tiers": TIERS,
    }
    section.update(overrides)
    return {"agent": {"adaptive_routing": section}}


def plan(message: str = NORMAL_MESSAGE, *, surface: str = "cli", budget_state=None, **overrides):
    return ar.plan_route_application(
        message=message,
        current_model=ORIGINAL_MODEL,
        current_runtime={"provider": ORIGINAL_PROVIDER},
        config=config(**overrides.pop("adaptive", {})) if "adaptive" in overrides else config(**overrides),
        explicit_pin=False,
        has_history=False,
        session_is_new=True,
        surface=surface,
        budget_state=budget_state,
    )


def signals(message: str, **kwargs) -> ar.TaskSignals:
    return ar.classify_task(message=message, **kwargs)


def write_records(home: Path, records) -> Path:
    path = home / ar.SHADOW_LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def record_for(tier: str, *, applied: bool = True, ts=None, **extra) -> dict:
    base = {
        "ts": float(ts if ts is not None else time.time()),
        "session_id": "sabc123",
        "platform": "cli",
        "tier": tier,
        "provider": "p",
        "model": "m",
        "mode": "balanced",
        "complexity_class": "NORMAL",
        "score": 2,
        "requires_vision": False,
        "reason_codes": ["default"],
        "escalation_chain": [],
        "effective_provider": "p",
        "effective_model": "m",
        "applied": applied,
        "matched": True,
        "surface": "cli",
        "router_version": ar.ROUTER_VERSION,
    }
    base.update(extra)
    return base


# ── config normalisation ─────────────────────────────────────────────────────


class TestConfigNormalisation:
    def test_surfaces_default_to_phase2_behaviour(self):
        cfg = ar.load_adaptive_config(config())
        assert cfg["surfaces"]["cli"] is True
        assert cfg["surfaces"]["gateway"] is True
        assert cfg["surfaces"]["cron"] is False
        assert cfg["surfaces"]["tui"] is False
        assert cfg["surfaces"]["delegation"] is False

    def test_surfaces_are_fail_closed_on_junk(self):
        cfg = ar.load_adaptive_config(
            config(surfaces={"cli": "yes", "cron": 1, "nonsense": True, "gateway": False})
        )
        # Non-boolean values keep the default; unknown surfaces are ignored.
        assert cfg["surfaces"]["cli"] is True
        assert cfg["surfaces"]["gateway"] is False
        assert cfg["surfaces"]["cron"] is False
        assert "nonsense" not in cfg["surfaces"]

    def test_budget_rejects_malformed_values(self):
        cfg = ar.load_adaptive_config(
            config(budget={"workhorse": 3, "premium": 0, "nope": 5, "local": -4, "free": True})
        )
        assert cfg["budget"] == {"workhorse": 3, "premium": 0}

    def test_data_policy_filters_to_known_enums(self):
        cfg = ar.load_adaptive_config(
            config(
                data_policy={
                    "local_only_classes": ["CRITICAL", "bogus", "critical"],
                    "forbidden_tiers": ["frontier", "not-a-tier"],
                }
            )
        )
        assert cfg["data_policy"]["local_only_classes"] == ("CRITICAL",)
        assert cfg["data_policy"]["forbidden_tiers"] == ("frontier",)

    def test_normalisation_is_idempotent(self):
        once = ar.load_adaptive_config(
            config(
                surfaces={"cli": True, "cron": True},
                budget={"workhorse": 2},
                data_policy={"forbidden_tiers": ["frontier"]},
            )
        )
        twice = ar.load_adaptive_config(once)
        assert twice == once


# ── surface allowlist ────────────────────────────────────────────────────────


class TestSurfaceAllowlist:
    def test_cli_surface_applies(self):
        result = plan(NORMAL_MESSAGE, surface="cli")
        assert result.should_apply is True
        assert result.tier == "workhorse"
        assert result.model == "deepseek-v4-flash"

    def test_gateway_surface_applies(self):
        result = plan(NORMAL_MESSAGE, surface="gateway")
        assert result.should_apply is True
        assert result.decision is not None
        assert ar._SURFACE_NOT_ALLOWED not in result.decision.reason_codes

    def test_cron_surface_is_denied_and_keeps_original_route(self):
        result = plan(NORMAL_MESSAGE, surface="cron")
        assert result.should_apply is False
        assert result.decision is not None
        assert ar._SURFACE_NOT_ALLOWED in result.decision.reason_codes
        # The surface's own route must be untouched: caches and credentials
        # stay exactly as they were resolved.
        assert result.model == ""
        assert result.provider == ""

    def test_unknown_surface_is_denied(self):
        result = plan(NORMAL_MESSAGE, surface="something_new")
        assert result.should_apply is False
        assert ar._SURFACE_NOT_ALLOWED in result.decision.reason_codes

    def test_surface_can_be_switched_on_by_config(self):
        result = plan(NORMAL_MESSAGE, surface="cron", surfaces={"cli": True, "gateway": True, "cron": True})
        assert result.should_apply is True
        assert result.tier == "workhorse"

    def test_gateway_can_be_switched_off_by_config(self):
        result = plan(NORMAL_MESSAGE, surface="gateway", surfaces={"cli": True, "gateway": False})
        assert result.should_apply is False
        assert ar._SURFACE_NOT_ALLOWED in result.decision.reason_codes

    def test_denied_surface_preserves_the_effective_route_identity(self):
        """The cache-preservation invariant: nothing but the route changes."""
        result = plan(NORMAL_MESSAGE, surface="cron")
        assert (result.model or None) != "deepseek-v4-flash"
        assert result.tier == ""
        assert result.reasoning_effort == ""

    def test_inactive_router_returns_the_plain_original_plan(self):
        result = plan(NORMAL_MESSAGE, surface="cli", enabled=False)
        assert result.should_apply is False
        assert result.decision is None
        assert result.model == ORIGINAL_MODEL
        assert result.provider == ORIGINAL_PROVIDER

    def test_history_never_applies(self):
        result = ar.plan_route_application(
            message=NORMAL_MESSAGE,
            current_model=ORIGINAL_MODEL,
            current_runtime={"provider": ORIGINAL_PROVIDER},
            config=config(),
            explicit_pin=False,
            has_history=True,
            session_is_new=True,
            surface="cli",
        )
        assert result.should_apply is False
        assert result.decision is None

    def test_explicit_pin_wins(self):
        result = ar.plan_route_application(
            message=NORMAL_MESSAGE,
            current_model=ORIGINAL_MODEL,
            current_runtime={"provider": ORIGINAL_PROVIDER},
            config=config(),
            explicit_pin=True,
            has_history=False,
            session_is_new=True,
            surface="cli",
        )
        assert result.should_apply is False
        assert result.model == ORIGINAL_MODEL


# ── daily budget ─────────────────────────────────────────────────────────────


class TestDailyBudget:
    def test_absent_tier_is_uncapped(self):
        state = {"date": "2026-09-23", "counts": {"workhorse": 99}}
        result = plan(NORMAL_MESSAGE, surface="cli", budget={"frontier": 1}, budget_state=state)
        assert result.should_apply is True
        assert result.tier == "workhorse"

    def test_zero_budget_downgrades_to_the_cheapest_funded_tier(self):
        state = {"date": "2026-09-23", "counts": {}}
        result = plan(NORMAL_MESSAGE, surface="cli", budget={"workhorse": 0}, budget_state=state)
        assert result.should_apply is True
        assert result.tier == "free"
        assert result.model == "nex-agi/nex-n2.5-mini:free"
        assert ar._BUDGET_DOWNGRADE in result.decision.reason_codes

    def test_spent_budget_downgrades(self):
        state = {"date": "2026-09-23", "counts": {"workhorse": 2}}
        result = plan(NORMAL_MESSAGE, surface="cli", budget={"workhorse": 2}, budget_state=state)
        assert result.should_apply is True
        assert result.tier == "free"

    def test_remaining_budget_still_applies(self):
        state = {"date": "2026-09-23", "counts": {"workhorse": 1}}
        result = plan(NORMAL_MESSAGE, surface="cli", budget={"workhorse": 2}, budget_state=state)
        assert result.should_apply is True
        assert result.tier == "workhorse"

    def test_budget_never_promotes(self):
        """A spent budget on the mapped tier must not push the route upwards."""
        state = {"date": "2026-09-23", "counts": {"workhorse": 5}}
        result = plan(NORMAL_MESSAGE, surface="cli", budget={"workhorse": 1}, budget_state=state)
        assert result.should_apply is True
        assert ar.TIER_ORDER.index(result.tier) < ar.TIER_ORDER.index("workhorse")

    def test_no_funded_cheaper_tier_denies_the_route(self):
        state = {"date": "2026-09-23", "counts": {}}
        result = plan(
            TRIVIAL_MESSAGE,
            surface="cli",
            budget={"local": 0, "free": 0},
            budget_state=state,
        )
        assert result.should_apply is False
        assert ar._BUDGET_EXHAUSTED in result.decision.reason_codes
        assert result.model == ""

    def test_vision_tier_is_denied_rather_than_downgraded(self):
        state = {"date": "2026-09-23", "counts": {}}
        result = ar.plan_route_application(
            message="descreva a imagem anexada",
            current_model=ORIGINAL_MODEL,
            current_runtime={"provider": ORIGINAL_PROVIDER},
            config=config(budget={"multimodal": 0}),
            explicit_pin=False,
            has_history=False,
            session_is_new=True,
            has_images=True,
            surface="cli",
            budget_state=state,
        )
        assert result.should_apply is False
        assert ar._BUDGET_EXHAUSTED in result.decision.reason_codes
        assert result.decision.tier == "multimodal"

    def test_budget_ignores_forbidden_tiers_when_downgrading(self):
        state = {"date": "2026-09-23", "counts": {}}
        result = plan(
            NORMAL_MESSAGE,
            surface="cli",
            budget={"workhorse": 0},
            data_policy={"forbidden_tiers": ["free"]},
            budget_state=state,
        )
        assert result.should_apply is True
        assert result.tier == "local"

    def test_load_budget_state_counts_only_applied_records_from_today(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            yesterday = time.time() - 86400 * 2
            write_records(
                home,
                [
                    record_for("workhorse", applied=True),
                    record_for("workhorse", applied=False),
                    record_for("premium", applied=True),
                    record_for("workhorse", applied=True, ts=yesterday),
                    {**record_for("free", applied=True), "ts": "not-a-number"},
                ],
            )
            path = home / ar.SHADOW_LOG_NAME
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("{not json\n")

            state = ar.load_budget_state(home=home)
        assert state["counts"] == {"workhorse": 1, "premium": 1}
        assert state["date"] == time.strftime("%Y-%m-%d")

    def test_load_budget_state_without_a_log_is_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            state = ar.load_budget_state(home=Path(tmp))
        assert state["counts"] == {}

    def test_load_budget_state_reads_the_tail_of_a_large_log(self):
        """Only the newest bytes are parsed, and the cut line is dropped."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            old = time.time() - 86400 * 3
            filler = [record_for("premium", applied=True, ts=old) for _ in range(3000)]
            write_records(home, filler)
            path = home / ar.SHADOW_LOG_NAME
            assert path.stat().st_size > ar._BUDGET_READ_BYTES
            write_records(home, [record_for("workhorse", applied=True)])
            state = ar.load_budget_state(home=home)
        assert state["counts"] == {"workhorse": 1}

    def test_budget_state_is_optional(self):
        result = plan(NORMAL_MESSAGE, surface="cli", budget={"workhorse": 0}, budget_state=None)
        assert result.should_apply is True
        assert result.tier == "workhorse"


# ── data policy ──────────────────────────────────────────────────────────────


class TestDataPolicy:
    def test_local_only_class_never_leaves_the_machine(self):
        decision = ar.resolve_route(
            signals(CRITICAL_MESSAGE),
            config=config(data_policy={"local_only_classes": ["CRITICAL"]}),
        )
        assert decision.tier == "local"
        assert decision.model == "qwen3.5-4b-local"
        assert decision.escalation_chain == ()
        assert ar._POLICY_LOCAL_ONLY in decision.reason_codes

    def test_local_only_class_without_a_local_tier_is_an_empty_route(self):
        tiers = {name: entries for name, entries in TIERS.items() if name != "local"}
        decision = ar.resolve_route(
            signals(CRITICAL_MESSAGE),
            config=config(tiers=tiers, data_policy={"local_only_classes": ["CRITICAL"]}),
        )
        assert decision.tier == ""
        assert decision.provider == ""
        assert ar._POLICY_LOCAL_ONLY_UNAVAILABLE in decision.reason_codes

    def test_forbidden_tier_is_never_selected(self):
        decision = ar.resolve_route(
            signals(CRITICAL_MESSAGE),
            config=config(data_policy={"forbidden_tiers": ["premium"]}),
        )
        # CRITICAL maps to premium in balanced mode; the router prefers the next
        # *stronger* configured tier rather than silently weakening the route.
        assert decision.tier == "frontier"
        assert ar._POLICY_TIER_FORBIDDEN in decision.reason_codes

    def test_forbidden_tiers_fall_down_the_ladder_when_none_stronger_exists(self):
        decision = ar.resolve_route(
            signals(CRITICAL_MESSAGE),
            config=config(data_policy={"forbidden_tiers": ["premium", "frontier"]}),
        )
        assert decision.tier == "workhorse"
        assert ar._POLICY_TIER_FORBIDDEN in decision.reason_codes

    def test_forbidden_tier_is_excluded_from_the_escalation_chain(self):
        decision = ar.resolve_route(
            signals(NORMAL_MESSAGE),
            config=config(data_policy={"forbidden_tiers": ["premium"]}),
        )
        assert decision.tier == "workhorse"
        pair = ("openai-codex", "gpt-5.6-sol")
        assert pair not in decision.escalation_chain
        assert ("openai-codex", "gpt-6-astra") in decision.escalation_chain

    def test_policy_does_not_override_an_explicit_pin(self):
        decision = ar.resolve_route(
            signals(CRITICAL_MESSAGE),
            config=config(data_policy={"local_only_classes": ["CRITICAL"]}),
            explicit_provider="anthropic",
            explicit_model="claude-sonnet",
        )
        assert decision.pinned is True
        assert decision.model == "claude-sonnet"

    def test_mode_is_still_respected_under_policy(self):
        economy = ar.resolve_route(
            signals(NORMAL_MESSAGE),
            config=config(mode="economy", data_policy={"forbidden_tiers": ["premium"]}),
        )
        assert economy.tier == "workhorse"

    def test_local_only_class_gate_applies_end_to_end(self):
        result = plan(CRITICAL_MESSAGE, surface="cli", data_policy={"local_only_classes": ["CRITICAL"]})
        assert result.should_apply is True
        assert result.tier == "local"
        assert ar._POLICY_LOCAL_ONLY in result.decision.reason_codes


# ── telemetry / end to end ───────────────────────────────────────────────────


class TestTelemetryContract:
    def test_recorded_line_carries_the_surface_and_no_user_text(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            decision = ar.resolve_route(signals(NORMAL_MESSAGE), config=config())
            path = ar.record_shadow_decision(
                decision,
                effective_provider="deepseek",
                effective_model="deepseek-v4-flash",
                session_id="sabc123",
                platform="cli",
                applied=True,
                surface="cli",
                home=home,
            )
            raw = Path(path).read_text(encoding="utf-8")
        record = json.loads(raw.strip())
        assert record["surface"] == "cli"
        assert record["applied"] is True
        assert record["matched"] is True
        assert record["router_version"] == ar.ROUTER_VERSION
        assert set(record) == set(ar._SHADOW_JSON_FIELDS)
        # No prompt text, no credentials: only enumerated/bounded fields.
        assert "contabilidade" not in raw
        assert "reason_codes" in record and isinstance(record["reason_codes"], list)

    def test_surface_is_bounded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            decision = ar.resolve_route(signals(TRIVIAL_MESSAGE), config=config())
            path = ar.record_shadow_decision(decision, surface="x" * 500, home=home)
            record = json.loads(Path(path).read_text(encoding="utf-8").strip())
        assert len(record["surface"]) == 32

    def test_applied_records_drive_the_next_budget_check(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = config(budget={"workhorse": 1})
            first = ar.plan_route_application(
                message=NORMAL_MESSAGE,
                current_model=ORIGINAL_MODEL,
                current_runtime={"provider": ORIGINAL_PROVIDER},
                config=cfg,
                explicit_pin=False,
                has_history=False,
                surface="cli",
                budget_state=ar.load_budget_state(home=home),
            )
            assert first.should_apply is True and first.tier == "workhorse"
            ar.record_shadow_decision(
                first.decision,
                effective_provider="deepseek",
                effective_model=first.model,
                applied=True,
                surface="cli",
                home=home,
            )
            second = ar.plan_route_application(
                message=NORMAL_MESSAGE,
                current_model=ORIGINAL_MODEL,
                current_runtime={"provider": ORIGINAL_PROVIDER},
                config=cfg,
                explicit_pin=False,
                has_history=False,
                surface="cli",
                budget_state=ar.load_budget_state(home=home),
            )
        assert second.tier == "free"
        assert ar._BUDGET_DOWNGRADE in second.decision.reason_codes

    def test_surfaces_do_not_share_state(self):
        state = {"date": time.strftime("%Y-%m-%d"), "counts": {"workhorse": 1}}
        cli_plan = plan(NORMAL_MESSAGE, surface="cli", budget={"workhorse": 1}, budget_state=state)
        cron_plan = plan(NORMAL_MESSAGE, surface="cron", budget={"workhorse": 1}, budget_state=state)
        assert cli_plan.tier == "free"
        assert cron_plan.tier == ""
        assert ar._SURFACE_NOT_ALLOWED in cron_plan.decision.reason_codes
        assert ar._BUDGET_DOWNGRADE not in cron_plan.decision.reason_codes
