"""Behavior tests for HAIR Phase 2 pre-construction adaptive routing."""

from __future__ import annotations

import json
import socket
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import adaptive_routing as ar
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


def _adaptive_config(
    *,
    enabled: bool = True,
    apply_routes: bool = True,
    shadow_mode: bool = False,
    tiers: dict | None = None,
) -> dict:
    if tiers is None:
        tiers = {
            "local": [
                {
                    "provider": "ollama",
                    "model": "qwen3:4b",
                    "reasoning_effort": "low",
                }
            ],
            "workhorse": [
                {
                    "provider": "openrouter",
                    "model": "workhorse/model",
                    "reasoning_effort": "medium",
                }
            ],
        }
    return {
        "agent": {
            "adaptive_routing": {
                "enabled": enabled,
                "apply_routes": apply_routes,
                "shadow_mode": shadow_mode,
                "mode": "balanced",
                "max_escalations": 2,
                "tiers": tiers,
            }
        }
    }


def _runtime(provider: str = "openrouter", *, key: str = "sk-original") -> dict:
    return {
        "api_key": key,
        "base_url": f"https://{provider}.example/v1",
        "provider": provider,
        "requested_provider": provider,
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }


def _plan(**overrides):
    args = {
        "message": "Renomeie essas variaveis",
        "current_model": "original/model",
        "current_runtime": _runtime(),
        "config": _adaptive_config(),
        "explicit_pin": False,
        "has_history": False,
        "has_images": False,
    }
    args.update(overrides)
    return ar.plan_route_application(**args)


@pytest.mark.parametrize(
    ("enabled", "apply_routes", "shadow_mode"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_application_gate_is_strict_and_inert(enabled, apply_routes, shadow_mode):
    plan = _plan(
        config=_adaptive_config(
            enabled=enabled,
            apply_routes=apply_routes,
            shadow_mode=shadow_mode,
        )
    )
    assert plan.should_apply is False
    assert (plan.model, plan.provider) == ("original/model", "openrouter")


def test_active_gate_selects_a_complete_candidate_without_mutating_runtime():
    original = _runtime()
    plan = _plan(current_runtime=original)

    assert plan.should_apply is True
    assert (plan.model, plan.provider, plan.tier) == (
        "qwen3:4b",
        "ollama",
        "local",
    )
    assert plan.reasoning_effort == "low"
    assert original == _runtime()


def test_apply_gate_off_remains_observe_only_even_when_shadow_flag_is_false():
    agent = SimpleNamespace(
        provider="openrouter",
        model="original/model",
        requested_provider="openrouter",
        requested_model="",
        session_id="observe-only-session",
        platform="cli",
    )
    config = _adaptive_config(apply_routes=False, shadow_mode=False)

    with patch.object(ar, "record_shadow_decision") as record:
        decision = ar.observe_shadow_route(
            agent=agent,
            user_message="Renomeie essas variaveis",
            conversation_history=[],
            config=config,
        )

    assert decision is not None
    assert decision.applied is False
    assert (agent.model, agent.provider) == ("original/model", "openrouter")
    record.assert_called_once()


@pytest.mark.parametrize("guard", ["explicit", "history", "resumed"])
def test_pins_and_existing_sessions_fail_open_to_original_route(guard):
    plan = _plan(
        explicit_pin=guard == "explicit",
        has_history=guard == "history",
        session_is_new=guard != "resumed",
    )
    assert plan.should_apply is False
    assert (plan.model, plan.provider) == ("original/model", "openrouter")


def test_malformed_route_returns_the_original_complete_route():
    plan = _plan(
        config=_adaptive_config(
            tiers={"local": [{"provider": "ollama", "model": ""}]}
        )
    )
    assert plan.should_apply is False
    assert (plan.model, plan.provider) == ("original/model", "openrouter")


def test_deterministic_application_plan_opens_no_sockets(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("deterministic routing must not touch the network")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)

    assert _plan().should_apply is True


def _cli_shell(**overrides):
    shell = SimpleNamespace(
        model="original/model",
        api_key="sk-original",
        base_url="https://openrouter.example/v1",
        provider="openrouter",
        requested_provider="openrouter",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        service_tier=None,
        config=_adaptive_config(),
        conversation_history=[],
        _resumed=False,
        _startup_route_explicit=False,
        session_id="cli-session",
    )
    for key, value in overrides.items():
        setattr(shell, key, value)
    return shell


def _resolved_ollama_runtime():
    return {
        "api_key": "no-key-required",
        "base_url": "http://127.0.0.1:11434/v1",
        "provider": "ollama",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "source": "provider",
    }


def test_cli_new_unpinned_route_is_resolved_before_build_and_stays_sticky():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    shell = _cli_shell()
    bound = CLIAgentSetupMixin._resolve_turn_agent_config.__get__(shell)

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=_resolved_ollama_runtime(),
    ) as resolve_runtime, patch.object(
        ar, "classify_task", wraps=ar.classify_task
    ) as classify:
        first = bound("Renomeie essas variaveis")
        shell.conversation_history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "done"},
        ]
        second = bound("Agora continue")

    assert first["model"] == "qwen3:4b"
    assert first["runtime"]["provider"] == "ollama"
    assert first["runtime"]["api_key"] == "no-key-required"
    assert first["signature"] == second["signature"]
    assert (shell.model, shell.requested_provider) == ("qwen3:4b", "ollama")
    resolve_runtime.assert_called_once_with(
        requested="ollama", target_model="qwen3:4b"
    )
    assert classify.call_count == 1


def test_cli_resumed_session_and_explicit_same_provider_model_pin_win():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    for shell in (
        _cli_shell(_resumed=True),
        _cli_shell(model="user/pinned-model", _startup_route_explicit=True),
    ):
        bound = CLIAgentSetupMixin._resolve_turn_agent_config.__get__(shell)
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as resolve_runtime:
            route = bound("Renomeie essas variaveis")
        assert route["model"] == shell.model
        assert route["runtime"]["provider"] == "openrouter"
        resolve_runtime.assert_not_called()


def test_cli_constructor_preserves_explicit_startup_route_intent():
    from cli import HermesCLI

    pinned = HermesCLI(
        model="user/pinned-model",
        provider="openrouter",
        compact=True,
    )
    unpinned = HermesCLI(compact=True)

    assert pinned._startup_route_explicit is True
    assert unpinned._startup_route_explicit is False


def test_cli_model_command_marks_the_route_as_explicit_before_first_user_turn():
    """A successful interactive /model choice must beat the router."""
    from cli import HermesCLI

    cli = HermesCLI(compact=True)
    cli.agent = None
    result = SimpleNamespace(
        success=True,
        new_model="user/pinned-model",
        target_provider="openrouter",
        api_key="sk-user",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
        model_info=None,
        warning_message="",
    )

    cli._apply_model_switch_result(result, persist_global=False)

    # Interactive /model pins THIS session; the process-level flag (set only by
    # --model/--provider at launch) is untouched so /new can release the pin.
    assert cli._session_route_explicit is True
    assert cli._startup_route_explicit is False
    plan = ar.plan_route_application(
        message="Renomeie essas variaveis",
        current_model=cli.model,
        current_runtime={"provider": cli.provider},
        config=_adaptive_config(),
        explicit_pin=bool(
            cli._startup_route_explicit or cli._session_route_explicit
        ),
        has_history=False,
    )
    assert plan.should_apply is False
    assert plan.model == "user/pinned-model"


def test_cli_candidate_auth_failure_fails_open_without_partial_mutation():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    shell = _cli_shell()
    original = (shell.model, shell.provider, shell.requested_provider, shell.api_key)
    bound = CLIAgentSetupMixin._resolve_turn_agent_config.__get__(shell)
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=RuntimeError("no target auth"),
    ), patch.object(ar, "record_shadow_decision") as telemetry:
        route = bound("Renomeie essas variaveis")

    assert route["model"] == "original/model"
    assert route["runtime"]["provider"] == "openrouter"
    assert (shell.model, shell.provider, shell.requested_provider, shell.api_key) == original
    assert telemetry.call_args.kwargs["applied"] is False
    assert telemetry.call_args.kwargs["effective_provider"] == "openrouter"
    assert telemetry.call_args.kwargs["effective_model"] == "original/model"


def test_cli_candidate_with_endpoint_but_no_credentials_fails_open():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    shell = _cli_shell()
    bound = CLIAgentSetupMixin._resolve_turn_agent_config.__get__(shell)
    credentialless = _resolved_ollama_runtime()
    credentialless["api_key"] = ""

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=credentialless,
    ):
        route = bound("Renomeie essas variaveis")

    assert route["model"] == "original/model"
    assert route["runtime"]["provider"] == "openrouter"
    assert getattr(shell, "_adaptive_route_owned", False) is False


def _gateway_source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _gateway_runner(source: SessionSource):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={source.platform: PlatformConfig(enabled=True, token="tok")}
    )
    runner._service_tier = None
    runner._session_model_overrides = {}
    runner.session_store = MagicMock()
    key = build_session_key(source)
    entry = SessionEntry(
        session_key=key,
        session_id="gateway-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=source.platform,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store._entries = {key: entry}
    return runner, key


def test_gateway_new_unpinned_route_is_sticky_before_cache_signature():
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=_resolved_ollama_runtime(),
    ) as resolve_runtime, patch.object(
        ar, "classify_task", wraps=ar.classify_task
    ) as classify:
        first = bound(
            "Renomeie essas variaveis",
            "original/model",
            _runtime(),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )
        second = bound(
            "continue",
            first["model"],
            first["runtime"],
            session_key=key,
            user_config=_adaptive_config(),
            has_history=True,
            source=source,
        )

    assert first["model"] == "qwen3:4b"
    assert first["runtime"]["provider"] == "ollama"
    assert first["signature"] == second["signature"]
    assert runner._session_model_overrides[key]["model"] == "qwen3:4b"
    assert runner._session_model_overrides[key]["adaptive_router"] is True
    assert runner._is_intentional_model_switch(key, "qwen3:4b") is True
    runner.session_store.set_model_override.assert_called_once()
    persisted = runner.session_store.set_model_override.call_args.args[1]
    assert persisted["model"] == "qwen3:4b"
    assert "api_key" not in persisted
    resolve_runtime.assert_called_once_with(
        "ollama", target_model="qwen3:4b"
    )
    assert classify.call_count == 1


def test_gateway_user_model_override_wins_and_is_not_overwritten():
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    user_override = {
        "model": "user/pinned-model",
        "provider": "openrouter",
        "api_key": "sk-user",
    }
    runner._session_model_overrides[key] = dict(user_override)
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)

    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider") as resolve_runtime:
        route = bound(
            "Renomeie essas variaveis",
            "user/pinned-model",
            _runtime(key="sk-user"),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )

    assert route["model"] == "user/pinned-model"
    assert runner._session_model_overrides[key] == user_override
    runner.session_store.set_model_override.assert_not_called()
    resolve_runtime.assert_not_called()


def test_gateway_api_request_surface_is_conservatively_skipped():
    from gateway.run import GatewayRunner

    source = _gateway_source(Platform.API_SERVER)
    runner, key = _gateway_runner(source)
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)

    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider") as resolve_runtime:
        route = bound(
            "Renomeie essas variaveis",
            "request/model",
            _runtime(),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )

    assert route["model"] == "request/model"
    assert not runner._session_model_overrides
    resolve_runtime.assert_not_called()


def test_gateway_candidate_auth_failure_fails_open_and_does_not_persist():
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=RuntimeError("missing target credentials"),
    ), patch.object(ar, "record_shadow_decision") as telemetry:
        route = bound(
            "Renomeie essas variaveis",
            "original/model",
            _runtime(),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )

    assert route["model"] == "original/model"
    assert route["runtime"]["provider"] == "openrouter"
    assert not runner._session_model_overrides
    runner.session_store.set_model_override.assert_not_called()
    assert telemetry.call_args.kwargs["applied"] is False
    assert telemetry.call_args.kwargs["effective_provider"] == "openrouter"
    assert telemetry.call_args.kwargs["effective_model"] == "original/model"


def test_gateway_candidate_with_endpoint_but_no_credentials_fails_open():
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)
    credentialless = _resolved_ollama_runtime()
    credentialless["api_key"] = ""

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=credentialless,
    ):
        route = bound(
            "Renomeie essas variaveis",
            "original/model",
            _runtime(),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )

    assert route["model"] == "original/model"
    assert route["runtime"]["provider"] == "openrouter"
    assert not runner._session_model_overrides


def test_gateway_persistence_failure_keeps_a_coherent_in_process_route():
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    runner.session_store.set_model_override.side_effect = OSError("read-only store")
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=_resolved_ollama_runtime(),
    ):
        route = bound(
            "Renomeie essas variaveis",
            "original/model",
            _runtime(),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )

    assert route["model"] == "qwen3:4b"
    assert route["runtime"]["provider"] == "ollama"
    assert runner._session_model_overrides[key]["model"] == "qwen3:4b"


def test_gateway_persists_non_secret_adaptive_ownership_marker():
    from gateway.session import sanitize_model_override

    cleaned = sanitize_model_override(
        {
            "model": "qwen3:4b",
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "must-not-persist",
            "adaptive_router": True,
        }
    )

    assert cleaned == {
        "model": "qwen3:4b",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "adaptive_router": "true",
    }
    assert "api_key" not in cleaned


def test_gateway_feature_flag_rollback_clears_only_router_owned_override():
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    adaptive_override = {
        "model": "qwen3:4b",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "adaptive_router": True,
    }
    runner._session_model_overrides[key] = dict(adaptive_override)
    bound = GatewayRunner._apply_session_model_override.__get__(runner)

    with patch(
        "agent.adaptive_routing.load_adaptive_config",
        return_value={"enabled": True, "apply_routes": False, "shadow_mode": True},
    ):
        model, runtime = bound(key, "original/model", _runtime())

    assert model == "original/model"
    assert runtime["provider"] == "openrouter"
    assert key not in runner._session_model_overrides
    runner.session_store.set_model_override.assert_called_once_with(key, None)

    user_override = {
        "model": "user/pinned-model",
        "provider": "openrouter",
    }
    runner._session_model_overrides[key] = dict(user_override)
    runner.session_store.set_model_override.reset_mock()
    model, runtime = bound(key, "original/model", _runtime())
    assert model == "user/pinned-model"
    assert runner._session_model_overrides[key] == user_override
    runner.session_store.set_model_override.assert_not_called()


def test_applied_telemetry_is_bounded_and_contains_no_user_text(tmp_path):
    secret = "private-user-text-zqx9f3a"
    plan = _plan(message=f"Renomeie essas variaveis {secret}")
    path = ar.record_shadow_decision(
        plan.decision,
        effective_provider=plan.provider,
        effective_model=plan.model,
        session_id="session-1",
        platform="cli",
        applied=True,
        home=tmp_path,
    )

    raw = open(path, encoding="utf-8").read()
    assert secret not in raw
    record = json.loads(raw)
    assert record["applied"] is True
    assert record["matched"] is True
    assert record["effective_provider"] == "ollama"
    assert record["effective_model"] == "qwen3:4b"


def test_default_config_has_separate_apply_gate_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    section = DEFAULT_CONFIG["agent"]["adaptive_routing"]
    assert section["enabled"] is False
    assert section["apply_routes"] is False


# ── review follow-ups: session pins, privacy, telemetry identity ─────────────


def test_cli_session_pin_is_released_by_new_session_but_startup_flag_persists():
    from cli import HermesCLI
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    cli = HermesCLI(compact=True)
    cli.agent = None
    cli._session_db = None
    # A real CLI reads the live config section; inject the adaptive section so
    # the test exercises routing rather than the default-off config.
    cli.config = _adaptive_config()
    cli.model = "original/model"
    cli.provider = "openrouter"
    cli.requested_provider = "openrouter"
    cli.api_key = "sk-original"
    cli.base_url = "https://openrouter.example/v1"
    cli.api_mode = "chat_completions"
    cli.conversation_history = []
    cli._resumed = False
    cli._adaptive_route_owned = False
    cli._startup_route_explicit = False
    cli._session_route_explicit = True
    bound = CLIAgentSetupMixin._resolve_turn_agent_config.__get__(cli)

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=_resolved_ollama_runtime(),
    ):
        pinned = bound("Renomeie essas variaveis")
    # Compare against literals: a routed call would rewrite cli.model/provider,
    # so self-comparison would pass even when the pin is ignored.
    assert pinned["model"] == "original/model"
    assert pinned["runtime"]["provider"] == "openrouter"
    assert cli.model == "original/model"
    assert cli.provider == "openrouter"

    with patch("hermes_cli.model_switch.switch_model") as switch:
        switch.return_value = SimpleNamespace(success=False)
        cli.new_session(silent=True)
    assert cli._session_route_explicit is False
    assert cli._adaptive_route_owned is False

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=_resolved_ollama_runtime(),
    ):
        routed = bound("Renomeie essas variaveis")
    assert routed["model"] == "qwen3:4b"
    assert routed["runtime"]["provider"] == "ollama"

    # A process-level --model/--provider is NOT session scoped.
    cli._startup_route_explicit = True
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=_resolved_ollama_runtime(),
    ):
        still_pinned = bound("Renomeie essas variaveis")
    assert still_pinned["model"] == cli.model


def test_gateway_routing_telemetry_never_records_the_chat_key():
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=_resolved_ollama_runtime(),
    ), patch.object(ar, "record_shadow_decision") as telemetry:
        route = bound(
            "Renomeie essas variaveis",
            "original/model",
            _runtime(),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )

    assert route["model"] == "qwen3:4b"
    recorded = telemetry.call_args.kwargs["session_id"]
    assert recorded
    assert recorded != key
    assert "c1" not in recorded and "u1" not in recorded
    assert ar.session_ref(key) == recorded
    assert source.chat_id not in recorded


def test_gateway_routed_runtime_keeps_the_operator_output_cap(monkeypatch):
    from gateway.run import GatewayRunner

    source = _gateway_source()
    runner, key = _gateway_runner(source)
    bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)
    monkeypatch.setenv("HERMES_MAX_TOKENS", "4321")
    capped = _resolved_ollama_runtime()
    capped["max_tokens"] = 4321

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=capped,
    ):
        route = bound(
            "Renomeie essas variaveis",
            "original/model",
            _runtime(),
            session_key=key,
            user_config=_adaptive_config(),
            has_history=False,
            source=source,
        )

    assert route["model"] == "qwen3:4b"
    assert route["runtime"]["max_tokens"] == 4321
    assert route["signature"] == route["signature"]


def test_privacy_local_only_contract_is_honoured_by_the_planner():
    config = _adaptive_config()
    config["agent"]["adaptive_routing"]["privacy"] = "local_only"
    config["agent"]["adaptive_routing"]["tiers"] = {
        "local": [{"provider": "ollama", "model": "qwen3:4b"}],
        "workhorse": [{"provider": "openrouter", "model": "workhorse/model"}],
    }

    # Even a long, code-heavy prompt (which would otherwise route to workhorse)
    # must stay on the local tier.
    plan = _plan(
        message="Refatore este modulo inteiro com testes e migracao " * 6,
        config=config,
    )
    assert plan.should_apply is True
    assert plan.tier == "local"
    assert plan.provider == "ollama"
    assert "privacy_local_only" in plan.decision.reason_codes

    # With no local tier configured the contract must not silently escalate.
    config["agent"]["adaptive_routing"]["tiers"] = {
        "workhorse": [{"provider": "openrouter", "model": "workhorse/model"}]
    }
    unavailable = _plan(message="Renomeie essas variaveis", config=config)
    assert unavailable.should_apply is False
    assert unavailable.model == "original/model"


def test_session_ref_is_stable_bounded_and_not_reversible():
    key = "telegram:dm:12345:user-1"

    assert ar.session_ref(key) == ar.session_ref(key)
    assert ar.session_ref(key) != ar.session_ref("telegram:dm:12345:user-2")
    assert key not in ar.session_ref(key)
    assert len(ar.session_ref(key)) <= 33
    assert ar.session_ref("") == ""
    assert ar.session_ref(None) == ""


def test_provider_scoped_runtime_resolution_honours_max_tokens_env(monkeypatch):
    """The provider-scoped resolver must mirror the default one's output cap."""
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider

    monkeypatch.setenv("HERMES_MAX_TOKENS", "7777")
    resolved = {
        "api_key": "sk-x",
        "base_url": "http://127.0.0.1:11434/v1",
        "provider": "ollama",
        "requested_provider": "ollama",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=resolved,
    ):
        runtime = _resolve_runtime_agent_kwargs_for_provider("ollama")

    assert runtime["max_tokens"] == 7777

    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)
    resolved["max_output_tokens"] = 512
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=resolved,
    ):
        fallback = _resolve_runtime_agent_kwargs_for_provider("ollama")
    assert fallback["max_tokens"] == 512
