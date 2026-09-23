"""Testes do shadow do Jev como classificador candidato do HAIR.

Contratos sob teste (em ordem de importância):

1. **Inerte por padrão**: desligado, não cria thread, não faz rede, não escreve.
2. **Nunca quebra um turno**: erro de rede/HTTP/parse/disco vira linha de erro
   enumerada, nunca exceção.
3. **Sem texto no log**: o arquivo de telemetria não pode conter a mensagem.
4. **Formato de fio correto**: o payload de ``/api/alpha/decisions`` é
   ``{model, state, questions}`` e a resposta é lida de ``answers.<nome>``,
   com fallback de slug quando o alias "latest" falha.
5. **Fora do caminho do turno**: ``spawn`` retorna imediatamente (thread daemon).
"""

from __future__ import annotations

import json
import threading
import urllib.error
from pathlib import Path

import pytest

from agent import jev_router_shadow as jrs

TIER_CFG = {
    "agent": {
        "adaptive_routing": {
            "enabled": True,
            "apply_routes": True,
            "shadow_mode": False,
            "tiers": {
                "local": [{"provider": "custom:local-qwen", "model": "qwen3.5-4b-local"}],
                "workhorse": [{"provider": "deepseek", "model": "deepseek-v4-flash"}],
                "premium": [{"provider": "openai-codex", "model": "gpt-5.6-sol"}],
            },
            "jev_shadow": {"enabled": True},
        }
    }
}

ANSWER_PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {"tier": {"type": "choice", "choice": "workhorse", "confidence": 0.91}},
    "usage": {"input_tokens": 300, "output_tokens": 40, "cost": 1.26e-05},
}


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _read_lines(home: Path) -> list:
    path = home / jrs.SHADOW_LOG_NAME
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── 1. inércia por padrão ────────────────────────────────────────────────────


def test_disabled_by_default_does_nothing(tmp_path, monkeypatch):
    called = {"n": 0}

    def _boom(*a, **kw):
        called["n"] += 1
        raise AssertionError("nao deveria chamar a rede quando desligado")

    monkeypatch.setattr(jrs, "_classify_remote", _boom)
    cfg = {"agent": {"adaptive_routing": {"enabled": True, "jev_shadow": {}}}}
    thread = jrs.spawn_jev_shadow(message="oi", config=cfg, home=tmp_path, join=True)
    assert thread is None
    assert called["n"] == 0
    assert _read_lines(tmp_path) == []


def test_missing_jev_section_is_disabled():
    cfg = jrs.load_shadow_config({"agent": {"adaptive_routing": {"enabled": True}}})
    assert cfg["enabled"] is False
    assert cfg["model"] == jrs.DEFAULT_MODEL


def test_config_normalisation_clamps_and_rejects_junk():
    cfg = jrs.load_shadow_config(
        {"agent": {"adaptive_routing": {"jev_shadow": {"enabled": "yes", "timeout_s": -9, "model": "  "}}}}
    )
    assert cfg["enabled"] is False  # só `True` boolean liga
    assert cfg["timeout_s"] == 2.0  # clamp inferior
    assert cfg["model"] == jrs.DEFAULT_MODEL  # string vazia cai no default
    cfg2 = jrs.load_shadow_config(
        {"agent": {"adaptive_routing": {"jev_shadow": {"enabled": True, "timeout_s": 99999}}}}
    )
    assert cfg2["enabled"] is True and cfg2["timeout_s"] == 120.0


def test_empty_or_non_text_message_never_calls(tmp_path, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(jrs, "_classify_remote", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    jrs.spawn_jev_shadow(message="   ", config=TIER_CFG, home=tmp_path, join=True)
    jrs.spawn_jev_shadow(message={"image": "x"}, config=TIER_CFG, home=tmp_path, join=True)
    assert called["n"] == 0


# ── 2/3. caminho feliz, sem texto no log ─────────────────────────────────────


def test_enabled_records_one_bounded_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "a", "workhorse": "b"})
    monkeypatch.setattr(
        jrs,
        "_classify_remote",
        lambda *a, **k: {
            "tier": "workhorse",
            "confidence": 0.91,
            "latency_ms": 412,
            "cost_usd": 1.26e-05,
            "served_model": "typesafe/jev-1.13-20260917",
        },
    )
    secret = "zqx9clienteconfidencial42"
    thread = jrs.spawn_jev_shadow(
        message=f"Renomeie essas variaveis do cliente {secret}",
        session_id="sess-1",
        config=TIER_CFG,
        home=tmp_path,
        join=True,
    )
    assert isinstance(thread, threading.Thread)
    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["jev_tier"] == "workhorse"
    assert rec["jev_confidence"] == pytest.approx(0.91)
    assert rec["cost_usd"] == pytest.approx(1.26e-05)
    assert rec["latency_ms"] == 412
    assert rec["served_model"].startswith("typesafe/jev")
    assert rec["agree"] is None  # heurística não informada neste chamador
    assert set(rec) == set(jrs._JSON_FIELDS)
    raw = (tmp_path / jrs.SHADOW_LOG_NAME).read_text(encoding="utf-8")
    assert secret not in raw
    assert "Renomeie" not in raw


def test_agreement_is_computed_when_heuristic_is_known(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "a", "workhorse": "b"})
    monkeypatch.setattr(
        jrs,
        "_classify_remote",
        lambda *a, **k: {"tier": "workhorse", "confidence": 0.8, "latency_ms": 300, "cost_usd": 0.0, "served_model": "m"},
    )
    jrs.spawn_jev_shadow(message="texto", heuristic_tier="workhorse", config=TIER_CFG, home=tmp_path, join=True)
    jrs.spawn_jev_shadow(message="texto", heuristic_tier="local", config=TIER_CFG, home=tmp_path, join=True)
    lines = _read_lines(tmp_path)
    assert [l["agree"] for l in lines] == [True, False]
    assert lines[0]["heur_tier"] == "workhorse"


def test_surface_and_session_ref_are_bounded_not_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "a", "workhorse": "b"})
    monkeypatch.setattr(
        jrs,
        "_classify_remote",
        lambda *a, **k: {"tier": "local", "confidence": 0.5, "latency_ms": 100, "cost_usd": 0.0, "served_model": "m"},
    )
    jrs.spawn_jev_shadow(
        message="texto",
        platform="signal",
        session_id="agent:main:signal:dm:+5562998005176",
        config=TIER_CFG,
        home=tmp_path,
        join=True,
    )
    rec = _read_lines(tmp_path)[0]
    assert rec["surface"] == "gateway"
    assert "5562998005176" not in json.dumps(rec)
    assert rec["session_ref"].startswith("s")


# ── 4. nunca quebra o turno ──────────────────────────────────────────────────


def test_provider_error_becomes_an_error_line_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "a", "workhorse": "b"})

    def _raise(*a, **kw):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(jrs, "_classify_remote", _raise)
    thread = jrs.spawn_jev_shadow(message="texto", config=TIER_CFG, home=tmp_path, join=True)
    assert isinstance(thread, threading.Thread)
    rec = _read_lines(tmp_path)[0]
    assert rec["error"] == "RuntimeError"
    assert rec["jev_tier"] == ""
    assert rec["agree"] is None


def test_missing_api_key_is_recorded_and_never_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "a", "workhorse": "b"})
    called = {"n": 0}
    monkeypatch.setattr(jrs, "_classify_remote", lambda *a, **k: called.__setitem__("n", 1))
    jrs.spawn_jev_shadow(message="texto", config=TIER_CFG, home=tmp_path, join=True)
    assert called["n"] == 0
    assert _read_lines(tmp_path)[0]["error"] == "no_api_key"


def test_single_tier_install_is_not_comparable(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "só isso"})
    jrs.spawn_jev_shadow(message="texto", config=TIER_CFG, home=tmp_path, join=True)
    assert _read_lines(tmp_path)[0]["error"] == "insufficient_tiers"


def test_log_failure_is_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "a", "workhorse": "b"})
    monkeypatch.setattr(
        jrs,
        "_classify_remote",
        lambda *a, **k: {"tier": "local", "confidence": 0.4, "latency_ms": 90, "cost_usd": 0.0, "served_model": "m"},
    )
    result = jrs.record_shadow_comparison(home="\x00invalido", jev_tier="local")
    assert result is None  # engolido, sem exceção


# ── 4. formato de fio ────────────────────────────────────────────────────────


def test_wire_payload_and_parsing(monkeypatch):
    seen = {}

    def _fake_urlopen(request, timeout=None):
        seen["body"] = json.loads(request.data.decode())
        seen["url"] = request.full_url
        seen["auth"] = request.headers.get("Authorization")
        return _FakeResponse(ANSWER_PAYLOAD)

    monkeypatch.setattr(jrs.urllib.request, "urlopen", _fake_urlopen)
    out = jrs._classify_remote(
        "texto de teste",
        model="~typesafe/jev-latest",
        timeout_s=10,
        criteria={"local": "a", "workhorse": "b"},
        api_key="sk-test",
    )
    assert seen["url"] == jrs.ENDPOINT
    assert seen["body"]["model"] == "~typesafe/jev-latest"
    assert seen["body"]["state"] == "texto de teste"
    assert seen["body"]["questions"]["tier"]["type"] == "choice"
    assert seen["body"]["questions"]["tier"]["criteria"] == {"local": "a", "workhorse": "b"}
    assert seen["auth"] == "Bearer sk-test"
    assert out["tier"] == "workhorse"
    assert out["confidence"] == pytest.approx(0.91)
    assert out["served_model"].startswith("typesafe/jev")
    assert out["latency_ms"] >= 0


def test_first_slug_failure_falls_back_to_the_next(monkeypatch):
    calls = {"n": 0}

    def _fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)
        return _FakeResponse(ANSWER_PAYLOAD)

    monkeypatch.setattr(jrs.urllib.request, "urlopen", _fake_urlopen)
    out = jrs._classify_remote("t", model="~typesafe/jev-latest", timeout_s=5, criteria={"a": "b", "c": "d"}, api_key="k")
    assert calls["n"] == 2
    assert out["tier"] == "workhorse"


def test_all_slugs_failing_raises_once(monkeypatch):
    def _always_fail(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)

    monkeypatch.setattr(jrs.urllib.request, "urlopen", _always_fail)
    with pytest.raises(RuntimeError):
        jrs._classify_remote("t", model="x", timeout_s=5, criteria={"a": "b", "c": "d"}, api_key="k")


# ── apoio ────────────────────────────────────────────────────────────────────


def test_tier_criteria_only_lists_configured_tiers():
    criteria = jrs.tier_criteria(TIER_CFG["agent"]["adaptive_routing"])
    assert set(criteria) == {"local", "workhorse", "premium"}
    assert "deepseek-v4-flash" in criteria["workhorse"]
    assert "multimodal" not in criteria


def test_summarize_counts_agreements_and_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(jrs, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(jrs, "tier_criteria", lambda config=None: {"local": "a", "workhorse": "b"})
    monkeypatch.setattr(
        jrs,
        "_classify_remote",
        lambda *a, **k: {"tier": "workhorse", "confidence": 0.9, "latency_ms": 200, "cost_usd": 1e-05, "served_model": "m"},
    )
    jrs.spawn_jev_shadow(message="t", heuristic_tier="workhorse", config=TIER_CFG, home=tmp_path, join=True)
    jrs.spawn_jev_shadow(message="t", heuristic_tier="premium", config=TIER_CFG, home=tmp_path, join=True)
    summary = jrs.summarize(home=tmp_path)
    assert summary["calls"] == 2
    assert summary["compared"] == 2
    assert summary["agreements"] == 1
    assert summary["cost_usd"] == pytest.approx(2e-05)
    assert summary["tiers"] == {"workhorse": 2}
