"""Jev (System One) como classificador candidato do HAIR — em SHADOW.

Compara, sem mudar NADA no turno, o tier escolhido pela heurística determinística
do HAIR com o tier que o Jev (modelo de decisões da TypeSafe, servido via
OpenRouter) escolheria para a mesma mensagem. É o passo de evidência que a Fase 4
exige antes de qualquer troca de classificador.

Invariantes de segurança deste módulo:

* **Inerte por padrão**: só age quando ``agent.adaptive_routing.jev_shadow.enabled``
  for explicitamente ``true`` em ``config.yaml``. Desligado, custo zero — sem
  thread, sem rede, sem arquivo.
* **Nunca no caminho do turno**: o trabalho roda numa thread daemon; um provedor
  lento não pode atrasar a resposta ao usuário.
* **Nunca levanta**: qualquer erro (rede, HTTP, parse, disco) é engolido; um
  *hint* de roteamento jamais quebra um turno.
* **Telemetria enumerada**: o log grava tier, confiança, latência, custo e o
  comprimento da mensagem — **nunca o texto da mensagem** nem credenciais.
* **Privacidade explícita**: a mensagem É enviada à TypeSafe (via OpenRouter)
  quando ligado. Essa é uma decisão de política de dados do dono da instalação,
  não um default — por isso o default é desligado.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Endpoint de decisões do OpenRouter. O endpoint de chat/completions RECUSA o
#: Jev com HTTP 400 ("is a decisions model and cannot be used with the
#: chat/completions endpoint"); o slug do alias "latest" exige o "~" literal.
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"
FALLBACK_MODELS: Tuple[str, ...] = ("typesafe/jev-1.13",)

#: Um JSON line por comparação, ao lado dos outros artefatos do HAIR.
SHADOW_LOG_NAME = "jev_router_shadow.jsonl"

#: Bump quando o prompt/critério mudar, para separar gerações de evidência.
SHADOW_VERSION = "jev-shadow-1"

_JSON_FIELDS: Tuple[str, ...] = (
    "ts",
    "shadow_version",
    "platform",
    "surface",
    "session_ref",
    "heur_tier",
    "jev_tier",
    "jev_confidence",
    "agree",
    "message_chars",
    "latency_ms",
    "cost_usd",
    "served_model",
    "error",
)

_TIER_ORDER: Tuple[str, ...] = (
    "local",
    "free",
    "workhorse",
    "multimodal",
    "premium",
    "frontier",
)


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _resolve_home() -> Optional[Path]:
    for module_name in ("hermes_cli.config", "hermes_constants"):
        try:
            module = importlib.import_module(module_name)
            resolver = getattr(module, "get_hermes_home", None)
            if callable(resolver):
                resolved = resolver()
                if resolved:
                    return Path(resolved)
        except Exception:
            continue
    env_home = os.environ.get("HERMES_HOME", "").strip()
    return Path(env_home) if env_home else None


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    root = _resolve_home()
    candidates = [root / ".env"] if root else []
    candidates.append(Path.home() / "AppData" / "Local" / "hermes" / ".env")
    for candidate in candidates:
        try:
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def load_shadow_config(config: Any = None) -> Dict[str, Any]:
    """Seção normalizada de ``agent.adaptive_routing.jev_shadow``.

    Aceita ``None`` (lê o config.yaml), um config completo ou a própria seção.
    Malformado nunca levanta: cai no default (desligado).
    """
    defaults: Dict[str, Any] = {
        "enabled": False,
        "model": DEFAULT_MODEL,
        "timeout_s": 20.0,
    }
    section = config
    try:
        if section is None:
            from agent.adaptive_routing import load_adaptive_config  # noqa: WPS433

            section = load_adaptive_config() or {}
        elif isinstance(section, dict) and "agent" in section:
            agent = section.get("agent")
            if isinstance(agent, dict):
                section = agent.get("adaptive_routing") or {}
        elif isinstance(section, dict) and "adaptive_routing" in section:
            section = section.get("adaptive_routing") or {}
        if not isinstance(section, dict):
            return defaults
        raw = section.get("jev_shadow")
        if not isinstance(raw, dict):
            return defaults
        enabled = raw.get("enabled")
        model = _clean(raw.get("model")) or DEFAULT_MODEL
        timeout = raw.get("timeout_s")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            timeout = defaults["timeout_s"]
        timeout = float(min(max(float(timeout), 2.0), 120.0))
        return {
            "enabled": enabled is True,
            "model": model,
            "timeout_s": timeout,
        }
    except Exception:
        return defaults


def tier_criteria(config: Any = None) -> Dict[str, str]:
    """Descrição de cada tier configurado, para as opções do ``choice``.

    Só tiers que existem de verdade na configuração entram como opção — o Jev
    nunca deve escolher um nível que esta instalação não tem.
    """
    criteria: Dict[str, str] = {}
    try:
        from agent.adaptive_routing import load_adaptive_config  # noqa: WPS433

        section = load_adaptive_config() if config is None else config
        if isinstance(section, dict) and "agent" in section:
            agent = section.get("agent")
            section = (agent or {}).get("adaptive_routing") if isinstance(agent, dict) else {}
        tiers = (section or {}).get("tiers") if isinstance(section, dict) else None
        if not isinstance(tiers, dict):
            return criteria
        for name in _TIER_ORDER:
            entries = tiers.get(name)
            if not entries:
                continue
            if isinstance(entries, dict):
                entries = [entries]
            models = [
                _clean(entry.get("model"))
                for entry in entries
                if isinstance(entry, dict)
            ]
            models = [m for m in models if m]
            if name == "multimodal":
                criteria[name] = "tarefa exige ver/interpretar imagem (" + ", ".join(models[:2]) + ")"
            elif models:
                criteria[name] = "tarefa de força '%s' (%s)" % (name, ", ".join(models[:2]))
            else:
                criteria[name] = "tarefa de força '%s'" % name
    except Exception:
        return criteria
    return criteria


def _classify_remote(
    message: str,
    *,
    model: str,
    timeout_s: float,
    criteria: Dict[str, str],
    api_key: str,
) -> Dict[str, Any]:
    """Uma chamada ao endpoint de decisões. Levanta em erro (chamador captura)."""
    questions = {
        "tier": {
            "type": "choice",
            "instructions": (
                "Qual nivel de modelo deve atender esta mensagem: o mais barato que "
                "da conta, ou um mais forte quando o risco/ambiguidade justifica"
            ),
            "criteria": criteria,
        }
    }
    body = json.dumps({"model": model, "state": message, "questions": questions}).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hermes.local",
        "X-Title": "Hermes HAIR jev shadow",
    }
    started = time.time()
    last_error = ""
    for candidate in (model, *FALLBACK_MODELS):
        payload_body = json.dumps(
            {"model": candidate, "state": message, "questions": questions}
        ).encode()
        request = urllib.request.Request(
            ENDPOINT, data=payload_body, headers=headers, method="POST"
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode())
            answers = (payload.get("answers") or {}).get("tier") or {}
            usage = payload.get("usage") or {}
            return {
                "tier": _clean(answers.get("choice")),
                "confidence": float(answers.get("confidence") or 0.0),
                "latency_ms": round((time.time() - started) * 1000),
                "cost_usd": float(usage.get("cost") or 0.0),
                "served_model": _clean(payload.get("model")),
            }
        except urllib.error.HTTPError as exc:  # slug inexistente: tenta o fallback
            last_error = "HTTP %s" % exc.code
            try:
                exc.read()
            except Exception:
                pass
        except Exception as exc:
            last_error = type(exc).__name__
    raise RuntimeError(last_error or "jev shadow failed")


def record_shadow_comparison(
    *,
    platform: Any = "",
    surface: Any = "",
    session_ref: Any = "",
    heur_tier: Any = "",
    jev_tier: Any = "",
    jev_confidence: Any = 0.0,
    latency_ms: Any = 0,
    cost_usd: Any = 0.0,
    served_model: Any = "",
    message_chars: Any = 0,
    error: Any = "",
    home: Any = None,
) -> Optional[str]:
    """Append de UMA linha enumerada (sem texto de mensagem). Nunca levanta."""
    try:
        heur = _clean(heur_tier)
        jev = _clean(jev_tier)
        record = {
            "ts": float(time.time()),
            "shadow_version": SHADOW_VERSION,
            "platform": _clean(platform)[:32],
            "surface": _clean(surface)[:32],
            "session_ref": _clean(session_ref)[:128],
            "heur_tier": heur[:32],
            "jev_tier": jev[:32],
            "jev_confidence": float(jev_confidence or 0.0),
            "agree": (heur == jev) if (heur and jev) else None,
            "message_chars": int(message_chars or 0),
            "latency_ms": int(latency_ms or 0),
            "cost_usd": float(cost_usd or 0.0),
            "served_model": _clean(served_model)[:64],
            "error": _clean(error)[:64],
        }
        root = Path(home) if home is not None else _resolve_home()
        if root is None:
            return None
        root.mkdir(parents=True, exist_ok=True)
        path = root / SHADOW_LOG_NAME
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(path)
    except Exception:
        return None


def _surface_for_platform(platform: Any) -> str:
    try:
        from agent.adaptive_routing import surface_for_platform  # noqa: WPS433

        return surface_for_platform(platform) or ""
    except Exception:
        name = _clean(platform).lower()
        return name if name in ("cli", "gateway", "tui", "cron", "delegation") else ""


def _session_ref(raw: Any) -> str:
    try:
        from agent.adaptive_routing import session_ref  # noqa: WPS433

        return session_ref(raw or "")
    except Exception:
        return ""


def _run_shadow(
    *,
    message: Any,
    platform: Any,
    heuristic_tier: Any,
    session_id: Any,
    cfg: Dict[str, Any],
    home: Any,
) -> None:
    text = message if isinstance(message, str) else ""
    chars = len(text)
    if not text:
        return
    api_key = _api_key()
    if not api_key:
        record_shadow_comparison(
            platform=platform,
            heur_tier=heuristic_tier,
            message_chars=chars,
            error="no_api_key",
            home=home,
        )
        return
    criteria = tier_criteria()
    if len(criteria) < 2:
        record_shadow_comparison(
            platform=platform,
            heur_tier=heuristic_tier,
            message_chars=chars,
            error="insufficient_tiers",
            home=home,
        )
        return
    try:
        result = _classify_remote(
            text[:4000],
            model=cfg["model"],
            timeout_s=cfg["timeout_s"],
            criteria=criteria,
            api_key=api_key,
        )
    except Exception as exc:
        record_shadow_comparison(
            platform=platform,
            heur_tier=heuristic_tier,
            message_chars=chars,
            error=type(exc).__name__,
            home=home,
        )
        return
    record_shadow_comparison(
        platform=platform,
        surface=_surface_for_platform(platform),
        session_ref=_session_ref(session_id),
        heur_tier=heuristic_tier,
        jev_tier=result.get("tier", ""),
        jev_confidence=result.get("confidence", 0.0),
        latency_ms=result.get("latency_ms", 0),
        cost_usd=result.get("cost_usd", 0.0),
        served_model=result.get("served_model", ""),
        message_chars=chars,
        home=home,
    )


def spawn_jev_shadow(
    *,
    message: Any,
    platform: Any = "",
    heuristic_tier: Any = "",
    session_id: Any = "",
    config: Any = None,
    home: Any = None,
    join: bool = False,
) -> Optional[threading.Thread]:
    """Dispara a comparação em background. Retorna a Thread (or ``None``).

    Devolve ``None`` — sem thread, sem rede, sem arquivo — quando o recurso está
    desligado, quando a mensagem não tem texto ou quando o config não carrega.
    ``join=True`` existe para testes determinísticos; em produção nunca se espera.
    """
    try:
        cfg = load_shadow_config(config)
        if not cfg["enabled"]:
            return None
        if not isinstance(message, str) or not message.strip():
            return None
        thread = threading.Thread(
            target=_run_shadow,
            kwargs={
                "message": message,
                "platform": platform,
                "heuristic_tier": heuristic_tier,
                "session_id": session_id,
                "cfg": cfg,
                "home": home,
            },
            name="jev-shadow",
            daemon=True,
        )
        thread.start()
        if join:
            thread.join(timeout=max(5.0, cfg["timeout_s"] + 2.0))
        return thread
    except Exception:
        return None


def summarize(home: Any = None) -> Dict[str, Any]:
    """Resumo do log de comparação, para o relatório de evidência."""
    out: Dict[str, Any] = {"calls": 0, "agreements": 0, "compared": 0, "cost_usd": 0.0, "tiers": {}}
    try:
        root = Path(home) if home is not None else _resolve_home()
        if root is None:
            return out
        path = root / SHADOW_LOG_NAME
        if not path.is_file():
            return out
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            out["calls"] += 1
            out["cost_usd"] += float(record.get("cost_usd") or 0.0)
            tier = record.get("jev_tier") or "-"
            out["tiers"][tier] = out["tiers"].get(tier, 0) + 1
            if record.get("agree") is not None:
                out["compared"] += 1
                out["agreements"] += 1 if record.get("agree") else 0
    except Exception:
        return out
    return out
