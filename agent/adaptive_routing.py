"""HAIR adaptive routing — pure decisions with surface-owned application.

This module answers one question per first turn: *"which tier would a router
have picked for this message?"* — and records the answer next to the provider
and model that were actually used. Phase 1 remains an observe-only mode;
Phase 2 lets CLI and gateway surfaces apply the route before construction;
Phase 3 makes that expansion explicit and bounded — per-surface allowlist,
per-tier daily budget and a per-class data policy:

* this module never switches a live agent or resolves credentials;
* it makes **zero** LLM/provider calls — classification is a pure heuristic
  over the message text (bilingual PT-BR + EN);
* it is opt-in and inert by default (``agent.adaptive_routing.enabled`` is
  false): no file, no log line, no extra work;
* it never raises — every public entry point is fail-open, because a routing
  *hint* must never be able to break a turn.

Design constraints encoded here:

* All provider/model IDs, modes and tier membership come from ``config.yaml``
  (``agent.adaptive_routing``) — nothing is hardcoded in the logic.
* Telemetry is bounded and enumerated: reason codes, a tier, a score, a
  boolean. No user text, no prompt, no tool arguments, no URLs, no keys.
* A malformed or partially configured tier ladder must degrade to *something
  real* (an existing configured tier) or to an explicitly empty route — never
  to a half-empty provider/model pair.

Escalation chains remain suggestions and are not wired into the runtime
fallback mechanism.

Phase 3 gates (all opt-in, all defaulting to today's behaviour):

* ``surfaces`` — allowlist of surfaces allowed to apply (default: cli+gateway);
* ``budget`` — per-day cap of applied routes per tier (``0`` blocks a tier);
* ``data_policy.local_only_classes`` — classes that must stay on the local tier;
* ``data_policy.forbidden_tiers`` — tiers never selected nor escalated into.

Observation is per surface: while application is live, a surface that owns its
own telemetry — the two appliers (``cli``/``gateway``, which record both applied
and denied decisions) and any surface the allowlist authorises to apply — is
never *also* observed, so one first turn can never produce two log lines. Every
other surface (``cron``, ``delegation``/``subagent``, ``tui``, an unmapped front
end) keeps observing, recording a non-applied decision tagged
``observe_only_surface`` — the evidence the next phase needs from exactly the
surfaces that may not apply yet. With application off (shadow mode or
``apply_routes: false``) every surface observes, as in Phase 1.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import time
import unicodedata
from hashlib import sha256
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Ordered tier ladder, weakest → strongest. Tier *membership* (which concrete
#: provider/model fills each tier) is user config, never source code.
TIER_ORDER: Tuple[str, ...] = (
    "local",
    "free",
    "workhorse",
    "multimodal",
    "premium",
    "frontier",
)

#: Complexity ladder produced by :func:`classify_task`.
COMPLEXITY_CLASSES: Tuple[str, ...] = (
    "TRIVIAL",
    "LOW",
    "NORMAL",
    "HIGH",
    "CRITICAL",
)

#: Routing modes. ``balanced`` is the default.
MODES: Tuple[str, ...] = ("economy", "balanced", "quality", "maximum")

#: Bumped whenever the classifier/route heuristics change, so shadow logs from
#: different generations stay distinguishable.
ROUTER_VERSION = "phase3-2"

#: One JSON line per observation, appended under HERMES_HOME.
SHADOW_LOG_NAME = "adaptive_routing_shadow.jsonl"

_SCORE_BY_CLASS: Dict[str, int] = {
    "TRIVIAL": 0,
    "LOW": 1,
    "NORMAL": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

#: Per-tier default reasoning effort, used when a tier entry omits the key.
_DEFAULT_EFFORT_BY_TIER: Dict[str, str] = {
    "local": "low",
    "free": "low",
    "workhorse": "medium",
    "multimodal": "medium",
    "premium": "high",
    "frontier": "high",
}

_VALID_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})

# Explicit privacy contracts the router understands. ``local_only`` pins the
# local tier so no prompt content can leave the machine.
PRIVACY_VALUES = frozenset({"normal", "local_only"})

#: Surfaces that know how to apply a route. Observing is independent of this
#: list: a surface absent here simply may not *change* the route.
SURFACES: Tuple[str, ...] = ("cli", "gateway", "tui", "cron", "delegation")

#: Surfaces allowed to apply when config does not say otherwise. Deliberately
#: the two surfaces exercised end-to-end in Phase 2; every other surface stays
#: observe-only until switched on explicitly (the Phase 3 allowlist).
_DEFAULT_SURFACES: Dict[str, bool] = {"cli": True, "gateway": True}

#: Surfaces whose own pre-construction applier writes the routing telemetry:
#: ``applied=True`` when it applied the route, ``applied=False`` with a gate
#: reason when a Phase 3 gate denied it. These are exactly the callers of
#: :func:`plan_route_application` (the CLI turn-config builder and
#: ``gateway/run.py``), so the shared prologue observer must stay out of them —
#: otherwise every first turn on those surfaces would be logged twice.
_APPLIER_SURFACES: Tuple[str, ...] = ("cli", "gateway")

#: ``platform`` values that mean the ``delegation`` surface. The delegation tool
#: stamps its child agents ``platform="subagent"``, which is not a surface name.
_DELEGATION_PLATFORMS: Tuple[str, ...] = ("subagent", "subagents", "delegate")

#: Platforms served by the gateway whose turns it never plans a route for
#: (``gateway/run.py`` treats them as pinned, request-oriented builders and
#: returns before any decision exists). Nothing else records them, so the shared
#: observer must keep doing it.
_GATEWAY_UNPLANNED_PLATFORMS: frozenset = frozenset({"local", "api_server", "webhook"})

#: Bounded tail read for the daily budget counter: telemetry is append-only, so
#: only the newest bytes can hold today's applied routes.
_BUDGET_READ_BYTES = 512 * 1024

_EXPLICIT_PIN = "explicit_pin"
_PRIVACY_LOCAL_ONLY = "privacy_local_only"
_PRIVACY_LOCAL_ONLY_UNAVAILABLE = "privacy_local_only_unavailable"
_TIER_FALLBACK = "tier_fallback"
_NO_TIER_CONFIGURED = "no_tier_configured"
_SHORT_PROMPT = "short_prompt"
_SIMPLE_VERB = "simple_verb"
_CODE_INTENT = "code_intent"
_LONG_PROMPT = "long_prompt"
_HAS_HISTORY = "has_history"
_HAS_IMAGES = "has_images"
_PRIOR_FAILURE = "prior_failure"
_NON_TEXT_PAYLOAD = "non_text_payload"
_DEFAULT_REASON = "default"
_POLICY_LOCAL_ONLY = "policy_local_only"
_POLICY_LOCAL_ONLY_UNAVAILABLE = "policy_local_only_unavailable"
_POLICY_TIER_FORBIDDEN = "policy_tier_forbidden"
_BUDGET_DOWNGRADE = "budget_downgrade"
_BUDGET_EXHAUSTED = "budget_exhausted"
_SURFACE_NOT_ALLOWED = "surface_not_allowed"
_OBSERVE_ONLY_SURFACE = "observe_only_surface"

_TRIVIAL_MAX_CHARS = 120
_LOW_MAX_CHARS = 320
_LONG_PROMPT_CHARS = 400
_MAX_CLASSIFY_CHARS = 4000

# ── heuristic vocabulary ─────────────────────────────────────────────────────
#
# Every pattern is matched against an ASCII-folded, lowercased copy of the
# message, so accented PT-BR and plain EN spellings share one keyword list.
# Word-boundary anchored: "list" matches "listar"/"lista" but not "palavra".

_RISK_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "risk_payment",
        (
            "pagamento", "payment", "pagar", "faturamento", "billing",
            "cobranca", "checkout", "stripe", "invoice", "nota fiscal",
            "boleto", "pix", "reembolso", "refund", "assinatura",
        ),
    ),
    ("risk_webhook", ("webhook", "callback de terceiro")),
    ("risk_idempotency", ("idempot", "exactly once", "duplicidade")),
    (
        "risk_auth",
        (
            "autentica", "authorization", "autorizacao", "authentication",
            "login", "logout", "senha", "password", "credencial", "credential",
            "oauth", "jwt", "token", "escalacao de privilegio", "privilege escalation",
        ),
    ),
    ("risk_rbac", ("rbac", "permissoes", "permission", "autorizacao de acesso", "scope")),
    (
        "risk_concurrency",
        (
            "concorr", "concurrency", "race condition", "condicao de corrida",
            "corrida", "mutex", "lock contention", "transacao simultanea",
        ),
    ),
    (
        "risk_data_loss",
        (
            "perda de dados", "data loss", "perder dados", "delete sem",
            "destruicao de dados", "irreversivel", "corrupcao",
        ),
    ),
    ("risk_migration", ("migracao", "migration", "migrate", "backfill de dados")),
    ("risk_backup", ("backup", "restore", "restauracao", "rollback de banco")),
    ("risk_production", ("producao", "production", "clientes reais", "live traffic")),
    (
        "risk_customer_data",
        (
            "dados de cliente", "dados do cliente", "customer data", "dados pessoais",
            "personal data", "pii", "titular dos dados",
        ),
    ),
    (
        "risk_secret",
        ("segredo", "secret", "api key", "chave de api", "credencial secreta", "vazamento de chave"),
    ),
    ("risk_lgpd", ("lgpd", "gdpr", "hipaa", "compliance regulatorio")),
)

_HIGH_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("architecture", ("arquitetura", "architecture", "design do sistema", "system design", "trade-off")),
    (
        "intermittent",
        (
            "intermitente", "intermittent", "flaky", "as vezes falha", "sometimes fails",
            "nao reproduz", "cannot reproduce", "sem reproducao",
        ),
    ),
    ("distributed", ("distribuid", "distributed", "cluster", "fila de mensagens", "message queue")),
    ("deadlock", ("deadlock", "impasse", "livre de deadlock")),
    ("leak", ("vazamento", "leak", "memory leak")),
    ("performance", ("performance", "desempenho", "latencia", "latency", "otimizacao de query", "lento demais")),
    ("refactor_broad", ("refatorar", "refactor", "reescrever o modulo", "reestruturar o sistema")),
    ("escalability", ("escalabilidade", "escalability", "escala horizontal", "throughput")),
)

_PRIOR_FAILURE_KEYWORDS: Tuple[str, ...] = (
    "ja falhou", "falhou de novo", "falhou novamente", "continua falhando",
    "ainda falha", "segue falhando", "already failed", "failed again",
    "still failing", "keeps failing", "continues failing", "failed twice",
)

_CODE_INTENT_KEYWORDS: Tuple[str, ...] = (
    "codigo", "codigos", "code", "codar", "implementar", "implement",
    "funcao", "funcoes", "function", "script", "scripts", "refatorar",
    "refactor", "bug", "bugs", "erro", "erros", "error", "errors",
    "stack trace", "traceback", "deploy", "sql", "api", "endpoint", "teste",
    "testes", "test", "testing", "log", "logs", "logar", "compilar",
    "compile", "regex", "typescript", "python", "javascript", "classe",
    "class", "modulo", "module", "commit", "pull request", "repo",
)

#: Verbs that imply a single, well-scoped operation ("do exactly this one thing").
_SIMPLE_VERB_KEYWORDS: Tuple[str, ...] = (
    "resumir", "resuma", "resume", "summarize", "summarise", "extrair", "extraia",
    "extract", "classificar", "classifique", "classify", "formatar", "formate",
    "format", "renomear", "renomeie", "rename", "traduzir", "traduza", "translate",
    "listar", "liste", "list", "count", "contar", "conte",
)


def _compile(keywords: Iterable[str]) -> re.Pattern:
    """Build one word-boundary-alternation regex from a keyword tuple."""
    parts = sorted({kw for kw in keywords if kw}, key=len, reverse=True)
    return re.compile("|".join(r"\b" + re.escape(kw) for kw in parts))


def _compile_whole_words(keywords: Iterable[str]) -> re.Pattern:
    """Whole-word variant, for short keywords that must not match as stems.

    ``\brepo`` would match "report" and ``\blist`` would match "listen", which
    silently pushed a one-line summarisation request to NORMAL. Stems
    (``concorr``, ``autentica``, ``distribuid``) keep the prefix matcher; short
    words get both boundaries.
    """
    parts = sorted({kw for kw in keywords if kw}, key=len, reverse=True)
    return re.compile("|".join(r"\b" + re.escape(kw) + r"\b" for kw in parts))


_RISK_PATTERNS = tuple((code, _compile(words)) for code, words in _RISK_PATTERNS)
_HIGH_PATTERNS = tuple((code, _compile(words)) for code, words in _HIGH_PATTERNS)
_PRIOR_FAILURE_PATTERN = _compile(_PRIOR_FAILURE_KEYWORDS)
_CODE_INTENT_PATTERN = _compile_whole_words(_CODE_INTENT_KEYWORDS)
_SIMPLE_VERB_PATTERN = _compile_whole_words(_SIMPLE_VERB_KEYWORDS)

_SHADOW_JSON_FIELDS: Tuple[str, ...] = (
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
)


# ── value objects ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskSignals:
    """Deterministic classification of one first-turn message."""

    complexity_class: str
    score: int
    requires_vision: bool
    reason_codes: Tuple[str, ...]


@dataclass(frozen=True)
class RouteDecision:
    """What the router *would* do. ``applied`` is always False in Phase 1."""

    tier: str
    provider: str
    model: str
    reasoning_effort: str
    escalation_chain: Tuple[Tuple[str, str], ...]
    complexity_class: str
    reason_codes: Tuple[str, ...]
    mode: str
    pinned: bool
    shadow: bool
    applied: bool


@dataclass(frozen=True)
class RouteApplicationPlan:
    """Pure pre-construction plan; surfaces own credential resolution."""

    model: str
    provider: str
    reasoning_effort: str
    tier: str
    should_apply: bool
    decision: Optional[RouteDecision]


# ── small pure helpers ───────────────────────────────────────────────────────


def _dedupe(codes: Iterable[str]) -> Tuple[str, ...]:
    seen: Dict[str, None] = {}
    for code in codes:
        if isinstance(code, str) and code and code not in seen:
            seen[code] = None
    return tuple(seen)


def _clean_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _bounded(value: str, limit: int) -> str:
    text = _clean_str(value)
    return text[:limit]


def _fold(text: str) -> str:
    """Lowercase + strip accents so PT-BR and EN share one keyword list."""
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text[:_MAX_CLASSIFY_CHARS])
    return decomposed.encode("ascii", "ignore").decode("ascii").lower()


def _extract_text(payload: Any) -> str:
    """Best-effort bounded text from a non-str payload (never echoes raw blobs)."""
    if isinstance(payload, dict):
        for key in ("text", "content", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""
    if isinstance(payload, (list, tuple)):
        chunks: List[str] = []
        total = 0
        for part in payload:
            if isinstance(part, str):
                chunk = part
            elif isinstance(part, dict):
                chunk = part.get("text") if isinstance(part.get("text"), str) else ""
            else:
                chunk = ""
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= _MAX_CLASSIFY_CHARS:
                break
        return " ".join(chunks)
    return ""


# ── classification ───────────────────────────────────────────────────────────


def classify_task(
    *,
    message: str,
    history_len: int = 0,
    has_images: bool = False,
) -> TaskSignals:
    """Classify a first-turn message. Pure, deterministic, bilingual, no I/O.

    The only inputs are the message text, the number of prior turns and whether
    the payload carries imagery. No LLM, no network, no clock, no randomness.
    """
    if isinstance(message, str):
        text = _fold(message)
        length = len(message)
        payload_code = ""
    else:
        text = _fold(_extract_text(message))
        length = len(text)
        payload_code = _NON_TEXT_PAYLOAD

    try:
        history = int(history_len)
    except (TypeError, ValueError):
        history = 0
    if history < 0:
        history = 0

    reasons: List[str] = []
    if payload_code:
        reasons.append(payload_code)
    if has_images:
        reasons.append(_HAS_IMAGES)
    if history:
        reasons.append(_HAS_HISTORY)

    risk_hits = [code for code, pattern in _RISK_PATTERNS if pattern.search(text)]
    high_hits = [code for code, pattern in _HIGH_PATTERNS if pattern.search(text)]
    prior_failure = bool(_PRIOR_FAILURE_PATTERN.search(text))
    code_intent = bool(_CODE_INTENT_PATTERN.search(text))
    simple_verb = bool(_SIMPLE_VERB_PATTERN.search(text))

    if risk_hits:
        # Life-safety / money / data-loss surface area outranks everything.
        complexity = "CRITICAL"
        if prior_failure:
            reasons.append(_PRIOR_FAILURE)
        reasons.extend(risk_hits)
    elif high_hits or prior_failure:
        complexity = "HIGH"
        if prior_failure:
            reasons.append(_PRIOR_FAILURE)
        reasons.extend(high_hits)
    elif (
        length <= _TRIVIAL_MAX_CHARS
        and not code_intent
        and not has_images
        and history == 0
    ):
        complexity = "TRIVIAL"
        reasons.append(_SHORT_PROMPT)
        if simple_verb:
            reasons.append(_SIMPLE_VERB)
    elif simple_verb and not code_intent and length <= _LOW_MAX_CHARS:
        complexity = "LOW"
        reasons.append(_SIMPLE_VERB)
        if length <= _TRIVIAL_MAX_CHARS:
            reasons.append(_SHORT_PROMPT)
    else:
        complexity = "NORMAL"
        if length > _LONG_PROMPT_CHARS:
            reasons.append(_LONG_PROMPT)
        if history:
            reasons.append(_HAS_HISTORY)
        if code_intent:
            reasons.append(_CODE_INTENT)

    if not reasons:
        reasons.append(_DEFAULT_REASON)

    return TaskSignals(
        complexity_class=complexity,
        score=_SCORE_BY_CLASS[complexity],
        requires_vision=bool(has_images),
        reason_codes=_dedupe(reasons),
    )


# ── config ───────────────────────────────────────────────────────────────────


def _read_config_from_disk() -> Dict[str, Any]:
    """Read config.yaml through the cached readonly loader (no mutation)."""
    try:
        from hermes_cli.config import load_config_readonly

        loaded = load_config_readonly()
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return {}


def _extract_adaptive_section(raw: Any) -> Dict[str, Any]:
    """Pull ``agent.adaptive_routing`` out of whatever shape we were handed."""
    if not isinstance(raw, dict):
        return {}
    direct = raw.get("adaptive_routing")
    if isinstance(direct, dict):
        return direct
    agent = raw.get("agent")
    if isinstance(agent, dict):
        if "adaptive_routing" in agent:
            nested = agent.get("adaptive_routing")
            return nested if isinstance(nested, dict) else {}
    return raw


def _normalise_max_escalations(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 2
    if value < 0:
        return 2
    return min(value, len(TIER_ORDER))


def _normalise_entry(entry: Any, tier: str) -> Optional[Dict[str, str]]:
    if not isinstance(entry, dict):
        return None
    provider = _clean_str(entry.get("provider"))
    model = _clean_str(entry.get("model"))
    if not provider or not model:
        return None
    effort = _clean_str(entry.get("reasoning_effort")).lower()
    if effort not in _VALID_EFFORTS:
        effort = _DEFAULT_EFFORT_BY_TIER.get(tier, "medium")
    return {"provider": provider, "model": model, "reasoning_effort": effort}


def _normalise_tiers(raw: Any) -> Dict[str, Tuple[Dict[str, str], ...]]:
    tiers: Dict[str, Tuple[Dict[str, str], ...]] = {tier: () for tier in TIER_ORDER}
    if not isinstance(raw, dict):
        return tiers
    for tier, entries in raw.items():
        if not isinstance(tier, str) or tier not in TIER_ORDER:
            continue  # unknown tier names are ignored, never guessed at
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, (list, tuple)):
            continue
        cleaned: List[Dict[str, str]] = []
        for entry in entries:
            normalised = _normalise_entry(entry, tier)
            if normalised is not None:
                cleaned.append(normalised)
        tiers[tier] = tuple(cleaned)
    return tiers


def _normalise_surfaces(raw: Any) -> Dict[str, bool]:
    """Allowlist of surfaces permitted to apply a route.

    Fail-closed: a value that is not a boolean keeps the default, and the
    defaults only let the two Phase 2 surfaces apply. Unknown surface names are
    ignored rather than guessed at.
    """
    surfaces: Dict[str, bool] = {
        name: bool(_DEFAULT_SURFACES.get(name, False)) for name in SURFACES
    }
    if isinstance(raw, dict):
        for name in SURFACES:
            value = raw.get(name)
            if isinstance(value, bool):
                surfaces[name] = value
    return surfaces


def _normalise_budget(raw: Any) -> Dict[str, int]:
    """Per-day cap on *applied* routes, per tier.

    A missing tier is uncapped; ``0`` blocks the tier outright. Only real,
    non-negative integers are honoured (``True`` is not ``1`` here).
    """
    budget: Dict[str, int] = {}
    if not isinstance(raw, dict):
        return budget
    for tier, value in raw.items():
        if not isinstance(tier, str) or tier not in TIER_ORDER:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        budget[tier] = value
    return budget


def _normalise_data_policy(raw: Any) -> Dict[str, Tuple[str, ...]]:
    """Per-class data policy: classes that stay local, tiers never used."""
    policy: Dict[str, Tuple[str, ...]] = {
        "local_only_classes": (),
        "forbidden_tiers": (),
    }
    if not isinstance(raw, dict):
        return policy
    classes = raw.get("local_only_classes")
    if isinstance(classes, (list, tuple)):
        policy["local_only_classes"] = _dedupe(
            [
                item.strip().upper()
                for item in classes
                if isinstance(item, str) and item.strip().upper() in COMPLEXITY_CLASSES
            ]
        )
    tiers = raw.get("forbidden_tiers")
    if isinstance(tiers, (list, tuple)):
        policy["forbidden_tiers"] = _dedupe(
            [
                item.strip().lower()
                for item in tiers
                if isinstance(item, str) and item.strip().lower() in TIER_ORDER
            ]
        )
    return policy


def _normalise_section(section: Any) -> Dict[str, Any]:
    if not isinstance(section, dict):
        section = {}
    enabled = section.get("enabled")
    apply_routes = section.get("apply_routes")
    shadow_mode = section.get("shadow_mode")
    mode = section.get("mode")
    privacy = section.get("privacy")
    normalised_mode = mode.strip().lower() if isinstance(mode, str) else ""
    if normalised_mode not in MODES:
        normalised_mode = "balanced"
    normalised_privacy = privacy.strip().lower() if isinstance(privacy, str) else ""
    if normalised_privacy not in PRIVACY_VALUES:
        normalised_privacy = "normal"
    return {
        "enabled": enabled if isinstance(enabled, bool) else False,
        "apply_routes": apply_routes if isinstance(apply_routes, bool) else False,
        "shadow_mode": shadow_mode if isinstance(shadow_mode, bool) else True,
        "mode": normalised_mode,
        "privacy": normalised_privacy,
        "max_escalations": _normalise_max_escalations(section.get("max_escalations")),
        "tiers": _normalise_tiers(section.get("tiers")),
        "surfaces": _normalise_surfaces(section.get("surfaces")),
        "budget": _normalise_budget(section.get("budget")),
        "data_policy": _normalise_data_policy(section.get("data_policy")),
    }


def load_adaptive_config(config: Any = None) -> Dict[str, Any]:
    """Return the normalised ``agent.adaptive_routing`` section.

    Accepts ``None`` (read ``config.yaml``), a full Hermes config dict, or an
    already-normalised adaptive section. Malformed values fall back to
    defaults; nothing here raises, and the input is never mutated.
    """
    raw = _read_config_from_disk() if config is None else config
    return _normalise_section(_extract_adaptive_section(raw))


# ── surfaces: who may apply, who owns the telemetry ──────────────────────────


def _application_is_live(cfg: Dict[str, Any]) -> bool:
    """True when an enabled, non-shadow config changes routes before build."""
    return bool(cfg.get("apply_routes")) and not bool(cfg.get("shadow_mode"))


def surface_allowed_to_apply(config: Any, surface: Any) -> bool:
    """True when the ``surfaces`` allowlist lets ``surface`` apply a route.

    Single source of truth for the Phase 3 gate: the applier
    (:func:`plan_route_application`) and the observer must never disagree about
    which surfaces may change a turn's route. Fail-closed — an unknown surface
    name, a malformed value or an unreadable config all answer ``False``.
    """
    try:
        cfg = load_adaptive_config(config)
        name = _clean_str(surface).strip().lower()
        if name not in SURFACES:
            return False
        return bool((cfg.get("surfaces") or {}).get(name, False))
    except Exception:
        return False


def surface_owns_its_telemetry(config: Any, surface: Any) -> bool:
    """True when ``surface`` writes its own routing telemetry.

    Two cases, each of which would produce a duplicate log line if the shared
    prologue observer also recorded the turn:

    * a surface with a pre-construction applier (``cli``/``gateway``) — it
      records the applied route *and* the denied one, so it owns its telemetry
      even when the allowlist turns it off;
    * a surface the config authorises to apply — its applier is expected to
      record the same way, so silence there is the configuration's own
      statement that the surface owns routing.

    Everything else (``cron``, ``delegation``, ``tui``, unmapped front ends) is
    observed instead.
    """
    name = _clean_str(surface).strip().lower()
    if name in _APPLIER_SURFACES:
        return True
    return surface_allowed_to_apply(config, name)


def _is_gateway_channel(name: str) -> bool:
    """True when ``name`` is a platform the gateway serves (structural check).

    A gateway agent carries the *channel* it serves (``telegram``, ``signal``,
    …) rather than the surface name, so channels have to be recognised to keep
    the observer out of their applier's telemetry. Membership is decided by
    constructing ``gateway.config.Platform`` — the same structural test the TUI
    gateway uses — so registered plugin platforms are covered without a
    hardcoded list. When the gateway package cannot be imported there is no
    gateway applier in this process either, so answering ``False`` can only cost
    an extra observation, never a duplicate record.
    """
    if not name:
        return False
    try:
        from gateway.config import Platform
    except Exception:
        return False
    try:
        Platform(name)
    except Exception:
        return False
    return True


def surface_for_platform(platform: Any) -> str:
    """Map an agent's ``platform`` to a routing surface (``""`` when unmapped).

    Agents are stamped with what they are talking to — ``cli``, ``tui``,
    ``cron``, ``subagent``, or the channel a gateway agent serves — while the
    routing config names surfaces. Only a mapping returns a name in
    :data:`SURFACES`; anything unmapped (``desktop``, ``acp``, a bare plugin
    tag) answers ``""``, which is treated as a surface that may not apply. That
    is the fail-open direction for evidence: an unnamed surface is by definition
    not on the apply allowlist, and observing it costs a log line, while
    suppressing it would lose the turn.
    """
    name = _clean_str(platform).strip().lower()
    if not name:
        return ""
    if name in SURFACES:
        return name
    if name in _DELEGATION_PLATFORMS:
        return "delegation"
    if name in _GATEWAY_UNPLANNED_PLATFORMS:
        # Served by the gateway, but its turn-config builder returns before a
        # decision exists (pinned/request-oriented), so observe rather than
        # defer to an applier that will not write anything.
        return ""
    if _is_gateway_channel(name):
        return "gateway"
    return ""


def _tier_entries(
    tiers: Any, tier: str
) -> Tuple[Dict[str, str], ...]:
    """Validated entries for one tier, tolerating any messy input shape."""
    if not isinstance(tiers, dict) or tier not in TIER_ORDER:
        return ()
    raw = tiers.get(tier)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    cleaned: List[Dict[str, str]] = []
    for entry in raw:
        normalised = _normalise_entry(entry, tier)
        if normalised is not None:
            cleaned.append(normalised)
    return tuple(cleaned)


def _first_entry(tiers: Any, tier: str) -> Optional[Dict[str, str]]:
    entries = _tier_entries(tiers, tier)
    return entries[0] if entries else None


# ── tier selection ───────────────────────────────────────────────────────────


def _resolve_mode(mode: Any, fallback: str = "balanced") -> str:
    if isinstance(mode, str) and mode.strip().lower() in MODES:
        return mode.strip().lower()
    if isinstance(fallback, str) and fallback.strip().lower() in MODES:
        return fallback.strip().lower()
    return "balanced"


def _target_tier(complexity_class: str, requires_vision: bool, mode: str) -> str:
    """Map a complexity class (+vision) to a tier name for the given mode."""
    if requires_vision:
        return "multimodal"
    balanced: Dict[str, str] = {
        "TRIVIAL": "local",
        "LOW": "local",
        "NORMAL": "workhorse",
        "HIGH": "workhorse",
        "CRITICAL": "premium",
    }
    if mode == "economy":
        # Never escalate to premium/frontier on cost-sensitive installs, except
        # for genuinely critical work (which is exactly where a retry is worst).
        return balanced.get(complexity_class, "workhorse")
    if mode == "quality":
        floor = {
            "TRIVIAL": "workhorse",
            "LOW": "workhorse",
            "NORMAL": "workhorse",
            "HIGH": "workhorse",
            "CRITICAL": "premium",
        }
        return floor.get(complexity_class, "workhorse")
    if mode == "maximum":
        ceiling = {
            "TRIVIAL": "workhorse",
            "LOW": "workhorse",
            "NORMAL": "premium",
            "HIGH": "frontier",
            "CRITICAL": "frontier",
        }
        return ceiling.get(complexity_class, "premium")
    return balanced.get(complexity_class, "workhorse")


def _select_tier(
    mapped: str,
    tiers: Any,
    excluded: Iterable[str] = (),
    requires_vision: bool = False,
) -> Tuple[Optional[str], Tuple[str, ...]]:
    """Resolve the mapped tier, falling through to the nearest configured one.

    Prefers a *stronger* configured tier (an unconfigured tier must never
    silently weaken the route), then a weaker one. Tiers in ``excluded`` (the
    Phase 3 data policy) are never selected, and the reason codes say so.
    Returns ``(None, codes)`` when nothing eligible is configured — the caller
    then reports an explicitly empty route instead of inventing a half-empty
    provider/model pair.
    """
    blocked = {name for name in excluded if isinstance(name, str)}
    if (
        mapped in TIER_ORDER
        and mapped not in blocked
        and (mapped != "multimodal" or requires_vision)
        and _tier_entries(tiers, mapped)
    ):
        return mapped, ()
    start = TIER_ORDER.index(mapped) + 1 if mapped in TIER_ORDER else 0
    for candidate in list(TIER_ORDER[start:]) + list(reversed(TIER_ORDER[:start])):
        # ``multimodal`` is a capability tier, not a strength tier: a text-only
        # task must never be walked into a vision model.
        if candidate in blocked or (candidate == "multimodal" and not requires_vision):
            continue
        if _tier_entries(tiers, candidate):
            if mapped in blocked:
                return candidate, (_POLICY_TIER_FORBIDDEN, _TIER_FALLBACK)
            return candidate, (_TIER_FALLBACK,)
    if mapped in blocked:
        return None, (_POLICY_TIER_FORBIDDEN, _TIER_FALLBACK, _NO_TIER_CONFIGURED)
    return None, (_TIER_FALLBACK, _NO_TIER_CONFIGURED)


def _escalation_pairs(
    tier: str,
    provider: str,
    model: str,
    tiers: Any,
    max_escalations: int,
    requires_vision: bool = False,
    excluded: Iterable[str] = (),
) -> Tuple[Tuple[str, str], ...]:
    """Stronger configured tiers, in ladder order, deduped and truncated.

    ``multimodal`` is a *capability* tier (vision), not a strength tier: it sits
    after ``workhorse`` in :data:`TIER_ORDER`, so a naive walk would tell a
    text-only task to escalate into a vision model. It is therefore only
    reachable when the task actually requires vision.
    """
    if max_escalations <= 0 or tier not in TIER_ORDER:
        return ()
    blocked = {name for name in excluded if isinstance(name, str)}
    seen = {(provider, model)}
    chain: List[Tuple[str, str]] = []
    for candidate in TIER_ORDER[TIER_ORDER.index(tier) + 1 :]:
        if candidate == "multimodal" and not requires_vision:
            continue
        if candidate in blocked:
            continue
        for entry in _tier_entries(tiers, candidate):
            key = (entry["provider"], entry["model"])
            if key in seen:
                continue
            seen.add(key)
            chain.append(key)
            if len(chain) >= max_escalations:
                return tuple(chain)
    return tuple(chain)


# ── routing ──────────────────────────────────────────────────────────────────


def resolve_route(
    signals: TaskSignals,
    *,
    config: Any,
    mode: Any = None,
    explicit_provider: Any = "",
    explicit_model: Any = "",
    privacy: Any = "normal",
) -> RouteDecision:
    """Map :class:`TaskSignals` to a :class:`RouteDecision` (never applies it)."""
    cfg = load_adaptive_config(config)
    tiers = cfg["tiers"]
    shadow = bool(cfg["shadow_mode"])
    resolved_mode = _resolve_mode(mode, cfg["mode"])
    policy = cfg.get("data_policy") or {}
    local_only_classes = tuple(policy.get("local_only_classes") or ())
    forbidden_tiers = tuple(policy.get("forbidden_tiers") or ())

    base_codes: Tuple[str, ...] = ()
    if isinstance(signals, TaskSignals):
        base_codes = signals.reason_codes
    complexity = "NORMAL"
    if isinstance(signals, TaskSignals) and signals.complexity_class in COMPLEXITY_CLASSES:
        complexity = signals.complexity_class
    requires_vision = bool(getattr(signals, "requires_vision", False))

    pin_provider = _clean_str(explicit_provider)
    pin_model = _clean_str(explicit_model)
    if pin_provider or pin_model:
        # A user-forced model is authoritative; we only report it.
        return RouteDecision(
            tier="pinned",
            provider=pin_provider,
            model=pin_model,
            reasoning_effort="",
            escalation_chain=(),
            complexity_class=complexity,
            reason_codes=_dedupe((_EXPLICIT_PIN,) + base_codes),
            mode=resolved_mode,
            pinned=True,
            shadow=False,
            applied=False,
        )

    if _clean_str(privacy).lower() == "local_only":
        entry = _first_entry(tiers, "local")
        if entry is None:
            return RouteDecision(
                tier="",
                provider="",
                model="",
                reasoning_effort="",
                escalation_chain=(),
                complexity_class=complexity,
                reason_codes=_dedupe((_PRIVACY_LOCAL_ONLY_UNAVAILABLE,) + base_codes),
                mode=resolved_mode,
                pinned=False,
                shadow=shadow,
                applied=False,
            )
        return RouteDecision(
            tier="local",
            provider=entry["provider"],
            model=entry["model"],
            reasoning_effort=entry["reasoning_effort"],
            escalation_chain=(),
            complexity_class=complexity,
            reason_codes=_dedupe((_PRIVACY_LOCAL_ONLY,) + base_codes),
            mode=resolved_mode,
            pinned=False,
            shadow=shadow,
            applied=False,
        )

    if complexity in local_only_classes:
        # Data policy: this class must never leave the machine. Same shape as
        # the privacy contract, decided per class instead of globally.
        entry = _first_entry(tiers, "local")
        if entry is None:
            return RouteDecision(
                tier="",
                provider="",
                model="",
                reasoning_effort="",
                escalation_chain=(),
                complexity_class=complexity,
                reason_codes=_dedupe((_POLICY_LOCAL_ONLY_UNAVAILABLE,) + base_codes),
                mode=resolved_mode,
                pinned=False,
                shadow=shadow,
                applied=False,
            )
        return RouteDecision(
            tier="local",
            provider=entry["provider"],
            model=entry["model"],
            reasoning_effort=entry["reasoning_effort"],
            escalation_chain=(),
            complexity_class=complexity,
            reason_codes=_dedupe((_POLICY_LOCAL_ONLY,) + base_codes),
            mode=resolved_mode,
            pinned=False,
            shadow=shadow,
            applied=False,
        )

    mapped = _target_tier(complexity, requires_vision, resolved_mode)
    tier, tier_codes = _select_tier(
        mapped, tiers, forbidden_tiers, requires_vision=requires_vision
    )
    codes = _dedupe(tuple(tier_codes) + base_codes)

    if tier is None:
        return RouteDecision(
            tier="",
            provider="",
            model="",
            reasoning_effort="",
            escalation_chain=(),
            complexity_class=complexity,
            reason_codes=codes,
            mode=resolved_mode,
            pinned=False,
            shadow=shadow,
            applied=False,
        )

    entry = _first_entry(tiers, tier) or {}
    provider = entry.get("provider", "")
    model = entry.get("model", "")
    effort = entry.get("reasoning_effort") or _DEFAULT_EFFORT_BY_TIER.get(tier, "medium")
    return RouteDecision(
        tier=tier,
        provider=provider,
        model=model,
        reasoning_effort=effort,
        escalation_chain=_escalation_pairs(
            tier,
            provider,
            model,
            tiers,
            cfg["max_escalations"],
            requires_vision=requires_vision,
            excluded=forbidden_tiers,
        ),
        complexity_class=complexity,
        reason_codes=codes,
        mode=resolved_mode,
        pinned=False,
        shadow=shadow,
        applied=False,
    )


def build_escalation_chain(decision: RouteDecision, config: Any) -> List[Dict[str, str]]:
    """Fallback-shaped dicts for the decision's escalation chain, in order.

    Phase 1 exposes this for tests and for Phase 2. It is deliberately **not**
    wired into any runtime fallback: the chain stays a suggestion in the shadow
    log until Phase 2 opts in.
    """
    try:
        if getattr(decision, "pinned", False):
            return []
        cfg = load_adaptive_config(config)
        effort_by_key: Dict[Tuple[str, str], str] = {}
        for tier in TIER_ORDER:
            for entry in _tier_entries(cfg["tiers"], tier):
                key = (entry["provider"], entry["model"])
                effort_by_key.setdefault(key, entry["reasoning_effort"])
        chain: List[Dict[str, str]] = []
        seen: set = set()
        for pair in tuple(getattr(decision, "escalation_chain", ()) or ()):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            provider, model = _clean_str(pair[0]), _clean_str(pair[1])
            if not provider or not model or (provider, model) in seen:
                continue
            seen.add((provider, model))
            chain.append(
                {
                    "provider": provider,
                    "model": model,
                    "reasoning_effort": effort_by_key.get((provider, model), "medium"),
                }
            )
        return chain
    except Exception:
        return []


def _plan_from_decision(decision: RouteDecision) -> RouteApplicationPlan:
    """The applying-plan shape for a usable decision."""
    return RouteApplicationPlan(
        model=_clean_str(decision.model),
        provider=_clean_str(decision.provider),
        reasoning_effort=_clean_str(decision.reasoning_effort),
        tier=_clean_str(decision.tier),
        should_apply=True,
        decision=decision,
    )


def _denied_plan(decision: RouteDecision, code: str) -> RouteApplicationPlan:
    """Keep the surface's own route, but say *why* it was kept.

    The returned plan carries no provider/model — ``should_apply`` is false —
    while still exposing ``decision`` (with the gate's reason code) so callers
    can record a non-applied observation. That is what makes expansion
    evidence-driven: a denied surface is visible instead of silently inert.
    """
    annotated = replace(
        decision,
        reason_codes=_dedupe((code,) + tuple(decision.reason_codes)),
        applied=False,
    )
    return RouteApplicationPlan(
        model="",
        provider="",
        reasoning_effort="",
        tier="",
        should_apply=False,
        decision=annotated,
    )


def _budget_remaining(
    counts: Dict[str, Any], tier: str, caps: Dict[str, int]
) -> Optional[int]:
    """Applies still available for ``tier`` today; ``None`` means uncapped."""
    cap = caps.get(tier)
    if cap is None:
        return None
    try:
        used = int(counts.get(tier, 0))
    except (TypeError, ValueError):
        used = 0
    return max(0, cap - used)


def _budget_gate(
    plan: RouteApplicationPlan,
    decision: RouteDecision,
    cfg: Dict[str, Any],
    budget_state: Any,
) -> RouteApplicationPlan:
    """Downgrade or deny a route whose tier has spent today's budget.

    The ladder is walked *downwards* only: a spent budget may never promote a
    route. A vision tier is never downgraded to a text model — the image would
    silently stop being read — so it is denied instead.
    """
    caps = cfg.get("budget") or {}
    if not caps or not isinstance(budget_state, dict):
        return plan
    raw_counts = budget_state.get("counts")
    counts: Dict[str, Any] = raw_counts if isinstance(raw_counts, dict) else {}
    tier = _clean_str(decision.tier)
    if tier not in TIER_ORDER:
        return plan
    remaining = _budget_remaining(counts, tier, caps)
    if remaining is None or remaining > 0:
        return plan
    if tier == "multimodal":
        return _denied_plan(decision, _BUDGET_EXHAUSTED)
    forbidden = tuple((cfg.get("data_policy") or {}).get("forbidden_tiers") or ())
    for candidate in reversed(TIER_ORDER[: TIER_ORDER.index(tier)]):
        if candidate in forbidden or candidate == "multimodal":
            continue
        entry = _first_entry(cfg["tiers"], candidate)
        if entry is None:
            continue
        cheaper = _budget_remaining(counts, candidate, caps)
        if cheaper is not None and cheaper <= 0:
            continue
        downgraded = replace(
            decision,
            tier=candidate,
            provider=entry["provider"],
            model=entry["model"],
            reasoning_effort=entry["reasoning_effort"],
            escalation_chain=_escalation_pairs(
                candidate,
                entry["provider"],
                entry["model"],
                cfg["tiers"],
                cfg["max_escalations"],
                requires_vision=False,
                excluded=forbidden,
            ),
            reason_codes=_dedupe((_BUDGET_DOWNGRADE,) + tuple(decision.reason_codes)),
        )
        return _plan_from_decision(downgraded)
    return _denied_plan(decision, _BUDGET_EXHAUSTED)


def plan_route_application(
    *,
    message: Any,
    current_model: Any,
    current_runtime: Any,
    config: Any,
    explicit_pin: bool,
    has_history: bool,
    session_is_new: bool = True,
    has_images: bool = False,
    surface: Any = "",
    budget_state: Any = None,
) -> RouteApplicationPlan:
    """Return a deterministic first-turn candidate or the original route.

    This helper is deliberately pure: it does not resolve credentials, write
    telemetry, mutate session state, or construct/switch an agent. CLI and
    gateway adapters validate the candidate runtime before applying it.

    Phase 3 routes every application through explicit gates: ``surface`` must be
    enabled in ``agent.adaptive_routing.surfaces`` (default: only ``cli`` and
    ``gateway``) and the target tier must still have daily budget left. A denied
    gate returns a non-applying plan that still carries the decision, so the
    caller can record *why* nothing changed.
    """
    original_model = _clean_str(current_model)
    original_provider = ""
    if isinstance(current_runtime, dict):
        original_provider = _clean_str(current_runtime.get("provider"))
    original = RouteApplicationPlan(
        model=original_model,
        provider=original_provider,
        reasoning_effort="",
        tier="",
        should_apply=False,
        decision=None,
    )
    try:
        cfg = load_adaptive_config(config)
        active = bool(cfg["enabled"]) and _application_is_live(cfg)
        if (
            not active
            or bool(explicit_pin)
            or bool(has_history)
            or not bool(session_is_new)
        ):
            return original
        signals = classify_task(
            message=message,
            history_len=0,
            has_images=bool(has_images),
        )
        decision = resolve_route(signals, config=cfg, privacy=cfg.get("privacy"))
        provider = _clean_str(decision.provider)
        model = _clean_str(decision.model)
        if not provider or not model or decision.pinned:
            return original
        surface_name = _clean_str(surface).lower()
        if not surface_allowed_to_apply(cfg, surface_name):
            return _denied_plan(decision, _SURFACE_NOT_ALLOWED)
        return _budget_gate(_plan_from_decision(decision), decision, cfg, budget_state)
    except Exception:
        return original


def load_budget_state(
    *,
    config: Any = None,
    home: Any = None,
    now: Any = None,
) -> Dict[str, Any]:
    """Count *today's applied routes* per tier from the routing log.

    Append-only telemetry is the cheapest honest ledger: no new file, no new
    write path, and the counter can never claim a route that was not recorded
    as applied. Only the newest ``_BUDGET_READ_BYTES`` of the log are read, so
    the cost stays constant as the file grows. Returns
    ``{"date": "YYYY-MM-DD", "counts": {tier: n}}`` and never raises.
    """
    empty: Dict[str, Any] = {"date": "", "counts": {}}
    try:
        moment = float(now) if now is not None else time.time()
        local = time.localtime(moment)
        day_start = time.mktime(
            (local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1)
        )
        root = Path(home) if home is not None else _resolve_home()
        if root is None:
            return empty
        path = root / SHADOW_LOG_NAME
        if not path.is_file():
            return empty
        counts: Dict[str, int] = {}
        size = path.stat().st_size
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            if size > _BUDGET_READ_BYTES:
                handle.seek(size - _BUDGET_READ_BYTES)
                handle.readline()  # drop the truncated first line
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, dict) or record.get("applied") is not True:
                    continue
                try:
                    recorded_at = float(record.get("ts"))
                except (TypeError, ValueError):
                    continue
                if recorded_at < day_start:
                    continue
                tier = _clean_str(record.get("tier"))
                if tier:
                    counts[tier] = counts.get(tier, 0) + 1
        return {"date": time.strftime("%Y-%m-%d", local), "counts": counts}
    except Exception:
        return empty


# ── first-turn gate ──────────────────────────────────────────────────────────


def should_observe_first_turn(
    *,
    config: Any,
    has_history: bool,
    explicit_provider: str = "",
    explicit_model: str = "",
    surface: Any = "",
) -> bool:
    """Observe when routing is enabled, this is the first turn, and no other
    writer already owns this surface's telemetry.

    ``explicit_provider``/``explicit_model`` are accepted for callers that want
    to reason about pins later; today a pinned turn is still worth observing
    (the shadow log records that a pin overrode the router).

    ``surface`` is the surface this turn belongs to
    (:func:`surface_for_platform`). With application live, a surface that owns
    its telemetry (``cli``/``gateway``, or anything the allowlist authorises) is
    *not* observed — its applier records the same decision, and observing it too
    would double-log every first turn. Any other surface, including an unnamed
    one (``""``: an agent whose platform maps to no surface is not on the apply
    allowlist either), is observed. With application off (shadow mode or
    ``apply_routes: false``) every surface observes, exactly as in Phase 1.
    """
    try:
        cfg = load_adaptive_config(config)
    except Exception:
        return False
    if not bool(cfg["enabled"]) or bool(has_history):
        return False
    if not _application_is_live(cfg):
        return True
    return not surface_owns_its_telemetry(cfg, surface)


# ── telemetry (bounded, enumerated, no user text) ────────────────────────────


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
    if env_home:
        return Path(env_home)
    return None


def session_ref(raw: Any, *, length: int = 12) -> str:
    """Return a stable, non-reversible reference for a session identifier.

    Used when a caller's only available identifier is an external one (a
    gateway chat key embeds ``platform:chat_id:user_id``), so decisions stay
    correlatable across turns without writing a durable chat/user identifier to
    the routing log. Routing surfaces that already hold a locally-generated
    session id (timestamp + uuid) may record it directly.
    """
    text = _clean_str(raw)
    if not text:
        return ""
    try:
        size = max(6, min(int(length), 32))
    except Exception:
        size = 12
    return "s" + sha256(text.encode("utf-8")).hexdigest()[:size]


#: Content-block types that carry pixels rather than text.
_IMAGE_PART_TYPES = ("image", "image_url", "input_image")
#: Dict keys that mark an image payload even without a ``type`` field.
_IMAGE_KEYS = (
    "image_url",
    "image",
    "input_image",
    "image_path",
    "image_base64",
    "image_data",
)


def _part_is_image(part: Any) -> bool:
    if isinstance(part, dict):
        if str(part.get("type", "")).strip().lower() in _IMAGE_PART_TYPES:
            return True
        return any(key in part for key in _IMAGE_KEYS)
    # A non-text object inside a content list is an attachment payload.
    return not isinstance(part, str)


def message_has_images(message: Any) -> bool:
    """Return True when a user message carries an image payload.

    Control-path detection: it must err towards "no image" (stay on the cheap
    tier, where the existing ``vision_analyze`` fallback still describes the
    pixels) rather than towards the vision tier. Callers that *know* the turn
    has attachments pass ``has_images=True`` explicitly; this helper is the
    safety net for content-list payloads.

    Note the shadow *observer* (``agent.turn_context``) intentionally stays
    fail-open on non-str payloads — it only measures, and a false positive
    there costs nothing.
    """
    try:
        if message is None or isinstance(message, str):
            return False
        if isinstance(message, (list, tuple)):
            return any(_part_is_image(part) for part in message)
        if isinstance(message, dict):
            return _part_is_image(message)
        # Unknown provider-native object type: not provably an image.
        return False
    except Exception:
        return False


def record_shadow_decision(
    decision: RouteDecision,
    *,
    effective_provider: Any = "",
    effective_model: Any = "",
    session_id: Any = "",
    platform: Any = "",
    applied: bool = False,
    surface: Any = "",
    home: Any = None,
) -> Optional[str]:
    """Append exactly one bounded JSON line describing a shadow decision.

    Returns the path on success, ``None`` on any failure — this is telemetry
    and must never be able to break a turn. The record carries enumerated
    fields only: no message text, no prompt, no tool arguments, no URLs, no
    credentials.

    Callers must pass a NON-IDENTIFYING ``session_id``: a gateway chat key
    embeds ``platform:chat_id:user_id``, so routing surfaces hash it with
    :func:`session_ref` before recording. The value is bounded to 128 chars.
    """
    try:
        provider = _clean_str(getattr(decision, "provider", ""))
        model = _clean_str(getattr(decision, "model", ""))
        effective_provider = _clean_str(effective_provider)
        effective_model = _clean_str(effective_model)
        reason_codes = tuple(
            code
            for code in (getattr(decision, "reason_codes", ()) or ())
            if isinstance(code, str) and code
        )
        complexity = _bounded(getattr(decision, "complexity_class", ""), 16)
        escalation = [
            f"{_bounded(pair[0], 64)}:{_bounded(pair[1], 128)}"
            for pair in (getattr(decision, "escalation_chain", ()) or ())
            if isinstance(pair, (tuple, list)) and len(pair) == 2
        ]
        record = {
            "ts": float(time.time()),
            "session_id": _bounded(session_id, 128),
            "platform": _bounded(platform, 32),
            "tier": _bounded(getattr(decision, "tier", ""), 32),
            "provider": _bounded(provider, 64),
            "model": _bounded(model, 128),
            "mode": _bounded(getattr(decision, "mode", ""), 32),
            "complexity_class": complexity,
            "score": _SCORE_BY_CLASS.get(complexity, 2),
            "requires_vision": bool(
                _bounded(getattr(decision, "tier", ""), 32) == "multimodal"
                or _HAS_IMAGES in reason_codes
            ),
            "reason_codes": list(reason_codes),
            "escalation_chain": escalation,
            "effective_provider": _bounded(effective_provider, 64),
            "effective_model": _bounded(effective_model, 128),
            "applied": bool(applied),
            "surface": _bounded(surface, 32),
            "matched": bool(
                provider
                and model
                and provider == effective_provider
                and model == effective_model
            ),
            "router_version": ROUTER_VERSION,
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


# ── single integration entry point ───────────────────────────────────────────


def observe_shadow_route(
    *,
    agent: Any,
    user_message: Any,
    conversation_history: Any,
    config: Any = None,
    has_images: bool = False,
) -> Optional[RouteDecision]:
    """Classify a first turn, record the shadow decision, change nothing.

    Observation is decided per surface, so the appliers and this observer never
    write the same turn twice: while application is live, ``cli``/``gateway``
    record their own decision (applied, or denied with a gate reason) and are
    skipped here, while every other surface — ``tui``, ``cron``,
    ``delegation``/``subagent``, unmapped front ends — records a non-applied
    observation prefixed with the ``observe_only_surface`` reason code. When
    application is off (shadow mode / ``apply_routes: false``) every surface is
    observed, exactly as in Phase 1.

    Returns ``None`` when routing is disabled, when this is not a first turn,
    when the surface owns its own telemetry, or on any error. It never mutates
    ``agent``, never changes the effective provider/model, and never calls a
    model.
    """
    try:
        cfg = load_adaptive_config(config)
        surface = surface_for_platform(getattr(agent, "platform", ""))
        if not should_observe_first_turn(
            config=cfg,
            has_history=bool(conversation_history),
            surface=surface,
        ):
            # Disabled, not a first turn, or a surface that records its own
            # route (no classification, no routing, no file, no log line).
            return None

        effective_provider = _clean_str(getattr(agent, "provider", ""))
        effective_model = _clean_str(getattr(agent, "model", ""))
        # A pin is only a pin when the user's request actually differs from the
        # runtime that is in use.
        explicit_provider = _clean_str(getattr(agent, "requested_provider", ""))
        explicit_model = _clean_str(getattr(agent, "requested_model", ""))
        if explicit_provider == effective_provider:
            explicit_provider = ""
        if explicit_model == effective_model:
            explicit_model = ""

        signals = classify_task(
            message=user_message,
            history_len=0,
            has_images=bool(has_images),
        )
        decision = resolve_route(
            signals,
            config=cfg,
            explicit_provider=explicit_provider,
            explicit_model=explicit_model,
        )
        if _application_is_live(cfg) and not surface_owns_its_telemetry(cfg, surface):
            # Application is running and this surface may not apply: say so in
            # the record, so the evidence reads "would have routed, was not
            # allowed to" instead of looking like an ordinary shadow turn.
            decision = replace(
                decision,
                reason_codes=_dedupe(
                    (_OBSERVE_ONLY_SURFACE,) + tuple(decision.reason_codes)
                ),
            )
        record_shadow_decision(
            decision,
            effective_provider=effective_provider,
            effective_model=effective_model,
            session_id=getattr(agent, "session_id", ""),
            platform=getattr(agent, "platform", ""),
            surface=surface,
        )
        return decision
    except Exception:
        return None
