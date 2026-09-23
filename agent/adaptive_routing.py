"""HAIR adaptive routing — pure decisions with surface-owned application.

This module answers one question per first turn: *"which tier would a router
have picked for this message?"* — and records the answer next to the provider
and model that were actually used. Phase 1 remains an observe-only mode;
Phase 2 lets CLI and gateway surfaces apply the route before construction:

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
"""

from __future__ import annotations

import importlib
import json
import os
import re
import time
import unicodedata
from hashlib import sha256
from dataclasses import dataclass
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
ROUTER_VERSION = "phase2-prebuild-1"

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
    }


def load_adaptive_config(config: Any = None) -> Dict[str, Any]:
    """Return the normalised ``agent.adaptive_routing`` section.

    Accepts ``None`` (read ``config.yaml``), a full Hermes config dict, or an
    already-normalised adaptive section. Malformed values fall back to
    defaults; nothing here raises, and the input is never mutated.
    """
    raw = _read_config_from_disk() if config is None else config
    return _normalise_section(_extract_adaptive_section(raw))


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
    mapped: str, tiers: Any
) -> Tuple[Optional[str], Tuple[str, ...]]:
    """Resolve the mapped tier, falling through to the nearest configured one.

    Prefers a *stronger* configured tier (an unconfigured tier must never
    silently weaken the route), then a weaker one. Returns ``(None, codes)``
    when nothing at all is configured — the caller then reports an explicitly
    empty route instead of inventing a half-empty provider/model pair.
    """
    if mapped in TIER_ORDER and _tier_entries(tiers, mapped):
        return mapped, ()
    start = TIER_ORDER.index(mapped) + 1 if mapped in TIER_ORDER else 0
    for candidate in list(TIER_ORDER[start:]) + list(reversed(TIER_ORDER[:start])):
        if _tier_entries(tiers, candidate):
            return candidate, (_TIER_FALLBACK,)
    return None, (_TIER_FALLBACK, _NO_TIER_CONFIGURED)


def _escalation_pairs(
    tier: str,
    provider: str,
    model: str,
    tiers: Any,
    max_escalations: int,
    requires_vision: bool = False,
) -> Tuple[Tuple[str, str], ...]:
    """Stronger configured tiers, in ladder order, deduped and truncated.

    ``multimodal`` is a *capability* tier (vision), not a strength tier: it sits
    after ``workhorse`` in :data:`TIER_ORDER`, so a naive walk would tell a
    text-only task to escalate into a vision model. It is therefore only
    reachable when the task actually requires vision.
    """
    if max_escalations <= 0 or tier not in TIER_ORDER:
        return ()
    seen = {(provider, model)}
    chain: List[Tuple[str, str]] = []
    for candidate in TIER_ORDER[TIER_ORDER.index(tier) + 1 :]:
        if candidate == "multimodal" and not requires_vision:
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

    mapped = _target_tier(complexity, requires_vision, resolved_mode)
    tier, tier_codes = _select_tier(mapped, tiers)
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
) -> RouteApplicationPlan:
    """Return a deterministic first-turn candidate or the original route.

    This helper is deliberately pure: it does not resolve credentials, write
    telemetry, mutate session state, or construct/switch an agent. CLI and
    gateway adapters validate the candidate runtime before applying it.
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
        active = (
            cfg["enabled"]
            and cfg["apply_routes"]
            and not cfg["shadow_mode"]
        )
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
        return RouteApplicationPlan(
            model=model,
            provider=provider,
            reasoning_effort=_clean_str(decision.reasoning_effort),
            tier=_clean_str(decision.tier),
            should_apply=True,
            decision=decision,
        )
    except Exception:
        return original


# ── first-turn gate ──────────────────────────────────────────────────────────


def should_observe_first_turn(
    *,
    config: Any,
    has_history: bool,
    explicit_provider: str = "",
    explicit_model: str = "",
) -> bool:
    """Observe only when routing is enabled and this is the first turn.

    ``explicit_provider``/``explicit_model`` are accepted for callers that want
    to reason about pins later; today a pinned turn is still worth observing
    (the shadow log records that a pin overrode the router).
    """
    try:
        cfg = load_adaptive_config(config)
    except Exception:
        return False
    observe_only = bool(cfg["shadow_mode"]) or not bool(cfg["apply_routes"])
    return bool(cfg["enabled"]) and observe_only and not bool(has_history)


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

    Gateway chat keys embed platform, chat id and user id. Hashing keeps
    decisions correlatable across turns without writing a durable chat/user
    identifier to the routing log.
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


def message_has_images(message: Any) -> bool:
    """Return True when a user message carries an image payload.

    Mirrors the detection used by the shadow observer: a non-str/sequence
    payload is assumed multimodal, and a list/tuple is multimodal when any part
    is an OpenAI-style image content block.
    """
    try:
        if message is None or isinstance(message, str):
            return False
        if isinstance(message, (list, tuple)):
            for part in message:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type", "")).strip().lower() in _IMAGE_PART_TYPES:
                    return True
            return False
        # dict / attachment / provider-native payloads that are not plain text
        return True
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

    Returns ``None`` when routing is disabled, when this is not a first turn,
    or on any error. It never mutates ``agent``, never changes the effective
    provider/model, and never calls a model.
    """
    try:
        cfg = load_adaptive_config(config)
        if not cfg["enabled"] or (
            cfg["apply_routes"] and not cfg["shadow_mode"]
        ):
            # Inert path: no classification, no routing, no file, no log line.
            return None
        if not should_observe_first_turn(
            config=cfg, has_history=bool(conversation_history)
        ):
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
        record_shadow_decision(
            decision,
            effective_provider=effective_provider,
            effective_model=effective_model,
            session_id=getattr(agent, "session_id", ""),
            platform=getattr(agent, "platform", ""),
        )
        return decision
    except Exception:
        return None
