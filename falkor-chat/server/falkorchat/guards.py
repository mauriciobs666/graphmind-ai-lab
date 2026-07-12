"""Transition-guard evaluation (M3, K-022 — Phase 2 / U7).

A `TRANSITION.guard` is an opaque, set-on-create string (QUERIES §11.1). Its content
is a serialized discriminator, parsed **app-side only** (rule 8 — never filtered in
Cypher). `evaluate_guard` dispatches on the serialized `kind` to exactly **two** live
branches plus a raising seam (plan §2.5):

  * **Unconditional** — the empty string ``""`` fires whenever the transition is reached
    (the deterministic default; **lowest priority** — the caller orders it last). This is
    how the **D5** research→answer transition is expressed (unconditional, not LLM-judged).
  * ``{"kind":"llm","text":…}`` — the fuzzy branch: the **Q1 extract-then-judge** method
    (`docs/plans/m3-executor-ml.md`). The compact `understanding` object (`{request, known,
    missing}`) the node emitted is extracted and handed — *not the raw transcript* — to the
    **injected** `judge`, which returns a ``{decision, rationale}`` verdict. The Q1 ambiguity
    policy is applied here: parse failure / non-bool decision / a rationale that contradicts a
    `true` decision → **bias-to-suspend** (`decision=False`), safe for the human-unblockable
    intake guard (a false-suspend costs one more cheap clarifying question). The verdict +
    rationale are returned for the tracer (FR-4).
  * **Any other kind** (e.g. a would-be ``expr``) → a pure `NotImplementedError` seam (M7 —
    no dead code). A full expression evaluator is a deferred slice; do not build it here.

The judge is injected so this whole module is stub-testable offline — no LLM, no network.
The production judge (LLM-backed, building the DS §Q1 prompt) is wired in a later unit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

# Rationale cues that contradict a `decision:true` — an internally inconsistent verdict is
# treated as false (bias-to-suspend). The over-suspend direction is the safe error for a
# human-unblockable guard (DS note Q1); a stronger judge / more concrete guard text is the
# real fix if this trips too often.
_NEGATION_CUES: tuple[str, ...] = (
    "not enough", "insufficient", "still missing", "cannot ", "unable",
    "unclear", "no relevant", "not yet", "more info", "need more",
)


@dataclass(frozen=True)
class GuardVerdict:
    """A guard judgment: a boolean `decision` plus a traced `rationale` (FR-4)."""

    decision: bool
    rationale: str = ""


# The injected fuzzy-guard judge: scores a CONDITION against the compact CURRENT STATE and
# returns a ``{decision, rationale}`` mapping (DS §Q1). Kept a plain callable so a stub drives
# it offline; the production LLM-backed judge builds the §Q1 prompt behind this same shape.
Judge = Callable[..., Any]


def evaluate_guard(
    guard: str,
    *,
    ctx: dict[str, Any],
    run: dict[str, Any],
    step_output: str,
    thread: Any,
    judge: Judge,
) -> GuardVerdict:
    """Evaluate one transition guard → a `GuardVerdict` (see module docstring)."""
    if guard == "":
        return GuardVerdict(decision=True, rationale="unconditional (empty guard)")

    parsed = _load_obj(guard)
    kind = parsed.get("kind")
    if kind == "llm":
        understanding = _extract_understanding(step_output, ctx)
        raw = judge(
            parsed.get("text", ""),
            understanding=understanding, ctx=ctx, step_output=step_output,
        )
        return _coerce_verdict(raw)

    raise NotImplementedError(
        f"guard kind {kind!r} is not supported in this cut "
        f"(expr/deterministic-expression seam, M7)"
    )


def _coerce_verdict(raw: Any) -> GuardVerdict:
    """Normalize an injected judge's output into a `GuardVerdict`, biasing to suspend.

    Only a clean, internally-consistent `{"decision": True, ...}` advances; anything else —
    a non-mapping, a missing/non-bool `decision`, or a rationale that contradicts a `true`
    decision — resolves to `decision=False` (Q1 bias-to-suspend for the intake guard).
    """
    if not isinstance(raw, Mapping):
        return GuardVerdict(False, "malformed judge output (bias-to-suspend)")
    rationale = raw.get("rationale")
    rationale = rationale if isinstance(rationale, str) else ""
    # Strict: only the real bool True advances (not "true", not 1, not None/missing).
    if raw.get("decision") is not True:
        return GuardVerdict(False, rationale or "not clearly met (bias-to-suspend)")
    if _rationale_contradicts(rationale):
        return GuardVerdict(
            False, f"rationale contradicts advance (bias-to-suspend): {rationale}"
        )
    return GuardVerdict(True, rationale)


def _rationale_contradicts(rationale: str) -> bool:
    low = rationale.lower()
    return any(cue in low for cue in _NEGATION_CUES)


def _extract_understanding(step_output: str, ctx: dict[str, Any]) -> dict[str, Any]:
    """The 'extract' half of extract-then-judge: pull the compact `understanding` object.

    The intake node emits its `{request, known, missing}` state as structured output; prefer
    that (unwrapping an `understanding` envelope if present), then fall back to an
    `understanding` carried in `ctx`, then to an empty object. Judging this compact state —
    not the raw transcript — is the Q1 reliability decision.
    """
    obj = _load_obj(step_output)
    if obj:
        inner = obj.get("understanding")
        return inner if isinstance(inner, dict) else obj
    inner = ctx.get("understanding") if isinstance(ctx, dict) else None
    return inner if isinstance(inner, dict) else {}


def _load_obj(raw: str | None) -> dict[str, Any]:
    """Deserialize an opaque string to a dict; `None`/``""``/non-object → ``{}``."""
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}
