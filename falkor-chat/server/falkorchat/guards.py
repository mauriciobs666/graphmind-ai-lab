"""Transition-guard evaluation (M3 — K-022 Phase 2 / U7; the `cmp` family, K-024 U1).

A `TRANSITION.guard` is an opaque, set-on-create string (QUERIES §11.1). Its content
is a serialized discriminator, parsed **app-side only** (rule 8 — never filtered in
Cypher). `evaluate_guard` dispatches on the serialized `kind` to exactly **three** live
branches plus a raising seam (plans `m3-executor.md` §2.5, `m3-process-flow.md` §3.2):

  * **Unconditional** — the empty string ``""`` fires whenever the transition is reached
    (the deterministic default; **lowest priority** — the caller orders it last). This is
    how the **D5** research→answer transition is expressed (unconditional, not LLM-judged).
  * ``{"kind":"llm","text":…}`` — the fuzzy branch: the **Q1 extract-then-judge** method
    (`docs/plans/m3-executor-ml.md`). The evidence handed to the judge is **two-tier**:
    the compact `understanding` object (`{request, known, missing}`) the node emitted is the
    **primary** signal; when the node emitted none (prose output — the shipped intake def),
    the last `RECENT_TURNS_N` thread turns carried out on `StepResult.thread` are handed over
    as the DS-specified **degraded fallback** ("omit if understanding is present"). Judging a
    compact state beats judging a transcript, but a judge with *no* evidence at all can only
    bias to suspend forever — that was Defect A. Evidence goes to the
    **injected** `judge`, which returns a ``{decision, rationale}`` verdict. The Q1 ambiguity
    policy is applied here: parse failure / non-bool decision / a rationale that contradicts a
    `true` decision → **bias-to-suspend** (`decision=False`), safe for the human-unblockable
    intake guard (a false-suspend costs one more cheap clarifying question). The verdict +
    rationale are returned for the tracer (FR-4).
  * ``{"kind":"cmp"|"all"|"any"|"not", …}`` — the **deterministic** branch (K-024 D-A): a
    structured comparator over already-parsed JSON. **No parser, no `eval`, no dependency**
    — every op is a whitelisted Python callable (`_OPS`), every path root is one of two
    (`PATH_ROOTS`), and depth/node/width caps make a pathological guard structurally
    impossible. `validate_cmp` is the same rule set applied at **publish** time, so a
    typo'd op is an authoring error rather than a live run that parks forever.
  * **Any other kind** (e.g. a would-be ``expr``) → a pure `NotImplementedError` seam (M7 —
    no dead code). The comparator above is named `cmp` **precisely so `expr` stays shut**:
    DESIGN §13's resolution "no expression library is built" remains literally true, and
    the door to a DSL/`simpleeval` stays visibly closed. Do not implement `expr` here.

The judge is injected so this whole module is stub-testable offline — no LLM, no network.
The production judge (LLM-backed, building the DS §Q1 prompt) is wired in a later unit.

Two rules of the `cmp` family that are easy to get wrong, both deliberate:

**1. Totality — a missing path, or a type that cannot be compared, is `False` for every
op** (including `exists` and `ne`), and no `cmp` guard ever raises for *data* reasons.
This bias-to-not-fire mirrors `_coerce_verdict`'s bias-to-suspend: the safe error is to
park, because a parked run is unblockable by a human while a wrongly-advanced one is not.
Guards raise `WorkflowConfigError` only for *structural* faults — an unknown op, a bad
combinator arity, a cap breach — which are authoring defects, not run data.

**2. De Morgan does NOT hold on a missing path.** ``{"op":"ne","path":p,"value":v}`` and
``{"kind":"not","of":[{"op":"eq","path":p,"value":v}]}`` are **not** interchangeable when
`p` is absent: `ne` is a comparison against a value that is not there → `False` (rule 1),
while `not` negates a *verdict* — the inner `eq` is `False`, so the negation is `True`.
Both are correct for what they mean; they simply do not mean the same thing. Prefer `ne`
when you want "present and different"; use `not(eq)` only when a missing value should
fire the transition. The contrast is pinned by a dedicated test pair in
`tests/test_guards.py` (plan m-10) so it stays a documented decision, not a surprise.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

# DS §Q1: "RECENT TURNS (context only) … N = 6, newest last" — the fallback evidence tier.
RECENT_TURNS_N = 6
# Per-turn char cap (rule 6): one long turn must not crowd the window out of the judge's
# prompt. 6 × 400 ≈ 2.4k chars, comfortably inside the DS ~1500-token judge budget.
TURN_TEXT_MAX = 400

# Rationale cues that contradict a `decision:true` — an internally inconsistent verdict is
# treated as false (bias-to-suspend). The over-suspend direction is the safe error for a
# human-unblockable guard (DS note Q1); a stronger judge / more concrete guard text is the
# real fix if this trips too often.
#
# These are **deficiency assertions**, and each must be matched *in its polarity* — see
# `_rationale_contradicts`. A transcript-fed judge writes wordier rationales in which a cue
# often appears NEGATED ("no more info is needed", "nothing is unclear") and therefore
# AFFIRMS the advance. A bare substring match turns those into forced suspends — a failure
# indistinguishable from Defect A ("the guard never fires"). Hence the negator rule below.
_NEGATION_CUES: tuple[str, ...] = (
    "not enough", "insufficient", "still missing", "still need", "cannot ", "unable",
    "unclear", "not yet", "more info", "need more",
)
# Dropped from the cue set: "no relevant". It **embeds its own negator**, so the negator rule
# cannot resolve its polarity — "no relevant details are missing" (advance) and "no relevant
# information was provided" (suspend) are indistinguishable to it. It is the one cue that is
# unfixable rather than tightenable; "still need" replaces the coverage it was carrying.

# A cue immediately preceded by one of these is negated → it affirms rather than contradicts.
_NEGATORS: tuple[str, ...] = ("no ", "not ", "nothing ", "never ", "n't ")
# Only the text *immediately* before the cue counts. Wide enough for a copula ("nothing IS
# unclear"), deliberately too narrow to span a clause boundary: in "did not provide the
# version; more info is needed" the "not " belongs to another clause and must NOT negate the
# cue. Erring narrow keeps the failure on the safe (over-suspend) side — DS Q1.
_NEGATOR_WINDOW = 12


# ── the deterministic `cmp` comparator (K-024 U1, plan §3.2) ─────────────────

# The guard kinds the deterministic comparator owns. `expr` is deliberately NOT here:
# naming this family `cmp` is what keeps DESIGN §13's "no expression library is built"
# literally true, and keeps `{"kind":"expr"}` a visible, still-raising seam.
CMP_KINDS: frozenset[str] = frozenset({"cmp", "all", "any", "not"})

def _order(fn: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    """Wrap an ordering comparison so a mismatched/non-comparable pair is `False`.

    A raised `TypeError` would escape into the drive and fail the run; a guard that
    cannot decide must decline to fire (bias to not-fire), never crash.
    """
    def compare(left: Any, value: Any) -> bool:
        try:
            return bool(fn(left, value))
        except TypeError:
            return False
    return compare


def _contains(left: Any, value: Any) -> bool:
    """`value` ∈ the list/str at `path`. A non-container (or an unhashable probe) ⇒ False."""
    if not isinstance(left, (list, tuple, str)):
        return False
    try:
        return value in left
    except TypeError:
        return False


def _in(left: Any, value: Any) -> bool:
    """The value at `path` ∈ the list literal `value`. A non-list literal ⇒ False."""
    return _contains(value, left)


# The **closed** whitelist of comparison callables. Every op is a plain Python function of
# `(left, value)`; there is no parser and nothing is ever `eval`ed. An op outside this dict
# is a named, loud `WorkflowConfigError` — never a silent False.
_OPS: dict[str, Callable[[Any, Any], bool]] = {
    # `eq`/`ne` use plain Python `==` on JSON-native types — no coercion.
    "eq": lambda left, value: bool(left == value),
    "ne": lambda left, value: bool(left != value),
    "lt": _order(lambda left, value: left < value),
    "le": _order(lambda left, value: left <= value),
    "gt": _order(lambda left, value: left > value),
    "ge": _order(lambda left, value: left >= value),
    "in": _in,
    "contains": _contains,
    # Value-free ops: reaching them at all means the path resolved (see `_eval_leaf`).
    "exists": lambda left, value: True,
    "truthy": lambda left, value: bool(left),
}

# Ops that compare against a `value` literal; the rest (`exists`, `truthy`) are unary.
_VALUE_OPS: frozenset[str] = frozenset(_OPS) - {"exists", "truthy"}


class WorkflowConfigError(Exception):
    """A workflow def / executor misconfiguration surfaced loudly at drive time.

    Raised (m-3) when an `{"kind":"llm"}` guard is reached but the executor was built
    without a `guard_judge` — calling `None(...)` would otherwise be a bare `TypeError`
    that names no seam. The executor's M-1 fault net converts this into a `fail_run`
    carrying this clear message, so the run reaches a defined terminal instead of a
    zombie `running`. A subclass of `Exception`, so that net catches it.
    """


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
    guard: str | None,
    *,
    ctx: dict[str, Any],
    run: dict[str, Any],
    step_output: str,
    thread: Any,
    judge: Judge | None,
) -> GuardVerdict:
    """Evaluate one transition guard → a `GuardVerdict` (see module docstring).

    `thread` is the recent thread window carried out of the node on `StepResult.thread`
    (no extra graph read — m-C neutral). `None`/`[]`/malformed → the understanding-only
    path; it is only consulted when no `understanding` was emitted (the DS omit rule).

    A `cmp`-family guard consults neither the judge nor the thread — it reads only `ctx`
    and `step_output`, and is therefore fully deterministic and offline. Note the two
    module-docstring rules that govern it: **totality** (missing/uncomparable → `False`)
    and the **De Morgan asymmetry** between `ne` and `not(eq)` on a missing path.
    """
    # `None` and `""` both mean "no condition" → unconditional. Treating a null
    # guard here (not just the empty string) is the n-2 safety net: a hand-crafted
    # or pre-existing transition with a null `guard` fires-when-reached rather than
    # falling through to the `NotImplementedError` seam (which would orphan a drive).
    if not guard:
        return GuardVerdict(decision=True, rationale="unconditional (empty guard)")

    parsed = _load_obj(guard)
    kind = parsed.get("kind")
    if kind == "llm":
        # m-3: fail loudly and by name when no judge is wired for an llm guard, rather
        # than calling `None(...)` → an opaque TypeError. The M-1 net turns this into a
        # `fail_run` carrying the message.
        if judge is None:
            raise WorkflowConfigError(
                "no guard_judge is wired for an llm guard — the executor was built "
                "without a judge; wire a production judge before running llm-guarded defs"
            )
        understanding = _extract_understanding(step_output, ctx)
        # DS §Q1: RECENT TURNS is the *fallback* — "omit if understanding is present".
        # Truthiness, not presence: an emitted-but-empty `{}` understanding knows nothing,
        # so the turns are still the only evidence available. The precedence lives here,
        # not in the judge — it is a method decision every judge should inherit rather
        # than re-derive, and it stays unit-testable offline against a stub judge.
        recent_turns = [] if understanding else _recent_turns(thread)
        raw = judge(
            parsed.get("text", ""),
            understanding=understanding, recent_turns=recent_turns,
            ctx=ctx, step_output=step_output,
        )
        return _coerce_verdict(raw)

    if kind in CMP_KINDS:
        return _evaluate_cmp(parsed, ctx=ctx, step_output=step_output)

    raise NotImplementedError(
        f"guard kind {kind!r} is not supported in this cut "
        f"(expr/deterministic-expression seam, M7)"
    )


_MISSING = object()


def _evaluate_cmp(
    spec: dict[str, Any], *, ctx: dict[str, Any], step_output: str
) -> GuardVerdict:
    """Validate, then evaluate, a `cmp`-family guard into a traced `GuardVerdict`.

    Validation runs first and is the **same** `validate_cmp` the publish path calls
    (M-4): one implementation of the structural rules, two call sites. Everything the
    evaluator below assumes about shape is therefore already guaranteed.
    """
    # Structural rules are enforced at drive time too (an unknown op must never be a
    # silent False) — but paths stay total here; see `validate_cmp`.
    _validate_node(spec, depth=1, counter=[0], check_paths=False)
    decision = _eval_node(spec, ctx=ctx, step_output=step_output)
    return GuardVerdict(
        decision=decision,
        rationale=f"{render_label(spec)} → {'true' if decision else 'false'}",
    )


def _eval_node(spec: Any, *, ctx: dict[str, Any], step_output: str) -> bool:
    """Evaluate one **already-validated** guard node."""
    kind = _node_kind(spec)
    if kind in _COMBINATORS:
        children = spec["of"]
        if kind == "not":
            return not _eval_node(children[0], ctx=ctx, step_output=step_output)
        # `all([]) → True` / `any([]) → False`: the identity of each operator, chosen
        # deliberately rather than inherited by accident (plan §7 case 11).
        reduce = all if kind == "all" else any
        return reduce(
            _eval_node(child, ctx=ctx, step_output=step_output) for child in children
        )
    return _eval_leaf(spec, ctx=ctx, step_output=step_output)


_COMBINATORS: frozenset[str] = frozenset({"all", "any", "not"})


def _node_kind(spec: Any) -> str:
    """The kind of one guard node. Nested children may omit `kind` when they carry `op`."""
    kind = spec.get("kind") if isinstance(spec, Mapping) else None
    if isinstance(kind, str):
        return kind
    return "cmp"


def _eval_leaf(spec: Mapping[str, Any], *, ctx: dict[str, Any], step_output: str) -> bool:
    op = spec.get("op")
    left = _resolve_path(spec.get("path"), ctx=ctx, step_output=step_output)
    if left is _MISSING:
        return False
    return _OPS[op](left, spec.get("value"))


# Structural DoS caps (rule 6). A guard is data an author writes into a def; these make
# a pathological guard a *structural* impossibility rather than a runtime gamble. Guards
# are additionally bounded by `MAX_CONFIG_LEN` at the API boundary.
MAX_GUARD_DEPTH = 5    # maximum nesting levels (a bare leaf is depth 1)
MAX_GUARD_NODES = 32   # maximum total nodes in one guard
MAX_GUARD_WIDTH = 8    # maximum children of one `all`/`any`/`not`

# The two whitelisted path roots. `ctx.<key>…` reads the run ctx; `output.<key>…` reads
# the current step's JSON output and a bare `output` its raw string. There is no third
# root, no list indexing, no attribute access and no callable — traversal is dict-key
# lookup, so a guard can never reach anything but data.
PATH_ROOTS: frozenset[str] = frozenset({"ctx", "output"})


def validate_cmp(spec: Any) -> None:
    """Structurally validate a `cmp`-family guard; raise `WorkflowConfigError` if unsound.

    Takes an **already-parsed dict** — normalizing a string-shaped guard is the caller's
    job (`services._validate_def_spec` for the publish path). Called both at publish time
    (so a typo'd `op` is an authoring error, not a dead live run) and at drive time, so
    the rules exist exactly once.

    Rejects: an unknown `op`; a `path` whose root is not `ctx.`/`output.` (bare `output`
    allowed, bare `ctx` not — it is the whole run state, not a value); a missing `path`,
    or a missing `value` for an op that compares against one; a combinator without a list
    `of`; a `not` whose arity is not exactly 1; and any depth / node-count / width breach.

    **Paths are strict here and total at drive time**, deliberately: an unresolvable path
    is an authoring defect worth rejecting at publish, but at drive time it is just a
    value that is not there → `False` (see the module docstring's totality rule). Same
    validator, one flag — `_evaluate_cmp` passes `check_paths=False`.
    """
    _validate_node(spec, depth=1, counter=[0], check_paths=True)


def _validate_node(
    spec: Any, *, depth: int, counter: list[int], check_paths: bool
) -> None:
    if depth > MAX_GUARD_DEPTH:
        raise WorkflowConfigError(
            f"guard nesting depth exceeds the cap of {MAX_GUARD_DEPTH}"
        )
    counter[0] += 1
    if counter[0] > MAX_GUARD_NODES:
        raise WorkflowConfigError(
            f"guard node count exceeds the cap of {MAX_GUARD_NODES}"
        )
    if not isinstance(spec, Mapping):
        raise WorkflowConfigError(
            f"guard node must be an object, got {type(spec).__name__}"
        )

    kind = _node_kind(spec)
    if kind in _COMBINATORS:
        of = spec.get("of")
        if not isinstance(of, list):
            raise WorkflowConfigError(
                f"guard combinator {kind!r} requires a list `of` "
                f"(got {type(of).__name__})"
            )
        if len(of) > MAX_GUARD_WIDTH:
            raise WorkflowConfigError(
                f"guard combinator {kind!r} width {len(of)} exceeds the cap of "
                f"{MAX_GUARD_WIDTH}"
            )
        if kind == "not" and len(of) != 1:
            raise WorkflowConfigError(
                f"guard combinator 'not' takes exactly one child, got {len(of)}"
            )
        for child in of:
            _validate_node(
                child, depth=depth + 1, counter=counter, check_paths=check_paths
            )
        return

    if kind != "cmp":
        raise WorkflowConfigError(
            f"guard kind {kind!r} is not part of the cmp family "
            f"({', '.join(sorted(CMP_KINDS))})"
        )
    op = spec.get("op")
    if op not in _OPS:
        raise WorkflowConfigError(
            f"unknown guard op {op!r} — allowed ops: {', '.join(sorted(_OPS))}"
        )
    if check_paths:
        _validate_path(spec.get("path"))
    if op in _VALUE_OPS and "value" not in spec:
        raise WorkflowConfigError(f"guard op {op!r} requires a `value`")


def _validate_path(path: Any) -> None:
    if not isinstance(path, str) or not path:
        raise WorkflowConfigError(
            f"guard `path` must be a non-empty string, got {path!r}"
        )
    head, _, rest = path.partition(".")
    if head not in PATH_ROOTS:
        raise WorkflowConfigError(
            f"guard path root {head!r} is not whitelisted "
            f"(allowed: {', '.join(sorted(PATH_ROOTS))})"
        )
    if head == "ctx" and not rest:
        raise WorkflowConfigError("guard path 'ctx' must name a key, e.g. 'ctx.decision'")
    if rest and any(not segment for segment in rest.split(".")):
        raise WorkflowConfigError(f"guard path {path!r} has an empty segment")


def render_label(spec: Any) -> str:
    """A compact, human-readable rendering of a guard — the M-6 trace label.

    A `cmp` guard has no `text` field, so the executor's trace line would otherwise start
    with a bare `" -> "`. Deliberately total and side-effect-free: it never raises, so a
    malformed guard still traces (the loud failure is `validate_cmp`'s job, not a
    renderer's).
    """
    if not isinstance(spec, Mapping):
        return repr(spec)
    kind = _node_kind(spec)
    if kind in _COMBINATORS:
        of = spec.get("of")
        children = of if isinstance(of, list) else []
        return f"{kind}({', '.join(render_label(child) for child in children)})"
    op = spec.get("op")
    path = spec.get("path")
    if op in _VALUE_OPS:
        return f"{path} {op} {spec.get('value')!r}"
    return f"{path} {op}"


def _resolve_path(path: Any, *, ctx: dict[str, Any], step_output: str) -> Any:
    """Resolve a whitelisted `ctx.…` / `output.…` path → the value, or `_MISSING`."""
    if not isinstance(path, str):
        return _MISSING
    head, _, rest = path.partition(".")
    if head == "ctx":
        # A bare `ctx` is not a value — it is the whole run state. Only `ctx.<key>…`
        # resolves, and `validate_cmp` rejects the bare root at publish time to match.
        if not rest:
            return _MISSING
        cursor: Any = ctx
    elif head == "output":
        cursor = step_output if not rest else _load_obj(step_output)
    else:
        return _MISSING
    if not rest:
        return cursor
    for key in rest.split("."):
        if not isinstance(cursor, Mapping) or key not in cursor:
            return _MISSING
        cursor = cursor[key]
    return cursor


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
    """True when the rationale *asserts a deficiency*, i.e. contradicts a `decision:true`.

    Polarity-aware by necessity (R-1): a cue is only a contradiction when it is **not
    negated**. "Not enough information" contradicts an advance; "no more info is needed"
    affirms it, yet both contain a cue. The check is a backstop against an internally
    inconsistent verdict — it is not a prose grep for negative-sounding words.
    """
    low = rationale.lower()
    for cue in _NEGATION_CUES:
        start = low.find(cue)
        while start != -1:
            if not _is_negated(low, start):
                return True
            start = low.find(cue, start + 1)
    return False


def _is_negated(low: str, cue_start: int) -> bool:
    """Is the cue at `cue_start` immediately preceded by a negator (→ it affirms)?"""
    window = low[max(0, cue_start - _NEGATOR_WINDOW):cue_start]
    return any(neg in window for neg in _NEGATORS)


def _recent_turns(thread: Any, n: int = RECENT_TURNS_N) -> list[dict[str, str]]:
    """The 'fallback' half of extract-then-judge (DS §Q1): the last `n` thread turns,
    newest last, normalized to a compact `{speaker, role, text}`.

    Input rows are `repository.read_thread` shape and already **chronological**
    (`ORDER BY m.createdAt`), so `thread[-n:]` *is* "newest last". Tolerant by design —
    `None` / non-list / malformed rows → `[]`; a guard must never crash a drive (the
    offline stub path and non-agent steps legitimately carry no thread).
    """
    if not isinstance(thread, list):
        return []
    turns: list[dict[str, str]] = []
    for row in thread[-n:]:
        if not isinstance(row, Mapping):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        speaker = row.get("displayName") or row.get("authorId") or "member"
        role = row.get("role") or "user"
        turns.append({
            "speaker": str(speaker), "role": str(role), "text": text[:TURN_TEXT_MAX],
        })
    return turns


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
