"""The runner: capture-order orchestration, the two driving loops, and `LatencyBlock` accumulation.

Design: `docs/plans/small-model-benchmarking-runner-spec.md` (the "spec") §4/§5/§7 Step 1, itself
synthesized from `docs/plans/small-model-benchmarking.md` §3.4.4a, §3.6, §3.8.4, §4 S1/S2, and
`docs/plans/small-model-benchmarking-ml-dispatch-failure.md`. Every signature and invariant below
is the spec's unless a comment says otherwise; **three genuine gaps in the spec's own pseudocode**
were found and resolved while building this — each is called out at its own site, not buried here,
because the spec's own §9 already asks that of the next reader:

1. **`design_effect` for `_drive_conversations`.** The spec's pseudocode calls
   `stats.design_effect(...)` with the arguments elided (`§4`, `§9`'s own flagged risk). Reading
   `docs/plans/small-model-benchmarking-ml.md` directly (as §7 Step 1 instructs) settles it a
   different way than a bootstrap-width call: §4.5.1's table and R1's row both state
   **"DEFF 1.00 by construction"** for the 12×1 tool-caller design — the sampling unit (script) and
   the analysis unit (script) coincide, exactly the `_drive_single_call_items` argument generalised
   to the one pack that clusters turns *within* a script but never scripts against each other.
   `stats.design_effect(bootstrap_width, naive_width)` is Rule 5's general **variance-ratio**
   utility for a design that *does* cluster its sampling unit — it is not this run's number to
   compute, so `_drive_conversations` returns `1.0` unconditionally, exactly like the item-level
   loop, and never calls `stats.design_effect`.
2. **`latency_block`'s call-surface blindness (§5 rule iv-a).** The spec's signature is
   `latency_block(items) -> LatencyBlock | None` — no call-surface parameter. Rule (iv-a) requires
   `statsCoveredCount` to read `None` on a surface that returns no `stats` at all (embeddings) and a
   real `0` on a chat surface where every call happened to lack `stats` — two states that are
   **structurally identical** in `items` alone (both are "every call's `ttftMs` is `None`"), so no
   post-hoc reading of `items` can recover which one occurred. `latency_block` therefore takes a
   keyword-only `call_surface` here, threaded from `run_pack`'s own already-computed value; nothing
   else in the nine invariants needed it.
3. **`aggregates`'s source for `RunResult`.** The spec's `run_pack` pseudocode constructs
   `RunResult(..., aggregates=aggregates, ...)` from a variable `_drive_scored_items` never returns
   for the item-level branch — `ItemScorer.score_item` (§3.2) yields one `ItemResult` per item and
   no run-level aggregation method at all, unlike `ConversationScorer.score_conversations`, which
   does return `(items, aggregates)`. `ItemScorer` gains one method beyond the spec's own two,
   `aggregate(items, *, pack) -> Aggregates`, so `run_pack` has a symmetric source on both loops.
   Both scorer seams are live: `_load_item_scorer` resolves a real pack's scorer since S3, and
   `_load_conversation_scorer` does the same as of S5 — `_drive_conversations` calls a real
   `ConversationScorer` end to end (spec §3.2's own scorer-seam caveat, resolved rather than open).

**Everything here is offline-testable.**
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Protocol

import modelbench
from modelbench import hostinfo
from modelbench.convo import (
    ChatMessage,
    Conversation,
    ConversationTrace,
    PromptConfig,
    ToolDispatchFailed,
    drive,
)
from modelbench.decisions import (
    BAR_KIND_BY_ROLE,
    DECISION_POINT_CHAT_TIMEOUT_SECONDS,
    HANDOFF_FAILURE_BY_ROLE,
    Candidate,
    CandidateError,
    FitManifest,
    apply_policy,
    eval_set_hash,
    is_over_limit,
    leaked_items,
    load_fit_manifest,
    question_spec_hash,
)
from modelbench.fingerprint import Fingerprint
from modelbench.lmstudio import (
    WARMUP_DECISION_INPUT,
    ChatResult,
    DecisionAnswer,
    DecisionResult,
    EmbedResult,
    LMStudio,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LMStudioUnreachable,
    LoadedInstance,
    LoadResult,
    ModelInfo,
    ResidentModel,
    check_tool_calling_eligibility,
)
from modelbench.packs import (
    Pack,
    answer_space_problem,
    derive_call_surface,
    derive_decision_support,
)
from modelbench.results import (
    BENCH_SCHEMA_VERSION,
    Aggregates,
    BinaryMetric,
    CallTiming,
    ItemResult,
    ItemTiming,
    LatencyBlock,
    RunResult,
)
from modelbench.stats import LEVEL_P50, LEVEL_P95, Basis, percentile
from modelbench.tooling import ToolEnvironment

#: `-ml` §11.5.1's ruling: `unexplainedMs` withholds a `"load"` cause once the item's summed
#: per-call gap exceeds this (milliseconds; **not** the 3 485.6 ms figure test 15b's fixtures use,
#: which only bounds the threshold from above — spec §4/§9, `-ml` §11.5.1 read directly).
UNEXPLAINED_MS_THRESHOLD: float = 1000.0

#: `-ml` §11.6's two named floors, in percentage points of the *nominal* level (50 / 95).
_P50_FLOOR_PCT: int = 45
_TAIL_FLOOR_PCT: int = 90

#: Must equal `hostinfo._residency_source_after_a_successful_probe()`'s own literal — the value
#: `attest` writes into `host.json`'s `observedAtAttestation.residencySource`, which
#: `hostinfo.check_attestation_staleness` reads back at capture-order step 6. A second literal here
#: would silently mark every run stale from the first `"compared"` run on.
_RESIDENCY_SOURCE: str = "lmstudio-api-v0"

#: The chat/embeddings call budget when `run` is given no `--request-timeout` (plan §3.6).
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 120.0

_PROBE_MESSAGES: dict[str, str] = {
    "v1-only": (
        "LM Studio serves an OpenAI-compatible API but not its native /api/v0 catalog. "
        "model-bench fingerprints from that catalog and will not run against a server it cannot "
        "fingerprint (plan §3.4.4a)."
    ),
    "unreachable": "LM Studio is not reachable (plan §3.4.4a).",
}


# --- public shapes (spec §4) -------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """`run`'s flags (plan §3.6a `run --pack <id> --model <key>`) plus the two budgets §3.6 sizes
    for the runner to pass (never a pack/adapter default — both are required keyword args on
    `LMStudio.chat`/`embed`/`warm_up` already).

    **The arm is exactly one of `modelKey` (a chat/embeddings arm) and `candidateId` (a decision-
    model candidate, `run --candidate`; typed-decision-models plan v1.3 §3.3)** — refused here
    otherwise, so no path can build a run with both or neither. `candidateId` defaults to `None`
    so every pre-existing chat/embeddings construction site is unchanged.

    **`requestTimeoutSeconds` is the explicit `--request-timeout`, or `None` when none was given**
    — `run_pack` resolves it (`_resolve_request_timeout`): `DEFAULT_REQUEST_TIMEOUT_SECONDS`, or
    the pinned `DECISION_POINT_CHAT_TIMEOUT_SECONDS` on a decision-point pack, where any other
    explicit value refuses (typed-decision-models plan2 §3.2). The `None` is what tells "no flag"
    from an explicit `--request-timeout 120`."""

    modelKey: str | None
    sessionId: str | None
    referenceKey: str | None
    warmupExtra: int = 0
    firstCallTimeoutSeconds: float = 300.0
    requestTimeoutSeconds: float | None = None
    candidateId: str | None = None

    def __post_init__(self) -> None:
        if (self.modelKey is None) == (self.candidateId is None):
            raise ValueError(
                "RunConfig: exactly one of modelKey and candidateId is set "
                f"(modelKey={self.modelKey!r}, candidateId={self.candidateId!r})"
            )

    @property
    def armKey(self) -> str:
        """The arm's identity (v1.3 §3.3 "Arm identity"): the candidate id on a candidate run,
        else the model key — what the run id is built from."""
        if self.candidateId is not None:
            return self.candidateId
        assert self.modelKey is not None  # __post_init__: exactly one is set
        return self.modelKey


class RunRefused(RuntimeError):
    """The run stopped before or during capture order and wrote nothing (exit 3/4/5 territory,
    plan §3.6a). Carries `exitCode` so the CLI does not re-derive it from message text."""

    def __init__(self, message: str, *, exitCode: int) -> None:
        super().__init__(message)
        self.exitCode = exitCode


@dataclass(frozen=True)
class DispatchFailureDisclosure:
    """One censored conversation, for the CLI's `PACK DISPATCH FAILURES` block (dispatch-failure
    note §4(d)). `(scriptId, turn, tool, reason)`, verbatim."""

    scriptId: str
    turn: int
    tool: str
    reason: str


# --- the scorer seam (spec §3.2) ----------------------------------------------------------------


class ItemScorer(Protocol):
    """One item-level role's scorer (embedder, guard-judge, nlq-generator, chat-responder).

    `aggregate` is **not** in the spec's own §3.2 sketch — added here because `run_pack` cannot
    build a `RunResult` without a source for `aggregates` and no other one exists for this loop
    (module docstring, gap 3). `ConversationScorer.score_conversations` already returns its
    aggregates; this gives the item-level loop the same shape rather than a special case.
    """

    def score_item(
        self,
        item_input: Mapping[str, Any],
        result: ChatResult | EmbedResult | None,
        timing: ItemTiming,
        *,
        pack: Pack,
    ) -> ItemResult: ...

    def aggregate(self, items: Sequence[ItemResult], *, pack: Pack) -> Aggregates: ...

    def prime(
        self,
        *,
        lmstudio: LMStudio,
        model_info: ModelInfo,
        call_surface: Literal["chat", "embeddings"],
        timeout_s: float,
        pack: Pack,
    ) -> None:
        """Called once, before the per-item loop, iff the scorer defines it (spec §3.2's own
        extension to the shipped Protocol, S3). The one hook that gives a scorer model access
        outside a scored item — the embedder's own use is embedding and caching the reference
        corpus; no other role needs one yet. Optional: `getattr`-guarded at the one call site
        (`_drive_single_call_items`), so a scorer that omits it is untouched."""
        ...

    def embed_text(self, item_input: Mapping[str, Any], *, pack: Pack) -> str:
        """Called instead of the hardcoded `json.dumps(item_input, sort_keys=True)` on the
        embeddings call branch (spec §3.2/§2.2). Required in practice for any embeddings-surface
        role's scorer — reached with no `getattr` guard, since it is only ever called on the
        embeddings branch, which today only the embedder ever declares."""
        ...

    def build_messages(
        self, item_input: Mapping[str, Any], *, pack: Pack,
    ) -> list[ChatMessage]:
        """Called instead of the generic, scorer-blind `_item_chat_messages` on the chat branch,
        iff the scorer defines it (optional, `getattr`-guarded — S4 spec §4.1, closing
        itemscorer-extension.md finding 1). `guard-judge`'s real prompt is a labelled multi-block
        render, `nlq-generator`'s is a schema-filled structured-completion instruction set — neither
        is a JSON dump of the item row. Falls back to `_item_chat_messages` when absent, so every
        existing S1/S2 test (`FakeItemScorer`, which defines neither) is untouched."""
        ...

    # --- typed-decision-models (plan v1.3 §3.3 "Scorer seam"; plan2 §3.3; plan3 §3.3) ---------
    # Three optional hooks, needed only on the `decisions`/`handoff` surfaces. A chat or
    # embeddings run never reaches them, so a scorer without them is untouched;
    # `_decision_scorer_problem` refuses a decisions run whose scorer lacks one, up front.

    def decision_input(self, item_input: Mapping[str, Any], *, pack: Pack) -> str:
        """The decision model's `input` string for one item: the role's §3.2 renderer."""
        ...

    def answer_from_chat(self, result: ChatResult, *, pack: Pack) -> tuple[Any, str]:
        """The role's hand-off chat parse alone (§3.2): `(answer, outcome)` for one reply."""
        ...

    def score_answer(
        self,
        item_input: Mapping[str, Any],
        answer: Any,
        outcome: str,
        timing: ItemTiming,
        *,
        pack: Pack,
        decided_by: str | None,
        probability: float | None,
        failed_call: str | None,
        reason: str | None,
        input_tokens: int | None,
    ) -> ItemResult:
        """One item's `ItemResult` from an answer, wherever it came from. The five keywords have
        no default (plan2 §3.3, §3.4; plan3 §3.3): `decided_by` (`"decision-model"`/`"handoff"`),
        the decision's `probability` (a predicate's p, a choice's confidence), `failed_call`,
        `reason` (`"over-limit"`) and `input_tokens` (`usage["input_tokens"]`). Their legal
        combinations are the scorer's to enforce (`classification.score_answer`)."""
        ...


class ConversationScorer(Protocol):
    """`tool-caller`'s scorer — needs the whole (possibly censored) trace for cross-turn state
    (the hazard, the per-position table, `cleanThroughTurnH`'s censoring, `-ml` §4.3 rule 5)."""

    def score_conversations(
        self,
        scored: Sequence[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]],
        probes: Sequence[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]],
        *,
        pack: Pack,
    ) -> tuple[tuple[ItemResult, ...], Aggregates]: ...


def _load_item_scorer(pack: Pack) -> ItemScorer:
    """Resolve `pack.manifest["scorer"]` to a `modelbench.scoring.<name>` module (S3 spec §7.1's
    own naming choice, now wired) and return the module itself — a module satisfies `ItemScorer`
    structurally (its `score_item`/`aggregate` module-level functions ARE the Protocol's methods;
    never `isinstance`-checked). Raises `RunRefused(exitCode=4)` on an absent `"scorer"` key or an
    unresolvable module name — a pack-config defect, not a programming-contract one (unlike
    `embed_text`'s uncaught `AttributeError`, S3 spec §3.2)."""
    name = pack.manifest.get("scorer")
    if not name:
        raise RunRefused(f"pack {pack.packId!r} declares no \"scorer\"", exitCode=4)
    try:
        return importlib.import_module(f"modelbench.scoring.{name}")
    except ImportError as exc:
        raise RunRefused(
            f"pack {pack.packId!r} declares scorer {name!r}, which does not resolve to "
            f"modelbench.scoring.{name}: {exc}",
            exitCode=4,
        ) from exc


def _load_conversation_scorer(pack: Pack) -> ConversationScorer:
    """Resolve `pack.manifest["scorer"]` to a `modelbench.scoring.<name>` module — `_load_item_
    scorer`'s own resolution, mirrored exactly (S5 spec §5 Step 6): a module satisfies
    `ConversationScorer` structurally (its `score_conversations` module-level function IS the
    Protocol's method), and the same `RunRefused(exitCode=4)` contract applies to an absent
    `"scorer"` key or an unresolvable module name — a pack-config defect, not a programming-contract
    one."""
    name = pack.manifest.get("scorer")
    if not name:
        raise RunRefused(f"pack {pack.packId!r} declares no \"scorer\"", exitCode=4)
    try:
        return importlib.import_module(f"modelbench.scoring.{name}")
    except ImportError as exc:
        raise RunRefused(
            f"pack {pack.packId!r} declares scorer {name!r}, which does not resolve to "
            f"modelbench.scoring.{name}: {exc}",
            exitCode=4,
        ) from exc


# --- shared timing helpers (spec §3.3, §4) --------------------------------------------------


def _is_resident(model_id: str, resident: Sequence[ResidentModel]) -> bool:
    return model_id in {m.id for m in resident}


def _load_withheld_for(
    model_info: ModelInfo, resident_before: list[ResidentModel]
) -> Literal["load"] | None:
    """The between-item residency probe (plan §3.6): withheld for load iff the PRECEDING snapshot
    did not show the model resident. Never evaluated during a timed call."""
    return None if _is_resident(model_info.id, resident_before) else "load"


def _between_item_residency(lmstudio: LMStudio, *, where: str) -> list[ResidentModel]:
    """The between-item residency snapshot, with a failed probe ruled (typed-decision-models plan4
    §3.10). `residency()` failing in any of its three ways (`LMStudioUnreachable`,
    `LMStudioCallFailed`, a socket `TimeoutError`) re-probes the server: anything but `"api-v0"`
    is an outage and refuses the run with exit 3 and step 2's message, storing nothing (SMB plan
    §3.6 clause (iv)). Otherwise the snapshot is `[]` — "not shown resident", never "resident" —
    and one stderr line names `where` (the probe's place in the run) and the exception class. What
    `[]` does is each caller's existing load rule: a completed chat call is withheld for
    `"load"`, a failed one keeps its own disposition, a decision item takes `"load"`."""
    try:
        return lmstudio.residency()
    except (LMStudioUnreachable, LMStudioCallFailed, TimeoutError) as exc:
        probe_result = lmstudio.probe()
        if probe_result != "api-v0":
            raise RunRefused(_PROBE_MESSAGES[probe_result], exitCode=3) from exc
        print(
            f"model-bench: the between-item residency probe {where} failed "
            f"({type(exc).__name__}: {exc}); LM Studio still answers, so the snapshot reads "
            "as not shown resident (typed-decision-models plan4 §3.10)",
            file=sys.stderr,
        )
        return []


def _gap_ms(call: CallTiming) -> float | None:
    """`-ml` §11.5.1's per-call gap: `wallClockMs − (ttftMs + generationMs)`. `None` if any operand
    is `None` (a chat-surface-only figure). This is `results._call_gap_ms`'s own formula — the
    plan's to state at the unit boundary, not `-ml`'s (§3.6 `:1487-1512`) — reused here rather than
    duplicated so the two copies cannot drift."""
    if call.wallClockMs is None or call.ttftMs is None or call.generationMs is None:
        return None
    return call.wallClockMs - (call.ttftMs + call.generationMs)


def _gap_withheld_for(item_timing: ItemTiming) -> Literal["load"] | None:
    """Sums `_gap_ms` over `item_timing.calls` via `ItemTiming.unexplainedMs`; `None` unless
    EVERY call yields a gap (plan `:3013-3014`). Withholds iff the sum exceeds `-ml` §11.5.1's
    threshold."""
    gap = item_timing.unexplainedMs
    return "load" if gap is not None and gap > UNEXPLAINED_MS_THRESHOLD else None


def _call_timing_from_result(call: ChatResult | EmbedResult) -> CallTiming:
    """One `ChatResult`/`EmbedResult` as a `CallTiming` — the unit-boundary fields are already
    normalised on `ChatResult` (`lmstudio.py`'s own docstring); an `EmbedResult` carries no `stats`
    at all, so its three sibling figures are `None` (§3.6, rule iv-a's call-surface condition)."""
    if isinstance(call, ChatResult):
        usage = call.usage if isinstance(call.usage, Mapping) else {}
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            prompt_tokens = None
        return CallTiming(
            wallClockMs=call.wallClockMs,
            ttftMs=call.ttftMs,
            generationMs=call.generationMs,
            promptTokens=prompt_tokens,
            tokensPerSecond=call.tokensPerSecond,
        )
    return CallTiming(
        wallClockMs=call.wallClockMs,
        ttftMs=None,
        generationMs=None,
        promptTokens=None,
        tokensPerSecond=None,
    )


def _item_chat_messages(pack: Pack, item_input: Mapping[str, Any]) -> list[ChatMessage]:
    """Generic single-call message assembly for an item-level role's chat surface — the pack's
    system prompt (if any) plus the item's own declared content, JSON-encoded.

    **Deliberately role-agnostic and not specified anywhere** (the spec's own `_drive_single_call_
    items` pseudocode elides this call's messages with `...`). Each role's real prompt template
    (`prompts/*.md`, built by S3-S7's own scorer units) is a scorer concern this document does not
    invent; this is only what lets the runner issue a real call for a stub or fixture item today.
    """
    cfg = pack.prompt_config()
    messages: list[ChatMessage] = []
    if cfg.systemPrompt:
        messages.append({"role": "system", "content": cfg.systemPrompt})
    messages.append({"role": "user", "content": json.dumps(item_input, sort_keys=True)})
    return messages


def _drive_single_call_items(
    pack: Pack,
    cfg: RunConfig,
    *,
    lmstudio: LMStudio,
    model_info: ModelInfo,
    call_surface: Literal["chat", "embeddings"],
    baseline_residency: list[ResidentModel],
) -> tuple[tuple[ItemResult, ...], float, Basis]:
    """The four item-level roles. One call per item; `ItemTiming.calls` is always length 0 or 1.
    `designEffect == 1.0`, `basis == "by-construction"` — no clustering: sampling unit == item
    (plan §3.8.1's "designEffect 1.00 by construction" for the embedder, generalised here to every
    item-unit role since none of them cluster observations — spec §4's own inference, not quoted
    per-role)."""
    scorer = _load_item_scorer(pack)
    prime = getattr(scorer, "prime", None)
    if prime is not None:  # NEW — S3 spec §3.2: the corpus-embedding hook, once before the loop
        # typed-decision-models plan5 §3.1 (U2.4): `prime` runs before any item and the run
        # cannot continue without it, so a failed corpus embed takes the warm-up rule — exit 3,
        # no re-probe. The catch is the runner's (scorers do not know `RunRefused`); any other
        # exception passes through.
        try:
            prime(
                lmstudio=lmstudio,
                model_info=model_info,
                call_surface=call_surface,
                timeout_s=cfg.requestTimeoutSeconds,
                pack=pack,
            )
        except (LMStudioCallTimeout, LMStudioCallFailed) as exc:
            raise RunRefused(
                f"corpus embed of {model_info.id!r} failed before any item ({exc}); "
                "nothing was scored",
                exitCode=3,
            ) from exc
    prompt_config = pack.prompt_config()
    resident = baseline_residency
    results: list[ItemResult] = []
    for item_input in pack.iter_items():
        resident = _between_item_residency(  # between-item probe — never during a timed call
            lmstudio, where=f"before item {item_input.get('itemId')!r}"
        )
        try:
            if call_surface == "chat":
                build_messages = getattr(scorer, "build_messages", None)  # NEW — S4 spec §4.1
                messages = (
                    build_messages(item_input, pack=pack)
                    if build_messages is not None
                    else _item_chat_messages(pack, item_input)
                )
                call: ChatResult | EmbedResult = lmstudio.chat(
                    messages,
                    model=model_info.id,
                    temperature=prompt_config.temperature,
                    max_tokens=prompt_config.maxTokens,
                    timeout_s=cfg.requestTimeoutSeconds,
                )
            else:
                # NEW — S3 spec §3.2/§2.2: the pack's own `embed_text`, not a hardcoded dump of
                # the whole item row. No `getattr` guard: reached only on the embeddings branch,
                # which today only the embedder role ever declares.
                embed_text = scorer.embed_text
                call = lmstudio.embed(
                    [embed_text(item_input, pack=pack)],
                    model=model_info.id,
                    timeout_s=cfg.requestTimeoutSeconds,
                )
        except LMStudioCallTimeout:
            timing = ItemTiming(wallClockMs=None, calls=(), withheldFor="timeout")
            results.append(scorer.score_item(item_input, None, timing, pack=pack))
            continue
        except LMStudioCallFailed:
            timing = ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")
            results.append(scorer.score_item(item_input, None, timing, pack=pack))
            continue
        call_timing = _call_timing_from_result(call)
        probe_timing = ItemTiming(
            wallClockMs=call.wallClockMs, calls=(call_timing,), withheldFor=None
        )
        withheld = _load_withheld_for(model_info, resident) or _gap_withheld_for(probe_timing)
        timing = ItemTiming(
            wallClockMs=call.wallClockMs, calls=(call_timing,), withheldFor=withheld
        )
        results.append(scorer.score_item(item_input, call, timing, pack=pack))
    return tuple(results), 1.0, "by-construction"


# --- the `decisions` surface (typed-decision-models plan v1.3 §3.3, §4 U1d) -------------------


#: The hooks a scorer needs on the `decisions` surface. `answer_from_chat` is the hand-off parse,
#: so only the `handoff` surface needs it (`build_messages` stays optional there, as on chat).
_DECISIONS_SURFACE_HOOKS: tuple[str, ...] = ("decision_input", "score_answer", "aggregate")
_HANDOFF_SURFACE_HOOKS: tuple[str, ...] = (*_DECISIONS_SURFACE_HOOKS, "answer_from_chat")


def _decision_scorer_problem(scorer: ItemScorer, *, handoff: bool) -> str | None:
    """The decision-surface hooks the scorer lacks, named, or `None` — checked before the
    warm-up, so a scorer without them refuses up front rather than after a call per item."""
    hooks = _HANDOFF_SURFACE_HOOKS if handoff else _DECISIONS_SURFACE_HOOKS
    missing = [hook for hook in hooks if getattr(scorer, hook, None) is None]
    if not missing:
        return None
    return f"its scorer has no {', '.join(missing)} hook (plan v1.3 §3.3 'Scorer seam')"


def _input_tokens(usage: Mapping[str, Any] | None) -> int | None:
    """Plan2 §3.4: `usage["input_tokens"]` when it is an `int` (a `bool` is not), else `None` —
    never another usage key, and never `0` for an absent one."""
    value = usage.get("input_tokens") if isinstance(usage, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _decision_call_timing(result: DecisionResult) -> CallTiming:
    """A decisions call as a `CallTiming`: `wallClockMs` only — the response has no `stats`
    (v1.3 §2.3), so the three sibling figures are `None`, as on embeddings."""
    return CallTiming(
        wallClockMs=result.wallClockMs,
        ttftMs=None,
        generationMs=None,
        promptTokens=None,
        tokensPerSecond=None,
    )


def _the_answer(result: DecisionResult, name: str) -> DecisionAnswer:
    """The answer to the one question asked. `decide` has already refused a response that does
    not answer every asked question exactly once (`lmstudio._parse_decision_answers`)."""
    return next(a for a in result.answers if a.name == name)


def _drive_decision_items(
    pack: Pack,
    *,
    candidate: Candidate,
    scorer: ItemScorer,
    lmstudio: LMStudio,
    model_info: ModelInfo,
    baseline_residency: list[ResidentModel],
    clock: Callable[[], float],
) -> tuple[ItemResult, ...]:
    """The `decisions` surface: a decision-model candidate with no hand-off model, alone (v1.3
    §3.3 "Per-item flow", the decision-only subset). Per item: the between-item residency probe;
    the scorer's `decision_input`; the item clock started; one `decide` call with the candidate's
    question and `decisionTimeoutSeconds` (plan2 §3.2); `apply_policy`; the clock stopped. The
    item's `wallClockMs` is the item clock, its `calls` the one decision call, and it is
    `withheldFor="load"` iff the preceding snapshot did not show the model resident. No gap check
    runs: a decisions call has no `stats`, so `unexplainedMs` is always `None`.

    `score_answer` gets `decided_by="decision-model"`, the decision's `probability` (a
    predicate's p, a choice's confidence) and `input_tokens` from `usage` (plan2 §3.4).

    **A failed decision call (timeout or failure) is kept** (plan2 §3.3): the runtime acts on it,
    so the item stays in the run with outcome `"fail"`, `ItemTiming(wallClockMs=<the item clock at
    the raise>, calls=(), withheldFor=None)` — or `"load"` when the residency probe withholds it —
    and goes through `score_answer` with `failed_call="decision"` and every other decision keyword
    `None`. Its `withheldFor` is never a call-failure value: that is the "withheld, as today"
    shape plan2 §3.3 replaced, and `score_answer` refuses it.

    **An over-limit call** (plan3 §3.3: `decisions.is_over_limit` on the `LMStudioCallFailed`'s
    `status`/`body` — a 500 whose body carries the marker) is the same kept item plus
    `reason="over-limit"`; any other 500 stays a plain decision failure. There is no hand-off
    model on this surface, so nothing takes the item over (plan3 §3.3, `decisions` surface)."""
    question = dict(candidate.question)
    results: list[ItemResult] = []
    for item_input in pack.iter_items():
        resident = _between_item_residency(  # between-item probe — never during a timed call
            lmstudio, where=f"before item {item_input.get('itemId')!r}"
        )
        text = scorer.decision_input(item_input, pack=pack)
        started = clock()
        try:
            result = lmstudio.decide(
                text,
                [question],
                model=model_info.id,
                timeout_s=candidate.decisionTimeoutSeconds,
            )
        except (LMStudioCallTimeout, LMStudioCallFailed) as exc:
            over_limit = isinstance(exc, LMStudioCallFailed) and is_over_limit(
                exc.status, exc.body
            )
            kept = ItemTiming(
                wallClockMs=(clock() - started) * 1000.0,
                calls=(),
                withheldFor=_load_withheld_for(model_info, resident),
            )
            results.append(
                scorer.score_answer(
                    item_input,
                    None,
                    "fail",
                    kept,
                    pack=pack,
                    decided_by=None,
                    probability=None,
                    failed_call="decision",
                    reason="over-limit" if over_limit else None,
                    input_tokens=None,
                )
            )
            continue
        answer = _the_answer(result, question["name"])
        decision = apply_policy(candidate.policy, answer)
        elapsed_ms = (clock() - started) * 1000.0
        if decision.handoff:
            # `Candidate.load`'s one-way rule makes this unreachable from a loaded candidate.
            raise RunRefused(
                f"candidate {candidate.candidateId!r}'s policy handed off item "
                f"{item_input.get('itemId')!r}, but it has no hand-off model (plan3 §3.2)",
                exitCode=4,
            )
        timing = ItemTiming(
            wallClockMs=elapsed_ms,
            calls=(_decision_call_timing(result),),
            withheldFor=_load_withheld_for(model_info, resident),
        )
        results.append(
            scorer.score_answer(
                item_input,
                decision.answer,
                "pass",
                timing,
                pack=pack,
                decided_by="decision-model",
                probability=answer.probability if answer.type == "predicate" else answer.confidence,
                failed_call=None,
                reason=None,
                input_tokens=_input_tokens(result.usage),
            )
        )
    return tuple(results)


def _decision_call_metrics(items: Sequence[ItemResult]) -> tuple[BinaryMetric, ...]:
    """The candidate arm's exploratory decision-call counts (plan2 §3.3 "Counts", plan3 §3.6), over
    every item the surface attempted (`n` = all items: each made its decision call, a load-
    withheld one included), read from the `detail` `score_answer` wrote — never re-derived from
    the call:
    - `decisionCallFailures`: kept `failedCall == "decision"` items **without**
      `reason == "over-limit"`;
    - `overLimit`: items with `reason == "over-limit"`."""
    over_limit = sum(1 for i in items if i.detail.get("reason") == "over-limit")
    failures = sum(
        1
        for i in items
        if i.detail.get("failedCall") == "decision" and i.detail.get("reason") != "over-limit"
    )
    return (
        BinaryMetric(name="decisionCallFailures", successes=failures, n=len(items), unit="item"),
        BinaryMetric(name="overLimit", successes=over_limit, n=len(items), unit="item"),
    )


def _with_decision_call_metrics(
    aggregates: Aggregates,
    items: Sequence[ItemResult],
    *,
    extra: Sequence[BinaryMetric],
) -> Aggregates:
    """The scorer's aggregates plus `_decision_call_metrics` and `extra` (the `handoff` surface's
    `_handoff_metrics`, `()` on `decisions`), appended to `perClass`. Never in `verdictMetrics`,
    so `report.py` renders them exploratory. A decision role's aggregates need a `perClass` block
    to carry them; one without refuses rather than dropping the counts."""
    per_class = getattr(aggregates, "perClass", None)
    if per_class is None:
        raise TypeError(
            f"{type(aggregates).__name__} has no perClass block for the decisions surface's "
            "decision-call counts (plan2 §3.3)"
        )
    return replace(
        aggregates, perClass=(*per_class, *_decision_call_metrics(items), *extra)
    )


# --- the `handoff` surface (typed-decision-models plan v1.3 §3.3, §4 U2; plan2 §3.3–§3.4; plan3
# §3.3, §3.6) ----------------------------------------------------------------------------------


def _models_withheld_for(
    model_infos: Sequence[ModelInfo], resident_before: Sequence[ResidentModel]
) -> Literal["load"] | None:
    """The between-item residency probe over every model of the arm (v1.3 §3.3, "must show
    **both** resident"): withheld for load iff the preceding snapshot lacks any of them."""
    if all(_is_resident(m.id, resident_before) for m in model_infos):
        return None
    return "load"


@dataclass(frozen=True)
class _HandoffRoute:
    """Why and with what an item goes to the hand-off chat model: the decision call it completed
    (`()` on an over-limit item, whose decision call failed), that call's `probability` and
    `inputTokens` (`None` on an over-limit item, plan3 §3.3), and `reason` (`"over-limit"` or
    `None` for a policy hand-off)."""

    decision_calls: tuple[CallTiming, ...]
    probability: float | None
    input_tokens: int | None
    reason: Literal["over-limit"] | None


def _hand_off(
    pack: Pack,
    item_input: Mapping[str, Any],
    route: _HandoffRoute,
    *,
    scorer: ItemScorer,
    lmstudio: LMStudio,
    handoff_info: ModelInfo,
    handoff_failure: Literal["withhold", "fail-closed"],
    request_timeout_s: float,
    load_withheld: Literal["load"] | None,
    started: float,
    clock: Callable[[], float],
) -> ItemResult:
    """Steps 4–6 of v1.3 §3.3's per-item flow for one handed-off item: the chat call with the
    scorer's `build_messages` (else the generic `_item_chat_messages`), at the run's chat budget
    `request_timeout_s` — the pinned 180 s on a decision-point pack, **never**
    `decisionTimeoutSeconds` (plan3 §3.2), an over-limit item's included — then the scorer's
    `answer_from_chat`, then the item clock stopped.

    **A failed hand-off chat call** (timeout or failure) is decided by `handoff_failure`
    (`HANDOFF_FAILURE_BY_ROLE`, plan2 §3.3):
    - `withhold` (the guard): withheld exactly as a failed chat call is today — `wallClockMs=None`,
      `calls=()`, `withheldFor="timeout"`/`"no_response"`; a failing call takes its own
      `withheldFor` and no load producer runs. Both kinds go through `score_answer` with no
      answer, no `failed_call`, no `decided_by` or `probability`, and the outcome `score_item`
      would give (architect rulings, 2026-10-31): an over-limit one is the withheld over-limit
      record (plan3 §3.3's `withhold` column, `reason="over-limit"`, no `input_tokens`); a policy
      one carries `reason=None` and its decision call's `input_tokens` (plan2 §3.4). Its `detail`
      is the scorer's, never patched here.
    - `fail-closed` (the new abilities): kept — `ItemTiming(wallClockMs=<item clock at the raise>,
      calls=<the completed decision call, if any>, withheldFor=<residency's>)` and
      `failed_call="handoff"`, no `decided_by`; the scorer scores the fail-closed answer.
    A successful one is `decided_by="handoff"` with the decision's `probability` (`None` when
    over-limit). No gap check runs on this surface (plan3 §3.3): the decision call has no
    `stats`, and an over-limit item's lone chat call would otherwise be gap-checked alone."""
    build_messages = getattr(scorer, "build_messages", None)
    messages = (
        build_messages(item_input, pack=pack)
        if build_messages is not None
        else _item_chat_messages(pack, item_input)
    )
    prompt_config = pack.prompt_config()
    try:
        chat = lmstudio.chat(
            messages,
            model=handoff_info.id,
            temperature=prompt_config.temperature,
            max_tokens=prompt_config.maxTokens,
            timeout_s=request_timeout_s,
        )
    except (LMStudioCallTimeout, LMStudioCallFailed) as exc:
        if handoff_failure == "withhold":
            timed_out = isinstance(exc, LMStudioCallTimeout)
            withheld = ItemTiming(
                wallClockMs=None, calls=(), withheldFor="timeout" if timed_out else "no_response"
            )
            return scorer.score_answer(
                item_input,
                None,
                "fail" if timed_out else "unrunnable",
                withheld,
                pack=pack,
                decided_by=None,
                probability=None,
                failed_call=None,
                reason=route.reason,
                input_tokens=route.input_tokens,
            )
        kept = ItemTiming(
            wallClockMs=(clock() - started) * 1000.0,
            calls=route.decision_calls,
            withheldFor=load_withheld,
        )
        return scorer.score_answer(
            item_input,
            None,
            "fail",
            kept,
            pack=pack,
            decided_by=None,
            probability=None,
            failed_call="handoff",
            reason=route.reason,
            input_tokens=route.input_tokens,
        )
    answer, outcome = scorer.answer_from_chat(chat, pack=pack)
    timing = ItemTiming(
        wallClockMs=(clock() - started) * 1000.0,
        calls=(*route.decision_calls, _call_timing_from_result(chat)),
        withheldFor=load_withheld,
    )
    return scorer.score_answer(
        item_input,
        answer,
        outcome,
        timing,
        pack=pack,
        decided_by="handoff",
        probability=route.probability,
        failed_call=None,
        reason=route.reason,
        input_tokens=route.input_tokens,
    )


def _drive_handoff_items(
    pack: Pack,
    *,
    candidate: Candidate,
    scorer: ItemScorer,
    lmstudio: LMStudio,
    model_info: ModelInfo,
    handoff_info: ModelInfo,
    handoff_failure: Literal["withhold", "fail-closed"],
    request_timeout_s: float,
    clock: Callable[[], float],
) -> tuple[tuple[ItemResult, ...], bool]:
    """The `handoff` surface: a decision-model candidate with a hand-off chat model (v1.3 §3.3
    "Per-item flow"). Per item: the between-item residency probe over **both** models; the
    scorer's `decision_input`; the item clock started; one `decide` at `decisionTimeoutSeconds`;
    then one of four paths.

    - **Decided** (the policy answers): as on the `decisions` surface — `calls=(decision,)`,
      `decided_by="decision-model"`, `probability`, `inputTokens`.
    - **Policy hand-off** (the policy abstains): `_hand_off`, with the decision call completed —
      `calls=(decision, chat)` on success.
    - **Over-limit** (plan3 §3.3: `is_over_limit` on the `LMStudioCallFailed`'s `status`/`body`):
      the stakeholder's rule hands it off as a policy hand-off, with `reason="over-limit"`, no
      completed decision call, and no `probability` or `inputTokens`.
    - **Any other decision-call failure** (a timeout, a non-marker 500): kept and **not** handed
      off (plan2 §3.3), exactly as on the `decisions` surface.

    `wallClockMs` is the item clock, end to end; `withheldFor` is `"load"` iff the preceding
    snapshot lacked either model, except on a withheld hand-off failure (`_hand_off`).

    Returns the items and whether any preceding snapshot lacked a model. The flag is what
    `_within_load_basis` needs: a withheld hand-off failure takes its own `withheldFor`, so its
    record alone cannot show the reload that note §Q6's rule ("only if nothing shows a reload")
    must see."""
    question = dict(candidate.question)
    models = (model_info, handoff_info)
    results: list[ItemResult] = []
    snapshot_lacked_a_model = False
    for item_input in pack.iter_items():
        resident = _between_item_residency(  # between-item probe — never during a timed call
            lmstudio, where=f"before item {item_input.get('itemId')!r}"
        )
        load_withheld = _models_withheld_for(models, resident)
        snapshot_lacked_a_model = snapshot_lacked_a_model or load_withheld is not None
        text = scorer.decision_input(item_input, pack=pack)
        started = clock()
        try:
            result = lmstudio.decide(
                text,
                [question],
                model=model_info.id,
                timeout_s=candidate.decisionTimeoutSeconds,
            )
        except (LMStudioCallTimeout, LMStudioCallFailed) as exc:
            if not (isinstance(exc, LMStudioCallFailed) and is_over_limit(exc.status, exc.body)):
                kept = ItemTiming(
                    wallClockMs=(clock() - started) * 1000.0,
                    calls=(),
                    withheldFor=load_withheld,
                )
                results.append(
                    scorer.score_answer(
                        item_input,
                        None,
                        "fail",
                        kept,
                        pack=pack,
                        decided_by=None,
                        probability=None,
                        failed_call="decision",
                        reason=None,
                        input_tokens=None,
                    )
                )
                continue
            route = _HandoffRoute(
                decision_calls=(), probability=None, input_tokens=None, reason="over-limit"
            )
        else:
            answer = _the_answer(result, question["name"])
            decision = apply_policy(candidate.policy, answer)
            probability = (
                answer.probability if answer.type == "predicate" else answer.confidence
            )
            decision_call = _decision_call_timing(result)
            if not decision.handoff:
                timing = ItemTiming(
                    wallClockMs=(clock() - started) * 1000.0,
                    calls=(decision_call,),
                    withheldFor=load_withheld,
                )
                results.append(
                    scorer.score_answer(
                        item_input,
                        decision.answer,
                        "pass",
                        timing,
                        pack=pack,
                        decided_by="decision-model",
                        probability=probability,
                        failed_call=None,
                        reason=None,
                        input_tokens=_input_tokens(result.usage),
                    )
                )
                continue
            route = _HandoffRoute(
                decision_calls=(decision_call,),
                probability=probability,
                input_tokens=_input_tokens(result.usage),
                reason=None,
            )
        results.append(
            _hand_off(
                pack,
                item_input,
                route,
                scorer=scorer,
                lmstudio=lmstudio,
                handoff_info=handoff_info,
                handoff_failure=handoff_failure,
                request_timeout_s=request_timeout_s,
                load_withheld=load_withheld,
                started=started,
                clock=clock,
            )
        )
    return tuple(results), snapshot_lacked_a_model


_CALL_FAILURE_WITHHELD: frozenset[str] = frozenset({"timeout", "no_response"})


def _handoff_metrics(
    items: Sequence[ItemResult],
    *,
    handoff_failure: Literal["withhold", "fail-closed"],
    verdict_metrics: Sequence[str],
) -> tuple[BinaryMetric, ...]:
    """What the `handoff` surface adds to `_decision_call_metrics` (v1.3 §3.5; plan2 §3.3; plan3
    §3.3, §3.6), all exploratory and read from each item's `detail`/`timing`:
    - `handoffShare`: `decidedBy == "handoff"` items over the attempted ones — every item except
      one withheld by a failed hand-off call (plan3 §3.3's `withhold` column: "not attempted"). A
      kept hand-off failure counts as not handed off;
    - on `fail-closed` roles only, `handoffCallFailures`: `failedCall == "handoff"` items, an
      over-limit one included, over every item (plan3 §3.3: "not applicable (guard)");
    - the per-path diagnostics (v1.3 §3.5, AC-2): each verdict metric over the items the decision
      model decided (`<metric>DecidedByModel`) and over the handed-off ones
      (`<metric>HandedOff`)."""
    attempted = [
        i for i in items if i.timing is None or i.timing.withheldFor not in _CALL_FAILURE_WITHHELD
    ]
    handed_off = sum(1 for i in attempted if i.detail.get("decidedBy") == "handoff")
    metrics = [
        BinaryMetric(name="handoffShare", successes=handed_off, n=len(attempted), unit="item")
    ]
    if handoff_failure == "fail-closed":
        failures = sum(1 for i in items if i.detail.get("failedCall") == "handoff")
        metrics.append(
            BinaryMetric(name="handoffCallFailures", successes=failures, n=len(items), unit="item")
        )
    for metric in verdict_metrics:
        for decided_by, suffix in (("decision-model", "DecidedByModel"), ("handoff", "HandedOff")):
            outcomes = [
                outcome
                for i in items
                if i.detail.get("decidedBy") == decided_by
                and (outcome := i.scored_outcome(metric)) is not None
            ]
            metrics.append(
                BinaryMetric(
                    name=f"{metric}{suffix}",
                    successes=sum(outcomes),
                    n=len(outcomes),
                    unit="item",
                )
            )
    return tuple(metrics)


def _same_decision(a: DecisionAnswer, b: DecisionAnswer) -> bool:
    """Plan2 §3.4's equality for the re-check, exact: a predicate's `probability`; a choice's
    `choice` and `confidence`. Answers of different types are never equal."""
    if a.type != b.type:
        return False
    if a.type == "predicate":
        return a.probability == b.probability
    return (a.choice, a.confidence) == (b.choice, b.confidence)


def _within_load_basis(
    items: Sequence[ItemResult],
    *,
    candidate: Candidate,
    lmstudio: LMStudio,
    model_info: ModelInfo,
    warm_up: DecisionResult | None,
    snapshot_lacked_a_model: bool,
) -> Basis:
    """Plan2 §3.4's load rule (note §Q6), run after the last item and before
    `residentModelsAtEnd`: did every scored item come from the one load the warm-up made?

    It probes residency first. If the decision model is absent, any item was withheld for
    `"load"`, or `snapshot_lacked_a_model` (a between-item snapshot lacked a model of the arm —
    on the `handoff` surface a withheld hand-off failure keeps its own `withheldFor`, hiding that
    snapshot from the items; note §Q6: one load "only if nothing shows a reload"), it makes
    **no** call and the run is `"assumed"` (not one load). Otherwise it repeats
    the mandatory warm-up's call — `decide(WARMUP_DECISION_INPUT, [question])` at
    `decisionTimeoutSeconds` — and the run is `"by-construction"` iff the answer is exactly the
    mandatory warm-up's (`_same_decision`; never a `--warmup N` extra's). A raise, an unequal
    answer, a probe that could not run, or no warm-up answer to compare against is `"assumed"`:
    the fail-safe value, which only ever weakens what the run may claim. "A probe that could not
    run" is any of the three ways `residency()` fails: no response (`LMStudioUnreachable`), a
    non-200 status or a malformed catalog body (`LMStudioCallFailed`, impl review P7-n1), or a
    socket timeout."""
    try:
        resident = lmstudio.residency()
    except (LMStudioUnreachable, LMStudioCallFailed, TimeoutError):
        return "assumed"
    if not _is_resident(model_info.id, resident):
        return "assumed"
    if snapshot_lacked_a_model:
        return "assumed"
    if any(i.timing is not None and i.timing.withheldFor == "load" for i in items):
        return "assumed"
    if warm_up is None:
        return "assumed"
    name = candidate.question["name"]
    try:
        recheck = lmstudio.decide(
            WARMUP_DECISION_INPUT,
            [dict(candidate.question)],
            model=model_info.id,
            timeout_s=candidate.decisionTimeoutSeconds,
        )
    except (LMStudioCallTimeout, LMStudioCallFailed, LMStudioUnreachable):
        return "assumed"
    same = _same_decision(_the_answer(recheck, name), _the_answer(warm_up, name))
    return "by-construction" if same else "assumed"


#: The three failing `TurnDisposition` values, and the `ItemTiming.withheldFor` each maps to
#: (§3.6/§3.8.4's disposition table) — a failing disposition wins over both load producers, and
#: neither producer runs on these three (plan §3.6 `:1759-1796`, P15-3's ruling).
_FAILING_TURN_WITHHELD_FOR: dict[str, Literal["timeout", "no_response"]] = {
    "timed-out": "timeout",
    "no-response": "no_response",
    "server-rejected": "no_response",
}


def _turn_timings(
    trace: ConversationTrace,
    *,
    resident_before: list[ResidentModel],
    model_id: str,
    lmstudio: LMStudio,
) -> tuple[tuple[ItemTiming, ...], list[ResidentModel]]:
    """One `ItemTiming` per turn of `trace`, from each `TurnTrace.chatResults` (one `CallTiming` per
    completed call, in order — plan §4 S2 `:2716-2731`); applies the residency guard and the gap
    detector; returns the updated `resident` snapshot for the **next conversation's** probe.

    `drive()` runs a whole conversation atomically with no hook for a mid-conversation residency
    probe, so the between-item probe's granularity is necessarily per-*conversation* here rather
    than per-turn: every turn of one conversation is guarded against the same `resident_before`
    snapshot, taken before this conversation started, and one fresh probe is taken after it
    finishes, for the next conversation's turns (spec §4's own pseudocode: `_turn_timings` returns
    one updated `resident` per call, not one per turn).

    A turn whose disposition is not `replied`/`cap-hit` takes its `withheldFor` from that
    disposition and **no load producer runs on it** — the precedence rule §3.6/§3.8.4 state
    repeatedly, and the one this step must mutation-test.
    """
    timings: list[ItemTiming] = []
    for turn in trace.turns:
        calls = tuple(_call_timing_from_result(cr) for cr in turn.chatResults)
        failing = _FAILING_TURN_WITHHELD_FOR.get(turn.turnDisposition)
        if failing is not None:
            timings.append(ItemTiming(wallClockMs=None, calls=calls, withheldFor=failing))
            continue
        # replied / cap-hit: the turn is complete and timed; both load producers may run.
        probe_timing = ItemTiming(wallClockMs=turn.wallClockMs, calls=calls, withheldFor=None)
        withheld = (
            None if _is_resident(model_id, resident_before) else "load"
        ) or _gap_withheld_for(probe_timing)
        timings.append(ItemTiming(wallClockMs=turn.wallClockMs, calls=calls, withheldFor=withheld))
    return tuple(timings), _between_item_residency(
        lmstudio, where=f"after conversation {trace.scriptId!r}"
    )


def _drive_conversations(
    pack: Pack,
    cfg: RunConfig,
    *,
    lmstudio: LMStudio,
    model_info: ModelInfo,
    baseline_residency: list[ResidentModel],
) -> tuple[tuple[ItemResult, ...], tuple[DispatchFailureDisclosure, ...], float, Basis, Aggregates]:
    """`tool-caller` only. One fresh `ToolEnvironment` per conversation (plan §4 S2, v1.31 addition
    — "never one reused across a pack's items"), drives the scored scripts then the determinism-
    probe scripts (§3.8.4), catches `ToolDispatchFailed` per script and censors.

    Returns `(items, disclosures, basis, design_effect, aggregates)` — `design_effect` is `1.0`
    unconditionally (module docstring, gap 1: `-ml` §4.5.1/R1 — DEFF 1.00 by construction under the
    12×1 design, `stats.design_effect` is not this number).
    """
    scorer = _load_conversation_scorer(pack)
    prompt_config = pack.prompt_config()
    resident = baseline_residency
    scored: list[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]] = []
    disclosures: list[DispatchFailureDisclosure] = []

    def llm(
        messages: Sequence[ChatMessage], *, tools: Any, temperature: float, max_tokens: int
    ) -> ChatResult:
        return lmstudio.chat(
            messages,
            model=model_info.id,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=cfg.requestTimeoutSeconds,
        )

    for script in pack.iter_scripts():
        env: ToolEnvironment = pack.load_tool_module().build_environment()  # one per conversation
        try:
            trace = drive(env, script, llm, prompt_config)
        except ToolDispatchFailed as exc:
            trace = ConversationTrace(
                scriptId=script.scriptId,
                shape=script.shape,
                replicate=script.replicate,
                turns=exc.completedTurns,
            )
            disclosures.append(
                DispatchFailureDisclosure(
                    scriptId=script.scriptId,
                    turn=exc.turnIndex,
                    tool=exc.toolName,
                    reason=f"{type(exc.__cause__).__name__}: {exc.__cause__}",
                )
            )
        item_timings, resident = _turn_timings(
            trace, resident_before=resident, model_id=model_info.id, lmstudio=lmstudio
        )
        scored.append((script, trace, item_timings))

    probes: list[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]] = []
    probe_script_ids = pack.manifest["sampling"]["determinismProbeScripts"]
    for script_id in probe_script_ids:
        script = pack.find_script(script_id)
        env = pack.load_tool_module().build_environment()  # fresh env here too — same obligation
        try:
            trace = drive(env, script, llm, prompt_config)
        except ToolDispatchFailed:
            continue  # a probe re-run that itself censors is "not identical" by construction
        item_timings, resident = _turn_timings(
            trace, resident_before=resident, model_id=model_info.id, lmstudio=lmstudio
        )
        probes.append((script, trace, item_timings))

    items, aggregates = scorer.score_conversations(scored, probes, pack=pack)
    ran = len(probes) == len(probe_script_ids)
    replicates_one = pack.manifest["sampling"]["replicatesPerScript"] == 1
    determinism_probe = getattr(aggregates, "determinismProbe", None) or {}
    basis: Basis = (
        "by-construction"
        if replicates_one and ran and determinism_probe.get("identical")
        else "assumed"
    )  # plan §3.8.4 :2250-2261, test 12b's four cases :6169-6176
    return items, tuple(disclosures), basis, 1.0, aggregates


# --- LatencyBlock accumulation (spec §5) --------------------------------------------------------


def _rank(x: int, level: Fraction) -> int:
    """`-ml` §11.2.1's own integer `ceil(level * x)`, clamped to `[1, x]` — the same expression
    `stats.percentile` uses internally, reproduced here (not imported: it is private there) because
    the coverage gate below needs the rank itself, not only the value at that rank."""
    return max(1, min(x, -(-level.numerator * x // level.denominator)))


def _coverage_ok(x: int, y: int, *, level: Fraction, floor_pct: int) -> bool:
    """`-ml` §11.6's level floor, evaluated in integers exactly as the note states it:
    `100 * r >= floor_pct * y`. `x == 0` refuses unconditionally (no rank to compute)."""
    if x <= 0:
        return False
    return 100 * _rank(x, level) >= floor_pct * y


def _wall_clock_trio(
    timed_values: list[float], *, x: int, y: int
) -> tuple[float | None, float | None, float | None]:
    """`latencyMsP50`/`latencyMsP95`/`latencyMsMax`, gated (`-ml` §11.6) and, for the tail figure,
    renamed to `max` under §11.3's identity floor (`r95 == X`) — mutually exclusive with
    `latencyMsP95`, per the ruling: *"when r == X ... latencyMsP95 is None."*"""
    p50 = (
        percentile(timed_values, level=LEVEL_P50)
        if _coverage_ok(x, y, level=LEVEL_P50, floor_pct=_P50_FLOOR_PCT)
        else None
    )
    if _coverage_ok(x, y, level=LEVEL_P95, floor_pct=_TAIL_FLOOR_PCT):
        if _rank(x, LEVEL_P95) == x:
            p95, vmax = None, max(timed_values)
        else:
            p95, vmax = percentile(timed_values, level=LEVEL_P95), None
    else:
        p95, vmax = None, None
    return p50, p95, vmax


def latency_block(
    items: Sequence[ItemResult],
    *,
    call_surface: Literal["chat", "embeddings", "decisions", "handoff"] | None,
) -> LatencyBlock | None:
    """One run's `LatencyBlock`, built in one pass over `items` and returning `None` iff every
    item's `timing is None` (spec §5 rule v). `call_surface` is required (module docstring, gap 2):
    rule (iv-a) needs to tell "this surface has no `stats`" from "this surface has `stats` and none
    arrived", which `items` alone cannot distinguish.

    `"decisions"` and `"handoff"` are treated like `"embeddings"` for every stats-derived field
    (typed-decision-models plan v1.3 §3.3): a decisions response has no `stats`, and a median of
    the hand-off chat calls' stats alone over a mixed-surface arm would mislead, so
    `statsCoveredCount` and the three medians are `None`, never `0`. The wall-clock figures apply
    unchanged.
    """
    if all(i.timing is None for i in items):
        return None

    latency_item_count = len(items)
    call_count = sum(len(i.timing.calls) for i in items if i.timing is not None)
    latency_withheld_for_load = sum(
        1 for i in items if i.timing is not None and i.timing.withheldFor == "load"
    )
    latency_withheld_for_no_response = sum(
        1
        for i in items
        if i.timing is not None and i.timing.withheldFor in ("timeout", "no_response")
    )
    latency_timed_count = sum(1 for i in items if i.latencyMs is not None)
    # rule (iii): every withheld item lands in exactly one of the two counters.
    assert (
        latency_withheld_for_load + latency_withheld_for_no_response
        == latency_item_count - latency_timed_count
    )

    call_attempted_count = call_count + latency_withheld_for_no_response

    timed_values = [i.latencyMs for i in items if i.latencyMs is not None]
    p50, p95, vmax = _wall_clock_trio(
        timed_values, x=latency_timed_count, y=latency_item_count
    )

    unexplained_readings = [
        i.timing.unexplainedMs
        for i in items
        if i.timing is not None and i.timing.unexplainedMs is not None
    ]
    unexplained_ms_max = max(unexplained_readings) if unexplained_readings else None

    if call_surface in ("embeddings", "decisions", "handoff"):
        # rule (iv-a): no `stats` on this surface at all — `None`, never `0`.
        stats_covered_count: int | None = None
        ttft_median = prefill_median = tps_median = None
        unexplained_ms_max = None  # no `stats` -> no gap ever computed on this surface
    else:
        calls_flat = [c for i in items if i.timing is not None for c in i.timing.calls]
        # rule (iv)/(iv-c): a call counts only with BOTH a usable `stats` object (proxied by
        # `ttftMs is not None`) AND a usable `promptTokens` — co-presence, per call, all three
        # sibling figures share the one exclusion.
        covered = [
            c
            for c in calls_flat
            if c.ttftMs is not None
            and isinstance(c.promptTokens, int)
            and not isinstance(c.promptTokens, bool)
            and c.promptTokens > 0
        ]
        stats_covered_count = len(covered)
        if _coverage_ok(
            stats_covered_count, call_attempted_count, level=LEVEL_P50, floor_pct=_P50_FLOOR_PCT
        ):
            ttft_median = percentile([c.ttftMs for c in covered], level=LEVEL_P50)
            prefill_median = percentile(
                [c.ttftMs / (c.promptTokens / 1000) for c in covered], level=LEVEL_P50
            )
            tps_values = [c.tokensPerSecond for c in covered if c.tokensPerSecond is not None]
            tps_median = percentile(tps_values, level=LEVEL_P50) if tps_values else None
        else:
            ttft_median = prefill_median = tps_median = None

    return LatencyBlock(
        latencyMsP50=p50,
        latencyMsP95=p95,
        latencyMsMax=vmax,
        latencyTimedCount=latency_timed_count,
        latencyItemCount=latency_item_count,
        latencyWithheldForLoad=latency_withheld_for_load,
        latencyWithheldForNoResponse=latency_withheld_for_no_response,
        statsCoveredCount=stats_covered_count,
        callCount=call_count,
        ttftMsMedian=ttft_median,
        prefillMsPer1kMedian=prefill_median,
        tokensPerSecondMedian=tps_median,
        unexplainedMsMax=unexplained_ms_max,
    )


# --- run_pack: capture order (spec §4) -----------------------------------------------------------


def _utc_stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _find_model(catalog: list[ModelInfo], model_key: str) -> ModelInfo:
    for m in catalog:
        if m.id == model_key:
            return m
    raise RunRefused(
        f"model {model_key!r} is not in the LM Studio catalog (plan §3.4.4a)", exitCode=4
    )


def _check_call_surface_cross_check(call_surface: str, model_info: ModelInfo) -> None:
    """plan §3.4.4a: `callSurface == "chat"` requires catalog `type in {"llm", "vlm"}`;
    `"embeddings"` requires `type == "embeddings"`. Runs on every `model` run, before the load."""
    ok = (call_surface == "chat" and model_info.type in ("llm", "vlm")) or (
        call_surface == "embeddings" and model_info.type == "embeddings"
    )
    if not ok:
        raise RunRefused(
            f"pack declares callSurface={call_surface!r} but model {model_info.id!r} has catalog "
            f"type {model_info.type!r} (plan §3.4.4a)",
            exitCode=4,
        )


def _runtime_identity(load_result: LoadResult) -> tuple[str | None, str | None]:
    """`LoadResult.runtime`'s `name`/`version` — chat surface only; `(None, None)` on embeddings or
    a malformed/absent `runtime` object (plan §3.4.4a capture-order step 5)."""
    runtime = load_result.runtime
    if not isinstance(runtime, Mapping):
        return None, None
    name = runtime.get("name")
    version = runtime.get("version")
    return (
        name if isinstance(name, str) else None,
        version if isinstance(version, str) else None,
    )


def _resident_dicts(resident: Sequence[ResidentModel]) -> list[dict[str, str]]:
    return [{"id": r.id, "state": r.state} for r in resident]


def _build_fingerprint(
    *,
    pack: Pack,
    model_info: ModelInfo,
    call_surface: Literal["chat", "embeddings", "decisions", "handoff"],
    host: Mapping[str, Any],
    started_at: datetime,
    ended_at: datetime,
    resident_at_start: list[ResidentModel],
    resident_at_end: list[ResidentModel],
    residency_source: str,
    runtime_name: str | None,
    runtime_version: str | None,
    catalog_after_load: list[ModelInfo],
    prompt_config: PromptConfig,
    decision_fields: Mapping[str, Any] | None,
    handoff_fields: Mapping[str, Any] | None,
) -> Fingerprint:
    """The run's fingerprint. Model identity is read from the LM Studio catalog entry of the
    model the arm ran (`model_info` — the decision model on a candidate arm), never from a request
    string; `loadedContextLength` from the catalog read after the load. `decision_fields` is
    `_decision_fingerprint_fields`' output on the `decisions` and `handoff` surfaces and `None`
    on every other one; `handoff_fields` is `_handoff_fingerprint_fields`' on `handoff` only —
    each refused otherwise, so a profile's additions can neither go missing nor leak onto
    another profile."""
    if (decision_fields is not None) != (call_surface in ("decisions", "handoff")):
        raise ValueError(
            f"_build_fingerprint: decision_fields are given iff call_surface is 'decisions' or "
            f"'handoff' (call_surface={call_surface!r})"
        )
    if (handoff_fields is not None) != (call_surface == "handoff"):
        raise ValueError(
            f"_build_fingerprint: handoff_fields are given iff call_surface == 'handoff' "
            f"(call_surface={call_surface!r})"
        )
    loaded = next((m for m in catalog_after_load if m.id == model_info.id), None)
    attested = host.get("attested") or {}
    fields: dict[str, Any] = {
        "modelKey": model_info.id,
        "modelPublisher": model_info.publisher,
        "arch": model_info.arch,
        "quantization": model_info.quantization,
        "compatibilityType": model_info.compatibility_type,
        "maxContextLength": model_info.max_context_length,
        "loadedContextLength": loaded.loaded_context_length if loaded is not None else None,
        "modelType": model_info.type,
        "modelCapabilities": (
            list(model_info.capabilities) if model_info.capabilities is not None else []
        ),
        "modelCapabilitiesPresent": model_info.capabilities is not None,
        "residencySource": residency_source,
        "residentModelsAtStart": _resident_dicts(resident_at_start),
        "residentModelsAtEnd": _resident_dicts(resident_at_end),
        "packId": pack.packId,
        "packVersion": pack.packVersion,
        "packContentHash": pack.contentHash,
        "benchVersion": modelbench.__version__,
        "benchSchemaVersion": BENCH_SCHEMA_VERSION,
        "pythonVersion": platform.python_version(),
        "hostOs": platform.platform(),
        "startedAt": _utc_stamp(started_at),
        "endedAt": _utc_stamp(ended_at),
        "lmStudioAppVersion": attested.get("lmStudioAppVersion"),
        "kvCacheSetting": attested.get("kvCacheSetting"),
        "hostRamGb": attested.get("hostRamGb"),
        "otherResidentWorkloads": attested.get("otherResidentWorkloads", []),
    }
    if call_surface == "chat":
        fields["runtimeName"] = runtime_name
        fields["runtimeVersion"] = runtime_version
        fields["temperature"] = prompt_config.temperature
        fields["maxTokens"] = prompt_config.maxTokens
    if decision_fields is not None:
        fields.update(decision_fields)
    if handoff_fields is not None:
        fields.update(handoff_fields)
    return Fingerprint(armKind="model", callSurface=call_surface, fields=fields)


def _handoff_fingerprint_fields(
    handoff_info: ModelInfo,
    *,
    catalog_after_load: Sequence[ModelInfo],
    chat_warm_up: LoadResult,
    prompt_config: PromptConfig,
) -> dict[str, Any]:
    """What `model:handoff` adds to `model:decisions` (v1.3 §3.3, §4 U2): the hand-off model's
    identity from its own catalog entry, `handoffLoadedContextLength` from the catalog read after
    the load, its runtime from **its own** chat warm-up (the only chat call outside a scored
    item), and the pack's sampling settings, which only the hand-off chat call obeys — flat and
    `handoff`-prefixed, never under the bare chat names."""
    loaded = next((m for m in catalog_after_load if m.id == handoff_info.id), None)
    runtime_name, runtime_version = _runtime_identity(chat_warm_up)
    return {
        "handoffModelKey": handoff_info.id,
        "handoffModelPublisher": handoff_info.publisher,
        "handoffArch": handoff_info.arch,
        "handoffQuantization": handoff_info.quantization,
        "handoffCompatibilityType": handoff_info.compatibility_type,
        "handoffMaxContextLength": handoff_info.max_context_length,
        "handoffLoadedContextLength": loaded.loaded_context_length if loaded is not None else None,
        "handoffModelType": handoff_info.type,
        "handoffRuntimeName": runtime_name,
        "handoffRuntimeVersion": runtime_version,
        "handoffTemperature": prompt_config.temperature,
        "handoffMaxTokens": prompt_config.maxTokens,
    }


def _decision_fingerprint_fields(
    candidate: Candidate, *, eval_set: str, fit_manifest: FitManifest, instance_config: Any
) -> dict[str, Any]:
    """What `model:decisions` adds to the embeddings set (v1.3 §3.3, §3.5; plan2 §3.2; plan3
    §3.7), all read from the loaded candidate, the pack and the decision model's instance — the
    report reads them from the run record, never from the candidate file. `fitManifestHash`/
    `fitManifestSize` are measured from the manifest file `_prepare_decision_arm` verified
    against the candidate's declaration (AC-3). `instance_config` is
    `_decision_instance_config`'s, verbatim."""
    return {
        "candidateId": candidate.candidateId,
        "candidateContentHash": candidate.contentHash,
        "questionSpecHash": question_spec_hash(candidate.question),
        "decisionPolicy": dict(candidate.policy),
        "fitManifestHash": fit_manifest.sha256,
        "fitManifestSize": fit_manifest.size,
        "evalSetHash": eval_set,
        "fitOutcome": candidate.fit["outcome"],
        "fitQualityTarget": candidate.fit["qualityTarget"],
        "decisionInstanceConfig": instance_config,
    }


# --- explicit loading (typed-decision-models plan3 §3.7) ------------------------------------------


def _arm_model_ids(model_info: ModelInfo, decision_arm: _DecisionArm | None) -> tuple[str, ...]:
    """The models a run's arm uses (plan3 §3.7: "the chat model, the decision model, or both"):
    the decision model, then the hand-off model if any, on a candidate run; the chat/embeddings
    model otherwise."""
    if decision_arm is None:
        return (model_info.id,)
    if decision_arm.handoff_info is None:
        return (decision_arm.model_info.id,)
    return (decision_arm.model_info.id, decision_arm.handoff_info.id)


def _load_absent_models(
    lmstudio: LMStudio,
    model_ids: Sequence[str],
    *,
    resident_at_start: Sequence[ResidentModel],
    timeout_s: float,
) -> dict[str, LoadedInstance]:
    """Plan3 §3.7's load step, on a decision-point pack, before the warm-ups: every model of the
    arm absent from `residentModelsAtStart` gets one `load_model` call, at `timeout_s`
    (`firstCallTimeoutSeconds`); a model resident at start is used as found and gets none. LM
    Studio exposes no load identity, so the runner's own load is the evidence that nothing it
    scores was JIT-loaded. A load that times out, fails or finds no server refuses the run with
    exit 3 (an environment state) before any warm-up. Returns what was loaded, by model id."""
    loaded: dict[str, LoadedInstance] = {}
    for model_id in model_ids:
        if _is_resident(model_id, resident_at_start):
            continue
        try:
            loaded[model_id] = lmstudio.load_model(model_id, timeout_s=timeout_s)
        except (LMStudioCallTimeout, LMStudioCallFailed, LMStudioUnreachable) as exc:
            raise RunRefused(
                f"could not load {model_id!r} before the warm-up ({exc}); nothing was scored "
                "(typed-decision-models plan3 §3.7)",
                exitCode=3,
            ) from exc
    return loaded


def _refuse_unless_resident(model_ids: Sequence[str], resident: Sequence[ResidentModel]) -> None:
    """Plan3 §3.7's co-residency refusal: after the warm-ups, every model of a candidate arm must
    be resident, or the run refuses with exit 3 before any item. On the `decisions` surface the
    arm has one model, so co-residency reduces to the decision model being resident after its own
    warm-up — the one-model case of the same rule, not a second one."""
    absent = [model_id for model_id in model_ids if not _is_resident(model_id, resident)]
    if absent:
        raise RunRefused(
            f"{', '.join(repr(m) for m in absent)} not resident after the warm-up; nothing was "
            "scored (typed-decision-models plan3 §3.7, co-residency)",
            exitCode=3,
        )


def _decision_instance_config(lmstudio: LMStudio, model_id: str) -> Mapping[str, Any]:
    """Plan3 §3.7's `decisionInstanceConfig`: the decision model's `loaded_instances[0].config`
    from `GET /api/v1/models`, read **after** the load step and the co-residency check, verbatim.

    Every path that does not yield a `config` mapping refuses the run with exit 3 before any item:
    a failed read, no instance listed under `model_id`, or an instance with no `config` mapping.
    Co-residency has just shown the model resident and LM Studio's `config` is required, so each
    of those is an anomaly — an eviction between steps 7 and 8, or an `/api/v0` `id` that is not
    the `/api/v1` `key` (a pre-resident model under a custom identifier) — never a captured
    answer. This deviates from plan3's "`None` if absent" (impl review Pass 8, P8-M1): a `None`
    here could only ever mean "not captured", which FR-7 refuses."""
    source = f"{model_id!r}'s instance config from GET /api/v1/models"
    try:
        instances = lmstudio.loaded_instances(model_id)
    except (LMStudioUnreachable, LMStudioCallFailed, TimeoutError) as exc:
        raise RunRefused(
            f"could not read {source} ({exc}); nothing was scored (typed-decision-models plan3 "
            "§3.7)",
            exitCode=3,
        ) from exc
    if not instances:
        raise RunRefused(
            f"could not read {source}: no loaded instance listed under that key, although the "
            "model was resident (evicted, or its /api/v0 id is not its /api/v1 key); nothing "
            "was scored (typed-decision-models plan3 §3.7)",
            exitCode=3,
        )
    config = instances[0].get("config")
    if not isinstance(config, Mapping):
        raise RunRefused(
            f"could not read {source}: its first instance has no 'config' object; nothing was "
            "scored (typed-decision-models plan3 §3.7)",
            exitCode=3,
        )
    return config


@dataclass(frozen=True)
class _DecisionArm:
    """A `run --candidate` arm that passed every pre-flight refusal: the loaded candidate, its
    decision model's catalog entry, the scorer, the pack's `evalSetHash`, and the verified fit
    manifest. On the `handoff` surface also the hand-off model's catalog entry and what a failed
    hand-off chat call does on this role (`HANDOFF_FAILURE_BY_ROLE`); both are `None` iff the
    candidate has no hand-off model (the `decisions` surface)."""

    candidate: Candidate
    model_info: ModelInfo
    scorer: ItemScorer
    eval_set: str
    fit_manifest: FitManifest
    handoff_info: ModelInfo | None
    handoff_failure: Literal["withhold", "fail-closed"] | None

    @property
    def call_surface(self) -> Literal["decisions", "handoff"]:
        return "decisions" if self.handoff_info is None else "handoff"


def _prepare_decision_arm(
    pack: Pack, cfg: RunConfig, *, root: Path, catalog: list[ModelInfo]
) -> _DecisionArm:
    """v1.3 §3.3's refusals for `run --pack P --candidate ID` (exit 4 each), before any call:
    an unloadable candidate (`Candidate.load`'s own refusals) or one for another role; a pack
    with no `decisionPoint` block or no `lmstudio-decisions` token; a question outside the
    pack's answer space (`packs.answer_space_problem`, a `score` question included); a pack whose
    items cannot give an `evalSetHash` (an item with no `stratum`); a fit manifest that cannot be
    read or does not match the candidate's declared `sha256`/`size` (`decisions.load_fit_manifest`);
    **AC-3's leakage refusal** — any pack item, `eval` or `side-check`, whose `input_hash` is in
    the fit manifest (v1.3 §3.5); a scorer without the decisions hooks; the decision model absent
    from the catalog, or its catalog `type` not `"decision"`.

    **A candidate with a hand-off model runs on the `handoff` surface** (v1.3 §3.3's third row),
    with three more refusals: a pack whose `environment.requires` does not declare
    `lmstudio-chat`; a hand-off model absent from the catalog or of catalog `type` not in `{llm,
    vlm}`; and, on a speed-up pack (`BAR_KIND_BY_ROLE`), a hand-off model that is not
    `decisionPoint.incumbent` (FR-3). Its scorer also needs `answer_from_chat`, the hand-off parse.

    The manifest and leakage checks run for **every** candidate, with or without a hand-off model
    (impl review Pass 6, P6-m2): both surfaces share this function, and a decision-only candidate
    is exactly the one a hand-off-only check would miss."""
    assert cfg.candidateId is not None
    try:
        candidate = Candidate.load(root, cfg.candidateId)
    except CandidateError as exc:
        raise RunRefused(str(exc), exitCode=4) from exc

    def refuse(reason: str) -> RunRefused:
        return RunRefused(
            f"{pack.packId}: cannot run candidate {candidate.candidateId!r}: {reason}", exitCode=4
        )

    if candidate.role != pack.role:
        raise refuse(f"it is a {candidate.role!r} candidate and the pack's role is {pack.role!r}")
    if "decisionPoint" not in pack.manifest:
        raise refuse("the pack has no decisionPoint block (plan v1.3 §3.3)")
    environment = pack.manifest.get("environment") or {}
    if not derive_decision_support(environment.get("requires")):
        raise refuse("environment.requires does not declare 'lmstudio-decisions' (plan v1.3 §3.3)")
    handoff_model = candidate.handoffModel
    if handoff_model is not None:
        if derive_call_surface(environment.get("requires")) != "chat":
            raise refuse(
                f"it has a hand-off model ({handoff_model!r}) and environment.requires does not "
                "declare 'lmstudio-chat' (plan v1.3 §3.3)"
            )
        incumbent = pack.manifest["decisionPoint"].get("incumbent")
        if BAR_KIND_BY_ROLE.get(pack.role) == "speed-up" and handoff_model != incumbent:
            raise refuse(
                f"its hand-off model {handoff_model!r} is not the speed-up point's incumbent "
                f"{incumbent!r}; any other hand-off model goes to the existing judge (FR-3, plan "
                "v1.3 §3.3)"
            )
    problem = answer_space_problem(pack.manifest["decisionPoint"], candidate.question)
    if problem is not None:
        raise refuse(problem)
    try:
        eval_set = eval_set_hash(pack.iter_items(), pack.role)
    except ValueError as exc:
        raise refuse(f"no evalSetHash: {exc}") from exc
    try:
        fit_manifest = load_fit_manifest(root, candidate.fitManifest)
    except CandidateError as exc:
        raise refuse(str(exc)) from exc
    leaked = leaked_items(pack.iter_items(), pack.role, fit_manifest)
    if leaked:
        raise refuse(
            f"its fit manifest holds the input of {len(leaked)} of this pack's items "
            f"({', '.join(leaked[:5])}{', …' if len(leaked) > 5 else ''}): it was fitted on an "
            "input it would be measured on (AC-3, plan v1.3 §3.5)"
        )
    scorer = _load_item_scorer(pack)
    problem = _decision_scorer_problem(scorer, handoff=handoff_model is not None)
    if problem is not None:
        raise refuse(problem)
    model_info = _find_model(catalog, candidate.decisionModel)
    if model_info.type != "decision":
        raise refuse(
            f"its decision model {model_info.id!r} has catalog type {model_info.type!r}, not "
            "'decision' (plan v1.3 §3.3)"
        )
    handoff_info: ModelInfo | None = None
    handoff_failure: Literal["withhold", "fail-closed"] | None = None
    if handoff_model is not None:
        handoff_info = _find_model(catalog, handoff_model)
        if handoff_info.type not in ("llm", "vlm"):
            raise refuse(
                f"its hand-off model {handoff_info.id!r} has catalog type {handoff_info.type!r}, "
                "not 'llm' or 'vlm' (plan v1.3 §3.3)"
            )
        # Keyed exactly like `BAR_KIND_BY_ROLE`, which `Candidate.load` checked the role against.
        handoff_failure = HANDOFF_FAILURE_BY_ROLE[pack.role]  # type: ignore[assignment]
    return _DecisionArm(
        candidate=candidate,
        model_info=model_info,
        scorer=scorer,
        eval_set=eval_set,
        fit_manifest=fit_manifest,
        handoff_info=handoff_info,
        handoff_failure=handoff_failure,
    )


def _resolve_request_timeout(pack: Pack, cfg: RunConfig) -> float:
    """The run's chat-call budget (typed-decision-models plan2 §3.2, note §Q4.3). On a decision-
    point pack (a `decisionPoint` block) it is `DECISION_POINT_CHAT_TIMEOUT_SECONDS`, and an
    explicit `--request-timeout` of any other value refuses up front (exit 4): a run at another
    value can never carry a verdict. Elsewhere it is the explicit value, or
    `DEFAULT_REQUEST_TIMEOUT_SECONDS`."""
    if "decisionPoint" not in pack.manifest:
        return (
            cfg.requestTimeoutSeconds
            if cfg.requestTimeoutSeconds is not None
            else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
    pinned = float(DECISION_POINT_CHAT_TIMEOUT_SECONDS)
    if cfg.requestTimeoutSeconds is not None and cfg.requestTimeoutSeconds != pinned:
        raise RunRefused(
            f"{pack.packId}: --request-timeout {cfg.requestTimeoutSeconds:g} on a decision-point "
            f"pack; its chat calls run at {pinned:g} s and no other value can carry a verdict "
            "(typed-decision-models plan2 §3.2) — drop the flag",
            exitCode=4,
        )
    return pinned


def _warm_up(
    lmstudio: LMStudio,
    model_id: str,
    *,
    call_surface: Literal["chat", "embeddings", "decisions"],
    system_prompt: str | None,
    was_resident_before: bool,
    cfg: RunConfig,
    **question: Any,
) -> LoadResult:
    """One model's mandatory warm-up at `firstCallTimeoutSeconds` (capture-order step 4), then
    its `--warmup <n>` extras, additive and discarded. A failed mandatory warm-up refuses the run
    with exit 3: nothing was scored. Returns the mandatory warm-up's `LoadResult` — never an
    extra's (plan2 §3.4's re-check compares with it)."""
    try:
        load_result = lmstudio.warm_up(
            model_id,
            call_surface=call_surface,
            system_prompt=system_prompt,
            was_resident_before=was_resident_before,
            timeout_s=cfg.firstCallTimeoutSeconds,
            **question,
        )
    except LMStudioCallTimeout as exc:
        raise RunRefused(
            f"warm-up of {model_id!r} did not complete within --first-call-timeout "
            f"({cfg.firstCallTimeoutSeconds}s); nothing was scored (plan §3.6)",
            exitCode=3,
        ) from exc
    except LMStudioCallFailed as exc:
        raise RunRefused(f"warm-up call failed: {exc}", exitCode=3) from exc
    for _ in range(cfg.warmupExtra):
        lmstudio.warm_up(
            model_id,
            call_surface=call_surface,
            system_prompt=system_prompt,
            was_resident_before=True,
            timeout_s=cfg.firstCallTimeoutSeconds,
            **question,
        )
    return load_result


def run_pack(
    pack: Pack,
    cfg: RunConfig,
    *,
    lmstudio: LMStudio,
    root: Path,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[RunResult, tuple[DispatchFailureDisclosure, ...]]:
    """The CLI `run` command's one entry point (plan §3.6a). Raises `RunRefused` for every refusal
    in capture order (exit 3/4/5); returns the assembled `RunResult` plus zero or more dispatch-
    failure disclosures otherwise. Sequencing is §3.4.4a's ten capture-order steps, verbatim.

    **A candidate run** (`cfg.candidateId`, typed-decision-models plan v1.3 §3.3) takes the
    same steps on the `decisions` surface: step 3a runs `_prepare_decision_arm`'s refusals instead
    of the chat/embeddings cross-check; step 4's warm-up is one `decide` with the candidate's own
    question (attestation trip-wire `"unavailable"`, as for embeddings); step 9 is
    `_drive_decision_items`, timed by `clock` (seconds, monotonic — the item clock). Its
    `designEffect` is `1.0`; its `basis` is the load rule's (`_within_load_basis`, plan2 §3.4):
    `"by-construction"` for one load, otherwise `"assumed"`.

    **A candidate with a hand-off model** runs on the `handoff` surface (v1.3 §3.3, §4 U2): step
    4 warms up the decision model, then the hand-off model on the chat route; step 6's trip-wire
    is fed by that chat warm-up (`"compared"`); step 9 is `_drive_handoff_items`; the fingerprint
    is `model:handoff` (`_handoff_fingerprint_fields`), and the load rule is the decision
    model's, as on `decisions`.

    **Every run records its chat-call budget** (`RunResult.requestTimeoutSeconds`), resolved up
    front by `_resolve_request_timeout` — pinned, with its exit-4 refusal, on a decision-point
    pack (plan2 §3.2).

    **On a decision-point pack the runner loads** (typed-decision-models plan3 §3.7): after step
    3a, before the warm-up, every model of the arm absent from `residentModelsAtStart` gets one
    `load_model` (`_load_absent_models`, exit 3 on failure). A candidate arm's models must then be
    resident at step 7 (`_refuse_unless_resident`, exit 3), and after step 8 the decision model's
    instance config is read for `decisionInstanceConfig` (`_decision_instance_config`)."""
    started_at = now()  # step 0
    # Up front, before any call: the chat-call budget, pinned on a decision-point pack.
    cfg = replace(cfg, requestTimeoutSeconds=_resolve_request_timeout(pack, cfg))

    try:
        host = hostinfo.read_host_info(root)  # step 1
    except hostinfo.HostInfoError as exc:
        raise RunRefused(str(exc), exitCode=5) from exc

    probe_result = lmstudio.probe()  # step 2
    if probe_result != "api-v0":
        raise RunRefused(_PROBE_MESSAGES[probe_result], exitCode=3)

    resident_at_start = lmstudio.residency()  # step 3
    residency_source = _RESIDENCY_SOURCE

    catalog = lmstudio.catalog()  # step 3a
    decision_arm: _DecisionArm | None = None
    call_surface: Literal["chat", "embeddings", "decisions", "handoff"] | None
    if cfg.candidateId is not None:
        decision_arm = _prepare_decision_arm(pack, cfg, root=root, catalog=catalog)
        model_info = decision_arm.model_info
        call_surface = decision_arm.call_surface
    else:
        assert cfg.modelKey is not None  # RunConfig: exactly one of the two is set
        model_info = _find_model(catalog, cfg.modelKey)
        call_surface = derive_call_surface(  # type: ignore[assignment]
            pack.manifest.get("environment", {}).get("requires")
        )
        if call_surface is None:
            raise RunRefused(
                f"{pack.packId}: environment.requires does not declare exactly one call surface "
                "(plan §3.4.4a)",
                exitCode=4,
            )
        _check_call_surface_cross_check(call_surface, model_info)
        if pack.role == "tool-caller":
            try:
                check_tool_calling_eligibility(pack.role, model_info)
            except Exception as exc:  # ToolCallingIneligible
                raise RunRefused(str(exc), exitCode=4) from exc

    # Plan3 §3.7's load step, on a decision-point pack only: the arm's absent models are loaded
    # here, so the warm-ups find them loaded and nothing the run scores is JIT-loaded.
    arm_models = _arm_model_ids(model_info, decision_arm)
    loaded_now = (
        _load_absent_models(
            lmstudio,
            arm_models,
            resident_at_start=resident_at_start,
            timeout_s=cfg.firstCallTimeoutSeconds,
        )
        if "decisionPoint" in pack.manifest
        else {}
    )

    prompt_config = pack.prompt_config()
    # The decisions warm-up is one `decide` with the candidate's own question (v1.3 §3.3); the
    # keyword is passed on that surface only, where `warm_up` requires it. A candidate arm's
    # first warm-up is its decision model's, on LM Studio's `decisions` route, whichever
    # surface the arm is.
    warm_up_surface: Literal["chat", "embeddings", "decisions"] = (
        "decisions" if decision_arm is not None else call_surface  # type: ignore[assignment]
    )
    warm_up_question: dict[str, Any] = (
        {"question": dict(decision_arm.candidate.question)} if decision_arm is not None else {}
    )
    # Step 4 — the mandatory, once-per-model warm-ups. On the `handoff` surface both models are
    # warmed up here, the decision model first, and only then does step 7 check that both are
    # still resident (plan3 §3.7: "After both warm-ups") — so a chat warm-up that evicts the
    # decision model refuses the run before any item.
    load_result = _warm_up(
        lmstudio,
        model_info.id,
        call_surface=warm_up_surface,
        system_prompt=prompt_config.systemPrompt if warm_up_surface == "chat" else None,
        was_resident_before=(
            _is_resident(model_info.id, resident_at_start) or model_info.id in loaded_now
        ),
        cfg=cfg,
        **warm_up_question,
    )
    handoff_info = decision_arm.handoff_info if decision_arm is not None else None
    chat_load_result: LoadResult | None = None
    if handoff_info is not None:
        chat_load_result = _warm_up(
            lmstudio,
            handoff_info.id,
            call_surface="chat",
            system_prompt=prompt_config.systemPrompt,
            was_resident_before=(
                _is_resident(handoff_info.id, resident_at_start) or handoff_info.id in loaded_now
            ),
            cfg=cfg,
        )

    # The attestation trip-wire is fed by a chat warm-up: the arm's own on a chat arm, the
    # hand-off model's on the `handoff` surface (`"compared"`, v1.3 §3.3); a surface with no chat
    # warm-up (embeddings, decisions) has no runtime to compare (`"unavailable"`).
    chat_warm_up = chat_load_result if handoff_info is not None else load_result
    attestation_surface: Literal["chat", "embeddings", "decisions"] = (
        "chat" if handoff_info is not None else warm_up_surface
    )
    runtime_name, runtime_version = (
        _runtime_identity(chat_warm_up)  # type: ignore[arg-type]
        if attestation_surface == "chat"
        else (None, None)
    )

    check = hostinfo.check_attestation_staleness(  # step 6
        host,
        call_surface=attestation_surface,
        residency_source=residency_source,
        runtime_name=runtime_name,
        runtime_version=runtime_version,
    )
    if check.stale:
        raise RunRefused(check.message or "attestation is stale", exitCode=5)
    if check.updated_host is not None:
        hostinfo.write_host_info(root, check.updated_host)

    baseline_residency = lmstudio.residency()  # step 7 — item-1's baseline, never resident_at_start
    if decision_arm is not None:
        _refuse_unless_resident(arm_models, baseline_residency)  # plan3 §3.7, before any item
    catalog_after_load = lmstudio.catalog()  # step 8 — loadedContextLength
    instance_config = (
        _decision_instance_config(lmstudio, decision_arm.model_info.id)
        if decision_arm is not None
        else None
    )

    if decision_arm is not None:
        if handoff_info is not None:
            items, snapshot_lacked_a_model = _drive_handoff_items(  # step 9
                pack,
                candidate=decision_arm.candidate,
                scorer=decision_arm.scorer,
                lmstudio=lmstudio,
                model_info=model_info,
                handoff_info=handoff_info,
                handoff_failure=decision_arm.handoff_failure,  # type: ignore[arg-type]
                request_timeout_s=cfg.requestTimeoutSeconds,  # type: ignore[arg-type]
                clock=clock,
            )
            handoff_metrics = _handoff_metrics(
                items,
                handoff_failure=decision_arm.handoff_failure,  # type: ignore[arg-type]
                verdict_metrics=pack.manifest["metrics"]["verdictMetrics"],
            )
        else:
            items = _drive_decision_items(  # step 9
                pack,
                candidate=decision_arm.candidate,
                scorer=decision_arm.scorer,
                lmstudio=lmstudio,
                model_info=model_info,
                baseline_residency=baseline_residency,
                clock=clock,
            )
            handoff_metrics = ()
            snapshot_lacked_a_model = False  # every item records its own load withholding
        # Sampling unit == item (v1.3 §3.3); `by-construction` is final iff the run is one load
        # of the decision model (plan2 §3.4) — the load rule, before step 10's
        # `residentModelsAtEnd`, on both decision surfaces.
        design_effect = 1.0
        basis = _within_load_basis(
            items,
            candidate=decision_arm.candidate,
            lmstudio=lmstudio,
            model_info=model_info,
            warm_up=load_result.decision,
            snapshot_lacked_a_model=snapshot_lacked_a_model,
        )
        disclosures = ()
        aggregates = _with_decision_call_metrics(
            decision_arm.scorer.aggregate(items, pack=pack), items, extra=handoff_metrics
        )
    elif pack.role == "tool-caller":
        items, disclosures, basis, design_effect, aggregates = _drive_conversations(  # step 9
            pack,
            cfg,
            lmstudio=lmstudio,
            model_info=model_info,
            baseline_residency=baseline_residency,
        )
    else:
        items, design_effect, basis = _drive_single_call_items(
            pack,
            cfg,
            lmstudio=lmstudio,
            model_info=model_info,
            call_surface=call_surface,
            baseline_residency=baseline_residency,
        )
        disclosures = ()
        scorer = _load_item_scorer(pack)
        aggregates = scorer.aggregate(items, pack=pack)

    resident_at_end: list[ResidentModel] | None = None
    run_id = f"{pack.packId}-{cfg.armKey}-{_utc_stamp(started_at)}".replace("/", "_")
    for attempt in range(4):  # step 10, 3 retries
        try:
            resident_at_end = lmstudio.residency()
            break
        except (LMStudioUnreachable, LMStudioCallFailed, TimeoutError):  # P7-n1's class too
            if attempt == 3:
                raise RunRefused(
                    "residentModelsAtEnd could not be captured after three retries; transcript "
                    f"retained at results/transcripts/{run_id}.jsonl",
                    exitCode=5,
                )
            sleep(1.0)
    ended_at = now()

    fingerprint = _build_fingerprint(
        pack=pack,
        model_info=model_info,
        call_surface=call_surface,
        host=host,
        started_at=started_at,
        ended_at=ended_at,
        resident_at_start=resident_at_start,
        resident_at_end=resident_at_end or [],
        residency_source=residency_source,
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        catalog_after_load=catalog_after_load,
        prompt_config=prompt_config,
        decision_fields=(
            _decision_fingerprint_fields(
                decision_arm.candidate,
                eval_set=decision_arm.eval_set,
                fit_manifest=decision_arm.fit_manifest,
                instance_config=instance_config,
            )
            if decision_arm is not None
            else None
        ),
        handoff_fields=(
            _handoff_fingerprint_fields(
                handoff_info,
                catalog_after_load=catalog_after_load,
                chat_warm_up=chat_load_result,  # type: ignore[arg-type]
                prompt_config=prompt_config,
            )
            if handoff_info is not None
            else None
        ),
    )
    latency = latency_block(items, call_surface=call_surface)
    run = RunResult(
        runId=run_id,
        sessionId=cfg.sessionId,
        role=pack.role,
        armKind="model",
        fingerprint=fingerprint,
        items=items,
        aggregates=aggregates,
        designEffect=design_effect,
        basis=basis,
        attestationTripWire=check.outcome,
        latency=latency,
        requestTimeoutSeconds=cfg.requestTimeoutSeconds,
    )
    return run, disclosures
