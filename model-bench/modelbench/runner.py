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
   Inert today: `_load_item_scorer`/`_load_conversation_scorer` raise `NotImplementedError`
   unconditionally (no scorer ships before S3), so this is a seam for S3 to confirm or revise
   (spec §3.2's own scorer-seam caveat), not a load-bearing decision anything already depends on.

**Everything here is offline-testable and nothing here calls a real pack's scorer** — S2's own
"Done when" never runs one (spec §3.2).
"""

from __future__ import annotations

import importlib
import json
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from modelbench.fingerprint import Fingerprint
from modelbench.lmstudio import (
    ChatResult,
    EmbedResult,
    LMStudio,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LMStudioUnreachable,
    LoadResult,
    ModelInfo,
    ResidentModel,
    check_tool_calling_eligibility,
)
from modelbench.packs import Pack, derive_call_surface
from modelbench.results import (
    BENCH_SCHEMA_VERSION,
    Aggregates,
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
    `LMStudio.chat`/`embed`/`warm_up` already)."""

    modelKey: str
    sessionId: str | None
    referenceKey: str | None
    warmupExtra: int = 0
    firstCallTimeoutSeconds: float = 300.0
    requestTimeoutSeconds: float = 120.0


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
    raise NotImplementedError(
        f"no scorer module ships yet for role {pack.role!r} (pack {pack.packId!r}); "
        "S2's runner ships the scorer seam only (spec §3.2) — the first concrete "
        "ConversationScorer is S5's"
    )


# --- shared timing helpers (spec §3.3, §4) --------------------------------------------------


def _is_resident(model_id: str, resident: Sequence[ResidentModel]) -> bool:
    return model_id in {m.id for m in resident}


def _load_withheld_for(
    model_info: ModelInfo, resident_before: list[ResidentModel]
) -> Literal["load"] | None:
    """The between-item residency probe (plan §3.6): withheld for load iff the PRECEDING snapshot
    did not show the model resident. Never evaluated during a timed call."""
    return None if _is_resident(model_info.id, resident_before) else "load"


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
        prime(
            lmstudio=lmstudio,
            model_info=model_info,
            call_surface=call_surface,
            timeout_s=cfg.requestTimeoutSeconds,
            pack=pack,
        )
    prompt_config = pack.prompt_config()
    resident = baseline_residency
    results: list[ItemResult] = []
    for item_input in pack.iter_items():
        resident = lmstudio.residency()  # between-item probe — never during a timed call
        try:
            if call_surface == "chat":
                call: ChatResult | EmbedResult = lmstudio.chat(
                    _item_chat_messages(pack, item_input),
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
    return tuple(timings), lmstudio.residency()


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
    items: Sequence[ItemResult], *, call_surface: Literal["chat", "embeddings"] | None
) -> LatencyBlock | None:
    """One run's `LatencyBlock`, built in one pass over `items` and returning `None` iff every
    item's `timing is None` (spec §5 rule v). `call_surface` is required (module docstring, gap 2):
    rule (iv-a) needs to tell "this surface has no `stats`" from "this surface has `stats` and none
    arrived", which `items` alone cannot distinguish.
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

    if call_surface == "embeddings":
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
    call_surface: Literal["chat", "embeddings"],
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
) -> Fingerprint:
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
    return Fingerprint(armKind="model", callSurface=call_surface, fields=fields)


def run_pack(
    pack: Pack,
    cfg: RunConfig,
    *,
    lmstudio: LMStudio,
    root: Path,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[RunResult, tuple[DispatchFailureDisclosure, ...]]:
    """The CLI `run` command's one entry point (plan §3.6a). Raises `RunRefused` for every refusal
    in capture order (exit 3/4/5); returns the assembled `RunResult` plus zero or more dispatch-
    failure disclosures otherwise. Sequencing is §3.4.4a's ten capture-order steps, verbatim."""
    started_at = now()  # step 0

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
    model_info = _find_model(catalog, cfg.modelKey)
    call_surface = derive_call_surface(pack.manifest.get("environment", {}).get("requires"))
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

    prompt_config = pack.prompt_config()
    try:
        load_result = lmstudio.warm_up(  # step 4 — the mandatory, once-per-arm warm-up
            cfg.modelKey,
            call_surface=call_surface,
            system_prompt=prompt_config.systemPrompt if call_surface == "chat" else None,
            was_resident_before=model_info.id in {m.id for m in resident_at_start},
            timeout_s=cfg.firstCallTimeoutSeconds,
        )
    except LMStudioCallTimeout as exc:
        raise RunRefused(
            f"warm-up did not complete within --first-call-timeout "
            f"({cfg.firstCallTimeoutSeconds}s); nothing was scored (plan §3.6)",
            exitCode=3,
        ) from exc
    except LMStudioCallFailed as exc:
        raise RunRefused(f"warm-up call failed: {exc}", exitCode=3) from exc
    for _ in range(cfg.warmupExtra):  # --warmup <n>, additive, discarded
        lmstudio.warm_up(
            cfg.modelKey,
            call_surface=call_surface,
            system_prompt=prompt_config.systemPrompt if call_surface == "chat" else None,
            was_resident_before=True,
            timeout_s=cfg.firstCallTimeoutSeconds,
        )

    runtime_name, runtime_version = (
        _runtime_identity(load_result) if call_surface == "chat" else (None, None)
    )

    check = hostinfo.check_attestation_staleness(  # step 6
        host,
        call_surface=call_surface,
        residency_source=residency_source,
        runtime_name=runtime_name,
        runtime_version=runtime_version,
    )
    if check.stale:
        raise RunRefused(check.message or "attestation is stale", exitCode=5)
    if check.updated_host is not None:
        hostinfo.write_host_info(root, check.updated_host)

    baseline_residency = lmstudio.residency()  # step 7 — item-1's baseline, never resident_at_start
    catalog_after_load = lmstudio.catalog()  # step 8 — loadedContextLength

    if pack.role == "tool-caller":
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
    run_id = f"{pack.packId}-{cfg.modelKey}-{_utc_stamp(started_at)}".replace("/", "_")
    for attempt in range(4):  # step 10, 3 retries
        try:
            resident_at_end = lmstudio.residency()
            break
        except (LMStudioUnreachable, TimeoutError):
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
    )
    return run, disclosures
