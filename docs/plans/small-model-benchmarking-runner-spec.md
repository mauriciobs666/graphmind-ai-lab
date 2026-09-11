# `model-bench` runner and CLI `validate`/`run` — implementation spec

> **Status:** archived · **Owner:** `architect` · **Tracks:** — · **Extends:** `docs/plans/small-model-benchmarking.md` (S2)

## 1. Goal & scope

`docs/plans/small-model-benchmarking.md` (the "plan") assigns `modelbench/runner.py` and the CLI's
`validate`/`run` commands to stage S2 (§4 S2, `docs/plans/small-model-benchmarking.md:4985`), but
never gives `runner.py` a code-skeleton block the way it gives `packs.py`/`lmstudio.py`
(`:4990-5050`). Its requirements are scattered across §3.6, §3.6a, §3.8.4, §4 S1's `LatencyBlock`/
`CallTiming`/`ItemTiming` definitions, §4 S2's nine invariants, the v1.31 per-conversation-
`ToolEnvironment` addition, and `docs/plans/small-model-benchmarking-ml-dispatch-failure.md`
(the "dispatch-failure note", `Status: active`, `Owner: data-scientist`).

This document is that missing skeleton: a concrete, citable specification for `runner.py`'s
functions and types, and for the CLI's `validate`/`run` commands, synthesized from those scattered
sections into one place — **not** a revision of the plan (the plan gate is closed and this document
does not touch it) and **not** an implementation (no source is written here).

**In scope:** `modelbench/runner.py`'s entry points and internal shape; the `LatencyBlock`/
`CallTiming`/`ItemTiming` additions to `modelbench/results.py` and the conversion of
`ItemResult.latencyMs` from a stored field to a derived property; the CLI's `validate` and `run`
subcommands in `modelbench/cli.py`; the exit-code set (§3.6a, as amended by the dispatch-failure
note); the seam between the runner and each role's (not-yet-built) scorer module.

**Out of scope:** the scorer modules themselves (`scoring/toolcalls.py` is S5's, `scoring/
retrieval.py` etc. are S3/S4/S7's); `tools/sim.py` (S5/S6); `attest`'s CLI wiring (already shipped,
`modelbench/hostinfo.py`/`modelbench/cli.py`); anything in `stats.py`'s clustering-aware surface
(owned by `docs/plans/small-model-benchmarking-ml.md`, cited never restated per the plan's §7
rule 2). Where this document names a `-ml`-owned constant or formula it is not restated —
the one exception, flagged explicitly in §5, is a numeric threshold this document could not verify
because the `-ml` note was not opened in this session.

**CPG:** considered, not relevant — `model-bench` is a Python component with no loaded CPG
(`cpg_model-bench` does not exist among the graphs on this FalkorDB instance, confirmed by query);
this is a code-level task, so the applicable case is "considered, not relevant" rather than "not
applicable".

## 2. Context & findings

### 2.1 What already exists (read directly, not inferred from the plan)

Everything in this subsection was confirmed by reading `model-bench/modelbench/*.py` as shipped,
not by reading the plan's illustrative code sketches, because two real gaps between the plan's
S1/S2 skeleton and shipped code only show up that way (§2.2 below).

- `modelbench/packs.py` — `Pack`, `PackRef`, `load_pack`, `content_hash`, `validate_pack`,
  `check_sampling_contract`, `derive_call_surface`, `Pack.prompt_config()`,
  `Pack.load_tool_module()` are shipped and tested. `validate_pack` covers four of the plan's five
  named axes plus the `historyReplay`/`maxIterationsPerTurn` role scoping; **route (iii) of the
  `sampling` contract — `pairingKey[0] == roles.analysis_unit_field(role)`, swept over every role
  via a computed `roles.ANALYSIS_UNIT_FIELD_BY_ROLE` (plan §4 S2's "Done when", `:5078-5082`) — is
  not present in shipped `packs.py` or `roles.py`.** `roles.py` ships `ROLES`, `UNIT_KIND_BY_ROLE`,
  `MULTI_CALL_TURN_BY_ROLE` and `unit_kind()` only. This is `packs.py`'s gap, not `runner.py`'s —
  flagged here because CLI `validate` wraps `validate_pack` directly (§4 below) and is therefore
  incomplete relative to §4 S2's own done-conditions until it is closed. Not this document's to fix.
- `modelbench/lmstudio.py` — `LMStudio` (`probe`, `catalog`, `residency`, `chat`, `embed`,
  `warm_up`), `ChatResult` (with the derived `ttftMs`/`generationMs`/`tokensPerSecond` trio and raw
  `stats`), `EmbedResult`, `LoadResult`, `ModelInfo`, `ResidentModel`, `LMStudioCallTimeout`,
  `LMStudioCallFailed` (carrying `status: int | None`, required keyword, no default),
  `tool_calling_eligible`/`check_tool_calling_eligibility` are shipped and tested, including the
  `-m live` tests.
- `modelbench/hostinfo.py` — `attest`, `read_host_info`/`write_host_info`/`validate_host_info`,
  and **`check_attestation_staleness`**, which already implements capture-order step 6 / §3.4.5
  point 3's trip-wire as a pure function (`host`, `call_surface`, `residency_source`,
  `runtime_name`, `runtime_version` → `AttestationCheck(outcome, stale, message, updated_host)`).
  The runner does not reimplement the trip-wire; it calls this.
- `modelbench/convo.py` — `PromptConfig`, `Turn`, `Conversation`, `TurnTrace`, `ConversationTrace`,
  `TurnDisposition`/`TURN_DISPOSITIONS` (five members), `assemble`, `drive`, `TraceContractViolated`,
  `ToolDispatchFailed` are shipped and tested. `drive()`'s per-turn loop, its five-way disposition
  set, and its precedence (exception tested before the cap) all match §3.8.4 as built.
- `modelbench/tooling.py` — `ToolEnvironment` (a `@runtime_checkable` `Protocol`), `DispatchRecord`
  are shipped.
- `modelbench/results.py` — `ItemResult`, `RunResult`, `Aggregates`, `BinaryMetric`,
  `ContinuousMetric`, `DistributionSummary`, `TurnPositionRate`, `store`/`load_history`/
  `rebuild_index` are shipped and tested. **`LatencyBlock`, `CallTiming`, `ItemTiming`,
  `ItemResult.timing`, `RunResult.latency` and `RunResult.attestationTripWire` do not exist.**
- `modelbench/cli.py` — `compare`, `index rebuild`, `models --tested` (stored-records half),
  `attest` are shipped. Its own module docstring already states *"`validate` and `run` are a later
  S2 unit's and are deliberately still absent"* and its exit-code constants
  (`EXIT_OK=0`, `EXIT_USAGE=2`, `EXIT_LMSTUDIO_UNREACHABLE=3`, `EXIT_BAD_PACK=4`,
  `EXIT_FINGERPRINT=5`) are already defined for `validate`/`run` to reuse. Its docstring's exit-`4`
  line ("invalid pack") predates the dispatch-failure note's amendment and needs the "or, after
  `run`'s artifacts are written, a dispatch-censoring finding" clause (§3.6a,
  `docs/plans/small-model-benchmarking.md:1933-1938`).

### 2.2 Two genuine plan-vs-shipped-code gaps found in this session, not previously flagged anywhere

These are not contradictions inside the plan — the plan is internally consistent on both points —
they are places where shipped S1 code diverged from what the plan's own owning sections (§7 rule 4:
where a derived surface disagrees with the section that owns the type, the section is right)
already decided. Both are required by `runner.py`'s own dependencies, so both are load-bearing for
this spec and are called out as their own implementation step (§6, step 0).

1. **`ItemResult.latencyMs` is currently a stored constructor field (`float | None`), not a derived
   `@property` over `ItemTiming`.** The plan's own §4 S1 (`:3087-3097`) and Appendix A's
   `ItemResult` row (`:7825`) are unambiguous: *"`ItemResult.timing` replaces the stored `latencyMs`,
   which becomes a **property** over it."* Shipped `results.py:164` still declares
   `latencyMs: float | None` as a constructor parameter, and it is used that way at roughly thirty
   call sites across `tests/conftest.py`, `tests/test_results.py`, `tests/test_report.py` and
   `tests/test_cli.py` (`grep -rn latencyMs model-bench/modelbench model-bench/tests`, counted in
   this session). Converting it is required to build `LatencyBlock` at all — `ItemTiming` is its
   only source of item-level timing — and every one of those call sites must move from
   `latencyMs=X` to `timing=ItemTiming(wallClockMs=X, calls=(), withheldFor=None)` (or
   `timing=None` for a `None` `latencyMs`). This is squarely an "edit to shipped, tested code"
   in the same category the plan already asks for elsewhere (e.g. the `TurnDisposition` widening,
   `docs/plans/small-model-benchmarking.md:5222-5277`) — it is called out here because the plan
   never wrote it as a numbered edit anywhere, since `runner.py`/`LatencyBlock` did not exist yet
   when S1 shipped.
2. **`ToolDispatchFailed.__init__(self, message, *, toolName, turnIndex)` does not carry the
   partial `ConversationTrace` the dispatch-failure note requires it to carry.** The note's §4(b)
   (`docs/plans/small-model-benchmarking-ml-dispatch-failure.md:105-111`) rules: *"`drive` wraps the
   `env.dispatch` call site, raises the named class... carrying **the completed turns'
   `ConversationTrace`**, the turn index, the tool name and the parsed arguments and the original
   exception."* Shipped `convo.py:352` carries only `toolName`/`turnIndex` (plus `__cause__` via
   `raise ... from exc`); it does not carry the completed `TurnTrace`s or the parsed arguments. The
   runner needs the completed turns to build the censored `ConversationTrace` it stores (§4
   below), and cannot reconstruct them itself — `drive()` builds `turn_traces` as a private local
   and only returns a `ConversationTrace` on a clean finish, so once `ToolDispatchFailed` propagates
   out of `_drive_turn`, the only turns the caller has are whatever the exception carries. **This is
   a required, small edit to `convo.py`, specified in §4 below rather than left to the coder to
   improvise**, since the exact payload shape is a genuine synthesis decision (§4, step 1).
   `ToolDispatchFailed`'s docstring also still frames the abort-vs-record question as *"deliberately
   not decided"* (`convo.py`, the class docstring) — it is decided, by the note; the docstring
   needs a one-paragraph rewrite to stop asserting otherwise, on the same principle as the plan's own
   shipped-docstring sweeps (`docs/plans/small-model-benchmarking.md:5228-5250`).

### 2.3 What the plan gives verbatim, and where

Every fact restated as a signature or invariant below is drawn from one of these citable homes; §7
below lists them again per implementation step so a chunk's dispatch brief can point at one place:

| Topic | Plan citation |
|---|---|
| `LatencyBlock`/`CallTiming`/`ItemTiming` shapes | §4 S1 `:2985-3025`, Appendix A `:7826-7828` |
| The nine `LatencyBlock` invariants | §4 S2 `:5416-5576` |
| The unit boundary (`ttftMs`/`generationMs`/`tokensPerSecond`/`wallClockMs`) | §3.6 `:1487-1512` |
| The mandatory warm-up, two timeout budgets, contamination guard, `unexplainedMs` | §3.6 `:1524-1832` |
| Capture order (steps 0–10) | §3.4.4a `:1166-1215` |
| `callSurface` cross-check and tool-calling eligibility gate scoping | §3.4.4a `:1217-1247`, §3.6 `:1874-1904` |
| Attestation trip-wire (three outcomes) | §3.4.4a step 6 `:1200-1209`, §3.4.5 point 3 `:1357-1407` (already implemented, `hostinfo.check_attestation_staleness`) |
| Three enforcement points (write refuses / read quarantines / attestation staleness) | §3.4.5 `:1348-1363` |
| The tool-caller per-turn loop and five-way disposition set | §3.8.4 `:2480-2679`, already implemented (`convo.drive`) |
| One `ToolEnvironment` per conversation | §4 S2, v1.31 addition `:5641-5652` |
| A raising `dispatch` censors the conversation at `t`, exit `4` after artifacts written | `-ml-dispatch-failure.md` §2–§4 (whole document), plan §3.6a `:1933-1938` |
| The determinism probe and `basis` wiring | §3.8.4 `:2233-2264` |
| CLI surface table, exit codes | §3.6a `:1905-1945` |

## 3. Design & rationale

### 3.1 The runner is a capture-order orchestrator plus two driving loops, not a monolith

`runner.py`'s job, read across every section in §2.3, decomposes into three layers that the plan
never names together but that its own citations force apart:

1. **Capture-order orchestration** (§3.4.4a's ten steps) — sequencing calls into `hostinfo`,
   `lmstudio.LMStudio`, and `fingerprint.Fingerprint` construction. Every one of these calls
   already exists as a tested function; the runner's job here is purely sequencing and refusal
   (which exit code, which message, at which step).
2. **Two driving loops, selected by `roles.MULTI_CALL_TURN_BY_ROLE[pack.role]`** — a **single-call
   item loop** (the four item-level roles: one `chat()`/`embed()` call per item) and a
   **per-conversation loop** (`tool-caller` alone: one fresh `ToolEnvironment` and one `convo.drive`
   call per script, `ToolDispatchFailed` caught and censored). Both loops produce the same two
   things per unit of work: a raw call/turn result, and a `CallTiming`/`ItemTiming` built from it
   under the residency guard and the `unexplainedMs` gap detector (§3.6, shared by both loops —
   the guard and the detector are stated once in the plan, over "an item", and apply identically to
   a single-call item and to a `tool-caller` turn, which the plan itself calls "an item" on that
   role, `:1826`).
3. **`LatencyBlock` accumulation and `RunResult` assembly** — one pass over the finished `items`
   tuple, per §4 S2's nine invariants (§5 below), independent of which loop produced the items.

### 3.2 The scorer seam — a genuine synthesis decision, marked as such

Neither loop can decide `ItemResult.outcome`/`scoreable`/`counts`/`measures` itself: that is each
role's scorer module's job (`scoring/toolcalls.py` for S5, `scoring/retrieval.py` etc. for
S3/S4/S7 — none shipped yet), and the plan states this as a general rule (*"S2's scorers derive
`aggregates` from the same `items` they emit, in one pass"*, `:5594-5613`) without ever writing
`runner.py`'s side of that seam. **This document's own design choice, since the plan does not fix
one:** the runner calls one role-resolved scorer function per unit of work (single-call item) or
once per completed run (tool-caller, which needs cross-turn state for the hazard/per-position
table) and receives back complete `ItemResult`s carrying the `timing` the runner supplies as an
input, never as something the scorer invents:

```python
# my design choice — not specified by the plan, which names the scorer contract in prose
# (docs/plans/small-model-benchmarking.md:5594-5613) but never the runner-side call shape.

class ItemScorer(Protocol):
    """One item-level role's scorer (embedder, guard-judge, nlq-generator, chat-responder)."""
    def score_item(
        self, item_input: Mapping[str, Any], result: ChatResult | EmbedResult | None,
        timing: ItemTiming, *, pack: Pack,
    ) -> ItemResult: ...

class ConversationScorer(Protocol):
    """tool-caller's scorer — needs the whole (possibly censored) trace for cross-turn state
    (the hazard, the per-position table, cleanThroughTurnH's censoring, -ml §4.3 rule 5)."""
    def score_conversations(
        self, scored: Sequence[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]],
        probes: Sequence[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]],
        *, pack: Pack,
    ) -> tuple[tuple[ItemResult, ...], ToolCallAggregates]: ...
```

`ConversationScorer.score_conversations` takes the 12 scored conversations and the 2 determinism-
probe re-runs **separately** because the probe's conversations are diagnostic and must never enter
`items`/`aggregates`' counted population (§3.8.4: *"excluded from every denominator, appear only in
the report's own probe line"*) — the scorer is what builds `ToolCallAggregates.determinismProbe`
from comparing the two populations' outcome vectors (§4.4 below), so it needs both. `result` is
`None` on `ItemScorer.score_item` exactly when the call did not complete (timeout / no-response),
matching `ItemTiming.withheldFor is not None`; a scorer decides the failed item's `outcome` from
that, not the runner (§3.4 below explains why the runner does not decide it either).

A pack names its scorer by the manifest's `"scorer"` key (e.g. `"toolcalls"`, the tool-caller
example manifest, `docs/plans/small-model-benchmarking.md:441`); resolving that string to a
`modelbench.scoring.<name>` module and a `SCORE_ITEM`/`SCORE_CONVERSATIONS` entry point is this
document's own naming choice (no plan citation), stated because `run_pack` (§4) needs a concrete
`import` line and none exists. **Until S3 ships the first scorer module, `run --pack <any>` fails
at that import** — expected and correct, since S2's own "Done when" (`:5073-5126`) never exercises
a real pack run; S2's runner tests are entirely offline against a stub LLM and a fixture pack.

### 3.3 `ItemTiming`/`CallTiming` construction is the runner's, unconditionally, on both loops

Every rule in §3.6 (the unit boundary, the warm-up, the two timeout budgets, the residency guard,
the co-presence exclusion, `unexplainedMs`) is stated once, generically, and the plan is explicit
that it must not be read as tool-caller-only (`:1737-1801`'s "Withholding is at the unit of the
figure" block states the per-call/per-item split as a property of *any* item, and test 15b case
(f), `:6259-6271`, is a single-call `guard-judge` fixture pinning exactly this). So both loops
share one internal helper for the residency guard and one for the gap detector; only the shape of
"one call" vs. "one turn of several calls" differs.

### 3.4 Why the runner, not the scorer, decides `withheldFor` (and why that is not a scoring decision)

`withheldFor` is a timing-record concern — *what does `latencyMs` reflect* — and is decided by two
producers that are properties of the **run's** execution, not of any one item's content: the
between-item residency snapshot (cross-item state only the runner accumulates) and the per-call gap
sum (computable from `CallTiming` alone). Neither needs the pack's scoring rules. What a
`withheldFor`-carrying item **scores** (`fail` for a timeout, `unrunnable` for everything else
non-completing, per `-ml` §4.3 rule 4, cited not restated per §3.8.4's own citation discipline) is
the scorer's, because that mapping is role-specific in general (a `tool-caller` turn's disposition
table, `:2518-2524`, has no analogue yet defined for a single-call role's bare
`LMStudioCallTimeout`/`LMStudioCallFailed`) and this document does not invent one for the four
item-level roles — that is each role's scorer's decision at S3/S4/S7, informed by whatever `-ml`
rules for that role, not fixed here.

### 3.5 Alternative considered and rejected: one driving loop, dispatched on a `turnDisposition`-shaped abstraction

Forcing the four single-call roles through a fake one-iteration "turn" (so both loops share
`convo.drive`'s machinery) was considered and rejected: `-ml` §4.3 rule 4's five-way disposition
vocabulary is `TurnTrace`'s and is explicitly a **mechanism vocabulary for a bounded multi-call
loop** (§3.8.4 `:2509-2532`); forcing it onto a single call would be exactly the two-vocabularies
collision the plan has refused three times already (`BinaryMetric.unit`/`PackRef.analysisUnit`,
`unit_kind`/`analysis_unit_field`, and the dispatch-failure note's own refusal of a sixth
disposition member, `-ml-dispatch-failure.md` §4(c)). Two loops sharing two small helpers
(residency guard, gap detector) is more code than one loop, and is the design that does not invent
a mapping the plan has three times ruled against inventing.

## 4. `runner.py` — signatures

```python
# runner.py — my synthesis of docs/plans/small-model-benchmarking.md §3.4.4a, §3.6, §3.8.4, §4 S1,
# §4 S2, and docs/plans/small-model-benchmarking-ml-dispatch-failure.md. Citations inline; anything
# uncited in a docstring below is this document's own naming/sequencing choice.

@dataclass(frozen=True)
class RunConfig:
    """--run's flags (plan §3.6a `run --pack <id> --model <key>`), plus the two budgets §3.6
    requires the runner to size (never a pack or adapter default — both are required keyword args
    on LMStudio.chat/embed/warm_up already)."""
    modelKey: str
    sessionId: str | None
    referenceKey: str | None           # --reference; not read by run_pack itself (§7's pairing
                                        # concern belongs to `compare`, already shipped)
    warmupExtra: int = 0               # --warmup <n>: EXTRA warm-up calls before item 1 (plan
                                        # §3.6a: "additional warm-up calls before the first item,
                                        # never items removed after the fact")
    firstCallTimeoutSeconds: float = 300.0   # plan §3.6 default
    requestTimeoutSeconds: float = 120.0     # plan §3.6 default


class RunRefused(RuntimeError):
    """The run stopped before or during capture order and wrote nothing (exit 3/4/5 territory,
    plan §3.6a). Carries `exitCode` so the CLI does not re-derive it from message text."""
    def __init__(self, message: str, *, exitCode: int) -> None:
        super().__init__(message)
        self.exitCode = exitCode


@dataclass(frozen=True)
class DispatchFailureDisclosure:
    """One censored conversation, for the CLI's `PACK DISPATCH FAILURES` block (dispatch-failure
    note §4(d)). `(scriptId, turn, tool, reason)` — the note's own tuple, verbatim."""
    scriptId: str
    turn: int
    tool: str
    reason: str


def run_pack(
    pack: Pack,
    cfg: RunConfig,
    *,
    lmstudio: LMStudio,
    root: Path,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[RunResult, tuple[DispatchFailureDisclosure, ...]]:
    """The CLI `run` command's one entry point (plan §3.6a). Raises `RunRefused` for every
    refusal in capture order (exit 3/4/5); returns the assembled `RunResult` plus zero or more
    dispatch-failure disclosures otherwise — the caller (§6 below) is what turns a non-empty
    disclosure tuple into exit `4` *after* `results.store()` has written the record (dispatch-
    failure note §4(d): "the record is written, then run exits 4").

    Sequencing is §3.4.4a's ten capture-order steps, verbatim (cited per step below); nothing here
    reorders them, because where the staleness trip-wire fires depends on the order (§3.4.4a
    :1166-1168).
    """
    started_at = now()                                                    # step 0 (§3.4.4a)

    host = hostinfo.read_host_info(root)                                  # step 1
    # absent/invalid -> RunRefused(exitCode=5); hostinfo.read_host_info already raises
    # HostInfoError on schema-invalid — wrap it, do not reimplement validate_host_info here.

    probe_result = lmstudio.probe()                                       # step 2
    if probe_result != "api-v0":
        raise RunRefused(_PROBE_MESSAGES[probe_result], exitCode=3)       # two distinct messages,
                                                                            # §3.4.4a's own wording

    resident_at_start = lmstudio.residency()                              # step 3
    residency_source = "api-v0"    # the surface that answered step 2/3 — see note below

    catalog = lmstudio.catalog()                                          # step 3a
    model_info = _find_model(catalog, cfg.modelKey)                       # exit 4 if absent
    call_surface = packs.derive_call_surface(pack.manifest["environment"]["requires"])
    _check_call_surface_cross_check(call_surface, model_info)             # §3.4.4a — exit 4
    if pack.role == "tool-caller":
        check_tool_calling_eligibility(pack.role, model_info)             # §3.6 — exit 4,
                                                                            # role-scoped

    load_result = lmstudio.warm_up(                                      # step 4
        cfg.modelKey, call_surface=call_surface,
        system_prompt=pack.prompt_config().systemPrompt if call_surface == "chat" else None,
        was_resident_before=model_info.id in {m.id for m in resident_at_start},
        timeout_s=cfg.firstCallTimeoutSeconds,
    )
    # LMStudioCallTimeout here -> RunRefused(exitCode=3, "--first-call-timeout"); the warm-up's
    # own timeout is never a scored call (§3.6 clause (iv), "nothing written").
    for _ in range(cfg.warmupExtra):                                      # --warmup <n>, additive
        lmstudio.warm_up(cfg.modelKey, call_surface=call_surface, ...)    # same call, discarded

    runtime_name, runtime_version = _runtime_identity(load_result)        # step 5, chat-surface
                                                                            # only (None, None) on
                                                                            # embeddings

    check = hostinfo.check_attestation_staleness(                        # step 6
        host, call_surface=call_surface, residency_source=residency_source,
        runtime_name=runtime_name, runtime_version=runtime_version,
    )
    if check.stale:
        raise RunRefused(check.message, exitCode=5)                      # "and host.json is not
                                                                            # written" on the
                                                                            # first-observation+stale
                                                                            # sub-case (§3.4.4a)
    if check.updated_host is not None:
        hostinfo.write_host_info(root, check.updated_host)               # only on first-observation

    baseline_residency = lmstudio.residency()                            # step 7 — the guard's
                                                                            # item-1 baseline,
                                                                            # NEVER resident_at_start

    catalog_after_load = lmstudio.catalog()                              # step 8 — loadedContextLength

    items, disclosures, basis, design_effect = _drive_scored_items(       # step 9
        pack, cfg, lmstudio=lmstudio, model_info=model_info,
        call_surface=call_surface, baseline_residency=baseline_residency,
    )

    for attempt in range(4):                                              # step 10, 3 retries
        try:
            resident_at_end = lmstudio.residency()
            break
        except (LMStudioUnreachable, TimeoutError):
            if attempt == 3:
                raise RunRefused(
                    "residentModelsAtEnd could not be captured after three retries; "
                    f"transcript retained at results/transcripts/{run_id}.jsonl", exitCode=5,
                )
            sleep(1.0)
    ended_at = now()

    fingerprint = _build_fingerprint(                                    # assembles all 30 fields
        pack=pack, model_info=model_info, call_surface=call_surface,
        host=host, started_at=started_at, ended_at=ended_at,
        resident_at_start=resident_at_start, resident_at_end=resident_at_end,
        residency_source=residency_source, runtime_name=runtime_name,
        runtime_version=runtime_version, catalog_after_load=catalog_after_load,
    )
    latency = latency_block(items)                                       # §5 below, None-able
    run = RunResult(
        runId=..., sessionId=cfg.sessionId, role=pack.role, armKind="model",
        fingerprint=fingerprint, items=items, aggregates=aggregates,
        designEffect=design_effect, basis=basis,
        attestationTripWire=check.outcome, latency=latency,
    )
    return run, disclosures
```

**Two internal driving functions**, selected once inside `_drive_scored_items` on
`roles.MULTI_CALL_TURN_BY_ROLE[pack.role]` (§3.3 above):

```python
def _drive_single_call_items(
    pack: Pack, cfg: RunConfig, *, lmstudio: LMStudio, model_info: ModelInfo,
    call_surface: Literal["chat", "embeddings"], baseline_residency: list[ResidentModel],
) -> tuple[tuple[ItemResult, ...], float, Basis]:
    """The four item-level roles. One call per item; `ItemTiming.calls` is always length 0 or 1.
    designEffect == 1.0, basis == "by-construction" — no clustering: sampling unit == item
    (plan §3.8.1's "designEffect 1.00 by construction" for the embedder, :3363, generalised here to
    every item-unit role, since none of them cluster observations — this generalisation is this
    document's own inference, not quoted from a per-role bullet for the other three roles)."""
    scorer = _load_item_scorer(pack)
    resident = baseline_residency
    results: list[ItemResult] = []
    for item_input in pack.iter_items():          # naming/shape: this document's own — packs.py's
                                                    # per-role item iteration is not spec'd anywhere
                                                    # either and is out of this document's scope
        resident = lmstudio.residency()            # between-item probe — never during a timed call
        try:
            call = (lmstudio.chat(..., timeout_s=cfg.requestTimeoutSeconds) if call_surface == "chat"
                    else lmstudio.embed(..., timeout_s=cfg.requestTimeoutSeconds))
        except LMStudioCallTimeout:
            timing = ItemTiming(wallClockMs=None, calls=(), withheldFor="timeout")
            results.append(scorer.score_item(item_input, None, timing, pack=pack))
            continue
        except LMStudioCallFailed:
            timing = ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")
            results.append(scorer.score_item(item_input, None, timing, pack=pack))
            continue
        call_timing = _call_timing(call)
        withheld = _load_withheld_for(model_info, resident) or _gap_withheld_for(call_timing)
        timing = ItemTiming(wallClockMs=call.wallClockMs, calls=(call_timing,), withheldFor=withheld)
        results.append(scorer.score_item(item_input, call, timing, pack=pack))
    return tuple(results), 1.0, "by-construction"


def _drive_conversations(
    pack: Pack, cfg: RunConfig, *, lmstudio: LMStudio, model_info: ModelInfo,
    baseline_residency: list[ResidentModel],
) -> tuple[tuple[ItemResult, ...], tuple[DispatchFailureDisclosure, ...], float, Basis]:
    """tool-caller only. One fresh ToolEnvironment per conversation (plan §4 S2, v1.31 addition,
    :5641-5652 — "never one reused across a pack's items"), drives the 12 scored scripts then the
    2 determinism-probe scripts (§3.8.4 :2233-2264), catches ToolDispatchFailed per script."""
    scorer = _load_conversation_scorer(pack)
    resident = baseline_residency
    scored: list[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]] = []
    disclosures: list[DispatchFailureDisclosure] = []
    llm = _bind_llm(lmstudio, model_info.id, timeout_s=cfg.requestTimeoutSeconds)

    for script in pack.iter_scripts():             # "scripts" naming per plan §3.8.4's own term
        env = pack.load_tool_module().build_environment()   # ONE per conversation — the fresh-env
                                                              # obligation this whole loop exists for
        try:
            trace = drive(env, script, llm, pack.prompt_config())
            censored = False
        except ToolDispatchFailed as exc:
            # See §2.2 item 2 and §6 step 1: exc must carry `completedTurns` (this document's own
            # required extension of ToolDispatchFailed's payload).
            trace = ConversationTrace(
                scriptId=script.scriptId, shape=script.shape, replicate=script.replicate,
                turns=exc.completedTurns,
            )
            disclosures.append(DispatchFailureDisclosure(
                scriptId=script.scriptId, turn=exc.turnIndex, tool=exc.toolName,
                reason=f"{type(exc.__cause__).__name__}: {exc.__cause__}",
            ))
            censored = True
        item_timings, resident = _turn_timings(trace, resident_before=resident, model_id=model_info.id,
                                                lmstudio=lmstudio)
        scored.append((script, trace, item_timings))

    probes: list[tuple[Conversation, ConversationTrace, tuple[ItemTiming, ...]]] = []
    for script_id in pack.manifest["sampling"]["determinismProbeScripts"]:  # §3.8.4
        script = pack.find_script(script_id)
        env = pack.load_tool_module().build_environment()   # fresh env here too — same obligation
        try:
            trace = drive(env, script, llm, pack.prompt_config())
        except ToolDispatchFailed:
            continue   # a probe re-run that itself censors is "not identical" by construction;
                        # the scorer's comparison (§3.2) handles this from `probes`' shorter trace
        item_timings, resident = _turn_timings(trace, resident_before=resident, model_id=model_info.id,
                                                lmstudio=lmstudio)
        probes.append((script, trace, item_timings))

    items, aggregates = scorer.score_conversations(scored, probes, pack=pack)
    ran = len(probes) == len(pack.manifest["sampling"]["determinismProbeScripts"])
    replicates_one = pack.manifest["sampling"]["replicatesPerScript"] == 1
    basis: Basis = (
        "by-construction"
        if replicates_one and ran and aggregates.determinismProbe["identical"]
        else "assumed"
    )  # plan §3.8.4 :2250-2261, test 12b's four cases :6169-6176
    design_effect = stats.design_effect(...)  # -ml's signature, not restated (§7 rule 2)
    return items, tuple(disclosures), basis, design_effect
```

**`_turn_timings`** builds one `ItemTiming` per turn of a `ConversationTrace` from
`TurnTrace.chatResults` (one `CallTiming` per completed call, in order — plan §4 S2
`:2716-2731`), applies the residency guard and the gap detector, and returns the updated
`resident` snapshot for the next conversation's probe. `withheldFor` on a turn follows the
precedence §3.6/§3.8.4 state repeatedly: a turn whose `turnDisposition ∉ {"replied", "cap-hit"}`
takes its `withheldFor` from that disposition (`"timeout"` for `timed-out`, `"no_response"` for
`no-response`/`server-rejected`) and **no load producer runs on that item** — neither the
residency guard nor the gap detector may override a failing disposition (plan §3.6 `:1759-1796`,
P15-3's ruling).

**`_load_withheld_for`/`_gap_withheld_for`** are the two load producers, each returning
`"load" | None`, shared by both driving loops (§3.3 above):

```python
def _load_withheld_for(model_info: ModelInfo, resident_before: list[ResidentModel]) -> Literal["load"] | None:
    """The between-item residency probe (plan §3.6): the item is withheld for load iff the
    PRECEDING snapshot did not show the model resident. Never during a timed call."""
    return None if model_info.id in {m.id for m in resident_before} else "load"

def _gap_ms(call: CallTiming) -> float | None:
    """-ml §11.5.1's per-call gap: wallClockMs minus (ttftMs + generationMs). None if any operand
    is None — a chat-surface-only detector (plan §3.6 :1730-1736: no `stats` on embeddings, so no
    gap). Formula stated in the plan's own docstring-sweep note (:5384-5392); NOT restated as
    arithmetic anywhere the plan calls out as owned by -ml, because this particular line — unlike
    the threshold below — IS the plan's own to state (it names the operands at the unit boundary,
    §3.6 :1487-1512)."""
    if call.wallClockMs is None or call.ttftMs is None or call.generationMs is None:
        return None
    return call.wallClockMs - (call.ttftMs + call.generationMs)

def _gap_withheld_for(item_timing: ItemTiming) -> Literal["load"] | None:
    """Sums _gap_ms over item_timing.calls; None unless EVERY call yields a gap (plan :3013-3014).
    Withholds iff the sum exceeds -ml §11.5.1's threshold. **This document did not verify the
    threshold's numeric value** — the -ml note was not opened this session; §6 step 2 requires the
    implementer to read docs/plans/small-model-benchmarking-ml.md §11.5.1 directly rather than
    trust a number guessed here. Test 15b's fixtures use 3 485.6 ms as an over-threshold gap
    (plan :6237, :6251), which bounds the threshold from above but does not state it."""
```

## 5. `LatencyBlock`'s nine invariants — concrete, checkable assertions

Source: `docs/plans/small-model-benchmarking.md:5416-5576` (rules), Appendix A `:7826-7828`
(shapes). Field names and computations exactly as the plan states them; nothing here is this
document's invention.

```python
@dataclass(frozen=True)
class CallTiming:                                          # results.py, plan §4 S1 :2987-2996
    wallClockMs: float | None
    ttftMs: float | None
    generationMs: float | None
    promptTokens: int | None
    tokensPerSecond: float | None

@dataclass(frozen=True)
class ItemTiming:                                           # results.py, plan §4 S1 :2999-3025
    wallClockMs: float | None
    calls: tuple[CallTiming, ...]
    withheldFor: Literal["load", "timeout", "no_response"] | None

    @property
    def callCount(self) -> int: return len(self.calls)

    @property
    def unexplainedMs(self) -> float | None:
        gaps = [_gap_ms(c) for c in self.calls]
        return None if not self.calls or any(g is None for g in gaps) else sum(gaps)

@dataclass(frozen=True)
class LatencyBlock:                                         # results.py, plan §4 S2/Appendix A
    latencyMsP50: float | None
    latencyMsP95: float | None
    latencyMsMax: float | None
    latencyTimedCount: int
    latencyItemCount: int
    latencyWithheldForLoad: int
    latencyWithheldForNoResponse: int
    statsCoveredCount: int | None
    callCount: int
    ttftMsMedian: float | None
    prefillMsPer1kMedian: float | None
    tokensPerSecondMedian: float | None
    unexplainedMsMax: float | None

    @property
    def callAttemptedCount(self) -> int:
        return self.callCount + self.latencyWithheldForNoResponse   # rule (ii)'s 4th assertion
```

`latency_block(items: Sequence[ItemResult]) -> LatencyBlock | None` computes every field **in one
pass over `items`** and asserts each rule below against a **recomputation** from the same `items`
(plan: *"every figure and every count is computed from `run.items` in one pass and asserted against
a recomputation from them"*, `:5413-5415`). Returns `None` iff every item's `timing is None` (a
`deterministic` arm, or an S1 fixture — rule (v)).

| # | Invariant | Exact computation |
|---|---|---|
| **(i)** | Wall-clock trio `None` only on `-ml` §11's gate refusal, never `0`. The three `stats`-derived medians are `None` under a gate refusal **or** absence-of-input, and the block does not distinguish the two — a reader checks `statsCoveredCount`. `unexplainedMsMax` is **split out**: a maximum over items with a reading, no coverage gate, `None` only on absence-of-input. | `latencyMsP50/P95/Max = -ml.percentile(...)` over `{i.latencyMs for i in items if i.latencyMs is not None}` (gated); `unexplainedMsMax = max((i.timing.unexplainedMs for i in items if i.timing and i.timing.unexplainedMs is not None), default=None)` — **no gate** |
| **(ii)** | `latencyItemCount == len(items)`. `callCount == Σ len(i.timing.calls)`. Each item's `len(timing.calls) == TurnTrace.iterations` (tool-caller only). **4th assertion:** `callAttemptedCount == Σ_items (len(timing.calls) + [that item's last call did not return])` — recomputed from dispositions, asserted equal to the derived `callCount + latencyWithheldForNoResponse`. **Independent branch on single-call model arms** (`armKind == "model"` and `roles.MULTI_CALL_TURN_BY_ROLE[role] is False`): `callAttemptedCount == latencyItemCount`, asserted from disjoint inputs — catches the mis-file P16-1 named (an item whose one call failed but whose `withheldFor` was wrongly set `"load"`). **Not asserted on tool-caller**, where the two are meant to differ (§4 S5 *Done when* item 7). | `latencyItemCount = len(items)`; `callCount = sum(len(i.timing.calls) for i in items if i.timing)` |
| **(iii)** | `latencyWithheldForLoad + latencyWithheldForNoResponse == latencyItemCount − latencyTimedCount` | derived counts, always, for every withheld item |
| **(iv)** | `statsCoveredCount` = count of **calls** with both a usable `stats` object and usable `promptTokens` (rule iv-c). Inequality `statsCoveredCount ≤ callCount` (not equality — a load-contaminated call still counts) | `sum(1 for i in items if i.timing for c in i.timing.calls if c.ttftMs is not None and c.promptTokens is not None and c.promptTokens > 0)` |
| **(iv-a)** | On a call surface with no `stats` (embeddings; every `deterministic` arm), `statsCoveredCount` and the four sibling figures (three medians + `unexplainedMsMax`) are `None`, never `0`. Condition is the **call surface**, not `armKind`. | `statsCoveredCount = None` when `call_surface == "embeddings"` |
| **(iv-b)** | The three sibling medians take `-ml` §11.6's p50 gate against **their own coverage** — `X = statsCoveredCount`, `Y = callAttemptedCount` (never `latencyItemCount`, never `callCount`) | gate formula is `-ml`'s, cited not restated |
| **(iv-c)** | Co-presence: a **call** with `stats` but `promptTokens` absent/`≤0` is excluded from `statsCoveredCount` **and** from all three medians' inputs — per call, never per item; nothing raises | disposition, not assertion — four edit sites: the count and each median's input set |
| **(v)** | `LatencyBlock` is `None` (not an empty instance) iff every item's `timing is None` | `return None if all(i.timing is None for i in items) else LatencyBlock(...)` |
| **(vi)** | `latencyTimedCount` = count of items whose derived `i.latencyMs is not None`. `latencyWithheldForLoad` = count of `timing.withheldFor == "load"`. `latencyWithheldForNoResponse` = count of `timing.withheldFor in ("timeout", "no_response")` — **one counter over two item states** | all three derived from `items`, never accumulated by a second running counter |

`ItemResult.latencyMs` (the derived property, §2.2 item 1) is:

```python
@property
def latencyMs(self) -> float | None:
    t = self.timing
    return None if t is None or t.withheldFor is not None else t.wallClockMs
```

`censoringExact` (`-ml` §11.5.1, plan `:5577-5593`) is computed **at render time in `report.py`**,
never stored — out of `runner.py`'s scope entirely; listed here only so its two inputs
(`ItemTiming.wallClockMs`, surviving on a `"load"`-withheld item; `timing.withheldFor`) are
confirmed present on the record the runner writes.

## 6. The CLI's `validate` and `run` commands

Both wrap already-shipped machinery; neither reimplements a check. Signature convention matches
`cli.py`'s existing `_cmd_compare`/`_cmd_attest` shape (`argparse.Namespace -> int`).

### 6.1 `validate --pack <path> [--strict]`

```python
def _cmd_validate(args: argparse.Namespace) -> int:
    """plan §3.6a's `validate` row (:1917). Structural only — no LM Studio connection, no model
    catalog (validate_pack's own docstring already states this scope, packs.py :735-737)."""
    try:
        pack = load_pack(Path(args.pack))
    except (OSError, PackConfigError) as exc:
        print(f"model-bench: cannot load pack at {args.pack}: {exc}", file=sys.stderr)
        return EXIT_BAD_PACK
    problems = validate_pack(pack)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        return EXIT_BAD_PACK
    print(f"{pack.packId} {pack.packVersion} ({pack.role}): valid")
    return EXIT_OK
```

`--strict` is named in §3.6a's table with no further elaboration anywhere in the plan — **this
document did not find a distinguishing rule for it** and does not invent one; flagged as an open
question in §8 rather than guessed.

`validate` never calls LM Studio and never writes anything (§3.6a: *"Not here: the `callSurface`-
versus-catalog-`type` cross-check and the tool-calling eligibility gate are `run`'s"*, `packs.py`
`:735-737`).

### 6.2 `run --pack <id> --model <key> [--session <id>] [--reference <key>] [--warmup <n>] [--first-call-timeout <s>] [--request-timeout <s>]`

```python
def _cmd_run(args: argparse.Namespace) -> int:
    """plan §3.6a's `run` row (:1918): 'calls validate first and fails closed'."""
    pack = load_pack(_pack_root(args.root, args.pack))
    problems = validate_pack(pack)
    if problems:
        for p in problems: print(p, file=sys.stderr)
        return EXIT_BAD_PACK

    lmstudio = LMStudio(base_url=hostinfo.read_host_info(Path(args.root))["apiBaseUrl"])
    cfg = RunConfig(
        modelKey=args.model, sessionId=args.session, referenceKey=args.reference,
        warmupExtra=args.warmup or 0,
        firstCallTimeoutSeconds=args.first_call_timeout or 300.0,
        requestTimeoutSeconds=args.request_timeout or 120.0,
    )
    try:
        run, disclosures = run_pack(pack, cfg, lmstudio=lmstudio, root=Path(args.root))
    except RunRefused as exc:
        print(f"model-bench: {exc}", file=sys.stderr)
        return exc.exitCode

    path = store(run, Path(args.root))     # raises InvalidFingerprint — a runner defect, not
                                            # handled here per §3.4.5 point 1 (no bypass anywhere)
    print(f"stored: {path}")

    if disclosures:
        print("PACK DISPATCH FAILURES")    # dispatch-failure note §4(d)'s funnel-head block
        for d in disclosures:
            print(f"  {d.scriptId} turn {d.turn}: {d.tool} — {d.reason}")
        return EXIT_BAD_PACK               # exit 4, AFTER store() has already written the record
                                            # (dispatch-failure note §4(d): "the record is written,
                                            # then run exits 4")
    return EXIT_OK
```

The ordering — `store()` before the exit-`4` check — is the dispatch-failure note's own load-bearing
clause (§4(d): *"a data-quality finding discovered only once a partial record exists, not an
aborted run"*) and plan §3.6a's amendment (`:1933-1938`): `results/runs/<runId>.json` and the
disclosure both land before the process reports `4`.

### 6.3 Exit codes — the closed set, as amended

Reusing `cli.py`'s already-defined constants (§2.1); `EXIT_BAD_PACK` (4) now covers two distinct
situations at two different points in the same command, both cited:

| Code | Constant | When (source) |
|---|---|---|
| `0` | `EXIT_OK` | Tool ran and reported, whatever the scores (§3.6a) |
| `2` | `EXIT_USAGE` | Bad arguments/usage |
| `3` | `EXIT_LMSTUDIO_UNREACHABLE` | `probe()` returns `v1-only`/`unreachable` (two distinct messages); warm-up does not answer within `--first-call-timeout`; re-probe after a scored-call timeout shows the server gone (§3.6 clause (iv)) |
| `4` | `EXIT_BAD_PACK` | `validate` failure; pack load error; unmet `environment.requires`; a `callSurface`/catalog-`type` contradiction; **or, after `run`'s artifacts are already written**, a `tool-caller` conversation censored by a dispatch raise (§3.6a `:1933-1938`, dispatch-failure note §4(d)) |
| `5` | `EXIT_FINGERPRINT` | `host.json` absent/schema-invalid (step 1); attestation trip-wire stale (step 6); `residentModelsAtEnd` unobtainable after three retries (step 10) |

`compare` is unaffected — it still exits `0` on every stored record being invalid, per its own
already-shipped behaviour.

## 7. Step sequence for implementation

Three chunks, each independently dispatchable, sequenced so the tree stays buildable and each
chunk's tests are a superset of the previous chunk's fixtures rather than a rewrite of them.

### Step 0 — `results.py`: `LatencyBlock`/`CallTiming`/`ItemTiming`, and the `latencyMs` conversion

Closes §2.2 item 1. Add `CallTiming`, `ItemTiming`, `LatencyBlock` (§5's shapes) to `results.py`;
add `ItemResult.timing: ItemTiming | None` (replacing the constructor's `latencyMs` parameter);
convert `latencyMs` to the `@property` in §5; add `RunResult.latency: LatencyBlock | None` and
`RunResult.attestationTripWire: Literal["compared","first-observation","unavailable"] | None`
(required, no default, `None` iff `armKind == "deterministic"` — plan `:3145-3147`); update
`to_dict`/`from_dict` on both (`to_dict` still emits `latencyMs` for a stored record's readability;
`from_dict` ignores it and re-derives — plan `:3093-3094`). Update every existing
`ItemResult(..., latencyMs=X, ...)` call site (`tests/conftest.py`, `tests/test_results.py`,
`tests/test_report.py`, `tests/test_cli.py` — enumerated in §2.2) to `timing=...`. This is a
self-contained, mechanically-checkable chunk: `grep -rn 'latencyMs=' model-bench/modelbench
model-bench/tests` before/after brackets it, and every existing S1 test must still pass unmodified
in its *assertions*, only its *fixtures'* construction changes.

`convo.py`: extend `ToolDispatchFailed.__init__` to accept `completedTurns: tuple[TurnTrace, ...]`
and `parsedArguments: Mapping[str, Any] | None` alongside the existing `toolName`/`turnIndex`
(§2.2 item 2 — this document's required, concretely-specified extension, since the note names the
payload but not the exact keyword shape); update the one raise site in `_drive_turn` to pass
`observed` (already in scope there, the turns completed before this one) and the parsed arguments;
rewrite the class docstring's "deliberately not decided" paragraph to state the ruling
(dispatch-failure note §4(b)-(c)) instead.

### Step 1 — `runner.py`'s core: capture order + `LatencyBlock` accumulation + both driving loops, offline only

The bulk of §4/§5 above. Ship `RunConfig`, `RunRefused`, `run_pack`, `_drive_single_call_items`,
`_drive_conversations`, `_turn_timings`, `_load_withheld_for`, `_gap_ms`, `_gap_withheld_for`,
`latency_block`. **Before writing `_gap_withheld_for`'s threshold, read
`docs/plans/small-model-benchmarking-ml.md` §11.5.1 directly** (§4's flagged gap) — do not use the
3 485.6 ms figure from test 15b's fixtures as the threshold; that number is only a fixture value
known to exceed it.

The scorer seam (§3.2) ships as the `ItemScorer`/`ConversationScorer` protocols only — no concrete
scorer exists yet, so `_load_item_scorer`/`_load_conversation_scorer` may raise
`NotImplementedError` for every role until S3 lands the first one; this is expected and matches
S2's own "Done when", which never runs a real pack.

Tests: §5's item 15b **drives this step** (plan `:6317`: *"`15b` drives it — it is offline, it is
the acceptance surface for the runner's timing design, and it is written first"*) — write 15b's
fixtures (the fifteen-odd cases enumerated at `docs/plans/small-model-benchmarking.md:6177-6274`)
before the accumulation code, red→green. Then the v1.31 `ToolEnvironment`-per-conversation tests
(`:5646-5652`): twelve distinct instances across a twelve-conversation fixture pack, and an empty
cart in conversation `k+1` after conversation `k` placed an order. Then test 10c's four disposition
cases plus the precedence case (already covered by `convo.drive` itself, per §2.1 — this step's own
tests are only that the runner catches and stores correctly around it). Then the dispatch-failure
note's E1–E5 evaluation designs (§5 of that note) — E2/E3/E4 exercise `_drive_conversations`'
censoring directly; E1 is `tools/sim.py`'s (S5, out of scope) and E5 is `compare`'s (already
shipped, exercise only its read side here). This step needs no real LM Studio, no real pack, and
no scorer — every fixture is a stub `LMStudio`/`ToolEnvironment` per the plan's own established
pattern (§4 S2 `:5647`: "this stage's own pattern, no real pack required").

### Step 2 — CLI `validate`/`run` wiring, exit codes, capture-order refusal messages

Ships §6's `_cmd_validate`/`_cmd_run`, wires them into `_build_parser` (already has `attest`'s
pattern to follow), updates `cli.py`'s module docstring's exit-`4` line (§2.1). Depends on step 1
being complete since `run` calls `run_pack`. Tests: capture-order refusals 13/14/15 (`-m live`,
adapter-level, already partly covered by existing `-m live` `lmstudio` tests — this step adds the
`run`-level wiring around them: exit 3 on each of `probe()`'s two negative outcomes, exit 4 on the
`callSurface` cross-check and the tool-calling gate, exit 5 on `host.json` absent/stale); the
`--strict` open question (§8) should be resolved (or explicitly deferred with a `NotImplementedError`
and a `TODO` citing this document) before this step closes, not silently ignored.

## 8. Test strategy

This is an implementation-plan test *sequence*, not a new test list — every numbered item below is
the plan's own (`docs/plans/small-model-benchmarking.md`'s §5), owed to S2's row of the stage table
(`4, 10, 10b, 10c, 12, 12b, 13, 14, 15, 15b`, `:5978`). Drive them in this order, per stage
guidance (`:6314-6321`: *"the red→green sequence is per stage... `15b` drives it"*):

1. **15b** (offline, drives step 1) — the runner's timing discipline: warm-up-once-per-arm,
   two-budget completion, `coldLoadSeconds` presence/absence, residency-withholding-while-keeping-
   siblings, `unexplainedMs` withholding, timeout scores `fail`/no-response scores `unrunnable` with
   no timing figure either way, the co-presence exclusion, a `stats`-less `ChatResult` never
   raising, `censoringExact`'s four branches, a clean cold run withholding nothing, the six
   multi-call cases (a)-(f) — full case list at `docs/plans/small-model-benchmarking.md:6177-6274`.
2. **v1.31's two `ToolEnvironment`-per-conversation tests** (§4 S2 `:5646-5652`) — twelve distinct
   instances, empty cart after an order.
3. **10, 10b, 10c** — already exercised by `convo.py`'s own shipped test suite; this step's job is
   only to confirm the runner's `_drive_conversations` reads `TurnTrace`/`ConversationTrace`
   correctly around them, not to re-test `drive()` itself.
4. **The dispatch-failure note's E1-E5** (`docs/plans/small-model-benchmarking-ml-dispatch-failure.md`
   §5) — E2 (censoring wired), E3 (headline vs. hazard discrimination, shared machinery with test
   10c/§4 S5's item (3a)), E4 (negative control: censoring ≠ scored failure — the same fixture with
   a normal scored failure at the same turn must render differently), E5 (disclosure survives
   storage, exercised through `store()`+`compare` reading).
5. **12, 12b** — pack integrity (already `packs.py`'s, step 2 only wires `validate` around it —
   this document does not re-derive `validate_pack`'s own test suite) and the determinism probe's
   `basis` wiring: four cases (probe ran+identical → by-construction; probe ran+differed → assumed;
   probe did not run → assumed; `replicatesPerScript > 1` → assumed regardless), asserted against a
   stub `ConversationScorer` returning a canned `determinismProbe` dict — the outcome-vector
   comparison itself is S5's (§3.2 above) and is not tested here.
6. **13, 14, 15** — `-m live` adapter tests, largely already shipped at the `lmstudio.py` level;
   step 2 adds only the `run`-level exit-code assertions around a live LM Studio.

Edge cases worth calling out because they are exactly where the plan's own review history found
defects (§2.3's citations trace each to a specific plan-gate finding): a turn that raises on its
third iteration (`callCount == 2`, `callAttemptedCount == 3`, P15-2); a residency-guard-then-
disposition collision on a single-call role (P16-1's `guard-judge` fixture, test 15b case (f)); a
`cap-hit` turn that must not be folded into "incomplete" (it is timed and complete); a dispatch
raise at `t = 5` with `H = 4` (in the headline's denominator, out of the hazard — the discriminating
pair §4 S5 *Done when* item (3a) gates).

## 9. Risks & open questions

- **`--strict`'s semantics are not stated anywhere in the plan.** §3.6a's CLI table names the flag
  with no elaboration and no other section mentions it. This is a genuine gap, not merely
  scattered — flagged rather than guessed, per this task's own instruction. **Recommendation:**
  route to `architect`/`tico` for a one-line ruling before step 2 closes (candidates: promote a
  `validate_pack` warning class that does not yet exist to a failure — but `validate_pack` today
  has no warning tier, only pass/fail — or make it a no-op pending a future warning tier). Low
  blast radius either way: it is additive to an already-closed function signature.
- **`packs.py`'s `validate_pack` does not yet implement route (iii) of the `sampling` contract**
  (§2.1) — CLI `validate` will therefore under-enforce relative to §4 S2's own stated done-condition
  until that lands. Not this document's fix; flagged so the coder dispatched on step 2 does not
  read a green `validate` test suite as proof the plan's S2 done-conditions are fully met.
- **The `unexplainedMs` withholding threshold's numeric value was not verified in this session**
  (§4, `_gap_withheld_for`'s docstring). The coder must read `docs/plans/small-model-benchmarking-
  ml.md` §11.5.1 directly before implementing it; do not infer it from test 15b's 3 485.6 ms
  fixture value, which is only known to exceed it.
- **The scorer seam (§3.2) is this document's own design, not the plan's.** It is a reasonable,
  minimal seam given everything the plan does fix about scorer responsibilities (`ItemResult`
  shape, the per-item/per-run split, `ToolCallAggregates.determinismProbe`'s shape), but it has
  never been reviewed. **Recommendation:** when S3 (the embedder pack, the first stage to build a
  real `ItemScorer`) is dispatched, its brief should treat this seam as a proposal to confirm or
  revise, not as settled fact the way §4/§5 above are.
- **`pack.iter_items()`/`pack.iter_scripts()`/`pack.find_script()` are named here for readability
  but are not shipped `Pack` methods and are not specified by the plan either.** Whatever S2's
  actual implementer builds them as, they are pure data-iteration helpers with no cross-cutting
  design stakes — low risk, but named so nobody mistakes them for already-existing API.
- **`design_effect(...)`'s exact call signature for the tool-caller path is cited, not filled in**
  (§4, `_drive_conversations`) — it is `-ml`'s function (listed among the `stats.py` functions
  §4 S1 names without a full signature, plan `:3195`) and this document did not open the `-ml` note
  to confirm its parameters. The coder implementing step 1's tool-caller branch needs that
  signature from `docs/plans/small-model-benchmarking-ml.md` directly.

## Ready to implement

Document: `docs/plans/small-model-benchmarking-runner-spec.md` (this file). Three dispatchable
steps (§7): **step 0** (`results.py`'s timing types + the `latencyMs`/`ToolDispatchFailed` shipped-
code edits, §2.2), **step 1** (`runner.py`'s core — capture order, both driving loops, `LatencyBlock`
accumulation, offline-only, driven by test 15b), **step 2** (CLI `validate`/`run` wiring + exit
codes). Two items need a human/specialist call before step 2 closes: `--strict`'s semantics (§9)
and confirmation of the scorer seam at S3 (§3.2, §9). One item needs the coder to consult
`docs/plans/small-model-benchmarking-ml.md` §11.5.1 directly rather than trust a number in this
document (§4, §9).
