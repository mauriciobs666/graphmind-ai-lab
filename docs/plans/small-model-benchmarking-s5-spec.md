# `model-bench` S5 — `tool-caller` pack, part 1: environment and scoring — implementation spec

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `docs/plans/small-model-benchmarking.md` (S5)

## 1. Goal & scope

Build the pieces `docs/plans/small-model-benchmarking.md`'s "### S5 — `tool-caller` pack, part 1:
environment and scoring" section requires **before** S6 authors the real 12 conversation scripts:
`packs/tool-caller-shop-assistant/{pack.json,catalog.json,tools/sim.py,tools/schemas.json,
prompts/system.md}` and `modelbench/scoring/toolcalls.py`, proved entirely by hand-built synthetic
`ConversationTrace`s — no live LM Studio call, no real script. **Out of scope**: `conversations.jsonl`
and its `PROVENANCE.md` (S6's own `Create:` line), the ~20 labelled prose-detector replies (S6 step 2,
plan `:5891`), any live end-to-end `run` of this pack (S6's own done-condition, plan `:5901`), and the
`validate --strict` pass on the real pack (S6's, since `conversations.jsonl` does not exist yet — see
§2.6).

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(confirmed live by `teco`'s brief); this is a code-level task in a component with no CPG, so
"considered, not relevant" applies rather than "not applicable."

## 2. Context & findings

### 2.1 What already exists and needs no S5 work — confirmed by reading the shipped code directly

- **`modelbench/convo.py` already implements the full dispatch-raise mechanism.** `ToolDispatchFailed`
  (`convo.py:325-372`) carries `toolName`, `turnIndex`, `completedTurns`, `parsedArguments`; `_drive_turn`
  raises it at the `env.dispatch` call site (`convo.py:756-768`) and lets it propagate; `drive`'s own
  docstring states the ruling in full. `TURN_DISPOSITIONS` is already the **five**-member set
  (`replied`, `cap-hit`, `timed-out`, `no-response`, `server-rejected`) — the `timed-out`/`no-response`
  split `-ml` §4.3.1 item 2 asked for already shipped.
- **`modelbench/runner.py`'s `_drive_conversations` already wires E2's censoring end to end.** It builds
  one fresh `ToolEnvironment` per conversation (`runner.py:485`, `512` — the dispatch-failure note §7's
  named gap is already closed), catches `ToolDispatchFailed`, stores a `ConversationTrace` truncated to
  `exc.completedTurns` (`runner.py:488-494`), and appends a `DispatchFailureDisclosure`
  (`scriptId, turn, tool, reason`). `tests/test_runner.py:1442-1500` already exercises this (asserts
  exactly 3 `TurnTrace`s on a raise at `t=4`, and a second fixture proving a raise on one script does
  not affect a sibling script's disclosure count) — **this is E2's censoring-is-wired half**, built
  ahead of S5 and needing no further runner-side work.
- **`modelbench/cli.py` already implements E5's ordering and disclosure half.** `_cmd_run` stores the
  record, *then* prints `PACK DISPATCH FAILURES` and returns exit `4` when `disclosures` is non-empty
  (`cli.py:434-436`); `tests/test_cli.py:1010-1039` pins the load-bearing ordering claim (the record is
  on disk before exit `4`). **What is not built**: the per-arm dispatch-failure *count surviving into
  the stored record* so a later, separate `compare` invocation can re-read it (§2.3 below) — `run`'s own
  disclosures are an in-process return value, never persisted.
- **`ConversationScorer` (`runner.py:212-222`) and the scorer-loading seam are ready for the
  conversation branch**, and the `itemscorer-extension.md` review's finding 2 (the
  `prompt.systemPrompt`/`toolSchemas` path-to-content resolution) is fixed at `Pack.prompt_config()`
  (`packs.py:341-436`), which is the **one** method both `_drive_single_call_items` and
  `_drive_conversations` call (`runner.py:352`, `467`) — so the fix already covers the conversation
  branch too. Finding 1 (`build_messages`) is chat-branch-only and has no `ConversationScorer`
  analogue: `convo.assemble`/`drive` already build the tool-caller's messages generically from
  `PromptConfig`, and no per-item scorer hook is missing there. **Nothing further is owed to the
  `ConversationScorer` side of the seam beyond what S4 Step 0 already resolved.**
- **`_load_conversation_scorer` (`runner.py:245-250`) still raises `NotImplementedError`
  unconditionally** — confirmed by reading it directly, exactly as the brief states. This is S5's own
  seam to fill (§4, Step 6 below).
- **`report.py` has zero tool-caller-specific rendering today** (confirmed:
  `grep -n "funnel\|ToolCallAggregates\|cleanThroughTurn\|hazard\|perTurnPosition\|restraint"
  modelbench/report.py` returns one hit, an unrelated comment). Everything the plan's *Done when*
  clause asks of `report.py` — the funnel table, the per-turn-position table, the hazard curve — is
  net-new S5 work (§4.4 below).
- **Baseline, run this session**: `.venv/bin/python -m pytest -q` → `1450 passed, 1 failed, 3
  deselected`; `.venv/bin/ruff check .` → clean. The one failure is the tripwire named next.

### 2.2 The tripwire is confirmed already red — this stage's own Step 0

`tests/test_convo.py::test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5` is red right
now, reproduced this session:

```
AssertionError: /home/.../model-bench/modelbench/scoring now exists: wire S5's branch set into
this file as the third leg of §4 S2's disposition probe, then delete this tripwire.
```

`modelbench/scoring/` exists because S4 built `classification.py`/`extraction.py`. The plan's own S5
prose names this as this stage's **own first, unavoidable finding**, not a surprise, and its own item
(1) of the *Done when* list requires resolving it in the **same change** that writes the third leg
(`ITERATION_SUMMARY_DISPOSITIONS | ITERATION_SUMMARY_EXCLUDED == convo.TURN_DISPOSITIONS`, item (4)) —
so Step 0 below builds both at once, mirroring S4's own Step 0 seam-fix precedent (`s4-spec.md` §7
Step 0: a small, offline, backward-compatible precursor before the pack's own body).

### 2.3 A real gap: the per-arm dispatch-failure count does not survive storage (E5)

E5 requires: *"a stored run carrying at least one dispatch failure, re-read by `compare`, prints the
per-arm dispatch-failure count."* Tracing this directly: `RunResult` (`results.py:600-686`) has no
field for it, `run_pack` returns `disclosures` as a **separate, in-process** tuple never merged into
the `RunResult` it stores (`runner.py:906-920`), and `RunResult.to_dict()`/`from_dict()` do not
mention it. `compare` only ever reads `results/runs/*.json` files written by a *prior, separate*
invocation of `run` — it never sees the in-memory `disclosures` tuple `_cmd_run` printed at the time.
**So today, nothing a `compare` run reads back carries the per-arm dispatch-failure count** — this is
a genuine plan-vs-shipped-code gap, not something already covered by E2's runner-level wiring.

**Resolution, and it needs no new top-level `RunResult` field.** A conversation censored by
`ToolDispatchFailed` is exactly one whose stored `ConversationTrace` is shorter than its script
(`len(trace.turns) < len(script.turns)`) — the only mechanism that produces a truncated trace, since
every other disposition still records one `TurnTrace` per scripted turn (`drive`'s per-turn loop never
skips a turn, `-ml` §4.1). So `ConversationScorer.score_conversations` can count these directly from
its own `scored` argument while building `ToolCallAggregates`, and store the count as
`FunnelCounts.unrunnableToolChannel` (§3.1) — the same integer the funnel table's own
"`unrunnable (tool channel)`" line prints (rule 3's own second `unrunnable` source, item E2). `compare`
then reads it straight off the stored, re-loaded `RunResult.aggregates` — no new field, no duplicate
count, one number serving both the funnel table and E5's disclosure line (§4.4).

### 2.4 A real design gap: `ToolCallAggregates`'s shape cannot carry what the note requires — resolved below

Read `results.py:511-529` directly. `ToolCallAggregates` today has `cleanThroughTurn`, `perTurnPosition`,
`funnel`, `restraint`, `hazard: tuple[BinaryMetric, ...]` — and **no `determinismProbe` field**, even
though the plan states its exact shape verbatim: *"Carried in the record:
`ToolCallAggregates.determinismProbe = {scriptIds, ran: bool, identical: bool, differingTurns: [...]}`"*
(plan `:2262-2264`). Confirmed this is not merely undocumented: `runner._drive_conversations` already
reads `getattr(aggregates, "determinismProbe", None)` (`runner.py:525`) and `tests/test_runner.py`'s
`FakeConversationScorer` (`test_runner.py:1213-1248`) fakes this attribute on a **bespoke stand-in
class**, never on the real `ToolCallAggregates` — because the real class does not have the field yet.
This is S5's own addition to make (§3.1).

Separately, `-ml` §4.3.1 item 11 rules *"what the scorer stores per position: three integers, `f_t`,
`r_t`, `c_t` — never a rate. The report divides."* `BinaryMetric(successes, n, unit)` can carry `f_t`
(as `successes`) and `r_t` (as `n`), but has no field for `c_t` (the censored-count). `hazard:
tuple[BinaryMetric, ...]` as shipped cannot represent this without either dropping `c_t` (violating the
note's own rule) or overloading `BinaryMetric.successes`/`.n` to mean something they do not mean
elsewhere in this codebase. §3.1 below adds a `HazardPoint` type for this, following the same pattern
`TurnPositionRate` already established for the per-position table (`results.py:143-152`).

### 2.5 A real design gap: `named_metrics()` currently pulls `funnel`/`hazard` into the generic Arms table, which is wrong for the funnel and inert-but-wrong for the hazard

`ToolCallAggregates.named_metrics()` (`results.py:522-528`) returns `[*found, *self.funnel,
*self.hazard]` — meaning every entry of `funnel` and `hazard` would render as a flat row in `report.py`'s
generic "## Arms" table (`report.py:764-808`), which prints a bare `successes/n` rate with a suppressed
interval for a pooled unit. That is the right generic treatment for the FR-8 (a)-(g) **rate** metrics
(§4.2's `native`/`rightToolChosen`/`allArgsCorrect`/etc. — each a real `k/n` over turns or calls, exactly
what the existing pooled-unit branch at `report.py:784-790` already renders correctly with no new code).
It is the **wrong** treatment for the funnel-head's own **structural** counts (*"turns driven 360"* is
not a rate — there is no meaningful `k` distinct from `n`) and for `hazard` (rule 5's own text: *"the
scorer stores `f_t`, r_t, c_t... never a rate"* — the whole reason the note insists on this is that a
naive `successes/n` reading is exactly the rate a reader is told not to compute from these numbers
directly, since censoring makes the risk set shrink at every position). **Resolution (§3.1, §4.4)**:
split the funnel-head's structural counts into a new `FunnelCounts` field (excluded from
`named_metrics()`), keep `funnel: tuple[BinaryMetric, ...]` for the FR-8 rate metrics only (unchanged
generic treatment), and change `hazard` to `tuple[HazardPoint, ...]` (excluded from `named_metrics()`,
same treatment `perTurnPosition` already correctly gets today). This mirrors `perTurnPosition`'s own
precedent exactly — it was already, correctly, left out of `named_metrics()` because it needs its own
per-position table, not a flat row; `funnel`'s head and `hazard` needed the same exclusion and did not
get it, because nothing had tried to build their renderer yet.

### 2.6 A real design gap: how `ItemResult`s must be shaped for `cleanThroughTurnH` to pair correctly — traced through the shipped machinery, not guessed

This is the least obvious and most load-bearing finding in this document, so it is traced in full.

The plan states `sampling.pairingKey` for the tool-caller is the **3**-component
`["scriptId", "replicate", "turnIndex"]` (plan `:456`, `:691-693`), which reads, on its own, as "one
`ItemResult` per turn." But tracing `compare_report`'s actual pairing machinery directly shows that
cannot be the whole design for the pack's **one** verdict metric:

- `pack.metrics.verdictMetrics == ["cleanThroughTurnH"]` (plan `:2312`) — the *only* metric that ever
  reaches `_paired_rows`/`PairedOutcomes.from_units` (`report.py:830`: `family =
  list(pack.metrics.verdictMetrics)`).
- `_paired_rows(a, b, metric, pack)` (`report.py:140-189`) iterates **every** item in `a.items`
  unconditionally (`for item, unit_id in zip(a.items, a_units)`) — there is no per-metric
  pre-filtering of the item list. For each item it calls `item.scored_outcome(metric)`, and
  `scored_outcome` returns `None` uniformly whenever `metric` is **absent from `scoreable`** *or*
  **declared `False`** (`results.py:404-405`: `if not self.scoreable.get(metric, False): return
  None`) — the two are indistinguishable to this function.
- Every item for which either arm's outcome is `None` is folded into `asymmetry_a`/`asymmetry_b`/
  `unscoreable_both` (`report.py:166-173`) — there is no fourth "does not apply to this metric at all,
  do not even tally it" bucket.
- Consequence: **if the scorer emits one `ItemResult` per turn** (≈80 across 12 conversations) and only
  the turn that decides `cleanThroughTurnH` (say, turn `H-1`) declares it, the other ≈68 turn-records
  return `None` for `cleanThroughTurnH` on both arms and would inflate `unscoreable_both` by ≈68 —
  drowning the real signal (an early-`unrunnable` conversation, of which there are at most a handful)
  in a meaningless count that misrepresents `-ml` §4.3 rule 2's own "n/a tally, printed next to the
  rate" as 60+ conversations instead of the handful that are real. **If, instead, every turn of a
  conversation declares the identical `cleanThroughTurnH` value**, `unit_ids` (all `pairingKey[0]` =
  the same `scriptId`, repeated once per turn) hits `PairedOutcomes.__post_init__`'s duplicate guard
  and raises `DuplicateAnalysisUnit` on the very first comparison run (`stats.py:701-708`) — the
  backstop the note calls "a backstop, not the mechanism," firing exactly because a per-turn record
  *is* the cluster key here.
- Independently, `_aggregate_item_mismatches` (`report.py:291-378`, DC-10's cross-check) only inspects
  a `BinaryMetric` whose `.unit == unit_kind_for_role(pack.role)` (`"conversation"` for tool-caller,
  `roles.py:27`) **and** whose `.name in pack.metrics.verdictMetrics` — so it too only ever touches
  `cleanThroughTurnH`, confirming the FR-8 (a)-(g) rate metrics (`unit="turn"`/`"call"`) never need a
  per-item recount at all and are correctly, entirely, aggregate-only.

**Resolution, adopted below, marked explicitly as this document's own synthesis**: `items` (the
`tuple[ItemResult, ...]` half of `score_conversations`'s return) carries **exactly one `ItemResult`
per scored (non-probe) conversation** — 12 of them on the real pack, fewer only if a script list
itself were shorter (never fewer than the number of scripts driven; censoring changes what a
conversation's item *says*, never whether one exists). `pairingKey = (scriptId, replicate, H - 1)`
— the pack-declared `metrics.cleanThroughTurnH.H` minus one, i.e. the zero-based index of the last
turn the headline considers. This is not an arbitrary filler value for the manifest's declared
3-component shape: it is the one turn index `cleanThroughTurnH` is genuinely *about*, every
conversation shares the same `H`, so the value is constant across items and never collides.
`itemId = scriptId`. `scoreable = {"cleanThroughTurnH": True}` for a conversation that reached turn
`H` with nothing scoring `fail`/`unrunnable` at `t ≤ H` in the model-channel/turn-mechanism sense
(rule 4/§4.6), and `{"cleanThroughTurnH": False}` for rule 5's third state — a conversation carrying
an `unrunnable` turn (model-**or** tool-channel; §4.3 rule 4's discriminator and dispatch-failure note
§4(c)'s tool-channel row both route here) at any `t ≤ H`. `counts = {"cleanThroughTurnH": 1}` when
`True` and clean, `{"cleanThroughTurnH": 0}` when `True` and it failed by turn `H` without being
`unrunnable`. `outcome` is set to `"pass"`/`"fail"` for the two ordinary cases and `"unrunnable"` for
the third state — the same informal `Outcome` extension already in live use elsewhere in this codebase
(`retrieval.py`, per `s4-spec.md` §9's own risk note that `Outcome`'s `Literal` is already narrower
than shipped usage; this document does not re-open that type, consistent with that precedent).

This design is verified against every consumer traced above: `_paired_rows` sees exactly one row per
`scriptId`, no duplicate, no phantom exclusion; `_aggregate_item_mismatches`'s recount of "items for
which `scored_outcome("cleanThroughTurnH")` is not `None`" equals the run's own declared `n` by
construction, because both are derived from the same 12-item pass in `score_conversations`; rule 2's
n/a tally (rendered by the existing, generic `_pairing_tally`, `report.py:381-394` — confirmed no new
code needed here) reports exactly the conversations rule 5 actually excludes, nothing else. **No other
metric needs an `ItemResult` at all** — `restraint` and the FR-8 (a)-(g) rates are pooled, non-verdict,
computed and stored directly on `ToolCallAggregates.funnel`/`.restraint`, and read generically by the
existing Arms table with no per-item backing (§2.5).

### 2.7 The prose-detector calibration corpus is S6's, not S5's — confirmed against the plan text directly

`-ml` §4.2(a)+(b) requires the pack to ship "~20 labelled replies" and the report to print the
detector's own precision/recall. Plan `:5891` assigns *drafting* those 20 replies to **S6** ("Also draft
the ~20 labelled replies the prose-vs-native detector is scored against"), alongside the real
conversation scripts — both are human-verified content authored against the real storefront. **S5 owns
the detector *machinery* and the precision/recall computation**, not the real calibration data: the
scorer's `detect_prose_pseudo_call(reply_text) -> bool` heuristic and its
`prose_detector_precision_recall(labelled_pairs) -> tuple[float, float]` companion are pure functions
S5 unit-tests against small, hand-built fixtures of its own; the real corpus, once S6 authors it, is
read through a pack `data.*` path this document declares (`data.prosePseudoCallCalibration`, §3.2) but
does not populate.

## 3. Design & rationale

### 3.1 `results.py` — three additive, backward-compatible extensions to `ToolCallAggregates`

```python
@dataclass(frozen=True)
class FunnelCounts:
    """§4.3 rule 3's funnel-head, one integer per line of the illustrated table, in order. Never a
    rate — `report.py`'s dedicated funnel renderer computes a "-> k/n" annotation for the lines that
    have one (restraint, the (a)+(b) partition) by cross-referencing `restraint`/`funnel`, never by
    storing a second, derivable copy here (§7 rule 4)."""
    turnsDriven: int
    unrunnableModelChannel: int          # D(t) in {no-response, server-rejected}
    unrunnableToolChannel: int           # a censored-by-ToolDispatchFailed conversation count
    turnsScoredAfterUnrunnable: int      # rule 5's disclosure line (3a-ii); 0 by construction for
                                          # the tool channel (dispatch-failure note §4(b))
    restraintTurns: int                  # R(t) = 0
    requiredCallTurns: int               # R(t) >= 1
    nativeCallEmitted: int
    prosePseudoCall: int
    noAttempt: int
    turnsWithAnyCall: int                # |E(t)| >= 1 — denominator for (c), (e), (f)
    dispatchedCalls: int                 # denominator for (d); calls, not turns
    factBearingReturns: int              # denominator for (g)
    unscoreableReturns: int


@dataclass(frozen=True)
class HazardPoint:
    """`-ml` §4.3.1 item 11's per-position hazard triple, stored as three integers per position —
    never a rate (rule 5). `metric.successes = f_t` (failures at this position), `metric.n = r_t`
    (risk-set size), `metric.unit = "conversation"`; `censored = c_t`, the count newly censored at
    this exact position (not cumulative). Mirrors `TurnPositionRate`'s own shape (`results.py:143`)
    one field further."""
    turnIndex: int
    metric: BinaryMetric
    censored: int


@dataclass(frozen=True)
class ToolCallAggregates:
    kind: Literal["toolcalls"] = "toolcalls"
    cleanThroughTurn: BinaryMetric | None = None
    perTurnPosition: tuple[TurnPositionRate, ...] = ()
    funnel: tuple[BinaryMetric, ...] = ()          # FR-8 (a)-(g) RATE metrics only (§2.5) — unchanged type
    funnelCounts: FunnelCounts | None = None       # NEW — the funnel-head's own structural counts
    restraint: BinaryMetric | None = None
    hazard: tuple[HazardPoint, ...] = ()           # CHANGED from tuple[BinaryMetric, ...] (§2.4)
    determinismProbe: Mapping[str, Any] | None = None  # NEW — plan `:2262-2264`'s exact shape

    def named_metrics(self) -> tuple[MetricValue, ...]:
        found = []
        if self.cleanThroughTurn is not None:
            found.append(self.cleanThroughTurn)
        if self.restraint is not None:
            found.append(self.restraint)
        return tuple([*found, *self.funnel])   # CHANGED — hazard/funnelCounts/perTurnPosition all
                                                # excluded; each gets its own report.py renderer (§2.5)
```

`hazard`'s type change is safe: nothing ships a populated `ToolCallAggregates.hazard` today (confirmed
§2.1 — no scorer exists yet), so there is no round-tripping consumer to break; `_encode`/`_decode`
(`results.py:791-811`) gain one case each, following `TurnPositionRate`'s existing pattern exactly —
`HazardPoint`'s dict carries a `"censored"` key `TurnPositionRate`'s never does, so `_decode` checks for
it **before** the existing `"turnIndex" in value` branch (order matters; both dicts carry `turnIndex`):

```python
def _encode(value: Any) -> Any:
    if isinstance(value, (BinaryMetric, ContinuousMetric, DistributionSummary)):
        return _metric_to_dict(value)
    if isinstance(value, HazardPoint):
        return {"turnIndex": value.turnIndex, "metric": _metric_to_dict(value.metric),
                "censored": value.censored}
    if isinstance(value, TurnPositionRate):
        return {"turnIndex": value.turnIndex, "metric": _metric_to_dict(value.metric)}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(v) for v in value)
    if isinstance(value, dict) and "type" in value:
        return _metric_from_dict(value)
    if isinstance(value, dict) and "censored" in value:          # NEW — check before TurnPositionRate
        return HazardPoint(turnIndex=value["turnIndex"], metric=_metric_from_dict(value["metric"]),
                            censored=value["censored"])
    if isinstance(value, dict) and "turnIndex" in value:
        return TurnPositionRate(turnIndex=value["turnIndex"], metric=_metric_from_dict(value["metric"]))
    return value
```

`_AGGREGATE_BY_KIND["toolcalls"]` needs no change (still `ToolCallAggregates`); `determinismProbe`
(a plain JSON-native `dict`/`None`) and `funnelCounts` (a small dataclass — give it the same
`_encode`/`_decode` treatment as `HazardPoint`, keyed on a distinguishing key such as `"turnsDriven"`)
round-trip through the same generic `_aggregates_to_dict`/`_aggregates_from_dict` (`vars(agg).items()`)
with no further change.

### 3.2 `packs/tool-caller-shop-assistant/pack.json`

```json
{
  "packId": "tool-caller-shop-assistant",
  "packVersion": "0.1.0",
  "role": "tool-caller",
  "schemaVersion": 1,
  "description": "Simulated storefront: catalog lookups, cart mutation, order placement, scripted multi-turn conversations.",
  "scorer": "toolcalls",
  "environment": {"requires": ["lmstudio-chat"]},
  "prompt": {
    "systemPrompt": "prompts/system.md",
    "toolSchemas": "tools/schemas.json",
    "representToolSchemasEachTurn": false,
    "historyReplay": "structured",
    "historyTurns": 0,
    "maxIterationsPerTurn": 8,
    "temperature": 0.0,
    "maxTokens": 512
  },
  "data": {
    "catalog": "catalog.json",
    "conversations": "conversations.jsonl",
    "prosePseudoCallCalibration": "prose_calibration.jsonl"
  },
  "tools": {"module": "tools/sim.py", "entrypoint": "build_environment", "schemas": "tools/schemas.json"},
  "sampling": {
    "seed": 20260913,
    "pairingKey": ["scriptId", "replicate", "turnIndex"],
    "analysisUnit": "scriptId",
    "scripts": 12,
    "replicatesPerScript": 1,
    "determinismProbeScripts": ["A-01", "B-01"]
  },
  "metrics": {
    "verdictMetrics": ["cleanThroughTurnH"],
    "headlineMetric": "cleanThroughTurnH",
    "cleanThroughTurnH": {"H": 4}
  },
  "provenance": "PROVENANCE.md"
}
```

**Two fields declared here are deliberately not yet backed by a file S5 creates**:
`data.conversations` (S6's) and `data.prosePseudoCallCalibration` (S6 step 2, §2.7) — this pack cannot
pass `load_pack`/`validate_pack` until S6 lands, which is expected and matches the plan's own
`Create:` line for this stage (no `conversations.jsonl`, no `--strict` done-condition here; both are
S6's). No test in this stage's own suite calls `load_pack` on this real directory — every scorer/sim
test below builds synthetic `Conversation`/`ConversationTrace` fixtures in Python directly, the same
pattern `test_convo.py` already uses throughout.

**`determinismProbeScripts` names two scripts that do not exist yet.** This is fine for the same
reason: nothing in S5 drives them for real. The field is declared now, per plan `:2242-2244`'s "which
two is pack data and cannot be chosen after seeing a result," so S6 inherits a decision rather than
making one under its own deadline pressure — flagged in §6 as an open item for confirmation, since the
concrete script ids (`A-01`, `B-01` above) are this document's own placeholder pending S6's actual
authored set.

**`tools.schemas` and `prompt.toolSchemas` deliberately point at the same file.** One canonical
`tools/schemas.json` is both what the model is told (via `PromptConfig.toolSchemas`, resolved to
content by `Pack.prompt_config()`) and what the scorer reads `boundaryRule` extensions from (§3.4) —
a second, hand-copied schema file is exactly the two-homes-for-one-fact shape `model-bench/AGENTS.md`
warns against.

### 3.3 `packs/tool-caller-shop-assistant/tools/schemas.json` and `tools/sim.py` — the storefront

Seven tools, transcribed from `falkorchat/tools.py` per plan `:2271-2279`: `lookup_product_fact`,
`filter_products`, `view_cart`, `add_to_cart`, `remove_from_cart`, `clear_cart`, `place_order`. Each
schema entry is an OpenAI-shaped function definition (`{"type":"function","function":{"name":...,
"parameters": {...JSON Schema...}}}`, matching `_tool_schema_message`'s expected shape in `convo.py`).

**`boundaryRule` is a per-parameter JSON Schema extension key**, per plan `:2299-2302`'s explicit
placement ("the per-argument boundary rule supplied by the pack's tool schemas") — not in
`conversations.jsonl`'s `expect` block. Shape (this document's own synthesis; no prior art in this
tree states one at this granularity):

```json
{"maxPrice": {"type": "number", "boundaryRule": {"confusedWith": [49, 5000], "inclusive": true}}}
```

`confusedWith` lists alternate values that, if produced by the model in place of the pack's declared
correct value, are `boundary_unit`-classified (a subset of `wrong_value`, never a sibling — §4.2(d))
rather than a generic wrong-value miss; `inclusive` documents (for a human reader, not consumed by the
scorer directly) which inclusive/exclusive reading the correct value assumes. `scoring/toolcalls.py`
reads this straight off the tool's own declared schema for the argument being checked — never a regex,
per the note's own rule.

`tools/sim.py`'s `build_environment()` returns an object satisfying `modelbench.tooling.ToolEnvironment`
structurally (no import beyond `modelbench.tooling` — `validate_pack`'s AST allowlist, `packs.py`). Per
conversation state: a cart (`{productId: quantity}`) and a list of placed orders, both empty at
construction — `_drive_conversations` already builds one fresh environment per script (§2.1), so no
cross-conversation leak is this module's concern to guard against a second time.

**The dispatch-totality contract is the load-bearing rule** (dispatch-failure note §4(a), plan's own
citation at `:2874` area / `tooling.py`'s own docstring): `dispatch(name, arguments)` **never raises**
on anything the model can produce. Every input-shaped problem — unknown tool name that still reached
`dispatch` (should not happen given `drive` only dispatches a name it parsed off the model's own
`tool_calls`, but the sim must not assume it), a missing/extra/wrong-typed argument, a product id not
in the catalog, a quantity of `0` or negative, removing an item not in the cart — is a **returned**
`{"error": "<code>", ...}` value, recorded as an ordinary `DispatchRecord` like any other call.
`raise` is reserved for a condition that does not depend on the model's arguments at all (the catalog
JSON failed to load, an internal invariant broke) — the one-line test from the note: *"if the model's
arguments can change whether it raises, it must not raise."* The sim does **not** schema-validate
before executing (note §4(a)'s corollary) — a wrong-typed argument still reaches the tool's own logic
and gets a data-driven `{"error": "wrong-type", ...}` or is coerced per the tool's own rule, never
rejected pre-dispatch, since rejection there would make the call *undispatchable* and destroy
FR-8(d)'s own measurement.

`catalog.json` is a small, versioned product list (`~10-15` products spanning a price range that
exercises `maxPrice`'s boundary case, at least one product whose stock/quantity fields exercise a
`0`/negative-quantity boundary) — hand-authored data, not derived from anything in this repo; content
only, no design decision beyond "small enough for the pack's `contentHash` and readable in a code
review."

### 3.4 `modelbench/scoring/toolcalls.py` — the scorer

One module (`ItemScorer`/`ConversationScorer`-shaped, structural, no base class — matching
`retrieval.py`'s/`classification.py`'s own precedent). Its public surface, each piece unit-testable in
isolation against hand-built `TurnTrace`/`ConversationTrace` fixtures, none needing a loaded pack:

- **`ITERATION_SUMMARY_DISPOSITIONS`/`ITERATION_SUMMARY_EXCLUDED`** (`-ml` §4.2(f), §4.3.1 item 8) —
  the two scoring constants, written out literally (never derived from each other or from
  `convo.TURN_DISPOSITIONS`), plus the module-level assertion binding their union to
  `convo.TURN_DISPOSITIONS` with disjointness (item (4)) — this is also the tripwire's own third leg
  (§4 Step 0).
- **`turn_disposition_scores(disposition: TurnDisposition, has_dispatch: bool) -> Literal["scored",
  "unrunnable"]`** (or an equivalent small pure function) — the single place `-ml` §4.3 rule 4's table
  is consulted; every other function below calls this rather than re-deriving the mapping (closing
  §4.3.1 item 1's own instruction: "the recommended fix is to delete the column, not split the row" —
  this module is rule 4's one home on the scorer side).
- **`emission_form(required: bool, dispatched_count: int, prose_detected: bool) -> Literal["native",
  "prose_pseudo_call", "no_attempt"]`** — the (a)+(b) three-way partition (§4.2).
- **`detect_prose_pseudo_call(reply_text: str) -> bool`** and
  **`prose_detector_precision_recall(labelled: Sequence[tuple[str, bool]]) -> tuple[float, float] |
  None`** — the heuristic and its calibration (§2.7); `None` when no labelled corpus is available
  (S5's own tests exercise this with a small hand-built list; the real ~20-reply corpus is S6's).
- **`right_tool_chosen`, `argument_correctness`, `spurious_and_duplicate`, `stopping_when_done`,
  `iteration_summary` (`I(t)` mean/p95 via `stats.percentile`), `reply_matches_tool`, `restraint`** —
  one function per §4.2 letter, each returning the numerator/denominator pair(s) it owns, each reading
  `boundaryRule` off the pack's own `tools/schemas.json` content for `argument_correctness`'s
  `boundary_unit` subset (never a regex).
- **`clean_through_turn(trace: ConversationTrace, *, h: int) -> Literal["clean", "failed", "n_a"]`** —
  rule 5's third state, computed over the pack-declared `H`; also the single place that decides an
  `ItemResult`'s `scoreable["cleanThroughTurnH"]`/`counts` (§2.6's design).
- **`hazard_points(traces: Sequence[ConversationTrace], *, h: int | None) -> tuple[HazardPoint, ...]`**
  — rule 5's censoring: risk set, `f_t`, `c_t`, computed position by position, `c_t == 0` leaving the
  point's own bound-eligibility flag unset (no bound rendered — a report-layer concern, §4.4) and never
  storing a rate.
- **`per_turn_position(traces, *, both_arms_available: ...)`** — not this module's to compute per-arm
  in isolation beyond producing `TurnPositionRate` tuples over one arm's own traces; §4.4 covers
  the observed-vs-structural `n` distinction at render time.
- **`outcome_vectors_differ(a: ConversationTrace, b: ConversationTrace) -> tuple[int, ...]`** — the
  determinism probe's own pure function (plan `:2239`, item 12b's "outcome-vector half," S5's Done-when
  item naming it explicitly). **Design decision, stated as this document's own** (the plan and `-ml`
  note name the requirement — *"report whether the outcome vector is identical"* — but not the
  comparand's exact composition): per turn, compare `(turnDisposition, finalReplyText, tuple((d.name,
  d.parsedArguments) for d in that turn's dispatch slice))` — disposition and reply text because they
  are what scoring rules 4/5 branch on, dispatched `(name, parsedArguments)` pairs because they are
  what FR-8(c)/(d) score, and deliberately **not** `wallClockMs` (always differs), not
  `DispatchRecord.timestamp`/`.returnValue` (derivable from a deterministic sim's `parsedArguments`,
  and comparing them adds no discriminating power over a truly non-deterministic model while adding
  false positives from clock-driven fields), not raw `messagesSent`/tool-call ids (an LM Studio
  request id or similar could differ trivially without any behavioural difference). Returns the tuple
  of turn indices where the two traces' per-turn tuples differ — `()` when identical, used directly as
  `differingTurns` in `ToolCallAggregates.determinismProbe`.
- **`score_conversations(scored, probes, *, pack) -> tuple[tuple[ItemResult, ...], Aggregates]`** — the
  `ConversationScorer` Protocol method, assembling everything above: builds the twelve per-conversation
  `ItemResult`s (§2.6), the `FunnelCounts` (including `unrunnableToolChannel` via the
  `len(trace.turns) < len(script.turns)` test, §2.3), the FR-8 rate `BinaryMetric`s in `funnel`, the
  `hazard`/`perTurnPosition` tuples, `restraint`, `cleanThroughTurn`, and `determinismProbe` (comparing
  each `probes` entry against its same-`scriptId` counterpart in `scored` via
  `outcome_vectors_differ`, setting `ran`/`identical`/`differingTurns` per plan `:2262-2264`'s shape).

### 3.5 `packs/tool-caller-shop-assistant/prompts/system.md`

Plain prose instructing the model it is a shop assistant with the given tools, written once, resolved
to content by `Pack.prompt_config()` (§2.1 — no new resolution code needed, the fix already ships).
Content-only; no design decision beyond keeping it short enough that `representToolSchemasEachTurn:
false` (§3.2) does not force it into every turn's token budget.

## 4. File/module layout

| File | New/changed | Owner in this stage |
|---|---|---|
| `modelbench/results.py` | changed | `FunnelCounts`, `HazardPoint`, `ToolCallAggregates.determinismProbe`/`funnelCounts`, `hazard` type change, `named_metrics()` fix, `_encode`/`_decode` cases (§3.1) |
| `modelbench/scoring/toolcalls.py` | new | the scorer (§3.4) |
| `modelbench/runner.py` | changed | `_load_conversation_scorer` resolves `modelbench.scoring.<name>` the way `_load_item_scorer` already does (one function, mirrored) |
| `modelbench/report.py` | changed | three new renderers + wiring (§4.4) |
| `packs/tool-caller-shop-assistant/pack.json` | new | §3.2 |
| `packs/tool-caller-shop-assistant/catalog.json` | new | §3.3 |
| `packs/tool-caller-shop-assistant/tools/schemas.json` | new | §3.3 |
| `packs/tool-caller-shop-assistant/tools/sim.py` | new | §3.3 |
| `packs/tool-caller-shop-assistant/prompts/system.md` | new | §3.5 |
| `tests/test_convo.py` | changed | tripwire deleted, third leg added (§4 Step 0) |
| `tests/test_results.py` | changed | round-trip tests for the three `ToolCallAggregates` additions |
| `tests/test_tools_sim.py` | new | E1 + ordinary sim behaviour |
| `tests/test_scoring_toolcalls.py` | new | the bulk of the scorer's own tests |
| `tests/test_report.py` | changed | funnel/per-position/hazard renderer tests |
| `tests/test_runner.py` | changed | `_load_conversation_scorer` resolution test, end-to-end synthetic integration (Step 6) |

### 4.4 `report.py` — the three new renderers, and where they sit

Inserted into `compare_report`, gated on `pack.role == "tool-caller"` (equivalently
`run.aggregates.kind == "toolcalls"`, checked per-run since `_comparison_pair` may not yet have run):

1. **`_render_funnel(run, arm_label) -> list[str]`** — one per arm, placed **before** the "## Arms"
   section (rule 3: *"the report opens with a funnel table, not a metric table"*), reading
   `run.aggregates.funnelCounts` for the raw hierarchy and cross-referencing `.restraint`/`.funnel`
   for the "-> k/n" annotations shown in the plan's own illustration (`:2098-2111`) — no duplicate
   storage of a derivable rate (§7 rule 4).
2. **`_render_per_turn_position(pack, runs) -> list[str]`** — §4.4's Wilson-per-position table, one
   column per arm, `n` = the **observed** count (`TurnPositionRate.metric.n`, already censoring-aware
   per §3.4's `per_turn_position`), the **structural** `n` (12/8/4 by position under the real pack)
   printed beside it, every position with observed `n < 10` marked `descriptive at this n — no
   significance claim` (§4.4's own literal text) — placed after "## Arms".
3. **`_render_hazard(runs) -> list[str]`** — both arms' curves **side by side**, each with its own
   `c_t` column, reading `HazardPoint.censored` directly (never a recomputed rate) — **and no
   cross-arm hazard difference string on any path** (item 3a-i): the renderer must not compute or
   print `a_rate - b_rate` for any position, because the two curves are conditioned on different,
   arm-specific risk sets after censoring (`-ml` §4.3 rule 5's own closing clause). Placed after the
   per-position table.
4. **The `I(t)` summary block and the `Y_calls/Y` vs. `I(t)`-mean distinction sentence** (§4.2(f),
   §11.7 slot 6's second sentence, §4.3.1 item 7) render as part of the existing latency block
   machinery's own per-pack extension point — `latency_block`'s `LatencyBlock` already carries
   `callAttemptedCount`-equivalent data (`callCount`, `latencyWithheldForNoResponse`); this document
   defers the exact wiring of that one sentence to Step 5 below rather than pre-designing it here,
   since it is a small, mechanical addition once `iteration_summary`'s mean/p95 exist (§3.4) — flagged
   in §6 as a low-risk item, not an open question.
5. **E5's per-arm dispatch-failure count** — read directly off
   `run.aggregates.funnelCounts.unrunnableToolChannel` and printed in the "## Arms" section's own
   funnel line (item 1 above already carries it; no separate line is needed).

## 5. Step sequence

Given the section's own size (E1-E5's five gated done-conditions, the ten named adversarial traces,
the two specifically-named discriminating pairs, three new report renderers, and a from-scratch
storefront), this is split into **seven** steps, each independently red-then-green and each buildable
and testable **entirely offline** except where flagged. No step in this stage requires a live LM
Studio call — the plan's own text states synthetic traces are "the only way to test it
deterministically," and S5's *Done when* list states no live-run done-condition (the first live
`tool-caller` run is S6's, per plan `:5901`/`:5931` area, item 16's per-pack obligation).

### Step 0 — Tripwire + `results.py` shape extensions (offline)

`FunnelCounts`, `HazardPoint`, `ToolCallAggregates.determinismProbe`/`funnelCounts`, the `hazard` type
change, the `named_metrics()` fix, `_encode`/`_decode` cases (§3.1) — round-trip tests in
`tests/test_results.py`. Then create `modelbench/scoring/toolcalls.py` with **only**
`ITERATION_SUMMARY_DISPOSITIONS`/`ITERATION_SUMMARY_EXCLUDED` and the cross-module union+disjointness
assertion (§3.4's first bullet, S5 Done-when items (1) and (4)); add the third leg to
`tests/test_convo.py` (`test_turn_dispositions_scoring_branch_set_is_exactly_the_plan_table` or
similar, asserting `ITERATION_SUMMARY_DISPOSITIONS | ITERATION_SUMMARY_EXCLUDED ==
convo.TURN_DISPOSITIONS` with disjointness) and delete
`test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5`. **Done when:** `pytest -q` is
`1451 passed, 0 failed, 3 deselected` (one net new test replacing the deleted tripwire), `ruff check .`
clean, and both mutation directions on the union/disjointness pin hold per `model-bench/AGENTS.md`'s
guard-reach convention (move a member between the two sets — reddens; add a member to neither —
reddens via the `TURN_DISPOSITIONS` side).

### Step 1 — `tools/sim.py`, `catalog.json`, `tools/schemas.json` (offline)

The storefront (§3.3). Tests, in order: (a) each tool's ordinary behaviour against a small in-memory
catalog fixture (add/remove/clear cart, place an order, filter/lookup); (b) **E1** — the property test
over every tool in `schemas()` crossed with the eleven-case adversarial argument set (absent key,
extra key, wrong type per declared field, `""`, `0`, `-1`, a large int, unicode, a nested object, `{}`)
— **zero raises, one `DispatchRecord` per call**, asserted as a totality claim (one raise fails the
whole test); (c) the `boundaryRule` extension round-trips off `tools/schemas.json` for at least one
`$`/cents-shaped and one inclusive/exclusive-bound-shaped parameter. **Done when:** every case above is
green and `tools/sim.py` imports nothing beyond stdlib + `modelbench.tooling` (confirmed by running
`validate_pack`'s AST-allowlist check directly against the file, even though the surrounding pack
cannot fully validate yet, §3.2).

### Step 2 — `pack.json`, `prompts/system.md` (offline, small)

§3.2, §3.5. No new test beyond `pack_ref_from_manifest`/`metrics_from_manifest` accepting the manifest
shape standalone (the S1-level manifest-only read, `packs.py:223-236`, which does **not** need
`conversations.jsonl` to exist) — confirms the `sampling`/`metrics` blocks are well-formed ahead of
S6's own full `validate_pack` pass.

### Step 3 — `scoring/toolcalls.py`'s per-turn pure functions (offline)

`turn_disposition_scores`, `emission_form`, `detect_prose_pseudo_call` +
`prose_detector_precision_recall`, `right_tool_chosen`, `argument_correctness` (incl. the
`boundaryRule`-driven `boundary_unit` subset), `spurious_and_duplicate`, `stopping_when_done` +
`iteration_summary`, `reply_matches_tool`, `restraint` — each unit-tested against hand-built
single-turn/short fixtures per §4.2's own denominator rules. Includes item (6)'s swept exactness rule
(over `X ≤ 200` and every `c ≤ X`, both directions) and item (7)'s `I(t)`-mean-vs-`Y_calls/Y` distinct
assertion. **Done when:** every FR-8 letter (a)-(g) plus restraint has at least one synthetic case that
moves its count and one that does not (the plan's own first *Done when* clause), tested in isolation
from any conversation-level assembly.

### Step 4 — the ten named adversarial traces + `clean_through_turn`/`hazard_points` (offline)

Builds the ten hand-built `ConversationTrace` fixtures the plan names in prose (no tool call, prose
pseudo-call, wrong tool, omitted required argument, mis-translated boundary, duplicated within-turn
call, cross-turn re-issue, kept going after done, contradicted the tool result, called a tool when
none required) plus the **two v1.27/v1.28-named discriminating pairs**: (1) a `cap-hit` turn with an
*empty* dispatch trace, asserting all five consequences at once (`no_attempt` partition,
`iteration_cap_hit_rate` entry, `I(t)` summary entry, absence from `stopping_when_done`'s denominator,
absence from (g)'s `unscoreable` bucket); (2) a `timed-out` turn beside an otherwise-identical
`no-response` turn, asserting `fail` against `unrunnable`. Then `clean_through_turn`'s third state
(a conversation with one `unrunnable` turn at `t ≤ H`) and `hazard_points`'s censoring (the 9-turn,
one-`unrunnable`-at-`t=2` fixture; the `H=4` discriminating pair at `t=5` vs `t=3`; both mutation
directions per `model-bench/AGENTS.md`; `c_t == 0` rendering the bound absent). Also
`outcome_vectors_differ` on an identical pair and a one-turn-differing pair (item 12b's own
requirement). **Done when:** every S5 Done-when item (1)-(7), (3a), (3a-i)'s data half, (3a-ii)'s data
half is covered by a named test (cited by the plan's own numbering in the test file's docstrings, per
`model-bench/AGENTS.md`'s guard-reach convention).

### Step 5 — `score_conversations` assembly + E2-E4 against the real scorer (offline)

Wires Steps 3-4's functions into the one-`ItemResult`-per-conversation design (§2.6), builds
`FunnelCounts` (including `unrunnableToolChannel`, §2.3), and the `determinismProbe` field. Tests:
**E2** re-run against the real scorer (not just the runner-level trace-truncation already tested in
`test_runner.py`, §2.1) — the synthetic 9-turn/raise-at-`t=4`/`H=4` fixture, asserting the funnel
prints **1** under `unrunnableToolChannel`, the headline denominator is **3** with 1 in its `n/a`
tally, the hazard risk set is 3 at `t≥4` with `c_4==1`. **E3** — the same fixture with the raise moved
to `t=5`, asserting it is **in** `cleanThroughTurn4`'s denominator and **out** of the hazard from
`t=5`. **E4** — the E2 fixture rendered beside one where turn 4 is an ordinary scored failure rather
than a raise, asserting the headline/hazard/per-position figures **differ**. **Done when:** all three
pass and the laundering test (a trace that collapses at turn 2 must not out-score one that reaches
turn 8 on any conditional count, plan item 7) is green.

### Step 6 — `report.py` renderers + `runner.py` wiring + E5 + full integration (offline)

`_load_conversation_scorer`'s real implementation (mirrors `_load_item_scorer`, resolving
`modelbench.scoring.<pack.manifest["scorer"]>`). The three renderers (§4.4). **E5** — a stored run
(hand-built `RunResult` with a populated `funnelCounts.unrunnableToolChannel > 0`) re-read by
`compare`, asserting the per-arm count prints. Then one **end-to-end synthetic integration test**:
a `FakePack`-shaped tool-caller pack (mirroring `test_runner.py`'s existing `_tool_caller_pack`
fixture) driven through `_drive_conversations` with the **real** `scoring.toolcalls` module in place
of `FakeConversationScorer`, over a small set of hand-built `Conversation` scripts and a stub LLM,
producing a `RunResult` that `store()`/`load_history()`/`compare_report()` accept and render without
error — the first point in this stage where every new piece runs together. **Done when:** the full
suite is green, `ruff check .` is clean, and the funnel table / per-turn-position table / hazard curve
all render on this synthetic integration run's own output (visually confirmed in the test, not merely
"did not raise").

## 6. Test strategy

Per the coordination's stage-ownership table (plan `:5980`), S5 owes numbered items **7**, **10c**
(third leg only), **12b** (outcome-vector half), and **16**; plus its own named done-conditions
(1)-(7), (3a)/(3a-i)/(3a-ii), and E1-E5 — done-conditions rather than numbered items, cited by name per
`model-bench/AGENTS.md`'s "gated by name, not referenced in prose" convention:

- **Item 7** (`scoring/toolcalls` — the synthetic-trace matrix, denominator edge cases, the laundering
  test) → Steps 3-5.
- **Item 10c, third leg** (the coverage probe's `TURN_DISPOSITIONS`-vs-scorer-branch-set binding) →
  Step 0.
- **Item 12b, outcome-vector half** → Step 4 (`outcome_vectors_differ`'s own unit tests), consumed by
  the `basis`-wiring tests already shipped in `test_runner.py` (§2.4, `determinism_probe` dict shape).
- **Item 16** (one full run per pack, end to end) — **not owed by S5.** The stage-ownership table
  states item 16 is "one arm of a per-pack obligation: each of S3-S7 owes the end-to-end run for the
  pack it builds" — for `tool-caller-shop-assistant` that run needs `conversations.jsonl` (S6's), so
  S6 owes it, not S5. Step 6's own synthetic integration test is this stage's *offline* substitute and
  does not discharge item 16.
- **E1-E5** (dispatch-failure note §5's evaluation table) → E1 in Step 1, E2 (censoring wiring, already
  partly tested at the runner level per §2.1) fully re-tested against the real scorer plus E3/E4 in
  Step 5, E5 in Step 6.
- **Everything else** (the ten named adversarial traces, `clean_through_turn`/`hazard_points`'s own
  correctness, the three report renderers, `FunnelCounts`/`HazardPoint`/`determinismProbe`'s
  round-trips) is new unit-test surface this stage introduces and owns outright, following S4's own
  precedent for un-numbered scorer-module tests (`s4-spec.md` §8).

**No acceptance-tier test is owed by this stage** — S5's own done-conditions are fully covered by the
items above plus this document's unit tests; the acceptance-tier known-answer validation (plan item 19)
is S6's, since it needs the real scripts.

## 7. Risks & open questions

- **`determinismProbeScripts`'s two placeholder script ids (`A-01`, `B-01`) are this document's own
  guess, not a decision.** The plan requires "one shape-A and one shape-B script" but the concrete ids
  do not exist until S6 authors the twelve scripts. Confirm with whoever plans/executes S6 that
  `pack.json`'s declared ids are updated to match the real authored scripts *before* S6's determinism
  probe runs — a stale id here would fail `pack.find_script` loudly (an S6-time defect, not silent,
  but worth flagging now so it is not a surprise).
- **The `I(t)` summary's exact render wiring (§4.4 item 4) is deferred to Step 5's own design, not
  pre-specified here.** Low risk: the mean/p95 values and the `Y_calls/Y` distinction are both fully
  designed in §3.4/§4.2(f); only the report-string assembly (which existing latency-block rendering
  hook it piggybacks on, if any, versus a fully separate block) is left to the implementer's judgement
  at Step 5, consistent with `-ml` §4.2(f)'s own "no refusal gate... report-only" framing (no new
  statistical machinery, only presentation).
- **The `boundaryRule` JSON shape (§3.3) is this document's own synthesis** — the plan and `-ml` note
  both require the mechanism ("data-driven, not heuristic... an optional `boundaryRule`") but neither
  states a JSON shape at this granularity. The `{"confusedWith": [...], "inclusive": bool}` shape
  proposed here is a reasonable, minimal, testable starting point; if `data-scientist` or a future
  reviewer finds it under-expressive once real catalog data is authored (S6), it is a local, low-blast-
  radius change (one file, one scorer function) to widen.
- **The outcome-vector comparator's exact field set (§3.4) is this document's own decision, not the
  plan's.** Reversing it later (e.g. adding `messagesSent` comparison, or dropping dispatched-argument
  comparison) is cheap — it is a pure function with no stored-record consumer beyond `differingTurns`,
  itself only ever printed as diagnostic text (`-ml` §4.5.1(iii): "diagnostic, outside `n`, never
  pooled into it") — but is called out explicitly per this document's own honesty rule rather than
  presented as settled.
- **`prose_calibration.jsonl`'s manifest key name (`data.prosePseudoCallCalibration`, §3.2) is this
  document's own naming choice**, since S6 is the file's actual author. If S6's implementer prefers a
  different key or a different location (e.g. embedding the 20 replies directly in `pack.json` rather
  than a separate data file), that is a one-line manifest change with no scorer-side consequence
  beyond the path string `prose_detector_precision_recall`'s caller resolves it through.
- **The real catalog/storefront domain content (product names, prices, boundary values) is
  hand-authored data**, not derivable from anything in this repository — sized small deliberately
  (§3.3) so it stays reviewable, but it is still real authoring work this stage's implementer owes,
  not a mechanical step.

## Ready to implement

Document: `docs/plans/small-model-benchmarking-s5-spec.md` (this file). Seven steps (§5), all offline:
**Step 0** (tripwire fix + `results.py` shape extensions — `FunnelCounts`, `HazardPoint`,
`determinismProbe`, `named_metrics()` fix); **Step 1** (`tools/sim.py`/`catalog.json`/
`tools/schemas.json`, E1); **Step 2** (`pack.json`/`prompts/system.md`); **Step 3** (the scorer's
per-turn pure functions); **Step 4** (the ten named adversarial traces, the two discriminating pairs,
`clean_through_turn`, `hazard_points`, the outcome-vector comparator); **Step 5**
(`score_conversations` assembly, E2 re-tested against the real scorer, E3, E4); **Step 6**
(`report.py`'s three new renderers, `_load_conversation_scorer`'s real implementation, E5, one
synthetic end-to-end integration test). Five real plan-vs-shipped-code gaps found and resolved as
design rulings, not left open: the per-arm dispatch-failure count's persistence (§2.3, via
`FunnelCounts.unrunnableToolChannel`, no new top-level field); `ToolCallAggregates`'s missing
`determinismProbe` field and its `hazard` shape's inability to carry `c_t` (§2.4, §3.1); `funnel`/
`hazard` wrongly reaching the generic Arms table via `named_metrics()` (§2.5); and the load-bearing
`ItemResult`-shape design for `cleanThroughTurnH`'s correct pairing — one record per conversation,
`pairingKey = (scriptId, replicate, H-1)` — traced directly through `_paired_rows`/
`_aggregate_item_mismatches`/`PairedOutcomes` rather than guessed (§2.6). Three open items flagged for
confirmation, none blocking: the determinism-probe script-id placeholders, the `boundaryRule` JSON
shape, and the outcome-vector comparator's exact field set — all cheap to revise later since none has
a downstream consumer yet beyond this stage's own tests.
