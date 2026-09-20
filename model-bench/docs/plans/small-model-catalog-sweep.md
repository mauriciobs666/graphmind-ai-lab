# Small-Model Catalog Sweep — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** — (M9) · **Version:** 2

**2026-09-19 revision:** addressed `analyst`'s and `data-scientist`'s "needs changes" reviews
(`docs/reviews/small-model-catalog-sweep.md`, `docs/reviews/small-model-catalog-sweep-ml.md`) —
added `stats.verdict()`'s `correction_k` parameter (blocker), replaced Q1 with a new
`mean_bootstrap_interval` function, resolved Q3 to one combined guard-judge family and Q4 to match,
fixed §3.5's marker-count contract, folded FR-11's concrete figures into guard-judge's/tool-caller's
caveats, and closed two minor gaps (footprint value coercion, latency-`None` handling); noted AC-3's
wording fix is in flight with `tico`.

Requirements: `model-bench/docs/requirements/small-model-catalog-sweep.md` (`tico`, Status: Ready
for design). Coordination ledger: `model-bench/docs/plans/small-model-catalog-sweep-coordination.md`
(`teco`, unit U1 = this document). This plan covers **Track A** only — the report feature FR-6
through FR-12 need to produce five per-pack reports plus one consolidated document. It assumes a
shared sweep session id and a populated `results/runs/` directory as *given inputs*; it does not
design the live 70-run sweep itself (FR-1–FR-5, Track B, held for `devops`).

## 1. Goal & scope

Build the reporting capability the sweep's five deliverables need: (a) a per-pack **ranked table**
across every in-scope model with a stored result for that pack, each with its own confidence
interval, latency p95, estimated footprint, and the pack's own restated ceiling/adequacy caveat
(FR-6, FR-7, FR-11, FR-12); (b) an **optional**, pre-registered, reference-anchored Holm-Bonferroni
family of pairwise verdicts against one designated reference model — never a full pairwise matrix
(FR-8); (c) a design for assembling the five resulting reports into **one consolidated document**
with a non-arithmetic index and a narrative "insights and recommendations" section (FR-9, FR-10).

**Out of scope** (per the brief): the live sweep's execution mechanics (session tagging, the 70-run
loop, continue-on-failure recording — FR-1–FR-5); writing any production code myself; deciding the
statistical validity of any new formula — flagged for `data-scientist`'s review (U2b in the
coordination ledger), not decided here.

**CPG:** considered, not relevant — `GRAPHS` lists `cpg_falkorchat`, `kaizen_team`, `reference`,
`ws:*`; no `cpg_model-bench` graph is loaded, and this task's scope (two files read in full,
`report.py`/`stats.py`/`cli.py`/`packs.py`/`results.py`) needed no call-graph tooling beyond direct
reading.

## 2. Context & findings

Read in full: `model-bench/AGENTS.md`, `docs/requirements/small-model-catalog-sweep.md`,
`docs/plans/small-model-catalog-sweep-coordination.md`, `docs/plans/small-model-benchmarking-ml.md`
(repo root), `modelbench/report.py`, `modelbench/stats.py`; substantially:
`modelbench/cli.py` (`_select_arms`, `_cmd_compare`, the command table), `modelbench/packs.py`
(`PackRef`, `PackMetrics`), `modelbench/results.py` (`RunResult`, `BinaryMetric`, `ContinuousMetric`,
`DistributionSummary`, `load_history`, `models_with_stored_results`), `modelbench/scoring/
classification.py`, `docs/BACKLOG.md`, `docs/HISTORY.md`, `README.md`, and the five
`packs/*/pack.json` manifests.

### 2.1 The two-arm ceiling, confirmed

`report.py::_comparison_pair` (`:594-603`) unconditionally returns `runs[0], runs[1]`; `cli.py::
_select_arms` builds whatever arm list `--models`/`--session`/`--negative-control` select, but every
verdict downstream (`compare_report`'s Holm/`stats.verdict`/`stats.continuous_verdict` machinery)
is written for exactly two arms. Matches teco's finding exactly.

**Also confirmed, and this changes one framing in the brief:** `runner.RunConfig.referenceKey`
(`run --reference <key>`) is accepted by the CLI but **read nowhere** — grepped `referenceKey`
across `modelbench/`, the only two hits are the field declaration (`runner.py:127`) and the call
site that assigns it into `RunConfig` (`cli.py:399`); `run_pack` never consults it, and no
`RunResult`/fingerprint field carries it. So "reference" is not a run-time concept `run` records
today — the reference-anchored family (FR-8) has to be a **report-time selection**: an operator
names a `modelKey` to the new report command, and the renderer treats whichever *stored* run has
that key as the anchor. This plan's design does not depend on `--reference` becoming wired up.

### 2.2 What already generalizes to N arms, and what doesn't

`compare_report`'s per-arm sections are written as `for r in runs` (or set comprehensions over
`runs`) and are **already N-arm-generic**: the version/hash/schema-mismatch banners (`:995-1021`),
the invalid/excluded-arm block (`:1022-1042`), the funnel table (`:1045-1046`), the "## Arms"
descriptive table with its per-metric Wilson interval (`:1049-1094`), "## Speed" (`_render_speed`),
the per-turn-position table, and the hazard curve. Only the "## Verdicts" section — everything from
`_comparison_pair` on (`:1105-1399`) — is fixed at two arms. This means the new ranked-table
renderer does **not** need to reinvent per-arm rendering; it needs a new *selection and ordering*
layer (rank by one metric's value, restated caveat, footprint column) plus, optionally, a new
*pairwise* layer built from the same two-arm primitives `stats.py` already exposes
(`PairedOutcomes`, `stats.verdict`, `stats.continuous_verdict`, `stats.holm_steps`,
`stats.resolving_power`) — never a second copy of any of them.

### 2.3 The five packs' shape (read from each `pack.json` + `-ml`)

| Pack | `headlineMetric` | `verdictMetrics` | metric kind | polarity | own caveat (`-ml`) |
|---|---|---|---|---|---|
| `embedder-graphrag-retrieval` | `mrr` | `[mrr]` | continuous (`ContinuousMetric.mean`) | higher is better | recall@10 = 37/38 — can detect a *worse* embedder, never certify a *better* one (`-ml` §7.4) |
| `guard-judge-understanding` | `None` | `[falseAdvanceRate, falseSuspendRate]` | binary | **lower is better** (§2.4 below) | two co-equal class-conditional error rates, no single headline; floor 15.0/20.0 pp, MDD₈₀ 21.9/28.7 pp at the two-member `alpha_mdd=0.025` (`-ml` §7.3) |
| `nlq-structured-query` | `layer1ExactMatchRate` | same | binary | higher is better | true denominator is **34**, not 40 — 6 items are structurally unanswerable and excluded (`-ml` §7.2 table, v1.25 note) |
| `tool-caller-shop-assistant` | `cleanThroughTurnH` | same | binary | higher is better | analysis unit is **scripts**, n=12 (3 shapes × 4 scripts); floor 50.0 pp, MDD₈₀ 57.8 pp (`-ml` §7.2/§4.5) |
| `chat-responder-grounded-answers` | `groundingRate` | same | binary | higher is better | reply *quality* is not measured, only containment/format/latency (`-ml` §3.8.5; already rendered today by `report._render_role_caveat`, reusable verbatim) |

### 2.4 A pre-existing defect this feature is at risk of first surfacing (read-verified, not yet observed live)

`modelbench/scoring/classification.py:209-210,259`: for `falseAdvanceRate`/`falseSuspendRate`, an
item's own `counts[metric]` is `1` when the **error itself** occurred (`count = int(not advanced)
if metric == "falseSuspendRate" else int(advanced)`, with the tier comments confirming "a
False->True flip IS the false-advance event"). So a model's own `BinaryMetric.rate` for these two
metrics is the **rate of the undesired event** — higher is worse — unlike every other pack's
verdict metric, where "successes" is the desired outcome.

`stats.verdict()`'s generic winner selection (`:1320`, `winner, loser = (a_label, b_label) if diff
>= 0 else (b_label, a_label)`) and `report.py`'s reused text are polarity-blind: they call whichever
arm has the numerically higher raw rate "better," unconditionally. For `falseAdvanceRate`/
`falseSuspendRate` this means a **distinguishable** verdict would print "X is better than Y" when X
in fact has the *higher* error rate — backwards. I checked every shipped `reports/guard-judge-*.md`
(`grep -n "is better than"` across all three) and none has yet reached a distinguishable verdict on
either metric, so this has never been visible in a rendered report — it is a latent defect, verified
by reading the two source files above, not by observing it fire.

**This is out of scope for this feature to fix** — it lives in already-shipped, heavily-tested,
shared `stats.verdict()`/`report.py` machinery no FR here asks me to touch, and a fix there needs
its own dedicated review given this module's mutation-testing bar. I recommend a **new**
`docs/BACKLOG.md` entry (I cannot write it myself — architect's write access is limited to
`docs/plans/`) naming it: *"`stats.verdict()`'s 'is better than' wording assumes every verdict
metric's raw rate is higher-is-better; `falseAdvanceRate`/`falseSuspendRate` are rates of an
undesired event, so a future distinguishable guard-judge verdict would print backwards. Read-
verified in `docs/plans/small-model-catalog-sweep.md` §2.4, never yet observed live."* — filed by
whichever agent lands this plan's implementation step.

**What this plan does instead:** the new ranked-table/family code (§3 below) is polarity-aware from
the start, via one small, explicit, testable constant — it does not touch or reuse `Verdict.text`
for the family rendering (§3.2).

## 3. Design & rationale

### 3.1 The ranked table (FR-6, FR-7, FR-11, FR-12)

**New function, not a `compare_report` branch.** `compare_report` is already ~450 lines carrying
five honesty-rule-critical branches (mixed-kind family, continuous family, homogeneous-binary
Holm). Adding a "print an N-row table instead of a pairwise verdict" branch inside it risks
entangling an orthogonal rendering mode with Rule 7/Holm logic under edit — exactly the shape this
module's own docstring warns about ("two copies of a formula is one copy and one bug" cuts the same
way for control flow). A **new function**, `report.rank_report(...)`, reuses the already-N-arm-
generic pieces (§2.2) as **library calls**, not copy-paste: the exclude-on-mismatch pass
(`_aggregate_item_mismatches`), the version/hash/schema banners, `_arm_names`. It does **not** call
`_render_funnel`/`_render_per_turn_position`/`_render_hazard` — those are two-arm deep-dive tables;
a 17-row ranking overview does not need per-arm funnels, and omitting them keeps the new function
focused (rejected alternative: include them anyway "for completeness" — rejected because it would
make a 17-model report enormous and because FR-7 explicitly scopes the ranked table to rank + CI +
latency + footprint + caveat, nothing else).

```python
def rank_report(
    runs: Sequence[RunResult],
    *,
    pack: PackRef,
    invalid: Sequence[InvalidRecord] = (),
    reference: str | None = None,
    footprints: Mapping[str, str] | None = None,
) -> str:
    """FR-6/FR-7/FR-11/FR-12: one ranked table per pack (two, for a pack with no headline
    metric — one per verdictMetrics member), covering every in-scope model with a stored,
    consistent result — never a pairwise matrix. `reference`, when given, additionally renders
    FR-8's optional reference-anchored Holm-Bonferroni family (§3.2)."""
```

Caller contract: `runs` must carry **unique `modelKey`s** — `rank_report` raises (a new
`DuplicateModelInReport` `ValueError`, mirroring `PairedOutcomes.from_units`'s duplicate-unit-id
guard) rather than silently keeping one. Deduplication is the **caller's** job (§3.3's
`_select_rank_arms`), exactly the "backstop, not the mechanism" split `PairedOutcomes` already uses.

**Per pack with a headline metric** (`embedder`, `nlq-generator`, `tool-caller`, `chat-responder`):
one ranked table, sorted by that metric's own value, polarity-aware (§3.1.1). **Per pack with none**
(`guard-judge`): two ranked tables, one per `verdictMetrics` member, each sorted by its own value —
there is no single number to rank by, matching `packs.py`'s own "co-equal, no headline" design
already enforced elsewhere in this module (`compare_report`'s headline-block branch, `:1356-1378`).

Columns: rank | model | k/n | rate (or mean, for a continuous metric) | 95% CI | latency p95 |
footprint. Reused/new per column:

- **k/n, rate, Wilson CI** (binary metrics): `stats.wilson_interval`, exactly the existing "## Arms"
  table's own per-arm cell (`report.py:1064-1079`) — same descriptive-only framing, same
  `_DESCRIPTIVE_NOTE` reused verbatim underneath the table.
- **mean CI** (continuous metrics — today, only `mrr`): a **new** function, `stats.
  mean_bootstrap_interval` (§3.2's `stats.py` change also covers this) — **not** a reuse of
  `paired_bootstrap`, resolved by `data-scientist`'s review (`docs/reviews/small-model-catalog-
  sweep-ml.md` §3, replacing this plan's original Q1 recommendation): `paired_bootstrap`'s
  difference-support clamp `(lo-hi, hi-lo)` is the wrong clamp for a one-sample mean — MRR's own
  support `(0.0, 1.0)` must clamp the interval directly, never as a difference — and a single arm's
  own mean is a descriptive quantity with no verdict shape to borrow. `mean_bootstrap_interval`
  shares the resample *engine* with `paired_bootstrap` (factored into a private helper both call)
  but takes `support` as the direct clamp; `ContinuousMetric.support` (`results.py:120-123`) already
  carries the metric's own bounds, so no new field and no new judgment call about what the bound is.
  Carries its own descriptive-only caveat, stronger than Wilson's (verbatim per the review): *"this
  interval describes this model's own mean; it is not a comparison, and two such intervals
  overlapping or not overlapping is not itself a basis for a verdict — see FR-8's optional
  reference-anchored family for an actual test, when one was run."*
- **latency p95**: `RunResult.latency.latencyMsP95`, already-shipped (`_render_speed`'s own field).
  When `run.latency is None` (a deterministic arm, or a fixture predating latency capture — never a
  live sweep run), the cell renders `—`, matching `_render_speed`'s own per-field guard
  (`report.py:920-944`) rather than inventing a new convention.
- **footprint**: an opaque, pass-through **string**, never a number the renderer parses or compares
  — see §3.3's footprint-plumbing decision. This is a deliberate simplicity trade-off: FR-12 itself
  says footprint is "not something model-bench itself measures," so treating it as inert display
  text removes any risk of a footprint figure silently feeding a computation the honesty rules would
  otherwise have to police. A **non-string** footprint value is coerced to `str()` at load time
  (§3.3, Unit B's job) before it ever reaches this renderer — the renderer itself never sees
  anything but a string or a missing key.
- **restated ceiling/adequacy caveat** (FR-11): a per-pack (or per-`role`) constant string, printed
  as a block-quote above the table — the same pattern `_render_role_caveat` already uses for
  `chat-responder` (reusable **verbatim** for that one pack), generalized to a small dict covering
  all five packs, sourced from §2.3's table above (which is itself sourced from `-ml`, cited by
  section in each string, matching this module's existing citation discipline). **Guard-judge's and
  tool-caller's strings carry their concrete floor/MDD figures**, not just the qualitative shape —
  per `data-scientist`'s review (§7): guard-judge states floor 15.0/20.0 pp, MDD₈₀ 21.9/28.7 pp
  (`falseAdvanceRate`/`falseSuspendRate`, at the two-member `alpha_mdd=0.025`, `-ml` §7.3);
  tool-caller states floor 50.0 pp, MDD₈₀ 57.8 pp at n=12 scripts (`-ml` §7.2/§4.5). §2.3's table
  above is updated to match.
- **resolving-power sentence** (FR-7's "recomputed for the model count actually in that report"):
  resolved by `data-scientist`'s review (`-ml` review §6; restated as this plan's §3.4 Q4) —
  printed unconditionally whenever the ranked table has ≥2 models, **alongside**, never instead of,
  the pack's own already-published single-comparison resolving power from `-ml` §7.1–§7.4
  (unchanged, via the existing `resolving_power_line`). The new sentence states what a *maximal*
  reference-anchored family (FR-8) would cost at this table's own model count, with `k` matching
  §3.2's family-size resolution (`N-1` for a single-metric pack, `2·(N-1)` for guard-judge) and
  names no specific anchor model.

#### 3.1.1 `_metric_value` and polarity

```python
def _metric_value(run: RunResult, metric: str) -> float | None:
    """The one sortable number for `metric` on `run`'s own aggregate: `BinaryMetric.rate` or
    `ContinuousMetric.mean`. Raises `PackConfigError` for a `DistributionSummary` — no shipped
    pack's headline/verdictMetrics member resolves to one today (separationRaw/Z are exploratory
    only), and silently picking median over p10 would be a guess this module refuses elsewhere.
    `None` when `run` declares no aggregate for `metric` at all (excluded from ranking, not ranked
    last — a model with no data for this pack is not "worst," it is absent)."""

#: Verdict metrics whose raw rate is a rate of an UNDESIRED event (§2.4) — declared explicitly
#: because no manifest field carries polarity. Scoped to this ranked-table/family code only; it
#: does not change `stats.verdict()`'s own text (§2.4's flagged, unfixed defect). Confirmed
#: complete against every pack.json shipped today by `data-scientist`'s review (`-ml.md` §4,
#: checked against each pack's own scorer). **If `falseAdvanceRateBoundary` (currently
#: exploratory-only, `classification.py::_METRIC_BY_TIER`) is ever promoted into a pack's
#: `verdictMetrics`, this set must gain it in the same change** — same undesired-event polarity,
#: confirmed by the same review; cross-referenced at `_METRIC_BY_TIER`'s own declaration site too
#: (implementation step, §4).
_LOWER_IS_BETTER: frozenset[str] = frozenset({"falseAdvanceRate", "falseSuspendRate"})

def _better(metric: str, value: float, other: float) -> bool:
    """True if `value` ranks at or above `other` on `metric`, honoring `_LOWER_IS_BETTER`."""
```

A guard-shaped constant per this module's own convention (AGENTS.md, "A guard's reach lives in an
asserted constant"): the test strategy (§5) mutates `_LOWER_IS_BETTER` both ways (shrink: does
`falseSuspendRate` sort backwards if removed; widen: does a normal higher-is-better metric sort
backwards if wrongly added).

### 3.2 The optional reference-anchored family (FR-8), and the `stats.py` change it needs

Triggered by `reference: str | None` — a `modelKey` the caller names (via the new CLI's
`--reference` flag, §3.3), not `RunConfig.referenceKey` (§2.1, inert).

#### 3.2.1 [BLOCKER, resolved] A required `stats.py` change: `verdict()`'s new `correction_k`

**Found independently by both reviewers** (`docs/reviews/small-model-catalog-sweep.md` §2.1,
`docs/reviews/small-model-catalog-sweep-ml.md` §2) — **resolved by `data-scientist`, not re-derived
here.** `stats.verdict()`'s Rule 4 preconditions 3 and 5 (`stats.py:1189-1195,1207-1219`) tie the
Holm-correction divisor to `len(family)` — the pack's own pre-registered *metric* count (1 or 2 for
every shipped pack) — because the only existing call site (`report.py:1236-1294`) runs its Holm
ladder over exactly that axis, where the two coincide by construction. FR-8's family needs a
different, larger axis: up to 16 *candidates* against one reference. Reusing `verdict()` unmodified,
as this plan originally (incorrectly) assumed, raises `ValueError` on the very first candidate at
the sweep's own real scale, whichever of `resolving.alpha_mdd`/`family` is adjusted to try to match
— worked through in full, with concrete numbers, in both reviews; not repeated here.

**Resolution: `stats.verdict()` gains one new, keyword-only, optional parameter:**

```python
def verdict(
    outcomes: PairedOutcomes,
    *,
    resolving: ResolvingPower,
    metric_name: str,
    family: Sequence[str],
    correction_k: int | None = None,   # NEW — defaults to len(family); today's call site unchanged
    a_label: str = "A",
    b_label: str = "B",
    alpha_step: float | None = None,
    holm_tested: bool = True,
) -> Verdict: ...
```

Precondition 3 becomes `k = correction_k if correction_k is not None else len(family)`, checked
against `resolving.alpha_mdd == resolving.alpha_family / k`; every other precondition, and Rule 7's
floor enforcement, is untouched — `metric_name in family` still requires a genuinely pre-registered
metric, so this cannot smuggle an unregistered metric into a verdict. **Rejected alternative** (the
reviews' own other sketched option): reimplementing Rule 7's floor check and the winner label
directly from `PairedOutcomes`/`mcnemar_exact`/`resolving_power.observable_floor`, bypassing
`verdict()` — rejected because it creates a **second enforcement site** for "no path returns
`distinguishable` below the observable floor," exactly the class of duplication this module's own
convention refuses ("two copies of a formula is one copy and one bug").

A default (rather than a required keyword) is correct here despite this module's general "nothing
that shapes a decision carries a default" rule: the default reproduces the one behavior already
shipped and audited (`correction_k = len(family)`), which is not the anti-conservative-by-omission
shape that rule exists to block, and FR-8's own call site always passes it explicitly. **This is a
`stats.py` contract change and needs its own review pass given this module's mutation-testing bar**
— its own implementation step and test obligation, §4/§5, not folded silently into Unit A's other
work.

**Scope note for later, not now:** `continuous_verdict()` has the identical `family`/`alpha_family`
coupling and would need the same parameter if a future pack ever needs FR-8's family on a
continuous headline metric with more than two in-scope models. Not needed today — the only
continuous-headline pack (`embedder`) has 2 in-scope models total, so FR-8's family is never
meaningful there regardless of this change.

#### 3.2.2 The family itself: one combined candidate×metric ladder, not one ladder per metric

For **each** `verdictMetrics` member: build one candidate-vs-reference two-arm comparison per other
model, using the **existing** pairwise primitives exactly as `compare_report`'s homogeneous-binary
path already does (`_paired_rows`/`PairedOutcomes.from_units`, then `stats.verdict` with §3.2.1's
new `correction_k`; the continuous path is out of scope per §3.2.1's scope note) — never a second
copy of that arithmetic.

**One combined Holm ladder across (candidate × metric), not one ladder per metric** — resolved by
`data-scientist`'s review (`-ml` review §5), **reversing this plan's original per-metric default,
which under-corrects.** `-ml` §3.3's own reasoning for why guard-judge's two metrics must share one
ladder for a single A-vs-B comparison ("two co-equal verdicts at α=0.05 each carry a ~9.75% chance
of at least one false 'better' under the null") applies with the same force, and more so, to the
candidate axis: two independent 16-candidate ladders bound the **combined** false-positive risk
across all 32 tests only by the union bound (~9.75%), reintroducing at a larger scale precisely the
risk guard-judge's own pre-registration was built to close. FR-8's own wording ("**one**
pre-registered, reference-anchored family per pack," singular) supports the combined reading, not
the split one.

```python
p_values: list[float] = []          # length 2*(N-1) for guard-judge, N-1 for every other pack
cells: list[tuple[str, str]] = []   # (candidate_key, metric_name), same order as p_values
for metric in family_members:       # 1 member for four packs, 2 for guard-judge
    for candidate in candidates:    # N-1 models, excluding the reference
        ...build PairedOutcomes, append mcnemar_exact(b, c) to p_values...
steps = stats.holm_steps(p_values, alpha=pack.metrics.alpha_family)   # ONE call, unchanged primitive
```

`holm_steps` is already fully generic over `p_values` (`k = len(p_values)` internally, no notion of
which axis contributed which entry, verified by reading its body — `stats.py:1447-1486`) — this is
strictly **simpler** than this plan's original two-ladder design, not more code. Render one table
per metric by filtering `cells`/`steps` back apart after the one combined call.
`correction_k = len(p_values)` — `N-1` for a single-metric pack's family, `2·(N-1)` for guard-judge's
combined one — passed once per `stats.verdict()` call, matching whichever `p_values` list produced
that candidate's `alpha_step`. Holm-Bonferroni controls the family-wise error rate under **arbitrary
dependence** among the p-values, so combining a metric axis scored over disjoint item subsets
(guard-judge's `falseAdvanceRate`/`falseSuspendRate` already are) with the candidate axis inherits
that same guarantee — no new dependence-structure assumption is introduced (`-ml` review §5).

**Cost, stated plainly:** guard-judge's tightest Holm step moves from `α/16 = 0.003125` (this plan's
original, incorrect default) to `α/32 = 0.0015625` (this resolution) — a real power cost, and the
price FR-8's own "one family per pack" design already commits to for the one pack with two co-equal
metrics, not a new cost this revision introduces.

**Deliberately does not reuse `Verdict.text`/`ContinuousVerdict.text` verbatim.** Those strings bake
in the polarity-blind "X is better than Y" framing (§2.4). The family renderer builds its own
sentence from the structured fields (`Verdict.diff`, `.ci`, `.distinguishable`, `.decided_by`,
`.mcnemar_p`, `.b`, `.c`), choosing the winner label via `_better`/`_LOWER_IS_BETTER` from §3.1.1
rather than `diff >= 0`. This is more code than "print `.text`," but it is the only way to render
`falseAdvanceRate`/`falseSuspendRate` verdicts correctly without touching the shared function that
has the defect.

**Rendering shape:** one table per metric — candidate | diff (signed, polarity-corrected so a
positive number always reads "candidate is better") | 95% CI | Holm-adjusted threshold | decision
(`distinguishable` / `not distinguishable` / `not tested (Holm stops here)` / `no verdict — no
paired data`) — the same four-state vocabulary `_decision()` already renders in `compare_report`'s
family-wise table (`:501-511`), reused directly (it takes a `Verdict | None` and a `HolmStep`,
neither of which changes shape here).

**Everything outside this family is labelled `exploratory — no significance claim`** (the
acceptance criterion's exact phrase, already a literal string in `report.py`, e.g. `:1396`) —
mechanically true by construction here, since the ranked table itself (§3.1) never computes or
prints a p-value or a Holm decision; only this optional family does, and only for the metrics it
covers.

### 3.3 CLI shape: a new `rank` command, not a `compare --rank` flag

**Decision: new subcommand.** `compare`'s existing flags (`--models`, `--session`,
`--negative-control`) are wired through `_select_arms` into a fixed two-arm (or `[x, x]`) selection
that `_comparison_pair` then narrows further; overloading `--rank` onto `compare` would need a
second, incompatible selection path inside the same function and a second incompatible rendering
branch inside `compare_report` (§3.1's own rejected alternative). A new verb also matches this
codebase's existing one-verb-per-shape precedent (`compare`, `index rebuild`, `models --tested`,
`attest`, `validate`, `run` are six distinct commands, not flags on one). Rejected alternative:
teach `compare_report` an `arms: Sequence[RunResult]` len-dispatch — rejected because it makes one
function branch on both "how many arms" and "what kind of report," which is exactly the kind of
entanglement §3.1 already argues against for the renderer; the same argument applies one layer up,
at command dispatch.

```
./run.sh rank --pack <pack-id> [--session <id>] [--reference <model-key>]
    [--footprints <path.json>] [--out <path>]
```

- `--session`, `--out`: same semantics as `compare`'s.
- `--reference`: a stored `modelKey` for this pack (and session, if given) to anchor FR-8's family;
  omitted → ranked table only, no family, no per-model p-value anywhere in the report.
- `--footprints <path.json>`: optional; a flat `{"<modelKey>": "<display string>"}` map, read from
  disk and passed through to `rank_report`'s `footprints` parameter — **never parsed for a number**,
  so a malformed or missing entry can only ever produce a missing/wrong-looking display cell (`—`),
  never a computation error. A **malformed file** (unparseable JSON, or not a JSON object) is a
  usage error, exit 2 — the same shape `_gather_attested_fields` already uses for a bad `attest`
  input. A **present but non-string value** under a modelKey (a JSON number, object, or array) is
  coerced to `str()` at load time, in `cli.py`, before it is ever passed to `rank_report` — never an
  exception: a malformed *value* degrades to a wrong-looking display cell exactly as a missing one
  does (`analyst` review §2.4). Absent flag → every footprint cell prints `—`.

**AC-3 reconciliation.** The requirements doc's AC-3 literally names `./run.sh compare --session
<sweep-session-id>`; this plan's new `rank` command satisfies AC-3's *intent* (one report per pack
covering every in-scope model with a stored result) but not its *literal wording* — `compare`
remains fixed at two arms by design (§2.1), so AC-3 as written cannot be satisfied by the verb it
names (`analyst` review §2.3). `teco` is dispatching `tico` to amend AC-3's wording to name `rank`;
no plan or requirements-doc edit is needed from `architect` beyond this note.

`_select_rank_arms(runs, *, session)`: filter to `session` if given (identical to `_select_arms`),
then dedupe by `modelKey` keeping the newest-stored run (`by_key = {r.modelKey: r for r in
candidates}`, last value wins) — the exact one-liner `_select_arms`'s `--models` path already uses
and AGENTS.md already documents the semantics of, reused rather than reinvented. `_cmd_rank`
mirrors `_cmd_compare`'s structure (load manifest → `load_history` → select arms → validate
`--reference` is present, `UnknownModelKey`-shaped error, exit 2, if not → render → write via a new
`_rank_report_path(root, pack_id)`, identical to `_report_path` but with a `-rank-` filename infix
(`reports/<pack-id>-rank-<date>-<n>.md`) so a later consolidation step (§3.5) can find "the ranked
report" unambiguously and never picks up an ordinary two-arm `compare` report by accident).

**Footprint data itself is out of scope for this plan to author.** The 19-model footprint table is
sweep-specific research (LM Studio catalog metadata or manual lookup per model), not a general
`model-bench` capability — hardcoding it into `modelbench/` would tie a reusable library to one
one-time sweep's model list, which conflicts with this component's own "no CI hook, no leaderboard,
compare within a role generically" framing. The JSON file itself is produced by whoever executes the
report-rendering step (coordination ledger's U6), sourced from the requirements doc's own footprint
figures where given and researched for the rest; this plan only specifies the plumbing (flag,
schema, pass-through).

### 3.4 Statistical design questions (Q1–Q4) — resolved by `data-scientist`'s review (U2b)

This plan originally flagged four questions rather than deciding them. All four are now resolved by
`docs/reviews/small-model-catalog-sweep-ml.md`; each resolution below is cited from that review, not
re-derived here.

- **Q1 — no existing single-arm CI for a continuous metric, resolved: a new function, not a reuse.**
  `paired_bootstrap` is not valid unchanged for a one-sample mean — its difference-support clamp
  `(lo-hi, hi-lo)` is the wrong clamp for a single arm's own mean, which must clamp directly to the
  metric's own support (`ContinuousMetric.support`, already stored, no new field). Build
  `stats.mean_bootstrap_interval(values, *, B, seed, levels, support)`, sharing the resample engine
  with `paired_bootstrap` via a private helper, never calling `paired_bootstrap` itself with raw
  values. Fully specified in §3.1's CI-column bullet and §4's implementation step (`-ml` review §3).
- **Q2 — `_LOWER_IS_BETTER` is complete and correctly scoped, no change** — confirmed by
  `data-scientist` against every pack's own scorer (`-ml` review §4's table). One latent-risk note
  folded in: `falseAdvanceRateBoundary` (currently exploratory-only) shares the same undesired-event
  polarity and must be added to `_LOWER_IS_BETTER` in the same change the day it is ever promoted
  into a `verdictMetrics` list — cross-referenced at both declaration sites (§3.1.1, and
  `_METRIC_BY_TIER`'s own comment, an implementation-step addition, §4).
- **Q3 — one combined candidate×metric family for guard-judge, not two independent per-metric
  ladders** (reverses this plan's original default, which under-corrected) — full mechanism in
  §3.2.2; `k = 2·(N-1)` for guard-judge, `N-1` for every other in-scope pack.
- **Q4 — the no-`--reference` resolving-power sentence: `k` must follow Q3, and the "top-ranked
  model" anchor is dropped.** This plan's original generic `k = N-1` regardless of pack is wrong for
  guard-judge (must be `2·(N-1)`, matching Q3). The sentence must not name a specific anchor model —
  "the top-ranked model" is a fact about this run's own results, and naming it as the hypothetical
  anchor reads as a post-hoc, choose-after-you-see-the-data pre-registration, exactly what `-ml`'s
  `verdictMetrics` discipline exists to foreclose. Model-agnostic wording instead: *"if this pack's
  optional reference-anchored family (FR-8) were run — any one of the {N} models here designated as
  the reference, the other {N-1} compared against it{, jointly across both verdict metrics, for
  guard-judge} — that family of {k} tests would resolve differences of >= X pp with 80% power
  (α_mdd = {alpha_family}/{k})."* Printed **alongside**, never instead of, the pack's own
  already-published single-comparison resolving power (`-ml` §7.1–§7.4, via the existing,
  unmodified `resolving_power_line`) — the two answer different questions ("what could one
  comparison show" vs. "what would scanning every model cost") and neither supersedes the other.
  `resolving_power_line`'s existing "Best case — assumes the candidate wins every {unit} the models
  differ on" sentence (`report.py:432-434`) is written for one already-run comparison and is false
  or misleading applied to this hypothetical; the new sentence needs its own variant wording ("each
  of the other models would win every {unit} it differs from the reference on") rather than reusing
  that clause verbatim — implement as a `hypothetical: bool` branch on `resolving_power_line` or a
  small parallel function, implementer's call (`-ml` review §6, this plan's §4).

### 3.5 The consolidated document (FR-9, FR-10)

**Not a `modelbench` runtime code path.** The structural invariant (`load_history()` takes one
`packId`, no API loads across packs) is about **computation** — no function anywhere may combine a
score from more than one pack's stored results. Reading five *already-rendered* markdown files as
opaque text and stitching them under a cover section touches that invariant's letter only if a
naive design re-invokes `load_history`/`compare_report`/`rank_report` across packs in one call; a
pure text-assembly step over five finished artifacts does not, and preserves the invariant's actual
purpose (never compute a cross-pack number) while avoiding manual copy-paste transcription risk.

**Design: a small extraction script, `scripts/consolidate_sweep_reports.py`, plus mandatory
human/agent-authored narrative — not full templating.** Three options considered:

1. **Fully manual document.** Simplest, zero new code, but a human/agent hand-copying five top-row
   figures into an index table is exactly the kind of transcription step this codebase's own
   philosophy (refuse rather than risk a silent wrong number) argues against.
2. **Mechanical index extractor + human/agent narrative (chosen).** The script does zero
   arithmetic — pure text extraction of each ranked report's own top row(s) — and leaves every
   judgment call (the preamble's honest sample-size statement, the per-role recommendation with its
   trade-offs) to whoever executes the coordination ledger's U7, unavoidably a narrative-writing
   task no template can discharge faithfully (FR-10 wants a *rationale*, not a mechanically-derived
   sentence).
3. **Full templating, including narrative.** Rejected: generating "recommended model per role"
   prose from a template either overclaims (reads as a hardcoded verdict, in a tool that has none by
   design — README's "no gate" invariant) or requires an LLM call this zero-runtime-dependency,
   standalone tool does not make anywhere else in its report path.

**Mechanism — one or more marker lines per report *file*, one per rendered table.** `rank_report`
(§3.1) emits, immediately after each ranked table, one machine-readable HTML comment naming that
table's own top row — e.g. `<!-- rank-report: pack=<packId> metric=<metricName> top=<modelKey>
value=<v> ci=[<lo>,<hi>] -->` — so the extractor never depends on markdown table formatting (column
order, cell text) staying stable. **Four packs' report files carry exactly one marker each; the
guard-judge file carries two** — one per `verdictMetrics` member (§3.1's own "two ranked tables" for
a pack with no headline) — so at this sweep's five files the extractor reads **six** markers total,
not five (`analyst` review §2.2, which found §3.1 and this section disagreeing on the count). The
extractor is written for a variable marker count per file from the start: `consolidate_sweep_reports.
py --reports <path1.md> ... <path5.md> --out <path>` (explicit paths, not auto-discovered by glob —
this is a one-time, human-triggered assembly step, and explicit inputs cannot silently pick up a
stale or wrong-session report) parses **every** `<!-- rank-report: ... -->` line found across the
five files — one per line, keyed by `(pack, metric)`, never assumed 1:1 with files — and writes a
skeleton: a preamble section left as a `<!-- TODO -->` placeholder (the sample-size honesty
statement AC-6 requires is a judgment call, not extractable), a mechanically-filled index table with
one row per marker (pack, metric, top model, value, CI, a relative link to the full per-pack report
— never a second computation of anything; guard-judge contributes **two** rows, one per metric, and
the worked example in Unit C's own docstring should show this explicitly so an implementer doesn't
discover the variable count mid-implementation), and one `### <role>` heading per role under "##
Insights and recommendations," each holding a `<!-- TODO: narrative, grounded only in this role's
own within-pack report -->` placeholder for whoever does U7 to fill in by hand.

Output path: `reports/<sweep-session-id>-consolidated.md` — `reports/` is already a committed,
durable-output directory (git-tracked, per `.gitignore`), matching every other rendered report's
home; no new docs-lifecycle home is needed since this is a rendered artifact, not engineering-
process documentation.

## 4. Step-by-step implementation

Sized per this component's own convention (split past ~3 sequential steps or ~5 files). Three
units, each independently reviewable and buildable:

### Unit A — `report.py` + `stats.py` core (statistical/rendering): recommend `tdd-engineer`

Files: `modelbench/stats.py`, `modelbench/report.py`, `tests/test_stats.py`,
`tests/test_report.py`, plus a one-line docstring cross-reference in
`modelbench/scoring/classification.py` (Q2's `_METRIC_BY_TIER`/`_LOWER_IS_BETTER` coupling note, no
behavior change). Recommended over `coder` because the behavior contract is clear (§3.1–3.2 above)
but the exact sequencing of edge cases — empty ranking, a model present in `runs` with no aggregate
for the target metric, a mixed-kind guard-judge family, Holm's stop-at-first-non-rejection
interacting with a polarity-corrected label, a `reference` key absent from `runs` — is large enough,
and similar enough in shape to this file's existing edge-case-driven test suite
(`tests/test_report.py`'s ~3000 lines, one behavior-named test per edge case), that test-first
discovery is the efficient path, matching how this exact file's prior stages were built.

**`stats.py`'s two additions (`correction_k`, §3.2.1; `mean_bootstrap_interval`, §3.4 Q1) are their
own reviewed, mutation-tested change and naturally come first** in the TDD sequence — `report.py`'s
ranked-table and family code cannot be written against them until they exist. Not a rigid step table
otherwise (TDD sequencing beyond that dependency is the implementer's call) — an ordered list of
behaviors to drive red→green is in §5 below.

### Unit B — CLI wiring: recommend `coder`

Files: `modelbench/cli.py`, `tests/test_cli.py`, `README.md`. Fully specified, mechanical:

1. `_build_parser()`: add the `rank` subparser (`--pack`, `--session`, `--reference`,
   `--footprints`, `--out`, plus `with_root`), mirroring `compare`'s registration exactly.
2. `_select_rank_arms(runs, *, session)` and `_rank_report_path(root, pack_id)` (§3.3) — two small
   functions beside `_select_arms`/`_report_path`.
3. `_cmd_rank(args)` mirroring `_cmd_compare`'s structure (§3.3); wire into `main()`'s dispatch
   table beside `_cmd_compare`.
4. `README.md`'s "What the CLI does today" section: add `rank`'s command line and a short
   paragraph, matching the existing style for each command.
5. Depends on Unit A's `rank_report` signature (§3.1) being stable — sequence after Unit A, or in
   parallel against an agreed signature if schedule pressure requires it (no shared file with Unit
   A, so no merge conflict either way).

### Unit C — consolidation script: recommend `coder`

Files: `scripts/consolidate_sweep_reports.py`, `tests/test_consolidate_sweep_reports.py` (light
touch, matching `scripts/refresh_golden.py`'s precedent of a modest fixture-based test file rather
than the core library's mutation-testing bar — this is a one-time convenience script, not part of
the honesty-rule surface).

1. `rank_report` emits the `<!-- rank-report: ... -->` comment lines, **one per rendered table —
   two for guard-judge's file, one for every other pack's** (this line item actually belongs to
   Unit A — call it out there too, since Unit C's script is unusable without it).
2. `consolidate_sweep_reports.py`: parse **every** comment line found across `--reports <paths...>`
   (not one per file — §3.5's fixed count), emit the skeleton document (§3.5) to `--out`.
3. A handful of fixture tests: two or three small stub rendered-report files with known comment
   lines in — **including one guard-judge-shaped stub carrying two markers** — assert the extracted
   index table matches, with the correct row count (six rows across five files, not five).

Depends on Unit A (the comment-line contract, including its variable-per-file count) but not on
Unit B (no CLI involvement) — can run in parallel with Unit B once Unit A's comment format is fixed.

### Documentation

Per this repo's standing doc-curator convention, whoever lands each unit updates: `README.md` (Unit
B, above), `docs/BACKLOG.md` (the §2.4 polarity-defect entry — must be added by whoever lands Unit
A, since I cannot write it myself), `docs/HISTORY.md` (a dated entry per unit landed, per this
component's own convention), and the coordination ledger (`teco`'s standing responsibility, not an
implementer's).

## 5. Test strategy

**Unit A (`stats.py` first, then `report.py`) — behaviors to drive red→green, roughly in dependency
order:**

0. **`stats.py` additions, before any `report.py` behavior below depends on them:**
   - `verdict(..., correction_k=None)` (default): behaves exactly as today — a regression pin on the
     one existing call site, asserted both by passing `correction_k` explicitly equal to
     `len(family)` (should match the omitted-parameter path) and by omitting it (per
     `data-scientist`'s stated test obligation, "assert both the passing and the omitted path").
   - `verdict(..., correction_k=16)` (or another value `!= len(family)`) with an `alpha_step` that
     would fail today's `family`-derived precondition but is valid at `k=16`: succeeds, and Rule 7's
     floor check still fires correctly against `resolving.observable_floor` (unaffected by
     `correction_k`).
   - `mean_bootstrap_interval(values, B=..., seed=..., levels=..., support=(0.0, 1.0))`: clamps the
     *result* directly to `support` (never to a difference-support conversion); refuses `len(values)
     < 2` and a non-finite value, mirroring `paired_bootstrap`'s/`continuous_verdict`'s own refusals;
     `support=None` leaves the interval unclamped; a dedicated test confirms the shared resample
     engine (factored out of `paired_bootstrap`) is the same code path, not a second implementation
     (e.g. via a monkeypatched/instrumented RNG call count, or by asserting bit-identical output at a
     fixed seed against a hand-computed expectation).
1. `_metric_value`: `BinaryMetric` → `.rate`; `ContinuousMetric` → `.mean`; `DistributionSummary` →
   raises; no aggregate for the metric → `None`.
2. `_better`/`_LOWER_IS_BETTER`: a normal (higher-is-better) metric ranks descending; a
   `_LOWER_IS_BETTER` metric ranks ascending; **mutation-tested both ways** (shrink the constant to
   `frozenset()` → a `falseAdvanceRate` ranking test reddens; widen it to include, say,
   `groundingRate` → a `chat-responder` ranking test reddens), per this component's own guard-
   constant convention.
3. `rank_report` with 3+ runs, one binary headline metric, no `reference`: rows sorted correctly,
   each with `k/n`/rate/Wilson CI, the pack's own restated caveat block-quoted at the top, latency
   column populated from `RunResult.latency`, footprint column reading `—` when `footprints` is
   `None`/omits the key and the supplied string when present. A separate fixture with one run whose
   `latency is None`: that row's latency cell reads `—`, the rest of the table is unaffected.
4. Continuous headline (`embedder`/`mrr`): rows sorted by `.mean` descending; the CI column calls
   `stats.mean_bootstrap_interval` with `support=(0.0, 1.0)` (from `ContinuousMetric.support`) and
   renders its own stronger descriptive caveat (§3.1/§3.4 Q1) — a distinct footnote from
   `_DESCRIPTIVE_NOTE`, asserted by its own text-presence test so the two can't silently merge.
5. No-headline pack (`guard-judge`): renders **two** ranked tables, one per `verdictMetrics` member,
   each independently sorted and polarity-correct (this is the test that would have caught §2.4's
   defect had it existed in this new code path).
6. A model in `runs` with no aggregate at all for the target metric: excluded from that table, not
   ranked last, and not silently dropped from the report (present in a version/hash banner if its
   pack version differs, exactly as `compare_report` already handles this).
7. `_aggregate_item_mismatches`-excluded arm: same "excluded and named" block `compare_report`
   already renders, reused, asserted via a fixture identical in shape to `compare_report`'s own
   existing exclusion tests.
8. Duplicate `modelKey` in `runs`: `rank_report` raises `DuplicateModelInReport` — never silently
   drops one (mirrors `PairedOutcomes.from_units`'s duplicate-unit-id backstop test shape).
9. `reference` given, present, a single-metric pack (e.g. `chat-responder`), 3+ candidates: **one**
   Holm ladder over `N-1` p-values, `correction_k=N-1` passed to each `verdict()` call, `_decision()`
   states match a hand-computed `holm_steps` table, at least one fixture reaching each of the four
   decision states (`distinguishable`, `not distinguishable`, `not tested`, `no verdict — no paired
   data`).
10. `reference` given, `guard-judge` (no headline, two metrics): **one combined** Holm ladder over
    the flattened `2·(N-1)` p-values (§3.2.2) — assert `holm_steps` is called exactly once, with a
    p-value list interleaving both metrics' candidates in the documented `(metric, candidate)` order,
    `correction_k=2·(N-1)` passed to every `verdict()` call, and the two rendered tables' Holm
    thresholds match what a hand-computed 32-entry ladder produces (**not** two independent 16-entry
    ladders — a regression test for this plan's original, rejected default). Each table correctly
    polarity-labelled via `_better` — this is also the test that most directly exercises §2.4's
    fix-in-new-code-only design: assert a candidate with a *higher* `falseAdvanceRate` than the
    reference is labelled "not better," never "better."
11. `reference` named but absent from `runs` (or absent from this session): a usage-shaped error at
    the `rank_report`/CLI boundary — decide at implementation time whether this is `rank_report`'s
    own raise or `_cmd_rank`'s pre-check (§3.3 already designs it as `_cmd_rank`'s, mirroring
    `UnknownModelKey`) and pin whichever is chosen with a test.
12. Every ranked table's rows print with the `<!-- rank-report: ... -->` comment lines Unit C
    depends on — pinned by a dedicated test, including the guard-judge fixture emitting **two**
    marker lines (§3.5), since Unit C has no other contract to test against.
13. Version/hash/schema-mismatch banners still fire across a 17-model ranked table exactly as they
    do across two arms (reusing the existing set-based checks needs no new logic, but deserves an
    N-arm-scale fixture, not just the existing two-arm ones).
14. The no-`--reference` resolving-power sentence (§3.4 Q4): printed whenever the ranked table has
    ≥2 models; `k` is `N-1` for a single-metric pack and `2·(N-1)` for guard-judge; no specific model
    name appears in the sentence; and it is printed **alongside**, not instead of, the pack's own
    unmodified `resolving_power_line` single-comparison sentence — a fixture asserts both sentences
    are present and distinct.

**Unit B (`cli.py`) — coder-authored tests, not necessarily test-first, but exhaustive over the
mechanical surface:**

- `rank` subcommand argument parsing (each flag, required vs. optional).
- `_select_rank_arms`: session filter, dedupe-keeps-newest (mirror the existing `--models` dedup
  test shape in `tests/test_cli.py`).
- `_rank_report_path`: same-day sequence numbering, `-rank-` infix distinguishes it from `compare`'s
  own same-day files for the same pack.
- Exit codes: `0` on a normal render (whatever the ranking, per this tool's own "no score-driven
  exit code" rule); `2` on an unknown `--reference` key or a malformed `--footprints` file; `4` on
  an invalid pack manifest (reusing `_cmd_compare`'s existing path).
- `--footprints` passthrough: a JSON file with one string entry renders exactly that string in the
  footprint column; a JSON file with a **non-string** value under a modelKey (a number, object, or
  array) renders that value's `str()` — never an exception; a missing key renders `—`; a missing
  `--footprints` flag renders `—` everywhere; an unparseable/non-object JSON file is the one
  malformed-*file* usage error (exit 2).

**Unit C (`consolidate_sweep_reports.py`) — light fixture tests:** two or three stub rendered-report
files with hand-written `<!-- rank-report: ... -->` lines; assert the generated index table's rows
match; assert the narrative section emits exactly one `<!-- TODO -->` placeholder per role, no more,
no fewer (a missing placeholder is a silently-incomplete consolidated document; an extra one is a
role that was never supposed to exist).

**Behavior-contract clarity vs. judgment split, restated:** Unit A's rendering/selection logic is
contract-clear (§3.1–3.2 fully specify shapes and reuse points) but edge-case-dense — test-first
discovery earns its keep. Unit B and C are mechanical wiring with no statistical judgment calls —
`coder` can execute both directly from this plan's file/function list without re-deriving anything.

## 6. Risks & open questions

- **`stats.verdict()`'s new `correction_k` parameter (§3.2.1) is a shared-machinery change and
  needs its own review pass**, independent of Unit A's other tests — it is small, additive, and
  defaulted to reproduce today's behavior exactly, but it touches the one function this module's
  mutation-testing bar is strictest about (Rule 7's floor enforcement). Both reviewers independently
  found the underlying conflict and `data-scientist` decided the resolution (§3.2.1); the
  implementation still needs its own scrutiny for correctness, not just design agreement.
- **The §2.4 polarity defect is a known landmine independent of this feature.** If `stats.verdict()`
  itself is ever asked to render a guard-judge verdict that reaches significance (possible under
  FR-8's own reference-anchored family, since it multiplies the effective sample of comparisons run
  against a designated reference), the **existing, unmodified** two-arm `compare` command could
  print a backwards "better" claim for the first time, on data this very sweep produces. This plan's
  own new code avoids the defect (§3.1.1, §3.2.2), but a `compare --models
  <candidate>,<reference> --session <sweep-session>` run against guard-judge, done by hand later,
  would not be protected. Worth a line in whatever briefs the operator running ad hoc `compare`
  invocations against this sweep's data. A `docs/BACKLOG.md` entry for this has been drafted and
  handed to `teco` to add (`architect` cannot write outside `docs/plans/`).
- **AC-3's literal wording (`./run.sh compare --session <id>`) does not match this plan's `rank`
  verb** (`analyst` review §2.3) — `compare` stays fixed at two arms by design (§2.1), so AC-3 as
  written cannot be satisfied by the command it names. `teco` is dispatching `tico` to amend AC-3 to
  name `rank`; no further action needed here (§3.3's "AC-3 reconciliation" note).
- **Footprint data provenance is genuinely unscoped here** (§3.3) — whoever executes U6 needs a
  source for 19 models' estimated on-disk footprint at their configured quantization; this plan only
  specifies the plumbing, not the research.
- **FR-8's "one designated reference model" being `qwen/qwen3-4b-2507`** is the requirements doc's
  own recommendation, not a hard requirement — the CLI's `--reference` flag makes this an operator
  choice at report-render time, so no part of this design hard-codes it.
- **`rank_report`'s comment-line contract (§3.5) is a new inter-tool interface** (Unit A produces it,
  Unit C consumes it) that no existing test protects beyond this plan's own §5. It now has a
  variable per-file count (one marker for four packs, two for guard-judge) — §5's behavior 12 and
  Unit C's own fixture (§5) are what pin both the format and the count, so a rename of the comment's
  field names, or a regression back to an assumed 1:1 file→marker mapping, needs those tests to
  catch it.
- **Q3's resolution (one combined `2·(N-1)`-size ladder for guard-judge) is a real power cost, not
  a free correctness fix** (§3.2.2) — accepted as the honest price of FR-8's own "one family per
  pack" wording applied to the one pack with two co-equal metrics, not a design choice this plan
  could avoid without under-correcting.
