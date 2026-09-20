# Small-Model Catalog Sweep — Report-Generation Plan Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M9)

## 1. Scope & verdict

Reviewed `model-bench/docs/plans/small-model-catalog-sweep.md` (Track A only — the report-
generation feature, FR-6 through FR-12) against `model-bench/docs/requirements/small-model-catalog-
sweep.md` (Status: Ready for design) and against the actual source it claims to build on:
`modelbench/report.py`, `modelbench/stats.py`, `modelbench/cli.py`, `modelbench/packs.py`,
`modelbench/results.py`, `modelbench/scoring/classification.py`, and `model-bench/AGENTS.md`. Track B
(the live 70-run sweep) is out of scope here, per the brief. This review does not duplicate
`data-scientist`'s parallel U2b methodology review (`docs/reviews/small-model-catalog-sweep-ml.md`)
of the four flagged statistical questions (§3.4 Q1–Q4) — where a finding below touches that ground I
say so and stop short of adjudicating it.

**Verdict: needs changes.** One blocker (§2.1) means Unit A cannot be implemented as designed without
either an unaddressed decision or touching `stats.py` machinery the plan explicitly disclaims
touching. Two more findings (§2.2, §2.3) are correctness/consistency gaps in the plan itself, not
statistical-validity questions, so they don't overlap `data-scientist`'s lane. Everything else the
plan claims about the existing codebase, I independently verified as accurate (§4).

**CPG:** considered, not relevant — queried `GRAPHS` directly: `cpg_falkorchat`, `kaizen_team`,
`reference`, `ws:*` are loaded; no `cpg_model-bench` graph exists. This review's scope (five source
files read in full/substantially, plus the requirements/plan docs) needed no call-graph tooling
beyond direct reading, matching the plan's own CPG note.

## 2. Findings

### 2.1 [BLOCKER] `stats.verdict()`'s own precondition rejects the candidate-axis Holm ladder §3.2 requires — will raise `ValueError` under the sweep's actual scale

**Route: `architect` (design rework), with a code-contract question for whoever lands Unit A.**

`stats.verdict()` (`modelbench/stats.py:1146-1220`) enforces, unconditionally:

```
if metric_name not in family:                                                   # :1185-1188
    raise ValueError(...)
if abs(resolving.alpha_mdd - resolving.alpha_family / len(family)) > 1e-12:     # :1189-1195
    raise ValueError(...)
if alpha_step is not None and not (resolving.alpha_mdd - 1e-12 <= alpha_step
        <= resolving.alpha_family + 1e-12):                                     # :1207-1219
    raise ValueError(...)
```

`family` here is **the pack's own pre-registered metric family** — the *only* existing call site
(`report.py:1236-1294`, the homogeneous-binary path §3.2 says the new code reuses) passes
`family=pack.metrics.verdictMetrics` and `resolving=stats.resolving_power(..., alpha_mdd=
pack.metrics.alpha_mdd)`, where `pack.metrics.alpha_mdd == pack.metrics.alpha_family / len(
pack.metrics.verdictMetrics)` by construction (`packs.py:77-95`). `len(family)` is always 1 or 2 for
every shipped pack (§2.3's own table). The Holm ladder at that call site runs **across metrics**
(`p_values` has one entry per metric in `family`, `steps = stats.holm_steps(p_values, alpha=
pack.metrics.alpha_family)`), so every `step.threshold` it produces is guaranteed to land in
`[alpha_mdd, alpha_family]` — the two axes (the ladder's multiplicity and `family`'s length) are
literally the same variable.

FR-8's family (§3.2) needs a **different, larger** axis: one Holm ladder across **N-1 candidates**
for one fixed metric. §3.4 Q4's own recommended default is `resolving_power(..., alpha_mdd=
pack.metrics.alpha_family / (N-1))` — i.e. `resolving.alpha_mdd` scaled to the *candidate* count, not
the metric count. But `verdict()`'s own precondition requires `resolving.alpha_mdd == resolving.
alpha_family / len(family)`, and `family` must also satisfy `metric_name in family` — `metric_name`
is a string like `"falseAdvanceRate"`, never a candidate's `modelKey`. There is no `family` argument
that is simultaneously (a) a real metric family containing `metric_name`, (b) length `N-1`, without
degrading into a padded/duplicated list that violates this module's own guard-constant conventions
(AGENTS.md "A guard's reach lives in an asserted constant") and would print a nonsensical
`{len(family)}-member verdict family` error message if it ever tripped.

Concretely, for the 17-model chat/vlm sweep, N-1 = 16 candidates against one reference, but every
in-scope pack's metric family has `len(verdictMetrics)` = 1 or 2. Holm's tightest candidate-ladder
step is `alpha_family/16 = 0.003125`; if `resolving.alpha_mdd` is left at the pack's *own* metric-
level value (0.05 or 0.025), every candidate-ladder `alpha_step` below that value raises the third
check above. If instead `resolving.alpha_mdd` is scaled to `alpha_family/16` (Q4's own recommendation)
and `family` stays `pack.metrics.verdictMetrics` (length 1 or 2), the second check now fails instead
(`0.003125 != 0.05/2`). **Either way, the very first candidate not ranked last in the ladder raises.**
This is not a hypothetical edge case — it fires on ordinary, in-scope data at the sweep's own stated
scale.

**Verified, not inferred:** I read `stats.verdict()`'s three preconditions (`:1185-1219`) and the
*only* existing call site that builds `family`/`resolving` together (`report.py:1236-1268`), and
traced both axes (metric-count vs. candidate-count) through the actual numbers the plan's own Q4
recommendation would produce for this sweep. This is a code-contract conflict, not a statistical-
validity question — it's why I'm not treating it as `data-scientist`'s Q3/Q4 territory, though it's
adjacent to both.

**Suggested resolution (for `architect` to pick, not for me to decide):** either (a) design a
low-level entry point in `stats.py` that keeps Rule 7's floor logic but takes an explicit correction
size decoupled from `family`'s membership check — a scoped, reviewed change to shared machinery,
exactly the kind of change §2.4 already flags as needing "its own dedicated review given this
module's mutation-testing bar" — or (b) have the new family renderer compute the McNemar p-value,
the Rule 7 floor check, and the winner label directly from `PairedOutcomes`/`mcnemar_exact`/
`resolving_power.observable_floor` without going through `verdict()` at all, accepting that this is a
second, smaller, deliberately narrower reading of Rule 7 (and needs its own mutation tests, since it
duplicates a small slice of `verdict()`'s logic rather than reusing it). §3.2's current text ("using
the existing primitives exactly as `compare_report`'s homogeneous-binary path already does... never a
second copy of that arithmetic") rules out neither option explicitly and doesn't acknowledge the
conflict exists.

### 2.2 [MAJOR] §3.1 and §3.5 disagree on how many marker lines a guard-judge report emits

**Route: `architect`.**

§3.1 (`rank_report`'s design): "Per pack with none [headline metric] (`guard-judge`): **two** ranked
tables, one per `verdictMetrics` member". §5 test 5 pins this explicitly ("renders **two** ranked
tables"), and test 12 says every ranked table emits its own `<!-- rank-report: ... -->` comment line
— so guard-judge's *one report file* carries **two** marker lines.

§3.5 (the consolidation script's parsing contract): "the extractor never depends on markdown table
formatting... it reads **five fixed comment lines**, not five tables" — describing exactly one marker
line per report file, for five files.

These are inconsistent: `consolidate_sweep_reports.py --reports <path1.md> ... <path5.md>` takes five
*file* paths (one per pack, correctly), but guard-judge's file contains two marker lines, not one, so
the extractor must handle a variable count of markers per file (1 for four packs, 2 for guard-judge)
— six markers total across five files, not five. This isn't cosmetic: it changes the index table's
row count (FR-9's "list...each pack's top-ranked model(s)" already anticipates the plural for
guard-judge; the consolidation design's prose doesn't). An implementer following §3.5's literal "five
fixed comment lines" would build a 1:1 file→row extractor and then have to special-case guard-judge
anyway, or silently drop one of its two verdict metrics from the index.

**Suggested fix:** revise §3.5 to state the parsing contract as "one or more marker lines per report
file, one per rendered table" and show the guard-judge case in the worked example, so Unit C's
implementer designs for the variable count from the start rather than discovering it mid-
implementation.

### 2.3 [MAJOR] AC-3's literal command name doesn't match the plan's chosen CLI verb

**Route: `architect` to reconcile with `tico` (the requirements doc's owner), or amend the plan to
call this out explicitly.**

Requirements doc AC-3: *"Given the sweep is complete, **when** `./run.sh compare --session
<sweep-session-id>` is run for each of the five packs, **then** each renders a comparison report to
`reports/` covering every in-scope model with a stored result for that pack."* This literally names
the existing `compare` verb.

The plan's §2.1 finding (confirmed independently, §4 below) is that `compare`/`_comparison_pair`
is fixed at exactly two arms — `runs[0], runs[1]` unconditionally — and §3.3 explicitly rejects
extending `compare` in favor of a **new** `rank` subcommand. That's a sound design call on its own
terms, but the plan never reconciles it against AC-3's literal wording: running
`./run.sh compare --session <id>` for a 17-model in-scope pack, exactly as AC-3 says, would still only
compare two of those models, not "every in-scope model with a stored result" — the acceptance
criterion as written is not satisfiable by the design as designed. Nothing in the plan flags this;
§2.1 reconciles the *other* AC-adjacent mismatch in the same requirements doc (FR-8's `run
--reference` assumption) but doesn't apply the same scrutiny to AC-3.

This matters because AC-3 is exactly the kind of literal, executable acceptance criterion
`qa-engineer` would run verbatim at acceptance time. Left as-is, it's a nearly-certain gate failure
or a confused re-interpretation at that point, for a design decision that was actually correct.

**Suggested fix:** the plan should add one sentence stating that AC-3's `compare` reference is
superseded by the new `rank` command and flag it for `tico` to amend the requirements doc's AC-3
wording (or the plan itself, since `architect`'s write access is scoped to `docs/plans/`, could at
least name the discrepancy explicitly the way §2.4 already models for the polarity defect —
"I cannot write the requirements doc myself; whoever lands Unit A/B should raise this with `tico`").

### 2.4 [MINOR] `--footprints` malformed *values* (not just a malformed file) are unaddressed

**Route: `coder` (Unit B), informational.**

§3.3 says a malformed `--footprints` *file* is a usage error (exit 2), and a missing *key* renders
`—`. It doesn't say what happens to a **present but non-string value** (e.g. a JSON number, object,
or array under a modelKey) — the stated intent ("never parsed for a number... can only ever produce a
missing/wrong-looking display cell") suggests it should render as-is, but a dict/array value passed
straight into a markdown table cell can break the table's own formatting (embedded `|` or newlines),
not just look "wrong." This is low-stakes (worst case is a garbled cell in one report, not a
computation error), but §5's Unit B test list doesn't include a case for a non-string footprint value,
so nothing currently pins the behavior either way.

**Suggested fix:** one line in §3.3 saying a non-string value is coerced to its `str()` (or rejected
at load time as a second usage-error shape) plus one test in Unit B's list.

### 2.5 [MINOR] Ranked table's latency column has no stated behavior for `run.latency is None`

**Route: `tdd-engineer` (Unit A), informational.**

`RunResult.latency` is optional (`results.py:712`, `None` for a deterministic arm or any run whose
timing was never wired). `_render_speed` (the function §3.1 says the ranked table's latency column
reuses conceptually, though not by direct call) explicitly guards this per-run and per-field
(`report.py:920-944`, prints `—` throughout). §3.1's own column list doesn't state what the ranked
table's latency cell shows when a *model* arm's `latency` is `None` — plausible today only for an old
or synthetic fixture, not a live sweep run, but §5's test list has no fixture for it, unlike
`_render_speed`'s explicit `[]`-when-absent guard.

**Suggested fix:** add one line to §3.1 stating the ranked table treats a `None` latency the same way
`_render_speed` does (`—`), and one fixture to §5's Unit A list.

## 3. What's solid

- **Grounding is excellent everywhere I independently checked it.** Every specific code citation
  I verified against the actual files matched: `_comparison_pair`'s unconditional `runs[0], runs[1]`
  (`report.py:600`); `RunConfig.referenceKey` accepted but read nowhere (confirmed via `grep -rn
  referenceKey` — exactly the two hits the plan names, plus one unrelated test default); the
  `falseAdvanceRate`/`falseSuspendRate` "stores the error itself" claim (`classification.py:209-210,
  259`, `count = int(not advanced) if metric == "falseSuspendRate" else int(advanced)`); `stats.
  verdict()`'s polarity-blind `winner, loser = (a_label, b_label) if diff >= 0 else (b_label,
  a_label)` (`stats.py:1320`, `diff = (b - c) / n` at `:1223`) — this is a real, latent defect exactly
  as described, and the plan's own new code is correctly designed to avoid inheriting it (§3.1.1's
  `_LOWER_IS_BETTER`/`_better`, §3.2's explicit refusal to reuse `Verdict.text`). `BinaryMetric.rate`/
  `ContinuousMetric.mean`/`DistributionSummary`'s median+p10-no-mean shape, `_decision()`'s four-state
  vocabulary, `_render_role_caveat`'s chat-responder-only scope, `_select_arms`'s dedup-keeps-newest
  semantics, `_report_path`'s two-digit sequence scheme, and the pack-level facts in §2.3's table
  (recall@10 = 37/38, nlq n_eff=34, tool-caller n=12) all checked out against the cited sources.
- **FR-6, FR-7 (modulo Q4), FR-9, FR-11, FR-12 all have concrete, buildable answers**, each reusing
  an identified existing primitive rather than inventing a parallel one.
- **The polarity-defect discovery (§2.4) is a genuinely valuable, independently-confirmed find** —
  correctly scoped as out-of-fix-scope for this feature, correctly designed around in the new code,
  and correctly routed to `docs/BACKLOG.md` (with the honest caveat that `architect` can't write
  there itself).
- **Unit sizing and routing are sound.** Three units, ≤3 sequentially-dependent steps, Unit A's
  edge-case density genuinely favors `tdd-engineer`'s test-first discovery (matches this file's own
  ~3000-line precedent, confirmed by `wc -l`), Units B/C are mechanical and correctly routed to
  `coder`. Unit C's "lighter test bar" precedent (`scripts/refresh_golden.py`) exists and is real.
- **The consolidated-document design (§3.5) correctly respects the no-cross-pack-arithmetic
  invariant** — zero arithmetic in the extraction script, narrative left to a human/agent judgment
  call rather than templated into a false-precision sentence.

## Pass 2 — 2026-09-19

Re-reviewed `docs/plans/small-model-catalog-sweep.md` **Version 2** (not this summary, not Pass 1's
memory) against the live source, alongside `data-scientist`'s `docs/reviews/small-model-catalog-
sweep-ml.md` (verdict: needs changes, Q1–Q4) and the requirements doc's own committed fix.

**Verdict: approve.** All five Pass-1 findings are resolved, re-derived independently rather than
taken on the plan's word (details below). No new blocker/major surfaced from either the fixes
themselves or their interaction with `data-scientist`'s Q3 resolution. Two nits, neither gating.

**Disposition of Pass 1 findings:**

1. **[BLOCKER] `stats.verdict()` precondition conflict — fixed, re-derived independently.** Plan
   §3.2.1's new `correction_k: int | None = None` keyword changes precondition 3 to `k =
   correction_k if correction_k is not None else len(family)`, checked against `resolving.alpha_mdd
   == alpha_family / k`. I re-traced this against the actual precondition code
   (`stats.py:1185-1219`): `family`'s only remaining job is the unchanged `metric_name in family`
   membership check, fully decoupled from the correction-size math; `correction_k=None` reproduces
   `len(family)` exactly, matching the one existing call site's behavior unchanged
   (`report.py:1236-1294`, which never passes it). I independently re-read `observable_floor`'s
   construction (`stats.py:836-844`, `ResolvingPower.observable_floor` built at `:941` from
   `alpha_family`/`n_eff` alone) to confirm Rule 7's floor enforcement is untouched by `correction_k`
   — confirmed, it shares no variable with it. Also checked composition with `data-scientist`'s Q3
   combined ladder (§3.2.2's `correction_k = len(p_values)`, 16 or 32 at this sweep's N): `correction_k`
   is a bare, shape-agnostic `int`, so it carries either value with no further change needed —
   composes correctly.
2. **[MAJOR] §3.1/§3.5 marker-count inconsistency — fixed.** §3.5 now states explicitly "the
   guard-judge file carries two [markers]... six markers total, not five"; §4 Unit C step 1 and §5
   test 12 both updated to match, and Unit C's fixture list now names a guard-judge-shaped stub.
3. **[MAJOR] AC-3 literal-command mismatch — fixed, verified against the committed requirements doc
   directly, not the plan's claim about it.** `docs/requirements/small-model-catalog-sweep.md:147-150`
   now reads `./run.sh rank --pack <pack-id> --session <sweep-session-id>`, with a dated 2026-09-19
   decision-log entry (`:200-201`) citing this review by name. Plan §3.3's "AC-3 reconciliation" note
   is consistent with the corrected doc.
4. **[MINOR] footprint non-string value — fixed.** §3.3 specifies `str()` coercion at load time in
   `cli.py`, never an exception; §5's Unit B test list adds the case.
5. **[MINOR] latency-`None` handling — fixed.** §3.1 states the `None` → `—` behavior explicitly,
   matching `_render_speed`'s own per-field guard; §5 test 3 adds the fixture.

**New observations this pass (nits, not gating):**

- The plan's own risk section flags `correction_k` as needing "its own review pass... given this
  module's mutation-testing bar" but doesn't create a distinct coordination-ledger unit for it —
  in practice this is moot now, since both `analyst` (this pass) and `data-scientist` have already
  scrutinized the mechanism in design detail before any code exists; `teco`'s standing U4 code-gate
  review covers it. Not worth a ledger change on its own.
- Unit A's file list is now `stats.py`, `report.py`, `test_stats.py`, `test_report.py`, plus a
  one-line docstring touch in `classification.py` — five files, at this component's own sizing
  convention's edge but not past it, and the added file (`test_stats.py`) is a natural consequence of
  `stats.py` now changing at all. Not a resizing recommendation.

I found nothing else needing changes.

## 4. Open questions

- **§3.4's Q1–Q4 are `data-scientist`'s to resolve**, per the brief. Of the four, Q3 (one combined vs.
  two independent Holm ladders for guard-judge) is the one I'd flag as carrying the most risk if its
  provisional default ships unreviewed: two independent per-metric ladders under-corrects the true
  family-wise error rate across `2 × (N-1)` tests, which is the exact shape of overclaim this module's
  five honesty rules exist to prevent. That said, the coordination ledger (`small-model-catalog-sweep-
  coordination.md`) already gates U3 (implementation) behind U2b (`data-scientist`'s review) landing
  first, so none of Q1–Q4's defaults will actually be implemented against before review — this
  mitigates the risk the brief asked me to check for. None of the four strike me as *unsafe to leave
  provisional* given that gate; §2.1's precondition conflict is a separate, code-level question that
  sits beneath all four and needs resolving regardless of how Q1–Q4 land.
  I flag this as a *note*, not a duplicate finding.
- **Whether §2.1's fix should extend `stats.py` or duplicate a slice of `verdict()`'s logic** is a
  real design fork this review surfaces but doesn't resolve — it's `architect`'s (and likely
  `data-scientist`'s, since either path touches Rule 7) to decide before Unit A starts.

## Pass 3 — Consolidated document, final gate — 2026-09-20

Reviewed `reports/catalog-sweep-2026-09-19-consolidated.md` — the FR-9/FR-10 consolidated
document `data-scientist` just produced from the five gated per-pack reports
(`reports/*-rank-20260920-02.md`) — against `docs/requirements/small-model-catalog-sweep.md`'s
FR-9, FR-10, FR-11 and its acceptance criteria. This is the coordination's final gate: I checked
the assembly and narrative, not the underlying report correctness (already double-gated by two
prior review rounds, `docs/reviews/small-model-catalog-sweep-impl.md`).

**Verdict: approve with suggestions.** No blocker. One major finding (§ below) in the exact spot
the brief asked me to scrutinize hardest — the guard-judge section's explanation for sidelining its
reference-anchored family table. Everything else I checked (FR-9's no-cross-pack-arithmetic rule,
the resolving-power table's verbatim-copy discipline, all five FR-11 restated caveats, the index
table's byte-for-byte fidelity to each report's marker line, and roughly two dozen specific numeric
claims spot-checked line-by-line against the five source reports) held up.

**CPG:** not applicable — this review is of a generated markdown report's narrative accuracy
against its own cited source reports and the requirements doc; no code changed and no call-graph
question was in scope.

### Findings

**[MAJOR] Guard-judge section mischaracterizes why its reference-anchored family's diff column is
sidelined — the column is not unreliable, it's already the fixed rendering.**

The guard-judge subsection (lines 100-109) says the family table's diff column shows a sign
"reversed" relative to marginal per-arm rates and calls this "consistent with the documented,
non-blocking `stats.verdict()` polarity-blind-wording caveat" — implying the column is suspect and
justifying avoiding it. I confirmed the raw observation is correct: I cross-checked every row of
both `falseAdvanceRate` and `falseSuspendRate` family tables against the per-arm rate table
(`reports/guard-judge-understanding-rank-20260920-02.md:7-24,60-77` vs. `:38-54,91-107`) and the
sign is indeed flipped relative to a naive `candidateRate - referenceRate` in every non-zero row
(e.g. `gemma-3-4b-vl-it-...`: candidate 0.950 vs. reference 0.100 = +85.0pp marginal, family shows
-85.0pp; `mistralai/ministral-3-3b`: 0.000 vs 0.100 = -10.0pp marginal, family shows +10.0pp). I
also confirmed this reversal is specific to guard-judge's two `_LOWER_IS_BETTER` metrics and does
*not* occur on the other three packs' higher-is-better metrics — exactly as the narrative claims.

But the cause is not the live bug. `report.py:1226-1243`'s `_polarity_corrected` — the function
that builds this exact family table — was written specifically to *avoid* inheriting
`stats.verdict()`'s polarity-blind `winner, loser` labeling for `_LOWER_IS_BETTER` metrics
(`report.py:122`: `_LOWER_IS_BETTER = frozenset({"falseAdvanceRate", "falseSuspendRate"})`): for
these two metrics it deliberately applies **no flip**, because the raw `diff = (b-c)/n`
(`stats.py:1305`, `a`=reference, `b`=reference-only-correct, `c`=candidate-only-correct) already
reads `positive = candidate ahead` once you account for `scored_outcome`'s `True` meaning "the bad
event happened." That's exactly why it differs in sign from a naive rate subtraction — it's the
*correct*, already-applied fix, not a defect. The plan that shipped this code says so directly:
`docs/plans/small-model-catalog-sweep.md:119-121` — "the new ranked-table/family code... is
polarity-aware from the start... it does not touch or reuse `Verdict.text`" — and the `BACKLOG.md`
entry itself (`:108-111`) scopes the live bug to `Verdict.text`/`stats.verdict()`'s generic winner
text, citing this very plan section as "a scoped design that avoids the defect in new code."

The consequence is not a wrong claim about any model's standing (the section's ultimate
recommendation — `qwen/qwen3-4b-2507` or `nemotron-3-nano-4b` as the only two balanced picks — is
unaffected either way), but it does cause the section to silently discard real evidence: once the
sign convention is understood correctly (positive = candidate better), the family tables show 2
Holm-confirmed `distinguishable` verdicts on `falseAdvanceRate` and 8 on `falseSuspendRate` — real,
certified findings the chat-responder section below (lines 185-193) *does* use from its own,
correctly-signed family table ("does add one certified finding... Holm-confirmed distinguishably
worse than the reference"). Guard-judge's narrative could have made the identical move and didn't,
for a reason that turns out not to hold.

**Suggested fix:** rewrite lines 100-109 to state the true mechanism — `_polarity_corrected`
already reorients the diff for `_LOWER_IS_BETTER` metrics to "positive = candidate better," which
is why it doesn't match a naive rate subtraction — and then either (a) cite the family's actual
`distinguishable` rows as additional certified evidence, matching the chat-responder section's
pattern, or (b) if the two-metric family is judged harder to read than the per-arm table for other
reasons, state that reason honestly instead of implying the column is unreliable.

**[MINOR] Embedder section's latency-ratio range slightly undersells granite's actual speedup.**
"3-4x lower p95 latency" (line 64) describes both `nomic-embed-text-v1.5` (17ms) and
`granite-embedding-278m-multilingual` (12ms) against the top model's 55ms. `55/17 ≈ 3.2x` fits, but
`55/12 ≈ 4.6x` doesn't — the true range is closer to "3-5x." Low-stakes (doesn't change the
section's conclusion that both are much faster), but worth a one-number fix.

### What's solid

- **FR-9's hard rule holds everywhere.** Read every sentence of all five "Insights and
  recommendations" subsections plus the preamble; no cell, column, or sentence combines a score
  from more than one pack anywhere in the document.
- **The resolving-power table (lines 18-25) and the index table (lines 35-42) are both genuinely
  verbatim.** I checked all 6 resolving-power rows and all 6 index rows against each source
  report's own `resolves differences of >=...` sentence and `<!-- rank-report: ... -->` marker line
  respectively (`reports/*-rank-20260920-02.md`) — byte-for-byte matches throughout, zero
  arithmetic performed across rows.
- **All five FR-11 caveats are genuinely restated, not cited.** Embedder's recall@10 ceiling,
  guard-judge's two-class-conditional shape, nlq's true n=34 denominator, tool-caller's
  underpowering, and chat-responder's grounding-is-not-quality distinction each appear as full,
  paraphrased sentences carrying the same content as the source report's own caveat, not a "see
  report" pointer.
- **Numeric accuracy is otherwise excellent.** Spot-checked roughly two dozen specific claims —
  latency ratios, footprint ratios, CI values, pp gaps, k/n counts, the "only two models under
  0.20 on both guard-judge rates" claim (a nontrivial 16-row × 2-table cross-reference, verified
  correct), the tool-caller 7-of-16/9-of-16 split, nlq's five-perfect-scorers-on-different-bases
  breakdown — against the five source reports directly, not the document's own citations. All
  checked out except the one minor latency-ratio rounding above.
- **The preamble (lines 3-13) states plainly, before any per-role data, what this sweep's sample
  sizes can and cannot prove**, in language that tracks the acceptance criterion's own wording
  closely enough to leave no ambiguity about intent.

### Open questions

- None. The major finding above has a concrete, self-contained fix that doesn't require anyone
  else's input — `data-scientist` (or whoever revises the document) can apply it directly.

## Pass 4 — Fix verification, final gate — 2026-09-20

Re-read the revised `reports/catalog-sweep-2026-09-19-consolidated.md` guard-judge (lines 74-130)
and embedder (lines 62-66) sections against the two Pass-3 findings.

**Verdict: approve.** Both findings resolved, independently re-verified against
`reports/guard-judge-understanding-rank-20260920-02.md`'s two family tables — not taken on the
document's or the coordinator's word.

**Disposition of Pass 3 findings:**

1. **[MAJOR] Guard-judge mechanism mischaracterization — fixed.** Lines 100-106 now correctly
   attribute the unflipped diff to `report.py`'s `_polarity_corrected`/`_LOWER_IS_BETTER` handling,
   explicitly distinguished from "the unrelated, still-open `stats.verdict()` wording caveat in
   `docs/BACKLOG.md`" — matching the mechanism I traced in Pass 3 (`report.py:122,1226-1243`).
   I independently re-counted both family tables row-by-row: falseAdvanceRate has exactly 2
   `distinguishable` rows (`gemma-3-4b-vl-it-...`, `stablelm-zephyr-3b`, both -85.0 pp) and
   falseSuspendRate has exactly 8 (`google/gemma-4-e2b` -80.0, `mistralai/ministral-3-3b` -53.3,
   `mistralai_ministral-3-3b-instruct-2512` -53.3, `prism-ml/bonsai-27b` -86.7,
   `qwen/qwen3-4b-thinking-2507` -83.3, `qwen2.5-3b-instruct` -80.0,
   `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` -73.3, `stable-code-instruct-3b` -63.3) — 10
   total, no overlap between the two sets, matching the document's list exactly. I also verified
   the document's stronger claim — "every one of the seven models that led either metric's own top
   row is Holm-confirmed worse... on the other metric" — against all seven: true in every case. And
   I confirmed no row in either table shows a positive diff reaching `distinguishable` (i.e. no
   candidate is certified better than the reference), supporting "no row... shows
   `qwen/qwen3-4b-2507` certified worse than any candidate." The recommendation logic (§ item 3 of
   the brief) holds together: `qwen/qwen3-4b-2507` as the certified-stronger pick (10 confirmed
   worse, 0 confirmed better) and `nemotron-3-nano-4b` correctly described as untested against the
   reference (its own diffs, +0.0 pp and -3.3 pp, both fall in Holm's untested region) rather than
   disproven or confirmed similar.
2. **[MINOR] Embedder latency-ratio wording — fixed.** Line 64-65 now reads "3-5x lower p95
   latency (55 ms vs. 17 ms and 12 ms respectively — ~3.2x and ~4.6x)" — both multipliers check out
   (55/17 ≈ 3.24, 55/12 ≈ 4.58) and the stated "3-5x" range now covers both instead of undershooting
   the larger one.

**One non-blocking nit, new this pass:** line 105-106's clause "...which this family renderer never
calls" is ambiguous on a literal read — `_render_reference_family` *does* call `stats.verdict()`
(for the correctly-computed CI/p-value), it just never reuses `stats.verdict()`'s own
polarity-blind `Verdict.text` wording. The intended meaning ("never exhibits/reuses that wording
caveat") is recoverable from context and the surrounding sentence is otherwise accurate, so this is
a clarity polish, not a correctness issue — worth a one-word tighten ("never reuses" instead of
"never calls") if the document is touched again, not worth reopening it for.

I found nothing else needing changes.
