# Methodology review — `rank --reference` on a continuous verdict metric, implementation plan

> **Status:** archived · **Owner:** `data-scientist` · **Tracks:** —

**Reviewing:** `docs/plans/rank-continuous-reference.md` (architect), against `docs/plans/
rank-continuous-reference-ml.md` (this role's own prior method note, "`-ml`"), grounded against
the current `modelbench/stats.py`/`modelbench/report.py` tree (not either document's claims about
it). Scope: statistics/methodology only — `analyst`'s parallel pass covers completeness/line
numbers/code quality.

## Verdict: **approve with suggestions**

The plan implements `-ml`'s statistical design faithfully — I found no drift on the load-bearing
formulas (`correction_k` semantics, the combined divisor, the rendering shape, the two adjacent
defect fixes). Its resolution of open question (a) is a genuine, correctly-diagnosed gap, not
over-engineering, and its fix is the right shape. Its resolution of open question (b) reaches the
right operational outcome but over-states the argument for it — §3.5's "mechanically" framing
borrows the pre-registered-mixed-family's multiplicity rationale for a problem that is actually a
data-integrity failure, and the two are not the same kind of uncertainty. Neither of these findings
blocks implementation; the second is worth a documentation fix before or shortly after, so a future
reader doesn't inherit an argument that doesn't hold if it is ever leaned on again.

## 1. Faithfulness check — no drift found on the specified formulas

Read directly against the current tree (`modelbench/stats.py:1216-1301`, `1571-1769`;
`modelbench/report.py:98-242`, `1246-1486`, `1650-1930`):

- **§3.1 `correction_k`** — the plan's diff (`k = correction_k if correction_k is not None else
  len(family)`, inserted verbatim where `-ml` specifies, keyword position after `support` and
  before `a_label`) is character-for-character `-ml`'s own signature and body change. `verdict()`'s
  already-shipped `correction_k` (`stats.py:1216-1301`) uses the identical `k = correction_k if
  correction_k is not None else len(family)` shape at line 1270 — the plan's mirror is exact, not
  approximate.
- **§3.3/§3.2 the combined divisor** — `k = len(family) * len(candidates)`, computed directly with
  no per-metric ladder, exactly `-ml` §3.2's formula and its "general formula even when
  numerically indistinguishable from a per-metric one today" instruction (`-ml` §3.2's rejected
  alternative). Confirmed the plan does not special-case `len(family) == 1` away anywhere in step 7.
- **§3.6 the rendering shape** — four decision states (`distinguishable`, `not distinguishable`,
  `no verdict — no paired data`, `no verdict — one paired unit`), no Holm-step column, caption
  stating the numeric correction (`alpha_family=…, k=…, alpha_used=… (…% CI)`). `-ml` §3.4's own
  four bullet decision-state strings are reproduced verbatim in the plan's `_render_reference_
  family_continuous` (plan §4 step 5). The column header simplifies `-ml`'s symbolic `(1 −
  α_family/2k) CI` to a plain `CI` header with the numeric correction moved to the caption — this
  is not a deviation from `-ml`'s intent, it is exactly the caption-carries-the-number pattern
  `-ml` §3.4 itself demonstrates in its own worked example (`` `alpha_family=0.05, k=4,
  alpha_used=0.0125 (98.75% CI)` ``).
- **§3.2 `ContinuousVerdict.text`'s coverage label (`-ml` §3.6)** — `coverage_label = f"{100 * (1 -
  alpha_used):g}% CI"` replacing both literal `"95% CI"` occurrences, computed from the same
  `alpha_used` the dataclass already carries. Matches `-ml` §3.6's minimal fix exactly, including
  the `k=1` byte-identical / `k=4` → `98.75% CI` worked example.
- **Plan step 6, `_rank_resolving_power_lines` gating (`-ml` §3.5)** — "gate the call, not redesign
  the sentence" is implemented as a `return []` for a continuous metric, mirroring `-ml`'s own
  chosen fix and its rejected alternative (no invented continuous power-preview sentence). One
  genuine, small drift from `-ml`'s literal instruction, flagged in §4 below — not a correctness
  bug, but worth naming since the task asked for drift "however small."
- **`_polarity_corrected` reuse (plan §2.2)** — verified by reading `stats.mover_d_interval`
  directly (`stats.py:126-146`) and `continuous_verdict`'s own `diff = sum(diffs)/len(diffs)`
  (`stats.py:1730`): both are `a`-minus-`b` (reference-minus-candidate, since `_paired_diffs` is
  called `(reference_run, candidate, ...)`), and `_polarity_corrected` is pure sign/interval
  arithmetic with no kind-specific logic (`report.py:1226-1243`). The plan's claim that it is
  "directly reusable, unchanged" is correct — confirmed algebraically, not just by inspection: for
  `mrr` (not in `_LOWER_IS_BETTER`), a candidate with a higher raw MRR produces `cv.diff < 0`
  (reference minus a larger candidate), and `_polarity_corrected` flips it to positive — "positive =
  candidate is better," the stated convention, holds.

No drift found on any of the load-bearing statistical formulas. The implementation plan is faithful
to the method note on every point the note was prescriptive about.

## 2. Open question (a) — real gap, correctly diagnosed; the fix is the right shape

Traced `_metric_kind` directly (`report.py:229-242`): `metric = _metric_aggregate(a, name) or
_metric_aggregate(b, name)` — it returns `a`'s own declared kind whenever `a` declares *any*
aggregate for that metric, **never inspecting `b`'s declaration in that case**. This is not a
shape-extension question at all; it is a detection failure. Concretely: if `reference_run` declares
an aggregate for `mrr` at all (the normal case for every shipped pack), then `_metric_kind
(reference_run, cand, "mrr")` returns `reference_run`'s kind for *every* candidate regardless of
what any individual candidate itself declares — two candidates disagreeing with each other on
`mrr`'s kind would be invisible to `-ml`'s own literal §3.3 step 1 sketch, because that sketch never
lets a candidate's own declaration override or even register against another candidate's.

Worth stating plainly since it is my own note's gap: `-ml` §3.3 step 1's literal pseudocode (`kinds
= {metric: _metric_kind(reference_run, cand, metric) for metric in family for cand in candidates}`)
has a second, independent problem beyond the one the plan names — as a dict comprehension keyed
only by `metric`, each candidate's iteration *overwrites* the previous one's entry, so even a
by-construction cross-candidate check would need to accumulate a *set* per metric, not a single
dict value, to detect disagreement at all. The plan's chosen fix — a genuinely N-ary
`_resolve_reference_kinds` that inspects every arm's own declared aggregate directly, independent
of any pairwise anchoring — sidesteps both problems at once rather than patching the sketch. I
traced its logic against the diagnosis: for each metric, it collects `{modelKey: kind}` from every
arm (reference + every candidate) that declares an aggregate at all, and flags a metric only when
that collected set of kinds has more than one distinct value. This is the correct predicate for "do
all arms that speak to this metric agree" — not over-engineered, and not narrower than the actual
gap. **Verdict on (a): the plan's diagnosis is correct and its fix is proportionate — approved.**

## 3. Open question (b) — right outcome, an argument that doesn't fully hold as stated

`-ml` §5 left this open, leaning toward treating a genuine cross-arm kind disagreement as a
*sharper* failure (closer to the version/hash/schema banners) rather than assuming it must default
to the same "refuse whole" treatment `-ml` §3.3 specifies for a *pre-registered* mixed-kind family.
The plan's §3.5 resolves it as "both, on different axes": identical mechanical refusal, distinct
louder framing. **The outcome is right — the argument for its mechanical half is not as tight as
written.**

**What the plan's §3.5 claims:** tracing `-ml` §3.3's own rationale for refusing an ordinary mixed
family whole ("`k` is `len(verdictMetrics)` and pre-registered, so dropping the minority kind would
shrink `k` after the results exist and under-correct the survivors") and asserting "the identical
mechanical problem applies to a disagreeing metric."

**Why this doesn't transfer cleanly.** The two cases are different kinds of uncertainty, and the
"`k` shrinks" argument is *keyed to the multiplicity correction*, which is not actually the binding
constraint in the disagreement case:

- For an ordinary **pre-registered** mixed family, every metric's kind is a resolved, known fact —
  the problem is purely structural: there is no combined correction procedure that spans a
  Holm-stepped p-value ladder and a bootstrap-interval-widening family in one pass, because no such
  procedure is designed anywhere in this codebase. Refusing the family whole here is a scoping
  decision about an *undesigned mechanism*, not a forced mathematical consequence of `k`.
- For a genuine **cross-arm disagreement**, the binding constraint is different: the metric's own
  type cannot be established at all, so no instrument — binary or continuous — can be legitimately
  applied to it, independent of any multiplicity concern. Critically, `k = len(family) *
  len(candidates)` (§3.2's formula) is a pure count, computed from `family`'s and `candidates`'
  sizes — it does **not** depend on any metric's kind resolving cleanly. Nothing about `k`
  mechanically "shrinks" if you refuse only the disagreeing metric's own cells while still deciding
  a *sibling*, cleanly-resolved metric's ladder at the full, correctly-sized `k` — this codebase
  already has the mechanism for exactly that shape: `rank_report`'s existing binary combined ladder
  keeps a zero-paired-data `(candidate, metric)` cell *in* the combined `p_values` list (via
  `mcnemar_exact(0, 0) = 1.0`, `report.py:1467`, comment: "`k` is fixed by pre-registration, never
  by how much data arrived"), consuming a Holm rank without ever being decided. A disagreeing
  metric's cells could, in principle, take the identical treatment — refused individually, without
  refusing siblings or shrinking `k` for them.

So the plan's "mechanically requires the same whole-family refusal" is an overstatement: a
partial-refusal-with-preserved-`k` design is not mathematically ruled out the way the argument
implies. What actually forces (or at least strongly motivates) whole-family refusal for the
disagreement case is a *different* argument, closer to the one `-ml` §5 itself favored: a cross-arm
kind disagreement means the pack's own record is internally self-contradictory for that arm — a
data-integrity failure in the same family as the schema/version/hash mismatches
(`report.py:1396-1420`, "visible, never silent, never a reason to drop a record"), and once one
metric's type declaration is shown to be unreliable for some arm, there is a legitimate reason to
distrust the same arm's *other* declared aggregates enough to withhold the whole reference-anchored
verdict, not just the one broken cell. That is a conservative response to *epistemic* uncertainty
about the data, not a *multiplicity-correction* argument at all.

**Practical effect of this review finding: none on the implementation.** The plan's actual code
(step 7, the `reference_family_refused` flag) does the right thing either way — refuse the whole
family, and it does so with a distinct, louder banner exactly as `-ml` leaned. I am not asking for
a different behavior. I am flagging that **§3.5's stated justification should be rewritten before
or shortly after this unit lands**, so a future reader (an implementer revisiting §3.7/§3.8's
rejected alternatives, or a reviewer checking whether a partial-refusal design is safe) does not
inherit a "mechanically impossible" claim that isn't actually true. The corrected framing: whole-
family refusal for a genuine disagreement is a *conservative, deliberate response to a data-
integrity failure* (primary reason, mirroring the schema/version banners' own justification, which
itself invokes no multiplicity argument at all) — **not** a forced consequence of the same `k`-
shrinkage mechanics that govern the ordinary pre-registered mixed case (at most a secondary,
supporting observation). This is a minor/moderate documentation-quality finding, not a blocker.

## 4. Everything else — one small, low-stakes drift; no other statistical-behavior changes found

- **Minor drift from `-ml` §3.5's literal instruction, step 6.** `-ml` §3.5 says
  `_rank_resolving_power_lines` "should resolve each metric's kind (**reusing the same
  `_metric_kind`-based resolution §3.3 introduces**)" — i.e., the full N-ary `_resolve_reference_
  kinds` cross-check. The plan instead resolves kind from the *first* non-`None` aggregate across
  `runs` (`aggs`, `first_agg`, plan §4 step 6) — a narrower check that does not detect a cross-arm
  disagreement the way `_resolve_reference_kinds` would. This is low-stakes: the function only
  gates an informational preview *sentence* (never a rendered verdict), the disagreement case is
  unreached by any shipped pack today (same status as the rest of that code path), and reusing the
  cheaper `_aggregate_kind` helper here avoids paying `_resolve_reference_kinds`'s full N-arm cost
  for every metric of every pack on every rank report, continuous or not, with or without
  `--reference` — a real cost/complexity trade worth making differently than the note's literal
  words. But it is a drift, however small, and worth naming per the brief: if this function is ever
  revisited, note that its kind-resolution is coarser than the reference-family path's, and a future
  cross-arm disagreement would silently pick whichever arm's aggregate happens to be first in `runs`
  order for this sentence alone, rather than being caught and banner-flagged the way the reference
  family itself would catch it.
- **No other statistical-behavior change found.** I checked the two spots most likely to hide a
  quiet behavior change: (i) the binary combined-ladder block inside step 7's dispatch is a literal
  copy of the current unconditional code (`report.py:1453-1472`), now behind the `resolved_kinds ==
  {"binary"}` guard — confirmed identical line-for-line, so `guard-judge-understanding`'s rendering
  is untouched by construction, not merely by test; (ii) `design_effect`/`basis` selection in the
  new continuous renderer (`max(...)`/`min(..., key=_BASIS_STRENGTH.__getitem__)`) matches the
  existing binary reference table's and `compare_report`'s continuous branch's conventions exactly
  — no new provenance rule introduced.
- The test strategy (plan §5, extending `-ml` §4's eight obligations to nineteen) is proportionate:
  tests 7-8 specifically target the diagnosed §3.4 gap (two candidates disagreeing with each other,
  and a candidate disagreeing with a *declaring* reference — the case a pairwise-anchored check
  would miss precisely because the reference's own declaration would otherwise win by default),
  tests 13-16 exercise both refusal paths and their distinct messaging, and test 19 is the
  byte-for-byte non-negotiable regression pin the coordination ledger requires, backed by the two
  existing mechanism-level tests rather than the string diff alone. No gaps in the eval design worth
  flagging.

## 5. Summary for the implementer

Build exactly what plan §4 (steps 1-8) specifies — nothing found here should change a single line
of it. Two follow-ups, neither blocking:

1. Revise plan §3.5 (or a short addendum) so the whole-family-refusal argument for a genuine
   disagreement rests on the data-integrity/self-contradictory-record reasoning as primary, with
   the `-ml` §3.3 multiplicity argument demoted to a secondary note that does not claim disagreement
   "mechanically requires" the same treatment as a pre-registered mixed family.
2. If `_rank_resolving_power_lines`'s coarser kind check (plan §4 step 6) is ever revisited for a
   pack that might have a genuine cross-arm disagreement, know that it will not be caught there —
   only the reference-family path (`_resolve_reference_kinds`) catches it today.
