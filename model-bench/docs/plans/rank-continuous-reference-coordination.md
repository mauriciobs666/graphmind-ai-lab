# `rank --reference` on a continuous verdict metric — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (backlog item, `docs/BACKLOG.md`, closed via `docs/HISTORY.md` U188)

## Goal

Fix the real, shipped product bug `qa-engineer` found while testing the user manual (D-1,
`docs/test-reports/small-model-benchmarking-manual-additions-report.md`): `./run.sh rank --pack
<id> --reference <modelKey>` crashes (exit 2, uncaught `MetricKindError` naming `scored_value`) on
any pack whose headline/verdict metric is continuous rather than boolean — e.g.
`embedder-graphrag-retrieval` (metric: `mrr`). The manual already carries a known-issue callout +
`compare`-as-workaround (committed `460057d3`); this coordination fixes the underlying code defect
and closes the `docs/BACKLOG.md` item filed alongside it.

**Investigation to date (by this coordinator, reading `modelbench/report.py`/`stats.py` directly,
not yet delegated) found the defect is deeper than "branch to `_paired_diffs` instead of
`_paired_rows`":** `compare_report` already branches correctly by `_metric_kind`
(`report.py:1672-1770`), refusing a *mixed*-kind family whole rather than guessing
(`report.py:1666-1671`), and its continuous path calls `stats.continuous_verdict`
(`stats.py:1651`). But `continuous_verdict`'s own multiplicity correction is `k = len(family)` —
the pack's own metric-count axis — with **no `correction_k` decoupling parameter**, unlike the
binary path's `stats.verdict()`, which explicitly exposes `correction_k` precisely so FR-8's
reference-anchored family (an independent candidate-count axis) can pass it in
(`stats.py:1230-1238`: "FR-8's reference-anchored family... always passes this explicitly").
`rank_report`'s reference-family builder (`report.py:1447-1470`) computes one combined
`stats.holm_steps` call over the full metric×candidate p-value matrix for the binary case — there
is no continuous equivalent of that matrix today. Extending `rank`'s reference-anchored family to
a continuous metric correctly is a statistical-design decision (does the candidate-count axis
correct in the interval, per-candidate, or across the whole metric×candidate grid; does a mixed
binary/continuous family within one reference-family call get refused whole, mirroring
`compare_report`'s existing invariant) before it is an implementation change.

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `data-scientist` | `ad9b70127f174f7a7` | delivered | `docs/plans/rank-continuous-reference-ml.md` | — → — | 156.7k tok / 33 tools |
| U2 | `architect` | `a9a3fa58f7d09e64f` | accepted | `docs/plans/rank-continuous-reference.md` | `analyst`(`a21cc72f34c1d08e3`, **approve with suggestions**)+`data-scientist`(`a3f33215c664e083f`, **approve with suggestions**) → both approve, no blockers | 232.3k tok / 45 tools |

**`analyst` plan review** (`docs/reviews/rank-continuous-reference.md`): approve with suggestions.
Reproduced D-1 live, confirmed the current suite green (1778 passed) as baseline. One **major**
(M1): the new continuous caption independently recomputes `alpha_used` rather than sharing a
formula with `stats.continuous_verdict` — a real duplicate-formula seam, violating
`AGENTS.md`'s "stats.py is the only home for every formula" invariant; suggested fix: a shared
`stats.alpha_used(alpha_family, k)` helper. Four **minor**: (m1) the plan's blanket "no partial
table under refusal" over-reads the `-ml` note for the *ordinary pre-registered mixed* case
specifically — `compare_report`'s own precedent still renders a per-metric diagnostic tally there
(kind is still cleanly resolved per metric in that case, unlike the disagreement case); (m2) the
continuous branch's `elif` should test `resolved_kinds == {"continuous"}` literally rather than by
elimination; (m3) `_rank_resolving_power_lines`'s weaker kind-check is a risk-section note, not a
blocker; (m4) a fixture-helper attribution nit (`tests/conftest.py`, not `tests/test_report.py`).
No blockers — decided to fold M1+m1+m2+m4 into the `tdd-engineer` brief directly (all concrete,
unambiguous, no new design judgment needed) rather than loop back to `architect` for a plan
rewrite; m3 stays a documented, accepted risk.

**Decision: proceed directly to U3 (`tdd-engineer`)** — both gates cleared with no blockers; every
finding from both reviews is concrete and self-contained enough to fold into the implementation
brief rather than costing another architect round-trip.

**`data-scientist` methodology review** (`docs/reviews/rank-continuous-reference-ml.md`): approve
with suggestions, no blockers. Confirmed no drift between the plan and the method note on any
load-bearing formula. Confirmed open question (a)'s N-ary gap is real, not over-engineering.
Open question (b): agrees with the plan's *outcome* (refuse whole + louder banner) but found its
"mechanically the same as k-shrinkage" argument overstated — the tighter, correct justification is
that a cross-arm disagreement is a data-integrity failure, not a multiplicity problem (mirroring
the schema/version-mismatch banners' own justification, which invokes no multiplicity argument at
all) — a documentation-quality fix to §3.5's rationale, not a code change, non-blocking. Also noted
plan step 6's `_rank_resolving_power_lines` fix uses a cheaper first-non-null-aggregate kind check
rather than the full N-ary `_resolve_reference_kinds`, acceptable since that function only gates an
informational sentence, never a rendered verdict — named for the record, not a blocker.
| U3 | `tdd-engineer` | `a5f6f0e888be72401` | accepted | code + tests | `analyst`+`data-scientist` → both approve, no blockers | 388.8k tok / 195 tools (3 rounds) |
| U4 | `teco` | — | accepted | `docs/BACKLOG.md` item removed, `docs/HISTORY.md` U188 added | — → — | — |

**Round 3 (self-initiated by `tdd-engineer`, per teco's ask to self-check for the same blind
spot):** found and closed a third same-class gap — `support_metric`'s preference-with-fallback
(`_metric_aggregate(reference_run, ...) or _metric_aggregate(candidate, ...)`) was also untested.
teco independently re-ran the design_effect/basis mutation against the tracked tree directly (not
a scratch copy) at the correct line (983, disambiguated from the unchanged binary-path's identical
expression at line 1474) and confirmed the new test fails for the right reason, then confirmed
clean revert and full suite green (1800 passed). Production code (`report.py`/`stats.py`)
unchanged in this round — test-only addition — so no third external gate round was dispatched;
closing here per the two-same-class-catches stopping condition set earlier, satisfied.

## Closeout

U1-U4 all accepted. `docs/BACKLOG.md`'s D-1 item removed; `docs/HISTORY.md` U188 entry added
(2026-09-20). All six family documents (`docs/plans/rank-continuous-reference.md`,
`docs/plans/rank-continuous-reference-ml.md`, `docs/reviews/rank-continuous-reference.md`,
`docs/reviews/rank-continuous-reference-ml.md`, `docs/reviews/rank-continuous-reference-impl.md`,
`docs/reviews/rank-continuous-reference-ml-impl.md`) flipped to `Status: archived` (mechanical
flip only, no other change). This coordination doc is now `Status: archived` too — see below.

**Follow-up handed to `tico` (not this coordination's to do):** `docs/manuals/
small-model-benchmarking.md`'s known-issue callout for D-1 (committed `460057d3`) is now stale —
either remove it or add a "fixed as of `docs/HISTORY.md` U188" pointer.

**Pre-existing drift noted, out of this coordination's scope, not chased:** `docs/HISTORY.md` has
no entry for the manual-extension commit (`460057d3`) that preceded this coordination — flagged
for whoever next touches `model-bench/docs/HISTORY.md`, not fixed here.

**`analyst` post-implementation review** (`docs/reviews/rank-continuous-reference-impl.md`):
approve with suggestions. Faithful implementation confirmed, suite green for real (1798 passed),
both live-run checks reproduced independently, the two named regression tests (round-2's
`correction_k` pin, the byte-for-byte guard-judge golden) hand-traced and confirmed real, not
vacuous. **New Major finding (own independent mutation pass, same class as teco's round-1 catch):**
`_render_reference_family_continuous`'s `design_effect=max(...)`/`basis=min(...)` selection
(passed into `continuous_verdict`) is untested — every continuous fixture uses identical values
for reference and candidate, so flipping `max`↔`min` in an isolated scratch copy left the full
suite green (174 relevant tests, unchanged). Code itself confirmed correct (mirrors the
pre-existing binary-path pattern); this is a coverage gap, not a live bug — same silent-drift risk
class as round 1's `correction_k` gap. Decision: one more round, folding this in, then stop —
this is a real, cheap, same-class gap worth closing (not diminishing-returns churn), but two
same-class catches in a row is also the natural stopping point; no third round unless this one
surfaces something new.

**`data-scientist` post-implementation methodology confirmation** (`docs/reviews/
rank-continuous-reference-ml-impl.md`): approve, no blockers. Confirmed `correction_k`/combined
`k` formula matches design exactly; §3.5's amended rationale correctly reflects the earlier
methodology finding (not a superficial reword); `stats.alpha_used` used consistently at both call
sites; live-verified `k=4, alpha_used=0.0125 (98.75% CI)` by hand against all four candidate rows.
One flagged, non-blocking deviation: shipped code renders a `_pairing_tally` line under the
ordinary mixed-family refusal (analyst's m1 fold-in from the plan gate) where the original plan's
§3.8 had rejected any tally — inert, computes no statistic, no correctness impact.

**U3 (verified by teco):** applied all fold-ins (M1/m1/m2/m4, data-scientist's finding 3). Own
mutation test (data-plumbing argument, not the delegate's own control-flow table) found a real gap
round 1: `_render_reference_family_continuous`'s per-candidate `correction_k` had no integration
test pinning it — full suite stayed green even with `correction_k=None` silently substituted.
Resumed `tdd-engineer` (same agent id); round 2 added
`test_rank_report_reference_family_continuous_row_uses_the_real_combined_k_not_k1`. **teco
independently re-ran the exact same mutation against round 2's code — the new test now fails for
the right reason** (`assert 'not distinguishable' in '... | distinguishable |'`), reverted cleanly,
full suite green (1798 passed, 3 deselected) confirmed by teco directly, not just reported. Files
touched: `modelbench/report.py`, `modelbench/stats.py`, `tests/test_report.py`,
`tests/test_stats.py`, `docs/plans/rank-continuous-reference.md` (§3.5 rationale amended in place).
Live acceptance check (`./run.sh rank --pack embedder-graphrag-retrieval --reference bm25` → exit
0, `k=4, alpha_used=0.0125 (98.75% CI)`) reported by the delegate, not yet independently re-run by
teco — queued for integration step alongside the post-implementation gate.
| U4 | `teco` | — | queued | `docs/BACKLOG.md` removal + `docs/HISTORY.md` entry | — → — | — |

**U2 note (verified by teco):** spot-checked several load-bearing claims directly against the
tree — `_metric_kind`'s actual current body (`report.py:229-242`) exactly matches the plan's
described pre-refactor state (confirms step 3's "pure refactor" claim); `mover_d_interval`'s
`diff = p1 - p2` orientation (`stats.py:126-140`) supports the polarity-reuse argument in §2.2;
all five cited test names/line ranges (`tests/test_report.py:1741`, `:1828`, `:3558`, `:3659`,
`tests/test_stats.py:567`) exist with matching content. Plan resolves both of `-ml`'s open
questions itself (N-ary kind resolution via a new `_resolve_reference_kinds`, and a
disagreement-vs-mixed-family framing split) rather than punting them to the implementer — sending
to both `analyst` (completeness/code-quality) and `data-scientist` (methodological soundness of
§3.4/§3.5's resolutions and fidelity to the `-ml` note) in parallel, since both are read-only
reviews of the same document.

**U1 note (verified by teco):** method note significantly widened scope beyond the original crash
fix — spot-checked its three key evidentiary claims directly against the repo (all confirmed): (1)
`_rank_resolving_power_lines` is already live and silently prints McNemar/Wilson-shaped "pp"/"80%
power" language for `embedder-graphrag-retrieval`'s continuous `mrr` metric today, independent of
`--reference` (`reports/embedder-graphrag-retrieval-rank-20260920-03.md` lines 19/21, grepped
directly); (2) `report.py:1472`, `correction_k = len(combined_p_values)`, matches the note's cited
binary-path formula exactly; (3) `stats.py`'s `ContinuousVerdict.text` (~1740-1753) hardcodes the
literal `"95% CI"` in both branches while never reading its own computed `alpha_used` field —
confirmed by reading the two f-strings directly. Recommendation: `continuous_verdict` gains
`correction_k: int | None = None` mirroring `verdict()`'s existing parameter exactly; combined `k =
len(family) * len(candidates)` (same formula/reasoning as the binary path); `rank_report`'s
reference path needs the same mixed-kind "refuse whole" guard `compare_report` already has; fix
the two adjacent defects (§2.3, §2.4 of the note) in the same unit, not separately. Two open
questions left explicitly for the architect (§5 of the note): whether `_metric_kind`'s two-arm
signature extends cleanly to N arms, and how a genuine cross-arm kind disagreement (vs. a
pre-registered mixed family) should be handled.

U2 depends on U1 (the plan must build on the method note's correction design). U3 depends on U2.
U4 depends on U3's gate passing.

## Notes

- Precedent to mirror, not reinvent: `compare_report`'s mixed-family-refused-whole invariant
  (`report.py:1666-1671`) and its continuous path (`report.py:1708-1745`).
- `rank_report`'s existing binary reference-family flow (`report.py:1436-1470`) must keep working
  identically for `guard-judge-understanding` — regression coverage, not just new-path coverage.
- Manual (`docs/manuals/small-model-benchmarking.md`) already documents the workaround; once fixed,
  the known-issue callout should be revisited (does it still apply, does it need a "fixed as of
  ..." pointer) — fold into U3's or U4's done-condition, whichever lands the fix.
