# `rank --reference` on a continuous verdict metric — implementation methodology confirmation

> **Status:** archived · **Owner:** `data-scientist` · **Tracks:** —

**Confirming:** the delivered code (`modelbench/stats.py`, `modelbench/report.py`, working-tree
diff against `HEAD`, uncommitted at review time) against this role's own method note
(`docs/plans/rank-continuous-reference-ml.md`, "`-ml`"), this role's own prior plan review
(`docs/reviews/rank-continuous-reference-ml.md`), and `analyst`'s plan review
(`docs/reviews/rank-continuous-reference.md`). Scope: statistics/methodology only — `analyst`'s
parallel diff-scoped pass covers completeness/code quality/line-number accuracy.

## Verdict: **approve**

The shipped code implements `-ml`'s statistical design with no drift on any load-bearing formula,
the amended plan §3.5 correctly reflects this role's methodology finding, the shared `alpha_used`
helper closes the plan-review's M1 finding cleanly at both call sites, and a live re-run against
the real stored corpus produces numbers that check out by hand against the note's own formulas. No
blockers, no suggestions that change behavior.

## 1. `correction_k`/`k` formula — matches the design exactly, no drift

Read directly from the working-tree diff (`git diff modelbench/stats.py`):

```python
k = correction_k if correction_k is not None else len(family)
levels = _family_ci_levels(alpha_family, k)          # alpha/(2k), 1 - alpha/(2k)
used_alpha = alpha_used(alpha_family, k)              # alpha_family / k
```

This is character-for-character `-ml` §3.1's specified change and mirrors `stats.verdict()`'s
already-shipped `correction_k` line (`k = correction_k if correction_k is not None else
len(family)`, `stats.py:1270`) exactly — the same "generalize the parameter, not the effect"
pattern `-ml` §3.1 required. `family`'s only remaining job is the `metric_name in family`
membership check, untouched.

Confirmed the correction lands **in the interval's own quantile levels**, not a Holm step — exactly
`-ml` §2.2's mechanism argument: `_family_ci_levels` computes `alpha/(2k)` and `1 - alpha/(2k)` as
exact `Fraction` arithmetic (recovering `alpha_family` via `Fraction(str(alpha_family))`, per this
function's own long-standing decimal-recovery invariant, unchanged by this diff) and feeds those
levels to `paired_cluster_bootstrap`. There is no p-value, no `alpha_step`, and no Holm ladder
anywhere on this path — `rank_report`'s new continuous branch (`report.py`, `resolved_kinds ==
{"continuous"}`) computes `correction_k = len(family) * len(candidates)` as a plain count and
passes it straight through to `continuous_verdict`, never touching `stats.holm_steps`. This is the
combined `N x M` divisor `-ml` §3.2 specifies, computed identically to (and justified identically
to) the binary path's `correction_k = len(combined_p_values) = len(family) * len(candidates)`
(`report.py:1472`, unchanged) — I re-confirmed the plan does not special-case `len(family) == 1`
away anywhere in the shipped `elif resolved_kinds == {"continuous"}:` branch (the m2-fixed explicit
condition, §3 below).

## 2. Amended plan §3.5 — correctly reflects the methodology finding, not a superficial reword

Read the revised section in full (`docs/plans/rank-continuous-reference.md` §3.5, under the
`> **Revision (2026-09-20, tdd-engineer...)**` marker). My prior review's finding 3
(`docs/reviews/rank-continuous-reference-ml.md` §3) was: the plan's original "mechanically the
same as k-shrinkage" framing borrowed the pre-registered-mixed-family's multiplicity rationale for
a disagreement case that is actually a data-integrity failure, and the two arguments do not
transfer.

The rewrite gets this right, on both halves:

- **Ordinary pre-registered mixed family** — kept as "a scoping decision about an undesigned
  mechanism... a real, load-bearing reason, but a mechanical/multiplicity one." Unchanged from
  before, correctly, since my finding never disputed this half.
- **Genuine cross-arm disagreement** — rewritten to state plainly that `correction_k` "is a pure
  count... does **not** mechanically depend on any metric's kind resolving cleanly," that "nothing
  forces `k` to shrink if only the disagreeing metric's own cells were refused," and that the
  refusal is instead "for a data-integrity reason, primary": a metric resolving to different kinds
  depending on which arm answers it is "a self-contradictory record," in the same class as the
  pack/version/schema banners, "none of which invoke a multiplicity argument either." The
  `k`-fixed property is explicitly demoted — "a real, secondary property this plan's design already
  delivers... not the forcing reason for the disagreement case's own refusal."

This is exactly the primary/secondary reordering my finding asked for, not a word-level rephrase
that keeps the same (wrong) logical structure. The behavior the section specifies (whole-family
refusal, distinct louder banner) is unchanged, as the revision note itself states, and as confirmed
directly in the shipped code (§4 below): `_REFERENCE_KIND_DISAGREEMENT_EXPLANATION`
(`report.py`) carries the identical data-integrity-primary framing verbatim, down to the
"self-contradictory record"/"owed the same distrust" phrasing, with a doc-comment above it citing
`docs/reviews/rank-continuous-reference-ml.md` §3 by name. Good practice: the argument now lives in
exactly one place (the plan) and the code's own comment cites it rather than re-deriving it.

## 3. `stats.alpha_used` — closes plan-review M1, used consistently at both call sites, formula exact

`git diff modelbench/stats.py` shows the new top-level function:

```python
def alpha_used(alpha_family: float, k: int) -> float:
    return alpha_family / k
```

placed directly beside `_family_ci_levels`, its natural home. Confirmed both call sites use it,
not a re-derivation:

- **Inside `continuous_verdict` itself** — `used_alpha = alpha_used(alpha_family, k)`, and the
  dataclass's own `alpha_used` field is set from `used_alpha` (the local variable renamed to avoid
  shadowing the module-level function of the same name — a small but real correctness detail: a
  naive `alpha_used = alpha_used(...)` would have worked once but been a landmine for the next
  editor).
- **Inside `_render_reference_family_continuous`'s caption** (`report.py`) — `used_alpha =
  stats.alpha_used(pack.metrics.alpha_family, correction_k)`, computed once before the
  per-candidate loop (the caption prints before any `cv` exists, exactly as the plan-review's
  M1 suggested fix required — "the caption still prints before any `cv` exists... it must call
  this helper directly rather than reading `cv.alpha_used` off a candidate's result").

Formula is exactly `alpha_family / k` at both call sites, no divergence, no second independent
computation anywhere in the diff — the "two copies of a formula is one copy and one bug" seam
`analyst`'s M1 flagged is closed by construction, not by the drift-detection test alone (test 10 in
the plan's own numbering still exists as a regression pin, but is no longer the only thing keeping
the two numbers in sync).

## 4. Live verification — ran the real command, checked one row by hand

```
$ ./run.sh rank --pack embedder-graphrag-retrieval --reference bm25
```

Exits 0. Reference-anchored family caption:

```
alpha_family=0.05, k=4, alpha_used=0.0125 (98.75% CI)
```

Checked by hand: `len(family) = 1` (`verdictMetrics: ["mrr"]`, confirmed from
`packs/embedder-graphrag-retrieval/pack.json`), `len(candidates) = 4` (5 stored model keys minus
the `bm25` reference) → `k = 1 * 4 = 4`. `alpha_used = 0.05 / 4 = 0.0125` exactly. Coverage
`= 100 * (1 - 0.0125) = 98.75`. Both match the printed caption exactly.

One candidate row, `text-embedding-qwen3-embedding-4b`: `diff = +0.088`, `CI = [-0.026, +0.212]`,
`decision = not distinguishable`. Checked the decision predicate by hand against
`ContinuousVerdict.distinguishable`'s own rule (`ci[0] > 0 or ci[1] < 0`): `-0.026 <= 0` and
`+0.212 >= 0`, so the interval covers zero — `not distinguishable` is the correct call, consistent
with the printed row. Repeated the same check on the other three candidate rows (`granite`, `nomic`,
`qwen3-0.6b`) — all four intervals cover zero and all four print `not distinguishable`, consistent.
I did not re-derive the bootstrap's exact percentile boundaries by hand (that requires re-running
`B=10000` resamples at the pinned seed, which is what the unit tests already pin) — the check here
is the formula (`k`, `alpha_used`, coverage) and the decision predicate against the printed CI, both
of which are exactly reproducible without re-running the bootstrap.

Also ran the full suite as a sanity check on top of `analyst`'s parallel pass: `1798 passed, 3
deselected` (up from the plan-review's recorded baseline of `1778 passed, 3 deselected` — a net 20
new tests, consistent with the plan's 19 newly-specified tests plus the M1-fix's own
`test_alpha_used_is_alpha_family_over_k`). The `correction_k`/`_resolve_reference_kinds`/
`alpha_used`/`kind_disagreement`-named tests (26 of them) all pass.

## 5. Anything else that changes statistical behavior — one inert deviation from the plan, not a concern

The shipped code renders a `_pairing_tally` line under the **ordinary pre-registered mixed-family**
refusal (`report.py`, the `elif reference_kind_disagreements:`/`else` branch inside `rank_report`'s
message-selection chain, comment citing "m1"). The architect's plan (§3.8) explicitly rejected this
— "this plan does **not** mirror that for `rank_report`'s refusal paths... this plan renders no
tally in the ordinary-mixed sub-case either" — but `analyst`'s plan review (m1 finding) judged that
rejection's citation of `-ml` §3.3 as reading more into the note than it says, and offered fixing it
as one of two acceptable alternatives; the implementer took that option.

This is a real behavior change from the plan as written, worth naming since the brief asked for
anything that changes statistical behavior — but it does **not** change any statistic: no `k`, no
`alpha_used`, no CI, and no verdict is computed in either the tally or no-tally rendering, since
both are the *refused* path where no `continuous_verdict`/`mcnemar_exact` call happens at all. The
tally is a purely descriptive per-candidate paired-n count, mirroring `compare_report`'s existing
precedent exactly. It is also unreached by any pack shipped today (no pack currently mixes binary
and continuous `verdictMetrics`), so it has zero live effect. No objection from this review.

The `m2` fix (explicit `elif resolved_kinds == {"continuous"}:` rather than a bare `elif` under an
`# resolved_kinds == {"continuous"}` comment) is present in the diff exactly as `analyst` suggested
— confirmed it forecloses the `family == []` silent-zero-`k` edge case the finding named, with no
change to any real pack's behavior (no shipped pack has an empty `verdictMetrics`).

No other statistical-behavior change found. The binary combined-ladder block inside the new
three-way dispatch is untouched line-for-line (confirmed by direct read, not just by the passing
regression test), so `guard-judge-understanding`'s existing rendering is unaffected by construction.

## 6. Summary

Nothing here blocks landing this change. The two follow-ups `analyst`'s plan review asked for (M1,
m1, m2) are all visibly closed in the delivered diff, and the one follow-up this role's own plan
review asked for (the §3.5 rationale rewrite) is closed correctly, not superficially. The live
run's numbers are exactly what `-ml`'s formulas predict.
