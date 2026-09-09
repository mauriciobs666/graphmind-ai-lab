# Fixture design for a computed estimate — what the input lets a mutation move

> **On-demand knowledge base for `tdd-engineer`.** How to choose the *input* for a test over a
> value the code **computes** — a resampled confidence interval, a level a helper derives from a
> collection's length, anything scaled by a correction constant. These are the tests that pin the
> right quantity and still cannot see it move. Guards over other code's *text* are a different
> problem and live in `guard-testing-techniques.md`; ordinary unit practice lives in
> `tdd-engineer.md`.
>
> Origin: distilled 2026-09-09 from `kaizen_team` via `agent-maintenance` skill §5.

## A degenerate fixture is a precision instrument in one direction and a blindfold in the other

**The mechanism, stated once.** A resampling estimator over an all-identical sample has nothing to
resample: every draw is the same value, so the resample distribution collapses to a single atom
and **every percentile is that value exactly** — for any `B`, any seed. The interval is a
zero-width point, and it is *deterministic*, not merely likely.

That single property has two opposite consequences, and the pairing is the lesson: **what makes a
fixture decisive for one assertion is exactly what makes it blind to another.**

**Direction 1 — it constructs an exact boundary with no luck involved.** A strict-vs-non-strict
comparison against a bound (`ci[0] > 0` vs `>= 0`) looks untestable, because a continuous sample
never lands a percentile precisely on the bound. It does not have to: an all-zero sample puts it
there by construction. So **do not defer a comparison mutation as "needs Monte-Carlo luck to
construct" before trying a degenerate input.**

**Direction 2 — it silently disables anything that scales the interval.** A zero half-width scaled
by `sqrt(design_effect)` is still zero, at any design effect; a support clamp over a point already
inside the support never fires. A test built on such a fixture can assert the exact expected
bounds and **stay green with the clamp deleted outright.**

**One precision the original capture did not state, and it matters when you copy this.** The
blindfold needs *two* conditions, not one: zero width **and** the point inside the support. A
constant sample sitting outside it is clamped normally — the transform is observable again. Verify
which case your fixture is in rather than assuming degeneracy alone hides the clamp.

Re-derived here 2026-09-09 by standalone reproduction (stdlib `random`/`statistics`, percentile
bootstrap of the mean, no project code):

| fixture | observation |
|---|---|
| `[0.0]*8`, support `(0,1)` | `ci == (0.0, 0.0)` exactly, on all 12 combinations of `B ∈ {200, 2000, 20000}` × `seed ∈ {0,1,42,12345}` |
| same, comparison mutated | strict `ci[0] > 0 or ci[1] < 0` → `False`; mutated `>=`/`<=` → `True`. One fixture kills it |
| a continuous 8-sample straddling zero | an exact `0.0` bound in **0 of 2000** seeds — "needs luck" is true only *off* the degenerate case |
| `[1.0]*10`, support `(0,1)` | half-width `0.0`; widened at `DEFF ∈ {1, 4, 9, 100, 10⁶}` → `(1.0, 1.0)` every time; clamped ≡ unclamped at all five |
| `[1.5]*10` / `[-0.2]*10`, support `(0,1)` | zero width, but the clamp **does** bite — the blindfold needs in-bounds too |
| same shape, real spread, `DEFF=9` | unclamped upper `1.28` vs clamped `1.0` — the mutation dies |

**The fix is a second fixture, not a rewritten assertion.** Keep the degenerate case for the
boundary it pins honestly, and add one with real spread at a design effect large enough that the
unclamped bound leaves the support. Say in the docstring which of the two proves what: a
zero-variance case that *reads* as a worked example of the clamp is a false certification, and the
next reader will trust it.

**Both errors were made by one implementer in one session** — a degenerate sample used as the
clamp's worked case (blind), and a degenerate sample *declined* for the boundary test as
unconstructible (the tool, unused). They are the same fact, which is why they are one section.

## To pin a constant a helper derives from `len(collection)`, vary the length at one seed

A helper that computes an internal correction level from the size of a collection it is handed
(`k = len(family)`, level `alpha/(2k)`) has a wiring failure mode with no exception and no wrong
type: a caller that passes a **one-element** collection instead of the whole one silently drops the
correction and returns a plausible, conventional, wrong interval.

**Why a large suite stays silent.** The tests that look like they cover it assert on the rendered
explanation — which is generated from the metric name and the *nominal* alpha, not from the
derived level, and is therefore byte-identical under the mutation. Asserting on narration cannot
see a constant change.

**The technique.** Render **identical underlying data twice through the caller**, varying only the
collection's length (`k=1` vs `k=2`), holding the data and the RNG seed fixed, and assert a
**monotonic effect on the number the constant moves** — here, that the `k=2` interval is strictly
wider. Same-seed pairing removes resampling noise from the comparison, so the assertion is an
exact inequality rather than a tolerance.

Re-derived here 2026-09-09, same standalone harness (`alpha/len(family)`, 40 paired diffs, one
seed):

| wiring | `k=1` width | `k=2` width | `k2 > k1` | explanation text identical across `k` |
|---|---|---|---|---|
| correct | `0.078465` | `0.090482` | **True** | yes |
| collapsed to one member | `0.078465` | `0.078465` | **False** | yes |

The width assertion passes clean and fails on the mutant; the text assertion passes on **both**.
Generalise it as: **assert on the quantity the constant actually moves, never on prose the code
generates from the un-corrected inputs** — and choose the observable by asking which number a
wrong constant would change, not which output the feature is named after.
