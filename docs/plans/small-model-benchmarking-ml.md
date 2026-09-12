# Small-Model Benchmarking — Statistics and Metric Definitions

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Version:** 1.25

2026-09-11 (v1.25, `data-scientist`) — corrects this note's `nlq-generator` **analysis-unit count**,
which was the raw item count rather than the scoring denominator. Plan §3.8.3 excludes
structurally-unanswerable items from `layer1ExactMatchRate`'s denominator, and
`model-bench/docs/reviews/nlq-conflicting-facts-answerability-ml.md` establishes that the excluded
set is **6 of 40** (4 `relationship-traversal` + 2 `conflicting-facts`), not the plan's stated 4 —
restoring the ruling `falkor-chat/docs/plans/workflow-nl-query-generation-ml.md` §5 already made
over this same golden file. So **`n_eff = 34`, not 40**: §7.2's row and §7.1's table are corrected
(**floor 15.0 → 17.6 pp, MDD₈₀ 19.1 → 22.3 pp**), and §3's composition fact gains the
answerable/unanswerable split. **The adequacy verdict does not move** — the reference effects this
lab has needed to resolve (97.5-vs-0 pp; ~30 pp) both still clear 22.3 pp. **No power is lost by the
exclusion, and the note should not be read as if it were:** an item both arms score 0 on is a
*concordant* pair contributing to neither `b` nor `c`, so McNemar's exact p and the 6-net-discordant-
wins count floor are **identical** at 34 and at 40 — the change is a re-expression of the same test
in units that describe the metric beside it. **Two denominators, both correct, neither
interchangeable:** `n_eff = 34` governs every inferential bound, while `Y = 40` — the executed-item
count — governs §11.6's latency-coverage census and the `Y ≤ 21` reach bound, and is **unchanged**.
§7.1's discordance-mix sensitivity table is indexed by `n` generically rather than by pack; its
n=40 column no longer corresponds to a declared pack, and `nlq-generator` now reads between its
n=30 and n=40 columns.

2026-09-10 (v1.24, `data-scientist`) — rules the s3-spec's (`docs/plans/small-model-benchmarking-
s3-spec.md` §9) two open questions, both disambiguations of formulas this note already states
rather than new methodology. **§5.1: `recall_at_k`/`precision_at_k` binarize into `BinaryMetric`'s
`counts[metric]` as "at least one relevant doc in top k" (`recall_at_k(...) > 0`), not "all relevant
docs found" (`== 1.0`)** — matches this note's own §5.2 precedent (`sep_raw(q) > 0 ⟺ P@1 = 1`, the
one place this note already turns a continuous per-query quantity into a discrete success), matches
the standard IR reduction of recall@k to a binary outcome (Hit-Rate@k/Success@k; `== 1.0` has no
standard name and is strictly more conservative), and keeps recall@10's stated role (§7.4: "harness
sanity floor... and regression detector, not a comparison metric") from getting more trigger-happy
on exactly the 2 items where the two readings disagree. **Recommend storing the raw hit count**
(`|top-k ∩ R|`, an int in `[0, |R|]`) in `counts[metric]` rather than a pre-binarized 0/1 flag —
`ItemResult.scored_outcome` already reads `counts[metric] > 0` generically (`results.py:416`), so
the aggregate is identical either way, and the raw count is free provenance a flattened flag
discards. **New honesty-line requirement: model-bench's reported recall@10 is not the same
statistic as falkor-chat's `retrieval_baseline.json`/FR-12's quoted 0.974** — that figure is
`sum(recall_at_k(...) for each item) / n` (`falkor-chat/server/tests/eval/test_retrieval_eval.py:
134`, a mean of the continuous fraction), and no binarization of a `BinaryMetric` reproduces a
continuous mean in general (only coincidentally when every item's fraction is already 0 or 1, which
is why the two readings agree on the pinned 37/38 baseline but need not agree on a future run with a
genuinely partial-credit multi-relevant item). Report this footnote alongside the existing
precision@k one. **§5.2: `sd({cos(q, d) : d in corpus})` is the population standard deviation**
(`statistics.pstdev`), confirming the s3-spec's own draft reasoning: the 121-doc corpus is
enumerated in full and scored exactly for every query — it is not a sample used to estimate an
unknown, larger population's spread, which is the condition that would call for Bessel's correction
(sample stdev, `ddof=1`). This is a different statistical object from §7.4's `sd_d` (the sample
stdev of per-query MRR *differences*, legitimately estimating variability for inference over the
38-query sample) — no contradiction between the two uses. At n=121 the two estimators differ by
`sqrt(121/120) ≈ 1.004`, immaterial to any reported figure, but the estimator is still part of the
published claim and is now pinned rather than left to `sd`'s ambiguity. Ruled here, in place,
rather than in a new `-ml`-role sibling file: both questions disambiguate formulas §5.1/§5.2 already
state, not new methodology, matching this note's own revision precedent (e.g. v1.22/v1.23 folding
in rulings raised elsewhere).

2026-09-10 (v1.23, `data-scientist`) — folds this note's half of the raising-`dispatch` ruling
(`docs/plans/small-model-benchmarking-ml-dispatch-failure.md`, accepted at `a65d288`): §4.1's
`unrunnable` gains the **tool channel** beside the model channel; §4.3 rule 4 gains its row and the
discriminator sentence (same *unattributable, no observation* argument, plus why the bucket is
`unrunnable` and never *undispatchable* — the latter partitions as `no_attempt`, a failure charged
to the model, more often for the weaker arm); rule 5 gains its one exception, **state**
contamination reaching (a)–(g) and the `I(t)` summary where **history** contamination does not, so
a conversation censored by a raise has no turns after `t` at all; and rule 3's funnel splits its
`unrunnable` line by channel. **`ITERATION_SUMMARY_DISPOSITIONS`/`_EXCLUDED` and their union
assertion are deliberately untouched**: the ruling adds no sixth `TurnDisposition` member — a
dispatch raise is conversation-scoped and is carried as a conversation-level censoring marker — so
the union against `convo.TURN_DISPOSITIONS` still binds five members and must not be widened.

2026-09-10 (v1.22, `data-scientist`) — plan-gate Pass 15's blocker **P15-1**, plus plan v1.27's two
`§7 rule 3` raises; all three ruled, and two of the three are against this note.
**P15-1: v1.21 closed one hole and there were three, so the fix is a rule and not a second
clause.** §4.3 gains **rule 5** — an `unrunnable` turn ends its conversation's *trajectory*, and
every statistic indexed by turn position drops that conversation from that position **onward**,
with a total map over the three consumers. The per-turn hazard's treatment is **censoring, not a
third state**: the conversation keeps its observations at `1 … t−1` and leaves the risk set from
`t`, which is deliberately *not* `cleanThroughTurnH`'s per-conversation ternary — at `H = 4` an
`unrunnable` turn at `t = 5` leaves the headline untouched and censors the hazard, and that pair is
the test that proves the two are two rules. Carry-forward rather than a per-position hole because
the estimand is a trajectory **and** the replayed history downstream of a reply-less turn is not
the pack's declared stimulus. **The censoring may be informative** — an HTTP 400 from a model's own
runaway message list is caused by what the hazard measures — so §4.6 prints `c_t` at every position
and a two-sided imputation bound wherever `c_t > 0`, and §4.4's per-position `n` becomes the
**observed** count with the structural one beside it. Rule 5 deliberately does **not** reach §4.2's
turn-level counts, which measure behaviour rather than a trajectory; that boundary is disclosed
with one funnel line instead. §4.3.1 item 11 carries the plan's edits and the four tests, including
a replayed-history gap the sweep found: nothing anywhere says what a reply-less turn contributes to
the next turn's history, and the natural implementation repairs it from the script.
**The two raises, both ruled true.** **R-1: `Y_calls` was netted by accident, and it
unnets.** Under this note's own pin `callCount == len(chatResults)`, the quantity §11.4 called *"the
run's total model calls"* counts only the calls that **returned** — so the coverage base for the
three `stats`-derived figures loses exactly the calls that could not have carried `stats`, which is
§4.3's laundering pattern one unit below where §11.4's own `Y` refuses it. It is not cosmetic: a
netted base leaves `X_calls == Y_calls` on every run whose only losses are failures, so rule
(iv-b)'s p50 gate cannot fire, and `Y_calls == 0` is reachable on the run with the least data. So
**`Y_calls` is the count of calls the run *attempted*** — `callCount + latencyWithheldForNoResponse`,
**derived, not stored** — while `callCount` keeps its pin and its meaning (calls that returned).
Five sentences of this note were false under the netted reading; the two §11.4 owns are rewritten
here (its `callCount is 1` claim, and its bound, which under the correction **is** plan v1.27's
`statsCoveredCount ≤ callCount`), and §11.7, §11.8, §4.2(f) and §4.3.1 were already written for the
attempted reading and become true rather than changing. §11.10 (7d)'s fixture is re-derived: it set
`Y_calls == Σ callCount` by construction and so could not separate the two counts it exists to pin.
**R-2: the cite drifted, and the durable defect is the form** — `modelbench/lmstudio.py:225` is
`:247` four days later. Measured at `d71c83e`: two of this note's three unpinned `modelbench` line
cites no longer resolve, both sha-pinned ones do, so a code citation here is a symbol plus an
enumerating command with its count, or a line pinned to a named sha. One ruling arrives unasked and
is R-1's dependency: plan-gate **P15-3**'s missing precedence is settled here — a turn that ended on
a raise takes `withheldFor` from the **failing disposition**, never `"load"` — because the
attempted-call identity is off by the overlap without it.

2026-09-10 (v1.21, `data-scientist`) — plan-gate Pass 14's open question 1 (`P14-6`'s second half),
ruled — and it is **not** downstream of `P14-1`: the dependency runs the other way. Ruling the
question surfaced a **live three-site contradiction inside plan v1.26** that no gate has caught,
and §4.3 rule 4 cannot be written around it, so it is ruled here too.

**The `I(t)` question.** **The mean and p95 of `I(t)` are computed over turns dispositioned
`replied` or `cap-hit`, and over no others.** Four reasons, the second and third each decisive
alone: those two are the mechanisms in which the model's own behaviour or the harness's declared
cap ended the turn, so they are the only ones carrying an observation of *stopping* behaviour; a
non-completing turn contributes `I(t) = 0` — **forced**, since v1.20 pinned
`iterations == len(chatResults)` — and **`0` there means *not observed*, never *zero
iterations***, this component's absent-never-zero rule (`coldLoadSeconds`, `scored_outcome`'s
refused absent count) arriving inside a mean; including them biases the mean **down**, so a server
that rejects a model reports that model as *better* at stopping, §4.3's laundering with the sign
reversed; and nothing is lost, because the *cost* question already has its own unrestricted
denominator in §11.4's `Y_calls / Y`. **§4.2(f)'s denominator statement did need correcting, and
that is what made the question unanswerable** — it named one denominator and reported three
statistics, only one of which used it. It now carries three, over **three different subsets** of
the mechanism set. That **sharpens `P14-1`'s prescribed fix rather than waiting on it**: three
figures are keyed on the mechanism rather than on `E(t)`, not one. **I concur with `P14-1` and add
what it does not use: §4.3's funnel already routes its case correctly** — a `cap-hit` turn with
`|E(t)| = 0` lands under *no attempt* — so the plan's four-row table is a second home for a mapping
that already had one, and it is the second home that is wrong.

**The contradiction, and it is the plan's, not this note's.** Plan v1.26 §3.6's fourth disposition
rules that *"a non-2xx response, a dropped connection or an unparseable body"* scores **`fail`,
never `n_a`**, while §3.8.4's table and §4 S5 both route a status-bearing `LMStudioCallFailed` to
**`unrunnable`** — opposite dispositions for one event, and not a wording difference: `fail` keeps
the turn in the denominator as a loss, `unrunnable` removes it. Verified live at committed
`1842b1d` (`:1577`, `:2243`, `:4978`). **Rule 4 decides it with a discriminator rather than a
case list: a turn scores `fail` only where the harness gave the model its whole declared budget and
observed nothing come back** — which is the **timeout**, and nothing else. Every other
non-completion is a channel failure the harness cannot attribute (a `400` from a model's runaway
message list and a `400` from a malformed harness payload are the same status code) and is
`unrunnable`. **This is §11.5.1's own distinction reaching a second consumer** — *a timeout is a
censored observation and a call that failed at 40 ms is a missing one*, ruled at v1.14 for
`censoringExact`. Two consequences: the mechanism set is **five**, not four (v1.26 folds `timed-out`
into `no-response`, which leaves the scorer unable to tell a `fail` from an `unrunnable`); and
§4.6's `cleanThroughTurnH` gains a **third state**, since *"zero failure of any kind"* would
otherwise read an `unrunnable` turn as **clean** — the very escape §3.6 feared, arriving at the
headline. P4-7's accounting fix is untouched: only the sentence assigning the scored outcome moves.

**Nothing already committed is invalidated.** `-ml` v1.20 stands unchanged, and none of §11.9 ask
7's twelve items moves — the *timing* withholding split (`withheldFor: timeout | no_response`) is
orthogonal to the *scoring* outcome and rule 4 cites it as precedent rather than touching it. New
**§4.3 rule 4** and new **§4.3.1**, the plan-side edit list.

2026-09-09 (v1.20, `data-scientist`) — plan-gate Pass 13's `P13-2`, ruled, together with the two
arithmetic errors of this note's own that the finding inherited. **Plan v1.25 stopped a scored item
being a model call, so every FR-11 figure is redefined on the unit it is a property of** — the wall
clock on the **item**, the three `stats`-derived figures on the **call** — with two coverage
numbers, two denominators and two nouns in §11.7's block, and never mixed *(§11.4, §11.7)*.
**§11.5.1's detector was never a per-call rule; it is a matched-bracket accounting identity, and the
loop is the first place the two spellings differ.** `unexplainedMs` becomes the **sum over the
item's calls** of `wallClockMsᵢ − (ttftMsᵢ + generationMsᵢ)` — v1.9's expression verbatim at
`callCount == 1`, so the 3 485.6 ms / −11.3 … +7.6 ms measurement, the 1 000 ms threshold and every
fixture survive untouched — and it is the **only** form that preserves the detector's own
false-negative bound: under a per-call *maximum* rule an 8-iteration turn retains up to 8 × 999 ms of
foreign time with every one of its calls passing. **The gate's mechanism is confirmed; its size was
understated in one direction and overstated in the other.** Understated: the withheld set is not
three items but **every multi-iteration turn**, which on the `tool-caller` pack is correlated with
the behaviour the pack scores, so the surviving latency sample would be the turns where the model
did *not* loop. Overstated: at `Y = 38` three withholdings refuse the **tail** figure alone and four
refuse both, so this note's own *"a clean run prints no latency summary at all"* was off by one —
corrected, with the per-pack budget tabulated in §11.6 rather than asserted from one example.
**`Y` for the `tool-caller` pack is 80, not 12** — its items are turns and have been since plan
v1.4 — so §11.3's *"the tool-caller pack's 12 conversations"*, §11.6's *"every pack in §3.3's table
has `Y ≥ 12`"* and §11.9 ask 4's *"every tool-caller run"* were all reading the **analysis-unit**
count as the **item** count, three instances of one substitution. Corrected, and it closes a
reachability question nobody had asked: the identity floor's `max` label is reachable only at
**`Y ≤ 21`** (exact, swept), therefore on **no** declared pack. Every reach claim this revision
introduces or repairs is pinned in an asserted constant, per the component's convention — the
threshold's margin against `maxIterationsPerTurn`, the `Y ≤ 21` bound, and the two denominators'
nouns (§11.10). §4.5.2's minutes are restated as a **floor** (plan-gate P13-7), and §11.9 gains
**ask 7**, the whole plan-side edit list this ruling implies, in one place.

2026-09-08 (v1.19, `data-scientist`) — plan-gate Pass 8's `P8-1`, adjudicated at the altitude the
reviewer routed it to *(new §3.4 Rule 4a)*. **A support is the parameter space of the estimand, so
it is applied once, to the interval that is printed, and never to an arm of a composition** — which
holds for both carriers, since §3.2d's continuous path already does exactly that and only the
paired-binary envelope moves. **The arms-versus-composed choice is immaterial to the statistics:**
clamping and composing commute exactly (`max(L,·)`/`min(U,·)` are non-decreasing and `min`/`max`
select rather than compute, so the identity is exact in IEEE-754), verified over all 173 472
`(table, DEFF)` combinations at n ∈ {12, 30, 38, 40} — zero differences, and zero changes to
coverage, width, point containment, zero-exclusion or any verdict. **So this is a reporting
correction, not a statistical one**, which scopes it as cheaper work than `P8-1` implies. What is
*not* cheap to get right is the audit: the defect is **larger** than the tie-break `P8-1` measured —
every bound the clamp moves is printed beside a sentence naming an arm that did not produce it, 58
of them at n=40 / DEFF 1.5 against `P8-1`'s 6, **all 58 on `distinguishable` verdicts** — and
**neither** of `P8-1`'s two suggested fixes closes it, with `(0, 0, 38, 2)` at DEFF 1.5 the
separating case where attributing from the unclamped arms is confidently wrong. `bound_by` therefore
takes a **third token, `"support bound"`**, on a strict comparison against the support, with ten
named assertions and the tie-break pinned at last (Pass 8's mutation 6). Closes `P8-5` as collateral:
compose-and-clamp gets one home.

2026-09-07 (v1.18, `data-scientist`) — plan-gate Pass 8's `P8-1`, the half that is this note's
under §7 rule 3. **A quantile level is an exact rational — `percentile(values, *, level:
Fraction)` — and `stats.py:159` is not exempted from it.** `permille: int` cannot express a
family-corrected `α/(2k)` (12.5 ‰ at k=2, 8.33 ‰ at k=3) and **no decimal unit can**: `1/120 ·
10^m` is never an integer, so widening the unit fails at the first k divisible by 3. Rounding the
level outward is sound and is **rejected on price** — it owes a printed attained level in every
continuous verdict string and borrows §11.6's machinery for a quantity §11.6 does not measure —
where a numerator/denominator pair is exact for every k and keeps §11.2.1's integer rank verbatim.
The **exemption is refused on a measured cost, not on scope**: what `:159` would keep is not a
different unit but the different *estimator* §11.2 rejects, which at `B = 10 000` selects the
**adjacent** order statistic to type 1's at the lower bound for k = 1, 2 and 5 and the same one at
k = 3 and 4 — a one-sided, k-dependent, anti-conservative shift on the one path where the interval
**is** the test. **§11.2.1's own attribution is corrected**: over levels `n/1000` and `X ≤ 3000`
the **1626** divergences belong to the percent spelling, the level-first spelling gives **755**,
and the expression v1.17 printed gives **0** — a sweep run against an expression the code does not
use, this note's own warning landing in the section that quotes it. **§11.10(3)'s
one-implementation rule is restated repo-wide** *(new §11.2.2)*, which is the clause the cheap fix
leaned on.

2026-09-07 (v1.17, `data-scientist`) — plan v1.13's §7 rule 3 raise, ruled. **Rule 8 gains
`support: tuple[float, float] | None`, keyword-only and required with no default — and still takes
no `clamp`.** That does not cut against v1.16's refusals: those four split into *the quantity does
not exist on this path* and *the quantity is derivable from what the function already holds*, and
`support` is neither — it is an **irreducible fact about the metric**, undiscoverable from `diffs`
(a sample of MRR differences in `[−0.3, 0.3]` is indistinguishable from a sample of z-differences),
which is `design_effect`'s own principle: `_widen` can no more discover a support than a resample can
discover clustering. The **clamp is derived inside the function** and exposed nowhere, so the sign
and order of `(lo−hi, hi−lo)` are written once. `support=None` is a **stated** value meaning
*unbounded* — `sep_z`'s case, no clamping — never an absent one. The plan's bounded/non-blocking
classification is **endorsed with its premise tightened**: rest it on the scale-1.0 identity, which
holds for every metric, not on the pack census, which a new pack changes silently.

2026-09-07 (v1.16, `data-scientist`) — four items back from plan v1.12 (`5b67416`). **§3.2f is
swept** (item 1): both *"Decided by the cluster-bootstrap CI…"* variants, their selecting condition
and the two prose sentences in §3.2e and §3.4 that named the same instrument now say **conservative
envelope** and what it is an envelope *of*, closing §3.4 Rule 4's own sweep obligation against the
one surface an implementer copies verbatim. **New §3.4 Rule 8** (item 2) specifies the
continuous-verdict producer `continuous_verdict()` — its parameters, its **negative** parameters
(no `ResolvingPower`, no `alpha_step`, no McNemar *p*), a `ContinuousVerdict` return that is a
sibling type rather than a `Verdict` with five meaningless fields, four refusals, and the two engine
changes `paired_cluster_bootstrap` needs to serve it. **Item 3 confirmed with the note's wording
moving**: a manifest carries no records, so `validate` cannot decide a metric's kind and a manifest
`kind` field would be the second declaration §3.2d refuses — enforcement moves to `compare_report`
pass 1, and a mixed family **refuses the whole family's verdicts** rather than shrinking `k` after
the results exist. **Item 4 specified** (§5.2): `sep_raw` prints median, p10 and the fraction above
zero, no mean anywhere, and **no cross-model difference is printed for `sep_raw` at all** — the
shared carrier makes it comparable, and it is the quantity `sep_z` exists because you cannot compare.

2026-09-07 (v1.15, `data-scientist`) — plan-gate Pass 6's blocker P6-1, both halves. **The
continuous instrument's carrier is specified in §3.2d**: a *second* per-item map,
`measures: Mapping[str, float]`, never a widened `counts` — widening makes the booleanisation
type-legal without making it wrong; finite floats with **no domain constraint at the carrier**; a
metric name in `counts` **or** `measures` and never both, which is what makes instrument selection
total; absence stays `scoreable`'s job, so `0.0` is a measurement and `measures` is **not**
`float | None`; `scored_value` as `scored_outcome`'s sibling, and `scored_outcome` **raises** on a
continuous metric, which is the one line that turns P6-1's silent booleanised-MRR verdict into a
loud failure; `diffs` taken **per analysis unit**, with a one-arm-only unit excluded and counted as
§4.3 asymmetry. `separationZ` takes the **same per-item carrier and a different aggregate one** —
§5.2 publishes a median, a p10 and a fraction, none of which is `ContinuousMetric`'s mean — and it
is reported, not verdicted, so Rule 4's `_widen` clamp is due with the `sep_z` comparison rather
than with S3. **§3.2e owes not one string but two** (open question 2): the decision stays binary, so
verdicts 4 and 5 render *distinguishable* and *not distinguishable* with no `pp`, no McNemar clause
and no MDD — a continuous metric has no observable floor, and the resolving sentence is the
interval's own width, stated descriptively and never as a power claim. §3.3 gains the rule that
makes both safe: **a `verdictMetrics` family is homogeneous in kind**, because Holm orders by
p-value and a continuous verdict has none.

2026-09-07 (v1.14, `data-scientist`) — the plan gate's Pass 5 routed three method questions here.
**Co-presence (P5-5): the conservative single count, not a separate `prefillCoveredCount`** — an item
with `stats` but no usable `promptTokens` leaves `statsCoveredCount` **and** all three medians, since
excluding it from the count alone would print a denominator that does not describe its numerator; the
case is defensive against something that should not occur, and a permanent second denominator is the
wrong price for a rarity. One consequence the plan must carry: §4 S2 rule (iv)'s identity becomes an
**inequality**. **`censoringExact` (P5-6): clause 1 survives in substance and needs a third
`withheldFor` value** — v1.10's merge is right for the counter and wrong for the item, because a
timeout is a *censored* observation and a call that failed at 40 ms is a *missing* one; the predicate
gains an explicit false branch for the second, and slot 3's weaker string names both mechanisms.
**`paired_cluster_bootstrap` (P5-3): keep** — the closed form retires the binary paired interval and
only that one. It is **§3.2d's entry point for every continuous verdict**, `paired_bootstrap` is its
engine and not a second entry point, and it takes the same discriminator as `sampling.seed`. One
condition stated with it: `_widen`'s `[-1, 1]` clamp is a difference-of-proportions assumption and is
wrong for `sep_z`.

2026-09-06 (v1.13, `data-scientist`) — plan v1.10 (`3e5dc50`) routed two items back. **§4 S2's rule
(iv-b) is confirmed — and it corrects this note rather than merely applying it.** §11.4 sent `ttftMs`
and prefill to §11.6's p50 gate while leaving `tokensPerSecond` with a denominator only; the three
share **one** coverage number, so they print or refuse together by construction, and exempting the
diagnostic would print a median over precisely the subset the gate had just declared too short to
describe the run. The split is **withdrawn** and all three take the gate. The transfer itself is
sound for the reason v1.12 changed: §11.5's level bound is now distribution-free, so it needs nothing
about *why* an item lacks `stats` — at v1.11 it would have needed the missingness direction, which
for a `stats`-less item is unknown. Confirmed **conditionally on co-presence**, with the assertion
that makes the condition checkable. And §11.2's citation of the second shipped `_percentile` is
corrected to **`results.py:573`**, both copies pinned at `5878014`.

2026-09-06 (v1.12, `data-scientist`) — the plan gate's Pass 4 open question 2 and its two routed
minors, ruled. **§11.5's exactness argument does not extend to §11.5.1's detector**: it withholds on
a *covariate* rather than on the wall clock, so *"every withheld call was slower than every timed
call"* is not true of every render — and it fails on the ordinary case of the very pack that will
fire it, not a corner one. The attained-level bound is re-derived without the ordering assumption
(it needs only that the timed items are a subset of the run), so §11.6's floor, its 5-point constant
and §11.5's table are untouched; §11.7's slot 3 gains a second variant and a **computed** selector,
and two slot-4 clauses that had the ordering baked into them are rewritten. §11.5.1's threshold
**survives at 1 000 ms** and its *"~3.5× below the smallest cold load"* margin is **withdrawn** —
plan-gate P4-11 is right that the two cold loads differ in model, quantization *and* route, so the
data bounds no load from below — replaced by the asymmetry of the detector's two error costs.
§11.7's second denominator line renders only where the call surface produces a `stats` object, on a
`None`-never-`0` carrier (plan-gate P4-13, confirmed, on the surface rather than on the profile).

2026-09-03 (v1.11, `data-scientist`) — §3.4 Rule 4's closed-form percentile becomes **binding**, and
the reason is not the one v1.8 gave: measured against the shipped `stats.conservative_envelope`
(`5878014`), the envelope's seed dependence reaches the **verdict**, not only the printed digits —
at `(a=1, b=25, c=12, d=2)`, n=40, DEFF=1.2, past both the McNemar veto and Rule 7's floor, the
published sentence alternates between *distinguishable* and *not distinguishable* on 80/70 of 150
seeds and on 85/65 of 150 row permutations **at one fixed seed**. Rule 4's *"the wider of"* is
published as a **bound-by-bound** envelope, its *"reduces to MOVER-D exactly at `DEFF = 1.00`"* is
withdrawn as false (the resample arm binds a bound on 83–87% of tables), the closed form's quantile
levels are pinned at `permille` 25 and 975 in exact integer arithmetic, and the four printed strings
that call the interval *"cluster-bootstrap"* are corrected — the interval is an envelope, and naming
one of its two arms is M-ML-8's defect one layer over. `PackRef.seed` and `sampling.seed` **stay**;
only their object moves, to §3.2d's continuous-metric bootstrap.

2026-09-03 (v1.10, `data-scientist`) — §11.7's renderings are **measured** (50 warm calls against
`qwen/qwen3-4b-2507`, following one cold load), and taking them reversed two of v1.9's own rulings:
LM-Studio-side TTFT **excludes** the JIT load (cold call 3 625.0 ms wall against 49.7 ms `ttft`), so
§11.4 keeps `ttftMs`/prefill/`tokensPerSecond` on a contaminated item with **their own denominator**
rather than withholding them, and slot 3 stops naming a load cost — v1.9's *"about 21 s"* is false
of a 3.625 s load. New **§11.5.1**: the same gap is a detector for R-14's in-call-reload residual
(3 485.6 ms cold against −11.3…+7.6 ms warm, 461×), with a 1 000 ms threshold and its basis.

2026-09-03 (v1.9, `data-scientist`) — plan R-13 closed on all three of its inputs: new **§11**
settles the latency percentile as **Hyndman-Fan type 1** (inverse ECDF, integer-ceiling rank, one
implementation shared by both call sites), fixes the denominator at the unnetted item count with the
withholding applied per *item* across the whole FR-11 timing block (review G3-3), and sets **two
floors** — an identity floor at X = 20 that **renames** a tail figure `max` where a p95 would be the
maximum (review G3-11), and a level floor that makes a figure **absent** where its attained level
falls more than 5 points below its nominal one. The report block is published as a verbatim clause
grammar. §3.2d gains a pointer to the estimator; nothing else changes.

2026-09-03 (v1.8, `data-scientist`) — the clustered path's four unowned or false sentences: §3.2e
publishes the cluster-path label verbatim (it was prose invented in code) and stops it asserting
clustering at DEFF 1.00, verdict 2's alternate clause becomes *"at or above that"* (equality is
reachable), §7.1's floor sentence gains its effective-unit qualifier where DEFF > 1, and §3.4
Rule 4 replaces the bare percentile interval with the **conservative envelope** — review Pass 4
(M-ML-8, m-ML-9, m-ML-10, m-ML-11).

2026-09-03 (v1.7, `data-scientist`) — four stale or unhandled corners the v1.6 fix round surfaced:
§3.4 Rule 2's dataclass sketch and Rule 4's precondition 3 now say **two** αs on `ResolvingPower`
(not three) and Rule 4 gains its fifth precondition, §3.2e verdict 2's closing clause becomes
conditional (it was false whenever the observed difference exceeded the strict-dominance MDD),
§7.1 states that the floor sentence prints independently of the MDD's attainability, and §7.2's
rendered example is brought onto §7.1's v1.6 template. §3.2a's `3.0 × 10⁻⁴ pp` becomes `3.1` (it
was below the value it bounded — review n-ML-2, the last open Pass 1 item).

2026-09-03 (v1.6, `data-scientist`) — the observable floor moves to the **unadjusted α** (review
M-ML-6), §3.4 Rule 3 is generalised into the printed-bound principle that governs it, Rule 7 splits
by path, and Rule 4 states that McNemar is permitted as a **veto**.

2026-09-03 (v1.5, `data-scientist`) — §3.2c's fixtures republished at 10 dp so the mandated 1e-9
tolerance is assertable (review m-ML-1), the floor/MDD rounding directions separated and three §7.1
floor cells corrected, and §3.4 gains Rule 7 (no `distinguishable` below the observable floor).

2026-09-03 (v1.4, `data-scientist`) — §4.6 adopts the plan's `H` semantics (manifest-declared,
validated `H ≤ min(script length)`, no longer *equal to* the minimum), closing review N-3; pairs
with plan v1.4.

2026-09-02 (v1.3, `data-scientist`) — `primaryMetrics` renamed **`verdictMetrics`** throughout
(`architect`'s naming authority, plan v1.3); semantics unchanged, `headlineMetric` unchanged.

2026-09-02 (v1.2, `data-scientist`) — tool-caller resampled to **12 distinct scripts × 1 run**,
`guard-judge` given **no headline metric**, S3's self-check pinned as diagnostic-only (three
stakeholder decisions), and the statistics-module contract fixed in **§3.4** so neither the
resolving-power line nor `verdict()` can be built from a raw *n* (gate B-1, M-1).

2026-09-02 (`architect`, outcome only — no method changed) — §8's three flags were accepted into
the requirements (commit `afe4aef`): §3.2's paired instrument is now what FR-15/AC-4 *require*,
§4.2's nesting is now FR-8(d), §4.5's floor is now FR-22a, and §6's judged layer is **deferred** by
FR-21a, so §6.2/§6.3 describe a design to be built later rather than one in first delivery.

## 1. The question and the decision it serves

The architect is writing the implementation plan for `model-bench/` against
`docs/requirements/small-model-benchmarking.md` (Status: Ready for design). Five method
questions in that document are under-specified or in genuine tension, and an implementer would
otherwise have to invent the statistics. This note settles them so the plan can name exact
formulas, exact denominators, exact decision wording and exact sample sizes.

**Scope boundary.** This note does not redesign the requirements' measurement architecture (seven
tool-calling counts, per-turn reporting, system-ground-truth scoring, four retrieval indicators,
paired design). It resolves how those are *computed and reported*. Where a requirement is
methodologically unbuildable as written, §8 says so and names the amendment.

**Settled inputs this version encodes (stakeholder, 2026-09-02 — decided, not open):**

1. **Tool-caller sampling is 12 distinct conversation scripts, one run each, at temperature 0** —
   replacing 4 scripts × 4 replicates per shape. Taken on §4.5's own argument. §4.5 and §7.2 are
   rewritten around it; §4.5 also states, in full, what the lab gives up by taking it.
2. **`guard-judge` declares no headline metric.** Both class-conditional error rates carry a verdict,
   with equal weight and no single number above them (§3.3, §7.3).
3. **The S3 embedder self-check is a diagnostic, never a gate** (§5.4).

**Assumed design decisions (architect's, accepted here as sound):** in-process brute-force exact
cosine over a copied ~121-doc corpus with a BM25 keyword arm; simulated deterministic tools with
in-harness state as the tool-caller's ground truth; pack-versioned prompt assembly. All three are
methodologically correct for the stated object of measurement — the *model*, not a product
pipeline. Nothing below asks to change them.

**Dependency budget.** Every statistic in this note is implementable in **Python 3.12 stdlib
only** (`math.sqrt`, `math.comb`, `random`, `statistics`). **No scipy, no statsmodels.** `numpy`
is worth taking for the embedder pack alone (a 121×1024 similarity matrix per query set); the
statistics layer must not import it, so the stats module stays pure and unit-testable exactly
the way `falkor-chat/server/tests/eval/metrics.py` and `nlq_scoring.py` already are.

---

## 2. Findings from the real system

Read in full: the requirements document; `falkor-chat/server/tests/eval/{metrics.py,
nlq_scoring.py,judge.py}`; `retrieval_baseline.json`; `corpus_provenance.json`;
`golden_retrieval.jsonl` (38); `golden_guards.jsonl` (85); `golden_judge_calibration.jsonl` (10);
`nlq_golden_set.jsonl` (40); `judge_calibration.json`; and
`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.

Measured/derived facts that drive the answers below (all computed in this session from the
committed artifacts, none quoted from memory):

| Fact | Value | Where from |
|---|---|---|
| Golden-retrieval relevance cardinality | 36 of 38 items have exactly **one** relevant id; 2 have two | counted from `golden_retrieval.jsonl` |
| Pinned baseline | recall@10 = 0.9737 (37/38), recall@5 = 0.8947 (34/38), MRR = 0.6259, n=38 | `retrieval_baseline.json` |
| Corpus | 121 messages, 12 topics, `text-embedding-qwen3-embedding-0.6b`, dim 1024 | `corpus_provenance.json` |
| Guard set composition | `clear_suspend` 40 · `clear_advance` 30 · `boundary` 15; `expected=False` 55, `expected=True` 30 — **every boundary item is an expected-suspend** | counted from `golden_guards.jsonl` |
| NLQ set composition | **40 items** (21 scalar / 13 set / 6 not_found across 7 shapes) — of which **34 answerable** (17 scalar / 11 set / 6 not_found across 5 shapes; 21 catalog / 13 knowledge_base) and **6 structurally unanswerable** (4 `relationship-traversal` + 2 `conflicting-facts`, plan §3.8.3). **`n_eff = 34`** for every inferential bound (§7.1, §7.2); **`Y = 40`** for the executed-item counts §11.6's latency coverage and the `Y ≤ 21` reach bound use — different denominators, both correct *(v1.25)* | counted from `nlq_golden_set.jsonl`; exclusion per `model-bench/docs/reviews/nlq-conflicting-facts-answerability-ml.md` |
| Judge calibration, faithfulness | raw agreement 9/10 = 0.90, Wilson95 [0.596, 0.982], **Cohen's κ = 0.833** | recomputed from `judge_calibration.json` |
| Judge calibration, relevance | raw agreement 7/10 = 0.70, Wilson95 [0.397, 0.892], **Cohen's κ = 0.211**; false-positive rate (judge says relevant when gold says not) **2/3** | recomputed from `judge_calibration.json` |
| Judge conflict of interest | `"sameModelAsAgentUnderTest": true` — judge and agent-under-test were both `qwen/qwen3-4b-2507` | `judge_calibration.json` |
| Effect sizes this lab has actually cared about | qwen3-4b turn-4 collapse 97.5% vs ministral 0/176; ministral duplicate-instruction 30% | review §8.2/§8.4 |

Three of these are decision-changing and are used repeatedly below: the **relevance-axis κ of
0.21**, the **saturation of recall@k on the 38-item set**, and the **class-conditional smallness of
the 85-item guard set**.

---

## 3. Q1 — FR-15 vs FR-16: what the tool computes and prints

### 3.1 The finding: FR-15's literal rule is not conservative, it is inert

FR-15 as written ("two models are declared different only when their intervals don't overlap")
reads as a safe, conservative rule. At the sample sizes FR's own out-of-scope section commits to
(~20–40 per arm), **it is not conservative — it is incapable of ever firing in the regime this
lab actually operates in.** Computed here, minimum second-arm score needed for two marginal
Wilson intervals to separate:

| n | baseline 0.50 | 0.70 | 0.90 | 0.95 |
|---|---|---|---|---|
| 20 | needs 0.950 (Δ 45 pp) | **impossible** | **impossible** | **impossible** |
| 30 | 0.867 (Δ 36.7 pp) | 1.000 (Δ 30 pp) | **impossible** | **impossible** |
| 40 | 0.800 (Δ 30 pp) | 0.950 (Δ 25 pp) | **impossible** | **impossible** |
| 85 | 0.706 (Δ 21.2 pp) | 0.871 (Δ 17.6 pp) | 1.000 (Δ 10.6 pp) | **impossible** |

"Impossible" means: *even a model scoring 100% cannot separate from the baseline under this rule.*
At n=40 with an incumbent at 0.90, a candidate at a flawless 40/40 still "overlaps."

The worked case to put in the plan:

> **40/40 vs 34/40, perfectly nested (the candidate gets every item the incumbent gets, plus 6).**
> Marginal Wilson: [0.912, 1.000] and [0.709, 0.929] — **overlap**, so FR-15's literal rule prints
> "not distinguishable."
> Paired: b=6, c=0. McNemar exact two-sided **p = 0.031**. Paired difference **+15.0 pp,
> 95% CI [3.2, 29.1] pp** — excludes zero.
>
> The candidate strictly dominates on every item and the literal rule cannot say so. That is not
> caution; it is a rule that discards the entire experimental design FR-16 mandates.

The reason is mechanical: two marginal intervals overlapping is a *much* stronger condition than
their difference covering zero, and the marginal intervals throw away the item-level covariance
that pairing exists to capture. FR-16 pays the cost of pairing; FR-15 as literally worded then
refuses to bank the return.

### 3.2 Recommendation

**Read "the interval" in FR-15 and AC-4 as the 95% confidence interval on the *paired
difference*, not as two marginal intervals.** This preserves AC-4's intent exactly ("don't rank
on noise") and states it correctly. §8 carries this back as a requirement amendment for `tico`.

Compute and print, in this order:

**(a) Per-arm reporting — Wilson score interval, unchanged.**
Reuse `nlq_scoring.wilson_interval` verbatim (`_Z_95 = 1.959963984540054`; this lab's convention,
not Clopper-Pearson, not rule-of-three). Every rate prints as `k/n = p̂ [lo, hi]` — never a bare
percentage, never without its denominator. These are **descriptive**: they say what each model
scored. They are explicitly **not** the comparison instrument, and the report must say so in one
line under the table.

**The constant, settled (gate M-1).** **`z = 1.959963984540054`** is authoritative. It is
`Φ⁻¹(0.975)` — verified in this session: `statistics.NormalDist().inv_cdf(0.975)` returns
`1.9599639845400536` — and it is what `falkor-chat/server/tests/eval/nlq_scoring.py:59` already
pins as `_Z_95`. **`1.96` is a typographic rounding of that number, not a competing convention**, so
the apparent split in this lab's prose (the salesperson review says "1.96") is a split in how the
same constant is written, not in which constant is meant. Implement it as a module constant with
`z` **keyword-only**, exactly as `nlq_scoring` does.

**What the precision is worth — measured, not asserted.** Recomputing §3.2c's five regression
fixtures at both values moves every MOVER-D bound by **at most 3.1 × 10⁻⁴ pp** (measured
3.0167 × 10⁻⁴ pp, on the `34,6,0,0` row's **upper** bound; v1.2–v1.6 printed `3.0 × 10⁻⁴`, which is
*below* the value it claims to bound — an *"at most"* bound rounds **up**, §3.4 Rule 3, review
n-ML-2). That is invisible at the 0.1 pp the report prints and invisible at any
tolerance looser than ~10⁻⁵. **So the fixtures pass under either constant, and M-1 is not a
numerical defect.** The reason to pin the exact value is reproducibility of an *equality* assertion:
two modules carrying two constants disagree in the fifth decimal forever, and the day someone
tightens the fixture tolerance (§9.1 does — see below) the failure appears in a module nobody
changed. Pin one constant; pin this one.

**(b) The decision rule — McNemar's exact test on the paired 2×2, for binary metrics.**

Build the paired table over items scored by *both* models in the *same session* (FR-16):

```
                 B correct   B wrong
  A correct         a           b
  A wrong           c           d      n = a+b+c+d
```

`b` = items A wins, `c` = items B wins. Only the **discordant** pairs carry information.

```python
from math import comb
def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact conditional (binomial-sign) test. b, c = discordant counts."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(m, i) for i in range(k + 1)) * (0.5 ** m)
    return min(1.0, 2.0 * tail)
```

Exact, not the chi-square approximation — the chi-square version is invalid at the discordant
counts this lab will actually see (b+c often < 10). Stdlib only.

Its small-sample floor, computed here, is the single most useful number for the whole tool:

| discordants against (`c`) | minimum `b` to reach p ≤ 0.05 |
|---|---|
| 0 | **6** |
| 1 | 8 |
| 2 | 10 |
| 3 | 12 |
| 4 | 13 |

So **no paired binary comparison can ever be declared significant on fewer than 6 net wins**,
whatever n is. That is the honest hard floor and it should be printed.

**(c) The effect size — MOVER-D (Newcombe) confidence interval on the paired difference.**

Reuses the Wilson function the lab already has, so it is consistent with (a) by construction:

```python
def paired_diff_ci(a, b, c, d, z=1.959963984540054):
    """Newcombe's square-and-add (MOVER-D) 95% CI for p1 - p2 on paired binary data."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n          # A's rate, B's rate
    diff = p1 - p2                              # == (b - c) / n
    l1, u1 = wilson_interval(a + b, n, z=z)
    l2, u2 = wilson_interval(a + c, n, z=z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = ((a * d - b * c) / den) if den > 0 else 0.0     # margin-zero -> phi = 0
    lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, lo), min(1.0, hi)
```

The `max(0.0, …)` clamps are required, not cosmetic: the radicand goes slightly negative under
extreme `phi`. Worked outputs verified in this session:

| a | b | c | d | n | diff | MOVER-D lower (pp) | MOVER-D upper (pp) | McNemar p (exact rational) |
|---|---|---|---|---|---|---|---|---|
| 34 | 6 | 0 | 0 | 40 | +15.0 pp | 3.1762869443 | 29.0723243665 | 1/32 = 0.03125 |
| 30 | 6 | 0 | 4 | 40 | +15.0 pp | 3.8506738324 | 27.7026867131 | 1/32 = 0.03125 |
| 33 | 6 | 1 | 0 | 40 | +12.5 pp | −0.9864868353 | 26.8581964973 | 1/8 = 0.125 |
| 20 | 8 | 2 | 10 | 40 | +15.0 pp | 0.1708978316 | 28.7785182732 | 7/64 = 0.109375 |
| 72 | 10 | 2 | 1 | 85 | +9.4 pp | 1.4800198994 | 18.2130920778 | 79/2048 = 0.03857421875 |

Use these five rows as the implementer's regression fixtures for the statistics module. **Ten
significant decimal places in percentage points, i.e. 1e-12 as a proportion** — three orders inside
the tolerance below, which is the point (see the trap at the end of this subsection). All ten bounds
were re-derived at 60-digit precision at `z = 1.959963984540054` in this session and agree with the
independent re-derivation in `docs/reviews/small-model-benchmarking-ml.md`; the p-values are exact
rationals and are published as rationals so no decimal expansion is load-bearing.

**Assertion tolerance — settled, replacing both "exactly" and "3 decimal places" (gate m3).**
- `mcnemar_exact` is rational arithmetic over `math.comb` and `0.5**m`: assert to **1e-12
  absolute**. It is exact; a looser tolerance hides an implementation that is not.
- MOVER-D bounds: assert to **1e-9 absolute on the proportion** (i.e. 1e-7 pp). That is ~7 orders
  above double-precision noise for values of this magnitude, so operation-order differences cannot
  trip it, and it is ~4 orders tighter than the z-constant divergence above — meaning **this
  tolerance is what makes M-1's constant load-bearing.** "Exact match" is not assertable across
  platforms; 3 decimal places is loose enough to pass a genuinely wrong Wilson.

**The trap this note fell into, stated once because it generalises (review m-ML-1).** v1.2–v1.4
published these bounds at 4 dp *in pp* and mandated a 1e-9 *proportion* tolerance in the same
subsection. Those two statements are incompatible: 4 dp in pp is 1e-6 as a proportion, so the
published `3.1763` sits 1.31e-7 from the true `3.1762869443…` — **131× the tolerance it was supposed
to be asserted against**. The delivered implementation was never the problem; it agrees with the
60-digit value to 1.44e-16, about 7×10⁶ *inside* the mandate. The table was under-precise, not wrong.

> **A fixture table published for display cannot carry a tolerance tighter than its own printed
> precision.** The two numbers are one decision, not two, and they are usually written in different
> sentences by different people at different times — which is exactly how they drift apart. Whenever
> a tolerance is tightened, the fixtures must be republished at a precision that clears it, and when
> a table is published, the tolerance it can support is fixed at that moment. The safe default is to
> publish **at least 2–3 orders finer than the tolerance** so a later tightening has headroom, and
> to prefer **exact rationals** wherever the quantity has one (as the p-values above now do), since
> a rational carries no precision claim to get out of step.

**(d) Continuous metrics — paired percentile bootstrap on the per-item difference.**

McNemar does not apply to MRR, score separation, or latency. Use a seeded paired bootstrap:
resample the *items* (with replacement, n draws), recompute the mean per-item difference each
time, take the two quantile levels §3.3 fixes — `1/40` and `39/40` at `k = 1`, **exact rationals
and never percent floats** *(v1.18, §11.2.2)* — at B = 10 000 (**the percentile estimator is §11's,
and there is one of it, package-wide**). ~15 lines of stdlib (`random.Random(seed)`).
The seed goes into the environment fingerprint (FR-7) so a report is reproducible. Decision:
**the CI excludes zero.** No separate significance test — for continuous metrics the CI *is* the
test, and reporting both would be redundant, not extra rigour.

For **clustered** data, resample the **outermost independent unit, never the observation**. Under
the settled 12×1 sampling design (§4.5) conversation and script are the same unit, so the tool-caller
resample is **one-level over the 12 conversations**. The two-level (script → replicate) resample
returns the moment any pack declares `replicatesPerScript > 1`; §3.4 makes that a **validation
error** rather than a silently one-level approximation. See §4.4.

**What this instrument needs on the record, and none of it is there** *(v1.15, plan-gate P6-1)*. The
gate found the embedder's only verdict metric unbuildable: `counts` is `Mapping[str, int]` and
`scored_outcome` returns `counts[metric] > 0`, so a reciprocal rank of 0.5 is unstorable and any
positive one booleanises into *did this query retrieve anything*. The ruling, concrete enough to
specify a field:

- **A second per-item map, not a widened one.** `counts` stays `Mapping[str, int]`; continuous
  per-item values live in a new **`measures: Mapping[str, float]`**. Widening `counts` to `float` is
  the tempting one-word fix and it is the wrong one — it makes the booleanisation *type-legal*
  without making it wrong, and it puts two kinds of quantity in one key space: a count that §4.2's
  denominators count, and a measurement that they must not.
- **`float`, finite, and the carrier constrains no domain.** MRR is in `[0, 1]`, `sep_z` is a
  difference of z-scores and is unbounded — §3.4 Rule 4's `_widen` clamp is that same assumption
  made one layer up, and it must not be repeated here. A **non-finite** value is refused at the
  carrier, because one `NaN` propagates through the mean and both percentiles and arrives as a
  rendered interval rather than as an error.
- **A metric name lives in `counts` or in `measures`, never in both.** This is what makes the
  instrument selection **total**: the map a metric is in *is* the declaration of which instrument
  decides it, so nothing infers a kind from a name or from a pack field, and a name in both maps is
  refused rather than resolved by code order.
- **Absence stays `scoreable`'s job — this is the absent-never-zero answer for this field, and it
  is answered by machinery that already exists.** The three states are unchanged: absent from
  `scoreable` → no declaration, no value; `scoreable[m] is False` → a declared precondition failure
  and §4.3's asymmetry; `scoreable[m] is True` → the value **must** be in `measures` or the record is
  refused (`IncompleteItemRecord`), never read as `0.0`. **An MRR of `0.0` is a measurement** — the
  query retrieved nothing relevant in the top *k* — and it must be storable and distinguishable from
  a query nobody judged. So `measures` is `Mapping[str, float]` and **not**
  `Mapping[str, float | None]`: a second home for absence is a second thing to keep in step
  (§3.2c's trap), and the first home is already correct.
- **`scored_outcome` gains a sibling and a refusal.** `scored_value(metric) -> float | None`, the
  same three states. And **`scored_outcome` raises when called on a metric that lives in
  `measures`**, rather than returning `value > 0`. That is one line, and it is the line that turns
  P6-1's first silent outcome — a booleanised MRR printed as a McNemar `+X pp` verdict, a different
  metric under the same name — into a loud failure.
- **What the bootstrap consumes.** `diffs` is one difference **per analysis unit**, never per
  observation (the clustered rule above): for every unit present in **both** arms,
  `value_A(u) − value_B(u)`, where a unit's value is its item's measure when the unit is one item —
  the embedder's case, unit ≡ query ≡ item — and the **mean over its items** when it is not. A unit
  scoreable in one arm and not the other is **excluded from `diffs` and counted as §4.3's
  asymmetry**, exactly as the paired binary table already does; dropping it silently is the
  laundering §4.3 exists to prevent, arriving on the continuous path.
- **`separationZ` takes the same per-item carrier and a *different* aggregate one.** §5.2's
  comparison is a paired bootstrap on per-query `sep_z` differences, so it needs
  `measures["separationZ"]` (and `separationRaw`) exactly as MRR does. Its **aggregate** side does
  not fit `ContinuousMetric`, which carries a mean: §5.2 publishes a **median, a p10 and the fraction
  of queries with `sep_raw > 0`**, and none of the three is a mean, so one bare float cannot carry
  them — that is P6-1's defect one layer up and it needs a distribution summary. It is **not** on the
  embedder's verdict path (`verdictMetrics = ["mrr"]`, §3.3), so it is reported rather than
  verdicted, and §3.4 Rule 4's `_widen` clamp is therefore due **with the `sep_z` comparison**, not
  as a precondition of S3.
- **Latency does not go here, and the exclusion is worth one clause.** It is continuous and it is
  per-item, so it reads like a `measures` member — and it has its own carrier (§11.8's `ItemTiming`)
  and is in no pack's `verdictMetrics` by §11.7 slot 6. Putting it in `measures` would duplicate a
  number that already has a home and would make it eligible for a verdict this note has ruled it
  cannot have (§11.7's closing paragraph: no paired latency interval is printed on any path).

**How the carrier is proven right, and all of it is offline.** (1) `scored_outcome` on a metric in
`measures` **raises**, asserted from both maps — the test that fails if the booleanisation is ever
reintroduced. (2) A measure of `0.0` on a `scoreable: True` item survives a `to_dict`/`from_dict`
round trip **as `0.0` and not as absent**, and the same item with the key missing raises
`IncompleteItemRecord`: the absent-never-zero boundary, asserted on both sides. (3) A non-finite
measure is refused at construction. (4) Over two arms where one unit is scoreable in only one of
them, `diffs` has length `n_units − 1` **and** the dropped unit appears in the §4.3 asymmetry tally —
asserted together, because either assertion alone passes on a silent drop.

**(e) The decision wording, which AC-4 must be checked against.**

Three verdicts for the **binary** instrument, exactly these strings — the continuous path's two are
published below them *(v1.15)*, so this section carries five:

1. **Distinguishable.**
   `A is better than B on <metric>: +15.0 pp (95% CI [3.2, 29.1] pp), n=40 paired items (unit: item, design effect 1.00), McNemar exact p=0.031 (b=6, c=0).`
2. **Not distinguishable.**
   `Not distinguishable at this sample size. Observed difference +12.5 pp, 95% CI [-1.0, 26.9] pp covers zero (b=6, c=1, McNemar exact p=0.125). This pack resolves differences of >=19.1 pp with 80% power at n=40 effective items (40 units, design effect 1.00, by-construction, alpha=0.05); the observed 12.5 pp is below that. Neither model is ranked above the other.`

   **The closing clause is conditional, not fixed prose (v1.7).** `the observed X pp is below that`
   is only true when `|diff| < mdd80`, and the case where it is false is not exotic — it is §7.1's
   own *normal case for a model swap*, a candidate that wins more than it loses without strictly
   dominating. At n=30 with `b=5, c=13` the difference is 26.7 pp against an MDD₈₀ of 25.1 pp and
   McNemar's p is 0.096, so the sentence as fixed prose reads *"resolves differences of >=25.1 pp
   with 80% power; the observed 26.7 pp is below that"* — two numbers in one sentence that refute
   it. When `|diff| >= mdd80`, render instead:

   > `…; the observed 26.7 pp is at or above that, but the MDD assumes strict dominance and this comparison is not strictly dominant (b=5, c=13), so the difference required for 80% power at this discordance mix is larger (§7.1).`

   The discordance counts are mandatory in that form: they are the reason the two numbers point
   opposite ways, and without them the sentence looks like the instrument contradicting itself.

   ***"at or above", not "above" (v1.8, review m-ML-9).*** The branch is `|diff| < mdd80`, so
   **equality takes this wording** — which is the right branch, since `mdd_clause` claims the pack
   resolves differences of **≥** `mdd80` and at equality the difference is resolvable. But `mdd80`
   is ceilinged onto a 0.1 pp grid while an observed difference is `(b−c)/n`, so the two coincide
   exactly whenever the grid and the lattice meet, and **that is reachable at this component's own
   sample sizes** — measured this session: `n_units = 85`, `k = 2`, `DEFF = 1.9` gives
   `mdd80 = 20.0 pp` and `|b−c| = 17` gives exactly 20.0 pp; sweeping `6 ≤ n_eff ≤ 200`, equality
   occurs at (n=90, 100, 120, 150, 180) for k=2 and (n=200) for k=1, and at n=90 it is 2.4% of
   every table that reaches this clause. A strict *"is above that"* is then false in the same way
   *"is below that"* was, one branch over. **Do not add a third wording for equality** — one
   comparative that is true across the whole branch is worth more than a third string to keep in
   step with the other two.
3. **Instruments disagree** (MOVER-D excludes zero, McNemar does not — row 4 of the table above; real and not rare):
   `Not distinguishable at this sample size. The effect-size interval [0.2, 28.8] pp excludes zero but the exact paired test does not reach alpha=0.05 (b=8, c=2, p=0.109). Reported as not distinguishable: the exact test is the decision rule.`

   **The interval's name in that string is a substitution, and this is where it is published**
   *(v1.16 — it was named only in §3.4 Rule 4's sweep list, so an implementer copying this section
   never learned the word varies: the same trip hazard §3.2f had)*. It renders **`effect-size`** when
   `decided_by` is `mcnemar-exact` and **`conservative envelope`** when it is
   `conservative-envelope` — never `cluster-bootstrap`, which named one arm of a two-arm interval.

**McNemar exact is the decision; MOVER-D is the effect size.** One instrument decides, one
quantifies. Do not AND them into a bloc — but *always print both individual outcomes in the
prose*, as verdict 3 does, so a reader never sees an aggregate verdict without the two components
that produced it.

**The continuous path owes its own strings, and there are two of them** *(v1.15, plan-gate P6-1's
open question 2)*. The three above all carry `pp`, a McNemar clause and an MDD-or-floor sentence, and
a continuous verdict has none of the three: the unit is the metric's own (an MRR difference is not
percentage points), §3.2d rules there is no significance test, and §7.2 gives the embedder's floor as
`n/a (continuous)` because the observable floor is a property of a binary paired table's discordance
lattice and does not exist here. Rendering a continuous verdict through string 1 or 2 would be false
in three places at once. The decision is still binary — **the CI excludes zero** — so the path owes
**two** strings and not three: *"instruments disagree"* cannot arise where there is one instrument.

4. **Distinguishable (continuous).**
   `A is better than B on mrr: +0.048 (95% CI [0.011, 0.086]), n=38 paired queries (unit: query, design effect 1.00, by-construction). The interval excludes zero; for a continuous metric the interval is the test and no significance test is run. Interval: paired bootstrap, B=10000, seed=<n> from the pack's sampling.seed.`
5. **Not distinguishable (continuous).**
   `Not distinguishable at this sample size. Observed difference +0.021 on mrr (95% CI [-0.014, 0.057]), n=38 paired queries (unit: query, design effect 1.00, by-construction). The interval covers zero. No power threshold is computed for a continuous metric — the observable floor and the MDD are properties of the paired binary table — so the resolving statement is the interval's own width: differences much below 0.036 are not separable from zero by this instrument at this n. Neither model is ranked above the other. Interval: paired bootstrap, B=10000, seed=<n> from the pack's sampling.seed.`

**Four properties of that pair, each a decision rather than a wording preference.** (i) **The unit is
the metric's own and never `pp`** — an MRR difference rendered in percentage points invites a reader
to compare it against a rate difference elsewhere in the same report, and they are not the same
quantity. (ii) **The absence of a significance test is printed, not implied**, because a reader
trained on strings 1–3 reads a missing *p* as an omission. (iii) **The resolving sentence is
descriptive and never a power claim** — a half-width is not an MDD, which is why it says *"not
separable from zero by this instrument"* and not *"with 80% power"*; §7.4's `sd_d`-dependent table
stays what it is, a sizing aid computed from a run rather than a verdict clause. (iv) **The resample
provenance is mandatory in both** — an interval that does not name `B` and its seed is not
reproducible, which is the whole reason §3.4 Rule 4 kept `sampling.seed` alive after the binary path
stopped resampling.

*(The numbers in strings 1–5 are **grammar, not measurements**: §3.2e publishes shapes an
implementer tests against, and only §11.7's renderings are pinned to a measured sample. The one real
figure above is §7.4's incumbent MRR of 0.6259, which is what makes `+0.048` a plausible shape rather
than an arbitrary one.)*

**One precondition on all three binary strings, and it is the whole of gate B-1:** McNemar exact and
MOVER-D are valid **only when each row of the paired table is one independent analysis unit**. When
the analysis unit contains correlated observations (design effect > 1), both are anti-conservative
and the decision rule becomes *"the conservative envelope on the paired difference excludes zero"*
— MOVER-D and the exact paired bootstrap, the wider of the two at each bound (Rule 4) — with the
strings rendered against the envelope instead of McNemar and the design effect and its basis
printed. §3.4 makes this a property the code cannot get wrong by omission.

**(f) The clustered-path label — published here, because Rule 4 required it printed and this note
published no string for it (v1.8, review m-ML-10).** It is **appended to whichever of the three
strings above was rendered**, on every verdict whose `decided_by` is **`conservative-envelope`**
*(v1.16 — the token and the prose are both swept to Rule 4's ruling; `cluster-bootstrap` named one
arm of a two-arm interval and, after the closed form, a resample that no longer runs)* — not on two
of them, because a reader who sees one verdict must still be told which instrument produced it. Two
variants, and the condition is the **design effect**, never the basis:

1. **A widening was applied (`design_effect > 1.0`):**

   > `Decided by the conservative envelope on the paired difference — MOVER-D and the exact paired bootstrap, the wider of the two at each bound — widened by sqrt(DEFF)=1.41 for the declared clustering, in conjunction with McNemar's exact test (p=0.031) as a necessary condition: under clustering McNemar rejects too readily, so it may withhold a verdict but never carries one on its own.`

2. **No widening was applied (`design_effect == 1.0`), which is *every* comparison until a
   determinism probe establishes the basis:**

   > `Decided by the conservative envelope on the paired difference — MOVER-D and the exact paired bootstrap, the wider of the two at each bound — the instrument here because this comparison's design effect is assumed rather than established by construction, with no widening applied (sqrt(DEFF)=1.00), in conjunction with McNemar's exact test (p=0.031) as a necessary condition: a design effect that was never established cannot license the exact test to carry a verdict, so it may withhold one but never carries one on its own.`

   `assumed` is the `basis` field verbatim (`measured` is the other value that reaches this path).

Two things the second variant fixes, both of them the same defect this note keeps ruling on — a
clause that is true on the rare path and false on the common one:

- **The reason clause must attach to the instrument choice, not to the widening.** What displaced
  McNemar is the **basis**; what left `sqrt(DEFF)` at 1.00 is the **design effect**. A single
  *"not widened …, because the design effect is assumed"* reads as the basis explaining the
  widening, which is not the causal chain and is the nearest-attachment parse.
- **Do not say *"under clustering"* where no clustering is declared.** At `design_effect == 1.00`
  the comparison asserts none, so the rationale has to be the one that is actually true there: an
  unestablished design effect cannot license the exact test to carry a verdict. The
  clustering rationale is right in variant 1 and false in variant 2, and variant 2 is the default
  path.

**FR-15's literal marginal-overlap check should still be computed and printed as a diagnostic
line** ("marginal Wilson intervals overlap: yes/no"), because the requirement asks for it and it
costs nothing — but it must be labelled *diagnostic*, never the verdict, with a one-line footnote
naming why (§3.1's inertness). That keeps the requirement visibly honoured while the decision is
made correctly.

### 3.3 Multiple comparisons

Seven tool-calling counts × ~9 turn positions × several metrics is dozens of tests; at α=0.05 each,
a false "better" is close to certain. Cheapest correct handling, and the recommendation:

**Each task pack pre-registers, in its versioned config, a `verdictMetrics` list of one or more
metrics, and a `headlineMetric` that is either exactly one member of that list or `null`.** Only
members of `verdictMetrics` receive a better / not-distinguishable *verdict*; every other number is
printed with its CI and labelled `exploratory — no significance claim`. Pre-registration in pack
config is what stops the verdict-carrying metrics from being chosen after the results exist.

*(v1.2/v1.3 — this **retires** the singular `primaryMetric` rather than redefining it.
Stakeholder decision 2 gives `guard-judge` two co-equal verdict metrics and no headline, so a
one-metric schema cannot express the pack that most needs expressing; and re-pointing an
established name at "may now be `null`" is a trap, since the old meaning is already fixed in the
requirements, the review and two plan versions. The replacement is named `verdictMetrics`
(`architect`, plan v1.3) — a plural one character away from the retired singular would be
indistinguishable in a JSON manifest and in a diff, which is intolerable for the one field whose
entire job is pre-registration. The two fields are separable and both are needed:
**`verdictMetrics` controls inference** — `len(verdictMetrics)` *is* the multiplicity *k*, hence the
correction below — while **`headlineMetric` controls presentation**, i.e. what a reader is entitled
to read as "the" number. A pack with `headlineMetric: null` prints its verdict metrics side by side,
in a fixed declared order, with no summary line above them and no arithmetic combining them.)*

| Role | `verdictMetrics` | `headlineMetric` | Unit |
|---|---|---|---|
| tool-caller | `cleanThroughTurnH` (`H` manifest-declared, validated `≤ min(script length)`; 4 here) | same | conversation |
| **guard-judge** | **`["falseAdvanceRate", "falseSuspendRate"]`** — scored on the 40 `clear_suspend` and 30 `clear_advance` items respectively | **`null`** | item |
| nlq-generator | Layer-1 exact-match rate | same | item |
| chat-responder | deterministic checklist pass rate | same | item |
| embedder | MRR | same | query |

**When `len(verdictMetrics) > 1`, family-wise error control is mandatory, not optional.** Two
co-equal verdicts at α=0.05 each carry a ~9.75% chance of at least one false "better" under the
null — which would hand the stakeholder exactly the fishing artefact pre-registration exists to
prevent. Apply **Holm–Bonferroni across the declared family** (order the p-values; test the
smallest at α/k, the next at α/(k−1), …, stopping at the first non-rejection) and print the
adjusted threshold beside each p-value.

**And it changes what the resolving-power line must print — for the MDD only.** The
guaranteed-detectable threshold for a k-member family is computed at **α/k**, since a metric can be
required to clear that step. For `guard-judge` (k=2) that is α=0.025, and §7.3 carries the
recomputed figures — materially worse than the α=0.05 ones, which is the honest price of two verdict
metrics rather than one. §3.4 Rule 4 is the mechanism: `verdict()` raises unless
`resolving.alpha_mdd == alpha_family / len(family)`, so a k-member family cannot report its MDD at
α=0.05 by oversight.

**The observable floor takes the opposite α, on purpose — do not "fix" the two to match.** The floor
stays at the **unadjusted α=0.05** whatever *k* is. The two bounds carry different sentences, and
each takes the α that makes its own sentence true (§3.4 Rule 3): the MDD promises power *whatever
rank the member draws*, so it needs the tightest step; the floor asserts an impossibility *at any
step the member could face*, so it needs the loosest. Setting the floor to α/k understates nothing —
it **overstates** what is impossible, and prints a threshold that an ordinary Holm rejection walks
straight under. It also double-charges the multiplicity: the price of a second verdict metric is
already paid in the MDD, and paying it again in the floor collapses Holm to Bonferroni for any
difference in `[6/n, 7/n)` — exactly the band §7.3 prices.

*Reversal trigger:* if the stakeholder later ranks the two guard-judge errors, drop the loser out of
`verdictMetrics` into the exploratory block and the family collapses back to k=1 with α=0.05
(floor 15.0 pp instead of 17.5 pp on `clear_suspend`). That is a real gain in resolving power and
it is available whenever the product question "which error costs more?" becomes answerable.

**A family is homogeneous in kind, and `validate` refuses a mixed one** *(v1.15, plan-gate P6-1)*.
Holm orders a family **by p-value**, and a continuous verdict has none — §3.2d rules that the
interval *is* the test. So a `verdictMetrics` list mixing a binary member with a continuous one has
no ordering, hence no ladder, and the correction silently fails to happen for one of them. Two
consequences, both free today: the mixed family is **refused where the records are**, and **an
all-continuous family with `k > 1` takes its correction in the interval rather than in a ladder** — each member's
bootstrap percentiles are taken at `α/(2k)` and `1 − α/(2k)` instead of `1/40` and `39/40`. **The
levels are exact rationals on the unit interval, never percent floats** *(v1.18, §11.2.2)*: at
α = 0.05 the pair is `Fraction(1, 40k)` and `Fraction(40k − 1, 40k)` — `1/80` and `79/80` at k = 2,
`1/120` and `119/120` at k = 3 — and the k = 3 pair is expressible in no decimal unit at all, which
is why the estimator's level parameter is a rational rather than a finer integer one. That
is Bonferroni and it is deliberately not Holm: Holm's gain comes from ordering by p-value, and there
is nothing here to order. The embedder is `k = 1`, so this binds nothing today — and it is written
now for the same reason pre-registration is written now.

**Where that refusal lives — `compare_report`'s first pass, not `validate`** *(v1.16; the plan's
objection is right and the note's v1.15 wording is what moves)*. A manifest carries no records, so
`validate_pack` cannot see which map a metric's values arrive in, and a manifest `kind` field would
be a **second declaration of kind** — precisely what §3.2d refuses when it rules that the map a
metric is in *is* the declaration. A kind is a fact about the run, so it is checkable only where the
run is. **And the refusal is of the whole family's verdicts, not of the offending members:** `k` is
`len(verdictMetrics)` and it is pre-registered, so dropping the minority kind would shrink `k` after
the results exist and hand the survivors a *weaker* correction than the one declared — the fishing
artefact pre-registration exists to prevent, arriving as a repair. Choosing which kind survives is
itself a post-hoc instrument choice. So: name the members and their kinds, print no verdict for any
of them, and let every member's number through as `exploratory — no significance claim` under this
section's standing rule. A mixed family is a pack-authoring defect with a one-line fix, not a data
condition the report should paper over.

### 3.4 The statistics-module contract (`stats.py`) — the B-1 guard

Gate B-1 is not "the plan chose the wrong formula". It is that **the plan's signatures made the
wrong thing the easy thing**: `min_detectable_difference(n: int)` accepts a turn count, a
conversation count and a replicate count identically, and `verdict()` accepts 48 correlated rows as
readily as 12 independent ones. A note that only says "be careful about clustering" reproduces the
defect at the next pack. So the contract below is written so that **the anti-conservative version
does not typecheck, and the honest one is the only one that runs.** Types are illustrative; the
seven rules are binding.

**Rule 1 — the paired table is constructible only from independent analysis units.**

```python
@dataclass(frozen=True)
class PairedOutcomes:
    unit_kind: str                    # pack-declared: "conversation" | "item" | "query"
    unit_ids: tuple[str, ...]         # the cluster keys, one per row
    a_correct: tuple[bool, ...]
    b_correct: tuple[bool, ...]

    @classmethod
    def from_units(cls, unit_kind: str,
                   rows: Iterable[tuple[str, bool, bool]]) -> "PairedOutcomes":
        """Raises DuplicateAnalysisUnit if any unit id appears more than once."""

    @property
    def table(self) -> tuple[int, int, int, int]: ...   # (a, b, c, d)
    @property
    def n_units(self) -> int: ...                       # == len(unit_ids)
```

There is **no other constructor**, and `from_units` raising on a repeated unit id is the mechanism:
the old 48-conversations-from-12-scripts design cannot reach `verdict()` at all, because the script
id repeats four times. A pack that legitimately buys replicates must collapse them first —
`collapse_replicates()` returning one *rate* per script — at which point the data are continuous and
McNemar no longer applies, which is the correct outcome, not an inconvenience.

**Rule 2 — resolving power is computed from effective units, and its inputs have no defaults.**

```python
@dataclass(frozen=True)
class ResolvingPower:
    n_units: int
    unit_kind: str
    alpha_family: float          # unadjusted 0.05 — the FLOOR's alpha, whatever k is (Rule 3)
    alpha_mdd: float             # 0.05/k for a k-member verdictMetrics family — the MDD's (§3.3)
    design_effect: float         # Kish; >= 1.0
    basis: Literal["by-construction", "measured", "assumed"]
    n_effective: float           # n_units / design_effect
    observable_floor: float      # b_min(alpha_family, c=0) / n_effective
    mdd80: float | None          # exact, see rule 3; None when no effect size attains the power

def resolving_power(n_units: int, *, unit_kind: str, design_effect: float, basis: str,
                    alpha_family: float, alpha_mdd: float,
                    power: float = 0.80) -> ResolvingPower: ...
```

`design_effect`, `basis`, `unit_kind` and both αs are **keyword-only with no default value**. A
default of `1.0` would rebuild B-1 by omission — the caller who forgets clustering is exactly the
caller the gate found. `min_detectable_difference` takes **`n_effective: float`** (never `n: int`),
so passing a raw observation count is a visible mislabel at the call site rather than an invisible
one inside the function. `observable_floor` still takes an `alpha` parameter — `b_min` is a function
of α in general, not the constant 6 — but **`resolving_power` must call it with `alpha_family`, the
unadjusted α, never with `alpha_mdd`** (Rule 3's α row). Keeping the parameter while fixing the
call site is deliberate: the formula stays general, and the *choice* lives in one auditable place.

**Rule 3 — the printed-bound principle. Every printed bound takes the rounding direction, the α,
and the denominator that keep *its own claim* true.**

This is one rule, not a list of special cases, and it is the rule three separate adjudications in
this document's review history each rediscovered the hard way. A printed bound is a *sentence*, and
the sentence is what has to be true — so every free parameter in computing it is fixed by asking
"which value makes this sentence true?", never by consistency with the bound printed next to it.
Two bounds sitting side by side routinely take **opposite** values of the same parameter, and that
is correct rather than sloppy. The three instances, all live in this note:

| parameter | MDD — *"resolves differences ≥ X with 80% power"* | observable floor — *"differences below Y cannot reach significance at any observed outcome"* |
|---|---|---|
| **rounding** | **up** — at the printed value the power claim must hold (19.0 pp has power 0.798) | **down** — a printed 15.8 at n=38 is false, since the attainable `6/38 = 15.789` reaches significance |
| **α** | **tightest** step, `α/k` — the claim must hold whatever rank the member takes under Holm | **loosest** step, unadjusted `α` — a member *can* be tested at 0.05, so `7/n` asserts a false impossibility |
| **denominator** | **floored** `n_effective` — flooring is what keeps X conservative | **unfloored** `n_effective` — flooring here would *raise* Y and overstate what is impossible |

Each cell is the conservative choice **for that sentence**; unifying any row would make one of the
two bounds anti-conservative, which is why the denominator nit (`n-ML-1`) was correctly declined
rather than harmonised. When a new printed bound is added, derive its three parameters from its
sentence before writing the formula.

**Rule 3a — MDD and floor are computed exactly and rounded in *opposite* directions; the `8/n` rule
of thumb is not code.**
`min_detectable_difference` bisects on δ for the smallest difference at which
`P(reject) ≥ power` under the exact McNemar rejection region, then **rounds up to the printed
precision**. Both halves matter, and this settles gate m3's "8/n gives 20.0 pp where the note says
19.1 pp":
- exact δ at n=40 is **19.046 pp**, and `8/n` = 20.0 pp — the rule of thumb is *conservative but
  wrong*, and printing two different numbers for the same quantity in the same report is the defect;
- rounding **to nearest** would print 19.0 pp, at which measured power is **0.798** — below the
  0.80 the sentence claims. Rounding up to 19.1 pp gives 0.8023.

**And the observable floor rounds the other way — *down*.** One principle, two directions: **round
each printed bound in the direction that keeps its own claim true.**
- MDD carries *"resolves differences ≥ X with 80% power"*, so X must round **up**: at the printed
  value the power claim must hold.
- The floor carries *"differences below Y cannot reach significance at any observed outcome"*, so Y
  must **truncate**: at n=38 the exact floor is `6/38 = 15.789 pp`, and printing the ceiling 15.8
  makes the sentence **false**, because the attainable observed difference `6/38 = 15.789` is below
  15.8 and *does* reach significance (b=6, c=0, p=1/32). Truncating to 15.7 keeps it true.
- The failure mode is not academic: at n=12, α=0.025 the exact floor is `7/12 = 58.333 pp`, which a
  report displays as an observed `58.3`. Ceiling the floor to 58.4 produces a report whose verdict
  line says *distinguishable* while its own honesty line says the difference cannot reach
  significance. Truncation to 58.3 removes the contradiction.
- Exact values (`6/40 = 15.0`) print exactly under either rule; only inexact ones diverge.
- **Truncate with the same expression the caller uses.** `math.floor(x/precision)` and
  `math.floor(x*1000)` are not interchangeable: the double nearest `0.001` is slightly *above* it,
  so `(7/40)/0.001 = 174.99999999999997` truncates across the bin edge to **17.4 pp** where
  `(7/40)*1000 = 175.0` gives the correct 17.5. Swept in this session over `n ≤ 2000`: with
  `b_min = 7` this bites at **n = 5, 10, 20, 40**; with `b_min = 6` it **never** bites.
  **So under v1.6's floor — always `6/n_eff` — the guard is defensive, not load-bearing on any cell
  this note prints.** It was load-bearing under the `α/k` floor that v1.6 removes (17.5 pp at n=40
  was a published cell), and the two changes arrived in the same review pass, which is precisely how
  a justification outlives the thing that justified it. Keep the guard anyway — `b_min` is a
  function of α, not the constant 6, so any future α reopens the hazard, and the cost is one
  expression — and pin it with the **code's own expression**, never an equivalent-looking one.

*(v1.5 corrects three §7.1 cells where v1.2–v1.4 ceilinged an inexact floor — 15.8→15.7 at n=38 and
7.1→7.0 at n=85 in the α=0.05 column, 46.7→46.6 at n=15 in the α=0.025 column. The note was
internally inconsistent, ceiling-rounding the floor in one column and truncating it in the other;
this rule is what makes the direction derivable rather than remembered.)*

**Rule 4 — `verdict()` asserts its own preconditions and refuses when McNemar is invalid.**

```python
def verdict(outcomes: PairedOutcomes, *, resolving: ResolvingPower,
            metric_name: str, family: Sequence[str]) -> Verdict: ...
```
Raises — never warns, never silently proceeds — unless all five hold:
1. `resolving.n_units == outcomes.n_units` (the printed resolving power belongs to *this* table);
2. `resolving.unit_kind == outcomes.unit_kind`;
3. `resolving.alpha_mdd == alpha_family / len(family)` and `metric_name in family` (§3.3's
   multiplicity). **This precondition is unchanged by v1.6 and must not be "fixed" to match the
   floor**: `alpha_mdd` is the *pre-registration* α — the tightest step, which is what makes the MDD
   claim rank-independent;
4. `resolving.design_effect >= 1.0`;
5. `alpha_step`, when supplied, lies in `[alpha_mdd, alpha_family]`. Holm's own steps are
   `α/(k−i)`, which lie in that range by construction, so a step outside it is a caller defect —
   and left unchecked it surfaces one module later as Rule 7's `mcnemar-exact` raise, which
   announces a bug in the wrong place. Checking it here is what makes Rule 7's theorem a *checked*
   premise rather than an assumed one *(v1.7: shipped with the m-ML-6 fix round and adopted here)*.

**Three αs coexist deliberately; `ResolvingPower` carries two of them.** `alpha_family`
(unadjusted, 0.05 — the **floor's**, Rule 3) and `alpha_mdd` (`α/k` — the **MDD's**) are both known
when the object is built and rendered, so both are fields. The third, `alpha_step` (`α/(k−i)`,
Holm's actual data-dependent threshold for this member at decision time), is **not** a field: it is
known only after the family's p-values are ranked, which is *after* this object is built and
printed, so a field would be `None` until it was not and the number would have two homes. It
reaches the one function that uses it as `verdict()`'s parameter. *(v1.7 corrects v1.6, which said
the dataclass carries all three in the same sentence that said the third is unknowable at
construction time. The two-field shape is the one the note requires.)*

And the decision rule branches on the design effect, which is the fix B-1 asks for:
- **`design_effect == 1.0` and `basis == "by-construction"`** → McNemar exact decides, MOVER-D
  quantifies (§3.2b/c). This is the only configuration in which McNemar is valid.
- **otherwise** → McNemar is anti-conservative and must not **decide alone**. The decision is *"the
  conservative envelope on the paired difference excludes zero"* (Rule 4); McNemar's p may still be
  printed, labelled `anti-conservative under clustering — not the decision`.

**McNemar as a *veto* is permitted on that path, and is the recommended form.** Making the
non-`by-construction` decision a **conjunction** — distinguishable iff the widened CI excludes zero
**and** `mcnemar_exact(b, c) ≤ alpha_step` — does not violate this rule, and the reason is worth
stating because the rule reads as if it forbids what the fix requires. **The objection to McNemar
under clustering is that it *rejects* too readily.** A necessary condition can only ever *remove*
rejections, so the conjunction is uniformly at least as conservative as either instrument alone: at
DEFF = 1 it restores the exact test's calibration exactly, and at DEFF > 1 the widened interval
remains the binding constraint it already was. What the rule forbids is McNemar deciding *for*
distinguishability under clustering; using it to withhold that verdict is the opposite operation.

**The interval printed on that path is the conservative envelope, not the resample alone (v1.8,
review M-ML-8). This corrects v1.6–v1.7, whose *"the strings rendered against the bootstrap"*
authorised a substitution it never checked the width of.** The veto fixed the **decision** on this
path and left the **quantification** on the bare percentile interval, and that interval is the
narrower of the two available — measured this session, exactly (every paired table enumerated, and
every bootstrap resample outcome computed analytically rather than sampled):

| n, regime | MOVER-D coverage / mean width | percentile bootstrap at DEFF 1.00 | narrower than MOVER-D on |
|---|---|---|---|
| 30, strict dominance δ=15 pp | 0.983 / 30.60 pp | **0.942** / 24.57 pp | **100%** of the probability mass |
| 40, strict dominance δ=15 pp | 0.976 / 25.63 pp | **0.939** / 21.23 pp | **100%** |
| 85, strict dominance δ=12 pp | 0.969 / 15.32 pp | **0.940** / 13.56 pp | **100%** |
| 40, null | 0.962 / 29.55 pp | 0.956 / 26.80 pp | 91% |

So on the path taken **because the design effect was never established** — i.e. every comparison
until a determinism probe runs — the report prints a *tighter* effect size than the instrument
§3.2c mandates, in the strict-dominance regime this tool exists to catch. Worse, the percentile
bootstrap **degenerates at the sparse discordant counts §3.2b says this lab will actually see**:
at n=30 with `b=4, c=0` it returns `[3.3, 26.7] pp`, excluding zero, where McNemar's exact p is
**0.125** and MOVER-D is `[−0.6, 29.7]`. The mechanism is not subtle — with four non-zero rows,
`P(no +1 drawn) = (26/30)³⁰ = 1.4% < 2.5%`, so the 2.5th percentile *cannot* be zero. At n=30
that fires on **20.3%** of the probability mass under strict dominance, each time rendering
verdict 3's *"the interval excludes zero but the exact paired test does not"* on the strength of a
degenerate interval rather than a genuine disagreement.

**The rule.** On any non-`by-construction` path the printed interval is the **envelope**, taken
**bound by bound**, of

- the `√DEFF`-widened percentile bootstrap (Rule 6), and
- the `√DEFF`-widened MOVER-D interval, half-widths scaled about the point estimate the same way:

> **the printed lower bound is the more negative of the two lower bounds and the printed upper
> bound is the more positive of the two upper bounds, each chosen independently of the other. It is
> *not* "whichever of the two intervals is wider".**

*(v1.11 publishes that sentence because v1.8's "the wider of" does not state it. Read as a choice
between two whole intervals — the more natural reading of the words — Rule 4's own claim below is
**false**, and this note's own worked table is the separating case. At `(a=4, b=5, c=3, d=0)`,
n=12, DEFF 1.00: the resample interval `[−25.0, 58.3]` is the wider of the two whole, 83.3 pp
against MOVER-D's 80.8 pp, while MOVER-D's lower bound **−27.1 pp** is the more conservative one —
so a whole-interval choice takes the resample and prints a bound tighter than an instrument the
rule claims to dominate. Bound by bound each bound is conservative separately, which is exactly
what makes "uniformly at least as conservative as either alone" true. Confirmed against the shipped
`stats.conservative_envelope` at `5878014`, whose implementation already takes the bound-by-bound
reading — this paragraph brings the note into line with it rather than the reverse, and the check
that fixes the reading is the `(4, 5, 3, 0)` table, not a preference.)*

It is uniformly at least as conservative as either instrument alone, keeps the interval responsive
to a declared design effect where one exists, and removes the sparse-count degeneracy because
MOVER-D covers zero exactly when the counts are too thin to support excluding it. It can only
*remove* rejections, so it disturbs neither the veto nor Rule 7.

**Withdrawn as false: "reduces to MOVER-D exactly at `DEFF = 1.00`"** *(v1.11, measured — every
paired table enumerated at n = 12, 30 and 40, with the exact bootstrap quantiles computed
analytically rather than sampled).* At DEFF 1.00 the **resample** arm binds at least one bound on
**83.5%** of the 455 tables at n=12, **87.3%** of the 5 456 at n=30 and **87.4%** of the 12 341 at
n=40. The envelope is a genuine mixture at every design effect, not a MOVER-D restoration with a
dormant second arm. What is true is the weaker statement the construction actually delivers —
**neither printed bound is ever tighter than MOVER-D's**, so §3.2c's instrument is a *floor on
conservatism* rather than the interval itself. The correction matters beyond accuracy: the withdrawn
sentence is what made the resample arm look inert on the default path, and that arm is the one
carrying the seed dependence below.

**And the percentile itself must be computed in closed form, not resampled.** *(v1.11 promotes this
from v1.8's "should". v1.8 argued it on cost and purity, which is a preference; the three findings
in the v1.11 block below make it binding, and the third of them was already spent by §11.2 before
this paragraph was implemented.)* For a paired
**binary** table the per-unit differences take three values, so the resample distribution is
exactly multinomial and its quantiles are a ~30-line exact computation (verified this session
against the shipped `B=10 000` resample). The resampled version is not merely slower — **it is a
Monte-Carlo estimate of an *atomic* quantile, so whenever the target percentile lands within
Monte-Carlo error of an atom boundary the printed bound flips by a whole atom, `1/n_units`.** State
it precisely, because the loose version ("the bound moves with row order") is not reproducible on an
arbitrary table and a rationale nobody can reproduce is a defect in waiting:

- **The predictor is the exact CDF's distance from the target percentile.** At `(b=8, c=0, n=85)`
  the exact bootstrap CDF is `0.97281` at `13/85` and `0.98738` at `14/85`, so the 97.5% quantile
  is `14/85` by only **0.0022** of probability, against a Monte-Carlo standard error of
  `sqrt(0.9728·0.0272/10 000) = 0.0016` — about a 9% chance per draw that the empirical CDF crosses
  first and the bound prints `15.3` instead of `16.5`. Measured: **16 of 200 seeds** and **23 of 200
  row permutations** at a fixed seed. Where the quantile is *not* near a boundary the bound is
  perfectly stable — `(b=4, c=0, n=30)` is identical across 60 seeds.
- **Row order matters for the same reason the seed does, and the reason is not obvious.**
  `random.Random.choice` draws an **index**; a fixed seed fixes the index sequence, and a
  permutation of the same multiset re-maps those indices onto different values. So "same multiset,
  therefore same resample" is false.
- **At the tool-caller pack's own n it is a coin flip, not a tail event.** `(b=5, c=3, n=12)`:
  `paired_cluster_bootstrap(..., B=10 000)` prints a lower bound of `-25.0 pp` at seed 0 and
  `-33.3 pp` at seed 5 — **8.3 pp apart**, at **107/93 of 200 seeds** and **102/97 of 200 row
  permutations**. That is the last *two* digits of a bound printed to 0.1 pp, decided by the seed
  and by the order rows happen to sit in.

A bound at that mercy is not reproducible in the sense §3.2d's seed requirement is asking for, and
the closed form removes both dependencies at no cost.

---

**v1.11 — why the closed form is binding, and the residual it closes is bigger than the note said.**
All three findings measured this session against the shipped `stats.conservative_envelope` at
`5878014`. The seed distribution of the shipped bound was computed **exactly**, not simulated:
`_percentile` selects the order statistic `ordered[int(round(pct/100·(B−1)))]`, so for `B = 10 000`
the bound is the 251st (lower) and 9 750th (upper) order statistic of B draws from a known atomic
distribution, and `P(X₍ₖ₎ ≤ x) = P(Binom(B, F(x)) ≥ k)` gives its full distribution over seeds in
closed form. Cross-checked against 400 shipped seeds at `(4, 5, 3, 0)`: predicted 55.7%/44.3%,
observed 228/172.

**(1) The envelope did not absorb the instability; it narrowed it, and not by much.** The claim that
MOVER-D binds "the overwhelming majority of tables" is the mirror image of the measurement — the
resample arm binds a bound on 83–87% of tables (above). What limits the damage is not the envelope
but the fact that the exact bootstrap CDF is usually far from the target level. Tables whose
**printed** bound moves with the seed at DEFF 1.00, by the probability of the minority rendering:

| n | tables | ≥ 1% | ≥ 5% | ≥ 20% | ≥ 40% |
|---|---|---|---|---|---|
| 12 | 455 | 69 (15.2%) | 61 (13.4%) | 47 (10.3%) | 27 (5.9%) |
| 30 | 5 456 | 2 586 (47.4%) | 1 815 (33.3%) | 1 046 (19.2%) | 265 (4.9%) |
| 40 | 12 341 | 6 815 (55.2%) | 5 168 (41.9%) | 2 915 (23.6%) | 956 (7.7%) |

So at the two item-level pack sizes a **majority** of reachable tables print a bound decided by the
seed. `(4, 5, 3, 0)` — where the rendered lower bound is `−27.1 pp` on 55.7% of seeds and
`−33.3 pp` on 44.3% — is a representative member of that set, not an exotic corner.

**(2) The seed reaches the *verdict*, not only the printed digits, and neither the veto nor Rule 7
catches it.** This is the finding that makes the closed form mandatory on its own, and it is a
different defect from M-ML-8. The envelope's zero-exclusion is a conjunct of `raw_significant`, so a
seed-dependent bound is a seed-dependent **verdict** wherever the McNemar veto and the observable
floor both pass. Two cases, both re-run through the shipped code:

| table `(a,b,c,d)` | n | DEFF | \|diff\| | McNemar p | floor | verdict over 150 **seeds** | verdict over 150 **row permutations** at seed 20260902 |
|---|---|---|---|---|---|---|---|
| `(1, 25, 12, 2)` | 40 | 1.2 | 32.5 pp | 0.047 | 18.0 pp | **80 distinguishable / 70 not** | **85 / 65** |
| `(2, 19, 7, 12)` | 40 | 1.5 | 30.0 pp | 0.029 | 22.5 pp | **101 / 49** | **97 / 53** |

On the first, the report alternates between
*"A is better than B on `<metric>`: +32.5 pp (95% CI [0.0, 62.6] pp) …"* and
*"Not distinguishable at this sample size. Observed difference +32.5 pp, 95% CI [−0.4, 62.6] pp
covers zero …"*. **The row-permutation column is the one that settles it:** the pack's declared
`sampling.seed` is fixed, and the verdict still moves — so `sampling.seed` does not make this
verdict reproducible, and P3-5's contract does not deliver what it claims on the path it governs.
Row order is not a pack author's choice; it is result-file iteration order.

**Reachability, stated exactly** (every table passing both the veto at the loosest Holm step
α = 0.05 and Rule 7's floor, counting those whose verdict flips with probability ≥ 1%):

| n | DEFF 1.00 | 1.2 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| 12 | 0 | 0 | 0 | 0 | 0 |
| 30 | 0 | 0 | 20 (worst 3.7%) | 0 | 0 |
| 40 | 0 | 26 (worst **43.1%**) | 62 (worst 31.7%) | 0 | 0 |

Read this honestly in both directions. **At DEFF 1.00 — today's default on every comparison until
the determinism probe lands — no verdict flips** (worst minority probability over n ∈ {12, 30, 40}
is 3.9 × 10⁻¹⁰), and at n=12 **no** reachable table flips at any design effect in the row above —
78 tables clear the veto and the floor at DEFF 1.00 and none of them is seed-dependent, because the
atoms there are 8.3 pp apart and the 50.0 pp floor removes the region where a bound could sit beside
zero. The tool-caller pack's exposure is the printed digits only. The defect is **latent, not live**
at the design effects in force today. It
becomes live the moment any design effect above 1.0 is declared, and nothing gates that:
`results.py` reads `designEffect` from a run record (defaulting to 1.0) and `report.py` propagates
`max(a.designEffect, b.designEffect)` straight into `resolving_power`. A determinism probe that
measures DEFF 1.2 on the guard-judge pack's 40-item slice lands directly in the worst row of that
table. Under the standing principle that defects are not carried into later stages, a latent verdict
flip that a *scheduled* deliverable activates is the case the principle exists for.

**(3) §11.2 has already spent the closed form, in the present tense.** §11.2's second reason for
Hyndman–Fan type 1 — one of four reasons that carried R-13's gated estimator ruling, and the one
that rejects linear interpolation on the ground that "no third estimator enters the document" —
reads: *"It is the same functional §3.4 Rule 4 already computes in closed form. Rule 4 (v1.8)
replaces the resampled bootstrap quantile with the exact quantile of the multinomial resample
distribution … So adopting it makes v1.8's closed form a **substitution of computation, not of
definition**."* With Rule 4 left resampled, that sentence is false of the built system and §11.2's
reason 2 has no referent. This is not an argument the closed form is *nice*; it is that a second,
separately accepted ruling was already written on its existence.

**Was `(4, 5, 3, 0)` sufficient on its own? No — and the honest answer matters.** Taken alone it is
a printed-digit defect on a table whose verdict is *not distinguishable* on every seed, and a
reviewer could reasonably grade it minor and defer it. Finding (2) is what makes the closed form
mandatory, and finding (3) is an independent second reason. An implementer who stops at the
`(4, 5, 3, 0)` residual has under-measured the problem, not over-measured it.

**The closed form, pinned — an implementer must not have to choose any of this.**

For a paired binary table `(a, b, c, d)` with `n = a+b+c+d`, the per-unit differences take exactly
three values, so a resample of `n` rows with replacement gives
`(N₊, N₀, N₋) ~ Multinomial(n, (b/n, (a+d)/n, c/n))` and the resample mean is `S/n` with
`S = N₊ − N₋` on the integer support `[−n, n]`.

- **The estimator is §11.2's, and there is one of it.** `Q(p) = inf{ s/n : F(s) ≥ p }` — Hyndman–Fan
  type 1, the inverse CDF, applied to the **exact** distribution where §11.2 applies it to an
  empirical sample. This is precisely what §11.2 reason 2 already claims, so the two agree by
  construction.
- **The levels are `LEVEL_CI95_LO = 1/40` and `LEVEL_CI95_HI = 39/40`, exactly — never the level
  the resample happened to estimate** *(spelled as exact rationals at v1.18, §11.2.2; they were
  `permille` 25 and 975)*. The shipped `ordered[int(round(pct/100·(B−1)))]` picks the 251st and
  9 750th of 10 000, which estimate levels `251/10 001` and `9 750/10 001`, not `0.025`/`0.975`.
  Measured: the atom selected at those two level pairs differs on **2** `(b, c)` pairs at n=30,
  **12** at n=40 and **113** at n=85 — one whole atom, `1/n`, on a printed bound each time. `B` is
  an artefact of the resample and must not survive into a formula that has no `B`. Both are already
  among the four literal levels §11.2.1 sweeps.
- **Integer arithmetic, so there is no tolerance to choose and no bin edge to guard.** Every atom's
  probability is a rational with denominator `n**n`:
  `P(S = s) = Σ_{n₊−n₋ = s} multinomial(n; n₊, n₀, n₋)·b^n₊·(a+d)^n₀·c^n₋ / n**n`.
  Accumulate the integer numerators in ascending `s` and select the first atom satisfying
  `level.denominator · cum ≥ level.numerator · n**n` *(v1.18: one representation for both
  estimators; the expression was `1000 · cum ≥ permille · n**n`)*. That is an exact integer
  comparison: **the atom is never chosen by
  a float tie-break**, which is Rule 3a's and §11.2.1's hazard *removed* rather than guarded — the
  third time this document meets it and the first time it can be deleted instead of pinned. Verified
  against a float-CDF implementation on all 455 + 5 456 + 12 341 tables at n = 12/30/40: zero
  disagreements.
- **Cost, measured:** 3.2 ms at n=85 against 216 ms for the shipped `B = 10 000` resample — **68×
  faster**, in pure stdlib, with `math.comb`. The enumeration is `O(n²)` terms.
- **Degenerate input:** `b = c = 0` gives the single atom `0`, and both quantiles are `0`. Correct,
  and it needs no special case.
- **The boundary of applicability is three values, not "binary versus continuous".** The cost of the
  exact enumeration is `C(n + v − 1, v − 1)` for `v` distinct per-unit difference values: `v = 3`
  at n=85 is 3 741 terms and trivial; `v = 10` at n=38 is 1.6 × 10⁹ and infeasible. §3.2d's
  continuous metrics are excluded by that arithmetic, and so is anything else with more than three —
  **apply the closed form when and only when `v = 3`, which in this note is the paired binary table
  and nothing else.** MRR is the trap worth naming: its per-item values are discrete (`0`, `1/r`),
  so "it is not continuous" invites the wrong generalisation; its `v` is far above 3.

**What retires, what stays, and what replaces P3-5's contract.**

*Retires.* Nothing on the paired-binary path resamples any more, so:

- `stats.verdict(..., bootstrap_seed=...)` — the parameter goes, and with it the raise that demands
  one on the clustered path.
- `stats.conservative_envelope(diffs, table, *, design_effect, B, seed)` collapses to
  `conservative_envelope(table, *, design_effect)`. The `diffs` argument goes too: the exact
  bootstrap distribution is a function of `(b, c, n)` alone and MOVER-D of `(a, b, c, d)`, so there
  is no second argument left to disagree with the first — which **retires the
  `n != len(diffs)` guard by making the error it catches unrepresentable**, the outcome this note
  prefers to a guard every time.
- `report.py`'s `- decided by: cluster-bootstrap (seed N, from the pack's `sampling.seed`)`
  parenthetical, and the four tests that pin it (P3-5's three in `test_report.py` plus the
  `bootstrap_seed=None` precondition-ordering test in `test_stats.py`).

*Stays — `PackRef.seed` and `sampling.seed` are **not** removed, and no manifest field changes.*
P3-5's finding was that *the pack's own declaration could not reach the decision it governs*; that
finding is **satisfied, not reversed**, and only its object moves:

- **§3.2d's continuous-metric bootstrap is untouched and stays seeded.** MRR, score separation and
  latency have no closed form at any tolerable cost (the `v` arithmetic above), and the embedder
  pack's `verdictMetrics = ["mrr"]` is decided by exactly that interval. `PackRef.seed` keeps its
  no-default rule and `pack_ref_from_manifest`'s `sampling.seed` refusal stays; only the docstring's
  justification moves from the paired table to §3.2d.
- **Rule 6's `cluster_bootstrap` and `paired_bootstrap` stay seeded** — Rule 5's `design_effect`
  measurement needs a bootstrap width, and the two-level resample arrives with
  `replicatesPerScript > 1`.
- **`paired_cluster_bootstrap` stays too, and this note owed it a name** *(v1.14, plan-gate P5-3)*.
  The closed form retires the **binary** paired interval and only that one, so the chain
  `conservative_envelope → paired_cluster_bootstrap → paired_bootstrap` loses its *current* caller,
  not its *designed* consumer. **`paired_cluster_bootstrap` is §3.2d's entry point for every
  continuous verdict**, called with the pack's declared `design_effect` — the identity widening at
  1.00 — and **`paired_bootstrap` is its engine, not a second entry point.** Naming the entry point
  is the load-bearing half: an implementer who wires MRR straight to `paired_bootstrap` gets a
  correct interval that **silently ignores a declared design effect**, which is the one failure
  direction this note refuses everywhere else. It is unreferenced today because §3.2d's continuous
  path is unbuilt — the same state `sampling.seed` is in above, and it takes the **same
  discriminator**: both go if the embedder pack's continuous verdict is ever cut. Its four direct
  tests are the only executable statement of Rule 6's `√DEFF` exactness argument and stay with it.
- **One condition on that reuse, stated here because leaving it unstated is a defect.** `_widen`'s
  clamp is `[-1, 1]` — correct for the difference of proportions the envelope was written for, and
  **wrong for `sep_z`**, whose per-query differences are differences of z-scores and are not bounded
  by 1. Wired as it stands, a `sep_z` interval whose true upper bound exceeds 1 is silently clamped
  to it and the point estimate can land outside its own interval; the *verdict* survives (a positive
  difference's exclusion of zero is decided by the lower bound) but the printed interval is false.
  So: MRR and any rate difference go through unchanged, and **`sep_z` needs that clamp parameterised
  or absent** before §5.2's comparison is wired. Blocked on nothing — it is one argument on
  `_widen`. *(v1.17 ships that argument as `support` on Rule 8's producer; v1.19's Rule 4a rules
  **where** it is applied — once, on the printed interval, never on an envelope arm — and the
  binary path's own support stays a constant rather than becoming a second parameter.)*
- **P3-5's rule is unchanged: the seed is named only where a resample actually decided.** The same
  predicate, evaluated against a system with one fewer resample in it, now selects the continuous
  verdicts and not the binary ones.
- **The honest cost, named rather than hidden:** until §3.2d's continuous path is built, `sampling.seed`
  is a required manifest field with no live consumer, which is the shape this codebase's P2-4
  principle distrusts. It stays anyway, for a reason P2-4 does not cover: a seed chosen **after**
  results exist is not a pre-registration, and a content-hashed manifest is where pre-registration
  lives (plan §3.3(iii)). Removing and re-adding a required field churns every pack fixture twice
  to buy one release of tidiness. **Discriminator, so this is checkable rather than asserted:** the
  field stays only while the embedder pack's MRR verdict is a committed deliverable. If that is ever
  cut, `sampling.seed` goes with it.
- **Unrelated, flagged so it is not mistaken for collateral of this ruling:** §3.2d says *"the seed
  goes into the environment fingerprint (FR-7)"* and it does not — `fingerprint.py` at `5878014`
  contains no seed field (`git grep seed` finds it only in `stats.py`, `report.py`, `packs.py`).
  That gap predates this change and is unaffected by it either way.
- **For `architect`:** no field is added or removed, and `sampling.seed`'s semantics — which the
  plan's §3.3 currently states only by appearing in a JSON example — should be written down there as
  *the continuous-metric bootstrap seed*. The two signatures that change are `stats.verdict` and
  `stats.conservative_envelope`, both above.

**What the tool prints, which is where this project's defects have lived for five passes.**

- **The CI digits move.** On every non-`by-construction` verdict where the resample arm binds and
  the Monte-Carlo estimate missed the atom, the printed bound changes. Against the current *modal*
  rendering the change is small — the closed form differs from the MC mode on 0 of 455 tables at
  n=12, 46 of 5 456 at n=30 and 68 of 12 341 at n=40 — but against *a given seed's* rendering it is
  the instability table above. Any fixture asserting a rendered interval on this path must be
  re-derived, not adjusted.
- **§3.2e verdict 1's published string does not move.** At `(a=0, b=6, c=0, d=34)`, n=40, DEFF 1.00
  the closed-form envelope renders `[3.2, 29.1] pp` — MOVER-D's own interval, and verdict 1's
  published string exactly. Checked because it is the one string a reader would notice.
- **The knife-edge verdicts resolve deterministically, and not all to the same side.** The closed
  form is exact, not conservative: at `(1, 25, 12, 2)` DEFF 1.2 it renders `[0.0, 62.6]` and
  **distinguishable**; at `(6, 16, 35, 28)` n=85 DEFF 1.9 it renders `[−43.4, 0.4]` and **not
  distinguishable**. An implementer expecting the closed form to be uniformly one way or the other
  will mis-write the tests.
- **Four printed strings call the interval "cluster-bootstrap" and must stop** *(v1.11, and this is
  a defect in the shipped v1.8 envelope, independent of the closed form)*. The printed interval is
  an envelope of two instruments; naming one arm is exactly M-ML-8's error one layer over, and it is
  wrong today on the 12.6–16.5% of tables where MOVER-D binds both bounds. The strings are
  Rule 7's floor-demotion sentence (*"The cluster-bootstrap interval …"*), §3.2e verdict 3's `named`
  substitution, both variants of the trailing *"Decided by the cluster-bootstrap CI on the paired
  difference …"* clause, and the `- decided by:` bullet. Each must name **the conservative envelope**
  and, after the closed form, say what it is an envelope *of*. The `- decided by:` bullet loses the
  seed parenthetical and has room for the audit that replaces it — **name which arm bound each
  bound**, which is cheap, deterministic once the resample is gone, and makes Rule 4's mixture
  legible to a reader instead of inferable only from the code:

  > `- decided by: conservative envelope (lower bound: MOVER-D; upper bound: exact paired bootstrap, p=0.975)`

  **A third token joins those two at v1.19** — `support bound`, for a bound the support clamp moved,
  which neither arm produced. See **Rule 4a**, which also settles where the clamp is applied and why
  that choice changes no number.

  Whether the machine token `DecidedBy = "cluster-bootstrap"` is renamed with the prose was
  `architect`'s call (27 occurrences across `stats.py`, `report.py` and two test modules), and it is
  **settled: renamed to `conservative-envelope`** (plan v1.11, §4 S1e Table D; §3.2f and Rule 7 above
  are written against the new token). The recommendation stood because a token naming a resample that no longer runs is precisely the
  failure Rule 7's own docstring warns about — *"such a substitution changes the instrument's name
  and not its interval, and an interval alone cannot report that"* — with the sign reversed. What is
  **not** optional either way is the prose: a sentence naming an instrument that did not produce the
  number beside it is the defect class this document exists to remove.

**How to prove it — the acceptance the implementer inherits.**

1. **Determinism, by construction and by test.** `conservative_envelope` takes no seed and no `B`,
   so the property is structural; assert it anyway on the two tables above, over 20 row permutations
   each — the test that would have failed before this change, and the only one that reproduces the
   defect rather than the fix.
2. **Agreement with the resample where the resample is stable.** For every table at n=12 whose exact
   bootstrap CDF sits more than 5 Monte-Carlo standard errors from both target levels, the closed
   form must equal `paired_bootstrap(..., B=10 000, seed=s)` at 0.1 pp for `s ∈ {0..9}`. This is the
   substitution check: same definition, better arithmetic. **Run this session and it passes** — 21
   `(b, c)` pairs qualify at n=12, over all their `a`/`d` splits and all ten seeds, zero
   disagreements. Twenty-one pairs is thin coverage; widen to n=30 if the runtime is affordable.
3. **The two published anchors, verbatim:** `(0, 6, 0, 34)` at DEFF 1.00 renders `[3.2, 29.1] pp`,
   and `(4, 5, 3, 0)` at DEFF 1.00 renders `[−27.1, 58.3] pp` — the second being the case where
   MOVER-D's lower bound is the more conservative one, so it also pins the bound-by-bound reading.
4. **Rule 4's conservatism property, asserted rather than asserted-about.** Over every table at
   n ≤ 40 and `DEFF ∈ {1.0, 1.2, 2.0}`: `envelope_lo ≤ mover_lo` and `envelope_hi ≥ mover_hi`, and
   the same against the exact bootstrap arm. That is the property whose whole-interval reading is
   false, so the test is what stops a future implementer taking it. **Run this session and it
   passes** — 54 756 `(table, DEFF)` combinations at n ∈ {12, 30, 40}, zero violations.
5. **Render the page and read the English.** Every one of the five verdict strings on this path,
   printed and read, for the naming change above. Every defect in this document's five review passes
   was found that way and none by reading assertions.

**Rule 4a — the support bound belongs to the *printed interval*, applied once; and a bound the
support moved is not attributable to an arm.** *(v1.19, at plan-gate Pass 8's `P8-1`, routed here
because the reviewer correctly declined to adjudicate it. The ruling holds for **both** carriers —
the paired-binary envelope and §3.2d's continuous interval — and it is stated once, as a property of
the estimand, so neither carrier owns a special case.)*

**The rule, in four lines an implementer can execute without re-deriving any of what follows.**

1. **A support is the parameter space of the estimand.** It is a fact about the *metric*, never
   about an *instrument*, so it is applied **once, to the interval that is printed**, and never to
   an input of a composition. It is never a "plausible range" or a display bound — `clamp=(0.9,
   1.5)` is legal as a test pin and illegal as a production value (see assertion 9).
2. **On the paired-binary path:** `envelope_arms` widens both arms with **`clamp=None`** and returns
   them unclamped; the composition — `min` of the two lower bounds, `max` of the two upper bounds —
   clamps its **result** to `(-1.0, 1.0)`. The support here is **not** a new parameter: a difference
   of two proportions read off one paired table lies in `[-1, 1]` by construction, which is v1.17's
   *second* category (*derivable from what the function already holds*), so it is a constant in one
   place, exactly as the percentile levels are. Only `sep_z`'s support is v1.17's *third* category
   and stays declared.
3. **On the continuous path nothing changes**, and that is the point of stating the rule at this
   altitude: `paired_cluster_bootstrap` already clamps the one interval it returns, and
   `continuous_verdict` already derives that clamp from the metric's `support` (Rule 8). There is no
   composition there, so "once, on the printed interval" is what it already does. `support=None`
   → `clamp=None` → no clamping, unchanged.
4. **`bound_by` becomes a three-token closed set** — `Literal["MOVER-D", "exact paired bootstrap",
   "support bound"]` — and is computed from the composed **unclamped** value `u` against the support
   `(L, U)`, with a **strict** comparison:
   - lower: `u_lo < L` → `"support bound"`; otherwise `"MOVER-D"` if `mover_lo <= exact_lo` else
     `"exact paired bootstrap"`;
   - upper: `u_hi > U` → `"support bound"`; otherwise `"MOVER-D"` if `mover_hi >= exact_hi` else
     `"exact paired bootstrap"`.

   Strictness is load-bearing: a bound sitting *at* the support because both instruments genuinely
   produced it is an **arm's** bound. `(0, 0, 12, 0)` at DEFF 1.00 is the witness — both arms return
   exactly `-1.0` with no widening applied at all.

**Compose-and-clamp gets exactly one home, which closes `P8-5` as collateral.** The composition rule
is written twice today — `conservative_envelope` and the inline `min`/`max` in `verdict()`. One
private composer takes the two unclamped arms and returns `(interval, bound_by)`;
`conservative_envelope` returns its first element, `verdict()` takes both. No public signature
changes and no new required parameter.

**Why the location is immaterial to the number — and this is measured, not argued.** Clamping and
composing **commute exactly**. `max(L, ·)` and `min(U, ·)` are non-decreasing, and a non-decreasing
`f` satisfies `f(min(x, y)) = min(f(x), f(y))` and `f(max(x, y)) = max(f(x), f(y))`; `_widen` applies
`max(L, ·)` only to the lower component and `min(U, ·)` only to the upper, so no cross-clamp breaks
the correspondence, and `min`/`max` *select* an operand rather than compute one, so the identity is
exact in IEEE-754 too rather than merely to a tolerance. Verified exhaustively this session over
**every** table at n ∈ {12, 30, 38, 40} × DEFF ∈ {1.0, 1.2, 1.5, 2.0, 4.0, 7.0} — 28 912 tables,
173 472 combinations: **zero** differences between compose-then-clamp and clamp-then-compose.

*The premise that makes the one-sided clamp safe is separately measured, because it is the kind of
thing that is true of today's caller and quietly false of tomorrow's.* `_widen` applies `max(L, ·)`
to the lower component **only** and `min(U, ·)` to the upper **only**, so an arm whose whole widened
interval sat above `U` would come back **inverted** (`lo > hi`) with nothing checking it. It cannot:
widening moves each bound *away* from the point estimate, and **both arms always contain the point
estimate** — verified over every table at n ∈ {12, 30, 38, 40, 85}, 138 648 tables and 277 296 arm
intervals, zero exceptions. Rule 4a therefore needs no ordering guard; but a future arm that does
not contain the point estimate reopens this, which is why the property is written down rather than
assumed.

**So state the honest headline plainly, because it scopes the work.** *Arms-versus-composed is
immaterial to the statistics and material only to the audit bullet.* The printed interval, its
coverage, its width and every verdict are **bit-identical** under either choice. This is a
**reporting** correction, not a statistical one — cheaper and different work than `P8-1`'s framing
implies. Three further properties, over the same 173 472 combinations, zero violations each:

- the printed interval never excludes the point estimate (`(b−c)/n ∈ [-1, 1]` by construction, and
  `max(-1, lo) ≤ max(-1, θ̂) = θ̂`);
- **clamping can never change zero-exclusion, hence never a verdict** — `max(-1, x) > 0` iff `x > 0`
  and `min(1, x) < 0` iff `x < 0`, so the clamp cannot move a bound across zero;
- Rule 4's own conservatism property survives verbatim: the envelope is never tighter than the
  clamped MOVER-D arm.

And truncation at the support is coverage-preserving **in principle**, not merely in this sweep: the
estimand lies in `[-1, 1]` by definition, so removing the region outside it cannot remove it. The
clamp is a free narrowing — it buys width at no coverage cost, which is why it stays.

**But the reporting defect is larger than `P8-1` measured, and *neither* of `P8-1`'s two suggested
fixes closes it.** The tie-break is the visible symptom of a wider one: **every** bound the clamp
moves is printed beside a sentence naming an arm that did not produce it, whether or not the
tie-break also picks the wrong arm. Measured this session, at n=40, counting printed bounds:

| DEFF | 1.0 | 1.2 | 1.5 | 2.0 | 4.0 | 7.0 |
|---|---|---|---|---|---|---|
| bounds whose printed value came from the **support** | 0 | 38 | 58 | 132 | 570 | 2 038 |
| of which `P8-1`'s wrong-arm subset | 0 | 0 | 6 | 12 | 166 | 954 |
| support-pinned tables whose verdict is **`distinguishable`** | 0 | — | 58 of 58 | 132 of 132 | 540 of 570 | — |

Two things that row three settles. The false sentence is **not** confined to the not-distinguishable
path — at the design effect a determinism probe is most likely to return, *all* of it lands beside a
published, positive verdict. And the tie-break subset is between 10% and 47% of the false sentences,
so fixing only it leaves the majority in place.

**The separating case, and it is the one that decides between the three options.** `(a=0, b=0, c=38,
d=2)` at n=40, DEFF 1.5 — a plausible measured design effect on the guard-judge pack's own slice:

- unclamped MOVER-D `[-0.99431, -0.77289]` — **inside** the support;
- unclamped exact paired bootstrap `[-1.01124, -0.85814]` — outside it;
- composed unclamped lower bound `-1.01124`, printed `-1.0`;
- verdict: **distinguishable**, CI `[-1.0, -0.77289]`.

The shipped code prints `lower bound: exact paired bootstrap`. **`P8-1`'s suggested fix — attribute
from the unclamped arms — prints the same thing**, because the exact arm *is* the more negative one.
Both are false: the exact arm produced `-1.0112`, and `-1.0` came from the support. That is M-ML-8's
error a third time, and only the third token removes it.

**What a support-pinned bound means, stated so the reader is not left to infer it.** It says the
`√DEFF` widening ran off the parameter space — the declared design effect has consumed more than the
whole resolvable range on that side, so the interval carries **no information in that direction**.
This is a labelling change, not a new gate: Rule 7's floor already refuses the degenerate
configurations that matter (at n=12, DEFF 7 the floor is 350 pp and nothing is resolvable). A reader
seeing `[-1.0, -0.77]` should know the `-1.0` is a boundary rather than an estimate, and the bullet
is where that is cheapest to say.

**The rendered bullet, verbatim.** The `p=` clause attaches only to the exact-bootstrap arm, exactly
as today; **a `support bound` token never carries a level, because no level produced it**:

> `- decided by: conservative envelope (lower bound: support bound (-1); upper bound: MOVER-D)`

**The tie-break stays, and is now pinned.** Ties between *unclamped* arms are real but rare and
structural — swept at DEFF 1.00 through 7.0, they are exactly the boundary tables `(0, n, 0, 0)` and
`(0, 0, n, 0)` (both arms reach ±1) and `b = c = 0` with `a, d > 0` (both arms are the point `0`):
3 at n=12, 2 at n=30, 3 at n=38, 3 at n=40, and the count does **not** move with the design effect,
which is the signature of a structural tie rather than a manufactured one. In every one of them
**both arms attain the printed bound**, so naming either is a true sentence and no fourth token is
owed. Keep `<=` / `>=`, and assert it: Pass 8's mutation 6 shows nothing pins it today.

**Rejected alternatives, with what decided each.**

1. **Fix the tie-break only** — attribute from the unclamped arms, keep the arm-level clamp
   (`P8-1`'s first suggestion). *Rejected on the separating case above*: it repairs 10–47% of the
   false sentences and is *confidently wrong* on the rest, and it leaves two versions of each arm
   in flight — one clamped for the interval, one unclamped for the audit — which is the
   one-arithmetic-two-homes shape this note refuses everywhere else.
2. **Keep the arm clamp and give a clamp-pinned bound its own token there** (`P8-1`'s second
   suggestion). *Rejected as under-determined*: pinning is per-arm, and the composed bound can be
   pinned in one arm and not the other — `(0, 0, 38, 2)` is exactly that — so the rule would still
   owe an adjudication of "one arm pinned ⇒ is the composed bound a support bound?". Composing
   first makes the question not arise.
3. **Do not clamp at all; print `[-1.03, 0.13]`.** *Rejected*: a printed CI outside the parameter
   space is a false statement about a difference of proportions, and it is strictly wider than
   necessary. The clamp is a coverage-free narrowing (above).
4. **Widen on a transformed scale — logit or arcsine — so the interval cannot leave the support.**
   *Rejected on a cost far larger than what it buys*: Rule 5's `√DEFF` is exact **on the difference
   scale**, because DEFF is a variance ratio there; re-scaling would change the printed interval on
   *every* table, where the clamp touches **none** at DEFF 1.00 and 0.31%/0.47%/1.07% of the 12 341
   tables at n=40 for DEFF 1.2/1.5/2.0 — and it would break §3.2c's *floor on conservatism* property
   against MOVER-D. Reopen only if a metric arrives whose support makes
   truncation bind at DEFF ≈ 1.00 — nothing in §3.8's packs does.

**What must be asserted — the fix is not evidence until one of these would catch it being wrong.**
Values below were computed against the shipped module this session; the two witness tables are
`(5, 0, 7, 0)` (`P8-1`'s own) and `(0, 0, 38, 2)` (the separating case).

1. **The arms are not clamped.** `envelope_arms((5, 0, 7, 0), design_effect=4.0)` returns
   `mover ≈ (-1.030146, +0.133341)` and `exact ≈ (-1.083333, -0.083333)`; assert **both lower bounds
   are `< -1.0`**. Kills a reinstated `clamp=(-1.0, 1.0)` inside `envelope_arms` — the mutation this
   whole ruling is about.
2. **The printed interval did not move.** `conservative_envelope((5, 0, 7, 0), design_effect=4.0)
   == (-1.0, 0.13334065646719284)` and `conservative_envelope((0, 0, 38, 2), design_effect=1.5)
   == (-1.0, -0.7728921326614777)`. Kills dropping the clamp from the composer (alternative 3), and
   is the assertion that makes "the number is unchanged" evidence rather than a claim.
3. **`P8-1`'s own cell, sharpened — and the ruling deliberately disagrees with `P8-1` here.**
   `verdict(_outcomes(5, 0, 7, 0), resolving=_rp(12, deff=4.0, basis="measured"), metric_name="m",
   family=["m"]).bound_by == ("support bound", "MOVER-D")`. `P8-1` asks only that `[0]` stop being
   `"MOVER-D"`; it must **not** become `"exact paired bootstrap"` either.
4. **The cell `P8-1`'s suggested fix gets wrong.** `verdict(_outcomes(0, 0, 38, 2), resolving=_rp(40,
   deff=1.5, basis="measured"), …).bound_by[0] == "support bound"`, with `.distinguishable is True`
   and `.ci[0] == -1.0` asserted in the same test — the false sentence sat beside a *positive*
   verdict, and that is the fact the test has to carry.
5. **Strictness of the support comparison.** `verdict(_outcomes(0, 0, 12, 0), resolving=_rp(12,
   deff=1.0, basis="measured"), …)` has `ci[0] == -1.0` and `bound_by == ("MOVER-D", "MOVER-D")`.
   Kills `<=` substituted for `<` in the support test — a bound at the support that no clamp moved
   is an arm's.
6. **The tie-break is pinned.** `verdict(_outcomes(0, 12, 0, 0), resolving=_rp(12, deff=1.0,
   basis="measured"), …).bound_by == ("MOVER-D", "MOVER-D")`; the upper bound is a genuine unclamped
   tie (both arms return exactly `1.0`). Kills Pass 8's surviving mutation 6 (`<=,>=` → `<,>`).
7. **The token is about the clamp, not about the table.** `verdict(_outcomes(5, 0, 7, 0),
   resolving=_rp(12, deff=1.0, basis="measured"), …).bound_by == ("exact paired bootstrap",
   "MOVER-D")` — the *same* table one design effect down, where no clamp binds. Assertions 3 and 7
   are a pair and are worth writing adjacently.
8. **The commutation property, asserted rather than asserted-about** — the shape of Rule 4's
   acceptance item 4. Over every table at n=12 and `DEFF ∈ {1.0, 1.5, 4.0}` (455 × 3), composing the
   unclamped arms then clamping equals clamping each arm then composing. This is the property that
   makes the move safe, and it is what stops a future reader "restoring" the arm clamp in the belief
   that it changes the number.
9. **The clamp cannot change a verdict.** Over every table at n=12 and `DEFF ∈ {1.5, 4.0}`,
   `distinguishable` is identical whether the composer clamps or not. This is the assertion that
   catches a *support that is not the parameter space*: a clamp inside the data's own range would
   change verdicts, and nothing else in the suite would notice.
10. **The rendered string.** `"- decided by: conservative envelope (lower bound: support bound (-1);
    upper bound: MOVER-D)"` appears verbatim in the markdown for a report built on `(0, 0, 38, 2)` at
    DEFF 1.5. Kills a renderer that formats the new token through the `p=` branch — the current
    `zip(v.bound_by, (LEVEL_CI95_LO, LEVEL_CI95_HI))` would print `support bound, p=0.025`.

Assertions 1–7 and 10 are single-table and cheap; 8 and 9 are the two sweeps, and both run in
seconds at n=12. `P8-2`'s missing `mcnemar-exact` assertion is a separate finding and is not
discharged by any of these.


**Rule 5 — design effect is a variance ratio, not a width ratio.** (A correction to v1.1 §4.4,
which called the width ratio "the design effect"; an implementer following v1.1 literally would have
divided by 2.6 where the truth was 7, over-stating effective *n* by ~2.7×.)

```python
def width_inflation(bootstrap_width: float, naive_width: float) -> float: ...   # the ratio
def design_effect(bootstrap_width: float, naive_width: float) -> float: ...     # ratio ** 2
def effective_n(n_observations: int, design_effect: float) -> float: ...        # n / DEFF
```
CI half-width scales as `1/√n`, so `DEFF = (bootstrap width ÷ naive Wilson width)²` and
`n_eff = n_obs / DEFF`. Check it against §4.4's real case: 280 turns in 40 conversations with
within-conversation correlation ρ ≈ 1 and m = 7 turns each gives `DEFF = 1 + (m−1)ρ = 7`, width
ratio `√7 = 2.646` — v1.1's "≈2.6" — and `n_eff = 280/7 = 40`, which is exactly the conversation
count. The identity is the check: **when ρ = 1, effective n must equal the cluster count.** Make
that a unit test; it is the one assertion that catches a squaring error in either direction.

**Rule 7 — no verdict path may return `distinguishable` when `|diff| < observable_floor`.** This is
a **required property of the verdict path itself, asserted in `verdict()`**, not a test the
implementer may or may not write. It costs one comparison and it is the only cheap check that ties
together two quantities computed by completely independent routes — the floor comes from
`b_min(α)/n_effective`, the verdict from McNemar's tail or the cluster bootstrap — so a defect in
either surfaces as a contradiction rather than as a plausible number. It is what would have caught a
bootstrap that silently resamples correlated rows i.i.d.: such a substitution changes the
instrument's name and not its interval, and an interval alone cannot report that.

Three details that decide whether it works:
- Compare against the **exact** floor, never the display-rounded one (Rule 3) — otherwise the
  invariant inherits the presentation layer's rounding and can fire or fail to fire by 0.05 pp.
- **The converse is not an invariant.** `|diff| ≥ observable_floor` does *not* imply
  `distinguishable`; the floor is necessary, never sufficient. Asserting the converse would be a
  bug that hides the discordance structure McNemar exists to read.
- **The response splits by path, because the invariant has two different statuses.**
  - On **`mcnemar-exact`** it is a **theorem, so a fire is a module bug → raise.** With the floor at
    the unadjusted α (Rule 3), `p ≤ α_step ≤ 0.05` implies `|b−c| ≥ b_min(0.05) = 6`, hence
    `|diff| ≥ 6/n`, at *every* Holm step. Verified exhaustively in this session over all `(b, c)`
    with `b + c ≤ 400`: **zero violations at α=0.05 and zero at α=0.025.** Silently demoting here
    would discard exactly the detector property this rule exists for.
  - On **`conservative-envelope`** it is a **guard, so a fire demotes and names** — the verdict becomes
    not-distinguishable with the floor breach printed as the reason. Raising would abort on ordinary
    clustered data: at DEFF = 2 on `(34, 6, 0, 0)` the widened interval still excludes zero at a
    15.0 pp difference while the floor has moved to 30.0 pp, which is a legitimate disagreement
    between a widened interval and a shrunken effective *n*, not a defect.
  - **This split only works with the floor at the unadjusted α.** At `α/k` the McNemar branch is
    reachable — a member rejected at a 0.05 Holm step with `|b−c| = 6` sits below a `7/n` floor —
    so the theorem is not available and `raise` would fire on correct data.

**Rule 6 — `cluster_bootstrap` is one-level, and a pack that needs two must fail validation.**

```python
def cluster_bootstrap(units: Sequence[Sequence[bool]], *, B: int = 10_000,
                      seed: int) -> BootstrapResult: ...
```
Each inner sequence is the observations belonging to one independent unit (for a turn-pooled count:
the turns of one conversation). Under 12×1 there are 12 members. `validate` **fails any pack
declaring `replicatesPerScript > 1`** while only this one-level function exists — the two-level
resample (§4.5) is then required and its absence must be an error, not an approximation.

**On the `√DEFF` widening used as the interim substitute, and why it is safe rather than merely
cheap.** Widening a percentile interval by `√DEFF` is exact for what it claims — DEFF is a variance
ratio and a CI half-width scales as `1/√n`, so `√DEFF` is precisely the factor carrying an interval
computed at `n_units` to one computed at `n_units/DEFF` — but it **rescales a variance it cannot
discover**, and in particular it does **not reproduce the few-clusters degeneracy** a genuine
cluster bootstrap has (§4.5.1(ii) rejects bootstrapping 3 shapes for exactly that reason). It only
widens; it never becomes unusable. **That gap is closed by a different mechanism, and the two
compose:** the floor is computed at `n_eff = n/DEFF`, so a too-few-clusters configuration is caught
by Rule 7 or by `UnattainablePower` rather than by the interval. Verified: at n=12, DEFF=7 the floor
is `6/1.714 = 350 pp`, i.e. no difference is resolvable at all and the run says so. **Neither
mechanism is sufficient alone** — the widening handles variance the floor cannot see, the floor
handles degeneracy the widening cannot — which is what makes the interim defensible until a pack
with `replicatesPerScript > 1` makes the structural primitive buildable against real data.
**A third mechanism was missing and v1.8 adds it:** neither the widening nor the floor sees the
*sparse-discordant-count* degeneracy of a percentile interval — at four non-zero rows in thirty the
2.5th percentile cannot be zero, whatever the design effect — and that is what Rule 4's
conservative envelope with MOVER-D closes (review M-ML-8).

**Rule 8 — the continuous verdict has its own producer, and it is not `verdict()` with fields left
empty.** *(v1.16, at plan v1.12's ask. §3.2e's strings 4 and 5 need a function to render them, and
the plan now calls one; its signature is this note's because the preconditions are.)*

```python
def continuous_verdict(
    diffs: Sequence[float],          # one difference per analysis unit (§3.2d), never per observation
    *,
    metric_name: str,
    family: Sequence[str],
    alpha_family: float,
    unit_kind: str,
    design_effect: float,
    basis: Basis,
    B: int,
    seed: int,
    support: tuple[float, float] | None,   # v1.17 — the METRIC's support, required, no default
    a_label: str = "A",
    b_label: str = "B",
) -> ContinuousVerdict: ...
```

**What it returns is a sibling type, not a `Verdict`.** `Verdict` carries `mcnemar_p`, `b`, `c`,
`marginal_overlap`, `floor_demoted` and `holm_tested`; **none of the six exists on this path**, and
filling them with `None` or a sentinel is the *"a field that is `None` until it is not"* shape this
note has already refused once for `alpha_step`. `ContinuousVerdict` carries `metric_name`,
`distinguishable`, `text`, `diff`, `ci`, `n_units`, `unit_kind`, `design_effect`, `basis`, `B`,
`seed`, `alpha_used`, and `decided_by: Literal["paired-bootstrap"]` so the `- decided by:` bullet has
something to print: `- decided by: paired bootstrap on per-<unit> differences (B=10000, seed=N)`.
`report.py` handles a union of the two verdict types; that seam is the plan's.

**The four parameters it deliberately does *not* take, each of which an implementer would otherwise
pass.** (i) **`resolving: ResolvingPower`** — it exists to make the observable floor's and the MDD's
sentences true, and a continuous metric has neither (§3.2e); passing it would put two meaningless
headline numbers within reach of the renderer, which is how string 2's grammar leaks onto string 5.
The four provenance fields it would have supplied are passed directly instead. (ii) **`alpha_step`**
— Holm's data-dependent threshold, and there is no ladder here (§3.3). (iii) **a McNemar *p*** —
§3.2d rules the interval *is* the test. (iv) **percentile levels** — see the next paragraph.

**The multiplicity correction is made unrepresentable rather than guarded.** The function takes
`alpha_family` and `family` and computes its own quantile levels as the **exact rationals** `α/(2k)`
and `1 − α/(2k)` with `k = len(family)` (§3.3), α recovered as `Fraction(str(alpha_family))` and
never as `Fraction(alpha_family)` — the second is the double's exact value rather than the declared
decimal, and §11.2.2 measures what that costs. It exposes **no percentile parameter**, so a caller
cannot render a `k = 3` family at `1/40` / `39/40` by omission. This is Rule 4's `n != len(diffs)` lesson
applied at the signature: remove the guard by making the error it catches unrepresentable. It
**raises** when `metric_name not in family`, for the same reason `verdict()` does.

**Four refusals, all of them cheap and all of them silent failures otherwise.** (1) `diffs` empty —
already `paired_bootstrap`'s behaviour and inherited here. (2) any element **non-finite** — a single
`NaN` propagates through the mean and both quantiles and arrives as a *rendered interval* rather than
as an error, which is §3.2d's carrier rule enforced at the second place it can be enforced.
(3) `design_effect < 1.0`. (4) `len(diffs) < 2` — a one-unit interval is a point and the string would
report a CI of zero width as though it were a measurement.

**Two changes the engine needs before it can serve this** (§3.4 Rule 4's `paired_cluster_bootstrap`
is still §3.2d's entry point, and `paired_bootstrap` still its engine, not a second one): the
**quantile levels must be parameters** rather than the hard-coded 2.5/97.5, or the `k > 1` correction
above has nowhere to land — and their type is `tuple[Fraction, Fraction]`, since `permille: int` and
a percent `float` both fail at `k ≥ 2` *(v1.18, §11.2.2)*; and **`_widen`'s `[-1, 1]` clamp must be conditional**, since it is a
difference-of-proportions assumption and `sep_z` is unbounded (Rule 4, and Rule 4a for *where* a
support is applied — on this path, already correctly, to the one interval returned). Both are noted there; they
are named here because they are this function's preconditions, and a plan that calls it without them
calls something that cannot render string 4 correctly for two of the three continuous metrics.

**`support` is required, and it takes no `clamp`** *(v1.17, at plan v1.13's §7 rule 3 raise; the
plan's recommendation, accepted with a sharper shape)*. Adding a parameter to a function whose
**negative** parameters were the point needs a reason, and the reason is that the four refused above
are two kinds and `support` is a third:

- **The quantity does not exist on this path** — `ResolvingPower`, `alpha_step`, a McNemar *p*.
  Refusing these refuses a category error.
- **The quantity exists and is derivable from what the function already holds** — the percentile
  levels, computed from `alpha_family` and `family`. Refusing this removes a way to get it wrong.
- **The quantity exists, is needed, and is *irreducible*** — `support`. It cannot be recovered from
  `diffs`: a sample of MRR differences lying in `[−0.3, 0.3]` is indistinguishable from a sample of
  z-differences lying there, and inferring a support from an observed range is the silent-wrong
  default the clamp rule exists to refuse. **This is `design_effect`'s own principle** (Rule 2), and
  the symmetry is exact: `_widen` can no more discover a metric's support than a resample can
  discover clustering the declaration did not state. Both are facts about the *metric*, known to the
  party that defined it, so both are **declared and forwarded, never inferred**.

So the parameter is keyword-only and **required with no default**, and its `None` is a **stated
value** — *this metric is unbounded, and I am telling you so* — never an absent one. That is this
note's standing absent-never-defaulted discipline at a fourth field, and it is what makes wiring an
unbounded metric a decision the caller must take rather than one it inherits.

**What it does with each value.** The clamp is derived **inside** the function and exposed nowhere:
`None` when `support is None`, otherwise `(lo − hi, hi − lo)` — the support of the *difference*, not
of the metric, which is the conversion worth writing once rather than at each call site where its
sign and order can be transposed. Making `clamp` unrepresentable at this surface is the same move as
the percentile levels, so v1.16's grain is preserved rather than broken. `sep_z` passes
`support=None` and is never clamped; `mrr` passes `(0.0, 1.0)` and is clamped to `(−1.0, 1.0)`. A
fifth refusal joins the four above: **`support` with `lo >= hi` raises** — a degenerate support
derives a clamp of `(0, 0)` and would pin every bound to zero silently.

**Two callers, and they are not the same surface.** A **verdict** metric reaches the engine only
through this function, which is why `support` belongs here. `sep_z` is reported and not verdicted
(§3.2d), so its comparison calls `paired_cluster_bootstrap` directly and states that engine's
`clamp` itself. Neither surface should be "simplified" into the other: one takes the metric's
support and derives, the other takes the derived value because it has no metric to ask.

**Why nothing built today turns on it, stated so the premise cannot go stale.** `_widen` scales
half-widths by `sqrt(design_effect)`, so at **1.00** it returns its input interval unchanged and no
clamp can bind — for *every* metric, not merely for `mrr`, since an unwidened bound is a bootstrap
percentile of per-unit differences and already lies inside the difference's own support. So an
interim `clamp=None` is correct for an unbounded metric always and inert for a bounded one at
`design_effect == 1.00`. **Rest the interim on that identity and not on a census of which packs
declare a continuous verdict metric** — the embedder's `designEffect` is 1.00 by construction in
§7.2's sense (unit ≡ query ≡ item, one observation per unit, preserved by §7.4's own 38 → 60 growth
path), but that is a fact about the packs declared today, and a pack census is the premise that goes
stale without anyone editing the sentence that rests on it. The question goes live at
`design_effect > 1.0` on a bounded continuous verdict metric, which nothing refuses.

**Rendering precision, decided once so it is not decided three times:** `diff`, both bounds and
string 5's half-width all print at **three decimal places**, rounded to nearest — a printed
difference asserts no bound, so §3.4 Rule 3's question returns *nearest* here exactly as it does for
§11.7's latencies. A half-width printed at a different precision from the bounds it is derived from
is an arithmetic inconsistency a reader will find before a test does.

---

## 4. Q2 — FR-8/FR-9 denominators, precondition handling, per-turn reporting

### 4.1 Notation

Per turn `t` of a scripted conversation, the pack supplies ground truth and the harness records:

- `R(t)` — the required tool calls (tool name + expected arguments). Possibly **empty** (an
  abstention turn, a chit-chat turn) — those turns are first-class, not filler.
- `E(t)` — the calls actually dispatched, from the harness's own dispatch trace (FR-10).
- `P(t)` — whether the reply contained a prose-shaped pseudo-call (harness heuristic).
- `I(t)` — the number of model calls in the turn that **completed**, i.e. returned a response the
  harness could record. It is `0` on a turn whose first call raised, and **`0` means *no iteration
  was observed*, never *the model used no iterations*** — §4.2(f) is where that distinction is
  spent. It is one number with `-ml` §11.4's `callCount` and plan §4 S1's `TurnTrace.iterations`,
  bound by assertion (§11.10 (7d)) rather than by convention.
- `D(t)` — the **mechanism** that ended the turn, over a closed **five**-member set:
  `replied`, `cap-hit`, `timed-out`, `no-response`, `server-rejected`. A mechanism vocabulary,
  deliberately not a scoring one; §4.3 rule 4 is the single place it maps onto this section's
  denominators, and the three subsets that matter are all different. *(Plan v1.26 records four,
  folding `timed-out` into `no-response` — a fold this note's own §11.5.1 unfolded at v1.14 for
  `censoringExact`, and one that leaves the scorer unable to tell a `fail` from an `unrunnable`;
  §4.3.1 item 2 is the ask.)*
- `S(t)` — simulated tool state after the turn.

**Hard design rule that makes every denominator below well-defined: the harness always drives the
full script.** A turn is never skipped because a previous turn failed. This is what the prior §8.2
run did (turns 5–9 recorded after the turn-4 collapse) and it is what keeps the per-turn-position
denominator equal to `n` conversations at every `t`, rather than a selection-conditioned subset.
If a turn cannot be driven at all it is recorded as **`unrunnable`** and reported in its own
count — never as a failure, never silently dropped. **Two channels reach it, not one** *(v1.23)*.
The **model channel**: LM Studio 400 / crash, as `gpt-oss-20b` produced — `D(t) ∈ {no-response,
server-rejected}`. The **tool channel**: a pack's `ToolEnvironment.dispatch` raised, so the
harness could not execute the call the model chose. The discriminator is rule 4's and is the same
on both — the harness holds no observation of the model at that turn, and cannot attribute the
failure — and the second channel is **not** a sixth `D(t)` member: a dispatch raise is
conversation-scoped (it invalidates the environment, not just the turn), so it is carried as a
conversation-level censoring marker and `D(t)` stays at five
(`docs/plans/small-model-benchmarking-ml-dispatch-failure.md` §4(c)).
A model whose comparison rests on 8 runnable turns out of 64 is not comparable to one with 64, and
the report must make that visible rather than averaging it away.

### 4.2 The seven counts — exact denominators

FR-8's (a) and (b) should **not** be two independently-reported rates. They share a denominator and
"did the model intend a call?" is unanswerable, so reporting them separately forces the
implementer to invent an intent heuristic. Collapse them into one 3-way partition:

**(a)+(b) — Emission form.** Denominator: turns with `|R(t)| ≥ 1`. Three mutually exclusive,
exhaustive outcomes, printed as a partition that sums to the denominator:

| outcome | condition |
|---|---|
| `native` | `|E(t)| ≥ 1` |
| `prose_pseudo_call` | `|E(t)| = 0` and `P(t)` fired |
| `no_attempt` | `|E(t)| = 0` and not `P(t)` |

FR-8(a) "called a tool when required" = the `native` rate (a prose pseudo-call is *not* a
dispatched call). FR-8(b) "native form rather than prose" = the `prose_pseudo_call` vs `no_attempt`
split of the failures. Both requirements satisfied, no intent inference needed.

`P(t)` is a heuristic, so the pack must ship a small labelled set of replies (~20, human-verified)
and the report prints **the prose detector's own precision/recall** next to the partition. Prior
art: review §8.3 did exactly this for `_note_possible_fabrication` and it is what turned a shipped
signal into a trustworthy one. An uncalibrated detector's number is not a measurement.

**(c) — Right tool chosen.** Denominator: turns with `|R(t)| ≥ 1` **and** `|E(t)| ≥ 1`
(arguments/tool identity are undefined when nothing was called). Numerator: `E(t)` contains at
least one call to every tool name in `R(t)` — *coverage of the required names*. Extra calls are
(e)'s business, not (c)'s; keeping them out of (c) is what stops one failure being counted twice.

**(d) — Argument correctness. Unit shifts to the *call*, not the turn.** Denominator: dispatched
calls whose **tool name is correct** (arguments of a wrong-tool call are meaningless). Print
`n_calls` explicitly — it is a different denominator from every other count and a reader will
otherwise assume turns.

The requirement's "split three ways" is not three disjoint buckets on equal footing. The correct
structure, and the one to implement:

- **Headline:** `all_args_correct` — binary per call.
- **Failure decomposition, per *argument* not per call:**
  - `omitted_required` — a required parameter is absent.
  - `wrong_value` — present but ≠ expected after canonicalization (reuse
    `nlq_scoring._scalar_equal`'s discipline: numeric epsilon for numbers, casefold+whitespace-
    collapse for strings, **never coerce across types**).
  - `boundary_unit` — **a named subset of `wrong_value`**, not a sibling: the wrong value is
    explained by a declared boundary/unit rule (inclusive-vs-exclusive bound, `$`/cents,
    kg/g). Report it as `wrong_value: 12, of which boundary/unit: 7`.

Make it data-driven, not heuristic: each expected argument in the pack carries an optional
`boundaryRule` listing the alternate encodings that count as boundary confusions. Otherwise the
classifier is a regex nobody calibrated. (This is a live falkor-chat failure class — K-057's
inclusive-bound wording fix — so the pack should carry those cases.)

**(e) — Spurious and duplicate calls. Two separate counts, not one.** Denominator for both: turns
with `|E(t)| ≥ 1`.

- `spurious_turn_rate` — turns where `E(t)` contains a call to a tool not in `R(t)`.
- `duplicate_turn_rate` — turns where `E(t)` repeats an already-satisfied call, **either
  within-turn or re-issuing a prior turn's completed call**. Both sub-shapes are real observed
  defects (K-061 same-turn `add_to_cart`; §8.4's ministral turn-2 re-issue of turn 1). Report the
  within-turn and cross-turn variants as a named breakdown — the ministral defect was
  turn-2-specific and a pooled duplicate rate would have hidden it.

**(f) — Stopping when done, and the two iteration figures beside it. Three statistics, three
denominators, and they are not the same denominator** *(v1.21, plan-gate P14-6. Until now this
bullet named **one** denominator and then reported three figures, only one of which used it —
which is exactly why "does a non-`replied` turn enter the `I(t)` summary?" had no answer in the
text. Each figure now carries its own.)*

- **`stopping_when_done`.** Denominator: turns with `|E(t)| ≥ 1`, less the two `unrunnable`
  mechanisms (`no-response`, `server-rejected` — rule 4 below). *(v1.23: still **two**, and the
  tool channel does not make it three. `D(t)` has five members and this bullet's three denominators
  are all over turns that were driven; a turn lost to a raised `dispatch` is never driven to a
  record at all, so it subtracts here by absence rather than by an exclusion clause. The same
  reading carries to the two figures below and to `ITERATION_SUMMARY_EXCLUDED`, which stays at
  three members.)* Numerator: `D(t) == replied`
  **and** no call dispatched after `R(t)` was fully satisfied. A `cap-hit` turn is **in** the
  denominator and fails it — that is the count's whole purpose. A `timed-out` turn is in it and
  fails it too: outcome `fail`, never `n_a`, per §3.6's third disposition and rule 4's
  discriminator.
- **`iteration_cap_hit_rate`.** Denominator: **all turns driven**, less the two `unrunnable`
  mechanisms. Numerator: `D(t) == cap-hit`. **Keyed on the disposition alone and on nothing about `E(t)`** — a
  turn that reached the cap emitting only malformed tool calls reached the cap. This figure and the
  two below it are why the bullet is not a single binary rate: the `gpt-oss-20b` message-spam defect
  (§8.4) presented **purely** as cap-hits, and a `stopping_when_done` rate alone would have scored
  it as a failure without saying what kind.
- **Mean and p95 of `I(t)`.** Denominator: turns with **`D(t) ∈ {replied, cap-hit}`**, and no
  others. All **three** non-completion mechanisms are excluded — including `timed-out`, which is
  in the two denominators above and out of this one, because a turn cut off by the request budget
  carries no observation of where the model would have stopped. The combined excluded count is
  printed beside the figures (§4.3 rule 2). **Three figures, three different subsets of `D(t)`**,
  which is the whole reason this bullet no longer states one denominator.

**Why those two dispositions and not all four — four reasons, of which the second and third are
each decisive alone.**

1. **The estimand.** `I(t)`'s summary describes how much the model loops before it stops. `replied`
   and `cap-hit` are the two mechanisms in which *the model's own behaviour or the harness's
   declared cap* ended the turn, so they are the only two that carry an observation of stopping
   behaviour. A turn ended by a dropped socket carries none — and neither does one ended by the
   request budget, which is why `timed-out` is excluded here while scoring `fail` everywhere else
   (rule 4). A timeout bounds *how long*, not *how many*.
2. **The arithmetic, and it is this component's own recorded defect.** Such a turn contributes
   `I(t) = 0` — **forced**, not chosen: v1.20 pinned `callCount == len(ItemTiming.calls) ==
   TurnTrace.iterations` (§11.4), and a turn whose first call raised has no `ChatResult`. A `0` that
   means *not observed* entering a mean is the absent-as-zero shape this note has already ruled
   twice — `coldLoadSeconds` is **absent, never `0`** (§11.6), and `ItemResult.scored_outcome`
   **refuses** an absent count rather than reading it as a zero, the default that once turned an arm
   holding no data at all into *"+100.0 pp, p=0.002"*.
3. **The direction.** Including them biases the mean **down**, so a run in which the server rejected
   the model repeatedly reports that model as *more* efficient at stopping. That is §4.3's
   laundering with the sign reversed, and it is not hypothetical: `salesperson-tool-reliability-ml.md`
   §8.4 lost **6 of 8** `gpt-oss-20b` conversations to HTTP 400.
4. **Nothing is lost by excluding them**, which is what makes the exclusion cheap rather than a
   trade. The *cost* question — how many model calls did this run actually make — has its own
   unrestricted denominator since v1.20: `Y_calls / Y` over **every** item including the failed
   ones (§11.4). Two questions, two denominators, both printed. **They will differ on any run with a
   non-`replied` turn, and that difference is a testable consequence, not a discrepancy** — a report
   that shows them equal on such a run has substituted one for the other.

**Two censoring properties, computed and printed rather than assumed.** A `cap-hit` turn's `I(t)` is
**right-censored**: the observation is *the model wanted at least `maxIterationsPerTurn`*, not *the
model used `maxIterationsPerTurn`*. So:

> **The mean prints as `>= <value>` whenever the cap-hit count `c > 0`**, and as a bare value only
> when `c == 0`.
>
> **The p95 is exact iff `r ≤ X − c`**, where `X` is the summary population, `c` its cap-hit count
> and `r` the integer rank of §11.2.1 — and prints as `>= <value>` otherwise. Integer arithmetic,
> no float and no comparison of values: censoring that pushes the top `c` observations upward
> cannot move an order statistic that has `c` observations above it, and must move one that does
> not. The same one-sided argument §11.5 makes for the attained level, and the budget it buys is
> generous where it matters — at `X = 80`, `r = 76`, so the p95 is exact up to **4** cap-hits.

**The p95 is `stats.percentile(level=LEVEL_P95)` and inherits §11.3's identity floor.** §11.2.2's
one-implementation rule is package-wide, and an `I(t)` p95 is precisely the second quantile a reader
would otherwise hand-roll — the first cost this component an `index.csv` column computing
`latencyMsP95` at the 50th percentile. Inheriting §11.3 means that when `r == X` the figure is the
sample maximum and is labelled `max`, which is reachable here in a way it is not for latency:
a degraded run can leave the summary population well under 20 turns.

**No refusal gate, and the reason is that §4.3 already carries the instrument.** §11.6 refuses a
latency figure below a coverage floor because `index.csv` strips a prose qualifier from a bare
column; the `I(t)` figures are report-only — they are in no `index.csv` column — and §4.3's rules 1
and 2 already oblige every rate to print its `k/n` inline and its excluded count beside it. A third
floor here would add a refusal without adding information.

**The reach of `{replied, cap-hit}` is pinned in constants, in both directions** *(the component's
guard-reach convention, both halves)*. `ITERATION_SUMMARY_DISPOSITIONS = frozenset({"replied",
"cap-hit"})` **and** `ITERATION_SUMMARY_EXCLUDED = frozenset({"timed-out", "no-response",
"server-rejected"})` are written out **separately and literally** — the second is *not* derived as
the first's complement, because a derived complement makes the union assertion a tautology, which
is the exact defect the convention was amended twice to close. Three assertions:

1. **Union, against an independent declaration.** `ITERATION_SUMMARY_DISPOSITIONS |
   ITERATION_SUMMARY_EXCLUDED == convo.TURN_DISPOSITIONS`, and the two are disjoint. `TURN_DISPOSITIONS`
   is authored in `convo.py` by another unit, so this binds two independent declarations and
   **reddens on a widen**: a fifth disposition added by the rework unit belongs to neither set and
   fails here before it can be silently included or silently dropped.
2. **A distinct behavioural consequence per member**, five cases, each observable and each
   different: appending a `replied` turn moves the mean; appending a `cap-hit` turn moves the mean
   **and** flips it to the `>=` form; appending a `timed-out` turn leaves the mean and p95
   **unchanged** while still incrementing `stopping_when_done`'s denominator and its failures;
   appending a `no-response` turn leaves the mean and p95 unchanged **and** leaves that denominator
   untouched **and** increments `unrunnable`; appending a `server-rejected` turn does the same by a
   different mechanism token. Moving any one member between the two sets reddens exactly one of
   these, so the pin **reddens on a shrink** as well. The third and fourth cases are the pair that
   pins rule 4's `fail`/`unrunnable` discriminator — they differ in nothing but the mechanism.
3. **The exactness rule swept, both directions** — over `X ≤ 200` and every `c ≤ X`, raising each
   cap-hit observation to any larger value leaves the `r`-th order statistic unchanged **iff**
   `r ≤ X − c`. A test asserting only the *iff*'s forward half would bless a report that printed a
   bare p95 where a `>=` was owed.

**(g) — Final reply matches what the tool returned.** Denominator: turns with ≥1 dispatched call
whose return value is **fact-bearing** — i.e. the pack declares at least one checkable value for
that turn. Turns where nothing checkable came back go to an explicit `unscoreable` bucket that is
printed, not silently excluded.

Scoring is deterministic, **not a judge**: normalized-substring containment of every
`mustContain` value **and** absence of every `mustNotContain` value, both listed per turn in the
pack. This is the lab's established convention (`nlq_scoring.layer2_contains`, review §8.1's
value-containment ground truth). The negative list is the half that matters: it is what catches
"successfully removed from your cart" narrated over an unchanged cart — the exact failure whose
price-shaped-token blind spot §8.3 documented.

**Plus one count the requirement omits and needs — restraint.** Denominator: turns with
`R(t) = ∅`. Numerator: `|E(t)| = 0`. Without it, a model that calls tools indiscriminately scores
perfectly on (a) and the abstention turns contribute nothing. Cheap to add, and conditions A and C
already contain abstention turns.

### 4.3 Precondition failures must never be laundered

Five rules, all mandatory *(three until v1.21, four until v1.22)*:

1. **Every printed rate carries its denominator inline** — `k/n`, never a bare percentage.
2. **A turn excluded from a conditional denominator is counted in that count's own `n/a` tally**,
   printed next to the rate.
3. **The report opens with a funnel table**, not a metric table (the counts below are an
   *illustration of the shape*, not this pack's sizing — the 12×1 design drives ~80 turns per model,
   §4.5.2 — and the funnel must additionally print, at its head, the **conversation count and the
   analysis unit**, because every rate under it is a turn or call count while every verdict above it
   is computed over 12 conversations):

```
turns driven                  360
  unrunnable (model channel)    0   -> no-response / server-rejected
  unrunnable (tool channel)     0   -> dispatch raised; conversation censored at t
  R(t) = 0 (restraint turns)   40   -> restraint rate 38/40
  R(t) >= 1                   320
    native call emitted       142   -> (a)+(b) partition over 320
    prose pseudo-call          31
    no attempt                147
  turns with >=1 call         142   -> (c), (e), (f) denominators
  dispatched calls            167   -> (d) denominator  [NOTE: calls, not turns]
  fact-bearing returns        118   -> (g) denominator
  unscoreable returns          24
```

This is the mechanism that stops "100% argument correctness" on three calls from being read as
comparable to 95% on two hundred. Without it, a model that collapses early looks *better* on every
conditional count, because it never generated the calls that could be wrong. That is the single
most likely way this harness lies, and the funnel is the fix.

4. **The turn's *mechanism* maps onto every denominator in one place, and this is it** *(v1.21;
   plan-gate P14-1's shape, answered from the note's side, plus one contradiction P14-1 did not
   reach)*. `D(t)` is a mechanism (§4.1) and each §4.2 count is a scoring question, so the mapping
   is a rule here rather than a column on the plan's table of mechanisms.

   **The discriminator between `fail` and `unrunnable`, stated once.** A turn scores **`fail`**
   only where **the harness gave the model its whole declared budget and observed nothing come
   back**. That is one mechanism: the **timeout**. Every other non-completion is a failure of the
   request/response channel, about which the harness holds **no observation of the model at all** —
   and it must not be guessed at, because a `400` produced by a model's runaway message list and a
   `400` produced by a malformed harness payload are the same status code. So they are
   **`unrunnable`**: out of every scoring denominator, counted, and printed. **This is §11.5.1's
   own distinction, arriving at a second consumer** — *a timeout is a **censored** observation and
   a call that failed at 40 ms is a **missing** one* — ruled at v1.14 for `censoringExact` and
   binding here for the same reason: a censored observation has a known bound and is evidence; a
   missing one is neither.

   | mechanism | in a scoring denominator? | how it scores where it is in |
   |---|---|---|
   | `replied` | yes, wherever its own `R(t)`/`E(t)` condition admits it | on its own FR-8 counts |
   | `cap-hit` | yes, wherever its own `R(t)`/`E(t)` condition admits it | **fails** `stopping_when_done`; enters `iteration_cap_hit_rate`'s numerator and the `I(t)` summary; enters (g)'s `unscoreable` bucket **only if `|E(t)| ≥ 1`** |
   | **`timed-out`** | yes | outcome **`fail`**, never `n_a` — §3.6's third disposition, and the only non-completion that earns it. **Out of the `I(t)` summary** all the same (§4.2(f)): a budget bounds *how long*, not *how many* |
   | `no-response` (dropped connection, unparseable or unusable body) | **no** | §4.1's **`unrunnable`** count |
   | `server-rejected` (the server answered with a status and refused) | **no** | §4.1's **`unrunnable`** count |
   | **a raised `dispatch`** (the pack's tool environment could not execute the model's call) — *not a `D(t)` member; see below* | **no** | §4.1's **`unrunnable`** count, and the conversation is censored from `t` (rule 5) |

   **The tool channel takes the same discriminator, and the row above is where it lands**
   *(v1.23, folding `docs/plans/small-model-benchmarking-ml-dispatch-failure.md`)*. A pack's
   `dispatch` that raises is the model channel's argument one channel over: the trigger is
   **partly model-chosen** — the model picks the arguments, so a weaker arm drives a sim into a
   raise more often — and the harness cannot attribute the raise between *the model produced an
   argument the sim did not anticipate* and *the sim is simply broken*, exactly as it cannot
   attribute a `400` between a runaway message list and a malformed harness payload. So it is
   `unrunnable`, and emphatically **not** *undispatchable*: an undispatchable call contributes
   nothing to `E(t)`, so a turn whose only call landed there partitions as `no_attempt` under
   §4.2(a) — **a failure charged to the model** — which would launder a harness fault into a
   scored model failure, silently, and more often for the weaker arm. **The primary defence is
   upstream and belongs to the pack**: `dispatch` is *total* over `(str, dict)`, every
   input-shaped problem being a returned error value rather than a raise, so this row governs a
   residual rather than a design path.

   **Three consequences, and the second and third are where the plan is currently wrong.**

   *(i)* An `unrunnable` turn leaves **before** any `R(t)`/`E(t)` condition is evaluated, which is
   already the funnel's shape above — subtracted at the top, one line under *turns driven*.

   *(ii)* **A disposition does not by itself decide what scores a turn.** Except for
   `iteration_cap_hit_rate` and the two `I(t)` figures — keyed on the mechanism alone, and on **two
   different subsets** of it (§4.2(f)) — every §4.2 count is conditioned on `R(t)` or `E(t)` too,
   so the mapping is a function of the **pair**. The case that proves it is plan-gate P14-1's: a
   `cap-hit` turn with `|E(t)| = 0`, which the loop reaches whenever the model emits only
   undispatchable tool calls. That turn is `no_attempt` under §4.2(a) — a **failure** — and is
   outside (f)'s `stopping_when_done` denominator and outside (g)'s denominator entirely, so it
   cannot occupy (g)'s `unscoreable` bucket. It is **not** *absent, not failed*. The funnel already
   routes it correctly, under *no attempt*; the plan's four-row table is a second home for a
   mapping that already had one, and it is the second home that is wrong.

   *(iii)* **A second row of that table is not a single scoring population either, and this one is
   a live contradiction between two plan sections rather than an omission.** Plan §3.6's fourth
   disposition rules that *"a non-2xx response, a dropped connection or an unparseable body"*
   scores **`fail`, never `n_a`**; plan §3.8.4's v1.26 table routes a status-bearing failure to
   **`unrunnable`** while citing that same fourth disposition for the row beside it. Both are live
   at v1.26 and they cannot both stand. **This rule decides it against §3.6's outcome clause and
   for §4.1's**, on the discriminator above — and the reversal costs §3.6 nothing else it was
   written for: P4-7's fourth disposition exists to close a `LatencyBlock` accounting hole (an item
   withheld under neither named cause falsified invariants (iii) and (iv)), and every part of that
   fix — the third `withheldFor` value, the one counter, the `ItemTiming` that exists without a
   figure — is untouched. Only the sentence assigning the **scored outcome** moves. **The
   discriminator needs no new field**: `drive` catches `LMStudioCallTimeout` and `LMStudioCallFailed`
   separately already, so the mechanism is known where the disposition is assigned, and the plan's
   own record has carried the distinction since plan-gate P5-6 for `censoringExact`. What it does
   need is for the mechanism set to **name** it — see §4.3.1 item 2.

   **Why `unrunnable` cannot be an escape hatch, which is the objection §3.6 raised and answered
   the other way.** §3.6's argument is that scoring a non-completion `n_a` *"would let a model that
   hangs out-score one that answers wrongly"*. That is true of a **rate whose denominator silently
   shrinks**, and it is exactly why §4.3 rules 1–3 exist. Under those rules an `unrunnable` turn
   costs the model **`n`**, visibly: the conversation leaves the paired table for **both** arms
   (§4.3's intersection corollary), the funnel prints the count at its head, and §7.1's
   resolving-power line renders the collapsed `n_effective` and the widened floor in its own words —
   §8.4's `gpt-oss-20b`, which lost **6 of 8** conversations to HTTP 400, would arrive at the
   headline with an `n` the report states rather than a rate it invents. Losing power and saying so
   is §4.5.3's own trade; scoring a box's bad twenty minutes as a model's failure at `n = 12`
   manufactures a difference, and one dropped connection is worth **8.3 pp** on the headline and can
   flip a McNemar discordant pair. This note has ruled that comparison once already, one field over:
   storing the timeout constant in the p95 *"would print a figure about the configuration rather
   than about the model"* (§11.5).

   **And the holes that must close with it, or the escape is real.** Under the rule above an
   `unrunnable` turn is not a failure, so **every statistic whose predicate is *this conversation
   has not failed yet* reads it as clean** unless told otherwise. `cleanThroughTurnH` is one such
   statistic — *"the fraction of conversations with zero failure of any kind through turn `H`"*,
   saying nothing about `unrunnable` — and v1.21 ruled it in isolation: **`cleanThroughTurnH` takes
   a third state**, a conversation with an `unrunnable` turn at any `t ≤ H` being **neither clean
   nor failed**, out of the headline's denominator and into the headline's own `n/a` tally (rule
   2), printed beside it. That ruling stands unchanged, and no new gate is needed to stop the
   resulting `n` from being read as sound: `verdict()` already refuses below
   `resolving.observable_floor` (§3.4 Rule 7) and the floor is computed from the surviving `n`, so a
   collapsed denominator refuses itself. **What v1.21 got wrong is the count** — it called this the
   *one* remaining hole, and there are three consumers of that predicate. **Rule 5 is the total
   map**, and it exists because a fix applied only where the defect was noticed is the failure this
   coordination has paid for most *(v1.22, plan-gate P15-1)*.

5. **An `unrunnable` turn ends its conversation's *trajectory*: every statistic indexed by turn
   position drops that conversation from that position onward** *(v1.22, plan-gate P15-1 — the
   generalisation of v1.21's `cleanThroughTurnH` third state, which was one consumer of this
   predicate ruled as though it were the only one)*. Rule 4 maps a turn onto the **turn-level**
   denominators; rule 5 maps the same turn onto the **conversation-level** ones. Two different
   questions about one turn, and answering only the first is what left the hazard reading an
   unobserved turn as a pass.

   **Why *from `t` onward* and not *at `t` alone*.** Two independent reasons, either sufficient.
   **(i) The estimands are conditional on history.** The hazard's *clean through `t−1`* and the
   headline's *zero failure through `H`* are claims about a **trajectory**, and a trajectory with an
   unobserved turn in it answers neither; carrying it forward as clean **imputes a pass** on a turn
   nobody observed, which is rule 4's laundering one level up. **(ii) The stimulus downstream is
   contaminated.** A turn with no final reply has no assistant message to replay, so under any
   `historyReplay` but `"none"` every later turn of that conversation is answered against a history
   the pack does not declare — the outcome at `t+1` is then partly the harness's, which is §4.1's
   own reason for `unrunnable` arriving one turn later. Reason (ii) is what makes this a
   **carry-forward** rather than a per-position hole, and it is the one that is conditional:
   *reversal trigger* — a pack declaring `historyReplay: "none"` **and** a per-turn hazard could
   narrow the exclusion to position `t` alone; no declared pack does, and the branch is not built
   until one exists.

   **This is censoring, and it is the survival answer rather than a convenience.** A conversation
   censored at `t` **keeps its observations at `1 … t−1`** — they are real, and discarding them
   would throw away good data and widen every earlier position for nothing. Only its risk-set
   membership at `≥ t` goes. The alternative — carrying it in the denominator under a distinct
   state — puts a non-observation in a rate's base and deflates the hazard, which is the same
   defect as imputing a pass with an extra label on it.
   **And the censoring may be informative, which this note will not assume away.** An HTTP 400
   raised by a model's own runaway message list is *caused by* the degeneration the hazard exists to
   measure, so the conversations that leave are plausibly the ones about to fail and the bias runs
   **downward** — toward the flat, low curve that reads as "gradual degradation", which is the exact
   misreading FR-9 exists to prevent. Censoring is unbiased only under independent censoring, an
   assumption this design cannot make, so §4.6 prints a **bound** beside the point estimate rather
   than claiming one.

   **The three consumers, and the map is total.**

   | consumer | what an `unrunnable` turn at `t` does | mechanism |
   |---|---|---|
   | **`cleanThroughTurnH`** (§4.6, headline) | `t ≤ H`: **neither clean nor failed** — out of the denominator, into the `n/a` tally (rule 2). `t > H`: **nothing** — its first `H` turns were fully observed, so it stays clean-or-failed | a per-conversation **ternary**; one observation per conversation |
   | **the per-turn hazard** (§4.6, always printed) | out of the **risk set at every position `≥ t`**, numerator and denominator both; **in** at `1 … t−1` | position-indexed **censoring**, with `c_t` and §4.6's bound printed |
   | **the per-position table** (§4.4) | out of that position's `n` at every position `≥ t`; the printed `n` is the **observed** count, never the structural 12 / 8 / 4 | the same censoring, on an unconditional rate |

   **The two are not the same state, and the difference is testable**: with `H = 4`, an
   `unrunnable` at `t = 5` leaves the conversation **in** `cleanThroughTurn4`'s denominator and
   **out** of the hazard from `t = 5`. An implementation that collapses the two rules into one
   passes every test built only from an early `unrunnable` turn (§4.3.1 item 11).

   **What rule 5 does *not* reach: §4.2's turn-level counts (a)–(g) and the `I(t)` summary.** Their
   estimand is **per-turn behaviour**, not a trajectory, and a turn answered against a shortened
   history is still an observation of what the model does with the turn it was given. Excluding
   those turns would discard most of the evidence on exactly the runs where behaviour is most in
   question, to remove a contamination whose effect on a *behavioural* rate this note has no basis
   to sign — and an exclusion whose direction is unknown is not conservative, it is just smaller.
   It is **disclosed rather than assumed away**: the funnel (rule 3) prints, under its
   **model-channel** `unrunnable` line, the count of turns scored **after** an `unrunnable` turn in
   the same conversation — under the tool channel that count is `0` by construction (v1.23's
   exception below leaves no such turns), which is what makes the two lines worth separating. One
   integer from the same pass, no gate and no second denominator. *(This asymmetry is the answer to
   "the same third state, or a different treatment?" — one predicate, three consumers, two
   mechanisms, and a fourth group deliberately untouched.)*

   **One exception, and it is the tool channel's alone** *(v1.23)*. The carve-out above rests on
   **history** contamination: a turn answered against a shortened replay is still an observation of
   what the model does with the turn it was given, and excluding it would discard evidence to
   remove a bias whose direction this note cannot sign. That reasoning does not survive **state**
   contamination. After a raised `dispatch` the environment's state is unknown — the raise may have
   mutated the cart, may have recorded a `DispatchRecord`, and `drive` cannot tell — so FR-10's
   ground truth is itself unreliable from `t` onward, and the later turns of that conversation are
   not observations of *anything*, behavioural or trajectory-level. **So for this cause, and only
   this cause, the censoring reaches (a)–(g) and the `I(t)` summary too.** The implementation is
   not an exclusion rule: those turns are **never driven**, the conversation ending at `t` with
   turns `1 … t−1` kept, which is also what returns their inference budget (§4.5.2's binding
   constraint). Discriminator, stated so the two are not collapsed: a conversation censored by
   `no-response` at `t` still contributes turns `> t` to (a)–(g); one censored by a raised
   `dispatch` at `t` has no turns `> t` at all.

   **The sweep, and its result — P15-1 asks for exhaustive rather than sampled.** Every occurrence
   of the token was read at v1.22 (`grep -ni clean docs/plans/small-model-benchmarking-ml.md`) and
   dispositioned. **Three sites condition on the predicate** and are the table above. **Two sites
   depend on the headline's surviving `n`** — §4.5.1's *"`cleanThroughTurnH` has 12 rows, so
   McNemar exact is a valid decision rule"* and §4.5's sampling-table cell — and both stand
   unchanged: rule 5 shrinks the headline no further than v1.21's third state already did, and
   `verdict()` refuses below the floor computed from the surviving `n` (§3.4 Rule 7). **One site**
   is §3.1's metrics-table row, which names the headline and not its denominator rules. The
   remainder — every occurrence under §11, plus §4.4's *"a clean binomial"* — uses the word in the
   unrelated sense of an uncontaminated measurement or a fully covered run, and none of them
   conditions on a conversation's history. **One consequence the sweep turned up that is not a
   rewrite of anything:** under censoring two arms' hazard curves are conditioned on **different
   risk sets**, so **no cross-arm hazard difference is printed** — the two curves go side by side,
   each with its own `c_t`, which is the discipline §11.7 slot 7 already applies to two arms'
   latency figures at unequal coverage.

**Paired-comparison corollary:** for the conditional counts (c)–(g), pairing only works on items
where *both* models produced a scoreable outcome. The paired `n` is the **intersection** and must
be printed separately from each arm's own `n`. Items scoreable for exactly one model are reported
as an `asymmetry` count — they are not missing data, they are a finding about the model that could
not produce them.

#### 4.3.1 What §4.2(f) and rule 4 need from the plan — `architect`'s, and precisely scoped

Stated here in one place because a fresh `architect` is writing plan v1.27 concurrently and this
note may not edit that document. **Nothing in §11.9 ask 7 is invalidated by anything below** — that
list is about *timing* and rule 4 is about *scored outcomes*, and the two meet only at item 3's
warning. Residual classification is at the end; nothing is deferred by choice.

1. **§3.8.4's four-row table, column 4 — the recommended fix is to delete the column, not to split
   the row.** The table declares itself *"the only home of that mapping"* and it is not: §4.3's
   funnel has routed these turns since v1.1 and rule 4 now states the mapping outright. Two homes
   for one mapping is what produced plan-gate P14-1, and the deeper reason is §7 rule 2 — the plan
   owns the harness surface and `TurnTrace`'s **mechanism** vocabulary, while which denominator a
   mechanism lands in is a scoring question and is this note's. So column 4 should become a
   **citation to §4.3 rule 4**, and the sole-ownership sentence should be deleted rather than
   corrected. **If the architect prefers the mapping visible in the plan**, P14-1's row split is the
   fallback and must be exactly rule 4's. Either way **P14-1's own fix sentence needs one correction
   as it lands**: it calls `iteration_cap_hit_rate` *"the one count keyed on the disposition alone"*.
   There are **three** — the cap-hit rate and the mean and p95 of `I(t)` — keyed on **two different
   subsets**, `all-less-unrunnable` for the first and `{replied, cap-hit}` for the other two. A
   reader who takes the singular literally puts the `I(t)` summary back under `E(t)`.
2. **The mechanism set is five, not four, and this one is time-critical.**
   `TurnDisposition`/`TURN_DISPOSITIONS` gains **`timed-out`**, split out of `no-response`. The
   reason is rule 4's discriminator: a timeout scores `fail` and every other non-completion scores
   `unrunnable`, so a token that folds the two leaves the S5 scorer unable to tell them apart. The
   information is already in hand where the disposition is assigned — `drive` catches
   `LMStudioCallTimeout` and `LMStudioCallFailed` separately — and this note's own §11.5.1 unfolded
   exactly this pair at v1.14 for `censoringExact`, so re-deriving the scoring split from
   `ItemTiming.withheldFor` would be the two-vocabularies collision the plan has paid for twice.
   **It must reach P14-3's precursor unit before `TURN_DISPOSITIONS` lands**, or a four-member set
   ships and the widen has to be caught later by the union assertion of §4.2(f)'s pin — which will
   catch it, at the cost of a rework. *The fallback — keep four members and have the scorer read
   `withheldFor` — is workable and is the collision shape, so if it is taken it should be recorded
   as a deliberate trade rather than arrived at by default.*
3. **§3.6's fourth disposition: the outcome clause moves, and only that clause.** *"A non-2xx
   response, a dropped connection or an unparseable body … scored per the pack's rule with outcome
   `fail`, never `n_a`"* becomes **`unrunnable`** (§4.1's count), leaving the **timeout** — §3.6's
   third disposition — as the only non-completion scoring `fail`. **Everything else P4-7 bought is
   untouched and must not be re-opened**: the third `withheldFor` value, the single
   `latencyWithheldForNoResponse` counter, invariants (iii)/(iv), the `ItemTiming` that exists
   without a figure, the re-probe/exit-3 asymmetry. P4-7's fix was a *record-keeping* fix and the
   outcome sentence rode along with it. **This is the one place ask 7 is adjacent**: ask 7's §3.6
   item is about withholding a *timing* and says a partial turn is withheld under
   `timeout`/`no_response` — that stays true verbatim, and it must not be merged with this edit,
   because the two sentences now give the same turn different answers to different questions.
4. **§4 S5's restatement (`:4978` at `1842b1d`) is the contradiction's third site and must be
   rewritten, not patched.** *"`no-response` is §3.6's `fail` and `server-rejected` is `-ml` §4.1's
   `unrunnable`"* becomes rule 4's five-row mapping; *"Only `turnDisposition == 'cap-hit'` is
   absent, not failed"* becomes conditioned on `|E(t)| ≥ 1`. The synthetic-trace set must carry the
   two discriminating cases, neither of which a four-disposition list produces by accident: **a
   cap-hit turn with an empty dispatch trace** (assert all three at once — it is `no_attempt`, it
   **does** enter `iteration_cap_hit_rate`, it **does** enter the `I(t)` summary), and **a
   timed-out turn beside a dropped-connection turn** (identical in every respect but the mechanism,
   asserting `fail` against `unrunnable`).
5. **The S5 scorer implements `cleanThroughTurnH`'s third state** — and, from v1.22, §4.3 rule 5's
   two other consumers with it, which land in the same scorer and are item 11. A conversation with an
   `unrunnable` turn at any `t ≤ H` is neither clean nor failed: out of the headline's denominator,
   into its own `n/a` tally, printed (rule 4, §4.6). No new gate is needed to keep the shrunken `n`
   honest — `verdict()` already refuses below `resolving.observable_floor` and the floor is computed
   from the surviving `n`, so a collapsed denominator refuses itself.
6. **§4 S1 and Appendix A — `iterations` is not an open choice, and P14-6's first half closes by
   consequence.** `iterations == len(chatResults)` is already forced: `-ml` §11.4 (v1.20) binds
   `callCount == len(ItemTiming.calls) == TurnTrace.iterations` and ask 7 builds `ItemTiming.calls`
   from `chatResults` in order. State it as a consequence with that citation rather than as a fresh
   decision, and add the value it forces: a turn whose first call raised records `iterations == 0`.
   **`0` is a real recorded value on the record and a non-observation in the statistic** — it stays
   on `TurnTrace`, and §4.2(f) is what keeps it out of the mean. Appendix A's `TurnDisposition` row
   moves with item 2.
7. **The report surface.** The `I(t)` summary prints its own denominator inline and the excluded
   count beside it (rules 1–2); the mean prints `>= v` whenever `c > 0` and the p95 `>= v` whenever
   `r > X − c`. The two iteration figures are **both** printed and never substituted for one
   another: `Y_calls / Y` (§11.4) is the unrestricted calls-per-item **cost** figure, the mean of
   `I(t)` the restricted **behaviour** figure, and on any run with a non-`replied` turn they differ.
8. **Where the two scoring constants live, and it is not `convo.py`.**
   `ITERATION_SUMMARY_DISPOSITIONS` and `ITERATION_SUMMARY_EXCLUDED` are a **scoring** vocabulary
   and belong beside the scorer; `TURN_DISPOSITIONS` is a **mechanism** vocabulary and stays in
   `convo`. The union assertion reaching across the two modules is the point, not an inconvenience —
   it is what makes the pin bind two independent declarations.
9. **§5 test 10c and the S5 scorer tests** carry §4.2(f)'s three pins verbatim: the cross-module
   union with disjointness; the **five** per-member behavioural cases, one distinct observable each;
   and the swept exactness rule in **both** directions. **The union assertion should land in
   P14-3's precursor unit**, on that finding's own corrected rationale — it buys nothing in round 1,
   where one author writes every artifact, and everything when the **rework** unit widens the enum.
10. **One sweep this ruling does not cause but sits beside.** §3.5's `index.csv` bullet still reads
    that `latencyMsMax` *"carries the tail figure for … every tool-caller run (`-ml` §11.3)"*. That
    is the substitution v1.20 corrected in three places — the tool-caller's `Y` is 80 turns, not 12
    conversations, so `latencyMsMax` is `None` on every declared pack — and §3.5 is a section ask 7
    already opens. Fix it in the same pass rather than leave the fourth instance standing.

11. **New at v1.22: §4.3 rule 5's plan-side consequences** *(plan-gate **P15-1**, raised under §7
    rule 3 and ruled here; the plan transcribes and does not re-derive)*.
    - **§3.8.4 and §4 S5 — the *"one remaining escape"* sentence is false as written**, and the
      correction is not a second clause bolted on: `cleanThroughTurnH`'s third state is **one of
      three** consumers of one predicate, and §4.3 rule 5 is the map. Replace the sentence, cite
      rule 5, and restate nothing of it. §4 S5's *Done when* gains **the hazard's censoring** and
      **§4.4's observed `n`** beside v1.27's item (3) — gated by name, not referenced in prose, for
      plan-gate P14-3's reason.
    - **What the scorer stores per position: three integers, `f_t`, `r_t`, `c_t` — never a rate.**
      The report divides. A stored rate whose base is not stored beside it is a denominator nobody
      can audit, which is rule 1 and §11.8's own argument for storing `X` and `Y`.
    - **§3.8.4, and this one is a gap rather than a sweep: what a failed turn contributes to the
      replayed history is specified nowhere.** Under `structured-replies-only` a turn with
      `finalReplyText is None` has no assistant message to replay. The plan must state that it
      contributes **nothing**, and that the harness **never** substitutes the script's `expect` for
      it. The repair is the more natural implementation and it is the worse one three ways over: it
      is hidden state §3.8.4 already forbids, it silently un-contaminates the trajectory rule 5
      censors, and it makes the pack's declared `historyReplay` false on exactly the conversations
      under study.
    - **§4.4's per-position `n`** is the observed count with the structural one printed beside it,
      and the `n < 10` marking is evaluated on the observed one.
    - **The funnel** (rule 3) gains one line under `unrunnable`: turns scored **after** an
      `unrunnable` turn in the same conversation. One integer, no gate, no new denominator.
    - **Tests — S5's, and gated by name in *Done when*.** **(a)** A 9-turn conversation with one
      `unrunnable` turn at `t = 2`, three others clean: it is in the risk set at `t = 1` and in
      **no** risk set at `t ≥ 2`; `c_2 == 1`; the hazard at `t ≥ 3` is over **3** conversations, not
      4. **(b)** The **discriminating pair**, which is what proves the two exclusions are two rules:
      at `H = 4`, an `unrunnable` at `t = 5` leaves the conversation **in** `cleanThroughTurn4`'s
      denominator and **out** of the hazard from `t = 5`, while one at `t = 3` takes it out of both.
      A suite built only from an early `unrunnable` turn passes on an implementation that collapses
      the two into one rule. **(c)** Both mutation directions, per `model-bench/AGENTS.md`: an
      implementation censoring the conversation from **every** position (including `1 … t−1`) must
      redden, and one censoring it from **none** must redden — the second is the behaviour P15-1
      found, the first is the natural over-correction. **(d)** `c_t == 0` renders the bound
      **absent** and the point estimate bare; `c_t > 0` renders it. Asserted on the absence, for
      §11.10 (7c)'s reason.

**Residuals, classified.** The five-member set, the two scoring constants, the cross-module union
assertion and the swept exactness rule are buildable now and are **not blocked**: they need
`TURN_DISPOSITIONS` and `stats.percentile`, which exist or land in the precursor unit. The five
per-member behavioural cases, the two discriminating traces, `cleanThroughTurnH`'s third state and
**all of item 11's** need the **S5 scorer**, which does not exist — **blocked on unbuilt work**, and
gated by being named
in S5's *Done when* list rather than referenced in its prose, which is plan-gate P14-3's own finding
applied to this ruling's own tests. Nothing here is deferred by choice. **The one non-test residual
is item 11's replayed-history gap**: it is a *plan* statement about an unbuilt behaviour, closeable
in the plan today and therefore not deferred either — if the rework unit reaches the replay before
the plan says this, it will choose the repair, because the repair is what makes the transcript look
right.

### 4.4 Per-turn-position reporting when turns are not independent

The non-independence is real but it is **not** a problem for the per-turn-position slice, and
saying why matters:

**At a fixed turn position `t`, each conversation contributes at most one observation.** Across
conversations, those observations *are* independent. So a per-turn-position rate is a clean
binomial over conversations, and a **Wilson interval over conversations (never over turns) is
meaningful and correct**. FR-9's per-position slicing is precisely the slicing that restores
independence. This is exactly what review §8.2's table did.

**Pooling across turns within a conversation is what breaks.** Any statistic that sums turns
(overall accuracy, "280 turns") has an effective sample size far below the turn count, because
turn 4's outcome is nearly determined by turn 3's. In §8.2's data the within-conversation
correlation was essentially 1.0 after onset — 121 post-onset turns produced zero independent
information. **A pooled per-turn CI at n=280 in that dataset was a fiction; the real n was 40.**

Two mandatory consequences:

1. **Never print a Wilson interval over a turn-pooled count.** Where a pooled rate is wanted,
   compute its CI by **cluster bootstrap resampling conversations** (which, under the settled 12×1
   design, *are* the scripts — §4.5), recomputing the pooled rate inside each resample. ~20 lines of
   stdlib. Print two numbers next to it: the **width inflation** (`bootstrap width ÷ naive Wilson
   width`, ≈2.6 in §8.2's data) and the **design effect**, which is that ratio **squared** (≈7
   there). *(v1.2 correction: v1.1 called the ratio itself the design effect. It is not — the Kish
   design effect is a variance ratio, and `n_eff = n_obs / DEFF`. §3.4 rule 5 carries the formula and
   the ρ=1 identity that tests it.)*
2. **The per-position table is the primitive and is always printed**, with per-position `n` =
   conversations. **Under the 12×1 design those `n`s are small and unequal by position**, and the
   report must print each one rather than a single header figure: positions 1–4 have n=12 (all three
   shapes), 5–7 have n=8 (shapes A and B), 8–9 have n=4 (shape A only). Computed Wilson widths at
   those `n` (z=1.959963984540054): `0/12 → [0.000, 0.242]`, `12/12 → [0.758, 1.000]`,
   `0/8 → [0.000, 0.324]`, `0/4 → [0.000, 0.490]`, `4/4 → [0.510, 1.000]`. **Mark every position with
   n < 10 `descriptive at this n — no significance claim`**; that is positions 5 onward, i.e. most of
   the deep-turn region.
   **The printed `n` is the *observed* count, and 12 / 8 / 4 are the design rather than the
   denominator** *(v1.22, plan-gate P15-1)*. A conversation censored by §4.3 rule 5 at `t' ≤ t` is
   not in position `t`'s `n`, so what the row carries is `structural(t) − censored(≤ t)` — with the
   structural figure printed beside it, because the gap between the two **is** the run's
   contamination made visible and is what a reader needs to judge the row at all. The widths above
   are computed at the structural `n`s and illustrate the design; the interval actually printed
   takes the observed `n`, and **the `n < 10` marking is evaluated on the observed one** — so
   censoring can push a position out of significance-claiming territory, which is correct and is
   exactly what printing 12 would hide.
   A deterministic collapse still shows (12/12 vs 0/12 at a position is
   unmistakable), but a 30-pp difference at turn 7 is not resolvable and the table must not look as
   though it were.

### 4.5 The clustering hazard, and the sampling design that answers it (settled 2026-09-02)

**The argument, unchanged from v1.1.** FR-22's conversation scripts are **fixed**. §8.2's design ran
15 replicates of the *same* condition-A script. That is **trial replication**, not item replication:
the resulting CI describes "if I run this one script again," not "if I wrote a different 9-turn
script." A tool that prints ±5 pp from 40 replicates of 3 scripts is reporting an interval that
could move 40 pp if someone wrote a fourth script.

**The decision (stakeholder, on that argument): 12 distinct conversation scripts — 4 per shape ×
the three A/B/C shapes — run once each at temperature 0.** `replicatesPerScript = 1`.
**n = 12 conversations per model per arm**, one observation per script.

#### 4.5.1 What 12×1 does to the clustering problem

**Conversation-level clustering dissolves — for conversation-level statistics only.** With one run
per script, each cluster contributes exactly one observation, so there is nothing to inflate the
variance: `DEFF = 1.00` **by construction, not by assumption**, `n_eff = 12`, the paired table for
`cleanThroughTurnH` has 12 rows, and **McNemar exact becomes a valid decision rule rather than an
anti-conservative one.** That is what the design bought. It did not buy precision — see §4.5.3.

*(The "12 rows" is 12 only while `H ≤ min(script length)` holds — §4.6. That is a validated pack
invariant, not an assumption of this section: violate it and the paired table silently loses the
conversations too short to reach turn `H`, which would make both `n_eff` and the resolving-power
line wrong in the optimistic direction. It is the only place `H` touches a denominator in this
note.)*

**Three things do not dissolve, and the implementer must handle all three.**

**(i) Turn-level dependence inside a conversation.** Every statistic that pools turns — all seven
FR-8 counts, every per-call count — still has ~7 positively-correlated observations per
conversation, for exactly the reason §4.4 gives. **The cluster bootstrap requirement survives, at
one level: resample the 12 conversations, recompute the pooled rate inside each resample** (§3.4
rule 6). The consequence worth printing: with 12 clusters, **the effective n of any turn-pooled
count is capped at 12** no matter how many turns feed it — 80 turns at ρ=1 is 12, not 80 — so a
turn-pooled count resolves ~50 pp at best and can never be a verdict metric. It isn't one (§4.6);
this is the arithmetic that says it never can be at this pack size.

**(ii) Shape-level correlation, which now has nowhere to go.** The 12 scripts are 4 per shape × 3
shapes. Scripts within a shape share tool set, length and task pattern, so they are *not*
exchangeable draws from "all conversation scripts" — a model that fails shape B's write-mutating
pattern plausibly fails all four B scripts. The tempting fix is a cluster bootstrap over shapes, and
it is wrong: **3 clusters is too few for any bootstrap** (the resample distribution is degenerate and
the resulting interval is noise). The honest handling, and the recommendation:

> Treat **shape as a fixed blocking factor, not a random cluster.** The inference is *conditional on
> these 12 fixed scripts* — which is exactly what FR-22's "fixed, versioned scripts" specifies —
> per-shape tables are always printed, and the report carries one line: `Inference is conditional on
> the 12 scripts in pack <id>@<version>; generalization to unwritten scripts is not certified by any
> interval in this report.` No number this tool prints, at any n, certifies that generalization; the
> only thing that would is more distinct scripts (§4.5.3's reversal trigger).

**(iii) Run-to-run variability becomes unmeasurable.** This is the one real loss. LM Studio at
temperature 0 is near-deterministic but not guaranteed bit-deterministic (batching and GPU
reduction order), and with one run per script the harness can no longer see the difference between
"this model is flaky" and "this script is hard". Cheap mitigation, and it should be built:
**a determinism probe — re-run 2 of the 12 scripts a second time, once per model, and report
whether the outcome vector is identical.** Two conversations of budget; **diagnostic, outside `n`,
never pooled into it.** If the probe comes back non-identical, every conversation-level statistic in
that run carries an unmeasured extra variance source and the report must say so in the same words.

Keep printing `temperature` and `replicatesPerScript` adjacent to every conversation-level `n`
(FR-18), now with `replicatesPerScript = 1` as the value that says the design effect is 1 by
construction rather than by hope.

#### 4.5.2 Cost, measured basis

12 scripts = 4×9 + 4×7 + 4×4 = **80 turns per model**. At the prior run's measured ~1.3 s/turn that
is ≈1.7 min per model, ≈3.5 min for both arms of a paired comparison. (The old 48-conversation
design was 320 turns ≈ 7 min per model, ≈14 min paired — v1.1's estimate, same basis.)

**Those minutes are a floor, not a central estimate** *(v1.20, plan-gate P13-7)*. The ~1.3 s/turn was
measured on falkor-chat's real multi-step executor, whose node declares `maxIterations: 8`
(`falkor-chat/server/falkorchat/proof_defs.py:415`), so plan §3.8.4's per-turn loop is **not a new
cost** and no sizing decision is reopened by it. What the figure will not carry is *typical*: it was
measured on `qwen/qwen3-4b-2507`, the model that stops calling tools at turn 4 in 39 of 40
conversations, so a large share of those turns ran a single iteration and the average is a mixture
rather than a per-turn cost. A model that calls tools on every turn runs more iterations per turn
and costs proportionally more; the multiplier is **bounded above by `maxIterationsPerTurn`**, which
puts the paired run at **≤ ≈28 min** at this pack's declared cap of 8. Plan against that bound, not
against the 3.5. **The typical case is unmeasured and this note will not put a number on it** — an
inference from §8.2's turn-position table is not a measurement of a run nobody has made — and it
costs nothing to obtain: §4.2(f) already reports the mean and p95 of `I(t)`, and §11.4's
`Y_calls / Y` is the **cost-side** reading of it off the latency block — unrestricted, over attempts
and over every item, which is the right base for a run-time estimate (a call that times out costs
its whole budget) and is deliberately not §4.2(f)'s restricted mean *(v1.22: "the same quantity"
overstated it, and §4.3.1 item 7 forbids substituting either for the other)*. **The sizing decision stands**:
§4.5.3's reversal trigger is denominated in *scripts*, and the binding constraint is FR-19 human
verification, not compute.

**A correction the stakeholder is owed: this is not the same run budget, it is one quarter of it.**
The authoring budget is unchanged — 12 human-verified scripts either way, which is the expensive
half — but the *inference* budget drops from ~320 to ~80 turns per model. What that freed budget can
and cannot buy is §4.5.3.

#### 4.5.3 The honest consequence, stated plainly

| | old design (12 scripts × 4 reps) | **settled design (12 × 1)** |
|---|---|---|
| nominal `n` | 48 conversations | **12 conversations** |
| honest `n_eff` at temperature 0, ρ≈1 within script | `48 / (1 + 3·1)` = **12** | **12** |
| observable floor (α=0.05) | claimed 12.5 pp / honest **50.0 pp** | **50.0 pp** |
| MDD₈₀ | claimed 16.0 pp / honest **57.8 pp** | **57.8 pp** |
| is McNemar valid? | **no** — anti-conservative over 48 correlated rows | **yes** |
| is the design effect measurable? | yes, from replicates | no — it is 1 by construction |

**So the tool did not lose resolving power. It lost a claim it could not support.** The honest
figures are identical because the old design's effective n *was* 12; all that changed is that the
report now says 12 where it used to say 48. (One v1.1 figure is corrected here: §7.2 put the fully
clustered MDD at "~65 pp", which came from the `8/n` rule of thumb. The exact value at n_eff=12,
recomputed this session, is **57.8 pp** — §3.4 rule 3. The direction of gate B-1 is unaffected; the
gap it names was 16.7 vs 57.8 pp, not 16.7 vs 65.)

**What this design genuinely cannot do, in the lab's own terms.** The effects this lab has cared
about split across the floor:
- the `qwen3-4b` turn-4 collapse (97.5% vs 0%) → at 12×1 that is b=12, c=0, McNemar exact
  p = 0.00049. **Comfortably detected.** The pack still does the job it was commissioned for.
- the ministral duplicate-instruction defect (~30 pp) → **below the 50.0 pp floor. Not resolvable,
  at any observed outcome.** It would have been *claimed* resolvable under the old nominal 48
  (MDD₈₀ 16.0 pp) and would not actually have been. The loss is of a false claim, but a reader who
  remembers the old sizing should be told the 15–50 pp band is dark.

***Reversal trigger, costed.*** The freed inference budget (36 conversations, ≈11 min paired)
converts into resolving power **only** by authoring **36 more distinct human-verified scripts** —
48 total, 16 per shape — which puts the floor at 12.5 pp and MDD₈₀ at 16.0 pp *honestly*. The
binding constraint is FR-19 human verification of scripts, not compute; the compute is already
paid for. **Trigger: the first tool-caller comparison that returns "not distinguishable" with an
observed difference in the 15–50 pp band.** That is precisely the band 48 distinct scripts would
resolve and 12 cannot, and it is the evidence that makes the authoring cost worth funding rather
than a hypothetical.

### 4.6 Aggregating across conditions of different length without a confounded headline

Do **not** pool turns across conditions — a 9-turn script contributes 2.25× the turns of a 4-turn
one and the headline becomes a weighted average of script lengths.

**Recommended headline: a survival statistic at a pack-declared turn depth.**

- Headline (`headlineMetric`): **`cleanThroughTurnH`** — the fraction of conversations with zero
  failure of any kind through turn `H`. **`H` is declared in the pack manifest
  (`metrics.cleanThroughTurnH.H`) and validated `H ≤ min(script length)` across all conditions in
  the pack**; it is *bounded by* the minimum, not *equal to* it. `H = 4` for the A/B/C set, whose
  minimum is also 4. *(v1.4: v1.1–v1.3 defined `H = min(script length)`. The plan's semantics are
  authoritative and are adopted here; the derived definition was the gate's M-11 — it let a future
  pack version that adds a 3-turn script silently redefine the headline from `cleanThroughTurn4` to
  `cleanThroughTurn3` under an unchanged name.)*

  **The `≤` is not a formality — it is exactly the precondition that makes the denominator
  honest.** Because every conversation in the pack is at least `H` turns long, every conversation
  of every condition contributes exactly one observation, so the statistic is length-independent by
  construction, is a proper binomial over the full 12 conversations, and is the statistic that
  would have caught `qwen3-4b` on the first run. **If `H > min(script length)` the denominator
  silently becomes selection-conditioned** — short scripts cannot reach turn `H`, so the headline
  quietly turns into a rate over long conversations only, which is §4.3's laundering failure in the
  one place the report calls its headline. That is the methodological reason `validate` must fail
  the pack rather than clamp `H`, and it must fail rather than warn.

  **What declaring `H` strictly below the minimum costs, since it is now allowed.** It is a
  legitimate choice — it keeps the headline's meaning stable across pack versions, which is the
  point of declaring it — but turns `H+1 … min` are then scored and excluded from the headline, so
  a failure at turn 5 does not count against `cleanThroughTurn4`. The metric gets coarser, not
  wrong. Two consequences, both cheap: **print `H` beside the metric name in every report** so
  `cleanThroughTurn4` is never read as `cleanThroughTurn7`, and treat any gap between `H` and
  `min(script length)` as discriminating information deliberately left on the table — worth a line
  in the report when the gap is non-zero, and worth nothing when it is zero (as it is today).
- Diagnostic, always printed: the **per-turn hazard** — `P(first failure at t | clean through
  t-1)`, over the **risk set at `t`**: the conversations that reached `t` clean **and whose turn
  `t` was observed**. A conversation leaves that risk set by either of two exits — its first
  `fail`, or an `unrunnable` turn, and the second removes it from `t` **onward** while keeping its
  observations at `1 … t−1` (§4.3 rule 5, which is where the reasoning lives and is not restated
  here). Hazard is what distinguishes
  "gradual degradation" (flat, low hazard) from "deterministic collapse at a fixed position"
  (hazard ≈ 0, 0, 0, 1.0 — §8.2's actual shape). A pooled accuracy number cannot tell those apart
  and the difference is the whole reason FR-9 exists.

  **Three numbers per position, not one** *(v1.22, plan-gate P15-1)*: the rate `f_t / r_t` inline
  per rule 1; the **censored count `c_t`** beside it, printed **even at zero** so the row's shape is
  constant (§11.7's reason, one report over) — `c_t` counting only conversations that **entered `t`
  clean**, since one that had already failed is out of the risk set and its later `unrunnable` turn
  censors nothing; and — **when and only when `c_t > 0`** — the
  two-sided imputation bound

  > `[ f_t / (r_t + c_t) , (f_t + c_t) / (r_t + c_t) ]`

  the hazard's value if every censored conversation had passed at `t` and if every one had failed.
  Two divisions, no resample, and it **collapses onto the point estimate at `c_t == 0`**, so a
  clean run pays nothing for it. It is printed because rule 5's censoring may be **informative**:
  the point estimate is the value under an assumption this design cannot support, and the bound is
  what is actually known. On §8.4's shape — 6 of 8 conversations lost to HTTP 400 — it is wide
  enough to be visibly uninformative, which is the honest rendering of that run and the one thing a
  bare flat curve cannot be. **The bound is per position and does not propagate**: a censored
  conversation's counterfactual at `t+1` needs a second assumption, and the per-position bound is
  already the honest statement at each point. **No gate, no interval, no verdict** — the hazard
  carries none of those (§4.4 rule 1 forbids a Wilson interval over turns, and refusing the curve
  below some coverage would delete the one picture FR-9 is for); §4.4's `n < 10` marking on the
  observed risk set is what says a position is thin.
- Per-condition tables are always printed underneath. A cross-condition headline other than
  `cleanThroughTurnH` is not reported at all.

---

## 5. Q3 — Embedder metrics, exact definitions

### 5.1 recall@k, MRR, precision@k

Preprocessing that is not optional: **L2-normalize every embedding before cosine**, for both
queries and documents, and record the distribution of raw `‖v‖` as a diagnostic (some LM Studio
embedding endpoints return unnormalized vectors; a silent scale difference would corrupt score
separation without touching ranking, so it must be visible).

- **recall@k** = `|top-k ∩ R| / |R|` — `metrics.recall_at_k` verbatim, semantics unchanged
  (standard, handles multi-relevant, correctly raises on empty `R`).
- **MRR** = reciprocal rank of the *first* relevant id — `metrics.mrr` verbatim.
- **precision@k** = `|top-k ∩ R| / k`.

**Finding on precision@k: on this golden set it carries no information.** Because
`precision@k = recall@k · |R| / k`, and `|R| = 1` for 36 of 38 items, precision@k is (to within
the two two-relevant items) a fixed rescaling of recall@k — an exact algebraic identity, not an
opinion. Its ceiling at k=10 is 0.10. Report it once for FR-12 compliance with the footnote
`precision@k = recall@k · |R|/k; |R|=1 for 36/38 items, so this is a rescaling of recall@k and
adds no discriminating information on this pack`, and treat **P@1** (= "is the top hit relevant")
as the informative member of the family. *Reversal trigger:* extend the golden set with genuinely
multi-relevant items (|R| ≥ 3) and precision@k becomes informative again.

**Binarizing `recall_at_k`/`precision_at_k` into a `BinaryMetric`'s per-item count (v1.24).**
`recall_at_k`/`precision_at_k` are continuous fractions — genuinely fractional only on the golden
set's 2 multi-relevant items (`|R| = 2`; §5.1 above). A per-item success flag is **"at least one
relevant doc retrieved in the top k"** (`recall_at_k(...) > 0`), not "every relevant doc retrieved"
(`== 1.0`). Rationale: (1) it matches §5.2's own precedent below, the one place this note already
collapses a continuous per-query quantity to a discrete success (`sep_raw(q) > 0 ⟺ P@1 = 1`); (2)
it is the standard IR reduction of recall@k to a binary outcome (Hit-Rate@k / Success@k); a
`== 1.0` threshold has no standard name and is strictly more conservative, i.e. a subset of `> 0`'s
successes; (3) it keeps recall@10 in its stated role (§7.4: harness sanity floor and regression
detector, never a comparison metric) from becoming more trigger-happy on exactly the 2 items where
the two readings can disagree. **Store the raw hit count** `|top-k ∩ R|` (an int in `[0, |R|]`) in
`counts[metric]`, not a pre-binarized 0/1 flag: `ItemResult.scored_outcome` already reads
`counts[metric] > 0` generically, so the aggregate `BinaryMetric` is identical either way, and the
raw count is free provenance a flattened flag would discard. **Honesty line, required alongside the
precision@k footnote above:** model-bench's reported `recall@10` is **not the same statistic** as
falkor-chat's `retrieval_baseline.json` / FR-12's quoted `0.974` — that figure is `sum(recall_at_k(
...) for each item) / n` (`falkor-chat/server/tests/eval/test_retrieval_eval.py:134`, a mean of the
continuous fraction), and no `BinaryMetric` binarization reproduces a continuous mean in general.
The two happen to agree on the pinned `37/38` baseline only because that particular run's two
multi-relevant items land at exactly 0 or 1 each (verified: `retrieval_baseline.json`'s
`recall_at_10 = 0.9736842105263158` is exactly `37/38`, and the 36 single-relevant items can only
contribute integers, so the 2 multi-relevant items' fractions must already sum to an integer here) —
a future run with a genuinely partial-credit multi-relevant item (one of two relevant docs found)
will diverge between the two statistics, and the report must not imply they are the same number.

### 5.2 Score separation

**Per query, over the full corpus (brute force gives every score anyway, so no top-k truncation):**

```
sep_raw(q)  = max_{d in R(q)} cos(q, d)  -  max_{d not in R(q)} cos(q, d)
```

Can be negative — that is the informative case (an irrelevant document outranks every relevant
one). Note the exact identity, worth printing once so nobody double-counts:
**`sep_raw(q) > 0` ⟺ the global top-1 document is relevant ⟺ P@1 = 1 for that query.** So the
*sign* is redundant with P@1; the **magnitude is the new information** — the margin by which the
ranking is right or wrong, which is what predicts whether a similarity threshold will hold up.

**Normalization is required, and this is not a judgement call.** Cosine scales genuinely differ
across embedding families — some compress almost everything into [0.6, 1.0] — so a raw gap of 0.05
from one model and 0.15 from another are not comparable quantities. Report both:

```
sep_z(q) = sep_raw(q) / sd({cos(q, d) : d in corpus})
```

**`sd` is the population standard deviation** (`statistics.pstdev`, v1.24 ruling). The 121-doc
corpus is enumerated in full and scored exactly for every query — it is not a sample drawn from
some larger, unobserved population whose spread is being estimated, which is the condition that
calls for Bessel's correction (sample stdev, `ddof=1`). This is a different statistical object from
§7.4's `sd_d` (the sample stdev of per-query MRR *differences*, legitimately estimating variability
for inference over the 38-query sample used to compute a margin of error) — no contradiction between
the two. At n=121 the two estimators differ by `sqrt(121/120) ≈ 1.004`, immaterial to any reported
figure, but the estimator is now pinned rather than left to `sd`'s ambiguity.

Per-query z-scoring against that query's own similarity distribution over the 121-doc corpus.
Scale- and offset-free, so it is the **cross-model comparable** number; `sep_raw` stays as the
within-model, product-actionable number (it is what you would set a threshold against).
*Rejected alternative:* per-query min-max normalization — driven by the single worst document in
the corpus, unstable at N=121.

**Aggregation:** per model report **median `sep_z`**, **10th percentile `sep_z`**, and **fraction
of queries with `sep_raw` > 0**. Median over mean because the distribution is skewed and a single
catastrophic query would dominate a mean; the p10 is the tail statistic that actually predicts
retrieval failures. Comparison across models uses the **paired bootstrap on per-query
`sep_z` differences** (§3.2d).

**`sep_raw`'s published figures, and the one comparison that is forbidden** *(v1.16, at plan v1.12's
ask — the plan gives `sep_raw` the same per-item carrier as `sep_z`, which is right, and asks which
of its figures print)*. Print **median `sep_raw`, 10th percentile `sep_raw`, and the fraction above
zero** — the same three shapes, for the same reason and one more: `sep_raw` is the number a threshold
is set against, and a threshold has to survive the bad queries, so the p10 is the figure that
decides it and the median alone would be the wrong summary to publish for its only product use.
**No mean, for either quantity** — which is the concrete reason `ContinuousMetric` is the wrong
aggregate carrier for both (§3.2d).

> **No difference between two models' `sep_raw` figures is printed, on any path.** `sep_raw` is
> scale-dependent per model — that is the entire reason `sep_z` exists — so a difference of two
> models' raw separations is a difference of two quantities measured on different scales. Giving
> `sep_raw` the same per-item carrier as `sep_z` makes such a difference **computable**, and
> therefore makes the prohibition worth writing down rather than leaving to the reader of §5.2's
> first paragraph. The same shape as §11.7 slot 6's latency rule: a per-item continuous quantity
> that is reported per arm and never differenced.

Worth noting for the pack's own documentation: with 12 topics over 121 messages, the "irrelevant"
pool contains same-topic near-misses — genuine hard negatives. That is a feature; the separation
number is meaningfully hard rather than trivially large.

### 5.3 The BM25 arm

**Construction, all of it versioned as pack data so the arm is reproducible without a network
download:**

- Tokenization: Unicode-aware `re.findall(r"\w+", text.casefold())`.
- **No stemming.** A Porter/Snowball stemmer means a dependency and a silent behaviour change
  between versions; declaring "no stemming" as pack config makes the arm reproducible. *Reversal
  trigger:* if BM25 beats the embedder on recall and the suspicion is morphology, add a stemmer
  **as a second, separately-named arm**, never by mutating the existing one.
- Stopwords: a **small English list committed in the pack** (not an `nltk` download).
- Parameters: `k1 = 1.2`, `b = 0.75` (standard Okapi defaults), both in pack config.
- IDF: use the always-positive variant `idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5))`. At N=121 a
  term appearing in >60 documents is entirely plausible, and the classic `ln((N-df+0.5)/(df+0.5))`
  goes negative there, which produces documents *penalized* for containing a query term. Use the
  `+1` form.

**Arm or reference line? Both, precisely stated.** BM25 is **deterministic** given (corpus version,
query set, parameters) — zero run-to-run variance — but it still has **sampling variance over the
query population**, so its recall/MRR on 38 queries legitimately carries a CI. Recommendation:
**report it as a full paired arm with its CI**, labelled `reference arm (deterministic given pack
version — re-running will not change it)`. It participates in the paired comparison normally
(its half of the pairing contributes no noise, which only tightens the interval). This satisfies
AC-5 and gives the "quality read against search-without-embeddings" the requirement asks for.

### 5.4 Is `retrieval_baseline.json` usable as a validation target?

**Not as a metric target. Yes as a bug detector and as an implementation cross-check.** The
reasoning, which the plan should carry so nobody re-litigates it:

The pinned numbers came from falkor-chat's `hybrid_search`: **approximate** in-graph ANN **plus**
full-text, over the 121-message corpus. The new harness is **exact brute-force, vector-only**. Two
differences pointing in **opposite** directions — exact ≥ ANN on recall (no approximation loss),
vector-only ≤ hybrid on recall (loses the keyword contribution). A disagreement of either sign is
therefore uninterpretable as a quality signal. Comparing them as if they were the same measurement
would be the classic pipeline-vs-model confound the whole tool exists to avoid.

Two narrower uses that are genuinely valuable:

1. **Sanity floor / bug detector.** Running the *same* model (`text-embedding-qwen3-embedding-0.6b`,
   dim 1024) on the *same* 121-doc corpus and the *same* 38 queries, exact brute-force recall@10
   should land at or above the ANN-based 0.974, and certainly not below ~0.85. A materially lower
   number is a harness defect — wrong prefix, unnormalized vectors, a truncated corpus — not a
   model finding. Print it as `harness self-check`, explicitly **not** a quality gate (the
   requirements rule out hard gates and are right to).

   **Settled 2026-09-02 (stakeholder decision 3), for an implementer reading only this note:
   the self-check is a diagnostic and never blocks.** If it lands below the ~0.85 reference,
   **S3 still completes** — the run is stored, the comparison renders, the exit code is unchanged —
   and the deviation plus its investigation are written into the test report. There is no
   configuration in which this number fails a build, fails a stage, or suppresses a result. The
   reason is §5.4's own argument: exact-vs-ANN and vector-only-vs-hybrid push in opposite
   directions, so a disagreement of either sign is uninterpretable as *quality*, and a number you
   cannot interpret must not hold a gate. Its job is to make a harness defect visible, and it does
   that by being printed and read, not by halting anything.
2. **Implementation cross-check on the pure metric functions.** Feed `test_metrics.py`'s existing
   fixtures through the new implementation and require byte-identical outputs — assertable here,
   unlike §3.2c's bounds, because recall@k and MRR are ratios of small integers and the values are
   exactly representable. Cheap, and it
   removes "did we reimplement recall@k subtly differently" as an explanation for any future
   divergence.

**Also copy `golden_retrieval.embeddings.json` (1.1 MB) as a fixture.** It contains the query
embeddings for that exact model, so the harness can run its ranking/cosine/metric path against
known vectors and isolate "is my ranking code right" from "is my embedding call right." That is a
one-time self-test with real diagnostic value at essentially zero cost.

**Provenance at the copy boundary (FR-6/FR-19).** Every copied file must gain
`copiedFrom` (repo-root path), `copiedAt`, and the **source git SHA**. Without those the copied
data has a provenance chain that dead-ends at the copy, and FR-6's comparability check silently
degrades to "same filename."

### 5.5 FR-14 prefixes — one correctness trap

Per-model `queryPrefix` / `docPrefix` in pack config, as FR-14 requires. The trap:
**the corpus must be re-embedded per model with that model's document prefix.** A cached corpus
embedding cannot be shared across models, and a cached one shared across *prefix settings of the
same model* is equally wrong. The cache key must include `(model id, quantization, docPrefix,
corpus version)` — all four. Getting this wrong produces a plausible-looking, entirely invalid
comparison, and it is exactly the kind of error that leaves no visible trace.

---

## 6. Q4 — The chat-responder role

### 6.1 The honest position

The role is **defensibly measurable at first delivery, but not as an open-ended "chat quality"
score.** As a judge-only score with the current 10-item calibration and a same-model judge it is
not defensible, and the committed artifacts prove it rather than merely suggesting it.

**Evidence, recomputed here from `judge_calibration.json`:**

| axis | raw agreement | Wilson95 | Cohen's κ |
|---|---|---|---|
| faithfulness | 9/10 = 0.90 | [0.596, 0.982] | **0.833** |
| relevance | 7/10 = 0.70 | [0.397, 0.892] | **0.211** |

Three things follow directly:

1. **The relevance axis is close to worthless.** κ = 0.21 is "slight" agreement. The committed file
   reports `"relevanceAgreement": 0.70`, which reads far better than it is — raw agreement is
   inflated by the skewed marginals (gold 7 relevant / 3 not; judge 8 / 2). Its Wilson CI includes
   0.5. Its class-conditional failure is the informative number: the judge called **2 of 3**
   genuinely-irrelevant answers relevant.
2. **The faithfulness axis is usable.** κ = 0.833 on a comparison-against-supplied-context task —
   near-extractive, which is what a small model can actually do.
3. **The 10-item calibration cannot support a gate on either axis.** Every interval above is
   ~40 points wide.

### 6.2 Recommended minimum defensible design

**Make chat-responder ~80% ground-truth-matched and only ~20% judge-mediated.**

**Golden set: 30 items.** Each item = (conversation prefix, user turn, retrieved context or
explicitly none), with a **checklist ground truth** rather than a reference answer:

```json
{
  "id": "cr-01",
  "context": ["…"],                    // may be [] for the abstention items
  "mustContain":    ["2000ms", "p99"], // normalized-containment, deterministic
  "mustNotContain": ["5000ms"],        // fabrications the context does not support
  "mustAbstain": false,                // true for the "not in context" items
  "provenance": {"corpusVersion": "…", "draftedBy": "…", "verifiedBy": "…", "date": "…"}
}
```

Deterministic scoring, no judge: all `mustContain` present **and** no `mustNotContain` present
**and** abstention matches `mustAbstain`. Reuse `nlq_scoring.layer2_contains`'s canonicalization
and `_ABSTENTION_MARKERS`. This converts the majority of "chat quality" into ground-truth
matching — the same move `nlq_scoring` already made for NL queries, and it is why that role has a
defensible number today.

*Why 30:* the McNemar floor means 6 net wins are needed regardless of n, so at n=30 the tool
resolves 20 pp observed / 25 pp at 80% power. At n=20 that becomes 30.0/36.7 pp — too coarse to be
worth building. 30 is the smallest n where the role can distinguish anything the lab has cared
about, against a human-verification cost of 30 items.

*Drafted how:* LLM-drafted from the **copied 121-message corpus** (real, already
provenance-tracked, and the retrieval contexts are then genuine), **every item human-verified**
per FR-19. Questions must be paraphrases, never verbatim.

**Judge component: faithfulness only.** Reuse `judge.py`'s faithfulness axis — including its
conservative parse handling and its unconditional `faithfulness=None` when context is empty, both
of which are correct and hard-won. **Drop the relevance axis entirely** and let the deterministic
`mustContain`/`mustAbstain` checks measure what relevance was trying to measure. κ = 0.21 is the
justification; this is not a preference.

**Which model judges — enforce it in code.**

- `judgeModel` is pinned in pack config and recorded in the fingerprint.
- **The harness hard-errors when `judgeModel == candidateModel`.** Cheap, enforceable, and the
  committed `"sameModelAsAgentUnderTest": true` shows the collision happens by default rather than
  by accident.
- If no non-candidate judge is loadable in a given session, the run still proceeds but sets
  `judgeIsCandidate: true` and **every judge-mediated number is suppressed from the comparison** and
  printed as diagnostic-only.
- Practically on this box: judge from a different family than the candidate (qwen candidate →
  ministral judge, or vice versa). `gpt-oss-20b` is not a judge option until its LM Studio
  crash is resolved (review §8.4).

**Two distinct self-preference caveats, never one blanket one.** The judge scoring **fixed,
human-authored calibration items** carries little self-preference risk — that pass is a legitimate
rubric-following signal. The judge scoring the **candidate's own live replies** carries it fully.
One undifferentiated caveat would let a reader extend trust from the first to the second. The
report must carry them separately, worded to the sub-pass.

**Calibration: extend to 40 items, and gate on class-conditional rates, not κ.**

- 10 → **40 items**, deliberately **balanced**: ~20 gold-faithful, ~20 gold-unfaithful, plus the
  empty-context abstention cases. Balance is required so class-conditional rates are estimable at
  all; the current set has only 2 gold-`False` faithfulness items.
- `judge.py` is **conservative by design** (its own docstring: prefer `None` over a guessed `True`).
  A deliberately-biased judge is mis-gated by any symmetric statistic: specificity sits near its
  ceiling by construction, which decouples κ from the error class that matters, and κ additionally
  moves with the hand-picked case mix. **Gate on the class-conditional rates:**
  - **`false-pass rate` = P(judge says faithful | gold unfaithful)** — the error that actually
    corrupts a score.
  - **`unfaithful-recall` = 1 − false-pass rate.**
  - **`parse-failure rate`** (`judge.py` already tracks it and must never be dropped from the
    denominator).
- κ and raw agreement are **reported as diagnostics with their marginals**, never as the gate.
- **Threshold (computed here, at 20 gold-unfaithful items):** judge usable if **false-passes ≤ 2/20**
  (rate 0.10, Wilson95 [0.028, 0.301]) **and** parse-failure rate ≤ 0.05. At 3/20 the Wilson upper
  bound reaches 0.36 — a judge that could be wrong on a third of the cases that matter is not a
  measurement instrument. *Reversal trigger:* if no available local judge clears 2/20, the
  faithfulness component is dropped and chat-responder ships deterministic-checks-only.

**Report language.** Judge-mediated numbers live in their own table, **below** the deterministic
ones, never summed into a headline, each carrying:

> `judge-mediated — not ground-truth-matched. Judge <model>, calibrated <date> on <n> items:
> false-pass rate k/n [lo, hi], parse-failure rate k/n. Not comparable in strength to the
> deterministic counts above.`

### 6.3 The flag to carry to the stakeholder

**Not "this role is unmeasurable" — a costed conditional:**

> Chat-responder at first delivery costs **one new 30-item golden set** plus **30 new
> judge-calibration items** (10 → 40, balanced), all human-verified per FR-19 — roughly the same
> drafting effort as FR-22's conversation scripts, which the requirements already accept as the
> main new golden-data cost. If that is funded, the role ships with a defensible, mostly
> ground-truth-matched number. **If it is not funded, chat-responder should ship
> deterministic-checks-only (no judge, no faithfulness axis) or be deferred — it must not ship as a
> judge-only score**, because the only calibration evidence the lab has says the judge's
> open-ended axis agrees with human labels at κ = 0.21.

---

## 7. Q5 — Sample sizes and the honest minimum detectable difference

### 7.1 The two numbers every report must print

The requirements' "roughly 15 percentage points" is the **observable floor at n=40** — the smallest
difference that *could* be called significant if it landed perfectly. It is not the difference the
tool reliably *detects*. Both numbers follow from the McNemar exact floor and were computed here:

- **Observable floor = `b_min(α_family)/n_eff` = `6/n_eff`, computed at the *unadjusted* α=0.05 —
  never at `α/k`** (Rule 3's α row; review M-ML-6). Under Holm a member's threshold is
  data-dependent, so no single α makes the floor's sentence true by construction and one must be
  chosen: *"below this, nothing can reach significance"* is true only at the **loosest step a member
  can face**, which is the unadjusted α. Printed at `α/k` it is `7/n_eff` and is **false** — at
  n=40, `b=6, c=0` gives p=0.031, which a rank-2 member *does* clear at its 0.05 Holm step, and
  15.0 pp sits below the printed 17.5. Same falsity class as the `15.8` withdrawn in v1.5.
  Floor-at-0.05 is additionally conservative for every member, and it is what makes Rule 7's
  McNemar branch a theorem.
- **MDD₈₀ — computed exactly, never from a rule of thumb** (§3.4 rule 3). The `n·δ ≈ 7.7` mnemonic
  holds only over n≈20–120 (recomputed: 7.33 at n=20 → 7.81 at n=120) and **breaks below it —
  6.94 at n=12** — which is exactly the range the tool-caller pack now lives in.

**All figures recomputed in this session** by exact search over the McNemar rejection region.
**MDD columns are ceilinged to 0.1 pp; the floor column is truncated** — opposite directions, for
the reason in §3.4 Rule 3. The exact MDD values are shown so nobody re-derives them; the floor is
exactly `6/n_eff`, truncated with the caller's own expression (Rule 3a's bin-edge guard):

| n_eff | **observable floor** (α=0.05, *every* k) | MDD₈₀ at α=0.05 (k=1) | (exact) | MDD₈₀ at α=0.025 (k=2) |
|---|---|---|---|---|
| **12** | **50.0 pp** | **57.8 pp** | 57.794 | 65.6 pp |
| 15 | 40.0 pp | 47.6 pp | 47.559 | 54.2 pp |
| 20 | 30.0 pp | 36.7 pp | 36.646 | — |
| 30 | 20.0 pp | 25.1 pp | 25.075 | 28.7 pp |
| **34** | **17.6 pp** | **22.3 pp** | 22.258 | — |
| 38 | 15.7 pp | 20.1 pp | 20.009 | — |
| 40 | 15.0 pp | 19.1 pp | 19.046 | 21.9 pp |
| 48 | 12.5 pp | 16.0 pp | 15.972 | — |
| 60 | 10.0 pp | 12.9 pp | 12.857 | — |
| 85 | 7.0 pp | 9.2 pp | 9.142 | — |
| 120 | 5.0 pp | 6.6 pp | 6.509 | — |

**There is one floor column and there always will be**: the floor does not move with *k*, because
its α is the unadjusted one whatever the family size. Only the MDD pays for multiplicity. *(v1.6
deletes v1.2–v1.5's second floor column — 58.3 / 46.6 / 23.3 / 17.5 pp at α=0.025 — which asserted
impossibilities that are attainable.)*

**Every report prints, computed from its own `n_effective` and its own α, never hardcoded:**

> `This pack resolves differences of >=X pp with 80% power at n=N effective <unit>s (<U> units,
> design effect D, <basis>, alpha=A_mdd). Differences below Y pp cannot reach significance at any
> observed outcome, at any Holm step (alpha <= A_family).`

Every field is mandatory, **including both αs** — they differ whenever k > 1, and a reader shown
only one cannot tell which bound it governs. The unit, the raw unit count, the design effect and its
basis are what make the line auditable — a bare `n` is the shape gate B-1 rejected, because 48 turns, 48
conversations and 12 scripts print the same sentence and mean three different things.

**The floor sentence takes an effective-unit qualifier whenever `design_effect > 1.0` (v1.8, review
m-ML-11).** The floor is `b_min/n_eff` while the McNemar p printed beside it is computed over the
`n_units` raw rows, so above DEFF 1 the two quantities live on different denominators and the
sentence as templated above reads as a flat contradiction of the number three lines from it.
Measured: at `n_units = 12, DEFF = 2` the floor prints **100.0 pp** while `b=11, c=0` is a 91.7 pp
difference at `p = 0.00098`; at `n_units = 40, DEFF = 2` the floor prints **30.0 pp** beside
`b=6, c=0` at `p = 0.031`. Both are decided correctly — Rule 7 demotes them — and both *print* a
p below α underneath a sentence saying no observed outcome can reach significance. Where
`design_effect > 1.0`, render the floor sentence as:

> `differences below 30.0 pp cannot reach significance at any observed outcome over the 20 effective items this design supports, at any Holm step (alpha <= 0.05) — the McNemar p printed beside it is computed over the 40 raw rows, where clustering makes it anti-conservative, so it can fall below that alpha without lifting the difference above this floor`

At `design_effect == 1.00` the two denominators coincide, the qualifier is noise, and the template
above stands unchanged — which is why §7.2's rendered line does not move.

**The two sentences are independent, and the second prints whatever happens to the first (v1.7).**
When no effect size attains the power the MDD sentence is replaced by the *unattainable* wording
(review M-ML-1), and it is tempting to let that one replacement speak for both bounds — *"no
difference is resolvable, so nothing can reach significance."* **That combined sentence is false in
a reachable corner**, and it is reachable for the same reason the two bounds take different αs: with
`k >= 2` and `n_eff ∈ [6, 7)`, the rejection region at `alpha_mdd = 0.025` is empty (`b_min = 7`
exceeds the 6 floored units), so `mdd80` is `None` — while at `alpha_family = 0.05` the floor is
`6/n_eff ≤ 100 pp` and is *attained*. Worked, at `n_units = 13, DEFF = 2, n_eff = 6.5`: the floor
prints 92.3 pp, and the attainable outcome `b=12, c=0` is a 92.3 pp difference at
`p = 2·2⁻¹² = 0.00049`, which clears a rank-2 member's 0.05 Holm step. So the unattainable clause
speaks **only for power**, the floor sentence prints **either way**, and where the floor itself
exceeds 100 pp it says so in words rather than quoting an unreachable threshold.

**One assumption behind every MDD above, and it is optimistic.** The power model is **strict
dominance** (`π_c = 0`): every item on which the models differ favours the candidate. That is the
most favourable case, so these are *lower bounds* on the difference actually needed. Measured
sensitivity, computed this session at α=0.05:

| discordance mix | n=12 | n=30 | n=40 | n=48 |
|---|---|---|---|---|
| strictly dominant (`π_c = 0`) | 57.8 pp | 25.1 pp | 19.1 pp | 16.0 pp |
| candidate wins 4:1 (`π_c = 0.25·π_b`) | unattainable (max power 0.56) | 45.2 pp | 34.0 pp | 28.4 pp |
| candidate wins 2:1 (`π_c = 0.5·π_b`) | unattainable (max power 0.18) | unattainable (0.43) | unattainable (0.53) | unattainable (0.58) |

"Unattainable" means 80% power is not reached **at any effect size** under that mix. So a candidate
that is genuinely better but *also* loses some items — the normal case for a model swap — needs
substantially more than the headline MDD, and at n=12 essentially needs to win every discordant
conversation. **The report prints the strict-dominance MDD** (one number, comparable across packs)
**labelled `best case — assumes the candidate wins every item the models differ on`**, and any pack
whose `n_eff < 20` additionally prints the power ceiling from the 2:1 row, because that is where the
label stops being a caveat and starts being the finding.

Reassurance, still true and now precise: the effects this lab has needed to resolve — the qwen3-4b
turn-4 collapse (97.5% vs 0%) — are 90–100 pp and clear every floor in the table including n=12's.
The ~30 pp ministral duplicate-instruction defect does **not** clear the tool-caller pack's 50.0 pp
floor (§4.5.3). The tool is fit for catching collapses and honest about not resolving anything in
the 15–50 pp band at this sample size.

### 7.2 Per role

| Role | n | Unit of n | Instrument | Observable floor | MDD₈₀ | Existing golden data adequate? |
|---|---|---|---|---|---|---|
| **tool-caller** | **12** (3 shapes × 4 distinct scripts × **1 run**) | **conversation ≡ script**, one observation per cluster, **DEFF 1.00 by construction** | McNemar exact on `cleanThroughTurn4` — **valid at this design**, plus one-level cluster bootstrap over the 12 conversations for any turn-pooled count | **50.0 pp** | **57.8 pp** | **No — must be built (FR-22/FR-22a).** 12 distinct human-verified scripts; §4.5 is the sizing and §4.5.3 the honest consequence. |
| **guard-judge** | 85 total, but the decision is **class-conditional** and the family has **two verdict metrics** (MDD at α=0.025; floor at the unadjusted α=0.05) | item | McNemar per class, Holm across the two | see below | see below | **Partly** — see §7.3 |
| **nlq-generator** | **34** (40 items less the 6 structurally unanswerable — §3, plan §3.8.3; `Y = 40` still governs latency coverage) | item | McNemar on Layer-1 exact match | **17.6 pp** | **22.3 pp** | **Yes, marginally.** Answers "clearly better" only. *(v1.25: n corrected from the raw item count to the scoring denominator. The verdict is unchanged — both reference effects still clear 22.3 pp — and no power is lost, because the 6 excluded items are concordant-incorrect pairs that never entered `b` or `c`.)* |
| **chat-responder** | 30 (new) | item | McNemar on checklist pass | 20.0 pp | 25.1 pp | **No — does not exist** (§6.2) |
| **embedder** | 38 queries | query | paired bootstrap on per-query MRR | n/a (continuous) | see §7.4 | **Yes for MRR; no for recall@k** — see §7.4 |

*(`chat-responder`'s row describes the design to build when FR-21a's judged layer is funded; it does
not ship in first delivery.)*

**The tool-caller pack's resolving-power line, verbatim — this is what §7.1's template renders to
under the settled design, and the string an implementer should test against:**

> `This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations
> (12 units, design effect 1.00, by-construction, alpha=0.05). Differences below 50.0 pp cannot
> reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case —
> assumes the candidate wins every conversation the models differ on; if it loses one for every
> two it wins, 80% power is not reached at any effect size at this n. Inference is conditional on
> the 12 scripts in <packId>@<packVersion>; generalization to unwritten scripts is not certified by
> any interval in this report.`

*(v1.7: the floor sentence gains `at any Holm step (alpha <= 0.05)`. §7.1's template mandated it
from v1.6 and this rendering was left on the pre-v1.6 wording — a stale example presenting itself as
"the string an implementer should test against" is worse than no example, so **§7.1's template is
authoritative wherever the two ever diverge again.** This pack is k=1, so both αs are 0.05 and only
the wording moves.)*

Four sentences, all mandatory, none derivable from a bare `n` — which is the whole of gate B-1's
fix. Compare what the plan's `min_detectable_difference(48)` would have printed: **16.7 pp**, a
number 3.5× more optimistic than the design supports, with no unit, no design effect and no
conditionality.

### 7.3 Guard-judge — n=85 is misleading, and this pack has two verdict metrics and no headline

The decision-relevant statistic is class-conditional (a bias-to-suspend judge is mis-gated by
pooled accuracy for the same reason §6.2's judge is), and the class slices are small.

**Stakeholder decision 2, encoded:** `guard-judge` has **no `headlineMetric`**. Both
class-conditional error rates are co-equal members of `verdictMetrics`, reported with equal weight,
in a fixed declared order, with no summary number above them and no arithmetic combining them.
**Nothing in this note depends on that pack having a single verdict-carrying metric** — v1.1's §7.3
called false-advance "the primary" and §3.3's table named it; both are corrected, and no formula,
denominator or threshold anywhere in the note ever took the retired singular `primaryMetric` as
input.

**As built (plan v1.3), and nothing here conflicts with it:**
`verdictMetrics = ["falseAdvanceRate", "falseSuspendRate"]`, `headlineMetric = null`, bare metric
names with **no `@slice` suffix** — the slice is the metric's denominator, stated once in the table
below, not part of its identity — and `advanceRecall` **printed as a labelled complement that
carries no verdict**.

**Naming, settled (gate nit).** Report **both verdict metrics as error rates in the same
direction**, so neither reads as "better is higher" beside one that reads "better is lower":

- **`falseAdvanceRate`** = P(judge advances | gold says suspend), on the 40 `clear_suspend` items.
- **`falseSuspendRate`** = P(judge suspends | gold says advance), on the 30 `clear_advance` items.

`advance-recall` (v1.1's name for the second) is the **complement**: `advanceRecall = 1 −
falseSuspendRate`. Same quantity, and it stays printed as the complement so a reader looking for
recall finds it — but it **carries no verdict**, because a metric and its own complement are one
test, not two, and counting both would inflate *k* against a difference that is by definition
identical. The verdict is rendered on the error rate, because two co-equal verdict metrics pointing
the same way is the only presentation in which "worse on one, better on the other" is readable at a
glance.

**The MDD is recomputed at α=0.025** — the tightest Holm step a two-member family can be required to
clear (§3.3) — with the k=1 figure alongside so the price of the second verdict metric is visible.
**The floor stays at the unadjusted α=0.05** and is therefore the same number it would be for a
one-metric pack:

| slice | n | verdict metric | observable floor (α=0.05) | MDD₈₀ @α=0.025 | (MDD₈₀ @α=0.05, for contrast) |
|---|---|---|---|---|---|
| `clear_suspend` | 40 | **`falseAdvanceRate`** | **15.0 pp** | **21.9 pp** | 19.1 pp |
| `clear_advance` | 30 | **`falseSuspendRate`** | **20.0 pp** | **28.7 pp** | 25.1 pp |
| `boundary` (all expected-suspend) | 15 | false-advance on near-misses — **not a verdict metric** | 40.0 pp | — | 47.6 pp |

*(v1.6: the floor column was `17.5 / 23.3 / —` at α=0.025 in v1.2–v1.5 and is now `15.0 / 20.0 /
40.0` at the unadjusted α — review M-ML-6. All three are exact `6/n`, so neither the truncation rule
nor Rule 3a's bin-edge guard has anything to do here; `17.5` at α=0.025 *was* the cell where the
guard mattered, and moving the floor to α=0.05 retired that hazard rather than the guard. MDDs are
ceilinged, the floor truncated — §3.4 Rule 3.)*

So: **the naive read "n=85 → ~9 pp" is wrong**, and the honest figures are now 21.9 / 28.7 pp
rather than v1.1's 19.1 / 25.1 — the ~2.8–3.6 pp of resolving power that a second co-equal verdict
metric costs. (v1.1's boundary-tier MDD of "53 pp" was the `8/n` rule of thumb; the exact value at n=15 is
**47.6 pp**, recomputed here.) The `boundary` tier is **descriptive only** at n=15 and must be
printed with `no significance claim` — it is also the tier where disagreement is legitimate by
construction, so pooling it into an overall accuracy would let a model look better or worse for the
wrong reason. Report the three tiers separately, always; never a pooled 85-item accuracy as a
headline. Pooled accuracy and κ stay as diagnostics with marginals.

### 7.4 Embedder — the ceiling finding

**recall@k on the current 38-item set has zero power to certify a better model.** Computed here:

- recall@10 = 37/38. Only **1** item is available to win. McNemar needs **6**. It can **never**
  fire in the "candidate is better" direction, at any effect size.
- recall@5 = 34/38. Only **4** items available. Same conclusion.
- Marginal Wilson intervals do not separate even against a perfect 38/38 (Wilson [0.908, 1.000] vs
  [0.865, 0.995]).

The asymmetry is worth stating plainly because it is decision-relevant: **on recall, this set can
detect a materially *worse* embedder but cannot certify a *better* one.** Report recall@10 as the
harness sanity floor (§5.4) and as a regression detector, not as a comparison metric.

**MRR is the discriminating metric — the requirement already said so and the data confirm it.**
0.6259 leaves 0.374 of headroom. Its MDD is `1.96 · sd_d / √n` where `sd_d` is the standard
deviation of per-query MRR differences, which is **data-dependent and must be computed and printed
from the actual run**, not assumed:

| sd of per-query MRR difference | MDD at n=38 | MDD at n=60 |
|---|---|---|
| 0.10 | 0.032 | 0.025 |
| 0.20 | 0.064 | 0.051 |
| 0.30 | 0.095 | 0.076 |
| 0.40 | 0.127 | 0.101 |

**Verdict: 38 queries are adequate for MRR and for score separation; inadequate for recall@k
(saturated) and uninformative for precision@k (algebraically redundant, §5.1).** The cheapest
improvement, if the embedder role becomes decision-critical, is **~22 additional harder queries
(38 → 60), several with |R| ≥ 3** — which simultaneously de-saturates recall@k, makes precision@k
informative, and tightens the MRR interval by ~20%. Name it as a follow-up, not a first-delivery
requirement.

---

## 8. Places where the requirements are methodologically unbuildable as written

Three, in severity order. All three route to `tico` as requirement amendments.

**8.1 (blocker) — FR-15 / AC-4's marginal-overlap rule.** "Two models are declared different only
when their intervals don't overlap," read as two marginal Wilson intervals, **cannot fire at all**
at n ≤ 40 whenever the baseline is ≥ 0.90 — the regime this lab actually operates in (retrieval
recall@10 = 0.974; ministral 0/176). §3.1's worked case: 40/40 vs 34/40, perfectly nested, McNemar
p = 0.031, and the literal rule prints "not distinguishable." It also silently discards the paired
design FR-16 mandates and pays for. **Amendment:** FR-15/AC-4 should read *"the 95% confidence
interval on the **paired difference** must exclude zero"*. Same intent, correctly stated. §3.2's
verdict strings satisfy AC-4's wording under that reading.

**8.2 (major) — FR-8's "argument correctness, split into omitted / wrong value / boundary-unit."**
These are not three disjoint categories: boundary/unit translation is a **subset** of wrong value.
An implementer coding them as siblings will either double-count or arbitrarily prioritize, and two
runs will not be comparable. **Amendment:** state the nesting explicitly, and require each expected
argument in a pack to carry a `boundaryRule` so the classification is data-driven rather than a
regex nobody calibrated.

**8.3 (major) — FR-9/FR-22 do not name the clustering/replication problem.** Fixed scripts plus
replicates means the reported CI describes "this script run again," not "a script of this kind."
§8.2's own precedent (15 replicates of one script per condition) would produce an interval that
could move tens of points if a fourth script were written, with nothing in the report signalling it.
**Amendment:** FR-22 should require **≥3 distinct scripts per conversation shape**, and FR-15
should require the CI on any conversation-level statistic to come from a **cluster bootstrap over
scripts**, with the design effect printed.

*(Status at v1.2: **accepted and superseded by something stronger.** FR-22a carries the distinct-
scripts requirement, and the stakeholder's 12×1 sampling decision removes replication entirely, so
conversation-level statistics need no bootstrap at all — the design effect is 1 by construction
(§4.5.1). The bootstrap requirement survives only for **turn-pooled** counts, one level, over the
12 conversations. FR-15 does not need amending for it; §3.4 rule 4 enforces it in the code, which is
the more durable place.)*

**Two smaller items, not blockers, worth folding into the plan rather than amending:**

- **FR-12's precision@k** is algebraically redundant with recall@k on the copied golden set
  (§5.1). Keep it for compliance, footnote it.
- **FR-8 has no "restraint" count** — no measure of *not* calling a tool when none was required
  (§4.2). Add it; it is one line and without it a trigger-happy model scores perfectly.

---

## 9. Evaluation design — how to prove the harness itself is right

The harness is an instrument; an uncalibrated instrument produces confident wrong numbers. Eight
checks, all cheap, all implementable as unit/integration tests. Checks 1–4 are the statistics
module's own contract (§3.4) and are the ones that keep gate B-1 from recurring:

1. **Statistics module regression fixtures.** The five worked (a,b,c,d) rows in §3.2c, with their
   expected McNemar p and MOVER-D bounds, plus the McNemar floor table at both alphas
   (α=0.05: c=0→b=6, c=1→b=8, c=2→b=10, c=3→b=12, c=4→b=13; **α=0.025: c=0→b=7, c=1→b=9,
   c=2→b=11**). Pure functions, no network. *Threshold:* **1e-12 on p, 1e-9 on MOVER-D bounds as
   proportions** (§3.2c) — not "exactly", which is not assertable, and not 3 decimal places, which
   is loose enough to pass a wrong Wilson. **Assert against §3.2c's 10-dp table, which is the only
   published form of these fixtures precise enough to carry that tolerance** (§3.2c's trap); assert
   the p-values against the exact rationals, not their decimal expansions.
2. **The four contract tests that keep gate B-1 fixed** (§3.4), each of which must fail loudly
   rather than degrade: (a) `PairedOutcomes.from_units` raises `DuplicateAnalysisUnit` on a repeated
   unit id — feed it 48 rows drawn from 12 script ids, which is the exact shape the gate rejected;
   (b) `resolving_power` cannot be called without `design_effect` and `basis` (a `TypeError`, i.e.
   the absence of a default is itself the test); (c) `verdict` raises when
   `resolving.n_units != outcomes.n_units`, when the unit kinds differ, or when
   `alpha_mdd != alpha_family/len(family)`, or when `alpha_step` falls outside
   `[alpha_mdd, alpha_family]` (Rule 4's fifth precondition); (d) `verdict` refuses to let McNemar
   decide when `design_effect > 1.0` and renders the bootstrap decision instead.
3. **The MDD computation itself**, three assertions: it reproduces §7.1's exact column
   (19.046 pp at n_eff=40, 57.794 at 12, 47.559 at 15); it **ceilings** to the printed precision, so
   n=40 prints 19.1 and never 19.0 (measured power at 19.0 pp is 0.798 — the test is that the
   printed number's power is ≥ 0.80); and it is not a constant, not `8/n`, and rejects an `int`
   observation count passed where `n_effective` belongs.
4. **The design-effect identity** (§3.4 rule 5): for a synthetic cluster set with ρ=1 and m
   observations per cluster, `effective_n` must equal the **cluster count**, and `design_effect`
   must equal the squared width ratio. This is the one test that catches the squaring error in
   either direction; v1.1's own wording would have failed it.
5. **Metric cross-check against falkor-chat.** Run `test_metrics.py`'s existing fixtures through
   the copied recall@k/MRR. *Threshold:* byte-identical outputs.
6. **Embedding-path self-test.** Rank the 38 golden queries using the copied
   `golden_retrieval.embeddings.json` vectors and the copied corpus. *Reference line, not a
   threshold:* recall@10 is compared against ~0.85 and **printed either way** — below it, the run
   still completes and the deviation is investigated and written into the test report (§5.4,
   stakeholder decision 3). It is a bug detector; it fails no stage.
7. **Prose-call detector calibration.** ~20 human-labelled replies; report the detector's own
   precision/recall in every tool-caller run. *Threshold:* the numbers are printed; no pass/fail
   gate, because the requirements rule out hard gates and §4.2's use is diagnostic.
8. **Judge calibration gate** (chat-responder only, §6.2): false-pass rate ≤ 2/20 on gold-unfaithful
   items **and** parse-failure rate ≤ 0.05, else judge-mediated numbers are suppressed.

**A negative control worth the twenty minutes:** run the paired comparison with **the same model in
both arms — two independent runs, not two copies of one run** (two copies give b = c = 0 by
construction and cannot fail). The correct output is "not distinguishable" with b ≈ c and a
difference CI centred on zero. Anything else means the pairing, the session handling, or the RNG
seeding is wrong. It catches an entire class of harness bugs that would otherwise present as
plausible model differences.

**What the 12×1 design does to that control, and it is worth knowing before it is misread.** At
temperature 0 with one run per script, two independent runs of the same model are *expected* to
produce **b = c = 0** — not because the statistics work, but because the model is deterministic. So
on the tool-caller pack the negative control degenerates into the determinism probe of §4.5.1(iii),
and it exercises the pairing and session handling but **not** the decision arithmetic. Two
consequences: (1) the control still earns its place — `b ≠ c` here is a genuine finding, either a
harness bug or non-determinism at temperature 0, and both need to be known; (2) the decision
arithmetic must be exercised by the §9.1 fixtures and the synthetic traces instead, never by
"the negative control passed".

---

## 10. Risks and open questions

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Script-level clustering treated as independent replication → confidently narrow, wrong CIs | **high** *(largely retired by the 12×1 design; the residual is R1a)* | §4.5: one run per distinct script, so DEFF = 1 by construction; §3.4 rules 1–4 make the correlated-rows version unconstructible in code |
| R1a | The *residual* of R1: a turn-pooled count, or a future pack that buys replicates, silently re-inherits the old defect | **high** | §3.4 rule 6: one-level cluster bootstrap over conversations for turn-pooled counts, and `validate` **fails** any pack declaring `replicatesPerScript > 1` while only the one-level resample exists |
| R1b | Shape-level correlation (4 scripts per shape, 3 shapes) read as generalizing to unwritten scripts — 3 clusters is too few to bootstrap | medium | §4.5.1(ii): shape is a **fixed blocking factor**, per-shape tables always printed, and one mandatory report line saying the inference is conditional on the 12 fixed scripts |
| R2 | Conditional-denominator laundering — a model that collapses early scores *better* on (c)–(g) | **high** | §4.3: mandatory funnel table, `k/n` on every rate, printed `n/a` tallies, paired-n intersection |
| R3 | Judge-mediated chat-responder number read as equal in strength to a ground-truth one | **high** | §6.2: separate table, fixed banner, split self-preference caveats, hard error on judge==candidate |
| R4 | Corpus embedding cache shared across models or prefix settings → invalid comparison, no visible trace | medium | §5.5: cache key = (model, quantization, docPrefix, corpus version) |
| R5 | Multiple comparisons across 7 counts × turn positions → a false "better" is near-certain | medium | §3.3: pre-registered `verdictMetrics` (1..k) per pack, Holm–Bonferroni when k > 1, everything else labelled exploratory |
| R6 | Temperature-0 replicates counted as independent n | **retired** — the design no longer buys replicates | §4.5: `replicatesPerScript = 1`; still printed next to every conversation-level n (FR-18), now as the field that says DEFF = 1 by construction |
| R6a | The *inverse* of R6: with replicates gone, run-to-run variability is unmeasurable, so model flakiness at temperature 0 is invisible | medium | §4.5.1(iii): a 2-script determinism probe per model, diagnostic and outside `n`; a non-identical result is stated in the report as an unmeasured variance source |
| R9 | The 50.0 pp tool-caller floor read as "no difference exists" when it means "this pack cannot see one" | medium | §7.1's mandatory resolving-power line with unit, unit count, DEFF, basis and α; §4.5.3's named 15–50 pp dark band and its costed reversal trigger |
| R7 | Copied golden data loses its provenance chain at the copy boundary | low | §5.4: `copiedFrom` + `copiedAt` + source git SHA on every copied artifact |
| R8 | A copied `judge.py` drifts from falkor-chat's and the two get conflated | low | Record `judgePromptVersion` in the fingerprint; state in the pack docs that the two are independent by design |

**Open questions — all three of v1.1's are now closed. Closed in place, with what closed them:**

1. ~~Does the stakeholder fund the chat-responder golden data (30 items + 30 calibration items)?~~
   **Closed by FR-21a (2026-09-02): the judged layer is deferred**, and `chat-responder` ships its
   deterministic layer only. §6.2's 30-item checklist set and 40-item balanced calibration set
   describe the design to build *when* it is funded; §6.3's costed conditional stands as the
   trigger, not as an open ask.
2. ~~`primaryMetric` per pack (the retired singular) — the stakeholder may prefer a different
   guard-judge headline.~~
   **Closed by stakeholder decision 2 (2026-09-02): `guard-judge` gets no headline**; both
   class-conditional error rates are co-equal verdict metrics with equal weight. Encoded in §3.3
   (`verdictMetrics` + `headlineMetric: null`) and §7.3 (recomputed at α=0.025). The stakeholder
   declined to rank the two errors; §3.3's reversal trigger says what to do the day that changes.
3. ~~Extending the retrieval golden set to ~60 queries.~~ **Closed as a backlog follow-up**, not a
   first-delivery ask (§7.4). Trigger unchanged: the embedder role becoming decision-critical, or a
   run where recall's saturation blocks a real comparison.

**And v1.1's largest open methodological question, also closed:** ~~how the tool-caller pack should
spend its conversation budget between distinct scripts and replicates~~ — **closed by stakeholder
decision 1 (2026-09-02): 12 distinct scripts × 1 run at temperature 0** (§4.5).

**One new question this version raises, for the stakeholder rather than the architect** — it changes
data cost, not method, and nothing is blocked on it:

- **Is the freed inference budget meant to buy more distinct scripts?** The decision was recorded as
  "same total run budget", but 12 scripts × 1 run is **one quarter** of the previous inference cost
  (≈80 turns per model against ≈320; §4.5.2). If ~48 conversations of inference were genuinely
  budgeted, the statistically correct spend is **48 distinct scripts × 1 run** (16 per shape), which
  moves the floor from 50.0 pp to 12.5 pp and MDD₈₀ from 57.8 pp to 16.0 pp *honestly*. The binding
  constraint is **FR-19 human verification of 36 additional scripts**, not compute. If that authoring
  is not fundable, 12×1 is the right design and §4.5.3's trigger is the right way to revisit it —
  this question needs no answer before S1, and none before S6 either, but it should be asked out
  loud rather than settled by the phrase "same budget".


---

## 11. Q6 — The latency percentile: estimator, denominator, and the two floors (plan R-13)

### 11.1 The question and the decision it serves

Plan §6 R-13 asks for `_percentile`'s definition. Two further inputs arrived before it could be
answered, and all three are settled here because each one moves the others:

1. **The definition** — which estimator, precisely enough that two implementers cannot disagree at
   n = 12, where this component actually operates.
2. **The denominator** *(plan v1.8)*. Under LM Studio's JIT auto-load a **cold first call was
   measured at 21.068 s** (plan §2.5) against sub-second warm calls, so plan §3.6 runs a residency
   probe between items and sets **`ItemResult.latencyMs = None`** on any item a model load touched.
   The latency sample is therefore a subset of the items, **missing exactly the slow ones, because
   they were slow.**
3. **A minimum surviving-sample floor** *(review Pass 3, G3-11)*: a bad run can leave four
   latencies, and a nearest-rank p95 over four points is the maximum — "a number the report will
   print, honestly denominated, that means nothing."

What the caller does differently: S2 stores the first latency figure the day this is answered, and
FR-11's p50/p95 are compared across runs stored months apart, so the definition freezes on first
write.

**The altitude this section takes, stated once.** Latency is **descriptive**: it appears in no
pack's `verdictMetrics` (§3.3's table), it decides nothing, and no verdict rests on it. So the bar
here is deliberately *not* §3.2's bar. The paired instrument refuses because a wrong verdict is a
false claim about two models; the latency block refuses because a number labelled `p95` that is
really a `p85` is a **false label on a true measurement**, and it travels into `index.csv` as a bare
column where no qualifier can follow it. Those are different harms. The second is answered by a
floor looser than §3.2's, by conditional strings that carry the qualifier, and — for the third input
above — by **renaming rather than refusing**, which is a move the comparison instrument never has
available to it.

### 11.2 The estimator — the empirical quantile function (Hyndman–Fan type 1), one implementation

**Recommendation: nearest-rank with ceiling, no interpolation.** For a sample of `X ≥ 1` values
sorted ascending `x₍₁₎ ≤ … ≤ x₍X₎` and a percentile level `p`, the reported value is

> `P_p = x₍r₎`,  `r = ceil(p·X/100)`, clamped to `[1, X]`.

This is the **inverse of the empirical CDF** — `inf{ v : F̂(v) ≥ p/100 }` — and it is a named,
externally documented definition rather than a house convention: **Hyndman–Fan type 1**, R's
`quantile(type = 1)`, NumPy's `method = "inverted_cdf"`. Naming it is half the point: an
implementer can check the implementation against a published definition instead of against this
paragraph.

Four reasons it is the right one *here*, in the order they decided it:

1. **It returns a measurement that was actually taken.** Latency is right-tailed and the report's
   job is "how slow was a slow call". An interpolated p95 at n = 38 is a value strictly between the
   36th and 37th observed calls — a number no call took. For a descriptive statistic that is a
   fabrication with a decimal point on it, and this note has spent four review passes removing
   sentences that were not true of the numbers beside them.
2. **It is the same functional §3.4 Rule 4 already computes in closed form.** Rule 4 (v1.8) replaces
   the resampled bootstrap quantile with the exact quantile of the multinomial resample
   distribution, i.e. `inf{ v : F(v) ≥ p }`. Type 1 applied to an empirical sample is that same
   operator applied to the empirical distribution. So adopting it makes v1.8's closed form a
   **substitution of computation, not of definition** — the two agree by construction rather than by
   coincidence, and no third estimator enters the document.
3. **It is total and exactly reproducible at every X including X = 1**, in integer arithmetic, with
   no platform-dependent float rounding (§11.2.1).
4. **Its bias direction is the safe one for this component.** Type 1's p95 is the smallest observed
   value with at least 95% of the sample at or below it, so it is never *below* an interpolated
   estimate. Against a missingness mechanism that already biases the tail **low** (§11.5), an
   estimator that does not additionally shave it down is the one whose printed sentence survives.

**Rejected: linear interpolation** (`statistics.quantiles`, NumPy's default, Hyndman–Fan type 7).
It is the better estimator of a *population* quantile from a large sample, and that is not what is
being reported: FR-11 asks what the calls cost, over 12–85 of them. It also fails reason 2 — a
second quantile definition in a document whose Rule 4 already fixed one — and it makes the printed
number depend on floating-point interpolation between two observations, reopening the
reproducibility question R-13 exists to close.

**Rejected: the shipped `int(round(p/100·(X−1)))`** (both copies at `5878014`, `stats.py:296` and
`results.py:573`). Beyond being a third definition, `round` is half-to-even, so its tie-break
direction **alternates with X**: measured this session, at X = 4 the p50 index is 2 (the *upper* of
the two middle values) and at X = 6 it is 2 again (the *lower* of the middle pair). An estimator
whose tie-break flips with the sample size is not a definition anyone can reason about.

**One implementation, and it is the only quantile in the package** *(scope restated at v1.18 —
plan-gate P8-1's cheapest fix read the narrower v1.17 wording as licence to put one call site on a
second estimator; §11.2.2 refuses that and §11.10(3) carries the check)*. `stats.percentile` is the
**only** percentile or quantile implementation anywhere in `modelbench`: no module defines a private
helper, `results.py` imports this one, and both bootstrap call sites in `stats.py` call it too. The
plan's own §3.9 says why — *two copies of a
formula is one copy and one bug* — and the S1 implementation review is the evidence: a duplicated
helper is exactly what let `index.csv` compute `latencyMsP95` at the 50th percentile and stay green
(review M27). The function **sorts a copy of its input internally**; requiring a pre-sorted argument
is a precondition a caller can silently violate, and `results.py`'s copy sorted while `stats.py`'s
did not.

```python
def percentile(values: Iterable[float], *, level: Fraction) -> float:
    """Hyndman-Fan type 1 (inverse empirical CDF). `level` is an EXACT RATIONAL in (0, 1].

    `fractions.Fraction`, stdlib, inside §1's dependency budget. p95 is `Fraction(19, 20)`; a
    k-member continuous family's lower level is `Fraction(1, 40 * k)` at alpha = 0.05 (§11.2.2).

    Raises TypeError on a `float` level — a float level reopens the bin-edge hazard the integer
    rank exists to close (§11.2.1) — ValueError outside (0, 1], and ValueError on an empty input:
    whether a figure exists at all is decided by `latency_summary` (§11.6), never by returning
    None from here.
    """
```

#### 11.2.1 The rank is computed in integers, and this is the same hazard as Rule 3a

The rank is one integer expression over the level's numerator and denominator, and it is the whole
of the estimator. Same bin-edge class as Rule 3a's `(7/40)/0.001` case, one operation over:

```python
r = max(1, min(X, -(-level.numerator * X // level.denominator)))   # ceil(level * X), exactly
```

**Three float spellings, and only one of them is safe — v1.17 pinned the safe one and attributed the
unsafe one's number to it** *(corrected at v1.18)*. Re-measured this session over the same lattice
and range v1.17 swept — levels `n/1000` for `n = 1…999`, `X ≤ 3000` — counting ranks that differ
from the integer form:

| Float spelling | Divergences | First (by X) |
|---|---|---|
| `math.ceil(pct / 100 * X)` — the percent form, and the shipped `_percentile`'s | **1626** | `X = 25`, `p = 28.0` |
| `math.ceil(float(level) * X)` — level-first | **755** | `X = 25`, `level = 7/25` |
| `math.ceil(level.numerator * X / level.denominator)` — numerator-first | **0** | — |

v1.17's `0.28 * 25 == 7.000000000000001` is measured and true, and it belongs to the **percent** and
**level-first** spellings; v1.17 printed it under `math.ceil(permille * X / 1000)`, which is the
numerator-first spelling and diverges **nowhere** in that sweep — `permille * X` is an exact integer
and the division that follows is correctly rounded. The number and the expression printed beside it
came from two different spellings. That is `format_floor_pp`'s own docstring warning — *"a sweep run
against the expression the code does not use is how this was missed the first time"* — landing in
the section that cites it, which is why this subsection now names every spelling it measured.

**The numerator-first spelling is safe here and is still not the mandate.** It is exact only while
`level.numerator · X` stays an exactly representable integer, which is a property of the *levels*
reaching it and not of the expression: on a level built from a float — `Fraction(0.05)`, denominator
`2**56` — it diverges from the integer form on **1000** of `X ≤ 20 000`, first at `X = 20`. One
expression that needs no reasoning about which levels reach it is cheaper than a second
correct-for-now spelling.

**Status of the guard, stated the way Rule 3a states its own — and v1.18 does not upgrade it.**
Measured this session: over the four levels this tool takes literally (`1/2`, `19/20`, `1/40`,
`39/40`) at `X ≤ 3000`, and over the whole family lattice `α/(2k)` and `1 − α/(2k)` for `k ≤ 25` at
`X ≤ 20 000`, the level-first spelling diverges **21** times (first at `X = 2520`, on `k = 21`'s
lower level `1/840`) and the numerator-first spelling **0** times; at `X = B = 10 000` — the
resample size §3.2d fixes, and the only `X` a family level meets — **no spelling diverges at any
`k ≤ 25`**. So the integer form stays **defensive, not load-bearing**, exactly as Rule 3a's guard is
under the v1.6 floor. **P8-1's new call site does not make it load-bearing and this note declines to
claim it does**: the reason to mandate it is that the level lattice is now open in two directions —
`k` is any family size and α is any declared level — and the cost of the guard is one expression.
**Pin the code's own expression, never an equivalent-looking one.**

#### 11.2.2 The level is an exact rational, because a family-corrected level is not a decimal

**The question, and the decision it serves** *(v1.18, plan-gate P8-1(c) — a §7 rule 3 raise)*. The
plan's §4 S1e Table C makes `stats.percentile` the package's single percentile, and Table G makes
`paired_bootstrap`'s quantile levels required parameters because an all-continuous `k > 1` family
takes its Bonferroni correction **in the interval** (§3.3) and has nowhere else to put it. The two
collide on `stats.py:159`: at α = 0.05 that line's level is `α/(2k)` — **12.5 ‰ at k = 2, 8.33 ‰ at
k = 3** — and `permille: int` holds neither. The implementer needs one answer before writing the
line, and it is a statistics decision.

**Ruling: the level is an exact rational — `level: Fraction`, `fractions` being stdlib and inside
§1's dependency budget — and `stats.py:159` is not exempted from it.** Three things follow, each a
decision rather than a consequence.

**(1) Widening the integer unit fails, and it fails at k = 3 rather than late.** Per-10⁴ buys k = 2
(125) and loses k = 4 (62.5) and k = 3 outright: `1/120 · 10^m` is `25 · 10^(m−3) / 3`, never an
integer for any `m` — measured at m = 3, 4, 6, 9 and 12. A **fixed** decimal unit expressing
`α/(2k)` for every k does not exist, because the denominator is `40k` and 3 divides one of them. A
unit re-chosen per family is not a unit.

**(2) Rounding the level outward is sound and is rejected on price.** It works, and for the record
its shape is: round the lower level **down** and the upper level **up** onto the chosen lattice, so
the interval only widens — the conservative direction on the one path where the interval is the test
— with the error bounded by one lattice step per bound (`10⁻⁴` at per-10⁴, so at most `2 × 10⁻⁴` of
tail mass across the interval), and the attained pair printed in strings 4 and 5 beside the nominal
one. What it costs is permanent: a second number on every continuous verdict whose entire content is
that the tool could not represent its own level (*"interval taken at 0.8333 %/99.1667 % against a
nominal 0.8333… %"*), and a **second meaning for "attained level"** in one report — §11.6's is a
*coverage* shortfall (how many items were timed) and has nothing to do with where a quantile sits,
so a reader who meets both has two ways to read each. The exact form costs one integer division and
prints nothing. Keep outward rounding named as the fallback for a level that is genuinely
irrational; **no level in this note is** — every one is a ratio of declared quantities — so the
rational form is *closed* over the tool's whole level space, and that closure is what decided it.

**(3) The exemption is refused on a measured cost, not on scope.** P8-1(b) reads §11.10(3) correctly
— its v1.17 wording obliged only `modelbench.results` — so the exemption is legal against the letter
and it is refused anyway. What `stats.py:159` would keep is not a different *unit*; it is the
different **estimator** §11.2 rejects, `int(round(pct/100·(X−1)))`. Measured at `B = 10 000` against
type 1 at the family levels, the two select the **adjacent** order statistic at the lower bound for
`k = 1` (index 250 against 249), `k = 2` (125 against 124) and `k = 5` (50 against 49), and the
**same** one at `k = 3` and `k = 4`; every upper bound coincides. So the exemption ships a
*k*-dependent, one-sided, anti-conservative shift of the bound that decides, on the one path where
§3.2d rules the interval **is** the test, at the only call site in the package whose level is not a
literal — the M27 defect class re-entering exactly where the most is at stake. That it can reach the
published verdict is demonstrated rather than assumed: on a constructed 38-unit sample shifted so
the resample distribution carries an atom at zero, the two estimators disagree on *excludes zero* on
**2 of 60 seeds** (type 1's lower bound `0.0`, the shipped estimator's the next atom up). The
construction is deliberate, as v1.11's `(a=1, b=25, c=12, d=2)` was; the frequency on real data is
**not measured and is not claimed**.

**The signature, and the two places a level is written.** `percentile(values, *, level: Fraction)`;
`levels: tuple[Fraction, Fraction]` on `paired_bootstrap` and `paired_cluster_bootstrap`,
keyword-only and required with no default — Table G's ruling stands and only the element type moves
— plus one refusal Table G could not have: **`levels[0] >= levels[1]` raises**, because a transposed
pair returns an inverted interval that no other check sees.

- **Literal levels are module constants, never call-site expressions.** `LEVEL_P50 = Fraction(1, 2)`,
  `LEVEL_P95 = Fraction(19, 20)`, `LEVEL_CI95_LO = Fraction(1, 40)`, `LEVEL_CI95_HI =
  Fraction(39, 40)` — names recommended, naming being the architect's as `DecidedBy`'s tokens were.
  Those four are the whole literal space, and a constant cannot be built from a float by accident.
- **Exactly one level is computed, and the computation is Rule 8's.** `continuous_verdict()` derives
  `α/(2k)` and `1 − α/(2k)` from `alpha_family` and `len(family)` and forwards them; nothing else
  derives a level. **The conversion is `Fraction(str(alpha_family))`, and it is the one line where
  this can go wrong.** `Fraction(0.05)` is legal and is *not* `1/20` — it is the double's exact
  value, `3602879701896397/72057594037927936` — and its ranks differ from `1/20`'s on **1000** of
  `X ≤ 20 000`, first at `X = 20`. `str()` on a float is its shortest round-tripping decimal, so
  `Fraction(str(0.05)) == Fraction(1, 20)` exactly (measured), which recovers the decimal the pack
  author declared. Its precondition is that `alpha_family` is **declared, never computed** — it is
  read from the pack manifest, and `Fraction(str(0.1 + 0.2))` would recover
  `0.30000000000000004` faithfully and uselessly.
- **The check is an equality on the level, not on the interval** (§11.10(2b)). A `k = 2` family at
  α = 0.05 derives exactly `(Fraction(1, 80), Fraction(79, 80))` — Fraction equality, no tolerance —
  so `Fraction(alpha_family)` in place of `Fraction(str(alpha_family))` fails at the level rather
  than three functions later at a bound nobody can hand-check.

**The closed form takes the same representation** (§3.4 Rule 4). Its atom selector was written
`1000 · cum ≥ permille · n**n` and becomes `level.denominator · cum ≥ level.numerator · n**n` —
still one exact integer comparison, still no tolerance and no bin edge, and now over the same object
the sample estimator takes, which is what §11.2 reason 2 claims and v1.17 could not quite deliver.
Its own levels do not move: the paired **binary** path takes its `k` correction in the Holm ladder
and never in the interval (§3.3), so it passes `LEVEL_CI95_LO` and `LEVEL_CI95_HI` and no `k`
reaches it.

### 11.3 Floor 1 — the identity floor at X = 20, which renames rather than refuses

At this component's sample sizes a "p95" is a near-maximum order statistic. Ranks computed this
session with the integer expression above:

| X (timed items) | rank of the tail figure | timed calls slower than it | rank of the p50 figure |
|---|---|---|---|
| 12 | 12 | **0** | 6 |
| 19 | 19 | **0** | 10 |
| 20 | 19 | 1 | 10 |
| 38 | 37 | 1 | 19 |
| 40 | 38 | 2 | 20 |
| 85 | 81 | 4 | 43 |
| 100 | 95 | 5 | 50 |

**`r = X` — the tail figure *is* the sample maximum — for every `X ≤ 19`** (swept `X ≤ 200`).
Review G3-11 is right that printing that
number under the name `p95` is meaningless; it is wrong only in the remedy, and the difference
matters. The number itself is fine — the largest of 12 timed calls is a real measurement and a
legitimate, low-biased estimator of the population 95th percentile (`E[F̂(max)] = 12/13 = 0.923`).
What is false at `X ≤ 19` is the **label**: `p95` promises "one call in twenty was slower", and
**no** timed call was slower. Refusing the number would discard a measurement the operator wants and
can read out of the record anyway; refusing the *label* costs nothing.

> **The ruling: when `r == X`, the report prints `max`, not `p95`, and `latencyMsP95` is `None`.**
> The figure is the maximum, named as the maximum, carrying the same denominator and the same
> level clause as any other tail figure.

**This floor is derived, not chosen.** `X = 20` is not a convention: it is exactly the smallest `X`
at which `ceil(0.95X) < X`, i.e. where a 95th percentile stops being the maximum. Nothing about it
is tunable, which is what distinguishes it from §11.6's floor and is why the two are separate rules
rather than one blended threshold.

**Its *reach* is a second, composed question, and this section had it wrong** *(v1.20)*. The
withdrawn clause read *"the tool-caller pack's 12 conversations sit inside that range"*, which took
the pack's **analysis-unit** count for its **item** count. They are different numbers by a factor of
6.7: a scored `tool-caller` item is a **turn** — `pairingKey = ["scriptId", "replicate",
"turnIndex"]`, plan §3.3, unchanged since plan v1.4 — so `Y = 80` there (4×9 + 4×7 + 4×4, §4.5.2),
beside 85, 40, 38 and 30 for `guard-judge`, `nlq-generator`, `embedder` and `chat-responder`. `X ≤ 19`
is then far below §11.6's level floor at every one of them, and the two rules compose to a closed
bound:

> **The `max` label is reachable only at `Y ≤ 21`** — and at `Y = 21` only at `X = 19`. Computed in
> the gate's own integer arithmetic and swept over `Y ≤ 200`: at `Y = 22` the level floor already
> requires `X ≥ 21`, where `r = 20 < X`, so no surviving `X` has `r == X`.

**No declared pack is in range, so `max` and `latencyMsMax` are unreachable today** — which does not
retire the rule (a pack with `Y ≤ 21` is buildable and the label would then be false without it) but
does change what it is for and what §11.9 ask 4 may claim for it. **`21` is an asserted constant,
not a sentence** (§11.10 (4a)): the reachability bound and the per-pack `Y` census are the two
things that make the paragraph above true, and a prose reach claim that outlives its mechanism is
this component's most-recorded defect.

**Two consequences worth stating before someone rediscovers them.**
- **`p50` here is not `statistics.median`.** At even X, type 1 returns the **lower** of the two
  middle observations, not their mean. An implementer must not "fix" `latencyMsP50` to the median:
  the mean of two observations is again a value no call took, and it would diverge from the same
  document's Rule 4 quantile.
- **Cross-run comparability is safe within a pack and only within a pack.** `Y` is fixed by the
  pack, so at equal coverage both arms' tail figures are the *same order statistic* and the
  comparison is like-for-like. Two packs' figures are different order statistics of different item
  sets and are not comparable; nothing in the report may place them in one column.

### 11.4 The denominators — `Y` is the item count, `Y_calls` the attempted-call count, and which fields the withholding governs

**`Y = len(run.items)`** — every item the run recorded — and **`X` = the count of items whose timing
survived**. No item is removed from `Y` for any reason.

**There are two units now, and plan v1.25 is what separated them** *(v1.20)*. A scored item is no
longer one model call: a `tool-caller` item is a **turn**, and a turn is a bounded loop of `I(t)`
calls (plan §3.8.4). So each FR-11 figure is denominated in the unit it is a property of, and the
two counts are printed side by side and never substituted for one another:

| figure(s) | unit | counts | printed denominator |
|---|---|---|---|
| `latencyMs` → `latencyMsP50` / `latencyMsP95` / `latencyMsMax` | **item** | `Y = len(run.items)`; `X` = items whose wall clock survived | §11.7 slot 2, in **items** |
| `ttftMs`, `prefillMsPer1kPromptTokens`, `tokensPerSecond` | **call** | `Y_calls` = the calls the run **attempted** (below, v1.22 — *not* `Σ_items callCount`); `X_calls = statsCoveredCount` | §11.7 slot 2's second line, in **calls** |
| `unexplainedMs` → `unexplainedMsMax` | **item** (a sum over that item's calls, §11.5.1) | items with a readable gap | disclosed with the figure; **no gate** (§11.5.1) |

**The wall clock is an item figure because the item is what the report ranks and what the operator
waits through.** On a `tool-caller` that is a whole turn, every iteration and the tool dispatches
between them — the user-visible response time, and the only latency figure an operator can act on.
Redefining it as one call of the turn (the final one, say) would make a model that loops eight times
report a *smaller* latency than one that answers in a single call, which inverts the quantity.

**The three `stats`-derived figures are call figures because a generation is what they measure**, and
they are **pooled over calls** rather than averaged up to the item first: a median of per-item
medians is not a median, and prefill in particular is not even constant within a turn — the prompt
grows with each appended `tool` message, so per-iteration prefill is the signal, not noise to be
averaged out. Pooling weights an item by its own `I(t)`, which is admissible **here and nowhere else
in this note** for one stated reason: these three carry no verdict, no confidence interval and no
Holm step (§11.7 slot 6), so the within-item clustering that §4.4 makes fatal for an inferential
statistic has nothing to invalidate in a descriptive median. Anything that ever puts an interval on
them inherits §4.5.1(i)'s cluster bootstrap over the analysis unit — and nothing does today.

**`callCount` counts the calls that *returned*, and on every role but `tool-caller` an item makes at
most one call — so it is `1` where that call completed and `0` where it did not.** It is
`len(ItemTiming.calls)` and it must equal `TurnTrace.iterations`; two independently maintained
counts of the same thing is how several of this component's defects started, so the agreement is one
assertion (§11.10 (7d)) rather than a convention. The `≤ 1` is a **consequence** of
`maxIterationsPerTurn` and never a second declaration, and on a `deterministic` arm `callCount` is
`0` for every item because that arm makes no model call at all — a real count, not an absence.
*(v1.22, plan R-1: v1.20 wrote "`callCount` is `1` for every role but `tool-caller`", which the pin
`callCount == len(chatResults)` falsifies on both of those cases. The same one word is owed to
§11.9 ask 7's §3.3 item, and ask 8 carries it.)*

**`Y_calls` counts the calls the run *attempted*, and `Σ callCount` is not that number** *(v1.22 —
plan v1.27's raise R-1, ruled: the netting is real, it is not intended, and it does not survive)*.
The two differ by exactly the calls that did not return, and those are precisely the calls that
could not have carried `stats`. Denominating the sibling coverage in **completed** calls therefore
removes a rate's losses from its own base — Rule 3's question, one unit below where `Y` answers it:
the sentence `Y_calls` carries is §11.6's **refusal threshold** and §11.7 slot 2's printed coverage,
and a threshold is weakened by every member netted out of its base whether the member is an item or
a call. Three consequences make it structural rather than cosmetic:

- **A netted base cannot refuse.** `X_calls = statsCoveredCount` is a subset of the *completed*
  calls, so under netting the only shortfalls left are (iv-c)'s co-presence exclusions and a
  completed call whose body carried no `stats` — both of which this section itself calls defensive
  rarities. The systematic shortfall, the calls that never returned, is deleted from the base
  instead of counted in it. A run of 38 single-call items in which 37 calls failed reports
  `X_calls = Y_calls = 1`, clears §11.6's gate at full nominal coverage, and prints a median of one
  call as `n = 1 of 1 calls`. Unnetted it is `1 of 38` and the gate refuses.
- **`Y_calls == 0` is reachable** — the run in which every call failed — and §11.6's integer gate
  `100·r ≥ 45·Y` is vacuously **true** at `Y == 0`. A denominator that collapses to zero on the run
  carrying the least data is the wrong denominator.
- **The invariant the report is written against inverts.** §11.7 slot 6 renders its second sentence
  on `<Yc> > <Y>` and says a `tool-caller` run exceeds it *by construction*; netted, a `tool-caller`
  run with failures can sit at or below `<Y>` and drop the sentence on exactly the run that needs
  it. Unnetted, every item that ran a turn contributes at least one attempt, so `Y_calls ≥ Y` holds
  wherever every item reached the call boundary and the condition means what it says.

**The arithmetic, and it needs no new stored count.** A turn's loop terminates on the first call
that raises (plan §3.8.4, P14-5), so an item has **at most one** non-returning call; the three
failing dispositions map totally onto `withheldFor ∈ {"timeout", "no_response"}` (plan §3.8.4's
table), and `latencyWithheldForNoResponse` counts exactly those items (plan Appendix A: three item
states, one counter). So:

> **`Y_calls = callCount + latencyWithheldForNoResponse`** per run, and
> `a_i = callCount_i + [D(t_i) ∈ {timed-out, no-response, server-rejected}]` per item.

`Y_calls` is therefore **derived, never stored** (§11.8) — the call §11.9 ask 7 already makes for
`unexplainedMs` and `callCount` on `ItemTiming`, for the same reason: a derived number cannot
disagree with the record it is reconstructed from. Three things to state rather than leave to
inference. The derivation **crosses units** — an item count standing in for a call count — and is
valid only under the at-most-one lemma, so **a within-turn retry breaks it**, and that is the
reversal trigger: the day one exists, the attempt is recorded per call and `Y_calls` is summed
directly. It rests on a **precedence the plan states in effect and nowhere states as a rule**, and
plan-gate P15-3 is right that the gap is live: keeping the completed calls' `CallTiming`s on an
incomplete turn (plan v1.27 §3.6(ii)) newly makes `unexplainedMs` computable there, so a load guard
and a failing disposition can both fire on one turn while `withheldFor` holds one value. **Ruled,
because this derivation makes it load-bearing rather than tidy: the failing disposition wins, and
`"load"` is reachable only on a turn every one of whose calls returned.** The cause a withholding
names is the reason the wall clock is absent, and on that turn the reason is that the turn never
finished — the load reading on an earlier call is a real datum about a call, not a cause of the
item's missing measurement, and it survives in that call's own gap. It is also the direction the
counters need: `latencyWithheldForNoResponse` must count exactly the items with a non-returning
call or the identity above is off by the overlap. §11.10 (7d) therefore asserts it by **two
routes** rather than one. And
`Y_calls` sums over `run.items`, so the warm-up call, the residency probe, §3.6's post-timeout
re-probe and any corpus-embedding pass are outside it by construction, as they always were.

**What this does not touch, and the harmonisation that would bring the netting back.** §4.3 rule
4's `fail`/`unrunnable` split partitions the **scoring** denominators. `Y` and `Y_calls` are
**coverage** denominators: they count every item and every attempt whatever the scorer does with
them, which is why an `unrunnable` item is already in `Y` and why a `server-rejected` call — scored
out of every rate — is still in `Y_calls`. The two vocabularies answer different questions (*did the
model succeed?* against *does this figure describe the run?*) and must not be aligned.

**Rejected: giving the non-returning call a `CallTiming` of its own**, so that `calls` holds attempts
and `callCount` needs no companion. It is the tidier record, and it costs the pin
`callCount == len(ItemTiming.calls) == len(chatResults) == TurnTrace.iterations` that plan-gate
P14-6 has just closed and that §4.2(f) reads directly. The only datum it buys — the failed call's own
duration — has no consumer: a timeout's is `requestTimeoutSeconds` by construction, and §11.5.1's
`censoringExact` takes its fail-safe branch on a no-response item without reading any wall clock
(§11.10 (7b)). *Reversal trigger:* the first consumer that needs a per-call duration for a call that
did not return.

**Back to `Y` itself.** The temptation is to net out items that were never timed, so that coverage
reads better. Rule 3's
question settles it: *which denominator keeps this sentence's claim true?* The sentence the
denominator carries is a **refusal threshold** (§11.6), and a threshold is weakened by every item
netted out of its base — a conditional denominator would let a run that timed 8 of 38 items report
"8 of 8, full coverage", which is §4.3's laundering pattern with a different numerator. The
unnetted, largest denominator is the conservative one *for this sentence*, the same way §3.4 Rule 3
gives the observable floor the **unfloored** `n_effective`. It also keeps the latency line on the
same base as every other rate in the report (§4.3's `k/n` discipline).

**And which fields the withholding governs is settled by measurement, not by a conservative
default — review G3-3.** `ttftMs`, the `prefillMsPer1kPromptTokens` input and `tokensPerSecond` are
read off the **same response** as the discarded `latencyMs`, and v1.9 withheld all four on the
grounds that whether LM-Studio-side TTFT includes the JIT load was *not established*. **It is now,
and it does not.** Measured this session against **`qwen/qwen3-4b-2507` (Q4_K_M)** from a genuinely
non-resident start — `residency()` returned `[]` before the call and named the model after it:

| | client wall clock | `stats.time_to_first_token` | `stats.generation_time` | wall − (ttft + gen) |
|---|---|---|---|---|
| **cold call (the JIT load)** | **3 625.0 ms** | 49.7 ms | 89.8 ms | **3 485.6 ms** |
| 50 warm calls, same model | 55.1 / 78.2 / 114.6 ms (min/median/max) | — | — | **−11.3 … +7.6 ms** |

The load is 3 485.6 ms that LM Studio's own `stats` never sees. So:

> **Only `latencyMs` — the client wall clock — is withheld on a contaminated item.** `ttftMs`, the
> prefill figure and `tokensPerSecond` measure generation that happened *after* the load, are
> **kept**, and are each printed with **their own denominator** — which is the other half of G3-3's
> fix, and the half that survives the measurement.

**This reverses v1.9's ruling, and the order matters more than the outcome.** The conservative
default — withhold everything until a field is shown clean — was correct *while the question was
open*, and keeping it once the question closed would have been the same defect one document up:
discarding good measurements to honour a caveat that no longer describes reality. **The wall clock
is the contaminated field and `stats` is the clean one**, and their difference is a detector
(§11.5.1). Aggregates over all three are **medians**, so **all three** take §11.6's **p50**
gate against **their own** coverage, which will normally be `Y of Y` while the wall-clock block is
short. *(v1.13: this note exempted `tokensPerSecond` as FR-11 diagnostic-only, owing its denominator
rather than a refusal. Withdrawn — the three share **one** coverage number, so they print or refuse
together by construction, and a `tokensPerSecond` median printed over exactly the subset the gate has
just judged too short to describe the run is the unstated-subset defect with the subset stated and
then overruled. Plan v1.10's §4 S2 rule (iv-b) already gates all three, and it is right.)*

**One coverage number for three figures is correct only while the three are co-present, and that is
an invariant rather than an assumption.** `ttftMs` and `tokensPerSecond` need a `stats` object; the
prefill figure additionally needs a usable `usage.prompt_tokens`. A call carrying `stats` but no
usable token count puts prefill's true `X_calls` below `statsCoveredCount`, so both the gate and
§11.7 slot 2's single denominator line would overstate coverage for one figure of the three —
silently, and in the direction that prints. The closure is one assertion beside the recomputation
plan §4 S2 already mandates: **the three medians are computed over the same call count, and that
count is `statsCoveredCount`.**

**The runtime disposition, ruled** *(v1.14, plan-gate P5-5; restated per call at v1.20, since v1.25
made an item several of them)*. A **call** carrying `stats` whose `promptTokens` is absent or `≤ 0`
is **excluded from `statsCoveredCount` and from all three sibling medians** — the conservative
single count, not a second count for prefill. **Both halves of that sentence are load-bearing:**
dropping the call from the count while still letting it into the
`ttftMs` and `tokensPerSecond` medians would print a denominator that does not describe its own
numerator, which is §4.3's laundering with the sign reversed and worse than either clean option. The
cost is two good measurements discarded on such a call; the purchase is one number that is true of
all three figures, one gate evaluation, one printed line, and a §11.7 slot 2 whose grammar does not
fork. It is a **disposition, not an assertion** — nothing raises, so an implementer cannot turn it
into a run-ending `assert` (plan-gate P4-7's shape). **The exclusion is per call and never per
item**: on a multi-call item the sibling figures of its other calls are good measurements and
nothing about the excluded one contaminates them. The item's `unexplainedMs` is a separate question
with the opposite answer, and §11.5.1 gives it.

**Why not the separate `prefillCoveredCount` this note leaned toward at v1.13.** `usage.prompt_tokens`
is a standard field of every chat completion and every call this harness makes carries a non-empty
prompt, so the
case is **defensive against something that should not occur** rather than a regime the design serves.
A permanent second denominator — a second gate evaluation, a second stored count, a second line in
slot 2 — is the wrong price for a rarity, and this note has ruled the same way before (§11.6 takes
one 5-point tolerance rather than a second level-scaled constant). *Reversal trigger, observable for
free:* the exclusions are countable from `run.items`, so the first run in which they are **not** a
rarity is the run that buys `prefillCoveredCount`.

**One consequence the plan must carry, or its own invariant refutes the disposition.** §4 S2 rule
(iv) pins `statsCoveredCount == latencyItemCount − latencyWithheldForNoResponse` on a `stats`-bearing
surface. That identity fails twice over now. It fails on the co-presence exclusions — an excluded
call **returned a response and was timed**, so it sits on neither side of the subtraction — and it
fails on the units, since the left side counts calls and the right side items, which differ by
construction on any pack whose `maxIterationsPerTurn` exceeds 1. **The bound that survives is
`statsCoveredCount ≤ callCount`, which is the same statement as
`statsCoveredCount ≤ Y_calls − (calls that returned no response)`** once `Y_calls` counts attempts
— one identity, not two forms *(v1.22: at v1.20 this sentence carried only the second spelling, and
under the netted reading its subtrahend either did nothing or, read literally as a count of every
non-returning call, made the bound **false** on the first run with a failed call and full `stats`
coverage — an assertion that crashes on the data the harness exists to characterise. Plan v1.27
rule (iv) chose the first spelling and is right; its hedge that the two differ is superseded here)*
— and the equality worth
asserting is against a recomputation over `run.items`: `statsCoveredCount` is the count of **calls**
carrying **both** a usable `stats` and a usable `promptTokens`, which §4 S2's one-pass recomputation
already mandates and which is now a sum over each item's `calls`.

### 11.5 The missingness is informative, and how far its direction is known

**Three producers of a withheld timing, and only two of them censor the quantity being summarised**
— which is what the strings have to be built on:

- **Model-load contamination caught by the between-item probe** (plan §3.6). The call ran with a JIT
  load inside it. **The load's cost varies by an order of magnitude** and no report may name a single
  figure for it: plan §2.5 measured **21.068 s** and this session measured **3.625 s** on the same
  box — but on a different model, a different quantization *and* a different route (plan-gate
  P4-11), so **nothing in the pair identifies a cause**, and this note's earlier reading of it as
  page-cache state named one confound out of three. Warm turns are ~1.3 s in plan §2.2's experiment
  and 55–115 ms for the minimal calls of §11.4's table. That range is why §11.7's slot 3 names a
  magnitude and not a number.
- **A scored call that hits `requestTimeoutSeconds`** (review G3-5, whose recommended disposition —
  scored per the pack's rule, `latencyMs = None`, timeout count printed on its own line — this
  section adopts and depends on; §11.9). A censored observation is **not a measurement**: storing
  `latencyMs = 120000` would put the timeout *constant* into the p95, where it is by construction
  the largest value, so the report would print a figure about the configuration rather than about
  the model. That is the opposite bias to contamination and it is the more dangerous of the two
  wall-clock producers, because it arrives as a number rather than as a gap.
- **The in-call reload detector** (§11.5.1), and it is different **in kind** from the two above:
  it withholds on `unexplainedMs`, a **covariate**, never on the wall clock. Nothing ties a withheld
  item's latency to a timed item's — an item with a 1.1 s gap around 0.2 s of generation is withheld
  at 1.3 s while a clean 2.0 s item beside it is timed.

**What all three share is weaker than what two of them share, and every published sentence has to sit
on the weaker one.** Producers 1 and 2 censor the summarised quantity itself, so their withheld calls
do sit above every timed call — **by construction** for the timeout (a timed call returned inside the
budget) and **on measured magnitudes** for the probe-detected load. Producer 3 does not, and its
failure is not exotic: it needs only a timed item slower than the threshold, which on §2.2's ~1.3 s
pack turns is the ordinary case. **So the ordering stops being argued and starts being computed**
(§11.5.1's `censoringExact`), and §11.7's slot 3 selects on the result.

**One consequence survives the loss with no qualification at all, and it is the load-bearing one.**
The printed figure is the `r`-th smallest of the `X` timed items, so **at least `r` of the run's `Y`
items are ≤ it** and its position in the whole run is at worst level

> **`L = r / Y`**, the **attained level**, with `r` the integer rank of §11.2.1.

That argument uses only that the timed items are a **subset** of the run — no ordering, no producer,
no distributional assumption — so §11.6's floor, its 5-point constant and the table below are
untouched by the ruling above, and the floor stays conservative in the safe direction (it can refuse
a figure whose true level was fine, never bless one whose was not). `L` is a *lower bound* on the
figure's true level (exactly `p` when nothing was withheld), so the clause that prints it must
**truncate**, never round to nearest — §3.4 Rule 3's rounding row, applied to a new printed bound.
The **gate compares the exact integer rank, never the display-truncated level** (Rule 7's precedent:
an invariant that inherits the presentation layer's rounding fires or fails to fire by a display
unit).

**Two further consequences are conditional on the ordering, and therefore on the same computed
predicate** — which is exactly why they live in a selected string rather than in the standing one:

- **`p50` is robust.** When the withheld points all sit above the median, the reported median moves
  by about `M/2` ranks in the densest part of the sample — the smallest displacement available
  anywhere in the distribution.
- **The printed figures are lower bounds on the run's own figures.** This is slot 3's claim, and it
  is the one that inverts when a fast call is withheld on its gap.

Worked, exactly, at `Y = 38` — including plan §3.6's own `34 of 38` sketch:

| X of 38 | r (tail) | L (tail) | r (p50) | L (p50) | printed |
|---|---|---|---|---|---|
| 38 | 37 | 97.3 | 19 | 50.0 | both |
| 37 | 36 | 94.7 | 19 | 50.0 | both |
| 36 | 35 | 92.1 | 18 | 47.3 | both |
| 35 | 34 | **89.4** | 18 | 47.3 | **p50 only** |
| **34** | 33 | **86.8** | 17 | **44.7** | **neither** |

#### 11.5.1 The in-call reload detector — R-14's residual, and what it becomes once an item is a loop

Plan R-14 accepts one residual it cannot see: a reload that **begins and ends inside a single timed
call** is invisible to a between-item residency probe, which only ever looks *between* items.
§11.4's table closes it, and the separation is not marginal:

> **`wallClockMs − (ttftMs + generationMs)` is the load, isolated.** Measured: **3 485.6 ms** on the
> cold call against **−11.3 … +7.6 ms** across 50 warm calls — the cold gap is **461×** the largest
> warm gap observed.

**The gap is an accounting identity over one wall clock, and both operands must bracket the same
interval.** That was always the rule. Until plan v1.25 a scored item was one call, so the
matched-bracket form and a per-call form were the same expression and nothing in this note
distinguished them — which is why plan-gate P13-2 could read the rule as per-call and be neither
wrong nor right. They stop being the same expression the moment a `tool-caller` turn runs a bounded
loop (plan §3.8.4) and a scored item is `I(t)` calls: the **item's** wall clock brackets every
iteration plus the tool dispatches between them, while `ttftMs` and `generationMs` bracket **one**
generation. Subtract one call's pair from the turn's wall clock and the residual is the non-final
iterations' entire duration — so the detector fires on **iteration count**, the quantity §4.2(f)
exists to measure, rather than on a load.

> **The ruling: `unexplainedMs` is the sum over the item's own calls of each call's own gap.**
>
> `unexplainedMs = Σᵢ [ wallClockMsᵢ − (ttftMsᵢ + generationMsᵢ) ]`, over the `ChatResult`s the item
> comprises, in order. `runner` withholds `latencyMs` as a contamination (§11.5) whenever it exceeds
> **1 000 ms**. **At `callCount == 1` this is v1.9's expression unchanged**, so every existing
> fixture, the measurement above and the threshold's whole basis survive verbatim — and a
> single-call role cannot tell the two rules apart, which is exactly why the plan may not leave the
> choice to an implementer.

The negative warm gaps are expected rather than anomalous — the client wall clock and the server's
own timers bracket different work — which is why the rule stays a one-sided threshold on a
magnitude and is never `gap > 0`.

**The measurement is per call and the decision is per item, and neither level is a matter of
taste.** The gap can only be *computed* per call: its two subtrahends are properties of one
generation and of nothing larger. The withholding can only be *decided* per item, because the thing
withheld is the item's admitted wall clock and `latencyMs` aggregates over items (§11.4). So
`ChatResult` carries the operands, `ItemTiming` carries the ordered calls, and exactly one number
per item meets the threshold. **Nothing is withheld at the call level**: a load inside call 3
contaminates the turn's wall clock, which is the figure that has to go, while the three
`stats`-derived figures of all `I(t)` calls stay — the load is outside them, which is §11.4's
measured ruling and is unaffected by how many calls an item has.

**Why the sum and not the largest call's gap.** The two agree on every case the detector was built
for: a 3.5 s load inside one call clears 1 000 ms under either rule. They differ on exactly the case
the loop creates, and this section's own false-negative argument decides it — *a false negative is
continuous and **bounded by the threshold itself**; the retained wall clock carries at most that
much foreign time, by the detector's own definition*. That sentence is true of the sum and **false**
of a per-call maximum: at `maxIterationsPerTurn = 8` a turn could retain up to 8 × 999 ms of
unaccounted time with every one of its calls passing, and the bound the threshold's whole
justification rests on would be silently void. A rule that voids its own justification is not the
same rule at a different granularity.

**A partial sum is never formed** — §11.4's co-presence discipline, one unit down. A call yields a
gap only if its `wallClockMs`, `ttftMs` and `generationMs` are all readable. If **any** call of an
item does not, the item's `unexplainedMs` is **`None`**, and the detector does not fire on that item:
it has no reading, and a sum over the calls that happened to report is a number that does not
describe its own item. The item's `latencyMs` stands, guarded by the between-item probe alone —
which is precisely the state a single-call item with no `stats` is already in today, stated here
rather than left to be inherited. **`unexplainedMsMax` therefore takes no coverage gate and opens no
third denominator.** §11.6 gates *medians* because a median over a selected subset misrepresents the
run; a **maximum** over a subset is a **lower bound** on the run's largest gap, which is the useful
direction for the one job that figure has — re-checking the threshold against real payloads.
Wherever it is printed it names the count of items that had a reading, because a bound whose base is
unstated reads as a point estimate.

**The threshold's false-positive margin now scales with the iteration cap, so the claim is pinned to
the cap and not asserted in prose.** The margin was *"~130× the largest warm gap"* — one call's.
Summed over `I(t)` calls the clean ceiling is `I(t) × 7.6 ms`, so the margin is
`1 000 / (7.6 · I(t))`: **~16× at this pack's declared cap of 8**, and **gone entirely at a cap of
132** (`1 000 / 7.6 = 131.6`). That is a reach claim, and a reach claim outliving its mechanism is
this component's most-recorded defect class, so it does not live in this sentence:
**`WARM_GAP_CEILING_MS = 7.6` (§11.4's measurement) and the 1 000 ms threshold are named constants,
and one test asserts `WARM_GAP_CEILING_MS × cap < threshold` for the `maxIterationsPerTurn` of every
pack the suite loads, rendering the margin from the constants rather than quoting it**
(§11.10 (7a)). *Reversal trigger:* the first pack that fails that assertion. The disposition then is
to raise the threshold, or to move to a per-call rule **with its weaker bound stated in the
report** — never to keep this paragraph.

**The basis for 1 000 ms is the asymmetry of the detector's two errors, not a margin against the
smallest load** *(revised at v1.12; plan-gate P4-11)*. The withdrawn clause read the threshold as
sitting *"~3.5× below the smallest cold load measured anywhere"*. It cannot: the two cold loads are
one observation each and differ in model, quantization and route (§11.5), so **the data bounds no
load from below**, and the page-cache-warm reload this note itself hypothesised is precisely the case
a sample of two unattributed cold starts cannot exclude from landing under a second. What does size
the value is that its two errors do not cost alike:

- **A false positive** — a warm item misread as a load — is **discrete and can be severe**: it
  withholds a good wall clock *and* misreports it under §11.7 slot 2's model-load cause. **At
  `Y = 38`, three of them refuse the tail figure and four refuse both**, the second being the run
  that prints no latency summary at all; §11.6 tabulates the budget for every declared pack and the
  smallest is **2 (tail) / 3 (p50)**. *(v1.20 corrects this bullet's own arithmetic. It read "three
  of them at `Y = 38` take the run below §11.6's floor, so a clean run prints no latency summary at
  all", conflating the tail floor with both floors — three is the tail, four is the summary — and
  plan-gate P13-2 quoted it faithfully and inherited the error. The correction makes the false
  positive *cheaper by one item* and changes nothing about the ruling above, whose cost is not three
  items but every multi-iteration turn.)*
- **A false negative** — a load smaller than the threshold — is **continuous and bounded by the
  threshold itself**: the retained wall clock carries at most that much foreign time, by the
  detector's own definition, and it is the **sum** form above that keeps that true at every `I(t)`.

A bounded, disclosed error against an unbounded, discrete one puts the threshold **high**, and
1 000 ms is the highest value that still has a *measured* false-positive margin at every declared
iteration cap. So the value stands and its second margin does not. **What it must not be read as
claiming:** 1 000 ms is roughly three-quarters of a §2.2 pack turn, so this detects a
**load-magnitude** event — R-14's residual, which is seconds — and does not protect a latency figure
to within a fraction of its own scale.

**It remains a starting value with a named basis, not a derived constant, and the re-check is
scheduled rather than deferred — it needs no new instrumentation.** The plan already
stores `unexplainedMs` on every item and reports its maximum below the threshold; that maximum, on
the first real pack run, *is* the false-positive margin on realistic payloads, which is what §11.4's
50 minimal calls of one model cannot supply. The asymmetry above says the value should not move
**down** before that measurement exists. **What the loop adds to that re-check is free and must be
read with it:** `Y_calls / Y` is the run's mean iterations per item, so the observed margin and the
`I(t)` it was observed at arrive together, and a margin read without its `I(t)` is not a margin.

**What the loop does *not* open, stated so that nobody builds a guard for it.** A reload that happens
*between* two iterations of one turn is invisible to the between-item probe (which runs between
items) **and** sits outside every call's wall clock, so the sum above would miss it. It costs
nothing, because LM Studio's load is **JIT on the request**: whatever was unloaded is re-loaded
inside the *next* call, where that call's own gap sees it — the same mechanism §11.4 measured, a
cold call whose 3 485.6 ms is entirely invisible to the server's own `stats`. What genuinely does
sit between the calls is the harness's own tool dispatch and message assembly, and that is **not**
foreign time to be subtracted: it is real time the operator waits through and it belongs inside the
turn's latency. The identity worth asserting rather than assuming is
`ItemTiming.wallClockMs ≥ Σᵢ wallClockMsᵢ`, the difference being harness-side (§11.10 (7d)).
*(On `tool-caller-shop-assistant` the tools are an in-process simulated storefront, so that
difference is microseconds. A pack whose tool module performed network or disk I/O would put that
I/O inside its latency figure — the honest place for it, and a fact the report must state rather
than let a reader discover. Reversal trigger: the first pack declaring such a module owes a
dispatch-time figure of its own.)*

**The detector is not right-censoring, so slot 3's ordering is computed rather than assumed**
*(v1.12; plan gate Pass 4 open question 2)*. The predicate, evaluated per render whenever `M > 0`:

> **`censoringExact`** — **true** when **every** withheld item is either an item whose withholding
> reason is a **timeout**, or a **load**-withheld item whose wall clock is readable and exceeds the
> largest wall clock among the timed items. **False** otherwise — in particular false whenever any
> withheld item is a **no-response failure that is not a timeout**, and false whenever a
> load-withheld item's wall clock is not readable from the record.

A timeout needs no comparison — it exhausted a budget every timed call returned inside. **The
no-response branch is new at v1.14** (plan-gate P5-6), and it is not a fine point of evaluability:
**a timeout is a *censored* observation and a call that failed at 40 ms is a *missing* one.** The
first has a known bound; the second has no value in either direction, so a string asserting it was
slower than every timed call asserts something about a measurement that never existed.

**The loop changes how often the predicate comes back false, and that is the predicate working**
*(v1.20)*. Items are now heterogeneous in `I(t)`, so a clean 8-iteration turn is legitimately slower
than a load-contaminated 1-iteration one, and the second clause — a load-withheld item's wall clock
above every timed item's — fails more often than it did when every item was one call. Slot 3's
stronger string would be **untrue** on such a run, so the weaker one is owed and is what renders.
Nobody should "fix" the increased false rate by comparing within iteration count: the claim the
string makes is about the run's items as printed, and stratifying the comparison would make it a
claim about a stratum the block never shows.

**What the plan must preserve, and it is one field value.** Plan v1.10 merged timeout into
`no_response`, which is **right for the counter** — §11.7's cause split prints one `no response`
label either way, already ruled — and **wrong for the item**, because clause 1 is a statement about
one item's mechanism rather than about a total. `withheldFor` therefore needs a **third value,
`timeout`**, distinct from `no_response`. Without it the predicate is unevaluable and a timeout-only
run silently renders the weaker string: the safe direction, but not the true one, and nothing in
either document would record that it had happened. **The counter stays one; the item state becomes
three.**

The comparison needs the withheld wall clock to survive on the item, which §11.6 already promises the
reader it does (§11.9 item 2b); where it does not, the predicate **fails safe** to the weaker string
rather than to the stronger one. One comparison per render, over numbers the record already holds.

**§11.5's level bound is undisturbed by any of this**, and it earns a sentence because the natural
worry is that an item with no timing at all breaks a statement about *the run's `Y` items*: the bound
counts only items known to be **≤** the printed figure, so an item with no value — or with a value
nobody will ever know — can fail to be counted but can never falsify the count. That one-sidedness is
what made the bound distribution-free at v1.12, and it is what makes it survive a third item state
here — and a multi-call item, which changes what an item *is* without changing that it is one member
of the ranked set.

This detector is **additive to** the residency probe, not a replacement: the probe catches a reload
that happened *before* an item, the gap catches one *inside* any of its calls, and neither sees what
the other does. Placing it is `architect`'s (§11.9); the metric, the threshold and its basis are here.

### 11.6 Floor 2 — the level floor, which refuses; and how the two floors divide the work

**Yes, a latency figure is refused below a coverage threshold, and the refusal is at the *value*,
not in the prose.** The reason it cannot be a prose qualifier is structural: `index.csv` carries
`latencyMsP50` / `latencyMsP95` as **bare columns** and `compare` renders them in a summary table,
so any qualifier living in a sentence is stripped the moment the number is read the way the tool
intends it to be read. This component already has the right precedent — `coldLoadSeconds` is
**absent, never `0`**, when it was not measured — and this is the same move.

> A printed figure's **attained level may fall short of its nominal level by at most 5 percentage
> points**. Below that the figure is **absent** — `None` in the record, an empty cell in
> `index.csv`, and a refusal clause in the report.

Evaluated in integers, with no float anywhere in the gate:

```python
print_p50  = (100 * r50 >= 45 * Y)      # nominal 50.0, floor 45.0
print_tail = (100 * r95 >= 90 * Y)      # nominal 95.0, floor 90.0
```

**The 5-point tolerance is a chosen convention, and it is named as one.** Its basis is the band the
headline figure's own name claims: `p95` names the top 5% of calls, so a figure whose attained level
has slipped a further 5 points is sitting at the outer edge of a band **twice as wide as its name** —
the last point at which the label still describes the number. The same 5 points serve `p50` rather
than a second, level-scaled tolerance, for §3.2c's reason: two constants written in two sentences
drift apart, and the tolerance's job — bounding how far the label may travel from the number — is
measured in the same units at every level. *Reversal trigger:* a stakeholder who wants latency
reported at any coverage moves this constant and nothing else; the strings, the ranks, the
denominator and §11.3's identity floor are all unaffected by its value.

**How the two floors divide the work — G3-11's question answered directly, with the numbers.**
A ratio floor and a count floor fail in different regimes, and here they are **complementary rather
than redundant — only one of them refuses — with one honesty correction at v1.20: at the pack sizes
actually declared, the level floor is always reached first, so the identity floor is a live rule
with no run to fire on rather than a co-equal gate:**

- The level floor implies a **count** floor, because `r ≤ X` forces `100·X ≥ 90·Y`, i.e.
  **`X ≥ 0.9·Y`** — a floor that is re-derived per pack for free, because it is expressed against
  `Y`. **G3-11's four-surviving-latencies case is refused outright at every declared pack**, not
  merely renamed. **The per-pack budget, computed in the gate's own integer arithmetic** *(v1.20 —
  the withdrawn version of this bullet said "every pack in §3.3's table has `Y ≥ 12`" and put the
  tool-caller at `X ≥ 11`, reading its **analysis-unit** count for its **item** count; §11.3
  carries the correction and the general rule)*:

  | pack (role) | `Y` items | tail prints while | `p50` prints while | withheld budget, tail / p50 |
  |---|---|---|---|---|
  | `tool-caller` | **80** turns (4×9 + 4×7 + 4×4) | `X ≥ 75` | `X ≥ 71` | **5 / 9** |
  | `guard-judge` | 85 | `X ≥ 81` | `X ≥ 77` | 4 / 8 |
  | `nlq-generator` | 40 | `X ≥ 37` | `X ≥ 35` | 3 / 5 |
  | `embedder` | 38 | `X ≥ 36` | `X ≥ 35` | 2 / 3 |
  | `chat-responder` | 30 | `X ≥ 28` | `X ≥ 27` | 2 / 3 |

  **This is the table §11.5.1's false-positive cost is priced against**, and it is why the
  detector's unit had to be settled before a rework unit built the loop. The tool-caller's budget is
  the largest in the component, and a gap taken against the turn's wall clock would have exhausted
  both halves of it on the first clean run.
- What the level floor **cannot** see is a *clean* small sample: at `X == Y == 12` the shortfall is
  zero and the gate passes, correctly. That is exactly the regime §11.3's identity floor governs,
  and it governs it by **renaming**, because the number is sound and only the label is not.
  **No declared pack is in that regime** — the identity floor's `max` label is reachable only at
  `Y ≤ 21` (§11.3) and the smallest declared `Y` is 30 — so today the identity floor is a rule with
  no run to fire on. It stays, because the label it refuses would be false on the pack that
  eventually has one, and because its cost is a comparison.

So: **one floor for a sample that is short (refuse), one for a sample that is small (rename).** They
are checked in that order, and the identity floor is evaluated first so that a refusal message names
the figure the run would have had.

**Four properties of the level floor, all checked this session, each falsifiable:**

1. **It can never fire on a clean run.** At `X == Y`, `r/Y = ceil(p·X/100)/X ≥ p/100` by the
   definition of `ceil`, so the shortfall is never positive. A run that timed every item always
   prints both figures, at every X including X = 1.
2. **A single withheld item never suppresses either figure at `Y ≥ 10`** (swept `Y ≤ 399`,
   contiguous from 10). Every pack has `Y ≥ 30`, so the common disturbance — one TTL expiry — costs a
   printed qualifier, not a printed number. **This is what makes the floor safe against G3-4:** if
   the guard's first comparand is left as written and every cold-start run discards item 1, the floor
   still prints both figures on every pack. The floor does not depend on G3-4 being fixed — but
   G3-4 should be fixed anyway (§11.9), because a systematically absent item 1 is a missing
   measurement dressed as a contamination event.
3. **The two figures are gated independently, and the band between them is real.** `p50` survives to
   lower coverage than the tail figure — at `Y = 38`, both print at `X ≥ 36`, `p50` alone at
   `X = 35`, neither at `X ≤ 34`. That asymmetry is §11.5's robustness fact made visible rather than
   averaged away, and it is why the p50-only string exists.
4. **It bites on the plan's own example.** `latency n = 34 of 38` — §3.6's sketch — prints **no
   figure at all** under this ruling. Four withheld items in one run is a disturbed run, and the
   honest output is that it has no latency summary, not a summary of its fast calls.

**What replaces the number.** Not silence: the refusal clause names the coverage, the mechanism and
the direction of the bias, states that **the per-item timings remain in the run record — the summary
is withheld, not the data** — and names the operator's remedy. The scored outcomes are untouched in
every case and the report says so, because a reader who sees a refused latency block must not infer
that the run's quality numbers are also suspect.

### 11.7 The published block — the clause grammar, verbatim, with the condition selecting each

**Why a grammar and not one string per case.** The block varies on three independent conditions
(tail named `p95` or `max`; each of two figures printed or refused; anything withheld or not), which
is twelve monolithic strings to keep in step — more surface than the defect they prevent, and this
note has already paid once for two statements about the same thing drifting apart (§3.2c's trap). So
the **slots and their order are fixed**, each variant is published verbatim, and §11.10 pins five
fully rendered examples as test targets.

**Substitutions.** `<X>` timed **item** count · `<Y>` item count · `<M> = Y − X` · `<ML>` withheld
for model load · `<MT>` withheld for a **non-returning call** — timeout *or* no response, plan-gate
P4-7's widened `latencyWithheldForNoResponse`, which is why it is also the attempt count below
*(v1.22: the gloss said "timeout", naming one of the two things the counter counts)* ·
`<Xc>` = `statsCoveredCount`, a **call** count ·
`<Yc>` = `Y_calls`, the run's **attempted** model calls — `callCount + <MT>`, never
`callCount` alone (§11.4, v1.22) · `<TAIL>` = `max` when `r95 == X` (§11.3)
else `p95` · `<p50>`/`<tail>` in **milliseconds, rounded to the nearest integer** · `<s> = X − r95` ·
`<L95>`/`<L50>` **truncated** to 1 dp.

**Two nouns, one per field group, and they are bound to the group rather than typed into a string**
*(v1.20)*. Every wall-clock line is denominated in **items** and every `stats`-derived line in
**calls**, because that is what §11.4 denominates the figures in. On four of the five packs the two
numbers are equal (one call **attempted** per item, whether or not it returned — v1.22), so a
fixture built on any of them cannot tell the nouns apart — which makes the multi-call fixture of §11.10 (7d) the only thing that pins this, and
makes typing the noun into each string the one implementation that will pass the suite and print a
false denominator on the pack that matters. The same trap, one vocabulary over, already cost this
component a silent comparison (`BinaryMetric.unit` against `PackRef.analysisUnit`).

**The values in the renderings below are measured, not illustrative.** Source: **50 warm
`POST /api/v0/chat/completions` calls against `qwen/qwen3-4b-2507` (Q4_K_M)** on this box,
`temperature: 0`, `max_tokens: 16`, one short prompt each, following the single cold warm-up call of
§11.4's table on a model `residency()` had just reported non-resident. The **first 38** calls are the
`Y = 38` sample and the **last 12** the `Y = 12` sample; each partial-coverage rendering withholds
the **k slowest** calls of that same sample, which is exactly what the guard does to it. These are
minimal chat calls and so run an order of magnitude faster than plan §2.2's ~1.3 s pack turns — what
is being pinned is the **shape** of the block, and the ranks and levels in it are exact.

**The unit is milliseconds and the block never prints seconds.** `ItemResult.latencyMs` is a float
in ms; `coldLoadSeconds` is the only latency field in this tool measured in seconds, it is reported
separately (plan §3.6) and it never appears inside this block. **The values round to nearest**, not
up and not down: unlike the MDD and the observable floor, a printed latency asserts no bound — it
displays a measurement — so Rule 3's question returns "nearest" for this cell, and the direction
must not be copied from the bounds beside it.

**Slot 1 — the figures.** Exactly one variant, prefixed `Latency (client wall clock): `.

- both printed → `p50 = 78 ms, p95 = 112 ms.`  *(or `max = 104 ms` when `<TAIL>` is `max`)*
- p50 only → `p50 = 76 ms; no p95 is reported.`  *(`no max is reported` under `<TAIL>` = `max`)*
- neither → `not reported for this run.`

**Slot 2 — the denominator.** Always present. Two variants, on `M == 0`:

> `latency n = 38 of 38 items; no timing was withheld.`

> `latency n = 36 of 38 items; timings withheld: 2 (model load 2, request timeout 0).`

A **second denominator line follows it when the arm's call surface produces a `stats` object and the
two counts differ**, because §11.4 keeps the server-side timings on an item the wall clock was
withheld for:

> `ttft/prefill/tokens-per-second n = 38 of 38 calls; these are LM-Studio-side figures and a model load is outside them (see the note's 11.4).`

**Its denominator is `<Xc>` of `<Yc>` and its noun is `calls`, on every pack** — not `items`, and
not `items` on the four packs where the two counts coincide. The condition selecting the line is
unchanged (the surface produces a `stats` object and the two *coverages* differ); what v1.20 changes
is that "the two counts differ" becomes the ordinary case rather than the exception, since on a
`tool-caller` run `<Yc>` exceeds `<Y>` by construction.

**The surface condition is a ruling, not decoration** *(v1.12; plan-gate P4-13, confirmed — and
placed on the surface rather than on the arm profile, which is the wider condition)*.
`POST /api/v0/embeddings` returns no `stats`, so on an embedder arm the three figures do not exist
and a coverage line for them is a line about nothing; a `deterministic` arm is the second case, and a
profile-shaped condition would miss it. So the carrier must distinguish *this surface has no such
figure* from *this surface has them and none arrived*: `statsCoveredCount` is **`None`** wherever the
surface returns no `stats` and the line is **not rendered at all**, while `0` keeps its meaning on
the chat surface, where it says every response lacked `stats` and is a real signal (§11.9 item 5).
Absent-never-zero is the move `coldLoadSeconds` already makes (§11.6). One consequence worth naming:
with no `stats` there is no `unexplainedMs`, so §11.5.1's detector does not run on those arms and
their only load producer is the between-item probe.

The cause split is mandatory and both counts print even at zero: the two causes carry very different
operator actions — a TTL reload is a re-run, a timeout is a hung model — and a reader who sees only
the total cannot tell which run they have. Printing both at zero also makes the line's shape
constant, so a reader who has learned it once reads every run the same way. *(If the plan widens the
second counter from `requestTimeoutSeconds` alone to any scored call that returned no response —
plan-gate P4-7 — the cause reads `no response <MT>` and nothing else in this grammar moves: the label
names what the counter counts, and the split stays exhaustive.)*

**Slot 3 — the mechanism.** Present only when `M > 0`. **Two variants, selected on §11.5.1's
computed `censoringExact`** — not on the producer, and not on an argument *(v1.12)*:

- `censoringExact` true →
  > `Every withheld item was slower than every timed item — a model load adds seconds to a call, and a timed-out call by definition exceeded the request budget — so the figures below are lower bounds.`
- `censoringExact` false →
  > `The withheld items are not all slower than the timed ones: an item is also withheld when too much of its wall clock is unaccounted for by the server's own timers, and one whose call returned no response has no timing to compare at all — neither is necessarily among this run's slowest. So the figures below are computed over the surviving items only and are not lower bounds on the run's own figures; the levels below hold either way.`

**Why two strings rather than one weaker one that is always true.** The true branch is the ordinary
case — both wall-clock-censoring producers land in it, and §11.7's own measured fixtures withhold the
`k` slowest calls, so it is what the assembled rendering below shows — and it carries the reading an
operator acts on. Deleting it would cost every render the statement that is true on most of them; the
selector is *computed*, so neither string is ever rendered where it is false, which is the property
four review passes have been spent buying for this block.

**Both branches say *item* where v1.11 said *call*, and the true branch has lost one clause**
*(v1.20)*. The withheld and timed objects are items, which are calls only on four of the five packs,
so the noun had to move. The deleted clause is *"a call that otherwise takes tens to hundreds of
milliseconds"*: that magnitude was measured on §11.4's minimal warm calls and is false of a
`tool-caller` turn at ~1.3 s and up, and this section's own rule below — a string carrying a figure
is a string that is false on most of the runs that render it — condemns it. What survives is the
*load's* magnitude (*"adds seconds"*), which §11.5's measurements do support and which is the half
the sentence needs.

**Slot 3 names a magnitude and never a figure — in both branches**, and that is a correction to
v1.9, which wrote
*"a model load costs about 21 s"* into the string. §11.5's own measurements refute it: the cold load
was **3.625 s** this session and **21.068 s** in plan §2.5. A string that carries a load cost is a
string that is false on most of the runs that render it — the exact defect class four review passes
have been spent removing, reproduced by this note in the act of documenting it. The same rule is why
the false branch says *"too much of its wall clock"* and never *"more than a second"*: §11.5.1's
threshold is a starting value, and a string carrying it would be the same defect one constant over.

**Slot 4 — the levels.** Omitted entirely when `M == 0` (nothing was withheld, so the levels are
nominal and a clause restating that would be noise that drifts). Otherwise one variant — and **every
sentence in them is a statement about a *level*, never about a value**, which is what makes slot 4
independent of `censoringExact`: at v1.11 two of these clauses said *"stands as a lower bound"* and
*"this run's faster calls"*, both of which are value claims that the ordering carried, so both are
rewritten *(v1.12)*.

- both printed →
  > `The p95 figure is at worst percentile 92.1 of the run's 38 items, and the p50 figure at worst percentile 47.3.`
- p50 only →
  > `The 95th percentile of the timed calls is at worst percentile 89.4 of the run's 38 items, more than 5 points below the level its name claims, so no p95 is reported for this run. The p50 figure is still reported: at worst percentile 47.3.`
- neither →
  > `The 95th percentile of the timed calls is at worst percentile 86.8 of the run's 38 items and the 50th at worst percentile 44.7, both more than 5 points below the level their names claim, so the 34 surviving timings are a selected subset of this run rather than a sample of it. The per-item timings are in the run record — the summary is withheld, not the data. A latency summary for this model needs a re-run in which the model stays resident throughout, and the scored outcomes are unaffected either way.`
- `X == 0` → replaces slots 2–4 entirely:
  > `latency n = 0 of 38 items; no item's timing survived. The scored outcomes are unaffected; only the timing is absent.`

*"at worst percentile 92.1"*, not *"the 92.1st percentile"*, deliberately: an ordinal suffix on a
decimal is a rule an implementer has to encode and will get wrong, and the phrase carries no less.

**Slot 5 — the tail figure's own reading.** Present only when a tail figure was printed. Two
variants, and the condition is `<TAIL>`:

- `p95` (i.e. `X ≥ 20`) → `Timed items slower than the p95 figure: 1 of 36.`
- `max` (i.e. `X ≤ 19`) → `At 12 timed items the 95th percentile is the largest observation, so this run reports the maximum and no p95.`

**Slot 6 — the standing label, on every render whatever the coverage.** One sentence always, plus a
second **when and only when `<Yc> > <Y>`** — a computed condition over the record, never a pack
declaration, so a pack that declares an iteration cap and never reaches it renders the short form.

> `Latency is descriptive: it is in no pack's verdictMetrics, it carries no verdict, no confidence interval and no Holm step, and no difference between two arms' latency figures is printed unless both arms report the same order statistic.`

> *(`<Yc> > <Y>` only)* `A timed item on this pack is a whole turn — every model call the turn made, and the tool dispatches between them; this run averaged 2.4 calls per turn. A latency difference between two arms may therefore be a difference in how many calls they made rather than in how fast they generate: the per-call figures above and the iteration counts in the tool-calling block are what separate the two.`

**The second sentence exists because the estimand changed and the column name did not.**
`index.csv`'s `latencyMsP50` holds a per-call figure on four packs and a per-turn figure on one, and
§11.3 already forbids comparing across packs — but a reader comparing two *arms* of the same pack
can read a real turn-latency difference as a speed difference when it is an `I(t)` difference. The
report needs no new statistic to separate them: `ttftMsMedian` and `tokensPerSecondMedian` are the
speed figures and §4.2(f)'s mean and p95 of `I(t)` are the behaviour figures, and both are already
printed. What was missing was the sentence saying so. **The mean is rendered from `<Yc>/<Y>` to
1 dp and never typed**, for §3.2c's reason. It is **attempts per item over every item** *(v1.22)*,
which is neither the mean over the run's *timed* turns nor §4.2(f)'s restricted `I(t)` mean — the
sentence around it already sends a reader to the second, and the two differ on any run with a
non-`replied` turn (§4.3.1 item 4).

**Slot 7 — `compare`, per figure, when either arm has none.** Both arms' blocks always print in
full, side by side; only the *difference* is withheld:

> `No p95 comparison: qwen/qwen3-4b-2507 has no reportable p95 (latency n = 34 of 38 items).`

**The condition on slot 6's last clause is `r_A == r_B`, not full coverage on both arms, and the
difference matters.** The reason a difference is withheld is that at unequal coverage the two arms'
figures are *different order statistics* (§11.3), so their difference is not a latency difference.
Full coverage is a sufficient condition for equal rank, not the necessary one — and under G3-4 as
written, no cold-start run reaches full coverage at all, so a full-coverage condition would suppress
every latency difference this tool ever prints. The condition must be the one its own rationale
names. **And no paired latency interval is printed on any path:** the intersection of two arms'
timed items is doubly selected by the same informative mechanism, so §3.2d's paired bootstrap has no
valid sample here even though latency is a continuous metric.

**One block assembled end to end**, the `X = 36, Y = 38` case (measured sample, two slowest calls
withheld), because a grammar is only checkable against at least one full rendering:

> `Latency (client wall clock): p50 = 76 ms, p95 = 105 ms.`
> `latency n = 36 of 38 items; timings withheld: 2 (model load 2, request timeout 0).`
> `ttft/prefill/tokens-per-second n = 38 of 38 calls; these are LM-Studio-side figures and a model load is outside them (see the note's 11.4).`
> `Every withheld item was slower than every timed item — a model load adds seconds to a call, and a timed-out call by definition exceeded the request budget — so the figures below are lower bounds.`
> `The p95 figure is at worst percentile 92.1 of the run's 38 items, and the p50 figure at worst percentile 47.3.`
> `Timed items slower than the p95 figure: 1 of 36.`
> `Latency is descriptive: it is in no pack's verdictMetrics, it carries no verdict, no confidence interval and no Holm step, and no difference between two arms' latency figures is printed unless both arms report the same order statistic.`

### 11.8 Where the figures live

- **`RunResult` / the record:** `latencyMsP50` and the tail figure are **`None` exactly when the
  gate refuses them** (and `latencyMsP95` is additionally `None` whenever `<TAIL>` is `max`, §11.3),
  never `0` and never a figure carrying a hidden qualifier. `X`, `Y`, `ML`, `MT` and — v1.20 —
  the run's **call** count are stored beside them, so every clause in §11.7 is reconstructible from
  the record alone. The fifth stored count is `callCount`; `Y_calls`, which is what slot 2's second
  line and slot 6's second sentence are actually computed from, is **derived** from `callCount` and
  `MT` and is not a sixth *(v1.22, §11.4)* — a denominator built out of two stored counts in one
  named place cannot drift from them, and storing it would be the third home for a number this
  block has already substituted once.
- **`index.csv`:** the cells are **empty exactly when the record's fields are `None`**. This is the
  property that makes a qualifier-free column honest: *a populated `latencyMsP95` cell is always a
  genuine 95th percentile whose attained level is within 5 points of its name.* That single sentence
  is the whole argument for gating at the value rather than in the prose. **Three count columns join
  it at v1.20 and the third is a correctness requirement, not a readability one** — `latencyTimedCount`
  and `latencyItemCount` are the coverage a CSV reader cannot otherwise see (§11.9 ask 4's
  recommendation), and **`callCount`** is what tells that reader whether the `latencyMsP50` cell in
  front of them is a per-call figure or a per-turn one. Without it the column silently holds two
  estimands, which is the same defect the gate closes at the value: a number whose qualifier lives
  somewhere the reader is not.
- **`compare` and the per-run report:** §11.7's slots.

### 11.9 What this needs from the plan — `architect`'s, and precisely scoped

R-13 closes on the method side with this section. Four things it depends on are **plan-owned**
(§7 rule 2 — the result schema and the harness surface are the plan's). The first three are review
Pass 3 findings already routed there; they are restated here only as the *shape* this section needs,
so one plan revision can serve both.

1. **G3-5 — the timeout disposition, and it is load-bearing here, not merely adjacent.** §11.5
   depends on a timed-out scored call yielding **`latencyMs = None`** (scored per the pack's rule,
   run continues), which is G3-5's own recommendation. Under the alternative reading —
   `latencyMs = 120000` stored as a measurement — **§11.5's mechanism inverts**: the tail figure is
   then biased *high* by a configuration constant, which is the opposite of what §11.7's slot 3
   asserts, and **no computed predicate catches it**, because such an item is not withheld at all and
   so never reaches `censoringExact`. If the architect chooses to abort the run instead, everything
   above still holds with `MT` pinned at 0.
2. **G3-3 — the three siblings need their own denominator; they must *not* be nulled.** G3-3's
   free measurement has been taken (§11.4): LM-Studio-side TTFT **excludes** the JIT load, so
   `ttftMs`, prefill and `tokensPerSecond` stay on a contaminated item and print `n = Y of Y` beside
   the wall clock's `n = X of Y`. The defect G3-3 names is real and its fix is the **denominator**,
   not the nulling. **This reverses what this section asked for at v1.9**, on the evidence G3-3
   itself proposed gathering.
2a. **The in-call reload detector (§11.5.1)** *(v1.12; the arithmetic restated at v1.20 for plan
   v1.25's loop)*. `runner` computes each call's own gap — `wallClockMsᵢ − (ttftMsᵢ +
   generationMsᵢ)` — **sums them over the item's calls**, and withholds the item's `latencyMs`
   above **1 000 ms**. At `callCount == 1` that is the v1.9 expression unchanged. It closes R-14's
   acknowledged residual — a reload inside a single timed call, which the between-item probe
   structurally cannot see — at the cost of one subtraction per call, and it needs no new probe.
   Its placement is the plan's; the metric, the threshold and the unit each is applied at are
   §11.5.1's, and ask 7 below lists what the plan must say to carry them.
2b. **New at v1.12: a withheld item's wall clock must stay readable on the record.** §11.5.1's
   `censoringExact` is one comparison between the withheld and the timed wall clocks, and §11.6
   already promises the reader that *the summary is withheld, not the data* — so the number has to
   survive somewhere other than `latencyMs`, whose absence is what keeps it out of every aggregate.
   **The field's name is the plan's**, and the per-item fields plan-gate P4-3 asks for already
   reconstruct it exactly (`unexplainedMs + ttftMs + generationMs`) if the architect prefers adding
   none. **This is not a blocker:** without it the predicate fails safe and §11.7's slot 3 prints its
   weaker variant on every render that withheld anything for load.
3. **G3-4 — the guard's first comparand.** §11.6(2) shows the floor holds either way, so this is not
   a blocker for R-13. It should still be fixed as G3-4 specifies (baseline = a probe taken *after*
   the warm-up returns), because otherwise `MT`/`ML` count a systematically absent item 1 as a
   contamination event and slot 2's line reports a mechanism that did not occur.
4. **A `latencyMsMax` field and column — kept, on a corrected justification** *(restated at
   v1.20)*. The v1.12 wording read *"§11.3 makes `latencyMsP95` `None` for every run with `X ≤ 19` —
   which is every tool-caller run, the pack that is the long pole"*. That is false, and the way it
   is false is this component's signature defect: it took the tool-caller's **analysis-unit** count
   (12 conversations) for its **item** count (80 turns). §11.3 now carries the general correction
   and the exact reachability bound, and the consequence for this ask is that **`latencyMsMax` is
   `None` on every run of every declared pack** — the `max` label is reachable only at `Y ≤ 21` and
   the smallest declared `Y` is 30. The field still earns its home, on the narrower argument: §11.3's
   rename is the rule that keeps the `p95` **label** honest, a pack with `Y ≤ 21` is buildable and
   costs nothing to be ready for, and the alternative — discovering at that point that the maximum
   has nowhere to live — is the state this ask was written to prevent. What must **not** survive is
   the claim that the component's headline pack needs it today. **Alongside it:** columns carrying
   `X`, `Y` and `callCount` (§11.8) — the first two a readability improvement, the third a
   correctness requirement now that one column holds two estimands.
5. **New at v1.12: `statsCoveredCount` is `None`, never `0`, on a call surface that returns no
   `stats`** — today `POST /api/v0/embeddings`, and every `deterministic` arm. §11.7's second
   denominator line renders on exactly that condition, and §4 S2's invariant over the field is then
   scoped to the surfaces where it is a number — **by surface, not by arm profile**, which is the
   wider and therefore the correct condition. The distinction is the field's whole content: `0` on
   the chat surface is a real signal and must stay distinguishable from *no such figure exists here*.

6. **New at v1.18: what §11.2.2's ruling changes in §4 S1e, stated as the plan's edit list**
   *(plan-gate P8-1; the note rules the method, the tables are `architect`'s)*. **(a)** Table C's
   replacement is **not** scoped away from `stats.py:159` — there is no exemption, and the reason is
   §11.2.2(3) rather than the width of §11.10(3), which is now package-wide anyway. **(b)** The new
   signature Table C's four site rows move to is `percentile(values, *, level: Fraction)`, and
   `stats.py:296`'s replacement is the public `percentile` with the four `LEVEL_*` constants beside
   it. **(c)** Table G's `levels` parameter becomes `tuple[Fraction, Fraction]`; every row that
   passes `(2.5, 97.5)` passes `(LEVEL_CI95_LO, LEVEL_CI95_HI)` instead, and its new `k = 2` test's pair is
   `(Fraction(1, 80), Fraction(79, 80))` — a pair `permille: int` could not express and this one
   can. Its **both-bounds-move-outward** assertion survives intact and is checkable in advance: at
   `B = 10 000` the ranks move from 250/9 750 to 125/9 875, measured. **(d)** Table G's two
   residuals are restated over the surviving spelling of `:159`, which after Table C is
   `percentile(means, level=…)` and carries neither retired literal — so the pair must be re-derived
   rather than re-scoped, and the two tables' collision is named on both rows the way Table D/E's is
   on **their own shared site** *(v1.22: v1.18 wrote that as a bare `stats.py:263`, which is a blank
   line at `d71c83e`; the plan's tables own those pins and re-base them, so this note names the
   rows)*. **(e)** Table C's residual gains the `stats.py` half plan-gate P8-3 asks for,
   over all three sites, plus §11.10(3)'s package-wide command as the one that cannot be passed by a
   half-application. **(f)** Nothing here reopens Table G's *"required rather than defaulted"*
   argument, which is unaffected by the element type.

7. **New at v1.20: the whole plan-side consequence of the loop ruling, section by section — this is
   the one place it is stated** *(plan-gate P13-2)*. Nothing here is optional and nothing is
   deferred; every item is closeable in the plan today, because no implementation unit has built
   any of it. **Nothing stored becomes unreadable:** `ItemTiming` is designed and unbuilt (DC-11 has
   not landed — `modelbench/results.py` still carries a bare `ItemResult.latencyMs`), no run record
   has ever been written with a timing block, and no `benchSchemaVersion` moves. **No run budget
   moves either** — the ruling adds no call, no probe and no pass over the data.

   - **§3.3.** State that `callCount` per item is `I(t)` for a `tool-caller` and `1` for every other
     role, and that this is a *consequence* of `maxIterationsPerTurn`, not a second declaration. No
     new manifest key.
   - **§3.5 (`index.csv`).** Add `latencyTimedCount`, `latencyItemCount` and **`callCount`**;
     §11.8 gives the reason the third is not cosmetic.
   - **§3.6, the FR-11 table.** Four rows move and one is added. `latencyMs` is an **item** figure
     and on a `tool-caller` it is the turn's wall clock, tool dispatches included — say so in the
     row, because "client wall clock" no longer disambiguates it. `ttftMs`,
     `prefillMsPer1kPromptTokens` and `tokensPerSecond` are **call** figures aggregated over calls
     with a call denominator. `unexplainedMs` is §11.5.1's **sum over the item's calls**, `None`
     unless every call yields a gap. New row: `callCount`, `int`, `len(ItemTiming.calls)`, asserted
     equal to `TurnTrace.iterations`.
   - **§3.6, the withholding dispositions.** Withholding is at the unit of the figure: `latencyMs`
     per item, the sibling coverage per call, and **no call is ever withheld for load** (§11.4's
     measurement is unchanged by the loop). Add the one sentence that closes the residual an
     implementer will otherwise go looking for: a reload *between* two iterations costs nothing
     because LM Studio loads JIT on the request, so it lands inside the next call's own gap
     (§11.5.1). **And one the loop creates that P13-1's disposition split does not cover:** a turn
     whose loop terminated on a call that timed out or returned nothing has an **incomplete** wall
     clock — it measures only the iterations that happened to succeed — so it is **not timed**. Its
     `latencyMs` is withheld under `timeout`/`no_response` exactly as a single-call item's is, and
     the partial turn wall clock is never stored as a measurement. A **cap-hit** turn is the
     opposite case and must not be folded into it: every one of its `maxIterationsPerTurn` calls
     returned, its wall clock is complete, and it is timed and ranked like any other item.
   - **§3.8.4.** The runner builds one `ItemTiming` per turn from `TurnTrace.chatResults`, in
     order — the loop bullet currently ends at `capHit` and says nothing about timing at all. Narrow
     the *"No sizing consequence"* clause to what §4.5.2 now says: the loop is not a *new* cost, and
     the derived minutes are a floor bounded above by `maxIterationsPerTurn` (plan-gate P13-7).
   - **§4 S1, `ItemTiming`.** New shape, and it is **smaller** than v1.10's, not larger:
     `(wallClockMs: float | None, calls: tuple[CallTiming, ...], withheldFor: … | None)`, with
     `CallTiming = (wallClockMs, ttftMs, generationMs, promptTokens, tokensPerSecond)`, each field
     `| None` and never `0`. `unexplainedMs` and `callCount` become **derived properties** over
     `calls`, exactly as `ItemResult.latencyMs` is already a derivation over `ItemTiming` — §7 rule
     4's preference for a derivation over two stored copies, applied to the same record a second
     time. The five scalars v1.10 put on `ItemTiming` move onto `CallTiming` and have no second
     home. An item that returned no response still carries an `ItemTiming`, now with `calls == ()`.
   - **§4 S2, `LatencyBlock`.** Add **`callCount`** (`Y_calls`) beside `latencyItemCount`.
     `statsCoveredCount` becomes a count of **calls** and rule (iv)'s recomputation is a sum over
     each item's `calls`; rule (iv)'s surviving bound is
     `statsCoveredCount ≤ Y_calls − (calls that returned no response)` (§11.4). Rule **(iv-b)**'s
     gate takes `X = statsCoveredCount, Y = callCount` — **not** `latencyItemCount`, which is the
     one-word substitution that will otherwise ship. Rule **(iv-c)** is restated per call, both
     halves intact. Rules (ii), (iii), (v) and (vi) are item-level and do not move; state that,
     because "one unit changed" is how the other five get changed by accident.
   - **§4 S2, `unexplainedMsMax`.** It is a maximum over the items that had a reading, takes **no**
     coverage gate, and is printed with that count. Rule (i)'s sentence currently folds it in with
     the three medians under (iv-b); split it out.
   - **§5 test 15b.** Three cases the existing list cannot express, all offline against the stub
     clock and stub LLM: **(a)** a warm 3-iteration turn, and one at the cap of 8, report **no**
     model-load contamination and keep their `latencyMs` — this is the regression P13-2 names and
     every current timing fixture is single-call, so it is invisible without it; **(b)** a
     3-iteration turn one of whose calls carries a 3 485.6 ms gap **is** withheld, counted under
     model load, with the other two calls' sibling figures kept; **(c)** a 3-iteration turn one of
     whose calls has no `stats` has `unexplainedMs is None`, is **not** withheld, and contributes
     its two readable calls to `statsCoveredCount`.
   - **§5 test 10b.** Beyond P13-5's own fix: assert `callCount == len(chatResults) ==
     TurnTrace.iterations` and `ItemTiming.wallClockMs >= Σᵢ wallClockMsᵢ` on a multi-call fixture.
   - **Appendix A.** `ItemTiming`, the new `CallTiming`, and `LatencyBlock`'s new `callCount`.
   - **One shipped docstring, and it is a one-word sweep, not a rewrite.**
     **`_coerce_finite_float`'s docstring** in `modelbench/lmstudio.py` quotes the detector's gap as
     `latencyMs - (ttftMs + generationMs)` — the module's only occurrence of that token, so the
     sweep is enumerable rather than remembered: `grep -c latencyMs modelbench/lmstudio.py` → **1**
     today, **0** after it *(v1.22, plan raise R-2: v1.20 pinned this as `:225`, which is `:247`
     four days later; the plan pinned it by symbol and count and was right to)*.
     The operand at that level is the **call's** `wallClockMs`; `latencyMs` is the item's admitted
     figure and after this ruling is not an operand of the gap at all. The docstring's actual claim
     — that a non-finite value would send the gap to `-inf` and the detector could never fire — is
     unaffected and correct.

8. **New at v1.22: R-1's resolution, as the delta on ask 7 and not a rewrite of it.** Ask 7 landed
   in plan v1.27; every item of it stands except where named here. Nothing below is blocked and
   nothing is deferred — `ItemTiming`, `CallTiming` and `LatencyBlock`'s new fields are all still
   unbuilt — `git grep -n -E 'ItemTiming|LatencyBlock' d71c83e -- 'model-bench/modelbench/*.py'`
   returns **two** lines and both are prose cross-references (`convo.py:48`, `results.py:819`), so
   no stored record moves and no `benchSchemaVersion` does.
   - **§4 S2, `LatencyBlock`.** `callCount` is unchanged — completed calls, `Σ_items len(calls)`,
     the pin and rule (ii)'s third assertion exactly as delivered. What ask 7 called `Y_calls` is
     **not** that number: add **`callAttemptedCount`** as a **derived** property,
     `callCount + latencyWithheldForNoResponse` (§11.4), carrying the
     at-most-one-non-returning-call lemma and its reversal trigger (a within-turn retry) where an
     implementer reads it, plus a fourth assertion under rule (ii) — recomputed from `run.items`
     over the **dispositions**, it equals what the two stored counts give. Do not store it as a
     seventh count. *(Rejected: renaming `callCount` to `callCompletedCount` so the pair cannot be
     confused at all. It is the strongest fix and it is refused on price — a rename across six plan
     sections, Appendix A, an `index.csv` column and the tests, whose **half**-application is this
     component's signature defect and would be worse than the ambiguity. The guard replaces it:
     §11.10 (7d)'s fixture makes the three counts pairwise distinct, so the substitution is caught
     rather than avoided. Reversal trigger: one more review or implementation that confuses the
     two.)*
   - **§4 S2 rule (iv-b) — the one place v1.27 must change what it bolded.** The gate is
     `X = statsCoveredCount, Y = callAttemptedCount`: **not** `callCount`, and still not
     `latencyItemCount`. v1.27 refused the second substitution and stopped one short of the first,
     which is the same one-word failure one count over. All three are pairwise distinct on §11.10
     (7d)'s fixture, which is what makes the refusal testable rather than editorial.
   - **§4 S2 rule (iv).** The bound does not move — `statsCoveredCount ≤ callCount` is right. What
     goes is the hedge: the parenthetical reasoning that ask 7's subtrahend is identically zero was
     true only of the netted reading, and the two spellings are now one identity (§11.4).
   - **§3.3.** One word: `callCount` is `I(t)` for a `tool-caller` and **at most** `1` for every
     other role — `0` where the call did not return, `0` throughout a `deterministic` arm. The
     *attempt* count is exactly `1` per item on any model-calling arm, and that is what `Y_calls`
     sums.
   - **§3.5 (`index.csv`) — no new column.** The CSV carries no `statsCoveredCount`, so an
     attempted-call column would be a denominator with no numerator in its own row; `callCount`
     keeps the job ask 7 gave it (which estimand the `latencyMs*` cells hold).
   - **§3.6 / §3.8.4 — one sentence, and it is plan-gate P15-3's.** The precedence this
     derivation rests on must be **written**, not left implied by §3.8.4's table: a turn that ended
     on a raise takes `withheldFor` from the failing disposition, never `"load"`, whatever an
     earlier call's gap showed — ruled in §11.4 with the reason, because the counter identity
     `Y_calls = callCount + latencyWithheldForNoResponse` is off by the overlap without it. That
     is the answer to P15-3's *"nothing says which wins"*, and §11.10 (7d)'s two-route assertion is
     what keeps it true rather than remembered.
   - **§5 test 10b and Appendix A.** Test 10b's multi-call assertion gains the failed-turn leg
     (§11.10 (7d)); Appendix A's `LatencyBlock` row gains the derived property beside `callCount`.
   - **R-2, whose durable half is the citation form.** Ask 7's last item is corrected in place.
     Measured at `d71c83e`: of this note's three unpinned `modelbench/*.py` line cites **two no
     longer resolve** (`lmstudio.py:225` → the text is at `:247`; `stats.py:263` → a blank line),
     while **both** cites pinned to an explicit sha resolve exactly — `results.py:573` and
     `stats.py:296` at `5878014` are still the two `_percentile` copies, days after that symbol left
     the package entirely. So a citation into shipped code is **a symbol plus an enumerating command
     with its count, or a line number pinned to a named sha — never a bare line.** The five
     `stats.py:159` cites still resolve and are left alone; they resolve because the edit they
     prescribe landed on that very line, which is luck rather than form.

Two consequences that are the standing sweep obligation of plan §7 rather than new asks:
**§3.6's `latency n = X of Y` sketch and §5 test 15b's assertion are superseded by §11.7's slots** —
the plan cites them, as §3.9 already cites §3.2e's verdict strings, and does not restate them; and
**test 15b's scenario still renders a figure** (§11.6(2)), so the test remains satisfiable under
either G3-4 disposition.

### 11.10 Evaluation design — how to prove this implementation right

Pure functions and rendered strings; no network, no model. Asserted **exactly** — every quantity
here is an integer, a string, or a value copied from the input, so no tolerance is appropriate and
any tolerance would hide the defect it was meant to catch.

1. **Rank fixtures.** `rank(LEVEL_P95, X)` for `X ∈ {1, 12, 19, 20, 38, 40, 85, 100}` equals
   `{1, 12, 19, 19, 37, 38, 81, 95}`, and `rank(LEVEL_P50, X)` equals
   `{1, 6, 10, 10, 19, 20, 43, 50}`. Integer equality. This is §11.3's table and it is the one test
   that pins the estimator itself. **Both rows were re-measured at the rational level form** *(v1.18
   — a unit change invalidates a fixture table until it is re-run, §3.2c's trap)* and neither moved:
   `19/20` and `1/2` are exactly the numbers `950` and `500` named, and only the spelling moved.
2. **The bin-edge guard, and it is now two assertions, because one float spelling is invisible to
   the other's fixture** *(v1.18)*.
   **(2a) The rank expression.** `percentile(range(25), level=Fraction(7, 25))` returns the **7th**
   value — `6` — where `math.ceil(float(level) * X)` returns the 8th. Basis:
   `float(Fraction(7, 25)) * 25 == 7.000000000000001`, measured. **This fixture cannot catch the
   numerator-first spelling**, which returns 7 here and diverges nowhere over any level this tool
   reaches (§11.2.1) — so its scope is stated with it: it guards against a *level-first*
   simplification and against nothing else, and saying so is the difference between a guard and a
   guard believed to be wider than it is.
   **(2b) The level's construction.** `percentile(range(20), level=Fraction(1, 20))` returns the
   **1st** value — `0` — and the same call with the level built as `Fraction(0.05)` returns the
   **2nd**. Measured: those two levels disagree on **1000** of `X ≤ 20 000`, first at `X = 20`. The
   assertion that carries the rule rather than its symptom is on the level itself: a `k = 2` family
   at α = 0.05 derives exactly `(Fraction(1, 80), Fraction(79, 80))`, asserted as Fraction equality
   with no tolerance, which fails the moment `Fraction(str(alpha_family))` becomes
   `Fraction(alpha_family)` (§11.2.2).
3. **One implementation, package-wide** *(scope corrected at v1.18 — the v1.17 wording named
   `modelbench.results` only, and plan-gate P8-1's cheapest fix read the gap as licence to leave the
   family-dependent call site on the rejected estimator)*. **No module in `modelbench` defines a
   percentile or quantile helper; `stats.percentile` is the only one, and `results.py`'s percentile
   *is* that object** (identity, not equality of behaviour). Checkable in one line:
   `grep -rEn 'def [A-Za-z_]*(percentile|quantile)' modelbench tests --include='*.py'` returns
   **2 today** — `stats.py:296` and `results.py:573`, both the rejected `int(round(…))` estimator —
   and must return **1**, the public `percentile`, after the edit. It is stated over the *estimator*
   rather than over one module's private helper because that is the shape of review M27's defect,
   and a rule scoped to one module licenses the second copy in the other.
4. **The identity floor.** At `X = 19` the record's `latencyMsP95` is `None` and the rendered block
   contains `max = ` and slot 5's `max` variant; at `X = 20` it contains `p95 = ` and
   `Timed calls slower than the p95 figure: 1 of 20.` The boundary is asserted on both sides,
   because a floor tested on one side is a floor with an untested inequality.
4a. **The identity floor's reach, as a constant rather than a sentence** *(v1.20, §11.3)*. Two
    assertions, both integer. **(i)** `IDENTITY_FLOOR_MAX_Y == 21`, bound to the sweep that
    produces it: over `Y ≤ 200`, the set of `Y` for which some `X` passes the level floor **and**
    has `r95(X) == X` is exactly `1…21`, and at `Y = 21` that `X` is uniquely 19. **(ii)** No
    declared pack is in range — `min(Y over the loaded packs) > IDENTITY_FLOOR_MAX_Y`, driven
    through the pack loader rather than over a hand-written list, so a pack added later fails the
    assertion rather than the paragraph. Together these are what license §11.3's *"`max` is
    unreachable on every declared pack"* and §11.9 ask 4's narrowed argument; without them both are
    prose reach claims of exactly the kind the component's convention forbids.

5. **Level-floor boundaries, at `Y = 38`.** `X = 36` → both figures present; `X = 35` → `p50`
   present and the tail `None`; `X = 34` → both `None`. Exactly §11.5's table, and it is the test
   that fails if the 5-point tolerance is silently moved. **The same table's p50 column gates the three sibling
   medians** (§4 S2 (iv-b)) — read with `X = statsCoveredCount` and, **v1.20, `Y = callCount`,
   never `latencyItemCount`**: the gate is over the coverage of the unit those three figures are
   computed in, and on a single-call pack the two denominators are equal, so only a multi-call
   fixture distinguishes the correct gate from the one an implementer will write. They are asserted
   to refuse **together**, never one of them, which is the property their shared coverage number
   claims *(v1.13)*.
6. **The clean-run invariant.** For every `X` in `1…200`, a run with `X == Y` has both figures
   present. §11.6(1) is a theorem; a fire means the gate is wrong, not the data.
7. **The two denominators (§11.4).** An item withheld by the guard contributes to **no**
   `latencyMs` aggregate, and its calls **do** contribute to `ttftMs`, prefill and
   `tokensPerSecond`; the rendered block prints **both** denominator lines, they differ
   (`n = 36 of 38 items` against `n = 38 of 38 calls`), **and their nouns differ** *(v1.20)*. A test
   asserting one denominator for all four fields pins v1.9's withdrawn ruling and must fail.
   **And the three siblings are co-present** *(v1.13, §11.4; per **call** from v1.20)*: the count of
   calls contributing to each of the three medians is one number and it is `statsCoveredCount`. A
   call carrying `stats` but no usable `usage.prompt_tokens` is absent from that count **and** from
   all three medians while its sibling calls in the same item stay, and **the run does not raise**
   *(v1.14, §11.4's disposition)* — the assertion is on the exclusion being applied to all four
   places at the call level, never on the call being impossible.
7a. **The in-call reload detector (§11.5.1), and every one of these needs a multi-call fixture
   except the first** *(rewritten at v1.20)*. **(i)** A single-call item whose gap exceeds 1 000 ms
   has its `latencyMs` withheld and is counted under model load; one at 7.6 ms — the largest warm
   gap measured — does not. Both sides of the threshold, since a threshold tested on one side is an
   untested inequality. **(ii)** A **3-iteration** item whose three calls are each warm — gaps at
   the measured warm ceiling — is **not** withheld, and neither is one at the cap of 8. This is the
   regression plan-gate P13-2 names: it fails under a gap taken against the turn's wall clock and
   passes under the sum, and every timing fixture that exists today is single-call and cannot see
   it. **(iii)** A 3-iteration item one of whose calls carries the measured 3 485.6 ms cold gap
   **is** withheld, counted under model load, with the other two calls' sibling figures kept.
   **(iv)** A 3-iteration item one of whose calls has an unreadable gap has `unexplainedMs is None`
   and is **not** withheld — the partial sum is never formed. **(v)** The margin claim, as a
   constant rather than a sentence: `WARM_GAP_CEILING_MS * cap < threshold` for the
   `maxIterationsPerTurn` of **every** pack the suite loads, with the quoted margin computed from
   those two constants. A test that asserts the number `16` rather than computing it is the defect
   this item exists to prevent.
7b. **The censoring predicate, all four branches (§11.5.1).** A run whose one load-withheld item has
   a wall clock **below** the largest timed one renders slot 3's `censoringExact == false` variant;
   the same run with that wall clock above renders the true variant; a run whose only withholding is
   a timeout renders the **true** variant without reading any withheld wall clock; a
   load-withheld item whose wall clock is unreadable renders the **false** variant; and a run whose
   only withholding is a **no-response failure that is not a timeout** renders the **false** variant
   *(v1.14)* — the branch plan v1.10's merged `withheldFor` category could not express. The last two
   are the fail-safes, and they are the branches an implementer will skip because no fixture produces
   them by accident.
7c. **A surface that returns no `stats` (§11.7).** On such an arm `statsCoveredCount` is `None`, the
   rendered block contains **no** `ttft/prefill/tokens-per-second` line and no item carries an
   `unexplainedMs`; on a chat arm with `statsCoveredCount == 0` that line **is** rendered. Asserted
   on the **absence** — a line that should not exist is invisible to a test that only checks the
   lines that should.
7d. **Three counts, and they are separable only on a fixture where all three differ** *(v1.20;
   re-derived at v1.22, because the v1.20 fixture could not separate two of them)*. `Y`,
   `callCount` and `Y_calls` are three numbers this block prints or gates on, and v1.20 set
   `Y_calls == Σ callCount` **by construction** — so a `callCount`/`Y_calls` substitution passed
   that fixture silently, which is how the netted denominator R-1 found reached plan v1.27 as a
   bolded instruction. A guard hole first, a wrong number second. **One `tool-caller`-shaped run of
   three items** — one `replied` after 3 returning calls, one that returned 2 and then timed out,
   one whose first call was `server-rejected` — gives `Y = 3`, `callCount = 5`, `Y_calls = 7`,
   `<MT> = 2` and `X = 1` timed item: pairwise distinct, so each of the three one-word
   substitutions reddens on its own. On it: `callCount == len(ItemTiming.calls) ==
   TurnTrace.iterations` **per item** (`3, 2, 0`); `ItemTiming.wallClockMs >= Σᵢ wallClockMsᵢ` on
   the timed item, the difference attributed to the harness and subtracted nowhere, and `is None`
   on the other two; **`Y_calls == callCount + latencyWithheldForNoResponse`, recomputed over the
   `TurnTrace` dispositions and asserted against the block's value recomputed over
   `ItemTiming.withheldFor`** — two routes, so what the pin actually tests is the runner's
   disposition→`withheldFor` map, and a turn that ended on a raise but was filed under `"load"`
   fails it; `Y_calls > Σ callCount`, which the netted implementation cannot satisfy; the rendered
   block's wall-clock lines read `of <Y> items` and its sibling line `of <Yc> calls`; and slot 6's
   second sentence renders with the mean computed from `<Yc>/<Y>`. **Mutate both ways**: netting
   `Y_calls` to `5` and inflating it to `8` (a second attempt counted for one failed turn) must
   **each** redden — a fixture that catches only the shrink is being covered by something else for
   the widen. **Two negatives in the same suite.** *(a)* The single-call run **with no
   failures**: `Y_calls == Y == callCount`, the three numbers coincide, the two nouns still differ,
   and slot 6's second sentence is **absent** — asserted on the absence, for (7c)'s reason.
   *(b)* A chat-surface run in which
   **every** item's only call failed: `callCount == 0`, `Y_calls == Y`, `statsCoveredCount == 0` (a
   real `0`, not `None` — (iv-a) keys on the surface, not on what arrived), the three medians
   **refused by the gate** rather than absent for want of input — which is what rule (i)'s
   discriminator says when the count is a number — and slot 2's second line rendering
   `0 of <Y> calls`. Under a netted `Y_calls`
   that same run prints `0 of 0` and evaluates §11.6's gate at `Y == 0`, where `100·r ≥ 45·Y` is
   vacuously true; this is the fixture that forbids both.

8. **String rendering.** Five blocks rendered against fixtures and asserted **verbatim**, the way
   §7.2's resolving-power line is, with §11.7's measured sample as the fixture:
   `(X=Y=38)` → `p50 = 78 ms, p95 = 112 ms`, `1 of 38` slower ·
   `(X=Y=12)` → `p50 = 77 ms, max = 104 ms`, the `max` case ·
   `(X=36, Y=38)` → `p50 = 76, p95 = 105`, levels 92.1 / 47.3 ·
   `(X=35, Y=38)` → `p50 = 76` only, levels 89.4 / 47.3 ·
   `(X=34, Y=38)` → both refused, levels 86.8 / 44.7.
   Plus one with `MT > 0`, so the cause split is exercised rather than assumed, and one on the
   **false** branch of §11.5.1's predicate, so both slot-3 strings are pinned verbatim rather than
   one of them *(v1.12)*. **Plus, from v1.20, one multi-call rendering** (`Y_calls > Y`), pinning
   the `calls` noun on the second denominator line and slot 6's second sentence verbatim. All five
   existing renderings change in two places and must be **re-pinned rather than re-derived**: slot
   2's second line now ends `of 38 calls`, and slot 3's true branch has lost the *"tens to hundreds
   of milliseconds"* clause (§11.7). A fixture table is invalidated by a wording change exactly as
   it is by a unit change — §3.2c's trap, which this note has now paid for twice. **The fixture is the measured sample, so the ranks and levels are checkable
   by hand against it** — which is the property a hand-invented fixture does not have.
9. **The unit guard.** Every printed figure in the block carries ` ms`, and no latency in the block
   is printed in seconds. The ms/s boundary sits one field away from `coldLoadSeconds`, and a figure
   printed in the wrong unit is a defect no reader can detect from the report.
10. **Empty input, and the level's own preconditions.** `percentile([], level=LEVEL_P95)` raises;
    `latency_summary` over a run with no timed item renders the `X == 0` variant. The
    absent-or-present decision lives in exactly one place. *(v1.18)* Three more refusals are
    asserted at the same boundary, each one line: a `float` level raises `TypeError`, a level
    outside `(0, 1]` raises `ValueError`, and `paired_bootstrap` with `levels[0] >= levels[1]`
    raises — the transposed pair being the one error that otherwise returns a plausible inverted
    interval (§11.2.2).

**Acceptance:** every item above passes, and no stored record carries a `latencyMsP95` whose attained level is
below 90.0 or whose rank equals its `X` — both checkable over `results/runs/` at any time, and
together they are the invariant §11.8 sells.
