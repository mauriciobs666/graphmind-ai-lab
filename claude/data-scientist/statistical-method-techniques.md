# Statistical & evaluation-design techniques — on-demand

> **On-demand knowledge base for `data-scientist`.** Situational statistics/eval-design techniques
> — gate slack, judge-bias metric selection, repeated-measures analysis units, exact-arithmetic and
> bootstrap traps — consulted when that specific method question arises, not needed for every
> consult. Live-verified LM Studio/small-model facts for this lab's stack live separately in
> `claude/data-scientist/lm-studio-model-notes.md`; general method doctrine lives in
> `data-scientist.md`.
>
> Origin: extracted 2026-09-17 from `data-scientist.md` as part of K-030 Stage 0 (interim
> knowledge-base relief) — a prompt restructure, not a kaizen distillation.

## A zero-tolerance gate needs its slack checked against the metric's own confidence interval

Before blessing a **hard, zero-tolerance pass/fail gate** on a small golden set, compute the
one-unit delta (`1/n`) and compare it to the metric's own confidence-interval width at that `n` —
a gate with no slack below a near-ceiling baseline fails on measurement noise as readily as on a
real regression, and can't register a genuine improvement either.

## A gated probe set should still report each probe's individual outcome, not just the boolean

When a probe **set** is gated with AND/OR logic, still report each probe's individual outcome in
the summary prose, not just the boolean bloc result — the aggregate can mask a real
partial-failure pattern (e.g. 2-of-3 probes failing) at exactly the small-N scale where the
qualitative read matters most.

## A regression-fixture table's printed precision caps the tolerance anyone can assert against it

When you publish a **regression-fixture table** next to an assertion tolerance, the table's
printed precision **caps** the tolerance anyone can assert against it — publish the fixtures two
to three orders finer than the tolerance you mandate, and as exact rationals wherever the
quantity has one, or the implementer is forced to substitute a looser check than your note
demands.

## A verdict-biased judge needs class-conditional rates, not a symmetric agreement metric

For a judge deliberately biased toward one verdict (e.g. bias-to-suspend / abstention-favoring by
design), a symmetric agreement metric (Cohen's κ, plain accuracy) mis-gates it: specificity near
its ceiling by construction decouples κ from the error class that actually matters, and κ
additionally moves with hand-picked case-mix prevalence. Gate on **class-conditional rates** (e.g.
false-advance rate + advance-recall for a suspend-biased judge) and demote κ/accuracy to reported
diagnostics with marginals, not the gate itself.

## A self-preference caveat must be split by whether the judge scores fixed content or its own output

When the judge collapses onto the same model as the agent-under-test, don't write one blanket
self-preference caveat over everything that sub-pass reports — split it: a sub-pass judging fixed,
independently-authored content carries little self-preference risk (still a legitimate
rubric-following signal), while a sub-pass judging the model's own live output does; one
undifferentiated caveat lets a reader extend trust from the first kind to the second.

## In a repeated-measures design, name which count is the analysis unit

In a repeated-measures design (several turns/calls/items per independently-sampled unit — a
multi-turn conversation, a multi-query session), name which count is the **analysis unit**: `n`, a
design-effect (DEFF) adjustment, and a paired test (McNemar and similar) are computed over the
independently-sampled unit, never over the larger raw per-turn/per-item count, which is instead
the right denominator for a coverage or latency **rate**. The two counts are easy to swap silently
in a benchmark's own terminology — treat any such benchmark's "how many X" as ambiguous until
you've confirmed which of the two it means.

## Two non-overlapping marginal intervals is not a difference test at small n near a ceiling

Two marginal intervals failing to overlap is not a difference test, and at small n against a
near-ceiling baseline it is inert rather than conservative: at n ≤ 40 with a baseline ≥ 0.90, no
candidate score separates at all — not even a perfect one. Decide a **paired** binary comparison
with **McNemar's exact test** and size the effect with a paired difference interval (Newcombe
MOVER-D over the same Wilson bounds); no paired result reaches α = 0.05 on fewer than **6 net
discordant wins**, so 6/n is the observable floor and ≈7.7/n the 80%-power minimum detectable
difference.

## A published statistic's arithmetic and provenance label are both part of the claim

Compute every rank, bin or quantile **index** in exact integer/rational arithmetic
(`-(-num * X // den)`), never a float multiply then `ceil`/`floor`: the float spellings diverge
from the exact integer on reachable inputs, and *operand order* decides how often — so quoting one
spelling's divergence count while pinning a different one claims a guard you do not have. Verify
such a guard by sweeping **the exact expression the code uses**; an algebraically equivalent
spelling reports zero misfires for a truncation the shipped one gets wrong. A provenance label is
part of that claim too: compute the which-instrument-produced-this-number audit **before** any
value-modifying post-transform (support clamp, rounding, normalisation), never after — the
transform can collapse two distinct candidates onto one printed value, and a tie-break over the
transformed pair then names a source that produced neither. Moving it is free wherever the
transform is bound-wise and monotone: it commutes exactly with a bound-by-bound `min`/`max`
composition, so no printed number moves — only whether the audit is answerable.

## A seeded percentile bootstrap over a discrete outcome still moves with row order

A seeded percentile bootstrap over a discrete outcome is not a function of the data alone. Its
bound jumps by a whole atom (`1/n`) wherever the target percentile lands within Monte-Carlo error
of an atom boundary — so at a *fixed* seed it still moves with the **row order** of the input
vector, because `random.Random.choice` draws an index. The instability is conditional, not
universal: predict it from the exact-CDF distance to the nearest atom against
`sqrt(p(1−p)/B)`, note that a smaller `B` makes it more common rather than less, and where the
resample distribution has a closed form — a paired binary table's is exactly multinomial —
compute the percentile from that instead of resampling.
