# Small-Model Catalog Sweep — Consolidated Report

## What this sweep can and cannot prove

This is a 68-run, single-session sweep across 20 small models and five evaluation packs, with
per-pack sample sizes in the tens (12 to 40 effective units, depending on the pack — see the table
below, each row copied verbatim from that pack's own report). At this scale, every pack's own
resolving-power sentence says the same thing in different numbers: the design can reliably catch a
**large** collapse — a model dramatically worse than the field — but it cannot certify
fine-grained ranking among the closely matched 3-4B-class models that make up most of this list. A
model ranked #1 in the index above is, in almost every case, the top **point estimate**, not a
model proven better than #2, #3, or #4 at this sample size; read every "top model" cell, and every
recommendation below, with that qualifier attached.

Each pack's own resolving power, verbatim from that pack's own report and never combined across
packs (the same "list side by side, compute nothing across rows" discipline as the index above):

| Pack | Metric | n (effective units) | Floor (no verdict below) | MDD80 (80% power) |
|---|---|---|---|---|
| `embedder-graphrag-retrieval` | mrr | 38 | 15.7 pp | 20.1 pp |
| `guard-judge-understanding` | falseAdvanceRate | 40 | 15.0 pp | 21.9 pp |
| `guard-judge-understanding` | falseSuspendRate | 30 | 20.0 pp | 28.7 pp |
| `nlq-structured-query` | layer1ExactMatchRate | 34 | 17.6 pp | 22.3 pp |
| `tool-caller-shop-assistant` | cleanThroughTurnH | 12 | 50.0 pp | 57.8 pp |
| `chat-responder-grounded-answers` | groundingRate | 30 | 20.0 pp | 25.1 pp |

`tool-caller-shop-assistant` is the extreme case, entirely on its own numbers: at n=12 scripts,
the best-case gap needed for 80% power (57.8 pp) exceeds the entire observed range in that pack's
own table (0.500 down to 0.000, i.e. 50.0 pp) — no outcome in that pack could have reached 80%
power, and only the single largest possible gap sits at the edge of what could reach nominal
significance at all.

## Index

| Pack | Metric | Top model | Value | 95% CI | Report |
|---|---|---|---|---|---|
| `embedder-graphrag-retrieval` | `mrr` | `text-embedding-qwen3-embedding-4b` | 0.6610 | [0.5491, 0.7684] | [embedder-graphrag-retrieval-rank-20260920-02.md](embedder-graphrag-retrieval-rank-20260920-02.md) |
| `guard-judge-understanding` | `falseAdvanceRate` | `google/gemma-4-e2b` | 0.0000 | [0.0000, 0.0876] | [guard-judge-understanding-rank-20260920-02.md](guard-judge-understanding-rank-20260920-02.md) |
| `guard-judge-understanding` | `falseSuspendRate` | `gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf` | 0.0000 | [0.0000, 0.1135] | [guard-judge-understanding-rank-20260920-02.md](guard-judge-understanding-rank-20260920-02.md) |
| `nlq-structured-query` | `layer1ExactMatchRate` | `mistralai/ministral-3-3b` | 1.0000 | [0.8928, 1.0000] | [nlq-structured-query-rank-20260920-02.md](nlq-structured-query-rank-20260920-02.md) |
| `tool-caller-shop-assistant` | `cleanThroughTurnH` | `qwen/qwen3-4b-thinking-2507` | 0.5000 | [0.2538, 0.7462] | [tool-caller-shop-assistant-rank-20260920-02.md](tool-caller-shop-assistant-rank-20260920-02.md) |
| `chat-responder-grounded-answers` | `groundingRate` | `qwen2.5-coder-3b-instruct` | 0.8000 | [0.6269, 0.9049] | [chat-responder-grounded-answers-rank-20260920-02.md](chat-responder-grounded-answers-rank-20260920-02.md) |

## Insights and recommendations

### embedder

**No single model is statistically distinguished from the others in the top tier — choose by
footprint/latency budget, not by the nominal top row.**

Within `embedder-graphrag-retrieval` (mrr, n=38, resolving power: floor 15.7 pp / MDD80 20.1 pp),
the four in-scope embedding models cluster tightly: `text-embedding-qwen3-embedding-4b` (0.6610,
CI [0.5491, 0.7684], 55 ms p95, ~2.2-2.5 GB Q4_K_M) leads the point estimate, but
`text-embedding-nomic-embed-text-v1.5` (0.6407, 17 ms, ~0.08 GB), `text-embedding-granite-
embedding-278m-multilingual` (0.6317, 12 ms, ~0.28 GB) and `text-embedding-qwen3-embedding-0.6b`
(0.6278, 35 ms, ~0.6 GB) all sit within 3.3 pp of it — well inside this pack's own 20.1 pp
resolving power, so none of the four can be called better or worse than another from this report
alone. The bm25 lexical baseline trails at 0.5734; even that gap (up to 8.8 pp above bm25) does not
clear the pack's own 15.7 pp floor, so this report does not statistically certify that any
embedding model beats bm25 either — directionally consistent, not proven, at this n.

Given that indistinguishability, footprint and latency become the deciding factors:
`nomic-embed-text-v1.5` and `granite-embedding-278m-multilingual` deliver a mean MRR within ~2-3 pp
of the nominal top model at roughly 1/8 to 1/28 the footprint and 3-5x lower p95 latency (55 ms vs.
17 ms and 12 ms respectively — ~3.2x and ~4.6x), making either a stronger default for a
footprint- or latency-constrained deployment.
`qwen3-embedding-4b` is the right choice only if raw point-estimate score is prioritized ahead of
footprint, with the 20.1 pp resolving-power caveat explicitly accepted.

Caveat (restated from the per-pack report): this pack's item set has a recall@10 ceiling of 37/38
— only one item is available to "win" beyond that ceiling — so this ranking can catch a materially
worse embedder but can never certify a better one, independent of the sample-size point above.

### guard-judge

**`qwen/qwen3-4b-2507` — the only model in this pack never certified worse than any rival on
either co-equal metric, and Holm-confirmed better than 10 of the other 15 on at least one axis.
`nvidia/nemotron-3-nano-4b` remains a plausible, descriptively similar second option, though the
family test never actually compared it to the reference. No single-metric "top row" model is a
safe pick alone.**

`guard-judge-understanding` has two co-equal class-conditional metrics with no single headline
(falseAdvanceRate, falseSuspendRate — restated from the per-pack report). Reading the index above
in isolation is actively misleading here: the top rows on falseAdvanceRate
(`gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf` and `stablelm-zephyr-3b` also lead
falseSuspendRate at 0.000) look excellent on the metric that ranks them first, but both sit at
0.950 falseAdvanceRate (38/40) on the *other* metric within this same report — a near-always-
advance policy that trivially minimizes false suspends by almost never suspending.
`prism-ml/bonsai-27b` shows the opposite trivial policy (0.000 falseAdvanceRate, 1.000
falseSuspendRate, 30/30) and is this pack's own latency outlier at 12167 ms p95 (the rest of the
field is 599-4975 ms) — not a viable choice on either axis despite a perfect score on one metric.

Reading both metrics together from this report's own per-arm rate table: `qwen/qwen3-4b-2507`
(falseAdvanceRate 0.100 [4/40], falseSuspendRate 0.133 [4/30], 1079 ms p95, ~2.2-2.5 GB) and
`nvidia/nemotron-3-nano-4b` (falseAdvanceRate 0.100 [4/40], falseSuspendRate 0.167 [5/30], 3792 ms
p95, ~2.2-2.5 GB) are the *only* two models in the 16-row table with both rates below 0.20 — every
other model has at least one of the two at or above 0.20. Between the two, `qwen/qwen3-4b-2507`
has the latency edge (1079 ms vs. 3792 ms) at the same footprint class.

This pack's reference-anchored family (anchored on `qwen/qwen3-4b-2507`) adds real, certified
weight to that reading rather than undermining it. Its diff column is signed "positive = candidate
better" for these two `_LOWER_IS_BETTER` metrics by deliberate, purpose-built design
(`modelbench/report.py`'s `_polarity_corrected`, which leaves the raw McNemar diff unflipped for
`falseAdvanceRate`/`falseSuspendRate` specifically because that raw diff already carries the right
polarity for them — a different, already-fixed mechanism from the unrelated, still-open
`stats.verdict()` wording caveat in `docs/BACKLOG.md`, which this family renderer never calls).
Reading the column correctly: on `falseAdvanceRate`, the two trivial-always-advance models
(`gemma-3-4b-vl-it-...`, `stablelm-zephyr-3b`) are Holm-confirmed distinguishably worse than
`qwen/qwen3-4b-2507` (both -85.0 pp). On `falseSuspendRate`, eight models are Holm-confirmed
distinguishably worse than the reference: all five of the falseAdvanceRate co-leaders —
`google/gemma-4-e2b` (-80.0 pp), `mistralai/ministral-3-3b` and its `-2512` variant (-53.3 pp
each), `prism-ml/bonsai-27b` (-86.7 pp), `qwen/qwen3-4b-thinking-2507` (-83.3 pp) — plus
`qwen2.5-3b-instruct` (-80.0 pp), `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` (-73.3 pp), and
`stable-code-instruct-3b` (-63.3 pp). In other words: every one of the seven models that led either
metric's own top row is Holm-confirmed worse than `qwen/qwen3-4b-2507` on the *other* metric — the
family test turns the descriptive "reading the index alone is misleading" point above into a
certified one. `nvidia/nemotron-3-nano-4b` is not among the 10 distinguishable-worse rows, but it
was never tested against the reference either (both its diffs, +0.0 pp and -3.3 pp, are too small
for Holm's sequential procedure to reach) — its similarity to `qwen/qwen3-4b-2507` stays
descriptive, not disproven and not confirmed. No row in either family table shows
`qwen/qwen3-4b-2507` certified *worse* than any candidate.

Net effect on the recommendation: the family test doesn't change the two-model shortlist, but it
does not leave the two on equal footing either — `qwen/qwen3-4b-2507` now carries certified
support (better than 10 rivals on at least one axis, worse than none) that `nemotron-3-nano-4b`
lacks, so it is the stronger of the two picks. Both remain subject to this pack's own resolving
power (floor 15.0/20.0 pp, MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate) — the
un-Holm-tested rows above (including nemotron's own comparison) simply didn't reach a large enough
gap to clear that bar, which is expected at this n and not a contradiction of the certified rows
that did clear it.

### nlq-generator

**`qwen/qwen3-4b-2507` — the lead pick on evidentiary strength, since its perfect score rests on
the pack's full, unreduced denominator. `qwen2.5-3b-instruct` is the lowest-latency perfect
scorer if speed is prioritized over the fuller evidentiary base.**

`nlq-structured-query`'s true denominator is 34, not 40 (6 items are structurally unanswerable and
excluded — restated from the per-pack report), and per-model denominators vary further because two
models (`qwen/qwen3-4b-thinking-2507`, `stable-code-instruct-3b`) produced zero scoreable
observations for this metric and are excluded from the ranking entirely (visible, not silently
dropped, per FR-5). Among the models that do have a stored result, five post a perfect
layer1ExactMatchRate of 1.000, but on very different evidentiary bases within this same report:
`qwen/qwen3-4b-2507` (34/34, CI [0.898, 1.000], 1408 ms p95) is the only one scored against the
pack's full 34-item denominator; `mistralai/ministral-3-3b` and
`mistralai_ministral-3-3b-instruct-2512` are each 32/32 (CI [0.893, 1.000], ~2.6-2.8 s p95);
`qwen2.5-3b-instruct` is 30/30 (CI [0.886, 1.000], 1265 ms p95 — the fastest of the perfect
scorers); and `prism-ml/bonsai-27b` is only 8/8 (CI [0.676, 1.000] — visibly the widest and least
certain of the five, and this pack's own latency outlier at 24422 ms p95).

This pack's resolving power (floor 17.6 pp, MDD80 22.3 pp at n=34) means none of the five perfect
scorers are distinguished from each other, nor from `nvidia/nemotron-3-nano-4b` immediately below
them (30/31 = 0.968, a 3.2 pp gap) — the practical choice among this top tier should be driven by
denominator completeness, latency, and footprint rather than the tied point estimate.
`qwen/qwen3-4b-2507` is the strongest pick on that basis: perfect score, full 34-item denominator,
and mid-pack latency (1408 ms) at ~2.2-2.5 GB. If raw speed is the priority and the smaller
30-item evidentiary base is accepted, `qwen2.5-3b-instruct` is faster still (1265 ms).
`prism-ml/bonsai-27b` should not be read as "tied for first" in practice — its perfect score sits
on the smallest, widest-CI base in the table and comes with a 15-20x latency cost relative to the
other perfect scorers.

### tool-caller

**No recommendation with confidence — this is the sweep's most underpowered pack by a wide margin,
and its result should be read as directional at best, never a basis for selection on its own.**

`tool-caller-shop-assistant`'s analysis unit is scripts, n=12 (3 shapes x 4 scripts — restated from
the per-pack report), giving a floor of 50.0 pp and an MDD80 of 57.8 pp. The entire observed range
in this pack's own table is exactly 50.0 pp (`qwen/qwen3-4b-thinking-2507` at 0.500, down to nine
models tied at 0.000) — even the single largest possible gap in this data sits right at the edge of
what could reach nominal significance, and no comparison in this pack reaches 80% power. Latency is
not reported for this pack (every row's p95 column is `—`), so no latency trade-off can be drawn
from it either.

With that qualifier: seven of the sixteen in-scope models completed at least one script cleanly
through this metric's tracked point in this run, ranging from `qwen/qwen3-4b-thinking-2507`
(6/12 = 0.500) and `google/gemma-4-e2b` (5/12 = 0.417) at the top, through
`mistralai/ministral-3-3b`, `mistralai_ministral-3-3b-instruct-2512`, and
`nvidia/nemotron-3-nano-4b` (4/12 = 0.333 each), down to `prism-ml/bonsai-27b` and
`qwen2.5-3b-instruct` (1/12 = 0.083 each); the remaining nine — including the sweep's own reference
model, `qwen/qwen3-4b-2507` — recorded zero clean-through-turns in this run (one of the nine,
`llama-3.2-3b-instruct`, on a reduced 0/3 attempted count, not the full 12). That split is a real,
directionally consistent observation about this specific run, but given this pack's own floor it is
not a statistically certified claim that the zero-scoring group is worse in general. Selecting a
tool-caller model on this pack's evidence alone is not advisable; if a decision is needed sooner
than a larger tool-caller pack can be built, the five models at 0.333 or better above are the most
defensible starting shortlist, understood as unproven.

### chat-responder

**`qwen2.5-coder-3b-instruct` — the lead pick on grounding score and latency together, with the
standing reminder that this pack does not measure reply quality.**

`chat-responder-grounded-answers`'s `groundingRate` is a deterministic containment check against
retrieved context, not a judgment of how good, helpful, or well-written a reply is — the
judged-quality layer is deferred (FR-21a, restated from the per-pack report); every statement below
is about grounding only, not reply quality. `qwen2.5-coder-3b-instruct` (24/30 = 0.800, CI [0.627,
0.905], 1180 ms p95, ~3.0 GB) ties the top score with `qwen/qwen3-4b-thinking-2507` (24/30 = 0.800,
7796 ms p95 — 6.6x slower for the same point estimate), making `qwen2.5-coder-3b-instruct` the
clearly better pick between the two once latency is considered. Three models sit close behind at
23/30 = 0.767: `google/gemma-4-e2b`, `llama-3.2-3b-instruct`, and `qwen/qwen3-4b-2507` — a 3.3 pp
gap below this pack's own 20.0 pp resolving floor, so none of these four are statistically
distinguished from `qwen2.5-coder-3b-instruct` by this report alone; it is the top point estimate,
not a proven winner.

This pack's own reference-anchored family (anchored on `qwen/qwen3-4b-2507`, whose diff column
checks out against this same report's marginal rates for this metric — unlike guard-judge's two
metrics above) does add one certified finding: `mistralai/ministral-3-3b` and its `-2512` variant
(both 9/30 = 0.300), `nvidia/nemotron-3-nano-4b` (8/30 = 0.267), `prism-ml/bonsai-27b` (3/30 =
0.100, and this pack's own latency outlier at 23737 ms p95), and
`qwen3.5-2b-claude-4.6-opus-reasoning-distilled` (0/30 = 0.000) are all Holm-confirmed
distinguishably worse than the reference — a real, within-this-pack finding, not merely a
point-estimate gap. The family does not reach `qwen2.5-coder-3b-instruct` (Holm's sequential
procedure stops before testing it), so its lead over the reference stays descriptive, not proven.

Caveat: before this or any candidate is used for the actual chat-responder role, its output still
needs a reply-quality pass this pack does not perform — a grounded-but-unhelpful reply scores 1.0
here.
