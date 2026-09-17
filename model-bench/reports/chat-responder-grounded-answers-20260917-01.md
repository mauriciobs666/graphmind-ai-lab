# Comparison — chat-responder-grounded-answers@0.1.0 (chat-responder)

> **Reply quality is not measured by this pack.** `groundingRate` is a deterministic containment check against the retrieved context, never a judgement of how good, helpful, or well-written a reply is (FR-21a — the judged-quality layer is deferred, `docs/BACKLOG.md`).

> **NEGATIVE CONTROL (WIRING SMOKE CHECK)** — both arms are the *same stored record*, so `b = c = 0 by construction` and this comparison **cannot fail**. It proves the mode is wired; it says nothing about whether the harness is sound. The real negative control is two **independent** runs of the same model and is an acceptance step, not this (`-ml` §9, plan §5 test 19a).

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| qwen/qwen3-4b-2507 | groundingRate | 15/30 | 0.500 | [0.332, 0.668] |
| qwen/qwen3-4b-2507 | formatMaxWords | 30/30 | 1.000 | [0.886, 1.000] |
| qwen/qwen3-4b-2507 | formatSingleParagraph | 30/30 | 1.000 | [0.886, 1.000] |
| qwen/qwen3-4b-2507 | formatNoForbiddenPatterns | 30/30 | 1.000 | [0.886, 1.000] |
| qwen/qwen3-4b-2507 | groundingRate | 15/30 | 0.500 | [0.332, 0.668] |
| qwen/qwen3-4b-2507 | formatMaxWords | 30/30 | 1.000 | [0.886, 1.000] |
| qwen/qwen3-4b-2507 | formatSingleParagraph | 30/30 | 1.000 | [0.886, 1.000] |
| qwen/qwen3-4b-2507 | formatNoForbiddenPatterns | 30/30 | 1.000 | [0.886, 1.000] |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

## Speed

| arm | p50 | p95/max | timed/n | withheld (load/no-resp) | TTFT median | prefill ms/1k | tok/s median (diagnostic) |
|---|---|---|---|---|---|---|---|
| qwen/qwen3-4b-2507 | 575 | 1142 | 30/30 | 0/0 | 39 | 129.8 | 55.7 |
| qwen/qwen3-4b-2507 | 575 | 1142 | 30/30 | 0/0 | 39 | 129.8 | 55.7 |

*Descriptive only — decode tokens/sec is a diagnostic, never a comparison instrument (FR-11).*

## Verdicts

Comparison kind: **paired, same session** (§3.7).

### groundingRate

Not distinguishable at this sample size. Observed difference +0.0 pp, 95% CI [-0.0, 0.0] pp covers zero (b=0, c=0, McNemar exact p=1.000). This pack resolves differences of >=25.1 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.05); the observed 0.0 pp is below that. Neither model is ranked above the other.

- paired n: 30 of 30 items (`asymmetry`: 0 scoreable for qwen/qwen3-4b-2507 only, 0 scoreable for qwen/qwen3-4b-2507 only; 0 unscoreable in both; 0 present in qwen/qwen3-4b-2507 only, 0 in qwen/qwen3-4b-2507 only) — §4.3
- marginal Wilson intervals overlap: yes
- decided by: mcnemar-exact

This pack resolves differences of >=25.1 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.05). Differences below 20.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 30 items in chat-responder-grounded-answers@0.1.0; generalization to unwritten items is not certified by any interval in this report.

**Headline (groundingRate):** Not distinguishable at this sample size. Observed difference +0.0 pp, 95% CI [-0.0, 0.0] pp covers zero (b=0, c=0, McNemar exact p=1.000). This pack resolves differences of >=25.1 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.05); the observed 0.0 pp is below that. Neither model is ranked above the other.

### Exploratory metrics

- `formatMaxWords` — exploratory — no significance claim
- `formatSingleParagraph` — exploratory — no significance claim
- `formatNoForbiddenPatterns` — exploratory — no significance claim

_The marginal-overlap line is a **diagnostic**, never the verdict: at this lab's sample sizes two marginal intervals overlapping is a far stronger condition than their difference covering zero, and the literal rule cannot fire at all at n<=40 with a baseline >=0.90 (`-ml` §3.1)._

