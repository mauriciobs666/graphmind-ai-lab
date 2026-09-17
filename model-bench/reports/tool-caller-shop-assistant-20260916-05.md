# Comparison — tool-caller-shop-assistant@0.2.0 (tool-caller)

### Funnel — mistralai/ministral-3-3b

```
turns driven                    81
  unrunnable (model channel)    0   -> no-response / server-rejected
    turns scored after unrunnable  0
  unrunnable (tool channel)     0   -> dispatch raised; conversation censored at t
  R(t) = 0 (restraint turns)    6   -> restraint rate 6/6
  R(t) >= 1                     75
    native call emitted         73   -> (a)+(b) partition over 75
    prose pseudo-call           0
    no attempt                  2
  turns with >=1 call           73   -> (c), (e), (f) denominators
  dispatched calls              75   -> (d) denominator (calls, not turns)
    args omitted required       2   -> per-argument split of (d)'s 69 calls w/ correct tool
    args wrong value            1   -> of which boundary/unit: 0
  fact-bearing returns          34   -> (g) denominator
  unscoreable returns           39
```

- prose-pseudo-call detector: precision 1.000, recall 1.000 (n=20 calibration replies)
- `I(t)` (iterations/turn, replied+cap-hit only, n=81): mean 1.91, p95 2.00
- `Y_calls / Y` (unrestricted, every turn driven): 155/81 = 1.91
- Different statistics, both printed: `Y_calls / Y` pools every driven turn including ones that never completed, while `I(t)`'s mean/p95 count only replied/cap-hit turns (`-ml` §4.2(f)/§11.4) — they will differ whenever a turn did not complete, and neither substitutes for the other.

### Funnel — qwen/qwen3-4b-2507

```
turns driven                    81
  unrunnable (model channel)    0   -> no-response / server-rejected
    turns scored after unrunnable  0
  unrunnable (tool channel)     0   -> dispatch raised; conversation censored at t
  R(t) = 0 (restraint turns)    6   -> restraint rate 6/6
  R(t) >= 1                     75
    native call emitted         59   -> (a)+(b) partition over 75
    prose pseudo-call           0
    no attempt                  16
  turns with >=1 call           59   -> (c), (e), (f) denominators
  dispatched calls              59   -> (d) denominator (calls, not turns)
    args omitted required       0   -> per-argument split of (d)'s 56 calls w/ correct tool
    args wrong value            1   -> of which boundary/unit: 0
  fact-bearing returns          24   -> (g) denominator
  unscoreable returns           35
```

- prose-pseudo-call detector: precision 1.000, recall 1.000 (n=20 calibration replies)
- `I(t)` (iterations/turn, replied+cap-hit only, n=81): mean 1.73, p95 2.00
- `Y_calls / Y` (unrestricted, every turn driven): 140/81 = 1.73
- Different statistics, both printed: `Y_calls / Y` pools every driven turn including ones that never completed, while `I(t)`'s mean/p95 count only replied/cap-hit turns (`-ml` §4.2(f)/§11.4) — they will differ whenever a turn did not complete, and neither substitutes for the other.

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| mistralai/ministral-3-3b | cleanThroughTurnH | 9/12 | 0.750 | [0.468, 0.911] |
| mistralai/ministral-3-3b | restraint | 6/6 | 1.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | iterationCapHitRate | 0/81 | 0.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | native | 73/75 | 0.973 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | rightToolChosen | 68/73 | 0.932 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | allArgsCorrect | 66/69 | 0.957 | — (n is calls; not the analysis unit) |
| mistralai/ministral-3-3b | spurious | 6/73 | 0.082 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | duplicateWithinTurn | 0/73 | 0.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | duplicateCrossTurn | 15/73 | 0.205 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | stoppingWhenDone | 55/73 | 0.753 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | replyMatchesTool | 31/34 | 0.912 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | cleanThroughTurnH | 6/12 | 0.500 | [0.254, 0.746] |
| qwen/qwen3-4b-2507 | restraint | 6/6 | 1.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | iterationCapHitRate | 0/81 | 0.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | native | 59/75 | 0.787 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | rightToolChosen | 56/59 | 0.949 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | allArgsCorrect | 55/56 | 0.982 | — (n is calls; not the analysis unit) |
| qwen/qwen3-4b-2507 | spurious | 3/59 | 0.051 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | duplicateWithinTurn | 0/59 | 0.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | duplicateCrossTurn | 6/59 | 0.102 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | stoppingWhenDone | 50/59 | 0.847 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | replyMatchesTool | 23/24 | 0.958 | — (n is turns; not the analysis unit) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

_A count whose denominator is not the analysis unit prints **without an interval**: `-ml` §4.4's first mandatory consequence is *"Never print a Wilson interval over a turn-pooled count"*, because the turns of one conversation are not independent observations and the resulting interval is understated several-fold. The honest bound is a one-level cluster bootstrap over the conversations (`stats.cluster_bootstrap`, Rule 6), which needs the per-unit observations a stored aggregate does not carry — S2's runner does._

## Per-turn position

| position | mistralai/ministral-3-3b (observed k/n, structural n) | qwen/qwen3-4b-2507 (observed k/n, structural n) |
|---|---|---|
| t=0 | 0/12 (structural 12) | 1/12 (structural 12) |
| t=1 | 1/12 (structural 12) | 2/11 (structural 12) |
| t=2 | 0/11 (structural 12) | 0/9 (structural 12) — descriptive at this n — no significance claim |
| t=3 | 2/11 (structural 12) | 3/9 (structural 12) — descriptive at this n — no significance claim |
| t=4 | 3/7 (structural 12) — descriptive at this n — no significance claim | 1/5 (structural 12) — descriptive at this n — no significance claim |
| t=5 | 1/4 (structural 12) — descriptive at this n — no significance claim | 2/4 (structural 12) — descriptive at this n — no significance claim |
| t=6 | 2/3 (structural 12) — descriptive at this n — no significance claim | 1/2 (structural 12) — descriptive at this n — no significance claim |
| t=7 | 0/1 (structural 12) — descriptive at this n — no significance claim | 0/1 (structural 12) — descriptive at this n — no significance claim |
| t=8 | 0/1 (structural 12) — descriptive at this n — no significance claim | 0/1 (structural 12) — descriptive at this n — no significance claim |
| t=9 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |

## Hazard (time-to-first-failure)

| position | mistralai/ministral-3-3b (f_t, r_t, c_t) | qwen/qwen3-4b-2507 (f_t, r_t, c_t) |
|---|---|---|
| t=0 | f=0, r=12, c=0 | f=1, r=12, c=0 |
| t=1 | f=1, r=12, c=0 | f=2, r=11, c=0 |
| t=2 | f=0, r=11, c=0 | f=0, r=9, c=0 |
| t=3 | f=2, r=11, c=0 | f=3, r=9, c=0 |
| t=4 | f=3, r=7, c=2 | f=1, r=5, c=1 |
| t=5 | f=1, r=4, c=0 | f=2, r=4, c=0 |
| t=6 | f=2, r=3, c=0 | f=1, r=2, c=0 |
| t=7 | f=0, r=1, c=0 | f=0, r=1, c=0 |
| t=8 | f=0, r=1, c=0 | f=0, r=1, c=0 |
| t=9 | f=0, r=0, c=1 | f=0, r=0, c=1 |

## Verdicts

Comparison kind: **paired, same session** (§3.7).

### cleanThroughTurnH

Not distinguishable at this sample size. Observed difference +25.0 pp, 95% CI [-8.7, 58.3] pp covers zero (b=4, c=1, McNemar exact p=0.375). This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations (12 units, design effect 1.00, assumed, alpha=0.05); the observed 25.0 pp is below that. Neither model is ranked above the other. Decided by the conservative envelope on the paired difference — MOVER-D and the exact paired bootstrap, the wider of the two at each bound — the instrument here because this comparison's design effect is assumed rather than established by construction — with no widening applied (sqrt(DEFF)=1.00), in conjunction with McNemar's exact test (p=0.375) as a necessary condition: a design effect that was never established cannot license the exact test to carry a verdict, so it may withhold one but never carries one on its own.

- paired n: 12 of 12 conversations (`asymmetry`: 0 scoreable for mistralai/ministral-3-3b only, 0 scoreable for qwen/qwen3-4b-2507 only; 0 unscoreable in both; 0 present in mistralai/ministral-3-3b only, 0 in qwen/qwen3-4b-2507 only) — §4.3
- marginal Wilson intervals overlap: yes
- decided by: conservative envelope (lower bound: MOVER-D; upper bound: exact paired bootstrap, p=0.975)

This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations (12 units, design effect 1.00, assumed, alpha=0.05). Differences below 50.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every conversation the models differ on; if it loses one for every two it wins, 80% power is not reached at any effect size at this n. Inference is conditional on the 12 scripts in tool-caller-shop-assistant@0.2.0; generalization to unwritten scripts is not certified by any interval in this report.

**Headline (cleanThroughTurnH):** Not distinguishable at this sample size. Observed difference +25.0 pp, 95% CI [-8.7, 58.3] pp covers zero (b=4, c=1, McNemar exact p=0.375). This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations (12 units, design effect 1.00, assumed, alpha=0.05); the observed 25.0 pp is below that. Neither model is ranked above the other. Decided by the conservative envelope on the paired difference — MOVER-D and the exact paired bootstrap, the wider of the two at each bound — the instrument here because this comparison's design effect is assumed rather than established by construction — with no widening applied (sqrt(DEFF)=1.00), in conjunction with McNemar's exact test (p=0.375) as a necessary condition: a design effect that was never established cannot license the exact test to carry a verdict, so it may withhold one but never carries one on its own.

### Exploratory metrics

- `restraint` — exploratory — no significance claim
- `iterationCapHitRate` — exploratory — no significance claim
- `native` — exploratory — no significance claim
- `rightToolChosen` — exploratory — no significance claim
- `allArgsCorrect` — exploratory — no significance claim
- `spurious` — exploratory — no significance claim
- `duplicateWithinTurn` — exploratory — no significance claim
- `duplicateCrossTurn` — exploratory — no significance claim
- `stoppingWhenDone` — exploratory — no significance claim
- `replyMatchesTool` — exploratory — no significance claim

_The marginal-overlap line is a **diagnostic**, never the verdict: at this lab's sample sizes two marginal intervals overlapping is a far stronger condition than their difference covering zero, and the literal rule cannot fire at all at n<=40 with a baseline >=0.90 (`-ml` §3.1)._

