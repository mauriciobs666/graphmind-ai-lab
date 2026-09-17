# Comparison — tool-caller-shop-assistant@0.2.0 (tool-caller)

### Funnel — qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z)

```
turns driven                    81
  unrunnable (model channel)    0   -> no-response / server-rejected
    turns scored after unrunnable  0
  unrunnable (tool channel)     0   -> dispatch raised; conversation censored at t
  R(t) = 0 (restraint turns)    6   -> restraint rate 6/6
  R(t) >= 1                     75
    native call emitted         23   -> (a)+(b) partition over 75
    prose pseudo-call           0
    no attempt                  52
  turns with >=1 call           23   -> (c), (e), (f) denominators
  dispatched calls              23   -> (d) denominator (calls, not turns)
    args omitted required       0   -> per-argument split of (d)'s 20 calls w/ correct tool
    args wrong value            0   -> of which boundary/unit: 0
  fact-bearing returns          6   -> (g) denominator
  unscoreable returns           17
```

- prose-pseudo-call detector: precision 1.000, recall 1.000 (n=20 calibration replies)
- `I(t)` (iterations/turn, replied+cap-hit only, n=81): mean 1.28, p95 2.00
- `Y_calls / Y` (unrestricted, every turn driven): 104/81 = 1.28
- Different statistics, both printed: `Y_calls / Y` pools every driven turn including ones that never completed, while `I(t)`'s mean/p95 count only replied/cap-hit turns (`-ml` §4.2(f)/§11.4) — they will differ whenever a turn did not complete, and neither substitutes for the other.

### Funnel — qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z)

```
turns driven                    81
  unrunnable (model channel)    0   -> no-response / server-rejected
    turns scored after unrunnable  0
  unrunnable (tool channel)     0   -> dispatch raised; conversation censored at t
  R(t) = 0 (restraint turns)    6   -> restraint rate 6/6
  R(t) >= 1                     75
    native call emitted         23   -> (a)+(b) partition over 75
    prose pseudo-call           0
    no attempt                  52
  turns with >=1 call           23   -> (c), (e), (f) denominators
  dispatched calls              23   -> (d) denominator (calls, not turns)
    args omitted required       0   -> per-argument split of (d)'s 20 calls w/ correct tool
    args wrong value            0   -> of which boundary/unit: 0
  fact-bearing returns          6   -> (g) denominator
  unscoreable returns           17
```

- prose-pseudo-call detector: precision 1.000, recall 1.000 (n=20 calibration replies)
- `I(t)` (iterations/turn, replied+cap-hit only, n=81): mean 1.28, p95 2.00
- `Y_calls / Y` (unrestricted, every turn driven): 104/81 = 1.28
- Different statistics, both printed: `Y_calls / Y` pools every driven turn including ones that never completed, while `I(t)`'s mean/p95 count only replied/cap-hit turns (`-ml` §4.2(f)/§11.4) — they will differ whenever a turn did not complete, and neither substitutes for the other.

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | cleanThroughTurnH | 0/12 | 0.000 | [0.000, 0.242] |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | restraint | 6/6 | 1.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | iterationCapHitRate | 0/81 | 0.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | native | 23/75 | 0.307 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | rightToolChosen | 20/23 | 0.870 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | allArgsCorrect | 20/20 | 1.000 | — (n is calls; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | spurious | 3/23 | 0.130 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | duplicateWithinTurn | 0/23 | 0.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | duplicateCrossTurn | 1/23 | 0.043 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | stoppingWhenDone | 20/23 | 0.870 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) | replyMatchesTool | 6/6 | 1.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | cleanThroughTurnH | 0/12 | 0.000 | [0.000, 0.242] |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | restraint | 6/6 | 1.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | iterationCapHitRate | 0/81 | 0.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | native | 23/75 | 0.307 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | rightToolChosen | 20/23 | 0.870 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | allArgsCorrect | 20/20 | 1.000 | — (n is calls; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | spurious | 3/23 | 0.130 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | duplicateWithinTurn | 0/23 | 0.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | duplicateCrossTurn | 1/23 | 0.043 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | stoppingWhenDone | 20/23 | 0.870 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) | replyMatchesTool | 6/6 | 1.000 | — (n is turns; not the analysis unit) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

_A count whose denominator is not the analysis unit prints **without an interval**: `-ml` §4.4's first mandatory consequence is *"Never print a Wilson interval over a turn-pooled count"*, because the turns of one conversation are not independent observations and the resulting interval is understated several-fold. The honest bound is a one-level cluster bootstrap over the conversations (`stats.cluster_bootstrap`, Rule 6), which needs the per-unit observations a stored aggregate does not carry — S2's runner does._

## Per-turn position

| position | qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) (observed k/n, structural n) | qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) (observed k/n, structural n) |
|---|---|---|
| t=0 | 1/12 (structural 12) | 1/12 (structural 12) |
| t=1 | 7/11 (structural 12) | 7/11 (structural 12) |
| t=2 | 3/4 (structural 12) — descriptive at this n — no significance claim | 3/4 (structural 12) — descriptive at this n — no significance claim |
| t=3 | 1/1 (structural 12) — descriptive at this n — no significance claim | 1/1 (structural 12) — descriptive at this n — no significance claim |
| t=4 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=5 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=6 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=7 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=8 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=9 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |

## Hazard (time-to-first-failure)

| position | qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) (f_t, r_t, c_t) | qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) (f_t, r_t, c_t) |
|---|---|---|
| t=0 | f=1, r=12, c=0 | f=1, r=12, c=0 |
| t=1 | f=7, r=11, c=0 | f=7, r=11, c=0 |
| t=2 | f=3, r=4, c=0 | f=3, r=4, c=0 |
| t=3 | f=1, r=1, c=0 | f=1, r=1, c=0 |
| t=4 | f=0, r=0, c=0 | f=0, r=0, c=0 |
| t=5 | f=0, r=0, c=0 | f=0, r=0, c=0 |
| t=6 | f=0, r=0, c=0 | f=0, r=0, c=0 |
| t=7 | f=0, r=0, c=0 | f=0, r=0, c=0 |
| t=8 | f=0, r=0, c=0 | f=0, r=0, c=0 |
| t=9 | f=0, r=0, c=0 | f=0, r=0, c=0 |

## Verdicts

Comparison kind: **paired, same session** (§3.7).

### cleanThroughTurnH

Not distinguishable at this sample size. Observed difference +0.0 pp, 95% CI [-24.2, 24.2] pp covers zero (b=0, c=0, McNemar exact p=1.000). This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations (12 units, design effect 1.00, assumed, alpha=0.05); the observed 0.0 pp is below that. Neither model is ranked above the other. Decided by the conservative envelope on the paired difference — MOVER-D and the exact paired bootstrap, the wider of the two at each bound — the instrument here because this comparison's design effect is assumed rather than established by construction — with no widening applied (sqrt(DEFF)=1.00), in conjunction with McNemar's exact test (p=1.000) as a necessary condition: a design effect that was never established cannot license the exact test to carry a verdict, so it may withhold one but never carries one on its own.

- paired n: 12 of 12 conversations (`asymmetry`: 0 scoreable for qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) only, 0 scoreable for qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) only; 0 unscoreable in both; 0 present in qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:11:26Z) only, 0 in qwen/qwen3-4b-2507 (run tool-caller-shop-assistant-qwen_qwen3-4b-2507-2026-09-17T01:13:38Z) only) — §4.3
- marginal Wilson intervals overlap: yes
- decided by: conservative envelope (lower bound: MOVER-D; upper bound: MOVER-D)

This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations (12 units, design effect 1.00, assumed, alpha=0.05). Differences below 50.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every conversation the models differ on; if it loses one for every two it wins, 80% power is not reached at any effect size at this n. Inference is conditional on the 12 scripts in tool-caller-shop-assistant@0.2.0; generalization to unwritten scripts is not certified by any interval in this report.

**Headline (cleanThroughTurnH):** Not distinguishable at this sample size. Observed difference +0.0 pp, 95% CI [-24.2, 24.2] pp covers zero (b=0, c=0, McNemar exact p=1.000). This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations (12 units, design effect 1.00, assumed, alpha=0.05); the observed 0.0 pp is below that. Neither model is ranked above the other. Decided by the conservative envelope on the paired difference — MOVER-D and the exact paired bootstrap, the wider of the two at each bound — the instrument here because this comparison's design effect is assumed rather than established by construction — with no widening applied (sqrt(DEFF)=1.00), in conjunction with McNemar's exact test (p=1.000) as a necessary condition: a design effect that was never established cannot license the exact test to carry a verdict, so it may withhold one but never carries one on its own.

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

