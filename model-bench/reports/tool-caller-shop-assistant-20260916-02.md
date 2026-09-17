# Comparison — tool-caller-shop-assistant@0.2.0 (tool-caller)

### Funnel — qwen/qwen3-4b-2507

```
turns driven                    81
  unrunnable (model channel)    0   -> no-response / server-rejected
    turns scored after unrunnable  0
  unrunnable (tool channel)     0   -> dispatch raised; conversation censored at t
  R(t) = 0 (restraint turns)    6   -> restraint rate 6/6
  R(t) >= 1                     75
    native call emitted         42   -> (a)+(b) partition over 75
    prose pseudo-call           0
    no attempt                  33
  turns with >=1 call           42   -> (c), (e), (f) denominators
  dispatched calls              44   -> (d) denominator (calls, not turns)
    args omitted required       0   -> per-argument split of (d)'s 39 calls w/ correct tool
    args wrong value            0   -> of which boundary/unit: 0
  fact-bearing returns          15   -> (g) denominator
  unscoreable returns           27
```

- prose-pseudo-call detector: precision 1.000, recall 1.000 (n=20 calibration replies)
- `I(t)` (iterations/turn, replied+cap-hit only, n=81): mean 1.54, p95 2.00
- `Y_calls / Y` (unrestricted, every turn driven): 125/81 = 1.54
- Different statistics, both printed: `Y_calls / Y` pools every driven turn including ones that never completed, while `I(t)`'s mean/p95 count only replied/cap-hit turns (`-ml` §4.2(f)/§11.4) — they will differ whenever a turn did not complete, and neither substitutes for the other.

### Funnel — mistralai/ministral-3-3b

```
turns driven                    81
  unrunnable (model channel)    69   -> no-response / server-rejected
    turns scored after unrunnable  0
  unrunnable (tool channel)     0   -> dispatch raised; conversation censored at t
  R(t) = 0 (restraint turns)    1   -> restraint rate 1/1
  R(t) >= 1                     11
    native call emitted         11   -> (a)+(b) partition over 11
    prose pseudo-call           0
    no attempt                  0
  turns with >=1 call           11   -> (c), (e), (f) denominators
  dispatched calls              11   -> (d) denominator (calls, not turns)
    args omitted required       0   -> per-argument split of (d)'s 11 calls w/ correct tool
    args wrong value            0   -> of which boundary/unit: 0
  fact-bearing returns          1   -> (g) denominator
  unscoreable returns           10
```

- prose-pseudo-call detector: precision 1.000, recall 1.000 (n=20 calibration replies)
- `I(t)` (iterations/turn, replied+cap-hit only, n=12): mean 1.92, p95 2.00
- `Y_calls / Y` (unrestricted, every turn driven): 92/81 = 1.14
- Different statistics, both printed: `Y_calls / Y` pools every driven turn including ones that never completed, while `I(t)`'s mean/p95 count only replied/cap-hit turns (`-ml` §4.2(f)/§11.4) — they will differ whenever a turn did not complete, and neither substitutes for the other.

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| qwen/qwen3-4b-2507 | cleanThroughTurnH | 2/12 | 0.167 | [0.047, 0.448] |
| qwen/qwen3-4b-2507 | restraint | 6/6 | 1.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | iterationCapHitRate | 0/81 | 0.000 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | native | 42/75 | 0.560 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | rightToolChosen | 37/42 | 0.881 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | allArgsCorrect | 39/39 | 1.000 | — (n is calls; not the analysis unit) |
| qwen/qwen3-4b-2507 | spurious | 5/42 | 0.119 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | duplicateWithinTurn | 1/42 | 0.024 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | duplicateCrossTurn | 3/42 | 0.071 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | stoppingWhenDone | 33/42 | 0.786 | — (n is turns; not the analysis unit) |
| qwen/qwen3-4b-2507 | replyMatchesTool | 11/15 | 0.733 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | restraint | 1/1 | 1.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | iterationCapHitRate | 0/12 | 0.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | native | 11/11 | 1.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | rightToolChosen | 11/11 | 1.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | allArgsCorrect | 11/11 | 1.000 | — (n is calls; not the analysis unit) |
| mistralai/ministral-3-3b | spurious | 0/11 | 0.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | duplicateWithinTurn | 0/11 | 0.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | duplicateCrossTurn | 0/11 | 0.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | stoppingWhenDone | 11/11 | 1.000 | — (n is turns; not the analysis unit) |
| mistralai/ministral-3-3b | replyMatchesTool | 1/1 | 1.000 | — (n is turns; not the analysis unit) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

_A count whose denominator is not the analysis unit prints **without an interval**: `-ml` §4.4's first mandatory consequence is *"Never print a Wilson interval over a turn-pooled count"*, because the turns of one conversation are not independent observations and the resulting interval is understated several-fold. The honest bound is a one-level cluster bootstrap over the conversations (`stats.cluster_bootstrap`, Rule 6), which needs the per-unit observations a stored aggregate does not carry — S2's runner does._

## Per-turn position

| position | qwen/qwen3-4b-2507 (observed k/n, structural n) | mistralai/ministral-3-3b (observed k/n, structural n) |
|---|---|---|
| t=0 | 1/12 (structural 12) | 0/12 (structural 12) |
| t=1 | 3/11 (structural 12) | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=2 | 0/8 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=3 | 6/8 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=4 | 1/2 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=5 | 0/1 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=6 | 1/1 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=7 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=8 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |
| t=9 | 0/0 (structural 12) — descriptive at this n — no significance claim | 0/0 (structural 12) — descriptive at this n — no significance claim |

## Hazard (time-to-first-failure)

| position | qwen/qwen3-4b-2507 (f_t, r_t, c_t) | mistralai/ministral-3-3b (f_t, r_t, c_t) |
|---|---|---|
| t=0 | f=1, r=12, c=0 | f=0, r=12, c=0 |
| t=1 | f=3, r=11, c=0 | f=0, r=0, c=12 |
| t=2 | f=0, r=8, c=0 | f=0, r=0, c=0 |
| t=3 | f=6, r=8, c=0 | f=0, r=0, c=0 |
| t=4 | f=1, r=2, c=0 | f=0, r=0, c=0 |
| t=5 | f=0, r=1, c=0 | f=0, r=0, c=0 |
| t=6 | f=1, r=1, c=0 | f=0, r=0, c=0 |
| t=7 | f=0, r=0, c=0 | f=0, r=0, c=0 |
| t=8 | f=0, r=0, c=0 | f=0, r=0, c=0 |
| t=9 | f=0, r=0, c=0 | f=0, r=0, c=0 |

## Verdicts

Comparison kind: **paired, same session** (§3.7).

### cleanThroughTurnH

**No verdict: no paired data.** No conversation is scoreable for `cleanThroughTurnH` in both arms, so there is no paired table, no interval and no verdict. An arm carrying no data for a metric is not an arm that failed every conversation of it (`-ml` §4.3). The tally below says where the rows went.

- paired n: 0 of 12 conversations (`asymmetry`: 12 scoreable for qwen/qwen3-4b-2507 only, 0 scoreable for mistralai/ministral-3-3b only; 0 unscoreable in both; 0 present in qwen/qwen3-4b-2507 only, 0 in mistralai/ministral-3-3b only) — §4.3

**Headline (cleanThroughTurnH):** **No verdict: no paired data.** No conversation is scoreable for `cleanThroughTurnH` in both arms, so there is no paired table, no interval and no verdict. An arm carrying no data for a metric is not an arm that failed every conversation of it (`-ml` §4.3).

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

