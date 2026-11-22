# ccg P1-34 flow re-measurement: Part F1 (pair validity)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** P1-34 (U6)

Prereg `docs/plans/connected-code-graph-p1-flow-prereg.md` §2.4; P0-ML Part F (`docs/plans/connected-code-graph-p0-ml.md`). Input: `cpg/spikes/frontend-fitness/labels/python-flow-p1.jsonl` (read only, not edited), its sample, the corpus root `cpg/.ccg-work/corpora/python/src`, and the `flowaud-p1` draw (F1 only; no F2 file opened). Every pair was judged from the source against the brief's definitions (`cpg/spikes/frontend-fitness/flow-brief.template.md`). No Joern, parser, results, facts, or scorer output was read or run.

## Verdict: F1 not flagged

| Set | n | valid | invalid | endpoint-naming-error |
|---|---|---|---|---|
| Uniform (60 pairs, 25 methods) | 60 | 60 | 0 | 0 |
| Unsure (targeted, one-sided; 11 pairs) | 11 | 11 | 0 | 0 |

Uniform rate 0/60 (Wilson 95% upper bound 0.060). Flag rule `invalid/60 > 0.05, >= 4`: not met. No pair leaves `G` from either set.

**INVALID keys (ordinal,pairIndex): none.** **Endpoint-naming errors: none** (every call ordinal, line, param index, ret ordinal and cond code was recomputed from the file; checked at the line numbers of `falkorchat/guards.py:464-485`, `tests/test_storefront_api.py:2085-2089`, `tests/test_querygen.py:844-848`).

The unsure set is a targeted draw of pairs the labeller itself doubted: it can only show labeller errors it was suspicious of, so it bounds no population rate.

## Rulings that were forks (decided, with the sensitivity)

1. **Early exit with several exits before the sink** (brief: after an exit, "every later point in that block that the exit can skip is decided by `c`"). Ruled valid as written: `guards.py` (11,33) cond 466 to the for header 481; (11,50), (11,51) cond 475 to cond 481 and ret 5; unsure (11,49). A strict "certain to execute" or CDG reading would give those points only to the nearest exit's cond (479) and make (11,33), (11,50), (11,51) invalid (uniform 3/60 = 0.05, still not above 0.05 and below 4; unsure +1). The verdict does not change under either reading.
2. **Subscript operand reaches the subscript result** (a key into `node[...]`, `cursor[key]`): ruled valid (an indexing operator, brief's "language operators … elements"). Touches only unsure pairs (2,10), (11,13); if rejected, 2 unsure invalid, the uniform set is untouched.
3. **Callee result as the receiver of a dynamic call** (`getattr(...)(...)`, `test_storefront_api.py:2089`, (16,2)): valid, the brief's "receiver".
4. **A value stored only as a dict key of a returned container** (`test_input_bound_parity.py:519-524`, (30,7)): valid (the key is contents of the returned dict). Borderline.
5. **A return inside an `except` handler after an early exit** (`guards.py:129-134`, (13,7)): valid by the early-exit bullet; exception edges are not control flow but the exit does skip the handler. Borderline.

Rule for borderline calls: an invalid is recorded only when the pair is clearly not directly reaching or directly decided, because removing a real pair from `G` biases recall upward while keeping a doubtful one biases it down (conservative).
