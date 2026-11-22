# ccg P1-34 flow re-measurement: F2 omission audit (python)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** P1-34

**Verdict: F2 not flagged.** Omission rate 0/104 = 0.000 (flag needs > 0.05 with >= 20 valid pairs). No pair is added to the truth.

## Rule used

Prereg `docs/plans/connected-code-graph-p1-flow-prereg.md` §2.4 F2 / `docs/plans/connected-code-graph-p0-ml.md` Part F: omission rate = valid pairs the golden set lacks (valid F2-only) ÷ valid pairs in the union of the two label sets. Flag iff > 0.05 with >= 20 valid pairs; fewer: extend by 3 methods once, then `insufficient`. Endpoint identity = kind + source point + sink point (param index/name; call calleeName/ordinal/line; ret ordinal/line; cond line/code). Judged against the brief `cpg/spikes/frontend-fitness/sample/python-flow-p1.brief.md` from `cpg/.ccg-work/corpora/python/src`. Only F2 and the original labels, the sample subset and the brief were read.

## Diff (6 methods: 3, 11, 16, 18, 21, 30)

| Ordinal | Original | F2 | Agreed | F2-only | Original-only |
|---|---|---|---|---|---|
| 3 | 13 | 13 | 13 | 0 | 0 |
| 11 (`guards.py::_resolve_path`) | 52 | 46 | 45 | 1 | 7 |
| 16 | 7 | 7 | 7 | 0 | 0 |
| 18 | 11 | 11 | 11 | 0 | 0 |
| 21 | 10 | 10 | 10 | 0 | 0 |
| 30 | 11 | 11 | 11 | 0 | 0 |

Union 104 + 1 invalid = 105 pairs; agreed pairs (97) are taken as valid. Only method 11 differs. No endpoint-naming errors.

## One-sided pairs (all in `falkorchat/guards.py`, method lines 464-485)

F2 only (possible original omission), 1 pair:

1. ord 11 control `cond 481 "key in rest.split(".")"` -> `ret 4 @483`: **invalid.** The return is inside `if` 482 inside the `for`; "decided by `b`, not `a`" and the nearest-governing-cond rule give it to cond 482, which both sets already have (482 -> ret 4).

Original only (F2 omissions), 7 pairs, all **valid**:

2. data `param 0 path` -> `cond 466 "not isinstance(path, str)"`: valid, `path` is read inside the condition's text (:466); F2 has it for the same shape at cond 482 (`cursor`).
3. data `call partition#0 @468` -> `cond 481`: valid, `rest` is destructured from the partition result (:468) and read in the `for` header (:481); F2 has partition -> `split` and the other conds.
4. data `call isinstance#1 @482` -> `cond 482`: valid, a call that is an operand of the condition goes to it (brief); F2 has the same for isinstance#0 -> cond 466.
5-8. control `cond 475 "head == \"output\""` -> `cond 479`, `call split#0 @481`, `cond 481`, `ret 5 @485` (original: unsure): valid. The `else: return _MISSING` (:477-478) is an early exit of the chain whose nearest cond is 475; P0-ML §8.4 "nearest governing cond" gives points after the chain that this exit can skip to the exit's cond. Valid but defensible only by that ruling, hence correctly `unsure`.

## Counts and flag

- Valid F2-only pairs not in the original: **0**. Valid pairs in the union: 104 (>= 20, so no extension).
- Omission rate = 0/104 = 0.000. Not flagged; no `--extend` needed.
- Keys to ADD to the truth: **none**.
- Informational (not the flag): F2 missed 7 of 104 valid union pairs (all method 11, one data/loop cluster and the else-return cluster), so the second labeller's recall is about 0.93 on this draw; the original's is 1.00 here.

## Observation (outside the one-sided scope, not counted)

By the same nearest-governing-cond ruling, cond 472 `not rest` (`return _MISSING` at :473, an early exit) would also decide 479, `split#0 @481`, cond 481 and `ret 5 @485`. Neither labeller has these 4 pairs, so the diff cannot see them. They are common-mode omissions, consistent with the prereg caveat that the same model and brief bias this rate downward; F2 stays a qualifier.
