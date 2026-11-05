# S28 flow adjudication tools (data-scientist)

These tools replay the per-pair attributions in `adjudication/<lang>-flow-misses.adj.jsonl` and `adjudication/<lang>-flow-spurious.adj.jsonl`, and the figures in report §a2.2 (`docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md`).
Work files go to `$FLOW_WORK` (default `/tmp/ccg-flow-adj`). Nothing is written under `cpg/.ccg-work`.

| Step | Command (from `cpg/`) | Output |
|---|---|---|
| 0 | `python3 spikes/frontend-fitness/adjudication/tools/flow/make_variants.py` (optional) | regenerates `rdrelax.sc`, `rdwide.sc` and `rdfix.sc` from `s28-extract-bodiless-guard.sc`; byte-identical to the checked-in copies |
| 1 | `spikes/frontend-fitness/adjudication/tools/flow/run_probes.sh <lang> rdrelax`, then `... <lang> rdfix` | one facts file each in `run/<lang>/` and `runfix/<lang>/` |
| 2 | `.venv/bin/python spikes/frontend-fitness/adjudication/tools/flow/variants.py <lang>` | `var/<lang>.status.json`: per-pair shipped, R1, R2 and R3 status, CDG hop count and call-is-cond flag |
| 3 | `.venv/bin/python spikes/frontend-fitness/adjudication/tools/flow/classify.py <lang>` | the two `.adj.jsonl` files (overwritten) |
| 4 | `.venv/bin/python spikes/frontend-fitness/adjudication/tools/flow/metrics.py`; `gocontra.py` | `metrics.json` (§a2.2.5 strata, omission rates, census reach); the Go Part-F-invalid pairs |

The three projection variants:
- **R1** (`rdrelax.sc`, `d2` records): `defsOf` also follows a REACHING_DEF from an earlier, non-assignment same-name identifier.
- **R2** (`rdfix.sc`, `d4` records): R1, plus the receiver of `<init>` is defined by that call.
- **R3**: R2, plus frontend-synthetic calls are transparent. This is a scoring-time relaxation in `variants.py`, not a Joern variant.

Each probe also emits the shipped walk (`d` records), which reproduces §a2.1.
Step 2 requires exactly one facts file in each of `run/<lang>/` and `runfix/<lang>/`.

The `rdDrop` stats record counts the reaching-definition sources the shipped walk drops. The attribution has three layers:
- the counterfactual verdicts in `variants.py`;
- the source-pattern rules and the explicit tables `EP`, `CS_DROPPED` and `OV` in `classify.py`, read from source against the projection's points;
- the spurious verdicts in `spur_verdicts.py`, which were judged by hand from source.

`cdgprobe.sc` and `idprobe.sc` are the one-off CFG/CDG and reaching-definition probes cited in §a2.2.2.
