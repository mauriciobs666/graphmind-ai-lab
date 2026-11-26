# Connected code graph P1 — AC-5 gate, CO-9 probe-producing dev rehearsal at record R4 (report)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1 close-out, CO-9)

CPG: considered, not relevant — a black-box execution of the committed AC-5 gate harness on the dev gold against a throwaway store; no test-gap query was needed.

**Not a tree and not a labelled rehearsal** (`noiseFix.probeRehearsal`): same engine tree F2 `cccf3328`, same dev gold, lists and key-errors as t3; it writes the dev-probe the held-out run needs and runs the listed-set comparator. Nothing was drawn, labelled or spot-checked. Record `ac5-gate-1-r4`, commit `bbaf281d`; counters unchanged (noise-motivated fixes 1 of cap 1; dev-noise trees 3 of cap 3). The dev result is a fit statistic, never AC-5 evidence.

## Verdict

**Comparator exit 0: the listed set is identical to t3's.** Entry item 1 (listing) MET on dev: http 18/18, mcp 36/36, scripts 64 = 50 hits + 10 withheld-not-listed + 4 key-errors-not-listed; 0 misses, 0 harness errors, 0 engine errors, 0 stops. No newly unlisted, no changed item, no lost hit beyond t3. The probe carries `contentSnapshotId`. No stop condition fired.

## Commands (all exit 0)

`S` = `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/co9-r4`.

| Step | Command / observation |
|---|---|
| Tree | `git rev-parse HEAD` = `bbaf281d` (R4 itself); `git diff --name-only bbaf281d HEAD -- cpg` empty; `git status --short cpg` clean |
| Clone | `git clone --no-hardlinks` + `git checkout bbaf281d` into `$S/repo` (the build is at EVAL_COMMIT regardless) |
| Store | `ss -ltn` / `docker ps -a` first: 16650 free. `CCG_FALKORDB_PORT=16650 cpg/infra/ccg-falkordb.sh --throwaway up` -> `ccg-falkordb-tw-16650`; private `CCG_CONFIG_DIR=$S/cfg`, `CCG_CREDENTIALS_DIR=$S/cred`, both mode 700 |
| Provision | `02_provision.py`: 4 ACL users, 162 dev records, 3 unit probes, scratch custodian under `$S/home/.local/share/ccg-eval/rehearsal` (real custodian dirs never read) |
| Run | `gate_run.py run --gate-id ac5-gate-1-r4 --lists dev-all.jsonl --custodian <cust> --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port 16650` -> `{"asked": 34, "label": "first-contact", "records": 162}`; **wall 1102 s** |
| Probe | `gate.py dev-probe --run <run> --out $S/dev-probe.json --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port 16650` -> `{"complete": true, ...}` |
| Comparator | `python cpg/eval/results/co9-t3/compare_listed.py --run $S/run.json` (from the clone, repo venv python), **exit 0**, output below |
| Analysis | `analyze.py` (read-only scoring and burden block) exit 0 |
| Teardown | `ccg-falkordb.sh --throwaway down` -> removed; `ccg-falkordb`, `falkordb-dev`, `ccg-falkordb-tw-16390` untouched |

Not run: `freeze-check`, any held-out verb, custodian dirs, noise samples, labelling, committing.

## Comparator output (verbatim)

```
{"added":0,"baselineItems":503,"baselineMatchesPinned":true,"chainChanged":0,"duplicatePairs":0,"fieldsChanged":0,"groupDiffs":0,"groups":0,"identical":true,"items":503,"removed":0,"sha256Equal":true}
```
Exit status 0.

## Snapshot ids and artefacts

- Content snapshot id (probe and run `contentSnapshotId`): `4e6bb90c24ccb833d76bd4b39d6b5b5a`.
- Stamped snapshot id (run and probe `snapshotId`): `0d06c30304d0ffbc79d366abef245fcf` (t3: `a965482a84f90cecff7b66a484642e38`; differs by design).
- Probe `recordCommit`: `bbaf281daf0f6963db2ff00ea6f8ec476328a984`.
- Persisting artefacts: `$S/dev-probe.json`, `$S/run.json` (copy of the custodian's `run-*.json`; the original and its answers file stay under the scratch custodian `$S/home/.local/share/ccg-eval/rehearsal/`).
- Probe hit ids and key-error ids per stratum equal t3's (http, mcp, scripts: all equal).

## Raw hits per stratum

| Stratum | Items | Hits | Withheld (not listed) | Key-error (not listed) | Misses |
|---|---|---|---|---|---|
| http | 18 | 18 | 0 | 0 | 0 |
| mcp | 36 | 36 | 0 | 0 | 0 |
| scripts | 64 | 50 | 10 (B 2, C 2, I 2, J 2, K 1, M 1) | 4 | 0 |

`hit@100` = 18, 36, 50. The predicate flags 14 dev scripts items, 4 listed by the engine (counted among the 50). Raw counts only.

## Burden block A (counts; listed = items + group members)

| Scope | Answers | Listed | Median | p90 | Max | Possible by dist 1 / 2 / >=3 | Confirmed by dist 1 / 2 / >=3 | Prompt-mention chains (possible 1 / 2) | Hits | Listed per hit | Largest answer share | Answers >100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| http | 12 | 73 | 5.5 | 8 | 10 | 67 / 0 / 0 | 6 / 0 / 0 | 1 / 0 | 18 | 4.06 | 0.137 | 0 |
| mcp | 5 | 122 | 22 | 42 | 42 | 98 / 0 / 0 | 24 / 0 / 0 | 78 / 0 | 36 | 3.39 | 0.344 | 0 |
| scripts | 19 | 338 | 13 | 32 | 79 | 166 / 156 / 0 | 16 / 0 / 0 | 132 / 0 | 50 | 6.76 | 0.234 | 0 |
| pooled | 34 | 503 | 6.5 | 32 | 79 | 317 / 156 / 0 | 30 / 0 / 0 | 208 / 0 | 104 | 4.84 | 0.157 | 0 |
| pooled without scripts | 17 | 195 | 6 | 33 | 42 | 165 / 0 / 0 | 30 / 0 / 0 | 79 / 0 | 54 | 3.61 | 0.215 | 0 |

Identical to the t3 block. Q ceiling (upper bound, never noise): scripts 5.76, pooled 3.84, pooled without scripts 2.61, http 3.06, mcp 2.39. Labelled Q: not computed here; the t3 labelled Q carries only because the comparator exited 0.

## Observations

1. No harness defect; no workaround; default deps. Wall time 1102 s (t3: 936 s).
2. Scratch custodian holds the run's `gate-*` dirs and marker; a rerun needs clean scratch.
3. Run count at this record: probe-producing rehearsal #1 at R4; not a tree, not a counter.
