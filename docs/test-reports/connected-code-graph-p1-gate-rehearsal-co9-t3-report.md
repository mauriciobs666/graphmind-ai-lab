# Connected code graph P1 — AC-5 gate, CO-9 dev rehearsal #1 at record R3 (report)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1 close-out, CO-9)

CPG: considered, not relevant — a black-box execution of the committed AC-5 gate harness on the dev gold against a throwaway store; no test-gap query was needed.

**Rehearsal count: dev rehearsal #1 at record R3** (tree ordinal `t3`, the last dev-noise tree; record `ac5-gate-1-r3`, F2 `cccf3328`, R3 `31f60bf9`). Counters (`noiseFix.fixCounters`): noise-motivated engine fixes 1 of cap 1; dev-noise trees t1, t2, t3 of cap 3 (this is t3). No tuning against dev, no fix, no earlier run at this tree. The dev result is a fit statistic, never AC-5 evidence.

## Verdict

**Entry-criterion item 1 (listing): MET on dev.** Every dev item of a gated stratum is a hit, a withheld shape or a user-confirmed key-error: http 18/18, mcp 36/36, scripts 64 = **50 raw hits** + 10 withheld-not-listed + 4 key-errors-not-listed; 0 misses, 0 harness errors, 0 engine errors, 0 stops. **Stopping-rule listing clause: held** — the only dev items newly unlisted against the t2 hits are the four named (`7fc6a23c`, `c649a7ac`, `d105f096`, `f7a50995`); no fifth. **Item 3 (labelled Q <= 4): NOT MET at this time** (no labelled Q exists; reads NOT MET until the labelling unit supplies it). The rehearsal as a whole therefore waits on labelling; the stopping rule's Q clause is not yet decidable. Item 2 (scan) is not mine; items 5 and 6 per the record (no `freeze-check` run).

## Setup and commands (all exit 0 unless stated)

`S` = `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/co9-t3` (scripts copied from `.../co9-t2/` and adapted: port, gate ids, paths; originals untouched).

| Step | Command / observation | Exit |
|---|---|---|
| Tree | `git rev-parse HEAD` = `337910eb` (descendant of R3 `31f60bf9`; the only commit since, `337910eb`, touches `docs/plans/connected-code-graph-coordination.md`); `git diff --stat 31f60bf9 HEAD -- cpg` empty; `git status --short cpg` empty | 0 |
| Clone | `git clone --no-hardlinks <repo> $S/repo; git checkout 337910eb` (the harness ran on this committed-tree clone; `cpg/` identical to R3; the working repo was never written by the run) | 0 |
| Store | `ss -ltn` and `docker ps -a` first: 16620 free (6379, 6390, 16390 held by others). `CCG_FALKORDB_PORT=16620 cpg/infra/ccg-falkordb.sh --throwaway up` -> `ccg-falkordb-tw-16620`, 127.0.0.1:16620, TIMEOUT_MAX 10000, MAX_INFO_QUERIES 0, QUERY_MEM_CAPACITY 268435456; `CCG_CONFIG_DIR=$S/cfg` and `CCG_CREDENTIALS_DIR=$S/cred` both mode 700 (the developer's `~/.config/ccg` not used) | 0 |
| Provision | `02_provision.py` (reads the password from the private config dir): 4 ACL users; dev list `dev-all.jsonl` = 162 records (18+36+64+16+28); 3 dev unit probes; scratch custodian `$S/home/.local/share/ccg-eval/rehearsal` (the real `~/.local/share/ccg-eval/` never read) | 0 |
| Run | `gate_run.py run --gate-id ac5-gate-1-r3 --lists dev-all.jsonl --custodian <cust> --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port 16620` -> `{"asked": 34, "label": "first-contact", "records": 162}`; wall 936 s; no workaround, default deps | 0 |
| Probe | `gate.py dev-probe --run run-7c7636da....json --out dev-probe.json --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port 16620` -> `{"complete": true, ...}` | 0 |
| Analysis | `analyze.py` (scratch, read-only: `gate.score_gate` on the committed dev gold, predicate, key-error record; `log/analyze.json`) | 0 |
| Samples | `prep_sample.py` (`noise.build_frame/allocate/draw`; nothing labelled), then `unchanged.py` (t3 vs t2 pair map) | 0, 0 |
| Teardown | `CCG_FALKORDB_PORT=16620 ccg-falkordb.sh --throwaway down` -> removed. `docker ps -a` afterwards: `ccg-falkordb`, `falkordb-dev`, `hopeful_brattain`, `ccg-falkordb-tw-16390` untouched; 16620 free | 0 |

Snapshot id: `a965482a84f90cecff7b66a484642e38`. The probe's harness shas and dev gold shas equal t2's; `recordCommit` `31f60bf974f8dbd2a542b15c23e0faa9baec6513`. `freeze-check`, any held-out verb, the custodian dirs, and committing: not done. The harness `--gate-id` is `ac5-gate-1-r3` everywhere; `ac5-gate-1-dev-t3` is the noise-draw label only. No setup failure this time.

## Entry criterion item 1: per stratum (dev probe, counts)

| Stratum | Items | Hits | Withheld shape (not listed) | Key-error (not listed) | Misses |
|---|---|---|---|---|---|
| http | 18 | 18 | 0 | 0 | 0 |
| mcp | 36 | 36 | 0 | 0 | 0 |
| scripts | 64 | **50** | 10 (B 2, C 2, I 2, J 2, K 1, M 1) | 4 | 0 |

Matches the record's expected arithmetic exactly (50 + 10 + 4 = 64). The predicate flags 14 dev scripts items; the engine lists 4 of them (over-withholding on dev, counted among the 50 hits) and does not list 10. The 4 key-errors: `7fc6a23c`, `f7a50995`, `e8c36e15`, `f33cbe42` (probe `keyErrorIds`). `hit@100` equals `hit` per stratum (18, 36, 50). Scripts raw hits are 50 of 64; no concordant-only percentage is stated, and the sentence "the resolver recovers cross-script links" is not written (the record forbids it at P1).

## Set difference against the t2 hits (stop-rule check)

Computed from `hitIds` of the t2 probe (`.../co9-t2/dev-probe.json`, snapshot `20d29e20...`) and the t3 probe (`$S/dev-probe.json`).

| Stratum | t2 hits | t3 hits | In t2 hits, not in t3 (newly unlisted) | In t3, not in t2 |
|---|---|---|---|---|
| http | 18 | 18 | none | none |
| mcp | 36 | 36 | none | none |
| scripts | 54 | 50 | `7fc6a23c205a1a82a393d0918dbae59c`, `c649a7aca2de27884ddee78e4cd099b4`, `d105f096e63283a09511fcf69297d3ed`, `f7a509957611e2328bb39ac439749782` | none |

Exactly the four named items; no fifth newly unlisted item. `7fc6a23c` and `f7a50995` are now confirmed key-errors, `c649a7ac` is shape B and `d105f096` shape I (withheld), as the record says. First exposure (moved items, descriptive): mcp 16 of 16 and scripts 15 of 15 still hit.

## Burden block A (label-free; listed = items + group members, counts)

| Scope | Answers | Listed | Median | p90 | Max | Possible by dist 1 / 2 / >=3 | Confirmed by dist 1 / 2 / >=3 | Prompt-mention chains (possible 1 / 2) | Hits | Listed per hit | Largest answer share | Answers >100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| http | 12 | 73 | 5.5 | 8 | 10 | 67 / 0 / 0 | 6 / 0 / 0 | 1 / 0 | 18 | 4.06 | 0.137 | 0 |
| mcp | 5 | 122 | 22 | 42 | 42 | 98 / 0 / 0 | 24 / 0 / 0 | 78 / 0 | 36 | 3.39 | 0.344 | 0 |
| scripts | 19 | 338 | 13 | 32 | 79 | 166 / 156 / 0 | 16 / 0 / 0 | 132 / 0 | 50 | 6.76 | 0.234 | 0 |
| pooled | 34 | 503 | 6.5 | 32 | 79 | 317 / 156 / 0 | 30 / 0 / 0 | 208 / 0 | 104 | 4.84 | 0.157 | 0 |
| pooled without scripts | 17 | 195 | 6 | 33 | 42 | 165 / 0 / 0 | 30 / 0 / 0 | 79 / 0 | 54 | 3.61 | 0.215 | 0 |

p90 by index `-(-9*n//10)`. For reference t2 scripts: listed 556, distance-2 possible 374, prompt-mention distance-2 230, largest share 0.142; t3 pooled 503 listed (t2 721), largest answer still 79 items, answers over 100: 0. Largest-answer share beside Q: pooled 0.157, scripts 0.234 (t2: 0.110, 0.142).

**Unexplored** (the answer's `depthBound.unexplored`, counts): total 978 over the 34 answers; 14 answers carry a non-zero value, all scripts-anchored (19 scripts answers: 978, maximum 184); the 15 remaining answers carry 0. t2 comparison not computed (the t2 store is gone).

## Dev noise figures

- **Labelled Q (pooled and scripts): not available; reads NOT MET** until the labelling unit runs. No labelled figure, possible precision, S1 reduction, replay or gold-aware sensitivity was computed or printed (the record forbids it from this run).
- **Q_ceiling** = (listed - hit-matched)/hit-matched, an upper bound, never noise: scripts 5.76, pooled 3.84, pooled without scripts 2.61, http 3.06, mcp 2.39 (t2: scripts 9.30, pooled 5.68). Scripts is above 4, so counts alone cannot show the bound met; the labelled Q decides. The ceiling counts every unmatched listed item as wrong; the gold is incomplete.

## Prepared, not run: labelling samples and the unchanged-pair map (outside the repo)

Directory `S/labelling/` = `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/co9-t3/labelling/` (`README.txt`, `exclusions.json`; per draw `pooled/` and `scripts/`: `answers.jsonl`, `noise-frame.jsonl`, `noise-sample.jsonl`, `noise-census.jsonl`, `noise-out-of-scope.json`, `seed.txt`, `unchanged-pairs-sample.jsonl`, `unchanged-pairs-census.jsonl`). Label gateId `ac5-gate-1-dev-t3`, snapshot id `a965482a84f90cecff7b66a484642e38`.

| Draw | Seed | Answers | Frame | Census (confirmed) | Possible in frame / strata | Sample |
|---|---|---|---|---|---|---|
| pooled (all 34 dev answers) | `ccg-noise-v1|ac5-gate-1-dev-t3|a965482a84f90cecff7b66a484642e38` | 34 | 503 | 30 | 473 / 6 | 100 |
| scripts-only (19 answers) | same seed + `|scripts` | 19 | 338 | 16 | 322 / 6 | 60 |

Floors met (>= 100 and >= 60 possible items, min 5 per stratum). Method as at t2 (EP §11.2 allocation, hash-sort draw; the scripts draw calls the module functions directly because the CLI target is fixed at 100). A rerun at this tree must reuse this seed and these labels; the spot-check draw comes after labelling.

**Unchanged-pair map** (`unchanged.py`): a t3 pair (anchorId, itemId) is "unchanged" when the t2 answers hold the same anchor, item and an identical canonical-JSON `chain`. Each row of `unchanged-pairs-*.jsonl` carries a status and, for a reusable one, `t2LabelKey` (anchorId, itemId, the t2 `noise-labels.jsonl` label; the t2 labels of both draws agree, 0 conflicts). Counts:

| Draw | Rows | Unchanged and labelled at t2 (label reusable) | Unchanged but never labelled at t2 (needs a fresh label) | Chain changed (relabel blind) | New pair (absent at t2) |
|---|---|---|---|---|---|
| pooled sample | 100 | 24 | 73 | 3 | 0 |
| pooled census | 30 | 30 | 0 | 0 | 0 |
| scripts sample | 60 | 20 | 38 | 2 | 0 |
| scripts census | 16 | 16 | 0 | 0 | 0 |

So the pooled draw reuses 54 of 130 t2 labels and needs 76 fresh labels; the scripts draw reuses 36 of 76 and needs 42 fresh labels (by hash: 73+3 and 38+2 sample rows; the census rows are all reusable). No t3 pair is absent from t2 (the fix only removes items), but the t3 samples are fresh draws, so most sampled pairs were not in t2's samples and carry no t2 label ("unchanged" there means same triple, not "labelled"). The reused-label share beside Q will be 54/130 (pooled) and 36/76 (scripts) before spot-check selection.

## Observations

1. No harness defect; the gate build passed with all units built, no workaround. The run was faster than the 32 min estimate (936 s).
2. The run left its `gate-*` dirs and marker under the scratch custodian root (a rerun needs clean scratch or resume semantics).
3. Dev-fit caveat applies: the withheld set and predicate came after the CO-1 read of the first 49 dev scripts items; the 15 moved items joined dev after the contact move; the noise rule came after the t2 labels. A t3 pass is a refit statistic.
4. Counters stated: rehearsal #1 at R3; noise-motivated fixes 1 of cap 1; dev-noise trees 3 of cap 3 (t3 last). Artifacts (scripts, `log/*.out|err|rc`, `log/analyze.json`, `dev-probe.json`, the run file) are under `S/`.
