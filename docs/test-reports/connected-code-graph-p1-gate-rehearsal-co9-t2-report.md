# Connected code graph P1 — AC-5 gate, CO-9 dev rehearsal #1 at record R2 (report)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1 close-out, CO-9)

CPG: considered, not relevant — a black-box execution of the committed AC-5 gate harness on the dev gold against a throwaway store; no test-gap query was needed.

**Rehearsal count: dev rehearsal #1 at record R2** (tree ordinal `t2`; record `ac5-gate-1-r2`, F `db7f7f45`, R2 `67ea5fd4`). No tuning against dev, no fix, no earlier run at this tree. The t1 rehearsal (`docs/test-reports/connected-code-graph-p1-gate-rehearsal-co9-report.md`) is superseded by the contact move. Dev result is a fit statistic, never AC-5 evidence.

## Verdict

**Entry-criterion item 1 (listing): MET on dev.** Every dev item of a gated stratum is a hit, a withheld shape or a user-confirmed key-error: http 18/18, mcp 36/36, scripts 64 = 54 hits + 8 withheld-not-listed + 2 key-errors-not-listed; 0 misses, 0 harness errors, 0 engine errors, 0 stops. **Item 3 (labelled Q <= 4): NOT MET at this time** (no labelled Q exists; reads NOT MET per the record until the labelling unit supplies it). The rehearsal as a whole is therefore not yet passed; it waits on labelling. Item 2 (scan) is not mine; items 5 and 6 per the record.

## First-exposure report (the 31 moved mcp and scripts dev items; counts, descriptive, not AC-5 evidence)

Moved items = dev-gold itemIds absent from `git show f4ef2239:cpg/eval/dev/ac5-{mcp,scripts}.jsonl` (mcp 36 - 20 = 16; scripts 64 - 49 = 15).

| Stratum | Moved items | Hits | Withheld shapes (not listed) | Misses |
|---|---|---|---|---|
| mcp | 16 | 16 | n/a (predicate is scripts-only) | 0 |
| scripts | 15 | 15 | 0 | 0 |

Scripts detail: the predicate flags 4 of the 15 moved scripts items (the record's "4 flagged, all family C"); the engine lists all 4, so they count among the 15 hits and are over-withholding on dev, none is a withheld-not-listed item. No miss exists, so there is no miss diagnosis to give. Zero-miss over 3 clusters bounds nothing about held-out (record §6: 0.44); read as an optimistic first look.

## Setup and commands (all exit 0 unless stated)

| Step | Command (scratch dir `S=.../scratchpad/co9-t2`, copied from `co9/`, originals untouched) | Exit |
|---|---|---|
| Tree | `git rev-parse HEAD` = `81fba860` (descendant of R2 `67ea5fd4`); `git diff --stat 67ea5fd4 HEAD -- cpg` empty; `git status --short cpg` empty | 0 |
| Clone | `git clone --no-hardlinks <repo> $S/repo; git checkout 81fba860` (the harness ran on this committed-tree clone, `cpg/` identical to R2; the repo itself was never written by the run) | 0 |
| Store | `ss -ltn` and `docker ps -a` first: 16610 free. `CCG_FALKORDB_PORT=16610 cpg/infra/ccg-falkordb.sh --throwaway up` -> `ccg-falkordb-tw-16610`, 127.0.0.1:16610, TIMEOUT_MAX 10000 (config and password dir private under `$S/cfg`, so the developer's `~/.config/ccg` was not touched; key mtimes unchanged) | 0 |
| Provision | `02_provision.py`: 4 ACL users in a private `CCG_CREDENTIALS_DIR`; dev list `dev-all.jsonl` = 162 records (18+36+64+16+28), S47 probe map with 3 dev unit probes, in the scratch custodian `$S/home/.local/share/ccg-eval/rehearsal` (the real `~/.local/share/ccg-eval/` never read). First two provision attempts failed on my own setup (password file read from `~/.config`; credentials dir needed mode 700); no store state mutated by them | 1, 1, then 0 |
| Run | `gate_run.py run --gate-id ac5-gate-1-r2 --lists dev-all.jsonl --custodian <cust> --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port 16610` -> `{"asked": 34, "label": "first-contact", "records": 162}`; wall 916 s, no workaround, default deps | 0 |
| Probe | `gate.py dev-probe --run run-7c7636da....json --out dev-probe.json ...` -> `{"complete": true, ...}` | 0 |
| Analysis | `analyze.py` (scratch, read-only: `gate.score_gate` on the committed dev gold, predicate, key-error record) | 0 |
| Samples | `prep_sample.py` (scratch, `noise.build_frame/allocate/draw`; nothing labelled) | 0 |
| Teardown | `CCG_FALKORDB_PORT=16610 ccg-falkordb.sh --throwaway down` -> removed. `docker ps -a` afterwards: `ccg-falkordb`, `falkordb-dev`, `hopeful_brattain`, `ccg-falkordb-tw-16390` untouched; 16610 free | 0 |

Snapshot id: `20d29e2096dbc916e1987f4c3d43b921`. `freeze-check`, any held-out verb, the custodian dirs and committing: not done. The harness `--gate-id` is `ac5-gate-1-r2` everywhere; `ac5-gate-1-dev-t2` is the noise-draw label only.

## Entry criterion item 1: per stratum (dev probe, counts)

| Stratum | Items | Hits | Withheld shape (not listed) | Key-error (not listed) | Misses |
|---|---|---|---|---|---|
| http | 18 | 18 | 0 | 0 | 0 |
| mcp | 36 | 36 | 0 | 0 | 0 |
| scripts | 64 | 54 | 8 (B 1, C 2, I 1, J 2, K 1, M 1) | 2 | 0 |

Scripts: the predicate flags 14 dev items (B 2, C 6, I 2, J 2, K 1, M 1); the engine lists 6 of them (over-withholding on dev, counted among the 54 hits), 8 not listed. Of the 3 confirmed key-errors 1 is listed (a hit), 2 are not. `hit@100` equals `hit` per stratum (18, 36, 54).

## Scripts hits: concordant vs coincident

`H_scripts = 54 (concordant 50, coincident 4)`. The 4 coincident hits ("listed through a prose caller") are ids `7fc6a23c`, `c649a7ac`, `d105f096`, `f7a50995`, the same four as at t1; none is among the moved items. Per the record's claim rule the sentence "the scripts resolver recovers cross-script links" is **not** written, because coincident hits exist; the verdict follows EP §5.2 unchanged.

## Label-free burden block A (listed = items + group members, counts)

| Scope | Answers | Listed | Median | p90 | Max | Possible by dist 1 / 2 / >=3 | Confirmed by dist 1 / 2 / >=3 | Prompt-mention chains (possible 1 / 2) | Hits | J = listed/hits | Largest answer share | Answers >100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| http | 12 | 73 | 5.5 | 8 | 10 | 67 / 0 / 0 | 6 / 0 / 0 | 1 / 0 | 18 | 4.06 | 0.137 | 0 |
| mcp | 5 | 122 | 22 | 42 | 42 | 98 / 0 / 0 | 24 / 0 / 0 | 78 / 0 | 36 | 3.39 | 0.344 | 0 |
| scripts | 19 | 556 | 27 | 48 | 79 | 166 / 374 / 0 | 16 / 0 / 0 | 132 / 230 | 54 | 10.30 | 0.142 | 0 |
| pooled | 34 | 721 | 18 | 47 | 79 | 317 / 374 / 0 | 30 / 0 / 0 | 208 / 230 | 108 | 6.68 | 0.110 | 0 |
| pooled without scripts | 17 | 195 | 6 | 33 | 42 | 165 / 0 / 0 | 30 / 0 / 0 | 79 / 0 | 54 | 3.61 | 0.215 | 0 |

p90 by index `-(-9*n//10)`. Answers are per asked anchor (hits count items; one answer can serve several items).

## Dev noise figures

- **Labelled Q (pooled and scripts): not available; reads NOT MET** until the labelling unit runs. No labelled figure, possible precision, S1 reduction or baseline was computed or printed (the record forbids it from this run).
- **Q_ceiling** = (listed - hit-matched)/hit-matched, an upper bound, never noise: scripts 9.30, pooled 5.68, pooled without scripts 2.61, http 3.06, mcp 2.39. Scripts and pooled exceed 4, so counts alone cannot show the bound met; the labelled Q decides. The ceiling counts every unmatched listed item as wrong; the gold is incomplete.

## Prepared, not run: labelling samples (outside the repo)

Directory `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/co9-t2/labelling/` (README.txt, `exclusions.json`; per draw `pooled/` and `scripts/`: `answers.jsonl`, `noise-frame.jsonl`, `noise-sample.jsonl`, `noise-census.jsonl`, `noise-out-of-scope.json`, `seed.txt`). Label gateId `ac5-gate-1-dev-t2`, snapshot id `20d29e2096dbc916e1987f4c3d43b921`.

| Draw | Seed | Answers | Frame | Census (confirmed) | Possible in frame / strata | Sample |
|---|---|---|---|---|---|---|
| pooled (all 34 dev answers) | `ccg-noise-v1|ac5-gate-1-dev-t2|20d29e2096dbc916e1987f4c3d43b921` | 34 | 721 | 30 | 691 / 8 | 100 |
| scripts-only (19 answers) | same seed + `|scripts` | 19 | 556 | 16 | 540 / 8 | 60 |

Method as at t1 (EP §11.2 allocation, min 5 per stratum, hash-sort draw; the scripts draw calls the module functions directly because the CLI target is fixed at 100). Both floors met (>=100 and >=60 possible items). A rerun at this tree must reuse this seed and these labels; the spot-check draw comes after labelling. Run artefacts (scripts, logs, `analyze.json`, `dev-probe.json`, the run file) are under `.../scratchpad/co9-t2/` (`log/`), and the run left its `gate-*` dirs and marker under the scratch custodian root (a rerun needs clean scratch or `--resume` semantics).

## Observations

1. No harness defect recurred (t1 D1-D4); the gate build passed with all units built, no workaround.
2. Setup trap (mine, not a product defect): the `ccg-falkordb.sh` password and config dir follows `CCG_CONFIG_DIR`, and the ACL credentials dir must be mode 700; a script that reads `~/.config/ccg/falkordb-admin.pass` while the store was started with a private `CCG_CONFIG_DIR` fails authentication.
3. Dev-fit caveat applies: the withheld set and predicate came after the CO-1 read of the first 49 dev scripts items, and the 15 moved items joined dev only after the contact move.
4. Run count at this record: 1; no tuning, no fix.
