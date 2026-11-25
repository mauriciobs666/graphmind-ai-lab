# Connected code graph P1 — AC-5 gate, CO-9 dev rehearsal at the record commit (report)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1 close-out, CO-9)

CPG: considered, not relevant — a black-box execution of the committed AC-5 gate harness on the dev gold against a throwaway store; no test-gap query was needed.

**Rehearsal count: dev rehearsal #1 at this tree** (tree ordinal `t1`; record `ac5-gate-1`, F `db7f7f45`, R `f4ef2239`). No tuning against dev, no fix, no earlier run at this tree. Dev result is a fit statistic, never AC-5 evidence.

## Verdict

**Entry-criterion item 1 (listing): MET on dev.** Every dev item of a gated stratum is a hit, a withheld shape or a user-confirmed key-error; 0 misses, 0 harness errors, 0 engine errors, 0 stops. **Entry-criterion item 3 (labelled Q <= 4): NOT MET at this time** (no labelled Q exists; reads NOT MET per the record until the separate labelling unit supplies it). The rehearsal is therefore not yet passed as a whole: it waits on the labelling unit.

## Setup and commands

- Tree: `git rev-parse HEAD` = `85211d57`, a descendant of R `f4ef2239`; `git diff f4ef2239 HEAD -- cpg` lists nothing (the one commit touches only `docs/plans/connected-code-graph-coordination.md`); `git status --short cpg` empty. The harness ran on a clone of that HEAD (scratch `.../scratchpad/co9/repo`, same `cpg/` tree as R), as the earlier rehearsal did; the repo itself was never written to by the run.
- Store: `CCG_FALKORDB_PORT=16590 cpg/infra/ccg-falkordb.sh --throwaway up` (exit 0; `ccg-falkordb-tw-16590`, 127.0.0.1:16590, `TIMEOUT_MAX` 10000, lab-standard `requirepass`). Port checked free with `ss -ltn` first. Four ACL users created by `ccg.store.acl.create_users` into a private `CCG_CREDENTIALS_DIR` (exit 0, 4 users; the developer's `~/.config/ccg/keys` mtimes unchanged).
- Custodian-style work dir: `HOME=<scratch>/co9/home`, custodian `<scratch>/co9/home/.local/share/ccg-eval/rehearsal`, holding the dev list (126 records: the five dev files) and the S47 probe map (1 dev unit probe). The real `~/.local/share/ccg-eval/` was never read.
- Run (exit 0, wall 952 s, ~16 min; no workaround, default deps): `gate_run.py run --gate-id ac5-gate-1 --lists <dev-all.jsonl> --custodian <cust> --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port 16590` -> `{"asked": 31, "label": "first-contact", "records": 126}`. The harness requires `--gate-id` to equal the record's stratum `ac5-gate-1` (`find_prereg_line`); the record's `ac5-gate-1-dev-t1` is the labelling gateId/seed only, used below. The earlier defects D1-D4 did not recur: the gate build passed on a password-protected store with all units built, no workaround.
- Probe (exit 0): `gate.py dev-probe --run run-<key>.json --out dev-probe.json --repo <clone> --system graphmind-ai-lab --host 127.0.0.1 --port 16590` -> `{"complete": true, ...}` (below).
- Analysis (exit 0): `analyze.py` (scratch, read-only: `gate.score_gate` on the committed dev gold, the predicate, the key-error record, answers from `run.json`).
- Sample prep (exit 0): `prep_sample.py` (scratch, `noise.build_frame/allocate/draw`, no labelling).
- Teardown (exit 0): `CCG_FALKORDB_PORT=16590 ccg-falkordb.sh --throwaway down` -> removed. `docker ps -a` afterwards: `ccg-falkordb`, `falkordb-dev`, `hopeful_brattain` untouched; `ccg-falkordb-tw-16390` (not mine) still present; port 16590 free.

## Entry criterion item 1: per stratum (dev probe, counts)

| Stratum | Items | Hits | Withheld shape (not listed) | Key-error (not listed) | Misses |
|---|---|---|---|---|---|
| http | 18 | 18 | 0 | 0 | 0 |
| mcp | 20 | 20 | 0 | 0 | 0 |
| scripts | 49 | 39 | 8 (B 1, C 2, I 1, J 2, K 1, M 1) | 2 | 0 |

Scripts detail: the predicate flags 10 dev items (B 2, C 2, I 2, J 2, K 1, M 1); the engine lists 2 of them (one B, one I), reported as over-withholding on dev, counted among the 39 hits. Of the 3 confirmed key-errors, 1 is listed (a hit) and 2 are not. No missed dev item exists, so there is no id list to give. `hit@100` equals `hit` in every stratum (18, 20, 39).

## Scripts hits: concordant vs coincident

`H_scripts = 39 (concordant 35, coincident 4)`. The 4 coincident hits ("listed through a prose caller"; ids `7fc6a23c`, `c649a7ac`, `d105f096`, `f7a50995`, the last a confirmed key-error item). Per the record's claim rule, the sentence "the scripts resolver recovers cross-script links" is **not** written, because coincident hits exist; the verdict itself follows EP §5.2 unchanged. Dev baseline at CO-2c was 34 hits (30/4).

## Label-free burden block A (listed = items + group members, counts)

| Scope | Answers | Listed | Median | p90 | Max | Possible by dist 1 / 2 / >=3 | Confirmed by dist 1 / 2 / >=3 | Prompt-mention chains (possible 1 / 2) | Hits | J = listed/hits | Largest answer share | Answers >100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| http | 12 | 73 | 5.5 | 8 | 10 | 67 / 0 / 0 | 6 / 0 / 0 | 1 / 0 | 18 | 4.06 | 0.137 | 0 |
| mcp | 3 | 84 | 22 | 42 | 42 | 61 / 0 / 0 | 23 / 0 / 0 | 49 / 0 | 20 | 4.20 | 0.500 | 0 |
| scripts | 18 | 477 | 25 | 48 | 48 | 87 / 374 / 0 | 16 / 0 / 0 | 57 / 230 | 39 | 12.23 | 0.101 | 0 |
| pooled | 31 | 604 | 17 | 46 | 48 | 201 / 374 / 0 | 29 / 0 / 0 | 104 / 230 | 77 | 7.84 | 0.080 | 0 |
| pooled without scripts | 15 | 157 | 6 | 22 | 42 | 128 / 0 / 0 | 29 / 0 / 0 | 50 / 0 | 38 | 4.13 | 0.268 | 0 |

p90 by index `-(-9*n//10)`. Answers are per asked anchor (hits count items; one answer can serve several items).

## Dev noise figures

- **Labelled Q (pooled and scripts): not available; reads NOT MET** until the labelling unit runs. No labelled figure, possible precision, wrong-`possible` bound or S1 reduction was computed or printed: each needs labels or a baseline, and the record's S1 rule (aggregate noise only, nothing before the record) gives no licence to print a reduction from this run; I did not construct a baseline.
- **Q_ceiling** = (listed - hit-matched)/hit-matched, an upper bound, never noise: scripts 11.23, pooled 6.84, pooled without scripts 3.13, http 3.06, mcp 3.20. Scripts and pooled exceed 4, so counts alone cannot show the bound is met; the labelled Q decides (record, recall ruling §2B).
- The ceiling counts every listed item not matched to a gold item as wrong; the gold is incomplete.

## Prepared, not run: labelling sample (custodian-style work dir, outside the repo)

`<scratch>/co9/labelling/` = `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/co9/labelling/` (README.txt, `exclusions.json`, and per draw `answers.jsonl`, `noise-frame.jsonl`, `noise-sample.jsonl`, `noise-census.jsonl`, `noise-out-of-scope.json`, `seed.txt`). gateId `ac5-gate-1-dev-t1`, snapshot id `4001b00c4d8e3b72fb5ffb5784d0b27d`.

| Draw | Seed | Answers | Frame | Census (confirmed) | Possible in frame / strata | Sample |
|---|---|---|---|---|---|---|
| pooled (all 31 dev answers) | `ccg-noise-v1|ac5-gate-1-dev-t1|4001b00c...|` (snapshot id, no suffix) | 31 | 604 | 29 | 575 / 8 | 100 |
| scripts-only (18 answers) | same seed + `|scripts` | 18 | 477 | 16 | 461 / 8 | 60 |

Method: EP §11.2 allocation (`noise.allocate`, min 5 per (reason, band) stratum, target 100 pooled, 60 scripts-only; the CLI `draw` has a fixed target of 100, so the scripts draw called the module functions directly), hash-sort draw. Both floors are met (>=100 and >=60 possible items). Not labelled; the spot-check draw (`noise.py spot`) comes after labelling. A rerun at this tree must reuse this seed and these labels. Note the sample is derived from this run's answers only; it was drawn on dev answers (no held-out data).

## Observations and open points

1. The harness `--gate-id` must be `ac5-gate-1`, not the record note's `ac5-gate-1-dev-t<j>`; the note's wording ("rehearsal gate id") is only right for the labelling seed. Suggest the note says so.
2. Dev-fit caveat applies: the withheld set and predicate came after the CO-1 read of all 49 dev items.
3. Not done by design: `freeze-check`, any held-out verb, the custodian dirs, committing.
4. The run left its `gate-*` build, work and key dirs and the marker/lock under the scratch custodian root (a rerun at this tree needs `--resume` semantics or clean scratch; a probe rerun reuses `run-<key>.json`). Artifacts: scripts and logs in `.../scratchpad/co9/` (`log/*.out|err|rc`, `analyze.json` output at `log/analyze.json`, `dev-probe.json`).
