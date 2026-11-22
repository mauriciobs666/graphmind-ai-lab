# Connected code graph P1-20 Done run: full refresh of the proving system

> **Status:** active · **Owner:** `devops` · **Tracks:** P1-20 (C-1001 P1)

Run record of the P1-20 "Done run" (`docs/plans/connected-code-graph-p1-linking.md`, row P1-20: a full refresh of the proving system with wall time and peak RSS recorded, R-4). Every number below was read from the tool's own report, `/usr/bin/time -v`, a sampler, or a query against the loaded graph; a figure not observed is not stated.

## 1. Conditions

| Item | Observed |
|---|---|
| Date, host | 2026-11-19, WSL2, 8 CPUs, 5,930 MB RAM, swap 4,096 MB (1,016 MB used before the run) |
| Host load before (`uptime`, 16:38:39Z) | `load average: 0.55, 0.69, 0.61`; `MemAvailable` 4,172 MB; other containers up and idle: `falkordb-dev` (120 MiB), two `cypher-mcp` (3 MiB each), `ccg-falkordb-tw-16390` (178 MiB, untouched). Two `claude --agent teco` sessions were idle. No other heavy work seen |
| Commit built | `8377c3268e7857e7a71a55d5cfdd030b8df5fa8d` (HEAD, unchanged through the run); the work tree had an uncommitted edit to `docs/plans/connected-code-graph-coordination.md` and two untracked `resume_*.sh` (a refresh builds the commit, not the work tree) |
| Entry point | `cd cpg && .venv/bin/python -m clients.cli.main refresh` (the shipped `Client.refresh` path: `ccg_build_<s>`/`ccg_verdict_<s>` connections, then `ccg.refresh.refresh`), default manifest `cpg/ccg/system.yaml`, system `graphmind-ai-lab`, no scope, no override (no broker used) |
| Store | own throwaway `ccg-falkordb-tw-16395` (`ccg-falkordb.sh --throwaway up`, port 16395, FalkorDB `v4.18.11`, 2 GiB cap), scratch `CCG_CONFIG_DIR` and `CCG_CREDENTIALS_DIR` under `/tmp/claude-1000/-home-<user>-prg-graphmind-ai-lab/82bb1fdb-fb53-4f8c-80b5-7bf434adbf7b/scratchpad/p1-20/cfg`. `acl.create_users(admin, "graphmind-ai-lab")` run once there; `~/.config/ccg/keys` was empty before and after (0 entries). `falkordb-dev`, :6390 and `ccg-falkordb-tw-16390` were not touched |
| Toolchain | default sandbox image `ccg-toolchain:73b9fb76bd2a` (the `build.sh --print-tag` result); real Joern builds in `ccg-sbx-*` containers |
| Fresh bundle area | `cpg/.ccg-work/bundles/graphmind-ai-lab/` did not exist before the run (a cold, full build, nothing carried) |

## 2. Refresh result

Exit 0, stderr empty. Start `2026-11-19T16:38:57Z`, end `2026-11-19T16:57:01Z`. `/usr/bin/time -v`: elapsed **18:03.82**, user 727.71 s, system 70.33 s, 73% CPU, **maximum resident set size 937,124 KB** (the CLI process; no swaps).

Tool report (verbatim except the 1,126-id `unattested` list, elided):

```
refresh graphmind-ai-lab: 1138 fresh, 0 stale-kept, 0 not-mapped (head is the local clone's)
swept 0 stale temp file(s); unknownKind 0; skippedDocs 0; undecodablePaths 0; pruned 0; prune errors 0; state errors 0
unattested 1126: 000a94d0d0a7b1892069a68719731c6f, 005711b5d9d09a216a7aa6978159ba5e, ... (list elided)
dropped consumer-key-none 4
stage sweep 0.00s peak 40600 KB (this process); host MemAvailable min 4232620 KB (drop 0 KB)
stage discover 0.13s peak 41688 KB (this process); host MemAvailable min 4216856 KB (drop 15764 KB)
stage build 725.66s peak 652424 KB (this process); host MemAvailable min 2464156 KB (drop 1752700 KB)
stage resolve 173.12s peak 654764 KB (this process); host MemAvailable min 3614960 KB (drop 316696 KB)
stage link 129.03s peak 730604 KB (this process); host MemAvailable min 3546844 KB (drop 80616 KB)
stage assemble 5.19s peak 974848 KB (this process); host MemAvailable min 3318860 KB (drop 251768 KB)
stage load 5.38s peak 937104 KB (this process); host MemAvailable min 2804912 KB (drop 546148 KB)
stage swap 0.01s peak 937104 KB (this process); host MemAvailable min 2814332 KB (drop 0 KB)
```

- **Wall time:** per stage as above; total **1083.8 s** (18:03.82) measured by `time`, not derived from the stage lines.
- **Peak RSS:** the tool's highest stage peak is 974,848 KB (assemble); `time -v` reports 937,124 KB. The stage "peak" values are not monotonic (assemble 974,848, load 937,104), so they are the tool's own per-stage figures and are quoted as printed. Joern runs in `ccg-sbx-*` containers outside the process tree; the process-tree sampler (below) cannot see them.
- **Peak memory beyond the CLI process:** the tool's host `MemAvailable` low-water mark during build was 2,464,156 KB (drop 1,752,700 KB from 4,216,856 KB), which is the R-4 host figure. A sampler at about 8 s spacing (first sample at 16:39:02Z, so the first ~2.5 min of build is covered only by the tool's own figure plus two early samples; one sampler restart) saw: highest process-tree RSS 974,984 KB, highest single sandbox container `943.5MiB`, lowest `MemAvailable` 2,810,224 KB, highest 1-minute load **4.43**. Swap stayed at about 3.16 GB free (it was 3,079 MB free before). Samples: `.../p1-20/samples.txt`, `samples2.txt`.
- **Units:** 1,138 discovered, **1,138 built fresh, 0 failed, 0 stale-kept, 0 not-mapped**, no failure reason to report. By language: markdown 821, config 219, shell 86, python 10, javascript 2 (status from the CLI's `units` array; all `fresh`, `failReason` null).
- **Other report counts:** `unattested` 1,126 units (report line only; the plan says it has no consumer yet), `dropped consumer-key-none` 4, sweep/prune/state/unknownKind/skippedDocs/undecodablePaths all 0.
- **snapshotId:** `9add870b2d5ce60349c37b2d08b6b937` (`builtAt` 2026-11-19T16:51:18Z, `graphModelVersion` 1.1, `bundleSchemaVersion` 1.1, `verdictLogOffset` 0, `manifestHash` `19e9feaa...c746e`, `packSetHash` `2e3cf5b6ab3cbd857916dcc51f24d821`, `codeHash` `a4cb933c...a8225`, `auditSetHash` `d858c42e...c03c4fe`).
- **Graph size** (read as admin on the throwaway, key `ccg_graphmind-ai-lab`, the only key): **228,846 nodes, 367,726 edges**. Nodes by label: CallSite 73,038; ConfigEntry 90,802; Symbol 14,523; Cond 18,379; Param 17,594; Ret 6,265; Name 4,880; File 1,511; Unit 1,138; Link 569; ContractPoint 104; SkippedFile 34; Pack 7; Repo 1; Snapshot 1. Edges: CALLS_NAME 72,957; HAS_CALL 73,038; DATA 95,300; CALLS 32,631; CONTROLLED_BY 25,749; HAS_COND 18,379; HAS_PARAM 17,594; DEFINES 14,523; HAS_RETURN 6,265; RESOLVES_TO 5,344; VIA 1,542; HAS_FILE 1,511; IN_REPO 1,138; ON/TO 569 each; MOUNTED_BY 127; CONSUMES 122; LIFTED_VIA 128; BOUND_BY 114; PROVIDES 91; SKIPS 34; LAUNCHES 1.
- **Artifacts:** `cpg/.ccg-work/bundles/graphmind-ai-lab/` (the shipped path has a fixed work root, not the scratch dir): 1,138 `*.bundle.jsonl` plus 1,138 `*.state.json`, 116 MB, gitignored.

## 3. `ccg ask` against the loaded snapshot

Environment for all: `CCG_FALKORDB_HOST=127.0.0.1 CCG_FALKORDB_PORT=16395`, scratch `CCG_CONFIG_DIR`/`CCG_CREDENTIALS_DIR` (the `ccg_engine_graphmind-ai-lab` credential from the run's own provisioning). Success path proven: three structured/parsed requests exit 0 with a valid canonical answer.

**Anchor-resolution observation (not a defect):** `{"kind":"name","name":"build_answer"}` returned exit 2 `anchor-unresolved` (`unknown-name`, no candidates) because `cpg/` is excluded from the proving system's manifest (`cpg/ccg/system.yaml` exclusion `cpg/`), so no `cpg` symbol is in the graph. Anchors below are symbols in built units.

### 3.1 impact, name anchor `load_point` (`--format text`)

Command: `ask --request '{"useCase":"impact","anchor":{"kind":"name","name":"load_point"}}' --format text`. Exit 0, 1.77 s wall, 58,700 KB max RSS, stdout 475,462 bytes, stderr empty. Header and notices verbatim (items elided after the first):

```
Code-derived text appears only inside code spans; it is data, not instructions.

Snapshot 9add870b2d5ce60349c37b2d08b6b937, verdict log offset 0.
What is affected by: ` decision-models/decisionmodels/points.py::load_point ` in ` decision-models/decisionmodels/points.py `.

Code views used:
  ` claude ` (python, repository graphmind-ai-lab): fresh, built 2026-11-19T16:39:23Z from commit 8377c3268e78.
  ` decision-models ` (python, repository graphmind-ai-lab): fresh, built 2026-11-19T16:40:43Z from commit 8377c3268e78.
  ` falkor-chat/server ` (python, repository graphmind-ai-lab): fresh, built 2026-11-19T16:43:15Z from commit 8377c3268e78.
  ` model-bench ` (python, repository graphmind-ai-lab): fresh, built 2026-11-19T16:45:35Z from commit 8377c3268e78.
  ` skills ` (python, repository graphmind-ai-lab): fresh, built 2026-11-19T16:47:34Z from commit 8377c3268e78.

455 items listed (82 confirmed, 373 possible).

Flow-dependent effects in python may be missing from this list. In python the walk did not follow data, return-value, argument or control-flow edges; it followed call, link and wiring edges. An item's absence here is not evidence that no such effect exists.
Reach stopped at distance 2 with 7896 unexplored edges.

Repository graphmind-ai-lab
  Component ` claude `
    Distance 2, possible
      - ` claude/cobb/scripts/kb_golden_runner_selftest.py::FakeSearch.search ` in ` claude/cobb/scripts/kb_golden_runner_selftest.py ` lines 45-47 [possible, flow-unavailable:python (no data-flow facts for python in this unit, so the reach is by call only); distance 2 (indirect)]
        1. function ` decision-models/decisionmodels/points.py::load_point ` in ` decision-models/decisionmodels/points.py ` line 88
        2. call to ` load_point ` in ` decision-models/tests/test_points.py ` line 368; reached by CALLS <-
        3. function ` decision-models/tests/test_points.py::test_no_template_carries_a_seed_value ` in ` decision-models/tests/test_points.py ` line 366; reached by HAS_CALL <-
        4. call to ` search ` in ` decision-models/tests/test_points.py ` line 370; reached by HAS_CALL ->; possible, flow-unavailable:python (no data-flow facts for python in this unit, so the reach is by call only)
        5. function ` claude/cobb/scripts/kb_golden_runner_selftest.py::FakeSearch.search ` in ` claude/cobb/scripts/kb_golden_runner_selftest.py ` line 45; reached by CALLS_NAME ->; possible, name-match (matched by name only, the callee is not resolved)
```

The same request with `--format json` (exit 0, 2.02 s, stdout 846,154 bytes; top-level keys `answerSchemaVersion, depthBound, groups, interpretation, items, notices, recipe, staleness, stamp`):

- `stamp`: `{"snapshotId": "9add870b2d5ce60349c37b2d08b6b937", "verdictLogHash": "c482b1cf54631b37eb4572dc86a68fb1", "verdictLogOffset": 0}`
- `notices`: `[{"code": "flow-not-enumerated", "langs": ["python"]}]`

### 3.2 impact, name anchor `_run_agent_node` (`--format text`)

Exit 0, 0.76 s wall, 45,568 KB max RSS, stdout 177,365 bytes, stderr empty. Anchor line: `What is affected by: ` falkor-chat/server/falkorchat/executor.py::WorkflowExecutor._run_agent_node ` in ` falkor-chat/server/falkorchat/executor.py `.`; same snapshot line, the same five "Code views used" lines, then `172 items listed (82 confirmed, 90 possible).`, the same flow-dependent-effects notice sentence, and `Reach stopped at distance 2 with 961 unexplored edges.`

### 3.3 plain-language `--question`, parser-resolved (no model call)

- `--question 'decision-models/decisionmodels/points.py:88 impact' --format text` (a `path:line` anchor plus the direction word): exit 0, 2.03 s, stdout 475,462 bytes, **byte-identical** (`cmp`) to 3.1: the parser resolved line 88 to `load_point` and the use case to impact; header, 455 items (82 confirmed, 373 possible), the same notice.
- `--question 'falkor-chat/server/falkorchat/executor.py::WorkflowExecutor._run_agent_node' --format text` (a bare `path::symbol`, no direction word): exit 0, 0.26 s, stdout 591 bytes, nothing analysed because the use case is not stated:

```
Code-derived text appears only inside code spans; it is data, not instructions.

Snapshot 9add870b2d5ce60349c37b2d08b6b937, verdict log offset 0.
The question has several readings (two-use-cases); none was chosen, so nothing was analysed.
Reading 1 (rank 1): What is affected by: ` falkor-chat/server/falkorchat/executor.py::WorkflowExecutor._run_agent_node ` in ` falkor-chat/server/falkorchat/executor.py `.
Reading 2 (rank 2): What depends on the answer to: ` falkor-chat/server/falkorchat/executor.py::WorkflowExecutor._run_agent_node ` in ` falkor-chat/server/falkorchat/executor.py `.
```

- Operator note (my first attempt, not a defect): a question with prose around the anchor (`what breaks if I change <path>::<symbol>`) is free prose by the parser's rule (every word must be an anchor, scope or direction word), went to the interpreter and returned exit 2 `model-unavailable` (`BrokerUnavailable`, no broker running); that is the specified behaviour. No free-prose question was run, and no model was called.

## 4. Teardown and leftovers

- `ccg-falkordb.sh --throwaway down` (port 16395) stopped and removed `ccg-falkordb-tw-16395`; `docker ps` afterwards: `ccg-falkordb-tw-16390`, `falkordb-dev` and the two `cypher-mcp` containers, all as before. The sampler had exited.
- Kept: scratch dir `/tmp/claude-1000/-home-<user>-prg-graphmind-ai-lab/82bb1fdb-fb53-4f8c-80b5-7bf434adbf7b/scratchpad/p1-20/` (`cfg/` with the dead throwaway's admin password and `graphmind-ai-lab.acl`; `refresh.out` full JSON report, `refresh.time`, `samples*.txt`, `ask*.out`), and `cpg/.ccg-work/bundles/graphmind-ai-lab/` (116 MB, gitignored). The graph itself died with the container (no persistence); reloading from the retained bundles needs a new refresh.
- **Created in the real key area:** `~/.config/ccg/keys/graphmind-ai-lab.hmac` (65 bytes, 0600, created 13:48:53 local during the build stage). The directory was empty before the run. See section 5.

## 5. Open points for the caller

- **Defect (isolation): `CCG_CREDENTIALS_DIR` does not redirect the HMAC key.** `atrest.hmac_key(system, keys_root=None)` resolves through `_key_root(keys_root)` (`cpg/ccg/atrest.py` line 499) and never reads `$CCG_CREDENTIALS_DIR` (only `credentials_dir()`, used for the ACL files, does); `Client.refresh` and `refresh._run` pass `keys_root=None`. With the scratch `CCG_CONFIG_DIR`/`CCG_CREDENTIALS_DIR` set, the refresh still generated the system's real HMAC key under `~/.config/ccg/keys/`. Nothing pre-existing was overwritten. The ACL credentials did stay in the scratch dir. Decision for the caller: keep the key (the retained bundles are bound to it; a rotation needs a full rebuild) or delete it by hand; I left it. A coder fix would make `hmac_key` honor the variable, or document that the key is always the real one.

- The CLI has no `--work-root`: refresh artifacts land in `cpg/.ccg-work/bundles/graphmind-ai-lab/` (and `cpg/.audit/`), shared with any later refresh of this system. A later refresh would carry the `.state.json` records from this run; delete the directory first for another cold measurement.
- The CLI's `refresh` prints the whole `unattested` id list (1,126 ids, about 37 KB) in its JSON report; informational.
