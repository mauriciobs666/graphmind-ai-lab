# Connected code graph P1 close-out CO-1: dev probe and scope-ruling evidence

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1 close-out)

CPG: considered, not relevant — black-box measurement of the connected-code-graph engine against dev gold on a freshly built store; no `cpg_*` graph answers any of it.

Unit CO-1 of `docs/plans/connected-code-graph-p1-closeout.md` (§2, §3.1-§3.4). Counts and classifications only. Scripts, drivers and raw results: `cpg/eval/results/co1/` (`build.py`, `asks.py`, `callsites.py`, `classify.py`, `wiring_probe.py`, `gen_tables.py`; `co1-asks.json`, `callsites.json`, `classification.json`, `wiring-probe.json`, `build.json`, `table-*.md`). Dev gold only (`cpg/eval/dev/ac5-*.jsonl`); no held-out file or custodian directory was opened.

## 1. Headline

- **§3.1 premise, split verdict.** A shell `template` valueSet **does** keep the literal path tail and marks the dynamic head (`{"unknown": "point-derived"|"runtime-built"}`), and the token screen did not withhold any of them, **for an argument** (`. "$(dirname "$0")/shlib/x.sh"`, `exec "$(dirname ...)/../../scripts/x.sh"`): 11 shell items. **It fails for a computed command word** (`"$REPO_DIR/scripts/x.sh" args`, `"$HERE/x.sh" args`, `"${core_dir}/x.sh"`): the call site has `calleeName: "<dynamic>"` and its valueSets `a0..` start at the first *argument*; the command word's value is in **no record**. That is **10 of the 49 scripts items**, the largest single group. It changes CO-2: the extractor must emit the command word's value (record shape to be checked against bundle-spec3 §5.1 `argPath`, which has a `recv` segment; not tried here), or those 10 stay misses. Evidence: raw call-site and valueSet records of the real build for every impacted element, `callsites.json`; extractor source `cpg/extractors/shell/extract.py` `_callee` and `_arguments`.
- **Scripts 0/49** (every anchor's answer lists 0 items; the anchor `<script>`/`<module>` symbol exists in the store; 0 `CALLS` edges point into any of the 20 symbol anchors). **http 17/18.** **wiring 0/7 asked** (4 unit-anchored items not asked).
- **DEF-1 (§3.3):** the one http miss is the inline `curl` span in `claude/teco/coordination-techniques.md:25`: its call site exists (`curl`, `possible`/`prompt-mention`) with **no valueSet at all**. The three shell `curl` items carry the URL in a valueSet (`http://127.0.0.1:` + unknown + `/health`, or unknown + `/health`).
- **Wiring (§3.4):** 0 of 11 dev items expressible now: every config element has `ConfigEntry` rows in the store (1 to 48 under its keyPath) and **no `Link` is bound to any**; the three contract-anchored answers (22 items) list no `config` item.

## 2. System under test

| Item | Observed |
|---|---|
| Clone | `git clone --no-hardlinks` of the working repo; HEAD **be5d8b2fb44e46ef2015519f7e101e0b1fbb2dea** (the dispatch cited 94382dc8; one docs-only ledger commit, be5d8b2f, landed before the clone; `cpg/` is identical between the two) |
| Store | `cpg/infra/ccg-falkordb.sh up --throwaway`, container `ccg-falkordb-tw-16510`, port 16510, private `CCG_CONFIG_DIR`; `GRAPH.CONFIG` reported `QUERY_MEM_CAPACITY 268435456` |
| Build | one full refresh, `cpg/eval/results/co1/build.py` (copy of p1t-20): **1347 units, 1347 fresh, 0 not fresh**, `real 31m50s`; snapshot id `804ea74b44b820f4adb59d2fc486f7b0` |
| Private areas | ACL users in a private `CCG_CREDENTIALS_DIR`; work, keys, audit dirs passed explicitly; none under the developer defaults |
| Teardown | throwaway container stopped and removed (`ccg-falkordb-tw-16510 stopped and removed`); `ccg-falkordb`, `ccg-falkordb-tw-16390`, `falkordb-dev` and the two other containers untouched and still up |
| Sentinels | `cpg/.ccg-work` and `cpg/.audit`: no file newer than the run start. `~/.config/ccg`: every file, size and mtime unchanged, but **the directory's own mtime moved at 14:15:14** (an entry created or removed transiently). I did not use that directory (`CCG_CONFIG_DIR` was private) and cannot attribute the change; a concurrent session on this host is the likely cause. Disclosed, not resolved. |
| Noise printing (S1) | No tool I ran printed noise, a reduction, a burden sum or a baseline [k, n]; I computed and wrote no noise figure at all. Nothing to disclose. |
| Method | `asks.py` is `p1t-20/asks.py` unchanged in `request_of`/`ask_anchor`/`matches`/`judge` plus a per-anchor answer view; one structured `impact` ask per anchor through the real client; hit = the EP H1 identity of the dev item among the answer's items, status confirmed/possible, chain with file and line on every located hop. Production code not changed. |

## 3. One denominator per stratum

| Stratum | Items | Anchors | Anchor kinds (items) | Asked | Hit now |
|---|---|---|---|---|---|
| scripts | **49** | **22** (20 symbol, 2 contract) | symbol 45; contract 4 (3 http `GET /health`, 1 mcp `send_message`) | 49 | **0** |
| http | **18** | **12** | contract 18 | 18 | **17** |
| wiring | **11** | **6** (1 unit, 4 symbol, 1 contract) | unit 4, symbol 4, contract 3 | **7** (symbol 4 + contract 3; 5 anchors) | **0** |

Reconciliation, arithmetic only: 49 minus the 3 http-contract items is 46 (the RCA's figure; I cannot see how the RCA derived it); 49 minus all 4 contract-anchored is the plan's 45 symbol-anchored; wiring's 7/11 is the 4 unit-anchored items, which need the custodian-side unit probe and were not asked here. Scripts impacted elements by language: Markdown 17 items (10 distinct documents), shell 31 items (30 `<script>`, 1 function `run_block` that is absent from the store), Python 1.

## 4. Scripts: classification of every dev item (49)

Families (item count; the table in §4.1 carries each item). "Record carries" = what the real extractor wrote for the impacted element.

| Family | Items | Record carries | Mechanism a fix needs |
|---|---|---|---|
| A. Markdown inline token, >= 2 segments, unique snapshot suffix | 13 | call site `possible`/`prompt-mention`, `calleeName` = the token as written (`./scripts/x.sh`, `claude/scripts/x.sh`) | edge only (assembler); the call site exists |
| B. Markdown bare basename only (`export-cpg.sh`, `run.sh`) | 2 | call site with a 1-segment token (1 and 5 snapshot matches); the gold mention is in a fenced block | not reachable by the 2-segment rule; `run.sh` is also ambiguous |
| C. Markdown front-matter hook `command: $HOME/.claude/agents/<a>/hooks/x.sh` | 2 | nothing (no call site for a front-matter `command`) | new front-matter extraction; an installed path, suffix `hooks/x.sh` would be unique |
| D. shell `. "$(dirname "$0")/shlib/x.sh"` | 4 | valueSet `a0` template [`point-derived`, `/shlib/opencode_config.sh`], 2 segments, 1 match | tier 2 as designed |
| E. shell `exec "$(dirname ..)/../../scripts/x.sh" agent` | 5 | valueSet `a0` template [`point-derived`, `/../../scripts/guard-*.sh`]; after `..` normalisation 2 segments, 1 match | tier 2 as designed |
| F. shell literal command word `./scripts/bootstrap_schema.sh` | 1 | call site `calleeName` = the literal | tier 1 (the gold line, 103, holds `echo ""`; the call is on line 114) |
| G. shell **computed command word** (`$REPO_DIR/scripts/x.sh` 7, `$HERE/export-cpg.sh` 1, `${core_dir}/guard-destructive-ops.sh` 1, `$ROOT/run.sh` 1) | **10** | `calleeName` `<dynamic>`, `possible`/`shell-dynamic`; **no record of the path** | extractor change first. Then: 7 have a 2-segment unique tail (tier 2); 3 have a 1-segment tail (`export-cpg.sh` 1 match, `guard-destructive-ops.sh` 4, `run.sh` 5) |
| H. shell `$HERE/x` as an argument, 1-segment tail (`. "$HERE/git-provenance.sh"`, `exec "$PY" "$HERE/server.py"`) | 2 | valueSet template [`runtime-built`, `/git-provenance.sh` or `/server.py`]; 1 and 5 snapshot matches | the 2-segment rule rejects it; needs own-directory resolution, but the head's kind is not recorded as "caller's own directory" |
| I. `test-stamp-wiring.sh` | 3 | parse-error warning, `<script>` symbol only, **no call sites** (`parse-error` at line 351; `run_block` symbol absent) | CO-2a parse fix; and the path sits inside a `bash -c '...'` string (items 0, 34) or is an `awk` file argument, not a launch (item 47) |
| J. `-m` package (`exec "$PY" -m modelbench`, `-m mcp_monitor`) | 2 | valueSet `-m`, `modelbench`; no path | P2 item (`-m` package resolution) |
| K. Python `subprocess.run(["bash", str(_BOOTSTRAP), ...])`, `_BOOTSTRAP = _REPO_ROOT / "scripts" / "bootstrap_schema.sh"` | 1 | call site `run` with kw args; `str(_BOOTSTRAP)` is a `ref` | Python extractor reading a subprocess argument and a Path expression |
| L. shell `curl` URL (contract `GET /health`) | 3 | valueSet on the `curl` site: `http://127.0.0.1:`+unknown+`/health`, or unknown+`/health` | DEF-1 shell pack |
| M. embedded Python in a shell file calling `session.call_tool('send_message', ...)` (contract `send_message`) | 1 | no call site near the gold lines | withheld (Python inside a shell string) |
| **Total** | **49** | | |

Buildable under the plan's current rules from records that exist today: A 13 + D 4 + E 5 + F 1 = **23** (plus L 3 with the pack). With a command-word record: +7 (G, 2-segment tails) = 30. Not reachable under the 2-segment rule or needing new extraction: B 2, C 2, G 3, H 2, I 3, J 2, K 1, M 1 = **16**. Every tier-2 site I checked has exactly one snapshot match and that match is the anchor (no checked site points elsewhere); the 1-segment tails are the ambiguous ones.

### 4.1 Per-item table

| # | item | anchor | impacted element | family | record carries | now | why it misses (the answer for the anchor) |
|---|---|---|---|---|---|---|---|
| 0 | 0fbf43 | git-provenance.sh::<script> | test-stamp-wiring.sh::run_block | I parse defect / path inside `bash -c` string / file-read | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 1 | 106e87 | guard-tdd-broad-bash.sh::<script> | tdd-engineer.md::<document> | C md front-matter hook | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 2 | 107ea0 | test_queries.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/test_queries.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 3 | 177448 | verify_salesperson.sh::<script> | start_demo.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 4 | 1ebbf6 | guard-broad-write.sh::<script> | guard-tdd-broad-write.sh::<script> | E shell exec arg `../../scripts/x` | valueSet `/../../scripts/guard-broad-write.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 5 | 28b763 | guard-destructive-ops.sh::<script> | guard-destructive-ops.sh::<script> | E shell exec arg `../../scripts/x` | valueSet `/../../scripts/guard-destructive-ops.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 6 | 3c0917 | health | load_test.sh::<script> | L shell curl URL (http contract) | valueSet `/health` (1 seg, 0 snapshot match) | miss | answer lists 10 items, none is this element |
| 7 | 46cba1 | opencode_config.sh::<script> | start_agent_team.sh::<script> | D shell `.` arg `$(dirname)/shlib/x` | valueSet `/shlib/opencode_config.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 8 | 4886ea | create_workspace.sh::<script> | start_server.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 9 | 4b9e00 | opencode_config.sh::<script> | ensure_demo_models.sh::<script> | D shell `.` arg `$(dirname)/shlib/x` | valueSet `/shlib/opencode_config.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 10 | 52fb50 | guard-destructive-ops.sh::<script> | guard-destructive-ops.sh::<script> | E shell exec arg `../../scripts/x` | valueSet `/../../scripts/guard-destructive-ops.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 11 | 5b355b | opencode_config.sh::<script> | start_demo.sh::<script> | D shell `.` arg `$(dirname)/shlib/x` | valueSet `/shlib/opencode_config.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 12 | 5dbd4b | health | seed_nlq_eval_corpus.sh::<script> | L shell curl URL (http contract) | valueSet `/health` (1 seg, 0 snapshot match) | miss | answer lists 10 items, none is this element |
| 13 | 6fee71 | export-cpg.sh::<script> | pipeline.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 14 | 7d9d8a | opencode_config.sh::<script> | start_server.sh::<script> | D shell `.` arg `$(dirname)/shlib/x` | valueSet `/shlib/opencode_config.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 15 | 7fc6a2 | bootstrap_schema.sh::<script> | test_queries.sh::<script> | F shell literal command word `./scripts/x.sh` | callee `./scripts/bootstrap_schema.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 16 | 8f9035 | load_test.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/load_test.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 17 | 914902 | audit-team.sh::<script> | cobb.md::<document> | A md inline token, >=2 segments, unique | callee `claude/scripts/audit-team.sh` (3 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 18 | 9180d6 | __main__.py::<module> | run.sh::<script> | J `-m` package | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 19 | 9302f3 | create_workspace.sh::<script> | start_agent_team.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 20 | 94471e | bootstrap_schema.sh::<script> | create_workspace.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 21 | 9b9925 | audit-team.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `scripts/audit-team.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 22 | a4672e | guard-broad-write.sh::<script> | guard-coder-broad-write.sh::<script> | E shell exec arg `../../scripts/x` | valueSet `/../../scripts/guard-broad-write.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 23 | a7b52a | bootstrap_schema.sh::<script> | seed_eval_corpus.py::main | K Python subprocess argument | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 24 | aa752d | send_message | demo_falkor_chat.sh::<script> | M embedded Python MCP call in shell (mcp contract) | none | miss | answer lists 22 items, none is this element |
| 25 | ace49b | test_queries.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/test_queries.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 26 | aed9b8 | guard-destructive-ops.sh::<script> | guard-destructive-ops.sh::<script> | E shell exec arg `../../scripts/x` | valueSet `/../../scripts/guard-destructive-ops.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 27 | af3d4a | verify_salesperson.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/verify_salesperson.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 28 | b0f627 | guard-tico-repo-writes.sh::<script> | tico.md::<document> | C md front-matter hook | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 29 | b3547e | pipeline.sh::<script> | SKILL.md::<document> | A md inline token, >=2 segments, unique | callee `scripts/pipeline.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 30 | b35838 | verify_catalog.sh::<script> | start_demo.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 31 | c12efd | bootstrap_schema.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/bootstrap_schema.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 32 | c386af | git-provenance.sh::<script> | pipeline.sh::<script> | H shell `$HERE/x` argument, 1-segment tail | valueSet `/git-provenance.sh` (1 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 33 | c649a7 | export-cpg.sh::<script> | SKILL.md::<document> | B md bare basename only | callee `export-cpg.sh` (1 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 34 | d105f0 | git-provenance.sh::<script> | test-stamp-wiring.sh::<script> | I parse defect / path inside `bash -c` string / file-read | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 35 | d65e21 | __main__.py::<module> | run.sh::<script> | J `-m` package | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 36 | d74079 | guard-destructive-ops.sh::<script> | guard-broad-bash.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 37 | d7b414 | run.sh::<script> | AGENTS.md::<document> | B md bare basename only | callee `run.sh` (1 seg, 5 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 38 | df42c4 | health | ensure-environment.sh::<script> | L shell curl URL (http contract) | valueSet `/health` (1 seg, 0 snapshot match) | miss | answer lists 10 items, none is this element |
| 39 | df9bdc | verify_catalog.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/verify_catalog.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 40 | e2d5f0 | create_workspace.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/create_workspace.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 41 | e6c6a3 | test_queries.sh::<script> | qa-engineer.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/test_queries.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 42 | e6e80b | audit-team.sh::<script> | SKILL.md::<document> | A md inline token, >=2 segments, unique | callee `claude/scripts/audit-team.sh` (3 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 43 | e8c36e | server.py::<module> | run.sh::<script> | H shell `$HERE/x` argument, 1-segment tail | valueSet `/server.py` (1 seg, 5 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 44 | f15550 | verify_workflows.sh::<script> | AGENTS.md::<document> | A md inline token, >=2 segments, unique | callee `./scripts/verify_workflows.sh` (2 seg, 1 snapshot match) | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 45 | f25577 | bootstrap_schema.sh::<script> | load_test.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 46 | f33cbe | run.sh::<script> | demo_falkor_chat.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 47 | f7a509 | pipeline.sh::<script> | test-stamp-wiring.sh::<script> | I parse defect / path inside `bash -c` string / file-read | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |
| 48 | fa829e | create_workspace.sh::<script> | start_demo.sh::<script> | G shell computed command word (path unrecorded) | none | miss | answer lists 0 items; anchor symbol present; 0 CALLS edges in |

All 49 miss for the same observable reason: the anchor's answer lists no item for the impacted element (0 items for the 20 symbol anchors; 10 and 22 items, none the dev element, for the two contract anchors).

## 5. §3.3 call-site check

- **Markdown (17 items, 10 documents): 15 carry a call site whose `calleeName` names the anchor** (path or basename as written); 2 carry none (family C). Only 7 of the 15 have that site within one line of the gold line; for the other 8 the site is elsewhere in the same document. Token forms: 13 are a >= 2-segment unique path, 2 a bare basename.
- **Shell (31 items): 1 carries a call site naming the anchor** (item 15, literal command word). 11 more carry the anchor's path tail in an **argument** valueSet (D 4, E 5, H 2). 10 carry a `<dynamic>` call site with the path unrecorded (G). 3 + 1 are contract-anchored (L, M), 3 are the parse defect (I), 2 are `-m` (J).
- Note, not a classification: the gold line holds no mention of the anchor within one line for items 2, 15, 16, 27, 39, 41, 42 (the mention is 2 to 11 lines away in the same file at this HEAD). I do not claim a key-error; the architect or custodian may want to look.

## 6. http (18): DEF-1 and the rest

17 hit (6 `confirmed`, 11 `possible`), all at distance 1, through the existing packs (JS `api("/path")` wrappers and template URLs, TS `apiFetch` with `API_PREFIX`, Python client calls with literal and f-string URLs). One miss:

| # | item | anchor | impacted element | mechanism | now | status/distance or reason |
|---|---|---|---|---|---|---|
| 0 | 021708 | GET /health | coordination-techniques.md::<document> | curl URL in inline Markdown span (carrier) | miss | answer lists 10 items; the Markdown `<document>` is not among them (its `curl` call site has no valueSet) |
| 1 | 0414ef | GET /workflow-defs | app.js::loadDefsList | already built (hit) | hit | possible, distance 1 |
| 2 | 0a94dc | GET /health | load_append.py::main | already built (hit) | hit | confirmed, distance 1 |
| 3 | 1f4925 | GET /workspaces/{}/snapshots/{}/versions/{} | app.js::renderWaitingForm | already built (hit) | hit | possible, distance 1 |
| 4 | 40c8c6 | POST /shop/api/session | endpoints.ts::joinSession | already built (hit) | hit | confirmed, distance 1 |
| 5 | 47f126 | POST /threads/{}/messages | app.js::postMessage | already built (hit) | hit | possible, distance 1 |
| 6 | 57f718 | GET /health | seed_nlq_eval_corpus.py::_check_server_reachable | already built (hit) | hit | confirmed, distance 1 |
| 7 | 7546b0 | GET /channels | app.js::loadChannels | already built (hit) | hit | possible, distance 1 |
| 8 | 7f3682 | POST /threads/{}/messages | load_append.py::_worker | already built (hit) | hit | possible, distance 1 |
| 9 | 8ffa9f | POST /workflow-runs/{}/input | app.js::submitRunInput | already built (hit) | hit | possible, distance 1 |
| 10 | 9703a1 | GET /documents/{} | seed_nlq_eval_corpus.py::_poll_until_terminal | already built (hit) | hit | possible, distance 1 |
| 11 | c05679 | POST /shop/api/session | language_salience_probe.py::run_participant | already built (hit) | hit | possible, distance 1 |
| 12 | c58283 | GET /channels/{}/threads | app.js::loadThreads | already built (hit) | hit | possible, distance 1 |
| 13 | e25af0 | GET /shop/api/health | endpoints.ts::getHealth | already built (hit) | hit | confirmed, distance 1 |
| 14 | f2d84c | GET /shop/api/messages | endpoints.ts::getMessages | already built (hit) | hit | confirmed, distance 1 |
| 15 | f3f1d5 | GET /shop/api/messages | language_salience_probe.py::_assistant_rows_since | already built (hit) | hit | possible, distance 1 |
| 16 | f65a45 | POST /documents | seed_nlq_eval_corpus.py::main | already built (hit) | hit | confirmed, distance 1 |
| 17 | f9041b | POST /shop/api/session | load_demo.py::run_participant | already built (hit) | hit | possible, distance 1 |

**Carrier finding:** the Markdown `curl` call site carries no valueSet, so the carrier change is to write the first URL token of the inline span (`http://127.0.0.1:8200/health`) as a valueSet on that call site. The span is inline (no fenced carrier needed for this dev item). The shell side is already present (family L).

## 7. Wiring probe (§3.4)

Store totals on this build: 347,733 `ConfigEntry`, 934 `Link`, 124 `BOUND_BY`, 82 `Link`s bound to a `ConfigEntry`. For the 11 dev items: `ConfigEntry` rows exist under every keyPath (1 to 48) and **0 `Link`s are bound to any of them**.

| # | item | anchor type | config element | mechanism | askable | now | store (ConfigEntry under keyPath / Link BOUND_BY it) | answer |
|---|---|---|---|---|---|---|---|---|
| 0 | 129dc9 | unit | compose.yaml#services.server | server URL (compose service; unit anchor) | no | not asked | 11 / 0 | not asked |
| 1 | 284562 | unit | falkor-chat-demo.json#mcpServers.falkor-chat | server URL (kiro mcpServers; unit anchor) | no | not asked | 1 / 0 | not asked |
| 2 | 2ae2af | unit | config.example.toml#server.falkor-chat | server URL (mcp-monitor server block; unit anchor) | no | not asked | 2 / 0 | not asked |
| 3 | 32da04 | contract | falkor-chat-demo.json#allowedTools[0] | tool name in an allow-list (kiro allowedTools) | yes | miss | 1 / 0 | 22 items, 0 of type config |
| 4 | 3efb7f | contract | falkor-chat-demo.json#tools[0] | tool name in a tools list (kiro tools) | yes | miss | 1 / 0 | 22 items, 0 of type config |
| 5 | 81fd8c | unit | .mcp.json#mcpServers.falkor-chat-agent-team | server URL (.mcp.json; unit anchor) | no | not asked | 2 / 0 | not asked |
| 6 | 9c7140 | symbol | opencode.json#agent.tank | launches a script (tank bash allow-list path) | yes | miss | 48 / 0 | 0 items |
| 7 | ae355d | symbol | config.example.toml#server.fake-test | launches a script (command in TOML server block) | yes | miss | 4 / 0 | 0 items |
| 8 | da060c | symbol | opencode.json#agent.tank | launches a script (tank prompt + allow-list) | yes | miss | 48 / 0 | 0 items |
| 9 | f16850 | symbol | opencode.json#agent.tank | launches a script (tank prompt + allow-list) | yes | miss | 48 / 0 | 0 items |
| 10 | f8844c | contract | falkor-chat-demo.json#prompt | tool name in a prompt (kiro `prompt` field) | yes | miss | 1 / 0 | 22 items, 0 of type config |

Classes: server URL 4 (all unit-anchored, not askable here), tool name 3 (contract-anchored), launches-a-script **4** (not the plan's 3: the `mcp-monitor` `server.fake-test` command also runs a script). Expressible now: **none** of 11. The three contract-anchored items fail because no `Link` reaches their `ConfigEntry` and the answer-spec admits a `config` item only through a bound `Link`; the four symbol-anchored items fail because the anchor symbol has no inbound edge from the allow-list/prompt/command entry.

## 8. Defects and findings for the ruling

1. **Computed command word is unrecorded** (10 items, §1): CO-2 as designed cannot reach them. Needs an extractor change.
2. **1-segment tails** (B 2, G 3, H 2 = 7 items): the 2-segment rule rejects them by design; own-directory resolution would need the extractor to say the head is the caller's directory, and the head kinds recorded (`point-derived`, `runtime-built`) do not say so.
3. **CO-2a is wider than the quoting fix:** two of the three `test-stamp-wiring.sh` items name the path inside a `bash -c '...'` string; fixing the parse yields a `bash` call site with a long string argument, not a path argument. The third is a file read, not a launch.
4. **Gold line drift** on 7 items (§5), possible, unverified.
5. **Scope-changing decision, returned to the architect:** under the plan's rules 16 of 49 scripts items are not built or not reachable (§4); Q8 applies to them.

## 9. Coverage and limits

Covered: every dev item of scripts, http, wiring; real extractor output for all 49 scripts impacted elements; one build. Not covered: mcp and flow strata (not asked), the 4 unit-anchored wiring items (custodian-side probe), held-out anything, noise. `callsites.py` reads the build's bundles at `$CO1_BUNDLES` (scratch copy, not kept); the committed `callsites.json` and `classification.json` are its output. The family labels are my reading of each source line; counts, record contents and answers are observed.
