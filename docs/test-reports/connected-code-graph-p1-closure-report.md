# Connected code graph P1: AC-closure table (P1-38, first deliverable)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1-38)

CPG: considered, not relevant — the subject is the AC-closure bookkeeping of the `ccg` P1 phase (documents, reports, reviews and test runs of the `ccg` pipeline); no `cpg_<component>` Joern graph answers it.

Plan: `docs/plans/connected-code-graph-p1-gates.md` section "P1-38 · phase exit"; `docs/plans/connected-code-graph-p1.md` §8 "AC map at phase exit" and the P1-37/P1-38 rows; AC wording `docs/requirements/connected-code-graph.md`; test plan `docs/test-plans/connected-code-graph-p1.md` §5. Counts and verdicts only: no held-out content is quoted; neither the custodian root `~/.local/share/ccg-eval/cf7b2042cce7` nor any held-out file was read for this report (held-out counts are cited from `docs/test-reports/connected-code-graph-p1-heldout-strata-counts-report.md`).

## 1. How to read this report

- **Observed** = a command run for this report (section 3, exit status, cwd `cpg/`, HEAD `dacc6958`). **Cited** = read from the named file and not re-derived; the file owns the figure. This includes the live runs (closing items, AC-8 harness, fixture and AC-level drivers) and the coordinator's full-suite run: they were executed by other agents and I re-ran nothing live. The live AC-level run (P1-42: AC-9, 9b, 16, 17, 18, 21, 25) was made at `a81a597e` plus the then-uncommitted driver; after it `cc53b3de` (verbatim-key retry in `resolve_contract`), `4cda0ed5` and `2bf875bd` (interpreter prompt) changed, and its per-row ledger was never committed (only `cpg/eval/results/p1t-20/p1t20.json` is). The P1T-20 run at `57be01a2` post-dates `cc53b3de`; the AC-level driver was not repeated on the final engine.
- Status vocabulary: **closed**; **closed for a stated sub-scope**; **reported not closed, because ...**. Stakeholder decisions are recorded in the words the coordinator gave them.
- This report decides nothing; decisions already taken are recorded in section 6, what stays open is stated there.

## 2. Verdict

**Of the 16 AC rows: 4 closed (AC-10 flagged "behaviour unobserved", AC-11, AC-18 with its reading recorded, AC-21); 8 closed for a stated sub-scope (AC-1 17/18, AC-2 MCP half, AC-6, AC-9, AC-9b, AC-16 with a sufficiency-only raw leg, AC-17 two of four groups, AC-25 data-flow half); 4 reported not closed (AC-4, AC-5 in every stratum, AC-8 part 1, AC-24 local model).** AC-5's held-out gate is deferred to P2 (user, 2026-11-23). No new red: R1 to R8 observed green, and the coordinator's full default suite (R9, cited) has the nine known reds, two of which are P1's own and are being fixed (section 3). Details per AC in section 4.

## 3. Runs observed for this report

All from `cpg/` with `.venv/bin/python -m pytest -q -p no:cacheprovider`, default markers (live tier deselected), HEAD `dacc6958` (the work tree differs from HEAD only in three untracked `docs/` files of another feature). Output logs: session scratchpad `cl/*.log` (not committed).
| Run | Files | Exit | Last line |
|---|---|---|---|
| R1 AC driver (offline tier) | `tests/test_fixture_gate_p1_ac.py tests/test_fixture_gate_p1.py` | 0 | `11 passed, 120 deselected` (the 120 are the live tier) |
| R2 AC-16 pin-and-replay (offline) | `tests/test_pin_replay.py` | 0 | `12 passed, 4 deselected` |
| R3 AC-8 harness unit tier | `tests/test_ac8_assertions.py test_ac8_detector.py test_ac8_lib.py test_ac8_window.py` | 0 | `360 passed` (the `test_ac8_lib` flake did not occur this run) |
| R4 AC-24 smoke harness | `tests/test_ac24_smoke.py` | 0 | `177 passed` (stubs and synthetic items only: no model, no held-out) |
| R5 renderers and clients (AC-11, AC-16) | `tests/test_render.py test_clients.py test_front_clients.py` | 0 | `290 passed, 1 warning` |
| R6 ACL and at-rest | `tests/test_acl_unit.py tests/test_atrest.py` | 0 | `273 passed` |
| R7 fixtures and manifests | `tests/test_fixtures.py` | 0 | `80 passed, 3 deselected` |
| R8 AC-5 gate harness | `tests/test_gate.py test_gate_score.py test_gate_run.py test_gate_fail.py` | 0 | `309 passed` |
| R9 full default suite (**cited**, run by the coordinator on the committed tree, not by me) | `cpg/`, `pytest tests`, live tiers deselected | 1 | 9 failed, 17330 passed, 10 skipped, 531 deselected, 1 xfailed, 2026-11-24 22:01 (ledger 2026-11-24) |

My own earlier attempt at R9 was stopped at about 42% and is superseded by the coordinator's complete run. The nine reds are, per the ledger entry (names cited; attribution below is the reviewer's, `docs/reviews/connected-code-graph-p1-impl.md` M1, by execution at `231d5b35`, cited not re-derived by me): **P1's own, being fixed now** (coordinator, Q4): `test_p1_11b_the_repos_own_mcp_json_names_its_script` (P1-27, `e200a8a2`, added the `ccg` `bash -c` entry to `.mcp.json`; the P1-11b test asserts no `-c` arg) and `test_every_data_file_under_ccg_is_covered_by_a_package_data_glob` (P1-26, `8377c326`, added `front/prompts/*.txt` and `front/grammar/interpret.json` with no `pyproject.toml` glob; a non-editable install would ship without the interpreter prompt and grammar). **Not P1's:** `test_units` golden (the monorepo grew). **Reproduced, cause untraced, owner unverified:** the leaf-harness survivor-tell. **Did not reproduce in isolation** (environment- or order-dependent): `test_scip_parse` x4. **Known flake:** `test_ac8_lib`. So: no new red, but two of the nine are P1's, not "known and not P1's". A concurrent devops build touched `ac8` files during that run (ledger). `test_queries.sh` (falkor-chat, shared FalkorDB) was not run: out of this component.

## 4. AC-closure table

Closing steps are the plan's step ids (`docs/plans/connected-code-graph-p1.md` §3 table, `-p1-gates.md`). Evidence paths are repo paths; "cited" figures are not re-derived here.

### AC-1 (HTTP): **closed for a stated sub-scope: 17 of 18 dev items listed; the one miss (DEF-1) is carried to P2**

- Closing step: P1-42, item P1T-20. Cited, executed (`docs/test-reports/connected-code-graph-p1-closing-items-report.md` §3, qa `a35970ea267d82d24`, committed `d1542439`; HEAD `57be01a2`, own throwaway store, proving-system build 1335 of 1335 units `fresh`, 1772 s): 12 dev anchors, **17 of 18** items listed (`confirmed` 6, `possible` 11, all distance 1), 0 chain violations over the 17 hits; results `cpg/eval/results/p1t-20/p1t20.json`; two runs gave identical per-item outcomes. Dev anchors were used for development: not AC-5 evidence.
- The AC's literal scenario, a falkor-chat endpoint's `salesperson/` callers, is met: the 6 `salesperson/` dev items are all hits (reviewer-verified in the dev gold); the one miss is the `claude/teco` `curl` line. Under the plan's pass rule (every dev item, no recall floor) the row is a fail. Stakeholder decision (2026-11-24): AC-1 closed for sub-scope, DEF-1 carried to P2. DEF-1: the miss is a shell/Markdown `curl` URL source (`claude/teco/coordination-techniques.md` line 25, anchor `GET /health`), the P2-deferred class (ledger characterisation; the closing-items report itself made no root-cause claim, only that no http Link starts from that file).
- Fixture-level (cited): HTTP rows pass, 81 of 81 P1 rows, closed world 0 unplanted (`docs/test-reports/connected-code-graph-p1-report.md` "P1-41 dispatch C"; `cpg/tests/test_fixture_gate_p1.py`, offline tier observed R1).

### AC-2 (MCP + wiring): **closed for a stated sub-scope: MCP half closed (20 of 20); wiring half not closed (P2)**

- MCP half, cited (closing-items report §3; composition per review M3): 3 anchors (`get_document`, `delete_document`, `send_message`), **20 of 20** dev items listed; all 20 impacted files are Markdown prompt files, all `possible`, distance 1, none bound on a Link (`cpg/eval/results/p1t-20/p1t20.json`). The `kiro/` and `mcp-monitor/` configuration callers named in the AC are the wiring half (0), so "MCP half" means prompt mentions of three tools.
- Wiring half: 0 of 3 in the plan's filtered sample, 0 of 7 asked in the stratum (4 items of one unit anchor not asked: that needs the custodian-side probe); no `config` item is ever listed (cited; `docs/reviews/connected-code-graph-p1-gate-zero-hit-rca.md`: engine gap, `LAUNCHES` beyond `launchUnit` is P2). The `kiro/`, `mcp-monitor/` wiring in the AC text is not listed. Carried to P2 (stakeholder 2026-11-23).

### AC-4 (scripts): **reported not closed, because shell script-by-path invocation and prompt-to-script links are not built at P1 (P2)**

- Cited (closing-items report §3): 0 of 1 in the plan's filtered sample, **0 of 49** in the scripts stratum (22 of 22 anchors asked); 20 of 22 script anchors answer with zero items; no `config` item and no `Link.bound[]` in the 54 answer items (DEF-2, the known engine gap, reproduced on HEAD). The launch-chain mechanism works on fixtures (`BOUND_BY` 10 on the fixture build, `docs/test-reports/connected-code-graph-p1-report.md` §2).

### AC-5 (never miss), per stratum: **reported not closed at P1 — the held-out gate is deferred to P2 by stakeholder decision (2026-11-23)**

No held-out item has been exposed to the gate: no `ac5-gate-1` record exists in `cpg/eval/preregistration.jsonl` (grep count 0, observed), P1-37 was not run and P1-43 does not apply (ledger 2026-11-23). The AC text requires >= 75 counting items; the http+mcp pair has **72** counting items in 19 clusters (3 short), scripts and wiring are not linkable at P1 (cited: `docs/test-reports/connected-code-graph-p1-heldout-strata-counts-report.md`; manifest `cpg/eval/heldout-manifest.json`, `verify` exit 0, checked 77, mismatches 0, cited).

| Stratum | Held-out items (cited) | Dev items and dev hits, P1T-20 on the HEAD build (cited, optimistic, dev was used for development) | Status |
|---|---|---|---|
| http | 26 (15 clusters) | 18; **17/18** | not measured at P1 (gate deferred); dev only |
| mcp | 46 (4 clusters; lumpy) | 20; **20/20** | not measured at P1; dev only |
| scripts | 79 (36 clusters) | 49; **0/49** | engine gap (RCA); not delivered at P1, P2 |
| wiring | 8 (5 clusters; exactly the floor of 8, "screened" not "certified") | 11 in the file; **0/7** asked | engine gap (RCA); not delivered at P1, P2 |
| flow | 99 (12 clusters; 0 counting under flag false) | 28; not run (flag false) | **withheld**: `python.flowAvailable` false (`FALSE:measured`), see below |


- Flow stratum: reported not closed, flow-unavailable. P1-34 concluded `FALSE:measured`: decision cell 296/334 = 0.8862, Wilson 95% [0.8477, 0.9160], rule 20y >= 19G failed (M = -389), no registry flip (cited: `cpg/eval/decisions/ccg-p1-flow-python-v2-decision.json`; review `docs/reviews/connected-code-graph-p1-34-impl.md`). JavaScript was not measured (plan).
- Closing steps that did complete (harness, not the gate): P1-32a verbs (`docs/test-reports/connected-code-graph-p1-32a-report.md`), P1-32b `gate.py`/`gate_run.py` gated at P1 (analyst Pass 5, approve with suggestions; `docs/reviews/connected-code-graph-p1-32b-impl.md`), dress rehearsal RED then fixed by rounds E and F (`-gate-rehearsal-report.md`, ledger 2026-11-22/23). Harness tests observed green: R8, 309 passed. Not re-run after round E: the no-workaround dev rehearsal (the delegate's own rerun only), as the ledger says it is P2's when the list is spent.
- Disclosures a P1-37 report would have restated, carried here (all cited from the ledger): labels are AI labels; the EP §7.3 redo of 40 items was done by an AI reviewer on the `fable` model in about 4 minutes, 39 correct 1 incorrect, three `doubt:` items kept as marked (a stricter reading gives 4 of 40 and would fire the escalation trigger), item 26 key-error removal accepted as the AI's call (`docs/test-reports/connected-code-graph-p1-s36-redo-report.md`); the flow-draw custodian role was filled by the coordinator, not the human owner; same-family adjudicators agreed 120 of 120.
- Open P2 items before the gate may spend the list (cited, ledger): Pass 5 minors P5-1..P5-3, `STRATA_GATED`/`MIN_COUNTING` and the `ac5-gate-1` record stale, dev-gold pre-gate probe and per-kind breakdown, rerun of the no-workaround rehearsal.

### AC-6 (doubt flagged), P1 kinds: **closed for a stated sub-scope (URL, example config, prompt mention); dynamic import reported not met at P1**

- Closing steps: P1-41, P1-42 (+ P1-55, P1-56, P1-57). Evidence: `cpg/tests/test_fixture_gate_p1_ac.py`, `cpg/tests/test_fixture_gate_p1.py`; `docs/test-reports/connected-code-graph-p1-report.md` "P1-42" table (cited, live run on a throwaway): URL 3 of 3 (`possible`/`unknown-key`), example config 2 of 2 (`possible`/`example-config`), prompt mention 3 of 3 (`possible`/`prompt-mention`); none `confirmed`, none with a `flow-unavailable` reason. Offline checkers observed green (R1).
- Dynamic import: 0 `library_api` contract points, 0 Links, 0 `CONSUMES` from `plugins/loader.py::load_plugin`; `AC6-DYN-1..3` and `TW-1` moved to P2 (user decision Q-DYN B, 2026-11-21): **not met at P1**.
- User decision Q-AC6 (2026-11-20): PM-3 = B (Markdown list item, floor 3 kept); EX-1 = B' (a Markdown mention `mcp__fx-svc-remote__list_orders` bound through a second `.mcp.example.json` URL entry, floor 2 kept, no engine change). Caveat: the reason shows only because the same document's plain mention shares the Link (ledger 2026-11-20; `docs/reviews/connected-code-graph-p1-56-impl.md` observes the two added Links, 84 to 86). The launch x possible matrix cell is empty at P1 (an exception recorded; P2 re-adds). Example config has 2 instances at P1 (EX-1, EX-4); the ledger of 2026-11-10 cites the evaluation protocol §12.2 floor as >= 3 instances in 3 shapes (the third, AC6-EX-3, gates P3): stated, not hidden. Stakeholder decision (2026-11-24 ledger): AC-6 example config at 2 instances accepted at P1.

### AC-8 part 1 (configurable): **reported not closed: harness live-tested 72/72 including the authorised host probes; the real-component run (P1-40) is carried to P2**

- Cited (`docs/test-reports/connected-code-graph-p1-ac8-report.md`, devops `a2d545b24797ca5b0`, committed `4a83e430`; I did not re-run `test_ac8.sh` or any host probe): `cpg/sandbox/tests/test_ac8.sh` in full, **72 passed, 0 failed, 0 not run**, exit 0, with the user-authorised host-address probes (cases 3 to 5) and `.invalid` DNS; 0 `conn` records, no leak, cleanup verified. (History: "no, later" on 2026-11-23, then "sorry, actually i want to run"; the user confirmed the authorisation on 2026-11-24, so the probes ran on the user's authorisation.) Observed here: the offline Python tier, R3, 360 passed.
- The real P1-40 run was not possible: `cpg/sandbox/ac8/run.sh` lacks the four spec §8.1 features (a real store with `REPLICAOF`/`MIGRATE`/`MODULE` disabled, probing the real P1-25/P1-26 containers, a fixed-address probe container, an exit-0 `selfTest == []` run); the real components have no container image and the local model is not reachable from a no-route network (ledger 2026-11-24). Stakeholder decision: carry P1-40 to P2.
- Rows (cited, §3 of that report): 0 computed on real components, rows 8.2/8.4/8.12 vacuous (8.13 mostly), 13 rows unrun; the test-3 "pass" column is stand-in evidence that the harness computes rows, not a result for the real components. Under option C (git fetch deferred) the `R2` clauses and proxied rows are vacuous. The hosted-default dependency download, git-remote fetch and mixed-system local-model paths have no real run behind them. **Q-E (how AC-8 closes without the fetch path) is still unanswered.**
- Also closed: P1-29 broker core (`docs/reviews/connected-code-graph-p1-29.md`, `-29-security.md`) and the P1-31 harness (`docs/reviews/connected-code-graph-p1-31.md` Pass 2; `-31-security.md`; `cpg/sandbox/ac8/`).

### AC-9 (staleness, code-snapshot stamp): **closed for a stated sub-scope (code-snapshot half); decision-state half vacuous until P4**

- Closing steps: P1-19, P1-39 (a/b), P1-42. Evidence (cited, live): all 9 answers carry the same stamp and every `staleness.units` entry has `builtCommit`, `builtAt`, `buildStatus`; same commit gives the same `snapshotId` and equal `facts_bytes`; a changed commit gives a different `snapshotId` (`docs/test-reports/connected-code-graph-p1-report.md` P1-42; `cpg/tests/test_fixture_gate_p1_ac.py`). The decision-state half is only `verdictLogOffset` 0 on an empty verdict log: the stamp form is closed, the content is open to P4.

### AC-9b (partial failure): **closed for a stated sub-scope: crossings a binding chain reaches; an unbound crossing into a never-built unit is not flagged (P2)**

- Closing steps: P1-20, P1-41 (rows AB-1, AB-2, LNK-42, LNK-43), P1-57. Evidence (cited): 81 of 81 rows pass live incl. these four (`-p1-report.md` "P1-41 dispatch C"; `cpg/tests/test_fixture_gate_p1.py`); LNK-42/43 assert `possible`/`target-not-mapped`, provider None, `TO_UNIT` `not-mapped`. Composition (review M3): the rows fail a **unit** (`fx-svc` `legacy`) inside one repository where the AC says "a repository"; of the four rows only LNK-42 was mutated (killed); AB-1, AB-2, LNK-43 were not. AB-3/AB-4 gate P3. The residual is the architect's disclosed limit (P1-57), recorded in the ledger 2026-11-21 ("flag at close"); it was not observed by any run I read.

### AC-10 (in the flow, impact stages): **closed at P1 on the analyst's review, flagged "behaviour unobserved"**

- Closing step: P1-27 (`claude/teco/teco.md`, `skills/cpg-analysis/SKILL.md`, `.mcp.json`; `docs/reviews/connected-code-graph-p1-27-impl.md`, Pass 3 approve with suggestions; allowlist test observed R6, 273 passed). Stakeholder decision (Q-4 answered, ledger 2026-11-23): AC-10 closes at P1 on the analyst's review, flagged "behaviour unobserved". No real architect or implementer brief on a covered system was inspected (P1T-39 has no finding file); the closure rests on prompt-level review alone. Decision: the behaviour is to be observed by a qa pass with a live model via the broker (not yet done).

### AC-11 (answer forms, impact): **closed (P1T-19 executed, pass); one-repository answer and a self-written render check are the stated limits**

- Closing steps: P1-24, P1-24b, P1-26, P1T-19. Cited (closing-items report §4, `cpg/eval/results/p1t-20/p1t19.json`): a parser-resolved question (no model call, no `provenance`) returns 599 items, grouped by repository, component, distance in both text and Markdown (0 differences against `ccg.render.common.component_groups`; 599 of 599 items in the text and in the Markdown; 0 located hops without a line; interpretation line first); CLI `--format text` and `--format markdown` exit 0 with bytes equal to the library renderings; 3 planted render failures each caught in both forms. Limits: the proving system is one repository (no repository-order mutant), the check is the executor's own and compares with the engine's grouping function, and the F-3 plant used an item `qname` because no `Cond` hop exists at P1. Observed: R5 290 passed (`cpg/tests/test_render.py`, `test_clients.py`).

### AC-16 (clients): **closed for MCP, CLI, engine and the parser-path plain-language client; the raw-recipe leg closes as sufficiency only, with narrow coverage**

- Closing steps: P1-25 (a/b), P1-26, P1-42. Evidence: `cpg/tests/test_pin_replay.py` (offline tier observed R2, 12 passed; live legs cited); `cpg/tests/test_fixture_gate_p1_ac.py` (R1 offline observed). Cited, live (`-p1-report.md` P1-42): engine, CLI, MCP give byte-equal `facts_bytes` on 9 of 9 requests (6 walk, 2 contract-point, 1 location); plain language through engine, CLI, MCP equals the structured facts; echoed request replayed as a pin gives the same facts for `impact` and for `depends-on`.
- **Raw leg, named as the review requires** (`docs/reviews/connected-code-graph-p1-42-impl.md` F1): `test_pin_replay.py` covers the **call column of a flag-true unit only** (flag-true impact, `CALLS`/`CALLS_NAME`/`HAS_CALL`, one world, one pin); any other shape fails, not skips. The P1-42 replay adds 179 hops of 51 items on 9 answers with 13 recipe reads, and the first level of the 3 symbol/location asks. It proves the recipe text run as the raw role returns every emitted hop (**sufficiency of what was emitted**); it does **not** detect an omitted item past level 1, an omitted contract-point or depends-on item, nor a changed item `status`, `distance` or grouping, and `flow-unavailable` is checked as a consistent fact, not derived. The plain-language leg is the parser path only (no model call); the free-prose model leg is AC-24's.

### AC-17 (starting points): **closed for a stated sub-scope: code location and contract point (2 of 4 groups); queue name to P2; branch and stack trace not built**

- Closing steps: P1-21, P1-26, P1-42. Cited, live: `fx-svc core/flow.py:17` and a TypeScript location resolve, contract `GET /api/search` resolves (5 items), plain-language forms resolve without naming a repository, language or graph (fixture repositories injected, O-2); 3 of 3 non-code files answer `unresolved`/`not found`. Branch and stack-trace asks return a typed `NotBuilt` (asserted, not only recorded). **The queue-name group was not asked** and moves to P2's exit (review `-p1-42-impl.md`; ledger 2026-11-21). Structured location anchors require a `repo` id (`-p1-report.md` O-2); only the plain-language path omits it, and that was run with injected fixture repositories. The AC text names four groups.

### AC-18 (evidence): **closed (for the impact answers asked), with a reading of "each step"**

- Closing steps: P1-22, P1-23, P1-24, P1-42. Cited, live: 51 items on 9 answers, 0 violations; every `Symbol` (116 hops), `CallSite` (76) and `ConfigEntry` (1) hop carries file and line, checked against the materialized fixture tree (file exists, line in range, call-site line names the callee); 26 `Link` hops carry `lifted` entries with file and line or an adjacent code hop. The 11 `ContractPoint` hops carry a key, not a file: P1-42 read "file and line at each step" as not applying to a contract point (see section 5, item 3). Stakeholder decision (ledger 2026-11-23): "each step" excludes contract-point hops; closed with that reading recorded.

### AC-21 (read-only): **closed**

- Closing steps: P1-19, P1-25, P1-42. Cited, live: 25 of 25 write attempts refused by the database for the access-control reason (6 write shapes x `GRAPH.QUERY`/`GRAPH.PROFILE` with `ccg_raw_<s>`, the same 6 through `raw_query`, 7 further commands), view unchanged (stamp, 1346 nodes, 2045 edges equal before and after); MCP `isError`, CLI exit 2 `raw-refused`; positive control: the build credential's `CREATE` succeeds on a scratch key (`-p1-report.md` P1-42; `docs/reviews/connected-code-graph-p1-25a-impl.md`). Observed offline: R6 273 passed (`cpg/tests/test_acl_unit.py`, `test_atrest.py`).
- `raw_query` memory limit (required statement, `-p1-gates.md` P1-38; stakeholder decision 2026-11-19, cited `docs/plans/connected-code-graph-p1-raw-query-memory-graph.md`, graph-dba's read-only look on its own throwaway): a raw query can exhaust instance memory and drop the no-persistence snapshot. `GRAPH.RO_QUERY` is not gated by `maxmemory` (no `denyoom`); only the 2g container cap bounds it, and an OOM kill loses the snapshot. Controls on: `maxmemory` 1536mb/`noeviction`, container 2g, `TIMEOUT_MAX` 10 s, `RESULTSET_SIZE` 10000 (per `cpg/infra/ccg-falkordb.sh`; the live store's config was not read). Absent: `MEM_LIMIT`, per-user limits (P4, FR-24). `QUERY_MEM_CAPACITY` exists but is off (recommendation 256 MiB, not applied). This is a documented P1 limit, **not** a safety claim.

### AC-24 (understanding), smoke: **reported not met at P1 for the local model (stakeholder decision 2026-11-22); no held-out run was spent**

- Closing step: P1-35. Cited (ledger 2026-11-21/22; `docs/reviews/connected-code-graph-p1-35-impl.md`, `docs/plans/connected-code-graph-p1-interpreter-design-fault.md`, `-interpreter-model2-ml.md`; results `cpg/eval/results/ac24-dev2/`): dev entry check (needs >= 11/12 unambiguous and 2/2 ambiguous) failed on `qwen/qwen3-4b-2507` at 9/12 [0.468, 0.911] and 1/2; dev2 (sealed `cpg/eval/dev/ac24-r2.jsonl`, 14 lines) qwen3 8/12 [0.391, 0.862], 1/2; ministral-3-3b 5/12; nemotron probe failed; 7 attempts in all. Arithmetic (reviewer-recomputed, `docs/reviews/connected-code-graph-p1-impl.md` M2): exact P(8 or fewer of 12 | true 0.92) = 0.012; 9 of 12 gives 0.065 (not distinguishable from 0.92 at 5%). The 8/12 and 5/12 results are evidence against the 0.92 target at this n; they come from 14 dev items used for development with the prompt revised between runs, not a held-out measurement. The verdict "not met at P1" stands. `ccg-prereg-ac24-v1` never appended (grep count 0, observed); held-out unspent. User decision "Accept: not met at P1, carry to P2"; a lower local target is set from a P2 measurement. Disclosures: prompt revision between dev and dev2, dev2 shape.
- The smoke harness itself is sound: observed R4, 177 passed (stubs and synthetic items).

### AC-25 (impact shape): **closed for the data-flow half, the shape and the reverse question; the control half is reported not closed (V-both fallback), and AC-25's confirmed form is not met**

- Closing steps: P1-22, P1-23/23b/23d, P1-39 (a/b/d), P1-41, P1-42. Cited, live: shape on 9 answers, 0 violations (direct iff distance 1, within bound 2, ordered repository/component/distance, confirmed before possible, every `possible` carries a reason); data-flow effect DF-1, DF-2, DF-4 at their P1 forms in the AC-25 test, and DF-3 (the one effect that crosses a `Link` into another repository, so the only one where grouping by repository discriminates) in the row gate (`cpg/tests/test_fixture_gate_p1.py`, 81/81); reverse question for `apply_discount`: `base_price` d1 `confirmed`, `find_hook`, `is_member`, `load_rate` d2 `possible`/`flow-unavailable:python`. **Control half: `CF-CD` lists `finalize` and `notify` only and carries the `flow-not-enumerated` notice; the guarded-code rows CD-1, CD-3, CD-4 are not listed.** Reason: python flow is unavailable (`FALSE:measured`), the shipped answer is V-both (stakeholder 2026-11-19; `docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md`). Data-flow items are `possible`, never `confirmed`, at P1.

## 5. Items found contradicting the AC text (for the owner to judge)

1. **AC-5 floor.** AC text: >= 75 individual known impacts. The narrowed http+mcp pair is 72 counting items, 3 short (cited counts report); scripts/wiring cannot be linked at P1. The deferral treats the 75 floor as applying to the P2 combined gate.
2. **Plan wording still saying AC-1/2/4 closed.** The proposal's P1 row (`docs/plans/connected-code-graph.md` §4: AC-1 ✓, AC-2 ✓, AC-4 ✓, AC-5 on HTTP/MCP/scripts/flow, AC-25 ✓) and `-p1-gates.md` P1-38 ("AC-1, AC-2, AC-4 closed") contradict the recorded statuses (AC-1 sub-scope, AC-2 MCP half only, AC-4 not closed, AC-5 deferred, AC-25 data-flow half). A **plan-wording fix is owed**. The ledger of 2026-11-22 records an AC-25 "data-flow half only" fix; I did not recheck every plan line.
3. **AC-18 "file and line at each step".** `ContractPoint` hops carry a key, not a file and line (11 hops). Decided: excluded from the reading (section 6); the AC text does not say so.
4. **AC-16 "returns the same facts".** The raw leg cannot detect an omitted item past level 1 nor a changed status/distance/grouping: "same facts" is shown for three of four clients and for sufficiency only on the fourth.
5. **AC-17 "each of the four groups".** Two groups answered; branch and stack trace `NotBuilt`; queue name not asked.
6. **AC-25 "both effects are listed".** The control-dependency effect is not listed for CD-1/3/4; data-flow items are `possible`.
7. **AC-6 "dynamic import".** Three of the four P1 kinds met; dynamic import not (Q-DYN B). Example config has 2 instances where the evaluation-protocol floor was cited as 3 (accepted at P1, section 6).
8. **AC-9 "which decision state".** At P1 an empty log (`verdictLogOffset` 0): vacuously met.
9. **AC-8 clauses on git-remote fetch, dependency download and hosted override.** No built path to exercise them at P1; no real-component run exists.
10. **AC-1 "listed".** 17 of 18: one dev item, a prompt-embedded `curl` URL, is not listed (DEF-1).
11. **AC-5 gold drift.** The strict gold provider line span matches 3 of 8 dev items (`api.py`/`storefront_api.py` changed after the gold was pinned, D7); symbol identity 8 of 8. Gold not repinned (ledger 2026-11-20).
12. **Dev optimism.** Every dev figure here was used for development; a dev pass is not AC-5 evidence.

## 6. Stakeholder decisions recorded, and what stays open

Recorded exactly as the coordinator stated them (ledger 2026-11-23/24):
- AC-10 closes at P1 on the analyst's review, flagged "behaviour unobserved" (Q-4 answered).
- AC-6 example-config at 2 instances accepted at P1.
- AC-18 "each step" excludes contract-point hops (closed with that reading recorded).
- AC-1 closed for sub-scope: 17/18 on dev; the one miss is a shell/Markdown `curl` URL source (the P2-deferred class), carried as DEF-1.
- AC-2 MCP half closed, wiring half not (P2). AC-4 not closed (P2).
- AC-8 part 1 = harness live-tested 72/72 incl. host probes; the real-component run (component image + in-scenario local-model path, spec §8.1 features 1-4) carried to P2; rows 8.2/8.4/8.12 vacuous and 13 rows unrun.
- AC-5 held-out gate deferred to P2 (user, 2026-11-23). AC-24 not met for the local model (user, 2026-11-22).

- Carried work (AC-4, AC-2 wiring, AC-5 held-out gate, AC-8 real-component run, DEF-1, AC-24 local model, the P2 items above) goes to a **P1 close-out unit before P3** (user, Q1).
- The AC-8 host probes were authorised by the user (confirmed 2026-11-24).
- AC-10 behaviour is to be observed by a qa pass with a live model via the broker.
- The two P1-caused reds (`test_p1_11b_the_repos_own_mcp_json_names_its_script`, the package-data-glob red) are being fixed now (Q4).

Disclosures carried (cited; reviewer m1): oracle capture limits (`docs/reviews/connected-code-graph-p1-47-impl.md`): no `mcp_tool` oracle edge at P1, SPA edges at file grain, `salesperson/tests/e2e` not run, `/order/advance` one key for three transitions, second `/session` join not driven (S10 defect). P1-34 flow decision (`cpg/eval/decisions/ccg-p1-flow-python-v2-decision.json`): labellers worked from sample source text alone (a recorded deviation), the entry value had zero margin (465/481), the two adjudicators agreed 120 of 120 and are one model family. S36 redo: items 40, 10, 17 count as correct only in one configuration the rubric counts.

Still open: Q-E (AC-8 closure without the git-fetch path); the P2 backlog items named under AC-5; the plan-wording fix of section 5 item 2.

## 7. Coverage, gaps, recommendations

- Not covered by any run of mine: all live tiers (cited only), the held-out gate, P1T-39 (no brief inspected), the queue-name starting point, the wiring `unit`-anchor items (custodian probe), live-tier flakiness (one run each; the closing items ran twice with identical results on one store), build-to-build determinism (one build).
- Feedback from the closing-items run (cited): the plan's AC-4 and wiring filters select 1 anchor/1 item and 1 anchor/3 items, far smaller than the strata; the P1T-19 render check and its F-3 `Cond.code` plant need a shipped checker and a flow-true fixture; the HEAD build took 29.5 min on this host against the 16 to 21 minute estimate (one sample).
- Recommendations: fix the stale proposal and `-p1-gates.md` P1-38 wording at the close; before P2 spends the held-out list, land P5-1..P5-3, repoint `STRATA_GATED`/`MIN_COUNTING`, add the dev-gold pre-gate probe, rerun the no-workaround rehearsal; keep the `REVIEW.md` lines the ledger holds for the close (replay checks are sufficiency only; guard tests plant every named untouchable).
- Parallel P1-38 deliverables not covered here: spike dispositions (`docs/test-reports/connected-code-graph-p1-spike-dispositions-report.md`), the `analyst` whole-phase review `docs/reviews/connected-code-graph-p1-impl.md`.
