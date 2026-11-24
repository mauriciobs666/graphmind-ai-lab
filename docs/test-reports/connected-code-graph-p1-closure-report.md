# Connected code graph P1: AC-closure table (P1-38, first deliverable)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1-38)

CPG: considered, not relevant — the subject is the AC-closure bookkeeping of the `ccg` P1 phase (documents, reports, reviews and test runs of the `ccg` pipeline); no `cpg_<component>` Joern graph answers it.

Plan: `docs/plans/connected-code-graph-p1-gates.md` section "P1-38 · phase exit"; `docs/plans/connected-code-graph-p1.md` §8 "AC map at phase exit" and the P1-37/P1-38 rows; AC wording `docs/requirements/connected-code-graph.md`; test plan `docs/test-plans/connected-code-graph-p1.md` §5. Counts and verdicts only: no held-out content is quoted; neither the custodian root `~/.local/share/ccg-eval/cf7b2042cce7` nor any held-out file was read for this report (held-out counts are cited from `docs/test-reports/connected-code-graph-p1-heldout-strata-counts-report.md`).

## 1. How to read this report

- **Observed** = a command run for this report (section 3, with exit status, cwd `cpg/`, HEAD `dacc6958`). **Cited** = read from the named file and not re-derived; the file owns the figure. Every figure below is one of the two, and cited figures say so.
- Status vocabulary: **closed**; **closed for a stated sub-scope**; **reported not closed, because ...**. No AC is marked closed by step completion alone.
- The live tiers (throwaway FalkorDB, Docker sandbox, real model) were **not** re-run here: that would start containers and models (environment mutation) and is hours of wall time. Their figures are cited from the reports of the runs that executed them.
- This report decides nothing. The forks that need a stakeholder or `teco` decision are in section 6.

## 2. Verdict

**Of the 16 AC rows: 2 closed (AC-18, AC-21); 7 closed for a stated sub-scope (AC-6 three of four kinds, AC-9 code-snapshot half, AC-9b bound crossings, AC-11 renderers, AC-16 with a sufficiency-only raw leg, AC-17 two of four groups, AC-25 data-flow half); 7 reported not closed (AC-1, AC-2, AC-4, AC-5 in every stratum, AC-8 part 1 not run, AC-10 unobserved, AC-24 local model not met).** AC-5's held-out gate is deferred to P2 by stakeholder decision (2026-11-23). No new red in R1 to R8; the full-suite run was incomplete (section 3). Details per AC in section 4.

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
| R9 full default suite | `cpg/` whole | 143 (terminated by me at about 42%, after about 15 min, on a host shared with other agents' suites) | progress lines only; no summary line exists |

R9 is therefore **not a result**: its progress log shows one `F` at about 29% of the collected tests, not identified (no failure summary was produced, so it could be a known red or a new one). Rerun on a quiet host to attribute it. R1 to R8 are complete and green.

**New reds: none seen in R1 to R8; R9 incomplete (one unidentified `F`).** The known reds the brief lists (`test_p1_11b_*`, leaf-harness survivor-tell, package-data-glob, `test_scip_parse` x4, `test_units` golden, `test_ac8_lib` flake, `test_queries.sh` section 14.8) are not P1's. `test_queries.sh` (falkor-chat, live FalkorDB) was not run: out of this component and needs the shared instance.

## 4. AC-closure table

Closing steps are the plan's step ids (`docs/plans/connected-code-graph-p1.md` §3 table, `-p1-gates.md`). Evidence paths are repo paths; "cited" figures are not re-derived here.

### AC-1 (HTTP): **reported not closed, because the plan's own closing item (P1T-20 on dev anchors) was never executed**

- Closing step per plan: P1-42, item P1T-20 ("closed on P1-42 alone", `docs/test-plans/connected-code-graph-p1.md` §5, §7). The P1-42 driver `cpg/tests/test_fixture_gate_p1_ac.py` does not contain P1T-19/P1T-20; the report says so: `docs/test-reports/connected-code-graph-p1-report.md` P1-42 O-4 ("Not run ... P1T-19/P1T-20").
- Indicative evidence only (cited, dev anchors, optimistic by the plan's own words): proving-repo store, the 8 dev script-to-endpoint items link to their gold provider symbol 8 of 8 on both instruments (`docs/test-reports/connected-code-graph-p1-report.md` §3; strict gold-line span 3 of 8 = gold drift D7, not an engine miss). Gate-harness rehearsal on dev anchors, real engine, http hits 16/17 (healthy runs) and 17/18 (flow-true plant) (`docs/test-reports/connected-code-graph-p1-gate-rehearsal-report.md` item 4; this was before round E and the D5 engine fix).
- Fixture-level (cited): the HTTP walk and Link rows pass, 81 of 81 P1 rows, closed world 0 unplanted (`docs/test-reports/connected-code-graph-p1-report.md` "P1-41 dispatch C"; test file `cpg/tests/test_fixture_gate_p1.py`; offline tier observed R1).
- Fork: section 6 Q1.

### AC-2 (MCP + wiring): **reported not closed — MCP half not executed under its pass rule, wiring half an engine gap deferred to P2**

- MCP half: same P1T-20 gap as AC-1. Indicative (cited): rehearsal mcp hits 20/20 and 19/19 on dev anchors (`-gate-rehearsal-report.md` item 4); fixture MCP Links pass in the 81 rows.
- Wiring half: dev wiring 0/11 (cited, `docs/reviews/connected-code-graph-p1-gate-zero-hit-rca.md`: engine builds no `config` item for a config entry that names a tool/server and no hook-launches-script hop at P1; `LAUNCHES` beyond `launchUnit` is P2). The `kiro/`, `mcp-monitor/` configuration wiring in the AC text is therefore not listed. Stakeholder decision 2026-11-23 (ledger): carried to P2.

### AC-4 (scripts): **reported not closed, because shell script-by-path invocation and prompt-to-script links are not built at P1**

- Dev scripts 0/49 items (0/46 counting, 0/21 anchors) on the rehearsal (cited: `-gate-rehearsal-report.md` item 4; RCA above: 0 of 1703 answer items are non-`symbol`; the same RCA says the root cause is an engine gap plus a plan inconsistency, not a harness defect). Same P1T-20 gap. Carried to P2 by the 2026-11-23 stakeholder decision. The launch-chain mechanism itself works on fixtures (`BOUND_BY` 10 on the fixture build, cited from `docs/test-reports/connected-code-graph-p1-report.md` §2).

### AC-5 (never miss), per stratum: **reported not closed at P1 — the held-out gate is deferred to P2 by stakeholder decision (2026-11-23)**

No held-out item has been exposed to the gate: no `ac5-gate-1` record exists in `cpg/eval/preregistration.jsonl` (grep count 0, observed), P1-37 was not run and P1-43 does not apply (ledger 2026-11-23). The AC text requires >= 75 counting items; the http+mcp pair has **72** counting items in 19 clusters (3 short), scripts and wiring are not linkable at P1 (cited: `docs/test-reports/connected-code-graph-p1-heldout-strata-counts-report.md`; manifest `cpg/eval/heldout-manifest.json`, `verify` exit 0, checked 77, mismatches 0, cited).

| Stratum | Held-out items (cited) | Dev items and dev hits on the rehearsal (cited, optimistic, dev was used for development) | Status |
|---|---|---|---|
| http | 26 (15 clusters) | 18; hits 16/17 healthy, 17/18 flow-true | not measured at P1 (gate deferred); dev only |
| mcp | 46 (4 clusters; lumpy) | 20; hits 20/20 healthy, 19/19 flow-true | not measured at P1; dev only |
| scripts | 79 (36 clusters) | 49; **0/49** | engine gap (RCA); not delivered at P1, P2 |
| wiring | 8 (5 clusters; exactly the floor of 8, "screened" not "certified") | 11; **0/11** | engine gap (RCA); not delivered at P1, P2 |
| flow | 99 (12 clusters; 0 counting under flag false) | 28; hits 9/28 only in the flow-true plant | **withheld**: `python.flowAvailable` false (`FALSE:measured`), see below |

- Flow stratum: reported not closed, flow-unavailable. P1-34 concluded `FALSE:measured`: decision cell 296/334 = 0.8862, Wilson 95% [0.8477, 0.9160], rule 20y >= 19G failed (M = -389), no registry flip (cited: `cpg/eval/decisions/ccg-p1-flow-python-v2-decision.json`; review `docs/reviews/connected-code-graph-p1-34-impl.md`). JavaScript was not measured (plan).
- Closing steps that did complete (harness, not the gate): P1-32a verbs (`docs/test-reports/connected-code-graph-p1-32a-report.md`), P1-32b `gate.py`/`gate_run.py` gated at P1 (analyst Pass 5, approve with suggestions; `docs/reviews/connected-code-graph-p1-32b-impl.md`), dress rehearsal RED then fixed by rounds E and F (`-gate-rehearsal-report.md`, ledger 2026-11-22/23). Harness tests observed green: R8, 309 passed. Not re-run after round E: the no-workaround dev rehearsal (the delegate's own rerun only), as the ledger says it is P2's when the list is spent.
- Disclosures a P1-37 report would have restated, carried here (all cited from the ledger): labels are AI labels; the EP §7.3 redo of 40 items was done by an AI reviewer on the `fable` model in about 4 minutes, 39 correct 1 incorrect, three `doubt:` items kept as marked (a stricter reading gives 4 of 40 and would fire the escalation trigger), item 26 key-error removal accepted as the AI's call (`docs/test-reports/connected-code-graph-p1-s36-redo-report.md`); the flow-draw custodian role was filled by the coordinator, not the human owner; same-family adjudicators agreed 120 of 120.
- Open P2 items before the gate may spend the list (cited, ledger): Pass 5 minors P5-1..P5-3, `STRATA_GATED`/`MIN_COUNTING` and the `ac5-gate-1` record stale, dev-gold pre-gate probe and per-kind breakdown, rerun of the no-workaround rehearsal.

### AC-6 (doubt flagged), P1 kinds: **closed for a stated sub-scope (URL, example config, prompt mention); dynamic import reported not met at P1**

- Closing steps: P1-41, P1-42 (+ P1-55, P1-56, P1-57). Evidence: `cpg/tests/test_fixture_gate_p1_ac.py`, `cpg/tests/test_fixture_gate_p1.py`; `docs/test-reports/connected-code-graph-p1-report.md` "P1-42" table (cited, live run on a throwaway): URL 3 of 3 (`possible`/`unknown-key`), example config 2 of 2 (`possible`/`example-config`), prompt mention 3 of 3 (`possible`/`prompt-mention`); none `confirmed`, none with a `flow-unavailable` reason. Offline checkers observed green (R1).
- Dynamic import: 0 `library_api` contract points, 0 Links, 0 `CONSUMES` from `plugins/loader.py::load_plugin`; `AC6-DYN-1..3` and `TW-1` moved to P2 (user decision Q-DYN B, 2026-11-21): **not met at P1**.
- User decision Q-AC6 (2026-11-20): PM-3 = B (Markdown list item, floor 3 kept); EX-1 = B' (a Markdown mention `mcp__fx-svc-remote__list_orders` bound through a second `.mcp.example.json` URL entry, floor 2 kept, no engine change). Caveat: the reason shows only because the same document's plain mention shares the Link (ledger 2026-11-20; `docs/reviews/connected-code-graph-p1-56-impl.md` observes the two added Links, 84 to 86). The launch x possible matrix cell is empty at P1 (an exception recorded; P2 re-adds). Example config has 2 instances at P1 (EX-1, EX-4); the ledger of 2026-11-10 cites the evaluation protocol §12.2 floor as >= 3 instances in 3 shapes (the third, AC6-EX-3, gates P3): stated, not hidden.

### AC-8 part 1 (configurable): **reported not run: host-probe allowance declined for now (stakeholder, 2026-11-23: "no, later"); carried until the user allows it. P1-40 has NOT been run; harness built, vacuous rows by design**

- P1-40 needs the user's explicit allowance for the host-address probes `test_ac8.sh` 3 to 5; the stakeholder declined it for now (coordinator message, 2026-11-23, "no, later"; earlier ledger 2026-11-12: "ok go ahead" was not treated as it). Nothing in this report ran `test_ac8.sh` or any host-address or `.invalid` DNS probe; only the offline Python files of R3 ran. No P1-40 report exists in `docs/test-reports/`, no results directory for it under `cpg/eval/results/` (observed `ls`), and the ledger has no delivery entry for it.
- What closed: P1-29 broker core (`docs/reviews/connected-code-graph-p1-29.md`, `-29-security.md`) and the P1-31 harness (`docs/reviews/connected-code-graph-p1-31.md` Pass 2, approve with suggestions; `-31-security.md`; `cpg/sandbox/ac8/`). Harness unit tier observed green: R3, 360 passed. Stand-in runs only (delegate-reported, cited from the ledger 2026-11-12): live subsets 1, 2, 6; the analyst's N-A: a pass is unreachable from `run.sh` with the store stand-in, so "passes only with zero connections" is met at the incomplete level only.
- Vacuous or not exercised under option C (git-fetch deferred, Q1 = C): the `R2` clauses and the proxied rows are vacuous and must be reported as such; the git-remote fetch, hosted-default dependency download and "hosted override refused in a mixed system" paths have no run behind them. Q-E (how AC-8 closes without the fetch path) is unanswered (section 6).

### AC-9 (staleness, code-snapshot stamp): **closed for a stated sub-scope (code-snapshot half); decision-state half vacuous until P4**

- Closing steps: P1-19, P1-39 (a/b), P1-42. Evidence (cited, live): all 9 answers carry the same stamp and every `staleness.units` entry has `builtCommit`, `builtAt`, `buildStatus`; same commit gives the same `snapshotId` and equal `facts_bytes`; a changed commit gives a different `snapshotId` (`docs/test-reports/connected-code-graph-p1-report.md` P1-42; `cpg/tests/test_fixture_gate_p1_ac.py`). The decision-state half is only `verdictLogOffset` 0 on an empty verdict log: the stamp form is closed, the content is open to P4.

### AC-9b (partial failure): **closed for a stated sub-scope: crossings a binding chain reaches; an unbound crossing into a never-built unit is not flagged (P2)**

- Closing steps: P1-20, P1-41 (rows AB-1, AB-2, LNK-42, LNK-43), P1-57. Evidence (cited): 81 of 81 rows pass live incl. these four (`-p1-report.md` "P1-41 dispatch C"; `cpg/tests/test_fixture_gate_p1.py`); LNK-42/43 assert `possible`/`target-not-mapped`, provider None, `TO_UNIT` `not-mapped`. AB-3/AB-4 gate P3. The residual is the architect's disclosed limit (P1-57), recorded in the ledger 2026-11-21 ("flag at close"); it was not observed by any run I read.

### AC-10 (in the flow, impact stages): **reported not closed — reviewed at prompt level, behaviour unobserved; closure by review alone awaits stakeholder question Q-4**

- Closing step: P1-27 (`claude/teco/teco.md`, `skills/cpg-analysis/SKILL.md`, `.mcp.json`; `docs/reviews/connected-code-graph-p1-27-impl.md`, Pass 3 approve with suggestions; `test_atrest` allowlist, observed R6 273 passed). Item P1T-39 has no recorded finding file, no real architect/implementer brief on a covered system was inspected, and Q-4 (`docs/test-plans/connected-code-graph-p1.md` §13) is unanswered in the requirements and the ledger. The plan says: until answered, state "reviewed at prompt level, behaviour unobserved", and do not state AC-10 closed.

### AC-11 (answer forms, impact): **closed for a stated sub-scope: grouped text and Markdown renderers and CLI modes; the end-to-end item P1T-19 was not executed**

- Closing steps: P1-24, P1-24b, P1-26 (reviews `docs/reviews/connected-code-graph-p1-24.md`, `-24b-impl.md`). Observed: R5 290 passed (`cpg/tests/test_render.py` goldens for text and Markdown; `cpg/tests/test_clients.py::test_the_cli_text_and_markdown_modes_are_the_renderers_and_carry_the_notice`). Not done: P1T-19 as an item (ask one parser-resolved question, render both forms) is not in the P1-42 driver and no report records it (`-p1-report.md` P1-42 O-4). Plain-language model interpretation is AC-24's.

### AC-16 (clients): **closed for MCP, CLI, engine and the parser-path plain-language client; the raw-recipe leg closes as sufficiency only, with narrow coverage**

- Closing steps: P1-25 (a/b), P1-26, P1-42. Evidence: `cpg/tests/test_pin_replay.py` (offline tier observed R2, 12 passed; live legs cited); `cpg/tests/test_fixture_gate_p1_ac.py` (R1 offline observed). Cited, live (`-p1-report.md` P1-42): engine, CLI, MCP give byte-equal `facts_bytes` on 9 of 9 requests (6 walk, 2 contract-point, 1 location); plain language through engine, CLI, MCP equals the structured facts; echoed request replayed as a pin gives the same facts for `impact` and for `depends-on`.
- **Raw leg, named as the review requires** (`docs/reviews/connected-code-graph-p1-42-impl.md` F1): `test_pin_replay.py` covers the **call column of a flag-true unit only** (flag-true impact, `CALLS`/`CALLS_NAME`/`HAS_CALL`, one world, one pin); any other shape fails, not skips. The P1-42 replay adds 179 hops of 51 items on 9 answers with 13 recipe reads, and the first level of the 3 symbol/location asks. It proves the recipe text run as the raw role returns every emitted hop (**sufficiency of what was emitted**); it does **not** detect an omitted item past level 1, an omitted contract-point or depends-on item, nor a changed item `status`, `distance` or grouping, and `flow-unavailable` is checked as a consistent fact, not derived. The plain-language leg is the parser path only (no model call); the free-prose model leg is AC-24's.

### AC-17 (starting points): **closed for a stated sub-scope: code location and contract point (2 of 4 groups); queue name to P2; branch and stack trace not built**

- Closing steps: P1-21, P1-26, P1-42. Cited, live: `fx-svc core/flow.py:17` and a TypeScript location resolve, contract `GET /api/search` resolves (5 items), plain-language forms resolve without naming a repository, language or graph (fixture repositories injected, O-2); 3 of 3 non-code files answer `unresolved`/`not found`. Branch and stack-trace asks return a typed `NotBuilt` (asserted, not only recorded). **The queue-name group was not asked** and moves to P2's exit (review `-p1-42-impl.md`; ledger 2026-11-21). The AC text names four groups.

### AC-18 (evidence): **closed (for the impact answers asked), with a reading of "each step"**

- Closing steps: P1-22, P1-23, P1-24, P1-42. Cited, live: 51 items on 9 answers, 0 violations; every `Symbol` (116 hops), `CallSite` (76) and `ConfigEntry` (1) hop carries file and line, checked against the materialized fixture tree (file exists, line in range, call-site line names the callee); 26 `Link` hops carry `lifted` entries with file and line or an adjacent code hop. The 11 `ContractPoint` hops carry a key, not a file: P1-42 read "file and line at each step" as not applying to a contract point (see section 5, item 3).

### AC-21 (read-only): **closed**

- Closing steps: P1-19, P1-25, P1-42. Cited, live: 25 of 25 write attempts refused by the database for the access-control reason (6 write shapes x `GRAPH.QUERY`/`GRAPH.PROFILE` with `ccg_raw_<s>`, the same 6 through `raw_query`, 7 further commands), view unchanged (stamp, 1346 nodes, 2045 edges equal before and after); MCP `isError`, CLI exit 2 `raw-refused`; positive control: the build credential's `CREATE` succeeds on a scratch key (`-p1-report.md` P1-42; `docs/reviews/connected-code-graph-p1-25a-impl.md`). Observed offline: R6 273 passed (`cpg/tests/test_acl_unit.py`, `test_atrest.py`).
- `raw_query` memory limit (required statement, `-p1-gates.md` P1-38; stakeholder decision 2026-11-19, cited `docs/plans/connected-code-graph-p1-raw-query-memory-graph.md`, graph-dba's read-only look on its own throwaway): a raw query can exhaust instance memory and drop the no-persistence snapshot. `GRAPH.RO_QUERY` is not gated by `maxmemory` (no `denyoom`); only the 2g container cap bounds it, and an OOM kill loses the snapshot. Controls on: `maxmemory` 1536mb/`noeviction`, container 2g, `TIMEOUT_MAX` 10 s, `RESULTSET_SIZE` 10000 (per `cpg/infra/ccg-falkordb.sh`; the live store's config was not read). Absent: `MEM_LIMIT`, per-user limits (P4, FR-24). `QUERY_MEM_CAPACITY` exists but is off (recommendation 256 MiB, not applied). This is a documented P1 limit, **not** a safety claim.

### AC-24 (understanding), smoke: **reported not met at P1 for the local model (stakeholder decision 2026-11-22); no held-out run was spent**

- Closing step: P1-35. Cited (ledger 2026-11-21/22; `docs/reviews/connected-code-graph-p1-35-impl.md`, `docs/plans/connected-code-graph-p1-interpreter-design-fault.md`, `-interpreter-model2-ml.md`; results `cpg/eval/results/ac24-dev2/`): dev entry check (needs >= 11/12 unambiguous and 2/2 ambiguous) failed on `qwen/qwen3-4b-2507` at 9/12 [0.468, 0.911] and 1/2; dev2 (sealed `cpg/eval/dev/ac24-r2.jsonl`, 14 lines) qwen3 8/12 [0.391, 0.862], 1/2; ministral-3-3b 5/12; nemotron probe failed; 7 attempts in all. 8 of 12 is not distinguishable from a 0.92 model at this n: "not met on this protocol", not a measured ceiling. `ccg-prereg-ac24-v1` never appended (grep count 0, observed); held-out unspent. User decision "Accept: not met at P1, carry to P2"; a lower local target is set from a P2 measurement. Disclosures: prompt revision between dev and dev2, dev2 shape.
- The smoke harness itself is sound: observed R4, 177 passed (stubs and synthetic items).

### AC-25 (impact shape): **closed for the data-flow half, the shape and the reverse question; the control half is reported not closed (V-both fallback), and AC-25's confirmed form is not met**

- Closing steps: P1-22, P1-23/23b/23d, P1-39 (a/b/d), P1-41, P1-42. Cited, live: shape on 9 answers, 0 violations (direct iff distance 1, within bound 2, ordered repository/component/distance, confirmed before possible, every `possible` carries a reason); data-flow effect DF-1, DF-2, DF-4 at their P1 forms; reverse question for `apply_discount`: `base_price` d1 `confirmed`, `find_hook`, `is_member`, `load_rate` d2 `possible`/`flow-unavailable:python`. **Control half: `CF-CD` lists `finalize` and `notify` only and carries the `flow-not-enumerated` notice; the guarded-code rows CD-1, CD-3, CD-4 are not listed.** Reason: python flow is unavailable (`FALSE:measured`), the shipped answer is V-both (stakeholder 2026-11-19; `docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md`). Data-flow items are `possible`, never `confirmed`, at P1.

## 5. Items found contradicting the AC text (for the owner to judge)

1. **AC-5 floor.** AC text: >= 75 individual known impacts. The narrowed http+mcp pair is 72 counting items, 3 short (cited counts report); scripts/wiring cannot be linked at P1. The deferral decision treats the 75 floor as applying to the P2 combined gate.
2. **AC-5 wording vs the proposal.** The proposal's P1 row (`docs/plans/connected-code-graph.md` §4) says AC-1 ✓, AC-2 ✓, AC-4 ✓, AC-5 on HTTP/MCP/scripts/flow, AC-25 ✓ at P1, and `-p1-gates.md` P1-38 lists "AC-1, AC-2, AC-4 closed": all contradict the 2026-11-23 deferral and the per-AC statuses above. The ledger of 2026-11-22 records an AC-25 "data-flow half only" wording fix; I did not recheck every plan line for the stale form.
3. **AC-18 "file and line at each step".** `ContractPoint` hops carry a key, not a file and line (11 of 51 items' chains). P1-42 read this as the AC not applying to a contract point; the AC text does not say so.
4. **AC-16 "returns the same facts".** The raw leg cannot detect an omitted item past level 1 nor a changed status/distance/grouping, so "same facts" is demonstrated for three of four clients and for sufficiency only on the fourth.
5. **AC-17 "each of the four groups".** Two groups answered; branch and stack trace `NotBuilt`; queue name not asked.
6. **AC-25 "both effects are listed".** The control-dependency effect is not listed for CD-1/3/4; data-flow items are `possible`, not the AC's implied confirmed.
7. **AC-6 "dynamic import".** The P1 row names four kinds; three are met, dynamic import is not (Q-DYN B); example config stands at 2 instances where the evaluation-protocol floor was cited as 3.
8. **AC-9 "which decision state".** At P1 the decision state is an empty log (`verdictLogOffset` 0): the clause is vacuously met.
9. **AC-8 clauses on git-remote fetch, dependency download and hosted override.** No built path exists to exercise them at P1 (option C) and no P1-40 run exists.
10. **AC-5 gold drift.** The strict gold provider line span matches 3 of 8 dev items because `api.py`/`storefront_api.py` changed after the gold was pinned (D7); symbol identity matches 8 of 8. Gold files were not repinned (ledger 2026-11-20): a held-out set with the same drift would score misses under a strict-span reading.
11. **Dev optimism.** Every dev figure in this report was used for development; the plan states a dev pass is not AC-5 evidence.

## 6. Decisions and questions returned (not decided here)

- **Q1 (AC-1, AC-2 MCP half, AC-4):** the plan closes these "on P1-42 alone by P1T-20's pass rule", but P1T-20 (one structured impact request per sampled dev anchor, every dev item listed with a chain) was never executed; only the gate-rehearsal's dev scoring exists (http 16/17, mcp 20/20, scripts 0/49, wiring 0/11, pre-fix, pre-round E). Options: run P1T-20 (a ~20-37 minute proving-system build on a throwaway plus the asks), or accept the rehearsal as the AC-1/AC-2 MCP-half evidence and say so, or leave them reported not closed as above.
- **Q2 (AC-8 part 1):** P1-40 stays not run until the user allows the host-address probes (declined for now). Q-E (how AC-8 closes without the git-fetch path) is also unanswered.
- **Q3 (AC-10, Q-4):** may AC-10 close on the P1-27 review alone, or does it wait for an observed `teco` brief on a covered system?
- **Q4 (AC-11):** P1T-19 end-to-end (one parser-resolved question rendered as text and Markdown) is cheap but was not run; run it, or accept the renderer tests?
- **Q5 (AC-6):** is example config at a floor of 2 (EP §12.2 says 3 shapes) an accepted P1 form? Q-AC6 B' keeps floor 2.
- **Q6 (AC-18):** is "file and line at each step" read as excluding contract-point hops?

## 7. Coverage, gaps, recommendations

- Not covered by any run in this report: the live tiers (cited only), P1T-19, P1T-20, P1T-39, P1-40, the held-out gate, the queue-name starting point, flakiness of the live tiers (one run each), a completed full-suite run (R9 stopped at about 42%).
- Recommendations: fix the stale proposal and `-p1-gates.md` P1-38 AC wording (section 5, items 2) at the close; before P2 spends the held-out list, land P5-1..P5-3, repoint `STRATA_GATED`/`MIN_COUNTING`, add the dev-gold pre-gate probe, and rerun the no-workaround rehearsal; keep the `REVIEW.md` lines the ledger holds for the close (replay checks are sufficiency only; guard tests must plant every named untouchable).
- Parallel P1-38 deliverables not covered here: spike dispositions (`docs/test-reports/connected-code-graph-p1-spike-dispositions-report.md`), the `analyst` whole-phase review `docs/reviews/connected-code-graph-p1-impl.md` (not yet written when this report was produced).
