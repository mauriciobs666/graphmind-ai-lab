# Connected code graph — P1 live QA pass (post P1-51b/52/53a/53b/53c/54)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** P1-41 prerequisites (P1-51b, P1-52, P1-53a/b/c, P1-54)

CPG: considered, not relevant — the subject is the `ccg` build/link pipeline and its stored output; no `cpg_<component>` Joern graph answers it.

Plan: `docs/test-plans/connected-code-graph-p1.md` (P1T-05, P1T-08, P1T-12) and `docs/test-plans/connected-code-graph-fixtures.md`.

## Summary

Run 2026-11-20 against HEAD `24cdff79` (contains P1-51b, 52, 53a, 53b, 53c, 54), store `ccg-falkordb` on 6390 only, new snapshot `483767a48aecd809219aa738f6c1f8ca` (1306 units, all fresh). **Verdict: partial pass, not green for the AC-6 floor.** Green: refresh, AC-5 8 of 8 by provider symbol on both instruments (store = bundle), LNK-11 `possible`/`partial-key`, the five 53b rows at their contract values, slice-live 3 passed, pin files 109 passed. Not green: **the example-config floor is 1 of 2 in the live run (D1, major)**; five more rows the manifests gate at P1 are not at expected at Link level (D2-D6) while no pin asserts them; AC-5 by the strict gold-line span is 3 of 8 on both instruments because the gold provider lines are stale (D7). The census recomputes 87 P1 rows (of 134). The census compared 34 of the 87 at Link level (28 at expected); 53 are out of this instrument. Method names and figures below are command output of this session.


## 1. Store refresh

Command (cwd `cpg/`, started 20:04, finished 20:41, wall about 37 min; the system has one repository, so `--scope graphmind-ai-lab` is the whole system): `CCG_SYSTEM=graphmind-ai-lab CCG_FALKORDB_HOST=127.0.0.1 CCG_FALKORDB_PORT=6390 .venv/bin/python -m clients.cli.main refresh --scope graphmind-ai-lab`. The units' commit is the local clone HEAD `24cdff79dc87377c06e8330d4159c9eba4c85ad2` (a store query returns it as the single `builtCommit`).

- New snapshot id **`483767a48aecd809219aa738f6c1f8ca`** (`builtAt` 2026-11-20T23:28:40Z; was `a4d7d0bed057e1ef270e5922c0192130`).
- Report line: `refresh graphmind-ai-lab: 1306 fresh, 0 stale-kept, 0 not-mapped (head is the local clone's)`; `unknownKind 0; skippedDocs 0; undecodablePaths 0; pruned 1233; prune errors 0; state errors 0`; `dropped consumer-key-none 4`.
- Per-unit freshness (from the CLI JSON `units`, 1306 entries): every unit `fresh`, `failReason` null; markdown 865, config 314, shell 113, python 12, javascript 2. Store check: `MATCH (u:Unit) RETURN u.buildStatus, count(*)` gives `fresh 1306`. The unit count was 1233 (python 10) at the previous snapshot; `git ls-tree -r --name-only` lists 3121 files at `01cb0dba` and 3448 at HEAD, so the increase tracks tree growth (I did not attribute it unit by unit).
- Stage times / peaks (report): build 1340.23 s, resolve 301.00 s, link 353.33 s, assemble 13.71 s, load 11.16 s, swap 0.02 s; process peak 1,609,244 KB; host MemAvailable low-water 136,104 KB (load). Previous run's link stage was 134.04 s: see Observation O-2.
- Store content after the swap (raw Cypher): Link 3165 (was 1814), Symbol 16948, CallSite 92117, ContractPoint 116, Pack 9 (was 8), `CONSUMES` 167, `PROVIDES` 172, `MOUNTED_BY` 207. Links by status/reason: possible/unknown-key 2254, possible/prompt-mention 514, possible/partial-key 191, confirmed/key-match 206.
- Only `ccg-falkordb:6390` was written (the CLI default); `falkordb-dev` and `ccg-falkordb-tw-16390` were not touched; no service was stopped or restarted.

## 2. Fixture census (P1T-05 instrument, Link level)

**Instrument** (scratch, not committed: `qa/test_qa_census.py` and `qa/test_qa_rows.py` in the session scratchpad): the fixture system is materialized from `cpg/fixtures/system1` and every unit built through the sandbox (`ccg.build.build_units`, the same fixture as `tests/test_fixture_binding_live.py`), then `run_packs` (the 9 default packs), `Binder` and `link`, then `build_rows` (assembler, no store write). No store was touched; Docker toolchain only. Each of the P1 rows (gate read through the committed manifests' merged attributes, `p1Form` where a row carries one, the manifest form otherwise) is compared with the Link(s) from its consumer locator. There is no committed census driver (P1-41 dispatch B is unbuilt): the existing offline/live pins assert only LNK-45, 16, 19, 12, 02, 06, AC6-EX-4.

**Row count (recomputed from the manifests): 134 distinct row ids, of which 87 have gate P1** (`LNK` 27, `SK` 18, `AC6` 11, `NC` 6, `KN` 5, `DF` 4, `CD` 4, `TW` 3, `RV` 3, `MR` 2, `AB` 2, `FT` 2). Matches the ledger's 87. The fixture build produced **48 Links** (store-side `Link|TO|Symbol` 48, `CallSite|VIA|Link` 48, `Link|BOUND_BY|ConfigEntry` 10).

| Outcome | Rows | Count |
|---|---|---|
| Link level, **at expected** status/reason (and provider set) | LNK-01, 02, 04, 06, 07, 08*, 11, 12, 13, 14, 15, 16, 17, 19, 20, 36, 37, 38, 45; KN-1..5; AC6-EX-4, AC6-PM-1, AC6-PM-2; TW-2* (* `p1Form.linked: false`: no Link observed, as expected) | 28 |
| Link level, **not at expected** (defects D1-D6 below) | AC6-EX-1, AC6-PM-3, AC6-URL-1, AC6-URL-2 (no consumer/Link observed); AC6-URL-3 (Link right, edge reason not observable); LNK-10 (status/reason right, consumer locator does not resolve) | 6 |
| Binder outcomes, not stored Links at P1 (plan P1-53); asserted by the offline launch pins, not compared here | LNK-21, LNK-22 | 2 |
| Edge-level rows (`CALLS_NAME`/call/import edges, no Link expected); not compared by this Link-level instrument | LNK-30, 39, 40, AC6-DYN-1, 2, 3, TW-1, TW-3 | 8 |
| Scenario rows (need SC-STEADY/SC-FAIL-UNIT: a build with `svc-legacy` failed and `svc-tools` unbuilt); this single all-units-built build links both providers `confirmed`/`key-match` | LNK-42, LNK-43 | 2 |
| Extractor/engine rows (skips, negatives, flow forms, answers, field depth, mis-resolution, unit staleness): out of this instrument | SK-1..18, NC-1/2/3/5/6/7, DF-1..4, CD-1..4, RV-1..3, MR-PY, MR-TS, AB-1, AB-2, FT-1, FT-2 | 41 |
| **Total** | | **87** |

(28 + 6 + 2 + 8 + 2 + 41 = 87.) So 34 rows were compared at Link level: 28 at expected, 6 not. The other 53 are not asserted by this pass; they remain P1-41 dispatch B's.

**The five rows the ledger says P1-53b changed** (committed fixture edits in `d8c5c815`: `fx-svc/scripts/Dockerfile` (new), `fx-svc/deploy/Dockerfile`, `fx-job/deploy/compose.dev.yaml` (new)): all observed at their 53b values. LNK-16 `possible`/`port-shared` to `trigger_server.py::status`; LNK-45 `possible`/`prompt-mention` with `BOUND_BY` ords [0, 1]; LNK-02 and LNK-06 `confirmed`/`binding-confirmed` (BOUND_BY ord [0]); LNK-12 `confirmed`/`binding-confirmed` to exactly `server.py::reindex` (no `job:job/mcp_admin.py::reindex` link). LNK-19 stays `confirmed`/`key-match` to both `reindex` providers with no `BOUND_BY`; LNK-20 `possible`/`shell-dynamic` to all four providers. I did not rebuild the pre-53b fixtures, so "exactly five rows changed" is **not independently reproduced**; what is observed is that these five are at the contract values and the other 23 comparable rows are at manifest values.

**Unplanted links:** none by consumer. Every one of the 48 Links has a consumer symbol that some manifest row names (JobPanel's Link belongs to the P3 row OV-2, "planted, not asserted at P1"); the one name that does not match its row is LNK-10 (D6).


## 3. AC-5 8-script count (P1-49 Done-when 2, P1-51b Done-when (a))

Population: the 8 dev items of `cpg/eval/dev/ac5-http.jsonl` whose impacted element is a Python script symbol (`impacted.path` ends `.py` and contains `/scripts/`: 8 found). Snapshot `483767a48aecd809219aa738f6c1f8ca`, commit `24cdff79`. Dev items only, nothing tuned.

| Method | Definition | Result |
|---|---|---|
| **Store method, link to gold provider symbol** | raw Cypher on 6390 as `ccg_raw_graphmind-ai-lab`: the impacted symbol's call site has a `VIA` edge to a `Link` that is `ON` the gold keyNorm and `TO` the symbol with the gold provider `qname`, any status | **8 of 8** |
| **Bundle method** (stored bundles of the 14 python/javascript units, real `run_packs` with the 9 default packs and `link`, no Binder) | a `Link` from the impacted symbol to the gold provider symbol, any status | **8 of 8**, equal item by item to the store method |
| Store method, `CONSUMES` edge carrying the **exact gold `keyNorm`** from the impacted symbol's call sites | | **6 of 8** |
| Store method, any `CONSUMES` edge from the impacted symbol's call sites | | 8 of 8 |
| Both instruments, **strict gold-line span** (provider symbol spans the gold chain's provider line) | the P1-49 wording "spans the gold provider line" | **3 of 8** (store and bundle agree) |

Item detail (store = bundle, status/reason of the Link; line spans are the provider symbol's current `lineStart`-`lineEnd` against the gold line): `load_append.py::main` GET /health confirmed/key-match (api.py 58-67, gold 58); `seed_nlq_eval_corpus.py::_check_server_reachable` GET /health confirmed/key-match (58-67); `load_append.py::_worker` POST /threads/{}/messages possible/partial-key (95-121, gold 95); `seed_nlq_eval_corpus.py::_poll_until_terminal` GET /documents/{} possible/partial-key (253-258, gold 252); `language_salience_probe.py::run_participant` and `load_demo.py::run_participant` POST /shop/api/session possible/partial-key (storefront_api.py `join` 1027-1092, gold 1020); `language_salience_probe.py::_assistant_rows_since` GET /shop/api/messages possible/partial-key (1110-1144, gold 1103); `seed_nlq_eval_corpus.py::main` POST /documents confirmed/key-match (167-198, gold 166).

Disagreements: (1) **`CONSUMES`-with-gold-key (6) vs link-to-provider (8)**: items `_worker` and `_poll_until_terminal` carry `CONSUMES` on the partial keys `POST /threads/{?}/messages` and `GET /documents/{?}` (not the gold `{}` forms) while their Links reach the gold provider; not a defect, a method difference. (2) **Strict line span (3) vs symbol identity (8)**: five gold provider lines no longer fall inside the provider symbol's span, because `falkor-chat/server/falkorchat/api.py` and `storefront_api.py` changed after the gold was pinned (commit `496c5d40`, K-094, touched both; the spans are 1 to 7 lines past the gold). Both methods agree, so this is gold drift, not an engine miss: defect D7.

P1-51b Done-when (d) (the `join` Link of item 5 no longer carries the keyless sites 397 and 438): observed. Store: the Links into `build_storefront_router.join` from `salesperson/scripts/` are exactly `language_salience_probe.py:384` and `load_demo.py:171` (`possible`/`partial-key`); no `VIA` edge exists from `language_salience_probe.py` lines 397 or 438. Done-when (b), the ten-unit before/after diff, was **not run** (needs the pre-51b build).


## 4. AC-6 example-config floor at P1 (floor 2: EX-1, EX-4)

Counted from the live fixture build (`build_rows` output, every edge and node carrying `reason = example-config`):

- **EX-4: shows it.** `CallSite|VIA|Link` edge with `{status: possible, reason: example-config, consumerKey: GET /api/health}` and the matching `Link` node (`possible`/`example-config`, matcher `exact`), consumer `svc:clients/peer.py::peer_health` to `svc:api/health.py::build_router.health`, `BOUND_BY` ords [0, 1, 2] (the 3-entry chain through `deploy/compose.example.yaml`).
- **EX-1: does not.** `fx-svc/.mcp.example.json` is built as a config unit: its `File` has `role: example` and its `configEntry` records carry `example: true`, but no edge or node with reason `example-config` exists for it: the only `ConfigEntry|LAUNCHES|Unit` edges are 3 (from `Dockerfile`, `deploy/Dockerfile`, `scripts/Dockerfile`; all `p` null), and `facts.launch_facts` holds 6 records, none from the example file. No Link has `.mcp.example.json` as consumer.
- Result: **1 instance shows `example-config` (EX-4); the floor of 2 is not met in the live run.** (The row-level floor in `tests/test_fixtures.py` counts manifest rows, not emitted reasons, and is green.) Defect D1.
- Real proving-repo store (snapshot `483767a4...`): 0 edges and 0 Links with reason `example-config` (`MATCH ()-[r]->() WHERE r.reason='example-config'` returns no rows), so the real store offers no second source.


## 5. LNK-11 live check and slice-live

**LNK-11 (live fixture build, P1T-08):** `job:job/run_report.py::requeue` has exactly one Link, to `svc:api/orders.py::build_router.create_order`, **`possible`/`partial-key`**, key `POST /api/v1/orders`, matching the manifest `p1Form` (`possible`/`partial-key`, the P1 form: the wrapper's base is a caller-unseen `self.base`). Underlying facts: the wrapper `ApiClient.post` (`job/api_client.py:12`) is a lifted non-leaf (`\x01\x01`); the leaf at `requeue:38` is `possible`/`partial-key` with alternatives `["/api/v1/orders"]`; the consumer fact is `POST /api/v1/orders` `possible`/`partial-key`; the `CONSUMES` edge's reason is not `lifted-role`. The caller stays the consumer (no informative wrapper slot), as P1-51b's regression note requires. **PASS.**

**slice-live (P1-51b Done-when (e), P1T-02):** `cd cpg && CCG_FALKORDB_PORT=6390 .venv/bin/python -m pytest -m "slice and live" -q -p no:cacheprovider` (the two selected files, `tests/slice/test_normalize_equivalence.py` and `tests/slice/test_extract_equivalence.py`, hold no store access; the port variable was set per the README line and is unused): `3 passed, 16442 deselected in 118.22s`. **PASS** (0 skipped: the oracle artifact was present).

**Pin run** (cwd `cpg/`, `.venv/bin/python -m pytest tests/test_fixture_binding_live.py tests/test_fixture_binding_offline.py tests/test_fixtures.py tests/test_linker_bind_real_shapes.py -q -p no:cacheprovider -m "live or not live"`): `109 passed in 190.79s`, including the two live-tier tests (LNK-45/16/19/12/02/06/AC6-EX-4 links and the `jobs.py` environ-read shape). The pins are green while D1-D6 stand, because none of them asserts those rows (see Coverage).


## Defects

Severity by user impact on the P1 acceptance bar (AC-6, P1T-05/P1T-12). Reproduction for D1-D6: build the fixture system as in section 2 (`tests/test_fixture_binding_live.py::fixture_build` plus `ccg.assemble.rows.build_rows`) and look at the named consumer.

| ID | Sev | Title | Expected (manifest/plan) | Actual (observed) |
|---|---|---|---|---|
| D1 | **Major** | AC-6 `example-config` floor is 1 of 2 in the live run: EX-1 shows no `example-config` | AC6-EX-1 (`fx-svc/.mcp.example.json`, kind `launch`) `possible`/`example-config`; with EX-4 the floor of 2 (architect ruling, ledger 53c) | `build_rows` over the fixture build has exactly one element with reason `example-config` (EX-4's `VIA` edge and its `Link`). `.mcp.example.json` yields `File.role=example` and `configEntry.example=true` records, no `LAUNCHES`/`VIA`/`Link` row (the three `LAUNCHES` edges are Dockerfile-derived, `p` null; 6 launch facts, none from the example file). The ledger's "engine emits example-config for EX-1" is not what the live build does |
| D2 | Medium | AC6-PM-3 (Python triple-quoted prompt naming `list_orders`) yields nothing | `svc:prompts/system.py::<module>` `possible`/`prompt-mention` (gate P1), true target `mcp/server.py::list_orders` | no consumer fact, no Link from `prompts/system.py`; the prompt-mention extractor is Markdown-only (`docs/reviews/connected-code-graph-p1-test-plan.md` Pass 5 notes "AC6-PM-3 stays a Python file") and no plan text defers the row |
| D3 | Medium | AC6-URL-1: consumer is partial-key with no Link, not runtime-built with an unknown-key Link | `clients/jobs.py::call_named` edge `runtime-built`, link `unknown-key` (fan-out) | consumer fact `GET /{?}` `possible`/`partial-key` at line 21; zero Links from `call_named` (cause not investigated) |
| D4 | Medium | AC6-URL-2: no consumer attributed to `dyn.ts::go` | `possible`, `runtime-built / binding-narrowed` (gate P1, no `p1Form`; the base is Compose-bound, which may be a P2 expectation left at P1 in the manifest) | `go` is only a non-leaf lifted fact (`\x01\x01`, `confirmed`/`lifted-role`); no consumer, no Link |
| D5 | Minor | AC6-URL-3 edge reason not as the row states | `dyn.ts::forward`: `runtime-built / unknown-key` | the Link half holds (12 `possible`/`unknown-key` Links to 12 distinct providers); the leaf's lifted reason is `partial-key`, not `runtime-built` |
| D6 | Minor | LNK-10 consumer locator does not resolve to an extractor symbol | consumer `web:app/src/api/index.ts::api.search.query` | the extractor's symbol is `app/src/api/index.ts::query`; the Link (to `search`, `confirmed`/`key-match`, ordered `LIFTED_VIA` entries `index.ts:5` -> `client.ts::request` -> `client.ts:3 fetch`) is right. A locator-level P1-41 assertion would find no symbol |
| D7 | Minor (instrument) | AC-5 strict gold-line span is 3 of 8 at HEAD on both instruments | provider symbol spans the gold provider line | falkor-chat's `api.py` and `storefront_api.py` moved after the gold was pinned (commit `496c5d40`): the provider spans start 1 to 7 lines after the gold lines. By provider symbol identity the count is 8 of 8 on both instruments. Not an engine defect; the P1-49 wording needs the symbol-identity form or the gold needs repinning |

No new red anywhere else in the scope I ran: slice-live 3 passed, pin run 109 passed. The full suite was **not** run in this pass (it was run elsewhere by the implementer; the known reds from the brief are not re-reported).

## Observations (not defects)

- O-1: Link count 3165 (was 1814 at the previous snapshot): possible/unknown-key 2254, possible/prompt-mention 514, possible/partial-key 191, confirmed/key-match 206. The 2254 keyless fan-out Links are the P1-36 trade-off the plan already tracks; I have no per-reason baseline from the previous snapshot to say which part is new.
- O-2: refresh link stage 353.33 s against 134.04 s at snapshot `a4d7d0be...`, build 1340.23 s against 813.59 s, with 1306 units against 1233; host MemAvailable low-water 136,104 KB during load (previous 1,139,204 KB). Another process was using memory during part of the run (free memory was about 0.9 GB at 20:26), so the timings are not a clean regression signal.
- O-3: LNK-42/43 need the SC-STEADY/SC-FAIL-UNIT scenario builds (P1T-04); this single build cannot assert them.


## Coverage, gaps, recommendations

Covered: the refresh against 6390 (1306 units fresh), AC-5 by two instruments and four definitions, the Link-level fixture census (34 of 87 rows compared, 28 at expected), the five 53b rows, LNK-11, EX-4, 51b Done-when (a) and (d), slice-live, and the four pin files. Not covered: the 53 rows of section 2 that are not Link-level (edge, Binder, scenario, extractor and engine rows), 51b Done-when (b) (the ten-unit before/after diff), the pre-53b fixture rebuild (so "five rows changed" is not reproduced), the full suite, and any `ask`/answer-level assertion (P1-41 dispatch B).

Recommendations: (1) architect/tdd: decide D1 first, since it decides whether the AC-6 floor is 2 or 1 at P1: either emit `example-config` for `.mcp.example.json` (EX-1) or restate the floor as 1 and take it to the stakeholder (the ruling counted EX-1 on a ledger claim the live build does not bear out). (2) D2 to D5 are rows the manifest gates at P1 that nothing asserts; each needs a call: fix the engine, move the row to P2 with a `p1Form`, or correct the manifest. AC6-URL-2 looks like a P2 (Compose-binding) expectation left at P1. (3) D6: fix the locator (`...index.ts::query`) or the extractor naming; add a live-tier locator-resolution check, since TP-002 passes it. (4) D7: re-pin `ac5-http.jsonl` provider lines or define the hit by provider `qname`. (5) The live pins assert 7 of the 34 Link-level rows; a committed census driver of the section 2 instrument would have shown D1 to D6 before this pass.

