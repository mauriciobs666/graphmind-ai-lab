# Connected code graph — P1 live QA pass (post P1-51b/52/53a/53b/53c/54)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** P1-41 (prerequisites P1-51b, P1-52, P1-53a/b/c, P1-54; dispatch B)

*2026-11-21: section "P1-41 dispatch C" appended (its 81-row, all-pass verdict supersedes dispatch B's 87-row verdict). 2026-11-20: section "P1-41 dispatch B" appended; the two sentences of sections 2 and Recommendations that called the census driver unbuilt are rewritten.*

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

**Instrument** (scratch, not committed: `qa/test_qa_census.py` and `qa/test_qa_rows.py` in the session scratchpad): the fixture system is materialized from `cpg/fixtures/system1` and every unit built through the sandbox (`ccg.build.build_units`, the same fixture as `tests/test_fixture_binding_live.py`), then `run_packs` (the 9 default packs), `Binder` and `link`, then `build_rows` (assembler, no store write). No store was touched; Docker toolchain only. Each of the P1 rows (gate read through the committed manifests' merged attributes, `p1Form` where a row carries one, the manifest form otherwise) is compared with the Link(s) from its consumer locator. The committed driver is `cpg/tests/test_fixture_gate_p1.py` (P1-41 dispatch B, section below); this section records the scratch instrument of the earlier pass, whose rows D1-D6 below are since fixed or ruled.

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

Recommendations: (1) architect/tdd: decide D1 first, since it decides whether the AC-6 floor is 2 or 1 at P1: either emit `example-config` for `.mcp.example.json` (EX-1) or restate the floor as 1 and take it to the stakeholder (the ruling counted EX-1 on a ledger claim the live build does not bear out). (2) D2 to D5 are rows the manifest gates at P1 that nothing asserts; each needs a call: fix the engine, move the row to P2 with a `p1Form`, or correct the manifest. AC6-URL-2 looks like a P2 (Compose-binding) expectation left at P1. (3) D6: fix the locator (`...index.ts::query`) or the extractor naming; add a live-tier locator-resolution check, since TP-002 passes it. (4) D7: re-pin `ac5-http.jsonl` provider lines or define the hit by provider `qname`. (5) Done: the census driver is committed (`cpg/tests/test_fixture_gate_p1.py`, section "P1-41 dispatch B").



## P1-41 dispatch B: the 87 P1 fixture rows (2026-11-20)

CPG: considered, not relevant — the subject is the `ccg` build/link/answer pipeline over the materialized fixtures; no `cpg_<component>` Joern graph answers it.

**Superseded by dispatch C (end of this report): this verdict was 76 of 87 P1 rows at expected, 11 red (defects D8-D14 below), all since fixed or moved.**  Run 2026-11-20 against HEAD `039d5785` plus the uncommitted driver files, one run of the live tier on the throwaway store `ccg-falkordb-tw-16393` (started and used by this run; `ccg-falkordb:6390`, `falkordb-dev:6379` and `ccg-falkordb-tw-16390` untouched, nothing stopped). Command (cwd `cpg/`): `CCG_GATE_P1_OUT=<ledger.json> CCG_FALKORDB_PORT=16393 CCG_CONFIG_DIR=<scratch> CCG_REQUIRE_LIVE=1 .venv/bin/python -m pytest tests/test_fixture_gate_p1.py -m "live or not live" -q -p no:cacheprovider` -> `12 failed, 79 passed in 264.27s` (91 tests: 3 offline, 87 row tests, 1 closed-world test; 11 red row tests plus the red closed-world test). The ledger JSON written by the run holds 87 row verdicts: 76 pass, 11 fail.

### 1. Row set (two definitions)

Recomputed from `cpg/fixtures/system1/expected/*.yaml` (`M.holders` and the merged `gate`): **134 distinct row ids, 87 gate P1** (`LNK` 27, `SK` 18, `AC6` 11, `NC` 6, `KN` 5, `DF` 4, `CD` 4, `TW` 3, `RV` 3, `MR` 2, `AB` 2, `FT` 2); P2 18, P3 25, P4 1, P5 3. Second definition (raw manifest entries, the holder copies in `fx-svc`/`fx-web`/`fx-job` counted separately): 175 entries, **111 with gate P1**. The 87 equals the ledger's and the plan's; no `B1F-*`/`B1S-*` row is P1. The offline test `test_the_p1_row_set_is_the_87_distinct_ids_of_the_manifests_and_every_row_has_one_means` asserts 134/87/175/111 and that the 87 ids equal the union of the driver's per-family means (no row without a means, none twice).

### 2. Driver and means

`cpg/tests/test_fixture_gate_p1.py` and `cpg/tests/fixture_gate_p1_helpers.py` (uncommitted, for the coordinator to commit). One pytest test per row id; every expectation is read from the manifests (`CCG_GATE_P1_EXPECTED` names another manifest directory: the mutation hook); the status/reason is the row's `p1Form` where it carries one. Instruments: **world** = fixtures materialized, every unit built through the sandbox, `run_packs` (9 default packs), `Binder`, `link`, `build_rows`; **store** = the SC-STEADY scenario refreshed into the throwaway store (the real build of the materialized HEAD builds `svc-legacy` *fresh*, so `svc-legacy` and `svc-tools` are failed by a `refresh(build=...)` wrapper to reach the plan's steady state, see O-1) and read by Cypher and `Client.ask`; **SC-FAIL-UNIT** for AB-1 (fx-svc at H1, then H4 with `svc-legacy` failing).

| Means | Rows | n | Reuse of an existing test |
|---|---|---|---|
| Link-level, world + store (provider set, status, link reason, `BOUND_BY` ordinals, consumer key for KN) | LNK-01, 02, 04, 06, 07, 08, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 36, 37, 38, 45; KN-1..5; AC6-EX-1, EX-4, PM-1..3; TW-2 | 31 | `test_fixture_binding_offline.link_violations` (LNK-45/16/19/12/02/06, AC6-EX-4 chains) and `p56_link_violations` (AC6-PM-3, EX-1) are called on the same build |
| Binder outcome (world) | LNK-21, LNK-22 | 2 | `check_chain`/`DOCKERFILE_ENTRY` of the offline pin |
| Keyless fan-out + CONSUMES fact + `build_rows` (world + store) | AC6-URL-1, 2, 3 | 3 | none |
| Call edges (store: `CALLS`, `CALLS_NAME`) | LNK-30, 39, 40, TW-3, MR-PY, MR-TS | 6 | none |
| `library_api` consumer fact (world) | AC6-DYN-1, 2, 3, TW-1 | 4 | none |
| `DATA` edge (store) | FT-1, FT-2 | 2 | none |
| Negative controls (store) | NC-1, 2, 3, 5, 6, 7 | 6 | NC-5 composes LNK-12/19/20 |
| `SkippedFile` + `File` (store, OFF form) | SK-1..18 | 18 | the unit tier is `test_skipped_files.py` (not rerun) |
| Scenario build (store) | AB-1, AB-2, LNK-42, LNK-43 | 4 | none |
| `Client.ask` (store) | DF-1..4, CD-1..4, RV-1..3 | 11 | none |

Total 87. The AC6-URL keyless assertion is made on the pack's `CONSUMES` fact (kind `http`, key `None`, `possible`/`runtime-built`), never the lifted fact (QA report D5); `build_rows` holds **no** `CONSUMES` edge for a keyless site (it drops it, `consumer-key-none`), so the edge-level reason `runtime-built` is observable only on that fact: the driver asserts that absence too. LNK-22 binds a synthetic site `http://api:8000/api/health` through the real `Binder` (no stored Link at P1).

### 3. Results by row id

**Pass (76):** LNK-01, 02, 04, 06, 07, 08, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 30, 36, 37, 38, 39, 40, 45; KN-1..5; AC6-URL-1, AC6-URL-2, AC6-EX-1, AC6-EX-4, AC6-PM-1, AC6-PM-2, AC6-PM-3; TW-2, TW-3; DF-1, DF-2, DF-4; CD-1, CD-2, CD-3, CD-4; RV-1, RV-2, RV-3; NC-1, 2, 3, 5, 6, 7; FT-1, FT-2; AB-1, AB-2; SK-1..9 and SK-11..18. **Red (11):** AC6-DYN-1, AC6-DYN-2, AC6-DYN-3, AC6-URL-3, DF-3, LNK-42, LNK-43, MR-PY, MR-TS, SK-10, TW-1.

Per family (pass/total): LNK 25/27, KN 5/5, AC6 7/11, TW 2/3, DF 3/4, CD 4/4, RV 3/3, NC 6/6, FT 2/2, MR 0/2, AB 2/2, SK 17/18. Observations behind the passes: LNK-02/06/12 each link to exactly their `svc` provider, `confirmed`/`binding-confirmed`, `BOUND_BY` [0]; LNK-45 `possible`/`prompt-mention`, [0, 1]; LNK-16 `port-shared`, [0, 1]; AC6-EX-4 `example-config`, [0, 1, 2]; AC6-EX-1 `example-config`, [0, 1]; LNK-19 both `reindex` providers `confirmed`/`key-match`, no `BOUND_BY`; LNK-20 four providers `shell-dynamic`; LNK-08 and TW-2 no Link, consumer `CONSUMES` `possible`/`partial-key` with the p1Form key. The walk rows: `CF-DF` holds exactly DF-1, DF-2, DF-4 with notices `[python]`, `depthBound.bound` 2, `unexplored` >= 1; `CF-CD` exactly `finalize` (d1 `confirmed`) and `notify` (d2 `possible`/`flow-unavailable:python`), CD-1/3/4 not listed; RV-1, RV-2 and RV-3 each hold exactly their P1-form items. AB-1: after H1 then H4, `svc-legacy` is `stale-kept` at the H1 commit with `failedAt`, the other `fx-svc` units `fresh` at H4, and the answer for `legacy/app.py::fee` carries that unit in `staleness.units` (`buildStatus` `stale-kept`, `builtCommit` H1, `reason` `extract:failed`, `failedAt`). AB-2: `svc-tools` `not-mapped`, every other unit `fresh` in SC-STEADY. SK: OFF form (see O-2).

**Pass the two-instrument comparison:** the 31 Link-level rows are green under both the world and the store instrument (no row is red under one only).

### 4. Defects (rows asserted and red; routed back, no assertion weakened, no source, fixture or manifest edited)

| ID | Sev | Rows | Title | Expected (manifest) | Observed |
|---|---|---|---|---|---|
| D8 | Medium | DF-3 (item), closed world | `DbStore.load` is a false-positive HTTP consumer: `self.session.fetch(item_id)` is read as a keyless fetch | no consumer at `svc:state/stores.py::DbStore.load` (no row names it as a consumer; it is only LNK-40's candidate) | 12 `possible`/`unknown-key` Links (world; 10 in the store, which has two providers fewer) to every http provider; it is an unplanted item in the `CF-DF3` answer (d2). `state/stores.py:14` |
| D9 | Medium | DF-3 | `CF-DF3` answer differs from DF-3's P1 form in four items | exactly `place`, `create_order`, `accept_quote`, `requeue` (d2 `possible`/`flow-unavailable:python`), `forward`, three `save` callees | `requeue` is d2 `possible`/`partial-key`; unplanted `clients/jobs.py::call_named` and `app/src/api/dyn.ts::go` (d2 `possible`/`unknown-key`: the P1-55 keyless Links, so the manifest's p1Form pre-dates P1-49/P1-55); plus D8's `DbStore.load`. Likely manifest-fix (requeue, call_named, go) |
| D10 | Major | AC6-DYN-1, 2, 3; TW-1 | No `library_api` consumer is produced at P1: AC-6 dynamic import is 0 of 3 | `possible`/`dynamic-import` consumers (Python `import_module(name)`, `__import__("plugins." + name)`, TS `` import(`./plugins/${name}`) ``); TW-1 `confirmed`/`key-match` for the literal | no pack in `cpg/packs` emits kind `library_api` (9 default packs run); no consumer fact at `plugins/loader.py::load_plugin`, `load_legacy`, `load_audit`, `app/src/plugins/load.ts::loadPlugin`; the calls exist only as bare callsites (`import_module`, `__import__`, `import`) with no edge. The fixtures plan §6 floor for the group at P1 is 3 |
| D11 | Minor | AC6-URL-3 | `dyn.ts::forward` has no keyless `CONSUMES` fact | `runtime-built` consumer (as URL-1 `call_named` and URL-2 `go` show) | only a lifted fact `possible`/`partial-key` with no alternatives; the 12 `unknown-key` Links from `forward` exist, but the edge-level reason `runtime-built` is not on any fact for it |
| D12 | Major | LNK-42, LNK-43 | No `target-not-mapped` Link is produced for a consumer of an unmapped unit's provider | `possible`/`target-not-mapped` Link from `legacy_fee` / `run_tool` | SC-STEADY (`svc-legacy`, `svc-tools` `not-mapped`): zero Links from either consumer; the consumer's `CONSUMES` edge (`confirmed`/`lifted-role`, `GET /legacy/fee`, `GET /tools/run`) points at a contract point with 0 providers. With no bundle there is no provider fact to link to; the plan assigns AC-9b's closure to these two rows. Decision for `architect`: produce the Link from a stale-kept bundle only, or restate the rows |
| D13 | Medium | SK-10 | A production file with `test` inside its name is omitted | `app/src/ui/latest.ts` included with a `file` record | listed `frontend-ignore`, no `File` node; its sibling decoy `e2e-helper.ts` (SK-11) is included |
| D14 | Minor | MR-PY, MR-TS | Precondition of the row not met (fixtures `TP-015`): BLOCKED, not passed | the frontend resolves the call to Y (`Rules.fee`, `Formatter.render`) | `price_with` site `fee` and `formatTotal` site `render` have no resolved `CALLS` edge; only `CALLS_NAME` candidates (`Rules.fee`, `PremiumRules.fee`; `Formatter.render`, `FancyFormatter.render`), so the `resolution-unverified` form cannot arise (it would read `name-match`) |

### 5. Closed world

A Link is unplanted only if it matches no row of any manifest at any gate. **Definition B** (the Link's consumer is a consumer-ish locator of some row; the verdict): world 86 Links, 12 unplanted, all from one consumer; store 76 Links, 10 unplanted, same consumer: `svc:state/stores.py::DbStore.load` (D8). **Definition A** (the consumer is any locator of any row): 0 unplanted in both instruments (`DbStore.load` appears as LNK-40's candidate), so the two definitions differ by exactly D8. `JobPanel.<anon>#1` (`possible`/`partial-key` to `job/trigger_server.py::status`) matches OV-2 (P3) and is "planted, not asserted at P1".

**Planted, not asserted at P1 (47 rows gated later, never red):** P2 (18): AC6-EX-2, AC6-TOP-1..3, LNK-05, 09, 18, 23, 24, 25, 31, 32, 33, 34, 35, 41, NC-4, SQC-JAVA. P3 (25): AB-3, AB-4, AC6-EX-3, AC6-QL-1..3, B1F-R, B1F-RU, B1F-W, B1F-W2, B1S-D, B1S-R, B1S-W, B1S-W2, LNK-03, 26, 27, 28, 29, 44, OV-1..5. P4 (1): SQC-PY. P5 (3): TF-1..3. Also the `OrderCache.entries` field rows (named by the gate text; no manifest row id).

### 6. Instrument spread and counts under more than one definition

| Definition | Count |
|---|---|
| P1 rows, distinct ids | 87 (76 pass, 11 red) |
| P1 rows, raw manifest entries (holder copies counted) | 111 (100 on green rows, 11 on red rows: each red id has one holder) |
| P1 rows by means | Link-level 31 (31 green), Binder 2 (2), fan-out 3 (2), call edges 6 (4), library_api 4 (0), DATA 2 (2), negative controls 6 (6), skipped files 18 (17), scenario 4 (2), `Client.ask` 11 (10) |
| Links, world instrument (all units built) | 86 |
| Links, store instrument (SC-STEADY) | 76; the 10 fewer = `legacy_fee` and `run_tool` (confirmed in the all-built world, absent in the store, D12) + 8 fan-out Links (four keyless consumers x two providers fewer, the providers of the unmapped units) |
| Link-level rows red under world only / store only | 0 / 0 |
| Unplanted Links, definition B / A | world 12 / 0; store 10 / 0 |

### 7. Mutation test of the driver

Copy-aside (manifest directory `cpg/fixtures/system1/expected` copied to the scratchpad; `fixture_gate_p1_helpers.py` copied aside; nothing restored by `git`). Three silent corruptions applied at once: (a) a status: `LNK-45` `possible` -> `confirmed` in the copy of `fx-svc.yaml`; (b) a provider id: `LNK-12`'s provider `mcp/server.py::reindex` -> `mcp/server.py::get_item`; (c) a `BOUND_BY` ordinal: `BOUND_ORDS["LNK-16"]` `[0, 1]` -> `[0, 2]` in the helper. Run `CCG_GATE_P1_EXPECTED=<copy> ... pytest tests/test_fixture_gate_p1.py -m "live or not live" -k test_link_level_row` -> `3 failed, 28 passed in 165.59s`: `LNK-45` ("Link ... is possible/prompt-mention, expected confirmed/prompt-mention", reported by the world, store and pin checks), `LNK-12` ("go to [reindex], expected exactly [get_item, reindex]"), `LNK-16` (ordinals) red and the other 28 Link rows green. The helper was restored from the copy (sha256 equal) and the manifests are unmodified (`git status` shows only the two new driver files). An offline test (`test_a_mutated_manifest_status_turns_the_link_checker_red_and_the_unmutated_one_green`) keeps the same three mutants on synthetic records in the default suite. The ordinal mutant was not run on the other rows of `BOUND_ORDS`.

### 8. Observations and what was not executable

- **O-1 (precondition, SC-STEADY):** the manifests say `svc-legacy` is unbuildable at HEAD (`legacy/broken.py`, syntax error) and `svc-tools` never built. The real refresh of the materialized HEAD builds both `fresh` (`legacy` and `tools` python units, status `fresh`, `failReason` None), so the steady state is reached only through the injected failure; the Link and `Client.ask` rows are asserted in that injected state. Whether the real extractor should fail on `broken.py` is for `architect`.
- **O-2 (SK, ON column):** the runner always passes `override=False` and `jssrc2cpg` has no ignore override (`extractors/joern/skipped.py`), so the ON form (SK-5 `included`) is **not executable here**; SK rows are asserted in the OFF form, which equals the ON form for all rows but SK-5. SK-8 holds: no `scripts` unit exists, no `SkippedFile`/`File` for `scripts/build.test.ts`.
- **O-3:** AC6-PM-1..3 are asserted as "the true target is among the Links, every Link `possible`/`prompt-mention`" (superset allowed). AC6-PM-2 (`reindex` in a table cell) links to both `reindex` providers (`svc:mcp/server.py::reindex` and `job:job/mcp_admin.py::reindex`), the Q-NC5 same-name shape the P1-36 plan already tracks; AC6-PM-3 and AC6-EX-1 are additionally exact through `p56_link_violations`.
- **O-4:** AC6-URL-1 and URL-2 pass at their P1 forms (`possible`/`runtime-built` fact, no stored `CONSUMES` edge, `unknown-key` fan-out to 12 providers in the world, 10 in the store).
- Not run: the whole `cpg/` suite, the unit tiers of SK/NC/MR, the ON-override column, the flag-true form of the walk rows, rendering of the notice text.

### 9. Recommendations

(1) `architect`: D12 first (AC-9b's closure rests on LNK-42/43; produce a `target-not-mapped` Link from what, when the unit never built?), then D10 (a `library_api` pack or move AC6-DYN-1..3 and TW-1 to P2: the P1 AC-6 dynamic-import floor is 0 of 3 as built). (2) `tdd-engineer`: D8 (the Python `fetch` matching `self.session.fetch(...)` is a false-positive consumer, a closed-world defect that also pollutes DF-3), D13 (SK-10) and D11. (3) Manifest/plan owner: D9 (re-derive DF-3's p1Form after P1-49/P1-55) and D14 (MR rows blocked at TP-015; decide whether the precondition is a P1 or P3 matter). (4) The driver is red by design until these land: commit it with the red rows listed above, or gate on the defect ids.


## P1-41 dispatch C: the 81 P1 rows against the final manifest (2026-11-21)

CPG: considered, not relevant — the subject is the `ccg` build/link/answer pipeline over the materialized fixtures; no `cpg_<component>` Joern graph answers it.

**Verdict: 81 of 81 P1 rows at expected, 0 red; closed world 0 unplanted Links in both instruments.** HEAD `e40e2f17` plus the uncommitted driver edits (`cpg/tests/test_fixture_gate_p1.py`, `cpg/tests/fixture_gate_p1_helpers.py`; commits 57a, 57c, 57d are in). One live run on the throwaway `ccg-falkordb-tw-16394` (started and removed by this run; `ccg-falkordb:6390`, `falkordb-dev:6379`, `ccg-falkordb-tw-16390` were running before and after, untouched). Command (cwd `cpg/`): `CCG_GATE_P1_OUT=<ledger> CCG_FALKORDB_PORT=16394 CCG_CONFIG_DIR=<scratch> CCG_REQUIRE_LIVE=1 .venv/bin/python -m pytest tests/test_fixture_gate_p1.py -m "live or not live" -q -p no:cacheprovider` -> `86 passed in 340.41s`. The 86 are 3 offline, 81 row tests, the closed-world test and the new DbStore.load test. The ledger JSON holds 81 row verdicts, 81 pass.

### Driver changes
- Row set: `len(P1) == 81`, `(raw, raw_p1) == (175, 105)`; `LIB_IDS` (AC6-DYN-1/2/3, TW-1), `test_library_api_row`, and MR-PY/MR-TS (and their branch of `test_call_edge_row`) removed (P2). `sorted(MEANS) == sorted(ROW_IDS)` holds (offline test green). Means counts from the ledger: link 31, skipped-file 18, `Client.ask` 11, negative control 6, scenario 4, call-edge 4, fan-out 3, DATA edge 2, Binder 2.
- AC6-URL-3: `expected_form` carries `consumes_fact` (default true; the p1Form's false); `test_keyless_url_row` then skips the edge-reason and dropped-edge checks and asserts the pack emitted no http CONSUMES fact at `forward`; the Links stay `possible`/`unknown-key` fan-outs.
- LNK-42/43 (`test_target_not_mapped_row`): suspected bug (provider compared with `unit:<rootPath>` against a linker `unit:<hex id>` anchor) is **not real** for the instrument that test reads: the store's Link has no `TO` Symbol and a `TO_UNIT` Unit whose `rootPath` the driver reads, so `unit:legacy`/`unit:tools` is right (the hex form appears only in `world_recs`, which this test does not use). Added regardless an explicit store assertion: Link `possible`/`target-not-mapped`, provider Symbol None, `TO_UNIT` = Unit `fx-svc` with rootPath `legacy` (LNK-42) / `tools` (LNK-43), buildStatus `not-mapped`; the contract-walk check is unchanged.
- New `test_dbstore_load_is_no_consumer` (D8 regression): no consumer fact at `svc:state/stores.py::DbStore.load` in the world, no Link from it in world or store, no CONSUMES edge in the store.

### Results
All 81 pass (per-row verdicts in the run's ledger JSON): the LNK rows (incl. LNK-42 and LNK-43 via the scenario test), KN-1..5, AC6-URL-1/2/3, AC6-EX-1/4, AC6-PM-1/2/3, TW-2, TW-3, FT-1/2, NC-1/2/3/5/6/7, SK x18, DF-1..4, CD-1..4, RV-1..3, AB-1, AB-2 (the ledger holds the per-row evidence; defects D8, D9, D11, D12, D13 and the D10/D14 moves are cleared by 57a/57c/57d and the P2 move). Closed world: world 74 Links, store 68 Links, 0 unplanted under both definitions (A and B). DbStore.load: 0 facts, 0 Links.

### Mutation spot-check
A copy of the manifests (`CCG_GATE_P1_EXPECTED`) with two expectations broken, run on the same throwaway with `-k "LNK-42 or URL-3"`: `2 failed in 229.31s`, both killed. LNK-42 set to `confirmed`/`key-match`: `Link is possible/target-not-mapped, provider symbol None, to_unit fx-svc:'legacy' (not-mapped); expected confirmed/key-match`. AC6-URL-3 `consumes_fact: true`: `no http CONSUMES fact for ['web:app/src/api/dyn.ts::forward'] ... lifted facts [('possible', 'partial-key', [])]`. (A carrier-level break, e.g. removing the Compose `legacy` service, was not run.)

### Gaps and notes
Not covered: a carrier-level mutation; the known reds (`test_p1_11b_*`, leaf-harness survivor-tell, package-data-glob, `test_scip_parse` x4, `test_units` golden, `test_ac8_lib` flake) were not run or touched; the DF-3 manifest note `pending_57a` still sits in `fx-web.yaml` although the live CF-DF3 answer now matches its p1Form (DF-3 passed): suggest dropping the note (a manifest edit, not done here). One run only, so flakiness is unmeasured.
