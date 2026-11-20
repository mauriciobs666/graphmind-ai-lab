# Connected code graph — post-P1-50 rebuild, measurements and fixture-row re-derivation

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 (P1-50, P1-49 Done-when 2)

CPG: considered, not relevant — the subject is the `ccg` build/link pipeline and its stored output over the proving repository; no `cpg_<component>` Joern graph answers it.


## Summary

Run 2026-11-20 against HEAD `01cb0dba` (contains the P1-50 commits `31e4021f` and `deb7c7db`), `NORMALIZER_VERSION` `0.7`. Verdict: **A done; B green except the 8 known reds (no new failure); C 6 of 8 (target 8, not met, causes stated); D Part B changed no provider or mount key and nothing became newly `confirmed`; E walk rows equal their forms, `UNIT_FLOORS` svc and four plan items re-derived, and three high defects (D1-D3) found on the fixture build that keep the P1-41 rows red.** Artifacts: this report, `docs/test-plans/connected-code-graph-p1.md` (revised in place), `cpg/tests/test_fixtures.py` (one constant). Nothing was committed.

## A. Store refresh (default system)

Command (cwd `cpg/`, `CCG_SYSTEM=graphmind-ai-lab CCG_FALKORDB_HOST=127.0.0.1 CCG_FALKORDB_PORT=6390`): `.venv/bin/python -m clients.cli.main refresh` (no `--scope`: the whole system; the units' commit is the local clone HEAD `01cb0dba01a829ac0984a60eae6d7e60d9160d87`, which contains deb7c7db). Started 10:38:48, finished about 10:58 (wall 20 min 31 s at last check).

- New snapshot id **`a4d7d0bed057e1ef270e5922c0192130`** (`builtAt` 2026-11-20T13:53:10Z; was `03d0d41b85712818043f901b72e71b29`).
- Report line: `refresh graphmind-ai-lab: 1233 fresh, 0 stale-kept, 0 not-mapped (head is the local clone's)`; `unknownKind 0; skippedDocs 0; undecodablePaths 0; pruned 1223; prune errors 0; state errors 0`; `dropped consumer-key-none 4`.
- Per-unit staleness: 1233 units, all `fresh`, `failReason` null (markdown 833, config 302, shell 86, python 10, javascript 2). Store check: `MATCH (u:Unit) RETURN u.builtCommit, count(*)` -> one row, `01cb0dba…`, 1233.
- Stage timings / peak RSS: build 813.59 s (peak 649 MB), resolve 250.42 s, link 134.04 s, assemble 8.86 s, load 8.70 s, swap 0.01 s; process peak 1,376,648 KB; host MemAvailable low-water 1,139,204 KB (load).
- Store content after the swap (raw Cypher): Link 1814, Symbol 14535, CallSite 73038, ContractPoint 114, Pack 8, `CONSUMES` 153, `PROVIDES` 91, `MOUNTED_BY` 127, `CALLS` 26342, `CALLS_NAME` 72957, `VIA` 2973.
- Only `ccg-falkordb:6390` was written; `falkordb-dev` and `ccg-falkordb-tw-16390` untouched; no service stopped or restarted.

## B. Whole non-live suite plus `slice and live` (P1-50 Done-when 6)

Command (cwd `cpg/`, run alone, 11:01-11:24): `.venv/bin/python -m pytest -q -p no:cacheprovider -m "not live or (slice and live)"`.
Result line: **`8 failed, 15944 passed, 10 skipped, 381 deselected, 1 xfailed, 1 warning in 1403.07s (0:23:23)`**.

| Red test (named) | Known before P1-50? | Observed cause |
|---|---|---|
| `tests/test_ac8_lib.py::test_a_signal_cleans_up_once_and_exits_nonzero_without_resuming[1-129]` | yes (brief: 2 rows; this run **1** row failed, the other signal row passed) | `subprocess.TimeoutExpired: ... s.sh timed out after 20 seconds` (a timing test under load) |
| `tests/test_config_extractor.py::test_p1_11b_the_repos_own_mcp_json_names_its_script` | yes | `assert not True` over the repo's own `.mcp.json` |
| `tests/test_rule_pack_leaf_mutation.py::test_the_mutation_harness_can_tell_a_killed_leaf_from_a_survivor` | yes | `survives(...)` returned False on the harness's own `can_tell` row |
| `tests/test_rule_pack_schema.py::test_every_data_file_under_ccg_is_covered_by_a_package_data_glob` | yes | package-data glob misses `front/grammar/interpret.json`, `front/prompts/interpret.txt`, `front/prompts/summary.txt` |
| `tests/test_scip_parse.py::test_a_file_of_empty_sub_messages_is_a_scip_error_not_a_memory_error_under_a_1_5_gb_limit[...]` x3 (`a.py` occurrence cap, `a.py` symbol cap, `../x` occurrence cap) | yes | `assert 546 < 300` (child `peak_mb 546`: a memory-ceiling assertion, not an OOM kill) |
| `tests/test_scip_parse.py::test_an_occurrence_with_millions_of_range_elements_is_rejected_without_collecting_them` | yes | `assert (True and 546 < 300)` on `peak_mb 546` |

No red outside the known list: **no new defect from the suite**. Skipped (10, all opt-in, named in the run's summary): `test_flow_p1.py:757` and `test_s28_metrics.py:380` x6 (`CCG_S28_RECOMPUTE=1`), `test_fstring_constants.py:90` (`CCG_FSTRING_LIVE=1`), `test_normalize_self_rule_live_golden.py:71` (`CCG_SELFRULE_LIVE=1`), `test_rule_pack_leaf_mutation.py:758` (`CCG_LEAF_MUTATION=1`). XFAIL (1): `tests/slice/test_linker_conformance.py::test_a_client_on_8200_binds_the_server_unit_binding_confirmed_on_real_extractor_output` (no extractor emits falkor-chat's launch facts). The `slice and live` tests ran in this invocation (no `slice` skip of the oracle).

## D. Part B measured effect (P1-50 Done-when 4), recomputed independently

Method (script not committed; scratch `partb_qa.py`): for each of the **ten Python units**, take the raw Joern facts and stage tree that the section A refresh left under `cpg/.ccg-work/joern/<unitId>/` (HEAD `01cb0dba`), rebuild the bundle through `extractors.joern.normalize.assemble_bundle.assemble` twice, **Part B on** (shipped code) and **Part B off** (`Resolver.element` called with `fmt_piece=False`, the code's own switch), then run the same eight default packs (`ccg.refresh.DEFAULT_PACKS`) through `ccg.packrt.run_packs` on each set and diff the facts as canonical JSON multisets. I did not use the implementer's scratch (`partb_effect.py`, its draw facts at `8377c326`): this uses a different commit and different raw facts.

Validation of the instrument: the Part-B-on rebuild of every unit **equals the stored bundle of section A as a record multiset** (10 of 10: claude 24,242, falkor-chat/server 92,762, mcp-monitor 2,382, salesperson 8,394, model-bench 89,887, cypher-mcp 3,125, falkor-chat/scripts 3,028, docs 876, skills 17,574, decision-models 84,784 records; only record order differs).

Observed (command output of `partb_qa.py`):

| Quantity | Part B off | Part B on | Changed by Part B |
|---|---|---|---|
| template valueSets (all units) | 1,835 | 1,835 | 1,866 bundle records differ, record counts equal |
| template valueSets with a const piece | 1 | **1,828** | (of 1,835) |
| provider facts (`providers`) | 91 | **91** | **0 records changed** (no key, status or reason) |
| mount facts (`mounts`) | 3 | **3** | **0 records changed** |
| consumer facts | 122 | **128** | 17 on-side records differ from 11 off-side (6 net new); every changed consumer is `possible`/`partial-key` |
| lifted facts | 67 | 73 | 21 on-side differ from 15 off-side; every changed one is `confirmed`/`lifted-role` on **both** sides |
| launch facts / marks / diagnostics | 1 / 0 / 1 | 1 / 0 / 1 | 0 |

**Result: Part B changed no provider key and no mount key on the ten Python units, and nothing became newly `confirmed`** (changed consumer facts are all `possible`/`partial-key`; the changed lifted facts were `confirmed`/`lifted-role` before and after, the value set only gained its constant text, e.g. `"\x01\x01"` -> `"/channels/\x01/threads"`). The implementer's priors (providers 91->91, mounts 3->3, consumers 122->128) are reproduced exactly, from independently rebuilt facts at a different commit. Limit: the Part-B-off arm is the code's own `fmt_piece=False` switch (the analyst gate noted the same), not a separate pre-P1-50 build.

## C. AC-5 8-script `CONSUMES` count (P1-49 Done-when 2; P1-50 Done-when 5)

Definition applied (plan P1-49 Done-when 2): the 8 dev items of `cpg/eval/dev/ac5-http.jsonl` whose impacted element is a Python script symbol (8 found by the filter `impacted.path` ends `.py` and contains `/scripts/`); an item **hits** when the impacted symbol has an `http` link (a `CALLS`-site `VIA` edge into a `Link`) whose contract key equals the gold `anchor.keyNorm` and whose provider symbol lies in the gold chain's provider file and spans the gold provider line, **at any status**. Dev items only; no held-out file read. Never tuned.

Two independent measurements, equal results:
1. **The refreshed store** (read-only raw Cypher as `ccg_raw_graphmind-ai-lab` on 6390, snapshot `a4d7d0bed057e1ef270e5922c0192130`, links produced by the real refresh including the `Binder`; scratch `ac5_store.py`).
2. **The stored bundles at HEAD `01cb0dba`** (units falkor-chat/scripts, salesperson, falkor-chat/server) through the real `run_packs` (8 default packs) and `link` (no `Binder`; scratch `c_bundles.py`).

**Count: 6 of 8** (store and bundles agree item by item).

| # | Gold item (key; impacted symbol; gold provider) | Result | Evidence (observed) |
|---|---|---|---|
| 1 | `GET /health`; `load_append.py::main`; `api.py:58` | HIT | link `confirmed`/`key-match` to `api.py::build_router.health`; consumer fact `GET /health` `confirmed`/`lifted-role`, line 90 |
| 2 | `GET /health`; `seed_nlq_eval_corpus.py::_check_server_reachable`; `api.py:58` | HIT | link `confirmed`/`key-match`; consumer `GET /health` `confirmed`/`lifted-role`, line 254 (through the `_http_json` wrapper) |
| 3 | `POST /threads/{}/messages`; `load_append.py::_worker`; `api.py:95` | HIT | link `possible`/`partial-key` to `build_router.post_message`; consumer `POST /threads/{?}/messages`, line 60 (own site, `ownSite`) |
| 4 | `GET /documents/{}`; `seed_nlq_eval_corpus.py::_poll_until_terminal`; `api.py:252` | **MISS** | no http link and no consumer fact for the symbol (cause below) |
| 5 | `POST /shop/api/session`; `language_salience_probe.py::run_participant`; `storefront_api.py:1020` | HIT, weak at link level | link to `build_storefront_router.join` is `possible`/`unknown-key`: its three sites are 384 (`partial-key`, `consumer_key` `POST /shop/api/session`), 397 and 438 (keyless `unknown-key` fan-out). The keyed site is among the link's `VIA` sites; the link-level reason is the keyless one |
| 6 | `GET /shop/api/messages`; `language_salience_probe.py::_assistant_rows_since`; `storefront_api.py:1103` | **MISS** | no http link and no consumer fact for the symbol (cause below) |
| 7 | `POST /documents`; `seed_nlq_eval_corpus.py::main`; `api.py:166` | HIT | link `confirmed`/`key-match` to `build_router.ingest_document` (sites 369 `key-match`, 377 `unknown-key`) |
| 8 | `POST /shop/api/session`; `load_demo.py::run_participant`; `storefront_api.py:1020` | HIT | link `possible`/`partial-key` (line 171, matcher `exact` in the store) to `join` |

**Cause of the two misses (items 4 and 6), from the fact dump** (`c_facts.py`): both wrappers build the URL from a template that mixes a parameter with other pieces (`f"/documents/{document_id}"`; `f"{base_url}/shop/api/messages?since={since}&limit={MESSAGE_PAGE}"`). The Part B constant text is present: the intermediate lifted facts at `seed_nlq_eval_corpus.py:289` (`/documents/\x01`) and `language_salience_probe.py:366` (`\x01/shop/api/messages?since=\x01&limit=\x01`) are `confirmed`/`lifted-role`. But each wrapper itself has a lifted caller (`main` line 377 calls `_poll_until_terminal(document_id)`; `run_participant` lines 397/438 call `_assistant_rows_since(base_url, token, ...)`), and the plan's `ownSite` text says "a wrapper with at least one lifted caller is still not a leaf". So the consumer facts are attributed to the **callers**, with the key taken from the leaf slot alone: `main:377` is the bare `document_id` (`\x01`, `possible`/`partial-key`, no key, keyless `unknown-key` fan-out to 60 providers), and `run_participant:397` carries the literal argument `0` as `GET /0`. Neither wrapper symbol gets a `CONSUMES`, so the gold impacted element has no link. This is the "template that mixes a parameter with another unresolved piece" case that plan P1-49 Done-when 2 says to return to the architect "before any engine change"; I changed nothing.

Side observation (not a plan row, for the architect): the consumer fact `GET /0` at `language_salience_probe.py:397` is a wrong URL (the `since` argument substituted as the whole URL by the leaf-slot rule, status `possible`/`partial-key`, `evidence.opaqueSlot` set); it links to no provider. And a link's reason is the weakest across its sites when sites differ in kind of key (item 5: keyed `partial-key` site inside a link reported `unknown-key`), whereas item 7's link keeps `key-match` with a keyless site inside it.

**Target 8/8 not met; 6/8 is the observed count; no figure was tuned.**

## E. Fixture-row re-derivation after 5d / 49b / 50

**Instrument.** A throwaway store of my own (`cpg/infra/ccg-falkordb.sh --throwaway up`, `ccg-falkordb-tw-16391`, scratch config/credentials/work/audit dirs; removed after the run; `ccg-falkordb:6390`, `falkordb-dev` and `ccg-falkordb-tw-16390` untouched), the four fixture repositories materialized by `fixtures/materialize.py` into a scratch dir, `ccg.refresh.refresh` over `cpg/fixtures/system1/system.yaml` with the default packs: **`refresh fixture-system1: 31 fresh, 0 stale-kept, 0 not-mapped`**, snapshot `01bb489cf77c5fb3e741cb4050eef87f`, `dropped consumer-key-none 1`, `dropped provider-key-none 1`, build stage 134.16 s. Store content: 37 links (http 27, mcp_tool 10; by reason `key-match` 12, `unknown-key` 12, `partial-key` 5, `prompt-mention` 4, `shell-dynamic` 4), `BOUND_BY` 0, `LAUNCHES` 1, `binding-confirmed` 0 (re-linked offline with a real `Binder` over the stored fixture bundles: the same 37 links). Expectations are the manifests' `p1Form` and the plan text; where a form was missing I derived it from the fixture source and the plan's stated rule, never from the output.

### E.1 UNIT_FLOORS (`cpg/tests/test_fixtures.py`)
Live run `pytest tests/test_fixtures.py -m live -k tp006_every` **before the edit**: `AssertionError: ('svc', {'methods': 202, 'named': 85, 'expected_callable': 68, 'expected_found': 67, ...})  assert 68 == 67`; all other units equal their floors (job 15/15, lib-cli 3/3, lib-pricing 5/5, svc-common 4/4, svc-legacy 1/1, svc-tools 1/1, web 33/33). Derivation of the 68: the manifests' callable locators in unit `svc` over the materialized tree (`expected_symbol_names`): 68, of which the eight handlers are now `build_router.<h>` and `db/repo.py::Repo.save` is the DF-3 callee run; the one not found, `expected_missing == ["db/settings.py::Settings"]`, is the class without methods the floor comment already documents. **Changed:** `UNIT_FLOORS["svc"]` (67, 66) -> **(68, 67)** with the comment replaced. After: `pytest tests/test_fixtures.py -m live` -> `2 passed, 78 deselected in 64.81s`; non-live `tests/test_fixtures.py tests/test_fixture_keynorm_mutants.py` -> `85 passed, 2 deselected`.

### E.2 Walk rows (asks through `Client.ask`, depth bound 2 on every answer) - all equal their forms
| Ask | Observed items | Equals form? |
|---|---|---|
| `CF-DF` (`base_price`) DF-1, DF-2, DF-4 | `apply_discount` d1 `confirmed`; `accept_quote` d2 `possible`/`name-match`; `build_router.create_order` d2 `possible`/`flow-unavailable:python`; notice `flow-not-enumerated` `["python"]`; `unexplored` 10 | yes (DF-4 now `name-match`: the 5d demotion removed the Joern-linked `resolved-call`) |
| `CF-CD` (`eligible`) CD-2 | `finalize` d1 `confirmed`; `notify` d2 `possible`/`flow-unavailable:python`; CD-1/3/4 not listed | yes |
| `RV-1` (`base_price`) | `is_member` d1 `confirmed`, `load_rate` d1 `confirmed`, `find_hook` d1 `possible`/`name-match`, `connect` d2 and `OrderCache.get` d2 `possible`/`flow-unavailable:python` | yes |
| `RV-2` (`eligible`) | `audit`, `tier_of`, `OrderCache.get` d1 `confirmed`; `find_hook` d1 `possible`/`name-match`; `load_rate` d2 `possible`/`flow-unavailable:python` | yes (`OrderCache.get` is `confirmed` now; it was `possible`/`name-match` before 5d) |
| `RV-3` (`OrderTotal.place`) | `create_order` d1 and `apply_discount` d2 `confirmed`; `Repo.save`, `SqlRepo.save`, `MemRepo.save` d2 `possible`/`name-match`; notices `["javascript", "python"]` | yes |
| `CF-DF3` (`apply_discount`) DF-3 | `create_order` d1 `confirmed`; `accept_quote` d1 `possible`/`name-match`; three `save` callees d2 `possible`/`flow-unavailable:python`; `dyn.ts::forward` d2 `possible`/`unknown-key`; `OrderTotal.place` d2 `possible`/`flow-unavailable:python`; notices `["javascript", "python"]` | **7 of the 8 planned items; `requeue` (d2) is absent** (defect D1 below). The plan's "nine items" is a miscount: its own list is eight |

A25-3 on `CF-DF3`: the d1 group holds `create_order` `confirmed` before `accept_quote` `possible` (observed order); the `CF-DF` and `CF-CD` pairs are vacuous as the plan states.

### E.3 Link rows (store links joined to the manifest rows; consumer matched as symbol or `symbol.<anon>#n`)
| Row | Expected (source and rule) | Observed | Verdict |
|---|---|---|---|
| LNK-01, LNK-04, LNK-07, LNK-36, LNK-38, KN-1..4 | `confirmed`/`key-match` | `confirmed`/`key-match` (consumers `Health.<anon>#1`, `getOrder`, `fetchFile`, `OrderTotal.place`, `Related.<anon>#1`, the Probe functions) | match |
| LNK-13 | `possible`/`partial-key` (P1-49 rule 3: base from `VITE_API_BASE`) | `possible`/`partial-key` (`getDefaultItem`) | match; plan text was stale |
| LNK-17, KN-5 | `possible`/`partial-key` | same | match |
| LNK-20 | `possible`/`shell-dynamic` | same (4 links, `dyn.sh`) | match |
| LNK-08, LNK-37, TW-2 | `confirmed`/`key-match` (constant bases) | **no consumer, no link** | red: D1 |
| LNK-11, LNK-14, AC6-EX-4, DF-3 `requeue` | `possible`/`partial-key` | **no consumer, no link** | red: D1 |
| LNK-02, LNK-06, LNK-12 | `confirmed`/`key-match` (LNK-12 `binding-confirmed`) | **no consumer, no link** | red: D2 |
| LNK-18, LNK-21, LNK-22, LNK-45, LNK-14/12 binding half | `binding-confirmed` | `BOUND_BY` 0, `binding-confirmed` 0, `LAUNCHES` 1 (the job Dockerfile); LNK-45's mention links `possible`/`prompt-mention` | red: D3 |
| LNK-19 | one provider, `svc:mcp/server.py::reindex`, `key-match` (NC-5) | `scripts/refresh.sh::<script>` links `confirmed`/`key-match` to **both** `mcp/server.py::reindex` and `job/mcp_admin.py::reindex`, no `BOUND_BY` | red: D3 |
| LNK-16 | `possible`/`port-shared` to `job/trigger_server.py::status` | no link from `JobStatus` (its consumer fact `GET /job/status` exists; the counts report `loopback_port_residual: 1`) | red: D3 (not root-caused) |
| AC6-EX-2 | `possible`/`example-config` | link `possible`/`partial-key`; the `example-config` reason lives on the binding edge, which D3 removes | unverified |

**Rows changed (E):** `cpg/tests/test_fixtures.py` UNIT_FLOORS svc; test plan `docs/test-plans/connected-code-graph-p1.md` P1T-06 (consumer-locator rule for `<Component>.<anon>#n`, LNK-38 included), P1T-08 (LNK-11 `possible`/`partial-key`), P1T-09 (LNK-13, LNK-14 `possible`/`partial-key`; LNK-15 left `key-match` and marked to re-derive), P1T-17 (eight items, not nine), and the revision line (the stale 2026-11-11 LNK-13/LNK-15 sentence removed). **Not changed, owed:** manifests `fx-svc.yaml`/`fx-job.yaml`/`fx-web.yaml` carry no `p1Form` for LNK-11, LNK-13, LNK-14, AC6-EX-4 (P1T-05 compares the manifest form where there is none, so these rows would be red on a correct build), and `docs/test-plans/connected-code-graph-fixtures.md` rows LNK-11/13/14 still state the old forms; LNK-01/17/38/AC6-EX-2 locators name the component, not the nested arrow. These are fixture-data edits tied to `test_fixtures.py` consistency checks, so I did not edit them under this unit's scope.

## Defects

**D1 (high, blocks P1-49 Done-when 1 and P1-41 rows): the `python-http-client` pack never seeds a typed `requests.<verb>` call.** Steps: materialize the fixtures, build `fx-job`'s Python unit; the bundle's callsites for `job/run_report.py` lines 20, 24, 28, 32 are `calleeName: get`, `resolution: external`, `externalTarget: "requests.py:<module>.get"` (and `post` at `job/api_client.py:12`); `packs/python-http-client/pack.yaml` lines 24-44 seed `get|post|put|patch|delete` only with `resolution: [unresolved]`. Expected: consumers for LNK-08, 11, 14, 15, 37, TW-2, AC6-EX-4 and `requeue` (plan P1-49 Done-when 1). Actual: `run_packs` returns 20 http consumers, all TypeScript; no Python one; the fixture links list has no `job/` or `clients/` consumer; `CF-DF3` lists 7 items, not 8. The plan states "no `requests`/`httpx` verb call is typed anywhere in the stored bundles, so the pack has no typed verb seed (adding one is a leaf no test could exercise)": true of the proving-repository bundles, false of the fixtures, which are the tests. Recommend: architect decides a typed seed (`externalTarget` glob `requests.py:*get` and the other verbs, `httpx.py`), with a fixture-backed test.

**D2 (high): no TypeScript MCP-client consumer pack, so P1 rows LNK-02, LNK-06 and LNK-12 cannot link.** `grep callTool packs ccg/packrt ccg/linker` finds only `packs/fastmcp/pack.yaml:35` (`call_tool`, Python snake_case). The fixture's `fx-web/app/src/mcp/client.ts` calls `c.callTool({name: "get_item"...})`; the run yields no `mcp_tool` consumer from it (consumers of kind `mcp_tool`: only the two shell ones). Expected per `P1T-06`/`P1T-08`: `confirmed`/`key-match` (LNK-12 `binding-confirmed`). Actual: no consumer, no link. No plan step names such a pack (P1-46 is the shell `mcp-call` pack); recommend an architect ruling: add the pack to P1 or move the three rows to P2.

**D3 (high): bindings do not bind in the fixture build.** `BOUND_BY` 0, `binding-confirmed` 0, `LAUNCHES` 1 (only the job `Dockerfile`; the launch facts are `launchUnit` and `port 8300` for `job.trigger_server:app`), although the proving-system store has 114 `BOUND_BY` edges; a `Binder` built with the units and the system key reproduces the same result offline. Consequences observed: LNK-45 `possible`/`prompt-mention` instead of `binding-confirmed`; LNK-19 links to both `reindex` providers (NC-5 asks for exactly one), each `confirmed`/`key-match`; LNK-16 gets no link (`loopback_port_residual: 1`). Not root-caused (which launch/config facts the fixture's `.mcp.json` and Compose do not yield is the next step); severity is high because it takes LNK-12/14/18/19/21/22/45 and LNK-16 red.

**Finding (not a defect against a plan row): AC-5 wrapper items.** See section C: wrappers with lifted callers are not leaves, so `_poll_until_terminal` and `_assistant_rows_since` get no `CONSUMES`; and the lifted key `GET /0` at `language_salience_probe.py:397` is a wrong URL.

## Coverage and gaps
Covered: refresh of the real system on the dedicated store; whole non-live suite plus `slice and live`; AC-5 count on dev items by two methods; the Part B effect on the ten Python units; the fixture build and every impact ask (CF-DF, CF-CD, RV-1..3, CF-DF3); the link rows listed above; `tp006` live. Not covered: the 89-row P1-41 run itself (out of scope, analyst re-gate first); `LNK-23..29`, `AB-*`, `OV-*`, `MR-*`, `TW-1/3`, `SK-*`, `FT-*`, `NC-*` and the other P1 AC6 rows were not compared in this unit; no held-out data was opened; the Part-B-off arm of D is the code's own switch, not a separate pre-P1-50 build; AC6-EX-2's reason is unverified (binding edge absent).

## Feedback and recommendations
1. P1-41 cannot go green on the fixtures until D1, D2 and D3 are ruled on; run the analyst test-plan re-gate with this report attached, and have the architect decide D1/D2 before dispatch B.
2. Add `p1Form` to the manifests for LNK-11, LNK-13, LNK-14, AC6-EX-4 and align the fixtures plan rows (owner `qa-engineer`, a separate small unit with `test_fixtures.py` consistency runs).
3. Decide the consumer-locator convention for nested anonymous functions (E.3 first row) at the re-gate; either the manifests name the arrow or the comparison accepts nesting as the plan now states.
4. For AC-5: an architect decision on lifting through a wrapper whose url template mixes a parameter with constant text (the case P1-49 Done-when 2 pre-announced): either the wrapper stays a consumer at its own site with the template key (`/documents/{}`), or the gold items for these wrappers are re-pointed to the caller.
5. A cheap guard: a test asserting that every Python unit of the fixture system yields at least one http consumer from the `requests` calls would have caught D1 at 49b.
