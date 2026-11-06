# Connected code graph: P0 vertical-slice report (S14, P0-12)

> **Status:** archived · **Owner:** `coder` · **Tracks:** P0-12 (S14)

This is step S14 of `docs/plans/connected-code-graph-p0-2.md`: it scores the slice (S11 extraction,
S12 assembly, S13 load and impact) against SL-1..SL-7 (§4.4) and carries the findings S15 needs.
**It declares no freeze and no "provisional"; that is S15's** (`analyst` + `graph-dba`).

Contract read: `docs/plans/connected-code-graph-bundle-spec3.md` (the plan says "bundle spec v1.3";
`-spec3` is its consolidation with `-spec2`, so deltas below are against `-spec3`), graph note
`docs/plans/connected-code-graph-graph.md` §3.2, §3.4, §3.5, §6.2-§6.4 (the plan cites it as
`-p0-graph.md`, which does not exist), review `docs/reviews/connected-code-graph-p0.md` (S11, S12 Pass 1-2).

CPG: considered, not relevant -- the code under test is untracked-until-S11..S13 spike code that no `cpg_*` graph indexes; the slice's own `ccg_p0slice` graph is the subject, and I read and queried it directly.

## 1. Verdicts

| # | Criterion | Verdict | One-line evidence |
|---|---|---|---|
| SL-1 | AC-1 link and chain | **PASS** | one confirmed `http` Link `GET /shop/api/health`, 4 `LIFTED_VIA` hops in the required order, each hop's line holds the expected text at `defca60f` (§2.1) |
| SL-2 | Mount composition + counterfactual | **PASS** | 3 `/health` providers present; only the storefront one links; **real staged-copy re-extraction** with `API_PREFIX = "/x/api"` gives `GET /x/api/health`, 0 links, 11 unmatched (§2.2) |
| SL-3 | Load integrity | **PASS** | 10 of 10 constraints `OPERATIONAL`, 0 `FAILED`; 1 `:Snapshot`; `RENAME` done, no build key left (§2.3) |
| SL-4 | Impact answers | **PASS, with a design caveat on one clause** | contract mode: getHealth confirmed, distance 1, chain expanded; value mode from `buildUrl` lists `apiFetch` and its `endpoints.ts` callers; the callers are **not** reached "through path-matched `DATA`" (§2.4, §6.8) |
| SL-5 | Cost and density | **PASS on density; PASS on time under one reading of "projection stage", FAIL under the widest** | 0.196 / 0.024 MB per file (limit 0.7); `extract.sc` 0.99x / 1.10-1.19x parse; extract+normalize 4.1-4.2x parse for the Python unit (§2.5) |
| SL-6 | Determinism | **PASS** | bundles byte-identical modulo `builtAt`/`sandboxAuditIds` across two independent extractions; link facts and BFS answer facts equal (§2.6) |
| SL-7 | Schema accounting | **PASS (the accounting is complete); the deltas are for S15 to rule on** | 0 bundle-schema field deltas; 8 graph-side deltas and 5 spec-silent items listed; 0 `qname` mismatches; all bundles validate with the schema unmodified (§2.7) |

## 2. Evidence by criterion

### 2.0 What was run

| Run | Command | Result |
|---|---|---|
| Slice unit tests | `cd cpg && .venv/bin/python -m pytest -q spikes/vertical-slice/tests` | 254 passed, 11 deselected (live), 73.59 s |
| Live tests, S11 run dir `run1` | `... -m live spikes/vertical-slice/tests/test_live_slice.py` | 11 passed, 65 deselected, 43.31 s |
| Live tests, my fresh extraction `s14a` | same, with `CCG_SLICE_RUN=.../s14a` | 11 passed, 65 deselected, 47.66 s |
| Fresh full extraction (both units) | `cpg/spikes/vertical-slice/run_extract.sh --tag s14a` | both units `ok`; audit check ok (`refused=false, exitCode=0, egressMode=offline`, `extractScriptHash == sandbox.script.sha256`); fcs `qnameMatch=4686`, sp `qnameMatch=257`, all mismatch counts 0 |
| Counterfactual extraction (fcs only) | staged copy of `run_extract.sh` (scratchpad), tag `s14cf` | `ok`, `qnameMatch=4686`, 0 mismatches (§2.2) |
| Assembly | `assemble.py <fcs bundle> <sp bundle>` | `links=11 providers=61 consumers=11 unmatched=0` on `run1` and on `s14a` |
| Full cpg suite, after my one new file | `cd cpg && .venv/bin/python -m pytest -q` | **1313 passed** in 10.85 s (1313 at start) |

The live test rewrites `<run dir>/s13-measurements.json`. I re-ran it on `run1`, so that file is
now my re-run; it differs from the committed-era copy (saved before the re-run) in timings and
`instanceUsedMemoryDelta*` only. All counts, link ids, BFS answers and the snapshot id are equal.
Run dirs are gitignored: `cpg/.ccg-work/vertical-slice/{run1,run2,s14a,s14cf}`.

### 2.1 SL-1: AC-1 link and chain (PASS)

Graph query on `ccg_p0slice` (loaded from `s14a`): exactly one `Link {kind:'http'}` `ON` the
contract point `GET /shop/api/health`; `status` `confirmed`, `reason` `key-match`, `matcher` `exact`;
`TO` `falkor-chat/server/falkorchat/storefront_api.py::build_storefront_router.health` (the
`@router.get(` decorator at line 954 inside `build_storefront_router()`); consumer call site
`salesperson/src/api/endpoints.ts` line 17, in `getHealth`. Its `LIFTED_VIA` rows by `ord`:

| ord | node | file:line | urlPath | verbPath | text at `defca60f` (`git show`) |
|---|---|---|---|---|---|
| 0 | CallSite `apiFetch` | `endpoints.ts:17` | `a0.path` | `a0.method` | `return apiFetch<HealthResponse>({` |
| 1 | Symbol `client.ts::apiFetch` | `client.ts:107` (`viaLine` 108) | `p0.path` | `p0.method` | `export async function apiFetch<T>(...)`; 108: `const url = buildUrl(opts.path, opts.query);` |
| 2 | Symbol `client.ts::buildUrl` | `client.ts:73` | `p0` | none | `function buildUrl(path: string, query?: ...): string {` |
| 3 | CallSite `fetch` | `client.ts:118` | `a0` | `a1.method` | `response = await fetch(url, {` |

This is the order, parameters and verb paths SL-1 requires. The same chain is in the assembled
facts (`assemble.py` output) and in the live-test assertions (`test_sl4_...`, hop lines checked
against `git show`). Provider-side mount: `app.py:521` `app.include_router(storefront_api.build_storefront_router(shop), prefix=storefront_api.API_PREFIX)`
under the condition `shop is not None` (`app.py:516`).

### 2.2 SL-2: mount composition and the counterfactual (PASS)

**Three `/health` providers** (graph query `MATCH (s:Symbol)-[:PROVIDES]->(cp) WHERE cp.keyNorm ENDS WITH '/health'`):

| Provider | Key | Mount facts (assembler output) |
|---|---|---|
| `storefront_api.py::build_storefront_router.health` (line 954) | `GET /shop/api/health` | `mounted` via `app.py:521`, condition `shop is not None` |
| `api.py::build_router.health` (line 58) | `GET /health` | `mounted` twice: `app.py:485` under condition `dev_surface` (line 484), and `tests/test_input_bound_parity.py:538` with no condition |
| `app.py::create_app.health` (line 497) | `GET /health` | `how: app` (the `@app.get("/health")` registration) |

Only the first links; `GET /health` has no consumer, so its two providers have no link. Note the
mounts and their conditions live in the assembler facts only; the loaded graph's `PROVIDES` edges
carry no property (graph query: `keys(e)` is empty for all 61), so "dev_surface recorded" holds in the
facts, not in `ccg_p0slice` (§6.9).

**Counterfactual, as a real re-extraction (not the S12 record patch).** `run_extract.sh` stages with
`git archive` and offers no hook, so I ran a scratchpad copy (`.../scratchpad/cf/run_extract_cf.sh`,
identical to `run_extract.sh` except `HERE` pinned to the repo path and one inserted block after the
`git archive` line). The block asserts the line `API_PREFIX = "/shop/api"` occurs exactly once in the
staged `falkorchat/storefront_api.py` and rewrites it to `"/x/api"`. Its diff (written by the script):
`135c135 < API_PREFIX = "/shop/api"  --- > API_PREFIX = "/x/api"`, one line, nothing else. Then the
whole S11 path ran in the offline sandbox on that tree (`joern-parse`, content check, `extract.sc`,
`normalize.py`, verify; audit check ok; qname re-derivation 4686/4686 with 0 mismatches).

Bundle diff against the unmodified `s14a` fcs bundle, bodies only: 6 differing lines = 3 records changed,
each exactly what a one-constant edit must change: the `file` record of `storefront_api.py`
(`contentHash`), its `<module>` symbol (`fingerprint`), and the `constant` record
`storefront_api.py::API_PREFIX` (`values: [{const: "/x/api"}]`, was `"/shop/api"`).

Assembled with the unchanged `s14a` salesperson bundle: `links=0 providers=61 consumers=11 unmatched=11`.
The storefront `/health` provider key is **`GET /x/api/health`** (the other two stay `GET /health`); all
11 storefront provider keys are now under `/x/api` (11 keys contain `/x/api`, 0 contain `/shop/api`); the
SL-1 consumer `getHealth` is among the 11 unmatched consumers. The S12 record patch and this re-run
agree; the re-run additionally shows that extraction carried the changed constant through the
`{ref}` in `app.py`'s `kw.prefix`.

### 2.3 SL-3: load integrity (PASS)

`db.constraints()` on `ccg_p0slice`: `Link.linkId`, `ContractPoint.cpId`, `Name.nameKey`,
`CallSite.callSiteId`, `Cond.pointId`, `Ret.pointId`, `Param.pointId`, `Symbol.symbolId`,
`File.fileId`, `Unit.unitId`: all 10 `OPERATIONAL`, none `FAILED` (`Repo` carries an index, no
constraint, as `load.py` `CONSTRAINED` defines). `MATCH (s:Snapshot) RETURN count(s)` = 1
(`snapshotId baa84a25...`, equal in `run1` and `s14a`). `KEYS *` on the instance after the runs:
`ccg_p0slice` and `telemetry{ccg_p0slice}` (the latter is the engine's own, not mine); no
`ccg_p0slice__build_*` key remains. Load sequence per the live test: indexes, nodes, edges, constraints
(polled), snapshot, `RENAME`; `TIMEOUT_MAX` 0 so no query `timeout` was used (`query_timeout_used: false`);
largest batch 18,591 rows (<= 20,000, D16). Counts (second load, both runs equal): 38,121 nodes,
90,867 edges, 37 batches, 0 dropped. Load wall time, second load: 1.45 s (`run1` re-run) and 1.37 s (`s14a`); first load 1.68 s and 1.45 s.

### 2.4 SL-4: impact answers (PASS; one clause caveated)

*Contract mode*, from `GET /shop/api/health`'s `ContractPoint`: lists `salesperson/src/api/endpoints.ts::getHealth`,
`confirmed`, distance 1, 4 chain edges (`PROVIDES <-`, `TO <-`, `VIA <-`, `HAS_CALL <-`), and the rendered
chain expands the 4 `LIFTED_VIA` hops of §2.1. 11 contract walks (one per link key) each give 5 states,
peak frontier 1.

*Value mode*, from `buildUrl`: 21 items, max distance 4; `apiFetch` at distance 1; 12 of the 21 items are in
`endpoints.ts` (the live test asserts the 9 direct `apiFetch` callers there at distance 2 and `getHealth` among
them). The AC-1 rows of bundle spec §5.10 are present both in the bundle records and in the graph (live
test `test_sl4_ac1_data_rows_are_in_the_graph`: `fetch.a0 <- buildUrl.r`, `buildUrl.a0 <- apiFetch.p0.path`,
`fetch.a1.method <- apiFetch.p0.method`, `buildUrl` `ret <- p0`; each `confirmed/intra-flow`). The
path-match filter is exact against the bundle records (anchor `apiFetch.p0`, paths `p0`..`p1`: DATA fan-out
17 / 4 / 3 / 3 / 1 / 0 / 0 for `p0`, `p0.body`, `p0.method`, `p0.path`, `p0.query`, `p0.nothing`, `p1`).

**Caveat (SL-4's words "through path-matched `DATA`").** The chains of `apiFetch` and `getHealth` in
the `buildUrl` walk are `buildUrl -CALLS<- call buildUrl@108 -HAS_CALL<- apiFetch -CALLS<- call apiFetch@17 -HAS_CALL<- getHealth`:
no `DATA` edge. Path-matched `DATA` does run (it takes the `buildUrl` call's result to `fetch`'s `a0`, and
only there), but the item list and distances come from function-level continuation (§6.8). I score the
criterion PASS because every listed outcome (apiFetch and its `endpoints.ts` callers) is delivered; whether
"through path-matched DATA" is a required mechanism is S15's reading.

### 2.5 SL-5: cost and density

**File denominator.** The graph's `File` nodes are the bundle `file` records: **148** (107 fcs + 41 sp),
`Unit-HAS_FILE->File` = 148. They are not Joern `FILE` nodes (fcs has 108 incl. a placeholder `N/A`). S11's
count of non-empty sources: fcs 106 (+1 empty `.py`), sp **74**, of which `jssrc2cpg` never parses **33**
(re-measured in `s14a/sp/perfile.json`: `nonEmptySourceFiles 74`, `nonEmptyNotInCpg 33`, `fileNodesInCpg 41`).
Plan SL-5 says "per source file"; I report both.

| Unit | Graph MB (`GRAPH.MEMORY USAGE`, integer) | MB / File node | MB / non-empty source | `used_memory` delta | delta MiB / File node | delta MiB / non-empty source | nodes, edges | nodes / file, edges / file (File nodes) |
|---|---|---|---|---|---|---|---|---|
| fcs (`run1` re-run) | 21 | 0.1963 (/107) | 0.1981 (/106) | 32,171,256 B (30.68 MiB) | 0.2867 | 0.2894 | 36,579; 87,878 | 341.9; 821.3 |
| sp (`run1` re-run) | 1 | 0.0244 (/41) | 0.0135 (/74) | 3,321,512 B (3.17 MiB) | 0.0773 | 0.0428 | 1,471; 2,837 | 35.9; 69.2 |
| both, one graph | 22 | 0.1486 (/148) | 0.1222 (/180) | n/m | n/m | n/m | 38,121; 90,867 | 257.6; 614.0 (/148) |

(`s14a` re-run: `used_memory` delta 32,376,608 B fcs, 3,321,512 B sp; same MB and counts; the 0.6% fcs
spread is the repeat noise I saw.) Every cell is far below the 0.7 MB limit under either denominator and
under either memory figure. **Coarseness:** `total_graph_sz_mb` is an integer; the sp unit's single "1 MB"
could be anything the allocator rounds to 1, so sp's 0.0244 is a bound, not a measurement (the
`used_memory` delta, 0.0773, is the finer, noisier cross-check; it includes allocator and index overhead of
the whole instance, not only the graph). For scale, graph note §6.4 estimates 455 nodes, 2,400 edges and
0.3-0.35 MB per file for the v2 summary; the measured fcs density (342 nodes, 821 edges per file) is lower
in nodes and edges, with MB slightly below that range by the `used_memory` figure (0.287).

**Timing (K7: projection at most 2x Joern parse).** "Projection stage" is not defined in the plan; I report
three readings, wall seconds, `run1` (S11) / `s14a` (mine):

| Reading | fcs | sp |
|---|---|---|
| per-method projection alone (`projectionMs`) / parse | 0.366 s / 12.89 s = 0.028; 0.321 s / 12.48 s = 0.026 | 0.044 / 7.13 = 0.006; 0.043 / 6.57 = 0.007 |
| `extract.sc` whole script wall / parse | 12.74 / 12.89 = 0.99; 12.34 / 12.48 = 0.99 | 8.49 / 7.13 = 1.19; 7.20 / 6.57 = 1.10 |
| extract + host `normalize.py` / parse | (12.74+40.49) / 12.89 = **4.13**; (12.34+40.16) / 12.48 = **4.21** | (8.49+2.02) / 7.13 = 1.47; (7.20+1.65) / 6.57 = 1.35 |

SL-5 holds for the first two readings and fails for the third on the Python unit. The `normalize.py` stage
is dominated by `git blame` for `lastChanged*` (S11 notes: "normalize (host, incl. `git blame`)"; I did not
split it). The load into FalkorDB is 1.26 s (fcs) and 0.065 s (sp) = 0.098x and 0.009x of parse. Per-method
histogram (`s14a`): fcs 6,236 methods < 1 ms and 18 in 1-10 ms; sp 226 and 6. Single host, two runs, no
repeat statistics.

### 2.6 SL-6: determinism (PASS)

1. **Two full extractions** (`run1`, 2026-11-01 18:39Z, and `s14a`, 21:50Z; each re-ran parse, extract,
   normalize in the sandbox): `compare_bundles.py` -> `identical (85071 lines) apart from builtAt/sandboxAuditIds`
   and `identical (2775 lines) ...`; the bodies (everything after line 1) are also `cmp`-identical. The
   headers differ in exactly `builtAt` and `sandboxAuditIds` (checked key by key), the two §8.5 exempt fields.
   S11's own `run1` vs `run2` pair also compares identical (both units, re-run by me).
2. **Link sets:** `assemble.py` output on `run1` and on `s14a` is `cmp`-identical (11 links, same ids).
3. **BFS answer facts:** the live test writes all answers into `s13-measurements.json`; the `bfs`,
   `linkChains`, `sl4Contract`, `sl4Value`, node/edge counts per label and type, constraint list and
   `pathMatchFanOut` sections compare equal between `run1` and `s14a` (python `==`); the snapshot differs only
   in `builtAt`, which is the bundles' exempt `builtAt`; `snapshotId` and `manifestHash` are equal.
4. S13's own check (two loads of one bundle set into one instance): link, VIA, TO and `LIFTED_VIA` rows
   (11, 11, 47 rows) and 13 BFS answers identical.

### 2.7 SL-7: schema accounting

**Bundle schema (bundle-v1, `-spec3`): no field or record added, removed or retyped.**
`cpg/ccg/schemas/bundle-v1.schema.json` and `cpg/ccg/bundle.py` are unmodified in the tree; all bundles of this
report validated on write and on a separate `read_bundle` re-read (fcs 85,071 records, sp 2,775, `s14a`;
the counterfactual bundle likewise on write). `header.tools` has the seven keys `-spec3` §2.2 lists. 7 of
the 19 record types are emitted with count 0 (`configEntry`, `dependency`, `fieldAccess`, `launchFact`,
`schemaEntry`, `secretSighting`, `testMarker`; the sp unit also has 0 `decorator`).

**`qname` re-derivation (bundle spec §4.0): no mismatch, no SL-7 candidate from the grammar.** `verify_qnames.py`
on `s14a`: fcs 4686 source = 4686 bundle, all equal; sp 257 = 257; 0 span, 0 param mismatches; same 4686 on
the counterfactual tree. I also recomputed every symbol's `fingerprint` (§5.3) from `git show defca60f:<file>`
with code that shares nothing with `normalize.py` (`probe_s14.py`): 4686 and 257 symbols, 0 mismatches.

**Graph-side deltas (S13), against graph note §3.2/§6.2 and bundle spec §9 (S17 must adopt or rule on each):**

| # | Delta | Against | Observed |
|---|---|---|---|
| G1 | `LIFTED_VIA` carries `{ord, siteId, line, urlPath}` and, when present, `verbPath`, `viaSiteId`, `viaLine` (five props added beyond `{ord}` / spec §9 `{ord, siteId}`) | graph note §3.2/§6.3 `{ord}`; spec3 §9.2, BF6 `{ord, siteId}` | `keys(r)` on a `LIFTED_VIA` row: `ord, siteId, line, urlPath, verbPath, viaSiteId, viaLine`; 47 edges |
| G2 | `PROVIDES <-` is traversed in contract mode (CP -> provider) | §6.2 row `PROVIDES`: contract column `-` (only impact `->` for a provider anchor) | contract walks fail to start without it; `impact.py` docstring lists it |
| G3 | `VIA` carries `{status, reason, consumerKey}` | graph note §3.2 bare `VIA`; spec3 BF6 "per-site status stays on each `CONSUMES`" | 11 `VIA` edges with those 3 props |
| G4 | `CONSUMES {status, reason}` (11) and `ON` (11) are written; neither is read by the BFS | the S13 plan bullet lists only `VIA <-`, `TO <-`, `PROVIDES ->` as the rows the slice needs; both edges exist in graph note §3.2 | edge counts by type in `loads[1].edges` |
| G5 | `ownedRegions` stored as a flat int list (`[[a,b],..]` -> `[a,b,..]`) | spec3 §9.1: encoding is S17's ("for example a flattened list") | a unit test pins `_flatten_regions`; **0 of the 4,943 symbols have `ownedRegions`**, so the real data never exercised it (graph: `count(s.ownedRegions IS NOT NULL) = 0`) |
| G6 | 61 provider records -> 60 `ContractPoint`s; `GET /health` has two providers (`api.py::build_router.health`, `app.py::create_app.health`) | one `PROVIDES` per provider; `cpId` per (kind, keyNorm) | 61 `PROVIDES` edges, 60 CPs, one CP with 2 providers |
| G7 | the fcs `DATA` self-edge is kept: 1 `data` record with `use == def` (a call site, `fromPath r`, `toPath recv`, `confirmed/intra-flow`) | spec3 §5.10 says both points belong to one symbol, and nowhere forbids `use == def`; G-4 drops only the CDG self-edge | graph: `MATCH (a)-[d:DATA]->(a)` returns 1 row (`CallSite`, `r`, `recv`); the BFS visited set stops it |
| G8 | `PROVIDES`, `ON`, `TO` carry no property; provider `status`, `mounts` and their conditions are **not** in the graph | spec3 §9.1/§6.2 do not require them; SL-2 asks for "the `dev_surface` condition recorded" | `keys(e)` empty on all 61 `PROVIDES`, 11 `ON`, 11 `TO`; mounts exist only in the assembler facts |

**Spec-silent or spec-conflicting items from S11/S12** (quoted from `cpg/spikes/vertical-slice/S11-notes.md` and the review unless marked re-measured):

| # | Item |
|---|---|
| S1 | **Spec3 §4.1 rule 4 ("every code file has exactly one file-scope symbol") vs `jssrc2cpg`'s file selection.** Re-measured: 33 of 74 non-empty sp sources are absent from the bundle (23 by `*.test.ts(x)` only, 3 by test suffix plus `i18n`, **5 production files under `src/i18n/` by directory rule alone**, 2 `tests/e2e/*.spec.ts`); `jsRulePredictionMismatches 0`. fcs: 106 of 106 present. |
| S2 | `complexity = 1 + count(own cond records)`, not CFG McCabe (§5.3). S11 measured the gap against an `ast` count on 4,046 functions; not re-measured here. |
| S3 | An empty file (`tests/eval/__init__.py`): `lastChanged*` falls back to the path's last commit, and the `<module>` span is `1..1` while `lineCount` is 0. Spec §5.3 silent. |
| S4 | Expression-bodied arrow functions have no `RETURN` in `jssrc2cpg`; S11 synthesized 14 `ret` records (ordinal 0) at the function's start line. Spec §5.5 silent. |
| S5 | `header.tools` records no host Python version although Python spans and qnames depend on the host `ast` (S11 finding 16); not a schema delta (the seven keys are as specified). |

## 3. `Link` count, BFS state space and chain lengths (M4)

**`Link` count by status** (assembled facts and graph agree): 11 `confirmed`, 0 `possible`, 0 other; by
reason 11 `key-match`; by matcher 11 `exact`. Providers: 61 (59 `confirmed`, 2 `possible`). Consumers: 11, 0
unresolved, 0 unmatched. 1 provider alternative dropped (`unresolved-part`, a handler in
`tests/test_storefront_api.py`, line 2239).

**BFS** (all from `s13-measurements.json`; equal in the two runs): **peak frontier max 1,070** and
**field-path states max 52**, both on the anchor `Repository._graph` (callers=119) in value mode (9 levels,
2,704 states, 697 items); states max 2,704 over the 20 walks (11 contract, 6 value, 3 `apiFetch.p0`). Other walks:

| Walk | levels | peak frontier | states | field-path states | items |
|---|---|---|---|---|---|
| 11 contract walks (one per link key) | 5 | 1 | 5 | 0 | n/a |
| `apiFetch.p0` @ `p0` | 8 | 18 | 79 | 8 | 21 |
| `apiFetch.p0` @ `p0.path` | 8 | 18 | 65 | 3 | 21 |
| `apiFetch.p0` @ `p0.method` | 8 | 18 | 65 | 4 | 21 |
| value `buildUrl` | 9 | 18 | 64 | 0 | 21 |
| value `Repository.ensure_user` (130 callers) | 9 | 149 | 738 | 3 | 351 |
| value `WorkflowExecutor._run_agent_node` (84 callers) | 11 | 84 | 232 | 0 | 110 |
| value `tests/test_services.py::FakeRepo.__init__` (250) | 7 | 574 | 898 | 0 | 274 |
| value `tests/test_services.py::make_service` (253) | 5 | 506 | 797 | 0 | 265 |
| value `tests/test_tools.py::StubServices.__init__` (91) | 5 | 180 | 291 | 0 | 98 |

No walk reached the 64-level guard (the live test asserts every walk has < 64 levels) or the result-set limit
(the reader raises on it). "Field-path states" counts visited states whose path has a `.` or `[`.

**Hop counts of every `Link` chain.** `LIFTED_VIA` hops per link (each link has one consumer site; `BOUND_BY`
is not produced by the slice, so 0 `BOUND_BY` hops): **4 hops for 8 links, 5 hops for 3.** Shape of the
4-hop chains: `apiFetch` call (endpoints.ts) -> `apiFetch` -> `buildUrl` -> `fetch` call. The three 5-hop
chains add a wrapper, `participantGet` (call in `endpoints.ts`, then the `participantGet` symbol at `endpoints.ts:85`
and `apiFetch`, `buildUrl`, `fetch`):

| Link key | consumer | LIFTED_VIA hops | BFS contract-answer edges |
|---|---|---|---|
| `GET /shop/api/health` | `getHealth` | 4 | 4 |
| `POST /shop/api/messages` | `postMessage` | 4 | 4 |
| `POST /shop/api/reset` | `resetMine` | 4 | 4 |
| `POST /shop/api/session` | `joinSession` | 4 | 4 |
| `POST /shop/api/presenter/session` | `presenterSession` | 4 | 4 |
| `POST /shop/api/presenter/reset-all` | `presenterResetAll` | 4 | 4 |
| `GET /shop/api/presenter/participants` | `presenterParticipants` | 4 | 4 |
| `POST /shop/api/order/advance` | `advanceOrder` | 4 | 4 |
| `GET /shop/api/messages` | `getMessages` | 5 | 4 |
| `GET /shop/api/catalog` | `getCatalog` | 5 | 4 |
| `GET /shop/api/state` | `getState` | 5 | 4 |

(Source of the second column: `linkChains[].answerHops`; the BFS answer chain is always `PROVIDES <-`,
`TO <-`, `VIA <-`, `HAS_CALL <-`, independent of the lifted chain's length.) Prior for P1: chain length 4-5 on this repo.

## 4. Proving-repo anchors the slice touched (eval-protocol §8.2)

Applying §8.2's S14 row literally: "every anchor of a link or impact the slice computed or asserted" is
SL-1's contract point, SL-2's three `/health` providers' contract points, SL-4's two anchors, and the
SL-2 counterfactual's anchor; the wrapper and chain hops (`apiFetch`, `buildUrl` as a hop, the `fetch` call)
and the mount call in `app.py` are excluded. Locators in bundle spec §3.5 form (`repoId` `graphmind-ai-lab`,
pinned commit `defca60ff16534aa12a74e35856de9be98bfc710`):

| # | Source | Locator |
|---|---|---|
| 1 | SL-1, SL-2, SL-4 (provider contract point) | `{kind: "http", keyNorm: "GET /shop/api/health"}` (cpId `a359de51...`) |
| 2 | SL-2 (the dev-router and `app`-level providers share one contract point) | `{kind: "http", keyNorm: "GET /health"}` (cpId `aefbd69a...`) |
| 3 | SL-4 value mode | `{repoId: "graphmind-ai-lab", path: "salesperson/src/api/client.ts", qname: "salesperson/src/api/client.ts::buildUrl", line: 73}` |
| 4 | SL-2 counterfactual | `{kind: "http", keyNorm: "GET /x/api/health"}` (cpId `fe6fd542...`); the route does not exist at the commit (staged copy only) |

So the strict set is **4 locators over 3 contract points and 1 code element** (SL-2 names three
providers but they have two contract points). The providers themselves, as code elements, are
`storefront_api.py::build_storefront_router.health`, `api.py::build_router.health`, `app.py::create_app.health`
(SL-2's providers; §8.2 names their *contract points*; I list them as context, not as extra anchors).
Two items §8.2's table does not enumerate and S15/GC should rule on (open question Q1): (a) the
consumer endpoint `endpoints.ts::getHealth` and the other 10 links, which the slice computed (all 11 contract
keys were assembled and walked); (b) the S13 scale-probe walks, not asserted by any SL criterion: contract
walks from the other 10 link keys, and value walks from `Repository._graph`, `Repository.ensure_user`,
`WorkflowExecutor._run_agent_node` (all under `falkor-chat/server/falkorchat/`), and `FakeRepo.__init__`,
`make_service`, `StubServices.__init__` (under `falkor-chat/server/tests/`), plus `apiFetch.p0`. I excluded
them from the strict set because no SL criterion asserts them.

## 5. Bundle spec §10 [P-slice] items

"Exercised" means a real slice record or run touched it, shown by the figure given. All figures are from the
`s14a` bundles unless marked.

| [P-slice] item | Exercised? | Evidence |
|---|---|---|
| Python, TS/JS and shell `qname` grammar (§4.0-§4.2), re-derived | **Python, TS/JS yes; shell no** | 4686/4686 and 257/257 re-derived, 0 mismatches; the slice parses no shell |
| ... evaluation-scope rule for decorator lambdas (§4.1 rule 2) | yes | `tests/test_fusion.py::<anon>#1` at line 241 is in the bundle; 185 fcs and 47 sp `<anon>` symbols, 0 `#k` duplicates |
| ... `Protocol`/abstract stubs as symbols (rule 8; "five") | yes, count differs | my `ast` heuristic (Protocol/ABC class, `...`-only body) finds 7 in `falkorchat/`, all 7 have a symbol (one is the property `Tool.schema.<get>`); the spec says five; my definition may differ from the spec's, so treat the count as unreconciled |
| Ancestor-plus-line locator conversion (§3.5) | **no** | no converter exists in `cpg/`; only the `<anon>#1` qname above was derived |
| Vendored-path exclusion (§5.0), vacuously | yes, vacuously | headers list `vendoredDirs`; no file record under them |
| Rule 8's TS row for type-alias method signatures | **no** | not exercised as far as I know: spec §10 says the proving repo's TS has none, and I did not search the sources for one |
| Access-path grammar incl. `recv`, `kw.`, depth cap | yes | `recv` heads: 1,057 fcs `data` fromPaths; `kw.` in 9,469 fcs value sets; cap `fieldDepthCap 3`; `field-truncated`: 3 fcs + 1 sp `data` records |
| `param` receiver index `-1` (§5.4) | yes (Python) | 880 fcs params; 0 in sp (no implicit `this` param, by design) |
| `ret`/`cond` ordinals, definition of a condition (§5.5) | yes | ret 1,903 + 222, cond 3,816 + 218; `cond` = CDG sources (S2/S4 above) |
| `calleeName`, call-site ordinal ordering (§5.6-§5.7) | yes | callsite 18,129 + 462; every call site's line holds its callee name (`callsiteLineWithoutCalleeName=0`) |
| Value-set element forms and caps (§5.8), `constant` (§5.9) | forms yes; **caps no** | all 6 element forms appear in fcs (`const` 12,194, `unknown` 11,343, `ref` 2,751, `template` 475, `concat` 79, `importRef` 65); sp has 4 (`unknown` 214, `const` 178, `template` 13, `ref` 10); `constant` 184 + 5. **Every value set has exactly 1 alternative (max 1 in both units)**, and the only `unknown` reason is `point-derived`: the 16-alternative and depth-4 caps and `value-set-cap` are never reached, and no multi-alternative value set exists to drive S12's `partialAlternatives` path on real data. |
| `data`, `controlledBy` (§5.10-§5.11); AC-1 rows | yes | data 15,983 + 441; controlledBy 5,699 + 451; the four AC-1 rows present (§2.4) |
| `decorator` (§5.12), FastAPI path | yes | 213 fcs decorator records; 0 in sp (none there) |
| Symbol `fingerprint` normalization (§5.3) | yes | independent recomputation, 0 of 4,943 mismatches |
| File `kind`/`role` (§5.2) | **partly** | all 148 files are `kind code`, `role primary`; no `example`, `doc`, `config` or `script` file in either unit |
| `http` `keyNorm` (§6), SL-1, SL-2 counterfactual | yes | §2.1-§2.2. No slice key needed `{#}` or `{?}` or percent-decoding (0 keys with `{#}`, 0 with `{?}`; 29 keys with `{}`); step 8a ran on all keys and masked none |
| §8.5 exempt fields (SL-6) | yes | §2.6 |
| Header fields (§2.2) | yes | 15 header keys per bundle, validated; `flowProjected` true, `history` available |
| §1.13: `nameNorm` recomputation; step 8a on slice keys | yes / vacuous | the validator recomputes `nameNorm` on every write and read; 8a masked nothing |
| `ownedRegions` (§4.1 rule 4) | **no** | 0 symbols carry any (G5) |
| `fieldAccess` ([P-slice] "if the slice emits fields") | **no** | 0 emitted; AC-1 needs none |

## 6. Carry-forward findings

### 6.1 S12 finding 3: providers and mounts in `tests/`
Re-measured: 5 of 61 providers are defined under `tests/` (3 `confirmed`, 2 `possible`), and 44 providers
(all of them `api.py::build_router.*`) have a mount at `tests/test_input_bound_parity.py:538`. 0 providers
are mounted *only* from test files, so no `confirmed` provider rests on a test mount in this corpus. No bundle
field marks a file or call site as test: all 148 files have `role primary`, `testMarker` is 0. The distinction
survives only in the path. The S12 recommendation (an `isTest` flag on provider and mount records) is
unimplemented; P1 inherits the problem as S12 stated it.

### 6.2 S12 finding 4: a provider with an unresolved include prefix vanishes
1 provider alternative dropped as `unresolved-part` (a handler nested in
`tests/test_storefront_api.py::test_the_service_error_re_shaper_answers_only_on_shop_api._shop`, line 2239). It is listed in
`droppedProviderAlternatives`; no consumer could match it, no graph element exists for it. The spec has no
provider-side form for an unresolved part (spec3 §6 defines `{#}` for a screen failure and `{?}` for consumers
only). Unchanged; a `-spec4`/S17 item.

### 6.3 S12 finding 5: SL-2 counterfactual was a record patch
Closed by §2.2: a real staged-copy re-extraction through the offline sandbox.

### 6.4 S12 finding 7: link-level choices
Three choices the spec leaves open, each unexercised by data because the 11 links are uniform:
(a) `Link.keyNorm` is the provider's key, though a `possible` link's consumer key can differ; all 11 links are
`exact` so the two are equal; (b) `candidateCount` (§6.5); (c) a `fetch` seed is matched by bare
`calleeName == "fetch"` with no `resolution` check: the corpus has exactly 1 `fetch` call site
(`client.ts:118`, `resolution external`, 0 targets) and 0 in fcs.

### 6.5 The `candidateCount` per-key question (for S17)
By code (`assemble.py` `assemble()`): `candidateCount = len(cands)` where `cands` is the list of candidate
providers for the (site, consumer key) that *first creates* the link; later sites and keys never update it.
So it is neither per key nor per link: it is the candidate count of the first (site, key) pair. The
bundles cannot discriminate: all 11 links have `candidateCount 1`, one site, one key. S12's pinned behavior:
for a consumer `GET|POST /x` against providers `api_route /x` and `get /x`, link A reports 2 though its
POST key had one candidate. Recommendation to S17: define it as "providers matching this link's
`keyNorm` for this link's consumer symbol" (per link, order-independent) or drop it from `Link`; the graph
currently stores the first-pair value (`Link.candidateCount`, 11 rows, all 1).

### 6.6 N16a: the symbol-hop `lineStart` fallback
**Still unpinned.** `_symbol_hop` (`assemble.py`) takes the hop line from the parameter record at that index and falls
back to `sym["lineStart"]` when none exists. I mutated the fallback to `lineStart + 1` on a scratchpad copy
and ran all of `test_assemble.py` with the real bundles enabled: **162 passed**, the mutant survives. It
cannot be killed by the real corpus either: my probe finds 25 symbol hops across the 11 chains, 0 without a
param record (the fallback is never taken); 3 hops have a param line different from `lineStart` (those are the cases the real corpus does
exercise; I did not check which tests pin them). No test added by me (out of scope: no edits to code under `cpg/spikes` beyond the new
probe); the fix is one hand-built case with a `p<i>` path and no `param` record.

### 6.7 The `src/i18n` note
The sp bundle has no `file` or `symbol` for `src/i18n/` (graph and bundle: 0 file records under it): 8
source files exist there at `defca60f`, 5 production (`LanguageChooser.tsx`, `Provider.tsx`, `config.ts`,
`format.ts`, `useLocale.ts`), all excluded by `jssrc2cpg`'s directory rule `i18n` (S11 finding 3, reproduced in
`perfile.json`). Consequence, measured: the sp bundle holds call sites named after those exports
(`formatCurrency` 5, `useLocale` 4, `formatDate` 1) and all 10 have empty `targets`. None is on the AC-1 chain.
The question S11 left open ("is `src/i18n/` meant to be in the slice?") stands (Q3) and bears on spec3 §4.1
rule 4 (S1 above) and the per-file content rule: over the CPG's own files it holds (0 failures), over all
non-empty sources it fails for 33 of 74.

### 6.8 S13's design finding: `HAS_* <-` makes the enclosing function an item
**Re-verified.** With §6.2 as written, `HAS_PARAM/CALL/RETURN/COND <-` takes a reached point to its function
in `value` state, and a `value`-state `Symbol` continues `CALLS <-` and `TO <-`. Item sets therefore do not
depend on the access path of the anchor. Probe (`probe_s14.py`, live, anchor `apiFetch` `p0`):

| anchor state | items | states | field-path states |
|---|---|---|---|
| `p0` | 21 | 79 | 8 |
| `p0.path` | 21 | 65 | 3 |
| `p0.method` | 21 | 65 | 4 |
| `p0.query` | 21 | 63 | 1 |
| `p0.nothing` (matches no `DATA` edge) | 21 | 62 | 1 |

The item-set hash is identical across all five (`a0058110df7b`), and equals the value-mode `buildUrl` item
set (21). Path matching narrows the *states* (79 -> 62..65) and the `DATA` fan-out (17 -> 3), not the items.
Mechanism (read from `impact.py` and shown in the chains of §2.4): the anchor `Param` reaches its function by
`HAS_PARAM <-` regardless of path, and the interprocedural `arg(i)` rows of §6.2 (`CALLS ->` into a callee,
`HAS_PARAM ->`) are not implemented ("Not followed" list in `impact.py`), so the function-level `value` state is
the only carrier of impact past a call. A `DATA` hop that lands on a call's argument (`path a0`) cannot
continue to the call's result (path `r`) without those rows.

**What-if experiment (probe `Gate`, V2; run on the live graph, not a proposal of this report):** end the walk at
the function (state `end`) when `HAS_* <-` comes from a point that is not a `Ret`. Result: every anchor above
gives **1 item** (`apiFetch` itself), 5 states for `p0.path` and `p0.method`, and the value walk from `buildUrl`
still lists `apiFetch` but drops all 12 `endpoints.ts` items (`itemsInEndpointsTs 0`): SL-4's callers clause
would then fail. So gating continuation without the missing `arg(i)` rows loses recall wholesale.

**Recommendation for graph-dba / S17 (labelled recommendation; not decided here):**
(1) Do not gate `HAS_* <-` continuation on path in model v1 as an isolated change; V2 shows it empties SL-4.
(2) Either accept function-level continuation as the P0/P1 baseline and say so in §6.2 (state: "an item is a
function containing a reached point; path-narrowing reduces states and `DATA` fan-out, not items"), or specify
and implement the `arg(i)` rows (`CALLS ->`, `HAS_PARAM ->` index-matched, a call-result summary
`arg path -> r`) first and gate afterwards, re-measuring items per path.
(3) Whichever is chosen, make the answer carry which items were reached by a chain containing a `DATA` hop and
which only by function-level continuation, so the 21 are not all presented as "path-matched". In the `buildUrl` walk
every item is `confirmed` with 0 `possible` hops (asserted by the live test).

### 6.9 Further S17-relevant observations
- Provider `status`, mounts and mount conditions are not in the graph (G8): the information SL-2 reads from the facts
  would need properties on `PROVIDES` or `CONTROLLED_BY`-style edges for an operator query to render it.
- S12's reviewed fix for an unresolved alternative (`partialAlternatives`, a `possible/partial-key` cap) has no real-data case: every slice value set has one alternative (§5). It is covered by mini-bundle tests only.
- Test-file mounts raise confidence (6.1) and `possible` providers (2) are both test-defined.
- No loader loss: `DATA` has 16,424 edges in the graph for 16,424 `data` records (15,983 + 441); `CONTROLLED_BY` 6,150
  for 6,150 records (5,699 + 451): the loader dropped none (`dropped: {}`).

## 7. Open questions for S15

1. **Q1 (anchors).** Does "touched" for contact set C (§8.2) cover only the 4 locators of §4, or also the 11 link consumers and S13's scale-probe anchors? GC needs the answer before S47.
2. **Q2 (SL-5 reading).** What is the "projection stage" in K7: per-method projection (0.03x), `extract.sc` (0.99-1.19x), or extract + host normalize (4.1-4.2x on Python)? The criterion is met on the first two and failed on the third. If the third counts, `lastChanged*` via `git blame` is the cost to attack.
3. **Q3 (`src/i18n`, rule 4).** Is `src/i18n/` in the slice? Does spec3 §4.1 rule 4 get a frontend-exclusion carve-out (spec states which files may be absent), or does the emitter synthesize `file` + `<module>` for skipped sources? Does the D15 per-file rule for JS run over CPG files only?
4. **Q4 (SL-4 wording).** Is "through path-matched `DATA`" a required mechanism for the value-mode callers (it is not what produces them, §6.8), or only the filter on the `DATA` hop?
5. **Q5 (§6.2 amendments).** Rule on G1-G4 and G8 as model-v1 deltas; add `PROVIDES <-` to the contract column; decide where provider mounts and conditions live.
6. **Q6 (self-edge).** Is a `DATA` self-edge (`use == def`, different paths, G7) legal in spec3 §5.10, kept in the graph, or dropped like the CDG self-edge?
7. **Q7 (G5).** `ownedRegions` is unexercised by data: S17's anchor rule (narrowest containing interval) has no real case in the slice; first test is a P1 fixture.
8. **Q8 (`candidateCount`, N16a, S12 findings 3/4/7)** as in §6: take into S17/`-spec4` or defer.
9. **Q9 (SL-2 wording).** "The bare re-registration" is read here as the `@app.get("/health")` in `create_app`; confirm.
10. **Q10 (Protocol stub count).** Spec §10 says five stubs; my heuristic finds seven, all with symbols. Reconcile the count or the definition in a spec edit.

## 8. New and touched files

- New: `cpg/spikes/vertical-slice/probe_s14.py` (read-only probe: file denominators, DATA self-edges, test mounts, symbol-hop fallback use, `src/i18n`, [P-slice] counts, independent fingerprint check, link chains, BFS item sets and the V2 what-if). Run: `cd cpg && .venv/bin/python spikes/vertical-slice/probe_s14.py [RUN_DIR] [--no-db]`.
- New: this report.
- Not committed or kept in the repo: the counterfactual script `.../scratchpad/cf/run_extract_cf.sh`, the N16a mutant copy, and the scratch helpers (`sl13.py`, `chains.py`, `q2.py`) under the session scratchpad. Gitignored run data: `cpg/.ccg-work/vertical-slice/{s14a,s14cf}` and the re-run `run1/s13-measurements.json`.
- Reservation R-B: only `ccg_p0slice` (and its build keys, created and removed by the loader) was written; `falkordb-dev` and port 6379 were not touched; the credential file was read, not printed.
