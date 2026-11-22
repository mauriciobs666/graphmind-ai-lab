# Connected code graph — P1 plan, part 3: answer engine, clients, plain language (W0 spec, W3)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1)

Revised 2026-11-19: V-both ships (P1-23, P1-39 rows) and the flow notice is built by P1-45b, P1-23d, P1-39d and P1-24b, in that order (`-answer-spec2.md` §5, §5.1); `docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md` is the decision's source.

Part of `docs/plans/connected-code-graph-p1.md` (decisions D1–D11, the flow gate §5, step index §7). Contracts:
answers `docs/plans/connected-code-graph-answer-spec.md` (canonical answer v1, `ccg/answer.py` is its validator),
traversal `-p0-graph.md` §3 (`bfs-v1`), graph `-graph.md` §6.2, evaluation `-eval-protocol-ml.md` §10. The slice's
walk is `cpg/spikes/vertical-slice/impact.py` (`walk`, `Reader`, `CypherReader`, `Item`, `render_item`), ported
under equivalence tests and then extended. Engine code is Python 3.12 under `cpg/ccg/engine/`; every answer
passes `ccg.answer` validation before it leaves the engine. Briefing line: dev items and fixtures only;
no writes to any shared database.

## P1-00 · answer-spec successor `-answer-spec2.md` — `architect`

`Extends:` `-answer-spec.md` (executed against by S34, so a successor with an ordinal, not an in-place edit);
the earlier gains `Extended by:`. It states **only** the changes:
- the stale text the ledger flagged at S50 (the engine reads/hop wording that predates graph model v1);
- the engine's **`cacheBypass` input** (EP §10.4, recorded in provenance as `cacheBypass`), citing the predicate
  rather than restating it (N-3);
- hop kinds: for each of `DEFINED_BY`, `LAUNCHES`, `DEPENDS_ON`, `FROM_DEP`, either the hop kind and its
  fields or an explicit prohibition (EP §5.2.1's four class-X edge types are carried by no P1 answer hop; `LAUNCHES`
  is the `.mcp.json` launch chain and is needed in P1, the others are P3 and are forbidden until then);
- the `algorithm.version` value and bump rule for P1's rows (C6): P1 ships `bfs-v1` at the version this document
  names (`1.1`, then `1.2` for V-both; `-answer-spec2.md` §1.2) and retires `function-level-continuation` with the `arg(i)`/`Ret` rows
  (`ENGINE_REASON_CODES` then drops it);
- the depth-bound default (index D3) and `unexplored` counting (C4);
- §5 of the index: how a `flow-unavailable:<lang>` item is built, so P1-23 has one definition.
**Gate:** `analyst`. The code and schema edits it requires are P1-45's.

## P1-45 · answer validator and schema edits — `tdd-engineer`

- **Files:** `cpg/ccg/answer.py`, `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/tests/test_answer.py`, `cpg/tests/golden/answer-ac1.json` (only if a golden changes).
- **Behaviours, failing validator tests first (as P0 did; the pin and the field are not authored in one step),
  in the order and with the bases of `-answer-spec2.md` §8:** (1) schema 1.1 (`"1.0"` still accepted) and
  `Provenance.cacheBypass` (`const: true`, requiring `cacheHit: false`), every other unknown key still rejected;
  (2) `Bound` with exactly one of `keyPath`/`launch`, the port evidence (`port`, `portOrigin`, `packId`; no `siteId`, `bound[]` is per site by the read, spec2 §3.2); the `Unit` hop kind, `TO_UNIT` and `UnitSubject` (spec2 §3.5); V5's wiring, depends-on, per-site and
  `Unit` clauses, V9's `not-mapped` check, and the 1.1 `facts_bytes` projection (spec2 §3.2, §3.3, §3.5, §4.3);
  V17 (version labels, the 1.1 bound), V18 (spec2 §5's closed row list, entries and exits) and V19 (`DEFINED_BY`,
  `LAUNCHES`, `DEPENDS_ON`, `FROM_DEP` on no 1.1 hop); the expected-value tables named in spec2 §8 (`REQUIRED`,
  the `$defs` set, `ENUMS`) are the only existing tests that change; (3) the validator accepts `algorithm.version` `"1.1"` (a characterization pin; the test that a row added
  without a bump fails is P1-22's); (4) `ENGINE_REASON_CODES` is unchanged until P1-22 removes
  `function-level-continuation` (one-line edit and its test belong to P1-22); (5) the stored golden still
  validates and is unchanged; `reasons.yaml` is not edited. **Needs:** 00. **Blocks:** P1-21, P1-22, P1-24, P1-26. **Gate:** `analyst`.

## P1-21 · reader layer and anchors — `tdd-engineer`

- **Files:** `cpg/ccg/engine/__init__.py`, `config.py` (`DEFAULT_DEPTH_BOUND = 2`, `LEVEL_GUARD = 64`,
  `CHUNK = 1000`, `RESULTSET_SIZE = 10000`), `reader.py` (`Reader` protocol, `CypherReader` bound to the
  `ccg_engine_<s>` credential (`ccg.sysname.acl_user('engine', s)`), recording every query text for the recipe), `anchors.py`, `cpg/tests/test_engine_reader.py`,
  `cpg/tests/test_engine_anchors.py`.
- **Behaviours:** (1) the reader returns for `Param`/`Ret`/`Cond` their `path`, `repo` and `unit` (the slice's
  reads returned only `pointId`, `line`: C3), for a `LIFTED_VIA` expansion its `unit`, `repo` and `siteId`, for
  an edge its `reason`, and, per link, `bound_of_link` (`-p1-graph.md` §5, `graph-dba`'s addition: each
  `configEntry`-origin `BOUND_BY` target's `siteId`, `ord` and port evidence, ordered for a first-row pick); each query is parameterized, read-only and named by the recipe pattern
  `^[a-z_]+(:[A-Za-z]+)?$`; ids chunk at ≤ 1,000 and a full reply halves the chunk; a reply reaching
  `RESULTSET_SIZE` raises, never truncates; (2) **code-location anchor:** `resolve_location(reader, repo, path,
  line=None, symbol=None)` runs the three repo-scoped reads in order — `File` → mapped; else `SkippedFile` → "not
  mapped" with its reason; else "not found" (`.d.ts`, `.pyi` and vendored paths answer "not found") — and for a
  file+line fetches **every** symbol of the file via the `File.path` index and `DEFINES` and picks the narrowest
  containing interval, `ownedRegions` included, ties by `depth`; (3) **contract-point anchor:** by `keyNorm`,
  with prefix and template matching client-side through `ccg.linker.matchers` (one implementation); (4) a
  symbol anchor carries `qname` (EP §15.2 Q1); (5) an ambiguous anchor returns `options`, never a guess;
  (6) an unknown repo or a name outside the whitelist is a clean error before any query; (7) the three reads of
  carry-over C3 are enough to derive an item set from a raw recipe re-run (AC-16's raw leg).
- **Done / Gate:** reads exercised on `ccg-falkordb` with the P1-19 loader on a small fixture graph; `analyst`;
  `graph-dba` reviews the query texts (C3).

## P1-22 · BFS rows, guards, depth bound — `tdd-engineer`

- **Files:** `cpg/ccg/engine/bfs.py` (the walk), `cpg/ccg/answer.py` (remove the `function-level-continuation` entry of `ENGINE_REASON_CODES` and its test in `cpg/tests/test_answer.py`), `cpg/ccg/engine/state.py` (`value`, `path(P)`, `contract`,
  `key`, `end`, `site(siteId)` and `function`, kept for `-answer-spec2.md` §5's flag-false rows), `cpg/tests/test_engine_bfs.py`,
  `cpg/tests/test_engine_bfs_guards.py`, `cpg/tests/slice/test_engine_equivalence.py`.
- **Behaviours (ordered):** (1) equivalence first: the slice walks (anchor `p0`, `p0.path`, `p0.method`, `p0.query`,
  `p0.nothing`, value walk from `buildUrl`) reproduce the slice's item sets under the slice's recipe; (2) the
  rows of graph note §6.2 as written, plus `PROVIDES <-` in the contract column (distance 0), and contract mode
  reaching consumers by **`CONSUMES <-`** with the `Link` attached through `ON <-`/`VIA`, so a consumer with no
  provider is found and reported `unmatched` (the slice walked `PROVIDES <-`, `TO <-`, `VIA <-` and never read
  `CONSUMES`/`ON`): a fixture graph where a contract walk from a provider-less contract point lists its consumers;
  **not walked in P1** (the module docstring and `cpg/README.md` list them): depends-on `BOUND_BY ->` (a `Link` is
  left only by `TO ->`/`TO_UNIT ->`, spec2 §3.3 V5), `CONSUMES ->` (no item kind carries it), `VIA ->` in `arg(i)`
  state (needs `paramMap`, not stored, `-p1-graph.md` G12), and from a `Param`, `←` then `CALLS ←` (P5's RCA);
  a **`ConfigEntry` anchor** (impact) reaches only the sites it binds: `BOUND_BY ←` enters the `Link` in a
  `site(siteId)` state carrying the `siteId` that `bound_in` returns (`-p1-graph.md` G7 and §5, which supersede
  graph note §6.2's "the links it binds → their consumers"), `VIA ←` from that state lists only that site's
  `CallSite`, an entry bound at two sites of one link lists exactly those two consumers, and a `bound_in` row with
  no `siteId` is a malformed row; a `Link` is never an anchor (`WalkError` before any read);
  (3) the **`CONTROLLED_BY` row** with a visited set seeded with the anchor keyed `(label, id, state kind, state
  path)`, keeping a state's arrivals that no other arrival dominates on `(possible, distance)` and expanding the state once per
  kept arrival (`-answer-spec2.md` §4.1): a `cond → cond` cycle terminates (C1, F-12; a mutant that expands an
  equal arrival, `<` replaced by `<=`, is killed), a two-point `DATA` cycle and a `DATA` self-edge
  terminate, a chain deeper than 64 levels raises; (4) the **`arg(i)`/`Ret` rows** (`CALLS ->` into a callee,
  index-matched `HAS_PARAM ->`, a `Ret`-reached callee summary onto the call site's `r`; receiver as index `-1`,
  `kw.<n>` by `Param.name`; **the `Ret` summary is context-insensitive**: a function reached through a `Ret` lists every
  caller of it as a resolved `CALLS ←` hop, not only the call site the change entered by. Accepted and documented for
  P1 (a test pins it): no P1 acceptance criterion measures it, because every P1 language's
  `flowAvailable` is false (index §5, D2), so these rows are not followed in any measured unit, and flag-true units
  exist only in fixtures. A flag that flips true by a decision record (`ccg-p1-flow-python-v2-decision`) first needs
  a call-string version of the row, a table change and a version bump, before that population is gated): in a flag-true unit items are then precise, no hop enters the `function` state, and a
  caller of a function whose return the change does not reach is no longer listed through it; each of the six slice walks
  is pinned as a named delta against the slice's 21-item set (`-p0-graph.md` §3.3): the items dropped and the items
  added, each with its reason, an empty delta being a legal value and no walk special-cased. Callees the change
  enters through `CALLS ->`/`HAS_PARAM ->`, which the slice never walked, are added, so a walk's item and state counts may grow (the anchor `p0` walk: 21 → 22
  items, 98 → 157 states, accepted). The V2-gated counts (19, 5, 5, 3, 2, 4) are state counts of the S17-L2 gate
  model, a prior; the engine is not required to approach them, and the equivalence test reproduces them only from
  that gate's own table. `apiFetch` stays `confirmed` (a resolved `CALLS ←` from the anchor `Symbol` is not a labelled row,
  `-answer-spec2.md` §5). **Every fixture sets `Unit.flowAvailable` explicitly** (`true` for these rows), so
  P1-23's flag gate changes no pin here; the `function` state and its labelled rows in flag-false units are
  P1-23's; (5) **depth bound:** `DEFAULT_DEPTH_BOUND = 2` always applies (the
  1.1 request has no bound field, `-answer-spec2.md` §4.1); items at distance ≤ bound as `-answer-spec2.md` §4.1 defines (everything reachable within it, not only the
  lexicographically best path), every edge the bound cuts is counted by §4.2 into
  `depthBound.unexplored` (a cut edge is counted once per edge, not per path) and `unexplored > 0` implies `bound`;
  a distance-2 `flow`-shaped item is reached by default (the reason the bound is 2); (6) `Name` is one compound
  hop carrying the `DISMISSED` anti-join (no verdict log in P1: the join returns empty) and never a BFS node;
  (7) best chain per item among the chains within the bound: fewest `possible` hops, then distance, then length, then
  fewer hops whose own edge is `possible`, then stable id (`-answer-spec2.md` §1.2, §5); every item's chain
  recomputes to the item's own cost (possible hops, distance, length) through the chain's own arrivals; (8) `algorithm.version`
  per `-answer-spec2.md` §1.2: the walk executes its row table as a data record (so P1-23's candidate walks are table variants, and only the shipped
  one is pinned), `ALGORITHM_VERSIONS` maps
  each version to its record's pinned fingerprint (`1.1`: `5cf047c14e43da44d0363707d3036080` in `ccg/engine/bfs.py`, published by `ccg/engine/answer_build.py`; `1.2`: the V-both record, P1-23d), and a test fails if a
  row is added without changing it.
- **Done / Gate:** `analyst`; mutation check on the arrival-dominance comparison, the bound and the `unexplored` count,
  and on every row of the table × each field (a field no code reads is removed from the table, not pinned).

## P1-23 · flow fallback, distance and grouping, reverse question — `tdd-engineer`

- **Files:** `cpg/ccg/engine/flow.py`, `cpg/ccg/engine/group.py`, `cpg/tests/test_engine_flow.py`,
  `cpg/tests/test_engine_group.py`; for behaviour (0) `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/ccg/answer.py`, `cpg/tests/test_answer.py`.
- **Behaviours:** (0) **the `site` hop state, first, failing tests first** (`-answer-spec2.md` §3.6): `Hop.state` gains `site`
  (schema, and the `ENUMS` row `("Hop", "state")` of `test_answer.py`); V5 gains the `site` shape clause and V17 the 1.0 rejection, with
  the rejections of spec2 §3.6 (a `site` hop that is not a `Link`; a `Link` entered by `BOUND_BY <-` from a `ConfigEntry` in another state;
  a `site` `Link` followed by a wiring `ConfigEntry`; `site` on a 1.0 document) and an acceptance of a config-anchor chain
  (`ConfigEntry` anchor, `Link` `site`, its `CallSite`). `ccg/render/` needs no change; (1) **flow fallback (index §5), as `-answer-spec2.md` §5 defines it:** from `Unit.flowAvailable`
  (assembler, D2): inside a unit with the flag true `DATA` and `CONTROLLED_BY` rows are followed `confirmed`;
  inside a unit with the flag false `DATA`, `CONTROLLED_BY` and the `arg(i)`/`Ret` rows are not followed, and only
  the closed list of labelled rows (continuations leaving the `function` state) is `possible`, with
  `flow-unavailable:<lang>` when the hop's own edge is confirmed; resolved edges with no labelled row on the chain
  stay `confirmed`; **the labelled set is the closed row table of `-answer-spec2.md` §5**: the rows continuing from a
  `function`-state hop, in both columns (a direct resolved caller of an anchor `Symbol` is not labelled), plus the
  added-row hops of a candidate walk (`-p1-ml.md` §6.6); **candidate walks:** besides the as-built table, P1-23
  built V-flow, V-both and V-union (`-p1-ml.md` §3.5) as row-table variants chosen by an engine-internal parameter
  (never a request field), each added hop `possible`/`flow-unavailable:<lang>`, one test per table; **V-both ships**
  (P1-39's pre-registered rule, `docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md` §2), as `-answer-spec2.md` §5's callee run at `algorithm.version` `1.2`, and
  P1-23d adds the table and its `1.2` pin and P1-39d makes it the default; the caller-only table (`1.1`, `flow.candidate_table("as-built")`), V-flow and V-union stay as non-shipped candidate tables, one test per table; a test plants one caller reached through a resolved and a name-matched site, **the name-matched
  site sorting first by stable key**, and expects the resolved chain (the mutant it kills: the own-possible
  tie-break dropped; P1-22 computes one `possible` variable for the hop's labelled status and its own edge's status, and
  P1-23 splits them where the first labelled row lands, so that test can fail); P1-22 already built the wiring row,
  `TO_UNIT ->` and the depends-on column, so P1-23 inherits and extends their tests and rebuilds nothing; **pinned at the P1-22 gate** (`docs/reviews/connected-code-graph-p1-22.md` N2): the equal-cost chain tie-break by the predecessor's stable key (spec2 §1.2) has a test planting two equal-cost arrivals whose reader and insertion order is opposite to their key order, under the shuffled-reply harness, so the mutant that drops the sort of a key's candidates fails; and the own/hop split above makes the own-possible test one that can fail, run at the first labelled row; **P1-23's gate re-runs V18 over real walk output** (P1-45's carry), not only planted documents; a missing `candidate_count` row (`-p1-graph.md` §5: absent `consumerKey`, or a site with no owner) reads as "N unknown", never `0` or `1`, with one live test for the absent-key row; **wiring items** (spec2 §3.3): at a `Link` reached by `TO ←`, each `configEntry`-origin
  `BOUND_BY` target is a `config` item (`BOUND_BY ->`, distance +1, the link's status) listed once, its single `Bound`
  taken from the entry's lowest-`(siteId, ord)` edge as `bound_of_link` returns it (spec2 §3.3); the **depends-on rules**:
  a walk row ending on a `launchFact`-origin entry yields no item, and from a `Link` with no `TO` the
  `TO_UNIT ->` row reaches a `Unit` hop, listed as a `unit` item, `possible`/`target-not-mapped`, with the unit
  `not-mapped` in `staleness.units` (spec2 §3.5; AC-9b's crossing case); (2) distance counts inter-function hops
  only (call, link, shared state, config); steps inside a function count 0; direct = distance 1; (3) items are
  grouped repository → component → distance, `confirmed` before `possible`, deterministic order, folded groups
  enumerate every member with its own chain; resolved calls' other candidates stay as `possible`/
  `resolution-unverified`; `group.py` folds an item only when its **own arrival** is the candidate hop (`-answer-spec2.md` §6, the §5.4 row;
  items past a candidate stay listed), into one group per resolved site, and only for a language in the `fold_languages` set it is given
  (`fold_languages(measured)`: `-p0-ml.md` §8.3's rule, exact 19/20, at least 60 resolved sites, every configuration measured; a language not in
  `measured` is listed; keys are `Unit.language` values of the site's unit, TS/JS being `"javascript"`); P1-39 (5) supplies the measured pairs;
  nothing is dropped, and the renderers list every group member with its own chain (spec §6), receiving no precision figure; the sort key is
  spec v1 §6's `items` row exactly, with no trailing kind slot (V8 makes ids unique across `items` and members); (4) the **reverse question** ("depends on") is the depends-on column with the same marking and grouping.
- **Gate:** `analyst`. **Needs:** 22, 18.

## P1-45b · schema 1.2, `notices`, V20, the 1.1-or-later helper — `tdd-engineer`

`-answer-spec2.md` §1.1, §5.1, §7, §8.1. **Needs:** 45. **Blocks:** P1-23d, P1-39d, P1-24b. **Gate:** `analyst`. First of the four: it changes no document that `"1.0"` or `"1.1"` accepts or rejects today.

- **Files:** `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/ccg/answer.py` (`ANSWER_SCHEMA_VERSIONS`, `ARRAY_RULES`, `canonicalize`, `facts_bytes`, `_check_labels`, `_check_flow_rows`, `_check_hops`'s flag, the V19 check, new `_check_notices`, `validate`, and the docstrings/messages that say "1.1" for "1.1 or later"), `cpg/tests/test_answer.py`.
- **Behaviours:** the schema, V17 and V20 of §5.1/§7; one 1.1-or-later helper replaces every `== "1.1"`; the golden `answer-ac1.json` stays `"1.0"` and unchanged. Tests in the order of §8.1's P1-45b list, test-first.
- **Done at its commit:** the builder still emits `"1.1"` and every existing engine test is green unchanged; every `"1.0"` and `"1.1"` base validates as before; `"1.2"` bases validate and V20 bites; the §8.1 mutants are killed.

## P1-23d · the V-both table: constant, `1.2` pin, V18's callee run — `tdd-engineer`

Stakeholder decision 2026-11-19 (`-answer-spec2.md` §5): V-both ships. **Needs:** 23, 39, 45b. **Blocks:** P1-39d, P1-41, P1-42, P1-37. **Gate:** `analyst`. It does not flip the default walk or the builder (P1-39d does), so no commit emits or accepts a `"1.1"` V-both document.

- **Files:** `cpg/ccg/engine/bfs.py` (`ROWS`, new V-both constant and `SHIPPED_ROWS`, `definition`, `ALGORITHM_VERSIONS`), `cpg/ccg/engine/flow.py` (`candidate_table`, `CANDIDATES`), `cpg/ccg/answer.py` (new `CALLEE_ROWS`, `_check_flow_rows`),
  `cpg/tests/test_engine_flow.py`, `test_engine_bfs_guards.py`, `test_answer.py`, `test_engine_superset_dev.py`.
- **Behaviours:** (1) `bfs.ROWS` stays the caller-only table (published `1.1`); the four V-both rows move out of `flow.candidate_table` into a `bfs` constant and `SHIPPED_ROWS` is `ROWS` plus them; `candidate_table("V-both")` returns `SHIPPED_ROWS`, the other three names keep their meaning
  (`candidate_table("as-built") == bfs.ROWS`, its record's `algorithm_version` is still `"1.1"`); `walk(table=None)`, `definition()` with no argument and `build_answer` are **unchanged**; (2) `ALGORITHM_VERSIONS` gains `"1.2"`, the fingerprint of the record of `SHIPPED_ROWS`, pinned as a
  literal computed by the implementer, `"1.1"` untouched; (3) V18 per `-answer-spec2.md` §5 and §8.1, **`"1.2"` only**: `CALLEE_ROWS` and clauses (b)–(d) in a run, (f), (g); `FLOW_ROWS` and `"1.1"` behaviour unchanged, so a `"1.1"` document with a `HAS_CALL →` exit stays rejected by (e);
  (4) the live measuring test `test_engine_superset_dev.py` and `cpg/eval/superset_dev.py` refuse once a pinned engine file differs, so only `test_the_real_phase` is marked run-at-the-pinned-engine history (skipped with that reason; the committed result is the record; `test_the_digest_and_the_candidate_measurement_run_over_the_store` keeps running), and a re-run gets its own `v2` record; every assertion that takes the default table as its baseline is re-derived in the unit that flips it (P1-39d), and `test_a_candidate_walk_has_no_algorithm_version_only_the_shipped_table_has` (`test_engine_flow.py`) is re-derived here, since `candidate_table("V-both")` now returns the shipped table; (5) one test per table stays in `test_engine_flow.py`.
- **Done at its commit:** suite green; `"1.1"` documents are accepted and rejected exactly as before (the relabelled `B-both` is rejected); the `"1.2"` `B-both` fixture validates; the fingerprint of `SHIPPED_ROWS` maps to `"1.2"`; the analyst gate's mutation check covers every added row × field, V18's new clauses and the `1.2` pin.
  P1-41's expected P1 forms are re-derived once P1-39d lands (the callee items are new `possible` items at distance ≤ 2 in a flag-false answer).

## P1-39 · staleness, stamp, answer builder, superset measurement — `tdd-engineer`

- **Files:** `cpg/ccg/engine/answer_build.py` (assembles the canonical answer and runs `ccg.answer` validation),
  `cpg/ccg/engine/staleness.py` (**imports** `ccg.staleness` from P1-20; no second implementation),
  `cpg/tests/test_engine_answer.py`, `cpg/tests/test_engine_superset_dev.py` (marked `live`).
- **Behaviours:** (0) the builder maps a walk state to the hop's `state` by its kind, `site` included (`-answer-spec2.md` §3.6), adding `accessPath` for `path` only; (1) staleness: each unit on any chain with its built commit and time, and whether it is stale-kept
  or not mapped with the reason (FR-7, FR-7a; AC-9 code-snapshot stamp from `:Snapshot`); (2) the stamp carries
  `verdictLogOffset` 0 and the empty-log hash until P4; (3) `facts_bytes` is byte-identical for the same request run
  twice and across rebuilds from the same commits; (4) the **flow-fallback measurement, on development flow items only**
  (`cpg/eval/dev/ac5-flow.jsonl`), on the snapshot built at `EVAL_COMMIT` (where the dev items' locators were
  validated): the share of those items that each of the four walks of `-p1-ml.md` §3.5
  (as-built, V-flow, V-both, V-union) lists **at any status** (`possible` included; stated in advance, so a call edge that
  `-p1-extraction.md` 5d demotes from `confirmed` to `possible`/`name-match` does not change what "listed" counts), by language and signature, with the unlisted items' ids and the burden
  per anchor; the rule there picked V-both (`safeClaim` false: 9 of 28 listed, no walk at n/n;
  `cpg/eval/results/ccg-p1-flow-fallback-v1.json`; reading in `docs/plans/connected-code-graph-p1-flow-fallback-result-ml.md`), which P1-23d and P1-39d ship at `algorithm.version` `1.2`
  (`-answer-spec2.md` §1.2); the measurement is done and nothing in this unit waits on a pick; the `ccg-p1-flow-fallback-v1` record was committed before it ran; it runs on a throwaway instance (index §8); (5) **the fold's inputs** (P1-23 (3)): the builder
  passes `group.listing`/`group_items` the set `group.fold_languages(measured)`, where `measured` maps a `Unit.language` value to its
  `(correct, resolved sites)` pairs, one per configuration, from the **headline** table of the P0 fitness report
  (`docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md` §a1.2, "P off / on"; the override runs of §a1.3 are secondary and
  not read), transcribed as a committed constant with its citation; TS/JS is keyed `"javascript"`. For a language P1-15 adopted an indexer for,
  the pairs are P1-15's M2 results on the merged output (`-p1-ml.md` §2.3; the record `ccg-p1-indexer-v1` holds only the pre-registered
  criteria, not these figures) and replace the P0 pairs; a language P1-15 did not adopt, did not run or ended "not adopted" for keeps its
  P0 pairs. A language with no pairs (Kotlin) is absent from `measured`, hence listed, never folded. A test asserts that the derived set
  equals the report's fold/list column (today `javascript` and `go` fold; `python`, `java`, `csharp`, `cpp`, `kotlin` list). P1-39 does not wait on P1-15;
  (6) the order of `Listing.items`, `groups` and `members` is by raw code point and is **not** the canonical order (the canonical one is by NFC
  strings, spec v1 §6, which `answer.canonicalize` applies): the builder NFC-normalizes and strips before emission and never takes `group`'s
  order as the bytes' order.
- **Gate:** `analyst`; `data-scientist` read the measurement (the result note above). **Needs:** 23.

## P1-39d · the engine flip and the flow notice — `tdd-engineer`

`-answer-spec2.md` §5.1 (trigger), §8.1 (the moved assertions). **Needs:** 23d, 45b. **Blocks:** P1-25, P1-41, P1-42. **Gate:** `analyst`. One commit makes the shipped walk V-both, the label `"1.2"` and `notices` present together.

- **Files:** `cpg/ccg/engine/bfs.py` (`walk`, `definition` default to `SHIPPED_ROWS`; `Walk`: new `flow_langs`, derived from `Walk.visited`), `cpg/ccg/engine/flow.py` (`Units.ensure`, the existing `units` read, for the visited code nodes' units),
  `cpg/ccg/engine/answer_build.py` (`build_answer`: `"answerSchemaVersion": "1.2"`, `notices`, the version of a `table=None` document), `cpg/tests/test_engine_answer.py`, `test_engine_answer_v18.py`, `test_engine_flow.py`, `test_engine_bfs_guards.py`.
- **Behaviours:** (1) the default table, label and version flip as above; (2) `flow_langs` is the set of `Unit.language` of flag-false units hosting a visited `Symbol`, `CallSite` or point node, the anchor included; a unit with no `Unit` row or no unit on a visited code node is a `GraphDataError`, a missing language is one only for a flag-false unit (§5.1);
  (3) `build_answer` sets `notices` to `[{"code": "flow-not-enumerated", "langs": sorted(flow_langs)}]` when non-empty and omits the key otherwise; (4) tests: a flag-false python anchor with callers lists them and carries `[python]`; **a flag-false anchor with no listed item carries it**;
  a flag-true anchor and its callers carry none; a mixed python (false) and javascript (true) chain names python only; an unbound config anchor and an edge-less contract point carry none, while a contract anchor with providers or consumers in flag-false units carries their languages; a bound config anchor reaching a flag-false `CallSite` carries that language; the depends-on column behaves the same;
  a flag-true unit with no language does not raise; the notice is identical for the same request run twice and in `facts_bytes` across rebuilds; the new `units` read adds a name to `recipe.queries` where none was read, and those pins change here;
  (5) the assertions that move, by name (`-answer-spec2.md` §8.1): `test_engine_answer.py::test_a_direct_caller_builds_a_valid_resolved_1_1_answer_whose_chain_is_the_walks` (label) and `::test_the_recipe_names_the_shipped_definition_the_guards_and_every_query_the_build_ran` (algorithm `"1.2"`), and in `test_engine_answer_v18.py` the shipped-population label check and
  `test_a_candidate_table_document_carries_the_unpublished_version_and_the_shipped_one_1_1` (the population's shipped table inverts: `table=None` is V-both, as-built passes `bfs.ROWS`). The snapshot-row `"1.1"` pins (graph-model version) stay. `build_answer(table=...)`'s `CANDIDATE_VERSION` labelling is re-derived: V-both now fingerprints to `1.2`, so a document from an explicit table takes the version its record maps to (`CANDIDATE_VERSION` only when none maps). The stale docstrings are reworded: `answer_build.py` and `engine/staleness.py` ("a 1.1 answer's `facts_bytes`" becomes "1.1 or later"; "V1-V19" becomes "V1-V20"). **The unit re-derives every assertion that takes the default table as its baseline**, at least these families: `test_engine_bfs_guards` (the default definition, rows and version tests), `test_engine_answer` (the depth-bound and `unexplored` test now lists the callee), `test_engine_answer_v18` (the other candidate-table tests), `test_engine_flow` (the V-flow, V-both and candidate-version tests and the read-count cases).
- **Done at its commit:** every answer the builder returns validates under V20 and V18's `"1.2"` rules; a flag-false V-both answer lists callees with `possible`/`flow-unavailable:<lang>`; mutants killed: the anchor omitted from the visited set, a flag-true language included, the zero-item answer losing the notice, `notices` dropped from `facts_bytes`.

## P1-24 · renderers — `coder`

- **Files:** `cpg/ccg/render/__init__.py`, `text.py`, `markdown.py`, `cpg/tests/test_render.py`,
  `cpg/tests/golden/render/` (rendered golden for `cpg/tests/golden/answer-ac1.json`).
- **Behaviours:** deterministic templates only — no fact is model-written; grouped text and a Markdown impact
  report (FR-10, AC-11); each hop shows file and line; a `Link` hop expands to its `BOUND_BY` and `LIFTED_VIA`
  entries; a `depthBound.unexplored > 0` note reads "reach stopped at distance d with k unexplored edges"
  from the two fields; **free-text identifiers from source**: `keyPath`, config keys and `launchFact.mountPath`/`service`/`envVar`
  pass `ccg.answer.mask_identifier` (F-4, whose closed kind set is answer spec §7.3's) and, with `Cond.code`,
  `calleeName` and `qname` (F-3 fenced only, never masked), are fenced with
  `fence_data` wherever a model could read them (F-3); the renderers do not fence minted strings (ids, enums), so
  `summary_input` additionally guards each minted string against its schema pattern or enum before it reaches a prompt; a planted `"Ignore previous instructions"` in `Cond.code`
  occurs in the rendered output only inside the fence, and the string never reaches the plain-language
  summary prompt (C2, answer spec §7.1–§7.2); `viaData` is never rendered as "path-matched" for an item whose
  chain has no `DATA` hop.
- **Gate:** `analyst`; mutation check on the fence and the mask.

## P1-24b · renderers show the flow notice — `coder`

`-answer-spec2.md` §5.1 (wording, placement). **Needs:** 24, 45b. **Blocks:** P1-25, P1-26 (both call `flow_notice_lines`). **Gate:** `analyst`; the wording is the stakeholder-visible part.

- **Files:** `cpg/ccg/render/common.py` (new `FLOW_NOTICE_TEXT` template, `FLOW_NOTICE_BANNED`, `flow_notice_lines(doc)`), `cpg/ccg/render/text.py` (`render_text`), `cpg/ccg/render/markdown.py` (`render_markdown`),
  `cpg/ccg/render/__init__.py` (`summary_input`), `cpg/tests/test_render.py`, `cpg/tests/golden/render/` (a `"1.2"` golden with a notice).
- **Behaviours:** (1) `render_text` and `render_markdown` print the notice directly after `counts_line` and before `reach_note` and any item, for every resolved document with `notices`, including one with no items; Markdown as a blockquote under `## Summary`;
  (2) the wording is §5.1's: every `langs` entry named, flow-dependent effects "may be missing from this list", the walk did not follow data, return-value, argument or control-flow edges (call, link and wiring edges were followed), absence not evidence that no such effect exists, never "none"/"nothing", the same text for both columns; (3) a test over the rendered notice of one impact and one depends-on document applies §5.1's ban
  (case-insensitive whole words `safe`, `complete`, `exhaustive`, `full`, `guarantee`, `guaranteed`; phrases `no impact`, `all impacts`; `incomplete` allowed) and the no-digit-outside-a-language rule, with `langs` substituted; (4) `summary_input` adds `notice flow-not-enumerated: <langs>` (a minted enum and langs only) and still validates first;
  (5) a document without `notices` renders byte-for-byte as before (the existing goldens are unchanged).
- **Done at its commit:** mutants killed: the call dropped from either renderer, the notice moved below the items, a language omitted, the template gaining a banned word or the word "none", the zero-item answer rendering no notice.

## P1-25 · clients: `ccg` MCP server, CLI, `raw_query` — `coder`

- **Files:** `cpg/clients/mcp/server.py` (the `mcp` SDK, stdio, FastMCP style; tools `ask`, `resolve`, `refresh`,
  `raw_query`; index D6), `cpg/clients/cli/main.py` (`ccg ask|resolve|refresh|raw` emitting canonical JSON,
  optional `--fail-on possible|any` exit codes), `cpg/clients/common.py` (system resolution, identity, the `ccg_raw_<s>` connection),
  `cpg/tests/test_clients.py`, `cpg/tests/test_pin_replay.py`, `cpg/tests/test_no_network.py`; a repo-root `.mcp.json` entry is a **proposal** in
  the step's report (the wiring file is `teco`'s/`cobb`'s to change, P1-27).
- **Behaviours:** (1) `ask(request)` takes the structured request `{useCase, anchor}` (use case `impact` or `depends-on`; no depth bound, `-answer-spec2.md` §4.1; **`scope` is not a P1 input**: the engine takes none, so a request carrying it is the typed `not-built` error, never ignored, and the answer's `request` carries no `scope`) or a plain-language question, plus the boolean `cacheBypass`
  (default `false`; the CLI flag `--cache-bypass`; no effect on a structured request; `-answer-spec2.md` §2), and
  returns the canonical answer; **other use cases return a typed "not built in this
  phase" error**, never an empty answer; **a structured ask whose anchor is ambiguous or unresolved is a typed error** (`anchor-ambiguous`/`anchor-unresolved`) carrying `resolve`'s candidate shape, and no structured path builds a canonical `ambiguous` or `unresolved` document: those statuses describe the reading of a *question*, so **P1-26 owns them** (plain-language path, `options` with `rank`); a structured ask names its anchor, and a wrong one is the caller's error; (2) `resolve(anchor)` returns candidates with ids (code location,
  contract point); (3) `refresh(system, scope)` calls P1-20; (4) `raw_query(cypher)` runs `GRAPH.RO_QUERY` with
  the `ccg_raw_<system>` credential only; a write, `GRAPH.QUERY` and `GRAPH.PROFILE` are refused by the database
  and the stamp is unchanged (AC-21); a time limit is sent on every call, its default is the pinned `TIMEOUT_MAX`, and the server refuses a larger one (the client never reads the instance value); a time limit bounds wall time, not memory (a documented P1 limit: below); (5) the asker is the launching OS
  user; access control is P4 (FR-24), so `raw_query` is available to the local user in P1 and the report says so;
  (6) **AC-16 pin-and-replay** (`-eval-protocol-ml.md` §10.6): the structured request submitted through MCP, the
  CLI and the engine gives identical `facts_bytes`; a raw-recipe re-run of `recipe.queries` reproduces the item set
  (subject ids with `status` and chain `(kind, id)`), reported as the weaker leg by decision; (7) **no client holds a network credential or opens a socket except to FalkorDB**: `cpg/tests/test_no_network.py`
  parses every module under `cpg/ccg/` except `cpg/ccg/broker/`, plus `cpg/extractors/` and `clients/`
  (`-egress-spec3.md` §10.1, I2's one import-boundary test)
  with `ast` and fails on any import or call that can reach the network outside a per-module allow-list
  (`falkordb`/`redis` for exactly `cpg/ccg/store/`, `cpg/ccg/engine/reader.py` and `cpg/clients/common.py`, the one
  client module that opens `raw_query`'s `ccg_raw_<s>` connection; `-egress-spec3.md` §10.1), including `import urllib.request as u` inside a function,
  `__import__("socket")`, `importlib.import_module`, `subprocess` of `curl`/`wget`, and an import of `ccg.broker`
  outside `ccg/refresh.py`, `ccg/front/interpret.py` and `ccg/front/summary.py` (`ccg.broker.client` only; the
  clients themselves open no broker run); each is a named mutant the test must kill; **the boundary is also a run-time one** (`-egress-spec3.md` §2.1, "import no broker module"): importing `clients.common` and running `ask`, `resolve` or `raw` loads no `ccg.broker*` module, so `Client.refresh` imports `ccg.refresh` inside the method (and `DEFAULT_MANIFEST` comes from where `ccg.refresh` takes it); a test asserts `sys.modules` holds none after those calls (`subprocess` and `socket` are not asserted: the lazy `falkordb` import brings `socket`); this fix belongs to the open P1-25a round;
  (8) it calls `validate` first and renders only a valid document; a test passes an invalid answer and asserts
  nothing is rendered; (8b) **`--fail-on` and unresolved documents:** with `--fail-on` given, an `ambiguous` or `unresolved` document exits **3** (neither 0 nor the tripped code 1), so a gate never reads "could not understand or locate" as success; without `--fail-on` the document is delivered with exit 0; a test with a document double covers it until P1-26 can produce one; (9) **the flow notice is never dropped** (Needs P1-24b for the text path) (`-answer-spec2.md` §5.1): the MCP `ask` and the CLI return the canonical document unchanged, so `notices` is in the JSON; a text-mode
  output is `ccg.render.render_text`; a test asserts a document with `notices` comes back from MCP and CLI with the field intact, and that `--fail-on` exit codes are unchanged by it (§9, Q-1).
- **Raw access and memory (stakeholder decision 2026-11-19, a documented P1 limit):** a raw query inside its time limit can exhaust the FalkorDB instance's memory (`raw --cypher 'UNWIND range(1,100000000) AS x RETURN count(x)' --timeout-ms 200` OOM-killed a throwaway at a 2 GB cap). On the shared, no-persistence `ccg-falkordb` that drops every loaded snapshot until the next `refresh`. P1 accepts it: one local user, `refresh` rebuilds, and per-user limits belong to P4 (FR-24). The database is the enforcer, so there is no client-side query filter. The P1 exit report states the limit (P1-38).
- **Gate:** `analyst`; AC-16, AC-21 evidence is P1-42's.

## P1-26 · plain-language front — `coder`

Blocked on `-p1-ml.md` question 3 (model, prompt, grammar, sampling, smoke protocol, `CAP`), on the broker core
(P1-29), on P1-24b (the summary calls `flow_notice_lines`), on P1-25 (the `ccg/front` modules sit under its import-boundary test) and on P1-45 (`cacheBypass` in provenance). Dispatch 1 (parsers, resolver, draft types, path-suffix reads) is committed (`13e56cbc`); dispatch 2 is the interpreter, cache, summary, prompts and grammar.
- **Files:** `cpg/ccg/front/parsers.py` (deterministic: stack traces are P5, so P1 parses routes, `file:line`,
  `path::symbol`, qnames, and diffs only as far as "not built"), `resolve.py` (deterministic resolver over
  `engine/anchors.py`), `interpret.py` (one-shot schema-constrained extraction of the use case and anchor spans),
  `cache.py` (key `ccg.answer.cache_key(normalized_question, snapshot_id, model_fingerprint, prompt_hash, …)`),
  `summary.py` (optional aggregate-only summary), `prompts/*.txt`, `grammar/*.json` (**never `.md`**, index D5),
  `cpg/tests/test_front_*.py`, and `cpg/clients/common.py` plus `cpg/clients/cli/main.py` (the default `front` wiring: `Client` and the CLI factory import `ccg.front` lazily, so `--question` and the MCP `ask` stop being `not-built`).
- **Behaviours:** (1) deterministic parsers run first; a question they fully resolve makes no model call; (2) for
  free prose, one model call through the broker's structured path (`component: answer.interpret`,
  `purpose: model-call`) with a grammar-constrained JSON reply validated against the canonical `Request` schema;
  (3) ambiguity is answered with the possible readings as `options` with `rank`, never a guess, and **P1-26 builds the canonical `ambiguous` and `unresolved` documents** (`-answer-spec.md` §4; the structured path of P1-25 returns typed errors instead); a question naming a repository scope is the typed `not-built` (scope is not a P1 input); the pre-registered cap (`CAP` = 5) applies to the did-you-mean lists only, while a complete reading (several-symbols, two-use-cases) is uncapped, bounded only by the engine's `OPTIONS_MAX` = 20 whole candidates (`-answer-spec2.md` §3, "every reading"); (3b) **draft and resolver** as `-p1-ml.md` §4.2 fixes them (span-group `InterpretDraft`, `direction: null` expanded into both
  directions as `options`, the verbatim-substring guard, the per-group join rule; the resolver resolves a partial path such as `falkorchat/api.py:603` through the engine reader's `file_by_suffix`/`skipped_by_suffix`, `-p1-graph.md` §5, after an exact repo-relative miss), with its **named resolver tests** on dev
  questions: QW-30, QW-27, QW-13, QW-7, QW-47, and negatives QW-3 and QW-11; (4) the interpretation provenance (`endpoint`, model fingerprint, quantization for
  local, prompt/grammar/sampling hashes, `cacheHit`) is in the answer; a cache entry lives exactly as long as its
  snapshot, and the `cacheBypass` input (`-answer-spec2.md` §2) makes the ask neither read nor write the cache, calls the
  interpreter, and records `cacheBypass: true` with `cacheHit: false` (EP §10.4); (5) the optional summary is
  aggregate-only and **dropped on any added entity** (the notice's `langs` values, minted `lang` strings, are allowed echoes, not added entities); (6) the broker consequences of `-egress-spec3.md` §2.1 and §10.2: one `answer` run per question, opened at the first
  model call from `interpret.py` or `summary.py` and reused for later calls, none for a question the parsers resolve;
  the `local` entry's `paths` fixed; a test that a missing or latched broker fails the call with no direct connection
  attempted; and P1-25's `cpg/tests/test_no_network.py` passes unchanged with `ccg/front` present; (7) a
  question about a use case P1 does not build is understood and answered "not built in this phase" with the
  request echoed; (8) no file under `cpg/ccg/front/` is `.md` (D5); (9) `summary.py` calls `validate` first and
  renders only a valid document, with a test on an invalid one; (10) (Needs P1-24b) a summary of a document with `notices` is always followed by `ccg.render.common.flow_notice_lines(doc)` verbatim, assembled by
  code, not restated by the model; a test with a stub model that omits it still shows it (`-answer-spec2.md` §5.1).
- **Gate:** `analyst`; `data-scientist` for the method. The AC-24 smoke run is P1-35.

## P1-27 · `teco` v1 and the usage skill — `cobb`

The unit's questions are in the index §9 (N-COBB). **Files** (all agent-definition or skill text, so `cobb`'s):
`claude/teco/teco.md` (the architect and implementer dispatch seams: an impact answer and its stamp in place of the
`CPG: <graph>, built …` line **for systems `ccg` covers**; falls back to the `cpg-analysis` freshness line for
`cpg_*` graphs), `skills/cpg-analysis/SKILL.md` (usage of the four tools; recipes for optional `cpg_*` graphs and
the freshness recipe stay), `claude/teco/kaizen/` per `claude/AGENTS.md`'s update rule, `claude/AGENTS.md` and
`skills/README.md` rows if wording changes, and the repo-root `.mcp.json` entry for the `ccg` server, which carries **no credential** (the server reads `CCG_CREDENTIALS_DIR`, P1-02; Claude-Code-only wiring, like `cypher`; OpenCode/Kiro wiring is C-310's problem, not P1's).
- **Behaviours (reviewable, not executable):** a brief for an architect or implementer on a covered system names the
  answer, the stamp, the staleness note and the answer's `notices` (the flow notice, `-answer-spec2.md` §5.1) verbatim when present; an uncovered system keeps the old line; a refresh failure is surfaced
  as "stale-kept" and never silently omitted.
- **Gate:** `analyst`; `cobb` keeps final say on prompts.
