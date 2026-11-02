# Connected code graph: graph model v1 (S17)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** P0-14 (S17) · **Extends:** `docs/plans/connected-code-graph-graph.md`

Graph model v1 frozen with schema v1 (`docs/plans/connected-code-graph-bundle-spec3.md` +
`docs/plans/connected-code-graph-bundle-spec4.md`, freeze commit `88eb134d`). This note states **only the
change against** `docs/plans/connected-code-graph-graph.md` §3.2-§3.5, §6.2, §6.3 (the "graph note"):
anything not named here is as the graph note says. Evidence: the slice graph `ccg_p0slice` on `ccg-falkordb`
(read-only), `docs/test-reports/connected-code-graph-p0-slice-report.md` (G1-G8, Q1-Q10) and the S15 graph check in
`docs/reviews/connected-code-graph-p0.md`. It is a successor file, not an edit of the graph note: the graph note has
been executed against (collision rule 5); the graph note gains an `Extended by:` header pointer only.

CPG: considered, not relevant: a design note; the subject is the loaded `ccg_p0slice` slice graph, which no `cpg_*` graph indexes.

## 1. Model v1

### 1.1 What the slice proved and what is frozen untested

Loaded and read live (38,121 nodes, 90,867 edges, 10 `UNIQUE` constraints `OPERATIONAL`): `Repo`, `Unit`, `File`,
`Symbol`, `Param`, `Ret`, `Cond`, `CallSite`, `Name`, `ContractPoint`, `Link`, `Snapshot`; edges `IN_REPO`,
`HAS_FILE`, `DEFINES`, `HAS_CALL/PARAM/RETURN/COND`, `CALLS`, `CALLS_NAME`, `RESOLVES_TO`, `DATA`, `CONTROLLED_BY`,
`PROVIDES`, `CONSUMES`, `VIA`, `TO`, `ON`, `LIFTED_VIA`. **Frozen, untested** (first testable phase in brackets):
`ownedRegions` and the anchor rule [P1 fixture]; multi-candidate links [P1]; `MOUNTED_BY`, `SkippedFile`,
`PROVIDES` props [P1 loader]; `fieldAccess` `READS`/`WRITES`, `ConfigEntry`, `SchemaEntry`, `Dependency`, `Mark`,
`Finding`, `RAN`, `LogTemplate` [P1-P5]; `TIMEOUT_MAX`-nonzero loads and multi-batch splitting [S22/S23, P1];
`arg(i)`/`Ret` traversal rows (§3) [P1]; a `DATA` cycle beyond the self-edge [S20 synthetic graph].

### 1.2 Deltas against the graph note

| # | Element | v1 rule |
|---|---|---|
| D1 | `LIFTED_VIA` | `(:Link)-[:LIFTED_VIA {ord, siteId, line, urlPath, verbPath?, viaSiteId?, viaLine?}]->(:CallSite\|:Symbol)`. Hop nodes are shared by many links, so the per-link line and path pair live on the edge. Optional props are absent, never empty. Replaces `{ord}` (graph note §6.3) and `{ord, siteId}` (spec3 BF6). |
| D2 | `VIA` | `(:CallSite)-[:VIA {status, reason, consumerKey}]->(:Link)`: **per-site** status and reason of that site on that link, and the consumer-side `keyNorm`. `Link.status` stays "best over its sites"; `CONSUMES {status, reason}` stays the raw recognizer result. `consumerKey` is kept: it is what groups sibling Links (D6). |
| D3 | `CONSUMES`, `ON` | Not deltas: the graph note's edges, loaded and correct (`ON` target equals `CONSUMES` target in 11/11). They are **read** by contract mode (§3.2). |
| D4 | `PROVIDES` | `(:Symbol\|:CallSite\|:ConfigEntry)-[:PROVIDES {status, reason}]->(:ContractPoint)`. `status` ∈ {confirmed, possible}, `reason` the recognizer's code. Several providers per CP is the design (61 providers, 60 CPs). |
| D5 | `MOUNTED_BY` (new) | `(:Symbol provider)-[:MOUNTED_BY {ord}]->(:CallSite mount)`, `ord` by (mount path, line). The mount call site and its condition are already in the graph (`app.py:521` under `shop is not None`, `app.py:485` under `dev_surface`, via `CONTROLLED_BY`); this edge adds only the provider-to-mount link. **Assembler requirement for P1, outside the first loader slice:** the assembler must emit mount facts as ids (provider `symbolId`, mount `callSiteId`); today they exist only as assembler facts. |
| D6 | `Link.candidateCount` | **Dropped from `Link`.** A Link is one provider (spec3 BF6: `linkId` includes the provider anchor), so "may call any of N" is a property of a **`VIA` edge**, not of the Link (one Link can be reached by sites with different consumer keys). For the `VIA` edge being rendered, N = the number of Links, **the rendered one included**, that have a `VIA` edge from a `CallSite` of the same consumer symbol (reached through `HAS_CALL <-`) with an equal `VIA.consumerKey`. Derived at render time; one home. |
| D7 | `Symbol` extras | Carried, unindexed: `complexity`, `lang`, `signature` (overloading languages only), `depth` (nesting depth, computed by the loader from `parentId`, for the anchor tie-break), `ownedRegions` (flat int list `[s0,e0,s1,e1,…]`, absent when empty; pair decoding works in this build: `[i IN range(0,size(o)-1,2) \| [o[i],o[i+1]]]`). `parentId` is not stored. `File` carries `lang`, `contentHash`, `kind` and `role` (orthogonal, BF7), unindexed. |
| D8 | `DATA`, `CALLS` status | `status`/`reason` stored **only when not the default** (`confirmed`/`intra-flow` for `DATA`, `confirmed`/`resolved-call` for `CALLS`); a reader coalesces absent to the default. The slice stores both on all 16,424 `DATA` edges (4 are `possible`/`field-truncated`); the bundle always carries both. (BF13: +56 B per edge, +42 % on a probe.) |
| D9 | `DEPENDS_ON` | adds `scope`. |
| D10 | Constraints | 11, not 7: the graph note's `Symbol`, `CallSite`, `ContractPoint`, `Link`, `Name`, `Unit`, `File` plus `Param`, `Ret`, `Cond` on `pointId` (S-1) and `Dependency` on `depId` (the loader matches `DEPENDS_ON` targets by it; measured in `cpg/spikes/store/MEASUREMENTS.md`). The full index and constraint set is the table in §1.6. |
| D11 | `Name.nameKey` | `h(["name", lang, calleeName])`, constrained (BF14); `lang`, `name` readable and unconstrained. `<dynamic>` sites get no `CALLS_NAME`. |
| D12 | `DATA` self-edge | `use == def` is legal and **kept** (spec4 §2). Slice instance: `tests/test_querygen.py:844`, `tail = tail.split(…)[0]` in a loop (loop-carried). Every traversal carries a visited set (§3.1). |
| D13 | Path grammar | spec3 §5.1 (`recv`, `kw.<n>`, quoted field segments) replaces the graph note's four forms. `arg(i, path)` becomes `arg(head, path)`: `a<i>` binds the callee `Param` by `index` after the receiver, `recv` to index `-1`, `kw.<n>` by `Param.name` (including `**kwargs`). |

### 1.3 File population and the "is X mapped?" lookup (spec4 §1)

`header.skippedFiles` has one source, the header. The graph holds it as nodes so the lookup is one indexed read:

```
(:Unit)-[:SKIPS]->(:SkippedFile {repoId, unitId, path, reason})   reason ∈ {test, frontend-ignore}
```

The store holds many repos and a path such as `src/index.ts` repeats across them, so **both lookups are repo-scoped**.
`SkippedFile` has no constraint (the validator guarantees sorted, unique per bundle); indexes are in §1.6. Rule, three
separate queries in this order (no `OR`, no `IS NULL OR` filters; the index seeks on `path` and `repoId` filters the rows):
1. `MATCH (f:File {path:$p, repoId:$r}) RETURN f.unitId` → **mapped**.
2. else `MATCH (s:SkippedFile {path:$p, repoId:$r}) RETURN s.unitId, s.reason` → **not mapped** (with the reason).
3. else → **not found**. Declaration files (`.d.ts`, `.pyi`), vendored paths and anything outside the units are in
   neither set, so they answer "not found", never "not mapped".

`SkippedFile` has no `DEFINES`, no `HAS_FILE` and no `kind`/`role`: it is a coverage fact, not a code element, so
no file-scoped query (`File`-based staleness, redaction, evidence) can pick it up by accident. The slice
`salesperson/` bundle predates the field; its 33 absent files load as `SkippedFile` only after a P1 re-extract.

### 1.4 `ownedRegions` and the File+line anchor (spec3 §9.1)

Replaces the graph note §3.4 span-only filter. The query fetches **every** symbol of the file (≈36) via the
`File.path` index (`repoId` filtered, as §1.3) and `DEFINES`, because an owned region can lie outside the span and a Cypher span predicate would
miss it:

`MATCH (f:File {path:$p, repoId:$r})-[:DEFINES]->(s:Symbol) RETURN s.symbolId, s.lineStart, s.lineEnd, s.ownedRegions, s.depth`

The engine picks, for line `l`, the symbol with the **narrowest** containing interval (span or owned region); on a
tie a span match beats an owned-region match, then larger `depth` wins. The rule is engine-side and unindexed by
design; the first real case is a P1 fixture (0 of the 4,943 slice symbols carry `ownedRegions`).

### 1.5 `launchFact`, `configEntry`, `fieldAccess` (S17 rulings F-11)

- **G-1:** the `launchFact`-origin `ConfigEntry` carries `launchKind`, `protocol`, `port`, `envVar`, `mountPath`,
  `service` and `argv0`. `argv0` is the bundle's `command` field (spec3 §1.6: argv[0] after the argv gate, or
  `"<redacted>"`), present on `kind: command` records only. It never carries `args`, `default` or `defaultHash`.
  Role-relevant argument pairs are **not carried**: §1.6.3's allow-list is the gate's closed non-redaction list of
  single-dash flags, not a role list, and copying pairs would copy the AV-68 residual (`-host hunter2`) into the graph.
  If S33 needs pairs, define a separate closed list and an AV-68-style test first. **S33 is told:** a `launchFact` hop renders the carried fields it has
  and, for an absent one, says nothing (never file and line "instead of" a field that exists); rendered `mountPath`,
  `service`, `envVar` pass the F-4 token screen at render.
- **P2-8:** `ConfigEntry.keyPath` is stored with each screen-failing segment replaced by `[#…]`; `entryId` is
  computed from the raw `keyPath`. `bindValue`, `format`, `example` stay out of the graph and `bindValue` is never rendered.
- **F-2 closed:** no cross-entry `valueHash` equality is stored or queried; a cross-entry hash is bundle-only.
- `fieldAccess` → `READS`/`WRITES {path, status, reason, linkId}`; two paths on one point are parallel edges with
  one `linkId`. `Cond` is a source of `CONTROLLED_BY` (one Cond→Cond group).

### 1.6 Index and constraint set (the DDL is checked against this table)

Range indexes are created on the empty build key **before** the load (edge loading `MATCH`es endpoints by them);
`UNIQUE` constraints **after** the load, polled to `OPERATIONAL`. `db.indexes()` after the build equals the index column.

| Label | Indexed properties (before load) | `UNIQUE` (after load) |
|---|---|---|
| `Repo` | `repoId` | – |
| `Unit` | `unitId` | `unitId` |
| `File` | `fileId`, `path` | `fileId` |
| `SkippedFile` | `path`, `repoId` | – |
| `Symbol` | `symbolId`, `nameNorm`, `qname` | `symbolId` |
| `CallSite` | `callSiteId` | `callSiteId` |
| `Param`, `Ret`, `Cond` | `pointId` (each) | `pointId` (each) |
| `Name` | `nameKey` | `nameKey` |
| `ContractPoint` | `cpId`, `keyNorm` | `cpId` |
| `Link` | `linkId` | `linkId` |
| `Dependency` | `depId` | `depId` |

Deferred to the phase that loads the label or edge: `ConfigEntry`/`SchemaEntry`/`Mark`/`Finding`/
`LogTemplate` id indexes, the `READS`/`WRITES`/`DISMISSED` `linkId` relationship indexes, and the full-text index.

## 2. Disposition of spec3 §9 and BF2-BF15

§9.1 and §9.2 are adopted as the v1 mapping except as amended in §1 above and by spec4. Per finding:

| BF | Disposition |
|---|---|
| BF2 | `Symbol.qname` stays path-anchored and indexed. Stack frames match by `File.path` suffix plus line (§1.4). A native name is an additive `nativeQname` in P5. |
| BF3 | Accepted: the edge props are `fromPath` (on the def) and `toPath` (on the use); the bundle names are `use`/`def`. |
| BF4 | Accepted; D13. |
| BF5 | `constant` and `schemaEntry` have no node; `fieldAccess` and `schemaEntry` map as §1.5 and §9.1. |
| BF6 | One `Link` per (consumer symbol, provider anchor, key); several `CallSite`s `VIA` it; D1, D2; `Link.status` best over sites. |
| BF7 | `kind` and `role` both on `File`, unindexed (D7). |
| BF8 | `valueHash` is an HMAC; `bindValue` out of the graph (§1.5). [P-later] |
| BF9 | `library_api` key is the readable signature; no hash. [P-later] |
| BF10 | `keyNorm` unscoped in v1; store scoping is P3 and a major change. |
| BF11 | `Symbol.lang` is the file's token. |
| BF12 | `lastChangedAt`/`lastChangedCommit` on `Symbol`, computed outside the sandbox. |
| BF13 | Carry `complexity`, `lang`, `signature`, `File.lang`, `contentHash`, `DEPENDS_ON.scope`; `DATA` status rule D8; not carried: `tools`, `egressMode`, `sandboxAuditIds`, `counts`, `history`, `parentId`, `ordinal`, `scope` on `Dependency` nodes, `framework`/`evidence`, `op`, `format`. |
| BF14 | D11. |
| BF15 | `sandboxAuditIds` is bundle provenance, not in the graph; `:Snapshot.manifestHash` covers it. |

## 3. Traversal: `bfs-v1`

The algorithm name and version are published in every answer's recipe (graph note §6.1 item 3, FR-19).

### 3.1 Shape

Level-synchronous BFS; one parameterized, index-anchored, read-only query per (edge type, direction) per level,
ids chunked ≤ 1,000, reply never silently truncated (`RESULTSET_SIZE` 10,000: raise on reaching it). State per
frontier entry: `value`, `path(P)`, `contract`, `key`, `end`. Cost `(possible hops, distance, length)`, ties broken
on the predecessor's stable key. A **visited set**, seeded with the anchor and keyed `(label, id, state kind, state
path)`, stops every cycle: `DATA` self-edge and cycles, cond→cond `CONTROLLED_BY` (F-12), recursion. A node is
re-expanded only on a strictly smaller `(possible, distance)`. A level guard (64) fails loudly.

### 3.2 Rows implemented in `bfs-v1`

Graph note §6.2 as written, plus: `PROVIDES <-` in the **contract** column (CP → its providers, listing and
rendering; distance 0); contract mode reaches consumers by **`CONSUMES <-`** ("matched or not"), and attaches the
`Link` (status, evidence chain) through `ON <-` / `VIA`, so a consumer with no provider is found and reported
`unmatched`. The slice instead walks `PROVIDES <-`, `TO <-`, `VIA <-` and never reads `CONSUMES`/`ON`: P1 must change
this and add a fixture (a contract walk from a provider-less CP lists its consumers). `DATA <-` is path-matched
(`path_overlaps`: equal, or a segment prefix, `p1` is not a prefix of `p10`).

### 3.3 Baseline: function-level continuation

`bfs-v1` does not implement the interprocedural `arg(i)` rows (`CALLS ->` into a callee, index-matched
`HAS_PARAM ->`, a `Ret`-reached callee summary onto the call site's `r`). An **item is a function containing a
reached point**; path-narrowing reduces states and `DATA` fan-out, not items (measured in
`docs/test-reports/connected-code-graph-p0-slice-report.md` §6.8; reproducible from `probe_s14.py`). This is a sound over-approximation (recall preserved); the stored edges already
support the precise walk (live: `apiFetch`'s `Ret` reaches `parse`, `text`, `fetch`, `buildUrl@108`, `Param opts`
over `DATA`, paths `r→ret`, `r→a0`, `r→recv` consistent), so closing it adds traversal rows, no node, edge or prop.

**Labelling rule (M2).** Until those rows exist, a continuation `HAS_PARAM|HAS_CALL|HAS_COND <-` followed by
`CALLS <-` or `TO <-` is a **`possible`** hop, `reason: function-level-continuation` (weight 1,000). A continuation
from a `Ret` (`HAS_RETURN <-`) and `CALLS <-` from an anchor `Symbol` stay `confirmed`. Each item carries `viaData`
(its best path contains a `DATA` hop) so "path-matched" is never claimed for function-level items. Impact precision is
not scored before the `arg(i)`/`Ret` rows land. Consequence for the slice: the 21 `buildUrl` items become `possible`,
which `test_live_slice.py` currently asserts `confirmed`.

## 4. Load protocol (D16 / S22 / S23)

Graph note §3.5 as written, with: the indexes of §1.6's index column on the empty `ccg_p0slice__build_<id>`; nodes then edges in `UNWIND` batches **≤ 20,000 rows**, read from one constant (`BATCH_MAX`).
**20,000 is provisional and carried to S23, which replaces it** with the ceiling its `PING`-stall probe records;
S17 closes without that value; one connection, retries explicitly zero (a `Retry(NoBackoff(), 0)` object, never the
client default), socket timeout 300 s; an **explicit query `timeout` = `TIMEOUT_MAX` on every load query whenever it
is nonzero**; a created-count check per edge batch; constraints after the load, polled to `OPERATIONAL`, any `FAILED`
fails the build; `:Snapshot` last; `RENAME`. Verified live on the slice with `TIMEOUT_MAX` 0 only; the nonzero branch
and a batch split at scale are first tested by S22/S23.

## 5. Tests that change or are added

Cited by test name; paths are under `cpg/spikes/vertical-slice/` unless stated.

- **`candidateCount` (D6, mutant A2).** `tests/test_assemble.py::test_candidate_count_counts_the_providers_a_site_key_reaches`
  and `::test_probe_links_every_field` pin the first-pair value and must change; `tests/test_live_slice.py::test_link_facts_become_nodes_and_edges_with_lifted_chain_props`
  and the live link-row selections read `Link.candidateCount`, as does `load.py` (`build_rows`'s link row). New: a render-helper
  test with **one Link reached by two sites with different `consumerKey`s** asserting two different N (self included),
  and one with a sole candidate asserting N = 1.
- **Lookup scope (§1.3, §1.4).** A live-store test loading **the same path in two repos**, mapped in one and in `SkippedFile`
  in the other, asserting the File+line anchor query returns only the queried repo's symbols, and "mapped", "not mapped" and "not found" per repo, and that a `.d.ts` path answers "not found".
- **Value-mode labelling (§3.3).** `tests/test_live_slice.py::test_sl4_contract_mode_lists_the_endpoints_caller_confirmed_at_distance_one`
  is unchanged; the `buildUrl` value-walk assertion that every item is `confirmed` with no `possible` hop changes to
  `possible` / `function-level-continuation`. New, on hand-built graphs against `impact.py`: (a) a `Ret` reached by `DATA`
  and then `HAS_RETURN <-`, `CALLS <-` → the caller is `confirmed` (kills "label everything `possible`"); (b) a planted
  case where the reached point does not reach the callee's `Ret` → the caller is `possible`; (c) `viaData` is true exactly
  when the best path contains a `DATA` hop.
- **Traversal guards, against `impact.py`'s `bfs-v1`** (not S20's `cpg/spikes/store/bfs.py`, a benchmark implementation):
  a two-point `DATA` cycle with matching paths asserts termination and the state count; a strictly-smaller-cost
  re-expansion test pins `<` against `<=` (the survivor mutant I7); a reader returning 10,000 rows asserts the
  `RESULTSET_SIZE` raise; a chain deeper than 64 levels asserts the level guard raises.
- **D8.** `impact.py`'s `data_in` coalesces an absent `status`/`reason` to the default; a test loads an edge without them.
- **Retries.** A test asserts the loader's connection carries a zero-retry object, independent of the client default.

## 6. Carried and open

1. **Batch ceiling** → S23 (§4). Not held for.
2. **Not in this note's delivery:** `cpg/ccg/store/ddl_v1.cypher`, `cpg/tests/test_ddl_live.py` and the loader changes
   D1-D8; they follow from §1 and need a live `__build_` key.
3. **`MOUNTED_BY`** waits on the assembler emitting mount ids (D5), P1.
4. **S20/S21 measurements (2026-11-02), inputs to S23's decisions table:** `cpg/spikes/store/MEASUREMENTS.md`. BFS
   (`bfs-v1`) beats `algo.SSpaths` (item sets differ on every hub anchor, reply cut at 10,000 rows); `MULTI` swap
   without a guard does not roll back; `MAX_INFO_QUERIES 0` disables telemetry; `GRAPH.DELETE` removes it at once;
   `Dependency(depId)` indexed (§1.6, D10).
