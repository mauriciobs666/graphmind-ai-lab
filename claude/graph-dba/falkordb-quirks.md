# FalkorDB verified engine quirks — this lab's pinned build

> **Live-verified knowledge base for `graph-dba`.** Facts confirmed by hands-on
> testing against this lab's running FalkorDB instance, not just docs — they
> diverge from the general documentation or from Neo4j assumptions. Treat them as
> **ground truth for this build**.
>
> **This is a cache, not the source of truth.** It is pinned to a specific build:
> `falkordb/falkordb:v4.18.11` (tagged release, 2026-06-24), Redis 8.6.3, graph
> module reporting version **`41811`**. **Re-verify every entry against the live
> instance on any release upgrade**, and re-stamp the date below.
>
> **Verified: 2026-07-05 against the live `falkordb:edge` instance (module 999999,
> tracking `main` past the v4.18.11 tag).** Deployment pinned to `v4.18.11` on
> 2026-07-09; the falkor-chat query suite (193/193, which encodes the DDL/ordering/
> anchor quirks below) re-ran green on the pinned build. Entries not exercised by
> that suite still carry their edge-build verification dates — re-probe individually
> if one becomes load-bearing.

`graph-dba.md` (the always-on prompt) points here and stays lean; this file holds
the perishable, growing fact list. When another lab project accumulates its own
"live-verified FalkorDB facts" against this same build, **fold the generic ones in
here** rather than letting them sit siloed in that project's docs — keep the
project-specific corollaries in the project's own `AGENTS.md`, each pointing back
to the general fact here.

---

## Indexing, constraints & DDL

- **Vector index creation is DDL, not a procedure:**
  ```sql
  CREATE VECTOR INDEX FOR (n:Label) ON (n.prop)
  OPTIONS {dimension: N, similarityFunction: 'cosine'}
  ```
  `db.idx.vector.createNodeIndex` is **not** a registered procedure on this build.
- **Index before constraint, always** — `GRAPH.CONSTRAINT CREATE` requires a
  pre-existing exact-match index on the same property, or it fails with
  `"missing supporting exact-match index"`.
- **Composite constraints** (`PROPERTIES 2 key version`) are supported and operational.
- **Fulltext** (`db.idx.fulltext.createNodeIndex` / `queryNodes`) confirmed working.
- **Vector dimension is enforced at query time and index-membership time, but NOT at write
  time** (verified 2026-07-08, module 999999). A wrong-dimension `SET n.embedding =
  vecf32([...])` is **silently accepted** (`Properties set: 1`, no error) — but the node then
  **drops out of the ANN index** and never appears in `db.idx.vector.queryNodes` results.
  Querying the index with a mismatched query vector *does* error
  (`Vector dimension mismatch, expected N but got M`), which is the reliable way to prove an
  index's dimension (`db.indexes()` does not expose it). There is **no `vec.dimension()`
  function** on this build to check a stored vector's length. Consequence: validate embedding
  length client-side before writing — a buggy worker sending wrong-size vectors produces
  permanently unretrievable nodes with no error surfaced.
- **ANN kNN returns *up to* `k`, not exactly `k`** — on a small/near-empty HNSW index,
  `db.idx.vector.queryNodes(…, k, …)` may return fewer than `k` (approximate recall of distant/
  orthogonal candidates). Near neighbors are returned and correctly ordered; don't treat
  "returns exactly k" as an invariant.

## Cypher dialect & query behavior

- **Cross-graph edges silently no-op** — no error, `MATCH` just returns 0 rows.
  There is nothing to catch.
- **`(:A | :B)` union-label syntax** in a pattern is unverified on this build —
  use `coalesce()` over two label-specific `OPTIONAL MATCH`es instead.
- **`length(path)` in `ORDER BY`** is not supported — order by a property instead.
- **`STARTS WITH` on an indexed string property does NOT plan as an index range scan**
  (verified 2026-07-09, module 999999). A prefix predicate like `WHERE n.key STARTS WITH
  $prefix` on an indexed `n.key` profiles as `Node By Label Scan` + `Filter`, not an index
  scan. Consequence: don't use a synthetic-composite-key prefix (`"{a}:{b}:"`) as an
  index-anchored scoping predicate — model an explicit edge (e.g. a `HAS_STEP` containment
  edge) and traverse from an indexed anchor instead.
- **`STARTS WITH` with a concatenated prefix needs explicit parentheses on the RHS**
  (verified 2026-07-09). `x STARTS WITH $a + ':' + $b` errors *"Type mismatch: expected
  Boolean but was String"* — `STARTS WITH` binds tighter than `+`. Write
  `x STARTS WITH ($a + ':' + $b)`.
- **`algo.*` procedures confirmed:** `BFS`, `WCC`, `pageRank`, `SPpaths`,
  `SSpaths`, `MSF`, `betweenness`, `labelPropagation`.
- **Empty `UNWIND` collapses the row stream.** `WITH x UNWIND [] AS y …` drops
  every row that reached it, even ones written earlier in the same query. Guard
  with `UNWIND (CASE WHEN $list = [] THEN [null] ELSE $list END) AS item` + a
  `FOREACH` that never filters.
- **`FOREACH (x IN CASE WHEN … THEN [1] ELSE [] END | CREATE …)`** is the working
  idiom for conditional writes without dropping rows. Nested `FOREACH`, and
  `DELETE` inside a `FOREACH`, both work.
- **`exists((n)-[:REL]->())` in a pattern returns `true` even when the edge is
  absent** (broken on this build); `count{ … }` subquery syntax is unsupported.
  For existence checks use `OPTIONAL MATCH (n)-[:REL]->(x) RETURN x IS NOT NULL`
  instead.
- **`labels(coalesce(a, b))[0]`** subscripting works, for reading the resolved
  label off a `coalesce()` of two optionally-matched nodes.
- **A map-projection cannot be a `CREATE` relationship endpoint** (verified
  2026-07-08, module 999999). `FOREACH (rec IN recs | CREATE (m)-[:R]->(rec.node))`
  where `rec` is a map with a `node` field **errors** (`Invalid input '.': expected
  a label, '{', a parameter or ')'`). The endpoint must be a **bound node
  variable**. To attach per-edge properties while iterating: collect the endpoints
  as **nodes** (`collect(DISTINCT s)`) and pull props from **map parameters keyed by
  the node's own property** — `CREATE (m)-[:R {score: $scoreBy[s.id], rank:
  $rankBy[s.id]}]->(s)`. Dynamic map-parameter indexing by a node property
  (`$scoreBy[s.id]`) works, including inside a `FOREACH`.
- **Two sequential guarded `UNWIND`s** in one query (each followed by its own
  `collect(...)` back to one row) do **not** row-multiply — the first `collect`
  collapses before the second `UNWIND` expands. Pattern: `UNWIND (CASE …) AS a …
  collect(…) AS as  UNWIND (CASE …) AS b … collect(…) AS bs`. Verified for two
  distinct edge blocks (e.g. `MENTIONS_MEMBER` + `EMITTED`) inside one guarded write.
- **Sequential `UNWIND` blocks *without* an intervening collapse row-multiply the
  final `RETURN`** (verified 2026-07-09). `WITH d UNWIND $steps … WITH d UNWIND
  $transitions … RETURN d.key` emits `steps × transitions` duplicate rows. Collapse
  each block back to one row with an aggregation (`WITH d, count(st) AS stepCount`)
  so the query returns a single clean status row. The write itself is unaffected —
  this is a result-cardinality issue, not a correctness one.

## Query tuning

- **An `OR` across two label-specific properties as the scan anchor**
  (`WHERE n.propA = $x OR n.propB = $x`) profiles as an `All Node Scan` even when
  both properties are indexed. Use two separate `OPTIONAL MATCH`es (one indexed
  lookup per label) + `coalesce()` instead. The `OR` form is fine once `n` is
  already bound by an indexed/traversal anchor — it's only a scan-anchor problem.

## Ops, config & tooling

- **`GRAPH.RO_QUERY`** routes to read replicas — use it for all read-only traffic.
- **A read via `GRAPH.QUERY` materializes an empty graph key** — running e.g.
  `MATCH (n) RETURN count(n)` against a *non-existent* graph creates the key (it
  then shows up in `GRAPH.LIST` with 0 nodes). `GRAPH.RO_QUERY` on the same
  non-existent graph instead returns `ERR Invalid graph operation on empty key`
  and creates **nothing**. So to test "does this graph already hold data?"
  without side effects, probe with `RO_QUERY` and treat the `empty key` error as
  "absent/empty"; never scan the whole `redis-cli` reply for digits (the
  `Query internal execution time: 0.179153 milliseconds` line makes everything
  look non-empty — parse the count from the lone pure-integer output line).
  (Verified 2026-07-17 on v4.18.11; surfaced building the `joern` CPG loader.)
- **Bolt port is `65535`** per `GRAPH.CONFIG` (not the Bolt default).
- **Default `TIMEOUT` is 1000ms — and writes ignore it entirely**; a write runs to
  completion regardless of clause or default. Reads enforce it batch-granularly
  (slightly-over queries can slip through). The client `timeout=` pass-through
  (`g.ro_query(q, params=…, timeout=…)`) works and is **uncapped while
  `TIMEOUT_MAX=0`**.
- **`GRAPH.MEMORY USAGE` under-reports vector-index memory** (reports
  `indices_sz_mb: 0` with a live HNSW index holding real vectors) — size
  vector-heavy workspaces from `INFO memory` deltas instead, until fixed upstream.
