# FalkorDB verified engine quirks — this lab's edge build

> **Live-verified knowledge base for `graph-dba`.** Facts confirmed by hands-on
> testing against this lab's running FalkorDB instance, not just docs — they
> diverge from the general documentation or from Neo4j assumptions. Treat them as
> **ground truth for this build**.
>
> **This is a cache, not the source of truth.** It is pinned to a specific build:
> `falkordb/falkordb:edge`, Redis 8.2.2, graph module reporting version
> **`999999`** (FalkorDB's edge/untagged sentinel — tracks latest `main`, a moving
> target). **Re-verify every entry against the live instance on any tagged-release
> upgrade** (e.g. a move to a `v4.x` build), and re-stamp the date below.
>
> **Verified: 2026-07-05 against the live `falkordb:edge` instance (module 999999).**

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

## Cypher dialect & query behavior

- **Cross-graph edges silently no-op** — no error, `MATCH` just returns 0 rows.
  There is nothing to catch.
- **`(:A | :B)` union-label syntax** in a pattern is unverified on this build —
  use `coalesce()` over two label-specific `OPTIONAL MATCH`es instead.
- **`length(path)` in `ORDER BY`** is not supported — order by a property instead.
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

## Query tuning

- **An `OR` across two label-specific properties as the scan anchor**
  (`WHERE n.propA = $x OR n.propB = $x`) profiles as an `All Node Scan` even when
  both properties are indexed. Use two separate `OPTIONAL MATCH`es (one indexed
  lookup per label) + `coalesce()` instead. The `OR` form is fine once `n` is
  already bound by an indexed/traversal anchor — it's only a scan-anchor problem.

## Ops, config & tooling

- **`GRAPH.RO_QUERY`** routes to read replicas — use it for all read-only traffic.
- **Bolt port is `65535`** per `GRAPH.CONFIG` (not the Bolt default).
- **Default `TIMEOUT` is 1000ms — and writes ignore it entirely**; a write runs to
  completion regardless of clause or default. Reads enforce it batch-granularly
  (slightly-over queries can slip through). The client `timeout=` pass-through
  (`g.ro_query(q, params=…, timeout=…)`) works and is **uncapped while
  `TIMEOUT_MAX=0`**.
- **`GRAPH.MEMORY USAGE` under-reports vector-index memory** (reports
  `indices_sz_mb: 0` with a live HNSW index holding real vectors) — size
  vector-heavy workspaces from `INFO memory` deltas instead, until fixed upstream.
