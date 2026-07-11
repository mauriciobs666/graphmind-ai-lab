# FalkorDB reference — graph-dba's on-demand playbook

> Companion to [`graph-dba.md`](./graph-dba.md): the detailed practice reference kept out of the
> always-loaded prompt. Read the section relevant to the task before deep work in that area.
> This file is general FalkorDB practice; the **pinned-build divergences** (what this lab's
> engine actually does when it disagrees with the docs) live in
> [`falkordb-quirks.md`](./falkordb-quirks.md) — quirks win. When a detail is version-sensitive,
> verify against docs.falkordb.com for the deployed engine release.

## Graph data modeling (LPG)

- Model the **domain as it is traversed**, not as a normalized relational schema with arrows. Nouns → nodes, verbs → relationships (typed, directed, with properties), labels group nodes by role. Ask "what questions must this graph answer cheaply?" and let the hot traversal paths drive the model.
- Property vs. relationship vs. label is a recurring judgment call — decide it on traversal cost and query shape. Promote a value to a node when it's a shared join target you traverse *to*.
- Know the canonical patterns and when each earns its keep: intermediate/hyper-nodes for n-ary relationships, time-tree/linked-list for temporal ordering, and **dense-node ("supernode") awareness**. (On FalkorDB a supernode is a dense matrix row/column — less catastrophic than pointer-chasing engines because traversal is matrix algebra, but dense matrices still cost memory and compute; mitigate with relationship-type partitioning and direction-specific patterns.)
- **Model for multi-tenancy.** When the workload is per-user/per-tenant, favor **one graph per tenant** over one giant graph with a tenant property — it isolates blast radius, keeps each graph in one shard's RAM, and exploits FalkorDB's multi-graph strength. Decide deliberately.

## Cypher on FalkorDB (OpenCypher dialect)

- Write idiomatic, performant Cypher within FalkorDB's supported surface: `MATCH`/`OPTIONAL MATCH`, `WHERE`, `WITH` pipelining, `MERGE` (+ `ON CREATE`/`ON MATCH`), variable-length paths, `UNWIND`, aggregations, `CALL {}` subqueries, list/string/math functions, `CASE`. **Verify** any function/clause against FalkorDB's docs before relying on it — the dialect is a subset and evolves.
- Execute via `GRAPH.QUERY <graph> "<cypher>"` (read-write) or `GRAPH.RO_QUERY` (read-only, routable to replicas). Use query **parameters** (`CYPHER name=$value`) — never string-concatenate input.
- **Tune with `GRAPH.EXPLAIN` (plan) and `GRAPH.PROFILE` (executed plan + records-per-op).** FalkorDB has its own command syntax — it's not the Neo4j `PROFILE` keyword prefix. Read the operator tree: spot label scans vs. index scans, cartesian products, and dense expansions; anchor the query on an indexed start point and shrink the frontier early.
- **Algorithms are built-in `algo.*` procedures, not GDS:** `algo.pageRank`, `algo.BFS`, `algo.SPpaths` (shortest paths), `algo.SSpaths`, `algo.WCC` (weakly connected components), `algo.MSF`, `algo.betweenness`, `algo.labelPropagation`. Call them with `CALL algo.…`. There is no APOC — if you reach for an APOC procedure, stop and find the FalkorDB-native equivalent or do it client-side.
- **Big writes:** batch with `UNWIND $rows AS row …` over chunked parameter lists, or use a client bulk loader — don't push millions of creates in one query. Make loads idempotent with `MERGE` backed by a constraint.

## Indexing & constraints

- **Range/exact indexes:** `CREATE INDEX FOR (n:Label) ON (n.prop)` (and relationship indexes `FOR ()-[r:TYPE]-() ON (r.prop)`). String, numeric, and geospatial types index. An index helps the **anchor** of a traversal, not every hop — always confirm it's used with `GRAPH.PROFILE`.
- **Full-text** (RediSearch under the hood): `db.idx.fulltext.createNodeIndex`, query with `db.idx.fulltext.queryNodes` / `queryRelationships`, drop with `db.idx.fulltext.drop`.
- **Vector indexes** (the GraphRAG enabler): create vector indexes on node/relationship properties storing embeddings (`vecf32`), query nearest neighbors with `db.idx.vector.queryNodes` / `queryRelationships` using cosine or euclidean similarity. This is how you fuse semantic search with graph traversal.
- **Constraints** via the `GRAPH.CONSTRAINT CREATE` command — **unique** and **mandatory** (existence) constraints on node labels / relationship types. A unique constraint needs a supporting exact-match index; pair `MERGE` with a uniqueness constraint for correctness under concurrency. Introspect with `CALL db.constraints()` and `CALL db.indexes()`.

## Architecture & operations

- **Memory sizing first.** Estimate graph RAM (nodes + relationships + properties + per-type matrices + indexes/vectors) and provision for it plus Redis overhead and headroom; a single graph must fit in **one shard's** RAM. Watch `GRAPH.MEMORY USAGE` / `GRAPH.INFO` and Redis `INFO memory`. Configure `maxmemory`/eviction deliberately (a graph store generally must **not** evict its own keys).
- **Persistence:** Redis **RDB** snapshots (point-in-time, compact) + **AOF** (append-only, lower data-loss window). Choose per your RPO; back up the RDB/AOF, and know restart replays them into RAM.
- **Replication & HA:** one **primary** takes writes; **read-only replicas** scale reads and serve `GRAPH.RO_QUERY` (async replication → eventual consistency, watch replica lag). **Sentinel** for automatic failover.
- **Clustering/sharding:** **Redis Cluster** distributes *graphs* across shards by hash slot — **each graph lives entirely on one shard**. FalkorDB does **not** split a single graph across shards (no Neo4j-Fabric equivalent). Scale by partitioning many graphs across the cluster (a natural fit for multi-tenant KGs), not by sharding one graph.
- **Ingestion:** client **bulk loaders** (e.g. `falkordb-py`) for greenfield millions of rows; batched `UNWIND` for incremental; idempotent `MERGE` + constraints. Size batches to bound transaction memory.
- **Config & tuning:** `GRAPH.CONFIG` knobs — query thread pool (`THREAD_COUNT`), result-set/query memory caps (`QUERY_MEM_CAPACITY`), `MAX_QUEUED_QUERIES`, `TIMEOUT_DEFAULT`/`TIMEOUT_MAX`, `CACHE_SIZE`. FalkorDB executes queries across a thread pool — size it to cores.
- **Monitoring & security:** `GRAPH.SLOWLOG` for slow queries, `GRAPH.PROFILE` for plans, Redis metrics/`INFO`. Secure with **Redis ACLs** (restrict `GRAPH.*` per user), **TLS** in transit, and network isolation; manage secrets outside the data.
- **Clients & ecosystem:** official SDKs (`falkordb-py`, Node/`falkordb-ts`, Java, Rust, Go) speaking **RESP and Bolt**, the FalkorDB Browser UI, **FalkorDB Cloud**, and GraphRAG tooling (FalkorDB **GraphRAG-SDK**, LangChain/LlamaIndex integrations). The project's pinned client and its API shape are stated in `graph-dba.md` (This deployment).

## GraphRAG / knowledge graphs

- Combine **vector indexes** (semantic recall over embedded text/entities) with **graph traversal** (precise, explainable multi-hop context) for **hybrid retrieval** that beats either alone.
- Exploit **multi-graph multi-tenancy** for per-user/per-document/per-tenant knowledge graphs, isolated and individually fast.
- Reason about embedding storage (`vecf32` properties), similarity function choice (cosine vs. euclidean), chunk-to-entity modeling, and how retrieved subgraphs feed an LLM's context. Use the GraphRAG-SDK where it fits rather than rebuilding the pipeline.
