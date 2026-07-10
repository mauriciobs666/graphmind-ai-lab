# Kaizen — Change History: graph-dba

> Dated log of actual changes to the `graph-dba` agent. Most recent first.

## 2026-07-09 — Deployment pinned to v4.18.11 (edge retired)
- **What:** The lab's FalkorDB moved from `falkordb/falkordb:edge` (module `999999`) to the tagged release **`v4.18.11`** (module `41811`, Redis 8.6.3, released 2026-06-24). Rewrote the "This deployment" bullet (pinned release, reason from v4.18.11's documented behavior instead of moving-target/latest-`main` caveats; `vectorset` still loaded) and updated the quirks-section pointer. Re-stamped `falkordb-quirks.md`'s header: pinned build identified, quirks re-verified via the falkor-chat query suite (193/193 green on the new build); entries not exercised by the suite keep their edge-build dates pending individual re-probes. Catalog current-state refs updated (root `AGENTS.md`, `claude/AGENTS.md`, falkor-chat docs).
- **Why:** User decided to pin the latest release (cost/verification churn of tracking edge; the prompt's verify-live posture existed largely because the build was a moving target). The quirks file's own rule — re-verify on any tagged-release upgrade — was executed via the canonical suite.
- **Plan items:** none.

## 2026-07-09 — devops boundary clause (description + ops bullet)
- **What:** Frontmatter `description` and a new "Architecture & operations" bullet state the split with `devops`: graph-dba *designs* the deployment (RAM sizing, persistence choice, replication/cluster topology, ACLs) and owns everything inside the database; the container/Compose plumbing that runs it (service bring-up, volumes, networking, CI wiring) routes to `devops` — mirroring devops's existing deferral of data-model/query design here. The pair is mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry). Catalogs synced (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): "spin up FalkorDB" plausibly matched both agents, and only devops's description named the boundary.
- **Plan items:** none.

## 2026-07-09 — Subagent-awareness on "ask one sharp question" (teco interface review follow-up)
- **What:** "How you work" step 1's "ask one sharp question" now carries the delegated-run fallback: when running as a subagent (e.g. delegated by `teco`), return the sharp question as the result instead of trying to ask mid-run — subagents can't ask. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** Sweep after the 2026-07-09 teco interface review found the "ask" phrasing assumed an interactive session across several delegates (same fix applied to coder, tdd-engineer, qa-engineer the same day).
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-07-05 — Absorbed generic FalkorDB engine quirks from falkor-chat/AGENTS.md
- **What:** `falkor-chat/AGENTS.md` had a "Live-verified FalkorDB facts" section mixing generic
  engine/dialect quirks (vector index DDL, index-before-constraint ordering, composite
  constraints, cross-graph edge no-op, union-label syntax, `length(path)` in ORDER BY, fulltext +
  `algo.*` confirmation, `GRAPH.RO_QUERY`/Bolt port, `TIMEOUT` default + write-path behavior,
  empty-`UNWIND` row collapse, the `FOREACH(CASE...)` idiom, the `exists()` pattern bug,
  `OR`-as-scan-anchor tuning, `GRAPH.MEMORY USAGE` under-reporting, `labels(coalesce())[0]`
  subscripting) with falkor-chat-specific corollaries (repository function names, mention
  write-block internals, keyset predicate profiling), generalized away from falkor-chat's specific
  property/label names. `falkor-chat/AGENTS.md` was trimmed to keep only the project-specific
  corollaries, each pointing back here for the general fact.
- **Mechanism (revised same day):** first draft inlined the ~20 quirks as a "Verified engine
  quirks" subsection in `graph-dba.md`; on review that bloats the always-on prompt with a
  *perishable, growing* fact list. Split instead into a **resource file** —
  `claude/graph-dba/falkordb-quirks.md` — modeled on the `agent-standards` skill's discipline
  (dated verification stamp, "cache not source of truth," re-verify on tagged-release upgrade,
  build sentinel `999999`). `graph-dba.md` keeps only a short stable-framing pointer that tells the
  agent to read the KB before writing/debugging Cypher/DDL/ops against this build. The whole agent
  folder is symlinked into `~/.claude/agents/graph-dba`, so the sibling file is reachable at both
  the repo path and `~/.claude/agents/graph-dba/falkordb-quirks.md`. `falkor-chat/AGENTS.md`'s
  back-reference was repointed from the prompt section to the resource file.
- **Why:** User: "the section ## Live-verified FalkorDB facts should be part of
  ../claude/graph-dba" — these are reusable DBA knowledge for *any* project on this FalkorDB
  build, not just falkor-chat, and belong on the agent so other projects benefit too. Resource-file
  form (not inline, not a shared skill) was the user's explicit call: keeps the prompt lean and the
  KB in the agent's own folder as a growing, curated store.
- **Plan items:** —

## 2026-06-05 — Deferred K-001 & K-002 (documentation-only for now)
- **What:** No agent/prompt change. User said "just document for now," so recorded the decision: **K-001** (tool permissions) → keep tools unconstrained, no `tools` key; **K-002** (live-FalkorDB profiling skill) → not building it yet, agent stays advice-only. Both marked ⚪ deferred with revisit triggers; active backlog is now empty.
- **Why:** User chose the documentation-only path rather than building tooling or restricting permissions. Logged so the items aren't re-proposed.
- **Plan items:** K-001 ⚪ deferred, K-002 ⚪ deferred.

## 2026-06-05 — Identified the deployment (edge engine on Redis 8 + Vector Sets); closed K-003
- **What:** User ran `redis-cli MODULE LIST` / `GRAPH.QUERY`. Findings: the **`graph` module reports version `999999`** = FalkorDB's **edge/untagged build** sentinel (a tagged release encodes as an integer, e.g. `41809` = v4.18.9), so the engine tracks latest `main`. It runs on **Redis 8.x**, evidenced by the separately-loaded **`vectorset`** module = **Redis Vector Sets** (`VADD`/`VSIM`), confirmed via redis.io docs. Module args observed: `MAX_QUEUED_QUERIES=25`, `TIMEOUT=1000`, `RESULTSET_SIZE=10000`. Edits to the agent: expanded the "This deployment" note (edge build → assume newest but verify + test live; Redis 8 base; `vectorset` present) and added a GraphRAG bullet distinguishing **FalkorDB's in-graph vector index** (`db.idx.vector.*` over `vecf32`, fuses with traversal — default for hybrid retrieval) from **standalone Redis Vector Sets** (`vectorset`/`VADD`/`VSIM`, not traversable — only when embeddings needn't live on the graph).
- **Why:** Closes K-003. An edge build can't be pinned to a semver, so the right move is to record the deployment reality and lean on verify-and-test rather than a release's notes. The dual vector stores on one box are a real GraphRAG footgun worth disambiguating.
- **Plan items:** K-003 ✅ (done). Active backlog now: K-001 (tool permissions, open), K-002 (optional live-FalkorDB skill, low).

## 2026-06-05 — Pinned the falkordb-py client + added version-line literacy (K-003 partial)
- **What:** User answered "version is 1.6.0." Verified via PyPI that this is the **`falkordb-py` Python client** (1.6.0 = 2026-02-21; 1.6.1 latest), not an engine version — the FalkorDB **module/server is on a separate `v4.x` line** (v4.18.9 as of 2026-06). Edited the agent's "Clients & ecosystem" bullet to pin the project's client at **`falkordb-py` 1.6.x** (with the `FalkorDB(...) → select_graph → query/ro_query` API shape and RESP+Bolt), and added a new bullet **"Mind the two version lines"** so the agent never conflates a client version with an engine version and reasons about dialect from the engine (v4.x) but client code from the SDK (1.6.x).
- **Why:** "1.6.0" is exactly the trap that makes an agent assume a wrong engine version; the dialect specifics it encodes are governed by the engine line, not the client. The doc-verified dialect details remain valid for current FalkorDB.
- **Plan items:** K-003 → 🟡 in-progress (client identified/pinned; remaining: confirm the deployed engine v4.x version and reconcile `GRAPH.*`/dialect specifics).

## 2026-06-05 — Repivoted from Neo4j-first to FalkorDB-first (major overhaul)
- **What:** Rewrote the agent to specialize in **FalkorDB** instead of Neo4j, after the user confirmed the lab uses FalkorDB. Verified specifics against docs.falkordb.com (two web searches + two doc fetches) before writing. Changes: new `description` (FalkorDB/Redis-module/GraphBLAS/GraphRAG triggers); added a **"What makes FalkorDB different"** section (sparse-matrix/GraphBLAS traversal as matrix multiplication, in-memory RAM-bound sizing, Redis-module ops model, multi-graph multi-tenancy, OpenCypher *subset* with no APOC/GDS/Fabric, `GRAPH.*` command surface). Reworked all core-expertise sections: modeling (added matrix-aware supernode reasoning + one-graph-per-tenant guidance); **Cypher on FalkorDB** (OpenCypher dialect, `GRAPH.QUERY`/`GRAPH.RO_QUERY`, `GRAPH.EXPLAIN`/`GRAPH.PROFILE` instead of Neo4j `PROFILE` prefix, built-in `algo.*` procedures replacing GDS, batched `UNWIND` writes); **indexing & constraints** (range/full-text `db.idx.fulltext.*`/vector `db.idx.vector.*`, `GRAPH.CONSTRAINT` unique/mandatory); **architecture & operations** (RAM sizing first, RDB/AOF persistence, primary/read-replica async replication, Redis Cluster with *graph-per-shard* and no single-graph sharding, Sentinel, `GRAPH.CONFIG`/`THREAD_COUNT`, `GRAPH.SLOWLOG`, Redis ACL/TLS security, SDK/Cloud ecosystem); and a dedicated **GraphRAG/knowledge-graphs** section (vector+graph hybrid retrieval, multi-tenant KGs, GraphRAG-SDK). Updated working method, principles, and communication style to FalkorDB realities. Kept LPG cross-awareness (Neo4j/openCypher/GQL) for porting and the RDF/SPARQL boundary.
- **Why:** User: "we will use falkordb not neo4j, please review everything." Almost every Neo4j-specific claim (APOC, GDS, Fabric, causal cluster/Raft, `neo4j-admin import`, page cache) was wrong for FalkorDB and had to be replaced.
- **Plan items:** reframes K-002 (companion skill now FalkorDB-specific: `redis-cli`/`GRAPH.PROFILE`) and promotes GraphRAG from parking-lot idea to a core section. Updated README.md and CLAUDE.md catalogs.

## 2026-06-05 — Dropped tenure-boast framing
- **What:** Removed "with decades of hands-on experience running graph databases in production" from the opening line; it now reads "You are a **graph database administrator and data architect** who runs graph databases in production." Kept the role label (sets altitude) but cut the tenure brag.
- **Why:** User feedback — the "decades of experience" framing makes agents sound cocky and adds nothing to behavior. Applied collection-wide (also tdd-engineer, dra-claudia).
- **Plan items:** —

## 2026-06-05 — Agent created
- **What:** Initial authoring of the `graph-dba` agent — a senior graph database administrator and data architect. Frontmatter `name: graph-dba`, `model: opus`, and a routing-oriented `description` with proactive-use triggers (design a graph model, write/optimize Cypher/GQL, plan cluster architecture/sizing/sharding, set up indexes/constraints, tune slow traversals, plan migrations/imports, ops questions). Body covers four core-expertise areas (graph data modeling; Cypher/GQL mastery; indexing & constraints; architecture & operations incl. GraphRAG/vector), a six-step working method (access-patterns-first, match existing conventions, show the model concretely, justify by traversal cost, prove perf via PROFILE, respect engine/version/edition boundaries), seven principles, and a communication style. Scoped **Neo4j/Cypher-first** but explicitly aware of the wider LPG world (openCypher, ISO GQL, Memgraph, Neptune) and honest about the RDF/SPARQL boundary.
- **Why:** User asked for a new agent: "graph database administrator who knows cypher query, data modeling, architecture and best practices." Fits the lab's focus (repo `graphmind-ai-lab`).
- **Plan items:** seeded K-001 (tool-permissions decision), K-002 (optional live-Cypher companion skill), plus parking-lot ideas (GraphRAG depth, PROFILE operator cheat-sheet, opus-vs-sonnet, multi-engine portability).

## 2026-06-05 — Docs updated (discoverability)
- **What:** Registered `graph-dba` in the collection catalog `claude/README.md` (table row + kaizen index link) and in the agent-context file `claude/CLAUDE.md` (Agents list).
- **Why:** Dual-audience documentation rule — keep humans (README) and other agents (CLAUDE.md) in sync the moment the agent is created.
- **Plan items:** —
