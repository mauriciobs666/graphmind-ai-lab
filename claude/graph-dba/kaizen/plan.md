# Kaizen — Improvement Plan: graph-dba

> Forward-looking backlog for the `graph-dba` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-28 (joern-agent merge; last full pass 2026-07-11, team-coherence certification)

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-005 | 2026-07-28 | med | 🔵 | Streaming loader for large-repo CPGs — `joern-cpg`'s transformer dedupes in memory, fine for moderate repos but a risk at repo scale (inherited from the retired `joern` agent's K-003) |
| K-006 | 2026-07-28 | low | 🔵 | CPGQL script library (`skills/joern-cpg/scripts/queries/*.sc`) for common security/taint/call-graph queries (inherited from `joern` K-004) |

> K-005/K-006 inherited 2026-07-28 from the retired `joern` agent when its CPG-generation
> capability folded into this agent (see history.md). K-001/K-002 deferred (below), K-003
> done, K-004 done 2026-07-11 (design-note handoff contract — see history.md).

> K-001 and K-002 deferred (below), K-003 done, K-004 done (2026-07-11, same-day — design-note
> contract + destructive-ops guard; the guard also answers K-001's revisit trigger with a
> destructive-shapes-only gate rather than a tool allowlist).

> Done: K-003 (2026-06-05) — deployment identified (edge `graph` build on Redis 8 + `vectorset`); details below and in history.md.
> Deferred: K-001, K-002 (2026-06-05) — documentation-only for now; keep tools unconstrained and the agent advice-only. Revisit triggers below.

### K-001 — Tool permissions decision  ⚪ DEFERRED (2026-06-05)
- **Status:** ⚪ deferred — user chose "just document for now."
- **Decision:** No `tools` key; the agent keeps inheriting all tools (matches `tdd-engineer`'s deliberate choice). The read-mostly allowlist (`Read, Grep, Glob, WebFetch, WebSearch`, ± `Write/Edit`) was considered and declined for now.
- **Revisit if:** broad tool access causes surprise/unwanted actions, or the agent starts mutating live FalkorDB data in ways that warrant a guardrail.

### K-002 — Companion "live FalkorDB" skill  ⚪ DEFERRED (2026-06-05)
- **Status:** ⚪ deferred — user chose "just document for now."
- **Rationale:** Much DBA value comes from actually running `GRAPH.PROFILE`/`GRAPH.EXPLAIN` against a real instance, and a live FalkorDB exists (edge build on Redis 8). A progressive-disclosure skill documenting how to connect (`redis-cli`, `falkordb-py`), run profiling, and capture plans would make tuning advice concrete — but not being built yet.
- **Proposed change (when revived):** Scope a `.claude/skills/falkordb-profiling/` skill.
- **Revisit if:** the user wants the agent to tune against real plans rather than stay advice-only.

### K-003 — Verify dialect specifics against the installed version  ✅ DONE (2026-06-05)
- **Status:** ✅ done — see history.md.
- **Outcome:** "1.6.0" = `falkordb-py` **client** (pinned 1.6.x). `redis-cli MODULE LIST` showed the **`graph` engine reporting `999999`** — FalkorDB's **edge/untagged build** sentinel (not a tagged `v4.x`), tracking latest `main` — running on **Redis 8.x**, with the standalone **`vectorset`** (Redis Vector Sets) module also loaded. Because it's an edge build, exact-version pinning isn't meaningful; the prompt instead documents: assume newest documented behavior but **verify + test against the live instance**, the Redis 8 base, and the **in-graph vector index vs. Redis Vector Sets** choice for GraphRAG. The observed module args (`MAX_QUEUED_QUERIES=25`, `TIMEOUT=1000`, `RESULTSET_SIZE=10000`) confirmed real config knob names (legacy `TIMEOUT` rather than newer `TIMEOUT_DEFAULT`/`TIMEOUT_MAX`).
- **Revisit if:** the deployment moves to a tagged `v4.x` release — then spot-check `GRAPH.*`/dialect specifics against that exact version.

### K-005 — Streaming loader for large-repo CPGs
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** `skills/joern-cpg/scripts/cpg-to-falkordb.py` collects all nodes/edges into
  memory to dedup before emitting Cypher. Fine at the scale exercised so far (a real
  41-file Python subtree: 110k nodes / 735k edges, well within RAM) but a full-repo CPG
  (millions of AST/CFG/REACHING_DEF edges) could exhaust it.
- **Proposed change:** stream per-file with an on-disk/rocksdb-style seen-id set, or dedup
  via `MERGE` at load time; benchmark against a genuinely large real repo first rather than
  guessing the threshold.

### K-006 — CPGQL script library
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** common Joern REPL queries (dangerous-sink reachability, unsanitized-input
  flow, call-chain to a function) get rewritten each time CPG generation is invoked.
- **Proposed change:** add `skills/joern-cpg/scripts/queries/*.sc` runnable via
  `joern --script`, referenced from the skill's CPGQL cheat-sheet.

## Parking lot / ideas
- **The agent owns one recurring `Status: archived` flip it isn't told about yet (noted 2026-07-27).** Root `AGENTS.md`'s routing table makes `graph-dba` the performer for `plans/<slug>-graph.md` at milestone close, on `teco`'s coordination; today that reaches the agent only through the closing unit's brief. Zero `-graph.md` files exist so far, so there is nothing to fix yet — revisit once the first one ships.
- If another project in the lab (or a future one) accumulates its own "live-verified FalkorDB
  facts" against this same edge build, fold the generic ones into the `falkordb-quirks.md`
  knowledge base rather than letting them sit siloed in that project's docs; keep only the
  project-specific corollaries in that project's `AGENTS.md`, pointing back here (2026-07-05).
- On any FalkorDB tagged-release upgrade (edge → `v4.x`), re-verify every entry in
  `falkordb-quirks.md` against the live instance and re-stamp its `Verified:` date; retire any
  quirk the new build fixes (2026-07-05).
- Add a concrete `GRAPH.PROFILE` operator cheat-sheet (label scan vs. index scan, cartesian product, dense expansion → matrix-density reasoning) — possibly as a skill rather than bloating the always-loaded prompt.
- ~~Consider whether `opus` is warranted vs. `sonnet` for routine query help~~ — **resolved 2026-07-27**: the `model` pin was removed team-wide; the agent inherits the session/system default.
- Neo4j/openCypher/GQL portability is currently kept for *porting models into FalkorDB*; deepen only if the lab targets multiple engines.
- RedisGraph migration note: FalkorDB is the drop-in successor — could add explicit migration guidance if any legacy RedisGraph data is in play.
