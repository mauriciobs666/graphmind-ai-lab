# Kaizen — Improvement Plan: graph-dba

> Forward-looking backlog for the `graph-dba` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-05 (absorbed falkor-chat's verified-quirks section)

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| _(none active)_ | | | | User chose "just document for now" — K-001 and K-002 deferred (below), K-003 done. |

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

## Parking lot / ideas
- If another project in the lab (or a future one) accumulates its own "live-verified FalkorDB
  facts" against this same edge build, fold the generic ones into the `falkordb-quirks.md`
  knowledge base rather than letting them sit siloed in that project's docs; keep only the
  project-specific corollaries in that project's `AGENTS.md`, pointing back here (2026-07-05).
- On any FalkorDB tagged-release upgrade (edge → `v4.x`), re-verify every entry in
  `falkordb-quirks.md` against the live instance and re-stamp its `Verified:` date; retire any
  quirk the new build fixes (2026-07-05).
- Add a concrete `GRAPH.PROFILE` operator cheat-sheet (label scan vs. index scan, cartesian product, dense expansion → matrix-density reasoning) — possibly as a skill rather than bloating the always-loaded prompt.
- Consider whether `opus` is warranted vs. `sonnet` for routine query help (matches the collection's opus default for now).
- Neo4j/openCypher/GQL portability is currently kept for *porting models into FalkorDB*; deepen only if the lab targets multiple engines.
- RedisGraph migration note: FalkorDB is the drop-in successor — could add explicit migration guidance if any legacy RedisGraph data is in play.
