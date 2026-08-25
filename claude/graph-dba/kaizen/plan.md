# Kaizen — Improvement Plan: graph-dba

> Forward-looking backlog for the `graph-dba` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-18 (K-007 opened from a kept-open kaizen-graph distillation entry;
> last full pass 2026-07-11, team-coherence certification; joern-agent merge 2026-07-28)

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-005 | 2026-07-28 | med | 🔵 | Streaming loader for large-repo CPGs — `joern-cpg`'s transformer dedupes in memory, fine for moderate repos but a risk at repo scale (inherited from the retired `joern` agent's K-003) |
| K-006 | 2026-07-28 | low | 🔵 | CPGQL script library (`skills/joern-cpg/scripts/queries/*.sc`) for common security/taint/call-graph queries (inherited from `joern` K-004) |
| K-007 | 2026-08-18 | low | 🔵 | Unreconciled relationship-count discrepancy on a scoped `DETACH DELETE` of a workflow-snapshot subgraph (34 deleted vs. ~15 expected) — investigate if it recurs |

### K-001 — Tool permissions decision  ⚪ DEFERRED (2026-06-05)
- **Status:** ⚪ deferred — user chose "just document for now."
- **Decision:** No `tools` key; the agent keeps inheriting all tools (matches `tdd-engineer`'s deliberate choice). The read-mostly allowlist (`Read, Grep, Glob, WebFetch, WebSearch`, ± `Write/Edit`) was considered and declined for now.
- **Revisit if:** broad tool access causes surprise/unwanted actions, or the agent starts mutating live FalkorDB data in ways that warrant a guardrail.

### K-002 — Companion "live FalkorDB" skill  ⚪ DEFERRED (2026-06-05)
- **Status:** ⚪ deferred — user chose "just document for now."
- **Rationale:** Much DBA value comes from actually running `GRAPH.PROFILE`/`GRAPH.EXPLAIN` against a real instance, and a live FalkorDB exists (edge build on Redis 8). A progressive-disclosure skill documenting how to connect (`redis-cli`, `falkordb-py`), run profiling, and capture plans would make tuning advice concrete — but not being built yet.
- **Proposed change (when revived):** Scope a `.claude/skills/falkordb-profiling/` skill.
- **Revisit if:** the user wants the agent to tune against real plans rather than stay advice-only.

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

### K-007 — Unreconciled `DETACH DELETE` relationship-count discrepancy
- **Status:** 🔵 proposed (opened from a kept-open kaizen-graph entry, cobb distillation
  pass 2026-08-18)
- **Priority:** low
- **Rationale:** a 2026-08-16 scoped `DETACH DELETE` of `ws:acme`'s `triage@v1` snapshot
  (1 `WorkflowDefSnapshot` + 3 `Step` nodes, 6 structural edges + 7 `OF_DEF` + ≤2 `AT_STEP`
  ≈ 15 relationships expected) reported **34** relationships deleted. Follow-up scoped
  queries confirmed the deletion was otherwise correctly scoped (0 remaining
  `triage:v1:*` `Step`s, the sibling `access-request@v1` snapshot and its 14 `OF_DEF`-linked
  runs untouched) — this is a discrepancy in the *count*, not evidence of an incorrect blast
  radius — but the extra ~19 relationships were never reconciled (the data was already gone
  by the time the count was noticed).
- **Proposed change:** next time a similar scoped `DETACH DELETE` is run on this schema,
  count relationships on the target subgraph *before* deleting (`OPTIONAL MATCH (n) WHERE n
  IN [...] -[e]-() RETURN count(DISTINCT e)`) rather than trusting structural-edge
  arithmetic, and/or check whether `advance_run`'s `AT_STEP`/`LAST_STEP_RUN` FOREACH-guarded
  writes (`repository.py`) can leave stale edges on nominally-terminal runs under some race.
- **Notes:** kept open (not promoted, not discarded) during the 2026-08-18 kaizen-graph
  distillation pass — `graph-dba` itself flagged the source entry `unsure`, and `cobb`
  could not independently verify it (the pre-delete state is gone and the anomaly isn't
  reproducible without a live repro). The raw `:KaizenEntry` also stays live in
  `kaizen_graph_dba` (entryId `6e5d6451-72fa-400c-b002-52757727f805`) alongside this backlog
  item, in case a future occurrence supplies the missing pre-delete count.

## Parking lot / ideas
- **Judged and kept, do not re-litigate (2026-08-24, C6 lint).** Five passages will read as class-6/7
  waste to a future sweep; all are keeps.
  - **The single-shard-per-graph rule, in "FalkorDB fundamentals" and again in "Principles."** The
    fundamental carries the **mechanism** (Redis Cluster distributes whole graphs across shards,
    never splits one); the principle carries the **action** (estimate up front, watch it in
    production). Two further mentions — step 4's trade-off list and the communication-style flag
    list — are **checklists of what to raise with the user**, a third function, not a third
    statement of the rule.
  - **Verify-against-`docs.falkordb.com`, stated three times** (fundamentals' "never assume
    Neo4j-only syntax works", step 6's version-sensitive check, communication style's "never
    present a fabricated function… as fact"). Three decision points: while writing Cypher, while
    judging a version-gated feature, while reporting. Finding 5's shape.
  - **`(successor to RedisGraph)`** — not lineage trivia but a live **anti-trigger**: it is what
    makes RedisGraph-era documentation and Stack Overflow answers legible as applicable.
  - **"mirroring the data-scientist's `-ml.md` convention"** (step 7) and **"(Mirrors its deferral
    of data-model/query design to you.)"** (the `devops` boundary) — both make a *reciprocal*
    contract visible from this side, which is what `agent-maintenance` §4's check-5 boundary
    reciprocity reads.
  - **"Both also resolve at `~/.claude/agents/graph-dba/` via the deployment symlink."** Mechanism:
    the knowledge-base links above it are repo-relative, and this is the fallback resolution path.
- **This file has the lowest total residual of the eleven measured (2026-08-24, C6).** ~20 w, against
  a band of 20–40. Not because it is leaner per rule but because its top layer is *reference
  mechanism* with no workflow counterpart re-aiming the same rules at a second altitude. Nothing to
  do — recorded so a future sweep reads a low number as structural, not as a missed opportunity.
- **The agent owns one recurring `Status: archived` flip it isn't told about yet (noted 2026-07-27).** Root `AGENTS.md`'s routing table makes `graph-dba` the performer for `plans/<slug>-graph.md` at milestone close, on `teco`'s coordination; today that reaches the agent only through the closing unit's brief. Zero `-graph.md` files exist so far, so there is nothing to fix yet — revisit once the first one ships.
- If another project in the lab (or a future one) accumulates its own "live-verified FalkorDB
  facts" against this same edge build, fold the generic ones into the `falkordb-quirks.md`
  knowledge base rather than letting them sit siloed in that project's docs; keep only the
  project-specific corollaries in that project's `AGENTS.md`, pointing back here (2026-07-05).
- On any FalkorDB tagged-release upgrade (edge → `v4.x`), re-verify every entry in
  `falkordb-quirks.md` against the live instance and re-stamp its `Verified:` date; retire any
  quirk the new build fixes (2026-07-05).
- Add a concrete `GRAPH.PROFILE` operator cheat-sheet (label scan vs. index scan, cartesian product, dense expansion → matrix-density reasoning) — possibly as a skill rather than bloating the always-loaded prompt.
- Neo4j/openCypher/GQL portability is currently kept for *porting models into FalkorDB*; deepen only if the lab targets multiple engines.
- RedisGraph migration note: FalkorDB is the drop-in successor — could add explicit migration guidance if any legacy RedisGraph data is in play.
