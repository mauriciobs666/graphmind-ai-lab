# Kaizen — Improvement Plan: graph-dba

> Forward-looking backlog for the `graph-dba` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-08 (K-009 opened from analyst chunk F / U24; prior review
> 2026-09-07, kaizen_team distillation, U6 of pass 2 — 9 current-shape entries:
> 6 promoted to falkordb-quirks.md (one a merged/corrected refinement pair), 1 promoted into
> qa-engineer's knowledge base, 2 kept open as K-008 for remit reasons; see history.md. Prior
> pass 2026-08-25 (U10). K-007 opened 2026-08-18; last full certification pass 2026-07-11;
> joern-agent merge 2026-07-28)

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-005 | 2026-07-28 | med | 🔵 | Streaming loader for large-repo CPGs — `joern-cpg`'s transformer dedupes in memory, fine for moderate repos but a risk at repo scale (inherited from the retired `joern` agent's K-003) |
| K-006 | 2026-07-28 | low | 🔵 | CPGQL script library (`skills/joern-cpg/scripts/queries/*.sc`) for common security/taint/call-graph queries (inherited from `joern` K-004) |
| K-007 | 2026-08-18 | low | 🔵 | Unreconciled relationship-count discrepancy on a scoped `DETACH DELETE` of a workflow-snapshot subgraph (34 deleted vs. ~15 expected) — investigate if it recurs |
| K-008 | 2026-09-07 | med | 🔵 | Two CPG-freshness facts from U6, overtaken by `6012ddb`: one delivered, one superseded — `graph-dba` to confirm, then close or re-scope |
| K-009 | 2026-09-08 | high | 🔵 | `pipeline.sh`'s `rq()` returns **0** on a bare FalkorDB runtime-error reply — a failed query reads as success at any call site with no expected-substring argument |

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

### K-008 — Two CPG-freshness facts from U6, overtaken by the provenance rewrite
- **Status:** 🔵 proposed — **`graph-dba` to confirm the two dispositions below, then close or
  re-scope.** Both facts were verified true when this item was opened (U6, 2026-09-07). `6012ddb`
  has since rewritten the very mechanism they describe: it delivered one and falsified the
  other's premise. Neither is a doc edit to apply as written.
- **Priority:** medium
- **Fact 1 (raw entry `b701038b-f723-4c8c-81b8-c981c1819de9`) — superseded; do not apply.** As
  opened, this held that a parse root could be both `.venv`-pruned and a real git working tree:
  stage the pruned copy inside the repo under a gitignored path
  (`cpg/.cpg-artifacts/src/<name>`) and `pipeline.sh` would resolve `SOURCE_COMMIT`/`SOURCE_DIRTY`
  where a `/tmp` copy could not. That only ever worked because the pre-`6012ddb` stamp inherited
  the *containing* repo's `HEAD` for an untracked copy — which was itself the defect `6012ddb`
  removed. Capture now returns non-zero for such a copy and the build stamps `PROVENANCE=none`
  with a warning in the first seconds of the run. `--source-origin <the tracked directory the copy
  was staged from>` is the supported route, and `skills/joern-cpg/SKILL.md`'s "No `--exclude`"
  bullet already says exactly that. Writing the fact as opened would re-document behaviour the fix
  deliberately removed.
- **Fact 2 (raw entry `4f1976fc-b7e0-4509-a274-c3994dbb7083`) — delivered; nothing to apply.** As
  opened, `pipeline.sh` ran `git -C "$SRC" status --porcelain` with no pathspec, so unrelated
  untracked or modified files anywhere in the repo stamped `SOURCE_DIRTY=true` over a parsed
  source that was clean at `HEAD`. `git-provenance.sh:105` now scopes that check with a
  `:(literal)` pathspec, and `skills/cpg-analysis/references/freshness.md` already documents
  `sourceDirty` as scoped — "it says nothing about the rest of the repo".
- **Notes:** each raw entry was cleared from `kaizen_team` in the pass that opened this item, so
  this item is their durable record; the `entryId`s are quoted above so a later distillation pass
  grepping this file finds them.
- 2026-09-09 (`cobb`, U26): both dispositions above re-derived by execution against the current
  tree, while distilling `f3c1a27e-9b64-4d18-a5e2-7c0b91d4e8a3`.

### K-009 — `pipeline.sh`'s `rq()` treats a bare runtime-error reply as success
- **Status:** 🔵 proposed (kept open from the `kaizen_team` distillation pass U24, 2026-09-08 —
  **confirmed by execution, twice, independently**; the blocker is write remit, not doubt)
- **Priority:** high — it is a silent-pass in a guard, in the shipped pipeline
- **The defect.** `skills/joern-cpg/scripts/pipeline.sh`'s `rq()` helper classifies failure by
  matching a prefix set at **line 304**:
  `errMsg:*|ERR\ *|WRONGTYPE*|*"read only"*|*"read-only"*`. FalkorDB returns some runtime errors
  **bare**, with no prefix at all, and `redis-cli` exits 0 for every error reply — so `rq()`
  returns 0 on them. A call site that passes no expected-substring third argument therefore treats
  a failed query as a success. Extracted verbatim and pointed at a live graph, 2026-09-08:

  | probe | reply | `rq()` |
  |---|---|---|
  | `RETURN (((` | `errMsg: Invalid input …` | **1** (caught) |
  | `RETURN nosuchfunc(1)` | `Unknown function 'nosuchfunc'` | **0** (missed) |
  | `MATCH (n:KaizenEntry) RETURN keys(n.fact)` | `Type mismatch: …` | **0** (missed) |
  | `UNWIND [1,0] AS x RETURN 1/x AS stray` | `Division by zero` | **0** (missed) |
  | `MATCH (n:Agent) RETURN count(n)` (control) | header + trailer | 0 (correct) |

- **Why this is a reopening, not a new bug.** `9124a1f` closed the *discarded-output* half of the
  same trap (the stamp write was `redis-cli … >/dev/null`, so a failed stamp was invisible after a
  multi-hour build) and replaced it with this prefix `case` — which reopens the identical
  silent-pass class in a new shape.
- **Proposed change:** classify **affirmatively**, not by prefix. A successful `GRAPH.RO_QUERY`
  reply carries a column header plus the `Query internal execution time:` trailer; every error
  reply — parse, runtime, mid-stream abort, timeout — is a single bare line with neither. Requiring
  the trailer makes the negative "no stray rows" assertion fail-closed. Keep any prefix `case` as a
  courtesy message, never as the check. (Both the general behaviour and this affirmative
  discriminator are already published in `claude/graph-dba/falkordb-quirks.md`, "Ops, config &
  tooling" — this item is the **code** fix, not a doc gap.)
- **Notes:** raw entries `b7f3c2a1-9d4e-4c11-8a52-6e0f1d3b7c94` (the prefix taxonomy) and
  `4f9c21ae-7b30-4d62-9c18-6ea5d0b73c41` (the mid-stream abort that makes the trailer test sound),
  both produced by `analyst`. Their `PRODUCED` edges were resolved in U24; both nodes stay **alive**
  in `kaizen_team` on a `MENTIONS`→`graph-dba` edge, so this agent's own distillation pass meets
  them again. `entryId`s are quoted here so a later pass grepping this file finds them.

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
