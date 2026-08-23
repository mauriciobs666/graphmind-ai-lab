# Ingestion Pipeline & Entity Fusion — Graph Design

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** K-050 (M5)

Graph-side design companion to `docs/plans/document-ingestion.md` (architect) and
`docs/plans/document-ingestion-ml.md` (data-scientist, read for the extraction/matching-method
context this note assumes: closed 7-value `Entity.type` enum, deterministic exact-match auto-merge,
no `Entity.embedding` in v1). Settles the plan's §0(a)–(d) delegation. All Cypher shapes below marked
**[verified]** were run live against this lab's `falkordb-dev` instance (`v4.18.11`, module `41811`)
on throwaway probe graphs (`gdba_probe_*`, `gdba_ram_probe`), each deleted after measurement — not
against `reference` or any `ws:*` graph.

> **Revision note (2026-08-22, post plan-gate review).** `docs/reviews/document-ingestion.md` found
> one BLOCKER routed here (the exact-tier check-then-act race in `create_or_reopen_match`'s original
> two-round-trip shape for the FR-8 auto-merge tier), one MAJOR routed here jointly with `architect`
> (the auto-merge tier's missing audit/discovery surface — `list_matches`), and one MINOR routed here
> (the §3.3/§10.3 reverse-read, previously prose-only). All three closed below: §1.8
> (`create_entity_with_auto_match`, new), §1.7 (extended with `list_matches`), §3.3 (completed and
> live-verified). `architect`'s companion revision is `document-ingestion.md` §3.4's "Concurrency
> note" — read it first for the decision rationale (why an atomic query, not a processing-order
> guarantee); this note only covers the resulting Cypher. Nothing else in this document changed.

---

## 0. Summary — where this diverges from the plan

**(a) Diverges.** The plan recommends a reified `MatchSuggestion` node
(`CANDIDATE_A`/`CANDIDATE_B` edges) and explicitly left "does this build support indexed
relationship properties" unverified. **It does** — live-verified below (`Edge By Index Scan`,
`RELATIONSHIP`-scoped `UNIQUE` constraints, both fully operational). With that blocker cleared, I
recommend a plain property-bearing edge instead: `(:Entity)-[:SAME_AS {matchId, status, confidence,
technique, createdAt, decidedAt, decidedBy, resuggestCount, lastResuggestedAt}]->(:Entity)`. Reasons
in §1 — in short: RAM is a wash (measured, not assumed), and the edge model is one hop instead of two
for the traversal this schema exists to eventually support ("given this entity, what did we fuse it
with"), while every access pattern the plan actually needs (matchId-addressed CRUD, global
status-listing) is equally cheap on both shapes once relationship indexing is on the table.

**(b)/(c) Confirm and extend.** Exact Cypher below, all live-verified, including one genuinely new
write shape for this codebase (a guarded find-or-reopen-or-create on an `OPTIONAL MATCH`-ed
relationship variable) and one real planner trap discovered while verifying it (§1.4) that changes
how every `SAME_AS`-touching query in this note is written.

**(d) Agree with the plan.** `extraction` becomes a fifth `ModelGateway` kind. From the graph side
the cost of that decision is exactly zero: `WorkspaceConfig` is a singleton node, and a sixth-now-
seventh nullable scalar property costs no new index, no new constraint, no measurable RAM (§4).

---

## 1. (a) `MatchSuggestion` node vs. `SAME_AS` edge — final schema

### 1.1 The blocker the plan flagged is cleared: relationship-property indexing works

**[verified 2026-08-22, module `41811`]**

```cypher
CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.matchId)
GRAPH.CONSTRAINT CREATE <graph> UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId
CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.status)
```

`db.indexes()` reports `entitytype: RELATIONSHIP` for both; the constraint attaches with no error
following the same index-before-constraint ordering rule as node constraints, going `PENDING` →
`OPERATIONAL` the same way. A query filtering on the indexed relationship property profiles as
**`Edge By Index Scan`** — confirmed for a pattern-property match (`{matchId:$id}`), a `SET` after
one, and a `WHERE`-filtered global scan (`{status:'pending'}`), directed or undirected. This closes
the plan's explicitly-unverified question outright: relationship-property indexing is not a
theoretical alternative on this build, it is a proven, first-class one — this is new information
worth folding into `claude/graph-dba/falkordb-quirks.md` (done, §1.4 below explains the one real trap
that comes with it).

### 1.2 RAM: measured, and it's a wash — not the deciding factor

I did not want to guess here, so I measured both shapes on a throwaway graph (`gdba_ram_probe`),
`INFO memory` delta, 300 shared `Entity` nodes + 1,000 suggestion records with realistic property
values (uuid `matchId`, `status`, `confidence` float, `technique` string, four timestamp/count
fields), same methodology `docs/DESIGN.md` §11 already uses for the `Message` line:

| Model | Shape | Measured bytes/suggestion |
|---|---|---|
| Edge | `(:Entity)-[:SAME_AS {...9 props}]->(:Entity)` | **840** |
| Node | `(:Entity)<-[:CANDIDATE_A]-(:MatchSuggestion {...9 props})-[:CANDIDATE_B]->(:Entity)` | **791** |

My back-of-envelope expectation going in was that the node model costs meaningfully more (extra
node + 2 extra typed relationships). The live number says otherwise — the two are within measurement
noise of each other, node model very slightly cheaper. **RAM is not a valid argument for either
shape on this build; don't cite it either way.** Both are negligible next to the plan's own dominant
line, `Chunk.embedding` (§6).

### 1.3 What actually decides it: hop count on the read this schema exists to serve, and write-path fit

FR-6/§3.1 of the plan is explicit that fusion **never merges edges** — a reader who wants the full
picture on a fused entity must traverse to its confirmed-same siblings and pull their `ABOUT`/
`RELATES_TO` too. That traversal isn't built in M5 (stage 4 only ships suggestion CRUD), but it's the
obvious next read once this ships, and schema is the one thing that's expensive to change after the
fact. Edge model: one hop, direction-agnostic (`(e)-[:SAME_AS {status:'confirmed'}]-(other)`). Node
model: two hops through a `MatchSuggestion` intermediary, with an extra `CANDIDATE_A|CANDIDATE_B`
union pattern on both ends.

The plan's own precedent argument ("mirrors `WorkflowRun.status`") is weaker than it looks once you
compare shapes: `WorkflowRun` is a hub entity with five distinct outgoing edge types to five
different node kinds (`OF_DEF`, `AT_STEP`, `TRIGGERED_BY`, `HAS_STEP_RUN`, `LAST_STEP_RUN`) — a real
independent process record. A match suggestion connects exactly two nodes of the *same* label via
two near-identical edge types. That's structurally `EMITTED {score, rank}` (a property-bearing edge
recording a point-in-time decision between two existing nodes), not `WorkflowRun`. Applying the
`EMITTED` precedent to a case that additionally needs global status-filtering is exactly what §1.1
proves is now safe to do.

**Write-path fit** (verified live, §1.5): the OQ-3 "reuse by pair, reopen if rejected" requirement
maps directly onto this codebase's dominant idiom — an `OPTIONAL MATCH` + guarded `FOREACH(CASE...)`
— with zero new mechanism. It does **not** need a bare `MERGE` (which would need a direction
convention baked into the query itself to avoid missing a suggestion created in the opposite
direction — see §1.5).

### 1.4 Live-verified planner trap: don't label the endpoints of a relationship-index-anchored query

**[verified 2026-08-22, module `41811`, at 1000-row `Entity` cardinality]** This is the one thing
that makes the edge-property choice non-trivial to get right, and it shapes every `SAME_AS` query
below:

```
MATCH (a:Entity)-[r:SAME_AS {matchId:$id}]->(b:Entity) SET r.status = 'confirmed' ...
```
```
Update
  Filter
    Edge By Index Scan | [r:SAME_AS]        Records produced: 1
      Node By Label Scan | (a:Entity)       Records produced: 1000   ← full label scan!
```

Even though `matchId` is uniquely indexed and the edge scan alone is fully selective (1 row), putting
a **bare label with no property predicate** on either endpoint node in the pattern (`a:Entity` and/or
`b:Entity`) makes the planner *additionally* run a full `Node By Label Scan` on that label — at
production `Entity` cardinality (which grows with corpus size, unbounded) this is a real cost, not a
rounding error. Drop the label entirely and the plan collapses to a clean, single-operator scan:

```
MATCH (a)-[r:SAME_AS {matchId:$id}]->(b) SET r.status = 'confirmed' ...
```
```
Update
  Project
    Edge By Index Scan | [r:SAME_AS]        Records produced: 1
```

Omitting the label costs nothing semantically — `SAME_AS` only ever connects `Entity` nodes by
construction, and `a.entityId`/`b.entityId` (or `labels(a)[0]` if ever needed) still read correctly
off the unlabeled pattern variable. **Every query below that anchors on a `SAME_AS` property uses
bare `()`/`(a)`/`(b)`, never `(a:Entity)`, for exactly this reason.** This is now recorded in
`claude/graph-dba/falkordb-quirks.md` as a general finding (any relationship-index-anchored query on
this build, not just this feature).

### 1.5 Final schema

```
(:Entity)-[:SAME_AS {
  matchId, status, confidence, technique,
  createdAt, decidedAt, decidedBy,
  resuggestCount, lastResuggestedAt
}]->(:Entity)
```

- **Direction is a write-time convention, not a semantic claim**: always created
  `(newlyExtractedEntity)-[:SAME_AS]->(existingCandidateEntity)`. Reads that need to be
  direction-agnostic (the find-or-reopen write itself, and any future "expand to fused siblings"
  read) use an **undirected** pattern; reads that always know the direction because they wrote it
  (`list_pending_matches`) use a directed one. Verified: an undirected pattern between two
  **already-bound, id-anchored** nodes (`(a:Entity{entityId:$x})-[r]-(b:Entity{entityId:$y})`)
  returns exactly one row per edge — the "undirected patterns double-count" behavior only bites when
  at least one endpoint is a free/unbound pattern variable (standard Cypher semantics, not a
  FalkorDB-specific quirk; verified live, not restated in `falkordb-quirks.md` for that reason).
- **`status ∈ {pending, confirmed, rejected}`** — same three values, same meaning as the plan's
  original design.
- **`confidence`**: `1.0` for the FR-8 exact/auto tier (a deterministic identity check, not a score —
  see `document-ingestion-ml.md` §4.1); the raw RediSearch relevance score for the FR-9 suggested
  tier, per `document-ingestion-ml.md` §4.3 ("store the raw score... but document what it is not: a
  calibrated probability").
- **`technique`**: implementer-tunable string, e.g. `'exact_normalized_name_type'` /
  `'fuzzy_fulltext'` — illustrative, not mandated.
- **`decidedAt`/`decidedBy`**: set at creation time for the auto tier (`decidedBy:'system'`); `null`
  until `confirm_match`/`reject_match` for the suggested tier — matches the plan's audit intent
  exactly.
- **`resuggestCount`/`lastResuggestedAt`**: OQ-3's automatic-reopen bookkeeping.

**DDL** (index-before-constraint, mirrors `bootstrap_schema.sh`'s existing style):

```bash
echo "[index] SAME_AS.matchId"
gquery "$g" "CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.matchId)"

echo "[index] SAME_AS.status"
gquery "$g" "CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.status)"

echo "[constraint] SAME_AS unique {matchId}"
gconstraint "$g" UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId
```

No new node label, no new `Entity`-side index for this piece.

### 1.6 Write: find-or-create-or-reopen (`create_or_reopen_match`) — **[verified live, exact shape]**

**Post-review scope note:** as of §1.8 below, this write serves the **fuzzy/suggested tier only**
(`status='pending'`) — the exact/auto tier no longer calls it (it's folded into
`create_entity_with_auto_match`, §1.8, to close the plan-gate review's concurrency blocker). Called
once `find_fuzzy_candidates` (§2.3) returns a top-ranked candidate for a newly-extracted entity that
did **not** exact-match (both `Entity` nodes already exist and are id-resolved by the caller before
this runs — the new entity via `create_entity_with_auto_match`, the candidate via
`find_fuzzy_candidates`).

```cypher
// $newEntityId, $candidateEntityId = the two Entity.entityId values (write direction:
//   new -> existing; reads elsewhere don't rely on this direction, see §1.5)
// $matchId    = server-minted uuid; consumed only if a fresh edge is actually created
// $status     = 'confirmed' (FR-8 auto tier) | 'pending' (FR-9 suggested tier)
// $confidence, $technique, $createdAt
MATCH (a:Entity {entityId: $newEntityId})
MATCH (b:Entity {entityId: $candidateEntityId})
OPTIONAL MATCH (a)-[existing:SAME_AS]-(b)
WITH a, b, existing,
     (existing IS NULL) AS isNew,
     (existing IS NOT NULL AND existing.status = 'rejected') AS reopen
FOREACH (_ IN CASE WHEN isNew THEN [1] ELSE [] END |
  CREATE (a)-[:SAME_AS {
    matchId: $matchId, status: $status, confidence: $confidence,
    technique: $technique, createdAt: $createdAt,
    decidedAt: CASE WHEN $status = 'confirmed' THEN $createdAt ELSE null END,
    decidedBy: CASE WHEN $status = 'confirmed' THEN 'system' ELSE null END,
    resuggestCount: 0, lastResuggestedAt: null
  }]->(b)
)
FOREACH (_ IN CASE WHEN reopen THEN [1] ELSE [] END |
  SET existing.status = 'pending',
      existing.resuggestCount = coalesce(existing.resuggestCount, 0) + 1,
      existing.lastResuggestedAt = $createdAt
)
RETURN isNew AS created, reopen AS reopened,
       coalesce(existing.matchId, $matchId) AS matchId,
       coalesce(existing.status, $status)   AS status
```

**Why not a bare `MERGE (a)-[:SAME_AS]->(b)`**: `SAME_AS` is semantically symmetric, but a `MERGE`
with a fixed direction only matches an existing edge written in *that* direction — a suggestion that
happens to exist as `(b)-[:SAME_AS]->(a)` (opposite discovery order on an earlier ingest) would be
invisible to it and get duplicated. The `OPTIONAL MATCH ... undirected` lookup above sidesteps that
by construction (direction-agnostic read, canonical-direction write), and reuses the guarded-`CREATE`
idiom this codebase already uses everywhere over a computed-condition `MERGE` (`docs/QUERIES.md`'s
own "every `MERGE` is backed by a uniqueness constraint, or it's a duplicate-node bug" rule, §9 —
note that a bound-node-to-bound-node relationship `MERGE` between two already id-resolved nodes
*would* actually be safe by a different argument — no fresh node identity is at stake — but the
guarded-`CREATE` shape above is what this codebase already reaches for, and it's what handles the
reopen branch in the same query, so I kept it consistent rather than introducing a second write
idiom for one feature).

**Verified live** (`gdba_probe_ingestion2`): call 1 on a fresh pair → `created=true`; call 2 (same
pair, no change) → `created=false, reopened=false` (idempotent no-op); manually flip to `rejected`,
re-derive same pair → `created=false, reopened=true`, `resuggestCount` bumps to `1` on the
**original** edge (`matchId` unchanged) — no duplicate edge; querying from the reversed argument
order (`candidate` first, `new` second) finds the same edge. All exactly per OQ-3's contract.

### 1.7 Reads

```cypher
// confirm_match(match_id, decided_by, decided_at) — FR-10
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
SET r.status = 'confirmed', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

```cypher
// reject_match(match_id, decided_by, decided_at) — FR-10
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
SET r.status = 'rejected', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

```cypher
// recheck_match(match_id, at) — OQ-3 manual reopen; rejected -> pending only
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
WHERE r.status = 'rejected'
SET r.status = 'pending',
    r.resuggestCount = coalesce(r.resuggestCount, 0) + 1,
    r.lastResuggestedAt = $at
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

```cypher
// list_pending_matches(limit) — OQ-2 review surface. Directed (matches the canonical
// write direction, §1.5) — no undirected-doubling risk, and no label on the endpoints
// (§1.4) even though the SAME_AS.status index alone is the plan's whole point.
MATCH (a)-[r:SAME_AS {status: 'pending'}]->(b)
RETURN r.matchId AS matchId,
       a.entityId AS entityA, a.name AS nameA,
       b.entityId AS entityB, b.name AS nameB,
       r.confidence AS confidence, r.technique AS technique, r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```

All four **[verified]**: `confirm`/`reject`/`recheck`-shaped SET-by-matchId profiles as a clean
`Edge By Index Scan` (no label scan, §1.4); `list_pending_matches`' shape profiled identically clean
at 1,000-`Entity` scale (`gdba_ram_probe`).

**`list_matches(status=None, limit)` — the review's MAJOR finding, added post-review.** Naive
instinct is to parameterize `list_pending_matches`'s literal into an optional filter:
`WHERE $status IS NULL OR r.status = $status`. **Don't** — verified live (`gdba_probe_listmatches`,
1000-`Entity`/300-`SAME_AS` scale) that this exact idiom profiles as `All Node Scan | (a)` (1000
records) → `Conditional Traverse` → `Filter`, **even when `$status` is bound to a real value** —
it silently discards the `SAME_AS.status` index for the common filtered call, not just the
unfiltered one. This is a new instance of the same family as the already-documented `OR`-as-
scan-anchor trap (`falkordb-quirks.md`), now added there as its own entry. **Fix: branch at the
repository layer into two distinct query strings, not one query with a null-guard:**

```cypher
// list_matches(status=<value>, limit) — filtered branch. Same shape as
// list_pending_matches, status is now the parameter instead of a literal. Verified
// [Edge By Index Scan] — identical clean plan to the hardcoded-literal version.
MATCH (a)-[r:SAME_AS {status: $status}]->(b)
RETURN r.matchId AS matchId,
       a.entityId AS entityA, a.name AS nameA,
       b.entityId AS entityB, b.name AS nameB,
       r.status AS status, r.confidence AS confidence, r.technique AS technique,
       r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```

```cypher
// list_matches(status=None, limit) — unfiltered branch. A full scan is genuinely
// unavoidable here — verified there is no relationship-type-only scan operator on
// this build; even fully anonymous, unbound endpoints (`MATCH ()-[r:SAME_AS]->()`)
// still root the plan at `All Node Scan`. Same result shape as the filtered branch,
// no WHERE at all.
MATCH (a)-[r:SAME_AS]->(b)
RETURN r.matchId AS matchId,
       a.entityId AS entityA, a.name AS nameA,
       b.entityId AS entityB, b.name AS nameB,
       r.status AS status, r.confidence AS confidence, r.technique AS technique,
       r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```

This closes the review's finding: `status='confirmed', decidedBy='system'` rows (the auto-merge
tier, previously undiscoverable via `list_pending_matches`'s hardcoded `'pending'` filter) are now
listable via `list_matches(status='confirmed')`, at the same index-anchored cost as the pending-only
listing. The unfiltered call is the one genuinely expensive path here (`All Node Scan` on `Entity`,
which grows unboundedly with corpus size) — reasonable for an infrequent admin/audit "show me
everything" call, not for a hot path; `limit` bounds the *result* size but not the *scan* cost, worth
flagging to the implementer as a rate-of-use assumption, not a hard cap.

---

### 1.8 `create_entity_with_auto_match` — the concurrency-fix atomic query (BLOCKER fix)

**What this closes.** The plan-gate review found the exact tier's original shape — a separate
`find_exact_candidate` read, then `create_entity`, then `create_or_reopen_match` — as three
independent `GRAPH.QUERY` round trips with nothing binding them atomically. Two entities extracted
around the same wall-clock time (same `ingest_documents` batch, or two concurrent MCP calls — this
codebase's real concurrency model per `background.py`, no serialized single-worker queue anywhere)
could each run the candidate read before either sibling's write commits, so **neither** sees the
other and the pair that should auto-merge silently doesn't — the one fusion action with zero
human/agent review (`decidedBy='system'`), so nothing downstream would ever catch the miss.
`architect`'s decision (`document-ingestion.md` §3.4 "Concurrency note"): close it at the database
layer by folding the candidate lookup, the entity `CREATE`, and the conditional auto-link into one
atomic `GRAPH.QUERY` — FalkorDB/Redis serializes command execution, so two concurrent calls against
the same `(nameNormalized, type)` can never interleave; each call's internal read-then-write is
atomic by construction, the same reason `create_or_reopen_match` (§1.6) already didn't need a
separate lock for its own read-then-write. This is not a new technique — it's the existing
guarded-write idiom applied one hop earlier.

**The one thing worth not assuming**, per the coordinator's explicit ask: that folding a `MATCH`
and a `CREATE` into one query string actually guarantees the `MATCH` sees pre-query state only,
never the new node its own later `CREATE` is about to add. Cypher clause order suggests it, but this
codebase has already found genuine same-query-ordering surprises on this build (the map-projection-
as-`CREATE`-endpoint bug, the label-scan-on-relationship-index trap in §1.4) — so this was verified
directly, not assumed.

```cypher
// create_entity_with_auto_match(entity_id, name, name_normalized, type, created_at, match_id)
// $matchId is server-minted; consumed only if a candidate is actually found. Oldest-first
// tie-break matches find_entity_candidates' original exact-tier semantics (§2.3).
OPTIONAL MATCH (candidate:Entity {nameNormalized: $nameNormalized, type: $type})
WITH candidate
ORDER BY candidate.createdAt ASC
LIMIT 1
CREATE (e:Entity {
  entityId: $entityId, name: $name, nameNormalized: $nameNormalized,
  type: $type, createdAt: $createdAt
})
WITH e, candidate
FOREACH (_ IN CASE WHEN candidate IS NOT NULL THEN [1] ELSE [] END |
  CREATE (e)-[:SAME_AS {
    matchId: $matchId, status: 'confirmed', confidence: 1.0,
    technique: 'exact_normalized_name_type', createdAt: $createdAt,
    decidedAt: $createdAt, decidedBy: 'system',
    resuggestCount: 0, lastResuggestedAt: null
  }]->(candidate)
)
RETURN e.entityId AS entityId,
       candidate IS NOT NULL AS exactMatched,
       candidate.entityId AS candidateEntityId,
       CASE WHEN candidate IS NOT NULL THEN $matchId ELSE null END AS matchId
```

**Why no `reopen` branch** (the plan's flagged simplification, confirmed): a brand-new `Entity`
node, `CREATE`d fresh inside this same query, cannot structurally already have a `SAME_AS` edge to
anything — there is no prior state to reopen. `create_or_reopen_match`'s reopen branch exists
because *its* two endpoints are both pre-existing, id-resolved entities that could already carry a
`rejected` edge from an earlier ingestion; that case cannot arise here by construction. Confirmed,
kept the query simpler accordingly.

**Ordering guarantee — verified two ways, not assumed** (`gdba_probe_atomic`):

1. **Black-box behavioral test, the sharpest form of it.** First call against a **brand-new**
   `(nameNormalized, type)` pair with zero pre-existing entities → `exactMatched=false`. If the
   query's `CREATE` were somehow visible to its own `MATCH` (the failure mode worth ruling out), this
   first-ever call would incorrectly self-match and report `true`. It didn't. A second call with the
   same pair then correctly reports `exactMatched=true, candidateEntityId=<first call's id>` — never
   its own just-created node. A third call, with two now-eligible pre-existing candidates, correctly
   picks the **older** one (`createdAt` tie-break), never the newer sibling and never itself. Final
   graph state: 3 `Entity` nodes, exactly 2 `SAME_AS` edges, both pointing at the original (oldest)
   entity — no self-loops, no duplicate/stray edges.
2. **`GRAPH.PROFILE`, structural proof.** The execution plan is one linear pipeline, bottom to top:
   `Node By Index Scan (candidate:Entity)` → `Filter` → `Optional` → `Sort` → `Limit` → **`Create`
   (the new entity)** → `Foreach` (the conditional `SAME_AS` edge). The candidate scan is a fully
   separate, strictly earlier stage of the pipeline — it resolves to exactly one row (real candidate
   or `null`) *before* the `Create` operator for the new entity ever runs. This is the FalkorDB-level
   proof behind the behavioral result above, not just a plausible reading of clause order.

   Note also: `candidate:Entity` (unlike §1.4's endpoints) profiles as a clean `Node By Index Scan`,
   not a label scan — because `candidate` itself carries the selective property predicate
   (`{nameNormalized, type}`) directly, rather than being an unlabeled-selectivity bystander next to
   a relationship-property filter. §1.4's trap is about *where the plan roots when the only real
   predicate lives elsewhere in the pattern*; that's not the shape here.

**Stop-and-ask check, resolved without needing to escalate**: the ordering guarantee held exactly as
`architect`'s spec assumed — no fork to report.

---

## 2. (b) Exact Cypher for the new writes

### 2.1 `Document`/`Chunk`/`HAS_CHUNK` (stage 1)

Per plan §2.4: **no HEAD/TAIL race** — `documentId` is server-minted, so a retried call mints a
second `Document` (accepted, matches the "create channel/thread" precedent, `docs/DESIGN.md` §9's
write-paths table). Because of that, this query carries **no `dup` guard** — deliberately, matching
the plan's explicit non-idempotent-on-retry posture rather than silently hardening past it. The
guard that *does* exist is the actor check (FR-4) — mirrors the message write path's `authorFound`
contract exactly, including the "unknown actor ⇒ nothing written" behavior.

`Document.sourceKind` is **derived server-side from which label resolved the actor** (`'document'`
if `$ingestedBy` is a `User`, `'agent'` if an `Agent`) — same rationale, same "never trust the
caller" posture as `Message.role` (`docs/DESIGN.md` §5.1). No `Document.ingestedBy` property is
stored redundantly — same convention as `Message`, which carries no author-id property either,
only the `POSTED_BY` edge (`docs/QUERIES.md` §4's "Get a single message" traverses it). A new
`INGESTED_BY` edge (`UPPER_SNAKE`, subject → actor, same direction as `POSTED_BY`) is the sole
source of that fact.

```cypher
// create_document — $chunks = [{chunkId, text, seq}, ...], pre-split app-side
// (chunking.split_into_chunks). $documentId/$chunkIds server-minted uuids.
OPTIONAL MATCH (u:User  {userId:  $ingestedBy})
OPTIONAL MATCH (a:Agent {agentId: $ingestedBy})
WITH u, a, coalesce(u, a) AS ingestor, (coalesce(u, a) IS NOT NULL) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (d:Document {
    documentId: $documentId, title: $title, text: $text,
    sourceFormat: $sourceFormat,
    sourceKind: CASE WHEN u IS NOT NULL THEN 'document' ELSE 'agent' END,
    status: 'processing', createdAt: $createdAt
  })
  CREATE (d)-[:INGESTED_BY]->(ingestor)
  FOREACH (ch IN $chunks |
    CREATE (d)-[:HAS_CHUNK]->(:Chunk {
      chunkId: ch.chunkId, text: ch.text, seq: ch.seq, documentId: $documentId
    })
  )
)
RETURN ok AS written, ingestor IS NOT NULL AS ingestorFound
```

**Verified live** (`gdba_probe_docs`): 3-chunk document with a known `User` actor → 1 `Document` +
3 `Chunk` nodes, 1 `INGESTED_BY` + 3 `HAS_CHUNK` edges, all `seq`/`text`/`chunkId` correctly
populated from the `$chunks` list-of-maps parameter (a genuinely new shape for this codebase's
query library — `FOREACH` iterating a **parameter list of maps**, reading `.field` per item inside
a nested `CREATE`; distinct from the existing verified shapes, which only cover `FOREACH` over
**collected node** lists and **map-parameter key lookups** — this combination wasn't proven before).
Unknown-actor case → zero rows written, confirmed via a follow-up `MATCH...count()`.

`get_document(document_id)`:
```cypher
MATCH (d:Document {documentId: $documentId})
OPTIONAL MATCH (d)-[:INGESTED_BY]->(actor)
RETURN d, labels(actor)[0] AS ingestedByKind, coalesce(actor.userId, actor.agentId) AS ingestedById
```

### 2.2 `Entity`/`ABOUT`/`RELATES_TO` (stage 3)

**Post-review note:** `create_entity` below stays the underlying plain-create primitive, but as of
the concurrency fix (§1.8) it is **no longer the entity-creation call site `IngestionPipeline` uses**
— that call site now goes through `create_entity_with_auto_match` (§1.8), which creates the entity
*and* resolves/links the exact-tier candidate in one atomic query. `create_entity` remains correct
and usable standalone (e.g. any future call site that genuinely doesn't need the exact-tier check).

Both `entityId` and every edge here are freshly minted by *this same pipeline run* moments apart
(the `Chunk` from stage 1, the `Entity` a line earlier in stage 3) — a `MATCH` miss would indicate a
real bug, not a routine 4xx-worthy caller input, so unlike `Document`/message writes these don't
carry a guard-and-status-row contract. Plain `CREATE`s, matching the "server-minted id, no HEAD/TAIL
race" precedent.

```cypher
// create_entity — always a NEW node; fusion (§1) never blocks creation, only decides
// linking after the fact. $nameNormalized = case-folded + whitespace-collapsed $name,
// computed app-side by the SAME normalization convention document-ingestion-ml.md §3.2
// already specifies for extraction's own subject/object stub-repair — one shared helper,
// not two independent normalizers drifting apart.
CREATE (e:Entity {
  entityId: $entityId, name: $name, nameNormalized: $nameNormalized,
  type: $type, createdAt: $createdAt
})
RETURN e.entityId AS entityId
```

```cypher
// link_chunk_about_entity — plain CREATE, never MERGE. A chunk is extracted exactly
// once per ingestion, so a duplicate (chunk, entity) ABOUT pair shouldn't occur under
// normal operation; if it ever did, an extra co-occurrence edge is harmless (no
// properties to conflict), same "never deduplicated" posture the plan already commits
// to for RELATES_TO (§3.1), applied here by the same reasoning.
MATCH (c:Chunk {chunkId: $chunkId})
MATCH (e:Entity {entityId: $entityId})
CREATE (c)-[:ABOUT]->(e)
```

```cypher
// create_entity_relationship — the RELATES_TO fact edge. label = the LLM-extracted
// predicate, free text (never its own relationship type — plan §3.3). Never
// deduplicated: every extracted fact is independent provenance (FR-6/§3.1).
MATCH (subj:Entity {entityId: $subjectId})
MATCH (obj:Entity  {entityId: $objectId})
CREATE (subj)-[:RELATES_TO {
  label: $label, sourceChunkId: $sourceChunkId,
  sourceDocumentId: $sourceDocumentId, createdAt: $createdAt
}]->(obj)
```

**No DDL for `ABOUT` or `RELATES_TO`** — both are always traversed *from* an already-anchored
`Chunk`/`Entity`, never independently scanned by their own properties. Exactly the `EMITTED`/
`MENTIONS_MEMBER` precedent (`docs/DESIGN.md` §9/QUERIES §10: "no relationship-property index is
needed... FalkorDB traverses the typed edge from the anchored [node] via its adjacency matrix").

### 2.3 `Entity.name` full-text index + `Entity.nameNormalized` + candidate lookup

**Why two properties, not one.** The plan specifies a fuzzy full-text index on `Entity.name` for the
suggested tier (§2.5, mirroring `Message.text`) but leaves the FR-8 exact-tier mechanism unspecified.
Running the exact-tier check *through* the same RediSearch fulltext index (e.g., treating "fuzzy
score happens to be the max possible" as "exact") would make a deterministic identity check depend
on a search engine's tokenizer/stemmer/stopword behavior — exactly the kind of unvalidated,
non-deterministic mechanism `document-ingestion-ml.md` §4.1 argues against for the one fusion action
with no human review. A second, plain, RANGE-indexed `nameNormalized` property gives a real `=`
comparison, decoupled from RediSearch scoring semantics entirely, at negligible extra cost (one
short string property, one RANGE index — no constraint, since distinct real entities can and do
share a normalized name+type before fusion runs).

**Sequencing note for the implementer:** set `nameNormalized` on every `Entity` starting in **stage
3** (§2.2 above already does), even though fusion doesn't read it until **stage 4** — this avoids a
backfill migration when the fusion stage lands one PR later.

```bash
# ── Entity fulltext + normalized-name lookup (K-050) ──
echo "[fulltext] Entity.name"
gquery "$g" "CALL db.idx.fulltext.createNodeIndex('Entity', 'name')"

echo "[index] Entity.nameNormalized"
gquery "$g" "CREATE INDEX FOR (n:Entity) ON (n.nameNormalized)"
```

**Post-review update: the exact-tier query below is superseded, kept only for its semantics.** The
original design called this as a standalone `find_exact_candidate` read, separate from entity
creation — exactly the shape the plan-gate review found racy (§1.8). As of the concurrency fix, the
identical `MATCH ... ORDER BY createdAt ASC LIMIT 1` logic is embedded **inside**
`create_entity_with_auto_match` (§1.8's `OPTIONAL MATCH (candidate:Entity {...})` block) — it is
**not** called as a separate query at runtime any more. Shown here once more, standalone, only
because it's the clearest statement of the tie-break semantics `create_entity_with_auto_match`
implements — do not add a separate `find_exact_candidate` repository method that duplicates it.

```cypher
// exact-tier semantics (FR-8) — embedded in create_entity_with_auto_match (§1.8), not
// a standalone call. Deterministic, index-anchored, no RediSearch involved. Multiple
// pre-existing entities can share (nameNormalized, type) if they were never fused —
// oldest-first is a reasonable, implementer-tunable tie-break, not load-bearing.
MATCH (e:Entity {nameNormalized: $normalizedName, type: $type})
RETURN e.entityId AS entityId, e.name AS name, e.type AS type
ORDER BY e.createdAt ASC
LIMIT 1
```

`find_fuzzy_candidates` — the suggested tier, still a genuinely separate read (unaffected by the
concurrency fix, §1.6's post-review scope note):

```cypher
// suggested tier (FR-9) — RediSearch fuzzy full-text. $fuzzyQuery is built app-side,
// one RediSearch fuzzy term per name token: '%acme%' (1-edit fuzzy) or '%%acme%%'
// (2-edit) per docs.falkordb.com's RediSearch syntax — both forms verified live on
// this build (a typo'd query term still matched 'Acme Corporation').
CALL db.idx.fulltext.queryNodes('Entity', $fuzzyQuery) YIELD node AS candidate, score
WHERE candidate.type = $type
RETURN candidate.entityId AS entityId, candidate.name AS name, candidate.type AS type, score
ORDER BY score DESC
LIMIT $limit
```

The `type` filter on the fuzzy tier is a routine judgment call, not plan-mandated: avoids a fuzzy
name hit surfacing a nonsensical cross-type suggestion (e.g. the organization "Acme" fuzzy-matching
a product also named "Acme"). Cheap — it's a post-`YIELD` filter over an already-small candidate set.

### 2.4 `Chunk.seq`/`Chunk.documentId` — confirmed, no DDL

Agree with the plan: both stay plain, unindexed properties. `seq` is read-time ordering only, always
reached by traversing from an already-anchored `Document`/already-matched `Chunk` — never a lookup
key. `documentId` is navigation metadata for reporting a search hit's source without a traversal —
the exact `Message.threadId` precedent (`docs/DESIGN.md` §5.1: "navigation metadata, not an anchor").
No change to `bootstrap_schema.sh` needed for either.

---

## 3. (c) Generalizing `EMITTED` provenance to `Chunk` seeds

### 3.1 The resolution shape: bare-id `coalesce`, not a namespaced id scheme

The plan flags two options: a `kind` field per seed, or a namespaced id scheme (`msg:<uuid>` /
`chunk:<uuid>`). **Recommendation: neither — resolve `$seedIds` against both `Message.msgId` and
`Chunk.chunkId` directly, via the exact two-label `OPTIONAL MATCH` + `coalesce` idiom this query
already uses twice** (once for author resolution, once for mention resolution, `docs/QUERIES.md`
§4/§10.1). This works because both id spaces are server-minted `uuid4` values from disjoint
generators — collision odds are the same "astronomically negligible" already accepted for
`User.userId`/`Agent.agentId` sharing one namespace (`docs/QUERIES.md` §2: "member ids are
namespace-unique across `User`/`Agent`" — a locked rule). Extending that same locked convention to
`Message.msgId`/`Chunk.chunkId` needs no new mechanism, no string-manipulation risk (a namespaced
scheme would need stripping a prefix before the actual id-property `MATCH`, adding surface for a
subtle bug into a write query already carrying real complexity), and no parallel-list zipping (a
`kind` field would need `$seedIds`/`$seedKinds` UNWIND'd together, which this build has no clean
idiom for beyond index-zipping).

### 3.2 Write — generalized seed resolution inside the guarded `EMITTED` write

One-line change to `docs/QUERIES.md` §10.1's existing seed-resolution block:

```cypher
// was:
// OPTIONAL MATCH (s:Message {msgId: sid})
// becomes:
UNWIND (CASE WHEN $seedIds = [] THEN [null] ELSE $seedIds END) AS sid
OPTIONAL MATCH (sm:Message {msgId: sid})
OPTIONAL MATCH (sc:Chunk {chunkId: sid})
WITH ..., collect(DISTINCT coalesce(sm, sc)) AS seeds
```

Everything downstream (the guarded `FOREACH (seed IN seeds | CREATE (m)-[:EMITTED {...}]->(seed))`)
is unchanged — `seed` is still a bound node variable, `CREATE` doesn't care which label it carries.
**Per-edge property map keys change from `$scoreBy[seed.msgId]` to `$scoreBy[coalesce(seed.msgId,
seed.chunkId)]`** (and same for `$rankBy`) — exactly one of the two properties is non-null per node,
so `coalesce` picks the right key regardless of seed type; the caller (`services.hybrid_search`'s
merge step) already has to key its score/rank maps by *some* id per seed item, so this changes
nothing about how that dict gets built, only which property FalkorDB reads back off the node.

**Verified live** (`gdba_probe_emitted`): one answer `EMITTED` to one `Message` seed and one `Chunk`
seed in a single guarded write — both edges created correctly, no row multiplication from the
`UNWIND`, no interference with the mention block's own `UNWIND` (the "two sequential guarded
`UNWIND`s don't row-multiply" quirk, already documented, holds for a third block too).

### 3.3 Read — label dispatch + the `Chunk → Document` provenance hop (AC-5)

```cypher
// generalized version of docs/QUERIES.md §10.2 (forward — "what did this answer cite")
MATCH (a:Message {msgId: $msgId})-[e:EMITTED]->(s)
OPTIONAL MATCH (s)<-[:HAS_CHUNK]-(d:Document)
RETURN labels(s)[0]              AS seedKind,       // 'Message' | 'Chunk'
       coalesce(s.msgId, s.chunkId) AS seedId,
       s.text                    AS text,
       s.role                    AS role,           // null when seedKind = 'Chunk'
       d.documentId              AS documentId,     // null when seedKind = 'Message'
       d.title                   AS documentTitle,
       e.score, e.rank
ORDER BY e.rank
```

`s` is deliberately **unlabeled** in the pattern — it's discovered by traversal from the
index-anchored `a`, not independently searched, so there's no `Node By Label Scan` risk here (unlike
§1.4's trap, which is specifically about a node being **searched via its own label** alongside an
indexed relationship property; this is a plain traversal endpoint, the existing precedent this
codebase already uses via `labels(coalesce(a,b))[0]` for author/mention resolution). The
`OPTIONAL MATCH (s)<-[:HAS_CHUNK]-(d:Document)` only ever matches when `s` is a `Chunk` (a `Message`
has no incoming `HAS_CHUNK` edge) — this is what satisfies AC-5's "traverses back through
`(:Chunk)<-[:HAS_CHUNK]-(:Document)`" requirement, folded into the same read, one extra hop.

**Verified live** (`gdba_probe_emitted2`): a `Message`-seeded row returns `seedKind:'Message'`,
`role:'user'`, `documentId:null`; a `Chunk`-seeded row returns `seedKind:'Chunk'`, `role:null`,
`documentId`/`documentTitle` correctly populated from the extra hop. `GRAPH.PROFILE`:
`Node By Index Scan | (a:Message)` → `Conditional Traverse (a)-[:EMITTED]->(s)` →
`Optional Conditional Traverse (d)` — clean, no label scan, matches the existing §10.2 profile shape
with one added hop.

### 3.4 `docs/QUERIES.md` §10.3 (reverse-read — "which answers cited this seed")

Flagged MINOR by the plan-gate review: the previous draft only described this in prose ("needs the
mirror-image change"). Completed and live-verified to the same standard as §3.2/§3.3:

```cypher
// read_citing_answers($seedId) — generalized §10.3. Resolve the anchor via the same
// two-label coalesce as the write (§3.2) and forward-read (§3.3) — bare id, not a
// namespaced scheme, same locked-convention argument as §3.1.
OPTIONAL MATCH (sm:Message {msgId: $seedId})
OPTIONAL MATCH (sc:Chunk {chunkId: $seedId})
WITH coalesce(sm, sc) AS s
MATCH (a:Message)-[e:EMITTED]->(s)
RETURN a.msgId AS msgId, a.role AS role, a.createdAt AS createdAt,
       e.score AS score, e.rank AS rank
ORDER BY a.createdAt DESC
```

**Verified live** (`gdba_probe_1103`), including the specific question the review's finding didn't
get to: whether `a:Message`'s label triggers §1.4's planner trap here too, now that `a` is the
**unbound** endpoint (traversed *to*, not *from*) and `s` is the one resolved by the earlier
`OPTIONAL MATCH` pair. Populated 1,000 candidate `Message` answer nodes all citing the same `Chunk`
seed (the worst case for a label-scan risk on `a`) plus one citing a `Message` seed. Both profiled
identically clean: `Node By Index Scan (sm:Message)` / `Node By Index Scan (sc:Chunk)` (exactly one
produces a row per call) → `Conditional Traverse` from the resolved `s` outward to `a` → `Project` →
`Sort` — **no `Node By Label Scan` at any point**, despite `a` carrying a label. This is the
opposite shape from §1.4's trap: there, *two* pattern endpoints were unbound with only a
relationship property supplying selectivity, and the planner rooted on one endpoint's label instead
of the relationship index. Here `s` already has its own strong index-anchor (resolved by the
`OPTIONAL MATCH` pair one step earlier, exactly like §3.3's forward-read anchors on `a:Message
{msgId:$msgId}`), so `a`'s label costs nothing — it's a pure traversal-discovery target, not a
competing anchor candidate. Data correctness also confirmed: the `Chunk`-seed call returned all
1,000 citing answers correctly ordered `createdAt DESC`; the `Message`-seed call returned exactly
the one correct citing answer.

---

## 4. (d) `ModelGateway` fifth kind: `extraction` — agree, zero graph cost

Agree with the plan: `extraction` is a genuinely distinct workload from `step` (single-shot
structured-JSON generation vs. an agentic multi-turn loop inside a workflow), and coupling its model
choice to the executor's `step` override would force an operator who wants a cheap/fast model for
bulk extraction to also redirect real workflow-step reasoning to it, or vice versa — an unwanted
coupling `document-ingestion.md` §2.3 already flags. From the schema side there is nothing to weigh
this against: `WorkspaceConfig` (`docs/DESIGN.md` §14.8, `docs/QUERIES.md` §13) is a single-row
singleton per workspace with **zero new index/constraint per additional override property** — the
existing `workspaceConfigId` index/constraint already covers the whole node, exactly as it already
does for the four current override columns. Adding `extractionModelOverride` costs one more nullable
scalar, no new DDL line, no measurable RAM (§13.3's own framing already covers this: "One node per
workspace, six scalar properties... below `GRAPH.MEMORY USAGE`'s 1 MB resolution" — a seventh scalar
doesn't change that assessment).

**Property name**: `extractionModelOverride` — follows the `<kind>ModelOverride` pattern the way
`guardModelOverride`/`embeddingModelOverride` already do (not the two renamed exceptions,
`agentModelOverride`↔`step` and `responderModelOverride`↔`agent`, which exist only because those two
kind strings collided with pre-existing naming — `extraction` has no such collision, so it gets the
straightforward name). Add the crosswalk entry at `server/falkorchat/modelconfig.py`'s
`_KIND_TO_OVERRIDE_KEY` (code-level, not graph-level — noted here only so it isn't missed).

```cypher
// write_model_overrides — widened, one added SET term
MERGE (c:WorkspaceConfig {workspaceConfigId: 'default'})
SET c.agentModelOverride      = $agent,
    c.guardModelOverride      = $guard,
    c.embeddingModelOverride  = $embedding,
    c.responderModelOverride  = $responder,
    c.extractionModelOverride = $extraction,
    c.modelOverrideUpdatedAt  = $at,
    c.modelOverrideUpdatedBy  = $by
RETURN c.agentModelOverride AS agent, c.guardModelOverride AS guard,
       c.embeddingModelOverride AS embedding, c.responderModelOverride AS responder,
       c.extractionModelOverride AS extraction
```

```cypher
// read_model_overrides — widened, one added RETURN term
MATCH (c:WorkspaceConfig {workspaceConfigId: 'default'})
RETURN c.agentModelOverride      AS agentModel,
       c.guardModelOverride      AS guardModel,
       c.embeddingModelOverride  AS embeddingModel,
       c.responderModelOverride  AS responderModel,
       c.extractionModelOverride AS extractionModel
```

`bootstrap_schema.sh` needs **no change** for this piece — no new index line, no new constraint line.

---

## 5. DDL summary (net new vs. `bootstrap_schema.sh` today)

```bash
# ── Entity fulltext + normalized-name (K-050, §2.3) ──
gquery "$g" "CALL db.idx.fulltext.createNodeIndex('Entity', 'name')"
gquery "$g" "CREATE INDEX FOR (n:Entity) ON (n.nameNormalized)"

# ── SAME_AS relationship-scoped indexes + constraint (K-050, §1.5) ──
gquery "$g" "CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.matchId)"
gquery "$g" "CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.status)"
gconstraint "$g" UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId
```

Everything else this feature needs (`Document`/`Chunk`/`Entity` node indexes+constraints,
`Chunk.embedding` vector index) already exists in `bootstrap_schema.sh` since M2 — confirmed by
reading it, not changed here (plan §2.1 already established this; scope boundary for this note is
design, not editing the script).

**Post-review: no DDL delta from either fix.** `create_entity_with_auto_match` (§1.8) reuses the
existing `Entity.nameNormalized` index and `SAME_AS.matchId`/`SAME_AS.status` indexes verbatim — it
changes *how* the exact-tier lookup is called (folded into one atomic query), not what it's indexed
by. `list_matches` (§1.7) reuses `SAME_AS.status` too. The DDL block above is unchanged from the
first pass.

---

## 6. RAM/scale (rule 6)

- **`Chunk.embedding`** stays the dominant new line — not re-derived here, see the plan's own §6
  estimate (`docs/DESIGN.md` §11's ~12.4 KB/vector at 1024 dims, applied per-chunk).
- **`SAME_AS` (§1.2, measured)**: ~840 bytes/suggestion. Suggestion count is bounded by the plan's
  own per-chunk extraction cap (§3.3, ≤20 entities/chunk) times however often a candidate is actually
  found — realistically well under one suggestion per extracted entity. Even 100k suggestions ≈
  84 MB, a rounding error against the vector line.
- **`Entity.nameNormalized`**: one short string property + one RANGE index entry per `Entity` —
  same order of magnitude as any other id-shaped property already on the node, negligible.
- **`Entity.name` fulltext index**: real RediSearch inverted-index overhead, structurally identical
  to the existing (never separately measured) `Message.text` fulltext line — this codebase has never
  broken that cost out on its own either (`docs/DESIGN.md` §11 folds it into the aggregate per-message
  line). Flagging rather than fabricating a number: measure via `INFO memory` delta once real entity
  volume exists, same posture as the plan's own "re-measure `Chunk.embedding` with real ingestion
  volume" action item (§6).
- **`Document.text` duplicating `Chunk.text`**: plan already calls this out (§6, ~2× raw text size,
  negligible against vectors) — nothing to add.
- **Supernode watch, extended**: `docs/DESIGN.md` §5.4 already flags `Entity` fan-out via `MENTIONS`
  as a risk to re-evaluate with `GRAPH.PROFILE` once real data lands. `ABOUT`, `RELATES_TO`, and now
  `SAME_AS` are three more edge types that can accumulate on a frequently-mentioned/frequently-fused
  `Entity` (a common name like a well-known company). No new mitigation proposed here beyond the
  existing note — same "re-evaluate with real data" posture, now with one more edge type to include
  in that future pass.
- **`WorkspaceConfig.extractionModelOverride`**: zero (§4).

---

## 7. Handoff notes for implementation

- The `gdba_probe_*`/`gdba_ram_probe` throwaway graphs used for live verification — across both this
  pass and the plan-gate-review fix-up pass (`gdba_probe_atomic`, `gdba_probe_listmatches`,
  `gdba_probe_listmatches2`, `gdba_probe_1103`, plus the originals) — were all deleted at the end of
  each check (`GRAPH.DELETE`). Nothing left behind on the shared instance.
- Four live-verified, build-specific facts are now in `claude/graph-dba/falkordb-quirks.md` (dated
  2026-08-22): relationship-property indexing support (§1.1 here), the label-on-relationship-endpoint
  planner trap (§1.4 here), RediSearch fuzzy-term syntax (§2.3 here), and — added this round — the
  `$param IS NULL OR prop = $param` optional-filter trap (§1.7 here). Read all four before writing any
  *other* `SAME_AS`-shaped, relationship-index-anchored, or optionally-filtered listing query in this
  codebase.
- **The exact-tier candidate lookup is no longer a separate repository call** (post-review, §1.8) —
  it's embedded inside `create_entity_with_auto_match`. Do not reintroduce a standalone
  `find_exact_candidate` method that duplicates that logic; `find_fuzzy_candidates` (§2.3, unaffected
  by the fix) remains the one genuinely separate read, kept apart from the exact tier's mechanism so
  `document-ingestion-ml.md` §4.1/§4.3's "don't conflate the two tiers" framing stays true at the
  call-site level, not just in prose.
- **`list_matches` must be two query strings in `repository.py`, not one parameterized query** —
  §1.7's verified finding: an `IS NULL OR` guard silently drops the `SAME_AS.status` index even on
  the filtered call. Branch on `status is None` in Python, not in Cypher.
- `nameNormalized`'s normalization function should be the **same** one `extraction.py`'s
  subject/object stub-repair uses (`document-ingestion-ml.md` §3.2: "case-fold + whitespace-collapse
  compare") — one shared helper, not two independently-written normalizers that can drift.
