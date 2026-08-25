# Ingestion Pipeline & Entity Fusion — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-050 (M5)

Turns `docs/requirements/document-ingestion.md` (FR-1..FR-14, AC-1..AC-10, OQ-1..OQ-3) into an
ordered, staged build. This plan resolves OQ-2 and OQ-3 as concrete design decisions and makes a
concrete recommendation for OQ-1, but the two axes the requirements doc explicitly named as
outside architect-only territory — the **extraction technique** (FR-7a) and the **matching
technique/threshold** (FR-7/OQ-1) — were delegated to `data-scientist` for a method note, and the
**graph schema/Cypher** for the new node/edge shapes this design introduces was delegated to
`graph-dba`. **Both notes have now landed** (`docs/plans/document-ingestion-ml.md`,
`docs/plans/document-ingestion-graph.md`) and this plan has been reconciled against them — see §0.

## 0. Delegation summary — both notes landed, reconciled

| Follow-on note | Owner | What it settled |
|---|---|---|
| `docs/plans/document-ingestion-ml.md` | `data-scientist` | **No schema-affecting divergence from this plan's original defaults.** Confirmed: LLM-based extraction over a structured JSON schema, reusing the K-027-proven fence-tolerant parse helper (`llm.extract_own_line_json_object`); the FR-8/OQ-1 "very-high confidence" default (exact normalized-name+type match, no calibrated numeric threshold — §3.4). Refinements added, not requiring a plan edit: a closed 7-value `Entity.type` taxonomy; app-side schema validation the parser helper doesn't itself provide; a stub-entity repair rule for dangling relationship references (subject/object names that don't resolve to an already-extracted entity in the same chunk); embedding-based semantic matching explicitly **deferred to a scoped v2**, not a v1 precondition (§7). |
| `docs/plans/document-ingestion-graph.md` | `graph-dba` | **One real divergence, reconciled into this plan (§3.4).** This plan's original §3.4 recommended a reified `MatchSuggestion` node (`CANDIDATE_A`/`CANDIDATE_B` edges) and explicitly left "does this build support indexed relationship properties" unverified — the premise the node-model recommendation leaned on. `graph-dba` live-verified that it **does** (`Edge By Index Scan`, working `RELATIONSHIP`-scoped `UNIQUE` constraints) and, with that blocker cleared, recommends a plain property-bearing edge instead: `(:Entity)-[:SAME_AS {matchId, status, confidence, technique, createdAt, decidedAt, decidedBy, resuggestCount, lastResuggestedAt}]->(:Entity)` — RAM measured as a wash between the two shapes (not assumed), decided instead on hop count for the eventual "expand to fused siblings" read and write-path fit, plus a live-verified planner trap (a bare label on a `SAME_AS`-anchored query endpoint forces a full label scan) that shapes every query touching it. **Adopted throughout this plan — see §3.4 for the full reconciliation note.** Also confirmed, unchanged: `Document`/`Chunk`/`Entity`/`RELATES_TO` write shapes (§2.1/§2.2), the `Entity.name` fulltext index plus a new `Entity.nameNormalized` RANGE-indexed property for the deterministic exact tier (§2.3/§2.5/§3.4), the generalized two-label `EMITTED` seed resolution for FR-2 (§3.7), and the fifth `ModelGateway` kind, `extraction` (§2.3) — "zero graph cost," per graph-dba's §4. |
| (future) `docs/plans/document-ingestion-coordination.md` | `teco` | Sequencing/gating log once implementation starts — not authored here. |

---

## 1. Goal & scope

**Goal.** Build an ingestion pipeline that takes external text-based content (documents, and
agent-generated text treated identically) and turns it into fused, GraphRAG-usable graph
knowledge: chunked for retrieval, entity/relationship-extracted, and fused against what the graph
already knows at three confidence tiers (auto-merge / suggested-pending / confirm-or-reject, with
rejection reversible). The result is retrievable both through the existing chat-grounding path and
as a standalone knowledge base, and writable by a connected MCP agent as persistent memory.

**In scope:** FR-1..FR-14, AC-1..AC-10, resolving OQ-2/OQ-3 and proposing OQ-1's default.

**Out of scope** (per the requirements doc): binary/non-text formats (PDF, images, Office docs);
a unified chat+knowledge-base search index (FR-14 — the two stay separate capabilities); UI work
(no `web/` changes are planned here — a future item, not blocking M5 green, mirroring how M3
shipped before M3.5's web coverage).

---

## 2. Context & findings

**CPG:** considered, not relevant — this is new-code design for `falkor-chat/server`, not an
impact analysis over existing call graphs; the coordinator's brief already flagged the loaded
`cpg_falkorchat` as stale-but-uninvolved (no ingestion-related commits since its build) and
recommended reading the current tree directly, which is what this plan is grounded in.

### 2.1 The dormant schema this feature populates (`docs/DESIGN.md` §5.1, §7.1)

```
(:Document {documentId})-[:HAS_CHUNK]->(:Chunk {chunkId, text, embedding: vecf32})
(:Chunk)-[:DERIVED_FROM]->(:Message)          // NOT this feature's edge — see below
(:Entity {entityId, name, type})<-[:MENTIONS]-(:Message)   // NOT this feature's edge — see below
(:Chunk)-[:ABOUT]->(:Entity)                  // THIS feature populates this one
```

`Document`/`Chunk`/`Entity` already carry a range index + UNIQUE constraint on `{label}Id` in
every workspace graph (`scripts/bootstrap_schema.sh:97-104,184-191`), and `Chunk.embedding` already
has a live `CREATE VECTOR INDEX` at the workspace's configured dimension
(`bootstrap_schema.sh:239-240`). **No DDL changes are needed for the base `Document`/`Chunk`/
`Entity` node shapes or the `Chunk` vector index** — this feature is finishing wiring that was
bootstrapped since M2 and never populated (`docs/QUERIES.md:472` calls it out explicitly: "`Chunk`
is bootstrapped DDL, never populated").

**Two edges in the dormant schema are *not* this feature's concern**, and the plan does not
populate them: `(:Chunk)-[:DERIVED_FROM]->(:Message)` (a *message*-derived chunk provenance —
relevant only if chat messages themselves are ever chunked, which is not part of this feature) and
`(:Entity)<-[:MENTIONS]-(:Message)` (message-level entity mentions — `docs/BACKLOG.md`'s K-008 note
already recorded this as "OUT OF SCOPE for M2... parked, M3-adjacent" and it stays parked here too;
this feature's entities are mentioned by **`Chunk`**, via `ABOUT`, not by `Message`, via `MENTIONS`).
Do not conflate the two `Entity`-pointing edges — see §4.6 for how retrieval nonetheless reaches
both without merging them.

### 2.2 Layering and where the pipeline lands (`docs/DESIGN.md` §14.2, `falkor-chat/AGENTS.md`)

`repository.py` is the only place Cypher lives (1:1 with `QUERIES.md`); `services.py` owns
invariants and id/timestamp generation; `api.py`/`mcp.py` are thin adapters over the same
`Services` methods (`mcp.py:1-20`); background/out-of-band work is scheduled through
`background.py`'s `_safe_*` wrappers, one per transport-neutral policy
(`background.py:1-13`, e.g. `_safe_embed`). This plan's new pipeline follows the same shape: a new
`IngestionPipeline` component (peer of `AgentResponder`/`EmbeddingWorker`/`WorkflowExecutor`),
injected with a `ModelGateway` and the repository via `Services`, invoked synchronously for the
fast/deterministic part (Document + Chunk creation) and asynchronously for the LLM-bearing part
(extraction, fusion, embedding) — mirroring the existing "embed messages: async worker, decoupled
from the write path" row in `docs/DESIGN.md` §9's write-paths table.

### 2.3 The model-resolution seam already exists and should be reused (`docs/DESIGN.md` §14.8)

Every LLM/embedding consumer resolves through one `ModelGateway` seam (K-042); four closed kinds
exist today (`agent`, `step`, `embedding`, `guard`). Extraction is a genuinely distinct LLM
workload (structured-output generation, not chat or a workflow step), so this plan recommends
adding a fifth kind, `extraction`, with its own per-kind default in `config/models.json` and its
own `WorkspaceConfig` override property — done **deliberately**, per the AGENTS.md note that a
fifth kind either gets its own override property or silently escapes FR-17's hard-cap enforcement.
Final call and the exact `WorkspaceConfig`/`ResolvedModel` change: `graph-dba`'s `-graph` note.

### 2.4 The message write-path discipline this pipeline must (and needn't) match

`docs/DESIGN.md` §5.3/§9 and `QUERIES.md` §4: every write touching a `HEAD`/`TAIL` pointer is one
atomic guarded `CREATE` (never a conditional `MERGE`), because concurrent posters race for the same
tail pointer. **Document/Chunk creation has no such race** — a document is a fresh, self-contained
subtree with no shared mutable pointer to fight over — so it is closer to the "create channel /
thread" precedent (`docs/DESIGN.md` §9's write-paths table: "plain `CREATE` (server-minted uuid
ids)... **Non-idempotent** — a retried create mints a new id; the uniqueness constraints backstop").
This plan adopts the same posture for `ingest_document`: a retried call (e.g. after a network
timeout) mints a second `Document`. This is a deliberate, lower-stakes trade-off, not an oversight —
noted as an accepted risk in §7, not a blocker.

### 2.5 RediSearch fuzzy full-text is already a proven pattern in this codebase

`Message.text` already has a full-text index (`db.idx.fulltext.createNodeIndex('Message', 'text')`,
`bootstrap_schema.sh:216`) queried via `db.idx.fulltext.queryNodes` (`QUERIES.md` §5). RediSearch
(the module backing it) supports fuzzy term matching syntax out of the box. This plan's recommended
entity-matching default (§3.4) reuses this exact mechanism against a new `Entity.name` full-text
index for the **suggested tier only** — a materially cheaper RAM line than adding
`Entity.embedding`, discussed in §6. **Narrowed post-graph-dba-reconciliation:** the FR-8 exact
tier does **not** also run through this fulltext index — `graph-dba`'s note gives it its own
deterministic mechanism instead, a plain RANGE-indexed `Entity.nameNormalized` property compared
with `=` (`document-ingestion-graph.md` §2.3), reasoning that routing a deterministic identity
check through a search engine's tokenizer/stemmer/stopword behavior would make the one fusion
action with no human review depend on non-deterministic scoring semantics. See §3.4 for the tier
mechanics.

---

## 3. Design & rationale

### 3.1 Two orthogonal axes, stated up front (this is the key design insight)

The requirements doc's FR-6 (keep both conflicting facts, always) and FR-7..FR-10 (fuse *entities*
at three confidence tiers) are **not the same axis**, and conflating them is the most likely design
mistake here:

- **Entity identity fusion** (FR-7/8/9/10) decides whether two extracted entity mentions refer to
  the same real-world thing — this **may** merge/link **nodes**.
- **Fact/relationship provenance** (FR-6) **never merges edges** — every extracted relationship
  fact is written as its own edge, carrying its own source (chunk/document) and timestamp, forever.
  Even after two entities are confirmed as the same, their previously-attached facts are not
  deduplicated or reconciled — a reader traverses both entities' fact edges and weighs them.

This falls directly out of never physically merging `Entity` nodes (§4.3) — since nothing is ever
migrated or deleted, FR-6 is true *by construction*, not by a separate "keep-both" code path.

### 3.2 Chunking (FR-13)

**Recommendation (routine judgment call, not delegated):**

- **Target chunk size:** 1,000 characters (~200–250 tokens for English/Portuguese prose) — a
  standard RAG default, small enough that `search_documents` (§4.6) surfaces one relevant passage
  rather than a whole document, matching FR-13's stated rationale.
- **Overlap:** 150 characters (~15%) — carries enough trailing context across a chunk boundary that
  a fact split mid-sentence is still extractable from at least one chunk.
- **Boundary rule, in priority order:** (1) prefer splitting on paragraph breaks (`\n\n`); (2) if a
  single paragraph exceeds the target size, split on sentence boundaries (`. `, `! `, `? `, or a
  bare `\n`); (3) if a single "sentence" still exceeds the target size (e.g. a large embedded
  JSON/CSV blob, a long Mermaid diagram block — real cases per FR-1's format list), hard-cut at the
  target size and carry the overlap forward regardless of boundary.
- **Why not format-aware chunking (e.g. Markdown-heading-aware, CSV-row-aware) for v1:** FR-13
  explicitly records chunking as "not fixed here," and the format list (plain text, Markdown,
  Mermaid, CSV, JSON, "not a fixed/closed list") is too open to justify one parser per format now.
  The three-tier boundary rule degrades gracefully on every format (CSV/JSON fall through to
  sentence-then-hard-cut) without needing per-format code. A future format-aware pass is a
  documented enhancement, not a blocker.
- **Implementation:** a pure function, `chunking.split_into_chunks(text: str, *, size: int = 1000,
  overlap: int = 150) -> list[str]`, in a new `server/falkorchat/chunking.py` — no I/O, trivially
  unit-testable (§5).
- **Reconciling with the dormant schema:** `Chunk` gains `Chunk.seq` (0-based position within its
  document, **no index needed** — read-time ordering only, not a lookup key) and a denormalized
  `Chunk.documentId` (unindexed, mirroring the live `Message.threadId` precedent — DESIGN §5.1's
  "navigation metadata, not an anchor" pattern) so a chunk hit can report which document it came
  from without a traversal.

### 3.3 Extraction (FR-7a) — delegated, recommendation only

Extraction turns chunk text into `Entity` nodes plus `RELATES_TO` fact edges (new relationship
type, §3.4). **Recommendation, pending `data-scientist` review:**

- **Technique:** LLM-based extraction over a structured JSON output schema (`{"entities":
  [{"name","type"}], "relationships": [{"subject","predicate","object"}]}`), one call per chunk.
  An NLP-pipeline (spaCy-style NER) alternative exists but this codebase has no such dependency
  today and the LLM path reuses the already-proven `ModelGateway` seam end to end — data-scientist
  should confirm this is still the right call for accuracy/cost, not just convenience.
- **Parse robustness — a hard-won lesson, reuse it.** K-027 item 1 (`docs/BACKLOG.md`) found that a
  bare `json.loads` on a local model's JSON output breaks on **every** response when the model wraps
  its JSON in a ` ```json ` fence, and fixed it with `llm.extract_own_line_json_object(...,
  require_key=...)` — a parser tolerant of fencing but conservative about ambiguous prose. This
  plan's extraction parser **must** reuse that exact helper (or an equally reviewed sibling), not a
  fresh `json.loads`, on the same reasoning: this codebase runs against small local models, and this
  exact failure mode is proven to recur.
- **Predicate representation:** relationship predicates are **not** their own Cypher relationship
  types. Every extracted fact lands as one relationship type, `RELATES_TO`, carrying the predicate
  as a free-text property (`label`). This avoids an unbounded, LLM-controlled set of graph
  relationship types (a real risk — LLM output is not a closed vocabulary) and follows the existing
  "opaque string, parsed app-side" convention already used for `Step.config`/`TRANSITION.guard`
  (`falkor-chat/AGENTS.md` rule 8).
- **Bounding extraction per chunk (RAM/supernode risk, §6):** cap extracted entities and
  relationships per chunk (e.g. 20 entities / 20 relationships) — mirrors `docs/DESIGN.md` §5.4's
  existing mitigation ("capping entity extraction per message"), now applied to chunks.

### 3.4 Fusion (FR-6, FR-8/FR-9/FR-10) — the mechanism, OQ-1/OQ-2/OQ-3 resolved

**Never physically merge `Entity` nodes.** FalkorDB has no APOC-style node-merge/refactor
procedure, and physically merging (migrating every edge off one node onto another, then deleting
it) is exactly the kind of destructive graph surgery this codebase avoids elsewhere (compare how
`WorkflowDefSnapshot`/`Step` deletion blast radius is carefully reasoned about in `docs/DESIGN.md`
§6.2 rather than casually mutated). Instead, **every fusion decision — auto or suggested — is
recorded as a property-bearing edge**, per `graph-dba`'s finalized schema
(`docs/plans/document-ingestion-graph.md` §1.5):

```
(:Entity)-[:SAME_AS {matchId, status, confidence, technique,
                      createdAt, decidedAt, decidedBy,
                      resuggestCount, lastResuggestedAt}]->(:Entity)
```

**Revision note (post-graph-dba reconciliation).** This plan's first draft recommended a reified
`MatchSuggestion` node (`CANDIDATE_A`/`CANDIDATE_B` edges), explicitly flagging as unverified
whether this build supports indexed relationship properties — the premise a `WorkflowRun.status`-
style node model leaned on. `graph-dba`'s note live-verifies that it **does**
(`Edge By Index Scan`, working `RELATIONSHIP`-scoped `UNIQUE` constraints,
`document-ingestion-graph.md` §1.1) and, with that blocker cleared, recommends the plain
`SAME_AS` edge instead — RAM measured as a wash between the two shapes (~840 B/suggestion edge vs.
~791 B node model, live-measured, not assumed, §1.2), decided instead on hop count for the
eventual "expand to fused siblings" read this schema exists to serve (one hop for the edge model
vs. two through a node intermediary, §1.3) and on write-path fit (the OQ-3 find-or-reopen-or-create
write, §1.6, maps directly onto this codebase's dominant guarded-`CREATE`-inside-`FOREACH(CASE...)`
idiom with no new mechanism). `graph-dba`'s note also documents a live-verified planner trap that
makes the edge shape non-trivial to get right: a bare label on either endpoint of a `SAME_AS`-
anchored query forces a full `Node By Label Scan` even though the relationship-property scan alone
is fully selective (`document-ingestion-graph.md` §1.4) — every `SAME_AS` query must therefore
match its endpoints unlabeled (`(a)-[r:SAME_AS {...}]->(b)`, never `(a:Entity)-[r:SAME_AS
{...}]->(b:Entity)`). `status ∈ {pending, confirmed, rejected}`, exactly as originally designed —
only the physical shape changed, not the tier semantics below.

**The three tiers, mechanically:**

- **FR-8 (auto-merge, "very-high confidence"):** a `SAME_AS` edge is written with
  `status='confirmed'`, `decidedAt`/`decidedBy='system'`, at ingestion time — no human/agent action
  needed. The new entity node still exists (never merged into the matched one); the two are linked
  as "the same" via the confirmed `SAME_AS` edge. **OQ-1 default (flagged to `data-scientist`, §0;
  confirmed with no schema-affecting divergence in `document-ingestion-ml.md`):** "very-high
  confidence" = an exact match on **normalized name (case-folded, whitespace-collapsed) + identical
  `Entity.type`** — a deterministic identity check, not a fuzzy score, chosen because this pipeline
  has **no calibration data** (unlike the guard judge, which K-027 item 3 calibrated against a
  golden set before being trusted with silent behavior) — an unvalidated numeric threshold
  auto-linking two *possibly-different* real entities is a correctness risk this default avoids by
  only auto-linking what is *definitionally* the same string+type. `confidence` is stored as `1.0`
  for this tier (a deterministic identity check, not a score — `document-ingestion-graph.md` §1.5).
  Mechanically, this check is **not** run through the RediSearch fulltext index at all: `graph-dba`
  gives it its own deterministic mechanism, a plain RANGE-indexed `Entity.nameNormalized` property
  compared with `=` (§2.5 below) — decoupled from a search engine's tokenizer/stemmer behavior,
  which the exact tier's "no calibration data" argument above depends on staying deterministic.
  **This tier's candidate check, the new entity's creation, and its auto-link are one atomic
  `GRAPH.QUERY`** — see the concurrency note immediately below the tier list; this is the one
  correction this section makes relative to the reconciled draft.
- **FR-9 (suggested, "plausible but not very-confident"):** a `SAME_AS` edge is written with
  `status='pending'`. Candidate generation: a RediSearch fuzzy full-text query against the
  `Entity.name` full-text index (§2.5) — this is the suggested tier's *own* mechanism, distinct
  from the exact tier's `nameNormalized` check above, not a fallback path off it. `confidence`
  stores the raw RediSearch relevance score for audit/UI purposes but does not gate a second
  numeric threshold in v1 — `document-ingestion-ml.md` explicitly defers embedding-based semantic
  matching (e.g. catching "IBM" vs. "International Business Machines") to a scoped v2, not v1.
  **This tier's lookup stays a separate read followed by `create_or_reopen_match` (unchanged,
  §3.4/§4 Stage 4) — it does not need the exact tier's atomicity fix**, because a missed/duplicated
  suggestion under concurrent timing still lands in the reviewed `pending` queue either way (worst
  case: two near-simultaneous fuzzy suggestions for the same pair, or a suggestion silently missed
  and re-derivable on the next ingestion) — it never silently defeats a zero-review guarantee the
  way the exact tier's race did, since nothing here is ever auto-confirmed without a human/agent
  decision. See the concurrency note below for why the two tiers are treated differently.
- **FR-10 (confirm/reject):** `confirm_match`/`reject_match` (§4.4) flip `status` to `confirmed` /
  `rejected`, stamping `decidedAt`/`decidedBy` (a `User`/`Agent` id — never `'system'` on this
  path, so an audit trail can always tell an automatic decision from a human/agent one). Rejecting
  does **not** delete the `SAME_AS` edge — it stays as a `rejected` record, which is what makes
  OQ-3 answerable without a second mechanism.
- **OQ-2 (where a pending match surfaces) — resolved: a dedicated review surface, not chat.**
  `list_pending_matches` (§4.4), reachable via both MCP and REST. Rejected alternative: posting a
  message into a channel — rejected because a pending match has no natural channel/thread anchor
  (fusion happens per-workspace, not per-conversation) and would conflate knowledge-base curation
  with chat, which FR-14 already keeps as a separate concern. This mirrors how `WorkflowRun`'s own
  "awaiting" state is discovered by a read endpoint (`GET /workflow-runs/{id}/step-runs`), not a
  chat post.
- **OQ-3 (how a rejected match gets re-evaluated) — resolved, two paths, both real:**
  1. **Automatic re-open on corroboration.** If a later ingestion independently re-derives the same
     candidate pair (same two entity ids, in either discovery order — the write's lookup is
     direction-agnostic, `document-ingestion-graph.md` §1.5/§1.6), the existing `SAME_AS` edge is
     reused (found by an undirected `OPTIONAL MATCH` between the two id-anchored entities, never
     re-created) — a `rejected` one flips back to `pending` (never straight to `confirmed`, even on
     an exact-match re-derivation: a human/agent said no once, so one more explicit confirmation is
     required) and `resuggestCount`/`lastResuggestedAt` are bumped for visibility.
  2. **Manual recheck on demand.** A new `recheck_match(match_id)` tool (§4.4) lets a human/agent
     force a `rejected` suggestion back to `pending` without waiting for new content — satisfies
     AC-7's "or a human/agent chooses to."

**Concurrency note (post plan-gate review) — closing the exact-tier check-then-act race.** The
plan-gate review (`docs/reviews/document-ingestion.md`, BLOCKER finding) caught that the original
draft above described the exact tier as three independent round trips — `find_exact_candidate`
(read), `create_entity` (write), `create_or_reopen_match` (write) — with nothing binding them
together. Two entities extracted around the same wall-clock time (two documents in one
`ingest_documents` batch, AC-8's own scenario; or two concurrent MCP `ingest_document` calls, FR-5's
own use case) could each run `find_exact_candidate` before either sibling's create/link commits,
so **neither** sees the other and no `SAME_AS` edge is ever written for a pair that should have
auto-merged — silently defeating FR-8's "no confirmation required" guarantee on the one fusion
action with zero human/agent review (`decidedBy='system'`).

**Decision: close the race at the database layer (the review's option (a)), not with a processing-
order guarantee (option (b)).** Reasons:

- **This codebase's own concurrency model has no serialization point to hang (b) off of.**
  `background.py` schedules per-request/per-call — FastAPI `BackgroundTasks` (concurrent across
  requests) for REST, a bare `threading.Thread` (fire-and-forget, no shared queue) for MCP
  (`background.py:1-11`) — and this plan's own §2.2 explicitly models `IngestionPipeline` on that
  same pattern. Enforcing "one entity at a time, never concurrently" for extraction+fusion alone
  (but not embedding) would mean inventing a new primitive this codebase doesn't have anywhere
  today — a lock or single-worker queue — asymmetric with every other background policy, and a new
  failure mode of its own (a stuck/slow lock holder blocking every other ingestion in the
  workspace). That is real new machinery for a fix that (a) gets for free.
- **(a) is the established discipline, not a special case.** Every other race-prone write in this
  codebase — Thread HEAD/TAIL, member-ensure, model-override CAS (`falkor-chat/AGENTS.md` rule 4)
  — is already a single atomic guarded query, precisely *because* FalkorDB/Redis serializes command
  execution: two concurrent `GRAPH.QUERY` calls against the same graph can never interleave their
  reads and writes, only run strictly one after the other. `graph-dba`'s own `create_or_reopen_match`
  (§1.6 below) already proves the idiom for the write half; folding the read one step earlier is the
  same technique applied one hop sooner, not a new one.
- **The fuzzy/suggested tier does not need the same fix** (see the FR-9 bullet above) — extending
  atomicity to it would be solving a problem that doesn't exist there, since nothing on that path is
  ever silently auto-confirmed.

**What changes mechanically:** the exact tier's candidate lookup, the new entity's creation, and its
auto-link (when a candidate exists) become **one** atomic write — replacing the plan's original
three-round-trip sequence for this tier only. The fuzzy tier's two-step shape (`find_fuzzy_candidates`
read, then `create_or_reopen_match` write) is unchanged. See §4 Stage 4 for the resulting file-list
change and the exact interface `graph-dba` is designing the Cypher for.

### 3.5 MCP/REST write surface (FR-5)

New MCP tools (peers of the existing table in `docs/DESIGN.md` §15.2), mirrored as REST routes
(peers of §14.4) for symmetry with the existing dual-transport convention — REST parity is a
judgment call, not FR-5-mandated (FR-5 only requires the MCP path), and can be trimmed if scope
needs tightening:

| MCP tool | REST | Service method |
|---|---|---|
| `ingest_document(text, title=None, source_format="text", source_label=None)` | `POST /documents` | `ingest_document` |
| `ingest_documents(items: list[dict])` | `POST /documents/batch` | `ingest_documents` |
| `get_document(document_id)` | `GET /documents/{id}` | `get_document` |
| `search_documents(query, limit=20)` | `GET /documents/search?q=` | `search_documents` |
| `list_pending_matches(limit=50)` | `GET /matches/pending` | `list_pending_matches` |
| `list_matches(status=None, limit=50)` | `GET /matches?status=&limit=` | `list_matches` |
| `confirm_match(match_id)` | `POST /matches/{id}/confirm` | `confirm_match` |
| `reject_match(match_id)` | `POST /matches/{id}/reject` | `reject_match` |
| `recheck_match(match_id)` | `POST /matches/{id}/recheck` | `recheck_match` |

- **`list_matches` audit surface (plan-gate review, MAJOR finding).** `list_pending_matches` only
  ever shows `status='pending'` rows; the auto-merged tier (`status='confirmed', decidedBy='system'`)
  had no discovery surface at all — an operator could not enumerate it without a raw Cypher query,
  even though `reject_match`/`recheck_match` are only reachable once a `match_id` is already known.
  `list_matches(status=None, ...)` fills that gap: same shape as `list_pending_matches`, but
  status-filterable (or unfiltered when `status` is omitted) — reuses the `SAME_AS.status` index
  already indexed for the pending-only listing (`document-ingestion-graph.md` §1.5). Cypher
  authored/verified by `graph-dba`, mirroring `list_pending_matches` (§1.7) with the status literal
  parameterized.
- **Actor attribution (FR-4):** `ingest_document`/`ingest_documents` stamp `Document.ingestedBy`
  from `get_context()`'s actor — exactly the existing MCP posture ("MCP ignores any client-supplied
  `frm`; every call is attributed to the `get_context()` actor," `docs/DESIGN.md` §15.2). This is
  what makes agent-authored content "treated the same as a human-supplied document" (FR-4)
  mechanical, not a special case: the actor resolves to a `User` or `Agent` node exactly as message
  authorship already does (`coalesce(userId, agentId)`, `docs/DESIGN.md` §1.2).
- **Bounds (mirroring the existing REST size caps, `docs/DESIGN.md` §14.4):** a suggested
  `MAX_DOCUMENT_CHARS = 500_000` and `MAX_BATCH_SIZE = 20` — implementer-tunable, not load-bearing.
  The two caps compound: at both ceilings, one `ingest_documents` call can queue on the order of
  600 chunks/document (500,000 chars ÷ the §3.2 1,000-char target) × 20 documents ≈ 12,000
  background extraction LLM calls from a single MCP/REST call, with no rate-limiting/backpressure
  mechanism proposed here. Acceptable for v1 (each bound is independently reasonable and neither is
  load-bearing) — revisit if real usage makes the compounded fan-out a practical problem, since
  FR-5 opens this surface to any connected agent, not just a human-paced UI.

### 3.6 Bulk ingestion (FR-11) and retention (FR-12)

- **Bulk (FR-11):** `ingest_documents` loops the single-document path per item, returning one
  receipt per item; each document's background processing (extraction/fusion/embedding) is
  independent, so cross-document fusion (AC-8) happens naturally once each item's background work
  runs — no special "batch-aware" fusion logic is needed, because fusion always checks the graph's
  *current* state (including sibling documents from the same batch, once their entities land),
  never a batch-local view.
- **Retention (FR-12/AC-9):** `Document.text` stores the full original source text verbatim, as a
  flat string property (consistent with the `ctx`/`input`/`output` "flat serialized strings"
  convention, `falkor-chat/AGENTS.md` rule 8 — `Document.text` is never queried *inside*, only read
  whole). `get_document` returns it verbatim. No separate archival store — single-store philosophy
  (`docs/DESIGN.md` philosophy header) applies here exactly as everywhere else in this project.

### 3.7 Retrieval integration (FR-2, FR-3, FR-14)

**FR-3 (standalone KB, independent of chat):** `search_documents` runs a `Chunk`-only ANN query
(`db.idx.vector.queryNodes('Chunk', 'embedding', ...)`, the existing vector index, §2.1) and returns
ranked chunks with their `documentId` (denormalized, §3.2) and, via `(:Chunk)-[:ABOUT]->(:Entity)`,
the entities each chunk mentions — satisfies FR-3 without touching `Message` at all, which is what
keeps this a genuinely separate capability from chat search (FR-14, "not required" to unify).

**FR-2 (chat-grounding integration, mirroring `EMITTED`):** `responder.AgentResponder.maybe_respond`
(`server/falkorchat/responder.py:84-119`) currently does exactly one retrieval call —
`services.hybrid_search` (`Message`-seeded ANN) — then posts the answer with `(msgId, score)`
provenance via `services.post_agent_answer`, written as `(:Message)-[:EMITTED]->(:Message)` inside
the same guarded write (`QUERIES.md` §10.1). This plan extends that flow with a **second**,
app-layer-fanned-out retrieval call against `Chunk`, matching the decision log's own resolution of
the unified-search question ("app-layer fan-out+merge... consistent with the existing hybrid-
retrieval pattern," requirements doc decision log, 2026-08-22): `services.hybrid_search` gains a
sibling (or an internal merge, service-layer's call) that also queries `Chunk.embedding`, merges the
two ranked seed lists (interleave/rank by score), and passes the merged list to
`_build_prompt`/`post_agent_answer`. **The generalization, resolved by `graph-dba`
(`document-ingestion-graph.md` §3.1-§3.3):** today's `EMITTED` write query resolves seed ids
against `Message {msgId}` only (`QUERIES.md` §10.1's `OPTIONAL MATCH (s:Message {msgId: sid})`);
it now resolves against **either** `Message.msgId` **or** `Chunk.chunkId` via a bare-id `coalesce`
— the same two-label `OPTIONAL MATCH` + `coalesce` idiom already used twice in this query for
author and mention resolution (`docs/QUERIES.md` §4/§10.1), not a new mechanism. Both options this
section originally floated (a `kind` field per seed, or a namespaced id scheme like `msg:<uuid>`/
`chunk:<uuid>`) were **rejected** — `document-ingestion-graph.md` §3.1 reasons that `Message.msgId`/
`Chunk.chunkId` are already disjoint server-minted uuid4 spaces (the same "astronomically
negligible" collision posture already accepted for `User.userId`/`Agent.agentId` sharing one
namespace, `QUERIES.md` §2), so a bare `coalesce` needs no new string-manipulation surface and no
parallel-list zipping. The read side gains a matching `seedKind`/`documentId`/`documentTitle`
projection (`document-ingestion-graph.md` §3.3) so a `Chunk` seed's row also carries its source
document without a second query. AC-5 ("that answer's provenance traces back to the source
document") is satisfied once a `Chunk` seed's `EMITTED` edge is traversable back through
`(:Chunk)<-[:HAS_CHUNK]-(:Document)`, folded into the same read (§3.3's `OPTIONAL MATCH
(s)<-[:HAS_CHUNK]-(d:Document)`).

---

## 4. Step-by-step implementation

Staged so the tree stays buildable and each stage is independently testable; stages 1–2 need no
delegated note, stage 3 needs the ML note, stage 4 needs both notes.

### Stage 0 — Design prerequisites — ✅ both notes delivered and reconciled (§0)

- `graph-dba`: `document-ingestion-graph.md` — delivered, reconciled into §3.4/§2.3/§3.7/§4 above.
  Gate: `analyst` plan-gate review of the full three-document design set (this plan + both notes),
  not yet run.
- `data-scientist`: `document-ingestion-ml.md` — delivered, no schema-affecting divergence (§0).
  Same `analyst` gate as above.

### Stage 1 — Chunking + Document/Chunk write path (no LLM)

**Files:**
- `server/falkorchat/chunking.py` — new. `split_into_chunks` (§3.2), pure function.
- `server/falkorchat/repository.py` — new methods: `create_document(ws, *, document_id, title,
  text, source_format, source_kind, ingested_by, created_at, chunks: list[str]) -> dict`
  (writes `Document` + all its `Chunk`s + `HAS_CHUNK` edges, one `GRAPH.QUERY`, plain guarded
  `CREATE` per §2.4's posture — no HEAD/TAIL race to guard against); `get_document(ws,
  document_id) -> dict | None`. Cypher authored/verified by `graph-dba` (stage 0).
- `server/falkorchat/services.py` — `ingest_document(ctx, *, text, title, source_format,
  source_label) -> dict`: validates length (`MAX_DOCUMENT_CHARS`), splits via `chunking`, calls
  `repository.create_document`, returns `{documentId, chunkCount, status: 'processing'}`; sets
  `Document.status='processing'` (flips to `'ready'` at the end of stage 3's background pipeline —
  §3.6). `get_document(ctx, document_id) -> dict | None`.
- `server/falkorchat/mcp.py` / `api.py` / `schemas.py` — `ingest_document`/`get_document` per §3.5.
- `scripts/bootstrap_schema.sh` — no change needed for this stage (`Document`/`Chunk` DDL already
  exists, §2.1); `graph-dba` confirms in the -graph note whether `Chunk.seq` needs any DDL (it
  shouldn't — a plain unindexed property).

**Done:** a document ingested via MCP/REST is split into chunks, both retained verbatim
(`Document.text`) and split (`Chunk.text` + `seq` + `documentId`); `get_document` round-trips the
full text (AC-9 provable in isolation, before any entity work exists).

### Stage 2 — Chunk embeddings + standalone search (FR-3)

**Files:**
- `server/falkorchat/embedding.py` or a small sibling — extend/reuse `EmbeddingWorker` to also
  embed chunks (`set_chunk_embedding`, mirroring `set_embedding` for messages, `QUERIES.md` §6).
- `server/falkorchat/background.py` — `_safe_embed_chunk`, mirroring `_safe_embed`.
- `server/falkorchat/repository.py` — `search_chunks(ws, *, q_vec, k, limit) -> list[dict]`
  (`Chunk`-only ANN, §3.7).
- `services.py`/`mcp.py`/`api.py` — `search_documents` per §3.5.

**Done:** an ingested document's chunks are embedded out-of-band (readable before embedding
lands, same posture as messages); `search_documents` returns ranked chunks. AC-6 (MCP write, then
read by any agent) is provable end-to-end at this stage, without entities yet.

### Stage 3 — Extraction (FR-7a) — needs the ML note

**Files:**
- `server/falkorchat/extraction.py` — new. Prompt template + `extract(chunk_text, llm) ->
  ExtractionResult` (dataclass: `entities: list[{name, type}]`, `relationships:
  list[{subject, predicate, object}]`), parsed via `llm.extract_own_line_json_object` (§3.3).
- `server/falkorchat/repository.py` — `create_entity(ws, *, entity_id, name, name_normalized, type,
  created_at) -> dict` (`name_normalized` = case-fold + whitespace-collapse of `name`, computed
  app-side by the **same** normalization helper `extraction.py`'s subject/object stub-repair uses —
  one shared function, not two independently-written normalizers that can drift, per
  `document-ingestion-graph.md` §2.2/§7 and `document-ingestion-ml.md` §3.2 — written starting this
  stage even though fusion (stage 4) is the first reader, so no backfill migration is needed later);
  `link_chunk_about_entity(ws, *, chunk_id, entity_id)`; `create_entity_relationship(ws, *,
  subject_id, object_id, label, source_chunk_id, source_document_id, created_at)` (the `RELATES_TO`
  fact edge, §3.3 — never deduplicated, per §3.1).
- `server/falkorchat/ingestion.py` — new `IngestionPipeline` (or `IngestionWorker`) component,
  peer of `AgentResponder`/`EmbeddingWorker`: per chunk, calls `extraction.extract`, then (stage 3:
  unconditionally) creates a fresh `Entity` per extracted mention + `ABOUT` edge + `RELATES_TO`
  edges for extracted relationships between entities resolved so far. **Fusion is deliberately not
  wired yet in this stage** — every extraction creates a new entity, so FR-7a/AC-10 is provable in
  isolation before fusion's added complexity (§3.1's two-axes framing makes this split natural: a
  system with fusion permanently at "always create new" is a valid, testable degenerate case).
- `server/falkorchat/background.py` — `_safe_extract`, mirroring `_safe_embed`'s try/except-log-
  never-raise discipline: an extraction failure for one chunk must not corrupt the `Document` or
  block sibling chunks (§5's "Background-job failure isolation" test bullet already expects this;
  this is the file-list entry that was missing for it, per the plan-gate review's MINOR findings).
- `config/models.json` — add the `extraction` kind's default (per graph-dba's stage-0 note, §2.3).
- `scripts/bootstrap_schema.sh` — `Entity.name` full-text index (`CALL
  db.idx.fulltext.createNodeIndex('Entity', 'name')`) and a plain RANGE index on
  `Entity.nameNormalized` (no constraint — distinct real entities can share a normalized name+type
  before fusion runs) — per `document-ingestion-graph.md` §2.3/§5. Bundled here (rather than stage
  4) because `nameNormalized` starts being written this stage; the fulltext index is inert until
  stage 4's fuzzy lookup uses it, same "bootstrapped, not yet populated/queried" posture the
  original `Chunk` scaffolding already demonstrated (§2.1).

**Done:** an ingested document's chunks yield `Entity` nodes and `RELATES_TO` fact edges, each
traceable to its source chunk/document (AC-10) — no fusion behavior yet (every entity is new).

### Checkpoint — extraction-quality qualitative review (advisory, not blocking)

Per `document-ingestion-ml.md` §6's "firm follow-up, not optional": once Stage 3 is live
end-to-end against real ingested content, `data-scientist` reviews a qualitative sample of real
extraction output (~20-30 chunks across a few real documents) for the two named, unmeasured
failure modes — under-extraction (missed real entities) and over-extraction (hallucinated ones) —
**before** Stage 4 (fusion) is trusted with the entity population extraction actually produced.
This is explicitly **advisory, not a hard gate**: Stage 4 implementation can proceed in parallel
(fusion's own correctness doesn't depend on extraction's output quality, only on what it's handed),
matching the ML note's own stance that a pre-launch calibration pass is reasonable to skip for a
net-new capability with no prior production traffic. What changes here is visibility: `teco`'s
coordination log should record this review's completion (and any findings, e.g. a fallback to the
two-stage entities-then-relationships extraction call named in the ML note §6) before fusion is
considered validated against real data, not unit/fixture tests alone — the plan-gate review found
this checkpoint named only in a sibling document, with nothing in this plan's own build sequence
making it visible to whoever is gating the Stage 3→4 boundary.

### Stage 4 — Fusion (FR-6/7/8/9/10, OQ-1/2/3) — needs both notes

**Files:**
- `server/falkorchat/fusion.py` — new. **The exact tier (FR-8) is folded into entity creation
  itself as of the concurrency fix below (§3.4) — it is no longer a separate `find_exact_candidate`
  pre-check called from here.** `fusion.py` now holds: `find_fuzzy_candidates(repo, ws, name, type,
  limit) -> list[MatchCandidate]` (the FR-9 tier, RediSearch fuzzy full-text, §2.5/§3.4, unchanged
  from the original draft) and `classify_fuzzy(fuzzy) -> Literal['suggested', 'none']` (the
  OQ-1 default's fuzzy branch only — the exact/'auto' branch is now decided inside the atomic
  repository call, not in Python, so the three-way `classify(exact, fuzzy)` this section originally
  named collapses to this two-way helper).
- `server/falkorchat/ingestion.py` — **wiring changed by the concurrency fix (§3.4).** Per
  extracted entity mention: call `repository.create_entity_with_auto_match(...)` (below) — this
  single atomic call replaces stage 3's plain `create_entity` at this call site *and* the original
  `find_exact_candidate` pre-check; it always creates the entity, and additionally auto-links it
  (`status='confirmed'`) if an exact candidate existed, in one round trip. If the call reports
  `exactMatched=false`, then (and only then) call `fusion.find_fuzzy_candidates`; if any candidate
  is found, call `repository.create_or_reopen_match` (below) with `status='pending'` for the
  top-ranked one. `create_entity` (stage 3, §2.2 of the graph note) stays in `repository.py` as the
  underlying plain-create primitive but is no longer `IngestionPipeline`'s entity-creation call site
  once this stage lands.
- `server/falkorchat/repository.py`:
  - **New — `create_entity_with_auto_match(ws, *, entity_id, name, name_normalized, type,
    created_at, match_id) -> dict`** (returns `{entityId, exactMatched, candidateEntityId,
    matchId}`). This is the blocker fix: one atomic `GRAPH.QUERY` that (1) looks up the FR-8 exact
    candidate (`Entity.nameNormalized`/`type` `=` lookup, oldest-first tie-break, same semantics as
    the original `find_exact_candidate` read), (2) creates the new `Entity` node, and (3), only if a
    candidate was found, creates a `SAME_AS{status:'confirmed', decidedBy:'system', confidence:1.0,
    technique:'exact_normalized_name_type', ...}` edge from the new entity to the candidate — all
    three inside one round trip, so no other concurrent `GRAPH.QUERY` can observe a state between
    "candidate checked" and "entity created + linked." Closes the plan-gate review's BLOCKER
    finding by construction: FalkorDB/Redis serializes command execution, so two concurrent calls
    against the same `(nameNormalized, type)` can never both miss each other's not-yet-committed
    sibling the way the original three-round-trip sequence could. Exact Cypher, live verification,
    and the precise ordering guarantee (the candidate `MATCH` must bind before the new entity's
    `CREATE`, so the new entity can never appear as its own candidate) are `graph-dba`'s to design —
    see the handoff spec in this plan's decision record (§3.4's concurrency note) and the
    coordinator's dispatch. A brand-new entity cannot already have a `SAME_AS` edge to anything, so
    this write does **not** need `create_or_reopen_match`'s "reopen a rejected edge" branch — a
    deliberate simplification, not an oversight, for `graph-dba` to confirm rather than copy the
    full reopen-capable shape unnecessarily.
  - `create_or_reopen_match(ws, *, new_entity_id, candidate_entity_id, match_id, status, confidence,
    technique, created_at) -> dict` (returns `{created, reopened, matchId, status}` — the guarded
    find-or-reopen-or-create write, `document-ingestion-graph.md` §1.6: an `OPTIONAL MATCH`
    undirected lookup between the two id-anchored entities, then a guarded `FOREACH(CASE...)` that
    either creates a fresh `SAME_AS` edge or, if a `rejected` one already exists for this pair,
    flips it back to `pending` and bumps `resuggestCount`/`lastResuggestedAt` — **not** a bare
    `MERGE`, since `SAME_AS` is semantically symmetric but a fixed-direction `MERGE` would miss an
    edge written in the opposite discovery order, §1.6). **Unchanged by the concurrency fix — still
    called for the fuzzy/suggested tier only** (`status='pending'`); no longer called for the exact
    tier, which now goes through `create_entity_with_auto_match` above.
  - `confirm_match(ws, *, match_id, decided_by, decided_at)`, `reject_match(ws, *, match_id,
    decided_by, decided_at)`, `recheck_match(ws, *, match_id, at)` (each a matchId-anchored `SET`,
    §1.7 — `recheck_match` only transitions a `rejected` edge, a no-op otherwise), `list_pending_matches(ws,
    limit)` (§1.7, directed on the canonical write direction, no endpoint label per §1.4's
    planner-trap note), and **`list_matches(ws, *, status=None, limit)`** (§3.5 — the same shape,
    status-filterable instead of hardcoded to `'pending'`, closing the plan-gate review's
    auto-merge-tier discoverability finding).
- `server/falkorchat/background.py` — `_safe_fuse`, mirroring `_safe_embed`/`_safe_extract`'s
  try/except-log-never-raise discipline, if fusion's candidate lookups/writes are scheduled as a
  step distinct from extraction's own background task (implementer's call on granularity — either
  one combined `_safe_extract` covering extract+fuse per chunk, or two separate wrappers; either
  way, a fusion failure for one entity must not corrupt the `Document` or block sibling entities).
- `services.py`/`mcp.py`/`api.py` — `confirm_match`/`reject_match`/`recheck_match`/
  `list_pending_matches`/`list_matches` per §3.5.
- `scripts/bootstrap_schema.sh` — `SAME_AS` relationship-scoped indexes (`r.matchId`, `r.status`)
  and `UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId` — per `document-ingestion-graph.md` §1.5/
  §5. (The `Entity.name`/`Entity.nameNormalized` DDL already landed in stage 3, above — not
  repeated here.)

**Done:** AC-1..AC-4, AC-7, AC-8 all provable — conflicting facts survive fusion (never merged
edges), auto-merge needs no confirmation, suggested matches are listable and confirm/reject-able,
rejection is reversible both automatically and on demand, and a batch of documents fuses against
each other as well as existing knowledge.

### Stage 5 — Chat-grounding integration (FR-2)

**Files:**
- `server/falkorchat/services.py` — extend/add `hybrid_search` to also query `Chunk` seeds and
  merge (§3.7), per graph-dba's finalized generalized `EMITTED` write.
- `server/falkorchat/responder.py` — `maybe_respond` consumes the merged seed list; `_build_prompt`
  needs no change (already generic over `seeds: list[dict]` with a `text` field).
- `server/falkorchat/repository.py` — the generalized `EMITTED`-write/read (per graph-dba).
- `server/tests/test_provenance.py` — **must be updated, not just extended.** The generalized read
  (`document-ingestion-graph.md` §3.3) changes `read_provenance`/`read_citing_answers`'s asserted
  response shape from `{seedMsgId, text, role, score, rank}` to one that adds `seedKind`/
  `documentId`/`documentTitle` (verified against the current shape at
  `server/falkorchat/repository.py:581-620` and `server/tests/test_provenance.py:56-60`). The
  existing suite fails loudly if this isn't updated, but the plan-gate review flagged that this
  stage's file list didn't name the file explicitly, unlike every other stage's exhaustive list.

**Done:** AC-5 — an agent's chat answer grounded in ingested content carries provenance back to the
source chunk/document, traversable exactly like today's `Message`-seeded `EMITTED` edges.

### Stage 6 — Batch hardening + QA acceptance

- Confirm `ingest_documents` batch semantics (§3.6) under a real multi-document fixture with
  cross-document entity overlap (the concrete AC-8 scenario).
- `qa-engineer` acceptance pass, mirroring the K-015/K-025/K-036 pattern — versioned test plan +
  report (`docs/test-plans/document-ingestion.md`, `docs/test-reports/document-ingestion-report.md`).

---

## 5. Test strategy

| AC | What proves it | Altitude |
|---|---|---|
| AC-1 (conflicting facts kept) | Ingest two chunks with differing facts about the same entity name; assert both `RELATES_TO` edges exist, each with distinct `sourceChunkId`/timestamp, neither deleted/overwritten | repository/service integration |
| AC-2 (auto-merge, no confirmation) | Ingest an entity that exact-matches (normalized name + type) an existing one; assert a `SAME_AS{status:'confirmed', decidedBy:'system'}` edge exists immediately, no pending step | repository/service integration |
| AC-3 (suggested, unlinked until confirmed) | Ingest a fuzzy-but-not-exact match; assert a `SAME_AS{status:'pending'}` edge; assert the two entities are NOT resolved as "the same" by any confirmed-only read | repository/service integration |
| AC-4 (confirm/reject) | `confirm_match`/`reject_match` on a pending suggestion; assert status transitions and audit fields (`decidedBy`, `decidedAt`) | service/API contract |
| AC-5 (chat-grounding provenance) | Ingest a document, `@mention` the agent with a question the ingested content answers; assert the answer message's `EMITTED`-generalized edge resolves back to the source chunk/document | responder integration (mocked LLM) + a live-marked e2e mirroring `test_workflow_live.py`'s pattern |
| AC-6 (MCP write → any agent read) | One MCP client calls `ingest_document`; a second calls `search_documents`/`get_document` and finds it — mirrors the existing K-041 cross-transport send/read pattern | MCP integration |
| AC-7 (rejected, not permanent) | Reject a match; re-derive the same pair via a second ingestion (assert auto-reopen to `pending`, never straight to `confirmed`); separately, call `recheck_match` directly on a rejected pair | repository/service integration |
| AC-8 (bulk, cross-document fusion) | `ingest_documents([doc_a, doc_b])` where both mention the same entity; assert both processed and a `SAME_AS` edge links their extracted entities, not just doc_a's | service integration, background-completion-aware (poll `Document.status` or run the pipeline synchronously in the test) |
| AC-9 (full document retained) | `get_document` returns `text` byte-identical to the ingested input, including for a large multi-chunk document | repository/service unit |
| AC-10 (entities/relationships traceable) | Ingest a document mentioning two entities and a relationship; assert both `Entity` nodes and the `RELATES_TO` edge exist and traverse back to their source chunk/document | repository/service integration |

**Additional, non-AC-mapped test coverage:**
- `chunking.split_into_chunks` — pure unit tests: short text (one chunk), exact-boundary text, a
  long single paragraph (sentence-split path), a long single "sentence" (hard-cut path),
  empty/whitespace-only input (rejected upstream at the service boundary, not the chunker's job).
- `extraction.extract` — parser robustness tests mirroring K-027 item 1's fixture shape: bare JSON,
  fenced JSON, and a deliberately ambiguous/prose-wrapped reply that must resolve to "no result,"
  not a silently wrong one (reuse `test_llm.py`'s existing `extract_own_line_json_object` fixtures
  as the template).
- **Background-job failure isolation** — an extraction or embedding failure for one chunk must not
  corrupt the `Document` or block other chunks; `Document.status` flips to a `failed`/`partial`
  state rather than leaving a silently-stuck `'processing'` document, mirroring `_safe_embed`'s
  try/except-log-never-raise discipline (`background.py`).
- **Exact-tier auto-merge race (plan-gate review BLOCKER, §3.4 concurrency note)** — a concurrency
  test proving the fix actually closes the race: two entities with the same `(nameNormalized,
  type)` created via two concurrent `create_entity_with_auto_match` calls (real threads/tasks, not
  sequential calls dressed up as concurrent) must produce exactly one `SAME_AS{status:'confirmed'}`
  edge between them, never zero. This is the regression test for AC-2/AC-8's concurrent variant —
  the sequential case AC-2's row above already exercises does not, by itself, prove the race is
  closed.
- **`test_queries.sh` additions** — graph-dba's gate raises the enumerated baseline for every new
  Cypher shape introduced (Document/Chunk/Entity/`SAME_AS` writes, the exact + fuzzy candidate
  lookups, the generalized `EMITTED` write/read, the `Chunk` ANN query) — the plan does not
  enumerate an exact new count here since the exact queries are graph-dba's to author (already
  live-verified in `document-ingestion-graph.md`, including the §1.4 no-endpoint-label discipline
  every `SAME_AS` query must follow).
- **A default (offline) `pytest` run stays network-free** — extraction/matching are exercised with
  injected fakes (mirroring `llm=`/`embedder=` injection sites throughout the existing suite,
  `docs/DESIGN.md` §14.8); a live-marked test (`pytest -m live`) proves the real LLM extraction
  prompt/schema end-to-end, following the `test_workflow_live.py`/`test_services_live.py` pattern.

---

## 6. RAM/scale considerations (rule 6)

- **`Chunk.embedding` vector index** — the dominant new RAM line, same empirical shape as
  `Message.embedding` (`docs/DESIGN.md` §11: ~12.4 KB/vector at 1024 dims, ~85% of which is the
  vector+HNSW overhead, not the node itself). Chunk count scales with corpus size independent of
  message volume — a single large document at the recommended 1,000-char chunk size (§3.2)
  produces roughly `len(text) / 1000` chunks (e.g. a 250,000-character document → ~250 chunks ≈
  3 MB at the 1024-dim line). **No new DDL is needed** (the index already exists, §2.1), but this
  is a genuinely new, ingestion-driven growth axis the existing per-workspace RAM budget
  (`docs/DESIGN.md` §11.2) did not account for — worth re-measuring once real ingestion volume
  exists, same posture as the existing "re-measure with the real embedding model" action item.
- **No new vector index for entity matching** (§3.4/§2.5) — the shipped default (confirmed by both
  delegated notes: `document-ingestion-graph.md`'s `Entity.nameNormalized`/fulltext mechanism for
  §2.3, `document-ingestion-ml.md`'s explicit v1/v2 split) reuses RediSearch full-text on
  `Entity.name` plus a plain RANGE-indexed `Entity.nameNormalized` rather than adding
  `Entity.embedding`, which would have doubled the vector-RAM growth axis. Embedding-based semantic
  matching is deferred to a scoped v2 in `document-ingestion-ml.md` — when/if that lands, the RAM
  trade-off this bullet describes should be made visible again, not silently reopened.
- **`Document.text` duplicates chunk text** — the full source is stored once on `Document.text` and
  again, split, across its `Chunk.text` properties (~2× the raw text size, but text is cheap
  relative to vectors — negligible against the `Chunk.embedding` line above). Called out per rule 6
  rather than left implicit.
- **`Entity`/`SAME_AS` growth is extraction-volume-bound**, capped by the per-chunk extraction
  limits recommended in §3.3 (20 entities/relationships per chunk) — without a cap, extraction
  volume is LLM-output-bound and not something the app otherwise controls. `graph-dba` live-measured
  the `SAME_AS` edge shape at ~840 bytes/suggestion (`document-ingestion-graph.md` §1.2/§6) — even
  100k suggestions is ≈84 MB, a rounding error against the `Chunk.embedding` line above; `Entity`
  node cost itself is unmeasured here but structurally similar to any other id-anchored entity node
  already in this schema (`nameNormalized` adds one short string + one RANGE index entry, also
  negligible per `document-ingestion-graph.md` §6).
- **Supernode watch (`docs/DESIGN.md` §5.4 already flags `Entity` fan-out as a risk):** a
  frequently-mentioned entity (e.g. a company mentioned across hundreds of ingested documents)
  accumulates `ABOUT`, `RELATES_TO`, **and now `SAME_AS`** edges without bound
  (`document-ingestion-graph.md` §6 makes the same extension). This plan does not introduce a new
  mitigation beyond the existing DESIGN §5.4 note (re-evaluate with `GRAPH.PROFILE` once real data
  lands) — flagged here so it isn't silently forgotten now that a real write path exists to trigger
  it.

---

## 7. Risks & open questions

- **Non-idempotent document creation on retry (§2.4/§3.5).** A retried `ingest_document` call (e.g.
  after a client-side timeout) mints a second `Document` with duplicate content. Accepted, mirroring
  the existing channel/thread creation precedent — not a blocker, but worth a coordinator sign-off
  if the stakeholder later wants exactly-once ingestion semantics (a content-hash dedup key would be
  the natural fix, not designed here since it wasn't asked for).
- **Resolved: `SAME_AS` edge vs. `MatchSuggestion` node (§3.4).** This plan's first draft flagged the
  schema as a recommendation pending `graph-dba`, and explicitly left "does this build support
  indexed relationship properties" unverified. It does — live-verified
  (`document-ingestion-graph.md` §1.1) — and with that blocker cleared, `graph-dba` recommends and
  this plan now adopts the plain `SAME_AS` edge (§3.4), decided on hop count for the future
  "expand to fused siblings" read and write-path fit, not RAM (measured a wash, §1.2). No longer an
  open risk; kept here only as a decision-log pointer for anyone who read the earlier draft.
- **Resolved: exact-tier auto-merge check-then-act race (plan-gate review BLOCKER, §3.4 concurrency
  note).** This plan's original draft described the FR-8 exact tier as three independent round
  trips (candidate read, entity create, link write) with nothing binding them together — two
  entities extracted around the same time could each miss the other's not-yet-committed sibling and
  end up with no `SAME_AS` edge at all, silently defeating FR-8's zero-review guarantee. Closed by
  folding the candidate check, entity creation, and auto-link into one atomic `GRAPH.QUERY`
  (`create_entity_with_auto_match`, §4 Stage 4) rather than adding a new one-at-a-time processing
  guarantee this codebase's background-scheduling model has no primitive for. `graph-dba` is
  designing/verifying the exact Cypher next (dispatched by the coordinator, conditioned on this
  decision). No longer an open risk on the design-decision axis; the remaining work is
  implementation-grade verification, tracked in Stage 4.
- **Extraction/matching validity is unvalidated in v1** — there is no golden set for either axis
  (unlike the guard judge, which K-027 calibrated before being trusted). This is a stated,
  deliberate v1 posture (ship a defensible deterministic default, iterate with real data), not an
  oversight — confirmed acceptable by `document-ingestion-ml.md` (embedding-based semantic matching
  explicitly scoped to a v2, not a v1 precondition).
- **Background-pipeline completion visibility** — `Document.status` is the only signal a caller has
  that ingestion (extraction/fusion/embedding) has finished; no push notification exists. A caller
  that needs "wait until fully searchable" must poll `get_document`. Acceptable for v1 (mirrors the
  existing "message is readable before its embedding lands" eventual-consistency posture already
  used for chat), not redesigned here.
- **A fifth `ModelGateway` kind (`extraction`, §2.3)** reopens K-042's "four closed kinds" framing.
  This is additive (a new kind, not a change to the existing four) and mirrors how K-042 Landing 2
  itself was additive-only, so it is not expected to be controversial — `graph-dba` has already
  confirmed it explicitly (`document-ingestion-graph.md` §4: "zero graph cost," a single nullable
  `WorkspaceConfig` scalar, `extractionModelOverride`, no new DDL) and named the exact
  `modelconfig.py` crosswalk entry to add. No longer an open risk on the graph side.
- **No web UI work is planned** — FR-1..FR-14/AC-1..AC-10 are all reachable via MCP/REST; a `web/`
  surface for ingestion/fusion review is a natural follow-up (mirroring how M3 shipped before M3.5's
  web coverage pass) but is not proposed as part of M5's done-condition here. Flag to the
  coordinator if the stakeholder expects UI in this milestone.
