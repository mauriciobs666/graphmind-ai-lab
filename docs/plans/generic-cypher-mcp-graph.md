# Generic Cypher MCP — graph data model for `graph-dba`'s kaizen working memory

> **Status:** archived · **Owner:** `graph-dba` · **Tracks:** — (M5 proposed)

Design for the data-shape slice of
[`../requirements/generic-cypher-mcp.md`](../requirements/generic-cypher-mcp.md) (FR-2, FR-3,
FR-4, FR-7, FR-8, FR-9, FR-10; AC-1, AC-2, AC-4, AC-5, AC-6), per
[`generic-cypher-mcp-coordination.md`](./generic-cypher-mcp-coordination.md) unit U1. This
document owns **the entry schema, the attribution data shape, curator-clear semantics, the
one-time import, and indexing/footprint** — it does not decide the MCP tool's mechanism, FR-8's
enforcement logic, or FR-4's frozen-marker signal on `inbox.md`; those are `architect`'s U2, which
cites this document by path. No files are edited or graphs mutated by this note — design only.

---

## 0. Graph key

**Recommendation: `kaizen_graph_dba`** — one dedicated graph, following this repo's existing
`cpg_<component>` underscore convention for component-scoped graphs (not `falkor-chat`'s `ws:{id}`
colon convention, which names per-*tenant* workspaces sharing one schema — this is a different
shape: one graph per *agent*, name baked in now so a future non-`graph-dba` pilot doesn't collide
or force a rename). This is a naming call, not a hard requirement of the schema below — `architect`
can override it if the generic MCP tool's graph-discovery UX wants something else, but every
example in this document assumes it.

---

## 1. Entry schema

One node per working-memory entry, carrying exactly the fields today's markdown entry does
(FR-7), plus two identity fields the markdown format doesn't need but a graph does.

**Label:** `:KaizenEntry` (PascalCase, matching this repo's established label convention —
`falkor-chat/docs/DESIGN.md` §3.1).

| Property | Type | Source | Notes |
|---|---|---|---|
| `entryId` | string (UUID v4) | generated at write time | primary key — see §1.1 |
| `date` | string, `YYYY-MM-DD` | markdown `## ` heading date | the fact's discovery date, lexically sortable |
| `fact` | string | markdown `## ` heading text (after the date) | one line |
| `evidence` | string | `- **Evidence:**` body | verbatim, no truncation |
| `context` | string | `- **Context:**` body | verbatim |
| `suggestedHome` | string | `- **Suggested home:**` body | free text, but the inbox template constrains it to `prompt` \| `knowledge base` \| `project docs` \| `unsure` — every current entry conforms (verified, §4) |
| `author` | string | the writing agent's slug | `"graph-dba"` for every entry in this pilot — see §2 |
| `createdAt` | string, ISO-8601 timestamp | write time | when the *node* was created (import time for migrated entries, real write time for new ones) — distinct from `date`, see §4 |

Properties are `camelCase`, matching the repo's established convention (`falkor-chat/docs/DESIGN.md`
§3.1: labels `PascalCase`, relationship types `UPPER_SNAKE`, properties `camelCase`) — **not** the
CPG schema's `UPPER_CASE` convention (`cpg-model.md`), which is a Joern-export artifact this pilot
has no reason to inherit.

### 1.1 Why `entryId` exists (integrity, not just convenience)

Nothing in the markdown format needs a stable id — a human reads the file top to bottom. A graph
that a **curator agent points a targeted clear at** (FR-9, AC-5) does: `cobb` needs to say "delete
*this* entry" unambiguously, and a `MERGE` (or a retried `CREATE`) with no uniqueness anchor is
exactly the "duplicate-node bug waiting for concurrency" this agent warns about generally. `entryId`
is that anchor — see §5 for the backing constraint.

### 1.2 Traversal, and the stretch goal

AC-1's "via graph traversal" is satisfied by a plain `MATCH` — this schema doesn't need multi-hop
to be traversal-queryable, and forcing one in for its own sake would be over-modeling a six-entry
pilot:

```cypher
MATCH (e:KaizenEntry)
RETURN e.date, e.fact, e.evidence, e.context, e.suggestedHome, e.author
ORDER BY e.date
```

**Stretch goal (FR-7, explicitly optional, not required for this delivery).** Genuine
semantic/similarity search would add a `vecf32` property and a vector index over the `fact`/
`evidence` text:

```cypher
CREATE VECTOR INDEX FOR (n:KaizenEntry) ON (n.embedding)
OPTIONS {dimension: N, similarityFunction: 'cosine'}
```

`N` and the embedding model are a `data-scientist` call (my boundary: I own the in-graph vector
mechanics, not which model or how to chunk `fact`+`evidence` into it). Flagging the shape only —
building it is out of scope here.

**A further, later evolution (not needed now, noted so nobody "discovers" it as a limitation):**
`context` is a free-text property today. If the deferred falkor-chat integration (Out of scope,
requirements doc) ever lands, `context` is the natural field to promote into a real edge — e.g.
`(:KaizenEntry)-[:SURFACED_IN]->(:Task)` — at which point traversal actually earns its keep. Left
as a property now; no reason to pre-build a node type with nothing to point at yet.

---

## 2. Attribution data shape (FR-8's foundation)

**Recommendation: a plain property, `author: "graph-dba"` — not a relationship to an `:Agent`
node.**

Considered the relationship shape (`(:KaizenEntry)-[:AUTHORED_BY]->(:Agent {name, kind})`) and
rejected it for *this* delivery: it's the more graph-native choice in the abstract (author is a
shared join target across many entries — my own modeling principle says that's a node-shaped
fact), but nothing in this pilot's actual query surface uses it that way yet. This pilot has
exactly one author. A property:

- Satisfies FR-8's foundation identically — whatever tool-level check `architect` designs (FR-8's
  enforcement, not mine) compares the caller's claimed identity against `e.author`, a plain
  equality predicate, no traversal or extra lookup needed.
- Leaves clean room for FR-10 (a future human-authored entry) with **zero redesign**: a human
  entry is just `author: "<some human identifier>"`. If the human/agent *distinction itself* ever
  needs to be queried (not just displayed), add `authorKind: "agent" | "human"` then — an additive
  property change, not a schema migration.
- Costs one string per node instead of one string per node **plus** a shared node **plus** a
  relationship — smaller footprint (§6), and there's no dense-fan-in risk to design around at
  N authors = 1.

**Revisit trigger:** if this pattern ever extends past `graph-dba` to genuinely multiple
authoring agents (the broader, explicitly-deferred vision) and a consumer wants "give me every
entry from agent X across N entries" as a first-class, frequently-run query, or wants to attach
metadata to the author itself (e.g., which agents are currently active), that's the point where an
`:Agent`/`:Actor` identity node earns its complexity. Not before.

---

## 3. Curator-clear semantics (FR-9)

**Recommendation: hard delete (`DETACH DELETE`), not a `promoted: true` retention flag** — and a
**non-negotiable ordering**: `cobb` appends to `history.md` and confirms the write succeeded
*before* deleting the graph node, never the reverse.

**Why hard delete.** A retained `promoted: true` node is, functionally, a second permanent copy of
something `history.md` already durably holds — exactly the "graph mirrors/indexes a still-
authoritative markdown file" model the requirements doc's decision log explicitly superseded
(round 1 → round 2: "the graph is now the write target for new raw entries... while `history.md`
stays markdown, exactly as today"). Letting cleared entries linger recreates that rejected shape
in miniature and reopens the stakeholder's memory-footprint worry for no benefit — nothing in FR-9
or AC-5 asks for a post-promotion audit trail *in the graph*; `history.md`'s promotion entry is
already that record. The graph's job under this design is to hold **only what hasn't been promoted
yet** — a hard delete makes that literally true rather than true-by-convention (a filtered
`WHERE promoted = false` on every read, which every future consumer would have to remember to
add).

```cypher
MATCH (e:KaizenEntry {entryId: $entryId})
DETACH DELETE e
```

(`DETACH DELETE` rather than bare `DELETE` as a matter of habit — this node has no relationships
under §2's chosen shape, so it's a no-op safety margin, not a requirement.)

**Why the ordering is load-bearing, not a style preference.** The two writes (`history.md` append,
graph delete) are not one transaction — they're two separate operations, quite possibly two
separate tool calls. If the delete ever ran *before* the `history.md` append was confirmed durable
and something failed in between, the fact would be gone from **both** places — permanent data
loss, the one failure mode this whole design cannot tolerate. Append-then-delete fails safe
instead: a crash between the two steps leaves the entry duplicated (still in the graph *and* now
in `history.md`), which is merely a no-op on `cobb`'s next pass, never a loss. This ordering
constraint belongs in whatever `architect`/`cobb` designs as the actual distillation-workflow tool
sequence (`agent-maintenance` skill §5, step 4) — flagging it here because the data model is what
makes the failure mode possible, even though enforcing the order is implementation, not schema.

---

## 4. One-time import (FR-3 / AC-2)

Read `claude/graph-dba/kaizen/inbox.md`, parse its six current entries (verified by re-reading the
file for this design: 2026-08-16 ×4, 2026-08-17 ×2 — every one already in the exact
`## date — fact` / `- **Evidence:**` / `- **Context:**` / `- **Suggested home:**` shape the schema
above expects), and write one `:KaizenEntry` node per entry. **All six map cleanly onto §1's
schema with no field loss** — nothing in the current inbox uses a shape the template doesn't
define (no entry is missing a section, none has extra sections), so there's no lossy edge case to
flag here.

Migration shape (`UNWIND` over a parameter list — the implementer's actual script, not written
here, builds `$entries` by parsing the markdown and generating `entryId`/`createdAt` client-side;
see the note below on why client-side, not in-Cypher):

```cypher
UNWIND $entries AS e
CREATE (k:KaizenEntry {
  entryId:        e.entryId,
  date:           e.date,
  fact:           e.fact,
  evidence:       e.evidence,
  context:        e.context,
  suggestedHome:  e.suggestedHome,
  author:         'graph-dba',
  createdAt:      e.createdAt
})
```

Where each `$entries[i]` is a map like:

```json
{
  "entryId": "‹uuid4, generated by the migration script›",
  "date": "2026-08-16",
  "fact": "`META_DATA` (and `FILE`/`TYPE`/`NAMESPACE`) are absent from both live pysrc2cpg-built graphs...",
  "evidence": "‹verbatim Evidence body›",
  "context": "‹verbatim Context body›",
  "suggestedHome": "knowledge base (`skills/joern-cpg/references/cpg-model.md`) — ...",
  "createdAt": "‹import-run ISO-8601 timestamp›"
}
```

**`entryId` is generated in the migration script (Python `uuid.uuid4()`), not in Cypher.** I'm not
asserting a `randomUUID()`-shaped Cypher function exists on this build — I haven't verified one,
and this document doesn't present unverified functions as fact (per this agent's own standing
rule). Client-side generation is also simply simpler: the migration script already has to parse
markdown in Python, so generating the id there costs nothing extra. This mirrors the precedent in
`docs/plans/cpg-agent-adoption-graph.md` §1.2 (the `CpgBuildInfo` freshness marker), which
likewise generates its timestamp client-side (`date -u ...`) rather than assuming an in-Cypher
equivalent.

**All six imported entries get the same (or near-identical) `createdAt`** — the import run's
timestamp, not each fact's original discovery time (which the markdown only ever recorded to
day-granularity, in `date`). This is a deliberate, useful side effect, not a data-quality gap:
every entry sharing one `createdAt` batch is a visible, queryable signature that these six are the
backfill, distinguishing them for free from any organically-written entry that comes after (whose
`createdAt` will be its own real write time). No extra "imported" flag needed to get that signal.

The migration script itself (parse `inbox.md`, build `$entries`, run the `UNWIND`, confirm six
nodes exist) is implementation — an `architect`-sequenced unit, not written here.

---

## 5. Indexes

**One index, worth doing regardless of scale: an exact-match index + uniqueness constraint on
`entryId`.** Not for query performance (six rows don't need an index to scan fast) — for
integrity, per this agent's own standing principle ("index the anchor, constrain for integrity").
`entryId` is the curator's clear-by-id anchor (§1.1, §3); a constraint makes "two entries can't
silently share an id" an engine-enforced fact instead of a hoped-for one. Recall the
index-before-constraint ordering this build requires (`falkordb-quirks.md`):

```cypher
CREATE INDEX FOR (e:KaizenEntry) ON (e.entryId)
```

```cypher
GRAPH.CONSTRAINT CREATE kaizen_graph_dba UNIQUE LABEL KaizenEntry PROPERTIES 1 entryId
```

**Everything else — deferred, explicitly.** No index on `author`, `date`, or `suggestedHome` at
this entry count. A full label scan over single-digit-to-low-dozens of `KaizenEntry` nodes is
sub-millisecond regardless; adding indexes here would be tuning a query pattern that doesn't exist
yet (nobody has asked "give me all `graph-dba` entries" as a *filtered* query — today, with one
author, `MATCH (e:KaizenEntry)` already *is* that query). Revisit only if/when this pattern
extends past one agent and `author`-filtered reads become a real, frequent access pattern — the
same trigger as §2's revisit note.

---

## 6. Memory-footprint note

Order-of-magnitude only — I don't have a verified per-node/per-index-entry byte figure for this
build to cite as fact, so this is a property-size estimate, not a measured one.

- **Per-entry raw property size:** `evidence` and `context` are the largest fields (roughly
  200–1,500 characters each across the six current entries; `fact` ~100–200 chars; the rest are
  short fixed-shape strings — `entryId` 36 chars, `date` 10, `author`/`suggestedHome`/`createdAt`
  well under 50). Call it **~1–2 KB of string data per entry**, generously.
- **Six entries today, bounded going forward by design — not by an entry-count cap.** This is the
  point that actually answers the stakeholder's stated worry: this graph doesn't accumulate. Every
  entry that lands in it either gets promoted-and-hard-deleted (§3) or stays as a small,
  currently-un-distilled backlog. Even a generous working-set assumption — say, a few dozen
  un-promoted entries at once, well past this agent's actual "moderate churn" history — is on the
  order of **tens of KB total**, plus whatever fixed per-node/per-index-entry engine overhead
  FalkorDB's sparse-matrix representation adds (not zero, but not the dominant term here either).
- **Compare to the instance's real memory consumers:** the same instance holds
  `cpg_falkorchat` at ~167K nodes / ~1.1M edges (this agent's own kaizen inbox, 2026-08-17 entry)
  and `falkor-chat`'s live workspace graphs with vector indexes. This pilot's footprint is several
  orders of magnitude below either — noise, not a sizing concern.
- **The one thing that would change this materially:** the stretch-goal vector index (§1.2). A
  384–768-dim `vecf32` embedding is ~1.5–3 KB per entry **on its own** — bigger than all the text
  properties combined — plus HNSW index overhead. Still trivial at six-to-dozens of entries, but
  worth flagging now so nobody is surprised later that the optional stretch goal, not the base
  schema, is what would actually move this number.

---

## 7. Summary for the implementer (`architect`, U2)

- Graph key: `kaizen_graph_dba` (§0, overridable).
- One label, `:KaizenEntry`, eight properties (§1) — the five markdown-sourced fields
  (`date`/`fact`/`evidence`/`context`/`suggestedHome`), plus `entryId` (generated),
  `author` (assigned attribution, §2), and `createdAt` (generated).
- Attribution is the `author` string property (§2), not a relationship — FR-8's enforcement check
  is a plain equality predicate against it.
- Curator-clear is `DETACH DELETE` by `entryId`, run only **after** the `history.md` append is
  confirmed (§3) — this ordering constraint must survive into whatever tool-call sequence U2
  designs for `cobb`'s distillation workflow.
- Migration is a six-row `UNWIND … CREATE` (§4); all six current `inbox.md` entries map cleanly,
  no field loss.
- One index + one uniqueness constraint, both on `entryId` (§5); everything else deferred.
- Footprint is negligible at this pilot's scale and bounded by the clear-on-promote design itself,
  not by a cap (§6).
