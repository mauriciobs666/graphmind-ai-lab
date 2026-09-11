# Agent knowledge-base strategy — implementation plan

> **Status:** active — substrate stage (everything past Stage 0) blocked pending `falkor-chat/docs/requirements/document-ingestion2.md` · **Owner:** `architect` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

**Block, 2026-09-10 (relayed by `teco`, requirements doc `e6b4a3e`):** the stakeholder reviewed
§1's Option A recommendation and declined it — not on the evidence, which stands, but on
principle: falkor-chat is *meant* to be the one substrate for agent-and-human knowledge
interaction generally, not one graph among several with overlapping GraphRAG capability, so the
update/delete gap §1 found is worth closing in falkor-chat itself rather than routed around with a
parallel system on `kaizen_team`. **Only Stage 0 (§3) is authorized to proceed right now.**
Everything from §2 onward is the correct design *if* Option A is eventually chosen again, kept
intact for exactly that reason, but **none of it is authorized to build** until a new, separate
requirements doc, `falkor-chat/docs/requirements/document-ingestion2.md` (adding real update/delete
to falkor-chat's document ingestion), is specified and built, at which point this document's
Option A/Option B choice is revisited on the merits — not assumed to flip to Option B just because
the capability then exists. See §8 for the full note.

Implements `claude/docs/requirements/agent-knowledge-base-strategy.md` (Ready for design,
`3a094fd`; substrate stage blocked as of `e6b4a3e` — see above) FR-1 through FR-7. Covers: the
graph schema/content model for distilled agent
knowledge, the write-authorization shape it needs, how (and whether) falkor-chat's GraphRAG
machinery gets reused, sequencing against K-030's interim relief, and both open questions the
requirements doc left for design — including a **substrate fork the stakeholder reopened after
handoff** (Open question #1), which this plan resolves with a recommendation, not a coin flip.

**CPG:** used `cpg_falkorchat` — confirmed via a direct query (`MATCH (m:METHOD) WHERE
toLower(m.NAME) CONTAINS 'document' ...`) that no `update_document`/`delete_document`/
`list_documents` method exists anywhere in `falkorchat/{services,repository,mcp,api}.py`, and that
`test_create_document_is_non_idempotent_on_retry` pins document creation as non-idempotent by
design — the evidentiary basis for §1's substrate recommendation. `cypher-mcp` (also touched by
this plan) has no loaded CPG (`cpg_cypher-mcp` absent from `GRAPH.LIST`; only `cpg_falkorchat` and
`cpg_deprecated_salesperson` are loaded on this instance) — considered, not relevant there; its
`server.py` extension in §4 relies on direct source reading instead.

**ML method note:** `data-scientist` returned a method note,
`claude/docs/plans/agent-knowledge-base-strategy-ml.md` (delegated per this plan's own
guardrails — embedding/chunking/retrieval-parameter/evaluation-design calls are its lane, not
architect's to guess), covering the four questions §2/§4.3/§6/§7 below depend on. Its bottom line,
quoted once here and cited (not re-argued) at each section it decides: **keep Qwen3-Embedding-0.6B,
but use its documented asymmetric query-instruction prefix on the query side only; chunk at one
distinct claim per node, not one Markdown heading per node — `review-techniques.md` already
violates the heading-per-technique assumption this plan originally carried; retrieve with top-K=5
and a cosine-distance floor calibrated from a golden-set run, no conversation-style traversal,
only a `familyId` sibling link where a heading splits into several claim-nodes; and gate AC-2 on a
~40-pair, independently-authored golden set scored as recall@5 with a Wilson-interval CI.**

---

## 1. The substrate fork — resolved, with rationale

The requirements doc's decision log (2026-09-10, after handoff) reopened what was originally a
settled call: is the graph-backed store `kaizen_team` (raw kaizen capture, reached via
`mcp__cypher__query`/`cypher-mcp`), or is it falkor-chat's own graph, reached by *ingesting* the
flat KB Markdown files as a document corpus through falkor-chat's own MCP server (Streamable-HTTP)
and its existing `ingest_document`/`ingest_documents`/`search_documents` tools? The stakeholder
was explicitly undecided and asked both to be weighed. I investigated both against this specific
corpus (distilled agent-team technique entries — not chat) and recommend **`kaizen_team`
(Option A)**. This is the single highest-leverage design decision in this plan; the rest of the
plan is written against it.

### Option B (falkor-chat ingestion) — what it buys, and the defect that rules it out

The appeal is real: `falkorchat.services.ingest_document`/`ingest_documents`/`search_documents`
are already-shipped, already-tested MCP tools (`falkor-chat/server/falkorchat/{services,mcp}.py`)
that do exactly "text in → chunked, embedded, vector-searchable corpus" with **zero new code** —
the most literal reading of the stakeholder's "reuse the actual machinery" preference. `search_documents`
already does the text→embed→`db.idx.vector.queryNodes`→ranked-chunks round trip
(`services.py:1229-1240`, confirmed via the CPG method list above), which is precisely the
capability this whole project needs and that `cypher-mcp` alone cannot supply (see §4.3).

But three concrete properties of that pipeline fight this corpus, not just inconvenience it:

1. **No update/delete/list capability — verified via the CPG, not inferred.** `MATCH (m:METHOD)
   WHERE toLower(m.NAME) CONTAINS 'document'` returns 140 rows spanning every method and test in
   the codebase that touches `Document`; there is no `update_document`, `delete_document`, or
   `list_documents` among them, and `repository.create_document` mints a fresh `documentId` on
   every call — pinned non-idempotent by `test_create_document_is_non_idempotent_on_retry`
   (`falkor-chat/server/tests/test_repository.py`). This mirrors the codebase's own stated
   convention (`DESIGN.md` §9: "Create channel / thread — Non-idempotent — a retried create mints
   a new id"). Chat/document ingestion was built for **finite, one-shot** corpora — a message, a
   file dropped in once. A distilled KB file is the opposite: it is a **living document, edited
   and re-distilled routinely** (this repo's own `AGENTS.md` states the general rule bluntly: "An
   open item is rewritten, not appended to" — and a KB file is exactly that kind of document).
   Re-ingesting an edited file after every distillation pass would `CREATE` a brand-new `Document`
   + `Chunk`s alongside the old ones, with **no way to retract the superseded chunks** —
   `search_documents` would keep surfacing stale, possibly-reversed advice indefinitely. This is a
   correctness defect for this use case, not a style preference.
2. **Chunking granularity mismatch — worse than a first read suggests.** `falkorchat.chunking.split_into_chunks`
   (paragraph → sentence → hard-cut at ~1000 chars, 150-char overlap, no heading awareness) is a
   generic prose splitter with no notion of this corpus's actual unit boundaries. The requirements
   doc's own working assumption — "one `##`-heading-delimited, self-contained technique per
   section, 50–400 words" — turns out not to hold even for the file it's modeled on:
   `data-scientist`'s method note (full text: `agent-knowledge-base-strategy-ml.md`) read
   `claude/analyst/review-techniques.md` in full and found several headings bundling five or more
   independently-attributed sub-claims past 1,500 words. A mechanical chunker with **no** heading
   awareness at all does categorically worse than a heading-aware one already would — it will
   routinely pack unrelated techniques together or sever a technique's own caveat paragraph from
   the claim it bounds. This corpus needs a **claim-aware** unit, which falkor-chat's chunker
   cannot produce and this plan's own schema (§2) is designed to hold directly instead.
3. **Tenancy and side-effect mismatch.** `ingest_document` targets a `ws:{workspaceId}` tenant
   graph — there is no natural workspace for team-wide agent-engineering knowledge (`ws:acme`/
   `ws:globex` are demo tenants for the salesperson/chat proofs), so this option would need a new,
   purpose-built workspace carrying the full chat/thread/channel schema overhead for a corpus that
   has none of that shape. It also triggers `IngestionPipeline`'s LLM-based entity/relationship
   extraction on every chunk by default (`background._safe_extract`, scheduled alongside
   `_safe_embed_chunk` from `mcp.ingest_document`) — machinery for building an entity graph out of
   prose, which has no clear payoff for a corpus of imperative engineering rules and would need to
   be selectively suppressed per call, one more deviation from "the machinery as-is."

None of these individually is fatal, but stacked they mean Option B's literal-reuse convenience
is bought at the cost of a capability this use case structurally needs (retract-and-replace) that
the reused machinery was never built to have — and reusing it as-is would mean either accepting
permanently accumulating stale content, or first extending falkor-chat's own `Document`/`Chunk`
model with update/delete semantics it doesn't have today, which is no longer "reuse," it's new
development inside someone else's component.

### Option A (`kaizen_team`) — the recommendation

`kaizen_team` is already the substrate every agent's raw capture lives in and is already wired
into every agent session via `cypher-mcp` (§4.3, §4.4 spell out the two small new pieces this
needs). Choosing it means:

- **Reuse the falkor-chat *decisions*, not its package.** Same embedding model (Qwen3-Embedding-0.6B,
  1024-dim, `falkor-chat/docs/DESIGN.md` §1.3), same LM Studio OpenAI-compatible endpoint
  (`falkor-chat/config/opencode.example.json`: `http://localhost:1234/v1`,
  `text-embedding-qwen3-embedding-0.6b`), same in-graph vector-index DDL shape, same
  vector-then-traversal hybrid Cypher shape (`falkor-chat/docs/QUERIES.md` §6) — everything about
  *how* falkor-chat does GraphRAG is reused faithfully. What is **not** reused is its Python
  package (`falkorchat.embedding.EmbeddingWorker`) or its running server process, because neither
  exists for `kaizen_team` and importing across `falkor-chat/server`'s own venv boundary would
  break this monorepo's stated "independent, self-contained components" convention (root
  `AGENTS.md`) for a saving of maybe 60 lines of client code (§4.2).
- **Retract-and-replace is native**, not missing: `cobb`'s existing curator role already has an
  exact-skeleton `DETACH DELETE`-by-`entryId` write shape for `:KaizenEntry`
  (`cypher-mcp/server.py::_CURATOR_CLEAR_RE`); §4.4 adds the twin for `:KnowledgeEntry`. Revising
  a technique is delete-old + create-new, the same "rewritten, not appended to" discipline this
  repo already applies to every living document, translated one layer down into the graph.
- **Claim-level chunking is a curator judgment call either way, and Option A is the only one that
  can actually make it.** `data-scientist`'s finding (above) means neither option gets clean
  chunking for free — but Option A's write path is a `cobb`-issued Cypher write per node (§4.1),
  so a curator judging "this heading is really three claims" writes three nodes directly. Option B
  has no such seam: its chunker runs mechanically over whatever text it's handed, with no way for
  a human/agent judgment call to override where one chunk ends and the next begins.

**Stakeholder review outcome (2026-09-10, post-dispatch — see the block notice at the top of this
document and §8): declined, on principle rather than on the evidence.** The CPG-confirmed
create-only limitation stands unchallenged; what the stakeholder rejected is treating it as a
reason to *route around* falkor-chat with a second graph. Their stated reasoning: falkor-chat is
meant to be *the* substrate for agent-and-human knowledge generally, so a gap in its document
store is worth fixing in falkor-chat itself — exactly the "willing to sponsor adding document
update/delete to falkor-chat itself" branch this section already named as what would flip the
recommendation, now taken. That work is scoped separately
(`falkor-chat/docs/requirements/document-ingestion2.md`), and **this document's Option A/Option B
choice is revisited once it lands** — not assumed to flip automatically to Option B just because
the capability then exists (the stakeholder was explicit on that point). §2 onward stays as
Option A's correct design, on hold, not re-cast against falkor-chat's schema pre-emptively: there
is no falkor-chat update/delete capability to design against yet, and prescribing one is that
successor requirements doc's job, not this plan's.

---

## 2. Graph schema — distilled knowledge in `kaizen_team`

`kaizen_team` today holds only `(:Agent {agentId})`/`(:KaizenEntry {...})` with `PRODUCED`/
`MENTIONS` edges (`claude/AGENTS.md`; `skills/agent-maintenance/SKILL.md` §5). This plan adds one
new label and two new edge types, kept structurally distinct from the raw-capture shapes so
retrieval never has to filter pending-review noise out of a search:

```
(:Agent {agentId:'cobb'})-[:CURATED {sessionId}]->(:KnowledgeEntry {
  entryId, topic, title, text, embedding: vecf32, sourceEntryIds, distilledAt, familyId
})
(:KnowledgeEntry)-[:FOR]->(:Agent {agentId:'<owning-agent>'})
```

- **`CURATED`** (provenance: who distilled it) mirrors `PRODUCED`'s direction (curator → entry).
- **`FOR`** (which agent's KB this belongs to) mirrors `MENTIONS`'s direction (entry → agent) —
  reusing the already-indexed/constrained `Agent.agentId` anchor from M8 (`kaizen-agent-ontology.md`),
  so no new index is needed on the agent side.
- **Node granularity is one distinct, independently-actionable claim, not one Markdown heading**
  (`-ml.md` Recommendation 2) — the common case is still one node per existing `##`-headed
  technique (most headings already are one claim), but a heading bundling several independently
  "Origin:"-attributed sub-claims, or materially longer than ~500 words spanning more than one
  verifiable claim, becomes several nodes. This is a curator (`cobb`) judgment call, not a
  mechanical rule — see §6 for how it lands in migration.
  - `title` — the claim's short name (a `##` heading's text, or a sub-claim's own bolded
    sub-header when a heading was split).
  - `text` — the full distilled prose for that one claim, including its own worked
    example/counter-example (`-ml.md`: "do not go finer than one claim... the worked example is
    what makes a rule actionable").
  - `familyId` — shared across every node produced by splitting one original heading, so a
    retrieval hit can optionally pull its siblings (§4.3) — the one traversal edge this corpus
    earns (`-ml.md` Recommendation 3). `null`/omitted for a heading that stayed one node.
- **Fields**, one node per claim (per the granularity rule above):
  - `entryId` — `uuid4`, same convention as `KaizenEntry`.
  - `topic` — the owning KB file's slug (e.g. `review-techniques`), for future filtering/export —
    a plain filterable property, not an edge (`-ml.md`: "don't build a graph edge for something a
    property filter already answers").
  - `embedding` — `vecf32`, **written by a separate step, not at creation** (§4.2) — mirrors
    falkor-chat's own decoupling: `cobb`'s write never blocks on an embedding HTTP call. **Stored
    unprefixed** (`-ml.md` Recommendation 1 — the asymmetric instruction prefix applies to the
    *query* side only, at retrieval time; see §4.3).
  - `sourceEntryIds` — comma-joined `KaizenEntry` UUIDs this claim was distilled from, for
    traceability (plain string, not a list property — **graph-dba should verify FalkorDB's list-typed
    property support live before switching this to a native array**; nothing here depends on it).
  - `distilledAt` — ISO-8601 write time.
- **No `topic` index at creation** — `graph-dba`'s own rule applies (`falkor-chat/AGENTS.md` "Rules
  for future work" #3: profile before tuning). Add one only if `GRAPH.PROFILE` on the shipped
  retrieval query (§4.3) shows it's needed; today's expected corpus size (a few hundred entries
  across ~10 agents) makes a full vector-index seed cheap enough that a `topic` pre-filter is
  unlikely to matter.
- **Vector index** — `CREATE VECTOR INDEX FOR (k:KnowledgeEntry) ON (k.embedding) OPTIONS
  {dimension: 1024, similarityFunction:'cosine'}`, plus a range index + `UNIQUE` constraint on
  `KnowledgeEntry.entryId` (mirrors `KaizenEntry.entryId`'s existing pair). **Both are schema
  DDL, and `cypher-mcp` refuses all DDL unconditionally regardless of agent** (`server.py`
  comment: "schema DDL included ... no carve-out for schema statements over data statements, even
  from a valid, recognized agent slug" — already hit twice, for `kaizen_team`'s original
  `KaizenEntry.entryId` index and M8's `Agent.agentId` index). `graph-dba` runs this DDL directly
  via `redis-cli`/`GRAPH.QUERY`, one-time, same as those two precedents.

---

## 3. Sequencing

Per the requirements doc's explicit instruction (FR-1, decision log): K-030's four agents get
interim relief **now**, independent of this plan's build-out, then migrate later alongside the
existing six.

| Stage | What | Owner | Blocks on | Authorized to proceed? |
|---|---|---|---|---|
| **0** | K-030's four interim flat-file KBs (`teco`, `architect`, `data-scientist`, `tdd-engineer`), authored per the existing pattern | `cobb` | nothing — proceed immediately; not part of this plan's deliverable (out of scope per the requirements doc), but see the FR-7 authoring constraint below | **Yes — proceed now, unaffected by the block** |
| **1** | Schema DDL: `KnowledgeEntry` indexes + vector index | `graph-dba` | §2 | **No — blocked** |
| **2** | `cypher-mcp` write-shape extension (curator distill-write + knowledge-clear) | `coder`/`graph-dba` | Stage 1 | **No — blocked** |
| **3** | Embedding-backfill script (out-of-band, direct FalkorDB) | `devops` | Stage 2 | **No — blocked** |
| **4** | `kaizen-rag-mcp` — new read-only retrieval MCP server | `coder`/`devops` | Stage 1 (needs the vector index to exist to test against) | **No — blocked** |
| **5** | `cobb`'s `agent-maintenance` SKILL.md §5 gains a graph-write routing option + the KB-export step | `cobb` | Stages 2–4 (explicitly **out of scope** for this plan to redesign — see requirements doc's "Out of scope" list — but §5 below names the exact hook point so that follow-up is a small, well-specified edit, not a rediscovery) | **No — blocked** |
| **6** | Migrate the five/six existing KBs (FR-6/AC-4) + the four interim ones from Stage 0 (FR-7/AC-5) — a `cobb`-led curator judgment pass, not a mechanical script (§6) | `cobb`, tooled by `graph-dba`/`coder` | Stages 1–4 | **No — blocked** |
| **7** | Generated Markdown export wired into `cobb`'s distillation pass (§6, answering Open question #2) | `cobb`/`devops` | Stage 6 | **No — blocked** |

**Stages 1–7 are blocked on `falkor-chat/docs/requirements/document-ingestion2.md`** (see the
notice at the top of this document and §1's revised recommendation section) — not on any
technical dependency internal to this plan. Nothing below this point in the document authorizes
work; it documents the design that stays ready to resume if/when Option A is chosen again after
that successor feature lands.

**FR-7 authoring constraint for Stage 0 (flag to whoever authors the four interim files, not a
redesign of that work):** author them with the same convention the existing six already use — one
`##`-heading-delimited technique per section, each holding one claim where possible (visible in
`claude/analyst/review-techniques.md` today, imperfectly — see §6). This keeps FR-7's bar
("must not cost materially more migration effort... than the pre-existing five/six") satisfied by
construction: both sets go through the **same** Stage-6 curator judgment pass, so neither is
privileged or penalized relative to the other, even though (per `-ml.md` Recommendation 2) that
pass is not the purely mechanical script this plan originally assumed for either of them.

---

## 4. Design & rationale — the two new pieces of machinery

`cypher-mcp`'s `query` tool is deliberately Cypher-only (`server.py`: "one tool, two required
parameters plus one optional... no `params`, no `mode`"; frozen by `docs/plans/cpg-query-access.md`
§4.4) — it cannot turn free text into a vector, because FalkorDB itself doesn't compute embeddings,
only stores/searches ones already computed. That gap is real regardless of which substrate is
chosen (Option B's `search_documents` solves it only because falkor-chat's server does the
embedding call inside its own process, not through any generic tool) — so Option A needs two new,
small, narrowly-scoped pieces, neither of which is a `cypher-mcp` change to its frozen one-tool
contract.

### 4.1 The write shape — `cypher-mcp` extension (answers Open question #3)

Two new curator-only shapes, following the exact style of the six that exist today
(`cypher-mcp/server.py::authorize_write`, `cypher-mcp/README.md` "Writing through this tool"):
narrow, exact skeletons, matched via brace-balanced text scanning (not a real parser), curator-gated
via the existing `CYPHER_MCP_CURATOR_AGENTS` set (default `cobb`). Neither constrains the map's
*field* names — like the existing shapes, only the structural skeleton (which clauses, which
labels, which edge types, in which order) is checked; the property list inside `{...}` is free-form,
matching how `KaizenEntry.fact`/`evidence`/`context` are convention-enforced, not regex-enforced.

**Shape 7 — curator distilled-knowledge write** (create only; no embedding — that's §4.2):

```cypher
MERGE (c:Agent {agentId: 'cobb'})
MERGE (t:Agent {agentId: '<owning-agent-slug>'})
CREATE (c)-[:CURATED {sessionId: '...'}]->(k:KnowledgeEntry {
  entryId:'<uuid4>', topic:'<topic-slug>', title:'<short title>', text:'<distilled prose>',
  sourceEntryIds:'<comma-joined KaizenEntry uuids>', distilledAt:'<ISO-8601>',
  familyId:'<shared id across siblings split from the same heading, or omit this key entirely>'
})
CREATE (k)-[:FOR]->(t)
```
`familyId` is optional — like `sessionId` on the existing producer-write shape, omit the key
entirely rather than writing `null` when a claim was not split from a larger heading (§2). `agent`
must be a recognized curator (today, `cobb` only) — same rejection message pattern as the
existing MENTIONS-write/edge-resolve shapes when it isn't. No `RETURN` allowed to follow (mirrors
the producer-write shape's own trailing-clause restriction and its specific near-miss message —
`server.py::_producer_write_trailer_message`); issue a separate follow-up read for the `entryId`.

**Shape 8 — curator knowledge-clear** (mirrors the existing `_CURATOR_CLEAR_RE` exactly, new label):

```cypher
MATCH (k:KnowledgeEntry {entryId: '...'}) DETACH DELETE k
```
Used for §1's retract-and-replace revision (delete old, shape-7-create new with a fresh `entryId`)
and for outright retirement of a technique that no longer applies.

**Implementation footprint in `cypher-mcp/server.py`:** two new brace-matched skeleton recognizers
(mirroring `_producer_write_shape_match`/`_CURATOR_CLEAR_RE`), two new branches in
`authorize_write()`, and updates to the two **frozen, test-pinned** strings (`TOOL_DESCRIPTION`,
`SERVER_INSTRUCTIONS`) from "6 shapes" to "8 shapes" with the two new ones described — the MCP
server's own `SERVER_INSTRUCTIONS` string is what `claude/AGENTS.md` and this session's own MCP
instructions block quote verbatim today, so that quoted copy needs the same edit in the same
change. `cypher-mcp/README.md`'s "Writing through this tool" numbered list (1–6) gains entries 7–8
in the same style, plus a note in `docs/plans/kaizen-agent-ontology.md`'s lineage (or a new small
`docs/plans/agent-knowledge-base-strategy-graph.md`, `graph-dba`'s call) documenting the two new
regexes' exact skeleton, matching how the M8 shapes were documented.

### 4.2 Embedding — out-of-band, direct-FalkorDB, not through `cypher-mcp`

`cobb` (an agent) has no embedding-model tool available to it, so shape 7 above never carries an
embedding — exactly mirroring falkor-chat's own posture (`DESIGN.md` §9: "Embed messages: async
worker... Decouple embedding latency from the post path"). A new script,
**`claude/scripts/kaizen_embed_backfill.py`**, owned by `devops` (precedent: K-017's
`ensure-services.sh`, and `devops`'s own remit — "automation, automation scripts" — `claude/AGENTS.md`
roster), does the rest:

- Connects to FalkorDB directly (the `falkordb` Python client, same dependency `cypher-mcp` already
  pins), **not through `cypher-mcp`** — this is a batch/service-owned write, the same category as
  `falkor-chat/scripts/bootstrap_schema.sh`'s DDL and the `joern-cpg` pipeline's bulk loads: none
  of those go through `cypher-mcp` either, because `cypher-mcp`'s write authorization exists to
  attribute *agent-session* writes, and this script runs outside any single agent's session.
- Queries `kaizen_team` for `:KnowledgeEntry` nodes with no `embedding` property.
- For each, calls LM Studio's `/v1/embeddings` directly (`http://localhost:1234/v1`, model
  `text-embedding-qwen3-embedding-0.6b`, per `falkor-chat/config/opencode.example.json` — the same
  endpoint/model falkor-chat's own `EmbeddingWorker` calls, just via a small ~30-line
  self-contained HTTP client in this script rather than an imported `OpenAICompatibleEmbedder`,
  per §1's self-containment reasoning) and `SET k.embedding = vecf32($v)`.
- **Idempotent and safe to re-run** — the "no `embedding` property" filter is itself the
  idempotency guard; running it twice in a row is a no-op the second time.
- **Invocation**: on-demand, run by `devops` (or `cobb`, as the last step of a distillation pass —
  see §5) after new/revised `:KnowledgeEntry` nodes land. No standing daemon: unlike falkor-chat's
  live chat traffic, this corpus grows in small, human-cadence batches (one distillation pass at a
  time), so a background loop is disproportionate infrastructure for the volume.
- Verify a multi-line `text` property write byte-exact via `size()`/re-read, never by eyeballing a
  truncated tool response (`cypher-mcp/README.md`'s own documented gotcha from the 2026-08-20
  migration — same trap applies here).

### 4.3 Retrieval — a new, small, read-only MCP server: `kaizen-rag-mcp`

One new top-level component, sibling to `cypher-mcp/` and `mcp-monitor/`, following `cypher-mcp`'s
own containerization pattern (`Dockerfile`, `docker-run.sh`, `build.sh` with a content-hash tag,
`setup.sh`/`run.sh` host-venv fallback — copy and adapt that scaffolding; it's infra boilerplate,
not application logic, so reusing it directly is in the spirit of "reuse what exists," unlike the
Python-package question in §1/§4.2). **Deliberately not a new tool bolted onto `cypher-mcp`** —
that tool's one-tool, two-required-parameter contract is explicitly frozen
(`cpg-query-access.md` §4.4), and mixing "generic Cypher passthrough" with "domain-specific
semantic retrieval workflow" in one tool would blur both. Exposes exactly one tool:

```
mcp__kaizen_rag__search(agentId: str, situation: str, topK: int = 5) -> str
```

Body, per `-ml.md` Recommendations 1/3:

1. **Embed `situation` with the asymmetric query-instruction prefix**, not raw text — the one
   concrete, free lever `-ml.md` identifies: `falkor-chat`'s own `EmbeddingWorker` never needed
   this (chat-to-chat similarity is near-symmetric), but this system's retrieval is sharply
   asymmetric (a short informal situation sentence vs. a long, formal, technique-dense entry).
   Call LM Studio with:
   ```
   "Instruct: Given a coding agent's description of its current situation, retrieve the "
   "distilled technique or rule that applies to it.\nQuery: {situation text}"
   ```
   Stored `:KnowledgeEntry.embedding` values stay **unprefixed** (§2) — only the query side changes.
2. **Run the hybrid vector query**, adapted from falkor-chat's canonical shape
   (`falkor-chat/docs/QUERIES.md` §6): the `FOR`-edge scope replaces that query's channel scope, a
   mandatory score-floor `WHERE` replaces open-ended top-K, and the `MENTIONS`-based "related
   context" expansion is dropped (no analogue — a `KnowledgeEntry` isn't part of a conversation
   thread, `-ml.md` Recommendation 3 confirms), replaced by an **optional `familyId` sibling pull**
   for a split entry:
   ```cypher
   CALL db.idx.vector.queryNodes('KnowledgeEntry', 'embedding', $seedK, $qVec)
   YIELD node AS k, score
   MATCH (k)-[:FOR]->(:Agent {agentId: $agentId})
   WHERE score <= $scoreFloor
   OPTIONAL MATCH (sibling:KnowledgeEntry {familyId: k.familyId})
     WHERE k.familyId IS NOT NULL AND sibling.entryId <> k.entryId
   RETURN k.entryId, k.title, k.text, score,
          collect(DISTINCT sibling.entryId) AS familyMembers
   ORDER BY score ASC
   LIMIT $topK
   ```

**Parameters, `-ml.md` Recommendation 3, not this plan's own guess:** `topK = 5` (not
falkor-chat's `10` — that figure is a pre-traversal ANN *seed* budget there, not a final answer
count; with no traversal narrowing the result here, top-K is directly what reaches the querying
agent's context). `$scoreFloor` is a **calibrated config value, not an asserted constant** — derived
from the golden-set pilot (§7) by finding where the true-positive and known-irrelevant
cosine-distance distributions cross, documented with the run that produced it, and re-derived
whenever the corpus or the query prefix changes materially. Ship a conservative placeholder
(disabled / no floor) only until that first pilot run produces a real value — do not invent one to
unblock Stage 4's build; the floor is what stops FR-4's real failure mode ("an agent trusted an
irrelevant top-1"), not a nice-to-have.

Read-only throughout (`GRAPH.RO_QUERY`, mirroring `cypher-mcp`'s own read-path guarantee) — this
server never writes, so it needs no `authorize_write()`-style authorization model of its own, a
meaningfully smaller surface than `cypher-mcp`. Rendered as plain text, same rationale as
`cypher-mcp::format_result` (JSON roughly doubles token cost). **Degraded fallback if this server
is ever unavailable**: `cypher-mcp` (already wired everywhere) still serves a plain keyword/`CONTAINS`
Cypher query over `KnowledgeEntry.text` — worse recall, but zero new dependency, mirroring
`cypher-mcp`'s own documented `redis-cli` fallback posture.

**New operational dependency, flagged, not silently absorbed:** every knowledge-base lookup now
needs LM Studio reachable (Windows host via WSL2 mirrored networking, the same path `severino`
already depends on) — a failure mode flat-file reads never had. See §7.

### 4.4 The file counterpart — recommended answer to Open question #2

**Recommendation: yes, but generated, never hand-edited.** A small script (part of Stage 7,
`devops`/`cobb`) queries all of one agent's `KnowledgeEntry` nodes (via the `FOR` edge, ordered by
`topic`/`title`) and renders them back into the **exact Markdown shape the flat files use today**
(`## <title>` + body), overwriting `claude/<agent>/<topic>.md` wholesale — never appended to,
matching this repo's own "a document that freezes does not move... a living document is compacted,
not only appended to" discipline, here taken to its logical end: the file is a pure, regenerated
export. `cobb` commits it as part of every distillation pass's existing doc-curation duty (the
same "doc updates are part of every unit's done-condition" convention `teco` already runs by).

This resolves the tension the requirements doc names directly: human review/diffability via git
history (every rewrite is a normal, diffable commit) **without** reintroducing FR-4's own
"read-whole-file" problem for retrieval, because **no agent ever reads this file for retrieval** —
only `kaizen-rag-mcp` (§4.3) is ever queried at runtime; the file exists solely so a human can `git
diff`/`git blame` what changed, exactly like every other generated-but-committed artifact in this
repo. `cobb`'s authoring surface moves from hand-editing the `.md` file to issuing shape-7 writes
(§4.1) — a real workflow change, named here so `agent-maintenance` SKILL.md §5's eventual update
(Stage 5, out of this plan's scope) has an unambiguous target.

---

## 5. `cobb`'s distillation hook point (named, not redesigned)

Per the requirements doc, redesigning SKILL.md §5 step-by-step is explicitly out of scope here —
it "follows once the storage target is decided." This plan decides the target (§1) and names the
exact points that future edit will touch, so it's a small, well-specified follow-up rather than a
rediscovery:

- Step 3's routing list ("An on-demand knowledge base — the `graph-dba/falkordb-quirks.md`
  pattern...") gains a new destination: **the graph-backed knowledge base**, i.e. a shape-7 write
  (§4.1) instead of an `Edit` to a flat file, for any entry routed there once Stage 2 ships.
  Routing to the *prompt* or to *project docs* is unchanged.
  - **What is genuinely new here, named up front because the deferral needs a place to land, not
    just a mention:** whether a given promotion, once graph-write becomes available, should write
    *directly* to `kaizen_team` or continue landing in the flat file with Stage 7's export handling
    the graph side — i.e., which of FR-2/FR-5's two legalized paths ("writing it directly, or...
    the store ingesting the flat file") `cobb`'s day-to-day workflow actually follows — is not
    decided by this plan and is squarely Stage 5's call, not mine to silently pick either way now.
    §4.1/§4.4 already assume "write directly, export is one-way" throughout (the simpler of the
    two, and the one Option A's own reasoning in §1 argues for), so Stage 5 should encode that
    choice explicitly rather than re-derive it — but if Stage 5's implementer finds a concrete
    reason to prefer file-first-then-ingest instead, that is theirs to make, not inherited from
    this plan by default.
- Step 4's clear sequence gains the `:KnowledgeEntry` equivalent of the existing count-and-decide
  logic **only if** an entry ever needs to be `MENTIONS`-tagged onto a second agent post-distillation
  (unclear today whether `KnowledgeEntry` needs a `MENTIONS`-equivalent at all, since `FOR` is
  singular-owner by construction) — flag for Stage 5, not resolved here.
- Stage 7's export step (§4.4) is a new step 6, after step 4's log-and-clear, scoped to "which
  agents had a `KnowledgeEntry` touched this pass" (not a full re-export every pass).

---

## 6. Migration (FR-6/AC-4, FR-7/AC-5) — Stage 6

**Revised from a mechanical script to a `cobb`-led curator judgment pass**, per `-ml.md`
Recommendation 2's finding that the requirements doc's "one heading = one technique" assumption is
false for the actual files (`review-techniques.md` alone has at least two headings bundling 5+
independently-attributed sub-claims past 1,500 words). A splitter keyed on `##` boundaries alone
would silently ship oversized, multi-topic nodes in exactly the file that most needs fine
retrieval. The process instead:

1. **`graph-dba`/`coder` build a candidate-flagging tool** (not a splitter) — per file, per `##`
   section: word count, count of bolded sub-headers/enumerated checks, count of "Origin:" mentions.
   Flags a heading as a split candidate past `-ml.md`'s stated trigger (~500 words spanning more
   than one verifiable claim, or multiple independently-attributed sub-claims) — mechanical,
   scriptable, and exactly the kind of thing a tool should do: surface candidates, not decide them.
2. **`cobb` reviews every flagged heading and decides the actual split boundary** — the same
   editorial judgment it already exercises during ordinary distillation (§5), applied retrospectively
   here. An unflagged heading migrates as a single node unchanged.
3. **`cobb` issues one shape-7 write per resulting node**, attributed `agent='cobb'` (the standing
   curator; the migration runs on its behalf even when scripted assistance drives the mechanics —
   same posture the `cypher-mcp` README documents for prior bulk kaizen-graph migrations: "three
   independent agents migrating free-text kaizen entries into `kaizen_team` in one rollout"), with
   a shared `familyId` (§2) across every node produced from the same original heading. Verify each
   multi-line `text` write byte-exact via `size()` (§4.2's documented gotcha).
4. Stage 3's embedding-backfill script then picks up every new node with no special-casing (it
   already filters on "no `embedding` property," regardless of how the node was created or how
   many nodes one heading became).

**Content-loss check (AC-4/AC-5's own bar), revised to match:** a plain node-count-equals-heading-count
check no longer applies once headings can become several nodes. Instead: every word of a source
section's body must be accounted for by exactly one resulting node's `text` (a `cobb`-verified
partition, not merely a count match) — spot-checked via a scripted diff of "source section text"
against "concatenation of its family's node texts," not a manual read of the whole corpus.

**Effort, sized rather than assumed away (`-ml.md`'s own flagged risk):** this is real, additional
migration work for the pre-existing six KBs relative to this plan's original "mechanical split"
assumption — but because Stage 0's four interim KBs go through the identical process (§3's revised
FR-7 constraint), the *relative* cost FR-7 actually gates stays even; only the *absolute* cost of
Stage 6 as a whole rose, which is `data-scientist`'s finding to weigh, not a violation of FR-7's
own bar.

Whether the source `.md` files are then deleted or left in place as the *first* generation of
§4.4's export artifact (functionally identical once migrated, since the export renders back to the
same shape) is Stage 6/7's implementation call, not this plan's.

---

## 7. Test strategy

- **`cypher-mcp` shapes 7/8** — offline, `fake_client`-fixture unit tests in
  `cypher-mcp/tests/test_server.py`, mirroring the existing naming/coverage pattern for shapes 1–6
  (`test_producer_write_with_matching_agent_succeeds`, `..._mismatched_agent_is_rejected`,
  `test_curator_clear_shape_with_cobb_succeeds`, `..._with_non_curator_is_rejected`, plus a
  trailing-clause near-miss test mirroring `test_producer_write_with_trailing_return_clause_gives_specific_trailer_message`).
  Both the in-container test gate and the host-venv suite must stay green (`cypher-mcp/README.md`
  "Smoke check").
- **`kaizen_embed_backfill.py`** — unit: idempotency (already-embedded nodes untouched, a fixed
  stub embedder for the HTTP call); a live-marked smoke test (mirrors `cypher-mcp`'s own `-m live`
  convention) against a throwaway graph + real LM Studio.
- **`kaizen-rag-mcp`** — offline: query-shape correctness against a fake FalkorDB client (same
  `fake_client` pattern as `cypher-mcp/tests/conftest.py`) and a stub embedder; live: an end-to-end
  round trip against a throwaway `kaizen_team`-shaped scratch graph + real LM Studio, mirroring
  `cypher-mcp`'s own unique-scratch-graph-name discipline (C-321) to stay concurrency-safe.
- **AC-2 (semantic match under reworded queries)** — the golden set design is `-ml.md`
  Recommendation 4, quoted once here: a **~40-pair, independently-authored** (never `cobb` — a
  curator's own paraphrase would echo the stored entry's vocabulary and understate the real gap
  AC-2 exists to close; prefer real situation descriptions pulled from session transcripts/raw
  `kaizen_team` entries over synthetic ones) golden set, **stratified** across (a) every KB
  weighted by entry count, so no single dense file dominates or is silently absent, (b)
  code/regex/shell-heavy entries vs. prose-only entries — this is what actually answers whether
  Qwen3-Embedding-0.6B handles this corpus's density, rather than leaving it asserted either way,
  (c) 4–6 explicit **negative** queries with no matching entry, to confirm the score floor rejects
  rather than force-feeds, and (d) a handful of near-duplicate stress pairs to confirm the
  embedding discriminates rather than clusters broadly. Scored as **recall@5 primary, MRR
  secondary**, reported with a **Wilson-interval CI** (this lab's established small-n convention),
  never a bare percentage. **Sequencing:** a ~15–20-pair pilot on one dense agent's KB once the
  pipeline plus a first embedded slice exists (this is also what produces the first calibrated
  `$scoreFloor`, §4.3); the full ~40-pair set becomes the AC-2 regression gate once Stage 6
  migration covers both the four interim KBs and the pre-existing five/six, re-run whenever a bulk
  migration tranche lands or the embedding model/prefix convention changes.
- **AC-4/AC-5 (migration, no content loss)** — the scripted partition-diff check in §6 (every
  source word accounted for by exactly one resulting node, verified per family, not by a
  node-count comparison), run once per migrated file, not a manual read.
- **Full pipeline smoke** — one hand-run distillation entry through shape-7 write → backfill script
  → `kaizen-rag-mcp` search, before Stage 6's bulk migration, so a schema/query mistake is caught
  on one entry rather than discovered mid-migration.
- **`kaizen-rag-mcp`'s query-prefix template (§4.3)** gets its own unit test asserting the exact
  string sent to the embedder for a given `situation` — a silent drift in that template (e.g. a
  future edit dropping the `Instruct:`/`Query:` framing) would degrade retrieval quality with no
  loud failure anywhere else in the pipeline.

**Two questions `-ml.md` explicitly leaves open, carried forward rather than resolved here**
(Recommendation 1's own "what would change my mind," and the risks section of that note): whether
Qwen3-Embedding-0.6B's code-retrieval sub-score actually holds up on this corpus's code/regex-dense
entries — answered only once the stratified golden-set run in this section produces a number, not
asserted now either way — and the exact `$scoreFloor`/`topK` values, which are provisional
(§4.3) until that same run.

---

## 8. Risks & open questions

- **BLOCKED, 2026-09-10 — this is now a status, not a risk, but recorded here as the authoritative
  note.** The stakeholder reviewed §1's evidence and declined Option A on principle: falkor-chat is
  meant to be the one substrate for agent-and-human knowledge generally, so its document-ingestion
  gap (confirmed real, via the CPG) is worth closing in falkor-chat itself rather than routed
  around. A new requirements doc, `falkor-chat/docs/requirements/document-ingestion2.md` (successor
  to the archived `document-ingestion.md`), specifies that work; §1–§7 here resume only once it
  exists and the substrate choice is revisited on the merits — explicitly not auto-flipped to
  Option B. **Only §3 Stage 0 is authorized to proceed in the meantime.** This entry supersedes the
  "flagged for stakeholder confirmation" framing below, which is kept as the historical record of
  what was flagged and resolved, not as still-open.
- **Substrate call (§1) — was flagged for stakeholder confirmation; now resolved, see above, not
  in the direction this plan recommended.** Strong evidence (CPG-verified) was never in question;
  the stakeholder weighed a principle (one substrate, not several) over the gap-avoidance framing
  this plan offered.
- **New runtime dependency.** Every KB lookup now needs LM Studio reachable — a failure mode flat
  files never had. This is a real trade-off (prompt-bloat relief vs. new fragility), worth a
  deliberate stakeholder nod rather than an implicit one, especially since `severino`'s own kaizen
  history already documents WSL2↔LM Studio connectivity as a live, recurring source of friction.
  `kaizen-rag-mcp`'s `cypher-mcp` keyword fallback (§4.3) bounds the blast radius but doesn't
  eliminate the dependency.
- **`.mcp.json` project-scope approval.** A new server needs the same one-time trust approval
  `cypher-mcp` needed (`cypher-mcp/README.md`'s documented per-directory approval-scoping quirk) —
  no new risk, just a step that's easy to forget when standing up a second MCP server.
- **File counterpart recommendation (§4.4)** is architect's call within its remit, but since it
  resolves Open question #2 by construction of choosing §1, it's presented for stakeholder
  confirmation alongside the substrate call, not assumed accepted.
- **Migration effort rose materially from this plan's original assumption** (§6) — Stage 6 is now
  a `cobb`-led curator pass, not a mechanical script, because `-ml.md` Recommendation 2 found the
  "one heading = one technique" premise false against the real files. Sizing/scheduling that pass
  (especially for `review-techniques.md`, the densest KB) is worth a deliberate look before Stage 6
  is dispatched, not an assumption it's a quick batch job.
- **Code-retrieval quality on this corpus is genuinely unverified** (`-ml.md` Recommendation 1) —
  Qwen3-Embedding-0.6B's fit for code/regex-dense entries specifically stays open until the
  stratified golden-set run (§7) produces a number; the fallback if it disappoints is a documented,
  cheap, re-embed-only upgrade (`qwen3-embedding:4b`, same 1024-dim family, no schema change), not
  a redesign.
- **`$scoreFloor`/`topK` are provisional** (§4.3, §7) until the golden-set pilot run — do not ship
  Stage 4 with an invented threshold; ship with the floor disabled and calibrate from the pilot.
- **Stage 5's file-first-vs-direct-write question (§5)** is named but not resolved — genuinely
  Stage 5's call once that implementer is looking at the live system, not something to guess now.

---

## Ready to implement — Stage 0 only; everything else is blocked

Plan at `claude/docs/plans/agent-knowledge-base-strategy.md`, with its ML method note at
`claude/docs/plans/agent-knowledge-base-strategy-ml.md`.

**What's actually actionable right now: §3 Stage 0 only** — K-030's four agents (`teco`,
`architect`, `data-scientist`, `tdd-engineer`) get interim flat-file KB relief per the existing
six-agent pattern, unaffected by anything below. Dispatch that; do not dispatch anything else from
this plan.

**Everything from §2 onward is blocked** (top-of-document notice, §1, §3, §8) on a new, separate
requirements doc, `falkor-chat/docs/requirements/document-ingestion2.md` — the stakeholder declined
this plan's §1 recommendation (`kaizen_team` as substrate) on principle, not on the CPG evidence
behind it: falkor-chat is meant to be the one substrate for agent-and-human knowledge generally,
so its document-ingestion update/delete gap is worth closing there rather than routed around. Once
that successor feature is specified and built, this plan's Option A/Option B choice is revisited
on the merits — not assumed to flip to Option B automatically.

**Kept intact, not thrown away, for that eventual revisit:** `:KnowledgeEntry`/`CURATED`/`FOR`
schema with claim-level (not heading-level) granularity and `familyId` sibling links (§2); two new
curator-only `cypher-mcp` write shapes (§4.1); an out-of-band embedding-backfill script design
(§4.2); a `kaizen-rag-mcp` retrieval-server design with an asymmetric query-instruction prefix,
top-K=5, and a golden-set-calibrated score floor (§4.3); a generated Markdown export answering Open
question #2 (§4.4); a `cobb`-led curator judgment pass for migration, not a mechanical script,
because the "one heading = one technique" premise the requirements doc originally carried turned
out false against the real files (§6); and a ~40-pair, independently-authored, Wilson-CI-scored
golden set for AC-2 (§7) — all per the ML method note's four recommendations. None of it is
authorized to build today.
