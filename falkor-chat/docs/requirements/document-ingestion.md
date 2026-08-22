# Document Ingestion & Fusion — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-22

## Intent
The stakeholder wants to bring knowledge from **multiple external text-based sources** — documents
and agent-generated text alike — into the graph for GraphRAG, with a **fusion** step: merging/
reconciling facts that come from more than one source during ingestion, rather than treating each
source as a fully independent, unrelated blob of content. The ingested knowledge should serve
**both** (a) the AI agent answering questions in a chat channel, using the same retrieval path
chat messages use today, and (b) a **standalone queryable knowledge base**, usable independent of
any chat channel. A connected AI agent should also be able to **write** into this store via the
existing MCP front door — using it as **persistent memory** — not just read from it.

## Problem & current state
Today, falkor-chat's GraphRAG has exactly one knowledge source going into the graph: chat
messages, embedded as they're posted (`falkor-chat/docs/DESIGN.md` §8). There is no path for
ingesting knowledge from outside the chat itself.

## User stories
- As an **AI agent answering in a channel**, I want to ground my replies in ingested document
  knowledge, not just chat history, so that I can answer questions chat alone can't.
- As a **human or agent operator**, I want to query the ingested knowledge base directly,
  independent of any chat channel, so that retrieving knowledge doesn't require going through chat.
- As a **connected AI agent (via MCP)**, I want to write things I've learned into the shared
  store, so that I have persistent memory usable across sessions.
- As a **contributor (human or agent) submitting multiple documents**, I want overlapping or
  conflicting facts across those documents merged/reconciled, so that the graph doesn't end up
  with messy duplicates or silently-inconsistent knowledge.
- As a **human or connected AI agent**, I want to confirm or reject a suggested entity match, so
  that ambiguous fusion decisions don't happen silently/incorrectly behind my back.

## Functional requirements
- **FR-1** — The system can ingest content from text-based sources (plain text, Markdown,
  Mermaid, CSV, JSON, and similar — see Out of scope) into the graph as GraphRAG-retrievable
  knowledge.
- **FR-2** — Ingested knowledge is retrievable by the AI agent answering in a chat channel, via
  the same retrieval path used for chat messages today.
- **FR-3** — Ingested knowledge is also retrievable **independent of chat**, as a standalone
  queryable knowledge base.
- **FR-4** — Content authored by an AI agent is a valid ingestion source, treated the same as a
  human-supplied document.
- **FR-5** — The ingestion (write) capability is reachable via the existing MCP front door, so a
  connected agent can use it as persistent memory, not only as a reader.
- **FR-6 (fusion — conflicting facts)** — When two sources state conflicting facts about the same
  subject, **both are kept**, each carrying its provenance (source + when), rather than one
  silently overwriting the other. Readers (human or agent) weigh the conflicting facts themselves.
- **FR-7 (fusion — same-entity matching)** — The system attempts to recognize when new content
  refers to an entity/subject that already exists in the graph (e.g. "Acme Corp" vs. "Acme
  Corporation"), at some confidence level. **The matching technique itself is a design decision,
  out of scope for this document** — see Open questions.
- **FR-8 (fusion — auto-merge tier)** — A **very-high-confidence** match (e.g. an exact shared
  identifier, or near-identical content) is linked/merged **automatically**, no confirmation
  required.
- **FR-9 (fusion — suggested-match tier)** — A match that is plausible but not very-high-confidence
  is **not** linked/merged automatically. Instead it is surfaced as a **pending suggestion**;
  nothing is linked/merged until it is confirmed.
- **FR-10 (fusion — confirmation)** — A pending match suggestion can be **confirmed or rejected**
  by either a **human user** or a **connected AI agent**. Confirming links/merges the two;
  rejecting leaves them separate. **Rejection is not permanent** — a rejected pair can later be
  reconsidered (e.g. re-suggested, or manually linked) if warranted; rejecting does not
  permanently forbid a future match between the same two things.
- **FR-11 (bulk ingestion)** — The system supports ingesting **multiple documents/sources in one
  batch**, not only one document at a time.

## Out of scope
- **Binary / non-text document formats** (PDF, images, Office docs, etc.) that require dedicated
  parsing/extraction machinery — v1 handles **text-based formats** (plain text, Markdown,
  Mermaid, CSV, JSON, and similar; not a fixed/closed list) (stakeholder decision, 2026-08-22).

## Acceptance criteria
- **AC-1** — Given two sources describing conflicting facts about the same subject are both
  ingested, when the knowledge base is queried, then both facts are returned with their
  originating source and ingestion time, and neither has silently overwritten the other.
- **AC-2** — Given new content is ingested that matches existing knowledge at very-high
  confidence, when ingestion completes, then the two are linked/merged with no pending
  confirmation required.
- **AC-3** — Given new content is ingested that plausibly, but not very-confidently, matches
  existing knowledge, when ingestion completes, then the match appears as a pending suggestion
  and the two remain unlinked until confirmed.
- **AC-4** — Given a pending match suggestion, when a human user or a connected AI agent confirms
  it, then the two become linked/merged; when rejected, they remain permanently separate (or until
  re-suggested by a future ingestion, per Open questions).
- **AC-5** — Given knowledge was ingested from a document, when the AI agent grounds a chat-channel
  answer in it, then that answer's provenance traces back to the source document (mirroring how
  chat-message grounding already works today via `EMITTED` edges).
- **AC-6** — Given knowledge was ingested via the MCP front door by a connected agent, when that
  or another agent later queries the knowledge base, then the previously-written content is
  retrievable.
- **AC-7** — Given a suggested match was previously rejected, when new corroborating content later
  arrives (or a human/agent chooses to), then the two can still be linked — a past rejection does
  not permanently block a future match between the same pair.
- **AC-8** — Given a batch of multiple documents is submitted together, when ingestion completes,
  then all of them are processed (including fusion against each other and against existing
  knowledge), not just the first one.

## Related work (not part of this feature)
- `falkor-chat/docs/requirements/summary-nodes.md` (Status: Interviewing, unfinished) — condenses
  *existing* graph content into retrievable summary nodes. This feature is about bringing *new*
  knowledge in from outside the graph. Different problem; cross-referencing because both feed the
  same GraphRAG retrieval path.

## Open questions
- **OQ-1** — What counts as "very-high confidence" for auto-merge (FR-8), and what
  matching technique produces it (shared IDs, fuzzy name matching, embedding similarity, LLM
  confirmation, some layered combination)? Design decision — architect/data-scientist territory,
  not decided here. This document only fixes the *behavior* at each confidence tier.
- **OQ-2** — Where/how does a pending match suggestion (FR-9) actually surface to a human or agent
  for confirmation — e.g. a message in a channel, a dedicated review surface, an MCP tool response?
  Affects the user experience of fusion, so worth a stakeholder decision once the architect has
  options to weigh in with; not yet settled.
- **OQ-3** — Where/how does a rejected-but-reconsiderable match get re-evaluated in practice — does
  it need new corroborating content to resurface, or can a human/agent force a re-check on demand?
  Left to design; FR-10/AC-7 only fix that rejection isn't permanent.

## Decision log
2026-08-22 — Scope of source formats → **text-based formats broadly** (plain text, Markdown,
Mermaid, CSV, JSON, etc. — not a closed list) plus agent-generated text, for v1. Excludes only
formats needing real parsing/extraction machinery (PDF, images, Office docs).
2026-08-22 — Purpose of ingested knowledge → **both** grounding the AI agent's chat-channel
answers (same retrieval path as messages today) **and** a standalone knowledge base queryable
independent of chat.
2026-08-22 — "Input from agents" → confirmed: AI-agent-generated text is a valid ingestion source,
treated the same as a human-supplied document.
2026-08-22 — Interface → the ingestion (write) capability will be offered via the **existing MCP
front door**, so a connected agent can use it as **persistent memory** (write, not just read).
Recorded as a stated interface preference/need — exact MCP tool shape is the architect's call.
2026-08-22 — Conflicting facts → **keep both**, with provenance, and let the reader (human or
agent) weigh them — no silent overwrite, no automatic winner.
2026-08-22 — Same-entity matching → confidence-tiered: **very-high confidence auto-merges**;
anything less confident is **surfaced as a pending suggestion** requiring confirmation before
anything links/merges. Matching technique itself deferred to design (OQ-1).
2026-08-22 — Who can confirm a pending match → **either** a human user **or** a connected AI
agent — not human-only.
2026-08-22 — Rejection permanence → **not permanent** — a rejected match can be reconsidered/
re-linked later if warranted.
2026-08-22 — Ingestion volume → **bulk import supported from day one**, not just one document at
a time.
