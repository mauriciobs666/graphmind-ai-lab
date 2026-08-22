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
- **FR-6 (fusion, core)** — _(eliciting mechanics)_ When newly ingested content overlaps or
  conflicts with already-ingested knowledge, the system reconciles it rather than creating
  disconnected duplicates or silently inconsistent facts.

## Out of scope
- **Binary / non-text document formats** (PDF, images, Office docs, etc.) that require dedicated
  parsing/extraction machinery — v1 handles **text-based formats** (plain text, Markdown,
  Mermaid, CSV, JSON, and similar; not a fixed/closed list) (stakeholder decision, 2026-08-22).

## Related work (not part of this feature)
- `falkor-chat/docs/requirements/summary-nodes.md` (Status: Interviewing, unfinished) — condenses
  *existing* graph content into retrievable summary nodes. This feature is about bringing *new*
  knowledge in from outside the graph. Different problem; cross-referencing because both feed the
  same GraphRAG retrieval path.

## Open questions
- What is the ingested knowledge *for* — grounding the AI agent's answers in chat (same retrieval
  path as messages today), a separate queryable knowledge base, or both?
- What exactly is meant by "input from agents" as a source — agent-generated text (e.g. a
  summarizer or external tool's output) treated as an ingestible document, same as a
  human-provided text file?

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
