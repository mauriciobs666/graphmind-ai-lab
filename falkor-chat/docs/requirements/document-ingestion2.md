# Document ingestion — update & delete — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Extends:** `falkor-chat/docs/requirements/document-ingestion.md` · **Last updated:** 2026-09-10

## Intent
`falkor-chat`'s document-ingestion pipeline (`falkor-chat/docs/requirements/document-ingestion.md`,
shipped, archived) can only **create** a document — `services.create_document` mints a fresh
`documentId` on every call, pinned non-idempotent by its own test
(`test_create_document_is_non_idempotent_on_retry`), and there is no `update_document`,
`delete_document`, or `list_documents` anywhere in the codebase (confirmed via the `cpg_falkorchat`
Code Property Graph during a design pass for a different, related feature — see below). That's a
correctness gap for any source content that gets **edited and re-ingested over time**, rather than
ingested once and left alone: re-ingesting an edited version currently creates a second, independent
`Document` + `Chunk`s alongside the original, with no way to retract the superseded content — search
would keep surfacing stale, possibly-reversed material indefinitely, alongside the current version,
forever.

This surfaced concretely from a separate, `claude/`-scoped feature
(`claude/docs/requirements/agent-knowledge-base-strategy.md`) that wanted to ingest the Claude Code
agent team's knowledge-base Markdown files — files that get **routinely re-edited** — into
`falkor-chat`'s GraphRAG corpus for semantic search. That plan's evidence exposed this gap and
recommended routing around it (a parallel graph). The stakeholder's view, stated directly: **`falkor-chat`
is meant to be *the* substrate for agent-and-human knowledge interaction** — not one graph among
several with overlapping GraphRAG capability — so this gap is worth closing in `falkor-chat` itself,
rather than building a workaround elsewhere. `agent-knowledge-base-strategy.md`'s substrate choice
is blocked on this feature; once this ships, that choice gets revisited on the merits.

## Problem & current state
- **No update, delete, or list capability exists today.** `create_document` is the only write path;
  it mints a new `documentId` every call and is pinned non-idempotent by its own test. There is no
  `update_document`/`delete_document`/`list_documents` method anywhere in
  `falkorchat/{services,repository,mcp,api}.py` (confirmed by a direct CPG query,
  `MATCH (m:METHOD) WHERE toLower(m.NAME) CONTAINS 'document'`, 140 rows, none matching).
- **Consequence:** re-ingesting an edited document creates a second `Document`+`Chunk`s alongside
  the original. Search surfaces both indefinitely — including the stale, possibly-reversed one —
  with no mechanism to retract it.
- **This blocks any use case where the source content is a living document**, not a one-shot drop-in
  — exactly the shape of an agent-team knowledge-base file, which is what surfaced this gap.
- **Falkor-chat already has an established, non-destructive convention** that any update/delete
  design needs to stay consistent with: the original ingestion feature's FR-6 (conflicting facts
  from different sources are **kept side-by-side with provenance**, never one silently overwriting
  another) and FR-12 (the **full original document is always retained**, never discarded after
  processing, specifically so it can be re-inspected or re-processed later). A destructive
  overwrite-in-place would be new, inconsistent behavior for this system.

## User stories
- As a contributor (human or agent) who edits a previously-ingested document, I want the updated
  content to become what search surfaces, so stale content doesn't keep appearing indefinitely.
- As a contributor, I want the system to recognize when new content is really an edit of something
  already ingested, the same way it already recognizes likely-duplicate entities, so I don't have
  to track and pass an ID myself for the common case.
- As an operator, I want to actually and permanently remove a document I deliberately choose to
  delete (e.g. ingested by mistake, or content that shouldn't persist at all), distinct from
  ordinary supersession by an update.
- As a human or agent investigating provenance, I want to look up what a document used to say
  before it was superseded, consistent with the retention principle the original ingestion feature
  already established.
- As a connected AI agent, I want the same update/delete access I already have for ingestion via
  MCP, so document maintenance doesn't require a separate, more restricted path.

## Functional requirements
- **FR-1 (update).** The system can update an existing document's content. The new version becomes
  what default search surfaces; the superseded version is **retained, not destroyed**, for later
  inspection (see FR-5) — never silently overwritten in place, consistent with this system's
  existing non-destructive convention.
- **FR-2 (update detection, confidence-tiered).** The system can detect, at some confidence level,
  that newly submitted content is an edited version of an already-ingested document — following the
  **same confidence-tiered pattern** the original feature already uses for entity fusion (its
  FR-8/FR-9/FR-10): a **very-high-confidence** match auto-treats the submission as an update; a
  **lower-confidence** match surfaces as a **pending suggestion** requiring confirmation before the
  existing document is superseded. The detection technique itself is a design decision (see Open
  questions), mirroring the original feature's own OQ-1 posture for entity matching.
- **FR-3 (update confirmation).** A pending update suggestion (FR-2's lower-confidence case) can be
  confirmed or rejected by a human or a connected AI agent — mirrors the original feature's FR-10
  for entity-match confirmation. Confirming supersedes the existing document; rejecting leaves both
  as independent documents.
- **FR-4 (delete).** The system supports **true deletion** of a document — actual removal, not
  supersession. Deletion is triggered **only by an explicit, deliberate action naming the specific
  document** — never automatic or detected, unlike FR-2's update path.
- **FR-5 (version history).** A document's prior (superseded) version(s) remain inspectable/
  retrievable on request, even though only the current version is what default search surfaces —
  consistent with the original feature's "always retain the full original" principle (FR-12),
  extended across versions rather than just the single original.
- **FR-6 (access parity).** Update and delete are reachable through the **same access surface** as
  ingestion itself (human or a connected AI agent, via the existing MCP front door) — no separate,
  more restricted access tier for these operations.
- **FR-7 (entity/relationship consequence — deferred).** What happens to entities/relationships
  that were extracted from a document and fused into the graph, when that document is later
  updated or deleted, is **explicitly not decided here** — the stakeholder deferred this to
  whoever designs it, since it needs to be weighed against fusion's actual mechanics (the original
  feature's FR-6–FR-10). See Open questions.
- **FR-8 (delete audit trail).** A deletion (FR-4) leaves a record that it happened — who deleted
  which document and when — even though the document's own content is gone. Deletion is not a
  silent, untraceable removal, consistent with this system's general provenance-first posture
  (FR-6/FR-12 of the original feature).

## Out of scope
- The actual technique for detecting "this is likely an edit of an existing document" (embedding
  similarity, title/metadata matching, LLM-based comparison, or some layered combination) — a
  design decision, same posture as the original feature's OQ-1 for entity-match technique.
- Where/how a pending update-suggestion (FR-2/FR-3) actually surfaces for confirmation — same
  posture as the original feature's OQ-2.
- Whether/how entity-level cascade on update/delete is implemented (FR-7) — explicitly deferred to
  design, see Open questions.
- Any change to falkor-chat's existing entity-fusion matching logic itself, beyond what's needed to
  support document-level update/delete.
- `agent-knowledge-base-strategy.md`'s own substrate decision — that document is a downstream
  consumer of this capability, revisited once this ships; not decided here.

## Acceptance criteria
- **AC-1.** Given new content is submitted that matches an existing document at very-high
  confidence, when ingestion completes, then the existing document is superseded automatically —
  no confirmation required — and default search surfaces only the new version.
- **AC-2.** Given new content is submitted that plausibly, but not very confidently, matches an
  existing document, when ingestion completes, then the match appears as a pending suggestion and
  the existing document is **not** superseded until confirmed.
- **AC-3.** Given a pending update suggestion, when a human or a connected agent confirms it, then
  the existing document is superseded by the new version; when rejected, the existing document is
  unaffected and the new content stands as an independent document.
- **AC-4.** Given a document has been superseded by an update, when default search runs, then only
  the current version's content is returned — never the superseded version's.
- **AC-5.** Given a document has been superseded, when someone looks up its history, then the prior
  version's full content is still retrievable.
- **AC-6.** Given a document is explicitly deleted by its documentId, when deletion completes, then
  that document's content is no longer retrievable by any means — distinct from AC-5's
  "superseded but still inspectable" case.
- **AC-7.** Given update/delete access, when exercised by a human versus by a connected AI agent via
  MCP, then both succeed identically — no access-tier difference between them.
- **AC-8.** Given a document is deleted, when someone looks up whether/when/by whom it was deleted,
  then that record is retrievable, even though the document's own content is not.

## Open questions
1. **Entity/relationship cascade on update/delete** — when a document is superseded or deleted,
   should entities/relationships extracted solely from it be retracted from current knowledge, or
   should fused entities always survive independent of their originating document(s)? Explicitly
   deferred by the stakeholder to whoever designs this, since it needs to be weighed against the
   original feature's actual fusion mechanics (FR-6–FR-10) rather than decided in the abstract.
2. **Update-detection technique** (embedding similarity, metadata/title matching, LLM comparison,
   or a layered combination) — design decision, same posture as the original feature's OQ-1.
3. **Where a pending update suggestion surfaces for confirmation** (a channel message, a dedicated
   review surface, an MCP tool response) — design decision, same posture as the original feature's
   OQ-2.

## Decision log
- 2026-09-10 — Opened as a successor to the archived `document-ingestion.md`, surfaced by `agent-knowledge-base-strategy.md`'s substrate investigation. Stakeholder wants falkor-chat's document store to gain real update/delete as a feature with value of its own, not only as a workaround for the agent-knowledge-base use case.
- 2026-09-10 — What happens to old content on update? → Explored versioning vs. hard overwrite; grounded in falkor-chat's existing "never silently overwrite, keep provenance" convention (FR-6/FR-12 of the original feature) → **versioned**: old content retained but excluded from default search, not destroyed.
- 2026-09-10 — Is true hard-delete needed at all, separate from update/versioning? → **Yes** — for a mistaken ingestion or content that must be genuinely removable, distinct from ordinary supersession.
- 2026-09-10 — How does the caller indicate "this is an edit of X" vs. "this is new"? → **System detects it automatically**, not explicit-ID addressing.
- 2026-09-10 — Should detection follow the same confidence-tiered pattern already used for entity matching (auto vs. pending-suggestion)? → **Yes**, same tiered pattern.
- 2026-09-10 — How is true deletion triggered? → **Explicit action only**, by documentId — never automatic/detected.
- 2026-09-10 — What happens to fused entities/relationships derived from an updated/deleted document? → **Deferred to architect** — stakeholder not sure, wants the trade-off weighed against actual fusion mechanics.
- 2026-09-10 — Should superseded versions stay inspectable? → **Yes**, history should be retrievable, consistent with the original feature's retention principle.
- 2026-09-10 — Should update/delete access match ingestion's access (human + connected agent via MCP)? → **Yes**, same access, no separate tier.
- 2026-09-10 — Should a deletion leave an audit trail even though content is gone? → **Yes** — recorded as FR-8/AC-8, no longer an open question.
- 2026-09-10 — Readback confirmed: FR-1 through FR-8, the acceptance criteria, and the three remaining deferred-to-design open questions (cascade behavior, detection technique, suggestion surface) are correct as drafted. Status flipped to Ready for design.
