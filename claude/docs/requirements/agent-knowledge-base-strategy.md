# Agent knowledge-base strategy — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`) · **Last updated:** 2026-09-10

## Intent
Four custom agent prompts (`teco.md`, `architect.md`, `data-scientist.md`, `tdd-engineer.md`)
have crossed a size threshold where every new learning gets absorbed directly into the
always-loaded prompt, because — unlike `analyst`/`graph-dba`/`qa-engineer`/`devops`/
`tdd-engineer`(partial)/`data-scientist`(partial) — they have no on-demand "knowledge base" file
to route rare-path learnings into instead. `teco.md` alone grew ~2,500 words across four recent
distillation passes. `claude/cobb/kaizen/plan.md`'s K-030 flags this as a policy decision reserved
for the stakeholder.

Beyond K-030 itself, the stakeholder has a **real near-term plan**: route agent on-demand
knowledge — both the four new cases and the five/six agents that already have flat-file knowledge
bases — through the team's existing `kaizen_team` FalkorDB graph, retrieved the way `falkor-chat`
already retrieves chat context (GraphRAG: ingestion → embeddings → hybrid vector + graph-traversal
retrieval), rather than an agent reading a named flat file whole. The driver isn't "flat files are
inherently wrong" — it's that flat files **don't stay findable as they grow**: an agent still has
to read a whole file to locate the relevant part, and `cobb` has to manually route/dedupe entries
into it by hand. This document captures the requirement for that graph-backed retrieval system,
with K-030's four agents as its first concrete consumer — not a document about K-030 alone.

## Problem & current state
- **Today's pattern (six agents already have it):** `analyst`, `graph-dba`, `qa-engineer`,
  `devops`, `data-scientist`, and `tdd-engineer` each carry one or two on-demand knowledge-base
  files (e.g. `claude/analyst/review-techniques.md`). The agent's always-loaded prompt carries a
  one-line pointer ("consult `<file>` when `<situation>` arises"); the agent reads that whole file
  when the trigger applies. `cobb`'s periodic distillation pass is what routes a raw learning into
  one of these files versus the prompt itself versus discarding it.
- **Four agents have no such file** (`teco`, `architect`, `data-scientist` for one class of
  learnings, `tdd-engineer` for one class) — so every rare-path learning about them has nowhere to
  go except the always-loaded prompt. `teco.md` is the clearest case: prior editorial compaction
  attempts (`claude/teco/kaizen/plan.md` K-016) found a hard floor around ~5,200–5,250 words with
  every rule intact — the file is ~60 distinct rules, not padded prose, so there's nothing further
  to cut without actually moving content out.
- **Even where a knowledge-base file already exists, it's a growing flat file with no
  retrieval beyond "read it whole" or grep.** As one grows, the agent still reads the entire file
  to find the applicable part, and `cobb` routes/dedupes new entries into it by hand — the same
  underlying problem (no way to find "just the relevant part") that the flat-file pattern was
  invented to solve for the *prompt*, recurring one layer down.
- **What already exists, mechanically:** the team's raw kaizen capture already lives in a shared
  FalkorDB graph, `kaizen_team`, reachable by every agent via the `mcp__cypher__query` tool
  (`cypher-mcp`). Today that graph holds only **raw, undistilled** `:KaizenEntry` nodes pending
  `cobb`'s review; there is no embedding/semantic-search capability over it, and no distilled
  "knowledge base" content lives there — only flat files do. Writes to it are already restricted to
  specific authorized shapes (a producer's own capture, and a small set of curator-only shapes for
  `cobb`).
- **What's proven elsewhere:** `falkor-chat` already runs a working GraphRAG pipeline on its own
  graphs (message ingestion → out-of-band embedding via an `EmbeddingWorker`/LM Studio → in-graph
  vector index → hybrid vector+traversal retrieval, `falkor-chat/docs/DESIGN.md` §6/§8). The
  stakeholder's stated preference is to **reuse that actual machinery** against `kaizen_team`,
  rather than building an independent embedding/retrieval setup for it — a preference to carry
  forward to design, not a decision this document makes.

## User stories
- As an agent whose situation matches a rare-path learning, I want to retrieve just the relevant
  distilled knowledge for that situation, so that I don't have to carry it in my always-loaded
  prompt or read an entire knowledge-base file to find it.
- As `cobb`, distilling raw kaizen entries, I want to write curated knowledge into a shared,
  queryable store, so that it's findable by relevance to any agent's situation rather than only by
  which file happens to hold it.
- As the stakeholder, I want K-030's four agents relieved of their prompt-bloat problem now,
  without that interim work being thrown away once the graph-backed system exists.
- As the stakeholder, I want the eventual system to cover every agent's knowledge base — not just
  the four new ones — so the team ends up with one consistent mechanism instead of two permanent,
  parallel ones.

## Functional requirements
- **FR-1.** K-030's four agents (`teco`, `architect`, `data-scientist`, `tdd-engineer`) must be
  relieved of always-loaded-prompt growth now, independent of the graph-backed system's timeline —
  an interim on-demand knowledge base, following the existing flat-file pattern, is acceptable and
  must not wait on the larger system.
- **FR-2.** Distilled agent knowledge (existing flat-file content, plus anything created to satisfy
  FR-1) must become storable in `kaizen_team` in a form other than "only inside a flat file."
- **FR-3.** An agent must be able to query `kaizen_team` for knowledge relevant to its current
  situation and receive back the relevant distilled entries, without reading an entire
  knowledge-base file end-to-end.
- **FR-4.** Retrieval must surface relevant entries even when the querying agent's situation is
  worded differently from the stored entry's text — semantic match, not only exact-string/grep
  match. This is the specific gap flat files cannot close as they grow, and the reason a graph/
  embedding-backed approach is being pursued instead of, say, a bigger or better-organized file.
- **FR-5.** `cobb`'s distillation workflow must be able to write curated/distilled knowledge into
  `kaizen_team` (not only append to a flat file), while raw-capture writes continue to work as
  today.
- **FR-6.** The five/six agents' existing flat-file knowledge bases must be migratable onto the
  graph-backed system without content loss, so the team ends up with one retrieval mechanism for
  all agent knowledge, not two indefinitely-parallel ones.
- **FR-7.** Whatever interim solution satisfies FR-1 must not cost materially more migration effort
  later than the pre-existing five/six agents' knowledge bases cost under FR-6 — the point of doing
  it now is relief, not a second throwaway system.

## Out of scope
- The actual graph schema / node-and-edge design for storing distilled knowledge in `kaizen_team`
  — an architecture decision for `architect`/`graph-dba`.
- The specific mechanism for pointing falkor-chat's embedding/retrieval machinery at `kaizen_team`
  — captured above only as a stated stakeholder preference, not specified here.
- Redesigning `cobb`'s distillation procedure step-by-step (the `agent-maintenance` skill) — follows
  once the storage target is decided.
- Whether distilled knowledge also needs a git-tracked Markdown counterpart — open question below,
  not decided here.
- `BACKLOG.md` content moving to the graph — explicitly a separate, related future item; this
  document stays scoped to agent on-demand knowledge bases.
- The actual authoring/content of K-030's four interim knowledge bases — ordinary distillation
  work, once FR-1 is greenlit, following the pattern the existing six already use.

## Acceptance criteria
- **AC-1.** Given K-030's four agents have no on-demand knowledge base today, when the interim step
  (FR-1) is executed, then each of the four has a flat-file knowledge base created following the
  existing pattern, without waiting on the graph-backed system.
- **AC-2.** Given the graph-backed system exists, when an agent queries `kaizen_team` for knowledge
  relevant to its situation using wording that differs from a stored entry's exact text, then the
  relevant entry is still returned.
- **AC-3.** Given the graph-backed system exists, when `cobb` distills a raw entry destined for an
  agent's knowledge base, then the result is stored in `kaizen_team` in a form retrievable by
  query — not only appended to a flat file.
- **AC-4.** Given the five/six agents' existing flat-file knowledge bases, when the graph-backed
  system is available, then each can be migrated onto it without content loss.
- **AC-5.** Given the interim knowledge bases created under AC-1, when the graph-backed system
  becomes available, then migrating those four requires no more rework than migrating the
  pre-existing five/six under AC-4.

## Open questions
1. Should distilled knowledge also exist as a git-tracked Markdown file (human-reviewable,
   diffable, consistent with every other doc kind in this repo), or can it live purely as graph
   data with no file counterpart? Stakeholder is not yet sure — flagged for design.
2. What is the actual write-authorization shape for a "distilled knowledge" write into
   `kaizen_team`? Today's `cypher-mcp` write authorization covers raw producer-capture and a small
   set of curator shapes (MENTIONS-write, edge-resolve, full-node clear) — a distilled-knowledge
   write is a new shape that doesn't exist yet.
3. Timeline/sequencing for actually starting the graph-backed system's build, beyond "a real
   near-term plan" — not yet specified.

## Decision log
- 2026-09-10 — Which K-030 (falkor-chat workflow item vs. cobb's agent-prompt backlog item) is this about? → cobb's agent-prompt compaction backlog.
- 2026-09-10 — Stakeholder wants a requirements interview opened on the K-030 decision, explicitly flagging that they may later want to route this through falkor-chat's ingestion/GraphRAG machinery, and wants the decision made carefully with that in mind.
- 2026-09-10 — Is the GraphRAG idea scoped to just K-030's four agents, or all agent/team knowledge? → all agent/team knowledge; K-030 is the first concrete consumer, not the whole scope.
- 2026-09-10 — How concrete is the GraphRAG plan? → a real near-term plan, not a hedge.
- 2026-09-10 — Is `kaizen_team` the intended substrate, or something separate? → `kaizen_team` is the substrate.
- 2026-09-10 — Should this document cover the future retrieval system itself, or stay narrowly about K-030 with the future direction as a constraint? → cover the future retrieval system itself; K-030 becomes its first acceptance scenario.
- 2026-09-10 — What does GraphRAG solve that today's flat knowledge-base files don't? → flat files don't stay findable as they grow — an agent still reads the whole file, and `cobb` routes/dedupes by hand.
- 2026-09-10 — Who/what decides when to query the graph — the agent, or something automatic? → the agent still decides when to look, same as today's prompt pointers, but runs a targeted search instead of opening a named file.
- 2026-09-10 — Do the five/six agents with existing flat-file KBs need to migrate onto the new system too? → yes, in scope now (though actual migration execution is downstream design/implementation work).
- 2026-09-10 — Should distilled knowledge still exist as a git-tracked file? → not sure; left as an open question.
- 2026-09-10 — Does "use falkor-chat's ingestion and GraphRAG" mean reusing its actual running machinery, or just the same technique as independent infrastructure? → reuse the actual machinery (stated preference, not a decision made here).
- 2026-09-10 — Does K-030 wait for the graph system, or proceed now with an interim flat-file KB and migrate later? → proceeds now; migrates later, with FR-7's no-extra-rework expectation.
- 2026-09-10 — Should this document also cover `BACKLOG.md`'s parallel "headed for the graph" direction? → no; stays scoped to agent knowledge bases only.
- 2026-09-10 — Readback confirmed: FR-1 through FR-7 are correct as drafted; both open questions (file counterpart, write-authorization shape) stay genuinely open for design. Status flipped to Ready for design; handing off to `architect`.
