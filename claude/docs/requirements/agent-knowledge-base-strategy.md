# Agent knowledge-base strategy — Feature Requirements
> **Status:** Ready for design — two sequenced tracks: raw kaizen-capture migration into a new "agent team" falkor-chat workspace (FR-8/FR-9, first), then distilled-knowledge-base ingestion into that same workspace (FR-2–FR-7, Option B) · **Owner:** `tico` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`) · **Last updated:** 2026-09-17

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
bases — through a graph-backed store, retrieved the way `falkor-chat` already retrieves chat
context (GraphRAG: ingestion → embeddings → hybrid vector + graph-traversal retrieval), rather than
an agent reading a named flat file whole. The driver isn't "flat files are inherently wrong" — it's
that flat files **don't stay findable as they grow**: an agent still has to read a whole file to
locate the relevant part, and `cobb` has to manually route/dedupe entries into it by hand. This
document captures the requirement for that graph-backed retrieval system, with K-030's four agents
as its first concrete consumer — not a document about K-030 alone.

**Which graph, and whether the flat files stay authoritative — now settled (2026-09-17, see the
resolution below and the decision log).** The stakeholder first named `kaizen_team` (the team's
existing raw-capture graph) as the substrate, then floated keeping the flat Markdown files as the
authoritative, git-versioned artifact `cobb` distills into (unchanged from today), with that
content additionally *ingested* into `falkor-chat`'s own GraphRAG corpus for searchability —
reached via `falkor-chat`'s own MCP server, not `cypher-mcp`/`kaizen_team`. That second framing —
Option B — is the one chosen: the flat files stay authoritative and get ingested into falkor-chat's
corpus for search.

**Resolved as a sequencing decision (2026-09-10, see decision log):** the stakeholder's underlying
reasoning is that `falkor-chat` is *meant* to be the one substrate for agent-and-human
interaction/knowledge generally — not one graph among several with overlapping GraphRAG
capability — so a gap in its document-ingestion pipeline (no update/delete, found by `architect`'s
plan below) is worth closing rather than routing around with a parallel system on `kaizen_team`.
This document's substrate stage (everything past Stage 0's interim K-030 relief) was therefore
**blocked** on a separate, successor falkor-chat feature,
`falkor-chat/docs/requirements/document-ingestion2.md`, adding real update/delete to document
ingestion.

**Block lifted, substrate decided (2026-09-17, see decision log):** `document-ingestion2` shipped
and archived (`falkor-chat/docs/plans/document-ingestion2-coordination.md`, closed 2026-09-13) —
real update/delete/versioning now exists in falkor-chat's document store, live-verified end-to-end
(AC-1 through AC-8). The choice was re-run on the merits rather than auto-flipped: `architect`'s
original Option B rejection named three separate problems, not one — no update/delete (now
closed), a chunking-granularity mismatch (falkor-chat's generic paragraph/sentence splitter has no
claim-awareness), and a tenancy/side-effect mismatch (ingestion targets a chat-shaped
`ws:{workspaceId}` tenant graph and triggers automatic entity/relationship extraction by default).
`document-ingestion2` was scoped purely to update/delete/versioning and left the other two
untouched. Presented with that, the stakeholder chose **Option B (falkor-chat ingestion) anyway**
— the original principle (one substrate, not several) still governs — accepting or resolving the
two remaining costs as part of this feature's own design work, not as a reason to revert to Option
A. **This document's substrate choice is now settled as Option B**, not still open.

**Scope expanded (2026-09-17, see decision log): raw kaizen capture migrates first, ahead of
distilled knowledge.** Every prior section of this document treated `kaizen_team` (the standalone
graph holding every agent's raw, undistilled `:KaizenEntry` capture) as an untouched given —
FR-5 originally assumed its writes "continue to work as today," and migrating it was never in
scope. The stakeholder's reasoning for changing that: raw kaizen capture **is, at bottom, a
knowledge base too** — just an earlier, undistilled stage of the same thing the rest of this
document is about — so under the governing one-substrate principle (falkor-chat is *the* substrate
for agent-and-human knowledge, not one graph among several) there's no principled reason to leave
it on a separate graph while only the *distilled* knowledge moves. So the first concrete piece of
work this document now calls for is **moving kaizen_team's functionality into a falkor-chat
workspace** — not the distilled-knowledge-base retrieval work (FR-2 through FR-7), which comes
after. Clarified: `kaizen_team` is **not** decommissioned as part of this — it keeps running in
parallel (retirement timing is an open question, not decided here) — and the migration is
**team-wide**: every agent that writes raw capture today (not only K-030's four), moves to writing
through the new falkor-chat-workspace venue.

**One workspace for both tracks, not two (2026-09-17, see decision log).** The new workspace is a
**dedicated, new one** — not an existing chat/demo workspace — directionally named something like
**"agent team"** (the stakeholder's naming intent; the exact literal workspace-id string is
`architect`/`graph-dba`'s naming-convention call, following `falkor-chat`'s own `ws:{workspaceId}`
pattern). It starts with raw kaizen entries (FR-8) but is explicitly meant to **generalize beyond
that** — the stakeholder's own framing: "kick off with the Kaizen entries and generalize to all
kind of stuff." Confirmed directly: **the distilled-knowledge-base ingestion (FR-2) targets this
same workspace**, not a separate one — so this document's two tracks converge on one shared
destination, consistent with the whole one-substrate premise, rather than trading `kaizen_team` for
a different kind of graph proliferation.

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
- **What already exists, mechanically — two graphs, one of them now a migration target:**
  - The team's raw kaizen capture already lives in a shared FalkorDB graph, `kaizen_team`,
    reachable by every agent via the `mcp__cypher__query` tool (`cypher-mcp`). Today that graph
    holds only **raw, undistilled** `:KaizenEntry` nodes pending `cobb`'s review; there is no
    embedding/semantic-search capability over it, and no distilled "knowledge base" content lives
    there — only flat files do. Writes to it are already restricted to specific authorized shapes
    (a producer's own capture, and a small set of curator-only shapes for `cobb`). **This is now
    the first thing to move** (see above) — into a falkor-chat workspace, running in parallel with
    `kaizen_team` rather than replacing it outright.
  - `falkor-chat` already runs a working GraphRAG pipeline on its own graphs (message ingestion →
    out-of-band embedding via an `EmbeddingWorker`/LM Studio → in-graph vector index → hybrid
    vector+traversal retrieval, `falkor-chat/docs/DESIGN.md` §6/§8), exposed over its own MCP
    server (Streamable-HTTP, distinct from `cypher-mcp`, `falkor-chat/docs/SERVER.md`). The
    stakeholder's stated preference throughout is to **reuse that actual machinery**, not build an
    independent embedding/retrieval setup — resolved (2026-09-17, see above and the decision log)
    to mean ingesting the knowledge-base Markdown files as a document corpus *into `falkor-chat`'s
    own graph*, with the files themselves staying the authoritative, versioned source. This is a
    preference to carry forward to design, not a schema/mechanism this document specifies.

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
- As the stakeholder, I want the team's raw kaizen capture to land in the same substrate its
  distilled knowledge eventually will, so falkor-chat is genuinely *the* one substrate rather than
  the destination for only the "finished" half of the team's knowledge.
- As any agent producing a raw kaizen entry, I want to keep writing it without disruption while
  this migration happens, whether that write currently lands in `kaizen_team` or its new
  falkor-chat-workspace home.

## Functional requirements
- **FR-1.** K-030's four agents (`teco`, `architect`, `data-scientist`, `tdd-engineer`) must be
  relieved of always-loaded-prompt growth now, independent of the graph-backed system's timeline —
  an interim on-demand knowledge base, following the existing flat-file pattern, is acceptable and
  must not wait on the larger system.
- **FR-2.** Distilled agent knowledge (existing flat-file content, plus anything created to satisfy
  FR-1) must become **searchable from a graph-backed store** — decided (2026-09-17) to be
  `falkor-chat`'s own graph, reached by ingesting the knowledge-base content as a document corpus
  through `falkor-chat`'s own MCP server (Option B), **into the same dedicated "agent team"
  workspace FR-8 creates for raw kaizen capture** — in a form other than "only inside a flat
  file." **The flat file stays the authoritative, git-versioned artifact `cobb` distills into**,
  unchanged from today; ingestion is what makes its content searchable, not a replacement for it.
- **FR-3.** An agent must be able to query the graph-backed store for knowledge relevant to its
  current situation and receive back the relevant distilled entries, without reading an entire
  knowledge-base file end-to-end.
- **FR-4.** Retrieval must surface relevant entries even when the querying agent's situation is
  worded differently from the stored entry's text — semantic match, not only exact-string/grep
  match. This is the specific gap flat files cannot close as they grow, and the reason a graph/
  embedding-backed approach is being pursued instead of, say, a bigger or better-organized file.
- **FR-5.** `cobb`'s distillation workflow must be able to route curated/distilled knowledge into
  the graph-backed store so it becomes searchable — via the store ingesting the flat file `cobb`
  still edits (see FR-2, Option B) — while raw-capture writes keep working uninterrupted throughout
  (whether still via `kaizen_team` or via FR-8's falkor-chat-workspace successor, depending on
  where FR-8's migration has reached).
- **FR-6.** The five/six agents' existing flat-file knowledge bases must become searchable through
  the same graph-backed mechanism without content loss, so the team ends up with one retrieval
  mechanism for all agent knowledge, not two indefinitely-parallel ones.
- **FR-7.** Whatever interim solution satisfies FR-1 must not cost materially more migration effort
  later than the pre-existing five/six agents' knowledge bases cost under FR-6 — the point of doing
  it now is relief, not a second throwaway system.
- **FR-8 (added 2026-09-17).** The team's raw kaizen-capture writes — every agent's `:KaizenEntry`
  producer/curator writes, today landing in the standalone `kaizen_team` graph — must be able to
  land inside a **new, dedicated falkor-chat workspace** instead, team-wide (not only K-030's four
  agents). The workspace is directionally named something like **"agent team"** (naming intent, not
  a literal string this document mandates) and is meant to generalize beyond kaizen entries over
  time, not stay kaizen-only (see FR-2, which targets this same workspace). `kaizen_team` is
  **not** decommissioned as part of this FR — it keeps operating in parallel; retirement is a
  separate, undecided question (see Open questions).
- **FR-9 (added 2026-09-17).** FR-8's raw-capture migration is sequenced **before** FR-2 through
  FR-7's distilled-knowledge-base retrieval work — it is this effort's first concrete deliverable,
  not a parallel or later track.

## Out of scope
- The schema/content-model design for how distilled knowledge lives inside `falkor-chat`'s graph
  once ingested (Option B, decided 2026-09-17) — an architecture decision for `architect`/
  `graph-dba`, informed by Open question #1 above.
- The specific mechanism for routing ingestion through falkor-chat's existing MCP tools — captured
  above only as the decided direction, not specified here.
- Redesigning `cobb`'s distillation procedure step-by-step (the `agent-maintenance` skill) — follows
  now that the storage target is decided.
- `BACKLOG.md` content moving to the graph — explicitly a separate, related future item; this
  document stays scoped to agent on-demand knowledge bases.
- The actual authoring/content of K-030's four interim knowledge bases — ordinary distillation
  work, once FR-1 is greenlit, following the pattern the existing six already use.
- The structural shape raw kaizen capture takes once inside the new "agent team" workspace (a new
  node type mirroring today's `:KaizenEntry`, reuse of falkor-chat's existing message/document
  shapes, or something else), the workspace's exact literal id, and the write-access mechanism for
  every agent (FR-8) — architecture decisions, see Open questions.
- When or whether `kaizen_team` is ever retired once FR-8's parallel-run migration lands — not
  decided; FR-8 only requires that it keep running, not that it be decommissioned.

## Acceptance criteria
- **AC-1.** Given K-030's four agents have no on-demand knowledge base today, when the interim step
  (FR-1) is executed, then each of the four has a flat-file knowledge base created following the
  existing pattern, without waiting on the graph-backed system.
- **AC-2.** Given the graph-backed system exists, when an agent queries it for knowledge relevant
  to its situation using wording that differs from a stored entry's exact text, then the relevant
  entry is still returned.
- **AC-3.** Given the graph-backed system exists, when `cobb` distills a raw entry destined for an
  agent's knowledge base, then the result becomes retrievable by query through that system —
  whether by a direct write or because the flat file it was distilled into gets ingested — not only
  readable by opening a flat file.
- **AC-4.** Given the five/six agents' existing flat-file knowledge bases, when the graph-backed
  system is available, then each can be migrated onto it without content loss.
- **AC-5.** Given the interim knowledge bases created under AC-1, when the graph-backed system
  becomes available, then migrating those four requires no more rework than migrating the
  pre-existing five/six under AC-4.
- **AC-6 (added 2026-09-17).** Given every agent's raw kaizen capture writes to `kaizen_team` today,
  when FR-8's migration lands, then every agent (not only K-030's four) can write and read its raw
  capture through a falkor-chat-workspace destination, with `kaizen_team` still operational in
  parallel, not decommissioned.
- **AC-7 (added 2026-09-17).** Given this document's two tracks (raw-capture migration, FR-8/FR-9;
  distilled-knowledge retrieval, FR-2 through FR-7), when work is sequenced, then AC-6 is satisfied
  before any distilled-knowledge-retrieval capability (AC-2/AC-3) is built.

## Open questions
1. What is the actual ingestion-authorization/trigger shape for a non-chat document corpus going
   through `falkor-chat`'s ingestion MCP tools (Option B, decided 2026-09-17)? `document-ingestion2`
   added update/delete/versioning to that pipeline but didn't design for a corpus with this
   feature's shape (routinely re-edited, one-claim-per-node granularity desired, no natural chat
   workspace) — a design decision for `architect`, informed by the chunking-granularity and
   tenancy/extraction-noise costs named in the Intent section above, which the stakeholder chose to
   accept or resolve within this feature's own design rather than treat as disqualifying.
2. Timeline/sequencing for actually starting the graph-backed system's build, beyond "a real
   near-term plan" — not yet specified.
3. **(added 2026-09-17) What does "raw kaizen capture inside a falkor-chat workspace" actually
   look like structurally?** A new node type mirroring today's `:KaizenEntry`/`PRODUCED`/`MENTIONS`
   shape, reuse of falkor-chat's existing message or document shapes, or something else — an
   architecture decision for `architect`/`graph-dba`.
4. **(added 2026-09-17) What is the write-access mechanism for every agent** once raw capture also
   targets a falkor-chat workspace — does every agent's write path change from `cypher-mcp`'s
   authorized shapes to falkor-chat's own MCP tools, or does something else preserve today's access
   pattern? This touches every agent's session wiring team-wide, not just this feature's four —
   flagged for `architect` given the breadth.
5. **(added 2026-09-17) When or whether `kaizen_team` is eventually retired** once FR-8 lands and
   both run in parallel — not decided; no trigger or timeline specified yet.

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
- 2026-09-10 — After handoff, stakeholder floated a refinement: keep the flat Markdown knowledge-base files as the authoritative, versioned artifact `cobb` distills into (unchanged from today), and use `falkor-chat`'s existing ingestion mechanism to make that content searchable via `falkor-chat`'s own MCP server, rather than storing distilled knowledge directly in `kaizen_team`. Asked whether this replaces or sits alongside the earlier "`kaizen_team` is the substrate" answer → stakeholder not sure yet, wants both weighed. Recorded as new Open question #1 (substrate choice); FR-2/FR-3/FR-5/FR-6 and AC-2/AC-3 generalized to not presuppose `kaizen_team` specifically. `architect` (already dispatched before this arrived) notified of the update via follow-up message.
- 2026-09-10 — `architect`'s plan (`claude/docs/plans/agent-knowledge-base-strategy.md`) resolved Open question #1 with CPG-backed evidence: falkor-chat's document ingestion (`ingest_document`/`create_document`) has no update/delete/list capability and is pinned non-idempotent by its own test — a correctness defect against a corpus that gets re-edited constantly (knowledge-base entries), since re-ingesting an edited entry would leave stale, superseded content permanently searchable alongside the new version with no way to retract it. Recommended `kaizen_team` (Option A): reuse falkor-chat's model/endpoint/query-style decisions, not its running document-store code. Flagged explicitly for stakeholder sanity-check, since it reads "reuse the actual machinery" more narrowly (pattern-level) than the most literal reading (falkor-chat's actual running server).
- 2026-09-10 — Asked whether the `kaizen_team` recommendation matches stakeholder intent → **no, wants to reconsider.** Asked why literal reuse specifically matters → **falkor-chat's document store gaining real update/delete has independent value beyond this feature** (not just an interim workaround) — reframes the question from "accept kaizen_team's gap-avoidance" to "should falkor-chat's document ingestion gain update/delete as its own feature, and should this effort depend on it."
- 2026-09-10 — Sequencing decision → **block this feature's substrate work (everything past Stage 0) on a separate falkor-chat feature adding real update/delete to document ingestion.** K-030's four agents still get interim flat-file relief per Stage 0, unaffected either way. A new, separate requirements interview opens for the falkor-chat side (`falkor-chat/docs/requirements/document-ingestion2.md` — successor to the archived `document-ingestion.md`, same topic family); once that feature is specified/built, this document's substrate choice (Option A vs. Option B) is revisited, not assumed to flip automatically to Option B.
- 2026-09-10 — Status changed from "Ready for design" to "Ready for design — substrate stage blocked" pending the new falkor-chat requirements doc. `architect` notified so its plan can reflect the block.
- 2026-09-10 — Stakeholder's underlying rationale for the block, stated directly: this **is** falkor-chat's intended purpose — being *the* substrate for agent-and-human interaction/knowledge generally, not one graph among several with overlapping GraphRAG capability. Building a second, parallel semantic-retrieval system on `kaizen_team` runs against that, even though `kaizen_team` is technically capable and CPG-evidence-backed today. This is why the gap in falkor-chat's document store is worth closing rather than routing around.
- 2026-09-17 — `teco` relayed that `document-ingestion2` shipped and archived (`falkor-chat/docs/plans/document-ingestion2-coordination.md`, closed 2026-09-13, commit `8906878`), satisfying the 2026-09-10 block condition, and asked for the Option A/Option B choice to be re-run on the merits — not auto-flipped to Option B just because the capability now exists (per this document's own prior instruction). Verified independently: read the coordination ledger (all 8 stages + QA acceptance `accepted`, AC-1..AC-8 live-verified) and `document-ingestion2`'s own requirements doc before treating the claim as settled.
- 2026-09-17 — Re-examined `architect`'s original Option B rejection (`claude/docs/plans/agent-knowledge-base-strategy.md` §1): it named three separate problems with routing this feature through falkor-chat's ingestion pipeline, not one — (1) no update/delete, (2) a chunking-granularity mismatch (generic paragraph/sentence splitter, no claim-awareness), (3) a tenancy/side-effect mismatch (chat-shaped `ws:{workspaceId}` tenant graph, automatic entity/relationship extraction by default). Checked `document-ingestion2`'s requirements doc and plan directly: its FR-1 through FR-8 are entirely about update/delete/versioning/audit — it never touched chunking behavior or the tenancy/extraction model. So only problem 1 is resolved; problems 2 and 3 stand exactly as `architect` found them. Presented this to the stakeholder before asking for a decision, so the choice wasn't made on the mistaken premise that the whole objection was closed.
- 2026-09-17 — Given that framing, which substrate? → **Option B (falkor-chat ingestion)**, chosen anyway. The original principle (falkor-chat as the one substrate, not several) still governs; the stakeholder accepts or resolves the remaining chunking-granularity and tenancy/extraction-noise costs as part of this feature's own design work, not as grounds to fall back to Option A. Status flipped from "blocked" to "Ready for design — substrate resolved to Option B"; FR-2/FR-5, the Intent section, and Open questions #1/#2 (substrate, file-counterpart) rewritten to reflect the settled choice — the file-counterpart question resolves to "yes" as Option B's own side effect (flat file stays authoritative, ingestion is what makes it searchable), consistent with what this document already named as the expected outcome if Option B were chosen.
- 2026-09-17 — Before reporting back to `teco`, stakeholder reopened scope: wants to **start by moving `kaizen_team`'s functionality (raw kaizen capture) into a falkor-chat workspace, as this effort's first use case** — ahead of the distilled-knowledge-base retrieval work the document was otherwise ready to hand off on. Status reverted from "Ready for design" to "Interviewing" pending a fresh readback of the expanded document.
- 2026-09-17 — Does this retire `kaizen_team`, or run both in parallel? → **run both in parallel for now**; not decommissioned as part of this work. Retirement timing left open (new Open question).
- 2026-09-17 — Is the raw-capture migration scoped to every agent, or narrower? → **every agent, team-wide** — not limited to K-030's four.
- 2026-09-17 — Why fold raw capture into this document/effort at all, rather than treat it as separate? → **"because it is what it is, a knowledge base"** — raw kaizen capture is, at bottom, an earlier/undistilled stage of the same knowledge this document is already about, so under the governing one-substrate principle it belongs in falkor-chat too, with no principled reason to carve it out. Recorded as new FR-8/FR-9 (migrate, sequenced first), AC-6/AC-7, new user stories, and four new Open questions (structural shape, workspace choice, write-access mechanism, retirement timing) — this document's original distilled-knowledge scope (FR-1 through FR-7, AC-1 through AC-5) is unchanged and still stands as Option B/settled; only the sequencing and the document's overall scope grew.
- 2026-09-17 — Which workspace hosts the raw-capture migration? → **a new, dedicated one**, directionally named something like **"agent team"** (naming intent, exact literal id left to `architect`/`graph-dba`'s naming convention). Stated explicitly not kaizen-only forever: "we will kick off with the Kaizen entries and generalize to all kind of stuff" — recorded in FR-8 and the Intent section. Resolves former Open question #4 (which workspace); removed from Open questions, list renumbered.
- 2026-09-17 — Does the distilled-knowledge-base ingestion (FR-2, Track 2) target this same new workspace, or a different one? → **same workspace, confirmed directly** ("yes, exactly, same workspace for both"). FR-2 rewritten to name the shared destination explicitly — this document's two tracks (raw capture, distillation) now converge on one shared workspace rather than each getting its own.
- 2026-09-17 — Readback confirmed ("perfect, sir, please close and commit"): the two-track structure (FR-8/FR-9 raw-capture migration into the new "agent team" workspace, sequenced first; FR-2–FR-7 distilled-knowledge ingestion into that same workspace, second), the parallel-run/no-decommission posture for `kaizen_team`, the team-wide breadth, and the five remaining Open questions (all architecture-only, none stakeholder-level) are all correct as drafted. Status flipped back to **Ready for design**. Next step: report back to `teco` so the coordination can resume — `architect`'s existing plan needs a real revision, not a resume from §2, since it predates both the Option B choice and the whole raw-capture-migration track.
