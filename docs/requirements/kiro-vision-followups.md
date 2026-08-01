# Kiro Multi-Agent Ecosystem — Vision Follow-ups — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (—) · **Last updated:** 2026-08-01

## Intent
`kiro/DESIGN.md` sketched an early vision — before `falkor-chat` existed — for a team of Kiro
agents collaborating through a chat MCP server backed by FalkorDB. `falkor-chat/docs/DESIGN.md`
is now the canonical, built reference, and its actual design diverged from the vision in several
deliberate ways. This document exists to capture the vision's ideas that are **genuinely not yet
incorporated**, so they aren't lost, without re-litigating the ideas the real system already
resolved differently.

## Problem & current state
`kiro/DESIGN.md` (status: Draft) used a simpler schema than what was actually built — flat
`Agent`/`Message`/`Task`/`Artifact` nodes plus `meta/*.yaml` side files. Comparing it against
`falkor-chat/docs/DESIGN.md` + `docs/BACKLOG.md`:

**Already resolved differently — not gaps, recorded here so nobody re-proposes them:**
- Task lifecycle / "room state" → modeled as a general `WorkflowDef` (`kind:'process'`) over
  `Step`/`TRANSITION`/`StepRun`, not a flat `Task` node or a presence field —
  `falkor-chat/docs/DESIGN.md` §6.3, full ADR in `falkor-chat/docs/archive/plans/m1-chat-mcp.md`
  Appendix B.
- `get_room_state` / an agent "status" field → same ADR; deliberately not built as a parallel
  primitive.
- "How does a human participate in the chat?" → answered: a human is just a `User` member,
  already live.
- "What prevents two agents claiming the same task simultaneously?" → answered differently:
  `WorkflowRun.status` uses CAS-guarded flips (`running↔waiting`) already.
- "Multiple chat rooms?" → answered: Channel/Thread already supports many rooms per workspace.

**Genuinely unbuilt — real candidates for future work:**
1. **Deliverable/artifact provenance.** No equivalent to the vision's `Artifact` node (linking a
   git commit/file to the agent and task/workflow-step that produced it). Today's `EMITTED` edge
   covers only an AI answer's retrieval context, not "what did this agent's work produce."
2. **Turn-taking/backoff among several simultaneously-responding AI agents.** An open question in
   the original vision, never answered. Today's one shipped responder (K-013) only guards against
   answering its own messages — it doesn't address two AI agents both firing on the same trigger.
3. **Real-time push for agent wake-up** (SSE/pub-sub instead of polling). Already tracked as
   `falkor-chat/docs/BACKLOG.md` K-018 (proposed, deferred) — pointed at here, not re-proposed.
4. **An actual Kiro agent connected to falkor-chat.** Nothing exists yet. The first slice of this
   gap is being captured immediately, in its own requirements doc (the minimal demo agent).

## User stories
_(draft, for future prioritization — not yet committed work)_
- As the repo owner, I want an agent's work (a commit, a file, a deliverable) to be traceable in
  the graph back to the agent and the task/workflow-step that produced it, so that I can query
  "what did this agent actually deliver" the way I can already query message provenance.
- As an operator running multiple AI agents in the same channel, I want only one to respond to a
  given trigger (or a defined resolution order), so that agents don't step on each other or flood
  a thread.

## Functional requirements
Not yet drafted. These are candidate future work, not committed requirements — to be developed in
a future interview once prioritized against the rest of the backlog.

## Out of scope
- The "already resolved differently" list above — re-litigating the §6.3 ADR (flat `Task` node,
  presence field, `get_room_state`) is explicitly not on the table.
- Real-time push mechanics — tracked at `falkor-chat/docs/BACKLOG.md` K-018; not re-specified here.
- The minimal Kiro demo agent itself — captured in its own requirements doc.

## Acceptance criteria
None yet — no committed scope.

## Open questions
1. Turn-taking/backoff among multiple simultaneously-responding AI agents — stakeholder wants to
   think about this later; not resolved.
2. Deliverable/artifact provenance — shape, priority, and whether it merits its own milestone is
   unscoped.
3. When should this document get its own interview pass to move to "Ready for design"?

## Decision log
2026-08-01 — Is the artifact-provenance gap real and worth tracking? → Yes.
2026-08-01 — Turn-taking/backoff open question — resolve now or defer? → Bring it into this
document as an open question; stakeholder will think about it later.
2026-08-01 — Real-time push — re-propose or point at K-018? → Point at K-018, don't re-propose.
2026-08-01 — Is "a Kiro agent wired to falkor-chat" the first slice of the Kiro-integration gap? →
Yes; captured separately as the immediate demo-agent requirements doc.
