# Kaizen — Improvement Plan: tico

> Forward-looking backlog for the `tico` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | high | 🔵 | Live e2e spin: a real `claude --agent tico` interview on a genuine feature request |
| K-002 | 2026-07-09 | — | ⚪ | ~~SendMessage continuation for interview rounds~~ — moot in first-order mode |
| K-003 | 2026-07-09 | low | 🔵 | Requirements→plan traceability (architect plan cites FR-ids) |

### K-001 — Live e2e spin (first-order)
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** the live-interview design is unexercised; same validate-by-running discipline as teco's K-001.
- **Proposed change:** launch `claude --agent tico` on a genuinely vague feature request (e.g. in `falkor-chat/`) and run the interview to "Ready for design". Verify: `initialPrompt` kicks off correctly, the doc is updated *during* the conversation (not batched), the guard hook passes conforming writes silently, one-thread-at-a-time pacing holds, and the readback/explicit-confirmation gate fires before the status flip.
- **Notes:** also worth one delegated invocation to see the subagent fallback degrade gracefully.

### K-002 — SendMessage continuation for rounds
- **Status:** ⚪ rejected 2026-07-09
- **Rationale (original):** make re-invoked interview rounds cheaper by continuing the spawned agent.
- **Why rejected:** the first-order redesign removed the rounds protocol from the primary path — as the main-session agent tico converses natively. The subagent fallback keeps doc-as-state and needs no continuation machinery.

### K-003 — Requirements→plan traceability
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** FR-ids exist in tico's template but nothing downstream references them; an architect plan that cites FR-ids makes coverage checkable.
- **Proposed change:** once a real tico→architect handoff has run, consider asking the architect (prompt or convention) to map plan steps / test strategy to FR-ids.
- **Notes:** don't build until a real handoff shows the need.

## Parking lot / ideas
- A `docs/requirements/` template file vs the inline template (only if the inline one drifts across features).
- Non-functional requirements section (performance, security) — add when a feature actually needs one rather than padding every doc.
- Project-scoped `agent` setting to make tico the default session agent in a requirements-heavy phase — only if launching via `--agent` proves to be friction.
