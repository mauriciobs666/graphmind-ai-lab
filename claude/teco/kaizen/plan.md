# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-08

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-06-20 | high | 🔵 | Validate nested delegation end-to-end (teco → architect → tdd-engineer) and confirm depth/quality holds in practice. |
| K-002 | 2026-06-20 | medium | 🔵 | Evaluate Claude Code *agent teams* / *background agents* as a better substrate for teco than nested subagent calls. |

### K-001 — Validate nested delegation end-to-end
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** The docs say subagents can use the `Agent` tool, but real nested orchestration (teco spawning architect, then handing its plan to tdd-engineer) is unproven here — depth limits, context-passing fidelity, and result quality need a live test.
- **Proposed change:** Run a real multi-step feature through teco; capture where briefs lost information or delegation failed; tighten the prompt.
- **Notes:** Pairs with architect K-002 and coder K-002 (handoff validation). **Update 2026-07-08:** the architect handoff is now path-based (plan doc at `docs/plans/<slug>.md`, path passed to the implementer — no more verbatim copy-through), which removes the main brief-fidelity risk this item was meant to catch; the live run should now focus on delegation depth and result quality.

### K-002 — Consider agent teams / background agents
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Claude Code now has first-class *agent teams* (sessions that communicate) and *background agents* — possibly a cleaner orchestration substrate than a subagent calling subagents via the `Agent` tool, especially for parallel or long-running work.
- **Proposed change:** Read `code.claude.com/docs/en/agent-teams` and `/en/agent-view`; assess whether teco should be reframed as a team lead / use background tasks. Update `agent-standards` skill with findings.
- **Notes:** Surfaced during creation-time verification (2026-06-20). New primitives not yet in cobb's resident notes.

## Parking lot / ideas
- A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
