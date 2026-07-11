# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-10

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Evaluate Claude Code *agent teams* / *background agents* as a better substrate for teco than nested subagent calls. |

> **K-001 — validate nested delegation end-to-end — ✅ done 2026-07-09** (moved to history.md).
> Live run: falkor-chat M3 slice 1 through teco → architect → graph-dba → tdd-engineer, all
> checklist items passed. Launch brief + observation checklist preserved at
> [`k001-run-brief.md`](./k001-run-brief.md).

### K-002 — Consider agent teams / background agents
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Claude Code now has first-class *agent teams* (sessions that communicate) and *background agents* — possibly a cleaner orchestration substrate than a subagent calling subagents via the `Agent` tool, especially for parallel or long-running work.
- **Proposed change:** Read `code.claude.com/docs/en/agent-teams` and `/en/agent-view`; assess whether teco should be reframed as a team lead / use background tasks. Update `agent-standards` skill with findings.
- **Notes:** Surfaced during creation-time verification (2026-06-20). New primitives not yet in cobb's resident notes.

## Parking lot / ideas
- ~~A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.~~ *(✅ Resolved 2026-07-09: the roster is now an explicit routing table — task shape → owner → tie-breaker — with the coder-vs-tdd efficiency rule on both implementer rows, plus a separate handoff-contracts list. See history.md.)*
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
