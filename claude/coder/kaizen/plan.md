# Kaizen — Improvement Plan: coder

> Forward-looking backlog for the `coder` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-20

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-06-20 | high | 🔵 | Watch for routing overlap with `tdd-engineer`; sharpen descriptions if both auto-trigger on the same requests. |
| K-002 | 2026-06-20 | medium | 🔵 | Validate the architect→coder handoff end-to-end and confirm the coder can execute an architect plan without re-investigating. |

### K-001 — Routing overlap with `tdd-engineer`
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Both are "senior engineer who implements." If Claude Code auto-delegates ambiguously between them, the user gets unpredictable behavior. The descriptions try to disambiguate (coder = plan-driven/pragmatic; tdd-engineer = strict test-first) but real routing should be observed.
- **Proposed change:** Monitor invocations; if they collide, tighten trigger keywords or merge/retire one. Decide whether "plan-driven implementer" and "TDD implementer" warrant two agents or one with a mode.
- **Notes:** Surfaced at creation. The user has a documented TDD preference (see auto-memory) — TDD may be the default expectation, which affects this.

### K-002 — End-to-end handoff validation
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The architect→coder contract is designed but unproven; the coder should be able to pick up an architect plan cold (isolated context) and build it.
- **Proposed change:** Run a real feature through architect→coder; capture what the plan was missing; feed back into both prompts.
- **Notes:** Paired with architect K-002.

## Parking lot / ideas
- A "definition of done" checklist (suite green, behavior covered, no scope creep, honest run report) the coder self-checks before reporting completion.
- Consider whether the coder should delegate the test-writing step to `tdd-engineer` when strict TDD is required, rather than doing it itself.
