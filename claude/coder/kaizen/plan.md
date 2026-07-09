# Kaizen — Improvement Plan: coder

> Forward-looking backlog for the `coder` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Validate the architect→coder handoff end-to-end and confirm the coder can execute an architect plan without re-investigating. |

> Done: K-001 (2026-07-09) — efficiency-based routing boundary with `tdd-engineer`; descriptions now use objective task-shape triggers (detailed plan → coder; bug fix / safety-net refactor / test work / clear-contract feature → tdd-engineer) and cross-reference symmetrically. See history.md.

### K-002 — End-to-end handoff validation
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The architect→coder contract is designed but unproven; the coder should be able to pick up an architect plan cold (isolated context) and build it.
- **Proposed change:** Run a real feature through architect→coder; capture what the plan was missing; feed back into both prompts.
- **Notes:** Paired with architect K-002. **Update 2026-07-08:** the handoff transport is now fixed — the coder receives the architect's plan as a document path (`<component>/docs/plans/<slug>.md`) and reads the file itself; the live validation run remains. **Update 2026-07-09:** the *contract* is now proven live — in the teco K-001 run (falkor-chat M3 slice 1), `tdd-engineer` executed an architect plan cold from its path with no re-investigation (architect K-002 ✅). What remains for this item is coder-specific only: one live run with the **coder** as the implementer half.

## Parking lot / ideas
- A "definition of done" checklist (suite green, behavior covered, no scope creep, honest run report) the coder self-checks before reporting completion.
- Consider whether the coder should delegate the test-writing step to `tdd-engineer` when strict TDD is required, rather than doing it itself.
