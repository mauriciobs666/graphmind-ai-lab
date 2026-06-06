# Kaizen — Improvement Plan: tdd-engineer

> Forward-looking backlog for the `tdd-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-05 (review #2)

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| _(none active)_ | | | | All tracked items resolved — see Done/Deferred below and parking lot. |

> Done: K-001, K-002, K-004, K-005 (2026-06-05) — see history.md.
> Deferred: K-003 (2026-06-05) — keep tools unconstrained for flexibility; see history.md.

### K-003 — Tool permissions decision  ⚪ DEFERRED (2026-06-05)
- **Status:** ⚪ deferred — user chose to keep `tools` unconstrained for now.
- **Decision:** No `tools` key; the agent continues to inherit all tools, preserving flexibility to spawn subagents and fetch docs mid-task. The focused-set restriction (`Read, Edit, Write, Bash, Grep, Glob`) was considered and declined.
- **Revisit if:** the agent's broad tool access causes surprise or unwanted actions in practice.

## Parking lot / ideas
- State explicitly that the agent does **not** auto-commit (the harness rule is "commit only when asked") — avoids surprise commits given the "commit-sized increments" language.
- Add a one-liner that coverage % is a guide, not a goal — pin behaviors, don't chase numbers.
- Note on flaky tests: quarantine + diagnose root cause rather than re-run until green.
- Consider whether `opus` is warranted vs. `sonnet` for cost — TDD benefits from careful reasoning, but routine cycles may not need it.
- Optional enrichment: a brief nod to advanced test techniques where they fit — table-driven/parameterized tests for boundary sweeps, property-based testing for invariants, and mutation testing as a *coverage-quality* check (does the suite actually catch injected faults?). Low priority; the prompt is deliberately lean, so only add if it earns its keep.
