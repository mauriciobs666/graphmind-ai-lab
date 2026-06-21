# Kaizen — Change History: architect

> Dated log of actual changes to the `architect` agent. Most recent first.

## 2026-06-20 — Dropped "senior" framing
- **What:** Removed "senior" from the `description` ("Senior software architect" → "Software architect") and the body opener ("You are a senior software architect" → "You are a software architect"). Mirrored in the catalog entries (`claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md`).
- **Why:** User flagged the overconfidence concern with seniority framing. Evidence (persona-prompting studies, e.g. Zheng et al. 2024) shows role labels are weak-to-neutral for correctness and authority framing can dent calibration; behavior is driven by the concrete process + guardrails, not the title. Chose the most conservative option (drop the word entirely) over keeping it. Note: this goes one step *further* than the 2026-06-05 collection precedent, which dropped boasts but **kept** "Senior" as an altitude signal — so architect/coder are now inconsistent with cobb/tdd-engineer/graph-dba/dra-claudia until those are harmonized (flagged to user).
- **Plan items:** —

## 2026-06-20 — Created
- **What:** Created the `architect` subagent (`architect/architect.md`, `model: opus`). Read-only design/planning agent: investigates the codebase, weighs trade-offs, and produces a step-by-step implementation plan/spec (goal & scope, context & findings, design & rationale, ordered steps, test strategy, risks & open questions). Tools restricted to `Read, Grep, Glob, Bash, Write, WebFetch, WebSearch, Agent`; **no `Edit`/`NotebookEdit`** and a hard guardrail that `Write` is for the plan document only — it never edits source/tests/config. Designed as the planning half of an **architect→coder handoff**: the plan stands alone so an isolated-context implementer can execute it.
- **Why:** User asked to create two complementary Claude Code subagents, "the architect" and "the coder," distinct from the existing `tdd-engineer` (strict TDD implementer) and the OpenCode `coding-senior`, with a sequential architect→coder handoff.
- **Plan items:** seeded K-001..K-003.

## Decisions recorded at creation
- **Why `Write` but no `Edit`:** the agent must be able to emit a *durable* plan document for the handoff (an isolated coder context won't see the architect's investigation otherwise), but must not surgically modify existing code. Omitting `Edit`/`NotebookEdit` + a prompt guardrail signals "planning only" while still allowing the deliverable. `Bash` is investigation-only by guardrail. Tool gating can't fully enforce "plan docs only" (Write can overwrite any path) — the guardrail carries that.
