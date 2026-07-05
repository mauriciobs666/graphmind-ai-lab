# Kaizen — Change History: teco

> Dated log of actual changes to the `teco` agent. Most recent first.

## 2026-07-05 — Added `Edit` (scoped to the coordination doc)
- **What:** Added `Edit` to teco's frontmatter tools (`Read, Grep, Glob, Bash, Agent, Write, Edit, WebFetch, WebSearch`). Updated the guardrail to `Write`/`Edit` = **coordination/work-breakdown document only** (Write to create, Edit to revise in place as steps complete) — still **never** source/tests/config. Also tightened "How you work" step 2 to mention editing the doc in place. Mirrored the wording in `claude/AGENTS.md`.
- **Why:** User asked to give teco the `Edit` tool. With `Write` only, teco could create a coordination doc but had to overwrite it wholesale to update it; `Edit` lets it surgically revise the doc across a long-running orchestration (mark steps done, append findings). Scoped deliberately to the coordination doc — parallels `architect`, which carries `Write`+`Edit` guardrailed to its plan doc — so teco's "coordinate, don't implement" identity is preserved.
- **Plan items:** none (out-of-band user request); relevant to K-003 (coordination-doc convention).

## 2026-06-20 — Created
- **What:** Created the `teco` subagent (`teco/teco.md`, `model: opus`). Technical coordinator / tech lead: decomposes a multi-step goal into a sequenced work breakdown and **delegates each unit to the right specialist** (architect, coder, tdd-engineer, graph-dba, cobb; Explore/Plan built-ins) via the `Agent` tool, then integrates and verifies. **Hybrid mode:** delegates execution itself by default but pauses and returns to the user at genuine decision points / blockers / ambiguity. Tools: `Read, Grep, Glob, Bash, Agent, Write, WebFetch, WebSearch` — **no `Edit`/`NotebookEdit`** (it coordinates, doesn't implement); `Write` is for the coordination doc only; `Bash` read-only by guardrail.
- **Why:** User asked for a third agent on top of the architect→coder pair — "teco the technical coordinator" — to orchestrate the specialist roster.
- **Plan items:** seeded K-001..K-003.

## Decisions & verification recorded at creation
- **Subagents CAN delegate to subagents — verified 2026-06-20** against `code.claude.com/docs/en/sub-agents`. The doc enumerates the tools withheld from subagents (`AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode`, `ScheduleWakeup`, `WaitForMcpServers`); the `Agent`/Task tool is **not** withheld, so an orchestrator subagent is viable. (Older lore said subagents couldn't spawn subagents — that constraint no longer holds per the live doc. Claude Code now also has first-class *agent teams* and *background agents*.)
- **Key limitation baked into the prompt:** `AskUserQuestion` is unavailable to subagents, so teco **cannot ask interactively** — the hybrid design has it *return* to the user with the decision instead of guessing. teco also doesn't see the parent conversation, and delegated agents don't see teco's or each other's context → the prompt mandates **self-contained briefs** (pass the architect's plan verbatim to the implementer, etc.).
- **No `name`-conflict / collection consistency:** dropped any "senior" framing to match the 2026-06-20 harmonized collection. Defaults implementation routing toward `tdd-engineer` given the user's documented TDD preference.
