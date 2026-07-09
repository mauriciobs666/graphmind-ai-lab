# Kaizen — Change History: tico

> Dated log of actual changes to the `tico` agent. Most recent first.

## 2026-07-09 — Redesigned as a first-order conversational agent
- **What:** tico now runs as the **main-session agent** (`claude --agent tico`) — verified 2026-07-09 against `code.claude.com/docs/en/sub-agents`: the main thread takes on the definition's prompt/tools/model, frontmatter hooks still fire in main-session mode, and `initialPrompt` auto-submits as the first user turn. Prompt rewritten around a **live interview**: one thread at a time, reflect-back confirmations, `AskUserQuestion` (added to `tools`) for option picks, the doc updated as the conversation progresses, and a readback + explicit stakeholder confirmation gating the "Ready for design" flip. The round-based protocol shrank to a degraded subagent fallback ("If you are invoked as a subagent anyway"). teco no longer delegates to tico — it consumes the doc by path and pauses to the user when requirements need capturing.
- **Why:** User ruling on the initial design: tico is not a subagent but a first-order agent meant to be conversational — the rounds protocol optimized for the wrong constraint.
- **Plan items:** K-002 ⚪ rejected (continuation machinery moot in first-order mode); K-001 retargeted to a live `claude --agent tico` session.

## 2026-07-09 — created
- **What:** initial version of `tico` — conversational product-owner subagent that interviews the user/stakeholder about a feature request and owns the feature requirements document (`<component>/docs/requirements/<slug>.md`). Round-based interview protocol (subagents can't `AskUserQuestion`, so each invocation folds answers into the doc and returns the next question batch as its deliverable; the doc is the durable state between rounds). Write/Edit scoped to the requirements doc, harness-enforced by `hooks/guard-requirements-doc-writes.sh` (same PreToolUse "ask"-escalation pattern as architect/teco/devops). Model `opus`; tools mirror the architect's investigation set.
- **Why:** the team had design (architect) through delivery (coder/tdd/qa) covered, but nothing upstream capturing WHAT/WHY before the architect decides HOW — vague feature requests went straight to design. Requested by the user 2026-07-09.
- **Plan items:** seeded K-001 (e2e spin), K-002 (SendMessage rounds), K-003 (FR-id traceability).
