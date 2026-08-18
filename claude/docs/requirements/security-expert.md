# Security expert (new agent) — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M?) · **Last updated:** 2026-08-17

## Intent
Introduce a dedicated security-expert team member to close two gaps the current roster doesn't
cover:

1. **Code/app security review**, deeper than the security/perf step already folded into
   `analyst`'s general code review.
2. **Agent/prompt-safety review** — judging whether an agent's own artifacts (prompts, skills,
   `kaizen/inbox.md` entries, plans) are safe to keep or promote. No agent currently owns this
   judgment: on 2026-07-31, `analyst` flagged a kaizen inbox entry as instruction-poisoning-shaped,
   and `teco` (no adjudication authority of its own) had to route the call to `cobb` ad hoc.

## Problem & current state
- `analyst` reviews code correctness → tests → convention fit → clarity → **security/perf**, in
  that priority order — security is one checklist item among several, not a deep pass.
- `devops` owns secrets hygiene and infra hardening, but not application-level vulnerability
  analysis or prompt/agent-safety judgment.
- Nobody owns agent/prompt-safety adjudication as a standing responsibility — the 2026-07-31
  incident (`claude/analyst/kaizen/history.md`, `claude/cobb/kaizen/history.md`) was handled
  one-off, outside any agent's stated remit.

## User stories
- As a stakeholder, I want a dedicated deep-dive on code-level security issues, so that
  vulnerability-shaped problems don't ride along as a single checklist line in a general review.
- As a stakeholder, I want a standing owner for judging whether an agent's own artifacts
  (prompts, skills, kaizen entries) are safe, so incidents like 2026-07-31 have a clear home
  instead of ad hoc routing.

## Functional requirements
_(to be drafted as the interview proceeds)_

## Out of scope
_(to be drafted as the interview proceeds)_

## Acceptance criteria
_(to be drafted as the interview proceeds)_

## Open questions
- What specifically triggered this request now — was there a recent moment that exposed the gap?
- Where's the boundary with `analyst`'s existing security/perf check — does this agent replace
  that step, or does `analyst` stay the first pass and this agent goes deeper on request?
- Which components are in scope for the code-security side — all of them, or the ones handling
  untrusted/external input specifically (e.g. `cpg`'s Cypher execution surface, `falkor-chat`'s
  chat/workflow inputs, `salesperson`'s chatbot inputs)?
- For the agent/prompt-safety side, what's the actual surface reviewed — kaizen inbox entries
  before promotion, agent/skill prompts themselves, plans, all of the above?
- When does this agent get invoked — proactively as a standing review gate (like `analyst` today),
  on demand for a dedicated audit, or both?

## Decision log
2026-08-17 — What's the intent? → Introduce a security-expert agent covering both (1) deeper
code/app security review than `analyst`'s current security/perf checklist step, and (2)
agent/prompt-safety review. For the code-security side, the stakeholder wants the agent able to
use the project's existing Code Property Graph (the `cpg` MCP tool / `cpg-analysis` skill pattern
other agents already follow) when one exists for the component under review — noted here as a
stated preference for reusing an existing project capability, not a new design decision.
2026-08-17 — What triggered this now? → No specific incident; proactive risk reduction ("ensure
we won't have any problems") rather than a reaction to a known close call. This is about standing
coverage, not patching one discovered hole.
2026-08-17 — Does the security expert replace or layer on top of `analyst`'s existing
security/perf checklist step? → **Layer on top.** `analyst` keeps its lightweight security/perf
check on every review as the first line of defense; the security expert is a separate, deeper
pass invoked when security is the actual concern.
