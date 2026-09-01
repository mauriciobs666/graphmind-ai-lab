# DevOps: headless, local-model OpenCode variant

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** —

**Note:** drafted by Claude from a prior conversation with the stakeholder, ahead of an actual
`tico` interview — capture only, not yet reviewed. `tico` should treat this as a starting point
for its own interview, not a finished requirement.

## What

A second way to run the `devops` role (`claude/devops/devops.md`), alongside its existing
interactive/delegated Claude Code form:

- **Headless / schedulable** — runnable without a live Claude Code session or a human present
  (e.g. on a cron-style schedule), via OpenCode's non-interactive `opencode run --agent devops`
  invocation (confirmed working in this repo).
- **Routed to a cheaper/local model** — via OpenCode's LM Studio provider support, rather than
  every run going through a metered cloud model.

Both are wanted together, not either/or — confirmed directly with the stakeholder.

## Why

The current `devops` agent only runs interactively or as a delegated subagent inside a live
Claude Code session — there's no way to get a routine devops check (health/hygiene, disk usage,
container status, etc.) done on a schedule without a human opening a session. Routing that kind
of routine, lower-stakes work through a local/cheaper model also avoids spending cloud-model
budget on checks that don't need frontier reasoning.

## Constraints already settled with the stakeholder (this conversation)

These came out of direct back-and-forth, not inference — carry them into the actual interview
rather than re-deriving them:

- **One canonical prompt/persona, not two that drift.** The stakeholder chose "shared prompt, two
  thin wrappers" over either a full move (dropping the Claude Code side) or two independently
  maintained copies.
- **Headless destructive-ops posture: auto-deny, report only.** The existing `devops` agent
  approval-gates destructive/shared-state operations (via a `PreToolUse` hook,
  `claude/scripts/guard-destructive-ops.sh` — volume wipes, `docker system prune`, `docker rm
  -f`, `compose down -v`, Redis/FalkorDB flush/delete) by asking a human. A headless/cron run has
  no human to ask, so the stakeholder chose: refuse the action automatically and report what
  would have been done, rather than attempting any form of unattended approval.

## Background worth carrying into the interview (not decisions, just findings)

- OpenCode's own non-interactive-run permission handling is not officially documented for the
  "nobody can answer an `ask` gate" case; corroborating evidence (a GitHub issue, a third-party
  write-up) points at hanging/silent blocking rather than a safe default. This is *why* the
  auto-deny constraint above is a real requirement, not just caution.
- The existing kaizen learning-capture step (writing to the shared `kaizen_team` FalkorDB graph)
  cannot carry over to an OpenCode-run variant as-is — OpenCode has no MCP wiring in this repo at
  all yet (root `AGENTS.md` tracks this as backlog **C-310**). Whether that's an acceptable gap
  for a first version, or a blocker, is a real open question (see below).
- No scheduler (cron, systemd timer, or similar) exists anywhere in this repo today — this would
  be new automation infrastructure, not reuse of an existing one. `mcp-monitor` exists but is an
  event/content-triggered watcher, not a wall-clock scheduler, and there's no existing
  devops-relevant event source to hook it to instead.
- A rough technical design (exact OpenCode agent frontmatter, a permission-glob translation of
  the existing destructive-ops guard patterns, a wrapper script for headless invocation, doc
  update list) was sketched informally in the conversation that produced this draft. It is
  **not** included here since it's HOW, not WHAT/WHY — once this requirement reaches "Ready for
  design," that sketch is available as a head start for whoever (`cobb`, most likely, given the
  cross-tool agent-porting nature of the work) writes the actual `plans/` document.

## Open questions for the actual `tico` interview

- What should the first scheduled task actually check/do? (A conservative read-only-flavored
  health/hygiene check was suggested as a starting example, not a decision.)
- What cadence — how often should it run?
- Is the kaizen-learning-capture gap (blocked on C-310) acceptable for a first version, or does
  this requirement need to wait on C-310, or get its own interim answer?
- Should the local-model config be scoped globally (every project, matching the existing
  `devops` agent's user-scoped reach) or just to this repo initially?
- What's the acceptable risk tolerance for the destructive-ops guard translating from Claude
  Code's regex-based hook to OpenCode's simpler glob-pattern permission matching — is a
  best-effort glob translation sufficient, or does this need a stronger mechanism (e.g. routing
  risky operations through a wrapper script instead of allowing direct `docker`/`redis-cli`
  calls)?
