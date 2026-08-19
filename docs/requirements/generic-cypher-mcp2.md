# Generic Cypher MCP — team-wide kaizen inbox rollout — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M6) · **Last updated:** 2026-08-18

## Intent
M5 (`docs/requirements/generic-cypher-mcp.md`) proved the graph-backed working-memory pattern on
one pilot slice — `graph-dba`'s kaizen inbox — and explicitly deferred the question of extending
it to the rest of the team as "a follow-on decision made after this one is evaluated." The
stakeholder is now making that call: roll the same pattern out to the other agents' kaizen
inboxes, the way it was done for `graph-dba`.

## Problem & current state
Every agent except `graph-dba` still captures raw kaizen learnings by appending to its own
`claude/<agent>/kaizen/inbox.md` (append-only markdown), distilled periodically by `cobb`
(verify → route → append to `history.md` → clear). `graph-dba` alone has moved its raw-capture
layer to a FalkorDB graph (`kaizen_graph_dba`) via `mcp__cpg__query`'s write path, while its
`history.md` stays markdown and unchanged. The other eleven agents' raw learnings are siloed in
per-agent markdown files that only `cobb` reads across agents, and are invisible to any other
agent until a distillation pass promotes them.

Current per-agent `kaizen/inbox.md` sizes (context, not yet a requirement):
`analyst` 40, `architect` 19, `cobb` 19, `coder` 18, `data-scientist` 118, `devops` 18,
`frontend-engineer` 18, `qa-engineer` 43, `tdd-engineer` 40, `teco` 41, `tico` 47 lines.

## User stories
*(to be filled in as the interview proceeds)*

## Functional requirements
*(to be filled in as the interview proceeds)*

## Out of scope
*(to be filled in as the interview proceeds)*

## Acceptance criteria
*(to be filled in as the interview proceeds)*

## Open questions
- Which agents are actually in scope for M6 — literally all eleven remaining, or a subset/staged
  rollout?
- One-shot migration for all in-scope agents, or phased (M5 deliberately picked one low-risk
  pilot first)?

## Decision log
- 2026-08-18 — Session opened. Stakeholder: "extend the [graph-backed kaizen] inbox to all
  agents like we did with graph-dba." `tico` confirmed this is the M5 requirements doc's own
  named follow-on (its Out of scope: "Extending to the rest of the team is a follow-on decision
  made after this one is evaluated"), grounded in `docs/requirements/generic-cypher-mcp.md`,
  `docs/BACKLOG.md` M5 section, and `claude/AGENTS.md`. Opened as a successor document on the
  same slug family (`generic-cypher-mcp2.md`), since M5's doc is `archived` (approved/gated/
  executed against) — not a fresh, unrelated topic.
