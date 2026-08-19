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
- Single cross-team query surface vs. per-agent graphs (mirroring today's one-file-per-agent
  shape) — does "collective memory" need to mean one query reaches every agent's raw entries?
- Does `cobb`'s distillation role/cadence need to change to handle 11x the raw-capture volume?
- Sequencing/batching of the rollout across the eleven agents.
- Anything agent-specific that breaks the graph-dba template (e.g. `cobb`'s own dual role as
  both an author of its own inbox *and* the curator of everyone else's).

## Decision log
- 2026-08-18 — Session opened. Stakeholder: "extend the [graph-backed kaizen] inbox to all
  agents like we did with graph-dba." `tico` confirmed this is the M5 requirements doc's own
  named follow-on (its Out of scope: "Extending to the rest of the team is a follow-on decision
  made after this one is evaluated"), grounded in `docs/requirements/generic-cypher-mcp.md`,
  `docs/BACKLOG.md` M5 section, and `claude/AGENTS.md`. Opened as a successor document on the
  same slug family (`generic-cypher-mcp2.md`), since M5's doc is `archived` (approved/gated/
  executed against) — not a fresh, unrelated topic.
- 2026-08-18 — Scope? → **All eleven remaining agents** (analyst, architect, cobb, coder,
  data-scientist, devops, frontend-engineer, qa-engineer, tdd-engineer, teco, tico) — not a
  subset, not deferred as a separate later batch.
- 2026-08-18 — Why now? → **M5 proved out end-to-end** (8/8 acceptance criteria, no defects) —
  no reason to wait for more evidence the pattern works.
