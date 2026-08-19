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
- Atomic cutover (nothing is "done" until all eleven are live) vs. incremental (each agent's
  migration is independently valuable/verifiable as it lands)?
- Does the FalkorDB memory-footprint concern that shaped M5's working-memory-vs-permanent-record
  split need re-examining now that the pattern scales to twelve agents' working memory at once?
- Verification depth: an independent acceptance pass per agent (mirroring M5's AC-1…AC-8), or one
  consolidated pass that spot-checks a few agents plus programmatically confirms the rest, since
  the underlying mechanism is already proven?

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
- 2026-08-19 — Team-wide query surface? → **Needs a team-wide view** — an agent (or the
  stakeholder) should be able to reach raw learnings across the whole team in one query, not loop
  over eleven (now twelve, counting `graph-dba`) separate graphs one at a time. This is a new
  capability M5 didn't need (its pilot had only one agent's working memory to query).
- 2026-08-19 — `cobb`'s distillation cadence at 11x the raw-capture volume? → **No change** — same
  cadence, same workflow, just a different backend, exactly like M5 did for `graph-dba` alone.
- 2026-08-19 — Is `cobb`'s own kaizen inbox (author of its own learnings *and* curator of
  everyone else's) a special case? → **No** — `cobb` migrates too, on the same template: it
  authors its own entries into the graph and later curator-clears them itself, same as it already
  does for the rest of the team.
