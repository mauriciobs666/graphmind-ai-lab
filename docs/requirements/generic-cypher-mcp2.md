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
- As **any agent**, I want to query any other agent's working-memory graph directly — including
  entries `cobb` hasn't distilled yet — so I benefit from what a teammate discovered without
  waiting for a distillation pass.
- As **any agent**, I want to ask one question that reaches raw learnings across the *whole*
  team, not loop over eleven-plus separate graphs myself, so "collective memory" is actually
  collective, not just parallel silos.
- As **each of the eleven remaining agents**, I want to record a new learning by writing it into
  my own graph instead of appending to `kaizen/inbox.md`, exactly like `graph-dba` already does,
  so it's immediately queryable by the rest of the team.
- As **`cobb`**, I want my existing distillation workflow (verify → route → append to
  `history.md` → curator-clear) to keep working unchanged, now against every agent's graph
  (including my own), so my role doesn't change — only where the raw material lives, for the
  whole team instead of one pilot agent.
- As **whoever creates a new agent**, I want that agent to start directly on the graph-backed
  pattern, so nobody has to retrofit it later the way this delivery is retrofitting today's
  eleven.
- As the **stakeholder**, unchanged from M5: I still watch graph state via FalkorDB's existing
  web console; my own direct read/write access to the knowledge plane remains a later phase.

## Functional requirements
- **FR-1** — Each of the eleven agents not yet migrated — `analyst`, `architect`, `cobb`,
  `coder`, `data-scientist`, `devops`, `frontend-engineer`, `qa-engineer`, `tdd-engineer`,
  `teco`, `tico` — gets a graph-backed kaizen working-memory raw-capture layer, on **exactly the
  mechanism M5 built and proved on `graph-dba`** (the generic Cypher MCP write path, the same
  author/curator enforcement). No new write mechanism is required by this delivery beyond FR-7.
- **FR-2** — For each of the eleven, going forward that agent writes a new raw learning
  **directly into its own graph**, attributed to itself — this replaces appending to
  `kaizen/inbox.md`, exactly as FR-2 did for `graph-dba` in M5.
- **FR-3** — Each agent's current `kaizen/inbox.md` content, as of that agent's migration, is
  imported into its graph **once**, so nothing already captured is lost in the cutover.
- **FR-4** — After that import, each agent's `kaizen/inbox.md` is kept, **frozen**, as a
  historical snapshot — no longer written to, and clearly distinguishable (to a reader) from a
  live file, same signal `graph-dba`'s already uses.
- **FR-5** — Each agent's `kaizen/history.md` is **unchanged**: `cobb`'s promotions are still
  appended there, in the same format, at the same cadence, as today.
- **FR-6** — Any agent can read any other migrated agent's working-memory graph directly —
  including entries `cobb` hasn't distilled yet — not gated behind distillation. Extends M5's
  FR-6 from one agent's graph to all twelve.
- **FR-7** — There is a **single query surface that reaches every migrated agent's raw working
  memory in one query** — an agent (or the stakeholder) does not have to know about, and loop
  over, each agent's graph individually to ask "what has the team learned about X." This is new
  relative to M5, whose pilot had only one agent's working memory to query.
- **FR-8** — A working-memory entry captures the same fields as today's markdown entry (the
  dated fact, its evidence, the context it surfaced in, a suggested home), queryable via graph
  traversal, for every migrated agent — same shape as M5's FR-7.
- **FR-9** — Write/modify access keeps M5's two shapes, applied per agent: **author** (an agent
  creates entries attributed only to itself — including `cobb` authoring its own) and
  **curator** (`cobb` clears/marks-as-promoted an entry it doesn't own, across all migrated
  agents, exactly as it already does for `graph-dba`'s graph today). No new access shape is
  introduced.
- **FR-10** — `cobb`'s distillation workflow continues to function end to end, **unchanged in
  cadence and process**, for every migrated agent including itself — the only change is that raw
  capture now targets a graph instead of a markdown file, for eleven more agents.
- **FR-11** — Every doc describing the standing kaizen-inbox convention (`claude/AGENTS.md`,
  `claude/README.md`, `docs/BACKLOG.md`, and each migrated agent's own operative-prompt "Learning
  capture" section) is updated to describe that agent's **actual** post-migration behavior — no
  remaining unconditional claim that an agent appends to `inbox.md`.
- **FR-12** — The standing convention for creating a new agent is updated so that a newly
  created agent is born directly on the graph-backed pattern — no `kaizen/inbox.md` is ever
  created for it, and no future retrofit is needed.
- **FR-13** — Rollout is **incremental, not atomic**: each agent's migration is independently
  complete and valuable as it lands (a state where, say, 8 of 11 have migrated and 3 have not is
  a valid in-progress state of this delivery, not a failure of it) — this delivery is considered
  fully done once all eleven have landed, but is not blocked from making real progress before
  that point.

*Context for the architect (not a requirement):* how FR-7's team-wide query surface is
technically achieved — one shared graph with an agent-partition property, per-agent graphs plus
a federated query helper, something else — is a design decision, not specified here. So is
whether each agent's graph follows `graph-dba`'s `kaizen_<agent>` naming pattern. Sequencing/
batching of which agents migrate in what order is also left to implementation planning, given
FR-13 accepts incremental delivery.

## Out of scope
- **falkor-chat data integration** — unchanged from M5: linking knowledge-plane entries to chat
  threads/messages is a later phase.
- **Design/requirements/plan documents becoming graph data** — unchanged from M5: still deferred.
- **Migrating any module's `docs/BACKLOG.md` / task-backlog state to the graph** — unchanged from
  M5: the "project-management" half of the original motivation stays deferred.
- **The stakeholder's own direct read/write access to the knowledge plane** — unchanged from M5:
  FalkorDB's existing web console remains sufficient monitoring for now.
- **Guaranteed semantic/similarity search** — unchanged from M5: structural traversal is what
  "done" requires; vector-indexed retrieval remains a stretch goal, not a requirement, and is not
  newly required by the team-wide query surface (FR-7) either.
- **Hardened/cryptographic access control** — unchanged from M5: same trusted
  self-identification level the rest of the repo runs at.
- **Deleting any agent's `kaizen/inbox.md`.** Every migrated agent's file stays, git-tracked,
  frozen — not removed.
- **Redesigning the write mechanism itself.** FR-1 reuses M5's author/curator mechanism as-is;
  this delivery is a rollout of an existing mechanism to more agents, not a mechanism redesign
  (FR-7's query surface is additive, not a change to the write path).

## Acceptance criteria
- **AC-1** — Given a migrated agent's working-memory graph, when another agent queries it, it
  gets back entries with the same fields today's markdown format carries (date, fact, evidence,
  context, suggested home) — including entries `cobb` has not yet distilled.
- **AC-2** — Given an agent's `kaizen/inbox.md` entries as of its migration, when the one-time
  import for that agent runs, every one of them is present in its graph afterward.
- **AC-3** — After an agent's import, its `kaizen/inbox.md` still exists, unchanged in content,
  and a reader can tell it is no longer live.
- **AC-4** — Given a migrated agent discovers a new learning, when it records it, the entry
  appears in that agent's graph (not `inbox.md`), attributed to it, immediately queryable by
  another agent.
- **AC-5** — Given a raw entry in any migrated agent's graph, when `cobb` runs its distillation
  workflow, the promotion is appended to that agent's `kaizen/history.md` in the existing format,
  and the entry is cleared from the graph.
- **AC-6** — Given the trusted self-identification model, when a tool call claims to be one
  migrated agent and attempts to create or curator-clear an entry attributed to another (outside
  `cobb`'s recognized curator role), the system rejects it or otherwise does not accept it as
  authored by the claimed-against agent.
- **AC-7** — Given the team-wide query surface (FR-7), when any agent asks for raw learnings
  across the team, it gets results spanning every migrated agent's graph in one query — not
  eleven-plus separate lookups.
- **AC-8** — No reader finds `claude/AGENTS.md`, `claude/README.md`, `docs/BACKLOG.md`, or any
  migrated agent's own operative prompt silent about or contradicting that agent's actual
  behavior (FR-11 executed).
- **AC-9** — The documented convention for creating a new agent (whatever doc governs that
  process) directs a new agent to be born on the graph-backed pattern, with no `kaizen/inbox.md`
  step in it (FR-12 executed).
- **AC-10** — At any point before all eleven have migrated, the set of already-migrated agents
  independently satisfies AC-1…AC-6 for themselves — partial progress is verifiably real
  progress, not merely claimed (FR-13).

*How thoroughly each individual acceptance criterion is exercised per agent — an independent
pass for all eleven vs. a sampled/consolidated pass with programmatic checks for the rest — is
left to `qa-engineer`'s test strategy when this reaches that stage; the stakeholder had no
preference (see Decision log).*

## Open questions
*(none)*

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
- 2026-08-19 — Atomic cutover vs. incremental? → **Incremental** — each agent's migration is
  independently delivered/valuable as it lands; a partial state (e.g. 8 of 11 migrated) is real
  progress, not a failure (FR-13/AC-10).
- 2026-08-19 — Does the FalkorDB memory-footprint concern (flagged in M5) need re-examining at
  12-agent scale? → **No** — same accepted trade-off as M5, not a new sizing/verification
  concern for this delivery.
- 2026-08-19 — Verification depth: independent acceptance pass per agent, or one consolidated
  pass? → **Left to `qa-engineer`'s test strategy** — not decided here; the requirements doc
  states the truth conditions (AC-1…AC-10), not the rigor level used to check them.
- 2026-08-19 — Should the graph-backed pattern become the standing default for any future new
  agent? → **Yes, now** — the agent-creation convention itself is updated (FR-12/AC-9), not
  deferred to whenever the next agent happens to be created.
