# Kaizen agent/learning-note ontology — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M8) · **Last updated:** 2026-08-20

## Intent
Make `cobb`'s distillation work better by making the kaizen graphs' agent/note structure explicit
and queryable instead of implicit in a string property and free-text `context`. Three concrete
payoffs the stakeholder wants: (1) **one-query, team-wide attribution lookups** — "every note
agent X ever emitted" as a real traversal, not a per-graph string match; (2) **cross-agent
provenance tracing** — when a note is really about another agent's behavior, that link is an
explicit, queryable edge instead of buried in free text; (3) **spotting cross-agent duplicate or
related learnings** — two agents independently noticing something related becomes discoverable
instead of missed.

## Problem & current state
Every agent's raw working-memory graph (`kaizen_<agent>`, one per agent, rolled out team-wide as
of 2026-08-20 per `docs/requirements/generic-cypher-mcp2.md`, M7) stores learning notes as
`:KaizenEntry` nodes. Attribution today is a **plain string property**, `author: "<agent-slug>"`
— there is no `:Agent` node and no relationship connecting a note to who emitted it, or to any
other agent or note it might be about.

This was a deliberate, considered decision, not an oversight: `docs/plans/generic-cypher-mcp-graph.md`
§2 (M5, `graph-dba`) explicitly weighed a relationship shape
(`(:KaizenEntry)-[:AUTHORED_BY]->(:Agent {name, kind})`) against the plain property, and chose the
property **for that pilot delivery**, with a stated **revisit trigger**: *"if this pattern ever
extends past `graph-dba` to genuinely multiple authoring agents … and a consumer wants 'give me
every entry from agent X across N entries' as a first-class, frequently-run query, or wants to
attach metadata to the author itself … that's the point where an `:Agent`/`:Actor` identity node
earns its complexity."* M7's team-wide rollout (all 12 agents now write to their own graph) is a
plausible instance of that trigger firing — flagged here as prior-decision provenance, not
re-litigated by this document.

## User stories
- As **`cobb`**, I want every note's producing agent to be a real, queryable graph node (not just a
  string property), so I can ask "every note agent X ever produced" as one traversal.
- As **`cobb`**, I want a note that's really about a different agent than its producer to be
  explicitly linked to that agent, so the note surfaces in that agent's own review queue instead
  of only ever being seen through its producer's queue.
- As **`cobb`**, I want two agents' independently-produced notes that both concern the same
  third agent to be discoverable through that shared connection, so I can notice a cross-agent
  pattern I'd otherwise miss reading each agent's notes in isolation.
- As **`cobb`**, I want a note that mentions several agents to stay available to each of them in
  turn, so no agent's relevant note disappears just because another agent's pass happened first.

## Functional requirements
- **FR-1** — Each team agent is represented as its own node in the graph (not merely a string
  value), so it can be a shared traversal target across every note that names it.
- **FR-2** — Every new note (created after this feature ships) is connected to the agent that
  produced it via a queryable graph relationship — the producer link. Preferred name:
  `PRODUCED_BY` (naming preference, not binding — see Decision log). Historical notes created
  before this feature are **not** retrofitted (they keep working via the existing `author`
  property).
- **FR-3** — A note may additionally be connected to zero or more other agents it is about, via a
  separate queryable relationship — the reference link. Preferred name: `MENTIONS` (naming
  preference, not binding). This is optional per note ("when applicable"), and is distinct from
  the producer link (FR-2): a note's producer and the agent(s) it references are never the same
  relationship type.
- **FR-4** — The reference link (FR-3) is set by `cobb`, during its existing per-agent
  distillation pass — not by the note's producing agent at write time.
- **FR-5** — A note connected to another agent via the reference link (FR-3) is included in that
  mentioned agent's own future distillation review, in addition to its producer's — i.e. `cobb`'s
  per-agent review scope covers both "notes this agent produced" and "notes that mention this
  agent," regardless of who produced them.
- **FR-6** — A note is only removed from the graph once every relationship pointing at it has been
  consumed: the producer relationship (FR-2) is always resolved on the producing agent's own pass,
  independent of how many reference links remain; each reference link (FR-3) is independently
  resolved (deleted) only when `cobb` reviews the note in that specific mentioned agent's pass —
  removing just that one edge, never the note, while any other reference link still remains. The
  note node itself is deleted only when its last remaining relationship (of either kind) is
  consumed.
- **FR-7** — This feature is designed and delivered against the graph topology M7
  (`docs/requirements/generic-cypher-mcp2.md`) actually ships — **M8 does not start design until
  M7's plan gate closes and its consolidation lands.**

*Context for the architect (not a requirement):* exact relationship-type names (`PRODUCED_BY` vs.
`EMITTED_BY`, `MENTIONS` vs. `REFERS_TO`), whether the producer link fully replaces the `author`
property or coexists with it, and the query mechanics behind FR-5/FR-6 (how `cobb`'s per-agent
pass is scoped, how partial-edge deletion is expressed in Cypher) are design decisions, informed
by `falkor-chat`'s locked naming precedent (`PRODUCED`, `MENTIONS`) but not dictated by it — a
different graph/component, no literal namespace collision.

## Out of scope
- **Retrofitting historical entries** with the new relationships (FR-2's note) — new entries only.
- **Design/requirements/plan documents becoming graph data** — unchanged from M5/M7.
- **Any change to `cobb`'s distillation cadence or its markdown `history.md` output format** —
  this feature changes what's queryable in the raw graph and when a note is safe to delete, not
  what gets promoted or how often.
- **Starting before M7 lands** (FR-7) — this document can reach "Ready for design," but the
  architect should not begin designing against a substrate that's still changing under M7.

## Acceptance criteria
- **AC-1** — Given a new note created after this feature ships, when any agent queries the graph,
  the note's producer is reachable via a graph traversal (not a string-equality check on a
  property) to a real agent node.
- **AC-2** — Given a new note that `cobb` determines is about a different agent than its producer,
  when `cobb` tags it during the producer's or a prior pass, that note subsequently appears when
  `cobb` reviews the mentioned agent's own queue.
- **AC-3** — Given a note mentioning two or more agents, when `cobb` reviews it in the context of
  one of those agents and no other relationship remains, the note is fully removed; when at least
  one other relationship remains (mention or unresolved producer link), only that one relationship
  is removed and the note persists.
- **AC-4** — Given the producing agent's own distillation pass runs on a note, the producer
  relationship is resolved regardless of how many mention relationships still point at the note.
- **AC-5** — Given two notes, produced by two different agents, that both mention a third common
  agent, a query starting from that third agent's node reaches both notes.
- **AC-6** — No design work for this feature starts before M7's coordination log records its plan
  gate closed and its consolidation delivered.

## Open questions
1. Exact relationship-type names (`PRODUCED_BY`/`MENTIONS` vs. alternatives) — left to the
   architect (see FR-7's "Context for the architect" note); not blocking readiness.
2. Whether the `author` property is dropped once the producer relationship exists, or the two
   coexist for a transition period — left to the architect.
3. Any properties beyond identity on the new agent node (e.g. role/kind) — not raised by the
   stakeholder; assume identity-only unless a concrete need surfaces at design time.

## Decision log
- 2026-08-20 — Session opened. Stakeholder: new milestone M8, "graph adoption" — improve the
  kaizen ontology with `:Agent` nodes and relationships (`EMITTED_BY` and `REFERS_TO`, the latter
  "when applicable", connecting agents and learning notes in both directions).
- 2026-08-20 — `tico` checked `docs/requirements/` and `docs/BACKLOG.md` for prior decisions this
  might reverse: found `docs/plans/generic-cypher-mcp-graph.md` §2 (M5) considered and rejected an
  `:Agent` node in favor of a plain `author` property, with an explicit revisit trigger that M7's
  team-wide rollout plausibly satisfies (see Problem & current state). Not a blocker — recorded as
  provenance the eventual design must address as a supersession, not a fresh, unrelated FR.
- 2026-08-20 — What's the payoff for `cobb` specifically? → **All three offered**: one-query
  team-wide attribution lookups, cross-agent provenance tracing, and spotting cross-agent
  duplicate/related learnings. Not mutually exclusive; all three motivate the feature.
- 2026-08-20 — Stakeholder asked `tico` (Mode 2) to check `falkor-chat`'s relationship-naming
  standards for consistency. Found: `UPPER_SNAKE` type names (matches); a **locked decision (D2)**
  in `falkor-chat/docs/DESIGN.md` naming `PRODUCED` (not `EMITTED`) for "who/what created this
  artifact," reserving `EMITTED` for a distinct Message→Message citation-provenance edge; and
  `(:Message)-[:MENTIONS]->(:Entity)` — content node → what it's about, not entity-to-entity.
  Presented as informal Mode-2 findings, not a binding cross-component rule (different graphs, no
  literal namespace collision) — offered so the stakeholder's naming choice here is informed.
- 2026-08-20 — In light of the `MENTIONS` precedent, reconsidered the "refers to" shape (was:
  Agent→Agent derived from a note's content) → **now Note → Agent**, matching falkor-chat's
  content-node-points-at-referent pattern: a `:KaizenEntry` points directly at the `:Agent` it's
  about, not two `:Agent` nodes pointing at each other. **Supersedes the previous answer.**
- 2026-08-20 — Authorship edge name: keep `EMITTED_BY`, or align to `falkor-chat`'s `PRODUCED`
  vocabulary? → **Align to `PRODUCED_BY` (or similar)** — stated as a naming preference for the
  architect, not a locked requirement.
- 2026-08-20 — How does M8 relate to M7 (`generic-cypher-mcp2`), which is still in flight and
  currently (uncommitted plan revision) redirecting from per-agent `kaizen_<agent>` graphs to one
  shared `kaizen_team` graph, `author`-partitioned? → **M8 waits for M7 to land** — the ontology is
  designed once, against the real final substrate. **M8 is sequenced after M7; do not start design
  before M7's plan gate closes and its consolidation ships.**
- 2026-08-20 — Retrofit existing entries with the new edges, or new entries only? → **New entries
  only, going forward.** Historical entries keep working via the `author` property; no one-time
  migration of old entries into the new edge shape.
- 2026-08-20 — Who sets the `MENTIONS` edge (a note → another agent it's about)? → **`cobb`, during
  distillation** — not the writing agent at creation time. This surfaced a real timing conflict
  with curator-clear's hard `DETACH DELETE` (see next two entries).
- 2026-08-20 — Reconciling `MENTIONS` with curator-clear's hard delete: stakeholder's resolution —
  **`cobb` reviews one agent's universe at a time** (its existing per-agent distillation cadence).
  When a note isn't really for the agent whose queue `cobb` is currently distilling but mentions
  another agent, `cobb` tags it `MENTIONS`→that agent; the note then **surfaces again later as part
  of the mentioned agent's own queue listing** — `MENTIONS` is a routing mechanism into another
  agent's review, not decoration.
- 2026-08-20 — When is it safe to actually delete the note node? → A note may mention **multiple**
  agents (multiple `MENTIONS` edges). `cobb` counts remaining `MENTIONS` edges before deleting: if
  more than one remains, it deletes **only the edge** for the agent whose pass it's currently
  reviewing, leaving the node (and other edges) intact; the node itself is deleted only once its
  last remaining edge is consumed.
- 2026-08-20 — Does that same defer-and-count rule apply to `PRODUCED_BY` (the producer's own
  review), or is `PRODUCED_BY` special? → **`PRODUCED_BY` is special**: the producing agent's own
  pass always fully resolves/removes the `PRODUCED_BY` edge when it runs, regardless of how many
  `MENTIONS` edges remain. Only `MENTIONS` edges follow the count-and-defer-deletion rule; the node
  survives independently of `PRODUCED_BY` for as long as any `MENTIONS` edge remains.
