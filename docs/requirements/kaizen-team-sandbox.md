# Agent-team graph sandbox — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M<n> TBD) · **Last updated:** 2026-08-31

## Intent

The stakeholder is about to start a bigger integration of the agent team with the graph (in the
spirit of what already exists for `kaizen_team`), and cannot risk that ongoing development work
interfering with the stability of the agent team's real, currently-relied-upon graph data. In the
stakeholder's own words: **agent development needs to happen in a separate sandbox environment so
it can't collapse the real thing.**

Today there is one shared FalkorDB instance (`falkordb-dev`) used for local development across
the repo, and `kaizen_team` — the graph every agent's raw learnings are written into via
`(:Agent)-[:PRODUCED]->(:KaizenEntry)`, and that `cobb` periodically distills — lives on it
alongside everything else. There is no separation today between "trying something out" and "the
data the team is actually relying on right now."

## Problem & current state

- `kaizen_team` is live, continuously written data: every agent's raw learnings capture writes
  directly into it during normal operation (`claude/AGENTS.md`).
- The FalkorDB instance it lives on is shared dev infrastructure with no environment boundary —
  a query, schema change, or heavy workload run while developing the graph integration runs
  directly against the same engine and, if scoped to `kaizen_team`, the same data other agents
  depend on right now.
- Three distinct ways development work could hurt production were identified and are all in
  scope: accidental writes/deletes against the real data, schema/shape changes that conflict with
  or corrupt what's there, and engine-level load or crashes that degrade or interrupt access for
  everyone — not just the data itself.
- There is currently no mechanism to try out a structural change to the agent-team graph without
  it being, by construction, a change to the live thing.

**Stated preference (not a requirement):** the stakeholder's own framing is "separate FalkorDB
instances" for production and development. That's their suggested shape for the isolation and is
recorded here as context for the architect — the actual requirement is the isolation behavior
described below (data, schema, and engine-level), not a mandate on the specific mechanism.

## User stories

- As the person building the agent/graph integration, I want to develop and test schema and query
  changes for the agent team's graph in a sandbox that can't touch the real data, so that a
  mistake during development can't corrupt or lose data the team currently relies on.
- As the person doing this work, I want engine-level isolation, not just data isolation, so that
  heavy querying or a crash while experimenting doesn't degrade or take down agents' real-time
  access to the real `kaizen_team` graph.
- As the stakeholder, I want the decision of whether a given piece of work needs to go through the
  sandbox to be based on that work's risk/blast radius, not a blanket rule, so that minor changes
  aren't slowed down by unnecessary process.
- As the stakeholder, I want any structural (schema-shape) change destined for the real graph, and
  the decision to isolate a change in the first place, to be negotiated with me before it happens
  or before it's applied, so I stay in control of what changes the team's shared memory.
- As any agent or contributor doing agent/graph work — not just this specific integration project,
  and not just the stakeholder — I want to be able to request a sandbox for my own work, so I
  don't have to build isolation from scratch or need special standing to get it.
- As the requester of a sandbox, I want to specify what my work needs (scope, isolation
  requirements, the risk/blast-radius call from FR-5) without having to build or run the
  environment myself, so that infrastructure work stays with the people who own infrastructure.

## Functional requirements

- **FR-1 — Data isolation.** A development environment for agent-team graph work must be isolated
  from the environment agents' real, ongoing usage writes to ("production"), such that a write or
  delete made in development cannot affect production data.
- **FR-2 — Schema isolation.** The isolation must cover schema/index/constraint changes — a
  schema experiment made in development must not alter or conflict with production's existing
  schema.
- **FR-3 — Engine-level isolation.** The isolation must cover availability/performance — load,
  crashes, or restarts caused by development work must not degrade or interrupt production's
  availability to agents operating normally.
- **FR-4 — Unaffected normal operation.** Agents' ordinary kaizen-writing behavior (a `PRODUCED`
  write via `mcp__cypher__query` during a normal session) must continue to land in production
  exactly as it does today, without requiring reconfiguration triggered by the sandbox's
  existence.
- **FR-5 — Risk-based use, not a blanket rule.** Using the sandbox is not mandatory for every
  change. Before work that touches the agent team's graph begins, its risk/blast radius must be
  evaluated to decide whether it warrants sandbox development first; a minor, low-risk change may
  proceed directly against production. This call is made case by case for now — no fixed
  checklist exists to mechanize it yet.
- **FR-6 — Stakeholder negotiation.** Both (a) the decision that a given piece of work needs
  sandbox isolation, and (b) any structural (schema-shape) change destined for production, must be
  negotiated with and approved by the stakeholder before it happens.
- **FR-7 — Migration-impact analysis before promotion.** Before a validated sandbox change is
  applied to production, a migration-impact analysis must determine what specifically needs to
  move (schema only, schema plus data, or something else) — decided per feature, not by one fixed
  universal rule.
- **FR-8 — Standing capability, open to anyone.** Once built, the sandbox is a standing
  capability available to any agent or contributor whose future work touches the agent team's
  graph — not a one-time setup solely for the upcoming integration project, and not limited to
  the stakeholder who requested it.
- **FR-9 — Responsibility split.** Specifying what a given piece of work needs from its
  environment (scope, isolation requirements, and the risk/blast-radius call from FR-5) is the
  **requester's** responsibility; creating, deploying, and maintaining the actual environment(s)
  is **devops's** responsibility.

## Out of scope

- The `cpg_*` graphs (Joern Code-Property-Graphs) — explicitly not part of the protected scope;
  only `kaizen_team` is.
- `falkor-chat`'s own workspace/production deployment concerns — a different system, not what
  triggered this request.
- The specific promotion mechanism/tooling for moving schema or data between environments — that
  is a design (HOW) question for the architect, not this document.
- A fixed, universal promotion recipe — FR-7 deliberately leaves this per-feature, decided by each
  change's own migration-impact analysis.
- Continuous/automatic data sync between sandbox and production.
- A mandatory sandbox-for-everything policy — FR-5 explicitly rules this out.

## Acceptance criteria

- Given a destructive operation is deliberately run against the sandbox environment (e.g.
  dropping or mutating its copy of the agent-team graph), when checked afterward, then production
  `kaizen_team` and its data are unaffected.
- Given a schema/index/constraint change is made in the sandbox, when checked afterward, then
  production's schema is unaffected unless and until the change is deliberately promoted.
- Given development work generates heavy query load or a crash in the sandbox, when checked
  afterward, then agents' real-time reads/writes against production `kaizen_team` continue to
  succeed with no degradation attributable to the sandbox incident.
- Given an agent performs its normal kaizen-writing behavior with no special configuration, when
  checked, then the write lands in production, unaffected by the sandbox's existence.
- Given a piece of work touching the agent-team graph is proposed, when its risk/blast radius is
  evaluated, then there is a recorded decision (with the stakeholder) on whether it goes through
  the sandbox — not an assumption either way.
- Given a change is ready to move from sandbox to production, when the promotion is prepared,
  then a migration-impact analysis exists identifying what needs to move, and any structural
  change carries recorded stakeholder sign-off before being applied.

## Open questions

None outstanding — all three prior open questions were resolved in conversation (see decision
log): the risk/blast-radius call is case-by-case for now, the sandbox can land in parallel with
early low-risk exploration rather than gating it, and it's open to any agent/contributor, not
just the stakeholder.

## Decision log

- 2026-08-30 — What is "graphmind-ai-lab" scope of this request? → Not `falkor-chat` workspace
  data; it's about the shared `falkordb-dev` instance hosting `kaizen_team`, which every agent
  writes to via `cypher-mcp`. Confirmed no prior requirement/decision blocks or reverses this —
  fresh ground (checked `falkor-chat/docs/DESIGN.md` §1, `docs/BACKLOG.md`).
- 2026-08-30 — Which data needs protecting? → `kaizen_team` graph specifically. `cpg_*` graphs
  and the engine as a general resource were offered but not selected as in-scope data; the engine
  concern resurfaced separately as an availability requirement (FR-3), not a data-protection one.
- 2026-08-30 — What kind of interference is feared? → All three: accidental writes/deletes,
  schema/shape changes, and engine load/availability — informs FR-1/FR-2/FR-3.
- 2026-08-30 — Restated in stakeholder's own words → "agent development is done in a separate
  sandbox environment so it doesn't collapse." Adopted as the Intent statement.
- 2026-08-30 — Should dev start from a copy of production data or clean? → "Depends on the
  feature requirements and data available in the prod environment" — left as a per-feature call,
  not fixed here; not elevated to an FR since it's a promotion/seeding mechanism (HOW).
- 2026-08-30 — Is promoting a validated change to production in scope of this document? → Yes,
  in scope (FR-7).
- 2026-08-30 — What actually needs to move on promotion? → "Migration impact analysis should be
  done according to the feature" — adopted as FR-7 (per-feature analysis, not a fixed universal
  rule).
- 2026-08-30 — Is this a one-off need or a standing capability? → Standing capability, but any
  structural change is negotiated with the stakeholder when needed (FR-6/FR-8).
- 2026-08-30 — Follow-up: is sandbox use mandatory for every change? → No — "should evaluate risk
  and suggest isolated sandbox according to the blast radius of the feature." Adopted as FR-5,
  and folded into FR-6 (the risk decision itself is also negotiated with the stakeholder, not
  assumed).
- 2026-08-30 — Readback follow-ups: (1) Is risk/blast-radius evaluation a fixed checklist or
  case-by-case? → Case by case for now (FR-5). (2) Must the sandbox exist before the integration
  project starts? → No, it can land in parallel with early low-risk exploration — not a gate on
  starting the project. (3) Is sandbox use scoped to the stakeholder alone? → No, open to any
  agent/contributor (FR-8).
- 2026-08-30 — Responsibility split volunteered by the stakeholder → Specifying a piece of work's
  environment requirements is the requester's job; creating, deploying, and maintaining the
  actual environment(s) is devops's job. Adopted as FR-9.
