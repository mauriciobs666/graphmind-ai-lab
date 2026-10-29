# Agent-team workspace sandbox — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Supersedes:** `docs/requirements/kaizen-team-sandbox.md` · **Last updated:** 2026-10-28

## Intent

Carried forward from the predecessor, to be re-confirmed: **agent development needs to happen in a
separate sandbox environment so it can't collapse the real thing.** The "real thing" is no longer
the retired `kaizen_team` graph; it is the agent team's live data in the falkor-chat workspace
`ws:agent-team`.

## Problem & current state

- The predecessor (`docs/requirements/kaizen-team-sandbox.md`) protected `kaizen_team`. Its plan
  (`docs/plans/kaizen-team-sandbox.md`, V5) was approved but never built, and `kaizen_team` has
  since been retired and deleted (`claude/AGENTS.md`).
- `ws:agent-team` is now the agent team's live store. It holds every agent's raw learnings capture
  (written via falkor-chat's `ingest_document`) and the distilled knowledge base that agents search
  on demand (366 documents as of 2026-10-20, per `claude/docs/requirements/coordination-graph.md`).
- It shares one FalkorDB instance with the other falkor-chat workspaces (`ws:acme`, `ws:eval`,
  `ws:test`, `ws:demo`, …) and `cpg_falkorchat`.
- _(to be confirmed in the interview)_

## User stories

- As anyone developing against falkor-chat (agent or human), I want a throwaway copy of the whole
  stack to experiment in, so that a mistake can't damage the real workspaces the team and the
  storefront rely on.
- As that developer, I want the sandbox to behave like production, so that what works in it works
  for real.
- As the stakeholder, I want a runaway experiment to be unable to starve or crash the real system,
  even on the same machine.
- As the stakeholder, I want sandboxes used where the risk warrants it, not for everything, and I
  want sign-off on that call and on any structural change before it goes live.
- As a requester, I want to say what I need (including what data to start with) and have devops
  build it, so that I don't have to build infrastructure myself.

## Functional requirements

- **FR-1 — Protected scope.** "Production" means all live falkor-chat data: every real workspace
  (e.g. `ws:agent-team`, `ws:acme`), not only the agent team's.
- **FR-2 — Containerized (stakeholder constraint).** The sandbox runs as containers. The
  stakeholder mandated this explicitly, so it is a constraint and not just a preference. The
  outcomes it must deliver are FR-3..FR-6.
- **FR-3 — Disposable.** A sandbox can be created and removed completely in one step each. Removal
  leaves nothing of it behind on the machine (no leftover data, volumes, or running processes).
- **FR-4 — Whole stack.** A sandbox contains the full path that agents and users exercise: the
  database and the falkor-chat server they talk to. It does not contain only the database.
- **FR-5 — Reproducible and production-like.** Every sandbox starts from the same known state, with
  the same component versions as the real setup, so behavior seen in the sandbox predicts behavior
  in production.
- **FR-6 — Hard resource limits, configurable.** A sandbox runs under a CPU and memory ceiling, so
  that a runaway experiment on the same machine cannot starve or crash the real system. The ceiling
  is configurable per sandbox: the requester may set it in the request (FR-13), and otherwise
  defaults apply. The architect proposes the default values and the production-health threshold
  that AC-2 checks against, and the stakeholder approves them.
- **FR-7 — Unaffected normal operation.** Ordinary agent and user activity against production
  (learnings capture, knowledge-base search, storefront use) works exactly as today, with no
  reconfiguration caused by a sandbox existing.
- **FR-8 — Requester-chosen starting data.** Each sandbox request states what the sandbox starts
  with: empty structure, the repo's seed/demo data, or a copy of specific real workspaces. A copy of
  real data, once made, is subject to FR-1's isolation: changes to it never flow back on their own.
- **FR-9 — Isolation against all three harms.** Nothing done in a sandbox can (a) write to or
  delete production data, (b) change or conflict with production's structure (schema, indexes,
  constraints), or (c) degrade or interrupt production's availability through load, crashes, or
  restarts.
- **FR-10 — Risk-based use, not a blanket rule.** A sandbox is not mandatory for every change.
  Before work that touches live falkor-chat data begins, its risk/blast radius is judged case by
  case. Low-risk work may go straight to production.
- **FR-11 — Stakeholder sign-off.** The stakeholder approves both (a) the call that a given piece
  of work needs a sandbox and (b) any structural change before it is applied to production.
- **FR-12 — Check before promoting.** Before a validated sandbox change is applied to production,
  an analysis determines what has to move (structure only, structure plus data, or something
  else), decided per change.
- **FR-13 — Open to anyone; devops builds.** Any agent or contributor may request a sandbox. The
  requester states what the work needs (scope, starting data per FR-8, resource limits per FR-6,
  the FR-10 risk call).
  Creating, running, and removing sandboxes is devops's responsibility.

## Out of scope

- `cpg_*` code-property graphs: still not protected scope (carried over).
- The specific promotion tooling, and a fixed universal promotion recipe: per-change (FR-12), the
  HOW is the architect's.
- Continuous or automatic sync between a sandbox and production.
- A mandatory sandbox-for-everything policy (FR-10).
- The predecessor's "falkor-chat workspace concerns are out of scope" line, which is **reversed**
  (decision log, 2026-10-28).

## Acceptance criteria

- **AC-1:** Given a sandbox started from a copy of real workspaces, when data is deleted and
  structure is changed inside it, then production's data and structure are unchanged afterwards.
- **AC-2:** Given a sandbox running a deliberately runaway workload (or crashing), when production
  is exercised at the same time (learnings capture, knowledge-base search, a storefront request),
  then production keeps answering within the health threshold approved under FR-6. The same
  check passes at the default limits and at a requester-chosen limit.
- **AC-3:** Given a sandbox is removed, when the machine is checked afterwards, then no container,
  volume, data, or process of it remains.
- **AC-4:** Given two sandboxes created from the same request, when compared, then they start
  identical and on the same component versions as production.
- **AC-5:** Given a sandbox exists, when an agent captures a learning or a user uses the storefront
  without any special setup, then it lands in production exactly as it does today.
- **AC-6:** Given a sandbox request, then it records the requester's stated needs (scope, starting
  data, risk call) and the stakeholder's approval. Given a change ready to promote, then a
  what-has-to-move analysis exists, and any structural change carries recorded stakeholder sign-off.
- **AC-7:** Given the doc set, when this document closes, then
  `docs/requirements/kaizen-team-sandbox.md` is marked superseded by it, and the predecessor plan
  (`docs/plans/kaizen-team-sandbox.md`) is flagged to its owner (`architect`) as not carried
  forward.

## Open questions

None.

## Decision log

- 2026-10-28 — Closeout check found `docs/requirements/kaizen-team-sandbox.md` designed but never
  built, with its protected graph (`kaizen_team`) retired. Retire or re-scope? → **Re-scope.** This
  document is the successor. It reverses the predecessor's out-of-scope line excluding
  "falkor-chat's own workspace concerns", because `ws:agent-team` is itself a falkor-chat workspace.
- 2026-10-28 — What does the sandbox protect? (offered: `ws:agent-team` only / plus its server
  path / all live falkor-chat data) → **All live falkor-chat data.** Every real workspace (e.g.
  `ws:acme` storefront data), not just the agent team's, gets the same protection from development
  work. This widens the predecessor's scope beyond agent-team data. Open question 1 is closed.
- 2026-10-28 — What's driving this now? (offered: coordination-graph work / general safety net /
  something else) → **General safety net.** Isolation is wanted as standing practice, not tied to
  one upcoming project. Open question 2 is closed. Urgency is not driven by a dated project.
- 2026-10-28 — Stakeholder statement: "I want this sandbox to be containerized." Recorded
  verbatim. Whether it is a hard constraint or a preference depends on the need behind it (being
  asked next). Context for the architect: FalkorDB already runs as a Docker container
  (`falkordb-dev`), and falkor-chat already ships a `Dockerfile` and `compose.yaml`.
- 2026-10-28 — What should containerizing deliver? (multi-select: disposable / whole stack inside
  / reproducible and production-like / hard resource limits) → **All four.** Containerization
  becomes a stakeholder constraint (FR-2) with the outcomes FR-3..FR-6. FR-1 and FR-7 were drafted
  from the earlier answers.
- 2026-10-28 — What data does a fresh sandbox start with? (offered: requester chooses / seed-demo
  only / copy of real data / empty structure) → **Requester chooses** per request (FR-8). Copying
  real workspaces is allowed.
- 2026-10-28 — Which predecessor process rules still hold? (multi-select) → **All four**:
  risk-based use, stakeholder sign-off, check before promoting, and anyone asks / devops builds
  (FR-10..FR-13).
- 2026-10-28 — Readback. Numbers for the resource limits and the production-health threshold? →
  **The architect proposes them, but they are configurable / the user's choice** per sandbox
  (FR-6, FR-13, AC-2 updated). Do all three harms still apply? → **Yes, all three** (FR-9
  confirmed). No open questions remain.
- 2026-10-28 — Readback confirmed by the stakeholder → **Ready for design.** The predecessor `docs/requirements/kaizen-team-sandbox.md` is marked superseded.
