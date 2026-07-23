# LLM-Native Workflows — Feature Requirements
> Status: Ready for design · Last updated: 2026-07-09

## Intent
The stakeholder wants falkor-chat workflows to be **LLM-native** rather than driven by rigid,
coded logic. The vision: each workflow node is essentially a **system prompt the model executes
with full reasoning**, and the transitions between nodes are **fuzzy, natural-language conditions
the LLM judges** — not formulas in an expression language. Workflows should be **authored in plain
natural language**, so the value is in expressing intent and semantic judgment that a formula
(`priority == "high"`) could never capture.

This surfaced from the M3 open question (DESIGN §13) "workflow guard expression language — expr lib
vs. minimal DSL," which the stakeholder reframed: guards (and nodes) should be reasoned about by an
LLM, not evaluated by an expression library.

## Problem & current state
- The workflow engine's **run/step-run executor** (K-022) is not yet built. The definition model and
  snapshot materialization (Slice 1) shipped.
- The current design (DESIGN §6) models a `Step.type` ∈ {prompt|tool|decision|human|message|wait}
  and a `TRANSITION.guard` expression evaluated against run `ctx`. Guards are stored today as
  **opaque strings, never yet evaluated** — the executor is the first thing that must interpret them,
  which is exactly the decision point being reframed here.
- An expression-library / DSL approach cannot express the fuzzy, semantic conditions the stakeholder
  wants ("customer sounds frustrated", "answer is complete enough to send"), nor be authored in
  plain language by non-engineers.

## User stories
- As a **workflow author**, I want to write each step as a plain-language system prompt, so that I can
  define agent behavior without learning a node-type taxonomy or code.
- As a **workflow author**, I want to write transition conditions in natural language, so that I can
  branch on fuzzy/semantic judgments a formula can't express.
- As a **workflow author**, I want to choose **per node/edge** whether it is LLM-native or
  deterministic, so that I can use fuzzy reasoning where it helps and predictable logic where the
  process demands it — within a single workflow.
- _(more to elicit)_

## Functional requirements
- **FR-1** — A workflow author can define a node as a plain-language system prompt that the model
  executes with full reasoning (an "LLM-native" node).
- **FR-2** — A workflow author can define a transition condition ("guard") in natural language,
  evaluated by an LLM against the run's context to decide whether the transition fires.
- **FR-3** — For each node and each transition, the author can choose whether it is **LLM-native**
  or **deterministic** (typed step / formula guard). A single workflow may mix both.
- **FR-4** — Tracing is **configurable per run instance**. Serviceability is a **primary value**: when
  tracing is enabled ("debug instance"), the run records **all aspects** of execution into its audit
  trail — node reasoning/rationale, transition judgments (with the "why"), every tool/MCP call (inputs
  and results), GraphRAG retrievals, the LLM prompts and responses, and step timing/status — so a run
  can be fully reconstructed and diagnosed after the fact. When disabled, the run executes leanly
  without recording this detail.
- **FR-5** — An LLM-native node can, as part of its execution:
  - **a. Post a message** into the chat thread (workflow is visible to participants as it runs).
  - **b. Retrieve from the graph** via GraphRAG (the M2 vector + traversal search) to ground its
    reasoning.
  - **c. Call a tool**, including tools exposed over **MCP** (Model Context Protocol).
  - **d. Hand off to / wait for a human**, pausing the run and resuming on the person's input.
  > _Context for architect (not a design decision here):_ falkor-chat today is an MCP **server**
  > (§15). FR-5c requires a workflow node to act as an MCP **client** consuming external MCP tools —
  > a new capability/direction, not the existing front door.
- **FR-6** — Each LLM-native node is an **autonomous agent operating within author-set bounds**: the
  author gives the node a system prompt **and a scoped toolset/permissions** (which tools it may use,
  what it may/may not do); the model then decides freely — which actions to take (retrieve, call
  which tool, post, hand off) and in what order — **within that fence**. Not every node gets the full
  toolset; boundaries are set per node.
- **FR-7** — A workflow run is **triggered by a chat `@mention`** of the agent in a thread; the
  triggering message anchors the run (`TRIGGERED_BY`, per DESIGN §6.2).

## Out of scope
- The deterministic model is **not** removed — it coexists (see decision log). This feature adds the
  LLM-native option alongside it.
- **Business-process proof flow is deferred.** This first cut delivers only the conversational proof
  flow (triage). The deterministic business-process proof the M3 DoD also wants is left for later.
- **Human approval / hand-off in the flow** — the capability exists (FR-5d) but is not used by the
  first proof flow.
- **No cost/latency budget or step cap is specified** for the first cut (see open questions for the
  runaway-loop safety note).
- Creating the seed/test conversation data, fixtures, and the automated tests themselves — that is
  downstream implementation/QA work (coder / qa-engineer), not part of this requirements doc. This
  doc only records the *need* for seeded data as a test precondition (see Acceptance criteria).

## Proof flow (first target)
A **conversational triage** workflow, triggered by an `@mention`:
1. **Intake agent** — asks the user clarifying questions in the thread until it judges it has enough
   information (an LLM-native transition guard: "enough info to proceed?").
2. **Research agent** — retrieves relevant context via GraphRAG and produces findings.
3. **Answer** — posts a reply to the thread grounded in the research findings. **No human approval**
   in this first cut — the run answers directly from what the research node found.

This doubles as the M3 milestone's required conversational proof flow. (A business-process proof flow
is still to be chosen — see open questions.) Human hand-off remains a platform *capability* (FR-5d)
but is **not exercised** by this proof flow.

## Acceptance criteria
- **AC-1 (trigger)** — Given a defined triage workflow, When a user `@mentions` the agent in a
  thread, Then a `WorkflowRun` starts and is linked to the triggering message.
- **AC-2 (intake loop + fuzzy guard)** — Given the intake agent lacks enough information, When it
  runs, Then it posts clarifying questions to the thread and does **not** advance; When it judges it
  has enough information, Then the run transitions to the research node.
- **AC-3 (retrieval)** — Given seeded conversation data exists in the workspace *(test
  precondition — data created by the implementer/QA, not this doc)*, When the research agent runs,
  Then it retrieves relevant context via GraphRAG and produces findings grounded in that data.
- **AC-4 (answer / done)** — Given the research agent has produced findings, When the answer node
  runs, Then a reply grounded in those findings is posted to the thread, and the run is marked done.
  (No human approval in this first cut.)
- **AC-5 (tracing)** — Given a run started as a **debug instance** (FR-4), When any LLM-native node
  or transition executes, Then its rationale is recorded in the run's audit trail; a non-debug run
  records no rationale.
- **AC-6 (per-node bounds)** — Given a node granted a scoped toolset (FR-6), When the agent runs,
  Then it can only use the tools/permissions granted to that node.

## Open questions
_None blocking the first cut. Parked items (for the architect / a later slice):_
- **Runaway-loop safety** — no step cap / termination budget is specified. Since every node and
  branch is an LLM call, the architect should consider a runaway guard (max steps or similar). Low
  priority; serviceability tracing (FR-4) aids diagnosis but doesn't prevent loops.
- **Business-process proof flow** (deferred, out of scope here) — which deterministic flow proves the
  process side for the full M3 DoD.

## Decision log
2026-07-09 — Guard evaluation: expression library vs. LLM? → Stakeholder: **LLM**, not an
expression library.
2026-07-09 — What's driving the LLM preference? → **Both** fuzzy/semantic conditions *and* plain
natural-language authoring; stakeholder envisions **each workflow node as a system prompt with full
reasoning**.
2026-07-09 — Replace or coexist with the deterministic model? → **Coexist**, and the author picks
**per node/edge** whether it is LLM-native or deterministic (a single workflow may mix both).
Deterministic model stays for business processes needing predictable branching + audit.
2026-07-09 — Is recording LLM decision rationale required? → **Configurable.** Stakeholder wants to
bring up **debug instances with tracing enabled** that record the rationale; the default (non-debug)
run does not. (FR-4.)
2026-07-09 — What can an LLM-native node do beyond reasoning? → **All of:** post messages, retrieve
via GraphRAG, call tools **including MCP tools**, and hand off to / wait for a human. (FR-5.)
2026-07-09 — Node autonomy: LLM-decides vs author-wires? → **Every node is an agent**, but the
author grants each node a **scoped toolset/permissions** and the agent decides freely *within that
fence* (the "mix" option, corrected from an earlier full-autonomy reading). Boundaries are per node.
(FR-6.)
2026-07-09 — Trigger? → **Chat `@mention`** of the agent. (FR-7.) First proof flow = a conversational
triage: intake agent → research agent → post answer.
2026-07-09 — Human approval in the proof flow? → **No** — too much complexity for now. The run posts
an answer directly from the research node's findings. Human hand-off stays a capability (FR-5d) but
isn't used in the first proof flow. (AC-4 simplified; approval-mechanics OQ dropped.)
2026-07-09 — Include a business-process proof flow in this first cut? → **No, leave for later.** First
cut is conversational-only. (Out of scope.)
2026-07-09 — Cost/latency limits? → **No hard budget.** Stakeholder instead values **serviceability**
highly: **trace all aspects when tracing is enabled** (FR-4 broadened). Runaway-loop guard left as a
parked note for the architect.
2026-07-09 — Readback confirmed by stakeholder → **Status: Ready for design.**
