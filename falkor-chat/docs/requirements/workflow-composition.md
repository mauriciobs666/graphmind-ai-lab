# Workflow composition (sub-workflow orchestration) — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of five sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — originally six documents; two of them,
durable mutable business state and deterministic computation, were later merged into one — see
`docs/requirements/workflow-cart-and-totals.md`). The others:
`docs/requirements/workflow-catalog-lookup.md`,
`docs/requirements/workflow-cart-and-totals.md` (supersedes `workflow-business-entities.md` and
`workflow-deterministic-compute.md`), `docs/requirements/workflow-durable-profile.md`,
`docs/requirements/workflow-nl-query-generation.md`. This document is the youngest of the six —
it was opened mid-interview, once the stakeholder confirmed a longer-term intent to compose
several dedicated workflows under one orchestrating agent, rather than only ever granting one
agent step every tool directly.

## Intent
A deliberately future-phased capability: the near-term plan (see
`docs/requirements/workflow-durable-profile.md` and sibling docs) is one orchestrating `agent`
step granted many tools (catalog lookup, cart/order, compute, profile), mirroring how
`salesperson` and `triage` already work today — no new engine primitive required for that. But
the stakeholder wants to expand into other markets/domains later by composing *dedicated
workflows* (e.g. a profile-interview workflow, a catalog workflow) under a higher-level
orchestrator, rather than growing one flat agent step's tool list indefinitely. That composition
— one workflow run starting, waiting on, and resuming from a child workflow run, with something
like a call stack and its own state machine — does not exist in falkor-chat today. This document
tracks that as a real, named future capability so the intent isn't lost, without blocking or
being required by the other five capabilities.

## Problem & current state
Today a workflow run's step types (`agent`, `human`, `wait`, `decision`) all execute within one
run; there is no step type or mechanism for one workflow run to start a *separate* workflow run
as a child, wait on its outcome, and resume with its result. Composing specialized workflows —
e.g. a top-level orchestrator delegating "collect the customer's profile" to a dedicated
profile-interview workflow, or "look up the catalog" to a dedicated catalog workflow — is not
possible; the only existing way to combine capabilities inside one run is granting a single
`agent` step many tools.

## User stories
- As a workflow author, I want to compose specialized workflows as a call stack — a parent
  workflow calling a child, which can itself call a grandchild, arbitrarily deep — so domain
  logic can be organized into focused, reusable units instead of one ever-growing agent step's
  tool list.
- As a workflow author, I want the customer to experience the whole composition as one seamless
  conversation with no visible handoff, agent-identity change, or discontinuity, so composing
  behind the scenes never leaks into the customer-facing experience — "we are a cognitive
  backend."
- As a workflow author, I want every level of a composition chain to share visibility into the
  same session-scoped state (e.g. the customer's identity, their cart) without each level having
  to explicitly forward it to the next, so nesting one level deeper never means re-plumbing
  context by hand.
- As a workflow author, I want the parent to receive an explicit outcome when a child run
  finishes (not just infer what happened from a state change), so it can clearly branch on what
  the child accomplished.
- As a workflow author, I want to be notified if a child run fails or gets stuck, so a parent
  never hangs indefinitely waiting on a broken descendant.
- As someone judging whether this is safe to build, I want runaway or cyclic recursion
  structurally guarded against, so a broken workflow definition can never hang a customer's
  session forever.
- As someone planning future work, I want this capability's intent captured now, during this
  design cycle, even though it won't be built until a later phase — so it isn't lost or
  re-discovered from scratch.

## Functional requirements
- **FR-1** — A workflow run can start a child workflow run as part of its execution — the closest
  analogy is a function call: the parent invokes the child and later resumes with its result.
- **FR-2** — A child run can itself start further child runs (a grandchild, and so on) — the
  mechanism supports multiple levels of composition, a call stack, not a single fixed level.
- **FR-3** — A parent run fully pauses when it starts a child (synchronous call semantics) and
  resumes only once that child finishes. This is the required near-term behavior; asynchronous/
  non-blocking delegation is a genuine future direction beyond even this document's own phase
  (see Out of scope).
- **FR-4** — Every run in a composition chain (parent, child, grandchild, …) shares visibility
  into common session-scoped state (e.g. customer identity, cart) without needing it explicitly
  forwarded level-by-level — mirroring how nested function calls share an enclosing scope, not a
  strict explicit-arguments-only call boundary.
- **FR-5** — When a child run finishes, it returns an explicit outcome to its parent (distinct
  from any change to shared session state), so the parent can branch clearly on what happened.
- **FR-6** — If a child run fails or never completes, its parent is notified of the failure — the
  parent does not hang indefinitely waiting on a broken or stuck child.
- **FR-7** — The mechanism structurally guards against runaway or cyclic recursion (e.g. a
  workflow chain calling itself, or a cycle of workflows calling each other indefinitely), so a
  broken composition can never hang a customer's session forever.
- **FR-8** — The entire composition, across every level, is invisible to the customer — no
  visible handoff, agent-identity change, or perceptible discontinuity in the conversation.
- **FR-9** — This document's acceptance criteria describe what a *future* proof-of-concept demo
  must show when this capability is actually picked up for design/implementation. Building and
  running that demo is explicitly **not** required to satisfy this document's own gate.

## Out of scope
- Asynchronous/non-blocking delegation (a parent starting a child and continuing without waiting,
  resuming later — closer to a fire-and-forget message than a function call) — a real future
  direction the stakeholder wants eventually, phrased explicitly as "later" relative to even this
  already-deferred capability; only synchronous call/return is required when this is built.
- The specific mechanism for shared session-scoped visibility (the stakeholder floated a "context
  node" per session as one idea) — an architecture decision, not fixed here; this document
  requires the *capability* (shared visibility without hand-forwarding), not that particular
  design.
- A fixed maximum stack depth — left to the architect; this document only requires "more than one
  level" (FR-2) and that runaway recursion is guarded against (FR-7), not a specific numeric cap.
- Building and running an actual proof-of-concept demo as part of this document's own gate (FR-9)
  — deferred to whenever this capability is picked up for design.
- Blocking, or being required by, any of the other five sibling capabilities — this remains
  purely additive infrastructure for a later phase.
- Rebuilding or replacing `salesperson`.

## Acceptance criteria
- **AC-1** (FR-1, FR-2) — Given a top-level workflow delegates to a child, and that child itself
  delegates to a grandchild, when the composed run executes, then all three levels run in
  sequence and the top-level run correctly resumes once the full chain completes.
- **AC-2** (FR-3) — Given a parent has started a child run, when the child has not yet finished,
  then the parent is demonstrably paused — not proceeding, not double-executing — until the child
  completes.
- **AC-3** (FR-4) — Given a fact established at the top level (e.g. the customer's identity),
  when a nested child or grandchild run needs it, then it is visible there without the parent
  having explicitly forwarded it as an argument.
- **AC-4** (FR-5, FR-6) — Given a child run completes, when the parent resumes, then it receives
  an explicit outcome; given a child run instead fails or gets stuck, when that happens, then the
  parent is notified rather than hanging indefinitely.
- **AC-5** (FR-7) — Given a workflow definition that would cause unbounded or cyclic recursion,
  when it is run, then the mechanism halts or rejects it before it can hang a customer's session
  indefinitely.
- **AC-6** (FR-8) — Given a customer chatting through a multi-level composed workflow, when
  observed from the customer's side, then no visible sign of a handoff is detectable at any point.

## Open questions
- Whether this is genuinely required, or whether "one agent, many tools" scales far enough in
  practice that composition is never needed — an open question the stakeholder wants tracked,
  not necessarily built, in this phase.

## Decision log
2026-08-22 — Opened mid-interview (during `workflow-durable-profile.md`'s interview) once the
stakeholder confirmed both: (a) the near-term approach is one orchestrating agent step with many
tools (no new primitive needed), and (b) a longer-term intent to compose dedicated sub-workflows
exists and should be tracked now, even though it is explicitly deferred to a future phase and
does not block any of the other five capabilities. Not yet interviewed in depth.
2026-08-22 — Interviewed sixth and last. Rigor level: stakeholder chose full treatment (testable
FRs/ACs) over a light-touch capture, despite the capability being deferred — so it's genuinely
ready for design the moment it's picked up, no re-interview needed.
2026-08-22 — Customer-visibility: the whole composition must be seamless — "we are a cognitive
backend." No visible handoff, ever.
2026-08-22 — Composition depth: a true multi-level call stack ("like a programming language") —
parent→child→grandchild, not limited to one level.
2026-08-22 — Call semantics: synchronous (parent blocks until child returns) is the near-term
requirement. Asynchronous/non-blocking delegation (compared by the stakeholder to a
`SendMessage`-style fire-and-forget pattern) is a genuine future direction, phrased as later than
even this deferred capability — out of scope here.
2026-08-22 — Data crossing the call boundary: not strict explicit-args-in/explicit-result-out only
— every level needs shared visibility into common session-scoped state (customer identity, cart,
etc.) without hand-forwarding it level by level. Stakeholder floated a "context node" per session
as a possible mechanism; captured as an architect-facing idea, not a fixed requirement — the
requirement is the shared-visibility capability itself (FR-4).
2026-08-22 — Despite the session-state visibility above, the parent still needs an **explicit
outcome** from a finished child (not just an inferred state change) to branch on, and must be
**notified on child failure** rather than hang.
2026-08-22 — Safety: runaway/cyclic recursion must be structurally guarded against — stated as a
requirement (FR-7), not left silent.
2026-08-22 — Demo/proof-of-concept: acceptance criteria describe what a *future* demo must show;
building one is explicitly not required to satisfy this document's own gate (FR-9) — matches the
"full treatment now, build later" rigor choice above.
2026-08-22 — Readback confirmed with no changes, but stakeholder deliberately kept `Status:` at
**Interviewing** rather than flipping to Ready for design — this document is complete and
consistent, but the stakeholder wants to consciously revisit it before it's handed off, since the
open question above (whether it's needed at all) is still live. Not a gap in the interview; a
deliberate hold.
