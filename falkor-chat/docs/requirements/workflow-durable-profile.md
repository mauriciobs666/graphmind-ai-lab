# Durable user-profile data for workflows — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of five sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — originally six documents; two of them,
durable mutable business state and deterministic computation, were later merged into one — see
`docs/requirements/workflow-cart-and-totals.md`). The others:
`docs/requirements/workflow-catalog-lookup.md`,
`docs/requirements/workflow-cart-and-totals.md` (supersedes `workflow-business-entities.md` and
`workflow-deterministic-compute.md`), `docs/requirements/workflow-nl-query-generation.md`
(added mid-interview, spun off from `workflow-catalog-lookup.md`),
`docs/requirements/workflow-composition.md` (opened mid-*this* document's own interview, still
`Interviewing`, on hold at the stakeholder's request). Read this one for durable, cross-conversation
profile data about the person a workflow is talking to (e.g. a name, a delivery address)
specifically — not for cart/order-shaped transactional state.

## Intent
Close a structural gap ahead of any specific consumer — same proactive-infrastructure framing as
the other sibling capabilities. Proven as part of the **single combined "salesperson" demo
agent** that also covers catalog lookup, cart/order, and deterministic totals
(`workflow-catalog-lookup.md`, `workflow-cart-and-totals.md`): this capability adds durable
name/address capture to that same agent, rather than shipping as its own separate demo. Building
one complete agent now and
splitting it into composed sub-workflows later is an explicit, deliberate phasing decision —
tracked as its own future capability in `workflow-composition.md`, not required here.

## Problem & current state
Today, information a workflow collects about the person it's talking to (e.g. a name, an
address) has nowhere durable to live except a run's `ctx` — flat, run-scoped, and discarded once
the run ends. The harder version of this problem — whether `identity` (the global, read-mostly
user/auth graph) should start accepting writes, or whether per-workspace snapshots are the right
answer instead — was flagged by both the architect and graph-dba as a genuinely unresolved
design question. This document deliberately **does not resolve it**: it reuses the same shortcut
`workflow-business-entities.md` already took for the cart — scope persistence to *within one
workspace* — deferring the cross-workspace/identity-write question to a later refactor once an
actual need for it materializes.

## User stories
- As a workflow author, I want a customer's name and delivery address to be durably recorded the
  first time they're given, so the same customer isn't asked again in a later, separate
  conversation within the same workspace.
- As a workflow author, I want profile capture to be one capability the single orchestrating
  "salesperson" agent can use, alongside catalog lookup and cart/order, so a complete, coherent
  demo can be built and judged end-to-end rather than as disconnected fragments.
- As someone judging whether this capability works, I want to see: the agent asks once, the
  answer persists, and it does not ask again in a new conversation with the same customer in the
  same workspace — proven inside the combined demo agent, not a separate toy example.

## Functional requirements
- **FR-1** — A workflow can durably write and later read a customer's name and delivery address,
  scoped to *one workspace* (the same workspace-local shortcut `workflow-business-entities.md`
  uses for the cart) — not scoped to a single run or thread.
- **FR-2** — Once captured, a customer's name/address are available to any later conversation
  with that same customer in the same workspace, without asking again.
- **FR-3** — If a customer provides updated name/address information later, the stored profile is
  updated, not frozen after the first write.
- **FR-4** — This capability is proven as part of the single combined "salesperson" demo agent
  shared with `workflow-catalog-lookup.md` and `workflow-cart-and-totals.md` — no separate
  standalone demo workflow is required.

## Out of scope
- Cross-workspace profile persistence — deferred to a future refactor once an actual need
  materializes, same deferral pattern as the cart's own workspace scoping.
- Any change to how `identity` works today (e.g. accepting direct writes) — sidestepped
  entirely by the workspace-scoped shortcut, not resolved here.
- Automatically attaching a saved profile to a placed order — a real future direction the
  stakeholder wants to build toward, but explicitly not required to ship this capability.
- An extensible/open-ended profile schema ("collect arbitrary new kinds of facts about a person
  later without touching this capability again") — kept to the simple fixed fields (name,
  delivery address) for this phase.
- A standalone demo workflow separate from the combined single-agent demo.

## Acceptance criteria
- **AC-1** (FR-1, FR-2) — Given a customer provides their name and delivery address during one
  conversation, when they start a new, separate conversation in the same workspace, then the
  agent already has their name/address and does not ask again.
- **AC-2** (FR-3) — Given a customer already has a stored name/address, when they provide an
  updated address in a later conversation, then the stored profile reflects the update.
- **AC-3** (FR-4) — The demo proving this capability is the same combined "salesperson" agent
  used to prove catalog lookup, cart/order, and deterministic totals — not a separate workflow.

## Open questions
None outstanding — pending stakeholder confirmation at readback.

## Decision log
2026-08-22 — Split out as one of the sibling capabilities; interviewed fourth (after
deterministic-compute).
2026-08-22 — Proactive infrastructure again, no concrete consumer.
2026-08-22 — Stakeholder's original intent was broader than "save name and address": a dedicated
interview-style workflow recording data into a new "sub-ontology." Narrowed, on discussion, to
the simple fixed fields (name, delivery address) for this phase — an extensible/open-ended
profile schema is out of scope here.
2026-08-22 — Surfaced a larger architectural direction: the stakeholder wants to eventually
compose multiple dedicated workflows (profile interview, catalog, cart/order) under a
higher-level orchestrating agent, to support expanding into other markets later. Near-term
decision: build one complete, richly-tooled single agent now (matching `salesperson`'s and
`triage`'s existing pattern — one `agent` step, many tools); defer actual workflow composition
(a call-stack/state-machine for one run to invoke and resume from a child run) to a new,
explicitly future-phased sibling capability, `workflow-composition.md`, opened mid-interview.
2026-08-22 — Demo scoping revised mid-interview: originally planned as a standalone
interview-style workflow; once the single-agent-with-many-tools direction was settled, folded
into the same combined demo agent as catalog-lookup/business-entities/deterministic-compute
instead of a separate workflow.
2026-08-22 — Connecting saved profile data to placed orders is a genuine future direction the
stakeholder wants, but is explicitly not required to ship this capability.
2026-08-22 — Profile persistence reuses the cart's workspace-scoping shortcut (not
cross-workspace, not writing to `identity`) — the harder identity-write-path question flagged by
the architect and graph-dba is deliberately deferred, to be revisited "when needed," not resolved
now.
2026-08-22 — Stakeholder confirmed the readback with no changes; flipped to Ready for design.
2026-08-22 — `workflow-business-entities.md` and `workflow-deterministic-compute.md` merged into
`docs/requirements/workflow-cart-and-totals.md` (stakeholder-requested, per a cross-document
coherence review). Sibling references and FR-4 above updated to point at the merged document; no
content of this document's own requirements changed.
