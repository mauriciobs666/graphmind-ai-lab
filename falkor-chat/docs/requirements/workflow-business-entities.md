# Durable mutable business entities for workflows — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of six sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-deterministic-compute.md`,
`docs/requirements/workflow-durable-profile.md`, `docs/requirements/workflow-nl-query-generation.md`
(added mid-interview, spun off from `workflow-catalog-lookup.md`),
`docs/requirements/workflow-composition.md` (opened later still, mid-interview of
`workflow-durable-profile.md`). Each is independently motivated and independently shippable —
read this one for cart/order-shaped mutable state specifically.

## Intent
Close a structural gap ahead of any specific consumer: falkor-chat workflows today have no way
to represent durable, mutable, queryable business state that outlives a single run — the kind of
thing a shopping cart or a placed order is. This is proactive infrastructure work, proven via a
runnable demo — built on the consumer-electronics catalog already scoped in
`workflow-catalog-lookup.md` — rather than left as an untested primitive, following the same
pattern falkor-chat already uses for other engine capabilities (`triage`, `access-request`).

## Problem & current state
Everything a workflow carries today lives in `ctx` — a flat, run-scoped, serialized JSON blob
that cannot be filtered or queried (`falkor-chat/AGENTS.md` rule 8) and is discarded once the run
ends. There is no way to represent a cart that gets edited over several turns, needs an accurate
running total, and must still be there if the customer closes the conversation and comes back —
nor a placed order that needs to durably outlive the run that created it, be queryable
afterward, and track its own lifecycle (e.g. shipped, delivered, cancelled) via further explicit
steps over time, exactly as `access-request` already demonstrates for a *decision* workflow but
nothing today demonstrates for a *stateful business record*.

## User stories
- As a workflow author, I want a customer's cart to survive across separate conversations within
  the same workspace, so returning to chat later doesn't lose what they'd already added.
- As a workflow author, I want to add/remove/adjust items in a cart and see an accurate running
  total, so a shopping-style interaction can be built without hand-rolling ephemeral state (the
  way `salesperson` currently does, in process memory, lost on restart).
- As a workflow author, I want placing an order to create a durable, immutable record of exactly
  what was ordered (items, quantities, prices at that moment), so it can be queried or audited
  later even if the catalog's prices change afterward.
- As an operator, I want to advance a placed order through its lifecycle (fulfilled/shipped,
  then delivered — or cancelled before fulfillment) via explicit steps, mirroring how
  `access-request` already uses human/wait steps for a decision, so order fulfillment can be
  demonstrated end-to-end, not just "placed" as a single terminal event.
- As someone judging whether this capability actually works, I want a runnable proof-of-concept
  — a demo cart-and-order workflow built on the consumer-electronics catalog — so I can verify it
  end-to-end rather than trust it in the abstract.

## Functional requirements
- **FR-1** — A workflow can add an item+quantity to a cart, remove/decrease an item's quantity,
  view the cart's current contents and running total, and clear it entirely.
- **FR-2** — A cart persists across separate conversations/threads within the same workspace —
  anchored to the customer within that workspace, not to any one thread. Starting a new
  conversation with the same identity in the same workspace shows the same cart. (Persisting a
  cart *across different workspaces* is out of scope — see below.)
- **FR-3** — A cart's displayed prices/total reflect the catalog's *current* prices as items sit
  in the cart (it is not yet finalized) — this is the "live" phase, distinct from FR-4's
  point-in-time snapshot at placement.
- **FR-4** — Placing an order converts a cart into a durable, immutable Order record: a snapshot
  of each line's name/price/quantity as of that moment (so a later catalog price change never
  retroactively alters a past order), and clears the cart it was created from.
- **FR-5** — A placed order has a lifecycle: `placed → fulfilled/shipped → delivered`, or
  `placed → cancelled` (cancellation only possible before fulfillment). Every transition is
  driven by an explicit step (an operator/human input or an external signal) — mirroring
  `access-request`'s human/wait pattern. No transition happens automatically or on a timer.
- **FR-6** — An order's current lifecycle status is queryable at any time after it's placed.
- **FR-7** — Cart-building and checkout are proven inside the same single combined "salesperson"
  demo agent as `workflow-catalog-lookup.md` and `workflow-deterministic-compute.md` (one
  orchestrating `agent` step, many tools — see the decision log). Order *fulfillment* — the
  human/operator-driven lifecycle in FR-5 — is a separate, process-kind workflow, mirroring
  `access-request`'s own conversation/process split (an inherent distinction in falkor-chat's
  engine, not a new one introduced here): materialized and verifiable the same way
  `triage`/`access-request` are today.

## Out of scope
- Cart persistence across *different* workspaces — that would require resolving the
  identity-write-path question that is still open and un-interviewed in
  `workflow-durable-profile.md`; deliberately deferred rather than blocking this capability on
  that one.
- Payment processing, returns, and refunds — the lifecycle as scoped is
  placed/fulfilled-shipped/delivered/cancelled only.
- Handling concurrent edits to the same cart (e.g. two open tabs editing it at once).
- A runtime management/admin API for carts or orders beyond what the demo needs to prove the
  capability — no general CRUD surface is required.
- Automatic or timer-driven lifecycle transitions — consistent with falkor-chat's existing
  no-timers design, every transition needs an explicit input.
- The mechanism behind computing totals/snapshots (e.g. how arithmetic is actually performed
  inside a step) — that is `workflow-deterministic-compute.md`'s territory; this document only
  requires that totals and snapshots be *correct*, not how that's achieved.
- Rebuilding or replacing `salesperson`.

## Acceptance criteria
- **AC-1** (FR-1) — Given an empty cart, when a user adds an item with a quantity, then the cart
  shows that item, its quantity, and the correct subtotal/total.
- **AC-2** (FR-1) — Given a cart with items, when a user removes an item or decreases its
  quantity, then the cart reflects the change and an updated total; when a user clears the cart,
  it is empty afterward.
- **AC-3** (FR-2) — Given a cart with items added during one conversation, when the same customer
  starts a different conversation in the same workspace, then the same cart (same items) is
  visible there.
- **AC-4** (FR-4) — Given a cart with items, when the customer places an order, then a durable
  Order record is created capturing the items/prices/quantities as of that moment, and the cart
  is cleared.
- **AC-5** (FR-4) — Given an order has been placed, when the catalog's price for one of its items
  changes afterward, then the order's recorded price is unaffected — it still shows the price at
  placement time.
- **AC-6** (FR-5, FR-6) — Given a placed order, when an operator advances it through
  fulfilled/shipped and then delivered via explicit steps, then the order's queryable status
  reflects each stage in turn, and it never advances on its own.
- **AC-7** (FR-5) — Given a placed order that has not yet been fulfilled, when an operator
  cancels it, then its status becomes cancelled and no further fulfillment transition is
  possible.
- **AC-8** (FR-7) — The demo workflow(s) and the catalog data they depend on can be seeded and
  verified the same way `triage`/`access-request` are today (a seed script, plus a verification
  check).

## Open questions
None outstanding — pending stakeholder confirmation at readback.

## Decision log
2026-08-22 — Split out as one of five sibling capabilities; interviewed second (after
catalog-lookup).
2026-08-22 — Proactive infrastructure again, no concrete consumer; "done" means a runnable
proof-of-concept, reusing the consumer-electronics catalog from `workflow-catalog-lookup.md`
rather than a separate demo domain.
2026-08-22 — Cart must survive across conversations/threads, which raised a real dependency risk
on the still-open identity-anchoring question in `workflow-durable-profile.md`. Resolved by
scoping the cart to *within one workspace* (anchored to the workspace-local customer identity,
not the thread) — this capability does not depend on `workflow-durable-profile.md` being
resolved first. Cross-workspace cart persistence is explicitly out of scope.
2026-08-22 — Order lifecycle: "full life-cycle" means `placed → fulfilled/shipped → delivered`,
with a `cancelled` branch possible only before fulfillment — no payment/returns/refunds states.
2026-08-22 — Order lifecycle transitions are driven by explicit human/operator or external-signal
steps, mirroring `access-request`'s human/wait pattern — no automatic or timer-driven
transitions, consistent with falkor-chat's existing no-timers design.
2026-08-22 — Stakeholder confirmed the readback with no changes; flipped to Ready for design.
2026-08-22 — Retroactive clarification (surfaced during `workflow-durable-profile.md`'s
interview): cart-building/checkout is proven inside one single combined "salesperson" demo
agent shared with `workflow-catalog-lookup.md`, `workflow-deterministic-compute.md`, and
`workflow-durable-profile.md` — not a demo of its own. Order fulfillment (FR-5) remains a
separate process-kind workflow, since that split is inherent to falkor-chat's engine (conversation
vs. process workflows), not a new decision. FR-7 updated accordingly; no other content changed.
