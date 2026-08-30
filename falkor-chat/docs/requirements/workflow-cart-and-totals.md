# Cart, orders, and deterministic totals — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** — (M<n> TBD) ·
> **Supersedes:** `docs/requirements/workflow-business-entities.md`, `docs/requirements/workflow-deterministic-compute.md`

This is one of five sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — originally six; two of them,
durable mutable business state and deterministic computation, are merged into this single
document — see decision log below). The others: `docs/requirements/workflow-catalog-lookup.md`,
`docs/requirements/workflow-durable-profile.md`, `docs/requirements/workflow-nl-query-generation.md`,
`docs/requirements/workflow-composition.md` (still `Interviewing`, on hold at the stakeholder's
request). Read this one for durable, mutable cart/order state **and** the exact, non-AI
computation its totals and snapshots depend on — the two were interviewed as separate documents
but are tightly enough coupled (deterministic-compute never had an acceptance bar independent of
this capability's own demo) that keeping them separate obscured the complete picture.

## Intent
Close a structural gap ahead of any specific consumer: falkor-chat workflows today have no way to
represent durable, mutable, queryable business state that outlives a single run — the kind of
thing a shopping cart or a placed order is — nor a way to compute an exact number (a running
total) as part of a run without either hand-rolling arithmetic or routing it through an LLM call.
This is proactive infrastructure work, proven via a runnable demo — built on the
consumer-electronics catalog already scoped in `workflow-catalog-lookup.md` — rather than left as
an untested primitive, following the same pattern falkor-chat already uses for other engine
capabilities (`triage`, `access-request`). The computation requirement exists specifically
*because* the cart/order capability needs its totals and snapshots to be exact, not
LLM-approximated — that dependency is why the two were merged into one document.

## Problem & current state
Everything a workflow carries today lives in `ctx` — a flat, run-scoped, serialized JSON blob
that cannot be filtered or queried (`falkor-chat/AGENTS.md` rule 8) and is discarded once the run
ends. There is no way to represent a cart that gets edited over several turns, needs an accurate
running total, and must still be there if the customer closes the conversation and comes back —
nor a placed order that needs to durably outlive the run that created it, be queryable
afterward, and track its own lifecycle (e.g. shipped, delivered, cancelled) via further explicit
steps over time, exactly as `access-request` already demonstrates for a *decision* workflow but
nothing today demonstrates for a *stateful business record*.

Separately, today's workflow step types are `agent` (an LLM turn), `human`/`wait` (park for a
reply), and `decision` (a pure branch — its `cmp` guard only *compares* two values, it never
*computes* a new one). There is no primitive for exact, deterministic computation as part of a
run — today it would have to be routed through an LLM `agent` turn calling a tool, which still
involves invoking the model even if the arithmetic itself is deterministic. A general
expression-evaluation guard (`expr`) already exists as a documented, unimplemented
`NotImplementedError` in the guard language — a related, broader idea (arbitrary expression
evaluation) that was apparently attempted and abandoned; this capability is deliberately narrower
(fixed arithmetic on line items), not a revival of general `expr` evaluation. Cart/order totals
and order-placement snapshots are exactly where this gap surfaces first and most concretely.

## User stories
- As a workflow author, I want a customer's cart to survive across separate conversations within
  the same workspace, so returning to chat later doesn't lose what they'd already added.
- As a workflow author, I want to add/remove/adjust items in a cart and see an accurate running
  total, so a shopping-style interaction can be built without hand-rolling ephemeral state (the
  way `salesperson` currently does, in process memory, lost on restart).
- As a workflow author, I want that running total to be an exact, non-LLM-approximated number —
  the same inputs must always produce the same output — so a cart or order's total can be trusted.
- As a workflow author, I want that computation to happen without requiring an LLM call at all,
  so it's fast, free of model cost, and never varies between runs given the same inputs.
- As a workflow author, I want placing an order to create a durable, immutable record of exactly
  what was ordered (items, quantities, prices at that moment), so it can be queried or audited
  later even if the catalog's prices change afterward.
- As an operator, I want to advance a placed order through its lifecycle (fulfilled/shipped, then
  delivered — or cancelled before fulfillment) via explicit steps, mirroring how `access-request`
  already uses human/wait steps for a decision, so order fulfillment can be demonstrated
  end-to-end, not just "placed" as a single terminal event.
- As someone judging whether this capability actually works, I want a runnable proof-of-concept —
  a demo cart-and-order workflow, with its totals and snapshots produced by the deterministic
  mechanism — built on the consumer-electronics catalog, so I can verify it end-to-end rather than
  trust it in the abstract.

## Functional requirements
- **FR-1** — A workflow can add an item+quantity to a cart, remove/decrease an item's quantity,
  view the cart's current contents and running total, and clear it entirely.
- **FR-2** — A cart persists across separate conversations/threads within the same workspace —
  anchored to the customer within that workspace, not to any one thread. Starting a new
  conversation with the same identity in the same workspace shows the same cart. (Persisting a
  cart *across different workspaces* is out of scope — see below.)
- **FR-3** — A cart's displayed prices/total reflect the catalog's *current* prices as items sit
  in the cart (it is not yet finalized) — this is the "live" phase, distinct from FR-5's
  point-in-time snapshot at placement.
- **FR-4** — A workflow can compute a running total from a set of line items (price × quantity
  per line, summed across all lines) as part of a run. The same set of inputs always produces the
  same, exact output — no variance run to run.
- **FR-5** — Placing an order converts a cart into a durable, immutable Order record: a snapshot
  of each line's name/price/quantity as of that moment (computed via FR-4's mechanism, so a later
  catalog price change never retroactively alters a past order), and clears the cart it was
  created from.
- **FR-6** — A placed order has a lifecycle: `placed → fulfilled/shipped → delivered`, or
  `placed → cancelled` (cancellation only possible before fulfillment). Every transition is
  driven by an explicit step (an operator/human input or an external signal) — mirroring
  `access-request`'s human/wait pattern. No transition happens automatically or on a timer.
- **FR-7** — An order's current lifecycle status is queryable at any time after it's placed.
- **FR-8** — Performing FR-4's computation does not require an LLM/model call — it must not cost
  a model invocation purely to add up numbers. (This requirement bears directly on the
  step-type-vs-tool design fork the architect flagged during the earlier reflection on this
  effort — a tool dispatched from inside an `agent` step still involves an LLM turn to decide to
  call it, which this FR rules out as the *only* way to satisfy the requirement. This document
  does not choose the mechanism; it does constrain it.)
- **FR-9** — Cart-building and checkout — including the line-item totals and order snapshots
  produced by FR-4/FR-8's mechanism — are proven inside the same single combined "salesperson"
  demo agent as `workflow-catalog-lookup.md` and `workflow-durable-profile.md` (one orchestrating
  `agent` step, many tools). Order *fulfillment* — the human/operator-driven lifecycle in FR-6 —
  is a separate, process-kind workflow, mirroring `access-request`'s own conversation/process
  split (an inherent distinction in falkor-chat's engine, not a new one introduced here):
  materialized and verifiable the same way `triage`/`access-request` are today.

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
- Anything beyond price × quantity summation in the computation itself — no discounts, tax
  percentages, rounding-mode rules, or other arithmetic. A future need can extend this later.
- Aggregation across multiple orders/carts (e.g. "total sales this week") — this is per-run
  computation only, not a reporting/analytics capability.
- A general expression-evaluation language (reviving `expr`-style arbitrary evaluation) — this
  capability is a fixed, narrow arithmetic operation, not a programmable formula engine.
- The specific mechanism behind the computation (a new step type vs. a tool vs. something else)
  — left entirely to the architect; FR-8 constrains it without choosing it.
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
- **AC-4** (FR-4) — Given a set of line items with known prices and quantities, when the total is
  computed, then it equals the exact hand-calculated sum, and repeating the computation with the
  same inputs yields the identical result every time.
- **AC-5** (FR-5) — Given a cart with items, when the customer places an order, then a durable
  Order record is created capturing the items/prices/quantities as of that moment (via FR-4's
  mechanism), and the cart is cleared.
- **AC-6** (FR-5) — Given an order has been placed, when the catalog's price for one of its items
  changes afterward, then the order's recorded price is unaffected — it still shows the price at
  placement time.
- **AC-7** (FR-6, FR-7) — Given a placed order, when an operator advances it through
  fulfilled/shipped and then delivered via explicit steps, then the order's queryable status
  reflects each stage in turn, and it never advances on its own.
- **AC-8** (FR-6) — Given a placed order that has not yet been fulfilled, when an operator
  cancels it, then its status becomes cancelled and no further fulfillment transition is
  possible.
- **AC-9** (FR-8) — Given the computation runs as part of a workflow, when it executes, then no
  LLM/model call is made solely to perform it (verifiable via the run's trace or an equivalent
  check the architect/QA define).
- **AC-10** (FR-9) — The demo workflow(s) and the catalog data they depend on can be seeded and
  verified the same way `triage`/`access-request` are today (a seed script, plus a verification
  check).

## Open questions
None outstanding — pending stakeholder confirmation at readback.

## Decision log
2026-08-22 — Split out as two of six sibling capabilities (`workflow-business-entities.md`
interviewed second, `workflow-deterministic-compute.md` interviewed third); each independently
motivated, proactive infrastructure, no concrete consumer.
2026-08-22 — Cart must survive across conversations/threads, which raised a real dependency risk
on the still-open identity-anchoring question in `workflow-durable-profile.md`. Resolved by
scoping the cart to *within one workspace* (anchored to the workspace-local customer identity,
not the thread) — this capability does not depend on `workflow-durable-profile.md` being
resolved first. Cross-workspace cart persistence is explicitly out of scope.
2026-08-22 — Order lifecycle: "full life-cycle" means `placed → fulfilled/shipped → delivered`,
with a `cancelled` branch possible only before fulfillment — no payment/returns/refunds states.
Transitions are driven by explicit human/operator or external-signal steps, mirroring
`access-request`'s human/wait pattern — no automatic or timer-driven transitions, consistent with
falkor-chat's existing no-timers design.
2026-08-22 — Deterministic-compute scope kept deliberately simple for this pass: plain price ×
quantity summation only, no discounts/tax/rounding — can extend later if a real need arises.
"Deterministic" means both things: exact/repeatable correctness, *and* not requiring an LLM call
just to compute it (a cost/latency/reliability concern, not just a correctness one) — the latter
directly bears on, without resolving, the step-type-vs-tool fork the architect flagged earlier in
this effort.
2026-08-22 — Both documents confirmed at readback with no changes; each independently flipped to
Ready for design.
2026-08-22 — Retroactive clarification (surfaced during `workflow-durable-profile.md`'s
interview): both demos converged onto the single combined "salesperson" agent shared with
`workflow-catalog-lookup.md` and `workflow-durable-profile.md`. FR language updated accordingly in
both original documents at the time; no other content changed.
2026-08-22 — **Merged** `workflow-business-entities.md` and `workflow-deterministic-compute.md`
into this single document, at the stakeholder's request, once a cross-document coherence/
dependency review confirmed what the decision log above already showed: deterministic-compute
never had an acceptance bar independent of business-entities' own demo (its own AC-3 said so
explicitly) — the two were effectively one delivery unit described as two documents. This merge
introduces no new product decisions; every FR/AC here is carried forward unchanged in substance
from the two originals, renumbered and reworded to read as one coherent document. Both originals
flipped to `superseded`, pointing here. Status carried forward as **Ready for design** without a
fresh stakeholder readback, since no requirement changed — this is a reorganization, not a new
interview.