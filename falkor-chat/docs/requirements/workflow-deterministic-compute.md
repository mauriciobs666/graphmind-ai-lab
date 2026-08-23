# Deterministic computation inside a workflow — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of six sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-business-entities.md`,
`docs/requirements/workflow-durable-profile.md`, `docs/requirements/workflow-nl-query-generation.md`
(added mid-interview, spun off from `workflow-catalog-lookup.md`),
`docs/requirements/workflow-composition.md` (opened later still, mid-interview of
`workflow-durable-profile.md`). Read this one for exact, non-AI computation (totals, quantity math) inside a workflow run specifically.

## Intent
Close a structural gap ahead of any specific consumer — same proactive-infrastructure framing as
the other sibling capabilities. Rather than a standalone demo, this one is proven through
`workflow-business-entities.md`'s own cart/order proof-of-concept: that capability already needs
its line-item totals and order snapshots to be exact, not LLM-approximated, so making that math
demonstrably deterministic *is* this capability's acceptance bar.

## Problem & current state
Today's workflow step types are `agent` (an LLM turn), `human`/`wait` (park for a reply), and
`decision` (a pure branch — its `cmp` guard only *compares* two values, it never *computes* a new
one). There is no primitive for exact, deterministic computation as part of a run — today it
would have to be routed through an LLM `agent` turn calling a tool, which still involves invoking
the model even if the arithmetic itself is deterministic. A general expression-evaluation guard
(`expr`) already exists as a documented, unimplemented `NotImplementedError` in the guard
language — a related, broader idea (arbitrary expression evaluation) that was apparently
attempted and abandoned; this capability is deliberately narrower (fixed arithmetic on line
items), not a revival of general `expr` evaluation.

## User stories
- As a workflow author, I want a step to compute an exact running total from a set of line items
  (price × quantity per line, summed across lines), so a cart or order's total is never an
  LLM-approximated number.
- As a workflow author, I want that computation to happen without requiring an LLM call at all,
  so it's fast, free of model cost, and never varies between runs given the same inputs.
- As someone judging whether this capability works, I want to see it proven inside the
  `workflow-business-entities.md` demo (its cart/order totals), not a separate toy example, so
  the proof is grounded in a real consumer rather than an abstract one.

## Functional requirements
- **FR-1** — A workflow can compute a running total from a set of line items (price × quantity
  per line, summed across all lines) as part of a run.
- **FR-2** — The same set of inputs always produces the same, exact output — no variance run to
  run.
- **FR-3** — Performing this computation does not require an LLM/model call — it must not cost a
  model invocation purely to add up numbers. (This requirement bears directly on the
  step-type-vs-tool design fork the architect flagged during the earlier reflection on this
  effort — a tool dispatched from inside an `agent` step still involves an LLM turn to decide to
  call it, which this FR rules out as the *only* way to satisfy the requirement. The document
  does not choose the mechanism; it does constrain it.)
- **FR-4** — This capability's proof-of-concept is `workflow-business-entities.md`'s own cart/order
  demo: its line-item totals and order snapshots are computed via this mechanism. No separate
  demo workflow is required.

## Out of scope
- Anything beyond price × quantity summation for this pass — no discounts, tax percentages,
  rounding-mode rules, or other arithmetic. A future need can extend this later.
- Aggregation across multiple orders/carts (e.g. "total sales this week") — this is per-run
  computation only, not a reporting/analytics capability.
- A general expression-evaluation language (reviving `expr`-style arbitrary evaluation) — this
  capability is a fixed, narrow arithmetic operation, not a programmable formula engine.
- The specific mechanism (a new step type vs. a tool vs. something else) — left entirely to the
  architect; FR-3 constrains it without choosing it.

## Acceptance criteria
- **AC-1** (FR-1, FR-2) — Given a set of line items with known prices and quantities, when the
  total is computed, then it equals the exact hand-calculated sum, and repeating the computation
  with the same inputs yields the identical result every time.
- **AC-2** (FR-3) — Given the computation runs as part of a workflow, when it executes, then no
  LLM/model call is made solely to perform it (verifiable via the run's trace or an equivalent
  check the architect/QA define).
- **AC-3** (FR-4) — Given the `workflow-business-entities.md` demo (cart building through order
  placement), when its totals and order snapshots are inspected, then they were produced by this
  mechanism — this capability has no acceptance criteria of its own beyond that demo's own
  AC-1/AC-2/AC-4/AC-5.

## Open questions
None outstanding — pending stakeholder confirmation at readback.

## Decision log
2026-08-22 — Split out as one of five sibling capabilities; interviewed third.
2026-08-22 — Proactive infrastructure again; proof-of-concept piggybacks on
`workflow-business-entities.md`'s own cart/order demo rather than a standalone one.
2026-08-22 — Scope kept deliberately simple for this pass: plain price × quantity summation only,
no discounts/tax/rounding — can extend later if a real need arises.
2026-08-22 — "Deterministic" means both things: exact/repeatable correctness, *and* not requiring
an LLM call just to compute it (a cost/latency/reliability concern, not just a correctness one).
The latter (FR-3) directly bears on — without resolving — the step-type-vs-tool fork the
architect flagged earlier in this effort.
2026-08-22 — Stakeholder confirmed the readback with no changes; flipped to Ready for design.
2026-08-22 — Retroactive note (surfaced during `workflow-durable-profile.md`'s interview): no
content change needed here — `workflow-business-entities.md`'s demo, which this capability was
already piggybacking on (FR-4), is itself now the single combined "salesperson" demo agent
shared with `workflow-catalog-lookup.md` and `workflow-durable-profile.md` too.
