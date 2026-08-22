# Durable mutable business entities for workflows — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of five sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-deterministic-compute.md`,
`docs/requirements/workflow-durable-profile.md`, `docs/requirements/workflow-nl-query-generation.md`
(the last one added mid-interview, spun off from `workflow-catalog-lookup.md`). Each is
independently motivated and independently shippable — read this one for cart/order-shaped
mutable state specifically.

## Intent
_Being drafted — see decision log._

## Problem & current state
Today falkor-chat's workflow engine has no way to represent durable, mutable, queryable
business state (the kind of thing a shopping cart or a placed order is: a structured record
that gets edited over multiple turns, needs to be queried/filtered, and must outlive a single
run). Everything a workflow carries today lives in `ctx` — a flat, run-scoped, serialized JSON
blob that cannot be filtered or queried (`falkor-chat/AGENTS.md` rule 8), and is discarded once
the run ends. The only built-in tools are `post_message`, `graphrag_retrieve` (semantic search
over chat-like text, not exact/filterable facts), and `human_handoff` — none of them create,
mutate, or query structured domain state.

This gap surfaced from comparing against the standalone `salesperson/` component (a Streamlit
pastel-shop chatbot with its own ad-hoc, purely in-process cart) during a didactic walkthrough,
followed by an architect + graph-dba primitive-level reflection in this conversation.
`salesperson` is a motivating illustration only — the stakeholder confirmed the goal is a
general-purpose capability for future workflows, not a commitment to rebuild `salesperson`
specifically.

## User stories
_To be captured._

## Functional requirements
_To be captured._

## Out of scope
_To be captured._

## Acceptance criteria
_To be captured._

## Open questions
_To be captured._

## Decision log
2026-08-22 — Interview opened following an architect + graph-dba primitive-level reflection on
representing catalog/cart/order/compute/profile in falkor-chat's paradigm. Stakeholder confirmed:
(a) this is a major feature spanning multiple infrastructure deliveries, tracked as separate
requirements documents per capability, not one combined document; (b) the goal is a
general-purpose capability — `salesperson` is illustrative, not a target to rebuild; (c) the
four capabilities are catalog/reference lookup, mutable business entities (cart+order together,
this document), deterministic computation, and durable user-profile writes.
