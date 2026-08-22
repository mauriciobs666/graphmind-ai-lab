# Structured catalog/reference lookup for workflows — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of five sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-business-entities.md`, `docs/requirements/workflow-deterministic-compute.md`,
`docs/requirements/workflow-durable-profile.md`, `docs/requirements/workflow-nl-query-generation.md`.
Read this one for read-only, exact/filterable domain data via a fixed set of query shapes (e.g.
a product catalog) specifically — not for mutable state (the business-entities doc) and not for
answering arbitrarily-phrased questions (spun off into its own sibling,
`workflow-nl-query-generation.md`, once that turned out to be a distinct project).

## Intent
Close a structural gap ahead of any specific consumer: today a falkor-chat workflow has no way
to answer an exact, filterable question against structured reference data — only fuzzy semantic
search over chat-like text. This is proactive infrastructure work, not a response to a concrete
requester, so — following the pattern falkor-chat already uses to prove other engine
capabilities (the `triage` and `access-request` demo workflows) — "done" means a runnable
proof-of-concept, not just tests against an abstract primitive.

## Problem & current state
The only workflow tool that reaches domain-adjacent data today is `graphrag_retrieve` —
embedding-based semantic search over chat-like text, with a distance-cutoff/abstention policy.
It cannot answer an exact, filterable question ("what is the price of X", "which items are
under $Y") because nothing like structured reference data exists in the schema at all —
`reference` (the read-mostly, replicated graph that already holds `WorkflowDef` templates) has a
placeholder box for "domain reference data / ontology / catalogs" in the topology diagram, but
nothing is built there.

A related tension surfaced during the interview: `salesperson` (the comparison case that
originally surfaced this gap) answers arbitrarily-phrased questions by having an LLM generate its
own Cypher against the schema, fenced off only by a keyword blocklist — a materially different,
harder problem (safety, schema-awareness, accuracy evaluation) than a fixed-shape lookup tool.
That capability has been split out into its own sibling document,
`docs/requirements/workflow-nl-query-generation.md`, and is explicitly out of scope here (see
below).

## User stories
- As a workflow author, I want a step to retrieve an exact fact (name, category, price) about a
  specific catalog item, so that I can answer factual questions without relying on fuzzy semantic
  search or hand-rolling a one-off tool per fact.
- As a workflow author, I want a step to filter/list catalog items matching a criterion (e.g. by
  category, by price range), so that I can answer "which items match X" questions, not just
  single-item lookups.
- As someone judging whether this capability actually works, I want a runnable proof-of-concept —
  a demo workflow plus a seeded demo catalog, in the same spirit as `triage`/`access-request` —
  so I can verify it end-to-end rather than trust it in the abstract.

## Functional requirements
- **FR-1** — A workflow can retrieve exact facts (name, category, price) about a specific,
  named catalog item.
- **FR-2** — A workflow can retrieve the set of catalog items matching a filter criterion (e.g.
  category, price range) — not single-item lookup only.
- **FR-3** — The lookup supports a fixed, author-defined set of query shapes: exact-name lookup,
  filter by category, and filter by price range. A question's phrasing must map onto one of
  these shapes to succeed — answering genuinely arbitrary phrasing is explicitly out of scope
  (see below).
- **FR-4** — When a question names an item/category that does not exist in the catalog, the
  workflow states plainly that nothing matched — it does not fabricate an answer. (Mirrors the
  abstention behavior `graphrag_retrieve` already has.)
- **FR-5** — The demo catalog's data is seeded via a one-time script (comparable to
  `seed_workflows.sh`) — no runtime API for creating/editing catalog entries is required.
- **FR-6** — The catalog is a single, shared, global dataset — not scoped per workspace.
- **FR-7** — A demo proof-of-concept ships with the capability: a runnable workflow definition
  plus a seeded demo catalog of consumer electronics (flat shape — name, category, price only),
  materialized and verifiable the same way `triage`/`access-request` are today.

## Out of scope
- Mutating catalog data from within a workflow (create/update/delete) — that belongs to the
  sibling `workflow-business-entities.md` capability.
- A runtime API to publish/manage catalog entries — this feature is seed-script-only.
- Per-workspace catalogs or per-workspace catalog overrides.
- Attributes beyond name/category/price for the demo catalog (e.g. detailed specs, stock,
  images) — the demo is intentionally flat.
- Answering arbitrarily-phrased natural-language questions (i.e. anything that doesn't map onto
  one of FR-3's fixed query shapes) — that is `docs/requirements/workflow-nl-query-generation.md`,
  a distinct, harder capability (safety, schema-awareness, accuracy evaluation) split out once
  the stakeholder recognized it as its own project, not a detail of this one.
- The specific mechanism behind FR-3's fixed query shapes (e.g. how the model is guided to pick
  and parameterize one) is an architecture decision, not fixed here.
- Rebuilding or replacing `salesperson`.

## Acceptance criteria
- **AC-1** (FR-1, FR-7) — Given the demo catalog is seeded and the demo workflow is
  materialized, when a user asks a single-item factual question about a named product (e.g. "how
  much does the X cost"), then the workflow returns the correct exact answer.
- **AC-2** (FR-2) — Given the same setup, when a user asks a filtering/listing question (e.g.
  "which laptops are under $1000"), then the workflow returns the correct matching set.
- **AC-3** (FR-4) — Given a question naming a product/category absent from the catalog, when
  asked, then the workflow clearly states no match was found rather than fabricating an answer.
- **AC-4** (FR-3) — Two differently-worded questions that both map onto the *same* fixed query
  shape (e.g. "how much is the X" and "what's the price of the X" — both an exact-name lookup)
  succeed with the correct answer. This checks the fixed-shape lookup tolerates reasonable
  wording, not that it handles arbitrary phrasing in general (that's
  `workflow-nl-query-generation.md`).
- **AC-5** (FR-5, FR-7) — The demo catalog and workflow can be seeded and verified the same way
  `triage`/`access-request` are today (a seed script, plus a verification check confirming the
  data/definition are actually in place).

## Open questions
None outstanding — pending stakeholder confirmation at readback.

## Decision log
2026-08-22 — Split out as one of four sibling capabilities; prioritized first among the four.
2026-08-22 — Proactive infrastructure, no concrete consumer yet; stakeholder wants a runnable
proof-of-concept (like `triage`/`access-request`) as the acceptance bar, not tests alone.
2026-08-22 — Demo dataset: a consumer-electronics catalog (stakeholder changed from an initially
proposed pastel-flavor menu), flat shape (name, category, price only).
2026-08-22 — Query surface: both single-item exact lookup and multi-item filtering/listing are
required (not lookup-only).
2026-08-22 — Catalog management: seed-script-only, no runtime CRUD API, for this feature.
2026-08-22 — Catalog scope: one shared/global catalog, not per-workspace.
2026-08-22 — Arbitrary natural-language phrasing was initially requested (FR-3) but, once the
stakeholder recognized it as a distinct project (safety + accuracy-evaluation concerns, not a
lookup detail), split out into its own sibling document,
`docs/requirements/workflow-nl-query-generation.md`. This document now scopes FR-3 down to a
fixed set of author-defined query shapes (exact-name lookup, category filter, price-range
filter).
2026-08-22 — Stakeholder confirmed the readback with no changes; flipped to Ready for design.
