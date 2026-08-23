# Natural-language query generation over structured graph data — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of six sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-business-entities.md`,
`docs/requirements/workflow-deterministic-compute.md`, `docs/requirements/workflow-durable-profile.md`,
`docs/requirements/workflow-composition.md`. This document was spun off mid-interview from
`workflow-catalog-lookup.md`, not scoped out on day one; `workflow-composition.md` is younger
still, opened mid-interview of `workflow-durable-profile.md`.

## Intent
Close a structural gap ahead of a concrete consumer, same proactive-infrastructure framing as
the other siblings — but the stakeholder has flagged this one as **a major aspect** of the whole
effort, not a minor add-on: it needs proof at two levels, not one. First, live: the same combined
"salesperson" demo agent (`workflow-catalog-lookup.md`, `workflow-business-entities.md`,
`workflow-deterministic-compute.md`, `workflow-durable-profile.md`) must be able to answer
arbitrarily-phrased questions, not just the fixed shapes `workflow-catalog-lookup.md` already
proves. Second, rigorously: a golden-set evaluation methodology (question/answer pairs, accuracy
scoring), in the spirit of `docs/plans/graphrag-eval.md`, because "does it answer correctly" is
not something a single live demo can establish on its own. A full adversarial/security review of
the mechanism is explicitly treated as a **separate, later, independent step** — building a
structurally safe mechanism is in scope here; proving it adversarially is not.

## Problem & current state
falkor-chat's tools are deliberately fixed, author-defined schemas — nothing today lets an LLM
generate its own query against structured graph data from a free-form question.
`workflow-catalog-lookup.md` covers a fixed set of pre-defined query shapes only; genuinely
arbitrary phrasing is out of scope there. `salesperson` (the comparison case that originally
surfaced this whole effort) already does something like this with its `cypher_qa` tool, but with
weak safety (a regex keyword blocklist is the only guard against a generated query doing
something it shouldn't) and no accuracy-evaluation methodology at all — neither of those is
acceptable to simply copy forward.

## User stories
- As a workflow author, I want a step to answer a natural-language question against structured
  graph data phrased however the user likes, not limited to a fixed set of question shapes, so
  the assistant feels like it understands the question rather than pattern-matching it.
- As a workflow author, I want this capability to generalize across structured datasets falkor-chat
  might host, not just the demo electronics catalog, so it doesn't need rebuilding for every new
  domain.
- As someone judging whether this is safe enough to build on top of, I want the mechanism
  structurally prevented from mutating or destroying graph data — not merely discouraged from
  doing so by a blocklist — even though a full adversarial security review happens later, as its
  own independent step.
- As someone judging whether this is accurate enough, I want a golden-set evaluation (question /
  expected-answer pairs) with a defined passing bar, designed by `data-scientist`, so "does it
  actually answer correctly" is measured, not assumed.
- As someone judging whether this works at all, I want to see it live: the combined "salesperson"
  demo agent correctly answering arbitrarily-phrased catalog questions that don't match any of
  `workflow-catalog-lookup.md`'s fixed shapes.

## Functional requirements
- **FR-1** — A workflow step can answer a natural-language question against structured graph data
  phrased arbitrarily — not limited to a fixed set of pre-defined question templates.
- **FR-2** — The mechanism generalizes across structured datasets; it is not hardcoded to the
  electronics catalog's specific schema.
- **FR-3** — The mechanism is **structurally** prevented from performing a mutating or
  destructive graph operation (create/update/delete/etc.), regardless of how a question is
  phrased — a baseline, load-bearing safety property built into the mechanism's design, not a
  best-effort filter layered on top. (This is distinct from — and does not substitute for — the
  adversarial security review noted as out of scope below.)
- **FR-4** — A golden-set evaluation methodology exists (question/expected-answer pairs) with a
  defined passing accuracy bar. The specific metric and threshold are a `data-scientist` design
  decision made during the design phase, not fixed in this document.
- **FR-5** — This capability's live proof-of-concept is the same combined "salesperson" demo
  agent as the other sibling capabilities — now able to answer arbitrarily-phrased catalog
  questions beyond `workflow-catalog-lookup.md`'s fixed shapes.

## Out of scope
- **A full adversarial/red-team security review of the mechanism.** Deliberately treated as a
  separate, later, independent step (likely a dedicated `security-expert` pass once the
  mechanism is built) — not part of this document's acceptance bar. FR-3 requires a structurally
  safe design; it does not require that design be adversarially proven here.
- Fixing the specific accuracy metric/threshold in this document — left to `data-scientist`.
- The specific query-generation technology/mechanism (e.g. LLM-generated Cypher vs. another
  approach) — an architecture decision, not fixed here.
- Any write/mutating capability — this document is about answering questions, not writing data;
  FR-3 makes that a hard boundary, not merely an implementation detail.
- Fixing catalog-lookup's fixed-shape queries (`workflow-catalog-lookup.md`'s own scope) — this
  document only concerns phrasing *beyond* those shapes.

## Acceptance criteria
- **AC-1** (FR-1) — Given the demo catalog, when two structurally different, arbitrarily-phrased
  questions are asked (not matching any of `workflow-catalog-lookup.md`'s fixed shapes), then
  both are answered correctly.
- **AC-2** (FR-2) — Given a second structured dataset with its own schema (not the electronics
  catalog), when the same mechanism is pointed at it, then it can answer questions against that
  dataset too, without being rebuilt specifically for that schema. (The concrete verification
  approach — e.g. a second small demo dataset — is left to the architect/QA.)
- **AC-3** (FR-3) — Given a question phrased to attempt a mutating/destructive operation, when it
  is processed, then no such operation occurs, by construction — verified as part of building the
  mechanism; a dedicated adversarial test suite is the separate, later security review, not
  gated here.
- **AC-4** (FR-4) — A golden-set evaluation exists and the mechanism meets whatever passing bar
  `data-scientist` defines for it, before this capability is considered complete.
- **AC-5** (FR-5) — The combined "salesperson" demo agent correctly answers at least one
  arbitrarily-phrased question that would not match any of `workflow-catalog-lookup.md`'s fixed
  query shapes.

## Open questions
- The specific accuracy metric and passing threshold (FR-4) are intentionally undefined here —
  they are a `data-scientist` design decision to be made during the design phase, not a gap in
  this interview.

## Decision log
2026-08-22 — Split out from `workflow-catalog-lookup.md`'s original FR-3 mid-interview (see that
document's decision log).
2026-08-22 — Interviewed fifth (after durable-profile); proactive infrastructure, no concrete
consumer, same as the other siblings.
2026-08-22 — Stakeholder marked this "a major aspect" of the whole effort, requiring both a live
demo proof (inside the combined "salesperson" agent) and a rigorous golden-set evaluation — not
either/or.
2026-08-22 — Safety scope: a full adversarial/red-team security review is explicitly deferred as
a separate, later, independent step. This document still requires (FR-3) that the mechanism be
*structurally* incapable of mutating/destroying data — a baseline design property — but does not
require adversarial-test-suite acceptance criteria as part of this capability's own gate.
2026-08-22 — Accuracy bar: the specific metric/threshold is left to `data-scientist` to propose
during design; this document only requires that an evaluation methodology and a passing bar
exist and are met.
2026-08-22 — Generality: the mechanism must be schema-agnostic — it must work against future
structured datasets, not just the demo electronics catalog, matching this whole effort's
general-capability goal (not `salesperson`-specific).
