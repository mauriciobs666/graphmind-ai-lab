# Document size discipline — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-10-29

## Intent

Stop engineering-process documents from growing round after round. Two goals of equal weight:
the agents who must read them (reviewers above all) stop spending most of their tokens re-reading
accumulated text, and the stakeholder can still read and review the documents themselves.

"Solved" means **growth stops**: a revision replaces text rather than adding to it, so a document
does not get bigger with each round. A fixed per-document word limit is not the success test.

## Problem & current state

Evidence from the connected-code-graph family (measured 2026-10-29, `tico` RCA in conversation):

- The family is ~134,000 words (~180k tokens). Architect runs cost 140k–444k tokens, analyst runs
  190k–301k, mostly spent reading documents (`docs/plans/connected-code-graph-coordination.md`,
  Cost column).
- Growth comes from review rounds: requirements 3,983 → 8,995 w; graph note +53% and ML note +44%
  in one revision, each added as a new section (§6 3,556 w, §8 4,198 w) beside the old text.
- Design documents carry their own production history: main plan §2 finding→resolution tables
  (2,346 w, 16%); P0 plan §11+§15 findings/dispositions (2,584 w, 12%); requirements decision log
  (3,148 w, 35%).
- The same fact is restated across documents (e.g. the AC-12 trace+log rule in four documents).
- Individual requirements accrete clauses (AC-12 = 327 words vs. a 40-word median).
- No agent prompt (`architect`, `analyst`, `graph-dba`, `data-scientist`, `teco`, `tico`) carries
  guidance on growth across revisions (`analyst.md` caps finding length; root `AGENTS.md` keeps a
  later review pass compact); `analyst`'s 20k-word `review-techniques.md` has never been opened in 29
  analyst runs.

**Prior decision this touches:** `claude/docs/plans/prompt-waste-reduction.md` Stage G, now in root
`AGENTS.md`: read-whole documents (`BACKLOG.md`, `DESIGN.md`, `AGENTS.md`, `README.md`) are bounded;
read-by-lookup documents (`HISTORY.md`, `reviews/`, closed `plans/`, `test-reports/`) "may grow
without bound, and that is correct". Active plans, specialist notes, specs and requirements are
not classified. During a live chain, reviews and active plans are in practice read whole.

## User stories

- As a **reviewer agent**, I want the documents I gate to hold only their current content, so that
  I spend my tokens judging the design rather than re-reading its history.
- As a **plan or note author** revising after a review, I want to change the text in place rather
  than add sections, so that the next round starts from one coherent document.
- As the **stakeholder**, I want each document to read as one present-tense statement, so that I
  can review it myself without reconstructing what changed across rounds.

## Functional requirements

- FR-1 — **Supersedes in part** the Stage G ruling (root `AGENTS.md`, "Module documentation
  convention"; `claude/docs/plans/prompt-waste-reduction.md` §4, Stage G) for the FR-2 kinds only: a
  document of those kinds that is not yet `archived` is treated as read whole, so it is held to
  "growth stops" even if its kind is otherwise read by lookup (e.g. `reviews/`). Once `archived`,
  it is read by lookup and may stay as it is. Every other document (`HISTORY.md`, `QUERIES.md`,
  test reports, …) keeps its Stage G classification unchanged.
- FR-2 — Covers three engineering-process document kinds: `requirements/`; `plans/` (architect
  plans, `-graph`/`-ml` specialist notes, spec documents — not `-coordination` ledgers); and
  `reviews/`.
- FR-3 — When an in-scope document is revised, only genuinely new scope (a new requirement, a new
  design area) may add words. Reworking existing content — answering review findings, applying
  stakeholder decisions, fixing errors — replaces the affected text in place; growth that comes
  from rework is a violation. In a review, a new compact `## Pass N` (a verdict on a new revision)
  counts as new content; the existing compact-pass rule bounds it, and earlier passes stay.
- FR-4 — A plan or specialist note reads as the current design only. The record of how it was
  produced — which review finding was handled how, finding→resolution cross-references — lives in
  the review: the reviewer records each finding's outcome in its next pass. The reasons for a
  design choice belong in the plan as part of the design (why it is this way), not as a response
  to a finding. A plan may cite a finding ID; it does not restate the finding. Open stakeholder
  questions may sit in the plan while open and are removed, with the answer folded into the
  design, once settled; the reviewer's next pass records each such answer in one line (question →
  stakeholder answer), so a stakeholder ruling stays distinguishable from an author's choice.
- FR-5 — A requirements decision log stays strictly append-only, and each entry is one question
  and one answer, plus the FR/AC it affects, with no narrative or rationale. An entry that reverses
  an earlier decision names the entry it overturns; earlier entries are never edited.
- FR-6 — A fact is stated once, in the document that owns it; any other in-scope document that
  needs it cites the owner instead of restating it.
- FR-7 — No separate cleanup of existing documents: an in-scope, non-archived document is brought
  in line with FR-3–FR-6 the next time it is revised for its own reasons.
- FR-8 — Every review gate on an in-scope document also checks the revision against FR-3–FR-6 and
  reports any violation (growth from rework, restated history, restated facts) as a finding. There
  is no numeric growth threshold: the reviewer judges whether any growth comes from new scope or
  from rework.
- FR-9 — A requirements document gets an `analyst` review gate before it is flipped to Ready for
  design; that gate includes the FR-8 check.

## Out of scope

- `test-plans/`, `test-reports/` and `manuals/` (stakeholder, 2026-10-29).
- Rewriting any document that is already `archived` (FR-1).
- Coordination ledgers (`plans/*-coordination.md`): exempt and out of scope.
- A dedicated retrofit pass over existing documents (FR-7).
- A document's first-version size, and splitting large scope across documents.
- Fixed per-document word limits, and token-cost targets (Intent).

## Acceptance criteria

- AC-1 (doc consistency) — After delivery, no repo convention or agent prompt still states that a
  non-archived `requirements/`, `reviews/` or `plans/` document may grow without bound, and none
  bounds a document outside the FR-2 kinds on FR-1's account; the Stage G
  read-whole/read-by-lookup distinction, wherever it is written, reflects FR-1.
- AC-2 (rework replaces) — Given an in-scope document revised in response to review findings or
  stakeholder decisions, then the gating review states, for any growth, whether it comes from new
  scope or from rework, and reports growth from rework as a finding; apart from a review's new
  compact `## Pass N`, the document has no new section that exists only to record the revision.
- AC-3 (new scope may add) — Given a revision that adds genuinely new scope, then growth is allowed
  for that new content, and the parts that were only reworked still meet AC-2.
- AC-4 (plans hold current design) — Given a plan or specialist note after a revision, then it
  contains no finding→resolution table, no disposition list, and no stakeholder question that has
  already been answered; each finding's outcome and each stakeholder answer folded in since the
  last pass appear, one line each, in the reviewer's next pass, and the plan cites findings by ID
  at most.
- AC-5 (decision log) — Given a requirements document, then every decision-log entry written after
  delivery states one question and one answer (with the FR/AC affected) and nothing else, and every
  entry reversing an earlier decision names the entry it overturns.
- AC-6 (one home per fact) — Given a fact owned by one in-scope document, then no other in-scope
  document restates it; each cites the owner.
- AC-7 (reviewer checks) — Given a review gate on an in-scope document whose revision violates
  FR-3–FR-6, then the review reports each violation as a finding.
- AC-8 (archived untouched; existing docs at next revision) — No archived document is edited for
  this feature; a non-archived existing document meets AC-2–AC-6 after its next ordinary revision.
- AC-9 (requirements gated) — Given a requirements document about to be flipped to Ready for
  design, then an `analyst` review of it exists and covers the FR-8 check; the flip happens only
  after that review.

## Open questions

*(none)*

## Decision log

- 2026-10-29 — Interview opened from `tico`'s RCA of connected-code-graph document growth (option 1
  of three offered).
- 2026-10-29 — Prior-decision check: `claude/docs/plans/prompt-waste-reduction.md` Stage G (root
  `AGENTS.md` living-document compaction rule) is the closest precedent; it bounds read-whole docs
  and explicitly leaves read-by-lookup docs unbounded. No requirement on active plan/requirements
  size exists.
- 2026-10-29 — Primary goal: token cost or readability? → **Both equally.**
- 2026-10-29 — What does "solved" look like? → **Growth stops** (a revision replaces, it doesn't
  add); not a fixed per-document limit, not a measured run-cost target.
- 2026-10-29 — Does this supersede Stage G's "reviews/closed plans may grow unbounded"? → **Bound
  while active**: a non-archived document of any kind counts as read whole; archived ones stay
  lookup-only and untouched (FR-1, AC-1).
- 2026-10-29 — Which document kinds are in scope? → **All four offered:** requirements, plans and
  specialist notes (incl. specs), reviews, coordination ledgers (FR-2).
- 2026-10-29 — What growth does "growth stops" allow? → **Only new scope adds**; rework of existing
  content replaces text (FR-3). Not net-zero, not "any growth if justified".
- 2026-10-29 — Test plans, test reports, manuals? → **Out of scope.**
- 2026-10-29 — Where does a plan's production history live? → **Review and ledger only**; the plan
  reads as the current design (FR-4).
- 2026-10-29 — Requirements decision log? → **One line per decision**, still append-only (FR-5).
- 2026-10-29 — "Growth stops" for coordination ledgers? → **Exempt ledgers**; they may grow freely.
- 2026-10-29 — Same fact restated across documents? → **One home, others cite** (FR-6).
- 2026-10-29 — Ledgers in this feature at all? → **Dropped entirely**; FR-4's questions go to the
  review (FR-2, FR-4 revised).
- 2026-10-29 — Existing in-flight documents? → **Trim at next revision**, no cleanup pass (FR-7).
- 2026-10-29 — Is a first version's size in scope? → **Out of scope.**
- 2026-10-29 — How is compliance checked? → **Reviewer checks it** at each review gate (FR-8).
- 2026-10-29 — Growth tolerance for rework-only revisions? → **Reviewer judges**; no number (FR-8, AC-2).
- 2026-10-29 — Who checks requirements documents? → **Analyst gate added** before Ready for design (FR-9, AC-9).
- 2026-10-29 — Readback confirmed by stakeholder; close with an analyst gate first. Consult: `analyst` asked to review this document (testability, consistency, FR-1 vs root `AGENTS.md`/Stage G, gaps) → pending.
- 2026-10-29 — `analyst` consult result → **needs changes**: B1, B2 blockers; M1, M2 major; m1–m4, n1 (`docs/reviews/document-size-discipline.md`). m1, m2, m4, n1 fixed by `tico`.
- 2026-10-29 — Review passes under "growth stops"? → **A compact Pass N counts as new content**; earlier passes stay (FR-3, AC-2; resolves B2).
- 2026-10-29 — Who records dispositions? → **The reviewer, in its next pass**; design reasons live in the plan as design (FR-4; resolves M1).
- 2026-10-29 — Where do a new plan's open questions live? → **In the plan until answered**, then removed (FR-4, AC-4; resolves M2).
- 2026-10-29 — FR-1's reach? → **Only the FR-2 kinds**; other documents keep Stage G (FR-1, AC-1; resolves B1).
- 2026-10-29 — Reversed log entries? → **The new entry names what it overturns**; earlier entries untouched (FR-5, AC-5). Overturns: "ledgers in scope (FR-2)" and "review and ledger only (FR-4)" entries above, per the "Ledgers … Dropped entirely" entry.
- 2026-10-29 — Meaning of "one line"? → **One question, one answer** plus FR/AC, regardless of wrap (FR-5, AC-5; resolves m3).
- 2026-10-29 — Close now or analyst Pass 2? → **Pass 2 first**; same `analyst` consult resumed on the fixes.
- 2026-10-29 — Correction (Pass 2 m5): the "FR-1's reach? → Only the FR-2 kinds" entry overturns, by narrowing, the "Bound while active … of any kind" entry.
- 2026-10-29 — Where is a stakeholder ruling on a plan question recorded once folded in? → **The reviewer's next pass logs it**, one line (FR-4, AC-4; resolves Pass 2 m6).
- 2026-10-29 — Analyst Pass 2 → approve with suggestions; m5, m6 folded. Status → **Ready for design** per the stakeholder's "Pass 2 first, then flip" decision.
