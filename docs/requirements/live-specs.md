# Live specs — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-10-30

## Intent

Make each component's live spec — a document stating how the system works **now** — the source of
truth agents and the stakeholder read to understand a component, as `falkor-chat/docs/DESIGN.md`
was originally meant to be. Engineering-process documents (requirements, plans, reviews, test
artifacts) become a record of how a change was made, not something anyone must read to learn
current behavior. The aim is to cut the volume of documentation read and written per change.

## Problem & current state

Measured 2026-10-30:

- falkor-chat: live spec (`DESIGN.md` 8.5k w, `SERVER.md` 9.8k w, `QUERIES.md` 29k w) ≈ 48k w
  against ≈ 790k w of `requirements/`, `plans/`, `reviews/`, `test-plans/`, `test-reports/`.
- model-bench: no live spec; `README.md` + `AGENTS.md` ≈ 4k w. Current design lives in the
  root `docs/plans/small-model-benchmarking.md` (122k w) and its family (coordination 89k w,
  `-ml` 61k w, reviews 129k w), plus `model-bench/docs/` (≈ 154k w).
- `falkor-chat/docs/DESIGN.md`: 41 commits ever vs 318 to `plans/`+`reviews/`; untouched since
  2026-09-20 across 58 falkor-chat commits, several changing behavior contracts (e.g. K-079's
  "Layer 1 contract", stated in `falkor-chat/docs/plans/salesperson-query-graph-data.md`, a
  docstring and `HISTORY.md`, but not in the live spec).
- No step in the team's pipeline makes updating the live spec part of shipping a change.

Related: `docs/requirements/document-size-discipline.md` (growth across revisions); this feature
covers ground its Out of scope leaves open.

## User stories

- As an agent starting work in a component, I want one current spec to read, so that I don't
  reconstruct current behavior from old plans and reviews.
- As an agent doing an RCA or rolling back one feature, I want that feature's archived plan and
  reviews still in the repo, so that I can see how and why it was built.
- As the stakeholder, I want to read what a component does today in one place, so that I can
  review it without archaeology.

## Functional requirements

- FR-1 — A shipped feature's engineering-process documents stay in the repo under the existing
  archive rule (root `AGENTS.md`); no agent or reader needs to open one to learn the system's
  current behavior.
- FR-2 — A feature is one backlog item in the component's own numbering or, where no backlog item
  tracks the work, one requirements document; any other change recorded in `HISTORY.md` that
  alters a contract the spec states also counts. It closes when its delivery is recorded in
  `HISTORY.md`, and not before the component's live spec states the behavior it shipped; the spec
  update happens once, at that close. Archiving its documents stays at milestone close.
- FR-3 — The live spec states only current behavior, at the level of externally observable
  contracts (e.g. tool/API/CLI surface, graph or file schemas, workflow or scoring semantics,
  configuration); it carries no change history (git and
  `HISTORY.md` hold that).
- FR-4 — A component's live spec is a small named set of documents (falkor-chat: `DESIGN.md`,
  `SERVER.md`, `QUERIES.md`); the component's `AGENTS.md` names that set. model-bench gains a
  live spec of its own.
- FR-5 — The repo has a root `REVIEW.md` holding repo-wide, review-only standing instructions
  (what counts as serious, what to always check, what not to report, the evidence bar), and
  falkor-chat and model-bench each have their own `REVIEW.md` adding component-specific ones.
- FR-6 — Every agent that writes a review or gate verdict reads the root `REVIEW.md` and the `REVIEW.md` of each
  component the reviewed work touches, and applies them.
- FR-7 — `teco` is accountable for the spec update at feature close: it ensures the update is
  written, delegating the writing, and that `analyst` checks the updated spec against the shipped
  code (not the plan). The feature does not close while that check's verdict is "needs changes".
  Supersedes `claude/teco/teco.md`'s rule that implementers update docs in the same change as the
  code, for live-spec content only.
- FR-8 — A reviewer who finds an issue that should become a standing review rule proposes the
  `REVIEW.md` line in its review; `teco` ensures proposed lines are applied at feature close,
  alongside the spec update (FR-7).
- FR-9 — `REVIEW.md` files stay small: they hold only instructions that change review behavior,
  and whoever adds a rule also removes rules now enforced elsewhere (tests, CI, a hook). Size
  smells (not gates): root `REVIEW.md` ~1,000 words, each component's ~800 words.
- FR-10 — A one-time catch-up brings each in-scope component's live spec up to date with the
  shipped code: falkor-chat's spec set absorbs what changed since it was last updated, and
  model-bench's spec is written. The catch-up also removes revision notes and delivered-ticket
  annotations from the specs; references to open backlog items stay. `teco` ensures it is
  written and checked as in FR-7.
- FR-11 — FR-1–FR-9 and FR-12–FR-13 are team-wide conventions: any component adopts them the
  first time a feature closes in it. A component with no live spec gets a whole-component spec,
  written from its code and checked as in FR-7, before that first feature closes. The FR-10
  catch-up covers falkor-chat and model-bench only.
- FR-12 — Where a component has a live spec, a new plan states only the change against it,
  citing the affected spec sections; it does not restate spec content. A review reports restated
  spec content as a finding.
- FR-13 — For behavior that an open (not yet closed) feature changes, a new plan cites that
  feature's plan alongside the spec.

## Out of scope

- Deleting shipped features' process documents (decision log 2026-10-30).
- A one-time catch-up of any component other than falkor-chat and model-bench (FR-11).
- Wiring `REVIEW.md` into a hosted pull-request review service (e.g. Claude Code Review); only
  the team's own reviewers read it.

## Acceptance criteria

- AC-1 (catch-up) — Given the catch-up is done, when `analyst` compares falkor-chat's spec set and
  model-bench's spec against the shipped code, it reports no externally observable contract
  (FR-3) that the spec omits or contradicts, and its verdict is not "needs changes".
- AC-2 (spec index) — falkor-chat's and model-bench's `AGENTS.md` each name their live spec set.
- AC-3 (no history in spec) — The spec documents carry no revision notes, delivered-ticket
  annotations or "changed on" clauses.
- AC-4 (feature close) — Given the first feature to close after delivery, when its delivery is
  recorded in `HISTORY.md`, its spec update and an `analyst` check against shipped code with a verdict other
  than "needs changes" are both recorded.
- AC-5 (`REVIEW.md` exists) — Root, `falkor-chat/` and `model-bench/` each have a `REVIEW.md`
  within its FR-9 size smell, holding only review-behavior instructions.
- AC-6 (reviewers read it) — The instructions of every agent that writes reviews or gate verdicts
  (incl. `analyst`, `qa-engineer`, `security-expert`, `data-scientist`, `graph-dba`, `cobb`)
  direct it to read and apply the root and touched-component `REVIEW.md` files.
- AC-7 (rule feed) — A review can carry a proposed `REVIEW.md` line; the feature-close step names
  applying such lines and pruning rules now enforced elsewhere.
- AC-8 (delta plans) — The first plan written after delivery for falkor-chat or model-bench cites
  spec sections for existing behavior and restates none, and cites an open feature's plan where it
  depends on behavior that feature changes (FR-13); its review checks this.
- AC-9 (doc consistency) — No repo convention or agent prompt still allows closing a feature
  without the spec update, still tells implementers to update live-spec content in the same change
  as the code, or describes `DESIGN.md` as something other than the live spec.
- AC-10 (new component) — Given a component with no live spec, when its first feature
  closes, a whole-component spec exists, is named in its `AGENTS.md`, and has passed the FR-7
  check.

## Open questions

_None._

## Decision log

- 2026-10-30 — Fate of a shipped feature's process documents? → keep, archived, as audit trail;
  nobody needs to read them to know current behavior (FR-1, Out of scope).
- 2026-10-30 — When is the spec brought up to date? → once, at feature close (FR-2); git keeps the
  history, so the spec carries none (FR-3).
- 2026-10-30 — Does "git keeps the history" reverse the keep-archived answer? → no; process documents
  stay archived in the repo, for later RCA and for rolling back a specific feature (FR-1).
- 2026-10-30 — What is the live spec per component? → a small named set indexed in `AGENTS.md`;
  falkor-chat keeps `DESIGN.md`+`SERVER.md`+`QUERIES.md`, model-bench gets one (FR-4).
- 2026-10-30 — What is `REVIEW.md`? → the emerging convention (Claude Code Review, Qodo, Devin):
  review-only standing instructions; one at the root plus one per component, read by the team's
  reviewers on every review (FR-5, FR-6).
- 2026-10-30 — Who updates the spec at feature close, and who checks it? → `teco` updates it;
  `analyst` checks it against the shipped code (FR-7).
- 2026-10-30 — How does a review lesson reach `REVIEW.md`? → the reviewer proposes the line; it is
  applied at feature close; `REVIEW.md` stays small (FR-8, FR-9).
- 2026-10-30 — Size bar for `REVIEW.md`? → smells of ~1,000 words root, ~800 per component (FR-9).
- 2026-10-30 — Catch up existing spec drift once? → yes, `teco` writes, `analyst` checks (FR-10).
- 2026-10-30 — Scope? → conventions team-wide, adopted at a component's first feature close;
  catch-up for falkor-chat and model-bench only (FR-11, Out of scope).
- 2026-10-30 — Must new plans state only the change against the spec? → yes; restated spec content
  is a review finding (FR-12).
- 2026-10-30 — Hosted PR review service integration? → out of scope (Out of scope).
- 2026-10-30 — Who writes the spec update? → `teco` ensures it and delegates the writing;
  overturns "`teco` updates it" (entry 6) and "`teco` writes it" (entry 9) (FR-7, FR-8, FR-10, AC-9).
- 2026-10-30 — What is a feature? → one `K-`numbered backlog item; archiving stays at milestone
  close (FR-2, AC-4).
- 2026-10-30 — What does a parallel plan cite while the spec lags? → the spec plus open features'
  plans (FR-13).
- 2026-10-30 — What does a component with no spec owe at its first feature close? → a
  whole-component spec (FR-11, AC-10).
- 2026-10-30 — Consult: `analyst` on readiness (Pass 1, `docs/reviews/live-specs.md`) → needs
  changes: B1, M1–M4 resolved by the four entries above; m1–m4, n1, n3 fixed; n2 not applied
  (log is append-only).
- 2026-10-30 — What counts as a feature outside `K-` numbering? → a backlog item in the
  component's own numbering, or a requirements document where none tracks it; refines entry 14
  (FR-2, AC-4, AC-9, AC-10).
- 2026-10-30 — Depth of "spec covers the code"? → externally observable contracts only (FR-3, AC-1).
- 2026-10-30 — Does a direct contract-changing fix with only a `HISTORY.md` entry count as a
  feature? → yes; refines entry 18 (FR-2).
- 2026-10-30 — Ready for design? → confirmed by stakeholder; `analyst` Pass 3 approve with
  suggestions (`docs/reviews/live-specs.md`).
