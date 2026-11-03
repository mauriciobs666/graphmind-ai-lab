# Cross-initiative coordination — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Version:** 4 · **Extends:** `claude/docs/requirements/coordination-graph.md` · **Last updated:** 2026-11-03 · **Reviews:** `docs/reviews/cross-initiative-coordination.md`

## Intent
`claude/docs/requirements/coordination-graph.md` already puts a mirror of every coordination
ledger in the graph (file authoritative, rebuildable, drift visible, in-flight view across
components, per-unit lookup). This document adds only what that one does not cover, for two needs:
(1) **overlap** — a new idea, ledger or plan touching ground another in-flight initiative already
covers, today found only by chance (the `document-ingestion` overlap that opened this topic);
(2) **parallel runs** — telling which units can be dispatched now without colliding.

## Problem & current state
Ledgers record a unit's owner, status and deliverable but not what it needs or what it touches;
dependencies live in the plan's step table. Nothing says an agent must check what is in flight
before starting related work. The in-flight view from `coordination-graph.md` shows initiatives
and units, but not whether two initiatives overlap, nor which queued units are safe to run now.

## User stories
- As the stakeholder, I want any overlap between a new idea or initiative and one in flight raised
  to me, so I decide to merge, sequence or run independently.
- As a `tico`, `teco` or `architect`, I want to check what is in flight for overlap before I
  start, so adjacent work surfaces early.
- As a `teco`, I want to know which queued units are ready to run in parallel without colliding.

## Functional requirements
Terms and mechanisms not defined here come from `coordination-graph.md`: ledger mirror and
freshness (its FR-2 to FR-5), the in-flight view (FR-6), leaving the view when `archived` (FR-9),
and "touched areas" as files, graph keys or facts (FR-13, FR-14).

- **FR-1** — Each in-flight entry carries a one-line description of what the initiative is, so
  overlap can be judged without reading the ledger. (Its ledger path is already required by
  `coordination-graph.md` AC-5.)
- **FR-2** — `tico` (requirements interview), `teco` (before opening a ledger) and `architect`
  (while planning) check the in-flight view for overlap with the work in hand. `teco`'s own
  ledger-opening conditions apply (`claude/teco/teco.md`).
- **FR-3** — On a plausible overlap the finder stops and raises it to the stakeholder; it never
  decides to merge, sequence or run independently.
- **FR-4** — A ledger unit carries, as recorded facts, the units it **needs** and the **areas it
  touches**, in the same sense as `coordination-graph.md` FR-14. A unit whose facts are missing or
  unparseable stays visible in the in-flight view, is excluded from FR-5, and is listed with what
  is missing; it is never silently dropped. An explicit "none" is a recorded fact, not a missing
  one.
- **FR-5** — A ready-now view lists units that are `queued` and whose needs are all met, and flags
  any that share a touched area with a unit already running. A need is met when that unit is
  `accepted`, including a unit in another ledger, and including one whose ledger is archived (its
  status is read from the record, not only from the in-flight view). A need that cannot be found,
  or whose status is unknown, is listed as unresolvable and the unit is not ready. A unit is
  running while `in-flight`, `delivered`, `gated` or `paused`. Two units overlap as in
  `coordination-graph.md` FR-13(b). It flags; it does not decide.
- **FR-6** — Any reader can obtain the current text of a ledger's doc-impact scan list and of its
  closing report from the view, each addressable on its own. Whether the view points to the file, holds a copy, or both is optional and
  unprescribed. A copy is never authoritative and shows its sync time (its FR-5).
- **FR-7** — Applies to ledgers opened after this ships. Existing ledgers gain FR-4 facts as each
  is next touched, starting with the `connected-code-graph` ledger; `teco` records them.

## Out of scope
- An actively-scanning watcher that flags overlaps on its own.
- Registering a still-`Interviewing` requirements document. Two concurrent interviews colliding
  is not caught by this mechanism. Noted, not solved here.
- Automatic conflict resolution of any kind (FR-3); an automated similarity score for overlap.
- Any change to how a single initiative's units are sequenced, delegated or reviewed.
- Everything `coordination-graph.md` already specifies (mirror, freshness, in-flight view).
- Block format, ingest trigger, graph home, schema, write path and any file-reading service for
  FR-6: `architect` decisions; a file-reading service is a new capability and `security-expert`
  should weigh its path scoping first.
- Editing the agent definitions that carry the FR-2 and FR-4 duties (`tico`, `teco`, `architect`):
  routed through `cobb`, and FR-2 and FR-4 for new ledgers depend on it.

## Acceptance criteria
- **AC-1** (FR-1) — Given an in-flight initiative, when the view is queried, then its entry shows a
  one-line description and its ledger path.
- **AC-2** (FR-2, FR-3) — Given two initiatives with overlapping scope, when a `tico` interview, a
  `teco` kickoff and an `architect` plan for a third, related piece of work each runs its check,
  then each surfaces the existing initiative and raises it to the stakeholder without resolving it.
- **AC-3** (FR-4) — Given a unit with a malformed `needs` fact, when the view is queried, then the
  unit appears, is absent from the ready-now view, and is listed with what is missing.
- **AC-4** (FR-5) — Given running unit R (`in-flight`) and queued units: A (all needs `accepted`,
  shares an area with R), B (all needs `accepted`, shares no area), C (one need not `accepted`), D
  (needs only a `accepted` unit in another ledger) and E (needs a unit that cannot be found), then
  the view lists A, B and D; flags A against R and neither B nor D; does not list C; and lists E as
  unresolvable. Repeating with R `delivered`, `gated` or `paused` still flags A; with R `accepted`
  it does not. A need on a unit of an archived ledger counts as met.
- **AC-5** (FR-6) — Given a closing report edited in the file, when a reader asks the view for it,
  then it obtains the edited text, and any copy shown carries its sync time.
- **AC-6** (FR-7) — Given the `connected-code-graph` ledger after `teco` records FR-4 facts, then
  its units, status and needs are queryable from the view.

## Open questions
None.

## Decision log
2026-08-23 — Opened. Stakeholder framed the need as "our work units are getting bigger, we need
a project manager." Clarified via options: the gap is **cross-initiative oversight** — a role
above individual `teco` instances, watching multiple concurrent initiatives for overlap/conflict
— not a request to change how a single initiative's own build gets coordinated (that's already
`teco`'s job, unchanged).
2026-08-23 — Detection mechanism preference: **a shared, checked-on-demand registry** of
in-flight initiatives, not an actively-scanning watcher role. Recorded as the stakeholder's
preferred shape; the underlying requirement is that an overlap like today's is discoverable
without a one-off manual read of another instance's ledger — the specific mechanism (registry
vs. something else) is still, formally, an architect-facing design choice, but the stakeholder
has a clear preference here so it's captured directly.
2026-08-23 — Registration ownership: `teco` itself, automatically, at the point it already writes
a coordination ledger — matches how `teco` already treats doc updates as part of every unit's
done-condition (root `AGENTS.md`), no new manual step for a human or another agent to remember.
2026-08-23 — Consultation points: all three offered were selected — `tico` during a requirements
interview, `teco` before opening a new coordination ledger, and `architect` while producing a
plan. Each catches the same class of overlap at a different, progressively later stage (idea →
build kickoff → design).
2026-08-23 — On overlap found: always escalate to the stakeholder — mirrors exactly how this
session's own overlap (falkor-chat business-entities vs. document-ingestion) was actually
handled: investigated, then brought to the stakeholder, not resolved unilaterally by the finder.
2026-08-23 — Registry scope: only initiatives that already cross `teco`'s own ledger threshold
(3+ units or any gated unit), repo-wide. Deliberately narrower than also listing in-progress
`tico` interviews — accepted as a real, explicit gap (two concurrent interviews could still
collide unnoticed) rather than expanding scope to close it now.
2026-08-23 — Entry lifecycle: removed on completion, not kept as closed history — the record
stays a small "what's live right now" view; the ledger itself remains the historical record of a
finished initiative.
2026-08-29 — Storage-location preference surfaced during readback: co-locate with `kaizen_team`
(possibly renamed to reflect broader agent-team scope). Captured as a non-binding preference for
the architect (see Preferences section) — storage mechanism stays an architect design decision
per Out of scope; not treated as reopening that scope call.
2026-08-29 — Full readback delivered and confirmed by stakeholder. Status flipped to Ready for
design.
2026-08-29 — Reopened before any downstream architect work started. Stakeholder wants the concept
broadened beyond the lightweight overlap registry: `teco`'s full coordination ledger (the
unit-by-unit table, not just a pointer to it) moved into the graph. Since no plan/build exists yet
against v1, revised in place rather than forked into a successor document — Status reverts to
Interviewing pending the broadened scope; Out-of-scope's "any change to how units get sequenced,
delegated, or reviewed" line and FR-8's ledger-is-a-pointer framing are both under active
re-examination in this pass, not carried forward as settled.
2026-08-29 — Does the coordination ledger move fully into the graph, or does the graph only carry
a pointer to a markdown ledger as before? → Fully into the graph — confirmed. Exact boundary (just
the unit rows vs. the whole coordination doc including the doc-impact scan list and closing
report) still open; paused before that question was put, session paused by stakeholder request.
2026-11-02 — Overturns the 2026-08-29 "fully into the graph" entry. Is the graph the home of the
ledger, or derived from it? → The markdown ledger in git is the source of truth; the graph is
derived by incremental ingest on file change. Reason: keeps git history and review and makes the
graph rebuildable. Affects FR-2, FR-4, AC-6, AC-7.
2026-11-02 — What of the ledger is machine-readable (former Q1)? → A block per unit (id, owner,
status, deliverable, gate verdict, needs, resources); prose stays free text. FR-3.
2026-11-02 — Does a separate small registry survive (former Q2)? → No; the ledger-derived
initiative nodes are the registry. FR-1, FR-9.
2026-11-02 — Migrate existing ledgers (former Q3)? → No; prospective only, with the
`connected-code-graph` ledger as first adopter. FR-10, AC-9.
2026-11-02 — Does the graph support parallel dispatch (new)? → Yes: dependencies and resources are
queryable, a ready-now view flags collisions, and never decides. FR-3, FR-6, AC-8.
2026-11-02 — Prose as a text property on the node, or a pointer to the file (new)? → Pointer only;
no copy of the prose in the graph, any reader obtains the current text from the record. A
file-reading service is optional. Refines FR-3's "stays free text in the file". FR-9, FR-11, AC-10.
2026-11-02 — Overturns the previous entry ("Pointer only"). Pointer or copy of the prose? → No
preference; both valid and both optional, left to the architect. The requirement is only that a
reader can obtain the current text from the record, and that a copy is never authoritative. FR-11,
AC-10.
2026-11-02 — The analyst review (`docs/reviews/cross-initiative-coordination.md` Pass 1) found
that `claude/docs/requirements/coordination-graph.md` already covers the ledger mirror. Merge,
extend or supersede? → Extend: this document keeps only the delta (overlap check, unit needs and
areas, ready-now view, prose access). FR-1 to FR-7 and AC-1 to AC-6 replace v3's.
2026-11-02 — Overturns the 2026-08-23 "Detection mechanism" and "Registry scope" entries as
wording only: the in-flight view of `coordination-graph.md` is the registry, with no separate
index. The 2026-08-29 "Fully into the graph" entry was already overturned above; the
file-as-truth rule now lives in `coordination-graph.md` FR-2 to FR-5.
2026-11-02 — Which statuses count as running? → `in-flight`, `delivered` and `gated`; a unit holds
its areas until accepted. FR-5.
2026-11-02 — When is a need met? → When the needed unit is `accepted`; a need on a unit in another
ledger counts. FR-5.
2026-11-02 — Also overturns the 2026-08-23 "Registration ownership" and "Entry lifecycle"
entries: entries now come from the file sync, and leave when the ledger is `archived`
(`coordination-graph.md` FR-9).
2026-11-02 — A need on a unit whose ledger is archived? → Counts as met; its status is read from
the record. A need that cannot be found is listed as unresolvable. FR-5.
2026-11-02 — Does a `paused` unit hold its areas? → Yes. FR-5.
2026-11-03 — Readback of v4 delivered; confirm and flip to Ready for design? → Confirmed by the stakeholder. Status flipped to Ready for design (latest `analyst` verdict: approve with suggestions, Pass 2). The last edit round after Pass 2 was not re-reviewed. FR-1 to FR-7.
