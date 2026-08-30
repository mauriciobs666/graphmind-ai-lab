# Cross-initiative coordination — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD) · **Version:** 2

## Intent
Multiple `teco`-led initiatives can now be in flight on this repo at the same time, each
tracked in its own coordination ledger, each scoped to its own initiative. Nothing today gives
any one of them visibility into what the others are doing. The trigger: while scoping the
falkor-chat "business entities in workflows" work into sibling requirement documents, this
`tico` instance discovered — only by manually reading another `teco` instance's coordination
ledger on request — a second, independently-running initiative (`document-ingestion`) touching
adjacent ground (structured/extracted entity data). No conflict existed this time, but nothing
structural caught the overlap; it was found by a one-off manual check, not by anything that
would catch it by default next time.

The stakeholder's own framing was "we need a project manager" — but the gap isn't in how one
initiative gets coordinated (that's `teco`'s job today, and stays that way); it's the absence of
any shared view *across* initiatives. The stakeholder's preferred shape for closing that gap is
a shared, checked-on-demand registry of what's in flight, not a new standing watcher role —
lighter-weight than a new agent, consistent with catching an overlap earlier next time rather
than only by chance.

## Problem & current state
Today, `teco` writes one coordination ledger per initiative (`<component>/docs/plans/<slug>-
coordination.md`), and each ledger is self-contained — it records that initiative's own units,
owners, and status, with no pointer to or awareness of any other initiative's ledger. There is no
single place listing what's currently in flight across the repo. Finding out that two initiatives
touch adjacent ground currently depends entirely on a person or agent happening to already know
about both and thinking to compare them — exactly what happened this session, and only because
the stakeholder mentioned the other initiative directly.

## User stories
- As the stakeholder, I want to know when a new requirements idea or a new `teco`-coordinated
  initiative overlaps with something already in flight, so I can decide whether to merge, sequence,
  or run them independently — before real design/build work happens twice or diverges.
- As a `tico` instance opening a new requirements interview, I want to check what's currently
  being built or coordinated elsewhere in the repo, so I can catch an adjacent-scope situation at
  the idea stage, before any build coordination even exists for it.
- As a `teco` instance about to open a new coordination ledger, I want to check the same shared
  view before I start, so I don't duplicate or silently diverge from another initiative's work.
- As an `architect` designing a plan, I want to check it too, so a schema/interface overlap that
  wasn't visible at the requirements or kickoff stage still gets one more chance to surface before
  code gets written.
- As the stakeholder, I want any overlap that's found to come to me, not be silently resolved or
  silently ignored by whichever agent found it — matching how this session's actual overlap was
  handled (investigated, then brought to me).

## Functional requirements
- **FR-1** — A single, shared record exists of currently in-flight `teco`-coordinated
  initiatives, reachable by any agent instance, repo-wide (not scoped to one component).
- **FR-2** — An initiative is added to the record automatically when it reaches the point `teco`
  already writes it a coordination ledger (teco's existing 3-units-or-gated threshold) — no
  separate manual registration step for the stakeholder or another agent to remember.
- **FR-3** — An initiative's record entry is removed once that initiative's coordinated work is
  done — the record reflects what's live now, not a history of everything ever coordinated.
- **FR-4** — A `tico` instance consults the record during a requirements interview, to check
  whether the idea under discussion overlaps with something already in flight.
- **FR-5** — A `teco` instance consults the record before opening a new coordination ledger, to
  check whether the initiative it's about to start overlaps with one already in flight.
- **FR-6** — An `architect` instance consults the record while producing a plan, to check whether
  the design touches ground another in-flight initiative also touches.
- **FR-7** — When any of the above finds a plausible overlap with another in-flight initiative, it
  stops and raises the question to the stakeholder rather than deciding unilaterally how to
  proceed (merge, sequence, run independently, or treat as coincidental).
- **FR-8** — An entry carries enough information for another agent to judge, without reading the
  full initiative, whether it might overlap with what it's about to do (at minimum: what the
  initiative is, and a pointer to its coordination ledger for the full detail).

## Out of scope
- An actively-scanning watcher that proactively flags overlaps on its own — the stakeholder chose
  a checked-on-demand shared record instead of a standing monitoring role.
- Registering a `tico` requirements interview itself as an entry — the record tracks
  `teco`-coordinated initiatives (FR-1/FR-2's threshold); a still-`Interviewing` requirements doc
  is not listed. (This means two concurrent `tico` interviews colliding with each other is **not**
  caught by this mechanism — a `tico` instance checks the record for existing *build* initiatives,
  but nothing yet checks one interview against another interview. Noted, not solved here.)
- Any automatic conflict resolution — merging documents, re-sequencing work, or deciding one
  initiative supersedes another are all stakeholder calls (FR-7), never made by the agent that
  found the overlap.
- An automated similarity/matching algorithm that decides what counts as "overlap" — judging
  whether two initiatives are actually related is a human/agent reading call, not a scored match.
- Any change to how a single initiative's own units get sequenced, delegated, or reviewed — that
  is `teco`'s existing job, entirely unchanged by this feature.
- The record's concrete storage mechanism/format (a doc, a graph, something else) and exact entry
  schema — an architect design decision.

## Acceptance criteria
- **AC-1** (FR-1, FR-2) — Given a `teco` instance opens a coordination ledger for a new
  initiative, when that happens, then an entry for it appears in the shared record without a
  separate manual step.
- **AC-2** (FR-3) — Given an initiative's coordinated work is complete, when that happens, then
  its entry is no longer present in the shared record.
- **AC-3** (FR-4, FR-5, FR-6) — Given two initiatives with genuinely overlapping scope, when a
  `tico` interview, a `teco` kickoff, or an `architect` plan for a third, related piece of work
  checks the record, then the existing overlapping entry/entries are discoverable from that check
  alone (without already knowing to look for them).
- **AC-4** (FR-7) — Given a plausible overlap is found via the record, when that happens, then the
  finder raises it to the stakeholder and does not resolve it unilaterally.
- **AC-5** (FR-1) — Given initiatives exist in more than one component, when the record is
  consulted, then all of them are visible from the same single record — not one record per
  component.

## Open questions
1. Exact boundary of "the ledger" now that it moves fully into the graph: just the unit table
   (Unit/Owner/Agent id/Status/Deliverable/Gate→verdict/Cost), or the whole coordination doc
   (also the documentation-impact scan list and the closing narrative report)? Unresolved —
   session paused before this was asked.
2. Given the full ledger now lives in the graph with queryable detail, does the original
   lightweight registry concept (FR-1/FR-8: minimal entry + pointer to a markdown ledger) still
   exist as a separate, smaller index, or does it collapse — the ledger nodes themselves double
   as the cross-initiative record this document was originally about? Unresolved.
3. Do already-open/historical coordination docs (many exist today, e.g.
   `falkor-chat/docs/plans/workflow-salesperson-demo-coordination.md`) get migrated into the
   graph, or does this apply prospectively only, to coordinations opened after this ships?
   Unresolved.
4. FR-1–FR-8 and the Out-of-scope/Preferences sections below still describe the **v1, pointer-
   based** registry and have not yet been rewritten for the broadened scope — pending resolution
   of open questions 1–3 above.

## Preferences (non-binding — for the architect)
- Storage location: the stakeholder suggested co-locating the record in the same FalkorDB graph
  already used for `kaizen_team` (possibly under a broader name reflecting that it holds
  everything concerning the agent team, not just kaizen entries) rather than a new graph or a doc.
  Rationale offered: both are shared, checked-on-demand state across the agent team, and the
  graph already exists. This is a preference, not a requirement — the storage mechanism/format
  remains an architect design decision (see Out of scope); the architect should also weigh that
  `kaizen_team` is already live and written to by every agent's kaizen capture, so any rename has
  a blast radius beyond this feature.

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
