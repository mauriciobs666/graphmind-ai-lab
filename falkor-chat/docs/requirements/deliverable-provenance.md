# Deliverable Provenance — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (—) · **Last updated:** 2026-08-30

## Intent
`kiro/docs/requirements/kiro-vision-followups.md` (item 1) flagged a gap between the early Kiro
multi-agent vision and the built `falkor-chat` system: the vision imagined an `Artifact` node
linking a git commit/file back to the agent and task that produced it, but nothing like it exists
in the graph today. This document develops that gap into real requirements, under the name
**deliverable** (see Terminology) rather than the vision's original "artifact" term. The trigger is
not a concrete incident that already happened — it's getting ahead of the gap before more
autonomous agent work accumulates, so the repo owner can eventually query "what did this agent
actually deliver" the same way message provenance can already be queried today.

## Terminology
This document is self-contained — every term below is used consistently throughout, in place of
looser synonyms ("artifact," "produced," "emitted") that appeared in earlier drafts and in the
originating vision document.

- **Deliverable** — a durable, non-temporary thing an agent's work produces or changes: a file (in
  or out of git) or a git commit. Never a temporary/scratch file (see FR-3). Identified by a git
  commit id, a file name/path, or both (FR-2).
- **Touch** — one recorded event where a workflow step acted on a deliverable. A single deliverable
  can accumulate many touches over its lifetime (FR-6), each with its own timestamp (FR-7) and
  kind (below). "Touch" replaces "produced"/"emitted"/"recorded producing" used in earlier drafts.
- **Touch kind** — what happened in a given touch. A closed set of three values: `created` (the
  deliverable didn't exist before this touch), `modified` (an existing deliverable was changed),
  `deleted` (the deliverable was removed). Every touch records exactly one kind (FR-10).

## Problem & current state
`falkor-chat`'s graph already has a `PRODUCED` edge (`StepRun -[:PRODUCED]-> Message`, DESIGN.md
§6.2, locked D2) — but it is scoped narrowly to a workflow step's **emitted chat message**. It has
no reach into the kind of deliverable this document is about: a durable file or git commit an
agent's work produces or changes outside the chat itself. Today, answering "what did this workflow
run/step actually produce" or "who touched this file, and how" means digging through git log/diff
and hand-written coordination-doc tables — there is no graph-queryable record, and no record at all
of *what kind* of touch happened (created vs. modified vs. deleted).

## User stories
- As the repo owner, I want to look at a workflow run/step and see every deliverable it touched, so
  I can audit an agent's actual output without digging through git/chat history by hand.
- As the repo owner, I want to look at a deliverable (by commit id or file name) and find every
  agent/workflow step that touched it, so I can trace a piece of work back to its origin.
- As the repo owner, I want the full history of every touch to a given deliverable over time — not
  just the most recent one — so nothing about its provenance is silently lost.
- As the repo owner, I want to know what kind of touch each one was (created, modified, or
  deleted), so a deliverable's history reads as a real story of what happened to it, not just a
  list of anonymous events.

## Functional requirements
- **FR-1.** An agent or workflow step must be able to explicitly record a touch on a deliverable —
  a deliberate action, not something the system infers automatically from git or filesystem
  activity.
- **FR-2.** A deliverable must be identifiable by **either** a git commit id **or** a file
  name/path — whichever is available. Neither is mandatory on its own; having at least one is
  sufficient to create and later look up the record. When both are available on the same touch,
  neither takes precedence over the other — they are independent, equally valid handles for that
  one record, not merged or resolved against other records (FR-2a).
- **FR-2a.** The system does not attempt to recognize two differently-identified touches as "the
  same underlying deliverable" (e.g. a file after a rename, or a file later folded into a commit).
  Each touch's identifiers are taken at face value; no cross-identifier identity resolution.
- **FR-3.** Temporary/scratch files (files meant to be deleted, e.g. a workspace scratch file) are
  never recorded as deliverables — only durable deliverables are in scope. Exclusion is a
  **convention**, not a system-enforced check: the agent/step simply never records a touch for a
  temp file; the system does not attempt to detect or refuse one automatically (this would require
  inferring "is this a temp file" from a path/extension pattern, which is exactly the kind of
  automated inference FR-1 already rules out).
- **FR-4.** Given a workflow run or step, it must be possible to list every deliverable it touched.
- **FR-5.** Given a deliverable (by commit id or file name), it must be possible to find every
  agent/workflow step that touched it.
- **FR-6.** When the same deliverable is touched by more than one step over time, every touch
  remains independently visible — a later touch never overwrites or removes an earlier one. This
  version keeps touch history unbounded, with no retention/purge mechanism (see Out of scope) —
  accepted as a non-issue at this system's real volumes.
- **FR-7.** Every touch carries the timestamp of when it happened, so a deliverable's touch history
  can be shown/ordered chronologically.
- **FR-8.** Every distinct deliverable carries the timestamp of when it first became known to the
  system, independent of any individual touch's own timestamp.
- **FR-9.** A deliverable's provenance is scoped to a single workspace — the same file or commit is
  not expected to be touched by workflow steps running in different workspaces. Cross-workspace
  lookup is not a requirement (see Out of scope).
- **FR-10.** Every touch records its kind — `created`, `modified`, or `deleted` (see Terminology) —
  so a deliverable's history distinguishes what actually happened at each touch, not just that
  something happened.

## Out of scope
- Automatic/inferred capture of touches from git activity or filesystem watching — recording is
  always an explicit, deliberate act by the agent/step (FR-1).
- The `claude/` subagent team's deliverables (e.g. via the `kaizen_team` graph) — this document is
  scoped to agents participating in `falkor-chat` workflows only, per the decision log.
- Temporary/scratch files (FR-3).
- Turn-taking/backoff among multiple simultaneously-responding agents — tracked separately as item
  2 of `kiro/docs/requirements/kiro-vision-followups.md`, not re-specified here.
- Real-time push — tracked separately at `falkor-chat/docs/BACKLOG.md` K-018, not re-specified here.
- The exact graph shape (node vs. edge, how it relates to the existing `PRODUCED`/`EMITTED`
  precedents) — that is the architect's/graph-dba's design call, not specified here.
- Cross-identifier identity resolution — recognizing a renamed file, or a file later folded into a
  commit, as "the same" deliverable it was before (FR-2a). Deliberately not attempted; this is the
  kind of problem the graph's existing entity-fusion mechanism (`SAME_AS`) solves elsewhere, and
  reusing/extending that here would be a materially bigger feature than this document's scope.
- Cross-workspace deliverable lookup (FR-9) — a deliverable is confirmed to realistically live
  within one workspace at a time; tracing the same file/commit across multiple workspaces is not a
  requirement. (This was flagged as a real design fork during an architect consult — see decision
  log — and resolved by confirming the single-workspace assumption rather than designing for the
  cross-workspace case.)
- Retention, purging, or archiving of touch history (FR-6) — this version keeps every touch
  forever, accepted as a non-issue at this system's real volumes (architect consult, 2026-08-29). A
  dedicated purge-and-archive tool/capability may be designed as a **separate, later feature** if
  real volume ever justifies it — not attempted here.

## Acceptance criteria
- **AC-1.** Given a workflow step records a `created` touch on a file, when that step is queried
  afterward, then the file appears among its touched deliverables.
- **AC-2.** Given a step records a touch using only a commit id (no file name), or only a file name
  (no commit id), when that deliverable is queried afterward, then the record exists and is
  findable either way — no requirement to supply both.
- **AC-3.** Given two different workflow steps each touch the same deliverable at different times,
  when that deliverable is queried, then both touches appear with their own timestamp, in
  chronological order, and neither is overwritten or removed by the other.
- **AC-4.** Given a deliverable is touched for the first time, when it is queried afterward, then it
  reports the timestamp of when it first became known to the system, distinct from the timestamp of
  any later touch to it.
- **AC-5.** Given a workflow run, when its touched deliverables are queried, then the result
  includes every deliverable any of its steps touched, traceable back to the specific step.
- **AC-6.** Given a deliverable is `created` by one step and later `modified` by another, when its
  touch history is queried, then each touch reports its own kind (`created`, `modified`) alongside
  its timestamp and the step that made it — the two touches are distinguishable, not just two
  identical anonymous entries.

## Open questions
None outstanding — see decision log for the resolution of the temp-file-refusal and
retention/volume questions.

## Decision log
2026-08-29 — Split out of `kiro/docs/requirements/kiro-vision-followups.md` item 1 into its own
requirements document → confirmed; that document now points here instead of carrying this scope
itself.
2026-08-29 — Is there a concrete incident driving this now, or is it proactive? → Proactive —
getting ahead of the gap before more autonomous agent work accumulates.
2026-08-29 — What counts as a "deliverable"? → Any durable written artifact, not just git-tracked
ones (broader than "commits and their files").
2026-08-29 — Which agents does this cover? → `falkor-chat` workflow agents only (e.g. Kiro
participating via MCP) — explicitly not the `claude/` subagent team's own `kaizen_team`-tracked
deliverables.
2026-08-29 — Which lookup direction matters — run/step → deliverable, or deliverable → run/step? →
Both directions required (FR-4, FR-5).
2026-08-29 — Should deliverables be captured automatically or self-reported? → Explicit
self-report only (FR-1) — mirrors how the existing `StepRun -[:PRODUCED]-> Message` edge is
written today, not an automated detection mechanism.
2026-08-29 — How is a deliverable identified? → Either a git commit id or a file name, whichever
is available — not both required (FR-2). Temp/scratch files are explicitly excluded (FR-3): they
are meant to be deleted, so they get no deliverable record or provenance edge.
2026-08-29 — If the same deliverable is touched by more than one step over time, what should a
lookup show? → Full history — every touch, never just the most recent (FR-6/AC-3).
2026-08-29 — Should timestamps be tracked? → Yes, on both: each individual touch (when that touch
happened, enabling chronological ordering, FR-7/AC-3) and each distinct deliverable itself (when it
was first recorded, independent of any later touch, FR-8/AC-4).
2026-08-29 — When both a commit id and file name are available for the same deliverable, does one
take precedence, and should the system try to recognize a renamed/re-identified file as the same
underlying deliverable? → No primacy between the two identifiers (FR-2), and no cross-identifier
identity resolution at all (FR-2a) — each touch's identifiers are taken at face value; recognizing
"this is the same deliverable under a new name" is explicitly out of scope, as it duplicates the
kind of problem the existing `SAME_AS` entity-fusion mechanism already solves for a different case.
2026-08-29 — Resolve the two remaining open questions (temp-file refusal mechanism, retention/
volume) now, or leave them for the architect? → Leave both for the architect to raise during
design; not resolved here.
2026-08-29 — **Architect consult** (review-shaped, initiated by `tico`): asked `architect` to read
this document plus `DESIGN.md` §6.2/§9/§11 and the `SAME_AS` entity-fusion precedent, and give a
buildability opinion — including a direct opinion on the temp-file-refusal and retention/volume
open questions — before this document is considered for "Ready for design." Findings: (a) the doc
was silent on workspace-scoping, which is a real design fork here (not merely "which node/edge
shape") because deliverable provenance is "many writers into one ever-growing shared record,"
structurally unlike the existing `reference`/`ws:{id}` definition/instance split (which works
because a `WorkflowDef` is small, shared, read-mostly, and copied once at publish time) — resolved
below by confirming single-workspace scope (FR-9); (b) opinion on the temp-file question →
convention, not enforced (folded into open question 1 above); (c) opinion on the retention/volume
question → non-issue at this system's real volumes (folded into open question 2 above); (d)
flagged FR-2/FR-2a's dual optional identifiers as expected real design work for `graph-dba` (not a
requirements defect — no change needed here); (e) flagged "produced" vs. "touched" terminology
ambiguity — resolved below by standardizing terminology and adding touch-kind tracking.
2026-08-29 — Follow-up to the architect consult: is the `reference`/`ws:{id}` "blueprints vs.
instances" split (WorkflowDef templates vs. live per-workspace state) an accurate model, and does
it help resolve the workspace-scoping question? → Architect confirmed the model is accurate (it is
literally `DESIGN.md` §4's definition/instance pattern) but does **not** transfer to deliverables:
defs are copied once at a fixed publish moment, while a deliverable accumulates new touches
indefinitely — a different problem shape (many-writers-into-one-shared-record vs.
many-readers-of-one-copied-definition). If cross-workspace were required, the architect's fallback
would be storing deliverables centrally (in `reference` or a dedicated shared graph) with each
workspace holding only a property reference, not a real edge — at the cost of FR-4/FR-5 becoming
an app-layer join instead of one graph traversal. Moot once workspace scope was confirmed (below).
2026-08-29 — Is cross-workspace deliverable touch realistic, or is this confined to one workspace
in practice? → Confined to one workspace at a time (FR-9); cross-workspace lookup is out of scope.
2026-08-29 — Should "produced" and "touched" be treated as one concept, or should create/modify/
delete be distinguished? → Distinguish them: every touch records a closed-set `touch kind`
(`created`/`modified`/`deleted`, FR-10/AC-6). Terminology across the whole document standardized on
"deliverable"/"touch"/"touch kind" (added a Terminology section), replacing the mixed
"artifact"/"produced"/"emitted"/"recorded producing" wording used in earlier drafts, so the
document reads as self-contained.
2026-08-30 — Accept the architect's recommendations on the two remaining open questions? → Yes,
both accepted as-is: (1) temp/scratch exclusion (FR-3) is enforced by convention, not a system
check — folded into FR-3's wording; (2) unbounded touch-history retention (FR-6) is accepted as a
non-issue at current real volumes, with no purge/retention mechanism in this version — a dedicated
purge-and-archive tool may be designed as a separate, later feature if real volume ever justifies
it (new Out of scope line). No open questions remain.
2026-08-30 — Final readback confirmed by stakeholder; no material assumptions unconfirmed → Status
flipped to **Ready for design**.
