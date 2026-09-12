# DevOps: headless, local-model OpenCode variant

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-09-12

## Terminology

Four related ideas that this document deliberately keeps distinct:

- **devops** — the role/capability itself (routine operational checks, environment management),
  independent of how it's run. Outlives any one implementation.
- **`claude devops`** — today's specific implementation: the interactive, Claude-Code-based agent
  at `claude/devops/devops.md`. This is the identity the long-term direction eventually retires.
- **`tank`** — the new, OpenCode-based implementation this requirement delivers (headless-only for
  this version); long-term, the one identity meant to absorb `claude devops` entirely.
- **interactive** — a human is actually present, steering turn by turn.
- **subagent** — runs inside a live Claude Code session, dispatched by another agent, with no
  human present. This is how `claude devops` can already run today (see Problem & current state) —
  neither "interactive" nor "headless."
- **headless** — no live Claude Code session at all; this version's `opencode run --agent tank`
  invocation.

## Intent

Give the `devops` role a way to run headless (no live Claude Code session, no human present) and
routed to a cheaper/local model — so a routine, lower-stakes devops check can happen without a
human opening a session, without spending frontier-model budget on work that doesn't need it. Both
properties (headless + local-model) are wanted together, not either/or, **for this initial
version**.

**Long-term direction (not this version's scope):** headless is a scope constraint on the *first*
implementation, not a permanent defining trait of `tank`. The stakeholder's intent is for `tank` to
become the **one identity** for the devops role — covering interactive and subagent use as well as
headless — with today's `claude devops` agent eventually retired and fully absorbed into `tank`.
This version delivers only the headless slice; interactive/subagent use by `tank`, and the actual
retirement of `claude devops`, are out of scope here (see Out of scope) but recorded so the design
doesn't foreclose the merge later.

## Problem & current state

The `claude devops` agent (`claude/devops/devops.md`) today only runs interactively or as a
subagent — always inside a live Claude Code session, always against a metered cloud model. There
is no way to get a routine check (health/hygiene, disk usage, container status, etc.) done
without a human present, and no way to route that kind of lower-stakes, routine work away from
frontier-model spend. OpenCode's non-interactive `opencode run --agent <name>` invocation is
confirmed working in this repo as the mechanism, and OpenCode's LM Studio provider support is the
confirmed local-model path, documented in `opencode/docs/manuals/local-llm.md`. `tank` is being
built greenfield: `opencode/`'s former custom agents are retired (`deprecated/opencode/`, preserved
for reference only, never a pattern to extend or copy from), so `tank` isn't built on or modeled
after any surviving OpenCode agent — only on the still-live LM Studio setup the manual documents.

No scheduler (cron, systemd timer, or similar) exists anywhere in this repo today, and building
one is explicitly **not** part of this requirement (see Out of scope) — `mcp-monitor` exists but
is an event/content-triggered watcher, not a wall-clock scheduler, and isn't a fit either.

## User stories

- As the stakeholder, I want to run a devops health/hygiene check without opening a live Claude
  Code session, so that I can get a routine report on demand with less friction.
- As the stakeholder, I want that run to use a local/cheap model rather than a metered cloud
  model, so that routine checks don't consume frontier-model budget.

## Functional requirements

- **FR-1.** The `devops` role must be runnable headlessly, via OpenCode's non-interactive
  `opencode run` invocation — without a live Claude Code session or a human present.
- **FR-2.** The headless run must be routed to a local model (via OpenCode's LM Studio provider
  support) rather than the metered cloud model used by `claude devops` today.
- **FR-3.** There is one canonical prompt/persona for the devops role — currently the
  `claude devops` prompt (`claude/devops/devops.md`) — shared across the interactive, subagent, and
  headless (`tank`) forms, not independently maintained copies.
- **FR-4.** The headless variant's OpenCode agent identifier is `tank` (stakeholder's naming
  choice, not `claude devops`).
- **FR-5.** The first headless check is read-only: it inspects and reports (container/service
  status, disk usage, dangling images/volumes, stale build cache, and similar hygiene signals) and
  attempts no changes on its own initiative.
- **FR-5a.** In addition to the read-only check, `tank` can bring up a demo/dev environment
  headlessly when asked to — bring-up is additive, not destructive, so it is not subject to the
  auto-deny safety net (FR-6/FR-7).
- **FR-5b.** `tank` can also shut down / release the resources of an environment **it itself
  brought up** headlessly (a symmetric bring-up → teardown lifecycle for the same environment) —
  this is the one explicit carve-out from FR-6/7's auto-deny posture. Any other destructive or
  shared-state operation — a volume wipe, a FalkorDB flush, removing something it didn't start
  itself — is not covered by this carve-out and stays auto-denied and reported per FR-6/7.
- **FR-6.** When a headless run encounters a destructive or shared-state operation (the same class
  `claude devops` approval-gates today via `guard-destructive-ops.sh`), it must refuse the action
  automatically and report what it would have done — never attempt any form of unattended
  approval.
- **FR-7.** The destructive-ops safety net is deny-by-default on ambiguity: if a command can't be
  confidently classified as safe, the headless run refuses and reports it rather than proceeding.
  This applies generally, not only to the read-only first check — a future headless task that
  isn't purely read-only inherits the same net.
- **FR-8.** The local-model routing config is scoped to this repo (`graphmind-ai-lab`) for this
  version, not global to every project the interactive `claude devops` agent reaches.
- **FR-9.** The headless run is not required to write to the shared `kaizen_team` learning graph
  in this version (see Out of scope).

## Out of scope

- **The actual retirement/merge of `claude devops` into `tank`.** This version delivers only the
  headless slice. Full replacement (one identity across interactive, subagent, and headless use) is
  the long-term direction (see Intent) but not something this version's design needs to complete —
  only to avoid foreclosing.
- **A wall-clock scheduler (cron, systemd timer, or similar).** This requirement delivers the
  headless + local-model *capability*; actually triggering it on a cadence is deferred until
  there's a proven check worth scheduling.
- **Kaizen learning capture from a headless run.** Blocked on backlog C-310 (OpenCode has no MCP
  wiring in this repo). Accepted as a gap for this version; a headless run's findings live only in
  its own report output. Revisit once/if C-310 lands.
- **Global (every-project) local-model config.** Deferred past this version — see FR-8.
- **Any headless task beyond the read-only health/hygiene check and demo-environment
  bring-up/teardown of its own environments.** Later checks are a separate follow-on, not scoped
  here.
- **Teardown of anything `tank` didn't bring up itself, and any other destructive/shared-state
  operation** (volume wipes, FalkorDB flush, removing a container it didn't start). Stays gated
  behind the destructive-ops safety net (FR-6/7) — no unattended action there, in this or any
  future version, without a separate decision to relax that posture further.

## Acceptance criteria

- Given OpenCode is configured for this repo, when `opencode run --agent tank` is invoked with no
  live Claude Code session and no human present, then it completes and produces a health/hygiene
  report without requiring any manual approval.
- Given the same invocation, when the local LM Studio provider is reachable, then the run uses the
  local model, not a metered cloud model.
- Given a destructive or shared-state operation would otherwise be attempted (e.g. a volume wipe,
  a FalkorDB flush), when it's encountered during a headless run, then the action is refused
  automatically and reported — never attempted, and never silently approved.
- Given a command the safety net cannot confidently classify as safe, when it's encountered during
  a headless run, then the run refuses and reports it rather than proceeding.
- Given a request to bring up a demo/dev environment, when made to a headless `tank` run, then it
  brings the environment up without requiring manual approval (bring-up is not
  destructive-shaped).
- Given `tank` brought an environment up itself in a headless run, when asked to shut it down /
  release its resources, then it does so without requiring manual approval (the one carved-out
  destructive action, FR-5b).
- Given a request to tear down or otherwise destructively act on something `tank` did not itself
  bring up, when made to a headless `tank` run, then it refuses automatically and reports what it
  would have done, same as any other destructive op.
- Given the `claude devops` prompt is later edited, when the change lands, then the interactive,
  subagent, and headless (`tank`) forms all reflect it — there is exactly one source of truth for
  the persona.

## Open questions

*(none — all resolved this session; see Decision log)*

## Decision log

2026-09-02 — Pre-interview capture reviewed against the repo (backlog C-310, `severino/` as
existing OpenCode+LM-Studio precedent, no existing scheduler). No prior decision found that this
request would reverse. Interview starting from the drafted constraints below, treated as already
confirmed with the stakeholder in the prior conversation that produced the draft:
- Headless + local-model are both wanted together (not either/or).
- One canonical prompt/persona, shared — not two that drift.
- Headless destructive-ops posture: auto-deny, report only (no unattended-approval mechanism).

2026-09-02 — First headless task scope → read-only health/hygiene report; no auto-fixes, even
non-destructive ones, in v1.

2026-09-02 — Cadence → none for v1; no scheduler is built as part of this requirement. Scheduling
is deferred to a later follow-on once a check is proven worth automating on a cadence.

2026-09-02 — Kaizen-capture gap (blocked on C-310) → acceptable gap for v1. Not a blocker on this
requirement; not given an interim workaround either. Tracked under existing C-310, not duplicated
here.

2026-09-02 — Local-model config scope → this repo only for v1, matching the `severino/` precedent
of project-scoped `opencode.json` over a global config. Widening to every project is a possible
later step, not committed to.

2026-09-02 — Stakeholder named the headless OpenCode agent identifier: `tank` (not `devops`).
Recorded as their naming choice.

2026-09-02 — Destructive-ops safety-net risk tolerance → zero tolerance for a slip-through:
deny-by-default on any command the guard can't confidently classify as safe. Applies to the
capability generally, not just the read-only first check.

2026-09-02 — Demo/dev environment bring-up → in scope for v1 alongside the read-only check
(FR-5a): `tank` can bring one up headlessly since bring-up isn't destructive-shaped. Teardown
stays out of scope, gated by the same destructive-ops safety net as any other shared-state-
destructive op (FR-6/7) — no unattended teardown, v1 or later, without a separate decision to
relax that posture. *(Superseded later this same session — see the next entry.)*

2026-09-02 — **Reversal of the entry immediately above.** The stakeholder asked for `tank` to also
be able to "shut down and release the resources" of an environment — i.e. headless teardown, not
just bring-up. Clarified the scope: this is **not** a general relaxation of the auto-deny
destructive-ops posture (FR-6/7 stand as-is for everything else). It is a narrow, symmetric
carve-out — new **FR-5b** — an environment `tank` itself brought up headlessly, `tank` may also
tear back down headlessly. Teardown of anything it didn't start itself, or any other
destructive/shared-state op, is unaffected and still auto-denies + reports.

2026-09-02 — Stakeholder confirmed the full readback (intent, FRs, out of scope, acceptance
criteria) with no open questions remaining. `Status` flipped to **Ready for design**.

2026-09-12 — Stakeholder asked to reopen the document. No `architect` plan exists yet, so nothing
downstream has consumed it. `Status` flipped back to **Interviewing**; awaiting the stakeholder's
change.

2026-09-12 — Stakeholder clarified intent: headless + non-interactive are scope constraints on
this *first* version, not permanent traits of `tank`. Long-term, `tank` is meant to **replace**
today's interactive `devops` agent, not run alongside it forever as a second mode. Intent section
rewritten to carry this; exact shape of "replace" was an open question.

2026-09-12 — "Replace" means one agent identity for both modes: `tank` eventually absorbs
interactive use too, and the `devops` identity is retired. Not two identities coexisting
indefinitely. Intent and Out of scope updated; the actual merge/retirement stays out of scope for
this version, recorded only so this version's design doesn't foreclose it.

2026-09-12 — No concrete trigger for moving toward the full merge — open-ended, revisit once this
first headless/local-model slice is proven out. No further action needed on this point.

2026-09-12 — Stakeholder flagged that mixing "interactive"/"headless" with "devops"/"tank" was
becoming ambiguous, given the newly-recorded long-term merge direction. Terminology settled and a
new **Terminology** section added: **devops** = the role/capability (outlives any one
implementation); **`claude devops`** = today's specific interactive/subagent Claude-Code identity
at `claude/devops/devops.md` (the one being retired); **`tank`** = the new/eventual identity;
**interactive** = a human actually present, turn by turn; **subagent** = a live Claude Code
session with no human present (how `claude devops` can already run today, distinct from both
interactive and headless); **headless** = no live Claude Code session at all. Every ambiguous
bare "devops" referring to the specific identity was rewritten to "`claude devops`" throughout
Intent, Problem & current state, FR-3/4/8, Out of scope, and the acceptance criteria; bare
"devops" referring to the role/capability was left as-is.

2026-09-12 — Stakeholder also asked to rename "interactive approval" (in the acceptance criteria)
to "manual approval," since it reused the word "interactive" in a different sense (a live
permission prompt) from the identity-mode term just defined in Terminology. All three occurrences
updated.

2026-09-12 — `cobb`: mechanical path fix only, made necessary by an unrelated cleanup — `severino/`
and `opencode/local-llm.md` were relocated to `deprecated/opencode/agents/severino/` and
`opencode/docs/manuals/local-llm.md` respectively; the Problem & current state citation updated to
match. No content/decision change.

2026-09-12 — Stakeholder (on behalf of `cobb`): `opencode/`'s former custom agents (`severino`,
`rpg`, `coding-senior`) are now fully deprecated to make `tank` a greenfield build — asked to
remove references to them from the document. Problem & current state no longer cites the retired
`severino/` agent as `tank`'s precedent/pattern; it still cites `opencode/docs/manuals/local-llm.md`
(the one artifact that stays live) as the source for the local-model setup. The historical mentions
of `severino/` in this Decision log (2026-09-02 entries, and the mechanical path-fix entry
immediately above) are left as-is — an append-only record of what was true at interview time, not
a claim about what's live now.
