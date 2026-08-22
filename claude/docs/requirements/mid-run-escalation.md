# Delegate mid-run escalation — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-21

## Intent
Now that `SendMessage` is proven as a live, working mechanism for resuming a delegate (K-007,
K-013 — `claude/teco/kaizen/history.md`), the stakeholder wants to relax `teco`'s "cannot ask
mid-run" rule for genuinely high-stakes forks: instead of a delegate guessing (and possibly baking
an unconfirmed assumption into its deliverable) or only flagging the question after the fact in
its final report, it should be able to stop early with the open question as its result, have
`teco` relay that question to the stakeholder, and — once answered — be resumed via `SendMessage`
(by its `agentId`, from its own transcript) with the answer folded in, continuing rather than
restarting. The goal is fewer assumptions reaching a deliverable and a more fluid, closer-to-real-time
conversation loop between delegate, `teco`, and the stakeholder on the decisions that actually
warrant it.

## Problem & current state
`teco`'s brief-writing step tells every delegate it cannot ask questions mid-run: "blockers,
questions, and approval requests come back as its deliverable" (`claude/teco/teco.md`, step 3,
"Subagent-awareness"). This is a standing, deliberate rule — it was generalized into the brief
template during a 2026-08-11-ish review pass (`claude/teco/kaizen/history.md`, "Brief template
generalized... no-mid-run-questions"), and predates `SendMessage` being exercised for real (K-013
closed 2026-08-16). Today, if a delegate hits a genuinely undecided fork partway through a run, its
only options are: state its reasoning and make the call itself, or flag it as a decision for
`teco`/the stakeholder in its final report — either way, `teco` only learns about the fork after
the delegate has already finished its run (right or wrong).

`SendMessage` today is used in one direction — `teco` (or a peer) messaging a named delegate to
resume it. The mechanism this document settles on (see FR-3, and the decision log) keeps it that
way: a delegate doesn't need `SendMessage` in its own tool grants to use this path — it stops its
run with the open question as its result, `teco` relays the question and later resumes the *same*
delegate via `SendMessage`, addressed by the `agentId` recorded in the coordination ledger at
dispatch. That resolves what would otherwise be an open question about the several specialists
that don't currently carry `SendMessage` (`analyst`, `architect`, `data-scientist`,
`security-expert`) — they're unaffected either way, since it's `teco` doing the resuming, not them.

## User stories
- As the stakeholder, I want a delegate facing a high-stakes, genuinely-undecided fork to stop and
  relay the question through `teco` rather than guess, so a costly-to-unwind decision isn't made on
  an assumption I never got to confirm.
- As the stakeholder, I want that delegate resumed with my answer once I've given it, rather than
  re-briefed cold, so it picks up exactly where it left off instead of re-deriving context (or me
  re-explaining it).
- As `teco`, I want a clear line between "routine ambiguity the delegate should just decide" and
  "stop and ask" so the loop doesn't turn into every delegate pinging back on everything it isn't
  100% sure of.

## Functional requirements
- **FR-1 (scope of the escalation path).** A delegate may stop mid-run and return an open question
  as its result — instead of guessing or deferring the question to its final report — only for a
  **high-stakes fork**: a decision that, if guessed wrong, would change scope, touch something
  irreversible, or waste substantial downstream work. Routine ambiguity is still resolved by the
  delegate stating its reasoning and making the call itself, same as today — this path does not
  replace that one, it sits alongside it for the narrower, costlier class of decision.
- **FR-2 (relay).** When a delegate stops with an open question, `teco` relays that question to the
  stakeholder rather than answering on the delegate's behalf or silently making the call itself.
- **FR-3 (resume, not restart).** Once the stakeholder answers, `teco` resumes the *same* delegate
  (addressed by its recorded `agentId`) with the answer, rather than re-briefing a fresh delegate —
  the delegate continues from its own prior work/transcript instead of starting over.
- **FR-4 (scope of the pause).** Stopping one unit to wait on an answer does not stall the rest of
  an in-flight coordination — `teco` keeps other independent, already-dispatched units moving; only
  the unit that raised the question sits paused pending the answer.
- **FR-5 (applies team-wide).** The "stop and ask" path is available to any specialist `teco`
  coordinates, not a subset — same rule for every delegate, independent of whether that delegate's
  own tool grants happen to include `SendMessage` (it doesn't need to: `teco` is the one that
  resumes it, per FR-3).

## Out of scope
- **Replacing "state reasoning and make the call" for routine ambiguity.** That path is unchanged
  — this feature adds a narrower option for high-stakes forks (FR-1), it doesn't remove the
  existing default.
- **A fixed cap on stop-and-ask round trips.** The stakeholder explicitly declined a numeric limit
  — left to `teco`'s judgment, same as its other in-run decisions, including whether a loop itself
  has become the problem worth escalating.
- **Standalone (non-`teco`-coordinated) agent runs.** This feature only changes behavior for a
  delegate `teco` dispatched — there's no `teco` in the loop to relay a question when a specialist
  is run directly (e.g. `claude --agent architect`). Those runs keep today's behavior (no mid-run
  question path) unchanged.
- **Changing any delegate's `SendMessage` tool grant.** Not needed by this feature's chosen
  mechanism (see Problem & current state) and not requested.
- **Any new ledger-status vocabulary, brief wording, or other implementation detail.** How `teco`
  represents "paused, waiting on an answer" in its coordination ledger, and how it phrases the
  brief-template rule change, is the architect's design, not this document's.

## Acceptance criteria
- **AC-1 (FR-1, scope).** Given a delegate hits a decision that would change scope, touch something
  irreversible, or waste substantial downstream work if guessed wrong, when it's genuinely unsure,
  then it stops its run and returns the open question as its result rather than guessing or only
  flagging it in a final report. Given a delegate hits ambiguity that doesn't meet that bar, when
  it's unsure, then it still states its reasoning and makes the call itself, unchanged from today.
- **AC-2 (FR-2, relay).** Given a delegate's result is an open question, when `teco` processes that
  result, then `teco` relays the question to the stakeholder rather than answering it or silently
  deciding on the delegate's behalf.
- **AC-3 (FR-3, resume not restart).** Given the stakeholder answers a relayed question, when
  `teco` continues that unit, then `teco` addresses the same delegate via `SendMessage` using its
  recorded `agentId` (not a fresh `Agent` call), and the delegate's continued work reflects the
  answer given.
- **AC-4 (FR-4, non-blocking).** Given one unit is paused waiting on an answer, when other
  independent units in the same coordination are in flight, then those units continue to be
  dispatched and progressed without waiting on the paused unit's answer.
- **AC-5 (FR-5, team-wide).** Given any specialist `teco` coordinates, when it faces a qualifying
  high-stakes fork, then the stop-and-ask path is available to it regardless of that delegate's own
  `SendMessage` tool grant.

## Open questions
None — every thread opened this round was resolved with the stakeholder (see Decision log).

## Decision log
- 2026-08-21 — Stakeholder: read a `teco` brief and wants to improve how coordination is done "now
  that we have the SendMessage functionality," pointing specifically at the "cannot ask questions
  mid-run" process note (`claude/teco/teco.md` step 3) → opened as a requirements interview (Mode
  1); grounded in `teco.md`'s current rule and the `SendMessage` history (K-007, K-013) before
  asking anything. Confirmed no prior decision is being reversed: K-002 (2026-07-24) evaluated and
  rejected an *agent-teams* reframe, and its `SendMessage` spinoff (K-007) is about `teco`-initiated
  resume of a delegate, not a delegate proactively messaging back mid-run — this is fresh ground,
  not a re-litigation.
- 2026-08-21 — Stakeholder proposed the mechanism directly: delegate returns to `teco` to ask the
  stakeholder a question, `teco` resumes the delegate via `SendMessage` once answered → captured as
  the stakeholder's stated preference for the mechanism (FR-3's "resume, not restart" shape);
  recorded as a requirement because it matches how `SendMessage` already provably works (K-013:
  resumes from the delegate's own transcript), not because the interview presumes any particular
  implementation.
- 2026-08-21 — Asked whether "stop and ask" replaces the delegate's existing two options (guess,
  or flag in the final report) or sits alongside them → Stakeholder: **reserved for high-stakes
  forks** — routine ambiguity is still resolved by the delegate making the call itself; "stop and
  ask" is a new, narrower option for the costly-to-guess-wrong case, not a replacement for
  everything. Settles FR-1's scope.
- 2026-08-21 — Asked what happens to the rest of an in-flight coordination while one delegate waits
  on an answer → Stakeholder: **only that unit stalls**; other independent, already-dispatched
  units keep moving. Settles FR-4.
- 2026-08-21 — Asked whether "stop and ask" is available to every delegate type or only some →
  Stakeholder: **all delegates, team-wide**. Noted that this doesn't depend on a delegate's own
  `SendMessage` grant, since `teco` (not the delegate) performs the resume. Settles FR-5.
- 2026-08-21 — Asked whether there should be a cap on stop-and-ask round trips per unit →
  Stakeholder: **`teco`'s judgment**, no fixed cap — moved to Out of scope rather than a
  requirement, since it's the absence of a constraint, not a new one.
- 2026-08-21 — Readback given (Intent, FR-1..5, Out of scope, AC-1..5, empty Open questions);
  stakeholder confirmed with no changes → Status → **Ready for design**.
