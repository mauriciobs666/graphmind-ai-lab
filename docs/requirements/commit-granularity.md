# Commit granularity — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** — (M-) · **Last updated:** 2026-09-13

## Intent
Stakeholder observes hundreds of commits across the team over recent days and suspects the git
history's *shape* is wrong — narrowed (2026-09-13) to a git-hygiene concern specifically, not the
amount of underlying process work itself.

## Problem & current state
`claude/AGENTS.md`'s "Git-commit authority" section currently mandates one artifact per commit
(path-limited `git commit -F - -- <path>`), justified by a documented 2026-08-21 incident: two
agents committing concurrently against a shared index, where a batched commit swept up another
agent's uncommitted, unrelated work. Separately, `claude/docs/plans/prompt-waste-reduction.md`
(§4.0, 2026-08-23 stakeholder ruling) already granted "one complete unit per commit" — but scoped
explicitly to that plan's own live prompt-editing rollout (single-agent, sequential; the working
tree there is production via `~/.claude/agents/<name>` symlinks). That precedent has never been
generalized to the team's ordinary docs-only coordination work (`teco`/`tico`-run chains: plan →
review → revision → re-review, each often landing as its own commit).

Quantitative snapshot gathered during investigation (14-day window): ~868 commits, 602 single-file,
~76% docs-prefixed; `model-bench` alone shows 323 commits against 50 feat/fix-labeled ones.

## User stories
- As the stakeholder, I want a docs-only coordination chain (requirements → plan → review →
  revision → re-review) to land as one commit once it completes, so that reading the git log for
  a feature shows one meaningful unit of work instead of one commit per intermediate round.
- As the stakeholder, I want a single small document (a header-status flip, one markdown file with
  no narrative) to not stand alone as its own commit when it's part of a larger unit of work.
- As the stakeholder, I want to be able to tell, just from looking at a dirty working tree, that
  pending files belong to a chain that's intentionally still in progress — not something missed.

## Functional requirements
- FR-1: A docs-only coordination chain (no source/tests/config touched) commits its constituent
  documents as a single commit once the chain reaches its terminal state (accepted / gated-closed /
  handed off to implementation) — not one commit per intermediate gate round, revision, or
  re-review pass.
- FR-2: This applies only to docs-only chains (requirements/plans/reviews, with or without a formal
  coordination ledger — a small 2-unit conversation-held chain batches the same as a ledger-tracked
  one). Code-implementation chains (`teco` coordinating `coder`/`tdd-engineer`) keep their current
  per-verified-unit commit granularity — out of scope here.
- FR-3: A chain that pauses mid-way (e.g. waiting on a review) across a session boundary leaves its
  pending documents uncommitted — the working tree is expected to stay dirty for the duration of
  the pause. This is deliberate, not a sign anything was missed.
- FR-4: The single terminal commit still stages by explicit path (never a blanket `git add -A`),
  preserving the existing concurrency-safety discipline — the batching applies to "how many
  commits," not to how paths are staged.

## Out of scope
- Rewriting existing git history (settled earlier: footer-cleanup and any commit-shape change are
  going-forward only).
- The volume/amount of underlying process work itself (explicitly ruled out of scope — stakeholder
  confirmed the concern is git hygiene, not process overhead, 2026-09-13).
- Code-implementation coordination chains (`teco` + `coder`/`tdd-engineer`) — per-verified-unit
  commit granularity there is unchanged (FR-2).
- Any new proactive-notification mechanism for chain status — the existing coordination ledger's
  `Status`/Notes fields are confirmed sufficient; the stakeholder checks it when curious rather than
  being told.

## Acceptance criteria
- AC-1: Given a docs-only chain (ledger-tracked or conversation-held) with multiple units and/or
  gate rounds, when the chain reaches its terminal state (accepted / gated-closed / handed off),
  then every constituent document lands in exactly one commit, staged by explicit path.
- AC-2: Given a chain that pauses waiting on a review or a stakeholder decision, when a session
  ends mid-pause, then the working tree remains dirty with the chain's pending files — no interim
  commit is made solely because the session is ending.
- AC-3: Given a paused chain with a coordination ledger, when the ledger's `Status`/Notes fields
  are read, then they state what's pending and why — confirmed as the sufficient mechanism, no new
  notification behavior required.
- AC-4: Given a code-implementation chain (`teco` + `coder`/`tdd-engineer`), when units complete
  and gate, then per-verified-unit commit practice is unchanged — this rule does not leak into
  code chains.
- AC-5: Given a chain in flight before this rule takes effect, its already-made commits are left
  untouched — the rule governs commits made going forward only (consistent with the earlier
  footer-cleanup ruling).

## Open questions
None outstanding — resolved through interview (see decision log).

## Decision log
- 2026-09-13 — Confirmed scope is git hygiene (commit granularity/shape), not process-work volume.
- 2026-09-13 — Batch unit is "one whole chain" (single commit once the chain reaches its terminal
  state), not "one commit per gate round."
- 2026-09-13 — Scope is docs-only chains only; code-implementation chains (`teco` +
  `coder`/`tdd-engineer`) keep current per-unit commit granularity, unchanged.
- 2026-09-13 — A chain that pauses mid-way leaves the working tree intentionally dirty until it
  completes; this is expected, not a sign of missed work.
- 2026-09-13 — No new proactive-notification mechanism needed; the coordination ledger's existing
  `Status`/Notes fields are sufficient for checking what's pending and why.
- 2026-09-13 — Applies to any docs-only chain regardless of whether it has a formal
  `-coordination.md` ledger or is held in conversation.
- 2026-09-13 — Applies going forward only; commits already made are left untouched.
- 2026-09-13 — Readback confirmed by stakeholder. Status flipped to Ready for design; handing off
  to `cobb` (agent-standards owner of `claude/AGENTS.md`'s commit-authority section) for the HOW.
