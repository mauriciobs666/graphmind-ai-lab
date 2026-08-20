# Agent permission-escalation friction — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-20

## Intent
Reduce permission-escalation prompts that fire on legitimate, safe agent actions, without
weakening the guardrails that exist to catch a genuine accidental drift (an agent editing
something outside its remit, or a coder-type agent doing something destructive/irreversible).
The stakeholder is running a live session and will relay concrete escalation instances as they
happen; this document is being built from real evidence rather than hypothetical ones.

## Problem & current state
Two escalation mechanisms are in play, and both have been flagged as firing too often or on the
wrong things:

1. **The five doc-scoped write guards** (`architect`, `analyst`, `data-scientist`, `teco`, `tico`
   — thin wrappers over `claude/scripts/guard-doc-writes.sh`) escalate any `Write`/`Edit` outside
   a narrow per-agent allowlist to a human "ask" prompt. Each allowlist currently exempts the
   agent's own `kaizen/inbox.md` — but since the 2026-08-20 kaizen consolidation, raw learnings
   capture writes directly to the shared `kaizen_team` FalkorDB graph via `mcp__cypher__query`
   (a call the `Write|Edit`-matched guard never sees), and `kaizen/inbox.md` itself is now a
   frozen historical snapshot nobody writes to. Meanwhile `kaizen/history.md` and `kaizen/plan.md`
   — per the current documented convention, curated by `cobb`, not self-edited by the agent — are
   **not** in any of the five allowlists. The stakeholder has hit an escalation trying to get an
   agent to add to its own `history.md` (and possibly other kaizen files).
2. **`coder`** (no doc-scoped write guard; `permissionMode: acceptEdits`, otherwise Claude Code's
   default tool-permission behavior) is, per the stakeholder, asking for permission "much" more
   than wanted. Specific trigger(s) not yet pinned down — collecting live examples.
3. **The plain default "confirm before Edit/Write" prompt** fires on any agent that lacks
   `permissionMode: acceptEdits` — independent of, and in addition to, both mechanisms above.
   `coder` is the only agent in the roster with that setting today; every other agent (including
   ones with no custom write guard at all, like `cobb`, and the five with a doc-scoped guard) gets
   a manual confirmation on every `Edit`/`Write`, even for an action squarely inside its normal,
   documented remit. First live instance (below) confirms this fires even when the guard mechanism
   doesn't apply at all.

## User stories
- As a stakeholder, I don't want to be interrupted approving an agent writing to its own kaizen
  files when that's a normal, expected part of its job, now that the inbox.md capture path has
  moved to the graph.
- As a stakeholder, I want `coder` to run with looser permission requirements so it isn't stopping
  to ask for things that don't warrant an interruption — specifics pending live examples.

## Functional requirements
_(Draft — validating against further live instances before finalizing wording.)_

- **FR-1:** `cobb` editing another agent's own definition file (`<name>/<name>.md`) — its core,
  stated job — must not require a manual per-edit confirmation. Well-evidenced: three-plus
  instances (below), all confirmed legitimate, none an accidental drift.
- **FR-2 (draft, broader form of FR-1):** More generally, an agent performing a `Write`/`Edit`
  squarely within its own normal, documented remit should not require a manual per-action
  confirmation — confirmation should be reserved for actions genuinely outside an agent's remit
  or genuinely risky. Currently anchored only by the `cobb` evidence; still validating whether the
  stakeholder wants this applied team-wide or agent-by-agent as friction is actually hit (see Open
  questions).

## Instances observed (live, from the concurrent `teco` session)
1. **2026-08-20 — `cobb`, `Edit` on `claude/analyst/analyst.md`.** No custom write guard applies
   to `cobb` (unrestricted tools). Stakeholder confirmed: this was `cobb` doing its normal
   agent-maintenance job; approved. Mechanism: default Claude Code "confirm before Edit" prompt
   (mechanism 3 above), not a custom guard hook. → supports FR-1.
2. **2026-08-20 — `cobb`, `Edit` on `claude/data-scientist/data-scientist.md`.** Same shape as
   instance 1 (no custom guard on `cobb`; default confirm-before-Edit prompt). Confirmed by the
   stakeholder as legitimate.
3. **2026-08-20 — several further `cobb` edits, all to other agents' own system-prompt files**
   (`<name>/<name>.md`). Stakeholder: "there were several from cobb all while trying to edit
   other agents' system prompts which is his purpose" — every one legitimate, none an accidental
   drift. Pattern is now well-evidenced: `cobb` editing another agent's own `<name>.md` is its core
   job, not an edge case.

## Out of scope
_(TBD — likely candidates to confirm with the stakeholder: any change to the destructive-ops
guards on `devops`/`graph-dba`/`qa-engineer`; any change to git-commit authority scoping.)_

## Acceptance criteria
_(TBD — will be derived per-FR once the FRs are settled.)_

## Open questions
1. For the kaizen-file thread: does "history and other relevant files" mean `kaizen/history.md`
   only, or `kaizen/plan.md` too — anything else? (asked, awaiting answer)
2. Is this a responsibility change (agents self-log their own history/plan entries going forward,
   instead of `cobb` curating) or purely a guard-widening (edits still land the same way, they
   just stop triggering an "ask")? (asked, awaiting answer)
3. For `coder`: which specific actions are triggering unwanted prompts? (awaiting live examples)
4. Should any loosening apply uniformly to all five doc-scoped agents, or only to the ones the
   stakeholder has actually hit friction with?

## Decision log
- 2026-08-20 — Stakeholder: "my agents ask too much permission when editing plans reviews and
  other files" → opened as a requirements interview (Mode 1); grounded in the existing
  `guard-doc-writes.sh` design (architect kaizen, 2026-07-08) before asking anything.
- 2026-08-20 — Stakeholder: kaizen inbox is now on the graph; wants agents able to add to
  `history.md` "and other relevant files"; separately, `coder` needs much looser permission rules
  → recorded as two threads in this doc; clarifying questions asked, not yet answered.
- 2026-08-20 — Stakeholder is running a concurrent `teco` session and will relay each
  permission-escalation instance here as it happens, rather than answering hypothetically →
  interview proceeds evidence-first; this doc updates per instance.
