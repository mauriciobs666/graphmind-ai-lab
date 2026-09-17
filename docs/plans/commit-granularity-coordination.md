# Commit granularity — closeout coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (M-)

## Context

Stakeholder asked to "start working on `docs/requirements/commit-granularity.md`." Investigation
found the requirement is **already implemented and committed**: `cobb` delivered it end-to-end in
commit `320f682` (2026-09-13, "feat(agent-standards): enforce no-footer via settings deny; batch
docs-only chain commits"), touching `claude/AGENTS.md` (new "docs-only coordination chain commits
once" paragraph), `claude/teco/teco.md`, `claude/tico/tico.md`, plus an unrelated no-footer
settings change bundled in the same commit. Full rationale trail: `claude/cobb/kaizen/history.md`,
2026-09-13 entry ("Commit granularity — docs-only chains batch to one commit").

What's actually still open, found during this investigation:

1. **No independent review gate.** `cobb`'s own kaizen entry says verification "rests on
   cross-reading the three files together, not a live run" — self-checked only. Per this team's
   default, a `cobb` agent/prompt artifact gets an `analyst` gate same as any other deliverable;
   that never happened here.
2. **No `docs/HISTORY.md` entry.** Investigated and resolved — see Ledger/Notes: `cobb` judged
   root `docs/HISTORY.md` out of scope (its header ties it to the CPG/code-graph component; `claude/`
   agent-prompt changes route to each agent's own `kaizen/history.md` by root `AGENTS.md`'s own
   convention, which already has the full record), and `analyst` independently agreed on re-check.
3. **Requirements doc `Status:` never flipped.** `docs/requirements/commit-granularity.md` still
   reads `Ready for design` despite being fully delivered — same drift pattern independently found
   this session in `docs/requirements/salesperson-ui.md` and
   `opencode/docs/requirements/devops-opencode-headless.md`.

This is a **docs-only chain** (review + history entry + a Status-field edit; no source/tests/config
touched) — per the very rule it is closing out, it commits **once**, at the end, not per unit.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `analyst` | `a1e422ede5c51a9a9` | accepted | `docs/reviews/commit-granularity.md` (2 passes) | Pass 1 → needs changes; Pass 2 → **approve with suggestions** | 125k+156k tok / 26+12 tools |
| U2 | `cobb` | `ae1fd935c91beb3cb` | accepted | fixed all 4 Pass-1 findings in `claude/tico/tico.md`, `claude/teco/teco.md`, `claude/AGENTS.md` (swept into a concurrent commit `702704f`, verified correct — see Notes), `claude/cobb/kaizen/{history,plan}.md` (K-033 corrected); declined `docs/HISTORY.md` entry, reasoned in kaizen history | U1 Pass 2 → approve with suggestions | 151k tok / 38 tools |
| U2b | `cobb` (resumed) | `ae1fd935c91beb3cb` | accepted | applied U1 Pass 2's 2 minor wording suggestions to `tico.md`/`teco.md`; addendum to same-day `claude/cobb/kaizen/history.md` entry | `teco` re-read both diffs directly → confirmed correct, no re-gate needed (suggestions-only, non-blocking) | 176k tok / 12 tools |
| U3 | `teco` | — | accepted | flipped `docs/requirements/commit-granularity.md` `Status:` → `archived` (mechanical) | n/a | — |

## Notes

- **U2's `claude/AGENTS.md` fix (finding 4) landed via a concurrent session's commit, not this
  coordination's own.** `cobb`(U2) left it uncommitted per the delegated-subagent rule; before
  hand-off, an unrelated concurrent commit (`702704f`, "chore(kaizen): distill
  frontend-engineer's 8 raw kaizen_team entries... (U2)" — a different U2, from the
  `kaizen-team-distillation-coordination.md` chain) swept it in. Independently verified by `teco`:
  `git diff 320f682 HEAD -- claude/AGENTS.md` shows the terminal-state-mapping clause present and
  correct in `HEAD`, `claude/AGENTS.md` is clean in `git status` (nothing left to commit for it).
  Content is right; only the commit message's attribution is off, and per guardrails that is not
  ours to fix by amending history. No action needed from this coordination for that file.
- **Shared-file three-way check performed** on `claude/cobb/kaizen/{history,plan}.md` (structurally
  shared with the concurrent kaizen-team-distillation coordination, which also dispatches `cobb`):
  `git diff HEAD -- <file>` for both shows a clean single-source diff (baseline == HEAD, delta is
  U2's alone) — no interleaving with the other coordination's writes to the same files.
- **Word-count figures re-verified independently a third time** (`teco`, this session, not copied
  from either `analyst` or `cobb`): `claude/AGENTS.md` 2,783→3,074 (+291), `claude/teco/teco.md`
  10,765→10,899 (+134), `claude/tico/tico.md` 5,623→5,809 (+186) — matches both prior figures
  exactly.
