# Write-guard classifier-bypass gap — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (—)

## Goal
Close the gap `cobb` root-caused in `agent-permission-friction2.md` open question 3: a
`PreToolUse` hook's explicit `"allow"` does not reliably suppress the auto-mode permission
classifier's confirmation prompt for a Task/Agent-tool-delegated subagent write, contradicting
phase 1's root-cause finding (`agent-permission-friction.md` §1.3). Stakeholder wants a real fix,
not just a documented caveat — this coordination tracks that follow-on work, separate from
`tico`'s still-`Interviewing` `agent-permission-friction2.md` (which covers `coder`'s own
friction specifically; this is the deeper mechanism issue phase-2 design would otherwise inherit).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` | `a283853afe8b7e6d2` | delivered | `skills/agent-standards/claude-code.md` + `claude/cobb/kaizen/history.md`, commit `6193083` | — (skipped: low-risk doc-only addition) → — | 201k tok / 41 tool uses |
| U2 | `cobb` | `a283853afe8b7e6d2` | delivered | `claude/docs/plans/write-guard-classifier-gap.md` | `analyst` → pending | 228k+241k tok / 5+3 tool uses |
| U3 | `analyst` | pending dispatch | queued | review of U2's plan | — | — |

## Notes
- Existing `~/.claude/settings.json` blanket `Edit`/`Write`/`NotebookEdit` allow rule, flagged as
  an unresolved risk in `agent-permission-friction.md` §1.2/§10.1, confirmed **already removed**
  (file mtime 2026-08-21 23:20:56, before the 2026-08-23 friction instances that motivate this
  coordination) — verified by `teco` directly reading the file. No interference with U2's design;
  that flagged decision is closed, not reopened by this work.
- U2's design splits the guard roster: allow-list-shaped guards (`guard-doc-writes.sh` callers) are
  candidates for a `permissions.allow` `Edit(path)` supplement; `guard-broad-write.sh`'s deny-list
  shape (`tdd-engineer`) is **not** a safe candidate (rules aren't agent-scoped — a literal
  translation would blanket-open the whole repo to every session). That's a real, deliberate
  limitation of the fix, not an oversight — expect the eventual design to leave `tdd-engineer`'s
  gap open pending a different mechanism.
- U2's design also flags that settings.json `Edit(path)` rules are **not agent-scoped** — adding
  one narrows AC-4's per-agent escalation guarantee to per-path for every session, a real tradeoff
  needing explicit sign-off per glob, not a mechanical translation of the existing hook allowlists.
- Empirical confirmation that a `permissions.allow` rule actually closes the classifier gap for a
  Task-delegated write is still open — `cobb` attempted a non-invasive test (isolated worktree +
  ephemeral `--settings`, no persisted changes) and was blocked by its own session's auto-mode
  classifier from spawning the nested test process. Needs the stakeholder's own interactive
  terminal: add `"permissions":{"allow":["Edit(**/docs/reviews/**)"]}` to
  `.claude/settings.local.json`, reproduce a Task-delegated `analyst` write to that path from a
  concurrent `auto`-mode session, observe whether it still prompts.
