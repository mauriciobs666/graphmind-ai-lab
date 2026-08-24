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
| U3 | `analyst` | `aa2dec35279344ddb` | delivered | `docs/reviews/write-guard-classifier-gap.md` | `analyst` → needs changes | 128k tok / 25 tool uses |
| U4 | `cobb` | `a283853afe8b7e6d2` | delivered | `write-guard-classifier-gap.md` v2, commit `df00237` | — | 274k tok / 4 tool uses |
| U5 | `analyst` | `aa2dec35279344ddb` | delivered | Pass 2 appended to `docs/reviews/write-guard-classifier-gap.md` | `analyst` → approve with suggestions | 148k tok / 4 tool uses |
| U6 | `teco` | — | delivered | `skills/README.md` added to §5.3/§8, per Pass 2's one Minor finding | — (trivial fix, no gate) → — | — |
| U7 | `teco` + stakeholder | `a69060e60424e0cf1` | delivered | Live empirical test: `.claude/settings.local.json` (project root) given `"Edit(**/docs/reviews/**)"`; `analyst` dispatched via `Agent` to write `claude/docs/reviews/_permission-test-scratch.md`; stakeholder directly observed the OS/CLI confirmation prompt ("Create file · from the analyst agent") fire anyway | — (empirical observation, not a review gate) → **hypothesis refuted** | 23k tok / 1 tool use (analyst side) |

## Close-out

Design gated **approve with suggestions** (`analyst`, Pass 2) as a reasoning framework, but the
empirical test it was gated on (§7) has now run and **refutes the core mechanism**: a
`permissions.allow` `Edit(path)` rule in `.claude/settings.local.json` did **not** suppress the
confirmation prompt for a Task/Agent-delegated write to a path the rule covered, even though the
target agent's own `PreToolUse` hook independently returns `"allow"` for that same path. Both
suppression mechanisms this design and phase 1 relied on — hook `"allow"` and settings.json
`Edit(path)` rules — fail to bypass the classifier for delegated writes specifically.

**Consequence:** the whole `permissions.allow`-supplement approach in `write-guard-classifier-gap.md`
§5 is not viable as designed and should not be implemented. The only remaining lever identified
across this investigation is the `defaultMode` tradeoff `cobb`'s original RCA flagged and never
resolved. Escalated to the stakeholder, who chose to investigate it (U8 below). Scratch test
artifact (`claude/docs/reviews/_permission-test-scratch.md`) was cleaned up by the same `analyst`
dispatch that created it.

| U8 | `cobb` | `a3b4c777bd3b19360` | delivered | `claude/docs/plans/permission-default-mode.md`, commit `773328c` | `analyst` → pending | 195k tok / 28 tool uses |
| U9 | `teco` | — | delivered | Fixed pre-existing `audit-team.sh` FAIL (leaked home path) in `docs/reviews/write-guard-classifier-gap.md`, commit `773328c` | — (trivial fix, no gate) → — | — |
| U10 | `analyst` | `a21ab3bb588d7f368` | delivered | `claude/docs/reviews/permission-default-mode.md` | `analyst` → approve with suggestions | 135k tok / 23 tool uses |
| U11 | `cobb` | `a3b4c777bd3b19360` | in-flight | Revision of `permission-default-mode.md` per U10's 2 Major + 1 Minor findings | — | — |

**U10 summary:** core mechanism claims verified byte-for-byte against fresh doc fetches; the
"dead configuration" finding independently re-confirmed. No blockers — recommendation to stay on
`auto` holds. Two Major findings (a false "gap in the docs" claim resolvable by elimination; an
internal inconsistency with this repo's own already-documented finding that hook `"ask"`
enforcement is unconfirmed under Auto Mode) and one Minor (undercounted `Agent`-tool-carrying
agents: 6 named vs. actually all 13). Neither Major finding overturns the recommendation. Sent
back to `cobb` (U11) to fold in, same in-place precedent as the earlier v2 revision.

| U11 | `cobb` | `a3b4c777bd3b19360` | delivered | `permission-default-mode.md` v2, commit `7fc1e8d` | — | 221k tok / 28 tool uses |
| U12 | `teco` | — | delivered | One-line fix: stale "backport open" claim in v2's Cross-references section, commit `7fc1e8d` | — (trivial fix, no gate) → — | — |

## Close-out

Both levers this whole investigation could find are now exhausted:

1. **`permissions.allow` settings-rule supplement** (`write-guard-classifier-gap.md`) — design
   reviewed to approve-with-suggestions, then **empirically refuted** (§U7): the rule does not
   suppress the confirmation prompt for a Task-delegated write, even stacked with a hook that
   independently returns `"allow"` for the same path.
2. **`defaultMode` switch away from `auto`** (`permission-default-mode.md`, v2, reviewed to
   approve-with-suggestions) — the mechanism is real and precisely documented (parent-mode
   inheritance), but the blast radius is whole-session/every-Bash-call at any persisted scope,
   with no narrower option available, and the cost-benefit doesn't clear the bar. **Recommended:
   stay on `auto`, accept the documented friction** — the same conclusion shape both investigations
   independently reached.

**Bottom line for the stakeholder:** there is no clean, narrowly-scoped fix for the
Task/Agent-delegated write classifier gap available today. The gap is now fully explained and
durably documented (`skills/agent-standards/claude-code.md`), not mysterious — but closing it
would cost more (in Bash-prompt volume, team-wide, indefinitely) than living with it. This
coordination is ready to close pending the stakeholder's sign-off on that recommendation.

**U8 summary:** precisely identifies the subagent parent-mode-inheritance mechanism
(`bypassPermissions`/`acceptEdits` on the parent takes precedence and can't be overridden;
`auto` on the parent forces `auto` on every dispatched subagent, discarding its frontmatter).
Surfaces that all 13 agents' `permissionMode: acceptEdits` frontmatter is dead configuration —
never consulted for a session's own start mode (only `--permission-mode`, `defaultMode` in a
settings file, or the built-in default are), and irrelevant at dispatch time too since the parent
is always itself in `auto`. Maps blast radius (no scope narrower than whole-session; global vs.
project vs. per-launch) and the Bash-friction trade `acceptEdits` would introduce team-wide.
**Recommends staying on `auto`, accepting the documented friction** — the same shape of
conclusion as U7's rules-approach close-out. This coordination stays **open** pending U10's
review gate and the stakeholder's final call.

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
