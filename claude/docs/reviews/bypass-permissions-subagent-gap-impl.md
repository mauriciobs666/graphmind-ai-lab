# `bypassPermissions` doesn't suppress Task-delegated subagent Write/Edit prompts — Implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (post-M3 follow-up, not a milestone gate)

## Scope & verdict

Reviewed: the U3 implementation diff against Gen 4's twice-approved design
(`claude/docs/plans/bypass-permissions-subagent-gap.md` §4, gated by
`claude/docs/reviews/bypass-permissions-subagent-gap.md` Pass 1/Pass 2 — both already reviewed by
me and not re-reviewed here). This is a distinct implementation-gate review (U4 of
`claude/docs/plans/bypass-permissions-subagent-gap-coordination.md`); no code exists in this repo
in the traditional sense — the "code" is `.claude/settings.json` plus three documentation
artifacts. Checked: `.claude/settings.json` diff, the plan doc's in-place wording fix, the new KB
entry in `skills/agent-standards/claude-code.md`, the untracked upstream-feedback draft, and the
`cobb/kaizen/{history,plan}.md` entries, each verified against the live repo state and, where
relevant, against Claude Code's own current documentation (fetched fresh today) rather than taking
the diff's own narration at face value.

**Verdict: approve.** Every piece of this implementation does exactly what the design specified,
nothing more and nothing less. The `.claude/settings.json` diff is a single-line, surgical removal
with byte-for-byte confirmation that the three `allow` rules and the `ask` rules are untouched. The
plan doc's wording softening matches Pass 2's suggested clause verbatim. The new KB entry is
accurate against the design doc it summarizes and matches the existing 2026-08-24 entry's style. The
"dangling forward-reference" `cobb` flagged as resolved is genuinely resolved — I confirmed via
`git show HEAD` that the committed baseline lacked the stamp text entirely, so the U1 session added
the forward-reference and this U3 session filled in the body, exactly as the kaizen history
describes. The upstream-feedback draft is well-evidenced, references the existing receipt without
re-drafting it, and — notably — sidesteps Pass 2's still-soft hook-`ask` finding entirely rather than
risk overclaiming it. I additionally verified the one open technical question the brief flagged
(whether removing `defaultMode` cleanly falls through to the global `auto` default) against Claude
Code's current settings-precedence documentation: confirmed, per-key merge, no surprise.

**CPG:** not applicable — this is a config/documentation-only change to the harness's own permission
mechanics, with no code-level component in this repo to load a CPG against.

## Findings

No blockers, majors, or minors found. Two small observations, below approval threshold:

### Nit — the settings.json revert leaves no comment/marker explaining why `defaultMode` is absent

`.claude/settings.json` now simply lacks the key — correct per the design (§4.2 explicitly allows
"remove... or set back to unset"), but a future session skimming the file with no context has no
inline signal that this was a deliberate revert rather than an oversight. The KB entry
(`skills/agent-standards/claude-code.md`) is the actual authoritative record and is easy to find via
the file's own citation trail, so this is genuinely optional — JSON has no comment syntax here
anyway (`$schema`-validated strict JSON per the settings docs), so the only real option would be a
sibling doc note, which the KB entry already serves as. Not worth a follow-up.

### Nit — `claude/cobb/kaizen/plan.md`'s new parking-lot entry undersells that U4 is this review

The new bullet says "U4 (`analyst`'s implementation review) is still queued" — true at the time it
was written, but by the time a reader hits it (post-U4), it'll read as stale unless `teco`'s
milestone-close pass catches it. Minor, self-resolving once the coordination ledger and this
document exist; not a defect in what shipped.

## What's solid

- **`.claude/settings.json` diff is exactly the single line the design specified**
  (`git diff -- .claude/settings.json`): `"defaultMode": "bypassPermissions",` removed, nothing else
  touched. The three `allow` rules (`Bash`, `Edit(**)`, `mcp__cypher__query`) and all `ask` rules
  (destructive-ops patterns, `Edit(**/docs/BACKLOG.md)`) are byte-identical to before.
- **`~/.claude/settings.json` (global) confirmed live**: `permissions.defaultMode: "auto"`, nothing
  else relevant — re-checked directly, matches the design's assumption exactly.
- **Precedence subtlety the brief asked me to check: confirmed clean.** Fetched
  `code.claude.com/docs/en/settings` fresh today: "When the same key appears in more than one place,
  Claude Code uses the value from the highest level that sets it" — explicit per-key resolution, not
  whole-file replacement, and shared-project settings (`.claude/settings.json`) sits above user
  settings (`~/.claude/settings.json`) in the stack but below project-local
  (`.claude/settings.local.json`). Removing the key from the shared-project file therefore falls
  through cleanly to the user file's `"auto"` — exactly what the design assumed, no undocumented
  precedence trap. I also checked `.claude/settings.local.json` (gitignored, personal) for a
  competing `defaultMode` override that would sit *above* the shared file and silently keep bypass
  alive: none present, and its Gen-2-era leftover `Edit(**/docs/reviews/**)` rule (§4.3, flagged as
  inert hygiene) is still there, untouched, exactly as the design left it. Also checked for a
  machine-level `managed-settings.json`: none found — no org policy could be silently overriding the
  project's own choice either way.
- **Plan doc's §4.2 wording fix matches Pass 2's suggested clause verbatim**: "directly refutes" is
  now "is inconsistent with, in a headless context this test can't fully separate from mode" —
  exactly the phrasing the review asked for, in the exact spot Pass 2 flagged.
- **KB entry (`skills/agent-standards/claude-code.md`) is accurate and correctly styled.** I compared
  every substantive claim in the new "Resolution/update, 2026-09-01" bullet against the design doc it
  summarizes (background-dispatch-default-since-v2.1.232, the file-edit-vs-Bash asymmetry, the
  inconclusive isolation-lever test, the headless-mode confound caveat, the practical revert
  guidance) — all match without embellishment or omission of a caveat. Structurally it mirrors the
  2026-08-24 entry it's modeled on (same "Resolution/update, DATE" lead, numbered sub-findings,
  closing "practical guidance" paragraph).
- **The dangling forward-reference is genuinely resolved, not just reported resolved.** `git show
  HEAD:skills/agent-standards/claude-code.md` confirms the committed baseline has no
  `bypassPermissions`-related stamp text at all — the working-tree stamp (lines 32-38) and the `##
  Hooks` entry body (lines 427+) were both added in this uncommitted work, consistent with the
  kaizen history's account that an earlier Gen 4 session added the stamp while a later one (this U3)
  filled in the body. The stamp's promised location ("the `## Hooks` section's 2026-09-01 entry")
  and the actual entry's dateline match exactly.
- **Upstream-feedback draft is well-evidenced and appropriately conservative.** Cross-checked its
  four numbered evidence points against the design doc's §1.2/§2.1-2.3/§3 — all traceable, none
  embellished. It correctly omits the hook-`ask`-under-bypass claim entirely (the one Pass 2 flagged
  as still not fully separable from a headless-mode confound) rather than risk overclaiming it in an
  external submission, and explicitly frames the "likely mechanism" as a hypothesis in its own
  closing reviewer note, not an assertion. Clearly marked DRAFT/NOT SUBMITTED, correctly deferring
  the actual filing to a human.
- **`cobb/kaizen/history.md` and `plan.md` entries are accurate.** Cross-checked every specific claim
  (the settings.json edit shape, the wording fix, the KB fold-in, the draft's location and rationale,
  "left everything staged, uncommitted" for `teco`) against the actual working-tree state —
  `git diff --cached --name-only` confirms nothing is staged, matching the history entry's claim.
- **Nothing outside the design's remit was touched**: the coordination doc and my own Pass 1/2 review
  doc show no sign of edits attributable to U3 (their content ends exactly where Pass 2 left it, with
  no U3-dated addition); `docs/BACKLOG.md` untouched; no other repo file references stale
  current-state claims about `bypassPermissions` that this revert would newly invalidate (the handful
  of other hits are archived/historical coordination docs recording what was observed at the time,
  which is correct to leave as-is).

## Open questions

None — this implementation is a clean, verified execution of an already twice-gated design, with no
scope drift and no unresolved technical question left over from the design gate.
