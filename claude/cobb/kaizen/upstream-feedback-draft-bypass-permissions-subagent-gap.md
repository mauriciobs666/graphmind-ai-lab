# DRAFT — NOT SUBMITTED — follow-up to feedback receipt `3ccd08fc`

> This is a draft only. It has not been filed. Filing external feedback under the user's identity
> is not something `cobb` does autonomously — this text is left here for the user or `teco` to
> review, edit, and submit themselves (e.g. via `/feedback` in Claude Code, or whatever channel is
> current). Source design: `claude/docs/plans/bypass-permissions-subagent-gap.md` §4.1. Written
> 2026-09-01 by `cobb`, per the design's own instruction: reference the existing receipt, do not
> re-draft its original content — this is additive.

---

## Suggested submission channel

`/feedback` (in-product), referencing receipt `3ccd08fc` as the original report this follows up on.
Claude Code version at time of writing: `2.1.252`.

## Suggested title

Follow-up to `3ccd08fc`: `bypassPermissions` does not suppress background-subagent Write/Edit
permission prompts either (file-edit-tool-specific, not `acceptEdits`-specific)

## Suggested body

The original report (`3ccd08fc`, filed 2026-08-28/29) was scoped to `acceptEdits`: a spawned
subagent's first `Edit` still prompted despite parent `acceptEdits`, a matching `Edit(**)` settings
allow rule, and a `PreToolUse` hook explicitly returning `"allow"`. We since escalated our own
project's `permissions.defaultMode` to `bypassPermissions` specifically to route around that —
your own docs state parent-mode `bypassPermissions` "takes precedence and can't be overridden" onto
a dispatched subagent. It doesn't work either. Fresh evidence, collected today across three
independently-dispatched subagent types under one live coordinated session:

1. **Clean, cross-agent-type reproduction.** Three subagent types (`architect`, `tdd-engineer`,
   `analyst`), dispatched via `Agent`/`Task` from one top-level interactive session independently
   confirmed — via its own transcript's `"type":"permission-mode"` records — to have stayed in
   literal `bypassPermissions` for the entire run (25/25 records, zero drift). Eight of ten
   non-protected, in-remit `Write`/`Edit` calls each produced a real, human-confirmation-scale
   `tool_use`→`tool_result` gap (6.2s-462.2s); the other two were repeat touches of a file already
   approved once in that same run, and there is no reliable per-run/per-file stickiness across
   agent types — one agent's run re-prompted on the *same file* four separate times with no carry-
   over at all.

2. **The bug is specific to file-editing tools, not a blanket mode-inheritance failure.** Across
   all evidence gathered — the three fresh transcripts above, plus a targeted live reproduction —
   not one `Bash` call ever produced a confirmation gap; a blanket `"Bash"` `permissions.allow` rule
   reached every background-dispatched subagent correctly, every time, including calls with long
   genuine runtimes (traced to real `pytest`/`git stash` execution, not a masked prompt — ruling out
   the obvious confound). Only `Write`/`Edit` (and presumably `NotebookEdit`, by the same
   `Edit(path)`-covers-`Write` semantics your docs already establish) fail to inherit the parent's
   `bypassPermissions` under this dispatch path. This is a sharper, more actionable characterization
   than "subagent writes don't inherit bypass."

3. **Likely mechanism: an undocumented interaction between background-subagent dispatch (default
   since v2.1.232) and parent-mode inheritance.** Two adjacent sections of
   `code.claude.com/docs/en/sub-agents` appear not to have been written with each other in mind:
   "Background Subagents" describes *where* a needed permission prompt surfaces ("Claude Code
   surfaces the prompt in your main session and names the subagent that is asking"), while the
   parent-mode-inheritance passage immediately adjacent states flatly that a parent's
   `bypassPermissions`/`acceptEdits` "takes precedence and can't be overridden," with no stated
   background-dispatch exception. Since non-teammate `Agent`/`Task` dispatches run in the background
   by default in every interactive session (changelog v2.1.232, v2.1.251), essentially every ordinary
   subagent dispatch today is silently exercising this undocumented interaction. Two independent
   fetches of the same doc page, specifically hunting for a stated carve-out, came back empty-handed.

4. **Reproduces at least two Task-hops deep, not just for a top-level session's direct dispatch.** A
   subagent dispatched by another subagent (itself already a dispatch of the top-level session) — no
   settings file or mode touched at any intermediate layer — showed the single longest gap recorded
   in this investigation (961s / 16m1s) for an otherwise ordinary, non-protected `docs/plans/*`
   Write. Rules out "specific to being the top-level session's own direct child" as an explanation.

**Net effect:** for any project relying on `bypassPermissions` (or `acceptEdits`) specifically to
get frictionless delegated subagent writes, that benefit currently does not exist for background
dispatches — which is the default dispatch mode for ordinary (non-teammate) `Agent`/`Task` calls in
every interactive session. We're reverting our own project's `defaultMode` pin back to unset/`auto`
as a result (no config-level workaround closes the gap; `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`
could not be conclusively tested by us — see below).

**One lever we could not conclusively test, in case it's a quicker fix on your end:**
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` is documented to force foreground subagent dispatch "in
every kind of session." We attempted to test it via a nested headless (`-p`) session, but headless
dispatch never appears to enter the background-dispatch code path at all regardless of the env var
(`subagent_stats.started_in_background: 0` on every arm we ran, lever-set and control alike) — so
our test never actually exercised the precondition needed to say whether the lever helps. A real
test needs the var set before an interactive, TTY-attached session starts, which we didn't have a
way to automate. If your team can run that test internally, it would be useful signal either way.

## Full evidence

Available on request — detailed transcript citations (JSONL paths, exact timestamps, raw
`tool_use`/`tool_result` gaps for every call referenced above) live in our own internal design
document; happy to share specifics if useful for your triage.

---

*(End of draft body. Reviewer note from `cobb`: keep item 3's "likely mechanism" framing as a
hypothesis, not an assertion — we can't see Anthropic's internal implementation, only the
doc-contradicting external behavior. If whoever submits this wants to trim it, items 1+2+4 are the
load-bearing evidence; item 3 is our best-guess mechanism and the "lever we couldn't test" note is
a good-faith assist, not something we're confident about.)*
