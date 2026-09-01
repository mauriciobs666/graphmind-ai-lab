# `bypassPermissions` doesn't suppress Task-delegated subagent Write/Edit prompts — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (post-M3 follow-up, not a milestone gate)

## Scope & verdict

Reviewed: `claude/docs/plans/bypass-permissions-subagent-gap.md` (Gen 4, `cobb`), against the brief
in `claude/docs/plans/bypass-permissions-subagent-gap-coordination.md` and the three prior
generations it builds on (`agent-permission-friction.md`, `write-guard-classifier-gap.md` +
`-coordination.md`, `permission-default-mode.md`, all archived). This is a design/investigation
gate — no code/config diff exists to review; the check is whether the investigation and
recommendation are sound enough to act on.

**Verdict: approve with suggestions.** The forensic evidence is exceptionally well-supported — I
independently recomputed every `tool_use`→`tool_result` gap in all four live transcripts
(`architect`, `tdd-engineer`, `analyst`, and `cobb`'s own live test) directly from the raw JSONL,
and every figure in §1.2 and §3 matches to the decisecond. The core empirical finding (bypass
doesn't suppress the prompt, reproduces two Task-hops deep) and the central recommendation (revert
`defaultMode`) are sound and safe to act on. Two Major gaps keep this from a clean approve: an
available, low-cost, previously-untried isolation lever that directly targets the document's own
proposed mechanism was never tried or even mentioned, and the single most load-bearing safety
claim behind the revert recommendation is sourced from a commit message rather than the
independent doc verification the brief specifically asked for — and my own check of the current
docs doesn't confirm it as an explicit, verbatim statement. Neither undermines the recommendation
to revert (which stands on the friction evidence alone), so this doesn't block the design; both
should be closed out before the upstream report ships and before U3 is treated as final.

**CPG:** not applicable — this is a process/investigation document about the harness's own
permission mechanics, with no code-level component in this repo to load a CPG against.

## Findings

### Major — an available, documented, low-cost isolation lever was never tried or mentioned

`code.claude.com/docs/en/sub-agents` (fetched today, same page §2.1-2.2 quotes from) states
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` forces a spawned subagent to run in the **foreground**
"in every kind of session and whether or not fork mode is on" — the exact lever needed to test
§2's proposed mechanism (background dispatch's interaction with `bypassPermissions` inheritance)
in isolation. §2.1 asserts "no caller-facing toggle on the `Agent`/`Task` tool itself," which is
true narrowly, but there is a documented **session-level** toggle, reversible by unsetting one env
var, that the same docs page cobb fetched links from its own background-subagent section. This
is exactly the "genuine, previously-untried mechanism" class of outcome the coordination brief
asked for (point 3), and it wasn't tried, and isn't listed among §4.4's "what this document does
not resolve" items either. Before treating "no clean, narrowly-scoped fix exists" as final: run
one more live test — dispatch a subagent to a `docs/plans/*` write with
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` set and the project's `bypassPermissions` pin still in
place, and see whether the prompt disappears. If it does, that's a materially cheaper fix than
reverting `defaultMode` (keeps `auto`'s classifier off *and* Bash friction low); if it doesn't,
it's still valuable negative evidence to add to the upstream report, since it would show the bug
is in mode-inheritance itself, not specifically the background-dispatch path §2's mechanism
proposes.

### Major — the central bypass-skips-hook-`ask` safety claim is not independently verified, contrary to what the document implies

§4.2's load-bearing justification for reverting ("that's strictly worse, not a wash") rests on:
*"Hook `"ask"` is explicitly documented to be skipped under `bypassPermissions`" (per the commit
`6f719ae` message and the user's own memory)*. The brief specifically asked this claim be checked
against current docs independently. I fetched and grepped `code.claude.com/docs/en/hooks`,
`.../permissions`, and `.../permission-modes` today for every mention of `bypassPermissions` and
of hooks: the literal string `"bypassPermissions"` appears exactly once on the hooks page (a field
enum, unrelated), and the permission-modes page's own **"Actions no mode auto-approves"** section —
the definitive, itemized list of what survives `bypassPermissions` — names explicit settings `ask`
*rules*, `AskUserQuestion`/`requiresUserInteraction` tools, critical-path `rm`/`rmdir`, and
cross-session-messaging safeguards, but **never mentions a `PreToolUse` hook's own `"ask"`
decision**. Nowhere in current docs is there a verbatim statement that a hook's `"ask"` is
"skipped" under bypass. The practical conclusion cobb draws is still *plausible* by inference
(bypass's own description is "skips permission prompts... except for [that list]," so anything not
on the list, arguably including a hook-forced ask, would be skipped) — but that's cobb's inference,
not a documented fact, and the document presents it as the latter. Recommend either softening the
claim to "inferred, not verbatim documented" in the upstream report, or — better, and cheap —
running one more live check: point a subagent at a path that trips one of the eight existing
`guard-doc-writes.sh`/`guard-broad-write.sh` `"ask"` guards (e.g. `cobb`'s own topic-mismatch case)
under the current bypass-pinned config, and see whether it silently proceeds or still prompts.
(See `claude/docs/plans/agent-permission-friction.md` §1.3 for why this specific hook-mode
interaction has burned this team before — it was flagged unconfirmed under `acceptEdits` too, and
never closed.)

### Minor — one evidence-table caption doesn't match what the transcript shows

§1.2's row for `tdd-engineer`'s `docs/BACKLOG.md` edit reads *"0.0s — by design — ...`ask` rule is
meant to always fire; not part of this bug."* I traced the raw call in
`.../subagents/agent-a900b002b6a314590.jsonl` (`tool_use` `toolu_01HfFYAZJBoNgZRWT3s1nH8f`,
`10:01:14.823Z` → `tool_result` `10:01:14.833Z`, 0.010s, `is_error: true`): it's an ordinary Edit
failure, `"String to replace not found in file."` — ordinary content-mismatch, not a
settings-`ask`-rule firing at all (the agent worked around it seconds later via `sed -i` over
Bash). The row's exclusion from the "part of this bug" tally is still correct, but the stated
reason is wrong — nothing in this transcript actually exercises or confirms the `BACKLOG.md` `ask`
rule under bypass. Fix the caption so a future reader doesn't cite this row as a working
observation of the mode-proof `ask` rules holding.

### Minor — the Bash-vs-file-edit asymmetry claim (§2.3) is correct but the timing-gap method's Bash-side confound isn't addressed in the document

I independently checked: `tdd-engineer`'s transcript has Bash `tool_use`→`tool_result` gaps up to
17.6s (`.venv/bin/python -m pytest -q`, `git stash && ... pytest`), well outside the ~0.1s
"silent" band the method calibrates on for Write/Edit. Tracing the commands confirms these are
genuine `pytest` execution time, not a masked prompt, so §2.3's "not one Bash call ever produced a
confirmation gap" holds up on inspection — but the document doesn't show this disambiguation, and
Bash's variable execution time is a real confound the Write/Edit gaps don't have (a Write/Edit
completes near-instantly whether auto-approved or not, so any non-trivial gap is unambiguously a
wait; a Bash command's own runtime can look identical to a wait). Worth one sentence in §2.3 noting
the check was made, so the asymmetry claim doesn't read as merely "no long Bash gaps observed" when
what was actually confirmed is "the long Bash gaps that do exist all trace to real command
runtime."

## What's solid

- **§0's process-gap framing is accurate**, verified against `git show` for all three commits: all
  by `mauriciobs@gmail.com`, at the times and with the content stated (`f10cedf` 08:27,
  `c994442` 10:42, `6f719ae` 11:18, all 2026-08-29) — the commit-message quotes in the table are
  verbatim, not paraphrased.
- **§1.1's parent-mode claim is confirmed**: `grep '"permissionMode":"bypassPermissions"'` against
  `teco`'s own session transcript returns 25 matching records, zero of any other mode.
- **§1.2's entire gap table is correct to the decisecond** — I recomputed every Write/Edit
  `tool_use`→`tool_result` gap directly from the three subagent JSONLs and every number matches
  cobb's table exactly (e.g. `architect`'s three edits: 0.085s/0.095s/0.11s vs. reported "0.1s"
  ×3; `tdd-engineer`'s five `llm.py`-family edits: 196.96/17.74/11.20/462.24/6.19s vs. reported
  197.0/17.7/11.2/462.2/6.2s).
- **§3's live test (961s gap) is confirmed byte-for-byte**: `Write` `tool_use` at
  `2026-09-01T11:03:42.699Z`, `tool_result` at `11:19:43.589Z` in
  `.../subagents/agent-adf4bdc16aea9e91f.jsonl` (`parentAgentId: af27ea334c6585063` = `cobb`,
  `spawnDepth: 2`) — 960.89s, matching the claimed "961-second (16 min 1s)" figure, and the
  transcript independently confirms the described `server_error: request timed out` cutting the
  dispatch off right after.
- **The three prior generations are characterized accurately** — Gen 1's hook-`allow`-silent-fix,
  Gen 2's §U7 empirical refutation, and Gen 3's mode-inheritance quotes and dead-frontmatter finding
  all match my own read of those archived documents; nothing here misrepresents what came before.
- **§2.1-2.2's changelog and docs quotes are verbatim**: `v2.1.232`'s "non-teammate agent spawns in
  interactive sessions now run in the background by default" and `v2.1.251`'s "background
  subagents, the default, still show status only" both appear character-for-character in
  `~/.claude/cache/changelog.md`; the "takes precedence and can't be overridden" mode-inheritance
  quote and the Background Subagents section text both match a fresh fetch of the same page today.
- **Live config claims all verified directly**: `.claude/settings.json` (project) matches exactly;
  `~/.claude/settings.json` (global) still `"defaultMode": "auto"`;
  `.claude/settings.local.json`'s leftover `Edit(**/docs/reviews/**)` rule from Gen 2's §U7 is
  still there, exactly as flagged.
- The document correctly declines to guess at §1.3's stickiness-pattern question rather than
  padding the verdict with an unconfirmed explanation — appropriately flagged, not resolved.

## Open questions

- Should the two Major findings' follow-up live tests (the `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS`
  isolation test, and the hook-`ask`-under-bypass confirmation) be run as an amendment to this same
  U1 before U3 proceeds, or folded into U3's own scope as a pre-check? Either is defensible; the
  coordination doc doesn't currently allocate a unit for it.
- If the `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` test does turn out to close the gap, does the team
  want that as a *replacement* for the `defaultMode` revert (keeping `bypassPermissions` plus the
  env var), or as an *additional* data point while still reverting `defaultMode` on the
  independent friction-didn't-improve grounds already established? That's a design call for
  `cobb`/`teco`, not something I should resolve from a static review.

## Pass 2 — 2026-09-01

**Scope:** re-review of `cobb`'s in-place amendment (new §2.4, rewritten §4.2, edited §1.2/§2.3
captions) addressing Pass 1's two Major and two Minor findings. I independently re-verified every
raw-evidence claim in the new material against the actual transcripts cited, the same way as
Pass 1 — not just read cobb's prose.

**Pass 1 disposition:**

- **Major 1 (untried `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` lever) — addressed, honestly.** Cobb
  ran it (new §2.4) and reported "inconclusive," not a clean result either way. I re-derived the
  evidence independently: the lever-set subagent's Write/Edit/`rm` gaps
  (`.../04570ad2-.../subagents/agent-a5853a46086e1d9de.jsonl`) are 0.073s/0.066s/0.056s, matching
  §2.4 point 3 exactly; and the raw `subagent_stats` JSON blobs embedded in this session's own
  transcript (`bff1000c-....../subagents/agent-a1570f691aaabd084.jsonl`) show
  `"started_in_background":0` on **every** run, lever-set and control alike, with `"requested"`
  reading `{"unset":1}` for the lever-set runs vs. `{"foreground":1}` for the control — exactly
  what §2.4 reports. The "inconclusive" framing is correct, not a dodge: the precondition the test
  needed (`started_in_background:1`) genuinely never fired under `-p`/headless dispatch, for either
  arm. This is a properly disambiguated negative result, not one dressed up to look positive.
- **Major 2 (hook-`ask`-skipped-under-bypass claim unverified) — partially addressed; one new
  confound found, see below.** Cobb ran a live test (rewritten §4.2) and got a real, positive data
  point: a `Write` to a `docs/plans/*` path blocked pre-write by `tdd-engineer`'s own deny-list
  guard. I pulled the raw transcript
  (`8d8ac924-.../subagents/agent-a8cda714a0a091f39.jsonl`): `Write` `tool_use` at
  `12:08:25.144Z` → `tool_result` at `12:08:25.199Z` (0.055s, `is_error: true`), content the
  guard's own escalation message verbatim, no file created — matches §4.2's description exactly.
  This is a real improvement over Pass 1 (a live test beats an uncited commit-message claim). But
  see the new finding below — I don't think this fully closes the gap the way §4.2 now claims.
- **Minor 1 (BACKLOG.md caption) — fixed.** §1.2's row now reads "ordinary content-mismatch, not
  an `ask`-rule firing," cites the same `tool_use`/`tool_result` IDs and timestamps I traced in
  Pass 1, and correctly notes the row doesn't confirm the `ask` rule under bypass. Matches.
- **Minor 2 (Bash-disambiguation not shown) — fixed.** §2.4's closing paragraph now states the
  17.6s Bash gaps were traced to `pytest`/`git stash` execution, not a masked prompt. Matches what
  I verified in Pass 1.

**New finding (Minor-to-Major) — §4.2's live test doesn't actually isolate "bypass mode" from
"headless mode has no way to show a prompt at all."** `code.claude.com/docs/en/headless` (fetched
today): *"For `-p`, the built-in starting permission mode is Manual on every plan"* and,
separately, `dontAsk` mode is documented to *"deny[] anything not in your `permissions.allow`
rules"* rather than prompt, specifically because there's no human to ask. §4.2's test ran headless
(`-p`, no TTY) — the same methodology §2.4 uses, and §2.4 explicitly reasons about a headless-mode
confound for the background-dispatch precondition, but §4.2 doesn't apply the same scrutiny to
itself: an instant (0.055s) denial on a hook-`ask`-shaped decision in a context with literally no
surface to show a prompt on is equally consistent with "`bypassPermissions` genuinely still
enforces hook `ask`" *and* with "`-p` mode auto-denies any would-be-prompt regardless of which
mode is configured, the same way `dontAsk` is documented to." The test as run cannot distinguish
these two explanations, so "directly refutes 'documented to be skipped'" (§4.2) overclaims what a
single headless data point can support — it's evidence against the *strongest* form of the
original claim (a blanket "skipped," which would predict the write silently succeeding, and it
didn't), but it does not establish that `bypassPermissions` *specifically* — as opposed to
`-p`'s own Manual-by-default/no-prompt-surface behavior — is what caused the block. **This doesn't
change the verdict**: §4.2 itself already frames the recommendation as resting on the friction
finding alone (§1.2/§3), with this test only as an ancillary "no longer a confirmed cost" argument
— so the gap here is in how confidently that ancillary point is stated, not in the load-bearing
evidence. Suggest one clause softening "directly refutes" to "is inconsistent with, in a headless
context this test can't fully separate from mode" before this goes into the upstream report
(§4.1), so the report doesn't claim more certainty than the test supports.

**The unexpected §2.4 side-finding (a nested session's own top-level agent's hooks appearing to
additionally constrain a subagent it dispatches) is fine to leave open.** It's explicitly flagged
in §4.4, doesn't bear on §4.2's recommendation either way, and chasing it now would be scope creep
on an already-large investigation — same category as §1.3's stickiness question, which Pass 1 also
accepted as legitimately deferred. No objection.

**Verdict: approve with suggestions, unchanged. U1 is ready to gate U3.** Both original Majors
were substantively and honestly addressed with new live evidence, not hand-waved; the one new
wrinkle I found (§4.2's headless-mode confound) affects only the strength of a supporting argument
cobb has already subordinated to the primary friction-based case, not the recommendation itself.
Before `4.1`'s upstream report ships, fold in the one-clause softening above; that can happen
alongside or after U3 starts — it is not a precondition for reverting `defaultMode`, which remains
justified on §1.2/§3 alone.
