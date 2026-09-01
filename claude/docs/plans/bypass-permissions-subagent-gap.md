# `bypassPermissions` doesn't suppress Task-delegated subagent Write/Edit prompts — Design

> **Status:** active · **Owner:** `cobb` · **Tracks:** — (post-M3 follow-up, not a milestone gate)

**Background (read in full, not re-derived here):**
`claude/docs/plans/agent-permission-friction.md` (Gen 1, archived — hook `"allow"` silent-`exit 0`
bug, fixed; root-caused that a subagent's frontmatter `permissionMode` is silently overridden by
ambient session mode in documented cases) ·
`claude/docs/plans/write-guard-classifier-gap.md` + `-coordination.md` (Gen 2, archived — a
`permissions.allow` `Edit(path)` settings rule was designed as a classifier-bypass supplement,
then **empirically refuted**: it does not suppress the prompt for a Task-delegated write, §U7) ·
`claude/docs/plans/permission-default-mode.md` (Gen 3, archived, v2 — precisely quoted the real
subagent-mode-inheritance mechanism, confirmed all 13 agents' `permissionMode: acceptEdits`
frontmatter is dead configuration for session-start purposes, and recommended **against** adopting
`acceptEdits`/`bypassPermissions` team-wide on Bash-friction cost grounds) ·
`skills/agent-standards/claude-code.md` (the durable KB all three folded findings into).

This document is Gen 4. Its trigger: sometime after Gen 3 closed with "stay on `auto`," someone
overrode that recommendation and pinned `bypassPermissions` anyway (§0 below) — and fresh,
clean, cross-agent-type evidence collected today (2026-09-01) shows it doesn't even deliver the
frictionless delegated-write behavior it was adopted for.

---

## 0. The process gap — what actually shipped, and how (documented per the coordination doc's ask)

Three commits, all 2026-08-29, all authored directly by the user (`mauriciobs@gmail.com`) — **not**
by `cobb`, and with **no** `claude/docs/plans/*` design doc and **no** `analyst` gate, breaking this
team's own established convention for this exact class of change (`agent-permission-friction.md`'s
header: *"Owner altitude: cobb (design + implementation) · Reviewer: analyst (gate before
implementation)"*):

| Commit | Time | What it did |
|---|---|---|
| `f10cedf` | 08:27 | Pinned `defaultMode: acceptEdits` + a itemized Bash allowlist (cypher MCP, venv python, uvicorn pkill, git add/commit) |
| `c994442` | 10:42 | Collapsed the itemized Bash allowlist into a blanket `"Bash"` allow rule — "end per-command prompt whack-a-mole" |
| `6f719ae` | 11:18 | Switched `defaultMode` from `acceptEdits` to **`bypassPermissions`**, and rebuilt the deliberate safety gates (destructive-ops patterns, `docs/BACKLOG.md`) as `permissions.ask` **rules** instead of hooks — because, per the commit message, "subagents' first Edit prompts even under an `acceptEdits` parent with `Edit(**)` allowed and a hook emitting allow (upstream bug)," and the docs guarantee explicit `ask` rules/`AskUserQuestion` survive bypass, unlike hook `"ask"`, which bypass skips |

The user's own Claude Code memory (`subagent-permission-mitigation`, session-local, not
committed) documents the reasoning trail behind these three commits in detail: transcript
forensics on 2026-08-28/29 sessions found (a) a frontmatter hook's `"allow"` is ignored for a
Task-spawned subagent under an `auto`-mode parent — matches Gen 3's already-established finding;
(b) settings `Edit(**)` allow rules also don't reach subagent calls — matches Gen 2's §U7
refutation; (c) even with the parent switched to `acceptEdits`, a spawned `coder`'s first Edit
still prompted (176s stall) despite all three layers that should have silenced it (parent
`acceptEdits` "takes precedence," settings `Edit(**)`, and the agent's own hook emitting
`"allow"`) — one un-silenceable yes/no per spawned run, after which later edits in that same run
sailed; (d) an upstream bug report was drafted (**receipt `3ccd08fc`**, "do not re-draft," not yet
independently confirmed fixed). The final move to `bypassPermissions` was the user's own
escalation past that partial mitigation, explicitly framed as "last resort" in the memory, applied
directly (per the memory: "the classifier blocks Claude from writing a blanket Bash grant itself"
— i.e. a human, not an agent, had to make at least the Bash-collapse edit).

**This is a materially informed decision, not a blind one** — the user had already independently
rediscovered most of Gen 2/3's findings by direct transcript forensics before pinning bypass. But
it was made outside this team's documented process (no `cobb` design, no `analyst` gate), and —
per today's evidence below — the specific problem it was adopted to solve (the "one un-silenceable
yes/no per spawned run" residual under `acceptEdits`) is **not actually solved** by
`bypassPermissions` either; today's reproduction is worse (no scope-stickiness at all in 2 of 3
runs, not even per-file). The record is now complete and honest: this document is that overdue
`analyst`-gated review, after the fact.

---

## 1. Today's fresh evidence, forensically re-examined

Source: `falkor-chat/docs/plans/bare-call-key-shadowing-coordination.md`, a `teco`-coordinated run
today (2026-09-01) dispatching `architect`, `tdd-engineer`, `analyst` via `Agent`/`Task`, live
project config `permissions.defaultMode: "bypassPermissions"` (`.claude/settings.json`, pinned
commit `6f719ae`).

### 1.1 `teco`'s own session never left `bypassPermissions`

`grep '"type":"permission-mode"' bff1000c-....jsonl` — every one of dozens of records across the
whole session reads `"permissionMode":"bypassPermissions"`. No drift to `auto`, ever. This directly
answers the brief's scoping question (c): `teco` genuinely is the top-level, primary interactive
session (`--agent teco`, a human at the terminal) — there is no further layer of indirection at
*that* level, and its own ambient mode is exactly what the project pin says it should be.

### 1.2 Gap analysis of the three transcripts (tool_use → tool_result wall-clock gap, per call)

Subagent (sidechain) transcripts carry **no** `"type":"permission-mode"` records at all — unlike a
top-level session transcript, this is not a place to directly read "what mode was ambient." The
only available signal is timing: a call that actually needed human confirmation shows a real,
human-reaction-scale gap between the model proposing the tool call and the tool result landing; a
call that was silently auto-approved completes in ~0.1s.

| Agent | Call | Target | Gap | Read |
|---|---|---|---|---|
| `architect` | Write | `docs/plans/bare-call-key-shadowing.md` | 0.1s | *(see 1.3 — likely a real prompt, answered near-instantly by a primed human, not silent)* |
| `architect` | Edit #1 | *same file* | 0.1s | sailed, no prompt |
| `architect` | Edit #2 | *same file* | 0.1s | sailed, no prompt |
| `tdd-engineer` | Edit | `server/tests/test_llm.py` | 197.0s | real prompt |
| `tdd-engineer` | Edit #1 | `server/falkorchat/llm.py` | 17.7s | real prompt |
| `tdd-engineer` | Edit #2 | *same file* | 11.2s | real prompt |
| `tdd-engineer` | Edit #3 | *same file* | 462.2s | real prompt |
| `tdd-engineer` | Edit #4 | *same file* | 6.2s | real prompt |
| `tdd-engineer` | Edit | `docs/HISTORY.md` | 226.9s | real prompt |
| `tdd-engineer` | Edit | `docs/BACKLOG.md` | 0.0s | **ordinary content-mismatch, not an `ask`-rule firing** — traced to the raw transcript (`.../subagents/agent-a900b002b6a314590.jsonl`, `tool_use` `toolu_01HfFYAZJBoNgZRWT3s1nH8f`, `10:01:14.823Z`→`10:01:14.833Z`, `is_error: true`, `"String to replace not found in file."`, worked around seconds later via `sed -i` over Bash); the row's exclusion from the "part of this bug" tally is still correct, but this transcript does not actually exercise or confirm the `Edit(**/docs/BACKLOG.md)` `ask` rule under bypass one way or the other — see §4.2 for a live test that does |
| `analyst` | Write | `docs/reviews/bare-call-key-shadowing.md` | 25.2s | real prompt |

This matches `teco`'s own user-observed tally exactly (architect: 1 prompt; tdd-engineer: 6, all
non-BACKLOG calls; analyst: 1) with one added nuance the tally didn't capture: **within
`tdd-engineer`'s own run, the *same file* (`llm.py`) was independently re-prompted four separate
times**, each a real multi-second-to-multi-minute wait — there is no per-file, per-run, or
per-session stickiness for that agent's run at all. `architect`'s run shows the opposite pattern
for its own repeat-touches-of-the-same-file (2nd/3rd calls sailed after the 1st). See §1.3 for
why this doesn't change the verdict.

### 1.3 The `architect` stickiness pattern is very likely a UI artifact, not a config difference — flagged, not resolved

I attempted to reach `teco` (`SendMessage`) to ask whether the human clicked a persistent
"don't ask again" style option on `architect`'s first Write prompt specifically, vs. a plain
one-shot "yes" on each of `tdd-engineer`'s six. `teco` was not reachable by that name from my
session at the time I asked ("No agent named 'teco' is reachable"), and no subagent transcript
records which option a human picked at a prompt (only the final tool_result). **This is
genuinely unresolved** — flagged rather than guessed at, per the stop-and-ask spirit of this
brief, though it does not block the verdict below (§2).

The current official docs (fetched today, `code.claude.com/docs/en/sub-agents`, "Background
Subagents" section — see §2) state: *"When you answer one of those prompts with a choice that
lasts beyond that one tool call, such as a grant that lasts for the rest of the session, Claude
Code applies your answer to the whole session, including your main conversation."* This is
consistent with a human-side UI choice (an optional "yes, and don't ask again [for this
file/session]" button) explaining the difference, entirely independent of `defaultMode` — not a
config-level inconsistency. If so, the honest count of *config-attributable* friction is "every
subagent's first touch of a new file always needs a fresh human decision" (11 of 11 non-BACKLOG,
non-repeat-file calls needed one), with the "does a repeat-touch of the same file in the same run
also need one" question answered differently by which optional button a human happened to click,
not by anything `bypassPermissions` controls one way or the other.

---

## 2. Root cause: a documented mechanism (background-subagent dispatch) whose parent-mode
   inheritance the fresh evidence shows is not actually holding

### 2.1 The mechanism Gen 1-3 never had in hand: non-teammate dispatches run in the *background* by default, and have since Claude Code **v2.1.232**

`grep -i "background" ~/.claude/cache/changelog.md`, version `2.1.232` (this repo's live version
today is `2.1.252`, so this has been in effect the entire time, including for every prior
generation's own testing on `2.1.238`-`2.1.241`):

> "Subagent forking is now on by default: a `subagent_type: "fork"` subagent inherits the full
> conversation and prompt cache, and **non-teammate agent spawns in interactive sessions now run
> in the background by default**."

This is a harness-level default with no caller-facing toggle on the `Agent`/`Task` tool itself —
`teco` dispatching `architect`/`tdd-engineer`/`analyst`, and `cobb` (this session) dispatching a
throwaway test agent (§3), are all, silently, **background subagent** dispatches, not the
"foreground, blocks the parent" shape the sub-agents doc describes first. Confirmed further by
`v2.1.251`'s changelog line: *"Added live streaming of a foreground subagent's tool calls...
(background subagents, **the default**, still show status only)."*

### 2.2 What the current docs say about background subagents and permission mode

Fetched fresh today (`code.claude.com/docs/en/sub-agents`, "Background Subagents" section — none
of Gen 1-3 quote this section; it either post-dates their reads or they didn't have reason to look
for it):

> "**Foreground subagents** block the main conversation until complete. Permission prompts are
> passed through to you as they come up.
> **Background subagents** run concurrently while you continue working. When a background subagent
> reaches a tool call that needs permission, Claude Code surfaces the prompt in your main session
> and names the subagent that is asking. Approve to let the subagent continue, or press Esc to deny
> that one tool call without stopping the subagent."
>
> "Background subagents run with a smaller built-in tool set than foreground subagents, except for
> conversation forks, and they surface every permission prompt in your main session."

And, immediately adjacent on the same page, the parent-mode-inheritance rule Gen 3 already
precisely quoted, unchanged and with no stated exception for background dispatch:

> "If the parent uses `bypassPermissions` or `acceptEdits`, this takes precedence and can't be
> overridden. If the parent uses auto mode, the subagent inherits auto mode and any
> `permissionMode` in its frontmatter is ignored..."

Read together, and read plainly: a background subagent's prompt-routing (§2.2's first quote) is a
statement about **where** a needed prompt appears, not a statement that a prompt is always needed
regardless of mode. Nothing in the page carves out bypass/background as an exception to the
inheritance rule. I confirmed this by asking two independent, narrowly-targeted fetches of the same
page specifically to hunt for such a carve-out; both came back empty-handed ("The documentation
contains no exception or limitation stating that background subagents bypass this rule... The
inheritance rule appears absolute").

### 2.3 Verdict: genuine harness bug/regression, not a config or scoping error on this repo's side

Per the coordination doc's three-way framing (§ "What's actually being asked of you," point 1):

- **(a) genuine harness bug/regression contradicting documented behavior — YES, this is the
  finding.** `teco`'s parent session is confirmed continuously in literal `bypassPermissions`
  (§1.1); the sub-agents doc states this "takes precedence and can't be overridden" onto a
  dispatched subagent, with no documented background-dispatch exception (§2.2); and yet every
  ordinary, non-protected, in-remit Write/Edit call from three independently-dispatched subagent
  types produced a real, multi-second-to-multi-minute human-confirmation gap today (§1.2), fully
  reproduced by my own live dispatch this session (§3). The mechanism the docs promise is not what
  the harness delivers for the (now-default) background-dispatch path.
- **(b) a config/precedence issue on this repo's side — ruled out, with specifics.** The three
  `.claude/settings.json` `allow` rules (`Bash`, `Edit(**)`, `mcp__cypher__query`) are exactly as
  documented; `~/.claude/settings.json` (global) still carries `defaultMode: "auto"` and nothing
  else relevant (re-confirmed today) — the project pin is the only thing setting `bypassPermissions`
  anywhere, and it is correctly the higher-precedence file. No `disableBypassPermissionsMode`
  managed-settings policy exists on this machine (checked; none found). The one stray artifact
  found — `.claude/settings.local.json` (gitignored, untracked, personal) still carries the
  `"Edit(**/docs/reviews/**)"` rule Gen 2 added for its now-refuted §U7 test and never removed — is
  a hygiene leftover, not a contributing cause: it's an `allow` rule and today's `analyst` Write to
  exactly that glob still prompted (25.2s gap, §1.2), consistent with (not contradicting) Gen 2's
  own refutation that a matching rule doesn't suppress a delegated write's prompt.
- **(c) a scoping issue specific to how `teco` itself is run — ruled out.** §1.1 confirms `teco` is
  genuinely the top-level primary session with no mode drift. §3's test also shows the gap
  reproducing one level *deeper* than `teco`'s own direct dispatches (a `cobb`-dispatched subagent,
  where `cobb` is itself already a `teco` dispatch) — so this isn't specific to being `teco`'s
  direct child either; it recurs at whatever depth a background subagent is spawned from.

### 2.4 Live test of the isolation lever (`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`) — inconclusive, and why

The review's Major finding 1 asked for a live test of `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`
(`code.claude.com/docs/en/sub-agents`: forces foreground subagent dispatch "in every kind of
session"). I ran it, and the result is genuinely inconclusive — not because the test failed, but
because the mechanism available to run it doesn't reach the code path in question. Documented here
in full rather than glossed over, since the honest negative result changes how much weight §4.2 can
put on this lever.

**Why a nested headless session, not a direct dispatch from this session:** a session-level env var
takes effect for a process at its own start; nothing available to a running Claude Code agent (no
tool here mutates the current process's own environment, and Bash's exported vars don't survive
past that one call, let alone reach a different tool's dispatch logic) can retroactively set it for
*this* already-running session. The only faithful way to actually exercise the var is a fresh
process that has it set from the start. I ran `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 claude -p
"<dispatch a tdd-engineer subagent to write/edit/rm an ordinary scratch file>" ` (headless/print
mode, `cwd` at the repo root, project `bypassPermissions` pin unchanged) as that fresh process, with
a matched no-env-var control run for comparison.

**Two false starts, then a clean pair:**
1. First attempt targeted `falkor-chat/docs/plans/_permission-test-scratch-cobb2.md` (reusing the
   brief's suggested path) with the nested session's own top-level `--agent cobb`. It was blocked
   pre-write by `tdd-engineer`'s **own** deny-list guard (`guard-tdd-broad-write.sh` — `docs/plans/*`
   is explicitly on that list, §4 of `claude/AGENTS.md`'s hook-machinery section) — an *expected*,
   structural block unrelated to background dispatch or bypass at all, not a useful data point for
   this question (though it *is* directly useful for §4.2 — see below).
2. Second attempt retargeted an ordinary, undeniably-in-remit path
   (`falkor-chat/server/falkorchat/_permission_test_scratch_cobb2.py`, a plain source file, still
   under nested `--agent cobb`) — and was blocked instead by **`cobb`'s own** topic-remit guard
   (`guard-cobb-topic-writes.sh`), even though the dispatched subagent was `tdd-engineer`, not
   `cobb`. This is a genuinely unexpected, previously-undocumented-here observation: a nested
   session's own top-level agent's hooks appear to additionally constrain a subagent it dispatches,
   on top of that subagent's own hooks — a real finding, but a different one from what this test set
   out to check, and outside this document's scope to chase further (flagged, not resolved, same
   spirit as §1.3's stickiness question).
3. Retargeting the same ordinary source-file path with **no top-level `--agent` flag** (a plain
   default session carrying no hooks of its own, so only `tdd-engineer`'s own guard could apply)
   finally isolated the intended question. Result, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` set:
   Write (0.073s), Edit (0.066s), `rm` via Bash (0.056s) — all three tool_use→tool_result gaps
   near-instant, `permission_denials: []`, subagent completed clean
   (`~/.claude/projects/.../04570ad2-.../subagents/agent-a5853a46086e1d9de.jsonl`).
4. **The control** — identical setup, env var *not* set — came back just as clean: same near-zero
   friction, `permission_denials: []`, all three steps completed with no prompt.

**Why this doesn't confirm the lever fixed anything:** both runs' own harness-reported
`subagent_stats.started_in_background` field read **`0`** — the lever-set run *and* the no-lever
control. Headless/print (`-p`) mode's subagent dispatch apparently never enters the
background-dispatch path the changelog quote in §2.1 scopes to "**interactive** sessions" — so
neither run exercised the precondition (`started_in_background: 1`) that §2's theorized mechanism
requires to produce the bug in the first place. A clean result under a lever that was never actually
tested against the failure mode it's meant to fix is not evidence the lever works — it's an artifact
of headless mode already defaulting to foreground, with or without the var. (One secondary
data point consistent with this: the control run's `subagent_stats.requested` field read
`{"foreground":1}` explicitly, vs. the lever-set run's `{"unset":1}` — suggestive that the var acts
below the layer the "requested" field reports on, but neither run ever flipped
`started_in_background` to `1`, which is the only field that would actually answer the question.)

A genuine test needs the var set before an **interactive**, TTY-attached session starts (not `-p`) —
something outside what a running subagent can arrange for itself (no TTY-driving capability here,
and, per above, no way to mutate a live process's own env). This is a capability gap in what a
delegated subagent can self-test, not a finding about the lever one way or the other. **My
judgment, stated per the review's open question:** given this is inconclusive rather than negative,
it should not be *presented* as a refutation of the lever in the upstream report (§4.1) — but it
also supplies no basis to let a hoped-for cheap fix delay or replace §4.2's revert recommendation,
which stands on its own already-solid friction evidence (§1.2/§3) independent of whether this lever
turns out to help. If a future session can actually drive an interactive test (e.g., a human running
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 claude` themselves, or a `pty`-based automation this
investigation didn't attempt), that's worth doing — as an *additional* mitigation to layer on top of
the revert if it works, not a reason to hold the revert open until it's tried. Recommendation
unchanged from §4.2's original framing; this section only tempers how much credit the upstream
report can take for having tried the lever.

**One further, sharper characterization worth logging precisely, because it narrows where a future
report or fix should look:** across all three fresh transcripts plus my own test, **not one `Bash`
call ever produced a confirmation gap** — the blanket `"Bash"` allow rule visibly worked, every
time, from a background-dispatched subagent, with zero exceptions. (Not merely "no long Bash gaps
observed": `tdd-engineer`'s transcript has Bash gaps up to 17.6s — `.venv/bin/python -m pytest -q`,
`git stash && ... pytest` — well outside the ~0.1s silent band; tracing those specific commands
confirms they're genuine `pytest` execution time, not a masked prompt, so the asymmetry claim holds
up under the one real confound Bash's variable runtime introduces that Write/Edit's near-instant
completion doesn't have.) The gap is specific to
**file-editing tools** (`Write`/`Edit`, and by the `Edit(path)`-covers-`Write` rule semantics Gen 2
already established, presumably `NotebookEdit` too) under background dispatch — not "rules/mode
never reach a background subagent" in general. This is a materially more precise bug description
than "subagent-delegated writes don't inherit bypass," and should be the framing carried into the
upstream report (§4).

---

## 3. Live empirical test (this session, `cobb`, 2026-09-01)

Per the brief's constraint 2 — the human (`mauricio`) was at the terminal, actively watching for
and approving permission popups, the same setup that produced today's fresh evidence.

**Test 1 (uninformative by construction, kept for the record):** dispatched a `tdd-engineer`
subagent to Write then Edit a file at this session's own `/tmp/claude-1000/.../scratchpad/`
path. Both calls completed in a combined **8.8s**, "no perceptible delay or permission prompt" per
the agent's own report. **This does not test the bug** — this repo's own scratchpad is documented
(this agent's own system prompt) as harness-exempt from permission review ("can generally be used
without permission prompts"), so a no-prompt result here is expected regardless of the
`bypassPermissions` question and doesn't bear on it either way. Logged so a future reader doesn't
mistake it for a (non-)reproduction.

**Test 2 (the real test):** dispatched a second `tdd-engineer` subagent — itself a `Task`-delegate
of `cobb`, which is itself a `Task`-delegate of `teco`, i.e. **two levels of delegation deep**, not
one — to Write, then Edit, then `rm` a scratch file at an in-repo path matching today's exact
reproduction shape (`falkor-chat/docs/plans/_permission-test-scratch-cobb.md`, a `docs/plans/*`
target, same glob `architect` used today).

Its own transcript's timestamps give the cleanest single data point in this whole investigation:
the `Write` tool_use fired at `11:03:42.699Z`; its `tool_result` didn't land until `11:19:43.589Z`
— a **961-second (16 min 1 s) gap**, the longest of any call in today's evidence (vs.
`tdd-engineer`'s own 462.2s max in the original three-agent run, §1.2), for the identical
call shape (`docs/plans/*`, hook-cleared, non-protected, in-remit) that `architect` produced in
0.1s earlier today (§1.3's still-open stickiness question). The dispatch itself then died to an
unrelated transient platform error (`server_error: request timed out`, per `teco`'s relay) a few
turns after the Write landed — before it reached the Edit or `rm` steps — leaving the scratch file
on disk with only its first line written; **cleaned up directly by `cobb`** once found (confirmed
removed). The platform timeout is infrastructure flakiness, unconnected to the permission finding:
the 961s figure comes entirely from the harness's own `tool_use`/`tool_result` timestamps, recorded
before the timeout occurred.

**What this test isolates that today's evidence alone didn't:** confirms the gap is not specific to
being `teco`'s own *direct* dispatch target (§2.3's ruled-out (c)) — it reproduces, if anything
*more severely*, for a subagent two Task-hops removed from the primary interactive session,
dispatched by another subagent (`cobb`) that itself never touched a settings file or changed mode.

---

## 4. Recommendation

No clean, narrowly-scoped fix exists — this closes the same way Gen 2 and Gen 3 both did, but with
a materially worse cost-benefit picture than either of them faced, because `bypassPermissions` was
adopted specifically to solve the problem it turns out **not to solve**.

### 4.1 Strengthen the upstream report

The existing report (receipt `3ccd08fc`, filed 2026-08-28/29 per the user's own memory, "do not
re-draft") predates today's evidence and, per that memory, was scoped to the `acceptEdits` case.
**Submit a follow-up** (new `/feedback` submission, referencing `3ccd08fc`) carrying:

1. Today's clean, cross-agent-type reproduction table (§1.2) — three independent subagent types,
   eight of ten non-BACKLOG file-edit calls producing a real human-confirmation gap (6.2s-462.2s;
   the other two — `architect`'s own repeat-edits to a file its own run had already gotten approved
   for — sailed, see §1.3's flagged-not-resolved stickiness question), under a parent session
   independently confirmed (via its own transcript's `permission-mode` records) to have stayed in
   literal `bypassPermissions` throughout.
2. The sharper bug characterization from §2.3: **file-editing tools specifically** (`Write`/`Edit`),
   not `Bash`, fail to inherit the parent's bypass — a `Bash` rule/mode reaches a background
   subagent correctly every time in this evidence; a file-edit does not, ever.
3. The newly-identified mechanism candidate (§2.1-2.2): non-teammate `Agent`/`Task` dispatches have
   run as **background subagents by default since v2.1.232**, and the sub-agents doc's own
   "Background Subagents" section, plus the parent-mode-inheritance passage immediately adjacent to
   it, together promise no exception for this path — worth naming explicitly as the most likely
   locus of the regression, since neither passage was written (or at least is not written) with an
   acknowledged interaction between the two.
4. This session's own two-level-deep reproduction (§3, Test 2) — rules out "specific to a direct
   `teco` dispatch" as the explanation.

### 4.2 Config recommendation: revert `defaultMode` away from `bypassPermissions`

**Concretely (for `analyst`'s gate, then a follow-up implementation unit — U3 per the coordination
ledger; not applied by this document):** in `.claude/settings.json`, remove the
`"defaultMode": "bypassPermissions"` key (or set it back to unset / `"auto"` explicitly) and revert
the three `permissions.ask` rules that were added specifically to survive bypass
(`Edit(**/docs/BACKLOG.md)` plus the destructive-ops mirror rules) back to hooks, OR simply leave
those `ask` rules in place — they're harmless and arguably a strict improvement under `auto` too,
since Gen 3's own §1.3 already flagged that hook `"ask"` enforcement is *itself* unconfirmed under
`auto` (a still-open, separate finding), so having the same guarantees expressed as settings rules
in addition to hooks is a strictly safer position, not a regression, regardless of which mode is
active. The three `allow` rules (`Bash`, `Edit(**)`, `mcp__cypher__query`) are also safe to keep
under `auto` — they only ever *narrow* what still needs a decision, never widen it.

**Why revert, not just "accept the friction" as Gen 3 did:** Gen 3 weighed a real trade — `auto`'s
Bash-classifier safety net vs. `acceptEdits`/`bypass`'s promised frictionless delegated writes —
and correctly judged the trade not worth it *even if the promised benefit had been real*. Today's
evidence shows the benefit **was never real for the delegated-write case that motivated the
2026-08-29 pin** — `bypassPermissions` produces the *exact same* per-write confirmation friction
`auto` does (§1.2's gaps are the same shape and scale as friction this team has always had under
`auto`). That alone is Gen 3's own logic applied to a premise Gen 3 didn't have available: no
promised benefit survives contact with today's evidence, so there is no trade left to weigh —
reverting removes complexity (three extra `permissions.ask` rules, a mode pin that contradicts the
team's last recorded recommendation) for zero cost, whether or not any *additional* safety argument
holds up. The three points below were this document's original attempt at such an additional
argument; the first is now corrected following a live test, so treat it as a wash rather than a
second reason to revert — the friction finding carries the recommendation by itself:

- **Hook `"ask"` is *not* silently skipped under `bypassPermissions` — corrected from this
  document's earlier draft, which cited only a commit message and the user's own memory, not a
  verbatim doc statement (analyst review, Major finding 2).** I ran the live check the review asked
  for: a headless `tdd-engineer` dispatch (same nested-process methodology as §2.4) targeting
  `falkor-chat/docs/plans/_permission-test-scratch-cobb2.md` — a path on `tdd-engineer`'s own
  deny-list (`guard-tdd-broad-write.sh`, `docs/plans/*`) — under the live, unchanged
  `bypassPermissions` pin. Result: the `Write` call was blocked pre-write, the guard's own message
  came back verbatim in the tool result (`.../subagents/agent-a8cda714a0a091f39.jsonl`, `tool_use`
  → `tool_result` gap 0.055s, recorded as a `permission_denials` entry in the run's own JSON
  output), and no file was created. Contrast this directly against §2.4's clean, zero-friction runs
  for an *allowed* target under the identical bypass-pinned, headless setup: an explicit hook `ask`
  still produces a block; an explicit hook `allow` still sails. If hook `ask` were silently skipped
  under bypass, the denied case should have looked exactly like the allowed one — it didn't. This is
  one data point in a foreground/headless dispatch context specifically (§2.4 establishes headless
  mode never reaches `started_in_background: 1`), so it doesn't rule out a *different* result for a
  genuinely background-dispatched call — and, per the review's Pass 2 (§4.2's overclaim finding), it
  is inconsistent with, in a headless context this test can't fully separate from mode, "documented
  to be skipped" — `-p` mode is itself documented to run with no prompt surface by default
  (`code.claude.com/docs/en/headless`: Manual mode on every plan; `dontAsk` denies rather than
  prompts), so the instant denial is equally consistent with "`bypassPermissions` genuinely still
  enforces hook `ask`" and with "`-p` auto-denies any would-be-prompt regardless of which mode is
  configured" — this test cannot distinguish the two. What it does give is live, positive evidence
  (not mere absence of evidence) that the mechanism this team's eight
  doc-scoped write guards and the one broad-implementer deny-list guard rely on for their
  **escalation half** (AC-4 — "don't let an out-of-remit write through silently") keeps working
  under bypass, at least for this tested path. Under `auto`, this same guarantee was already flagged
  as *unconfirmed* (Gen 1's K-019, `claude-code.md`'s "ask" enforcement-gap callout) — today's
  result doesn't resolve that either way for `auto`, but it removes this document's own claim that
  `bypass` is *worse* than `auto` on this specific dimension; call it a wash on current evidence,
  not a documented loss.
- **Protected-path writes (`.git`, `.claude`) still prompt under current Claude Code versions even
  in bypass** (changelog `v2.1.78`, long since shipped) — a genuine, real mitigation, orthogonal to
  the AC-4 hook-`ask` question above (that's about this *team's own* doc-kind boundaries, e.g.
  `tdd-engineer` writing into `docs/reviews/*`, not `.git`/`.claude`).
- **The `permissions.ask` rules the 2026-08-29 pin added (destructive ops + `BACKLOG.md`) do
  survive bypass** (per docs: explicit `ask` rules and `AskUserQuestion` are on the
  "actions no mode auto-approves" list) — this is real and correctly reasoned in the commit
  message, and now sits alongside rather than substituting for the general per-agent doc-scope
  guard AC-4, which §4.2's live test found still functioning under bypass too.

Net: `bypassPermissions` today buys **nothing** over `auto` on the dimension it was adopted for
(delegated-write friction — identical either way, per §1.2/§3), and this document no longer finds a
confirmed additional safety cost either (the hook-`ask` claim above was corrected by live test, not
merely softened in wording). The recommendation to revert stands on the friction finding alone —
zero benefit for real complexity (a mode pin against the team's last recorded recommendation, three
extra `permissions.ask` rules) is enough on its own, independent of any safety trade. Reverting is
not a re-litigation of Gen 3's judgment call; it's correcting a decision made after Gen 3 closed on a
premise (today's evidence proves false) that Gen 3's own analysis didn't have to weigh.

### 4.3 Minor hygiene, not blocking

`.claude/settings.local.json`'s leftover `"Edit(**/docs/reviews/**)"` rule (from Gen 2's §U7 test,
never removed) is inert either way (§2.3) — flagged for whoever next touches that file, not urgent,
and outside this document's write authority (personal, untracked, outside `docs/plans/*`'s remit).

### 4.4 What this document does not resolve

- §1.3's stickiness-pattern question (which UI option a human clicked, if any) — flagged, not
  guessed at. Doesn't change the verdict either way (§1.3).
- Whether the background-dispatch-vs-mode-inheritance interaction (§2.1-2.2) is Anthropic's actual
  root cause internally, or whether some other undocumented mechanism produces the same symptom —
  this document establishes the doc-contradicting behavior and the most precise characterization
  and best-supported mechanism candidate this team can reach from the outside; only Anthropic can
  confirm the actual internal cause. That's exactly why §4.1 recommends filing, not guessing
  further.
- §2.4's isolation-lever test: whether `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` actually suppresses
  the gap for a genuinely background-dispatched (not headless/`-p`) subagent — untestable by a
  delegated subagent's own available tooling (no TTY-driving capability, no way to set an env var
  for an already-running process). Left for whoever can drive a real interactive test, or for the
  human to try directly; not a blocker for §4.2's revert, which doesn't depend on the answer.
- The unexpected observation in §2.4 (point 2) that a nested session's own top-level agent's hooks
  appeared to additionally constrain a subagent it dispatched, beyond that subagent's own hooks —
  flagged as a real, reproducible-once oddity worth a future look, not chased further here (out of
  this document's scope, and not needed to support §4.2's recommendation either way).

---

## 5. Cross-references

- `claude/docs/plans/agent-permission-friction.md` — Gen 1 (archived).
- `claude/docs/plans/write-guard-classifier-gap.md` + `-coordination.md` — Gen 2 (archived);
  §U7's empirical-test method is the direct template §3 above follows.
- `claude/docs/plans/permission-default-mode.md` — Gen 3 (archived, v2); §1.2's mode-inheritance
  quote and §5's cost analysis are both built on directly here, not re-derived.
- `skills/agent-standards/claude-code.md` — carries the running KB entry this document's findings
  (background-subagent dispatch default since v2.1.232; the file-edit-vs-Bash asymmetry; the
  `bypassPermissions`-doesn't-close-the-gap-either finding) should be folded into once this design
  is gated, per this team's standing KB-update convention.
- `falkor-chat/docs/plans/bare-call-key-shadowing-coordination.md` — source of today's fresh
  evidence (§1).
- `claude/docs/plans/bypass-permissions-subagent-gap-coordination.md` — the ledger this document is
  U1 of; not edited by this document.
