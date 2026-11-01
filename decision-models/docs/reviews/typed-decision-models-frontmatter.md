# U5.7 agent definitions: Claude Code frontmatter and hook semantics (cobb consult)

> **Status:** active · **Owner:** `cobb` · **Tracks:** —

**Verdict: approve with suggestions.** No field in the three definitions or in `settings.json` is rejected, and every field means what plan4 §3.4 and plan6 §3.3 assume. I found no blocker and no major. Two findings are worth acting on (F1, F2).

Advisory only. Nothing under `claude/` was touched. Sources are the Claude Code docs read on 2026-11-01 (`code.claude.com/docs/en/sub-agents`, `/hooks`, `/settings-reference`, `/tools-reference`, `/model-config`, `/interactive-mode`) and Pass 15 (e) of `docs/reviews/typed-decision-models4-impl.md`. The installed CLI is 2.1.316, as the probe report records. The docs pages carry no "current version" of their own, so each version floor below is the one the docs state.

Marks: **V** verified from the docs · **P** verified by a probe (Pass 15 (e), 2.1.316) · **I** inferred · **U** unknown.

## Answers

### (1) Frontmatter

| Field / value | Mark | Finding |
|---|---|---|
| `model: claude-opus-5-5` | V | `model-config` uses it as its example of a full model id. Needs 2.1.280 or later. |
| `model: claude-sonnet-5-5` | V/P | The docs name Sonnet 5.5 but do not spell the id. It is probed on a driver and a subagent (F7), and needs 2.1.284 or later. Claude Code does not validate a model id up front, so a wrong id fails on the first request. |
| model resolution | V | The `Agent` call's per-invocation `model` (alias only) beats the definition's `model`, which beats `CLAUDE_CODE_SUBAGENT_MODEL`. The env var came first before 2.1.251, and the runner drops it anyway. The guard's key whitelist (no `model` key) is therefore the control that keeps the pin. |
| `effort: medium` / `high` | V | Opus 5.5 and Sonnet 5.5 both accept `low`, `medium`, `high`, `xhigh` and `max`. `medium` is the default for both models, so the generator pin changes nothing in effect and is still worth recording. The value overrides the session effort but not `CLAUDE_CODE_EFFORT_LEVEL`, which the runner's env allowlist drops. A `maxEffortLevel` cap in any scope would lower it silently; the audit's `output_config.effort` check catches that loudly. Only `high` is probed (P); `medium` is V only. |
| `maxTurns: 6` / `14` | V/U | The docs place `maxTurns` on subagents. At the limit the subagent returns its output **marked partial**, not as an error, and it is resumable. That marking needs 2.1.246 or later, and it fits the plan's "`tool_result` present but stopped at `maxTurns`" row. U: whether a `SubagentHandback` call counts as a turn, so whether `P_max + 4` has margin. U: the documented semantics of `maxTurns` on a `--agent` main session (the driver's `3` is probed, P, and has one turn of margin). |
| `omitClaudeMd: true` | V/P | Needs 2.1.271 or later. It drops the user, project and local `CLAUDE.md`; **managed-policy files still load**, and the field is **ignored on a `--agent` main session** (so the driver loads the hierarchy, correctly unset there). On a machine with a managed `CLAUDE.md` the empty-`instructions` expectation will halt. That is the right result. |
| `permissionMode: acceptEdits`, spawned by the driver | V | Under `--permission-mode acceptEdits` the main conversation is already `acceptEdits`, and the docs say the subagent then runs in that mode and **ignores** the frontmatter value. The field is a no-op here and matches the runner's mode, so it is harmless. It does not bound anything (plan6 §3.3 already says so). The "other agent spawns it" case is the same as the main session, because the spawner is the `--agent` main thread. |
| `tools: Agent(dm-generator, dm-labeler)` | V/P | The docs show this exact comma form. It restricts spawnable types **only for a `claude --agent` main thread**, and in a subagent definition the parenthesised list is ignored. That is how `dm-driver` runs here. `Read, Write` on the subagents cannot spawn further, because `Agent` is not listed. |
| Unknown frontmatter keys | V | They are ignored without an error. A misspelt `omitClaudeMD` or `maxturns` would silently do nothing (F3). |

### (2) Hooks

| Item | Mark | Finding |
|---|---|---|
| `onFailure: "block"` | V | It exists on `command` and `http` handlers, takes `"continue"` (default) or `"block"`, and needs **2.1.295 or later**. The docs state all three failure shapes: a **timeout** cancels the hook and discards its output, and on `PreToolUse` the call proceeds unless `onFailure` is `block`; **cannot-start** is a non-blocking error unless `block`; **non-zero exit other than 0 or 2** is non-blocking unless `block`. A guard that prints nothing and exits 1 therefore blocks. Only the exit-1 case is probed (P, D2); timeout and cannot-start are V only. A missing `python3` under `sh -c` exits 127, which is the same class. |
| `timeout: 30` | V | The unit is seconds. The default for `command` is 600. 30 is valid and generous for a stdlib script. |
| `$CLAUDE_PROJECT_DIR` in `command` | V/U | The docs say the variable is exported to the spawned process and the shell form goes through `sh -c`, so `"$CLAUDE_PROJECT_DIR/.claude/guard_hook.py"` expands in the shell (V). The docs' own examples use `${CLAUDE_PROJECT_DIR}` and **recommend the exec form (`command` plus `args`)** for a path placeholder (V). U: the docs say nothing about `hook_success`, so what `hook_success.command` records is unknown. No probe has used the variable, and the plan already routes a mismatch to a loud `hookMissing` halt. |
| `PreToolUse` input | V | `agent_type` is the agent's `name` and is present in a subagent or under `--agent`. `agent_id`, `cwd`, `tool_use_id`, `permission_mode` and `effort` are common fields. Settings-level hooks fire inside subagents (V), which is what the single wiring relies on (P, F4). |
| A hook event for `SubagentHandback` | U | There is no dedicated event. `SubagentStart` and `SubagentStop` exist and are matched by agent type. `tools-reference` lists the tool names as the matcher strings, so a `*` matcher probably fires for it (I), and nothing documents that. |

### (3) `autoContinueAtUsageLimit` at project scope

**V, and the plan's wording is accurate.** The settings index row says "User or managed". The key's own entry is more precise: it is read from user settings, `--settings` and managed settings, and **when none of those sets it, a project or local file that sets it turns the feature off** rather than being ignored. Default `true`, and it needs 2.1.234 or later. So `false` in `exchange/.claude/settings.json` is honoured as an off-switch, which is plan6 §3.2's "a project file can only turn it off". It is overridden by any user, `--settings` or managed value, which is why a user `true` is a forbidden key. A managed `true` is invisible to `install`; the residual is small, because `interactive-mode` says the automatic wait is on by default **in interactive sessions** and that `-p` runs do not offer it at all.

### (4) Silently ignored, or rejected

Nothing is rejected: all frontmatter parses (I ran a YAML parse on the three files), all fields are documented, and the hook handler fields are valid. What a current release silently no-ops or may never exercise:
- `permissionMode` on both subagents (see (1)).
- The guard's `SubagentHandback` allow line (F1).
- `description` on `dm-driver`, which is used for delegation only, and the driver is a main session.
- Any misspelt or future-renamed frontmatter key (F3).

## Findings

### F1 — minor — `SubagentHandback` is documented as auto-mode only; the campaign runs `acceptEdits`

*Owner: `architect` (plan6 §3.3 and §3.4 wording), then `coder` only if the smoke settles it.*
- `tools-reference` says `SubagentHandback` is **provided only in auto mode**, to subagents the `Agent` tool runs locally other than forks, from 2.1.271, and that the classifier reviews the report.
- The runner launches `claude -p --permission-mode acceptEdits`. On the docs, the dm subagents should end with an ordinary `end_turn` reply and no hand-back.
- The P8-F2 corpus (206 of 207 runs end with one) comes from the SDK-hosted coordinating session, which the plan already says is not evidence for `-p`. It is probably an auto-mode session, which would explain it. I could not confirm that (I).
- Consequence: the guard's allow line and its tests are probably dead but harmless, and plan6 §3.3's conditional wording is already right. Name the auto-mode condition in §3.3 and the smoke row, so a hand-back that does appear under `acceptEdits` is recorded as a surprise and not an expectation.
- Whether a hook sees the tool stays **U** until the smoke.

### F2 — minor — nothing enforces a Claude Code version floor, and the guard's fail-closed rests on one

*Owner: `architect` for the rule, `coder` for the check.*
- Every field has a floor: `onFailure` 2.1.295, `omitClaudeMd` 2.1.271, Opus 5.5 2.1.280, Sonnet 5.5 2.1.284, the partial marking 2.1.246, the model-resolution order 2.1.251.
- The one that matters is `onFailure`. On a CLI that predates it, I infer (the docs say unknown *frontmatter* keys are ignored; they say nothing for hook handler keys) the field is dropped and a guard crash or timeout **fails open** into the permission flow, where `acceptEdits` allows a `Write` in `exchange/` (plan6 §3.3, D3).
- Suggest `install` and `run` refuse a `claude --version` below 2.1.295 and record the version in the `expectations` event. 2.1.316 passes today, so this costs nothing and removes a downgrade trap.

### F3 — minor — `install` validates no frontmatter key, and a typo is silent

*Owner: `coder`. It extends P15-n1.*
- The docs state unknown fields are ignored without an error. A one-letter slip in `omitClaudeMd` or `maxTurns` would install, pass the hash check against the (equally wrong) source, and be caught only by the smoke or the audit's expectations.
- Suggest `install` check each definition's keys against the documented set and the pinned values (P15-n1's fix already adds `model` and `effort`). Cheap and deterministic.

### F4 — nit — `permissionMode: acceptEdits` on the subagents is a documented no-op under this runner

*Owner: `architect`.* Keep it or drop it, but state in §3.3 that the parent's mode wins, so no reader counts it as a control. If the runner's mode ever changes, the frontmatter only takes effect when the parent is `default` or `dontAsk`.

### F5 — nit — the `$CLAUDE_PROJECT_DIR` form and the recorded `command`

*Owner: `architect` and `coder`, at the smoke.*
- The shell form works (V). The docs' preferred form is `command: "python3"` with `args: ["${CLAUDE_PROJECT_DIR}/.claude/guard_hook.py"]`, which has no shell and is safe against a space in the path.
- Do not switch before the smoke. Switching changes what `hook_success.command` records, and the plan compares it against `guardCommand`. Let the smoke's first batch show the recorded string, then pin whichever form it matches. The plan's absolute-path fallback stays the answer if neither matches.
- Whichever form is kept, `python3` must resolve on the child's allowlisted `PATH`. A miss exits 127 and blocks every call (V, via `onFailure`), which halts loudly.

### F6 — nit — the `maxTurns` margin is unmeasured

*Owner: `qa-engineer` at the smoke.* Record the turn count of a clean 50-item batch next to the `maxTurns: 2` shape, so the `P_max + 4` margin is a number and not an argument. The docs say a stop is a partial result and not an error, which the planned U5.5 fixture already assumes.

## Not touched

No file outside this note was written. The source definitions, `settings.json` and `guard_hook.py` are unchanged, and no kaizen file was touched (an advisory consult, not a promotion).
