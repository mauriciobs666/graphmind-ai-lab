# Typed Decision Models — Harness Spawn-Mode Probe — Evidence Record

> **Status:** active · **Owner:** `devops` · **Tracks:** U6.1 (`docs/plans/typed-decision-models6.md` §3.6)

## 1. Summary and fact index

This record keeps the evidence from the 2026-11-01 harness probe, which `docs/plans/typed-decision-models6.md` §2 (facts F1–F7) cites. The probe asked how an **interactive** `claude --agent` session launches, ends and loses subagents. It ran on Claude Code 2.1.316 in a trusted scratch folder outside the repo, driven through tmux on a real TTY. The kit is reproduced in the appendices so the probe can be rerun. §9 is reserved for U6.2's headless (`claude -p`) results.

Every path in this document that begins with `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/` is the **original session scratchpad** where the probe ran. That folder is ephemeral. The kit's files hard-code these paths, so a rerun from a different folder must rewrite them (Appendix I).

| Fact (plan6 §2) | Where it is shown here |
|---|---|
| F1 interactive spawn is background by default; completion via `<task-notification>` | §5.1 rows S1, S2, S3B; §7 sessions `503f4b34`, `d764701f` (`toolUseResult`, notification tags, `origin`, meta values); §4 panes A, B2 |
| F2 `run_in_background: false` and `CLAUDE_CODE_FORK_SUBAGENT=0` stay background; only `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` is foreground | §5.1 rows S2, S3, S3B; §7 session `16d9c155` (S3 `toolUseResult` keys, meta values); §4 panes B vs. A and B2 |
| F3 every completed run ends `END_TURN`; no `SubagentHandback` | §5.1 `terminal` and `hook-log HB` columns, `last2` detail lines; §7 `last2` per subagent |
| F4 settings, driver-frontmatter and subagent-frontmatter hooks all fire on every subagent tool call; 9 `hook_success` per 3 tool uses | §6 hook-log counts; §7 `tool_uses=3 hook_success attachments=9` per subagent |
| F5 SIGKILLed driver kills its background subagent; meta lacks `finished`; restart adds `<status>stopped</status>` keyed by task-id with no `<tool-use-id>` | §5.1/§5.2 row S4 and the diff (§5.3); §7 session `fb3a881a`; §4.5 restart pane |
| F6 `SubagentStop` with the driver's `agent_type` and a foreign `agent_id` after every driver `Stop` | §6.2 timing table (8 events) |
| F7 attempt 1 on Opus 5.5: refusal entry, validation error, safety-classifier user entry, Model-switch dialog | §3 dialog text and panes; §7 attempt-1 session `c5ce5291` (`REFUSAL` entry keys, `str` `toolUseResult`, `isMeta` user entry) |

## 2. Method

- **Host and CLI.** WSL2 Linux, `/home/mauricio/.local/bin/claude` version `2.1.316 (Claude Code)`, logged in on a Claude Pro account.
- **Kit.** `cobb` built it (Appendices A–H). The driver `probe-driver` runs as the main session via `claude --agent probe-driver` and is allowed only `Agent(probe-sub)`. The subagent `probe-sub` mirrors `dm-labeler`: `Read, Write`, `model: claude-sonnet-5-5`, `omitClaudeMd`, `acceptEdits`. Allow-all, log-only `PreToolUse` hooks are wired three ways, each tagged in `hook-log.jsonl` by `src`: project settings (`settings`, which also wires `Stop` and `SubagentStop`), the driver's frontmatter (`fm-driver`) and the subagent's frontmatter (`fm-sub`).
- **Launch flags.** Every launch used `FLAGS="--strict-mcp-config --mcp-config ./empty-mcp.json --permission-mode acceptEdits"`, the `dm-driver` line from plan4 §3.5.
- **Terminal.** Each launch ran inside a tmux session on a private socket, at 200×50: `tmux -L probe new-session -d -s probe -x 200 -y 50`. Keystrokes went in with `send-keys`, and the pane was captured with `capture-pane -p` every 3 s into `panes/`.
- **Clean environment (deviation from `RUN.md`).** The tmux server was started under `env -i` with only `HOME`, `USER`, `LOGNAME`, `SHELL=/bin/bash`, `PATH=/home/mauricio/.local/bin:/usr/local/bin:/usr/bin:/bin`, `LANG` and `TERM=xterm-256color`. The coordinating session's own `CLAUDECODE`, `CLAUDE_CODE_CHILD_SESSION`, `CLAUDE_CODE_AGENT`, messaging-socket and similar variables would otherwise have been inherited, making the probe sessions nested child sessions. The pane confirmed `env | grep -c '^CLAUDE'` = `0`.
- **Launches.** Step 0 trusted the folder. In the trust prompt the cursor starts on "No, exit"; `Down` then `Enter` selected "Yes, I trust this folder". After that came launch A (S1, S2), launch B (S3, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`), launch B2 (S3b, `CLAUDE_CODE_FORK_SUBAGENT=0`), launch C (S4, SIGKILLed mid-run) and the restart (`claude -c --agent probe-driver $FLAGS "PHASE R"`). Each launch ended with `/exit` except C.
- **Two attempts.** Attempt 1 left the driver on the session default model (Opus 5.5). It stopped at an unanticipated Model-switch dialog (§3). The coordinator then added `model: claude-sonnet-5-5` to `probe-driver.md`. Attempt 2 ran from a clean slate: `hook-log.jsonl` and `ans/*` deleted, the project dir moved aside to `…-scratchpad-dm-probe-attempt1`, and attempt 1's panes moved to `panes/attempt1/`. Every launch in attempt 2 showed `Sonnet 5.5 · Claude Pro` in the banner.

## 3. Attempt 1 — stopped at the Model-switch dialog (F7)

Launch A only, driver on Opus 5.5 (banner `Opus 5.5 · Claude Pro`). S1 spawned, was backgrounded and completed. The driver's next `Agent` call for S2 had only `description` and `subagent_type`, and the harness answered `InputValidationError`. Then the safeguards message and the modal dialog appeared. Nothing was selected in the dialog, no `/config` setting was changed, and the session was ended by killing the probe's own tmux server.

Pane `panes/attempt1/A-07.txt`, the first capture showing the dialog (it is still shown in `A-08` through `A-40`), verbatim apart from blank lines:

````text
❯ Carry out exactly the phase named in this message, as your instructions define it.
  PHASE A
● probe-sub(probe S1)
  ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)
● waiting for S1
✻ Waiting for 1 background agent to finish
● Agent "probe S1" finished · 7s
  ⎿  Initializing…
  ⎿  Invalid tool parameters
● Opus 5.5's safeguards stopped the response above · continuing once with that noted
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 ☐ Model switch
│ Opus 5.5's safeguards flagged this session. You may be seeing this for the first time: Opus 5.5 is more capable and has stronger safeguards as a result, which can sometimes flag non-cybersecurity
│ work. We're improving these safeguards to reduce the amount of incorrectly flagged messages. Switch to Opus 4.8 and keep going whenever this happens? You can change this later in /config.
❯ 1. Switch automatically
     Continue on Opus 4.8 now, and switch without asking from now on
  2. Stay on Opus 5.5
     Stop here without switching, and ask me each time a message is flagged
  3. Type something.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. Chat about this
Enter to select · ↑/↓ to navigate · Esc to cancel
````

From `A-14` onward, the line `✻ Waiting for API response · will retry in 2m 40s · check your network` also showed under the dialog. Earlier panes: `A-03` shows `● probe-sub(probe S1)  ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)`, and `A-04` shows `● waiting for S1` / `✻ Waiting for 1 background agent to finish`. That is the same background shape attempt 2 recorded.

Attempt 1's partial `read_results.py` output (`panes/attempt1/A-partial-results.txt`), verbatim:

````text
project dir: /home/mauricio/.claude/projects/-tmp-claude-1000--home-mauricio-prg-graphmind-ai-lab-c6eed315-dc2c-4b2c-99e2-eece159b76c4-scratchpad-dm-probe
spawn  rib-key  toolUseResult.status  requestShape  launchedWithDefaults  finished  notification(kind/status)               terminal  hook-log HB
-----  -------  --------------------  ------------  --------------------  --------  --------------------------------------  --------  -----------
S1     -        async_launched        background    True                  True      queue-operation:enqueue/completed;api-  END_TURN  no         
?      -        str                   no meta       no meta               no meta   none                                    -         no         

S1: session=c5ce5291-c849-4d90-9c83-995bcbf162a5 tool_use=toolu_017kSnRXBrNxUQxyazif4xbY agent=aaf7e059180c0fe69 input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=10:56:07
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: [['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage']]
?: session=c5ce5291-c849-4d90-9c83-995bcbf162a5 tool_use=toolu_01BTurWpwnQuXjMU3sGcQBHU agent=None input_keys=['description', 'subagent_type']
    result_keys='InputValidationError: [\n  {\n    "expected": "string",\n    "code": "invalid_t
    meta=absent
    last2: no transcript
    hook entries in transcript for SubagentHandback: -
    notification tags: []

hook-log counts (src, event, agent_type, tool_name):
     1  ('fm-driver', 'PreToolUse', 'probe-driver', 'Agent')
     2  ('fm-driver', 'PreToolUse', 'probe-sub', 'Read')
     1  ('fm-driver', 'PreToolUse', 'probe-sub', 'Write')
     2  ('fm-sub', 'PreToolUse', 'probe-sub', 'Read')
     1  ('fm-sub', 'PreToolUse', 'probe-sub', 'Write')
     1  ('settings', 'PreToolUse', 'probe-driver', 'Agent')
     2  ('settings', 'PreToolUse', 'probe-sub', 'Read')
     1  ('settings', 'PreToolUse', 'probe-sub', 'Write')
     1  ('settings', 'Stop', 'probe-driver', None)
     1  ('settings', 'SubagentStop', 'probe-driver', None)
     1  ('settings', 'SubagentStop', 'probe-sub', None)
````

## 4. Attempt 2 — pane observations per launch

Panes are quoted verbatim from `panes/<launch>-NN.txt`, 3 s apart, with banner and rule lines left out. Every launch's banner read `Sonnet 5.5 · Claude Pro`.

### 4.1 Launch A — S1 (no key), S2 (`run_in_background: false`): immediate return, later notification

```text
A-02  ● probe-sub(probe S1)
        ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)
      ● waiting for S1
      ✻ Waiting for 1 background agent to finish
A-06  ● Agent "probe S1" finished · 15s
      ● probe-sub(probe S2)
        ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)
A-07  ● waiting for S2
A-09  ● Agent "probe S2" finished · 6s
      ● PHASE A done
      ✻ Baked for 27s · done 11:00 AM
```

No blocking spinner was shown for either spawn. The driver ended its turn with `waiting for <id>`, and the completion arrived as a separate `Agent "…" finished` notification that started a new driver turn. After each driver turn the input box showed prompt-suggestion ghost text predicting the next phase (`❯ PHASE B` in `A-02`, `A-07`, `A-09`; `❯ PHASE B2` after B; `❯ PHASE C` after B2). It was never submitted; typing `/exit` replaced it, and Ctrl+U cleared it before the restart command.

### 4.2 Launch B — S3, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`: blocking spinner

```text
B-01  ✶ Nucleating… (5s · ↓ 16 tokens)
B-02    probe-sub(probe S3)
      * Nucleating… (8s · ↓ 291 tokens)
B-03  ● probe-sub(probe S3)
      ✶ Nucleating… (11s · ↓ 522 tokens)
B-04  ✢ Nucleating… (14s · ↓ 556 tokens)
B-05    ⎿  Done (3 tool uses · 3.6k tokens · 9s)
      ● PHASE B done
      ✻ Sautéed for 16s · done 11:00 AM
```

One continuous spinner ran for the whole subagent run. The result appeared inline (`Done (3 tool uses …)`), with no "Backgrounded agent" line and no separate notification.

### 4.3 Launch B2 — S3b, `CLAUDE_CODE_FORK_SUBAGENT=0`: immediate return, later notification

```text
B2-02  ● probe-sub(probe S3b)
         ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)
B2-03  ● waiting for S3b
       ✻ Waiting for 1 background agent to finish
B2-05  ● Agent "probe S3b" finished · 13s
B2-06  ● PHASE B2 done
       ✻ Sautéed for 18s · done 11:01 AM
```

### 4.4 Launch C — S4, SIGKILLed mid-run

```text
C-after-kill  ● probe-sub(probe S4)
                ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)
              ● waiting for S4
              ✻ Waiting for 1 background agent to finish
              ❯ KilledR
              mauricio@g15:…/scratchpad/dm-probe$
```

The watch loop polled `grep -c '/long/f' hook-log.jsonl` every second. The count read 0, 0, 0, 3, 6, 9, 12, 18, 21, 24, then 30 at t = 33 s. At 30 it resolved the PID: the process whose `/proc/<pid>/comm` was `claude` and whose command line was `claude --agent probe-driver --strict-mcp-config --mcp-config ./empty-mcp.json --permission-mode acceptEdits PHASE C` (PID 3391456). It ran `kill -KILL` on it, then ran `read_results.py` at once. Three hooks fire per Read, so 30 lines = 10 chain files read. The count stayed at 30 afterwards (rechecked after 5 s), so the subagent died with the driver. No `ans/s4.txt` was written. Its transcript holds 12 tool uses: 2 Reads of `req/s4.txt` and `long/f001`–`f010`.

### 4.5 Restart — `claude -c --agent probe-driver $FLAGS "PHASE R"`

The CLI accepted `-c` together with `--agent`, so `RUN.md`'s fallback was not needed. It reopened session `fb3a881a-…`:

```text
R-01  ❯ Carry out exactly the phase named in this message, as your instructions define it.
        PHASE C
      ● probe-sub(probe S4)
        ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)
      ● waiting for S4
      ✻ Baked for 4s · done 11:01 AM
        PHASE R
      ● restarted
      ✻ Worked for 1s · done 11:02 AM
      ● Background agent "probe S4" didn't finish before the previous session ended
```

Nothing changed in `R-02` to `R-12` (36 s). S4 was neither resumed nor relaunched, and the `/long/f` count stayed at 30.

## 5. Results tables (verbatim)

Both files are `read_results.py` output (Appendix F), copied byte for byte. Nothing in them is re-derived. The `notification` column is cut at 38 characters by the script itself.

### 5.1 `results-after-kill.txt` — written immediately after the SIGKILL, before any restart

````text
project dir: /home/mauricio/.claude/projects/-tmp-claude-1000--home-mauricio-prg-graphmind-ai-lab-c6eed315-dc2c-4b2c-99e2-eece159b76c4-scratchpad-dm-probe
spawn  rib-key  toolUseResult.status  requestShape  launchedWithDefaults  finished  notification(kind/status)               terminal                   hook-log HB
-----  -------  --------------------  ------------  --------------------  --------  --------------------------------------  -------------------------  -----------
S1     -        async_launched        background    True                  True      queue-operation:enqueue/completed;api-  END_TURN                   no         
S2     false    async_launched        background    True                  True      queue-operation:enqueue/completed;api-  END_TURN                   no         
S3     -        completed             foreground    absent                True      none                                    END_TURN                   no         
S3B    -        async_launched        background    True                  True      queue-operation:enqueue/completed;api-  END_TURN                   no         
S4     -        async_launched        background    True                  absent    none                                    NONE(asst[tool_use]:Read)  no         

S1: session=503f4b34-36b5-40d1-854a-58d787adc4fa tool_use=toolu_01HYaLWpm8JbuZsSXb6kEMne agent=a94a1bdbc93a47426 input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=10:59:58
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: [['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage']]
S2: session=503f4b34-36b5-40d1-854a-58d787adc4fa tool_use=toolu_019EpkEMtRZF6iYhFHHSTvvz agent=a3f73dac011045e49 input_keys=['description', 'prompt', 'run_in_background', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:00:07
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: [['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage']]
S3: session=16d9c155-59af-485c-8882-273ad53ff62c tool_use=toolu_018tdeA6Ad2VS7EnNtJV3wRx agent=ad58529de637ffead input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'agentType', 'canContinueAgent', 'content', 'harnessNoteCount', 'har
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:00:43
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: []
S3B: session=d764701f-54f0-47eb-8697-9b84abcc187a tool_use=toolu_01VhBzu5RuzHDGU4P7MNqyn9 agent=a120561ff37cb1731 input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:01:14
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: [['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage']]
S4: session=fb3a881a-d3f1-4a9a-8817-87d30485f1bd tool_use=toolu_014iC3NrNvkPLrcU3mQMUuMp agent=ab929d875b9bf66dc input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:01:35
    last2: asst[tool_use]:Read | user:tool_result
    hook entries in transcript for SubagentHandback: n/a
    notification tags: []

hook-log counts (src, event, agent_type, tool_name):
     5  ('fm-driver', 'PreToolUse', 'probe-driver', 'Agent')
    20  ('fm-driver', 'PreToolUse', 'probe-sub', 'Read')
     4  ('fm-driver', 'PreToolUse', 'probe-sub', 'Write')
    20  ('fm-sub', 'PreToolUse', 'probe-sub', 'Read')
     4  ('fm-sub', 'PreToolUse', 'probe-sub', 'Write')
     5  ('settings', 'PreToolUse', 'probe-driver', 'Agent')
    20  ('settings', 'PreToolUse', 'probe-sub', 'Read')
     4  ('settings', 'PreToolUse', 'probe-sub', 'Write')
     7  ('settings', 'Stop', 'probe-driver', None)
     7  ('settings', 'SubagentStop', 'probe-driver', None)
     4  ('settings', 'SubagentStop', 'probe-sub', None)
````

### 5.2 `results-after-restart.txt` — written after the restart session exited

````text
project dir: /home/mauricio/.claude/projects/-tmp-claude-1000--home-mauricio-prg-graphmind-ai-lab-c6eed315-dc2c-4b2c-99e2-eece159b76c4-scratchpad-dm-probe
spawn  rib-key  toolUseResult.status  requestShape  launchedWithDefaults  finished  notification(kind/status)               terminal                   hook-log HB
-----  -------  --------------------  ------------  --------------------  --------  --------------------------------------  -------------------------  -----------
S1     -        async_launched        background    True                  True      queue-operation:enqueue/completed;api-  END_TURN                   no         
S2     false    async_launched        background    True                  True      queue-operation:enqueue/completed;api-  END_TURN                   no         
S3     -        completed             foreground    absent                True      none                                    END_TURN                   no         
S3B    -        async_launched        background    True                  True      queue-operation:enqueue/completed;api-  END_TURN                   no         
S4     -        async_launched        background    True                  absent    none                                    NONE(asst[tool_use]:Read)  no         

S1: session=503f4b34-36b5-40d1-854a-58d787adc4fa tool_use=toolu_01HYaLWpm8JbuZsSXb6kEMne agent=a94a1bdbc93a47426 input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=10:59:58
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: [['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage']]
S2: session=503f4b34-36b5-40d1-854a-58d787adc4fa tool_use=toolu_019EpkEMtRZF6iYhFHHSTvvz agent=a3f73dac011045e49 input_keys=['description', 'prompt', 'run_in_background', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:00:07
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: [['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage']]
S3: session=16d9c155-59af-485c-8882-273ad53ff62c tool_use=toolu_018tdeA6Ad2VS7EnNtJV3wRx agent=ad58529de637ffead input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'agentType', 'canContinueAgent', 'content', 'harnessNoteCount', 'har
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:00:43
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: []
S3B: session=d764701f-54f0-47eb-8697-9b84abcc187a tool_use=toolu_01VhBzu5RuzHDGU4P7MNqyn9 agent=a120561ff37cb1731 input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:01:14
    last2: user:tool_result | asst[end_turn]:text
    hook entries in transcript for SubagentHandback: n/a
    notification tags: [['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage']]
S4: session=fb3a881a-d3f1-4a9a-8817-87d30485f1bd tool_use=toolu_014iC3NrNvkPLrcU3mQMUuMp agent=ab929d875b9bf66dc input_keys=['description', 'prompt', 'subagent_type']
    result_keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', '
    meta=['agentType', 'description', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId'] mtime=11:01:35
    last2: asst[tool_use]:Read | user:tool_result
    hook entries in transcript for SubagentHandback: n/a
    notification tags: []

hook-log counts (src, event, agent_type, tool_name):
     5  ('fm-driver', 'PreToolUse', 'probe-driver', 'Agent')
    20  ('fm-driver', 'PreToolUse', 'probe-sub', 'Read')
     4  ('fm-driver', 'PreToolUse', 'probe-sub', 'Write')
    20  ('fm-sub', 'PreToolUse', 'probe-sub', 'Read')
     4  ('fm-sub', 'PreToolUse', 'probe-sub', 'Write')
     5  ('settings', 'PreToolUse', 'probe-driver', 'Agent')
    20  ('settings', 'PreToolUse', 'probe-sub', 'Read')
     4  ('settings', 'PreToolUse', 'probe-sub', 'Write')
     8  ('settings', 'Stop', 'probe-driver', None)
     8  ('settings', 'SubagentStop', 'probe-driver', None)
     4  ('settings', 'SubagentStop', 'probe-sub', None)
````

### 5.3 `diff results-after-kill.txt results-after-restart.txt` (exit status 1)

````text
50,51c50,51
<      7  ('settings', 'Stop', 'probe-driver', None)
<      7  ('settings', 'SubagentStop', 'probe-driver', None)
---
>      8  ('settings', 'Stop', 'probe-driver', None)
>      8  ('settings', 'SubagentStop', 'probe-driver', None)
````

The only change is the restart session's own driver turn (one `Stop` plus its paired `SubagentStop`, §6.2). S4's row did not change: `finished` stayed absent, and `notification` stayed `none`. `read_results.py` matches notifications on `<tool-use-id>`, and the restart's `stopped` notification has none (§7, session `fb3a881a`).

## 6. Hook-log counts

### 6.1 Counts by (src, event, agent_type, tool_name), attempt 2, after the restart

This is the closing block of `results-after-restart.txt` (§5.2), repeated here for lookup. `hook-log.jsonl` had 102 lines, the sum of these counts.

````text
hook-log counts (src, event, agent_type, tool_name):
     5  ('fm-driver', 'PreToolUse', 'probe-driver', 'Agent')
    20  ('fm-driver', 'PreToolUse', 'probe-sub', 'Read')
     4  ('fm-driver', 'PreToolUse', 'probe-sub', 'Write')
    20  ('fm-sub', 'PreToolUse', 'probe-sub', 'Read')
     4  ('fm-sub', 'PreToolUse', 'probe-sub', 'Write')
     5  ('settings', 'PreToolUse', 'probe-driver', 'Agent')
    20  ('settings', 'PreToolUse', 'probe-sub', 'Read')
     4  ('settings', 'PreToolUse', 'probe-sub', 'Write')
     8  ('settings', 'Stop', 'probe-driver', None)
     8  ('settings', 'SubagentStop', 'probe-driver', None)
     4  ('settings', 'SubagentStop', 'probe-sub', None)
````

Per subagent, from `hook-log.jsonl` matched on `agent_id`:
- S1, S2, S3 and S3b each made 2 Reads and 1 Write. Each tool call was logged once by each of `settings`, `fm-driver` and `fm-sub`.
- S4 (`ab929d875b9bf66dc`) logged `Read` 12 times under each of the three `src` values: 2 of `req/s4.txt` and 10 under `/long/`.
- No `SubagentHandback` tool name occurs anywhere in the log.

### 6.2 `Stop` / `SubagentStop` / driver `Agent` timing (F6)

All of attempt 2's `Stop` and `SubagentStop` events, plus the driver's `PreToolUse:Agent` events (`src` `settings`), as fields from `hook-log.jsonl`:

```text
13:59:43.897 503f4b34 PreToolUse:Agent   probe-driver  None
13:59:44.842 503f4b34 Stop               probe-driver  None
13:59:45.972 503f4b34 SubagentStop       probe-driver  ade7d3e86041e1c38
13:59:58.896 503f4b34 SubagentStop       probe-sub     a94a1bdbc93a47426
14:00:01.296 503f4b34 PreToolUse:Agent   probe-driver  None
14:00:02.335 503f4b34 Stop               probe-driver  None
14:00:03.547 503f4b34 SubagentStop       probe-driver  aca13565752ab1359
14:00:07.761 503f4b34 SubagentStop       probe-sub     a3f73dac011045e49
14:00:08.892 503f4b34 Stop               probe-driver  None
14:00:10.144 503f4b34 SubagentStop       probe-driver  aa38d24d03031dfed
14:00:33.975 16d9c155 PreToolUse:Agent   probe-driver  None
14:00:43.841 16d9c155 SubagentStop       probe-sub     ad58529de637ffead
14:00:44.704 16d9c155 Stop               probe-driver  None
14:00:45.880 16d9c155 SubagentStop       probe-driver  ab8f6185443bd4adf
14:01:01.214 d764701f PreToolUse:Agent   probe-driver  None
14:01:04.392 d764701f Stop               probe-driver  None
14:01:10.803 d764701f SubagentStop       probe-driver  a9883d3f1891deffa
14:01:14.726 d764701f SubagentStop       probe-sub     a120561ff37cb1731
14:01:16.404 d764701f Stop               probe-driver  None
14:01:18.939 d764701f SubagentStop       probe-driver  aac7541f30fbd3aa3
14:01:35.941 fb3a881a PreToolUse:Agent   probe-driver  None
14:01:37.446 fb3a881a Stop               probe-driver  None
14:01:38.676 fb3a881a SubagentStop       probe-driver  af1d8d9d7b4d1e5fa
14:02:43.084 fb3a881a Stop               probe-driver  None
14:02:44.211 fb3a881a SubagentStop       probe-driver  addff5e89e60b5441
```

Every driver `Stop` (8 of 8, the restart's included) is followed 1.1–6.4 s later (seven of the eight within 2.6 s) by a `SubagentStop` with `agent_type` `probe-driver` and a fresh `agent_id` that matches no spawn. None of those 8 ids appears anywhere in the project dir: no `subagents/agent-<id>.*` file exists, and a recursive grep finds nothing. The hook logs only key names (`input_keys` includes `agent_transcript_path` and `last_assistant_message`), not values, so the transcripts cannot say what this agent is. The timing matches the prompt-suggestion ghost text that appeared after each driver turn (§4.1). That link is **an inference, not a proven fact**: logging `last_assistant_message` and `agent_transcript_path` on `SubagentStop` would settle it. Either way, an audit that counts `SubagentStop` must filter on `agent_type`.

## 7. Key-and-shape excerpts from the transcripts

The excerpts below come from a stdlib extractor run over both project dirs: attempt 1 at `…-scratchpad-dm-probe-attempt1` and attempt 2 at `…-scratchpad-dm-probe`, both under `/home/mauricio/.claude/projects/`. The extractor prints only:
- entry and attachment **type names** with counts;
- `Agent` `tool_use` input **keys**, plus the `description` and `run_in_background` values;
- `toolUseResult` **keys** with `status`, `isAsync` and `agentId`;
- for a `<task-notification>` entry: its type, its `origin` object, its tag set, `<status>`, `<task-id>` and whether a `<tool-use-id>` tag exists;
- the `refusal` entry's top-level keys, `message` keys, content block types and `tool_use` input keys;
- the keys of the user entry that mentions the safety classifier;
- per subagent: meta keys and values, the number of distinct `tool_use` ids, the number of `hook_success` attachments with their keys, and the last two conversation entries' types.

It prints no message text, tool input or output, hook `content`/`stdout`, or attachment body. The `environment`, `session_context` and `credential_org` attachments appear **only as type names in the counts**; their contents were never read into this record.

Two notes on reading it. Each `Agent` `tool_use` is printed twice because the transcript writes it in two assistant entries with the same id; `tool_uses=` counts distinct ids. The `hook_success` counts on the session line are the **driver's** own (the `fm-driver` and `settings` `PreToolUse` on each `Agent` call); the per-subagent lines count the subagent transcript's.

````text
## project dir -tmp-claude-1000--home-mauricio-prg-graphmind-ai-lab-c6eed315-dc2c-4b2c-99e2-eece159b76c4-scratchpad-dm-probe-attempt1

### session c5ce5291-c849-4d90-9c83-995bcbf162a5
entry types: ['agent-setting', 'api-request', 'api-request-blob', 'api-request-shape', 'assistant', 'atis-latch', 'attachment', 'cost-state', 'file-history-snapshot', 'last-prompt', 'mode', 'permission-mode', 'queue-operation', 'system', 'tools-used', 'user']
attachment types (counts): {'agent_listing_delta': 1, 'credential_org': 1, 'date': 1, 'environment': 1, 'hook_success': 2, 'model': 1, 'prompt_snapshot': 2, 'session_context': 1}
- Agent tool_use id=toolu_017kSnRXBrNxUQxyazif4xbY input keys=['description', 'prompt', 'subagent_type'] description='probe S1' run_in_background=<absent>
  toolUseResult keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', 'outputFile', 'prompt', 'resolvedModel', 'status'] status='async_launched' isAsync=True agentId=aaf7e059180c0fe69
- Agent tool_use id=toolu_017kSnRXBrNxUQxyazif4xbY input keys=['description', 'prompt', 'subagent_type'] description='probe S1' run_in_background=<absent>
- notification entry type=queue-operation origin=null tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=aaf7e059180c0fe69 has tool-use-id=True
- notification entry type=user origin={"kind": "task-notification", "producer": "session-task", "runId": "0mvxw7u2n-f8228ff3"} tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=aaf7e059180c0fe69 has tool-use-id=True
- Agent tool_use id=toolu_01BTurWpwnQuXjMU3sGcQBHU input keys=['description', 'subagent_type'] description='probe S2' run_in_background=<absent>
- Agent tool_use id=toolu_01BTurWpwnQuXjMU3sGcQBHU input keys=['description', 'subagent_type'] description='probe S2' run_in_background=<absent>
- REFUSAL entry keys=['advisorModel', 'apiBlockIndex', 'cwd', 'effort', 'entrypoint', 'gitBranch', 'isSidechain', 'message', 'parentUuid', 'perTurnEffort', 'requestId', 'requestRef', 'requestedModel', 'sessionId', 'session_id', 'timestamp', 'type', 'userType', 'uuid', 'version', 'wireToolInputs'] message keys=['container', 'content', 'context_management', 'diagnostics', 'id', 'input_transformations', 'model', 'role', 'stop_details', 'stop_reason', 'stop_sequence', 'type', 'usage'] content block types=['tool_use'] tool_use input keys=[['description', 'subagent_type']]
  toolUseResult is str (len 191), starts 'InputValidationError: '
- user string entry mentioning 'safety classifier' (keys=['cwd', 'entrypoint', 'gitBranch', 'isMeta', 'isSidechain', 'message', 'parentUuid', 'promptId', 'refusedMessageId', 'sessionId', 'session_id', 'timestamp', 'turnCompanion', 'type', 'userType', 'uuid', 'version']; isMeta=True)

  subagent agent-aaf7e059180c0fe69: meta keys=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId']
   meta values: {'agentType': 'probe-sub', 'description': 'probe S1', 'finished': True, 'isBuiltIn': False, 'launchedWithDefaults': True, 'requestNonInteractive': True, 'requestShape': 'background', 'spawnDepth': 1}
   tool_uses=3 hook_success attachments=9 (distinct toolUseIDs 3) hookEvents=[('PreToolUse', True)]
   hook_success attachment keys: ['command', 'content', 'durationMs', 'exitCode', 'hookEvent', 'hookName', 'stderr', 'stdout', 'toolUseID', 'type']
   last2: user:tool_result | assistant[stop_reason=end_turn]:text
## project dir -tmp-claude-1000--home-mauricio-prg-graphmind-ai-lab-c6eed315-dc2c-4b2c-99e2-eece159b76c4-scratchpad-dm-probe

### session 16d9c155-59af-485c-8882-273ad53ff62c
entry types: ['agent-setting', 'api-request', 'api-request-blob', 'api-request-shape', 'assistant', 'atis-latch', 'attachment', 'cost-state', 'file-history-snapshot', 'last-prompt', 'mode', 'permission-mode', 'system', 'tools-used', 'user']
attachment types (counts): {'agent_listing_delta': 1, 'credential_org': 1, 'date': 1, 'environment': 1, 'hook_success': 2, 'model': 1, 'prompt_snapshot': 2, 'session_context': 1}
- Agent tool_use id=toolu_018tdeA6Ad2VS7EnNtJV3wRx input keys=['description', 'prompt', 'subagent_type'] description='probe S3' run_in_background=<absent>
- Agent tool_use id=toolu_018tdeA6Ad2VS7EnNtJV3wRx input keys=['description', 'prompt', 'subagent_type'] description='probe S3' run_in_background=<absent>
  toolUseResult keys=['agentId', 'agentType', 'canContinueAgent', 'content', 'harnessNoteCount', 'harnessSectionHash', 'harnessTailCount', 'prompt', 'resolvedModel', 'status', 'toolStats', 'toolsUsed', 'totalDurationMs', 'totalTokens', 'totalToolUseCount', 'usage'] status='completed' isAsync=<absent> agentId=ad58529de637ffead

  subagent agent-ad58529de637ffead: meta keys=['agentType', 'description', 'finished', 'isBuiltIn', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId']
   meta values: {'agentType': 'probe-sub', 'description': 'probe S3', 'finished': True, 'isBuiltIn': False, 'requestNonInteractive': False, 'requestShape': 'foreground', 'spawnDepth': 1}
   tool_uses=3 hook_success attachments=9 (distinct toolUseIDs 3) hookEvents=[('PreToolUse', True)]
   hook_success attachment keys: ['command', 'content', 'durationMs', 'exitCode', 'hookEvent', 'hookName', 'stderr', 'stdout', 'toolUseID', 'type']
   last2: user:tool_result | assistant[stop_reason=end_turn]:text

### session 503f4b34-36b5-40d1-854a-58d787adc4fa
entry types: ['agent-setting', 'api-request', 'api-request-blob', 'api-request-shape', 'assistant', 'atis-latch', 'attachment', 'cost-state', 'file-history-snapshot', 'last-prompt', 'mode', 'permission-mode', 'queue-operation', 'system', 'tools-used', 'user']
attachment types (counts): {'agent_listing_delta': 1, 'credential_org': 1, 'date': 1, 'environment': 1, 'hook_success': 4, 'model': 1, 'prompt_snapshot': 2, 'session_context': 1}
- Agent tool_use id=toolu_01HYaLWpm8JbuZsSXb6kEMne input keys=['description', 'prompt', 'subagent_type'] description='probe S1' run_in_background=<absent>
  toolUseResult keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', 'outputFile', 'prompt', 'resolvedModel', 'status'] status='async_launched' isAsync=True agentId=a94a1bdbc93a47426
- Agent tool_use id=toolu_01HYaLWpm8JbuZsSXb6kEMne input keys=['description', 'prompt', 'subagent_type'] description='probe S1' run_in_background=<absent>
- notification entry type=queue-operation origin=null tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=a94a1bdbc93a47426 has tool-use-id=True
- notification entry type=user origin={"kind": "task-notification", "producer": "session-task", "runId": "0mvxwcn55-6cd4f936"} tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=a94a1bdbc93a47426 has tool-use-id=True
- Agent tool_use id=toolu_019EpkEMtRZF6iYhFHHSTvvz input keys=['description', 'prompt', 'run_in_background', 'subagent_type'] description='probe S2' run_in_background=false
- Agent tool_use id=toolu_019EpkEMtRZF6iYhFHHSTvvz input keys=['description', 'prompt', 'run_in_background', 'subagent_type'] description='probe S2' run_in_background=false
  toolUseResult keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', 'outputFile', 'prompt', 'resolvedModel', 'status'] status='async_launched' isAsync=True agentId=a3f73dac011045e49
- notification entry type=queue-operation origin=null tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=a3f73dac011045e49 has tool-use-id=True
- notification entry type=user origin={"kind": "task-notification", "producer": "session-task", "runId": "0mvxwd0k7-4dadf10d"} tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=a3f73dac011045e49 has tool-use-id=True

  subagent agent-a3f73dac011045e49: meta keys=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId']
   meta values: {'agentType': 'probe-sub', 'description': 'probe S2', 'finished': True, 'isBuiltIn': False, 'launchedWithDefaults': True, 'requestNonInteractive': True, 'requestShape': 'background', 'spawnDepth': 1}
   tool_uses=3 hook_success attachments=9 (distinct toolUseIDs 3) hookEvents=[('PreToolUse', True)]
   hook_success attachment keys: ['command', 'content', 'durationMs', 'exitCode', 'hookEvent', 'hookName', 'stderr', 'stdout', 'toolUseID', 'type']
   last2: user:tool_result | assistant[stop_reason=end_turn]:text

  subagent agent-a94a1bdbc93a47426: meta keys=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId']
   meta values: {'agentType': 'probe-sub', 'description': 'probe S1', 'finished': True, 'isBuiltIn': False, 'launchedWithDefaults': True, 'requestNonInteractive': True, 'requestShape': 'background', 'spawnDepth': 1}
   tool_uses=3 hook_success attachments=9 (distinct toolUseIDs 3) hookEvents=[('PreToolUse', True)]
   hook_success attachment keys: ['command', 'content', 'durationMs', 'exitCode', 'hookEvent', 'hookName', 'stderr', 'stdout', 'toolUseID', 'type']
   last2: user:tool_result | assistant[stop_reason=end_turn]:text

### session d764701f-54f0-47eb-8697-9b84abcc187a
entry types: ['agent-setting', 'api-request', 'api-request-blob', 'api-request-shape', 'assistant', 'atis-latch', 'attachment', 'cost-state', 'file-history-snapshot', 'last-prompt', 'mode', 'permission-mode', 'queue-operation', 'system', 'tools-used', 'user']
attachment types (counts): {'agent_listing_delta': 1, 'credential_org': 1, 'date': 1, 'environment': 1, 'hook_success': 2, 'model': 1, 'prompt_snapshot': 2, 'session_context': 1}
- Agent tool_use id=toolu_01VhBzu5RuzHDGU4P7MNqyn9 input keys=['description', 'prompt', 'subagent_type'] description='probe S3b' run_in_background=<absent>
  toolUseResult keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', 'outputFile', 'prompt', 'resolvedModel', 'status'] status='async_launched' isAsync=True agentId=a120561ff37cb1731
- Agent tool_use id=toolu_01VhBzu5RuzHDGU4P7MNqyn9 input keys=['description', 'prompt', 'subagent_type'] description='probe S3b' run_in_background=<absent>
- notification entry type=queue-operation origin=null tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=a120561ff37cb1731 has tool-use-id=True
- notification entry type=user origin={"kind": "task-notification", "producer": "session-task", "runId": "0mvxweass-2f4481b5"} tags=['duration_ms', 'note', 'result', 'status', 'subagent_tokens', 'summary', 'task-id', 'task-notification', 'tool-use-id', 'tool_uses', 'usage'] status=completed task-id=a120561ff37cb1731 has tool-use-id=True

  subagent agent-a120561ff37cb1731: meta keys=['agentType', 'description', 'finished', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId']
   meta values: {'agentType': 'probe-sub', 'description': 'probe S3b', 'finished': True, 'isBuiltIn': False, 'launchedWithDefaults': True, 'requestNonInteractive': True, 'requestShape': 'background', 'spawnDepth': 1}
   tool_uses=3 hook_success attachments=9 (distinct toolUseIDs 3) hookEvents=[('PreToolUse', True)]
   hook_success attachment keys: ['command', 'content', 'durationMs', 'exitCode', 'hookEvent', 'hookName', 'stderr', 'stdout', 'toolUseID', 'type']
   last2: user:tool_result | assistant[stop_reason=end_turn]:text

### session fb3a881a-d3f1-4a9a-8817-87d30485f1bd
entry types: ['agent-setting', 'api-request', 'api-request-blob', 'api-request-shape', 'assistant', 'atis-latch', 'attachment', 'cost-state', 'file-history-snapshot', 'last-prompt', 'mode', 'permission-mode', 'queue-operation', 'system', 'tools-used', 'user']
attachment types (counts): {'agent_listing_delta': 1, 'credential_org': 1, 'date': 1, 'environment': 1, 'hook_success': 2, 'model': 1, 'prompt_snapshot': 2, 'session_context': 1}
- Agent tool_use id=toolu_014iC3NrNvkPLrcU3mQMUuMp input keys=['description', 'prompt', 'subagent_type'] description='probe S4' run_in_background=<absent>
  toolUseResult keys=['agentId', 'canContinueAgent', 'canReadOutputFile', 'description', 'isAsync', 'outputFile', 'prompt', 'resolvedModel', 'status'] status='async_launched' isAsync=True agentId=ab929d875b9bf66dc
- Agent tool_use id=toolu_014iC3NrNvkPLrcU3mQMUuMp input keys=['description', 'prompt', 'subagent_type'] description='probe S4' run_in_background=<absent>
- notification entry type=queue-operation origin=null tags=['note', 'status', 'summary', 'task-id', 'task-notification'] status=stopped task-id=ab929d875b9bf66dc has tool-use-id=False
- notification entry type=user origin={"kind": "task-notification"} tags=['note', 'status', 'summary', 'task-id', 'task-notification'] status=stopped task-id=ab929d875b9bf66dc has tool-use-id=False

  subagent agent-ab929d875b9bf66dc: meta keys=['agentType', 'description', 'isBuiltIn', 'launchedWithDefaults', 'requestNonInteractive', 'requestShape', 'spawnDepth', 'toolUseId']
   meta values: {'agentType': 'probe-sub', 'description': 'probe S4', 'isBuiltIn': False, 'launchedWithDefaults': True, 'requestNonInteractive': True, 'requestShape': 'background', 'spawnDepth': 1}
   tool_uses=12 hook_success attachments=36 (distinct toolUseIDs 12) hookEvents=[('PreToolUse', True)]
   hook_success attachment keys: ['command', 'content', 'durationMs', 'exitCode', 'hookEvent', 'hookName', 'stderr', 'stdout', 'toolUseID', 'type']
   last2: assistant[stop_reason=tool_use]:Read | user:tool_result
````

## 8. Deviations from `RUN.md`

1. **Clean environment, private tmux socket** (§2). `RUN.md` assumes a human terminal. Here, launching from the coordinating session would have inherited its `CLAUDE*` variables.
2. **Three hook lines per Read, not two.** `RUN.md` expects about 2 per S4 Read (settings and subagent), but `fm-driver` also fires inside the subagent (F4). So the kill "at about 30" came after **10** chain files, not 15. That was still well before the chain's end at 90.
3. **Scripted watch and kill.** One shell loop did the watching and killing instead of a second terminal. It killed only the PID whose `comm` was `claude` and whose command line contained `--agent probe-driver`, after checking that. No other `claude` process was touched.
4. **Prompt-suggestion ghost text** was left unsubmitted (§4.1). `RUN.md` does not mention it.
5. **Attempt 1 aborted** at the Model-switch dialog (§3). Nothing was selected, and no `/config` setting was changed. The coordinator pinned the driver to `claude-sonnet-5-5` (Appendix B, line 5), and attempt 2 reran from a clean slate.
6. **Optional step 2b was run** (launch B2).

## 9. Headless probe (U6.2) — reserved

> **Reserved for U6.2** (`docs/plans/typed-decision-models6.md` §4, U6.2 row). U6.2 appends its `claude -p` results for H1–H5 here, with the same per-spawn columns as `read_results.py`, guard `hook_success` per tool use, and whether `SubagentHandback` is hooked. Nothing below this note and above the appendices belongs to U6.1.

## Appendices — the probe kit

These are the kit's own files, copied byte for byte from `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/` (the original session scratchpad), as they stood after attempt 2. Paths inside them are unchanged and point at that scratchpad. Appendix I says how to rerun from elsewhere. Run outputs (`hook-log.jsonl`, `ans/`, `panes/`, the results files) are not kit files; their evidence is quoted in §3–§7.

## Appendix A — `RUN.md`

````markdown
# dm spawn-mode probe — how to run it

Settles four harness questions behind typed-decision-models plan4 §3.3/§3.4 (ledger
`docs/plans/typed-decision-models-coordination.md`, 2026-11-01 "Architect reconciliation of Pass 8
F1/F2"; review P8-F1/P8-F2):

1. In an interactive `claude --agent probe-driver` session, are spawns foreground or background?
   (S1 no key · S2 `run_in_background: false` · S3 no key under `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`)
2. Does a subagent run end with an `end_turn` reply or a `SubagentHandback` call?
3. Does a `PreToolUse` hook see `SubagentHandback`?
4. What does a killed run (S4) leave behind? Is `finished` absent from its `.meta.json`?

**Cost and time.** About 15 minutes of your time. Spends 4 tiny subagent runs (S1–S3, one Read +
one Write each; 5 with the optional S3b) and one longer one (S4: up to 90 one-line Reads, killed
part-way), all on `claude-sonnet-5-5`, plus a few driver turns.

**What it touches.** Only this folder, plus what any Claude Code session writes anyway: its
transcripts under `~/.claude/projects/<this folder's encoded path>/` and, when you accept the
folder-trust prompt, the trust record Claude Code keeps for this folder. It does not touch the repo
and changes no `~/.claude` settings file. Your user-level hooks and `~/.claude/CLAUDE.md` still load,
as in any session; the probe's own log ignores them.

All commands below run from this folder:

```bash
P=/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe
cd "$P"
FLAGS="--strict-mcp-config --mcp-config ./empty-mcp.json --permission-mode acceptEdits"
```

`FLAGS` mirrors the `dm-driver` launch line in plan4 §3.5. Re-run the three lines in each new
terminal.

## What is in the folder

| Path | Role |
|---|---|
| `.claude/agents/probe-driver.md` | the driver (mirrors `dm-driver`: `tools: Agent(probe-sub)`, frontmatter `PreToolUse` hook, `initialPrompt`). One phase per launch. |
| `.claude/agents/probe-sub.md` | the subagent (mirrors `dm-labeler`: `Read, Write`, pinned model, `omitClaudeMd`, `acceptEdits`, frontmatter hook) |
| `.claude/settings.json` | project hooks: `PreToolUse` (matcher `*`), `SubagentStop`, `Stop` → `hook.py settings` |
| `hook.py` | logs one JSON line per call to `hook-log.jsonl`; on `PreToolUse` always exits 0 with `permissionDecision: allow` (plan4 §3.4's guard shape) |
| `req/s*.txt`, `long/f001…f090.txt` | request files; S4's chain of 90 files, each naming the next |
| `ans/` | where the subagent writes its answers |
| `read_results.py` | the summary table; run it after the probe |

Each hook line carries `src`, which says which wiring fired: `settings`, `fm-driver` (the
driver's frontmatter hook) or `fm-sub` (the subagent's frontmatter hook). That is how the run
shows premises the docs leave open (see "Unverified premises" below).

## Steps

Each launch is a fresh process: answer the driver once, wait for the reply named, then type
`/exit`. S3 **does** need its own launch, because the env var applies to the whole process.

### 0. Trust the folder (once)

```bash
claude
```

Accept the "trust this folder" prompt, then `/exit`. Until the folder is trusted, Claude Code skips
the frontmatter hooks (code.claude.com/docs/en/sub-agents), and the hook log would be empty.

### 1. Launch A — S1 and S2 (no env var)

```bash
claude --agent probe-driver $FLAGS "PHASE A"
```

- The driver spawns S1, waits for it, spawns S2, waits, then replies `PHASE A done`.
- If it shows `waiting for S1` and nothing more arrives within a minute after the subagent
  finished, type `continue`.
- Note what you see: a blocking spinner (foreground) or an immediate return plus a later
  notification (background). Then `/exit`.

### 2. Launch B — S3 (with `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`)

```bash
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 claude --agent probe-driver $FLAGS "PHASE B"
```

Wait for `PHASE B done`, then `/exit`. The variable is set for this one command only.

### 2b. Optional launch B2 — S3b (with `CLAUDE_CODE_FORK_SUBAGENT=0`)

Not in the ledger's spec, but recommended. The docs (code.claude.com/docs/en/sub-agents,
verified 2026-11-01) say that while fork mode is on (the default), Claude "can't wait for a
subagent" in interactive sessions, and that turning it off with `CLAUDE_CODE_FORK_SUBAGENT=0` is
the other way to get foreground spawns. If S3 comes back background, this is the next candidate.

```bash
CLAUDE_CODE_FORK_SUBAGENT=0 claude --agent probe-driver $FLAGS "PHASE B2"
```

Wait for `PHASE B2 done`, then `/exit`.

### 3. Launch C — S4, killed mid-run (no env var)

Terminal 1:

```bash
claude --agent probe-driver $FLAGS "PHASE C"
```

Terminal 2 (same `cd`), watch progress. Each S4 Read logs about two lines (settings + subagent
hook):

```bash
cd "$P"; while sleep 3; do echo "$(grep -c '/long/f' hook-log.jsonl 2>/dev/null) chain-read hook lines"; done
```

**When the count reaches about 30** (about 15 of the 90 files read, well before the chain ends), kill
the driver process hard from terminal 2. Ctrl+C the watch loop first, then:

```bash
pgrep -af 'agent probe-driver'      # one line: the PID and the claude command line
kill -KILL <that PID>
```

`SIGKILL` gives no chance to clean up, the worst case for a dead driver. Do **not** use `/exit`
or Ctrl+C in terminal 1; those are clean exits. Terminal 1 drops back to its shell. If S4 finished
first (count stopped rising and the driver said `PHASE C done`), rerun step 3. Clear `hook-log.jsonl`
first only if you also want a clean count.

Immediately after the kill, before any restart:

```bash
python3 read_results.py | tee results-after-kill.txt
```

### 4. Restart, then inspect again

```bash
claude -c --agent probe-driver $FLAGS "PHASE R"
```

- `-c` reopens the most recent conversation in this folder, the killed one. The driver should
  reply `restarted` and do nothing.
- If the CLI refuses `-c` together with `--agent`, use `claude -c $FLAGS "PHASE R"`.
- Wait about 30 seconds, in case the restart resumes or closes out S4 on its own. Note anything
  the UI says about S4, then `/exit`.

```bash
python3 read_results.py | tee results-after-restart.txt
diff results-after-kill.txt results-after-restart.txt
```

### 5. Hand back

Send `results-after-kill.txt`, `results-after-restart.txt` and `hook-log.jsonl`, plus what you saw
in steps 1, 2 and 4 (spinner versus immediate return; anything said about S4 on restart).

## Reading the table

One row per spawn, from this folder's transcripts only:

| Column | Source | Answers |
|---|---|---|
| `rib-key` | the driver's `Agent` input | S2 should show `False`, the others `-` (no key) |
| `toolUseResult.status` | the driver's `tool_result` entry | Q1: `async_launched` = background; anything else = foreground (see `result_keys` below the table) |
| `requestShape`, `launchedWithDefaults`, `finished` | `subagents/agent-<id>.meta.json` | Q1 / Q4: `absent` means the key is missing |
| `notification(kind/status)` | `<task-notification>` entries carrying the spawn's `tool-use-id` | the background completion shape; `none` for a foreground run, or a killed one |
| `terminal` | the subagent transcript's last assistant entry | Q2: `HANDBACK` or `END_TURN` (`last2` below the table lists the last two entries) |
| `hook-log HB` | `hook-log.jsonl`, matched on `agent_id` | Q3: which wiring saw `SubagentHandback` (`settings`, `fm-sub`, `fm-driver`) or `no` |

Below the table, each spawn's detail also says whether the **transcript** holds a `hook_success`
for the `SubagentHandback` tool use. That is the record the audit reads (plan4 §3.3). The closing
`hook-log counts` block shows which wiring fired in which agent.

## Unverified premises: what the run shows

- **Project-settings hooks inside subagents.** The docs say they fire, with `agent_id`/`agent_type`
  set (code.claude.com/docs/en/hooks). Confirmed if the counts show `('settings', 'PreToolUse',
  'probe-sub', 'Read')`.
- **The driver's own frontmatter hooks inside its subagents.** plan4 §3.4 assumes this; the docs
  don't say. Shown by any `fm-driver` row with `agent_type` `probe-sub`.
- **`PreToolUse` on `SubagentHandback`.** Not documented. Shown by the `hook-log HB` column, by
  `src`.
- **`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` and `Agent` spawns.** The env-var docs say it removes the
  subagent tool's `run_in_background` parameter and that "Claude waits for subagents to finish".
  Docs are not behaviour: the S3 row shows it.
- **A killed run's meta.** Not documented. Shown by S4's `finished` after the kill and after the
  restart. The `diff` shows whether a restart rewrites it.
- **`-c` combined with `--agent`.** Not documented; step 4 has the fallback.

## Rerunning

Old sessions stay in the project dir, and the table lists every spawn, each detail line naming its
session. For a clean slate, delete `hook-log.jsonl` and `ans/*`, and move this folder's project dir
(`python3 read_results.py` prints its path) aside.
````

## Appendix B — `.claude/agents/probe-driver.md`

Line 5, `model: claude-sonnet-5-5`, was added by the coordinator between attempts 1 and 2. Attempt 1 ran without it, on the session default (Opus 5.5).

````markdown
---
name: probe-driver
description: Driver session for the dm spawn-mode probe. Runs as the main session via claude --agent probe-driver and spawns probe-sub runs, one phase per launch.
tools: Agent(probe-sub)
model: claude-sonnet-5-5
hooks:
  PreToolUse:
    - matcher: "*"
      hooks:
        - type: command
          command: "python3 /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/hook.py fm-driver"
initialPrompt: "Carry out exactly the phase named in this message, as your instructions define it."
---

You are a probe driver. Your only tool is Agent, and you use it only as listed
below. The human names one phase per launch (PHASE A, B, B2, C or R). Do that
phase and nothing else: no other tool calls, no extra spawns, no retries, no
commentary beyond the one-line replies named here.

## How to make a spawn

Each spawn is one Agent call whose input has exactly these keys:
`subagent_type`, `description`, `prompt` — plus `run_in_background` only where
a spawn below says so. Never add `model`, `effort`, `isolation`, `name` or any
other key. `subagent_type` is always `probe-sub`. Copy `description` and
`prompt` character for character from the spawn's entry.

## Waiting

After a spawn, if the Agent result says the run was launched in the background
(asynchronously) rather than returning its output, end your turn with exactly
`waiting for <spawn id>` and make no other call. When a notification that the
run completed arrives, continue with the next step. If the result already holds
the run's output, continue at once.

## Phases

**PHASE A** — spawn S1, wait, spawn S2, wait, then reply `PHASE A done`.
- S1: no `run_in_background` key.
  - description: `probe S1`
  - prompt (everything between the fences, verbatim):

    ```text
    Read /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/req/s1.txt with offset 1, limit 1. Its `pages` field lists [offset, limit] pairs: read each remaining page with exactly that offset and limit. Follow the instructions on line 2, write the answer where the header says, then reply `done`.
    ```

- S2: `run_in_background: false` (the boolean false).
  - description: `probe S2`
  - prompt (everything between the fences, verbatim):

    ```text
    Read /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/req/s2.txt with offset 1, limit 1. Its `pages` field lists [offset, limit] pairs: read each remaining page with exactly that offset and limit. Follow the instructions on line 2, write the answer where the header says, then reply `done`.
    ```


**PHASE B** — spawn S3, wait, then reply `PHASE B done`.
- S3: no `run_in_background` key.
  - description: `probe S3`
  - prompt (everything between the fences, verbatim):

    ```text
    Read /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/req/s3.txt with offset 1, limit 1. Its `pages` field lists [offset, limit] pairs: read each remaining page with exactly that offset and limit. Follow the instructions on line 2, write the answer where the header says, then reply `done`.
    ```


**PHASE B2** (optional phase) — spawn S3b, wait, then reply `PHASE B2 done`.
- S3b: no `run_in_background` key.
  - description: `probe S3b`
  - prompt (everything between the fences, verbatim):

    ```text
    Read /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/req/s3b.txt with offset 1, limit 1. Its `pages` field lists [offset, limit] pairs: read each remaining page with exactly that offset and limit. Follow the instructions on line 2, write the answer where the header says, then reply `done`.
    ```


**PHASE C** — spawn S4, then wait. The human will stop this session mid-run;
that is expected. If S4 completes anyway, reply `PHASE C done`.
- S4: no `run_in_background` key.
  - description: `probe S4`
  - prompt (everything between the fences, verbatim):

    ```text
    Read /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/req/s4.txt with offset 1, limit 1. Its `pages` field lists [offset, limit] pairs: read each remaining page with exactly that offset and limit. Follow the instructions on line 2, write the answer where the header says, then reply `done`.
    ```


**PHASE R** — make no tool call at all. Reply exactly `restarted`.

Any other message: make no tool call and reply `no such phase`.
````

## Appendix C — `.claude/agents/probe-sub.md`

````markdown
---
name: probe-sub
description: Probe subagent for the dm spawn-mode probe. Reads one request file and writes one answer file. Spawned only by probe-driver.
tools: Read, Write
model: claude-sonnet-5-5
effort: high
maxTurns: 200
omitClaudeMd: true
permissionMode: acceptEdits
hooks:
  PreToolUse:
    - matcher: "*"
      hooks:
        - type: command
          command: "python3 /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/hook.py fm-sub"
---

You carry out one request file. Line 1 of the request file is a JSON header;
its `answer` field is the absolute path to write your answer to. Line 2 holds
your instructions. Follow them exactly, using only Read and Write, on only the
paths they name. Then reply `done`.
````

## Appendix D — `.claude/settings.json`

````json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/hook.py settings"
          }
        ]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/hook.py settings"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/hook.py settings"
          }
        ]
      }
    ]
  }
}
````

## Appendix E — `hook.py`

````python
#!/usr/bin/env python3
"""Probe hook: log every call, always allow.

Usage: python3 hook.py <source-tag>
  source-tag says which wiring fired it: settings | fm-driver | fm-sub.

Mirrors plan4 §3.4's guard shape: always exit 0; for PreToolUse print
{"hookSpecificOutput": {"hookEventName": "PreToolUse",
 "permissionDecision": "allow", "permissionDecisionReason": ...}}.
Hook input fields per code.claude.com/docs/en/hooks (verified 2026-11-01):
session_id, transcript_path, cwd, permission_mode, hook_event_name,
agent_id/agent_type (inside a subagent, or under --agent), tool_name,
tool_input, tool_use_id.
"""
import json
import os
import sys
import time

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hook-log.jsonl")


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    raw = sys.stdin.read()
    try:
        d = json.loads(raw)
    except Exception:
        d = {"_unparsed": True}
    ti = d.get("tool_input") if isinstance(d.get("tool_input"), dict) else {}
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time()*1000)%1000:03d}Z",
        "src": src,
        "event": d.get("hook_event_name"),
        "tool_name": d.get("tool_name"),
        "tool_use_id": d.get("tool_use_id"),
        "agent_id": d.get("agent_id"),
        "agent_type": d.get("agent_type"),
        "session_id": d.get("session_id"),
        "permission_mode": d.get("permission_mode"),
        "input_keys": sorted(d.keys()),
    }
    if "file_path" in ti:
        rec["file_path"] = ti["file_path"]
    if d.get("tool_name") in ("Agent", "Task"):
        rec["agent_input_keys"] = sorted(ti.keys())
        rec["subagent_type"] = ti.get("subagent_type")
        if "run_in_background" in ti:
            rec["run_in_background"] = ti["run_in_background"]
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if d.get("hook_event_name") == "PreToolUse":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": f"probe allow-all ({src})",
        }}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never block the probe; the gap shows in the log
        try:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"src": "hook-error", "error": repr(e)}) + "\n")
        except Exception:
            pass
    sys.exit(0)
````

## Appendix F — `read_results.py`

````python
#!/usr/bin/env python3
"""Summarise the dm spawn-mode probe. Stdlib only; reads nothing but this
probe folder's own project dir under ~/.claude/projects/ and ./hook-log.jsonl.

Usage: python3 read_results.py [--project-dir DIR]
"""
import argparse
import glob
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPAWN_RE = re.compile(r"/req/(s\d+b?)\.txt")
TAG_RE = re.compile(r"<([a-z_-]+)>")


def project_dir(arg):
    if arg:
        return arg
    root = os.path.expanduser("~/.claude/projects")
    enc = re.sub(r"[^A-Za-z0-9]", "-", HERE)
    d = os.path.join(root, enc)
    if os.path.isdir(d):
        return d
    # Do not read anything else: only list names that look like this folder.
    cands = [n for n in os.listdir(root) if n.endswith("scratchpad-dm-probe")]
    raise SystemExit(f"project dir not found: {d}\ncandidates (pass one with --project-dir): {cands}")


def load(path):
    out, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append((json.loads(line), line))
        except ValueError:
            bad += 1
    return out, bad


def blocks(d):
    m = d.get("message")
    c = m.get("content") if isinstance(m, dict) else None
    return c if isinstance(c, list) else []


def describe(d):
    m = d.get("message") if isinstance(d.get("message"), dict) else {}
    if d.get("type") == "assistant":
        tools = [b.get("name") for b in blocks(d) if b.get("type") == "tool_use"]
        kinds = sorted({b.get("type") for b in blocks(d)})
        s = f"asst[{m.get('stop_reason')}]"
        if tools:
            s += ":" + "+".join(tools)
        elif kinds:
            s += ":" + "+".join(kinds)
        if m.get("model") == "<synthetic>":
            s += "(synthetic)"
        return s
    if d.get("type") == "user":
        kinds = sorted({b.get("type") for b in blocks(d)}) or ["text"]
        return "user:" + "+".join(kinds)
    return d.get("type", "?")


def terminal(conv):
    for d in reversed(conv):
        if d.get("type") != "assistant":
            continue
        tools = [b.get("name") for b in blocks(d) if b.get("type") == "tool_use"]
        if "SubagentHandback" in tools:
            return "HANDBACK"
        if (d.get("message") or {}).get("stop_reason") == "end_turn":
            return "END_TURN"
        return "NONE(" + describe(d) + ")"
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir")
    a = ap.parse_args()
    pd = project_dir(a.project_dir)
    print(f"project dir: {pd}")

    hooklog = []
    hp = os.path.join(HERE, "hook-log.jsonl")
    if os.path.exists(hp):
        hooklog = [r for r, _ in load(hp)[0]]

    metas = {}
    for mp in glob.glob(os.path.join(pd, "*", "subagents", "agent-*.meta.json")):
        try:
            m = json.load(open(mp, encoding="utf-8"))
        except ValueError:
            continue
        metas[m.get("toolUseId")] = (mp, m)

    spawns = []  # dicts
    for sp in sorted(glob.glob(os.path.join(pd, "*.jsonl")), key=os.path.getmtime):
        entries, bad = load(sp)
        sid = os.path.basename(sp)[:-6]
        uses = {}
        for d, _ in entries:
            for b in blocks(d):
                if b.get("type") == "tool_use" and b.get("name") in ("Agent", "Task"):
                    inp = b.get("input") or {}
                    mt = SPAWN_RE.search(str(inp.get("prompt", "")))
                    uses[b["id"]] = {
                        "tag": mt.group(1).upper() if mt else "?",
                        "session": sid, "bad_lines": bad, "id": b["id"],
                        "keys": sorted(inp.keys()),
                        "rib": inp.get("run_in_background", "-"),
                        "result": None, "notif": [],
                    }
        for d, raw in entries:
            for b in blocks(d):
                if b.get("type") == "tool_result" and b.get("tool_use_id") in uses:
                    uses[b["tool_use_id"]]["result"] = d.get("toolUseResult")
            for tid, u in uses.items():
                if f"<tool-use-id>{tid}</tool-use-id>" in raw and "task-notification" in raw:
                    kind = d.get("type")
                    if kind == "queue-operation":
                        kind += ":" + str(d.get("operation"))
                    elif kind == "attachment":
                        kind += ":" + str((d.get("attachment") or {}).get("type"))
                    st = re.search(r"<status>([^<]*)</status>", raw.replace("\\n", "\n"))
                    tags = sorted(set(TAG_RE.findall(raw.replace("\\n", "\n"))))
                    u["notif"].append((kind, st.group(1) if st else None, tags))
        spawns.extend(uses.values())

    rows = []
    notes = []
    for u in spawns:
        r = u["result"]
        status = (r.get("status") if isinstance(r, dict) else ("str" if r else "NO RESULT"))
        mp, m = metas.get(u["id"], (None, None))
        g = (lambda k: ("absent" if k not in m else str(m[k]))) if m else (lambda k: "no meta")
        conv, last2, term, hb_hook_tx, aid = [], "no transcript", "-", "-", None
        if mp:
            aid = os.path.basename(mp)[len("agent-"):-len(".meta.json")]
            tp = mp[: -len(".meta.json")] + ".jsonl"
            if os.path.exists(tp):
                ents, bad = load(tp)
                conv = [d for d, _ in ents if d.get("type") in ("user", "assistant")]
                last2 = " | ".join(describe(d) for d in conv[-2:]) + (f" (+{bad} unparseable)" if bad else "")
                term = terminal(conv)
                hb_ids = {b["id"] for d in conv for b in blocks(d)
                          if b.get("type") == "tool_use" and b.get("name") == "SubagentHandback"}
                hs = [d for d, _ in ents if d.get("type") == "attachment"
                      and (d.get("attachment") or {}).get("toolUseID") in hb_ids
                      and str((d.get("attachment") or {}).get("type", "")).startswith("hook")]
                hb_hook_tx = ("n/a" if not hb_ids else
                              ",".join(sorted({str(h["attachment"].get("type")) for h in hs})) or "none")
        hb_log = sorted({h.get("src", "?") for h in hooklog
                         if h.get("tool_name") == "SubagentHandback" and aid
                         and h.get("agent_id") and (aid in h["agent_id"] or h["agent_id"] in aid)})
        notif = ";".join(f"{k}/{s}" for k, s, _ in u["notif"]) or "none"
        rows.append([u["tag"], str(u["rib"]), status, g("requestShape"), g("launchedWithDefaults"),
                     g("finished"), notif[:38], term, ",".join(hb_log) or "no"])
        notes.append(f"{u['tag']}: session={u['session']} tool_use={u['id']} agent={aid} "
                     f"input_keys={u['keys']}\n    result_keys="
                     f"{sorted(r.keys()) if isinstance(r, dict) else r!r:.80}\n"
                     f"    meta={'absent' if not m else sorted(m.keys())}"
                     f"{'' if not mp else ' mtime=' + time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(mp)))}\n"
                     f"    last2: {last2}\n    hook entries in transcript for SubagentHandback: {hb_hook_tx}\n"
                     f"    notification tags: {[t for _, _, t in u['notif']][:1]}"
                     + (f"\n    driver transcript unparseable lines: {u['bad_lines']}" if u['bad_lines'] else ""))

    hdr = ["spawn", "rib-key", "toolUseResult.status", "requestShape", "launchedWithDefaults",
           "finished", "notification(kind/status)", "terminal", "hook-log HB"]
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)] if rows else [len(h) for h in hdr]
    fmt = "  ".join("{:<%d}" % n for n in w)
    print(fmt.format(*hdr))
    print(fmt.format(*["-" * n for n in w]))
    for row in rows:
        print(fmt.format(*row))
    print()
    for n in notes:
        print(n)
    print()
    # Hook visibility summary: which wiring fired, in which agent, for which tool.
    from collections import Counter
    c = Counter((h.get("src"), h.get("event"), h.get("agent_type"), h.get("tool_name")) for h in hooklog)
    print("hook-log counts (src, event, agent_type, tool_name):")
    for k, v in sorted(c.items(), key=lambda kv: [str(x) for x in kv[0]]):
        print(f"  {v:4d}  {k}")
    if not hooklog:
        print("  (hook-log.jsonl empty or missing: hooks did not run — folder untrusted?)")


if __name__ == "__main__":
    main()
````

## Appendix G — `empty-mcp.json` and `req/*.txt`

Each file is shown verbatim. Each request file is two lines: a JSON header on line 1, the instruction on line 2, and a trailing newline.

`empty-mcp.json`:

````text
{"mcpServers":{}}
````

`req/s1.txt`:

````text
{"pages": [], "answer": "/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/ans/s1.txt"}
Write the single word ok to the answer path. Do nothing else.
````

`req/s2.txt`:

````text
{"pages": [], "answer": "/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/ans/s2.txt"}
Write the single word ok to the answer path. Do nothing else.
````

`req/s3.txt`:

````text
{"pages": [], "answer": "/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/ans/s3.txt"}
Write the single word ok to the answer path. Do nothing else.
````

`req/s3b.txt`:

````text
{"pages": [], "answer": "/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/ans/s3b.txt"}
Write the single word ok to the answer path. Do nothing else.
````

`req/s4.txt`:

````text
{"pages": [], "answer": "/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/ans/s4.txt"}
Read /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/long/f001.txt. Each file names the next file to read. Read them one at a time, in order, one Read call per file, until a file says END. Then write the number of files you read to the answer path.
````

## Appendix H — `long/f001.txt` … `long/f090.txt`

The 90 chain files follow one exact pattern. It was checked byte for byte against all 90 files on 2026-11-01. For N = 1…89, `long/fNNN.txt` (N zero-padded to 3 digits) is the two lines below, each ending in a newline:

````text
File N of a chain.
Next: /tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe/long/f(N+1, zero-padded).txt
````

`long/f090.txt` is:

````text
File 90 of a chain.
END
````

For example, `f001.txt` is `File 1 of a chain.` followed by `Next: …/dm-probe/long/f002.txt`.

## Appendix I — Rerunning the kit

- **Location.** Every kit file hard-codes `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6eed315-dc2c-4b2c-99e2-eece159b76c4/scratchpad/dm-probe`: the hook commands in Appendices B, C and D, the prompts in B, the answer paths in G and the `Next:` lines in H. To rerun elsewhere, recreate the tree (`.claude/agents/`, `.claude/settings.json`, `hook.py`, `read_results.py`, `empty-mcp.json`, `req/`, `long/`, an empty `ans/`) and replace that path in every file. `read_results.py` and `hook.py` locate their files relative to themselves.
- **Trust.** The new folder must be trusted first (`RUN.md` step 0). Otherwise frontmatter hooks are skipped.
- **Environment.** Launch from a clean environment (§2) if the launching shell is itself inside a Claude Code session.
- **Kill threshold.** Count **3** hook lines per chain Read (§8, item 2) when choosing the kill threshold.
- **Clean slate.** Follow `RUN.md` "Rerunning". `read_results.py` lists every spawn in the project dir, so a dirty dir mixes runs.

