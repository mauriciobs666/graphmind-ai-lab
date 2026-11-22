# Post-edit hook: settings block and go-live (S8)

> **Status:** active · **Owner:** `cobb` · **Tracks:** coordination-graph S8 (plan `claude/docs/plans/coordination-graph.md` §3.2, §4 S8, Q5)

The stakeholder applies this by hand (plan Q5). Nothing here is applied, run or registered by an
agent; `.claude/settings*.json` is protected and is never edited through Bash or any other route.
Handler behaviour: `coordination-graph/README.md` "The post-edit hook".

## 1. The block

Both events, one handler, as plan §3.2 specifies. Paste into **`.claude/settings.json`** (project,
committed; plan §3.2 says so) or, to keep it private and gitignored, `.claude/settings.local.json`.
Both files currently have no `hooks` key (checked 2026-11-21).

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PROJECT_DIR}\"/coordination-graph/hooks/post-tool-sync.sh",
            "timeout": 60
          }
        ]
      }
    ],
    "PostToolUseFailure": [
      {
        "matcher": "Write|Edit|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PROJECT_DIR}\"/coordination-graph/hooks/post-tool-sync.sh",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

Choices:

- **Matcher `Write|Edit|Bash`** (plan). `Write|Edit` is a plain exact-name list; `MultiEdit` is not in
  the current tools reference, so it is omitted. `Bash` covers a script or `sed` that edits a
  mirrored file. `NotebookEdit` is not listed: no mirrored file is a notebook.
- **`PostToolUseFailure` too** (plan M1): a Bash that writes a ledger then exits non-zero may be
  classified under either event; the script is idempotent, so a double firing costs a no-op.
- **Shell form** (no `args`), with the placeholder in double quotes: the form the hooks doc's own
  example uses and what the S0 probe used. Needs the script's executable bit (it has it).
- **`timeout: 60`** seconds (the unit is seconds). The script bounds its own foreground sync at
  `timeout 50`, leaving 10 s of headroom so the shell can queue the retry and write its warning
  before Claude Code would kill it.
- Merge by hand if a `hooks` key exists later; identical handlers across settings files run once, but
  do not register the block in both files anyway.

## 2. Apply steps (stakeholder)

1. The S0 probe has run (section 6), so the contexts where the hook fires are known. Do not register
   a probe block alongside this one: remove any leftover probe `hooks` entries first.
2. Confirm `coordination-graph/hooks/post-tool-sync.sh` is executable (`ls -l`).
3. **Before registering**, create the kill switch: `touch coordination-graph/.state/disabled`
   (`mkdir -p coordination-graph/.state` first). With it present the script exits 0 before it looks
   at the stamp, so the first fire cannot start an unrecorded first-run hash-compare against the
   shared instance. Section 5 removes it at the end.
4. Add the block above to `.claude/settings.json` (or `.local`), valid JSON.
5. Run `/hooks` in the session: both `PostToolUse` and `PostToolUseFailure` should list the handler
   with its source file. Per the docs, settings edits are normally picked up by the file watcher; if
   `/hooks` does not show it, restart the session.
6. Go straight to section 5 (go-live); the hook stays inert until its last step.

## 3. Verify it fired

The fast path writes nothing, so a quiet hook is indistinguishable from a silent one. Use a
disposable check, not a real source edit:

1. `/hooks` lists both events (step 4 above).
2. Run `claude --debug` (or `/debug`) and look for the hook matching `Write|Edit|Bash` on a tool call.
3. Positive proof: after go-live, make a trivial edit to a mirrored file you own (a scratch
   `docs/reviews/*.md`, not a live ledger) and check that `coordination-graph/.state/stamp` mtime
   moved and `./coordgraph.sh drift --confirm coordination` shows no drift for it.
4. A fault is visible, not silent: exit 2 puts stderr ("not synced ... say so") in front of the agent.
5. Per context (main session, foreground, background and worktree subagents) the S0 probe saw the
   hook fire under permission mode `auto`; other modes and teammates are unverified; see section 6.

## 4. Turn it off

- **Kill switch (no settings edit, instant):** `touch coordination-graph/.state/disabled`
  (or set `COORDGRAPH_HOOK_DISABLED=1` in the session environment). The script exits 0 at once and the
  sync header reads `Hook: disabled`. Re-enable with `rm coordination-graph/.state/disabled`. Edits
  made while disabled stay newer than `.state/stamp`, so the first hook call after re-enabling syncs
  them; run `./coordgraph.sh sync --all --confirm coordination` from `coordination-graph/` if you
  want that done at once.
- **Deregister (stakeholder):** delete both event entries from the settings file, confirm with
  `/hooks`. Leave `.state/` in place (gitignored) or delete it to force a first-run sync on re-register.

## 5. S8 go-live sequence (after registration; the stakeholder runs it)

Written down, not executed by any agent. From `coordination-graph/`, with FalkorDB and the venv up and
`.state/disabled` still in place from section 2 step 3:

```bash
touch .state/stamp                                   # 1. stamp at the START of the re-sync
./coordgraph.sh sync --all --confirm coordination    # 2. re-sync files edited since the S8a load
./coordgraph.sh drift --confirm coordination         # 3. must report no drift
rm .state/disabled                                   # 4. only now let the hook act
```

Why this order: the stamp is taken first, so a file edited while `sync --all` runs is newer than the
stamp and the hook picks it up once enabled (the script's own "stamp = call start" rule). The kill
switch stays on until the end because the hook, not `sync --all`, owns `failed.json`; with it on, the
hook cannot fire a first-run hash-compare or race the manual sync. Edits made in that window (including
the agent's own) are not mirrored by the hook but remain newer than the stamp, so step 4 picks them up.

**A failed source is not retried by the hook.** `sync --all` and `sync <file>` write no failure state
(`failed.json`/`failed.pending` belong to `hook-sync`/`retry-failed`; `coordgraph/sync.py` has none),
and the stamp is already touched, so a source that failed in step 2 is never re-entered by the hook until
someone edits it again. If step 2 reports a failed source or step 3 shows drift, re-run
`./coordgraph.sh sync --all --confirm coordination` (or `sync <file> --confirm coordination`) until
`drift` is clean, and only then do step 4.

Then record in `coordination-graph/docs/HISTORY.md` one dated entry: the settings block registered and
when, the `sync --all` counts and any source that needed a re-run, the stamp time, the final clean
`drift`, the kill switch removed, and the S0 result (section 6).

**Guard refusal on first run.** If `.state/stamp` is absent and the guard refuses the endpoint, the hook
sets a 15-minute `retry-due`; until it passes, edits are neither mirrored nor warned about
(`coordination-graph/README.md`, "The post-edit hook"). Initialising the stamp in step 1 avoids the
first-run path, but if the stamp is ever lost, expect this window and run `sync --all` by hand.

## 6. S0 probe result and what remains unverified

The S0 probe ran 2026-11-21 (Claude Code, permission mode `auto` in every logged firing; a logging
hook on `PostToolUse` and `PostToolUseFailure`, matcher `Write|Edit|Bash`). Recorded in
`skills/agent-standards/claude-code.md` (Hooks, `Verified:` stamp). Result:

- `PostToolUse` fires for Write, Edit and Bash in the main session and in foreground, background and
  worktree subagents (`agent_id` set in the subagents).
- A **failing Bash** fires `PostToolUseFailure` and no `PostToolUse`; hence both events stay registered.
- **A failed Edit ("String to replace not found") fires no hook event at all**: FR-5 must treat it as
  producing no firing (nothing was written, so nothing is lost).
- In a worktree subagent `cwd` is `<repo>/.claude/worktrees/agent-<id>` while `CLAUDE_PROJECT_DIR` is the
  original project root.
- Exit 2 stderr reaches the agent as a separate system message ("PostToolUse:Bash hook blocking
  error ..."), not inside the tool result, in main, foreground, background and worktree subagents.

**Still unverified:** other permission modes (`acceptEdits`, `default`), teammate/agent-team sessions,
MCP tools (not in the matcher), and the `PostToolUse`/`PostToolUseFailure` payload field docs
(`tool_input.file_path`; page truncated in fetch; the script reads `file_path` leniently and the fast
path does not depend on it).

A silent context is not a stop (plan Q2): it falls back to FR-5 flagging only.

## 7. Sources checked (2026-11-21)

`code.claude.com/docs/en/hooks` (first ~100k chars of 294k: schema, `timeout` in seconds, matcher
syntax, `CLAUDE_PROJECT_DIR` quoting, exit-2 table, subagent firing, `/hooks`, `--debug`);
`code.claude.com/docs/en/tools-reference` (tool names: Write, Edit, Bash listed; no MultiEdit).
