# S0 hook-firing probe — runbook (temporary)

Plan: `claude/docs/plans/coordination-graph.md` S0 (no S0 change in `coordination-graph3.md`).
Delete this file and `hook-probe.sh` when the result is recorded.

## 1. Settings edit (stakeholder applies; `.claude/settings.local.json` is the least invasive home)

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Write|Edit|Bash",
        "hooks": [ { "type": "command", "command": "\"${CLAUDE_PROJECT_DIR}\"/claude/cobb/scripts/hook-probe.sh", "timeout": 10 } ] }
    ],
    "PostToolUseFailure": [
      { "matcher": "Write|Edit|Bash",
        "hooks": [ { "type": "command", "command": "\"${CLAUDE_PROJECT_DIR}\"/claude/cobb/scripts/hook-probe.sh", "timeout": 10 } ] }
    ]
  }
}
```

Merge with any existing `hooks` key; restart the session (or `/hooks` to confirm both events list the
handler). Verify first: `rm -f /tmp/coordgraph-hook-probe.log`, run any Bash in the main session,
`tail -1 /tmp/coordgraph-hook-probe.log`.

## 2. Legs (each does: Write a file, Edit it, Bash `echo hi`; scratch dir `/tmp/s0-probe/`)

Log is `/tmp/coordgraph-hook-probe.log`, one JSON line per firing (`event`, `tool`, `agent_id`,
`cwd`, `env_CLAUDE_PROJECT_DIR`, `permission_mode`). Note the permission mode of each leg.

| Leg | How | Expect to record |
|---|---|---|
| L1 main session | do the three calls yourself | 3 lines, `agent_id` null |
| L2 foreground subagent | `Agent` general-purpose, "Write, Edit, Bash in /tmp/s0-probe/" | 3 lines with `agent_id` set |
| L3 background subagent | same, `run_in_background: true` | same; if absent -> silent context |
| L4 failing Bash | Bash: `echo x > /tmp/s0-probe/f.txt; exit 1` | which event fires: PostToolUse or PostToolUseFailure (M1). Also run it from a subagent |
| L5 worktree | subagent with `isolation: "worktree"` does Write+Bash | `cwd` inside the worktree, `env_CLAUDE_PROJECT_DIR` = main checkout (m4) |
| L6 exit-2 stderr | Bash: `echo PROBE_EXIT2` in main session, then from a subagent | agent sees `HOOK-PROBE-EXIT2-SENTINEL` text? (main / subagent separately) |
| L7 failing Write/Edit | Edit with an `old_string` that does not exist | PostToolUseFailure line expected |

## 3. Record and clean up

1. Add a dated, `Verified:`-stamped result to the `## Hooks` section of `skills/agent-standards/claude-code.md`
   (per-leg fired/not, failure event for L4, CLAUDE_PROJECT_DIR in worktree, permission mode).
2. Any leg that did not fire -> note "FR-5 flagging only" for S9/S11 (plan Q2); it does not stop the work.
3. Stakeholder removes the two `hooks` entries; then delete `hook-probe.sh`, this file and `/tmp/coordgraph-hook-probe.log`.
