# `kiro-demo-agent2` — Test Plan (AC-2 re-verification)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-041 (—) · **Extends:** `kiro/docs/test-plans/kiro-demo-agent.md`

## 1. Scope & objective

Re-verify **AC-2** only, following the fix for Defect D-1 (`kiro/docs/test-reports/
kiro-demo-agent-report.md`): `tdd-engineer` wired `responder`/`embed_worker`/`trigger` through
`mcp.configure()` so MCP `send_message` now schedules the same background trigger/responder work
the REST route always did (`falkor-chat` commit `17c2fa0`, K-041, write-up at
`falkor-chat/docs/reviews/mcp-background-scheduling-impl.md`). The fix itself is not re-reviewed
here — only the live behavior it's supposed to restore.

AC-1 is re-run first since it's AC-2's precondition (post the mention-bearing message that
`assistant` is then expected to reply to). AC-3/AC-4 are spot-checked only (never touched by this
fix) — not a full re-run.

References: original plan `kiro/docs/test-plans/kiro-demo-agent.md` (full risk assessment/recipe,
still authoritative for anything not superseded here); `kiro/docs/plans/kiro-demo-agent.md` §5.

## 2. Test items

| ID | Title | Priority |
|---|---|---|
| TP2-001 | AC-1 precondition re-run: mention-bearing post lands (same recipe as original TP-001) | High |
| TP2-002 | AC-2: `assistant` replies to the MCP-posted mention, and the reply is readable back via `read_messages` | High |
| TP2-003 | Spot-check AC-3: `kiro-cli agent list` still clean (no full clone re-run) | Low |
| TP2-004 | Spot-check AC-4: `/tools` still shows exactly the two tools | Low |

### TP2-001 — AC-1 precondition

- **Preconditions:** falkor-chat restarted fresh (not relying on `--reload` picking up the fix
  silently) on the fixed code (commit `17c2fa0`+); reachable at `http://localhost:8000/`. LM Studio
  reachable. Record baseline message/run counts in `demo-welcome` (delta comparison, not absolute
  counts — the workspace has grown since the original pass per the coordinator's note).
- **Steps:** `cd kiro && kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "post 'hello from the kiro demo, take two' and mention assistant"`.
- **Expected:** Same as original TP-001 — `send_message` called with `mentions:["assistant"]`,
  exit 0, REST message count grows by exactly one, authored by `u1`/"Demo User".

### TP2-002 — AC-2

- **Preconditions:** TP2-001 done.
- **Steps:** Poll `GET /threads/demo-welcome/messages?ws=acme` and `GET
  /threads/demo-welcome/workflow-runs?limit=50` for up to ~60s for a new `assistant`-authored
  message or a new `WorkflowRun`. Once observed (or timeout), run `kiro-cli chat --agent
  falkor-chat-demo --no-interactive "read the latest messages"`.
- **Expected:** A new `assistant` reply appears (message and/or workflow run) within a reasonable
  wait, and the Kiro session's `read_messages` output surfaces that reply text back to the user.
  **If no reply appears within the timeout, this is a new, more urgent finding — report plainly,
  do not assume the fix worked.**

### TP2-003 / TP2-004 — AC-3/AC-4 spot-checks

- **Steps:** `cd kiro && kiro-cli agent list` (confirm `falkor-chat-demo` still listed, Workspace
  scope). Interactive `/tools` inside a session (confirm still exactly `send_message` +
  `read_messages`).
- **Expected:** Unchanged from the original pass — these were never touched by the K-041 fix.

## 3. Out of scope

- Re-reviewing the K-041 fix's implementation (`tdd-engineer`'s own test-first work, already
  independently code-reviewed per the coordinator).
- Re-running TP-002 (interactive `@`-completion), TP-004 (fresh clone), TP-005 (static config),
  TP-007 (no-mention post), TP-008 (cold start) — untouched by this fix, already PASS in the
  original report, not re-verified here.
