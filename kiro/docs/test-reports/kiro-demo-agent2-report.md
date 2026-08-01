# `kiro-demo-agent2` — Test Report (AC-2 re-verification)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-041 (—) · **Extends:** `kiro/docs/test-reports/kiro-demo-agent-report.md`

## Summary

Re-verification of **AC-2** (previously **FAIL**, Defect D-1) after the fix landed:
`tdd-engineer` wired `responder`/`embed_worker`/`trigger` through `mcp.configure()` so MCP
`send_message` now schedules the same background trigger/responder work the REST route always
did (`falkor-chat` commit `17c2fa0`, K-041). Executed per `kiro/docs/test-plans/kiro-demo-agent2.md`.
The fix's implementation was not re-reviewed here (already independently code-reviewed per the
coordinator, write-up at `falkor-chat/docs/reviews/mcp-background-scheduling-impl.md`) — only the
live behavior it restores.

falkor-chat was **explicitly stopped and restarted** before this pass (not left to `--reload` pick
up the fix silently) — confirmed running on commit `7977a8f` (ancestry includes fix commit
`17c2fa0`). LM Studio backend confirmed healthy (`GET :1234/v1/models` → 200) throughout.

**Verdict: AC-2 now PASSES.** A message posted via the Kiro agent's `send_message` MCP call, with
`mentions:["assistant"]`, triggered a real `WorkflowRun` (`triage@v1`) within seconds, which
produced a visible `assistant` reply directly addressing the posted text
("Hello from the kiro demo, take two! 👋"). The Kiro session's own `read_messages` call then
surfaced that reply back to the user, closing the loop AC-2 requires. AC-1 (re-run as AC-2's
precondition) and the AC-3/AC-4 spot-checks all remain PASS, unchanged from the original pass.

**All four acceptance criteria (AC-1 … AC-4) now PASS.** No new defects found.

## Results table

| ID | Ref | Result | Evidence |
|---|---|---|---|
| TP2-001 | AC-1 (precondition re-run) | **PASS** | `kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "post 'hello from the kiro demo, take two' and mention assistant"` → exit 0. Trace: `send_message` called with `body="hello from the kiro demo, take two"`, `re="demo-welcome"`, `mentions=["assistant"]`. `GET /threads/demo-welcome/messages?ws=acme` count 36→37, authored `u1`/"Demo User". |
| TP2-002 | AC-2 | **PASS** | Polling after TP2-001: within ~1 poll cycle (≤3s), `GET /threads/demo-welcome/workflow-runs?limit=50` showed a **new** `WorkflowRun` (`runId 2bb5c34c38b3491486de9c75cff7761d`, `defKey:"triage"`, `status:"running"`, `startedAt:1785624277916` — after TP2-001's post, absent from the pre-post baseline of 7 runs). Polled to terminal status (~72s): run reached `status:"waiting"` (parked, `waitsForHuman`, expected `triage@v1` shape — same as the K-039 report's documented behavior). `demo-welcome` message count grew 37→41; among the 4 new `assistant`-authored messages: `"Hello from the kiro demo, take two! 👋"` — a direct reply to TP2-001's exact posted text. Then `kiro-cli chat --agent falkor-chat-demo --no-interactive "read the latest messages"` → `read_messages(re="demo-welcome")` called, and the CLI's own rendered output includes `Assistant — "Hello from the kiro demo, take two! 👋"` among the returned messages, closing AC-2's "shown in the Kiro session" requirement. |
| TP2-003 | AC-3 (spot-check) | **PASS** | `cd kiro && kiro-cli agent list` → `falkor-chat-demo` listed, `Workspace` scope, unchanged from original pass. |
| TP2-004 | AC-4 (spot-check) | **PASS** | Interactive `tmux` session → `/tools` → `2 tools`: `read_messages` and `send_message`, both `mcp:falkor-chat`, both `● allowed`. Unchanged from original pass. |

**`demo-welcome` message/run counts:** 36→41 messages, 7→8 `WorkflowRun`s (`limit=50` query) over
this pass — consistent with the coordinator's note that the workspace has grown since the original
pass; all comparisons here are before/after deltas on this pass's own baseline, not the original
pass's absolute counts (per the coordinator's guidance and this component's own test-plan
convention).

## Defects

None found in this re-verification. **Defect D-1 from the original report is confirmed fixed** —
MCP-posted `@mention` messages now reliably trigger a response, observed on the first live attempt
(no need for the "3 independent attempts" rigor of a from-scratch reliability pass, since this is a
targeted regression-closure check on a specific, already-diagnosed code path, not a new-feature
statistical characterization).

## Coverage & gaps

**Covered:** AC-2 end-to-end (MCP post → workflow trigger → visible reply → read back through the
Kiro session), AC-1 precondition re-run, AC-3/AC-4 spot-checks.

**Not re-covered here (deliberately, per the re-verification's narrower scope — still valid from
the original pass, untouched by this fix):** TP-002 (interactive `@`-completion), TP-004 (full
fresh-clone flow), TP-005 (static config content), TP-007 (no-mention edge case), TP-008
(cold-start failure mode). The original report's results for these stand.

**Residual observation, not a defect:** the `read_messages` call in TP2-002 returned 8 messages
(all `assistant`/`u1` posts accumulated across this pass and prior sessions today), not just the
single reply to TP2-001's post — expected given the per-actor cursor hadn't been advanced by a
`read_messages` call since earlier in this same day's testing. Doesn't affect the AC-2 verdict
(the target reply was present and legible in the output either way).

## Feedback & recommendations

- Recommend closing K-041 as verified once this report lands, per the coordinator's dispatch.
- No new testability or process issues surfaced by this pass — the original test plan's
  delta/msgId-comparison convention (rather than absolute counts) held up cleanly against a
  workspace that had grown between passes, exactly as designed.

## Final state

falkor-chat is **left running** (`http://localhost:8000/` → `HTTP:200` at the end of this pass),
on the fixed code (commit `7977a8f`, ancestry includes `17c2fa0`). FalkorDB (Docker) untouched.
No falkor-chat file was modified by this pass.
