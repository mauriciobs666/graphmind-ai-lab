# `kiro-demo-agent` — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (—) · **Extended by:** `kiro/docs/test-reports/kiro-demo-agent2-report.md`

## Summary

Live acceptance pass for the minimal Kiro CLI demo agent (`kiro/.kiro/agents/falkor-chat-demo.json`,
commit `d33a8af`), executed per `kiro/docs/test-plans/kiro-demo-agent.md` against a real, locally
running `falkor-chat` (`./scripts/start_server.sh`, workspace `ws:acme`, `demo-general`/
`demo-welcome`, AI responder + M3 workflow trigger both enabled) and a real, authenticated
`kiro-cli` v2.14.1. LM Studio backend confirmed healthy throughout (`GET :1234/v1/models` → 200,
serving `prism-ml/bonsai-27b` and others).

**Verdict: 3 of 4 acceptance criteria PASS (AC-1, AC-3, AC-4). AC-2 FAILS** — not because of
anything in the Kiro-side config under test, but because of a real, reproducible defect in
`falkor-chat`'s own MCP transport: **messages posted via the MCP `send_message` tool never trigger
`assistant`'s reply**, because the responder/workflow `BackgroundTasks` wiring lives only in the
REST route handler (`api.py:post_message`), never in `Services.post_message()` or `mcp.py`. Every
prior QA pass that validated "`@mention` triggers a reply" (`falkor-chat/docs/test-reports/
mention-reply-delivery-report.md`) posted via the **REST** endpoint, not MCP — so this specific gap
in the MCP path had never actually been exercised until this pass. This is a **falkor-chat defect**,
outside this feature's own build scope ("Any change to falkor-chat's MCP server itself" is
explicitly out of scope for `kiro-demo-agent`), but it directly blocks this feature's AC-2 as
written and is the central finding of this report.

Everything Kiro-side worked cleanly: config containment (AC-4, both static and live), fresh-clone
availability (AC-3), message posting with and without mentions (AC-1, TP-007), and the one
previously-unverified interactive-TUI risk (`@`-file-completion, TP-002) did not reproduce.

Left running: **falkor-chat is left up** at the end of this pass (see §"Final state" below) — it was
stopped and cleanly restarted once, for TP-008 only.

## Results table

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-001 | AC-1 | **PASS** | `kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "post 'hello from the kiro demo' and mention assistant"` → exit 0. Trace: `send_message` called with `body="hello from the kiro demo"`, `re="demo-welcome"`, `mentions=["assistant"]`. `GET /threads/demo-welcome/messages?ws=acme` count 32→33; new message `msgId 2a1c16e6cc1e4bfeb25e8bd485b58551`, `authorId:"u1"`, `displayName:"Demo User"` (FR-6 — not a "Kiro" identity), text matches exactly. |
| TP-002 | AC-1 edge | **PASS** | Genuine interactive TUI driven via `tmux` (not `--no-interactive`). Typed literally `post 'testing literal at mention' to @assistant` and pressed Enter. Captured pane shows the input line rendered verbatim, no file-completion dropdown intercepted it; `send_message` called with `mentions=assistant`. REST confirms: count 34→35, `msgId c3b4ec0e6bcd476ab34980f957b4e909`, text exact match. Closes the risk the plan (§2.3/§7) flagged as unverified. |
| TP-003 | AC-2 | **FAIL** | `read_messages` tool mechanism itself works correctly — `kiro-cli chat ... --no-interactive "read the latest messages"` called `read_messages(re="demo-welcome")` and correctly surfaced prior historical `assistant` replies (author, text) in the Kiro session's own output text. But the precondition ("`@assistant` has replied") never materializes: waited 45s+ after TP-001's mention-bearing post with a healthy LLM backend, zero new message, zero new `WorkflowRun` (`GET /threads/demo-welcome/workflow-runs` most-recent `startedAt` unchanged at `1785543949993`, predating TP-001's `createdAt 1785619699056`). Root cause confirmed by reading `falkor-chat/server/falkorchat/mcp.py` (send_message calls `Services.post_message` directly, zero references to `BackgroundTasks`/`trigger`/`responder` in the whole 131-line file) vs. `falkor-chat/server/falkorchat/api.py:143-168` (the REST route schedules `background.add_task(_safe_run_workflow, ...)` / `_safe_respond`). See Defect D-1. |
| TP-004 | AC-3 | **PASS** | Real `git clone` (not the lighter adaptation) to a scratch path. `cd <clone>/kiro && kiro-cli agent list` → `falkor-chat-demo` listed, `Workspace` scope, zero extra setup. Mini round trip from the clone: `kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "post 'hello from the fresh clone' without mentioning assistant"` → exit 0, `send_message` called, REST count 35→36, `msgId a3d74f0740bb4bb8a7da55a4b496c6d5`. Confirmed no stray global config: `~/.kiro/agents/` holds only the pre-existing `agent_config.json.example`; no `~/.kiro/mcp.json` exists. |
| TP-005 | AC-4 (static) | **PASS** | Direct read of `kiro/.kiro/agents/falkor-chat-demo.json`: `tools == ["@falkor-chat/send_message", "@falkor-chat/read_messages"]`, `allowedTools` identical — no bare `@falkor-chat`, no `"*"`, no built-ins. |
| TP-006 | AC-4 (live) | **PASS** | Interactive session (`tmux`) → `/tools` → `2 tools`: `read_messages` (`mcp:falkor-chat`, `● allowed`) and `send_message` (`mcp:falkor-chat`, `● allowed`). Nothing else listed — no `create_thread`/`create_channel`/`list_channels`/`list_threads`/`search_messages`, no built-ins. |
| TP-007 | Edge | **PASS** | `kiro-cli chat ... --no-interactive "post 'plain post, no mention' without mentioning assistant"` → `send_message` called with `body` + `re` only, no `mentions` key at all. REST count 33→34, `msgId 2362e4eaa73149a288f036018688ff06`. Confirms the prompt handles the optional-mention case without breaking. (Also incidentally reconfirms D-1: this message likewise never got a reply, consistent with "no responder ever triggers via MCP" rather than "only mentioned messages fail to reply.") |
| TP-008 | Edge | **PASS** | falkor-chat's uvicorn stopped deliberately (own process, started by this pass — see §Environment notes). `kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "hello"` → `One or more mcp server did not load correctly ... Error: One or more MCP servers failed to start (--require-mcp-startup enabled)`, **exit code 3**. Fails loudly as documented, not silently. falkor-chat restarted cleanly immediately after (confirmed `HTTP:200` within ~10s). |

**Final `demo-welcome` state at end of pass:** 36 messages (started at 32; +4 from this pass: TP-001,
TP-007, TP-002, TP-004's clone message — all `authorId:"u1"`, zero new `assistant` messages, `GET
/threads/demo-welcome/workflow-runs` still 7 entries, unchanged from pre-pass).

## Defects

### D-1 — MCP `send_message` never triggers `assistant`'s reply (blocks AC-2)

- **Severity: High.** Blocks the core second half of this feature's demo promise (FR-5/AC-2 — "the
  person can then have the Kiro agent read back `@assistant`'s reply"). A live demo run following
  exactly `kiro/README.md`'s own instructions will post successfully (AC-1 works) and then find
  nothing to read back — the most visible, audience-facing failure mode possible for this feature.
- **Component:** `falkor-chat` (not the Kiro config under test — `kiro/.kiro/agents/
  falkor-chat-demo.json` and its system prompt behave exactly as designed; the defect is entirely
  server-side).
- **Steps to reproduce:**
  1. `cd falkor-chat && ./scripts/start_server.sh` (workspace `acme`, default `FALKORCHAT_ENABLE_AGENT=1`, `FALKORCHAT_WORKFLOW_ENABLED=1`).
  2. `cd kiro && kiro-cli chat --agent falkor-chat-demo --require-mcp-startup --no-interactive "post 'hello' and mention assistant"`.
  3. Wait ≥30s (LLM backend healthy, confirmed via `GET :1234/v1/models`).
  4. `curl -s "http://localhost:8000/threads/demo-welcome/messages?ws=acme"` and `curl -s
     "http://localhost:8000/threads/demo-welcome/workflow-runs"`.
- **Expected:** A new `assistant`-authored message appears in the thread (matching the behavior
  already verified via REST in `falkor-chat/docs/test-reports/mention-reply-delivery-report.md`),
  and/or a new `WorkflowRun` starts (workflow trigger is enabled by default in `start_server.sh`).
- **Actual:** Neither happens. Message count and `workflow-runs` list are unchanged from
  immediately before the call, indefinitely (confirmed at 45s in this pass; TP-007's later,
  unmentioned post 20+ minutes later still shows no reply to either message by the time of the
  final check).
- **Root cause (code-confirmed, not inferred):** `falkor-chat/server/falkorchat/api.py:143-168`'s
  REST `POST /threads/{thread_id}/messages` handler is the *only* place `background.add_task
  (_safe_run_workflow, ...)` / `background.add_task(_safe_respond, ...)` is scheduled — it takes a
  `BackgroundTasks` FastAPI dependency unavailable to a plain function call. `falkor-chat/server/
  falkorchat/mcp.py`'s `send_message` tool (lines 52-61) calls `_svc().post_message(ctx, ...)`
  directly and returns — the entire file has zero references to `BackgroundTasks`, `trigger`, or
  `responder`. `Services.post_message()` itself (the method both front doors call) does not
  schedule the trigger either — it's purely a REST-router-level concern, not a service-layer one.
  Since MCP is meant to be "the agent front door" (`falkor-chat/docs/DESIGN.md` §15's own framing),
  and every prior QA validation of the trigger mechanism went through REST (confirmed by re-reading
  `mention-reply-delivery-report.md`'s own evidence — every posted test message used `POST
  /channels/.../threads` or `POST /threads/{id}/messages`, never MCP), this gap in the MCP path
  appears to have simply never been exercised by a real MCP client before this pass.
- **Recommended fix (for whoever picks this up — not implemented by this QA pass):** wire the same
  trigger/responder scheduling into `mcp.py`'s `send_message` (or better, move the scheduling into
  `Services.post_message()` itself, if an async task-scheduling seam can be threaded through both
  front doors, so REST and MCP share one code path instead of two copies of the same policy).
  Flagging the two-copies risk explicitly: even a mechanical "add the same background call in
  `mcp.py`" fix would leave the policy duplicated in two files that must stay in sync by hand.

## Coverage & gaps

**Covered:** All 4 ACs live, both AC-4 static and live inspection, all 3 edge cases the plan named
(no-mention post, literal `@`-completion, cold-start failure mode), plus a *real* `git clone` for
AC-3 rather than the plan's lighter fallback adaptation (cheap enough to do faithfully; no second
falkor-chat instance was needed since the clone reused the already-running server, per the plan's
own reasoning that a second server isn't necessary to validate "no manual wiring").

**Gaps / not (re-)tested, deliberately, per the test plan's own scope:**
- falkor-chat's MCP tool *implementations* beyond the trigger-wiring gap found — `send_message`/
  `read_messages` argument handling, error surfaces, and the REST-path trigger mechanism itself are
  already covered by falkor-chat's own suite and `mention-reply-delivery-report.md`; not
  re-verified here except as needed to isolate D-1's root cause.
- LLM reply *content* quality — out of scope; AC-2 only requires surfacing, not judging the reply.
- A genuinely independent second falkor-chat instance for AC-3 — the real clone shared this pass's
  already-running server instead; judged sufficient since AC-3's actual concern (config
  availability, no manual wiring) doesn't depend on server independence.
- `kiro-cli doctor` — not re-run in this pass after the one incidental run during environment
  probing (see note below); its general reliability is out of scope for this feature.

## Feedback & recommendations

- **D-1 should be filed against falkor-chat, not against this Kiro feature.** The Kiro-side
  deliverable is correctly built to the spec it was given; the spec's assumption (implicit in FR-4/
  FR-5/AC-2) that "@mention via MCP produces a reply the same way @mention via REST does" turned out
  to be false in the running system. Recommend routing to `falkor-chat/docs/BACKLOG.md` as a new
  K-item; this report's Defect D-1 section has the reproduction steps and root cause ready to copy.
- **Demo-narrative risk if D-1 isn't fixed before the actual live demo runs:** `kiro/README.md`'s
  instructions, followed exactly as written, will appear to "hang" after the read-back step with no
  visible error — worth flagging to whoever presents next, even before a code fix lands, so they
  aren't caught off guard live.
- **Testability observation, not a defect:** `demo-welcome` is a shared, ever-growing fixture reused
  across multiple QA passes (32 messages already present before this pass started; 36 after). None
  of this pass's test items depended on the thread being empty (all compared before/after deltas
  and specific new `msgId`s, per this plan's own §4 setup note), but a future pass attempting a
  naive "assert thread is empty" or "assert exactly N messages" check would break. Recommend future
  QA passes continue the delta/msgId-comparison pattern rather than absolute counts.
- **Environment note, not scored as a test item:** during environment probing (before test
  execution began), `kiro-cli doctor` was run once and unexpectedly auto-remediated shell
  integration by appending sourcing lines to `~/.bashrc` and `~/.profile` (outside the repo, not
  destructive — idempotent one-line appends, confirmed via `tail`/`stat`). This was flagged to the
  user transparently as it happened. Recorded here so it isn't mistaken for a repo change; `doctor`
  was not run again for the remainder of this pass specifically to avoid a second unplanned
  mutation. Captured as a kaizen learning (see below) since it isn't documented anywhere the plan or
  requirements pointed to.

## Final state

falkor-chat is **left running** (`http://localhost:8000/`, confirmed `HTTP:200` at the end of this
pass) — stopped once, deliberately, for TP-008's cold-start check, and restarted immediately
afterward in the same terminal/log stream (`falkor-chat/scripts/start_server.sh`, same workspace).
FalkorDB (Docker) was never touched. No falkor-chat file was modified by this pass (test-only,
read/write against the running app's REST+MCP surfaces, no source edits).
