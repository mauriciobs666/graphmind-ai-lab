# Test Report — M1 Chat Core (REST + MCP)

> **Author:** qa-engineer (first-spin, proxy-run by cobb) · **Date:** 2026-07-01
> **Plan:** [`../test-plans/m1-chat-mcp.md`](../test-plans/m1-chat-mcp.md)
> **Build under test:** working tree at commit `def2959` (M1 server) · FalkorDB `falkordb-dev` @ `:6379`, tenant `ws:acme`.

## 1. Summary

**Verdict: PASS with one minor documentation/contract defect.**

All 22 functional/contract test items across both front doors passed against the **live** server,
on top of a green unit baseline (57/57). The REST API and the MCP Streamable-HTTP front door
produce identical, correct behavior over the shared service layer: channels, threads, thread-scoped
append, mention validation, read-cursors, full-text search, and error→status mapping all behave as
specified. One defect found: the MCP endpoint responds only at `/mcp/` (trailing slash); `POST /mcp`
— the spelling the README advertises — returns **405**.

## 2. Results

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-000 | Unit/integration baseline | ✅ PASS | `pytest -q` → **57 passed** in 1.34s |
| TP-001 | Create channel | ✅ PASS | 201, `channelId=b1a04745…` |
| TP-002 | List channels | ✅ PASS | array includes created channel |
| TP-003 | Create thread (happy) | ✅ PASS | 201, `threadId=2b9a783e…` |
| TP-004 | Thread on missing channel | ✅ PASS | **404** `{"error":"ChannelNotFoundError","detail":"nope"}` |
| TP-005 | List threads | ✅ PASS | array includes created thread |
| TP-006 | Post first message | ✅ PASS | 201, `authorId=u1, role=user` |
| TP-007 | Post subsequent message | ✅ PASS | 201, distinct `msgId` |
| TP-008 | Post with known mention `u2` | ✅ PASS | 201, `mentions:["u2"]` echoed |
| TP-009 | Post with unknown mention | ✅ PASS | **400** `{"error":"UnknownMemberError","detail":"['ghost']"}` |
| TP-010 | Post to missing thread | ✅ PASS | **404** `ThreadNotFoundError` |
| TP-011 | Read thread | ✅ PASS | 3 messages, insertion order preserved |
| TP-012 | Get message by id | ✅ PASS | 200, correct message |
| TP-013 | Get missing message | ✅ PASS | **404** |
| TP-014 | Search hit (`q=pineapple`) | ✅ PASS | 1 hit = the seeded message (full-text index works) |
| TP-015 | Search validation | ✅ PASS | missing `q` → **422**; `limit=0` → **422** |
| TP-020 | MCP tool discovery | ✅ PASS | tools: `create_thread`, `read_messages`, `send_message` |
| TP-021 | `create_thread` via MCP | ✅ PASS | `threadId=ae82121c…` |
| TP-022 | `send_message` via MCP | ✅ PASS | `authorId=u1`; client `frm="agent-x"` correctly **ignored** (plan Q#1) |
| TP-023 | Cross-door parity | ✅ PASS | MCP-posted message readable via REST `GET /threads/{id}/messages` (author `u1`) |
| TP-024 | Cursor advance | ✅ PASS | 1st read → 1 msg; 2nd read → **0** (cursor advanced to clock) |
| TP-025 | Explicit `since=0` is read-only | ✅ PASS | returns backlog (1) after cursor already advanced; cursor untouched |

**22/22 functional+contract items passed · baseline 57/57 · 1 defect (below).**

## 3. Defects

### DEF-1 — MCP endpoint 405s at `/mcp`; only `/mcp/` (trailing slash) works — **Severity: Low-Med**
- **Impact:** the README ("Agents connect to MCP at `http://localhost:8000/mcp`") and DESIGN Appendix A
  advertise `/mcp`, but a client that POSTs to `/mcp` gets **405 Method Not Allowed**. An agent author
  following the docs verbatim fails to connect; the official `mcp` Python client itself does not
  auto-append the slash and errors out. Only `/mcp/` completes the handshake.
- **Repro:**
  ```
  curl -s -o /dev/null -w "%{http_code}\n" -XPOST http://127.0.0.1:8000/mcp \
    -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}'
  # → 405
  # same POST to http://127.0.0.1:8000/mcp/  → 200
  ```
- **Cause (likely):** `app.mount("/mcp", mcp_app)` with FastMCP `streamable_http_path="/"` makes the
  real POST route `/mcp/`; Starlette redirects only bare GETs, so `POST /mcp` is unrouted → 405.
- **Recommended fix (defer to `coder`/`tdd-engineer`):** either (a) redirect/alias `POST /mcp` → `/mcp/`,
  or (b) update README + DESIGN Appendix A to state the endpoint is `/mcp/` and note clients must keep
  the trailing slash. Add a regression test asserting `POST /mcp` behaves as documented.

## 4. Coverage & gaps

- **Covered:** both front doors end-to-end against a live server; all M1 acceptance flows; the three
  error-mapping paths (404/404/400) + input validation (422); full-text search; cursor read/advance
  and read-only semantics; cross-door parity.
- **Not covered this spin (residual risk):** web UI (DOM/JS); concurrency/idempotency under parallel
  writes; large-payload / pagination limits (`limit` upper bounds beyond the 422 boundary);
  room-wide `read_messages` (no `re`) and its "never advances" guarantee; `quotedId`/reply threading
  shape; performance. Recommend a second pass for room-wide reads and concurrency.

## 5. Feedback & recommendations

1. **DEF-1 is the actionable item** — small fix, real doc-vs-behavior gap that will bite agent authors.
2. **Test-data seam:** `bootstrap_schema.sh` seeds no members, so the mention **happy path** can't be
   exercised without manual seeding (I added `u1`/`u2` to `ws:acme`). Consider a `seed_demo.sh` or a
   fixture that plants the configured actor + a peer, so acceptance runs are reproducible out of the box.
2b. The configured actor `u1` is **not required to exist** as a node for `post_message` to succeed
   (only *mentions* are validated). Fine for M1, but worth an explicit decision/test before auth lands.
3. **Response-shape consistency:** `POST …/messages` returns `{msgId,threadId,authorId,text,role,createdAt,mentions}`
   while `GET …/messages/{id}` returns `{msgId,text,role,createdAt,authorId,authorType,quotedId}` and
   thread-list projects `updatedAt` (not `createdAt`). All intentional-looking, but a documented
   response schema per endpoint would make the contract testable and stop drift.
4. **No uniqueness on channel name** — two channels named "general" coexisted. Likely by design for M1;
   flagging so it's a conscious choice.

## 6. How this run was executed

- Baseline: `cd server && .venv/bin/python -m pytest -q` → 57 passed.
- Live server: `.venv/bin/uvicorn falkorchat.app:app --host 127.0.0.1 --port 8000`.
- REST driven with `curl`; MCP driven with a real `mcp` Python client session
  (`streamablehttp_client` + `ClientSession`) against `/mcp/`.
- Tenant `ws:acme` bootstrapped via `./scripts/bootstrap_schema.sh acme`; two `User` members seeded
  for the mention happy-path. No other graphs touched; no code under test modified.

---

## Addendum — Follow-up pass (2026-07-01)

> Scoped second pass extending the plan's **Follow-up (2026-07-01)** items, closing two residual gaps
> from §4: room-wide `read_messages` (no `re`) and its "never advances a cursor" guarantee (TP-026),
> and a dated re-confirmation of DEF-1 (TP-027). Same build (`def2959`), live server, `ws:acme`,
> actor `u1`. Baseline re-established green before executing.

### A1. Results

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-000 (re-run) | Unit/integration baseline | ✅ PASS | `pytest -q` → **57 passed** in 1.27s |
| TP-026 | Room-wide `read_messages` spans threads & advances no cursor | ✅ PASS | Room-wide `read_messages()` (no `re`/`since`) returned **6** msgs (4 pre-existing + 2 new) containing **both** new msgIds — `680b30cb…` (thread `e4b66faf…`) and `35f565cc…` (thread `84ce5fa3…`) → spans both threads. Cursor untouched: subsequent `read_messages(re=A)` returned A's backlog (1 msg) and `read_messages(re=B)` returned B's backlog (1 msg); had the room-wide read advanced either per-thread cursor, these would have returned 0. |
| TP-027 | DEF-1 re-confirmation (`POST /mcp` vs `/mcp/`) | ✅ PASS (defect still reproduces) | `POST /mcp` → **405** `{"detail":"Method Not Allowed"}`; `POST /mcp/` → **200**, `content-type: text/event-stream`, `mcp-session-id` issued, valid `initialize` result (`serverInfo.name="falkor-chat"`). |

**2/2 follow-up items passed · baseline re-confirmed 57/57.**

### A2. Room-wide read semantics — confirmed contract (TP-026)

`services.read_messages` with `thread_id=None` reads workspace-wide from epoch 0 (or explicit
`since`) via `repository.read_ws_since` and returns rows spanning **all** threads. There is no
room-wide cursor in M1, so **no cursor is created or advanced** — verified behaviorally: two
brand-new threads whose per-thread cursors were still at 0 both returned their backlog on the
thread-scoped read that followed the room-wide read. Note: room-wide rows carry
`{msgId,text,role,createdAt,authorId,authorType,isMention}` and **omit `threadId`** — thread
membership was proven here by cross-referencing the msgIds returned by `send_message` (which does
carry `threadId`). A client that needs to bucket a room-wide read by thread cannot do so from the
read payload alone; flagging as a minor testability/usability observation (not a defect).

### A3. DEF-1 status

**Still open / reproduces** as of 2026-07-01 on build `def2959`. No change since the first pass; the
recommended fix (redirect/alias `POST /mcp` → `/mcp/`, or correct README + DESIGN Appendix A to the
trailing-slash spelling, plus a regression test) remains outstanding. TP-027 gives it a dated
re-confirmation.

### A4. How this follow-up was executed

- Baseline: `cd server && .venv/bin/python -m pytest -q` → 57 passed.
- Live server: `.venv/bin/uvicorn falkorchat.app:app --host 127.0.0.1 --port 8000` (started for this
  pass, stopped after).
- TP-026 driven with a real `mcp` Python client (`streamablehttp_client` + `ClientSession`) at
  `/mcp/`; channel created via REST (no `create_channel` MCP tool), threads + messages via MCP tools.
- TP-027 driven with `curl` (initialize handshake body) against `/mcp` and `/mcp/`.
- Only `ws:acme` touched (2 new threads + 2 messages appended, pre-authorized); no code modified.
