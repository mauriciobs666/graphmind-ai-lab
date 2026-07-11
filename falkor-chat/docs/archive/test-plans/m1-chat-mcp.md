# Test Plan — M1 Chat Core (REST + MCP)

> **Author:** qa-engineer (first-spin, proxy-run by cobb) · **Date:** 2026-07-01
> **Under test:** `falkor-chat/server/` — FastAPI REST API + MCP (Streamable-HTTP) front door, one process over FalkorDB.
> **Companion report:** [`../test-reports/m1-chat-mcp-report.md`](../test-reports/m1-chat-mcp-report.md)

## 1. Scope & objective

Verify the **behavior and contracts** of the M1 server from the outside in: both front doors
(browser REST + agent MCP) over the shared `Services` layer, against a live FalkorDB `ws:acme`
tenant. Confirm the M1 acceptance surface works end-to-end: channels, threads, thread-scoped
message append, @mentions validation, read-cursors, full-text search, and error mapping.

This is the **acceptance/behavior altitude** — the black-box complement to the existing unit/
integration suite (`server/tests/`, 57 tests). We establish that suite as a baseline, then drive
the *running* server (HTTP + a real MCP client session) to catch anything the in-process tests miss.

## 2. References (sources of truth)

- `falkor-chat/README.md` — "Run the M1 server" + M1 milestone row.
- `falkor-chat/docs/DESIGN.md` §14–§15 (client/server architecture).
- `falkor-chat/docs/archive/plans/m1-chat-mcp.md` — feature plan / acceptance criteria.
- Code: `server/falkorchat/{api,mcp,services,app,config,schemas}.py`.
- Existing tests: `server/tests/` (repository/services live, MCP, REST, app-mount).

## 3. Risk assessment (drives prioritization)

| Area | Risk | Why |
|---|---|---|
| Two front doors, one service | **High** | REST and MCP must produce identical behavior; a divergence is invisible to single-door unit tests. |
| Error → HTTP status mapping | **High** | `ServiceError` handled at app layer (`_register_error_handlers`), not the router. Regressions here surface as 500s. |
| Read-cursor advance semantics | **High** | `read_messages` is read-only with explicit `since`, read-write (advances cursor) without it. Easy to get subtly wrong; agent catch-up depends on it. |
| Mentions validation | **Med** | Unknown mention must be rejected (400), known mentions accepted; dedup order-preserving. |
| Full-text search | **Med** | Depends on the full-text index from bootstrap; keyword semantics. |
| Input validation | **Low-Med** | `q` min-length, `limit` bounds (1–200), missing bodies. |

## 4. Environment & data setup

- FalkorDB up (docker `falkordb-dev`, `:6379`). Schema bootstrapped: `./scripts/bootstrap_schema.sh acme`.
- Server: `.venv/bin/uvicorn falkorchat.app:app --host 127.0.0.1 --port 8000` (REST at `/`, MCP at `/mcp`).
- Hardcoded tenant `ws=acme`, actor `u1` (`config.py`).
- Test data: two `User` members seeded into `ws:acme` — `u1` (Alice), `u2` (Bob) — for the mention happy-path (none were seeded by bootstrap; see report feedback).

## 5. Entry / exit criteria

- **Entry:** baseline suite green (57 passed); server reachable (`GET /` → 200); schema present.
- **Exit:** every test item has a recorded outcome with evidence; defects filed with repro steps; verdict stated.

## 6. Test items

### REST front door
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| TP-001 | Create channel | `POST /channels {name}` | 201; body `{channelId,name,createdAt}` | High | functional |
| TP-002 | List channels | `GET /channels` after TP-001 | 200; array contains the created channel | High | functional |
| TP-003 | Create thread (happy) | `POST /channels/{cid}/threads {title}` | 201; `{threadId,channelId,title,createdAt}` | High | functional |
| TP-004 | Create thread on missing channel | `POST /channels/nope/threads` | **404** `ChannelNotFoundError` | High | contract |
| TP-005 | List threads | `GET /channels/{cid}/threads` | 200; array contains the created thread | Med | functional |
| TP-006 | Post first message | `POST /threads/{tid}/messages {text}` | 201; `{msgId,threadId,authorId:u1,role:user,...}` | High | functional |
| TP-007 | Post subsequent message | second `POST` into same thread | 201; distinct msgId (first-vs-subsequent variant) | High | functional |
| TP-008 | Post with known mention | `POST …/messages {text,mentions:[u2]}` | 201; `mentions:[u2]` echoed | Med | functional |
| TP-009 | Post with unknown mention | `mentions:[ghost]` | **400** `UnknownMemberError` | High | contract |
| TP-010 | Post to missing thread | `POST /threads/nope/messages` | **404** `ThreadNotFoundError` | High | contract |
| TP-011 | Read thread | `GET /threads/{tid}/messages` | 200; messages in order | High | functional |
| TP-012 | Get message by id | `GET /threads/{tid}/messages/{mid}` | 200; the message | Med | functional |
| TP-013 | Get missing message | `GET /threads/{tid}/messages/deadbeef` | **404** | Med | contract |
| TP-014 | Search hit | `GET /search?q=<word from a posted msg>` | 200; array includes the message | Med | functional |
| TP-015 | Search validation | `GET /search` (no `q`) and `?q=x&limit=0` | 422 (missing q) / 422 (limit<1) | Low | contract |

### MCP front door (real Streamable-HTTP client session at `/mcp`)
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| TP-020 | Tool discovery | MCP `initialize` + `list_tools` | tools: `send_message`, `read_messages`, `create_thread` | High | contract |
| TP-021 | create_thread via MCP | call `create_thread(channel_id,title)` | thread dict; visible via REST too | High | integration |
| TP-022 | send_message via MCP | call `send_message(body,re=tid)` | message dict; authored by configured actor `u1` (client `frm` ignored) | High | integration |
| TP-023 | Cross-door parity | message posted via MCP is readable via REST `GET /threads/{tid}/messages` | same message, one timeline | High | integration |
| TP-024 | read_messages cursor advance | `read_messages(re=tid)` twice | 1st returns backlog; 2nd returns empty (cursor advanced) | High | functional |
| TP-025 | read_messages explicit since = read-only | `read_messages(re=tid, since=0)` after cursor advanced | returns backlog; cursor NOT advanced | Med | functional |

### Baseline
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| TP-000 | Unit/integration baseline | `pytest -q` | 57 passed | High | regression |

### Follow-up (2026-07-01) — residual-gap items

Second pass extending §6, scoped to two residual gaps from the first report's §4 "Coverage & gaps":
room-wide `read_messages` (no `re`) and its "never advances a cursor" guarantee, plus a dated
re-confirmation of DEF-1. Same environment (live server, `ws:acme`, actor `u1`), driven via a real
MCP client session at `/mcp/` and `curl`.

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| TP-026 | Room-wide `read_messages` (no `re`) spans threads & advances no cursor | Create channel (REST) + 2 threads A,B (MCP `create_thread`); post one msg into each (MCP `send_message`); call `read_messages()` with no `re`/`since`; then thread-scoped `read_messages(re=A)` and `read_messages(re=B)` | Room-wide read returns messages spanning **both** threads (both msgIds present). Because it advances **no** cursor, each thread-scoped read still returns that thread's backlog | High | functional |
| TP-027 | DEF-1 re-confirmation — `POST /mcp` vs `POST /mcp/` | `curl -XPOST /mcp` and `/mcp/` with `initialize` handshake body | `POST /mcp` → **405**; `POST /mcp/` → **200** with a valid `initialize` result (defect still open) | Med | contract/regression |

## 7. Out of scope (this spin)

Web UI DOM/JS behavior; performance/load; auth (M1 is single hardcoded tenant, unauthenticated MCP
by design); concurrency/race conditions; M2+ features (embeddings, vector retrieval, workflows).
