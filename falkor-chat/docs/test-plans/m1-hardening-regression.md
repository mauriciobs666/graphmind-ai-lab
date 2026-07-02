# Test Plan — M1 Hardening Regression (K-004)

> **Author:** qa-engineer · **Date:** 2026-07-02
> **Under test:** `falkor-chat/server/` — FastAPI REST + MCP (Streamable-HTTP) + static web UI, one process over FalkorDB, at commit `707209a`.
> **Companion report:** [`../test-reports/m1-hardening-regression-report.md`](../test-reports/m1-hardening-regression-report.md)
> **Prior pass (build on, do not duplicate):** [`m1-chat-mcp.md`](m1-chat-mcp.md) · [`../test-reports/m1-chat-mcp-report.md`](../test-reports/m1-chat-mcp-report.md) (verdict PASS, one defect DEF-1).

## 1. Scope & objective

A **second, hardening-focused** black-box pass. Commit `707209a` (K-004) fixed five live-verified
defects via TDD (server suite 57→68). The prior QA pass (commit `def2959`) already established the
M1 acceptance surface works and is **not** re-run wholesale here. This pass verifies, from the
outside in (REST `curl` + a real MCP client session + the exact HTTP the web UI issues), that:

1. **The five `707209a` fixes actually hold live** — not just via the unit tests written for them,
   but through the REST/MCP surface at the acceptance altitude.
2. **Genuinely untested integration/acceptance behaviour** the unit suite doesn't reach: cursor +
   `limit` pagination loss across real MCP reads, cross-door parity *after* the ordering change,
   the web UI's three write flows through the real HTTP surface, and search error-mapping.
3. **Documented-but-unfixed residuals** (parked in `kaizen/plan.md`) still reproduce — recorded as
   dated confirmations/observations, not new defects.

## 2. References (sources of truth)

- `707209a` diff (five fixes + doc sync) — `git show 707209a`.
- `falkor-chat/AGENTS.md` — write-path invariants (zero-rows, chronological-cursor bullets).
- `falkor-chat/docs/QUERIES.md` §4 (write paths / zero-rows note), §9 (since-reads / ordering
  invariant / §9.3 no-member no-op), §5 (search).
- `falkor-chat/docs/DESIGN.md` §15.1 (trailing-slash gotcha), §15.2 (cursor semantics).
- Code: `server/falkorchat/{app,api,services,repository,mcp,config}.py`; `web/app.js`.
- `falkor-chat/kaizen/{history,plan}.md` — K-004 record + parked residuals.

## 3. Risk assessment (drives prioritization)

| Area | Risk | Why |
|---|---|---|
| Cursor + `limit` message loss (fix #2) | **High** | The worst prior defect class. Only reproduces with `limit` < backlog across *repeated* cursor reads — a combination the single-read happy paths never exercise. Silent, permanent data loss for an agent catching up. |
| Silent no-op writes (fix #1) | **High** | Previously 201-but-nothing-stored. Live proof must show the write **persists** (read-back), including on a *fresh* tenant where the actor was never manually seeded. |
| `POST /mcp` (fix #4 / DEF-1) | **Med** | Documented endpoint; prior pass had it 405. An agent following the docs must connect at both spellings now. |
| Search error mapping (fix #5) | **Med** | A malformed query must be a client 400, not a server 500 (crash-signalling to clients/monitoring). |
| Cross-door parity after ordering change | **Med** | The §9 ORDER BY changed; REST vs MCP must still present one coherent, chronological timeline. |
| Web UI write flows | **Med** | Never exercised at the HTTP altitude in the prior pass (was out of scope). The JS issues specific request shapes; a contract drift breaks the only human front door. |
| Parked residuals | **Low** | Known/deferred; confirm they still behave as documented so the backlog stays accurate. |

## 4. Environment & data setup

- FalkorDB `falkordb-dev` (`falkordb/falkordb:edge`) on `:6379`; schema bootstrapped for `reference`
  + `ws:acme` (`./scripts/bootstrap_schema.sh acme`).
- Running server: `.venv/bin/uvicorn falkorchat.app:app --host 0.0.0.0 --port 8000` (pre-existing;
  tenant `ws:acme`, actor `u1`). Confirmed live before starting.
- **Test data (additive, non-destructive):** a peer `User` `u2` seeded into `ws:acme` for the mention
  happy-path (bootstrap seeds no members; same approach as the prior pass). New channels/threads/
  messages created for this pass; pre-existing incidental data (`general`, `first thread`) left alone.
- **Fresh-tenant item (HR-002):** an isolated throwaway workspace `ws:qahard` bootstrapped and a
  second uvicorn started on `:8100` with `FALKORCHAT_WS_ID=qahard`, actor `u1` **never manually
  seeded** — torn down after. Does not touch `ws:acme`/`reference`.
- MCP driven with the real `mcp` python client (`streamablehttp_client` + `ClientSession`) at
  `http://127.0.0.1:8000/mcp/`; REST with `curl`.

## 5. Entry / exit criteria

- **Entry:** both baselines green (server `pytest -q` → 68; `./scripts/test_queries.sh` → 92/92);
  server reachable (`GET /` → 200); schema present.
- **Exit:** every item has a recorded outcome with real evidence (actual response/exit code/data
  state); defects filed with reproducible steps + severity by user impact; verdict stated.

## 6. Test items

### Baseline
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| HR-000 | Server unit/integration baseline | `pytest -q` | 68 passed | High | regression |
| HR-000b | Query suite baseline | `./scripts/test_queries.sh` | 92/92 | High | regression |

### Fix regressions — live black-box
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| HR-001 | DEF-1 closed — `POST /mcp` and `/mcp/` both work | `curl -XPOST` `initialize` to both paths | **both 200** (was 405 for `/mcp`) | Med | contract/regression |
| HR-002 | Fix#1 — fresh tenant persists out-of-the-box | Bootstrap `ws:qahard`; start server `:8100` (`FALKORCHAT_WS_ID=qahard`, actor `u1` never seeded); REST create channel→thread→post→read-back | post **201** *and* the message is **read back** (persisted, not a silent no-op) | High | integration |
| HR-003 | Fix#1 — write persists on `ws:acme` (no silent no-op) | REST post to a fresh thread, then `GET …/messages` | message present in thread read | High | functional |
| HR-004 | Fix#2 — cursor + `limit` pagination loses no message | MCP `send_message` ×5 to a fresh thread; `read_messages(re=T, limit=2)` repeatedly until empty | union of pages == all 5 msgIds, **no skips, no dups**, contiguous chronological | High | functional |
| HR-005 | Fix#2 — since-read chronological, mention flagged not resorted | Post plain, then a msg mentioning `u2`, then plain; `read_messages` as `u2` from since=0 | order == post order; the mention carries `isMention=true` **in place** (not hoisted first) | High | functional |
| HR-006 | Fix#2 — explicit `since` is a pure read (cursor untouched) | `read_messages(re=T, since=0)` after cursor advanced | returns backlog; a subsequent cursor read (no `since`) still reflects the un-mutated cursor | Med | functional |
| HR-007 | Fix#2 — empty page advances nothing | `read_messages(re=T)` twice with no new posts | 1st returns new backlog; 2nd returns `[]`; then a new post is delivered on the next read (cursor not jumped past it) | Med | functional |
| HR-008 | Fix#5 — search syntax error → 400 | `GET /search?q=hello"unbalanced` | **400** `InvalidSearchQueryError` (not 500) | High | contract |
| HR-009 | Fix#5 — valid search still 200 | `GET /search?q=<posted word>` | 200; array includes the message | Med | functional |

### Cross-door parity (after the §9 ordering change)
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| HR-010 | REST ↔ MCP one timeline | Post one msg via REST + one via MCP into one thread; read via both doors | both messages visible via both doors, chronological, author `u1` | Med | integration |

### Web UI write flows — through the exact HTTP the JS issues
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| HR-011 | UI create channel | `POST /channels {name}` (as `app.js` `new-channel`) | 201; channel listed | Med | e2e |
| HR-012 | UI create thread | `POST /channels/{cid}/threads {title}` (as `new-thread`) | 201; thread listed | Med | e2e |
| HR-013 | UI post message w/ mention | `POST /threads/{tid}/messages {text,mentions:[u2]}` (as `composer`+`parseMentions` for `@u2`) | 201; `mentions:["u2"]` echoed; readable in thread | Med | e2e |
| HR-014 | UI search | `GET /search?q=…` (as `search-form`) | 200; hits array shape (`text`,`createdAt`,`score`) | Low | e2e |

### Documented residuals — dated confirmation (parked in `kaizen/plan.md`, not new defects)
| ID | Title | Steps | Expected (as documented) | Pri | Type |
|---|---|---|---|---|---|
| HR-015 | `GET /threads/{tid}/messages/{mid}` ignores `tid` | Fetch a real msgId under a **wrong/nonexistent** thread path | message still returned 200 (nested route unvalidated) — residual still open | Low | exploratory |
| HR-016 | Web UI `parseMentions` 400s whole send on any `@token` | `POST …/messages {text:"hi @nobody", mentions:["nobody"]}` (what the JS would send) | **400** `UnknownMemberError` — send fails on a casual `@handle` — residual still open | Low | exploratory |
| HR-017 | `read_thread` (§4) omits `isMention` → UI highlight dead code | `GET /threads/{tid}/messages` on a thread with a reader-mention | rows carry no `isMention` key — UI `.mention` class never triggers — residual still open | Low | exploratory |

## 7. Out of scope (this spin)

Everything the prior pass already covered and that `707209a` did not touch (basic CRUD contracts,
404/422 mapping, tool discovery — see `m1-chat-mcp-report.md`); real browser DOM/JS execution
(flows are exercised via their HTTP requests); performance/load; auth; concurrency/idempotency
(parked residual, needs a concurrent-writer harness); M2+ features.
