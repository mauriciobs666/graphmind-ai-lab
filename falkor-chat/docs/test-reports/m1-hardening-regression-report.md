# Test Report — M1 Hardening Regression (K-004)

> **Author:** qa-engineer · **Date:** 2026-07-02
> **Plan:** [`../test-plans/m1-hardening-regression.md`](../test-plans/m1-hardening-regression.md)
> **Build under test:** commit `707209a` (K-004 hardening) · FalkorDB `falkordb-dev` @ `:6379`,
> tenant `ws:acme`, actor `u1`. Live server `uvicorn falkorchat.app:app :8000` (pre-existing).
> **Prior pass:** [`m1-chat-mcp-report.md`](m1-chat-mcp-report.md) (verdict PASS, defect DEF-1 — now closed here as HR-001).

## 1. Summary

**Verdict: PASS. All five `707209a` hardening fixes hold live from the outside in; no new defects.**

Every one of the five K-004 fixes was re-verified black-box through the REST and/or MCP surface
(not just via its unit test), on top of green baselines (server **68/68**, query suite **92/92**).
The worst prior defect class — **cursor + `limit` pagination silently losing messages** — is
confirmed fixed: five messages read back in three `limit=2` pages with **no skips and no
duplicates**. The previously-open **DEF-1** (`POST /mcp` → 405) is closed: both `/mcp` and `/mcp/`
now return 200. A **fresh tenant persists out-of-the-box** end-to-end (new workspace, actor never
manually seeded → post 201 **and** read-back confirms storage). Cross-door parity holds after the
§9 ordering change. All three write flows the web UI issues succeed through the real HTTP surface.

Three **already-documented, deferred residuals** (parked in `kaizen/plan.md`) were re-confirmed as
still-open — recorded here as dated observations, **not** new defects.

## 2. Results

| ID | Item | Result | Evidence |
|---|---|---|---|
| HR-000 | Server unit/integration baseline | ✅ PASS | `pytest -q` → **68 passed** in 1.26s (re-confirmed 68 at close) |
| HR-000b | Query suite baseline | ✅ PASS | `./scripts/test_queries.sh` → **92/92 passed** |
| HR-001 | DEF-1 closed — `POST /mcp` & `/mcp/` | ✅ PASS | `curl -XPOST /mcp` → **200**; `/mcp/` → **200** (prior pass: `/mcp` was 405). Live MCP session: `serverInfo="falkor-chat"`, tools `[create_thread, read_messages, send_message]` |
| HR-002 | Fix#1 — fresh tenant persists out-of-box | ✅ PASS | Bootstrapped fresh `ws:qahard` (0 users); server on `:8100` (`FALKORCHAT_WS_ID=qahard`). Lifespan auto-created `u1`. `POST …/messages` → **201**; `GET …/messages` → **count 1**, author `u1` — persisted, not a silent no-op |
| HR-003 | Fix#1 — write persists on `ws:acme` | ✅ PASS | REST post to fresh thread → read-back returns the message (`count 1, texts ['hello @u2 welcome']`) |
| HR-004 | Fix#2 — cursor + `limit` loses no message | ✅ PASS | 5 posts, MCP `read_messages(re,limit=2)` ×N → pages `[msg-0,msg-1] [msg-2,msg-3] [msg-4]`; `seen=5 unique=5 covers_all=true no_dups=true` |
| HR-005 | Fix#2 — chronological, mention flagged not resorted | ✅ PASS | Reader `u1` mentioned in the **middle** message: order `[before, poke u1, after]`, `isMention=[false, true, false]` — flagged **in place**, not hoisted to front |
| HR-006 | Fix#2 — explicit `since` is a pure read | ✅ PASS | `read_messages(re,since=0)` then plain cursor read → still full backlog `[a,b]`; cursor untouched by the explicit-since read |
| HR-007 | Fix#2 — empty page advances nothing | ✅ PASS | 1st read `[x1]`; 2nd read `[]`; after a new post, 3rd read `[x2]` — the new message is **not** skipped (cursor did not jump past it) |
| HR-008 | Fix#5 — search syntax error → 400 | ✅ PASS | `GET /search?q=hello"unbalanced` → **400** `{"error":"InvalidSearchQueryError","detail":"RediSearch: Syntax error at offset 6 near unbalanced"}` (not 500) |
| HR-009 | Fix#5 — valid search still 200 | ✅ PASS | `GET /search?q=welcome` → **200**, 1 hit (`score 4.0`) |
| HR-010 | Cross-door parity (after §9 change) | ✅ PASS | REST post + MCP post in one thread → both doors read `[via-rest, via-mcp]` chronologically; author `u1` |
| HR-011 | UI create channel | ✅ PASS | `POST /channels {name}` → **201**; appears in `GET /channels` |
| HR-012 | UI create thread | ✅ PASS | `POST /channels/{cid}/threads {title}` → **201**; appears in `GET …/threads` |
| HR-013 | UI post message w/ mention | ✅ PASS | `POST …/messages {text,mentions:[u2]}` → **201**, `mentions:["u2"]` echoed; readable in thread |
| HR-014 | UI search | ✅ PASS | `GET /search?q=welcome` → **200**; hit shape carries `text`, `createdAt`, `score` |
| HR-015 | Residual: nested route ignores `tid` | ⚠️ CONFIRMED OPEN | `GET /threads/deadbeefwrongthread/messages/{realMsgId}` → **200** returns the message under a nonexistent thread path (documented in `kaizen/plan.md`) |
| HR-016 | Residual: `parseMentions` 400s whole send | ⚠️ CONFIRMED OPEN | `POST …/messages {text:"hi @nobody…", mentions:["nobody"]}` → **400** `UnknownMemberError` — a casual `@handle` fails the whole send (documented) |
| HR-017 | Residual: `read_thread` omits `isMention` | ⚠️ CONFIRMED OPEN | `GET /threads/{tid}/messages` rows keys `[authorId, authorType, createdAt, displayName, msgId, role, text]` — **no `isMention`**; the web UI `.mention` highlight is dead code (documented) |

**17/17 active test items passed · 3/3 documented residuals re-confirmed still-open · baselines 68/68 + 92/92.**

## 3. Defects

**No new defects.** All five hardened defects verified fixed live. The three ⚠️ items above are
**previously-documented, intentionally-deferred residuals** (K-004 review, parked in
`kaizen/plan.md`) — re-confirmed here with dated evidence so the backlog stays accurate. Ranked by
user impact for when they're picked up:

1. **HR-015 — `GET /threads/{tid}/messages/{msg_id}` ignores `{tid}`** — *Severity: Low.* Any
   message resolves under **any** thread path (incl. a nonexistent one), returning 200. No cross-
   tenant leak (still workspace-scoped by graph), but a client can construct a URL implying a
   message belongs to a thread it doesn't. Fix: validate membership or drop the nested `{tid}`
   segment. (Parked: "`GET …/{msg_id}` ignores `tid`".)
2. **HR-016 — web UI `parseMentions` rejects casual `@handles`** — *Severity: Low.* `web/app.js`
   turns every `@token` into a mention id, so one unknown `@name` 400s the entire message send —
   a human typing "@bob thanks" when `bob` isn't a member id loses their whole message with a raw
   `UnknownMemberError` alert. Fix: only send resolvable mentions, or treat unknowns as plain text.
   (Parked: "Web UI mention polish (b)".)
3. **HR-017 — `read_thread` (§4) never returns `isMention`** — *Severity: Trivial (cosmetic).* The
   canonical thread read omits the flag, so `renderMessages`'s `m.isMention` highlight branch can
   never fire. Dead code, no user harm. (Parked: "Web UI mention polish (a)".)

## 4. Coverage & gaps

- **Covered this pass:** all five K-004 fixes end-to-end live (fix#1 no-op → persistence incl. a
  genuinely fresh tenant; fix#2 cursor/limit pagination loss + chronological-with-flag ordering +
  pure-read + empty-page semantics; fix#4 `/mcp` slash; fix#5 search 400-not-500); cross-door
  parity after the ordering change; the web UI's three write flows + search through the exact HTTP
  the JS issues; three documented residuals re-confirmed.
- **Not covered (residual risk):**
  - **Concurrency / idempotency** — the parked "write retry idempotency + first-post race" (two
    concurrent first posts → two HEADs; same-`msgId` retry duplicates `NEXT`/`TAIL`) needs a
    concurrent-writer harness; single-process HTTP testing can't provoke it reliably. **Highest
    residual risk** before M2 agents write in parallel.
  - **Agent authorship** — writes hardcode `role="user"` and the author `MATCH` is `{userId:…}`;
    an `Agent` cannot author yet. Untested by design (parked; arrives with M2 auth).
  - **Millisecond `createdAt` ties** — a cursor advanced to a tied `createdAt` could skip a same-ms
    sibling at a page boundary. Not provoked here (posts were naturally ms-distinct); low risk
    single-process, real risk under concurrent writers.
  - **Room-wide `read_messages` `threadId` omission** — unchanged since the prior pass addendum A2;
    not re-tested (no code change).
  - Performance/load; auth; large-payload pagination upper bounds.

## 5. Feedback & recommendations

1. **The K-004 hardening is solid** — every fix survives black-box scrutiny at the acceptance
   altitude, including the subtle cursor/limit pagination invariant (HR-004/HR-007) that is the
   easiest to regress and the most damaging to lose. Good, durable work.
2. **Prioritize the concurrency/idempotency residual before M2.** It's the one high-risk area no
   current test exercises, and M2's parallel agent writers are exactly the trigger. Worth a
   dedicated concurrent-writer test harness (two threads racing a first-post; a same-`msgId`
   retry) rather than another single-process pass.
3. **HR-015 is the residual most worth closing soon** — a one-line membership check (or dropping
   the `{tid}` segment) removes a contract inconsistency that will confuse API clients.
4. **Test-data seam still bites** — `bootstrap_schema.sh` seeds no members, so the mention happy
   path needs a manually-seeded peer (`u2`) every pass. The prior report already recommended a
   `seed_demo.sh`; re-flagging — it would make acceptance runs reproducible out of the box.
5. **Response-shape consistency across read endpoints** (prior report §5.3) is unchanged: `read_thread`
   returns `displayName` but no `isMention`; the since-reads return `isMention` but no `displayName`.
   A documented per-endpoint response schema would make these contracts testable and stop drift —
   and would have surfaced HR-017 as a spec mismatch rather than dead UI code.

## 6. How this run was executed

- **Baselines:** `cd server && .venv/bin/python -m pytest -q` → 68 passed (re-confirmed at close);
  `cd falkor-chat && ./scripts/test_queries.sh` → 92/92.
- **Live server:** the pre-existing `uvicorn …:8000` (`ws:acme`, actor `u1`) — not restarted.
- **REST/web-UI/residual items:** `curl` against `:8000`, replicating the request shapes `web/app.js`
  issues (`new-channel`, `new-thread`, `composer`+`parseMentions`, `search-form`).
- **MCP items:** the real `mcp` python client (`streamablehttp_client` + `ClientSession`) at
  `http://127.0.0.1:8000/mcp/`; fresh channel per scenario (via REST) + threads/messages via the
  MCP tools, so per-thread cursors started clean.
- **HR-002 fresh tenant:** isolated throwaway `ws:qahard` bootstrapped, a second `uvicorn` on
  `:8100` (`FALKORCHAT_WS_ID=qahard`), tested, then the server stopped and `ws:qahard` **deleted**
  (`GRAPH.DELETE`). `ws:acme`/`reference` untouched.
- **Test data:** peer `User` `u2` seeded into `ws:acme` (additive, idempotent `MERGE`) for the
  mention path; new channels/threads/messages created alongside pre-existing incidental data. No
  code under test modified; no destructive mutation of `ws:acme`/`reference`.
