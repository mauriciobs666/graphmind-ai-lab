# Kaizen — Change History: falkor-chat

> Dated log of actual changes to the `falkor-chat` component. Most recent first.

## 2026-07-01 — QA: functional test pass on M1 (REST + MCP)

- **What:** first black-box/acceptance QA pass on the M1 server, driving the *running* process
  (curl over REST + a real `mcp` Streamable-HTTP client session) on top of the 57-test baseline.
  Added `docs/test-plans/m1-chat-mcp.md` and `docs/test-reports/m1-chat-mcp-report.md`.
- **Result:** 22/22 functional+contract items PASS · baseline 57/57. Verified both front doors over
  one service layer, error→status mapping (404/404/400), input validation (422), full-text search,
  read-cursor advance vs. explicit-`since` read-only, and REST↔MCP cross-door parity.
- **Defect found (DEF-1, low-med):** MCP endpoint 405s at `POST /mcp`; only `/mcp/` (trailing slash)
  completes the handshake — but README/DESIGN Appendix A advertise `/mcp`. Fix = alias/redirect
  `/mcp`→`/mcp/` **or** correct the docs, plus a regression test. See the report §3.
- **Feedback:** `bootstrap_schema.sh` seeds no members, so the mention happy-path needs manual seeding
  (consider a `seed_demo.sh`); per-endpoint response shapes vary (documented schema would make them
  testable); channel names non-unique. Details in the report §5.
- **Why:** first spin of the new `claude/qa-engineer` agent (proxy-run). No code under test changed.

## 2026-07-01 — K-003: M1 chat core finish — full-text search endpoint + web UI

- **What:** Closed out M1 chat core on top of the K-002 server, TDD and search-first.
  - **Full-text search (red→green per layer):** `repository.search_messages` (workspace-wide
    `db.idx.fulltext.queryNodes('Message', …)`, `QUERIES.md` §5 with the channel-scoping MATCH
    omitted) → `services.search_messages` (thin passthrough) → REST `GET /search?q=&limit=`
    (`q` required via `Query(..., min_length=1)`; `limit` bounded 1–200). **+5 tests** (2 live repo,
    1 fake-repo service, 2 TestClient incl. the `422` missing-`q` guard).
  - **Web UI:** minimal `web/{index.html, app.js}` — vanilla `fetch` over the same-origin REST API:
    channels list/create, threads list/create, thread messages + composer (parses `@id` handles into
    `mentions[]`), and a full-text search panel. HTML-escaped throughout.
  - **Serving:** `app.py` gained a `web_dir` param and mounts `StaticFiles(html=True)` at `/`
    **last** — `/` is a catch-all that must sit behind the REST routes and the `/mcp` mount
    (Starlette matches in registration order). Same-origin ⇒ no CORS. Mount is skipped if `web/` is
    absent. **+1 test** pinning "serves index at `/` **and** `/channels` still returns JSON."
- **Verified:** full server suite **57 passed** (was 51); query suite regression **92/92**. Smoke:
  assembled app serves the real `web/index.html` at `/`, `web/app.js` as `text/javascript`, and
  `/channels` JSON alongside — one process, three front doors (web, REST, MCP).
- **Docs (same change):** `DESIGN.md` §12 roadmap + §14.5 layout/serving note + §14.6 build order
  (steps 3–4 ✅); `README.md` roadmap/layout/run + "open http://localhost:8000/"; `AGENTS.md` server
  surface (static-mount-last rule, `/search`) and test count 51→57.
- **Plan items:** K-003 ✅ → **M1 chat core code-complete.** Parking lot now: `search` over MCP,
  `create_channel` over MCP (Q#4).

## 2026-07-01 — K-002 Step 2: M1 server (repository → services → MCP + REST), one process

- **What:** Built the first application code for the component (greenfield `server/` tree), bottom-up
  and test-first, completing K-002 (`docs/plans/m1-chat-mcp.md`). All against live FalkorDB.
  - **`repository.py`** — every method 1:1 with a verified `QUERIES.md` query: channels/threads (§3),
    `ensure_user`/`ensure_agent` (§2/§7), both message write paths with the atomic `MENTIONS_MEMBER`
    block (§4), `read_thread` (§4), `read_thread_since` (§9.1), `read_ws_since` (§9.2),
    `advance_cursor`/`get_cursor` (§9.3/9.4), `get_message` (§4), plus validation reads
    (`thread_exists`/`channel_exists`/`existing_members`/`thread_has_head`).
  - **`services.py`** — invariants: id/clock generation (server clock), first-vs-subsequent write
    dispatch, mention validation (`UnknownMemberError`), RO/RW `read_messages` dispatch + `cursorId`
    construction, `Channel`/`ThreadNotFoundError`.
  - **`mcp.py`** — FastMCP adapter; tools `send_message`/`read_messages`/`create_thread`, injectable
    service + context (Q#1: `frm` ignored, actor = `get_context()`).
  - **`api.py` + `schemas.py`** — REST surface (DESIGN §14.4) incl. optional `mentions[]` parity;
    `ServiceError` → 404/400.
  - **`app.py`** — `create_app()` mounts REST + MCP on one FastAPI process.
- **Live gotchas found & mitigated (now in AGENTS.md):** (a) `exists((t)-[:HEAD]->())` returns `true`
  with no edge on this build and `count{}` is unsupported → existence via `OPTIONAL MATCH … IS NOT
  NULL`; (b) MCP lifespan wiring (python-sdk #1367) — forward `mcp_app.router.lifespan_context` to
  `FastAPI(lifespan=…)` or the session manager never starts; set `streamable_http_path="/"` so the
  mount lands cleanly at `/mcp`; (c) `call_tool` returns `(content, structured)` with list results
  wrapped as `{"result": […]}`.
- **Env:** no `uv` on the box → `server/.venv` via `python3 -m venv`; deps fastapi/uvicorn/falkordb
  1.6.1/mcp 1.28.1/pytest/httpx.
- **Tests:** **51 passed** — repository (24 live), services (12 unit fake-repo + 2 live), MCP (4
  in-memory), REST (7 TestClient), app-mount/lifespan (2). Query suite regression **92/92**.
- **Verified end-to-end:** REST round-trip through the assembled app; MCP tool discovery lists the
  three tools; mention-prioritised reads; monotonic cursor advance.
- **Plan items:** K-002 Step 2 ✅ → **K-002 complete.** Deferred: web UI (M1), `create_channel` over
  MCP (Q#4), full-text `search` REST endpoint.

## 2026-07-01 — K-002 Step 1 (gate): schema + queries for mentions & read-cursors

- **What:** Landed the graph-dba gate for the M1 Chat MCP transport (`docs/plans/m1-chat-mcp.md`),
  all live-verified against `falkordb/falkordb:edge`. (1) `bootstrap_schema.sh`: added
  `ReadCursor.cursorId` range index + uniqueness constraint (index-before-constraint). (2)
  `QUERIES.md` §4: both message write paths now carry a `$mentions` list and append a
  `MENTIONS_MEMBER` write-block, atomically inside the single write query. (3) `QUERIES.md` new §9:
  `read_messages` since-reads — §9.1 thread-scoped, §9.2 workspace-wide, §9.3 monotonic cursor
  advance, §9.4 cursor read. (4) `test_queries.sh`: +25 assertions.
- **Q#2 resolved (member-match index strategy).** `GRAPH.PROFILE` showed `WHERE n.userId=$x OR
  n.agentId=$x` as a scan anchor degrading to an `All Node Scan`; the write path instead resolves
  each mention with dual `OPTIONAL MATCH (u:User)/(a:Agent)` + `coalesce` → two `Node By Index
  Scan`s. The `OR` form is kept only where `me`/`mem` is already bound (mention-flag, cursor read).
- **Two live gotchas found & mitigated (now in AGENTS.md):** (a) a bare empty `UNWIND` collapses the
  row stream, so `RETURN m` came back empty on a `$mentions=[]` post despite the writes committing —
  guarded with `UNWIND (CASE WHEN $mentions=[] THEN [null] ELSE $mentions END)` + a non-filtering
  `FOREACH`; (b) `collect(DISTINCT coalesce(u,a))` gives free dedup + unknown-skip and collapses the
  per-mention rows back to a single result row. Both proven: `$mentions=[]` is byte-identical to a
  plain post; `['u3','u3','a7','nope']` → 2 edges `[u3,a7]`, one row.
- **Corrections vs. the plan's candidate Cypher:** mention-flag match handles **Agent** readers
  (`me.userId=$meId OR me.agentId=$meId`, not `me {userId:…}`); author id returned via
  `coalesce(author.userId, author.agentId)` so Agent authors aren't null. §9.3 monotonic guard
  (`CASE WHEN $now > coalesce(rc.lastReadAt,0) …`) verified on this build (300 → stale 200 stays
  300 → 400).
- **RAM (rule #6):** +1 range index and +1 constraint per workspace; growth term is one `ReadCursor`
  node per *(member, thread)* read and one `MENTIONS_MEMBER` edge per mention. No new vector
  dimension → no embedding-RAM change.
- **Tests:** suite green at **new baseline 67/67 → 92/92** (+25: mention write-path incl.
  empty/dedup/unknown, §9.1 prioritised since-read + exclusion, §9.2 index-scan proof, §9.3
  monotonic/idempotent cursor + constraint block, §9.4 read + index-scan proof).
- **Plan items:** K-002 Step 1 ✅ (gate passed); Step 2 (repository → services → `mcp.py`/`app.py`
  → REST parity) unblocked.

## 2026-06-11 — K-001: `list_channels` query (list channels in a workspace)

- **What:** Authored and live-verified a `list channels` query and added it to `docs/QUERIES.md`
  §3 (Channels & threads), with assertions in `scripts/test_queries.sh`. Query:
  `MATCH (c:Channel) WHERE c.channelId > '' RETURN c.channelId, c.name, c.createdAt ORDER BY
  c.createdAt DESC LIMIT $limit`. The always-true `c.channelId > ''` predicate (every `channelId`
  is a non-empty string) anchors the listing on the **`Channel.channelId` range index** —
  `GRAPH.PROFILE` confirms `Node By Index Scan`, not `NodeByLabelScan`. Ordered by `createdAt`
  (channel **creation** time, newest-first), which is free once the scan is index-backed. Marked
  `GET /channels → list_channels` resolved in `DESIGN.md` §14.4 (was "gap — owned by graph-dba")
  and flipped the §14.6 prerequisite step to done.
- **Why:** the M1 REST surface (`GET /channels`, DESIGN §14.4) needed a verified query and
  `QUERIES.md` had none — it covered channel *members* (§2) and recent *threads* (§3) but not
  channels. Unblocks the `list_channels` repository method (§14.6 build order).
- **Trade-off noted:** true activity-recency (most-recent message/thread per channel) would need a
  `HAS_THREAD` → `Thread.updatedAt` expansion per channel — the Channel-level edge traversal §5.2
  deliberately avoids — so the cheap, index-backed **creation-time** ordering is used instead, and
  this is documented inline in `QUERIES.md`. No new index or constraint added; **zero per-workspace
  RAM cost** (reuses the existing `Channel.channelId` index).
- **Tests:** suite green at the **new baseline 64/64 → 67/67** (one §3 functional assertion +
  the standard §8 `assert_index_scan` pair; the plan's "65/65" estimate predated counting each
  `assert_*` call — the PROFILE proof is a two-line assertion per the existing §8 convention).
- **Plan items:** K-001 ✅ done.

## 2026-06-11 — Defined the M1 client/server application architecture

- **What:** Pinned the M1 application architecture and documented it as a new `docs/DESIGN.md` §14.
  Decisions: **transport = REST/JSON over FastAPI** (chosen after explicitly re-evaluating and
  rejecting gRPC — the only M1 client is a browser, so gRPC's typed-contract/streaming/service-mesh
  wins go unused and gRPC-Web would be pure bridge tax; WebSocket/SSE is the stronger M2 real-time
  path); **client = minimal web UI**; **real-time deferred to M2**; **single hardcoded tenant**
  (`ws=acme`, `user=u1`) injected at one FastAPI dependency seam so real auth drops in later without
  touching services/repo. Captured the layering (router → service → repository → db → FalkorDB),
  the REST surface → service → `QUERIES.md` mapping, proposed `server/` + `web/` layout, and the
  bottom-up TDD build order. Updated the §12 + README roadmap rows to point at §14.
- **Why:** User wanted the M1 client/server architecture nailed down before any code; the DESIGN
  doc previously only sketched the *operational* topology (§10), not the application code shape.
- **Plan items:** seeded **K-001** (`list_channels` query gap, owned by graph-dba — the one piece of
  the M1 REST surface with no verified query yet).

## 2026-06-11 — Adopted the kaizen plan/history convention

- **What:** Created `kaizen/{plan.md,history.md}` for the component, mirroring the sibling
  `claude/<agent>/kaizen/` projects (forward-looking backlog + dated change log). Replaced a
  short-lived `BACKLOG.md` draft with this structure.
- **Why:** User asked the component to track work the same way the sibling projects do, rather than
  a standalone backlog file.
- **Plan items:** K-001 recorded as the first active item.

## (prior) — M0 baseline

- M0 — Engine up: FalkorDB running (`falkordb/falkordb:edge`, Redis 8.2.2, module `999999`),
  design locked, schema bootstrap + canonical query library live-verified (`test_queries.sh`,
  64/64). Predates this log; see git history (e.g. `feat(falkor-chat): schema, query library,
  tests, and working context`) and `docs/DESIGN.md`.
