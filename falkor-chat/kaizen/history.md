# Kaizen — Change History: falkor-chat

> Dated log of actual changes to the `falkor-chat` component. Most recent first.

## 2026-07-05 — K-010: QA DEF-1 + DEF-2 closed (K-008 prerequisites)

- **What:** closed both defects from the K-007 QA pass, clearing K-008's gate. Coordinated
  delivery: **graph-dba** authored + live-verified the query layer, **tdd-engineer** wired the
  Python (strict red→green), verification re-run independently.
  1. **DEF-1 — member-id namespace guard (K-008 prerequisite).** Locked rule: member ids are
     **namespace-unique across `User`/`Agent`**. `ensure_user`/`ensure_agent` are now v2
     guarded-CREATE single-query bodies (QUERIES.md §2/§7, verified `Node By Index Scan` on
     both legs) returning an always-present `(created, existed, collided)` status row —
     idempotent re-ensure is a structural no-op; a cross-label collision writes nothing and
     raises `MemberIdCollisionError` (repository-level, re-exported by services);
     `existed AND collided` is a distinct corruption alarm. App startup with a configured
     actor colliding with an existing Agent id now **fails loudly** instead of silently
     minting a shadow `User` that eclipsed the Agent in every `coalesce(u, a)` lookup (the
     exact QA S3 repro). Same-label uniqueness constraints remain the concurrency backstop;
     the one-query-wide cross-label race window is documented, not closed (no engine
     cross-label constraint exists).
  2. **DEF-2 — fail-fast on unreachable FalkorDB.** `db.connect()` now passes
     `socket_connect_timeout`/`socket_timeout` (config-resolved `FALKORDB_CONNECT_TIMEOUT=5`
     / `FALKORDB_SOCKET_TIMEOUT=10`, env-overridable) and wraps failures in
     `FalkorDBUnreachableError` naming host:port + timeout + a start-script hint; a new
     `db.LazyFalkorDB` defers the first connection out of import — **importing
     `falkorchat.app` never touches the network** (the module-level `create_app()` used to
     hang ≥90s with zero output on WSL2's closed-port blackhole). Smoke re-verified: dead
     port → clean exit in ~6s with the actionable error. `app.py` docstring now matches
     reality.
- **Files:** `server/falkorchat/{repository,services,config,db,app}.py`;
  `server/tests/{test_repository,test_services,test_app}.py` + new `test_db.py`;
  `docs/QUERIES.md` §2/§7 (v2 ensures + contract table + locked rule);
  `scripts/test_queries.sh` (11 new DEF-1 assertions incl. PROFILE index checks);
  `AGENTS.md` (new locked-decision row; baselines) + root `AGENTS.md` (baselines).
- **Baselines (independently re-verified):** pytest **98 → 110**, query suite
  **115/115 → 126/126**; `reference` schema restored post-suite; `ws:acme` untouched.
- **Why:** DEF-1's silent misattribution was exactly the failure class K-007 closed, and it
  gated wiring real agent identities in K-008; DEF-2 bought dev/ops diagnosability on the
  README bare-`uvicorn` path (Compose was already shielded by `service_healthy`).

## 2026-07-05 — QA: acceptance pass on K-007 M2 groundwork

- **What:** black-box/acceptance QA pass at `94ab746`, scoped to what the K-007 dev suites
  structurally can't reach: concurrency through the real HTTP stack (single- and
  **two-process** writers), MCP-driven cursor paging over millisecond ties, agent `role` on
  every read surface, `backfill_thread_ids.sh` against real legacy-shaped data, and the
  actor-seam edges. Added `docs/test-plans/k007-m2-groundwork.md` and
  `docs/test-reports/k007-m2-groundwork-report.md`. Isolated `ws:qa` (created + deleted);
  `ws:acme`/`reference` untouched.
- **Result: PASS with two low-severity defects** — 18/18 items executed, 16 clean passes, on
  green baselines (server **98/98**, query suite **115/115**). Highlights: 12-way REST
  first-post hammer and a 20-write race across **two server processes** both yielded exactly
  one HEAD/TAIL and a contiguous chain; the cross-process run produced a **natural same-ms
  `createdAt` tie** and MCP cursor paging (`limit=3`) still delivered all 20 exactly once;
  agent-authored messages read `role: "assistant"` consistently on all five read surfaces;
  backfill script: 2 backfilled, then 0 (idempotent), `threadId: null` tolerated pre-backfill.
- **Defects (parked in `kaizen/plan.md`, not fixed here):**
  - **DEF-1 (low now, K-008 hazard):** no cross-label member-id uniqueness — a configured
    actor colliding with an existing `agentId` silently MERGEs a shadow `User` that eclipses
    the Agent in every `coalesce(u, a)` lookup (role derivation, `POSTED_BY`, mentions).
  - **DEF-2 (low, ops):** with FalkorDB unreachable, `uvicorn falkorchat.app:app` hangs
    indefinitely with zero output — `FalkorDB()` connects eagerly (no socket timeout) inside
    the module-level `create_app()`, falsifying the "building the app never requires a
    reachable FalkorDB" intent (hang-vs-refuse is WSL2-flavored; the eager import-time
    connect is real everywhere).
- **Why:** the prior QA report's top residual risks (concurrency/idempotency, agent
  authorship, ms-ties) were exactly K-007's targets — this pass closes that loop before K-008
  puts real agent writers on the system. No code under test changed.

## 2026-07-05 — K-007: M2 groundwork — agent authorship, v2 write-path guards, threadId denorm, composite cursors

- **What:** the six pre-agent-writer correctness/completeness items, landed per the approved
  plan (`docs/plans/m2-groundwork.md`) over the graph-dba's live-verified query deliverable
  (`docs/plans/m2-groundwork-queries.md`); plus the two server fold-ins.
  1. **Agent authorship** — §4 write paths resolve the author label-specifically (two indexed
     `OPTIONAL MATCH`es + `coalesce`), closing the `All Node Scan` *and* the silent no-op that
     made `Agent` authors unwritable; `services.post_message` derives `role` from the author's
     label via the new `repository.resolve_member_kinds` (`User → user`, `Agent → assistant`;
     replaces `existing_members` — one round trip for author + mention validation + role).
  2. **v2 self-guarding write paths (two reproduced defects fixed)** — each path wraps its write
     in a `FOREACH`+`CASE` guard inside the single `GRAPH.QUERY` and always returns a
     `(written, hadHead, dupMsg, authorFound)` status row (`repository.MessageWriteStatus`).
     Defect A: a same-`msgId` retry replay re-ran the relink clauses (NEXT self-loop, doubled
     `POSTED_BY`) — now a structural no-op reported as `dupMsg` = idempotent success. Defect B:
     two racing first-posts created two HEADs — the loser now refuses with `hadHead` and the
     service re-dispatches as subsequent (bounded 4-attempt loop; `Message` writes carry **no
     MERGE** — the uniqueness constraint stays as the verified all-or-nothing backstop).
     `REPLY_TO`-inside-the-guard live-verified in `test_queries.sh` (OQ4); repository fold-in
     waits for a reply surface.
  3. **`Message.threadId` denorm** — stamped inline by both write paths, deliberately unindexed;
     surfaced in §9.1/§9.2 since-reads, `/search`, and `GET /messages/{id}`. One-off
     `scripts/backfill_thread_ids.sh` (QUERIES.md §4.x; idempotent, HEAD-anchored, orphan
     caveat) — run against `ws:acme`: 0 backfilled (expected no-op, 0 messages).
  4. **Millisecond-tie correctness (reproduced page-boundary skip fixed)** — deterministic total
     order `(createdAt, msgId)` on both since-reads; formulation-A composite keyset predicate
     (still a bare `Node By Index Scan`); composite monotonic `ReadCursor`
     (`lastReadAt`, `lastReadMsgId`) — five scenarios verified, pre-K-007 cursors covered by
     `coalesce(…, '')`, no schema change; plus a lock-guarded monotonic per-process message
     clock in `Services` (same-ms ties impossible at the source). Explicit REST `?since=` keeps
     plain-`>` semantics (documented, OQ3).
  5. **TIMEOUT posture (docs-only, live-probed)** — keep legacy `TIMEOUT=1000`; per-query client
     override for future GraphRAG reads; **writes ignore TIMEOUT on this build** — bounded
     batches + input caps are the only write-path protection (DESIGN §10).
  6. **RAM line re-costed at 1024 dims (empirical)** — 12,387 B/message observed ≈ 12.4 KB ⇒
     ~1.25 GB per 100k-message workspace; `GRAPH.MEMORY USAGE` under-reports vector-index memory
     (size from `INFO memory` deltas) — DESIGN §11 rewritten; bootstrap default stays 1536 with
     an explicit choose-before-creation comment.
  - Fold-ins: `db.connect()` late-binds `config.FALKORDB_*`; `create_channel`/`create_thread`
    are plain `CREATE` (server-minted ids — creates documented **non-idempotent**;
    `create_thread` raises on a missing channel anchor).
  - Docs: QUERIES.md §2/§3/§4(+§4.x)/§5/§9 rewritten as the canonical v2 bodies; DESIGN
    §5.1/§5.3/§9/§10/§11/§12 (role values fixed to `user`/`assistant`, the falsified
    "idempotent via MERGE" claim replaced by the status-row contract); AGENTS.md decisions/
    facts/write-path rewrite; README + root AGENTS.md baselines.
- **Why:** prerequisites for AI agents writing concurrently (K-008): agents couldn't author at
  all, a client retry corrupted the thread chain, a first-post race forked it, and same-ms
  `createdAt` ties silently lost messages at cursor page boundaries.
- **Verified:** server suite **98 passed** (was 75; +23 — the plan's ≈95 estimate, exceeded by
  finer-grained regression tests); query suite **115/115** (was 92; +23 exactly as enumerated);
  `ruff check .` clean; defect regressions were watched fail red against the old code (replay →
  `(2 NEXT, 1 self-loop, 2 POSTED_BY)`; race → 2 HEADs) before the v2 queries landed; live
  8-worker concurrency hammer green (1 HEAD, 1 TAIL, contiguous chain of 8); backfill no-op
  proven on `ws:acme`.
- **Plan items:** K-007 ✅ done; K-008 (GraphRAG proper) unblocked; parking-lot fold-ins
  (`db.connect` bind, uuid `MERGE`) delivered. OQ6 (upstream FalkorDB filings: `GRAPH.MEMORY
  USAGE` vector under-report; one-shot instant-timeout anomaly) recommended to the user, not
  filed.

## 2026-07-04 — K-009: containerization (Dockerfile/compose) + CI + `falkordb-data` persistence fix

- **What:** first delivery-lifecycle pass for the component — container images, a compose stack,
  path-filtered CI, dependency pinning, and a critical data-persistence bug fix.
  1. **`falkordb-data` persistence fix (critical)** — `scripts/start_falkordb.sh` mounted the
     named volume at `/data` (the image's legacy `VOLUME`), but `falkordb/falkordb:edge` actually
     writes its Redis `dir` to **`/var/lib/falkordb/data`** (`FALKORDB_DATA_PATH`) — so **no graph
     data ever survived a container stop**; the volume persisted nothing. Live-verified 2026-07-04:
     data written under the `/data` mount vanished on restart; remounted at `/var/lib/falkordb/data`
     it survives. Fixed in the script (with an inline warning comment) and used in `compose.yaml`.
     `ws:acme` schema was re-bootstrapped after the fix (12 indexes).
  2. **`Dockerfile`** — M1 server image (`python:3.12-slim`): build context is the component root
     so the `server/` + `web/` sibling layout survives (app.py resolves `parents[2]/web`), editable
     install, non-root `appuser` runtime (install stays root-owned/read-only), `EXPOSE 8000`, and a
     `HEALTHCHECK` against the K-006 `GET /health` (200 only when FalkorDB answers).
  3. **`compose.yaml`** — two services: `falkordb` (same image/ports/volume as the script; redis-cli
     ping healthcheck) and `server` (built image, `FALKORDB_HOST=falkordb`, `depends_on:
     service_healthy`). The `falkordb-data` volume is declared **`external: true`** — compose must
     never create/re-create/remove the shared dev volume, and `down -v` is explicitly warned
     against. Header warns the script-started `falkordb-dev` container and compose share :6379 and
     the volume — never run both.
  4. **`.dockerignore`** — only `server/` (minus tests/venv/egg-info) + `web/` enter the build
     context; docs, kaizen, scripts, markdown excluded.
  5. **CI (`.github/workflows/falkor-chat.yml`)** — path-filtered to `falkor-chat/**` + the
     workflow itself; single job on ubuntu-latest with a **FalkorDB service container**
     (`falkordb/falkordb:edge`, health-gated) mirroring the local commands: `ruff check server` →
     server pytest (75-baseline) → `./scripts/test_queries.sh` (92/92-baseline). Deliberately
     tracks the floating `:edge` tag — the project's live-verified facts are pinned to it.
     **Never run yet** — first push to GitHub will tell (parking-lot item).
  6. **Dependency pins + ruff adoption** (`server/pyproject.toml`) — compatible-range pins for
     reproducible installs: `fastapi>=0.139,<0.140`, `uvicorn>=0.49,<0.50`, `falkordb>=1.6,<1.7`,
     `mcp>=1.28,<1.29`, `pytest>=9.1,<10`, `httpx>=0.28,<0.29`, `ruff>=0.14,<0.15`; ruff config
     (E,F,W,I / target py312 / line 100). Behavior-neutral import-order (I) fixes across
     `falkorchat/{api,app,services}.py` and `tests/{conftest,test_app,test_repository,test_services}.py`.
  7. **README** — compose run section added alongside the script path.
- **Why:** the component had no image, no one-command stack, and no CI; and the persistence bug
  meant the "durable" dev volume was silently empty — any container stop lost every graph.
- **Verified (2026-07-04 resume session):** fixed script started FalkorDB from a cold stop and
  `GRAPH.LIST` returned **`ws:acme`** — live proof graphs now survive downtime (`ws:k007scratch`
  residue also present, left untouched for the K-007 relaunch). Pins install-verified in a clean
  reinstall (fastapi 0.139.0, uvicorn 0.49.0, falkordb 1.6.1, mcp 1.28.1, pytest 9.1.1,
  httpx 0.28.1, ruff 0.14.14); `ruff check .` clean; server suite **75 passed**; query suite
  **92/92**. Compose stack itself not booted locally (shares :6379 + the volume with the running
  `falkordb-dev`); its build is exercised by CI on first push.
- **Plan items:** K-009 ✅ done; parking lot gains "verify the CI workflow goes green on first
  push". K-007 (graph-dba relaunch) is the next action.

## 2026-07-04 — K-006: post-M1 review follow-ups (navigation, bounds, health)

- **What:** small, high-value fixes from a 2026-07-04 full-project review; the review's larger
  findings went to the parking lot. Adapter/boundary changes only — no `QUERIES.md` query bodies
  or schema touched, so the 92-suite stays a pure regression guard.
  1. **MCP navigation dead-end closed** — `list_channels(limit)` + `list_threads(channel_id,
     limit)` MCP tools (7 total). Before, an agent could not discover an existing channel or
     thread id (workspace-wide `read_messages` rows omit `threadId` — still parked); it could
     only create its own space. Thin wrappers over the existing `Services` methods; discovery
     test updated, list→post→read navigation roundtrip added.
  2. **Input size bounds (RAM rule 6)** — `schemas.py` Pydantic constraints (text ≤ 8000,
     name/title 1–200, mentions ≤ 50) and `Query` bounds on list `limit`s (1–200). Message text
     lands in graph RAM *and* the full-text index; nothing capped it.
  3. **REST thread-read pagination** — `GET /threads/{tid}/messages?since=&limit=` maps to the
     existing §9.1 `read_thread_since` as a **pure read** (`since` defaults to 0 explicitly, so
     a browser poll never consults/advances the member's cursor — cursors stay agent-owned).
     No params keeps the full §4 read contract. Mitigates the unbounded `NEXT*0..` walk vs the
     1000 ms default `TIMEOUT` cliff on long threads (full fix = web client adoption, parked).
  4. **`GET /health`** — `services.ping` → `repository.ping` (`RO_QUERY RETURN 1`); 503 when
     FalkorDB is unreachable. Probe target for compose/CI (both parked).
- **Doc drift fixed (root `AGENTS.md`):** query-suite baseline claims corrected 67/67 → **92/92**
  (×2) — the stale numbers were loaded into every agent session.
- **Verified:** server suite **75 passed** (was 70; +5: MCP navigation roundtrip, health, body
  bounds, limit bounds, pagination — the pagination test injects a counting clock to sidestep the
  known same-ms `createdAt` tie caveat); query suite **92/92**.
- **Docs (same change):** `DESIGN.md` §14.4 REST table (+`/health`, real `?since=&limit=` shape,
  bounds note) and §15.2 tools table (+2 rows); `README.md` tools list + counts 70→75;
  `falkor-chat/AGENTS.md` count 68→75 (was already stale); `plan.md` parking lot extended,
  Last-reviewed bumped; this entry.

## 2026-07-02 — K-005: M1-final cleanup

- **What:** four small parking-lot items from the 2026-07-02 review, resolved test-first. All
  server changes are **adapter-only** (`mcp.py`, `api.py`) — no `repository.py`, `services.py`,
  `QUERIES.md`, or `test_queries.sh` touched, so the 92-assertion suite stays a pure regression
  guard.
  1. **`search_messages` MCP tool** — the existing `services.search_messages` (REST `GET /search`,
     `QUERIES.md` §5) is now exposed as a 4th MCP tool so agents can keyword-search too. Thin
     adapter; roundtrip test added.
  2. **`create_channel` MCP tool** (Q#4) — 5th tool; agents can now set up their own space
     (channel → thread → post → read) without any REST seeding. Discovery test asserts all 5
     names; full-flow roundtrip added.
  4. **Flat `GET /messages/{msg_id}` route** — replaced the nested
     `GET /threads/{tid}/messages/{mid}`, which ignored `tid` and let a message resolve under any
     thread's URL (a false contract). `Message.msgId` is workspace-unique and `Message` has no
     `threadId`, so resolution is workspace-global by design; the flat route states that truth.
- **Two fork decisions (spec §0):**
  - **Fork 3(a) — dead `isMention` highlight:** *remove it from the JS* rather than make §4 return
    a per-reader `isMention`. `isMention` is a since-read (§9) concept computed only by
    `read_thread_since`/`read_ws_since` (which take `me_id`); the reader-agnostic §4 thread read
    the web UI uses never sends it, so the highlight was dead-falsy. Making §4 reader-aware would
    mutate the locked §4 query, add a per-reader traversal to the hot thread-read path (RAM rule
    6), and force a 92-suite assertion change — not worth restoring a cosmetic highlight on a
    request/response M1 UI. Revisit in M2 with real-time since-reads.
  - **Fork 4 — nested single-message route:** *drop the thread-scoped spelling* for a flat
    `GET /messages/{mid}`. Validating thread membership would need an O(thread-length) HEAD/NEXT
    traversal on a route the web UI does not use, purely to keep a URL shape; the O(1) fix
    (denormalised `Message.threadId`) is a parked schema change (RAM rule 6). Leaving it as-is
    ships a wrong-thread-resolution trap.
- **Verified:** server suite **70 passed** (was 68; +1 search roundtrip, +1 create_channel flow;
  discovery + 2 api tests edited net 0); query suite **92/92** (untouched — regression guard).
- **Docs (same change):** `DESIGN.md` §15.2 tools table (+2 rows), §14.4 REST surface
  (`/messages/{mid}`), §14 test-count 68→70; `README.md` MCP tools list (+`create_channel`,
  +`search_messages`) and counts 68→70; `plan.md` pruned (4 completed items removed, Last
  reviewed bumped); this entry.
- **Batch B (delivered separately by another implementer):** the two `web/app.js` items —
  removing the dead `isMention` class toggle in `renderMessages`, and making the composer submit
  handler retry a mention-rejected send (`400 UnknownMemberError`) as plain text with a
  non-blocking notice so a typo'd `@handle` no longer drops the whole message. No test harness for
  the web JS; verified manually.

## 2026-07-02 — K-004: M1 hardening — five live-verified defects + QA DEF-1 fixed

- **What:** a full-project review probed the M1 server live (isolated `ws:probe` graph) and
  confirmed five defects the 57-test suite missed — every failing scenario involved state the
  fixtures always seeded (the actor) or parameter combinations never tested (`limit` + cursor).
  All fixed TDD (11 red tests → green):
  1. **Silent no-op writes (worst).** The §4 write queries anchor on `MATCH (author {userId:…})`;
     with the author node absent the whole write no-ops and REST still returned **201 with a fresh
     `msgId`** — on a fresh tenant (nothing ensures `u1`) every send "succeeded" and every thread
     stayed empty. Fix at three layers: `repository._assert_written` raises on zero-row writes;
     `services.post_message` validates the actor resolves to a member (`UnknownActorError`, one
     shared membership lookup with mentions); `create_app`'s lifespan runs `services.ensure_actor()`
     (startup, not import — building the app still needs no live FalkorDB).
  2. **Cursor-vs-limit message loss.** `read_messages` advanced the cursor to the *server clock*,
     permanently skipping rows a `limit` truncated (probe: 5 posted, `limit=2` read → next read 0).
     Fix: since-reads (§9.1/§9.2) are now **chronological** — the truncated page is a contiguous
     prefix — with reader-mentions carried by the `isMention` flag instead of the old
     mention-first sort (which + `LIMIT` is what made pagination lossy); the cursor advances to the
     newest **delivered** `createdAt` (empty page → no write). Ordering change synced in
     `QUERIES.md` §9 (+ rationale note), `test_queries.sh` (1:1 assertion swap), DESIGN §15.2.
  3. **`advance_cursor` IndexError** when the member node didn't exist (empty result indexed) —
     now a no-op returning `None`; noted in QUERIES.md §9.3.
  4. **QA DEF-1 (from the 2026-07-01 report) closed.** `POST /mcp` 405'd (Starlette Mount serves
     only `/mcp/`) — `create_app` adds an ASGI path-alias middleware rewriting `/mcp` → `/mcp/`;
     regression pinned by tightening the existing app test (it had tolerated 405 via `< 500`).
  5. **Search syntax-error 500.** RediSearch parse errors (`q='hello"x'`) surfaced as unhandled
     500s — `services.search_messages` maps `ResponseError` → `InvalidSearchQueryError` → 400.
  - Also: removed a duplicated gotcha comment in `repository.thread_has_head`; fixed the stale
    `exists((t)-[:HEAD]->())` advice in QUERIES.md §4 (contradicted the AGENTS.md live gotcha).
- **Verified:** server suite **68 passed** (was 57; +11); query suite **92/92** (assertion count
  unchanged — ordering assertions swapped 1:1); live probe script re-run: all five defects gone.
- **Docs (same change):** `QUERIES.md` §4 zero-rows + HEAD-check notes, §9 ordering rationale,
  §9.3 no-member note; `AGENTS.md` write-path invariants (+ zero-rows, chronological-cursor
  bullets) and test count; `README.md` counts + `/mcp` slash note; `DESIGN.md` §12/§15.
- **Plan items:** K-004 ✅. Review findings **not** fixed here parked in `plan.md` (agent
  authorship, `threadId` in §9.2 rows, retry idempotency + first-post race, web-UI mention
  polish, nested-route validation, ms-tie ordering, dependency pins, lint/CI).

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
