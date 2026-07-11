# Test Report — K-007 M2 Groundwork (agent authorship · v2 write guards · threadId · composite cursors)

> **Author:** qa-engineer · **Date:** 2026-07-05
> **Plan:** [`../test-plans/k007-m2-groundwork.md`](../test-plans/k007-m2-groundwork.md)
> **Build under test:** commit `94ab746` (K-007) · FalkorDB `falkordb-dev` @ `:6379` (up 15h,
> never restarted) · isolated workspace `ws:qa` (bootstrapped `EMBEDDING_DIM=1024`, deleted after)
> · servers S1 `:8010` (`ws:qa`, actor `u1`), S2 `:8011` (`ws:qa`, actor `u2`), S3 `:8012`
> (collision item), S4 `:8013` (dead-DB smoke).
> **Prior pass:** [`m1-hardening-regression-report.md`](m1-hardening-regression-report.md) — its
> §4 residual-risk list (concurrency/idempotency, agent authorship, ms-ties) is exactly what
> K-007 fixed; this pass closes that loop.

## 1. Summary

**Verdict: PASS with two low-severity defects (neither in the K-007 core claims).**

All six K-007 behavior claims hold from the outside in, on green baselines (server **98/98**,
query suite **115/115**):

- **First-post race**: 12 parallel REST posts on one process and **20 posts racing from two
  separate server processes** all returned 201 and produced exactly one HEAD/TAIL and one
  contiguous chain — no forks, no self-loops, no lost writes.
- **Tie-safe cursors**: the cross-process run produced a **genuine same-millisecond
  `createdAt` tie across writers** (the exact M2 hazard), and MCP cursor paging (`limit=3`)
  still delivered all 20 messages exactly once. The deterministic tie fixture (page boundary
  splitting a 2000/2000 pair) was also lossless through the full MCP→service→repo stack.
- **Agent authorship**: an Agent-authored message reads back as `role: "assistant"`,
  `authorType: ["Agent"]` consistently on all five read surfaces; unknown actor is refused
  with nothing written; mentioning an Agent flags `isMention` for the agent reader in place.
- **`threadId`**: correct per-row across 6 threads in room-wide/`since`/search/point reads;
  legacy rows read back as `threadId: null` without error, and `backfill_thread_ids.sh` (first
  ever run against real legacy-shaped data) backfilled 2, then 0 on re-run.
- **Documented contracts confirmed dated**: explicit `?since=` plain-`>` (tied rows excluded,
  cursor untouched); pre-K-007 cursor upgraded in place (ties re-delivered, never lost);
  client-level retries of posts/creates mint new ids (non-idempotent by design); 404/400 error
  mapping survived the write-path rewrite.

The two defects are at the edges, not in the shipped behavior: **DEF-1** — nothing guards the
User/Agent member-id namespace, and a configured actor colliding with an existing Agent id
silently shadows the Agent everywhere (latent hazard for K-008); **DEF-2** — with FalkorDB
unreachable the server hangs indefinitely at import with zero output (eager connect in
`db.connect()`), contradicting `app.py`'s stated build-without-DB intent.

## 2. Results

| ID | Item | Result | Evidence |
|---|---|---|---|
| K7-000 | Server suite baseline | ✅ PASS | `pytest -q` → **98 passed** in 1.76s |
| K7-000b | Query suite baseline | ✅ PASS | `./scripts/test_queries.sh` → **115/115 passed** |
| K7-001 | Agent-authored msg on every read surface | ✅ PASS | `a1` (seeded `ensure_agent`) posted via service seam → payload `role:"assistant"`. Read back: REST full-thread + `?since=0` + `GET /messages/{id}` + MCP `read_messages` all show `role:"assistant"`, `authorId:"a1"`, `authorType:["Agent"]`; `/search` finds it with correct `threadId` (search shape carries no `role` — see §5.4). `u1`'s sibling msg `role:"user"`. One coherent timeline everywhere |
| K7-002 | User/Agent id collision (configured actor = existing Agent id) | ⚠️ **DEF-1** | S3 booted with `FALKORCHAT_USER_ID=qabot` (`qabot` pre-seeded as Agent): lifespan silently created `User {userId:"qabot"}` **alongside** the Agent (both exist, verified); post → `role:"user"`, `POSTED_BY→[User]`; mention `"qabot"` resolved to the shadow **User**. The Agent is shadowed in every `coalesce(u,a)` lookup |
| K7-003 | Unknown actor → clean refusal, nothing written | ✅ PASS | `CallContext(actor="ghost")` post → `UnknownActorError: ghost`; thread count unchanged (2 before/after) |
| K7-004 | Mentioning an Agent end-to-end | ✅ PASS | REST `mentions:["a1"]` → 201, echoed; `a1`'s cursor read flags exactly that row `isMention=true`, in place |
| K7-010 | REST first-post hammer (one process) | ✅ PASS | 12 parallel POSTs → **12× 201**, 12 distinct msgIds; graph: `HEAD=1, TAIL=1, NEXT=11, self-loops=0`, chain walk **12**, `POSTED_BY=12`; REST read: 12 rows, strictly increasing `createdAt` |
| K7-011 | Cross-process writers (S1+S2, one workspace) | ✅ PASS | 10+10 simultaneous posts from two uvicorn processes → **20× 201**, authors `u1`+`u2`, **1 natural cross-process `createdAt` tie**; `HEAD=1, TAIL=1`, contiguous chain of 20, 0 self-loops. MCP cursor paging `limit=3` → 7 pages `[3,3,3,3,3,3,2]`, **20 delivered / 20 unique**, globally ordered by `(createdAt, msgId)` incl. the tie, drain read → 0 rows |
| K7-012 | Client POST retry = new message (documented) | ✅ PASS | same-body POST ×2 → distinct msgIds `849064…`/`828760…`, 2 messages. Matches the documented contract (server-minted ids; same-`msgId` replay = repo-level `dupMsg`, dev-covered) |
| K7-013 | Creates non-idempotent (documented) | ✅ PASS | `POST /channels` ×2 → distinct channelIds; `POST …/threads` ×2 → distinct threadIds (QUERIES.md §3 note holds) |
| K7-020 | `threadId` cross-surface consistency | ✅ PASS | room-wide MCP read: 49 rows / 6 threads, **0 missing threadIds**, hammer rows all one threadId, xp rows all another; `?since=`/`/search`/point reads correct (K7-001/021 evidence). Full-thread read shape carries **no** `threadId` (documented asymmetry — caller supplied it) |
| K7-021 | Legacy rows + backfill script on real data | ✅ PASS | injected 2-message legacy chain (no `threadId`): `?since=0` + `/search` returned `threadId: null`, no error; `./scripts/backfill_thread_ids.sh qa` run 1 → **backfilled 2**, run 2 → **0**; rows then carry the correct threadId on both surfaces |
| K7-030 | Same-ms tie at page boundary — MCP cursor lossless | ✅ PASS | fixture 1000/2000/2000/3000; `read_messages(re, limit=2)` → pages `[(m1,1000),(ma,2000)] [(mb,2000),(m4,3000)] []` — the tied sibling straddles the boundary and **all 4 delivered exactly once** (the reproduced K-007 defect-4 shape, now through the full MCP stack) |
| K7-031 | Explicit `?since=` plain-`>`; cursor untouched | ✅ PASS | `?since=2000` → only `(m4, 3000)` (both tied rows excluded — OQ3 as documented); cursor `(3000,'…tie-m4')` identical before/after |
| K7-032 | Pre-K-007 cursor tolerated end-to-end | ✅ PASS | raw `ReadCursor {lastReadAt:2000}` (no msgId) → MCP read: no error, returns `(ma,2000),(mb,2000),(m4,3000)` — legacy ties **re-delivered, never lost**; cursor then upgraded to composite `(3000,'…tie-m4')` |
| K7-033 | Monotonic clock observable | ✅ PASS | 10 rapid sequential posts → strictly increasing `createdAt`, 0 ties |
| K7-040 | REST ↔ MCP parity on K-007 fields | ✅ PASS | one thread, one post per door → both doors read the identical timeline: same msgIds, `role`, `threadId`, order |
| K7-041 | Error mapping survived the v2 rewrite | ✅ PASS | POST to `nope` → **404** `ThreadNotFoundError`; unknown mention → **400** `UnknownMemberError` **and nothing written** (count unchanged); `q=hello"unbalanced` → **400** `InvalidSearchQueryError` |
| K7-050 | `/health` gates; boot without DB | ⚠️ PASS + **DEF-2** | live S1: `GET /health` → 200 `{"status":"ok"}`. S4 vs dead `:6399`: **hangs indefinitely** (≥90s) — no port bound, no log line, no error (see DEF-2) |

**18/18 plan items executed · 16 clean passes · 2 findings filed as defects · baselines 98/98 + 115/115.**

## 3. Defects

1. **DEF-1 — member-id namespace is not unique across `User`/`Agent`; a shadow User silently
   eclipses an Agent.** *Severity: Low today — High-leverage hazard for K-008.*
   **Repro:** `repo.ensure_agent("qa", agent_id="qabot")`; start the server with
   `FALKORCHAT_WS_ID=qa FALKORCHAT_USER_ID=qabot`; `POST /threads/{t}/messages`.
   **Actual:** startup `ensure_actor` MERGEs `User {userId:"qabot"}` next to the Agent (both
   nodes verified present); the post is attributed `role:"user"` with `POSTED_BY→User`, and a
   mention of `"qabot"` resolves to the shadow User — every `coalesce(u, a)` lookup (role
   derivation, author resolution, mentions, cursors) silently prefers the User. **Expected:**
   either a cross-label uniqueness guard (refuse to `ensure_user` an id that exists as an
   Agent, and vice versa) or an explicit documented precedence. Nothing in the schema or
   service prevents the collision, and the failure is silent misattribution — exactly the class
   K-007 was closing. **Recommend** fixing before K-008 wires real agent identities
   (`ensure_*` pre-check, or a locked "ids are namespace-unique across member labels" rule +
   validation). Defer implementation to coder/tdd-engineer.
2. **DEF-2 — server start hangs silently forever when FalkorDB is unreachable.**
   *Severity: Low (ops/diagnosability; no user data at risk).*
   **Repro:** `FALKORDB_PORT=6399 .venv/bin/uvicorn falkorchat.app:app --port 8013` (nothing
   listening on 6399). **Actual:** ≥90s with zero output — uvicorn never prints a line, the
   port never binds, `/health` never answers; isolated to `db.connect()`: the `FalkorDB()`
   constructor issues an eager command (`Is_Sentinel` check, falkordb-py 1.6.1) with **no
   socket/connect timeout**, and `app.py`'s module-level `app = create_app()` calls it at
   import. **Expected:** fail fast with a clear error (and ideally a serving `/health`→503).
   This also falsifies the `app.py` docstring intent "building the app never requires a
   reachable FalkorDB" — importing the module *is* building the app, and it requires one.
   **Caveat:** the indefinite hang (vs. instant `ECONNREFUSED`) is environment-flavored
   (WSL2 closed-port blackhole); the eager import-time connection is real everywhere.
   **Recommend:** pass `socket_connect_timeout` (and `socket_timeout`) in `db.connect()`
   and/or defer the connection to lifespan with an explicit startup error. Compose deployments
   are shielded by `depends_on: service_healthy`; the README dev path (`uvicorn` by hand) is not.

## 4. Coverage & gaps

- **Covered:** all six K-007 claims at the acceptance altitude — race via real HTTP (single- and
  **multi-process**, the M2 shape), cursor losslessness over both a natural cross-process tie
  and a deterministic boundary fixture, agent role on five read surfaces, `threadId` on four
  (+ the documented full-thread asymmetry), legacy-row null tolerance + the backfill script's
  first run against real data, legacy-cursor upgrade, explicit-`since` semantics, monotonic
  clock, cross-door parity, error contracts, non-idempotent creates/retries, `/health` + boot
  behavior. Dev suites re-run green as baselines.
- **Not covered (residual risk):**
  - **Agent writes through a real front door** — REST/MCP cannot express an agent actor in M1
    (`get_context` is the seam; MCP ignores `frm`); agent authorship was driven at the service
    seam. The transport-level agent path arrives with K-008 auth — test it then, together with
    DEF-1's fix.
  - **Server-side same-`msgId` replay black-box** — not expressible (ids server-minted);
    repo-level dev coverage stands; K7-012 records the client-visible contract instead.
  - **TIMEOUT posture & RAM line (K-007 items 5–6)** — doc claims from graph-dba's live probes;
    not re-verified (no server surface takes a timeout override; re-probing means multi-second
    load on the shared instance).
  - `REPLY_TO` (no API surface; suite-covered), GraphRAG/K-008, auth, browser DOM, load/soak,
    Docker image lifecycle (K-009's own concern).

## 5. Feedback & recommendations

1. **K-007 is solid where it matters most.** The concurrency work holds up under the harshest
   test this pass could construct — two independent server processes racing 20 writes produced
   a real cross-process millisecond tie, and the composite cursor delivered every message
   exactly once. That is precisely the M2 multi-agent-writer scenario, verified before any
   agent exists.
2. **Close DEF-1 before K-008.** Role derivation trusts `coalesce(u, a)`; one colliding id
   turns "agents author first-class" into silent misattribution. A cross-label uniqueness
   check in `ensure_user`/`ensure_agent` is small now and painful later.
3. **DEF-2 is a one-line-ish fix** (`socket_connect_timeout` in `db.connect()`) that buys real
   dev/ops diagnosability; consider also moving the first connection into lifespan so import
   never blocks.
4. **Shape asymmetries are now three-way** (prior reports flagged two): full-thread read has
   `displayName`+`authorType` but no `threadId`/`isMention`; since-reads have
   `isMention`+`threadId` but no `displayName`; search has neither `role` nor author fields.
   All documented/intentional individually, but a per-endpoint response schema (already
   recommended twice) would make the contract testable and stop accretion.
5. **The tie fixtures were only constructible via the repo seam** (forced `created_at`). Fine
   for QA, but if K-008 adds any client-influenced timing surface, revisit — the black-box
   tie coverage currently depends on gray-box seeding plus luck (K7-011's natural tie).

## 6. How this run was executed

- **Baselines:** `cd server && .venv/bin/python -m pytest -q` → 98 passed;
  `./scripts/test_queries.sh` → 115/115 (run sequentially — both use `ws:test`).
- **Workspace:** isolated `ws:qa` (`EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh qa`);
  members seeded via `repo.ensure_user("u2")` / `ensure_agent("a1","qabot")`. All fixtures
  (legacy chain, legacy cursor, tie messages) written only to `ws:qa`.
- **Drivers:** `curl` (REST), real `mcp` python client (`streamablehttp_client` +
  `ClientSession`) against `http://127.0.0.1:8010/mcp/`, `httpx` in threadpools for the
  hammers, `server/.venv/bin/python` + `falkorchat.{repository,services}` for seeding and the
  service-seam agent items, `redis-cli GRAPH.QUERY ws:qa` for graph-state evidence.
- **Cleanup (verified):** S1–S4 stopped; `GRAPH.DELETE ws:qa` → `GRAPH.LIST` = `ws:acme`,
  `reference` only; `redis-cli ping` → PONG; container `falkordb-dev` up, never restarted.
  Note: the baseline query suite deletes `reference` in its own teardown (its documented
  behavior); the subsequent `ws:qa` bootstrap recreated the `reference` schema — final state
  matches the documented topology, no QA data in it. `ws:acme` untouched (schema-only, as found).
