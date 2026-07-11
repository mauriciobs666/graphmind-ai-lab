# Test Plan — K-007 M2 Groundwork (agent authorship · v2 write guards · threadId · composite cursors)

> **Author:** qa-engineer · **Date:** 2026-07-05
> **Under test:** `falkor-chat/` at commit `94ab746` (K-007) — the v2 write paths, agent
> authorship/role derivation, `Message.threadId` denorm + backfill script, and tie-safe
> composite cursors, exercised through the M1 server (REST + MCP) and the ops scripts.
> **Companion report:** [`../test-reports/k007-m2-groundwork-report.md`](../test-reports/k007-m2-groundwork-report.md)
> **Prior passes (build on, do not duplicate):** [`m1-chat-mcp.md`](m1-chat-mcp.md) ·
> [`m1-hardening-regression.md`](m1-hardening-regression.md) (that report's §4 named
> concurrency/idempotency, agent authorship, and ms-ties as the top residual risks — K-007 is
> the fix; this pass closes the loop at the acceptance altitude).

## 1. Scope & objective

K-007 landed with heavy developer-level coverage (server pytest 98, query suite 115/115,
including repo-level replay/race/tie regressions and an in-process 8-worker hammer — see
`docs/archive/plans/m2-groundwork.md` §7). **This pass does not re-run that layer.** It verifies, from
the outside in, what those tests structurally cannot:

1. **Concurrency through the real HTTP stack** — parallel REST posts racing a first-post on a
   live uvicorn, and **two separate server processes** writing to one workspace (the actual M2
   multi-writer shape; the dev hammer was single-process, service-level).
2. **Cursor losslessness driven black-box** — MCP cursor paging across a seeded same-ms tie at
   a page boundary; explicit `?since=` plain-`>` semantics; legacy (pre-K-007) cursor shape
   tolerated end-to-end.
3. **Agent authorship as observable behavior** — an Agent-authored message read back through
   every read surface (REST thread read, `?since=`, `/messages/{id}`, `/search`, MCP), plus
   the *seam boundaries*: what the M1 configured-actor front doors can and cannot express, and
   the User/Agent id-collision edge nothing guards.
4. **`threadId` cross-surface consistency** — correct values with multiple threads, the
   documented shape asymmetries, null tolerance for legacy rows, and the
   `backfill_thread_ids.sh` **script** run against real legacy-shaped data (to date it has only
   run against an empty `ws:acme`; the suite tests the raw query, not the CLI).
5. **Error/idempotency contracts of the rewritten write path** — 404/400 mapping survived the
   v2 rewrite; documented non-idempotency of client-level retries (creates and posts).
6. **K-009 adjacency smoke** — `/health` gating and boot behavior against an unreachable
   FalkorDB.

## 2. References (sources of truth)

- `git show 94ab746` — the K-007 diff; `docs/HISTORY.md` 2026-07-05 entry.
- `docs/archive/plans/m2-groundwork.md` (§1 decisions, §7 dev-test enumeration = the do-not-duplicate
  list) · `docs/archive/plans/m2-groundwork-queries.md` (defect evidence + status-row contract table).
- `AGENTS.md` — "Message write paths" invariants (status-row contract, role derivation,
  since-read/cursor contract) + live-verified facts.
- `docs/QUERIES.md` §2/§3/§4(+§4.x)/§5/§9 v2 · `docs/DESIGN.md` §5.1/§9/§10.
- Code: `server/falkorchat/{config,app,api,services,repository,mcp}.py`;
  `scripts/{bootstrap_schema,backfill_thread_ids,test_queries}.sh`.

## 3. Risk assessment (drives prioritization)

| Area | Risk | Why |
|---|---|---|
| First-post race via real HTTP / two processes | **High** | The defect K-007 exists for (M2 = concurrent agent writers). Dev hammer was one process, one `Services`; real deployments are N processes with independent monotonic clocks and real threadpools. A fork here silently splits a thread's timeline. |
| Cursor loss across ms-ties, driven via MCP | **High** | Silent permanent message loss for a catching-up agent — the worst prior defect class (HR-004 lineage). Dev covered the repo layer; the service→MCP cursor wiring (advance-to-last-returned-pair) is new code exercised here over a real tie. |
| Backfill script on real legacy data | **Med-High** | Ops script shipped having only ever run against 0 messages. A wrong backfill mis-labels navigation metadata on every pre-K-007 row; legacy `threadId: null` rows must also not break reads/search meanwhile. |
| Agent role via every read surface + id collision | **Med** | `role` is derived server-side and flows to clients through five differently-shaped read paths. Nothing enforces User/Agent id uniqueness across labels, and `coalesce(u, a)` silently prefers User — an ambiguity worth probing before K-008 wires real agents. |
| Error contracts through the rewritten path | **Med** | `post_message` was rewritten wholesale (dispatch loop); the 404/400 mapping and nothing-written guarantees must have survived. |
| Client-retry non-idempotency (documented) | **Low** | Documented contract (server-minted ids); confirm and date it so the docs stay honest. |
| `/health` gating / boot without DB | **Low** | K-009 adjacency; smoke only. |

## 4. Environment & data setup

- FalkorDB `falkordb-dev` (`falkordb/falkordb:edge`) on `:6379` — pre-existing, **never
  restarted**; `ws:acme` (shared dev, schema-only) and `reference` **untouched**.
- **Isolated QA workspace `ws:qa`**: `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh qa`.
  All writes land here. Deleted (`GRAPH.DELETE ws:qa`) at the end, along with any other
  `ws:qa*` graph this pass creates.
- Servers (from `server/`, venv `.venv`):
  - **S1** `:8010` — `FALKORCHAT_WS_ID=qa` (actor defaults `u1`) — main front door.
  - **S2** `:8011` — `FALKORCHAT_WS_ID=qa FALKORCHAT_USER_ID=u2` — second *process*, second
    actor, same workspace (cross-process items).
  - **S3** `:8012` — `FALKORCHAT_WS_ID=qa FALKORCHAT_USER_ID=qabot` where `qabot` is
    pre-seeded as an **Agent** (collision item K7-002), started only for that item.
  - **S4** `:8013` — `FALKORDB_PORT=6399` (dead port) for the boot/health smoke, only for
    K7-050.
- Members seeded via the repo's idempotent surface (permitted): `ensure_user(u2)`,
  `ensure_agent(a1)`, `ensure_agent(qabot)` in `ws:qa`.
- Drivers: `curl` for REST; the real `mcp` python client (`streamablehttp_client` +
  `ClientSession`) at `http://127.0.0.1:8010/mcp/`; `server/.venv/bin/python` scripts using
  `falkorchat.repository`/`services` for seeding + tie fixtures (forced `created_at`) and for
  the service-seam agent items; `redis-cli GRAPH.QUERY ws:qa …` for graph-state evidence
  (HEAD/TAIL/NEXT counts) — read-only inspection except the two deliberate legacy fixtures
  (K7-021 legacy message chain, K7-032 legacy cursor), which are written by this pass into
  its own graph.

## 5. Entry / exit criteria

- **Entry:** FalkorDB answers (`redis-cli ping` → PONG); both baselines green
  (server `pytest -q` → 98; `./scripts/test_queries.sh` → 115/115); `ws:qa` bootstrapped;
  S1 serving (`GET /health` → 200).
- **Exit:** every item has a recorded outcome with real evidence; defects reproducible with
  severity by user impact; verdict stated; QA graphs deleted; `ws:acme`/`ws:test`/`reference`
  and the FalkorDB container intact.

## 6. Test items

### Baseline
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| K7-000 | Server suite baseline | `cd server && .venv/bin/python -m pytest -q` | **98 passed** | High | regression |
| K7-000b | Query suite baseline | `./scripts/test_queries.sh` | **115/115 passed** | High | regression |

### Agent authorship & the actor seam
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| K7-001 | Agent-authored message visible correctly on **every** read surface | Seed Agent `a1`; post via `Services` + `CallContext(ws="qa", actor="a1")` (the highest agent-reachable seam pre-K-008) into a thread that also has a `u1` REST post; read back via REST full-thread, REST `?since=0`, `GET /messages/{id}`, `GET /search`, MCP `read_messages` | agent msg: `role="assistant"`, `authorId="a1"`, `authorType=["Agent"]` where the shape carries them; `u1` msg `role="user"`; one coherent timeline on all surfaces | High | integration |
| K7-002 | User/Agent id collision — configured actor shadows an existing Agent | Seed Agent `qabot`; start S3 (`FALKORCHAT_USER_ID=qabot`); `POST /threads/{t}/messages`; inspect role + graph | **Exploratory — record actual.** Expected from code: lifespan projects a `User {userId:"qabot"}` alongside the Agent; `coalesce(u,a)` prefers User ⇒ `role="user"`; two nodes share one member id with no guard | Med | exploratory |
| K7-003 | Unknown actor → clean refusal, nothing written | `Services` + `CallContext(actor="ghost")` post to an existing thread; catch error; REST read-back | `UnknownActorError`; thread message count unchanged (no partial write) | Med | functional |
| K7-004 | Mentioning an Agent works end-to-end | REST `POST … {text:"ping @a1", mentions:["a1"]}` → 201; read the thread as `a1` (service seam, cursor mode) | 201, `mentions:["a1"]` echoed; `a1`'s read flags that row `isMention=true` in place | Med | functional |

### v2 write guards from the outside (race + retry)
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| K7-010 | REST first-post hammer — one process | Fresh thread; **12 parallel** `POST /threads/{t}/messages` against S1; then graph inspection + REST read | 12× **201**, 0× 5xx; graph: `HEAD=1, TAIL=1, NEXT=11`, chain walk = 12, `POSTED_BY` = 12, no self-loops; REST read returns all 12 chronologically | High | concurrency |
| K7-011 | Cross-process writers — two servers, one workspace | Fresh thread; 10 parallel posts to S1 (`u1`) **+** 10 to S2 (`u2`) simultaneously; graph inspection; then MCP cursor paging (`limit=3`) as `u1` until empty | 20× 201; `HEAD=1, TAIL=1`, contiguous chain of 20, both authors present; cursor pages deliver **all 20 exactly once** (no skip, no dup — union == set) | High | concurrency |
| K7-012 | Client-level POST retry is a **new** message (documented) | Same-body `POST …/messages` twice | two distinct `msgId`s, 2 messages in thread — server-minted ids make client retries non-idempotent by design (repo-level same-`msgId` replay = dev-covered `dupMsg`); dated confirmation | Low | contract |
| K7-013 | Channel/thread creates non-idempotent (documented) | `POST /channels {name:"dup"}` twice; `POST …/threads {title:"dup"}` twice | distinct `channelId`s / `threadId`s each time — matches QUERIES.md §3 note | Low | contract |

### `threadId` denorm + backfill
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| K7-020 | `threadId` cross-surface consistency (2 threads) | Post into two threads; MCP room-wide `read_messages()` (no `re`); REST `?since=0`; `/search`; `GET /messages/{id}`; REST full-thread read | room-wide/`since`/search/point rows each carry the **correct** `threadId` per row; full-thread read rows carry **no** `threadId` (documented asymmetry — caller supplied it); record shapes | High | contract |
| K7-021 | Legacy rows tolerated + `backfill_thread_ids.sh` on real data | Inject a 2-message legacy chain (no `threadId` property) into a fresh `ws:qa` thread via raw Cypher; read via `?since=0` + `/search` (expect `threadId: null`, no error); run `./scripts/backfill_thread_ids.sh qa` twice | pre-backfill reads work with `threadId: null`; run 1 backfills **2**; run 2 → **0** (idempotent); rows then carry the right `threadId` on all surfaces | High | integration/ops |

### Millisecond ties & cursors, black-box
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| K7-030 | Same-ms tie at a page boundary — MCP cursor paging is lossless | Seed 4 messages at `createdAt` 1000/2000/**2000**/3000 (repo seam, forced clock) in a fresh thread; MCP `read_messages(re, limit=2)` repeatedly as `u1` | pages `[m1,mA] [mB,m4] []` — **all 4 delivered exactly once**; the tied sibling at the boundary is not skipped (the K-007 defect-4 fixture, driven through the full MCP→service→repo stack) | High | functional |
| K7-031 | Explicit `?since=` keeps plain-`>` semantics; cursor untouched | On the K7-030 thread: REST `?since=2000`; then MCP cursor read | `?since=2000` returns only `createdAt > 2000` (both tied rows excluded — documented OQ3); the explicit read did not move the cursor | Med | contract |
| K7-032 | Pre-K-007 cursor (no `lastReadMsgId`) tolerated end-to-end | Fresh thread with the tie fixture; create `ReadCursor {lastReadAt:2000}` (no msgId property) for `u1` via raw Cypher; MCP cursor read | no error; returns the rows at `createdAt = 2000` (msgId > `''` — legacy ties re-delivered, never lost) + everything after; cursor then advances to the composite pair | Med | compatibility |
| K7-033 | Monotonic clock observable at the surface | Burst 10 sequential rapid REST posts to one thread | 10 strictly increasing `createdAt` values (no ties from one process) | Low | functional |

### Cross-door parity & error contracts
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| K7-040 | REST ↔ MCP parity on the K-007 fields | One thread: post via REST and via MCP `send_message`; read via both doors | both doors show one chronological timeline; same `msgId`/`threadId`/`role` values door-to-door | Med | integration |
| K7-041 | Error mapping survived the v2 rewrite | `POST /threads/nope/messages` → ?; `POST` with `mentions:["nobody"]` → ? + read-back; `GET /search?q=hello"unbalanced` → ? | 404 `ThreadNotFoundError`; 400 `UnknownMemberError` **and nothing written**; 400 `InvalidSearchQueryError` | Med | contract |

### K-009 adjacency smoke
| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| K7-050 | `/health` gates on FalkorDB; boot behavior without DB | `GET /health` on S1; start S4 against dead `:6399`, observe | S1 → 200 `{"status":"ok"}`; S4: **exploratory — record actual** (lifespan `ensure_actor` needs the DB at boot; expected fail-fast at startup rather than a serving-but-503 process) | Low | smoke/exploratory |

## 7. Out of scope (this spin)

- Everything in `m2-groundwork.md` §7's dev-test enumeration at its own altitude (repo-level
  status-row permutations, same-`msgId` replay mechanics, in-process hammer, FakeRepo dispatch
  unit tests) — re-verified only as green baselines.
- **Server-side msgId replay via REST/MCP** — not expressible black-box (ids are server-minted);
  covered at the repo layer; K7-012 records the client-visible consequence instead.
- TIMEOUT posture and RAM-costing claims (K-007 items 5–6) — doc/ops findings from live probes
  already run by graph-dba; no server read surface takes a timeout override yet; re-probing
  writes-ignore-TIMEOUT would mean multi-second load against the shared instance. Not re-verified.
- `REPLY_TO` — no API surface (repository fold-in deferred by OQ4); covered in `test_queries.sh`.
- GraphRAG/vector retrieval (K-008), auth/tenancy, real browser DOM, performance/load, Docker
  image/compose lifecycle (K-009's own pass), M3+ features.
