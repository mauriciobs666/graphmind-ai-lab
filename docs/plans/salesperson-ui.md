# The one salesperson UI — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (M<n> TBD) · **Version:** 1.20 · **Reviews:** `docs/reviews/salesperson-ui.md`, `docs/reviews/salesperson-ui-impl.md`

*2026-09-02 — v1.1: revised against `docs/reviews/salesperson-ui.md` (4 blockers, 9 majors, 15 minors) and the stakeholder's OQ-1…OQ-6 answers; the client component takes the `salesperson/` name and the retired app moves to `deprecated/salesperson/`.*
*2026-09-02 — v1.2: revised against that review's `## Pass 2` (approve with suggestions) — N1 pins `FALKORCHAT_WS_ID=demo` and adds a non-label survivor clause plus a positive non-participant survivor test, N2 assigns the SPA's shared entry files to S12a, N3 re-keys the route-table assertion onto the `storefront` parameter, plus both nits; `teco`'s `deprecated/` move is recorded as landed.*
*2026-09-02 — v1.3: the def version moves `v6`→`v7` throughout (`v6` is burned — §4.5), and §5.0's shared-file map is corrected against S2's delivered files (`docs/reviews/salesperson-ui-impl.md` F-4).*
*2026-09-02 — v1.4: S8 picks up F-6's stale `schemas.py` reserved-key list, S1's and S4's done-conditions pin their workspace to a throwaway probe (`teco`'s call), and §2.2's `v5` baseline is marked as plan-time rather than live.*
*2026-09-02 — v1.5: the header lists both reviews, §8 says where implementation-gate findings live, and §6.1's re-seed note is marked deliberately unpinned.*
*2026-09-02 — v1.6: §5.1's S4 row names **nine** repository methods — `ensure_participant` was missing and the row was the defect (`docs/reviews/salesperson-ui-impl.md` Pass 3, M-6).*
*2026-09-02 — v1.7: §5.1 adopts a **cite, don't re-list** rule; S7's and S10's quiesce done-conditions now cite `docs/plans/salesperson-ui-graph.md` §7 (a)–(d) instead of restating the wording that note supersedes, and §4.8/R4 are swept to match.*
*2026-09-02 — v1.8: three unabsorbed S0 hand-off mandates land — the post-reset profile-name re-write (S7), the `MAX_QUEUED_QUERIES` assertion (S15) and F8's reset-timeout rule (**misattributed to S12a — F8 is server-side; corrected in v1.10**) — plus the anomaly-response contract in §5.2/S8/S10 and a §5.0 row for `DESIGN.md` (`docs/plans/salesperson-ui-graph.md` §12).*
*2026-09-02 — v1.9: AC-5's presenter view gets an owner — new step **S12d** (roster, reset-everyone control, `incomplete`/`unresolved` rendering), with §5.0, §6.3 #8, §9, §10's AC-5 row and S15's dependencies swept to match (`teco`'s decomposition call).*
*2026-09-02 — v1.10: revised against `docs/reviews/salesperson-ui.md` `## Pass 3` (1 blocker, 3 majors, 8 minors, 2 nits) — F8 moves server-side to S7/S10 (M-1), the `Thread` UNIQUE signal moves to reset-mine (M-2), the presenter roster is trimmed to the delivered projection (M-3), S12b gains AC-5's participant control (M-4), plus every minor and both nits.*
*2026-09-02 — v1.11: revised against that review's `## Pass 4` (approve with suggestions) — S12d's scope column drops the three roster fields M-3 removed (P4-1), §4.8 specifies the `504` path's own re-read failing (P4-2) and records the client-layer no-retry premise with its reversal trigger (P4-4), and S12a's branch keys on **any** `504` rather than on the error string (P4-3).*
*2026-09-02 — v1.12: S12a's `504` re-read is per-path — `GET /shop/api/presenter/participants` after reset-all, not `/state`, whose token reset-all invalidates — and S12d's activity-holding fixture is labelled as the four-key negative control it is (`teco`'s call on two architect findings from the v1.11 pass).*
*2026-09-02 — v1.13: the client-side credential & session contract is consolidated into a new **§5.3** (C1–C8: two credentials + `localStorage` keys, per-credential `401`/`403`, the per-path `504` re-read, `409`, language step, polling), which S12a now cites instead of summarising — closing an undifferentiated `401` handler that broke §4.3's "presenter keeps driving through the reset" on every successful `reset-all`; §6.2 gains the client-tier bullet it never had (`teco`'s call).*
*2026-09-02 — v1.14: revised against `docs/reviews/salesperson-ui.md` `## Pass 5` (approve with suggestions) — §5.3 completes its matrix against §5.2 (C6 splits into C6a/C6b, new C9 for the quiesce `503`, plus a completeness table making the omission checkable), C8 gains a named test and the single-timer rule, `localStorage` is re-grounded on restart/cross-tab rather than XSS, S8's unreachable second `503` source is struck, and §6 covers the whole SPA track — plus C2 splitting the two meanings of a presenter `403`, which the new completeness table surfaced.*
*2026-09-02 — v1.15: revised against `docs/reviews/salesperson-ui.md` `## Pass 6` (approve with suggestions) — §5.3's completeness table is re-keyed on **(route, response)** so a row cannot span two routes (P6-1), which surfaced a fifth instance and three new rules: **C10** (`/order/advance`'s `404`/`409` are stale-button outcomes, not logouts), **C11** (`422`, now stated in §5.2 too — P6-2), **C12** (`5xx` never auto-retried, the browser being the layer nothing else covered); plus C1's route→credential clause (P6-3), C2 quoting S10's rate-limiter (P6-4) and the cross-tab discriminator re-labelled usability (P6-5).*
*2026-09-02 — v1.16: revised against `docs/reviews/salesperson-ui.md` `## Pass 7` (approve with suggestions) — the section's organising idea moves from enumeration to **two totality guards**: S8's error map becomes **total by type** (P7-3 — the unmapped query-time `TimeoutError` that answered a bare `500` on nine routes) with a decidable gate, and new **C13** makes any unruled `(route, response)` fail loudly on the client; C11 re-keys onto the error body's `field` (P7-1), C12 re-keys onto the transport axis and is verified against the pinned lock (P7-2, P7-4), S10's attempt counter is decided as observational (P7-5), the table becomes the source of truth with a stated generation rule (P7-6), S12c seeds the locale chooser from `/health`, and `docs/SERVER.md` gets its §5.0 row.*
*2026-09-02 — v1.17: revised against `docs/reviews/salesperson-ui.md` `## Pass 8` (approve with suggestions; **the plan gates stop here — no Pass 9**, review resumes at S8's and S12a's implementation gates) — §5.3 gains the **route-class table** S8's gate is computed over (P8-4), the cross-cutting `504` splits into one row per writing route each carrying its own re-read (P8-1/P8-2/P8-3), the grouping licence tightens to *one meaning **and** one action* (P8-3, C9 split by P8-5), C13 states what it does **not** catch and the mis-ruled residual is scored in the document (P8-6), S8 adds the per-route `responses={…}` static half of its gate (P8-7) and the `field` selection rule (P8-N1), **each of S12a's per-rule tests must enumerate the routes its rule spans** (the mis-ruled half's mechanical guard), join is decided **non-idempotent with the roster artifact accepted** (Pass 8 OQ-1), `GET /shop/api/health` is decided **graph-free** (Pass 8 OQ-2), and every env name is spelled in full — sweeping the presenter key prose to S6's delivered `FALKORCHAT_STOREFRONT_PRESENTER_KEY`; plus the S6 implementation gate's plan-side carry-forwards (`docs/reviews/salesperson-ui-impl.md` `## Pass 6`): S10 must reject an **unset** presenter key via `presenter_configured` and extend the constant-time tripwire to `presenter_login` (S6-2, S6-4), S8 gains two source tripwires, and `SERVER.md` §1.5 is explicitly **out of scope** for every step here.*
*2026-09-03 — v1.18: four corrections the S7 implementation gate proved (`docs/reviews/salesperson-ui-impl.md` `## Pass 7`, Rulings 1–3; **no Pass 9 — the plan's review gates closed at `docs/reviews/salesperson-ui.md` `## Pass 8`**) — §4.8's reset-mine row gains footnote **†** (the `Customer` is re-created name-only by §4.10's re-write, and `deliveryAddress: None` is the operative proof, not the empty `PLACED`/`Cart` subgraph), §4.8's quiesce bullet splits the wait (S7, delivered) from the cancellation (**S9**, in front of the wait, never in place of it — 30 s against a 180 s agent timeout is why), and **S8** takes both the `storefront_dir` forwarding §4.7 has no row for and Ruling 1's `productId` projection with `_catalog_rows`'s second read; §5.0, §6.1/§6.2 and §10's AC-11 row swept to match.*
*2026-09-03 — v1.19: Ruling 1's catalog fix is split out of S8 into its own **S7c** (`S7b` being already spent on the delivered Pass-7-minors unit) (projection + `_catalog_rows` simplification + `QUERIES.md` §15.2, with a done-condition that reddens if either half lands alone), so S8's `{handlers} × {routes}` gate — the place Pass 8's stopping rule sends review — judges one subject; and **S9 decides the per-participant record cache rather than carrying it as an open question: it is removed whole** (`lookup`, `_records`, `cached_ids`, every `_cache_put`/`_cache_drop`), which dissolves the S7 gate's S7-2 instead of fixing it. §5.0, §6.1/§6.2 and §9 swept — including §9's serialization list, which v1.18 left stale (`teco`'s call on both).*
*2026-09-03 — v1.20: three plan defects the S8 implementation gate proved (`docs/reviews/salesperson-ui-impl.md` `## Pass 10`, rulings 3/4/6) — §5.2 specifies the join greeting and its `en` fallback verbatim as built, its reset-all `5xx` is re-keyed onto **any unmapped graph error** (the `Thread` UNIQUE violation belongs to reset-mine, so naming it there was a row with no producer), and §5.3 gains a **`503 demo_not_seeded`** row on join (P10-2: reachable at request time, the preflight being a boot-time check) plus the two rules by which S8's gate reads the table — `field` granularity is proved by execution, and the `5xx` rows sit outside the handler cross product. No step row moves.*

## 1. Goal & scope

Build a business-facing, mobile-first storefront UI for the workflow-engine-backed `salesperson`
agent (falkor-chat M6), in which ~50 audience members each hold their own independent, isolated
conversation (with their own cart, order and profile) after joining with nothing but a display
name, in the language each of them picks. It replaces the retired Streamlit app.
Requirements: `docs/requirements/salesperson-ui.md` (FR-1…FR-11, AC-1…AC-11).

**Out of scope** — binding, per that document's own "Out of scope" section plus what this plan
adds:

- `falkor-chat/web/index.html` and `falkor-chat/web/app.js` — not touched, not restyled, not
  re-pointed. §4.9 does *un-mount* that page (and the legacy REST router, and `/mcp`) in the
  storefront deployment, which is a runtime wiring decision, not an edit to either file: both
  remain exactly as they are and are served unchanged by any non-storefront deployment.
- Real auth/access control (K-016/K-017/K-018). Participant tokens and the presenter key
  designed below are demo-session scoping, not authentication — §4.3 argues that boundary
  explicitly.
- Re-theming the electronics catalog; mid-chat language switching; the old app's
  `diagnostics.py` snapshot tooling.
- Driving the `order-fulfillment@v1` process def from the UI (§4.6 decides against it, with a
  reversal trigger).
- Real-time push (WebSocket/SSE). The storefront polls, like `web/app.js` does; K-018 stays
  deferred.
- The `claude/frontend-engineer/frontend-engineer.md` prompt refresh this work invalidates — it
  routes to `cobb`, not to any step here (`teco` dispatched it as U6; see §5.1 S16's note).

---

## 2. Context & findings

*§2 is the verified baseline the design was derived from, as of the design passes that produced it
(v1.0–v1.2) — **including the CPG freshness stamp below**. Line numbers, figures and the CPG's
contents have all moved with S1/S2/S4 and are deliberately not retro-fitted; §4 and §5 are the
current surface.*

**CPG: used `cpg_falkorchat` — rebuilt mid-flight and now fresh (`builtAt 2026-09-02T12:38:21Z`,
`sourceCommit 4bb96e1` = `HEAD`, full data-flow layer; `teco` verified four post-staleness symbols
against the working tree and confirmed its `SOURCE_DIRTY: true` stamp is a repo-wide false alarm,
not source drift). The v1.0 plan predated the rebuild and was written from source read directly
plus live FalkorDB verification; `analyst` independently re-verified ~20 of those source claims
and 2 live claims and found no false one, so this revision keeps that evidence rather than
re-deriving it, and uses the CPG only where this revision needed new impact analysis.**

### 2.1 What I verified live (FalkorDB `localhost:6379`, v4.18.11, 2026-09-02)

| Fact | Evidence |
|---|---|
| The catalog is 15 `Product` nodes in `reference` with **only** `productId`, `name`, `nameNormalized`, `category`, `categoryNormalized`, `price`. No image field of any kind. | `GRAPH.RO_QUERY reference "MATCH (p:Product) UNWIND keys(p) AS k RETURN DISTINCT k"` |
| `productId` is a deterministic name slug (`wireless-mouse-pro`, `27-inch-4k-monitor`, `usb-c-hub-7-in-1`, …). All 15 slugs are stable and known at build time. | same query, `RETURN p.productId ORDER BY p.name` |
| **Several independent `Customer` anchors already coexist inside one workspace graph, each with its own orders.** `ws:qa-cart-totals` holds `Customer{customerId:'u1'}` (2 orders: delivered, cancelled) and `Customer{customerId:'qa-tp14-cust'}` (1 order: placed). `qa-tp14-cust` has **no** `User` node — a `Customer` is created purely from `ctx.actor`. | `MATCH (c:Customer) OPTIONAL MATCH (c)-[:PLACED]->(o:Order) RETURN c.customerId, collect(o.orderId+':'+o.status)` |
| A bootstrapped workspace graph carries ~20 indexed label groups plus their uniqueness constraints. | `CALL db.indexes()` on `ws:acme` |
| An empty bootstrapped workspace is **sub-MB** (`GRAPH.MEMORY USAGE` reports 0 MB for `ws:test`/`ws:qa028`; `ws:acme`, with 544 `Entity` + 87 `Chunk` + 52 `Message`, reports 1 MB total). RAM is **not** the discriminator between the tenancy options in §4.3. | `GRAPH.MEMORY USAGE <key>` for `ws:test`, `ws:qa028`, `ws:acme`, `reference` |

Everything else below is read from source, not executed.

### 2.2 The platform seams the storefront lands on

- **`falkor-chat/server/falkorchat/config.py:CallContext` / `get_context()`** is the single
  auth/tenancy seam (`falkor-chat/docs/SERVER.md` §1.3). Today it returns a **process-constant**
  `CallContext(ws=WS_ID, actor=USER_ID)` from two env vars — *every* REST and MCP call resolves to
  the same actor. Per-participant identity is therefore the one thing the platform genuinely does
  not have; it is what this plan adds. **It is also the reason §4.9 exists**: any surface that
  still resolves through this seam is an unauthenticated reader of whatever workspace it points
  at.
- **`ctx.actor` *is* `customerId`.** `services.add_cart_item` / `get_cart` / `remove_cart_item` /
  `clear_cart` / `place_order` / `get_order_status` / `advance_order` / `get_profile` /
  `save_profile` (`falkor-chat/server/falkorchat/services.py:2653-2870`) all key on `ctx.actor`,
  stated explicitly in the §16 section comment at `services.py:2610`. Distinct actor ⇒ disjoint
  `Customer` / `Cart` / `CartItem` / `Order` / `OrderLine` subgraph, with no shared read path.
  **This is the single most load-bearing finding in the plan** and it is what makes §4.3's
  one-workspace design correct rather than merely convenient.
- **`services.advance_order`** (`services.py:2812`) already implements the guarded-CAS
  `fulfill` / `deliver` / `cancel` lifecycle, returning `None` when the CAS guard doesn't match.
  It has no REST route — confirmed against the full route list in
  `falkor-chat/server/falkorchat/api.py`.
- **There is no REST route for the catalog either.** `lookup_product_fact` / `filter_products`
  exist only as agent tools (`tools.py:385`, `:434`) over `services.lookup_product` /
  `services.filter_products`. FR-2/FR-11 need a read route.
- **There is no repository method that resolves a customer's orders at all.** The complete
  cart/order/profile repository surface is `ensure_customer`, `ensure_cart`, `add_to_cart`,
  `adjust_cart_item`, `read_cart`, `clear_cart`, `place_order`, `get_order` (by `orderId` only,
  `repository.py:2972`), `fulfill_order`, `deliver_order`, `cancel_order`, `upsert_profile`,
  `get_profile`; `services` adds only `get_order_status(ctx, order_id=…)` (`services.py:2795`).
  §4.6's state block and its ownership check both need primitives that do not exist — S4 builds
  them.
- **`repository.ensure_user`** (`repository.py:306`) exists and is a guarded, idempotent
  `User`-projection with a `MemberIdCollisionError` on an id already held by an `Agent`. No
  service or route calls it — `scripts/seed_demo.sh` is the only current producer of `User`
  nodes. There is **no** repository primitive for a `MEMBER_OF` edge (only the seed script's raw
  `redis-cli` `MERGE`).
- **Posting only requires the actor's node to exist**, not a `MEMBER_OF` edge:
  `repository.resolve_member_kinds` (`repository.py:1668`) is two `OPTIONAL MATCH`es on
  `User.userId` / `Agent.agentId`. `services._validate_and_derive_role` (`services.py:813-836`)
  raises `UnknownActorError` / `UnknownMemberError` from that one lookup **before any write** —
  which is why a missing demo `Agent` fails every participant's first message rather than
  degrading (§4.9, B3).
- **`app.py` mounts three surfaces, and `/` is a catch-all mounted last** (`app.py:290-340`):
  `api.build_router(...)` at the root path prefix, `/mcp` (gated by the existing `mount_mcp`
  parameter), then `StaticFiles(web_dir)` at `/`, "since `/` is a catch-all that must sit behind
  the REST routes and the `/mcp` mount". Anything registered after `create_app` returns is
  unreachable — so the `/shop` mount must go **inside** `create_app`.
- **`_lifespan` calls `services.ensure_actor(provider())`** (`app.py:264`), projecting
  `config.USER_ID` as a `User` into `ws:{WS_ID}` at startup. Under §4.3 that node exists in the
  storefront's workspace but is never a participant; §5.2's roster query filters on
  `User.tokenHash IS NOT NULL` so it can never appear in the presenter roster.
- **`_sweep_loop` resolves its own context from `provider()`** (`app.py:160-181`), i.e. from
  `config.get_context()` — so the periodic workflow sweep runs against `ws:{config.WS_ID}` and
  no other workspace. This is an independent reason the storefront's workspace must **be**
  `config.WS_ID` rather than a second variable (§4.9).
- **The `@mention` → agent path.** `api.post_message` (`api.py:92`) schedules
  `background._safe_run_workflow(trigger, …)` on FastAPI `BackgroundTasks` after the write.
  `trigger.WorkflowTrigger.maybe_trigger` applies an ordered rule: loop-guard → resume a
  `waiting` run in this thread → `@mention`-to-start the configured def → **fall through to the
  M2 `AgentResponder`**. **Step 2 matches only status `waiting`** — a run still `running` is
  invisible to it, so a second message during a turn falls through to step 3 and starts a
  *second* run on the same thread. Nothing in the platform guards start-while-running; only the
  `waiting→running` resume CAS is single-flight. §4.4 measure 1a closes this.
- **`salesperson@v5` was the current def at plan time** (`proof_defs.py:301`; S1 has since bumped
  it to `v7` — §4.5). One `agent` step + one terminal
  `decision` step, one never-firing `ctx.endConversation` guard; `config.model =
  "lmstudio/mistralai/ministral-3-3b"`; `maxIterations: 8`; tools `post_message,
  lookup_product_fact, filter_products, view_cart, add_to_cart, remove_from_cart, clear_cart,
  place_order, get_profile, save_profile, query_graph_data`. **`config.tools` / `systemPrompt` /
  `config.model` are create-only per version** — every bump must republish all three cumulatively
  or silently lose them (`proof_defs.py` module docstring; the `v3`/`v4`/`v5` notes).
- **v5's profile guidance is early-conversation, not order-time.** Its third `systemPrompt`
  paragraph reads *"Call `get_profile` once, early in the conversation… Ask for whichever of name
  or delivery address `get_profile` shows as missing — only once per conversation."* There is no
  order-time address confirmation, and `services.place_order` (`services.py:2748`) does not
  require an address. AC-8 asks for exactly that order-time behaviour — §4.10 supplies it.
- **`executor._assemble_messages`** (`executor.py:1243`) builds each LLM turn as: the step's
  `systemPrompt`, then the **thread-scoped** recent turns (`_read_thread_context` →
  `services.read_thread(ctx, thread_id=run_ctx["threadId"])`, `executor.py:1224-1241`), then a
  `CONTEXT:\n<json of run ctx>` user turn. **Anything in the run's `ctx` reaches the model on
  every turn** — that is the FR-3 language carrier (§4.5). `analyst` traced its durability
  further and confirmed it: the chat-path initial ctx is written once at `start_run` and
  `_drive_loop` reloads it from the run node on every resume (`executor.py:604`) without ever
  rewriting it, so `language` survives every turn of a long conversation *and* a re-start after
  a failed run.
- **`services.start_workflow_run`'s chat path ignores `run_ctx`** (`services.py:2023-2033`):
  when `trigger_msg_id` is given the initial ctx is hardcoded to `{"threadId": …}`. Only the
  process path (`trigger_msg_id is None`) honours the caller's `run_ctx`. §4.5 changes this.
- **`DEMO_EXPECTED_DEFS` is two pairs** (`services.py:650`) — `(config.TRIGGER_DEF_KEY,
  config.TRIGGER_DEF_VERSION)` *and* `access-request@v1` — so `GET /workspaces/{ws}/readiness`
  can never be green in a workspace that was never seeded with `seed_workflows.sh`. It is also
  a route on the legacy router, which §4.9 un-mounts. R9 relies on `verify_salesperson.sh`
  instead.

### 2.3 Concurrency facts (the FR-5 ceiling — read from source, not measured)

- **Every route handler in `api.py` is a sync `def`.** Starlette runs those — and sync
  `BackgroundTasks` callables — through `anyio.to_thread.run_sync`, bounded by anyio's **default
  capacity limiter of ~40 worker threads**. This is stated in falkor-chat's own source, at
  `mcp.py:57-63`: *"Starlette's `BackgroundTasks` runs a sync callable via
  `anyio.to_thread.run_sync`, which is bounded by a default capacity limiter (~40 concurrent
  worker threads)."* `analyst` executed the check on the pinned venv (anyio 4.14.1):
  `to_thread.current_default_thread_limiter().total_tokens == 40`, raising it to `100` works, and
  **calling it outside a running event loop raises `anyio.NoEventLoopError`** — it is
  event-loop-scoped, so the raise must happen inside the async lifespan (§4.4 measure 2).
- **Every LLM call is blocking `urllib.request`** (`transport.py` — one POST-JSON transport for
  all providers), with an `agent`/`step` timeout of **180 s** and an `embedding` timeout of 30 s
  (`falkor-chat/config/models.json`). One agent turn is up to `maxIterations: 8` sequential LLM
  round trips, each pinning one thread for its whole duration.
- **Every posted message is additionally embedded** out of band
  (`background._safe_embed`, scheduled from `api.post_message`) — one extra call per message to
  `text-embedding-qwen3-embedding-0.6b` on the same endpoint.
- **The LLM endpoint is one local LM Studio instance** on a 6 GB RTX 4050 with the embedder
  co-resident (`falkor-chat/docs/DESIGN.md` §1.3, "VRAM budget: 6 GB dedicated — embedder + 4B
  LLM co-resident. Do not plan around shared-RAM spill").
- **Graph reads are not the ceiling.** The measured M1 REST append path sustains ~614 msg/s at 16
  concurrent clients (`falkor-chat/docs/test-reports/capacity-report.md:61`); 50 clients polling
  two routes every **2 s** is ~50 req/s of millisecond-scale indexed reads. (v1.0 said 3 s /
  ~33 req/s here and 2 s elsewhere; 2 s is the single figure this plan uses throughout.)

### 2.4 Parity behaviours of the retired app

Citations are the **post-move** paths (`teco` holds the `salesperson/` → `deprecated/salesperson/`
move as U5 until this plan lands; the files are unchanged by the move).

From `deprecated/salesperson/chatbot.py:render_sidebar`, `deprecated/salesperson/cart.py`,
`deprecated/salesperson/customer_profile.py`:

| Old app | What FR-8/9/10 therefore mean |
|---|---|
| Sidebar **Cliente**: `Nome: …`, `Endereço: …` | Profile panel showing name + delivery address, em-dash when unset (FR-10) |
| A green `Pedido confirmado` badge when `is_order_ready()` | Order panel showing the current order and its status (FR-9) |
| Sidebar **Carrinho**: one `{qty}× {item} — {price}` line per item, then `**Total: …**`, or `Carrinho vazio.` | Cart panel with per-line quantity/name/price and a running total; explicit empty state (FR-8) |
| `st.chat_message` transcript, welcome message, `st.spinner("Pensando...")` | Chat transcript with a welcome turn and a visible "thinking" state |
| **The name is known from the start** — collected upfront, shown in the sidebar immediately | The join display name must reach the profile panel at once, not after the agent asks (§4.10) |
| `utils_common.format_currency` (single currency) | Currency formatting is per-locale in the new UI (the catalog is USD-priced) |
| `diagnostics.py` checkbox | Explicitly out of scope |

**Why Streamlit is not the stack (established from the code, per the brief).**
`deprecated/salesperson/session_manager.py:13` declares `_active_session_id: Optional[str] = None`
as a **module-level, process-global mutable**, and `ensure_session_id()` returns it *before* it
ever consults `st.session_state`:

```python
if _active_session_id:
    set_active_session(_active_session_id)
    ensure_session_log_handler(_active_session_id)
    return _active_session_id
```

Streamlit runs every browser session in one process, so the first participant to establish a
session pins that global, and every subsequent `get_cart_snapshot()` / `get_customer_profile()`
call made without an explicit id — which is every call `chatbot.py:render_sidebar` makes —
resolves to **that first participant's** session. `_cart_store`, `_profile_store` and
`_memory_store` are likewise process-global dicts keyed off it. That is a direct FR-4/AC-2
violation at n ≥ 2, in the shipped code, not a framework opinion. Structurally, `chatbot.py`
re-runs the whole script per interaction and calls `generate_response(message)` **synchronously
inside the script run**, so a participant's LLM turn blocks their session's script thread — a
poor shape for FR-5 even after the global is fixed.

---

## 3. Design overview

```
 phone browsers (≈50)                falkor-chat process (one uvicorn, storefront deployment)
┌──────────────────────┐  HTTP(S)  ┌───────────────────────────────────────────────┐
│ salesperson/dist     │◀─ static ─│ app.py  GET /health          (liveness only)   │
│  join · chat · cart  │           │         mount "/shop"        ← StaticFiles     │
│  order · profile     │──REST────▶│         router "/shop/api"   ← authenticated   │
│  catalog · presenter │  +Bearer  │                                                │
└──────────────────────┘  poll 2s  │   ✗ api.build_router   ── NOT mounted (§4.9)   │
                                   │   ✗ StaticFiles at "/" ── NOT mounted (§4.9)   │
                                   │   ✗ /mcp               ── NOT mounted (§4.9)   │
                                   │              │  get_participant() dependency   │
                                   │              ▼                                 │
                                   │ storefront.py  ── participant registry         │
                                   │                ── per-participant turn queue   │
                                   │                ── reset (mine / everyone)      │
                                   │                ── catalog + image manifest     │
                                   │      │                                         │
                                   │      ▼  CallContext(ws=config.WS_ID, actor=pId)│
                                   │ services.py ─▶ repository.py (all Cypher)      │
                                   │ trigger.py ─▶ executor ─▶ salesperson@v7       │
                                   └──────────────────────────┬─────────────────────┘
                                                              ▼
                                             FalkorDB: ws:{config.WS_ID} + reference
                                             LM Studio: one endpoint, bounded queue
```

One process, one origin, no CORS. The browser never sees a raw falkor-chat route and never
supplies a `threadId`, `customerId`, `orderId` or `ws` — every one of those is resolved
server-side from the bearer token. **In the storefront deployment there is no unauthenticated
read path into the workspace at all** (§4.9); `GET /health` is a liveness probe that touches
`services.ping` and returns no participant data.

TLS is a reverse proxy's job — plain uvicorn does not provide it. Under §4.3's key-based
presenter control a TLS-terminating proxy is fully supported (it was not, under the
loopback-binding variant this plan briefly carried; see §4.3). Over plain HTTP on a shared LAN
every participant bearer token is on the wire — R6's stated residual.

---

## 4. Decisions & rationale

### 4.1 Where the code lives — server inside `falkor-chat`, client takes the freed `salesperson/` name

**Settled by the stakeholder at the plan gate (OQ-3), and final.**

- **Server code** — a new, feature-flagged router + service module *inside* falkor-chat:
  `falkor-chat/server/falkorchat/storefront.py` (participant registry, turn queue, reset,
  catalog+images) and `falkor-chat/server/falkorchat/storefront_api.py` (the `/shop/api` router),
  wired in `app.py` behind `FALKORCHAT_STOREFRONT_ENABLED`. **Not** added to `api.py`: keeping it
  a separate router leaves the platform's documented REST surface (`falkor-chat/docs/SERVER.md`
  §1.4) unchanged, makes the demo surface independently mountable and independently testable,
  and — after §4.9 — makes "the legacy router is off" a single, checkable fact about the route
  table.
- **Client code** — the retired Streamlit app moves to **`deprecated/salesperson/`** (a
  history-preserving `git mv`, not a delete), and the new SPA component takes the freed
  **`salesperson/`** name, building to `salesperson/dist/`, which falkor-chat's `app.py` mounts
  by path from `FALKORCHAT_STOREFRONT_DIR`.

**Why the server half cannot live anywhere else.** Everything the storefront needs is
falkor-chat-internal and not remotely reachable in a participant-scoped way: the `Customer`/
`Cart`/`Order`/profile services keyed on `ctx.actor`, the workflow executor, the `ModelGateway`,
the FalkorDB connection, and above all the `get_context` tenancy seam. A separate process would
have to re-implement that seam and then carry identity across a hop that has no authentication
to carry it with — inventing exactly the trust boundary FR-1 says we don't need.

**Why the client half should not live in falkor-chat.** Different discipline, different
toolchain (Node/npm, absent from the dev box by falkor-chat's own note), different release
cadence — and `falkor-chat/docs/requirements/web-api-coverage.md` FR-9 ("coverage grows, visual
weight does not") is a standing commitment about that component's UI that a Tailwind/React
product surface inside `falkor-chat/web*` would quietly break. A separate component keeps that
promise literal, and the seam between the halves is one build-output directory — the same shape
as the existing `web/` mount.

**One correction carried from the review.** v1.0 rejected reusing the `salesperson/` name partly
"on top of a checked-in `.venv`". That is **factually false** and is struck: `git ls-files
salesperson/` returns 25 tracked files, **zero** under `.venv`. The name is free once U5's move
lands, and OQ-3 takes it.

**Consequences.**

- Root `AGENTS.md`: the `salesperson/` bullet is rewritten to describe the new SPA component; a
  `deprecated/` bullet is added for the retired app; the component-docs table row is replaced;
  the "Working in this repo" chatbot-tasks bullet is rewritten. `falkor-chat/`'s bullet gains a
  sentence about the `/shop` surface and the storefront deployment's un-mounted dev surface.
- Docs tree: the feature's document family stays at **repo-root `docs/`** (`teco`'s decision,
  matching where `tico` filed the requirements). `salesperson/` therefore carries only
  `README.md` + `AGENTS.md` and does **not** get its own `docs/` tree; delivery entries go to
  root `docs/HISTORY.md`. It adopts `salesperson/docs/` later, if it acquires topics of its own.

### 4.2 The stack — React 19 + TypeScript + Vite, built to a static bundle

**Recommendation, unchanged from v1.0:** Vite + React 19 + TypeScript, Tailwind CSS v4 for the
mobile-first layout, TanStack Query v5 for the polling/caching layer, `react-i18next` for FR-3,
Vitest + Testing Library for unit tests, Playwright (one mobile-viewport project) for AC-4.

**The decisive property is that it produces a pure static bundle.** The demo runtime stays exactly
one Python process — no Node server, no second port, no CORS, no extra failure mode on demo day —
while the *build* gets a mature responsive-component ecosystem. Node is a build-time dependency
only.

**Rejected, with the trade-off that decided each:**

| Option | Why not |
|---|---|
| **Streamlit** (the retired stack) | §2.4: the shipped code has a process-global session id that cross-wires participants at n ≥ 2, and a blocking LLM call inside the script run. Both are structural to how Streamlit is used here, and fixing them means writing the concurrency layer anyway — at which point Streamlit's rerun model is only a liability for FR-5/FR-6. |
| **HTMX + Jinja templates served by FastAPI** | Genuinely the cheapest option: no Node at all, one process, `hx-trigger="every 2s"` covers polling. Rejected on the product bar — the stakeholder's own stated interaction pattern (minimal icons opening bottom sheets for cart/order/profile detail) and a "modern, pleasant" transcript with optimistic sends are materially better in a component framework. **Kept as a fallback with a decision deadline, not a standing option: S5's done-condition is the last point at which it can be taken.** After S5 ships a working toolchain the fallback lapses; S12a–S14 *are* the client and re-deciding then would discard them. The server surface in §5.2 is identical either way, so the fallback costs only the client steps. |
| **Next.js / SvelteKit** | Their differentiator is SSR, which needs a Node runtime in the demo path. In static-export mode they reduce to option 1 with a smaller ecosystem and one more abstraction. |
| **Vanilla JS like `web/app.js`** | Fails the product bar and duplicates a codebase whose explicit design goal is minimalism. |

The stakeholder's stated "from scratch with the best available frameworks and security"
preference is honoured on the framework axis. On the **security** axis it is deliberately *not*
honoured as stated: FR-1 scopes this to a controlled demo and puts real auth out of scope, so the
plan adds demo-session scoping (§4.3), the structural removal of every unauthenticated read path
(§4.9), and standard hygiene (parameterised Cypher throughout — already the platform rule;
size-bounded request bodies via Pydantic, matching `schemas.py`; `textContent`-only rendering, no
`dangerouslySetInnerHTML`, so agent output can never inject markup; no secrets in the bundle),
and nothing more. Anything beyond that is K-016.

### 4.3 Participant identity & state isolation — one shared workspace, one actor per participant, server-resolved scope

**Recommendation:** one workspace graph, `ws:{config.WS_ID}` (§4.9 explains why it is
`config.WS_ID` itself and not a second `DEMO_WS` variable), bootstrapped once before the demo.
Each participant is:

- one `User {userId: <participantId>, displayName, tokenHash, threadId, channelId, language,
  joinedAt}` — `participantId = "p-" + uuid4().hex`;
- one `Channel` + one `Thread` of their own, both minted at join, with `MEMBER_OF` edges for the
  participant and the demo `Agent`;
- one `Customer`/`Cart`/`Order` subgraph — created eagerly at join by §4.10's profile write, and
  thereafter by the existing services, because `ctx.actor == participantId == customerId`;
- one opaque bearer credential: `participantToken = secrets.token_urlsafe(32)`, of which only
  `sha256` is stored (`User.tokenHash`), compared with `hmac.compare_digest`.

The storefront's FastAPI dependency `get_participant()` resolves `Authorization: Bearer
<participantId>.<token>` to a `ParticipantRecord`, and every storefront route builds
`CallContext(ws=config.WS_ID, actor=participantId)` from it. **No storefront route accepts a
client-supplied `threadId`, `customerId`, `orderId` or `ws`** — there is no parameter to tamper
with.

**The graph is the authoritative registry, not the in-process map.** `resolve_token()` re-reads
`User.tokenHash` (and `threadId`/`language`) from `ws:{config.WS_ID}`; the in-process
`dict[str, ParticipantRecord]` is a read-through cache and nothing more. This matters more than
it looks: R7 notes that a single file write under `falkor-chat/` restarts uvicorn under
`--reload`, and with an authoritative in-process map that restart would invalidate every token
and bounce every participant to a fresh `participantId` — losing their **cart and order**, not
just their session, because `customerId == participantId`. With the graph authoritative, a
restart is invisible to participants. S6 carries a restart-survival done-condition.

**Why not one workspace graph per participant.**

- **There is no runtime path that creates a workspace.** The DDL is a 400-line shell script
  (`falkor-chat/scripts/bootstrap_schema.sh`, ~20 indexed label groups plus constraints —
  verified live), and the def snapshot must additionally be materialised per workspace. Fifty
  joins would mean shelling out or re-implementing bootstrap DDL inside a request handler.
- **Teardown is 50 `GRAPH.DELETE`s**, and a participant who joins twice leaks a graph.
- **RAM is a non-argument** — measured sub-MB per empty workspace (§2.1), so this option is not
  *cheaper* on the axis it looks cheaper on.
- **It contradicts the platform's own tenancy model.** `falkor-chat/docs/DESIGN.md` §1.1: *"the
  graph boundary is a workspace, many users share it."* A participant is a user, not a tenant.
- **The sweep loop only sweeps one workspace** (`app.py:160-181`, `provider()` → `config.WS_ID`),
  so 49 of 50 per-participant workspaces would have no timer sweep at all.
- It is **not** a violation of `falkor-chat/AGENTS.md` rule 7 ("one graph per workspace; never
  add a `workspaceId` property to filter inside a shared graph") — we add no tenancy property; we
  use the per-actor anchors the schema already has. Live proof that this is the platform's
  intended shape: `ws:qa-cart-totals` already holds two independent `Customer` anchors with
  separate orders (§2.1).

**Why not one thread per participant in a shared channel.** Equivalent for isolation, but a
per-participant channel makes the roster a plain `Channel` scan and makes both resets a clean
subtree delete. The cost is 50 `Channel` nodes — negligible.

**The AC-2 isolation argument, in five parts.**

1. **Cart / order / profile** — anchored on `customerId = ctx.actor` in every one of the nine
   service methods (`services.py:2653-2870`); distinct participants share no node and no read
   path. *Verified live* that multiple such anchors coexist correctly in one graph.
2. **Messages** — each participant's thread id is resolved server-side from their token; the
   client cannot name another thread. `services.read_thread` is thread-scoped.
3. **Agent-visible context** — `salesperson@v7` grants **no** `graphrag_retrieve`, and
   `query_graph_data`'s dataset registry (`querygen.py:222`) exposes exactly two datasets:
   `catalog` (`Product` in `reference`) and `knowledge_base` (`Entity`/`Document`/`Chunk` in the
   workspace). **`Message`, `Customer`, `Cart`, `CartItem`, `Order`, `OrderLine` are unreachable
   through it**, and `querygen`'s compiler rejects any label/property outside the registry. The
   agent therefore cannot surface another participant's data even under prompt injection.
   `executor._read_thread_context` reads only the run's own `threadId`.
4. **The responder fall-through**, closed twice over. `trigger.maybe_trigger` step 4 hands an
   unhandled message to the M2 `AgentResponder`, whose retrieval is **workspace-wide**
   (`DESIGN.md` §M2; `services.hybrid_search` with `channel_id=None`). **S3** makes it
   structurally unreachable by wiring the trigger with `responder=None`. **And §4.4 measure 3 is
   a second, independent barrier**: with no participant message ever embedded, the responder's
   `Message` ANN pool contains no participant data at all, even if S3's flag were wrong. Both
   barriers are load-bearing — **a future reversal of measure 3 reopens this part**, so it must
   never be treated as a pure performance knob.
5. **No unauthenticated reader of the workspace exists** in the storefront deployment — the
   legacy REST router, the `/` web mount and `/mcp` are not in the route table (§4.9).

**Presenter identity — `FALKORCHAT_STOREFRONT_PRESENTER_KEY`, restored by stakeholder decision (OQ-5).**
One operator secret, typed once at `/shop/presenter`, exchanged for a presenter bearer token via
`POST /shop/api/presenter/session` (rate-limited, S10). It is not a login: no accounts, no
per-user credentials, no identity store, no authorization model — the same category of thing as
`FALKORCHAT_AGENT_ID`. **The name takes the `FALKORCHAT_STOREFRONT_` prefix like every other
storefront variable** — as delivered by S6 (`config.py:182`, pinned by
`tests/test_storefront.py`'s env-name test and documented in `falkor-chat/docs/SERVER.md` §1.3).
v1.16 dropped the `STOREFRONT` segment here, in R6 and in OQ-5, while §5.1's S6 row elided the whole
prefix; S10/S11 following the prose would have read the key from a variable nothing sets
and compared against `""`, so presenter login would **silently never authenticate** (see §5.1's
spell-in-full rule, and S10's `presenter_configured` clause for why an unset key must be rejected
*before* `hmac.compare_digest`). The alternative of letting any participant fire "reset everyone" fails
AC-5's evident intent and hands a demo-ruining button to the audience.

**Do not "simplify" this to a localhost check.** That variant was proposed, taken to the
stakeholder, and rejected **on evidence**, not on taste. `analyst` executed the check against the
pinned venv: uvicorn 0.49.0 wraps the app in `ProxyHeadersMiddleware` with
**`proxy_headers=True` by default**, trusting `os.environ.get("FORWARDED_ALLOW_IPS",
"127.0.0.1")`. Behind any TLS-terminating reverse proxy — the realistic way to get the HTTPS §3
contemplates — every request's immediate peer *is* `127.0.0.1`, which is trusted by default, so a
LAN client sending `X-Forwarded-For: 127.0.0.1` has `scope["client"]` rewritten to
`('127.0.0.1', 0)` and gets the presenter surface; with no XFF at all, every request simply *is*
loopback and reset-everyone is open to the whole audience unconditionally. A
`FORWARDED_ALLOW_IPS=*` inherited from the operator's shell produces the same inversion without
any proxy. That is **weaker than the key it would replace** — the key requires network reach
*and* a secret; the loopback check requires only that the server be wrong about one field — and
it forecloses HTTPS, which the key-based design does not. Two further alternatives were
considered and rejected: *first participant to join becomes presenter* (a participant refreshing
before the presenter joins steals it, unrecoverably) and *a token minted at startup and printed
to the operator's terminal* (strictly stronger than a standing secret, but it forces the presenter
to be at the server console at start time, and any process restart — a crash, an OOM kill, an
operator restarting to change a setting — silently invalidates it mid-demo with no way to recover
from a phone; the stakeholder chose the standing key with that trade-off stated). *(v1.1 grounded
that second reason on the `--reload` restart R7 describes, which is wrong: S11 sets a non-empty
`UVICORN_ARGS` precisely to disable `--reload`. Re-grounded on crash-restart, which S11 does not
remove.)* The standing
shared secret is R6's accepted residual.

**FR-4 puts the presenter in the audience too** ("every participant — presenter included — has
their own independent conversation, cart, and order state"). Settled: the presenter holds **two
independent credentials in one browser**, stored under separate keys — an ordinary participant
token from the normal join flow, and a presenter token from `/shop/presenter`. `reset_all` clears
the presenter's own *conversation* along with everyone else's (FR-7 says "every participant's")
and invalidates their *participant* token, but **not** their presenter token, so they can keep
driving the demo through the reset. The SPA's presenter view is a separate route in the same
bundle, not a separate app.

**This needs a `graph-dba` design note.** Yes, and narrowly scoped: `docs/plans/salesperson-ui-graph.md`
must fix (a) the exact provisioning Cypher (`ensure_user` + channel/thread + `MEMBER_OF`,
idempotent, constraint-backed); (b) **the two reset deletes** against the explicit keep/delete
inventory in §4.8, including the quiesce contract; (c) the two order reads B4 identified
(`get_customer_current_order`, `order_belongs_to_customer`); (d) a `GRAPH.PROFILE` confirmation
that the participant-scoped reads stay index-backed at 50 participants × ~40 messages, and
whether any new index or constraint is needed (I believe not — `User.userId`, `Thread.threadId`,
`Customer.customerId`, `Order.orderId` are all already indexed, verified live — but that is a
`graph-dba` call, not mine; note `User.tokenHash` is read only after a `userId` anchor, so it
needs no index of its own). It must **not** cover product images: §4.7 keeps those out of the
graph entirely.

### 4.4 Concurrency — the LLM endpoint is the ceiling, and this must be said plainly

**Finding for the stakeholder, unburied:** *~50 participants can join, browse, hold state and see
their panels update with no degradation. ~50 participants issuing an agent turn at the same
instant cannot be served at interactive latency by one local LM Studio instance on a 6 GB GPU.
Turns will queue.* That is a property of the hardware and the model-serving endpoint, not of the
UI, and no amount of front-end work changes it. What the plan can do — and does — is make the
queueing **fair, visible, and non-degrading to everything else**.

Four concrete measures, each a step in §5:

1. **A dedicated bounded turn executor, keyed by participant.** `storefront.py` owns its own
   `ThreadPoolExecutor(max_workers=FALKORCHAT_STOREFRONT_TURN_WORKERS, default 4)`, sized to LM
   Studio's configured parallelism, and schedules agent turns there instead of on
   `BackgroundTasks`. Agent turns then never touch anyio's ~40-thread limiter, so poll reads stay
   instant no matter how deep the turn queue is. Per-participant turn state (`idle` / `queued`
   with position / `thinking`) is exposed on `GET /shop/api/state` so the UI can show a queue
   position rather than an indefinite spinner.

   **1a — at most one in-flight turn per participant, enforced server-side.** This is not
   ergonomics, it is a correctness fix: `trigger.maybe_trigger` step 2 resumes only a run in
   status **`waiting`**, so a second message posted while the first turn is still `running` falls
   through to step 3 and starts a **second `WorkflowRun` on the same thread** — two runs driving
   the same `assistant` step, both calling `post_message`, both consuming the resource this
   section identifies as the ceiling. Nothing in the platform guards start-while-running. So
   `POST /shop/api/messages` **refuses with `409 TurnInProgress` before writing the message**
   when that participant already has a turn in flight or queued. Refusing before the write is
   deliberate: a written message with no reply would sit in the transcript forever. The client
   keeps the text in the composer and re-enables send when `turn.state` returns to `idle`.

   I deliberately did **not** take `analyst`'s "one in-flight *and* one pending" variant: a
   pending slot buys a marginal typing-ahead nicety and costs a second queue-position concept in
   both the API and the UI, while the `409`-plus-retained-composer path is the standard chat
   affordance and is trivially testable. A client-side disabled send button is *not* sufficient
   on its own — §6.4's load harness will not honour it, which is exactly how this defect would
   reach production.

2. **Raise the anyio limiter to 100** from `FALKORCHAT_THREAD_LIMIT`, **inside `create_app`'s
   `_lifespan`, before `yield`** — `to_thread.current_default_thread_limiter()` is event-loop
   scoped and raises `anyio.NoEventLoopError` outside a running loop (executed against the pinned
   venv). With measure 1 in place this is close to cosmetic — 50 clients × 2 polls / 2 s ≈ 50
   req/s of millisecond-scale indexed reads against 40 threads — so keep it as headroom, but it
   is **not** load-bearing and should not be cited as if it were.

3. **Do not embed storefront messages.** The storefront's own post path calls
   `services.post_message` directly and schedules **only** the turn — no `_safe_embed`. The
   `salesperson` def has no `graphrag_retrieve`, so every embedding call is pure GPU contention
   for zero benefit.

   **Sizing it honestly:** this removes **one** embedding call per posted message
   (`text-embedding-qwen3-embedding-0.6b`, 0.6 B, 30 s timeout) against **up to eight** chat
   completions on `mistralai/ministral-3-3b` (180 s timeout) per turn — roughly 1 of 2–9 endpoint
   calls, and a considerably smaller fraction of GPU-seconds. v1.0 said "roughly halves the load
   on the LLM endpoint"; that was wrong by 3–9× and is corrected here, because §4.4 is the section
   that feeds OQ-1's capacity conversation and a wrong number here costs hardware money. §6.4
   Run B measures the actual delta rather than the plan asserting one. The measure stays
   regardless of its size — **its primary value is not performance at all**, it is being the
   second structural barrier under §4.3 part 4.

4. **One uvicorn worker, stated deliberately.** `--workers > 1` is rejected: `LazyFalkorDB`, the
   in-process sweep tick and the turn queue are all per-process; multiple workers would multiply
   sweep ticks and fragment the queue. Recorded so nobody "optimises" it later.

**How AC-3 gets tested** — §6.4. AC-3's *literal* wording is addressed there and in §10: under
the stakeholder's chosen OQ-1 basis it is not fully satisfiable for agent turns, and the test
report must say so rather than record a pass against wording it does not meet.

### 4.5 Per-participant language (FR-3 / AC-9) — run-ctx carrier + one def version bump

The model already receives the run's whole `ctx` as a `CONTEXT:` turn on every LLM iteration
(`executor._assemble_messages`), and that ctx is written once at `start_run` and only ever
*reloaded* by `_drive_loop` (`executor.py:604`), never rewritten on the chat path — so a value
placed there survives every turn of a long conversation and a re-start after a failed run. So
language is data, not topology:

- **`salesperson@v7`** — a version bump that republishes the full cumulative
  `config.tools` / `systemPrompt` / **`config.model`** (all three are create-only; omitting
  `config.model` silently reverts K-056's Ministral re-point) and adds **two sentences** to
  `systemPrompt`: one for language (reply in the language named by `language` in the CONTEXT
  block; if none is named, reply in English) and one for §4.10's order-time delivery-address
  confirmation. Topology is byte-identical to v5, so the K-034 409 topology-conflict path is
  never hit. **The v5→v7 gap is deliberate — `v6` is a burned version number**: it denotes the
  reverted, never-shipped K-060 experiment already materialized into `ws:acme` from an
  uncommitted tree, which `config`'s create-only rule makes unoverwritable there, so it is never
  reused (`docs/reviews/salesperson-ui-impl.md` F-1).
- **`services.start_workflow_run`'s chat path learns to merge a caller `run_ctx`** into
  `{"threadId": …}`, applying the same reserved-key rejection the process path already applies
  (`threadId`, `error`). `trigger.WorkflowTrigger.maybe_trigger` grows an optional `run_ctx`
  parameter it forwards to the start branch.
- The storefront passes `run_ctx={"language": <participant's choice>}` when starting the
  participant's run.
- The UI's **own** chrome is localised independently with `react-i18next` (one JSON bundle per
  locale).

**Alternatives rejected.** *One def per language* (`salesperson-pt@v1`, …) gives the strongest
output quality — a fully localised system prompt — but multiplies the version-maintenance and
`config.model`-carry-forward obligation by N and needs per-participant def selection. **Kept as
the documented reversal path** if the live adherence check (§6.3 #7) fails. *Seeding the thread
with a hidden language message* needs no server change but pollutes the transcript and depends on
the UI hiding a message, which is fragile.

**Risk, stated up front:** a 3 B model honouring a language instruction carried in a JSON CONTEXT
block is exactly the class of thing K-057/K-060 shows this model getting wrong. **AC-9 is not
signed off on code review** — it requires a live adherence run (§6.3 #7).

Shipped locale set (OQ-2, confirmed): **English (default), Brazilian Portuguese, Spanish**.

### 4.6 Order lifecycle (FR-9 / AC-7) — REST over `services.advance_order`, not the `order-fulfillment` def

`services.advance_order` already implements the whole lifecycle as a guarded CAS; it needs a
route, an ownership check, and — per B4 — two repository reads that do not exist yet (§2.2). The
storefront adds:

- `GET /shop/api/state` returns the participant's current order (`orderId`, `status`, frozen
  lines, total) alongside cart and profile — one round trip per poll, which matters at 50×. The
  order block is populated from a **repository** read (`get_customer_current_order`), never
  composed from Cypher in `storefront.py`: `falkor-chat/AGENTS.md` rule 1 and `DESIGN.md` §14.2
  put all Cypher in `repository.py`, and this plan does not carve an exception.
- `POST /shop/api/order/advance {transition}` — `fulfill` | `deliver` | `cancel`, **ownership
  checked** via `order_belongs_to_customer` before the CAS, returning `409` when the CAS guard
  doesn't match (`advance_order` → `None`, i.e. a stale or out-of-order attempt) and `404` when
  the order isn't theirs. The order id comes from the server-side state, never the request body.

**Why not drive `order-fulfillment@v1`.** That def's own steps have **no side effect on the
`Order`** — `proof_defs.py`'s `ORDER_FULFILLMENT_DEF` comment and `falkor-chat/AGENTS.md`'s
script table both state the `Order.status` write is a separate `services.advance_order` call the
caller makes *alongside* `submit_workflow_input`. Driving it would add a `WorkflowRun` per order,
two REST calls per transition and a `maxSteps` budget to babysit, for zero user-visible
difference. **Reversal trigger:** if a demo ever needs to *show* the fulfillment workflow's step
progress, wire the def then — the storefront's advance route becomes the pair
(`submit_workflow_input` + `advance_order`) with no client change.

**Who advances it (OQ-4, confirmed): participant self-serve only.** No presenter-driven variant
is built. One UI consequence, which is a real one: a customer tapping "Fulfil" then "Deliver" on
their own purchase reads as a broken product to a business audience — precisely R2's amplification
concern. So S14 presents `fulfill`/`deliver` inside a visually distinct **"demo controls"**
affordance, labelled as a simulation of the warehouse, while `cancel` is presented as the
ordinary customer action it actually is.

### 4.7 Product images (FR-11 / AC-11) — static assets keyed by `productId`, no graph change

Images are authored at `salesperson/public/products/<productId>.<ext>` — 15 known, deterministic
slugs, verified live (§2.1) — and Vite copies `public/**` into the build output.

**The manifest is built from the *served* directory, not the source tree**:
`storefront.build_image_manifest()` lists `<FALKORCHAT_STOREFRONT_DIR>/products/` at startup and
intersects the basenames with the catalog's `productId`s. Accepted extensions are exactly
`{.webp, .jpg, .jpeg, .png}`, first match in that order. `GET /shop/api/catalog` returns each
product with `imageUrl: "/shop/products/<productId>.<ext>" | null`, consistent with Vite's
`base: "/shop/"`.

v1.0 pointed the manifest at `salesperson-ui/public/products/`, which only coincides with the
served directory when the source tree happens to sit beside the build output. Ship `dist/` alone
— the shape OQ-6 confirms — or point `FALKORCHAT_STOREFRONT_DIR` anywhere else, and the manifest
would be empty, every `imageUrl` `null`, and **AC-11's checks would still pass**, because its
negative branch ("no placeholder element") masks the total failure of its positive branch. So
S7's done-condition asserts a **non-empty** manifest against a fixture asset directory, and §6.3
#9 gains an explicit positive case. **S7's assertion is necessary and not sufficient**, because S7
delivered `Storefront(..., storefront_dir=None)` falling back to `config.STOREFRONT_DIR`: the
manifest is only ever as good as the directory its *caller* hands it, and that caller is
`create_app`. So the wiring — one directory value reaching both the `Storefront` and the `/shop`
mount — is **S8's** done-condition (§5.1), stated there rather than here.

**AC-11's no-placeholder rule is satisfied structurally, not cosmetically:** the client renders
the `<img>` element **only** when `imageUrl !== null`, and the card has a text-only variant. It
must **not** use an `onError` swap — that flashes a broken image before falling back, which is a
placeholder by another name.

**Why not a graph property.** `reference` is global and `seed_catalog.sh` is its sole writer with
a byte-identical-reseed guarantee (`test_queries.sh` and a default `pytest` run both wipe
`reference`); an image field would either bind the UI's asset set to a seed-script edit or drift
per deployment. Binary/base64 in the graph additionally violates DESIGN's RAM rule for no
benefit. **A URL property would be the right answer only if the catalog ever became
data-driven** — that is the reversal trigger. **No `graph-dba` note is needed for images.**

Two operational notes for the demo brief: the manifest is built **at startup only**, so dropping
an asset in later needs a restart; and image sourcing is OQ-6's confirmed answer — an agent
sources ~15 permissively-licensed stock photos and records the licence in `salesperson/README.md`.

### 4.8 The two resets (FR-7 / AC-5)

**"Reset mine" keeps the participant's identity.** §5.2 in v1.0 said the token was invalidated
while §4.8 said it survived; the contradiction is settled in favour of **survival**, for three
reasons. (i) Re-joining would mint a fresh `participantId`, orphaning the old `User`/`Channel`
in a graph the presenter roster reads. (ii) `customerId == participantId`, so a new id is a new
customer — which is right for "reset", but the *orphan* is not. (iii) It gives `reset_all` a real
and useful asymmetry: only the presenter-scoped reset deletes `User` nodes and therefore
invalidates tokens. The client, after its own reset, returns to a **language step** (not the full
join screen) with the previous language pre-selected — one tap to continue, and the choice is
genuinely re-offered, which was §4.8's original rationale for re-offering it at all.

| Control | Who | Deletes | Survives |
|---|---|---|---|
| **Reset mine** — `POST /shop/api/reset` | the participant, own token | their `Thread` and everything hanging off it — `Message` + `NEXT`/`HEAD`/`TAIL`/`POSTED_BY`/`MENTIONS_MEMBER`/`EMITTED`, `ReadCursor` + `HAS_CURSOR`, `WorkflowRun` + `StepRun` + `TraceEvent`; plus their `Cart` + `CartItem`s, `Order`s + `OrderLine`s, and their `Customer` **†** | their `User` (token, `displayName`, `language`) and `Channel`; a **fresh** `Thread` is minted and `User.threadId` repointed; `Agent`; `WorkflowDefSnapshot`/`Step`; `WorkspaceConfig`; `Document`/`Chunk`/`Entity`; every other participant's subgraph |
| **Reset everyone** — `POST /shop/api/presenter/reset-all` | presenter token only | the above for every participant, **plus** every participant `User` and `Channel` — so all participant tokens are invalidated and every client is bounced to the join screen | `Agent`; `WorkflowDefSnapshot`/`Step`; **`WorkspaceConfig`**; `Document`/`Chunk`/`Entity`; `config.USER_ID`'s lifespan-created `User`; the presenter's own presenter token; `reference` entirely |

**† The `Customer` is deleted and then immediately re-created, name-only.** The column is
*Deletes* and the delete does delete it — but the row reads as a post-reset inventory, and the
post-reset graph holds a `Customer` again: §4.10's profile re-write runs straight after the delete
and `services.save_profile` → `upsert_profile` **`MERGE`s** the node. What does *not* come back is
`deliveryAddress`. **That is the operative post-reset fact, and it is what the assertion must be
keyed on** — `profile == {"name": <displayName>, "deliveryAddress": None}`: the `None` address is
the only thing that distinguishes a re-written name from a survivor. The emptiness of the
`PLACED`/`Cart` subgraph is a corroborating count, never the proof (`docs/reviews/salesperson-ui-impl.md`
`## Pass 7`, Ruling 2 — S7 delivered exactly this assertion).

**The survivor rule that no label can express — and which S0 must state as a *scoping* rule, not
a label list.** `config.WS_ID` defaults to `"acme"` (`config.py:16`), and `ws:acme` is the repo's
primary dev/demo workspace: live inventory today is 2 `Channel`, 2 `Thread`, 52 `Message`,
1 `User`, 1 `ReadCursor`, alongside 544 `Entity` / 87 `Chunk` / 29 `Document`. S11 therefore pins
`FALKORCHAT_WS_ID=demo` (§4.9), but the pin is a mitigation, not the guarantee — an operator can
still point the storefront at a populated graph. The binding rule is:

> **Every `Channel`, `Thread` and `Message` not reachable from a participant `User` survives both
> resets** — `seed_demo.sh`'s `demo-general` channel and `demo-welcome` thread included, and every
> message in them. A participant `User` is one carrying `tokenHash` (the same predicate §5.2's
> roster filters on); nothing else is a reset target, whatever its label.

This is the one blind spot a label checklist structurally cannot cover: **victims and survivors
share the labels `Channel`, `Thread` and `Message`**, so "assert every survivor by label" passes
unchanged on a delete that wiped `demo-general`. S4 closes it with a *positive* assertion instead
(see its done-condition), not another label assertion.

**Three labels v1.0 never named, all of which S0 must also adjudicate explicitly.**
`bootstrap_schema.sh`'s inventory is `Agent, Cart, CartItem, Channel, Chunk, Customer, Document,
Entity, Message, Order, Product, ReadCursor, Step, StepRun, Thread, TraceEvent, User, WorkflowDef,
WorkflowDefSnapshot, WorkflowRun, WorkspaceConfig`, plus unindexed `OrderLine`.

- **`ReadCursor`** — per-member/thread, reached by `HAS_CURSOR`; orphaned by a thread delete, so
  it goes with the thread.
- **`WorkspaceConfig`** — the K-042 per-workspace model-override singleton, and a **must-survive**.
  A broad "delete everything that isn't an `Agent`" sweep would take it, silently changing model
  resolution mid-demo and **undoing K-056's Ministral re-point for every subsequent turn**. This
  is the single most expensive mistake available in the reset design.
- **`Document`/`Chunk`/`Entity`** — survivors of *both* resets, not just `reset_all`.

**The delete is thread-scoped, not author-scoped.** The agent's own replies live inside the
participant's thread and are authored by the `Agent`; an author-scoped delete would leave them
orphaned against a deleted `Thread`, and a "delete everything this actor authored" sweep would
additionally cross participant boundaries the moment the `Agent` is the author. S0 must state
this and S4 must test it.

**Quiesce contract — neither reset may race a turn in flight.** Both deletes remove `Thread`,
`WorkflowRun`, `StepRun`, `TraceEvent` and `Message` nodes that a turn executing on §4.4's thread
pool may be mid-write against: `services.post_message` raises `ThreadNotFoundError` on a vanished
thread, while `_record`/`suspend_run` writes against a deleted run silently no-op — a turn that
consumes an LLM call and posts nothing. (v1.0 also claimed those writes leave **orphan** rows;
S0's note disproved it by execution — all three are anchored on deleted nodes and create nothing.
`advance_cursor` is the one real orphan producer: `docs/plans/salesperson-ui-graph.md` §7, F3.)
"Reset everyone" mid-demo, with up to `turn_workers` turns in flight and a queue behind them,
is the realistic case — it is R4's "wrong and it either bricks the demo or wipes a bystander"
arriving through the *timing* dimension R4 did not consider. So:

- **Reset mine** cancels that participant's queued turn and, if one is in flight, waits for it
  bounded by `FALKORCHAT_STOREFRONT_QUIESCE_S` (default 30 s) before deleting; on a **quiesce**
  timeout it returns `503` and changes nothing (a FalkorDB socket timeout is a different case —
  see F8 below). **The two halves land in different steps, and the ordering between them is the
  point.** S7 delivered the wait alone, correctly: there is no ordering in which cancelling would
  have changed the *result*, because a queued turn reaches a worker, completes and clears its own
  entry, so the wait subsumes the cancel and differs only in latency — and S7 deliberately did not
  drop the turn-map entry as a stand-in, which would report idle while the job was still queued and
  let the delete race the very turn quiesce exists to prevent. **What waiting does not subsume is
  availability:** the 30 s budget sits against a **180 s** agent timeout, so a slow turn turns
  reset-mine into a `503` — a refusal, nothing reset — exactly where cancelling the queued work
  would have let it succeed. That is a designed outcome and the `503` contract is explicit, but it
  is *why* this bullet asks for cancellation at all. The queue is **S9**'s, so S9 owns the
  cancellation and must put it **in front of** the wait, never in place of it
  (`docs/reviews/salesperson-ui-impl.md` `## Pass 7`, Ruling 3).
- **Reset everyone** stops intake first (every subsequent post gets `409` until it completes),
  then drains, then deletes.
- **F8 — a reset that times out on the way to FalkorDB means *unknown*, not "nothing changed",
  and it is a server-side rule.** The timeout in question is falkor-chat's own Redis socket
  timeout to FalkorDB — `FALKORDB_SOCKET_TIMEOUT`, default 10 s (`config.py:29` → `db.py:44`'s
  `FalkorDB(socket_timeout=…)`) — **not** the browser's. `falkor-chat/docs/QUERIES.md` §18.7 states
  it as delivered: the module's `TIMEOUT` applies to reads only, so a slow reset is never truncated server-side,
  and if one ever crosses the socket bound **the client raises `TimeoutError` while the server
  commits the delete** — "client" there being falkor-chat's Redis client, which is what v1.8
  misread as the browser. So a `redis` `TimeoutError` from either reset must **never** map to the
  quiesce `503`: it gets its own **`504 reset_state_unknown`**, and S7/S10 re-read state and
  report from the graph rather than claiming nothing changed. S12a's half is only what the
  browser can still see (§5.2). **This argument is about the resets because they are what §4.8 is
  about, but nothing in it is reset-specific** — any write whose query crosses the socket bound has
  the same "may have committed" ambiguity. S8's error map generalises it to **the five writing routes
  of §5.3's route-class table** — not "every route", which is what made v1.16's gate uncomputable —
  and §5.3's cross-cutting table carries **one row per writing route**; the resets keep their own
  named code (`reset_state_unknown`) because C4's *action* differs per route, and on join it is not a
  re-read at all: the token was never delivered, so there is nothing to read with (§5.3 C4, R12).

  **The re-read is allowed to fail too, and it is the *likelier* fault.** It is another query
  against the same graph, and FalkorDB serialises writes per graph — the stalled reset that
  produced the first `TimeoutError` is precisely what will stall the re-read for another
  `FALKORDB_SOCKET_TIMEOUT`. A second `TimeoutError` must **not** escape as a `500`: S7 and S10
  catch it and still return **`504 reset_state_unknown`**, simply **with no state body**. The
  participant-facing meaning is identical either way — *unknown*, never "nothing changed"; the
  state block is a courtesy the response carries when it can, not the contract. Both orderings
  are named test cases in S7's and S10's done-conditions, because a fake repository that times
  out on the reset and *succeeds* on the re-read exercises only the easier half.

- **Stated premise — nothing beneath the application layer retries a reset, and that is a
  dependency default rather than something falkor-chat sets.** The "never retried" rule §5.2
  attaches to the `Thread` UNIQUE violation is only enforceable if the client library does not
  re-issue the command underneath it. On this build it does not: a connection built by
  falkor-chat's own `db.connect()` carries `retry._retries == 0` with `NoBackoff` — introspected
  on the pinned venv at redis-py 8.0.1 (`docs/reviews/salesperson-ui.md` `## Pass 4`, P4-4).
  That zero is **falkordb-py's** choice: a bare `redis.Redis(socket_timeout=…)` on the same
  redis-py yields `_retries == 10` with `redis.exceptions.TimeoutError` in its supported set, and
  `Redis._execute_command` routes every command through `conn.retry.call_with_retry`. Under that
  default a timed-out reset-mine would be re-issued with the same `$newThreadId` and surface as a
  `Thread` UNIQUE violation — *this plan's own "the graph needs repair" signal* — raised by a
  benign, already-committed reset. **Reversal trigger:** any bump of `falkordb-py` or `redis-py`,
  or any change to how `db.connect()` builds its connection, re-checks that the connection's
  `retry._retries` is still `0`; if it is not, pin it explicitly at the `db.connect()` seam, or
  the `504`/never-retried contract above no longer holds and must be re-derived. S8's contract
  test asserts only the **application** layer (call count); it structurally cannot see the
  library layer, which is why the premise is written down here.

S0 specifies the mechanism; S7 and S10 carry its done-condition, which is
`docs/plans/salesperson-ui-graph.md` §7's four conditions **(a)–(d)** — not restated here, and not
the v1.0 "no orphan rows" wording that note supersedes as vacuous.

### 4.9 The storefront deployment has no unauthenticated read path, and exactly one workspace variable

*(This section resolves B2 and B3 together. They are the same trap from opposite sides, and the
obvious fix for either one opens the other — so neither is fixed on its own.)*

**The problem, stated in full.** `api.build_router(...)` is mounted on the same app with **no
authentication of any kind**, resolving every call through the process-constant
`config.get_context()`. It exposes `GET /channels`, `GET /channels/{cid}/threads`,
`GET`+`POST /threads/{tid}/messages`, `GET /search?q=`, `GET /messages/{id}` and
`GET /threads/{tid}/participants`; `falkor-chat/web/app.js` drives exactly that surface from the
`/` mount — the URL a participant reaches by trimming `/shop` off the demo link — and `/mcp` is
the same seam again. If that surface reads the workspace the participants live in, any phone on
the LAN gets a roster of every participant's thread with full transcripts, a workspace-wide
full-text search over every message, and a post box that writes into anyone's thread. That is an
AC-2 failure by inspection.

**The trap.** The intuitive fix is to keep those surfaces mounted and point them at a *different*
workspace — i.e. require `FALKORCHAT_WS_ID ≠ FALKORCHAT_DEMO_WS`. But every seed script derives
its target from `FALKORCHAT_WS_ID` (`seed_demo.sh:41`: `WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"`),
so that configuration seeds the *wrong* graph: `ws:{DEMO_WS}` ends up with no
`Agent {agentId: config.AGENT_ID}` and no materialised def snapshot. Every storefront post carries
`mentions=[AGENT_ID]`, and `services._validate_and_derive_role` raises `UnknownMemberError` on an
unresolvable mention **before any write** — so every participant's first message 500s, the agent
never fires, and the bring-up script's "a reachable `/shop` with a working join" done-condition is
met while the demo is dead. The natural correction to *that* is `FALKORCHAT_WS_ID=$DEMO_WS`, which
re-opens the AC-2 hole exactly. `_sweep_loop` pulls the same way: it resolves its context from
`config.get_context()`, so it only ever sweeps `ws:{config.WS_ID}`.

**The decision: close it by construction, in two moves that cannot be un-made from the
environment.**

1. **The storefront deployment does not mount the unauthenticated surfaces at all.**
   `create_app` grows `dev_surface: bool = True`, and `_build_default_app` derives it as
   `not config.STOREFRONT_ENABLED` — alongside the existing `mount_mcp`, derived the same way.
   With `FALKORCHAT_STOREFRONT_ENABLED=1` the route table contains **only**: `GET /health`
   (liveness — `services.ping`, returns no participant data), the `/shop/api` authenticated
   router, and the `/shop` static mount. No `api.build_router`, no `/` `StaticFiles`, no `/mcp`.
   `dev_surface` is a **function parameter for tests, never an environment variable** — there is
   no env var an operator can set to put the legacy surface back while participants exist. This
   is the structural property: *the dangerous configuration is not expressible.*

2. **There is no `FALKORCHAT_DEMO_WS`. The storefront's workspace *is* `config.WS_ID`, pinned to
   a dedicated value by `start_demo.sh`.**
   Once move 1 removes the unauthenticated readers, `WS_ID` has no security role left, and a
   second workspace variable buys nothing while creating the only thing that made B3 possible:
   two variables that can disagree. With one variable, every seed script derives the storefront's
   workspace by its own existing default, the sweep loop sweeps the right graph, and
   `services.ensure_actor` at lifespan startup projects `config.USER_ID` into the graph the
   storefront actually uses (harmless — §5.2's roster filters on `User.tokenHash IS NOT NULL`, so
   that node is invisible to the presenter). **B3's misconfiguration is not expressible either,
   because there is no second value to get wrong.** *Both halves of this move are now delivered and
   test-enforced by S6: `dev_surface` has no environment variable (move 1), and a tripwire asserts
   that no module in the `falkorchat` package mentions `FALKORCHAT_DEMO_WS` at all — so
   reintroducing it by reflex fails the suite rather than passing review.*

   One consequence this carries, and it is the reason S11 *pins* the variable: `config.WS_ID`
   defaults to `"acme"` (`config.py:16`), so a `start_demo.sh` that merely *used*
   `"$FALKORCHAT_WS_ID"` without setting it would serve `ws:acme` — the repo's primary dev/demo
   workspace, which holds `seed_demo.sh`'s channel/thread and the M2/M5 hand-verification
   transcript. S11 therefore sets `FALKORCHAT_WS_ID=demo` explicitly. **This is still one
   variable, not the rejected two-variable split** — there is no second value for it to disagree
   with, so every argument above survives intact; pinning a single variable to a safe default is
   not the same shape of thing as requiring two variables to differ. And because a pin is a
   default rather than a guarantee, §4.8's non-label survivor rule (and S4's positive test for it)
   is what actually protects a populated workspace.

**Belt-and-braces, on top of the structure — not in place of it.** Both are startup failures, so
a wrong environment is loud at boot rather than silent until the first participant types:

- **Route-table assertion, keyed on the `storefront` *parameter*, not on `config.STOREFRONT_ENABLED`.**
  `create_app` raises when `storefront and dev_surface` — the two surfaces are mutually exclusive
  however the app was constructed. Keying it on the module constant would have made the guard dead
  in exactly the configuration the suite tests: `config.py` resolves every flag at **import**
  time, so an S8 `TestClient` test building the app through `create_app(storefront=True,
  dev_surface=False)` — the way every other test in this suite wires it — leaves
  `config.STOREFRONT_ENABLED` `False` and would skip the guard entirely. S8 asserts the resulting
  route table directly, which is a stronger test than probing for a 404.
- **Storefront readiness preflight** in the lifespan, before `yield`: the demo `Agent` resolves
  in `ws:{config.WS_ID}` (via `resolve_member_kinds`), `salesperson@v7`'s snapshot is present,
  and the catalog is non-empty — otherwise refuse to start, naming the exact fix command
  (`./scripts/seed_demo.sh "$FALKORCHAT_WS_ID"` etc.). A mis-seeded demo can no longer come up
  "green but dead".

**What I declined, and why.** `analyst`'s B2 fix (a) — *"make the storefront refuse to start when
`FALKORCHAT_WS_ID == FALKORCHAT_DEMO_WS`"* — is correct **only if** the legacy surface stays
mounted. Adopted alongside fix (b), it is not merely redundant: it *mandates* the two-variable
split and therefore mandates B3's trap, forcing every seed invocation to carry an explicit
workspace argument that a single missing `"$DEMO_WS"` silently defeats. So (b) is adopted in full
and generalised to `/mcp` and the `/` mount, and (a) is inverted — the two variables are collapsed
into one rather than being required to differ. `teco`'s constraint was that AC-2 must hold by
construction rather than because an env var happens to be right; a rule of the form "these two
variables must disagree" is exactly the class of assumption it rules out.

**Consequence for the non-storefront deployment: none.** `FALKORCHAT_STOREFRONT_ENABLED` is off
by default, so `uvicorn falkorchat.app:app`, `scripts/start_server.sh`, and the whole existing
test suite keep the current app shape — legacy router, `/mcp`, and `web/index.html` at `/`,
untouched. The two deployments are mutually exclusive and that is the point.

**Consequence for R9:** `GET /workspaces/{ws}/readiness` is a legacy-router route and is not
mounted in the storefront deployment, so it cannot be R9's mitigation. It could not have been
anyway — `DEMO_EXPECTED_DEFS` is *two* pairs including `access-request@v1`, which this demo never
seeds. R9 relies on `verify_salesperson.sh`, which is scoped to exactly the two defs that matter,
plus the readiness preflight above.

### 4.10 The join name reaches the profile, and the address is confirmed at order time (AC-8 / FR-10)

AC-8 is *"…**when they place an order**, then the UI's conversation prompts for and then displays
their name/delivery address."* v1.0 satisfied it by pointing at v5's carried-forward profile
guidance — but that guidance is *early-conversation* ("Call `get_profile` once, early in the
conversation… only once per conversation"), there is no order-time address confirmation anywhere
in the def, and `services.place_order` does not require an address. AC-8 was therefore not
actually covered. Two changes close it:

1. **`join()` writes the display name into the profile immediately** —
   `services.save_profile(ctx, name=display_name)` as part of provisioning. Without this, the
   profile panel shows an em-dash for `name` until the model gets round to asking for a name the
   participant typed thirty seconds earlier, which is a visible parity regression against the old
   app's upfront-name sidebar (§2.4's parity table sets that as the FR-10 bar). It also means the
   `Customer` anchor exists from the first moment, which simplifies every subsequent read.
2. **`salesperson@v7`'s second added sentence** covers order-time delivery-address confirmation:
   before placing an order, confirm the delivery address on file, or ask for it if absent; never
   invent one. This is the half v5 genuinely lacks.

**Half of this is a prompt-adherence claim on a 3 B model**, so it does not belong in the
code-review column. §6.3 #5 becomes a **measured** run alongside AC-9's, and AC-8's entry in §10
says so. If it fails, the honest fixes are (a) make it structural — have the storefront refuse
`place_order` without an address, which needs a services change and is a real scope addition, or
(b) accept the gap and record it, the same disposition K-060 already has. **Not** a third wording
guess: this lab's standing discipline is never to ship an unproven mitigation.

---

## 5. Step-by-step implementation

**P** marks steps that may run in parallel with their siblings in the same stage. **Cite, don't
re-list:** a row that implements a contract defined elsewhere (§4.8's reset inventory, S0's note)
**cites the section and names only what the row itself adds**. M-6 is what re-listing costs — S4's
row re-listed S0's method set, dropped one, and so disagreed with the row above it.
**Spell every environment-variable name in full, in every row and in every prose section** — no
prefix elision (a storefront variable written as its bare suffix under an implied
`FALKORCHAT_STOREFRONT_`), because the elided form and the prose form are then two spellings of one
name that drift independently: v1.16 shipped the S6 row elided and §4.3/R6/OQ-5 spelled out
**wrong**; the implementer had to pick one, picked the row — which is what the code implements — and only that made the prose *known* to be wrong (§4.3). The check is
`grep -o 'FALKORCHAT_[A-Z_]*PRESENTER[A-Z_]*' docs/plans/salesperson-ui.md | sort -u` returning a
single spelling — and it only works if no section quotes a wrong spelling even to disown it, which
is why the paragraphs describing this defect name the *segment* rather than the variable. This is
the same defect shape as the Pass 8 majors one level up — a rule stated in two places and
generalised in one — and the same remedy: one canonical statement, cited elsewhere. §5.0 is
regenerated mechanically from §5.1's Files column — it is what dispatch is gated on, so it lists
**every** file any step touches, not only the contested ones.

### 5.0 Shared-file map (regenerated from §5.1; read before dispatching in parallel)

| File | Touched by | Ordering |
|---|---|---|
| `falkor-chat/server/falkorchat/app.py` | S3, S8, S9 | **S3 → S8 → S9** |
| `falkor-chat/server/falkorchat/config.py` | S3, S6 | **S3 → S6** |
| `falkor-chat/server/falkorchat/services.py` | S2, S4 | **S2 → S4** |
| `falkor-chat/server/falkorchat/storefront.py` | S6, S7, S7c, S9, S10 | **S6 → S7 → S7c → S9 → S10** · S7c's touch is one method — `_catalog_rows` loses its second read once `filter_products` projects `productId` (§5.1 S7c). **S8 does not appear here**, which is the point of the S7c split |
| `falkor-chat/server/falkorchat/storefront_api.py` | S8, S9, S10 | **S8 → S9 → S10** |
| `falkor-chat/server/falkorchat/repository.py` | S4, S7c | **S4 → S7c** · S7c's touch is the additive `productId` projection on the delivered `filter_products` (§5.1 S7c) — the one place this plan reaches into a delivered step's file on purpose, which is why it is its own step and not a clause inside S8's |
| `falkor-chat/server/falkorchat/trigger.py` | S2 | — |
| `falkor-chat/server/falkorchat/schemas.py` | S8 | — |
| `falkor-chat/server/falkorchat/proof_defs.py` | S1 | — |
| `falkor-chat/server/tests/test_services.py` | S2, S4 | **S2 → S4** |
| `falkor-chat/server/tests/test_app.py` | S3, S8 | **S3 → S8** |
| `falkor-chat/server/tests/test_storefront.py` | S6, S7, S7c, S9 | **S6 → S7 → S7c → S9** |
| `falkor-chat/server/tests/test_storefront_api.py` | S8, S9, S10 | **S8 → S9 → S10** |
| `falkor-chat/server/tests/test_repository.py` | S4, S7c | **S4 → S7c** · S7c adds the `filter_products` row-key assertion only |
| `falkor-chat/server/tests/test_trigger.py` · `test_salesperson_scaffold.py` | S2 · S1 | — |
| `falkor-chat/server/tests/test_process_input.py` | S2 (**delivered**) | — · outside S2's §5.1 Files column, and accepted: that file already owns the real-graph chat-path harness the `run_ctx` merge needed, and **no other step touches it** (S4 takes `test_repository.py` + `test_services.py`, and S7c appends to the first) |
| `falkor-chat/docs/QUERIES.md` | S2 (§12.1/§12.12, **delivered**), S4 (new §18), S7c (§15.2's `RETURN` line) | **S2 → S4 → S7c** — three disjoint sections, and S2 → S4 is already satisfied by the `services.py` ordering. §12.1/§12.12 are **already edited**, so S4's §18 is an append rather than a merge, and S7c's §15.2 edit is a two-token change to one `RETURN` line in a section neither of the others touches |
| **`falkor-chat/AGENTS.md`** | S1, S11, S16 | **S1 → S11 → S16** |
| `falkor-chat/docs/DESIGN.md` (§5.1 arrow notation — `Channel {channelId, name, participantId, createdAt}`) | S4 (**delivered**) | — · mandated by the S0 note's §12 and absent from this map until v1.8; S4 shipped it anyway |
| **`falkor-chat/docs/HISTORY.md`** | S1, S2 (**delivered**), and every remaining falkor-chat step — one entry per delivered step (`teco`'s F-5 call) | **append-only, in dispatch order** — each step adds its own entry; never a merge |
| `falkor-chat/README.md` | S16 | — |
| **`falkor-chat/docs/SERVER.md`** | S6 (§1.3 env table, **delivered**), S16 (§1.4 + close-out) | **S6 → S16**, by disjoint section — **the third map gap found in review.** §4.1 leans on its §1.4 as "the platform's documented REST surface, unchanged", and §4.9 changes what that process actually serves (no legacy router, no `/`, no `/mcp`, plus `/shop` + `/shop/api`) — so the document §4.1 cites as the reason the surface is undisturbed is itself made stale by §4.9; S16 adds the storefront deployment to §1.4 and does the final pass. **v1.16 said "no earlier step may edit it" and S6 correctly did anyway** — it documented its own six env vars in §1.3 as it landed, which is the better rule for a document a reader consults *during* the build, so ownership is per-section rather than per-document. **§1.5's layout block is deliberately *not* in this plan's scope:** it is headed "Layout (as built, M1)" and lists 8 of the package's 27 modules — the nineteen it omits accumulated across M2–M6 and none of them is this plan's, so hanging it on S8 (or any step here) would make one step the owner of five milestones of documentation debt it did not create, and the `(as built, M1)` label makes the block a dated snapshot rather than a false claim about now. It is a standalone `falkor-chat/docs/BACKLOG.md` item — *refresh §1.5 to the current module set, or retitle it as an M1 snapshot* (`docs/reviews/salesperson-ui-impl.md` `## Pass 6`; `teco` files it) |
| `falkor-chat/scripts/{seed,verify}_salesperson.sh` | S1 | — |
| `falkor-chat/scripts/start_demo.sh` | S11 | — |
| `salesperson/` scaffold + toolchain config (`package.json`, `vite.config.ts`, `build.sh`, `.gitignore`) | S5 | — |
| `salesperson/playwright.config.ts` | S5, S12b | **S5 → S12b** · **single-owner after S12b**: S12d's `presenter.spec.ts` runs under the project S12b defines. If it needs its own project or viewport, that edit is **S12b's** to make — S12d never touches the config |
| **`salesperson/src/{main.tsx,App.tsx,index.css}`** — the SPA's shared entry files | S5 (scaffold), **S12a** (owns thereafter) | **S5 → S12a, and no later step edits them** — S12a lands the provider/layout **mount slots** so S12b and S12c never need to |
| `salesperson/src/**` (everything else) — `api/`+`session/`+`routes.tsx` (S12a) · `layout/`+`components/sheets/` (S12b) · `i18n/`+`locales/` (S12c) · `views/Chat*`+`components/message/` (S13) · `views/{Cart,Order,Profile,Catalog}*` (S14) · `views/Presenter*` (S12d) | S12a, S12b, S12c, S13, S14, S12d | **S12a first**, then S12b ‖ S12c, then S13 ‖ S14 ‖ S12d — the six subtrees named at left are disjoint, which is what makes the three parallel groups safe. The files that fall *outside* all six (the row above) are the collision the S12 split would otherwise reintroduce, and are assigned to S12a for exactly that reason |
| `salesperson/tests/e2e/**` | S12b, S12d | — · **separate spec files** — S12b owns the mobile-shell specs, S12d owns `presenter.spec.ts`; neither edits the other's |
| `salesperson/public/products/**` | S14 | — |
| `salesperson/scripts/load_demo.py` | S15 | — |
| `salesperson/{README,AGENTS}.md` | S5, S16 | **S5 → S16** |
| `docs/plans/salesperson-ui-graph.md` | S0 | — |
| `docs/test-plans/salesperson-ui.md` · `docs/test-reports/salesperson-ui-report.md` | S15 | — |
| root `AGENTS.md` · `docs/HISTORY.md` | S16 | — |

Four gaps in v1.0's map are closed here: `storefront.py` omitted S9 (which read as permitting
S9 ‖ S10 on one file), `storefront_api.py` omitted S9, **`falkor-chat/AGENTS.md` appeared in
no row at all** despite three steps writing it, and **`falkor-chat/docs/SERVER.md` likewise** — the
last found at the S3 gate. All four were missing *rows*, not missing orderings, which is the failure
mode a map regenerated from the Files column cannot catch on its own: a file no step lists is
invisible to the regeneration.

v1.3 closes three more, found once S2 shipped (F-4): `test_process_input.py` appeared in no row,
`QUERIES.md` was assigned to S4 alone, and `falkor-chat/docs/HISTORY.md` was absent entirely (only
root `docs/HISTORY.md` appeared, under S16). **The assessed collision risk of all three was nil** —
no sequencing changes as a result.

### 5.1 The table

| # | Step | Files | Interface / key symbols | Done-condition | Specialist | Parallel |
|---|---|---|---|---|---|---|
| **S0** | **Graph design note.** Scope fixed by §4.3/§4.6/§4.8: provisioning Cypher; **both reset deletes against §4.8's explicit keep/delete label inventory** (`ReadCursor` goes with the thread; **`WorkspaceConfig` must survive** — a sweep that takes it silently undoes K-056's Ministral re-point; `Document`/`Chunk`/`Entity` survive both), **thread-scoped not author-scoped**, and **§4.8's non-label scoping rule: only a `Channel`/`Thread`/`Message` reachable from a participant `User` (one carrying `tokenHash`) is ever a target — expressed in the Cypher's `MATCH`, not left to the caller**; the §4.8 quiesce contract; the two B4 order reads; a `GRAPH.PROFILE` check of participant-scoped reads at 50 participants; an explicit yes/no on new indexes/constraints. Excludes product images. | `docs/plans/salesperson-ui-graph.md` (new) | `ensure_participant`, `reset_participant`, `reset_all_participants`, `get_customer_current_order`, `order_belongs_to_customer` | Note exists, `Status: active`, every query live-verified against a throwaway `ws:` probe graph; keep/delete decided per label; DDL yes/no stated | `graph-dba` | **Blocks S4.** ‖ S1, S2, S3, S5 |
| **S1** | **`salesperson@v7`** (**not `v6` — burned, §4.5**) — bump `SALESPERSON_DEF["version"]`, republish full cumulative `config.tools` **and `config.model`** unchanged, add §4.5's language sentence **and §4.10's order-time address sentence** to `systemPrompt`. Bump both scripts' default version fallbacks. **Also update `falkor-chat/AGENTS.md` rows 82–83** (the script table narrating the `v1…v5` chain and `verify_salesperson.sh`'s expected version) so the doc is not stale for the whole S1→S16 window. `docs/BACKLOG.md`'s K-060/K-062 headings also pin `v5`; those belong to those defects' own tracks and are **deliberately not touched here**. | `falkor-chat/server/falkorchat/proof_defs.py`, `falkor-chat/scripts/seed_salesperson.sh`, `falkor-chat/scripts/verify_salesperson.sh`, `falkor-chat/server/tests/test_salesperson_scaffold.py`, `falkor-chat/AGENTS.md` | `SALESPERSON_DEF`, `SALESPERSON_MAX_STEPS` | `bootstrap_schema.sh <probe-ws>` (the seed script's own prerequisite — a fresh probe graph has no indexes or constraints), then `seed_salesperson.sh <probe-ws>`, then `verify_salesperson.sh <probe-ws>` exits 0 live **against a throwaway `ws:` probe graph, named explicitly, never a shared or populated one** (S0's shape; an unpinned `<ws>` falls back to `FALKORCHAT_WS_ID`, i.e. `acme` — which is how a working-tree def reached `ws:acme` and burned `v6`, §4.5. `reference` is written either way: it is the publish home, and only the *materialize* target is in question); a test asserts `config.model == "lmstudio/mistralai/ministral-3-3b"` and `tools ⊇` v5's; pytest green | `coder` | **P** |
| **S2** | **Chat-path `run_ctx` merge.** `services.start_workflow_run` merges a caller `run_ctx` into the chat path's `{"threadId": …}`, reusing the process path's reserved-key rejection (`threadId`, `error` → `WorkflowInputRejectedError`). `trigger.maybe_trigger` gains an optional `run_ctx` forwarded only to the start branch. | `falkor-chat/server/falkorchat/services.py`, `.../trigger.py`, `falkor-chat/server/tests/test_services.py`, `.../test_trigger.py` | `start_workflow_run(..., run_ctx: dict \| None)`, `maybe_trigger(..., run_ctx: dict \| None = None)` | Chat-path start with `run_ctx={"language":"pt-BR"}` yields a run whose ctx carries both keys; reserved keys rejected before any write; existing callers unchanged (default `None`); pytest green | `tdd-engineer` | **P** |
| **S3** | **Two wiring switches.** (a) `config.TRIGGER_RESPONDER_FALLTHROUGH` (`FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH`, default on) → `WorkflowTrigger(responder=None)` when off (§4.3 part 4). (b) **§4.9's `create_app(..., dev_surface: bool = True)`**, derived in `_build_default_app` as `not config.STOREFRONT_ENABLED` alongside `mount_mcp`; when false, neither `api.build_router` nor the `/` `StaticFiles` mount nor `/mcp` is registered, and a bare `GET /health` liveness route is added. **`dev_surface` is a parameter, never an env var.** | `falkor-chat/server/falkorchat/config.py`, `.../app.py`, `falkor-chat/server/tests/test_app.py` | `config.TRIGGER_RESPONDER_FALLTHROUGH`, `config.STOREFRONT_ENABLED`, `create_app(..., dev_surface=)` | With the fall-through flag off, an unmentioning non-resuming message reaches no responder. With `dev_surface=False`, **`app.routes` contains no legacy route and no `/`/`/mcp` mount** (asserted on the route table, not by probing 404s), and `GET /health` still answers. Default deployment byte-identical: full existing pytest suite green | `tdd-engineer` | **P**, owns `app.py`/`config.py` first |
| **S4** | **Repository + thin service primitives** — **the five queries S0 specifies verbatim** (`docs/plans/salesperson-ui-graph.md` §12: `ensure_participant` §3, `reset_participant` §4, `reset_all_participants` §5, **`get_customer_current_order`** §10.1, **`order_belongs_to_customer`** §10.2) **plus four this plan specifies and the note has no Cypher for** (`add_channel_member`, `get_participant_record`, `set_participant_record`, `list_participants` — designed into `QUERIES.md` §18.2/§18.3), plus `Services` wrappers (`get_current_order`, `order_belongs_to_customer`) so `storefront.py` never holds Cypher (`falkor-chat/AGENTS.md` rule 1, `DESIGN.md` §14.2). **Nine methods, not eight: `ensure_participant` is S4's, not S6's** (`docs/plans/salesperson-ui-graph.md` §12 hands it here twice, mandating its §3 verbatim; `docs/reviews/salesperson-ui-impl.md` M-6). It is the **only writer of `Channel.participantId` tree-wide** — the provenance marker both resets scope on — so without it neither reset ever resolves a participant and **both are inert**; and it stays **one atomic write**, because decomposing it opens a crash window leaving a `tokenHash`-carrying `User` whose channel is unmarked: exactly the unscoped participant the note's provenance reasoning depends on being unreachable. Every query added to `falkor-chat/docs/QUERIES.md` **§18** (§17 is the current highest). | `falkor-chat/server/falkorchat/repository.py`, `.../services.py`, `falkor-chat/docs/QUERIES.md`, `falkor-chat/server/tests/test_repository.py`, `.../test_services.py` | the nine repository methods + two service wrappers; all parameterised; `.query`/`.ro_query` split per the platform rule | Integration tests on an isolated `ws:test` graph prove: two participants' resets are disjoint; the delete is **thread-scoped** (an `Agent`-authored reply in the participant's thread is deleted, and no other participant's is); **every §4.8 survivor is asserted by label**, `WorkspaceConfig` included; **and — the assertion that label checks structurally cannot make — a non-participant `Channel` + `Thread` + `Message` (a `User` with no `tokenHash`, mirroring `seed_demo.sh`'s `demo-general`/`demo-welcome`) is seeded into the probe graph and asserted to survive `reset_all` intact**, because victims and survivors share those three labels; `reference` untouched; post-`reset_all` `verify_salesperson.sh <that same probe graph>` (**argument explicit** — unpinned it reads `ws:{FALKORCHAT_WS_ID}` and proves nothing about the graph under test) + `verify_catalog.sh` exit 0; every method idempotent | `coder` | **after S0** |
| **S5** | **Node toolchain + component scaffold** into the freed `salesperson/` (after U5's `git mv` to `deprecated/salesperson/`). Provision Node/npm (falkor-chat's own note: `node` is not on `PATH` on WSL2), scaffold Vite + React + TS + Tailwind + Vitest + Playwright, `build.sh`, `.gitignore` for `dist/`/`node_modules/`, initial `README.md` + `AGENTS.md`. | `salesperson/**` (new), `salesperson/{README,AGENTS}.md` | `npm run build` → `salesperson/dist/` with `base: "/shop/"` | `./salesperson/build.sh` produces `dist/index.html` + hashed assets from a clean checkout; `npm test` runs; Node version documented. **This done-condition is §4.2's HTMX-fallback decision deadline** — if it cannot be met, escalate before S12a rather than after | `devops` | **P**, after U5 |
| **S6** | **Storefront core** — participant registry + join + token verify + turn-state map. Token `secrets.token_urlsafe(32)`, `sha256` stored on `User.tokenHash`, `hmac.compare_digest`. **`resolve_token` re-reads the graph; the in-process map is a read-through cache only** (§4.3). Join also writes the display name into the profile (§4.10). Env, **spelled in full** (§5.1's rule — v1.16 elided five of these under the prefix and §4.3/R6/OQ-5 spelled the third one wrong): `FALKORCHAT_STOREFRONT_ENABLED`, `FALKORCHAT_STOREFRONT_DIR`, `FALKORCHAT_STOREFRONT_PRESENTER_KEY`, `FALKORCHAT_STOREFRONT_TURN_WORKERS`, `FALKORCHAT_STOREFRONT_QUIESCE_S`, `FALKORCHAT_STOREFRONT_LOCALES`, plus `FALKORCHAT_THREAD_LIMIT` (which does **not** take the prefix). **No `FALKORCHAT_DEMO_WS`** (§4.9). Documents these six in `falkor-chat/docs/SERVER.md` §1.3. | `falkor-chat/server/falkorchat/storefront.py` (new), `.../config.py`, `falkor-chat/server/tests/test_storefront.py` (new) | `Storefront(services, *, presenter_key, turn_workers, quiesce_s)`; `join(display_name, language) -> ParticipantRecord`; `resolve_token(bearer) -> ParticipantRecord \| None` | Join provisions `User`+`Channel`+`Thread`+profile-name idempotently; wrong/absent/malformed/deleted-participant tokens all resolve to `None`; **restart survival: a `Storefront` rebuilt from scratch resolves a token minted by the previous instance** | `coder` | **after S4** |
| **S7** | **Storefront state, reset, catalog, images.** `get_state(ctx)` composing `services.get_profile` + `get_cart` + **`services.get_current_order`** (a repository read, not composed here); `reset_participant` with §4.8's quiesce, **re-writing the profile name afterwards** — the `Customer` node goes with the reset while `User.displayName` survives, so the wrapper re-calls `services.save_profile(ctx, name=<User.displayName>)`; existing call, no new Cypher (`docs/plans/salesperson-ui-graph.md` §12 item 1); `list_catalog()` with an **explicit row bound** (`services.filter_products` defaults `limit=20` — correct for 15 products, silently wrong at 21); `build_image_manifest()` over **`<FALKORCHAT_STOREFRONT_DIR>/products/`** (§4.7); `advance_own_order()` via `services.order_belongs_to_customer` then `advance_order`. | `falkor-chat/server/falkorchat/storefront.py`, `falkor-chat/server/tests/test_storefront.py` | `get_state`, `reset_participant`, `list_catalog`, `build_image_manifest`, `advance_own_order` | State shape stable and the order block populated from the repository read; reset participant-disjoint; **manifest is non-empty against a fixture asset dir** and every `imageUrl` is `/shop/products/<id>.<ext>` or `null`; `list_catalog()` returns all 15 rows; advancing another participant's order refused; **after a self-reset the profile name is back, not an em-dash** (§2.4's FR-10 parity bar); **the quiesce done-condition is `docs/plans/salesperson-ui-graph.md` §7's four conditions (a)–(d)** (that note supersedes v1.0's "no orphan rows" wording as vacuous — those writes create nothing post-reset whether quiesce works or not), **read participant-scoped for reset-mine**: (b)/(c) over that participant's own posts, (d) over cursors owned by the reset participant — S7 has no global intake stop, which is what (b)'s "(intake stopped)" and (d)'s "after `reset_all`" are worded for; **and F8, both orderings** (§4.8, `falkor-chat/docs/QUERIES.md` §18.7): a FalkorDB socket `TimeoutError` returns **`504 reset_state_unknown`** after re-reading state, never the quiesce `503` — **and a stub whose re-read *also* raises `TimeoutError` still returns `504`, with no state body, never a `500`** | `coder` | **after S6** |
| **S7c** | **Ruling 1's catalog projection — `productId` on `filter_products`, and the removal of S7's workaround.** *Carried out of the S7 implementation gate as its own step rather than folded into S8* — **`S7b` is not a typo-gap: that id is taken by the delivered test-only unit that closed Pass 7's three minors (`d9d2f2b`), so this one is `S7c`** — (`docs/reviews/salesperson-ui-impl.md` `## Pass 7`, Ruling 1): Pass 8's stopping rule names **S8's `{handlers} × {routes}` assertion** as the specific place review resumes, and that gate is the payoff for closing eight plan passes without a Pass 9 — handing it an unrelated catalog refactor in two previously-gated files dilutes exactly the review those passes bought. **This is not S7's re-gate:** S7 is approved and delivered; this is an additive change beside it. `services.filter_products` projects `{name, category, price}` and **no `productId`** — the exact field §5.2's catalog row shape and §4.7's manifest intersection are keyed on — so S7 resolves each row's id with a second, index-anchored point read (`services.lookup_product`), a correct and documented `1+n`. The fix is the **identical additive change K-053 already made to `lookup_product`**: `p.productId AS productId` in `repository.filter_products`'s `RETURN` plus the matching row-mapping shift (`falkorchat/repository.py:2762`), after which `_catalog_rows` returns `services.filter_products`'s rows unchanged — dropping the second read **and the `if product is None: continue` silent-drop branch with it**, the catalog route's only unbounded failure path. `falkor-chat/docs/QUERIES.md` §15.2's `RETURN` line is edited in the same change. **The two halves ship together or not at all** — a projection without the simplification leaves the `1+n` and the silent drop in place while reporting the fix as done, which is what the done-condition is built to catch. **No def version bump:** `FilterProductsTool.run` returns its rows verbatim, so the slugs do reach the model — but `LookupProductFactTool` has returned `productId` verbatim since K-053 (`falkorchat/tools.py:428`), so this makes two sibling catalog tools consistent rather than putting a new kind of thing in front of the model, and `salesperson@v7`'s "(name, category, price)" phrasing already understates `lookup_product_fact`'s four fields without this. **The counterweight, on the record rather than only the conclusion:** the reviewer applied the fix live and measured **2473 passed, 14 deselected, zero test edits** — but *zero test breakage measures code, not model behaviour*, and **none of the 14 deselected `live` tests is a salesperson catalog conversation** (they are AC-5 grounding, querygen NLQ and triage), so no harness in this repo observes an LLM regression either way. The evidence for *safe* is the K-053 precedent, not the passing suite (Appendix K §2 and §7). **Reversal trigger:** if a live run after S11 shows the model quoting slugs at the customer, the answer is a prompt sentence in a later def version — never reverting the projection, which would restore the `1+n` and the silent drop. | `falkor-chat/server/falkorchat/repository.py`, `.../storefront.py`, `falkor-chat/server/tests/test_repository.py`, `.../test_storefront.py`, `falkor-chat/docs/QUERIES.md` (§15.2) | `Repository.filter_products(...) -> [{productId, name, category, price}]` (additive); `Storefront._catalog_rows` loses its `services.lookup_product` call | **One test binds both halves so they cannot drift:** with `services.lookup_product` patched to raise, `Storefront.list_catalog()` against the live 15-product `reference` catalog still returns all 15 rows carrying their real slugs — red by `KeyError` if the projection is missing, red by the raise if `_catalog_rows` still does the second read. Plus `repository.filter_products`' row keys asserted as exactly `{productId, name, category, price}` in `test_repository.py`, and §15.2's `RETURN` clause matching the query string in `repository.filter_products`. **S7's existing catalog tests stay green *unedited*** — they assert `list_catalog()`'s output against a live `catalog_repo` rather than against stubs, which is why the reviewer measured zero test edits (Appendix K §2); needing to edit one of them is the signal that this change did more than it should. Full suite green | `coder` | **after S7**, ahead of S8 in dispatch order — and **P** with S8 if wanted: the split leaves the two with no file in common. **Before S9**, which owns `storefront.py` next |
| **S8** | **The `/shop/api` router + mounts.** `storefront_api.build_storefront_router(...)`, the `get_participant()`/`get_presenter()` dependencies, size-bounded Pydantic models mirroring `schemas.py`, **the error map — which this step makes *total by type*, the plan's first totality guard**: today `_register_error_handlers` (`app.py:80`) maps `ServiceError` and the workflow errors only, and **`FalkorDBUnreachableError` (`db.py:47`) has no handler at all**, so a query-time `redis.exceptions.TimeoutError` escapes as a bare `500` — on `/state` and `/messages`, for **every polling participant at once**, in exactly the scenario §4.8's F8 exists for. S8 registers **typed** handlers, in this codebase's own stated idiom of typed handlers "without a blanket handler masking real bugs" (`app.py:136-137`): `FalkorDBUnreachableError` and `redis.exceptions.ConnectionError` → **`503 graph_unavailable`** (nothing was sent — the precedent is `api.py:63` / `app.py:345`); a query-time `redis.exceptions.TimeoutError` → **`503 graph_read_timeout`** on a `reads-only` route and **`504 <op>_state_unknown`** on a `writes` one, with `<op>` fixed per route (`join` / `post` / `order` / `reset`) by §5.3's cross-cutting table. **The classes are §5.3's route-class table, which this step is built against** — five `writes`, four `reads-only`, and **two routes (`GET /shop/api/health`, `POST /shop/api/presenter/session`) that issue no query at all** and therefore can produce **none** of the three; a handler firing on either is a defect, not a row. **Precedence, stated so it cannot be built the other way round:** the two reset routes catch the timeout themselves and answer F8's named `504 reset_state_unknown` (S7/S10); the typed handler is the backstop for the other **seven graph-touching** routes, and must not pre-empt them. Also **maps `RequestValidationError` to the storefront's own stable `{error: "validation_failed", field: "<name>"}`** rather than exposing FastAPI's `loc` shape, because §5.3 C11 dispatches on `field` and a framework-detail body is not a contract — **selection rule pinned: the *first* entry of `exc.errors()` (declaration order for a single request model), and `field` is the last element of its `loc`** (`displayName`, never `body.displayName`), so a multi-violation request produces one deterministic, client-facing name (§5.3 C11). Plus `schemas.py:256-257`'s reserved-key list, which still omits `timerFired`** (`docs/reviews/salesperson-ui-impl.md` F-6 — routed here because S8 edits `schemas.py` anyway), and the `create_app` wiring: include the router and mount `FALKORCHAT_STOREFRONT_DIR` at `/shop` **inside `create_app`** (`/` is a catch-all registered last and Starlette matches in registration order, so a mount added after `create_app` returns is unreachable). **`create_app` must pass its `storefront_dir` explicitly into the `Storefront` it constructs, and mount that same one value at `/shop`** — S7 shipped `Storefront(..., storefront_dir=None)` falling back to `config.STOREFRONT_DIR` (`falkorchat/storefront.py:364`), so a `create_app` that takes the parameter for the mount and lets the `Storefront` fall back to config builds the manifest from one tree while serving assets from another: every `imageUrl` `null` in the benign case, and keyed on the wrong tree in the worse one. §4.7 is written about exactly this failure, and AC-11's negative branch masks it. Also **call `build_image_manifest()` in `_lifespan`**, beside the preflight — §4.7's "built at startup only" is stated nowhere in a step until here, and leaving it to `list_catalog`'s first call makes the first participant's catalog fetch list the directory. It is **not** a preflight condition: an empty manifest is a legitimate deployment (the text-only card variant), so log the count and start. Plus §4.9's **route-table assertion** and the **readiness preflight** in `_lifespan`. Plus **a per-route `responses={…}` declaration on every one of the eleven routes**, naming exactly that route's own returns (its status codes and their bodies — *not* the three cross-cutting ones, which come from the handler set): FastAPI keeps them on the route object, so they are machine-readable and turn the second half of the gate below from a reading exercise into an assertion, at the cost of one dict per route. | `falkor-chat/server/falkorchat/storefront_api.py` (new), `.../schemas.py`, `.../app.py`, `falkor-chat/server/tests/test_storefront_api.py` (new), `.../test_app.py` | `create_app(..., storefront: bool = False, storefront_dir: Path \| None = None, dev_surface: bool = True)` — **`storefront_dir` is forwarded, one value, to both `Storefront(storefront_dir=…)` and the `/shop` mount**; routes per §5.2; `401` bad/absent token, `403` bad presenter key, `404` unknown order, `409` stale CAS / turn in flight / `unscoped_participant`, **`422 validation_failed` + `field`** from the Pydantic bounds on all five input-taking routes (§5.2, §5.3 C11), `503` quiesce timeout, **`503 graph_unavailable`** / **`503 graph_read_timeout`** / **`504 <op>_state_unknown`** from the typed handlers, **`504 reset_state_unknown`** on a reset's own FalkorDB socket timeout, and a `Thread` UNIQUE violation on **reset-mine** propagating as `5xx`, never retried (§5.2) | `TestClient` contract tests for every route, incl. the **auth matrix** and **the cross-participant probe**: with A holding cart items, messages and an order, every route called with B's token returns only B's data. **A stubbed repository raising `redis.exceptions.TimeoutError` from a reset is called exactly once** — the *application* layer never retries (§4.8's stated premise; the library layer is beyond this test's reach by construction, which is why the premise is written down rather than only asserted). **The gate is decidable, not a reading exercise, and it has two halves.** *(i) Handler half:* **`{registered handlers} × {routes-that-can-raise-them} ⊆ §5.3's completeness table`** — enumerate the handlers actually registered on the app object, cross them with the route table **filtered by §5.3's route classes** (a `503 graph_read_timeout` is expected on the four `reads-only` routes, a `504 <op>_state_unknown` on the five `writes`, and **neither on the two `no graph access` routes**), and assert every resulting `(route, response)` has a row; a handler with no row, or a row with no producer, fails the step. Without the class filter this half is not computable and its symmetric side falsely fails on `/health` and `presenter/session` — the classification is the input, not a nicety. *(ii) Declaration half:* **`{declared in each route's `responses={…}`} ∪ {handler-produced} == §5.3's table`**, read back off `app.routes`; every route must carry a declaration, so an omission fails loudly rather than silently shrinking the set. **What neither half closes** — a handler that returns a `JSONResponse(status_code=…)` directly never raises, so it is invisible to the handler set, and a declaration is itself an enumeration that can be wrong (a declared response nobody produces, or a produced one nobody declared). The declaration half narrows that residue and does not remove it; the per-route contract tests then prove each declared entry is actually producible, so the two disagree loudly instead of agreeing by omission — and **C13 is the backstop for whatever all three miss**, which is the reason the client guard exists rather than being redundant with this one. **Asserted by execution, not inspection:** a stubbed repository raising `redis.exceptions.ConnectionError` returns `503 graph_unavailable` on `/state`, and one raising `redis.exceptions.TimeoutError` returns `503 graph_read_timeout` on `/state` but `504` on `POST /messages` — **and no route anywhere answers a bare `500`**. **Two source tripwires carried from S6's gate** (`docs/reviews/salesperson-ui-impl.md` `## Pass 6`, answers 1 and 4): `storefront_api.py` **never calls `.lookup(`** — `lookup` and `resolve_token` return the identical `ParticipantRecord`, so a router authenticating through the read-through cache would be indistinguishable from one authenticating against the graph; and `create_app` constructs the `Storefront` **without passing `id_gen`** — S6's participant-id collision argument rests on no caller ever pinning it, and S8 is the first caller. `/shop` shadows nothing. **The two `no graph access` routes are asserted negatively**: with the repository stubbed to raise on *any* call, `GET /shop/api/health` and `POST /shop/api/presenter/session` still answer their normal `200`/`403`/`422` — proving they issue no query and so can produce none of the three cross-cutting responses. **The image wiring is asserted so that missing it goes red, which the obvious version of this test does not**: with `config.STOREFRONT_DIR` monkeypatched to a **different, also-populated** directory, `create_app(storefront=True, storefront_dir=<tmp>)` must serve `GET /shop/api/catalog` carrying **`<tmp>`'s** `imageUrl`s and `GET /shop/products/<id>.<ext>` must `200` out of that same tree — so a `create_app` that reads config instead of forwarding the parameter fails with *wrong* URLs and one that forwards only to the mount fails with `null` ones (a single populated directory against an unset config catches only the second, which is why the config default is pointed somewhere real). Preflight refuses to start on a missing `Agent`, missing snapshot or empty catalog, naming the fix command | `coder` | **after S7; owns `app.py` after S3** |
| **S9** | **Concurrency layer** (§4.4). The bounded `ThreadPoolExecutor` turn queue **keyed by `participantId`** with queue-position accounting and §4.4 measure 1a's `409 TurnInProgress` refusal *before* the message write; the storefront post path (`services.post_message` + enqueue trigger with `run_ctx={"language": …}`, **no `_safe_embed`**); raise the anyio limiter **inside `_lifespan` before `yield`**; graceful executor shutdown. **And the half §4.8 asked for that S7 could not build: cancellation of a *queued* turn runs in front of `_await_quiesce`, never in place of it.** S7 shipped the wait alone and that is correct for the *result* — a queued turn reaches a worker, completes and clears its own entry, so waiting subsumes cancelling and differs only in latency — but not for **availability**: `FALKORCHAT_STOREFRONT_QUIESCE_S` is 30 s against a 180 s agent timeout, so a turn that outlives the budget turns reset-mine into a `503` with nothing reset, exactly where dropping the queued work would have let it succeed (§4.8; `docs/reviews/salesperson-ui-impl.md` `## Pass 7`, Ruling 3). Cancel means the queued `Future` is actually cancelled and *then* its map entry cleared; a future already running cannot be cancelled and **falls through to the existing wait** — S7's refusal to clear the turn-map entry as a stand-in stands, because that reports idle while the job is still queued and lets the delete race the turn quiesce exists to prevent. **And one decision this row makes rather than leaving to the dispatch brief: the per-participant record cache goes.** S6 built `Storefront._records` as a read-through cache with `lookup` as its reader, and the S7 gate found **zero production callers** — all eight `.lookup(` call sites are in `tests/` (Appendix K §6) — while `cached_ids()` is diagnostics. Deleting `lookup` alone would leave `resolve_token`'s and `reset_participant`'s `_cache_put`s writing into a map nobody reads, which is worse than either end state, so this is **one decision, not two**. **Decided: remove.** S8's tripwire forbids the router reading it (`storefront_api.py` never calls `.lookup(` — the router authenticates against the graph, not the cache), and this row's own `enqueue_turn(ctx, participant, posted)` **receives** the `ParticipantRecord` from the request thread that already called `resolve_token`, so the worker never needs to resolve one either. With no reader available to any caller, the cache is dead code: S9 deletes `lookup`, `_records`, `cached_ids()` and every `_cache_put`/`_cache_drop` in one change, together with the tests that exist only to exercise them. **Consequence to bank rather than re-solve:** the S7 gate's **S7-2** (the two unpinned error-path `_cache_drop`s) is *dissolved* by this, not fixed — spend no tests on it. **Reversal trigger:** if the implementation turns up a real reader off the request thread — any path needing a `ParticipantRecord` it was not handed — keep `_records` **and** `lookup` together and pin both error-path evictions with tests (S7-2 names them); never keep one without the other. | `falkor-chat/server/falkorchat/storefront.py`, `.../storefront_api.py`, `.../app.py`, `falkor-chat/server/tests/test_storefront.py`, `.../test_storefront_api.py` | `Storefront.enqueue_turn(ctx, participant, posted)`; `turn: {state, queuePosition}` on `GET /shop/api/state` | With `turn_workers=1` and a stub 2 s LLM, three *different* participants' posts report queue positions 0/1/2 and complete in order; **two posts 100 ms apart from one participant produce exactly one `WorkflowRun` on that thread**, the second returning `409` with no `Message` written; poll latency unaffected while the queue is full; executor drains on shutdown; **cancellation is asserted by the case that separates it from waiting** — with `turn_workers=1`, A's stub turn in flight for 2 s and B's turn queued behind it, B's reset-mine at `quiesce_s=0.5` returns **`200`** and B's queued turn never runs (no `WorkflowRun` on B's fresh thread, no reply written), where the wait-only implementation returns `503`; **and no record cache survives** — `Storefront` defines no `lookup`, `_records` or `cached_ids`, `resolve_token` is its sole participant-resolution path, and the eight `tests/` call sites go with them, the suite green without replacements — **S8's `never calls .lookup(` tripwire goes vacuous with the symbol, and that is the correct end state**: it did its work at S8's gate, and with no cache to read the rule it enforced (*the router authenticates against the graph*) holds structurally rather than by assertion | `coder` | **after S8** |
| **S10** | **Presenter surface** — `POST /shop/api/presenter/session` (key → token, **entirely in-process: it reads `config.STOREFRONT_PRESENTER_KEY` (`FALKORCHAT_STOREFRONT_PRESENTER_KEY` — S6's delivered spelling, §4.3) and mints a token without touching the graph, which is what puts it in §5.3's `no graph access` class**; **an unset key must never authenticate** — `hmac.compare_digest("", "")` is `True`, so the route checks S6's delivered `Storefront.presenter_configured` *before* comparing and answers `403` when no key is configured (`falkor-chat/docs/SERVER.md` §1.3); rate-limited: **a fixed per-attempt delay, plus an attempt counter that is deliberately *observational only*** — it is logged and exposed to the operator so a brute-force attempt is visible, and it **never changes the response**, so the codes this route can answer stay `200` / `403` / `422` / the cross-cutting three. **Decided rather than left open (Pass 7, P7-5):** the alternative — a lockout after N attempts — is **rejected**, because there is exactly one shared presenter key (§4.3), so a lockout is a self-DoS: anyone on the LAN could lock the presenter out of their own demo mid-show, which is worse than the brute-force risk it mitigates, given that the fixed delay already bounds the attempt rate and R6 already accepts the standing key as a residual. It would also add an unlisted response (`429`/`423`, or a `403` that silently changes meaning from *wrong key* to *throttled*) that C2's second half would render as "your key is wrong" when the key may be right. **Reversal trigger:** if this is ever exposed beyond a controlled LAN, the answer is not a lockout but K-016), `GET /shop/api/presenter/participants` (roster filtered on `User.tokenHash IS NOT NULL`, so `config.USER_ID`'s lifespan node never appears; **it returns §5.2's four keys and nothing more** — the delivered `list_participants` (`falkor-chat/docs/QUERIES.md` §18.3) projects no activity data, and composing `messageCount`/`cartTotal`/`orderStatus` per participant would be ~150 extra graph queries per presenter poll at 50 participants, outside R10's budget and unprofiled by S0's §8. **Reversal trigger:** if the demo genuinely needs activity stats, they are **one aggregate roster query** designed by `graph-dba` as a §18.3 append, never a per-participant fan-out), `POST /shop/api/presenter/reset-all` with §4.8's stop-intake-then-drain quiesce. | `falkor-chat/server/falkorchat/storefront.py`, `.../storefront_api.py`, `falkor-chat/server/tests/test_storefront_api.py` | `presenter_login(key) -> token`, `list_participants()`, `reset_all()` | A participant token is refused on every presenter route; a wrong key is refused and counted; **an *unset* presenter key authenticates nobody: `presenter_configured` is checked *before* any `compare_digest`, asserted with a `presenter_key=""` storefront answering `403` to an empty submitted key** (`docs/reviews/salesperson-ui-impl.md` `## Pass 6`, S6-2 — the guard S6 delivered and tested is useless unless this row tells S10 to call it); **S6's constant-time tripwire pair is extended to `presenter_login`** — a `hmac.compare_digest` spy asserting exactly one call with the expected arguments, plus `assert "==" not in body` (same pass, S6-4: the shipped tripwire inspects `resolve_token` only, and `presenter_login` is the other `compare_digest` site); `reset-all` invalidates every participant token **but not the presenter's**, and clears the presenter's own conversation too; **the same §7 (a)–(d) quiesce conditions as S7**, applied to `reset_all`'s stop-intake-then-drain path; roster excludes non-participant `User`s **and its field set is exactly §5.2's four keys** — asserted, not assumed; **`unscopedCount == 0` returns no `incomplete` field at all**, not `incomplete: false`; **F8, both orderings** (§4.8): a FalkorDB socket `TimeoutError` returns **`504 reset_state_unknown`** after re-reading state, never the quiesce `503` — **and a stub whose re-read *also* raises `TimeoutError` still returns `504`, with no state body, never a `500`**; **§5.2's anomaly contract holds in full** | `coder` | **after S9** |
| **S11** | **Demo bring-up script** — `falkor-chat/scripts/start_demo.sh`, which **first pins `FALKORCHAT_WS_ID=demo`** (overridable, but never left to `config.py`'s `"acme"` default — that is the repo's populated dev/demo workspace; §4.9 move 2). Then: FalkorDB → `bootstrap_schema.sh "$FALKORCHAT_WS_ID"` at `EMBEDDING_DIM=1024` → `seed_demo.sh "$FALKORCHAT_WS_ID"` → `seed_catalog.sh` → `seed_salesperson.sh "$FALKORCHAT_WS_ID"` → preflight `verify_salesperson.sh` + `verify_catalog.sh` → build the SPA → uvicorn with `FALKORCHAT_ENABLE_AGENT=1`, `FALKORCHAT_WORKFLOW_ENABLED=1`, `FALKORCHAT_TRIGGER_DEF_KEY=salesperson`, `_VERSION=v7`, `FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH=0`, `FALKORCHAT_STOREFRONT_ENABLED=1`, `FALKORCHAT_STOREFRONT_DIR`, and a **non-empty** `UVICORN_ARGS` so `--reload` is off. Every seed **and verify** script gets the workspace **explicitly**, even though the pin makes its default correct — defence in depth, not a load-bearing requirement. `seed_workflows.sh` is deliberately **not** run: this demo needs neither `triage` nor `access-request`. Add the script-table row to `falkor-chat/AGENTS.md`. | `falkor-chat/scripts/start_demo.sh` (new), `falkor-chat/AGENTS.md` | — | From a cold box the script yields a reachable `/shop` with a working join **and a working first agent turn**, against `ws:demo` and not `ws:acme` (assert the resolved workspace in the startup banner); it fails loudly and specifically when Node, the bundle, FalkorDB, a def, the catalog or the demo `Agent` is missing | `devops` | **after S5, S8** |
| **S12a** | **Session + API client + routing.** **The three shared entry files and their mount slots (`main.tsx`, `App.tsx`, `index.css`) are this row's *first* deliverable** — S12b, S12c, S12d, S13 and S14 all block on them, so a half-finished S12a stalls five steps. Then: **§5.3's credential & session contract in full — C1–C13, stated there and deliberately not restated here** (the two credentials and their storage keys, per-credential `401`/`403` dispatch, the per-path `504` re-read, the two `409`s, the post-reset language step, `503`, the polling cadence — including the shared-constant and single-timer rules, which is why no cadence figure appears in this row); **typing** `reset-all`'s `incomplete`/`unresolved` response so **S12d** can render it (§5.2), TanStack Query as the polling layer, route shell for join / chat / presenter. | `salesperson/src/api/**`, `salesperson/src/session/**`, `salesperson/src/routes.tsx`, **`salesperson/src/{main.tsx,App.tsx,index.css}`** | `useSession()`, `useShopState()`, `apiClient` | Join → chat round-trips against a live server; **each of §5.3's C1–C13 has a test that goes red when the rule is broken, not merely green when it is kept** (C1 alone is emergent — no client can satisfy C2 and C3 together with a global handler); **and — the guard for the *mis-ruled* half of the defect class (§5.3 C13's residual) — each rule's test must enumerate by name the routes its rule spans, driving every one of them, so a rule whose domain was widened without widening its content fails here rather than in a review pass**: C4 drives all **five** writing routes, C9 both of its actions on both of its trigger kinds, C11 all six `(route, field)` cells, C2/C3 every route carrying their credential. A rule's route list is read off §5.3's route-class and cross-cutting tables, and a test that covers a proper subset is a failing test, not a partial one. Specifically: a participant-route `401` returns the participant view to join **while `/shop/presenter` stays mounted and the presenter credential stays in storage** (**C3** — drive it the way `reset-all` does, by invalidating *only* the participant token, so a global `401` handler fails); a presenter-route `401`/`403` returns only to key entry with the participant session intact, **and a `403` from `POST /shop/api/presenter/session` reports a bad key in place without clearing anything or navigating** (**C2**, both halves); a `409 TurnInProgress` retains the composer text and re-enables send at `turn.state === 'idle'` (**C6a**); **a `409 unscoped_participant` on the same route surfaces as a failure — no retry, no send re-enable, no language-step navigation — so a handler that dispatches on the `409` rather than on the error body fails** (**C6b**); a `503` on either reset keeps the credential and the view and offers a **retry control**, reporting *nothing changed*, **while a `503` arriving on a poll tick of `/state`, `/messages` or `/presenter/participants` renders a staleness indicator and offers *no* control — asserted on both branches, so a single undifferentiated `503` handler fails** (**C9**, both actions); a reset-mine `200` keeps the credential and lands on the language step, not join (**C7**); **`504` — the test names and drives all five writing routes, and never reports "nothing changed" on any of them**, asserted on **both** a named `<op>_state_unknown` body and a body-less proxy-style `504`, and on a browser fetch timeout (**C4**): reset-mine re-reads `/state`; **reset-all's re-read is asserted on the URL the client actually requests — it must be `GET /shop/api/presenter/participants`, so a client wired to `/state` for both paths fails rather than passing through the participant `401` path** (**C4/C5**); `/order/advance` re-reads `/state` and re-renders the order; **`POST /messages` re-reads `/messages` *and* `/state` and reconciles — with the message present and `turn.state === 'idle'` it must report the turn as lost and re-enable send, so a client that re-reads only `/messages` and leaves the participant waiting fails**; and **`POST /shop/api/session` performs *no* re-read at all** (there is no credential) and renders the "your join may not have completed — join again" report, so a client that attempts any authenticated call on that branch fails; **a `404` and a `409` from `POST /shop/api/order/advance` re-read `/state` and re-render the order while the participant stays signed in — a handler that routes either through the `401`/`404` participant path fails, because it logs them out for a stale button** (**C10**); a `422` carrying `field: displayName`, `text` **or `key`** shows an in-place field error and clears nothing, while one carrying `field: language`, `limit` or `transition` goes to the dev surface and is **not** retried — **asserted on the `field` value with the route held constant, so a route-keyed implementation fails** (**C11**); **no automatic retry fires anywhere except the one-shot catalog fetch — counted, with a `401` on `/state` dispatching to C3 on the *first* response rather than after a backoff ladder, and a `422` on `/messages` issued exactly once** (**C12**); **a reset mutation fired offline fails into C9's path rather than being queued and auto-resumed** (**C12**'s `networkMode: 'always'`); **an injected response no rule covers — a `418` on `/state` will do — renders the explicit "unhandled response" failure naming route and status, and clears no credential** (**C13**, the guard that fails when a fall-through silently swallows it); **a `5xx` from either reset fires no automatic retry — asserted by counting requests, since the browser is the one layer §4.8's premise and S8's call-count test do not reach** (**C12**); **both polling hooks read one shared exported constant — asserted by changing that constant in the test and observing both intervals move, so two literals fail — and the catalog query is fetched once across a multi-tick run** (**C8**). **For C3 and C4 alike, assert the intercepted request and the stored credentials, never the rendered outcome** — a global `401` handler and a wrong re-read endpoint each produce the *right* rendered outcome for the wrong reason, which is exactly how both defects survived four review passes; **`App.tsx` exposes an i18n-provider slot and a layout-shell slot, and `index.css` a Tailwind layer entry, each with a no-op default, so S12b and S12c mount into them without editing any shared entry file** (verified by S12b/S12c touching none); `npm test` green | `frontend-engineer` | **after S5 and S8** |
| **S12b** | **Mobile layout shell**, mounted into S12a's layout slot — **edits no shared entry file**. Sticky header with cart/order/profile icon buttons, bottom-sheet overlays, safe-area insets, no horizontal scroll at 360 px; **AC-5's participant half — the participant's own reset control in the profile sheet's *chrome* (`components/sheets/`, S12b's own subtree — **not** S14's profile card in `views/Profile*`), behind a confirm step**; the Playwright mobile project. | `salesperson/src/layout/**`, `salesperson/src/components/sheets/**`, `salesperson/tests/e2e/**`, `salesperson/playwright.config.ts` | — | Playwright at 360×740 and 390×844 shows no horizontal overflow and legible type; sheets open/close by icon; **the reset control is present, and confirming it calls `POST /shop/api/reset` and returns the client to the language step with the previous language pre-selected (§4.8; the client-side rule is **§5.3 C7**, built by S12a — S12b renders it), asserted on rendered state rather than on the fetch** | `frontend-engineer` | **after S12a**, ‖ S12c |
| **S12c** | **i18n**, mounted into S12a's provider slot — **edits no shared entry file**. `react-i18next` wiring, the three locale bundles, locale-aware currency/date formatting, and the join-screen language chooser feeding `POST /shop/api/session` — **seeded from `GET /shop/api/health`'s `locales`, which nothing consumed before**. That deletes §5.3 C11's config-drift trigger at source rather than handling it: the chooser can then only offer values the server accepts, so the UI-supplied `language` `422` becomes unreachable instead of merely well-rendered (C11's branch remains as the defence-in-depth behind it). | `salesperson/src/i18n/**`, `salesperson/src/locales/{en,pt-BR,es}.json` | `t()`, `useLocale()` | All three bundles complete (no missing-key fallbacks in a key-coverage test); chosen locale reaches the join request; UI chrome switches | `frontend-engineer` | **after S12a**, ‖ S12b |
| **S12d** | **Presenter view** (AC-5's UI half), mounted into S12b's layout shell — **edits no shared entry file**. The roster table over `GET /shop/api/presenter/participants` (one row per participant, **§5.2's four keys** — no activity data, see S10); the **reset-everyone** control behind a confirm step; and the rendering of `reset-all`'s `incomplete: true` / `unresolved` body as a named list of participants whose state is still live (§5.2). Presenter key entry, token storage and response typing stay in S12a — this row **renders**. | `salesperson/src/views/Presenter*`, `salesperson/tests/e2e/presenter.spec.ts` (its own spec file, not S12b's) | — | **The roster renders, not merely routes:** with three participants provisioned and one holding cart items and an order — **that activity is the point of the fixture and must not be "simplified" away: a participant who *has* data worth showing is the negative control for the four-key contract, proving the roster shows name and language and nothing more even then** — `/shop/presenter` shows three rows carrying each participant's display name **and language** (§5.2's roster keys — the roster carries no activity data, see S10), **asserted on rendered text rather than on the fetch**; an empty roster shows an explicit empty state, not a blank panel; a `reset-all` response carrying `incomplete: true` and two `unresolved` ids renders both ids and does **not** read as a clean sweep; the reset-everyone control requires a confirm step | `frontend-engineer` | **after S12b, S12c**, ‖ S13, S14 |
| **S13** | **Chat view** — transcript (`textContent` only, **no** `dangerouslySetInnerHTML`), optimistic send, thinking/queued indicator driven by `turn`, **welcome turn — the join response's `welcome` line (§5.2), never a greeting the client composes**, error/retry, autoscroll-when-at-bottom (mirroring `web/app.js`). | `salesperson/src/views/Chat*`, `salesperson/src/components/message/**` | — | A scripted 5-turn conversation renders correctly; a queued turn shows its position; agent-emitted markup renders as literal text | `frontend-engineer` | **after S12b, S12c**, ‖ S14 |
| **S14** | **Cart / order / profile / catalog panels** (FR-8/9/10/11 parity per §2.4). Cart lines + running total + empty state; profile card with em-dash placeholders; catalog grid with image-or-text-only cards; order card with a status chip, `cancel` as an ordinary customer action and **`fulfill`/`deliver` inside a visually distinct "demo controls" affordance labelled as a warehouse simulation** (§4.6). Sources the ~15 stock images and records their licence in `salesperson/README.md` (OQ-6). | `salesperson/src/views/{Cart,Order,Profile,Catalog}*`, `salesperson/public/products/**` | — | Panels match §2.4's parity table; a product **with** an asset renders an `<img>` and one **without** renders text-only with no `<img>` in the DOM (both asserted) | `frontend-engineer` | **after S12b, S12c**, ‖ S13 |
| **S15** | **Test suites & AC evidence** — the load harness (`load_demo.py`, stub-LLM and live-LLM modes, latency percentiles by route class, automated cross-participant isolation assertion on every response, and §6.4's queue-depth headroom check under `reset_all`), the live language-adherence run, the measured AC-8 run, and the mobile Playwright pass. Deliverable is a versioned test plan + report. | `salesperson/scripts/load_demo.py` (new), `docs/test-plans/salesperson-ui.md`, `docs/test-reports/salesperson-ui-report.md` | — | Every AC has recorded evidence; AC-3, AC-8 and AC-9 carry measured numbers, not assertions; **the report states plainly where AC-3's literal wording is not met** (§6.4); the `reset_all`-under-load run records the **observed** queue depth against §6.4's cap | `qa-engineer` | **after S11, S13, S14, S12d** |
| **S16** | **Docs close-out.** Root `AGENTS.md` (new `salesperson/` bullet, new `deprecated/` bullet, component-docs table row, "Working in this repo" bullet); root `docs/HISTORY.md`; `falkor-chat/README.md` + `AGENTS.md` **+ `docs/SERVER.md`** (the `/shop` surface, the storefront deployment's un-mounted dev surface, new env vars — `SERVER.md` because §4.1 cites its §1.4 as the documented REST surface this work leaves alone, while §4.9 changes what the process serves, so the citation goes stale unless it is updated here); `salesperson/{README,AGENTS}.md` final pass. **The `claude/frontend-engineer/frontend-engineer.md` refresh is NOT in scope** — an agent edit must land with its `kaizen/{plan,history}.md` and `claude/README.md` in the same change (`claude/AGENTS.md`), which routes to **`cobb`**; `teco` dispatched it as U6. | root `AGENTS.md`, `docs/HISTORY.md`, `falkor-chat/README.md`, `falkor-chat/AGENTS.md`, **`falkor-chat/docs/SERVER.md`**, `salesperson/{README,AGENTS}.md` | — | The command below returns **zero** matches (verified today it returns exactly the two `claude/frontend-engineer/frontend-engineer.md` lines U6 owns, and nothing else) | `coder` | **last** |

**S16's acceptance command.** v1.0's `rg -n 'salesperson/' --glob '!docs/**'` returns **36 matches
at `4bb96e1`** — `--glob '!docs/**'` contains a slash so it is root-anchored and excludes only the
root `docs/` tree — and under OQ-3's rename it can never be clean, since both the new component and
`deprecated/salesperson/` match the pattern. Replaced with a check that targets the retired app's
*modules* and is written in portable `grep` (`rg` is a shell function on this box, not reliably on
a delegate's `PATH`):

```bash
# Leading `!` is load-bearing: grep exits 1 on "no matches", so the bare command
# reports a FALSE FAILURE on success under `set -e`. Negate it, or capture the
# output and test for empty — never wire the bare form into a script.
! grep -rEn --exclude-dir=.git --exclude-dir=deprecated --exclude-dir=docs \
            --exclude-dir=kaizen --exclude-dir=node_modules \
            --exclude-dir=.venv --exclude-dir=dist \
  'salesperson/(chatbot|cart|customer_profile|session_manager|diagnostics|agent|graph|cypher|prompts|utils_common)\.py' .
```

(`.venv`/`dist` cannot match this pattern — they are excluded so a delegate copying the idiom to a
broader check inherits the right shape.)

### 5.2 The `/shop/api` surface (S8/S9/S10)

All participant routes require `Authorization: Bearer <participantId>.<token>`; all presenter
routes require `Bearer presenter.<presenterToken>`. **No route accepts `ws`, `threadId`,
`customerId` or `orderId` from the client.** Every bounded body and query below is a Pydantic model
(§4.2, built by S8), so a violation answers **`422`**, not `400` — stated per route because §5.3's
completeness table is the source of truth for the response set and this column is its prose view.
**Three responses are omitted from the rows below on purpose** — `503 graph_unavailable`,
`503 graph_read_timeout` and `504 <op>_state_unknown` are produced by S8's typed error handlers
rather than by any route's own body, and are listed once in §5.3's cross-cutting table rather than
repeated per route. **Which of the three a given route can produce is decided by its class**, and
the eleven routes are classified in §5.3's **route-class table** — `writes` (all three are
reachable, and the `504` carries that route's own `<op>` token and its own re-read endpoint),
`reads-only` (`503 graph_unavailable` and `503 graph_read_timeout` only — a read that times out
changed nothing), `no graph access` (**none of the three**, because no query is issued). That table
is the input to S8's `{registered handlers} × {routes}` gate; v1.16 said "every route" in all three
cases and the gate was therefore not computable.

| Route | Body / query | Returns |
|---|---|---|
| `GET /shop/api/health` | — | `{status, storefrontEnabled, locales}` |
| `POST /shop/api/session` | `{displayName ≤ 60, language ∈ locales}` | `{participantId, token, displayName, language, welcome}` — **`welcome` is specified under *The join greeting* below** · **`422`** on a bounds/enum violation (§5.3 C11) · **`503 demo_not_seeded`** when the demo `Agent` is absent from the workspace — `ensure_participant` wrote **nothing at all**, so this is a *nothing changed* `503` (§5.3 C9) naming `./scripts/seed_demo.sh`, and **join is the only route that can produce it**: `DemoNotSeededError` is raised from `Storefront.join` and from nowhere else (`falkor-chat/server/falkorchat/storefront.py`), which no other route calls. **It is not unreachable and must not be left to escape as a bare `500`** — §4.9's readiness preflight asks the same question at *boot*, which bounds the mis-seeded-deployment case and not the graph being altered out of band afterwards (the same posture as `409 unscoped_participant`: a graph can be unhealthy in ways the storefront did not cause) · **not idempotent** — a lost response can leave a ghost participant whose token nobody holds (§5.3 C4's join case, R12) |
| `GET /shop/api/state` | — | `{profile:{name,deliveryAddress}, cart:{items[],total}, order:{orderId,status,lines[],total}\|null, turn:{state,queuePosition}}` |
| `GET /shop/api/messages` | `?since=<ms>&limit=<1..200>` | message rows (participant's own thread, server-resolved) · **`422`** on an out-of-range `limit` (§5.3 C11) |
| `POST /shop/api/messages` | `{text ≤ 2000}` | the posted row · **`409 TurnInProgress`, nothing written**, when that participant already has a turn in flight · **`422`** when `text` exceeds the bound (§5.3 C11) |
| `GET /shop/api/catalog` | — | `[{productId,name,category,price,imageUrl\|null}]` |
| `POST /shop/api/order/advance` | `{transition ∈ {fulfill,deliver,cancel}}` | `{orderId,status}` · `409` stale CAS · `404` no order of theirs — **both ordinary stale-button outcomes, never auth failures (§5.3 C10)** · **`422`** on an unknown `transition` (§5.3 C11) |
| `POST /shop/api/reset` | — | `200 {threadId, language}` — **the participant's token survives** (§4.8); the client returns to the language step, not the full join screen · `503` on quiesce timeout, nothing changed · **`409 unscoped_participant`** when the repository reports `scoped=false` — nothing was reset and nothing will be until the graph is repaired, never a `200` · `404`/`401` on zero rows (indistinguishable from an already-deleted participant) · **`504 reset_state_unknown`** on a FalkorDB socket timeout — the delete may have committed, so re-read state and report from the graph, never "nothing changed" (§4.8 F8) · a `Thread` UNIQUE violation propagates as `5xx` and is **never retried**: this route is the one that can raise it, because reset-mine **re-mints** a thread (`falkor-chat/docs/QUERIES.md` §18.4) |
| `POST /shop/api/presenter/session` | `{key}` | `{token}` · `403` on a bad key (rate-limited) · **`422`** on a missing/blank `key` (§5.3 C11) |
| `GET /shop/api/presenter/participants` | — | `[{participantId,displayName,language,joinedAt}]` — a **subset of the delivered `list_participants` projection** (`falkor-chat/docs/QUERIES.md` §18.3: `participantId, displayName, channelId, threadId, language, joinedAt`); `channelId`/`threadId` are server-side ids no client needs (§4.3). **No activity stats** — see S10 |
| `POST /shop/api/presenter/reset-all` | — | `{clearedParticipants:<n>}`; every **participant** token invalidated, the presenter token is not · when the status row reports `unscopedCount > 0`, **`200` with `incomplete: true` and `unresolved: <unscopedIds>`** — not an error, but it must not read as clean · `unscopedCount == 0` is the normal path: **`200` with no `incomplete` field at all** — not `incomplete: false` · **`504 reset_state_unknown`** on a FalkorDB socket timeout (§4.8 F8) · **any unmapped graph error escaping the route propagates as `5xx` (a bare `500`) and is never retried** — *that* is what §5.3's `5xx` row on this route means, and the row is written on that producer rather than on the `Thread` UNIQUE violation, which **reset-all structurally cannot raise**: this query re-mints nothing (`falkor-chat/docs/QUERIES.md` §18.5), so the UNIQUE violation belongs to reset-mine alone (`docs/plans/salesperson-ui-graph.md` §12's anomaly contract). Naming it here would be a declared row with no producer, which S8's gate is specified to fail on |

**The join greeting (`welcome`).** A server-minted line, one per configured locale, interpolating
the display name the participant just typed. It is the **only participant-visible display string the
API supplies** — every other string the participant reads belongs to S12c's bundles and S13's views
— and it is server-side for one reason: **the join response is minted before the SPA knows its
language was accepted**, since `language ∈ locales` is checked on this very request (a rejection is
the `422` above), so the client cannot pick the line until it already holds the response.

| `language` | `welcome` |
|---|---|
| `en` | `Welcome to the store, {name}.` |
| `pt-BR` | `Bem-vindo à loja, {name}.` |
| `es` | `Bienvenido a la tienda, {name}.` |

`{name}` is the accepted `displayName`, interpolated verbatim — the client renders it as text
(§4.2's `textContent`-only rule), and the server escapes nothing.

**Fallback — a `language` with no line in the table answers the `en` line, and that is live
behaviour, not dead code.** The table covers `config.STOREFRONT_LOCALES`'s default
`("en", "pt-BR", "es")`, so the fallback is reachable exactly when a deployment widens the set
through `FALKORCHAT_STOREFRONT_LOCALES` — a real operator knob — and the join then **succeeds** in
that locale carrying the English greeting rather than failing. **It therefore takes a test of its
own** (join under a configured locale absent from the greeting table ⇒ the `en` line, with the name
interpolated): without one, replacing the fallback with a direct index turns that deployment's join
into a `500` and nothing goes red (`docs/reviews/salesperson-ui-impl.md` `## Pass 10`, mutant M-O).

**It is not session state.** §5.3's participant credential stores the response's other four keys and
deliberately not this one: the greeting belongs to the join moment and is not something the client
re-reads.

### 5.3 The client's credential & session contract (S12a)

*The client half of §5.2. It lives here, in one place, because it previously lived as five words in
a scope column plus three done-conditions — and two defects came out of that gap (the wrong `504`
re-read endpoint; undifferentiated `401` routing), which interact. **S12a builds all of it**; S12b,
S12d, S13 and S14 consume it and re-implement no part of it. This section states the **contract
only** — the decisions behind it are §4.3 (two credentials), §4.4 measure 1a (the `409`), §4.8
(what each reset does) and §5.2 (the wire) — and it changes none of them.*

**The two credentials.**

| | Participant | Presenter |
|---|---|---|
| **Storage key** | `salesperson.participant` | `salesperson.presenter` |
| **Stored value** | `{participantId, token, displayName, language}` — the `POST /shop/api/session` response **minus `welcome`**, which is a one-shot greeting rather than session state (§5.2) | `{token}` — the `POST /shop/api/presenter/session` response |
| **Header it produces** | `Authorization: Bearer <participantId>.<token>` | `Authorization: Bearer presenter.<presenterToken>` |
| **Sent on** | every route except `GET /shop/api/health` and the two `session` routes | the **two authenticated** presenter routes (`presenter/participants`, `presenter/reset-all`) — **not** `presenter/session`, which *mints* the token and cannot carry it |
| **Minted by** | the join flow at `/shop` | key entry at `/shop/presenter` |
| **Cleared client-side by** | a `401` on a participant route (C3) | a `401`/`403` on a presenter route (C2) |
| **Invalidated server-side by** | `reset-all` — **not** the participant's own reset (§4.8) | nothing the storefront does |

**Storage medium: `localStorage` for both keys — and this is a decision this section *makes*, not
one it records.** §4.3 said only "stored under separate keys". `localStorage` over
`sessionStorage` on two discriminators, both of which the design actually needs. **Restart and
tab survival:** §4.3's authoritative-registry argument (R7) exists so that a participant never loses
their **cart and order** to a session boundary, and `sessionStorage` reintroduces exactly that loss
on a closed tab — a realistic phone gesture. **Usability — one browser, two tabs, no re-authentication:**
`sessionStorage` is scoped per tab, so opening `/shop/presenter` in a second tab would demand the key
again; `localStorage` is shared origin-wide and does not. (This is a convenience property, **not** a
C3 requirement — C3 only asks that clearing one credential leave the other alone, which separate keys
give in either medium. R7 above carries the decision on its own.) *The standing XSS objection does **not** decide between the
two* — any script executing in the page reads either medium equally, so §4.2's `textContent`-only
rendering is a real mitigation but not a discriminator here. Under FR-1's controlled-demo scope the
choice sits inside R6's already-accepted residual rather than adding a new one; note also that the
presenter **key** is never stored — only the token exchanged for it (the table's `{token}`).

**The rules — C1…C13.** S12a's done-conditions cite these by number. **C13 is the one that does not enumerate** — it is the runtime guard against a `(route, response)` no rule covers. It is *not* a guard on whether the other twelve are **right**: see C13's own residual paragraph.

- **C1 — every `401`/`403` is dispatched *per credential*, never by one global handler.** Which
  credential the failed request carried decides what is cleared and where the user lands. A single
  app-wide `401 → rejoin` is the defect this rule exists to prevent, and it is not hypothetical —
  see C3. **C2 and C3 are stated by *route* rather than by credential, and that is equivalent here
  only because the credentials table's "Sent on" rows make route → credential a function:** each
  route carries exactly one credential, so the route the request went to identifies the credential it
  carried. C1 is the governing rule; C2 and C3 are its two cases, re-keyed onto the axis an
  implementer actually branches on. If a future route ever accepted either credential, that bijection
  breaks and C2/C3 must be re-stated on C1's key.
- **C2 — a presenter-route `401`/`403` clears only the presenter credential** and returns only the
  presenter view to key entry. The participant session and the participant's current view are
  untouched. **Two responses share these codes and mean different things** — the same trap C6b
  names, so keep them apart: on the **two authenticated** presenter routes it means *the presenter
  session is gone* (clear the credential, return to key entry); on **`POST /shop/api/presenter/session`**
  a `403` means *the key you just typed is wrong* — there is no credential to clear and the user is
  already on key entry, so the rule is to report the bad key in place. **A deployment with no key
  configured answers that same `403` deliberately** (S10 checks `presenter_configured` before any
  comparison, because `compare_digest("", "")` is `True`): the two *meanings* differ, but the client
  cannot and must not distinguish them — telling the LAN that no key is configured is worse than
  telling it the key was wrong — and the **action is identical**, so one row is correct under this
  section's grouping licence. The operator's signal is the server log, not the response. S10's rate-limiter is a
  **"fixed delay + attempt counter"** (its wording, quoted rather than paraphrased — it does *not*
  back off progressively), so the client must not assume growing backoff and must not add retry
  logic of its own around it.
- **C3 — a participant-route `401` clears only the participant credential** and returns only the
  participant view to join. **It must not clear the presenter credential and must not navigate away
  from `/shop/presenter`.** This fires on *every successful* `reset-all`: the sweep invalidates the
  presenter's own **participant** token while their presenter token survives (§4.3, §4.8), so their
  background `GET /shop/api/state` poll starts `401`-ing within one 2 s tick. A global handler then
  bounces the whole SPA to the join screen and yanks the presenter out of the demo mid-reset —
  precisely what §4.3's "they can keep driving the demo through the reset" forbids, on the most
  visible action in AC-5.
- **C4 — after a `504`, re-read from the endpoint the failed path's *surviving* credential can
  reach — and C4's content is *per writing route*, one case each, never a generalisation.** (§4.8's
  F8 is the server half; this is the browser's.) The branch keys on **the status
  code, not the error string**: §3 puts TLS behind a reverse proxy, which emits its own bare
  `504 Gateway Timeout` with an HTML body meaning exactly the same thing here (*unknown, re-read
  state*) and carrying no `reset_state_unknown` to match on. A fetch that times out in the browser
  takes the same branch. Never report "nothing changed".
  **The rule spans exactly the five writing routes of the route-class table below, and each gets its
  own case here.** v1.16 generalised F8 from "either reset" to "every write" and extended C4's
  *domain* — the cross-cutting `504` row, which said "every route that writes" — without extending
  its *content*: it listed four cases, so **join was missing entirely** and one of the four present
  (`POST /messages`) named a re-read that cannot answer the question. That asymmetry — widening the
  domain is one edit, widening the content is five — is the
  mis-ruled residual C13 does not catch, and the reason S12a's C4 test must **enumerate all five
  routes by name**.
  - after **`POST /shop/api/reset`** (reset-mine) → `GET /shop/api/state`. The participant credential
    survives its own reset (§4.8), so the call answers with state.
  - after **`POST /shop/api/presenter/reset-all`** → **`GET /shop/api/presenter/participants`**. If that
    delete committed, the participant credential is already dead, so `/state` would answer `401` rather
    than state — while the roster is what the surviving **presenter** credential can still reach, and is
    the thing that actually says whether the sweep happened.
  - after **`POST /shop/api/order/advance`** → `GET /shop/api/state`. The write may have left the order
    advanced; `/state` carries the order block, so the rendered order is the report.
  - after **`POST /shop/api/messages`** → **`GET /shop/api/messages` *and* `GET /shop/api/state`, and
    the two must be reconciled.** The post path is `services.post_message` **then** enqueue (S9), so a
    query-time timeout fires *during the write*, before the enqueue: the overwhelmingly likely committed
    state is **message written, turn never scheduled**. Re-reading only `/messages` shows the message
    present, says nothing about the turn, and leaves the participant waiting forever on a reply nobody
    queued — which is exactly the state §4.4 measure 1a refuses *before* the write ("a written message
    with no reply would sit in the transcript forever"), reintroduced through the back door.
    **The reconciliation:** message present **and** `turn.state === 'idle'` ⇒ *the turn was lost* — say
    so and re-enable send. Message present **and** `turn.state !== 'idle'` ⇒ the turn is running; wait,
    as normal. Message absent ⇒ nothing committed; the composer text is retained and send re-enabled.
    **State the cost of the only recovery the API offers:** re-sending posts a second line, so the
    transcript keeps a duplicate — tell the participant that rather than letting them discover it.
  - after **`POST /shop/api/session`** (join) → **there is no re-read, and there cannot be one.** Join
    writes (`User`+`Channel`+`Thread`+profile, S6), but the response that would have carried the token
    is what was lost — so the client holds **no credential with which to read anything**, and every
    other route answers `401`. The rule is therefore a *report*, not a re-read: **"your join may not
    have completed — join again"**, plus a warning that a stale roster row may appear. There is no
    variant of this case that re-reads, which is why the cross-cutting `504` row could not stay one row.
    **Consequence, decided rather than engineered away (Pass 8 OQ-1):** if the write did commit, the
    graph keeps a `User` with a `tokenHash` nobody holds — a ghost row in the presenter roster owning a
    `Channel` and `Thread`, while the person re-joins as a second identity. **Join stays
    non-idempotent and the artifact is accepted** (R12): the alternative is a client-supplied
    idempotency nonce, which §5.2's invariant does permit (it bans `ws`/`threadId`/`customerId`/
    `orderId`, not a nonce) but which would reopen **delivered** S6 — a new `join()` parameter, a
    uniqueness constraint and an S0 amendment — to close a window that requires a FalkorDB socket
    timeout during the one write a participant makes before they hold any state. The ghost costs one
    roster line, is self-healing (`reset-all` sweeps it: it is a participant `User`), and S12d renders
    it as an ordinary participant who never speaks. **Reversal trigger:** if join ever acquires a side
    effect that matters beyond the roster (payment, external provisioning, a per-participant quota), or
    the storefront is used outside a controlled demo, the nonce is the answer and it lands as its own
    step rather than as an S6 amendment.
- **C5 — a `401` from `/state` following a `504` on reset-all is *evidence the sweep committed*,**
  not an ordinary auth failure to swallow; it still routes through C3. **A client that wires
  `/state` for both C4 paths appears to work and is standing on luck** — the `401` falls into the
  participant-side handler and returning that view to join *is* the right outcome for a committed
  reset-all. Behaviourally acceptable, but incidental rather than designed. C3 and C4 are what make
  it designed; anyone changing either needs to know the other is leaning on it.
- **C6a — `409 TurnInProgress` retains the composer text** and re-enables send when `turn.state`
  returns to `idle` (§4.4 measure 1a, which argues why the refusal is server-side and pre-write).
  Nothing is cleared and nothing navigates.
- **C6b — `409 unscoped_participant` is an *alarm*, and must read as neither success nor busy.**
  It is a different response that happens to share C6a's status code, and **the handler dispatches on
  the error body, not on the `409`** — an undifferentiated `409` handler is the same defect shape as
  the undifferentiated `401` C1–C3 exist to prevent, one code over. It means *nothing was reset and
  nothing will be until the graph is repaired* (§5.2): surface it as a failure, **do not** retry, **do
  not** re-enable send as though a turn had merely been in flight, and **never** navigate to the
  language step as if C7 had fired. `docs/plans/salesperson-ui-graph.md` establishes the branch is
  unreachable on a healthy graph, so this is defence-in-depth — for the one state in which a lying
  client costs most.
- **C7 — after the participant's own reset (`200`), keep the credential** and return to the
  **language step** with the previous language pre-selected, not the join screen (§4.8). This is the
  one *success* response that navigates.
- **C8 — polling is 2 s for `GET /shop/api/state` and `GET /shop/api/messages`; the catalog is
  fetched once.** The 2 s figure is R10's budget basis (§2.3), so both hooks read **one shared
  exported constant, never two literals** — changing it is then a one-line change that re-opens R10
  deliberately rather than a per-view drift that re-opens it silently.
  **Poll the two on one shared interval so their ticks coincide** — this aligns them, it does **not**
  merge them into one request, so R10's two-routes-per-tick budget is unchanged and no combined
  endpoint is implied. It matters because C6a's re-enable is
  driven by `turn.state` from `/state` while the reply itself arrives on `/messages`: on independent
  timers the composer can stay disabled for up to a further poll interval *after* the participant can
  already see the answer. Aligning them collapses most of that window at no design cost. **The
  residual is accepted and is not a bug:** up to one interval of disabled composer after a reply
  lands is the price of C6a's server-authoritative gate, and S13's `turn`-driven thinking indicator
  covers the perception. The alternative — re-enabling optimistically on the client — is **rejected**,
  because it races the server's `409` and re-opens the double-post that §4.4 measure 1a exists to
  close.
- **C9 — a `503` means *nothing changed*, and the client says so. One meaning, but two actions, and
  the action keys on *who asked* — not on the route and not on the source.** **Four** sources share
  the meaning: the **quiesce timeout** on either reset (§4.8 — the reset never reached the graph);
  **`graph_unavailable`** on any route that touches the graph (it could not be reached at all, so
  nothing was sent); **`graph_read_timeout`** on a reads-only route (a query timed out, and a read
  that times out changes nothing by definition); and **`demo_not_seeded`** on join, where the write
  was refused before anything was written (§5.2). Common to all four: keep the credential, keep the
  view, navigate nowhere, and never silently swallow it. The action then splits:
  - **User-initiated request** (either reset, `POST /messages`, `/order/advance`, either `session`
    route, the one-shot catalog fetch) ⇒ **surface a transient refusal with a plain retry control.**
    A retry is safe **because** nothing changed, which is exactly what distinguishes this code from
    C4's `504`. A reset-everyone that quietly did nothing mid-demo is the failure the presenter must
    see. **`demo_not_seeded` is the one source that will not clear on its own** — the retry is still
    safe (nothing was written) but keeps failing until an operator re-seeds, so word it as a
    deployment fault rather than as a passing blip. The *action* is unchanged (keep the view, offer
    the retry), which is why it is one rule and one row rather than two: this section's grouping
    licence asks for one meaning **and** one action, and copy is neither.
  - **Background poll tick** (`/state`, `/messages`, `/presenter/participants` — C8's 2 s cadence)
    ⇒ **a staleness indicator and *no* control.** There is nothing to offer a retry button for: the
    next tick *is* the retry, and a button that duplicates it invites a second request during an
    outage, which is C12's amplification argument one rule over. What the participant or presenter
    needs is to know the view has stopped updating and since when — a poll that quietly freezes is
    the failure nobody notices until it matters.

  **Why this split is stated rather than left to taste:** it is the same defect as P8-3 one rule
  down — three sources with one meaning grouped under one rule, where the *action* was not in fact
  common. The discriminating axis here is genuinely not the route (`/state` takes both branches: the
  poll and a user-initiated refresh), which is why C9 may still be one rule and one table row where
  C4 may not.
- **C10 — `/order/advance`'s `404` and `409` are ordinary order outcomes, not auth failures and not
  alarms.** `404` is *no current order of theirs*; `409` is *the CAS guard did not match* (§5.2) —
  both are what a **stale button** produces when the order already moved on. Neither can mean
  "someone else's order": §4.6 takes the order id from server-side state and never from the request
  body. So **re-read `GET /shop/api/state`, render the current order, clear no credential and
  navigate nowhere.** Routing this `404` through C3 would **log a participant out for pressing a
  stale `cancel`** — the concrete reason the table below is keyed on *(route, response)*: `404` means
  something different here than on `POST /shop/api/reset`, and one row must not span both.
- **C11 — `422` dispatches on the *field*, not on the route.** Every bounded body and query in §5.2
  is a Pydantic model (§4.2's size-bounded hygiene, built by S8), so FastAPI answers **`422`**, not
  `400`. **The route is the wrong key and v1.15 got two cells wrong by using it** — one route can
  bound both a field a person typed and a field the UI chose, so the discriminator is *who supplied
  the value*, which only the error body carries. (C6b already dispatches on the body rather than the
  code; this is the same move one level down.)
  - **User-supplied — show it next to the field**, keep the input, clear nothing, navigate nowhere:
    `displayName` (`POST /shop/api/session`), `text` (`POST /shop/api/messages`), and **`key`
    (`POST /shop/api/presenter/session`)** — a blank key is a human pressing Enter on an empty box,
    the most ordinary mistake in the presenter flow, and v1.15 wrongly filed it as a client bug.
    The client bounds these three itself so the common case never round-trips; the `422` is the
    backstop, not the mechanism.
  - **UI-supplied — there is no field to blame**, so surface it on a dev-visible surface (console +
    an unobtrusive error state, not a form error the user is asked to fix) and **do not retry**; a
    retry resends the same invalid value: `limit` (`GET /shop/api/messages`), `transition`
    (`POST /shop/api/order/advance`), `language` (`POST /shop/api/session`). **`language` is
    UI-supplied and therefore *not* the user's to fix** — the chooser is S12c's bundle list, so a
    server `locales` narrower than the bundles makes this reachable by **demo bring-up config
    drift**, and showing a field error beside a picker the user cannot change would be the wrong
    render. S12c closes the drift at source by seeding the chooser from `GET /shop/api/health`'s
    `locales` (which nothing consumed before); C11's branch is the defence-in-depth behind it.
  - **This rule needs a body it can key on, and that is a contract, not an accident.** S8 does **not**
    expose FastAPI's raw `RequestValidationError` shape — whose `loc` array is a framework detail that
    a version bump may change — but maps it to the storefront's own stable
    **`{error: "validation_failed", field: "<name>"}`**, one field name per response. C11 branches on
    `field`; S8 owns the mapping.
  - **One field name, out of a report that carries all of them — so the selection rule is part of the
    contract, not an implementation detail.** Pydantic validates every field and FastAPI's
    `RequestValidationError.errors()` returns **all** violations at once. S8 takes the **first entry**
    of that list, which for a single request model is **declaration order** — so `displayName` before
    `language` on `POST /shop/api/session`, deterministically, and C11's highlighting cannot depend on
    dict ordering or on which field the user fixed last. And **`field` is the client-facing name**,
    the last element of the error's `loc` (`displayName`, never `body.displayName` or `query.limit`):
    the client knows its own input names, not FastAPI's parameter-source prefixes. A response
    therefore never carries two field names, and a user-supplied violation co-occurring with a
    UI-supplied one resolves to whichever is declared first — which is why the client bounds the
    user-supplied fields itself (above), so the common case never round-trips at all.
- **C12 — no automatic retry anywhere, except the one-shot catalog fetch — and the rule is keyed on
  the *transport*, not on the status.** §5.2's absolute — "a `Thread` UNIQUE violation propagates as
  `5xx` and is **never retried**" — is held at the library layer by §4.8's stated premise and at the
  application layer by S8's call-count test. **Neither reaches the browser**, which is the layer that
  actually re-issues the request. So surface a `5xx` as a failure that says the graph needs attention
  (§5.2: nothing was reset, and nothing will be until it is repaired), and offer no retry control.
  **But status is the wrong axis to state the rule on**, because the retry that fires is blind to it:
  - **Polling queries** (`/state`, `/messages`, `/presenter/participants`) pin **`retry: 0`**.
    `refetchInterval` *is* the retry — it re-issues every 2 s regardless — so a library retry buys no
    resilience and costs two things that matter: C3's `401` dispatch is delayed by the backoff
    (~7 s on the default ladder) during its headline scenario, every successful `reset-all`; and R10's
    two-requests-per-tick budget becomes up to eight per participant **precisely during an outage**,
    which is amplification when the server is already failing. It also silently retried C11's
    UI-supplied `422`, which C11 forbids — two §5.3 rules contradicting on one cell.
  - **Mutations** (both resets, `/order/advance`, both `session` routes, `POST /messages`) pin
    **`retry: 0` explicitly rather than inheriting it**, so the guarantee is in this codebase rather
    than in a dependency default.
  - **The one-shot catalog fetch** may keep a bounded retry with a **`5xx`-only predicate** — it is
    fetched once, has no `refetchInterval` behind it, and a transient failure there is the one case
    where a retry is the resilience.
  - **The reset mutations additionally pin `networkMode: 'always'`.** Under the default `'online'` a
    mutation fired while the browser is offline is **paused, not failed**, and auto-resumes on
    reconnect (focus-gated) — so the presenter presses "reset everyone", sees nothing at all (no
    error, so C9 never renders), and the sweep executes later, possibly after they have moved on.
    Failing fast into C9's path is the correct behaviour for a destructive action on a LAN demo.
  **Verified, not assumed** — read from the pinned installed source at
  `salesperson/node_modules/@tanstack/query-core` (`@tanstack/query-core@5.102.8`, matching
  `package-lock.json`; a source read, since `node` is absent from the box per R5): mutations default
  to no retry (`mutation.js:81`, `retry ?? 0`); queries default to three (`query.js:240` →
  `retryer.js:89`, `retry ?? (isServer() ? 0 : 3)`) with backoff `min(1000·2^n, 30000)` =
  1 s / 2 s / 4 s (`retryer.js:6-8`); **the retry predicate is blind to status** (`retryer.js:92` —
  it tests the count, never the error), which is why the rule cannot be stated over `5xx`; and offline
  mutations pause and auto-resume (`retryer.js:9-10, 51-52`; `mutationCache.js:103`
  `resumePausedMutations()`). **Reversal trigger — aimed at the query default, which is the one that
  bit:** if a future version changes the query default or makes `shouldRetry` status-aware, re-derive
  this rule; the explicit `retry: 0` pins above are what make that a re-derivation rather than a
  regression.
- **C13 — anything with no rule fails *loudly*.** Any `(route, response)` the client receives that matches no rule above renders an explicit
  **"unhandled response"** failure — naming the route and the status — instead of falling through to
  the nearest handler that happens to match. It clears no credential, navigates nowhere, and retries
  nothing.
  **Why this rule exists, stated plainly because it is the organising idea of this section.**
  Successive passes of this plan closed seven instances of one defect class — a client rule spanning
  server responses that share a status code but not a meaning — by *enumerating* better: first a rule
  list, then a table keyed on response, then a table keyed on `(route, response)`. Enumeration is
  necessary and it made the search mechanical, but it **cannot close the class**, for a reason worth
  quoting from the Pass 7 review: *"unexpressible in the table" is a document property;
  "unhandled ⇒ loud" is a runtime property, and only the second survives §5.2 being wrong.* Every one
  of those seven was a
  response the plan did not know about; a guard that only covers responses the plan knows about can
  never catch the next one. C13 does, and it catches it **in the demo, in front of the person who can
  act on it**, rather than in a review pass. It is deliberately the least clever rule here: a
  fall-through that shouts.

  **What C13 does not do, stated because the residual is real and ships.** C13 detects the
  **absence** of a matching rule. It is silent whenever a rule matches and is **wrong** — the branch
  runs, the client renders something plausible, and nothing shouts. That band is not hypothetical and
  it is not shrinking on its own. Scoring the class honestly, across eight review passes of this
  section:
  - **Unruled — closed.** Seven instances were found, five of them (1, 3, 4, 5, 7) responses the plan
    did not know it could receive. The pair that closes them is structural rather than enumerative:
    S8's error map is **total by type**, so the producible set is bounded by construction, and C13
    makes any survivor loud in the demo. Neither half depends on anyone having enumerated correctly.
  - **Mis-ruled — open, and *fed* by every generalisation.** Pass 8 found three more (C4 silent on
    join — the one writing route whose `504` admits no re-read at all; C4's `/messages` re-read
    confirming the write but saying nothing about the turn, which is the state §4.4 measure 1a
    exists to prevent; the one `504` row grouping five routes that share a meaning but not an
    action) — **all three created by the v1.16 delta that was meant to close the class**, because
    extending a rule's domain is one edit while extending its per-route content is five. C13 stays quiet on all three: a rule matches. The
    server map is innocent: it produces a documented `504`.
  - **The mitigation, labelled accurately: each rule states its own discriminator** ("on the body,
    not the code" — C6b; "on the `field`, not the route" — C11; "on the transport, not the status" —
    C12; "on the status code, not the error string" — C4) — which converts an implicit assumption
    into a visible claim and makes review fast. **That is a review aid, not a guard**: C4's
    discriminator is stated too, and C4 is precisely the rule that broke.
  - **The mechanical guard for this half is in S12a, not here.** Each rule's red-on-break test must
    **enumerate the routes the rule spans** — so C4's test names all five writing routes and a
    missing case fails at implementation time, without a reviewer. That is where the residual is
    carried; it is executable, and it is the reason this plan stops taking review passes and resumes
    at the implementation gates.

**Completeness — this table is the source of truth for the storefront's response set; §5.2's
`Returns` column is its prose view.** v1.15 had that backwards: it said the table certifies against
§5.2, while eight of its rows have no §5.2 counterpart at all (they come from the error map, not from
a route's own body). The direction is now fixed, and with it the **generation rule**, because at this
size a hand-maintained table needs one:

> A row exists for every `(route, response)` the server can produce, and that set is **derived, not
> remembered**: `{responses a route returns itself} ∪ {responses S8's typed error handlers can
> produce on it}`. The second half is enumerable **only because the error map is total by type**
> (S8, below) — which is what makes the S8 gate decidable rather than a reading exercise:
> **{registered handlers} × {routes *that route class permits them on*} ⊆ this table**. The class
> filter is not a refinement of the gate, it *is* the gate — without it the cross product is
> nonsense on the two routes that issue no query. S8 states the gate's two halves in full; this is
> the table they are evaluated against.

**Route classes — the gate's *other* input, and the reason it is computable.** "Every route",
"every route that **writes**" and "every route that only **reads**" are only usable if the plan says
which route is which; v1.16 did not, so the gate above could not actually be evaluated, and its
symmetric half ("a row with no producer fails the step") failed on the two routes that reach no
graph at all. **All eleven routes, classified:**

| Class | Routes | Which cross-cutting responses it can produce |
|---|---|---|
| **writes** (5) | `POST /shop/api/session` · `POST /shop/api/messages` · `POST /shop/api/order/advance` · `POST /shop/api/reset` · `POST /shop/api/presenter/reset-all` | all three — `503 graph_unavailable`, and a query-time `redis.TimeoutError` becomes that route's **own** `504 <op>_state_unknown` |
| **reads-only** (4) | `GET /shop/api/state` · `GET /shop/api/messages` · `GET /shop/api/catalog` · `GET /shop/api/presenter/participants` | `503 graph_unavailable` and `503 graph_read_timeout`; **never a `504`** — a read that times out changed nothing |
| **no graph access** (2) | `GET /shop/api/health` · `POST /shop/api/presenter/session` | **none of the three.** No query is issued, so no typed handler can fire and these routes take **no** cross-cutting row |

Two classification calls the plan owes an argument for, both decided here:

- **`GET /shop/api/health` does not touch the graph — deliberately, and unlike the platform's
  `/health` (`api.py:63`, which pings and answers `503`).** It returns `{status, storefrontEnabled,
  locales}` from `config` and the in-process `Storefront` (`Storefront.locales`, delivered by S6),
  and its two consumers are a liveness probe and **S12c's locale chooser** — which must render on
  the join screen *before* anything else works. Coupling it to the graph would mean a participant
  cannot even see the language list during an outage, and would buy nothing: the storefront
  deployment already carries the platform's graph-pinging liveness at the bare `GET /health`
  (§4.9 move 1, `app.py:338` — `services.ping`, `503` when FalkorDB does not answer). Two liveness
  routes with two different meanings is the intent, not an accident.
- **`POST /shop/api/presenter/session` does not touch the graph either** — the presenter is not a
  `User` (§4.3), the key comparison and the token mint are in-process (S10's `presenter_login`), and
  the attempt counter is in-process and observational. So it is **not** a writing route, and Pass 8's
  "six writing routes" is **five**: the sixth is excluded by classification rather than needing a
  sixth C4 case. `POST /shop/api/session` — the *participant* join — is the one that writes.

**Cross-cutting — produced by S8's typed handlers, on the routes each class above permits.** These
rows are grouped deliberately, and the licence for any grouping in this section is now stated in
full: **a row may span routes only when they share one meaning *and* one action.** Meaning alone is
not enough — v1.16's licence said only "their meaning is route-independent by construction", and the
one row whose *action* was per-route (the `504`, whose action is C4's re-read endpoint) hid two
missing cases and one wrong one under a single line. Where a rule's action varies on an axis that is
**not** the route — C9's user-initiated-vs-poll — the row may still span routes, provided the rule
names that axis explicitly. A row may never span two meanings, and **never two actions keyed on the
route**.

| Response | Where it can arise | Rule |
|---|---|---|
| `503 graph_unavailable` | the **9** routes of classes `writes` + `reads-only` — the graph could not be reached (`FalkorDBUnreachableError`, `redis.ConnectionError`); **nothing was sent** | C9 (action per C9's user-initiated-vs-poll split) |
| `503 graph_read_timeout` | the **4** `reads-only` routes — a query-time `redis.TimeoutError`; nothing changed | C9 (same split; on `/state`, `/messages` and `/presenter/participants` this is normally the poll branch) |
| `504 join_state_unknown` | `POST /shop/api/session` — the write may have committed, **and the token was not delivered** | C4 · **no re-read is possible**; report + join again; ghost roster row accepted (R12) |
| `504 post_state_unknown` | `POST /shop/api/messages` — the message may be written **and the turn never enqueued** (S9 writes before it enqueues) | C4 · re-read `GET /shop/api/messages` **and** `GET /shop/api/state`, reconcile on `turn.state` |
| `504 order_state_unknown` | `POST /shop/api/order/advance` — the transition may have committed | C4 · re-read `GET /shop/api/state` |
| `504 reset_state_unknown` | `POST /shop/api/reset` — the delete may have committed. **Producer is the route itself** (S7 catches its own `TimeoutError`, §4.8 F8), not the typed handler | C4 · re-read `GET /shop/api/state` |
| `504 reset_state_unknown` | `POST /shop/api/presenter/reset-all` — same, **producer is the route itself** (S10) | C4 + C5 · re-read `GET /shop/api/presenter/participants` |
| any unmapped response | — (must not exist once the map is total; C13 is the proof it does not) | **C13** |

Each `504` above is also reachable as a **bare proxy `504`** with an HTML body and no error token
(§3's reverse proxy) and as a browser fetch timeout; C4 keys on the status code precisely so those
take the same branch (P4-3). **The five `504` rows are one per writing route, not one row spanning
five**, because their action differs by route — including one route (`join`) whose action is *not a
re-read at all*. That is the licence above doing its job.

**Per route.**

| Route | Response | Rule |
|---|---|---|
| `GET /shop/api/health` | `200` | — (unauthenticated liveness; **class `no graph access`**, so this route's whole response set is this one row — it takes none of the three cross-cutting rows) |
| `POST /shop/api/session` | `200` | mints the participant credential (table above) |
| `POST /shop/api/session` | `422` `displayName` | C11 · user-supplied |
| `POST /shop/api/session` | `422` `language` | C11 · UI-supplied |
| `POST /shop/api/session` | `503 demo_not_seeded` — the demo `Agent` is absent, **nothing was written**; the only route that can produce it (§5.2) | C9 · user-initiated branch, and the one source that does not clear on its own |
| `GET /shop/api/state` | `200` | C8 (cadence) |
| `GET /shop/api/state` | `401` | C3 |
| `GET /shop/api/messages` | `200` | C8 (cadence) |
| `GET /shop/api/messages` | `401` | C3 |
| `GET /shop/api/messages` | `422` `limit` | C11 · UI-supplied |
| `POST /shop/api/messages` | `200` posted row | — |
| `POST /shop/api/messages` | `401` | C3 |
| `POST /shop/api/messages` | `409 TurnInProgress` | C6a |
| `POST /shop/api/messages` | `422` `text` | C11 · user-supplied |
| `GET /shop/api/catalog` | `200` | C8 (fetched once; the one retry exception — C12) |
| `GET /shop/api/catalog` | `401` | C3 |
| `POST /shop/api/order/advance` | `200` | C10 |
| `POST /shop/api/order/advance` | `401` | C3 |
| `POST /shop/api/order/advance` | `404` no order of theirs | **C10**, *not* C3 |
| `POST /shop/api/order/advance` | `409` stale CAS | **C10**, *not* C6a/C6b |
| `POST /shop/api/order/advance` | `422` `transition` | C11 · UI-supplied |
| `POST /shop/api/reset` | `200` | C7 |
| `POST /shop/api/reset` | `401` credential rejected | C3 |
| `POST /shop/api/reset` | `404` no such participant | C3 |
| `POST /shop/api/reset` | `409 unscoped_participant` | C6b |
| `POST /shop/api/reset` | `503` quiesce timeout | C9 |
| `POST /shop/api/reset` | `504 reset_state_unknown` (or a bare proxy `504`) | **its row is in the cross-cutting table above**, with its re-read endpoint — listed there rather than here so all five `504`s are read together |
| `POST /shop/api/reset` | `5xx` any unmapped graph error — **the `Thread` UNIQUE violation among them**, this being the route that re-mints a thread | C12 |
| `POST /shop/api/presenter/session` | `200` | mints the presenter credential (table above); **class `no graph access`**, so its response set is exactly these three rows |
| `POST /shop/api/presenter/session` | `403` bad key | C2 · second half |
| `POST /shop/api/presenter/session` | `422` `key` | C11 · **user-supplied** |
| `GET /shop/api/presenter/participants` | `200` | C8 (cadence); rendered by S12d |
| `GET /shop/api/presenter/participants` | `401` presenter session gone | C2 · first half |
| `GET /shop/api/presenter/participants` | `403` wrong credential type (a participant token — §6.2's auth matrix) | C2 · first half |
| `POST /shop/api/presenter/reset-all` | `200` clean | C3 · S12d |
| `POST /shop/api/presenter/reset-all` | `200` + `incomplete`/`unresolved` | typed by S12a, rendered by S12d (§5.2) |
| `POST /shop/api/presenter/reset-all` | `401` presenter session gone | C2 · first half |
| `POST /shop/api/presenter/reset-all` | `403` wrong credential type | C2 · first half |
| `POST /shop/api/presenter/reset-all` | `503` quiesce timeout | C9 |
| `POST /shop/api/presenter/reset-all` | `504 reset_state_unknown` (or a bare proxy `504`) | **its row is in the cross-cutting table above** (C4 + C5) |
| `POST /shop/api/presenter/reset-all` | `5xx` any unmapped graph error — **never the `Thread` UNIQUE violation**, which this query cannot raise (§5.2) | C12 |

**How S8's gate reads this table — the two places where the table's key and the gate's key differ.**
Both are *narrowings the gate is read through*, not changes to what the table says, and each is
carried by execution instead. Stated here because a literal reading without them makes the gate fail
on rows that are correct.

- **`field` is finer than any per-route declaration can be.** A `422` row carries the field name —
  `POST /shop/api/session` has **two** rows, `displayName` and `language` — because that is the axis
  C11 dispatches on. FastAPI keys `responses={…}` by **status code**, so a route declares **one**
  `422`: the field lives in the body, not in the declaration. The **declaration half** of S8's gate
  (§5.1) therefore compares at **status** granularity — six `(route, field)` cells over five
  validating routes, against five `422` declarations — and the **field axis is proved by execution**:
  one contract test per cell, plus S8's pinned first-violation selection rule. The table and the gate
  do not disagree; the gate is evaluated on the coarser of the two keys, and the finer one has its
  own evidence.
- **The `5xx` rows sit outside the handler cross product, by construction.**
  `{registered handlers} × {routes that class permits}` can only enumerate what some handler
  *produces*; an unmapped error has no handler — being unmapped is what makes it a `5xx` — so it can
  never appear on the gate's left-hand side, and neither `5xx` row is "a handler with no row" or a
  row the cross product should have generated. They are in the table because **C12 needs them on the
  client**, and they are proved producible the only way they can be: by execution, raising an
  unmapped graph error (`redis.exceptions.ResponseError`) from the repository on each of the two
  routes. **The same escape exists on every graph-touching route** — it is the residue *total by
  type* bounds but cannot remove — and it is **rowed on the two resets only, deliberately**: those
  are the destructive routes where C12's never-retry absolute is load-bearing and where §5.2 makes a
  claim about a specific producer. Elsewhere an unmapped `5xx` is C13's business on the client, which
  is exactly what C13 exists for. **This asymmetry is not an omission the gate should fail on.**

**A new `(route, response)` pair is not shipped until it has a row here, and §5.2 is updated to
match.** The table's own history is the argument for both the key it uses **and** for why the key is
not the closure. §5.3 was written to end one defect class — a client rule spanning responses that
share a status code but not a meaning — and its first version reproduced that class one code over. A
table keyed on *(response → rule)* caught a third instance in minutes **and created a fourth**. Keyed
on *(route, response)* it surfaced a fifth (`POST /shop/api/reset`'s unruled `5xx` → **C12**) — and
then a sixth appeared on two axes the key cannot express at all: **below** it, inside one cell,
discriminated by the error body's field (**C11**), and **beside** it, on the transport axis
query-vs-mutation, which the table has no column for (**C12**). Making the map total then surfaced a
seventh (see S8: a write's query-time timeout meant *unknown* on two routes and an unmapped `500` on
the rest). **And fixing *that* produced three more of a different kind** — rules that matched and
were wrong, created by the fix itself: they are scored, with the reason the arithmetic favours them,
in C13's residual paragraph, and they are why this section's licence now demands one action as well
as one meaning.

**So the enumeration is necessary and is not the guard.** Each re-key made the *search* cheaper —
finding instance six took one pass, not four — but every instance was a response the plan did not
know it could receive, and no table of known responses can bound the unknown ones. The two guards
that do are **total by type on the server** (S8, so the producible set is bounded by construction and
the gate above is decidable) and **loud by default on the client** (**C13**). This table is how the
two are checked against each other; it is not what makes them true. **Those two guards close the
*unruled* half only** — a rule that matches and is wrong passes both. The third mechanism, for that
half, is **S12a's per-rule tests enumerating the routes each rule spans**; it is the last piece and
it is executable, which is why plan review stops here and the remaining evidence is collected at the
S8 and S12a implementation gates.

**Invariants the server enforces vs. conventions the client upholds** — the split matters because
only the left column survives a buggy or hostile client:

| Enforced by the server (S8/S10) | Upheld by the client (S12a) |
|---|---|
| A participant token is refused on every presenter route and vice versa (§6.2's auth matrix) | C1–C3's dispatch: *which* session is cleared and *where* the user lands |
| No route accepts `ws`/`threadId`/`customerId`/`orderId` from the client (§5.2) | C4's per-route choice of re-read endpoint — and its refusal to re-read at all after a lost join, where no credential survives |
| `reset-all` invalidates participant tokens and not the presenter's (§4.8) | C5's reading of a `401` as evidence rather than as failure |
| `409 TurnInProgress` before any write when a turn is in flight (§4.4 1a); `409 unscoped_participant` rather than a false `200` when the graph is unrepaired (§5.2) | C6a's composer retention and C6b's refusal to render an alarm as busy; C7's language step; C8's cadence |
| The quiesce timer bounds both resets and returns `503` having touched nothing (§4.8) | C9's *nothing changed* report and its safe retry |
| `/order/advance` takes the order id from server-side state and CAS-guards the transition (§4.6), so `404`/`409` can only mean *stale*, never *someone else's* | C10's re-read-and-render, and its refusal to treat either as an auth failure |
| Pydantic bounds every request body and query, answering `422` (§4.2, S8) | C11's in-place field errors, and the client's own matching bounds |
| A `Thread` UNIQUE violation surfaces as `5xx` and the server never retries it (§4.8's premise, S8's call-count test) | C12's no-auto-retry **in the browser** — the layer neither of those covers |

A client that violates a right-hand row is **wrong, not dangerous**: every isolation guarantee in
§4.3 is a left-hand row. That is the intended split, and C1–C13 place no security obligation on the
browser.

---

## 6. Test strategy

### 6.1 Unit / offline

Runs with no FalkorDB and no network, in `falkor-chat/server/tests/`. Follow the suite's existing
review-safe pattern (`test_services.py` builds `Services(FakeRepo())`) so these can run against a
live shared instance with zero risk to `reference`. Hazards to respect, all documented in
`falkor-chat/docs/SERVER.md` §1.7: a default `pytest -q` run **wipes `reference` at fixture
setup**, and a green exit code with FalkorDB down means the integration half silently skipped —
always read `N passed, M skipped`. Re-run the seed sequence after any default pytest run —
**deliberately unpinned**: it restores your own dev workspace, so it must follow
`FALKORCHAT_WS_ID`, unlike S1's and S4's done-conditions.

- **S2** — chat-path ctx merge, reserved-key rejection, back-compat for `run_ctx=None`.
- **S3** — the trigger wired with `responder=None` never reaches the responder; **and the
  `dev_surface=False` route table**, asserted directly on `app.routes` rather than by probing for
  404s (a 404 could come from a typo in the probe; an absent route cannot).
- **S6/S7** — token verify (good / wrong / malformed / deleted participant); restart survival;
  join idempotency including the profile-name write; state composition; participant-disjoint
  reset **and its post-reset profile-name re-write**; a non-empty image manifest and exact `imageUrl` shape; the 15-row catalog bound;
  cross-participant order advance refused.
- **S7c** — the catalog read once, not `1 + n`: with `services.lookup_product` patched to raise,
  `list_catalog()` still returns every row with its real slug, and `filter_products`' row keys are
  exactly `{productId, name, category, price}`.
- **S9** — the bounded queue with `turn_workers=1` and a fake 2 s LLM: per-participant
  single-flight, `409` before write, global ordering, drain on shutdown, **and reset-mine
  cancelling a queued turn rather than waiting it out** (the `200`-vs-`503` case in S9's
  done-condition, which is the only one that tells the two apart).

**The client's unit tier runs separately and is not covered by the paragraph above** — Vitest +
Testing Library under `npm test` in `salesperson/`, no server and no FalkorDB, with the network
stubbed at the fetch boundary (which is what lets §6.2's S12a bullet assert *intercepted requests*).
Every SPA step carries its own unit tests in its done-condition; the two hazards above are
falkor-chat's and do not apply to it.

### 6.2 Integration / contract

- **S4** — repository tests against an isolated `ws:test` graph (the pattern `test_queries.sh`
  already uses), proving both resets and every provisioning primitive, with the negative
  assertions spelled out per label (§4.8's survivor column, `WorkspaceConfig` included), the
  thread-scoped-not-author-scoped rule, `reference` untouched, and a post-`reset_all`
  `verify_salesperson.sh <that same graph>` (argument explicit) + `verify_catalog.sh` exit 0.
- **S8/S10** — `TestClient` contract tests over the whole router: the auth matrix (no token /
  participant token on a presenter route / presenter token on a participant route), and **the
  cross-participant probe** — with A and B both provisioned and A holding cart items, messages
  and an order, every route called with B's token returns only B's data. Plus S8's image-wiring
  tripwire, built to go red rather than null: the `storefront_dir` `create_app` forwards is asserted
  against a *different*, also-populated `config.STOREFRONT_DIR`, so a `create_app` reading config
  instead of forwarding fails with *wrong* URLs and one forwarding only to the mount fails with
  `null` ones.
- **S12a** — the **client** tier's contract tests, which §6 named nowhere before v1.13 (the client's
  only appearance in the test strategy was `npm test` green inside four step rows): **§5.3's C1–C13,
  each with a test that goes red when the rule is broken**, asserted on **intercepted requests and
  stored credentials** rather than on rendered outcomes — §5.3 C5 is why that distinction is not
  pedantry — **and each rule's test enumerating by name the routes its rule spans** (S12a's
  done-condition; C4 over all five writing routes, C9 over both actions, C11 over all six
  `(route, field)` cells). This is the plan's only mechanical guard against a rule whose domain was
  widened without its content, and it is one of the two implementation gates review resumes at (the
  other is S8's `{handlers} × {routes}` assertion).
- **S12b/S12c/S12d/S13/S14 — the rest of the SPA track**, whose tests live in each step's own
  done-condition rather than in a bullet here, and run in two places: the **Playwright mobile
  project**, defined once by **S12b** and single-owner thereafter (§5.0), which S12d's
  `presenter.spec.ts` runs under; and each step's component tests under `npm test`. Named here so
  §6 accounts for the whole seven-step client track rather than only its first step.
- **§4.9's structural claim, tested as a claim about the app, not about a request:** with
  `dev_surface=False`, `app.routes` contains no `api.build_router` route, no `/` `StaticFiles`
  mount and no `/mcp` mount; `GET /health` still answers; and the *default* deployment is
  unchanged (the full existing suite is the regression test for that).
- **Mount ordering** — `/shop` shadows neither `/health` nor, in a dev-surface build, `/` or
  `/mcp`.

### 6.3 Live acceptance (`qa-engineer`, S15)

Ordered behaviours to drive black-box against a running `start_demo.sh`:

1. Join with a name only → own session, no login step; **the profile panel shows that name
   immediately** (§4.10). **(AC-1, part of AC-8)**
2. Two participants; A adds items and converses; B's cart and transcript show no trace. Then, from
   a second phone, open the demo host **root** and confirm there is no reachable chat UI, no
   `GET /channels`, no `GET /search` — §4.9's claim, verified from the network side.
   **(AC-2)**
3. Add / remove / change quantity → cart lines and running total update correctly. **(AC-6)**
4. Place an order → order card appears; advance through fulfilled → delivered via the demo
   controls, and a separate participant cancels theirs; both statuses reflected. **(AC-7)**
5. **Measured, n = 10:** order without a stored delivery address → the conversation asks for it
   *at order time* and then displays it. This is a prompt-adherence claim on a 3 B model (§4.10),
   not a code-review item; record the adherence rate. **(AC-8)**
6. Catalog shows the live 15-product electronics catalog. **(AC-9a)**
7. **Measured, n = 10 per locale:** three participants pick en / pt-BR / es simultaneously and each
   holds a 5-turn conversation; record per-turn adherence. A failure triggers §4.5's reversal path
   (one def per language), **not** a wording guess. **(AC-9b)**
8. Presenter enters the key on `/shop/presenter`, then "reset everyone" clears all state including
   their own conversation while their presenter token survives; a single participant's own reset
   clears only theirs and keeps them signed in. **The roster lists every joined participant by
   display name before the reset and is empty after it** — the presenter view must show data, not
   just load. Both are driven **from a phone** — which the key-based presenter control supports
   and the rejected loopback variant did not. **(AC-5)**
9. A product **with** an asset shows its picture **and** one **without** shows text only, with no
   placeholder element in the DOM. Both branches, because the negative branch alone passes
   vacuously when the manifest is empty (§4.7). **(AC-11)**
10. Phone viewport (360 × 740 and 390 × 844): chat, cart and order status all usable, no
    horizontal scrolling, no unreadably small text. **(AC-4)**

**Procedural hazard, non-negotiable:** do **not** write any file under `falkor-chat/` during a
live pass. `start_server.sh` runs uvicorn with `--reload` watching the whole tree, and a mid-pass
write silently kills in-flight background work, producing false negatives with nothing in the
response explaining why (`falkor-chat/docs/SERVER.md` §1.7, confirmed under K-050). `start_demo.sh`
must therefore set a **non-empty** `UVICORN_ARGS` — `UVICORN_ARGS=""` does *not* disable
`--reload` (bash `:-` treats empty as unset; also documented there).

### 6.4 Load / concurrency (AC-3, S15)

`salesperson/scripts/load_demo.py`, modelled on `falkor-chat/scripts/load_test.sh`'s
concurrent-REST-clients shape, drives N synthetic participants through join → scripted 5-turn
conversation → cart add → place order, reporting latency percentiles **split by route class**
(poll reads vs. agent turns) and asserting cross-participant isolation on every response. It must
honour the `409 TurnInProgress` contract rather than firing blind — a harness that ignores it is
the only client that would ever hit §4.4 measure 1a's defect.

- **Run A — stub LLM.** A local fake OpenAI-compatible endpoint with fixed latency, wired through
  `FALKORCHAT_OPENCODE_CONFIG` (`options.baseURL` is mandatory per provider — an example file that
  omits it parses but fails to *resolve*; `docs/SERVER.md` §1.7). This isolates the **server +
  graph**. Target: 50 participants, p95 `GET /shop/api/state` < 300 ms, zero errors, zero
  isolation violations.
  **Plus the queue-depth headroom check under `reset_all`:** FalkorDB's `MAX_QUEUED_QUERIES` is
  **25**, and the ~240 ms stop-the-world reset write against 50 participants polling at 2 s is
  estimated at **~18** queued — under the cap, but not by much. It is not measurable without
  concurrent load, which is why S0 handed it here (`docs/plans/salesperson-ui-graph.md` §12 item 3).
  **Observe it with `GRAPH.INFO`'s `Waiting queries` section** (present on this instance), polled for the duration of the `reset_all` call — a ~240 ms window is not sampleable by guesswork, and without a named mechanism the check degrades to "no query was rejected", which passes whenever the run never approached the cap and leaves the headroom number — the actual point — unmeasured. Report the peak depth against the cap; a breach is a rejected query, i.e. a failed poll.
- **Run B — live LM Studio.** The same harness, sweeping concurrency (1, 2, 4, 8, 16, 32, 50) and
  **publishing the measured reply-latency curve**, plus the measured delta from §4.4 measure 3
  (embedding on vs. off) rather than the plan's estimate of it.

**AC-3's literal wording is not fully satisfiable, and the report must say so.** AC-3 reads *"the
UI keeps responding without noticeable degradation **for any participant**."* Under the
stakeholder's chosen OQ-1 basis — stub-LLM pass plus a *published* live curve plus a staggered demo
script, explicitly **not** a live pass/fail threshold — that holds for every **read** path at 50
participants, and does **not** hold for **agent turns**, whose latency grows with concurrency by
the hardware property R1 names. The test report records: AC-3 **met for the UI's responsiveness**
(Run A, all read paths, 50 participants) and **not met as literally worded for agent-reply
latency**, with Run B's curve as the evidence and the staggered demo script as the operational
answer. Recording a bare "pass" against wording the system does not satisfy would be the wrong
outcome, and this is stated here so nobody has to decide it under time pressure on demo week.

---

## 7. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **The LLM endpoint, not the UI, is the FR-5 ceiling** — one LM Studio instance on a 6 GB GPU cannot serve ~50 simultaneous agent turns at interactive latency. | **High** | §4.4's four measures make reads unaffected and queueing fair + visible; §6.4 measures rather than asserts, and states plainly where AC-3's literal wording is not met. OQ-1's confirmed answer includes a **staggered demo script** — an operational answer, cheaper than any engineering. |
| R2 | **K-060 is open** — `salesperson@v5/v7` sometimes silently drops a genuine match from a mixed-category `filter_products` result (in progress, root-caused at n=75, low base rate). A polished demo UI amplifies every agent-reliability defect: a business audience reads a wrong answer as a broken product. | **High** | The catalog grid with pictures (FR-11) is a partial structural mitigation — participants browse visually instead of asking the model to enumerate. §4.6's "demo controls" framing keeps the self-serve fulfil/deliver from reading as product weirdness. Record the defect in the demo brief. K-060 has its own track; no wording guess here. |
| R3 | **Prompt adherence on two counts** — the language instruction carried in a JSON CONTEXT block (§4.5) and the order-time address confirmation (§4.10). Both are exactly the class of thing K-057/K-060 shows this 3 B model getting wrong. | Medium | §6.3 #5 and #7 gate AC-8 and AC-9 on **measured** runs; §4.5 and §4.10 each carry a pre-designed reversal path, and neither is a further wording guess. |
| R4 | **Reset is a destructive multi-label sweep on a shared graph**, with a **timing** dimension as well as a scoping one — wrong and it either bricks the demo (deleting the `Agent`, the def snapshot, or `WorkspaceConfig`) or wipes a bystander, and a reset racing an in-flight turn burns a turn (an LLM call consumed, nothing written) and can mint an orphan `ReadCursor` (`docs/plans/salesperson-ui-graph.md` §7, F3 — the `Message`/`StepRun`/`TraceEvent` orphans v1.0 feared do not occur). | **High** | S0's `graph-dba` note fixes the exact Cypher and the §4.8 quiesce contract before S4 is written; S4 asserts every survivor by label and the thread-scoped rule; S7/S10 assert the note's §7 (a)–(d) quiesce conditions under a stub-LLM turn. `WorkspaceConfig` is called out by name because taking it would silently undo K-056's Ministral re-point. |
| R5 | **Node is not on `PATH` on the dev box** (falkor-chat's own AGENTS.md note) — no bundle, no demo. | Medium | S5 is a `devops` step that provisions and documents it, and **S5's done-condition is the HTMX-fallback decision deadline** (§4.2) rather than a standing option. `start_demo.sh` fails loudly with the fix. `dist/` stays gitignored (OQ-6). |
| R6 | **No authentication, and one standing shared secret.** Anyone reachable on the network with the link can join under any name, including impersonating a display name; `FALKORCHAT_STOREFRONT_PRESENTER_KEY` is a long-lived secret in the server's environment, and anyone who learns it can reset the whole demo. Over plain HTTP on a shared LAN, every participant bearer token is on the wire. | Medium (accepted) | Bounded by FR-1's controlled-demo scope and "never real customer data". §4.9 removes every unauthenticated *read* path, so the residual is the key itself plus token interception, not a browsable surface. S10 rate-limits the key exchange. A TLS-terminating reverse proxy closes the on-the-wire half and **is compatible with the key-based design** (it was not with the rejected loopback variant). **Revisit reopens K-016.** |
| R7 | **`--reload` kills in-flight background work** when any file under `falkor-chat/` is written during a live run, silently. Its blast radius is smaller than it looks *because* §4.3 makes the graph the authoritative registry — a restart does not invalidate tokens or lose carts. | Medium | S11 sets a non-empty `UVICORN_ARGS`; §6.3 states the procedural rule; S6 carries the restart-survival done-condition. |
| R8 | **The `reference` graph is wiped** by `scripts/test_queries.sh`'s teardown *and* by a default `pytest -q` run's `wf_repo` fixture — taking the catalog **and** both def publications with it, mid-demo-prep. | Medium | `start_demo.sh` runs `verify_catalog.sh` + `verify_salesperson.sh` as a preflight and re-seeds on failure; §4.9's startup readiness check refuses to serve a workspace missing the def or the catalog; §6.1 states the re-seed obligation. |
| R9 | **`salesperson@v7` publish/materialize drift** between `reference` and `ws:{WS_ID}` — the workspace snapshot is what actually executes, and the two can diverge independently. | Low | `verify_salesperson.sh` in S11's preflight (scoped to exactly the two defs that matter) plus §4.9's startup snapshot check. **Not** `GET /workspaces/{ws}/readiness` — that route is unmounted in the storefront deployment and expects `access-request@v1`, which this demo never seeds. |
| R10 | **Poll load** — 50 clients × 2 routes / 2 s ≈ 50 req/s of graph reads, against a measured ~614 msg/s write path. | Low | Well inside budget; `GET /shop/api/state` deliberately composes profile+cart+order into one round trip. S0's `GRAPH.PROFILE` check confirms the reads stay index-backed. |
| R11 | **Retiring the Streamlit app.** Downgraded from v1.0: under OQ-3 it is a history-preserving `git mv` to `deprecated/`, not a delete, so the app survives **on disk**, not only in history — and the move (U5) happens *before* the new component is built, not after acceptance. | Low | The only residual is stale references to the old paths, which S16's acceptance command catches. |
| R12 | **Join is not idempotent, so a lost `POST /shop/api/session` response can leave a ghost participant** — the write commits, the token never reaches the browser, and the graph keeps a `User` with a `tokenHash` nobody holds, owning a `Channel` and `Thread`, while the person re-joins as a second identity. It requires a FalkorDB socket timeout (default 10 s) during that one write. | Low (accepted) | Decided in §5.3 C4's join case: **accepted rather than engineered away**, because the alternative (a client-supplied idempotency nonce, which §5.2's invariant does permit) reopens **delivered** S6 — new `join()` parameter, uniqueness constraint, S0 amendment — for a window this narrow. The client reports "your join may not have completed — join again"; the presenter is warned a stale roster row may appear; S12d renders it as a participant who never speaks; `reset-all` sweeps it, since it is an ordinary participant `User`. **Reversal trigger:** join acquiring a side effect beyond the roster (payment, external provisioning, a quota), or use outside a controlled demo — then the nonce lands as its own step. |

---

## 8. Open questions

**All six of v1.0's §8 questions are now answered** and folded into the plan; they are recorded
here as resolved rather than deleted, because several of them shaped decisions a future reader
will otherwise want to re-litigate.

| # | Question | Answer |
|---|---|---|
| OQ-1 | AC-3's acceptance basis | Stub-LLM pass + a **published** live curve + a staggered demo script. A live pass/fail threshold is explicitly **not** the bar. Consequences: §4.4, §6.4, §10's AC-3 row. |
| OQ-2 | Locale set | English (default), Brazilian Portuguese, Spanish. |
| OQ-3 | Component home | Retired app → `deprecated/salesperson/` (history-preserving move, `teco`'s U5, sequenced *before* S5); the new SPA component takes `salesperson/`. Server half stays inside `falkor-chat` as argued. §4.1. |
| OQ-4 | Who advances the order | Participant self-serve only; no presenter-driven variant. §4.6 adds the "demo controls" framing so it does not read as a product defect. |
| OQ-5 | Presenter distinction | `FALKORCHAT_STOREFRONT_PRESENTER_KEY` (S6's delivered spelling). The loopback-binding alternative was tried and **rejected on executed evidence** (uvicorn's default `proxy_headers=True` inverts it behind exactly the reverse proxy §3's HTTPS implies); §4.3 records that so it is not "simplified" back. |
| OQ-6 | Product images | An agent sources ~15 permissively-licensed stock photos; licence recorded in `salesperson/README.md`; `dist/` stays gitignored. S14 + S5. |

**Nothing in this plan is open** — a statement about §8's questions, not about outstanding work:
review findings from the S1/S2 implementation gate are tracked in
`docs/plans/salesperson-ui-coordination.md`, not here. `teco`'s U5 landed the `deprecated/salesperson/` move
(25 renames, zero deletions), so every `deprecated/salesperson/*.py` citation in §2.4 and §4.2
resolves on disk — re-checked, including that `session_manager.py:13` is still the line the
Streamlit rejection quotes.

---

## 9. Ready to implement

Plan: **`/home/mauricio/prg/graphmind-ai-lab/docs/plans/salesperson-ui.md`** (v1.19) — **21 steps**
(S0–S16, with S12 split into S12a/S12b/S12c/S12d and Ruling 1 carried as its own **S7c**): one
`graph-dba` design note (S0, dispatch first, blocks S4), a ten-step falkor-chat server track
(S1–S4, S6–S10 and S7c), one bring-up script (S11), a seven-step SPA track (S5, S12a–S12d, S13,
S14), plus QA (S15) and docs (S16).

**Dispatch order:** S0 · S1 · S2 · S3 · S5 in parallel → S4 → S6 → S7 → S7c → S8 → S9 → S10 →
S11 · S12a → S12b · S12c → S13 · S14 · S12d → S15 → S16.

**Sequencing constraints outside the file map:** S5 needs `teco`'s U5 (the `deprecated/` move) to
have landed; S12a needs both S5 and S8; S15 needs S11, S13, S14 and S12d. Within
`falkor-chat/server/falkorchat/`, `app.py` (S3 → S8 → S9), `config.py` (S3 → S6),
`services.py` (S2 → S4), `repository.py` (S4 → S7c), `storefront.py` (S6 → S7 → S7c → S9 → S10)
and `storefront_api.py` (S8 → S9 → S10) are the serialization constraints; `falkor-chat/AGENTS.md` is S1 → S11 → S16.
§5.0 has the complete map.

## 10. AC → step map

| AC | Requirement | Satisfied by |
|---|---|---|
| **AC-1** | name-only join, no login | S6 (registry/join), S8 (`POST /shop/api/session`), S12a (join flow) · verified §6.3 #1 |
| **AC-2** | two participants, zero cross-visibility | **S3** (responder kill switch **and** §4.9's `dev_surface=False`, which removes the legacy REST router, the `/` web mount and `/mcp` from the route table), S4 (provisioning), S6 (server-resolved scope, graph-authoritative tokens), S8 (no client-supplied ids; the cross-participant contract probe; the route-table assertion) · verified §6.2 and §6.3 #2 (including a network-side check that the demo host root exposes nothing), plus an isolation assertion on every load-harness response (§6.4) |
| **AC-3** | ~50 participants, no noticeable degradation | S9 (per-participant single-flight bounded turn queue, raised anyio limiter, no storefront embedding), S15 (harness) · verified §6.4 · **met for all read paths at 50 participants; not met as literally worded ("for any participant") for agent-reply latency** — §6.4 states the recording rule, per OQ-1's chosen basis |
| **AC-4** | phone-sized screens, no horizontal scroll | S12b (mobile-first shell, bottom sheets, safe-area insets, the Playwright mobile project), S13, S14 · verified §6.3 #10 at 360×740 and 390×844 |
| **AC-5** | presenter "reset everyone" + per-participant reset | S0 (delete design + quiesce contract), S4 (repository + survivor assertions), S7 (`reset_participant`), S10 (`reset_all`, presenter key exchange), **S12d** (the presenter view itself — roster, reset-everyone control, `incomplete`/`unresolved` rendering), S12a (**§5.3's credential & session contract C1–C13** — the two credentials and their storage, per-credential `401`/`403`, the per-path `504` re-read — plus `reset-all` response typing), **S12b** (the participant's own reset control in the profile sheet, its confirm step and the post-reset return to the language step — asserted on rendered state) · verified §6.2 and §6.3 #8, driven **from a phone** |
| **AC-6** | cart + running total update correctly | S7 (`get_state` over `services.get_cart`), S14 (cart panel) · verified §6.3 #3 |
| **AC-7** | order lifecycle status visible | **S4** (`get_customer_current_order`, `order_belongs_to_customer` — B4's missing primitives), S7 (`advance_own_order`, order in state), S8 (`POST /shop/api/order/advance`), S14 (order card + "demo controls" framing) · verified §6.3 #4 |
| **AC-8** | profile prompted for and displayed | **S6** (join writes the display name into the profile, so the panel is populated from second one — §4.10), **S1** (v7's order-time delivery-address sentence), S7 (profile in state), S14 (profile panel) · verified §6.3 #5 as a **measured** n=10 adherence run, not a code-review claim |
| **AC-9** | real electronics catalog + per-participant language | S1 (v7 language sentence), S2 (chat-path `run_ctx`), S6 (language on the participant record), S7 (`list_catalog`, explicitly bounded), S9 (`run_ctx={"language": …}` at turn start), S12c (i18n + the join-time choice), S14 (catalog grid) · verified §6.3 #6 and **#7 (measured, n=10 per locale — the real gate)** |
| **AC-10** | readiness gate on the first live demo | Not a build gate. S16 records it in `docs/HISTORY.md`; K-056 is resolved (2026-08-30) and K-060 is a separate open track (R2) |
| **AC-11** | picture when available, text-only with no placeholder otherwise | S7 (`build_image_manifest` over the **served** directory, non-empty assertion), **S8 (`create_app` forwards one `storefront_dir` to both the `Storefront` and the `/shop` mount — asserted against a different populated config default, since a mis-wire shows up as `null` or wrong URLs, never as an error)**, S14 (renders `<img>` only when `imageUrl !== null` — **no** `onError` swap; sources the assets) · verified §6.3 #9 with **both** branches asserted, since the negative branch alone passes vacuously on an empty manifest |
