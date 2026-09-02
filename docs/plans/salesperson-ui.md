# The one salesperson UI — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (M<n> TBD) · **Version:** 1.2 · **Reviews:** `docs/reviews/salesperson-ui.md`

*2026-09-02 — v1.1: revised against `docs/reviews/salesperson-ui.md` (4 blockers, 9 majors, 15 minors) and the stakeholder's OQ-1…OQ-6 answers; the client component takes the `salesperson/` name and the retired app moves to `deprecated/salesperson/`.*
*2026-09-02 — v1.2: revised against that review's `## Pass 2` (approve with suggestions) — N1 pins `FALKORCHAT_WS_ID=demo` and adds a non-label survivor clause plus a positive non-participant survivor test, N2 assigns the SPA's shared entry files to S12a, N3 re-keys the route-table assertion onto the `storefront` parameter, plus both nits; `teco`'s `deprecated/` move is recorded as landed.*

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
- **`salesperson@v5` is the current def** (`proof_defs.py:301`). One `agent` step + one terminal
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
                                   │ trigger.py ─▶ executor ─▶ salesperson@v6       │
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
3. **Agent-visible context** — `salesperson@v6` grants **no** `graphrag_retrieve`, and
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

**Presenter identity — `FALKORCHAT_PRESENTER_KEY`, restored by stakeholder decision (OQ-5).**
One operator secret, typed once at `/shop/presenter`, exchanged for a presenter bearer token via
`POST /shop/api/presenter/session` (rate-limited, S10). It is not a login: no accounts, no
per-user credentials, no identity store, no authorization model — the same category of thing as
`FALKORCHAT_AGENT_ID`. The alternative of letting any participant fire "reset everyone" fails
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

- **`salesperson@v6`** — a version bump that republishes the full cumulative
  `config.tools` / `systemPrompt` / **`config.model`** (all three are create-only; omitting
  `config.model` silently reverts K-056's Ministral re-point) and adds **two sentences** to
  `systemPrompt`: one for language (reply in the language named by `language` in the CONTEXT
  block; if none is named, reply in English) and one for §4.10's order-time delivery-address
  confirmation. Topology is byte-identical to v5, so the K-034 409 topology-conflict path is
  never hit.
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
#9 gains an explicit positive case.

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
| **Reset mine** — `POST /shop/api/reset` | the participant, own token | their `Thread` and everything hanging off it — `Message` + `NEXT`/`HEAD`/`TAIL`/`POSTED_BY`/`MENTIONS_MEMBER`/`EMITTED`, `ReadCursor` + `HAS_CURSOR`, `WorkflowRun` + `StepRun` + `TraceEvent`; plus their `Cart` + `CartItem`s, `Order`s + `OrderLine`s, and their `Customer` | their `User` (token, `displayName`, `language`) and `Channel`; a **fresh** `Thread` is minted and `User.threadId` repointed; `Agent`; `WorkflowDefSnapshot`/`Step`; `WorkspaceConfig`; `Document`/`Chunk`/`Entity`; every other participant's subgraph |
| **Reset everyone** — `POST /shop/api/presenter/reset-all` | presenter token only | the above for every participant, **plus** every participant `User` and `Channel` — so all participant tokens are invalidated and every client is bounced to the join screen | `Agent`; `WorkflowDefSnapshot`/`Step`; **`WorkspaceConfig`**; `Document`/`Chunk`/`Entity`; `config.USER_ID`'s lifespan-created `User`; the presenter's own presenter token; `reference` entirely |

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
thread, while `_record`/`suspend_run` writes against a deleted run silently no-op, leaving orphan
`StepRun`/`TraceEvent`/`Message` rows. "Reset everyone" mid-demo, with up to `turn_workers` turns
in flight and a queue behind them, is the realistic case — it is R4's "wrong and it either bricks
the demo or wipes a bystander" arriving through the *timing* dimension R4 did not consider. So:

- **Reset mine** cancels that participant's queued turn and, if one is in flight, waits for it
  bounded by `FALKORCHAT_STOREFRONT_QUIESCE_S` (default 30 s, comfortably under the 180 s agent
  timeout) before deleting; on timeout it returns `503` and changes nothing.
- **Reset everyone** stops intake first (every subsequent post gets `409` until it completes),
  then drains, then deletes.

S0 specifies the mechanism; S7 and S10 carry the done-condition that a reset issued while a
stub-LLM turn is in flight leaves **no orphan `StepRun`/`TraceEvent`/`Message`**.

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
   because there is no second value to get wrong.**

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
  in `ws:{config.WS_ID}` (via `resolve_member_kinds`), `salesperson@v6`'s snapshot is present,
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
2. **`salesperson@v6`'s second added sentence** covers order-time delivery-address confirmation:
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

**P** marks steps that may run in parallel with their siblings in the same stage. §5.0 is
regenerated mechanically from §5.1's Files column — it is what dispatch is gated on, so it lists
**every** file any step touches, not only the contested ones.

### 5.0 Shared-file map (regenerated from §5.1; read before dispatching in parallel)

| File | Touched by | Ordering |
|---|---|---|
| `falkor-chat/server/falkorchat/app.py` | S3, S8, S9 | **S3 → S8 → S9** |
| `falkor-chat/server/falkorchat/config.py` | S3, S6 | **S3 → S6** |
| `falkor-chat/server/falkorchat/services.py` | S2, S4 | **S2 → S4** |
| `falkor-chat/server/falkorchat/storefront.py` | S6, S7, S9, S10 | **S6 → S7 → S9 → S10** |
| `falkor-chat/server/falkorchat/storefront_api.py` | S8, S9, S10 | **S8 → S9 → S10** |
| `falkor-chat/server/falkorchat/repository.py` | S4 | — |
| `falkor-chat/server/falkorchat/trigger.py` | S2 | — |
| `falkor-chat/server/falkorchat/schemas.py` | S8 | — |
| `falkor-chat/server/falkorchat/proof_defs.py` | S1 | — |
| `falkor-chat/server/tests/test_services.py` | S2, S4 | **S2 → S4** |
| `falkor-chat/server/tests/test_app.py` | S3, S8 | **S3 → S8** |
| `falkor-chat/server/tests/test_storefront.py` | S6, S7 | **S6 → S7** |
| `falkor-chat/server/tests/test_storefront_api.py` | S8, S10 | **S8 → S10** |
| `falkor-chat/server/tests/test_trigger.py` · `test_repository.py` · `test_salesperson_scaffold.py` | S2 · S4 · S1 | — |
| `falkor-chat/docs/QUERIES.md` (new §18) | S4 | — |
| **`falkor-chat/AGENTS.md`** | S1, S11, S16 | **S1 → S11 → S16** |
| `falkor-chat/README.md` | S16 | — |
| `falkor-chat/scripts/{seed,verify}_salesperson.sh` | S1 | — |
| `falkor-chat/scripts/start_demo.sh` | S11 | — |
| `salesperson/` scaffold + toolchain config (`package.json`, `vite.config.ts`, `build.sh`, `.gitignore`) | S5 | — |
| `salesperson/playwright.config.ts` | S5, S12b | **S5 → S12b** |
| **`salesperson/src/{main.tsx,App.tsx,index.css}`** — the SPA's shared entry files | S5 (scaffold), **S12a** (owns thereafter) | **S5 → S12a, and no later step edits them** — S12a lands the provider/layout **mount slots** so S12b and S12c never need to |
| `salesperson/src/**` (everything else) — `api/`+`session/`+`routes.tsx` (S12a) · `layout/`+`components/sheets/` (S12b) · `i18n/`+`locales/` (S12c) · `views/Chat*`+`components/message/` (S13) · `views/{Cart,Order,Profile,Catalog}*` (S14) | S12a, S12b, S12c, S13, S14 | **S12a first**, then S12b ‖ S12c, then S13 ‖ S14 — the five subtrees named at left are disjoint, which is what makes the two parallel pairs safe. The files that fall *outside* all five (the row above) are the collision the S12 split would otherwise reintroduce, and are assigned to S12a for exactly that reason |
| `salesperson/tests/e2e/**` | S12b | — |
| `salesperson/public/products/**` | S14 | — |
| `salesperson/scripts/load_demo.py` | S15 | — |
| `salesperson/{README,AGENTS}.md` | S5, S16 | **S5 → S16** |
| `docs/plans/salesperson-ui-graph.md` | S0 | — |
| `docs/test-plans/salesperson-ui.md` · `docs/test-reports/salesperson-ui-report.md` | S15 | — |
| root `AGENTS.md` · `docs/HISTORY.md` | S16 | — |

Three gaps in v1.0's map are closed here: `storefront.py` omitted S9 (which read as permitting
S9 ‖ S10 on one file), `storefront_api.py` omitted S9, and **`falkor-chat/AGENTS.md` appeared in
no row at all** despite three steps writing it.

### 5.1 The table

| # | Step | Files | Interface / key symbols | Done-condition | Specialist | Parallel |
|---|---|---|---|---|---|---|
| **S0** | **Graph design note.** Scope fixed by §4.3/§4.6/§4.8: provisioning Cypher; **both reset deletes against §4.8's explicit keep/delete label inventory** (`ReadCursor` goes with the thread; **`WorkspaceConfig` must survive** — a sweep that takes it silently undoes K-056's Ministral re-point; `Document`/`Chunk`/`Entity` survive both), **thread-scoped not author-scoped**, and **§4.8's non-label scoping rule: only a `Channel`/`Thread`/`Message` reachable from a participant `User` (one carrying `tokenHash`) is ever a target — expressed in the Cypher's `MATCH`, not left to the caller**; the §4.8 quiesce contract; the two B4 order reads; a `GRAPH.PROFILE` check of participant-scoped reads at 50 participants; an explicit yes/no on new indexes/constraints. Excludes product images. | `docs/plans/salesperson-ui-graph.md` (new) | `ensure_participant`, `reset_participant`, `reset_all_participants`, `get_customer_current_order`, `order_belongs_to_customer` | Note exists, `Status: active`, every query live-verified against a throwaway `ws:` probe graph; keep/delete decided per label; DDL yes/no stated | `graph-dba` | **Blocks S4.** ‖ S1, S2, S3, S5 |
| **S1** | **`salesperson@v6`** — bump `SALESPERSON_DEF["version"]`, republish full cumulative `config.tools` **and `config.model`** unchanged, add §4.5's language sentence **and §4.10's order-time address sentence** to `systemPrompt`. Bump both scripts' default version fallbacks. **Also update `falkor-chat/AGENTS.md` rows 82–83** (the script table narrating the `v1…v5` chain and `verify_salesperson.sh`'s expected version) so the doc is not stale for the whole S1→S16 window. `docs/BACKLOG.md`'s K-060/K-062 headings also pin `v5`; those belong to those defects' own tracks and are **deliberately not touched here**. | `falkor-chat/server/falkorchat/proof_defs.py`, `falkor-chat/scripts/seed_salesperson.sh`, `falkor-chat/scripts/verify_salesperson.sh`, `falkor-chat/server/tests/test_salesperson_scaffold.py`, `falkor-chat/AGENTS.md` | `SALESPERSON_DEF`, `SALESPERSON_MAX_STEPS` | `seed_salesperson.sh <ws>` then `verify_salesperson.sh <ws>` exits 0 live; a test asserts `config.model == "lmstudio/mistralai/ministral-3-3b"` and `tools ⊇` v5's; pytest green | `coder` | **P** |
| **S2** | **Chat-path `run_ctx` merge.** `services.start_workflow_run` merges a caller `run_ctx` into the chat path's `{"threadId": …}`, reusing the process path's reserved-key rejection (`threadId`, `error` → `WorkflowInputRejectedError`). `trigger.maybe_trigger` gains an optional `run_ctx` forwarded only to the start branch. | `falkor-chat/server/falkorchat/services.py`, `.../trigger.py`, `falkor-chat/server/tests/test_services.py`, `.../test_trigger.py` | `start_workflow_run(..., run_ctx: dict \| None)`, `maybe_trigger(..., run_ctx: dict \| None = None)` | Chat-path start with `run_ctx={"language":"pt-BR"}` yields a run whose ctx carries both keys; reserved keys rejected before any write; existing callers unchanged (default `None`); pytest green | `tdd-engineer` | **P** |
| **S3** | **Two wiring switches.** (a) `config.TRIGGER_RESPONDER_FALLTHROUGH` (`FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH`, default on) → `WorkflowTrigger(responder=None)` when off (§4.3 part 4). (b) **§4.9's `create_app(..., dev_surface: bool = True)`**, derived in `_build_default_app` as `not config.STOREFRONT_ENABLED` alongside `mount_mcp`; when false, neither `api.build_router` nor the `/` `StaticFiles` mount nor `/mcp` is registered, and a bare `GET /health` liveness route is added. **`dev_surface` is a parameter, never an env var.** | `falkor-chat/server/falkorchat/config.py`, `.../app.py`, `falkor-chat/server/tests/test_app.py` | `config.TRIGGER_RESPONDER_FALLTHROUGH`, `config.STOREFRONT_ENABLED`, `create_app(..., dev_surface=)` | With the fall-through flag off, an unmentioning non-resuming message reaches no responder. With `dev_surface=False`, **`app.routes` contains no legacy route and no `/`/`/mcp` mount** (asserted on the route table, not by probing 404s), and `GET /health` still answers. Default deployment byte-identical: full existing pytest suite green | `tdd-engineer` | **P**, owns `app.py`/`config.py` first |
| **S4** | **Repository + thin service primitives**, implementing S0's Cypher verbatim: `add_channel_member`, `set_participant_record`, `get_participant_record`, `list_participants`, `reset_participant`, `reset_all_participants`, **`get_customer_current_order`**, **`order_belongs_to_customer`** — plus `Services` wrappers (`get_current_order`, `order_belongs_to_customer`) so `storefront.py` never holds Cypher (`falkor-chat/AGENTS.md` rule 1, `DESIGN.md` §14.2). Every query added to `falkor-chat/docs/QUERIES.md` **§18** (§17 is the current highest). | `falkor-chat/server/falkorchat/repository.py`, `.../services.py`, `falkor-chat/docs/QUERIES.md`, `falkor-chat/server/tests/test_repository.py`, `.../test_services.py` | the eight repository methods + two service wrappers; all parameterised; `.query`/`.ro_query` split per the platform rule | Integration tests on an isolated `ws:test` graph prove: two participants' resets are disjoint; the delete is **thread-scoped** (an `Agent`-authored reply in the participant's thread is deleted, and no other participant's is); **every §4.8 survivor is asserted by label**, `WorkspaceConfig` included; **and — the assertion that label checks structurally cannot make — a non-participant `Channel` + `Thread` + `Message` (a `User` with no `tokenHash`, mirroring `seed_demo.sh`'s `demo-general`/`demo-welcome`) is seeded into the probe graph and asserted to survive `reset_all` intact**, because victims and survivors share those three labels; `reference` untouched; post-`reset_all` `verify_salesperson.sh` + `verify_catalog.sh` exit 0; every method idempotent | `coder` | **after S0** |
| **S5** | **Node toolchain + component scaffold** into the freed `salesperson/` (after U5's `git mv` to `deprecated/salesperson/`). Provision Node/npm (falkor-chat's own note: `node` is not on `PATH` on WSL2), scaffold Vite + React + TS + Tailwind + Vitest + Playwright, `build.sh`, `.gitignore` for `dist/`/`node_modules/`, initial `README.md` + `AGENTS.md`. | `salesperson/**` (new), `salesperson/{README,AGENTS}.md` | `npm run build` → `salesperson/dist/` with `base: "/shop/"` | `./salesperson/build.sh` produces `dist/index.html` + hashed assets from a clean checkout; `npm test` runs; Node version documented. **This done-condition is §4.2's HTMX-fallback decision deadline** — if it cannot be met, escalate before S12a rather than after | `devops` | **P**, after U5 |
| **S6** | **Storefront core** — participant registry + join + token verify + turn-state map. Token `secrets.token_urlsafe(32)`, `sha256` stored on `User.tokenHash`, `hmac.compare_digest`. **`resolve_token` re-reads the graph; the in-process map is a read-through cache only** (§4.3). Join also writes the display name into the profile (§4.10). Env: `FALKORCHAT_STOREFRONT_ENABLED`, `_DIR`, `_PRESENTER_KEY`, `_TURN_WORKERS`, `_QUIESCE_S`, `_LOCALES`, `FALKORCHAT_THREAD_LIMIT`. **No `FALKORCHAT_DEMO_WS`** (§4.9). | `falkor-chat/server/falkorchat/storefront.py` (new), `.../config.py`, `falkor-chat/server/tests/test_storefront.py` (new) | `Storefront(services, *, presenter_key, turn_workers, quiesce_s)`; `join(display_name, language) -> ParticipantRecord`; `resolve_token(bearer) -> ParticipantRecord \| None` | Join provisions `User`+`Channel`+`Thread`+profile-name idempotently; wrong/absent/malformed/deleted-participant tokens all resolve to `None`; **restart survival: a `Storefront` rebuilt from scratch resolves a token minted by the previous instance** | `coder` | **after S4** |
| **S7** | **Storefront state, reset, catalog, images.** `get_state(ctx)` composing `services.get_profile` + `get_cart` + **`services.get_current_order`** (a repository read, not composed here); `reset_participant` with §4.8's quiesce; `list_catalog()` with an **explicit row bound** (`services.filter_products` defaults `limit=20` — correct for 15 products, silently wrong at 21); `build_image_manifest()` over **`<FALKORCHAT_STOREFRONT_DIR>/products/`** (§4.7); `advance_own_order()` via `services.order_belongs_to_customer` then `advance_order`. | `falkor-chat/server/falkorchat/storefront.py`, `falkor-chat/server/tests/test_storefront.py` | `get_state`, `reset_participant`, `list_catalog`, `build_image_manifest`, `advance_own_order` | State shape stable and the order block populated from the repository read; reset participant-disjoint; **manifest is non-empty against a fixture asset dir** and every `imageUrl` is `/shop/products/<id>.<ext>` or `null`; `list_catalog()` returns all 15 rows; advancing another participant's order refused; **a reset issued while a stub-LLM turn is in flight leaves no orphan `StepRun`/`TraceEvent`/`Message`** | `coder` | **after S6** |
| **S8** | **The `/shop/api` router + mounts.** `storefront_api.build_storefront_router(...)`, the `get_participant()`/`get_presenter()` dependencies, size-bounded Pydantic models mirroring `schemas.py`, the error map, and the `create_app` wiring: include the router and mount `FALKORCHAT_STOREFRONT_DIR` at `/shop` **inside `create_app`** (`/` is a catch-all registered last and Starlette matches in registration order, so a mount added after `create_app` returns is unreachable). Plus §4.9's **route-table assertion** and the **readiness preflight** in `_lifespan`. | `falkor-chat/server/falkorchat/storefront_api.py` (new), `.../schemas.py`, `.../app.py`, `falkor-chat/server/tests/test_storefront_api.py` (new), `.../test_app.py` | `create_app(..., storefront: bool = False, storefront_dir: Path \| None = None, dev_surface: bool = True)`; routes per §5.2; `401` bad/absent token, `403` bad presenter key, `404` unknown order, `409` stale CAS / turn in flight, `503` quiesce timeout or storefront disabled | `TestClient` contract tests for every route, incl. the **auth matrix** and **the cross-participant probe**: with A holding cart items, messages and an order, every route called with B's token returns only B's data. `/shop` shadows nothing. Preflight refuses to start on a missing `Agent`, missing snapshot or empty catalog, naming the fix command | `coder` | **after S7; owns `app.py` after S3** |
| **S9** | **Concurrency layer** (§4.4). The bounded `ThreadPoolExecutor` turn queue **keyed by `participantId`** with queue-position accounting and §4.4 measure 1a's `409 TurnInProgress` refusal *before* the message write; the storefront post path (`services.post_message` + enqueue trigger with `run_ctx={"language": …}`, **no `_safe_embed`**); raise the anyio limiter **inside `_lifespan` before `yield`**; graceful executor shutdown. | `falkor-chat/server/falkorchat/storefront.py`, `.../storefront_api.py`, `.../app.py` | `Storefront.enqueue_turn(ctx, participant, posted)`; `turn: {state, queuePosition}` on `GET /shop/api/state` | With `turn_workers=1` and a stub 2 s LLM, three *different* participants' posts report queue positions 0/1/2 and complete in order; **two posts 100 ms apart from one participant produce exactly one `WorkflowRun` on that thread**, the second returning `409` with no `Message` written; poll latency unaffected while the queue is full; executor drains on shutdown | `coder` | **after S8** |
| **S10** | **Presenter surface** — `POST /shop/api/presenter/session` (key → token, rate-limited: fixed delay + attempt counter, so the key is not trivially brute-forced on an open LAN), `GET /shop/api/presenter/participants` (roster filtered on `User.tokenHash IS NOT NULL`, so `config.USER_ID`'s lifespan node never appears), `POST /shop/api/presenter/reset-all` with §4.8's stop-intake-then-drain quiesce. | `falkor-chat/server/falkorchat/storefront.py`, `.../storefront_api.py`, `falkor-chat/server/tests/test_storefront_api.py` | `presenter_login(key) -> token`, `list_participants()`, `reset_all()` | A participant token is refused on every presenter route; a wrong key is refused and counted; `reset-all` invalidates every participant token **but not the presenter's**, and clears the presenter's own conversation too; **no orphan rows with turns in flight**; roster excludes non-participant `User`s | `coder` | **after S9** |
| **S11** | **Demo bring-up script** — `falkor-chat/scripts/start_demo.sh`, which **first pins `FALKORCHAT_WS_ID=demo`** (overridable, but never left to `config.py`'s `"acme"` default — that is the repo's populated dev/demo workspace; §4.9 move 2). Then: FalkorDB → `bootstrap_schema.sh "$FALKORCHAT_WS_ID"` at `EMBEDDING_DIM=1024` → `seed_demo.sh "$FALKORCHAT_WS_ID"` → `seed_catalog.sh` → `seed_salesperson.sh "$FALKORCHAT_WS_ID"` → preflight `verify_salesperson.sh` + `verify_catalog.sh` → build the SPA → uvicorn with `FALKORCHAT_ENABLE_AGENT=1`, `FALKORCHAT_WORKFLOW_ENABLED=1`, `FALKORCHAT_TRIGGER_DEF_KEY=salesperson`, `_VERSION=v6`, `FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH=0`, `FALKORCHAT_STOREFRONT_ENABLED=1`, `FALKORCHAT_STOREFRONT_DIR`, and a **non-empty** `UVICORN_ARGS` so `--reload` is off. Every seed script gets the workspace **explicitly**, even though the pin makes its default correct — defence in depth, not a load-bearing requirement. `seed_workflows.sh` is deliberately **not** run: this demo needs neither `triage` nor `access-request`. Add the script-table row to `falkor-chat/AGENTS.md`. | `falkor-chat/scripts/start_demo.sh` (new), `falkor-chat/AGENTS.md` | — | From a cold box the script yields a reachable `/shop` with a working join **and a working first agent turn**, against `ws:demo` and not `ws:acme` (assert the resolved workspace in the startup banner); it fails loudly and specifically when Node, the bundle, FalkorDB, a def, the catalog or the demo `Agent` is missing | `devops` | **after S5, S8** |
| **S12a** | **Session + API client + routing.** Bearer handling, `401 → rejoin`, `409 TurnInProgress` handling, TanStack Query polling (2 s state+messages; catalog fetched once), route shell for join / chat / presenter. | `salesperson/src/api/**`, `salesperson/src/session/**`, `salesperson/src/routes.tsx`, **`salesperson/src/{main.tsx,App.tsx,index.css}`** | `useSession()`, `useShopState()`, `apiClient` | Join → chat round-trips against a live server; a 401 returns to join; a 409 retains the composer text and re-enables send when `turn.state` is `idle`; **`App.tsx` exposes an i18n-provider slot and a layout-shell slot, and `index.css` a Tailwind layer entry, each with a no-op default, so S12b and S12c mount into them without editing any shared entry file** (verified by S12b/S12c touching none); `npm test` green | `frontend-engineer` | **after S5 and S8** |
| **S12b** | **Mobile layout shell**, mounted into S12a's layout slot — **edits no shared entry file**. Sticky header with cart/order/profile icon buttons, bottom-sheet overlays, safe-area insets, no horizontal scroll at 360 px; the Playwright mobile project. | `salesperson/src/layout/**`, `salesperson/src/components/sheets/**`, `salesperson/tests/e2e/**`, `salesperson/playwright.config.ts` | — | Playwright at 360×740 and 390×844 shows no horizontal overflow and legible type; sheets open/close by icon | `frontend-engineer` | **after S12a**, ‖ S12c |
| **S12c** | **i18n**, mounted into S12a's provider slot — **edits no shared entry file**. `react-i18next` wiring, the three locale bundles, locale-aware currency/date formatting, and the join-screen language chooser feeding `POST /shop/api/session`. | `salesperson/src/i18n/**`, `salesperson/src/locales/{en,pt-BR,es}.json` | `t()`, `useLocale()` | All three bundles complete (no missing-key fallbacks in a key-coverage test); chosen locale reaches the join request; UI chrome switches | `frontend-engineer` | **after S12a**, ‖ S12b |
| **S13** | **Chat view** — transcript (`textContent` only, **no** `dangerouslySetInnerHTML`), optimistic send, thinking/queued indicator driven by `turn`, welcome turn, error/retry, autoscroll-when-at-bottom (mirroring `web/app.js`). | `salesperson/src/views/Chat*`, `salesperson/src/components/message/**` | — | A scripted 5-turn conversation renders correctly; a queued turn shows its position; agent-emitted markup renders as literal text | `frontend-engineer` | **after S12b, S12c**, ‖ S14 |
| **S14** | **Cart / order / profile / catalog panels** (FR-8/9/10/11 parity per §2.4). Cart lines + running total + empty state; profile card with em-dash placeholders; catalog grid with image-or-text-only cards; order card with a status chip, `cancel` as an ordinary customer action and **`fulfill`/`deliver` inside a visually distinct "demo controls" affordance labelled as a warehouse simulation** (§4.6). Sources the ~15 stock images and records their licence in `salesperson/README.md` (OQ-6). | `salesperson/src/views/{Cart,Order,Profile,Catalog}*`, `salesperson/public/products/**` | — | Panels match §2.4's parity table; a product **with** an asset renders an `<img>` and one **without** renders text-only with no `<img>` in the DOM (both asserted) | `frontend-engineer` | **after S12b, S12c**, ‖ S13 |
| **S15** | **Test suites & AC evidence** — the load harness (`load_demo.py`, stub-LLM and live-LLM modes, latency percentiles by route class, automated cross-participant isolation assertion on every response), the live language-adherence run, the measured AC-8 run, and the mobile Playwright pass. Deliverable is a versioned test plan + report. | `salesperson/scripts/load_demo.py` (new), `docs/test-plans/salesperson-ui.md`, `docs/test-reports/salesperson-ui-report.md` | — | Every AC has recorded evidence; AC-3, AC-8 and AC-9 carry measured numbers, not assertions; **the report states plainly where AC-3's literal wording is not met** (§6.4) | `qa-engineer` | **after S11, S13, S14** |
| **S16** | **Docs close-out.** Root `AGENTS.md` (new `salesperson/` bullet, new `deprecated/` bullet, component-docs table row, "Working in this repo" bullet); root `docs/HISTORY.md`; `falkor-chat/README.md` + `AGENTS.md` (the `/shop` surface, the storefront deployment's un-mounted dev surface, new env vars); `salesperson/{README,AGENTS}.md` final pass. **The `claude/frontend-engineer/frontend-engineer.md` refresh is NOT in scope** — an agent edit must land with its `kaizen/{plan,history}.md` and `claude/README.md` in the same change (`claude/AGENTS.md`), which routes to **`cobb`**; `teco` dispatched it as U6. | root `AGENTS.md`, `docs/HISTORY.md`, `falkor-chat/README.md`, `falkor-chat/AGENTS.md`, `salesperson/{README,AGENTS}.md` | — | The command below returns **zero** matches (verified today it returns exactly the two `claude/frontend-engineer/frontend-engineer.md` lines U6 owns, and nothing else) | `coder` | **last** |

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
`customerId` or `orderId` from the client.**

| Route | Body / query | Returns |
|---|---|---|
| `GET /shop/api/health` | — | `{status, storefrontEnabled, locales}` |
| `POST /shop/api/session` | `{displayName ≤ 60, language ∈ locales}` | `{participantId, token, displayName, language, welcome}` |
| `GET /shop/api/state` | — | `{profile:{name,deliveryAddress}, cart:{items[],total}, order:{orderId,status,lines[],total}\|null, turn:{state,queuePosition}}` |
| `GET /shop/api/messages` | `?since=<ms>&limit=<1..200>` | message rows (participant's own thread, server-resolved) |
| `POST /shop/api/messages` | `{text ≤ 2000}` | the posted row · **`409 TurnInProgress`, nothing written**, when that participant already has a turn in flight |
| `GET /shop/api/catalog` | — | `[{productId,name,category,price,imageUrl\|null}]` |
| `POST /shop/api/order/advance` | `{transition ∈ {fulfill,deliver,cancel}}` | `{orderId,status}` · `409` stale CAS · `404` no order of theirs |
| `POST /shop/api/reset` | — | `200 {threadId, language}` — **the participant's token survives** (§4.8); the client returns to the language step, not the full join screen · `503` on quiesce timeout, nothing changed |
| `POST /shop/api/presenter/session` | `{key}` | `{token}` · `403` on a bad key (rate-limited) |
| `GET /shop/api/presenter/participants` | — | `[{participantId,displayName,language,messageCount,cartTotal,orderStatus}]` |
| `POST /shop/api/presenter/reset-all` | — | `{clearedParticipants:<n>}`; every **participant** token invalidated, the presenter token is not |

---

## 6. Test strategy

### 6.1 Unit / offline (every server step)

Runs with no FalkorDB and no network, in `falkor-chat/server/tests/`. Follow the suite's existing
review-safe pattern (`test_services.py` builds `Services(FakeRepo())`) so these can run against a
live shared instance with zero risk to `reference`. Hazards to respect, all documented in
`falkor-chat/docs/SERVER.md` §1.7: a default `pytest -q` run **wipes `reference` at fixture
setup**, and a green exit code with FalkorDB down means the integration half silently skipped —
always read `N passed, M skipped`. Re-run the seed sequence after any default pytest run.

- **S2** — chat-path ctx merge, reserved-key rejection, back-compat for `run_ctx=None`.
- **S3** — the trigger wired with `responder=None` never reaches the responder; **and the
  `dev_surface=False` route table**, asserted directly on `app.routes` rather than by probing for
  404s (a 404 could come from a typo in the probe; an absent route cannot).
- **S6/S7** — token verify (good / wrong / malformed / deleted participant); restart survival;
  join idempotency including the profile-name write; state composition; participant-disjoint
  reset; a non-empty image manifest and exact `imageUrl` shape; the 15-row catalog bound;
  cross-participant order advance refused.
- **S9** — the bounded queue with `turn_workers=1` and a fake 2 s LLM: per-participant
  single-flight, `409` before write, global ordering, drain on shutdown.

### 6.2 Integration / contract

- **S4** — repository tests against an isolated `ws:test` graph (the pattern `test_queries.sh`
  already uses), proving both resets and every provisioning primitive, with the negative
  assertions spelled out per label (§4.8's survivor column, `WorkspaceConfig` included), the
  thread-scoped-not-author-scoped rule, `reference` untouched, and a post-`reset_all`
  `verify_salesperson.sh` + `verify_catalog.sh` exit 0.
- **S8/S10** — `TestClient` contract tests over the whole router: the auth matrix (no token /
  participant token on a presenter route / presenter token on a participant route), and **the
  cross-participant probe** — with A and B both provisioned and A holding cart items, messages
  and an order, every route called with B's token returns only B's data.
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
   clears only theirs and keeps them signed in. Both are driven **from a phone** — which the
   key-based presenter control supports and the rejected loopback variant did not. **(AC-5)**
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
| R2 | **K-060 is open** — `salesperson@v5/v6` sometimes silently drops a genuine match from a mixed-category `filter_products` result (in progress, root-caused at n=75, low base rate). A polished demo UI amplifies every agent-reliability defect: a business audience reads a wrong answer as a broken product. | **High** | The catalog grid with pictures (FR-11) is a partial structural mitigation — participants browse visually instead of asking the model to enumerate. §4.6's "demo controls" framing keeps the self-serve fulfil/deliver from reading as product weirdness. Record the defect in the demo brief. K-060 has its own track; no wording guess here. |
| R3 | **Prompt adherence on two counts** — the language instruction carried in a JSON CONTEXT block (§4.5) and the order-time address confirmation (§4.10). Both are exactly the class of thing K-057/K-060 shows this 3 B model getting wrong. | Medium | §6.3 #5 and #7 gate AC-8 and AC-9 on **measured** runs; §4.5 and §4.10 each carry a pre-designed reversal path, and neither is a further wording guess. |
| R4 | **Reset is a destructive multi-label sweep on a shared graph**, with a **timing** dimension as well as a scoping one — wrong and it either bricks the demo (deleting the `Agent`, the def snapshot, or `WorkspaceConfig`) or wipes a bystander, and a reset racing an in-flight turn leaves orphan rows. | **High** | S0's `graph-dba` note fixes the exact Cypher and the §4.8 quiesce contract before S4 is written; S4 asserts every survivor by label and the thread-scoped rule; S7/S10 assert the no-orphan property under a stub-LLM turn. `WorkspaceConfig` is called out by name because taking it would silently undo K-056's Ministral re-point. |
| R5 | **Node is not on `PATH` on the dev box** (falkor-chat's own AGENTS.md note) — no bundle, no demo. | Medium | S5 is a `devops` step that provisions and documents it, and **S5's done-condition is the HTMX-fallback decision deadline** (§4.2) rather than a standing option. `start_demo.sh` fails loudly with the fix. `dist/` stays gitignored (OQ-6). |
| R6 | **No authentication, and one standing shared secret.** Anyone reachable on the network with the link can join under any name, including impersonating a display name; `FALKORCHAT_PRESENTER_KEY` is a long-lived secret in the server's environment, and anyone who learns it can reset the whole demo. Over plain HTTP on a shared LAN, every participant bearer token is on the wire. | Medium (accepted) | Bounded by FR-1's controlled-demo scope and "never real customer data". §4.9 removes every unauthenticated *read* path, so the residual is the key itself plus token interception, not a browsable surface. S10 rate-limits the key exchange. A TLS-terminating reverse proxy closes the on-the-wire half and **is compatible with the key-based design** (it was not with the rejected loopback variant). **Revisit reopens K-016.** |
| R7 | **`--reload` kills in-flight background work** when any file under `falkor-chat/` is written during a live run, silently. Its blast radius is smaller than it looks *because* §4.3 makes the graph the authoritative registry — a restart does not invalidate tokens or lose carts. | Medium | S11 sets a non-empty `UVICORN_ARGS`; §6.3 states the procedural rule; S6 carries the restart-survival done-condition. |
| R8 | **The `reference` graph is wiped** by `scripts/test_queries.sh`'s teardown *and* by a default `pytest -q` run's `wf_repo` fixture — taking the catalog **and** both def publications with it, mid-demo-prep. | Medium | `start_demo.sh` runs `verify_catalog.sh` + `verify_salesperson.sh` as a preflight and re-seeds on failure; §4.9's startup readiness check refuses to serve a workspace missing the def or the catalog; §6.1 states the re-seed obligation. |
| R9 | **`salesperson@v6` publish/materialize drift** between `reference` and `ws:{WS_ID}` — the workspace snapshot is what actually executes, and the two can diverge independently. | Low | `verify_salesperson.sh` in S11's preflight (scoped to exactly the two defs that matter) plus §4.9's startup snapshot check. **Not** `GET /workspaces/{ws}/readiness` — that route is unmounted in the storefront deployment and expects `access-request@v1`, which this demo never seeds. |
| R10 | **Poll load** — 50 clients × 2 routes / 2 s ≈ 50 req/s of graph reads, against a measured ~614 msg/s write path. | Low | Well inside budget; `GET /shop/api/state` deliberately composes profile+cart+order into one round trip. S0's `GRAPH.PROFILE` check confirms the reads stay index-backed. |
| R11 | **Retiring the Streamlit app.** Downgraded from v1.0: under OQ-3 it is a history-preserving `git mv` to `deprecated/`, not a delete, so the app survives **on disk**, not only in history — and the move (U5) happens *before* the new component is built, not after acceptance. | Low | The only residual is stale references to the old paths, which S16's acceptance command catches. |

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
| OQ-5 | Presenter distinction | `FALKORCHAT_PRESENTER_KEY`. The loopback-binding alternative was tried and **rejected on executed evidence** (uvicorn's default `proxy_headers=True` inverts it behind exactly the reverse proxy §3's HTTPS implies); §4.3 records that so it is not "simplified" back. |
| OQ-6 | Product images | An agent sources ~15 permissively-licensed stock photos; licence recorded in `salesperson/README.md`; `dist/` stays gitignored. S14 + S5. |

**Nothing in this plan is open.** `teco`'s U5 landed the `deprecated/salesperson/` move
(25 renames, zero deletions), so every `deprecated/salesperson/*.py` citation in §2.4 and §4.2
resolves on disk — re-checked, including that `session_manager.py:13` is still the line the
Streamlit rejection quotes.

---

## 9. Ready to implement

Plan: **`/home/mauricio/prg/graphmind-ai-lab/docs/plans/salesperson-ui.md`** (v1.1) — **19 steps**
(S0–S16 with S12 split into S12a/S12b/S12c), one `graph-dba` design note (S0, dispatch first,
blocks S4), a six-step falkor-chat server track, a six-step SPA track, plus bring-up, QA and docs.

**Dispatch order:** S0 · S1 · S2 · S3 · S5 in parallel → S4 → S6 → S7 → S8 → S9 → S10 →
S11 · S12a → S12b · S12c → S13 · S14 → S15 → S16.

**Sequencing constraints outside the file map:** S5 needs `teco`'s U5 (the `deprecated/` move) to
have landed; S12a needs both S5 and S8; S15 needs S11, S13 and S14. Within
`falkor-chat/server/falkorchat/`, `app.py` (S3 → S8 → S9), `config.py` (S3 → S6),
`services.py` (S2 → S4), `storefront.py` (S6 → S7 → S9 → S10) and `storefront_api.py`
(S8 → S9 → S10) are the serialization constraints; `falkor-chat/AGENTS.md` is S1 → S11 → S16.
§5.0 has the complete map.

## 10. AC → step map

| AC | Requirement | Satisfied by |
|---|---|---|
| **AC-1** | name-only join, no login | S6 (registry/join), S8 (`POST /shop/api/session`), S12a (join flow) · verified §6.3 #1 |
| **AC-2** | two participants, zero cross-visibility | **S3** (responder kill switch **and** §4.9's `dev_surface=False`, which removes the legacy REST router, the `/` web mount and `/mcp` from the route table), S4 (provisioning), S6 (server-resolved scope, graph-authoritative tokens), S8 (no client-supplied ids; the cross-participant contract probe; the route-table assertion) · verified §6.2 and §6.3 #2 (including a network-side check that the demo host root exposes nothing), plus an isolation assertion on every load-harness response (§6.4) |
| **AC-3** | ~50 participants, no noticeable degradation | S9 (per-participant single-flight bounded turn queue, raised anyio limiter, no storefront embedding), S15 (harness) · verified §6.4 · **met for all read paths at 50 participants; not met as literally worded ("for any participant") for agent-reply latency** — §6.4 states the recording rule, per OQ-1's chosen basis |
| **AC-4** | phone-sized screens, no horizontal scroll | S12b (mobile-first shell, bottom sheets, safe-area insets, the Playwright mobile project), S13, S14 · verified §6.3 #10 at 360×740 and 390×844 |
| **AC-5** | presenter "reset everyone" + per-participant reset | S0 (delete design + quiesce contract), S4 (repository + survivor assertions), S7 (`reset_participant`), S10 (`reset_all`, presenter key exchange), S12a/S12b (both controls) · verified §6.2 and §6.3 #8, driven **from a phone** |
| **AC-6** | cart + running total update correctly | S7 (`get_state` over `services.get_cart`), S14 (cart panel) · verified §6.3 #3 |
| **AC-7** | order lifecycle status visible | **S4** (`get_customer_current_order`, `order_belongs_to_customer` — B4's missing primitives), S7 (`advance_own_order`, order in state), S8 (`POST /shop/api/order/advance`), S14 (order card + "demo controls" framing) · verified §6.3 #4 |
| **AC-8** | profile prompted for and displayed | **S6** (join writes the display name into the profile, so the panel is populated from second one — §4.10), **S1** (v6's order-time delivery-address sentence), S7 (profile in state), S14 (profile panel) · verified §6.3 #5 as a **measured** n=10 adherence run, not a code-review claim |
| **AC-9** | real electronics catalog + per-participant language | S1 (v6 language sentence), S2 (chat-path `run_ctx`), S6 (language on the participant record), S7 (`list_catalog`, explicitly bounded), S9 (`run_ctx={"language": …}` at turn start), S12c (i18n + the join-time choice), S14 (catalog grid) · verified §6.3 #6 and **#7 (measured, n=10 per locale — the real gate)** |
| **AC-10** | readiness gate on the first live demo | Not a build gate. S16 records it in `docs/HISTORY.md`; K-056 is resolved (2026-08-30) and K-060 is a separate open track (R2) |
| **AC-11** | picture when available, text-only with no placeholder otherwise | S7 (`build_image_manifest` over the **served** directory, non-empty assertion), S14 (renders `<img>` only when `imageUrl !== null` — **no** `onError` swap; sources the assets) · verified §6.3 #9 with **both** branches asserted, since the negative branch alone passes vacuously on an empty manifest |
