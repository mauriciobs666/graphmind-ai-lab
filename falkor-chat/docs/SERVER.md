# falkor-chat — Server Application & MCP Transport

> **Status:** active · **Owner:** `architect` · **Tracks:** —

> **Scope.** The **server process** as it runs now: internal layering, the auth/tenancy seam and
> every configuration key, the REST and MCP front doors, the `/shop` storefront and `web/`
> mounts, the post-message handler policy (responder and workflow trigger), the agent tool
> surface, the on-disk layout, testing hazards, the model-resolution seam and logging. The
> **graph** — topology, data model, indexes, executor semantics, capacity, operations — is
> `falkor-chat/docs/DESIGN.md`. Every query's text and its guards are
> `falkor-chat/docs/QUERIES.md`; executable DDL is `scripts/bootstrap_schema.sh`. Change history
> is `falkor-chat/docs/HISTORY.md`; unbuilt work is `falkor-chat/docs/BACKLOG.md`.

---

## 1. Application architecture (client/server)

`DESIGN.md` §10 sketches the *operational* topology (app ⇄ FalkorDB). This section pins the
*application* code architecture: what the clients and the server are, the transport between
them, and the internal layering.

### 1.1 Scope decisions

| Axis | Decision | Rationale |
|---|---|---|
| **Transport** | **REST/JSON over FastAPI**, plus MCP Streamable-HTTP (§2) on the same process | Browser clients speak HTTP natively — no gRPC-Web bridge tax — and FastAPI gives a free OpenAPI console. |
| **Clients** | The **dev web UI** (`web/`, at `/`), the **storefront SPA** (`salesperson/`, at `/shop`), and **MCP agents** (at `/mcp`) | The smallest end-to-end paths that exercise the stack: a dev console, a business-facing storefront, and an agent front door. |
| **Real-time** | **Request/response only** | Clients poll (the dev UI re-fetches a thread window after posting; the storefront polls `GET /shop/api/state` every 2 s). A push path (Redis Pub/Sub → WebSocket/SSE) would slot onto the same service layer with no schema change — open as K-018. |
| **Auth / tenancy** | **One process-constant tenant** — `ws` and `actor` from the environment (§1.3); the storefront adds a per-participant bearer credential | Keeps the chat data path in focus. Injected at one seam (§1.3) so real auth (K-016) replaces it without touching services or the repository. |

> Transport is REST rather than gRPC: gRPC's wins (polyglot typed contracts, native streaming,
> service-to-service performance) are unused when the clients are browsers, and gRPC-Web cannot
> do client/bidi streaming in browsers anyway — WebSocket/SSE is the stronger real-time path. REST
> keeps the layers below the router transport-agnostic, so a gRPC servicer or a
> service-to-service hop can still be bolted onto the same `Services` if a non-browser consumer
> appears.

### 1.2 Layering

```
┌─ Clients ────────────────────┐            ┌─ Python server (FastAPI, one process) ─────────────────┐
│ dev web UI (web/, at /)      │ REST/JSON  │ api.py            dev REST router (thin: HTTP ⇄ Service) │
│ storefront SPA (/shop)       │ ─────────▶ │ storefront_api.py /shop/api router + credential deps    │
│ MCP agents (/mcp)            │ MCP/HTTP   │ mcp.py            MCP tools (thin: tool ⇄ Service)      │
└──────────────────────────────┘            │   ▲ CallContext = {ws, actor}  (§1.3)                   │
                                            │ storefront.py     participants, turn executor, resets   │
                                            │ background.py     out-of-band embed / trigger / ingest  │
                                            │ trigger.py · responder.py · executor.py · tools.py      │
                                            │ services.py       domain logic, invariants              │
                                            │ repository.py     Cypher ⇄ QUERIES.md (RO | RW)         │
                                            │ db.py             falkordb-py connection, select_graph  │
                                            │ modelconfig.py · transport.py  model resolution (§1.8)  │
                                            └───────────────────────────────────────────────────────┬─┘
                                                                                                    ▼  FalkorDB / LLM providers
```

- **`repository.py` is the only place Cypher lives.** Each method maps 1:1 to a verified query in
  `QUERIES.md`, always parameterised (`params=`), `ro_query` for reads / `query` for writes,
  `select_graph(f"ws:{id}")` for scoping.
- **`services.py` owns the invariants** the write-path rules describe: choosing the first-vs-subsequent
  append variant, id generation, `Thread.updatedAt` bumps, setting `role`/`POSTED_BY`.
- **The routers and `mcp.py` are thin.** They map transport ⇄ service calls and errors ⇄ status;
  `api.py`/`mcp.py` are the only layers that change if the transport is revisited.
  `storefront.py` is the storefront's own domain layer above `services.py` (§1.3).
- **`modelconfig.py`/`transport.py` (§1.8) are the model-resolution seam.** Every LLM/embedding
  consumer (`responder.py`, `executor.py`, `app._LlmGuardJudge`, `embedding.py`, `ingestion.py`,
  `tools.py`) holds a `ModelGateway` and resolves per call — never constructs `llm.py`'s or
  `embedding.py`'s OpenAI-compatible clients directly.

### 1.3 The auth/tenancy seam

The process-wide scope lives in **one FastAPI dependency**, not scattered through the code:

```python
# config.py — env-resolved once, at import
WS_ID   = os.environ.get("FALKORCHAT_WS_ID", "acme")
USER_ID = os.environ.get("FALKORCHAT_USER_ID", "u1")

def get_context() -> CallContext:        # the seam
    return CallContext(ws=WS_ID, actor=USER_ID)

# api.py — the FastAPI dependency routes actually depend on
from .config import get_context as _resolve_context
def get_context() -> CallContext:        # a wrapper, so tests can override it
    return _resolve_context()
```

It is **process-constant, not literal**: both values come from the environment at import time,
which is what lets a deployment point the process at its own workspace. What it is not is
*per-caller* — every call that resolves through `get_context`, which is the dev REST router and
`/mcp`, gets the same actor **and the same workspace**. So a second target workspace is a
**second server process** with its own `FALKORCHAT_WS_ID` (what `scripts/start_demo.sh` does for
`demo`), never a per-call or per-token parameter; the storefront's `/shop/api` routes do not
resolve through `get_context` at all (below) — their `actor` is per participant, their `ws` this
same constant. The `actor` half carries the mirror consequence: whatever a client says about
itself is ignored (MCP's `frm`, §2.2) — no call through `get_context`, including
`ingest_document` and `send_message`, reads any per-caller identity off `ctx.actor`. The one
place identity varies per caller is `Storefront.context_for` (below), which builds its own
`CallContext` beside this seam and varies only `actor`. The graph side needs nothing new for it:
`Repository.create_document` and `create_document_with_auto_supersede` resolve `INGESTED_BY`
against either a `User` or an `Agent` by the id they are handed.

`get_context` is the *only* place `ws` is fixed. Everything below the seam is
workspace-parametrized **per call**: services and the repository take `ws` / `actor` as
parameters, and the three components constructed once at startup — `ModelGateway` (`.llm()` /
`.embedder()` / `.resolve()` take `ws=`, §1.8), `EmbeddingWorker` (`embed_message(ws, …)` /
`embed_chunk(ws, …)`, its index-dimension cache keyed by `(ws, label)`) and `IngestionPipeline`
(`extract_chunk(ws, …)`) — resolve `ws` fresh on every call rather than baking it in at
construction. When real auth lands (token → user + workspace claim, or the `identity` graph as
source of truth — K-016) **only `get_context` changes**.

At startup `_lifespan` runs `services.ensure_actor(get_context())`, projecting the configured
actor into `ws:{WS_ID}` as a `User` so the write paths (which refuse an unknown author) have their
anchor. This is also the first FalkorDB round-trip: an unreachable instance aborts startup within
the connect timeout (`db.FalkorDBUnreachableError`, naming host:port), and an actor id colliding
with an existing `Agent` aborts with `MemberIdCollisionError`. Building the app itself never
touches the network.

**The dev REST router is unauthenticated**, which is fine only while every caller is trusted. It
stops being fine the moment mutually-untrusting people share one workspace: `GET /channels`,
`GET /threads/{tid}/messages`, `GET /search?q=` and the post box all resolve through the
process-constant `get_context()`, so any client on the network can read every transcript in the
workspace and write into any thread. `/mcp` and the `/` web mount are the same seam again.

The storefront deployment closes this **by construction rather than by configuration** —
`create_app` takes:

```python
create_app(services, dev_surface=True)   # the default: the dev app
create_app(services, dev_surface=False)  # no api.build_router, no `/` mount, no /mcp
create_app(services, dev_surface=False, storefront=True, storefront_dir=...)
                                         # + the /shop/api router, its typed error map, the /shop SPA mount
```

`dev_surface=False` un-mounts the entire unauthenticated dev surface. It dominates `mount_mcp`, so
`/mcp` is absent however the app was constructed. Because `GET /health` lives *inside*
`api.build_router`, that branch re-registers a bare liveness route with the same contract
(`services.ping`, `503 {"detail": "FalkorDB unreachable"}` when FalkorDB does not answer) —
exactly one `/health` exists in either configuration.

`create_app` **refuses `storefront=True` with `dev_surface=True`** (`ValueError`) — the
storefront deployment must not mount the unauthenticated dev surface, and that is enforced at
construction rather than left to the operator. `storefront_dir` (falling back to
`config.STOREFRONT_DIR`) is read **once** and forwarded to both the `Storefront` (which builds the
product-image manifest from `<dir>/products/`) and the `/shop` static mount, so the manifest is
always built from what is actually served.

**`dev_surface` is a function parameter and deliberately not an environment variable.**
`_build_default_app` derives it (and `mount_mcp`) as `not config.STOREFRONT_ENABLED`, and there is
no operator setting that can put the dev surface back while storefront participants exist. Assert
its effect on `app.routes`, never by probing for a 404 — a 404 is returned both when a route is
absent and when it exists but errors.

**Configuration keys.** Every key is an environment variable read once, at import, by `config.py`
(no reload path); the server reads no other environment variable apart from `{env:NAME}`
substitutions inside the model-config files (§1.8). A boolean flag is on for `1`/`true`/`yes`/`on`
(case-insensitive) and off for anything else.

| Env var | Default | Effect |
|---|---|---|
| `FALKORCHAT_WS_ID` | `acme` | The workspace (`ws:{id}` graph) every `get_context` call and the storefront run against. |
| `FALKORCHAT_USER_ID` | `u1` | The actor every `get_context` call is attributed to; projected as a `User` at startup. |
| `FALKORDB_HOST` / `FALKORDB_PORT` | `127.0.0.1` / `6379` | Where the FalkorDB client connects. |
| `FALKORDB_CONNECT_TIMEOUT` | `5` (s) | Client connect timeout: an unreachable instance fails startup fast rather than hanging (a dead port on WSL2 can blackhole instead of refusing). |
| `FALKORDB_SOCKET_TIMEOUT` | `10` (s) | Client-side bound on each command round-trip; long reads pass a per-query `timeout=` instead. |
| `FALKORCHAT_EMBEDDING_DIM` | `1536` | The vector dimension assumed when the resolved embedding model declares no `dim` in the overlay (§1.8), and the default length the repository validates an embedding against before writing it (`EmbeddingDimensionError` on a mismatch). Served GraphRAG workspaces are 1024-dim, so a served app runs with `1024` or a model that declares it (`scripts/start_server.sh` sets it). |
| `FALKORCHAT_OPENCODE_CONFIG` | unset (no product default) | Path to the pristine OpenCode `opencode.json` (providers) — §1.8. Required only when an LLM consumer is wired. |
| `FALKORCHAT_MODEL_CONFIG` | `falkor-chat/config/models.json` | Path to falkor-chat's model overlay (per-kind defaults, timeouts, roles, per-model settings) — §1.8. |
| `FALKORCHAT_AGENT_ID` | `assistant` | The workspace `Agent` the responder, the workflow trigger and the storefront post as and listen for; must match the `agentId` the seed scripts register. |
| `FALKORCHAT_AGENT_NAME` | `Assistant` | Parsed but read by nothing in the server. `scripts/seed_demo.sh` (via `start_server.sh`) reads the same variable directly as the seeded `Agent`'s display name. |
| `FALKORCHAT_ENABLE_AGENT` | off | **On:** `_build_default_app` builds one `ModelGateway` and wires the embedding worker (every posted message and ingested chunk is embedded), the `IngestionPipeline` (per-chunk entity extraction) and the `AgentResponder` (§1.12). **Off:** the app is network-free and nothing out-of-band runs. |
| `FALKORCHAT_WORKFLOW_ENABLED` | off | Read **only when `ENABLE_AGENT` is on**: also wires the workflow executor, the built-in tool registry (§1.11), the LLM guard judge, the `WorkflowTrigger` (§1.12) and the periodic due-run sweep. With `ENABLE_AGENT` off it changes nothing. Which routes answer `503` with the engine unwired: §1.4's run error map. |
| `FALKORCHAT_TRIGGER_DEF_KEY` / `FALKORCHAT_TRIGGER_DEF_VERSION` | `triage` / `v1` | The materialized def an `@mention` starts (§1.12); also the def `GET /workspaces/{ws}/readiness` checks and samples `postSuccess` from. Must match a seeded def or `@mention`-to-start never fires. |
| `FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH` | **on** | Read only where the workflow engine is wired. **Cleared (`0`)** builds `WorkflowTrigger(responder=None)`: a message matching no workflow reaches nothing instead of the responder, whose retrieval is workspace-**wide** (`hybrid_search` with `channel_id=None`) and would otherwise surface another participant's messages. With the engine unwired, the responder is wired directly and this flag changes nothing. |
| `FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S` | `30` | Seconds between ticks of the in-process due-run sweep (`Services.sweep_due_workflow_runs`), which runs only where the workflow engine is wired. A failing tick is logged and the loop continues. `POST /workflow-runs/due` is unaffected. |
| `FALKORCHAT_STOREFRONT_ENABLED` | off | **On:** `_build_default_app` builds the storefront deployment — `storefront=True` (the `/shop/api` router, its error map, the startup preflight, and the `/shop` SPA mount **only when `FALKORCHAT_STOREFRONT_DIR` names an existing directory**) with `mount_mcp=False`, `dev_surface=False`. With the workflow engine wired, the executor is built with `human_speaker_label=storefront.CUSTOMER_SPEAKER_LABEL`, so customer lines reach the model's transcript as `Customer: …`, never under the participant's display name; assistant lines keep their name. Independent of `ENABLE_AGENT`: without it the storefront serves with no agent behind it. |
| `FALKORCHAT_STOREFRONT_DIR` | unset (`None`) | The **served** SPA build directory (`salesperson/dist/`), mounted at `/shop` and the root of the product-image manifest (`<dir>/products/`). Unset resolves to `None`, never `""` (`Path("")` is the working directory). **Unset or naming a nonexistent directory skips the `/shop` mount silently**: the manifest is empty, the preflight logs `images=0` and continues, and `/shop/api` serves normally — so the default configuration is a storefront API with no SPA behind it. |
| `FALKORCHAT_STOREFRONT_PRESENTER_KEY` | `""` | The one operator secret, exchanged at `POST /shop/api/presenter/session` for a presenter bearer token — demo-session scoping, not authentication. **Empty means "no presenter surface" and never authenticates**: `hmac.compare_digest("", "")` is `True`, so the login path rejects an unset key *before* comparing (`Storefront.presenter_configured`). Unset, every login answers the same `403 bad_presenter_key` a wrong key gets; the operator's signal is a log line. |
| `FALKORCHAT_STOREFRONT_PRESENTER_LOGIN_DELAY_S` | `1.0` | A fixed delay `presenter_login` applies to every attempt, wrong key or right, before either check, so timing cannot distinguish them. Paired with `Storefront.presenter_login_failures`, a count logged on every refusal and never consulted. **There is no lockout after N attempts**: with one shared key a lockout is a self-DoS, so the response set stays `200`/`403`/`422` plus the cross-cutting three, never `429`/`423`. |
| `FALKORCHAT_STOREFRONT_TURN_WORKERS` | `4` | `max_workers` of the storefront's bounded turn executor, where every agent turn runs (§1.4 `/shop/api`). **It sets how many turns run at once, and with that the `turn.queuePosition` participants see**: that number is a participant's index in the *waiting* line, derived on every `GET /shop/api/state` from the turn map's booking ordinals; a running turn occupies a worker rather than a place in line, so more workers means fewer turns waiting ahead. Five simultaneous arrivals, the fifth one's position: `1` → `3`, `2` → `2`, `4` → `0`. Size it to the model server: LM Studio's KV pool is shared by every in-flight call, so `start_demo.sh` sets `2` (overridable). Turns never touch anyio's thread limiter, so a deep queue costs the 2 s poll nothing. |
| `FALKORCHAT_STOREFRONT_QUIESCE_S` | `30` | How long a reset waits for an in-flight turn to drain before giving up and changing nothing (`503 {"error":"quiesce_timeout"}`). Reset-**mine** first cancels the participant's turn if it is still queued (`Storefront._cancel_queued_turn`), so the wait applies only to a turn already on a worker or one reserved by a second tab mid-write; reset-**everyone** cancels nothing and waits on every turn, after stopping intake (every post answers `409 turn_in_progress` until `reset_all` exits, success or failure). A short window against a running turn answers the `503` and resets nothing; a window covering the turn waits it out and then resets. Raise it and a reset outlasts the turn instead of timing out. |
| `FALKORCHAT_STOREFRONT_LOCALES` | `en,pt-BR,es` | The languages a participant may join in, in the UI's offer order: the set `POST /shop/api/session` validates `language` against (anything else is `422 validation_failed`) and the list `GET /shop/api/health` reports for the SPA's chooser. A blank or all-separator value falls back to the default. |
| `FALKORCHAT_THREAD_LIMIT` | `100` | The size of anyio's default thread limiter — how many sync endpoints FastAPI offloads concurrently (anyio's own default is 40). Assigned in `_lifespan` before `yield`, the only place it can be (the limiter is event-loop scoped). Applies to every deployment, including the dev one. Headroom for the poll path, not load-bearing. |

The four legacy model env vars are refused at startup: §1.8.

**There is no `FALKORCHAT_DEMO_WS`.** The storefront's workspace *is* `config.WS_ID`: a second
workspace variable buys nothing once the unauthenticated surfaces are un-mounted, while creating
two variables that can disagree — the seed-the-wrong-graph trap. `tests/test_storefront.py`
carries the tripwire against a second one. `scripts/start_demo.sh` pins `FALKORCHAT_WS_ID=demo`,
never `config.py`'s `acme` dev default.

**The storefront adds a *second* auth path, and it is per-caller.** `get_context()` above is
process-constant, and **no `/shop/api` request resolves through it** — the one thing the storefront
takes from that seam is its *workspace*, read once at construction as `provider().ws`, so an
injected `context_provider` pins the storefront too. The `actor` half is
per-request:

```python
# storefront_api.py — the two credential dependencies
get_participant(authorization)   # Bearer <participantId>.<token>   -> ParticipantRecord, else 401
get_presenter(authorization)     # Bearer presenter.<presenterToken> -> None, else 403 / 401

# storefront.py
Storefront.context_for(pid) -> CallContext(ws=WS_ID, actor=pid)   # actor *is* customerId
```

- **The graph is the registry; the in-process map is only a cache, and the router never
  authenticates against it.** `Storefront.resolve_token` re-reads `User.tokenHash` through
  `Repository.get_participant_record` on **every** call and never consults `_records`. That is not
  style: with an authoritative in-process map, the single file write that restarts uvicorn under
  `--reload` would invalidate every token and bounce every participant to a fresh `participantId`,
  losing their **cart and order** rather than just their session (`customerId == participantId`),
  and a participant deleted by either reset would keep authenticating out of stale memory.
  `Storefront.lookup` returns the *identical* `ParticipantRecord` out of the cache and is documented
  as **not an auth path** — the router never calls it, pinned by an AST call-site tripwire.
- **Every failure is one answer.** `resolve_token` returns `None` — never an exception, never a
  partial answer — for an absent, malformed or wrong-scheme header; an unknown participant id; a
  `User` that is not a participant (`seed_demo.sh`'s `u1`, the lifespan's `config.USER_ID` node);
  a participant deleted by either reset; and a valid id carrying the wrong token, including another
  participant's. The middle three are **one** branch, not three: `get_participant_record`'s query
  carries `WHERE u.tokenHash IS NOT NULL`, so a non-participant and a deleted participant both come
  back as zero rows, indistinguishable from an id that never existed. The dependency maps every case
  to one `401 invalid_token`. The hash comparison is `hmac.compare_digest`, pinned by a spy at the
  comparison seam rather than by a timing measurement.
- **The two credentials cannot be used for each other's routes.** A presenter token presented on a
  participant route parses as participant id `presenter`, which no `User` carries, so it resolves to
  `None` and answers the ordinary `401` — held structurally, not by a special case. In the other
  direction `get_presenter` keeps two meanings apart, **and the `403` is narrower than it reads**:
  `parse_bearer` runs first, so **`403 wrong_credential_type`** fires only when the header *parses*
  as `Bearer <principal>.<token>` with a principal other than `presenter` — a participant token,
  typically. **`401 presenter_session_gone`** answers everything else: no header, any header
  `parse_bearer` rejects (a non-`Bearer` scheme, a missing `.`, an empty half — so `Basic abc`,
  `Bearer garbage` and `Bearer presenter` all land here), and a presenter token this process never
  minted. Presenter tokens
  are `secrets.token_urlsafe(32)` minted in-process by `_PresenterSessions` **in exchange for**
  `FALKORCHAT_STOREFRONT_PRESENTER_KEY` rather than derived from it, and are **not** graph state, so
  they do not survive a restart — the operator logs in again.
- **This is demo-session scoping, not authentication** (real auth is K-016). What it does give is the per-participant tenancy the platform seam
  cannot: `ctx.actor` is the participant id and every cart/order/profile service keys on it, so
  isolation between mutually-untrusting participants sharing one workspace is **structural rather
  than filtered**.
- **Startup refuses a mis-seeded deployment rather than coming up green and dead.**
  `storefront_preflight` runs from `_lifespan` and raises `StorefrontPreflightError` — naming the
  fix command for each problem — unless the demo `Agent` resolves in `ws:{WS_ID}` (asked with the
  *same* `resolve_member_kinds` lookup every post runs), the
  `FALKORCHAT_TRIGGER_DEF_KEY`@`_VERSION` snapshot is materialized into that workspace, and the
  catalog is non-empty. The image manifest is built here too but is
  deliberately **not** a condition: an empty manifest is the legitimate text-only deployment.

### 1.4 REST surface → service → verified query

The dev REST router (`api.build_router`) is mounted only under `dev_surface=True` (§1.3); every
route resolves `ctx` through `get_context`. `QUERIES.md` sections are cited where a route's query
lives.

**Chat.**

| Endpoint | Service method | `QUERIES.md` |
|---|---|---|
| `GET /health` | `ping` | liveness probe (trivial `RO_QUERY RETURN 1`) → `{"status":"ok"}`, or `503` when FalkorDB does not answer. Under `dev_surface=False` the same contract is re-registered bare by `create_app` (§1.3) |
| `POST /channels` | `create_channel` | §3 create a channel → **201** |
| `GET /channels[?limit=]` | `list_channels` | §3 list channels in a workspace |
| `POST /channels/{cid}/threads` | `create_thread` | §3 create a thread → **201** |
| `GET /channels/{cid}/threads[?limit=]` | `list_threads` | §3 list recent threads in a channel |
| `POST /threads/{tid}/messages` | `post_message` (+ the out-of-band policy, §1.12) | §2 member-kind lookup, then §4 first message / subsequent message (+ mentions) → **201** |
| `GET /threads/{tid}/messages[?since=&limit=]` | `read_thread` / `read_messages` | §4 full thread when neither param is given; either param → §9.1 window as a **pure read** (`since` defaults to 0, `limit` to 50; the browser never touches cursors). An explicit `since` is a plain `>` and may re-deliver or skip rows within that exact millisecond |
| `GET /messages/{mid}` | `get_message` | §4 get a single message (workspace-global by `msgId`; the body carries `threadId`); `404` when absent |
| `GET /search?q=[&limit=]` | `search_messages` | §5 full-text keyword search |
| `GET /threads/{tid}/participants` | `list_thread_participants` | §2 list thread participants |
| `GET /threads/{tid}/workflow-runs[?limit=]` | `list_workflow_runs_for_thread` | §12.14 every run this thread has had |

**Documents and the knowledge base.**

| Endpoint | Service method | `QUERIES.md` |
|---|---|---|
| `POST /documents` | `ingest_document` (+ per-chunk embed/extract and update detection, §1.12) | §14.9 `create_document_with_auto_supersede` → **201** `{documentId, chunkCount, status: "processing", …}` |
| `POST /documents/batch` | `ingest_documents` | §14.9 per item → **201**, one receipt per item in input order; a failing item comes back as its own `{"status":"error","error":…,"errorType":…}` receipt and is not scheduled |
| `GET /documents/search?q=[&limit=]` | `search_documents` | §14.3 `search_chunks` — chunks most-similar first, `score` = cosine distance; `503 SearchNotAvailableError` when no `ModelGateway` is wired |
| `GET /documents[?current_only=&limit=]` | `list_documents` | §14.7 (oldest first; `current_only=true` by default excludes superseded versions) |
| `GET /documents/{id}` | `get_document` | §14.2; `404` when absent |
| `DELETE /documents/{id}` | `delete_document` | §14.7 (hard delete, chunks included; `404 DocumentNotFoundError` for an unknown or already-deleted id) |
| `GET /documents/{id}/deletion` | `get_document_deletion` | §14.7 (the audit record `{deletedBy, deletedAt, …}`; `404` if never deleted) |
| `GET /documents/{id}/history` | `get_document_history` | §14.8 (every version in the confirmed `SUPERSEDES` chain containing `id`, oldest to newest) |
| `GET /document-updates/pending[?limit=]` | `list_pending_document_updates` | §14.8 (`SUPERSEDES` suggestions awaiting a decision) |
| `GET /document-updates[?status=&limit=]` | `list_document_updates` | §14.8 (filter `pending`/`confirmed`/`rejected`; unfiltered includes the auto-superseded tier, `decidedBy='system'`) |
| `POST /document-updates/{mid}/confirm` · `/reject` · `/recheck` | `confirm_document_update` · `reject_document_update` · `recheck_document_update` | §14.8 (confirm makes the old document non-current; reject keeps the edge as `rejected`; recheck reopens a `rejected` one to `pending`, `null` otherwise; `404 DocumentUpdateNotFoundError` for an unknown id on confirm/reject) |
| `GET /matches/pending[?limit=]` | `list_pending_matches` | §14.6 |
| `GET /matches[?status=&limit=]` | `list_matches` | §14.6 (unfiltered includes the auto-merged tier) |
| `POST /matches/{mid}/confirm` · `/reject` · `/recheck` | `confirm_match` · `reject_match` · `recheck_match` | §14.6 (`SAME_AS` is recorded as an edge, never a physical merge; `404 MatchNotFoundError` for an unknown id on confirm/reject) |

Static paths (`/documents/search`, `/documents/batch`) are registered before
`/documents/{document_id}`, so neither is read as a document id.

**Workflow definitions, snapshots and runs.**

| Endpoint | Service method | `QUERIES.md` |
|---|---|---|
| `POST /workflow-defs` | `publish_workflow_def` | §11.1 publish into `reference` → **201**; `400 WorkflowDefSpecError` for a bad spec or an unresolvable model (§1.8), `409 WorkflowDefConflictError` for a topology-differing re-publish |
| `GET /workflow-defs[?limit=]` | `list_workflow_defs` | §11.3 |
| `GET /workflow-defs/{key}[?version=]` | `get_workflow_def` | §11.3 — latest when no version; `404` when absent |
| `GET /workflow-defs/{key}/versions/{version}` | `get_workflow_def_structure` | §11.2 whole-def structure read |
| `POST /workflow-defs/{key}/versions/{version}/materialize` | `materialize_def` | §11.4 into `ws:{WS_ID}` → **201**; `409` on a topology conflict |
| `GET /workspaces/{ws}/snapshots[?limit=]` | `list_snapshots` | §11.6 |
| `GET /workspaces/{ws}/snapshots/{key}/versions/{version}` | `get_snapshot_structure` | §11.5 whole-snapshot structure read; `404` when absent |
| `GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff` | `diff_def_snapshot` | §11.2 + §11.5 |
| `GET /workspaces/{ws}/readiness` | `check_demo_readiness` | §11.2 + §11.5 per def, §12.15 for `postSuccess` (always `200`; below) |
| `POST /workflow-runs` | `start_workflow_run` (`trigger_msg_id=None`) | §12.12 start a run from a snapshot with **no** chat trigger `Message` → **201** |
| `POST /workflow-runs/{runId}/input` | `submit_workflow_input` | §12.7 + §11.5 (validate) then §12.13 merge-into-ctx **and** resume in one CAS → **200** |
| `POST /workflow-runs/due` | `sweep_due_workflow_runs` | §12.16, then §12.13 per due run → `{checked, due, resumed, raced, faulted}` |
| `GET /workflow-runs/{runId}` | `get_workflow_run` | §12.7; `404` when absent |
| `GET /workflow-runs/{runId}/step-runs` | `read_workflow_step_runs` | §12.8 (`[]` for an unknown run) |
| `GET /workflow-runs/{runId}/trace` | `read_workflow_trace` | §12.11 (`[]` for an unknown run) |

**The path `{ws}` is descriptive.** Every `/workspaces/{ws}/…` route resolves tenancy through
`get_context`, never from the path — the same posture as MCP ignoring a client-supplied `frm`.

**`GET /workspaces/{ws}/readiness`** is the HTTP form of `scripts/verify_workflows.sh`, checking
the same two defs (`FALKORCHAT_TRIGGER_DEF_KEY`@`_VERSION` and `access-request@v1`) with the
script's own problem wording. It answers `{ready, defs: [{key, version, defPresent,
snapshotPresent, inSync, problems}], postSuccess}`, always `200` — a cold graph reads as absent,
never as an error. `ready` is a configuration-only signal. `postSuccess` is a separate,
informational sample of the trigger def's last 20 runs — `{defKey, defVersion, sampleSize,
postedCount, rate, status}`, `rate` `null` and `status` `no-data` when there are none, `ok` when
every sampled run posted, else `degraded` — never folded into `ready`.

**Error envelope.** A `ServiceError` answers `{"error": <class name>, "detail": …}` — `404` for
`ChannelNotFoundError`, `ThreadNotFoundError`, `MatchNotFoundError`, `DocumentNotFoundError`,
`DocumentUpdateNotFoundError` and `AgentNotFoundError`, `400` for every other subclass (e.g.
`DocumentTooLargeError`, `BatchTooLargeError`, `UnknownActorError`, an unresolvable mention).
`WorkflowDefSpecError` → `400`, `WorkflowDefNotFoundError` → `404`, `WorkflowDefConflictError` →
`409`, `SearchNotAvailableError` → `503`, and the run errors below use the same envelope. A plain
`HTTPException` 404 (the "… not found" rows above) answers FastAPI's `{"detail": …}`; a request
failing Pydantic validation answers FastAPI's `422`.

**Bounds.** Request bodies are size-bounded at the Pydantic boundary (`schemas.py`): message text
≤ 8000 chars, channel name / thread title ≤ 200, mentions ≤ 50; document text ≤ 500,000 chars,
`title`/`sourceLabel`/`producedBy` ≤ 200, `sourceFormat` ≤ 50, a batch ≤ 20 documents. Path
segments are bounded ≤ 200 (`422` beyond) only for the document id on `DELETE`, `/deletion` and
`/history`, every `{mid}`, every `{runId}`, and the `{key}`/`{version}` of the def-structure,
materialize, snapshot-structure and diff routes; every other path id is unbounded. Message text lands in graph RAM *and* the full-text index, so the transport caps it (RAM
rule 6). The document and batch caps are also enforced in the service, so an MCP caller is held to
them too. Every `limit` is `Query`-bounded 1–200 (default 50; `GET /documents/search` defaults to
20), except the thread window (1–1000) and `GET /threads/{tid}/workflow-runs` (1–50, default 10);
`POST /workflow-runs/due` takes a body `limit` of 1–1000 (default 200).

The **two append variants** (`DESIGN.md` §5.3) stay hidden inside `post_message`: the service
checks whether the thread already has a `HEAD`/`TAIL` and dispatches the correct
single-`GRAPH.QUERY` write. The API only ever sees "post a message."

**The structure/diff reads — four operator-facing facts.** They answer *"is what I think is
published actually published"*, *"is the workspace running the same thing"*, and *"have `reference`
and `ws:{id}` gone stale independently"* without dropping to raw Cypher. (1) Both structure reads
are **whole-object and unpaginated** — there is no `?limit=`, deliberately: a truncated subgraph is
a *wrong* answer, not a partial one (an operator who gets 50 of 60 steps concludes ten are
missing). They are bounded upstream by the publish-time caps (`MAX_STEPS`, `MAX_TRANSITIONS`,
`MAX_CONFIG_LEN` — 200 steps, 500 transitions, 8000-char config/guard strings), matching the
unpaginated `QUERIES.md` §12 run reads; service-layer publishers bypass Pydantic, so those caps
are not universal. (2) The **diff** is bounded
instead by preview truncation (`MAX_DIFF_PREVIEW = 200`): its response is O(differences), never
O(def). (3) **The snapshot is what the executor drives** (`executor._drive_loop` → `get_snapshot`),
so `snapshot` is the operational truth and `def` (`reference`) is the intended truth. (4) The diff
is **version-qualified** — it answers "same version, different content", never "wrong version"; to
detect a stale *version*, compare `GET /workflow-defs` against `GET /workspaces/{ws}/snapshots`
first, or run `./scripts/verify_workflows.sh <wsId>`, which checks both seeded defs at their
expected versions in one command. There is no `latest` alias on the structure route: an operator
investigating a version mismatch must name the version.

> The publish/materialize receipt counts what was **submitted**; the structure read counts what is
> **stored**. **A divergence between the two is a signal, not an endpoint bug.**

`config`/`guard` come back **verbatim** as opaque strings (rule 8) — never parsed, re-serialized or
pretty-printed, so a whitespace-only divergence is still visible. The two structure bodies are the
same shape apart from `source` (so `jq`-diffing them by hand works) — **but that parity is the 200
body only**: the def route 404s through `WorkflowDefNotFoundError` (`{"error": …}`) while the
snapshot route raises a plain `HTTPException` (`{"detail": …}`), each mirroring its sibling
non-structure route's style. A client must not assume one error shape. These three routes are the
only ones on the router that declare a `response_model`; the rest deliberately do not (FastAPI's
`response_model` *filters* undeclared fields). The structure body omits `startKeys` in the
ordinary single-`START` case rather than sending `null`.

**`POST /workflow-runs` and `POST /workflow-runs/{runId}/input` drive the run synchronously**, not
on `BackgroundTasks` — they are the non-chat front door for a `kind:'process'` run, whose drive is
pure graph work with no LLM, so it is fast and deterministically testable.

**Bounds, and which layer owns which.** Pydantic bounds only what it can see: the
*submitted* dict (≤32 keys, key ≤ 200 chars, serialized ≤ 8000) and `maxSteps` (1…50). The
**merged** ctx bound, the reserved-key rule and the parked-step declaration check live in
`services.py`, because MCP tools and direct service callers never reach a pydantic model.

**Reserved run-ctx keys — `services.RESERVED_CTX_KEYS`: `threadId`, `error` and `timerFired` —
are rejected on both routes, in the service.**
`threadId` is the resume denorm anchor: a caller-set one would park a process run against a live
chat thread, and the trigger's step 2 would then advance it on the next ordinary human message
there — no input, no guard data. A process run parks with `waitingThreadId = ''`,
and the thread lookup short-circuits on an empty thread id from the other end.

**Error map.**

| Condition | Code |
|---|---|
| unknown run id; on `POST /workflow-runs`, a snapshot that is not materialized or has no START | **404** `WorkflowRunNotFoundError` |
| run is not parked, or lost the resume CAS (nothing written) | **409** `WorkflowRunNotWaitingError` |
| reserved key / undeclared key / value outside `config.expects` / oversized merged ctx | **400** `WorkflowInputRejectedError` |
| structurally malformed `cmp` guard (dominant source: publish-time validation) | **400** `WorkflowConfigError` |
| workflow engine not wired into this deployment: `POST /workflow-runs`, `POST /workflow-runs/{runId}/input`, `POST /workflow-runs/due` (the run reads still serve) | **503** `WorkflowEngineDisabledError` |
| **a fault *during* the drive**: `NotImplementedError` (unimplemented step type), `WorkflowConfigError` (malformed guard reaching evaluation), `ModelResolutionError` or `ProviderCallError` (any agent node or LLM guard) | **201/200** carrying `{"status":"failed","error":"<type>: <message>"}` |

That last row is the deliberate one: the executor's fault net has already stamped the run
`failed`, so the run *is* terminal and correct in the graph — a 500 traceback would misreport a
correctly-recorded terminal run as a server bug.

**Response bodies.** `POST /workflow-runs` answers `{runId, status, defKey, defVersion, trace[,
error]}`, with no `ctx`; `POST /workflow-runs/{runId}/input` answers `{runId, status, ctx[,
error]}`.

**The `/input` failed envelope reports graph truth, whole.** Its `status` **and** its `ctx` both come from
the *same* post-fault `get_run` re-read — never a re-read status beside the caller's submitted
input. So the `ctx` a caller gets back on the fault path is the engine's own state including the
diagnostic note `fail_run` stamped, not the merge that was attempted; on the clean path the two
are the same value by construction (the CAS wrote exactly that merge). Reporting one field from
the graph and the other from what the caller hoped happened would half-apply the rule, and the two
could disagree in exactly the situation where a reader most needs them consistent. If the re-read
status is anything other than `failed`/`done`/`waiting` the service **re-raises**, because
reporting a still-`running` zombie as success would be the worst outcome available.
**Step-budget exhaustion is not a fault**: the run is stamped `failed` with a `step budget
exceeded` note in its graph ctx, but the route answers through the clean path — `status: "failed"`,
no `error` field, and on `/input` the submitted merge as `ctx` rather than the graph's. The reason
is visible only through `GET /workflow-runs/{runId}`.

**The `/shop/api` storefront surface.** Eleven routes at `API_PREFIX = "/shop/api"`, present only
under `create_app(storefront=True)` (§1.3). The SPA build is mounted beside them at `/shop`, but
**only when `FALKORCHAT_STOREFRONT_DIR` names an existing directory** — unset (the default) or
mistyped and the mount is skipped silently while `/shop/api` serves normally.

**The `/shop` mount carries an HTML5-history SPA fallback, not a bare `StaticFiles(html=True)`.**
`salesperson/src/routes.tsx` uses `createBrowserRouter`, which needs the server to answer any
non-asset path under `/shop` with the SPA shell so the client-side router can take over — a plain
`StaticFiles(html=True)` only serves `index.html` at the mount's own root, and a Starlette `Mount`
is terminal (a 404 inside one never falls through to a route registered after it), so without
the fallback a deep route (`/shop/presenter`, a bookmark, a QR code, a page refresh) would 404
before the SPA loads. `app._SPAStaticFiles` answers any request under `/shop` that isn't a real
static asset and isn't under `/shop/api/*` with `index.html`; an unmatched API path still 404s,
never falls back to the shell.

**What holds for all eleven:** none resolves through the process-constant `get_context()`, and
`config.USER_ID` is reached by no route at all (it is touched once at startup, by
`services.ensure_actor` in `_lifespan`). **What does not hold for all eleven is the `actor`**, and
the table's `Cred` column is where the four shapes separate:

- **Five routes build `ctx` from the presented credential** — `GET /state`, `GET`/`POST /messages`,
  `POST /order/advance`, `POST /reset` — via `Storefront.context_for(participantId)`, so the cited
  query runs under that participant's own `actor` in `ws:{WS_ID}`.
- **`POST /session` builds `ctx` from the id it has just minted**, not from a credential; there is
  none yet.
- **`GET /catalog` authenticates a participant and then reads under the demo `Agent`.**
  `Storefront._catalog_ctx` is `CallContext(ws, actor=AGENT_ID)`, deliberately: the catalog is
  global `reference` data — `Repository.filter_products` takes **no `ws`** and no customer — so no
  participant identity is invented for a read that has nothing to do with one. That row's §15.2 is
  therefore reached under neither the participant's actor nor `ws:{WS_ID}`.
- **The four remaining routes build no `ctx` at all** — the two no-graph routes (`GET /health`,
  `POST /presenter/session`) issue no query, and the two presenter routes call
  `repo.list_participants(shop.ws)` / `reset_all_participants(shop.ws)` on the repository directly.

  (5 + 1 + 1 + 4 = 11.)

**Cred:** `P` = `get_participant`, `K` = `get_presenter`, `—` = none.

| Endpoint | Cred | Reaches | `QUERIES.md` |
|---|---|---|---|
| `GET /shop/api/health` | — | nothing — liveness plus the locale list the join screen's chooser reads | *(no query)* |
| `POST /shop/api/session` | — | `Storefront.join` → `ensure_participant`, `set_participant_record`, `services.save_profile` | §18.1, §18.3, §17.1 |
| `GET /shop/api/state` | P | `Storefront.get_state` → `get_profile` / `get_cart` / `get_current_order`, plus the in-process turn state. `get_cart` is **two** reads: `read_cart` in `ws:{WS_ID}`, then `lookup_products_by_id` against the **global `reference`** graph to price the lines — the only cross-graph read on the 2 s poll path | §17.2, §16.5 **+ §16.9**, §18.8 |
| `GET /shop/api/messages` | P | `services.read_messages` with an **explicit** `since` | §9.1 |
| `POST /shop/api/messages` | P | `services.post_message`, mentioning the demo agent — `thread_exists` + `resolve_member_kinds` **before** the write, which is where this route's `ServiceError` re-shapes come from — then `Storefront.enqueue_turn`, which books the turn and **submits** it. No further query runs on the request thread: the agent turn (`trigger.maybe_trigger` → `services.start_workflow_run`, §12.1) runs on a turn-executor worker (`storefront-turn_*`) after the `200` has been sent. **No message embedding** is scheduled | §2, then §4 (the turn's own §12.1 is off the request thread) |
| `GET /shop/api/catalog` | P | `Storefront.list_catalog` → `services.filter_products` + the startup image manifest | §15.2 |
| `POST /shop/api/order/advance` | P | `services.get_current_order`, then `Storefront.advance_own_order` → `order_belongs_to_customer` → `advance_order` | §18.8, §18.9, §16.10 |
| `POST /shop/api/reset` | P | `Storefront.reset_participant` — quiesce, then `repo.reset_participant`, then `services.save_profile` | §18.4, §17.1 |
| `POST /shop/api/presenter/session` | — | nothing — key → token, entirely in-process | *(no query)* |
| `GET /shop/api/presenter/participants` | K | `repo.list_participants`, projected to four keys | §18.3 |
| `POST /shop/api/presenter/reset-all` | K | `repo.list_participants` (pre-drain roster), then `repo.reset_all_participants` | §18.3, §18.5 |

**Request and response bodies.** `POST /session` takes `{displayName (1–60), language (1–32,
one of FALKORCHAT_STOREFRONT_LOCALES)}` and answers `{participantId, token, displayName, language,
welcome}` — the client presents `Bearer <participantId>.<token>` from then on. `GET /health`
answers `{status: "ok", storefrontEnabled: true, locales: […]}`. `GET /state` answers `{profile, cart, order, turn}`,
`turn` carrying `state`, `queuePosition` and `lastTurn` (`"failed"` after a turn that died without
a reply, else `null`). `GET /messages?since=&limit=` (`since` ≥ 0, default 0; `limit` 1–200,
default 50) answers the participant's own transcript; `POST /messages` takes `{text (1–2000)}` and
answers the posted message. `GET /catalog` answers every product with an `imageUrl`
(`/shop/products/<productId>.<ext>` when an asset exists, else `null`). `POST /order/advance`
takes `{transition: "fulfill" | "deliver" | "cancel"}`. `POST /presenter/session` takes
`{key (1–200)}` and answers `{token}`. `GET /presenter/participants` answers
`[{participantId, displayName, language, joinedAt}]`.

**No route accepts a client-supplied `threadId`, `customerId`, `orderId` or `ws`** — there is no
parameter to tamper with. (The *participant* id is client-supplied, inside
`Bearer <participantId>.<token>`, and becomes `ctx.actor` — which **is** the `customerId`; what makes
that safe is the `hmac.compare_digest` check against the graph row for that id, not the absence of
an id.) The thread id comes from the resolved `ParticipantRecord`, the order id from
`services.get_current_order` on the server side, the workspace from `Storefront.ws`. That is what makes `POST /order/advance`'s `404`/`409` mean *stale
button* and never *someone else's order* — and `advance_own_order` re-checks ownership before the
CAS anyway, so the guarantee holds at two layers rather than by one route's discipline.

**`GET /shop/api/messages` always passes `since` explicitly, and that is a classification fact.**
`services.read_messages` with no `since` takes its cursor path, which *advances the cursor* — a
write, on a route polled every 2 s.

**The cross-cutting error rule is computed, not listed.** `ROUTE_CLASSES` maps `(METHOD, path)` —
**not** path — to one of three classes (`writes`, `reads-only`, `no graph access`), and
`cross_cutting_response` turns a class plus a handler token into the answer: a lost connection is
always `503 graph_unavailable`; a query-time timeout is `503 graph_read_timeout` on a `reads-only`
route (nothing changed) and `504 <op>_state_unknown` on a `writes` route (the write may have
committed), while a `no graph access` route can produce none of the three. The method key is
load-bearing: `/shop/api/messages` is `reads-only` under `GET` and `writes` under `POST`, so a
path-keyed table would hand one of them the other's row. The live handlers and the test gate read
that **one** function, because a gate that agrees with a handler it does not share code with cannot
see that handler drift.

**Per-route errors are declared on the route and checked against what it can produce.** Each route
carries a `responses={…}` block naming its own statuses and their error tokens (`invalid_token`,
`turn_in_progress`, `no_current_order`, `order_transition_refused`, `quiesce_timeout`,
`unscoped_participant`, `unknown_participant`, `reset_state_unknown`, `demo_not_seeded`,
`validation_failed`, `bad_presenter_key`, `presenter_session_gone`, `wrong_credential_type`, plus
`unhandled` on the `500` rows of the two reset routes), and the gate reads them back off
`app.routes` rather than off the source — refusing a declared status
with no producer, a producer with no declaration, an unclassified route and a route with no
declaration at all. One token is emitted but declared nowhere: `_cross_cutting_json`'s defensive
`state_unknown` branch, which has **two** ways in — a route absent from `ROUTE_CLASSES` (the
`KeyError` arm), and a typed graph handler firing on a `no graph access` route, for which
`cross_cutting_response` returns `None`. Both are unreachable by construction —
the route-table assertion keeps `ROUTE_CLASSES` exactly the registered route set, and a
`no graph access` route issues no query for a typed handler to catch.

**The `ServiceError` handler wraps rather than replaces, and is path-scoped.**
`register_storefront_error_handlers` captures the incumbent handler and registers a wrapper that
re-shapes to the storefront's `{"error": token, "detail": …}` envelope only when the request path
starts with `API_PREFIX`, delegating everywhere else — so the dev envelope keeps exactly one
definition instead of a second copy that can drift. **No deployment carries both surfaces**, and
that — not the path check — is what keeps the dev app's error handling unchanged: this
handler is registered only when a `Storefront` exists, and `create_app` refuses `storefront` with
`dev_surface`. The path check is what makes the handler correct on an app that carries both, which
is a property of the handler rather than of any deployment. On `/shop/api` with **no** mapping it
logs and still delegates: a Python class name reaching a participant is a defect, but inventing a
token for a subclass nobody classified is a worse one. The set of routes on which a `ServiceError` can actually be raised is a **measurement**
(faults armed, all eleven routes driven), not a hand-list.

### 1.5 Layout

```
falkor-chat/
├── server/
│   ├── falkorchat/            # the package — one module per role, below
│   ├── tests/                 # pytest suite; tests/data/ holds the model-config fixtures,
│   │                          #   tests/eval/ the NLQ evaluation harness
│   └── pyproject.toml         # fastapi, uvicorn, falkordb, mcp; dev: pytest, httpx, ruff
├── web/{index.html, app.js, run-select.js}   # dev web UI; web/tests/ holds bare-node unit tests
├── config/{models.json, opencode.example.json, logging.json}
├── scripts/                   # bring-up, schema, seed and verify scripts (`falkor-chat/AGENTS.md`)
├── Dockerfile · compose.yaml
salesperson/dist/              # the storefront SPA build served at /shop (repo-root `salesperson/`)
```

| Module | Role |
|---|---|
| `app.py` | `create_app` / `_build_default_app`: wiring, mounts, error map, lifespan, the LLM guard judge |
| `config.py` | configuration keys and the `get_context` seam (§1.3) |
| `api.py` · `storefront_api.py` · `mcp.py` | the three front doors (§1.4, §2) |
| `schemas.py` | Pydantic request/response models and their bounds |
| `services.py` · `storefront.py` | domain logic; the storefront's participants, turn executor and resets |
| `repository.py` · `db.py` | every Cypher query; the FalkorDB connection |
| `background.py` | out-of-band job wrappers both transports schedule (§1.12) |
| `trigger.py` · `responder.py` | the post-message handlers (§1.12) |
| `executor.py` · `guards.py` · `proof_defs.py` | the workflow engine, transition guards, the shipped workflow defs |
| `tools.py` · `querygen.py` · `pricing.py` | the agent tool surface (§1.11), the constrained query DSL, cart arithmetic |
| `embedding.py` · `ingestion.py` · `chunking.py` · `extraction.py` · `fusion.py` · `update_detection.py` | embeddings and document ingestion |
| `modelconfig.py` · `transport.py` · `llm.py` · `retry.py` | model resolution (§1.8), HTTP transport, the OpenAI-compatible clients, provider retry |
| `logformat.py` | the log timestamp formatter (§1.10) |

`app.py` mounts every front door on one process and serves the dev UI `web/` as static files at
`/` (mounted **last**, since `/` is a catch-all that must sit behind the REST routes and the
`/mcp` mount), only under `dev_surface=True` and only when the directory exists. Serving the UIs
from the same process means there is no CORS seam. `web/` is plain HTML/JS with no bundler; it
calls the dev REST routes for channels, threads, messages, search, participants, workflow runs
(inline run cue and input form), the def list and def structure, snapshot structure and workspace
readiness, and its readiness
and snapshot calls name workspace `acme` in the path, which the server ignores (§1.4).

### 1.6 TDD build order

Not a current contract: build history is in `falkor-chat/docs/HISTORY.md`, and the test layout is
§1.5.

### 1.7 Testing hazards specific to `server/`

Gotchas that a green `pytest` run does not surface, distinct from the `test_queries.sh` teardown
hazard documented in `falkor-chat/AGENTS.md`'s "Key scripts" table:

- **`pytest -q` is destructive to the global `reference` graph too — a different mechanism than
  `test_queries.sh`'s teardown wipe.** The `wf_repo` fixture (`tests/conftest.py`) runs
  `MATCH (n) DETACH DELETE n` on `reference` at fixture **setup**, once per workflow test, to
  isolate it from earlier tests. Because the wipe never runs at teardown, a finished pytest
  session leaves the *last* workflow test's own published defs sitting in `reference` — so
  `already present — no-op` after a pytest run may be reporting a **test's** publish, not a real
  seed, while each `ws:<id>` snapshot still holds whatever it held before. Re-run
  `scripts/seed_workflows.sh <wsId>` after any pytest run, exactly as after `test_queries.sh`.
- **A green exit code is not evidence the graph-backed half of the suite ran.** With FalkorDB
  unreachable, `conftest._falkordb_reachable()` turns the whole integration suite into
  `pytest.skip` rather than failures, so the run still exits 0 with roughly half the tests
  silently skipped. Always read `N passed, M skipped`, never just the absence of failures.
- **`ruff check .` is clean but is not a wired gate.** `pyproject.toml` configures ruff and ships
  it as a dev dependency, but no script or hook runs it — a clean manual run is evidence of that
  one run only. The real gates here are `pytest` and (coordinator-run) `scripts/test_queries.sh`.
- **`ws:test`'s vector indexes are dim 4** (`conftest.TEST_EMBEDDING_DIM`), fixed at bootstrap and
  unrelated to the served workspaces' real dimension (1024/1536). Never point a real-embedder test
  at it: a wrong-dimension `vecf32` write is silently accepted by the engine and then drops out of ANN
  — the write "succeeds" and retrieval finds nothing, with no error anywhere in the chain.
- **`Services` and `WorkflowExecutor` each default to their own separately-defined `_default_clock`
  function** (`services._default_clock` vs. `executor._default_clock`), with zero wiring between
  them — including in production `app.py` wiring, which never passes `clock=` to either
  constructor. A test that
  injects `Services(clock=...)` to control run-parking timestamps silently fails to control
  `StepRun.startedAt`, since that field is stamped by the *executor's* clock, not the services
  clock — the test must construct `WorkflowExecutor` with the identical clock callable too, or the
  injection only half-takes.

**Verifying a claimed test count safely:** `pytest --collect-only -q` reports the suite's test
count with no FalkorDB connection and no writes — the correct way to check a plan's or review's
"N tests" baseline claim without triggering either the `wf_repo` setup-time wipe or
`test_queries.sh`'s teardown wipe above.

- **A wired agent requires the two config files of §1.8.** `tests/conftest.py`'s
  `_model_config_env` autouse fixture points both at `tests/data/` fixtures for every test — the suite must pass on a machine with **no**
  `~/.config/opencode/opencode.json` (verified: `HOME=<empty dir> pytest -q` is green). A test that
  needs a different value must override both the env var **and** the `falkorchat.config` module
  attribute (`monkeypatch.setattr`) — `config.py` resolves its env vars once at *import* time
  (no reload path), so a bare `monkeypatch.setenv` alone never reaches
  `ModelGateway.from_env()` once the module is already imported.

**QA/acceptance-testing gotchas, black-box-observed (distinct from the pytest hazards above):**

- **A `verify_workflows.sh` FAIL for `reference` (def MISSING) does not, by itself, block a live
  `@mention`-triggered workflow run.** `start_workflow_run`'s trigger/execute path reads only the
  workspace snapshot, never `reference` — the def routes (publish, list/get, structure, diff,
  readiness) and the catalog reads are what touch `reference`. Check which code path actually
  reads `reference` before treating a `verify_workflows.sh FAIL` as an environment blocker for a
  *behavioral* test.
- **A `WorkflowRun` parked `waiting` (`waitsForHuman`) resumes on the *next message posted to its
  thread*, whether or not that message `@mention`s the assistant.** A plain, non-mention message
  into a thread with an open `waiting` run silently resumes it. Only a fresh thread with **no**
  open run correctly exercises "an ordinary message never triggers a workflow" — reusing a thread
  from an earlier test item in the same pass will confound this check.
- **A step-budget failure carries no `error` in its response (§1.4).** A black-box check that
  sees a bare `status:"failed"` reads the reason with a follow-up `GET /workflow-runs/{id}`.
- **MCP-scheduled jobs do not survive a restart** (schedulers: §1.12). A black-box check that
  restarts the server mid-pass can lose MCP-scheduled replies, embeddings or extractions with
  nothing in any response saying so.
- **An `opencode.json` that omits `options.baseURL` parses but does not resolve** (the rule:
  §1.8; OpenCode's own `@ai-sdk/openai` has a default, this seam has none). It fails only at
  resolution (`ModelConfigError: ... no options.baseURL ...`). Any example
  or fixture file authored for this seam should be re-**resolved** once via `ModelGateway.resolve`,
  not just parsed, before being called documented or shipped.
- **`start_server.sh` runs uvicorn with `--reload` by default, watching the whole `falkor-chat/`
  tree — writing or editing ANY file there during a live acceptance pass (including your own
  `docs/test-plans/`/`docs/test-reports/` files) restarts the worker process and silently kills
  every in-flight background job** (the ingestion pipeline's extraction/embedding work, `mcp.py`'s
  daemon-thread dispatch). A file write mid-run produces false-negative-looking failures
  (entities/embeddings that never land) with nothing in the response indicating why. A reload
  mid-retry is worse: an ingestion job retrying a transient provider error stays in flight for
  typically 40–60 s and up to about 13 min, and a job killed then never reports, so its document
  stays `processing` for good. Disable reload for the pass (the `UVICORN_ARGS` gotcha below), and
  do all test-plan/test-report writes for a live pass *before* starting the server or after it's
  done — never interleaved.
- **Overriding `start_server.sh`'s `UVICORN_ARGS` to suppress `--reload` needs a genuinely
  non-empty value — `UVICORN_ARGS=""` does *not* disable it.** The script sets
  `UVICORN_ARGS="${UVICORN_ARGS:---reload}"`; bash's `:-` substitutes the default on **unset or
  empty**, so an explicit empty string is treated the same as never setting the variable at all.
  For a reload-free acceptance pass, pass any real, non-empty uvicorn arg string (e.g.
  `UVICORN_ARGS="--timeout-keep-alive 5"`) — never an empty override.

### 1.8 The model-resolution seam

**The rule, in one sentence:** every LLM/embedding consumer holds a `ModelGateway` and asks it
for a client; a directly-injected client (the `llm=`/`embedder=` constructor kwargs every consumer
accepts) is sugar `__init__` wraps into a `StaticModelGateway` — dependency injection for tests,
never a configuration route. There is exactly one internal path from "a kind + an optional
requested ref" to a working client, and no consumer reads an endpoint or model id from
`config.py` or any file directly. An AST check in `test_modelconfig.py` fails the suite if any
module outside `modelconfig.py`/`tests/` constructs `llm.OpenAICompatibleLLM`/
`embedding.OpenAICompatibleEmbedder` directly.

```
                     +-----------------------------------------------+
  opencode.json -->  | modelconfig.py                                |
  (pristine,         |   ProviderCatalog  <- parse + {env:}/{file:}   |
   shared)           |   Overlay          <- defaults . models        |
                     |   ModelGateway     <- resolve(kind, requested,  |
  models.json  -->   |                          ws, overrides)         |
  (falkor-chat       |                    <- .llm(...) / .embedder(...)|
   overlay)          +-------+-----------------------------------------+
                             | ResolvedModel(ref, base_url, model, key, timeout, params)
                             v
                     +------------------------------------+
                     | transport.make_http_transport()    |  timeout + headers + loud errors
                     +-------+----------------------------+
                             v
        OpenAICompatibleLLM / OpenAICompatibleEmbedder
                             ^
      +--------------+-------+--------+----------------+------------------+----------------+----------------+
   responder      judge            executor       query tool         embedding worker   retrieval tool   ingestion
   (kind=agent)   (kind=guard)     (kind=step)    (kind=step,        (kind=embedding)   + doc search     (kind=extraction)
                                                   inherits node                         (kind=embedding)
                                                   model)
```

**Consumer kinds and binding sites.** Resolution is **per call**, not at construction, so the
workspace override is a function of `ws` with no signature changes. The sites:

| Site | Kind |
|---|---|
| `AgentResponder.maybe_respond` | `agent` (answer) + `embedding` (query vector) |
| the executor's `_run_agent_node` | `step` |
| `QueryGraphDataTool.run` | `step`, inheriting the invoking node's `config.model` and override snapshot via `run["stepModel"]`/`run["modelOverrides"]`; with neither (e.g. `run={}` from the NLQ eval runner), the `step` default |
| `EmbeddingWorker.embed_message` / `embed_chunk` | `embedding` |
| `GraphragRetrieveTool.run`, `Services.search_documents` | `embedding` |
| the LLM guard judge (`app._LlmGuardJudge`) | `guard` |
| `IngestionPipeline` (per-chunk entity extraction) | `extraction` |

`modelconfig.KINDS` — the kinds a workspace override can reach — is the closed set `agent`,
`step`, `embedding`, `guard`. `extraction` resolves its default and timeout from the overlay like
any other kind but is outside `KINDS`, so **no workspace override reaches it** (open as K-080). A
new kind needs its own override property, or it escapes the hard cap the same way.

**Two hand-edited files** feed the gateway (`FALKORCHAT_OPENCODE_CONFIG` — a pristine, unmodified
OpenCode `opencode.json`, providers only, no product default; `FALKORCHAT_MODEL_CONFIG` —
falkor-chat's own overlay, default the shipped `config/models.json`, carrying per-kind `defaults`
and `timeouts`, per-model `models.<ref>` settings such as `dim` and `temperature`,
`providers.<id>` overrides and `roles`; unknown top-level keys are logged, never rejected). Both
are read once at wiring time (`ModelGateway.from_env()` — no reload path); `{env:NAME}` and
`{file:PATH}` references inside them are substituted, and an unresolved one is a startup error.
`config.assert_no_legacy_model_env()` refuses to start a wired gateway while any of
`config.LEGACY_MODEL_ENV_VARS` (`FALKORCHAT_LLM_BASE_URL`, `FALKORCHAT_LLM_MODEL`,
`FALKORCHAT_EMBEDDING_BASE_URL`, `FALKORCHAT_EMBEDDING_MODEL`) is set. Every provider
needs an explicit `options.baseURL` — there is no per-npm-package default.

**The `/v1` normalization rule.** LM Studio's `baseURL` convention omits `/v1`, and a missing
`/v1` is not an HTTP error — it is a `200` carrying an error envelope (a string on one
wrong-prefix path, an object on the right one, on the *same* server), so falkor-chat normalizes
rather than probes. Three ordered steps: **validate** (`scheme in {http,https}` and a non-empty
`netloc`, or reject at load naming the provider/file/value), **strip** every trailing `/`, then
**normalize** (append `/v1` only when the path is now empty; otherwise use it verbatim). An overlay
`providers.<id>.baseURL` override wins outright over both the file and the rule — used exactly as
declared, never auto-suffixed — and one INFO line per provider at startup names the declared
`baseURL`, the resolved API base, and which of {the rule, the override, verbatim} produced it.

**The `guard` kind's workspace carrier.** Three of the four override-capable kinds carry `ws` to
their resolution point via a `CallContext`; the LLM guard judge does not — `guards.evaluate_guard`'s
`ctx` is the *run* ctx dict, and `_select_transition` has no `CallContext` either.
`executor._drive` stamps `run["ws"] = ctx.ws` outside the SHA-locked `_drive_loop` boundary (a
fresh per-drive dict, never shared, never stored on `self`); `evaluate_guard` forwards `run=` to
the judge only when the judge advertises `accepts_run = True` (the production
`app._LlmGuardJudge`) and forwards `model=` only when the guard itself declared one
(`{"kind":"llm","text":…,"model":…}`) — both conditional kwargs, so a stub judge without them is
called without them.

**Roles and ordered fallback chains.** A ref with no `/` resolves as a **role name** — looked up
in the overlay's `roles` map and expanded to an ordered, settings-applied chain of
`provider/model` refs. A role name must not itself contain `/`, and a chain element that resolves
to another role is rejected at **load** time, not first use — roles never nest.
`ModelGateway.llm()`/`.embedder()` build a `FallbackClient` over the chain's resolved clients:
`.chat(...)`/`.complete(...)` try element 0, then element 1, … on a `ProviderCallError` (a
transport-layer `TimeoutError` converts to one), and raise naming **every** model tried only if all
fail. `FallbackClient` holds no mutable "last used" state (`__slots__` makes that structural) —
the answering model and whether it came from a fallback travel on the `ChatResult` return value:
`.model` (the answering ref) and `.fallback` (`True` iff a later element answered, `None` — never
`False` — for a one-element chain or a direct non-role ref).

**The resolved-model trace.** Each `StepRun` carries `resolvedModel`, `modelSource`
(`workspace`/`step`/`default` — the precedence rung that won) and `modelFallback` (nullable bool,
orthogonal to `modelSource` — a workspace override can itself resolve to a role with its own
fallback), written by the same atomic `record_step_and_advance` every run calls and surfaced on
`GET /workflow-runs/{id}/step-runs`. They are never a `TraceEvent`: those are debug-only (a
non-debug run writes none), so an audit field there would vanish on exactly the runs nobody
flagged for debugging. An agent node that loops and answers on more than one model records the
**last** iteration's three fields together, never a mix of iterations.

**The workspace override, precedence, and the hard cap.** A per-workspace `WorkspaceConfig`
singleton (one MERGE-backed node per `ws:{id}`) carries an optional per-kind override.
`ModelGateway.resolve()` is first-match-wins: **workspace → the consumer's own requested choice →
the per-kind default.** The workspace rung is a **hard cap**: when present it wins outright, even
over an explicit `requested=`, for all four `KINDS` — `guard` included, through the `run["ws"]`
carrier above. The executor and the guard judge read the override once per drive and pass it as
`run["modelOverrides"]`; the responder and the embedding consumers, which have no `run` dict, read
it per call through `GraphWorkspaceOverrides`.

**Publish-time rejection.** `publish_workflow_def` runs `_check_models_resolvable` after the
topology-conflict check and immediately before writing: every step's `config.model` and every
`{"kind":"llm"}` guard's `model` is resolved through the gateway (no `ws=`/`overrides=` — a global
publish is never gated by per-workspace state), and an unresolvable model or role fails the
publish with **`400 WorkflowDefSpecError`** naming the offending step key (or transition
endpoints) and the identifier — nothing is written. A def that fails **both** checks returns the
topology **409**, since the structural check runs first. A `Services` built without a gateway
(`FALKORCHAT_ENABLE_AGENT` off) skips the pass but logs a WARNING naming the def and its unchecked
identifiers if it declares any model or role.

**The embedding-dimension guard.** Before the first embed write for a `(workspace, label)` pair,
`EmbeddingWorker` compares the resolved embedding model's *declared* dimension (the overlay's
`dim`, else `FALKORCHAT_EMBEDDING_DIM`) against the workspace's *introspected* vector-index
dimension (`Repository.read_index_dimension`, cached per `(ws, label)` for the process lifetime —
never caching a failure) and raises `EmbeddingDimensionError` **before** calling the embedder on a
mismatch: no vector is written, no inference is wasted. A wrong-dimension `vecf32` write would
otherwise be accepted with no engine error and silently drop out of ANN.

### 1.9 Message assembly: `systemPrompt` vs. per-turn `CONTEXT`

Stated in `DESIGN.md` §6.1, agent node loop, "Prompt" bullet.

### 1.10 Logging

**The app installs no log configuration of its own; `config/logging.json` (repo path
`falkor-chat/config/logging.json`) is the configuration, applied by uvicorn's `--log-config`.**

- **Without it** the only handler on the root logger is the one FastMCP installs at import:
  `app.py` imports `mcp.py`, whose module-level `FastMCP(...)` calls `logging.basicConfig(level=INFO,
  format="%(message)s")`. INFO from every logger (`falkorchat.*`, mcp, httpx) then prints bare,
  with no time, level or logger name, and uvicorn's own lines carry no timestamp either.
- **With it:** every line starts with an ISO 8601 local timestamp with milliseconds and UTC offset
  (`YYYY-MM-DDThh:mm:ss.mmm-03:00`; the Docker container runs UTC, so `+00:00`), then the level.
  `falkorchat.*` lines also name their logger. `falkorchat.*` and uvicorn's error and access lines
  print at INFO, access lines on stdout and the rest on stderr. Every other logger prints at
  WARNING. The timestamp comes from `falkorchat/logformat.py`, uvicorn's formatter classes with
  one `formatTime` override, because a stdlib `datefmt` cannot put milliseconds before the offset.
- **Third-party INFO is dropped, deliberately.** That includes mcp's per-request
  `Processing request of type …` and `StreamableHTTP session manager …` lines, which would
  otherwise be most of a busy MCP server's INFO output. To get the per-request trace, give that
  logger its own handler in the file, e.g. `"mcp.server.lowlevel.server": {"handlers": ["app"],
  "level": "INFO", "propagate": false}`. A level alone is not enough, because root's handler only
  passes WARNING and above.
- **Who passes it.** `start_server.sh`, `start_agent_team.sh` and `start_demo.sh` resolve
  `UVICORN_LOG_CONFIG` (default: the shipped file, absolute) before step 1, fail loudly if it is
  missing, and append `--log-config` **after** `UVICORN_ARGS`, so a caller's override keeps it.
  A `--log-config` the caller puts in `UVICORN_ARGS` wins outright; the check normalizes
  whitespace first, so tabs count. The `Dockerfile` CMD passes `/app/config/logging.json`.
- **Who does not.** A bare `uvicorn falkorchat.app:app`, `load_test.sh`'s throwaway benchmark
  server, and pytest get none of it. A live run started outside the scripts passes
  `--log-config ../config/logging.json` from `server/`, or `--log-config config/logging.json`
  from `falkor-chat/`.
- **uvicorn's `--use-colors`/`--no-use-colors` are ignored under a file config** (uvicorn applies
  them only to a dict config), so colour follows whether stdout is a terminal. `--log-level`
  still reaches the `uvicorn.*` loggers but never `falkorchat`. Change either by editing the file
  or by pointing `UVICORN_LOG_CONFIG` at another one.
- **Configuring logging in app code was rejected.** It would also run under pytest, where
  `falkorchat.propagate: false` would blind every `caplog` test, and inside every embedding of
  the app.
- **Guard:** `server/tests/test_logging_config.py` applies the file, imports the app in a
  subprocess, and asserts how many handlers each (logger, level) reaches, 1 or 0, plus each line's
  stream and format.


### 1.11 Agent tool surface

A workflow `type:'agent'` node is offered the OpenAI function schemas of exactly the tools its
`config.tools` grants, and the executor dispatches each call through the one process-wide
`ToolRegistry` built by `tools.build_builtin_registry` (wired only with the workflow engine,
§1.3). Registry membership is not a grant: a tool a def never grants is never offered. A tool's
result is the string fed back to the model — a non-string is JSON-encoded. Tools hold no Cypher;
each goes through `services`. Every tool runs under the run's `ctx` (for a storefront turn, the
participant as actor); `post_message` posts as the agent.

| Tool | Arguments | Result |
|---|---|---|
| `post_message` | `text` (required), `mentions[]` (member ids) | Posts into the run's thread as the agent (role `assistant`) → `{"posted": msgId, "threadId": …}`; the executor links `StepRun -[:PRODUCED]-> Message` afterwards. `"error: …"` strings, not exceptions, for a run with no thread and for unknown mention ids (which names the id-vs-display-name confusion). |
| `graphrag_retrieve` | `query` (required) | Embeds the query (`embedding` kind), runs `services.hybrid_search` over messages and document chunks (k = 10), keeps seeds with cosine distance ≤ 0.5, at most 5 → `{"seeds": [{seedId, text, score, documentId}]}` (`documentId` `null` for a message seed), or `{"seeds": [], "finding": "no relevant context found"}`. |
| `human_handoff` | `reason` | Raises a suspend signal; registered, granted by no shipped def. |
| `lookup_product_fact` | `name` (required) | Exact catalog lookup (normalized name) → `{"found": true, …product row}`, or `{"found": false, "hint", "catalog": [{productId, name}]}` — the whole catalog by name, no prices or categories. |
| `filter_products` | `category`, `minPrice`, `maxPrice` (all optional, inclusive bounds; blank counts as omitted, numeric strings are coerced) | `{"items": [...]}` (at most 20). Empty: an unknown `category` → `{"items": [], "finding": "no catalog category has that name", "hint", "catalog": [{productId, name}]}`; any other empty result → `{"items": [], "finding": "no matching products found"}`. |
| `view_cart` | — | `services.get_cart` → `{"items": [...], "total": …}`, prices resolved live from the catalog; an empty cart is `{"items": [], "total": 0.0}`. |
| `add_to_cart` | `productId` (required), `quantity` (≥ 1, default 1) | Quantities accumulate → `{"found": true, …line}`, or `{"found": false, "reason": …}` for an id that does not resolve. The executor holds a `productId` no grounding tool returned this turn before the tool runs. |
| `remove_from_cart` | `productId` (required), `quantity` (omit to remove the whole line) | `{"found": true, "removed": …, …}` (`removed: false` when the product was not in the cart), or `{"found": false, "reason": …}`. |
| `clear_cart` | — | `{"cleared": true}` (also for an already-empty cart). |
| `place_order` | — | Snapshots the cart at current catalog prices with a computed total and clears the cart → the order as JSON; on an empty cart, the plain string `The cart is empty — add an item before placing an order.` |
| `get_profile` | — | `{"name", "deliveryAddress"}`, absent fields `null` — never a not-found shape. |
| `save_profile` | `name`, `deliveryAddress` (both optional) | Upserts; an omitted field keeps its stored value, never clears it → the saved profile. |
| `query_graph_data` | `question`, `dataset` (both required; `dataset` ∈ `catalog`, `knowledge_base`) | Below. |

**`query_graph_data`'s contract.** The tool's own arguments are not the answer: it makes a
second, internal LLM call (`step` kind, on the invoking node's model, §1.8) whose only accepted
output is a `querygen.QueryRequest`, compiles it with `querygen.compile` against the dataset's
allowlist and runs it read-only (the query shapes and guards are `QUERIES.md`'s). The datasets:

| `dataset` | Graph | Labels and queryable properties | Identity (first = prepended column) | Facets |
|---|---|---|---|---|
| `catalog` | `reference` | `Product{name, nameNormalized, category, price}` | `name`, `nameNormalized` | `category` |
| `knowledge_base` | `ws:{ws}` | `Entity{entityId, name, nameNormalized, type}`, `Document{documentId, title, sourceFormat}`, `Chunk{chunkId, text, seq, documentId}` | `Entity`: `name`, `nameNormalized`, `entityId`; `Document`: `title`, `documentId`; `Chunk`: `chunkId` | `Entity.type`, `Document.sourceFormat` |

A request is one label match with ≤ 4 filters (`=`, `<>`, `<`, `<=`, `>`, `>=` on a registered
property), 1–6 `returns` (`var.property` or `count`/`avg`/`min`/`max` of `var` or
`var.property`), an optional `order_by`/`order_dir`, and `limit` 1–50 (default 20). For the
`catalog` dataset the internal prompt also lists the catalog's stored categories.
A completion that omits `returns` entirely, with one match whose label has a `name` property, is
given `returns: ["<var>.name"]` and answers a name list, unless its filters make it a
single-entity lookup (an `=` filter on an identity property), which stays invalid and abstains
with `no matching data found`. An explicit empty `returns` is still rejected.

The result is `{"items": [row, …]}`, **each row a JSON object keyed by its `returns` expression
text** (e.g. `{"p.name": …, "p.price": …}`). **Every per-row attribute row names its entity**: when
the projection carries no identity property, the label's first identity column (`p.name` for
`Product`) is prepended, so it is each row's first key. Left as requested: an aggregate (alone or
grouped), a projection that already carries an identity property, a plain facet enumeration (only
facet properties, ordered by nothing outside `returns`), and a single-entity lookup (an `=` filter
on an identity property). The tool never surfaces a `productId`.

Empty results and abstentions:

- an empty `catalog` result whose `=` filter names a `category` not in the stored list, or a
  `name`/`nameNormalized` that does not resolve exactly as written → `{"items": [], "finding":
  "no catalog category or product has that exact name", "hint": …, "categories": [...]}`, the hint
  routing to `filter_products` and `lookup_product_fact`;
- an unknown `dataset` → `{"items": [], "finding": "unknown dataset"}`;
- every other empty result, and every unparseable, schema-invalid or allowlist-rejected
  completion → `{"items": [], "finding": "no matching data found"}`.

A model-resolution or provider-call failure is not caught: it propagates to the executor as an
infrastructure fault, as it does for `graphrag_retrieve`'s embedder call.

### 1.12 Post-message and ingestion out-of-band policy

A posted message is written, readable and answered before anything else runs; everything else is
scheduled out-of-band on wrappers in `background.py` that log and swallow every failure, so no
background fault reaches the poster.

**On every post** (REST `POST /threads/{tid}/messages`, MCP `send_message`), in order:

1. **embed** the message, when an embedding worker is wired (`FALKORCHAT_ENABLE_AGENT`);
2. **exactly one handler**: the `WorkflowTrigger` when the workflow engine is wired, else the
   `AgentResponder` when the agent is wired, else nothing. An `@mention` can never fire both a
   workflow and a direct reply.

**The trigger's ordered rule** (`WorkflowTrigger.maybe_trigger`) — the first that applies acts:

1. **loop guard** — an agent-authored message (`role == "assistant"`) is ignored;
2. **resume** — if the thread has a `waiting` run, the message resumes it, mention or not
   (single-flight CAS; a concurrent loser no-ops) and the rule stops there;
3. **start** — if the message `@mention`s `FALKORCHAT_AGENT_ID`, start a run of
   `FALKORCHAT_TRIGGER_DEF_KEY`@`_VERSION` triggered by this message;
4. **fall-through** — hand the message to the held responder, or to nothing when
   `FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH` is cleared.

**The responder** (`AgentResponder.maybe_respond`) answers only a non-agent message that
`@mention`s `FALKORCHAT_AGENT_ID`: it embeds the question, retrieves the 10 nearest seeds
workspace-wide (messages and document chunks, via `services.hybrid_search` with no channel scope), asks the `agent`-kind LLM, posts the
answer as the agent with the seeds recorded as `EMITTED` provenance, then embeds its own answer.
Any failure before the post posts nothing.

**On every ingest** (REST `POST /documents[/batch]`, MCP `ingest_document(s)`), per successfully
ingested document: when either is wired, one embed job and one entity-extraction job per chunk,
independent of each other; the document's `status` is `processing` until every job reports, then
`ready`, or `failed` if any failed. With either wired, and unless the ingest was
auto-superseded synchronously, a separate update-detection job looks for a `SUPERSEDES`
suggestion. With neither wired (`FALKORCHAT_ENABLE_AGENT` off, the default), nothing is
scheduled: the document stays `processing` and no update detection runs.

**Schedulers differ by transport.** REST schedules on FastAPI `BackgroundTasks`, which run after
the response on anyio's thread limiter (`FALKORCHAT_THREAD_LIMIT`). MCP spawns one daemon thread
per job, unbounded, inside the tool call before it returns — a max-size batch ingest can spawn
about 23,500 (20 documents × ⌈500,000 / 850⌉ chunks at size 1000, overlap 150 × 2 jobs) — and a failure to start a thread is logged and that one job skipped. MCP threads are
not tracked by `_lifespan` and die silently with the process. A storefront post schedules none of
this: its turn runs on the storefront's turn executor (§1.4) and its message is not embedded.

---

## 2. MCP transport — the agent front door

A second transport for AI agents: **MCP over Streamable-HTTP**, mounted at `/mcp` on the *same*
FastAPI process and calling the *same* `services.py` as the REST router. It exists only under
`dev_surface=True` (§1.3) — the storefront deployment has no `/mcp`.

### 2.1 Shape

```
browser ── REST/JSON ──┐
                       ├─▶ services.py ─▶ repository.py ─▶ FalkorDB
agents  ── MCP/HTTP ───┘   (all invariants here; both front doors call the SAME methods)
```

`mcp.py` is a thin adapter (peer of `api.py`), no business logic. `create_app()` passes
`mcp.configure(...)` the same `Services`, context provider and out-of-band collaborators
(embedding worker, ingestion pipeline, trigger or responder, repository) it gives the REST
router, so both transports run the identical post and ingest policy (§1.12), then:

```python
mcp_app = mcp.streamable_http_app()
app = FastAPI(lifespan=...)          # _lifespan enters mcp_app.router.lifespan_context
app.mount("/mcp", mcp_app)           # agents connect at /mcp
```

> **Lifespan:** the app's `_lifespan` runs the MCP app's lifespan
> (`mcp_app.router.lifespan_context`) around its own `yield`; without it the Streamable-HTTP
> session manager is never started and requests fail with "task group not initialized"
> (python-sdk #1367). The handler's own path is `/` (`mcp.settings.streamable_http_path = "/"`),
> so the mount yields `/mcp` rather than `/mcp/mcp`.
>
> **Trailing slash:** a Starlette `Mount` serves the sub-app only under `/mcp/`, and MCP clients
> don't auto-append the slash, so `create_app` adds an ASGI path-alias middleware rewriting
> `/mcp` → `/mcp/`; both spellings work.

### 2.2 Tools → service → query

Every tool resolves `ctx` through `get_context` (§1.3). A service error surfaces as an MCP tool
error.

| MCP tool | Service method | `QUERIES.md` |
|---|---|---|
| `send_message(body, re, mentions=None, frm=None)` | `post_message` (+ the out-of-band policy, §1.12) | §2 member-kind lookup, §4 first/subsequent (+ mentions) |
| `read_messages(re=None, since=None, limit=50, advance=True)` | `read_messages` | §9.1 (thread) / §9.2 (room-wide); §9.3/§9.4 cursor |
| `create_thread(channel_id, title)` | `create_thread` | §3 create a thread |
| `create_channel(name)` | `create_channel` | §3 create a channel |
| `list_channels(limit=50)` | `list_channels` | §3 list channels in a workspace |
| `list_threads(channel_id, limit=50)` | `list_threads` | §3 list recent threads in a channel |
| `search_messages(query, limit=50)` | `search_messages` | §5 full-text keyword search |
| `ingest_document(text, title=None, source_format="text", source_label=None, produced_by=None)` | `ingest_document` (+ §1.12) | §14.9 |
| `ingest_documents(items)` | `ingest_documents` (+ §1.12) | §14.9 per item |
| `get_document(document_id)` | `get_document` | §14.2 (`null` when absent) |
| `search_documents(query, limit=20)` | `search_documents` | §14.3 |
| `delete_document(document_id)` | `delete_document` | §14.7 |
| `list_documents(current_only=True, limit=50)` | `list_documents` | §14.7 |
| `get_document_deletion(document_id)` | `get_document_deletion` | §14.7 (`null` if never deleted) |
| `get_document_history(document_id)` | `get_document_history` | §14.8 |
| `list_pending_document_updates(limit=50)` | `list_pending_document_updates` | §14.8 |
| `list_document_updates(status=None, limit=50)` | `list_document_updates` | §14.8 |
| `confirm_document_update(match_id)` · `reject_document_update(match_id)` · `recheck_document_update(match_id)` | same-named service methods | §14.8 |
| `list_pending_matches(limit=50)` · `list_matches(status=None, limit=50)` | same-named service methods | §14.6 |
| `confirm_match(match_id)` · `reject_match(match_id)` · `recheck_match(match_id)` | same-named service methods | §14.6 |

Each document and review tool has the semantics of its REST twin (§1.4).

- **Actor identity:** MCP ignores any client-supplied `frm`; every call is attributed to the
  `get_context()` actor (§1.3), the configured `User`. `ingest_document`'s `produced_by`, when
  given, attributes the document to that existing `Agent` instead and fails with
  `AgentNotFoundError` if there is none — never a silent fall-back to the actor.
- **`read_messages` is RW when it advances a cursor.** With `re` and no `since`, the per-thread
  composite cursor `(lastReadAt, lastReadMsgId)` is read and, unless `advance=False`, advanced to
  the newest row actually delivered — never the server clock, which would permanently skip rows a
  `limit` truncated (an empty page advances nothing). Cursor reads never skip or re-deliver, even
  across same-millisecond messages. An explicit `since` is a pure, plain `>` read that may
  re-deliver or skip within that exact millisecond. Rows are chronological, carry `threadId`, and
  flag reader-mentions with `isMention` (`QUERIES.md` §9 ordering note). Room-wide reads (no `re`)
  default `since` to epoch 0 and never advance (there is no room cursor).
- **No schema layer on MCP.** Tool arguments reach the service without Pydantic, so only the
  service-enforced bounds apply: the document and batch caps hold, but a document `title` over
  200 characters, which REST rejects, is accepted (K-074). `ingest_documents` isolates a
  malformed item (missing or non-string `text`, a non-dict entry) as its own
  `{"status":"error","errorType":"MalformedItemError",…}` receipt.
- **REST mention parity:** `POST /threads/{tid}/messages` also accepts an optional `mentions[]`.

### 2.3 Client connection contract

Streamable-HTTP; a consuming agent points at the URL (no subprocess):

```json
{ "mcpServers": { "falkor-chat": { "type": "streamable-http", "url": "http://localhost:8000/mcp" } } }
```

Unauthenticated — bind to localhost or a trusted network only. Run:
`cd server && .venv/bin/uvicorn falkorchat.app:app` (bootstrap `ws:{FALKORCHAT_WS_ID}` first, by
default `ws:acme`).
