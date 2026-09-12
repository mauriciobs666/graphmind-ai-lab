# falkor-chat — Server Application & MCP Transport

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.0

> **Scope.** The **server process**: internal layering, the auth/tenancy seam, the REST and MCP
> front doors, the on-disk layout, testing hazards, and the model-resolution seam. The **graph**
> — topology, data model, indexes, capacity, operations — is `falkor-chat/docs/DESIGN.md`.
> Canonical Cypher is `falkor-chat/docs/QUERIES.md`; executable DDL is
> `scripts/bootstrap_schema.sh`. This document states the system as it is now; *when* something
> changed lives in `falkor-chat/docs/HISTORY.md`, unbuilt work in
> `falkor-chat/docs/BACKLOG.md`.

> **Moved from `DESIGN.md` on 2026-08-24.** §1 was `DESIGN.md` §14 and §2 was `DESIGN.md` §15;
> subsection numbers map straight across (old §14.7 → §1.7, old §15.3 → §2.3). `DESIGN.md`
> carries the redirect table for citations written before the move.

---

## 1. Application architecture (client/server)

`DESIGN.md` §10 sketches the *operational* topology (app ⇄ FalkorDB). This document pins the
*application* code architecture: what the client and server are, the transport between them, and
the internal layering. The scope decisions in §1.1 were locked for M1 and still hold.

### 1.1 Scope decisions locked for M1

| Axis | Decision | Rationale |
|---|---|---|
| **Transport** | **REST/JSON over FastAPI** | The only M1 client is a browser, which speaks HTTP natively — no gRPC-Web bridge tax. Free OpenAPI console to exercise the API. M2.5 real-time adds native WebSocket/SSE on the same server. |
| **Client** | **Minimal web UI** (channels list + thread view) | Smallest end-to-end path that exercises the full stack visually. |
| **Real-time** | **Deferred to M2.5** | M1 is request/response; the UI re-fetches a thread window after posting. The push path (Redis Pub/Sub → WebSocket) slots onto the same service layer in M2.5 with no schema change. |
| **Auth / tenancy** | **Single hardcoded tenant** — `ws=acme`, `user=u1` | Keeps M1 focused on the chat data path. Injected at one seam (see §1.3) so real auth replaces it without touching services/repo. |

> Transport was deliberately re-evaluated away from gRPC: gRPC's wins (polyglot typed contracts,
> native streaming, service-to-service perf) are all unused when the sole client is a browser, and
> gRPC-Web can't do client/bidi streaming in browsers anyway — WebSocket/SSE is the stronger M2.5
> real-time path. REST keeps the layers below the router transport-agnostic, so a gRPC servicer or
> a service-to-service hop can still be bolted onto the same `Service` later if a non-browser
> consumer ever appears.

### 1.2 Layering

```
┌─ Browser (minimal web UI) ─┐                ┌─ Python server (FastAPI, one process) ───────┐
│ channels | thread view     │   REST/JSON    │ api.py      router (thin: HTTP ⇄ Service)    │
│ post / read / search       │ ─────────────▶ │   ▲  CallContext dep = {ws:acme, actor:u1}  │
└────────────────────────────┘                │ services.py  domain logic, append dispatch   │
                                              │ repository.py  Cypher ⇄ QUERIES.md (RO|RW)   │
                                              │ db.py        falkordb-py conn, select_graph  │
                                              │ modelconfig.py  the model-resolution seam    │
                                              │   (§1.8) — every LLM/embedding call          │
                                              │ transport.py    ← one HTTP transport, §1.8   │
                                              └────────────────────────────────────────────┬─┘
                                                                                            ▼  FalkorDB / LLM providers
```

- **`repository.py` is the only place Cypher lives.** Each method maps 1:1 to a verified query in
  `QUERIES.md`, always parameterised (`params=`), `ro_query` for reads / `query` for writes,
  `select_graph(f"ws:{id}")` for scoping.
- **`services.py` owns the invariants** the write-path rules describe: choosing the first-vs-subsequent
  append variant, id generation, `Thread.updatedAt` bumps, setting `role`/`POSTED_BY`.
- **`api.py` is the only layer that changes** if the transport is ever revisited.
- **`modelconfig.py`/`transport.py` (§1.8) are the model-resolution seam.** Every LLM/
  embedding consumer (`responder.py`, `executor.py`, `guards.py`, `embedding.py`, `tools.py`) holds
  a `ModelGateway` and resolves per call — never constructs `llm.py`'s/`embedding.py`'s
  OpenAI-compatible clients directly (FR-4).

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

It is **process-constant, not literal**: both values come from the environment at import time, which
is what lets a deployment point the process at its own workspace (§4.9). What it is not is
*per-caller* — every REST and MCP call resolves to the same actor.

Services and the repository already take `ws` / `actor` as parameters, so when auth lands
(token → user + workspace claim, or the `identity` graph as source of truth) **only `get_context`
changes** — everything below is untouched.

**Until it lands, the whole REST router is unauthenticated**, and that is fine only while every
caller is trusted. It stops being fine the moment mutually-untrusting people share one workspace:
`GET /channels`, `GET /threads/{tid}/messages`, `GET /search?q=` and the post box all resolve
through the process-constant `get_context()`, so any client on the network can read every
transcript in the workspace and write into any thread. `/mcp` and the `/` web mount are the same
seam again.

The salesperson storefront (`docs/plans/salesperson-ui.md` §4.9) closes this **by construction
rather than by configuration** — `create_app` takes:

```python
create_app(services, dev_surface=True)   # the default: today's app, unchanged
create_app(services, dev_surface=False)  # no api.build_router, no `/` mount, no /mcp
create_app(services, dev_surface=False, storefront=True, storefront_dir=...)
                                         # + the /shop/api router, its typed error map, the /shop SPA mount
```

`dev_surface=False` un-mounts the entire unauthenticated dev surface. It dominates `mount_mcp`, so
`/mcp` is absent however the app was constructed. Because `GET /health` lives *inside*
`api.build_router`, that branch re-registers a bare liveness route with the same contract
(`services.ping`, 503 when FalkorDB does not answer) — exactly one `/health` exists in either
configuration.

`create_app` **refuses `storefront=True` with `dev_surface=True`** — the storefront deployment must
not mount the unauthenticated legacy surface, and that is enforced at construction rather than left
to the operator. `storefront_dir` is read **once** and forwarded to both the `Storefront` (which
builds the product-image manifest from `<dir>/products/`) and the `/shop` static mount, so the
manifest is always built from what is actually served.

**`dev_surface` is a function parameter and deliberately not an environment variable.** That is
the whole point: `_build_default_app` derives it (and `mount_mcp`) as
`not config.STOREFRONT_ENABLED`, and there is no operator setting that can put the legacy surface
back while storefront participants exist. Assert its effect on `app.routes`, never by probing for
a 404 — a 404 is returned both when a route is absent and when it exists but errors.

| Env var | Default | Effect |
|---|---|---|
| `FALKORCHAT_STOREFRONT_ENABLED` | off | **When set (`=1`)**, `_build_default_app` builds the storefront deployment — `storefront=True` (the `/shop/api` router, its error map, the startup preflight, and the `/shop` SPA mount **only when `FALKORCHAT_STOREFRONT_DIR` names an existing directory**) with `mount_mcp=False`, `dev_surface=False`. Unset, the app shape is unchanged |
| `FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH` | **on** | **Read only where the workflow engine is wired** (`app.py:553`, inside `if config.WORKFLOW_ENABLED:` on the `ENABLE_AGENT` path). There, **cleared (`=0`)** builds `WorkflowTrigger(responder=None)`: a message matching no workflow reaches nothing instead of the M2 responder, whose retrieval is workspace-**wide** (`hybrid_search` with `channel_id=None`) and would otherwise surface another participant's messages. Left on, the M2 fall-through behaves as it always has. **With either flag off — and both default off — setting this changes nothing**: that path wires the responder straight into `create_app` (`app.py:569`), never through a trigger |
| `FALKORCHAT_STOREFRONT_DIR` | unset (`None`) | The **served** SPA build directory (`salesperson/dist/`), mounted at `/shop` and the root of the product-image manifest (`<dir>/products/`). Unset resolves to `None`, never `""` — `Path("")` is the process working directory, which would silently serve whatever the operator happened to `cd` into. The manifest is built from what is *served*, never from the source tree. **Unset or naming a nonexistent directory skips the `/shop` mount entirely and silently** (`create_app` guards it with `Path(served_dir).is_dir()`): the manifest is empty, the preflight logs `images=0` and continues, and `/shop/api` serves normally — so the default configuration is a storefront API with no SPA behind it |
| `FALKORCHAT_STOREFRONT_PRESENTER_KEY` | `""` | The one operator secret, exchanged at `POST /shop/api/presenter/session` for a presenter bearer token. Demo-session scoping, not authentication. **That route is delivered, so setting a key today makes presenter login work** — by `curl` or a test; the `/shop/presenter` *screen* that types it is **not built yet — S12b/S12d**. **Empty means "no presenter surface" and must never authenticate**: `hmac.compare_digest("", "")` is `True`, so the login path rejects an unset key *before* comparing (`Storefront.presenter_configured` is that check). Left unset — the default — every login answers the same `403 bad_presenter_key` a wrong key gets, deliberately, and the operator's signal is a log line rather than the response |
| `FALKORCHAT_STOREFRONT_TURN_WORKERS` | `4` | **`max_workers` of the storefront's own bounded turn executor** (§4.4 measure 1), which is where every agent turn runs: `POST /shop/api/messages` reserves the turn, writes the message, calls `Storefront.enqueue_turn` and answers, and `trigger.maybe_trigger` → `services.start_workflow_run` runs on a worker named `storefront-turn_*`. **Setting it changes how many turns run at once — and with that, the `turn.queuePosition` participants are shown.** That number is a participant's index in the *waiting* line, derived on every `GET /shop/api/state` from the turn map's booking ordinals and never stored, so `turn_workers` is not a term in it — but a running turn occupies a worker rather than a place in line and is left out of the count, so more workers means fewer turns still waiting ahead of anyone. Measured on five simultaneous arrivals, the fifth one's reported position: `turn_workers=1` → `3`, `=2` → `2`, `=4` → `0`. Raising it to shorten the queue does shorten what the queue reports. Sized to LM Studio's configured parallelism, which is the reason for the default: it tracks the model server, not the web tier. Turns never touch anyio's thread limiter, so a deep queue costs the 2 s poll nothing |
| `FALKORCHAT_STOREFRONT_QUIESCE_S` | `30` | **How long either reset waits for a still-in-flight turn to drain before giving up and changing nothing** — which, since S9b, is *every* turn under reset-everyone and only the ones reset-mine's cancel could not reach. The measurements below were taken against a **held (running)** turn and are unaffected by S9b; what S9b narrowed is *which* turns reach this wait at all. Reset-**mine** now cancels the participant's turn first when it is still sitting in the executor's queue (`Storefront._cancel_queued_turn`, in front of the wait and never in place of it), so this budget no longer applies to a queued turn of theirs — a `quiesce_s=0` reset-mine over a queued turn resets rather than refusing (`tests/test_storefront.py::test_a_reset_cancels_a_queued_turn_where_it_used_to_refuse`), which is the availability §4.8 asked for. It still applies in full to a turn already on a worker, to one reserved by a second tab still inside its message write, and to **every** turn under reset-**everyone**, which cancels nothing. S9 wired `Storefront.set_turn_state` into the turn executor, so the turn map is populated for every turn and this wait now has something to act on, measured both ways: a short window (`0.4 s`) against a held turn answers `503 {"error":"quiesce_timeout"}` after ~0.41 s and resets nothing — message count and the original `Thread` unchanged; a window that comfortably covers the turn (`5.0 s` against a 0.8 s turn) instead waits it out (~0.77 s) and then resets, dropping the old `Thread`. While a turn is live, a second post from the same participant answers `409 {"error":"turn_in_progress"}` and is not written, and `GET /shop/api/state` reports `{"state": "thinking", "queuePosition": 0, "lastTurn": null}` rather than the idle payload — `lastTurn` (S9c) is a separate per-participant latch, not part of this map entry, so it survives the map write that later returns that participant to `idle` and is `"failed"` only after a turn that died without a reply. **Setting the value changes the outcome in both directions** — raise it and a reset outlasts the turn instead of timing out. **Reset-everyone's stop-intake precedes the wait, delivered at S10** (`Storefront.reset_all`): a set `_intake_stopped` refuses every participant's post with the same `409 turn_in_progress` a busy one already gets, before the message write, so nothing accepted after the pre-drain roster read can extend this budget — it comes back down on every exit from `reset_all`, success or failure alike, so a failed sweep does not lock the demo out of posting |
| `FALKORCHAT_STOREFRONT_PRESENTER_LOGIN_DELAY_S` | `1.0` | **`presenter_login`'s fixed per-attempt delay** (S10), applied to every attempt alike — wrong key or right — before either check, so response timing cannot distinguish the two beyond what `hmac.compare_digest` already refuses to. Paired with `Storefront.presenter_login_failures`, an observational-only count logged on every refusal and never consulted by `presenter_login` itself. **Deliberately not a lockout after N attempts** (Pass 7, P7-5): with exactly one shared presenter key, a lockout is a self-DoS — anyone on the LAN could lock the presenter out of their own demo mid-show — so the response set here stays `200`/`403`/`422` and the cross-cutting three, never a `429`/`423` |
| `FALKORCHAT_STOREFRONT_LOCALES` | `en,pt-BR,es` | The languages a participant may join in, in the UI's offer order. **Setting it changes two delivered answers today**: the set `POST /shop/api/session` validates `language` against (anything else is `422 validation_failed`) and the list `GET /shop/api/health` reports — the SPA locale chooser that reads that list, and the bundles behind it, are **not built yet — S12c**. Not a `Literal`: `JoinIn.language` is a length-bounded string and the route checks membership against this per-deployment list, raising the same `RequestValidationError` Pydantic would. A blank or all-separator value falls back to the default rather than yielding an empty set, which would reject every language a participant could pick |
| `FALKORCHAT_THREAD_LIMIT` | `100` | **The size of anyio's default thread limiter** (§4.4 measure 2) — how many sync endpoints FastAPI will offload concurrently; anyio's own default is 40. `create_app`'s `_lifespan` assigns it **before `yield`**, which is the only place it can be assigned: the limiter is event-loop scoped and raises `anyio.NoEventLoopError` outside a running loop. **It applies to every deployment this app builds**, not only the storefront one, so setting it changes the legacy surface's concurrency too. Headroom for the poll path and explicitly **not** load-bearing — the turn executor above is what keeps agent turns off this limiter, which is why the value is generous rather than tuned |

**There is no `FALKORCHAT_DEMO_WS`, and there never will be.** The storefront's workspace *is*
`config.WS_ID` (`salesperson-ui.md` §4.9 move 2): a second workspace variable buys nothing once
the unauthenticated surfaces are un-mounted, while creating the only thing that made the
seed-the-wrong-graph trap possible — two variables that can disagree. `tests/test_storefront.py`
carries the tripwire against reintroducing the second. **`scripts/start_demo.sh`, which is designed
to pin the one variable to a dedicated value, does not exist yet — S11**; until it does, an operator
pins `FALKORCHAT_WS_ID` by hand and `config.py`'s `"acme"` default is the repo's populated dev
workspace, not a demo one.

**The storefront adds a *second* auth path, and it is per-caller.** `get_context()` above is
process-constant, and **no `/shop/api` request resolves through it** — the one thing the storefront
takes from that seam is its *workspace*, read once at construction as `provider().ws`
(`app.py:324`), so an injected `context_provider` pins the storefront too. The `actor` half is
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
  minted. A client or test written from the looser reading asserts the wrong code. Presenter tokens
  are `secrets.token_urlsafe(32)` minted in-process by `_PresenterSessions` **in exchange for**
  `FALKORCHAT_STOREFRONT_PRESENTER_KEY` rather than derived from it, and are **not** graph state, so
  they do not survive a restart — the operator logs in again.
- **This is demo-session scoping, not authentication** (`docs/plans/salesperson-ui.md` §4.3;
  K-016/K-017/K-018 stay open). What it does give is the per-participant tenancy the platform seam
  cannot: `ctx.actor` is the participant id and every cart/order/profile service keys on it, so
  isolation between mutually-untrusting participants sharing one workspace is **structural rather
  than filtered**.
- **Startup refuses a mis-seeded deployment rather than coming up green and dead.**
  `storefront_preflight` runs from `_lifespan` and raises `StorefrontPreflightError` — naming the
  fix command — unless the demo `Agent` resolves in `ws:{WS_ID}` (asked with the *same*
  `resolve_member_kinds` lookup every post runs), the trigger def's snapshot is materialized into
  that workspace, and the catalog is non-empty. The image manifest is built here too but is
  deliberately **not** a condition: an empty manifest is the legitimate text-only deployment.

### 1.4 REST surface → service → verified query

| Endpoint | Service method | `QUERIES.md` |
|---|---|---|
| `GET /health` | `ping` | liveness probe (trivial `RO_QUERY RETURN 1`; 503 when FalkorDB is down). The only route of this table that survives `dev_surface=False`, re-registered bare by `create_app` since it otherwise lives inside `build_router` (§1.3) |
| `POST /channels` | `create_channel` | §3 create a channel |
| `GET /channels[?limit=]` | `list_channels` | §3 list channels in a workspace |
| `POST /channels/{cid}/threads` | `create_thread` | §3 create a thread |
| `GET /channels/{cid}/threads[?limit=]` | `list_threads` | §3 list recent threads in a channel |
| `POST /threads/{tid}/messages` | `post_message` | §4 first message / subsequent message |
| `GET /threads/{tid}/messages[?since=&limit=]` | `read_thread` / `read_messages` | §4 full thread; with `since`/`limit` → §9.1 window as a pure read (`since` defaults to 0 — the browser never touches cursors) |
| `GET /messages/{mid}` | `get_message` | §4 get a single message |
| `GET /search?q=` | `search_messages` | §5 full-text keyword search |

**Workflow-run drive surface** — the non-chat front door for a `kind:'process'`
run. (The `QUERIES.md` §11 def-authoring routes `POST/GET /workflow-defs…`, the `QUERIES.md` §11
**def/snapshot structure reads** `GET /workflow-defs/{key}/versions/{version}` and
`GET /workspaces/{ws}/snapshots/{key}/versions/{version}` plus their `…/diff` sibling —
QUERIES.md §11.2 / §11.5 — and the `QUERIES.md` §12 inspection routes
`GET /workflow-runs/{id}[/step-runs|/trace]` are also mounted; they are read/publish paths and are
described at their own sections.)

**The structure/diff reads — four operator-facing facts.** They answer *"is what I think is
published actually published"*, *"is the workspace running the same thing"*, and *"have `reference`
and `ws:{id}` gone stale independently"* without dropping to raw Cypher. (1) Both structure reads
are **whole-object and unpaginated** — there is no `?limit=`, deliberately: a truncated subgraph is
a *wrong* answer, not a partial one (an operator who gets 50 of 60 steps concludes ten are
missing). They are bounded upstream by the publish-time caps (`MAX_STEPS`, `MAX_TRANSITIONS`,
`MAX_CONFIG_LEN`), matching the unpaginated `QUERIES.md` §12 run reads; service-layer publishers bypass Pydantic,
so those caps are not universal — an accepted, documented residual. (2) The **diff** is bounded
instead by preview truncation (`MAX_DIFF_PREVIEW = 200`): its response is O(differences), never
O(def). (3) **The snapshot is what the executor drives** (`executor._drive_loop` → `get_snapshot`),
so `snapshot` is the operational truth and `def` (`reference`) is the intended truth. (4) The diff
is **version-qualified** — it answers "same version, different content", never "wrong version"; to
detect a stale *version*, compare `GET /workflow-defs` against `GET /workspaces/{ws}/snapshots`
first, or run `./scripts/verify_workflows.sh <wsId>`, which checks both seeded defs at their
expected versions in one command. There is no `latest` alias on the structure route: an operator
investigating a version mismatch must name the version.

> The publish/materialize receipt counts what was **submitted**; the structure read counts what is
> **stored**. **A divergence between the two is a signal, not an endpoint bug** — see
> `docs/plans/workflow-republish-semantics.md`.

`config`/`guard` come back **verbatim** as opaque strings (rule 8) — never parsed, re-serialized or
pretty-printed, so a whitespace-only divergence is still visible. The two structure bodies are the
same shape apart from `source` (so `jq`-diffing them by hand works) — **but that parity is the 200
body only**: the def route 404s through `WorkflowDefNotFoundError` (`{"error": …}`) while the
snapshot route raises a plain `HTTPException` (`{"detail": …}`), each mirroring its sibling
non-structure route's established style. A client must not assume one error shape. These three routes are the only
ones in the surface that declare a `response_model`; the rest are deliberately not retrofitted
(FastAPI's `response_model` *filters* undeclared fields), leaving a mixed convention recorded on the
standing response-schema backlog entry.

| Endpoint | Service method | `QUERIES.md` |
|---|---|---|
| `POST /workflow-runs` | `start_workflow_run` (`trigger_msg_id=None`) | §12.12 start a run from a snapshot with **no** chat trigger `Message` — → **201** |
| `POST /workflow-runs/{runId}/input` | `submit_workflow_input` | §12.7 + §11.5 (validate) then §12.13 merge-into-ctx **and** resume in one CAS — → **200** |

Both routes drive the run **synchronously**, not on `BackgroundTasks`: a process drive is pure
graph work with no LLM, so it is fast and — the deciding property — deterministically testable.
An LLM-bearing process def would want the background path; noted, not built.

**Bounds, and which layer owns which.** Pydantic bounds only what it can see: the
*submitted* dict (≤32 keys, key ≤ 200 chars, serialized ≤ 8000) and `maxSteps` (1…50). The
**merged** ctx bound, the reserved-key rule and the parked-step declaration check live in
`services.py`, because MCP tools and direct service callers never reach a pydantic model.

**Reserved run-ctx keys — `services.RESERVED_CTX_KEYS`, today `threadId`, `error` and
`timerFired` — are rejected on both routes, in the service.**
`threadId` is the resume denorm anchor: a caller-set one would park a process run against a live
chat thread, and the trigger's step 2 would then advance it on the next ordinary human message
there — no input, no guard data. A process run parks with `waitingThreadId = ''`,
and the thread lookup short-circuits on an empty thread id from the other end.

**Error map.**

| Condition | Code |
|---|---|
| unknown run id | **404** `WorkflowRunNotFoundError` |
| run is not parked, or lost the resume CAS (nothing written) | **409** `WorkflowRunNotWaitingError` |
| reserved key / undeclared key / value outside `config.expects` / oversized merged ctx | **400** `WorkflowInputRejectedError` |
| structurally malformed `cmp` guard (dominant source: publish-time validation) | **400** `WorkflowConfigError` |
| workflow engine not wired into this deployment | **503** `WorkflowEngineDisabledError` |
| **a fault *during* the drive** (unimplemented step type, malformed guard reaching evaluation) | **201/200** carrying `{"status":"failed","error":…}` |

That last row is the deliberate one: the executor's fault net has already stamped the run
`failed`, so the run *is* terminal and correct in the graph — a 500 traceback would misreport a
correctly-recorded terminal run as a server bug.

**The failed envelope reports graph truth, whole.** Its `status` **and** its `ctx` both come from
the *same* post-fault `get_run` re-read — never a re-read status beside the caller's submitted
input. So the `ctx` a caller gets back on the fault path is the engine's own state including the
diagnostic note `fail_run` stamped, not the merge that was attempted; on the clean path the two
are the same value by construction (the CAS wrote exactly that merge). Reporting one field from
the graph and the other from what the caller hoped happened would half-apply the rule, and the two
could disagree in exactly the situation where a reader most needs them consistent. If the re-read
status is anything other than `failed`/`done`/`waiting` the service **re-raises**, because
reporting a still-`running` zombie as success would be the worst outcome available.
**Step-budget exhaustion is not a fault**: it returns `"failed"` through the normal path and
reaches the same envelope without raising.

Request bodies are size-bounded at the Pydantic boundary (`schemas.py`: text ≤ 8000 chars,
name/title ≤ 200, mentions ≤ 50) — message text lands in graph RAM *and* the full-text index,
so the transport caps it (RAM rule 6). The `limit`s on the routes above are `Query`-bounded
(1–200; thread window 1–1000). Exactly one route on this router sets a different bound, and it is
worth naming precisely: `GET /threads/{tid}/workflow-runs` — `limit: int = Query(10, ge=1, le=50)`
at `api.py:294`, directly under its own `@router.get` at `api.py:291`. Its neighbour `GET
/threads/{tid}/participants` (`api.py:301`) takes **no `limit` at all**, which is how an earlier
revision of this sentence came to attribute the bound to it: `grep`ping for `le=50` confirms the
bound exists without saying whose it is. **Check the owner in the same step as the number** —
`grep -n -B3 'le=50' falkorchat/api.py` prints the route line above it — and read the route
rather than generalising from this one.

The **two append variants** (`DESIGN.md` §5.3) stay hidden inside `post_message`: the service checks whether
the thread already has a `HEAD`/`TAIL` and dispatches the correct single-`GRAPH.QUERY` write. The
API only ever sees "post a message."

**The `/shop/api` storefront surface.** Eleven routes at `API_PREFIX = "/shop/api"`, present only
under `create_app(storefront=True)` (§1.3). The SPA build is mounted beside them at `/shop`, but
**only when `FALKORCHAT_STOREFRONT_DIR` names an existing directory** — unset (the documented
default) or mistyped and the mount is skipped silently while `/shop/api` serves normally.

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
| `POST /shop/api/messages` | P | `services.post_message`, mentioning the demo agent — `thread_exists` + `resolve_member_kinds` **before** the write, which is where this route's `ServiceError` re-shapes come from — then `Storefront.enqueue_turn`, which books the turn and **submits** it. No further query runs on the request thread: the agent turn (`trigger.maybe_trigger` → `services.start_workflow_run`, §12.1) runs on a turn-executor worker after the `200` has been sent. **No `_safe_embed`** (§4.4 measure 3) | §2, then §4 (the turn's own §12.1 is off the request thread) |
| `GET /shop/api/catalog` | P | `Storefront.list_catalog` → `services.filter_products` + the startup image manifest | §15.2 |
| `POST /shop/api/order/advance` | P | `services.get_current_order`, then `Storefront.advance_own_order` → `order_belongs_to_customer` → `advance_order` | §18.8, §18.9, §16.10 |
| `POST /shop/api/reset` | P | `Storefront.reset_participant` — quiesce, then `repo.reset_participant`, then `services.save_profile` | §18.4, §17.1 |
| `POST /shop/api/presenter/session` | — | nothing — key → token, entirely in-process | *(no query)* |
| `GET /shop/api/presenter/participants` | K | `repo.list_participants`, projected to four keys | §18.3 |
| `POST /shop/api/presenter/reset-all` | K | `repo.list_participants` (pre-drain roster), then `repo.reset_all_participants` | §18.3, §18.5 |

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
route (nothing changed) and `504 <op>_state_unknown` on a `writes` route (§4.8's F8 — the write may
have committed), while a `no graph access` route can produce none of the three. The method key is
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
`cross_cutting_response` returns `None`. The second is the one the branch's own log line names
("*which §5.3 classifies as issuing no query at all*"). Both are unreachable by construction —
the route-table assertion keeps `ROUTE_CLASSES` exactly the registered route set, and a
`no graph access` route issues no query for a typed handler to catch.

**The `ServiceError` handler wraps rather than replaces, and is path-scoped.**
`register_storefront_error_handlers` captures the incumbent handler and registers a wrapper that
re-shapes to the storefront's `{"error": token, "detail": …}` envelope only when the request path
starts with `API_PREFIX`, delegating everywhere else — so the legacy envelope keeps exactly one
definition instead of a second copy that can drift. **No delivered deployment carries both
surfaces**, and that — not the path check — is what keeps the default app byte-identical: this
handler is registered only when a `Storefront` exists, and `create_app` refuses `storefront` with
`dev_surface`. The path check is what makes the handler correct on an app that carries both, which
is a property of the handler rather than of any deployment. On `/shop/api` with **no** mapping it
logs and still delegates: a Python class name reaching a participant is a defect, but inventing a
token for a subclass nobody classified is a worse one. The set of routes on which a `ServiceError` can actually be raised is a **measurement**
(faults armed, all eleven routes driven), not a hand-list.

### 1.5 Layout (as built, M1)

```
falkor-chat/
├── server/
│   ├── falkorchat/{config,db,repository,services,schemas,api,mcp,app}.py
│   ├── tests/{test_repository,test_services,test_services_live,test_mcp,test_api,test_app}.py
│   ├── pyproject.toml          # fastapi, uvicorn, falkordb, mcp, pytest, httpx
│   └── .venv/                  # python3 -m venv (no uv on the box)
└── web/{index.html, app.js}    # fetch() against REST; channels | threads | messages + search
```

`mcp.py` is the second front door — see §2. `app.py` mounts both on one process, and also
serves `web/` as static files at `/` (mounted **last**, since `/` is a catch-all that must sit
behind the REST routes and the `/mcp` mount). Serving the UI from the same process means there is
no CORS seam. The mount is skipped gracefully if the `web/` directory is absent.

### 1.6 TDD build order

Bottom-up, red → green per unit, reusing the isolated-`ws:test`-graph approach `test_queries.sh`
already uses:

0. **Prerequisite (graph-dba):** ✅ done — the `list_channels` query gap landed in
   `QUERIES.md` §3 + `test_queries.sh` (baseline 64/64 → 67/67). The `list_channels` repository
   method can now be built.
1. **`repository`** — integration tests against an isolated `ws:test` graph, one method at a time.
2. **`services`** — append-variant dispatch, id-gen, `updatedAt` bumps (fake repo + a few live checks).
3. **`api`** — FastAPI `TestClient` request/response contract tests. ✅ done — incl.
   `GET /search?q=` (full-text, `search_messages` → `QUERIES.md` §5).
4. **`web`** — ✅ done — minimal `web/{index.html,app.js}` (channels · threads · messages · search),
   served as static files by `app.py`; the mount seam is unit-tested, the UI itself verified
   manually against a running server.

> When this code lands, update `AGENTS.md` (key scripts/commands, working-context rules) and the
> README repo-layout/roadmap in the same change, per the repo's documentation rule.

### 1.7 Testing hazards specific to `server/`

Four gotchas that a green `pytest` run does not surface, distinct from the `test_queries.sh`
teardown hazard already documented at the `AGENTS.md` "Key scripts" table:

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
  at it: a wrong-dimension `vecf32` write is silently accepted (§2/§7.1) and then drops out of ANN
  — the write "succeeds" and retrieval finds nothing, with no error anywhere in the chain.
- **`Services` and `WorkflowExecutor` each default to their own separately-defined `_default_clock`
  function** (`services.py:244` vs. `executor.py:99`), with zero wiring between them — including in
  production `app.py` wiring, which never passes `clock=` to either constructor. A test that
  injects `Services(clock=...)` to control run-parking timestamps silently fails to control
  `StepRun.startedAt`, since that field is stamped by the *executor's* clock, not the services
  clock — the test must construct `WorkflowExecutor` with the identical clock callable too, or the
  injection only half-takes.

**Verifying a claimed test count safely:** `pytest --collect-only -q` reports the suite's test
count with no FalkorDB connection and no writes — the correct way to check a plan's or review's
"N tests" baseline claim without triggering either the `wf_repo` setup-time wipe or
`test_queries.sh`'s teardown wipe above.

- **A wired agent requires two config files.** `FALKORCHAT_ENABLE_AGENT=1` or
  `FALKORCHAT_WORKFLOW_ENABLED=1` builds a `ModelGateway.from_env()`, which reads
  `FALKORCHAT_OPENCODE_CONFIG` (no product default) and `FALKORCHAT_MODEL_CONFIG` (defaults to
  `config/models.json`). `tests/conftest.py`'s `_model_config_env` autouse fixture points both at
  `tests/data/` fixtures for every test — the suite must pass on a machine with **no**
  `~/.config/opencode/opencode.json` (verified: `HOME=<empty dir> pytest -q` is green). A test that
  needs a different value must override both the env var **and** the `falkorchat.config` module
  attribute (`monkeypatch.setattr`) — `config.py` resolves its env vars once at *import* time
  (FR-15, no reload path), so a bare `monkeypatch.setenv` alone never reaches
  `ModelGateway.from_env()` once the module is already imported.

**QA/acceptance-testing gotchas, black-box-observed (distinct from the pytest hazards above):**

- **A `verify_workflows.sh` FAIL for `reference` (def MISSING) does not, by itself, block a live
  `@mention`-triggered workflow run.** `start_workflow_run`'s trigger/execute path never reads
  `db.reference_graph` — only the observability/diff endpoints (`get_workflow_def_structure`,
  `diff_def_snapshot`) do. Three independent `@mention` triggers all started and completed
  `triage@v1` runs against `ws:acme` while `reference` was MISSING throughout. Check which code
  path actually reads `reference` before treating a `verify_workflows.sh FAIL` as an environment
  blocker for a *behavioral* test.
- **A `WorkflowRun` parked `waiting` (`waitsForHuman`) resumes on the *next message posted to its
  thread*, whether or not that message `@mention`s the assistant.** A plain, non-mention message
  into a thread with an open `waiting` run silently resumes it. Only a fresh thread with **no**
  open run correctly exercises "an ordinary message never triggers a workflow" — reusing a thread
  from an earlier test item in the same pass will confound this check.
- **`POST /workflow-runs/{id}/input`'s own response does not carry the `error` reason when that
  submission is what causes the run to fail** — only a follow-up `GET /workflow-runs/{id}` does
  (the reason lands in that run's `ctx`, not in the triggering call's response body). A caller that
  inspects only the `/input` response on a fault sees `status:"failed"` with no explanation.
- **MCP `send_message` never schedules the responder/workflow trigger — only the REST
  `POST /threads/{id}/messages` route does.** `api.py`'s REST handler is the only place
  `background.add_task(_safe_run_workflow/_safe_respond, ...)` is scheduled (via FastAPI's
  `BackgroundTasks`); `mcp.py`'s `send_message` tool has no such scheduling. A message posted via a
  real MCP client produces zero reply and zero `WorkflowRun`, confirmed live. Any black-box check of
  "does `@mention` produce a reply" must specify REST vs. MCP — they are not equivalent front doors.
- **`ModelGateway`/`modelconfig.py` requires an explicit `options.baseURL` for every provider —
  there is no implicit per-npm-package default** (unlike OpenCode's own `@ai-sdk/openai`, which
  has one). An example/fixture `opencode.json` that omits `baseURL` on an `openai`-kind entry
  parses fine but fails to *resolve* (`ModelConfigError: ... no options.baseURL ...`). Any example
  or fixture file authored for this seam should be re-**resolved** once via `ModelGateway.resolve`,
  not just parsed, before being called documented or shipped.
- **`start_server.sh` runs uvicorn with `--reload` by default, watching the whole `falkor-chat/`
  tree — writing or editing ANY file there during a live acceptance pass (including your own
  `docs/test-plans/`/`docs/test-reports/` files) restarts the worker process and silently kills
  every in-flight background daemon thread** (the ingestion pipeline's extraction/embedding work,
  `mcp.py`'s fire-and-forget `threading.Thread(daemon=True)` dispatch — see `background.py`). A
  document write mid-run produces false-negative-looking failures (entities/embeddings that never
  land) with nothing in the response indicating why. Confirmed K-050 M5: editing
  `docs/test-plans/document-ingestion.md` mid-pass logged uvicorn's "2 changes detected" and left
  several in-flight documents without their entities/embeddings; reproduced cleanly, then resolved
  by disabling reload for the pass (see the `UVICORN_ARGS` gotcha immediately below). Do all
  test-plan/test-report writes for a live pass *before* starting the server, or after it's done —
  never interleaved.
- **Overriding `start_server.sh`'s `UVICORN_ARGS` to suppress `--reload` needs a genuinely
  non-empty value — `UVICORN_ARGS=""` does *not* disable it.** The script sets
  `UVICORN_ARGS="${UVICORN_ARGS:---reload}"`; bash's `:-` substitutes the default on **unset or
  empty**, so an explicit empty string is treated the same as never setting the variable at all.
  Confirmed: `UVICORN_ARGS="" ./scripts/start_server.sh` still launched with `--reload` (`ps -ef`
  showed the flag); `UVICORN_ARGS="--timeout-keep-alive 5" ./scripts/start_server.sh` launched
  without it. For a reload-free acceptance pass, pass any real, non-empty uvicorn arg string —
  never an empty override.

### 1.8 The model-resolution seam

**The FR-4 rule, in one sentence:** every LLM/embedding consumer holds a `ModelGateway` and asks
it for a client; a directly-injected client (the legacy `llm=`/`embedder=` constructor kwargs
every consumer still accepts) is sugar `__init__` wraps into a `StaticModelGateway` — dependency
injection for tests, never a configuration route. There is exactly one internal path from "a
kind + an optional requested ref" to a working client, and zero consumers read an endpoint or
model id from `config.py` or any file directly. Enforced, not aspirational: an AST check in
`test_modelconfig.py` fails the suite if any module outside `modelconfig.py`/`tests/` constructs
`llm.OpenAICompatibleLLM`/`embedding.OpenAICompatibleEmbedder` directly.

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
      +--------------+-------+--------+------------------+----------------+
   responder      judge            executor         embedding worker   retrieval tool
   (kind=agent)   (kind=guard)     (kind=step)      (kind=embedding)   (kind=embedding)
```

**Four closed consumer kinds** (`agent`, `step`, `embedding`, `guard`) — adding a fifth means
adding its own override property, or it silently escapes FR-17's future hard cap (routed to
`tico`, `docs/plans/llm-provider-config.md` §9.3). Five binding sites resolve through them:
`AgentResponder.maybe_respond` (`agent` + `embedding`), the executor's `_run_agent_node` (`step`),
`EmbeddingWorker.embed_message` (`embedding`), `GraphragRetrieveTool.run` (`embedding`), and the
llm-kind guard judge (`guard`). Resolution is **per call**, not at construction — the workspace
override is then a function of `ws` with no signature changes.

**Two hand-edited files** feed the gateway (`FALKORCHAT_OPENCODE_CONFIG` — a pristine, unmodified
OpenCode `opencode.json`, providers only, no product default; `FALKORCHAT_MODEL_CONFIG` — falkor-
chat's own overlay, defaults to the shipped `config/models.json`), read once at wiring time
(`ModelGateway.from_env()` — no reload path). `config.assert_no_legacy_model_env()` refuses to
start if any of the four legacy per-provider/per-model env vars (`config.LEGACY_MODEL_ENV_VARS`)
is still set.

**The `/v1` normalization rule (AC-1).** LM Studio's `baseURL` convention omits `/v1`, and a
missing `/v1` is not an HTTP error — it is a `200` carrying an error envelope (a string on one
wrong-prefix path, an object on the right one, on the *same* server), so falkor-chat must
normalize rather than probe. Three ordered steps: **validate** (`scheme in {http,https}` and a
non-empty `netloc`, or reject at load naming the provider/file/value), **strip** every trailing
`/`, then **normalize** (append `/v1` only when the path is now empty; otherwise use it verbatim).
An overlay `providers.<id>.baseURL` override wins outright over both the file and the rule — used
exactly as declared, never auto-suffixed — and one INFO line per provider at startup names the
declared `baseURL`, the resolved API base, and which of {the rule, the override, verbatim}
produced it.

**The `guard` kind's workspace carrier.** Three of the four kinds carry `ws`
to their resolution point via `ctx`; the llm-guard judge does not — `guards.evaluate_guard`'s
`ctx` is the *run* ctx dict, not a `CallContext`, and `_select_transition` has no `CallContext`
either. `executor._drive` stamps `run["ws"] = ctx.ws` outside the SHA-locked `_drive_loop`
boundary (a fresh per-drive dict, never shared, never stored on `self`); `evaluate_guard` forwards
`run=` to the judge only when the judge advertises `accepts_run = True` (the production
`app._LlmGuardJudge`, not the closure it used to be) and forwards `model=` only when the guard
itself declared one (`{"kind":"llm","text":…,"model":…}`) — both zero-churn conditional kwargs, so
every stub judge in the test suite is called exactly as before.

**Roles + ordered fallback chains (FR-7/FR-18).** A ref with no `/` now resolves as a
**role name** — looked up in the overlay's `roles` map and expanded to an ordered, settings-applied
chain of `provider/model` refs, rather than being rejected. A role name must not itself contain
`/`, and a chain element that resolves to another role is rejected at **load** time, not first use
— the role namespace can never accidentally nest. `ModelGateway.llm()`/`.embedder()` build a
`FallbackClient` over the chain's resolved clients: `.chat(...)`/`.complete(...)` try element 0,
then element 1, … on a `ProviderCallError` (a transport-layer `TimeoutError` already converts to
one), and raise naming **every** model tried only if all fail. `FallbackClient` holds
no mutable "last used" state (`__slots__` makes that structural) — the answering model and
whether it came from a fallback travel on the `ChatResult` return value itself: `.model` (the
answering ref) and `.fallback` (`True` iff a later element answered, `None` — never `False` — for
a one-element chain or a direct non-role ref).

**The resolved-model trace (FR-8).** `StepRun` gains three durable properties —
`resolvedModel`, `modelSource` (`workspace`/`step`/`default` — the precedence rung that won) and `modelFallback` (nullable bool, orthogonal to `modelSource` — a workspace override
can itself resolve to a role with its own fallback) — written by the same atomic
`record_step_and_advance` every run already calls, and surfaced on `GET
/workflow-runs/{id}/step-runs`. This is never a `TraceEvent`: those are debug-only by construction
(a non-debug run writes zero), so an audit-relevant field placed there would silently vanish on
precisely the runs nobody thought to flag for debugging — see `docs/plans/llm-provider-config-graph.md`
§1.2 for the full rejection rationale. An agent node that loops and answers on more than one model
records the **last** iteration's three fields together, never a mix of iterations.

**The workspace override + precedence, and the guard-kind hard cap (FR-16/FR-17).** A per-workspace `WorkspaceConfig` singleton (one MERGE-backed node per `ws:{id}`)
carries an optional per-kind override, read once per drive/responder call and stamped onto
`run["modelOverrides"]` — never re-read per resolution. `ModelGateway.resolve()` now implements
the real, first-match-wins precedence: **workspace → the consumer's own requested choice → the
per-kind default.** The workspace rung is a **hard cap**: when present it wins outright, even over
an explicit `requested=`, for **all four consumer kinds — `guard` included**. `guard` is the kind
this section already flagged above as lacking a `CallContext`-borne `ws`; the `run["ws"]` carrier
documented there is exactly what makes the workspace override reachable at the guard-judge
resolution point too — the naive alternative would otherwise have had to reopen the SHA-locked
`_drive_loop`.

**Publish-time rejection (FR-9).** `publish_workflow_def` now runs
`_check_models_resolvable` **immediately before `self._repo.publish_def(...)`, after
`_check_no_structural_conflict`** (the topology-conflict check): every step's `config.model`
and every `{"kind":"llm"}` guard's `model` is resolved through the gateway (no `ws=`/`overrides=` —
a global publish is never gated by per-workspace state), and an unresolvable model or role fails
the publish with a **400** naming the offending step key (or transition endpoints) and the
identifier — nothing is written. A def that fails **both** checks (bad topology **and** an
unresolvable model) returns the topology **409**, not this check's 400, since the ordering runs the
structural check first. A `Services` built without a gateway skips the pass, but logs a WARNING
naming the def and its unchecked identifiers if it declares any model/role, so the skip is never
silently invisible.

**The embedding-dimension guard (FR-19).** Before the first embed write for a
`(workspace, label)` pair, `EmbeddingWorker` compares the resolved embedding model's *declared*
dimension against the workspace's *introspected* vector-index dimension (`Repository
.read_index_dimension`, cached per `(ws, label)` for the process lifetime — never caching a
failure) and raises `EmbeddingDimensionError` **before** calling the embedder on a mismatch: no
vector is written, no inference is wasted. This closes a real silent-failure mode — a wrong-
dimension `vecf32` write is accepted at `SET` with no engine-level error, then simply drops out of
ANN, so retrieval quietly finds nothing with no error anywhere in the chain.

Full design + rationale: `docs/plans/llm-provider-config.md` §3–§4, §7; graph design:
`docs/plans/llm-provider-config-graph.md`; requirements: `docs/requirements/llm-provider-config.md`.

---

## 2. MCP transport — the agent front door

M1 exposes a second, additive transport for AI agents: **MCP over Streamable-HTTP**, mounted on
the *same* FastAPI process and calling the *same* `services.py` as the REST router. Full spec and
rationale: `docs/archive/plans/m1-chat-mcp.md`. Two capabilities were folded into M1 to support it:
participant **@mentions** (`MENTIONS_MEMBER` edge) and per-member **read-cursors** (`ReadCursor`).

### 2.1 Shape

```
browser ── REST/JSON ──┐
                       ├─▶ services.py ─▶ repository.py ─▶ FalkorDB
agents  ── MCP/HTTP ───┘   (all invariants here; both front doors call the SAME methods)
```

`mcp.py` is a thin adapter (peer of `api.py`), no business logic. `app.py`'s `create_app()`
builds one `Services`, `mcp.configure(services)`, then:

```python
mcp_app = mcp.streamable_http_app()
app = FastAPI(lifespan=mcp_app.router.lifespan_context)  # MUST forward, or session mgr never inits
app.include_router(api.build_router(services))
app.mount("/mcp", mcp_app)                                # agents connect at /mcp
```

(Simplified: the shipped `create_app` also takes the opt-in `responder`/`embed_worker`/`trigger`/
`ingestion_pipeline`/`sweep_interval_s` collaborators. **Both lines above are skipped under
`dev_surface=False`** — the storefront deployment mounts neither the REST router nor `/mcp`; see
§1.3.)

> **Lifespan gotcha (python-sdk #1367):** forward the MCP app's lifespan to FastAPI or the
> Streamable-HTTP session manager is never started (requests 500 with "task group not
> initialized"). On this `mcp` build the lifespan is `mcp_app.router.lifespan_context`, and the
> handler's own path is set to `/` (`mcp.settings.streamable_http_path = "/"`) so mounting under
> `/mcp` yields a clean `/mcp` endpoint rather than `/mcp/mcp`. The app's lifespan also runs
> `services.ensure_actor()` so the configured actor node exists before the first write (the §4
> write paths anchor on the author node — QUERIES.md §4 zero-rows note).
>
> **Trailing-slash gotcha:** Starlette's Mount serves the sub-app only under
> `/mcp/`; a bare `POST /mcp` was 405 and MCP clients don't auto-append the slash. `create_app`
> adds an ASGI path-alias middleware rewriting `/mcp` → `/mcp/` so both spellings work.

### 2.2 Tools → service → query

| MCP tool | Service method | Query |
|---|---|---|
| `send_message(body, re, mentions=[], frm=None)` | `post_message` | §4 first/subsequent (+ mentions) |
| `read_messages(re?, since?, limit, advance=True)` | `read_messages` | §9.1 (thread) / §9.2 (room-wide) |
| `create_thread(channel_id, title)` | `create_thread` | §3 create a thread |
| `create_channel(name)` | `create_channel` | §3 create a channel |
| `list_channels(limit=50)` | `list_channels` | §3 list channels in a workspace |
| `list_threads(channel_id, limit=50)` | `list_threads` | §3 list recent threads in a channel |
| `search_messages(query, limit=50)` | `search_messages` | §5 full-text keyword search |

- **Actor identity:** MCP ignores any client-supplied `frm`; every call is attributed to the
  `get_context()` actor (§1.3). M1's actor is the single configured `User` (role `user`).
- **`read_messages` is RW when it advances a cursor.** Explicit `since` → pure read; otherwise the
  per-thread cursor is read and (unless `since` given) advanced to the newest `createdAt` actually
  delivered — never the server clock, which would permanently skip rows a `limit` truncated (an
  empty page advances nothing). Rows are chronological with reader-mentions carried by the
  `isMention` flag (see `QUERIES.md` §9 ordering note). Room-wide reads (no `re`) default `since`
  to epoch 0 and never advance (no room cursor in M1).
- **REST mention parity:** `POST /threads/{tid}/messages` also accepts an optional `mentions[]`.

### 2.3 Client connection contract

Streamable-HTTP; a consuming agent points at the URL (no subprocess):

```json
{ "mcpServers": { "falkor-chat": { "type": "streamable-http", "url": "http://localhost:8000/mcp" } } }
```

Unauthenticated in M1 — bind to localhost / a trusted network only. Run:
`cd server && .venv/bin/uvicorn falkorchat.app:app` (bootstrap `ws:acme` first).
