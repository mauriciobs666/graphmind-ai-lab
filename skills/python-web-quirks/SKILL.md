---
name: python-web-quirks
description: >-
  Live-verified Python gotchas beyond a quick docs read — mostly web/async, plus several
  pytest/import-timing traps: asyncio fire-and-forget GC-safety; FastAPI/Starlette
  BackgroundTasks' bounded thread pool (anyio's default limiter, total_tokens 40, raisable only
  from inside a running event loop, and silently accepting 0 — which deadlocks every sync call the
  app offloads) vs. unbounded threading.Thread; uvicorn defaulting
  proxy_headers=True so ProxyHeadersMiddleware rewrites scope["client"] from X-Forwarded-For,
  falsifying any "request came from localhost" gate; response_model_exclude_unset
  dropping defaulted nested-model fields; FastAPI's four built-in doc routes (and which
  constructor kwargs suppress them) falsifying any "registers only these routes" claim, while an
  included router's own routes are absent from app.routes entirely and its include_router(prefix=)
  never reaches the inner path, and a GET route not matching HEAD at all (starlette Route adds
  HEAD, fastapi APIRoute does not) so HEAD falls through to any path-matching Mount and reads as
  a 404 while every other method reads as an indistinguishable 405; responses={...}
  being keyed by status code alone, so two error bodies at one status collapse — though an x-
  extension inside a status entry does survive verbatim onto route.responses and into app.openapi();
  FastAPI/Starlette's exception-handler registry resolving by the raised class's MRO, keyed on
  starlette.exceptions.HTTPException rather than fastapi.HTTPException, and replaced silently (but
  readably) by a second add_exception_handler on the same type; pydantic
  Field(min_length=1) accepting whitespace-only strings; urllib's HTTPError/URLError/TimeoutError
  taxonomy and redis-py's sibling TimeoutError/ConnectionError, both falsifying a handler map
  claimed total by exception type; a test that gates itself on config.option.keyword rather than on
  what request.session.items collected, so a node id, --lf and --deselect all slip past it; an
  OpenAI-compatible server's HTTP-200 error envelope on a missing /v1; a bare json.loads LLM-judge
  parser failing silently on a fenced completion; an env var set after import (a monkeypatch.setenv
  or a script's own os.environ assignment in main()) being a no-op against an import-frozen
  constant; a bare pytest.raises with no match= passing on an unrelated same-type raise from an
  earlier check in the same validator; a function-local deferred import re-resolving each call vs. a
  def-time-bound default arg; a one-way circular import between two modules that fails in every
  load order unless the deferred import is inside a function body (not a class body); and
  starlette TestClient's teardown cancelling every still-running task regardless of whether the
  app's own lifespan cancels it, and its default raise_server_exceptions=True re-raising an unhandled
  server exception into the calling test so the plain-text 500 a real client receives cannot be
  asserted at all (and the parity the False setting buys comes from ServerErrorMiddleware, not from
  the flag: outside that middleware TestClient synthesizes a headerless, empty-bodied 500); bounding a call whose deadline the code under test computes with a
  daemon thread plus join(timeout) rather than an elapsed-time assert that a hang never reaches —
  stamping the start instant inside the thread body, since one taken before Thread.start()
  absorbs the scheduling gap and passes on a call that did nothing; and an application lock held across ThreadPoolExecutor.submit()
  delaying interpreter exit by the lock's hold time rather than deadlocking it (nothing holds
  _global_shutdown_lock across a join), while submit() enqueues the work item *before* it grows the
  pool, so its "can't start new thread" RuntimeError leaves the job running where the
  same-typed post-shutdown RuntimeError does not — one compensating except-branch cannot serve both. Use for asyncio.create_task scheduling, threading/executor
  lock-ordering questions, background-task dispatch, a
  FastAPI response model using exclude_unset, an assertion over an app's route table or its
  responses={...} declarations, a method-matching or static-mount-shadowing question, a test that
  must bound a possibly-hanging call, an HTTP client against urllib/OpenAI-compatible
  endpoints, an LLM-judge parser, a pytest monkeypatch touching an env var or deferred import, a
  circular-import fix, a TestClient-driven lifespan/background-task test, or an acceptance assertion
  on what a client actually receives from an unhandled server error — coder, tdd-engineer, architect,
  analyst, qa-engineer in a Python codebase.
allowed-tools: Read, WebFetch, WebSearch
---

# Python web/async-framework quirks

Facts confirmed by hands-on testing (source inspection, `inspect.getsource`, or a stress-test
script run to completion) against the versions cited per entry — not just the docs' prose. Treat
them as **verified for that version range**; re-check on a major upgrade of the framework in
question.

> **This is a cache, not the source of truth.** Origin: distilled 2026-08-09 from the `analyst`
> agent's learnings inbox (`claude/analyst/kaizen/inbox.md`) via `agent-maintenance` skill §5 —
> these are general Python/web-framework facts, not specific to any one project in this repo, so
> they live here rather than duplicated into each project's own docs. When a project-specific
> corollary shows up (e.g. "and here's how it bit falkor-chat's `mcp.py`"), keep that in the
> project's own docs and point back to the general fact here, mirroring how
> `claude/graph-dba/falkordb-quirks.md` and project `AGENTS.md` files cross-reference.

## `asyncio.create_task` fire-and-forget: GC risk is real per the docs, but didn't reproduce under stress

**The docs' warning:** `asyncio.create_task(coro)` with the returned `Task` never stored anywhere
is called out by the official asyncio reference docs as a way to lose the task to garbage
collection mid-execution: *"save a reference... a task that isn't referenced elsewhere may get
garbage collected at any time."*

**What was actually observed** (stress test, current CPython at time of testing): a 200-task
fire-and-forget script — `asyncio.create_task(worker(i))` per iteration, return value never
stored, `gc.collect()` forced every 10 creations — completed all 200 tasks with none lost. Root
cause of why it survived: `Task.__init__` schedules `self.__step` via `loop.call_soon`, a
bound-method closure that strongly references the `Task` from creation; once the task awaits, it
also registers as a callback on the awaited future, extending the reference chain for the task's
whole lifetime.

**Consequence for review:** the docs' warned failure mode is real per the contract (nothing
guarantees the closure-chain behavior across implementations or versions) and the anti-pattern
(`asyncio.create_task(...)` with no held reference, e.g. a fire-and-forget dispatch of a launch
or notification) is still worth flagging — but don't auto-escalate it to a confirmed correctness
blocker without reproducing on the actual runtime in question. It's a real latent-risk/idiom
finding, not a demonstrated live bug, unless you've seen it actually drop a task.

## Starlette/FastAPI `BackgroundTasks` (sync callable) is bounded; a raw `threading.Thread` per call is not

Verified by reading source: `starlette.background.BackgroundTask.__call__` (Starlette 1.3.1,
installed alongside FastAPI 0.139.0) routes a **sync** callable through
`starlette.concurrency.run_in_threadpool` → `anyio.to_thread.run_sync` (anyio 4.14.1). That
function accepts an optional `limiter: CapacityLimiter | None = None` and, when omitted, falls
back to anyio's **default limiter** — a bounded pool of exactly **40** worker threads
(`anyio.to_thread.current_default_thread_limiter().total_tokens == 40`, measured 2026-09-08 in
`falkor-chat/server/.venv`, anyio 4.14.1 — **name the venv when you re-check**: this repo carries
five, and `cypher-mcp/.venv` and `mcp-monitor/.venv` hold a newer set, anyio 4.14.2 /
starlette 1.6.0 / uvicorn 0.52.x, with no FastAPI at all).
So FastAPI/Starlette's `BackgroundTasks` **throttles** concurrent sync background work out of the
box.

**Raising that ceiling only works from inside a running event loop.** `total_tokens` is settable
(`limiter.total_tokens = 100` takes effect), but `current_default_thread_limiter()` is
**event-loop-scoped**: called with no loop running it raises `anyio.NoEventLoopError` (*"Not
currently running on any asynchronous event loop"*), so a bump written at module import, in a
plain `if __name__ == "__main__"` preamble, or anywhere else before the server starts, dies rather
than applying. Put it inside the async lifespan (or any async startup hook). A plan that says
"raise the limiter at startup" has not yet said *which* startup, and that ambiguity is the whole
defect.

**And the setter validates the type but not the *useful* range: `0` is accepted, and deadlocks.**
Measured 2026-09-08 (`falkor-chat/server/.venv`, anyio 4.14.1, inside `anyio.run`):
`total_tokens = -1` → `ValueError` *"total_tokens must be >= 0"*; `total_tokens = 0.5` →
`TypeError` *"total_tokens must be an int or math.inf"*; **`total_tokens = 0` raises nothing** — and
the next `to_thread.run_sync(...)` then never returns (a 2 s `move_on_after` around one had to
cancel it). Every sync endpoint FastAPI offloads, and every sync `BackgroundTask`, hangs with no
traceback anywhere. The shape that reaches this is an unvalidated env read assigned straight to the
property: `falkor-chat`'s `config.THREAD_LIMIT` is a bare
`int(os.environ.get("FALKORCHAT_THREAD_LIMIT", "100"))` (`config.py:243`), assigned to the limiter
in the lifespan at `app.py:369` — so `FALKORCHAT_THREAD_LIMIT=0` is a one-character silent outage.
**Review rule:** where a config value lands on a capacity limiter, the guard belongs on the config,
not on the setter — the setter's own refusal covers the wrong half of the domain.

A bare `threading.Thread(target=fn, ...).start()` has no such bound — every call spawns a new OS
thread unconditionally, with no ceiling.

**Consequence for review:** these two are *not* equivalent under load. "Runs the callable
off-thread, same as `BackgroundTasks`" is true of a hand-rolled `threading.Thread` dispatcher, but
"and therefore behaves like `BackgroundTasks`" is not — the off-thread property is easy to verify
by reading; whether it's *bounded* needs one level deeper (does the substitute have an explicit or
implicit cap on concurrent threads, or none?). Check this whenever a change compares a hand-rolled
`threading.Thread` fire-and-forget dispatcher against a framework's `BackgroundTasks` behavior —
e.g. an MCP tool handler that can't use FastAPI's per-request `BackgroundTasks` object and
substitutes a daemon thread per call.

## FastAPI `response_model_exclude_unset=True` drops defaulted fields on **nested** models silently

Verified against pydantic 2.13.4 (installed alongside FastAPI 0.139.0, which serializes responses
through the same `exclude_unset` path):

```python
class Inner(BaseModel):
    a: str
    b: str = "default"

class Outer(BaseModel):
    x: str
    inner: list[Inner]

Outer.model_validate({"x": "1", "inner": [{"a": "q"}]}).model_dump(exclude_unset=True)
# => {"x": "1", "inner": [{"a": "q"}]}   -- `b` is gone, not defaulted in
```

`exclude_unset` is a legitimate way to make "field absent" and "field explicitly null"
distinguishable on the **top-level** model (e.g. omitting an optional key entirely rather than
serializing it as `null`, without `exclude_none` swallowing a field that's legitimately `None`).
But it silently turns **every nested defaulted field** into an optional one too — any field on a
nested model that the caller didn't explicitly set drops out of the response, whether or not that
was the intent.

**Consequence for review:** whenever a response model uses `response_model_exclude_unset=True`
(or the equivalent `model_dump(exclude_unset=True)` call), check every **nested** model in the
response shape, not just the envelope — the guard against silent field loss there is an
exact-key-set contract assertion on the nested object, not just on the top-level one.

## Asserting over a FastAPI app's route table: four built-in doc routes you didn't register, and the included router whose routes aren't there

Verified against FastAPI 0.139.0: a bare `FastAPI()` already carries four routes — `/openapi.json`,
`/docs`, `/docs/oauth2-redirect` and `/redoc`. `/docs/oauth2-redirect` is the one routinely omitted
from a hand-written list of three. They are **defaults, not unconditional**, and the suppression
switches are not one-per-route: `FastAPI(openapi_url=None)` registers *none* of the four (the schema
route gates the other three), `docs_url=None` drops `/docs` **and** `/docs/oauth2-redirect`
together, `redoc_url=None` drops `/redoc`, and `swagger_ui_oauth2_redirect_url=None` drops the
redirect alone. So the count is a property of how the app was constructed, not of the framework — an
absolute claim about it needs the constructor's kwargs in hand.

**Consequence for a route-table assertion:** don't enumerate the exemption, derive it —
`frozenset(r.path for r in FastAPI().routes)` computed from a bare app at import time stays correct
across a framework upgrade that adds a fifth doc route, where a hardcoded list of four silently
red-fails an assertion about *your* app. The trade runs the other way too: a derived exemption means
a future framework default is exempted silently, so an app whose own surface is the thing under test
should assert its registered set by equality, not by count.

**On FastAPI 0.139 `app.routes` does not contain the included router's routes at all.**
`include_router()` stores the router as one opaque `fastapi.routing._IncludedRouter` entry rather
than splicing its `APIRoute`s in, so `[r.path for r in app.routes]` reports the whole REST surface
as **zero paths** — the same answer whether the router is mounted or not, which makes an
absent-route assertion written that way unfalsifiable rather than merely wrong. Flatten via
`route.original_router.routes`, or read `app.openapi()["paths"]` at the cost of missing
`include_in_schema=False` routes. Two traps inside the flatten itself:

- **The prefix lives on the wrapper, not on the path.** `include_router(prefix="/shop")` is stored
  as `route.include_context.prefix`; the inner `APIRoute.path` stays `"/join"`. A walk that
  recurses into `original_router.routes` and appends the raw `.path` therefore reports the same
  list for a router mounted at `/shop/api`, at `/` or at `/admin`. Thread the prefix through the
  walk and accumulate it — nested includes compose (`/shop` + `/api` + `/join`). Contrast
  `APIRouter(prefix=…)` declared at *construction*, which **is** baked into `.path` at include
  time; only the `include_router(prefix=…)` form is affected.
- **Not every entry is a router or a route.** `Mount` appears directly (a `"/"` static catch-all
  normalises to the empty string `""`), and `starlette.routing.Host` has neither
  `original_router` nor `.path` — so the unclassified branch should raise, not skip, or the
  surface silently shrinks.

**Write a positive control for the flatten before trusting it**: in the same test, assert it sees a
route you know exists, then use it to assert one doesn't. Every failure above is a *false green* —
the assertion still passes while proving nothing. Verified against fastapi 0.139.0 /
starlette 1.3.1; `create_app(mount_mcp=True)` yields 4 starlette `Route` (docs) + 1
`_IncludedRouter` (35 inner routes) + 2 `Mount`.

## FastAPI's `APIRoute` does not add `HEAD` to a `GET` route — Starlette's `Route` does, and the difference is only visible when something else matches the path

Verified 2026-09-08 against fastapi 0.139.0 / starlette 1.3.1 (`falkor-chat/server/.venv`).
`starlette.routing.Route('/x', h, methods=['GET'])` reports `methods == {'GET', 'HEAD'}`; the
`APIRoute` FastAPI builds for `@app.get('/x')` reports `{'GET'}`. So a `HEAD` request is only ever a
**partial** match on a FastAPI `GET` endpoint — it never reaches the handler.

**`HEAD` is not special; the *symptom* is.** On a bare app the partial match answers `405`, exactly
like `DELETE` or `PUT`. But a partial match loses to any later route or mount that matches the path
in full, and `starlette.routing.Mount` matches on path alone, ignoring the method — so on an app
whose API routes live under a prefix that also carries a `StaticFiles` mount, *every* non-`GET`
method falls through into the mount. Confirmed by replacing the mount with a permissive ASGI app,
which answered `MOUNT:OPTIONS` / `MOUNT:DELETE` / `MOUNT:PUT` while `GET` still reached the route.
`StaticFiles` then splits the outcomes and hides that: it *serves* `GET`/`HEAD`, so `HEAD` becomes a
**404** for a file that does not exist, while every other method gets `StaticFiles`' own **405** —
byte-identical to the `405` a real partial match produces.

```
FastAPI(); include_router(api, prefix="/shop/api"); mount("/shop", StaticFiles(...))
GET /shop/api/state -> 200 (the route)   HEAD -> 404 (the mount)   OPTIONS/DELETE/PUT -> 405 (the mount)
```

**Consequence for review.** A `405` on such an app is *not* evidence that the request reached the
route and was rejected on method — it may never have touched the route at all. And any handler
keyed on `(request.method, request.url.path)` against a `GET`-only route table cannot `KeyError` on
a `HEAD`, for a stronger reason than "HEAD is unhandled": the request is not routed there in the
first place. To restore `HEAD`, declare it (`methods=["GET", "HEAD"]`), or mount the static app on a
path that does not shadow the API prefix.

## FastAPI's `responses={...}` is keyed by status code only — two error bodies at one status collapse into one declaration

Verified against FastAPI 0.139.0: the parameter is typed `dict[int | str, dict[str, Any]]`, and the
string form is a wildcard *range* (`"4XX"`) — coarser than a status, never finer. There is no
declaration-side key for anything inside the body: not an error token, not the offending `field`.
Two documented 422s on one route that differ only by which field failed therefore reach
`route.responses` as a single `422` entry, and the collapse is correct as built, not a defect.

**Consequence for a contract gate:** a check that reads declarations back off `app.routes` compares
at **status** granularity and structurally cannot see a finer axis. A response table specified per
`(status, field)` has to say which half each side proves — the status half by the declaration, the
finer half only by executing the route. Left unsaid, the table and the gate read as contradicting
each other.

**The escape hatch, if you need the finer axis declared anyway:** the *key* set is closed, but the
per-status *value* is not validated — FastAPI stores the `responses` dict verbatim on
`route.responses` and passes unrecognised keys straight through into the generated OpenAPI
operation. So an `x-`-prefixed extension inside a status entry (`responses={409: {"description":
…, "x-storefront-error": ["ALREADY_JOINED", "THREAD_CLOSED"]}}`) survives **both** reads: off the
route object and out of `app.openapi()`, on a router route reached through `include_router` as
well. That turns a per-route declaration carrying a discriminator finer than the status code into
something a gate can assert against, instead of only executing the route. Verified against FastAPI
0.139.0; the string range form (`"4XX"`) carries extensions the same way. Two caveats: it is an
extension, so nothing validates it — a typo in the key is silently just another passthrough — and
a value that is not JSON-serialisable fails only at `app.openapi()` time, not at import.

## FastAPI/Starlette's exception-handler registry: MRO lookup, a class identity trap, and silent replacement

Verified against FastAPI 0.139.0 / Starlette 1.3.1. Four facts that decide whether an
error-envelope map does what its author thinks — the middle two are the ones that produce a green
test proving nothing.

- **Resolution walks the raised exception's MRO and takes the first *registered* class**
  (`starlette._exception_handler._lookup_exception_handler`: `for cls in type(exc).__mro__: if cls
  in exc_handlers`). Registration order is irrelevant; specificity is decided entirely by the
  exception hierarchy. The useful consequence: **subclassing** an exception claims its own envelope
  without changing what the base class does elsewhere on the same app — a `StorefrontHTTPError
  (HTTPException)` with its own handler returns a flat `{error, detail}` body while a plain
  `HTTPException` raised on the next route still gets FastAPI's `{detail}`.
- **FastAPI's built-in handler is registered on `starlette.exceptions.HTTPException`, which is not
  the same class object as `fastapi.HTTPException`.** On a bare `FastAPI()`,
  `fastapi.HTTPException in app.exception_handlers` is **`False`** and the Starlette one is `True`
  (`fastapi.HTTPException is starlette.exceptions.HTTPException` → `False`; `issubclass(...)` →
  `True`, which is why raising the FastAPI one still resolves). Two consequences: a test that
  *enumerates or classifies* `app.exception_handlers` must key on the Starlette class or it
  silently classifies nothing; and `add_exception_handler(fastapi.HTTPException, …)` **adds a
  fourth key rather than replacing the default** — your handler then wins for `fastapi.HTTPException`
  and its subclasses, while the bare `starlette` `HTTPException` that Starlette itself raises for
  routing 404s/405s keeps the untouched `{detail}` shape. The three keys a bare app starts with are
  `starlette.exceptions.HTTPException`, `fastapi.exceptions.RequestValidationError` and
  `fastapi.exceptions.WebSocketRequestValidationError`.
- **`add_exception_handler` on an already-registered type replaces the previous handler silently**
  — it is a dict assignment, not a chain; no warning, no second entry. The incumbent is still
  readable from `app.exception_handlers[Exc]` **immediately before** you overwrite it, which is the
  whole idiom for scoping an app-wide handler to one path prefix with no import cycle: capture the
  inherited handler, register a wrapper that re-shapes on its own prefix and awaits the captured one
  everywhere else. Assert the incumbent is present at wiring time rather than assuming it — an
  absent one otherwise surfaces as a `TypeError` *inside* an exception handler, on the first error a
  user provokes.
- **A handler map claimed "total by exception type" has to be checked against real MROs, because
  library exception classes that read as a hierarchy are often siblings.** `redis.exceptions.
  TimeoutError` and `redis.exceptions.ConnectionError` are **siblings** under `RedisError` (neither
  is a subclass of the other, and neither is the builtin `ConnectionError`) — so `except
  redis.exceptions.ConnectionError` catches no timeout, and catches nothing that subclasses the
  *builtin* `ConnectionError` either. Verified on redis-py 8.0.1; older redis-py majors did nest
  them, so a reader's memory of the hierarchy is exactly the wrong thing to trust. Each such class
  needs its own `add_exception_handler` registration, or the uncovered one escapes as a bare 500
  through the framework's catch-all.

## uvicorn rewrites `scope["client"]` from `X-Forwarded-For` by default — so "is this request from localhost?" is not an authorization check

Verified 2026-09-08 in `falkor-chat/server/.venv` (uvicorn 0.49.0 / starlette 1.3.1; the newer
`cypher-mcp/.venv` carries uvicorn 0.52.4 / starlette 1.6.0 and was not exercised here).
`uvicorn.config.Config` defaults
**`proxy_headers=True`**, and `forwarded_allow_ips` defaults to `None` in the signature but
resolves in `__init__` to `os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1")`. So a stock uvicorn
runs `ProxyHeadersMiddleware` trusting exactly one peer, `127.0.0.1` — **`::1` is not in it**
(`"::1" in _TrustedHosts("127.0.0.1")` → `False`).

Driving the middleware directly with a stub app (`trusted_hosts="127.0.0.1"`):

| peer | `X-Forwarded-For` | what the app sees as `scope["client"]` |
|---|---|---|
| `("127.0.0.1", 1234)` | *(none)* | `("127.0.0.1", 1234)` |
| `("127.0.0.1", 1234)` | `203.0.113.9` | **`("203.0.113.9", 0)`** |
| `("203.0.113.9", 1234)` | `127.0.0.1` | `("203.0.113.9", 1234)` |
| `("::1", 1234)` | `203.0.113.9` | `("::1", 1234)` |

**The unconditional half — a genuine loopback client is rewritten *away* from loopback the moment
it sends the header** (row 2), with no proxy in the picture at all. A route gated on
`scope["client"][0] == "127.0.0.1"` therefore denies a legitimate local caller that happens to
carry an `X-Forwarded-For`, and the check is a peer-address heuristic rather than an authorization
boundary. Note the rewritten port is `0`, so a gate keyed on the port breaks too.

**What does *not* follow, and is worth stating because it is the intuitive fear:** a remote client
cannot simply claim `X-Forwarded-For: 127.0.0.1` and be believed (row 3) — the peer must already
be trusted. `_TrustedHosts.get_trusted_client_address` walks the header **right-to-left and
returns the first untrusted host**, so a proxy that *appends* the real peer keeps an
attacker-supplied left-hand entry from winning. The spoof direction needs a real
misconfiguration — `forwarded_allow_ips="*"`, or a proxy that passes client-supplied
`X-Forwarded-For` through instead of appending to it — at which point every entry is trusted and
the function falls back to `x_forwarded_for_hosts[0]`, the attacker's value. Design against the
first row of this paragraph, but don't claim the second without checking the deployment.

## pydantic `Field(min_length=1)` does not reject whitespace-only strings

`" "` has `len` 1, so `Field(min_length=1)` accepts it — a common false-safety assumption when a
docstring or comment claims "the service validates non-empty input upstream" but the actual check
is only a REST-boundary `min_length`. Two consequences worth checking together: (1) any MCP or
other non-HTTP caller that bypasses the pydantic schema layer entirely is completely unguarded,
not just under-guarded; (2) even the REST path itself lets whitespace-only text through, since
`min_length` counts characters, not meaningful content. Verified against pydantic 2.x: `Field(
min_length=1)` on a `str` field accepts `"   \n\t  "` without validation error. If empty-vs-
whitespace-only matters, add an explicit `str.strip()` check at the service boundary (not just the
schema boundary) — the two guards are not redundant, they cover different callers.

## `urllib` failure taxonomy: `HTTPError ⊂ URLError`, but a read timeout is a bare `TimeoutError`, not a `URLError`

Verified against CPython 3.12: `issubclass(urllib.error.HTTPError, urllib.error.URLError)` →
`True`; `socket.timeout is TimeoutError` → `True`; `issubclass(TimeoutError,
urllib.error.URLError)` → **`False`**. A real `urllib.request.urlopen(req, timeout=0.5)` against a
slow server raised `TimeoutError` with MRO `(TimeoutError, OSError, Exception, BaseException)` —
no `URLError` anywhere in it. A schemeless URL (`"host:1234/x"`) raises `ValueError: unknown url
type`, in neither branch.

**Consequence for review:** (a) an `except URLError` clause placed before `except HTTPError` makes
the HTTP-status branch dead code and discards the response body — `HTTPError` must be caught
first; (b) a client that catches only `URLError`/`HTTPError` lets every read timeout escape
unclassified — a stdlib-only HTTP client's failure-mode enumeration must list `TimeoutError` and
`ValueError` (bad URL) as their own cases, not assume they land under `URLError`.

## An OpenAI-compatible local server can answer a missing `/v1` prefix with HTTP 200 + an error envelope, not a 404 or 400

Verified against LM Studio (`localhost:1234`): `POST /chat/completions` (no `/v1`) → **`200`**
`{"error":"Unexpected endpoint or method. (POST /chat/completions)"}`; `POST /v1/chat/completions`
→ `400` with a proper OpenAI error object. Same shape on `/embeddings` vs. `/v1/embeddings`. `GET
/models` and `GET /v1/models` **both** return `200` with the real model list, so probing `/models`
cannot discriminate the prefix. The `error` value's JSON shape also **differs by path** — a
**string** on the wrong-prefix response, an **object** on the correct one — so a classifier
written as `body["error"]["message"]` raises `TypeError` in exactly the case it exists to diagnose.

**Consequence for review:** any OpenAI-shaped client that omits `/v1` fails as a bare `KeyError:
'choices'`/`'data'` with no mention of the URL, not as a request error — a base-URL normalization
step (validate → strip trailing `/` → append `/v1` only when the resulting path is empty) belongs
ahead of the request, not a post-hoc status-code check. Related trap in the same code path:
`urllib.parse.urlparse("192.168.0.69:1234").path == "192.168.0.69:1234"` — **non-empty** — so an
"if the URL path is empty, append `/v1`" heuristic silently accepts a schemeless base URL instead
of rejecting it.

## An LLM-judge JSON parser that's a bare `json.loads(text)` is fence-fragile, and the failure mode is silent, not an exception you'd notice

A free-text LLM completion parsed with a bare `json.loads(text)` — no markdown-fence or
surrounding-prose stripping — breaks completely the moment a model wraps its answer in a ` ```json
...``` ` fence (common on the Mistral/Gemma model families): every response becomes
`json.JSONDecodeError`, caught and mapped to a generic "unparseable judge output," which a
deliberately bias-to-suspend judge design (see the asymmetric-judge note in this lab's method
notes) then resolves to `decision=False` — so the judge looks like it's working (no crash, a
plausible-looking verdict) while actually never parsing a single real answer. One capability probe
saw a model's advance-recall go from an apparent 0/10 to a real, still-mediocre 0.364 purely by
adding fence-tolerant parsing — the fence artifact had been masking (and looked identical to) a
genuine over-suspend weakness.

**Consequence for review:** any LLM-output JSON parser should strip a leading/trailing code fence
(or use the provider's structured-output / `response_format` mode) before `json.loads`, and a
judge/classifier's "unparseable" bucket is worth instrumenting separately from its substantive
verdicts — a spike in "unparseable" that silently resolves to one default verdict is a parser bug
wearing the shape of a model-quality problem.

## Setting an env var *after* import is a no-op for a module-level constant computed from `os.environ.get(...)` at import time — in a pytest fixture and in a plain script alike

A config module that resolves constants once at import (`WS_ID = os.environ.get("WS_ID", ...)` at
module scope, "read once, no reload path" by design) freezes those values before any per-test
`autouse` fixture runs, if the module was already imported by another test file at collection
time. `monkeypatch.setenv("WS_ID", ...)` inside the fixture never reaches code reading the frozen
module attribute — it was computed long before the fixture ran. The fix used elsewhere in the same
codebase (`monkeypatch.setattr(config_mod, "WS_ID", value)`) targets the **module attribute**
directly instead of the environment.

**Consequence for review:** an `autouse` fixture that repoints a config-driven test double via
`monkeypatch.setenv` alone is silently a no-op against any constant a module froze at import time.
The safe fixture sets **both** — the env var (for anything reading `os.environ` fresh, or a
subprocess) **and** `monkeypatch.setattr(module, "ATTR", value)` (for the frozen-at-import
constant).

**The same freeze bites a plain script with no pytest anywhere in sight** (verified 2026-08-30, a
`falkor-chat` eval harness): a top-of-file `from pkg import a, b` runs the config module's
import-time `os.environ.get(...)` immediately, so `os.environ["X"] = ...` inside `main()` — which
looks early enough, and is early relative to the call that consumes it — sets the variable *after*
the constant it feeds was already computed, and the consumer fails as if the variable were never
set. Here the fix is placement, not `setattr`: set the environment **above the package imports**,
at the very top of the file (`os.environ.setdefault(...)` before `from pkg import ...`). A
constant frozen at import can only be beaten before that import or at the module attribute — never
in between.

## A `pytest.raises(SomeError)` with no `match=` can pass on an unrelated raise from an earlier check in the same function

A validator that checks several fields in a fixed order and raises **one** exception type for all
of them will short-circuit on the first violation. If a test's fixture value plays two roles at
once — the field under test *and* some earlier-checked field, or a value that also trips a
pre-existing structural check — the bare `pytest.raises(SomeError)` is satisfied by whichever
check fires first, and the assertion never touches the behavior it claims to pin. The failure mode
is silent and one-directional: the test is green before the guard exists, green after, and green
when the guard is deleted.

**Consequence:** for any exception assertion where more than one code path can raise that type,
pin `match=` to the *specific* raiser's own wording (`r"^step key would be \d+ characters, over
the"`, not a bare `"key"` that a dangling-reference error also contains), and keep the fixture
value out of every role but the one under test. The check that this was done correctly is a
mutation, not a review: disable the guard's call sites and confirm **every** parametrized case
goes red — a case that stays green was passing for the wrong reason all along. (Observed
graphmind-ai-lab 2026-08-26, `falkor-chat` K-049; the surviving test comments in
`server/tests/test_services.py` around `OVERSIZED_ISOLATED_STEP` record the concrete instance.)

## A test that is only valid when its whole module ran must gate on what pytest **collected**, not on how the run was **selected**

A test whose assertion depends on side effects its module-mates recorded (a "every declared row was
provoked by some test in this file" completeness check, an accumulator asserted at the end) has to
skip itself on a partial run. Gating on the *selector* — `request.config.option.keyword`, i.e. what
`-k` was given — covers exactly one of the four ways a run can be partial. Verified on pytest 9.1.1:

| how the subset was chosen | `config.option.keyword` | `request.session.items` |
|---|---|---|
| `-k probe` | `'probe'` | just the matches |
| `pytest file.py::test_probe` (a node id) | `''` | just that test |
| `--lf` after a failure | `''` | just the failures |
| `--deselect file.py::test_a` | `''` | everything else |

So the `-k` guard is a no-op for the three cases a developer hits *most* — including `--lf`, which
is what you reach for right after this test fails, at which point it runs against a nearly empty
accumulator and reports every row as unproducible. Gate on collection instead:

```python
module = sys.modules[__name__]
defined   = {n for n in dir(module) if n.startswith("test_")}
collected = {getattr(i, "originalname", None) or i.name.split("[")[0]
             for i in request.session.items if getattr(i, "module", None) is module}
if defined - collected:
    pytest.skip("this module was only partly collected")
```

`originalname` is what collapses a parametrized `test_b[1]`/`test_b[2]` back to `test_b`; without
it every parametrized test in the module reads as uncollected and the guard skips always. **Subtract
marker-deselected tests from `defined` yourself** if the project's `addopts` carries a marker filter
(`-m "not live"`): the first `@pytest.mark.live` test added to the module would otherwise deselect
itself on every default run and switch the check off permanently — visible in the skip reason, but
off.

## A function-LOCAL `from .module import name` re-resolves fresh on every call — a function-DEFAULT bound to the same name does not

A deferred import placed as the first statement *inside* a function body (not module-level —
often done deliberately to break a circular import) performs a fresh `getattr(module, name)` every
call, so `monkeypatch.setattr("pkg.module.name", fake)` transparently intercepts every future call
through it — no changes needed to the caller's public API to inject a test double. Contrast: a
function whose **default argument** is bound to the same name (`def f(..., opener=urllib.request.
urlopen): ...`) binds it once at *definition* time (Python evaluates defaults once), so
monkeypatching the original name afterward does **not** reach it — that seam needs an explicit
`opener=` kwarg at the call site instead.

**Consequence for review:** when a module defers an import inside a function to break a cycle, that
import is a live monkeypatch seam by construction — reach for `monkeypatch.setattr` on the
*deferred import's source module* rather than restructuring the code to accept a new parameter.
When the same value instead reaches the function as a **default argument**, the seam is closed at
definition time and needs an explicit parameter to inject a test double.

## A one-way circular import fails in *every* load order, not just the "wrong" one — and a class-body import doesn't fix it, only a function-body one does

Two modules where one already imports a name from the other (`b.py: from .a import X`) cannot
gain the reverse direction (`a.py: from .b import Y`) even when `X`/`Y` are unrelated names and
neither module is otherwise self-referential — verified with `python3 -c "import pkg.a"` and
`import pkg.b"` on CPython 3.12: **both** orders raise `ImportError: cannot import name '...' from
partially initialized module`, not just the one you'd guess. Whichever module starts loading first
pauses mid-execution at its own top-level `from .other import NAME` before the other module has
reached the point where `NAME` is defined. Moving the reverse import into a **class body**
(`class Foo:\n    from .other import Y`) doesn't help — a class body executes immediately at
module-load time, so it fails identically (verified, same error). Only an import placed **inside a
function body**, executed on first *call* rather than at module-load time, avoids the cycle
(verified — succeeds once both modules have finished loading).

**Consequence for review:** the fix for a genuine one-way circular-import need is not "move the
import somewhere that looks deferred" — a class body doesn't count. Either (a) keep the shared name
in whichever module is already the "source" side of the existing one-way import (mirror the
existing direction, never invent the reverse), or (b) push the import inside a function/method body
if it's truly only needed at call time. Surfaced in falkor-chat: `services.py` already imported
constants from `schemas.py`; a plan that added new constants to `services.py` and had `schemas.py`
import them back failed in this exact shape (K-028 U3b, `coder`).

## `starlette.testclient.TestClient`'s teardown cancels every still-running task on its event loop — masking whether the app's own lifespan cancellation ever ran

Verified against starlette 1.3.1 / fastapi 0.139.0 / anyio 4.14.1: a lifespan that starts a
background `asyncio.Task` and **deliberately never cancels it on shutdown** still shows
`task.done() == True, task.cancelled() == True` immediately after the `with TestClient(app):`
block exits — reproduced with a minimal FastAPI app whose lifespan has no cancellation code at all.
`TestClient`'s anyio blocking portal tears down its event loop by cancelling every task still
running on it, independent of the app-under-test's own shutdown logic.

**Consequence for review:** a background-task-cancelled-at-shutdown test that only asserts
`task.cancelled()`/`task.done()` **after** the `with`-block exits passes even if the app's own
cancellation code is deleted entirely — the portal's teardown masks the bug. To actually pin
app-level shutdown-cancellation code, assert on something only the app code itself would produce
(e.g. that it stored the task reference on `app.state` at all — an `AttributeError` if that line
is dropped), not the task's final `cancelled()` state. Surfaced writing the lifespan smoke test for
falkor-chat's periodic sweep task (K-028 U3b, `coder`).

## `TestClient` re-raises an unhandled server exception into the calling test by default — so a test written on the default can never observe the 500 a real client receives

Second face of the class the section above belongs to: `TestClient` is not a real HTTP client, and
each convenience it adds is separately opt-outable. Verified against starlette 1.6.0 / httpx 0.28.1 /
uvicorn 0.52.1 / CPython 3.12.3; `fastapi.testclient.TestClient` is this same class re-exported
verbatim (`fastapi/testclient.py` is the single line `from starlette.testclient import TestClient`,
read out of the fastapi 0.141.1 wheel), so everything here applies identically under either import.

One Starlette app, one route raising an unhandled `RuntimeError`, driven three ways — a `/ok` route
returning 200 as a control in each arm:

| driver | what the caller gets |
|---|---|
| `TestClient(app)` — default `raise_server_exceptions=True` | `RuntimeError` raised **into the calling test**; no response object at all |
| `TestClient(app, raise_server_exceptions=False)` | `500`, `text/plain; charset=utf-8`, body `Internal Server Error` |
| real uvicorn server + real `httpx.Client` | `500`, `text/plain; charset=utf-8`, body `Internal Server Error` |

Mechanism (`starlette/testclient.py:348-363`): the portal call sits inside an
`except BaseException:` that re-raises when the flag is set. The flag does **not** construct the
500 — `ServerErrorMiddleware` has already sent that response before re-raising, so `response_started`
is true and the default simply *discards* the captured response in favour of the exception.

**The client-parity is the middleware's, not the flag's.** Against a bare ASGI callable with no
`ServerErrorMiddleware`, nothing ever sent a response, so `raise_server_exceptions=False` falls
through to the transport's synthesized fallback — `status_code` 500 with **no headers and an empty
body** — while a real client over the wire still gets uvicorn's own `text/plain; charset=utf-8` /
`Internal Server Error`. Verified with the same probe against a bare ASGI app: `content-type` `None`
and body `''` under `TestClient`, against `text/plain; charset=utf-8` / `Internal Server Error` from
the live server.

**Consequence for testing:** when the question is *what does a client receive?* rather than *did it
raise?*, construct the client with `raise_server_exceptions=False`. On the default, an acceptance
assertion about the status or body of an unhandled server failure is not merely failing — it is
unwritable, because the call never returns a response to assert on. Assert on the **body** only for a
real `Starlette`/`FastAPI` app, whose `ServerErrorMiddleware` is what makes the test's 500 and the
wire's 500 the same bytes; for anything outside that middleware's reach, assert the status alone.
Surfaced deciding which HTTP status a plan contract was actually violated by, in falkor-chat's S9
storefront concurrency acceptance pass (`qa-engineer`).

## Version rot ahead of the two sections above: starlette deprecates `httpx` under `testclient` in favour of `httpx2`

`from starlette.testclient import TestClient` emits a `StarletteDeprecationWarning` **at the import
line** whenever `httpx` is installed and `httpx2` is not: *"Using `httpx` with
`starlette.testclient` is deprecated; install `httpx2` instead."* Measured 2026-09-09, CPython
3.12.3, across this repo's venvs — starlette **1.3.1** (`falkor-chat/server/.venv`) and **1.6.0**
(`cypher-mcp/.venv`, `mcp-monitor/.venv`), both on httpx 0.28.1, with no `httpx2` installed
anywhere: identical warning, every run, both versions.

The warning itself is trivial to silence. What matters is that the two sections above are pinned to
exactly those versions, so neither is one release *ahead* of the migration — both already describe
the deprecated transport. And the behaviour those sections turn on is transport-side: the
headerless, empty-bodied 500 under `raise_server_exceptions=False` is `TestClient`'s own synthesis
rather than the app's, and response synthesis plus constructor surface are precisely what an
httpx→httpx2 swap can move. So re-derive both sections against the starlette actually installed
before citing them; do not read their version pins as "current".

## Holding an application lock across `ThreadPoolExecutor.submit()` does **not** deadlock at interpreter exit — it delays exit for exactly as long as the lock is held

Verified against CPython 3.12.3. `_python_exit` (`concurrent/futures/thread.py:23-31`) takes
`_global_shutdown_lock` **only** to set `_shutdown = True`; both the `q.put(None)` loop and the
`t.join()` loop sit *outside* that `with` block. `ThreadPoolExecutor.shutdown()` has the same shape
— `self._shutdown_lock` is released before its own join loop (`thread.py:220-238`). `submit()` does
take `self._shutdown_lock` and `_global_shutdown_lock` together (`thread.py:165`), but nothing ever
holds either lock *across* a join, so no cycle can form between an application lock and the
executor's shutdown machinery.

Staged the exact arrangement that a deadlock claim requires — an application lock acquired, then
`executor.submit(fn)` called while holding it, the submitted worker blocking on that same lock, and
the main thread falling off the end without releasing it. `submit()` returned in **0.2 ms** (it
never blocked), and the process exited cleanly (rc 0) after **exactly** the lock hold time: 0.52 s
wall for a 0.5 s hold, 3.03 s for a 3.0 s hold. With the lock never released the process does not
exit at all (killed at 30 s) — the same mechanism at its limit, not a deadlock.

**Consequence for review:** "holding a lock across `submit()` risks a shutdown deadlock" is false
and must not gate a design. The real cost is a **non-daemon worker that cannot finish**:
`_python_exit` joins every worker *unlocked*, so interpreter exit is bounded below by however long
the application lock stays held — a latency cost that becomes unbounded only if the lock is never
released. Surfaced as a retraction: a `salesperson-ui` S9 plan amendment asserted the deadlock,
`analyst` Pass 18 disproved it from source and by staging it (`architect`, 2026-09-08).

**A second ordering fact inside the same `submit()`, with a live consequence for error handling.**
`submit()` puts the work item on the queue **before** it tries to grow the pool —
`self._work_queue.put(w)` at `thread.py:178`, `self._adjust_thread_count()` at `:179`. So when the
pool cannot start a worker, `submit()` raises `RuntimeError("can't start new thread")` *with the
job already enqueued*, and any worker already running picks it up. Reproduced 2026-09-08 on CPython
3.12.3 by patching `threading.Thread.start` to raise while one worker was busy: `submit` raised, and
the "refused" job still ran. The trap is that the **other** `RuntimeError` `submit()` raises —
`cannot schedule new futures after shutdown`, from the checks at `:169-173`, which run *before* the
`put` — is a clean refusal and the same exception type. **So a compensating action in the caller's
`except RuntimeError:` — releasing a reservation, decrementing an in-flight counter, marking a turn
free — is right for one shape and corrupts state for the other**, orphaning a job that is about to
run. Discriminate on the message, or make the compensation idempotent against a job that runs anyway.

## Bounding a call whose deadline the code under test computes: a daemon thread, not an elapsed-time assert — and stamp the start instant *inside* the thread body

Where a venv has no `pytest-timeout` (confirmed absent from `falkor-chat/server/.venv`, 2026-09-08),
the reflex fix for "prove this call returns within its own deadline" is to time it and assert
afterwards. **That assertion is unreachable on the failure it exists to catch**: if the call hangs,
control never returns to the assert, and the test dies on the runner's own kill instead — an
observed exit 143 at 30 s, with no test name attributed. Run the call on a `daemon=True` thread,
`join(timeout=…)`, and assert `not worker.is_alive()`. Measured 2026-09-08: `join(timeout=1.0)`
around a 30 s sleep returns after **1.00 s** with `is_alive()` `True`, so the bound is the
assertion, and a failure names the test. The daemon flag is what stops the abandoned worker from
holding interpreter exit open.

**The trap inside the fix.** If the test also asserts an *ordering* (`started_at < finished_at`), a
start instant stamped on the calling thread — before `Thread.start()` — silently absorbs every
scheduling gap and any setup between the stamp and the call, so the assertion is satisfied by the
gap rather than by the work. With a realistic ms-resolution timestamp, a stubbed no-op call and a
300 ms gap injected before the start:

```
stamp OUTSIDE, before Thread.start()   started=…903964 finished=…904264  started<finished -> PASS
stamp INSIDE the thread body           started=…904565 finished=…904565  started<finished -> FAIL
```

The outside-stamped form passes on a call that did nothing. Re-stamp as the **first line of the
thread body** and the ordering assertion measures the call again. (A `perf_counter`-resolution clock
hides this — the delta comes back at ~1 µs and still passes; it is the coarse timestamp the code
under test actually records that makes the difference visible.)
