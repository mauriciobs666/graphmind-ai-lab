---
name: python-web-quirks
description: >-
  Live-verified Python gotchas beyond a quick docs read — mostly web/async, plus two
  pytest/import-timing traps: asyncio fire-and-forget GC-safety; FastAPI/Starlette
  BackgroundTasks' bounded thread pool vs. unbounded threading.Thread; response_model_exclude_unset
  dropping defaulted nested-model fields; urllib's HTTPError/URLError/TimeoutError taxonomy; an
  OpenAI-compatible server's HTTP-200 error envelope on a missing /v1; a bare json.loads LLM-judge
  parser failing silently on a fenced completion; monkeypatch.setenv as a no-op against an
  import-frozen constant; and a function-local deferred import re-resolving each call vs. a
  def-time-bound default arg. Use for asyncio.create_task scheduling, background-task dispatch, a
  FastAPI response model using exclude_unset, an HTTP client against urllib/OpenAI-compatible
  endpoints, an LLM-judge parser, or a pytest monkeypatch touching an env var or deferred import —
  coder, tdd-engineer, architect, analyst in a Python codebase.
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
back to anyio's **default limiter** — a bounded pool (roughly 40 concurrent worker threads by
default). So FastAPI/Starlette's `BackgroundTasks` **throttles** concurrent sync background work
out of the box.

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

## A pytest autouse fixture's `monkeypatch.setenv(...)` is a no-op for a module-level constant already computed from `os.environ.get(...)` at import time

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
