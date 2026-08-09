---
name: python-web-quirks
description: >-
  Live-verified Python web/async-framework gotchas that don't show up in a quick read of the
  docs: asyncio fire-and-forget task GC-safety, Starlette/FastAPI BackgroundTasks' bounded
  thread-pool concurrency vs. a raw threading.Thread's unbounded one, and FastAPI/pydantic's
  response_model_exclude_unset silently dropping defaulted fields on *nested* models. Use when
  writing, reviewing, or planning asyncio.create_task fire-and-forget scheduling, a
  background-task dispatch mechanism (BackgroundTasks vs. hand-rolled threading), or a FastAPI
  response model that relies on exclude_unset/exclude_none semantics — for coder, tdd-engineer,
  architect, or analyst working in a Python web/async codebase.
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
