# Review — MCP `send_message` background scheduling implementation (K-041)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-041

## Scope & verdict

Post-implementation code review of the uncommitted diff fixing D-1
(`kiro/docs/test-reports/kiro-demo-agent-report.md`) — messages posted via the MCP `send_message`
tool never scheduled the `assistant` responder or the M3 workflow trigger, while the REST route
did. Baseline: the working tree as of this session (`git -C falkor-chat status`/`diff`), against
`falkor-chat/docs/BACKLOG.md`'s K-041 entry and `falkor-chat/docs/HISTORY.md`'s 2026-08-01 K-041
entry, both written by the implementer.

Reviewed: the new `server/falkorchat/background.py`; the diffs to `server/falkorchat/{api,mcp,app,
services,executor}.py`; the new/changed tests in `server/tests/{test_mcp,test_process_input,
test_workflow_live}.py`; `docs/{BACKLOG,HISTORY}.md`'s K-041 entries; `docs/DESIGN.md` §15 (spot
check of the implementer's "no correction needed" claim). Ran `pytest -q` and
`./scripts/test_queries.sh` myself against a live FalkorDB (`falkordb-dev`, already running) rather
than trusting the reported counts. Did not review `kiro/` beyond confirming it isn't in the diff.

**Verdict: approve with suggestions.** No blockers. The core correctness property — exactly one of
{trigger, responder} fires, embed independent of both, replicated identically from `api.py`'s REST
route — is implemented correctly and is well covered by new tests. Two minor findings on the
threading mechanism's production-readiness posture, one doc-accuracy nit; none block landing this
fix.

## Findings

### Minor — `_default_schedule` spins an unbounded `threading.Thread` per call; the docstring's "mirrors Starlette" claim overstates it

`server/falkorchat/mcp.py:42-54`:

```python
def _default_schedule(fn: Callable[..., None], *args: Any) -> None:
    """...mirroring what Starlette does for a sync BackgroundTasks callable (runs it in
    a threadpool)."""
    threading.Thread(target=fn, args=args, daemon=True).start()
```

I checked what Starlette's `BackgroundTask.__call__` actually does for a sync callable
(`starlette/background.py`): it calls `run_in_threadpool(self.func, ...)`, which is
`anyio.to_thread.run_sync(...)` — and `anyio`'s `run_sync` uses a **bounded capacity limiter**
(default ~40 concurrent worker threads; verified against the installed `anyio` source in this
project's venv) when no explicit limiter is passed. A raw `threading.Thread()` per call has no such
bound: a burst of concurrent MCP `send_message` calls (e.g. several agents posting at once) would
spawn one new OS thread per call with no ceiling, unlike the REST path's throttled threadpool. This
is a real asymmetry between the two transports the fix was meant to make equivalent, and the
docstring's "mirrors Starlette" phrasing invites a reader to assume the bound exists too.

Given M1's stated posture (unauthenticated, single-tenant, lab-scale — `docs/DESIGN.md` §15.3),
unbounded-but-rare thread spawns are unlikely to matter in practice today. Suggest one of: (a) bound
`_default_schedule` with a small fixed-size `concurrent.futures.ThreadPoolExecutor` (module-level,
alongside the `_schedule` seam) so both transports share the same throttling posture, or (b) if the
unbounded version is intentionally accepted for M1's scale, soften the docstring's "mirroring what
Starlette does" claim to something like "mirrors Starlette's off-thread execution, without its
bounded-pool throttling" so a future reader doesn't assume parity that isn't there. Either is a
small change; not worth blocking the fix over.

### Minor — no shutdown/drain awareness for in-flight MCP background threads (asked for explicitly in the brief)

`app.py`'s `_lifespan` (`server/falkorchat/app.py:203-218`) has no hook that would wait for, or even
know about, threads `_default_schedule` has spawned. Daemon threads (`daemon=True`) are killed
outright when the interpreter exits, with no grace period — a `send_message` call whose background
thread hasn't finished `_safe_respond`/`_safe_run_workflow` when the process shuts down loses that
work silently (no exception, no log line, nothing to catch, since the thread is simply terminated).
Contrast with the REST path: Starlette's `Response.__call__` awaits `self.background` as part of
the same ASGI application coroutine that handles the request, so the request is not "done" (and a
graceful-shutdown drain that waits for in-flight requests would keep waiting) until the background
task completes.

I want to be precise about severity here: this is not a regression this fix introduces relative to
REST's *reliability* guarantee — both paths already swallow-and-log all exceptions inside
`_safe_*`, so neither is a strict at-least-once guarantee even normally. What's new is a
*shutdown-timing* gap that REST doesn't have: MCP's version can silently drop scheduled work at
process exit with literally no log entry, where REST's version would (at worst) log the failure
before the process actually stops. For M1's demo/lab scale this is likely acceptable as-is, but it
is exactly the class of gap the brief asked me to check for, so I'm flagging it rather than silently
accepting it. Suggest: either explicitly note this asymmetry in `background.py`'s or `mcp.py`'s
module docstring (so it's a documented, chosen trade-off rather than an implicit one), or — if it's
worth closing — track running threads in a small registry and join them with a timeout in
`_lifespan`'s shutdown branch.

### Nit — `docs/BACKLOG.md`'s K-041 entry misnames the swappable seam

`docs/BACKLOG.md`, K-041 entry, point (4):

> scheduling uses a daemon `threading.Thread` fire-and-forget by default — swappable via a
> module-level `_schedule` seam (`mcp._default_schedule`) that tests override for deterministic,
> non-racy assertions.

The parenthetical names the wrong symbol: `_default_schedule` is the *default value* assigned to
the seam, not the seam itself. The thing tests actually monkeypatch is `mcp_mod._schedule`
(confirmed: `test_mcp.py`'s `sync_schedule` fixture does `mcp_mod._schedule = lambda fn, *args:
fn(*args)`, never touching `_default_schedule`). `docs/HISTORY.md`'s K-041 entry gets this right
("swappable module-level seam (`mcp._schedule`)"), so it's just the BACKLOG.md wording that's
imprecise — a reader grepping for `mcp._default_schedule` expecting to find "the seam" would be
looking at the wrong name. Trivial fix: drop the parenthetical, or correct it to `mcp._schedule`.

## What's solid

- **The one-handler guarantee is correctly replicated.** `mcp.py`'s `_schedule_background`
  (`server/falkorchat/mcp.py:90-103`) schedules `_safe_embed` whenever `_embed_worker` is
  configured (independent of trigger/responder), then schedules exactly one of
  `_safe_run_workflow` (if `_trigger` is configured) or `_safe_respond` (elif `_responder` is
  configured) — line-for-line the same ordering as `api.py`'s `post_message` route
  (`server/falkorchat/api.py:96-107`). I traced both call sites side by side; the argument shapes
  match too (`ctx.ws`/`posted["msgId"]`/`posted["text"]` for embed; the full `posted` dict plus
  `ctx` for trigger/responder).
- **`background.py`'s relocation is behavior-preserving.** Diffed the removed `api.py` functions
  against the new `background.py` definitions directly: the three function bodies
  (`_safe_embed`/`_safe_respond`/`_safe_run_workflow`) are unchanged except for docstring wording
  ("Runs on `BackgroundTasks`" → "Runs off-band", to stay accurate for the now-shared module) — no
  logic, argument, or exception-handling change. `api.py` now imports from `background.py` and no
  longer defines a second copy anywhere (`grep` confirms no duplicate definitions remain outside
  the archived docs).
- **Test coverage is thorough and mirrors the existing REST-side pattern well**, including a case
  the REST suite doesn't have isolated: `test_send_message_embeds_independently_of_trigger_or_
  responder` wires `embed_worker` + `trigger` with **no** responder — the "plausible production
  shape" the brief called out — and confirms both fire. The `sync_schedule` fixture correctly
  resets `mcp_mod._schedule` to its original value after each test (no cross-test leakage), and the
  one genuinely-threaded test (`test_send_message_default_scheduling_runs_off_a_background_thread`)
  uses a `threading.Event` + `timeout=2` wait rather than a sleep-and-hope race, which is the right
  way to assert this deterministically.
- **`app.py`'s wiring is correct and minimal** — the exact `responder`/`embed_worker`/`trigger`
  objects already passed to `api.build_router(...)` are now also passed to `mcp_mod.configure(...)`
  (`server/falkorchat/app.py:187-193`), inside the same `if mount_mcp:` branch, so no new code path
  bypasses it.
- **Doc-sync claim verified**: `docs/DESIGN.md` §15.1's example code (`mcp.configure(services)`,
  `api.build_router(services)`) was already a simplified pre-K-013 sketch that omits
  `embed_worker`/`responder`/`trigger` for *both* transports, not just MCP — so the implementer's
  claim that no §15 correction is needed holds; this fix doesn't make an already-accurate section
  inaccurate.
- **Test/query suites verified green myself**: `pytest -q` → `696 passed, 1 deselected`, matching
  the HISTORY.md claim exactly; `./scripts/test_queries.sh` → `282/282 passed`, confirming no
  Cypher was touched. `ruff check` on all five touched/added `.py` files reports no issues.
- **Scope discipline held** — `git -C falkor-chat status`/`diff --stat` show only
  `falkor-chat/{server,docs}/...` paths touched; `kiro/docs/` is untracked and unrelated to this
  diff (the QA report that filed D-1, not touched by the fix).
- **Line-number citations in the BACKLOG/HISTORY entries check out** — I diffed against
  `git show HEAD:./server/falkorchat/{mcp,app}.py` and confirmed `mcp.py`'s pre-fix `configure()`
  and `app.py`'s pre-fix `mcp_mod.configure(...)` call sites are where the entries say they are
  (off by one line at most, immaterial).

## Open questions

None — the findings above are suggestions, not blockers, and the implementer/owner can decide
whether M1's current scale justifies deferring both threading-posture suggestions.
