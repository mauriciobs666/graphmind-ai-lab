# Workflow timers / scheduled wakeups (K-028)

> **Status:** active · **Owner:** `architect` · **Tracks:** K-028

## 1. Goal & scope

Give a parked workflow run a **durable due-time** and **something that ticks**, so an
SLA/escalation step ("if no approval in 48h, escalate") becomes expressible — while the
wakeup path reuses the **existing** `waiting→running` resume CAS, so a timer wakeup and a
concurrent human signal always resolve to exactly one resume, and the graph stays the sole
source of truth for run state.

**In scope:** a `WorkflowRun.wakeAt` due-time stamped at park time; an in-process asyncio
ticker (the recommended mechanism, §3.2) plus a `POST /workflow-runs/due` sweep endpoint as
its manual/cron fallback; a new due-guarded resume CAS; the `config.timeoutSeconds` def
declaration + publish invariant; the `ctx.wakeup` timeout marker guards can branch on;
tests, docs, ops surface.

**Out of scope:** multi-workspace sweeping (the M1 process is single-tenant, §7 open
question O-2); seeding a timeout-carrying def into `seed_workflows.sh` (K-029 owns def-seed
convergence — the escalation def ships as a test fixture + DESIGN example only); MCP
exposure of the sweep; a `ctxVersion` CAS (the pre-existing R-1 residual stays as is);
changing `wait`'s shipped signal-driven semantics (K-024 decision D-C stands — this *adds* a
release mechanism).

## 2. Context & findings

All paths repo-relative under `falkor-chat/` unless noted. Read directly from source
(2026-08-21):

- **The park:** `server/falkorchat/executor.py` — OUTCOME B in `_drive_loop` (`:492-500`)
  calls `repo.suspend_run(ctx.ws, run_id=..., thread_id=run_ctx.get("threadId",""))`. That
  call site is **inside the SHA-locked `_drive_loop` body** (lock `71055f756280`, recompute
  command in DESIGN §6.2's lock note). A second suspend call sits in `_drive`'s
  `HumanHandoffSignal` handler (`:442-446`), *outside* the lock. The step `config` dict is
  in scope at OUTCOME B — the timeout can be computed right there.
- **The CAS:** `server/falkorchat/repository.py` — `suspend_run` (`:1405`, QUERIES §12.3,
  guarded `running→waiting`), `resume_run` (`:1423`, §12.4, single-flight
  `waiting→running`), `resume_run_with_ctx` (`:1440`, §12.13 — §12.4 plus one `SET r.ctx`
  term; live-verified **zero-row contract**: a loser writes neither the flip nor the ctx).
  `executor.resume()` (`executor.py:367`) wraps both: CAS returns `None` → no drive.
  `find_waiting_run_for_thread` (`:1592`, §12.9) is the precedent this design copies: it
  anchors on the **existing `WorkflowRun.status` index** (point lookup on `'waiting'`) with
  a residual property filter — "no new index" is an accepted, PROFILE-verified pattern.
  Note the **`''` sentinel convention**: `waitingThreadId` is created as `''` and reset to
  `''`, never `NULL` — `wakeAt` follows it with a `0` sentinel (no reliance on
  SET-to-NULL property-removal semantics; absent-property comparisons null out of a
  `WHERE`, which is exactly what grandfathers pre-K-028 parked runs, §3.1).
- **The human input path:** `server/falkorchat/services.py` — `submit_workflow_input`
  (`:1430`): get_run → 404/409 checks → validate (D-H, free rejections) → flat ctx merge →
  `executor.resume(run_ctx_json=merged)` → CAS zero-row ⇒ 409
  `WorkflowRunNotWaitingError`, nothing written. `RESERVED_CTX_KEYS` (`:80`) =
  `{"threadId", "error"}` — engine-owned keys a caller may never submit.
  `_validate_def_spec` (`:884`) runs four invariants **deliberately last**, in order
  (waitsForHuman → requiredTools → cmp-guard soundness → at-least-one-transition); its
  docstring makes "new invariants append after existing ones" a stated constraint.
  `_drive_or_fault` (`:1598`) is the D-G fault envelope the wake path must also use.
- **Guards:** `server/falkorchat/guards.py` — cmp paths support **nested** segments
  (`_resolve_path:464` splits on `.`; `_validate_path:425` whitelists roots
  `ctx.`/`output.`), so `{"kind":"cmp","path":"ctx.wakeup.step","op":"eq","value":…}` is
  valid today with **zero guard-engine changes**. Missing path ⇒ `False`, total at drive.
- **App lifecycle:** `server/falkorchat/app.py` — `create_app` owns an
  `@asynccontextmanager` lifespan (`:202`) that already brackets startup/shutdown (actor
  ensure + MCP session manager). `_build_default_app` (`:244`) wires
  executor/trigger only inside `config.ENABLE_AGENT` **and** `config.WORKFLOW_ENABLED`
  (both flags needed — AGENTS.md). `scripts/start_server.sh` runs **one** uvicorn process
  (`--reload` by default, no `--workers`); FastAPI `BackgroundTasks` being request-scoped
  is why no scheduler exists today (BACKLOG K-028 preamble) — a **lifespan-owned asyncio
  task** is the seam that does outlive requests.
- **DDL:** `scripts/bootstrap_schema.sh` — `WorkflowRun` has `runId` (index + UNIQUE),
  `status` and `startedAt` range indexes. No DDL change is needed for this design.
- **Tenancy:** `config.get_context()` resolves the single M1 workspace
  (`FALKORCHAT_WS_ID`); the ticker sweeps that workspace only.
- CPG: considered, not relevant — `cpg_falkorchat` is stale per the teco brief (built
  2026-08-17, 6 commits behind the K-027 executor/guard work), so all findings above were
  read from current source files instead.

## 3. Design & rationale

### 3.1 Durable due-time: `WorkflowRun.wakeAt` property, no new index

**Decision: a property on the run, not a separate timer node.**

- `wakeAt` (epoch **ms**, integer; `0` = "parked with no timer") is written **inside the
  §12.3 suspend CAS itself** — the park and its deadline are one atomic write, so a crash
  can never leave a parked run whose timer silently vanished, and the scheduler never
  holds the deadline anywhere but the graph.
- **Rejected — separate `(:Timer)` node:** a new node type means new DDL (index + UNIQUE
  per rule "every MERGE backed by a constraint"), per-workspace RAM (rule 6), a second
  write per park, and — the killer — a second place run-wakeup truth lives, which is the
  exact "scheduler becomes a second source of truth" risk K-028 names. A property that
  rides the run's own CAS cannot drift from the run's status.
- **No new index.** The sweep anchors on the existing `WorkflowRun.status` index (point
  lookup `'waiting'`) with `wakeAt` as a residual filter — byte-for-byte the §12.9
  precedent (`waitingThreadId` denorm, "rides the existing status index"). Waiting-run
  cardinality per workspace is tiny (same argument §12.9 and the `startedAt` index note in
  `bootstrap_schema.sh` already accepted). **Rule 6 callout:** cost is one integer
  property on parked `WorkflowRun` nodes; zero new node types, zero new indexes, zero new
  edges, `bootstrap_schema.sh` untouched.
- **Def declaration:** a parking step declares `config.timeoutSeconds` (number > 0) inside
  its opaque config. Rule 8 says nothing may filter inside `config`, so the executor
  **materializes** it at park time: OUTCOME B computes
  `wake_at = clock_now_ms + int(timeoutSeconds * 1000)` from the already-deserialized
  `config` dict and passes it to `suspend_run`. Allowed on any step that declares
  `config.waitsForHuman: true` (`human`, `wait`, and the `agent` intake park) — the SLA
  motivating case is a `human` approval step, and `wait` ≡ `human` mechanically (§6.1).
- **Backward compatibility:** existing defs don't declare `timeoutSeconds` → their parks
  write `wakeAt = 0` → the sweep's `r.wakeAt > 0` never matches. Runs parked **before**
  this ships have no `wakeAt` property at all → `r.wakeAt > 0` evaluates null → filtered
  out. Both degrade to today's exact behavior; no migration, no backfill.
- The two run-creation queries (§12.1/§12.12) are **not** touched; `get_run` maps an
  absent `wakeAt` to `0` app-side.

### 3.2 The ticking mechanism — recommendation: in-process asyncio ticker (lifespan task), with the sweep also exposed as `POST /workflow-runs/due`

**Recommended: a lifespan-owned `asyncio` loop inside the served process.**

Why it wins:
1. **The ticker is stateless by construction.** Every deadline lives in the graph
   (`wakeAt`); the loop's only job is "periodically run one index-anchored read + one
   guarded CAS per due run." Restart, crash, `--reload`, redeploy — the first tick after
   startup *is* the catch-up sweep, with no special-case startup logic and no missed-wake
   ledger to reconcile. The scheduler cannot become a second source of truth because it
   *stores nothing*.
2. **The seam already exists.** `create_app`'s lifespan (`app.py:202`) is exactly the
   process-scoped bracket K-024's "BackgroundTasks are request-scoped" objection says is
   missing. Precise lifecycle (per `skills/python-web-quirks` discipline): create the task
   with `asyncio.create_task` inside the lifespan **and keep a strong reference**; each
   sweep runs the sync service call via `await asyncio.to_thread(...)` (the same
   threadpool posture FastAPI gives sync endpoints — a drive can do graph I/O and, for an
   `agent` re-execution, LLM I/O, and must never block the event loop); on lifespan exit,
   `task.cancel()` + `await` with `CancelledError` absorbed. The loop is **serial** (sweep,
   then sleep `interval`), so overlapping sweeps are impossible by construction.
3. **Deployment reality:** `start_server.sh` runs one uvicorn process, no workers, no
   cron/systemd surface anywhere in this repo, dev on WSL2 where cron isn't reliably
   running. An in-process ticker is the only candidate with zero new ops dependencies.
4. **Multi-process safety is the CAS's job, not the ticker's.** If someone runs
   `--workers N` (or the cron fallback fires beside the ticker), N sweeps race per due
   run and the §3.3 CAS admits exactly one winner; losers see zero rows and skip. Redundant
   work, never a double resume. State this in DESIGN §14 rather than pretending the
   process model forbids it.

**Rejected — external cron calling `POST /workflow-runs/due` as the *only* mechanism:**
minute-granularity at best; requires provisioning this repo has no surface for; and its
failure mode is silent (cron unprovisioned/dead ⇒ SLAs never fire, nothing logs). **But
the endpoint itself is kept** (§4 step 5): it is a ~10-line wrapper over the same service
method, gives ops a manual poke, QA a black-box driver, and makes cron a genuine fallback
for a deployment that wants belt-and-braces. Idempotent and safe to expose on this
unauthenticated M1 surface: it can only wake runs that are *genuinely due*, through the
same CAS.

**Rejected — Redis keyspace-notification consumer:** run state lives in graph node
properties, not Redis keys, so this requires shadow `SET key PX <ttl>` entries per parked
run — a literal second source of truth that can drift from the graph (the named K-028
risk, verbatim). Worse, Redis expired-event notifications are **fire-and-forget**: events
for keys that expire while the consumer is disconnected (or the server restarts) are lost,
so a durable-wakeup guarantee would *still* need a startup/periodic sweep — i.e., this
option costs a pub/sub consumer loop, shadow-key lifecycle management, and a drift-repair
job, and still contains the sweep it was supposed to replace. Strictly dominated.

**Ticker knobs:** `FALKORCHAT_TIMER_INTERVAL_S` (float, default `5.0`; `<= 0` disables the
loop even when the engine is wired). The ticker is constructed in `_build_default_app`'s
`WORKFLOW_ENABLED` branch only — same gate as executor/trigger, so the offline
pytest/import baseline never starts a loop. Wakeup precision is `interval`-coarse; for
48-hour SLAs, 5s is beyond ample.

### 3.3 CAS reuse — the due-guarded resume, and the one-winner proof

The existing §12.4/§12.13 queries are **left byte-identical**. The timer path gets one
sibling, following the codebase's "self-contained write path, never a conditional write"
doctrine (§4 first/subsequent; §12.12 vs §12.1):

**New `repository.resume_due_run` (QUERIES §12.16):**

```cypher
MATCH (r:WorkflowRun {runId: $runId})
WHERE r.status = 'waiting'
  AND r.wakeAt > 0 AND r.wakeAt <= $now
  AND r.ctx = $expectedCtx
SET r.status = 'running', r.waitingThreadId = '', r.wakeAt = 0, r.ctx = $ctx
RETURN r.runId AS runId, r.status AS status
```

Three guards, one atomic query (rule 4):
- `status = 'waiting'` — the **same single-flight CAS** as §12.4/§12.13. A timer wakeup
  and a concurrent human `POST /workflow-runs/{id}/input` contend on this one property
  flip; per-query atomicity means exactly one query observes `waiting` and flips it. The
  loser gets zero rows: the human submitter gets the existing 409 with nothing written
  (§12.13's verified contract), the timer logs at debug and skips. **Exactly one resume,
  no second source of truth** — the ticker never decides from memory; it re-reads and
  re-CASes every time.
- `wakeAt > 0 AND wakeAt <= $now` — closes the *re-park race*: between the sweep's read
  and its CAS, a human may resume the run and the run may re-park with a fresh future
  deadline. Without this term the timer would fire that new park early. With it, a
  re-parked-not-yet-due run fails the WHERE ⇒ zero rows ⇒ skip; the next sweep sees the
  new `wakeAt`.
- `ctx = $expectedCtx` (the ctx string the sweep's `get_run` read, passed back verbatim) —
  closes the *stale-merge race*: if the run was resumed, driven, and re-parked **already
  due again** inside the sweep's read→CAS window, a timer CAS built on the old ctx would
  overwrite (erase) the human's just-written keys — the exact silent-loss class D-F
  exists to prevent. Byte-comparing the opaque string is not "filtering inside ctx"
  (rule 8) — it is an equality fence on the whole opaque value, the same way the D-F
  argument treats ctx as an indivisible payload. Zero rows ⇒ skip ⇒ the next tick
  retries against fresh state. The timer can afford pure conservatism because it retries;
  a human can't, which is why §12.13 reports 409 instead.

**What a timer resume delivers:** the sweep merges an **engine-owned** marker into the
run ctx (flat merge, same as human input): `"wakeup": {"kind": "timeout", "step":
"<parked stepKey>", "at": <now ms>}`. `"wakeup"` joins `RESERVED_CTX_KEYS` so no caller
can forge or clobber it (free 400, existing mechanism). Defs branch on it with the
existing cmp family — nested paths verified supported (§2):

```json
{"kind": "cmp", "path": "ctx.wakeup.step", "op": "eq", "value": "approval"}
```

**Executor:** a new `WorkflowExecutor.resume_due(ctx, *, run_id, now, expected_ctx_json,
run_ctx_json) -> str | None` mirroring `resume()` (`executor.py:367`) exactly: call
`repo.resume_due_run`; `None` → `None`; else post-CAS `get_run` + `_drive`. Sits with
`resume` **outside** the SHA lock.

### 3.4 The park-time stamp — deliberate SHA-lock reopen

Computing `wakeAt` needs the step `config` at OUTCOME B, which lives inside the locked
`_drive_loop`. Chosen: **reopen the lock deliberately** — change the OUTCOME B call to

```python
self._repo.suspend_run(
    ctx.ws, run_id=run_id, thread_id=run_ctx.get("threadId", ""),
    wake_at=_wake_at_from(config, self._clock()),
)
```

with `_wake_at_from(config, now_ms) -> int` a module-level helper (outside the lock;
returns `0` when `timeoutSeconds` is absent/invalid — defensive like `_str_list`, a
hand-crafted bad config must not raise in the loop). Then recompute the SHA with the
line-number-independent `awk` command in DESIGN §6.2's lock note and update the recorded
value **in DESIGN §6.2** (the archived `docs/archive/plans/m3-process-flow.md` copy stays
as history — archived docs take header-pointer edits only).

**Rejected — post-park second write** (drive returns `"waiting"`, then a guarded
`set_wake_at` outside the lock): non-atomic; a crash between park and stamp silently
loses the SLA (durably — the run waits forever untimed), and it costs two extra reads per
park to rediscover which step parked. The lock exists to make loop changes deliberate,
not impossible; K-033 already treats "lands inside the locked `_drive_loop`" as "file it
as its own reviewed item," which this plan is. `_drive`'s `HumanHandoffSignal` suspend
(outside the lock) passes no `wake_at` (defaults to `0` — a handoff park is human-paced,
not SLA'd; revisit only if a def ever needs it).

`suspend_run` gains `wake_at: int = 0` and its query one term:
`SET r.status = 'waiting', r.waitingThreadId = $threadId, r.wakeAt = $wakeAt` (§12.3
amended in place — QUERIES.md is the living canonical library). Because **every** park
writes `wakeAt` (value or `0`), a stale deadline can never survive into a later park, and
§12.4/§12.13 need no clearing term (a non-waiting run's `wakeAt` is unreadable by the
sweep, whose anchor is `status='waiting'`).

**The sweep read (QUERIES §12.17, `find_due_runs`):**

```cypher
MATCH (r:WorkflowRun {status: 'waiting'})
WHERE r.wakeAt > 0 AND r.wakeAt <= $now
RETURN r.runId AS runId, r.ctx AS ctx, r.wakeAt AS wakeAt
ORDER BY r.wakeAt ASC
LIMIT $limit
```

Parameterized (rule 1), anchored on the `status` index (rule 3: verify with
`GRAPH.PROFILE`, expect `Node By Index Scan` on `WorkflowRun.status`, never
`NodeByLabelScan` — same expectation §12.9 documents). `LIMIT` (default 50) bounds one
sweep; leftovers are picked up next tick. Returning `ctx` here supplies `$expectedCtx`
without a second read.

### 3.5 After the wakeup — how a def expresses escalation

No def-model or schema change beyond the opaque-config key. The canonical SLA shape
(DESIGN §6 example + the test fixture def `escalation@v1`):

```python
steps = [
  {"key": "approval", "type": "human", "start": True,
   "config": {"waitsForHuman": True, "fields": ["decision"],
              "expects": {"decision": ["approve", "reject"]},
              "timeoutSeconds": 172800}},                      # 48 h
  {"key": "approved",  "type": "decision", "config": {}},      # terminal
  {"key": "rejected",  "type": "decision", "config": {}},      # terminal
  {"key": "escalated", "type": "decision", "config": {}},      # terminal
]
transitions = [
  {"from": "approval", "to": "approved", "order": 0,
   "guard": {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve"}},
  {"from": "approval", "to": "rejected", "order": 1,
   "guard": {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "reject"}},
  {"from": "approval", "to": "escalated", "order": 2,
   "guard": {"kind": "cmp", "path": "ctx.wakeup.step", "op": "eq", "value": "approval"}},
]
```

Authoring conventions to document in DESIGN §6 (both load-bearing):
1. **Order the timeout guard after the data guards** on the same step. Guards sort
   `(guard == "", order)`; if human data is present it should win even when a stale
   `wakeup` marker also matches (see residual R-T1, §7).
2. **Compare `ctx.wakeup.step` to the step's own key**, not merely
   `ctx.wakeup.kind == "timeout"` — a def with two timed parks must not let step A's
   timeout fire step B's escalation arm.
3. Prefer routing a timeout to a **distinct** step (here `escalated`) rather than
   re-parking the same step; re-parking the same timed step re-stamps a fresh `wakeAt`
   (correct) but keeps the old marker in ctx (R-T1).

**Publish validation** — a **fifth invariant** in `services._validate_def_spec`, appended
**after** the existing four (the docstring's LAST-ordering rule: older checks must keep
failing for their own reasons): when a step's config carries `timeoutSeconds`, it must be
a number `> 0` (bounded above, say `<= 10 * 365 * 24 * 3600`, to catch unit mistakes) and
the step must also declare `config.waitsForHuman: true` (a timeout on a non-parking step
is dead config — reject loudly at seed time, not a run that never fires). Non-retroactive:
no existing def or fixture declares the key.

**Budget interplay (document, don't change):** a timeout resume re-executes the parked
step and consumes one `stepCount` like any resume; a def whose timeout arm re-parks the
same step repeatedly will trip `maxSteps` — that is the tripwire working as specified
(K-031).

### 3.6 The service + ticker surface

**`services.wake_due_runs(ctx, *, now: int | None = None, limit: int = 50) -> dict`**
(new; `now=None` → `self._clock()` — the injectable-clock seam for offline tests):
1. `_require_executor()` (503 when unwired — same as every run path).
2. `repo.find_due_runs(ctx.ws, now=now, limit=limit)`.
3. Per due run, isolated (one bad run must not stop the sweep): build
   `merged = base_ctx + {"wakeup": {...}}` (base = the ctx string the sweep read;
   `wakeup.step` = the run's `atStepKey` from a `get_run` — or extend `find_due_runs`'s
   RETURN with `cur.key AS atStepKey` via the existing `AT_STEP` traversal to save the
   second read; implementer's choice, prefer the latter), then
   `_drive_or_fault(..., drive=lambda: executor.resume_due(ctx, run_id=..., now=now,
   expected_ctx_json=base, run_ctx_json=merged_json))`. `status is None` ⇒ lost the CAS ⇒
   count as `skipped`, log debug, **no error** (unlike `submit_workflow_input`'s 409 —
   the timer retries next tick by design).
4. Return `{"now": now, "due": n, "woken": [{runId, status, error?}...], "skipped": k}`.

**`server/falkorchat/timers.py` (new module) — `WorkflowTimerTicker`:**
`__init__(services, context_provider, *, interval_s: float)`; `async def run()`: loop
`{ sweep via asyncio.to_thread(services.wake_due_runs, provider()); log+swallow any
exception (the `background.py` isolation posture); await asyncio.sleep(interval_s) }` —
sweep **first**, so startup catch-up is immediate. `create_app` gains `ticker=None`;
when given, the lifespan creates/cancels the task as §3.2 describes.
`_build_default_app` constructs it in the `WORKFLOW_ENABLED` branch when
`config.TIMER_INTERVAL_S > 0`.

**REST:** `POST /workflow-runs/due` → `services.wake_due_runs(ctx)` — always 200 with the
sweep report envelope (a sweep is a report, like `/readiness`); 503 via the existing
`WorkflowEngineDisabledError` handler when unwired. No `now` override parameter — an
HTTP-reachable clock override could fire production timers early.

**Observability:** add `r.wakeAt AS wakeAt` to `get_run`'s RETURN (absent → `0`
app-side) so `GET /workflow-runs/{id}` shows a parked run's deadline; update any envelope
pins in existing tests.

## 4. Step-by-step implementation

Each step lands green on its own; sequence is bottom-up.

| # | Step | Files | Done when |
|---|---|---|---|
| 1 | **Repository layer.** `suspend_run(..., wake_at: int = 0)` + `SET r.wakeAt = $wakeAt`; new `find_due_runs(ws, *, now, limit=50)` (§3.4 query, incl. `atStepKey` via `AT_STEP` if chosen) and `resume_due_run(ws, *, run_id, now, expected_ctx, ctx)` (§3.3 query); `get_run` returns `wakeAt` (absent→0). Amend QUERIES.md §12.3 in place; add §12.16/§12.17 with the exact Cypher + PROFILE note. | `server/falkorchat/repository.py`, `docs/QUERIES.md` | Unit tests (live FalkorDB, throwaway ws) pin: park stamps wakeAt; park w/o timer stamps 0; due sweep finds only `0 < wakeAt <= now`; pre-K-028-shaped run (no property) excluded; `resume_due_run` zero-row on each of not-waiting / not-due / ctx-mismatch, and on success flips status + resets wakeAt to 0 + writes ctx atomically. `GRAPH.PROFILE` on §12.17 shows `Node By Index Scan`. |
| 2 | **Executor.** Module helper `_wake_at_from(config, now_ms) -> int` (defensive: absent/non-numeric/≤0 → 0); new `resume_due(...)` sibling of `resume()`; **deliberate SHA-lock reopen**: OUTCOME B passes `wake_at=_wake_at_from(config, self._clock())`; recompute lock (`awk` command in DESIGN §6.2) and update the recorded SHA there. `_drive`'s handoff suspend unchanged (defaults 0). | `server/falkorchat/executor.py`, `docs/DESIGN.md` §6.2 (SHA note) | `test_executor_process.py` additions: a `waitsForHuman`+`timeoutSeconds` step parks with the expected `wakeAt` under an injected clock; a non-timed step parks with 0; `resume_due` drives only when the repo CAS applies; `None` propagates without a drive. |
| 3 | **Services.** `RESERVED_CTX_KEYS += {"wakeup"}`; fifth `_validate_def_spec` invariant (§3.5, appended last); `wake_due_runs` (§3.6) with per-run isolation + `_drive_or_fault`. | `server/falkorchat/services.py` | `test_services.py`/`test_process_input.py` additions: submitting `wakeup` as input → 400; publish of `timeoutSeconds` on a non-parking step / non-positive / non-numeric → `WorkflowDefSpecError`, and one **ordering pin** (a spec violating an earlier invariant still fails for the earlier reason); `wake_due_runs` with injected `now` wakes a due run, skips a lost CAS, isolates a faulting run (others still swept), 503 when unwired. |
| 4 | **End-to-end offline acceptance + contention.** New `escalation@v1` fixture def (§3.5) in a new `server/tests/test_timers.py` (published/materialized to the test ws — **not** added to `seed_workflows.sh`). | `server/tests/test_timers.py` | See §5 — the escalation walk, the CAS-contention test, and the stale-merge fence test all green offline (FalkorDB up, no network). |
| 5 | **Ticker + app wiring + REST.** `timers.py`; `config.TIMER_INTERVAL_S` (`FALKORCHAT_TIMER_INTERVAL_S`, default 5.0); `create_app(ticker=...)` lifespan start/cancel; `_build_default_app` construction; `POST /workflow-runs/due` route. | `server/falkorchat/timers.py`, `config.py`, `app.py`, `api.py` | `test_app.py`/`test_api.py`/`test_timers.py`: TestClient lifespan starts and cleanly cancels a stub ticker; a short-interval ticker calls `wake_due_runs` at least once; default app with flags off constructs no ticker (offline baseline untouched); `POST /workflow-runs/due` returns the report envelope, 503 unwired. |
| 6 | **Docs + ops.** DESIGN §6.1 (wait wording: "timers exist via K-028, see §6.4"), new §6.4 subsection (wakeAt model, wakeup marker, authoring conventions 1–3, budget interplay, R-T1), §6.2 run-property list + §6.3 handoff note, §14.4 route row, §14 process-model note (multi-worker ⇒ redundant sweeps, CAS-safe); QUERIES.md done in step 1; `falkor-chat/AGENTS.md` (env var + "both flags + interval" note); `start_server.sh` (header doc, echo line, `export FALKORCHAT_TIMER_INTERVAL_S`); BACKLOG.md K-028 → 🟢 with pointer to this plan; HISTORY.md entry. | docs as listed, `scripts/start_server.sh` | `./scripts/test_queries.sh` green (rule 5 — schema untouched but run it anyway); `grep` finds no remaining "this system has no scheduler" claim stated as current fact. |

## 5. Test strategy

Offline-first (FalkorDB up, no network), mutation-testable — assert **graph state**, not
just return envelopes.

1. **Unit (steps 1–3):** as per the step table's done-columns. The load-bearing pins:
   the three independent zero-row causes of `resume_due_run`; wakeAt stamped atomically
   with the park (kill the process between? — not testable; instead pin that no code path
   writes wakeAt outside `suspend_run`/`resume_due_run` by grep-level review, and that
   `suspend_run` is one query); the validator ordering pin.
2. **Acceptance — escalation walk (step 4):** injected executor+services clocks; start
   `escalation@v1` → parks at `approval`, `GET`-level `wakeAt == start + 2_000` (fixture
   uses `timeoutSeconds: 2`); `wake_due_runs(now=start+1_999)` → `due: 0`, run untouched;
   `wake_due_runs(now=start+2_001)` → run terminal `done` at `escalated`, ctx carries
   `wakeup.step == "approval"`, StepRun trail `approval → escalated`, `wakeAt == 0`.
   Human-first variant: submit `decision=approve` before due → `approved`; a later sweep
   wakes nothing.
3. **CAS contention (the K-028 named test):** park a due run; two threads behind a
   `threading.Barrier` — A: `submit_workflow_input(decision="approve")`, B:
   `wake_due_runs(now=due)`. Assert **exactly one** advanced: terminal is `approved` XOR
   `escalated`; if A won, B's report shows `skipped: 1` and ctx has **no** `wakeup` key;
   if B won, A raised `WorkflowRunNotWaitingError` and ctx has **no** `decision` key;
   in both cases exactly the winner's StepRuns exist. Plus the deterministic split: flip
   the run via one path first, then call the other, pin the loser's exact behavior.
4. **Stale-merge fence:** park run, read it, human-resume + drive to re-park (same step,
   short timeout, already due), then call `resume_due_run` with the *old* ctx as
   `$expectedCtx` → zero rows, human's ctx intact byte-for-byte.
5. **Ticker (step 5):** asyncio-level — interval 0.02 + stub services with an event;
   lifespan cancel leaves no pending task (`asyncio.all_tasks` delta / task done);
   exception in one sweep doesn't kill the loop.
6. **Suite hazards (DESIGN §14.7 / AGENTS.md):** a default `pytest` run wipes `reference`
   at teardown — re-run `seed_workflows.sh` after; `ws:test` has the fixed dim-4 vector
   index — the new tests touch no embeddings; `test_timers.py` publishes its fixture def
   through the service layer into the test graphs the `wf_repo` fixture already manages,
   never into the shared `reference` outside the fixture's lifecycle.
7. **Live (optional, `-m live`):** one smoke: real ticker at 1s, `timeoutSeconds: 2`,
   observe escalation with no manual poke.

## 6. Ops surface

- **Env:** `FALKORCHAT_TIMER_INTERVAL_S` (float seconds, default `5.0`, `<= 0` disables).
  Effective only when `FALKORCHAT_ENABLE_AGENT=1` **and** `FALKORCHAT_WORKFLOW_ENABLED=1`
  (the ticker rides the executor wiring — extend the existing AGENTS.md "both flags"
  paragraph).
- **`start_server.sh`:** document the var in the header, export it, add it to the
  "Workflow:" echo line. No new process, container, or provisioning — **devops has
  nothing to stand up**. Optional hardening for a future deployment: a cron line hitting
  `POST /workflow-runs/due` as a second waker (safe by §3.3); document, don't build.
- **`--reload` note:** dev reloads restart the ticker with the process; durable `wakeAt`
  makes that a non-event.

## 7. Risks & open questions

- **R-T1 (residual, documented):** `ctx.wakeup` persists after a timeout (ctx merge is
  add-only and callers can't clear reserved keys). If the *same* timed step re-parks and
  a human then submits, a mis-ordered timeout guard could fire on the stale marker —
  mitigated by authoring conventions 1–3 (§3.5) and by the escalate-to-distinct-step
  shape. A future engine-side "strip `wakeup` at park" needs a ctx write on the park path
  (today parks don't write ctx) — deliberately deferred; note it beside K-029's ctxVersion
  follow-up family.
- **R-T2 — SHA-lock reopen** is the one deliberate high-ceremony edit: the diff inside
  `_drive_loop` is a single argument addition, but review must re-verify the §2.1 A/B/C
  semantics unchanged and recompute/record the new SHA (step 2). Reviewer checklist item,
  not a code risk per se.
- **R-T3 — sweep under a wrong clock:** `wakeAt` is stamped from the executor's clock and
  compared against the service clock — same process, same wall clock today. If these ever
  live in different processes, clock skew fires timers early/late by the skew. Acceptable
  at 5s granularity vs 48h SLAs; note in DESIGN §6.4.
- **O-1 (implementer's discretion):** `find_due_runs` returning `atStepKey` in-query vs a
  per-run `get_run` — §3.6 states the preference (in-query); either is correct.
- **O-2 (open, caller's call eventually):** multi-workspace sweeping. The ticker sweeps
  `config.WS_ID` only — correct for the M1 single-tenant seam; when multi-tenancy lands,
  the ticker needs a workspace enumeration source (the same problem every per-workspace
  background job will have). Not solvable inside K-028; flag in BACKLOG when closing it.
- **Performance:** one `RO`-class point-lookup query per interval on an all-but-empty
  index — negligible; the PROFILE gate in step 1 is the proof obligation (rule 3).
- **Security:** `POST /workflow-runs/due` adds no new capability beyond "make due things
  happen now"; the wakeup marker is forgery-proof via `RESERVED_CTX_KEYS`.
