# Workflow timers / scheduled wakeups — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-028 · **Version:** 3

> **Revision note — 2026-08-21 (v2 → v3).** `analyst` Pass 2 gated v2 *approve with suggestions*,
> but `teco`'s implementation dispatch (`coder`, U3a) surfaced — and `teco` independently
> re-verified against live source before routing here — a **load-bearing defect neither analyst
> pass caught**: v2's Finding-1 fix (§3.3's "must declare an unconditional fallback transition"
> invariant) does not actually work. `executor._drive_loop` calls `_select_transition` on **every**
> evaluation of a step, including its very first arrival, not only a resumed one; `evaluate_guard`
> (`guards.py:223-224`) fires an unconditional guard (`if not guard: return
> GuardVerdict(decision=True, ...)`) **unconditionally, whenever reached** — there is no
> "first arrival vs. resume" distinction anywhere in that evaluation. A step satisfying v2's
> invariant therefore never reaches OUTCOME B at all: on first arrival, before any real signal,
> the conditional guard is false, falls through to the unconditional fallback, which fires
> immediately — the step never parks. `coder`'s live probe against `ws:test` confirmed this
> behaviourally. This is why v2's own `analyst` Pass 2 review separately found a *second*,
> consequential problem with the same fix (its new-Major: the mandatory default arm forecloses
> the shipped `provision` step's documented "not yet, keep waiting" resumable-non-advancing
> pattern) — both problems trace to the same root cause: an *unconditional* fallback cannot tell
> "a real resume happened" apart from "we just arrived here" or "an explicit not-yet nudge
> happened," because `evaluate_guard` has no such concept.
>
> **v3 fix — Direction A (teco's framing), chosen over Direction B.** Replace the "unconditional
> fallback" invariant with a **conditional, engine-owned marker guard**: a step declaring
> `waitForSeconds`/`waitUntil` must declare an outgoing transition whose guard is
> `{"kind":"cmp","path":"ctx.timerFired","op":"eq","value":"<this step's own key>"}` — a guard
> that is false at first arrival, false on any ordinary human/system resume (nothing else ever
> writes `timerFired`), and true **only** when the sweep itself resumes the run, because the sweep
> now resumes via `executor.resume(ctx, run_id=rid, run_ctx_json=...)` →
> `resume_run_with_ctx` (§12.13) — the **same existing, already-CAS-guarded ctx-write-plus-resume
> primitive** `submit_workflow_input`'s human path already uses (D-F) — writing
> `ctx.timerFired = "<the parked step's own key>"` atomically as part of the resume, instead of
> the plain no-ctx `resume_run` (§12.4) v2 used. `timerFired` joins `RESERVED_CTX_KEYS`
> (`services.py:80`) so no human/API caller can ever set or spoof it. Rejected **Direction B**
> (a time-aware `cmp` guard kind, e.g. `now >= dueAt`, evaluated live inside `guards.py`) as
> larger surface for no added benefit here: it would touch `guards.py`'s evaluation contract and
> module docstring (a shared, heavily-precedented surface used by every guard in the system, not
> just timer steps) to solve a problem Direction A solves by writing one already-reserved ctx key
> through an already-existing primitive.
>
> **Consequence, verified below, not just claimed: Direction A also resolves Pass 2's "not yet"
> finding, not merely documents around it** (§3.3, §6 new test). Since the escalation guard is
> now genuinely conditional on a marker nothing but the sweep ever writes, an ordinary
> `{"provisioned": false}` "not yet" resume leaves `ctx.timerFired` unset, both guards evaluate
> false, and the step correctly re-parks — the mutual-exclusion problem Pass 2 found (a def
> author cannot combine "not yet, stay parked" with a timer on the same step) no longer exists;
> the two patterns compose cleanly. Pass 2's Minor (guard-normalization precision,
> `_serialize_opaque` vs `_normalize_opaque`) is **moot under v3** — that finding was specifically
> about recognizing `""`/`None` as "the unconditional guard," a comparison v3 no longer makes at
> all (the new invariant checks a specific `cmp` guard's fields, never an empty-string default).
>
> **A new defect self-caught while designing v3, closed before shipping — not left for a Pass 3.**
> A bare boolean `ctx.timerFired = true` marker would leak across **different** timer-bearing
> steps in the same run: once any step's timer fires and sets the marker, a *later*, *different*
> timer-bearing step reached afterward would see the (now-stale) marker already `true` on its own
> first arrival and immediately escalate, reproducing the original bug one level down. Closed by
> making the marker **step-scoped** — its value is the specific step's own `key`, not a bare
> boolean — so a later step's guard (`ctx.timerFired == "<its own key>"`) does not match an
> earlier step's stale marker value. The one residual this does not close (documented, not
> engineered around, §8): a def whose graph **cycles back to the same step** after that step's
> own timer already fired once would see its own stale marker and escalate immediately on
> re-arrival — narrower and less likely than the leakage case, and in the same spirit as the
> already-accepted `ctx` R-1 residual and Pass 2's own accepted "not yet" trade-off precedent
> (document, don't over-engineer, for a genuinely narrow case).
>
> **What did not change from v2:** the ticking mechanism (§3.1), the "derive dueness fresh, no
> `WorkflowRun.wakeAt` write, no `_drive_loop` body edit" property (§3.2 — still true; only the
> sweep's *resume call shape* and the *publish-time invariant* change), the RAM/index accounting
> beyond one small addition (§3.4 — `find_due_wait_candidates` gains one more, already-bound
> column, `s.key AS stepKey`; no new anchor, no new index), the batch-fault-isolation design, and
> the v2 scope decision (both `wait` and `human` steps, stakeholder-confirmed, §1). `coder`'s
> already-built and tested U3a work — `_wait_due_at`, the repository query's anchor/traversal
> shape, the sweep's overall batch/CAS-reuse/fault-isolation flow — is **still valid**; only the
> query's RETURN projection (additive), §3.3's invariant, and §3.5 step 5's resume call change.
> Full prior findings: `docs/reviews/workflow-timers.md` (Pass 1 + Pass 2).

## 1. Goal & scope

Give a `wait`/`human` step an optional, durable due-time so it can be released by elapsed time,
not only by an external signal on `POST /workflow-runs/{id}/input`. Concretely: a def author can
declare `config.waitForSeconds` (relative) or `config.waitUntil` (absolute epoch-ms) on any step
of type `wait` or `human` (`services.WAITING_STEP_TYPES`); a periodic in-process sweep, reusing
the **existing** guarded resume CAS family (`QUERIES.md` §12.4/§12.13 — v3 specifically calls
§12.13, the same one the human-input path already uses, §3.5), resumes any run whose due time has
passed, driving it exactly as a human reply would. A step that declares neither key keeps today's
forever-park, signal-only behaviour byte-for-byte — this item is additive only (backlog K-028,
`AGENTS.md` rule: adding a capability, not changing existing behaviour for defs that don't opt in).

**Scope, v2 (stakeholder-decided, `docs/reviews/workflow-timers.md` Finding 2).** v1 of this plan
scoped the feature to `wait` steps only, reading the backlog's mention of `wait`/`human`'s shared
mechanics as a scope boundary. The plan-gate review traced the backlog's own "why it exists" text
(`BACKLOG.md:635-642`) and found its single motivating example — *"if no approval in 48h,
escalate"* — describes the shipped `human`-typed `approval` step (`proof_defs.py:90-98`), not the
`wait`-typed `provision` step; nothing in the backlog item's own text restricts the capability to
`wait`. The stakeholder confirmed: **both `wait` and `human` steps are in scope.** Every design
section below (§3.3, §3.5) and the tests in §6 cover both step types under the existing
`WAITING_STEP_TYPES` set (`services.py:69`) — there is no special-casing between them anywhere in
this design, since they are already mechanically identical parking paths (both mandate
`config.waitsForHuman: true`, both suspend via the identical OUTCOME B).

**Out of scope** (per the backlog item and this plan's own judgment calls, both restated in §8):
- Changing `wait`'s signal-driven release path — untouched, still the primary mechanism.
- A dedicated MCP tool for triggering a sweep manually. The REST endpoint is the manual/cron
  entry point; MCP tools mirror REST 1:1 elsewhere in this codebase (`DESIGN.md` §15.2) but this
  plan does not add the mirror. Low-priority follow-up if a client ever needs it from that surface.
- Multi-workspace sweeping. `CallContext` resolves one hardcoded tenant per process today
  (`DESIGN.md` §14.3) — every other service method already operates on `ctx.ws` alone, and the
  new sweep follows the same grain. A future multi-tenant deployment would need the periodic loop
  to iterate workspaces; not built, flagged in §8.

`CPG: considered, not relevant — cpg_falkorchat exists but is stale (built 2026-08-17T00:40:42Z,
6 commits landed on server/ since, some engine-adjacent, per the coordinator's freshness
pre-check); read executor.py/services.py/repository.py/app.py/api.py/schemas.py/config.py
directly instead, which is what this plan's findings below are grounded in.`

## 2. Context & findings

- **The run model already carries everything a due-time computation needs, with no new write.**
  `WorkflowRun` keeps `AT_STEP` pointed at the parked step while `waiting` (cleared only on
  terminal, `DESIGN.md` §6.2), and `LAST_STEP_RUN` always points at the `StepRun` that
  `record_step_and_advance` (`QUERIES.md` §12.2) wrote for the **advance-to-self** record
  `executor._drive_loop`'s OUTCOME B makes immediately before calling `suspend_run` — every
  single time a run parks or re-parks, including a re-park after a failed/no-op resume attempt.
  `StepRun.startedAt` on that node is therefore always "the moment *this* park began," fresh on
  every suspend cycle. Verified by reading `executor.py:471-510` (`_drive_loop`'s while-loop:
  `_record` runs unconditionally before the `firing`/`waitsForHuman`/re-loop branch, so a `wait`
  or `human` step — both `WAITING_STEP_TYPES` and both mandating `config.waitsForHuman: true`,
  `services.py:69` — always either fires a guard on entry or suspends on the very first pass; it
  can never self-loop under OUTCOME C, so there is no stale prior-iteration `StepRun` to worry
  about; independently confirmed in the plan-gate review, `docs/reviews/workflow-timers.md`
  "What's solid").
- **`Step.config` is an opaque, already-materialized string** (`repository.py:1037-1056`,
  `_PUBLISH_CYPHER`: `st.config = s.config`) — read back verbatim by every existing consumer via
  app-side `json.loads` (`executor._load_json_obj`, `services._normalize_opaque`). A def's `wait`
  step config is therefore already sitting on the graph, one `Step` node away from any parked
  `WorkflowRun`, and rule 8 (`AGENTS.md`) already forbids filtering inside it via Cypher — so any
  due-time decision has to happen app-side regardless of design, which is what §3 below does.
- **`_drive_loop` is SHA-locked** (`71055f756280`, `DESIGN.md` §6.2, recompute recipe documented
  there) — `docs/archive/plans/m3-process-flow.md` §3.1. Its OUTCOME B branch is exactly:
  ```python
  if config.get("waitsForHuman"):
      self._repo.suspend_run(
          ctx.ws, run_id=run_id, thread_id=run_ctx.get("threadId", "")
      )
      return "waiting"
  ```
  (`executor.py:492-500`). The design in §3 deliberately needs **zero** edits inside this lock —
  see §3's "why no lock edit" callout for the reasoning and the two alternatives that would have
  required one.
- **`executor.resume()` (outside the lock, `executor.py:367-393`) is the single existing resume
  entry point**, with two existing, already-shipped modes selected by whether `run_ctx_json` is
  passed: omitted → `repo.resume_run` (§12.4, no ctx write, the chat/trigger resume path's call
  shape); supplied → `repo.resume_run_with_ctx` (§12.13, D-F, the human-input path's call shape,
  `submit_workflow_input`). **v3: the sweep calls the latter mode** (§3.3/§3.5), passing a merged
  ctx carrying the step-scoped `timerFired` marker — the same call shape
  `submit_workflow_input` already makes, not a new one. Either mode satisfies K-028's hard
  requirement — *"the sweep must reuse the existing `resume_run` CAS... do not invent a second
  resume path with separate semantics"* — since both are pre-existing methods on the same guarded
  CAS family; v3 simply resumes through the ctx-writing sibling instead of the plain one, for the
  reasons §3.3 gives. No new resume Cypher, no new resume method, either way.
- **`services._drive_or_fault`** (`services.py:1598-1666`) is a generic `(ctx, run_id, drive:
  Callable[[], str | None]) -> (status, error, fault_ctx)` helper, already used by both
  `start_workflow_run` and `submit_workflow_input`. It converts the four named drive-time faults
  (`NotImplementedError`, `WorkflowConfigError`, `ModelResolutionError`, `ProviderCallError`) into
  a `{"status": ..., "error": ...}` envelope instead of a 500, because the executor's own M-1
  fault net (`executor._drive`, `executor.py:397-449`) has **already** `fail_run`-stamped the run
  terminal before re-raising — "fail, don't zombie" holds regardless of caller. The sweep reuses
  this helper unmodified, per-candidate, inside a loop (§4).
- **`Services.__init__` already takes an injected `clock: Callable[[], int]`**
  (`services.py:530-543`), used today by `_next_ts` for message timestamps and by
  `start_workflow_run`/`submit_workflow_input` for `startedAt`/fault timestamps. Existing tests
  already construct `Services(repo, clock=lambda: 1000, id_gen=...)`
  (`server/tests/test_process_flow.py:60`). This is the exact seam K-028's "injected clock"
  test requirement needs — no new clock plumbing required, `sweep_due_workflow_runs` (§3) just
  calls `self._clock()` for "now" like every other timestamped write already does.
- **Publish-time validation lives in `services._validate_def_spec`** (`services.py:884-1019`),
  which runs a fixed list of invariants **deliberately last, in a fixed order** (documented in its
  own docstring: "Running them last is load-bearing... a new invariant can never mask... a
  pre-existing one"). The `human`/`wait` `waitsForHuman` check (`:949-958`) and the K-027
  `requiredTools`-must-be-`agent`-only check (`:962-989`) are the two closest precedents for the
  new timer-config invariant in §3.
- **Indexing precedent for a hot-filter property that needs a supporting predicate to anchor a
  plan**: `WorkflowRun.startedAt` (`scripts/bootstrap_schema.sh:148-156`) was added specifically
  because "a WHERE-filtered ORDER BY-only property does not pull the label-scan anchor by itself...
  but becomes a real `Node By Index Scan` once paired with that query's supporting predicate" —
  and `WorkflowRun.status` (`:145-146`) already anchors §12.9's `find_waiting_run_for_thread` on a
  point-value scan over a set explicitly documented as tiny ("at most a handful of parked
  conversations per workspace" — `QUERIES.md` §12.9). The sweep's read query (§3) reuses the
  `status` anchor the same way and needs **no new index or property at all** — see §3's RAM
  discussion for why this plan doesn't add a `WorkflowRun.wakeAt` property despite the backlog's
  own scope sketch floating one.
- **REST/schema/error-handling conventions**: `api.py` builds one `APIRouter` via `build_router`
  closing over `services`; every workflow-run route resolves `ctx` via `Depends(get_context)`,
  no `{ws}` path segment (`api.py:253-297`) — the run routes are single-tenant, unlike the
  `/workspaces/{ws}/...` read routes. `schemas.py` defines one `...In` pydantic model per POST
  body with a `MAX_ID_LEN`/`MAX_CONFIG_LEN`-bounded shape (`schemas.py:198-227`,
  `StartWorkflowRunIn`/`SubmitWorkflowInputIn`); POST handlers return the service's dict verbatim,
  no `...Out` model. `app._register_error_handlers` (`app.py:71-141`) maps named exceptions to
  status codes with one shared envelope shape — `WorkflowEngineDisabledError` → 503 is already
  registered and covers the new endpoint for free since `sweep_due_workflow_runs` calls
  `self._require_executor()` first, exactly like the two existing run-mutating methods.
- **`app._build_default_app`** (`app.py:244-317`) wires the executor/trigger only inside
  `if config.WORKFLOW_ENABLED:` (itself only reached when `config.ENABLE_AGENT` is also true —
  `AGENTS.md`'s documented double-flag gotcha). The periodic sweep task (§5) must be gated the
  same way — there is no executor to drive without it.
- **Blocking-sync-call-from-asyncio hazard**: `skills/python-web-quirks/SKILL.md` confirms
  (source-verified against Starlette 1.3.1 / anyio 4.14.1) that FastAPI's own `BackgroundTasks`
  routes a sync callable through `anyio.to_thread.run_sync` (via
  `starlette.concurrency.run_in_threadpool`) rather than calling it inline on the event loop.
  `Services.sweep_due_workflow_runs` is a synchronous, blocking-on-FalkorDB method (like every
  other `Services` method) — the periodic asyncio task in §5 must offload each tick through
  `anyio.to_thread.run_sync` (or the `run_in_threadpool` alias) for the same reason, not call it
  directly inside the loop coroutine.
- **`asyncio.create_task` fire-and-forget GC risk**: the same skill file documents that an
  unreferenced `Task` is a real (if hard to reproduce) risk per the asyncio docs. The lifespan
  wiring in §5 stores the task on `app_.state` and cancels it on shutdown — belt-and-suspenders,
  and also the only way to guarantee clean shutdown at all.
- **Existing CAS-contention test pattern to mirror** (§6): `test_repository.py:1353-1363`,
  `test_resume_run_flips_waiting_to_running_single_flight` — two **sequential** calls against the
  live `wf_repo` fixture (`first`/`second`), asserting the second is `None`. FalkorDB's per-query
  atomicity is what makes a sequential-call simulation a faithful proxy for true concurrency (the
  same argument the existing test already rests on) — the new timer-vs-human race test (§6) uses
  the identical shape, just with one leg going through the sweep path and the other through
  `resume_run_with_ctx`.
- **`wait` step config shape today** (`proof_defs.py:106-107`, the only shipped `wait` step):
  `{"waitsForHuman": True, "signal": "provisioned"}`. Adding `waitForSeconds`/`waitUntil` as two
  more optional keys on the same dict is a strict superset — no existing def's config needs any
  change.

## 3. Design & rationale

### 3.1 The ticking mechanism — an in-process asyncio periodic task, with a REST sweep endpoint as its single implementation

**Recommendation**: one new service method, `Services.sweep_due_workflow_runs`, that is:
1. exposed as `POST /workflow-runs/due` (manual trigger / external-cron entry point, satisfying
   the backlog's own "external cron calling a sweep" sketch without requiring one), **and**
2. called on a fixed interval by an `asyncio.Task` started at FastAPI lifespan startup and
   cancelled at shutdown, gated on `config.WORKFLOW_ENABLED` exactly like the executor/trigger
   themselves.

There is exactly **one** sweep implementation (the service method); the endpoint and the
periodic task are two thin callers of it, not two mechanisms. This is deliberate: the offline
test suite exercises the service method directly with an injected clock (§6) and never has to
touch asyncio at all, while production gets automatic ticking with no separate process, no new
infra dependency, and no manual `cron` step required to get a working demo — consistent with
`start_server.sh`'s existing "one-shot: starts everything" promise.

**Alternatives considered and rejected:**

- **External cron only, no in-process loop.** This is the backlog's own literal sketch
  ("an external cron calling a `POST /workflow-runs/due` sweep") and is still fully supported —
  the endpoint exists standalone and nothing prevents pointing a real cron at it instead of (or
  in addition to) the in-process loop. Rejected as the *sole* mechanism because it breaks the
  "single process, `start_server.sh` starts everything" operating model every other component in
  this repo follows (no other `falkor-chat` capability requires an external scheduler to be
  minimally functional) and it would ship K-028 with an untested, undemoable default — a fresh
  clone would need extra setup work to see a timer ever fire. Keeping the endpoint but adding the
  in-process task gets both: zero-setup default behaviour, and an escape hatch for an operator who
  wants an external, independently-schedulable trigger instead.
- **A separate worker process (APScheduler / Celery beat / a second `python -m` entry point).**
  Rejected: no component in this repo runs a second long-lived process today (`EmbeddingWorker`
  and `AgentResponder` are both in-process, invoked via `BackgroundTasks` or synchronously inside
  a request); adding one here would be new infra shape for a single backlog item, contradicting
  "no `uv`/long-lived-worker infra assumption beyond what's already true of this app" (the brief's
  own constraint) and the single-FastAPI-process posture `AGENTS.md`/`DESIGN.md` document
  throughout.
- **Redis keyspace-notification consumer** (e.g., a shadow key with a TTL = due-time, subscribing
  to `__keyevent@*__:expired`). Rejected on three grounds: (1) it requires enabling
  `notify-keyspace-events` on the shared Redis/FalkorDB instance — an ops-level config change with
  blast radius on every other consumer of that instance, not scoped to this feature; (2) it
  reintroduces exactly the "second source of truth for run state" risk the backlog names as the
  top risk (§8) — a shadow TTL key and `WorkflowRun.status` can drift, and reconciling them after
  a missed/duplicate event is genuinely harder than re-deriving dueness fresh on every sweep tick
  (§3.2); (3) it is much harder to test offline with an injected clock — a real or faked Redis
  expiry event is not something the existing `wf_repo`/`Services(clock=...)` test seam can drive
  deterministically, whereas a plain synchronous service method is trivial to call with any `now`.
- **A `WorkflowRun.wakeAt` property, written at suspend time** (the backlog's own scope sketch:
  "a durable due-time on a parked run — a `WorkflowRun.wakeAt` property + an index"). Considered
  in depth (see §3.2) and rejected in favor of deriving dueness fresh at sweep time from data the
  engine already, unconditionally writes (`StepRun.startedAt` + `Step.config`) — it needs no new
  write path, no new index, and critically, no edit inside the SHA-locked `_drive_loop` body. §3.2
  is the full argument, including the race this design avoids that a naive `wakeAt` write would
  have to solve some other way.

### 3.2 Deriving dueness at sweep time instead of writing `WorkflowRun.wakeAt` — and why this needs no `_drive_loop` edit

The backlog's scope sketch floats writing a `WorkflowRun.wakeAt` property at suspend time. Tracing
where that write would have to happen exposes a real problem: the **only** place inside the
engine that knows, synchronously, "this suspend event is happening right now, for this step's
config" is `_drive_loop`'s OUTCOME B branch (§2) — the exact call site that is SHA-locked. Every
alternative that avoids touching that line turns into a **second, non-atomic write** issued after
`_drive_loop` returns (from `_drive`, which is editable) — and that write needs its own guard
against a run that got resumed (by a human, or a competing sweep) in the gap between the suspend
commit and the follow-up write, or it can leave a **stale** `wakeAt` sitting on a run that has
since resumed and re-parked at a different, non-timed step — which would make the sweep fire a
spurious, unwanted resume later. Closing that race fully needs either a version counter on
`WorkflowRun` (a new field, and a new invariant to maintain — the same shape of problem `ctx`'s
own R-1 residual window already lives with, `DESIGN.md` §6.2) or a tie to the specific `StepRun`
the suspend created, which is not available outside the lock without an extra read.

**This plan sidesteps the whole problem**: `wait`/`human` are the only step types this item
targets (v2 scope, §1), and both handlers are pure and side-effect-free (`executor._run_wait_node`/
`executor._run_human_node`, `executor.py:576-613`) — each already receives `config` and returns
nothing but a descriptive `StepResult`. Its declared timer
(`config.waitForSeconds`/`waitUntil`, §3.3) is static def data, sitting on the `Step` node,
reachable from any parked `WorkflowRun` via the **existing, untouched** `AT_STEP` edge. And the
*when* — "when did the run start waiting" — is already captured, atomically, by the **existing,
untouched** `record_step_and_advance` call that `_drive_loop` makes immediately before every
suspend (§2's finding): `WorkflowRun -[:LAST_STEP_RUN]-> StepRun.startedAt` is always exactly the
timestamp of *this* park, refreshed automatically on every resume-then-re-park cycle, because
`record_step_and_advance` moves the tail pointer every single time a step (including a re-entered
`wait` step) is recorded.

So the sweep computes `dueAt = parkedAt + waitForSeconds * 1000` (or `dueAt = waitUntil`) **fresh,
every tick, from two reads that were already true of the graph** — no write happens at suspend
time at all, so there is no staleness window to close, no version counter to invent, and —
decisively — **no line inside `_drive_loop` needs to change**. The suspend call site
(`executor.py:492-500`) is untouched; the `_record` call site is untouched; `_run_wait_node` is
untouched. This is a strictly better outcome than the backlog's own sketch, not just a
workaround: it is simpler (fewer moving parts), cheaper (§3.4's RAM accounting), and provably
race-free using the codebase's own existing atomicity guarantees, rather than a new one this plan
would have to invent and then document as an accepted residual risk.

**Trade-off honestly stated**: this makes the sweep's per-candidate work O(waiting-run-count)
instead of an index-narrowed O(due-run-count) — every currently-`waiting` run is read and its
config parsed app-side on every sweep tick, not just the ones that are actually due. This is the
same trade-off §12.9's `find_waiting_run_for_thread` already accepts for the identical reason
(the `waiting` set is small by construction — "at most a handful of parked conversations per
workspace" today). If a deployment's waiting-run population ever grows past what a single sweep
tick should scan, §8 names the follow-up (an actual `wakeAt` index, at that point justified by
real numbers rather than anticipated ones) rather than building it speculatively now.

### 3.3 Declaring a due time in def-spec `config`

Two new, optional keys on a `WAITING_STEP_TYPES` step's `config` — `type == "wait"` **or**
`type == "human"` (v2 scope, §1) — a strict superset of today's shape (`proof_defs.py:90-107`):

- `config.waitForSeconds: int` — relative duration in whole seconds from the moment the run parks
  at this step. `dueAt = parkedAt(ms) + waitForSeconds * 1000`.
- `config.waitUntil: int` — absolute epoch-ms timestamp. `dueAt = waitUntil`.

Naming carries the unit explicitly (`waitForSeconds`, not a bare `waitFor`) to avoid an
ms-vs-seconds ambiguity the way `config.maxIterations`/`config.requiredTools` avoid ambiguity by
being self-describing. Neither key is required; **absent both is today's behaviour, unchanged** —
signal-only, forever-park until `POST /workflow-runs/{id}/input`. This is what makes K-028
additive-only: no existing def, including `access-request@v1`, needs a byte change to keep working
identically (its `wait` step, `proof_defs.py:106-107`, and its `human`-typed `approval` step,
`proof_defs.py:90-98`, both declare neither key).

**Publish-time validation** — a new invariant appended to `services._validate_def_spec`,
**deliberately last**, after the existing K-024/K-027 invariants (mirroring the existing
docstring's own stated ordering discipline, `services.py:896-906`, so a new rule can never mask
an older one):

- Both `waitForSeconds` and `waitUntil` declared on the same step → `WorkflowDefSpecError` (400,
  nothing written) — ambiguous, reject rather than silently pick one.
- `waitForSeconds` present but not `isinstance(int, float)` or `<= 0`, or `> MAX_WAIT_FOR_SECONDS`
  (a new constant, judgment call: `180 * 24 * 3600` = 180 days — generous for an SLA/escalation
  use case, defensively bounded against an authoring typo producing an absurd duration) →
  `WorkflowDefSpecError`.
- `waitUntil` present but not `isinstance(int, float)` or `<= 0` → `WorkflowDefSpecError`. No
  upper bound — it is an absolute timestamp, not a duration, and an arbitrarily-distant one costs
  nothing extra to store or evaluate.
- Either key declared on a step whose `type` is **not** in `WAITING_STEP_TYPES` (i.e., not
  `"wait"` and not `"human"`) → `WorkflowDefSpecError`, mirroring the K-027
  `requiredTools`-must-be-`agent`-only precedent exactly (`services.py:976-981`: "only an
  `'agent'` step has an executor code path that can ever satisfy the obligation" → here, "only a
  `wait`/`human` step's sweep-eligibility check ever reads these keys"). This closes off the
  otherwise-silent footgun of declaring a timer on, e.g., a `decision` or `agent` step and having
  it never take effect.
- **New (v3, closes plan-gate Finding 1 for real — v2's "unconditional fallback" fix was itself
  defective, see the header revision note).** A step declaring either `waitForSeconds` or
  `waitUntil` **must also declare at least one outgoing transition whose guard is exactly**
  `{"kind": "cmp", "path": "ctx.timerFired", "op": "eq", "value": "<this step's own key>"}`
  (field-by-field: `guard.get("kind") == "cmp"`, `guard.get("path") == "ctx.timerFired"`,
  `guard.get("op") == "eq"`, `guard.get("value") == step["key"]`, permissive of any *extra* field
  a def author might include) → otherwise `WorkflowDefSpecError`. `timerFired` is a new reserved
  ctx key, added to `RESERVED_CTX_KEYS` (`services.py:80`) alongside `threadId`/`error` — no
  human/API caller can ever set or spoof it (`_reject_reserved_keys` already runs at both
  `start_workflow_run`'s initial ctx and `submit_workflow_input`'s input, so one frozenset
  addition closes both paths with no other code change).

  **Why not a bare unconditional arm (v2's approach) — traced, not asserted.**
  `executor._drive_loop` calls `_select_transition` on **every** evaluation of a step, including
  its very first arrival — there is no "first arrival vs. resume" distinction anywhere in guard
  evaluation (`evaluate_guard`, `guards.py:223-224`, fires an unconditional guard *whenever
  reached*, unconditionally). A step with a bare unconditional fallback therefore never parks at
  all: on first arrival, before any real signal exists, the real conditional guard is false, falls
  through to the unconditional arm, which fires immediately — OUTCOME B (suspend) is never
  reached, so there is nothing for a timer or the sweep to ever act on. This is a functional defect
  in v2, not a documentation gap; verified against live source and `coder`'s own empirical probe
  (header revision note).

  **Why a `ctx.timerFired`-keyed conditional guard works instead.** The sweep no longer calls the
  plain `resume_run` (§12.4); it resumes via `executor.resume(ctx, run_id=rid,
  run_ctx_json=merged_ctx_with_timerFired)` → `resume_run_with_ctx` (§12.13, §3.5), writing
  `ctx.timerFired = "<the parked step's own key>"` atomically as part of the same CAS. The
  escalation guard is therefore **false** at first arrival (nothing has set `timerFired` yet),
  **false** on any ordinary human/system resume (nothing but the sweep ever writes it — the
  reserved-key guard closes this off structurally, not by convention), and **true** only on a
  genuine sweep-triggered resume of *this exact step* — at which point it is guaranteed to fire
  (a `cmp` guard against an exact match is deterministic and total, DESIGN §6.1), guaranteeing the
  advance the original Finding 1 needed, without ever risking a false-positive escalation on an
  unrelated resume. This closes the same churn risk v2 named (a `waitUntil` timer resuming forever
  with no budget check, a `waitForSeconds` timer silently becoming a repeating poll interval,
  `executor.py:493-496`'s OUTCOME-B budget exemption) — but by construction of a real, false-until-
  fired condition, not by forcing an always-true one.

  **Step-scoping closes a leakage bug this plan caught in its own design, before shipping it.** A
  bare boolean `ctx.timerFired = true` (rather than the step's own `key` as the value) would leak
  across *different* timer-bearing steps in the same run: once any step's timer fires, a later,
  different timer-bearing step reached afterward would see the stale `true` marker already set on
  its own first arrival and immediately escalate — reproducing the original bug one level down,
  just conditioned on "did *any* earlier step's timer ever fire in this run" instead of "did
  nothing at all fire yet." Scoping the marker's value to the firing step's own `key` means a
  later step's guard (`ctx.timerFired == "<its own key>"`) never matches an earlier step's stale
  value. The one residual this does **not** close — a def whose transitions cycle back to the
  *same* step after that step's own timer already fired once — is named explicitly in §8 as an
  accepted, documented trade-off (narrower than the leakage case it replaces, and in the same
  spirit as the already-accepted `ctx` R-1 residual, `DESIGN.md` §6.2, and Pass 2's own accepted
  "not yet" precedent: document a genuinely narrow case rather than engineer further complexity
  around it).

  **Bonus, verified not just claimed: this also resolves `analyst` Pass 2's "not yet" finding.**
  Pass 2 found v2's bare unconditional arm forecloses the shipped `provision` step's documented
  "not yet, keep waiting" pattern (`proof_defs.py:106-110`) — a human resuming with
  `{"provisioned": false}` would, under v2, fall through to the now-mandatory unconditional
  escalation arm and wrongly advance. Under v3 that resume writes no `timerFired` key at all (a
  human's `submit_workflow_input` never can — reserved-key rejection), so **both** the real
  `ctx.provisioned == true` guard and the new escalation guard evaluate false, and the step
  correctly re-parks exactly as it does today with no timer declared. The two patterns — "not yet,
  stay parked" and "SLA timer, escalate if nothing happens" — now compose on the same step with no
  special-casing, which Pass 2 identified as impossible under v2's design. Pass 2's Minor
  (`_serialize_opaque` vs `_normalize_opaque` precision for recognizing `""`/`None` as the default
  guard) is **moot under v3** — the new invariant never compares against an empty/null guard at
  all, only against a specific `cmp` guard's fields.

`config` arrives as a dict or a JSON string depending on caller (the M-7 shape-matrix concern
`services.py:191-218` already documents and normalizes via `_normalize_opaque`) — the new checks
reuse that same normalization, so they apply identically whether the def was published via REST
(`config` as a string) or a direct service/MCP call (`config` as a dict), closing the same
escape hatch the existing invariants already close.

### 3.4 The sweep's read query, RAM cost, and why no new index is added

New repository method `find_due_wait_candidates` (`QUERIES.md` §12.16 — the next slot after
§12.15 `read_recent_post_success`):

```cypher
// find_due_wait_candidates — the sweep's read half (K-028). $limit caps work per call.
// Anchors on the EXISTING WorkflowRun.status value index (the same anchor §12.9 already
// uses, on the same "waiting set is tiny" cardinality argument) -- no new index, no new
// WorkflowRun property. Dueness is derived app-side from data the ALREADY-EXISTING
// suspend/record path writes (see plan §3.2) -- this query only reads.
// v3: RETURN gained s.key AS stepKey -- s is already bound (no new traversal), needed so
// the sweep can write the step-scoped ctx.timerFired marker (§3.3/§3.5) without a second read.
MATCH (r:WorkflowRun {status: 'waiting'})-[:AT_STEP]->(s:Step)
OPTIONAL MATCH (r)-[:LAST_STEP_RUN]->(sr:StepRun)
RETURN r.runId AS runId, s.key AS stepKey, s.type AS stepType, s.config AS stepConfig,
       sr.startedAt AS parkedAt
LIMIT $limit
```

Expected to PROFILE as `Node By Index Scan | (r:WorkflowRun)` on the `status` value index
(point lookup on `'waiting'`), then two `Conditional Traverse`s (`AT_STEP`, `LAST_STEP_RUN`) —
the identical traversal shape `get_run` (§12.7) already uses for the same two edges, off an
already-bound node, so no fresh anchor is needed for either hop; the v3 `s.key` addition is a
free projection off the already-bound `s`, not a new traversal, so this shouldn't change the plan
shape `coder`'s U3a work already PROFILE-verified — the implementer should still re-confirm with
`GRAPH.PROFILE` (rule 3) after the RETURN-clause edit, exactly like every other §12 entry, and
record the result in `QUERIES.md` §12.16 the way every other entry does.

**RAM cost (rule 6): zero new properties, zero new indexes, zero new node/relationship types.**
This is the direct payoff of §3.2's design choice — the backlog's own scope sketch anticipated a
`WorkflowRun.wakeAt` property plus an index (a small but non-zero cost the backlog explicitly
flagged to call out); this design needs neither. The new bytes on the graph, v3: (1)
`config.waitForSeconds`/`waitUntil` on a `wait`/`human` step's already-existing, already-opaque
`config` string — a few extra characters on a node type (`Step`) that is rare compared to
`Message`; (2) the reserved `ctx.timerFired: "<stepKey>"` marker the sweep writes into
`WorkflowRun.ctx` on a genuine timer-triggered resume — a short string, same order of magnitude
as `resume_run_with_ctx`'s existing documented cost for a human-submitted ctx ("tens of bytes of
merged human input per run," `QUERIES.md` §12.13), and only ever written on the subset of runs a
timer actually fires for, never on every parked run. No different in kind from any other `config`
or `ctx` key this feature area already stores.

### 3.5 The sweep flow — who calls the executor, and what happens when one due run faults

`Services.sweep_due_workflow_runs(ctx, *, limit=DEFAULT_SWEEP_LIMIT)`:

1. `executor = self._require_executor()` — same 503-if-unwired guard `start_workflow_run`/
   `submit_workflow_input` already use.
2. `now = self._clock()` — the injected clock (§2, §6), never client-supplied (matches the
   `RESERVED_CTX_KEYS`/engine-owned-timestamp doctrine already established for every other
   server-minted stamp in this codebase).
3. `candidates = self._repo.find_due_wait_candidates(ctx.ws, limit=limit)`.
4. For each candidate, app-side (never in Cypher, rule 8): skip if `stepType not in
   WAITING_STEP_TYPES` (i.e., not `"wait"` and not `"human"` — v2 scope, §1; defensive — mirrors
   the `& granted_set` defense-in-depth pattern in `executor._run_agent_node`, `executor.py:697`,
   against a hand-crafted graph write bypassing publish-time validation); parse `stepConfig` via
   the existing `_normalize_opaque`/`_load_json_dict` helpers; compute `dueAt` per §3.3; skip if
   neither key is declared (today's forever-park def, correctly never swept) or `dueAt > now`.
5. For each **due** candidate, **sequentially, in-process** (no per-run background task, no
   threading — driving one run is already fast and synchronous everywhere else in this codebase,
   and batching sequentially keeps the whole sweep call trivially testable with a plain Python
   loop):
   1. **(v3)** `run = self._repo.get_run(ctx.ws, run_id=run_id)` — a fresh, immediate read, the
      same "read right before acting" doctrine `submit_workflow_input` already follows (D-H). If
      `run is None` or `run.get("status") != "waiting"`, the run has already moved on since the
      scan (a concurrent human reply, a concurrent sweep, or the run simply finished) — bucket
      directly as **raced** and move to the next candidate without attempting a resume at all.
   2. **(v3)** Merge the step-scoped marker onto the *current* ctx, never a stale one:
      `merged = _load_json_dict(run.get("ctx")); merged["timerFired"] = candidate["stepKey"];
      merged_json = self._dump_ctx(merged)` — the exact same merge shape
      `submit_workflow_input` already uses for human input (`services.py:1486-1488`), with one
      engine-owned key instead of caller-supplied ones.
   3. Reuse `self._drive_or_fault(ctx, run_id=run_id, drive=lambda rid=run_id, mj=merged_json:
      executor.resume(ctx, run_id=rid, run_ctx_json=mj))` — **the same `_drive_or_fault` helper
      as v2**, but now calling `executor.resume` **with** `run_ctx_json` (v3 — v2 called it with
      none), which routes to `resume_run_with_ctx` (§12.13) instead of the plain `resume_run`
      (§12.4). This is still **the exact same existing entry point and CAS family**
      `submit_workflow_input`'s human path already uses for exactly this "write ctx atomically as
      part of the resume" shape (D-F) — v3 does not invent a resume path, it converges the
      sweep's call shape onto the *same* primitive the human path already uses, rather than the
      plain no-ctx one v2 used. This is the concrete answer to "who calls the executor after a
      sweep resumes a run": the sweep's own call, synchronously, one run at a time, inside the
      same request (REST) or the same periodic-task tick (in-process loop) that found the run due.
6. Bucket each candidate's outcome:
   - **(v3)** a candidate bucketed **raced** at step 5.1 (run already gone or not `waiting` on the
     fresh read) never reaches `_drive_or_fault` at all — no wasted CAS attempt, not just a failed
     one.
   - `status is None and error is None` → the CAS lost the race (a concurrent human `POST
     .../input`, or — harmless — a second sweep tick/process racing this one, §8) → **raced**,
     not an error, no-op.
   - `error is not None` → `_drive_or_fault` caught one of its four named drive-time faults; the
     run is already correctly terminal in the graph (the M-1 net stamped it before
     `_drive_or_fault` observed it) → **faulted**, with the envelope's own diagnostic string.
   - otherwise → **resumed**, carrying the terminal/parked status the drive reached
     (`done`/`waiting`/`failed`).
7. **Batch isolation beyond `_drive_or_fault`'s own four named exceptions**: wrap the whole
   per-candidate call (step 5) in a broad `try/except Exception`, logging and recording a
   **faulted** entry, then `continue`ing the loop — this is a deliberately different posture from
   `start_workflow_run`/`submit_workflow_input`, which let an unnamed exception propagate to a
   500 for their single caller. A batch sweep cannot let run #3 of 50's unexpected exception stop
   runs #4–50 from ever being evaluated. This is safe precisely because `executor._drive`'s own
   M-1 fault net (`executor.py:397-449`) already `fail_run`-stamps the run terminal before
   re-raising **regardless of exception type** — the sweep's outer catch only needs to stop the
   *Python* exception from propagating, not re-derive terminal correctness, which the engine's
   existing safety net already guarantees. This directly answers the brief's stated failure-mode
   question ("sweep finds 50 due runs, one throws mid-drive — does the sweep continue to the
   rest?"): **yes**, by construction, and the faulted run is left in a defined terminal state, not
   a zombie.
8. Return `{"checked": len(candidates), "due": len(due), "resumed": [...], "raced": [...],
   "faulted": [...]}` — each of the three lists holding `{"runId": ..., ...}` dicts (status for
   `resumed`, `error` for `faulted`, nothing extra for `raced`). This dict flows straight through
   the REST layer unchanged, matching every other workflow-run POST endpoint's existing convention
   (`api.py:253-268`, no `...Out` pydantic model).

**Why this design satisfies the single-winner requirement for free**: step 5.3 goes through
`executor.resume` → `repo.resume_run_with_ctx` (§12.13), the **unmodified** guarded CAS — the same
one the human `POST /workflow-runs/{id}/input` path (`submit_workflow_input`) already uses (v3:
sweep and human-with-input now converge onto the identical primitive; v2 had them on two
*different* existing primitives, `resume_run` vs `resume_run_with_ctx`, both real but distinct).
Only one caller ever observes `status = 'waiting'` and flips it; every other caller's `WHERE` fails
and it does not re-enter the executor (`QUERIES.md` §12.13's own "verified... a run that is not
`waiting` matches the node but fails the `WHERE` ⇒ zero rows and NOTHING is written"). The loser's
merged ctx (whether the sweep's `timerFired` marker or a human's real input) is simply never
persisted — this is the same, already-accepted single-winner shape two racing humans already have
today, not a new residual-window class (§8 addresses this explicitly, since `teco`'s Direction-A
framing specifically asked whether the sweep's marker write reopens R-1). The plain, no-ctx
`resume_run` (§12.4) remains in use, unchanged, for the ordinary chat/trigger resume path
(`trigger.py`) — v3 does not touch it or retire it, it only changes which of the two *existing*
resume entry points the sweep itself calls.

### 3.6 The endpoint and the periodic task

**`POST /workflow-runs/due`** — sibling to `POST /workflow-runs`/`POST /workflow-runs/{id}/input`
in `api.py`, same `ctx: CallContext = Depends(get_context)` resolution, no `{ws}` path segment:

```python
@router.post("/workflow-runs/due")
def sweep_due_workflow_runs(
    body: SweepDueWorkflowRunsIn, ctx: CallContext = Depends(get_context)
):
    return services.sweep_due_workflow_runs(ctx, limit=body.limit)
```

New `schemas.py` model:

```python
class SweepDueWorkflowRunsIn(BaseModel):
    """`POST /workflow-runs/due` — sweep parked wait-timer runs past their due time."""
    limit: int = Field(DEFAULT_SWEEP_LIMIT, ge=1, le=MAX_SWEEP_LIMIT)
```

`DEFAULT_SWEEP_LIMIT = 200`, `MAX_SWEEP_LIMIT = 1000` — new constants in `services.py` alongside
the existing `RAG_QUERY_TIMEOUT_MS`-style module constants (`services.py:99-110`); `schemas.py`
imports `DEFAULT_SWEEP_LIMIT` the same way it already re-exports `MAX_CONFIG_LEN` in the other
direction (boundary ↔ service constant sharing, `services.py:47-52`'s existing precedent, mirrored
not duplicated). No new error type needed: `WorkflowEngineDisabledError` (→503) is already
registered (`app.py:112-127`); a malformed body (e.g., `limit` out of range) gets pydantic's
existing 422 for free.

**The periodic task** — `app.py`, gated identically to the executor/trigger:

```python
_DEFAULT_SWEEP_INTERVAL_S = 30.0

async def _sweep_loop(services, context_provider, *, interval_s, limit):
    while True:
        await asyncio.sleep(interval_s)
        try:
            ctx = context_provider()
            await anyio.to_thread.run_sync(
                lambda: services.sweep_due_workflow_runs(ctx, limit=limit)
            )
        except Exception:
            logging.getLogger(__name__).exception("workflow sweep tick failed")
```

`create_app` gains an opt-in constructor param mirroring the existing `responder`/`embed_worker`
pattern (`app.py:143-175`'s own documented convention: "opt-in out-of-band handlers... default to
`None` so building the default app stays network-free"):

```python
def create_app(
    services=None, *, context_provider=None, mount_mcp=True, web_dir=None,
    responder=None, embed_worker=None, trigger=None,
    sweep_interval_s: float | None = None, sweep_limit: int = DEFAULT_SWEEP_LIMIT,
) -> FastAPI:
    ...
    @asynccontextmanager
    async def _lifespan(app_: FastAPI):
        services.ensure_actor(provider())
        sweep_task = None
        if sweep_interval_s is not None:
            sweep_task = asyncio.create_task(
                _sweep_loop(services, provider, interval_s=sweep_interval_s, limit=sweep_limit)
            )
            app_.state.sweep_task = sweep_task  # held reference — python-web-quirks GC note
        if mcp_lifespan is not None:
            async with mcp_lifespan(app_):
                yield
        else:
            yield
        if sweep_task is not None:
            sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweep_task
```

`_build_default_app` passes `sweep_interval_s=config.WORKFLOW_SWEEP_INTERVAL_S` **only** inside
the existing `if config.WORKFLOW_ENABLED:` branch (`app.py:299-315`) — every other path (the
plain M1/M2 app, the disabled-engine default) leaves it `None`, so `create_app`'s default
behaviour and the pytest baseline are untouched, matching the file's own stated invariant
("building the default app stays network-free"). New `config.py` env var:
`WORKFLOW_SWEEP_INTERVAL_S: float = float(os.environ.get("FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S",
"30"))`, read only where `WORKFLOW_ENABLED` is already read.

## 4. Data model changes

- **No new label, no new relationship type, no new node property beyond `Step.config`'s two new,
  optional, opaque keys** (`waitForSeconds`/`waitUntil` — already-opaque string content, not a new
  graph-visible property). See §3.4 for the full RAM accounting.
- **No new index, no new constraint, no `bootstrap_schema.sh` change.** The sweep's read query
  reuses `WorkflowRun.status` (already indexed, `bootstrap_schema.sh:145-146`) and the `AT_STEP`/
  `LAST_STEP_RUN` edges (already traversed identically by `get_run`, §12.7).
- **Consequence for sequencing (§5): this is a single implementer unit, not two.** The backlog's
  own owner routing anticipated a possible `graph-dba` gate ("`graph-dba` only if a due-time index
  is added") — this design adds none, so there is no DDL-side work to split out. Everything is
  `server/falkorchat/*.py` + `server/tests/*.py` + the doc updates in §7, sized for one `coder` or
  `tdd-engineer` unit.

## 5. Step-by-step implementation sequence

All within `falkor-chat/server/falkorchat/` and `falkor-chat/server/tests/` unless noted. Ordered
so the tree stays buildable and each step is independently reviewable; steps 1–4 are pure
additions (no existing call site changes), step 5 changes one existing call site
(`services.py:1667-1678`'s neighbourhood is untouched — the sweep is a new method, not a rewrite
of `resume_workflow_run`), steps 6–8 wire it up.

1. **`services.py` — publish-time validation.** Add `MAX_WAIT_FOR_SECONDS`, `DEFAULT_SWEEP_LIMIT`,
   `MAX_SWEEP_LIMIT`, and (v3) `TIMER_FIRED_CTX_KEY = "timerFired"` constants near the existing
   `RAG_QUERY_TIMEOUT_MS`/K-039-sample-size block (`:99-110`); add `TIMER_FIRED_CTX_KEY` to
   `RESERVED_CTX_KEYS` (`services.py:80`). Add the new invariants to `_validate_def_spec` (§3.3's
   exact rules — timer-key structural validation, the `WAITING_STEP_TYPES`-only restriction,
   **and (v3, replacing v2's defective default-arm requirement)** the escalation-guard-shape
   requirement: at least one outgoing transition whose guard, once normalized via
   `_normalize_opaque` (M-7 shape matrix, matching the existing `cmp`-guard structural-check call
   site two lines above it, `services.py:992`), has `kind == "cmp"`, `path == "ctx.timerFired"`,
   `op == "eq"`, and `value == step["key"]`), appended **after** the existing K-024/K-027 blocks
   and **before** the K-024 U4b zero-transition check — or, if the implementer judges the
   zero-transition check should stay unconditionally last (it currently is), append the new
   invariants just before it, preserving "last" for the zero-transition check specifically since it
   is about publish-atomicity, not content, per its own docstring's ordering argument. Either
   placement is fine as long as the new checks still run strictly after every existing check they
   could otherwise mask — this is a judgment call for the implementer to make explicit in a code
   comment, not a hidden choice. The check needs the def's `transitions` list grouped by `from`
   step key (already a parameter of `_validate_def_spec`, `services.py:884-886`
   `*, kind, steps, transitions` — no new data has to be threaded in) and each step's own `key`
   (already available per-step in the same `steps` loop) to build the expected per-step guard
   shape.
2. **`services.py` — pure helpers.** A private, testable function (module-level, mirroring
   `_normalize_opaque`'s style) computing due-ness:
   `_wait_due_at(step_type: str, config_raw: Any, parked_at: int | None) -> int | None` — returns
   `None` when `step_type not in WAITING_STEP_TYPES`, `parked_at` is `None` (defensive — should not
   happen per §2's finding, but never trust a read), or neither timer key is declared; otherwise
   the computed `dueAt` epoch-ms. Unit-testable with no repo/graph at all (§6). **Unchanged from
   v2** — this helper has nothing to do with the resume-marker mechanism; `coder`'s U3a
   implementation of it stands as-is.
3. **`repository.py` — `find_due_wait_candidates`.** New RO method next to
   `find_waiting_run_for_thread`/`find_runs_for_thread` (`repository.py:1592-1636`), running §3.4's
   query verbatim (v3: RETURN gains `s.key AS stepKey`, an additive projection off the already-bound
   `s` — no traversal-shape change), returning `list[{"runId": ..., "stepKey": ...,
   "stepType": ..., "stepConfig": ..., "parkedAt": ...}]`. **`coder`'s U3a implementation needs
   only the one-line RETURN-clause edit**, not a rewrite — the MATCH/anchor is unchanged.
4. **`services.py` — `sweep_due_workflow_runs`.** Implements §3.5's flow, using steps 1–3's
   helpers plus the existing `_require_executor`/`_drive_or_fault`/`self._clock()`/`get_run`/
   `_load_json_dict`/`_dump_ctx` (the last three already used identically by
   `submit_workflow_input`, `services.py:1461,1486-1488`). Placed near `resume_workflow_run`
   (`services.py:1667-1679`) since it is its batch-mode sibling; leave `resume_workflow_run` itself
   untouched. **v3 changes this method's body relative to `coder`'s U3a draft**: step 5 now reads
   the run fresh, merges the step-scoped `timerFired` marker, and calls `executor.resume(...,
   run_ctx_json=...)` instead of the plain no-ctx call (§3.5) — the batch loop shape, bucketing
   logic (raced/faulted/resumed), and outer `try/except Exception` (§3.5 step 7) are unchanged.
5. **`schemas.py` — `SweepDueWorkflowRunsIn`.** Next to `SubmitWorkflowInputIn`
   (`schemas.py:218-227`).
6. **`api.py` — `POST /workflow-runs/due`.** Next to the existing `/workflow-runs` /
   `/workflow-runs/{run_id}/input` routes (`api.py:253-268`).
7. **`config.py` — `WORKFLOW_SWEEP_INTERVAL_S`.** Next to the existing `WORKFLOW_ENABLED`/
   `TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` block (`config.py:105-115`).
8. **`app.py` — the periodic task.** `_sweep_loop`, `create_app`'s new
   `sweep_interval_s`/`sweep_limit` params and lifespan wiring (§3.6), and
   `_build_default_app`'s one new keyword passed inside the existing `WORKFLOW_ENABLED` branch
   (`app.py:299-315`). Import `anyio`/`asyncio`/`contextlib` at module top as needed (`anyio` is
   already a transitive dependency via Starlette — confirm it is importable directly rather than
   only via `starlette.concurrency`, or use `starlette.concurrency.run_in_threadpool` instead,
   which is the same call one layer up and avoids a new direct dependency declaration; the
   implementer's call, functionally identical per the python-web-quirks finding in §2).
9. **Tests** (§6, detailed below) — can and should be written alongside steps 1–4 (offline,
   `wf_repo`-based) before steps 5–8 (REST/async wiring) land, since the REST/lifespan layer is a
   thin pass-through with nothing new to unit-test beyond "the route calls the service with the
   right args" and "the lifespan starts/stops a task," both cheap smoke checks.
10. **Docs** (§7) — update `DESIGN.md` §6.1/§6.2/§6.3, `QUERIES.md` §12.16,
    `scripts/start_server.sh`'s header comment (v2, Minor 4), `AGENTS.md` if the new
    env var needs a rule-adjacent callout, `BACKLOG.md`'s K-028 entry (mark delivered, per the
    repo's own doc-curation convention that every unit's done-condition includes doc updates),
    `HISTORY.md` (one dated entry).

No step requires touching `executor._drive_loop`'s locked body, `executor._record`,
`executor._run_wait_node`'s signature, or `repository.suspend_run`/`resume_run`/
`resume_run_with_ctx`'s existing Cypher — confirmed by tracing every call site touched above
against the lock boundary in §2's findings.

## 6. Test strategy

All offline (no LLM, no network beyond the local FalkorDB instance `AGENTS.md` already requires
for the default `pytest` run) — no `-m live` marker needed for any test below.

1. **`_wait_due_at` pure-function tests** (new, or added to an existing small-helper test module) —
   no repo, no clock injection needed beyond passing literal ints:
   - `stepType not in WAITING_STEP_TYPES` (e.g. `"decision"`, `"agent"`) → `None` regardless of
     config.
   - `stepType == "human"` with `waitForSeconds=60, parkedAt=1000` → `dueAt == 61000` (v2 scope —
     the pure function treats `wait` and `human` identically; this case is the one that would have
     silently regressed if the helper still hard-coded `== "wait"`).
   - neither `waitForSeconds` nor `waitUntil` in config → `None` (today's forever-park case,
     the additive-only requirement's own acceptance test), for both `wait` and `human`.
   - `waitForSeconds=60, parkedAt=1000` → `dueAt == 61000`.
   - `waitUntil=5000` → `dueAt == 5000` (parkedAt ignored for the absolute form).
   - `parkedAt is None` (defensive) → `None`.
2. **Publish-time validation tests** — extend `server/tests/test_executor_process.py` (the
   existing home for `_validate_def_spec` shape-matrix tests, `test_executor_process.py:1-17`
   already documents "the M-7 shape matrix... neither invariant may be escapable that way") with:
   - both `waitForSeconds` and `waitUntil` declared → `WorkflowDefSpecError`, both as a
     dict-shaped and a string-shaped `config` (the M-7 matrix).
   - `waitForSeconds=0` / negative / non-numeric → `WorkflowDefSpecError`.
   - `waitForSeconds` over `MAX_WAIT_FOR_SECONDS` → `WorkflowDefSpecError`.
   - either key declared on a step type **outside** `WAITING_STEP_TYPES` (e.g. `decision` or
     `agent`) → `WorkflowDefSpecError` (**not** a `human`-typed step — that is now the positive
     case below).
   - **v3, replacing v2's (defective) Finding-1 test:** a `wait` step declaring
     `waitForSeconds`/`waitUntil` with **no** outgoing transition matching the canonical
     `{"kind":"cmp","path":"ctx.timerFired","op":"eq","value":"<own key>"}` shape (mirroring
     `proof_defs.py`'s shipped `provision` step's own shape — a single `ctx.provisioned == true`
     transition, nothing else) → `WorkflowDefSpecError`. Repeat for a `human`-typed step with the
     same shape (both step types are in scope for this invariant). Also: a step declaring the
     escalation guard but with the **wrong** `value` (e.g. some other step's key, or a literal
     `true`/`""` — the v2 shape) → still `WorkflowDefSpecError`, proving the check is genuinely
     shape- and value-specific, not merely "has *a* conditional transition."
   - either key declared alone on a valid `wait` step **or** a valid `human` step (both with
     `waitsForHuman: true`, satisfying the pre-existing invariant too, **and** an outgoing
     transition carrying the exact canonical escalation guard keyed to that step's own `key`) →
     publishes cleanly, **positive** case — one for each step type. A **second** positive case:
     the same step also carries its own real conditional guard (e.g. `ctx.provisioned == true`)
     *alongside* the escalation guard — both declared, both legal, order between them irrelevant
     (both are conditional, §3.3).
   - a `wait`/`human` step declaring neither timer key still publishes cleanly with **no**
     escalation-shaped transition at all (regression guard: the new invariant must only fire when a
     timer key is actually declared — `access-request@v1`'s shipped `provision`/`approval` steps,
     neither declaring a timer key, must keep publishing unchanged).
   - a caller-supplied `run_ctx`/`input` carrying the reserved `timerFired` key, at either
     `start_workflow_run` or `submit_workflow_input` → `WorkflowInputRejectedError` (400, nothing
     written) — the same `RESERVED_CTX_KEYS` rejection path `threadId`/`error` already exercise,
     now covering the new key too.
3. **`find_due_wait_candidates` repository tests** — new tests in `test_repository.py`, next to
   the §12.3/§12.4 CAS tests (`test_repository.py:1330-1363`), against the live `wf_repo` fixture:
   - a `waiting` run parked at a `wait` step with `waitForSeconds` declared → one candidate row
     with the expected `stepKey`/`stepConfig`/`parkedAt` (v3: assert `stepKey` explicitly — the new
     projection).
   - a `waiting` run parked at a `human` step, or a `wait` step with neither timer key → still
     returned as a candidate (the query itself is due-agnostic, §3.4 — filtering happens app-side,
     step 4 covers the app-side skip).
   - a `running`/`done`/`failed` run → never returned (the `status:'waiting'` anchor).
   - `LIMIT $limit` respected with more waiting runs than the limit.
   - `GRAPH.PROFILE` check (recorded in `QUERIES.md` §12.16 by the implementer, per §3.4) —
     `Node By Index Scan | (r:WorkflowRun)`, no label scan, re-confirmed after the v3 `stepKey`
     RETURN-clause addition (expected unchanged, per §3.4's reasoning — verify, don't assume).
4. **`sweep_due_workflow_runs` service tests** — new file, `server/tests/test_workflow_timers.py`
   (a sibling to `test_executor_process.py`/`test_process_flow.py`, following the existing
   `_make_executor`/`_start` helper pattern from `test_executor_process.py:53-90`), against the
   real `wf_repo` fixture with a **real `WorkflowExecutor`** wired (so the sweep actually drives a
   run, not a stub):
   - **the injected-clock seam (v2, Finding 3 — load-bearing wiring detail, not optional).**
     `Services.__init__` and `WorkflowExecutor.__init__` each take their **own** `clock:
     Callable[[], int]`, defaulting to two **separately defined** module-level functions
     (`services.py:149`, `executor.py:99`); nothing in production wiring (`_build_default_app`,
     `app.py:299-315`) or anywhere else connects one to the other — both simply call their own
     real-clock default independently in production. `StepRun.startedAt` (the sweep's `parkedAt`)
     is minted by the **executor's** clock inside `_record` (`executor.py:1008`); the sweep's `now`
     comes from `self._clock()` on **`Services`** (§3.5 step 2). **This test must therefore
     construct both objects with the exact same `clock=` callable** — a single shared mutable
     counter, or a fixed `lambda: FIXED_NOW`/its stepped variant — passed to *both*
     `Services(wf_repo, clock=shared_clock, ...)` and the test's `WorkflowExecutor(..., clock=
     shared_clock)`, then wired together via `services.set_executor(...)`
     (`services.py:561-563`). Do **not** follow `_make_executor`'s existing pattern
     (`test_executor_process.py:53-60`) unmodified here — that helper deliberately gives the test
     executor its **own** independent `itertools.count(1000)` clock, disconnected from any
     `Services` instance, which is correct for every *other* executor test (they only care about
     internal StepRun ordering) but would make this test's `parkedAt` (executor clock) and `now`
     (`Services` clock) two unrelated numbers — silently passing or failing for the wrong reason,
     not proving the due-time comparison at all. Build a small local `_make_shared_clock()` or
     `itertools.count`-backed counter in this test file and pass the identical object to both
     constructors; do not reuse `_make_executor` for this specific test without first threading its
     `clock=` parameter through to match `Services`'s.
   - With that shared clock: publish/materialize/start a `process`-kind def with one `wait` step
     declaring `waitForSeconds=60` **and** the canonical `ctx.timerFired == "<own key>"` escalation
     transition (satisfying the v3 invariant, §3.3/§6 test 2), drive it to `waiting`, then call
     `sweep_due_workflow_runs` with the shared clock still returning a timestamp **before**
     `parkedAt + 60000` → asserts `due == 0`, `resumed == []`, run still `waiting`, and (v3) the
     run's `ctx` carries **no** `timerFired` key at all — the sweep took no action, wrote nothing.
     Advance the shared clock (or rebuild with a `Services`/`WorkflowExecutor` pair sharing a clock
     that now returns a timestamp **after** the due time) → asserts the run resumed and drove past
     the wait step, **and** (v3) `get_run`'s `ctx` now carries `timerFired` equal to that step's own
     `key` — the escalation guard's own condition, proving the marker mechanism, not just the
     dueness comparison, actually worked end to end. **No real sleep anywhere** — this is the
     concrete fulfilment of "an injected clock driving the sweep."
   - **repeat the same before/after due-time case for a `human`-typed step** declaring
     `waitForSeconds` and its own canonical escalation transition — proving the sweep resumes a
     `human` park exactly as it does a `wait` park, since both flow through the same
     `WAITING_STEP_TYPES`-scoped filter (§3.5 step 4).
   - a `wait`/`human` step with **no** timer key declared, parked, swept with any `now` → never
     appears in `resumed` (the additive-only guarantee, end-to-end this time, not just at the
     pure-function level).
   - **new (v3, resolves `analyst` Pass 2's "not yet" finding — verify, don't just document):** a
     `wait` step shaped exactly like `proof_defs.py`'s shipped `provision` step (a real
     `ctx.provisioned == true` transition, `config.fields`/`expects` deliberately absent so
     `{"provisioned": false}` is an accepted "not yet" input, per `proof_defs.py:106-110`'s own
     comment) **plus** a `waitForSeconds` timer and its canonical escalation transition. Resume it
     via `submit_workflow_input({"provisioned": false})` (the "not yet" nudge) **before** the due
     time passes → asserts the run **re-parks** (still `waiting`, same step), and its `ctx` carries
     `provisioned: false` but **no** `timerFired` key — proving the "not yet" pattern survives
     unbroken on a timer-bearing step, the exact composition Pass 2 found impossible under v2's
     bare-unconditional-arm fix. Then sweep it past the due time → asserts it now resumes via the
     escalation branch, `ctx.timerFired` set, exactly as the plain timer case above.
   - **new (v3, closes the self-caught leakage bug — see header revision note):** a `process`-kind
     def with **two** timer-bearing steps in sequence (step `A` then step `B`, both `wait`-typed,
     each with its own `waitForSeconds` and its own canonical `ctx.timerFired == "<own key>"`
     escalation transition, `A`'s escalation edge advancing to `B`). Sweep `A` past its due time
     (advances into `B`, `ctx.timerFired == "A"`) — then assert **`B` correctly parks** on arrival
     (its own escalation guard checks `ctx.timerFired == "B"`, which is false — the stale `"A"`
     value does not match) rather than immediately escalating on first arrival. This is the
     regression test for the bug a bare boolean marker would have reproduced one level down; it
     must never start failing.
   - `limit` respected — more due runs than `limit` → only `limit` are evaluated per call
     (a second call with a fresh clock/limit sweeps the rest).
   - a due run whose next step raises one of `_drive_or_fault`'s four named faults (reuse an
     existing fixture pattern from `test_process_flow.py`/`test_executor.py` that already
     provokes one, e.g. a `NotImplementedError`-typed unimplemented step type reached via the
     resumed drive) → appears in `faulted`, run is `failed` in the graph afterward (`get_run`
     assertion), and — the batch-isolation requirement — a **second** due run in the same sweep
     call (a `wait` step declaring a shorter `waitForSeconds` **and** its own canonical escalation
     transition, ordered so it's evaluated after the faulting one) still ends up in `resumed`,
     proving the sweep didn't stop at the first fault. Both runs' `WorkflowExecutor`/`Services`
     share the one clock per the injected-clock note above; the faulting step's own escalation
     transition is what lets the sweep-triggered resume advance into the unimplemented step type
     in the first place.
   - **the CAS-contention test** (mirrors `test_repository.py:1353-1363`'s two-sequential-calls
     shape, per §2's finding; v3: rewritten to assert *which branch fired*, a stronger proof than
     v2's version could give): park a run at a `wait` step declaring a due `waitForSeconds`, its
     own canonical escalation transition, **and** a second, conditional transition accepting a real
     domain signal (e.g. `ctx.provisioned == true`) — a step can be timer- and signal-eligible
     simultaneously, since nothing in this design makes them mutually exclusive. First call
     `submit_workflow_input({"provisioned": true})` (the human path, via `resume_run_with_ctx`) to
     resume it, **then** call `sweep_due_workflow_runs` with a clock past the due time → assert:
     the sweep's attempt lands in `raced` (the fresh `get_run` in §3.5 step 5.1 already sees
     `status != 'waiting'`, or, if it raced the CAS itself, `resume_run_with_ctx`'s `WHERE
     status='waiting'` fails because the human path already flipped it to `running`); the run
     advanced via the **domain** guard, never the escalation one; and `ctx.timerFired` was **never
     written** — the sweep's attempted merge never reached the graph. Then the mirror ordering:
     sweep resumes first (assert `resumed`, advanced via the **escalation** guard this time,
     `ctx.timerFired` set to the step's own key), then a same-run
     `submit_workflow_input({"provisioned": true})` call gets the existing `409
     WorkflowRunNotWaitingError` (already covered by existing tests for the human-vs-human race —
     this test only needs to confirm the sweep path produces the identical 409 for a *second*
     resume attempt, proving the sweep is not a distinguishable, special-cased resume path from the
     CAS's point of view — now demonstrated by which branch fired, not merely that *a* resume
     happened).
5. **REST smoke tests** — extend `test_api.py` (or wherever `/workflow-runs` POST routes are
   already tested) with one happy-path test hitting `POST /workflow-runs/due` against a throwaway
   `FastAPI` app built the way `test_api.py`'s existing zero-risk pattern does
   (`AGENTS.md`'s "Probing shared graph state without mutating it" note — a node that builds its
   own app rather than requesting the `wf_client` fixture), asserting the response shape and that
   `WorkflowEngineDisabledError` still 503s when the engine isn't wired (reusing the existing
   registered handler, no new test needed for the mapping itself — just that this new route hits
   it).
6. **Lifespan smoke test** (light) — a test constructing `create_app(services,
   sweep_interval_s=0.01, sweep_limit=1)` against a fake/stub `Services.sweep_due_workflow_runs`
   (a `MagicMock` or a counting stub) inside an `async with` `TestClient`/`httpx.AsyncClient`
   lifespan context, asserting the stub was called at least once within a short real-time bound
   and that the task is cancelled cleanly on app teardown (no `CancelledError` propagating, no
   "task was destroyed but it is pending" warning). This is the **one** test in this plan that
   necessarily touches real wall-clock time (proving the loop actually ticks) — keep it to a tiny
   interval and a generous timeout so it stays fast and non-flaky; everything that needs precise
   due-time behaviour is proven by test 4's injected clock instead, never by this test.

**Full-suite discipline** (rule 5): after landing, `./scripts/test_queries.sh` must stay green
(no query changes to existing numbered entries, one new §12.16 addition with its own assertions
per test 3 above) and the server's default `pytest -q` baseline count rises by the new tests
above, with zero existing test's behavior changed.

## 7. Documentation updates (implementer's done-condition, per `AGENTS.md`'s doc-curation rule)

- **`QUERIES.md`** — add §12.16 `find_due_wait_candidates`, transcribing §3.4's query verbatim
  plus the implementer's own live `GRAPH.PROFILE` finding (matching every other §12 entry's
  format).
- **`DESIGN.md` §6.1** — the `wait` bullet (`:304-308`) currently states flatly "Real
  timers/scheduled wakeups are backlog K-028, not a gap in this model." Update to describe the
  shipped mechanism: `config.waitForSeconds`/`waitUntil` on a `wait` **or** `human` step (v2
  scope), additive-only, and point at §6.2/§6.3's updated notes below rather than duplicating. The
  `human` bullet just above it (`:292-295`) should gain the same one-line pointer, since the
  mechanism is identical for both step types.
- **`DESIGN.md` §6.2** — no change needed to the `WorkflowRun`/`StepRun` property table (§4: no
  new properties); note the sweep as a new *reader* of `LAST_STEP_RUN`/`AT_STEP` alongside the
  existing consumers listed there, if the implementer judges it's worth a line for future
  readers — optional, not required, since nothing about those edges' write-side contract changed.
- **`DESIGN.md` §6.3** — the "handoff note for K-025" callout (`:485-490`) is now stale in its
  flat "There is no scheduler in this system (decision D-C)" framing; update to note the sweep
  exists but remains additive (a `wait`/`human` step that never advances on its own is still valid,
  specified behaviour for a def that doesn't opt in) — do not delete the note, correct it in
  place, since it is still the right pointer for a reader trying to understand `wait`'s baseline
  semantics.
- **`scripts/start_server.sh`** (v2, closes plan-gate Minor 4) — add
  `FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S` to the script's own header comment, the established
  sibling location for every other workflow env var (`FALKORCHAT_WORKFLOW_ENABLED`,
  `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`, `FALKORCHAT_ENABLE_AGENT`, all documented there per
  `AGENTS.md`'s own "runtime env vars are documented in the script's own header comment" rule for
  this script). The script already `${VAR:-default}`s, `export`s, and echoes each of the
  workflow-related vars at startup (`start_server.sh:88,154,172,182-183`) — mirror that shape for
  the new var too (a one-line header entry is mandatory; a matching startup echo, alongside the
  existing "Workflow: enabled=..." line at `:172`, is a natural addition but not itself required
  for the doc-curation done-condition).
- **`AGENTS.md`** — add the new `FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S` env var to the existing
  `WORKFLOW_ENABLED`/`ENABLE_AGENT` double-flag callout (`AGENTS.md:81-85`) if the implementer
  judges it belongs there (it's read only inside the same gate) — a judgment call, not mandatory,
  since `config.py`'s own header comment already documents every env var per the `start_server.sh`
  convention.
- **`BACKLOG.md`** — flip K-028's `🔵 proposed` marker to delivered/✅ per this repo's own
  milestone-tracking convention (see how K-020/K-021/K-022 etc. are marked in the `Active`
  section), pointing at this plan and the delivering commit/HISTORY.md entry.
- **`HISTORY.md`** — one dated entry per the doc-curation convention every delivered change
  follows.

## 8. Risks & open questions

- **The scheduler as a second source of truth for run state — the backlog's own named top risk.**
  This design does not introduce one: the sweep never writes anything except through the
  unmodified `resume_run_with_ctx` CAS (v3: the sweep now calls this one, not the plain
  `resume_run`, §3.5/§3.6) — no new `WorkflowRun` property, no shadow timer state anywhere (§3.2,
  §3.4). The one thing the sweep *does* now write, the step-scoped `ctx.timerFired` marker, rides
  **inside** that same existing atomic CAS — the exact same "the write and the flip cannot be
  split" property `resume_run_with_ctx` already gives the human-input path (D-F) — not a
  side-channel or a second write. There is no state for the scheduler and the graph to disagree
  about, because the scheduler has none of its own; every sweep tick re-derives dueness fresh from
  graph data that the existing suspend/record path already owns, and the one thing it writes is
  gated by, and lives inside, the same CAS everything else already trusts.
- **v3: does the sweep's `ctx.timerFired` write reopen the `ctx` R-1 residual window
  (`DESIGN.md` §6.2)? Checked explicitly, per `teco`'s own Direction-A framing — no new residual
  class.** R-1 is "two submitters read the same base ctx, merge onto it, and the loser's merge is
  never written — reported as a 409/no-op, not silently lost or corrupted." The sweep is exactly
  one more kind of submitter under that same shape: it reads `ctx` fresh (§3.5 step 5.1), merges
  one reserved key onto it, and races the identical CAS a human `submit_workflow_input` call
  already races. Whichever loses gets nothing written — never a partial merge, never an erased key
  a prior step branched on, because the CAS is atomic and single-winner regardless of how many
  kinds of caller are racing it. The one thing that's structurally *better* here than the general
  R-1 case: because `timerFired` is reserved, a human's own merge and the sweep's own merge can
  **never target the same key** — there is no scenario where a human's legitimate write and the
  sweep's marker write could ever collide on *value*, only on *which one gets to write at all*
  (which the CAS already resolves cleanly). Not a new risk; the same accepted shape, one more
  participant.
- **Closed, v3 (v2's fix for this was itself defective — see the header revision note; not a
  reopening of Finding 1, a correction of its fix): unbounded resume→re-park churn on a
  timer-bearing step, AND the `analyst`-Pass-2-found "not yet" foreclosure the v2 fix caused.**
  Root cause, traced to the actual source: `evaluate_guard`'s unconditional-fires-whenever-reached
  rule (`guards.py:223-224`) cannot distinguish "first arrival," "an ordinary not-yet resume," and
  "a genuine sweep-triggered timeout" — so **any** fix that relies on an *unconditional* fallback
  (v2's approach) necessarily collapses all three into "always fires," which is wrong for the
  first two. v3 replaces the unconditional fallback with a **conditional** one keyed to a
  step-scoped, engine-reserved ctx marker (`ctx.timerFired == "<this step's own key>"`, §3.3) that
  only the sweep can ever set (via `resume_run_with_ctx`, inside the same atomic CAS as the resume
  itself) — so the guard is genuinely false until a real timeout resume happens, true only then,
  and correctly ignores an ordinary or not-yet resume. This closes the original churn risk (a
  sweep-triggered resume is now guaranteed to advance exactly when, and only when, it should) and,
  verified rather than merely hoped, the Pass-2 foreclosure (an ordinary `{"provisioned": false}`
  resume writes no `timerFired`, so the step correctly re-parks) at the same time, with the same
  mechanism — not two separate fixes.
- **New, named residual accepted for v3 (narrower than what it replaces): a def whose transitions
  cycle back to a step *after that step's own timer already fired once* would see its own stale
  `ctx.timerFired` marker and escalate immediately on re-arrival, without re-parking.** Because the
  marker is step-scoped (its value is the firing step's own `key`), it cannot leak across
  *different* steps (§3.3, verified by §6's new two-step regression test) — the residual is
  narrower: only a def that authors a cycle back to the *same already-escalated* step would hit it.
  Not engineered around (would need clearing the marker via a second, non-atomic write outside the
  lock — reopening exactly the race-window problem this whole design exists to avoid, §3.2) —
  documented instead, in the same spirit as the already-accepted `ctx` R-1 residual and Pass 2's
  own accepted "not yet" trade-off precedent: a genuinely narrow, unusual authoring pattern
  (revisiting the exact step you just escalated away from) gets a documented boundary, not
  speculative complexity.
- **Multiple sweep sources racing each other (the periodic task, a manual `POST .../due`, an
  external cron, and — if ever run with multiple uvicorn workers — one sweep loop per worker
  process) is explicitly safe, not merely tolerated.** Every sweep attempt funnels through the
  same `resume_run_with_ctx` CAS a competing human reply already goes through (v3); the CAS being
  single-flight is what makes N racing sweepers cost nothing but wasted read/CAS-attempt work,
  never a double-resume and never two different `timerFired` writes landing. This is a genuine,
  load-bearing property of reusing the existing CAS rather than inventing a scheduler-specific
  lock or leader-election scheme — called out explicitly because a reviewer's first instinct on
  "an in-process loop per worker process" is usually "you need a distributed lock," and here you
  provably don't.
- **Settled, v2 (was an open fork in v1 — plan-gate Finding 2): scope covers both `wait` and
  `human` steps, stakeholder-decided.** v1 scoped this to `wait` only, citing a framing
  (`"'wait' steps today are mechanically identical to 'human'... K-028 needs..."`) that the
  plan-gate review traced and found does not actually appear in the K-028 backlog text — the
  backlog's own single motivating example (*"if no approval in 48h, escalate"*) is a `human`-typed
  approval scenario, not a `wait`-typed signal scenario. The stakeholder confirmed the broader
  scope; §1/§3.3/§3.5/§6 above are all written against `WAITING_STEP_TYPES` (both step types), not
  `wait` alone. No longer an open question.
- **Judgment call: sweep granularity (`30s` default interval, `200`/`1000` limit defaults).** The
  backlog states no SLA precision requirement; a lab-scale escalation timer measured in hours
  does not need sub-minute precision, but this is a guess, not a derived number. Both are
  environment-overridable (`FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S`, the endpoint's `limit` body
  field) — cheap to retune post-ship without a design change.
- **The O(waiting-run-count) sweep cost (§3.2's stated trade-off) could become real at a scale this
  plan doesn't anticipate.** Flagged, not built around: if a deployment's `waiting`-run population
  ever regularly exceeds a sweep's `limit`, or the app-side per-candidate JSON-parse cost becomes
  measurable, the documented escape hatch is a `WorkflowRun.wakeAt`-style property + index (the
  backlog's own original sketch), which becomes newly justified by real numbers instead of
  anticipated ones — deliberately not built speculatively now, per "prefer the simplest design
  that fully solves the problem."
- **Single-workspace sweep loop (§1's stated out-of-scope item) is a real limitation, not a
  simplification of convenience, if this deployment ever becomes multi-tenant.** `CallContext`'s
  single-hardcoded-tenant resolution is a pre-existing, documented boundary
  (`DESIGN.md` §14.3) this plan does not cross — flagged so a future multi-tenant change knows the
  periodic loop needs to become workspace-aware (iterate every `ws:*` graph, or accept a list) at
  the same time auth does.
- **Two-flag gate reuse (`WORKFLOW_ENABLED` + `ENABLE_AGENT`) inherits the existing, already-
  documented gotcha** (`AGENTS.md`: "`FALKORCHAT_WORKFLOW_ENABLED=1` alone is not enough") — the
  sweep loop only starts when both are set, same as the executor it depends on. Not a new risk,
  restated so the implementer doesn't rediscover it while wiring `_build_default_app`.
- **`anyio` vs `starlette.concurrency.run_in_threadpool` for the periodic task's blocking-call
  offload (§5 step 8)** is left as the implementer's call — both are correct per the
  python-web-quirks finding in §2; the plan does not have a strong opinion on which import path is
  more idiomatic for this codebase's existing dependency declarations, and it's a one-line choice
  either way.

## Ready to implement

Plan: `falkor-chat/docs/plans/workflow-timers.md` (this document, v3 — revised in place after
`teco`-routed implementation feedback found v2's Finding-1 fix functionally defective; see the
header revision note and `docs/reviews/workflow-timers.md` Pass 1 + Pass 2 for the full paper
trail).

**Digest**: a `wait` **or** `human` step (scope stakeholder-decided in v2, unchanged) gains two
optional, additive-only `config` keys (`waitForSeconds`/`waitUntil`); when either is declared, the
step must also declare an outgoing transition guarded on a **step-scoped, engine-reserved ctx
marker** — `{"kind":"cmp","path":"ctx.timerFired","op":"eq","value":"<own step key>"}` (v3,
replacing v2's defective "unconditional fallback" invariant, which — traced to `guards.py:223-224`
and `executor._drive_loop`'s uniform per-arrival guard evaluation — made the step never park at
all). `Services.sweep_due_workflow_runs` still derives dueness fresh on every call from data the
engine already writes (`StepRun.startedAt` + `Step.config`) — unchanged, no new `WorkflowRun`
property, no new index, no edit inside the SHA-locked `_drive_loop`. It resumes a due run through
the **existing** `executor.resume(..., run_ctx_json=...)` → `resume_run_with_ctx` CAS (§12.13) —
the same primitive `submit_workflow_input`'s human path already uses, writing the step-scoped
`timerFired` marker atomically inside the same CAS the resume itself uses (zero new resume
semantics, zero new residual-window class beyond the already-accepted `ctx` R-1 shape, §8) —
reusing `_drive_or_fault` for per-run fault isolation across a batch. Exposed as
`POST /workflow-runs/due` and ticked automatically by an in-process `asyncio` task (gated on
`WORKFLOW_ENABLED`), started/stopped at FastAPI lifespan — unchanged from v2. Single implementer
unit — no `graph-dba` DDL split, since no schema changes at all (still true: v3 adds one reserved
ctx key and one additive query-projection column, neither is a schema change). Test strategy
leans on the existing injected-clock (`Services(clock=...)`, one shared clock for `Services` and
`WorkflowExecutor`, v2 Finding 3) and sequential-call CAS-contention (`test_repository.py`'s own
precedent) patterns already living in this codebase; v3 adds two new regression tests earned by
this revision itself — the "not yet" pattern surviving on a timer-bearing step (resolving Pass
2's finding, verified not just documented) and a two-timer-step-in-sequence test proving the
marker's step-scoping actually stops cross-step leakage (the bug this plan caught in its own v3
design before shipping it).

All three of v3's own design choices are individually justified above, not just asserted: why
Direction A over Direction B (header note, §3.3), why the marker must be step-scoped rather than a
bare boolean (§3.3, closes a leakage bug found while designing the fix itself), and why the ctx
write doesn't reopen the R-1 residual window (§8, `teco`'s own explicit ask, checked and answered
"no, same accepted shape, one more participant"). The one narrower residual v3 does accept — a def
cycling back to the *same* already-escalated step — is named plainly in §8, not left implicit. The
sweep granularity defaults (30s interval, 200/1000 limits, §8) remain the sole open, low-stakes
judgment call from earlier passes — environment-overridable, not gating.
