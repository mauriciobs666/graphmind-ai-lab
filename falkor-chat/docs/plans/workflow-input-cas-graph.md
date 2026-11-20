# Workflow input CAS — `ctxVersion` on the resume (graph design)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** K-097 · **Reviews:** `falkor-chat/docs/reviews/workflow-input-cas.md`

Change against the live spec: DESIGN §6.2 ("Who writes the run `ctx`", "Residual window (R-1)",
"Timer release…", the `WorkflowRun` property list) and QUERIES §12 (model block, status-move
contract), §12.13, plus one term each in §12.4, §12.5 (`fail_run`) and §12.7. Engine
`falkordb:v4.18.11` (module `41811`), client `falkordb-py` 1.6.x. Outside the locked `_drive_loop`.

## 1. The ABA question — reproduced

**Verdict: the ABA holds on the shipped code, and it is worse than R-1.** A status-only CAS
matches `waiting` again after the winner's drive re-parks the run. The slower submitter's merge,
built on the pre-resume `ctx`, is written over the winner's. The submitter gets **200**, not 409,
and the winner's input is silently erased, even when a guard already branched on it.

**Evidence** (executed, not argued), in `falkor-chat/docs/reviews/workflow-input-cas-evidence/`:
- `aba_harness.py` (the review's copy differs only in its graph key).
- `aba_output_graph-dba.txt`, the original run on `ws:probe-k097` (deleted afterwards), and
  `aba_output.txt`, the analyst's re-run, which matches line for line.
- **The "follow-up probes" block ending `aba_output_graph-dba.txt` is manual** (`redis-cli`),
  not harness output. Its PROFILE claim was re-checked (`followup_probe.py`,
  `followup_probe_output.txt`); its RAM figure was not.

The harness drives the real `Services.submit_workflow_input` → `WorkflowExecutor.resume` →
`Repository.resume_run_with_ctx`. After B's read and merge, a proxy on the executor's repo runs
submitter A end to end just before B's CAS (the `_RacingRepo` technique of
`tests/test_workflow_timers.py`).

| Case | A's drive ends | B's response | Final run |
|---|---|---|---|
| A1, R-1 baseline | `done` | 409, nothing written | `ctx={"decision":"approve"}` |
| A2, re-park at another step (`approval`→`provision`) | `waiting` at `provision` | **200**, `ctx={"decision":"reject"}` | trail `approval, approval, provision, provision`; `ctx.decision = "reject"` |
| A3, re-park at the same step (bare `human`) | `waiting` at `note` | **200** | A's accepted key `comment` is gone from `ctx` |

A2 is the bad one. The trail records `approval → provision` fired on
`ctx.decision == 'approve'`, but the run's `ctx` now says `reject`. That breaks the property
DESIGN §6.2 claims ("which input advanced the run" and "which input is in `ctx`" can never
disagree). B's input was also validated against `approval`'s `expects` but applied at
`provision`, so per-step validation (D-H) was checked against a step the run had already left.
B's drive used a step of the budget (`stepCount` 3 → 4).

Two facts from the code shape the window:
- **The drive writes no `ctx`.** In `executor.py` only `fail_run` (terminal) writes it, so the
  ABA drops the **winner's input**, not drive output.
- **Every exit from `waiting` goes through one of two CASes**: `resume_run` (chat/trigger, no
  ctx) or `resume_run_with_ctx` (`/input` and the timer sweep); §2's table covers every
  `ctx`/`status` writer (grep, CPG-checked in the review). All three resumers can open the
  window for a pending `/input` or sweep CAS. The sweep's `atStepKey` re-check (step 5.1) narrows
  its own window but does not close it, because it is still a read outside the CAS.

## 2. The CAS shape

**Invariant.** `ctxVersion` is an integer on `WorkflowRun`, read everywhere as
`coalesce(r.ctxVersion, 0)`. A run is created without the property and reads as 0. Every query
that writes `ctx` **of an existing run**, or moves it out of `waiting`, bumps it. The service
refuses a read whose status is not `waiting`, so a read of
`(status='waiting', ctxVersion=v)` names one wait instance. A CAS carrying `v` succeeds only if
no resume, and no ctx write, has happened since that read. One compare closes R-1, the ABA, a
chat resume in the gap, and the sweep's read-to-CAS gap.

**The versioned resume (§12.13).** Verified live on `ws:probe-k097`:

```cypher
// $runId; $ctx = the FULL merged run ctx (opaque string, rule 8);
// $ctxVersion = the ctxVersion the service read together with the ctx it merged onto
MATCH (r:WorkflowRun {runId: $runId})
WHERE r.status = 'waiting' AND coalesce(r.ctxVersion, 0) = $ctxVersion
SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx,
    r.ctxVersion = $ctxVersion + 1
RETURN r.runId AS runId, r.status AS status
```

The return columns stay `{runId, status}`, the shape `_status_row` reads.

Probe results (harness Part B):
- **B1, the ABA on a legacy node (no `ctxVersion`):** B reads `v=0`. A's CAS(`v=0`) wins
  (version 1), and A's drive re-parks the run. B's CAS(`v=0`) returns zero rows,
  `properties_set=0`, and A's `ctx` survives.
- **B2:** a chat `resume_run` (with the bump) and a re-park in B's gap also refuse B.
- **B3:** a CAS with the fresh version succeeds.

**Every other writer:**

| Writer | `ctxVersion` | Why |
|---|---|---|
| `start_run` §12.1, `start_run_untriggered` §12.12 | untouched | No read can predate a run's creation, and a new run reads as 0 through the permanent `coalesce`. A `ctxVersion: 0` term would be indistinguishable from its absence. |
| `resume_run` §12.4 (chat/trigger) | `SET r.ctxVersion = coalesce(r.ctxVersion, 0) + 1`. It compares nothing. | It writes no ctx, but it ends a wait instance. Without the bump, a chat reply that re-parks reopens the ABA for a pending `/input` or sweep CAS (B2). There is no ctx read to compare. |
| `resume_run_with_ctx` §12.13 | compare and bump (above) | The fix. |
| `fail_run` §12.5 | `SET … r.ctxVersion = coalesce(r.ctxVersion, 0) + 1` | It writes `ctx`. Not load-bearing (terminal runs fail the `status` guard). It keeps "every ctx write to an existing run bumps" true for any future version-only reader. |
| `suspend_run` §12.3, `complete_run` §12.5, `record_step_and_advance` §12.2 | untouched | No ctx write. A re-park, or an `AT_STEP` move, is always preceded by a resume, which already bumped. That also pins a waiting reader's `atStepKey`, so D-H validation holds. |
| `get_run` §12.7 (RO) | projects `coalesce(r.ctxVersion, 0) AS ctxVersion` | The service reads the version together with the ctx, in one query. |

**Default for new and existing runs: `coalesce(r.ctxVersion, 0)`, no backfill.** A read-only
count on 2026-11-20 found 21 runs in `ws:acme` (7 waiting) and 2,195 in `ws:demo` (1,378
waiting), none carrying `ctxVersion`. `coalesce` heals itself (the first resume writes 1), needs
no migration across every `ws:*` and no deploy-ordering rule, and covers absent and null alike
(the same state on this build). A backfill would buy only the removal of `coalesce`, which
never-resumed legacy terminal runs would keep needing anyway. It is permanent in §12.4, §12.5,
§12.7 and §12.13.

**Plan (`GRAPH.PROFILE`, 500 padding runs, half of them `waiting`):**

```
# winner (v matches)                      # loser (stale v)                        # run not waiting, v matches
Results            | Records produced: 1  Results            | 0                   Results            | 0
 Project           | 1                     Project           | 0                    Project           | 0
  Update           | 1                      Update           | 0                     Update           | 0
   Filter          | 1                       Filter          | 0                      Filter          | 0
    Node By Index Scan | (r:WorkflowRun) | 1  Node By Index Scan | 1                    Node By Index Scan | 0
```

The scan anchors on `runId` (1 record among 501) with `status` folded in. The compare is a
residual `Filter` over that record; zero rows there means `Update` writes nothing.

**Index / RAM (rule 6).** No index: the property is only compared after a `runId` point lookup.
RAM is one integer attribute per `WorkflowRun`. A manual measurement put it at about 63–72 B per
run (50,000 scratch runs, instance-wide `INFO used_memory` delta, so noisy). That is about
150 KB at `ws:demo`'s 2,195 runs. No DDL change.

**Public addition: `GET /workflow-runs/{runId}` gains `ctxVersion`.** The route is a 1:1 RO
pass-through of §12.7's dict, so the new column surfaces there. It is kept deliberately (review
ruling Q4):
- it is a bare counter that reveals nothing sensitive;
- no reader breaks on an added key;
- it is the per-wait-instance identifier K-038 m7 needs.

The close names it in `HISTORY.md` as a public contract addition. A client-supplied
`If-Match`/`expectedCtxVersion` on `POST …/input`, which would protect a human working from a
stale view, is a separate contract change outside K-097.

## 3. Implementer-bound steps

**`server/falkorchat/repository.py`:**
- `resume_run_with_ctx(ws, *, run_id, ctx, ctx_version: int)`: the §2 query. `ctx_version` is
  keyword-only and required, so no caller can skip the compare.
- `resume_run`, `fail_run`: add the `coalesce(...) + 1` bump term.
- `get_run`: project `coalesce(r.ctxVersion, 0) AS ctxVersion` and add the dict key.
- The start queries are unchanged.

**`server/falkorchat/executor.py` — `WorkflowExecutor.resume(ctx, *, run_id, run_ctx_json=None,
ctx_version=None)`:** when `run_ctx_json` is given, `ctx_version is None` raises `ValueError`.
Test for `None`, not for falsiness, because 0 is the normal first value. The chat path is
unchanged apart from the repo-side bump.

**`server/falkorchat/services.py`:**
- `submit_workflow_input`: pass `ctx_version=run["ctxVersion"]`, taken from the same `get_run`
  whose `ctx` it merged onto.
- `sweep_due_workflow_runs`: pass `ctx_version=run["ctxVersion"]` from its step 5.1 fresh read.
  Keep the `atStepKey` check.
- In both docstrings, the zero-row cause becomes "no longer the wait instance that was read".

**Refused-resume contract.** A CAS miss (status moved, or version moved) makes the repo return
`None`, and `executor.resume` returns `None` without driving. `submit_workflow_input` raises
`WorkflowRunNotWaitingError` → **409** (the existing "resumed concurrently — the input was not
applied; re-read the run and retry"). The sweep buckets the run as `raced`. Nothing is written:
no flip, no ctx, no version bump, no `StepRun`. A version miss and a status miss are not
distinguished, because both mean "re-read".

**Fakes and proxies.** `tests/test_services.py`'s `FakeRepo` has no `resume_run_with_ctx`, and
that file never calls `submit_workflow_input` or the sweep. Every `/input` and sweep test runs
on the real `wf_repo`, so the new `get_run` key arrives without fake changes. No test compares
a whole `get_run` dict by equality. Existing `_RacingRepo`s pass through by `__getattr__`. Any
**new** injection proxy that overrides `resume_run_with_ctx` must accept and forward
`ctx_version` (the harness's `RacingRepo` would not).

**Tests** (offline, real `Repository` over `ws:test`, as in `tests/test_process_input.py`; the
injection is a proxy on `executor._repo`):
1. **ABA, re-park at the same step** (harness A3, 200 today). B reads and merges. Before B's CAS,
   A runs to a re-park at the same step. Assert B gets 409, `ctx` equals A's merge, and
   B leaves no `StepRun` and no `stepCount` change.
2. **ABA, re-park at a different step** (harness A2, 200 today). The `approval → provision`
   def. Assert B gets 409, `ctx.decision == "approve"`, `atStepKey == "provision"`, and the
   trail is A's.
3. **Chat resume in the gap.** The injected actor calls `repo.resume_run`, then
   `repo.suspend_run`. Assert B gets 409 (B2).
4. **Sweep in its read-to-CAS gap.** The injection fires on the executor-repo CAS, after step
   5.1's read. A human `submit_workflow_input` re-parks the run at the same step. Assert `raced`,
   and that `timerFired` is not in `ctx`. The existing `test_sweep_cas_contention_…` injects only
   before the read.
5. **Sweep, no race.** A normally started `wait` run, parked and due (no `ctxVersion` yet),
   lands in `resumed`.
6. **Version moves only where it should** (raw `RETURN r.ctxVersion`, no `coalesce`). A normally
   started run carries no property, so it is also the legacy case:
   - after start, absent;
   - the first `submit_workflow_input` returns 200, and the version is 1;
   - each later resume (input, chat, sweep), +1, and a park leaves it unchanged;
   - after a re-park, a stale `ctx_version=0` sent through the proxy gets 409;
   - a drive fault (`fail_run`), +1.

**Mutation gate.** Each mutant must turn the named test red:

| Mutant | Red |
|---|---|
| drop the `coalesce(r.ctxVersion, 0) = $ctxVersion` compare | 1, 2, 3, 4, 6 (stale 409) |
| drop `r.ctxVersion = $ctxVersion + 1` from §12.13 | 1, 2, 4, 6 |
| drop the bump from `resume_run` | 3, 6 |
| drop the bump from `fail_run` | 6 |
| compare without `coalesce` (`r.ctxVersion = $ctxVersion`) | 1–6 (every first CAS refused) |
| `get_run` projects `r.ctxVersion` without `coalesce` | 1–6 (every first resume raises) |

**`scripts/test_queries.sh`** (the §12.12/§12.13 block):
- Rewrite `RESUME_CTX` to the §2 query with `$ctxVersion`.
- Add these cases:
  - on a `START_UNTRIG` node (no `ctxVersion`), a CAS with `ctxVersion=0` flips and writes
    `ctx`, and the raw `r.ctxVersion` is 1;
  - the ABA replay: re-park via §12.3, then a stale `ctxVersion=0` returns zero rows, a
    `LOSER` ctx is not written, and the version stays 1;
  - the PROFILE still passes `assert_index_scan`.
- Add one bump assertion each to the §12.4 (`resume_run`) and §12.5 (`fail_run`) cases.

**Deploy (operator note, no code mitigation).** While old and new code share the instance, the
mixed window falls back to today's status-only behaviour in both directions. An old-code
winner doesn't bump, and an old-code submitter doesn't compare. That is never worse than today,
and nothing persists past the restart. Restart every server process together:
- the `:8200` agent-team server (MCP in-process);
- the `start_demo.sh` storefront uvicorn (its chat path calls `resume_run` via `trigger.py`);
- any `start_server.sh` uvicorn.

## 4. Spec sentences that change at close (not edited here)

**DESIGN §6.2:**
- The `WorkflowRun` property list (`ctx, trace, maxSteps, stepCount, waitingThreadId`) gains
  `ctxVersion`. The run-model notes gain one bullet stating the §2 invariant as scoped there:
  a run is created without the property and reads as 0; every ctx write to an existing run, and
  every exit from `waiting`, bumps it.
- "Who writes the run `ctx`":
  - "`resume_run_with_ctx` (QUERIES §12.13) is `resume_run` (QUERIES §12.4) plus one `SET`
    term, one query" becomes a compare and a bump, and §12.4 itself gains a bump.
  - "Folded, only the CAS **winner's** ctx is ever written: 'which input advanced the run' and
    'which input is in `ctx`' can never disagree" is **false today** (A2). It holds only with
    the version compare, and the sentence should say so.
- "Residual window, stated plainly (R-1)": the whole blockquote, including "not a wrong branch
  and not an erased key a prior step branched on" (contradicted by A2) and "the fix … is
  unbuilt". It becomes the closed statement: a stale read is refused (409), nothing written.
- "Timer release…": "It inherits R-1's exact shape … and adds no new residual class" becomes:
  the sweep passes the version it read.
- The design-rule blockquote: "closed generically by the ctx-write-inside-CAS discipline above"
  becomes the discipline **plus** the version compare.

**QUERIES §12:**
- The §12 model block, which repeats `WorkflowRun`'s property list, gains `ctxVersion`.
- **Status-move contract:** "zero rows = the anchor missed (run gone / wrong current state)"
  gains a third cause, a stale `ctxVersion` (§12.13).
- **§12.13:**
  - The opening "Parent: **§12.4** … which **remains in use unchanged**" and "§12.4 **plus one
    `SET` term**": §12.4 gains a bump, and §12.13 adds a compare and a bump.
  - The Cypher block and its `$ctx` comment gain `$ctxVersion`.
  - The zero-row contract: "a run that is not `waiting` matches the node but fails the `WHERE`"
    becomes "not `waiting`, or not at the read `ctxVersion`".
  - "'which input advanced the run' and 'which input is in `ctx`' can never disagree" gets the
    same correction as DESIGN.
  - "the `status` predicate is folded **into** the index scan (no residual `Filter` operator)":
    `status` still folds, but the compare is now a residual `Filter`. Replace the plan block
    with §2's.
  - "**Indexes (rule 6).** … The RAM cost is the merged `ctx` string" gains the one-integer cost.
- **One term each:** §12.4 (the bump), §12.5 `fail_run` (the bump), and §12.7 (the projected
  column). §12.7 is also the `GET /workflow-runs/{runId}` shape SERVER.md's route row cites, so
  SERVER.md itself needs no edit.
