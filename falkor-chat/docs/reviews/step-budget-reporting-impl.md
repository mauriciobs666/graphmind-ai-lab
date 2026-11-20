# Step-budget reporting (K-096) — implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-096

## Scope & verdict

**Reviewed:** the uncommitted diff of `falkor-chat/server/falkorchat/services.py`,
`server/tests/test_process_input.py` and `server/tests/test_workflow_timers.py`. The baseline
is `HEAD` (`78b5e647`). It was judged against `falkor-chat/docs/BACKLOG.md` `### K-096`, the
stakeholder decision of 2026-11-20 (report the failure, and budget is still not a fault),
SERVER.md §1.4/§1.7, and the approved K-097 design `falkor-chat/docs/plans/workflow-input-cas-graph.md`.
I applied root `REVIEW.md` and `falkor-chat/REVIEW.md`. I did not review the other modified
files in the tree.

**Verdict: approve with suggestions.** There are no blockers and no majors. The code meets the
backlog done-condition, and every premise I could check by tracing or running held, except
claim 5's "behaviour-equivalent" (F1). The three findings below are minor: two missing pins
and one claim that is only half true.

CPG: considered, not relevant — a three-file Python diff whose call sites (`_drive_or_fault`,
`get_run`, `fail_run` ctx writers, `start_workflow_run` consumers) were enumerated
exhaustively by grep and read in full. A call graph would add nothing to that.

**Checks run** (pytest, `server/` as cwd, `ws:test` + `reference-test`, so the live
`reference` was never touched and no re-seed is needed):
- `test_process_input.py test_workflow_timers.py test_api.py`: **208 passed**.
- `test_trigger.py test_storefront.py test_storefront_api.py test_executor_process.py
  test_executor.py test_services.py`: **671 passed**.
- Not run: the rest of the suite, `test_queries.sh` (no query changed), `-m live`.
- Mutation battery: services.py was copied aside, each mutant applied, and the file restored
  by copy. Its sha256 `26670d1d…` was re-verified after every run. Results are in the
  Appendix. Two temporary probe tests were run and then deleted, and the tree status was
  confirmed unchanged afterwards.

**Rulings on the implementer's claims**

1. **Holds (verified by mutant).** The read-back lives in `_drive_and_report`
   (`services.py:2644-2671`). The sweep still calls `_drive_or_fault` directly
   (`services.py:2903`). Routing the sweep through the wrapper turns
   `test_sweep_buckets_a_release_that_exhausts_the_budget_as_resumed_not_faulted` red.
2. **Holds (traced).** In `executor.py`, `"failed"` is returned literally in exactly one place,
   `_fail_budget` (`executor.py:1495-1501`). `_drive_loop`'s other returns are `"waiting"`,
   `"done"` and `run["status"]` (`:712-713`), and `_drive`'s returns are `_drive_loop` or
   `"waiting"` (`:686-693`). `run["status"]` would be `"failed"` only if the run were already
   failed with `AT_STEP` cleared when `get_run` reads it, and neither route can reach that
   state:
   - `run()` reads a run that `start_run*` has just created as `running` with an `AT_STEP`.
     No other actor writes a `running` run.
   - `resume()` reads right after its own `waiting→running` CAS. `suspend_run` keeps
     `AT_STEP` (`repository.py` §12.3 query). If `AT_STEP` were somehow absent, the status
     returned would be `running`, not `failed`.
3. **Holds (verified by mutant).** The `and error is None` guard keeps a caught fault's
   `"<Type>: <message>"`. Dropping the guard turns
   `test_a_caught_fault_keeps_its_exception_as_the_error_not_the_ctx_note` red. A clean
   `done`/`waiting` drive is unchanged: `test_the_clean_path_envelope_ctx_still_equals_the_persisted_merge`
   gained an assertion that `error` is absent.
4. **Half true (F3).** An unfiltered grep found these consumers:
   - `api.py:495/504`, the two routes.
   - `trigger.py:82`, the @mention start.
   - `storefront.py:1406`'s `failed` branch, through `maybe_trigger`.
   - `docs/reviews/workflow-input-cas-evidence/aba_harness.py`, which reads only `status`.
   - `web/app.js:559`, which ignores the `/input` body. No MCP tool wraps either route.

   No consumer breaks, since every one either gains a field or ignores it. However, the
   trigger's other branch, the resume (`trigger.py:78` → `resume_workflow_run`), still answers
   a bare `{runId, status}`.
5. **Not equivalent (F1, verified by probe).** M6 widens the read-back to `waiting`. A
   `waiting` run is open to concurrent writers the moment the drive returns, so the widened
   read can report another actor's ctx. Your hardcoded-error survivor is ruled on in F2.

## Findings

### F1 — minor — The `status == "failed"` restriction is load-bearing for concurrency, not just a saved read, and nothing pins it

**Evidence (verified by probe).** M6 (`if status is not None and error is None:`) passes all
208 tests. I added a temporary test with a proxy on `Services._repo` that does this on its 2nd
`get_run`, which is the read-back: a concurrent actor runs
`resume_run_with_ctx(ctx={"intruder": true})` + `suspend_run` before the read is delegated.
The submission is `{"decision": "reject"}`, which re-parks the run as `waiting`.
- **Shipped code:** the envelope is `{"decision": "reject"}`, and `get_run` is called once.
  The probe passes.
- **M6:** the envelope is `{'status': 'waiting', 'ctx': {'intruder': True}}`. The probe fails.

**Why it matters.** `failed` is terminal. The only writers of an existing run's `ctx` are
`resume_run_with_ctx` and `fail_run` (`repository.py:3040`, `:3077`). The first needs
`waiting`, and the second runs only inside a drive. So a read-back after `failed` is stable,
but a read-back after `waiting` is not. The docstring explains the restriction as cost ("costs
no read"). K-097 edits exactly this call next, and a later "simplify: always read back" would
bring in the race with the suite still green.

**Suggested improvement.** Add the probe as a permanent test in `test_process_input.py`. Use
the approval shape, submit `reject`, inject a concurrent resume and re-park on the second
`services._repo.get_run`, and assert that `out["ctx"]` equals the submitted merge. In the
`_drive_and_report` docstring, give the reason: `failed` is terminal and so stable to read,
while `waiting` is already open to the next writer. This is a test for an invariant of
already-shipped code, not a guard written in the same step it guards, so it can turn red now
(M6 is its mutant).

### F2 — minor — The survivor "`error = "step budget exceeded"`": pin the provenance, since the spec will state it

**Ruling.** Today it is equivalent: by claim 2, `_fail_budget` is the only non-raising
`failed` producer, and its note is that literal. The existing assertion
`json.loads(run["ctx"])["error"] == out["error"]` would catch the hardcode only if the
`_fail_budget` literal changed at the same moment. The contract being shipped is "error is the
graph's own `ctx.error` note" (the new `start_workflow_run` docstring, and the SERVER.md text
the close will write). That contract matters exactly when a second non-raising `failed` path
appears, such as a deadline or a different budget. The read-back would report that path
correctly, and the hardcode would misreport it. A spec claim that no test can tell apart from
the hardcode fails root `REVIEW.md`'s "can both fail and pass".

**Evidence (verified).** In a temporary test, I monkeypatched `WorkflowExecutor._fail_budget`
to stamp `"probe-note-k096"` through `_fail_with_note` and return `"failed"`, then asserted
that `start_workflow_run(...)["error"] == "probe-note-k096"`. It passes on the shipped code
and fails on the hardcode mutant.

**Suggested improvement.** Add that test, which is about 10 lines, next to
`test_budget_exhaustion_reaches_the_same_envelope_without_raising`. The expected value is a
test-local literal and is never read from `executor.py`.

### F3 — minor — Claim 4 covers only the @mention-start half of the trigger path; a budget failure on a resume still reports no reason

**Evidence (traced).** `trigger.py:78` sends a reply in a thread with a parked run to
`resume_workflow_run` (`services.py:2743-2754`). That method calls `executor.resume` directly
and returns `{"runId", "status"}`, with no `_drive_and_report` and no `error`. The salesperson
storefront is chat-driven: every turn after the first is a resume. So
`storefront.py:1407-1412`'s DEF-3 log line (`error=%s`) still logs `error=None` for a budget
failure on any later turn. It gets the reason only when the failure happens on the first,
start turn.

**Why it matters.** This is not a K-096 defect, because the done-condition names the two REST
routes. But the delivery claim, and any `HISTORY.md` line that says "the trigger path now logs
the reason", would overstate it.

**Suggested improvement.** Word the `HISTORY.md` entry as "the @mention-start path". Separately,
either open a backlog item to route `resume_workflow_run` through `_drive_and_report`, or record
an explicit no-change ruling. Routing it is cheap. Its envelope has no `ctx`, so only `error`
is gained. The resume route also lets real faults propagate (`executor.py:656`), so it would
call `_drive_and_report`'s read-back alone, not the fault catch. That design choice belongs to
the implementer and needs its own pin.

## What's solid

- The decision is kept exactly: nothing raises, the locked `_drive_loop` and `executor.py`
  are untouched, and the fix is service-side only, as the backlog's Owner line requires.
- The test design is good:
  - The decoy run in `_decoy_and_budget_parked_run` kills a mis-aimed read-back (M_wrongid
    turns red).
  - The `/input` test asserts both graph equality and the literal merge plus the note, so an
    expected value is never only read back from the code under test.
  - The pinning test was rewritten, not deleted.
- The sweep regression test documents why the wrapper is not folded into `_drive_or_fault`,
  and it is the test that catches that refactor.
- `error` is in `RESERVED_CTX_KEYS` (`services.py:115`), so a caller cannot plant a fake
  reason that the read-back would then report.
- **K-097 fit (read against plan §2–§3).** The read-back uses only `run["ctx"]`, so K-097's new
  `get_run` key `ctxVersion` cannot leak into either envelope. A K-097 CAS miss returns `None`,
  which `_drive_and_report` passes through unchanged, so the 409 path is unaffected. K-097's
  `fail_run` bump happens before this read, and the read is RO. The only overlap is textual:
  K-097 adds `ctx_version=` inside the same `executor.resume` lambda and rewords the same
  docstring, which is a trivial merge in either order. F1's test also pins this call against
  K-097's edit.

## SERVER.md sentences the close must change

These are current text that the shipped diff makes wrong. I also swept DESIGN.md, QUERIES.md,
`README.md`, `AGENTS.md` and `manuals/`. DESIGN §6.2 (`:639`) and QUERIES §12.5 (`:1385`) state
no envelope claim and stay true.

1. **§1.4, the "Step-budget exhaustion is not a fault" paragraph (`SERVER.md:450-453`).**
   Wrong are "the route answers through the clean path", "no `error` field", "on `/input` the
   submitted merge as `ctx` rather than the graph's" and "The reason is visible only through
   `GET /workflow-runs/{runId}`". Still true: "not a fault" and "stamped `failed` with a `step
   budget exceeded` note".
2. **§1.4, the "The `/input` failed envelope reports graph truth, whole" paragraph
   (`:441-449`).** It says status **and** ctx "both come from the *same* post-fault `get_run`
   re-read", and that "on the clean path the two are the same value by construction". After
   K-096, a budget `failed` drive also reports the graph's ctx, but its status is the drive's
   return value, not a re-read. "Clean path" no longer means "ctx = merge"; that now holds only
   for a `done`/`waiting` drive. The zombie re-raise sentence stays fault-path-only and is
   still true.
3. **§1.4, the error map (`:431`).** The `201/200 … {"status":"failed","error":"<type>:
   <message>"}` row covers faults only. A budget failure is also a 201/200 with `error: "step
   budget exceeded"` and has no row. Either add a row, or footnote it under "That last row is
   the deliberate one".
4. **§1.7, the bullet "A step-budget failure carries no `error` in its response (§1.4)"
   (`:679-680`).** It is wholly wrong now. Delete it, or invert it. F3's resume caveat belongs
   to the trigger and storefront text, not to this REST bullet.

`:437-439` ("Response bodies … `[, error]`") is still correct.

## Open questions

- F3: is the resume half of the trigger path (`resume_workflow_run`) wanted in a follow-up, or
  is the bare envelope there a deliberate ruling? Whoever owns the K-096 close should record one
  or the other.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. The source is F1 of this review: a widened
read-back condition survived as "behaviour-equivalent" under a serial suite and was not
equivalent under a concurrent writer.

- (addition) A surviving mutant called behaviour-equivalent is checked against concurrency as
  well as the serial suite. If it widens a condition to a state that another actor can write
  (a non-terminal status, an unlocked row), plant a concurrent write in that window before
  accepting the equivalence.

No prune: no test, CI job or hook enforces any line in that section that this addition
overlaps.

## Appendix — mutation battery

Each mutant was run against `test_process_input.py test_workflow_timers.py test_api.py`
(208 tests).

| Mutant | Result | Red test(s) |
|---|---|---|
| start route back to `_drive_or_fault` | 2 failed | `test_budget_exhaustion_reaches_the_same_envelope_without_raising`, `test_rest_start_reports_budget_exhaustion_with_its_reason` |
| `/input` route back to `_drive_or_fault` | 2 failed | `test_input_that_exhausts_the_budget_reports_the_graphs_ctx_and_reason`, `test_rest_input_that_exhausts_the_budget_reports_the_graphs_ctx` |
| sweep through `_drive_and_report` | 1 failed | `test_sweep_buckets_a_release_that_exhausts_the_budget_as_resumed_not_faulted` |
| drop `and error is None` | 1 failed | `test_a_caught_fault_keeps_its_exception_as_the_error_not_the_ctx_note` |
| M6: `status is not None and error is None` | **survives** | killed only by the F1 concurrency probe |
| read back a different run id (the decoy) | 1 failed | `test_input_that_exhausts_…_ctx_and_reason` |
| hardcode `error = "step budget exceeded"` | **survives** | killed only by the F2 monkeypatched-note probe |
| fill `error` but not `read_ctx` | 2 failed | both `/input` budget tests |
| fill `read_ctx` but not `error` | 4 failed | all four budget-envelope tests |

## Close check

**Scope.** This is step 3 of the root `AGENTS.md` live-spec close. I checked the uncommitted
SERVER.md edit against the shipped code at `068bc951`, with every code read done through
`git show 068bc951:<path>`. The diff baseline is `git diff -- falkor-chat/docs/SERVER.md`. The
committed `services.py` differs from the gated diff only in the `_drive_and_report` docstring,
which now states F1's concurrency reason. The commit also adds the F1 and F2 pins:
`test_a_waiting_envelope_is_never_read_back_so_a_concurrent_writer_cannot_leak_in` and
`test_budget_error_is_the_graphs_own_note_not_a_service_literal`. I ran no suites, as
instructed, so I did not re-run those two tests.

**Verdict: approve. The close may proceed.** No contract changed by the shipped diff is missing
or contradicted, and nothing below blocks.

**My four sentences.** All four are resolved:
1. The "not a fault" paragraph is replaced in place.
2. The "graph truth, whole" paragraph now separates the fault path's re-read status from the
   budget path's returned status.
3. The error map gains a budget row, and "That last row" became "The last two rows".
4. The §1.7 bullet is deleted.

I re-read the lines the edit left unchanged. The "Response bodies" line (`[, error]`) and the
zombie re-raise sentence, now scoped to the fault path, are still true. The other SERVER.md
sweep text (`:176`, `:311`, `:363`) states no bucketing that the new text contradicts.

**Claims and absolute words, checked against `068bc951`.**
- **"The only writers of an existing run's `ctx` are `resume_run_with_ctx` … and `fail_run`."**
  Holds (unfiltered grep). Across `falkorchat/` and `scripts/` outside the tests, I grepped for
  `.ctx =`, `SET x +=` / `= $map` and `DETACH DELETE`. The only `ctx` `SET`s are
  `repository.py:3040` (§12.13) and `:3077` (§12.5). The only other `ctx:` writes are the two
  `CREATE`s in `start_run*` (`:2856`, `:2901`), which create runs rather than write to an
  existing one. Each writer has one caller: `executor.py:628` and `:1511`. The § numbers match
  QUERIES `:1364` and `:1547`. The other `ctx` writes found are test-only (`test_queries.sh`,
  `docs/reviews/*-evidence/`).
- **"`faulted` holds only caught drive faults, escaped exceptions and oversized-merge
  rejections."** Holds (read in full). `faulted.append` occurs at three sites: the oversized
  merge (`services.py` about `:2905`), the `except Exception` around the drive (`:2921`) and
  `error is not None` from `_drive_or_fault` (`:2929`). `resumed` gets every other non-`None`
  status (`:2931`). "The periodic tick" exists (`app.py:214-229`).
- **"Read the run back once", "the graph's own `ctx.error`, not a service literal".** Holds;
  `_drive_and_report` makes exactly one `get_run`. "`error` is a reserved key" also holds:
  `RESERVED_CTX_KEYS` (`services.py:115`) is checked on both routes, the sweep merges only
  `timerFired`, and the chat `resume_run` writes no ctx.
- **"Explained only by `GET /workflow-runs/{runId}`".** Holds. `get_run` (`repository.py:3128`)
  is the only read that projects `r.ctx`. `find_runs_for_thread` does not project it (`:3218`),
  and no MCP tool fronts a run.
- **"Trigger's rule 2 (§1.12)".** Matches §1.12's ordered rule, where item 2 is resume.
- **"On any `failed` reply, `ctx` is the graph's"** and **"cannot race".** These are true about
  ctx writers. One residual is not a ctx write and is listed as CC-2.

**History prose and numbering.** The added lines contain no item IDs, dates or
"now/no longer/previously" framing. No heading was added, removed or renumbered.

**DESIGN/QUERIES need no change.** Confirmed by reading the only budget sentences in each:
- DESIGN §6.2 (`:639`, and the `maxSteps` note at `:657-664`) covers terminal semantics.
- QUERIES §12.5 (`:1375`, `:1385`) covers the `fail_run` note.

Neither states anything about the response envelope.

**Non-blocking findings.**
- **CC-1 — minor — the writer set now has two homes.** DESIGN §6.2 "Who writes the run `ctx`"
  (`:596`) owns the ctx-writer design. The new SERVER.md sentence enumerates the full writer
  set with "only", which K-097 (`workflow-input-cas-graph.md` §2) is about to revisit in that
  same DESIGN paragraph. K-097 adds no ctx writer, so the sentence stays true. Even so, a future
  writer would make SERVER's "only" false silently. Suggested fix: keep the reasoning, and cite
  the writer set as "(DESIGN §6.2; QUERIES §12.5/§12.13)" so the K-097 close re-checks it.
  Alternatively, record a no-change ruling.
- **CC-2 — nit — a deleted run degrades the reply (inferred, not run).** The storefront
  reset-mine query `DETACH DELETE`s a participant's runs (`repository.py` about `:4158`). If a
  REST `/input` on such a run is mid-drive at that moment, the read-back gets `None`. The reply
  is then `failed` with the submitted merge and no `error`, an edge case that "On any `failed`
  reply" does not cover. It is reachable only by REST-driving a storefront chat run during a
  reset. No spec change is needed unless the owner wants the edge named.

**Final proposed REVIEW.md line** (root `REVIEW.md`, `## Always check`; source: F1; addition, no
prune):

- A surviving mutant called behaviour-equivalent is checked against concurrency as well as the
  serial suite. If it widens a condition to a state that another actor can write (a
  non-terminal status, an unlocked row), plant a concurrent write in that window before
  accepting the equivalence.
