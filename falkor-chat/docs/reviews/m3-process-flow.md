# Plan review — M3 `kind:'process'` proof flow (K-024, remaining half)

> **Reviewer:** `analyst` · **Date:** 2026-07-19 · **Artifact:** `docs/plans/m3-process-flow.md`
> (architect, proposed) · **Baseline tree:** `4f69a16` (working tree clean apart from the untracked
> coordination ledger).
> **Gate:** pre-implementation plan gate. Static review — nothing executed that mutates state.

## 1. Scope & verdict

**Reviewed:** the plan end to end against the live tree — `server/falkorchat/{executor,guards,
services,repository,api,app,trigger,schemas,config}.py`, `server/tests/*`, `scripts/
{seed_workflows,bootstrap_schema,test_queries}.sh`, `docs/{DESIGN,QUERIES,BACKLOG}.md`,
`docs/plans/m3-process-flow-coordination.md`, and `falkor-chat/AGENTS.md` + root `AGENTS.md`.
Every file:line citation in plan §2 was checked individually.

**Not reviewed / not verifiable statically:** the two new queries' `GRAPH.PROFILE` plans and
zero-row contracts on this engine build (that is exactly what U0's graph-dba gate is for); the
actual post-change pytest/`test_queries.sh` counts; the RAM delta. See §5.

**Verdict: request changes.**

Two blockers, both in the same place: the plan's **F-3 ("the current silent no-op is load-bearing
for existing tests") is right in direction but materially incomplete**, and the new publish-time
invariant retro-invalidates existing test fixtures. Both make U2's stated done-condition ("the
existing 350 stay green") unachievable as written, and both are cheap to fix in the plan — this is
a scoping correction, not a design rethink.

The **central design claim of §3.1 — that park-and-branch works with `_drive_loop` untouched — is
correct.** I traced it against the real loop and every def shape in §4; it holds. The plan is
well-researched, honestly hedged, and its decision framing is largely sound. The blockers below are
about the *blast radius* the plan under-measured, not about the mechanism.

Counts: **2 blocker · 6 major · 10 minor · 5 nit.**

---

## 2. Findings

### Blockers

#### B-1 · F-3 is incomplete — a non-`agent` step type is also load-bearing for the 350 baseline

**Evidence.** Plan §2 F-3 and §3.3 scope the preserved stub to "`agent`, no LLM". But
`server/tests/test_executor.py:345`:

```python
TOOL_STEPS = [
    {"key": "answer", "type": "agent", "config": '{"tools":["post_message"],"maxIterations":4}'},
    {"key": "end", "type": "task", "config": "{}"},   # terminal, non-agent → stub
]
```

`end` is a **`type:'task'`** step that is *executed by the real `_drive_loop`* in
`test_hallucinated_mention_does_not_fail_the_run` (`test_executor.py:352`), which asserts
`status == "done"` and `[s["stepKey"] for s in trail] == ["answer", "end"]`. `task` is not even in
`services.STEP_TYPES` (`services.py:44`) — the test writes the snapshot straight through
`repo.materialize_snapshot`, bypassing publish validation.

Under plan §3.3's row *"unknown type — same `NotImplementedError` path"*, executing `end` raises →
`_drive`'s M-1 net (`executor.py:329`) `fail_run`s **and re-raises** → the test sees an exception out
of `ex.run(...)`, not `"done"`. The Defect-B regression pin dies.

**Why it matters.** U2's done-condition is "the existing 350 stay green", and §7 budgets zero
fixture work. An implementer following F-3 verbatim will break the baseline and then have to make an
unbriefed call about whether editing a *regression pin for a closed defect* is in scope.

**Remedy.** Amend F-3 to name both load-bearing stub cases, and add to U2's scope: change
`test_executor.py:345` `type:"task"` → `type:"agent"` (it is already driven with `llm=None`, so the
preserved F-3 stub covers it and the test's intent — "terminal, non-agent → stub", i.e. *a step the
node loop doesn't run* — is preserved exactly). Then re-state the U2 done-condition as
"350 → N, with the two named fixture edits, no other existing test modified".

#### B-2 · The new publish-time `waitsForHuman` invariant retro-invalidates existing def fixtures

**Evidence.** Plan §3.3: *"a step of type `human` or `wait` **must** declare
`config.waitsForHuman: true`, else `WorkflowDefSpecError`"*. Existing fixtures that violate it:

| Fixture | Declares | Effect |
|---|---|---|
| `tests/test_api.py:418` `DEF_BODY` | `{"key":"start","type":"human","config":"{}"}` | every publish through it becomes **400** |
| `tests/test_services.py:719` `VALID_STEPS` | `{"key":"start","type":"human","config":{"a":1},"start":True}` | `_publish` raises |
| `tests/test_services.py:808,822,836,851,866` | `type:"human"`, no config | raise **before** the condition under test |

Outright failures I can name: `test_api.py` — `test_publish_workflow_def_returns_201_and_counts`
(:428), `test_list_and_get_workflow_def` (:447), `test_get_workflow_def_specific_version` (:458),
and the materialize test at :472; `test_services.py` —
`test_publish_workflow_def_derives_start_and_serializes_config_and_guard` (:738),
`test_publish_workflow_def_returns_repo_result` (:759),
`test_publish_workflow_def_conversation_kind_allowed` (:781).

Worse than the failures: **five `pytest.raises(WorkflowDefSpecError)` tests start passing for the
wrong reason.** `test_publish_workflow_def_duplicate_step_key_raises_nothing_written` (:803),
`..._no_start_step...` (:817), `..._multiple_start_steps...` (:831),
`..._dangling_transition_from...` (:845), `..._dangling_transition_to...` (:860) all assert only the
exception *type*. With the new invariant firing first on their `type:"human"` steps, they become
vacuous — they would still pass if the duplicate-key / start-count / dangling-endpoint checks were
deleted. That is a silent regression-net hole, not a red test.

**Why it matters.** Same as B-1 — the plan's baseline arithmetic (§7, "350 → 395–405") assumes the
350 are untouched. They are not. And the vacuous-pass class is the kind of thing that never gets
noticed because the suite stays green.

**Remedy.** Add to U2's scope, explicitly: (a) add `"waitsForHuman": true` to the `human` steps in
`test_api.py:418` and `test_services.py:719/808/822/836/851/866`; (b) tighten the five
`pytest.raises` tests to assert on the message (`match="duplicate step key"` etc.) so they cannot go
vacuous again; (c) restate the U2 baseline as "350 modified-in-place + N new". Also state where in
`_validate_def_spec` the new check runs relative to the existing ones (recommendation: **last**,
after the key-uniqueness/start-count/endpoint checks, which minimises exactly this masking).

### Majors

#### M-1 · The `ctx` write and the resume CAS should be **one** query — D-B's mechanics were never weighed

**Evidence.** Plan §3.4 steps 4–5 do `repo.set_run_ctx(...)` then `executor.resume(...)`, i.e. two
`GRAPH.QUERY`s; §3.5 then accepts last-write-wins (R-1) and consoles itself that "the resume CAS is
still single-flight". `repository.resume_run` (`repository.py:1169`) is already
`MATCH (r:WorkflowRun {runId}) WHERE r.status='waiting' SET r.status='running', r.waitingThreadId=''
RETURN …`. Adding `, r.ctx = $ctx` to that same `SET` is a one-token change and makes
ctx-write-and-resume atomic.

**Why it matters.** With the split, the two writes can interleave: submitter B can land its `ctx`
between A's `set_run_ctx` and A's `resume_run`, so the drive that runs is A's but the data it reads
is B's — a *silent wrong branch*, not merely a lost input. With them folded, only the CAS winner's
ctx is ever written, so "which input advanced the run" and "which input is in `ctx`" can never
disagree. That is precisely the audit property §3.4's *Rejected* paragraph says is "the entire point
of §6.3". A stale-read lost update on the *merge* still exists (get_run → merge), but it shrinks from
"wrong branch taken" to "an earlier key overwritten", which is what R-1 actually claims.

There is a second, related hazard the plan does not name: with the split, a stale-read submitter can
write a `ctx` that **erases a key an earlier step already branched on** — e.g. overwriting
`{request, decision}` with `{request}` after the run has moved past `approval`. The run's own `ctx`
then no longer explains its own trail.

**Remedy.** Replace §3.4 steps 4–5 with a single `resume_run_with_ctx(ws, run_id=, ctx=)`
(QUERIES §12.12, same gate cost, same query count for U0 — it *replaces* `set_run_ctx` rather than
adding to it), returning the status row: zero rows ⇒ 409. Keep `set_run_ctx` out entirely, or keep
it only if a "write ctx without resuming" use case is actually needed (it isn't in this slice).
Update R-1 to describe the residual (a stale merge read) rather than the current, larger window.
This should be settled **before** U0 — it changes the gate's deliverable.

#### M-2 · The start body's `ctx?` has no reserved-key rule — a caller-set `threadId` hijacks the chat resume path

**Evidence.** §3.4 step 3 rejects `threadId`/`error` for `POST …/input`. The start route
`POST /workflow-runs {defKey, version, ctx?, …}` (§3.4, §6 U3, §7 U3) has **no such rule**. But
`_drive_loop`'s suspend passes `run_ctx.get("threadId", "")` straight into
`suspend_run(thread_id=…)` (`executor.py:379-381`), which denorms `waitingThreadId`
(`repository.py:1151`). And `trigger.py`'s step 2 (`trigger.py:76-79`) resumes *any* waiting run
whose `waitingThreadId` matches a posted message's thread — **before** it even looks at mentions.

So `POST /workflow-runs {"ctx": {"threadId": "<a real thread>"}}` parks an `onboarding` run against
that thread, and the next human chat message in it silently drives the approval process one step,
with no input and no guard data. This is F-5's latent collision made **live and reachable**, in the
opposite direction to the one the plan defends against.

**Remedy.** Apply the same reserved-key rejection (`threadId`, `error`) to the start body's `ctx`,
in `services.start_workflow_run` (not only in the pydantic schema — MCP/service callers bypass it).
State it in §3.4 and add it to U3's test list alongside case 6. Consider additionally having the
untriggered start path stamp `waitingThreadId` from a sentinel the trigger can never match, which
would make F-5 and this finding structurally impossible rather than validation-dependent.

#### M-3 · Synchronous drive + the M-1 net's **re-raise** ⇒ engine faults return HTTP 500, contradicting U3's error map

**Evidence.** `_drive` (`executor.py:329-331`) stamps `fail_run` and then **`raise`**s. §3.4 chose to
drive **synchronously** on the request thread ("deterministically testable"). The existing trigger
path is insulated by `api._safe_run_workflow` (`api.py:71`), which swallows and logs — the new routes
have no equivalent. Consequently:

- an unimplemented step type (§3.3's deliberate `NotImplementedError`),
- a malformed `cmp` guard (`guards.WorkflowConfigError`, §3.2's "loud, named" path),
- the snapshot/run tripwires,

all propagate out of `POST /workflow-runs` and `POST …/input` as unhandled exceptions → **500 with a
traceback**, not the "`503`-free error map (404 / 400 / 409)" U3 promises. Neither
`WorkflowConfigError` nor `NotImplementedError` is registered in `app._register_error_handlers`
(`app.py:63-86`).

Separately: **`WorkflowRunNotFoundError` has no handler either** (`app.py:71-86` maps only
`WorkflowDefSpecError` → 400 and `WorkflowDefNotFoundError` → 404), so §3.4 step 1's "404" does not
exist today. U3 lists only *one* new handler (`WorkflowRunNotWaitingError` → 409).

**Remedy.** Decide and write down the contract: either (a) the endpoints catch the drive fault and
return **200 with `{"status":"failed","error":…}`** — the run *is* correctly terminal in the graph,
so this is honest and keeps the audit story intact — or (b) map it to 500 deliberately with a named
envelope. Either way, U3 must add handlers for `WorkflowRunNotFoundError` (404) **and**
`WorkflowConfigError`, and §7 U3 needs a case for each.

#### M-4 · `cmp` guards are validated only at drive time; validate them at publish, where the plan already validates steps

**Evidence.** §3.2: an unknown `op` or a cap breach raises `WorkflowConfigError` **at evaluation**,
i.e. `fail_run`. Yet §3.3 adds a publish-time invariant for steps, and
`services._validate_def_spec` (`services.py:428`) already receives every `transitions[i]["guard"]`
before serialization (`services.py:504`).

**Why it matters.** The plan's own justification for the step invariant — *"Rejecting it at authoring
time costs ~6 lines and no runtime path"* — applies verbatim, and more strongly, to guards: a typo'd
`op` in a def is discovered when a manager clicks approve, killing a live run, rather than when the
def is published. This is also the difference between a 400 at seed time and a mysterious `failed`
run at K-025.

**Remedy.** Extend the U2 publish invariant to structurally validate any `kind:'cmp'|'all'|'any'|
'not'` guard (whitelisted op, `path` root in `{ctx., output.}`, depth/node/width caps, `not` arity)
by calling a `guards.validate_cmp(spec)` helper authored in U1 — one validator, two call sites, no
duplication. Note the one fixture consequence: `tests/test_services.py:721`'s
`{"guard": {"expr": "x>0"}}` has no `kind` and must stay accepted (validate only when
`kind` ∈ the cmp family).

#### M-5 · Park/re-park on invalid human input burns the run budget irreversibly — ~4 typos kill an `onboarding` run

**Evidence.** §4.2 note: *"a garbage `decision` value leaves the approval parked, which is the
correct, unblockable behaviour"*. It is unblockable, but not free: every park is an advance-to-self
`_record` that bumps `stepCount` (`executor.py:363`, `repository.py:1106`), `maxSteps` defaults to
`DEFAULT_STEP_BUDGET = 12` (`executor.py:58`), and §4.3's happy path already costs 8. That leaves
**four** mistaken submissions before the next advance trips `rec["stepCount"] > max_steps` →
`_fail_budget` → `failed`, with no restart path. `stepCount` never decreases.

The plan sees the shape of this (F-4, R-2) but frames it as "a *longer* process would silently hit
the budget"; the live failure mode is a *short* process plus ordinary human error. R-2's mitigation
(exposing `maxSteps` on the start endpoint) puts the lever in the hands of the caller, not the def
author, and `onboarding@v1` will be started with the default by anything that doesn't pass one.

**Remedy.** Pick one and state it: (a) give the untriggered start path a higher default budget for
`kind:'process'` runs (cheap, no engine change); or (b) validate the submitted input against the
parked step's declared `config.fields` / an expected value set **before** merging + resuming, so a
value that can fire no guard is a 400 that costs nothing; or (c) at minimum, raise `onboarding@v1`'s
`maxSteps` explicitly in §4 and document the "N mistakes allowed" number as part of the def. (b) is
the one that generalises. Do **not** silently rely on the reader noticing F-4.

#### M-6 · The `cmp` `guard_judgment` trace requires editing `_select_transition`, which no unit owns

**Evidence.** §3.2 promises `GuardVerdict.rationale` is traced "the same way it records LLM ones",
and §7 U2.9 tests it. But `_select_transition` appends to `judgments` **only** when
`parsed.get("kind") == "llm"` (`executor.py:636`), and `_trace_step` formats the payload as
`f"{text} -> {verdict.decision}: {verdict.rationale}"` using `parsed.get("text","")`
(`executor.py:637`, `:706`). U1's scope is "`guards.py` only; no other module touched"; U2's is
"`_execute_step` dispatch + three handlers" plus `services._validate_def_spec`. Neither names
`_select_transition`. As written, U2.9 has no owning edit, and if bolted on unbriefed the trace line
for a cmp guard renders as `" -> True: ctx.decision eq 'approve' → true"` (leading `" -> "`, because
a cmp guard has no `text`).

**Remedy.** Add `_select_transition` to U2's file scope explicitly, with the judgment filter widened
to the cmp family and a payload format that uses a guard *label* (`parsed.get("text") or
_render(parsed)`). Note reassuringly in the plan that this is **outside the SHA-locked region** — the
`awk` range ends at `# ── seams` (`executor.py:394`) and `_select_transition` is at `:613`, so the
lock is unaffected. Saying so pre-empts an implementer freezing at the §3.1 stop-and-escalate rule.

### Minors

- **m-1 · §0 D-D contradicts §4.** D-D says "Six steps, **seven** transitions … plus **three** `cmp`
  ops (`exists`, `in`, `eq`)". §4.2 lists **six** transitions (#1–#6) and §4.2's coverage line names
  **four** ops (`exists`/`in`/`eq`/`truthy`). Six steps and the 8/6/6 step counts in §4.3 recount
  correctly (I re-derived all three paths); only D-D's summary is wrong. Fix the summary.
- **m-2 · The def key `onboarding` collides with a long-standing test fixture key.**
  `tests/test_api.py:413`, `tests/test_services.py:731`, `tests/test_repository.py` all publish a def
  keyed `onboarding` (version `"1"`) into the shared `reference` graph. `onboarding@v1` will not
  collide on `(key, version)`, and `wf_repo` wipes `reference` per test, so this is not a live bug —
  but `get_def(key, version=None)` sorts `version DESC` across both, and every grep for
  "the onboarding def" now returns two unrelated things. Suggest `access-request@v1` or
  `onboarding-proof@v1`.
- **m-3 · §6's blanket done-condition is impossible for U0.** "Each unit ends with … server pytest
  count **up** from 350" — U0 is queries + `test_queries.sh` only and adds no pytest tests. Scope the
  clause per unit.
- **m-4 · DESIGN §6.2 has no owning unit.** R-1 says the last-write-wins posture is "documented in …
  DESIGN §6.2", but §6's doc assignments give U3 only §14.4. Assign §6.2 to U3 (or U5).
- **m-5 · The ctx size cap can't live in the schema.** U3 says `schemas.py` bounds "serialized ctx ≤
  `MAX_CONFIG_LEN`". Pydantic sees only the *input*, not the stored ctx it merges into; the merged
  cap must be enforced in `services.submit_workflow_input` (rule 6). State which layer owns which
  bound.
- **m-6 · F-1 is correct but incomplete.** I verified independently: nothing in the tree reads
  `TRANSITION.on` or `StepResult.on` — the only consumers are
  `services.publish_workflow_def` writing it through (`services.py:504`) and the three
  `StepResult(..., on="done")` constructions (`executor.py:412,466,484`); `_select_transition` never
  touches it. So DESIGN §6.1's *"`on` is the event/outcome that fires it"* is indeed false. But the
  **same sentence** continues *"guards are evaluated in `TRANSITION.order`, first-firing wins"*,
  which is also wrong: the sort key is `(t["guard"] == "", t["order"])` (`executor.py:627`) —
  conditional guards first, `order` only as a tie-break within each class. Fix both halves in U2's
  §6.1 edit.
- **m-7 · Under D-C/C1, `wait` is behaviourally identical to `human`.** Same park, same
  `waitsForHuman` requirement, same input endpoint, same guard mechanism — only the `awaiting.kind`
  string in the output envelope differs. §3.3 is honest about "zero new machinery" but never says the
  two step types are indistinguishable to the engine. Say it plainly in the DESIGN §6.1 edit and in
  the K-025 handoff, or QA will reasonably read `wait` as an unimplemented distinct mechanism (R-5
  only covers the missing timer).
- **m-8 · U4's verification steps need a running server.** "`GET /workflow-defs/onboarding`,
  `GET /workspaces/acme/snapshots`" require uvicorn up; the §4.4 order stops at `seed_workflows.sh`.
  Add `start_server.sh` (or say "via the service layer, not REST") to the order.
- **m-9 · Two def-source conventions in one script.** §4.4 puts `ONBOARDING_DEF` in
  `server/falkorchat/proof_defs.py` (importable, no-drift — good, and the right lesson from K-022
  U14) while leaving the triage spec inline in `seed_workflows.sh`'s heredoc (verified: it is a
  quoted `<<'PY'` block with the spec literal at lines ~85–190). The script then "loops over
  `[triage-spec, ONBOARDING_DEF]`" mixing a literal and an import. Either move both into
  `proof_defs.py` (additive, zero behaviour change, one convention) or say explicitly why triage
  stays put. Also worth a sentence on why proof/demo def content belongs in the **shipped** package
  rather than beside the script.
- **m-10 · `ne` on a missing path returning `False` breaks De Morgan.** §3.2 is explicit and the
  bias-to-not-fire rationale is sound, but it means `{"not":[{"op":"eq",…}]}` and `{"op":"ne",…}`
  differ on a missing path — a real trap for def authors. Require it in the `guards.py` docstring and
  add an explicit test pair to §7 U1 (currently case 3 tests `ne`-missing but nothing contrasts it
  with `not(eq)`-missing).

### Nits

- **n-1** — `_run_decision_node`'s output envelope key `decision` (§3.3) collides semantically with
  `ctx.decision`, the approval value in the very def being shipped (§4.2 #4/#5). Rename to `node`.
- **n-2** — §3.3 has `human`/`wait` return `on="await"`, inventing a new `on` value in the same plan
  that declares `on` vestigial (F-1). Harmless but inconsistent; `on="done"` or a note would do.
- **n-3** — A `decision` step *with* outgoing transitions but *no* unconditional default and no
  `waitsForHuman` self-loops (OUTCOME C, `executor.py:384-388`) to budget exhaustion — the same
  footgun the plan rejects at publish time for `human`/`wait`. `onboarding@v1` avoids it (#3 is
  unconditional), so this is advisory: consider the symmetric invariant, or at minimum a warning in
  the `falkor-chat/AGENTS.md` executor-invariants bullet U2 is already adding.
- **n-4** — D-A's cons for A1 should name the concrete limit: `cmp` compares a **path to a literal**
  only; there is no path-vs-path form (`ctx.approver != ctx.requester`). That is the expressiveness
  cliff a reader will hit first.
- **n-5** — "QUERIES.md §12.1b" breaks the numeric heading convention of `docs/QUERIES.md`
  (`### 12.1` … `### 12.11`). Prefer §12.12/§12.13 and cross-reference §12.1.

---

## 3. Are D-A…D-E well-posed?

**Mostly yes — one omission is material.**

- **D-A (guard language) — well-posed.** A0/A1/A2/A3 span the real space, the cons are honest
  (A3's "sandbox-escape history" and A2's "~200 lines to get right" are both fair), and the
  `cmp`-vs-`expr` naming genuinely does keep DESIGN §13 literally true: I checked §13's resolution
  text ("**No expression library is built** — a would-be `expr` kind is a `NotImplementedError`
  seam") and `guards.py:144` raises on any non-`llm` kind, so reserving `expr` preserves both the
  sentence and the seam. §13 still needs the amendment U1 assigns it. The lock-in cost in R-4 is
  stated accurately. Gaps: n-4 (path-vs-path), and M-4 — the option set says nothing about *when*
  a guard is validated, which is where the real quality difference lies.
- **D-B (input channel) — the option set is complete for *channels*, incomplete for *mechanics*.**
  B1 vs B2 vs B3 is the right framing and B2's "it hollows out the very §6.3 claim" is a genuinely
  good argument. But the whole D-B trade-off table is about *who submits*, and the plan then makes an
  unexamined **two-query** choice inside B1 that a one-query variant strictly dominates (M-1). Since
  R-1 is offered to the stakeholder as an accepted risk of B1, and that risk is largely an artifact of
  the mechanic rather than of B1 itself, the decision as posed **overstates B1's cost**. Fix before
  the stakeholder signs: add the folded-CAS variant as B1's recommended mechanic and rewrite R-1.
- **D-C (`wait` semantics) — well-posed and the best-argued of the five.** "This system has no
  scheduler" is verifiable (nothing in `server/` runs a periodic worker; `BackgroundTasks` is
  request-scoped, `api.py:160`) and C2's true cost — a new component, a `wakeAt` index, crash/
  duplicate-fire semantics — is named rather than hidden. Filing K-028 instead of implying a timer is
  the right call. Only gap: m-7 (say that C1 makes `wait` mechanically identical to `human`).
- **D-D (proof def) — well-posed but internally inconsistent.** The alternatives considered are
  reasonable and the def does exercise what it claims. m-1 (7-vs-6 transitions, 3-vs-4 ops) and m-2
  (key collision) need fixing before this is put to a stakeholder. Recount confirmed: 6 steps,
  6 transitions, 8/6/6 steps of a 12 budget — the §4.3 table is arithmetically correct.
- **D-E (scope boundary) — well-posed; the recommendation is right and the consequence honestly
  flagged.** R-3's premise checks out for *production*: `seed_workflows.sh` publishes exactly one
  def, `triage@v1`, whose three steps are all `type:'agent'`. What R-3 misses is the *test* estate —
  B-1 and B-2. E1 remains the right choice; the risk statement needs widening, not the decision.

**Materially better option omitted:** one — the folded ctx-write-into-resume-CAS mechanic under D-B
(M-1). Everything else is a refinement of a recommendation I agree with.

---

## 4. What's solid

Worth protecting from churn:

- **§3.1's park-and-branch claim is correct, and I verified it rather than trusted it.** I traced
  every §4 def shape through the real `_drive_loop` (`executor.py:333-392`): `submit`/`approval`
  (`human`, `waitsForHuman`) and `provision` (`wait`) all reach OUTCOME B with `firing is None` and
  park after an advance-to-self `_record`; `route` can never park or spin because #3 is unconditional
  and `_select_transition`'s `(guard == "", order)` sort guarantees #2 is tried first; `activate` and
  `rejected` have no outgoing transitions, so they fall to the terminal branch and `complete_run` →
  `done`. On resume, `executor.resume` re-reads the run via `get_run` **after** the CAS
  (`executor.py:292-298`) and `_drive_loop` reloads `run_ctx` from `run["ctx"]` at entry
  (`executor.py:347`), so a ctx written before `resume()` is visible — exactly as §2 claims. **No
  shape in §4 spins, parks wrongly, or falls through to the terminal branch unintentionally.**
  `_drive_loop` genuinely does not need to change.
- **The `_drive_loop` SHA procedure works.** I ran the §3.1 `awk | sed | sha256sum` command verbatim
  against the tree; it prints `71055f756280`. The line-number-independent form and the "SHA only,
  the byte counts are wrong" instruction are both good practice and should be kept.
- **Every file:line citation in §2 is accurate** — `executor.py:396/613/333-392`, `guards.py:97/144`,
  `repository.py:1067/1151/1169`, `services.py:44/428/578`, `conftest.py:85`. That is unusual and
  makes the plan cheap to verify.
- **§5's "no DDL, no new index" claim is correct.** `bootstrap_schema.sh` already creates
  `WorkflowDefSnapshot.key` (:117), `WorkflowDefSnapshot.version` (:120), `WorkflowRun.runId` (:123)
  and the `WorkflowRun` uniqueness constraint (:179) in every workspace graph, so both new queries
  anchor on existing indexes and `bootstrap_schema.sh` is genuinely untouched. The RAM statement
  (a longer `ctx` string on a rare node type) is proportionate under rule 6. Subject to §5 below.
- **The `cmp` design is injection-safe and total in the way it claims.** Already-parsed JSON, a
  closed op whitelist of Python callables, dict-key traversal only (no indexing, no attribute access,
  no callables), two path roots, structural depth/width/node caps, no `eval`, no dependency. The
  missing-path/type-mismatch → `False` rules make it total against the inputs that reach it. The
  bias-to-not-fire mirrors `guards._coerce_verdict`'s existing bias-to-suspend
  (`guards.py:150-168`) — consistent with the module's established policy.
- **Scope discipline against K-027 is clean.** I read K-027's scope (BACKLOG :233-263 — judge-parse
  robustness, terminal-node-must-post, judge calibration); nothing in this plan touches the judge,
  `triage@v1`'s content, `_build_llm_judge`, or the agent-node loop. The one adjacency —
  zero-outgoing `decision` nodes as terminals — is a different concern from K-027's terminal-post
  contract. No creep found, and no gold-plating: everything in §1's in-scope list traces to either
  proving §6.3 or unblocking K-025.
- **The `proof_defs.py` single-source-of-truth decision (§4.4)** is the right lesson drawn from the
  K-022 U14 shell-out, and the seeding/verification order in §4.4 correctly reflects the
  `reference`-wiped-but-`ws:<id>`-not trap (I confirmed `tests/conftest.py:85`'s `wf_repo` wipes
  `reference`, and `test_queries.sh` drops it at teardown).

---

## 5. Could not verify statically — needs execution

1. **The two new queries' plans and contracts.** That `start_run_untriggered` PROFILEs as a single
   `Node By Index Scan` on `WorkflowDefSnapshot.key`, that its zero-row contract holds when the
   snapshot has no `START`, and that `set_run_ctx`'s (or M-1's folded) `WHERE r.status = 'waiting'`
   is a clean zero-row no-op on this build. This is U0's job — I confirmed only that the required
   indexes exist and that no new DDL is implied.
2. **The `test_queries.sh` count.** The plan expects 241 → ~245–247. I did not run the suite (it
   drops the global `reference` graph, i.e. mutates shared state, and the AGENTS.md re-seed
   discipline applies).
3. **The pytest delta.** I confirmed the **baseline** by collection only —
   `pytest --collect-only -q` reports `350/351 tests collected (1 deselected)`, matching the plan.
   The 395–405 target is plausible arithmetic over §7's ~39 named behaviours **only if** several are
   parametrized, and it does not account for B-1/B-2's fixture edits (which change tests rather than
   add them). Treat 395–405 as an estimate, not a done-condition; "the count rises and no existing
   test was silently weakened" is the check that matters.
4. **RAM delta from the longer `ctx`.** Asserted as ≈ nil and plausible (process runs are rare vs.
   messages); no measurement attempted.
5. **The K-025 handoff.** Whether QA reads C1's signal-driven `wait` as sufficient proof of §6.3 is a
   judgement call for the coordinator, not a static finding (see m-7 and R-5).

---

## 6. Open questions for the coordinator

- **OQ-A (blocks U0).** M-1: fold the ctx write into the resume CAS? This changes U0's deliverable,
  so settle it before the gate runs, not after.
- **OQ-B (blocks U3).** M-3: on an engine fault during a synchronous drive, does the endpoint return
  200 + `status:"failed"`, or 500? A product call, not an implementation detail.
- **OQ-C.** M-5: what is the sanctioned budget for a `process` run, and does invalid input get
  validated before it costs a step? Affects §4's def content and U3's schema.
- **The plan's own OQ-1** (route reachable when `FALKORCHAT_WORKFLOW_ENABLED` is off → 503) is a
  reasonable call as recommended; note it is the same handler pass as M-3, so decide them together.
  **OQ-2** (MCP parity) as a follow-up is right.
