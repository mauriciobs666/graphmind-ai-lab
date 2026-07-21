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

---

# Re-gate (v2)

> **Reviewer:** `analyst` · **Date:** 2026-07-20 · **Artifact:** `docs/plans/m3-process-flow.md`
> **patch v2** (architect) · **Baseline:** the v1 gate above. **Scope:** delta review only — the
> central mechanism (§3.1 park-and-branch, the §2 citations, §5's no-DDL claim) was verified in v1
> and was **not** re-derived, except where the patch touched it. Static review; nothing executed
> that mutates state (no `pytest`, no `test_queries.sh`).

## Verdict: **approve with suggestions**

All 2 blockers and all 6 majors are **genuinely closed in the text**, not just in the gate-response
table — I checked each referenced section against the tree, and none of them moved rather than
closed. D-F/D-G/D-H are internally consistent and correctly propagated. The declines are honest.
**U0 and U1 can be dispatched now.**

Three new findings, **all carried into implementation** (one for U2, two for U3); none blocks U0/U1
and none blocks the plan.

## 1. Blocker / major closure — verified against the tree

| # | Closed? | Evidence I checked |
|---|---|---|
| **B-1** | **Yes** | §2 F-3 now names both stub cases; the fixture is real and the citation exact — `test_executor.py:345` is `{"key":"end","type":"task","config":"{}"}` driven by `test_hallucinated_mention_does_not_fail_the_run` (`:352`). The `task → agent` edit is in U2's file scope (§6) and in §7's baseline list. R-3 restated |
| **B-2** | **Yes** (one execution gap, M-7 below) | Fixture list is **exhaustive** — I swept the tree for `human`/`wait` step fixtures: the only ones reaching `services._validate_def_spec` are `test_api.py:418` and `test_services.py:719/808/822/836/851/866`, exactly as listed. The three others are unaffected and correctly not listed: `test_repository.py:734` (repo-level publish, bypasses validation), `test_services.py:881` (a **materialize** test — `materialize_def` never validates), `test_queries.sh:681/752` (raw Cypher). Invariant runs **last** (§3.3 box) with an ordering pin (§7 U2.8b); the five `pytest.raises` gain `match=` |
| **M-1 → D-F** | **Yes** | `set_run_ctx` is gone from §3.4, §3.5, §5 and U0/U3 — no residue anywhere. §12.13 is `resume_run` + one `SET` term. `executor.resume` is at `:287` and already re-reads via `get_run` at `:295` **after** the CAS, so the folded write is visible to the drive. R-1 correctly narrowed to the stale *merge read* |
| **M-2** | **Yes** | New F-6 documents the live path; §3.4 step 3 puts the reserved-key rule **in the service**; tests 6b/6c; R-8 split into F-5 (latent) / F-6 (reachable) |
| **M-3 → D-G** | **Yes** | Five handlers specified; each premise verified — `WorkflowRunNotFoundError` exists (`repository.py:64`) and is genuinely unhandled (`app._register_error_handlers` maps only `WorkflowDefSpecError`/`WorkflowDefNotFoundError`); `guards.WorkflowConfigError` is a plain `Exception` (`guards.py:72`), so an explicit handler is required; `_require_executor` (`services.py:570`) raises a bare `RuntimeError` today, and a subclass keeps `test_start_workflow_run_without_executor_raises` (`test_services.py:999`, `pytest.raises(RuntimeError)`) green exactly as D-G claims |
| **M-4** | **Yes**, with M-7 | `guards.validate_cmp` authored in U1, called from `_validate_def_spec` in U2, gated on `kind ∈ {cmp,all,any,not}`; the `{"expr":"x>0"}` fixture contract is pinned (§7 U2.11) |
| **M-5 → D-H** | **Yes** | Validation before merge, in the service; free-400 property asserted (§7 U3.14, U4 "typos are free"); `maxSteps = 24` with the arithmetic written out. The budget mechanics hold: park (OUTCOME B, `executor.py:374–382`) bumps `stepCount` with **no** budget check, and the *next advance* (OUTCOME A, `:369`) trips it — which is what D-H's "16 spare re-parks" describes |
| **M-6** | **Yes** | `_select_transition` + `_trace_step` are in U2's file scope, and the §3.1 box is the right anti-freeze device |

**SHA re-confirmed: `71055f756280`** (ran the §3.1 command verbatim). **The three "outside the lock"
edit sites genuinely are:** the locked region is `_drive_loop` (`:333`) → `# ── seams` (`:394`);
`resume` is `:287` (before it), `_execute_step` `:396`, `_select_transition` `:613`, `_trace_step`
`:684` (all after the marker).

## 2. New claims spot-checked

- **D-H "no new query" — confirmed.** `get_run` (`repository.py:1245`) returns `atStepKey`, `defKey`,
  `defVersion` (`:1255–1269`); `get_snapshot` (`:1403`) → `_read_subgraph` → `_READ_META_CYPHER`
  (`:945`) collects `{key, type, config}` per step. Also load-bearing and true: `suspend_run`
  (`:1151`) flips status only — it does **not** clear `AT_STEP`, so `atStepKey` survives the park and
  the parked step is resolvable.
- **`access-request@v1` rename — consistent everywhere.** The only remaining `onboarding` mentions in
  the plan are the four historical ones explaining the rename (D-D, §4 header, summary, gate table).
  §4, §6 U4, §7, `proof_defs.ACCESS_REQUEST_DEF`/`ACCESS_REQUEST_MAX_STEPS` and the seed section all
  use the new key.
- **`submit.fields = ["request"]` — matches §4.3.** Step 2 posts `{"request":{"role":…}}`, guard #1
  reads `ctx.request.role` (`exists`). The other two declarations line up too: `approval.fields
  ["decision"]` + `expects` vs. step 5's `{"decision":"approve"}`; `provision.signal "provisioned"`
  vs. step 7's `{"provisioned":true}` and guard #6's `ctx.provisioned truthy`.
- **Declines are honest, all three.** m-9: `triage@v1` really is a literal inside
  `seed_workflows.sh`'s `<<'PY'` heredoc (`:73`–`:220`), and it is the **live published** def — the
  create-only `MERGE … ON CREATE SET` risk is real; K-029 records it. n-3: `test_services.py:719`'s
  `review` `decision` step's only outgoing transition (`:726`) is guarded, so the symmetric invariant
  would indeed retro-reject it. Sentinel belt: `find_waiting_run_for_thread` (`:1299`) matches
  `r.waitingThreadId = $threadId` by **equality**, so with the reserved-key rejection in place `''`
  is structurally unmatchable from a real thread — the belt is genuinely redundant, not waved away.

## 3. New findings (carry into implementation — none blocks U0/U1)

### M-7 · major (U2) — the two new publish invariants assume dict-shaped `config`/`guard`, but the REST path delivers **strings**

**Evidence.** `_validate_def_spec` (`services.py:426–478`) today inspects only `key`/`type`/`start`;
it never touches `config` or `guard` — serialization happens after it, in `publish_workflow_def`
(`:494/:504`) via `_serialize_opaque`. The values it receives are **heterogeneously typed**: over
REST they are strings by schema (`schemas.py:48` `config: str | None`, `:58` `guard: str | None`) —
`test_api.py:418` is `"config": "{}"` — while service-layer callers pass dicts
(`test_services.py:719` `{"a": 1}`) or even non-JSON strings (`:720` `"raw-string"`).

**Why it matters.** Two failure directions, both easy to land silently:
(a) a naive `step.get("config", {}).get("waitsForHuman")` raises `AttributeError` on a str →
**500 on `POST /workflow-defs`**; (b) if the implementer guards with `isinstance(..., dict)`, then
**every def published over REST escapes both new invariants** — `access-request@v1` (seeded through
the service layer with dicts) is validated, the REST surface is not, and M-4's whole "caught at
authoring time" argument evaporates for the front door. Also, B-2's instruction "add
`"waitsForHuman": true` to `test_api.py:418`" is not literally applicable to a string config — it
must become `'{"waitsForHuman": true}'`.

**Suggested improvement (U2, ~5 lines of plan text).** State the normalization once, in §3.3 and
§3.2: *parse `config`/`guard` with `json.loads` when they arrive as `str`; a non-dict result is
treated as "no declaration" for the guard check (so `{"expr":"x>0"}` and `"raw-string"` keep
publishing) and as a **`WorkflowDefSpecError`** for a `human`/`wait` step (a step that must declare
`waitsForHuman` cannot carry an opaque config).* Keep `guards.validate_cmp(spec)` taking an
**already-parsed dict** — normalization belongs at the `_validate_def_spec` call site, so U1's
deliverable is unchanged. Restate the `test_api.py:418` edit in its string form.

### m-11 · minor (U3) — D-G's catch list names an exception that does not exist

`StepBudgetExceededError` appears nowhere in the tree, and budget exhaustion **never raises**:
`_fail_budget` (`executor.py:663`) stamps `fail_run` and *returns* `"failed"` through OUTCOME A/C
(`:370`, `:387`). As written, D-G invites an implementer to invent the exception and convert a clean
terminal return into a raise — a behaviour change to the existing engine, inside the unit that must
not touch it. **Fix:** drop it from D-G's list and add one clause: *budget exhaustion already returns
a terminal `failed` status and needs no catch — it reaches the same envelope through the normal
return path.*

### m-12 · minor (U3) — D-G should re-raise when the post-fault re-read is not terminal

D-G re-reads the run and "reports the graph's status, never a guessed one". If a fault ever escapes
*before* `_drive`'s net stamps `fail_run` (or if `fail_run` itself fails), that re-read returns
`running` and the endpoint would report a **zombie run as a 200/201 success envelope** — the worst
possible reporting for the audit property §6.3 exists to prove. **Fix:** one clause in D-G — *if the
re-read status is not in `{failed, done, waiting}`, re-raise (a 500 is the correct answer there)* —
plus an assertion in §7 U3.11 that the envelope's status came from `get_run`.

### n-6 · nit — three line citations drifted (inherited from my v1 review)

The five `pytest.raises(WorkflowDefSpecError)` tests are at `test_services.py:804/818/832/846/861`
(the plan's `:803/817/831/845/860` point at the blank line above each `def`), and the
`{"expr":"x>0"}` guard fixture is at **`:726`**, not `:721` (`:721` is the `done`/`message` step).
Tell the implementer to locate these by test name / content; the `test_executor.py:345`,
`test_api.py:418` and `test_services.py:719/808/822/836/851/866` citations are exact.

## 4. What to carry, and what not to block on

- **M-7** — worth a two-sentence plan patch before U2 starts, but U2 is three units away; it can also
  be handed to the U2 implementer as a written constraint. **Does not block U0/U1.**
- **m-11 / m-12 / n-6** — hand to the U3 implementer as-is; no plan patch required.
- Everything from the v1 gate is answered. **Dispatch U0 (graph-dba, §12.12 + §12.13) and U1
  (tdd-engineer, `guards.py`) now.**

## 5. Still not verifiable statically (unchanged from v1 §5)

Both new queries' PROFILE plans and zero-row contracts — now including §12.13's *nothing-written*
contract, which D-F's entire argument rests on and which the plan correctly flags as "verify, do not
assume". Plus the real suite counts and the RAM delta at integration.

---

# Implementation gate (U0–U4)

> **Reviewer:** `analyst` · **Date:** 2026-07-21 · **Artifact:** the K-024 implementation —
> commits `788e5bf` (U0+U1) · `efdeeb3` (U2) · `670474a` (U3) plus the **uncommitted U4
> working tree** (`server/falkorchat/proof_defs.py`, `server/tests/test_process_flow.py`,
> `scripts/seed_workflows.sh`, `AGENTS.md`), reviewed as one diff against `e3c09b3`.
> **Baseline:** `docs/plans/m3-process-flow.md` v2.1 (§3 design, §4 def, §6 done-conditions,
> §7 test strategy), `docs/plans/m3-process-flow-coordination.md` (D-A…D-H, O-1…O-6), and the
> two prior gates in this file.
> **Method:** static only. Per the brief I did **not** run `pytest` or `test_queries.sh` (both
> wipe the shared `reference` graph); no tree-mutating git command was used — baselines were
> read with `git show <ref>:<path>`. The only things executed were `py_compile` and a
> read-only import of `proof_defs` + `guards.validate_cmp` over the shipped def.

## Verdict: **approve with suggestions**

No blockers. The slice does what the plan says, in the places the plan says, and the test
estate binds it rather than decorating it. `_drive_loop` is untouched (SHA verified), the
three §4.3 paths trace correctly through the real engine, and every finding from both prior
gates is closed **in code**, not merely claimed. Two majors below are worth fixing before
U5 closes K-024; neither invalidates the delivery.

## 1. Verified (ran or traced end-to-end)

- **`_drive_loop` lock intact.** The §3.1 awk/sha command prints **`71055f756280`** on the
  current tree. The three sanctioned edit sites are all outside the locked region and none
  changes park-and-branch semantics: `resume` (`executor.py:308`) only chooses between
  `resume_run` and `resume_run_with_ctx` and still re-reads the run with `get_run` after the
  CAS, so `_drive_loop` still loads the just-written ctx; `_execute_step` (`:432`) is a pure
  dispatch whose `agent`+`llm is None` branch returns the identical stub as before;
  `_select_transition` (`:732`) changed only the *judgment-recording* filter and label
  (`TRACED_GUARD_KINDS`, `render_label`) — the `sorted(..., key=(guard == "", order))` rule and
  the first-firing return are byte-identical; `_trace_step` (`:812`) only prefixes the payload.
  No new `suspend_run`/`complete_run`/budget call sites exist outside the lock.
- **The three §4.3 paths, traced against the real loop** with `proof_defs.ACCESS_REQUEST_DEF`:
  privileged = park/advance ×3 → `activate`, **8** steps; standard hire = 6 (#2 loses, the
  unconditional #3 wins — the conditional-beats-unconditional rule exercised from the "loses"
  side); rejected = 6 → `rejected`, status `done`. Terminal `decision` nodes reach
  `complete_run` via OUTCOME C (`executor.py:426–428`). The step-count arithmetic matches
  §4.3 and matches what `test_process_flow.py` asserts.
- **The shipped def matches plan §4 exactly**: 6 steps, **6** transitions, ops
  `{exists, in, eq, truthy}` (checked by importing and by running `guards.validate_cmp` over
  every guard — all pass), `ACCESS_REQUEST_MAX_STEPS = 24`, key **`access-request`**, kind
  `process`, start `submit` (`proof_defs.py:46–144`).
- **U2 publish invariants.** `WAITING_STEP_TYPES` + the `waitsForHuman` check and the
  `validate_cmp` call both sit **after** the kind / step-type / duplicate-key / start-count /
  dangling-endpoint checks (`services.py:569–588`), i.e. last, and the ordering is *pinned* by
  `test_executor_process.py:392`. `_normalize_opaque` (`services.py:177`) is applied to both
  `config` and `guard` before either check, with the M-7 shape matrix covering all four
  directions (`test_executor_process.py:457–520`) — including that a **string** `'{"waitsForHuman":
  true}'` passes and `"{}"` is rejected, so the REST door cannot escape either invariant.
  `validate_cmp` is called only for `kind ∈ CMP_KINDS`, so `{"expr":"x>0"}` and `"raw-string"`
  still publish (`:431`). Strict-at-publish / total-at-drive (O-1) is real: one validator, the
  `check_paths` flag (`guards.py:256` vs `:331`).
- **Every prior-gate finding closed in code:** B-1 (`test_executor.py:345` `task`→`agent`);
  B-2 (exactly the named fixture edits, and the five `pytest.raises` now carry `match=`; the
  diff against `e3c09b3` shows **no other** pre-existing test touched); M-1/D-F
  (`resume_run_with_ctx`, `repository.py:1251`; `set_run_ctx` appears nowhere outside plan
  prose); M-2/F-6 (`RESERVED_CTX_KEYS`, rejected before any write on the start ctx *and* the
  input); M-3/D-G (five handlers registered via a table, no blanket `RuntimeError` handler,
  `app.py:93–127`); M-4; M-5/D-H; M-6; M-7; m-11 (**no** `StepBudgetExceededError` invented —
  `_fail_budget` still *returns* `"failed"`, pinned at `test_process_input.py:593`); m-12
  (`TERMINAL_OR_PARKED_STATUSES` re-raise, pinned twice, plus the teco ruling that `ctx` comes
  from the same post-fault `get_run`).
- **O-6 mitigation holds in fact.** The shipped def carries 6 transitions; all **9** `_publish`
  call sites in `test_executor_process.py` pass a non-empty `transitions`; `test_process_flow.py`
  publishes the real def; `test_process_input._materialize` (`:86`) documents the constraint in
  a comment. No new fixture publishes zero transitions.
- **The coder's pytest/`reference` learning is correct.** `conftest.wf_repo` (`tests/conftest.py:85`)
  wipes `reference` at **fixture setup**, once per test — so a pytest session *leaves behind* the
  last workflow test's published defs rather than emptying the graph. See m-C.
- **Test quality spot-check (the "9 mutations, 0 survivors" claim).** I did not re-run mutation
  testing, but the tests bind behaviour rather than execute it: the D-H "typos are free" test
  asserts `stepCount`, `status`, `atStepKey` **and** ctx equality before/after three rejections
  and then proves the run is still advanceable (`test_process_flow.py:244–279`); the failed-envelope
  tests assert `out["ctx"] != {"decision": "approve"}` and `== graph_ctx`, which kills the obvious
  `"ctx": merged` mutation; `test_the_failed_envelope_status_comes_from_a_post_fault_get_run`
  forces the graph to say `done` so a hardcoded `"failed"` cannot pass; the F-5 assertion
  `find_waiting_run_for_thread(thread_id="")` is made **while a parked process run exists**, so
  deleting the short-circuit fails it. Two assertions are tautological rather than behavioural
  (`test_process_flow.py:297–304` asserts properties of `ACCESS_REQUEST_DEF` against itself) —
  harmless as spec pins, but they are not evidence of engine behaviour.

## 2. Findings

### M-A · major — the D-G drive-fault catch also swallows the **chat/@mention** start path, silencing its only stack trace

**Evidence.** `services.start_workflow_run` routes *both* start paths through `_drive_or_fault`
(`services.py:748`), which catches `NotImplementedError` and `WorkflowConfigError` and returns a
dict. `trigger.py:76` (step 3, @mention-to-start) calls that same method, and `api._safe_run_workflow`
(`api.py:82–92`) is a `try/except` whose *whole purpose* is to log the stack of anything that
propagates out of the background trigger. Nothing in `_drive_or_fault` logs. `executor.py:352–355`
still documents the M-1 net as re-raising "so `_safe_run_workflow`'s isolation logs the stack" —
which is now false for the start path (it remains true for `resume_workflow_run`, which does not
use `_drive_or_fault`, so the two chat paths now behave differently).

**Why it matters.** A live `triage@v1` run that dies on an unwired judge (`WorkflowConfigError`) or
an unimplemented step type now leaves **no log line anywhere** — only a `failed` run with a ctx note
that someone has to go look for. That is exactly the live-triage diagnosability K-027 exists to
improve, degraded by this slice. Plan §6 U3's done-condition says "no change to the existing
`@mention` start path's behaviour (pinned by a test)"; the graph outcome is indeed unchanged, but
the propagation/observability contract is not, and no test pins it (`test_trigger.py:34` uses a fake
`Services`, so it cannot see this).

**Suggested improvement.** One line in `_drive_or_fault`'s `except`:
`logging.getLogger(__name__).exception("workflow drive fault for run %s", run_id)` before the
re-read — it keeps D-G's envelope exactly as specified while restoring the trace. Then correct the
`_drive` docstring (`executor.py:352–355`) to say the re-raise reaches `_safe_run_workflow` *unless*
a service-level D-G catch intercepts it. (Alternative, heavier: gate the catch on a `catch_faults`
flag the REST routes pass and the trigger does not — more faithful to D-G's "the *endpoints* catch
the drive fault", but more surface.)

### M-B · major — O-6 is reachable from the public publish route and **poisons** the `(key, version)`; the mitigation is documented only where implementers already knew

**Evidence.** `schemas.PublishWorkflowDefIn.transitions` is `Field(default_factory=list, …)`
(`schemas.py:71`), and `_validate_def_spec` has no non-empty-transitions rule, so
`POST /workflow-defs` with `"transitions": []` is a schema-legal, validation-passing body. It then
reaches `_PUBLISH_CYPHER` (`repository.py:937–956`), whose trailing `UNWIND $transitions` collapses
the row stream **after** the `WorkflowDef`, its `Step`s and the `START` edge are already MERGEd;
`publish_def` (`repository.py:1019`) indexes `res.result_set[0]` ⇒ `IndexError` ⇒ **HTTP 500**.
Because publish is `MERGE … ON CREATE SET`, the corrected retry on the same `key@version` is a
**silent no-op on the half-written def** — the version is permanently wrong and cannot be repaired
by re-publishing. Same for `materialize_snapshot` (`:1397`).

**Why it matters.** O-6 was raised as "latent, invisible today because every publish test carries ≥1
transition". That is true of the *test estate*, but the front door is open, and the failure mode is
worse than an `IndexError`: it is an unrepairable def version. Today the constraint is written down
in `proof_defs.py`'s docstring and a comment in `test_process_input.py:86` — both places an author
who already knows about it would look. It is **not** in `services.publish_workflow_def`'s docstring,
not in `AGENTS.md`'s workflow-def invariants list, and not in `QUERIES.md` §11.1.

**Suggested improvement (for U5, with the backlog filing).** (a) Add the constraint to
`services.publish_workflow_def`'s docstring and to the `AGENTS.md` executor/workflow-def invariants
bullets — one sentence each, the same slot the `waitsForHuman` bullet already occupies. (b) Consider
the cheap real fix rather than only filing it: a non-empty `transitions` check in `_validate_def_spec`
(running **last**, like the other two) turns a 500 + poisoned version into a clean
`WorkflowDefSpecError` → 400 with nothing written. Retro-cost is exactly **one** fixture:
`test_services.py:1080` (`test_agent_step_type_is_accepted_by_publish_validation`) is the only
existing test that publishes *successfully* with `transitions=[]` (the four others at `:802/:821/
:835/:850` already raise on an earlier rule). Note this rejects a legitimate shape — a single-step
def — so if that is wanted, fix the query instead (guard the `UNWIND` the way the §4 mention block
does) and keep the publish permissive.

### m-A · minor — an **empty** input body is accepted, wins the CAS and burns a step of budget

**Evidence.** `SubmitWorkflowInputIn.input` defaults to `{}` (`schemas.py:123`), and
`submit_workflow_input` (`services.py:759`) has no empty-payload rule: `{}` passes the reserved-key
check and the parked-step check (nothing is undeclared), merges to a no-op, wins the resume CAS,
re-executes the parked step, fires no guard and re-parks — costing one step of the 24. D-H's
"mistakes are free" argument covers wrong *values* and undeclared *keys*, but not the one mistake a
UI is most likely to emit (a submit with nothing filled in). 16 such calls kill an
`access-request` run.

**Suggested improvement.** Reject an empty `input` in `submit_workflow_input` with
`WorkflowInputRejectedError` ("no input submitted") — service layer, so MCP (OQ-2) inherits it, and
it is free by the same argument the other three rules use.

### m-B · minor — the `_publish` test helper's `transitions=()` default invites O-6 back in

**Evidence.** `tests/test_executor_process.py:344` — `def _publish(svc, *, steps, transitions=(), …)`.
All nine current call sites pass a non-empty list, so nothing is broken today, but the helper's
default is precisely the shape that produces a partial write plus `IndexError` in a future test, and
the failure would look like an unrelated engine bug.

**Suggested improvement.** Make `transitions` a required keyword (or default it to
`[SINK_TRANSITION]`), with a one-line comment naming O-6.

### m-C · minor — the new `AGENTS.md` seed row says pytest "drops `reference`"; it actually **leaves the last workflow test's defs behind**, which is what makes the `already present — no-op` verdict untrustworthy

**Evidence.** `wf_repo` (`tests/conftest.py:85–96`) wipes `reference` at **fixture setup**, per test.
A finished pytest session therefore leaves `reference` holding whatever the last `wf_repo` test
published — and `test_process_flow.py:68` publishes `ACCESS_REQUEST_DEF` under the **production**
`access-request@v1` key/version into that shared graph. Concretely, the hazard the row warns about
gets *harder* to see: edit `proof_defs.py` → run pytest (the test republishes the **new** def into
`reference`) → run `seed_workflows.sh acme` (prints `already present — no-op` for the def) while
`ws:acme` still holds the **old** snapshot, which is what the executor drives. The row's own
split-brain paragraph is right; its "both drop `reference`" premise is what misleads.

**Suggested improvement.** One clarifying clause in the `seed_workflows.sh` row and the script
header: *"pytest wipes `reference` per workflow test, so a finished run leaves the last test's defs
behind — `already present — no-op` after a pytest run may be reporting a test's publish, not a
seed."* Optionally have `test_process_flow.py` publish under a test-only version (e.g. `v1-test`)
via `{**ACCESS_REQUEST_DEF, "version": …}`; that keeps the no-drift property for content while
removing the production-key collision — at the cost of no longer proving the shipped `(key, version)`
pair publishes. Worth a deliberate call in U5, not a silent one.

### m-D · minor — doc drift for U5 (report-only, as briefed) — and the drift is **not** where it was expected

The brief anticipated stale prose in plan §4. It is not there: `m3-process-flow.md:88–97` states
"Six steps, six transitions … four `cmp` ops" and carries the m-1 recount note, and §4.2's table is
six. The stale copies are all in the **coordination ledger**:

- `m3-process-flow-coordination.md:69` — U4's row still reads "the `onboarding@v1` def".
- `:79` — D-D still reads "`onboarding@v1` … 6 steps / 7 transitions … 8 of the 12-step budget"
  (the rename, the recount and D-H's `maxSteps = 24` all post-date it).
- `:65` — U0's row still names "`set_run_ctx` (§12.12)" and "§12.1b", both dropped by D-F/n-5.
- `:324` — the v1 gate note "recount before U4" is now closed and should say so.

U5 already owns the Decisions table fill-in; folding these four in costs nothing.

### n-A · nit — the seed script's `**ACCESS_REQUEST_DEF` splat makes the constant's key set a signature

`scripts/seed_workflows.sh` builds `{**ACCESS_REQUEST_DEF, "key": …, "version": …}` and calls
`services.publish_workflow_def(ctx, **spec)`. Adding any field to `ACCESS_REQUEST_DEF` (a `notes`,
a `budget`) breaks the seed with a `TypeError` at run time — and only there, since the acceptance
test splats it the same way. A comment on `ACCESS_REQUEST_DEF` saying "these six keys are
`publish_workflow_def`'s signature — do not add fields" would cost one line.

### n-B · nit — `services.py` now imports `MAX_CONFIG_LEN` from `schemas.py`

Justified in the comment (`services.py:44–49`) and harmless today (`schemas` is a pydantic-only
leaf), but it is the first boundary→service constant import in this codebase. If a second one
appears, move both to `config.py` rather than growing the dependency.

## 3. What's solid

- The central design bet held: **`_drive_loop` never moved**, and a business process fell out of
  park-and-branch with no new outcome, no new state and no scheduler. That is the strongest thing
  in this slice and it is worth saying plainly.
- `guards.py` is the model deliverable of the four units: closed whitelists, no parser, no `eval`,
  total at drive, strict at publish, and the two easy-to-get-wrong rules (totality, the De Morgan
  asymmetry) written into the module docstring *and* pinned by a contrasting test pair.
- The D-G/m-12 envelope is now internally consistent — `status` and `ctx` from one post-fault
  observation — and the teco ruling that produced it is the right call.
- D-H's "mistakes are free" is a genuinely good property and is asserted, not assumed.
- The blast radius on the existing estate is exactly what the plan budgeted: the `git diff` against
  `e3c09b3` touches `test_api.py`, `test_executor.py` and `test_services.py` **only** at the named
  fixtures, and the five type-only `pytest.raises` are now message-asserting. The two remaining
  untightened `pytest.raises(WorkflowDefSpecError)` (`test_services.py:778, 801`) are not vacuous —
  their specs fail on the kind / step-type whitelist, which runs first.
- Doc discipline held: QUERIES §12.12/§12.13 carry PROFILEs and the zero-row contracts, DESIGN §7.1's
  pinned count moved 241→256, DESIGN §6.2 and §14.4 carry the ctx-write posture and the full error
  map, and the `AGENTS.md` invariants list (including the *unenforced* n-3 warning) is accurate.

## 4. Not verifiable statically / deliberately not checked

- The suites. Per the brief I ran neither `pytest` nor `test_queries.sh`; the reported
  **529 passed / 1 deselected** and **256/256**, the live re-seed, and both defs' presence in
  `reference` + `ws:acme` are the coordinator's evidence, not mine.
- The "9 mutations, 0 survivors" figure — I read the tests and confirmed they kill the obvious
  mutations named in §1, but I did not re-run mutation testing.
- Whether `seed_workflows.sh` prints `already present — no-op` for both defs on a clean re-run
  (it is structurally right — `_probe` reads before publishing, per def — but it was not executed
  here).
- Any RAM measurement; §5's "≈ nil" is inherited from U0's gate.

## 5. Open questions for the coordinator

1. **M-A's shape.** Log-and-swallow (one line, keeps D-G verbatim) or gate the catch to the REST
   callers (truer to D-G's wording, more surface)? This touches K-027's territory, so a decision
   now avoids two people fixing it differently.
2. **M-B(b).** Fix O-6 in this slice (one validator rule + one fixture edit) or file it only?
   Filing it leaves a public route that can permanently poison a def version.
3. **m-C's second half.** Should the acceptance test keep publishing the production
   `access-request@v1` into the shared `reference`, or move to a test-only version? Either is
   defensible; it should be a written choice in U5, since the current state makes the seed
   script's own success line unreliable.

---

# Re-gate (U4b)

> **Reviewer:** `analyst` · **Date:** 2026-07-21 · **Artifact:** the **uncommitted U4b working
> tree** on top of committed `670474a` — `server/falkorchat/{services.py, executor.py,
> proof_defs.py}`, `server/tests/{test_process_input, test_services, test_process_flow,
> test_executor_process}.py`, `AGENTS.md`, `scripts/seed_workflows.sh`.
> **Baseline:** the "Implementation gate (U0–U4)" section above — findings **M-A, M-B, m-A,
> m-B, m-C, n-A, n-B** and its §5 open questions, plus the coordinator's rulings on those
> three questions.
> **Scope:** diff-scoped re-gate only. U0–U4 were gated *approve with suggestions* and are not
> re-reviewed; m-D (coordination-ledger drift) is U5's and is untouched here by design.
> **Method:** static only. Per the brief I ran **neither** `pytest` nor `test_queries.sh`, and
> used no tree-mutating git command (baselines via `git diff` / `git show`). Nothing was
> executed; every claim below is from reading `services.py`, `executor.py`, `repository.py`,
> `app.py`, `trigger.py`, `schemas.py`, the four test modules and the two docs.

## Verdict: **approve with suggestions**

No blockers. All seven findings are closed, and closed **as the coordinator ruled** — not
more, not less. The two majors are fixed at the right layer with the right blast radius: M-A
is exactly one `logging.exception` line inside the existing `except`, with D-G's envelope
untouched and the catch *not* gated to REST callers; M-B is one rule, placed last, that
provably prevents the write rather than renaming the exception. The three new tests bind
behaviour rather than execute it. Five residual items below, all minor/nit; one of them
(**r-1**) is a real remaining hole on the *materialize* side of the same defect and is worth a
U5 line, and **r-2** is the backlog filing the brief asked me to rule on — it deserves one.

## 1. Per-finding disposition

### M-A — **closed** ✔ (log-and-swallow, exactly as ruled)

- One line added: `services.py:996` — `logging.getLogger(__name__).exception("workflow drive
  fault for run %s", run_id)`, placed **inside** the existing
  `except (NotImplementedError, WorkflowConfigError)` and **before** the post-fault
  `get_run`. The only other module change is `import logging` (`services.py:16`).
- **D-G's envelope is byte-unchanged.** The `git diff` on `_drive_or_fault` shows zero edits
  to the `run = get_run(...)` / `status` / `TERMINAL_OR_PARKED_STATUSES` re-raise / return
  tuple — the log line is a pure prefix. The `app.py:93–127` error map is untouched, so the
  200/201 `{"status":"failed", …}` shape and every mapped code are as gated.
- **Not gated to REST callers** — no `catch_faults` flag, no caller-dependent branch. K-027's
  territory is untouched, as ruled.
- **The corrected `executor.py:352–358` comment describes the actual two-path behaviour.** I
  verified both halves against the code: `resume_workflow_run` (`services.py:1009–1020`) calls
  `executor.resume` directly with **no** catch, so a fault propagates — and `trigger.py:72`
  (the chat resume leg) routes through it into `api._safe_run_workflow` (`api.py:73`). The
  other leg, `trigger.py:76`, calls `start_workflow_run`, which *does* go through
  `_drive_or_fault` (`services.py:775`). The docstring says precisely that, and its closing
  claim ("so a drive fault is never silent on either route") is now true.
- Pinned by `test_process_input.py:521` (see §3 for the mutation check).

### M-B / O-6 — **closed** ✔ (the partial write is genuinely prevented, not renamed)

Traced end-to-end:

- The rule is `services.py:604` (`if not transitions: raise WorkflowDefSpecError(...)`), inside
  the **static** `_validate_def_spec`.
- `publish_workflow_def` (`services.py:640`) calls `_validate_def_spec` **before** building
  `repo_steps`/`repo_transitions` and before `self._repo.publish_def(...)`. The raise therefore
  happens with **no** repository call at all — this is prevention, not a nicer exception.
- Against `repository._PUBLISH_CYPHER` (`repository.py:937–956`): the query MERGEs `d`, UNWINDs
  `$steps` (MERGE `Step` + `HAS_STEP`), `MATCH`es the start step and MERGEs `(d)-[:START]->`,
  and only **then** `UNWIND $transitions` → `WITH d, stepCount, count(rel)` → `RETURN`. With
  `$transitions = []` the stream collapses at that UNWIND, after three write clauses, and
  `publish_def`'s `res.result_set[0]` raises `IndexError`. The diagnosis in the prior gate is
  confirmed against the actual Cypher; the fix removes the only reachable route to it via
  publish.
- **The ordering pin is real**, on both sides: the rule sits after the kind / step-type /
  duplicate-key / start-count / dangling-endpoint checks *and* after the two U2 invariants
  (`services.py:569–602`), and `test_services.py:869` publishes a spec that violates **both**
  the start-count rule and the transitions rule and asserts it fails with
  `match="exactly one start step"`. Independently, the four pre-existing tests at
  `test_services.py:802/821/835/850` all pass `transitions=[]` and expect *earlier* errors — so
  a mis-placed rule would fail five tests, not one.
- **No existing fixture now fails for the wrong reason.** The only test in the estate that
  published successfully with `transitions=[]` was
  `test_agent_step_type_is_accepted_by_publish_validation` (`test_services.py:1113`); it gained
  a second `agent` step plus one transition, and still asserts exactly what it asserted before
  (`repo.published[0]["key"] == "triage"` — that `agent` is an accepted step type). Nothing
  about the assertion was weakened. A repo-wide grep for `transitions=[]` / `"transitions": []`
  finds no other publish-path caller.
- Documentation landed in the three places the finding named: `services.publish_workflow_def`'s
  docstring (`:624`), `_validate_def_spec`'s docstring (`:523`, "Three further invariants"), and
  the `AGENTS.md` executor/workflow-def invariant bullets ("All three run **last**"). QUERIES.md
  §11.1 was **not** updated — acceptable, since the rule is a service invariant, not a query
  contract, but see **r-2**.

### m-A — **closed** ✔

`services.py:828` — `if not input: raise WorkflowInputRejectedError("no input submitted …")`,
placed after the 404/409 checks and **before** `_reject_reserved_keys` /
`_validate_against_parked_step` / the merge / the CAS, i.e. before anything is written and
before any budget is spent. The docstring's order-is-load-bearing list (`:798`) was updated to
match the code. It is at the service layer, so MCP inherits it (OQ-2), and `app.py:107` already
maps `WorkflowInputRejectedError → 400`, so the REST answer is a clean 400 with no new mapping.
Side benefit: `input=None` (possible from a non-pydantic caller) now also yields the 400 rather
than a downstream `TypeError`.

### m-B — **closed** ✔

`tests/test_executor_process.py:344` — `def _publish(svc, *, steps, transitions, key=…)`, the
default removed, with a comment naming O-6. All **nine** call sites (`:364, 380, 403, 422, 441,
463, 480, 496, 514`) pass `transitions` explicitly, so the signature change is inert today and
purely preventive — which is what was asked.

### m-C — **closed** ✔, and §4.4's anti-drift property **survives the override**

I verified the anti-drift claim by reading, not by trusting the comment:

- `test_process_flow.py:53–54` — `VERSION = "v1-test"`, `TEST_DEF = {**ACCESS_REQUEST_DEF,
  "version": VERSION}`. **Only the version key is overridden**; `KEY` still comes from the
  constant (`:45`), and the fixture publishes `**TEST_DEF` (`:79`) then materializes
  `key=KEY, version=VERSION` (`:80`), and every run starts from that snapshot (`:86`).
- Consequence: a guard-`op` mutation in `proof_defs.py` still flows constant → publish →
  `validate_cmp` → snapshot → the three §4.3 path tests. A structurally invalid op fails at
  publish (fixture error, whole module red); a *valid but wrong* op changes the branch a run
  takes and fails the path assertions. The drift-proof property is intact.
- The production pair is pinned at `test_process_flow.py:301` (`(key, version) ==
  ("access-request", "v1")`). What is *lost* is the proof that the production pair itself
  publishes — negligible, since the spec is identical apart from a version string that is not
  semantically interpreted anywhere. Worth naming in the U5 note as a conscious trade, which
  the fixture docstring already does.
- The wording fix is correct and now distinguishes the two mechanisms properly:
  `AGENTS.md` seed row ("`test_queries.sh` deletes `reference` at **teardown** … `wf_repo`
  wipes `reference` at fixture **setup**, once per workflow test, so a finished pytest session
  *leaves the last workflow test's defs behind*") and the same in the script header
  (`scripts/seed_workflows.sh:56–63`). I re-checked `tests/conftest.py:85` — setup-time wipe,
  per test; the corrected wording matches the code.
- Collision check: the other workflow tests publish/materialize under version `"1"`
  (`test_process_input._materialize`, `test_executor_process._publish`), not `"v1"`, so nothing
  in the estate now writes the production `(access-request, v1)` pair into `reference`. The
  script's parenthetical ("anything else published by a test can") is correctly hedged.

### n-A — **closed** ✔

`proof_defs.py:49–53` — the five-line warning that the constant's key set *is*
`publish_workflow_def`'s keyword signature, with the correct remedy (a module-level constant,
"the way `ACCESS_REQUEST_MAX_STEPS` is").

### n-B — **not actioned, and that is right** ✔

The finding was explicitly conditional ("*if* a second boundary→service constant import
appears, move both to `config.py`"). U4b adds exactly one import to `services.py` — stdlib
`logging` — so the trigger condition did not fire. Acting now would be speculative churn
against a still-single import. Leave it as the standing note it was written to be.

## 2. New findings introduced by the fixes

### r-1 · minor — the O-6 guard is **publish-only**; `materialize_def` reaches the same collapse, and one existing test now asserts the opposite of the new invariant

**Evidence.** `services.materialize_def` (`services.py:655`) reads the def subgraph and calls
`repo.materialize_snapshot`, which reuses the *same* `_PUBLISH_CYPHER` shape
(`repository.py:1397`) — same trailing `UNWIND $transitions`, same `result_set[0]`. It performs
**no** spec validation. `read_def_subgraph` → `_read_subgraph` (`repository.py:976–997`) returns
a dict (not `None`) whenever the root `WorkflowDef` node exists, with `transitions = []` when the
transitions query yields nothing — which is exactly the shape a *pre-U4b half-written def* has.
So a def poisoned before this fix is still an unrepairable 500 on materialize, and the new rule
cannot see it. Concretely inconsistent artefacts left behind:

- `tests/test_services.py:916–925`
  (`test_materialize_def_two_phase_reads_reference_then_writes_workspace`) seeds
  `repo.defs[...] = {… "transitions": []}` and asserts materialize **succeeds**. It passes only
  because `FakeRepo` has no Cypher; against the real repository that same input is the 500 this
  slice just outlawed. The test isn't wrong for what it proves (two-phase read→write), but its
  fixture is now the one shape the codebase declares illegal.
- `tests/test_process_input.py:88–91` — the `_materialize` helper's comment still reads "a def
  with zero transitions trips the latent empty-`UNWIND` collapse … (out of U3's scope)". Post
  U4b that is stale in a misleading direction: the collapse is no longer latent on publish, but
  this helper calls `materialize_snapshot` **directly**, which is precisely the path that is
  *still* unguarded.
- `falkorchat/proof_defs.py:26–29` describes the constraint as "a zero-transition publish raises
  `IndexError` *after* the steps are written" — the pre-fix behaviour. It now raises
  `WorkflowDefSpecError` before any write.

**Why it matters.** Low likelihood (materialize is fed by publish, which is now guarded) but the
docs and one test now say three different things about the same invariant, which is how the next
author concludes the guard is symmetric when it isn't.

**Suggested improvement (U5, cheap).** (a) Two comment refreshes — `proof_defs.py:26–29` to name
`WorkflowDefSpecError` at publish, `test_process_input.py:88–91` to say the guard is
publish-side only and `materialize_snapshot` is still raw. (b) One sentence on the
`test_services.py:916` fixture noting it is a FakeRepo shape that the real query would reject.
(c) Optionally a `if not sub["transitions"]: raise WorkflowDefSpecError(...)` in
`materialize_def` — one line, turns the residual 500 into a 400 — but it is defensible to leave
it to the r-2 filing instead.

### r-2 · minor — the accepted trade-off (a single-step / transition-less def is now unpublishable) is stated as a **modelling rule**, never as a **limitation**; it deserves a U5 backlog filing

**Evidence.** All four new doc sites (`services.py:604–612` comment, `:624` docstring, `:523`
docstring, `AGENTS.md` bullet) say the same thing: "a terminal outcome is a step with no
*outgoing* transition, never a def with none." That is the *workaround*, correctly stated. What
none of them records is that (i) a genuinely single-step def is a legitimate shape that this
rule now rejects outright, and (ii) the alternative fix exists and is known — guard
`_PUBLISH_CYPHER`'s trailing `UNWIND` the way the §4 mention write-block guards its own
(`UNWIND (CASE WHEN $x = [] THEN [null] ELSE $x END)`), a pattern this codebase already relies
on and which `AGENTS.md` documents as load-bearing.

**Ruling on the brief's question 6:** yes, this deserves a backlog filing. The trade-off is
*documented as behaviour* but not *recorded as debt*, and the debt has a named, cheap, already-
proven remedy that only needs a graph-dba gate. Without a `K-` number, the next person who
needs a one-step def will rediscover the rule and either fight it or bypass validation.

**Suggested improvement.** File in `docs/BACKLOG.md` during U5 (one line, e.g. *"K-030 — allow
zero-transition defs: guard `_PUBLISH_CYPHER`/`materialize_snapshot`'s trailing `UNWIND` with
the §4 empty-`UNWIND` `CASE` pattern, then relax `_validate_def_spec`'s O-6 rule; graph-dba gate
+ re-PROFILE"*), and fold r-1's residual materialize gap into the same item.

### r-3 · nit — one of the two new `test_process_flow.py` assertions is vacuous

`test_process_flow.py:305` — `assert TEST_DEF == {**ACCESS_REQUEST_DEF, "version": VERSION}`
re-evaluates, verbatim, the expression that defined `TEST_DEF` at `:54`. It can only fail if the
constant is mutated at run time. The assertion one line above (`:301`, the production `(key,
version)` pair) *is* a real pin and should stay; this one is documentation wearing an `assert`.
Same class as the two tautological assertions flagged in the U0–U4 gate §1 — harmless, but it
inflates the apparent pin count.

### r-4 · nit — the `AGENTS.md` seed row still contains the sentence m-C objected to

The corrected explanation is now at the top of the row, but ~5 lines later the same paragraph
still reads "`test_queries.sh` and `server/tests`' `wf_repo` fixture **both wipe** `reference`
but not `ws:<id>`". Not false, but it is the exact phrasing that produced the wrong mental model,
now sitting downstream of its own correction in a row that has grown to ~25 lines. One clause
(`— by the two different mechanisms above —`) or a cut would settle it. U5 doc pass.

### r-5 · nit — the seed script's `access-request` / `v1` defaults still duplicate the constant

`scripts/seed_workflows.sh:89–90` hardcodes the shell defaults, and `:258–260` splats
`**ACCESS_REQUEST_DEF` while *overriding* key and version from the environment — so the seed
never reads the constant's own `key`/`version`. The duplication is now pinned on one side by
`test_process_flow.py:301`, which is a real improvement over U4, but the pin is a literal, kept
in sync by hand. Acceptable; noting it so the next `key`/`version` bump touches all three.

## 3. Mutation-claim spot-check (read, not re-run)

The "7 mutations / 0 survivors" claim holds for the mutations I could name from the code. What
each new test actually kills:

- **`.exception` → `.error`** — killed. `test_process_input.py:539–540` asserts
  `faults[0].exc_info is not None` **and** `faults[0].exc_info[0] is NotImplementedError`. A
  plain `.error(...)` produces a record with `exc_info = None`, so the stack — not merely the
  message — is what is pinned. (A mutation to `.error(..., exc_info=True)` would survive, but
  that is behaviourally identical, so it is not a real survivor.)
- **Delete the log line** — killed by `assert faults, "…produced no log record"` (`:536`).
- **Drop the run id from the message** — killed by `assert out["runId"] in
  faults[0].getMessage()` (`:538`), which also pins the lazy `%s` interpolation.
- **Envelope regression while adding the log** — killed by `assert out["status"] == "failed"`
  (`:534`) in the same test, plus the pre-existing D-G envelope tests.
- **Delete the O-6 rule** — killed by `test_services.py:855` (raises + `repo.published == []`).
- **Move the O-6 rule earlier / raise after the write** — killed respectively by
  `test_services.py:869` (must fail on the start count) and by `repo.published == []` in both
  new tests.
- **Delete the empty-input rule, or move it after the CAS** — killed by
  `test_process_input.py:687`: `pytest.raises(match="no input submitted")` **plus**
  `stepCount == before`, `ctx == {}`, `status == "waiting"`, and a follow-up real submit that
  reaches `done` — i.e. it proves the rejection was free *and* non-poisoning, not just that an
  exception was raised.
- **m-B (the removed `transitions` default)** is the one item no test can kill — a mutation
  reinstating the default is invisible while all nine call sites pass the argument. That is
  inherent to a lint-grade guard, not a gap.

No new `pytest.raises` is vacuous: all three carry `match=`, and in each case the asserted
message belongs to the rule under test (I checked the message strings against the raise sites at
`services.py:605`, `:834` and the start-count raise at `:559`).

## 4. Blast radius on the existing estate

Checked the whole `git diff` for the four test modules: the only edits to pre-existing tests are
(a) the `_publish` signature at `test_executor_process.py:344` (nine call sites already
compliant, no assertion touched) and (b) the `test_services.py:1113` fixture gaining a step and
a transition (assertion unchanged, and it still proves the `agent` step type is accepted). No
assertion anywhere was weakened, deleted or made conditional; no shared fixture (`conftest.py`,
`wf_repo`, `svc`, `_materialize`, `_parked_run`) changed behaviour. The `test_process_flow.py`
fixture change is a version-only override, analysed above. `executor.py` is docstring-only —
`_drive_loop` is not in the diff at all, consistent with the coordinator's SHA verification.

## 5. Not verifiable statically / deliberately not checked

- **Both suites.** Per the brief I ran neither. The reported **533 passed / 1 deselected**,
  **256/256**, the `_drive_loop` SHA `71055f756280`, and both defs' presence in `reference` +
  `ws:acme` are the coordinator's evidence, not mine. Note the +4 tests vs the U4 gate's 529
  matches exactly the four added here (`test_services` ×2, `test_process_input` ×2).
- **The log actually reaching a deployed handler.** `caplog` proves the record is emitted with
  `exc_info`; whether uvicorn's configuration surfaces `falkorchat.services` at ERROR in
  production is a deployment question (K-027's territory) and was not checked.
- **Mutation testing** was not re-run; §3 is a read of the tests against mutations I named.
- Whether `seed_workflows.sh` prints `already present — no-op` for both defs after this change
  (structurally unaffected — U4b changed only its header comment).
