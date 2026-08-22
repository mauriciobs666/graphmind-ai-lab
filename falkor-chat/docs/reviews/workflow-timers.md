# Workflow timers / scheduled wakeups — Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-028

## Scope & verdict

Plan-gate (pre-implementation, static) review of `falkor-chat/docs/plans/workflow-timers.md`
against `falkor-chat/docs/BACKLOG.md`'s K-028 section, `falkor-chat/docs/DESIGN.md` §6.1–§6.3,
`falkor-chat/docs/QUERIES.md` §12 (§12.2–§12.4, §12.9, §12.13, §12.15), `falkor-chat/AGENTS.md`'s
rules 1–8, and the real source (`executor.py`, `services.py`, `repository.py`, `app.py`,
`config.py`, `schemas.py`, `api.py`, `proof_defs.py`). Every file:line citation the plan makes was
checked against the live source, not taken on faith; every claimed query in `QUERIES.md` §12 was
read against the plan's paraphrase. I did not execute the (not-yet-written) test suite — there is
none to run yet — so "test strategy" findings below are static soundness checks on what the plan
proposes, not a run.

**Verdict: needs changes.** The central "derive dueness fresh, no `wakeAt` write" design choice is
sound — I traced the self-loop-impossibility claim through `_drive_loop` myself and it holds (see
"What's solid"). But three other findings are real and load-bearing enough that the plan should not
go to implementation as written: an unaddressed unbounded-churn risk in the sweep's resume
mechanics (Finding 1), a scope call that the backlog's own text contradicts — a genuine fork per
the brief's own instruction to flag one explicitly (Finding 2) — and a test-strategy gap that would
make the plan's flagship "injected clock" test not actually test what it claims (Finding 3). None
of these requires re-deriving the plan's core design; all three are fixable within the plan's own
constraints (a publish-time invariant, a scope decision, and a test-construction detail).

`CPG: considered, not relevant — cpg_falkorchat exists but is stale (built 2026-08-17T00:40:42Z, 6
commits landed on server/ since, per the coordinator's pre-check); every structural claim below was
verified by reading executor.py/services.py/repository.py/app.py/config.py/schemas.py/api.py/
proof_defs.py directly, matching the plan's own CPG note in its §1.`

## Findings

### Major 1 — The sweep can trigger unbounded resume→re-park churn on a `wait` step with no default transition, defeating the step-budget safety net the executor documents as relying on "a parked run cannot self-drive"

**Evidence.** `executor.py:492-500` (OUTCOME B, suspend) is explicitly exempted from the `maxSteps`
budget check, and the comment says why: *"No budget check here by design: the intake loop is
human-paced... a parked run cannot self-drive"* (`executor.py:493-496`). `_record`
(`executor.py:995-1020`) writes a **new** `StepRun` and bumps `stepCount` on **every** pass through
the loop — including a re-park — regardless of outcome (called unconditionally at line 481, before
branching). K-028 makes the exempted assumption false: `sweep_due_workflow_runs` calls
`executor.resume()` automatically, with no ctx write (plan §3.5 step 5), which only advances the
run if some **other**, already-satisfied guard fires. If the `wait` step's only outgoing transition
is a conditional guard tied to the human/system signal the timer is supposed to be a fallback for —
and the step declares no unconditional (`guard: ""`) default arm — a sweep-triggered resume just
re-executes `_run_wait_node`, no guard fires, and OUTCOME B fires again: a fresh `StepRun`, a fresh
`LAST_STEP_RUN`, `stepCount` bumped, **no budget check**, and the run re-parks at `waiting`.

This is not hypothetical: `proof_defs.py:106-110`, the **one shipped `wait` step in this
codebase** (`provision` in `access-request@v1`), has exactly this shape — a single conditional
transition (`ctx.provisioned == true`), no unconditional fallback arm (confirmed by reading
`proof_defs.py`'s transitions list, `:118-149`). If that step declared a `waitUntil` timer, every
sweep tick after the deadline passes would resume-and-immediately-re-park it, forever — because
`dueAt = waitUntil` (a fixed absolute value, plan §3.3) does not change on re-park the way
`waitForSeconds`'s relative `dueAt` does. Concretely: the sweep would call `executor.resume()` on
this run **every ~30s, forever**, each call writing a new `StepRun` node (unbounded growth,
contradicting the plan's own "zero new RAM cost" framing in §3.4/§4, which only accounts for the
two new `config` keys and never considers this churn) and burning a full CAS+drive cycle for no
effect. Even in the `waitForSeconds` (relative) case, this isn't harmless: because `_record` always
refreshes `LAST_STEP_RUN` on re-park, `parkedAt` resets to "now" every time the sweep fires and
fails to advance, so a relative timer silently turns into a **repeating poll interval** at period
`waitForSeconds`, not the one-shot deadline the def author's `waitForSeconds` name and the plan's
own §3.3 phrasing ("relative duration... from the moment the run parks") promise.

**Why it matters.** The plan's §8 risk list discusses the O(waiting-run-count) *read* cost and the
CAS-race safety, but never this orthogonal risk: an *automated*, unattended actor now exercises a
code path (OUTCOME B) whose own docstring explicitly assumes no automated actor exists. The result
is silent, unbounded per-run audit-trail growth for a plausible (indeed, the only currently-shipped)
`wait`-step authoring pattern, with no budget bound and no plan-side test that would catch it (§6
test 4's fault-isolation/CAS-contention cases don't cover "due run whose guard never fires and has
no default arm").

**Suggested fix.** Cheapest, and consistent with the plan's own "no `_drive_loop` edit" constraint:
add a fifth publish-time invariant to `_validate_def_spec` (`services.py:948-958`'s neighbourhood,
same "deliberately last" discipline) — a `wait`/`human` step declaring `waitForSeconds`/`waitUntil`
must also declare at least one unconditional (`guard: ""`) outgoing transition, so a sweep-triggered
resume is *guaranteed* to advance rather than possibly no-op forever. If the team prefers not to
force this (mirroring the already-accepted, deliberately-unenforced K-029 residual for `decision`
self-loops), the plan must at minimum add this to §8 as a named, undismissed risk rather than
leaving it undiscovered — the current §8 explicitly claims the scheduler has "no state to
disagree about" and is "provably" not a risk beyond the CAS race, which this finding shows is not
the whole picture.

### Major 2 — Genuine product-behavior fork: excluding `human` steps from timer scope is not backed by the backlog's own text, which is exactly what a plan gate is asked to catch

**Evidence.** `docs/BACKLOG.md:635-642` (K-028 "Why it exists") motivates the entire item with:
*"an SLA/escalation step ('if no approval in 48h, escalate') cannot be expressed today."* That
example is an **approval-with-a-deadline** scenario. In this codebase's own shipped def, "approval"
is a `human`-typed step (`proof_defs.py:90-98`: `"key": "approval", "type": "human", ...
"assignee": "manager"`, parked awaiting a manager's approve/reject) — **not** the `wait`-typed
`provision` step (a system signal, `proof_defs.py:106-110`). The plan's own §1/§8 defends scoping
to `wait` only by citing *"the backlog's own explicit '`wait` steps today are mechanically
identical to `human`... K-028 needs...' framing."* I traced that quoted framing: it does not appear
in the K-028 backlog section at all. The actual source is `DESIGN.md:486` / `BACKLOG.md:298`
("`wait` is signal-driven, not timer-driven, and mechanically identical to `human`") — both of
which describe **`wait`'s pre-K-028 baseline behavior** (in a K-025 QA handoff note and a workflow
DESIGN section respectively), not a scope boundary for what K-028 itself should build. Nothing in
the backlog item's own text restricts the feature to `wait`; its single motivating example is a
`human`-step scenario.

**Why it matters.** This is precisely the situation the brief asks me to flag rather than
rubber-stamp: the plan's scope call, if wrong, means K-028 ships without actually satisfying the
backlog item's own stated reason for existing (an SLA-escalation on a `human` approval step still
cannot be expressed — the gap the item was filed to close remains open). The plan itself concedes
in §8 that broadening scope is nearly free ("the mechanism generalizes cleanly... the sweep's
app-side filter... would just stop excluding `stepType == 'human'`... the publish-time check...
would drop its type restriction"), and mechanically `human`/`wait` are identical parking paths
(`WAITING_STEP_TYPES`, `services.py:69`; both mandate `config.waitsForHuman: true`; both suspend
via the identical OUTCOME B). There is no technical reason for the narrower scope — only a reading
of the backlog that the backlog's own words don't clearly support.

**This is a genuine fork, not a low-risk design call — surfaced per the brief's own instruction.**
Recommend one of: (a) pause back to the stakeholder to confirm `wait`-only is really what's wanted
given the SLA-escalation motivation reads as `human`-scoped, or (b) since the plan's own analysis
shows extending scope costs almost nothing, simply drop the `wait`-only restriction now and timer-
enable both `WAITING_STEP_TYPES` members, closing the ambiguity by building the broader (and,
per the backlog's own example, more clearly requested) capability.

### Major 3 — The proposed "injected clock" test (§6 test 4) doesn't account for `Services` and `WorkflowExecutor` holding two independent clocks; as described it would not test what it claims to

**Evidence.** `Services.__init__` (`services.py:530-543`) and `WorkflowExecutor.__init__`
(`executor.py:319-345`) each take their **own** `clock: Callable[[], int]` parameter, defaulting to
two **separately defined** module-level `_default_clock` functions (`services.py:149`,
`executor.py:99` — confirmed these are two distinct function objects, not a shared one). Nothing in
`_build_default_app` (`app.py:299-315`) or anywhere else wires one into the other — the executor
receives no `clock=` argument in production wiring either, so both simply call their own real-clock
default independently. `StepRun.startedAt` (the sweep's `parkedAt`) is minted by the **executor's**
clock inside `_record` (`executor.py:1008`, `self._clock()` → the executor's own monotonic
wrapper); `sweep_due_workflow_runs`'s `now` (plan §3.5 step 2) comes from `self._clock()` on
**`Services`**. The plan's §6 test 4 says to "build `Services(wf_repo, clock=lambda: FIXED_NOW)`...
drive it to `waiting`... call `sweep_due_workflow_runs` with the fixture's `_clock`" — but driving a
run to `waiting` requires a `WorkflowExecutor` wired via `services.set_executor(...)`
(`services.py:561-563`), and the plan never says that executor must be constructed with the exact
same clock callable. Every existing precedent for building a test executor
(`test_executor_process.py:53-90`'s `_make_executor` helper) gives it its **own** independent clock
(`itertools.count(1000)`), entirely disconnected from any `Services` instance.

**Why it matters.** If the implementer follows the existing `_make_executor` pattern (the plan's
own cited precedent) while also following §6 test 4's `Services(clock=lambda: FIXED_NOW)`
instruction, `parkedAt` (from the executor's clock) and `now` (from `FIXED_NOW`) become two
unrelated numbers — plausibly off by ~10^9ms if the executor defaults to real wall-clock time while
the test picks a small literal like `1000`. The "before due"/"after due" assertions the plan
describes as the concrete proof of "an injected clock driving the sweep, no real sleep anywhere"
would then either always trivially pass/fail for the wrong reason, or force the implementer to
discover this wiring gap by trial and error. This is exactly the kind of test-seam claim the brief
asked me to verify rather than assume.

**Suggested fix.** Add one sentence to §6 test 4: construct the test's `WorkflowExecutor` with the
**same** `clock=` callable passed to `Services` (a single shared mutable counter or `lambda:
FIXED_NOW`/its stepped variant), not the independent counter `_make_executor` uses elsewhere — since
this test specifically needs `parkedAt` and `now` to be comparable, unlike every prior executor test
that only cares about internal ordering.

### Minor 4 — Doc-curation list (§7) omits `scripts/start_server.sh`'s header comment, the established sibling location for every other workflow env var

**Evidence.** `falkor-chat/AGENTS.md`'s own scripts table states: *"Runtime env vars are documented
in the script's own header comment"* (for `start_server.sh`). Verified: the script's header
documents `FALKORCHAT_WORKFLOW_ENABLED`, `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`,
`FALKORCHAT_ENABLE_AGENT`, etc. (`scripts/start_server.sh:8-40`), and the script actively
`${VAR:-default}`s, `export`s, and echoes each of them at startup (`:88`, `:154`, `:172`, `:182-183`
for the workflow-related ones specifically). The plan's §7 documentation-updates list covers
`QUERIES.md`, `DESIGN.md`, `AGENTS.md`, `BACKLOG.md`, `HISTORY.md`, but not this script, even though
the new `FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S` is a direct sibling of the env vars already
documented there.

**Why it matters.** Not a correctness gap (config.py's own default of `30` applies whether or not
the script mentions it) — a completeness gap in the implementer's stated done-condition, given the
brief specifically asked me to check this list against what the plan actually changes.

**Suggested fix.** Add a one-line header-comment entry (and optionally a startup echo, matching the
existing "Workflow: enabled=..." line at `start_server.sh:172`) to §7's checklist.

## What's solid

- **The central "derive dueness fresh, no `wakeAt` write" design choice holds up.** I independently
  traced `_drive_loop`'s three outcomes (`executor.py:471-510`): a `wait`/`human` step's mandatory
  `config.waitsForHuman: true` (enforced at publish, `services.py:948-958`) means it can only ever
  reach OUTCOME A (a guard fired — advances away) or OUTCOME B (suspend) on any given pass; it can
  never reach OUTCOME C (self-loop on the same step), because OUTCOME C is reached only when
  `config.get("waitsForHuman")` is falsy. So the plan's claim that `StepRun.startedAt` via
  `LAST_STEP_RUN` is always fresh for the *current* park, with no stale prior-iteration `StepRun` to
  worry about, is correct.
- **The CAS-reuse claim is accurate and verified against `QUERIES.md` §12.4/§12.9/§12.13.** No new
  resume Cypher, no new resume semantics — `resume_run` (§12.4, `repository.py`) is byte-identical
  to what the plan quotes, and the sweep calls the same `executor.resume()` entry point
  `resume_workflow_run` already calls (`services.py:1667-1679`).
- **RAM/index claims check out.** `bootstrap_schema.sh:145-156` confirms `WorkflowRun.status` is
  already indexed; the proposed `find_due_wait_candidates` query's traversal shape
  (`status`-anchored MATCH → `AT_STEP` → `LAST_STEP_RUN`) matches `get_run`'s existing shape
  (`QUERIES.md` §12.7) exactly — no new index is needed for the stated reasons.
- **Batch fault isolation reasoning is correct.** Traced `_drive`'s catch-all (`executor.py:440-449`,
  catches *any* `Exception`, stamps `fail_run`, re-raises) against `_drive_or_fault`'s four named
  types (`services.py:1636-1666`) — confirmed every exception, named or not, is already fail-run-
  stamped before the sweep's own broad `try/except` around step 5 would ever see it, so the plan's
  "the sweep's outer catch only needs to stop Python propagation" claim is accurate.
- **Every file:line citation I spot-checked was accurate** — `repository.py` (`_PUBLISH_CYPHER`,
  `find_waiting_run_for_thread`), `services.py` (`_validate_def_spec`, `_drive_or_fault`, clock
  injection, module-constant precedent), `app.py` (`create_app`/`_build_default_app` line ranges,
  `WORKFLOW_ENABLED` gating), `config.py`, `api.py` (no-`{ws}`-segment routing), `schemas.py`
  (`MAX_ID_LEN`/`MAX_CONFIG_LEN`), and the `python-web-quirks` skill citations (bounded
  `run_in_threadpool`, `create_task` GC risk) all matched the real source. This is a well-grounded
  plan, and the effort that went into verifying its own claims shows.
- **The sweep default values (30s interval, 200/1000 limits) are a reasonable, low-risk design
  call, not a fork.** The backlog states no SLA-precision requirement; both are
  environment-overridable, and the plan is honest that they're guesses rather than derived numbers
  (§8). Approve as-is.

## Open questions

- Finding 2 (the `human`-step scope call) needs a stakeholder decision or an explicit plan revision
  before implementation — it is the one item in this review that fits the brief's "genuine fork"
  criterion, since the backlog's own primary example points at the excluded scope.
- Finding 1's mitigation (a publish-time default-arm invariant) is my recommendation, but the team
  may reasonably prefer the K-029 precedent instead (accept and document the residual risk rather
  than enforce it) — either is acceptable, but the plan must pick one explicitly rather than leave
  the risk undiscovered, which is the actual defect here.

## Pass 2 — 2026-08-21

Re-gate of v2 (`Status: active`, `Version: 2`), revised in place against Pass 1's three majors and
one minor. Read the full v2 document myself (not the revision note) and re-traced every claim
against the live source; I did not re-verify the Pass-1-confirmed-sound parts (core
dueness-derivation, CAS-reuse, RAM/index accounting, batch-fault-isolation) since v2 does not touch
that reasoning. Scope: did the four Pass-1 findings actually get closed, plus a fresh look for
anything the fix itself might have newly broken (teco's explicit ask).

**Updated verdict: approve with suggestions.** All three majors and the minor from Pass 1 are
genuinely closed, verified by re-tracing each fix through the real source, not by trusting the
revision note. Pass 2 surfaces one new major and one new minor — both are consequences of the fix
for Finding 1, not reopenings of anything from Pass 1 — and neither rises to blocker: they're a
documentation gap and an implementation-precision gap, not a functional defect in the mechanism
that ships. Recommend the plan take both suggestions before implementation starts (cheap: a
paragraph and a one-line clarification), but they do not warrant a third gate cycle.

### Finding 1 (unbounded churn) — CLOSED, verified

Re-traced independently, not taken on the plan's word: `evaluate_guard` (`guards.py:223-224`,
`if not guard: return GuardVerdict(decision=True, ...)`) always fires an unconditional guard when
reached and nothing higher-priority already fired; `_select_transition`'s sort
(`executor.py:974-975`, `(t["guard"] == "", t["order"], t["to"])`) evaluates conditional guards
first. So once a timer-bearing step is required to carry an unconditional fallback transition
(v2 §3.3's new fifth invariant), **any** sweep-triggered resume that doesn't satisfy the "real"
conditional guard is now *guaranteed* to fire the default and advance (OUTCOME A) — it can no
longer fall through to OUTCOME B and re-park. This closes both halves of the original finding: the
`waitUntil` forever-re-park case (now a single guaranteed resume, not infinite) and the
`waitForSeconds` silently-becomes-a-poll-interval case (now guaranteed to advance on the very first
due tick, `parkedAt` never gets the chance to reset). The fix is threaded consistently: §3.3 states
the invariant and its rationale, §5 step 1 tells the implementer how to wire it (transitions grouped
by `from`, already an available parameter), §6 test 2 adds the rejection case plus a regression
guard (a def declaring neither key must still publish with no default arm — verified against
`proof_defs.py`'s shipped `provision`/`approval`, neither of which declares a timer key today), and
§8 documents the closure with the correct reasoning (no natural budget backstop existed for this
path, so enforcement — not the K-029 "document only" alternative — was the right call given "worse
than the self-loop case," which I agree with: the self-loop case at least exhausts `maxSteps`
eventually, this one didn't).

### New Major — the default-arm requirement forecloses an existing, documented "not yet, keep
### waiting" resumable-non-advancing signal on any step that adopts a timer

**Evidence.** `proof_defs.py:106-110`'s comment on the shipped `provision` step is explicit and
deliberate: *"Deliberately no `expects`, so `{"provisioned": false}` ('not yet') stays expressible:
it re-parks and costs one step."* This is a real, shipped pattern — a human or system can resume a
parked run with a negative/not-yet signal and the run **stays parked** (OUTCOME B fires again,
because no guard matched). The new v2 invariant requires any step that adopts a timer to also carry
an unconditional fallback transition. But `evaluate_guard`'s unconditional-fires-when-reached rule
(`guards.py:223-224`) does not know *why* a resume happened — a sweep tick and a human's explicit
`{"provisioned": false}` "not yet" nudge are, by this design's own stated and load-bearing property
(§3.5's "the sweep is not a distinguishable, special-cased resume path from the CAS's point of
view," restated as a *feature* in the §6 test-4 CAS-contention test), **indistinguishable resume
attempts**. Once `provision` (or any step following this exact pattern) also declares a timer and
therefore must carry a default arm, the very next `{"provisioned": false}` resume — meant to keep
the run waiting — instead falls through to the unconditional guard and **advances the run into the
timeout/escalation branch**, exactly as if the timer had actually fired. There is no guard-language
construct available (`ctx`/`step_output` only, per `guards.py`'s module docstring) that can express
"fire only when nothing at all was submitted, not when something-but-not-yet was submitted" — the
guard evaluator sees the same `ctx` regardless of who called `resume()` or why.

**Why it matters.** This is a real, non-obvious semantic consequence of the very fix that closes
Finding 1, not a re-opening of it: the mechanism is otherwise correct, but a def author who
combines "explicit not-yet, please keep waiting" (the `provision` pattern) with a timer/escalation
fallback on the *same* step will silently lose the not-yet behavior the moment they add the timer
key — with no error, no warning, just a run that now escalates on the first not-yet reply instead of
staying parked. This doesn't block the backlog's own primary motivating example (the `approval`
step's `expects: {"decision": ["approve","reject"]}` already rejects any other value at 400 before
reaching the executor — `services.py` D-H rule 3 — so there is no pre-existing "not yet" pattern on
that step to break), but it is a foreseeable landmine for the next def author who reaches for both
patterns together, and the plan currently doesn't mention it anywhere.

**Suggested fix.** Not a redesign — document the trade-off explicitly, in the same place §3.3
documents the invariant's rationale and again in §8's risk list: a step that declares a timer key
gains "fire the real condition, or fall through to the timeout branch" semantics and **loses** the
ability to also treat an explicit-but-negative resume as "still not ready, stay parked" — the two
patterns are mutually exclusive on the same step under this design, because the sweep and a human
reply are deliberately indistinguishable at the guard-evaluation layer. A def wanting both would
need to route the "not yet" case to a real ctx-observable branch (e.g., an explicit intermediate
step) rather than relying on silence/re-park.

### New Minor — the invariant's "guard normalizes to `""`" check should be specified against
### `_serialize_opaque`, not `_normalize_opaque`, to match what the runtime actually compares

**Evidence.** `_serialize_opaque` (`services.py:159-170`) maps `None` → `""` (the literal stored
value); `_normalize_opaque` (`services.py:189-207`) does **not** — a `None` guard normalizes to
`None`, not `""` (it only special-cases strings; anything else, including `None`, "passes through
unchanged"). The runtime check that actually decides "is this transition the default" is
`t["guard"] == ""` on the **post-serialization** graph value (`executor.py:975`). A direct/MCP
caller may plausibly omit `guard` entirely on a transition dict (`tr.get("guard")` is used, not
`tr["guard"]`, at the two existing normalization call sites `services.py:431,992` — implying a
missing key is an anticipated shape), which publishes as `""` (via `_serialize_opaque`, matching the
runtime's idea of "unconditional") but would **not** be recognized as satisfying the new invariant
if the implementer follows the plan's own phrasing literally and checks
`_normalize_opaque(tr.get("guard")) == ""` (as the existing cmp-guard check two lines above it in
`_validate_def_spec` does, `services.py:992`) — `_normalize_opaque(None)` is `None`, not `""`. A def
whose only default arm is expressed by omitting `guard` (rather than writing `guard: ""` explicitly)
would then be wrongly rejected as "no unconditional transition," even though it publishes one.

**Why it matters.** A narrow, single-line ambiguity in an otherwise well-specified fix — but it's
specifically the kind of gap that survives to implementation unnoticed, since `proof_defs.py`'s own
transitions always write `"guard": ""` explicitly (line 133-134) rather than omitting the key, so
existing tests wouldn't catch a wrong choice here either way.

**Suggested fix.** §5 step 1 should say explicitly: check `_serialize_opaque(tr.get("guard")) == ""`
(matching what the graph will actually store and what the runtime actually compares), not
"`guard` normalizes to `""`" — or, equivalently, treat both `None` and `""` as satisfying the
unconditional-arm requirement.

### Findings 2, 3, and the Minor — CLOSED, verified

- **Finding 2 (scope fork).** Verified consistent everywhere it matters: §1's goal statement,
  §3.3's config-key scope, §3.5 step 4's app-side filter (`WAITING_STEP_TYPES`, not a hardcoded
  `"wait"` check), §6's test list (test 1's `human` dueness case, test 2's `human` positive/negative
  publish cases, test 4's `human`-typed before/after due-time case), §7's `DESIGN.md` §6.1 update
  instruction (explicitly calls out updating **both** the `wait` and `human` bullets), and §8's
  "Settled, v2" risk entry. I grepped the full document for stray `wait`-only scoping language;
  every remaining single-step-type "`wait` step" mention is a **concrete illustrative example**
  (mostly `proof_defs.py`'s shipped `provision` step, used correctly as "the one step this exact
  shape already exists for"), never a scope restriction — confirmed by reading each occurrence in
  context, not just grepping. No stale phrasing survived the edit.
- **Finding 3 (independent-clocks test gap).** §6 test 4's new opening bullet is unambiguous: it
  names the two independently-defaulting clock functions by file:line
  (`services.py:149`/`executor.py:99`), states plainly that the test must construct **both**
  `Services` and `WorkflowExecutor` with the same `clock=` callable, gives two concrete options (a
  shared mutable counter or a fixed lambda), and explicitly warns the implementer **against**
  reusing `_make_executor`'s existing independent-clock pattern unmodified for this specific test.
  This is exactly the level of explicitness Pass 1 asked for — an implementer would have to actively
  ignore the text to reintroduce the original bug.
- **Minor (doc checklist).** Verified: §5 step 10 and §7 both now list `scripts/start_server.sh`,
  correctly citing the same header-comment convention and the script's existing
  `${VAR:-default}`/`export`/echo shape (`start_server.sh:88,154,172,182-183`, matching what I
  checked in Pass 1).

## Pass 3 — 2026-08-21

Re-gate of v3, triggered by `teco`-routed implementation feedback: `coder` found, and `teco`
independently re-verified before routing here, that v2's Finding-1 fix (a bare unconditional
fallback transition) makes a timer-bearing step **never park at all** — a defect neither Pass 1 nor
Pass 2 caught. This pass does what was explicitly asked: traces the *mechanism* end to end against
live source myself, not a re-check of the sort-order claim alone, and does not approve on the
strength of the revision note.

**Updated verdict: approve with suggestions.** I traced the full v3 mechanism myself — guard
resolution, the reserved-key structural guarantee, the CAS write, and the step-scoping fix — through
the real source, not the plan's narrative. It works: a timer-bearing step correctly parks on first
arrival, correctly re-parks on an ordinary or "not yet" resume, and correctly (and only then)
escalates on a genuine sweep-triggered resume. This is not a re-approval of a claim; it is
independently reconstructed from `guards.py`/`executor.py`/`services.py`, detailed below. Two new,
narrow minors surfaced during that trace — neither reopens anything closed in Pass 1/2 and neither
blocks: a stale-candidate race window that collapses into the plan's own already-accepted "cycle
back" residual rather than creating a new class of harm, and two wrong file:line citations.

### Why v2 broke and why v3's replacement actually works — traced end to end, not asserted

**Root cause of the v2 break, confirmed against source, not just the revision note's claim.**
`_drive_loop` (`executor.py:471-510`) calls `_select_transition` on *every* pass through the loop,
first arrival included — there is no separate "am I resuming" code path. `evaluate_guard`
(`guards.py:223-224`, `if not guard: return GuardVerdict(decision=True, ...)`) fires an unconditional
guard the instant it's evaluated, with no notion of "first arrival" vs. "resume." So a step
satisfying v2's invariant hits OUTCOME A (advance via the unconditional arm) on its very first
evaluation, before `config.waitsForHuman`'s OUTCOME B branch is ever reached (`executor.py:492`,
the `if config.get("waitsForHuman")` check comes *after* the firing check at line 485) — the step
never suspends, so there is nothing for a resume or a sweep to ever act on. I reproduced this by
reading the actual control flow, not by trusting the header note; it is exactly right. **This is
also, precisely, why my own Pass 2 didn't catch it**: I verified the guard *sort order*
(`executor.py:974-975`, conditional-before-unconditional) and the *runtime semantics of an
unconditional guard once a step is legitimately parked* — but never asked whether the unconditional
arm would also fire on the step's *first-ever* evaluation, before any park had happened at all. The
sort order claim was true and irrelevant to the actual bug; the gap was scope, not a wrong trace.
Recording this because it's a real lesson for a future pass: when a fix changes what fires a
transition, check every evaluation site of `_select_transition`, not only the one the fix's own
narrative points at (the resume path) — `_drive_loop`'s first-arrival call site is not called out
anywhere in v2's own text, which is exactly why it was invisible to a review that (reasonably)
followed the plan's own framing.

**v3's marker-guard mechanism, traced independently against live source:**

- `guards.py:464-484` (`_resolve_path`) resolves `"ctx.timerFired"` by walking into the run's `ctx`
  dict; when the key is absent it returns the `_MISSING` sentinel, never raises. `guards.py:147`
  (`"eq": lambda left, value: bool(left == value)`) — `_MISSING == "<any step key>"` is `False`
  (confirmed: `_MISSING` is a fresh `object()`, never equal to a string). This is exactly the
  module's own documented "totality" rule (`guards.py:41-44`) — a missing path is `False` for every
  op, never an error. So the escalation guard is **false at first arrival** (nothing has written
  `timerFired` yet) — the step falls through to `config.waitsForHuman` and correctly suspends
  (OUTCOME B). This is the fix for the actual v2 defect, verified by reading the resolver and the
  op table myself, not by re-running the plan's argument.
- `services.py:1698-1704` (`_reject_reserved_keys`) is called at both `start_workflow_run`'s
  untriggered-ctx path (`services.py:1581`) and `submit_workflow_input`'s input path
  (`services.py:1658`) — confirmed these are the *only* two ctx-mutating entry points a human/API
  caller can reach, and both already reject any key in `RESERVED_CTX_KEYS` before merging anything.
  Adding `TIMER_FIRED_CTX_KEY` to that frozenset (§5 step 1) is therefore a real, structural
  guarantee — not a convention — that an ordinary resume can never write `ctx.timerFired`, confirmed
  by reading the two call sites, not assuming the frozenset addition "just works." So the guard is
  **false on every ordinary or "not yet" resume** (nothing but the sweep can ever set the key) —
  this is what makes v3's fix for Pass 2's "not yet" finding real, not just claimed: I checked the
  enforcement boundary directly, since that finding was specifically about a resume path silently
  breaking, and the fix's soundness rests entirely on that boundary holding.
- `resume_run_with_ctx` (`QUERIES.md` §12.13, `repository.py`, re-confirmed against the Cypher I
  already verified byte-for-byte in Pass 1) writes `r.ctx = $ctx` atomically inside the same
  `WHERE r.status = 'waiting'` CAS as the status flip — so the marker write and the resume can never
  be split, and a losing racer (human or a second sweep tick) writes nothing at all. The escalation
  guard is therefore **true only when the sweep itself won the CAS for this exact run**, and, because
  a `cmp` `eq` guard is deterministic and total (no judge, no LLM), it fires with certainty at that
  point — closing the churn risk Finding 1 named (a guaranteed advance now happens exactly when, and
  only when, the sweep resumes the run, never before, never never).
- **The step-scoping fix is sound.** The guard is `ctx.timerFired == "<this step's own key>"`, and
  `_validate_def_spec`'s existing, unrelated "duplicate step key" rejection
  (`services.py:928-933`, pre-existing, unchanged) already guarantees step keys are unique within one
  def — so a *different* step's stale marker value can never equality-match a *different* step's own
  key. The residual the plan names (a def that cycles back to the *same* step after its own timer
  already fired) is real and correctly scoped as narrower than the leakage case it replaces; I traced
  it myself rather than accepting the characterization, and it holds: only exact re-arrival at the
  literal same `Step` node reproduces it, not any other topology.

**Conclusion: the mechanism works end to end, independently verified.** I did not find a hole in the
central marker-guard design. The two items below are genuine, narrow gaps found while doing this
trace — not the "still doesn't work" outcome the brief asked me to gate hard against, so neither is
a blocker.

### New Minor — step 5.1's "raced" check tests run status, not run *position*; a stale scanned `stepKey` can still be merged after the run has moved on

**Evidence.** §3.5 step 5.1 reads `run = get_run(...)` fresh and buckets **raced** only when
`run is None or run.get("status") != "waiting"`. It does not compare `run.get("atStepKey")` against
the candidate's own `stepKey` from the original `find_due_wait_candidates` scan. Trace: a run parked
at timer-bearing step `X` is scanned as a candidate (`stepKey="X"`); before the sweep reaches step 5,
a human resumes it via `submit_workflow_input`, driving it forward into a *different* `wait`/`human`
step `Y`, which immediately parks (`status` is `'waiting'` again, just at `Y`, with `LAST_STEP_RUN`
now pointing at `Y`'s own fresh `StepRun`). The sweep's step-5.1 read sees `status == 'waiting'` (not
raced) and proceeds, merging `ctx.timerFired = "X"` (the stale scanned key, not `Y`'s) onto the
now-current ctx and winning the CAS (the run genuinely is `waiting` right now). This is not a
spoofing risk — `Y`'s own escalation guard checks `ctx.timerFired == "Y"`, `"X" != "Y"`, so it
correctly does not fire, and by the same step-key-uniqueness argument above this can only ever
misfire if the run later cycles back to literal step `X` — i.e., it collapses into the plan's own
already-accepted "cycle back" residual (§8), just reached through a race window instead of an
authored cycle, plus one wasted CAS win (a pointless resume-and-reparish of `Y`, one step-budget
increment, and a stale `"X"` value left sitting in `ctx.timerFired` until something overwrites it).

**Why it matters.** Not a new class of harm — it widens the trigger surface for a residual the plan
already accepts and documents — but it's a real gap in a check whose own name ("raced") implies it
catches exactly this kind of stale-scan situation, and it wasn't traced in the plan's own text.

**Suggested fix.** Cheap: step 5.1 should also compare `run.get("atStepKey") == candidate["stepKey"]`
and bucket as **raced** on a mismatch, not just on a status change — `get_run` already returns
`atStepKey` (`QUERIES.md` §12.7, no new read). Optional, not blocking; worth a one-line addition
alongside the existing status check.

### New Minor — §5 step 4 cites the wrong lines for `submit_workflow_input`'s ctx-merge shape

**Evidence.** The plan's §2/§5 step 4 cites `services.py:1461,1486-1488` for "`get_run`/
`_load_json_dict`/`_dump_ctx` already used identically by `submit_workflow_input`." Read directly:
those line numbers land inside `check_demo_readiness`'s snapshot-sync problem accumulation (an
unrelated method building a `results.append({...})` list), not the ctx-merge code. The actual merge
this section means to point at is `services.py:1658-1663` (`_reject_reserved_keys` → `_load_json_dict`
→ `.update(input)` → `_dump_ctx`), with `_dump_ctx` itself defined at `services.py:1708`.

**Why it matters.** A wrong citation an implementer would trip over while looking for the pattern to
mirror — low stakes since the surrounding prose describes the shape correctly and the actual method
is easy to find by name, but worth fixing given this whole plan's citations have otherwise held up
well under repeated, independent verification, and this review's standing practice is to check every
citation rather than assume the file:line is right.

**Suggested fix.** Correct the citation to `services.py:1658-1663,1708`.

## Diff Re-gate — 2026-08-21

Post-implementation code review of the working-tree diff (uncommitted; `git status`/`git diff` only,
never a tree-mutating command) delivering K-028 against Pass-3-approved plan v3. Scope: `services.py`,
`repository.py`, `executor.py`, `api.py`, `app.py`, `config.py`, `schemas.py`,
`scripts/start_server.sh`, `docs/QUERIES.md` §12.16, `docs/DESIGN.md` §6.1/§6.2/§6.3, and tests in
`test_workflow_timers.py` (new), `test_executor_process.py`, `test_repository.py`,
`test_process_input.py`, `test_app.py`. I read every changed line (not `coder`'s self-report),
re-ran the suites myself, independently re-verified the SHA-lock and the `GRAPH.PROFILE` claim
against a live instance, and performed my own mutation test rather than trusting the reported one.

**Verdict: approve with suggestions.** The marker-guard mechanism is implemented exactly as v3
specified and works — verified by tracing the code, running the full offline suite, and (per
teco's explicit ask) breaking the escalation-guard check myself and confirming a real def with a
wrong-step-key guard publishes successfully only when the check is gone, never against the actual
code. No blocker, no major. One low-priority nit below.

### What I verified directly, with evidence

- **The marker-guard invariant, as shipped, is exactly the v3 shape.** `services.py`'s new
  `_validate_def_spec` block (after the K-024/K-027 checks, before the K-024 U4b zero-transition
  check, matching the "deliberately last" discipline) requires, field-by-field on the normalized
  guard: `kind == "cmp"`, `path == "ctx.timerFired"`, `op == "eq"`, `value == step["key"]`. Confirmed
  by reading the code, not the plan's paraphrase.
- **`TIMER_FIRED_CTX_KEY = "timerFired"` is in `RESERVED_CTX_KEYS`**, and both ctx-mutating entry
  points (`start_workflow_run`'s untriggered ctx, `submit_workflow_input`'s input) still route
  through the same `_reject_reserved_keys` call unchanged — confirmed by reading both call sites
  and by the new parametrized reserved-key test in `test_process_input.py`
  (`@pytest.mark.parametrize("reserved", ["threadId", "error", "timerFired"])`), which I ran and
  which passes.
- **`sweep_due_workflow_runs` calls `executor.resume(ctx, run_id=rid, run_ctx_json=mj)`** —
  confirmed in the diff, always with a non-`None` `run_ctx_json`, so it always routes through
  `resume_run_with_ctx` (§12.13), never the plain `resume_run`. Matches v3 exactly.
- **The Pass-3 minor (the `atStepKey` race-check) is implemented precisely as I specified it**: step
  5.1's fresh read now buckets **raced** on `run is None or run.get("status") != "waiting" or
  run.get("atStepKey") != step_key` — the third clause is new and exactly closes the stale-scan gap
  I found in Pass 3. `coder` also wrote a dedicated regression test for it
  (`test_sweep_bucket_raced_on_a_stale_scanned_step_key_not_just_status`, using a `_RacingRepo`
  wrapper that injects a real human resume in the gap between the sweep's scan and its per-candidate
  act) — I ran it; it passes, and it genuinely exercises the race (not a sequential no-op): the
  wrapper's `on_scan()` callback fires `submit_workflow_input` moving the run to a different waiting
  step *between* the scan and the fresh read, and the assertion confirms `raced` with no marker
  written and the run correctly left at its new position.
- **The circular-import fix is real and correctly resolves the direction, not just moves the
  problem.** `schemas.py` is confirmed to import nothing from `services.py` (grep-verified: zero
  `from .services` in `schemas.py`), while `services.py` already imports `MAX_CONFIG_LEN`/
  `MAX_DIFF_PREVIEW` from `schemas.py` at module top (pre-existing, unrelated to this diff) — so the
  plan's originally-proposed direction (constants in `services.py`, imported by `schemas.py`) would
  have been a genuine import cycle (`schemas` → `services` → `schemas`) the moment `schemas.py`
  tried to import back. `coder`'s fix — `DEFAULT_SWEEP_LIMIT`/`MAX_SWEEP_LIMIT` defined in
  `schemas.py`, imported into `services.py` — mirrors the *existing* `MAX_CONFIG_LEN` precedent
  exactly and does not deadlock, confirmed by reading both files' import graphs and by the fact the
  module imports cleanly (the whole suite ran, which it could not do with a real cycle).
- **The `_run_wait_node` docstring correction sits entirely outside the SHA-locked `_drive_loop`
  body — verified by recomputing the lock hash myself against the live diff**, not by trusting the
  comment: `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py
  | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12` still returns `71055f756280` on
  the current working tree, byte-identical to the value pinned since M3. This is the strongest form
  of verification available for this specific claim and it holds.
- **CAS-contention, "which branch fired," verified for real.** `test_cas_contention_human_
  resolves_first_the_sweep_never_touches_it` asserts `ctx["provisioned"] is True` and
  `"timerFired" not in ctx` (the domain guard fired, not escalation) after a human wins first, and
  the subsequent sweep call finds zero candidates at all (the run left `waiting` before the sweep
  ever scans, a *stronger* proof of the single-winner property than a bare "raced" bucket — the
  test's own docstring explains why a sequential two-call test can't otherwise reproduce a genuine
  in-flight race, and correctly identifies that the real read-act-gap race needs the `_RacingRepo`
  injection technique used elsewhere). `test_cas_contention_sweep_resolves_first_a_second_human_
  submit_gets_409` asserts the mirror: `ctx["timerFired"] == "park"` (escalation fired, not domain)
  and a second human attempt gets `WorkflowRunNotWaitingError`. Both ran and pass. This is a more
  rigorous design than the plan's literal text asked for (a two-sequential-call mirror of the
  existing repo-level CAS test) — `coder` correctly recognized that scenario doesn't reproduce a
  genuine mid-flight race for the "human first" direction, and built `_RacingRepo` for the case that
  actually needs one.
- **The "not yet" regression and the two-step leakage regression both prove what they claim**, read
  end to end: `test_not_yet_resume_survives_unbroken_on_a_timer_bearing_step` drives a real
  `submit_workflow_input({"provisioned": False})` before the due time and asserts the run stays
  `waiting` at the same step with `"timerFired" not in ctx`, then sweeps it past due and confirms the
  escalation branch fires correctly afterward — composing both patterns on one step, exactly Pass 2's
  finding. `test_step_scoped_marker_does_not_leak_across_different_timer_steps` sweeps step `A` past
  due (advances into `B` via `A`'s own escalation guard, `ctx.timerFired == "A"`), then explicitly
  re-sweeps and asserts `B` stays parked (not an immediate false escalation from the stale `"A"`
  value), and only escalates once `B`'s own due time passes. Both ran and pass.
- **`GRAPH.PROFILE` claim re-verified live, independent of the recorded finding.** I ran `EXPLAIN`
  (this MCP's supported plan-only mode) on the exact `find_due_wait_candidates` query text against
  the real `ws:test` graph on this instance and got `Node By Index Scan | (r:WorkflowRun)` → two
  `Conditional Traverse`s → `Project` → `Limit` — identical shape to what `QUERIES.md` §12.16 records,
  confirming the `s.key AS stepKey` addition is genuinely a free projection with no new anchor, not
  an assumption carried over from before the RETURN-clause edit.
- **Full suite, run myself, not read from a report**: `pytest -q` on the five touched/new test files
  → `288 passed`; the whole server suite → `1528 passed, 3 deselected` (the `-m live` tests), zero
  failures. `ruff check` on every touched source file is clean except one **pre-existing** `I001`
  import-sort finding in `services.py` that I confirmed (via `git stash`) exists identically on the
  pre-diff file — not introduced by this change, and consistent with `DESIGN.md` §14.7's documented
  "ruff not a wired gate" posture.
- **My own mutation test (not `coder`'s reported one), per teco's ask.** I copied the working tree's
  `services.py` into an isolated scratch location (never touching the actual repo), removed the
  `g.get("value") == skey` clause from the new escalation-guard check, loaded the mutated module
  under `falkorchat.services` via `importlib` (real sibling modules unmutated) against a real
  FalkorDB connection, and published a def whose escalation guard names the *wrong* step key. With
  the mutation, `publish_workflow_def` **succeeded** (the bug the check exists to prevent,
  reproduced on demand); against the real, unmutated code the identical call is **rejected** with
  `WorkflowDefSpecError` naming the exact guard shape. Both throwaway probe defs were deleted from
  `reference` immediately after. This directly confirms the shipped check — and by extension the
  parametrized `test_publish_rejects_a_wrong_shaped_escalation_guard[wrong-step-key-...]` test that
  exercises it — is load-bearing, not a check that happens to pass regardless.

### Nit — the sweep's ctx merge doesn't apply the `MAX_CONFIG_LEN` bound `submit_workflow_input` enforces on its own merge

**Evidence.** `submit_workflow_input` checks `len(merged_json) > MAX_CONFIG_LEN` and raises
`WorkflowInputRejectedError` before ever calling the CAS (`services.py:~1709`). `sweep_due_workflow_
runs`'s own merge (`merged[TIMER_FIRED_CTX_KEY] = step_key; merged_json = self._dump_ctx(merged)`)
has no equivalent check before calling `executor.resume(..., run_ctx_json=merged_json)`.

**Why it matters.** Low probability, real in principle: a long-lived run whose ctx has already
accumulated close to `MAX_CONFIG_LEN` (8000 chars) worth of merged human input could, in principle,
be pushed over that bound by the sweep's own `"timerFired":"<stepKey>"` addition, writing an
oversized `ctx` through a path that has no size guard at all, unlike every other ctx-writing path in
this codebase. Not a correctness bug (nothing downstream currently enforces or depends on the bound
at read time) and not something any of the new tests would catch, since none constructs a
near-boundary ctx.

**Suggested fix.** Mirror `submit_workflow_input`'s check in `sweep_due_workflow_runs` before the
`_drive_or_fault` call — on breach, bucket the candidate as `faulted` (or a new, distinct outcome)
rather than writing an unbounded ctx. Low priority; does not block approval.

### Sanity pass on "what didn't change" — confirmed, no new inconsistency found

Per teco's ask (an edit for one fix has already broken something else once already): re-checked the
`find_due_wait_candidates` query's anchor/traversal shape after the `s.key AS stepKey` addition — it
is a projection off the already-bound `s` node (confirmed no new `MATCH`/traversal introduced), so
the `status`-indexed anchor and the `AT_STEP`/`LAST_STEP_RUN` conditional-traverse shape I verified in
Pass 1 stand unchanged. The batch-fault-isolation logic (§3.5 steps 6-7) is untouched from v2 except
for the new step-5.1 pre-check, which only adds an earlier `raced` exit, not a change to the existing
bucketing. The RAM accounting honestly adds the new `ctx.timerFired` marker's cost (§3.4) rather than
silently claiming zero-cost status is unchanged. The `wait`+`human` scope breadth (Pass 2's Finding 2)
is untouched and still consistently threaded — no stale phrasing reappeared in the v2→v3 edit. I did
not find anything the v3 edit disturbed beyond the two minors above.
