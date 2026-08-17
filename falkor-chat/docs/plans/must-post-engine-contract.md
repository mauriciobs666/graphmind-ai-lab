# `must-post-engine-contract` — engine-level "must-post" contract for agent nodes

> **Status:** active · **Owner:** `architect` · **Tracks:** K-027 (M3.5) · **Version:** 2 ·
> **Reviews:** `docs/reviews/must-post-engine-contract.md`

> **Revision note — 2026-08-16 (v1 → v2).** Revised in place against the `analyst` plan-gate
> review (`docs/reviews/must-post-engine-contract.md`, verdict *approve with suggestions*, no
> blockers; grounding independently re-verified as exact, including the CPG call-graph claims and
> the recomputed `_drive_loop` SHA-lock). All four findings adopted: **M1** — §11 now states
> plainly that for a non-debug run (every real `@mention` today) the violation signal is the
> process log only; nothing lands on the run/step-run, and the web UI's trace panel still reads
> "No trace events (not a debug run)" — this plan does not change the demo-visible symptom, only
> engineer-diagnosability. **M2** — two tests added to §10 (7-8, renumbering the publish-invariant
tests to 9-12): the `& granted_set`
> silent-drop behavior, and a `post_message`-required node ending on **empty** text (K-039's
> implicit dispatch never attempts on empty text, making the new check the sole defense on that
> path). **m1** — §3.2 now names `_execute_step`'s no-wired-LLM stub branch
> (`executor.py:517-518`) as a third, earlier return that silently skips the whole contract,
> deliberately out of scope. **m2** — §3.4/§9 now specify the new invariant lands after the
> existing `waitsForHuman` loop and before the `cmp`-guard validation loop. The two open questions
> the review raised (whether log-only diagnosability is an acceptable scope bar, and whether the
> §8 rollout caveat needs resolving before landing) are stakeholder-level and are not resolved
> here — they're routed to `teco`/the user separately, per the review's own framing.

## 1. Goal & scope

**Goal.** Give the executor an engine-level guarantee that an `agent`-typed node whose contract
is "must communicate" — today `post_message`, generalized to any future must-communicate tool —
either dispatches that tool before ending its turn, or the run records a visible, diagnosable
reason it didn't. This closes K-027 item 2 ("Terminal-node-must-post engine contract") and its
"Addendum from the K-025 QA pass" (the same failure on the non-terminal `intake` node, in the
worse "clarifying question never reached the thread" shape).

**In scope:**
1. A per-node declaration (`config.requiredTools`) naming which of a node's granted tools must be
   successfully dispatched at least once before the node may end its turn.
2. Enforcement inside `WorkflowExecutor._run_agent_node` (already outside the `_drive_loop`
   byte-lock) at both places a node currently ends: the non-tool-call-text branch and the
   `maxIterations`-exhaustion branch — the addendum's whole point is that the guarantee must not
   be terminal-node-specific or tool-name-specific.
3. A publish-time authoring invariant in `services._validate_def_spec`, mirroring the existing
   `waitsForHuman` check, so a `requiredTools` entry that isn't also granted (`config.tools`) or
   that lands on a non-`agent` step is a `WorkflowDefSpecError` at publish, not a live run that
   silently misbehaves.
4. Explicit layering with the already-shipped K-039 item 1 fallback (§4).
5. Updating `scripts/seed_workflows.sh`'s `triage@v1` literal to declare the contract on `intake`/
   `answer`, with the K-034 re-publish caveat spelled out (§8, §9).

**Explicitly out of scope** (per the coordination doc and K-027's own item split): judge
calibration (K-027 item 3), golden-set expansion (item 4), Ministral re-probe (item 5) —
`data-scientist` territory. K-033 (`maxSteps` off-by-one) and K-035 (argument-key shadowing) —
related nearby debt, not this contract. Modifying K-039's shipped fallback code — it stays exactly
as delivered (§4). Any change to `_drive_loop`, `_select_transition`, `_record`, `_trace_step`'s
signature, `resume`, or any repository/Cypher/DDL — none is needed (§6, §7).

## 2. Context & findings

**CPG: used `cpg_falkorchat` — confirmed the blast radius of the planned change before designing
it.** Freshness marker present (`CpgBuildInfo.BUILT_AT = 2026-08-17T00:40:42Z`, no `sourceCommit`
— the graph was built from a pruned scratch copy per the freshness recipe's documented limit, so
commit-pinned staleness can't be checked directly); `git log --oneline --since=2026-08-16 --
falkor-chat/server` returns zero commits, i.e. the source has not moved since the build — the
graph is current. Call-graph query (`(:METHOD)-[:CONTAINS]->(:CALL)` per the schema's documented
caller-resolution idiom) confirms: `_handle_tool_call` is called from exactly two sites, both
inside `_run_agent_node` (the real-call dispatch loop and the K-039 implicit-dispatch call);
`_run_agent_node` is called from exactly one production site, `_execute_step` (plus every test in
`server/tests/test_executor_agent.py`, which calls it directly as its own seam). No other
production code path reaches either function. This confirms the change described below is fully
contained to `executor.py`'s agent-node loop plus `services.py`'s publish-time validator, with no
ripple into `_drive_loop`, `repository.py`, or any other module.

**The defect, live-reproduced.** `docs/reviews/mention-reply-delivery-rca.md` §3/§4: a
`type:'agent'` node ends its turn on plain text with no tool call, `_run_agent_node`
(`executor.py:589-592` pre-K-039) returns `output=result.text, emissions=[]` with nothing
dispatched, and the engine has **no contract** that this is anything other than a normal, successful
`StepResult` (`on="done"` — DESIGN §6.1's own correction 1: `on` is vestigial, nothing in the
engine reads it). The run drives to `done` looking perfectly healthy from every place the UI or
`GET /workflow-runs/{id}` surfaces status. In the RCA's fresh, controlled repro, **all 3 of 3**
`triage@v1` steps hit this. The "Addendum from the K-025 QA pass" (`docs/BACKLOG.md`, search
`### K-027`) records the same failure on the **non-terminal** `intake` node, in a worse shape: the
run parks correctly (`waitsForHuman`) and looks healthy, but the clarifying question it was
supposed to post never reached the thread — nobody was ever shown what the run is waiting on. The
addendum's own words: "item 2's terminal-node scope is too narrow — the engine-level guarantee
must cover any node whose contract is 'post'."

**The K-039 fallback, read from the shipped code
(`server/falkorchat/executor.py:677-713`).** When `_run_agent_node`'s loop ends via the
non-tool-call branch (`not result.is_tool_call`) with non-empty `result.text`, and
`"post_message" in granted_set`, and nothing has been emitted yet this loop (`not emissions`), the
executor synthesizes `ToolCall(name="post_message", arguments={"text": result.text})` and
dispatches it through the same validated `_handle_tool_call` path a real call would take. This is
already **not terminal-node-specific** — the trigger condition is "was `post_message` granted,"
which is true for `intake` too (`scripts/seed_workflows.sh`'s `STEPS` grants `post_message` to
both `intake` and `answer`) — so today's fallback already gives the `intake` case some coverage,
just narrowly, only for the literal tool name `post_message`, and — this is the residual gap this
plan closes — **with no visibility if the implicit dispatch itself fails.** `_handle_tool_call`'s
AC-6/malformed/absorbed-`ServiceError` branches each append their own trace note, but `trace`
entries only reach a `TraceEvent` when the run is a debug run (`GraphTracer`, gated by
`run["trace"]`); a non-debug run — the common case — gets **zero** signal today if the implicit
fallback's own dispatch is rejected or errors. There is also no coverage at all today for the
`maxIterations`-exhaustion return path (`executor.py:724-732`): a node that repeatedly calls an
unrelated granted tool and never the required one exhausts its budget and returns
`last_text` with **no** must-post check of any kind, implicit or otherwise.

**Rule 8 (opaque config).** `Step.config` is already an opaque, app-side-only serialized string
(`falkor-chat/AGENTS.md` rule 8; DESIGN §6.1). `waitsForHuman` is the existing precedent for a
node declaring behavior this way, both at drive time (`executor.py` reads `config.waitsForHuman`)
and at publish time (`services._validate_def_spec`, `services.py:944-954`, using the
`_normalize_opaque` helper already used for exactly this purpose at `services.py:424`). This plan
adds a second key of the same shape and validates it the same way — no new Cypher-filtered field.

**DESIGN §6.1/§6.2.** `on` is vestigial (nothing reads it — correction 1); a `cmp`-family guard's
`output.<path>` resolution (`guards._resolve_path`) parses `step_output` as JSON only when a path
has a `rest` component, and today's `agent`-node `StepResult.output` is always the model's plain
text, never a JSON envelope — a design that changed that shape conditionally (only on violation)
would silently change what an existing or future `output.<key>` guard sees depending on whether a
violation occurred that run. §7 explains why this design avoids that entirely.

## 3. Design & rationale

### 3.1 Declaration: `config.requiredTools`

A `type:'agent'` step's `config` may carry `"requiredTools": [<tool name>, ...]` — a subset of
`config.tools` (the node's granted set) that the node must successfully dispatch at least once
during its execution. Absent or empty ⇒ no obligation, byte-identical to today's behavior (every
existing shipped def has no `requiredTools`, so nothing changes for them — see §7's compatibility
argument).

Chosen over an alternative "infer must-post from `type:'agent'` + `post_message` granted"
approach (i.e. treating every node that merely *holds* `post_message` as obligated) because: (a)
that's exactly K-039's fallback's own trigger condition already, and it is deliberately
permissive — a node might legitimately hold `post_message` without every turn being obligated to
use it; (b) it can't generalize to a future must-communicate tool without executor code knowing
that tool's name, which is precisely the K-025 addendum's complaint about the *terminal-node*
framing being too narrow, now transplanted onto a *tool-name* framing instead — same mistake, one
axis over; (c) an explicit per-node declaration is authorable and auditable (a def review can see
the obligation in the spec, not infer it from tool grants), and it is exactly the shape
`waitsForHuman` already established as the codebase's convention for "a step config bit the
executor keys a hard guarantee on."

### 3.2 Enforcement point: inside `_run_agent_node`, at both existing exit points

**Named out of scope: a third, earlier return this contract never reaches.**
`_execute_step`'s no-wired-LLM stub branch (`executor.py:517-518`,
`if not self._models.has_chat(): return StepResult(output="", on="done")`) returns *before*
`_run_agent_node` is ever called at all — a node declaring `config.requiredTools` and driven with
no chat LLM configured (`ModelGateway.has_chat() == False`) silently skips the whole contract: no
log, no trace. This is deliberate, not an oversight: it matches `_execute_step`'s own documented
rationale for that branch ("deliberate, not a fall-through accident... the affordance the offline
loop-engine tests are built on") — a node with no LLM has no agent loop to enforce anything
inside, and no shipped def or existing test combines `requiredTools` with a no-LLM executor. The
two exit points enforced below are `_run_agent_node`'s **own** two return points, not
`_execute_step`'s three-way dispatch.

`_run_agent_node` already tracks `emissions` (msgIds a dispatched `post_message` produced) across
its iteration loop. This plan adds one more piece of loop-local state, `satisfied: set[str]`, and
one small addition to `_handle_tool_call`: on the one success path that survives AC-6 rejection,
malformed-arg rejection, and an absorbed model-correctable `ServiceError` (i.e. a call that
genuinely reached `self._tools.dispatch(...)` and returned without raising), add `call.name` to
`satisfied`. `_handle_tool_call` gains one new parameter (`satisfied`); it has exactly two call
sites, both inside `_run_agent_node` (confirmed above via the CPG), so this is a two-line-call-site
change, not a public-surface change.

At both of `_run_agent_node`'s existing return points — the non-tool-call-text branch (after the
K-039 implicit-dispatch attempt, unchanged, has had its chance) and the `maxIterations`-exhaustion
fall-through — compute what's still missing and, if non-empty, make it visible (§3.3) without
otherwise changing what the node returns:

```python
required = set(_str_list(config.get("requiredTools", []))) & granted_set
```

(`_str_list` already exists in `executor.py`, used today by `_run_human_node`/`_run_wait_node` for
exactly this "defensively coerce an authored config list" job — reused, not reinvented. The `&
granted_set` intersection is defense-in-depth against a hand-crafted graph write that bypasses the
publish-time check in §3.4 — consistent with `guards.py`'s own "totality, never crash on bad
authored data at drive time" posture; publish-time is where an authoring mistake should be caught
loudly, per the codebase's repeated pattern.)

`post_message`'s satisfaction is **not** read off the generic `satisfied` set — it is read off the
richer, pre-existing `emissions` list (populated by `_buffer_emission` only when a dispatched
`post_message` call's result envelope actually carries a `"posted"` msgId). This one exception
matters: `PostMessageTool.run` (`tools.py:210-238`) can return a **string error result without
raising** (e.g. `"error: no thread is bound to this run"`) when the tool genuinely can't post —
that reaches `_handle_tool_call`'s success path (no exception), so the generic `satisfied` rule
would wrongly mark it done. `emissions` is the accurate "a `Message` actually got created" signal,
and it's already computed for free. Every other required tool name — hypothetical today, real
whenever a future must-communicate tool is granted — uses the generic `satisfied` membership,
because no richer signal exists for a tool the engine doesn't otherwise understand:

```python
def _missing_required_tools(required, satisfied, emissions):
    missing = set()
    for name in required:
        if name == "post_message":
            if not emissions:
                missing.add(name)
        elif name not in satisfied:
            missing.add(name)
    return missing
```

### 3.3 What happens on violation: trace-and-continue with a visible, always-on marker

Considered, and rejected:

- **Fail the step / fail the run.** Converts a reliability problem into an availability problem.
  K-039 existed specifically to *avoid* a `failed` run outcome for exactly this failure class; a
  design that now fails the run on the same trigger regresses that decision and is a strictly
  worse demo experience than today's silent-but-completed run.
- **Park the run.** There is no signal to wait *for* — the model already decided it was done.
  Parking would strand the run with no path to un-park short of a human blindly editing `ctx`,
  which is worse than the status quo, not better.
- **Retry the LLM call with a corrective nudge** ("you must call X before ending your turn").
  Genuinely appealing, and it is what a reader might expect "engine-level guarantee" to mean. Not
  built here, for three reasons: (1) the RCA's own evidence is that the shipped 4B model already
  ignores an equivalent instruction placed in its *system prompt* — a runtime corrective message is
  the same genre of intervention on the same model, with no evidence it would fare better;
  (2) it adds real complexity (a second retry budget or a slice of the existing `maxIterations`
  budget, a new corrective-message-content decision, more branches to test) for a payoff this
  model's own behavior casts doubt on; (3) the content/wording of a corrective prompt is exactly
  the kind of model-behavior lever this plan's own scope note excludes (`data-scientist` territory
  — K-027 items 3/4/5). **This is a considered-and-rejected alternative, not a closed door** — if
  K-027 item 3's calibration work later shows a corrective nudge measurably helps on the shipped
  model, it composes cleanly on top of what's built here (the retry would live in the same
  non-tool-call branch, before the missing-check runs) without touching this plan's mechanism.
- **Trace-and-continue with a visible marker — chosen.** The node still returns normally
  (`on="done"`, `output` unchanged — see §2's note on why `output`'s shape must not change
  conditionally). Two things happen that don't happen today:
  1. **Always** (regardless of whether this is a debug/traced run): `_log.warning(...)` naming the
     run id, step key, and missing tool name(s). This mirrors an existing, precedented pattern in
     the same file — `_link_emissions`'s "PRODUCED link gap" warning
     (`executor.py:1005-1009`) is exactly this shape: "a diagnosable gap, logged, never raised,
     never blocking." Nothing like this exists today for an implicit-dispatch failure; this closes
     that blind spot too, not just the new tool-name generalization.
  2. **When tracing is on** (debug runs only, same gate every other trace entry already uses): a
     new `("must_post_violation", "<message>")` tuple appended to the existing `trace` list that
     `_run_agent_node` already threads through `StepResult.trace`. `_trace_step` already forwards
     every entry in that list to the tracer verbatim (`executor.py:978-982`) — **zero changes** to
     `_trace_step`, `StepResult`'s dataclass shape, or any repository call are needed. This is
     literally what RCA §5 item 3 asked for: "an executor-level post-condition test asserting the
     node either dispatched that tool or the run recorded a traced, visible reason it didn't — not
     just a discarded `StepResult.output`."

  This is the smallest, safest, most testable increment that satisfies the addendum's actual ask
  (make the failure *visible*, everywhere it can occur, for any required tool) without inventing a
  new run outcome, a new `StepRun` property, or a new node/index (§7).

### 3.4 Publish-time invariant

`services._validate_def_spec` gains a fourth "deliberately LAST" invariant (alongside the existing
`waitsForHuman` check, `validate_cmp`, and the zero-transitions check — same docstring section,
`services.py:944-982`), mirroring the `waitsForHuman` check's exact shape and exception type
(`WorkflowDefSpecError`, nothing written): for every step, if `config.requiredTools` is present,
(a) it must be a list of strings, (b) the step's `type` must be `"agent"` (a `requiredTools` on a
`decision`/`human`/`wait`/`prompt`/`tool`/`message` step names an obligation no executor code path
for that type can ever satisfy — an authoring mistake worth rejecting loudly, not a live run that
quietly never fires the check), and (c) every named tool must also appear in that step's own
`config.tools`. This is the identical philosophy already stated in the `waitsForHuman` check's own
comment: "a parking step without it self-loops until the step budget fails the run" — catching the
mistake at authoring time is strictly cheaper than at drive time.

**Insertion order within the "deliberately LAST" block.** The new loop lands **after** the
existing `waitsForHuman` loop and **before** the `cmp`-family `validate_cmp` loop (i.e. second of
what becomes four, between `services.py:954` and `:956` today) — not because ordering changes
behavior here (the four checks inspect disjoint failure surfaces, so none can currently mask
another), but because the section's own docstring states the ordering discipline explicitly
("Running them last is load-bearing... so a new invariant can never mask... a pre-existing one")
and a plan that otherwise mirrors this codebase's conventions should be explicit about where a new
check sits, not leave it to whoever implements to guess.

## 4. Relationship to the K-039 fallback — explicit

**K-039's implicit-dispatch code is untouched, byte-for-byte** (`executor.py:677-713`). It stays
exactly as delivered: unconditional on any `post_message` grant (not gated by `requiredTools`),
narrowly scoped to the one tool shape it knows how to safely synthesize (`{"text": result.text}`,
`post_message`'s only required parameter). This plan does not modify its trigger condition, its
dispatch, or its tests.

This design is **a second, strictly more general layer that sits after K-039's, not a
replacement**:

- **Orthogonal trigger conditions.** K-039 fires whenever `post_message` is *granted* — an
  unconditional convenience, independent of whether the node's def author declared an obligation.
  This design's violation-visibility only engages when a node's def author opted in via
  `config.requiredTools` — an explicit, authored contract. A node that grants `post_message`
  without declaring it required still gets K-039's best-effort delivery attempt (unchanged) and
  generates zero new log/trace noise from this plan (its `required` set is empty).
- **Generalizes past the one tool name.** K-039 can only ever help `post_message`, because
  synthesizing `{"text": ...}` is only safe for a tool whose sole required argument is a text
  body. A future must-communicate tool with a different required-argument shape gets no help from
  K-039 — but it gets full coverage from this design's `satisfied`-tracking, because that tracking
  is generic (any tool name the def author lists).
- **Generalizes past the non-tool-call-ending branch.** K-039 only ever runs inside the "loop
  ended on plain text" branch. This design also checks the `maxIterations`-exhaustion return path,
  which K-039 never touches (§2's finding: a node that only ever calls an unrelated granted tool
  and never the required one exhausts its budget silently today).
- **Closes K-039's own residual blind spot.** If K-039's implicit dispatch itself fails (the tool
  dispatch raises, or — the concrete case found while reading `tools.py` — succeeds without
  raising but declines internally, e.g. no thread bound), today there is no log line and no trace
  entry unless the run happens to be a debug run. Because this design reads `post_message`'s
  satisfaction off `emissions` (not "did the call not raise"), that exact failure now surfaces as
  a `must_post_violation` — visible where today it is invisible.

Net effect: K-039 remains the fast, narrow, best-effort *recovery* attempt for the one tool it
knows how to safely help with; this plan is the general-purpose *detector* that runs regardless of
whether K-039 fired, applies to any declared tool, and never leaves a violation with zero signal.

## 5. The `_drive_loop` byte-identity lock

**Not touched.** Every change in this plan lands in `_run_agent_node` and `_handle_tool_call`
(both already below the `# ── seams ──` marker DESIGN §6.2 and `AGENTS.md` cite as outside the
lock) plus a new invariant inside `services._validate_def_spec` (never inside `_drive_loop` at
all — publish-time, a different module). No re-lock ceremony is needed; no line in `AGENTS.md`,
`BACKLOG.md`, `HISTORY.md`, or any archived plan asserting `71055f756280` needs to change. The
implementer should still reconfirm the lock unchanged before committing, per the existing
convention (`docs/HISTORY.md`'s repeated "SHA-lock reconfirmed unchanged" entries) — a cheap
confirmation, not a ceremony, since nothing in this plan's diff touches that function.

## 6. Rule 8 (opaque strings)

`config.requiredTools` is a new key inside the already-opaque `Step.config` string — parsed
app-side only, exactly like `waitsForHuman`/`tools`/`maxIterations`/`model` before it. It is never
filtered in Cypher; `_validate_def_spec`'s new check runs entirely in Python before any write, the
same way the `waitsForHuman` check already does. No new Cypher-filtered field is introduced.

## 7. Risks / RAM (rule 6)

**No new node, index, or property.** Verified, not assumed:
- `config.requiredTools` — a new key inside an existing opaque string property (`Step.config`).
  No DDL.
- The new `must_post_violation` trace kind — a new *value* for `TraceEvent.kind`, which is already
  an unindexed free-string property (confirmed against `scripts/bootstrap_schema.sh:128-133,
  202-203`: `TraceEvent` is indexed/constrained only on `traceId`). No DDL, no new index.
- `_log.warning(...)` — process log only, zero graph impact.
- `satisfied: set[str]` — in-process loop-local state inside `_run_agent_node`, never persisted.

**Backward compatibility.** Every shipped def today (`triage@v1`, `access-request@v1`, every
offline test fixture in `test_executor.py`/`test_executor_agent.py`) declares no
`config.requiredTools`, so `required` is always the empty set for them and `_missing_required_tools`
always returns empty — zero behavior change, zero new log lines, zero new trace entries for any
existing def or existing test. This is the compatibility argument the plan leans on instead of a
broad regression sweep: the new code path is unreachable without an explicit, new, opt-in config
key.

## 8. `scripts/seed_workflows.sh` — wiring the shipped demo, and its rollout caveat

`scripts/seed_workflows.sh`'s inline `STEPS` (search for `"tools": ["post_message"]`) grants
`post_message` to both `intake` and `answer`. Add `"requiredTools": ["post_message"]` to both —
this is the concrete change that makes the shipped `triage@v1` demo actually exercise the new
contract, not just a theoretical capability. `research` is unaffected (it grants only
`graphrag_retrieve`, nothing to require).

**Caveat, load-bearing, do not skip:** per `docs/plans/workflow-republish-semantics.md` §0/§1 and
`falkor-chat/AGENTS.md`'s own K-034 note, a config-only change to an already-published
`(key, version)` is a **silent no-op** on re-publish — `MERGE … ON CREATE SET` only writes on first
creation, and this is deliberately tested behavior
(`server/tests/test_api.py::test_republish_is_create_only_on_properties_structure_read_unchanged`,
K-031), not a bug this plan should route around. Concretely: re-running
`./scripts/seed_workflows.sh` against an environment where `triage@v1` is already published (the
shared dev box's `reference` graph, `ws:acme`) will **not** actually add `requiredTools` to the
stored def — the script will exit success, having changed nothing. It **will** take effect on any
environment that publishes `triage@v1` fresh: a new workspace bootstrap, the default `pytest -q`
run (which wipes `reference` at teardown per `AGENTS.md`'s testing-hazards note, so the next
`seed_workflows.sh` republishes clean), or a from-scratch dev box. Getting the change onto an
*already-seeded* shared environment needs either a full `reference` wipe + re-seed, or a version
bump (`triage@v1` → `v2`) — the latter is explicitly **not** recommended as part of this plan: `v1`
is tightly coupled to `config.TRIGGER_DEF_KEY`/`_VERSION`'s own defaults (K-037's whole reason for
existing is how fragile that coupling already is), and bumping it is a separate, larger,
stakeholder-visible operational change this plan should not fold in opportunistically. This is
flagged as an open rollout question in §10, not resolved here.

## 9. Step-by-step implementation

1. **`server/falkorchat/executor.py`**
   - Add module-level helper `_missing_required_tools(required, satisfied, emissions)` (§3.2),
     placed near `_str_list`/`_dumps` in the "value objects" / helper section.
   - `_handle_tool_call`: add a `satisfied: set[str]` parameter. At the single success return path
     (after `trace.append(("tool_result", _short(out)))`, before `return content`), add
     `satisfied.add(call.name)`. Update its docstring's contract description to mention the new
     parameter's bookkeeping role.
   - `_run_agent_node`: initialize `required = set(_str_list(config.get("requiredTools", []))) &
     granted_set` and `satisfied: set[str] = set()` alongside the existing `trace`/`emissions`/
     `last_text` initializations, before the iteration loop. Thread `satisfied` through both
     `_handle_tool_call` call sites (the main dispatch loop and the K-039 implicit-dispatch call,
     unchanged otherwise). Add a `_missing = _missing_required_tools(required, satisfied,
     emissions)` check, and on non-empty call a new small static helper `_note_must_post_violation`
     (mirrors `_buffer_emission`'s `@staticmethod` shape): logs via `_log.warning` and appends
     `("must_post_violation", "...")` to `trace`. Call this check at both existing return points:
     right after the K-039 implicit-dispatch attempt inside the non-tool-call branch, and right
     before the `maxIterations`-exhaustion `return` at the end of the function. Update the
     function's docstring (currently documents K-039's fallback inline) to describe the new
     contract and explicitly reference this plan.
   - No changes to `StepResult`'s dataclass shape, `_drive_loop`, `_execute_step`,
     `_select_transition`, `_record`, `_trace_step`, or `resume`.

2. **`server/falkorchat/services.py`**
   - `_validate_def_spec`: add the fourth "deliberately LAST" invariant described in §3.4,
     inserted **immediately after** the existing `waitsForHuman` loop (`services.py:944-954`) and
     **before** the `cmp`-family `validate_cmp` loop (`:956-963`) — iterate `steps`, for each with
     `config.requiredTools` present validate list-of-strings, `type == "agent"`, and
     `⊆ config.tools`, raising `WorkflowDefSpecError` with a message in the same style as the
     `waitsForHuman` one. Update the docstring's "Three further invariants" to "Four further
     invariants."

3. **`scripts/seed_workflows.sh`**
   - Add `"requiredTools": ["post_message"]` to the `intake` and `answer` step configs in the
     inline `STEPS` literal (§8). Add a short inline comment pointing at this plan and the K-034
     rollout caveat, so a future reader doesn't assume re-running the script onto a live
     environment is sufficient.

4. **Docs** (same change, per `AGENTS.md`'s doc-update-is-part-of-the-unit convention):
   - `falkor-chat/docs/BACKLOG.md`: flip K-027 item 2 to delivered once implemented + tested,
     recording the mechanism (declaration key, enforcement points, relationship to K-039) the way
     other delivered K-027/K-039 sub-items already do.
   - `falkor-chat/docs/HISTORY.md`: dated entry per the repo convention, including the test-count
     delta and the SHA-lock reconfirmation line (matching the style of the K-039 HISTORY entry).
   - `falkor-chat/docs/DESIGN.md` §6.1: add a short paragraph next to the existing `agent`-node
     description documenting `config.requiredTools` (mirroring how `waitsForHuman` is documented
     two paragraphs above it) and a one-line cross-reference to this plan from the `on`-is-vestigial
     correction, since that's exactly the fact this design leans on to justify never inventing a
     new `on` value.

## 10. Test strategy

Offline, no LLM/network, following `server/tests/test_executor_agent.py`'s existing stub
conventions (`StubChatLLM`, `StubRegistry`/`RaisingRegistry`, `_config()`/`_executor()` helpers) —
new tests land in that file, in a new section after the existing "K-039 / mention-reply-delivery
RCA #1" section (`test_executor_agent.py:508+`), and a new test in `server/tests/test_services.py`
alongside the existing `waitsForHuman` publish-invariant tests.

**`test_executor_agent.py` — executor-level (drives `_run_agent_node` directly, the file's
existing seam):**

1. `test_compliant_node_dispatching_required_tool_leaves_no_violation_trace` — a node with
   `requiredTools=["post_message"]` whose model calls `post_message` directly (tool-call branch,
   not the K-039 fallback). Assert: `result.output` unchanged from today's behavior, and no
   `must_post_violation` entry in `result.trace`. Pins the zero-behavior-change case for the
   common/compliant path.
2. `test_plain_text_ending_with_required_post_message_recovers_via_k039_no_violation_logged` —
   mirrors the existing `test_plain_text_with_granted_post_message_is_posted_as_implicit_fallback`
   but with `requiredTools=["post_message"]` added to config. Assert the K-039 implicit dispatch
   still fires exactly as before (unchanged assertions on `reg.dispatched`/`result.emissions`) **and**
   no `must_post_violation` trace entry — the fallback's success satisfies the contract via
   `emissions`.
3. `test_required_non_post_message_tool_never_dispatched_logs_and_traces_a_visible_violation`
   (the core new-behavior test, directly the RCA §5 item 3 ask) — a node with
   `requiredTools=["notify_owner"]`, `tools=["notify_owner"]` (a tool with a shape K-039 cannot
   help with), model ends on plain text without calling it. Assert: `result.on == "done"` (the run
   is not failed — pins the chosen "trace-and-continue" behavior over the rejected "fail the step"
   alternative), a `("must_post_violation", ...)` entry naming `notify_owner` is present in
   `result.trace`, and (via `caplog`, mirroring the existing
   `test_a_model_correctable_tool_error_is_logged_even_without_a_tracer` pattern) a `_log.warning`
   naming the missing tool is emitted **even with no tracer configured** — pins that visibility does
   not depend on the run being a debug run.
4. `test_required_post_message_whose_own_implicit_dispatch_declines_still_logs_a_violation` — the
   concrete gap found while reading `tools.py`: a registry double whose `post_message` dispatch
   returns an error string without raising (mirroring `PostMessageTool.run`'s "no thread bound"
   path) so the K-039 fallback "succeeds" at the dispatch layer but produces no `"posted"`
   envelope. Assert `result.emissions == []` and a `must_post_violation` naming `post_message` is
   present — pins that satisfaction is read off `emissions`, not off "the dispatch didn't raise."
5. `test_required_tool_never_dispatched_across_max_iterations_logs_and_traces_a_violation` — a
   node with `requiredTools=["notify_owner"]` whose model only ever calls a *different* granted
   tool every turn (an `AlwaysToolLLM`-style stub calling some other tool, never `notify_owner`)
   until `maxIterations` exhausts. Assert the existing exhaustion behavior is unchanged
   (`result.on == "done"`, best-current-text output, the existing `node_note` exhaustion trace
   entry still present) **and** a `must_post_violation` entry is also present — pins the
   `maxIterations` exit point, which K-039 never covered at all.
6. `test_undeclared_required_tools_is_fully_backward_compatible` — re-run (or parametrize) a
   representative slice of the *existing* K-039/plain-tool-loop tests with no `requiredTools` in
   config and assert zero `must_post_violation` entries appear — the explicit compatibility pin
   for §7's claim, not just an implicit consequence of not touching those tests' own assertions.
7. `test_required_tool_absent_from_granted_tools_is_silently_dropped_at_drive_time` (review
   finding M2.1) — a node whose `config.requiredTools` names a tool that is **not** in
   `config.tools` (the hand-crafted-graph-write shape publish-time validation, §3.4, is meant to
   catch — this test bypasses it deliberately, exercising `_run_agent_node` directly the way this
   file's other tests already do). Assert: no exception, no `must_post_violation` entry, and the
   node's `result.output`/`on` are unaffected — pins that the `& granted_set` intersection in
   §3.2 really does silently drop an ungranted required-tool name at drive time rather than
   crashing or falsely flagging a violation, exactly as §3.2's paragraph justifying that
   intersection claims.
8. `test_required_post_message_node_ending_on_empty_text_still_logs_a_violation` (review finding
   M2.2) — a node with `requiredTools=["post_message"]` whose model ends its turn via the
   non-tool-call branch with **empty** `result.text` (mirroring the existing
   `test_no_implicit_post_when_final_text_is_empty` fixture shape). K-039's implicit-dispatch
   attempt is gated on `result.text` being truthy (`executor.py:678`), so it never even attempts
   here — this is a materially different code path from test 4 above (where a dispatch is
   attempted and declines). Assert: `reg.dispatched == []` (K-039 correctly never fires on empty
   text, unchanged), `result.emissions == []`, and a `must_post_violation` entry naming
   `post_message` **is** present — pins that this contract's check is the *sole* defense on the
   empty-text ending, not a redundant restatement of an existing K-039 assertion.

**`test_services.py` — publish-time invariant (alongside the existing `waitsForHuman` tests):**

9. `test_publish_workflow_def_required_tool_not_granted_raises_nothing_written` — an `agent` step
   declaring `config.requiredTools=["x"]` without `"x"` in `config.tools`; assert
   `WorkflowDefSpecError` and (per the existing pattern in this file) that nothing was written.
10. `test_publish_workflow_def_required_tools_on_non_agent_step_raises_nothing_written` — a
    `decision`/`human` step declaring `config.requiredTools`; assert `WorkflowDefSpecError`.
11. `test_publish_workflow_def_required_tools_non_list_raises_nothing_written` — a malformed
    `config.requiredTools` (a string, a list containing a non-string); assert
    `WorkflowDefSpecError`.
12. `test_publish_workflow_def_required_tools_subset_of_granted_succeeds` — the valid case
    (`requiredTools ⊆ tools`) publishes cleanly; assert the stored `config` round-trips the key
    (mirroring `test_publish_workflow_def_derives_start_and_serializes_config_and_guard`'s
    byte-comparison style).

**Full offline suite** (`cd server && .venv/bin/python -m pytest -q`) must stay green throughout,
per `AGENTS.md` rule 5's project-wide convention (this repo's variant: the full offline suite, not
a schema-specific query suite). No `pytest -m live` change is needed or proposed by this plan —
the RCA's AC-4 assertion already exercises the *existing* K-039 path; extending it to also assert
`must_post_violation` visibility for a non-`post_message` required tool is optional follow-up
work, not required for this plan's done-condition, since offline pins 3–5 already characterize
that behavior deterministically.

## 11. Risks & open questions

- **For a non-debug run — every real `@mention` in the shipped demo today — this plan's only
  actual signal is the process log.** Say this plainly, because §3.3's "visible marker" language
  could otherwise be read as closing the RCA's user-facing symptom, and it does not. Nothing in
  this design writes onto the run or step-run itself for a non-debug run: the `must_post_violation`
  trace entry only ever reaches a `TraceEvent` when `run["trace"]` is set (debug runs only, the
  same gate every other trace entry already uses), and `trigger.py`/`app.py` do not start a debug
  run by default. The web UI's own trace panel renders the literal string
  `"No trace events (not a debug run)."` for exactly this case. Concretely: a demo presenter (or
  anyone watching `ws:acme`) sees **exactly what they see today** — a run that finishes `done`
  with no reply — even after this plan lands; only an engineer with server log access gains a
  signal. This plan does not change what the RCA calls "demo-blocking"; it only makes the failure
  diagnosable, not delivered. §3.3 names why the rejected alternatives (fail/park/retry) don't
  close that gap either, and no cheap alternative that avoids reopening the §2 `output`-shape
  constraint or adding a new persisted `StepRun` property (against rule 6) was found. Whether
  "diagnosable to an engineer with log access" is an acceptable scope bar for K-027 item 2, or
  whether a fast follow-up should also make the violation visible in the run's own queryable state
  regardless of debug-run status, is a stakeholder-level call this plan does not make — it's
  routed to `teco`/the user separately, alongside the rollout question below.
- **Rollout to the shared dev demo (`ws:acme`) is genuinely open** (§8) — this plan deliberately
  does not resolve whether/when to wipe-and-reseed or version-bump `triage@v1` to actually carry
  `requiredTools` on the live shared environment. Flagging this explicitly rather than picking
  silently, since it's an operational/stakeholder call (K-037's coupling makes a version bump
  non-trivial), not a design question this plan is positioned to answer alone.
- **`satisfied` is dispatch-success, not business-success, for any tool other than
  `post_message`.** A future required tool that can "succeed" at the dispatch layer while
  declining its actual effect (the same shape found in `PostMessageTool.run`'s no-thread-bound
  path) would be silently marked satisfied unless that tool also gets a `post_message`-style
  special case added to `_missing_required_tools`. This is named, not hidden: today no such tool
  exists (the only granted tools are `post_message`, `graphrag_retrieve`, `human_handoff`), so
  there is nothing to special-case yet. Flagged for whoever adds the first future
  must-communicate tool that isn't `post_message` — it may need the same treatment as §3.2 gives
  `post_message`, and this plan does not attempt to build a fully generic version of that richer
  signal (would require a project-wide convention for how tools report success in their result
  envelope, which is a larger design than this contract needs to justify).
- **Whether the corrective-retry alternative (§3.3, rejected here) should be revisited** once
  K-027 item 3 (judge/model calibration) has real data on whether this model responds to runtime
  correction prompts — an open question for `data-scientist`, not this plan, per the scope
  boundary in §1.
- **The `waitsForHuman`-adjacent publish invariant gap noted while reading `services.py`:** there
  is no existing test asserting a `human`/`wait` step *missing* `config.waitsForHuman` actually
  raises at publish (the code path exists at `services.py:944-954`, but a targeted test for it was
  not found in `test_services.py` during this investigation). Pre-existing, unrelated to this
  plan's scope — noted here only because the new `requiredTools` invariant sits in the same code
  block and an implementer following this plan's test list (§10, tests 9-12) will naturally close
  the analogous gap for the new invariant; closing the pre-existing `waitsForHuman` gap is a
  separate, small, opportunistic pick-up an implementer may fold in but this plan does not require
  it.
