# Review — M3 LLM-native workflow executor plan (`m3-executor.md`)

> Reviewer: analyst · 2026-07-10 · Static review-gate pass (R0 in the coordination doc).
> Artifact under review: `docs/plans/m3-executor.md` (architect, proposed).
> Baselines: `docs/requirements/llm-native-workflows.md` (FR-1…FR-7, AC-1…AC-6),
> `docs/plans/m3-executor-coordination.md` (U0–U15), and the live `falkor-chat/` codebase
> (`server/falkorchat/*`, `docs/DESIGN.md` §6/§13, `docs/QUERIES.md` §10/§11, `bootstrap_schema.sh`).
> ML/DS methodology (judge reliability, retrieval grounding/eval) is **out of this review's scope** —
> covered separately by the data-scientist at `docs/plans/m3-executor-ml.md` (U0).

## Scope & verdict

I judged the plan on grounding vs. the real codebase, completeness against FR/AC, soundness of the
execution/atomicity/trigger model, convention fit, and proportionality. I did **not** re-litigate the
four locked decisions (D1 §6/§13 reconciliation, D2 `PRODUCED` edge, D3 unit split, D4 function-calling
risk) — I only judged whether the plan implements them soundly, and it does.

**Verdict: approve with suggestions.**

The architecture is sound, the grounding is strong (I verified the plan's codebase claims — almost all
are accurate), and Phase 0 / U1–U2 (the graph-dba gate) can proceed now. There are **no blockers** —
nothing in the plan is wrong-and-unsafe such that no work may start. But there are **four majors** (M1–M4
below): each is a localized gap, not a redesign, and each must be closed in the plan **before the unit it
governs is implemented** (M1→U4, M2/M3→U11, M4→U2). I recommend the architect amend §2.1/§2.4/§3/§4/§6
to close them; none block the U1/U2 data-model gate.

## Grounding check — plan claims vs. the real codebase

Verified **accurate** (evidence in parentheses):

- The run/step-run DDL is already provisioned: `WorkflowRun.runId` index+UNIQUE, `WorkflowRun.status`
  index, `StepRun.stepRunId` index+UNIQUE, `StepRun.status` index, `Step.key`/`Step.stepUid`
  (`bootstrap_schema.sh:122-176`). The plan's §3 "already provisioned" table is correct.
- `EMITTED` is genuinely the K-013 `Message→Message` provenance edge (`QUERIES.md:625-649`,
  `services.post_agent_answer:288`) — the D2 `PRODUCED` disambiguation is well-grounded, not invented.
- `STEP_TYPES = {prompt,tool,decision,human,message,wait}` (`services.py:39-41`) — adding `'agent'` is
  the one-line whitelist add the plan claims.
- `_serialize_opaque` already JSON-encodes dicts with stable key order (`services.py:96-108`) — so the
  `{kind}`/config discriminators ride the existing publish path with no serialization change.
- `AgentResponder` loop-guards on `role == "assistant"` then triggers on `@mention`
  (`responder.py:62-68`) — the plan's §6 reconciliation mirrors the real trigger contract.
- `llm.LLM` is `complete(messages) -> str`, single-shot (`llm.py:29-32,66-71`) — the `chat(messages,
  tools)` extension is correctly identified as net-new.
- `api.post_message` schedules `_safe_embed` + `_safe_respond` on `BackgroundTasks`
  (`api.py:114-133`), config-gated — the plan's `_safe_run_workflow` peer slots in as described.
- `mcp>=1.28` is a real dependency (`server/pyproject.toml:15`) — the FR-5c MCP-**client** seam has a
  library to build on.

**Inaccurate / stale claims found:**

- **`MAX_CONFIG_LEN` already exists at 8000** (`schemas.py:39,45,55`), applied to both `Step.config` and
  `TRANSITION.guard`. The plan §3 poses "Bump `MAX_CONFIG_LEN`? (systemPrompts are longer)" as an open
  question, implying it may not exist. It does, and 8000 chars is ample for a system prompt — see M6.
- Minor doc drift the U1 reconciliation should sweep: `DESIGN.md` §7.1 still cites `test_queries.sh`
  "(126/126)" and DESIGN §6.2's `StepRun` property list omits `stepRunId` (which `bootstrap_schema.sh`
  indexes+constrains). Not the plan's error, but U1 touches these sections — sync them (see M8/nit).

## Findings (severity-ranked)

### M1 — Suspend semantics are underspecified; risk of permanently stuck runs (major)

**Evidence:** plan §2.1 loop and §2.4. The loop says: *"no guard fires + step is a wait/intake node →
status = waiting"* and *"no guard fires + terminal (no outgoing transitions) → done."* But **there is no
defined notion of a "wait/intake node"** for a `type:'agent'` step — intake, research, and answer are all
`type:'agent'` (§8 triage table). Two holes follow:

1. The executor has no signal to distinguish intake (should suspend and wait for a human) from research
   (whose guard "findings sufficient" may also not fire, but where no human reply is coming).
2. The case *"no guard fires + step has outgoing transitions + not a wait node"* is **undefined** — the
   pseudocode falls through with no action, implying either an infinite re-execution loop (bounded only
   by the step budget failing the run) or a silent hang.

A research node with a not-yet-satisfied guard would, under §2.1 as written, go to `waiting` and park
forever waiting for a human who will never reply — a stuck run that only the step budget eventually
fails. This directly threatens AC-2's correctness.

**Suggested improvement (→ architect, §2.1/§2.4).** Define the suspend trigger explicitly rather than by
the undefined "wait/intake node." Options: (a) a per-node config flag (e.g. `waitsForHuman: true` in the
LLM-native `config`) that means "on no-fire, suspend"; nodes without it **re-execute** (self-loop) on
no-fire, bounded by the step budget; or (b) make "suspend vs. re-loop vs. done" a three-way decision the
node's outcome carries, not an inference from graph shape. Either way, name the exact rule for all three
cases (fire / no-fire-with-outgoing / no-outgoing) so the executor is deterministic. This is the seam
U4 implements — it must be pinned before U4.

### M2 — §2.4 and §6 contradict each other on what resumes a waiting run (major)

**Evidence:** §2.4 says *"A subsequent **human** message in the run's thread **resumes** the run."* §6
orders the trigger as *"Loop-guard first … then the FR-7 trigger: the message @mentions the agent"* with
*"Resume-before-start"* listed as a **subsequent** bullet — i.e. resume is reached only **after** the
`@mention` gate passes. These are incompatible: §2.4 resumes on any human reply; §6 requires the human to
`@mention` the agent again on every intake answer.

If §6's ordering wins, the intake loop breaks AC-2's natural flow: the agent asks a question, the human
answers *in plain conversation* (no `@mention`), and the run never resumes. If §2.4 wins, any human
message in the thread resumes — including messages unrelated to the triage.

**Suggested improvement (→ architect, §2.4/§6).** Reconcile to one rule and state it in both sections.
The natural resolution: **resume is checked before the `@mention` gate** — for a thread with a `waiting`
run, any non-agent (`role != 'assistant'`) human message resumes it (no re-`@mention` required); the
`@mention` gate applies only to the **start** decision. Make the ordering in §6 explicit:
`loop-guard → resume-if-waiting → (else) @mention-to-start → (else) fall through`.

### M3 — Trigger wiring as a "peer to `_safe_respond`" risks a double response (major)

**Evidence:** §6 says *"Wire it in `api.post_message` as `_safe_run_workflow` (peer to
`_safe_embed`/`_safe_respond`)"* **and** *"If no workflow is configured → fall through to the existing
`AgentResponder`."* If `_safe_run_workflow` and `_safe_respond` are both registered as independent
background tasks (`api.py:127-131` pattern), then on an `@mention` with workflows enabled **both** fire:
the workflow starts *and* the responder posts a direct reply — two agent answers to one mention.

The "fall through to AgentResponder" only works if the trigger **owns** the fall-through (holds the
responder and calls it when no workflow is configured), i.e. exactly one of {workflow, responder} is
wired per request — not two peers each self-deciding.

**Suggested improvement (→ architect §6 / coder U11).** Pin the wiring: when `WORKFLOW_ENABLED`,
`app._build_default_app` registers **only** `_safe_run_workflow` and passes the `AgentResponder` **into**
`WorkflowTrigger` for the no-workflow fall-through; `_safe_respond` is **not** separately registered.
When disabled, the M2 wiring is unchanged. State this explicitly so U11 doesn't wire two peers.

### M4 — `record_step_and_advance` doesn't specify how the previous StepRun is anchored for the `NEXT` append (major)

**Evidence:** §2.1 and §3 (`record_step_and_advance`): *"CREATE `StepRun` …, `NEXT` from prev StepRun."*
The plan never says how the query locates the **previous** StepRun to hang `NEXT` from. `AT_STEP` points
to a `Step`, not a `StepRun`; `WorkflowRun` has no StepRun tail pointer. Finding the last StepRun then
requires either walking the `NEXT` chain to its tail (O(n) per step, and no anchor to start the walk) or
ordering all of the run's StepRuns by `startedAt` (label scan) — inside the atomic single-`GRAPH.QUERY`
that also relinks `AT_STEP` and bumps `stepCount`. This is a **data-model decision** (does `WorkflowRun`
carry a `LAST_STEP_RUN`/tail edge, analogous to `Thread`'s HEAD/TAIL, DESIGN §5.2?), not merely query
authoring the graph-dba gate can improvise.

**Suggested improvement (→ architect §3, then graph-dba U2).** Name the anchor in §3: add a
`WorkflowRun -[:LAST_STEP_RUN]-> StepRun` pointer (or equivalent) that `record_step_and_advance` reads
and relinks in the same query — mirroring the locked HEAD/TAIL pattern the codebase already uses for the
message linked list. Then U2 profiles it as an edge-anchored single query. Without this, the "single
atomic, index-anchored query" guarantee in §3/§8 is not actually achievable.

### M5 — `graphrag_retrieve` tool omits the embedding step (minor)

**Evidence:** §4 says `graphrag_retrieve` *"calls `services.hybrid_search` (§6 vector ANN…)."* But
`hybrid_search` takes a **query vector** `q_vec`, not text (`services.py:381-383`). The research node's
LLM will call the tool with a **text** query; something must embed it first (the responder does exactly
this: `embedder.embed(text)` then `hybrid_search`, `responder.py:96-99`). The plan's `ToolRegistry`/tool
wiring (§4, U9) doesn't inject an `Embedder`.

**Suggested improvement (→ architect §4 / coder U9).** State that `graphrag_retrieve` depends on an
injected `Embedder` (or an embed-then-search service helper) to turn the LLM's query string into `q_vec`
before `hybrid_search`. Also note the served app must run at the workspace's embedding dimension
(`ws:acme` = 1024, per AGENTS.md) or the ANN silently returns nothing — an AC-3 precondition.

### M6 — `MAX_CONFIG_LEN` open question is already answerable (minor)

**Evidence:** §3 poses *"Bump `MAX_CONFIG_LEN`?"* as open. It exists at 8000 (`schemas.py:39`), which
comfortably fits a node system prompt. **Suggested improvement (→ architect §3):** resolve the question
in the plan — either "8000 is sufficient, no change" or state the intended larger bound and its RAM
justification (rule 6). Don't leave it dangling into U3.

### M7 — The `expr` guard kind is unexercised by the triage proof (minor / proportionality)

**Evidence:** §2.5 defines `kind:'expr'` handling "trivial `outcome == '<literal>'` / unconditional"
cases, but the triage flow (§8) uses only `kind:'llm'` guards plus **unconditional** transitions — and
unconditional is the empty-string guard (`""`), not `expr`. So no `expr` guard is exercised anywhere in
this cut, making the "trivial literal" branch untested speculative code.

**Suggested improvement (→ architect §2.5 / tdd-engineer U7):** reduce `kind:'expr'` to a **pure**
`NotImplementedError` seam (dispatch on `kind`, `llm` implemented, `expr`/unknown → clear raise). This
satisfies FR-3's coexistence-seam requirement with zero untested branches; the full expr evaluator is
already deferred. Keeps the honest-seam intent without the dead code.

### M8 — DESIGN §6.2 `StepRun` shape and stale counts, to fold into U1 (minor / doc)

**Evidence:** DESIGN §6.2's `StepRun {stepKey,status,startedAt,endedAt,input,output}` omits `stepRunId`
(indexed+constrained in `bootstrap_schema.sh:126,176`); DESIGN §7.1 still says `test_queries.sh`
"(126/126)" vs. the plan's current 193. **Suggested improvement (→ graph-dba U1):** since U1 already
reconciles DESIGN §6.1/§6.2/§13, add `stepRunId` to the §6.2 property list and refresh the stale test
count in §7.1 in the same pass.

## What's solid

- **Grounding is strong.** The plan's structural claims about the existing schema, edges, provenance,
  and seams are almost all verifiably true — it is built on a real reading of Slice 1 and the M2 server,
  not a stale mental model. This is the hardest thing to get right and the plan gets it right.
- **D2 is correctly grounded.** The `EMITTED` overload is real (`QUERIES.md:625-649`); recommending a
  distinct `PRODUCED` edge is the right call and correctly routed to the graph-dba gate.
- **Atomicity discipline is applied correctly** (rule 4): every state-move is one `GRAPH.QUERY`; the
  two-step emission link (post, then `PRODUCED`) is honestly flagged as non-atomic-but-recoverable.
- **Rule 8 respected:** the `{kind}`/config JSON discriminators are parsed app-side only, never in
  Cypher — the plan is explicit about this and it holds.
- **AC-5 by construction:** the `NullTracer`/`GraphTracer` injected seam makes debug-vs-lean one wiring
  choice with no `if debug:` branches — a clean satisfaction of the tracing requirement.
- **Baseline preservation:** the config-gated trigger keeps the pytest baseline network-free and M2
  direct-reply behavior intact — consistent with the existing `ENABLE_AGENT` posture (`config.py:72`).
- **Runaway-safety reasoning is sound:** step budget + per-node cap (not a "never revisit" rule) — and
  the plan correctly recognizes the intake self-loop as a *legitimate* cycle the budget bounds.
- **FR/AC coverage is complete and explicitly mapped** (§8 table + §9 test strategy). FR-1…FR-7 and
  AC-1…AC-6 each trace to a concrete mechanism.
- **Proportionality is good:** MCP-client scoped to a stub, deterministic handlers as honest stubs,
  `TraceEvent` gated to debug runs — nothing materially over-built for the stated scope.

## Open questions (for the caller / owners)

1. **M1 + M2 together define the intake loop's core mechanic (AC-2).** Whoever owns the architect
   amendment should resolve suspend-signal (M1) and resume-trigger (M2) as a pair — they interact.
2. **Does resume require the human to be in the same thread only, or the same channel?** §6 anchors
   resume on the thread; confirm intake questions and answers are guaranteed same-thread (they are, via
   `post_message` into the run's thread) so `find_waiting_run_for_thread` is the right anchor.
3. **M4's tail-pointer** is a small additive edge with a per-run RAM cost (rule 6) — trivial, but the
   graph-dba gate should confirm it in the U1 RAM delta.

## Routing summary

| Finding | Severity | Owner (unit) |
|---|---|---|
| M1 suspend semantics | major | architect → U4 (tdd-engineer) |
| M2 resume-trigger contradiction | major | architect → U11 (coder) |
| M3 double-response wiring | major | architect §6 → U11 (coder) |
| M4 StepRun `NEXT` anchor / tail pointer | major | architect §3 → U2 (graph-dba) |
| M5 `graphrag_retrieve` embedder | minor | architect §4 → U9 (coder) |
| M6 `MAX_CONFIG_LEN` resolve | minor | architect §3 → U3 |
| M7 `expr` seam is dead code | minor | architect §2.5 → U7 (tdd-engineer) |
| M8 DESIGN §6.2/§7.1 sync | minor | graph-dba (U1) |

**Bottom line:** approve with suggestions. The plan is well-grounded and buildable; U1–U2 (the data-model
gate) may proceed. Close M1–M4 in the plan before U4/U11/U2 respectively — they are localized amendments,
not a redesign.
