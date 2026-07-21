# Change History — falkor-chat

> Dated log of actual changes to the `falkor-chat` component. Most recent first.
> (Formerly `kaizen/history.md` — older entries may say "kaizen" for what is now
> [`BACKLOG.md`](./BACKLOG.md) + this file; file paths in old entries have been
> updated so they still resolve.)

## 2026-07-21 — M3 K-024 (second half): the LLM-free `kind:'process'` proof flow, analyst-gated twice — **M3's last build item ✅**

Closes **K-024** — the `kind:'process'` business-process proof flow the DESIGN §6.3
"coordination is workflow" claim rests on — and with it the last **build** item of M3. Only
**K-025** (QA acceptance) now stands between the component and **M3 ✅**. Delivered by the
teco-coordinated chain over five units: **graph-dba** (U0), **tdd-engineer** (U1), **coder**
(U2, U3, U4, U4b), **teco** (U5 + every integration run), with the **mandatory analyst gate** run
**twice** as a non-negotiable done-condition. Plan `docs/plans/m3-process-flow.md` (v2.1);
coordination log `docs/plans/m3-process-flow-coordination.md`; all three gates in
`docs/reviews/m3-process-flow.md`. New baselines: server pytest **523 → 533 passed / 1 deselected**
(350 at the start of the slice); query suite **241 → 256**.

**The central design claim, and it held: `_drive_loop` was never modified.** SHA `71055f756280`
before the slice, after every edit, and at closeout. A business process fell out of *park-and-branch*
with **no new primitive, no new run state and no scheduler**: a `human` step is just a step whose
outgoing guard reads a `ctx` key that does not exist yet, so the executor's existing "no transition
fired" outcome parks it — and writing that key from outside makes the same guard fire on resume.
Only two capabilities were missing: **read `ctx` in a guard** (U1) and **write `ctx` from outside**
(U3).

- **U0 (graph-dba) — two additive queries, no DDL.** `QUERIES.md` §12.12 `start_run_untriggered`
  (a run with no chat `Message` anchor — finding F-2) and §12.13 `resume_run_with_ctx` (the ctx write
  **folded into** the existing resume CAS, decision D-F, so no window exists where one writer's ctx is
  read by another's in-flight drive). PROFILEs beat the plan's assumption — `start_run_untriggered`
  is a single `Node By Index Scan` and is *cheaper* than §12.1; `resume_run_with_ctx` has no residual
  `Filter`. Zero-row contracts **verified, not assumed** (the CAS loser wrote neither the flip nor the
  ctx). `bootstrap_schema.sh` untouched, RAM ≈ nil. Query suite 241 → 256.
- **U1 (tdd-engineer) — the deterministic `cmp` guard family** (decision D-A):
  `{kind:'cmp', path, op, value}` + `all`/`any`/`not`, whitelisted ops, two whitelisted path roots
  (`ctx.`/`output.`), depth/width/node caps, `validate_cmp` + `render_label`, populated `rationale`.
  **No parser, no `eval`, no new dependency** — and named `cmp`, not `expr`, so DESIGN §13's "no
  expression library is built" stays literally true (`kind:'expr'` still raises). **Strict at publish,
  total at drive** (open item O-1): an unwhitelisted path root is rejected at seed time, but a missing
  path at drive is simply `False`. `test_guards.py` 33 → 143, including a De Morgan contrast pair that
  pins the `ne`-vs-`not` asymmetry side by side.
- **U2 (coder) — typed step handlers + two publish invariants.** `_execute_step` became an explicit
  dispatch: `agent`+LLM → the agent loop; `agent` without an LLM → the preserved empty stub (finding
  F-3, load-bearing for the whole offline estate, now documented as to *why*); `human`/`decision`/`wait`
  → three pure handlers. **⚠️ Behaviour change (R-3): `prompt`/`tool`/`message` and any unknown type
  now raise `NotImplementedError`** naming the plan, where they previously fell through to a **silent
  no-op** — a `decision` node used to "succeed" doing nothing. The M-1 fault net stamps `fail_run` and
  re-raises. `_validate_def_spec` gained both invariants, running **last** and each preceded by
  `_normalize_opaque` so the REST front door (which types `config`/`guard` as `str`) cannot escape
  them. **Named fixture edits, exactly as budgeted by gate findings B-1/B-2:**
  `test_executor.py:345`'s `{"key":"end","type":"task"}` → `type:"agent"` (it is executed by the real
  loop inside the Defect-B regression pin, which the new `NotImplementedError` would otherwise have
  killed), and the `type:"human"` fixtures in `test_api.py` + `test_services.py` that declared no
  `waitsForHuman`; the five affected `pytest.raises` gained `match=` so none went vacuous. Six
  mutations, all killed.
- **U3 (coder) — start-without-trigger + the human-input channel** (decision D-B, REST):
  `POST /workflow-runs` and `POST /workflow-runs/{id}/input`, with the D-G five-handler error map
  (`WorkflowRunNotFoundError` 404 · `WorkflowRunNotWaitingError` 409 · `WorkflowInputRejectedError`
  400 · `WorkflowConfigError` 400 · `WorkflowEngineDisabledError` 503) registered via a table — no
  blanket `RuntimeError` handler. Reserved ctx keys (`threadId`, `error`) are rejected at the
  **service** layer so MCP inherits the rule, closing the latent bug where a caller-set `threadId`
  would let any chat message resume a process run. Submitted input is validated against the parked
  step *before* the merge (D-H), so a mistake costs no step budget. A drive fault reports the run's
  **graph truth** — `status` *and* `ctx` from the same post-fault re-read — and a non-terminal
  re-read re-raises rather than dressing a zombie run as success.
- **U4 (coder) — the proof def, its seed and an offline acceptance test.**
  `server/falkorchat/proof_defs.py` ships `ACCESS_REQUEST_DEF`: **`access-request@v1`**, six steps /
  six transitions, submit→route→approval→provision→activate\|rejected, `human`×2 / `decision`×3 /
  `wait`×1, the four `cmp` ops `exists`/`in`/`eq`/`truthy`, two terminal outcomes,
  `ACCESS_REQUEST_MAX_STEPS = 24`. (The key is **`access-request`**, not `onboarding` — that key
  collides with long-standing test fixtures.) `scripts/seed_workflows.sh` now loops over **both** defs.
  `server/tests/test_process_flow.py` drives all three §4.3 paths — privileged (8 steps), standard hire
  (6, exercising conditional-beats-unconditional from the *losing* side) and rejected (6) — through the
  service layer, **fully offline: no LLM, no network, no `live` marker**. It imports the same constant
  the seed script publishes, so seed and test cannot drift.
- **U4b (coder) — the implementation gate's findings, closed at the right layer.** A drive fault on
  the `@mention` start path was left with **no log line anywhere** once D-G's catch swallowed it —
  fixed with one `logging.exception` inside the existing `except`, envelope byte-unchanged and the
  catch deliberately *not* gated to REST callers. **Open item O-6 was fixed, not filed:** a def
  published with zero transitions collapsed `_PUBLISH_CYPHER`'s trailing `UNWIND` **after** the steps
  and `START` were written, raising `IndexError` on a partial write — and because publish is
  create-only, the corrected retry was a silent no-op on a permanently poisoned `(key, version)`. A
  `_validate_def_spec` rule (running last, like the other two) now rejects it **before any repository
  call**. An empty input body is likewise rejected before it can win the CAS and burn a step. The
  acceptance test moved to the test-only version **`access-request@v1-test`** so a finished pytest
  session can never leave a *test's* publish masquerading as a real seed of the production pair —
  overriding **only** the version key, so the anti-drift property survives.
- **U5 (teco) — closeout.** DESIGN §6.1 rewritten for what the engine actually executes (including
  finding **F-1**'s two-part correction: `TRANSITION.on`/`StepResult.on` are **vestigial and
  descriptive only**, and guards sort by `(guard == "", order)` — *conditional first*, `order` only a
  tie-break within each class), §6.3 gained the proof pointer + the K-025 handoff note, §13 amended
  with `cmp`-not-`expr`. K-024 → ✅, K-025 marked unblocked, and three items filed rather than folded
  in: **K-028** (workflow timers — `wait` is signal-driven because this system has **no scheduler**;
  decision D-C), **K-029** (converge the seed def sources into `proof_defs.py`; `triage@v1`'s literal
  is still inline in `seed_workflows.sh` — carries the unenforced symmetric `decision` publish
  invariant of nit n-3), **K-030** (allow zero-transition defs by guarding the `UNWIND` the way §4's
  mention block does, instead of rejecting them; folds in the residual **publish-only** asymmetry —
  `materialize_snapshot` reuses the same unguarded query).
- **Gates.** Plan gate: **request changes** (2 blocker / 6 major) → plan v2.1, all closed in text and
  re-verified diff-scoped. Implementation gate U0–U4: **approve with suggestions**, 0 blockers, two
  majors (M-A the swallowed stack trace, M-B the reachable O-6 route). Re-gate U4b: **approve with
  suggestions**, no blockers, all seven findings closed *as ruled* — not more, not less. Both gates
  independently re-verified the `_drive_loop` SHA and confirmed the blast radius on the existing test
  estate was exactly the named fixtures, with **no assertion weakened, deleted or made conditional**.
- **Process incident (contained, no work lost).** During U3 the implementer twice reached for a
  tree-mutating git command to *read* or *undo* something; the second (`git checkout <path>`) reverted
  `services.py` to HEAD and destroyed the unit's work in one command. It was reconstructed and
  independently verified against `efdeeb3` — no top-level symbol lost, every diff deletion accounted
  for. **Standing rule now in every implementer brief:** never `stash`/`checkout <path>`/`restore`/
  `reset` a working tree — read baselines with `git show <ref>:<path>`.

## 2026-07-19 — M3 K-022 Landing 2: trigger + triage proof flow (U11–U14), analyst-gated — **U15 not run**

Second landing of **K-022 — LLM-native workflow executor**: the `@mention` trigger, the triage proof
flow, and the two live defects that landing it exposed. Delivered by the teco-coordinated chain —
**tdd-engineer** (U11 trigger wiring + U12 REST run inspection, and later both defect reproduction
suites), **coder** (the U13 workflow seed, the D14 revert, and the gate's code major) — with the
**mandatory analyst review gate** as a non-negotiable done-condition. Plan `docs/plans/m3-executor.md`
+ the U11 design patch `docs/plans/m3-executor-landing2.md`; coordination log
`docs/plans/m3-executor-coordination.md`. New server baseline: **pytest 283 → 350 passed, 0 skipped,
1 deselected** (with FalkorDB up; teco re-verified independently). Query suite unchanged at
**241/241** — Landing 2 has **zero** graph/DDL/QUERIES surface.

- **U11–U12 (tdd-engineer):** `trigger.py` `WorkflowTrigger.maybe_trigger` (§6 ordered rule:
  loop-guard → resume-if-waiting → `@mention`-to-start → fall through to the M2 responder), one
  handler per request (trigger XOR responder), `WORKFLOW_ENABLED`/`TRIGGER_DEF_KEY` config **default
  off** so the baseline stays network-free; **Option B** emission linking (buffer during agent-node
  execution, `StepRun-[:PRODUCED]->Message` after `_record`, keeping the §2.1 A/B/C loop and
  `record_step_and_advance` byte-for-byte); the Landing-1 **M-1** fault net (`_drive` wraps
  `_drive_loop`, so a mid-drive fault can no longer leave a zombie `status='running'` run); agent-node
  thread context (`_read_thread_context`, window 20); `GET /workflow-runs/{id}` + `/step-runs` +
  `/trace`. Its own analyst gate (`docs/reviews/m3-executor-landing2-impl.md`) came back
  **approve-with-suggestions, 0 blocker / 0 major**.
- **U13 (coder):** `scripts/seed_workflows.sh` — publishes + materializes `triage@v1` (three
  `type:'agent'` steps, intake→research→answer) through the **service layer**, not raw Cypher;
  `start_server.sh` seeds it before uvicorn.
- **U14 (tdd-engineer):** `tests/test_workflow_live.py`, the `live`-marked AC-1…AC-4 e2e, delivered
  **deliberately RED** — it pinned two real defects rather than being bent to green.
- **Defect A — the intake→research guard could never fire (fixed).** Structural, not prompt
  calibration: `executor.py` passed `thread=None` and `guards.py` declared the parameter but never
  read it, so the DS-prescribed **recent-turns fallback (N=6) did not exist** and the judge always
  saw an empty understanding. Fixed **at the seam, not in a prompt**: the thread window the agent node
  already reads rides out on `StepResult.thread` → `thread=result.thread` → `guards._recent_turns`,
  with the DS precedence (understanding primary, turns only when empty). Zero extra graph reads; the
  locked `_drive_loop` untouched. Design: `docs/plans/m3-guard-thread-context.md`.
- **Defect B — a hallucinated `@mention` failed the whole run (fixed).** Tool errors are now
  survivable: a failed dispatch returns an error string the model can act on instead of propagating
  to `fail_run`. Pinned at drive level by
  `tests/test_executor.py::test_hallucinated_mention_does_not_fail_the_run`.
- **D14 — S5 reverted.** The intake `{"understanding":{…}}` JSON instruction did reach the *primary*
  extract-then-judge path, but **regressed live intake advancement 10/10 → 3/10** on the shipped
  Qwen3-4B (the model filled `missing` with forensic demands on every turn and the uncalibrated judge
  suspended). It was **surgically reverted** from `scripts/seed_workflows.sh`, with the removal site
  commented so it is not re-added; the separately-measured **Defect-C prompt mitigations were
  retained**. Consequence to carry: the shipped guard runs only on the **degraded RECENT-TURNS tier** —
  a `guard_judgment` citing turn text is expected, not a defect — and the `understanding` primary tier
  is unreachable in this cut by design.
- **D16 — tool-error split (propagate + log).** `UnknownActorError`/`ThreadNotFoundError` — and any
  future `ServiceError` subclass — **propagate** to the M-1 fault net; only an explicit fail-closed
  allowlist (`UnknownMemberError`, `InvalidSearchQueryError`) is absorbed as a model re-prompt, and
  every failed dispatch logs unconditionally. These are deployment misconfigurations, not
  model-correctable arguments; absorbing them produced a run reaching `done` having posted nothing.
  Closes analyst finding M-2.
- **Analyst gate: `approve with suggestions` — 0 blocker / 2 major / 3 minor / 3 nit**
  (`docs/reviews/m3-guard-thread-context-impl.md`, on commit `aa8b813`). All five mandatory
  confirmations affirmative, including the `_drive_loop` byte-identity (SHA `71055f756280`, unchanged
  across `514346b`/`c3cc239`/`aa8b813`). **Both majors closed before the commit:** **M-1** (doc drift —
  `m3-executor.md` §8 documented prompts the reverted script no longer seeds) and **M-2** (the silent
  `ServiceError` catch, closed by D16). The three minors + three nits are carried on
  [`BACKLOG.md`](./BACKLOG.md) under K-027 so they cannot rot.
- **D13 capability probe (data-scientist):** a fits-16GB comparison of the shipped `qwen/qwen3-4b-2507`
  against Ministral 3 3B (Q8_0), config/env-only. **Ministral loses — no model swap.** Intake
  advancement 3/10 vs 0/10; AC-4 terminal post 2/3 vs not-measurable. Two findings routed to K-027: the
  fuzzy-guard judge's **bare `json.loads` is model-fragile** (a fenced ```` ```json ```` reply made all
  26 golden cases unparseable — the shipped Qwen path is unaffected), and Ministral is actually
  *better* at the terminal tool call. Note: `docs/plans/m3-capability-probe-ml.md`.
- **D15 parity repair (graph-dba, user-authorized destructive op on this dev box).** The stale
  `ws:acme` `WorkflowDefSnapshot` was deleted and `triage@v1` republished into **both** `reference` and
  `ws:acme`, resolving a **split-brain** in which the def had been wiped while the stale snapshot — the
  thing the executor actually drives — survived. Throwaway graphs `ws:probe`/`ws:live` dropped. No
  `WorkflowRun` existed, so nothing was severed. Environment-specific authorization: on a shared graph,
  `DETACH DELETE` on a def's `Step`s severs live runs' `AT_STEP`/`OF_DEF` and is a data-loss event.
- **⚠️ Two environment hazards, now documented in `AGENTS.md`:** (1) `server/tests/conftest.py`'s
  `wf_repo` fixture wipes the global `reference` graph — so it is **not only `test_queries.sh`**: a
  plain `pytest` with the DB up destroys the published def while leaving the `ws:acme` snapshot, the
  same silent split-brain, from the command we treat as the routine baseline. Re-run
  `seed_workflows.sh` after either. (2) Published defs are effectively **immutable** —
  `repository._PUBLISH_CYPHER` is `MERGE (st:Step …) ON CREATE SET st.config`, so editing a prompt and
  re-seeding prints a clean `already present — no-op` while the old config stays live.
- **❗ Explicitly NOT done — U15 (qa-engineer acceptance, = K-025) was not run.** Per decision **D12-B**
  the executor **mechanism** is proven (Defect A dead; the flow reaches `done` with the judge reasoning
  from real evidence), while live-triage **reliability** is descoped to K-027: the terminal
  `post_message` call is unreliable on a 4B (**Defect C** — AC-4 posting measured ~2/8, then 0/3 after
  a strengthened prompt, then 2/3 in the probe replay: unreliable in every measurement) and the judge is
  still **uncalibrated**. **Landing 2 is delivered and gated, not accepted** — AC-2b/AC-3/AC-4 remain
  model-gated and only structurally demonstrated, and M3 does not reach ✅ on this landing.

## 2026-07-18 — WSL2 memory diagnostic produced; fix parked (not applied)

Read-only devops diagnostic of the WSL2 memory-overload crashes on the downgraded 16GB host,
persisted at `docs/plans/wsl2-memory-diagnostic.md`. **Verdict: ballooning confirmed by defaults** —
WSL2 runs uncapped at its 8GB default (50% of the host) with `autoMemoryReclaim` off, overcommitting
host RAM alongside Windows-side LM Studio (not reproduced live — FalkorDB was down during the run).
Recommended fix (`memory=6GB` + `swap=4GB` + `autoMemoryReclaim=gradual` in `C:\Users\mauri\.wslconfig`,
keeping `networkingMode=mirrored`; needs `wsl --shutdown`) was **parked, not applied, per the user's
decision** — un-park if the crashes recur. Tracked as a Parking-lot bullet in
[`BACKLOG.md`](./BACKLOG.md) (`## Parking lot / ideas`). Docs-only; no config or code changed.

## 2026-07-12 — M3 K-022 Landing 1: LLM-native workflow executor (U1–U10), analyst-gated

First landing of the reframed **K-022 — LLM-native workflow executor**: the offline executor +
node capabilities (Phases 0–3, units U1–U10). Delivered by the **teco-coordinated
graph-dba → tdd-engineer → coder** chain with a **mandatory analyst review gate** — the team's
first fully-gated coordinated run. Plan `docs/plans/m3-executor.md`; coordination log
`docs/plans/m3-executor-coordination.md`. Trigger + proof flow (Landing 2, U11–U15) is a separate
later run, **not started**. New baselines: **query suite 193 → 241/241**, **server pytest 196 → 283**,
both green (teco re-verified independently).

- **U1–U2 (graph-dba):** `bootstrap_schema.sh` adds `TraceEvent.traceId` index **then** UNIQUE
  (index-before-constraint, idempotent). DESIGN §5.1/§5.2/§6.1/§6.2/§7.1/§13 reconciled — LLM-judged
  guards + the `type:'agent'` node kind, §13 guard-language open question marked resolved, and the
  stale `EMITTED` on StepRun→Message corrected to **`PRODUCED`** (§5.1/§5.2). QUERIES §12 = twelve
  live-verified/PROFILEd run / step-run / trace queries. The M4
  `WorkflowRun-[:LAST_STEP_RUN]->StepRun` tail pointer makes `record_step_and_advance` an O(1)
  atomic advance (no chain-walk). No new index — resume rides the existing `status` index.
- **U3–U5 (tdd-engineer):** `repository.py` — the §12 methods 1:1 + `WorkflowRunNotFoundError` /
  `StepBudgetExceededError`. New `executor.py` — `WorkflowExecutor` (the §2.1 A/B/C loop),
  `Tracer`/`NullTracer`/`GraphTracer`, run-level step budget, monotonic StepRun clock.
  `services.py` — start/resume/read-run methods, tenant seam respected. The slice-1 `start_key` =
  `start:True` contract was kept.
- **U6 (coder):** `llm.chat(messages, tools) -> ChatResult` with dual-shape parsing (native
  `tool_calls` field primary, content-embedded-JSON fallback); `complete()` preserved byte-for-byte.
- **U7–U8 (tdd-engineer):** `guards.py` `evaluate_guard` (DS-note Q1 extract-then-judge,
  `{decision,rationale}`, bias-to-suspend on ambiguity; `""`=unconditional; `expr`/unknown =
  `NotImplementedError` seam, M7). `executor._run_agent_node` — a bounded, tool-scoped agent loop
  with defensive **AC-6** rejection of ungranted/malformed calls (re-prompt, never dispatched) and
  graceful `maxIterations` exhaustion. The §2.1 loop was left byte-for-byte unchanged.
- **U9–U10 (coder):** `tools.py` `ToolRegistry` + `post_message` (§4 write as the agent →
  `PRODUCED` link via `services.link_step_emission`), `graphrag_retrieve` (Q2 τ≈0.5 cutoff / cap 5 /
  floor 1 / abstain), `human_handoff` (registered capability, granted to no node). `McpToolClient`
  MCP-client seam — stub-tested in-memory; real external servers deferred.
- **Analyst gate:** `docs/reviews/m3-executor-impl.md` — **approve-with-suggestions, 0 blockers**
  (1 major, 3 minor, 3 nit). Major **M-1**: `executor._drive` has no top-level `try/except`, so an
  unexpected mid-drive exception leaves the run stuck at `status='running'` — a permanent
  un-resumable zombie once live defs/tools run (not a green-suite blocker; the offline path is
  deterministic). Both deliberately-deferred seams — live `PRODUCED`-link ordering and agent-node
  thread-message context — were ruled **acceptable for Landing 1** and carried to U11.
- Layering held (no Cypher outside `repository.py`); D1–D5 and the M4/M7 decisions honored; AC-5
  (trace on/off) and AC-6 hold by construction; the default app import stays network-free. Cost
  datapoint recorded in the coordination doc: **~1.20M subagent tokens / 238 tool uses / 6
  delegations + the gate** (the first measurable cost/benefit reading for the review gate).

## 2026-07-11 — Docs unification: kaizen/ retired into docs/ (repo module convention)

Unified the component's two documentation homes into one `docs/` tree — the repo-wide module
convention now defined in the root `AGENTS.md` (agent-folder `claude/<agent>/kaizen/` pairs are
a separate convention and unchanged):

- `kaizen/plan.md` → **`docs/BACKLOG.md`** (living backlog; K-numbered items unchanged) and
  `kaizen/history.md` → **`docs/HISTORY.md`** (this file); `kaizen/` removed.
- New **`docs/archive/{plans,test-plans,test-reports}/`** for frozen documents of closed
  milestones. Moved: plans `m1-chat-mcp`, `m2-groundwork`, `m2-groundwork-queries`,
  `m2-graphrag`, `m2-agent-participant`, `doc-consolidation-sweep` (delivered 2026-07-05 —
  header was stale); all four M1/M2 test-plans; all four test-reports. Active M3 plans,
  `m1-cleanup`, `graphrag-eval-ml` (K-026 pending), `demo-environment-bringup` (living
  runbook), and `reviews/` stay in place.
- **Rule going forward:** a plan/test-plan/report moves to `archive/` (same subdir name) when
  its milestone closes, with inbound links fixed in the same change.
- Inbound references rewritten repo-wide (docs, `AGENTS.md` key-docs table, `README.md` tree,
  server source comments, `test_queries.sh`, `.dockerignore`, `.claude/settings.local.json`,
  agent kaizen logs citing concrete paths). Old dated entries below keep their period prose but
  their paths were updated to resolve.

Moved the engine off the floating `falkordb/falkordb:edge` tag to the tagged release
**`v4.18.11`** (module `41811`, Redis 8.6.3): `scripts/start_falkordb.sh`, `compose.yaml`,
and the CI service container (`.github/workflows/falkor-chat.yml`) now pin it; the
salesperson component's `start_falkordb.sh` moved with it. Container recreated on the same
`falkordb-data` volume after an explicit `SAVE` — all graphs (`ws:acme`, `reference`,
`ws:test`) survived. Re-verification per the quirks-file rule: **query suite 193/193,
server pytest 196 passed** on the pinned build. Current-state docs re-stamped (`AGENTS.md`
header, README, `docs/QUERIES.md`, `docs/DESIGN.md` §2 callout,
`claude/graph-dba/falkordb-quirks.md`). Rationale: edge is a moving target that forced
verify-everything churn; a pin makes the live-verified facts durable until a deliberate
upgrade.

## 2026-07-09 — K-020 + K-021: M3 slice 1 (workflow defs + snapshot materialization) delivered

First slice of **M3 — Workflow engine**, delivered end-to-end by the **teco-coordinated
architect → graph-dba → tdd-engineer chain** (the teco K-001 nested-delegation validation run —
see `claude/teco/docs/HISTORY.md` 2026-07-09). Architect decomposed all of M3 into
**K-020…K-025** and wrote the slice-1 plan (`docs/plans/m3-workflow-engine.md`, Part A + Part B);
coordination log at `docs/plans/m3-workflow-engine-coordination.md`. Suites verified
independently after integration: **query suite 149 → 193/193, pytest 156 → 196.**

- **K-020 — def model in `reference`.** *graph-dba gate:* `Step.stepUid = "{defKey}:{version}:{stepKey}"`
  (architect's synthetic key — `Step.key` is unique only within a def) with index + UNIQUE in
  `reference`; one justified model addition — a `-[:HAS_STEP]->` containment edge, because the
  plan's `STARTS WITH stepUid` scoping PROFILEd as a label scan (HAS_STEP gives index-anchored
  O(steps-in-def) reads); canonical `QUERIES.md §11` (publish, read-def, list/get def), live-verified
  + PROFILEd; DESIGN §6.1/§7.1/§7.2 updated. *tdd impl:* `db.reference_graph` seam, reference-graph
  repository methods (1:1 with §11) + typed errors, `services.publish_workflow_def` with spec
  validation **before any write** — `start_key` resolved as "exactly one step declares `start: True`"
  (implementer's call, plan had no param; lock the contract at K-022).
- **K-021 — snapshot materialization.** *graph-dba gate:* workspace `Step.stepUid`/`Step.key` DDL
  in `bootstrap_schema.sh` (additive); materialize / read-snapshot / list-snapshot queries in §11.
  *tdd impl:* two-phase `services.materialize_def` (read `reference` → idempotent MERGE into
  `ws:{id}`; not atomic across the graph boundary, retry completes), size-bounded schemas, thin
  REST surface. **Structural parity proven** (publish → materialize → snapshot `==` reference def)
  + idempotency; reference-wiping test fixture added.
- **Scope discipline:** executor (K-022), chat linkage (K-023), proof flows (K-024) explicitly not
  built; `ws:acme`/`reference` additive-only; §13 guard-language decision confirmed **not forced**
  by slice 1 — returns to the user at K-022's architect pass.

## 2026-07-08 — K-008 + K-013 + K-014 + K-015: M2 GraphRAG delivered → milestone M2 done

End-to-end GraphRAG loop, delivered as the full graph-dba→tdd→coder→qa sequence and
**QA-accepted (K-015, PASS, zero defects)**. Prerequisite: a devops LM-Studio reachability spike
confirmed `http://localhost:1234/v1` reachable from WSL2, embedding dim **1024**, both models live
(`text-embedding-qwen3-embedding-0.6b`, `qwen/qwen3-4b-2507`).

- **K-008 — retrieval core.** *graph-dba gate:* verified the §6 hybrid ANN query + `SET m.embedding`
  live against a 1024-dim workspace, `GRAPH.PROFILE` confirmed the vector index is hit, Entity
  expansion no-ops cleanly; raised `test_queries.sh` **126 → 135**; deliverable `docs/archive/plans/m2-graphrag.md`.
  New quirk logged: a wrong-dimension `vecf32` write is *silently accepted* then drops the node out of
  the ANN index → validate length client-side. *tdd impl:* `repository.set_embedding` (client-side dim
  validation, `EmbeddingDimensionError`) + `repository.hybrid_search` (§6, channel/workspace variants)
  + `services.hybrid_search` (`RAG_QUERY_TIMEOUT_MS`) + `embedding.py` (`Embedder`/`LMStudioEmbedder`/
  `EmbeddingWorker`, injected transport). pytest **110 → 123**.
- **K-013 — AI Agent participant + `EMITTED` provenance.** *graph-dba gate:* defined
  `(answer:Message)-[:EMITTED]->(seed:Message)` with `score`+`rank` props, riding **inside the guarded
  §4 write** (exactly-once under `dupMsg` replay, no relationship constraint needed); canonical
  `QUERIES.md §10`; raised `test_queries.sh` **135 → 149**; deliverable `docs/archive/plans/m2-agent-participant.md`.
  *tdd impl:* `repository.post_agent_answer`/`read_provenance`/`read_citing_answers`, `llm.py`
  (`LMStudioLLM`), `responder.py` (`AgentResponder` — `@mention` trigger, loop guard on
  `role:"assistant"`, LLM/embedder before the guarded write ⇒ failure posts nothing). **Decisions
  (user):** trigger = agent `@mention` only; **every** posted message is embedded out-of-band (corpus
  grows) — both wired via FastAPI `BackgroundTasks`. pytest **123 → 154**.
- **K-014 — live wiring + web.** Served app builds the real embedder/worker/LLM/responder gated on
  `FALKORCHAT_ENABLE_AGENT` (default off → imports/tests stay network-free); `config` gained
  `AGENT_ID`/`AGENT_NAME`/`ENABLE_AGENT` + `LLM_*`; new `scripts/seed_demo.sh` registers the
  `assistant` agent + demo channel/thread; `start_server.sh` now exports `FALKORCHAT_EMBEDDING_DIM=1024`
  + enables the agent + seeds; `server/.env.example` documents runtime env. Web renders assistant
  replies (AI badge) + reader `isMention`; `displayName` added to since-reads (`QUERIES.md §9.1/§9.2`
  in lockstep, suite unaffected). pytest **154 → 156**.
- **Provisioning (ops):** served tenant `ws:acme` dropped and re-bootstrapped at `EMBEDDING_DIM=1024`
  (user-confirmed clean build); vector index verified at 1024.
- **K-015 — QA acceptance (the gate).** Black-box pass across REST + MCP + web + the running
  responder: out-of-band embedding, cosine-ASC ranking, agent answer with `EMITTED` provenance on all
  read surfaces, loop guard, failure isolation, dormant-Entity path — **PASS, no defects.** Plan/report:
  `docs/archive/test-plans/m2-graphrag.md`, `docs/archive/test-reports/m2-graphrag-report.md`.
- **Parked → M2.5 (deferred, not on the M2-green path):** real auth/tenancy (K-016), transport-level
  externally-authenticated agent actor (K-017, K-007 QA carry-over), real-time push (K-018);
  channel-scoped retrieval read (responder currently workspace-wide — trigger self-cites as rank-0);
  `ensure_agent` doesn't persist `displayName`; reverse-provenance not on a public route.
- **Suites:** pytest **156** / query suite **149/149**.
- **Milestone:** closes **milestone M2 — GraphRAG → ✅.** Next milestone: **M3 — Workflow engine.**

## 2026-07-06 — K-012: web request/response UX polish → M1 complete

- **What (client-side only, `web/` — no server/schema/query change):** de-staled the M1
  request/response web path. Three changes in `web/app.js` + `web/index.html`:
  1. **Incremental polling** — the open thread refreshes via `GET …?since=&limit=50` (bounded,
     `since`-anchored, no `NEXT*` walk, no cursor), replacing the full re-fetch-after-post.
  2. **Inline non-blocking toast errors** — replaced **both** `alert()` sites with inline toast
     rendering so a failed post/action no longer blocks the UI.
  3. **Clickable search results** — a search row now opens the message's thread via the `threadId`
     carried on search rows (K-007 denorm).
- **Scope guard:** `web/app.js` + `web/index.html` only — no `.py`, `QUERIES.md`, `test_queries.sh`,
  `bootstrap_schema.sh`, schema, or `scripts/` touched; suites unaffected. Manual-smoke-only per the
  K-005 precedent (no web test harness; `node` not on the box).
- **Parked follow-up → K-014:** polled (`?since=`) message rows carry `authorId` but no
  `displayName` (a `coder` left a code comment in `app.js`); resolving it needs a small server
  change to include `displayName` on since-read rows — folded into the K-014 web-M2 pass.
- **Suites:** pytest **110** / query suite **126/126** (unchanged — no code under test touched).
- **Milestone:** with K-011, closes **milestone M1 — Chat core → ✅**.

## 2026-07-06 — K-011: M1 DoD closeout — append-path load harness + hot-read PROFILE + RAM budget

- **What (devops, with a `graph-dba` PROFILE sub-pass):** closed the M1 append-path load-test +
  hot-read `GRAPH.PROFILE` DoD and folded a per-workspace RAM budget into DESIGN.
  1. **Load harness** — new `scripts/load_test.sh` + `scripts/load_append.py` drive the
     **service-layer append path through REST** (16 concurrent posters, 3000 msgs, 0 errors)
     against an isolated `ws:load` graph. Measured **~614 msg/s; p50/p90/p99 = 24.4/30.6/40.7 ms**.
  2. **Hot-read PROFILE** — `GRAPH.PROFILE` on the four hot reads (§4 thread read, §9.1 & §9.2
     since-reads, §5 search) — **all index-backed (`Node By Index Scan`), none degraded to a
     `NodeByLabelScan`**; raw plans archived by the harness.
  3. **RAM budget** — chat-core floor **~1.06 KB/msg** (measured `INFO memory` `used_memory`
     delta) ⇒ **~101 MB per 100k-msg workspace**; packing table folded into DESIGN §11.1/§11.2.
- **Files:** new `scripts/load_test.sh`, `scripts/load_append.py`; `docs/DESIGN.md` §11.1/§11.2;
  `AGENTS.md` Key-scripts row; `.gitignore` (`.load-out/`).
- **Scope guard:** read-only measurement + docs/harness — **zero new per-workspace RAM cost**;
  no `QUERIES.md`/`test_queries.sh`/`bootstrap_schema.sh`/schema change. Ran against `ws:load`
  (create + delete), never `ws:acme`.
- **Suites:** query suite **126/126** · pytest **110** (green).
- **Milestone:** with K-012, closes **milestone M1 — Chat core → ✅**.

## 2026-07-05 — K-021: §13 open-questions reconciliation + identity-authoritative decision

- **What (doc-only, no code/schema/query/script change):** recorded a newly-made design decision and
  brought `docs/DESIGN.md` §13 "Open questions" back in line with reality.
- **New locked decision — identity source of truth:** the **`identity` graph is authoritative
  (standalone)**, not a projection of an external IdP. The system is self-contained: the `identity`
  graph owns global user identity + auth principals; per-workspace `User` nodes remain membership
  projections of it (consistent with §3 topology). User-approved 2026-07-05; steers K-016 (real auth).
  - Added as a row in **DESIGN §1.2** (the authoritative detailed register; "Detailed in" → §3, §14.3).
  - Added a matching one-line pointer in **`AGENTS.md`**'s decisions index (`… → §3`, no rationale).
- **§13 pruned to genuinely-open questions only:** removed **Embedding model & dimension** (resolved;
  home §1.3) and **Identity source of truth** (now decided; home §1.2) — no resolved-pointers left in
  the "Open questions" list.
- **§13 reworded:** **Bolt vs. RESP** → **Real-time gateway transport** (M1 app driver settled = RESP
  via `falkordb-py`; only the M2.5 push-gateway transport is open, → K-018). **Live config defaults**
  → prefixed **Pre-production config review** and dropped TIMEOUT from the still-to-review set (TIMEOUT
  1000ms already reviewed & kept — K-007, §10; other knobs retained). The three genuinely-open bullets
  tagged with owners: workflow guard expr language (→ M3), retention (→ K-011 data), cross-workspace
  analytics (mechanism open, no milestone).
- **`docs/BACKLOG.md` reconciled:** K-016 "Inputs/prereqs"/Owner/scope now read as **decided** (identity
  graph authoritative; K-016 no longer needs the user for that axis — implements per §1.2); the
  `m2-auth-tenancy.md` recommended-doc row and the milestone-map note updated likewise; removed
  "identity source of truth" from the parking-lot "remaining open questions" line (real auth / K-016 stays).

## 2026-07-05 — K-019: documentation-inconsistency sweep (test counts, embedding decided, M2/M2.5 scope)

- **What (doc-only, no code/schema/query/script change):** reconciled stale numbers and
  contradictory milestone wording in `README.md` and `docs/DESIGN.md`. Counts sourced from a
  **live suite run** (`./scripts/test_queries.sh` → 126/126; `server && pytest -q` → 110 passed)
  with FalkorDB up.
  - **Test counts → true 110 pytest / 126 query suite.** `README.md`: `115/115 passed`→`126/126`
    (step 4 expected output); `(115 assertions)`→`(126)` (repo-layout comment); `(75 tests)`→
    `(110 tests)` and `# 98 passed`→`# 110 passed` (M1 row + pytest example). `DESIGN.md` §12 M1
    roadmap bullet `built and green (70 tests)`→`(110 tests)`. The README M0 roadmap figure
    `(92/92)` was **re-labelled historical** (`92/92 at M0 baseline`), not bumped — it records M0.
  - **Embedding model no longer "open."** `DESIGN.md` §11 RAM line: `default stays 1536
    (embedding model still open, §13)`→`(chosen per workspace); set EMBEDDING_DIM=1024 for the
    decided model (§1.3)`. `DESIGN.md` §13 open-questions "Embedding model & dimension" bullet
    replaced with a resolved pointer to §1.3 (Qwen3-Embedding-0.6B, `EMBEDDING_DIM=1024`). The
    `EMBEDDING_DIM=1536` *default* in scripts was intentionally **left untouched**.
  - **M2-vs-M2.5 scope aligned.** `DESIGN.md` §14.1 Transport/Real-time rows + §14.1 rationale
    note: real-time "deferred to M2"/"M2 real-time" → **M2.5** (agrees with §12 M2 = GraphRAG only
    and the kaizen deferred M2.5 track K-016/K-018). `README.md` M1 roadmap row
    "deferred to M2"→"deferred to M2.5". Auth references (§14.3 "when auth lands", §15.3
    "unauthenticated in M1") were already milestone-agnostic — no contradiction, left as-is.
- **Scope guard:** only `README.md` + `docs/DESIGN.md` (+ these kaizen files) touched — no `.py`,
  `QUERIES.md`, `test_queries.sh`, `bootstrap_schema.sh`, schema, or script changed; pytest 110
  and query suite 126/126 hold by construction (and were re-run green as the count source). The
  K-020 decision register (§1.1/§1.2/§1.3, AGENTS.md pointer index) was only *referenced*, not
  altered.

## 2026-07-05 — K-020: doc-architecture consolidation — DESIGN §1 single decision register

- **What (doc-only, no code/schema/query change):** applied the single-authoritative-home
  discipline (long applied to query bodies) to *design decisions*. `docs/DESIGN.md` §1 is now the
  one authoritative decision register; every other doc points to it.
  - **AGENTS.md decisions → DESIGN §1.2.** The 18-row "Decisions locked in" rationale table
    migrated into a new DESIGN **§1.2** detailed register (16 rows; `Message.role` inline + derived
    merged; "one graph per workspace" already lived in the §1.1 axes table). Each row is a
    statement + rationale + link to the body section (or QUERIES.md) that details the mechanics —
    no re-copied prose. AGENTS.md's section is now a terse two-column `Decision | Home` pointer
    index (rationale removed), kept — not deleted — as the quick do-not-reopen list.
  - **BACKLOG.md M2 stack → DESIGN §1.3.** The user-approved "Locked M2 stack decisions"
    (2026-07-04) graduated into a new DESIGN **§1.3** (embedding model/dim, agent LLM, runtime,
    VRAM, upgrade path); BACKLOG.md keeps a one-line pointer + the `EMBEDDING_DIM=1024` bootstrap
    reminder. K-0xx work items, sequencing, and parking lot untouched.
  - **A1 — GraphRAG dedup.** Deleted the drifted `cypher` block in DESIGN §8 (had lost its
    `LIMIT` and RETURN columns vs. the canonical QUERIES §6); §8 now points to QUERIES §6 in the
    §5.3 "shape-only, link the body" style. §8's design prose kept; QUERIES §6 untouched.
  - **A2 — coordination ADR promotion.** Added DESIGN **§6.3** (coordination is an M3 `WorkflowDef`
    of `kind:'process'`, not a flat `Task` node) with a back-link to `docs/archive/plans/m1-chat-mcp.md`
    Appendix B (which stays the ADR of record).
  - Added one new DESIGN §6.2 body line stating `ctx`/`input`/`output` are flat/serialised (D13).
- **Scope guard:** markdown docs only — no `repository.py`/`services.py`/`QUERIES.md` bodies/
  `test_queries.sh`/schema/scripts touched; pytest 110 and query suite 126/126 hold by
  construction. K-019 boundary respected (stale test counts, §13 "open"→"decided" wording, and
  §12/§14.1 scope left for K-019).

## 2026-07-05 — K-010: QA DEF-1 + DEF-2 closed (K-008 prerequisites)

- **What:** closed both defects from the K-007 QA pass, clearing K-008's gate. Coordinated
  delivery: **graph-dba** authored + live-verified the query layer, **tdd-engineer** wired the
  Python (strict red→green), verification re-run independently.
  1. **DEF-1 — member-id namespace guard (K-008 prerequisite).** Locked rule: member ids are
     **namespace-unique across `User`/`Agent`**. `ensure_user`/`ensure_agent` are now v2
     guarded-CREATE single-query bodies (QUERIES.md §2/§7, verified `Node By Index Scan` on
     both legs) returning an always-present `(created, existed, collided)` status row —
     idempotent re-ensure is a structural no-op; a cross-label collision writes nothing and
     raises `MemberIdCollisionError` (repository-level, re-exported by services);
     `existed AND collided` is a distinct corruption alarm. App startup with a configured
     actor colliding with an existing Agent id now **fails loudly** instead of silently
     minting a shadow `User` that eclipsed the Agent in every `coalesce(u, a)` lookup (the
     exact QA S3 repro). Same-label uniqueness constraints remain the concurrency backstop;
     the one-query-wide cross-label race window is documented, not closed (no engine
     cross-label constraint exists).
  2. **DEF-2 — fail-fast on unreachable FalkorDB.** `db.connect()` now passes
     `socket_connect_timeout`/`socket_timeout` (config-resolved `FALKORDB_CONNECT_TIMEOUT=5`
     / `FALKORDB_SOCKET_TIMEOUT=10`, env-overridable) and wraps failures in
     `FalkorDBUnreachableError` naming host:port + timeout + a start-script hint; a new
     `db.LazyFalkorDB` defers the first connection out of import — **importing
     `falkorchat.app` never touches the network** (the module-level `create_app()` used to
     hang ≥90s with zero output on WSL2's closed-port blackhole). Smoke re-verified: dead
     port → clean exit in ~6s with the actionable error. `app.py` docstring now matches
     reality.
- **Files:** `server/falkorchat/{repository,services,config,db,app}.py`;
  `server/tests/{test_repository,test_services,test_app}.py` + new `test_db.py`;
  `docs/QUERIES.md` §2/§7 (v2 ensures + contract table + locked rule);
  `scripts/test_queries.sh` (11 new DEF-1 assertions incl. PROFILE index checks);
  `AGENTS.md` (new locked-decision row; baselines) + root `AGENTS.md` (baselines).
- **Baselines (independently re-verified):** pytest **98 → 110**, query suite
  **115/115 → 126/126**; `reference` schema restored post-suite; `ws:acme` untouched.
- **Why:** DEF-1's silent misattribution was exactly the failure class K-007 closed, and it
  gated wiring real agent identities in K-008; DEF-2 bought dev/ops diagnosability on the
  README bare-`uvicorn` path (Compose was already shielded by `service_healthy`).

## 2026-07-05 — QA: acceptance pass on K-007 M2 groundwork

- **What:** black-box/acceptance QA pass at `94ab746`, scoped to what the K-007 dev suites
  structurally can't reach: concurrency through the real HTTP stack (single- and
  **two-process** writers), MCP-driven cursor paging over millisecond ties, agent `role` on
  every read surface, `backfill_thread_ids.sh` against real legacy-shaped data, and the
  actor-seam edges. Added `docs/archive/test-plans/k007-m2-groundwork.md` and
  `docs/archive/test-reports/k007-m2-groundwork-report.md`. Isolated `ws:qa` (created + deleted);
  `ws:acme`/`reference` untouched.
- **Result: PASS with two low-severity defects** — 18/18 items executed, 16 clean passes, on
  green baselines (server **98/98**, query suite **115/115**). Highlights: 12-way REST
  first-post hammer and a 20-write race across **two server processes** both yielded exactly
  one HEAD/TAIL and a contiguous chain; the cross-process run produced a **natural same-ms
  `createdAt` tie** and MCP cursor paging (`limit=3`) still delivered all 20 exactly once;
  agent-authored messages read `role: "assistant"` consistently on all five read surfaces;
  backfill script: 2 backfilled, then 0 (idempotent), `threadId: null` tolerated pre-backfill.
- **Defects (parked in `docs/BACKLOG.md`, not fixed here):**
  - **DEF-1 (low now, K-008 hazard):** no cross-label member-id uniqueness — a configured
    actor colliding with an existing `agentId` silently MERGEs a shadow `User` that eclipses
    the Agent in every `coalesce(u, a)` lookup (role derivation, `POSTED_BY`, mentions).
  - **DEF-2 (low, ops):** with FalkorDB unreachable, `uvicorn falkorchat.app:app` hangs
    indefinitely with zero output — `FalkorDB()` connects eagerly (no socket timeout) inside
    the module-level `create_app()`, falsifying the "building the app never requires a
    reachable FalkorDB" intent (hang-vs-refuse is WSL2-flavored; the eager import-time
    connect is real everywhere).
- **Why:** the prior QA report's top residual risks (concurrency/idempotency, agent
  authorship, ms-ties) were exactly K-007's targets — this pass closes that loop before K-008
  puts real agent writers on the system. No code under test changed.

## 2026-07-05 — K-007: M2 groundwork — agent authorship, v2 write-path guards, threadId denorm, composite cursors

- **What:** the six pre-agent-writer correctness/completeness items, landed per the approved
  plan (`docs/archive/plans/m2-groundwork.md`) over the graph-dba's live-verified query deliverable
  (`docs/archive/plans/m2-groundwork-queries.md`); plus the two server fold-ins.
  1. **Agent authorship** — §4 write paths resolve the author label-specifically (two indexed
     `OPTIONAL MATCH`es + `coalesce`), closing the `All Node Scan` *and* the silent no-op that
     made `Agent` authors unwritable; `services.post_message` derives `role` from the author's
     label via the new `repository.resolve_member_kinds` (`User → user`, `Agent → assistant`;
     replaces `existing_members` — one round trip for author + mention validation + role).
  2. **v2 self-guarding write paths (two reproduced defects fixed)** — each path wraps its write
     in a `FOREACH`+`CASE` guard inside the single `GRAPH.QUERY` and always returns a
     `(written, hadHead, dupMsg, authorFound)` status row (`repository.MessageWriteStatus`).
     Defect A: a same-`msgId` retry replay re-ran the relink clauses (NEXT self-loop, doubled
     `POSTED_BY`) — now a structural no-op reported as `dupMsg` = idempotent success. Defect B:
     two racing first-posts created two HEADs — the loser now refuses with `hadHead` and the
     service re-dispatches as subsequent (bounded 4-attempt loop; `Message` writes carry **no
     MERGE** — the uniqueness constraint stays as the verified all-or-nothing backstop).
     `REPLY_TO`-inside-the-guard live-verified in `test_queries.sh` (OQ4); repository fold-in
     waits for a reply surface.
  3. **`Message.threadId` denorm** — stamped inline by both write paths, deliberately unindexed;
     surfaced in §9.1/§9.2 since-reads, `/search`, and `GET /messages/{id}`. One-off
     `scripts/backfill_thread_ids.sh` (QUERIES.md §4.x; idempotent, HEAD-anchored, orphan
     caveat) — run against `ws:acme`: 0 backfilled (expected no-op, 0 messages).
  4. **Millisecond-tie correctness (reproduced page-boundary skip fixed)** — deterministic total
     order `(createdAt, msgId)` on both since-reads; formulation-A composite keyset predicate
     (still a bare `Node By Index Scan`); composite monotonic `ReadCursor`
     (`lastReadAt`, `lastReadMsgId`) — five scenarios verified, pre-K-007 cursors covered by
     `coalesce(…, '')`, no schema change; plus a lock-guarded monotonic per-process message
     clock in `Services` (same-ms ties impossible at the source). Explicit REST `?since=` keeps
     plain-`>` semantics (documented, OQ3).
  5. **TIMEOUT posture (docs-only, live-probed)** — keep legacy `TIMEOUT=1000`; per-query client
     override for future GraphRAG reads; **writes ignore TIMEOUT on this build** — bounded
     batches + input caps are the only write-path protection (DESIGN §10).
  6. **RAM line re-costed at 1024 dims (empirical)** — 12,387 B/message observed ≈ 12.4 KB ⇒
     ~1.25 GB per 100k-message workspace; `GRAPH.MEMORY USAGE` under-reports vector-index memory
     (size from `INFO memory` deltas) — DESIGN §11 rewritten; bootstrap default stays 1536 with
     an explicit choose-before-creation comment.
  - Fold-ins: `db.connect()` late-binds `config.FALKORDB_*`; `create_channel`/`create_thread`
    are plain `CREATE` (server-minted ids — creates documented **non-idempotent**;
    `create_thread` raises on a missing channel anchor).
  - Docs: QUERIES.md §2/§3/§4(+§4.x)/§5/§9 rewritten as the canonical v2 bodies; DESIGN
    §5.1/§5.3/§9/§10/§11/§12 (role values fixed to `user`/`assistant`, the falsified
    "idempotent via MERGE" claim replaced by the status-row contract); AGENTS.md decisions/
    facts/write-path rewrite; README + root AGENTS.md baselines.
- **Why:** prerequisites for AI agents writing concurrently (K-008): agents couldn't author at
  all, a client retry corrupted the thread chain, a first-post race forked it, and same-ms
  `createdAt` ties silently lost messages at cursor page boundaries.
- **Verified:** server suite **98 passed** (was 75; +23 — the plan's ≈95 estimate, exceeded by
  finer-grained regression tests); query suite **115/115** (was 92; +23 exactly as enumerated);
  `ruff check .` clean; defect regressions were watched fail red against the old code (replay →
  `(2 NEXT, 1 self-loop, 2 POSTED_BY)`; race → 2 HEADs) before the v2 queries landed; live
  8-worker concurrency hammer green (1 HEAD, 1 TAIL, contiguous chain of 8); backfill no-op
  proven on `ws:acme`.
- **Plan items:** K-007 ✅ done; K-008 (GraphRAG proper) unblocked; parking-lot fold-ins
  (`db.connect` bind, uuid `MERGE`) delivered. OQ6 (upstream FalkorDB filings: `GRAPH.MEMORY
  USAGE` vector under-report; one-shot instant-timeout anomaly) recommended to the user, not
  filed.

## 2026-07-04 — K-009: containerization (Dockerfile/compose) + CI + `falkordb-data` persistence fix

- **What:** first delivery-lifecycle pass for the component — container images, a compose stack,
  path-filtered CI, dependency pinning, and a critical data-persistence bug fix.
  1. **`falkordb-data` persistence fix (critical)** — `scripts/start_falkordb.sh` mounted the
     named volume at `/data` (the image's legacy `VOLUME`), but `falkordb/falkordb:edge` actually
     writes its Redis `dir` to **`/var/lib/falkordb/data`** (`FALKORDB_DATA_PATH`) — so **no graph
     data ever survived a container stop**; the volume persisted nothing. Live-verified 2026-07-04:
     data written under the `/data` mount vanished on restart; remounted at `/var/lib/falkordb/data`
     it survives. Fixed in the script (with an inline warning comment) and used in `compose.yaml`.
     `ws:acme` schema was re-bootstrapped after the fix (12 indexes).
  2. **`Dockerfile`** — M1 server image (`python:3.12-slim`): build context is the component root
     so the `server/` + `web/` sibling layout survives (app.py resolves `parents[2]/web`), editable
     install, non-root `appuser` runtime (install stays root-owned/read-only), `EXPOSE 8000`, and a
     `HEALTHCHECK` against the K-006 `GET /health` (200 only when FalkorDB answers).
  3. **`compose.yaml`** — two services: `falkordb` (same image/ports/volume as the script; redis-cli
     ping healthcheck) and `server` (built image, `FALKORDB_HOST=falkordb`, `depends_on:
     service_healthy`). The `falkordb-data` volume is declared **`external: true`** — compose must
     never create/re-create/remove the shared dev volume, and `down -v` is explicitly warned
     against. Header warns the script-started `falkordb-dev` container and compose share :6379 and
     the volume — never run both.
  4. **`.dockerignore`** — only `server/` (minus tests/venv/egg-info) + `web/` enter the build
     context; docs, kaizen, scripts, markdown excluded.
  5. **CI (`.github/workflows/falkor-chat.yml`)** — path-filtered to `falkor-chat/**` + the
     workflow itself; single job on ubuntu-latest with a **FalkorDB service container**
     (`falkordb/falkordb:edge`, health-gated) mirroring the local commands: `ruff check server` →
     server pytest (75-baseline) → `./scripts/test_queries.sh` (92/92-baseline). Deliberately
     tracks the floating `:edge` tag — the project's live-verified facts are pinned to it.
     **Never run yet** — first push to GitHub will tell (parking-lot item).
  6. **Dependency pins + ruff adoption** (`server/pyproject.toml`) — compatible-range pins for
     reproducible installs: `fastapi>=0.139,<0.140`, `uvicorn>=0.49,<0.50`, `falkordb>=1.6,<1.7`,
     `mcp>=1.28,<1.29`, `pytest>=9.1,<10`, `httpx>=0.28,<0.29`, `ruff>=0.14,<0.15`; ruff config
     (E,F,W,I / target py312 / line 100). Behavior-neutral import-order (I) fixes across
     `falkorchat/{api,app,services}.py` and `tests/{conftest,test_app,test_repository,test_services}.py`.
  7. **README** — compose run section added alongside the script path.
- **Why:** the component had no image, no one-command stack, and no CI; and the persistence bug
  meant the "durable" dev volume was silently empty — any container stop lost every graph.
- **Verified (2026-07-04 resume session):** fixed script started FalkorDB from a cold stop and
  `GRAPH.LIST` returned **`ws:acme`** — live proof graphs now survive downtime (`ws:k007scratch`
  residue also present, left untouched for the K-007 relaunch). Pins install-verified in a clean
  reinstall (fastapi 0.139.0, uvicorn 0.49.0, falkordb 1.6.1, mcp 1.28.1, pytest 9.1.1,
  httpx 0.28.1, ruff 0.14.14); `ruff check .` clean; server suite **75 passed**; query suite
  **92/92**. Compose stack itself not booted locally (shares :6379 + the volume with the running
  `falkordb-dev`); its build is exercised by CI on first push.
- **Plan items:** K-009 ✅ done; parking lot gains "verify the CI workflow goes green on first
  push". K-007 (graph-dba relaunch) is the next action.

## 2026-07-04 — K-006: post-M1 review follow-ups (navigation, bounds, health)

- **What:** small, high-value fixes from a 2026-07-04 full-project review; the review's larger
  findings went to the parking lot. Adapter/boundary changes only — no `QUERIES.md` query bodies
  or schema touched, so the 92-suite stays a pure regression guard.
  1. **MCP navigation dead-end closed** — `list_channels(limit)` + `list_threads(channel_id,
     limit)` MCP tools (7 total). Before, an agent could not discover an existing channel or
     thread id (workspace-wide `read_messages` rows omit `threadId` — still parked); it could
     only create its own space. Thin wrappers over the existing `Services` methods; discovery
     test updated, list→post→read navigation roundtrip added.
  2. **Input size bounds (RAM rule 6)** — `schemas.py` Pydantic constraints (text ≤ 8000,
     name/title 1–200, mentions ≤ 50) and `Query` bounds on list `limit`s (1–200). Message text
     lands in graph RAM *and* the full-text index; nothing capped it.
  3. **REST thread-read pagination** — `GET /threads/{tid}/messages?since=&limit=` maps to the
     existing §9.1 `read_thread_since` as a **pure read** (`since` defaults to 0 explicitly, so
     a browser poll never consults/advances the member's cursor — cursors stay agent-owned).
     No params keeps the full §4 read contract. Mitigates the unbounded `NEXT*0..` walk vs the
     1000 ms default `TIMEOUT` cliff on long threads (full fix = web client adoption, parked).
  4. **`GET /health`** — `services.ping` → `repository.ping` (`RO_QUERY RETURN 1`); 503 when
     FalkorDB is unreachable. Probe target for compose/CI (both parked).
- **Doc drift fixed (root `AGENTS.md`):** query-suite baseline claims corrected 67/67 → **92/92**
  (×2) — the stale numbers were loaded into every agent session.
- **Verified:** server suite **75 passed** (was 70; +5: MCP navigation roundtrip, health, body
  bounds, limit bounds, pagination — the pagination test injects a counting clock to sidestep the
  known same-ms `createdAt` tie caveat); query suite **92/92**.
- **Docs (same change):** `DESIGN.md` §14.4 REST table (+`/health`, real `?since=&limit=` shape,
  bounds note) and §15.2 tools table (+2 rows); `README.md` tools list + counts 70→75;
  `falkor-chat/AGENTS.md` count 68→75 (was already stale); `BACKLOG.md` parking lot extended,
  Last-reviewed bumped; this entry.

## 2026-07-02 — K-005: M1-final cleanup

- **What:** four small parking-lot items from the 2026-07-02 review, resolved test-first. All
  server changes are **adapter-only** (`mcp.py`, `api.py`) — no `repository.py`, `services.py`,
  `QUERIES.md`, or `test_queries.sh` touched, so the 92-assertion suite stays a pure regression
  guard.
  1. **`search_messages` MCP tool** — the existing `services.search_messages` (REST `GET /search`,
     `QUERIES.md` §5) is now exposed as a 4th MCP tool so agents can keyword-search too. Thin
     adapter; roundtrip test added.
  2. **`create_channel` MCP tool** (Q#4) — 5th tool; agents can now set up their own space
     (channel → thread → post → read) without any REST seeding. Discovery test asserts all 5
     names; full-flow roundtrip added.
  4. **Flat `GET /messages/{msg_id}` route** — replaced the nested
     `GET /threads/{tid}/messages/{mid}`, which ignored `tid` and let a message resolve under any
     thread's URL (a false contract). `Message.msgId` is workspace-unique and `Message` has no
     `threadId`, so resolution is workspace-global by design; the flat route states that truth.
- **Two fork decisions (spec §0):**
  - **Fork 3(a) — dead `isMention` highlight:** *remove it from the JS* rather than make §4 return
    a per-reader `isMention`. `isMention` is a since-read (§9) concept computed only by
    `read_thread_since`/`read_ws_since` (which take `me_id`); the reader-agnostic §4 thread read
    the web UI uses never sends it, so the highlight was dead-falsy. Making §4 reader-aware would
    mutate the locked §4 query, add a per-reader traversal to the hot thread-read path (RAM rule
    6), and force a 92-suite assertion change — not worth restoring a cosmetic highlight on a
    request/response M1 UI. Revisit in M2 with real-time since-reads.
  - **Fork 4 — nested single-message route:** *drop the thread-scoped spelling* for a flat
    `GET /messages/{mid}`. Validating thread membership would need an O(thread-length) HEAD/NEXT
    traversal on a route the web UI does not use, purely to keep a URL shape; the O(1) fix
    (denormalised `Message.threadId`) is a parked schema change (RAM rule 6). Leaving it as-is
    ships a wrong-thread-resolution trap.
- **Verified:** server suite **70 passed** (was 68; +1 search roundtrip, +1 create_channel flow;
  discovery + 2 api tests edited net 0); query suite **92/92** (untouched — regression guard).
- **Docs (same change):** `DESIGN.md` §15.2 tools table (+2 rows), §14.4 REST surface
  (`/messages/{mid}`), §14 test-count 68→70; `README.md` MCP tools list (+`create_channel`,
  +`search_messages`) and counts 68→70; `BACKLOG.md` pruned (4 completed items removed, Last
  reviewed bumped); this entry.
- **Batch B (delivered separately by another implementer):** the two `web/app.js` items —
  removing the dead `isMention` class toggle in `renderMessages`, and making the composer submit
  handler retry a mention-rejected send (`400 UnknownMemberError`) as plain text with a
  non-blocking notice so a typo'd `@handle` no longer drops the whole message. No test harness for
  the web JS; verified manually.

## 2026-07-02 — K-004: M1 hardening — five live-verified defects + QA DEF-1 fixed

- **What:** a full-project review probed the M1 server live (isolated `ws:probe` graph) and
  confirmed five defects the 57-test suite missed — every failing scenario involved state the
  fixtures always seeded (the actor) or parameter combinations never tested (`limit` + cursor).
  All fixed TDD (11 red tests → green):
  1. **Silent no-op writes (worst).** The §4 write queries anchor on `MATCH (author {userId:…})`;
     with the author node absent the whole write no-ops and REST still returned **201 with a fresh
     `msgId`** — on a fresh tenant (nothing ensures `u1`) every send "succeeded" and every thread
     stayed empty. Fix at three layers: `repository._assert_written` raises on zero-row writes;
     `services.post_message` validates the actor resolves to a member (`UnknownActorError`, one
     shared membership lookup with mentions); `create_app`'s lifespan runs `services.ensure_actor()`
     (startup, not import — building the app still needs no live FalkorDB).
  2. **Cursor-vs-limit message loss.** `read_messages` advanced the cursor to the *server clock*,
     permanently skipping rows a `limit` truncated (probe: 5 posted, `limit=2` read → next read 0).
     Fix: since-reads (§9.1/§9.2) are now **chronological** — the truncated page is a contiguous
     prefix — with reader-mentions carried by the `isMention` flag instead of the old
     mention-first sort (which + `LIMIT` is what made pagination lossy); the cursor advances to the
     newest **delivered** `createdAt` (empty page → no write). Ordering change synced in
     `QUERIES.md` §9 (+ rationale note), `test_queries.sh` (1:1 assertion swap), DESIGN §15.2.
  3. **`advance_cursor` IndexError** when the member node didn't exist (empty result indexed) —
     now a no-op returning `None`; noted in QUERIES.md §9.3.
  4. **QA DEF-1 (from the 2026-07-01 report) closed.** `POST /mcp` 405'd (Starlette Mount serves
     only `/mcp/`) — `create_app` adds an ASGI path-alias middleware rewriting `/mcp` → `/mcp/`;
     regression pinned by tightening the existing app test (it had tolerated 405 via `< 500`).
  5. **Search syntax-error 500.** RediSearch parse errors (`q='hello"x'`) surfaced as unhandled
     500s — `services.search_messages` maps `ResponseError` → `InvalidSearchQueryError` → 400.
  - Also: removed a duplicated gotcha comment in `repository.thread_has_head`; fixed the stale
    `exists((t)-[:HEAD]->())` advice in QUERIES.md §4 (contradicted the AGENTS.md live gotcha).
- **Verified:** server suite **68 passed** (was 57; +11); query suite **92/92** (assertion count
  unchanged — ordering assertions swapped 1:1); live probe script re-run: all five defects gone.
- **Docs (same change):** `QUERIES.md` §4 zero-rows + HEAD-check notes, §9 ordering rationale,
  §9.3 no-member note; `AGENTS.md` write-path invariants (+ zero-rows, chronological-cursor
  bullets) and test count; `README.md` counts + `/mcp` slash note; `DESIGN.md` §12/§15.
- **Plan items:** K-004 ✅. Review findings **not** fixed here parked in `BACKLOG.md` (agent
  authorship, `threadId` in §9.2 rows, retry idempotency + first-post race, web-UI mention
  polish, nested-route validation, ms-tie ordering, dependency pins, lint/CI).

## 2026-07-01 — QA: functional test pass on M1 (REST + MCP)

- **What:** first black-box/acceptance QA pass on the M1 server, driving the *running* process
  (curl over REST + a real `mcp` Streamable-HTTP client session) on top of the 57-test baseline.
  Added `docs/archive/test-plans/m1-chat-mcp.md` and `docs/archive/test-reports/m1-chat-mcp-report.md`.
- **Result:** 22/22 functional+contract items PASS · baseline 57/57. Verified both front doors over
  one service layer, error→status mapping (404/404/400), input validation (422), full-text search,
  read-cursor advance vs. explicit-`since` read-only, and REST↔MCP cross-door parity.
- **Defect found (DEF-1, low-med):** MCP endpoint 405s at `POST /mcp`; only `/mcp/` (trailing slash)
  completes the handshake — but README/DESIGN Appendix A advertise `/mcp`. Fix = alias/redirect
  `/mcp`→`/mcp/` **or** correct the docs, plus a regression test. See the report §3.
- **Feedback:** `bootstrap_schema.sh` seeds no members, so the mention happy-path needs manual seeding
  (consider a `seed_demo.sh`); per-endpoint response shapes vary (documented schema would make them
  testable); channel names non-unique. Details in the report §5.
- **Why:** first spin of the new `claude/qa-engineer` agent (proxy-run). No code under test changed.

## 2026-07-01 — K-003: M1 chat core finish — full-text search endpoint + web UI

- **What:** Closed out M1 chat core on top of the K-002 server, TDD and search-first.
  - **Full-text search (red→green per layer):** `repository.search_messages` (workspace-wide
    `db.idx.fulltext.queryNodes('Message', …)`, `QUERIES.md` §5 with the channel-scoping MATCH
    omitted) → `services.search_messages` (thin passthrough) → REST `GET /search?q=&limit=`
    (`q` required via `Query(..., min_length=1)`; `limit` bounded 1–200). **+5 tests** (2 live repo,
    1 fake-repo service, 2 TestClient incl. the `422` missing-`q` guard).
  - **Web UI:** minimal `web/{index.html, app.js}` — vanilla `fetch` over the same-origin REST API:
    channels list/create, threads list/create, thread messages + composer (parses `@id` handles into
    `mentions[]`), and a full-text search panel. HTML-escaped throughout.
  - **Serving:** `app.py` gained a `web_dir` param and mounts `StaticFiles(html=True)` at `/`
    **last** — `/` is a catch-all that must sit behind the REST routes and the `/mcp` mount
    (Starlette matches in registration order). Same-origin ⇒ no CORS. Mount is skipped if `web/` is
    absent. **+1 test** pinning "serves index at `/` **and** `/channels` still returns JSON."
- **Verified:** full server suite **57 passed** (was 51); query suite regression **92/92**. Smoke:
  assembled app serves the real `web/index.html` at `/`, `web/app.js` as `text/javascript`, and
  `/channels` JSON alongside — one process, three front doors (web, REST, MCP).
- **Docs (same change):** `DESIGN.md` §12 roadmap + §14.5 layout/serving note + §14.6 build order
  (steps 3–4 ✅); `README.md` roadmap/layout/run + "open http://localhost:8000/"; `AGENTS.md` server
  surface (static-mount-last rule, `/search`) and test count 51→57.
- **Plan items:** K-003 ✅ → **M1 chat core code-complete.** Parking lot now: `search` over MCP,
  `create_channel` over MCP (Q#4).

## 2026-07-01 — K-002 Step 2: M1 server (repository → services → MCP + REST), one process

- **What:** Built the first application code for the component (greenfield `server/` tree), bottom-up
  and test-first, completing K-002 (`docs/archive/plans/m1-chat-mcp.md`). All against live FalkorDB.
  - **`repository.py`** — every method 1:1 with a verified `QUERIES.md` query: channels/threads (§3),
    `ensure_user`/`ensure_agent` (§2/§7), both message write paths with the atomic `MENTIONS_MEMBER`
    block (§4), `read_thread` (§4), `read_thread_since` (§9.1), `read_ws_since` (§9.2),
    `advance_cursor`/`get_cursor` (§9.3/9.4), `get_message` (§4), plus validation reads
    (`thread_exists`/`channel_exists`/`existing_members`/`thread_has_head`).
  - **`services.py`** — invariants: id/clock generation (server clock), first-vs-subsequent write
    dispatch, mention validation (`UnknownMemberError`), RO/RW `read_messages` dispatch + `cursorId`
    construction, `Channel`/`ThreadNotFoundError`.
  - **`mcp.py`** — FastMCP adapter; tools `send_message`/`read_messages`/`create_thread`, injectable
    service + context (Q#1: `frm` ignored, actor = `get_context()`).
  - **`api.py` + `schemas.py`** — REST surface (DESIGN §14.4) incl. optional `mentions[]` parity;
    `ServiceError` → 404/400.
  - **`app.py`** — `create_app()` mounts REST + MCP on one FastAPI process.
- **Live gotchas found & mitigated (now in AGENTS.md):** (a) `exists((t)-[:HEAD]->())` returns `true`
  with no edge on this build and `count{}` is unsupported → existence via `OPTIONAL MATCH … IS NOT
  NULL`; (b) MCP lifespan wiring (python-sdk #1367) — forward `mcp_app.router.lifespan_context` to
  `FastAPI(lifespan=…)` or the session manager never starts; set `streamable_http_path="/"` so the
  mount lands cleanly at `/mcp`; (c) `call_tool` returns `(content, structured)` with list results
  wrapped as `{"result": […]}`.
- **Env:** no `uv` on the box → `server/.venv` via `python3 -m venv`; deps fastapi/uvicorn/falkordb
  1.6.1/mcp 1.28.1/pytest/httpx.
- **Tests:** **51 passed** — repository (24 live), services (12 unit fake-repo + 2 live), MCP (4
  in-memory), REST (7 TestClient), app-mount/lifespan (2). Query suite regression **92/92**.
- **Verified end-to-end:** REST round-trip through the assembled app; MCP tool discovery lists the
  three tools; mention-prioritised reads; monotonic cursor advance.
- **Plan items:** K-002 Step 2 ✅ → **K-002 complete.** Deferred: web UI (M1), `create_channel` over
  MCP (Q#4), full-text `search` REST endpoint.

## 2026-07-01 — K-002 Step 1 (gate): schema + queries for mentions & read-cursors

- **What:** Landed the graph-dba gate for the M1 Chat MCP transport (`docs/archive/plans/m1-chat-mcp.md`),
  all live-verified against `falkordb/falkordb:edge`. (1) `bootstrap_schema.sh`: added
  `ReadCursor.cursorId` range index + uniqueness constraint (index-before-constraint). (2)
  `QUERIES.md` §4: both message write paths now carry a `$mentions` list and append a
  `MENTIONS_MEMBER` write-block, atomically inside the single write query. (3) `QUERIES.md` new §9:
  `read_messages` since-reads — §9.1 thread-scoped, §9.2 workspace-wide, §9.3 monotonic cursor
  advance, §9.4 cursor read. (4) `test_queries.sh`: +25 assertions.
- **Q#2 resolved (member-match index strategy).** `GRAPH.PROFILE` showed `WHERE n.userId=$x OR
  n.agentId=$x` as a scan anchor degrading to an `All Node Scan`; the write path instead resolves
  each mention with dual `OPTIONAL MATCH (u:User)/(a:Agent)` + `coalesce` → two `Node By Index
  Scan`s. The `OR` form is kept only where `me`/`mem` is already bound (mention-flag, cursor read).
- **Two live gotchas found & mitigated (now in AGENTS.md):** (a) a bare empty `UNWIND` collapses the
  row stream, so `RETURN m` came back empty on a `$mentions=[]` post despite the writes committing —
  guarded with `UNWIND (CASE WHEN $mentions=[] THEN [null] ELSE $mentions END)` + a non-filtering
  `FOREACH`; (b) `collect(DISTINCT coalesce(u,a))` gives free dedup + unknown-skip and collapses the
  per-mention rows back to a single result row. Both proven: `$mentions=[]` is byte-identical to a
  plain post; `['u3','u3','a7','nope']` → 2 edges `[u3,a7]`, one row.
- **Corrections vs. the plan's candidate Cypher:** mention-flag match handles **Agent** readers
  (`me.userId=$meId OR me.agentId=$meId`, not `me {userId:…}`); author id returned via
  `coalesce(author.userId, author.agentId)` so Agent authors aren't null. §9.3 monotonic guard
  (`CASE WHEN $now > coalesce(rc.lastReadAt,0) …`) verified on this build (300 → stale 200 stays
  300 → 400).
- **RAM (rule #6):** +1 range index and +1 constraint per workspace; growth term is one `ReadCursor`
  node per *(member, thread)* read and one `MENTIONS_MEMBER` edge per mention. No new vector
  dimension → no embedding-RAM change.
- **Tests:** suite green at **new baseline 67/67 → 92/92** (+25: mention write-path incl.
  empty/dedup/unknown, §9.1 prioritised since-read + exclusion, §9.2 index-scan proof, §9.3
  monotonic/idempotent cursor + constraint block, §9.4 read + index-scan proof).
- **Plan items:** K-002 Step 1 ✅ (gate passed); Step 2 (repository → services → `mcp.py`/`app.py`
  → REST parity) unblocked.

## 2026-06-11 — K-001: `list_channels` query (list channels in a workspace)

- **What:** Authored and live-verified a `list channels` query and added it to `docs/QUERIES.md`
  §3 (Channels & threads), with assertions in `scripts/test_queries.sh`. Query:
  `MATCH (c:Channel) WHERE c.channelId > '' RETURN c.channelId, c.name, c.createdAt ORDER BY
  c.createdAt DESC LIMIT $limit`. The always-true `c.channelId > ''` predicate (every `channelId`
  is a non-empty string) anchors the listing on the **`Channel.channelId` range index** —
  `GRAPH.PROFILE` confirms `Node By Index Scan`, not `NodeByLabelScan`. Ordered by `createdAt`
  (channel **creation** time, newest-first), which is free once the scan is index-backed. Marked
  `GET /channels → list_channels` resolved in `DESIGN.md` §14.4 (was "gap — owned by graph-dba")
  and flipped the §14.6 prerequisite step to done.
- **Why:** the M1 REST surface (`GET /channels`, DESIGN §14.4) needed a verified query and
  `QUERIES.md` had none — it covered channel *members* (§2) and recent *threads* (§3) but not
  channels. Unblocks the `list_channels` repository method (§14.6 build order).
- **Trade-off noted:** true activity-recency (most-recent message/thread per channel) would need a
  `HAS_THREAD` → `Thread.updatedAt` expansion per channel — the Channel-level edge traversal §5.2
  deliberately avoids — so the cheap, index-backed **creation-time** ordering is used instead, and
  this is documented inline in `QUERIES.md`. No new index or constraint added; **zero per-workspace
  RAM cost** (reuses the existing `Channel.channelId` index).
- **Tests:** suite green at the **new baseline 64/64 → 67/67** (one §3 functional assertion +
  the standard §8 `assert_index_scan` pair; the plan's "65/65" estimate predated counting each
  `assert_*` call — the PROFILE proof is a two-line assertion per the existing §8 convention).
- **Plan items:** K-001 ✅ done.

## 2026-06-11 — Defined the M1 client/server application architecture

- **What:** Pinned the M1 application architecture and documented it as a new `docs/DESIGN.md` §14.
  Decisions: **transport = REST/JSON over FastAPI** (chosen after explicitly re-evaluating and
  rejecting gRPC — the only M1 client is a browser, so gRPC's typed-contract/streaming/service-mesh
  wins go unused and gRPC-Web would be pure bridge tax; WebSocket/SSE is the stronger M2 real-time
  path); **client = minimal web UI**; **real-time deferred to M2**; **single hardcoded tenant**
  (`ws=acme`, `user=u1`) injected at one FastAPI dependency seam so real auth drops in later without
  touching services/repo. Captured the layering (router → service → repository → db → FalkorDB),
  the REST surface → service → `QUERIES.md` mapping, proposed `server/` + `web/` layout, and the
  bottom-up TDD build order. Updated the §12 + README roadmap rows to point at §14.
- **Why:** User wanted the M1 client/server architecture nailed down before any code; the DESIGN
  doc previously only sketched the *operational* topology (§10), not the application code shape.
- **Plan items:** seeded **K-001** (`list_channels` query gap, owned by graph-dba — the one piece of
  the M1 REST surface with no verified query yet).

## 2026-06-11 — Adopted the kaizen plan/history convention

- **What:** Created `kaizen/{plan.md,history.md}` for the component, mirroring the sibling
  `claude/<agent>/kaizen/` projects (forward-looking backlog + dated change log). Replaced a
  short-lived `BACKLOG.md` draft with this structure.
- **Why:** User asked the component to track work the same way the sibling projects do, rather than
  a standalone backlog file.
- **Plan items:** K-001 recorded as the first active item.

## (prior) — M0 baseline

- M0 — Engine up: FalkorDB running (`falkordb/falkordb:edge`, Redis 8.2.2, module `999999`),
  design locked, schema bootstrap + canonical query library live-verified (`test_queries.sh`,
  64/64). Predates this log; see git history (e.g. `feat(falkor-chat): schema, query library,
  tests, and working context`) and `docs/DESIGN.md`.
