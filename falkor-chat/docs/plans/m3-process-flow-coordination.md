# K-024 (remaining half) — `kind:'process'` proof flow · teco coordination ledger

> Companion to the architect plan `m3-process-flow.md`. This is the **coordination** record:
> units, owners, gates, decisions, status. The design lives in the plan; the *why* of the
> workflow model lives in `../DESIGN.md` §6/§6.3.

**Goal:** close the open half of **K-024** — the LLM-free `kind:'process'` proof flow over
`human` / `decision` / `wait` steps (the DESIGN §6.3 "coordination is workflow" proof). This is
the last **build** item before **K-025** (QA acceptance) can run and take **M3 → ✅**.

**Explicitly out of scope:** **K-027** (live-triage reliability + carried gate minors m-1…m-3,
nits n-1…n-3). It is a parallel track, not an M3-green gate (decision D12-B). Do not fold its
items into any unit here.

## Entry state (verified by teco, 2026-07-19)

- Working tree clean; K-022 Landing 2 committed at `4f69a16`.
- Baselines: server pytest **350 passed / 0 skipped / 1 deselected**; query suite **241/241**.
- `reference` + `ws:acme` split-brain repaired and independently verified (see
  `m3-executor-coordination.md`, D15).

## Gaps this slice must close (teco investigation, pre-design)

1. **`executor._execute_step` dispatches only `type:'agent'`.** Every other type falls through
   to a silent no-op `StepResult(output="", on="done")` — *weaker* than plan `m3-executor.md`
   §2.3's stated intent (a documented `NotImplementedError`). A `decision` node today
   "succeeds" doing nothing.
2. **No deterministic guard language.** `guards.evaluate_guard` resolves only the empty/null
   guard (unconditional) and `{"kind":"llm"}`; every other `kind` hits the M7
   `NotImplementedError` seam. A `decision` node has nothing to branch with.
3. **No human-input channel for a non-chat process.** The existing `waitsForHuman`
   suspend/resume mechanic (§2.4) is chat-message-driven via `trigger.py`.

## Standing constraints carried into every brief

- **Deterministic + offline.** No `live` marker, no LM Studio. The network-free baseline must
  stay green and rise.
- **`_drive_loop` is locked** — verify by **SHA `71055f756280` only**; every byte count quoted
  for it in the docs is wrong (miscopied three different ways).
- **Published defs are create-only and split-brain-prone.** New def = new `key`/`version`,
  additive; never an edit of `triage@v1`. `pytest` and `test_queries.sh` both wipe `reference`
  but not `ws:<id>` ⇒ order is **pytest → re-seed → verify**, never the reverse.
- **Graph surface:** any new DDL / index / QUERIES entry requires a **graph-dba** gate before
  implementation, and its RAM impact must be called out (AGENTS.md rule 6).

## Documentation-impact scan (teco, folded into unit done-conditions)

| Doc | Expected impact |
|---|---|
| `docs/DESIGN.md` | §6 step-type semantics; §6.3 (the proof now exists); §13 if the deterministic-guard decision closes part of the open question |
| `docs/QUERIES.md` | Only if queries change — expected **none**; confirm at integration |
| `falkor-chat/AGENTS.md` | Key-scripts table (any new/changed seed script); new executor invariants |
| `docs/BACKLOG.md` | K-024 → ✅; K-025 unblocked; milestone map row |
| `docs/HISTORY.md` | Entry on delivery (repo convention: every delivered change) |
| `server/.env.example`, `scripts/start_server.sh` | Only if a config var is added |

## Units

Per `m3-process-flow.md` §6. Every unit's done-condition includes: suites green (server pytest
**up** from 350, `test_queries.sh` at its pinned count), the **`_drive_loop` SHA re-verified as
`71055f756280`**, and its named docs updated *in the same change*.

| U | Unit | Owner | Depends on |
|---|---|---|---|
| U0 | graph-dba gate — `start_run_untriggered` (§12.1b) + `set_run_ctx` (§12.12), queries only | `graph-dba` | — (blocks U3) |
| U1 | deterministic `cmp` guard (`guards.py` only) | `tdd-engineer` | — (parallel with U0) |
| U2 | typed step handlers (`human`/`decision`/`wait`) + publish invariant | `coder` | U1 (soft) |
| U3 | start-without-trigger + input endpoint | `coder` | U0 |
| U4 | the `onboarding@v1` def, seed & offline acceptance | `coder` | U1, U2, U3 |
| U5 | closeout (BACKLOG/HISTORY/DESIGN §6.3, file K-028) | `teco` | all |

## Decisions

| # | Decision | Status |
|---|---|---|
| D-A | Deterministic guard language | ✅ **settled (user, 2026-07-19): structured `cmp` comparator** — `{"kind":"cmp","path":…,"op":…,"value":…}` + `all`/`any`/`not`. Whitelisted ops, two whitelisted path roots (`ctx.`/`output.`), depth/width caps, no parser, no `eval`, no dependency. Named `cmp` deliberately so `kind:'expr'` keeps raising and DESIGN §13's "no expression library was built" stays literally true. |
| D-B | Human-input channel | ✅ **settled (user): REST** — `POST /workflow-runs` (start without a chat `Message`, per F-2) + `POST /workflow-runs/{id}/input` (body merges flat into run `ctx`, then the existing `resume_run` CAS drives). MCP `submit_workflow_input` is a non-blocking follow-up, not this slice. |
| D-C | `wait` semantics | ✅ **settled (user): external signal, not a timer.** No scheduler exists (BackgroundTasks are request-scoped). Must be stated **plainly** in DESIGN §6.1 rather than implied by the step name; real timers filed as new backlog item **K-028**. |
| D-D | The proof def | ⏳ taken as recommended — **`onboarding@v1`** (`kind:'process'`, brand-new key/version, `triage@v1` untouched): 6 steps / 7 transitions, `human`×2, `decision`×3, `wait`×1; ops `exists`/`in`/`eq`/`truthy`; conditional-beats-unconditional ordering; two terminal outcomes; 8 of the 12-step budget. Confirm against the analyst gate. |
| D-E | Scope boundary | ⏳ taken as recommended — implement `human`/`decision`/`wait`; `prompt`/`tool`/`message` become an explicit `NotImplementedError` seam (a **behaviour change** from today's silent no-op — R-3). Confirm against the analyst gate. |

## Gate plan

`architect` (design) → **`analyst` plan review** → user decision on D-A…D-E → implementer
(`coder` or `tdd-engineer`, per unit shape) → **`analyst` implementation review** → integration
(teco: suites + doc verification) → hands off to **K-025** (`qa-engineer`).

The review gate is non-negotiable — it was the pattern that caught both K-022 landings' majors.

## Status log

- **2026-07-19 · teco** — Entry state verified, gaps 1–3 established from source, ledger opened.
  `architect` briefed to design the slice and **surface** D-A…D-E as options rather than settle
  them. Deliverable: `docs/plans/m3-process-flow.md`.
- **2026-07-19 · architect** — Plan delivered (`m3-process-flow.md`, 608 lines). Central design
  choice: **park-and-branch, so `_drive_loop` is never touched** — a `human` step is just a step
  whose outgoing guard reads a `ctx` key that does not exist yet (existing OUTCOME B parks it);
  writing the key makes the same guard fire on `resume()`. Only two new capabilities are needed:
  *read `ctx` in a guard* (U1) and *write `ctx` from outside* (U3). graph-dba gate is **required
  but small** — two additive queries, no new label, no new index, no `bootstrap_schema.sh` change,
  RAM ≈ nil. Findings raised: **F-1** `TRANSITION.on`/`StepResult.on` are vestigial and DESIGN §6.1
  describes them inaccurately; **F-2** a run cannot start without a chat `Message` today
  (`repository.py:1067`); **F-3** the `agent`-without-LLM silent stub is load-bearing for the
  350-test baseline and must be preserved verbatim.
- **2026-07-19 · teco** — `analyst` dispatched for the **plan gate** (deliverable
  `docs/reviews/m3-process-flow.md`), briefed to verify the park-and-branch claim against the real
  loop, the `cmp` guard's totality/injection-safety, the two queries' additivity, the ctx-merge
  concurrency posture, the F-1/F-2/F-3 findings independently, and whether D-A…D-E are well-posed.
- **2026-07-19 · user** — **D-A, D-B, D-C settled as recommended** (see the Decisions table).
  D-D/D-E taken as recommended pending the gate. Implementation is **held until the analyst gate
  returns** — no implementer dispatched yet.
- **2026-07-19 · analyst (plan gate)** — **REQUEST CHANGES**: 2 blocker · 6 major · 10 minor ·
  5 nit → `docs/reviews/m3-process-flow.md`. The central mechanism is **verified sound** (see
  below); both blockers are the same class of error — the plan under-measured its blast radius on
  the **existing test estate** — and both are cheap plan-level fixes, not design rethinks.
- **2026-07-19 · teco** — Day-end stop at the user's request. **No implementer dispatched; U0–U5
  not started.** Next action is a plan patch (below), not code.

- **2026-07-20 · teco** — Three coordinator OQs **settled without a user round-trip** (all three
  were execution mechanics inside already-settled decisions, not new product choices):
  **OQ-A → fold** the ctx write into the resume CAS (analyst M-1; a mechanic refinement inside
  settled D-B). **OQ-B → 200/201 + `{"status":"failed",…}`** — the run *is* correctly terminal in
  the graph, so a 500 traceback would misreport it and break the very audit property §6.3 exists
  to prove. **OQ-C → analyst option (b) + (c)** — validate submitted input against the parked
  step *before* merge+resume, so a value that can fire no guard is a free 400 costing no budget,
  **and** state the def's budget explicitly.
- **2026-07-20 · architect (plan patch v2)** — Written in as decisions **D-F/D-G/D-H**. B-1, B-2,
  M-1…M-6 closed in text. `set_run_ctx` **dropped from the design entirely** (D-F replaces it with
  `resume_run_with_ctx`, QUERIES §12.13 — U0's deliverable changed, not grown). D-G specifies five
  handlers incl. a new `WorkflowEngineDisabledError` → 503, folding the plan's own OQ-1. D-H
  validates in `services.submit_workflow_input` (service layer, so MCP inherits it) against the
  parked step's declarations, via two **existing** RO reads — no new query. Def renamed
  **`onboarding@v1` → `access-request@v1`** (m-2: `onboarding` collides with a long-standing test
  fixture key), `maxSteps` raised to **24** with the mistakes arithmetic written out. m-1's recount
  fixed (6 transitions / 4 ops). **Declined with reasons:** m-9 (triage's def literal stays inline
  — moving a live published def during a split-brain-prone slice risks a byte-diff that
  `MERGE … ON CREATE SET` silently swallows; filed as proposed **K-029**), n-3 (symmetric
  `decision` invariant would retro-reject `test_services.py:719`; kept as an AGENTS.md warning),
  and the sentinel-`waitingThreadId` belt for M-2.
- **2026-07-20 · analyst (re-gate v2, diff-scoped)** — **APPROVE WITH SUGGESTIONS. U0/U1
  unblocked.** All 2 blockers + 6 majors verified **closed in the referenced text**, not merely
  claimed in the response table. B-2's fixture list confirmed *exhaustive* (the three unlisted
  `human`/`wait` fixtures provably never reach `_validate_def_spec`). D-F leaves no `set_run_ctx`
  residue; D-G's five handler premises each check out; D-H's "no new query" is real (`get_run`
  returns `atStepKey`/`defKey`/`defVersion`, `get_snapshot`'s steps carry `config`, and
  `suspend_run` does **not** clear `AT_STEP`, so the parked step is resolvable). Rename consistent
  everywhere; `fields:["request"]` matches §4.3's traced path. SHA re-confirmed `71055f756280`,
  and all three `executor.py` edit sites sit **outside** the lock (`resume` :287 before the loop;
  `_execute_step` :396, `_select_transition` :613, `_trace_step` :684 after the `# ── seams`
  marker :394). Four new findings, **all carried into implementation, none blocking**: **M-7**
  (U2 — the two publish invariants assume dict-shaped `config`/`guard`, but the REST path types
  them `str`; naive dict access → 500, a bare `isinstance` guard → every REST-published def
  silently escapes both invariants), **m-11** (U3 — `StepBudgetExceededError` doesn't exist),
  **m-12** (U3 — re-raise if the post-fault re-read isn't terminal, else a zombie `running` run
  reports as 200 success), **n-6** (drifted line citations).
- **2026-07-20 · teco** — **U0 (`graph-dba`) ∥ U1 (`tdd-engineer`) dispatched.** M-7/m-11/m-12/n-6
  sent back to `architect` in the same window (they land in U2/U3, neither of which is started, so
  they cost nothing on the critical path). **Shared-state protocol for this window:** `graph-dba`
  owns the live FalkorDB and the `reference` graph; `tdd-engineer` is restricted to
  `tests/test_guards.py` and explicitly barred from `test_queries.sh` and the full pytest run
  (whose `conftest` wipes `reference`). teco runs the integrated baseline once both land.

- **2026-07-20 · architect (plan patch v2.1)** — Re-gate findings closed. **M-7**: a shape
  normalization box opens §3.3 — `_validate_def_spec` must `json.loads` string-shaped
  `config`/`guard` first; a **non-dict guard is treated as no declaration** (so `{"expr":"x>0"}`
  and raw strings keep publishing) while a `human`/`wait` step whose config doesn't normalize to a
  dict is a `WorkflowDefSpecError`. Both failure directions are named in-plan. `validate_cmp` keeps
  its parsed-dict signature (stated twice) ⇒ **U1 in flight was unaffected**. §7 U2 gains a
  four-way shape matrix pinning that the REST front door cannot escape either invariant. **m-11**:
  `StepBudgetExceededError` dropped (budget exhaustion returns `"failed"`, never raises) with an
  explicit "U3 must not invent it" clause. **m-12**: D-G now requires the post-fault `get_run`
  re-read to **re-raise (500) unless status ∈ {failed, done, waiting}**, so a zombie `running` run
  can never be dressed as a success envelope. **n-6** citations corrected, with a "locate by test
  name, not line" instruction.
- **2026-07-20 · graph-dba (U0) — DONE, green.** QUERIES §12.12 `start_run_untriggered` + §12.13
  `resume_run_with_ctx`; **`set_run_ctx` appears nowhere** (D-F carried through). `test_queries.sh`
  **241 → 256** (+15), run twice green, `reference` re-seeded and `triage@v1` verified both times;
  DESIGN §7.1 pinned count updated. **PROFILEs pass and beat the plan's assumption**:
  `start_run_untriggered` is a single `Node By Index Scan` on `WorkflowDefSnapshot.key` and is
  *strictly cheaper* than §12.1 (dropping the `Message` anchor removes a scan branch);
  `resume_run_with_ctx` has **no residual `Filter`** — `status='waiting'` folds into the index
  scan (there is a RANGE index on `WorkflowRun.status`) — and was proven to remain a `runId` point
  lookup by seeding five other `waiting` runs (scan produced exactly 1 record). **Zero-row
  contracts verified, not assumed**: a snapshot with no `START` creates nothing (`Nodes created`
  absent); the CAS loser wrote **neither the status flip nor the ctx** (replayed with a marker ctx;
  the winner's value survived) — this is the empirical close of the M-1 silent-wrong-branch
  hazard. §5's no-DDL claim **confirmed**, `bootstrap_schema.sh` untouched, RAM ≈ nil. New
  engine quirk logged to `claude/graph-dba/falkordb-quirks.md`: `RETURN "x="+toString(count(m))`
  returns **zero rows with no error** (the implicit grouping key is the whole concatenated
  expression, which contains the aggregate) — an invisible failure mode.
- **2026-07-20 · tdd-engineer (U1) — DONE, green.** `cmp` + `all`/`any`/`not`, `validate_cmp`,
  `render_label`, populated `rationale`, whitelisted `_OPS`/`PATH_ROOTS`, caps 5/32/8. No parser,
  no `eval`, no new dependency; `kind:'expr'` still raises, pinned twice. `test_guards.py`
  33 → **143**. Six red→green cycles; two went green-on-arrival and were **mutation-tested rather
  than trusted** — one mutation (deleting the cmp-family check) survived, so a failing test was
  added first and confirmed RED→GREEN. Scope held to `guards.py` + its tests.
- **2026-07-20 · teco (integration)** — Integrated baseline run by teco (neither unit was permitted
  to): server pytest **460 passed / 1 deselected** = exactly 350 + 110, so **no existing test was
  weakened**; `test_queries.sh` **256/256**; `reference` re-seeded post-pytest and verified
  (`triage@v1` created, `ws:acme` snapshot consistent — no split-brain); `_drive_loop` SHA
  re-verified **`71055f756280`** with `executor.py` untouched.

## Open items carried out of U0/U1 (must be resolved in U2/U4/U5, not dropped)

| # | Item | Raised by | Lands in |
|---|---|---|---|
| O-1 | **§3.2 contradicted itself on unwhitelisted path roots** — the *evaluation* rule said "treat as missing" (§7 case 10 ⇒ `False`), the *validator* rule said `WorkflowConfigError` (§7 case 17). Once `evaluate_guard` calls `validate_cmp` (M-4's "one implementation") they collide. Resolved in code as **strict at publish, total at drive** (one validator, a `check_paths` flag), both halves pinned by test. **Ratify in the plan** so U2 doesn't re-derive it differently. | U1 | U2 (plan note) |
| O-2 | **`in` with a non-list literal** is unspecified. Implemented as `False` at drive, with **no** validator rule (adding one would invent a rule the plan doesn't state). Decide whether publish should reject it. | U1 | U2 |
| O-3 | **Bare `ctx` as a path** was unspecified (§3.2 blesses bare `output`). Implemented as *not* a value — rejected by `validate_cmp`, missing at drive — since it is the whole run state and would make `exists` trivially true. Confirm. | U1 | U2 |
| O-4 | **DESIGN §6.1 / §13 edits** are in U1's plan clause but were excluded from its dispatched scope (and `DESIGN.md` was held by graph-dba at the time). Includes F-1's two-part correction, D-C's "`wait` is signal-driven, and mechanically identical to `human`" (m-7), and §13's `cmp`-not-`expr` amendment. | U1 | U5 |
| O-5 | **K-029** (proposed): consolidate `triage@v1`'s inline def literal in `seed_workflows.sh` into `proof_defs.py` — declined inside this slice as too risky during a split-brain-prone change. File it. | architect | U5 |

## Gate outcome (v1) — what had to change before U0/U1 started

**Blockers (both = the plan's F-3 blast-radius miss):**

- **B-1 · F-3 is incomplete.** `server/tests/test_executor.py:345` defines a
  `{"key":"end","type":"task"}` step — a **non-`agent`** type executed by the *real* drive loop in
  `test_hallucinated_mention_does_not_fail_the_run`. Plan §3.3's "unknown type ⇒
  `NotImplementedError`" makes `_drive` `fail_run` **and re-raise**, killing that **Defect-B
  regression pin**. F-3 names only the `agent`-without-LLM stub. Remedy: name both stubs, and edit
  that fixture to `type:"agent"` inside U2.
- **B-2 · the new publish invariant retro-invalidates existing def fixtures.** `test_api.py:418`
  and `test_services.py:719/808/822/836/851/866` declare `type:"human"` with no `waitsForHuman`.
  ~7 tests fail outright and — worse — **5 `pytest.raises(WorkflowDefSpecError)` tests go
  vacuous** (they assert the exception type only, and the new check fires *before* the condition
  under test). U2's "the existing 350 stay green" is unachievable as written.

**Majors:**

- **M-1 · fold `set_run_ctx` into the `resume_run` CAS — one query, not two.** Removes the window
  where writer B's ctx is read by writer A's in-flight drive: that is a **silent wrong branch**,
  not merely a lost input, and it shrinks R-1 to what it actually claims. Same gate cost.
  **This is the one materially better option D-B omitted, and it blocks U0.**
- **M-2 · the start body's `ctx?` has no reserved-key rule.** A caller-set `threadId` makes
  `suspend_run` denorm a real thread, and `trigger.py` step 2 then resumes the *process* run on any
  chat message posted there — F-5's latent bug, live and reversed.
- **M-3 · error map is wrong.** Synchronous drive + the M-1 fault net's `raise` ⇒ unimplemented
  types and malformed `cmp` guards surface as **HTTP 500**, contradicting U3's "503-free error
  map"; `WorkflowRunNotFoundError` has no handler at all, so §3.4's promised 404 does not exist.
- **M-4 · validate `cmp` guards at publish**, where `_validate_def_spec` already sees them — the
  plan's own "~6 lines, no runtime path" argument applies *more* strongly to guards than to steps.
- **M-5 · budget.** ~4 mistaken approvals exhaust the 12-step budget irreversibly (§4.3 already
  costs 8; parks bump `stepCount`).
- **M-6 · the `cmp` `guard_judgment` trace requires editing `_select_transition`, which no unit
  owns** — it is safely *outside* the SHA-locked region, and the plan should say so explicitly.

**Verified rather than assumed (analyst traced these against the real tree):** the §3.1
park-and-branch claim **holds** — every §4 def shape traced through the real `_drive_loop`; nothing
spins, parks wrongly, or falls through to `done` unintentionally. The §3.1 SHA command reproduces
`71055f756280` verbatim. Every file:line citation in plan §2 is accurate. §5's "no DDL" is correct
(the required indexes already exist in `bootstrap_schema.sh`). Baseline confirmed by collection:
**350 passed / 351 collected, 1 deselected**. No K-027 creep, no gold-plating.

**Not statically verifiable — deferred to U0 execution:** both new queries' PROFILE plans and
zero-row contracts, the `test_queries.sh` count, the real pytest delta, the RAM figure. The analyst
deliberately did **not** run `pytest` or `test_queries.sh` — both wipe the shared `reference` graph.

**Decision hygiene:** D-A, D-C, D-E are complete and honestly stated (D-C strongest — "this system
has no scheduler" is verifiable and C2's cost is named, not hidden). **D-B's option set covered
*who submits* but never the *write mechanic*** ⇒ M-1. **D-D contradicts §4 internally**: prose says
"seven transitions" / "three ops", §4 shows **six** and **four** — recount before U4.

## Resume anchor — next session starts here

Plan is **approved** (v2, analyst re-gate 2026-07-20). U0 ∥ U1 are **in flight**.

1. Integrate **U0** (`graph-dba`: `start_run_untriggered` + `resume_run_with_ctx` in QUERIES §12,
   PROFILE evidence, zero-row contracts, `test_queries.sh` at a new pinned count, `reference`
   re-seeded) and **U1** (`tdd-engineer`: `cmp`/`all`/`any`/`not` + `validate_cmp` in `guards.py`).
   Run the **integrated full baseline** yourself — neither unit was allowed to, by the shared-state
   protocol. Order is always **pytest → re-seed → verify**.
2. Then **U2** (`coder`) — typed handlers + publish invariant. Carries **M-7** (normalize
   string-shaped `config`/`guard` before validating; `validate_cmp` keeps taking a parsed dict, so
   U1 is unaffected) and the B-1/B-2 fixture edits. U2 also owns `_select_transition`/`_trace_step`
   (M-6) — **outside** the SHA lock; say so in the brief so the implementer doesn't freeze at the
   stop-and-escalate rule.
3. Then **U3** (`coder`) — start-without-trigger + input endpoint, per D-G/D-H. Carries **m-11**
   (drop the non-existent `StepBudgetExceededError`) and **m-12** (re-raise when the post-fault
   re-read isn't terminal).
4. **U4** (the `access-request@v1` def, seed, offline acceptance) → **U5** (closeout: BACKLOG,
   HISTORY, DESIGN §6.3, file **K-028** timers and **K-029** def-literal consolidation).
5. Hands off to **K-025** (`qa-engineer`) ⇒ **M3 → ✅**.

D-A…D-H are all settled. **No user decision is pending.**
