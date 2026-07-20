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

## Gate outcome — what must change before U0/U1 start

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

1. **`architect` plan patch** (not code): close B-1, B-2, M-1…M-6 in `m3-process-flow.md`; fold
   M-1's single-query resume+ctx write into D-B and §5 (it changes U0's brief); fix the D-D
   transition/op recount. Minors + nits from the review to be triaged in the same pass.
2. Re-gate the patch with `analyst` (a diff-scoped re-review, not a full re-read).
3. Then, and only then, dispatch **U0** (`graph-dba`) ∥ **U1** (`tdd-engineer`) per §6.

D-A/D-B/D-C stay settled — the patch refines the *mechanic* inside D-B's chosen option, not the
option itself. No user decision is pending.
