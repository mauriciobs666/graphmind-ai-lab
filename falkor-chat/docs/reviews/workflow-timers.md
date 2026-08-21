# Review — K-028 workflow timers plan (`docs/plans/workflow-timers.md`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-028

## 1. Scope & verdict

Pre-implementation gate review of `falkor-chat/docs/plans/workflow-timers.md` (read in full)
against the `### K-028` section of `falkor-chat/docs/BACKLOG.md`, `falkor-chat/AGENTS.md`
(rules 1–8, testing hazards), `falkor-chat/docs/DESIGN.md` §6/§14, and the actual code:
`server/falkorchat/repository.py`, `executor.py`, `services.py`, `guards.py`, `api.py`,
`app.py`, `config.py`, `background.py`, `scripts/bootstrap_schema.sh`, plus
`docs/QUERIES.md` section numbering and the `server/tests/` tree. Static review only; no
suites run (no code exists yet to gate on — the plan's own acceptance commands are test
descriptions, not runnable commands, so there was nothing to execute verbatim).

**Verdict: approve with suggestions.** No blockers. One major (the SHA-lock re-lock
ceremony's documentation scope is narrower than the repo's own K-033-documented convention),
four minors, and nits. The core design — `wakeAt` inside the suspend CAS, the three-guard
`resume_due_run`, the stateless lifespan ticker — is sound and exceptionally well grounded:
every file:line citation I checked is accurate on the current tree.

CPG: considered, not relevant — the teco brief states `cpg_falkorchat` is stale (built
2026-08-17, 6 commits behind the K-027 executor/guard work), so all claims were verified by
reading current source files instead.

### What I verified (evidence, per the brief's scrutiny points)

1. **The one-winner property is airtight.** All three resume queries — `resume_run`
   (`repository.py:1431-1437`), `resume_run_with_ctx` (`:1461-1467`, with the live-verified
   zero-row contract in its docstring), and the proposed `resume_due_run` — guard the same
   `r.status = 'waiting'` flip; FalkorDB per-query atomicity means exactly one observes
   `waiting`. The ctx-equality fence needs no byte-stability assumption: `$expectedCtx` is
   the string read from the graph and passed back verbatim (plan §3.4 "Returning `ctx` here
   supplies `$expectedCtx` without a second read"), never re-serialized. Ctx is written
   nowhere else during a drive (grep: only §12.1/§12.12 create, §12.13, `fail_run`), so the
   fence's data-loss-prevention claim holds. One narrow residual the plan overstates —
   finding m-1.
2. **`wakeAt = 0` sentinel + no-new-index.** Absent-property `r.wakeAt > 0` nulling out of
   the `WHERE` is correct Cypher semantics (grandfathers pre-K-028 parked runs). The sweep
   query is shape-identical to the PROFILE-verified §12.9 precedent
   (`find_waiting_run_for_thread`, `repository.py:1601-1606`: inline `{status: 'waiting'}`
   anchor + residual filter), and `LIMIT $limit` has a working precedent (`:1635`, §12.14).
   The plan correctly makes `GRAPH.PROFILE` a step-1 gate rather than asserting the plan
   shape (rule 3). The rule-6 RAM callout (one integer property, zero new node types/
   indexes/edges, `bootstrap_schema.sh` untouched) is honest — verified against the DDL
   (`bootstrap_schema.sh:122-156,196-197`: `runId` index+UNIQUE, `status`, `startedAt`).
3. **The SHA-lock reopen is described correctly.** The lock exists as claimed: DESIGN
   §6.2:438-445, value `71055f756280`, with the line-number-independent `awk` recompute
   command. OUTCOME B (`executor.py:492-500`) is inside the locked body; the handoff
   suspend (`:442-446`) is outside; `resume` (`:367`) is outside; a module-level
   `_wake_at_from` is outside. The single-argument diff + recompute + record-in-§6.2
   process is correct **but under-scoped on the documentation side** — finding M-1.
4. **Fifth invariant: ordering rule respected, blast radius zero.**
   `_validate_def_spec` (`services.py:884`) runs exactly four invariants deliberately
   LAST, in the order the plan states, and its docstring makes append-after-existing a
   stated constraint (`:896-906`) — appending fifth-and-last complies. Grep of `server/`
   and `scripts/` finds **zero** occurrences of `timeoutSeconds`/`wakeup`/`wakeAt`: no
   existing fixture, seeded def, or test retro-rejects (unlike K-029's symmetric-invariant
   problem). The step-3 ordering pin is the right test.
5. **`ctx.wakeup` mechanism checks out.** `RESERVED_CTX_KEYS` (`services.py:80`) is
   `{"threadId", "error"}` and `_reject_reserved_keys` (`:1522-1530`) gives the free 400 as
   claimed. Nested guard paths are supported today (`guards._resolve_path` splits on `.`,
   `_validate_path` whitelists `ctx.`/`output.` roots) — zero guard-engine changes, as
   claimed. Guard ordering `(guard == "", order, to)` (`executor.py:974-976`, first-firing-
   wins) confirms authoring convention 1's premise. R-T1's residual is real, narrow, and
   honestly mitigated; see m-5 for one un-named consequence.
6. **Ticker lifecycle and route posture.** Lifespan seam (`app.py:202`), `_build_default_app`
   (`:244`) with the `WORKFLOW_ENABLED` branch inside `ENABLE_AGENT` (`:264,:299`) — all as
   described. `background.py`'s log-and-swallow isolation posture exists (`:33,:57,:80`).
   Multi-worker/cron redundant sweeps resolving via the CAS is correct. `POST
   /workflow-runs/due` adds no capability the unauthenticated M1 surface doesn't already
   grant (`POST /workflow-runs` itself is unauthenticated, `api.py:253`); the no-`now`-
   override decision is a good catch. `GET /workflow-runs/{id}` is a pass-through of the
   repo dict (`api.py:275-283` → `services.get_workflow_run:1695-1699`), so `wakeAt`
   surfaces in the REST envelope from step 1 — see n-4. One shutdown-semantics gap: m-3.
7. **Test strategy** is offline, graph-state-asserting, mutation-testable; the XOR
   contention test plus deterministic splits is exactly the K-028-named test; §5.6 states
   the suite hazards (reference wipe, dim-4 index) accurately per AGENTS.md/DESIGN §14.7.
   One test as described is unbuildable at the stated altitude — m-2. `test_executor_process.py`,
   `test_services.py`, `test_process_input.py` all exist; QUERIES §12.16/§12.17 and DESIGN
   §6.4 are indeed the next free numbers.
8. **Step table sequencing** is green at every step: `wake_at: int = 0` default keeps
   existing callers green in step 1; no test pins an exact run-envelope key set (grep found
   only per-key asserts), so the additive `wakeAt` key breaks nothing; "wakeup" and
   `timeoutSeconds` are unused anywhere, so steps 3's changes are non-retroactive; steps
   4–5 depend only on earlier steps.

The stakeholder-pre-approved ticking mechanism (in-process asyncio ticker) survives
scrutiny — I found no concrete technical defect in it; the stateless-by-construction /
catch-up-on-first-tick / CAS-absorbs-redundancy arguments are all real properties of the
design as specified.

## 2. Findings

### Major

- **M-1 · The re-lock ceremony's documentation scope contradicts the convention the plan
  itself cites.** Plan §3.4/step 2 updates the recorded SHA **only in DESIGN §6.2**. But
  the open K-033 backlog entry (`docs/BACKLOG.md:860-890`) — which the plan cites as its
  precedent for "file a locked-loop change as its own reviewed item" — documents the full
  re-lock ceremony and warns explicitly: after any reopen *"the lock stops being a single
  grep-able constant… **Decide that framing before editing.**"* Two concrete gaps:
  (1) K-033's own still-open entry quotes `71055f756280` as the *current* lock and
  enumerates a ceremony list keyed to it — after K-028 lands first, that entry is stale
  (its implementer would grep for the wrong constant and follow a wrong site list; note
  the list also names `falkor-chat/AGENTS.md`, which no longer quotes the SHA at all —
  verified by grep — so the list is already partially stale and should be re-derived by
  grep at execution time, not trusted). (2) DESIGN §6.2:438 points at
  `docs/archive/plans/m3-process-flow.md` §3.1, whose text says the command *"must still
  print `71055f756280`"* — after the reopen, the live value and the archived instruction
  diverge, and nothing in the plan states the framing ("as of K-028 the lock is `<new>`;
  archived and dated-historical records quote the pre-K-028 value"). **Suggested fix:**
  add to step 6 — (a) a dated note on BACKLOG K-033's entry recording the new SHA;
  (b) the explicit framing sentence in DESIGN §6.2 beside the new value; (c) a
  `grep -rn 71055f756280` sweep with the disposition rule *dated/archived quotes stay,
  current-fact claims update* (on my grep, DESIGN §6.2 ×2 and BACKLOG's K-033 entry are
  the only current-fact sites; everything else is dated history). Cost: one paragraph in
  the plan; skipping it recreates the exact doc-drift class BACKLOG.md:586 was filed for.

### Minor

- **m-1 · §3.3's stale-merge fence claim is slightly broader than what holds.** The fence
  closes the race only when the intervening resume *wrote ctx* (§12.13 human path — and,
  neatly, a competing timer's own `wakeup` write). A **ctx-less** resume (§12.4
  `resume_run` — the chat/trigger path, e.g. K-014's thread-message resume of a parked
  intake run) that re-parks the run *already due again* inside the sweep's read→CAS window
  passes all three guards with byte-identical ctx, and the timer then merges a
  `wakeup.step` naming the **previous** park's step (read pre-race). No data loss is
  possible — the fence still guarantees `base == current ctx` at CAS time — but the marker
  can misdirect the timeout arm for one cycle (self-healing next tick, costs one
  `stepCount`). Suggested: name this residual in §3.3 or beside R-T1, and promote O-1's
  in-query `atStepKey` from "preference" to the prescribed choice, since it makes the
  ctx+step read atomic and shrinks the window to a single query.
- **m-2 · The §5.4 stale-merge fence test is unbuildable as described.** "Human-resume +
  drive to re-park (same step, short timeout, already due)" cannot be produced through the
  service layer with `escalation@v1` as specified: `expects: {decision: [approve,reject]}`
  plus the two data guards means every accepted submission fires a guard and advances —
  the step never re-parks on valid input, and invalid input is a free 400 that writes
  nothing. Suggested: specify the test at **repository level** (park via `suspend_run`,
  read ctx, then `resume_run_with_ctx` + `suspend_run(wake_at=<already due>)` directly,
  then `resume_due_run` with the old ctx → zero rows, ctx intact byte-for-byte), or give
  the fixture a declared non-branching field. Either preserves the pin's intent.
- **m-3 · Cancellation does not interrupt an in-flight sweep — say so.** `task.cancel()`
  interrupts the `await asyncio.to_thread(...)`, but the underlying thread (potentially
  mid-drive, including LLM I/O for an `agent` re-execution) runs to completion, and
  process shutdown can block on the default-executor join until it finishes; the drive
  also continues concurrently with lifespan teardown for that interval. Not a defect —
  there is no safe way to abort a mid-drive sweep, and the CAS already landed — but §3.2's
  "on lifespan exit, `task.cancel()` + await" reads as a clean stop. Suggested: one
  sentence in §3.2 and the DESIGN §14 process-model note; and the step-5 "lifespan cancel
  leaves no pending task" pin should be worded so it isn't read as "no in-flight sweep".
- **m-4 · A timed step with no wakeup-matching outgoing guard ends in `failed`, not
  documented as such.** If a def declares `timeoutSeconds` but its author forgets the
  escalation arm (or convention 2's step-key comparison never matches), each timeout
  resume re-parks the step with a fresh `wakeAt`, looping timeout→re-park until `maxSteps`
  fails the run — the SLA converts into an eventual `failed` terminal. §3.5's budget-
  interplay note covers the *re-parking timeout arm* case but not this authoring mistake,
  which is the likelier one. Suggested: name it in the §6.4 authoring conventions
  ("a declared timeout with no matching arm eventually fails the run at the budget"); a
  publish-time check (timed step must have some outgoing guard referencing `ctx.wakeup`)
  is possible but I don't insist — the documented tripwire is a defensible posture.

### Nits

- **n-1** · `_validate_def_spec`'s docstring says "Four further invariants run **last**"
  (`services.py:896`) — step 3 must bump it to five; the plan doesn't mention the
  docstring edit and this codebase treats docstrings as load-bearing documentation.
- **n-2** · The timer merge's `MAX_CONFIG_LEN` posture is unstated: `submit_workflow_input`
  bounds the merged ctx (`services.py:1489-1493`); the timer path writes repo-level with
  no bound, so a ctx sitting at the bound grows past it by the marker's size. State the
  decision (accept — the marker is ~80 bytes — or bound and skip-with-log).
- **n-3** · Say explicitly that `wake_due_runs` serializes its merged ctx via the existing
  `Services._dump_ctx` (`services.py:1532-1534`, canonical `sort_keys` form) so ctx
  serialization stays uniform across the two merge paths.
- **n-4** · Because `GET /workflow-runs/{id}` is a pass-through, the `wakeAt` envelope
  change lands in **step 1**, not step 5 — §3.6's "update any envelope pins" should be
  folded into step 1's done column (my grep found no exact-key-set pins, so likely
  nothing breaks, but the obligation belongs to the step that changes the shape).
- **n-5** · Step 6's DESIGN §6.2 edit should also add `resume_due` to the
  "outside the lock" function list (`_execute_step`, `_select_transition`, `_trace_step`,
  `resume` — DESIGN §6.2:439-441), since §3.3 places it there.

## 3. What's solid

- **Grounding is exemplary.** Every file:line citation checked resolves on the current
  tree (`suspend_run:1405`, `resume_run:1423`, `resume_run_with_ctx:1440`, `resume:367`,
  OUTCOME B `:492-500`, handoff `:442-446`, `RESERVED_CTX_KEYS:80`,
  `_validate_def_spec:884`, `_drive_or_fault:1598`, lifespan `:202`,
  `_build_default_app:244`, guards' nested-path support, the `''`-sentinel convention,
  §12.9's index precedent, the DDL). The plan was clearly read from source, as it says.
- **The atomic park+deadline decision** (`wakeAt` inside the §12.3 CAS) is the strongest
  design choice here — it makes the "scheduler as second source of truth" failure mode
  structurally unrepresentable, and the every-park-writes-`wakeAt` rule elegantly removes
  any need for clearing terms in §12.4/§12.13.
- **The rejected-alternatives analysis** (timer node, cron-only, Redis keyspace
  notifications) is genuine engineering, not strawmen — the fire-and-forget
  expired-events point against the Redis option is accurate and decisive.
- **The test strategy** asserts graph state, names the three independent zero-row causes,
  pins the validator ordering, and includes both racing and deterministic-split
  contention tests — mutation-testable as required by the standing practice.
- Backward compatibility (no migration, null-filter grandfathering) and the honest
  R-T1/R-T2/R-T3 residual register.

## 4. Open questions

None requiring stakeholder input before implementation. The M-1 ceremony paragraph and the
m-findings are architect-actionable amendments to the plan (it is still `active` and
un-executed-against, so revise in place per the collision rules).
