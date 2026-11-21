# Review — Workflow input CAS (`ctxVersion` on the resume), graph design note

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-097

## Scope & verdict

Plan gate for `falkor-chat/docs/plans/workflow-input-cas-graph.md` (first version, so no
growth-stops check applies yet). Baseline: `HEAD` `1bf0af95`, plus the working tree. The working
tree's uncommitted `repository.py`/`services.py` hunks (U1, K-091) are at
`repository.py:21,1829-1984` and `services.py:1316-1326`, so none of the functions under review is
touched. Read: the note, BACKLOG `### K-097`, DESIGN §6.2, QUERIES §12 (model block, status-move
contract, §12.1–§12.7, §12.13), SERVER.md route table and error table, and the code at
`services.py` `submit_workflow_input`/`sweep_due_workflow_runs`/`_drive_or_fault`/`get_workflow_run`,
`repository.py` §12 writers and `get_run`, `executor.py` `run`/`resume`/`_drive`/`_fail_with_note`,
`trigger.py` step 2, `api.py` run routes, and the test fakes and proxies. Root and `falkor-chat/`
`REVIEW.md` were applied. No falkor-chat suite was run, because `ws:test` belongs to U1. All
probes ran on throwaway keys that were deleted afterwards (`ws:probe-k097-review`,
`ws:probe-k097-review2`).

**ABA: reproduced.** I re-ran graph-dba's harness unchanged except for the graph key. Part A and
Part B match graph-dba's output line for line, apart from execution times. My output is in
`falkor-chat/docs/reviews/workflow-input-cas-evidence/aba_output.txt`, next to the harness copy I
ran (`aba_harness.py`) and graph-dba's original output (`aba_output_graph-dba.txt`). The "follow-up
probes" block in graph-dba's output was added by hand, not printed by the harness. I re-checked
its plan claim separately (`followup_probe.py` / `followup_probe_output.txt`): a CAS on a
`running` run with a matching version gives 0 records at `Node By Index Scan` and writes nothing.
I did not re-measure the RAM figure.

**Verdict: approve with suggestions.** The diagnosis is right and the CAS is sound. Every
writer's ruling holds against an unfiltered sweep. The findings are about tests and
documentation completeness, not the design.

CPG: used cpg_falkorchat — cross-checked the grep's caller set for the seven §12 run writers (call sites match: production writers only in `executor.py`/`services.py`).

## Findings

### Major

**M1 — The legacy-run path is never pinned through the service, so the most damaging mutant
survives. (Verified by tracing; the mutant was reasoned, not run.)** Test 5 says "a `WorkflowRun`
created without `ctxVersion` resumes once" but does not name the layer. The
`test_queries.sh` legacy case runs raw Cypher. Suppose `get_run` loses its `coalesce`
(`r.ctxVersion AS ctxVersion`). Then a legacy run reads `ctxVersion: None`. In
`executor.resume` that is a `ValueError` (500), or, if it is passed through, `None + 1` / `= null`
in the CAS, which refuses the run forever. Either way, every legacy waiting run (1,378 in `ws:demo`
per the note's own count) becomes permanently unable to take `/input` or a timer release. Tests
1–4 and 6 all create runs on new code (`ctxVersion: 0` stored), so none of them goes red.
**Fix:** test 5 creates the legacy node raw, then drives it through `submit_workflow_input` (200,
version becomes 1), then sends a stale `ctx_version=0` via the proxy (409). Add the same legacy
setup to one sweep case (`resumed`). Add "drop `coalesce` in `get_run`'s projection" to the
mutation check, with test 5 as the test that must go red.

### Minor

**m1 — Test 6's "start → 0" cannot fail. (Verified by reading.)** `get_run` projects
`coalesce(r.ctxVersion, 0)`, so a run whose `CREATE` lacks `ctxVersion: 0` reads exactly like one
that has it. Removing the term from `start_run`/`start_run_untriggered` survives every listed
test (root `REVIEW.md`: a check that cannot fail is a finding). **Fix:** have test 6 read the raw
property (`RETURN r.ctxVersion`, no `coalesce`) for both start paths. Or drop the `CREATE` term
and its test row: with `coalesce` kept permanently, the term is redundant (§2's "new runs never
take the `coalesce` path" buys nothing observable). Either choice is fine. Make it in the plan,
not at implementation.

**m2 — The mutation check names one mutant, so it under-specifies the gate.** The note says
"dropping the compare term must turn tests 1–4 red". The other load-bearing terms each need a named
red test. Each pairing below is traced, not run:

| Mutant | Must turn red |
|---|---|
| drop `r.ctxVersion = $ctxVersion + 1` from §12.13 | 1, 2, 4 |
| drop the bump from `resume_run` | 3, 6 |
| drop the bump from `fail_run` | 6 |
| drop `coalesce` in the compare (`r.ctxVersion = $ctxVersion`) | 5 (first resume refused) |
| drop `coalesce` in `get_run` | 5 via the service (M1) |
| start `CREATE` term | 6 with a raw read (m1) |

**m3 — The fakes paragraph is not grounded. (Verified by grep.)** `tests/test_services.py`'s
`FakeRepo` has no `resume_run_with_ctx` (`grep -rn "def resume_run" tests/` matches nothing).
`test_services.py` never calls `submit_workflow_input` or the sweep. Its `get_run` returns
test-seeded dicts. Every `/input` and sweep test runs on the real `wf_repo`
(`test_process_input.py`, `test_workflow_timers.py`), so the new key arrives for free. No test
compares a whole `get_run` dict by equality (`test_process_flow.py:176` compares only `ctx`).
**Fix:** replace the paragraph with that statement. Keep the one real instruction: any new
injection proxy that overrides `resume_run_with_ctx` must accept `ctx_version`, as the harness's
`RacingRepo.resume_run_with_ctx(self, ws, *, run_id, ctx)` would not.

**m4 — The §4 list of spec sentences misses copies of the same facts.** Each of these becomes
false or incomplete under the note and is not listed:
- QUERIES §12 **model block** (`QUERIES.md:1211-1212`), which lists `WorkflowRun`'s properties. It
  is a second copy of DESIGN §6.2's list, and only the DESIGN copy is listed.
- QUERIES §12.13 opening: "Parent: **§12.4** … which **remains in use unchanged**" and "§12.4
  **plus one `SET` term**". §12.4 now changes (bump), and §12.13 adds a compare and a bump.
- DESIGN §6.2 "Who writes the run `ctx`": "`resume_run_with_ctx` … is `resume_run` … plus one
  `SET` term, one query". Only a later sentence of that paragraph is listed.
- QUERIES §12 **status-move contract**: "zero rows = the anchor missed (run gone / wrong current
  state)". A stale version is a third cause.

`SERVER.md:427` ("lost the resume CAS (nothing written) → 409") stays true, and the route row at
`:312` cites §12.7, so SERVER.md needs no edit unless M-Q1 below is answered "filter". **Fix:** add
the four items to §4. At close, a carried-over contradiction is a finding (root `REVIEW.md`).

**m5 — The note's evidence citations point to a scratch directory.** §1 cites
`/tmp/…/scratchpad/k097/…`, which can be cleared. **Fix:** re-point it to
`falkor-chat/docs/reviews/workflow-input-cas-evidence/` (this review's copy; the only change to
the harness is the graph key). Label the hand-added "follow-up probes" as manual. The PROFILE
half is now re-checked; the RAM half has not been re-checked.

### Nits

- **n1 — Mixed-deploy wording.** The risk is stated correctly but incompletely. In both
  directions (an old-code winner that doesn't bump, or an old-code submitter that doesn't compare),
  the mixed window falls back to **today's** status-only behaviour. It is never worse, and nothing
  persists after the restart. So an operator note is enough and no implementer mitigation is
  warranted. Say that, and name the processes: the `:8200` server (MCP is mounted in-process,
  `.mcp.json` → `http://localhost:8200/mcp`), the `start_demo.sh` storefront uvicorn (its chat
  path calls `resume_run` via `trigger.py:78`) and any `start_server.sh` uvicorn.
- **n2 — Test 1's title.** "R-1 interleave" is really harness case A3, an ABA. Pure R-1 (case
  A1) is already 409 today. Rename it to avoid an implementer "fixing" a test that was never red.
- **n3 —** `_status_row` returns `{runId, status}` only, so §2's new `ctxVersion` return column is
  dropped. That is harmless. Either drop it from the query or say the repo ignores it.
- **n4 —** `executor.resume`'s guard must test `ctx_version is None`, not falsiness. Version 0 is
  the normal first value. Tests 1–4 catch the falsy bug, so this is only worth one line in the
  step text.
- **n5 — Shared code with K-096.** U3 (this note) and U6 (K-096) both edit
  `submit_workflow_input`. K-096 adds a `get_run` read-back on a failed drive. The ledger already
  runs them one at a time, and nothing here depends on K-096's behaviour, so no citation is owed.
  Whichever lands second rebases onto the other's docstring.

## Rulings on the brief's questions

1. **Is the ABA real?** Yes. Reproduced as stated above. The proxy only reorders real calls: it
   delays B's `resume_run_with_ctx` (on `executor._repo`) until A's complete request has run, and
   B's read and merge used the real `services._repo`. Nothing in the shipped code prevents that
   order. No lock guards `/input` (the only locks are the two clock locks, `services.py:828`,
   `executor.py:589`). The routes are sync `def` in anyio's threadpool
   (`FALKORCHAT_THREAD_LIMIT`, default 100). Several processes can share the instance (n1). The
   window is short (B's validate-and-merge time), so the race is rare, but it is reachable. The
   sweep's step-5.1 read-to-CAS gap is a second real instance of it.
2. **Is the invariant complete?** Yes. An unfiltered grep over `falkor-chat/` (all files, docs
   excluded only after reading them) finds every `WorkflowRun.ctx`/`status` write in
   `repository.py`. The other hits are `test_queries.sh` copies of the same Cypher, and one
   test-only raw `SET r.status='done'` at `test_storefront.py:2713`. The CPG shows the same call
   sites. Rulings:

   | Writer | Note's choice | Ruling |
   |---|---|---|
   | `start_run` / `_untriggered` | `ctxVersion: 0` | Right in effect. Redundant with permanent `coalesce`, so the test must read raw (m1). |
   | `resume_run` (chat) | bump, no compare | Right. A chat reply writes no ctx, so it has nothing stale to compare. Its resume still ends a wait instance (probe B2). Process runs park with `waitingThreadId=''` and are never found by §12.9. The bump only matters for triggered runs that also receive `/input` or a sweep, and it costs nothing. |
   | `resume_run_with_ctx` | compare and bump | Right. |
   | `fail_run` | bump | Harmless, and consistent with "every ctx write bumps". Not load-bearing (terminal). |
   | `suspend_run` | none | Right. It is only reached from `running`, which only a bumped resume or a start (version 0, never read as `waiting` before the park) can produce. |
   | `complete_run` | none | Right. Terminal, and it writes no ctx. Unguarded on status, but its only caller is `_drive_loop` after a won CAS. |
   | `record_step_and_advance` | none | Right. It moves `AT_STEP` only while `running`. A waiting reader's `atStepKey` is pinned because any later `AT_STEP` change needs a resume first, which bumps. That makes D-H validation sound. |

   The service refuses a read whose status is not `waiting` (`services.py:2491`, sweep `:2833`).
   So no read taken during `running` can carry a version into a later re-park. This is the
   premise that makes "bump on exit from `waiting`" enough.
3. **Is the Cypher right?** Yes. The `coalesce(r.ctxVersion, 0)` default is consistent between the
   compare, the read and the bumps. Zero rows still means nothing written (B1:
   `properties_set=0`; my follow-up probe: a `running` run with a matching version is 0 at the
   scan, and `ctx` and version are unchanged). The service maps zero rows to `None` → 409
   (`services.py:2526-2532`), and the sweep maps it to `raced` (`:2886`). The plan claim
   reproduced: `status` still folds into the scan, and the version is a residual `Filter` over at
   most one record. The RAM figure (~63–72 B/run) is plausible for one int attribute; I did not
   re-measure it.
4. **Should `GET /workflow-runs/{runId}` gain `ctxVersion`?** **Expose it. Don't filter it out.**
   Reasons: (a) the route is a documented 1:1 RO pass-through of §12.7 (`api.py:522-530`,
   `SERVER.md:312`, with no `response_model`), and filtering one key would be its first divergence
   from the query; (b) an added JSON key breaks no reader (the web UI and `test_api.py:1998` read
   named keys); (c) the value is a bare counter and reveals nothing sensitive; (d) it is exactly
   the per-wait-instance identifier K-038 m7 needs. One condition: the close treats it as a public
   contract addition. §12.7 is listed already, and the `HISTORY.md` entry should name the new key.
   Telling the stakeholder is enough; this needs no decision from them. See M-Q1.
5. **The mixed-deploy risk** — see n1. The direction is correct, the window is no worse than
   today, and an operator note is enough.
6. **Can a faithful implementer carry out the step table, the six tests and the mutation check?**
   Yes, all three are reachable. Test 4's nested human submit passes through the proxy because the
   proxy fires once. The fixes needed are M1, m1 and m2 (what each test pins) and m3 (the fakes
   paragraph). The §4 sentence list is incomplete (m4).

## What's solid

- The ABA diagnosis is executed, not argued, and it is sharper than the backlog's guess. The
  backlog assumed the drive's ctx output is what gets lost. The note shows the drive writes no ctx
  (`fail_run` is the only ctx writer in `executor.py`), so it is the **winner's input** that is
  erased, and in A2 that input is one a guard already branched on.
- One compare covers R-1, the ABA, a chat resume in the gap, and the sweep's own gap. It needs no
  backfill, no DDL and no index. The no-index call is right: the property is only ever compared
  after a `runId` point lookup.
- The repo signature makes `ctx_version` keyword-only with no default, so a caller can't skip the
  compare by accident. The service does not split a version miss from a status miss, so the
  client contract stays "re-read and retry".
- `If-Match` is kept out of scope deliberately, with its reason stated.

## Open questions

- **M-Q1 (stakeholder, informational):** `GET /workflow-runs/{runId}` gains a `ctxVersion` key. My
  ruling is to expose it (question 4). If the stakeholder wants it hidden instead, the filter
  belongs in `services.get_workflow_run`. SERVER.md's route row would then stop being a pure §12.7
  pass-through and must say so.

## Pass 2

Re-gate of the note as revised in place (2,520 → 2,623 words). I re-read the whole note and the
code it cites, and checked the fixes against each other. No suite was run and no graph was
written in this pass. Every check below is by reading and tracing.

**Verdict: approve with suggestions.** All Pass 1 findings are closed. Two of the fixes create
new minor issues when read together (P2-1, P2-3), and the growth is unoffset rework (P2-2).
Nothing blocks implementation.

CPG: considered, not relevant — Pass 2 re-checks the note's text against code already traced in Pass 1; no new call-graph question.

### Dispositions

- **M1 — fixed.** Test 5 now goes through `submit_workflow_input` (200, then 409 on stale), plus
  a sweep variant, with the `get_run`-without-`coalesce` mutant in the gate. The sweep variant's
  setup is unreachable as written (P2-3).
- **m1 — fixed, by dropping the term.** Start queries are unchanged, and test 6 reads the raw
  property ("absent after start"). This introduces P2-1.
- **m2 — fixed.** A six-row mutant table is in place, and each row's red test traces. Two rows
  understate their reach; that is intended (P2-4).
- **m3 — fixed.** The fakes paragraph now matches `tests/` (rechecked: there is still no
  `def resume_run_with_ctx` under `tests/`).
- **m4 — fixed.** All four missing items are in §4. §12.1/§12.12 are correctly dropped after m1.
  The list is complete apart from P2-1's wording, which would carry into DESIGN.
- **m5 — fixed.** The evidence is re-pointed to `workflow-input-cas-evidence/`, the hand-added
  block is labelled, and the RAM figure is marked unverified.
- **n1 — fixed** (§3 Deploy). **n2 — fixed** (test 1 is renamed to the A3 ABA). **n3 — fixed**
  (§2 now returns `{runId, status}`). **n4 — fixed** (`is None`). **n5 — no change, correctly**:
  it asked for none.
- **Q4 ruling — folded in** (§2 "Public addition", with a `HISTORY.md` naming obligation at
  close).

### The coordinator's cross-checks

- **(a) Dropping the start term vs. the invariant: the invariant as written is now false.** §2
  says "Every query that writes `ctx` or moves the run out of `waiting` bumps it". Both start
  queries write `ctx` (`CREATE … ctx: $ctx`) and no longer write `ctxVersion`. The behaviour is
  still sound: no read of a run can predate its creation, and a new run reads 0 through
  `coalesce`. Only the sentence is wrong, and §4 copies it into DESIGN §6.2's run-model notes as
  an absolute. The `fail_run` row's "keeps 'every ctx write bumps' true" leans on the same false
  absolute. See P2-1.
- **(b) The mutants now that new and legacy runs look the same.** They look the same on purpose.
  There is one representation before the first resume, and that simplifies the design.
  Consequences:
  - The `get_run`-without-`coalesce` mutant now breaks **every** run's first `/input`.
    `ctxVersion` reads `None`, then `executor.resume` raises `ValueError`, which escapes
    `_drive_or_fault`'s four caught types, so the result is a 500, or `faulted` in the sweep.
  - Likewise, the compare-without-`coalesce` mutant refuses every first CAS (`null = 0`).
  - Both mutants therefore turn tests 1–4 red as well as 5. That is intended and stronger, and
    the table's "5" undercounts without being wrong (P2-4).
  - Test 6's raw read still discriminates: "absent after start" goes red if someone adds a
    start term, and the +1 assertions go red on the `resume_run`/`fail_run` bump mutants.
  - Test 5 no longer pins anything tests 1–4 don't (P2-3).
- **(c) n3 vs. the 409 mapping: fits.** `_status_row` (`repository.py:3431-3436`) reads
  `row[0], row[1]` and returns `None` on zero rows. That makes `executor.resume` return `None`,
  which becomes 409 (`services.py:2526-2532`) or `raced` (`:2886`). Nothing on the CAS path needed
  the dropped column. Test 5 reads the raw property separately.
- **(d) m4 after m1: complete, with one carry-over.** I re-swept DESIGN §6.2 and QUERIES §12
  (model block, status-move contract, locked shape decisions, build-quirks list, §12.4/§12.5/
  §12.7/§12.13) plus `SERVER.md:312,427`. No other sentence becomes false. The one gap is the
  invariant bullet itself (P2-1): when written into the spec, it must say that a new run carries
  no `ctxVersion` and reads as 0.

### Growth-stops check (root `REVIEW.md`)

+103 words. None of it is new scope; it is all rework answering Pass 1:
- the m2 mutant table;
- the m4 additions to §4;
- the M1 test 5 body;
- the n1 process list;
- the m5 evidence labels.

The offsets were removed only in part: the start row shrank and the fakes paragraph was
replaced. Fixed text that adds no information remains (P2-2).

### New findings

**P2-1 (minor) — The §2 invariant contradicts the start queries.** It is verified by reading
`repository.py:2836-2848,2881-2893` against note §2. **Fix:** state the invariant as "every
query that writes `ctx` **of an existing run**, or moves it out of `waiting`, bumps
`ctxVersion`. A run is created without the property and reads as 0 through `coalesce`".
Reword the `fail_run` row's reason to match. §4's DESIGN bullet then inherits the corrected
form.

**P2-2 (minor) — Rework growth is not offset.** Under root `AGENTS.md`'s growth-stops bullet,
rework replaces text in place. The available offsets:
- §1's writers table duplicates §2's "Every other writer" table plus the "Every exit from
  `waiting`" bullet;
- the "ccg list was call-only … name-match false positive" sentence is process history;
- test 5 and the third `test_queries.sh` case are now redundant (P2-3).

Dropping these takes the note below 2,520 words without losing a fact.

**P2-3 (minor) — Test 5 is redundant, and its sweep variant cannot pass as written.** It is
traced, not run.
- Since m1, a run started normally has no `ctxVersion`, exactly like a legacy node. Tests 1–4
  already drive such runs through the service. Test 5's raw setup buys nothing they don't pin.
- The sweep variant ("a due legacy `wait` run lands in `resumed`") builds a raw node with only
  `OF_DEF`/`AT_STEP`. `find_due_wait_candidates` takes `parkedAt` from
  `OPTIONAL MATCH (r)-[:LAST_STEP_RUN]->(sr)` (`repository.py:3244-3247`), so it reads `null`.
  For a `waitForSeconds` step, `_wait_due_at` then returns `None` and the run is never `due`.
  The test cannot reach `resumed` unless the setup also creates a `StepRun` tail or uses
  `waitUntil`.

**Fix:** drop test 5's raw-node setup. Move its two assertions (first `/input` 200 with raw
version 1; stale `ctx_version=0` → 409) into test 6. Add one sweep case on a normally started,
parked `wait` run (no version), asserting `resumed`. Point the two `coalesce` mutant rows at those
tests. In `test_queries.sh`, the "node with no `ctxVersion` is resumed by `ctxVersion=0`" case is
the same as the first new case on a `START_UNTRIG` node, so keep one.

**P2-4 (nit) — The two `coalesce` mutant rows undercount.** Since m1 they turn tests 1–4 red too,
because every run starts with no `ctxVersion`. That is the intended single representation.
Listing "1–5" makes the gate's strength visible.
