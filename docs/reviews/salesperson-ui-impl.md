# The one salesperson UI — Implementation Review (S1–S4, S6, S7)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M<n> TBD) · **Reviews:** `docs/plans/salesperson-ui.md` §5.1 rows S1, S2, S3, S4, S6, S7

## 1. Scope & verdict

**Reviewed:** the two uncommitted working-tree changes delivering `docs/plans/salesperson-ui.md`
(v1.2, `Status: active`) §5.1 steps **S1** (`salesperson@v6` def bump, owner `coder`) and **S2**
(chat-path `run_ctx` merge, owner `tdd-engineer`), each against **its own §5.1 row's scope, files
and done-condition** — not against the whole plan. Baseline: `git diff` against `HEAD`
(`4bb96e1`); nothing is committed, so the diff is exactly the review surface. 13 files, +400/−39.

**Also read for grounding:** `docs/plans/salesperson-ui.md` §4.5/§4.6/§4.10/§5.0/§5.1,
`docs/reviews/salesperson-ui.md` (2 passes), `falkor-chat/AGENTS.md`,
`falkor-chat/docs/BACKLOG.md` K-060/K-062, `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md`
§11, and the surrounding production code (`services.py`, `trigger.py`, `executor.py`, `api.py`,
`schemas.py`, `background.py`, `repository.py`, `tests/conftest.py`).

**Verdict: needs changes.**

One **blocker** — and it is not in the diff's *text*. The `salesperson@v6` **version identifier S1
chose is already taken in the live `ws:acme` workspace by a different, uncommitted def**, and
create-only materialize makes that unfixable by re-seeding. S1's own done-condition
(`seed_salesperson.sh <ws>` → `verify_salesperson.sh <ws>` exits 0) is therefore currently unmet
and unmeetable on `ws:acme`. This also **corrects the root cause `teco` had provisionally
established** (a self-inflicted double-seed): the stale snapshot is genuinely pre-existing. See
F-1 and Appendix A.

Everything else is one **major** (F-2, an unbounded ctx write S2 widens), three **minors** and two
**nits**. The S1 and S2 *code* is, on its own terms, correct, well-tested and idiomatic.

**CPG: used `cpg_falkorchat` — confirmed the complete production caller set of
`start_workflow_run` (exactly two: `api.py:394`, `trigger.py:76`) and of `maybe_trigger` (exactly
one: `background.py:265`), which is what makes S2's `None`-default back-compat claim checkable
rather than assumed.**

### What I verified by execution vs. judged statically

| Verified by running it | Judged statically |
|---|---|
| Full suite: **2330 passed, 14 deselected** (matches `teco`'s independent figure) | Diff-vs-plan-row conformance (files, interfaces, done-conditions) |
| `ruff check` clean on all 7 touched `.py` files (34 pre-existing errors are elsewhere) | Doc accuracy of `AGENTS.md` rows 82–83 and the `proof_defs.py` comments |
| 3 mutation spot-checks (guard removed / guard moved after the write / merge direction reversed) — Appendix B | §5.0 shared-file-map staleness |
| The exact process-path ordering delta of the hoist (id/clock consumption) — Appendix C | |
| `ws:acme`'s live `salesperson@v6` snapshot byte-compared against `proof_defs.SALESPERSON_DEF` | |
| `verify_salesperson.sh acme` (read-only; exit 1) | |
| `SALESPERSON_DEF` v5→v6 structural diff: `systemPrompt` only; `tools`/`model`/`requiredTools`/`maxIterations`/`waitsForHuman`/topology/`MAX_STEPS` all identical | |
| The `CONTEXT:` block mechanism S1's prompt sentence depends on (`executor.py:605`, `:1276`) | |

---

## 2. Findings

### F-1 · **Blocker** · S1 — `salesperson@v6` collides with a pre-existing, uncommitted `v6` in `ws:acme`; the step's done-condition cannot be met there

`ws:acme` holds a `WorkflowDefSnapshot {key:'salesperson', version:'v6'}` whose `systemPrompt`
contains **neither** of S1's two sentences. It is v5's prompt plus the K-060 *"check every returned
item, never drop a match"* safety-net wording that `falkor-chat/docs/BACKLOG.md:67` records as
**"Reverted, never shipped."** That string exists nowhere in the working tree and nowhere in
`git log --all -S` — it was published from a working-tree-only experiment. `config` is create-only
(`proof_defs.py` module docstring; `services.materialize_def`'s `_check_no_structural_conflict`
treats config-only differences as a **silent no-op**), so re-seeding can never overwrite it.

Consequence: on `ws:acme` the name `salesperson@v6` denotes the wrong prompt permanently, and the
S1 row's done-condition (`seed` → `verify` exits 0) is unmeetable. Blast radius is contained —
S11 pins `FALKORCHAT_WS_ID=demo` (§4.9 move 2) and QA uses throwaway `ws:qa-*` — but `ws:acme` is
the repo's populated dev workspace and the next person to drive `v6` there gets the reverted
experiment with no error.

**This supersedes the double-seed attribution.** The `git log -S` disproof was sound about git but
cannot see an uncommitted experiment; the author's "pre-existing defect" report was substantially
right. Full causal chain in **Appendix A**.

**Suggested fix — `teco`'s call, two options.** (a) **Bump S1 to `v7`** across `proof_defs.py`,
both scripts, the scaffold test, `AGENTS.md` rows 82–83, plan §4.5/§4.10/§5.1-S11 and
`salesperson-ui-coordination.md` — restores the create-only invariant's meaning, sidesteps the
poisoned snapshot, ~8 string edits, no graph surgery. **Recommended.** (b) Delete the
`salesperson@v6` `WorkflowDefSnapshot` + `Step` subgraph from `ws:acme` (approval-gated,
`graph-dba`), then re-run the done-condition. Cheaper in edits, but mutates a shared dev workspace
and leaves `v6` ambiguous in every historical doc. Either way, **record which workspace the
done-condition was finally satisfied against** — the author used a probe graph, `ws:s1v6`, which
does hold a v6 byte-identical to the file (verified).

### F-2 · **Major** · S2 — `start_workflow_run` still has no service-side `run_ctx` size bound, and S2 opens the first direct-service write path

`submit_workflow_input` (`services.py:2131`) and the timer sweep (`services.py:2458`) both enforce
`MAX_CONFIG_LEN` **in the service**, for the reason `schemas.py:256-257` states outright: "MCP and
direct service callers never see a schema." `start_workflow_run` is the one ctx-mutating entry
point that does not — it relies solely on `StartWorkflowRunIn._check_ctx`. Before S2 that was
tolerable: the chat path wrote a fixed-size `{"threadId": …}`. S2 makes the chat path accept caller
ctx, and its intended caller (S9's storefront, via `trigger.maybe_trigger`) is a **direct service
call** that bypasses `schemas.py` entirely. An oversized ctx is not merely a big graph property:
`executor._assemble_messages` re-emits the whole ctx as a `CONTEXT:` turn on every LLM call, so it
costs context window every turn, forever.

Not a blocker: no caller passes `run_ctx` on the chat path yet, and S6's `_LOCALES` allow-list will
constrain the one real value. But the guard belongs next to its sibling `_reject_reserved_keys`
call, not in a future step's discipline.

**Suggested fix:** four lines immediately after `self._reject_reserved_keys(caller_ctx, …)`,
mirroring `services.py:2131` verbatim — `if len(self._dump_ctx(caller_ctx)) > MAX_CONFIG_LEN:
raise WorkflowInputRejectedError(...)` — plus one test per path. Route to `tdd-engineer` as an S2
amendment (it is inside S2's own file column), **not** deferred to S8/S9.

### F-3 · **Minor** · S2 — the merge direction is untestable behaviour; make it an *invariant* test instead of a comment

Verified by execution: reversing the merge to `{"threadId": thread_id, **caller_ctx}` leaves the
**full suite green (2330 passed)**. The author's stated reasoning is therefore correct, and
declining to write a test that cannot fail was the right call. The residual risk is not today's
behaviour — it is that the merge's safety rests on an unstated, unenforced invariant: *every key
the engine writes into the start anchor is a member of `RESERVED_CTX_KEYS`*. Add a second anchor
key that is not reserved (a `channelId`, a `locale`) and the direction becomes silently
load-bearing with zero test coverage.

**Suggested fix:** hoist the anchor's key set to a module constant used by the merge —
`_CHAT_START_ANCHOR_KEYS: frozenset[str] = frozenset({"threadId"})` — and add one test that
**can** fail: `assert _CHAT_START_ANCHOR_KEYS <= RESERVED_CTX_KEYS`. Cheap, and it fails the moment
the invariant is broken rather than the moment a caller exploits it.

*Answering the brief's question 2 directly: no, I found no path where `caller_ctx` reaches the
merge unscreened.* `caller_ctx` is bound once and the guard and the merge are the only readers;
there is no mutation, no re-entry and no alternate entry point between them (CPG-confirmed caller
set). The guard is complete **for the current anchor**.

### F-4 · **Minor** · S2 — the plan's §5.0 shared-file map is now stale in three places

§5.0 is "what dispatch is gated on", regenerated from §5.1's Files column. S2 wrote three files it
does not list under S2: `tests/test_process_input.py` (listed nowhere at all),
`falkor-chat/docs/QUERIES.md` (§5.0 assigns it to **S4** only), and `falkor-chat/docs/HISTORY.md`
(absent from §5.0 entirely — only root `docs/HISTORY.md` appears, under S16).

**Collision risk is nil, but the map must not stay wrong.** `test_process_input.py` is touched by
no other step (S4 takes `test_repository.py` + `test_services.py`). `QUERIES.md` is safe only
because S2 → S4 is already ordered for `services.py`; S4's owner should be told §12.1/§12.12 were
already edited, so the S4 §18 addition is an append, not a merge.

**Suggested fix:** `architect` regenerates §5.0 from the *delivered* file lists (the coordination
doc already records them), adding `falkor-chat/docs/HISTORY.md` as a row with an explicit ordering
— every remaining step will append to it.

### F-5 · **Minor** · S1+S2 — inconsistent `HISTORY.md` discipline between two sibling steps

S2 added a 43-line `falkor-chat/docs/HISTORY.md` entry; S1 added none. Root `AGENTS.md` says
`HISTORY.md` is "a dated change log — append an entry for every delivered change", so as it stands
the delivered `salesperson@v6` bump has no history record at all, while a smaller sibling change
has a full one. Whichever convention this milestone adopts, S1 and S2 must match.

**Suggested fix:** `teco` picks one — per-step entries (then S1 owes one) or one entry at milestone
close (then S2's is early, but harmless and should stay). If per-step, note that S2's entry says
"both production callers (`api.py`'s `POST /workflow-runs`, `background.py`'s `_safe_run_workflow`)
are unchanged" — `background.py` calls `maybe_trigger`, not `start_workflow_run`; the claim is true
of the pair but reads as a single call-graph statement.

### F-6 · **Nit** · S2 — `schemas.py:256` still omits `timerFired` from its reserved-key list

S2 correctly fixed exactly this drift in `QUERIES.md` §12.12 ("gained the `timerFired` key it had
been missing since K-028"), but the identical stale list survives in `schemas.py:256-257`
("Reserved keys (`threadId`, `error`)…"). Out of S2's file column, so leaving it was right.
**Route to S8's owner**, who edits `schemas.py` anyway.

### F-7 · **Nit** · S2 — the hoist changes id/clock consumption on a rejected process-path start

Verified by execution (Appendix C): the *only* observable process-path delta is that a rejected
start no longer consumes an `id_gen()` / `clock()` tick. `_default_id` is `uuid.uuid4().hex` and
`_default_clock` is wall-clock — both pure — so there is **no production-observable change**:
exception type, message and precedence are all identical, and `WorkflowEngineDisabledError` still
wins over `WorkflowInputRejectedError` because `_require_executor()` still runs first. No existing
test depends on the consumed tick (suite green). Worth knowing only if a future test injects a
counting `id_gen` and asserts on the sequence. **No action.**

*Answering the brief's question 1 directly: the hoist changed nothing observable on the
pre-existing process path.* No repository call preceded the guard in either version, so "rejected
before any write" is unchanged there too — the hoist strictly *added* that property to the chat
path.

---

## 3. The two out-of-scope findings the S2 author raised (brief item 6)

**(a) Unbounded `run_ctx`.** Promoted to **F-2, Major** — sharper than filed: the service layer
already implements this bound twice for the sibling entry points, so the omission is an
inconsistency in a stated doctrine, not a missing feature. Fix it in S2, not later.

**(b) `conftest.py`'s `wf_repo` wipes `reference` with no teardown.** **Minor, no action now — but
it is a load-bearing contributor to F-1.** I reproduced it as a side effect of running the suite:
after my run, `reference` held 1 `WorkflowDef` + 4 `Step` nodes (one test's leftovers) and
`verify_salesperson.sh acme` reports both defs `reference def : MISSING`, exit 1. This is already
documented — `services.diff_def_snapshot`'s docstring names it "the documented trap", and
`falkor-chat/AGENTS.md` row 79 records the K-005 fix for its false-negative cousin — so it is a
known, accepted cost. What is *not* documented is its interaction with F-1: because the
`reference` side is wiped on every suite run and the `ws:` side is not, a stale workspace snapshot
can only ever be caught in the narrow window where both sides exist. Adding a teardown that
re-seeds `reference` is **not** worth it (it would need the seed scripts in the test path). The
proportionate fix is documentation plus F-8 below.

### F-8 · **Minor** · S1 — `verify_salesperson.sh` cannot detect the failure mode its own advisory warns about

The script already prints *"If they DIVERGE, do NOT re-seed — a create-only re-publish cannot
overwrite the stored def"* (pre-existing text, unchanged by S1). But it only ever compares
`reference` **against** `ws:<id>` (`services.diff_def_snapshot`), plus a per-def topology check.
It never compares either side against `proof_defs.SALESPERSON_DEF`. So the F-1 state — workspace
snapshot present, correct topology, wrong `systemPrompt`, `reference` absent — prints
`topology … OK` and diagnoses only the missing `reference` half. That is exactly the shape a
create-only trap produces, and exactly the shape the script's advisory addresses.

**Suggested fix (answers brief item 5).** Nothing *in the diff* makes the trap easier to fall into
— S1 changed only three version-default strings and comments, and the advisory was already there.
The gap is detection. Add one read-only check to `verify_salesperson.sh`: import
`falkorchat.proof_defs.SALESPERSON_DEF`, and for each present side compare the `assistant` step's
`config` dict against the file's, failing with `SNAPSHOT DIVERGES FROM proof_defs.py — this
version is create-only; bump the version` when they differ. ~10 lines in the existing embedded
Python heredoc, no new dependency, and it turns S1's done-condition into a check that can actually
detect the defect it is meant to gate. Route to `coder` alongside whichever F-1 option is chosen.

---

## 4. What's solid

- **S2's ordering is genuinely correct and genuinely tested.** Two independent mutations — removing
  the guard, and moving it after `repo.start_run` — both go red, the latter caught by *both* the
  offline `repo.calls == []` assertion and the live no-run/no-`Message` assertion (Appendix B).
  `FakeRepo.get_message` really does record into `calls`, so "nothing even read" means what it says.
- **The fifth file was justified (brief item 3).** `tests/test_process_input.py` is the file that
  already owns the real-graph chat-path harness (`_materialize` + `wf_repo`), already uses
  `wf_repo._graph("test").ro_query(...)` at four pre-existing sites, and is touched by no other
  step. Putting a real-DB "before any write" assertion anywhere else would have meant duplicating
  that harness. It is strictly additive to the offline coverage rather than a substitute.
- **S1's `v6` content is exactly what the plan asked for.** Structural diff against `HEAD`'s `v5`:
  `systemPrompt` is the *only* changed field — `tools` (11, identical), `model`
  (`lmstudio/mistralai/ministral-3-3b`, carried forward as the create-only rule demands),
  `requiredTools`, `maxIterations`, `waitsForHuman`, both step keys, the single transition and
  `SALESPERSON_MAX_STEPS` are all unchanged. No stray whitespace, no double spaces, both sentences
  read cleanly in context.
- **The `V5_TOOLS` baseline is a real baseline.** Hard-coded rather than derived, and it matches
  the shipped `v5` set exactly (verified). The `len(tools) == len(set(tools))` duplicate check is a
  good instinct for a hand-edited cumulative republish.
- **S1's premise is grounded.** I verified the mechanism the `language` sentence depends on:
  `executor._drive_loop` re-reads `run_ctx` from the run (`executor.py:605`) on every drive, and
  `_assemble_messages` emits it as a literal `CONTEXT:\n{…}` turn (`executor.py:1276`). The
  sentence's reference to "the CONTEXT block" resolves against a real string in the real prompt.
- **Doc scope discipline on S1 is exactly right.** `AGENTS.md` rows 82–83 updated; `BACKLOG.md`'s
  K-060/K-062 `v5` pins deliberately untouched, as the S1 row instructs. Every other surviving
  `salesperson@v5` reference in the repo is in an archived report or an owned-elsewhere doc.
- **Both prompt additions are correctly fenced as unproven.** `proof_defs.py`, the seed script and
  `AGENTS.md` each say plainly that these are prompt-adherence claims on a 3 B model, not signed
  off by code review, with §6.3 #5/#7 as the measured gate and a named fallback. That is the right
  posture and it is stated three times where a reader will hit it.
- `ruff` clean on every touched file; QUERIES.md §12.1's genuinely-wrong `"{}" at start` comment
  fixed as a bonus.

---

## 5. Open questions (need `teco`'s or the user's call)

1. **F-1: `v7` bump, or graph surgery on `ws:acme`?** My recommendation is (a) `v7`, but it ripples
   into the plan, the coordination doc and S11's env pin, so it is not mine to decide.
2. **F-5: per-step or per-milestone `HISTORY.md` entries** for this build?
3. **`ws:s1v6` is left in place** (it holds the correct `v6`, and is useful evidence for F-1). I
   created no probe graph of my own. Deleting it is approval-gated and I have not asked.

---

## Pass 2 — 2026-09-02 (re-gate of S1b + S2b)

**Reviewed:** the same working tree after S1b (`coder`) and S2b (`tdd-engineer`) actioned every
Pass 1 finding, plus plan `docs/plans/salesperson-ui.md` v1.5 and
`docs/plans/salesperson-ui-coordination.md`. Baseline unchanged: `git diff` against `4bb96e1`.

**Verdict: approve with suggestions.** No blockers, no majors. Every Pass 1 finding is disposed of
correctly, including the one that was deliberately *not* fixed. Four nits and one minor, all new,
all optional before the build continues.

**CPG: considered, not relevant — Pass 2's questions (bound arithmetic, mutation sensitivity,
shell logic) are not call-graph questions; Pass 1's `cpg_falkorchat` caller-set finding stands
unchanged, since S1b/S2b add no call sites.**

Verified by execution this pass: full suite **2336 passed, 14 deselected** (matches `teco`);
`ruff` clean on all 8 touched `.py` files; **6 new mutations** (Appendix D); `bash -n` plus a live
**negative and positive control** on the new `verify_salesperson.sh` check (Appendix E); the exact
anchor-overhead arithmetic; and `db.constraints()` on `ws:acme`.

### Pass 1 dispositions

| # | Disposition | Evidence rechecked |
|---|---|---|
| **F-1** Blocker | **Fixed** — `v6`→`v7` across all five S1 files, plan, coordination doc; `v6` recorded as burned in `proof_defs.py`, `AGENTS.md:94` and `HISTORY.md` | `SALESPERSON_DEF["version"] == "v7"`; both scripts default `v7`; scaffold test pins `v7`; `ws:s1v7` holds a v7 byte-identical to the file; `ws:acme` still holds only the old `v6` — no live collision at `v7` |
| **F-2** Major | **Fixed, with a deliberate deviation from both siblings** — bound at `services.py:2052-2055`. The deviation is *correct*; see Q1 | Mutation `>`→`>=` now fails the intended test (Appendix D) |
| **F-3** Minor | **Fixed** — `CHAT_START_ANCHOR_KEYS` + the `_chat_start_ctx` seam + three tests | The Pass 1 reverse-merge mutation that left the whole suite green now fails **exactly one** test, the intended one |
| **F-4** Minor | **Fixed** — plan v1.3 §5.0 carries all three missing rows with ordering, plus the explicit note to S4's owner | Read §5.0 rows 19, 20, 22 |
| **F-5** Minor | **Fixed** — per-step convention chosen and recorded in §5.0; S1's entry added | `HISTORY.md:8` (S1), `:58` (S2) |
| **F-6** Nit | **Routed, not fixed — correct** | Plan v1.4's S8 row now names `schemas.py:256-257`; the list is still stale, as intended |
| **F-7** Nit | **No action — correct** | Unchanged |
| **F-8** Minor | **Fixed** — `verify_salesperson.sh` check 6. Fires on the real F-1 artifact; no false positive on a clean def. One gap remains: **N-1** | Appendix E |

### N-1 · **Minor** · S1b — the new drift check compares step `config` but not transition `guard`, which is create-only too

`TRANSITION.guard` is written with `ON CREATE SET rel.guard = tr.guard`
(`repository.py:1748`) — the *same* create-only rule that motivates the whole check. The new loop
iterates `structure["steps"]` only, so a drifted guard is invisible: exactly the defect class F-8
exists to catch, one edge type over. It is not hypothetical — `SALESPERSON_DEF` has one guarded
transition and `ORDER_FULFILLMENT_DEF` has three.

Proven, not inferred: I drifted `"op": "truthy"` → `"op": "DRIFTED"` in a copy-aside
`proof_defs.py` and re-ran `verify_salesperson.sh s1v7`. The script printed **no divergence line**
and no new `RESULT` entry; only the pre-existing `reference`-missing failures. File restored
byte-identically (`git diff --stat` unchanged, 0 `DRIFTED` remaining).

**Suggested fix:** the structure read already exposes it (`services.py:409`,
`"guard": t["guard"]`), so this is a second small loop beside the existing one — key transitions
by `(from, to, on, order)` against `source_def["transitions"]` and compare `guard`, reusing the
same failure text. ~12 lines. Route to `coder` with S1b.

### N-2 · **Nit** · S2b — the call-site comment understates its own deviation by ~2.3×

`services.py:2048-2051` says the chat path "may exceed this by the anchor's own **~20**
characters". Measured: the anchor adds `,"threadId":"<tid>"` = **14 + len(thread_id)**. With a
real server-minted thread id (`_default_id` = `uuid4().hex`, 32 chars) that is **46**, so the
largest writable chat ctx is 8046, not ~8020. The `~20` looks borrowed from the *sweep's*
`timerFired` marker, which really is 20.

The conclusion is unaffected (see Q1 — the bound is soft), but a future reader sizing anything
against this comment would be wrong by 26 characters. **Suggested fix:** replace `~20` with
`14 + len(threadId)` (≈46 for a server-minted id).

### N-3 · **Nit** · S2b — `test_workflow_timers`'s boundary test is now coupled to the *start* bound

Observed: the `>`→`>=` mutation on `start_workflow_run`'s new bound turns
`test_sweep_faults_a_candidate_whose_merged_ctx_would_exceed_max_config_len` red as well as its
own intended test. Unavoidable — the fixture seeds through the public API, which is exactly what
made the resize necessary — but it means a future failure there no longer localises. **Suggested
fix:** one sentence in that test's existing comment saying a failure may indicate the *start*
bound rather than the sweep's. No code change.

### N-4 · **Nit** · S1b — `seed_salesperson.sh`'s version-chain comment jumps `v5` → `v7` silently

The header narrates the chain (`…K-057's wording fix bumped it to v5; the storefront demo bumped
it to v7`) with no mention of the gap, while `proof_defs.py` and `AGENTS.md:94` both explain it.
A reader in the script alone sees an unexplained skip. **Suggested fix:** one clause — *"`v6` is
burned, see `proof_defs.py`"*.

### N-5 · **Nit** · not S1/S2 — `docs/plans/salesperson-ui-graph.md:99` still pins `salesperson@v6`

S0's fixture inventory lists `WorkflowDefSnapshot salesperson@v6` among the survivors. Harmless
(any snapshot serves as a survivor), but S4 executes against that note. **Route to `graph-dba`**
with the next `-graph` doc touch; not worth its own dispatch.

### Answers to the three questions

**Q1 — is `MAX_CONFIG_LEN` a hard limit or a soft bound? Soft. Keep option (a); the author's
deviation is right.** Three lines of evidence. (i) `schemas.py:82-90` declares the whole block as
"the RAM guard (rule 6)". (ii) The *one* documented hard boundary in this engine is 4096 bytes for
a **UNIQUE-constrained** property (K-049, full-process SIGSEGV) — and `MAX_CONFIG_LEN` is 8000,
already nearly double it, which by itself proves the constant does not track an engine limit.
Verified live that it cannot apply here anyway: `db.constraints()` on `ws:acme` returns exactly
`WorkflowRun.runId` and `Step.stepUid`, both UNIQUE — **`ctx` and `config` are not constrained**.
(iii) `docs/archive/plans/m3-executor.md:288` resolved 8000 on fit-for-purpose grounds ("comfortably
fits a node system prompt; no bump, no RAM change"), not on a limit.

So a 46-character overrun is inside the noise of a sanity bound, and option (a) buys something
real that (b) and (c) cannot: **one call site, ahead of both paths and ahead of the `get_message`
read**, which keeps the F-2 guard co-located with `_reject_reserved_keys` and preserves the "both
paths screened before anything is read" property Pass 1 credited. Option (b) would split the
bound across two self-contained branches — against the §4 first/subsequent doctrine the docstring
cites — for a 46-character gain. Option (c) buys exactness by making the caller's usable budget
depend on a value the caller cannot see. **Keep (a); fix only the comment's number (N-2).**

**Q2 — is the sweep's bound still meaningfully covered? Yes, and the boundary is now tighter than
before; it has not been narrowed into vacuity.** Mutating the sweep's own bound to always-false
(`services.py:2505`, copy-aside, restored byte-identically) turns the resized test **red** — so
the test still proves the sweep faults an over-bound candidate. The construction is now *on* the
boundary rather than inside it: ctx = exactly 8000 (largest startable), merged = 8020 (smallest
over), and the two new asserts pin both numbers so it cannot silently drift off. The one thing it
does not catch is the sweep's *own* off-by-one (`>`→`>=` passes, verified) — but that was equally
uncovered before: the old seed serialized to 8011 and merged to 8031, where `>` and `>=` are
indistinguishable. **No regression; a strict improvement.** The residual cost is N-3.

**Q3 — is `_chat_start_ctx` a sound seam, and is the invariant test non-vacuous? Yes to both.**
The seam is not test-only scaffolding: it names a real domain concept (the initial chat ctx), it
is the single writer of the anchor, and it is what lets `CHAT_START_ANCHOR_KEYS` stand for the
function in the invariant. The three tests close the loop by construction, and I verified each
against its own mutation (Appendix D): drift the seam without the constant → the *constant-honesty*
test fails; grow both with a non-reserved key → the *invariant* test fails; reverse the merge →
the *ordering* test fails, and only that one. There is no route to a non-reserved anchor key that
leaves the suite green. The docstring's justification — "a guard that makes the behaviour below it
unreachable also makes it untestable from outside, so the seam is tested from inside" — is exactly
the right reasoning, and it is now demonstrably true rather than asserted.

**On the disclosed M11 gap: confirmed corrected.** `test_a_run_ctx_exactly_on_the_bound_is_accepted_on_both_paths`
sits *on* the bound — `json.dumps({"blob": "y" * (MAX_CONFIG_LEN - 11)}, separators=(",",":"))` is
exactly 8000 characters, and the test **asserts that equality inline** before exercising either
path, so it cannot drift off the boundary later without failing. The `>`→`>=` mutation now fails
it directly rather than incidentally in another file. Disclosing that gap rather than quietly
re-running was the right call and materially improved the test.

### What's solid (Pass 2)

The fixes are better than the findings asked for. F-3 was a request for one assertion; it came
back as a named constant, a documented seam and three tests that close every route. F-8 was a
sketch; the delivered check names the drifted fields, prints the only valid remedy, and fires on
the real `ws:acme` artifact unprompted. The plan's own v1.4 catch — S4's done-condition would have
run `verify_salesperson.sh` bare and asserted against `ws:acme` instead of the graph under test —
is a defect neither of my passes found, and it was the more dangerous one.

---

## Pass 3 — 2026-09-02 (S4: repository + service primitives, the two resets)

**Reviewed:** the S4 slice of the same uncommitted working tree — `falkorchat/repository.py`
(+636), the insert-only `§18 Storefront order reads` block in `falkorchat/services.py`,
`falkor-chat/docs/QUERIES.md` §18 (+598, append), `falkor-chat/docs/DESIGN.md` §5.1 (+6),
`falkor-chat/docs/HISTORY.md`'s S4 entry, `tests/test_repository.py` (+927) and the four new
`tests/test_services.py` cases — against `docs/plans/salesperson-ui.md` §5.1 row **S4** and the
approved graph note `docs/plans/salesperson-ui-graph.md` **v1.2** §3/§4/§5/§10/§12. The
`start_workflow_run` hunks in `services.py` and every other modified file are S1b/S2b's and were
**not** re-reviewed. Baseline: `git diff` against `4bb96e1`. **Fresh reviewer** — Pass 1/2
findings are not restated.

**Verdict: approve with suggestions.** No blockers. **One major** (M-1, a settable `threadId`
that contradicts its own method's stated scope rationale), six minors, two nits. The safety
argument is the strongest I have gated in this repo: I attacked the two guards from seven
directions and could not break either one.

**CPG: considered, not relevant — `cpg_falkorchat` was built at `4bb96e1` and contains none of
the S4 additions; the questions this pass turns on (does a guard hold at runtime, does a test
assert by identity) are answered by execution against the live graph, not by call-graph
queries.**

### What I verified by execution vs. judged statically (Pass 3)

| Verified by running it | Judged statically |
|---|---|
| Full suite **2379 passed, 14 deselected** in 17.2 s (matches `teco`; baseline 2336 + 43 new nodes, and 39 + 4 is what the diff contains) | Docstring/comment accuracy across the nine methods |
| **Verbatim, independently:** all five note blocks byte-identical to the shipped constants — *and* `QUERIES.md` §18 transcribes the same five byte-identically (Appendix F) | Whether `QUERIES.md` §18 covers everything note §12 mandates (it does — Q4) |
| **7 guard ablations**, in-process, source never edited (Appendix G): every one reddens, none is decorative | Proportionality of the nine-method surface |
| Note §2.3 **row B reproduced in S4's own fixture**: both guards off, `reset_participant('u2')` destroys `demo-welcome` + dm1/dm2/dm3 + 2 cursors while `Channel` **3→3**, `Thread` **3→3** and every survivor label is still present (Appendix G) | |
| Five **untested paths** probed directly — duplicate marker, marker forging, unscoped-way-2 under `reset_all`, cross-member cursor under `reset_all`, empty graph (Appendix G) | |
| The plan's own done-condition, verbatim: `./scripts/verify_salesperson.sh test` after `reset_all_participants` → **exit 0** | |
| `Channel.participantId` and `User.tokenHash` writer sets, by grep over the whole tree | |

### M-1 · **Major** · `set_participant_record` can repoint a participant's `threadId` at another participant's thread

`repository.py:3532-3546` accepts `thread_id` and writes
`u.threadId = coalesce($threadId, u.threadId)` with no check that the thread lies inside the
participant's own channel. **Verified by execution:** on the shipped fixture,
`set_participant_record("test", participant_id="p-aaa", thread_id="th-p-bbb")` returns
`{"threadId": "th-p-bbb"}` and the graph agrees. `User.threadId` is the server-resolved scope
denorm the storefront reads from (plan §4.3), so this is the one cross-participant lever in the
whole S4 surface — and it lands squarely on AC-2.

The method's own docstring rules `channelId` out for exactly this reason ("the channel a
participant owns is decided once, by `ensure_participant`, together with the marker G2 reads"),
and `QUERIES.md` §18.3 repeats it — but `threadId` is settable and neither text mentions it. The
argument applies unchanged: the thread a participant reads is decided by `ensure_participant` and
re-decided by `reset_participant`, both **in-query**. Today the parameter has **no caller and no
test** (no test in the diff passes `thread_id=`), so it is pure unexercised surface waiting for
S6/S7.

**Suggested fix (pick one):** drop the `thread_id` parameter and its `coalesce` line — nothing
needs it; or, if S7 is expected to, gate it in-query with
`OPTIONAL MATCH (:Channel {channelId: u.channelId})-[:HAS_THREAD]->(t:Thread {threadId: $threadId})`
and set only `WHEN $threadId IS NULL OR t IS NOT NULL`, with a test for the refusal.

### M-2 · **Minor** · the `reset_all` half of note P2's "one deliberate asymmetry" is untested

`reset_participant` **keeps** a participant's cursor on a surviving non-participant thread;
`reset_all` **deletes** it (the `User` goes, so the cursor would be unowned). The first half has a
dedicated test (`test_reset_participant_keeps_a_cursor_on_a_surviving_thread`). The second half
has none: no fixture seeds `p-ccc:demo-welcome` before a `reset_all`.

**Proved by mutation:** replacing `reset_all`'s wide `HAS_CURSOR` sweep with
`reset_participant`'s liveness-filtered form leaves **506 passed, 0 failed** across
`test_repository.py` + `test_services.py` (Appendix G). `test_reset_all_leaves_no_unowned_read_cursor`
only covers the *dead*-thread class (`th-gone`); the *live*-surviving-thread class has no witness.
The shipped behaviour is correct — confirmed directly: `p-ccc:demo-welcome` deleted,
`u1:demo-welcome`/`assistant:demo-welcome` kept, `cursorCount: 7`, zero unowned cursors after.

**Suggested fix:** two lines — add `repo.advance_cursor(..., cursor_id="p-ccc:demo-welcome", ...)`
to `test_reset_all_leaves_the_non_participant_subgraph_intact`'s setup and assert
`repo.get_cursor("test", cursor_id="p-ccc:demo-welcome") is None` after. That also gives
`test_reset_all_leaves_no_unowned_read_cursor` teeth for the live-thread class.

### M-3 · **Minor** · the duplicate-marker fail-safe is a §12 response contract with no test

Note §12 makes it one: *"Either raises `unique constraint violation on node of type Thread` →
propagate as a `5xx`; do not retry."* `reset_participant`'s docstring (`repository.py:3617-3624`)
promises both halves — it raises **and writes nothing**. Nothing asserts it.

**Verified by execution** that the promise holds: with a second `Channel {participantId:'p-aaa'}`
and a `MEMBER_OF` edge into it, `reset_participant('p-aaa')` raises
`ResponseError: unique constraint violation on node of type Thread`, node count **57 → 57**, the
old thread and its 3 messages intact, `th-X` absent. And `reset_all` still collects the
participant cleanly (`ch-dupe` gone, demo subgraph intact).

**Suggested fix:** one test that plants the duplicate marker, asserts `pytest.raises`, and asserts
node count + `User.threadId` unchanged. It is the only §12 contract row with no executable
witness, and it is the row whose *silent* failure mode is worst.

### M-4 · **Minor** · the new test plants a `Product` in the shared global `reference` graph and never removes it

`test_reset_all_never_touches_the_reference_graph` writes
`(:Product {productId:'prod1', name:'Widget', …})` into `reference` and has no teardown. The
`wf_repo` fixture wipes `reference` at *setup*, so the suite is self-consistent — but whichever
run ends on that test leaves the stray behind, and `Product` is the one label
**`verify_catalog.sh` counts**.

**Verified by execution:** after a clean suite run I re-seeded and ran the unit's own
done-condition — `./scripts/verify_catalog.sh` → `Product count : 16 (expected 15) — MISMATCH`,
**exit 1**; the extra node is `prod1 / Widget / Misc`. `./scripts/verify_salesperson.sh test`
after `reset_all_participants` exits **0**, so only the catalog half of the S4 done-condition is
affected, and it is affected by S4's own test rather than by S4's code.

**Suggested fix:** end the test with
`db.reference_graph(conn).query("MATCH (p:Product {productId:'prod1'}) DETACH DELETE p")`, or
prove non-interference with a label `verify_catalog.sh` does not count. **Live state, for
`teco`:** `reference` currently holds that stray (16 products); removing it restores
`verify_catalog.sh` to exit 0. I did not delete it — a write to shared global state is not mine to
make, and the deletion was blocked by the permission layer when I attempted it.

### M-5 · **Minor** · `Channel.participantId`'s "no index, no constraint" decision is absent from `DESIGN.md` §7.1

§7.1 is the component's index/constraint surface — the page a DBA reads before adding one. It
already carries the precedent this needs verbatim: `` > `Message.threadId` is **deliberately
unindexed** (§5.1) — nav metadata, not an anchor. `` `Channel.participantId` is a *load-bearing
predicate in a destructive query* that is deliberately unindexed and deliberately
un-constrained, and §7.1 does not mention it at all.

The decision itself is well documented — `QUERIES.md:3003-3007` states the property, the no-DDL
call and the declined `UNIQUE` — but note §9 also records a **reversal condition** (the constraint
was rejected "on scope, not on safety", after the gate review verified FalkorDB exempts
absent/`null` properties). Per root `AGENTS.md`, a rejected option with a reversal trigger is a
live constraint on the system and belongs on the owning design surface, not only in a plan
document that will eventually be archived.

**Suggested fix:** one `>`-note under §7.1's tables mirroring the `Message.threadId` line, plus
the reversal condition in a clause. Two lines.

### M-6 · **Minor** · the plan's §5.1 S4 row names eight methods; nine are required — the row is the defect

Full reasoning under Q1 below. `ensure_participant` is not optional scope: it is the **only**
writer of `Channel.participantId` anywhere in the tree (verified by grep — `create_channel` writes
a fixed three-property map, `set_participant_record` cannot touch `channelId`), so without it G2
never resolves for anyone and **both resets are permanent no-ops** returning `scoped=false` for
every participant. Note §12 hands it to S4 explicitly, twice.

**Suggested fix:** `architect` folds a v1.6 one-line correction into
`docs/plans/salesperson-ui.md` §5.1's S4 Interfaces column ("the nine repository methods…"), so a
later reader does not gate against a list that would have shipped an inert feature.

### M-7 · **Minor** · the §18 collision — the *pre-existing* header is the mis-numbered one

`repository.py:3761` and `services.py:2963` carry `# ── §18 Structured natural-language query
generation (K-055 M6) ──`. Every other section header in both files numbers itself by its
**QUERIES.md** section (§3 Channels, §4 Messages, §16 Cart/Order, §17 profile — all match), and
`QUERIES.md` had no §18 when that header was written; K-055's `run_readonly_query` is, by its own
comment, "the **only** repository method … that takes compiler-produced Cypher rather than a query
1:1-mapped to `QUERIES.md`", so it has no QUERIES.md section and never will.

So S4's `§18 Storefront participants & resets (QUERIES.md §18, …)` is the **correct** one under
the convention, and the K-055 header is a claim on a section number that was never used. **Do not
relabel S4's.**

**Suggested fix, and whose:** drop the number from the older header in both files —
`# ── Structured NL query generation (K-055 M6) — no QUERIES.md section: compiler-produced Cypher ──`.
It is a one-line comment edit in two files S4 already owns; `teco`'s call whether to fold it into
S4's pass or hand it to `coder` standalone. Leaving it is survivable, but it is exactly the kind
of ambiguity that costs a future reader ten minutes.

### N-6 · **Nit** · two test comments now assert something false

`tests/test_repository.py:28` — `_add_to_channel`'s docstring says "no repository method exists yet
— QUERIES.md §2 'Add user to channel' is a documented, verified query but not yet wrapped by a
repository method". S4 shipped that wrapper (`add_channel_member`). The same claim echoes at
`:2797`. Both are in S4's own file. (`tests/test_api.py:1554` repeats it and is *not* S4's file —
route separately or leave it.) `_add_to_channel` still has five callers, so it is not dead — only
its rationale is.

### N-7 · **Nit** · `QUERIES.md` §18.0 names the wrong test as the catcher

§18.0's ablation paragraph describes `reset_participant('u2')` with both guards removed and then
says *"`server/tests/test_repository.py::test_reset_all_leaves_the_non_participant_subgraph_intact`
is that assertion"*. That test guards **`reset_all`**. **Verified by ablation:** stripping G1+G2
from `_RESET_PARTICIPANT_CYPHER` reddens `test_reset_participant_is_a_no_op_for_non_participants[u1]`
and `[u2]`, the two cross-member/cursor tests and the mismatched-marker test — and leaves
`test_reset_all_leaves_the_non_participant_subgraph_intact` **green**. Cite the `[u2]` case, or
name both. One-line fix; it matters because §18.0 is where a future editor goes to learn which
test protects which guard.

### Answers to the seven questions

**1 — the ninth method. The author is right, and the plan's eight-method list was the defect.**
Three independent reasons, all checkable: (a) note §12, the *approved* hand-off, says "Implement
§3, §4, §5, §10.1 and §10.2 verbatim" and §3 **is** `ensure_participant`, then names it again in
the `.query()` routing bullet — S4's mandate is the note, and the note is unambiguous; (b) it is
the only writer of `Channel.participantId` in the tree, so without it G2 never resolves and both
resets are inert — not a style point, the feature simply would not exist; (c) plan §5.0 assigns
`repository.py` to **S4 alone**, and S6 (`storefront.py`/`config.py`/`test_storefront.py`) cannot
add it without breaking that map. The atomicity argument is also sound and I would keep it even if
(a)–(c) did not apply: decomposing the join needs a *new* marker-writing method (widening the
forging surface note §1.1's residual argument closes) and opens a crash window producing the
unscoped state whose "dead branch" status note §5/F2's whole trade-off rests on — it would fire
the note's own reversal trigger. Ship the nine; correct the plan (M-6).

**2 — the guards, audited hardest. They hold, including on paths no test reaches.** Both G1 and G2
are present in both destructive queries and nowhere is either weakened; the three participant
reads/writes carry G1 (`get_participant_record`, `set_participant_record`, `list_participants`);
the non-destructive methods need neither. Seven ablations all redden (Appendix G) — no guard is
decorative. The provenance premise is real: `Channel.participantId` has exactly **one** writer
tree-wide and `User.tokenHash` exactly two, the second of which (`set_participant_record`) is
itself G1-gated so it cannot *create* a participant. Five paths the tests never reach, probed
directly and all conforming to the note: duplicate marker → raises, **57→57 nodes, nothing
written**; `ensure_participant` handed `channel_id='demo-general'` → raises on `Channel` UNIQUE,
**nothing written, `demo-general.participantId` still `null`** (the marker cannot be forged even
by a caller choosing the id — a stronger result than the note claims); unscoped-way-2 under
`reset_all` → counted, left whole; completely empty graph → all four status-row contracts hold, no
`result_set[0]` IndexError. The one thing I did **not** find is a guard hole. M-1 is the sole
isolation defect in the unit and it is not in the resets — it is in a setter nobody calls yet.

**3 — the survivor assertions are by identity, everywhere it matters, and the claim reproduces.**
`_assert_demo_subgraph_intact` names `demo-general`, `demo-welcome`, `dm1`/`dm2`/`dm3`, the
HEAD/NEXT/TAIL chain, each message's author, `u1`, `u2`, `assistant`; `_assert_common_survivors`
names `doc1`, `chunk1`, `ent1`, the `triage` snapshot, `u1`'s
`Customer`/`Cart`/`CartItem`/`Order`/`OrderLine`, `orphan-1`, and `WorkspaceConfig` by its
`agentModelOverride` **value**. The one count-shaped check (`MATCH (s:Step) RETURN count(s)`) hangs
off a snapshot already asserted by identity. `test_reset_all_keeps_exactly_the_survivor_labels` is
the label check the plan asks for and its docstring says outright it is documentation, not the
safety net. And S0's ablation table reproduces **in this fixture**: guards off,
`reset_participant('u2')` → `Channel` 3→3, `Thread` 3→3, every §4.8 survivor label present,
`demo-welcome` and dm1–dm3 **gone** (Appendix G). `Message` 10→7 and `ReadCursor` 6→4 do move, so
a strict *count-of-every-label* check would catch it — but the plan's literal wording is "asserted
by label", and label-presence passes. The author's `HISTORY.md` wording ("the `Channel` and
`Thread` label counts unchanged (3 → 3)") is precisely right, not overstated.

**4 — the two out-of-column edits. One of them is not out of column, and a third is missing from
the count.** `QUERIES.md` **is** in §5.1's S4 Files column, so the §18 append is squarely in scope
(§5.0 even pre-clears it as "an append, not a merge"). `DESIGN.md` is genuinely outside — and
correct: note §9 and §12 both mandate it by name, the edit is exactly the arrow-notation line plus
a four-line gloss, and the property list matches `create_channel`'s actual three-property map plus
the new marker. `HISTORY.md` is also outside §5.1's Files column but mandated by §5.0's "one entry
per delivered step"; the entry is present, accurate, and its arithmetic (39 + 4; 2336 → 2379)
checks out against the suite. **What should have been updated and was not:** `DESIGN.md` §7.1
(M-5). I checked and found nothing owed to `SERVER.md` (it carries no method inventory),
`bootstrap_schema.sh` (no DDL, correctly), `BACKLOG.md`, or root `docs/HISTORY.md` (S16's).

**5 — the §18 collision: relabel the *older* header, not S4's.** See M-7. The file convention is
"§N = QUERIES.md §N", which S4 satisfies and K-055 never did — K-055's method is explicitly *not*
QUERIES.md-mapped. Owner: a two-file, one-line-each comment edit in files S4 already holds, so
folding it into S4 is cheapest; otherwise `coder`. It should not go to `graph-dba` — nothing about
it is a graph question.

**6 — no index and no constraint are the right calls, and index-before-constraint is not being
skipped.** Nothing is skipped because **no DDL is added at all** — `bootstrap_schema.sh` is
untouched and every existing workspace stays valid. On the index: `participantId` is only ever a
`Filter` on a `ch` already bound by a `MEMBER_OF` traversal from an index-anchored `u`, so an index
has nothing to anchor; at ~50 participants the whole `reset_all` plan visits 52 `User` records and
one 102-row `ReadCursor` scan. On the constraint: `Channel.channelId` already carries index **and**
UNIQUE (`bootstrap_schema.sh:123` index, `:247` constraint — ordering correct), and that constraint
is what makes marker-forging fail closed, as I confirmed above. A `UNIQUE` on `participantId` would
add nothing note §4's row-multiplication fail-safe does not already do loudly. **One caveat:** the
reason it can be declined is itself a fact worth keeping on the design surface — that is M-5, not
a disagreement with the call.

**7 — the mutation seam is sound, and I re-ran it independently.** Patching
`Repository._RESET_*_CYPHER` on the class from a pytest plugin at import time is a legitimate seam:
the constants are read per call through `self.`, so the patch takes effect for every test in the
session, and the file is never touched — `md5sum falkorchat/repository.py` is unchanged before and
after all eight of my runs, and `git status` shows no new or changed file. I did not reuse the
author's plugin; I wrote my own (Appendix G) and got equivalent reddening on all five of their
ablations plus two more. The ablations do cover the guards' failure modes: G1-off and G2-off,
independently and together, on **both** resets (the author's table covers G2-off-all, G1+G2-off,
G2-off-mine, G1-off and the author-scoping mutation; I added G1-off-all and the cursor-sweep
narrowing that found M-2). **The seam's one limit, worth stating:** it mutates *query text* only,
so it cannot exercise the Python-side branching — `ensure_participant`'s three
`MemberIdCollisionError` paths, `reset_participant`'s `None`-vs-`scoped` dispatch,
`get_customer_current_order`'s placeholder filter. Those have direct tests, so the gap is covered;
it just is not covered *by the mutation battery*, and the `HISTORY.md` entry reads as though the
battery is the whole evidence.

### What's solid (Pass 3)

- **The verbatim discipline is real and now double-checked.** Five blocks byte-identical in
  `repository.py` **and** in `QUERIES.md` §18 — 9 fenced blocks in §18, longest 77 lines, so the
  note's own "a three-figure-line block is self-evidently wrong" tripwire passes on both files.
- **The guards are the strongest thing here.** Seven ablations, five untested-path probes and a
  tree-wide writer census all agree with the note. The forging path is closed *harder* than the
  note claims: `Channel.channelId`'s UNIQUE constraint makes even a caller-chosen `channel_id` fail
  closed with nothing written.
- **The fixture is the note's own probe graph, faithfully.** Adversarial `u2`, a cross-member
  participant, both unscoped shapes, the off-chain message, `WorkspaceConfig` asserted by value,
  the GraphRAG boundary edge — and every survivor named.
- **The docstrings carry the *reasons*, not the mechanics** — why G2 is provenance, why the two
  cursor sweeps differ, why the duplicate marker is a fail-safe rather than a defect, what the
  caller must do with `scoped=false`. A rare case where a long docstring earns its length.
- `.query`/`.ro_query` split exactly as note §12 routes it; no Cypher escaped `repository.py`; the
  two service wrappers are thin and `ctx.actor`-scoped.

### Open questions (Pass 3)

1. **M-1's disposition is `teco`'s, not mine.** Dropping `thread_id` is the clean call *if* S7
   genuinely never needs it. If S7's "reset mine" wrapper is expected to repoint the thread from
   the service layer rather than rely on `reset_participant`'s in-query `SET`, the gated form is
   the right fix instead. `architect` can answer this from §5.1's S7 row faster than I can infer it.
2. **`reference` is left holding a stray `Product {productId:'prod1'}` (16, expected 15)** — see
   M-4. I could not remove it (blocked by the permission layer). Someone with write authority
   should, or re-seed `reference` from scratch, before anyone reads `verify_catalog.sh`'s exit code
   as a signal.
3. Unrelated to S4, flagged only because it changed under me: the two untracked
   `docs/plans/small-model-benchmarking*.md` files present at the start of this session are gone
   from the working tree now. Not mine; worth a glance if they were expected to survive.

---

## Pass 4 — 2026-09-02 (re-gate of S4b)

**Reviewed:** the same working tree after S4b actioned all nine Pass 3 findings — `repository.py`,
`tests/test_repository.py`, `QUERIES.md` §18.0/§18.3, `DESIGN.md` §7.1, `HISTORY.md`'s S4 entry, and
the two relabelled section headers in `repository.py`/`services.py`. Baseline unchanged. Compact by
rule: dispositions first, new findings in full.

**Verdict: approve.** All nine findings are correctly disposed of, six of them mutation- or
execution-proven by me rather than accepted. **One new nit** (N-8). Both flagged items ruled on
below, and I agree with the implementer on both.

**CPG: considered, not relevant — `cpg_falkorchat` is still at `4bb96e1` and contains none of the
S4/S4b additions; every question this pass turns on was settled by ablation and by running the
suite.**

**Verified by execution this pass:** full suite **2381 passed / 14 deselected** (serial run —
see the process note below); **10 ablations** on the current tree; the corrected §18.0 citation
table re-measured independently; the verbatim discipline re-checked in both directions after the
§18.3 edit; the S4 done-condition run end to end; the four Pass-3 guard probes re-run for
regression.

### Pass 3 dispositions

| # | Disposition | Evidence I rechecked |
|---|---|---|
| **M-1** Major | **Fixed by removal** — `thread_id` gone from the signature and from the Cypher's `SET`; docstring now rules out `channelId` **and** `threadId` together and names where a containment check would have to live | Signature is `participant_id`/`display_name`/`token_hash`/`language`; `QUERIES.md` §18.3's block and its `// …` param comment both dropped `$threadId` (`threadId` correctly stays in the `RETURN` projection — read-back, not settable). **No production caller of `set_participant_record` exists tree-wide** — every hit is a test, a doc, or this review. Ruling on the downstream half below |
| **M-2** Minor | **Fixed** — `p-ccc:demo-welcome` seeded into `…_leaves_the_non_participant_subgraph_intact`, with the delete asserted plus the two surviving members' cursors and the unowned-cursor sweep | My Pass 3 mutation now bites: narrowing `reset_all`'s sweep to the liveness-filtered form goes **0 → 1 failed** (that exact test), 507 passed |
| **M-3** Minor | **Fixed** — `test_reset_participant_with_a_duplicate_marker_raises_and_writes_nothing` pins raise + node-count + `User.threadId` + survivors | Re-ran the mutation that *removes* the raise (re-mint id made unique per channel, so the delete proceeds under a duplicate marker): the new test reddens. It has teeth against the behaviour it guards, not just against a constraint that would fire anyway |
| **M-4** Minor | **Fixed** — `try/finally` teardown, keeping `Product` as the label under test. I prefer this to my own alternate-label suggestion: it removes the pollution *and* keeps the assertion on the label that matters, and `finally` covers the failure path my version would not have | After a clean suite run, `reference` held **0** `Product` nodes; re-seeding gave exactly **15**, and `./scripts/verify_catalog.sh` → **exit 0**. First time this half of the done-condition has passed in the chain |
| **M-5** Minor | **Fixed, better than specified** — §7.1 gains a `>`-note mirroring the `Message.threadId` precedent, carrying both the no-index and no-`UNIQUE` reasoning **and** an explicit reversal condition | Read `DESIGN.md` §7.1; the reversal condition ("a second writer of the marker … or §18.4's raise stops being acceptable") is the durable half and it is there |
| **M-6** Minor | **Fixed** — plan §5.1's S4 row now reads "**Nine methods, not eight**", separates the five verbatim-from-note queries from the four this plan specifies, and states the inertness consequence | Read the revised row in `docs/plans/salesperson-ui.md` |
| **M-7** Minor | **Fixed as specified** — the *older* header relabelled in both files; S4's untouched | `repository.py:3768` and `services.py:2963` now read `# ── Structured NL query generation (K-055 M6) — no QUERIES.md section ──` |
| **N-6** Nit | **Fixed** — docstring rewritten to say the wrapper now exists and the helper is kept as a fixture tool. See **N-8** for one residual error | Read `tests/test_repository.py:26-34` and `:2793-2800` |
| **N-7** Nit | **Fixed, and my framing was the weaker one** — §18.0 now carries a measured two-row table plus the "not interchangeable, cite the one that matches the query you are changing" rule, which is better than the one-name correction I suggested | Re-measured independently; see ruling 1 |

### N-8 · **Nit** · the rewritten `_add_to_channel` docstring says "five callers"; there are four

`tests/test_repository.py:28-29` — "still has five callers that seed fixtures older than §18".
`grep -n "_add_to_channel(" tests/test_repository.py` returns **four** call sites: `:1554`, `:1555`,
`:1569`, `:1580`. (`tests/test_api.py` has its own separate local copy of the helper, not a caller
of this one.) The whole point of the N-6 rewrite was to stop this docstring asserting something
false, so it is worth the one-character fix — or drop the count, which is the part that will rot
again on the next fixture change.

### The two flagged items

**1 — the §18.0 citation table: I re-measured it my own way and it is exactly right.** Third
independent measurement, on the current tree, with **no `-k` filter** and both test files
collected (my Pass 3 runs used a `-k` filter, which is the methodological difference worth naming;
re-running the single-guard ablations unfiltered reproduced my Pass 3 counts exactly, so the filter
was not the source of any error). Stripping G1+G2 from `_RESET_PARTICIPANT_CYPHER` → **5 failed,
503 passed**; from `_RESET_ALL_PARTICIPANTS_CYPHER` → **5 failed, 503 passed**; the two sets are
disjoint and both match the shipped table **name for name, with no extras and none missing**
(Appendix H). We do not disagree a third time — the implementer's measurement is correct and mine
now agrees with it. The added "not interchangeable, cite the one that matches the query you are
changing" sentence is the part that stops the error recurring, and it is a better fix than the one
I asked for.

**2 — editing `HISTORY.md` outside the file list was right, and the corrected entry is accurate.**
The alternative was worse in every direction: the entry describes the *uncommitted* diff S4b had
just changed, so leaving it would have shipped four false statements into the permanent record, and
adding a *second* dated entry for a step that has not been delivered once would have violated
`AGENTS.md`'s "an open item is rewritten, not appended to" and left `HISTORY.md` narrating a fix
pass rather than a delivered change. Correcting in place is the convention's own answer. §5.0's map
already puts `HISTORY.md` outside §5.1's Files column and mandates one entry per step, so the file
was S4's to write in the first place — this is a correction to its own entry, not an incursion.
I checked all four corrections and the entry now reads true: "neither `channelId` nor `threadId`"
matches the shipped signature; **41 + 4 = 45** new nodes and 2336 + 45 = **2381** matches the suite
I ran; and the Docs paragraph now names §7.1 and the header relabel. One residual to watch, not a
finding: the entry's own mutation paragraph is now the longest unverified claim in it — I confirmed
five of its seven ablations directly.

### The discarded `CREATE`→`MERGE` ablation — no gap, and reporting it was the right call

**Measured: 508 passed, 0 failed.** The implementer's reasoning is correct and I confirmed the
mechanism: `MERGE` cannot match a `Thread` hung off the *other* channel, so it creates a second one
with the same `$newThreadId` and `Thread.threadId`'s UNIQUE constraint fires identically. That
makes it an **equivalent mutant** — it changes no observable behaviour on this path — and a
surviving equivalent mutant is a property of the mutation, not a hole in the test. The test's teeth
are demonstrated by the *non*-equivalent mutation instead: making the re-minted id unique per
channel removes the raise and lets the delete proceed under a duplicate marker, and
`test_reset_participant_with_a_duplicate_marker_raises_and_writes_nothing` reddens (Appendix H).
Reporting a discarded ablation rather than quietly dropping it is exactly the posture that makes a
mutation table worth reading.

### Verbatim discipline, re-checked after the §18.3 edit

Re-ran Pass 3's Appendix F check against the current tree: **9 fenced blocks in §18, max 77 lines**,
and all five class constants byte-identical to their §18 transcription *and* to the note's own
fenced blocks. The note is now **v1.3** (prose only, no Cypher touched), so "verbatim" is verbatim
against the current note, not a stale one. §18.3 sits next to those blocks and the edit did not
disturb them.

### Process note for `teco` (not a finding against S4)

These repository tests are integration tests against a **single shared `ws:test` graph**, so two
`pytest` processes cannot run concurrently — they wipe each other's fixtures mid-test. I produced a
spurious **"43 failed"** that way by running an ablation battery while a full-suite run was still in
flight; re-run serially, the same tree is **2381 passed / 14 deselected**. Worth knowing before
anyone parallelises verification, or reads a surprising red run as a real regression.

### Open questions (Pass 4)

None blocking. `ws:test` is left holding the post-`reset_all` fixture plus both re-materialized
defs; `reference` holds the clean 15-product catalog and both defs — I re-seeded both (the suite
run had emptied `reference`), and **the Pass 3 stray `prod1` is gone**, removed by M-4's own
teardown rather than by me. I created no graph key and deleted none.

---

## Pass 5 — 2026-09-02 (S3: the two wiring switches)

**Reviewed:** the uncommitted working-tree delivery of `docs/plans/salesperson-ui.md` (v1.15)
§5.1 step **S3** — `config.py`, `app.py`, `tests/test_app.py`, `docs/SERVER.md`,
`docs/HISTORY.md` — against that row's scope and done-condition and against §4.9 / §4.3 part 4.
Baseline: `git diff` vs `HEAD` (`5a5a257`). Not reviewed: S1/S2/S4 (Passes 1–4), and anything S8
or S16 owns.

**Verdict: approve with suggestions.** 0 blockers · 1 major (carry-forward, does **not** gate S3)
· 2 minor · 2 nits. The done-condition is met on evidence I re-derived independently: the
flattening helper is correct for every registration shape `create_app` actually uses, its positive
control genuinely closes the vacuity mode, `dev_surface` dominates `mount_mcp` across all four
`/mcp` seams, `/health` is exactly one route in both configurations, all three
`_build_default_app` return paths are independently covered, and my own nine-mutation battery
reproduced nine kills (the implementer's seven, plus two more isolating a partial surface).

CPG: considered, not relevant — `cpg_falkorchat` is loaded and reachable (285,547 nodes), but S3's
question is "what does this FastAPI app object register at *runtime*", a construction-time property
of `_IncludedRouter`/`Mount` instances rather than a static call/AST fact; I answered it by building
the real apps in the venv instead.

### P5-1 · **Major** · `_route_paths` reports **pre-prefix** paths — the helper S8 inherits cannot see a router's mount prefix

`tests/test_app.py:_route_paths` recurses through `route.original_router.routes` and appends each
route's raw `.path`. FastAPI 0.139 keeps the `prefix=` passed to `include_router` on the
`_IncludedRouter` wrapper (`route.include_context.prefix`), **not** on the inner routes — so the
helper reports a prefixed router by its bare paths. Run on an S8-shaped app (Appendix I, §3):

```
helper says: ['/health', '/join', '/orders/{oid}']
truth      : ['/health', '/shop/api/join', '/shop/api/orders/{oid}']
```

Nothing in S3 is weakened by this — S3's app has no prefixed include, and the exact-list assertion
still *counts* every included route, so no S3 assertion is vacuous. It gates **S8**: §4.9 and the
S5.1 S8 row hand this same helper the job of asserting the storefront route table, whose entire
content is a router at `/shop/api` and a mount at `/shop`. As written, an S8 assertion would pass
identically whether that router were mounted at `/shop/api`, at `/`, or at `/admin` — the seventh
instance of this build's recurring "green while asserting nothing" shape, pre-planted.

**Suggested improvement** (verified working, Appendix I §4 — reproduces all four S3 assertions
unchanged and gets nested prefixes right): thread an accumulator through the walk —
`walk(inner.routes, prefix + getattr(getattr(route, "include_context", None), "prefix", ""))`,
appending `prefix + path`. Owner: S8's implementer, as an S8 pre-condition; or `tdd-engineer` now.

### P5-2 · **Minor** · the helper silently drops any route object it cannot classify

The walk's final clause is `if path is not None: found.append(path)` — a route exposing neither
`original_router` nor `.path` vanishes without trace. `starlette.routing.Host` is exactly that
shape, and I confirmed it: a `Host` carrying a whole sub-app of routes is reported as `[]`
(Appendix I §2). No such route exists in `create_app` today, and the positive control would not
catch one, because the control only asserts the routes it was written against.

**Suggested improvement:** make the unclassifiable case loud rather than silent — `else: raise
AssertionError(f"route-table helper cannot classify {route!r}")` in place of the silent skip. One
line, and it converts a future blind spot from a passing test into a failing one.

### P5-3 · **Minor** · `falkor-chat/docs/SERVER.md` appears in **no row** of the plan's §5.0 shared-file map, and S8/S9 will falsify §1.3 again

The implementer's call to update `SERVER.md` was right: §1.3 documented the auth/tenancy seam and
§2.1 sketched `create_app`'s two mounting lines, and S3 falsified both. Leaving it to S16 would
have shipped a doc that describes an app shape the code no longer has. But the map is the
mechanism that keeps that from becoming a merge collision, and `SERVER.md` is absent from it —
`grep -n 'SERVER.md' docs/plans/salesperson-ui.md` returns five prose citations and zero Files-column
assignments. S8 adds `storefront`/`storefront_dir` to the same signature and S9 edits `app.py`
again; both will need the same §1.3/§2.1 edits.

**Suggested improvement** (`architect`, on the plan): add the row
`` `falkor-chat/docs/SERVER.md` | S3 (**delivered**), S8, S9 | **S3 → S8 → S9** `` — already
satisfied by the existing `app.py` ordering, so it changes no sequencing. This is the third
§5.0 map gap this review has found (F-4, M-6, now P5-3); the map is regenerated from §5.1, and
§5.1's Files columns are what omit doc files.

### P5-4 · **Nit** · the two `/health` routes diverge when the *context provider itself* raises

`app.py`'s bare route calls `provider()` **inside** the `try`, so a raising context provider yields
503. `api.py:56`'s route resolves `ctx` through `Depends(get_context)` — **outside** any `try` — so
the same failure yields 500. The docstring and `SERVER.md` both claim "the router's own contract".
Today `config.get_context()` cannot raise (it returns a constant `CallContext`), so this is
unreachable; it becomes reachable the moment real auth lands in that seam — which is precisely what
§1.3 says it is waiting for. Either hoist `ctx = provider()` above the `try` to match, or say in the
comment that this branch is deliberately stricter.

### P5-5 · **Nit** · `SERVER.md`'s new env table reads its own default backwards

`SERVER.md:115` pairs **Default: off** with **Effect: `_build_default_app` builds the storefront
deployment** — the Effect column describes the *set* state while the neighbouring column says the
var is unset. Row 116 has the same shape with the opposite polarity (`Default: **on**`, Effect
describes *Off ⇒ …*), so the two rows read in opposite directions. Prefix each Effect with the
condition (`When set (=1) …` / `When off (=0) …`).

### Answers to the six questions in the brief

1. **The flattening helper — my independent judgment.** *Correct and complete for presence, on
   every registration shape `create_app` uses; incorrect for path spelling under a prefix
   (P5-1); silently blind to one route class nobody uses (P5-2).* I exercised it directly against
   FastAPI 0.139 rather than reading it (Appendix I §1): plain `include_router`, `include_router`
   with a prefix, a router included into a router (two levels), `@app.get`, `add_api_route`,
   `@app.websocket`, a `Mount` of a sub-Starlette app, and `APIRouter(prefix=…)`. **Every one is
   reported.** The nested case matters and works — the walk recurses, and a two-level include
   yields the inner route, not an opaque wrapper. `Mount`s deliberately stop the walk and appear as
   one path each; the `/` static mount normalises to `""` and **does** land in `_registered_paths`,
   so `== ["/health"]` catches a surviving `/` mount by itself — the separate `isinstance(r, Mount)`
   assertion is belt-and-braces, not the load-bearing part.
   The vacuity mode is real and the positive control closes it: I re-ran the whole helper with its
   traversal attribute renamed, and the `dev_surface=False` assertion **still passed** (`["/health"]`)
   while the control **failed** — 37 registered paths collapse to 2. That is the exact defect shape
   this build keeps producing, and the control is the thing that makes it go red.
2. **`mount_mcp = mount_mcp and dev_surface` — domination is total.** All four `/mcp` seams sit
   behind the post-assignment `mount_mcp`: `mcp_mod.configure(...)` and `mcp_lifespan` (`app.py:270`),
   `app.mount("/mcp", mcp_app)` and `app.add_middleware(_McpPathAlias)` (`app.py:350-353`). Built
   for real with `mount_mcp=True, dev_surface=False`: `_registered_paths == ['/health']`, zero
   `Mount`s, **and `app.user_middleware == []`** — the `_McpPathAlias` shim goes with it, which the
   test suite does not assert but the code gets right. Nothing re-enables it downstream: `app.py`
   contains exactly four route-registration calls (`grep -n 'include_router\|\.mount(\|add_middleware\|add_api_route'`)
   and the only `create_app` callers in the component are `_build_default_app`'s three returns.
3. **`/health` — exactly one, in both configurations, contract genuinely matched.** Measured:
   default app = 37 registered paths, `count("/health") == 1`; `dev_surface=False` = `["/health"]`.
   There is no path that registers two — the bare route lives in the `else` of the same `if
   dev_surface` that includes the router. The failure behaviour is real, not approximated: my M-f
   mutation (return 200 without calling `services.ping`) is killed by
   `test_dev_surface_false_health_reports_503_when_falkordb_does_not_answer`, and that test drives a
   live `TestClient` through the lifespan against real FalkorDB, so the 503 comes from the route, not
   from a startup abort. One reachable-only-after-auth divergence: P5-4.
4. **All three `_build_default_app` return paths, verified separately.** Lines 390 (plain), 455
   (workflow), 462 (responder) each pass `mount_mcp=dev_surface, dev_surface=dev_surface`. I broke
   each one independently and each killed **only its own** parametrized id — `[plain-app]`,
   `[workflow-app]`, `[responder-app]` (Appendix I §5, M-d/M-e/M-g). The parametrization is not
   decorative.
5. **The mutation set — I re-ran nine of my own, all killed, and the set does cover a partial surface.**
   I re-derived the battery myself against a copy-aside of `app.py` rather than trusting the report
   (`app.py` restored byte-identical, md5 `fe8102d7…`, and the full suite re-run green afterwards).
   The partial-`dev_surface` question is the interesting one, and the answer is yes: I removed each
   of the three surfaces' guards **independently** — legacy router mounted regardless (M-h), `/`
   mount regardless (M-c), `/mcp` regardless (M-a) — and each is killed on its own by the single
   exact-list assertion, because `_registered_paths(app) == ["/health"]` is an equality over the
   whole table rather than three membership checks. The one gap the set leaves is the
   `_McpPathAlias` middleware: it is correctly gated, but no test would fail if it were not
   (removing `and dev_surface` is caught by the `/mcp` *route*, so the middleware is never the thing
   under test). Not worth a finding — the middleware only rewrites a path whose mount is asserted
   absent — but worth knowing that its correctness here is by construction, not by test.
6. **`SERVER.md` — right call, accurate content.** Updating it was correct: S3 falsified §1.3 and
   §2.1 directly, and it is assigned to no step (P5-3 is that the map should say so, not that S3
   should have skipped it). The load-bearing new claim — *"until [auth] lands, the whole REST router
   is unauthenticated"* — is **true and I verified it**: `api.py`'s only dependency across the entire
   router is `Depends(get_context)` (no `Security`, no `Header`, no `Authorization`), `api.py:45`
   forwards to `config.get_context`, and `config.py:162-170` returns a hardcoded
   `CallContext(ws=WS_ID, actor=USER_ID)`. The §1.4 `/health` row and the §2.1 note are both
   accurate. `README.md`/`AGENTS.md` untouched is right — neither carries an env-var table that S3
   made incomplete, and S16 owns the narrative.

### Carry-forward confirmed for S16

**Two** new env vars, not one: `FALKORCHAT_STOREFRONT_ENABLED` (`config.py:145`, default off) and
`FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH` (`config.py:124`, default **on**). Both are currently
documented only in `docs/SERVER.md` §1.3's table.

### What's solid (Pass 5)

- **The positive control is the right instrument and it works.** Not a formality: with the helper's
  traversal broken, the `dev_surface=False` assertion still passes and only the control fails. That
  is the first artifact in this build that structurally prevents the recurring defect rather than
  avoiding it once.
- **The exact-list equality over the whole route table** (rather than three `not in` checks) is what
  makes every partial-surface mutant die on a single assertion.
- **`mount_mcp = mount_mcp and dev_surface` as one dominating line** rather than a guard at each of
  the four `/mcp` seams — one place to be right, and the `dev_surface=False, mount_mcp=True` test
  proves the dangerous shape is inexpressible from a call site.
- **The `_IncludedRouter` trap was found by the implementer while writing the test**, and written
  into the helper's docstring where the next reader meets it. That is the correct home for it.
- `dev_surface` genuinely has no env var — `grep -rn 'dev_surface' falkorchat/` finds it only as a
  parameter and in comments. The structural claim in §4.9 move 1 holds as built.

### Open questions (Pass 5)

None blocking. **One routing decision for `teco`:** P5-1 is a defect in an S3 artifact whose
consequence lands entirely in S8. Fixing it now (a 2-line change to `_route_paths` plus a prefixed
assertion in the control) keeps S3's owner on it; deferring it makes it an S8 pre-condition that
S8's implementer must be told about explicitly, or it will be re-derived — or not.

**Environment left as found:** I ran the full suite once serially (**2391 passed, 14 deselected**,
16.8s) and seven mutation batteries against a copy-aside `app.py`, restored byte-identical before
the suite run. I created no graph key and deleted none. `ws:test` holds the suite's fixture state;
`reference` is empty of node data, as the brief said it would be — the suite's `wf_repo` fixture
wipes it with no teardown, so it is still empty now. `ws:s1v6`, `ws:s1v7`, `ws:probe-s0r3`,
`ws:probe-s4b` still await stakeholder cleanup; I touched none of them.

---

## Pass 6 — 2026-09-02 (S6: the storefront core)

**Reviewed:** commit **`2f7938d`** — `falkorchat/storefront.py` (new, 518 lines), `config.py`
(+61/−0), `tests/test_storefront.py` (new, 45 tests), `docs/SERVER.md` §1.3, `docs/HISTORY.md` —
against §5.1's **S6** row **as it stands at `acb5a2a` (plan v1.16)**, quoted below, plus §4.3, §4.9
and §4.10. An architect is producing v1.17 concurrently; if the row moves, this pass judged the
v1.16 text. The working tree at these three paths is byte-identical to the commit (`git diff
--quiet 2f7938d`), so tree and commit are the same review surface. Not reviewed: S7/S8/S9/S10 (not
written), and the S4 repository methods this module calls (Passes 3–4).

> **S6 done-condition (v1.16):** *Join provisions `User`+`Channel`+`Thread`+profile-name
> idempotently; wrong/absent/malformed/deleted-participant tokens all resolve to `None`; **restart
> survival: a `Storefront` rebuilt from scratch resolves a token minted by the previous
> instance**.*

**Verdict: approve with suggestions.** 0 blockers · 2 major (one in S6's tests, one a plan
carry-forward) · 2 minor · 0 nits. **The security property S6 exists to hold, holds** — I can state
it structurally, not just by inspection: `self._records` is touched at eight sites, all in this
module; its only two *readers* are `lookup` and `cached_ids`; `resolve_token` reaches it solely
through `_cache_put`/`_cache_drop`, both write-side; and no other module in the package calls
`lookup`/`cached_ids`/`forget` yet. The diff is careful work and the two mutation claims in the
commit message under-sell it — the cache-first mutant is killed by **five** tests, not one.

CPG: considered, not relevant — `cpg_falkorchat` is loaded and reachable, but `storefront.py` is
new in this commit and therefore absent from the graph; the reachability questions here ("can the
cache reach an auth decision", "can a client influence `participant_id`") were answered by
`grep`-complete enumeration of an 8-site private attribute plus live execution against `ws:test`.

### S6-1 · **Major** · `resolve_token`'s cache refresh is load-bearing for S9 and pinned by nothing — the mutant survives all 2439 tests

`storefront.py:~425` ends `resolve_token` with `self._cache_put(record)` before returning. Delete
that one line and **every test in the repository still passes** (`2439 passed, 14 deselected`,
Appendix J §2). Yet it is the only thing keeping `lookup` — the cache reader S9's turn workers and
S7's post-reset profile write are told to use — fresh: with it, a record changed behind the
storefront's back is visible to `lookup` on the next authenticated request; without it, `lookup`
serves the join-time record indefinitely. Demonstrated live (Appendix J §3):

```
shipped:  resolve_token -> language 'es'   lookup() -> 'es'
mutant:   resolve_token -> language 'es'   lookup() -> 'en'   # what an S9 worker sees
```

This is a deletion a *careful* reader is invited to make: the module docstring, the `resolve_token`
docstring and the `_records` comment all say, correctly and three times, that the cache is **never
consulted** by `resolve_token` — which reads as "so why is it writing to it?". S7 lands in this file
next.

**Suggested improvement** (`tdd-engineer`/S6's owner, before S9): one test —
`test_resolving_refreshes_the_cache_so_lookup_never_serves_a_stale_record`: join, resolve,
`repo.set_participant_record(..., language="es")`, resolve again, then assert
`shop.lookup(pid).language == "es"`. It is the `test_resolve_token_reads_the_graph_on_every_call`
you already have, with `lookup` as the observer instead of `resolve_token`, and it kills this
mutant. Add a clause to `_cache_put`'s call site saying why the write is there.

### S6-2 · **Major** · the empty-presenter-key contract is written everywhere except where S10 will look

`compare_digest("", "")` is `True`, `Storefront.presenter_configured` is the guard, and S6 states
the rule in three places: the property's docstring, `config.py`'s `STOREFRONT_PRESENTER_KEY`
comment, and `SERVER.md` §1.3's table. **None of them is the plan.** At v1.16 (`acb5a2a`):
`grep -c 'presenter_configured'` → **0**; `grep -i 'unset key\|empty key\|unconfigured'` → **0**;
§4.3's presenter paragraph, R6, OQ-5 and the S10 row all describe the key without the hazard, and
S10's done-condition says *"a wrong key is refused and counted"* — never *an unset one*. S10's
owner (`coder`, working from its §5.1 row and §5.2, in an isolated context) writes
`hmac.compare_digest(self._presenter_key, submitted)`, that is correct-looking code, and an
unconfigured deployment hands reset-everyone to whoever posts an empty key first. **S6 did its half
right** — the guard exists, is tested both ways, and is documented; the gap is that the obligation
to *call* it is not in the artifact S10 executes from.

**Suggested improvement** (`architect`, in the v1.17 sweep — it is already open for the
`FALKORCHAT_PRESENTER_KEY` → `FALKORCHAT_STOREFRONT_PRESENTER_KEY` spelling fix): add to S10's
Done-condition column — *"an **unset** presenter key authenticates nobody: `presenter_configured`
is checked **before** any `compare_digest`, asserted with a `presenter_key=""` storefront answering
`403` to an empty submitted key."* One clause, in the column S10 is gated on.

### S6-3 · **Minor** · the `FALKORCHAT_DEMO_WS` tripwire passes against a package directory that does not exist — this pass's pre-planted vacuity

`test_no_second_workspace_variable_exists_anywhere_in_the_package` asserts `offenders == []` over
`_PACKAGE_DIR.rglob("*.py")`. `Path.rglob` on a missing directory yields nothing and raises
nothing, so the test is green whether it scanned 27 files or zero. I repointed `_PACKAGE_DIR` at
`falkorchat_RENAMED` and the whole file stayed green (**45 passed**, Appendix J §2). The file's own
docstring makes this the finding it is: it argues, correctly, that a scan asserting *emptiness*
needs a positive control, and `test_join_stores_only_the_hash_of_the_token` duly carries one — this
sibling scan, structurally identical, does not. Its neighbour `test_dev_surface_has_no_environment_variable`
is safe by accident: it shares `_CONFIG_SOURCE` with the env-var test, which asserts a six-element
set equality, so emptying that constant reddens *that* test (verified).

**Suggested improvement:** two lines, either form — `scanned = list(_PACKAGE_DIR.rglob("*.py"));
assert len(scanned) > 20, _PACKAGE_DIR`, or the same positive control the token scan uses (assert a
string that *is* present, e.g. `FALKORCHAT_STOREFRONT_ENABLED`, is found by the identical scan).

### S6-4 · **Minor** · the constant-time tripwire is brittle where it should be loose, and absent where S10 needs it

The *decision* to pin `hmac.compare_digest` statically is right and I would not change it:
constant-time comparison has no observable behaviour, and I confirmed the consequence — replacing
it with `!=` reddens **exactly one test**, this one (Appendix J §2). Two problems with the form:

1. **Over-tight.** `assert "hmac.compare_digest(stored_hash, hash_token(token))" in body` matches
   exact call text; a formatter wrapping that line, or renaming the local `stored_hash`, reddens a
   correct implementation. The clause doing the real work is `assert "==" not in body`.
2. **Under-scoped.** It inspects `Storefront.resolve_token` only. The *other* `compare_digest` site
   is S10's `presenter_login` — the one carrying S6-2's hazard — and this tripwire will not cover it.

**Suggested improvement** (verified, Appendix J §4): a spy is strictly better on axis 1 —
`monkeypatch.setattr(storefront.hmac, "compare_digest", spy)`, resolve a valid token, assert the
spy was called **once** with `(stored_hash, hash_token(token))`. I ran it: 1 call, args exactly
those, resolution still succeeds. It survives reformatting, and unlike the source read it also goes
red if a future branch *skips* the comparison. Keep `assert "==" not in body` alongside it, and
when S10 lands, extend the same pair to `presenter_login`.

### Answers to the five questions in the brief

1. **Can the cache reach an authorization decision? No — and structurally, not by inspection.**
   `grep -n '_records' storefront.py` gives eight sites, all in this module: one init, one lock,
   and six accesses. Exactly two are **reads** — `lookup:439` and `cached_ids:456`. `resolve_token`
   reaches `_records` only through `_cache_put`/`_cache_drop`, both write-side, and
   `grep -rn 'lookup(\|cached_ids(\|forget(' falkorchat/ | grep -v storefront.py` is **empty**, so
   no module consumes them yet. The three routes you named: **S7/S9's `lookup`** — non-auth by
   construction, since it takes a `participant_id` the caller must already have resolved and returns
   no credential material; **an exception route** — `get_participant_record` is *not* wrapped in a
   `try`, so a FalkorDB outage **raises out of `resolve_token`** rather than falling back to memory:
   fail-closed, which is the right posture and worth keeping when S8 adds its error map; **a partial
   write** — if `save_profile` raises after `ensure_participant`, `_cache_put` never runs and the
   caller never receives the token, so the failure leaks an orphan participant node, not a
   credential. The residual is a **typing** risk, not a control-flow one: `lookup` and
   `resolve_token` return the identical `ParticipantRecord` with `token is None` in both cases, so a
   future author who writes `record = shop.lookup(pid)` holds an object indistinguishable from an
   authenticated one. Cheapest structural guard, and the same technique this suite already uses: a
   source tripwire in S8's `test_storefront_api.py` asserting `storefront_api.py` never calls
   `.lookup(`.
2. **The tripwire's shape** — see S6-4. Right decision, wrong form on two axes, both fixable in the
   test file alone.
3. **The presenter-key contract's placement** — see S6-2. The guard is correct and correctly
   tested; the contract is not in the artifact S10 executes from.
4. **The idempotency/rotation reachability claim is true, and the write-through opens nothing.**
   Two independent reasons, both checked: (a) `participant_id = self._id()` and
   `_default_participant_id` is `"p-" + uuid4().hex` — `join(display_name, language)` never derives
   the id from either argument, and no route accepts one, so the `else` branch needs a caller that
   pins `id_gen`; (b) even reached, it cannot escalate, because `set_participant_record`'s
   `MATCH … WHERE u.tokenHash IS NOT NULL` (`repository.py:3541`) means it can only overwrite an
   **existing participant's** hash — it cannot stamp a `tokenHash` onto `seed_demo.sh`'s `u1` or the
   lifespan's `config.USER_ID` node, and `ensure_participant` raises `MemberIdCollisionError` on
   that id shape before the branch is even reached. So the worst a pinned `id_gen` buys is
   overwriting a live participant's own credential with a fresh one the same caller just minted.
   One thing to carry, not a finding: that whole argument rests on **no caller ever passing
   `id_gen`**, and S8 is the one that will construct the `Storefront` in `create_app`.
5. **The pre-planted one is S6-3** — a scan asserting emptiness with no control, in the same file
   whose docstring explains why that is the trap. Everything else I probed holds: the negative
   credential set is genuinely paired (`test_a_valid_token_resolves_to_that_participant`, plus
   in-test controls on the two that matter); the two *transition* tests are real transitions; the
   restart test asserts `cached_ids() == frozenset()` before it answers and builds its second
   instance on its own `Repository` over its own `db.connect()`; and the token scan's control
   (`"Ada"` is found, the token is not) does what it claims.

### On the two carry-forwards you flagged

- **`FALKORCHAT_PRESENTER_KEY` vs `FALKORCHAT_STOREFRONT_PRESENTER_KEY`: agreed, and S6 is not in
  breach.** §5.1's S6 row lists it as `_PRESENTER_KEY` under the `FALKORCHAT_STOREFRONT_` elision it
  shares with `_DIR`/`_TURN_WORKERS`, which is exactly what the code implements; only §4.3, R6 and
  OQ-5's prose disagrees. Architect, v1.17 — right routing.
- **`SERVER.md` §1.5: I'd route it differently.** The block is headed *"Layout (as built, M1)"* and
  lists 8 modules; the package has **27**. It omits `trigger`, `responder`, `executor`, `llm`,
  `modelconfig`, `ingestion`, `chunking`, `embedding`, `extraction`, `fusion`, `guards`, `pricing`,
  `proof_defs`, `querygen`, `tools`, `transport`, `background` — nineteen modules, none of them
  S6's, accumulated across M2–M6. Hanging it on S8 makes S8 the owner of five milestones of doc
  debt it did not create, and the `(as built, M1)` label arguably makes the block an honest
  historical snapshot rather than a false claim about now. Suggest a standalone `BACKLOG.md` item
  ("refresh `SERVER.md` §1.5 to the current module set, or retitle it as an M1 snapshot") rather
  than an S8 obligation.

### What's solid (Pass 6)

- **The invariant is stated once, in the module docstring, with both consequences named** —
  restart survival and immediate revocation — and both are pinned by tests that go red under the
  matching mutation. That docstring is the best thing in the diff.
- **The cache-first mutant is killed by five tests**, not the one the commit message claims:
  `…_idempotent_when_the_participant_id_repeats`, `…_wrong_token_for_a_real_participant…`,
  `…_one_participants_token_never_resolves_under_anothers_id`, `…_deleted_participant_stops_resolving…`,
  `…_reads_the_graph_on_every_call`. Defence in depth that happened rather than being designed.
- **`resolve_token` is fail-closed on a DB error** (no `try` around the graph read) and
  **fail-closed on a partial join** (no token reaches the caller if `save_profile` raises).
- **`parse_bearer` returns `None`, never raises, for eleven malformed shapes**, and the
  `presenter-credential` and `unknown-participant` params quietly cover two cases the plan words as
  one — a credential that *looks* like S10's and an id that never existed.
- **`_env_csv`'s empty-value fallback is a real design call, not defensive padding**: an operator
  typo producing an empty locale tuple would reject every language a participant could pick, and the
  test says exactly that.
- **`set_participant_record`'s `tokenHash IS NOT NULL` anchor does the heavy lifting** for question
  4, and S6 relies on it rather than re-checking in Python — the right side of the "two places that
  can disagree" line.

### Open questions (Pass 6)

None blocking. **One sequencing call for `teco`:** S6-1's missing pin is cheap now (one test in a
file S7 is about to edit anyway) and expensive later — S9 is the consumer that would be bitten, and
by then `_cache_put` will look even more like a dead write. I would land it with S7 rather than
carry it.

**Environment left as found.** I ran the S6 file (45 tests) and the full suite twice serially —
**2439 passed / 14 deselected** on the pristine tree, and once more under the S6-1 mutant to
establish that it survives everything, not just `test_storefront.py`. Six mutations against
byte-copies of `storefront.py` and `test_storefront.py`; both restored and `md5sum -c` verified
after every one, and `git status --porcelain falkor-chat/` is clean. `ruff check` on
`storefront.py`, `config.py` and `test_storefront.py`: **All checks passed**. My probe scripts
wiped and re-populated `ws:test` (the suite's own fixture does this per test, so nothing was lost);
`reference` is empty of node data, as the default run leaves it. I created no graph key and deleted
none; `ws:s1v6`, `ws:s1v7`, `ws:probe-s0r3`, `ws:probe-s4b` are untouched.

---

## Pass 7 — 2026-09-03 (S7: state, reset, catalog, images)

**Reviewed:** commit **`dd78e70`** — `falkorchat/storefront.py` (+465/−9) and
`tests/test_storefront.py` (+950, 79 tests in the file) — against §5.1's **S7** row at plan
**v1.17** (`0ba772b`), §4.7, §4.8, and `docs/plans/salesperson-ui-graph.md` §7 (a)–(d) and §12.
Baseline: `git show dd78e70`, the working tree at both paths byte-identical to it. Not reviewed:
S8/S9/S10 (not written); the S4 repository methods (Passes 3–4); S6 (Pass 6, whose three findings
S6b closed at `5594134`).

**Verdict: approve with suggestions.** 0 blockers · 0 major · 3 minor · 1 nit, plus the three
rulings below. This is the largest diff in the build and the most carefully evidenced: the two
failure shapes that have bitten this build repeatedly — an assertion that cannot go red, and a
"nothing changed" that conflates two meanings — are both attacked head-on and mostly beaten. The
catalog workaround, which I expected to be the weak point, **verifies correct against the real
15-product catalog**: 15/15 rows resolved, 15 unique ids, **zero mis-bindings, zero drops**
(Appendix K §1).

CPG: considered, not relevant — `cpg_falkorchat` is loaded, but `storefront.py`'s S7 half is new
in this commit and absent from the graph; the reachability questions here (who calls `.lookup(`,
what the one-line `filter_products` fix breaks, whether the quiesce spy has teeth) were answered
by grep-complete enumeration plus live execution against `ws:test` and `reference`.

### Ruling 1 — take the one-line fix, at S8. The workaround is correct, and its stated blocker is already moot.

**The workaround is not a defect.** Its join key is `normalize_name(row["name"])` →
`Product.nameNormalized`, and both sides call the *same* `extraction.normalize_name`
(`scripts/seed_catalog.sh:75` imports it), so the round trip is exact by construction. Verified
live against the real catalog, not the fixture: **15 of 15 rows resolved, zero mis-bindings, zero
drops** (Appendix K §1). It could ship.

**But the second reason for declining the fix is already false.** `LookupProductFactTool.run`
returns `json.dumps({"found": True, **row})` (`tools.py:428`), and `services.lookup_product` has
projected `productId` since **K-053** — so the salesperson agent's context **already contains
product slugs**, from the sibling catalog tool, today. Adding `productId` to `filter_products`
makes the two tools consistent; it does not introduce a new category of thing into the prompt. The
def's own system prompt already names only "(name, category, price)" while `lookup_product_fact`
already returns four fields, so that drift exists now and this does not widen it.

**And the first reason costs nothing.** I applied the fix (projection + row mapping, 4 lines in
`repository.filter_products`) and ran the whole suite: **2473 passed, 14 deselected — zero
failures, zero test edits** (Appendix K §2). `test_repository.py` projects `[r["name"] for r in
rows]`, and `test_tools.py` drives a stub whose rows the caller supplies, so neither sees the extra
key.

**The honest counterweight, which I want on the record:** *zero test breakage measures code, not
model behaviour.* None of the 14 `live`-marked tests exercises a salesperson catalog conversation
(they are AC-5 grounding, querygen NLQ, and triage) — so **no harness in this repo would observe an
LLM regression either way**. The evidence for "safe" is the K-053 precedent, not a passing test.

**Recommendation:** take the fix as part of **S8** — which already reaches `app.py`/`schemas.py`,
is the next step in this area, and can drop `_catalog_rows`'s second read in the same change
(removing the `1+n` reads *and* the `if product is None: continue` silent-drop branch, the only
unbounded failure path on the catalog route). **If you'd rather not reach into a delivered file at
all, the workaround may ship unchanged** — but then add the tripwire it is missing: nothing asserts
`normalize_name(p.name) == p.nameNormalized` for the *real* seed, because `_catalog_rows(n)`'s
fixture satisfies it by construction.

### Ruling 2 — the substituted assertion is right, and stronger than reported. §4.8 needs a note, not a correction.

The author did not merely swap in "the `PLACED`/`Cart` subgraph is empty". The load-bearing
assertion is in `test_the_profile_name_is_back_after_a_self_reset_not_an_em_dash`:
`profile == {"name": "Ada", "deliveryAddress": None}` — and its docstring says exactly why that is
the right fact: *"`deliveryAddress` is asserted `None` in the same breath: it proves the `Customer`
really was deleted, so the name coming back is a re-write and not a survivor."* That is precisely
the fact §4.8's inventory was standing in for, and it survives the `MERGE` that made the naive
assertion false. Together with the `PLACED`/`Cart` counts and the anchor comment left at the
deleted assertion's site, this is better evidence than the plan asked for.

**§4.8 is not wrong.** Its column is headed *Deletes*, and the delete does delete the `Customer`.
What is missing is that the row reads as a post-reset inventory, and the post-reset graph holds a
**name-only `Customer`** again. So: a **note**, in the Survives column or as a footnote on the
reset-mine row — *"(a name-only `Customer` is re-created immediately afterwards by §4.10's profile
re-write; `deliveryAddress` does not come back)"* — not a correction. Fold into v1.18.

### Ruling 3 — "waiting subsumes cancelling" is sound for correctness, and explicitly not for availability. State that second half in S9's row.

**Sound.** A queued turn reaches a worker, completes, and clears its own entry; `_await_quiesce`
returns only once the map says idle; the delete follows. There is no ordering in which cancelling
would have made the *result* different — only sooner. And the author's refusal to drop the turn-map
entry as a stand-in is exactly right: that would report idle while the job was still queued and let
the delete race the turn quiesce exists to prevent. The docstring states the S9 constraint
correctly ("cancellation belongs *there*, in front of this wait, never in place of it").

**The boundary it does not state, and should.** Waiting subsumes cancelling for correctness but
**not for availability**. `STOREFRONT_QUIESCE_S` is 30 s and a turn may run to the 180 s agent
timeout, so a slow turn converts reset-mine into a `503` — a refusal, with nothing reset — where
cancellation would have dropped the queued work and let the reset succeed. That is a designed
outcome, not a defect (the `503` contract is explicit and tested), but it is the *reason* §4.8
wanted cancellation, and it is the thing that will be forgotten when S9 lands. **Suggested:** one
clause in §5.1's S9 row — *"cancellation of a queued turn runs in front of `_await_quiesce`, never
in place of it; without it a turn outliving `quiesce_s` turns reset-mine into a `503`."*

### S7-1 · **Minor** · the quiesce test asserts the post-conditions, not that the reset waited — and degrades to green, not red, under an adverse scheduler

The spy is real and has teeth: dropping the `_await_quiesce` call reddens two tests
(Appendix K §3, M1), and if the spy never ran, `at_delete["runStatus"]` would `KeyError`. So it
proves more than "the spy ran". But every assertion it makes —
`runStatus == "done"`, `messages == 2`, `posted: True` — is **also satisfied by the ordering in
which the worker finished before the reset even started**, i.e. with nothing to wait for. I ran
that ordering (joined the worker before calling `reset_participant`) and **all four quiesce tests
stayed green**. The safety today is timing, not assertion: the worker's `time.sleep(0.15)` against
a 5 s budget, and I measured the reset genuinely blocking **0.189 s / 0.189 s / 0.190 s** across
three runs. Non-flaky — 20 consecutive runs of the file, 79 passed every time — but "non-flaky" and
"asserts the wait" are different claims, and only the second one survives a loaded CI box.

**Suggested improvement:** two lines, and the floor is the worker's own sleep so it cannot be
tight — `t0 = time.monotonic()` before the reset, and after it
`assert time.monotonic() - t0 >= 0.15, "the reset did not wait"`. Measured margin 0.189 vs 0.15.

### S7-2 · **Minor** · the reset's two *error-path* cache evictions are unpinned, while the success path is

`reset_participant`'s success path writes the refreshed record through (`_cache_put`) and **is**
pinned — removing it reddens `test_reset_refreshes_the_cached_record_so_lookup_never_serves_a_dead_thread`
(Appendix K §3, A6). Its two error paths are not: removing `self._cache_drop(participant_id)` from
`_reset_state_unknown` (the F8/`504` path, where the delete **may have committed**, which is the
docstring's own stated reason for the line) and from the `status is None` branch both leave **79
passed**. Same shape as Pass 6's S6-1, on the paths where the cached `threadId` is *most* likely
wrong.

Honest mitigation, which is why this is minor and not major: `resolve_token` refreshes the cache on
every authenticated request and S8 calls it before anything else, so the stale window closes on the
participant's next request. The exposure is an S9 worker calling `lookup` **between** a failed
reset and that next request.

**Suggested improvement:** one test per path, using the existing A6 test as the template — after a
`ResetStateUnknownError`, assert `participant_id not in shop.cached_ids()`.

### S7-3 · **Minor** · a broken quiesce deadline hangs the suite instead of failing it

There is no `pytest-timeout` in the project. I extended `_await_quiesce`'s deadline by an hour and
`test_a_quiesce_timeout_changes_nothing_and_leaves_the_turn_running` **never returned** — it had to
be killed at 25 s (Appendix K §3, M3). The mutant is "caught", but as a hang: on CI that is a
job-level timeout with no failing test name, and locally it blocked my own battery for two minutes.
This is the only test in the suite that can wait on a wall clock it does not control.

**Suggested improvement:** bound it at the test rather than the module — either
`@pytest.mark.timeout(10)` (needs the `pytest-timeout` dev dependency) or, dependency-free and
sufficient here, assert the elapsed time: the `quiesce_s=0` storefront must refuse in well under a
second, so `t0`/`assert time.monotonic() - t0 < 1.0` turns the hang into a failure.

### S7-4 · **Nit** · `list_catalog`'s first call reads the catalog twice

When the manifest has not been built, `list_catalog` calls `build_image_manifest()` (which calls
`_catalog_rows()`) and then `_catalog_rows()` again — `2 × (1 + n)` reads, 32 against the real
catalog. Harmless in production, where S8 builds the manifest in the lifespan, and
`test_list_catalog_builds_the_manifest_when_nobody_did` covers the fallback. Vanishes entirely if
Ruling 1's fix is taken. Mentioned only so it isn't mistaken for the `1 + n` the call-site comment
documents.

### The other four things you asked about

1. **Does the spy prove what it claims?** — S7-1. It proves the delete is issued *after* the run
   reached `done` and the reply was written; it does not prove the reset *waited* for that. Both
   halves matter and only the first is asserted.
2. **`reset_participant(ParticipantRecord)` vs S8's route.** Correct, and the argument holds.
   §5.2's `POST /shop/api/reset` takes **no body**, and S8's `get_participant()` dependency resolves
   the bearer through `resolve_token`, which re-reads the graph — so the handler holds a
   graph-fresh `ParticipantRecord` at exactly that moment. The three fields the reset consumes
   (`participant_id`, `display_name`, `language`) are precisely the three the reset does not touch,
   so there is no window in which the record can be stale *for this use*. Passing a `ctx` plus a
   name would indeed be two sources that can disagree.
3. **The `catalog_repo` teardown cannot break another test file.** `wf_repo` wipes `reference` on
   **setup**, so every test that needs reference data seeds it inside its own test; a teardown wipe
   can therefore only remove data no later test relies on. The fixture's load-bearing claim — that
   the schema survives — I verified live: `reference` holds **4 indexes before and 4 after** a
   `MATCH (n) DETACH DELETE n`. And the problem it fixes is real: `seed_catalog.sh` `MERGE`s by
   `productId`, so a stray `widget-…` survives a re-seed and makes `verify_catalog.sh` report 17
   against 15.
4. **Did S7 need `Storefront.lookup`? Grep verified — no.** `grep -rn '\.lookup(' falkorchat/
   tests/` returns **eight hits, all in `tests/`**, none in the package; the production callers are
   zero. The staleness argument is also right, and `test_reset_refreshes_the_cached_record…` is its
   proof. **But do not delete `lookup` yet, and here is the sharper reason:** if it goes, the cache
   has no reader at all — `cached_ids()` is diagnostics — and `resolve_token`'s `_cache_put` (Pass
   6's S6-1) plus `reset_participant`'s `_cache_put` become writes into a map nobody reads. The
   right end state would then be deleting **`_records` entirely**, not just `lookup`; half-doing it
   leaves a write-only map, which is worse than either end. That is one decision — *does S9's
   executor want a per-participant record cache?* — and it belongs at **S9**, made once, not
   inferred from S7's silence.

### What's solid (Pass 7)

- **`get_state`'s order block is asserted to be the repository read** with a third product live in
  the cart, so a locally-composed block reports the wrong order and the test says so. That is a
  test written against a specific wrong implementation, not against the right one.
- **The four reset outcomes are four distinct exception types**, and F8's two orderings are both
  tested — including the one that matters more (`state=None` when the re-read *also* times out),
  with the reasoning for why that is the *likelier* fault, not the exotic one.
- **§7 (d) carries a false-positive control** (Bob's cursor), which rules out the "swept every
  cursor in the workspace" implementation that would satisfy (d) and be a worse defect.
- **`test_an_idle_participant_is_not_made_to_wait` is named as the control it is** — without it,
  "the reset waits" and "the reset refuses" are both satisfied by a reset that always refuses.
- **`_ticking_clock` removes a real coin flip** (`placedAt` ties broken by `orderId DESC`, and
  `orderId` is a `uuid4`) rather than betting on the wall clock ticking between two calls.
- **Four benign refactors of my own stayed green** (Appendix K §3, B1/B2 plus the two I folded in),
  so the suite is pinned to behaviour, not to source text — the failure mode Pass 6's S6-4 warned
  about.
- **The declined fix was documented at the call site with both alternatives and their costs**,
  which is what made this ruling a twenty-minute measurement instead of an archaeology exercise.

### Open questions (Pass 7)

None blocking. Ruling 1 is the only one needing your decision before S8 is dispatched; Rulings 2
and 3 are documentation touches for v1.18 and the S9 row.

**Environment.** Full suite pristine, serially: **2473 passed / 14 deselected** (18.1 s), matching
your solo run; `ruff check` on both S7 files: **All checks passed**; `test_storefront.py` run 20×
(5 whole-file, 15 single-test) with **79 passed** every time. Eleven mutations against byte-copies
of `storefront.py`, `test_storefront.py` and `repository.py`; all three restored and `md5sum -c`
verified, `git status --porcelain` clean at every falkor-chat path. **One deliberate change of
state, reported rather than restored:** `reference` was empty of node data when I began (as Pass 6
left it); I ran `./scripts/seed_catalog.sh` to test the workaround against the real catalog, and
**re-seeded it after my last suite run**, so `reference` now holds the clean 15-product catalog and
`./scripts/verify_catalog.sh` reports **OK — in sync (15 products)**. That is strictly better than
I found it, but it is a change: the next default `pytest` run will empty it again. I created no
graph key and deleted none; `ws:s1v6`, `ws:s1v7`, `ws:probe-s0r3`, `ws:probe-s4b` untouched.

---

## Pass 8 — 2026-09-03 (S7b: the two overrides, gated)

**Reviewed:** commit **`d9d2f2b`** — `tests/test_storefront.py` only (+166/−8), against Pass 7's
three minors **S7-1, S7-2, S7-3**. `falkorchat/storefront.py` is byte-identical to its parent and
to `dd78e70` (`md5 47ffe3abe13aafcad552c5f836a53921` at all three), so this is a test-only unit and
the production surface is unchanged. Gated **fresh**, by a different reviewer than Pass 7, for one
reason: the implementer **rejected the suggested fix on S7-1 and S7-3 and substituted its own**.
Everything below was executed, not read. Not reviewed: S7-4 (deferred to S8 with Ruling 1),
S8/S9/S10.

**Verdict: approve with suggestions.** 0 blockers · 0 major · 1 minor · 2 nits.

**Both overrides are substantively better than what they replaced, and the sharper claim is
correct.** I reproduced S7-3's counter-claim directly: Pass 7's suggested elapsed-assert form,
applied to the very test it was suggested for, **hangs** under Pass 7's own deadline mutant — killed
at 30 s, exit 143 (Appendix L §3). Pass 7's suggestion was structurally wrong and the implementer
was right to refuse it. S7-1's substitute is likewise stronger than the duration floor: its
detection power is genuinely duration-free — with the stub turn's sleep set to **zero** the no-wait
mutant is still caught **8/8** (§L §2), which no floor could survive. The one thing I found is where
one of the two instants is stamped.

CPG: considered, not relevant — `cpg_falkorchat` predates `storefront.py` entirely (established by
three prior units in this coordination); nothing here is a reachability question in any case, and
every claim below is settled by execution against the live `ws:test`.

### S8-1 · **Minor** · `started_at` is stamped on the *calling* thread, so the ordering assertion tolerates a scheduling gap — and tolerates it **green**

`_call_bounded` takes `started = time.monotonic()` at function entry (`tests/test_storefront.py:103`)
and stores it **before** the `Thread` is constructed and before `worker.start()`. So
`outcome["started_at"]` is *when the test asked for the reset*, not when `reset_participant` began
running, and `assert outcome["started_at"] < finished_at` tolerates the whole
main-thread-to-daemon-thread scheduling gap — in the direction that passes.

Demonstrated (§L §2): with `_await_quiesce` replaced by `return True` — no wait whatsoever — plus a
300 ms delay injected between the stamp and the call to emulate a descheduled daemon thread, the test
**passes**. Stamping inside `run()` instead turns the same probe **red**.

It matters because the commit message and the test comment claim "no margin at all". The margin is
the thread-start skew — the same quantity used to reject the duration floor — roughly `TURN_WORK_S`
wide, failing silently instead of loudly. Remote in practice (it needs a >150 ms thread-start delay),
hence minor.

**Suggested improvement:** one line — drop `box["started_at"] = started` and make
`box["started_at"] = time.monotonic()` the first statement of `run()`. Verified: closes the probe, and
10 consecutive runs of the 7 reset/quiesce tests stay green.

### S8-2 · **Nit** · `IMMEDIATE_S`'s comment mis-sizes the one bounded call that commits a write

The comment says the bounded calls take "~0.2 ms". Measured (§L §5): **0.142 ms**, **0.140 ms**, and
**2.435 ms**. The outlier is `test_an_idle_participant_is_not_made_to_wait`, which is the only
bounded call that performs a real delete-and-remint rather than an immediate refusal. The margin is
still 411×, so this is not a flake risk — but it is the one site where a tripped bound leaves a
daemon thread that will go on to **commit a graph mutation into the shared `ws:test`** after the
test that owned it has ended, and the assertion message would then misdiagnose it as "a wait bounded
by the code under test hung". Forcing `IMMEDIATE_S = 0.0001` trips exactly that test and no other
test was contaminated in that run — the per-test `ws:test` wipe absorbed the straggler — so the
exposure is narrow, not zero.

**Suggested improvement:** correct the figure and name the idle site in the comment ("~0.15 ms for
the two refusals, ~2.5 ms for the idle reset, which is the one that writes").

### S8-3 · **Nit** · `_call_bounded` inverts `pytest.raises` — a forgotten assertion is a silent pass

`box["error"] = exc` swallows the exception and returns normally, so a call site that forgets to
check `outcome` turns a raising call into a green test. All four current sites do check
(`assert "error" not in outcome` / `assert isinstance(outcome.get("error"), …)`), so nothing is wrong
today; it is a trap laid for the next site. **Suggested improvement:** re-raise `box["error"]` unless
the caller passes an explicit `expect_error=True` (or `expect=SomeError`), which restores the
default that a surprise exception fails. Separately, `box["seconds"]` is set and never read by any
test.

### Disposition of Pass 7's findings

- **S7-1** — **fixed, by substitution, and the substitution is stronger.** Adverse ordering (worker
  joined before the reset) now reddens **exactly one** test on the **first** assertion, 3 others
  green — the implementer's claim, reproduced verbatim (§L §1). `_await_quiesce → return True` is
  caught by the second assertion, margin 0.153 s. Residual: S8-1.
- **S7-2** — **fixed.** Both error-path evictions now pinned: removing `_cache_drop` from
  `_reset_state_unknown` reddens both parametrizations; removing it from the `status is None` branch
  reddens the new zero-row test (§L §4). The `[times-out-too]` parametrization pulls its weight — a
  mutant that moves the drop *after* the re-read is caught by **that param only**.
- **S7-3** — **fixed, by substitution; the reviewer's own suggestion was wrong.** Verified by running
  it: the elapsed-assert form still had to be killed at 30 s under the deadline mutant, exactly as
  the implementer reported. `_call_bounded` turns the same mutant into **2 failed in 3.34 s** with
  both test names printed, whole-file, no hang (§L §3). Coverage is complete — under the mutant the
  whole file terminates, and no other test file reaches `Storefront.reset_participant`.
- **S7-4** — **not addressed, correctly.** Deferred to S8 with Ruling 1's one-line fix, which
  removes it entirely. Still open.

### The three things you asked about

1. **Is `seconds=10` thin?** No. Measured 188 ms against it (53×), and the storefront's own 5 s
   `quiesce_s` sits *inside* the bound, so a genuine refusal still surfaces as a refusal rather than
   as a bound trip. Correctly chosen.
2. **Are the false-positive probes the right ones?** They are benign but they stress the *easy*
   direction — more time. The informative probes are the tight ones, and I ran them: `QUIESCE_POLL_S`
   at **0.00001** (2000× tighter, the real stress on `returned_at >= finished_at`) → green ×3; and
   `TURN_WORK_S` at **0.0** → green ×5 *while still catching the no-wait mutant 8/8*. That last
   result is the strongest evidence in the unit for the implementer's own "no duration" claim and is
   worth more than the doubling probe that was reported.
3. **Does any new assertion pass while proving nothing?** No. Each of the seven new or changed
   assertions is mutation-backed except `assert "error" not in outcome`, which is bookkeeping the
   helper forces rather than a claim — see S8-3. One residual neither form can reach: a
   `_await_quiesce` replaced by a **blind `time.sleep(0.5)`** satisfies both ordering assertions (and
   would satisfy the duration floor too), but the file still reddens on it — the two `quiesce_s=0`
   tests refuse to refuse (§L §6). Nothing to fix.

### What's solid (Pass 8)

The unit does the thing this coordination keeps asking for and rarely gets: it **ran the reviewer's
suggested fix against the reviewer's own mutant before rejecting it**, and the rejection is correct.
`_call_bounded`'s docstring states the structural reason (an assertion after a call that never
returns is never reached) rather than asserting a preference, and both new eviction tests carry the
mutation result that motivated them. Every one of the four claims in the commit message reproduced.

### Open questions (Pass 8)

None. S8-1 is a one-line change with a verified before/after; S8-2 and S8-3 are optional.

---

## Pass 9 — 2026-09-03 (S7b2 + S7c + S7c2 + S7c3: the catalog projection and the query gate)

**Reviewed:** four commits, oldest first — **`6fbe541`** (S7b2, test-only, `storefront.py`
byte-identical), **`f5291e6`** (S7c: `productId` on `filter_products`, S7's `1 + n` workaround and
its silent-drop branch removed, `QUERIES.md` §15.2), **`8aaeca3`** (S7c2: the `FILTER` constant and
§15.1's pre-existing K-053 drift), **`83af07c`** (S7c3: the `LOOKUP` constant) — against
`docs/plans/salesperson-ui.md` §5.1's **S7c** row at plan **v1.19** (`732f5e0`), and against Pass 8's
three findings. Everything below was executed. Not reviewed: S8 (not written), the untouched S7
surface.

**Verdict: approve with suggestions.** 0 blockers · 0 major · 1 minor · 2 nits. **This surface is
ready to build S8 on** — the projection is correct at every layer I could reach, the four §15 cells
(code, `QUERIES.md`, shell constant, ×2 queries) agree exactly under my own independent comparator,
and the third override in a row is right for the third time.

CPG: considered, not relevant — `cpg_falkorchat` predates `storefront.py` and is stale for
`repository.py` (established by five prior units); the consumer enumeration this pass needed was
answered by grep-complete search plus live execution.

### S9-1 · **Minor** · the S7c tripwire pins one *method name*, not "read once" — a `1 + n` through `self._repo` is invisible to it

`test_the_catalog_is_read_once_not_once_per_product` patches `stocked._services.lookup_product` and
asserts nobody calls it. That binds the three drift routes it was built for — all three verified red
(§M §2) — but not the property its name claims. Reproduced (§M §2, M-F): restore `_catalog_rows` to
a per-row loop *plus the silent-drop `continue`*, routed through `self._repo.lookup_product` instead
of `self._services.lookup_product`, projection left intact → **348 passed**, `test_storefront.py`
and `test_repository.py` both entirely green. The `1 + n` and the silent drop are back, and the
done-condition's own tripwire says nothing.

This is regression breadth, not a defect: the shipped code is correct and S7c's actual two halves
*are* bound. It matters because this test is the plan's single named guard against exactly this
regression, and S9 owns `storefront.py` next.

**Suggested improvement:** one line, verified — add
`monkeypatch.setattr(stocked._repo, "lookup_product", _boom)` beside the existing patch. Reddens
M-F, and 83 passed with no mutant.

### S9-2 · **Nit** · §15.1's new note dates the drift by milestones it can't support

`QUERIES.md` §15.1: *"§15.1 documented a projection `Repository.lookup_product` had not used for two
milestones."* The verified span is six days inside one milestone plus the current un-numbered work:
the code widened at **`bcd2dcc`** (2026-08-28, K-053) and the shell constant was last touched at
**`14891c9`** (2026-08-28, K-052), not again until S7c3 (2026-09-03) — with `test_queries.sh`
reporting **408/408 throughout**. **Suggested improvement:** cite the two commits and the 408/408,
which is a sharper indictment than the milestone count and is checkable.

### S9-3 · **Nit** · (pre-existing, outside this unit) `tests/test_repository.py` is not ruff-clean

`ruff check tests/test_repository.py` → **E741** at `3216` (`for l in order["lines"]`), present since
**`f020f90`** (K-053 cluster 1) and untouched by S7c — confirmed by linting the pre-S7c blob. Raised
only because lint has been reported per-file for the S7 files and this file has never been in that
set, so "lint clean" should not be read as covering it.

### The two claims, verified rather than accepted

- **"No test in this repo can observe what `FilterProductsTool` hands the model." — true.** The three
  `test_tools.py` tests drive `StubServices.filter_products`, which returns `list(self._filter_result)`
  — the test's own literal — and assert `out == {"items": rows}`: an identity check that passes under
  any projection. `FilterProductsTool.run` is `json.dumps({"items": rows})`, verbatim
  (`falkorchat/tools.py:499`), so the slug does reach the model. The only real wiring is
  `build_builtin_registry`, whose two callers are `app.py:440` and `tests/test_workflow_live.py:210`;
  `pytest -m live --collect-only` lists **14**: one AC-5 grounding, twelve querygen NLQ, one triage.
  None is a catalog conversation. The plan's counterweight stands as written.
- **"There is no third consumer of `filter_products`." — true.** `services.filter_products`: exactly
  two production callers, `tools.py:492` and `storefront.py:731`. `repository.filter_products`:
  exactly one, `services.py:2649`. A repo-wide sweep of `falkor-chat/` outside
  `server/falkorchat`, `server/tests` and `docs/` returns nothing.

### The four things you asked about

1. **S7b2's third override — nothing is lost, and it is stronger than what I suggested.** Re-raising
   beats the `expect_error` kwarg on the two checks that matter, both executed (§M §1): under the
   hang mutant the bound's `AssertionError` **escapes** `pytest.raises(QuiesceTimeoutError)` — 2
   failed in 3.31 s, no hang, so S7-3's protection survives the change; and when the bounded call
   both raises *and* overruns, the bound wins and is reported as a bound breach rather than
   swallowed. A site can no longer say nothing. My S8-3 suggestion was the weaker of the two.
2. **The refusal to widen the write site is right, on evidence I reproduced.** With `_await_quiesce`
   mutated to sleep a blind 2 s and report idle: at `IMMEDIATE_S = 1.0` the idle test **reddens**;
   widening that one call to `seconds=10` makes it **pass** (§M §1). The tightness is load-bearing.
   And naming *is* adequate for the hazard I raised: `conn` wipes `ws:test` on the **setup** of every
   test, so a straggler's commit is absorbed unless it lands after the next test's setup — and my own
   Pass 8 probe, which tripped exactly the writing site, contaminated nothing.
3. **The tripwire — closed against what it was built for, open on S9-1.** M-D reproduces the false
   negative the implementer caught: strip the `opaque-sku-42` override and the id-fabricating mutant
   goes **green**, so the override is load-bearing and the fix is real.
4. **The removed silent-drop branch — nothing depends on it, and there is no regression.** `Product`
   carries **UNIQUE, not MANDATORY**, on `productId` (`db.constraints()` live), so a node without the
   property is representable; such a row yields `{"productId": null, …, "imageUrl": null}` silently
   today — and produced the **identical row** under the reconstructed S7 code (§M §3). The deleted
   branch fired only when a row's *name* failed to re-resolve **between the two reads**, a condition
   that cannot arise now that there is no second read. Removing it with the read is correct.
5. **Any new assertion that proves nothing? No.** Every new assertion is mutation-backed (§M §2, §4),
   including the two key-set assertions, which a widened projection reddens. One is strictly
   subsumed — `rows[6]["productId"] == "opaque-sku-42"` is already covered by the id-list equality two
   lines above — deliberate emphasis, no cost.

### Disposition of Pass 8's findings

- **S8-1** — **fixed.** `started_at` is now the first statement of `run()`. Pass 8's own probe (no
  wait at all + 300 ms injected before the stamp) now **reddens**, and the adverse-ordering detection
  margin grew **54 µs → 142 µs**, exactly as claimed (§M §1).
- **S8-2** — **addressed by naming, and the refusal to bound is justified** — see (2). The constant's
  comment now carries the measured 0.2 ms / 2.5 ms split and names the writing site at both ends.
- **S8-3** — **fixed by a better substitute than I proposed** — see (1).
- **S7-4** (Pass 7's nit) — **reduced, not eliminated, and Pass 7 overstated its own fix.** Pass 7
  wrote that it "vanishes entirely if Ruling 1's fix is taken"; it does not. `list_catalog` still
  calls `_catalog_rows()` twice on the first call — once inside `build_image_manifest()` — so the
  count went from `2 × (1 + n)` = 32 to **2**, not 1. Harmless (first call only, S8 builds the
  manifest in the lifespan), and no action is asked for; recorded because the earlier claim is now
  demonstrably wrong.

### What's solid (Pass 9)

The §15 alignment is the strongest part and I checked it the hard way rather than accepting it: my
own AST comparator — independent of the implementer's and of `teco`'s — extracts each method's real
query text from `repository.py`, and **all four cells agree exactly** for both `lookup_product` and
`filter_products` (§M §5). The gate's internal coupling is real in three directions I mutated
(407/408 each), `./scripts/test_queries.sh` is **408/408** on my own run, and S7c's test diffs are
**pure insertions** (31/0 and 53/0), so S7's catalog tests really did stay green unedited. The
implementer found and fixed its own tripwire's false negative before shipping, and M-D confirms both
the negative and the fix.

### Open questions (Pass 9)

None. S9-1 is one verified line; the two nits are optional.

### Database state (read this before the next run)

**`reference` was not seeded when I started** — the coordinator's premise was already stale: the
first `pytest` run of this session emptied it (the `catalog_repo` fixture's teardown), and
`./scripts/test_queries.sh` deletes it outright at teardown. I ran `./scripts/seed_catalog.sh test`
at the end and `./scripts/verify_catalog.sh` reports **OK — in sync (15 products)**. Any default
`pytest` run empties it again.

---

## Appendix

### Appendix A — F-1 causal chain (all links verified by execution or by document)

1. **`ws:acme` `salesperson@v6` ≠ `proof_defs.SALESPERSON_DEF`.** Loaded both, compared field by
   field: `maxIterations`/`model`/`requiredTools`/`tools`/`waitsForHuman` identical,
   `systemPrompt` **different** (graph 2566 chars, file 2537). None of S1's three probe strings
   (`Reply in the language named by \`language\``, `Before you place an order, confirm the delivery
   address on file`, `Never invent a delivery address.`) is present in the graph's copy.
2. **The graph's `v6` is `v5` + one K-060 paragraph.** Diffed graph-v5 → graph-v6: a single added
   sentence, *"When a catalog tool's result spans more than one category (for example because you
   did not pass `category`), check every returned item's own category and price yourself before
   replying — list every item that actually matches what the customer asked for; never drop one
   that matches just because other items in the same result do not."*
3. **That wording is a rejected K-060 lever.** `falkor-chat/docs/BACKLOG.md:67` — *"a `systemPrompt`
   synthesis-time 'check every returned item, never drop a match' safety net … net wrong-reply rate
   went **up** (30% vs. the shipped fix's own 20%) … **Reverted, never shipped.**"*
4. **It exists nowhere on disk or in history.** `grep -rn "spans more than one category"` over the
   whole repo: 0 hits. `git log --oneline -S "spans more than one category" --all`: 0 commits. So
   it was published from a working-tree-only edit — which is precisely why `teco`'s
   `git log -S '"v6"'` disproof cannot settle the question either way.
5. **`ws:s1v6` (S1's own probe graph) holds a `v6` byte-identical to the file.** So S1's author did
   verify cleanly, on a clean workspace; the `ws:acme` copy was never theirs.
6. **The mechanism that made it permanent.** `services.materialize_def` docstring: *"Property-only
   differences stay a silent no-op (unchanged `MERGE … ON CREATE SET` behavior)"*;
   `_check_no_structural_conflict` filters to *structural* diffs, so a `config`-only difference
   raises nothing and writes nothing.
7. **Why it surfaced as "divergent" during S1's run.** `tests/conftest.py::wf_repo` wipes
   `reference` on every workflow test. So: experiment publishes v6 to `reference` + `ws:acme` → a
   later `pytest` run wipes `reference` → S1 re-seeds → publish **creates** the file's v6 in the
   now-empty `reference`, materialize **silently no-ops** against `ws:acme`'s existing v6 →
   `verify` compares the two and reports DIVERGENT. The author's report was accurate.

### Appendix B — mutation spot-checks (run without touching the working tree)

Implemented as three `pytest` plugins on `PYTHONPATH` that rebind `Services._reject_reserved_keys`
/ `Services.start_workflow_run` in `pytest_configure`. No file in the repo was modified.

| Mutation | Result |
|---|---|
| `_reject_reserved_keys` → no-op | **10 failed**, 290 passed — including all 3 new offline + all 3 new live chat-path cases, *and* the 4 pre-existing process-path cases (so the hoist did not weaken them) |
| Guard moved **after** `repo.start_run` on the chat path | **6 failed**, 294 passed — the 3 offline (`repo.started_runs == []`) and the 3 live (`test_process_input.py:239`, the `WorkflowRun` count) |
| Merge reversed to `{"threadId": …, **caller_ctx}` | **2330 passed, 14 deselected — fully green.** Confirms the author's claim; basis for F-3 |

### Appendix C — the hoist's exact process-path delta

`HEAD` order: `_require_executor()` → `_id()` → `_clock()` → `executor.step_budget` → *(else)*
`_reject_reserved_keys` → `_dump_ctx` → `start_run_untriggered`.
New order: `_require_executor()` → `_reject_reserved_keys` → `_id()` → `_clock()` →
`executor.step_budget` → *(else)* `_dump_ctx` → `start_run_untriggered`.

Driven with counting `id_gen`/`clock` and a `NullRepo` that raises on any write, a rejected
process-path start now leaves the generators at `id1` / `1000` where `HEAD` would have consumed
them. `_default_id` = `uuid.uuid4().hex`, `_default_clock` = `int(time.time() * 1000)` — both
pure. No repository method was reachable before the guard in either version.

### Appendix D — Pass 2 mutation battery

Four run as `pytest` plugins on `PYTHONPATH` (no file touched); two required a copy-aside of
`services.py`, restored from the copy and confirmed byte-identical by `md5sum`.

| Mutation | Result |
|---|---|
| New start bound `>` → `>=` (M11) | **2 failed** — `test_a_run_ctx_exactly_on_the_bound_is_accepted_on_both_paths` (intended) + the timers boundary test (N-3's coupling) |
| `_chat_start_ctx` merge reversed | **1 failed** — `test_chat_start_ctx_anchor_wins_a_caller_key_collision`, and only that. In Pass 1 this same mutation left all 2330 green |
| Seam grows a non-reserved anchor key, constant **not** updated | **7 failed**, incl. `test_chat_start_anchor_key_set_is_exactly_the_module_constant` — the constant cannot drift out of step |
| Seam **and** constant both grow a non-reserved anchor key | **7 failed**, incl. `test_chat_start_anchor_keys_are_all_reserved` — the invariant itself has teeth |
| Sweep bound (`services.py:2505`) forced always-false | **1 failed** — `test_sweep_faults_a_candidate_whose_merged_ctx_would_exceed_max_config_len`. The sweep's bound is still genuinely covered |
| Sweep bound `>` → `>=` | 26 passed — **not** caught. Equally uncaught before the resize (old seed: 8011 ctx / 8031 merged), so no regression |

### Appendix E — live controls for the new `verify_salesperson.sh` check

`bash -n` clean. Both runs read-only; `reference` shows `MISSING` in both because my own suite run
wiped it (expected, `conftest.py::wf_repo`), which is orthogonal to the new check.

**Negative control — the real F-1 artifact, not a construction.**
`FALKORCHAT_SALESPERSON_DEF_VERSION=v6 ./scripts/verify_salesperson.sh acme` → exit 1, printing:

```
    ⚠ ws:acme snapshot: step 'assistant' config differs from proof_defs.py on systemPrompt
  ✗ salesperson@v6: ws:acme snapshot step 'assistant' DIVERGES FROM proof_defs.py
    (systemPrompt) — this version is create-only; bump the version
```

Right step, right field, right remedy — and `order-fulfillment@v1` in the same graph produced no
drift line, so it is not simply failing everything.

**Positive control.** `./scripts/verify_salesperson.sh s1v7` → **no drift line for either def**;
the only failures are the `reference`-missing pair. No false positive against the current file.

**Note for `teco`:** `reference` currently holds one unrelated test def (my suite run), so
`verify_salesperson.sh` reports both defs missing on the `reference` side until someone re-seeds.
I did not re-seed — that is a write to shared global state and not mine to make. Probe graphs left
in place, none created by me: `ws:s1v6`, `ws:s1v7`.

### Appendix F — the verbatim check, run three ways (Pass 3)

Re-extracted every ```` ```cypher ```` block from both documents and compared to the shipped class
constants (`.venv/bin/python`, `falkorchat.repository.Repository`):

```
§18 cypher blocks: 9; max lines: 77
_ENSURE_PARTICIPANT_CYPHER         len= 1707 exact-match-in-§18: True [block 0]
_RESET_PARTICIPANT_CYPHER          len= 3909 exact-match-in-§18: True [block 5]
_RESET_ALL_PARTICIPANTS_CYPHER     len= 3232 exact-match-in-§18: True [block 6]
_CURRENT_ORDER_CYPHER              len=  475 exact-match-in-§18: True [block 7]
_ORDER_OWNERSHIP_CYPHER            len=  172 exact-match-in-§18: True [block 8]

note cypher blocks: 6; max lines: 77
… verbatim-from-note: True   (all five)
```

Both the note→code claim (`teco`'s) and the code→`QUERIES.md` claim (not previously checked) hold
byte-for-byte. Block count 6 / 9 and max length 77 satisfy note §12's fence tripwire on both files.

### Appendix G — Pass 3 mutation battery and untested-path probes

**Seam.** A pytest plugin (`-p ablate`, loaded from a scratch dir via `PYTHONPATH`) rewrites
`Repository._RESET_*_CYPHER` on the class at import time. `md5sum falkorchat/repository.py` is
identical before and after every run; `git status` shows no new or changed file in the repo.

| Ablation | Tests reddened |
|---|---|
| G2 off, `reset_participant` | `…_of_a_cross_member_leaves_the_demo_channel_whole`, `…_keeps_a_cursor_on_a_surviving_thread`, `…_with_a_mismatched_marker_is_a_total_no_op` (3) |
| G2 off, `reset_all` | `…_leaves_the_non_participant_subgraph_intact`, `…_reports_unscoped_participants_and_leaves_them_whole` (2) |
| G1 off, `reset_participant` | `…_is_a_no_op_for_non_participants[u1]`, `[u2]` (2) |
| G1 off, `reset_all` | `…_deletes_every_participant_subgraph`, `…_reports_unscoped_participants…`, `…_is_idempotent_and_returns_an_all_zeros_row_when_clean` (3) |
| G1+G2 off, `reset_participant` | 5 |
| G1+G2 off, `reset_all` | 5, **including** `…_keeps_exactly_the_survivor_labels` |
| message walk author-scoped instead of thread-scoped | `…_clears_its_own_subgraph_and_remints_the_thread`, `…_is_thread_scoped_not_author_scoped`, `…_is_idempotent` (3) |
| **`reset_all` cursor sweep narrowed to `reset_participant`'s liveness-filtered form** | **none — 506 passed** (this is M-2) |

**Note §2.3 row B, reproduced in S4's own fixture** (`reset_participant('u2')`, both guards
stripped, control run alongside):

|  | `Channel` | `Thread` | `Message` | `ReadCursor` | `demo-welcome` | `dm1`/`dm2`/`dm3` |
|---|---|---|---|---|---|---|
| control (shipped) | 3 → 3 | 3 → 3 | 10 → 10 | 6 → 6 | alive | alive |
| **G1+G2 stripped** | **3 → 3** | **3 → 3** | 10 → 7 | 6 → 4 | **GONE** | **GONE** |

Every §4.8 survivor **label** is still present in the stripped run. Label-presence passes; the
identity assertions do not.

**Untested-path probes** (direct calls against `ws:test`, fixture built from the shipped test
helpers):

| Probe | Result |
|---|---|
| two `Channel`s carrying `participantId:'p-aaa'` → `reset_participant('p-aaa')` | raises `ResponseError: unique constraint violation on node of type Thread`; **57 → 57 nodes**, `th-p-aaa` alive, its 3 messages alive, `th-X` absent (M-3) |
| same graph → `reset_all_participants()` | cleans up: `userCount 2`, `ch-dupe` gone, `p-aaa` `User` gone, demo subgraph intact |
| **marker forging** — `ensure_participant(channel_id='demo-general')` | raises `ResponseError: unique constraint violation on node of type Channel`; **56 → 56 nodes**, `demo-general.participantId` still `null`, no `p-zzz` `User` |
| unscoped **way 2** (no `MEMBER_OF`) under `reset_all` | `unscopedCount 1`, `unscopedIds ['p-aaa']`, `User` and `ch-p-aaa` both left whole |
| cross-member cursor under `reset_all` | `p-ccc:demo-welcome` **deleted**, `u1:demo-welcome` and `assistant:demo-welcome` **kept**, `cursorCount 7`, zero unowned cursors, `demo-welcome` alive — the note's P2 asymmetry, correct but untested (M-2) |
| completely empty `ws:test` (0 nodes) | `reset_all` → one all-zeros row (no `IndexError`); `reset_participant` → `None`; `order_belongs_to_customer` → `{"owned": False, "status": None}`; `get_customer_current_order` → `None`; `list_participants` → `[]` |
| `set_participant_record(participant_id='p-aaa', thread_id='th-p-bbb')` | returns `{"threadId": "th-p-bbb"}`, graph agrees — M-1 |

**The plan's done-condition, run verbatim.** After `seed_catalog.sh` + `seed_salesperson.sh test`,
seeding the fixture and running `reset_all_participants("test")`:
`./scripts/verify_salesperson.sh test` → `RESULT: OK — 2 defs in sync`, **exit 0**.
`./scripts/verify_catalog.sh` → `Product count : 16 (expected 15) — MISMATCH`, exit 1 — caused by
the test-planted `prod1` (M-4), not by S4's code.

**Graph state left behind, for `teco`.** `ws:test` holds a post-`reset_all` fixture from my last
probe. `reference` holds the 15-product catalog **plus** the stray `Product {productId:'prod1'}`
and both `salesperson@v7` / `order-fulfillment@v1` defs (re-seeded by me, since the suite run
wiped them). No graph key was created or deleted by this pass.

### Appendix H — Pass 4 re-measurement (current tree, serial, no `-k` filter)

Same seam as Appendix G: a pytest plugin rewrites `Repository._RESET_*_CYPHER` on the class at
import time. `md5sum falkorchat/repository.py` identical before and after every run; `git status`
shows no new or changed file from this pass.

**The §18.0 citation table, re-measured (`tests/test_repository.py` + `tests/test_services.py`,
508 collected):**

| Ablated query | Result | Tests that go red |
|---|---|---|
| G1+G2 off, `_RESET_PARTICIPANT_CYPHER` | **5 failed, 503 passed** | `…_of_a_cross_member_leaves_the_demo_channel_whole`, `…_keeps_a_cursor_on_a_surviving_thread`, `…_is_a_no_op_for_non_participants[u1]`, `[u2]`, `…_with_a_mismatched_marker_is_a_total_no_op` |
| G1+G2 off, `_RESET_ALL_PARTICIPANTS_CYPHER` | **5 failed, 503 passed** | `…_deletes_every_participant_subgraph`, `…_leaves_the_non_participant_subgraph_intact`, `…_keeps_exactly_the_survivor_labels`, `…_reports_unscoped_participants_and_leaves_them_whole`, `…_is_idempotent_and_returns_an_all_zeros_row_when_clean` |

Disjoint, five each, and identical to the shipped `QUERIES.md` §18.0 table name for name.

**Single-guard ablations, unfiltered — all reproduce Pass 3's counts** (so the `-k` filter used in
Pass 3 was not under-counting): G2-off-mine **3**, G2-off-all **2**, G1-off-mine **2**, G1-off-all
**3**, author-scoped walk **3**.

**M-2 and M-3 re-checks:**

| Mutation | Pass 3 | Pass 4 |
|---|---|---|
| `reset_all` cursor sweep narrowed to the liveness-filtered form | 0 failed (the gap) | **1 failed** — `test_reset_all_leaves_the_non_participant_subgraph_intact` |
| re-mint `CREATE` → `MERGE` (the implementer's discarded ablation) | — | **0 failed** — equivalent mutant; MERGE cannot match a `Thread` hung off the other channel, so it creates and UNIQUE fires identically |
| re-mint `threadId` made unique per channel (**removes** the raise, lets the delete proceed) | — | **3 failed**, including `test_reset_participant_with_a_duplicate_marker_raises_and_writes_nothing` — M-3's test has teeth |

**Guard probes re-run for regression** (unchanged from Pass 3): duplicate marker → raises, 57 → 57
nodes, nothing written; `ensure_participant(channel_id='demo-general')` → raises on `Channel`
UNIQUE, 56 → 56, `demo-general.participantId` still `null`; unscoped-way-2 under `reset_all` →
`unscopedCount 1`, left whole; `reset_all` under a duplicate marker → cleans up, demo subgraph
intact.

**The done-condition, end to end, in the order `HISTORY.md` states it:** fixture →
`seed_catalog.sh` → `seed_salesperson.sh test` → `reset_all_participants("test")` (userCount 2,
channelCount 2, messageCount 6) → `./scripts/verify_salesperson.sh test` **exit 0** and
`./scripts/verify_catalog.sh` **exit 0**. Both halves green for the first time.

**Verbatim, re-checked after the §18.3 edit:** 9 fenced blocks in `QUERIES.md` §18, max 77 lines;
all five constants byte-identical to their §18 transcription and to graph note **v1.3**'s own
blocks.

### Appendix I — Pass 5 route-helper probes and mutation battery (all run, FastAPI 0.139 / Starlette 1.3.1 / Python 3.12.3)

**§1 — every registration shape, against the shipped `_route_paths`.** Reported / truth:

| Shape | Helper reports | Correct? |
|---|---|---|
| `include_router(r)` | `['/a']` | ✅ |
| `include_router(r, prefix='/shop/api')` | `['/a']` (truth `/shop/api/a`) | ❌ **P5-1** |
| router → router, two levels | `['/deep']` (truth `/mid/deep`) | ✅ presence, ❌ prefix |
| `APIRouter(prefix='/pref')` | `['/pref/b']` | ✅ (baked into `.path` at include) |
| `@app.get` · `add_api_route` · `@app.websocket` | `['/direct']` · `['/added']` · `['/ws']` | ✅ |
| `app.mount('/m', sub_app)` | `['/m']` (walk stops — documented) | ✅ |
| `starlette.routing.Host('evil.example', app=sub)` | `[]` | ❌ **P5-2** |

`_IncludedRouter` has **no** `.path` attribute (`getattr(route,'path','<none>') == '<none>'`), which
is why a broken traversal drops the router entirely rather than reporting one wrong path.

**§2 — the real `create_app`, built for real.**

| Configuration | `_registered_paths` | `Mount`s | `user_middleware` |
|---|---|---|---|
| default (`mount_mcp=True`, `web_dir=falkor-chat/web`) | 37 paths, `/health` ×1, `/channels` ✓, `''` ✓, `/mcp` ✓ | `[('/mcp', None), ('', 'web')]` | `['_McpPathAlias']` |
| `mount_mcp=True, dev_surface=False` | `['/health']` | `[]` | `[]` |

**§3 — the vacuity mode, and the control that closes it.** The helper re-run with its traversal
attribute renamed (`original_router` → `SOME_FUTURE_NAME`), everything else identical:

| Traversal attr | default paths seen | positive control passes | `dev_surface=False` assertion passes |
|---|---|---|---|
| `original_router` | 37 | **True** | True |
| `SOME_FUTURE_NAME` | **2** | **False** | **True** |

The second row is the finding-shaped failure: the `dev_surface=False` assertion is *still green*
while asserting nothing. Only the positive control goes red. (The 2 survivors are the `/mcp` and
`''` mounts, which carry their own `.path`.)

**§4 — P5-1's suggested fix, run.** Threading `prefix + include_context.prefix` through the walk:
default app still 37 paths / 1 `/health` / `/channels` ✓ / `''` ✓; `dev_surface=False` still
`['/health']`; S8-shaped app now `['/health', '/shop/api/join']` = truth; two-level nested include
now `['/top/mid/deep']` = truth. All four S3 assertions unchanged.

**§5 — mutation battery** (`app.py` copied aside, mutated, targeted run, restored each time;
`md5sum` `fe8102d7ef2d4846b44860214a240a43` before and after, and the full suite green afterwards).
Baseline for every row: **10 passed, 33 deselected**.

| # | Mutation | Result | Killed by |
|---|---|---|---|
| M-a | delete `mount_mcp = mount_mcp and dev_surface` | 1 failed | `…_registers_no_legacy_router_no_web_mount_and_no_mcp` |
| M-c | `if web.is_dir():` — `/` mount ignores `dev_surface` | 1 failed | same |
| M-h | `if True:` — legacy router mounted regardless | 1 failed | same |
| M-b | bare `/health` registered in **both** configurations | 2 failed | `…_with_one_health_route` **and** `…_no_web_mount_and_no_mcp` |
| M-f | bare `/health` returns 200 without calling `services.ping` | 1 failed | `…_health_reports_503_when_falkordb_does_not_answer` |
| M-d | `_build_default_app` **plain** path → `mount_mcp=True` | 1 failed | `…_derives_both_switches…[plain-app]` |
| M-e | `_build_default_app` **workflow** path → `dev_surface=True` | 1 failed | `…[workflow-app]` |
| M-g | `_build_default_app` **responder** path → `dev_surface=True` | 1 failed | `…[responder-app]` |
| M-i | `WorkflowTrigger` ignores `TRIGGER_RESPONDER_FALLTHROUGH` | 1 failed | `test_trigger_responder_fall_through_is_gated_on_its_own_flag` |

M-a/M-c/M-h are the partial-`dev_surface` cases: each of the three surfaces, broken alone, dies on
the one exact-list assertion.

**§6 — suite and lint, re-run by me, serially.** `pytest -q` → **2391 passed, 14 deselected**
(16.8s). `ruff check falkorchat/app.py falkorchat/config.py tests/test_app.py` → **All checks
passed** (the 34 errors a repo-wide `ruff check falkorchat/ tests/` reports are all in files S3 did
not touch).

### Appendix J — Pass 6 evidence (commit `2f7938d`, live `ws:test`, run serially)

**§1 — the cache's readers, enumerated rather than eyeballed.**

```
$ grep -n '_records' falkorchat/storefront.py
249,250  init + lock          438,439  lookup()      <- READ
455,456  cached_ids()  <- READ 467,468  forget_all()
471,472  _cache_put()         475,476  _cache_drop()
$ grep -rn 'lookup(\|cached_ids(\|forget(' falkorchat/ | grep -v storefront.py
(no output)
```

`resolve_token` appears at none of the six access sites; it reaches the map only through
`_cache_put`/`_cache_drop`.

**§2 — mutation battery.** `storefront.py` and `test_storefront.py` byte-copied aside, mutated,
run, restored; `md5sum -c` clean after every row (`da2cb8cd…`, `3bfeb9f9…`). Baseline **45
passed**.

| # | Mutation | Result | Killed by |
|---|---|---|---|
| S6-a | `resolve_token` stops calling `_cache_put` | **45 passed — SURVIVES**; and **2439 passed / 14 deselected** across the whole repo | *nothing* → **S6-1** |
| S6-b | `resolve_token` stops calling `_cache_drop` on a missing row | 1 failed | `test_a_deleted_participant_stops_resolving_immediately` |
| S6-c | `hmac.compare_digest(…)` → `stored_hash != hash_token(token)` | 1 failed | `test_the_token_comparison_is_constant_time` **only** — no functional test sees it, exactly as its docstring says |
| S6-d | cache-first `resolve_token` (the commit message's claim) | **5 failed** | `…_idempotent_when_the_participant_id_repeats`, `…_wrong_token_for_a_real_participant…`, `…_one_participants_token_never_resolves_under_anothers_id`, `…_deleted_participant_stops_resolving…`, `…_reads_the_graph_on_every_call` |
| S6-e | `_PACKAGE_DIR` → `falkorchat_RENAMED` (nonexistent) | **45 passed — SURVIVES** | *nothing* → **S6-3**. `rglob` on a missing dir yields 0 files and raises nothing (27 files normally) |
| S6-f | `_CONFIG_SOURCE = ""` (control for S6-3's neighbour) | 1 failed | `test_config_reads_exactly_the_documented_storefront_env_vars` — so `test_dev_surface_has_no_environment_variable` *is* controlled |

**§3 — S6-1's consequence, executed.** Join → resolve → `repo.set_participant_record(…,
language="es")` → resolve again → read `lookup()`, the accessor S9's workers and S7's post-reset
write are pointed at:

```
shipped code:  resolve_token -> 'es'      lookup() -> 'es'
S6-a mutant :  resolve_token -> 'es'      lookup() -> 'en'    (stale, and the suite is green)
```

**§4 — S6-4's suggested spy, executed.** `storefront.hmac.compare_digest` swapped for a recording
wrapper, one valid token resolved: **1 call**, args exactly `(stored_hash, hash_token(token))`,
resolution still succeeds. Behavioural, formatter-proof, and red if a branch ever skips the call.

**§5 — plan greps behind S6-2** (against `git show acb5a2a:docs/plans/salesperson-ui.md`, v1.16):
`presenter_configured` → **0 hits**; `unset key|empty key|unconfigured|compare_digest("", "")` →
**0 hits**; `compare_digest` → 1 hit (line 372, the *participant* token). S10's Done-condition
column names a *wrong* key, never an unset one.

**§6 — suite and lint.** `pytest -q` on the pristine tree: **2439 passed, 14 deselected** (17.6 s),
matching your solo run. `ruff check falkorchat/storefront.py falkorchat/config.py
tests/test_storefront.py` → **All checks passed**. `git status --porcelain falkor-chat/` → clean,
and all three S6 paths `git diff --quiet 2f7938d`-identical.

### Appendix K — Pass 7 evidence (commit `dd78e70`, live `ws:test` + `reference`, run serially)

**§1 — the catalog workaround against the *real* 15-product catalog.** `./scripts/seed_catalog.sh`
→ 15 products, `verify_catalog.sh` **OK**. Then `Storefront.list_catalog()` on the live
`reference`, with each row's `productId` compared against `MATCH (p:Product) RETURN p.name,
p.productId`:

```
rows resolved by the name-join workaround: 15 of 15
unique ids: 15
name->id mis-bindings: []
products dropped by the join: []
ids: gaming-mouse-pad-xl, wireless-charging-pad, wireless-mouse-pro, laptop-stand-aluminum,
     usb-c-hub-7-in-1, bluetooth-speaker-mini, webcam-hd-1080p, fitness-tracker-band,
     smart-home-hub, mechanical-keyboard-k200, portable-ssd-1tb, action-camera-4k,
     noise-cancelling-headphones-x3, smartwatch-series-5, 27-inch-4k-monitor
```

The join is exact because both sides call the same function: `seed_catalog.sh:75`
`from falkorchat.extraction import normalize_name`, written to `nameNormalized` at seed time and
re-applied by `services.lookup_product` at read time. The fixture `_catalog_rows(n)` satisfies that
by construction (`"Widget 001"` / `"widget 001"`), which is why this live check was needed.

**§2 — blast radius of Ruling 1's one-line fix, measured.** Applied to
`repository.filter_products`: `RETURN p.productId AS productId, p.name AS name, …` plus the
matching row-mapping shift. Full suite, serially:

```
2473 passed, 14 deselected, 1 warning in 19.09s     # zero failures, zero test edits
```

`test_repository.py` asserts `[r["name"] for r in rows]` / `{r["name"] for r in rows}` — key
projections, not exact-dict equality; `test_tools.py` drives `FilterProductsTool` against a stub
whose rows the caller supplies. Neither sees the added key. `repository.py` restored, `md5sum -c`
OK.

**§3 — mutation battery** (byte-copies aside; `md5sum -c` clean after every row;
`storefront.py` `da2…`-family, restored). Baseline **79 passed**.

| # | Mutation | Result | Killed by |
|---|---|---|---|
| M1 | `reset_participant` drops the `_await_quiesce` call entirely | 2 failed | `…_waits_for_an_in_flight_turn_before_it_deletes`, `…_quiesce_timeout_changes_nothing…` |
| M2 | **vacuity probe** — worker `join()`ed *before* the reset, so nothing is left to wait for | **4 passed — SURVIVES** | *nothing* → **S7-1** |
| M3 | `_await_quiesce` deadline extended by an hour | **hangs** (killed at 25 s) | detected, but as a hang, not a failure → **S7-3** |
| A5 | `_reset_state_unknown` stops dropping the stale cached record | **79 passed — SURVIVES** | *nothing* → **S7-2** |
| A6 | success path stops refreshing the cached record | 1 failed | `test_reset_refreshes_the_cached_record_so_lookup_never_serves_a_dead_thread` |
| A7 | `status is None` path stops dropping the cache | **79 passed — SURVIVES** | *nothing* → **S7-2** |
| A9 | `CATALOG_LIMIT` 500 → 20 | 1 failed | `…_carries_an_explicit_bound_past_the_delivered_default` |
| B1 | *benign* — extract a local for the image URL | 79 passed | (correctly green) |
| B2 | *benign* — quiesce loop rewritten as `while True`, same semantics | 79 passed | (correctly green) |

**§4 — is the quiesce test flaky, or does it actually wait?** Instrumented the reset without
changing its behaviour:

```
RESET BLOCKED FOR 0.190s / 0.189s / 0.189s      # worker sleeps 0.15s; budget is 5s
```

So it genuinely waits, with a ~33× margin to the budget and a 0.039 s margin over the worker's
sleep. Stability: `test_storefront.py` **79 passed** on 5 consecutive whole-file runs (random
order, as the default run uses) and the quiesce test alone **passed 15/15**. Non-flaky — but M2
shows the wait is not what the assertions test.

**§5 — the `catalog_repo` teardown's load-bearing claim, verified live.**

```
reference indexes before/after MATCH (n) DETACH DELETE n:  4 / 4     nodes now: 0
```

Schema survives the data wipe, so the teardown cannot strand a later test needing the `Product`
index/constraint pair.

**§6 — `lookup` callers, grep-complete.** `grep -rn '\.lookup(' falkorchat/ tests/` → **8 hits, all
in `tests/`** (lines 391, 396, 399, 458, 464, 487, 1317, 1321); **zero in the package**. The
definition is `storefront.py:567`.

**§7 — the LLM-context claim behind Ruling 1.** `tools.py:428` —
`return json.dumps({"found": True, **row})`, where `row` is `services.lookup_product`'s output,
which has projected `productId` since K-053 (`repository.py:2681`). `pytest -m live --collect-only`
lists the 14 deselected tests: AC-5 chat grounding, 10 × querygen NLQ, and one triage workflow —
**none is a salesperson catalog conversation**, so no harness observes this either way.

**§8 — suite and lint.** Pristine tree, serially: **2473 passed, 14 deselected** (18.1 s).
`ruff check falkorchat/storefront.py tests/test_storefront.py` → **All checks passed**.

### Appendix L — Pass 8 evidence (commit `d9d2f2b`, live `ws:test`, `.venv/bin/python -m pytest`)

Every mutation below was applied to a byte-copy of the file, run, and reverted; both files were
`md5sum -c` verified against `8098db988561145ba07f6f16711bf125` (`tests/test_storefront.py`) and
`47ffe3abe13aafcad552c5f836a53921` (`falkorchat/storefront.py`) after each block, and
`git status --porcelain falkor-chat/` is empty. No tree-mutating git was run. Clean baseline:
**82 passed in 1.15 s** for the file; `ruff check` on both S7 files → **All checks passed**. The
whole-suite figure (2476 passed / 14 deselected) is taken as given from the dispatch, not re-run.

**§1 — S7-1, the adverse ordering.** `turn.join(timeout=5)` moved *before* the `_call_bounded` call,
so the turn is finished and there is nothing to wait for:

```
FAILED test_the_reset_waits_for_an_in_flight_turn_before_it_deletes
E  AssertionError: the reset was not issued while the turn was in flight …
E  assert 53950.693443919 < 53950.69338935
1 failed, 3 passed, 78 deselected in 0.57s
```

Exactly one test, on the **first** assertion, three quiesce tests still green — the implementer's
claim, reproduced. Detection margin at the moment of failure is 54 µs, but it is monotone-safe: any
extra delay in that ordering pushes `started_at` *later*, never earlier.

**§2 — S7-1, attacking the substitute.** Three probes.

| Probe | Result |
|---|---|
| `_await_quiesce` → `return True` (no wait at all) | **red** on the 2nd assertion, `53967.415 >= 53967.568` false — margin 0.153 s |
| `return True` **+** 300 ms sleep injected in `run()` before `fn(...)` (emulating a descheduled daemon thread) | **1 passed** — a false green, S8-1 |
| the same, with `started_at` re-stamped as the first line of `run()` | **red**, `53991.478 < 53991.341` false; with only that change and no mutant, 10 consecutive runs of the 7 reset/quiesce tests → 7 passed each time |
| `TURN_WORK_S = 0.0` alone | green ×5 |
| `TURN_WORK_S = 0.0` **+** `return True` | **red 8/8** — detection does not depend on the stub sleep |

**§3 — S7-3, both forms against the deadline mutant** (`deadline = … + self._quiesce_s + 3600`):

- Pass 7's suggested form — `t0 = time.monotonic()` / `with pytest.raises(QuiesceTimeoutError): shop.reset_participant(record)` / `assert time.monotonic() - t0 < 1.0`, on
  `test_a_quiesce_timeout_changes_nothing_and_leaves_the_turn_running`: `timeout 30` → **Terminated,
  `real 0m32.031s`, EXIT=143**. The suggestion cannot work; the implementer's structural argument is
  correct.
- `_call_bounded` as shipped, **whole file**: `2 failed, 80 passed in 3.34 s`, both names printed
  (`…quiesce_timeout…`, `…two_reset_failures…`), no hang. `grep -rl` over `tests/` shows only
  `test_storefront.py` and `test_repository.py` mention `reset_participant`, and the latter's are
  `Repository.reset_participant` — no quiesce, not hang-prone. Coverage complete.

**§4 — S7-2, the two eviction mutants and the parametrization's worth.**

| Mutant | Result |
|---|---|
| `_cache_drop` removed from `_reset_state_unknown` | **2 failed** — `…times_out_evicts…[succeeds]` and `[times-out-too]` |
| `_cache_drop` removed from the `status is None` branch | **1 failed** — `…finds_no_participant_evicts…` |
| `_cache_drop` **moved after** `get_state(ctx)` inside the `try` | **1 failed — `[times-out-too]` only** |

The third is why the parametrization is not decoration.

**§5 — measured bounded-call durations** (a `sys.stderr.write` added to `_call_bounded`, reverted):
188.028 ms (the waiting test, bound 10 s), 0.142 ms, **2.435 ms** (the idle reset — the only one that
writes), 0.140 ms (bound 1.0 s each). Leak probe: `IMMEDIATE_S = 0.0001` → `1 failed, 81 passed`,
the failure confined to `test_an_idle_participant_is_not_made_to_wait`, no downstream contamination.

**§6 — ruled out.** A blind-sleep `_await_quiesce` (`time.sleep(0.5); return True`) satisfies both of
the new ordering assertions — and would satisfy the duration floor too — but the file catches it
anyway: `2 failed, 80 passed in 9.14 s`, the two `quiesce_s=0` tests, because a reset that sleeps
blindly then succeeds never raises `QuiesceTimeoutError`. Not a finding.

### Appendix M — Pass 9 evidence (commits `6fbe541` … `83af07c`, live `ws:test` + `reference`)

Every mutation was applied from a byte-copy and reverted; `falkorchat/repository.py`,
`falkorchat/storefront.py`, `tests/test_storefront.py`, `tests/test_repository.py` and
`falkor-chat/scripts/test_queries.sh` all `md5sum -c` OK afterwards and `git status --porcelain` is
empty apart from this review. No tree-mutating git. Clean baselines: `test_storefront.py` +
`test_repository.py` + `test_tools.py` → **408 passed**; `./scripts/test_queries.sh` → **408/408**.
The whole-suite figure (2478 / 14 deselected) is taken as given from the dispatch.

**§1 — S7b2, the three claims about `_call_bounded`.**

| Probe | Result |
|---|---|
| Pass 8's own probe: `_await_quiesce → return True` + 300 ms injected before the (now in-thread) stamp | **red** on the first assertion, `56861.034 < 56860.896` false — S8-1 fixed |
| adverse ordering (worker joined before the reset) | **red**, `56849.45616648 < 56849.456024078` — margin **142 µs** (Pass 8 measured 54 µs) |
| hang mutant (`deadline + 3600`), whole file, with the two sites now `with pytest.raises(QuiesceTimeoutError):` | **2 failed in 3.31 s** — the bound's `AssertionError` escapes `pytest.raises`; S7-3 intact |
| bounded call **raises *and* overruns** (`sleep(2.0)` before `raise QuiesceTimeoutError`, bound 1.0 s) | **red on the bound**, "did not return within 1.0s" — the breach wins, the exception is not swallowed |
| blind-sleep mutant (`time.sleep(2.0); return True`) at `IMMEDIATE_S = 1.0` | **3 failed**, incl. `test_an_idle_participant_is_not_made_to_wait` |
| same mutant, idle site widened to `seconds=10` | **1 passed** — the detection is lost; the refusal to widen is correct |

**§2 — S7c, the tripwire under six mutants** (`test_storefront.py` + `test_repository.py`):

| # | Mutant | Result |
|---|---|---|
| M-A | projection reverted to `{name, category, price}` (both halves of `repository.filter_products`) | **12 failed** — incl. the tripwire and the new repo key-set test |
| M-B | projection kept, S7's `1 + n` loop restored via `services.lookup_product` | **1 failed** — the tripwire **alone**, on `_CatalogSecondRead` |
| M-C | projection kept, `_catalog_rows` fabricates the id as `name.lower().replace(" ", "-")` | **1 failed** — the tripwire alone, on the id-list equality |
| M-D | M-C **plus** the tripwire's own first draft (no `opaque-sku-42` override) | **6 passed — SURVIVES**, reproducing the false negative the implementer found and fixed |
| M-E | projection reverted **and** the `1 + n` restored through `self._repo.lookup_product` | `test_storefront.py` **83 passed**; caught only by `test_repository.py`'s key-set test |
| M-F | projection **kept**, `1 + n` **and** the silent-drop `continue` restored through `self._repo.lookup_product` | **348 passed — SURVIVES entirely** → **S9-1** |

S9-1's fix, verified: adding `monkeypatch.setattr(stocked._repo, "lookup_product", _boom)` reddens
M-F and leaves 83 passed with no mutant.

**§3 — the removed silent-drop branch.** `CALL db.constraints()` on `reference` returns
`['UNIQUE', 'Product', ['productId'], 'NODE', 'OPERATIONAL']` — unique, **not** mandatory, so a
`Product` without `productId` is accepted. Created one live and called `list_catalog()`:

```
shipped: [{'productId': None, 'name': 'Ghost Widget', ..., 'imageUrl': None}, {'productId': 'real-1', ...}]
S7 code: [{'productId': None, 'name': 'Ghost Widget', ..., 'imageUrl': None}]
```

Byte-identical row for the same node under the reconstructed S7 code (M-A + M-B applied together),
so the null-`productId` exposure is pre-existing and untouched, and the deleted branch guarded a
different condition (a *name* that failed to re-resolve between two reads) that no longer exists.

**§4 — a widened projection is caught.** Adding `p.categoryNormalized` to `filter_products`'s
`RETURN` → **3 failed**: `test_list_catalog_returns_all_fifteen_rows`, the tripwire's key-set
assertion, and the new repo key-set test. Neither key-set assertion is decoration.

**§5 — §15 fidelity, checked with an independent comparator.** A comparator written for this pass
(not the implementer's, not `teco`'s) parses `repository.py` with `ast`, walks each method for its
`ro_query`/`query` call and `literal_eval`s the concatenated first argument, then whitespace-
normalizes and compares against the `LOOKUP`/`FILTER` shell constants and against the fenced
`cypher` block under each `### 15.x` heading (comment lines stripped):

```
lookup_product   code==shell: True   code==docs: True
filter_products  code==shell: True   code==docs: True
ALL FOUR CELLS AGREE: True
```

Coupling, three directions, each a full gate run: `FILTER` constant narrowed alone → **407/408**;
`FILTER` abstention header narrowed alone → **407/408**; `LOOKUP` constant narrowed alone →
**407/408**. The gate does self-check its own `RETURN` list — and, as follow-up 16 says, cannot see a
constant and header that are wrong *together*, which is exactly the state `LOOKUP` sat in from
`bcd2dcc` (2026-08-28) to S7c3 while reporting 408/408.

---

## Pass 10 — 2026-09-03 (S8: the `/shop/api` router, the error map and the two-half gate)

**Reviewed:** commit **`81a1268`** — `falkorchat/storefront_api.py` (new, ~1090), `app.py`
(+102/−2), `schemas.py` (+6/−2), `tests/test_storefront_api.py` (new, ~1900, 99 node ids),
`tests/test_app.py` (+159/−4) — against `docs/plans/salesperson-ui.md` **v1.19** (`732f5e0`)
§5.1 S8, §5.2, §5.3, §4.7, §4.9, §6.2, and against `docs/reviews/salesperson-ui.md` `## Pass 8`'s
stopping rule, which named this gate as its payoff. Not reviewed: S9/S10/S12 content, the SPA,
`storefront.py` (byte-untouched — confirmed by `git diff --stat 81a1268^ 81a1268`, empty).

**CPG: considered, not relevant — `cpg_falkorchat` predates `storefront.py` and models none of
`storefront_api.py`, which this commit creates; every claim below is from direct read, execution
against the live FalkorDB, or source mutation.**

**Verdict: needs changes** — **2 blockers, 3 majors, 4 minors, 3 nits**. The gate is real: I ran all
eight failure demonstrations, and I could not weaken it from inside its own subject. But its handler
half enumerates the wrong set, and the response that slips through is not hypothetical — I reproduced
two of them against the delivered app. **The good news is that both blockers are small and land in
the same place**, and the gate's own machinery is what makes the fixes checkable.

### What I ran (all of it, solo)

Full suite **2582 passed / 14 deselected** (20.8 s) — matches the brief's figure, re-derived, not
inherited. `tests/test_storefront_api.py` alone: 99 passed, 3.6 s. **21 source mutations** of
`storefront_api.py` and `app.py`, each applied to a byte-restored copy and reverted after
(`md5sum` re-checked: `storefront_api.py` `edac9f1a…`, `app.py` `e6bf735a…`,
`test_storefront_api.py` `0374ffd2…`; `git status` on `falkor-chat/` clean). Four throw-away probe
files under `tests/`, all removed. Full mutation ledger: **Appendix P10-A**.

**Database.** `ws:test` and `reference` only. **`ws:acme` was never touched** — its node inventory
is identical before and after (14 labels, `Message` 52, `Entity` 544, `WorkflowRun` 21). See
*Environment note* at the end: `reference` was **already** stripped when I arrived, which the brief
did not know.

### The judgment you asked me to rule on — the S8/S10 fork

**Correctly not escalated, and the boundary is drawn in the right place.** Three independent pieces
of evidence, none of which is the implementer's own argument:

- **The gate is not evaluable on a partial surface, by its own construction.** `evaluate_gate`'s
  first check is `set(storefront_routes(app)) == set(ROUTE_CLASSES)`, and `ROUTE_CLASSES` *is*
  §5.3's eleven-row table. Eight or nine routes fails that line before either half runs. The
  declaration half is `{declared} ∪ {handler-produced} == FLAT_TABLE`, read off `app.routes` — a
  set equality over the whole table, which cannot be restricted to a subset without editing the
  table the gate exists to check.
- **S8's own done-condition names the route.** "*The two `no graph access` routes are asserted
  negatively*: … `GET /shop/api/health` **and `POST /shop/api/presenter/session`** still answer
  their normal `200`/`403`/`422`." A step whose done-condition names a route cannot be done without
  it. §6.2's `S8/S10` bullet hedges the same way; §5.1's S10 row is the only place that reads
  otherwise, and its Files column already contains `storefront_api.py`.
- **The strongest evidence is a negative one:** `storefront.py` — S10's *other* file, and the home
  of `presenter_login`/`list_participants`/`reset_all` — is byte-identical to its parent. The
  implementer built the three routes without touching the file S10 owns, which is exactly the line
  a scope fork should be held at.

**Is anything of S10's *content* pre-empted?** Only what could not be avoided. Absent, correctly:
the login's fixed per-attempt delay, the observational attempt counter, and reset-everyone's
**stop-intake** flag (`Storefront` state). Present-but-S10's-to-move: the constant-time tripwire
extension to `presenter_login` (S6-4) and the roster/`incomplete` assertions — all tests *of code
S8 wrote*, so they had to exist now. The S7-ships-the-wait/S9-ships-the-cancellation parallel holds:
`Storefront._await_quiesce` (`storefront.py:842`) documents the identical "waiting subsumes
cancelling for correctness" reasoning, and the router's reset-all drain is its per-roster twin.

**`_STEP_10_INTERIM` is right on the boundary and wrong on the inventory** — see **P10-6**. Cost to
S10: one re-derivation, not a rewrite.

### Blockers

#### P10-1 · **Blocker** · the handler half enumerates the *delta* against a baseline app, not the handlers on the app — and the twelve it subtracts include the one that provably fires

`registered_storefront_handlers` (`tests/test_storefront_api.py:203`) computes
`{handlers on this app} − {handlers on create_app(dev_surface=False)}`. §5.1 S8 asks for "the
handlers **actually registered on the app object**". The storefront app carries **17**; the gate
sees **5**. Subtracted: `ServiceError` and eleven workflow/search handlers — all registered by
`app.py:81`'s `_register_error_handlers`, which runs on every app including this one.

`ServiceError` is not inert on `/shop/api`. `services.post_message` →
`_validate_and_derive_role` raises `ThreadNotFoundError` / `UnknownActorError` / `UnknownMemberError`
(`services.py:833-846`), and `Storefront._await_quiesce`'s own docstring names the reset window in
which the first of those fires. Reproduced against the delivered app (Appendix P10-A, probe 3):

```
POST /shop/api/messages -> 404 {"error":"ThreadNotFoundError","detail":"th-swept-away"}
```

`404` is **not** in that route's `responses={…}` (200/401/409/422), **not** in `TABLE`, and produced
by a handler the gate subtracts — so all three of the gate, the declaration half and
`test_no_route_can_raise_a_refusal_it_does_not_declare` are silent. This is the unruled
`(route, response)` §5.3 spent eight passes closing, arriving from the server. C13 does shout, which
is the designed backstop working — but "the error map is total by type" is the claim this step
exists to make true, and as delivered it is not.

**Suggested fix (expect to beat it):** stop subtracting. Enumerate `app.exception_handlers` whole and
require every entry to be classified — `CROSS_CUTTING_HANDLERS`, `ENVELOPE_HANDLERS`, or a new
`INHERITED_HANDLERS` frozenset that must state, per type, either "cannot fire on a `/shop/api`
route" or the `(route, response)` rows it adds to §5.3's table. The `ServiceError` family needs the
second: at minimum `POST /shop/api/messages` gains rows for the reset-window refusals. A cheaper
90 % version is a single storefront-scoped `ServiceError` handler re-shaping them into the
storefront envelope with a plan token, so the client keys on a token rather than a Python class name.

#### P10-2 · **Blocker** · `POST /shop/api/session` answers a **bare `500`** for a condition `storefront.py` documents as a `503` — the unreachability argument is boot-time only

`DemoNotSeededError` (`storefront.py:103`) is the one `StorefrontError` subclass of seven with no
handler and no route `except`. Its own docstring says "**Maps to `503`**, naming `seed_demo.sh`".
The `join` route documents the omission and argues structural unreachability (preflight asks the
identical question; the demo `Agent` survives both resets). **The preflight is a boot-time check,
not an invariant.** Reproduced (Appendix P10-A, probe 2) — same app, lifespan entered, `Agent`
deleted out of band afterwards:

```
before delete -> 200
after  delete -> 500 Internal Server Error      # plain text, not even JSON
```

S8's done-condition says "**no route anywhere answers a bare `500`**", asserted only against a
stubbed repository, so the sweep never reaches this. This is also the plan's own C6b posture one
exception over: `409 unscoped_participant` is carried precisely because a graph can be unhealthy in
ways the storefront did not cause.

**Suggested fix:** catch it in `join` like the other six — `503 demo_not_seeded` (nothing was
written, so C9's "nothing changed" is exactly right), declare it, add the §5.3 row; the AST check
then picks it up for free. **And close the family structurally**, so the next one cannot hide: a
guard asserting every `StorefrontError` subclass is either caught by a route or handler-mapped. I
wrote and ran it — it flags `DemoNotSeededError` and nothing else (Appendix P10-A, probe 4).

### Majors

#### P10-3 · **Major** · `RequestValidationError` is classed as an envelope handler, so the one handler whose route set is *derivable* is excluded from the cross product

`ENVELOPE_HANDLERS` "contribute nothing to the `{handlers} × {routes}` cross product; what covers
them is the declaration half plus the per-route contract tests". For `StorefrontHTTPError` that now
holds (`test_no_route_can_raise_a_refusal_it_does_not_declare`). For `RequestValidationError` it does
not: it is produced by the framework, not by a route body, so no AST check sees it, and the
declaration half only compares two hand-written enumerations that can be wrong together.

**Mutation M-C, survived:** give `GET /shop/api/catalog` a `page: int = Query(1, ge=1)`. The route
now produces `422 validation_failed`, undeclared and untabled — **99 passed**.

**Suggested fix, verified before writing it.** `422`-producibility is mechanically derivable from
the route object: `route.body_field is not None or dependant.query_params or dependant.path_params`
(recursing into sub-dependencies). Run against the delivered app it reproduces **exactly** the five
routes that declare `422` and exactly the six that do not; asserted as
`{(m, p, 422, "validation_failed")} == {422 rows of FLAT_TABLE}` it is green today and red under
M-C. Full listing and both runs in Appendix P10-A.

#### P10-4 · **Major** · §4.8 F8's *second* ordering on reset-all is code without a test, and the obvious test cannot reach it

`presenter_reset_all` catches the sweep's `TimeoutError`, re-reads the roster, and has an inner
`except redis_exceptions.TimeoutError: unresolved = None` for the re-read failing too. That inner
branch is **untested** — mutation M-N (`unresolved = None` → `raise`) leaves **99 passed**. It is
S10's done-condition verbatim ("*a stub whose re-read **also** raises `TimeoutError` still returns
`504`, with no state body, never a `500`*"), but S8 shipped the code.

**And it is unreachable by the obvious stub**, which is this coordination's signature defect for the
third time: `_FailingMethodRepo` fails one method for all calls, so failing `list_participants`
breaks the *pre-drain* roster read (line 976) and the request never reaches the inner branch. The
test needs a repo whose `list_participants` raises **only on the second call**.

**Suggested fix:** a `_FailingAfterNCalls` variant, asserting `504`, `body["error"] ==
"reset_state_unknown"`, **`"participants" not in body`** (or `is None`) and `repo.calls == 1` on the
sweep. Whoever writes it should first confirm it goes red without the inner `except` — the branch
under test is the one an over-general stub silently skips.

#### P10-5 · **Major** · two declared rows have no producer, and the file's own claim that they do is what would stop the next reviewer looking

The module docstring narrows the acknowledged residue with: "*every declared entry is proved
producible by a contract test below, so the declaration and the implementation disagree loudly
instead of agreeing by omission*". Two counter-examples, both mutations that survive:

- **M-D** — delete `reset`'s `except UnknownParticipantError` branch entirely: **99 passed**.
  `(404, "unknown_participant")` is declared on `POST /shop/api/reset` and sits in `TABLE`; nothing
  produces it.
- **M-V** — delete `advance_order`'s `except UnknownOrderError` branch: **99 passed**. That escape
  is a `StorefrontError`, i.e. **a bare `500`** (P10-2's family).

The pairing is a convention, not a mechanism: nothing links a declared row to a test. **Suggested
fix:** either write the two missing contract tests (a `_FailingMethodRepo` raising
`UnknownParticipantError` from `reset_participant`; one raising `UnknownOrderError` from the
advance), or — better, because it does not decay — make the link mechanical by tagging each contract
test with the `(method, path, status, token)` it proves and asserting the tag set covers
`FLAT_TABLE`. That turns "proved producible" from a docstring into the gate's third half.

### Minors

- **P10-6 · `_STEP_10_INTERIM` miscounts the reads it hands over.** It says "*the three private
  reads in `build_storefront_router` → deleted with them*". There are **four** (`repo`, `services`,
  `agent_id`, `presenter_key`, lines 451–454) and exactly **two** go with S10: `repo` (used only at
  921/976/988/1004, all presenter routes) and `presenter_key` (only at 580). `services` is used by
  `GET /messages`, `POST /messages` and `/order/advance`; `agent_id` by `POST /messages`. The
  adjacent comment's "*S9/S10 delete every use of them*" is unestablished for `services`: neither
  S9's nor S10's row moves `read_messages` or `get_current_order` onto `Storefront`. Fix: say
  "two of the four", name them, and say `services`/`agent_id` stay.
- **P10-7 · the `welcome` fallback is untested and unspecified.** M-O (`WELCOME.get(language,
  FALLBACK)` → `WELCOME[language]`) survives: **99 passed**. `WELCOME` covers exactly
  `config.STOREFRONT_LOCALES`'s default `("en","pt-BR","es")`, so the fallback is only reachable
  through `FALKORCHAT_STOREFRONT_LOCALES` — a real deployment knob. One test joining under a
  configured locale absent from `WELCOME`, asserting the `en` line; plus the §5.2 spec line P10-R6
  asks for.
- **P10-8 · `cross_cutting_response` raises `KeyError` where its sibling path shouts.**
  `_cross_cutting_json` handles `answer is None` with a logged, conservative `504`; an unclassified
  `(method, path)` instead raises `KeyError` **inside an exception handler**. Unreachable today only
  because `app.py:408`'s bare `/health` catches `Exception` broadly — a property of another module.
  Fix: `ROUTE_CLASSES.get(...)` and route a miss down the same loud path.
- **P10-9 · a `writes` route's *pre-write* read failures are reported as "may have committed".**
  `presenter_reset_all`'s roster read (line 976) sits outside the `try`, so a `TimeoutError` there
  yields `504 reset_state_unknown` when nothing was attempted; `_TIMEOUT` freezes that as expected
  without naming it. The same holds for `get_participant`'s resolve on all five writing routes.
  Conservative and licensed by §5.3's class map — C4's action is a safe re-read — but the test that
  encodes it should say *why* it is right, or the next reader will read it as an attribution.

### Nits

- **P10-10** · `_PresenterSessions.verify`'s constant-time claim is unpinned — M-P (`compare_digest`
  loop → `token in candidates`) survives, **99 passed**. Note also that `any()` short-circuits, so
  the loop is constant-time *per comparison*, not per call; the docstring's "costs the same time" is
  true for a wrong token and not for a right one. Harmless (the response already reveals validity),
  but the sentence overstates.
- **P10-11** · `PARTICIPANT_ROUTES`/`PRESENTER_ROUTES` are hand-maintained, and the AST check's
  dependency attribution rides on them. `PARTICIPANT ∪ PRESENTER ∪ OPEN == ROUTE_CLASSES` catches a
  twelfth route but not a route that *gains* `Depends(get_participant)` while staying in `OPEN`.
  Derive the sets from each route's `dependant` instead — the same walk P10-3's fix needs.
- **P10-12** · `/openapi.json`, `/docs`, `/redoc` are reachable on the storefront deployment (all
  `200`, verified). §4.9's "the route table contains **only** …" is literally false and
  `_FASTAPI_BUILTIN_PATHS` subtracts them silently. No participant data leaks, so this is a wording
  fix in §4.9, not a change — but the exemption should be stated where the claim is.
  *(Side note in the other direction: the `x-storefront-tokens` extension keys survive into
  `/openapi.json` intact, which makes §5.3's completeness table machine-readable off a running
  server. That was not asked for and is worth keeping.)*

### The six plan/implementation calls you asked me to rule on

1. **The S8/S10 boundary — correct as built; the fork was rightly not escalated.** §6.2 already
   hedges `S8/S10`; §5.1's S10 row is the outlier and its own Files column contradicts it. Plan fix:
   S10's row gains "*the three presenter routes are delivered by S8 (its gate is evaluated over all
   eleven); S10 moves them onto `Storefront` and adds the delay/counter and stop-intake*". Carries
   P10-6.
2. **`DemoNotSeededError` — implementation defect** (P10-2). The unreachability argument is sound
   for the boot-time snapshot and unsound as an invariant; reproduced as a bare `500`. The plan owes
   the row, but the code should not have shipped the `500` while waiting for it.
3. **`422` field granularity — correct as built.** FastAPI keys `responses` by status; the
   collapse is a genuine narrowing, both fields are proved by execution, and the test file states
   the departure. Plan fix: one sentence in §5.3 saying the *field* axis is proved by execution, not
   by declaration. Optional and beat-able: an `x-storefront-fields` key alongside the tokens would
   make the table's own key declarable — I would not spend it unless the client tier wants it.
4. **`5xx` on `POST /presenter/reset-all` — plan defect (wording).** The implementation is right and
   tested: the row is made producible as *any* unmapped graph error (a `redis.ResponseError` from
   `reset_all_participants`), which is §5.2's stated stance. §5.3's row should say "any unmapped
   graph error" rather than naming the `Thread` UNIQUE violation it structurally cannot raise.
5. **`GET /shop/api/messages`'s `reads-only` class — correct as built, and the pin is sufficient.**
   `services.read_messages` keys on `since is not None` (`services.py:970`), and the route's
   `since: int = Query(0, ge=0)` makes `None` unreachable from **any** caller — so the classification
   is safe against a future client, not merely against a future implementer. The remaining risk is
   the route dropping `since=` from the call, and M-E reddens two tests including the zero-cursor
   graph assertion. No change needed.
6. **`welcome` — plan defect.** §5.2 names a field it never specifies. The invention (per-locale,
   `en` fallback, server-side because the line is minted before the SPA knows the language was
   accepted) is sound and I would adopt it verbatim into §5.2. Carries P10-7's missing test.

### What's solid

The gate is the real thing, and I attacked it rather than read it. All eight failure demonstrations
run and each matches on its own message; nine further mutations of my own — a seam reorder, a
mis-dispatching handler, a re-classified route, an op-token swap, a mount-ordering inversion, the
config-vs-parameter image wiring, the guard removal — were caught, several by three or four tests at
once. **Keying `ROUTE_CLASSES` on `(METHOD, path)` is the single best decision in the commit**:
`/shop/api/messages` genuinely is two classes, and a path-keyed table would have handed one of them
the other's row silently.

**The single-seam decision is safe, and I verified the property it rests on independently.**
`_UNAVAILABLE`/`_TIMEOUT` (`tests/test_storefront_api.py:1487-1506`) are literal dicts with a comment
saying exactly why. Mutation M-A reorders `cross_cutting_response` so a reads-only route swallows a
`ConnectionError` as `graph_read_timeout` — the gate goes red *and* two of the three execution
parametrisations go red, from expectations that never read the seam. The seam is shared; the
expectations are not.

**Both first-pass mutants are genuinely fixed, not narrated as fixed.** M-J (`errors()[0]` →
`[-1]`) reddens `test_join_reports_the_first_violation_by_declaration_order`; M-K (drop the
`presenter_configured` guard) reddens `test_an_unset_presenter_key_never_reaches_the_comparison`. The
two-violation fixture and the `compare_digest` spy both do the work claimed for them.

**The auth matrix is not a paper exercise.** 17 of its 33 cells assert the weak negative
`status not in (401, 403)`, which I distrusted — so I printed all 33 actual responses (Appendix
P10-A, probe 1). Every one is substantively right; none passes on a `500` or a spurious code, and
`POST /order/advance` → `404` is the only surprise and is correct (no order exists). M-G (403→401 on
the wrong credential type) reddens two cells.

The image-wiring test is **stronger than the plan asked for** — two populated trees with *different
extensions*, so all three ways of getting the forwarding wrong fail with wrong URLs rather than
null ones, and `app.state.storefront_preflight["images"] == 3` pins §4.7's "built at startup only"
in the same test. The three "beat the brief" items are real: the AST `.lookup(` check with its own
non-vacuity control is a strict improvement on the grep it replaces, `storefrontEnabled` reporting
the app's wiring closes the import-time-flag trap for a second surface, and
`test_no_route_can_raise_a_refusal_it_does_not_declare` closes a residue neither half of the gate
covers — it is what catches M-D2, and P10-5 is a request to finish it, not a criticism of it.

### Open questions

1. **Does the `ServiceError` family get storefront rows, or a storefront-scoped re-shaping handler?**
   (P10-1) Rows keep one envelope per response and cost §5.3 several lines; a re-shaping handler
   keeps the client on plan tokens but changes a response the legacy surface also serves. Architect's
   call — it is the only P10 finding with a design fork.
2. **Does `POST /shop/api/messages` need a *stated* response for the reset-all window at all**, or
   does S10's stop-intake flag change what that window produces? P10-1's row set depends on the
   answer, so the two may be worth deciding together rather than S8 guessing now.

### Environment note (the brief's DB claim was already stale)

The brief states `reference` held the 15-product catalog and that all three verify scripts exit `0`.
**Before I ran anything**, `reference` held only a stray `timers-stale-key@v1` `WorkflowDef` + 4
`Step`s, and `verify_catalog.sh` exited **1**. I re-seeded the half I disturbed —
`./scripts/seed_catalog.sh` → `./scripts/verify_catalog.sh` exit **0**. The `WorkflowDef` registry
(`triage@v1`, `access-request@v1`, `salesperson@v7`) is **still absent**, so
`verify_workflows.sh acme` and `verify_salesperson.sh acme` exit `1` — as they did on arrival.
I did **not** restore them: `seed_workflows.sh acme` / `seed_salesperson.sh acme` write into
`ws:acme`, which the brief told me to keep untouched. `ws:acme`'s inventory is identical to my
arrival snapshot. **`teco`'s call whether to re-seed.**

### Appendix P10-A — mutation ledger and probe transcripts

**Method.** `falkorchat/storefront_api.py` and `falkorchat/app.py` were mutated from a byte-copy held
outside the repo and restored after every run (`md5sum` re-verified; `git status` on `falkor-chat/`
clean at the end). Command: `.venv/bin/python -m pytest tests/test_storefront_api.py -q -rf`
(150-test runs for the `app.py` mutations, adding `tests/test_app.py`). Baseline: 99 passed.

| # | Mutation | Result |
|---|---|---|
| M-A | `cross_cutting_response`: reads-only branch before the `graph_unavailable` branch | **9 failed** — the gate **and** 2 of 3 execution parametrisations |
| M-B | `_handle_graph_timeout` dispatches `_GRAPH_UNAVAILABLE` (gate seam untouched) | **1 failed** — execution only, which is the point |
| M-C | `/catalog` gains `page: int = Query(1, ge=1)` | **survived** → P10-3 |
| M-D | `reset` loses its `except UnknownParticipantError` | **survived** → P10-5 |
| M-D2 | `reset` raises `404 "no_such_participant"` (undeclared token) | **1 failed** — `test_no_route_can_raise_a_refusal…` |
| M-E | `GET /messages` drops `since=since` | **2 failed** incl. the zero-`ReadCursor` assertion |
| M-E2 | `health` returns an undeclared `503` `JSONResponse` | **survived** — the plan's *admitted* residue, confirmed live |
| M-G | `get_presenter` answers `401` for a wrong credential type | **2 failed** — auth matrix |
| M-H | roster returns `list_participants` unprojected | **1 failed** |
| M-J | `422` selection rule takes `errors()[-1]` | **1 failed** |
| M-K | `presenter_configured` guard removed | **1 failed** |
| M-N | reset-all's second-timeout `unresolved = None` → `raise` | **survived** → P10-4 |
| M-O | `_welcome` loses its locale fallback | **survived** → P10-7 |
| M-P | presenter-token verify → `token in candidates` | **survived** → P10-10 |
| M-S | reset-all always reports `incomplete` | **1 failed** |
| M-U | `JoinIn` loses `_nonblank` | **1 failed** |
| M-V | `advance_order` loses its `except UnknownOrderError` | **survived** → P10-5 |
| M-X2 | `/shop` static mount registered **before** the `/shop/api` router (`app.py`) | **76 failed** |
| M-Y | `Storefront` built from `config.STOREFRONT_DIR`, mount from the parameter (`app.py`) | **2 failed** |
| M-Z | the `storefront and dev_surface` guard removed (`app.py`) | **1 failed** |

**Probe 1 — the 33 auth-matrix cells, actual responses.** Every cell substantively correct; the
weak-negative `not in (401, 403)` cells resolve to `200` except `POST /order/advance` → `404`
(no order exists) and the two `POST /…/session` cells → `200`. No `500`, no `422`.

**Probe 2 — `DemoNotSeededError` (P10-2).** Seeded `ws:test`, lifespan entered (preflight passed),
then `MATCH (a:Agent) DETACH DELETE a` out of band:

```
before delete -> 200
after  delete -> 500 Internal Server Error
post message  -> 400 {"error":"UnknownMemberError","detail":"['assistant']"}
```

The third line is P10-1's family from the same probe.

**Probe 3 — `ServiceError` on a live route (P10-1).** `Storefront.resolve_token` patched to hand back
a record whose `thread_id` no longer exists — the state `_await_quiesce`'s docstring describes for
the reset window:

```
POST  /shop/api/messages       -> 404 {"error":"ThreadNotFoundError","detail":"th-swept-away"}
GET   /shop/api/messages       -> 200 []
GET   /shop/api/state          -> 200 {...}
POST  /shop/api/reset          -> 200 {...}
POST  /shop/api/order/advance  -> 404 {"error":"no_current_order",...}
```

Handlers on the storefront app: `HTTPException`, `RequestValidationError`,
`WebSocketRequestValidationError`, `ServiceError`, `WorkflowDefSpecError`,
`WorkflowDefNotFoundError`, `WorkflowDefConflictError`, `WorkflowRunNotFoundError`,
`WorkflowRunNotWaitingError`, `WorkflowInputRejectedError`, `WorkflowConfigError`,
`WorkflowEngineDisabledError`, `SearchNotAvailableError`, `StorefrontHTTPError`,
`FalkorDBUnreachableError`, `ConnectionError`, `TimeoutError` — **17**.
What `registered_storefront_handlers` returns: `ConnectionError`, `FalkorDBUnreachableError`,
`RequestValidationError`, `StorefrontHTTPError`, `TimeoutError` — **5**.

**Probe 4 — the two candidate fixes, run before being written into this review.**

*P10-2's family guard* — subclasses of `StorefrontError` in `storefront.py` minus every name caught
in an `except` in `storefront_api.py` and every key of `CROSS_CUTTING_HANDLERS`/`ENVELOPE_HANDLERS`:

```
subclasses: ['DemoNotSeededError', 'OrderTransitionRefusedError', 'QuiesceTimeoutError',
             'ResetStateUnknownError', 'UnknownOrderError', 'UnknownParticipantError',
             'UnscopedParticipantError']
unmapped  : ['DemoNotSeededError']
```

*P10-3's `422` derivation* — `route.body_field is not None or dependant.query_params or
dependant.path_params` (recursing into sub-dependencies), against the delivered app:

| route | body | query | declares `422` |
|---|---|---|---|
| `GET /shop/api/health` | – | – | no |
| `POST /shop/api/session` | yes | – | **yes** |
| `GET /shop/api/state` | – | – | no |
| `GET /shop/api/messages` | – | `since`, `limit` | **yes** |
| `POST /shop/api/messages` | yes | – | **yes** |
| `GET /shop/api/catalog` | – | – | no |
| `POST /shop/api/order/advance` | yes | – | **yes** |
| `POST /shop/api/reset` | – | – | no |
| `POST /shop/api/presenter/session` | yes | – | **yes** |
| `GET /shop/api/presenter/participants` | – | – | no |
| `POST /shop/api/presenter/reset-all` | – | – | no |

Exact agreement, both directions. Asserted as
`{(m, p, 422, "validation_failed") for validating routes} == {422 rows of FLAT_TABLE}`: **1 passed**
on the delivered app, **1 failed** under M-C.

**Hypotheses ruled out** (recorded so the next pass does not re-walk them):

- *`HEAD` on a `GET` route reaches the endpoint and `KeyError`s in `cross_cutting_response`.* **No.**
  FastAPI's `APIRoute` — unlike Starlette's `Route` — does **not** add `HEAD` to a `GET` route, so
  `HEAD /shop/api/state` falls through to the `/shop` `StaticFiles` mount and answers `404`;
  `OPTIONS`/`DELETE` answer `405`. Verified against the delivered app.
- *The bare `GET /health` can reach `_cross_cutting_json` with an unclassified path.* **No.**
  `app.py:417-420` catches `Exception` around `services.ping`.
- *`since=0` is falsy and takes `read_messages`' cursor path.* **No.** `services.py:970` keys on
  `since is not None`.
- *`x-storefront-tokens` breaks OpenAPI generation.* **No.** `/openapi.json` → `200`, extensions
  emitted verbatim; `/docs` → `200`.

---

## Pass 11 — 2026-09-03 (S8b: the whole handler set, the three escapes, and the owed row)

**Reviewed:** commit **`18b675a`** — `falkorchat/storefront_api.py` (+310/−65),
`tests/test_storefront_api.py` and `tests/test_app.py` (+~1030) — against
`docs/plans/salesperson-ui.md` **v1.21** (`ac6741c`) §4.9, §5.1 S8/S10, §5.2, §5.3, and against
`## Pass 10`'s twelve findings. Not reviewed: S9/S10/S12 content, the SPA. `falkorchat/storefront.py`
and `falkorchat/app.py` are byte-unchanged — re-confirmed (`git status` on `falkor-chat/` clean at
start and end; `md5sum` on both files matches `HEAD`).

**I reviewed Pass 10 as a document, not as prior reasoning of my own** — three of its calls are
adjudicated below, and two of them do not survive intact.

**CPG: considered, not relevant — `cpg_falkorchat` returns `0` for
`MATCH (f:File) WHERE f.name CONTAINS 'storefront'`, so it models neither `storefront.py` nor
`storefront_api.py`; every claim below is from direct read, execution against the live FalkorDB, or
source mutation.**

**Verdict: approve with suggestions** — **0 blockers, 2 majors, 4 minors, 4 nits**. S8b's substance
is right and I could not weaken it where it was aimed: all **six** Pass 10 mutation survivors die,
the `ServiceError` route set really is a measurement (a second route reaching the family reddens it),
and `DemoNotSeededError` is closed three ways. The two majors are not wrong responses — they are two
places where the *new* guard excuses a handler by prose and one of them names a mechanism that cannot
check it. **P10-1's reported defect is closed; P10-1's defect class re-opens one bucket over**, and I
reproduced an escape through it. Both majors are cheap and both belong to S9's dispatch rather than
to a backlog: S9's own row is what makes one of them live.

### What I ran (all of it, solo, serially)

Full suite **2608 passed / 14 deselected** twice (48.3 s, 43.4 s) — re-derived, not inherited;
`tests/test_storefront_api.py` + `tests/test_app.py` alone: **176 passed** (baseline for every
mutation). **Fourteen source mutations**, each applied to a byte-copy held outside the repo and
restored from it (`md5sum` re-verified after every run; `storefront_api.py` `4e92d03a…`, `app.py`
`e6bf735a…`, `storefront.py` `a713e2c5…`, `test_storefront_api.py` `16e68f96…`, `test_app.py`
`fcf877ac…` — all matching `HEAD` at the end). Two throw-away probe tests, appended and removed.
`ruff check` clean on all three changed files. Ledger and transcripts: **Appendix P11-A**.

**Database.** `ws:test` and `reference` only. **`ws:acme` untouched** — 14 labels, `Message` 52,
`Entity` 544, `WorkflowRun` 21, identical before and after. The suite wiped `reference` as expected;
re-seeded (`seed_catalog.sh` → `verify_catalog.sh` exit **0**). `verify_workflows.sh` /
`verify_salesperson.sh` remain non-zero, unchanged from Pass 10's arrival state and deliberately not
repaired (they write into `ws:acme`).

### Majors

#### P11-1 · **Major** · `INHERITED_HANDLERS` is a prose exemption with no mechanism, and the file attributes its verification to a sweep that cannot perform it

Eleven of the seventeen handlers are excused from the cross product by a reason string.
`test_every_inherited_handler_states_why_it_produces_no_row` asserts only that the strings are
non-empty, and its docstring says "*the sweep above is what checks the reasons are true*" — the
sweep arms three `ServiceError` faults and no workflow fault, so it checks these eleven reasons in
an empty intersection. **Mutation N-F, survived (176 passed):** a `/shop/api` route raising
`WorkflowEngineDisabledError` answers `503 {"error":"WorkflowEngineDisabledError","detail":…}` — an
unruled `(route, response)` colliding in status with `graph_unavailable`/`demo_not_seeded`, the two
C9 dispatches on. This is P10-1's exact shape, one bucket over.

**It is not hypothetical past S8.** §5.1's S9 row adds the trigger enqueue to `POST /shop/api/messages`;
`services.start_workflow_run` → `_require_executor` raises `WorkflowEngineDisabledError`
(`services.py:1996`), plus `WorkflowInputRejectedError` on the `run_ctx` bound (`services.py:2056` —
F-2's unbounded write) and `WorkflowDefNotFoundError` on a missing snapshot. Three of the eleven
excuses become falsifiable in the next step.

**Suggested fix (cheap, and it reddens at exactly the right moment):** every excuse has the form
*"no storefront route calls layer X"*, which is AST-readable. Assert that the `services.<name>`
attribute accesses inside `build_storefront_router` are exactly a declared set — today
`{read_messages, post_message, get_current_order}` — so S9 adding a fourth reddens and has to
re-derive the exemption. Full generality (arming an inherited fault per route, the `_ArmedRepo`
pattern) is available and I would not spend it yet.

#### P11-2 · **Major** · a bare `HTTPException` is invisible to the check whose docstring names exactly that case, and `StarletteHTTPException`'s exemption cites that check as its proof

`_raised_refusals` (`tests/test_storefront_api.py:2739`) collects only `ast.Call` nodes whose func is
`StorefrontHTTPError`. `test_no_route_can_raise_a_refusal_it_does_not_declare`'s docstring says
"*A route that raised an undeclared `410` would sail through both*" and claims to close it.
**Mutation N-E, survived (176 passed):** `raise HTTPException(status_code=410, detail="gone")` in the
`join` body → the probe answers `410 {"detail":"gone"}` (transcript in P11-A) — no `error` token, no
row, no declaration, and green through both halves of the gate, the four-bucket partition and the
ownership check. `HTTPException` is already imported in `storefront_api.py:74`.

`INHERITED_HANDLERS[StarletteHTTPException]`'s reason asserts "*No `/shop/api` route raises a bare
`HTTPException` … (asserted by `_raised_refusals`)*". `_raised_refusals` asserts nothing of the kind.
A claim naming a mechanism that cannot check it is what P10-5 was, and it is the thing that stops the
next reviewer looking.

**Suggested fix:** in the same walk, collect `HTTPException(` calls and `raise` targets named
`HTTPException`, and assert the set is empty — every refusal must be a `StorefrontHTTPError`. Then
the reason string becomes true. ~8 lines, one control assertion.

### Minors

- **P11-3 · the `ServiceError` wrapper's `/shop/api` path scope is dead and untested.** **N-A**
  (`if request.url.path.startswith(API_PREFIX)` → `if True`) survives, 176 passed. The legacy
  surface is protected by `app.py:427`'s `if shop is not None` and `create_app`'s
  `storefront and dev_surface` refusal — the wrapper is never on an app that mounts the legacy
  router. The commit message ("*delegates elsewhere, so the legacy surface is byte-identical,
  asserted in `test_app.py`*") and §5.3's third narrowing ("*a `/shop/api`-scoped handler*") both
  credit the path check; `test_the_default_deployment_is_untouched_by_the_storefront_parameters`
  passes because the wrapper is absent, not because of the scope. Fix: say which mechanism is
  load-bearing, and pin the scope with an app carrying one non-`/shop/api` route that raises.
- **P11-4 · the "compares every candidate" test asserts "every" over a set of size one** — the sixth
  instance of this build's signature defect. `test_presenter_token_verification_compares_every_candidate_in_constant_time`
  mints **one** token, so **N-K** (`any(compare_digest(known, token) for known in candidates)` →
  `compare_digest(candidates[0], token)`) survives it; the mutant dies only *incidentally*, in
  `test_only_post_messages_can_raise_a_service_error[thread|actor]`, whose helper happens to mint a
  second presenter token. The property is real (a presenter who logs in twice must not be locked
  out, and S10 moves this code). Fix: mint two tokens, assert both verify and both appear in `seen`.
- **P11-5 · §5.3's new row did not reach §5.2's own row for that route.** §5.2's preamble names
  **exactly three** responses omitted on purpose (the cross-cutting ones) and §5.3 rules that "*a new
  `(route, response)` pair is not shipped until it has a row here, **and §5.2 is updated to match***".
  v1.21 added `503 demo_not_seeded` to §5.3's `POST /shop/api/messages` row and to §5.2's ***join***
  row, but §5.2's `POST /shop/api/messages` row still reads `the posted row · 409 TurnInProgress · 422`.
  A reader of that row — S12a's audience — does not see it. *(Same paragraph, lesser: `401` is absent
  from every §5.2 row and is not one of the three licensed omissions. Either license it or add it.)*
- **P11-6 · the one *behavioural* unreachability claim of the seven is pinned by nothing.**
  `SERVICE_ERRORS_UNREACHABLE[UnknownOrderTransitionError]` argues `422` answers first because
  `AdvanceOrderIn.transition` is a `Literal` of exactly the three `Services._ORDER_TRANSITIONS`
  accepts. **N-H** (widen the `Literal` by `"refund"`) survives, 176 passed, and would put an
  unmapped `ServiceError` — `400 {"error":"UnknownOrderTransitionError"}` — on
  `POST /shop/api/order/advance`. The other six reasons are "no storefront route calls that layer",
  which P11-1's fix covers. Fix, one line:
  `assert set(get_args(AdvanceOrderIn.model_fields["transition"].annotation)) == set(Services._ORDER_TRANSITIONS)`.

### Nits

- **P11-7** · `assert len(registered) == 17` duplicates what the partition assertion two lines below
  already catches, and it is the enumerated-vs-derived trade §4.9 explicitly decided the *other* way
  for `_FASTAPI_BUILTIN_PATHS` ("a framework upgrade must not red-fail an assertion about this app").
  Drop the count, keep the partition; the failure message is no worse.
- **P11-8** · `_takes_params`' sub-dependency recursion is dead today — **N-J** (`return False` for
  the recursive arm) survives. Correct forward-looking code; just not evidence of anything, and the
  `test_the_derived_422_routes_are_the_five_that_declare_one` control does not reach it.
- **P11-9** · `test_every_row_of_the_table_was_produced_by_execution` skips on `-k` only. Selecting a
  single node id (`pytest tests/test_storefront_api.py::test_health_reports_status_enabled_and_the_locale_list`)
  leaves `config.option.keyword` empty, so it runs and fails on a test the caller did not select.
  Key the skip on the subset actually observed (e.g. `if not _OBSERVED >= {some floor}`), or on
  `config.option.file_or_dir` carrying `::`.
- **P11-10** · `service_error_response`'s MRO walk is unexercised — **N-B** (walk → exact-type
  lookup) survives — and mildly contradicts its neighbour: a future subclass of `ThreadNotFoundError`
  placed in `SERVICE_ERRORS_UNREACHABLE` would still be *mapped* by the walk, so the partition guard
  would not mean what it says. Either drop the walk (the guard forces explicit classification anyway)
  or assert the two agree.

### Disposition of Pass 10's findings (rechecked, not inherited)

| # | Disposition | What I rechecked |
|---|---|---|
| P10-1 | **Fixed as reported; class re-opens** → P11-1 | `registered_handlers` returns all **17** (printed off the live app); N-D neuters `_assert_handler_ownership` → both ownership tests red, so they are non-vacuous |
| P10-2 | **Fixed** | M-Q (drop `except DemoNotSeededError`) → **3 red**, incl. the AST family guard and the produced-by-execution check |
| P10-3 | **Fixed** | M-C (`/catalog` gains `page: int = Query(1, ge=1)`) → **13 red**; the derivation control reproduces the five declaring routes exactly |
| P10-4 | **Fixed** | M-N (`unresolved = None` → `raise`) → 1 red; `_FailingAfterNCalls` asserts `roster.calls == 2`, so the re-read genuinely ran |
| P10-5 | **Fixed** | M-D → 3 red, M-V → 2 red. The `_OBSERVED` recording client is a better mechanism than the tagging I suggested |
| P10-6 | **Fixed** | `_STEP_10_INTERIM` now says "two of the four", names `repo`/`presenter_key`, and states `services`/`agent_id` stay; §5.1's S10 row carries it |
| P10-7 | **Fixed** | M-O → 1 red; §5.2 carries the fallback spec and the reachability argument |
| P10-8 | **Fixed** | N-C (remove the `except KeyError`) → 1 red |
| P10-9 | **Not fixed** | `repo.list_participants` still sits outside the `try` (`storefront_api.py:1230`); neither the code comment nor `test_a_reset_all_that_times_out_…` says why a pre-write read failure may honestly report `504`. Still a minor, still correct behaviour |
| P10-10 | **Fixed** | M-P → 1 red — but see P11-4: the new test's *name* outruns its fixture |
| P10-11 | **Fixed** | `test_the_credential_route_sets_are_the_ones_the_dependencies_declare` derives both sets from each route's `dependant` and asserts disjointness |
| P10-12 | **Fixed in v1.21** | `create_app(storefront=True, dev_surface=False).routes` carries `/docs`, `/docs/oauth2-redirect`, `/openapi.json`, `/redoc` — **four**, as §4.9 now says; `_FASTAPI_BUILTIN_PATHS` is derived from a bare `FastAPI()`, which is the same four |

Pass 10's two open questions are both answered by the delivery: the family got a re-shaping handler
(question 1), and the reset-window response is `401 invalid_token` — a row `POST /shop/api/messages`
already carried, so S10's stop-intake flag does not owe it anything (question 2).

### The seven questions in the brief

1. **Is P10-1 closed?** Yes as reported, and the four-bucket classification **does** have a hole the
   partition check cannot see — two, P11-1 and P11-2. The coder's argument that `__module__` is the
   axis that moves is **correct and I tested it rather than restating it**: N-D neuters
   `_assert_handler_ownership` and both directions go red, and the two negative tests set
   `__module__` explicitly and `match=` on the specific message. What the argument does not cover is
   a handler whose *key* was always there and whose excuse is prose.
2. **Is `SERVICE_ERROR_ROUTES` a measurement?** **Yes, and I proved it detects rather than merely
   reports.** **N-G** — `GET /shop/api/state` gains a `services.list_thread_participants` call, the
   realistic way a second route reaches the family — reddens
   `test_only_post_messages_can_raise_a_service_error[thread]`. On fault-set completeness: the
   storefront's whole service-call surface is `read_messages`, `post_message`, `get_current_order`
   plus `Storefront`'s own `save_profile`/`get_profile`/`get_cart`/`filter_products`/
   `lookup_product`/`order_belongs_to_customer`/`advance_order`/`get_snapshot`; of the ten
   `ServiceError` subclasses only `_validate_and_derive_role`'s three and `_dispatch_write`'s two
   (the same two types) are reachable from it, and `advance_order`'s `UnknownOrderTransitionError` is
   `Literal`-blocked. So a fourth fault would **not** move a second route — **subject to P11-6**,
   which is the one link in that chain nothing pins. The plan owing one row rather than a section is
   correct.
3. **Is P10-2 closed, and is the family guard sound?** Closed (M-Q → 3 red). The AST guard is sound
   for what it claims: it resolves `ast.Tuple` clauses and both `ast.Name`/`ast.Attribute` shapes,
   carries a positive control (`QuiesceTimeoutError` found) and a negative one (`NeverCaughtError`
   reported), and reads the live class tree rather than a list. Its limit is that "caught somewhere
   in the file" is not "caught on the route that can raise it" — acceptable here, because
   `_raised_refusals` + the declaration half + `_OBSERVED` cover the second question, and I would not
   spend more on it.
4. **Do v1.21's three §5.3 changes match the code?** Two yes, one incomplete. The new row
   (`POST /shop/api/messages` · `503 demo_not_seeded` · C9) matches: `mentions=[agent_id]` is
   server-side (`storefront_api.py:994`), and `_validate_and_derive_role` raises before any write
   (`services.py:846`, ahead of `_dispatch_write`), so C9's *nothing changed* is exact. The third
   narrowing's "whole contribution is two pairs on one route" matches `service_error_pairs` and
   both pairs are in `TABLE`. §5.2's *join* row's narrowed claim is true —
   `DemoNotSeededError` is raised at `storefront.py:461`, inside `join`, and nowhere else in the
   package. The incompleteness is **P11-5**, in §5.2's messages row.
5. **A sixth "test that cannot exercise the rule it names"?** **Found: P11-4.** Also P11-8 (a
   recursion no control reaches) and the near-miss in P11-2, where the check exists but reads the
   wrong node type.
6. **A third "stated in two places, updated in one"?** The test file's `TABLE` is **not** it — I
   compared all 62 `(route, status, token)` cells against §5.3's per-route and cross-cutting tables
   by hand and they agree in both directions, including the nine `graph_unavailable` rows, the four
   `graph_read_timeout` rows and the five `504`s. Nothing checks that agreement, which is a standing
   risk rather than a present defect. The live instance is **P11-5**, between §5.2 and §5.3.
7. **Was Pass 10 right where S8b says it was wrong?** Adjudicated below.

### Adjudicating S8b's three counter-claims against Pass 10

- **"P10-1 undercounts — three escapes, not one, and probe 2 printed the third."**
  **Upheld against Pass 10, and S8b overshoots in the other direction.** P10-1's body names one
  escape ("*the one that provably fires*") while its own Appendix P10-A probe 2 prints a second
  (`400 UnknownMemberError`) and labels it "*P10-1's family from the same probe*" without folding it
  into the count. A fix keyed on the reported `404` would indeed have left one behind — the criticism
  lands. But **three** is not the demonstrated number either: S8b's own
  `test_the_service_error_map_is_read_off_the_one_seam` docstring concedes `UnknownActorError` "*has
  no graph state that produces it without also failing `resolve_token` first, so it is unreachable
  through the wire*", and it is asserted at the seam, not driven. The honest count of escapes
  reachable on the delivered app is **two**; the third is type-reachable under fault injection only.
- **"P10-1's implied blast radius is larger than the truth — one route of eleven, measured."**
  **Not sustained as a correction.** P10-1 did not assert a radius; its suggested fix said "*at
  minimum `POST /shop/api/messages` gains rows*", which is where the measurement landed. What S8b
  built is a genuine strengthening — an unmeasured radius replaced by a measured one, and N-G shows
  the measurement bites — but it corrects an *absence* in Pass 10, not an error.
- **"P10-10's 'harmless' framing went untested; M-P is killable in six lines."**
  **Upheld, on the "untested" half only.** M-P dies (verified). The harmlessness itself is not
  disputed — S8b's own new comment repeats it verbatim ("*Harmless — the response already reveals
  validity*"). The real lesson is narrower and worth keeping: *"harmless"* is a statement about
  consequence and Pass 10 let it stand in for *"not worth pinning"*, which is a different judgment.
  And the replacement test inherits a smaller version of the same problem (P11-4).

### What's solid

**The `_OBSERVED` recording client is the best thing in this commit and better than what P10-5
asked for.** I suggested tagging each contract test with the `(method, path, status, token)` it
proves; S8b rebound `TestClient` module-wide and records what the server actually said. A tag is a
claim that can be wrong; a recorded response cannot be. It is what makes M-D, M-V and M-Q die on the
`⊇` direction, and it catches an unruled response on a route nobody wrote a contract test for. The
`⊆` direction is correctly stated as safe under any subset while the `⊇` direction skips — the
reasoning is right even where the skip predicate is not (P11-9).

**The measurement pattern is the right answer to P10-11's objection and generalises.** `_ArmedRepo`
arms at the *repository* seam — `thread_exists`, `resolve_member_kinds` — not by patching the
service function under test, so the sweep is not circular; and every response it sees is checked
against `TABLE`, which makes it a standing guard on any route added later, not a one-shot
derivation. P11-1's fix is available precisely because this machinery exists.

**`_FailingAfterNCalls` does exercise the rule it names**, which is the thing P10-4 warned it might
not: `roster.calls == 2` proves the re-read ran and `repo.calls == 1` proves nothing retried. The
choice of `participants: null` over an absent key is deliberate and documented, and it matches §5.1's
S10 done-condition ("*no state body, never a `500`*").

**The wiring-order refusal is the right shape.** `register_storefront_error_handlers` raising a named
`RuntimeError` at wiring time, rather than letting an absent incumbent surface as a `TypeError`
*inside* an exception handler on a participant's first `ServiceError`, is exactly the "loud at boot"
posture §4.9 argues for everywhere else.

**Ownership on `__module__` is the correct axis and it is tested in both directions** — a key-set
comparison cannot see an override, and N-D confirms the two negative tests are live rather than
decorative.

### Open questions (need `teco`'s or the architect's call)

1. **Do P11-1 and P11-2 go to S8c, or into S9's row as preconditions?** My recommendation: **S9's
   row**, as two named done-conditions. P11-1's fix reddens exactly when S9 adds its fourth
   `services.` call, which is the moment it is worth something; P11-2 is ~8 lines and can ride
   along. A third S8 round buys nothing that S9 does not already have to touch.
2. **Does §5.2's `POST /shop/api/messages` row get the `503 demo_not_seeded` line, and does the
   preamble license the `401` omission?** (P11-5) One is a plan edit the architect owns; the second
   is a choice — either list `401` per route or say once that credential responses live in §5.3.

### Appendix P11-A — mutation ledger and probe transcripts (Pass 11)

**Method.** Every mutation applied to `falkorchat/storefront_api.py` (or `tests/test_storefront_api.py`
where noted) from a byte-copy held outside the repo, restored from that copy after each run, `md5sum`
re-verified each time. Command: `.venv/bin/python -m pytest tests/test_storefront_api.py tests/test_app.py -q`.
**Baseline: 176 passed.** No `git` tree-mutating command was used at any point.

| # | Mutation | Result |
|---|---|---|
| N-A | `_handle_service_error`: `startswith(API_PREFIX)` → `True` (path scope removed) | **survived** → P11-3 |
| N-B | `service_error_response`: MRO walk → exact-type lookup | **survived** → P11-10 |
| N-C | `_cross_cutting_json`: the `except KeyError` guard removed (P10-8's fix) | **1 failed** |
| N-D | `_assert_handler_ownership` → no-op (test-file mutation) | **2 failed** — both ownership tests |
| N-E | `join` raises a bare `HTTPException(410)` on an unexercised branch | **survived** → P11-2 |
| N-F | `join` raises `WorkflowEngineDisabledError` on an unexercised branch | **survived** → P11-1 |
| N-F2 | same, `SearchNotAvailableError` (a `ServiceError` with its own inherited handler) | **survived** → P11-1 |
| N-G | `GET /state` gains a `services.list_thread_participants` call | **1 failed** — the sweep, `[thread]` |
| N-H | `AdvanceOrderIn.transition` `Literal` widened by `"refund"` | **survived** → P11-6 |
| N-I | reset-all's second-timeout `unresolved = None` → `raise` (P10 M-N) | **1 failed** |
| N-J | `_takes_params`: sub-dependency recursion → `False` (test-file mutation) | **survived** → P11-8 |
| N-K | `_PresenterSessions.verify` → `compare_digest(candidates[0], token)` | **2 failed** — but *not* the test that names the rule → P11-4 |
| M-C | `/catalog` gains `page: int = Query(1, ge=1)` (P10 survivor) | **13 failed** |
| M-D | `reset` loses its `except UnknownParticipantError` (P10 survivor) | **3 failed** |
| M-O | `_welcome` loses its locale fallback (P10 survivor) | **1 failed** |
| M-P | presenter-token verify → `token in candidates` (P10 survivor) | **1 failed** |
| M-Q | `join` loses its `except DemoNotSeededError` (P10-2's fix) | **3 failed** |
| M-V | `advance_order` loses its `except UnknownOrderError` (P10 survivor) | **2 failed** |

**Probe 1 — the escapes, driven through the wire.** Mutation in place, one throw-away test appended
to `tests/test_storefront_api.py`, run under `-k`, then both files restored from the byte-copies:

```
N-E  POST /shop/api/session -> 410 '{"detail":"gone"}'
N-F  POST /shop/api/session -> 503 '{"error":"WorkflowEngineDisabledError","detail":"engine off"}'
N-F2 POST /shop/api/session -> 503 '{"error":"SearchNotAvailableError","detail":"no index"}'
```

Note on N-F2: `SearchNotAvailableError` is a `ServiceError` subclass, but `app.py` registers a
handler for it *directly*, so Starlette's MRO walk picks that one and the storefront's re-shaper —
including its "no mapping, say so in the log" branch — is never entered. The re-shaper's safety net
does not cover family members that carry their own inherited handler.

**Probe 2 — the live app, enumerated.** `create_app(storefront=True, dev_surface=False)`:
17 exception handlers; top-level paths `''` (the `_IncludedRouter` carrying the eleven `/shop/api`
routes), `/health`, `/docs`, `/docs/oauth2-redirect`, `/openapi.json`, `/redoc`. A bare `FastAPI()`
registers the same four documentation paths — so §4.9's derived exemption and its stated value agree
today.

**Probe 3 — `TABLE` against §5.3, by hand, both directions.** All 62 cells agree: eleven route keys,
§5.3's per-route rows, the nine `503 graph_unavailable` rows (classes `writes` + `reads-only`), the
four `503 graph_read_timeout` rows (`reads-only`), the five `504 <op>_state_unknown` rows, and the
three documented departures (the `422` field collapse, `5xx` written `500`/`unhandled`, and
reset-all's two `200` rows). No drift found.

**Hypotheses ruled out** (so the next pass does not re-walk them):

- *The four-bucket partition can be defeated by dumping a storefront-registered handler into
  `INHERITED_HANDLERS`.* **No.** `_assert_handler_ownership` rejects it on `__module__`, in both
  directions, and N-D proves the two tests that cover it are live.
- *`SERVICE_ERROR_ROUTES` is circular — the sweep arms what only `POST /messages` calls.* **No.**
  Both arms are repository methods (`thread_exists`, `resolve_member_kinds`) available to every
  route, and N-G shows a second route reaching the family reddens the assertion.
- *`UnknownMemberError` → `503 demo_not_seeded` could mis-report a client error.* **No.** The route
  passes `mentions=[agent_id]` server-side (`storefront_api.py:994`); `PostMessageIn` carries only
  `text`, and pydantic ignores an extra `mentions` key rather than forwarding it.
- *The `503` the re-shaper mints could be confused with a quiesce `503` by C9.* **No** — they carry
  different tokens (`demo_not_seeded` vs `quiesce_timeout`) and §5.3 rules on the token. This is
  precisely what P11-1's `WorkflowEngineDisabledError` escape would break, since that one carries a
  class name where the token belongs.
- *`storefront.py`/`app.py` drifted.* **No.** `md5sum` matches `HEAD` at start and end;
  `git status` on `falkor-chat/` is clean.
