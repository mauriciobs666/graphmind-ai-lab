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

---

## Pass 12 — 2026-09-03 (S8c: does the mechanism reach the case it was built for?)

**Reviewed:** commit **`2e27835`** — `falkorchat/storefront_api.py` (+45/−3) and
`tests/test_storefront_api.py` (+434/−31, 7 new tests) — against `docs/plans/salesperson-ui.md`
**v1.23** §4.9, §5.1 S8/S9/S10, §5.2, §5.3, and against `## Pass 11`'s ten findings plus P10-9.
Not reviewed: S9/S10/S12 content, the SPA, and the five facts the coordinator verified
independently and asked me not to re-spend on (the three byte-unchanged files, the AST equivalence,
the full-suite 2615/14, `ruff`, `ws:acme`) — I re-checked `ws:acme` anyway because I hold the
instance: 14 labels, `Message` 52, `Entity` 544, `WorkflowRun` 21, unchanged.

**I reviewed Pass 11 as a document, not as prior reasoning of my own** — I did not write it, and its
two disputed calls are adjudicated below on evidence I generated.

**CPG: considered, not relevant — `cpg_falkorchat` models neither `storefront.py` nor
`storefront_api.py` (`MATCH (f:File) WHERE f.name CONTAINS 'storefront'` → 0, per the brief), so
every claim below is from direct read, AST analysis, or source mutation against the live suite.**

**Verdict: needs changes** — **0 blockers, 1 major, 1 minor, 2 nits**. Nine of the eleven closures
are real and I could not weaken them: every mutation Pass 11 left surviving now dies, N-K dies
*deterministically* where it used to die by luck, and P10-9 is pinned from the wire after two passes
open. **The major is the one that matters: P11-1's guard — the artifact the coordinator overrode
Pass 11 to build *before* S9, precisely so it could fire — does not redden on the shape S9 has
decided to take.** It reddens only on the shape S9 has decided *against*. The fix is ~15 lines in
one test and no source change, but it must land before S9 is dispatched, because a guard widened
during S9 is the "born accommodating" artifact the whole sequencing decision existed to avoid.

### What I ran (all of it, solo, serially)

Two-file baseline **183 passed** (`tests/test_storefront_api.py` + `tests/test_app.py`), matching
S8c's claim. **Fifteen mutations**, each applied to byte-copies held outside the repo and restored
from them, `md5sum` re-verified after every run (`storefront_api.py` `7415e5df…`, `storefront.py`
`a713e2c5…`, `app.py` `e6bf735a…`, `test_storefront_api.py` `1dba4209…`, `test_app.py` `fcf877ac…`
— all matching `HEAD` at the end; `git status` on `falkor-chat/` clean). Two throw-away probes,
appended and removed. **Sixteen extra suite runs across eight `PYTHONHASHSEED` values** on both the
Pass-11 and the fixed tree, to settle disagreement 2. No `git` tree-mutating command at any point.
Ledger and transcripts: **Appendix P12-A**.

**Database.** `ws:test` and `reference` only. `ws:acme` untouched (above). `reference` re-seeded at
the end — `seed_catalog.sh` → `verify_catalog.sh` exit **0**, "OK — product catalog in sync (15
products)". `seed_workflows.sh`/`seed_salesperson.sh` never run.

### Major

#### P12-1 · **Major** · the P11-1 guard fires on the `services.` shape S9 has *ruled out*, and stays green on the two it will actually take — so S9's own done-condition is unmeetable as written

`test_the_router_reaches_exactly_the_service_calls_the_exemptions_assume` pins
`{node.attr for node ∈ build_storefront_router if node is services.<attr>}` to three names. I applied
S9's decided shape — §5.1 S9's own `Storefront.enqueue_turn(ctx, participant, posted)` on
`storefront.py`, whose worker body calls `self._services.start_workflow_run`, plus
`shop.enqueue_turn(...)` in the router behind `services.post_message` — and the file stayed
**green, 183 passed**. So did `shop._services.start_workflow_run(...)` written directly in the
router (**183 passed**), a spelling the same file already uses at `storefront_api.py:1360`. Only
`services.start_workflow_run(...)` through the alias reddens (**1 failed**) — and that is the one
placement S9 explicitly rejects, because the trigger runs on the worker, not the request thread.

The root cause is the guard's name outrunning its assertion — this build's signature defect, **the
seventh instance**. The excuses it stands in for say *"no storefront route **calls layer X**"*, but
a route reaches that layer two ways. Measured off the AST: the router's direct `services.<name>`
surface is `{read_messages, post_message, get_current_order}`, while `shop.<method>` reaches
`{advance_order, get_cart, get_current_order, get_profile, order_belongs_to_customer, save_profile}`
one hop down and `filter_products` two. **The router's true one-hop reach is eight methods; the
guard measures three** — and S9's fourth call lands in the five it cannot see.

The blast radius is `docs/plans/salesperson-ui.md` §5.1's S9 row, which now states as a
done-condition that *"S8c's `services.` access assertion goes red on this step, and re-deriving it
is a required piece of work"*. On the delivered guard that is false: the S9 implementer will see
green and the eleven `INHERITED_HANDLERS` excuses revert to prose at exactly the moment three of
them (`WorkflowEngineDisabledError`, `WorkflowInputRejectedError`, `WorkflowDefNotFoundError`)
become falsifiable.

**Suggested fix (~15 lines, one test, no source change).** Widen the reader to the union it claims:
`services.<name>` **and** `shop._services.<name>` inside `build_storefront_router`, **plus**
`self._services.<name>` inside each `Storefront` method the router calls (the `shop.<attr>` set is
readable off the same walk). Pin that union — today the eight above, plus `filter_products` — with
`start_workflow_run` still deliberately absent. The `services = shop._services` control stays and is
genuinely live (renaming the binding reddens: mutation N-CTRL, 1 failed), it simply controls a
rename rather than an added path. Then re-run S9-REAL as the acceptance check: it must go red.

### Minor

- **P12-2 · the bare-`HTTPException` check is scoped to `raise` statements lexically inside the
  router, so one indirection escapes it.** `_raised_class_names` walks `build_storefront_router`
  only. **Mutation N-M, survived (183 passed):** a module-level `_refuse_retired_name(name)` in
  `storefront_api.py` that raises `HTTPException(410)`, called from `join`. Driven through the wire
  it answers `410 '{"detail":"gone"}'` — no `error` token, no row, green through both halves of the
  gate. That is P11-2's escape reproduced exactly, one call out; and
  `INHERITED_HANDLERS[StarletteHTTPException]`'s reason still reads "*No `/shop/api` route raises a
  bare `HTTPException`*", which is broader than what is checked. Materially smaller than P11-2 (the
  reflexive shape is now caught, and the reason cites a test that does something), which is why it
  is a minor. **Fix, checked not guessed:** parse the whole module rather than the router node and
  allowlist the three non-route raises — a module-wide walk today reports exactly
  `{RuntimeError, StorefrontPreflightError, ValueError}` outside the router, all at wiring/boot
  time. `_raised_class_names` already takes source rather than reading it, so its synthetic controls
  carry over unchanged.

### Nits

- **P12-3** · P11-9's new predicate self-disables on marker deselection, and S8c's docstring calls
  that "fine". It is *visible* rather than silent — `addopts = '-ra -m "not live"'` prints the skip
  reason — but adding one `@pytest.mark.live` test to this module turns the `⊇` direction off in
  every default run: I appended one and got `1 skipped … 1 of this module's tests were not
  collected`, `182 passed, 1 deselected`. Today there are **zero** `mark.live` uses in either file
  and the 14 deselected tests are elsewhere, so this is latent. One line closes it: subtract the
  marker-deselected items (`request.session.config.hook`-free version:
  `defined - {n for n in defined if _is_live_marked(n)}`) or, simpler, allowlist names carrying a
  `live` marker.
- **P12-4** · `test_the_service_error_map_resolves_through_the_class_tree` mutates process-global
  state — it mints a real `ServiceError` subclass into the live class tree — and relies on
  `del` + `gc.collect()` to undo it, because **two other tests assert the family is exactly ten**.
  It passes on pytest 9.1.1. If the reclaim ever fails (an assertion-rewrite temporary or a
  traceback holding the instance across a *failing* run) two unrelated tests go red with a
  confusing message. Cheaper and version-proof: have `_subclasses` skip classes whose `__module__`
  is not `falkorchat.services`, and drop the GC dance.

### Disposition of Pass 11's findings (re-derived, not inherited)

| # | Disposition | What I rechecked |
|---|---|---|
| P11-1 | **Mechanism built, but it cannot fire on S9's shape** → **P12-1** | S9-REAL **survived** (183), S9-ALT **survived** (183), S9-NAIVE **1 failed**; N-CTRL **1 failed** (the control is real); router's true layer reach measured at 8 vs the 3 asserted |
| P11-2 | **Fixed for the reflexive shape; scope gap remains** → P12-2 | N-E **1 failed**, N-F **1 failed**, N-F2 **1 failed**, all on the new test; N-M **survived**, wire `410 '{"detail":"gone"}'` |
| P11-3 | **Fixed** | N-A (`startswith(API_PREFIX)` → `True`) → **1 failed**, `test_the_service_error_re_shaper_answers_only_on_shop_api`; the two-surface app really does discriminate (`401 invalid_token` inside, `404 ThreadNotFoundError` outside) |
| P11-4 | **Fixed, and the finding is *stronger* than Pass 11 stated** | N-K → **1 failed**, the test that names the rule, on **all eight** seeds tried. See ruling 2 |
| P11-5 | **Fixed in the plan (v1.22/v1.23)** | §5.2's `POST /shop/api/messages` row now carries `503 demo_not_seeded` with the `UnknownMemberError` producer; the `401` omission is licensed by a named fourth-omission paragraph citing P11-5 |
| P11-6 | **Fixed** | N-H (`Literal` widened by `"refund"`) → **1 failed**, `test_the_only_behavioural_unreachability_claim_is_pinned_to_its_producer` |
| P11-7 | **Fixed** | `len(registered) == 17` is gone; the partition equality carries the `sorted(...)` message instead |
| P11-8 | **Fixed; the keep-rather-than-delete call is right** | N-J → **1 failed**. The new control borrows `GET /messages`' dependant onto `/catalog`'s and asserts the parameter is on the sub-dependency, so it reaches the recursive arm rather than the base case |
| P11-9 | **Fixed, and Pass 11's example was wrong** → ruling 1 | Full evidence below |
| P11-10 | **Fixed; keeping the walk is defensible** | N-B (MRO walk → exact-type lookup) → **1 failed**. The second half (nothing in `SERVICE_ERRORS_UNREACHABLE` is mapped through the walk) is a static cross-check rather than a behavioural one — correct, just weaker than its neighbour, and see P12-4 |
| P10-9 | **Fixed after two passes open** | Moving the roster read + drain inside the `try` → **1 failed**, `test_a_reset_all_whose_pre_drain_roster_read_times_out_never_enters_the_sweep`. The wire discriminator is real: the `except` arm always carries a `participants` key, the typed handler carries none |

### The two disagreements, ruled

**Ruling 1 — P11-9's reproduction. S8c is right, on both halves; Pass 11's finding stands but its
example and its suggested fix do not.**

- Pass 11's literal command against the **Pass-11 tree** (`git show 2e27835^`): `1 passed`. A single
  node id collects only that node, so the check is never collected and cannot run. **Does not
  reproduce.**
- S8c's reproducer against the same tree: selecting the check itself → `1 failed`, all 57 rows
  reported unproducible. **Reproduces.**
- And it is reachable by accident, not only on purpose. With an ordinary regression on the Pass-11
  tree (`if shop.turn_in_flight(...)` → `if False:` — the `409` gate lost), the full run gives
  **2 failed**: the real one, plus the check, because the failing test never recorded its row. Both
  land in `lastfailed`, so `pytest --lf` re-runs exactly those two and the check explodes with a
  **55-row** wall. One real failure, two reported, the second spurious. On the **fixed** tree the
  same regression gives the same honest `2 failed`, and `--lf` gives **`1 failed, 1 skipped`** with
  an informative reason. The fix works and is better than what Pass 11 asked for.
- Pass 11's suggested fix would not have caught it: under `--lf` I measured
  `config.option.file_or_dir=[]` and `keyword=''` (transcript in P12-A), so a predicate keyed on
  `file_or_dir` carrying `::` never fires there. S8c's "covers only half" is exact.

**Ruling 2 — the N-K count. S8c's attribution is right, and the truth is worse than either document
says: the number is `0`–`3` depending on `PYTHONHASHSEED`, and at seed 4 the mutant survives the
entire file.** On the Pass-11 tree, N-K's kill count across seeds 0–7 was
`1, 2, 1, 2, 0, 1, 1, 3`. Pass 11's "2" and S8c's "1" are both faithful observations of an
uncontrolled variable — neither document is *wrong* about what it saw, and there is no fault to
assign beyond this: **a mutation ledger that reports a bare count is reporting one draw from a
distribution, and Pass 11 should have said "seed-dependent" rather than "2".** `_PresenterSessions._tokens`
is a `set` of `str`, so candidate order is `PYTHONHASHSEED`-dependent, and no `pytest-randomly` is
installed (pytest 9.1.1, plain) — the variance is genuine string-hash randomisation, not plugin
shuffling. **P11-4 is thereby strengthened, not softened:** the incidental kill was not merely luck,
it was luck that fails outright one seed in eight. On the fixed tree the same mutant dies on every
seed tried, and `test_presenter_token_verification_compares_every_candidate_in_constant_time` is
red in all eight — the near-miss assertion (`{known for known, _ in seen} == {first, second}`) is
order-independent by construction, which is the right way to have written it.

### What's solid

**Nine of eleven closures are genuine, and three of them are better than what Pass 11 asked for.**
P11-4's two-token rewrite is order-independent *by construction* rather than by pinning a seed.
P11-8's control borrows a real `dependant` onto another route rather than synthesising one, so it
exercises the recursion against FastAPI's own object graph. P11-3's two-surface app is the only way
to test a property that no deployment exhibits, and it correctly records that the delivered legacy
surface is byte-identical **by absence** — the mechanism/claim separation P11-3 asked for, written
at the call site where the next reader meets it.

**P10-9's resolution is the right kind of answer to a finding that was twice "not fixed".** The
decision is defended at the call site (`forget_all`/`clear_all_turns` are correct only after a sweep
that may have committed) *and* pinned from the wire on a discriminator that exists independently of
the reasoning — the `participants` key. Two passes of prose became one mutation-killing test.

**The AST-over-grep argument is load-bearing and true.** `advance_order`'s docstring names
`services.order_belongs_to_customer` in prose, so a substring search reports a fourth call; the walk
does not see it. That is a real reason to parse, not a stylistic preference.

**Restraint on P11-7.** Dropping `len(registered) == 17` and moving the `sorted(...)` message onto
the partition equality is exactly the enumerated-vs-derived trade §4.9 already decided, applied
consistently rather than argued again.

### Open questions (need `teco`'s call)

1. **Does P12-1's fix go in as an S8d, or as the first done-condition of S9?** My recommendation:
   **S8d, before S9 is dispatched.** The coordinator's own argument decides it — a guard authored
   inside the step it guards is born accommodating. Fixing it inside S9 reproduces exactly the
   failure mode the S8c/S9 split existed to prevent, and the fix is ~15 lines in one test file that
   S9 will also touch, so serialising costs one short round rather than a merge.
2. **Does §5.1's S9 row need a wording change once the guard is widened?** It currently says
   "S8c's `services.` access assertion goes red on this step". After P12-1's fix the assertion is
   about the router's *service-layer reach*, not its `services.` alias — the obligation is
   unchanged but the sentence names the wrong thing, and the S9 implementer reads that sentence.
   That is an `architect` edit, not mine.

### Appendix P12-A — mutation ledger and transcripts (Pass 12)

**Method.** Every mutation applied from a byte-copy held outside the repo and restored from that
copy after each run, `md5sum` re-verified each time. Command:
`.venv/bin/python -m pytest tests/test_storefront_api.py tests/test_app.py -q`.
**Baseline: 183 passed.** No `git` tree-mutating command was used at any point.

| # | Mutation | Result |
|---|---|---|
| **S9-REAL** | `Storefront.enqueue_turn` (worker calls `self._services.start_workflow_run`) + `shop.enqueue_turn(...)` in the router — **§5.1 S9's decided shape** | **survived, 183 passed** → P12-1 |
| **S9-ALT** | `shop._services.start_workflow_run(...)` written directly in the router body | **survived, 183 passed** → P12-1 |
| S9-NAIVE | `services.start_workflow_run(...)` through the router's alias | **1 failed** — the guard (the one shape S9 rejects) |
| N-CTRL | the `services = shop._services` binding renamed to `svc` throughout the router | **1 failed** — the guard's control is live |
| N-A | `_handle_service_error`: `startswith(API_PREFIX)` → `True` | **1 failed** (P11-3's new test) |
| N-B | `service_error_response`: MRO walk → `SERVICE_ERROR_RESPONSES.get(type(exc))` | **1 failed** (P11-10's new test) |
| N-E | `join` raises `HTTPException(410)` on an unexercised branch | **1 failed** (P11-2's new test) |
| N-F | `join` raises `WorkflowEngineDisabledError` on an unexercised branch | **1 failed** |
| N-F2 | `join` raises `SearchNotAvailableError` on an unexercised branch | **1 failed** |
| **N-M** | a **module-level** `_refuse_retired_name` raises `HTTPException(410)`; `join` calls it | **survived, 183 passed** → P12-2 |
| N-H | `AdvanceOrderIn.transition` `Literal` widened by `"refund"` | **1 failed** (P11-6's new test) |
| N-J | `_takes_params`: recursive arm → `False` (test-file mutation) | **1 failed** (P11-8's new test) |
| N-K | `_PresenterSessions.verify` → `compare_digest(candidates[0], token)` | **1 failed**, deterministic across seeds 0–7 (P11-4's rewritten test) |
| N-G | `GET /state` gains a `services.list_thread_participants` call | **2 failed** — the sweep `[thread]` **and** the new guard |
| P10-9-MOVE | the pre-drain roster read + drain loop moved inside the `try` | **1 failed** (P10-9's new test) |

**Probe 1 — N-M driven through the wire.** Mutation in place, one throw-away test appended to
`tests/test_storefront_api.py`, run under `-k`, then both files restored from the byte-copies:

```
N-M  POST /shop/api/session -> 410 '{"detail":"gone"}'
```

**Probe 2 — the router's true service-layer reach, read off the AST.**

```
router -> services.<name> : ['get_current_order', 'post_message', 'read_messages']
Storefront.<method> -> self._services.<name>:
  _catalog_rows:     ['filter_products']
  advance_own_order: ['advance_order', 'order_belongs_to_customer']
  get_state:         ['get_cart', 'get_current_order', 'get_profile']
  join:              ['save_profile']
  reset_participant: ['save_profile']
union (one hop from the router): ['advance_order', 'get_cart', 'get_current_order',
  'get_profile', 'order_belongs_to_customer', 'post_message', 'read_messages', 'save_profile']
```

Eight, against the three `SERVICE_CALLS_TODAY` asserts. Module-wide `raise` walk, for P12-2's fix:
router `{RequestValidationError: 1, StorefrontHTTPError: 17}`, outside the router
`{RuntimeError: 1, StorefrontPreflightError: 1, ValueError: 1}`.

**Probe 3 — N-K across `PYTHONHASHSEED`** (`tests/test_storefront_api.py` + `tests/test_app.py`):

| seed | Pass-11 tree | fixed tree |
|---|---|---|
| 0 | 1 failed | 3 failed |
| 1 | 2 failed | 2 failed |
| 2 | 1 failed | 1 failed |
| 3 | 2 failed | 3 failed |
| 4 | **176 passed — survived** | 2 failed |
| 5 | 1 failed | 1 failed |
| 6 | 1 failed | 2 failed |
| 7 | 3 failed | 2 failed |

On the fixed tree the named test is red at every seed (spot-checked at seeds 2 and 5, where the
count is 1 and that one failure *is* `test_presenter_token_verification_compares_every_candidate_in_constant_time`).

**Probe 4 — the P11-9 cascade, both trees.** Regression: `if shop.turn_in_flight(...)` → `if False:`.

```
Pass-11 tree, full run : FAILED …test_a_second_post_while_a_turn_is_in_flight_is_409_with_nothing_written
                         FAILED …test_every_row_of_the_table_was_produced_by_execution
                         2 failed, 174 passed
Pass-11 tree, --lf     : 2 failed, 978 deselected  (the check re-fails with 55 rows)
fixed tree,   full run : same 2 failed, 181 passed
fixed tree,   --lf     : 1 failed, 1 skipped, 978 deselected
                         SKIPPED … 94 of this module's tests were not collected
option state under --lf: file_or_dir=[]  keyword=''  lf=True
```

`file_or_dir=[]` is the measurement that settles the second half of ruling 1.

**Probe 5 — the latent marker self-disable (P12-3).** One `@pytest.mark.live` test appended to
`tests/test_storefront_api.py`, nothing else changed:

```
SKIPPED [1] …:3247: 1 of this module's tests were not collected, so the run cannot cover the whole table
182 passed, 1 skipped, 1 deselected
```

Visible under the project's own `addopts = '-ra -m "not live"'`, which is why it is a nit and not a
minor. Zero `mark.live` uses exist in either file today; the suite's 14 deselected tests are
elsewhere (`2615/2629 collected`).

**Hypotheses ruled out** (so the next pass does not re-walk them):

- *The `services = shop._services` control is decorative.* **No.** N-CTRL (rename the binding
  throughout the router) reddens the guard on the control assertion, ahead of the set comparison —
  it does not silently measure an empty set. The control is real; it simply controls a *rename*, not
  an *added* access path, which is P12-1.
- *S9 might still add `services.start_workflow_run` through the alias, so the guard fires after
  all.* **No.** §5.1 S9 pins `Storefront.enqueue_turn(ctx, participant, posted)` as the interface and
  places `trigger.maybe_trigger` → `services.start_workflow_run` **on the worker**; the worker body
  belongs to `storefront.py`, which the guard does not read. The router gains `shop.enqueue_turn`,
  not a `services.` attribute.
- *P11-9's new predicate breaks the full-suite run because the 14 deselected tests hide in this
  module.* **No.** `pytest tests/test_storefront_api.py tests/test_app.py -q` reports no deselection
  at all, and `grep -c "mark.live"` is `0` in both files.
- *The N-K disagreement is a methodological error by one of the two documents.* **No.** Both counts
  reproduce, at different seeds, on the same tree. The variable is `PYTHONHASHSEED`.
- *`test_the_service_error_map_resolves_through_the_class_tree` leaks its synthetic subclass and
  makes the two "family is exactly ten" tests order-dependent.* **Not today** — the in-test
  `len(_subclasses(ServiceError)) == 10` passes on pytest 9.1.1, so the reclaim happens. Recorded as
  P12-4 because it is a GC-timing dependency, not a guarantee.
- *S8c changed behaviour under cover of a test-only commit.* **No.** Independent of the
  coordinator's AST check: N-A, N-B, N-H, N-K and P10-9-MOVE all reproduce the *pre-S8c* behaviour
  descriptions exactly, and the two-file baseline moved 176 → 183 on exactly seven new test
  functions.

## Pass 13 — 2026-09-07 (S8d: is the widening correct, and is it complete?)

**Reviewed:** commits **`769adc3`** (S8d partial) and **`1887180`** (S8d complete) as one unit —
`falkorchat/storefront_api.py` and `tests/test_storefront_api.py` — against
`docs/plans/salesperson-ui.md` **v1.23** §4.9, §5.1 S8/S9/S10, §5.2, §5.3, against `## Pass 12`'s
four findings, and against `docs/plans/salesperson-ui-coordination.md`'s record of why the guard was
built before S9. Not reviewed: S9/S10/S12 content, the SPA, `docs/HISTORY.md`/`SERVER.md`'s held
documentation debt.

**I am a fresh reviewer.** I did not write Pass 12; I treated it as a document and re-derived every
claim of it I relied on, including the two it got wrong.

**CPG: considered, not relevant — `cpg_falkorchat` models neither `storefront.py` nor
`storefront_api.py` (Pass 12 measured 0 `File` nodes matching `storefront`, and the brief asked me
not to re-spend on it), so every claim below is from direct read, AST analysis, or source mutation
against the live suite.**

**Verdict: needs changes** — **0 blockers, 2 majors, 1 minor, 1 nit**. S8d2's central judgement is
**right**: Pass 12's recommended fix was insufficient, both of the mutations it cites really do
survive `769adc3`, and widening to both storefront modules was the correct call. But the answer to
*"is it now complete?"* is **no**, and in the same shape for the third consecutive pass: **both
guards state a rule broader than the reach they implement.** The reach guard cannot see a
service-layer call written through a local alias — which is S9's decided shape plus one line, so
S9's done-condition is unmeetable again (P13-1, the ninth instance). The raise guard's exemption
still says "no `/shop/api` route raises a bare `HTTPException`" while stopping at `storefront.py`; a
bare `HTTPException(410)` in `services.save_profile` — one of the nine methods the sibling guard
itself measures as reached — survives at **183 passed** and answers `410 '{"detail":"gone"}'` from
`POST /shop/api/session` (P13-2, the tenth). Both fixes are small, both are in one test file, and
both must land before S9 dispatches and before §5.1's S9 row lifts S8d2's reach statement.

### What I ran (all of it, solo, serially)

Two-file baseline **183 passed** and full suite **2615 passed, 14 deselected**, both my own runs.
**Nine mutations** plus two wire probes, each applied to byte-copies held outside the repo and
restored from them, `md5sum` re-verified after every run — `storefront_api.py` `7b83eb14…`,
`storefront.py` `a713e2c5…`, `services.py` `a952f4ad…`, `app.py` `e6bf735a…`,
`test_storefront_api.py` `a95da701…`, `test_app.py` `fcf877ac…`, all matching `HEAD` at the end,
`git status` on `falkor-chat/` clean. Two baselines were re-created from
`git show 769adc3:<path>` and restored the same way; **no `git` tree-mutating command at any
point**. Ledger and transcripts: **Appendix P13-A**.

**Database.** `ws:test` and `reference` only. `ws:acme` re-checked at the end and unchanged — 14
labels with nodes, `Message` 52, `Entity` 544, `WorkflowRun` 21. `reference` was wiped by my
full-suite run (the documented default-`pytest` teardown) and re-seeded: `seed_catalog.sh` →
`verify_catalog.sh` exit **0**, "OK — product catalog in sync (15 products)".
`seed_workflows.sh`/`seed_salesperson.sh` never run.

### Major

#### P13-1 · **Major** · the reach guard reads three hard-coded spellings, not "every `Services` method a route can reach, by any path" — a one-line local alias makes S9's own shape invisible again

`_service_layer_reach`'s docstring says *"Every `Services` method a `/shop/api` route can reach, by
any path"* and the source comment says *"Nine today; a tenth reddens."* The implementation matches
`ast.unparse(child.value)` against the literal set `{"services", "shop._services"}` in the router and
`{"self._services"}` in each `Storefront` method, so a value bound to a local name escapes it:

- **P13-B, survived (183 passed):** `Storefront.enqueue_turn` written as `svc = self._services` /
  `return svc.start_workflow_run(ctx)`, called from the router — §5.1 S9's decided shape with one
  extra line. The control with `self._services.start_workflow_run(ctx)` written directly on the same
  injection point reddens (**1 failed**), so the difference is purely the alias.
- **P13-C, survived (183 passed):** a *second* router alias, `svc2 = shop._services` /
  `svc2.start_workflow_run(...)`. The `services == shop._services` binding control passes, because it
  checks a *rename* of the existing binding, not an *added* one.

This is P12-1's consequence intact: S9's implementer can meet §5.1's `enqueue_turn` interface, add the
trigger, and see green — and the eleven `INHERITED_HANDLERS` excuses revert to prose at the moment
three of them become falsifiable. **Suggested fix (verified, not proposed):** derive the prefixes
instead of listing them — a fixpoint over `ast.Assign` nodes whose `ast.unparse(value)` is already a
prefix, seeded with `{"shop._services"}` in the router and `{"self._services"}` in each method. I ran
it: it returns **the identical nine names** on the clean tree (so `SERVICE_LAYER_REACH_TODAY` needs no
re-baselining) and catches all four spellings — direct, router alias, method alias, and a two-hop
alias. The seeded literal `"services"` becomes derived, which makes the binding control structural
rather than a hand-written assumption. Pin P13-B and P13-C as synthetic controls beside the three
already there.

#### P13-2 · **Major** · the bare-`HTTPException` exemption is still broader than the walk: `services.py` is on the request path, and the reach guard does not cover what it raises

`INHERITED_HANDLERS[StarletteHTTPException]`'s reason is *"No `/shop/api` route raises a bare
`HTTPException`"*, and the widened guard reads `storefront_api.py` + `storefront.py`, stating that it
*"stops at the `services.py` boundary, **which the reach guard above covers instead**"*. That last
clause is not true: the reach guard measures **which** `Services` methods a route reaches, not **what
they raise**. Nothing in either file constrains the latter.

- **P13-A, survived (183 passed):** `raise HTTPException(status_code=410, detail="gone")` on a dead
  branch of `services.save_profile` — one of the nine names `SERVICE_LAYER_REACH_TODAY` itself
  lists, reached from `Storefront.join`. Driven through the wire it answers
  `410 '{"detail":"gone"}'` from `POST /shop/api/session`: no `error` token, no row, green through
  both halves of the gate. **Byte-identical to the answer S8d2 used to justify widening past Pass
  12** — the same argument, one layer further down.
- Secondary and latent: the router binds `repo = shop._repo` and calls `repo.list_participants` /
  `repo.reset_all_participants` directly, so `repository.py` is a fourth module on the request path
  under no guard at all. Both methods raise nothing today, so this leg is latent, not exercised.

**Two closures, either of which is sufficient — I verified the first.** (a) *Compose the two guards*:
run `_raised_class_names` over the `Services` methods named in `SERVICE_LAYER_REACH_TODAY`. Measured
today that reports exactly **`{UnknownOrderTransitionError}`**, which is already in
`SERVICE_ERRORS_UNREACHABLE` with a reason — a three-line addition and a one-name constant, with no
whole-module over-approximation of a file the legacy surface shares. (b) *Narrow the rule to the
reach*: reword the exemption to "no storefront route **body**…" and say in the same breath that the
service layer's raises are covered only for the `ServiceError` family, by the partition test. What is
not defensible is the current pairing of the broad sentence with the narrow mechanism.

### Minor

- **P13-3 · the declined `⊆ StorefrontError family` cross-check is worth more than the reason given
  for declining it.** S8d2 declined it as "not independently killable" and as a repeat of P11-7. Both
  reasons are off. P11-7 was about an enumerated `len(...) == 17` *restating* a derived partition;
  this is a cross-check between **two independent sources** — an AST read of the file and the live
  class tree — so it is not a restatement. And it is killable by exactly the mutation shape N-J
  already established as legitimate: add a `raise ValueError(...)` to `storefront.py` **and**
  `"ValueError"` to `STOREFRONT_RAISES_TODAY`, which is green today and red with the cross-check.
  That is not hypothetical maintenance behaviour — extending the allowlist is the cheapest way to
  silence this guard, and only `HTTPException` is separately fenced. The comment above the constant
  argues *"All seven are `StorefrontError` subclasses, and that is what makes `INHERITED_HANDLERS`'
  excuses true of this file"* — the family is fully handled (checked), and the raises are a subset of
  the family (**not** checked). Two lines close the unchecked premise:
  `assert set(STOREFRONT_RAISES_TODAY) <= {c.__name__ for c in _subclasses(storefront.StorefrontError)}`
  — which holds today, in fact as an equality (I checked both sides). The same exposure exists on
  `envelope | {"RuntimeError", "StorefrontPreflightError", "ValueError"}`, whose three members I
  verified are correctly described (a `field_validator`, a wiring-time refusal, a boot-time refusal).

### Nits

- **P13-4 · factory resolution has one silent shape; every other unresolvable shape fails loud.** I
  probed the reader directly: a factory whose `return`s carry no value resolves the whole `raise` to
  **`set()`** — `raise self._boom()` with `def _boom(self): return` contributes nothing at all. By
  contrast a factory returning a local (`e = HTTPException(410); return e`) yields `{"e"}` and an
  aliased import (`raise HX(410)`) yields `{"HX"}`, both of which redden the set equality, and a
  non-`Name`/`Attribute`/`Call` `raise` hits `_named_class`'s `AssertionError`. So the design is
  fail-loud with one gap that requires a factory that cannot actually raise anything — implausible,
  but one line ends it: assert the factory branch contributed at least one name.

### The two claims the brief asked me to check — both confirmed

1. **`769adc3`'s commit message is wrong about its own state.** Confirmed by reading the tree at that
   ref, not the message. `git show 769adc3:falkor-chat/server/tests/test_storefront_api.py` already
   contains the module-wide walk (`assert _raised_class_names(source) == envelope | {...}`, line
   3209), the three-name allowlist with its reasons (3202–3210), the N-M synthetic control
   (3230–3239), and P12-4's package filter in `_subclasses` (2618) with the `gc` dance already gone
   and its replacement documented in the comment at 2726–2727. The message says P12-2 is "STILL
   OPEN". **What "S8d delivered" means in the record:** `769adc3` = P12-1 + P12-2's
   `storefront_api.py`-wide half + P12-4; `1887180` = the widening to `storefront.py`, factory
   resolution, P12-3, and the docstring/comment corrections. A `HISTORY.md` entry written from the
   messages alone would misattribute three of those.
2. **Under `-m live` the table test is deselected, not skipped.** Confirmed:
   `pytest tests/test_storefront_api.py -m live -q` → **`131 deselected`**, zero collected, zero
   skipped. The corrected docstring is right and the previous wording was wrong.

### Is S8d2's guard-reach statement accurate as written? — **No, in two clauses**

I judge the statement in the form that will actually be lifted: the source comment above
`INHERITED_HANDLERS` (`storefront_api.py:433–452`, with the exemption string it governs at
`:455–467`), which the coordination row at
`docs/plans/salesperson-ui-coordination.md:143` paraphrases identically. S8d2's own report text is not
a file I hold; if it differs from the source comment, re-check it against these four rulings.

| Clause | Ruling |
|---|---|
| "over **both storefront modules whole** rather than the router node" | **Accurate** — verified by mutation on both files |
| "resolves `raise <factory>(...)` through the factory's `return`s" | **Accurate**, with P13-4's one silent shape |
| "**It stops at the `services.py` boundary, which the reach guard above covers instead**" | **Inaccurate** — the reach guard covers *which methods*, never *what they raise* (P13-2) |
| "the **whole set of `Services` methods a route can reach** … Nine today; a tenth reddens" | **Inaccurate** — three spellings, not every path; a tenth added through an alias does **not** redden (P13-1) |

**So the §5.1 S9 re-word must not lift the statement verbatim.** Lift it after P13-1 and P13-2 land,
and state the reach the *fixed* mechanism has. If for any reason the re-word must go first, the two
inaccurate clauses have to be narrowed to what is checked, not to what is intended — a plan row that
promises S9's implementer a red guard that stays green is P12-1's failure written into the plan
instead of into the test.

### The three judgement calls S8d2 made — all three upheld

1. **Widening past Pass 12's recommendation was right, and the two mutations are real.** I
   reproduced both on `769adc3`: **N-M2** (bare `HTTPException(410)` on a dead branch of
   `Storefront.join`) and **N-M3** (the same raise in a module-level `storefront.py` helper called
   from `join`) each **survived at 183 passed**, and I drove N-M2 through the wire —
   `POST /shop/api/session → 410 '{"detail":"gone"}'`. Both are killed at `HEAD` by
   `test_the_raises_a_route_can_reach_are_exactly_what_the_exemptions_assume`.
2. **Whole-module over reachability-filtered for `storefront.py` is the right trade.** I measured the
   two: every `raise` in that file today lives in a method a route reaches
   (`advance_own_order`, `join`, `reset_participant`), so the two readings agree on the same seven
   names — S8d2's "measured, not assumed" holds. The over-approximation's named cost (N-M4, a raise
   in `Storefront.lookup`) I reproduced: **1 failed**, as documented. That cost is even smaller than
   stated, because §5.1's S9 row **deletes `lookup` outright** with the record cache.
3. **P12-3's fix is the right shape.** Excluding `live`-marked names from `defined` rather than
   inspecting selector state keeps the predicate keyed on what was collected — the P11-9 lesson —
   and does not need the `-m live` case special-cased, since the whole module deselects there.

### Disposition of Pass 12's findings

| # | Disposition | What I rechecked |
|---|---|---|
| P12-1 | **Fixed for the three spellings it names; the defect class survives in a fourth** → **P13-1** | `direct_sf` **1 failed**; `alias_sf` **survived, 183**; `alias_router` **survived, 183** |
| P12-2 | **Fixed, and correctly widened past the recommendation — but still short of its own rule** → **P13-2** | N-M2/N-M3 survive on `769adc3` (183 each), both die at `HEAD`; P13-A survives at `HEAD` (183) and answers `410 '{"detail":"gone"}'` |
| P12-3 | **Fixed** | `-m live` → `131 deselected`, none skipped; `defined` excludes `pytestmark` `live` names |
| P12-4 | **Fixed** | `_subclasses` filters on `__module__.partition(".")[0] == "falkorchat"`; the minted subclass is in `ThreadNotFoundError.__subclasses__()` and absent from `_subclasses(ServiceError)`; no `gc` import anywhere in the file |

**Pass 12's evidence-discipline claim, upheld and extended.** S8d2 claims no figure it reports has
N-K's seed-dependent shape because every kill is a set-equality assertion. I checked rather than
accepted: both guard tests assert set equality or `in`/`not in` on a set, which is order-independent
by construction, and N-M2 gives **`1 failed` at `PYTHONHASHSEED` 0, 3, 4 and 7** — four of the eight
seeds Pass 12 used, including seed 4, the one where N-K survived the whole file. Every survival I
report is also a set-equality miss, so the survivals are as deterministic as the kills.

### What's solid

**S8d2 did the thing this build keeps failing to do — it asked whether the rule was broader than the
reach, one file out, and it was.** Pass 12 wrote "checked not guessed" about a scope, reproduced a
case, fixed the spelling that case exhibited, and stopped. S8d2 declined to apply the recommendation
it was handed and produced two wire-level survivors instead of an argument. That is the correct
reviewer-of-a-review behaviour, and the coordination note's framing of it ("*checked, not guessed*
names the method, not the scope") is the durable part.

**The synthetic controls are the right kind.** Every widening in `1887180` ships with a snippet that
exercises the reader on the exact mutation shape it was widened for, and `_raised_class_names` taking
a node *or* a source string is what makes that cheap. My own fix proposals for P13-1 slot in beside
them without new machinery.

**The three allowlisted `storefront_api.py` raises are described accurately.** I read all three sites:
`_nonblank`'s `ValueError` really is inside a `field_validator`, `register_storefront_error_handlers`'
`RuntimeError` really is wiring-time, and `storefront_preflight`'s `StorefrontPreflightError` really
is boot-time. The names are not tolerated, they are argued.

**`769adc3` was a coherent thing to commit.** A partial unit committed with an honest "PARTIAL,
UNGATED" header cost this pass nothing and let me build a real baseline; that it *understated* its own
delivery is a defect of the message, not of the decision to commit.

### Open questions (need `teco`'s call)

1. **Does P13-1 + P13-2 go in as an S8e, or as S9's first done-condition?** My recommendation:
   **S8e, before S9 dispatches**, on the coordinator's own argument — a guard authored inside the
   step it guards is born accommodating, and that argument is now two passes old and has been right
   twice. Both fixes are in one test file, and P13-1's is verified to leave
   `SERVICE_LAYER_REACH_TODAY` unchanged, so the round is short.
2. **Is P13-2 closed by widening (a) or by narrowing the rule (b)?** That is a scope call, not a
   review call. (a) buys a real check for three lines; (b) costs nothing and makes the exemption
   honest. What must not survive is the current pairing.
3. **Does the §5.1 S9 re-word wait for S8e?** It should. The statement it would lift is inaccurate in
   two clauses today, and correcting the plan twice is worse than dispatching the re-word once, after
   the mechanism matches its sentence.

### Appendix P13-A — mutation ledger and transcripts (Pass 13)

**Method.** Every mutation applied from a byte-copy held outside the repo and restored from that copy
after each run, `md5sum` re-verified each time. Command:
`.venv/bin/python -m pytest tests/test_storefront_api.py tests/test_app.py -q`. **Baseline: 183
passed** (`HEAD`), full suite **2615 passed, 14 deselected**. Baselines at `769adc3` were materialised
with `git show 769adc3:<path> > <path>` and restored from the `HEAD` byte-copies; no `git`
tree-mutating command was used at any point.

| # | Tree | Mutation | Result |
|---|---|---|---|
| **N-M2** | `769adc3` | bare `HTTPException(410)` on a dead branch of `Storefront.join` (`storefront.py`) | **survived, 183 passed** |
| **N-M3** | `769adc3` | module-level `_refuse_retired_name` in `storefront.py` raising `HTTPException(410)`, called from `join` | **survived, 183 passed** |
| N-M2 | `HEAD` | same | **1 failed** — the raise guard, at seeds 0/3/4/7 |
| N-M3 | `HEAD` | same | **1 failed** — the raise guard |
| N-M4 | `HEAD` | `HTTPException(410)` in `Storefront.lookup`, which no route reaches | **1 failed** — the documented over-approximation, reproduced |
| **P13-A** | `HEAD` | bare `HTTPException(410)` on a dead branch of `services.save_profile` | **survived, 183 passed** → P13-2 |
| direct_sf | `HEAD` | `Storefront.enqueue_turn` → `self._services.start_workflow_run`, called from the router | **1 failed** — the reach guard (control) |
| **P13-B** (alias_sf) | `HEAD` | the same, written `svc = self._services` / `svc.start_workflow_run(ctx)` | **survived, 183 passed** → P13-1 |
| **P13-C** (alias_router) | `HEAD` | a second router alias `svc2 = shop._services` / `svc2.start_workflow_run(...)` | **survived, 183 passed** → P13-1 |

**Probe 1 — N-M2 and P13-A driven through the wire.** Mutation in place, one throw-away test appended
to `tests/test_storefront_api.py`, run under `-k`, then both files restored from the byte-copies:

```
N-M2  (769adc3)  POST /shop/api/session -> 410 '{"detail":"gone"}'
P13-A (HEAD)     POST /shop/api/session -> 410 '{"detail":"gone"}'
```

**Probe 2 — the raises a route reaches, one layer past the guard's boundary.** `_raised_class_names`
(lifted verbatim out of the test file) applied to each `Services` method named in
`SERVICE_LAYER_REACH_TODAY`:

```
advance_order: ['UnknownOrderTransitionError']   filter_products / get_cart / get_current_order /
get_profile / order_belongs_to_customer / post_message / read_messages / save_profile: []
union: ['UnknownOrderTransitionError']           (already in SERVICE_ERRORS_UNREACHABLE)
repo.<name> in the router: ['list_participants', 'reset_all_participants'] — both raise nothing today
```

**Probe 3 — the alias-resolving reader, run before being proposed.** A fixpoint over `ast.Assign`
whose unparsed value is already a prefix, seeded `{"shop._services"}` / `{"self._services"}`:

```
clean tree   : identical to SERVICE_LAYER_REACH_TODAY (nine names)  -> True
alias_sf     -> {'start_workflow_run'}      alias_router -> {'start_workflow_run'}
direct_sf    -> {'start_workflow_run'}      two-hop alias -> {'start_workflow_run'}
```

**Probe 4 — whole-module vs reachability-filtered on `storefront.py`.**

```
reached methods that raise: advance_own_order, join, reset_participant
unreached methods that raise: (none) — __init__, cached_ids, clear_turn, forget, lookup,
  set_turn_state, storefront_dir, turn_workers all raise nothing
whole-module union == STOREFRONT_RAISES_TODAY == live _subclasses(StorefrontError) name set (7)
```

**Probe 5 — the raise reader's edge shapes.**

```
bare-return factory  raise self._boom(); def _boom(self): return      -> set()      (silent)
var-return factory   def _boom(self): e = HTTPException(410); return e -> {'e'}     (loud)
aliased import       from fastapi import HTTPException as HX; raise HX -> {'HX'}    (loud)
```

**Probe 6 — the `-m live` correction.**

```
pytest tests/test_storefront_api.py -m live -q  ->  131 deselected  (zero collected, zero skipped)
```

**Hypotheses ruled out** (so the next pass does not re-walk them):

- *The widening to `storefront.py` was unnecessary — Pass 12's module-wide walk of
  `storefront_api.py` was enough.* **No.** N-M2 and N-M3 both survive `769adc3` at 183 passed and
  both answer `410 '{"detail":"gone"}'` on the wire.
- *Whole-module reading of `storefront.py` over-approximates in a way that will bite.* **Not today,
  and less tomorrow.** Every raise in the file is in a route-reached method, and the one documented
  false-red (`lookup`) is a method S9 deletes.
- *`769adc3`'s message is right and S8d2 misread the tree.* **No.** The four features are at named
  lines in that ref's own test file.
- *The survivals I report might be seed-dependent, like N-K.* **No.** Every one is a set-equality
  miss, order-independent by construction; N-M2's kill was checked at four seeds including seed 4.
- *A raise inside `storefront.py`/`storefront_api.py` could still hide from the reader through an
  alias or an unusual `raise` expression.* **Only one shape, and it cannot raise** — probe 5. Every
  other unresolvable shape lands a foreign name in the set or trips `_named_class`'s
  `AssertionError`.
- *`repository.py`'s direct use from the router is an exercised hole.* **Latent, not exercised** —
  `list_participants` and `reset_all_participants` raise nothing today.

## Pass 14 — 2026-09-07 (S8e: is the *derivation*'s own reach as wide as its sentence?)

**Reviewed:** commit **`92bf842`** (S8e) — `falkorchat/storefront_api.py` (+45/−24, comment and one
reason string) and `tests/test_storefront_api.py` (+269/−41) — against
`docs/plans/salesperson-ui.md` **v1.23** §4.9, §5.1 S8/S9/S10, §5.2, §5.3, against `## Pass 13`'s
two majors, minor and nit, and against `docs/plans/salesperson-ui-coordination.md`'s rows 143–145
and its two 2026-09-07 sections. Not reviewed: S9/S10/S12 content, the SPA, the held
`HISTORY.md`/`SERVER.md` documentation debt.

**I am a fresh reviewer.** I wrote neither Pass 12 nor Pass 13; I read both as documents and
re-derived every claim of theirs I rely on below.

**CPG: considered, not relevant — `cpg_falkorchat` models none of the files in scope (0 `File`
nodes matching `storefront`, measured by Pass 12, accepted by Pass 13, and the brief asked me not
to re-spend on it), so every claim below is from direct read, AST analysis lifted verbatim out of
the delivered test file, or source mutation against the live suite.**

**Verdict: needs changes** — **0 blockers, 3 majors, 1 minor, 1 nit**. S8e's own work is the best
in this chain: the fixpoint is a real structural answer, the four-leg application went looking for
the sibling instance instead of waiting for a gate, the P13-3 reversal is correct, and **every
figure S8e reports reproduced exactly** — including the two-file 183, P13-3's 183/1-failed A/B, and
the seed-independence claim, which I confirmed at all eight seeds. But the answer to *"is the class
now closed?"* is **no**, in the same shape for the fourth consecutive pass and **twice inside
S8e's own fix**:

- **P14-1 (the eleventh instance).** The composed raise walk stops one hop *earlier* than the
  sentence governing it. A bare `HTTPException(410)` in a `Services` sibling helper reached from
  `save_profile`, and one in `Repository.ensure_participant` reached from `Storefront.join`, each
  **survive at 183 passed** and each answer `410 '{"detail":"gone"}'` from `POST /shop/api/session`
  — Pass 13's P13-A, one layer down, twice.
- **P14-2 (the twelfth).** `_alias_prefixes` closes over `ast.Assign` only, while the statement says
  "any local name transitively bound to it". `svc: object = self._services` — S9's decided shape
  **plus one type annotation** — **survives at 183**; drop the annotation and it is `1 failed`.
  Annotated local binding is a house idiom in the very files S9 edits (6 in `storefront.py`, 16 in
  `storefront_api.py`).
- **P14-3.** The composition claim that a separate unit will lift into §5.1's S9 row is **wrong
  about the plan it cites**: the walk reports *one* of the S9 row's three excuses, not two, plus a
  fourth the row does not name — and the row's own third is unaccounted for because the **plan**
  mis-names it.

All three are test-and-prose defects; production behaviour is untouched, which is why none is a
blocker. All three must land before S9 dispatches and before the §5.1 re-word lifts anything.

### What I ran (all of it, solo, serially)

Two-file baseline **183 passed** (`tests/test_storefront_api.py` + `tests/test_app.py`), my own run.
**Eight mutations** and **two wire probes**, each applied from byte-copies held outside the repo
and restored from them, `md5sum` re-verified after every run — `storefront.py` `a713e2c5…`,
`services.py` `a952f4ad…`, `repository.py` `584ce30c…`, `app.py` `e6bf735a…`, `storefront_api.py`
`3bddcc7e…`, `test_storefront_api.py` `15874a7f…`, `test_app.py` `fcf877ac…`, all matching `HEAD`
at the end and `git diff --stat -- falkor-chat/` empty. **Eleven extra suite runs across
`PYTHONHASHSEED` 0–7** to settle the evidence-discipline claim. Six static probes run against the
delivered readers **lifted verbatim** out of `tests/test_storefront_api.py` by `ast.get_source_segment`,
so nothing below is a paraphrase of the mechanism. `ruff check` on both touched files: clean.
**No `git` tree-mutating command at any point.** Ledger and transcripts: **Appendix P14-A**.

**I did not re-run the full suite.** The coordinator ran it (2615/14) and I report only figures I
observed; every mutation here is two-file-scoped, and a second full run would wipe `reference`
again for no evidence I need.

**Database.** `ws:test` only. `ws:acme` re-checked at the end and unchanged — **871 nodes**,
`Message` 52, `Entity` 544, `WorkflowRun` 21. `reference` was **already empty when I started** (the
coordinator's own documented wipe) and I left it exactly so — my two-file runs neither seeded nor
wiped it, and `teco`'s own note says the re-seed belongs after the last suite of the chain, not
before. No stray `timers-stale-key@v1` remains; `reference` holds zero nodes.
`seed_workflows.sh`/`seed_salesperson.sh` never run.

### Major

#### P14-1 · **Major** · the composed raise walk stops one hop earlier than its exemption claims — the eleventh instance, and it is exercised today, not latent

`INHERITED_HANDLERS[StarletteHTTPException]`'s reason (`storefront_api.py:475–478`) now reads *"no
bare `HTTPException` anywhere the guard above reaches — `storefront_api.py` and `storefront.py`
whole, plus the **`Services`/`Repository` methods a route reaches**"*. A route reaches more of both
than the walk reads, which lists nine `Services` names and the router's two `repo.<name>` calls:

- **P14-M1, survived (183 passed):** `Services._refuse_retired_name()` raising
  `HTTPException(410)`, called from `save_profile` — a sibling method of one of the nine. Wire:
  `POST /shop/api/session → 410 '{"detail":"gone"}'`.
- **P14-M2, survived (183 passed):** the same raise on a dead branch of
  `Repository.ensure_participant`, reached from `Storefront.join`, not from the router. Wire:
  identical `410 '{"detail":"gone"}'`.

The same comment states the principle it then abandons — *"a raise one helper call out of a route
body is a raise on the route"* — and applies it whole-module to the two storefront files only. It
is not a theoretical boundary: the nine reached methods raise **five** classes on the request path
today, not one (`RuntimeError`, `ThreadNotFoundError`, `UnknownActorError`, `UnknownMemberError` in
`post_message`'s two helpers, plus `UnknownOrderTransitionError`), so `SERVICE_RAISES_TODAY`'s own
headline — *"Everything the reached methods … raise"* — is measurably false. And the blind spot is
already hiding an **unclassified** reachable raise: `Repository.ensure_participant` raises
`MemberIdCollisionError`, which is in no §5.2/§5.3 row, no `SERVICE_ERROR_RESPONSES` entry and no
`INHERITED_HANDLERS` excuse.

**Suggested fix — measured, not proposed (Appendix P14-A, probe 4).** Reuse the frontier idiom
`_service_layer_reach` already contains: close the walked method set over `self.<name>` calls to a
fixpoint. On `Services` that grows 9 → 13 and yields exactly the five names above; on `Repository`,
adding the four methods `Storefront` reaches yields exactly `{MemberIdCollisionError}`. ~8 lines of
reader, five names and one name in the two constants — and it makes P14-3 go away for free.

#### P14-2 · **Major** · the fixpoint follows `ast.Assign` only, while the statement says "any local name transitively bound to it" — a type annotation is enough to hide S9's shape again

`_alias_prefixes` collects `(target.id, ast.unparse(child.value))` for `ast.Assign` nodes with
`ast.Name` targets. Six other ways of binding a local name to the service object are invisible.
Measured against the delivered reader (probe 2), on the `self._services` leg:

| shape | seen? |
|---|---|
| `svc = self._services` · `a = b = self._services` | seen (the controls) |
| `svc: object = self._services` · `svc, _ = self._services, None` · `(svc := self._services)` · `for svc in (self._services,)` · `with self._services as svc` · comprehension target | **missed** |

- **P14-M3a, survived (183 passed):** `svc: object = self._services` / `svc.start_workflow_run(...)`
  on the router-reached `Storefront.join`, i.e. S9's decided shape plus a type annotation.
- **P14-M3b, `1 failed`:** byte-identical but for `: object`. The difference is the annotation alone.

This is not an exotic spelling: the package carries **68** annotated local assignments, **6** in
`storefront.py` and **16** in `storefront_api.py`, and `storefront.py:966` is literally
`state: dict[str, Any] | None = self.get_state(ctx)`. So §5.1's S9 done-condition ("the assertion
goes red on this step") is once again unmeetable for a variation one token away from the shape S8e
was built for. **Fix (checked):** widen the binding harvest to `ast.AnnAssign`, `ast.NamedExpr`,
`ast.For`/`ast.comprehension` targets and `ast.withitem` `optional_vars`, and to `ast.Tuple`/`List`
targets when the value is a matching tuple — or, if that is judged over-built, **narrow the
sentence to "a local name bound by a plain assignment"** and pin the missed shapes as documented
non-reach, which is the (b) move Pass 13 named and this comment is otherwise careful to make.

#### P14-3 · **Major** · the composition claim is precise about the code and wrong about the plan row it cites — and that sentence is the one being lifted

The test docstring (`tests/test_storefront_api.py:3417–3423`) says the walk will report
`WorkflowInputRejectedError` and `WorkflowRunNotFoundError` — *"two of the three `INHERITED_HANDLERS`
excuses §5.1's S9 row says become falsifiable at that moment"*. I measured the walk: it reports
exactly those two (probe 1). But §5.1's S9 row names **`WorkflowEngineDisabledError`,
`WorkflowInputRejectedError` and `WorkflowDefNotFoundError`**. So it is *one* of the row's three,
plus a **fourth** excuse the row does not name; the row's third is dropped from the accounting
silently, and it can never be reported — `WorkflowDefNotFoundError` is raised only in
`materialize_def`, `get_workflow_def_structure` and the diff method, none reachable from
`start_workflow_run`.

The root cause is in the **plan**, not the guard: §5.1's S9 parenthetical *"(`services.py`
`_require_executor`, the `run_ctx` bound, the missing snapshot)"* maps *the missing snapshot* to
`WorkflowDefNotFoundError`, but `services.py:2085` raises `WorkflowRunNotFoundError` there. S8e read
the code correctly and then described it against a row that is wrong — which is exactly the failure
mode this chain keeps producing, one document over.

**Fix, two parts.** (1) `architect` corrects §5.1's S9 row to name `WorkflowRunNotFoundError` for the
missing snapshot and states that `WorkflowDefNotFoundError`'s excuse is untouched by S9. (2) The
docstring sentence names the exceptions, not a count: *"reports `WorkflowInputRejectedError` and
`WorkflowRunNotFoundError`; `WorkflowEngineDisabledError` is raised in `_require_executor`, which
this walk does not enter; `WorkflowDefNotFoundError` is unreachable from `start_workflow_run` and its
excuse is unaffected."* **P14-1's fix subsumes the concession** — with the sibling closure, the walk
reports `WorkflowEngineDisabledError` too (probe 4, measured), so two of the row's three become
mechanically measured and the "prose argument" disappears rather than being argued.

### Minor

- **P14-4 · P13-3's cross-check landed on one of the three allowlists, and the same commit
  introduced the other two.** `set(STOREFRONT_RAISES_TODAY) <= {StorefrontError family}` closes the
  allowlist-extension escape for `storefront.py` (verified: **P14-M4a `1 failed` with it, P14-M4b
  183 passed without it** — S8e's A/B reproduces exactly). `SERVICE_RAISES_TODAY` and
  `REPOSITORY_RAISES_TODAY` carry the identical exposure with no cross-check: **P14-M6, survived
  (183 passed)** — `raise ValueError` added to `services.save_profile` **plus** `"ValueError"` added
  to `SERVICE_RAISES_TODAY`, which is the cheapest way to silence the new leg. `services.py`'s
  reached raises are not all `ServiceError`s, so the cross-check cannot be copied verbatim; the form
  that fits is the one this file already uses for `storefront_api.py` —
  `<= {ServiceError family} | {named non-family allowances, each with its reason}`, which today is
  `{RuntimeError}` under P14-1's fix and `∅` without it.

### Nit

- **P14-5 · "duplicated so the two cannot drift" is currently unfalsifiable in the tree.**
  `grep -rn "any local name transitively bound"` over `*.md` and `*.py` returns **one** hit —
  `storefront_api.py:446`. There is no second in-repo copy to drift from; S8e's report text lives in
  a transcript. That is fine today, and it is exactly the moment to say what binds them tomorrow:
  the §5.1 S9 re-word *creates* copy two, and nothing will hold them together. One line in the
  comment naming §5.1's S9 row as its only licensed copy, and a plan cell that cites
  `storefront_api.py:437–471` rather than restating it, costs nothing now and is unwritable later.

### Is S8e's corrected guard-reach statement accurate as written? — **No, in four clauses of eleven**

I judge the statement in the form that will be lifted: the source comment at
`storefront_api.py:437–471`, plus the two docstring paragraphs it summarises
(`_service_layer_reach`'s *Where it stops*, `tests/test_storefront_api.py:3074–3078`; the raise
test's, `:3426–3433`) and the `INHERITED_HANDLERS[StarletteHTTPException]` reason string it governs.

**Reach guard (comment lines 442–456 — `storefront_api.py`):**

| # | Clause | Ruling |
|---|---|---|
| 1 | "(1) `<prefix>.<name>` anywhere in `build_storefront_router`, where `<prefix>` is `shop._services` **or any local name transitively bound to it**" | **Inaccurate** — `ast.Assign`/`ast.Name` only; six binding forms missed (P14-2) |
| 2 | "(2) the same, with prefix `self._services`, in every `Storefront` method the router reaches through `shop.<method>`/`self.<method>` calls, **transitively**" | **Accurate for the frontier**, verified; **inherits clause 1's defect for the prefix** — P14-M3a is this leg |
| 3 | "**Nine today**" | **Accurate** — the derived reader returns exactly `SERVICE_LAYER_REACH_TODAY` (probe 1) |
| 4 | "…**and a tenth added by any of those paths reddens**" | **Inaccurate** — P14-M3a adds a tenth on path (2) and stays green |
| 5 | "Prefixes are **derived from the files' own bindings**, not listed" | **Over-broad** — derived from the files' `ast.Assign` bindings |
| 6 | *Where it stops #1*: "it follows the service object only through attribute access, so a call made by handing that object somewhere else — **passed to a helper, returned, stored** — is outside it" | **Accurate for the three it names** (probe 2: `_go(self._services)` and `self._svc = self._services` both missed) but **not exhaustive**, and "stored" is ambiguous against a mechanism that *does* follow a local store and does not follow an attribute store |
| 7 | "That is a statement about the walk, not a claim about the code; the moment one is written, this comment is what has to change with it" | **Accurate and honestly framed** — nothing enforces it, and it does not pretend otherwise |

**Raise guard (comment lines 458–471, and the reason string at 475–478):**

| # | Clause | Ruling |
|---|---|---|
| 8 | "over four scopes: `storefront_api.py` and `storefront.py` **whole**" | **Accurate** — the file-whole reading verified; Pass 13's N-M2/N-M3 die at `HEAD` |
| 9 | "plus **the `Services` methods the reach guard above measures** and **the `Repository` methods the router calls through `repo.<name>`**, read at those methods only" | **Accurate literally** — this is the narrow, true reading, and it is the one the mechanism implements |
| 10 | *Where it stops #2*: "**It stops at what those collaborators call in turn** — `Services` into `Repository`, `Repository` into redis" | **Inaccurate** — it also stops at `Services` into `Services` and at `Storefront` into `Repository`, both measured escapes answering `410` on the wire (P14-1) |
| 11 | reason string: "no bare `HTTPException` anywhere the guard above reaches — … plus **the `Services`/`Repository` methods a route reaches**" | **Inaccurate** — a route reaches `Services._validate_and_derive_role` and `Repository.ensure_participant`; both are outside the walk (P14-M1, P14-M2) |

**So the §5.1 S9 re-word must not lift the statement verbatim, for the third gate running.** Clauses
3, 6, 7, 8 and 9 are the durable ones and can be lifted today. Clauses 1, 4, 10 and 11 must either
be narrowed to what is checked or backed by P14-1's and P14-2's fixes first; clause 5 needs four
words. The composition sentence must not be lifted at all until P14-3 is settled, because it is the
only clause that would put a **wrong exception name** in front of S9's implementer.

### The six claims the brief asked me to check

| # | Claim | Ruling |
|---|---|---|
| 1 | P13-1 closed by derivation; four legs; nine names need no re-baselining | **Confirmed for the derivation** (probe 1: derived reach ≡ the nine; probe 3: the four legs are alias-resolved) — **but the derivation's own reach is narrower than its sentence** → P14-2 |
| 2 | P13-2 closed by taking (a) *and* narrowing the sentence; four scopes | **Confirmed on the wire for the scopes named** — but the narrowed sentence is still broader than the four scopes → P14-1 |
| 3 | The `repo.<name>` leg cannot empty silently | **Confirmed** (probe 5): a post-S10 router yields `∅`, `assert repo_reach` fires, and without it `_raises_of(…, ∅) == REPOSITORY_RAISES_TODAY` would pass. The assertion is load-bearing |
| 4 | The two-of-three composition split | **Confirmed about the code, wrong about the plan row** → P14-3 |
| 5 | P13-3 adopted and killable — 183 without, 1 failed with | **Confirmed exactly** (P14-M4a/b). The reversal reasoning is right: two independent sources, not a restatement |
| 6 | P13-4 — the factory branch errors rather than resolving to nothing | **Confirmed** (probe 6): the bare-return factory raises `AssertionError`; the value-returning and aliased-import shapes stay loud |

**Evidence discipline, checked rather than accepted.** S8e claims every assertion involved is a set
equality or `not in` on a set and therefore order-independent. I confirmed by reading both guard
bodies, and measured **P13-A dying `1 failed, 182 passed` at all eight `PYTHONHASHSEED` values 0–7** —
including seed 4, where Pass 12 measured N-K surviving the entire file. My two headline **survivals**
are equally deterministic: P14-M1 and P14-M3a both give **183 passed at seeds 4 and 7** as well as at
the default. No figure in this pass is one draw from a distribution.

### Disposition of Pass 13's findings

| # | Disposition | What I rechecked |
|---|---|---|
| P13-1 | **Fixed for `ast.Assign`; the defect class survives in six other binding forms** → **P14-2** | P14-M3b (plain assign) **1 failed**; P14-M3a (annotated) **survived, 183**; probe 2 enumerates the six |
| P13-2 | **Fixed for the four scopes named; the sentence still outruns them by one hop** → **P14-1** | P14-M1 **survived, 183**, wire `410 '{"detail":"gone"}'`; P14-M2 **survived, 183**, same wire answer |
| P13-3 | **Fixed for `STOREFRONT_RAISES_TODAY`; not applied to the two constants the same commit added** → P14-4 | P14-M4a **1 failed** / P14-M4b **183 passed**; P14-M6 **survived, 183** |
| P13-4 | **Fixed** | Probe 6 — bare-return factory now raises `AssertionError` with the named message |

### What's solid

**The fixpoint is the right *kind* of answer, and the coordination note is right that this is the
first structural move in the chain.** The tell it names holds up under my own measurement rather
than by report: the derived reader returns `SERVICE_LAYER_REACH_TODAY` unchanged, so the change is
purely in what the reader *can* see. P14-2 narrows what that is; it does not make the move wrong.

**Applying the fixpoint on all four legs, unasked, is the behaviour the pattern should induce.** Pass
13 named two legs. S8e widened to four on an argument about the shape of the defect rather than the
shape of the report, and that argument is correct — the frontier walks are the same defect one field
over.

**The P13-3 reversal is worth more than being right the first time.** S8e's own earlier P11-7 analogy
was wrong, it re-derived the rebuttal, and it produced the A/B. Both halves reproduced for me at the
reported figures.

**Every leg that could empty silently now says so.** `assert repo_reach`, `_raises_of`'s `missing`
assertion, and P13-4's factory assertion are three different silent-empty shapes closed the same
way — a failed assertion with a sentence explaining what to do. That habit, not any one of the
three, is the durable part.

**And the self-report is real.** S8e retracted its own S8d clause (*"it stops at the `services.py`
boundary, which the reach guard above covers instead"*) in the file, in the place a reader meets it.
P14-1 says the replacement is still one hop short — but it is short by one hop, not by a layer.

### Open questions (need `teco`'s call)

1. **Does P14-1 + P14-2 go in as an S8f, or as S9's first done-condition?** My recommendation:
   **S8f, before S9 dispatches** — the same argument, now four passes old and right three times. But
   this is the point to weigh the alternative honestly: **for P14-2, narrowing the sentence is a
   defensible whole answer.** The escape needs a deliberate annotation on a two-line idiom, and S9's
   implementer will be reading a done-condition that says the guard must redden. P14-1 is different
   — it is exercised on the wire today and hides an unclassified raise, so it needs the mechanism.
2. **Who fixes §5.1's S9 parenthetical (P14-3)?** It is an `architect` edit to the plan, and it is a
   *prerequisite* of the re-word unit rather than part of it: the re-word cannot state the
   composition correctly against a row that names the wrong exception.
3. **Does the §5.1 S9 re-word wait again?** It should, for the third time — but the hold is now
   cheaper than it looks: clauses 3, 6, 7, 8 and 9 are lift-ready today and are most of the
   statement's useful content. Lifting those five now and the rest after S8f would end the hold
   without lifting anything false.

### Appendix P14-A — mutation ledger, probes and transcripts (Pass 14)

**Method.** Every mutation applied from a byte-copy held outside the repo and restored from that copy
after each run, `md5sum` re-verified each time. Command:
`.venv/bin/python -m pytest tests/test_storefront_api.py tests/test_app.py -q`. **Baseline: 183
passed.** Static probes ran against the delivered readers lifted out of `tests/test_storefront_api.py`
with `ast.get_source_segment`, never re-typed. No `git` tree-mutating command was used at any point.

| # | Mutation | Result |
|---|---|---|
| **P14-M1** | `Services._refuse_retired_name()` raises `HTTPException(410)`; `save_profile` calls it | **survived, 183 passed** (also at seeds 4, 7) → P14-1 |
| **P14-M2** | bare `HTTPException(410)` on a dead branch of `Repository.ensure_participant`, reached from `Storefront.join` | **survived, 183 passed** → P14-1 |
| **P14-M3a** | `svc: object = self._services` / `svc.start_workflow_run(...)` on `Storefront.join` | **survived, 183 passed** (also at seeds 4, 7) → P14-2 |
| P14-M3b | the same, written `svc = self._services` | **1 failed** — the reach guard (the control: the annotation is the whole difference) |
| P14-M4a | `raise ValueError` in `storefront.py` **+** `"ValueError"` in `STOREFRONT_RAISES_TODAY` | **1 failed** — P13-3's cross-check, as S8e reports |
| P14-M4b | the same, with the cross-check line deleted | **183 passed** — S8e's A/B reproduces exactly |
| **P14-M6** | `raise ValueError` in `services.save_profile` **+** `"ValueError"` in `SERVICE_RAISES_TODAY` | **survived, 183 passed** → P14-4 |
| P13-A | bare `HTTPException(410)` in `services.save_profile`'s **body** (Pass 13's escape) | **1 failed, 182 passed at every seed 0–7** |

**Probe 1 — the delivered readers against the delivered files.**

```
SERVICE_LAYER_REACH_TODAY == _service_layer_reach(api, sf)          -> True  (nine names)
_raises_of(services, "Services", the nine)   -> ['UnknownOrderTransitionError']  == SERVICE_RAISES_TODAY
_router_repository_reach(api)                -> ['list_participants', 'reset_all_participants']
_raises_of(repository, "Repository", those)  -> []                              == REPOSITORY_RAISES_TODAY
_raises_of(services, "Services", {"start_workflow_run"})
                     -> ['WorkflowInputRejectedError', 'WorkflowRunNotFoundError']
```

The last line is P14-3's measurement. §5.1's S9 row names `WorkflowEngineDisabledError`,
`WorkflowInputRejectedError`, `WorkflowDefNotFoundError`; `services.py:2085` raises
`WorkflowRunNotFoundError` for the missing snapshot, and `WorkflowDefNotFoundError` occurs only at
`services.py:1752/1814/1869` (`materialize_def`, `get_workflow_def_structure`, the diff), none
reachable from `start_workflow_run`.

**Probe 2 — every way of binding the service object, against `_alias_prefixes` (P14-2).** Router
stub + one `Storefront` method, reach expected `{"start_workflow_run"}`:

```
SEEN   self._services.start_workflow_run(ctx)          (control)
SEEN   svc = self._services
SEEN   a = b = self._services
MISSED svc: object = self._services                     <- P14-M3a, a house idiom
MISSED svc, _ = self._services, None
MISSED (svc := self._services)
MISSED for svc in (self._services,)
MISSED with self._services as svc
MISSED [s.start_workflow_run(ctx) for s in [self._services]]
MISSED def _go(svc): ...; _go(self._services)           (licensed by "passed to a helper")
MISSED self._svc = self._services                       (licensed by "stored")
```

Idiom census, `ast` over `falkorchat/`: **68** annotated local assignments (`storefront.py` 6,
`storefront_api.py` 16, `services.py` 23), 40 tuple-target assignments, 1 walrus.
`storefront.py:966` is `state: dict[str, Any] | None = self.get_state(ctx)`.

**Probe 3 — what a route actually reaches one hop past the walk (P14-1).**

```
sibling self.<name> calls from the nine reached Services methods:
  get_cart      -> _priced_cart_lines
  post_message  -> _validate_and_derive_role, _dispatch_write, _next_ts
  (start_workflow_run, at S9 -> _require_executor, _reject_reserved_keys, _drive_or_fault, …)
their raises: RuntimeError, ThreadNotFoundError, UnknownActorError, UnknownMemberError
              (+ WorkflowEngineDisabledError, WorkflowInputRejectedError at S9)
  Services._validate_and_derive_role -> ThreadNotFoundError, UnknownActorError, UnknownMemberError
  Services._dispatch_write           -> RuntimeError, ThreadNotFoundError, UnknownActorError
Repository reached from Storefront (not from the router):
  join -> ensure_participant, set_participant_record ; lookup/resolve_token -> get_participant_record
  reset_participant -> reset_participant
  their raises: MemberIdCollisionError   (Repository.ensure_participant — in no table anywhere)
```

**Probe 4 — the suggested fix, run before being suggested.** Closing the walked method set over
`self.<name>` calls to a fixpoint, reusing `_service_layer_reach`'s own frontier idiom:

```
Services closure 9 -> 13  (+_dispatch_write, _next_ts, _priced_cart_lines, _validate_and_derive_role)
  raises: RuntimeError, ThreadNotFoundError, UnknownActorError, UnknownMemberError,
          UnknownOrderTransitionError                         (five names, stable)
with start_workflow_run: closure 19, raises the five + WorkflowEngineDisabledError,
          WorkflowInputRejectedError, WorkflowRunNotFoundError   <- P14-3's concession disappears
Repository via router + via Storefront: raises {MemberIdCollisionError}  (one name)
Repository via Services (for scale, not proposed): 20 methods, raises nothing today
```

**Probe 5 — the `repo.<name>` leg cannot empty silently (brief claim 3).**

```
post-S10 router stub (shop.list_participants(), no `repo` binding):
  _router_repository_reach -> set()      -> `assert repo_reach` fires
  without that assert: _raises_of(repo, "Repository", set()) == set() == REPOSITORY_RAISES_TODAY -> passes
```

**Probe 6 — the factory reader's edge shapes at `HEAD` (P13-4).**

```
bare-return factory   raise self._boom(); def _boom(self): return       -> AssertionError (loud)
var-return factory    def _boom(self): e = HTTPException(410); return e -> {'e'}          (loud)
mixed-return factory  one bare `return` + one `return HTTPException(…)` -> {'HTTPException'}
                      (the value-less arm is dropped, but it cannot raise — same argument, accepted)
```

**Wire transcripts.** Mutation in place, one throw-away test appended to
`tests/test_storefront_api.py`, run under `-k`, both files then restored from the byte-copies:

```
P14-M1  POST /shop/api/session -> 410 '{"detail":"gone"}'
P14-M2  POST /shop/api/session -> 410 '{"detail":"gone"}'
```

**Hypotheses ruled out** (so Pass 15 does not re-walk them):

- *The class is closed — the fixpoint is a derivation, and a derivation cannot be too narrow.* **No.**
  A derivation is only as wide as the syntax it enumerates, and this one enumerates one statement
  node. P14-M3a is the eleven-character edit that proves it.
- *P14-1 is Pass 13's P13-2 not fixed.* **No.** P13-A itself dies at `HEAD` at every seed. What
  survives is the identical shape one call out — the same relationship P12-2 has to P11-2, and
  P13-2's N-M3 to N-M.
- *P14-M2 is latent like Pass 13's `repo` leg.* **No.** Pass 13's leg was latent because
  `list_participants`/`reset_all_participants` raise nothing. `Repository.ensure_participant` is on
  `POST /shop/api/session`'s path and raises `MemberIdCollisionError` today.
- *The `services`-binding control regressed when it became structural.* **No.** P14-M3b reddens on
  `test_the_routers_service_layer_reach_is_exactly_what_the_exemptions_assume`, and `_alias_prefixes`
  derives `services` off the delivered router (probe 1). Pass 12's N-CTRL rename case is subsumed.
- *S8e's reported figures might be one draw, like N-K.* **No.** P13-A is `1 failed` at all eight
  seeds, P13-3's A/B is a set-subset assertion, and my two survivals hold at seeds 4 and 7.
- *`SERVICE_RAISES_TODAY`'s exemption of `UnknownOrderTransitionError` is a list masquerading as a
  mechanism.* **No.** It is a `ServiceError` subclass, it is in `SERVICE_ERRORS_UNREACHABLE`, and the
  partition test covers it — both halves checked in the live class tree.
- *`92bf842` changed behaviour under cover of a test-and-comment commit.* **No.** The five
  must-be-unchanged files md5-match `HEAD`, the `storefront_api.py` diff is the comment block plus one
  reason string, and every production-side mutation I applied reproduced the pre-S8e behaviour
  descriptions exactly.

**What would convince me the class is closed.** Not a passing suite, and not a reproduction that
dies. Two things: (1) the guards' sentences state a **syntactic** scope — the node types walked, the
files read, the method sets closed — rather than a semantic one ("any path", "a route reaches"), so
that the smallest-edit question has a mechanical answer instead of a judgement; and (2) a probe of
the shape run here — enumerate every syntactic way to write the thing the sentence names, run the
delivered reader over each, and list the misses — comes back empty. Both probes in this pass took
one script each. Until a fix ships with its own version of that enumeration in the file, the prior
should stay that a thirteenth instance exists.

## Pass 15 — 2026-09-07 (S8f: does the *probe* reach as wide as the sentence it certifies?)

**Reviewed:** commit **`00827c2`** (S8f) — `falkorchat/storefront_api.py` (+103/−48, the comment
block at `:437–518` plus one reason string) and `tests/test_storefront_api.py` (+774/−94) — against
`docs/plans/salesperson-ui.md` **v1.25** §4.9, §5.1 rows S8/S9/S10, §5.2, §5.3; against `## Pass 14`'s
P14-1/2/3/4/5 and its stated convergence test; and against
`docs/plans/salesperson-ui-coordination.md`'s "Twelve instances…", "A compression of mine that was
wrong…", "Stakeholder decisions, 2026-09-07" and "S8f — the first unit to catch an instance of the
defect *itself*". Baseline for the diff: `92bf842` (S8e). Not reviewed: S9/S10/S12 content, the SPA,
the held `HISTORY.md`/`SERVER.md` documentation debt.

**I am a fresh reviewer.** I wrote none of Passes 12, 13 or 14; I read them as documents and
re-derived every claim of theirs I rely on below. This is the last gate on this artifact by
stakeholder decision.

**CPG: considered, not relevant — `cpg_falkorchat` models none of the files in scope (0 `File`
nodes matching `storefront`, measured by Pass 12, re-stated by the brief, and not re-spent here), so
every claim below is from direct read, from the delivered readers run **by import** rather than by
paraphrase, or from source mutation against the live two-file suite.**

**Verdict: needs changes** — **0 blockers, 3 majors, 1 minor, 1 nit**. Production behaviour is
untouched; all three majors are statement-and-test defects, which is why none is a blocker.

### The ruling the stakeholder acts on: the convergence test is **not passed**

Pass 14's test has two clauses. I judge them separately because they fail differently.

> **(1)** the sentences state a **syntactic** scope — node types walked, files read, method sets
> closed — rather than a semantic one.

**Partially met.** *Files read* and *method sets closed* are now genuinely syntactic and I verified
both (`storefront_api.py`/`storefront.py` whole; `Services` 9 seeds → 13 closed, `Repository` 6 seeds
→ 7 closed). *Node types walked* is syntactic **on the target side only**. The alias mechanism has
**two** axes — which node type binds the name, and which value expression counts as naming the
object — and the block states only the first. The second is stated as *"closed over the names bound
**to it**"* (`:453–454`), a semantic relation, over a mechanism that is
`ast.unparse(value) in prefixes`: exact source-text identity with a prefix already derived. Nothing
in the block says that.

> **(2)** the *enumerate-every-syntactic-form-and-run-the-reader* probe comes back empty.

**Not met.** The delivered probe is real and its target-axis half is excellent — `_ALIAS_FORM_SNIPPETS`
covers all eight walked node types, the classification is held against `ast`'s own enumeration
(27 = 8 + 19, verified), and it comes back empty. But **all eight snippets hold the value axis fixed
at the single spelling `self._services`.** Running the same-shaped probe over the value axis returns
**four misses on the delivered reader**, three of them confirmed on the delivered suite at
**185 passed** — including `me = self` / `me._services.start_workflow_run(...)`, written in
`ast.Assign`, the first entry in the walked list, and resolved by three of this guard's five legs but
not by the two that matter. Detail in **P15-1**; ledger in **Appendix P15-A**.

**So the probe comes back empty because it varies one of the mechanism's two axes.** That is the
failure mode the brief asked me to look for — *a mechanism that certifies a mechanism over-claims* —
and it is the fourteenth instance of this artifact's signature defect, in the same shape as the
previous thirteen: a rule stated wider than the reach beneath it.

**Per the stopping rule this chain stops here and escalates.** What that escalation needs is in
*"What kind of answer this needs instead of another pass"* below, and my answer is **not** a sixth
code cycle — for the first time in this chain the defect is closable **without touching the
reader**, and closing it with the reader would not converge.

### Major

#### P15-1 · **Major** · the alias mechanism has two axes and only one was enumerated — the fourteenth instance

`:454–455` promises *"a tenth added by either path, **written in any of the binding forms below**,
reddens."* Measured against the delivered reader and the delivered two-file suite (baseline
**185 passed**, my own run):

| shape, injected on the router-reached `Storefront.join` | reader | suite |
|---|---|---|
| `svc = self._services` (control) | seen | **1 failed, 184** |
| `svc: object = self._services` (P14-2's escape) | seen | **1 failed, 184** |
| **`me = self` ; `me._services.start_workflow_run(...)`** | **missed** | **185 (survives)** |
| `svc = self._services if x else self._services` (`IfExp` value) | **missed** | **185 (survives)** |
| `pair = [self._services]` ; `svc = pair[0]` (`Subscript` value) | **missed** | **185 (survives)** |
| `for svc in self._pool:` (non-literal iterable) | **missed** | **185 (survives)** |

All four are written in walked binding forms and **none is in the block's stop list** (`:469–477`),
which names call argument, return value, attribute store and `getattr`. The `me = self` half is the
sharper one: `_storefront_reach`'s two legs seed on the **receiver** (`shop`, `self`) and resolve it;
`_reached_methods` seeds on `self` and resolves it — and has a control asserting so
(`tests/test_storefront_api.py:4162`, *"via an alias of `self`"*). Only `_collaborator_reach`'s two
legs seed on `receiver.attr` and cannot. That is exactly the asymmetry `_reached_methods`' own
docstring warns against: *"a walk that resolves aliases on three legs and not the fourth is the
defect one field over."*

**Two different fixes, and they are not interchangeable.** (a) The receiver half is **~2 lines and
measured before being suggested** (Appendix P15-A, probe 5): derive each collaborator leg's seeds
from the receiver's own aliases — `{f"{r}._services" for r in _alias_prefixes(node, {"self"})}` — then
close as today. On the delivered files it returns the **identical nine** and the **identical
repository reach** (no re-baselining), catches `me = self` on both the `Storefront` leg and the
router leg (`sf = shop`), and leaves the documented attribute-store non-reach missed. (b) The value
half — `IfExp`, `Subscript`, non-literal iterable — is **alias analysis**, not another node type, and
must be written into the sentence instead: replace `:453–455` with the value rule the reader actually
implements (*"a name bound, in one of those eight forms, to an expression whose **source text is
exactly** a prefix already derived"*) and add those three shapes to the stop list at `:469–477`.
**(b) alone makes the block true;** see the escalation section for why I do not recommend (a).

#### P15-2 · **Major** · the delivered files state a plan fact the plan retracted, at 14 sites — including the block §5.1 is licensed to cite

`tests/test_storefront_api.py:3390` reads *"v1.22 decided it runs **on the turn worker** —
`shop.enqueue_turn(...)` in the router, `self._services.start_workflow_run(...)` in
`storefront.py`."* That is `teco`'s compression, which `teco` retracted in writing
(`docs/plans/salesperson-ui-coordination.md`, *"A compression of mine that was wrong"*) and which
v1.25 replaced: the plan specifies `trigger.maybe_trigger` → `services.start_workflow_run` **on the
worker**, call site `trigger.py:82`, outside all four scopes. `grep "S9's decided\|decided shape\|
placements S9 actually takes\|When S9 adds\|v1.22 decided"` returns **14 sites**, one of them in the
comment block itself (`storefront_api.py:466`). The load-bearing one is S8f's **newly written**
paragraph at `tests/test_storefront_api.py:3886–3892`: *"When S9 adds `start_workflow_run` to the
reach, this walk follows it with nobody re-pointing it."* Under v1.25 it will not — the walk reports
**none** of the three classes. That paragraph was written to fix P14-3 (a docstring making a claim
about the plan that the plan contradicts) and reproduces P14-3 one turn later.

**Mitigation, stated because it affects attribution, not the finding:** v1.25 landed at 08:49 and
`00827c2` at 09:10, so it was in the tree — but S8f's brief predates it and this is a race, not
carelessness. **Fix:** the §5.1 re-word unit corrects all 14 sites to cite v1.25's spelling, and
deletes the composition forecast at `:3886–3892` rather than re-wording it — the honest statement is
that the walk reports nothing at S9 and the evidence moves to the armed-fault measurement.

#### P15-3 · **Major** · *"none of it is excused here"* is contradicted by an entry in the dict it heads

`:498–503` states that the region outside the walk — *"`Services` into `Repository`, `Repository`
into redis, and a module-level helper in either collaborator file"* — is *"not walked, and **none of
it is excused here**: the graph faults are S8's typed handlers' own rows and the `ServiceError`
family is covered by the `SERVICE_ERROR_RESPONSES`/`SERVICE_ERRORS_UNREACHABLE` partition."*

`INHERITED_HANDLERS[WorkflowConfigError]` = *"guard evaluation, inside the executor — off the request
path"* is an excuse for a class raised **only in `guards.py`** (14 sites, verified), reachable only
through `Services → self._executor → executor → guards` — squarely the unwalked region. It is not a
graph fault, and it is **not** a `ServiceError` subclass (verified: none of the eleven inherited
handlers is). So the sentence is false in the dict it introduces, and the older headline at `:433`
— *"Every reason below is a checked claim, not prose"* — over-claims for that one entry. Nine of the
eleven **are** checked (the raise test's `Services`/`Repository` equalities plus the reach test);
`WorkflowConfigError` is prose, and `WebSocketRequestValidationError`'s is at best indirect —
`storefront_routes` iterates `entry.methods`, which a websocket route does not carry.

**Fix:** narrow `:500–503` to *"none of it is excused here **except `WorkflowConfigError`, whose
raise sites are `guards.py`'s and which no mechanism in this file checks**"*, and soften `:433` to
*"every reason below is a checked claim except that one."* Two sentences, no code.

### Minor

- **P15-4 · the census at `tests/test_storefront_api.py:3603` mixes two definitions.** *"68 annotated
  local assignments in this package, 16 in `storefront_api.py`"* — inherited from Pass 14. Counted
  **inside function bodies** the package total is **68** ✓ but `storefront_api.py` is **5**; counted
  **at all scopes** `storefront_api.py` is **16** ✓ but the package total is **215**. No single
  definition yields both. The substantive claim (annotated assignment is a house idiom) survives
  either way — 5 local in `storefront_api.py`, 3 in `storefront.py`, 14 in `services.py`. This is the
  brief's stale-figure question answered: the staleness is **not** confined to the "184" line, and
  this instance is in the repo rather than in a transcript. Fix: `(68 in this package, 14 of them in
  `services.py`)`, one definition.

### Nit

- **P15-5 · *"a binding form nobody classified fails rather than passing quietly"* (`:462–463`) is
  true only for a form the field-name heuristic keys on.** `binding_nodes` is derived by intersecting
  `_fields` with eight literal names (`target`, `targets`, `optional_vars`, `name`, `names`,
  `asname`, `arg`, `rest`). I verified the derivation is **complete for Python 3.12** — no ast class
  outside the 27 introduces a local name — so this is a claim about the future, not a live hole. But
  a future node spelled `_fields = ('var', 'value')` would open one silently. One clause: *"…so long
  as it carries one of the eight field names above."*

### Clause-by-clause ruling on the guard-reach statement (`storefront_api.py:437–519`)

Judged in the form a later unit will lift it. Lines are the delivered file's.

| # | Lines | Clause | Ruling |
|---|---|---|---|
| 1 | 437–444 | "stated … as the syntax it walks rather than as the behaviour it means to cover" | **Accurate as an intent, and delivered on the target axis; not on the value axis** (P15-1) |
| 2 | 448–452 | reach = (1) `<prefix>.<name>` in `build_storefront_router` + (2) the same in every `Storefront` method the router reaches through `shop.<method>`, closed over `self.<method>` to a fixpoint | **Accurate** — verified: 22 `Storefront` methods walked, 9 `Services` names |
| 3 | 452–454 | "`<prefix>` is the seed attribute — `shop._services`/`self._services` for the collaborator, `shop`/`self` for the frontier — closed over the names bound to it" | **Accurate literally**, and it is the narrow true reading. But "the names bound to it" is a **semantic** relation over a source-text-identity mechanism (P15-1b) |
| 4 | 454–455 | "**Nine today**…" | **Accurate** — the derived reader returns exactly `SERVICE_LAYER_REACH_TODAY` |
| 5 | 454–455 | "…**and a tenth added by either path, written in any of the binding forms below, reddens**" | **Inaccurate** — four measured counter-examples, three surviving the suite at 185 (P15-1) |
| 6 | 457–459 | "'Bound' is a closed list of eight `ast` node types … `Assign`, `AnnAssign`, `NamedExpr`, `For`, `AsyncFor`, `comprehension`, `withitem`, `MatchAs`" | **Accurate as the target-side list**, and genuinely closed. **Over-broad for `For`/`AsyncFor`/`comprehension`**, which are read only over a literal `Tuple`/`List`/`Set` |
| 7 | 459–463 | "Every *other* name-binding node … classified … held against `ast`'s own enumeration … so a binding form nobody classified fails rather than passing quietly" | **Accurate today** (27 = 8 + 19, derivation complete for 3.12); **future-tense clause needs one qualifier** (P15-5) |
| 8 | 463–467 | the P13-1/P14-2 history, incl. "S9's decided shape plus a type annotation" | **Mechanism accurate** (I reproduced 1 failed/184); **the plan fact is retracted** (P15-2) |
| 9 | 469–477 | "**Where it stops, as syntax:** … not followed into a call argument …, into a return value, onto an attribute …, or through `getattr`" | **Accurate for the four it names** (call-argument and attribute-store both survive at 185, measured) but **presented as the stop list and it is not** — it omits the receiver alias and the three value shapes (P15-1) |
| 10 | 475–477 | "That is a statement about the walk, not a claim about the code; the moment one of those is written, this comment is what has to change with it" | **Accurate and honestly framed** — nothing enforces it and it does not pretend otherwise |
| 11 | 479–485 | raise walk "over four scopes: `storefront_api.py` and `storefront.py` **whole** … plus, in each collaborator module, the methods the reach walk above measures **closed over the `self.<name>` calls those methods make**, to a fixpoint" | **Accurate** — verified `Services` 9→13 / `Repository` 6→7, and the closure catches P14-M1's and P14-M2's shapes |
| 12 | 485–490 | "the `Repository` seed is the union of the router's `repo.<name>` calls and the reached `Storefront` methods' `self._repo.<name>` calls, so S10 … re-points that leg instead of emptying it" | **Accurate** — the reach is exactly those six, two router-side and four `Storefront`-side |
| 13 | 490–496 | the P13-2/P14-1 history and what closing over `self.<name>` bought | **Accurate** — reproduced: `Services` 1→5 classes, `Repository` 0→1 |
| 14 | 498–500 | "**Where it stops, as syntax:** at any call that is not `self.<name>` on the walked class — `Services` into `Repository`, `Repository` into redis, and a module-level helper in either collaborator file" | **Accurate** — `_RAISE_ROUTES`' four STOP cases are the same list, run against the reader |
| 15 | 500–503 | "None of that is walked, and **none of it is excused here**…" | **Inaccurate** — `WorkflowConfigError` (P15-3) |
| 16 | 505–514 | the two non-family classes, their reasons' home, and the equality cross-check | **Accurate and killable** — I reproduced all three arms (Appendix P15-A, P15-M8) |
| 17 | 516–519 | "**This block is the statement's only home.** §5.1's S9 row is licensed to cite it by file and line" | **Accurate, and the right rule** — this is P14-5 closed properly |

**Lift-ready today: clauses 2, 4, 10, 11, 12, 13, 14, 16, 17** — nine of seventeen, and most of the
statement's useful content. **Must be narrowed before anything cites them: 5, 6, 9, 15.** **Must be
corrected against v1.25 first: 8**, and every other site P15-2 lists. Clause 3 is safe to lift only
if clause 5 is fixed, because clause 3 alone reads as complete.

### Ruling on `MemberIdCollisionError`: **classify, do not handle — I agree with S8f**, with two qualifications

I re-measured the wire answer independently (throwaway test on a byte-copy, restored,
`md5 b991fc5e…`): `POST /shop/api/session` with `ensure_participant` stubbed to raise →
**`500`, `text/plain; charset=utf-8`, `'Internal Server Error'`**. S8f's measurement reproduces exactly.

**The unreachability argument is airtight for the thing it claims, and the claim is one word too
wide.** `Storefront.join` is the sole caller (`_repository_reach` reaches `ensure_participant` from
nowhere else), `participant_id = self._id()` with `self._id = id_gen` defaulting to
`_default_participant_id()` = `'p-' + uuid4().hex`, and the one production caller is pinned **at the
construction seam** by `test_create_app_never_pins_the_participant_id_generator` (`tests/test_app.py:1111`)
— which asserts on the recorded kwargs, not on a grep, so it is a real pin. All three raise branches
need a pre-existing `Agent`, or a `User` without `tokenHash`, **at that exact freshly-minted uuid4**.
So no *client input* determines the id, and the reason string's *"**No request can reach it**"* is
sound. What it is one word wide about: an **operator** could — a hand-seeded node, or a
`FALKORCHAT_USER_ID` colliding with a minted id — and that is precisely the corruption the alarm
exists for. Suggested wording: *"no client input reaches the id, so no request can cause it; an
operator-created node at a minted uuid4 can, and that is what this alarm is for."*

**A bare `500` is the right answer here, on three grounds I checked rather than accepted.** (i) It is
consistent with the only precedent in the same table — `RuntimeError` (`services._dispatch_write`'s
invariant alarms) is classified identically and answers identically, and Pass 14 accepted that.
(ii) A typed handler would *mask* a namespace corruption, against `app.py:136-137`'s stated idiom of
typed handlers "without a blanket handler masking real bugs". (iii) The participant is not left
mute: §5.3 **C13** makes an unruled `(route, response)` render an explicit "unhandled response"
failure naming route and status, which is the plan's own designed answer for exactly this shape.
**One correction to the citation:** `app.py:357` is a **startup** posture (abort loudly rather than
silently shadow), not a request-path response posture. The comment block states this correctly
("This is `app.py`'s own posture **at startup**"); the commit message's shorter form does not.
Keep the block's wording.

**And the process call was right.** Refusing to write a `(route, response)` pair that would need a
plan row S8f was not authorised to write, and flagging it for disagreement instead of burying it, is
the correct move under this plan's own ownership rules.

### Disposition of Pass 14's findings

| # | Disposition | What I rechecked |
|---|---|---|
| P14-1 | **Fixed** | `_reached_methods` closes both legs to a fixpoint: `Services` 9→13 / 1→5 classes, `Repository` 6 seeds→7 / 0→1. `MemberIdCollisionError` is now in `REPOSITORY_RAISES_TODAY` and `NON_FAMILY_RAISES` |
| P14-2 | **Fixed on the axis it named; the defect class survives on the second axis** → **P15-1** | `svc: object = self._services` now **1 failed, 184**; four value-axis shapes still missed, three surviving at **185** |
| P14-3 | **Fixed in the file, re-created in the same paragraph against v1.25** → **P15-2** | The wrong exception names are gone; the replacement forecasts a composition v1.25 says will not occur |
| P14-4 | **Fixed, and killable on the `Services` leg** | Extending `SERVICE_RAISES_TODAY` (P14-M6's escape, 183-survivor on S8e) is now **1 failed, 184**; a blank reason is **1 failed, 184**; a written reason passes, which is the intended cost |
| P14-5 | **Fixed** | `:516–519` names the block as the statement's only home and licenses citation; no "cannot drift" claim remains |
| the thirteenth (self-caught) | **Fixed** | `_raises_of` hands the reached methods to `_raised_class_names` as one synthetic module; `raise self._mk(...)` resolves to `HTTPException` on the collaborator legs, and the delivered test pins it |

### What's solid

**S8f closed the target axis completely, and that is a real result.** The classification is derived
from `ast` rather than written down — I checked the derivation myself and it is **complete for Python
3.12**: no node class outside the 27 introduces a local name, and 8 + 19 = 27 exactly. That is the
right structural instinct, and it is the first check in this chain that ages correctly rather than
decaying.

**Every figure I could check reproduced**, and there are a lot of them: two-file **185** baseline,
exactly **2** new test functions (name-diffed against `92bf842`), 27/8/19, `Services` 9→13 and 1→5,
`Repository` 2→7 and 0→1, the P14-2 A/B at 1 failed/184, the P14-4 A/B on the `Services` leg, the
documented non-reaches surviving at 185, ruff clean on both files. The two exceptions are the "184
passed (survives)" `teco` already caught and P15-4's census.

**The self-catch is the strongest evidence in the chain that Pass 14's framing was right.** S8f's own
probe returned `MISSED ['_mk']` before any gate saw the commit, and it fixed it and re-ran. That is
the first instance in fourteen found by the producer rather than by a gate, and it cost one script.
The framing works; my finding is that it was applied to one of two axes, not that it is the wrong
framing.

**The MemberIdCollisionError handling is the model for how to disagree.** Measured, not argued;
classified, not swallowed; the plan-row constraint named as the reason for not handling; flagged for
disagreement rather than buried. I disagree with none of it.

### Honesty about the three excuses that go false at S9 — **partially, and the gap is a forecast, not an omission**

The brief flagged this as the one place a passing guard could still lie. My ruling:

- **The deferral is honest.** `SERVICE_LAYER_REACH_TODAY`'s comment says outright that *"which
  `INHERITED_HANDLERS` excuses that step falsifies is not stated here, and that is deliberate … the
  mapping is being settled in the plan."* That is the correct half of `teco`'s sentence-split, and
  S8f stayed inside it.
- **The forecast attached to the deferral is not.** The same comment continues *"after S9,
  `_raises_of` over the closed reach reports the classes"* — under v1.25 it reports **none of the
  three** — and `:3886–3892` says the walk will follow `start_workflow_run` into the reach. Those are
  P15-2, and they are the lie a reader would take away.
- **The absence of a re-worded reason string is correct, not a defect.** v1.25 assigns the
  replacement to S9 (*"each one's reason string is **replaced** by the reason that is true of it …
  and cited to the armed-fault test"*), so S8f writing it now would be a forecast too.
- **What is genuinely missing is one line.** Nothing at those three `INHERITED_HANDLERS` entries
  tells the next reader that their reasons have an expiry no test will catch. A single
  `# expires at S9 — see §5.1's S9 row; neither guard can see it` on each is available today, costs
  nothing, and is the only part of this that S8f could have shipped.

### What kind of answer this needs instead of another pass

**This is the escalation deliverable. It is deliberately not a patch list.**

**1. The class of mechanism has run out, and that is a finding, not a mood.** Every previous instance
was closed by *widening the reader along the target axis* — three spellings → two files → a fixpoint
over `ast.Assign` → eight node types — and S8f **finished** that axis: it is derived from `ast`,
complete for 3.12, and reddens rather than decaying. The remaining gap is the value axis: *which
expression counts as naming this object*. That is alias/points-to analysis. `me = self` needs a
receiver join, `svc = pair[0]` needs container contents, `svc = a if x else b` needs a branch join —
and each closure spawns the next. **A sixth cycle of the same kind does not converge**, which is
precisely the pre-agreed reading the stopping rule wrote down in advance.

**2. The right answer is docs-only, and it is the option this coordination has already adopted.**
Pass 14's option (b) — *narrow the sentence to what the walk does and freeze it* — is recorded in
`docs/plans/salesperson-ui-coordination.md` (U31) as **doctrine, not advice**. It applies here
verbatim, and for the first time in fourteen instances it closes the defect **without touching the
reader**: clauses 5, 6, 9 and 15 become true by being narrowed, and P15-2's 14 sites become true by
citing v1.25. No new test, no re-baselining, no mutation campaign, no gate.

**3. Why narrowing is now *sufficient* rather than a concession.** v1.25 already removed this guard's
load-bearing role: S9's done-condition is the **armed-fault measurement**, and the reach guard is a
**service-surface tripwire** whose stated job is to redden if S9 acquires a `Services` call through
`self._services` instead of the trigger. A tripwire does not need a complete rule — it needs a
narrow, true one, and it has one. Both direct spellings and both one-token variants redden
(measured). What survives are shapes nobody writes to enqueue a turn — and **zero** receiver-alias
bindings exist anywhere in `falkorchat/` (measured: 0 local bindings whose value is exactly `self` or
`shop`). So the residual operational risk of shipping the reader **exactly as delivered** is very
low. The defect is in the *sentence*, and the sentence is what §5.1 lifts.

**4. If the stakeholder wants the receiver half closed anyway, it is 2 lines and it is measured.**
Appendix P15-A probe 5: seed each collaborator leg from the receiver's own aliases. Identical nine,
identical repository reach, no re-baselining, closes `me = self` on both legs and `sf = shop` on the
router leg, leaves the documented attribute-store non-reach missed. **I do not recommend dispatching
it as a unit** — it buys a shape with zero precedent in the package, and the last five times this
chain dispatched a reader fix, the fix contained the next instance. If it is wanted, it belongs
**inside S9** as a one-line diff its own gate already covers, not as an S8g.

**5. The one thing that must not happen.** Do not let the §5.1 S9 re-word lift clauses 5, 6, 9 or 15,
and do not let it lift anything that cites `self._services.start_workflow_run(...)` as S9's decided
shape. Nine of seventeen clauses are lift-ready today; lifting those nine and narrowing the rest is
the whole remaining job, and it is one `architect`/`tico` edit.

### Open questions (need the stakeholder's call)

1. **Docs-only close (P15-1b + P15-2 + P15-3 + P15-4/5), or accept the block as delivered and record
   the four clauses as known-inaccurate in the review?** My recommendation: the docs-only close,
   folded into the §5.1 re-word unit that is already queued, so it is not a sixth cycle by another
   name.
2. **Does the receiver-alias 2-liner go into S9, or nowhere?** My recommendation: **into S9**, as a
   line in its diff, or nowhere. Not as its own unit.
3. **P15-2's 14 sites are in a `coder`-owned test file, and the correction is a plan-fact
   correction.** Who edits them — S9's implementer as part of its own touch on that file, or the
   re-word unit? This is a routing question I cannot decide.

### Appendix P15-A — ledger, probes and transcripts (Pass 15)

**Method.** Every mutation applied from byte-copies held at
`…/scratchpad/frozen/`, restored from those copies after each run, `md5sum` re-verified every time.
Command: `.venv/bin/python -m pytest tests/test_storefront_api.py tests/test_app.py -q`, working
directory `falkor-chat/server`, **run solo and serially**. **Baseline: 185 passed** (my own run).
The delivered readers were exercised **by importing `tests/test_storefront_api.py` as a module**, so
nothing below is a re-typed paraphrase of the mechanism. **No `git` tree-mutating command at any
point**; `git status --porcelain -- falkor-chat/` empty at the end, and all seven files md5-match
`HEAD`: `storefront.py` `a713e2c5…`, `services.py` `a952f4ad…`, `repository.py` `584ce30c…`,
`app.py` `e6bf735a…`, `storefront_api.py` `60aff74d…`, `test_app.py` `fcf877ac…`,
`test_storefront_api.py` `b991fc5e…`. `ruff check` on both touched files: **All checks passed**.

**I did not re-run the full suite.** The brief supplied 2617/14 as verified and asked me not to
re-derive it; every mutation here is two-file-scoped.

**Database.** `ws:test` only. `ws:acme` re-checked at the end and **unchanged — 871 nodes,
`Message` 52, `Entity` 544, `WorkflowRun` 21**. `reference` held **0 nodes when I started** — the
stray `timers-stale-key@v1` the brief describes was already gone — and holds 0 now.
`seed_workflows.sh` / `seed_salesperson.sh` never run.

| # | Mutation (injected on `Storefront.join`, dead branch, unless noted) | Result |
|---|---|---|
| **P15-M1** | `me = self` ; `me._services.start_workflow_run(None)` | **survived, 185** → P15-1 |
| P15-M1b | control: `svc = self._services` ; `svc.start_workflow_run(None)` | **1 failed, 184** — the reach guard |
| **P15-M2** | `svc = self._services if display_name else self._services` | **survived, 185** → P15-1 |
| **P15-M3** | `for svc in self._pool:` ; `svc.start_workflow_run(None)` | **survived, 185** → P15-1 |
| **P15-M4** | `pair = [self._services]` ; `svc = pair[0]` | **survived, 185** → P15-1 |
| P15-M5 | `svc: object = self._services` — **P14-2's escape**, 183-survivor on S8e | **1 failed, 184** — fixed |
| P15-M6 | `self._svc = self._services` — documented attribute-store non-reach | **survived, 185** (S8f reported 184 — the stale figure `teco` caught) |
| P15-M7 | `_go(self._services)` — documented call-argument non-reach | **survived, 185** |
| P15-M8a | `raise ValueError` in `services.save_profile` **+** `"ValueError"` in `SERVICE_RAISES_TODAY` — **P14-M6's escape**, 183-survivor on S8e | **1 failed, 184** — P14-4 fixed |
| P15-M8b | the same **+** `NON_FAMILY_RAISES["ValueError"] = "   "` | **1 failed, 184** — the blank-reason silence dies |
| P15-M8c | the same **+** a written reason | **185 passed** — correct by design: silencing now costs a written falsehood |
| P15-W1 | `ensure_participant` stubbed to raise `MemberIdCollisionError`, throwaway wire test | `POST /shop/api/session` → **`500`, `text/plain; charset=utf-8`, `'Internal Server Error'`** |

**Probe 1 — the derived binding-node list is complete for Python 3.12.** `27` classes carry one of
the eight field names; `8` walked + `19` excluded = `27`; the disjointness and equality assertions
both hold. I then listed every remaining `ast` class and its `_fields`: none introduces a local name
that is not a child of one of the 27 (`Name`/`Attribute`/`Subscript` in `Store` are the *sites*;
`Lambda`/`arguments` bind through `arg`, which is classified).

**Probe 2 — the value axis, against the delivered reader.** Router stub + one `Storefront` method,
expected `{"start_workflow_run"}`:

```
SEEN   self._services.start_workflow_run(ctx)            (control)
SEEN   svc = self._services  /  svc: object = ...  /  (svc := ...)  /  with ... as svc
SEEN   for svc in (self._services,)  /  [svc.start_… for svc in (self._services,)]
SEEN   match self._services: case svc  /  case object() as svc  /  case svc if ctx
SEEN   svc, _ = self._services, None   /  (svc, a), b = (self._services, 1), 2
MISSED me = self          ; me._services.start_workflow_run(ctx)      <- P15-M1
MISSED me: object = self  ; me._services.start_workflow_run(ctx)
MISSED (me := self)       ; me._services.start_workflow_run(ctx)
MISSED svc = self._services if ctx else self._services                <- P15-M2
MISSED for svc in self._service_pool                                  <- P15-M3
MISSED for svc in (s for s in (self._services,))
MISSED pair = [self._services] ; svc = pair[0]                        <- P15-M4
MISSED svc, *_ = self._services, None      (declined loudly, documented)
MISSED self._svc = ...  /  _go(self._services)   (documented stops)
```

**Probe 3 — census.** `0` local bindings anywhere in `falkorchat/` whose value is exactly `self` or
`shop`. Value-expression kinds for `Name`-target local bindings: `Call` 691, `Constant` 178,
`Subscript` 87, `IfExp` 53, `Attribute` 23, `Name` 21. Annotated local assignments — see P15-4.

**Probe 4 — the delivered readers over the delivered files.**

```
_storefront_reach            -> 22 methods
_service_layer_reach         ->  9  == SERVICE_LAYER_REACH_TODAY
_reached_methods(Services)   -> 13  (+_dispatch_write, _next_ts, _priced_cart_lines,
                                      _validate_and_derive_role)
_raises_of(Services)         -> RuntimeError, ThreadNotFoundError, UnknownActorError,
                                UnknownMemberError, UnknownOrderTransitionError   (five)
_repository_reach            ->  6  (ensure_participant, get_participant_record,
                                      list_participants, reset_all_participants,
                                      reset_participant, set_participant_record)
_reached_methods(Repository) ->  7  (+_graph)
_raises_of(Repository)       -> MemberIdCollisionError                             (one)
_raised_class_names(storefront.py) -> the seven StorefrontError subclasses
```

**Probe 5 — the receiver-seed fix, run before being suggested.** Replace each collaborator leg's
literal seed with `{f"{r}.{attr}" for r in _alias_prefixes(node, {receiver})}`, then close as today:

```
delivered reach == fixed reach on the delivered files : True   (nine, unchanged)
delivered repo reach == fixed repo reach             : True   (six, unchanged)
me = self / me: object = self                         : delivered [] -> fixed ['start_workflow_run']
router  sf = shop ; sf._services.start_workflow_run() : delivered [] -> fixed ['start_workflow_run']
self._svc = self._services (must stay missed)         : delivered [] -> fixed []
svc = self._services if ctx else ...  (value axis)    : delivered [] -> fixed []   (unclosed, by design)
```

**Probe 6 — the raise leg has the same value-axis gap, but its *shape* axis is closed.**
`_reached_methods` seeds on `self`, so `me = self ; me._h(ctx)` **is** seen (and the delivered test
pins it). `me = self if ctx else self` and `for me in self._pool` are missed — same class, and the
reason they matter less is that the shape that escapes the reach guard does **not** escape here.

**Hypotheses ruled out** (so nobody re-walks them):

- *The derived node list is incomplete, so the probe's "27 grammar nodes" is a floor.* **No.** Probe 1
  enumerates every `ast` class in 3.12 and none outside the 27 introduces a local name.
- *`_ALIAS_FORM_SNIPPETS` files a snippet under a node type it does not actually contain.* **No.** The
  test asserts `isinstance` per snippet before asserting the reach, and the negative control
  (binding removed, call kept) returns `set()`.
- *S8f's numbers were measured against a moving baseline generally.* **No.** Everything except the
  "184 passed (survives)" line and P15-4's census reproduced exactly, including the two-file 185, the
  2 new test functions, 27/8/19, and both collaborator method/class counts.
- *`Services` or `Repository` inherit methods the walk silently drops.* **No.** Both are plain classes
  (`class Services:`, `class Repository:`), so `_class_methods`' intersection loses nothing.
- *`P14-4`'s equality cross-check is unkillable in the wrong direction.* **No.** P15-M8a/b/c: the
  allowlist extension and the blank reason both die; a written reason passes, which is the intended
  and visible cost.
- *`MemberIdCollisionError` is reachable from some route other than `POST /shop/api/session`.*
  **No.** `_repository_reach` reaches `ensure_participant` from `Storefront.join` only, and `join` is
  that route's body.
- *`00827c2` changed behaviour under cover of a test-and-comment commit.* **No.** The five frozen
  production files md5-match `HEAD`, `storefront_api.py`'s diff is the comment block plus one reason
  string, and every production-side mutation I applied reproduced the documented behaviour exactly.

---

## Pass 16 — 2026-09-07 (the component documentation: `SERVER.md` §1.3/§1.4 and the S7→S8g `HISTORY.md` entries)

**Scope — this pass reviews the component documentation, not the guard-reach clauses.** Passes 1–15
judged the storefront's code and its self-describing guards; that clause-by-clause chain was closed
by a stakeholder stopping rule at Pass 15 and is **not reopened here**. No finding of Passes 10–15 is
re-adjudicated, and nothing below is a fifteenth cycle of it. What is under review is commit
**`9c51189`** — two documentation files, no code:

- `falkor-chat/docs/SERVER.md` **§1.3** (the storefront auth/tenancy block, `:139–186`) and **§1.4**
  (the eleven-route `/shop/api` block, `:297–358`), plus the two §1.3 corrections at `:99–100` and
  `:109–113`/`:123`;
- `falkor-chat/docs/HISTORY.md` — four new entries (S7, S7c, S8, S8b–S8g).

Judged against the code as delivered on `main`: `falkorchat/storefront_api.py`, `storefront.py`,
`app.py`, `repository.py`, `services.py`, `config.py`, `docs/QUERIES.md`, and the cited commits.
The commit's own claim that the **pre-existing** §1.3/§1.4 prose "was checked and left alone" is
treated as a claim under review (P16-4). Not reviewed: §1.1/§1.2/§1.5–§1.8, the SPA, any code.

**CPG: considered, not relevant — `cpg_falkorchat` is mid-rebuild by `graph-dba` and out of bounds
for this run per the brief; it also modelled no storefront file at Passes 12 and 15 (0 `File` nodes
matching `storefront`). Every claim below comes from direct read of the delivered files, from `git
show`/`git log` on the cited commits, or from small read-only AST/regex probes over the tree.**

**Not run:** the pytest suites (the stakeholder's standing constraint — a default run wipes the
freshly re-seeded `reference` graph). Every suite figure the entries record is therefore
**unverified here**; see *Open questions*.

**Verdict: needs changes** — **0 blockers, 5 majors, 3 minors, 4 nits.** No claim I found is
*unsafe*; all five majors are the same class the S8b–S8g entry itself names — **a stated rule
broader than the reach the mechanism implements** — now in prose rather than in a guard. Three of
them are in the new §1.3/§1.4 text, one is pre-existing §1.3 text the commit reports as checked, one
is in the S8b–S8g entry.

### Major

#### P16-1 · **Major** · §1.4's lead sentence claims per-credential `ctx` for all eleven routes; it holds for six, and its own `Cred` column contradicts it three lines later

`SERVER.md:299–301`: *"every one of them resolves `ctx` from the request's own credential, so the
`QUERIES.md` column below is reached under the **participant's** `actor`, never `config.USER_ID`."*
Measured route by route (full table in **Appendix P16-A**):

- Three routes carry **no credential at all** — the table's own `—` rows (`GET /health`,
  `POST /session`, `POST /presenter/session`). `POST /session` builds `ctx` from the id it has just
  *minted*, not from a request credential.
- **`GET /catalog` authenticates a participant and then reads under the demo `Agent`.**
  `Storefront.list_catalog` → `_catalog_rows` → `filter_products(self._catalog_ctx, …)`, and
  `_catalog_ctx` is `CallContext(ws, actor=self._agent_id)` (`storefront.py:694–705`). The §15.2
  query is `reference`-global besides — `Repository.filter_products` takes **no `ws`**
  (`repository.py:2724`). So that row's cited section is reached under neither the participant's
  actor nor `ws:{WS_ID}`, which `storefront.py`'s own docstring states deliberately and §1.4 denies.
- The two presenter rows call `repo.list_participants(shop.ws)` / `reset_all_participants` with **no
  `ctx`** (`storefront_api.py:1353`, `:1423`, `:1435`).

Only `GET /state`, `GET`/`POST /messages`, `POST /order/advance` and `POST /reset` match the
sentence. **Suggested replacement:** keep the true half — *no storefront route resolves through the
process-constant `get_context()`, and no route reaches `config.USER_ID`* — then state the three
shapes separately: participant-`ctx` routes (five), the catalog's deliberate `Agent` actor over
global `reference` data, and the presenter/no-graph routes that build no `ctx`.

#### P16-2 · **Major** · the `/shop` SPA mount is documented as unconditional; `create_app` mounts it only when `FALKORCHAT_STOREFRONT_DIR` names an existing directory — and its documented default is *unset*

`SERVER.md:123` lists what `FALKORCHAT_STOREFRONT_ENABLED=1` gives you as *"the `/shop/api` router,
its error map, **the `/shop` mount** and the startup preflight"*, and `:298` says the router is
*"mounted alongside a `StaticFiles` mount of the SPA build at `/shop`."* The code
(`app.py:436–443`) guards it:

```python
if served_dir is not None and Path(served_dir).is_dir():
    app.mount(SHOP_MOUNT, StaticFiles(directory=str(served_dir), html=True), name="shop")
```

`config.STOREFRONT_DIR` is `None` unless the operator sets it (`config.py:171`), and §1.3's own
`FALKORCHAT_STOREFRONT_DIR` row documents *unset* as the default — a combination §1.3 elsewhere
calls legitimate ("the text-only deployment"). So the documented default configuration produces a
running storefront **with no `/shop` at all**, and a mistyped path skips the mount silently (the
preflight logs `images=0` and continues; the manifest is deliberately not a precondition).
**Suggested fix:** in both places, condition the mount on `FALKORCHAT_STOREFRONT_DIR` naming an
existing directory, and add one clause to that env row saying an unset/nonexistent path skips the
`/shop` mount as well as emptying the manifest, while `/shop/api` still serves.

#### P16-3 · **Major** · the `403 wrong_credential_type` condition is stated broader than `get_presenter` implements — a malformed or wrong-scheme credential answers `401`

`SERVER.md:170–173`: *"**`403 wrong_credential_type`** when the request carried something that is not
the presenter principal (a participant token, typically), **`401 presenter_session_gone`** when it
carried no credential or a presenter token this process never minted."* The mechanism
(`storefront_api.py:917–930`) branches on `parse_bearer` **first**: `None` → `401
presenter_session_gone`; only a credential that *parses* into `<principal>.<token>` with
`principal != "presenter"` reaches the `403`. `parse_bearer` (`storefront.py:215–240`) returns
`None` for an absent header, a whitespace-only one, a non-`Bearer` scheme, a missing `.`, an empty id
half and an empty token half. So `Authorization: Basic abc`, `Bearer garbage` and `Bearer presenter`
— each of them "something that is not the presenter principal" — all answer **`401`**, not `403`.
A client or test written from this sentence asserts the wrong code. **Suggested fix:** *`403` when
the header parses as `<principal>.<token>` with a principal other than `presenter`; `401` for
everything else — no header, a header `parse_bearer` rejects, or a presenter token this process
never minted.* The same wording sits in `get_presenter`'s docstring and is worth the same edit.

#### P16-4 · **Major** · "existing content was checked and left alone" does not hold: four pre-existing §1.3 statements describe mechanisms that do not exist yet, one of them a script that is not in the tree

Each verified against the delivered tree, not inferred:

| `SERVER.md` | Claim | Reality |
|---|---|---|
| `:135` | "`start_demo.sh` pins the one variable to a dedicated value" | **The file does not exist** — `find` over the repo returns nothing, `falkor-chat/scripts/` has no such entry; it is plan step **S11** (`docs/plans/salesperson-ui.md:1070`, gated *after S5, S8*). `config.py:168` correctly writes it as future ("`scripts/start_demo.sh`, S11"). The paragraph's other half — `tests/test_storefront.py`'s `FALKORCHAT_DEMO_WS` tripwire — is real (`test_storefront.py:733`). |
| `:127` | "Size of the storefront's own bounded turn executor… Agent turns run there rather than on `BackgroundTasks`" | No executor exists. `turn_workers` is stored and exposed and read nowhere else; `storefront.py:313` says "**S9** adds a `ThreadPoolExecutor`". Today the value is inert. |
| `:130` | "The anyio thread limiter the storefront raises **inside `_lifespan`, before `yield`**" | `config.THREAD_LIMIT` has **no reader** in `falkorchat/`; no `current_default_thread_limiter()` call exists. Nothing raises it. |
| `:128` | "…after intake stops" | The stop-intake flag is **S10**; `presenter_reset_all`'s own comment says so (`storefront_api.py:1400–1406`). The drain is real; the stop-intake it is described as following is not. |

A document whose scope note says *"states the system as it is now"* (`SERVER.md:9`) cannot carry
four future mechanisms in the present tense — and these are in the env-var table, which is exactly
where an operator looks. **Suggested fix:** one step marker per row (`(S9 — not built; the value is
inert today)`, `(S11)`), which is the shape `config.py:168` already uses. The three code comments
`config.py:186–208` carry the same tense and are worth a sweep by their owner; that is a code change
and is out of this pass's scope.

#### P16-5 · **Major** · the S8b–S8g entry's blast-radius claim is false for one of the three files it names, and it is offered as a measurement

`HISTORY.md:14–17`: *"Two files only … with `storefront.py`, `app.py` and `tests/test_app.py`
byte-unchanged across the whole chain (md5-checked at every unit's close)."* `18b675a` (S8b) changes
`falkor-chat/server/tests/test_app.py` — `+22/−1`, adding the `ServiceError`-ownership assertion to
`test_the_default_deployment_is_untouched_by_the_storefront_parameters`. `git diff --stat 81a1268
b720bd3 -- tests/test_app.py` reports the same 22 insertions across the whole chain. The two
**production** files are byte-unchanged, and S8b's own commit message claims only that
("`falkorchat/storefront.py` and `falkorchat/app.py` are byte-unchanged") — the HISTORY entry widened
it. It is the sentence a later reader is least likely to re-check, because it advertises its own
evidence. **Suggested fix:** *"Three files — `storefront_api.py`, `tests/test_storefront_api.py`, and
`tests/test_app.py` once, in S8b (+22, the `ServiceError`-ownership assertion) — with
`falkorchat/storefront.py` and `falkorchat/app.py` byte-unchanged across the whole chain."*

### Minor

#### P16-6 · **Minor** · "No route takes an id from the client" is broader than the plan's sentence and than the mechanism

`SERVER.md:317`. The plan states it precisely (`docs/plans/salesperson-ui.md:386–388`): *"No
storefront route accepts a client-supplied `threadId`, `customerId`, `orderId` or `ws`."* The
compression drops the qualifier and is literally false: the **participant id is client-supplied**,
inside `Bearer <participantId>.<token>`, and becomes `ctx.actor` — which *is* the `customerId`
(`storefront.py:413–420`). What makes it safe is the `hmac.compare_digest` check against the graph
row for that id, not the absence of an id; the sentence's own three supporting clauses (thread,
order, workspace) never mention it. **Suggested fix:** restore the plan's four names, or write "no
route takes an id **as a parameter**", then keep the existing two-layer argument unchanged.

#### P16-7 · **Minor** · two rows of the `Reaches`/`QUERIES.md` columns under-cite what the route runs

- `GET /state` — `get_cart` is not one query. `Services._priced_cart_lines` (`services.py:2670`)
  reads `read_cart` (§16.5) **and then `lookup_products_by_id` against the global `reference`
  graph** (`QUERIES.md` §16.9). That is the only cross-graph read on the 2 s poll path and the row
  hides it; §16.9 is the section to add.
- `POST /messages` — `services.post_message` runs `thread_exists` and `resolve_member_kinds`
  (`QUERIES.md` §2) *before* the §4 write (`services.py:832–847`). That pre-write lookup is where
  the row's `401 invalid_token` / `503 demo_not_seeded` re-shapes come from
  (`SERVICE_ERROR_RESPONSES`), so citing §4 alone hides the only route in the table that can raise a
  `ServiceError`.

#### P16-8 · **Minor** · "There is no Pass 16" is now false at the file level

`HISTORY.md:110`. The sentence's argument is sound and specific — a re-review of the same 17
guard-reach clauses would be ceremony — but it is written as an unqualified statement about this
review document, and this section is a Pass 16 on a different subject. **Suggested fix:** *"No
sixteenth pass on the guard-reach clauses was opened, and none should be without a fresh stakeholder
decision"* — which keeps the ruling and survives later passes on other subjects.

### Nits

- **P16-9** · `SERVER.md:339–342`'s parenthetical reads as the complete declared-token set but omits
  **`unhandled`**, declared on the `500` rows of `POST /reset` and `POST /presenter/reset-all`
  (`storefront_api.py:1289`, `:1387`). `_cross_cutting_json`'s defensive branch can also emit a
  token no route declares (`state_unknown`, `:394`) — labelled unreachable by construction, so a
  parenthetical mention is enough.
- **P16-10** · `SERVER.md:173`: presenter tokens are not minted *"from
  `FALKORCHAT_STOREFRONT_PRESENTER_KEY`"* — they are `secrets.token_urlsafe(32)`
  (`storefront_api.py:837–841`), unrelated to the key; the key *gates* the mint. "in exchange for"
  rather than "from".
- **P16-11** · `SERVER.md:139–140`: *"the `/shop/api` surface never uses it [`get_context()`]"* is
  true per request, but the storefront's workspace comes from `provider().ws` once at construction
  (`app.py:324`) — i.e. from that same seam, resolved one time. One clause keeps it exact.
- **P16-12** · `SERVER.md:161–163` lists *"a `User` carrying no `tokenHash`"* as its own failure mode.
  The repository collapses it: `get_participant_record`'s `WHERE u.tokenHash IS NOT NULL`
  (`repository.py:3507`) returns zero rows, the same branch as an unknown id, so
  `resolve_token`'s `isinstance(stored_hash, str)` guard is unreachable through it. The prose is not
  wrong about the *outcome*; it implies a branch that does not exist.

### What's solid

Verified in full, not spot-checked — each of these is stated correctly and I could not break it:

- **The load-bearing invariant (§1.3) is exactly right, including on the error paths.**
  `Storefront.resolve_token` calls `self._repo.get_participant_record` on every call
  (`storefront.py:534`); `self._records` is *written* there (`_cache_put`/`_cache_drop`) and never
  read; the only reader is `lookup`, and `storefront_api.py` contains no `.lookup(` call site — the
  AST tripwire (`tests/test_storefront_api.py:2910`) carries its own non-vacuity test. I walked the
  reset, the F8 timeout and the `_reset_state_unknown` re-read: none of them lets a cached record
  answer an auth question. The "cache refresh is load-bearing" comment at `storefront.py:550–562` is
  accurate and worth keeping.
- **The single `401` is genuinely undifferentiated.** Every `None` from `resolve_token` becomes one
  `StorefrontHTTPError(401, "invalid_token", "no valid participant credential")` with no branch
  (`storefront_api.py:899–904`). The document does not claim "nine" — it enumerates five clauses
  covering eight causes over seven `return None` points, and that enumeration is complete against the
  code. One nuance worth keeping in mind (not a defect): the "every failure is one answer" rule is
  scoped to *credential* failures; a graph fault inside the same dependency reaches the cross-cutting
  handlers as `503`/`504`, which is correct and is §1.4's subject.
- **The presenter→participant direction of the isolation claim** (`presenter` parses as a participant
  id no `User` carries → ordinary `401`, structurally, not by a special case) — confirmed;
  `get_participant_record` is anchored on `userId` with the `tokenHash` guard, and participant ids
  are server-minted `p-<uuid4hex>`. Only the reverse direction is mis-stated (P16-3).
- **The preflight paragraph** — runs from `_lifespan` **before `yield`** (`app.py:362–366`), all
  three conditions and their fix commands as described, the manifest built there and deliberately not
  a condition (`storefront_api.py:1479–1534`), the `resolve_member_kinds` identity with the post path
  (`services.py:839`) exact.
- **`GET /messages` always passes `since`, and the classification argument is right** — no-`since`
  + `thread_id` takes `get_cursor` + `advance_cursor`, a write (`services.py:969–984`), which is why
  `ROUTE_CLASSES` is keyed on `(METHOD, path)`.
- **The `ServiceError`-wrapper paragraph is right about which mechanism does what** — the default app
  is byte-identical because the handler is *absent* (`app.py:427–431` registers it only
  `if shop is not None`; `app.py:278` refuses `storefront and dev_surface`), and the path check is a
  property of the handler. That distinction was P11-3's finding and it survived into the doc intact.
- **Both §1.3 corrections** (the `create_app` storefront form; `STOREFRONT_ENABLED` driving
  `storefront=True`) match `app.py:185–192`/`:278`/`:487–489`.
- **The eleven-route table's credentials and cited sections all check out** — every row's `P`/`K`/`—`
  matches the route's dependencies, the eleven paths match `ROUTE_CLASSES` exactly, and §18.1, §18.3,
  §17.1, §17.2, §16.5, §18.8, §18.9, §16.10, §18.4, §18.5, §15.2, §9.1 and §4 all resolve to the
  right `QUERIES.md` heading. The four-key presenter projection is real (`repository.py:3599–3614`
  projects six). P16-7 is an omission inside a correct table, not a wrong citation.
- **The error-rule paragraphs** — `ROUTE_CLASSES` keyed on `(METHOD, path)`, the live handler and the
  gate reading the one `cross_cutting_response` seam, and the gate reading declarations back off
  `app.routes` with all four refusals having real tests (`test_storefront_api.py:938`, `:962`,
  `:974`, `:2021`, `:4412`).
- **HISTORY's re-derived figures reproduce.** `INHERITED_HANDLERS` = 11; 17 registered handlers
  (11 + `ServiceError` + 2 envelope + 3 cross-cutting); `SERVICE_LAYER_REACH_TODAY` = 9;
  `_ALIAS_BINDING_NODES` = 8 and `_NON_ALIAS_BINDING_NODES` = 19; **zero** receiver-alias bindings in
  `falkorchat/` (re-measured by AST over all 28 modules: no `Assign`/`AnnAssign`/`NamedExpr` whose
  value is `self`); all 15 cited commits resolve, with dates and file scopes matching the entries
  (only P16-5's `test_app.py` claim fails); the follow-up's `timers-stale-key@v1` is really at
  `tests/test_workflow_timers.py:766` with `VERSION = "v1"`.
- **The one-entry decision for S8b–S8g is correct and should stand.** The entry's value is the shape
  — one defect, fourteen instances, each inside the artifact that closed the previous one — and six
  entries would bury it. It is also honest about the cost, which is rare enough in a change log to be
  worth protecting; the "so nobody later splits it into six" clause does that job.

### Open questions

1. **The suite figures are unverified.** 2582 → 2608 → 2615 → 2617, "14 deselected throughout", the
   two-file 176 → 183 → 185, and "`ws:acme` … 871 nodes" are recorded as measurements I was
   instructed not to re-run. They are *consistent* with Pass 15's independently recorded 185/184 and
   871-node figures, which is corroboration, not verification. If you want them re-derived, that is a
   `pytest` run plus a read-only node count — say the word and it routes to `qa-engineer` with the
   re-seed obligation attached.
2. **`start_demo.sh` is cited as delivered in two further places outside this commit's files** —
   `salesperson/README.md:73` and `salesperson/AGENTS.md:23`. Same false present tense, different
   component; flagging it rather than reviewing it, since it is outside this pass's scope.
3. **Where do the corrections land?** All twelve findings are documentation edits owned by the two
   files' author, except the `config.py:186–208` comment tense noted inside P16-4, which is a code
   change and would route to `coder`.

### Appendix P16-A — what each `/shop/api` route actually resolves (evidence for P16-1)

| Route | Credential | `ctx` built from | Graph reached |
|---|---|---|---|
| `GET /health` | none | — | none |
| `POST /session` | none | the **server-minted** participant id (`Storefront.join` → `context_for`) | `ws:{WS_ID}` |
| `GET /state` | participant | the credential | `ws:{WS_ID}` **+ global `reference`** (cart pricing, §16.9) |
| `GET /messages` | participant | the credential | `ws:{WS_ID}` |
| `POST /messages` | participant | the credential | `ws:{WS_ID}` |
| `GET /catalog` | participant | **not built from it** — `_catalog_ctx`, `actor = config.AGENT_ID` | **global `reference`** (`filter_products` takes no `ws`) |
| `POST /order/advance` | participant | the credential | `ws:{WS_ID}` |
| `POST /reset` | participant | the credential | `ws:{WS_ID}` |
| `POST /presenter/session` | none | — | none |
| `GET /presenter/participants` | presenter | **no `ctx`** — `repo.list_participants(shop.ws)` | `ws:{WS_ID}` |
| `POST /presenter/reset-all` | presenter | **no `ctx`** — `repo.list_participants` / `reset_all_participants` | `ws:{WS_ID}` |

`config.USER_ID` is reached by **no** route (it is touched once at startup, by
`services.ensure_actor(provider())` in `_lifespan`) — that half of §1.4's sentence is correct.

### Pass 16, second look — 2026-09-07 (re-gate of `c708423`)

**Reviewed:** `c708423` — `falkor-chat/docs/SERVER.md` (+118/−55 across §1.3/§1.4),
`falkor-chat/docs/HISTORY.md` (two sentences), and one docstring in
`falkorchat/storefront_api.py`. Every disposition below was re-derived against the tree at `HEAD`,
not read off the fix commit's message. Same constraints as the first look: no pytest, no seed
scripts, no `cpg_falkorchat`.

**Verdict: approve with suggestions** — **0 blockers, 0 majors, 2 minors, 1 nit.** All twelve
findings are genuinely fixed, several better than I proposed. Nothing below warrants another gate:
the three items are one-clause edits to take whenever §1.3/§1.4 is next opened, or to decline. **I
am not asking for a third pass.**

**The twelve, one line each.** All **fixed**; the two I would call improvements on my own suggestion
are marked.

| | Disposition |
|---|---|
| P16-1 | Fixed, and **better than proposed** — a four-bucket partition with the arithmetic shown (`5 + 1 + 1 + 4 = 11`) instead of a corrected sentence. Partition re-checked below. |
| P16-2 | Fixed at both sites; the `FALKORCHAT_STOREFRONT_DIR` row now carries the consequence ("skips the `/shop` mount entirely and silently", `Path(served_dir).is_dir()`, `images=0`), which is what an operator needs. |
| P16-3 | Fixed in the doc **and** in `get_presenter`'s docstring, stated as the branch (`parse_bearer` first) rather than as what the branch means, with the three concrete headers that answer `401`. |
| P16-4 | Fixed as markers, not deletions — judged separately below. |
| P16-5 | Fixed: "Three files … plus `test_app.py` **once**, in S8b (`+22/−1`)". Matches `git diff --stat 81a1268 b720bd3` exactly, and the md5 claim is now scoped ("from S8d onward") rather than dropped. |
| P16-6 | Fixed — the plan's own four names (`threadId`, `customerId`, `orderId`, `ws`) restored, with the credential-carries-the-participant-id caveat kept in parentheses rather than buried. |
| P16-7 | Fixed — `§16.5 **+ §16.9**` with "the only cross-graph read on the 2 s poll path" in the cell, and `§2, then §4` with the pre-write lookup named as the source of that route's re-shapes. |
| P16-8 | Fixed, and **better than proposed** — it names this pass by subject ("a later `## Pass 16` on a different subject … is not one"), so the ruling survives future passes without re-editing. |
| P16-9 | Fixed — `unhandled` added and attributed to the two reset routes' `500` rows; `state_unknown` now mentioned (one imprecision left, P16-15). |
| P16-10 | Fixed — "minted in-process … **in exchange for** … rather than derived from it", with `secrets.token_urlsafe(32)` named. |
| P16-11 | Fixed — "no `/shop/api` request resolves through it — the one thing the storefront takes from that seam is its *workspace*, read once at construction as `provider().ws` (`app.py:324`)". Verified. |
| P16-12 | Fixed — "The middle three are **one** branch, not three", with `WHERE u.tokenHash IS NOT NULL` named as the reason. Matches `repository.py:3507`. |

### Judgement 1 — P16-4's repair shape: right, and it works; one row it did not reach

**The shape is correct and I would not change it.** The test I applied is the one the brief names:
*would a reader skimming §1.3 for "how does this system work" come away with the right picture?*

- The markers are **not quiet**. `**Not built yet — S9**` leads the cell, which is the strongest
  position a table cell has; a skimmer reading nothing but bold text still gets it.
- Both marked rows say **what the value does today** — "nothing consumes it, so setting it changes
  nothing today", "read by nothing in `falkorchat/`". That is the sentence that actually protects a
  reader, and it is stronger than a bare step reference, which would leave "so is it live or not?"
  open. Both re-verified: `_turn_workers` has no consumer, and `config.THREAD_LIMIT` has no reader
  anywhere in `falkorchat/`.
- The `start_demo.sh` marker sits mid-paragraph rather than leading, but it carries the operator's
  actual workaround (pin `FALKORCHAT_WS_ID` by hand; the `"acme"` default is the populated dev
  workspace). That is the right trade for a paragraph rather than a table cell.

**What the pass did not achieve is the standing property, and one row shows where the line fell.**
The distinction now holds for the four sentences that were reported, not as a rule. Concretely —

#### P16-13 · **Minor** · `FALKORCHAT_STOREFRONT_QUIESCE_S` is as inert today as the two rows above it, and its row says the opposite

The row now reads: *"The wait is delivered; the **stop-intake** that is designed to precede it is not
— S10, so today a post landing mid-drain extends the wait instead of being refused."* The wait is
delivered code, but **nothing can populate the turn map**: `Storefront.set_turn_state` has no caller
anywhere in `falkorchat/` (S9 adds it, with the executor). Grep over all 28 modules — the only
readers are `turn_in_flight` at `storefront.py:865`, `storefront_api.py:1175` and `:1434`. So today:

- `_await_quiesce` and `presenter_reset_all`'s drain loop always pass on the **first** check, and
  `503 quiesce_timeout` cannot fire;
- `POST /shop/api/messages`'s `409 turn_in_progress` cannot fire;
- `GET /state`'s `turn` block is always `{"state":"idle","queuePosition":0}`;
- **setting `FALKORCHAT_STOREFRONT_QUIESCE_S` changes nothing observable** — exactly the property the
  two rows above it mark and this one does not.

**Suggested one-clause fix:** *"**The wait is delivered but cannot yet trigger — S9**: nothing
populates the turn map until the executor lands, so the drain passes on its first check and neither
`503 quiesce_timeout` nor `409 turn_in_progress` is reachable today. The **stop-intake** designed to
precede the wait is S10's."* **And the rule that makes it standing, if you want one:** *every §1.3
env row says what setting the value does today.* Three rows now do, one half does. That is a
convention to apply on the next touch, not a mechanism to build — a doc-level gate over eight rows
would cost more than the defect.

### Judgement 2 — the five new items: four hold, one is wrong

1. **The seam code block** — holds, and it was the right catch. `config.py:16–17` are
   `os.environ.get("FALKORCHAT_WS_ID", "acme")` / `("FALKORCHAT_USER_ID", "u1")`; `get_context` is
   `config.py:223`; `api.py:22` imports it as `_resolve_context` and `api.py:43` is the overridable
   wrapper routes depend on. The block now shows both, correctly labelled, and the
   "process-constant, not literal" sentence resolves the contradiction with the `start_demo.sh`
   paragraph.
2. **`RESERVED_CTX_KEYS`** — holds. `services.py:111` is
   `frozenset({"threadId", "error", TIMER_FIRED_CTX_KEY})`, `TIMER_FIRED_CTX_KEY = "timerFired"`
   (`:110`). Citing the constant rather than the members is the right repair.
3. **The `limit` counter-example — wrong route** (P16-14, below).
4. **The presenter screen** — holds. `salesperson/src/` is `App.tsx`/`main.tsx`/`App.css`/
   `index.css`/`assets`; no presenter view, no occurrence of "presenter" anywhere under `src/` or
   `index.html`. The plan's S12d is the presenter view, mounted into S12b's shell — so the
   `S12b/S12d` citation is right, not just plausible.
5. **"Three remaining routes" → four** — holds, and the general fix is sound. Partition checked
   below.

#### P16-14 · **Minor** · the `le=50` counter-example names a route that has no `limit` at all

`SERVER.md`: *"other routes on the same router set their own — `GET /threads/{tid}/participants` is
1–50 — so read the route rather than generalising from this one."* `GET
/threads/{thread_id}/participants` (`api.py:301–305`) takes **`thread_id` and `ctx` only — no
`limit` parameter**. The `Query(10, ge=1, le=50)` at `api.py:294` belongs to its neighbour, `GET
/threads/{thread_id}/workflow-runs` (`api.py:291`), 10 lines above. The two were added in the same
K-036 block under one comment, which is how they got swapped. The sentence's *point* survives — I
enumerated every `Query(` in `api.py` and that route is the **only** exception: lines 72, 87, 122,
229, 249, 257, 327, 445 are all `le=200`, line 131 is the `le=1000` thread window. **Suggested
fix:** name `GET /threads/{tid}/workflow-runs` instead, and it is safe to say it is the only one.

**The partition is correct** (`SERVER.md:342`, the load-bearing arithmetic). Re-derived from the
route bodies, disjoint and exhaustive: **5** build `ctx` from the credential — `GET /state`
(`shop.get_state(shop.context_for(...))`), `GET`/`POST /messages`, `POST /order/advance`
(`ctx = shop.context_for(...)`), `POST /reset` (inside `reset_participant`, `storefront.py:922`);
**1** from the id it just minted (`POST /session`, via `Storefront.join`); **1** under the demo
`Agent` (`GET /catalog`, `_catalog_ctx`); **4** build none — `GET /health` and `POST
/presenter/session` (no query at all) and the two presenter routes (`repo.list_participants` /
`reset_all_participants` on `shop.ws`). 5 + 1 + 1 + 4 = 11, and `ROUTE_CLASSES` has 11 entries.

#### P16-15 · **Nit** · `state_unknown`'s stated reachability names the branch the log line is *not* about

*"reachable only by a route absent from `ROUTE_CLASSES`"* covers one of the two ways
`_cross_cutting_json` reaches that branch: `answer is None` arises from the `KeyError` path
(unclassified route) **and** from `cross_cutting_response` returning `None` for a `no graph access`
route whose typed handler fired — which is the case the code's own log line describes ("*which §5.3
classifies as issuing no query at all*"). Both are unreachable by construction, so the conclusion is
right and only the reason is half. **Fix:** "…by an unclassified route, or by a typed handler firing
on a `no graph access` route".

### What I re-verified independently

- **The code touch is docstring-only, on my own evidence rather than on the re-run proof.** Parsing
  `storefront_api.py` at `9c51189` and at `HEAD` with every docstring replaced by a sentinel gives
  **equal `ast.dump`**, an identical docstring-owner set of **27**, and `git diff 9c51189..HEAD`
  shows `storefront_api.py` as the only changed file under `server/` (the test file byte-identical).
  All 23 changed raw lines sit inside `get_presenter`'s docstring, whose new text I checked against
  the branch it describes. **The method is sound and I would use it again** — stripped-AST equality
  plus an owner-set check is stronger than a prose-only claim, because it also catches a docstring
  silently added or deleted under cover of the strip.
- **A sample of the "fourteen holding" I could reach without the list** (I never saw it, so this is
  my own sweep of §1.4's unchanged mechanism claims, not a re-check of theirs): `MAX_DIFF_PREVIEW =
  200`, `MAX_STEPS`/`MAX_TRANSITIONS`/`MAX_CONFIG_LEN` all in `schemas.py`; `schemas.py`'s
  `MAX_TEXT_LEN = 8000` / `MAX_NAME_LEN = 200` / `MAX_MENTIONS = 50`; the workflow error map's
  404/409/400/400/503 against `app.py:126–139`; "no `latest` alias" against `api.py:346–349`;
  exactly **3** `response_model=` in `api.py`. All hold.

### The two deliberate non-fixes — my view

- **`config.py:186–208`'s comment tense: holding it is right.** It is code, it is outside the
  docstring-only licence, and a CPG snapshot of `server/` is in flight. One thing worth carrying into
  that unit's brief, because it is the reason and not just the task: **the doc's env table is
  transcribed from those comments**, so the derived artifact is now correct and its source is not —
  which is the configuration in which someone "re-syncs" the doc *backwards*. The unit is three
  comment edits and should not sit behind anything larger.
- **`salesperson/README.md` + `AGENTS.md`: follow-up, agreed — but not a low-priority one.** Those
  are that component's entry docs, so a reader there is likelier to *act* on the false present tense
  than a `SERVER.md` reader was: the natural next step after reading them is to run a script that
  does not exist. It is a two-line edit in each; worth doing whenever that component is next opened
  rather than queued behind S9.

### Open questions — closed

The suite figures I was barred from measuring were measured by `teco` (185 two-file, 2617/14, and
`ws:acme` at 871 nodes), which closes Pass 16's only open question. Nothing in this second look is
blocked on anything.

## Pass 17 — 2026-09-08 (S9a: the concurrency core — the turn off the request thread)

**Scope.** Commit **`e6fa20c`** only (nine files, +895/−44), diff-scoped, judged against
`docs/plans/salesperson-ui.md` **v1.26** §5.1's S9 row, §4.4 measures 1/1a/2/3, §5.2's `turn` block,
and `docs/plans/salesperson-ui-coordination.md`'s **S9a** scope row (the first of five units
splitting S9: concurrency core only). `HEAD` has moved past `e6fa20c`; everything below is read
from `git show e6fa20c` and from the files as that commit left them, not from the working tree.
**Not reviewed:** S9b–S9e content (cancellation, `turn.lastTurn`, the record-cache removal, the
three `INHERITED_HANDLERS` reason strings) — none of it is in this unit and its absence is not a
finding; the SPA; anything Passes 10–15 adjudicated.

**CPG: considered, not relevant — `cpg_falkorchat` is stamped from `b795f4c` and `storefront.py`
has since changed substantially (this commit alone adds ~200 lines to it), so it is stale for every
file in this diff and was not consulted for any claim below; the call-graph facts it could still
answer (`set_turn_state` had no production caller, `maybe_trigger`'s only caller was
`background._safe_run_workflow`) were re-established here by direct `grep` over the delivered tree
instead.**

**What I ran.** `tests/test_storefront.py` + `test_storefront_api.py` + `test_app.py` — **280
passed** at `e6fa20c`'s tree state. Five mutations of my own, each applied to `falkorchat/storefront.py`
and restored from a byte-copy held in the session scratchpad, `md5sum` re-verified as
`d032d3ed…c675a72` after every one (Appendix N §1). Two standalone read-only probes against a
`Storefront` built on a stub `Services`, no graph touched (Appendix N §2/§3). `ws:acme` untouched;
no seed script run.

**Verdict: needs changes** — 0 blockers, **2 majors**, 5 minors, 3 nits.

Both majors are worth stating plainly for what they are: **neither is the implementer failing the
row.** Every done-condition S9a owns is met, on evidence I re-derived rather than accepted (the
sweep below), and the mutation discipline is the best this chain has seen. P17-1 is a race the
row's done-condition does not sample and whose full fix costs one sentence in the row; P17-2 is a
semantic the plan never specified and the code had to pick. Both should be closed before S9b builds
on the same function, which is why this is *needs changes* rather than *approve with suggestions*.

### Findings

**P17-1 — major. Two concurrent posts from one participant do not merely start two runs; they
leave the turn map asserting `idle` while a turn is still running, which is the exact state
`_await_quiesce` exists to make impossible.** Reproduced (Appendix N §2). The `409` check is in the
route (`storefront_api.py:1194`) and the booking is in `enqueue_turn` (`storefront.py:743`), with
`services.post_message` — a FalkorDB round trip — between them, so the window is milliseconds
wide, not nanoseconds. Both posts book, and `self._turns[participant_id] = …` is a **single-slot
overwrite**: two turns, one entry. When the first worker's `finally: clear_turn` fires
(`storefront.py:800`) it deletes the entry belonging to the *second*, still-running turn. Observed:
`turn_in_flight("p-a") is False` and `_await_quiesce("p-a") is True` with a turn live on a worker —
so reset-mine proceeds to delete the thread underneath it, which `_await_quiesce`'s own docstring
(`storefront.py:998`, graph note §7.3) names as the failure the quiesce order exists to prevent.
Two independent fixes, and the cheap one needs no plan change: **(a)** carry a per-booking token on
the entry and have `_run_turn`'s `finally` clear only its own booking — the map can then never
under-report; **(b)** to close the window itself, reserve the slot atomically before
`services.post_message` and release it on a failed write. See *Ruling 1* for whether (b) is the
row's business.

**P17-2 — major. `queue_position = len(self._turns)` is a plausible integer under every
condition, and on the delivered default (`turn_workers=4`) it is wrong.** Reproduced (Appendix N
§3): with four turns *running* on four workers, the fifth arrival reports
`{"state": "queued", "queuePosition": 4}` while it is **first** in the waiting line. The count is
of all unfinished turns, running ones included, so it equals "how many are ahead of me" only at
`turn_workers=1` — which `enqueue_turn`'s docstring states honestly and which is the only setting
any test exercises, but is not the default and is not what §4.4 measure 1 promises the field for
("so the UI can show a **queue position** rather than an indefinite spinner"). Two further edges of
the same shape: the value is **never recomputed**, so a participant queued at 2 polls `2` until a
worker picks them up and then jumps to `thinking`/0 (no countdown for S12a to render); and
`presenter_reset_all`'s `clear_all_turns()` empties the map while workers are still running, so the
next arrival books `0` with N turns in flight. Suggested: define the field's contract in §5.2
(architect — one sentence: index among unfinished turns, or true waiting-line index) and then
compute it to match; `max(0, len(self._turns) - self._turn_workers)` is the waiting-line reading.
§5.2 currently specifies only the key's *presence* (`docs/plans/salesperson-ui.md:1137`), never its
meaning, so the code was not free to be right here.

**P17-3 — minor. A failed `executor.submit` leaves the booking behind, permanently.**
`enqueue_turn` books under the lock and then submits with no guard (`storefront.py:739–747`).
Reproduced (Appendix N §2, part 2): after `shutdown_turns()`, `submit` raises
`RuntimeError: cannot schedule new futures after shutdown` and the entry survives as
`TurnState(state='queued', queue_position=0)` — that participant is then `409`-refused forever and
every reset-mine of theirs answers `503 quiesce_timeout`, until the process restarts. The
reachable trigger is narrow (submit after shutdown; `RuntimeError: can't start new thread` under
resource exhaustion) but the blast radius is disproportionate to the guard. Suggested:
`try: return self._executor.submit(...)` / `except BaseException: self.clear_turn(participant_id); raise`.

**P17-4 — minor. The one diagnostic a dead turn produces is pinned by nothing, and deleting it
leaves the suite green.** Until S9c's `turn.lastTurn` lands, `_log.exception(...)`
(`storefront.py:794`) is the *only* evidence that a turn died — the plan says so itself in the S9
row ("the participant sees their message, no reply, and a composer that quietly re-enables"). I
replaced that call with `pass` and ran the three suites: **280 passed** (Appendix N §1, mutation D).
This mutant is not among the eleven the commit message lists, so it is a live gap rather than a
re-report. Suggested: extend `test_a_turn_whose_trigger_raises_is_isolated_and_still_clears_the_gate`
with `caplog` — assert the record is at `ERROR`, carries `exc_info`, and names both the
`participantId` and the `msgId`. There is no `caplog` anywhere in either storefront test file
today (`grep -rn caplog tests/test_storefront*.py` → no matches).

**P17-5 — minor. `shutdown(wait=True)` has no bound, and `app.py`'s comment states one that is too
small by a factor of the queue depth.** `app.py:409–413` says "a blocking join of up to the agent
timeout". `shutdown(wait=True)` with no `cancel_futures` drains the **whole** accepted queue, so the
bound is `ceil(queued / turn_workers) × 180 s` — at the plan's own ~50-participant scale and the
default 4 workers, ~37 minutes of uvicorn refusing to exit on SIGTERM. The *decision* to drain
rather than cancel is right and well argued (`storefront.py:801`) and a done-condition pins it; the
*claim about its cost* is the defect, and it is this chain's recurring shape — prose asserting a
reach the mechanism does not have. Suggested: correct the comment to the real bound, and consider
whether the drain should be time-boxed (an `architect` call, since bounding it edges toward the
cancel semantics the row rules out).

**P17-6 — minor. `FALKORCHAT_THREAD_LIMIT=0` now silently deadlocks every sync endpoint; before
this commit it was inert.** Verified against the pinned venv (anyio **4.14.1**):
`CapacityLimiter.total_tokens` rejects `-1` (`ValueError: total_tokens must be >= 0`) but **accepts
`0`**, which leaves zero tokens for the threadpool FastAPI offloads every sync route onto — an app
that starts cleanly and hangs on the first request, with nothing in the log. `config.py:222` parses
the value with a bare `int(os.environ.get(...))` and nothing validates it. Contrast
`STOREFRONT_TURN_WORKERS=0`, which is now *loudly* fatal (`ValueError: max_workers must be greater
than 0` out of `Storefront.__init__`) and is fine as-is. Suggested: `max(1, config.THREAD_LIMIT)`
at the assignment, or a bound at parse time in `config.py`; and a `SERVER.md` clause saying `0` is
not "off".

**P17-7 — minor. The `trigger is None` branch is documented in three places and asserted by
nothing.** `storefront.py:783–784` returns early when no workflow engine is wired; `Storefront.__init__`'s
docstring, `app.py:331–336` and `test_a_storefront_with_no_trigger_still_queues_and_clears_the_turn`
all describe it as "a turn that does nothing but still occupies its queue slot". Deleting the two
lines leaves **280 passed** (Appendix N §1, mutation E) — because without the guard
`None.maybe_trigger` raises `AttributeError`, which the isolation block catches, logs and clears,
satisfying every assertion that test makes (`future.exception() is None`, `turn_in_flight is False`).
The behavioural difference is an `ERROR` traceback per post in the no-engine deployment. Suggested:
have that test assert the trigger was *not* called and that nothing was logged — which is the claim.

**P17-8 — nit. §4.4 measure 3 ("no `_safe_embed`") is now asserted in three prose locations and
pinned by no test.** `storefront_api.py:1185`, the plan's S9 row and `SERVER.md` §1.4 all state it;
`grep -rn _safe_embed falkorchat/ tests/test_storefront_api.py` finds it in `api.py`/`mcp.py` and in
the storefront's *docstring* only. It is structurally true today (the router holds no `embed_worker`),
so the risk is drift, not defect. The module already carries AST tripwires; a one-line source
assertion in the same family would cost nothing.

**P17-9 — nit. `enqueue_turn`'s returned `Future` has no production retainer, which is a gap S9b
inherits.** The route discards it (`storefront_api.py:1203`) and `Storefront` keeps no handle, so
"cancel the queued turn in front of `_await_quiesce`" has nothing to cancel. S9b will need a
`participantId → Future` map — and it is the same map P17-1(a)'s booking token wants. Worth
briefing as one change rather than two.

**P17-10 — nit. `test_enqueue_books_the_turn_before_it_submits` discriminates, but by a thread race
it does not control.** I re-ran the implementer's own reordering mutation (submit before book) 10×
and got **10 failed / 10** (Appendix N §1, mutation A), so the fixed test is genuinely better than
the one it replaced and the 5-of-5 claim in the commit message holds at 10 of 10 here. It survives
because `Thread.start()` blocks until the worker has bootstrapped, giving the worker a head start
the caller's dict write cannot win — a real mechanism, but not one the test states or pins.
A deterministic spelling exists: wrap `shop._executor.submit` and assert the map already holds the
entry at the moment submit is entered, which *is* the ordering claim.

### Ruling 1 — the check-then-act `409`: severity, reachability, and whether the row is wrong

**Severity: major, and the reason is not the double run.** The implementer framed the exposure as
"two simultaneous posts both pass the check", i.e. two `WorkflowRun`s on one thread — which is what
§4.4 measure 1a names. That undersells it. The second, reproduced consequence is that the turn map
**cannot represent** two turns for one participant, so the first `finally` erases the second's
entry and `turn_in_flight`/`_await_quiesce` report `idle`/`True` under a live turn (P17-1). That is
a corrupted invariant, not a duplicated unit of work, and it reaches the reset path the whole
quiesce design exists to protect.

**Reachability: yes, and the plan already says so.** The window spans a graph write, so it is
milliseconds, not nanoseconds — a double-tap on send, or a `localStorage` credential open in two
tabs (§5.3's own cross-tab case), can hit it. More decisively, §4.4 measure 1a states the premise
outright: *"A client-side disabled send button is **not** sufficient on its own — §6.4's load
harness will not honour it, which is exactly how this defect would reach production."* §6.4's
harness is a guaranteed trigger if it posts concurrently per participant, and it is in this plan.

**Does the booking-under-lock narrow or widen the window?** Neither — it is orthogonal. The lock
guards the dict, not the check-then-act; the window is fixed by *where* the check is (route,
`storefront_api.py:1194`) versus where the book is (`enqueue_turn`, after `post_message`). What
S9a changes relative to the pre-S9a code is far larger and entirely in the right direction: before
this commit `set_turn_state` had no production caller, so `turn_in_flight` was `False` for
everyone and the `409` was **unreachable** — there was no single-flight at all, and no storefront
trigger to duplicate. S9a builds the enforcement; it just does not make it atomic.

**Is the row wrong? Under-specified, not wrong — and I do not think it costs you a decision to
fix.** The row spells the request thread's sequence as "the `409` single-flight check,
`services.post_message`, the turn-map bookkeeping and `executor.submit(...)`", and P17-1(b) moves
the bookkeeping ahead of the write. That is one clause. But **P17-1(a) — the per-booking token —
closes the corrupted-invariant half with no plan change at all, no route-shape change, and no
deviation from the sequence the row spells.** My recommendation: take (a) in S9b (which opens
`enqueue_turn` anyway and needs the same handle map, P17-9), and amend the row's sequence clause to
*reserve-then-write, release on a failed write* only if you want measure 1a's "enforced
server-side" to be literally true before §6.4's harness runs. The implementer's judgement not to
take (b) unilaterally was correct; its judgement that the exposure is bounded by the double run was
not.

### Ruling 2 — `STOREFRONT_QUIESCE_S`: is the correction bigger than two prose blocks?

**The prose rewrite is bigger than two blocks, and there is one code-behaviour item — but it is
S10's, already assigned, and nothing S9a delivered is wrong.**

*Prose.* `config.py:202–209` and `SERVER.md`'s `FALKORCHAT_STOREFRONT_QUIESCE_S` row are the two
blocks named, and both are now false in five separate clauses each (`set_turn_state` has no caller;
the map is never populated; both drains pass on their first check; `409 turn_in_progress` is
unreachable; "setting the value changes nothing observable"). Two further places carry the same
staleness and were not in the brief: `storefront.py:1004`'s `_await_quiesce` docstring **was**
updated by this commit and is correct, but `storefront_api.py:1440–1455`'s `presenter_reset_all`
comments describe the reset-all drain, which is now live too — check them. So: three or four
blocks, still prose.

*Code.* I found nothing S9a made *wrong*. Two things it made *real* that were previously
unreachable no-ops, both already owned elsewhere: **(i)** the reset-all sequence
`drain-loop → repo.reset_all_participants → clear_all_turns` (`storefront_api.py:1455–1493`) has an
intake window — a participant can post and book a fresh turn between the drain's last check and the
delete — which is exactly what §5.1's **S10** stop-intake flag is for, and which was invisible
while the map was always empty; **(ii)** `clear_all_turns()` now wipes entries whose workers are
still running, so a running turn's `finally: clear_turn` becomes a no-op and the position accounting
restarts from 0 (P17-2's third edge). Neither is a defect in S9a. **Conclusion for your follow-up
unit: brief it as prose-only, but tell it to check `presenter_reset_all`'s comments as well, and
note that S10 — not it — closes the intake window the honest version of that row will now describe.**

### The S9 row, clause by clause (S9a's share only)

| Row clause | Delivered | Evidence I re-derived |
|---|---|---|
| Bounded `ThreadPoolExecutor`, `max_workers` from config | ✅ | `storefront.py:401–403`; `turn_app` fixture drives `workers=1` and the 0/1/2 serialization depends on it |
| Keyed by `participantId` | ⚠️ | the **map** is keyed; the executor is a single FIFO queue and the map is single-slot — see P17-1 |
| `409 TurnInProgress` **before** the message write | ✅ (sequentially) | `test_two_posts_a_tenth_of_a_second_apart…`; `_counts` reads `(1 Message, 1 WorkflowRun)` from the graph. Concurrently: P17-1 |
| Queue-position accounting on `GET /shop/api/state` | ⚠️ | present and correct at `turn_workers=1`; P17-2 at the default |
| `enqueue_turn(ctx, participant, posted)` signature | ✅ | `storefront.py:694–699`, exact |
| Post path: `post_message` + enqueue, `run_ctx={"language": …}` | ✅ | `test_the_turn_worker_carries_the_participants_language_in_the_run_ctx` asserts all seven kwargs |
| **No** `_safe_embed` (measure 3) | ✅ unpinned | P17-8 |
| Trigger on the worker, never the request thread | ✅ | three independent assertions; `executor.threads[0].startswith(TURN_THREAD_PREFIX)` |
| Worker never resolves a `ParticipantRecord` | ✅ | spy on `repo.get_participant_record` **with a positive control** |
| anyio limiter inside `_lifespan`, before `yield` | ✅ | `app.py:368`; the test asserts `77`, neither anyio's 40 nor config's 100 |
| Graceful executor drain on shutdown | ✅ | `test_the_turn_executor_drains_on_shutdown`, asserted with no wait of its own; P17-5 is about its cost, not its correctness |
| S8c's `services.` reach stays **green** | ✅ **and not vacuous** | verified independently — see below |
| `run_ctx` carries only `language` | ✅ | asserted by equality, not membership |

**Nothing in S9a's scope is silently dropped.** The four clauses the row carries that are absent
here — cancellation, `turn.lastTurn`, the record-cache removal, the three reason strings — are
S9b–S9e by the coordination's own split, and `HISTORY.md`'s "Not in this unit" paragraph names all
four correctly.

### What's solid

- **The reach-guard claim is true, and I did not take it on report.** I injected
  `self._services.start_workflow_run(ctx)` into `_run_turn` and
  `test_the_routers_service_layer_reach_is_exactly_what_the_exemptions_assume` went red
  (Appendix N §1, mutation B). Worker code really is inside the guarded reach, so "green" here is a
  guard that works rather than one that stopped looking — which, after six passes of exactly the
  opposite finding, is the single most reassuring thing in this commit.
- **The isolation block is not decorative.** Removing the `except` entirely turns
  `test_a_turn_whose_trigger_raises_is_isolated_and_still_clears_the_gate` red and nothing else
  (mutation C) — tight, and the `finally` is separately pinned by two of the eleven mutations.
- **The stub-swap is argued, not silently taken.** The row asks for "a stub 2 s LLM"; the tests use
  a `threading.Event` gate and say why in a comment block — a sleep asserts in-flight-ness by
  hoping, a gate asserts it, and three 2 s sleeps would be 6 s on a 7 s suite. That is the right
  call and the right way to record it.
- **`_GatedExecutor` is stubbed at the correct seam.** One layer above the model, so
  `repository.start_run` still writes real `WorkflowRun` nodes and the measure-1a assertion can be a
  graph count rather than a mock call count.
- **The self-report on the two surviving mutants is accurate.** Both survivors are described
  correctly, and the booking test's replacement genuinely discriminates (10/10, P17-10). Honest
  reporting of a survived mutant is what let me spend my mutation budget on the five it did *not*
  run rather than re-checking the eleven it did.
- **`config.py` and `SERVER.md` were carried in the same change**, which is the U36 coupling
  `teco` identified — and the one row left stale was left stale *on instruction* and flagged
  in `HISTORY.md` rather than quietly.

### Open questions

1. **P17-2's contract is yours or the architect's, not the implementer's.** §5.2 specifies
   `queuePosition`'s presence and never its meaning. Someone has to say whether it is an index
   among unfinished turns (today's behaviour, correct only at `turn_workers=1`) or a waiting-line
   position (what §4.4 measure 1 promises the UI). S12a renders whichever it is.
2. **Does §6.4's load harness post concurrently per participant?** If yes, P17-1 is a
   *before-the-harness* fix rather than a before-production one, and the reserve-then-write clause
   should go into the row now. If the harness serialises per participant, P17-1(a) alone is enough
   for a while.

### Appendix N — Pass 17's measurements

**N §1 — mutations run against `falkorchat/storefront.py` at `e6fa20c`'s tree state.** Byte-copy
held at `<scratchpad>/storefront.orig.py`; `md5sum` re-verified as `d032d3ed3ae64f3fcc0ec4a88c675a72`
after each restore, and `git status --porcelain falkor-chat/` empty at the end. Suite =
`tests/test_storefront.py tests/test_storefront_api.py tests/test_app.py`, `-p no:randomly`.

| # | Mutation | Result |
|---|---|---|
| A | `enqueue_turn`: submit **then** book (the implementer's own reordering) | `test_enqueue_books_the_turn_before_it_submits` **failed 10/10 runs** |
| B | inject `self._services.start_workflow_run(ctx)` into `_run_turn` | `…service_layer_reach_is_exactly_what_the_exemptions_assume` **failed**; the raises guard passed |
| C | delete the whole `except Exception` arm (no isolation) | 1 failed, 226 passed — only `…is_isolated_and_still_clears_the_gate` |
| D | replace `_log.exception(...)` with `pass` (silent swallow) | **280 passed — survivor** (P17-4) |
| E | delete the `if self._trigger is None: return` guard | **280 passed — survivor** (P17-7) |

**N §2 — the duplicate-booking probe** (read-only, stub `Services`, no graph). Two `enqueue_turn`
calls for `p-a` with the first turn held on a worker, then the first released:

```
after double-book, map entry: TurnState(state='thinking', queue_position=0)
second turn running; turn_in_flight('p-a') = False
           _await_quiesce returns          = True
```

and, for P17-3, after `shutdown_turns()`:

```
submit raised: cannot schedule new futures after shutdown
leaked entry: TurnState(state='queued', queue_position=0)
turn_in_flight('p-b') = True   quiesce = False
```

**N §3 — `queue_position` at the delivered default** (`turn_workers=4`, four turns *running*, a
fifth arriving):

```
p5 reports -> {'state': 'queued', 'queuePosition': 4}   (true waiting-line index: 0)
p1 reports -> {'state': 'thinking', 'queuePosition': 0}
```

## Pass 18 — 2026-09-08 (v1.27: the reserve/release protocol and the derived queue position)

**Scope.** Commit **`d1eaa7f`** only — `docs/plans/salesperson-ui.md` v1.26 → v1.27, +58/−8, one
file. This is a **plan** gate, not a code gate: no implementation of the amendment exists yet, and
`e6fa20c` is not re-adjudicated (Pass 17 stands; P17-1, P17-2, P17-3 and P17-9 are what this
amendment answers). Read as a word-level diff of the S9 row against `d1eaa7f^` plus the seven other
hunks, and judged against §4.4, §5.2, §5.3 C6a/C6b, §6.1, §6.4, S13 and the pinned interpreter.
**Not reviewed:** everything Passes 10–17 closed; P17-4…P17-8 and P17-10, which are code findings
this amendment correctly does not touch.

**CPG: considered, not relevant — the artifact under review is a markdown plan document, which no
code-property graph models; `cpg_falkorchat` is additionally stale for `storefront.py` and is being
written to by a `graph-dba` unit, and was not queried.**

**What I ran.** `/usr/lib/python3.12/concurrent/futures/thread.py` on the pinned interpreter
(`.venv/bin/python` → **3.12.3**) read in full for the deadlock claim, plus two executable probes:
the alleged three-party deadlock staged exactly as the row describes it, and a two-thread
`TestClient` concurrency probe for the new linchpin done-condition (Appendix O). No graph touched,
no seed script, nothing written outside this document.

**Verdict: needs changes** — 0 blockers, **2 majors**, 4 minors, 2 nits.

**The amendment's substance is right and I would not send it back for redesign.** The reservation
closes the admission window, the token closes the ownership window, the two are correctly called
independent, the derived position is a better answer than the one I suggested, and the sweep is
complete. Both majors are **one sentence each** — a false justification attached to a rule that
should survive, and an invariant the row relies on but never states. This should be a fast round
trip.

### Findings

**P18-1 — major. The deadlock prohibition's cited mechanism does not exist on the pinned
interpreter. The rule should survive; its justification must not.** The row states: *"`submit`
acquires `concurrent.futures.thread`'s `_global_shutdown_lock`, which `_python_exit` holds while
joining worker threads that are themselves blocking on the turn lock — a three-party deadlock at
interpreter exit, read from the pinned venv (CPython 3.12.3 `concurrent/futures/thread.py`)."`
`_python_exit` does **not** hold that lock while joining. The pinned source (`thread.py:24–31`) is
four statements: `with _global_shutdown_lock: _shutdown = True`, then — **outside** that `with` —
`q.put(None)` and `t.join()` for each worker. The same is true of `shutdown()` (`:217–240`): the
`t.join()` loop is outside `self._shutdown_lock`. Neither of the two locks `submit` takes
(`:164–165`) is ever held by anyone across a worker join, so the cycle cannot form. I then staged
the arrangement — turn lock held across `submit`, a worker blocked on that same lock, the
interpreter exiting underneath both — and it **exits cleanly in 1.23 s**, with the submit-under-lock
returning in 0.6 ms (Appendix O §2). Suggested: keep the prohibition, replace the reason with the
two that are true — (i) there is nothing to buy, since booking order need not equal submit order
(the row already says this, and it is sufficient on its own); (ii) `_python_exit` **does** join
every `ThreadPoolExecutor` worker, outside any lock, so a worker blocked on the turn lock delays
process exit for as long as the lock is held — which is an argument for holding the turn lock
*briefly*, not for a deadlock. Optionally note that putting an application lock underneath two
`concurrent.futures` internals is a lock-ordering hazard whose current benignity is an
implementation detail rather than a contract.

**P18-2 — major. The arrival ordinal now does two jobs and the row never states the invariant
either one needs.** It is the ordering key ("the count of entries that are still `queued` and were
accepted **earlier** than this one") *and* the ownership token ("a worker only ever writes the slot
it still owns"), which the commit message celebrates as "one field, two defects". Both jobs require
a **process-global, strictly monotonic counter that is never reset and never reused**, and the row
says only "the booking's arrival ordinal". That is precisely the wrong-rather-than-absent shape this
amendment exists to eliminate: `len(self._turns)` is a plausible "arrival ordinal", it is what the
same implementer shipped for the same-shaped number six days ago (P17-2), and it **collides** —
after `clear_all_turns()` it restarts, and two live bookings can share a value. A colliding ordinal
gives wrong positions *and* lets a worker's ownership check pass against a booking that is not its
own, which re-opens P17-1 in a new spelling. Suggested: one clause in the S9 row — *the ordinal is a
process-global counter incremented under the turn lock, never reset and never reused
(`itertools.count()`); two bookings never share one, including across `clear_all_turns()`* — and a
done-condition that a booking cleared by a foreign worker reddens.

**P18-3 — minor. The ownership condition is stated for worker writes only; the release path has the
same exposure.** The row's prose scopes it to the worker ("the `thinking` flip and the `finally`
clear alike"), while the interface cell's `Storefront.release_turn(participant_id, booking)` takes
the booking and so *can* be conditional. Between a reservation and a failed `post_message`, a
`clear_all_turns()` plus a second-tab post can install a different booking in that slot, and an
unconditional release then deletes it — the identical defect the token exists to prevent, on the
one path the prose does not cover. Suggested: state the condition once over **every** map write
(reserve, release, `thinking`, `finally`) rather than over worker writes.

**P18-4 — minor. §5.2's definition and the row's own concession disagree.** §5.2 says the number
means "how many *other* accepted turns must **start** before theirs"; the row concedes that "two
turns booked microseconds apart may then reach workers in the other order". The derivation counts
earlier-*booked* queued turns, so under a reorder the number is not what §5.2 promises — a small
divergence (bounded by one place, and only inside the booking window), but the plan is the artifact
and its two sentences contradict each other. This is the chain's signature defect in miniature: a
stated rule slightly wider than the mechanism. Suggested: define it as *how many other accepted
turns are ahead of theirs in the line* and let the row's reorder concession stand, or state the
booking-order tie-break as the definition.

**P18-5 — minor. The one stated bound understates itself.** §5.2's *One bound* paragraph says the
post-`clear_all_turns()` window produces "an under-count, never an over-count, and it self-corrects".
It also produces an under-report of `turn_in_flight` — a running worker's booking has been wiped, so
`_await_quiesce` can return `True` under a live turn. That is P17-1's invariant, reachable through
reset-all rather than through a double post. It is **accepted rather than defective** (reset-all has
just deleted the graph those turns would write into; §7.3's ordering makes their writes silent
no-ops) and S10's stop-intake narrows the intake half — but the paragraph currently reads as though
only a display number is at stake. Suggested: say both consequences and why the second is accepted.

**P18-6 — minor. The redefinition falsifies two prose blocks S9a shipped six days ago, and nothing
in the sweep can reach them.** `config.py`'s `STOREFRONT_TURN_WORKERS` comment and `SERVER.md`
§1.3's matching row both say *"setting it changes … the `turn.queuePosition` … since a position is
how many accepted turns were unfinished when this one arrived"* — which is the **old** definition,
verbatim, and is false under v1.27 (`turn_workers` deliberately does not appear in the derivation,
and running turns are excluded). This is the U36 coupling shape exactly: a document true when
written and false the moment the next unit lands. It is outside a plan sweep's reach by
construction. Suggested: brief the implementing unit to carry both, in the same change — the same
rule that made S9a a `config.py` unit.

**P18-7 — nit. The linchpin test's mechanism is unnamed, and it is worth naming because it is the
one test the whole correction rests on.** "The concurrent pair held inside the write" needs two
threads driving one `TestClient` with the first blocked inside the handler, and the suite has no
precedent for it. I verified it works: a second request is served and returns `200` while the first
is still inside its handler (Appendix O §1). One clause naming the shape saves the implementer
discovering a portal-concurrency question at build time.

**P18-8 — nit. §9's stale version token, confirmed and cosmetic.** `docs/plans/salesperson-ui.md:2016`
still reads "(v1.19) — **21 steps**". The version token is stale; the count is not the issue (no
step rows moved in v1.27). Agreed with the architect that this is a follow-up, and it is
cosmetic — recorded so it is not re-discovered.

### The four questions, answered

**1. The deadlock prohibition — wrong as justified, right as a rule.** See P18-1. Verified two ways
against the pinned 3.12.3, not against general knowledge: source read (`_python_exit`'s joins are
outside the `with`) and execution (the staged arrangement exits in 1.23 s). The instinct to verify
this one rather than assert it was correct; the verification landed on the wrong two lines.

**2. Does reserve/release close P17-1, both halves? Yes, and the release is more load-bearing than
the row claims.** *Reserve*: `reserve_turn` as a test-and-set under the turn lock, with `None` **as**
the `409` raised before the write, closes the admission window completely — there is no longer a
read on one thread and a write on another with a FalkorDB round trip between them. *Token*: making
the `thinking` flip conditional as well as the `finally` is the half that matters and the architect
got the subtle case — a booking wiped by `clear_all_turns()` while still `queued` would otherwise
flip a *later* booking's slot to `thinking` when its work item finally ran. With both, the
single-slot overwrite I reproduced cannot occur; the residual is P18-2's collision and P18-3's
release path. *Completeness of "released on any path that never reaches a worker"*: the row states
it generically and names two instances, which is the right structure — I could not construct a third
path (the request thread is not cancellable mid-handler, nothing cancels futures until S9b, and
`shutdown(wait=True)` runs everything it accepted). *The `504` consequence*: **confirmed, and
stronger than stated.** §5.3's rule (`:1353–1355`) is *message present and `turn.state === 'idle'`
⇒ the turn was lost* / *`!== 'idle'` ⇒ wait, as normal*. A reservation that survived a failed write
does not make that rule undecidable — it makes it decide **wrongly**, sending the client into "wait,
as normal" forever. Note what that implies: the release is not an improvement the reservation
enables, it is a repair for a property the reservation would otherwise **break** (today, with no
reservation, C6b already decides correctly). That is why it must land in the same change, and the
row is right to bind them.

**3. `queuePosition` — implementable, and the rejection of my suggestion was correct.** Counting
earlier-booked entries that are still `queued` is strictly better than my `len(self._turns) -
turn_workers`, which assumed running turns fill workers evenly and breaks whenever the map holds a
mix of `queued` and `thinking`. Deriving on read is the right call and cannot go stale by
construction; the cost is one scan of a ≤50-entry map per 2 s poll. The arithmetic in both new
done-conditions checks out (`thinking`/0, `queued`/0, `queued`/1, the third falling to 0; and
`queued`/**0** for a fifth arrival behind four running turns at `turn_workers=4`). Excluding
`thinking` turns is what makes `turn_workers` genuinely unnecessary on the wire — that argument is
sound. **The stated bound is not the only one**: P18-4 (booking order ≠ start order) and P18-5 (the
same window under-reports `turn_in_flight`, not only the position) are two more, and P18-2 is a
third if the ordinal is implemented as anything resettable. None is disqualifying; all three are
sentences.

**4. The falsified done-condition and the concurrent-inside-the-write test — the claim holds.**
Replacing `0/1/2` was required, not optional: under the new definition the old expectation is
arithmetically false (three participants at `turn_workers=1` give 0/0/1, not 0/1/2), so leaving it
would have been an unmeetable done-condition. And the discrimination claim is right on both halves.
A sleep-timed pair passes under check-then-act **and** under reservation, because the first booking
exists by the time the second post is issued either way. A pair held inside the write separates
them exactly: under check-then-act the second request reads an empty map (the booking is behind the
write) and is answered `200`; under reservation it is answered `409` with no `Message`. The row even
names the failing observation (`the check-then-act answers that second post 200`), which is what
makes it a test rather than an assertion. Mechanism verified as implementable (P18-7, Appendix O §1).

**On §6.4 — I agree with the architect, and I over-read measure 1a in Pass 17.** §6.4 already binds
the harness: *"It must honour the `409 TurnInProgress` contract rather than firing blind — a harness
that ignores it is the only client that would ever hit §4.4 measure 1a's defect"* (`:1932–1934`).
Measure 1a's "§6.4's load harness will not honour it" refers to a **disabled send button**, not to
the `409` response; Pass 17 read it as making the harness a concurrent-post trigger, and that was an
over-read on my part. Declining to change §6.4 is correct. The amendment therefore rests on the
double-tap / second-tab case plus P17-3's leak plus the `504` decidability — three independent
motivations, none of which needs a race to be likely, which is a stronger footing than the one my
own finding offered.

### Sweep — complete, checked rather than assumed

I grepped every occurrence of `queuePosition`, `queue position`, `TurnInProgress`, `turn map`,
`set_turn_state`, `turn_in_flight` and `enqueue_turn` in the amended file and accounted for each.
§4.4 measure 1 (`:521`), measure 1a (`:533`), §5.1's S9 row and S13 row (`:1076`), §5.2's `/state`
shape row (`:1145`) and the new *The queue position* section (`:1184`), §5.3 C6b's `504`
reconciliation (`:1343`) and §6.1's test list (`:1834`) all moved. The sites that did **not** move
are correct not to have: C6a (`:1383`) and the response tables (`:1677`, `:1800`) key on
`turn.state`, which is unchanged; the `lastTurn` paragraph (`:1222`) names `queuePosition` only as a
sibling field; §6.4 is P18's *On §6.4* above. `set_turn_state`'s `queue_position=` parameter appears
only inside the S9 row, so removing it strands nothing in the plan. **The one thing a plan sweep
could not reach is P18-6** — two blocks in `config.py` and `SERVER.md`, outside the file.

### What's solid

- **"The two halves are independent and neither subsumes the other."** That sentence is the review's
  own conclusion stated better than the review stated it, and it is the thing an implementer is most
  likely to get wrong (taking the reservation and calling the token redundant).
- **Making the `thinking` flip conditional, not just the `finally`.** Pass 17 reproduced the
  `finally` case only; the flip case is subtler, real, and was found by the architect rather than
  copied from the finding.
- **Deriving rather than storing, with the reason named as the hazard class.** "A stored number
  being the wrong-rather-than-absent hazard itself" is the correct generalisation, and it is applied
  rather than merely quoted — the field it deletes (`set_turn_state`'s `queue_position=`) goes with
  the number.
- **Both P17 findings were answered inside the architect's authority and the judgment holds.**
  Measure 1a already said "enforced server-side" and argued it as correctness; making the mechanism
  satisfy an invariant the plan already asserts is a correction, not new scope. I would have
  escalated and I would have been wrong to.
- **`0` on a `queued` turn is called out as load-bearing and S13's done-condition was swept to
  render it as *first in line*.** That is the exact place a derived-from-zero contract gets
  mis-rendered as "no queue", caught before the client existed.
- **The falsified done-condition was replaced rather than quietly dropped**, and the replacement
  names the observation that fails.

### Open questions

None blocking. P18-2 and P18-3 are the two the implementer must not be left to infer; the rest can
travel as suggestions.

### Appendix O — Pass 18's measurements

**O §1 — two threads on one `TestClient`, first held inside the handler** (the linchpin
done-condition's mechanism; FastAPI + `starlette.testclient`, pinned venv):

```
b returned while a is still inside handler: 200 (b alive? False )
order at that instant: [('enter', 'a'), ('enter', 'b'), ('leave', 'b')]
final: {'b': 200, 'a': 200} [('enter','a'), ('enter','b'), ('leave','b'), ('leave','a')]
```

**O §2 — the alleged three-party deadlock, staged as the row describes it** (turn lock held across
`executor.submit`, a worker blocked on that same turn lock, the interpreter exiting underneath both;
`timeout 25` wrapper, CPython 3.12.3):

```
submit-under-lock returned in 0.0006s
main returning -> threading._shutdown() runs _python_exit now
exit=0
wall=1.226928486s
```

The 1.23 s is the probe's own 1.0 s hold plus interpreter start-up. No hang. The source it is read
from, `/usr/lib/python3.12/concurrent/futures/thread.py:24–31`:

```python
def _python_exit():
    global _shutdown
    with _global_shutdown_lock:
        _shutdown = True
    items = list(_threads_queues.items())
    for t, q in items:
        q.put(None)
    for t, q in items:
        t.join()
```
