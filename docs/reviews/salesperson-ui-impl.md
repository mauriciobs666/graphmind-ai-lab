# The one salesperson UI — Implementation Review (S1–S4)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M<n> TBD) · **Reviews:** `docs/plans/salesperson-ui.md` §5.1 rows S1, S2, S3, S4

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
