# Embedding-migration `pin` implementation — Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M6+)

## Scope & verdict

Reviewed U6 of `docs/plans/embedding-migration-coordination.md`: `coder`'s implementation of
`docs/plans/embedding-migration.md` §5 step 1 (FR-1/FR-2's `pin` mechanism), three new files —
`scripts/embedding_migration.py` (`pin()` + CLI), `scripts/pin_workspace_embedding_model.sh`,
`server/tests/test_embedding_migration.py` (10 tests) — against the plan's exact spec (§2.3, §3.2,
§5 step 1, §6 tests 1-4), the live `repository.py`/`modelconfig.py` source it calls, and the
`migrate` (U7)/`create_workspace.sh` (U8) follow-up units it must not foreclose. `migrate` itself
is explicitly out of scope (a separate, queued unit).

Baseline verified live, not by paraphrase: read `Repository.write_model_overrides`/
`read_model_overrides` (`server/falkorchat/repository.py:3237-3360`) and
`ModelGateway.resolve()`/`Resolution.primary`/`ModelConfigError`
(`server/falkorchat/modelconfig.py:121, 202-221, 625-655, 729-755`) directly; ran
`test_embedding_migration.py` alone (10 passed) and the full suite minus `test_services.py`
(2585 passed, 14 deselected, clean — matches the ledger's independently-reported counts exactly).

**Verdict: approve.**

**CPG:** considered, not relevant — `cpg_falkorchat` exists (built `07c252d`) but every claim below
was verified against live source directly (`repository.py`, `modelconfig.py`, `conftest.py`,
`tests/data/models.json`) and by running the actual suite; for three files this size, a direct read
settled every call-shape/kwarg-name question faster than a graph query would have.

## Findings

None at blocker, major, or minor severity. Two nits below.

**Nit — `sys.path.insert` for a direct scripts-import in a test has no prior precedent in this
suite.** `test_embedding_migration.py:31` (`sys.path.insert(0, str(_REPO_ROOT / "scripts"))` then
`import embedding_migration`) is a new pattern — every other script-under-test in this repo
(`test_seed_workflows_script.py`) shells out via `subprocess` instead. The docstring names the
reason (a spy/fake repo needs direct call-count assertions, not achievable through a subprocess),
which is a real and sufficient justification, and the insert doesn't collide with anything today
(`scripts/*.py` has no other module named `embedding_migration`, confirmed via `ls scripts/*.py`)
or destabilize the rest of the suite (full run above is clean). Worth a one-line note for whoever
adds U7's `migrate` tests to the same file, since the `sys.path` mutation is process-global and
unscoped — not worth a fixture/teardown for a single insert, but a second test file doing the same
insert for a different `scripts/` module would want to reuse this one rather than duplicate it.

**Nit — wrapper script's per-workspace-id loop lives in Python, not bash, diverging from
`backfill_thread_ids.sh`'s per-id bash loop with its own per-id progress banner.**
`pin_workspace_embedding_model.sh:67` passes the whole argument list through in one `exec`;
`embedding_migration.main()` loops over `args.workspace` and calls `pin(ws)` once per id. The plan
explicitly leaves this as "the implementer's call, either is a small, reviewable detail" (§5 step
1), so this isn't a deviation to flag as unauthorized — noting it only because a future reader
comparing the two scripts side-by-side might otherwise wonder why the shapes differ.

## What's solid

- **Every call shape into `repository.py`/`modelconfig.py` verified correct, kwarg-by-kwarg and
  key-by-key.** `write_model_overrides(ws, agent=, guard=, embedding=, responder=, at=, by=)`'s
  return shape (`{agent, guard, embedding, responder}`) and `read_model_overrides`'s
  (`{agentModel, guardModel, embeddingModel, responderModel}`) are both consumed with the exact
  right keys; `gateway.resolve("embedding").primary.ref` matches `Resolution.primary`'s
  `ResolvedModel.ref` and the plan's explicit "no `ws=`/`requested=`" precondition for reading the
  *global* default (not the workspace override).
- **The read-before-write discipline (§2.3's landmine) is followed correctly and is now
  well-tested against a swap in any of the three untouched kinds**, not just `agent`.
  `test_pin_preserves_guard_and_responder_overrides_in_their_own_slots` sets three distinct non-
  `None` values (agent/guard/responder) and asserts each survives in its own slot alongside a
  fourth, distinct `embeddingModel` value — this shape also incidentally re-verifies the return-
  dict's own key mapping (`written["guard"] → "guardModel"`, etc.), since a mislabeled pairing
  anywhere in either the write-call or the return-dict construction would produce a value mismatch
  against one of four distinct, always-present values. No further silent-corruption path found
  along this call chain.
- **Idempotency is tested at the right altitude — spy count, not just end state.**
  `test_pin_already_set_is_a_true_noop_write_never_issued` and the second `pin()` call in
  `test_pin_two_workspaces_each_keep_the_default_at_their_own_birth` both assert
  `write_calls == []` / `len(write_calls) == 1`, exactly the "no write issued at all" bar the brief
  asked for, not an inference from re-reading the same value back.
- **The `ModelConfigError`-degrades-to-WARNING path is verified not to touch the repo at all**,
  via an `_ExplodingRepo` that raises `AssertionError` on any call — `test_pin_gateway_construction_
  failure_never_touches_the_repo` passing *is* the proof, not a separate assertion.
  `at=int(time.time() * 1000)` also matches this codebase's existing `now_ms()`-shaped convention
  used identically in `ingestion.py`/`background.py`/`executor.py`/`storefront.py`/`services.py`.
- **Scope discipline held exactly**: `git status` shows only the three named new files; the
  `argparse` subparsers list contains only `pin`; `_default_repo()` is a two-line, unopinionated
  factoring (`Repository(db.connect())`) that neither forecloses nor complicates U7's `migrate`
  subcommand reusing it.
- **The wrapper script's one deliberate divergence from `seed_eval_corpus.sh`'s preflight**
  (missing `FALKORCHAT_OPENCODE_CONFIG` is a WARNING here, not a fatal `ERROR`) is explicitly
  documented in the script's own header comment and traces directly to §3.2 Option B's named
  mitigation — not an unexplained drift from the named precedent.

## Open questions

None — nothing here needs the caller's input; ready to proceed to U7/U8.
