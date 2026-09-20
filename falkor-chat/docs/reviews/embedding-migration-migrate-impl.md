# Embedding model migration — `migrate` (U7) and FR-2 enforcement (U8) — Implementation Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M6+)

## Scope & verdict

Reviewed together, one pass, per the brief: **U7** (`migrate` subcommand + helpers +
`scripts/migrate_embeddings.sh` + 26 tests in `server/tests/test_embedding_migration.py`) and
**U8** (`scripts/create_workspace.sh` + three call-site swaps + in-process `seed_eval_corpus.py`
call + five doc-only clauses + `AGENTS.md` row + 4 tests in `server/tests/test_create_workspace_script.py`),
against `docs/plans/embedding-migration.md` §3.2/§3.3/§3.4/§4/§5/§6 and
`docs/plans/embedding-migration-graph.md`'s confirmed Cypher shapes. `pin()`
(`scripts/embedding_migration.py`) was re-diffed against U6's committed version
(`a07e6805`), not re-reviewed.

**Verdict: approve, both units.**

**CPG:** considered, not relevant — `cpg_falkorchat` (built `07c252d`) is available, but every
claim below was verified against live source, a live test run, and one independent mutation
probe, none of which the CPG would have made faster for a change this size and this well-annotated.

## Findings

### U7 (`migrate`)

No blockers, no majors.

**Minor — multi-batch (keyset-page-boundary) execution is never actually exercised by any test.**
Every `migrate` test uses the default `batch_size=50` against ≤5 rows, so `_reembed_label`'s
`while True` loop never runs a second `_read_unmigrated_batch` call in any test — the keyset
cursor advance (`last_id = batch[-1][0]`, `embedding_migration.py:243`) is exercised zero times
across pages. I verified this is not a live correctness risk: I mutated the line to
`last_id = batch[0][0]` (a plausible off-by-one) and reran the full 26-test file — all 26 still
passed, confirming the gap — then wrote a probe test forcing two pages
(`batch_size=2` over 5 rows) and reran it against the same mutation; it *also* passed, because
`coalesce(embeddingModel,'') <> $target` (not `$lastId`) is the actual correctness gate, exactly
as §3.3 documents — a wrong cursor only causes redundant re-scanning, never wrong output. So this
is a coverage gap with no correctness consequence today, but it does mean a *different* batching
regression — e.g. `--batch-size` silently being ignored and a hardcoded page size used instead —
would also go undetected, since no test proves changing `batch_size` changes anything observable.
Suggested improvement: add one test that seeds ≥5 rows, calls `migrate(..., batch_size=2)`, and
asserts either the final state (already covered) *or*, more precisely, spies on
`_read_unmigrated_batch`'s call count/params to confirm more than one page was actually read —
otherwise the assertion can't distinguish "batching worked" from "one page happened to cover
everything." (Reverted my probe/mutation; suite confirmed clean and back to 26/26 before writing
this review — no working-tree changes from this verification.)

**Nit — CLI path can raise an uncaught `MigrationAbortedError`/traceback instead of a clean exit
code.** `main()`'s `migrate` branch (`embedding_migration.py:434-440`) doesn't catch
`MigrationAbortedError`; a non-interactive run with no `--i-have-stopped-traffic` and no attached
terminal will crash with a Python traceback rather than a one-line error + exit 1. Not a
correctness issue (the graph is still never touched — confirmed by test 6), just a rougher
operator experience than the rest of the script's error messages suggest. Worth a
`try/except MigrationAbortedError as exc: print(f"ERROR: {exc}", file=sys.stderr); return 1`
wrap in `main()` if this is ever operator-facing beyond a shared dev box, not blocking.

**What I independently confirmed, matching the brief's checklist exactly:**
- `pin()` is byte-identical to U6's committed version — `git diff a07e6805 -- scripts/embedding_migration.py`
  shows zero removed lines inside the function body (only additive code below it and a docstring
  update above it).
- All four Cypher helpers match `embedding-migration-graph.md` verbatim: `_read_unmigrated_batch`'s
  `>`-only predicate and keyset `ORDER BY ... LIMIT`, `_write_embedding`'s `properties_set > 0`
  no-op test, `_count_unmigrated`'s two-statement label-scan shape, `_rebuild_vector_index`'s
  `read_index_dimension(...) is not None` DROP guard.
- `resolve()`/`embedder()` call shapes in `migrate()` (`modelconfig.py:729-793`) pass no `ws=`/
  `overrides=`, matching §3.3's hard-cap bypass exactly — confirmed against the real signatures,
  not the plan's paraphrase.
- The step-0 traffic-stop precondition (`embedding_migration.py:285-291`) is the first statement
  in `migrate()`, before `_default_repo()`/`ModelGateway.from_env()` are ever called — confirmed
  by reading the function body, not just trusting test 6's assertion.
- Full suite (`--ignore=tests/test_services.py`): **2605 passed**; `test_embedding_migration.py`
  alone: **26 passed**; `test_create_workspace_script.py` alone: **4 passed** — reran independently,
  not taken from the ledger.
- The vanished-row skip-and-log path (`_VanishingRowGraph`, test 13) is a genuine engine no-op
  (a real `DETACH DELETE` mid-flight), not a faked result object — read the fixture in full.

### U8 (FR-2 enforcement)

No blockers, no majors.

**Minor (judgment, not a defect) — `create_workspace.sh`'s throwaway-workspace test id is a fixed
literal, not randomized.** `test_create_workspace_script.py` uses `WS = "k072-create-workspace-guard"`
rather than a uuid/pid-suffixed id. In isolation this looks like new surface for the
already-flagged live-instance-contention risk, but it is consistent with an existing accepted
convention in this suite (`test_vector_index_churn_guard.py`'s `PROBE_WS = "vector_churn_guard"`,
also a fixed literal) — not a regression this unit introduced, and both fixtures drop-before/
drop-after the same way. Not worth changing on its own; if the team ever moves to parallel test
execution (`pytest-xdist`) across this file and `test_vector_index_churn_guard.py` at once, both
would need the same fix together, not just this one.

**What I independently confirmed, matching the brief's checklist exactly:**
- All three call-site swaps (`start_server.sh:148`, `start_demo.sh:166`, `start_agent_team.sh:205`)
  forward `EMBEDDING_DIM`/`FALKORDB_HOST`/`FALKORDB_PORT` unchanged — `git diff HEAD` shows a
  single-token swap (`bootstrap_schema.sh` → `create_workspace.sh`) on each line, nothing else
  touched.
- `create_workspace.sh` never converts `pin()`'s non-fatal `ModelConfigError` WARNING into a hard
  failure: no error handling wraps the pin loop (`create_workspace.sh:57-59`), and
  `embedding_migration.main()`'s `pin` branch returns `0` unconditionally regardless of `pin()`'s
  return value (`None` on the WARNING path) — traced both sides, not just one.
- `seed_eval_corpus.py`'s `gateway` reuse: `gateway = modelconfig.ModelGateway.from_env()` at
  line 579 is the same object passed to `embedding_migration.pin(EVAL_WS, gateway=gateway)` at
  line ~606 — no second resolve, confirmed by reading the intervening code, not just the two line
  numbers in isolation.
- All five `seed_*.sh` doc clauses plus the `AGENTS.md` row are comment-only —
  `git diff HEAD -- scripts/seed_*.sh AGENTS.md` shows the diff hunks touch only `#`-prefixed
  lines and one new table row, zero executable lines changed.
- `start_server.sh:113`/`:200` sets `FALKORCHAT_OPENCODE_CONFIG` to its default path
  *unconditionally*, regardless of `FALKORCHAT_ENABLE_AGENT` — so the `ENABLE_AGENT=0`/no-config
  regression scenario the plan worries about is real and reachable through `create_workspace.sh`,
  and I confirmed (via `test_pin_missing_opencode_config_warns_and_does_not_raise`, plus reading
  `pin_workspace_embedding_model.sh`'s own non-fatal preflight) that it resolves to the WARNING
  path, not a crash, end to end.
- Point 6b (pre-existing `bootstrap_schema.sh` partial-DDL-failure gap): confirmed genuinely
  pre-existing and out of this chain's scope. `bootstrap_schema.sh` uses `set -euo pipefail`, so a
  genuine script-level failure (not a "Redis says already-exists" no-op, which exits 0 by design)
  propagates a nonzero exit; `create_workspace.sh` has no `||true`/trap swallowing that around its
  `bootstrap_schema.sh "$@"` call, so `set -e` there aborts the whole chain before the pin loop
  ever runs. The residual risk the ledger named — a *partial but non-erroring* DDL sequence (e.g.
  the process is killed mid-script) leaving a materialized-but-incomplete graph key that `pin()`
  would then "succeed" over — already existed identically for an operator running
  `bootstrap_schema.sh` then `pin_workspace_embedding_model.sh` by hand before this unit existed;
  `create_workspace.sh` changes neither the exposure nor the mitigation. Correctly left as a
  future backlog item, not this unit's to fix.

## Interaction check (U7 × U8, both depending on U6's `pin()`)

No conflict found. The two units touch disjoint files (`embedding_migration.py`'s new `migrate`
code vs. `create_workspace.sh`/call sites) and make compatible, non-overlapping assumptions about
`pin()`:

- U8 depends on `pin()`'s **non-fatal** `ModelConfigError` degrade (so workspace creation never
  hard-fails on missing config) — unaffected by U7, which never touches `pin()`.
- U7's `migrate()` independently reimplements `pin()`'s read-before-write discipline for its own
  FR-5 override move (`current = repo.read_model_overrides(ws); repo.write_model_overrides(ws,
  agent=current["agentModel"], ...)`) rather than calling `pin()` itself — correct, since `pin()`'s
  contract is specifically "set if unset," not "set unconditionally to `target_ref`," which is what
  FR-5 needs. The duplication is small (six lines) and each side has its own dedicated test
  (`test_pin_preserves_guard_and_responder_overrides_in_their_own_slots` /
  `test_migrate_write_back_preserves_other_override_kinds`) — an acceptable, reviewed trade-off,
  not a drift risk.
- Both U7 and U8 assume `Repository.read_model_overrides`/`write_model_overrides` keep their
  current signatures and clear-on-`None` semantics; neither this session's nor the concurrent
  unrelated session's changes (`services.py`/`mcp.py`/`repository.py` line ~1283, confirmed a pure
  insertion nowhere near the model-override methods at ~3237+/3334+) touch that surface.
- Both wrappers (`migrate_embeddings.sh`, `pin_workspace_embedding_model.sh` via
  `create_workspace.sh`) resolve `FALKORCHAT_OPENCODE_CONFIG` independently and diverge on
  purpose (fatal for `migrate`, non-fatal for `pin`/`create_workspace.sh`) — this asymmetry is
  documented in both scripts' own header comments, not an oversight.

## What's solid

- The hard-cap-bypass test (`test_migrate_bypasses_the_workspace_hard_cap`) and the
  embed-the-actual-text regression test (`test_migrate_writes_the_embedding_computed_from_each_rows_own_text`)
  are both genuinely load-bearing — I traced each assertion back to a real, distinguishable failure
  mode rather than a tautology (the latter uses a fresh, untouched second `_FakeEmbedder` instance
  as an independent oracle, not the same instance `migrate()` used).
- The DROP-guard resume test (`test_migrate_resume_after_index_dropped_but_not_recreated`)
  reproduces the exact intermediate state `graph-dba` found (index dropped, not recreated, data
  already fully migrated) rather than a simplified stand-in.
- `create_workspace.sh` and its doc updates are exactly as narrow as the plan specified — three
  call-site swaps, one in-process call, comment-only elsewhere — no scope creep.
- Both units' commit-boundary discipline held: `embedding_migration.py`'s `migrate`-only additions
  never touch `repository.py` (deliberately, to stay clear of the concurrent unrelated session),
  and I confirmed that file is untouched by this chain.

## Open questions

None — both units are ready to land as-is. The two minors above (U7's multi-batch coverage gap,
U8's fixed-literal test workspace id) are worth a follow-up ticket but do not block acceptance.
