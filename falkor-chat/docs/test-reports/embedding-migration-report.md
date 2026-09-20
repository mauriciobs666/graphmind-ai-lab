# Embedding model migration & index rebuild — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (M6+)

## Summary

Acceptance/behavior-level pass (U9 of `docs/plans/embedding-migration-coordination.md`) driving
the real, shipped `scripts/embedding_migration.py` `pin`/`migrate` CLI (plus
`scripts/create_workspace.sh`, `scripts/pin_workspace_embedding_model.sh`,
`scripts/migrate_embeddings.sh`) end-to-end against a **live FalkorDB workspace and real LM Studio
embedding models** — the one thing the 30 existing unit/integration tests (`_FakeEmbedder`,
hash-derived vectors) could not exercise. No test-plan document was written separately (task
judgment call: this pass's scope is contained enough that the plan lives in this report's Results
table, per the family convention's allowance for a scoped pass).

Run against commit-clean tree (working tree unmodified — confirmed via `git status --short` on
`falkor-chat/config`, `falkor-chat/scripts` before writing this report); no source, script, test,
or shipped-config file was changed. Environment: FalkorDB reachable (`PONG`), LM Studio at
`http://localhost:1234/v1` with both `text-embedding-qwen3-embedding-0.6b` (1024-dim, current
default) and `text-embedding-granite-embedding-278m-multilingual` (768-dim, candidate) loaded,
confirmed present both before and after the run. Throwaway workspace
`ws:qa-embmig-acceptance-20260920` created, exercised, and `GRAPH.DELETE`d — confirmed absent from
`GRAPH.LIST` both before and after; no other workspace (`ws:test`/`ws:eval`/`ws:acme`/`ws:demo`/
`ws:agent-team`/`ws:nlq-eval`) was touched.

**CPG:** considered, not relevant — `cpg_falkorchat` exists, but this is a black-box, live-behavior
pass driving the running CLI/database directly; nothing here is a static call-graph/data-flow
question the CPG would answer faster than executing the actual code.

**Verdict: PASS WITH FINDINGS.** Every functional requirement exercised in this pass (FR-2, FR-3,
FR-4, FR-5, FR-8a, FR-10, `pin`'s idempotency, `migrate`'s traffic-stop refusal) behaved exactly as
specified against real infrastructure. One finding, already flagged at *review* time
(`docs/reviews/embedding-migration-migrate-impl.md`, "Nit"), is now **confirmed live**: the
traffic-stop refusal path (FR-7) exits non-zero but via an uncaught Python traceback rather than a
clean error message — a real, reproducible UX defect, ranked minor since it does not affect
correctness (confirmed: zero graph writes occur on this path either way). One environmental
observation (not a code defect) is also recorded below.

## Test items & results

| ID | Item | FR(s) | Result | Evidence |
|---|---|---|---|---|
| TP-001 | `create_workspace.sh <ws>` bootstraps schema AND pins the embedding-model override in one call, zero separate operator action | FR-2 | **PASS** | `EMBEDDING_DIM=1024 ./scripts/create_workspace.sh qa-embmig-acceptance-20260920` → schema DDL ran, then printed `ws:qa-embmig-acceptance-20260920 pinned to 'lmstudio/text-embedding-qwen3-embedding-0.6b'`. Verified via direct read: `WorkspaceConfig{embeddingModelOverride:"lmstudio/text-embedding-qwen3-embedding-0.6b", modelOverrideUpdatedBy:"embedding_migration.pin", ...}` |
| TP-002 | Seed real `Message`/`Chunk` rows for `migrate` to act on | — | **PASS** (setup) | 2 `Message` + 2 `Chunk` rows written via `redis-cli GRAPH.QUERY` (direct Cypher — `mcp__cypher__query` refused the write, scoped to `kaizen_team`-shaped writes only, see Feedback). One `Chunk` seeded with `embedding: null, embeddingModel: null` deliberately, to also exercise the self-healing sweep. |
| TP-003a | `migrate` without `--i-have-stopped-traffic`, stdin redirected from `/dev/null` (non-interactive) refuses before any write | FR-7 | **PASS, with a confirmed defect** | Exit code 1; `MigrationAbortedError` raised and printed — but as an **uncaught traceback**, not a clean message (see Defects D-1). Confirmed zero graph mutation: `embeddingModel` counts unchanged before/after (`3` old-model rows, `1` null, same as seed state). |
| TP-003b | Interactive prompt, answering `n`, also refuses before any write | FR-7 | **PASS, same defect as TP-003a** | Ran under a real `tmux` pty (needed — a piped/`script`-wrapped stdin is still not a tty, so `sys.stdin.isatty()` stays `False` and the non-interactive path fires instead). Prompt appeared verbatim (`Has it been stopped? [y/N]`), `n` answered, same `MigrationAbortedError` traceback, exit 1, zero graph mutation confirmed by the same before/after count check. |
| TP-003c | `migrate` with target `dim` undeclared in the model-config overlay aborts before any HTTP call or graph write, naming the missing key | FR-12-shaped edge case (requirements doc, not FR-numbered) | **PASS, same UX defect** | Ran against the real shipped `config/models.json` (granite ref has no `dim` entry there) with `--i-have-stopped-traffic`: `MigrationAbortedError: models.'lmstudio/text-embedding-granite-embedding-278m-multilingual'.dim is not declared in the model-config overlay ...` — again via an uncaught traceback, exit 1, no write. |
| TP-004 | `migrate` WITH `--i-have-stopped-traffic` to a genuinely different-dimension real model re-embeds every row with real HTTP-sourced vectors, rebuilds both vector indexes at the new dimension, and moves the override | FR-3, FR-4, FR-5, FR-8a | **PASS** | Ran with `dim: 768` declared via a **temporary overlay file** (`FALKORCHAT_MODEL_CONFIG` pointed at a copy of `config/models.json` plus the one added `models.<ref>.dim` entry — the shipped file itself was never edited). Output: `Message: migrated 2 rows`, `Chunk: migrated 2 rows`, `migrated to '...granite...' at dim 768`, `Message: migrated 2/2 (skipped 0) in 3.3s`, `Chunk: migrated 2/2 (skipped 0) in 3.3s`. Independently verified via `Repository.read_index_dimension` — both labels now `768` — and by reading every row back: all 4 rows (**including the previously-`NULL`-embedding chunk — the self-healing sweep, confirmed live**) now carry distinct, non-zero, real 768-dim vectors (e.g. `[-0.0394, 0.0471, -0.0293, ...]`) and `embeddingModel == 'lmstudio/text-embedding-granite-embedding-278m-multilingual'`. `WorkspaceConfig.embeddingModelOverride` confirmed moved to the same ref, `modelOverrideUpdatedBy: "embedding_migration.migrate"`. |
| TP-005 | Idempotent resume/no-op: re-running the identical `migrate` command reports zero re-embed work | FR-10 | **PASS** | Second identical run completed in `0.176s` total (vs. `3.3s`+ for the real-embed run) with `migrated 2/2 (skipped 0) in 0.0s` for both labels — no HTTP calls were made (elapsed time is the tell; the keyset read matched zero unmigrated rows). |
| TP-006 | `pin` on an already-pinned workspace is a real no-op | FR-1/FR-2 idempotency | **PASS** | `./scripts/pin_workspace_embedding_model.sh qa-embmig-acceptance-20260920` → `ws:... already pinned to 'lmstudio/text-embedding-granite-embedding-278m-multilingual' — no-op`. Confirmed `modelOverrideUpdatedBy` unchanged (`"embedding_migration.migrate"`, not overwritten by `pin`) — proves no write was actually issued, not just that the printed message claimed one wasn't. |
| TP-007 | Live-only check beyond the fake-embedder tests' reach: real multi-batch keyset pagination | FR-3/FR-4 (§4 item 1 correctness), closes a coverage gap `analyst`'s review explicitly flagged as "never actually exercised by any test" | **PASS** | Migrated back from granite (768) to the original qwen3 model (1024) with `--batch-size 1` against the same 4 seeded rows (2 `Message`, 2 `Chunk`) — forces 2 real keyset pages per label. Output showed each row as its own batch line (`migrated 1 rows (batch ending at 'qa-msg-1')`, then `'qa-msg-2'`, etc.), 4 real HTTP round-trips, `1.7s` total. Verified index dimension reverted to `1024` for both labels and `embeddingModelOverride` moved back to the qwen3 ref. This is genuinely new coverage — the unit-test suite's own multi-batch case never runs against a real embedder or real network latency. |

## Defects

**D-1 (Minor, UX — confirmed live, previously flagged only at static review).** `migrate`'s CLI
entry point (`scripts/embedding_migration.py:main()`, the `migrate` branch) does not catch
`MigrationAbortedError`. Every refusal path this pass exercised (no traffic-stop confirmation —
both non-interactive and interactive-answered-`n`; undeclared target-model dimension) surfaces as
a raw Python traceback on stderr rather than a clean one-line error. Exit code is correctly
non-zero (`1`) in all cases, and — the important part — **zero graph reads/writes occur on any of
these paths**, confirmed by before/after row-count checks. So this is a rough operator experience,
not a correctness defect: a script/cron caller checking the exit code alone is unaffected; a human
operator watching the output sees a traceback where a one-line `ERROR: ...` message was clearly
intended (the message text itself is already good — `str(exc)` is a complete, actionable
sentence). `docs/reviews/embedding-migration-migrate-impl.md` already recommends the fix (a
`try/except MigrationAbortedError as exc: print(f"ERROR: {exc}", file=sys.stderr); return 1` wrap
in `main()`); this pass adds nothing new to that recommendation beyond confirming it reproduces
identically against a real workspace/real config, on all three paths that can raise it (traffic-stop
×2, undeclared-dim ×1), not just the one the review inspected by reading the code.

No functional defects found in `pin`, `migrate`'s happy path, the self-healing sweep, the override
move (FR-5), idempotent resume (FR-10), or the vector-index rebuild (FR-4) — all matched the
requirements doc's acceptance criteria exactly against real infrastructure.

## Coverage & gaps

**Covered live in this pass:** FR-2's one-call bootstrap+pin (TP-001); FR-7's traffic-stop refusal,
both the non-interactive and the interactive-`n` path (TP-003a/b); the undeclared-dimension
precondition (TP-003c); FR-3/FR-4/FR-5/FR-8a's full re-embed + index-rebuild + override-move
against a genuinely different real dimension, including the self-healing null-embedding sweep
(TP-004); FR-10's idempotent no-op re-run (TP-005); `pin`'s idempotency on an already-migrated
workspace (TP-006); and, as net-new coverage beyond what any existing test (fake or real) checks,
real multi-batch keyset pagination end-to-end (TP-007).

**Deliberately not re-covered** (already independently verified — two `analyst` review passes, 30
unit/integration tests, teco's ledger): Cypher shape correctness of the four `§4` helpers, the
DROP-guard resume hazard (index-rebuild crash between `DROP`/`CREATE`), the hard-cap-bypass
regression (workspace override never leaking into which model `migrate` actually calls),
vanished-row skip-and-log semantics, and `create_workspace.sh`'s `ModelConfigError`-degrades-to-
WARNING path. Re-deriving these live would have been redundant per this pass's explicit brief.

**Not covered, and out of scope for this pass:** FR-6/FR-8b (the `ws:eval` golden-set
recall/MRR comparison and a real retrieval-sanity check via the assistant) — those require an
actual `model-bench` run and a live-served workspace respectively, neither of which this pass's
throwaway workspace has (no server process was started against it, by design, to avoid any
live-traffic risk on the shared box). FR-1's one-time production sweep and FR-9's "reusable
on-demand" framing are process/runbook claims, not something a single throwaway-workspace run
proves or disproves further than TP-001/TP-006 already do. The full-outage/server-restart-clears-
cache check (plan §6 test 18) was not run — it needs a live-served process (`start_server.sh`
against a disposable ws), which this pass's scope didn't call for and starting one carries more
shared-box risk than this pass's brief asked for; flagged as a residual risk, not attempted.

**Residual risk:** none rated above minor. The one open item (D-1) is cosmetic and has a
already-recommended, unimplemented fix on file.

## Feedback & recommendations

- **Apply D-1's already-recommended fix** (`analyst`'s review) — a two-line `try/except` in
  `main()`'s `migrate` branch. Low effort, real operator-experience improvement, zero risk (the
  underlying abort logic and its "no writes" guarantee are unaffected either way — this pass
  independently reconfirmed that guarantee holds regardless of the traceback).
- **Testability note, not a defect:** `mcp__cypher__query` (the `cypher` MCP server) only
  authorizes writes shaped for the `kaizen_team` learnings graph (producer-write / curator
  MENTIONS-write / edge-resolve / entry-clear) — it refused a plain `CREATE` against
  `ws:qa-embmig-acceptance-20260920` outright ("Write detected but no `agent` parameter
  supplied" → still refused with `agent` set, since the shape doesn't match any of the 6
  authorized ones). Seeding real workspace data for a live QA pass therefore has to go through
  `redis-cli GRAPH.QUERY` or a `Repository`-based Python snippet, not the MCP tool, despite the
  tool's own description reading as generic ("not limited to `cpg_*` graphs"). Worth a one-line
  note wherever that tool's scope is documented for other agents, so a future QA pass doesn't
  waste a round-trip discovering this the same way.
- **Environmental observation, not a code defect:** the box's checked-in
  `$HOME/.config/opencode/opencode.json` (`FALKORCHAT_OPENCODE_CONFIG`'s default path) currently
  declares the `lmstudio` provider's `baseURL` as a LAN IP (`http://192.168.0.69:1234`) that this
  session's WSL2 network could not route to (`OSError: [Errno 113] No route to host`) —
  `http://localhost:1234` reached the same LM Studio instance without issue (consistent with this
  team's already-documented WSL2-mirrored-networking fallback). This pass worked around it with a
  session-local copy of the config (never touching the real file) rather than editing the shared
  file. Not this feature's defect — `modelconfig.py`'s error surfacing here was actually good (a
  clear `ProviderCallError` naming the exact URL and the underlying OS error) — but worth flagging
  since a DHCP-reassigned LAN IP silently breaking the shared config until someone hits exactly
  this error is a recurring-risk shape, and the fix (point the shared config at `localhost`, per
  the mirrored-networking fallback) is cheap and durable.
- **No testability issues found in the CLI surface itself** — `pin`/`migrate`'s printed output is
  detailed enough (per-batch progress, per-label migrated/total/skipped, explicit restart
  reminder) that this pass never had to guess at internal state; every claim was independently
  verifiable via `Repository.read_index_dimension`/`read_model_overrides` reads, which is exactly
  what made the live verification in TP-004/005/006/007 possible without instrumentation.

## Artifacts

- This report: `falkor-chat/docs/test-reports/embedding-migration-report.md`
- No separate `docs/test-plans/embedding-migration.md` was written (judgment call per this pass's
  scope — the plan is the Test items table above); flag to `teco` if a future, larger pass against
  this same feature (e.g. the eventual real `ws:eval`/production migration, FR-6/FR-9) should get
  a fuller standalone plan instead.
