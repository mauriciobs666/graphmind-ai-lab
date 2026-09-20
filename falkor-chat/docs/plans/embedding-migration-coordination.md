# Embedding model migration & index rebuild — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (M6+)

Coordinates delivery of `docs/requirements/embedding-migration.md` (Status: Ready for design,
2026-09-19): a reusable, on-demand capability to re-embed a workspace's `Message`/`Chunk` data and
rebuild its vector index when the embedding model changes, plus a workspace-level model-pinning
safety net (FR-1/FR-2/FR-5) so a global default swap never silently affects existing workspaces.
Triggered by an urgent LM-Studio memory-pressure swap off the current model
(Qwen3-Embedding-0.6B); destination model choice and which production workspace(s) to migrate are
deliberately out of scope for this chain (requirements doc, "Open questions").

CPG note: `cpg_falkorchat` exists, built from commit `07c252d`; `HEAD` has since moved by exactly
one unrelated commit (`999141f`, a salesperson `systemPrompt` language-salience fix, no relation to
embedding code) — usable for structural navigation, not rebuilt for this chain.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a2ce9950345d8a80a` | delivered | `docs/plans/embedding-migration.md` | `analyst` → — | 204k tok / 64 tool uses |
| U2 | `analyst` | `aea05bc9eb8f79ce7` | delivered | `docs/reviews/embedding-migration.md` | `analyst` → needs changes (2 blockers) | — |
| U3 | `graph-dba` | `a49343cae70f94e54` | delivered | `docs/plans/embedding-migration-graph.md` (+ `claude/graph-dba/falkordb-quirks.md` update) | teco (spot-check) → verified | 185k tok / 42 tool uses |
| U4 | `architect` (resume U1) | `a2ce9950345d8a80a` | delivered | `docs/plans/embedding-migration.md` rev. (in place) | `analyst` (re-review, U5) → **approve** | 343k tok / 125 tool uses (this turn; 547k/189 cumulative across U1+U4) |
| U5 | `analyst` (re-review of U4) | `a84297e88678205db` | delivered | `docs/reviews/embedding-migration.md` Pass 2 (in place) | teco (spot-check) → verified | 116k tok / 16 tool uses |
| U6 | `coder` | `a2cb87d5a804deb2b` | accepted | `scripts/embedding_migration.py` (`pin`) + `scripts/pin_workspace_embedding_model.sh` + `server/tests/test_embedding_migration.py` (§5 step 1) | `analyst` (`aefe5f0b369f3e4d4`) → **approve** | 182k tok / 12 tool uses (follow-up turn; 346k/66 cumulative) |
| U7 | `coder` | `a6bef11910ee45ad3` | accepted | `scripts/embedding_migration.py` (`migrate`) + `scripts/migrate_embeddings.sh` + interrupt/resume tests (§5 steps 4-5) | `analyst` (`a009a2e28690c7b65`) → **approve** | 234k tok / 16 tool uses (follow-up turn; 447k/88 cumulative) |
| U8 | `coder` | `ac2fd2b646978a9e3` | accepted | `scripts/create_workspace.sh` + call-site swaps + doc clauses (§5 step 2) | `analyst` (`a009a2e28690c7b65`, joint w/ U7) → **approve** | 199k tok / 8 tool uses (follow-up turn; 380k/104 cumulative) |
| U9 | `qa-engineer` | `a5c2e4d188f8f891b` | accepted | `docs/test-reports/embedding-migration-report.md` — live acceptance pass, real embedder | teco (independent re-verification) → **PASS WITH FINDINGS** | 160k tok / 62 tool uses |

_U4 correction, 2026-09-19: this row was logged `in-flight` before the revision brief was actually
sent — the agent sat idle since U1's handback until a status-check message (not a revision request)
reached it. Real revision brief (both blockers + 3 minors from `docs/reviews/embedding-migration.md`)
sent 2026-09-19 ~19:55 — genuinely in-flight from that point, delivered ~20:07._

**U4 spot-check (teco, before commit):** independently verified three of the revision's load-bearing
citations directly against source — `scripts/start_agent_team.sh:~202-205` and
`scripts/start_demo.sh:~164-166` both do carry a `bootstrap_schema.sh` call site as claimed (Option
B's named call sites), `scripts/test_queries.sh:~103` is the unchanged `ws:test` bootstrap call as
claimed, and `server/falkorchat/config.py:279`'s `get_context()` docstring does say "M1 resolves
every call to one hardcoded tenant" (backs §2.6's one-process-one-workspace claim). Section
numbering re-checked whole (`grep -n '^## \|^### '`): monotonic 1→7, 2.1→2.9, 3.1→3.5, no
duplicates/gaps — the renumbering claim holds. **Not yet independently re-derived:** whether Option
B's claimed regression (the `FALKORCHAT_ENABLE_AGENT=0` no-config mode) is real, or whether the new
§3.4 step 0 traffic-stop precondition and the `DROP VECTOR INDEX` resume guard are correctly wired
into §5's steps — left for U5's re-review rather than re-verified twice.

**U5 spot-check (teco, before committing/dispatching further):** independently re-verified all six
of Pass 2's newly-cited source facts against live source, not the review's paraphrase —
`start_server.sh:148`, `start_demo.sh:166`, `start_agent_team.sh:205` each do call
`bootstrap_schema.sh` as claimed; `start_server.sh:16-17`'s header text ("Set 0 to serve the UI/REST
without the AI loop") matches verbatim; the six `seed_*.sh` scripts each hold only a precondition
*comment*, never an actual `bootstrap_schema.sh` call (`grep -n bootstrap_schema.sh scripts/seed_*.sh`,
confirmed); and `server/falkorchat/app.py`'s executable `ModelGateway.from_env()` call (line 607) is
reached only after the `if not config.ENABLE_AGENT: return ...` early-return at line 588 — the one
other hit at line 289 is inside a docstring's illustrative code example, not live code, so it
doesn't undercut the claim. All confirmed exactly as Pass 2 states. Verdict **approve** stands.

**U6 verification (teco):** independently reread `pin()` against `repository.py`'s
`write_model_overrides`/`read_model_overrides` and `modelconfig.py`'s `resolve()`/
`Resolution.primary.ref` — every call shape and key name matches exactly, no bugs found there.
Per the "mutate one argument yourself, beyond the implementer's own table" rule, swapped
`guard=`/`responder=` in the write call — the delegate's own two mutations (agent/guard→`None`,
idempotent check disabled) didn't cover this. **All 9 tests still passed** — no test sets distinct
non-`None` `guardModel`/`responderModel` on the same workspace, so the swap was invisible. Mutation
reverted (confirmed clean diff, suite green again); sent back to the same delegate, who added
`test_pin_preserves_guard_and_responder_overrides_in_their_own_slots` and confirmed it catches the
exact swap. Independently reran: `test_embedding_migration.py` alone → 10 passed; full suite minus
`test_services.py` → 2585 passed, clean.

**Unrelated concurrent work detected, not ours — noted, not acted on.** `git status` shows
`server/falkorchat/services.py`/`mcp.py`/`repository.py` and `scripts/bootstrap_schema.sh` +
`server/tests/test_services.py` modified on disk by some other, unrelated session (a hybrid
full-text/vector retrieval feature — `search_chunks_fulltext` — mid-flight, judging by
`repository.py`'s diff hunk at line 1283 and the `test_services.py` failure shape changing between
two runs seconds apart). Confirmed disjoint from this chain: `repository.py`'s diff is a pure
insertion at line 1283, nowhere near `write_model_overrides`/`read_model_overrides` (~3237+,
content unaffected, only shifted); none of U6's three files overlap; `test_services.py`'s 2-3
failing tests are unrelated to `pin()` (`KeyError: 'score'` / `AttributeError` inside
`services.search_documents`, nothing this chain touches). **Do not stage or commit any of those
five files under this coordination** — not ours, not reviewed by us, actively being edited by
someone else.

**U8 verification (teco):** independently confirmed against source — all three call-site swaps
(`start_server.sh:148`, `start_demo.sh:166`, `start_agent_team.sh:205`) forward existing env vars
unchanged; `seed_eval_corpus.py`'s in-process `embedding_migration.pin(EVAL_WS, gateway=gateway)`
reuses the already-resolved `gateway` from line 579, no re-resolve; all five doc-only seed-script
clauses and `AGENTS.md`'s new `create_workspace.sh` row are comment-only, content matches claim
exactly; `create_workspace.sh` itself has no error-handling that would defeat `pin()`'s non-fatal
WARNING path. Ran `test_create_workspace_script.py` independently: 3 passed. Per the "mutate one
argument yourself" rule, tried a mutation distinct from the delegate's own two (order-swap,
swallow-failure): narrowed the pin loop from `for wid in "$@"` to `for wid in "$1"` — bootstrap
still runs against every given id, only the *pin* step silently narrows to the first. **All 3 tests
still passed** — none calls `create_workspace.sh` with more than one workspace id, despite the
script's own usage line advertising `<wsId> [<wsId> ...]`. Reverted (confirmed clean, suite green);
sent back to the same delegate for a two-workspace-id test closing that gap before `analyst`.

**Also flagged by U8, worth tracking but out of scope for this chain:** (1) the full test suite is
markedly more nondeterministic than the "2-3 `test_services.py` failures" baseline once a second
concurrent coder (U7) and two live `uvicorn` processes are also hitting the same shared FalkorDB
instance — a live-instance-contention risk, not a defect of this unit; U8's own `test_repository.py`
ran clean (328/328) both times isolation was checked. (2) A pre-existing, unexercised gap in
`bootstrap_schema.sh`: a *partial* DDL failure that still leaves the workspace's graph key
materialized would let `pin()` "succeed" over an incompletely-schemaed graph — not introduced by
this chain, not caught by any test here or elsewhere; noted for a future backlog item, not blocking
FR-2's delivery.

**U7 verification (teco):** independently confirmed `pin()` is byte-identical to U6's committed
version (diff shows only additions); all four Cypher helpers
(`_read_unmigrated_batch`/`_write_embedding`/`_count_unmigrated`/`_rebuild_vector_index`) match
`docs/plans/embedding-migration-graph.md`'s confirmed shapes exactly — predicate direction (`>`
only), no-op detection (`properties_set > 0`), the `DROP` guard (`read_index_dimension(...) is not
None`); `resolve()`/`embedder()` call shapes (no `ws=`/`overrides=`) match `modelconfig.py` exactly,
and `.embed(text)` matches this codebase's standard embedder interface (`responder.py`,
`services.py`, `tools.py`). Independently reran: `test_embedding_migration.py` alone → 25 passed;
full suite minus `test_services.py` → 2604 passed — matches the delegate's own counts exactly.

Per the "mutate one argument yourself" rule, tried a mutation distinct from the delegate's own
three (guard/responder swap, count-check-always-passes, drop-guard-always-skips): in the re-embed
loop, embedded each row's **id** instead of its **text** (`embedder.embed(row_id)` instead of
`embedder.embed(text)`). **All 25 tests still passed** — nothing asserts on the actual embedding
vector's value, only on `embeddingModel`/counts/index dimension, none of which would change under
this bug. In production this would silently ship every row with a nonsense, id-derived embedding —
indistinguishable from success by every signal `migrate()` reports, while destroying retrieval
quality on that workspace. Reverted (confirmed clean, suite green); sent back to the same delegate
for a test asserting the actual embedded vector against each row's own text before this goes to
`analyst` alongside U6/U8.

**U7/U8 joint review verified (teco), committed.** Independently re-read the joint review
(`docs/reviews/embedding-migration-migrate-impl.md`) and reran the suite myself (30 passed across
both new test files) before committing `09e30282`. Two non-blocking minors filed by the review, not
acted on further (both explicitly judged non-blocking by the reviewer's own analysis, not just
deferred by convenience): (1) `migrate`'s multi-batch keyset paging has no dedicated test proving
`batch_size` changes anything observable — proven to have no correctness consequence today (the
`coalesce(embeddingModel,'') <> $target` clause is the real correctness gate, not the cursor), so
left as a coverage follow-up rather than spinning up another implement→review round for a
zero-risk gap; (2) `main()`'s `migrate` branch surfaces `MigrationAbortedError` as an uncaught
traceback instead of a clean exit — a CLI-polish nit with no functional consequence (confirmed: the
graph is still never touched either way), left for whenever this tool is operator-facing beyond
this dev box.

**All of §5's code-bearing steps (1, 2, 4, 5) are now delivered, reviewed, and committed.** Step 3
(FR-1's one-time sweep) is explicitly not code — a runbook action (run
`pin_workspace_embedding_model.sh` against every existing real workspace once, before the global
default in `config/models.json` is ever edited) — an ops decision for whoever performs the actual
model swap, out of this chain's scope per the requirements doc's own "Open questions."

**U9 verification (teco):** independently confirmed the throwaway workspace
(`ws:qa-embmig-acceptance-20260920`) is absent from a live `GRAPH.LIST` — cleanup genuine;
`git status` shows only the new report file — no source/script/config was touched; the shared
`$HOME/.config/opencode/opencode.json`'s LAN-IP `baseURL` is exactly as the report describes,
confirming it wasn't edited. D-1 (the traceback-vs-clean-exit defect, now confirmed live on all
three trigger paths) was fixed directly by teco as a genuinely trivial single-function change — a
`try/except MigrationAbortedError` wrap in `main()`, exactly as both the code review and this QA
pass independently specified — with a new regression test, mutation-tested (reverting the fix
reproduces the uncaught traceback; the new test catches it). Full suite reran clean: 2606 passed
(minus `test_services.py`'s unrelated concurrent work). Committed `e933c4fd`.

**Coordination status: all planned units (U1-U9) delivered, reviewed/verified, and committed.**
`pin`, `migrate`, and FR-2's enforcement mechanism are built, code-reviewed clean (two `analyst`
passes), and live-verified against real infrastructure (`qa-engineer`, PASS WITH FINDINGS, one
Minor found and fixed). What remains is explicitly out of this chain's scope per the requirements
doc itself: FR-1's one-time production sweep (an ops runbook action for whoever performs the actual
model swap), and FR-6/FR-8b's `model-bench` golden-set validation + a live retrieval-sanity check
(both require an actual destination-model decision and a real migration target, neither of which
this chain was ever scoped to choose). Whether to flip `docs/requirements/embedding-migration.md`
and this coordination doc to `archived` now, or leave them `active` until an actual production
migration exercises the remaining items, is a stakeholder call (`tico` owns the requirements-doc
flip) — not made here.

## Pause (2026-09-19, user-requested) — resumed 2026-09-20

Paused at the user's request for exclusive `model-bench` machine access; resumed in a new session
per the plan (ledger + `git log`/`git show` against `efeb4a88` reconciled first, per this doc's own
resume instructions — U1-U4's committed artifacts matched exactly what this ledger described).
**U5 is now delivered and independently spot-checked — the design-review gate on
`embedding-migration.md` is closed.** Nothing about the plan itself is still blocked on review.

Two things remain open, tracked as U6-U8 above:

- **Implementation, unblocked now:** `graph-dba`'s companion note (U3) is final (item 6 verified
  safe), so both `pin` (§5 step 1, U6) and `migrate` (§5 steps 4-5, U7) can be built regardless of
  the item below. U7 is sequenced after U6 (both touch `scripts/embedding_migration.py`).
- **Stakeholder decision (resolved 2026-09-20):** Option B chosen for FR-2's enforcement mechanism
  (§3.2) — `create_workspace.sh` replaces `bootstrap_schema.sh` as the canonical entry point at the
  three named call sites, per the plan's own recommendation. U8 (§5 step 2) is queued behind U6
  (needs `pin()` to exist); once U6 lands, U7 and U8 can run in parallel (disjoint files).
