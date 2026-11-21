# Change History — coordination-graph

> Dated log of actual changes to the `coordination-graph` component. Most recent first.

## 2026-11-20 — S8-pre: deliberate shared-instance path; `PARSER_VERSION` 1 -> 2

Pre-work for S8a/S8 (`claude/docs/plans/coordination-graph.md` S8a/S8 rows; graph note v4 §5.5; impl review `## S3b`
decision on `connect_shared`, `## S6` F1). Stakeholder decisions (2026-11-20): build a separate, explicit shared path
so the ephemeral guard stays unweakened; bump `PARSER_VERSION` before the first shared load.

- `store.connect_shared(host, port, key, confirm)` / `require_shared`: loopback only, `confirm` must equal the graph
  key exactly, refused while `COORDGRAPH_EPHEMERAL_PORT` is set. `store.open_graph(..., confirm=None)` is the one
  choke point the commands use: `None` is `connect` (the ephemeral guard, byte-for-byte unchanged), anything else,
  even `""`, is `connect_shared`. KB counterpart `store.require_shared_kb_url` (loopback, port 8200 only); the
  engine picks the check from the mirror's mode, so a shared mirror never pairs with a scratch falkor-chat or the reverse.
- CLI: `--confirm GRAPHKEY` on every store-backed command; `COORDGRAPH_SHARED_CONFIRM` is the same value as a
  standing env setting, honoured by the routine commands only. `bootstrap`, `rebuild` and `kb-reconcile` refuse
  the env alone and need the flag. A detached `kb-finish` inherits a flag confirmation.
- `PARSER_VERSION` is 2: every graph stamped 1 re-syncs on the next `sync`/`drift`. `tests/test_requirements.py`
  now reads the constant instead of pinning 1; `tests/test_sync.py` pins 2 and the 1 -> 2 resync.
- How the non-interactive hook is authorized on the shared instance is not decided here (S8 design question).
- Verified: 1009 offline tests (955 before; 52 new in `tests/test_shared.py`, 2 in `test_sync.py`), no database
  connection in any new test.
- S8-pre gate fixes (F1, F2, F3): shared mode now also requires a shared port (`SHARED_PORTS`, so the two modes are
  exact complements) and the key to be the coordination mirror's (`SHARED_KEYS`), so a confirmation cannot name another
  component's graph (`cpg_*`, `kaizen_team`); documented mapping: the real falkor-chat (8200) fronts `falkordb-dev`
  on 6379. `kb-sync --detach-finish` hands `--confirm` and `--graph` to the child unconditionally. 1025 offline.

## 2026-11-20 — S7a: read views (`views.py`)

Delivered base-plan step S7a (`claude/docs/plans/coordination-graph.md` §3.4/§3.4a, FR-6/FR-8/FR-9/FR-12, AC-5/AC-6/AC-7/AC-9;
plan3 §3.1/§3.2/§3.4/§6 test 13; graph note v4 §7.2/§7.3/§7.4/§7.6). `coordgraph/views.py`: `inflight [--waiting]`,
`unit <slug|ID> [<unit>]`, `planned <component|agent> [--all]`, each through `RO_QUERY` only, each printing the sync
header (`drift.render_header` over every covered file, registry line included) with `⚠ drift` on the rows whose file
row moved. `unit` expands an ID argument through `Identity.resolve` and joins `IN $candidates` (feature ledgers by
`Source.itemId`, tasks by `Unit.itemId`), so the new ID, an alias and the old legacy name find the same ledger; the
slug read always runs too, and a non-resolving ID is reported with its reason, never guessed. `inflight` decides
"picked up" for a requirement by slug (§7.2(d)) and by its `Item:` candidates (§7.6(d)). The lookup rule is
`identity.select_units`; `~n` is tried as written, then stripped. `--waiting` is the stakeholder questions only
(§7.2(b)). `cli.py` gains three one-line insertions (`views` import, `views.register(sub)`, `**views.COMMANDS`); the
views import `identity` lazily, so the sync path still loads neither it nor `wilib`.
Tests: `tests/test_views.py` (38, offline, a fake `ro_query` graph with no `query` method) and
`tests/test_views_live.py` (21, ephemeral container): AC-5 four kinds with component, path, header; AC-7 an
all-accepted active ledger stays, an archived one and its paused unit never list, and after an archived flip `unit`
still finds the minimal units; FR-6(c); AC-6 paused question and date, repeats, `~n`, label vs exact key; AC-9
and the `⚪` trailing group; plan3 test 13 `unit` by new ID, old ID and task ID after a `wi move` on a scratch
registry, an `IN` index plan, `Registry: unavailable` under a bogus `COORDGRAPH_WI_ROOT`. Mutation run on a copy:
33 single-point mutants, all killed (two survivors on the first pass, a shared-slug archived paused unit and a
word-only/emoji-only `⚪` marker, each fixed by a new test).
S7 gate fixes (F1-F6, `claude/docs/reviews/coordination-graph-impl.md` `## S7`): `inflight` resolves a requirement's `Item:` in its own component's scope; an unsynced graph key now says to run `coordgraph bootstrap` and `sync`; new tests for headerless ledger/review, two-token `Depends on`, the closed-item filter, `src/a` vs `src/ab.py`, `M5.1`/`M5.2`, a live `propose` run over `ITEMS`/`DEP_ITEMS`, and propose's stdlib-only imports (offline 941 → 955, live views file 21 → 24; 17 mutants, all killed).

## 2026-11-20 — S7b: milestone proposal (`propose.py`, `commands_propose.py`)

Delivered base-plan step S7b (`claude/docs/plans/coordination-graph.md` §3.7, FR-13–FR-15, AC-11–AC-14; plan3 §3.1/§3.3
for identity and delivery). `coordgraph/propose.py`: `propose(items, resolver, per_milestone=6) -> Proposal`, pure
(no store, `wilib`, network or file read; `resolver(token, scope) -> DepResolution` answers whatever needs one) plus
`touch_overlap`. Steps 0–6 as written: fixed items and milestone ordering (`ahead-of-Mx` just before `Mx`), insertion
into the earliest not-started gap, classify, dependency resolution (internal / in-progress → wave ≥ 2 / external →
held / delivered / unknown → not plannable), cycles, holding propagation, effective-priority rank, list-scheduled
waves with conservative glob overlap, capacity cut, one criterion per placement, *approved placements now
impossible*. `commands_propose.py`: `coordgraph propose <scope> [--per-milestone N] [--graph K]` reads note §7.5
(i)/(ii) by `RO_QUERY` (the item query also returns `milestoneOrder`, `milestoneStarted` and the backlog path,
which §7.5 (i) lacks), resolves tokens through `identity` and the history files, prints under the sync header.
Choices the plan left open: a fixed `🟡` dependency imposes no wave constraint; unblocks counts plannable-or-fixed
transitive dependents; an in-progress dependency blocks insertion; an equal `order` between a fixed item and its
dependency is *impossible*; a gap whose `ahead-of-Mx` name is already an approved milestone is skipped.
Verified offline: `tests/test_propose.py` (pure) and `tests/test_propose_cli.py` (fake graph); a 39-mutant
check (plan §5's six, the rejected alternatives, and silently corrupted arguments) with every mutant killed. No
live-graph run of the two new statements yet.

## 2026-11-20 — S6: KB sync (`kbsync.py`, `commands_kb.py`)

Delivered base-plan step S6 (`claude/docs/plans/coordination-graph.md` §3.5, P2-M3, R10; graph note v4 §2.4, §6.4).
`coordgraph/kbsync.py`: falkor-chat REST writes (`KbHttp`, stdlib `urllib`, producer `kb-mirror`), read-only
observation of `$COORDGRAPH_KB_GRAPH` (`Observer`, §6.4 statements verbatim), and the per-section engine
(`KbSync`): append-before-delete with `COORDGRAPH_KB_BUDGET_S` and `pending`, the per-section lease with token
fencing and rules L1-L3, cancel-on-re-edit, duplicate-claim refusal plus the receipt check, split modes,
`needs-resplit`. CLI (`commands_kb.py`): `kb-sync [PATH...] [--detach-finish]`, `kb-status [--ids ID...]`,
`kb-reconcile`, `kb-finish [--once]` (`kb-finish.lock`, single instance). Exit 0 settled, 2 an update is not
complete, 1 the command could not run. The mirror connection is still `store.connect` (ephemeral only); the
sync path imports no `wilib` (plan3 test 12 now also lists `kbsync` and `commands_kb`). `falkor-chat/scripts/
seed_agent_team.sh` gains the `kb-mirror` row.

- **Decisions the plan left open.** Unchanged claims are kept, not re-posted: the pending fields describe the *new*
  documents only and the swap merges them with the kept old ones, so one edited bullet costs one POST. A heading
  rename with an identical body is accepted as a supersede of the vanished section's document (re-embeds, never
  refused as a duplicate). A client-timed-out POST is outcome-unknown (review P4-m1): the section fails, and the
  retry's receipt naming an unmapped `kb-mirror` document of the same text is accepted and that twin deleted. Added
  to graph note §2.4 (owner to fold in): `KbSection.mode`, `needsResplit`; while pending `pendingClaimTitles`,
  `pendingKeepHashes`, `pendingMode`.
- **Marker-stripped claims.** `kb-reconcile` adopts a stored document whose hash equals the claim text *without* its
  list marker, mapping it under its real (stripped) hash with no `sectionHash`, so the first `kb-sync` replaces it
  append-before-delete. Observed: `claude/graph-dba/falkordb-reference.md` parses to 17 bullets-mode claims in 3
  sections; a live run on a scratch falkor-chat (the 17 stored stripped, adopted, then synced) created 17 new
  documents and deleted the 17 stored ones. (`claude/frontend-engineer/frontend-quirks.md` also parses to 6
  bullets-mode claims; the plan names only the 17.) The shared instance was not touched: whether its stored claims
  are stripped is S8a's to confirm.
- **Tests.** Offline: `tests/test_kbsync.py` (engine over the fakes of `tests/kbfakes.py`), `tests/test_kb_cli.py`
  (`KbHttp` against a local stdlib server; the four subcommands). Live (ephemeral container + scratch falkor-chat,
  workspace `coordtest`): `tests/test_kb_live.py` including the R10 contract test (the observation reads find
  `HAS_CHUNK`, `INGESTED_BY`, `Chunk.embedding`, `Document.currentVersion`, `textNormalizedHash`; falkor-chat's
  normaliser equals ours; the seeded roster carries `kb-mirror`).

S6 gate fixes (`claude/docs/reviews/coordination-graph-impl.md` `## S6`, F1-F6; F7 skipped): the ephemeral mirror now
also refuses a falkor-chat URL that is non-loopback or on the shared port 8200 (the unset default), in
`store.require_ephemeral_kb_url`, called by `_engine` before any request; an unchanged re-sync of a pending section
is left to `kb-finish` (no cancel, `pendingSince` kept); the claims file text wins over the old mapping, and a
receipt superseding a document whose section no longer carries that claim is accepted, so two headings exchanging
claims in one edit both sync; each delete renews the lease; `kb-status` exits 2 when anything is flagged (1 is only
"could not run", as the other `kb-*` commands), and one file's unforeseen error in `kb-sync` is that file's `failed`
outcome. New tests pin the `_adopt` and twin-check guards (survivors of the gate's mutation run).

## 2026-11-20 — S3: store, sync, rebuild, drift, query (S3a `683eb212`, S3b)

Delivered base-plan step S3 (`claude/docs/plans/coordination-graph.md` §3.1-§3.3, §4; graph note v4
§3-§5). S3a: `store.py` (bootstrap, per-source atomic replace, the ephemeral-endpoint guard),
`dryrun.py` (`rebuild --dry-run`), `commands_store.py`. S3b: `sync.py` (`sync PATH.../--changed/--all`,
`rebuild`, the parity proof, orphan sweep, `SyncState`), `drift.py` (the §3.3 read-time check and sync
header), the `sync`, `rebuild`, `drift`, `query`, `bootstrap` subcommands, each taking `--graph`.

- **Parser gaps closed.** Every `:Source` now carries `rowCount` (`0` for a review/requirement; the
  backlog's `itemCount` is gone: graph note §2.2 names `rowCount` for every kind and has no `itemCount`).
  A `Priority` of the form `medium (note)` keeps its leading word. `test_real_repo_dry_run_is_clean` is no
  longer an expected failure. `PARSER_VERSION` stays 1: nothing was mirrored before this change.
- **Write boundary** (graph-dba gate m1, m2). An indexed non-unique value that is empty or over 512
  bytes is refused; a NUL in any string becomes U+FFFD and the source gains a warning. Parity expects
  the same scrubbed form (`store.write_form`).
- **Parity** compares a fresh parse with one export of the mirror, re-hashing every child (never the stored
  `rowHash`/`itemHash`) and the source's property map minus `syncedAt`/`mtimeNs`.
- **Offline path.** `sync`, `rebuild` and `drift` import neither `wilib` nor `identity`; `lint`,
  `history-coverage` and the header's `Registry:` line load `identity` (read side) for registry-backed joins.
- **Not yet:** the views (`inflight`, `unit`, `planned`: S7a), KB sync (S6) and the hook (S8). Every store
  command still refuses any endpoint but an announced ephemeral one; S8a needs its own override.

Live tests (ephemeral container only): AC-2, AC-4, AC-6 (the note 7.3 rows), AC-7 and plan3 section 6
test 13 (the legs that need no S7a view).

S3 gate fixes (`claude/docs/reviews/coordination-graph-impl.md` `## S3b`): an unsafe indexed value is a `store.UnsafeValue`
(a `StoreError` and a `KeyHygieneError`), so it fails one path instead of aborting `sync`, `parity` and `drift`; parity
also compares node labels and relationship types; `drift` reports `stale-parser`; `shapes.offline_candidates` drops the
bare string for a scope-qualified legacy token, as `identity` does. `PARSER_VERSION` still 1 (stakeholder decision).

## 2026-11-20 — S1e: backlog parser, history match, offline CLI; history coverage probe

Delivered the remainder of base-plan step S1 (`claude/docs/plans/coordination-graph.md` §4, with
`claude/docs/plans/coordination-graph3.md` §3.3, §3.5, §3.6): `parse/backlog.py`, `history.py`,
`shapes.py`, `scope.py`, `lint.py`, `migrate.py` and the CLI (`python -m coordgraph`,
`coordgraph.sh`): `lint [--count]`, `migrate --dry-run|--apply PATH`, `facts`,
`history-coverage`. Offline only; no graph code.

**Coverage probe** (`coordgraph history-coverage`, over every history file at repo HEAD `16d5120d`
plus the uncommitted working tree of 2026-11-20). For each ID a history file mentions that is open
in no backlog: matched = a delivery-shaped heading names it; unmatched = no such heading. Rate =
matched / (matched + unmatched).

| History file | Matched | Unmatched | Rate |
|---|---|---|---|
| `claude/analyst/kaizen/history.md` | 0 | 19 | 0% |
| `claude/architect/kaizen/history.md` | 3 | 19 | 14% |
| `claude/cobb/kaizen/history.md` | 3 | 58 | 5% |
| `claude/coder/kaizen/history.md` | 4 | 12 | 25% |
| `claude/data-scientist/kaizen/history.md` | 1 | 11 | 8% |
| `claude/devops/kaizen/history.md` | 0 | 9 | 0% |
| `claude/frontend-engineer/kaizen/history.md` | 1 | 4 | 20% |
| `claude/graph-dba/kaizen/history.md` | 2 | 12 | 14% |
| `claude/qa-engineer/kaizen/history.md` | 1 | 15 | 6% |
| `claude/security-expert/kaizen/history.md` | 1 | 9 | 10% |
| `claude/tdd-engineer/kaizen/history.md` | 2 | 12 | 14% |
| `claude/teco/kaizen/history.md` | 15 | 16 | 48% |
| `claude/tico/kaizen/history.md` | 4 | 6 | 40% |
| `decision-models/docs/HISTORY.md` | 0 | 1 | 0% |
| `docs/HISTORY.md` (cpg) | 9 | 79 | 10% |
| `falkor-chat/docs/HISTORY.md` | 21 | 77 | 21% |
| `mcp-monitor/docs/HISTORY.md` | 0 | 1 | 0% |
| `model-bench/docs/HISTORY.md` | 0 | 17 | 0% |
| `opencode/docs/HISTORY.md` | 0 | 0 | n/a |
| **Total** | **67** | **377** | **15%** |

Denominators (recounted 2026-11-20 over the same 444 IDs, none open in any backlog): 268 of them are
never named in a history *heading*, so no heading rule could match them. (A) any mention, strict
rule: 67/444 = 15%. (B) restricted to the 176 heading-named IDs, strict rule: 67/176 = 38%.
(C) the same 176, a heading carrying `✅` or any delivery verb (`closed`/`delivered`/`fixed`/
`shipped`/`done`, plus `implemented`/`resolved`/`landed`/`complete(d)`/`merged`): 123/176 = 70% (an
upper bound: `(K-023)` in "U46: made the stamp's closed list enforceable" counts; with only the
five rule verbs it is 110/176 = 62%). 15% is therefore not a defect rate of the rule; among
heading-named IDs it misses real forms (verb-first `Closed C-309`, `K-005 + K-006 closed`), which
`lint` reports as a visible `unresolvable-dependency` error, never a false delivery.

Reading: today's history files match few deliveries, as the plan expected (P2-M4). Correctness
rests on the delivery-heading convention (`<ID> delivered` / `<ID> closed`, slash-joined
siblings) and on `lint` flagging a `Depends on` ref that is open nowhere with no delivery
heading; existing deliveries are not backfilled. An unmatched ID here is an ID *mentioned*, not
necessarily one that was delivered.

Verified offline: the full suite from `coordination-graph/` (counts in the S1e hand-off); the
reformat applied only to scratch copies with their own registry, never to the repo's
`.work-items/`.

## S8a: first load into the shared instance (graph-dba, 2026-11-20/21) — PAUSED at note §5.5 step 4

Steps 1–3 passed; step 4 stopped on the "if `coordination` already exists, stop and ask" rule. Nothing was written to the shared instance except one `BGSAVE`.

- **Step 1, offline validation:** `rebuild --dry-run` exit 0, no violations (UNIQUE/indexed keys incl. archived `Unit.key`s and the KB registry, duplicates, parameter sizes). Sources: backlog 19, ledger 98, requirement 54, review 375 (546). `statusToken` tally: abandoned 5, accepted 662, delivered 413, gated 53, in-flight 40, paused 5, queued 18, unknown 106. Every `unknown` unit sits in an `archived` ledger (17 ledgers checked by `Status:`; none in a non-archived one, so the `H1` cell in `docs/plans/python-web-quirks-kb-coordination.md` no longer yields `unknown`). Finding: `coordgraph lint` exits 0 with `0 error(s), 0 warning(s)` and does not report `unknown` statuses (`coordgraph/lint.py` has no ledger/`statusToken` check); the evidence above comes from `rebuild --dry-run`.
- **Step 2, rehearsal** (`falkordb/falkordb:v4.18.11`, 127.0.0.1:6395, no volume, `COORDGRAPH_EPHEMERAL_PORT=6395`): `bootstrap` ok; `rebuild` #1 `546 synced`, `parity: 546 expected, 546 in graph`, `parity: ok`; `rebuild` #2 identical (`parity: ok`). Counts both runs: Source 546, Unit 1302, Item 246, SyncState 1; HAS_UNIT 1302, HAS_ITEM 246. `PING` PONG after both. `used_memory` 2,649,096 → 7,202,888 (after #1) → 7,244,232 (after #2). Views run once: `drift` (none), `inflight`, `inflight --waiting`, `unit`, `planned`, `propose`, `query`. Container stopped.
- **Step 3, freshness:** re-run dry-run manifest identical to step 1's (`diff` empty, 559 lines).
- **Step 4, shared instance:** `GRAPH.LIST` = `cpg_falkorchat ws:eval ws:k069probe ws:acme ws:nlq-eval ws:demo ws:agent-team ws:pwq-test reference coordination`; `used_memory` 504,190,584; `BGSAVE` ok (`LASTSAVE` 1795220455 → 1795220744, `rdb_last_bgsave_status:ok`); `:8200/health` 200. **`coordination` already exists** with schema (bootstrap-shaped indexes) and one node, `SyncState {id: mirror, schemaVersion: 4}`, no `Source`/`Unit`/`Item` data. Awaiting the stakeholder's ruling; steps 5+ and the KB seeding not run.

S8a resumed (2026-11-21, stakeholder ruling: load into the existing near-empty `coordination` key, no `GRAPH.DELETE`). A fresh `rebuild --dry-run` exited 0 first (review count 377; the manifest differed from step 1 in `coordination-graph-coordination.md`, `connected-code-graph-coordination.md` and a new `connected-code-graph-p1-55-impl.md`, all re-validated by that full run).
- **Step 5, live load** (`--confirm coordination`, port 6379): `bootstrap` printed `nothing to do` (schema already complete); the pre-existing `SyncState {id: mirror, schemaVersion: 4}` was not changed by `bootstrap`, and `rebuild` then added `lastRebuildAt` and `parserVersion: 2` to it. `rebuild` #1: `548 synced`, `parity: 548 expected, 548 in graph`, `parity: 2 difference(s)` (exit 1) — `falkor-chat/docs/BACKLOG.md` was edited by another session during the run (file mtime 08:18:42, after the parse); not a parser defect. `rebuild` #2: `548 synced`, `parity: ok` (exit 0). After: Item 246, Source 548, Unit 1302, SyncState 1, HAS_ITEM 246, HAS_UNIT 1302; `PING` PONG; `GRAPH.LIST` keys all present; `:8200/health` 200; `used_memory` 504,280,168 (before) → 508,096,856.
- **KB seeding (2026-11-21):** `seed_agent_team.sh` exit 0 (14 agents, incl. `kb-mirror`). `kb-reconcile --confirm coordination` (`COORDGRAPH_KB_URL=http://localhost:8200`): 199 results, `0 to review`; 169 adopted, 30 unadopted (`0 of 1 claims have a stored document`); the 3 `falkordb-reference.md` sections with stored text flagged `marker-stripped text: the next kb-sync re-ingests it` (5, 4, 8 documents). Check on the shared instance: the 19 stored `falkordb-reference.md` documents contain no `<!--` and no `kb-split` (0 of 19).
- **First `kb-sync` (shared, `--detach-finish`): INCOMPLETE, open.** 6 synced, 27 pending (`submitted, not yet searchable`), 166 unchanged; the three `falkordb-reference.md` sections synced as `5 new, 0 kept, 5 removed` / `4 new, 0 kept, 4 removed` / `8 new, 0 kept, 8 removed`. Observed afterwards: of the 19 documents mapped for `falkordb-reference.md`, 16 have `status: failed` and 3 `ready` (the failure reason is not exposed by `get_document`); one analyst claim is `failed` as well; `ws:agent-team` Document status counts `ready 511, processing 9, failed 33` (16 of the 33 predate the sync; `processing` drains about one document every 1–2 minutes). `kb-status` still lists 20 unmirrored claims and `last sync 2026-11-21T11:32:08Z`; the detached `kb-finish` (pid 3755855) was still running when this was written. Gate not closed; needs the `kb-finish` outcome and a decision on the failed documents.
- **S8a KB sync diagnosis (2026-11-21 08:42 -03, read-only):** `kb-finish` (pid 3755855) was still running at 08:42; `kb-status` unchanged (20 unmirrored). `ws:agent-team` Documents: ready 512, processing 8, failed 33. Failed by source: 16 `falkordb-reference.md` claims plus 1 other claim created in the sync window (17), and 16 older ones created 10-21 … 11-19. Failure reason is not stored on the `Document` (properties: `documentId`, `title`, `text`, `sourceFormat`, `sourceKind`, `status`, `pendingJobs`, `createdAt`, `currentVersion`, `textNormalizedHash`); it is only in the server log `/home/mauricio/.local/state/graphmind-ai-lab/agent-team-8200.log`. All chunks of the failed documents carry an `embedding` (39 of 39), so embedding succeeded. In the window the log shows 30 `background extract failed` errors, every one `ProviderCallError lmstudio/qwen/qwen3-4b-2507 … /v1/chat/completions: TimeoutError: timed out` (08:33–08:35), and no embedding errors; `docs/SERVER.md` §1 says an extract job's timeout is not retried and any failed job marks the document `failed`. Best-supported hypothesis (not proven): the sync submitted ~40 documents within about 90 s, each scheduling one extraction chat call per chunk against the single LM Studio chat model, the queue exceeded the extraction timeout, and the extracts timed out. A probe at 08:42 (`max_tokens: 3`, `qwen/qwen3-4b-2507`) got no answer within 60 s while a one-word embedding returned at once; both models are `loaded`; `processing` documents still carry `pendingJobs` (1–27), so the chat endpoint is still saturated.

## S8 (code half): the post-edit hook (coder, 2026-11-21)

Built `hooks/post-tool-sync.sh`, `hook-sync` and `retry-failed` (`coordgraph/commands_hook.py`, state in `coordgraph/hookstate.py`, wired in `cli.py`) to plan §3.2. **Not registered** (no `.claude/settings*.json` edit; `cobb` authors the block, the stakeholder applies it), and the go-live (`sync --all`, stamp initialisation) has not been run. Nothing touched the shared instance, `ws:agent-team` or :8200.

- **Contract.** Bash fast path: one pruned `find -newer .state/stamp` plus the KB registry, no Python; a changed mirrored file runs `hook-sync` under `timeout 50` (FalkorDB ping 1 s; for a KB file falkor-chat `/health` and embedder `/models`, 2 s each; a file whose hash already equals the graph's is a no-op); exit 2 names every file that is failed or pending, on every such edit (no throttle); an exit other than 0/2 from `hook-sync` is the hook's own fault (files appended to `failed.pending` under `flock state.lock`, `retry-due` +60 s, exit 2). The stamp goes to `max(stamp, call start)` regardless of failure. No stamp: one `--first-run` hash-compare, the stamp created by `hook-sync` only once FalkorDB answered (down: silent, `retry-due` +60 s).
- **Failure and backoff.** `failed.json` per repo-relative path (`sha256`, `error`, `attempts`, `nextRetryAt`); backoff 1, 2, 4, 8, then 15 minutes; a new edit (different sha256) restarts it. With no changed file and `retry-due` passed the shell bumps `retry-due` +60 s first, then launches `retry-failed` detached (`setsid`, all three streams redirected). `retry-failed` (one instance, `retry.lock`) retries only due entries, drops a gone file or one whose hash equals the graph's without deleting any source, and leaves a KB section inside its 15-minute `kb-finish` cap to `kb-finish`. A Write/Edit payload naming a file in the failure state re-enters the sync (R12).
- **Kill switch.** `COORDGRAPH_HOOK_DISABLED` in `1/true/yes/on` (config's set) or `.state/disabled`: exit 0.
- **Authorization (stakeholder decision, option A).** Both commands are routine in `commands_store.confirmation`; the script passes the LITERAL `--confirm coordination` itself (never `$COORDGRAPH_KEY`), unsets any ambient `COORDGRAPH_SHARED_CONFIRM` and exports none; `store.require_shared`/`require_shared_kb_url` unchanged, so the reach stays the coordination mirror and its KB mapping.
- **Plan vs code.** (1) The guard refuses `--confirm coordination` on an ephemeral port by design, so no live test can run the script's real flag: live tests use a shim that drops that one pair (the literal and its delivery are pinned by a text test and a recording fake). (2) Two test seams in the script, `COORDGRAPH_HOOK_CLI` and `COORDGRAPH_HOOK_TIMEOUT_S`. (3) §5's "slow embed" leg needs the health probe to pass while embeds are slow; the fake embedder gates `/models` too, so the test points `COORDGRAPH_EMBED_URL` at a tiny always-200 server (the real fake stays slow). (4) `hook-sync --first-run` takes no PATH (it discovers itself) and owns the stamp. (5) `retry-failed` honours `nextRetryAt` per entry, not only `retry-due`.
- **Fast path measured** 2026-11-21 on the real repo tree (4,424 files after pruning, CLI stubbed, scratch state dir): 20 calls 10.8-13.4 ms (median 11.4); five earlier calls 22-31 ms, first after idle 77 ms. Recorded in `README.md` ("The post-edit hook").
- **Verified.** Offline suite from `coordination-graph/`: 1115 passed, 86 deselected (1025 before this unit's 90 new cases: `tests/test_hook_sync.py` 38, `tests/test_hook_script.py` 52 from 28 functions). Live `tests/test_hook_live.py` on an ephemeral container + scratch falkor-chat: 10 passed (AC-3 on both events, no-change < 50 ms with no Python, FalkorDB outage and heal, first run, mirror-down first run, KB edit, embed down, slow embed with detached `kb-finish`, KB-down first run). 33 deliberate mutants, all killed.
- **S8 gate fixes** (2026-11-21, `claude/docs/reviews/coordination-graph-impl.md` `## S8` F1-F6): hook reads stdin with `cat` (300 KB payload 95.9 -> 15.9 ms to process exit, 2 MB 535 -> 33 ms); no-change timing tests assert best-of-20 and a relative large-payload bound (the old max<50 ms failed 3/3 under 3x CPU load, new forms pass 3/3); `store.require_endpoint` guard runs before the ping, a refusal is its own `Run.refused` (message, `hook.log` line, no `failed.json`, first-run re-ask 15 min); `COORDGRAPH_HOOK_TIMEOUT_S` validated; shell discovery now filtered to `discover.py`'s shapes (a test found two divergences, `docs/reviews/*/x.md` and nested `claude/`); faulting first run lists 20 files; README/AGENTS corrected (loaded, KB URL default, bash 5). Offline 1127 passed / 86 deselected; live hook tests 10 passed; mutants killed.
- **S8 re-gate F2 fix** (2026-11-21, `## S8 Pass 2`): the no-change timing assertions (offline and live) compare the best of 20 against a same-loop kill-switched baseline (< 6x + 30 ms) instead of an absolute wall time; 5/5 pass under 24 busy loops on 8 cores, a planted `sleep 0.15` in the fast path fails it; README states the 15-minute first-run refusal window.

S8a KB retry (graph-dba, 2026-11-21): `kb-finish` had exited and a 3-token chat probe to `qwen/qwen3-4b-2507` answered in 3.4 s (`ws:agent-team` processing 0). `kb-sync -v --confirm coordination claude/graph-dba/falkordb-reference.md` printed `5 section(s), 0 not complete` with all five sections `unchanged`: it compares only the section/mode hash (`kbsync.py` `_sync_locked`) and does not look at document `status`, and `kb-reconcile`'s `_verify` treats only missing or hash-mismatched ids as bad. Observed afterwards: of the 19 documents mapped for `falkordb-reference.md`, 3 `ready` and 16 still `failed`; `ws:agent-team` ready 516, failed 33. A second `kb-sync` is therefore not a retry path for `failed` documents. Stopped before the review-techniques.md path and the remaining files. `kb-status`: last sync `2026-11-21T12:16:25Z`, 17 unmirrored; `drift --confirm coordination`: `Drift: 15 source(s)` (files edited since the load; hook not registered).

KB failed-document retry (tdd-engineer, 2026-11-21, `## KB failed-retry` gate): `kb-sync`/`kb-reconcile` re-post a mapped `failed` document only when it is not searchable (chunks missing or unembedded), because the re-POST auto-supersedes the old same-text document (`documentCurrent=false`, not restored on delete); an embedded-but-failed one is `degraded` (also in `kb-status`, with `failed-doc` for the unsearchable), never re-posted. Retries are bounded by `:KbSection.retryCount`/`lastRetryAt` (3 attempts, 30 min doubling backoff, 4 sections per run; `COORDGRAPH_KB_RETRY_*`), cleared by a swap. 1148 offline passed, 12 KB live (ephemeral).

S8a finding, failed documents are still searchable (graph-dba, 2026-11-21, read-only): all 33 `failed` `ws:agent-team` Documents (16 `falkordb-reference.md` claims, 1 analyst claim, 16 older) have `currentVersion = true`, and every one has all its chunks carrying an `embedding` and `documentCurrent = true` (33 of 33; none without chunks, none partially embedded). `search_documents` with a claim's own opening phrase returned the failed document as the top hit in all three tries: Vector indexes (`cfd1052e…`, score 0.155), Persistence (`2c6b6b70…`, 0.128), Constraints (`dcd926c0…`, 0.239). So 33 are searchable now and 0 are unsearchable for lack of embeddings; what a `failed` document lacks is entity extraction (the extract jobs timed out). The old `falkordb-reference.md` versions the sync removed cannot be read, so their text was not compared.

S8a KB ingest of the 17 never-ingested claims (graph-dba, 2026-11-21): precondition read-only check found no `Document` (any version) sharing `textNormalizedHash` with any of the 17 claims (checked with `claim_hash` and the marker-stripped hash). Each file ran alone, `kb-sync -v --confirm coordination <path>` (no `--detach-finish`), a 3-token chat probe answering first (0.12–0.47 s), then waited until `processing` was 0 and ran `kb-finish --once --confirm coordination` (each reported `0 not complete`): `claude/analyst/review-techniques.md` (7 claims), `claude/architect/plan-authoring-techniques.md` (1), `claude/data-scientist/lm-studio-model-notes.md` (1), `claude/graph-dba/falkordb-quirks.md` (4; about 28 minutes of embedding/extraction), `claude/qa-engineer/qa-testing-techniques.md` (1), `claude/tdd-engineer/guard-testing-techniques.md` (2), `claude/teco/coordination-techniques.md` (1). The failed count stayed 33 after every file (no new failure); `ws:agent-team` is ready 533, failed 33, processing 0. Each of the 17 claims now has exactly 1 current `Document`, status `ready`. `kb-status`: 0 unmirrored, 17 `degraded` (the failed-but-embedded documents, informational); `drift --confirm coordination`: `Drift: 16 source(s)` (files edited since the load, left for go-live). `search_documents` with each claim's own opening wording found 16 of the 17 in the top 5: 15 under the 0.43 floor (scores 0.164–0.385) and `0dc7d197…` (the commit-grant claim) at 0.461, above the floor; `3537c199…` ("Findings get the scrutiny…") was not in the top 5 for a paraphrase nor for its literal opening (top hit 0.417), although it is `ready`, current, with 4 of 4 chunks embedded and current, so that is ranking, not indexing.

ledger `unknown`-status lint (tdd-engineer, 2026-11-21, S8a gate F1): `coordgraph lint` now errors (`unknown-status`) per unit whose parser-derived `statusToken` is `unknown` in a ledger whose header Status is not `archived` (header-less or odd Status counts as live; archived ledgers silent, the real tree's archived unknowns stay unreported); honours the PATH filter; `--count` stays backlog-only. Real tree: 0 errors, 0 warnings, rc 0. 7 tests, 8 mutants killed.

S9 rollout, cobb half (cobb, 2026-11-21, `claude/docs/plans/coordination-graph.md` S9 row): `teco.md`/`tico.md` rules, `agent-kb-retrieval` `kb-status` step, `agent-maintenance` §5 step 5 replaced, KB manifest frozen, `skills/coordination-graph/SKILL.md` brought current, `claude/AGENTS.md` project-settings hook class, and `claude/scripts/audit-team.sh` checks 13 (`coordgraph lint`), 14 (KB registry vs mirror) and 15 (hook registered on both events). Check 15 reports the hook unregistered by name until the stakeholder applies `docs/hook-settings.md`; the fail-safe rule (no hook report: run `coordgraph sync <file>`) stands until it is registered and S0 has run. Root `AGENTS.md` wording drafted, not applied (`claude/docs/plans/coordination-graph-s9-root-agents-draft.md`); the `tico` manual section and the `analyst` gate are still open.

S9 gate fixes (cobb, 2026-11-21, `## S9` of `claude/docs/reviews/coordination-graph-impl.md`): audit check 15 rewritten (it could never pass; now `claude/scripts/coordgraph-hook-registered.py` judges each event and the matcher, with a selftest whose positive control is the exact `docs/hook-settings.md` block), check 14 honours the `kb-status` exit code, `agent-kb-retrieval` step 5 cites the `cypher-mcp/.venv` fallback, the KB manifest is frozen for claim tracking only (`_dispositioned_raw_*` stays live), and a retired KB file needs its `##` headings emptied and `kb-sync`ed before it leaves `kb-files.txt` (read from `kbsync.py`, not live-tested; the engine has no cleanup for a file dropped from the registry first).

S10 acceptance defects D-1, D-2, D-4 (tdd-engineer, 2026-11-21, `claude/docs/test-reports/coordination-graph-report.md`): **deliberate deviation from plan §3.3 / AC-5 wording.** The sync header's `Hook: active|disabled` meant only "kill switch off" yet read as working same-moment freshness while no hook was registered (D-1). It now reads `Hook: registered|not registered|disabled` (`coordgraph/hookreg.py`, `drift.render_header` takes the state string): `disabled` (kill switch) wins; `registered` needs `.claude/settings.json`/`settings.local.json` to run `coordination-graph/hooks/post-tool-sync.sh` (command's executable, or the argument of `bash`/`sh`, resolved against the repo root with `$CLAUDE_PROJECT_DIR` expanded; `.bak`, `echo <path>`, another checkout and `..` detours do not count) for BOTH `PostToolUse` and `PostToolUseFailure` with a matcher covering Write, Edit and Bash (empty/`*`, an exact `|` name list, or a regex judged as a full match of each tool name; tokens unioned across entries and files); everything else, including unreadable or garbled settings, is `not registered`. A static reading only: it does not prove the hook fires in a given context. Own implementation, not an import of `claude/scripts/coordgraph-hook-registered.py` (`coordgraph` must not depend on `claude/scripts`); the anchored command match is that checker's open P2-1 finding (`## S9 Pass 2`), which is not changed here. D-2: `parse.source.decode` strips a leading UTF-8 BOM (hashes stay over the raw bytes; no `PARSER_VERSION` bump, no BOM exists in the corpus). D-4: a ledger with table-like lines (outside fences) but no pipe table with a Unit column gets the parser warning `no units parsed: ...`, and `lint` reports it as a `no-units` warning (not an error). D-3 (symlink hardening) deliberately not done; D-5/D-6 are not code. Tests: `tests/test_hookreg.py` (new), `test_ledger.py`, `test_lint.py`, `test_sync.py`, `test_sync_live.py`; offline 1155 -> 1202, live 88 -> 89. Doc strings updated: `coordination-graph/README.md` "Reading the sync header", `skills/coordination-graph/SKILL.md`, `claude/docs/manuals/agent-team.md`.

S10-fixes gate (tdd-engineer, 2026-11-21, `## S10 fixes` of `claude/docs/reviews/coordination-graph-impl.md`): M1 `hook_status` no longer says `registered` for `disableAllHooks: true` (local file over project file), a handler whose `type` is not `command` or that has an `if` filter, a relative command path, or a script that is missing or (direct form) not executable; name lists with spaces or commas (`Write | Edit | Bash`) are exact lists; user-scope and managed settings are named as unreadable in the README. M2 the registered-header live test is parametrized over all six header call sites (`drift`, `query`, `inflight`, `unit`, `planned`, `propose`); replacing `root` with `Path("/nonexistent")` at each of the four code sites (two in `commands_store.py`, `views._header`, `propose_cmd`) is now killed. Correction to the D-1 entry above and its commit message ("17/17 mutants killed"): that count did not cover the `root` plumbing, which survived at the `views` and `propose` sites until this change. M3 the golden table `tests/fixtures/hook-settings-cases.json` (57 cases: name, settings, optional settings_local, script, kill_switch, expected, why) is read by `tests/test_hookreg.py`; `hookreg` is the authority and the audit checker is aligned to it by `cobb`. M6 (other text reads keeping a BOM: `cli.py`, `migrate.py`, `lint.py`, `history.py`) not changed: no BOM exists in the corpus and `migrate` writes the text back. M4 (discovery slug collision) not addressed here.

Close-gate `## Live spec check` fixes (tdd-engineer, 2026-11-21): `rebuild --dry-run` accepts the `:Item` properties `milestoneOrder`/`milestoneApproved`/`milestoneStarted` the parser writes (FORMATS §7.3; `scopeKind`, written nowhere, dropped from the allowed `:Source` set), proved by a full-grammar backlog syncing to parity and passing the dry-run on an ephemeral container; `drift.check` (every view's header) now reads through `ro_query`, so a view on an absent key errors instead of materialising it; FORMATS §1 hash coverage, `Aliases:` token rule, DESIGN §5 exit table, S0 probe artifact, README `Hook:` bullet (cites DESIGN §4.6) and `docs/BACKLOG.md` header corrected.

## 2026-11-21 — Feature close: coordination graph in service except hook registration (GAL-F0002)

Delivered: the `coordgraph` mirror of ledgers, reviews, requirements, backlogs, agent plans and registered KB files; the shared `coordination` graph loaded with parity ok (S8a; 548 sources at the load) and the KB mirror seeded; the read views and `propose`; the shared-instance guard; the post-edit hook script and its `hook-sync`/`retry-failed` commands; the team rollout (S9: `teco`/`tico` prompts, skills, audit checks 13-15, root `AGENTS.md`, the user manual); acceptance (S10: `claude/docs/test-reports/coordination-graph-report.md`). Live spec written at the close: `docs/DESIGN.md`, `docs/FORMATS.md` (checked against the shipped code in `claude/docs/reviews/coordination-graph-impl.md`, `## Live spec check` and Pass 2: approve with suggestions). Reviews' `REVIEW.md` additions applied (7 lines, no prune, none narrowing). Open, carried in `docs/BACKLOG.md`: registering the hook (stakeholder), the S0 hook-firing probe, go-live, the unmeasured real-backend KB round trip, and the cross-initiative requirement (GAL-F0003), which is not built and stays open. Frozen with this entry: the coordination-graph requirements, plans, ledger, reviews and S10 test documents; the cross-initiative documents stay active.
