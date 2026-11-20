# Change History — coordination-graph

> Dated log of actual changes to the `coordination-graph` component. Most recent first.

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
