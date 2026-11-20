# Change History — coordination-graph

> Dated log of actual changes to the `coordination-graph` component. Most recent first.

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
