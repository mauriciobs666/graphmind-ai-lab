# Change History — coordination-graph

> Dated log of actual changes to the `coordination-graph` component. Most recent first.

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

Reading: today's history files match few deliveries, as the plan expected (P2-M4). Correctness
rests on the delivery-heading convention (`<ID> delivered` / `<ID> closed`, slash-joined
siblings) and on `lint` flagging a `Depends on` ref that is open nowhere with no delivery
heading; existing deliveries are not backfilled. An unmatched ID here is an ID *mentioned*, not
necessarily one that was delivered.

Verified offline: the full suite from `coordination-graph/` (counts in the S1e hand-off); the
reformat applied only to scratch copies with their own registry, never to the repo's
`.work-items/`.
