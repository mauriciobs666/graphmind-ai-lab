---
name: coordination-graph
description: >-
  How to read and write the coordination graph: the FalkorDB mirror of the repo's coordination
  files (teco ledgers, reviews, requirements, component BACKLOG.md files, agent kaizen plan.md),
  queried through the `coordgraph` CLI, plus the item format, delivery-heading and milestone
  conventions the mirror depends on. Use when asked what is in flight, waiting on the stakeholder,
  blocked or planned for a component/agent, for a milestone proposal, for the status of a ledger
  unit (by slug, work-item ID or label); when resuming a coordination; when writing or editing a
  backlog item, `Depends on`/`Touches`/`Milestone` field, a ledger `Item`/`Needs` column, or a
  history heading that delivers an item; or when a hook reports a mirror sync failure.
---

# coordination-graph — views, item format, conventions

The graph is a **read-only mirror**: files are the truth, a hook mirrors every edit, and the
views read the mirror. Never write to the graph; edit the file. Live design:
`claude/docs/plans/coordination-graph.md` (+ `-graph.md`, `coordination-graph3.md`).

**Entry point.** `coordination-graph/coordgraph.sh <sub>` (run from anywhere). Bare
`coordgraph <sub>` below abbreviates exactly that path. It needs the component venv
(`coordination-graph/setup.sh`); with none it exits 1 naming the fix.

## Build status — read before trusting a recipe

- **Built (offline, no store):** `lint`, `migrate`, `facts`, `history-coverage`.
- **Planned, not built:** the store and the views (S3 `bootstrap`/`sync`/`rebuild`/`drift`; S7a
  `inflight`/`unit`/`planned`/`query`; S7b `propose`), KB freshness (S6 `kb-*`), the post-edit hook
  (S8). Until S7a lands, answer "what is in flight / status of unit X" by reading the ledger file;
  this skill's rules about the sync header, drift and hook failures take effect when the views ship.
  When a planned subcommand errors as unknown, that is why, not a fault.

## Rules that never bend

1. **Relay the sync header.** Every read view prints one first (`Synced … · Hook: active|disabled`,
   `Drift: none|…`, `Registry: …`). Pass it on with the answer; never strip it.
2. **A drift flag means quote the file.** A `⚠ drift` row or a `Drift:` source is stale in the
   graph: the file wins; read it and quote that row from the file, and say the graph lagged.
3. **A hook exit 2 means the edit is not done.** The file is written but its mirror sync failed or
   is pending: say so in the same turn, name the file and the reason the hook printed, do not call
   the update complete. The hook retries by itself; `coordgraph sync <file>` forces it. *(Hook is
   planned, S8; the rule binds from the day it is registered.)*
4. **A proposal never takes effect without explicit stakeholder approval.** `propose` writes
   nothing; relay it for approval. Only after the stakeholder approves each placement in
   conversation do you record `Milestone` on the items, by a normal file edit. An agent that
   *requests* a proposal (`teco`) relays it and never applies it.
5. **Never guess an identity or a unit.** Report `ambiguous`/`unknown`/`unminted` with the
   reason printed; do not pick a candidate.

## The commands

| Question | Recipe | State |
|---|---|---|
| What is in flight / waiting on the stakeholder / open "needs changes" / unpicked requirements | `coordgraph inflight [--waiting]` | planned (S7a) |
| One unit's status, owner, deliverable, question | `coordgraph unit <ledger-slug\|ID> [<unit>]` | planned (S7a) |
| What is planned for a component or agent | `coordgraph planned <component\|agent> [--all]` | planned (S7a) |
| Propose milestones for a backlog | `coordgraph propose <component\|agent> [--per-milestone N]` | planned (S7b) |
| Anything else (ad-hoc Cypher) | `coordgraph query '<cypher>' [--covers PATH…]` — not raw `mcp__cypher__query`, so the sync header and drift flag still apply | planned (S7a) |
| Is the mirror current / force a sync | `coordgraph drift [PATH…]`, `coordgraph sync <PATH…>\|--changed\|--all`, `rebuild` | planned (S3) |
| Backlog format check | `coordgraph lint [--count] [PATH…]` (exit 1 on errors) | built |
| Fact tokens in use (reuse, never reinvent) | `coordgraph facts [--like X]` | built |
| One-time reformat of a backlog | `coordgraph migrate --dry-run [PATH…]`; `--apply PATH [--titles-reviewed]` (one file per run; mints IDs via `wi new`, refuses while `--dry-run`-flagged titles are unacknowledged — IDs are permanent) | built |
| Delivery-heading match rate | `coordgraph history-coverage [-v]` | built |
| KB mirror staleness of a search hit | `coordgraph kb-status --ids <documentIds>` | planned (S6) |

`unit` and `inflight` accept a feature ID, alias or task ID as well as a ledger slug. The ledger
slug is the file name minus `-coordination` (`connected-code-graph`, not
`connected-code-graph-coordination`). Unit tokens, labels, `~n` and `ambiguous` handling, the `Item`/
`Needs` columns and the degraded registry mode: `references/ids-and-units.md`.

## Writing a backlog item (the format `lint` and the mirror parse)

Each item is its own `###` heading and a field block, then free prose. Heading
`### <ID> — <title>`; `<ID>` is a `wi new` ID (never hand-numbered), or an existing `K-NNN`/`C-NNN`.

```markdown
### GAL-I02-K0042 — Example item title

- **Status:** 🔵 proposed — filed 2026-10-15
- **Priority:** medium
- **Depends on:** GAL-F0043, falkor-chat:K-070
- **Touches:** file:falkor-chat/server/*.py; graph:ws:test; fact:falkor-chat/docs/QUERIES.md#ingest-job-order (changes)
- **Milestone:** M7 — order 2 — approved 2026-10-21
```

- `Status`: `🔵 proposed` · `🟡 in-progress` · `⚪ deferred` · `⚪ rejected` · `✅ delivered` (keyed on
  the word, emoji must agree; `✅` is transient — a delivered item leaves the backlog, its record is
  `HISTORY.md`). There is no `blocked`: it is derived from `Depends on`.
- `Priority`: `high|medium|low`. `Depends on`: comma-separated refs or `none`. `Touches`:
  `;`-separated `file:` / `graph:` / `fact:<token> (changes|relies)` (mode mandatory) or `none`.
- **Absent field = not recorded** (the item is not plannable, the missing fact is named);
  **`none` = recorded empty.** Never backfill a fact you did not establish.
- **Parking-lot / ideas bullets are items too**: a `###` item under that section, default
  `🔵 proposed`; a settled judgment is `⚪ rejected — record, not work`. No bullet's text is dropped.
- **`Milestone` is fixed.** Written only when the stakeholder approves a proposal, or by the
  reformat converting a recorded milestone table; every recorded value is never re-placed by a tool.
- Cite a legacy item as `<home>:K-NNN` (`falkor-chat:K-070`), **never bare `K-NNN`** — a bare one is
  ambiguous across homes (`wi` exit 3); prefer the scheme ID once minted. A scheme ID is never
  prefixed with a home. Run `wi flush` per `skills/work-items/SKILL.md` after any allocation.

Full grammar (heading pattern, status words, fact tokens, milestone map markers), the delivery-heading
convention and the milestone `started` convention: `references/backlog-format.md`.

## Two conventions the planner depends on

- **Deliver with a heading.** When an item is delivered, its `HISTORY.md` (agent plans:
  `kaizen/history.md`) heading carries `<ID> delivered` or `<ID> closed`; several IDs are
  **slash-joined with no spaces**: `K-070/K-071 delivered`. A comma list, `and`, `+` or a space
  beside the slash does **not** deliver the earlier IDs. Or drop the dependent's ref.
- **`started`.** The first time any item of a milestone turns `🟡`, append `— started YYYY-MM-DD`
  to the `Milestone` field of **every** open item of that milestone, in the same edit.

## Ledger edits

A ledger edit is only done when the hook reports no failure (rule 3). Optional ledger columns
`Item` (the unit's task ID) and `Needs` (comma-separated IDs, `<ledger-slug>:<unit>`, a local unit
label, or `none`); `Status` cells use teco's seven words (`queued`, `in-flight`, `delivered`,
`gated`, `accepted`, `abandoned`, `paused`) — anything else reads `unknown` and is a cell to fix.
