---
name: coordination-graph
description: >-
  How to read and write the coordination graph: the FalkorDB mirror of the repo's coordination
  files (teco ledgers, reviews, requirements, component BACKLOG.md files, agent kaizen plan.md,
  registered KB files), queried through the `coordgraph` CLI, plus the item format,
  delivery-heading and milestone conventions it depends on. Use when asked what is in flight,
  waiting on the stakeholder, blocked or planned for a component/agent, for a milestone proposal,
  for a ledger unit's status (slug, work-item ID or label); when resuming a coordination; after
  `search_documents` (kb-status); when writing or editing a backlog item, `Depends on`/`Touches`/
  `Milestone` field, a ledger `Item`/`Needs` column, or a history heading that delivers an item;
  or when a hook reports a mirror sync failure or none appears after a mirrored-file edit.
---

# coordination-graph — views, item format, conventions

The graph is a **read-only mirror**: files are the truth, a hook mirrors every edit, and the
views read the mirror. Never write to the graph; edit the file. Design:
`claude/docs/plans/coordination-graph.md` (+ `coordination-graph3.md`); component:
`coordination-graph/README.md`.

**Entry point**, from the repo root: `coordination-graph/coordgraph.sh <sub>`. Bare `coordgraph <sub>`
below abbreviates that. It needs the component venv (`coordination-graph/setup.sh`); with none it
exits 1 naming the fix, and the equivalent is `PYTHONPATH=coordination-graph
cypher-mcp/.venv/bin/python -m coordgraph <sub>`. **Shared instance:** a store-backed command needs
`--confirm coordination`, or the standing `COORDGRAPH_SHARED_CONFIRM=coordination` for routine ones
(views, `drift`, `sync`, `kb-status`); `bootstrap`, `rebuild`, `kb-reconcile` need the flag and are
the distiller's/`graph-dba`'s, not yours. Never export `COORDGRAPH_EPHEMERAL_PORT`.

## Status — read before trusting a recipe

Every subcommand below is built and the shared mirror is loaded. **The post-edit hook is built and registered
in `.claude/settings.json`** (`coordination-graph/docs/hook-settings.md`); if it is not, or is disabled,
nothing mirrors an edit for you, and the header's `Hook:` says so (`registered`, `not registered` or
`disabled`; only `registered` means settings register both events). The S0 probe (2026-11-21, mode `auto`) saw it fire for Write/Edit/Bash in the main session and in
foreground, background and worktree subagents; a failing Bash fires `PostToolUseFailure`; **a failed
Edit fires no event**. Other permission modes and teammate sessions are unverified
(`coordination-graph/docs/hook-settings.md` §6).

## Rules that never bend

1. **Relay the sync header.** Every read view prints one first (`Synced … · Hook: registered|not registered|disabled`,
   `Registry: …`, `Drift: none|…`). Pass it on with the answer; never strip it.
2. **A drift flag means quote the file.** A `⚠ drift` row or a `Drift:` source is stale in the
   graph: the file wins; read it, quote that row from the file, and say the graph lagged.
3. **Never call a mirrored-file edit complete on silence.** After you edit a mirrored file (ledger,
   backlog, review, requirements, `plan.md`, registered KB file): if the hook reports exit 2, the
   file is written but not mirrored — say so in the same turn, name the file and the printed
   reason, and do not call the update done (the hook retries; `coordgraph sync <file>` forces it).
   **If no hook report appears** — the hook is unregistered, or you are in a context where it may not
   fire — run `coordgraph sync <file>` yourself after the edit, and `coordgraph drift <file>` when
   unsure; a non-zero exit is reported the same way. Do this for every such edit until the hook is
   registered and your context is one S0 covered (unverified: modes other than `auto`, teammates). A
   failed Edit never triggers the hook, but it also wrote nothing.
4. **A proposal never takes effect without explicit stakeholder approval.** `propose` is a pure
   planner that writes nothing; relay it for approval. Only after the stakeholder approves each
   placement in conversation do you record `Milestone` on the items, by a normal file edit. An agent
   that *requests* a proposal (`teco`) relays it and never applies it.
5. **Never guess an identity or a unit.** Report `ambiguous`/`unknown`/`unminted` with the
   reason printed; do not pick a candidate.

## The commands

| Question | Recipe |
|---|---|
| What is in flight / waiting on the stakeholder / open "needs changes" / unpicked requirements | `coordgraph inflight [--waiting]` |
| One unit's status, owner, deliverable, question | `coordgraph unit <ledger-slug\|ID> [<unit>]` |
| What is planned for a component or agent | `coordgraph planned <component\|agent> [--all]` |
| Propose milestones for a backlog | `coordgraph propose <component\|agent> [--per-milestone N]` (read-only) |
| Anything else (ad-hoc Cypher) | `coordgraph query '<cypher>' [--covers PATH…]` — not raw `mcp__cypher__query`, so the header and drift flag still apply |
| Is the mirror current / force a sync | `coordgraph drift [PATH…]` (exit 1 = drift); `coordgraph sync <PATH…>\|--changed\|--all` |
| Backlog format check | `coordgraph lint [--count] [PATH…]` (exit 1 on errors) |
| Fact tokens in use (reuse, never reinvent) | `coordgraph facts [--like X]` |
| One-time reformat of a backlog | `coordgraph migrate --dry-run [PATH…]`; `--apply PATH --titles-reviewed` (S4 only: `--dry-run` that file first; the owner fixes flagged titles before `--apply`; one file per run and per commit, stage `.work-items/registry.jsonl` with it; mints permanent IDs via `wi new`) |
| Delivery-heading match rate | `coordgraph history-coverage [-v]` |
| KB staleness of a search hit | `coordgraph kb-status --ids <documentIds>` (see `skills/agent-kb-retrieval/SKILL.md`) |

`unit` and `inflight` accept a feature ID, alias or task ID as well as a ledger slug. The ledger
slug is the file name minus `-coordination` (`connected-code-graph`, not
`connected-code-graph-coordination`). Unit tokens, labels, `~n` and `ambiguous` handling, the `Item`/
`Needs` columns and the degraded registry mode: `references/ids-and-units.md`.

## Writing a backlog item (the format `lint` and the mirror parse)

Each item is its own `###` heading and a field block, then free prose. Heading
`### <ID> — <title>`; `<ID>` is a `wi new` ID (never hand-numbered), or an existing `K-NNN`/`C-NNN`.

```markdown
<!-- IDs in this skill's examples are illustrative -->
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
  beside the slash does **not** deliver the earlier IDs.
- **`started`.** The first time any item of a milestone turns `🟡`, append `— started YYYY-MM-DD`
  to the `Milestone` field of **every** open item of that milestone, in the same edit.

## Ledger edits

A ledger edit is done when rule 3 is satisfied (no failure reported, or your own `sync` exited 0).
Optional ledger columns `Item` (the unit's task ID) and `Needs` (comma-separated IDs,
`<ledger-slug>:<unit>`, a local unit label, or `none`); `Status` cells use teco's seven words
(`queued`, `in-flight`, `delivered`, `gated`, `accepted`, `abandoned`, `paused`) — anything else
reads `unknown`, which `lint` reports as an error in a non-archived ledger: fix the cell.
