# Identity, ledger columns, unit lookup, degraded mode

Verified 2026-11-20 against `coordination-graph/coordgraph/{identity.py,shapes.py,parse/ledger.py}`
and `claude/docs/plans/coordination-graph3.md` §3.1-§3.4. The lookup rules below are implemented in `identity.select_units` / `resolve_need`, which the
`unit`/`inflight` views call. The code wins over this text.

## IDs are stored as written and resolved at read time

A file holds whatever ID was written; the registry (`.work-items/`, read through `wilib`) is
consulted only when a view reads, so a `wi move` renames nothing in files and old and new names still
join. Token shapes: bare legacy `K-070` (same backlog as the referrer); `home:K-070` (qualified
legacy); scheme `GAL-F0042`; in a `Needs` cell also `<ledger-slug>:<unit>` or a bare local unit label.
Resolution states: `resolved | unminted | ambiguous | unknown | retired | unavailable`. A bare legacy
ID is ambiguous when several homes declare it: relay the candidate list `wi` prints, never pick one.
An `unminted` legacy name prints its `wi mint-legacy <home>:K-NNN --type …` hint; minting is `wi`'s job
(`skills/work-items/SKILL.md`), not this skill's.

## Ledger columns

Mapped by header name, optional, absent = not recorded:

- **`Item`** — the unit's task ID (first token, as written).
- **`Needs`** — comma-separated tokens or `none`: a task or feature ID/alias; `<ledger-slug>:<unit>`;
  or a unit label of the same ledger (`U3`). An empty token between commas is a malformed cell
  (`needs not recorded`, a warning): fix the cell. A need never copies another ledger's status.
  Need is **met** when the task's unit is `accepted`, or (feature ID) every ledger carrying it is
  archived or it has at least one unit and all its units are `accepted` (an empty live ledger is unmet); `unknown`-status units are *unresolvable*, never met.
  A feature ID matches ledgers by their header `Item:`.

`Status` cells use teco's seven words: `queued`, `in-flight`, `delivered`, `gated`, `accepted`,
`abandoned`, `paused`; closed aliases `relayed`→`paused`, `done`→`delivered`, `superseded`→`abandoned`
(in every ledger; write teco's seven words, the aliases only absorb old cells). Anything else is `unknown` with a warning: a cell to fix, not a token to add.

## `unit <ledger-slug|ID> [<unit>]`

1. The first argument is a ledger slug (file name minus `-coordination`) **or** a feature/alias/task
   ID; an ID expands to every name it is known by.
2. The unit token matches `unitKey = <t>` **or** `unitLabel = <t>`, case-sensitive. A label is the
   Unit cell up to the first whitespace or `(`, trailing `:,;` trimmed (`U10b Note: …`→`U10b`,
   `U1(a)`→`U1`).
3. **Exact-key precedence:** if any row's full `unitKey` equals the token, only those rows are the
   answer (every repeat of that key); only when none does, all label matches are. So `S6` wins over
   `S6 gate`, and `S6 gate` stays reachable by its full text.
4. `<label>~<n>` (end-anchored) selects the n-th repeat of a repeated label (`U32~2`).
5. More than one row left → **`ambiguous`**: print all rows with paths and keys, ask or cite by task
   ID; never choose. Prose-led labels (`Pass`, `Stage`) are expected to be ambiguous.
6. Two ledgers sharing a slug are both listed with paths; a local `U3` in a `Needs` cell anchors on
   the referring ledger's file, not on the slug.

## Registry unavailable (degraded mode)

When `wilib` is not importable, `.work-items/` is absent or the registry is unreadable, the sync
header reads `Registry: unavailable — identities compared as written` (healthy:
`Registry: <n> items · last event <date> · <m> unflushed journal events`). Views still work, but
IDs are compared as written: a dependency or `Needs` token written under an old name will not join a
renamed item. Say so when relaying; treat a miss as "not found under that spelling", not "does not
exist", and cite the form the file uses. Sync (the hook) never reads the registry, so a broken
registry cannot stop a file from being mirrored. `COORDGRAPH_WI_ROOT` overrides where `wilib` is
loaded from.
