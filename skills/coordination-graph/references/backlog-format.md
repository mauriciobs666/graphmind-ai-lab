# Backlog format, delivery headings, milestones, fact tokens

Verified 2026-11-20 against `coordination-graph/coordgraph/{parse/backlog.py,history.py,lint.py}`
as built (S1/S1e) and `claude/docs/plans/coordination-graph3.md` §3.6. The code wins over this text.

## Item heading

`^### (?P<id>[A-Z]-\d{3,}|[A-Z]{2,6}(?:-(?:[IST]\d{2,}|[FDK]\d{4,}))+) — (?P<title>.+)$`
(em dash, one space each side), outside code fences. Examples that match: `K-004`, `C-310`,
`GAL-F0042`, `GAL-I02-K0001`, `GAL-I03-F0012-S01-T01`. Not items: `GAL-F42`, `K-04`, `GAL-F0042-FR2`
(`FR`/`AC` are citations). Shape only: canonical form (widths, parent matrix) is `lint`'s call via
`wilib.rules` (`malformed-id`). A `###` heading with a `Status` field but no recognised ID is an
*id-less item* (positional ref, warning) that exists only until `migrate --apply` mints it an ID.
A duplicate ID in one file is a parse error; the same scheme ID headed in two backlog files is a
`duplicate-ref` error (a `wi move` may be in progress). Nothing else under a `###` is an item
(a "Cross-references" section is not).

## Field block

Consecutive `- **Label:** value` lines directly under the heading (one blank line allowed);
continuation lines indented two spaces; unknown labels are kept, not parsed. Canonical token first,
free text after ` — `.

| Field | Value |
|---|---|
| `Status` | `🔵 proposed` · `🟡 in-progress` · `⚪ deferred` · `⚪ rejected` · `✅ delivered` |
| `Priority` | `high` · `medium` · `low` [` — text`] |
| `Depends on` | comma-separated refs, or `none`. Stored verbatim, never shape-validated at parse |
| `Touches` | `;`-separated `file:<path or glob>` · `graph:<key>` · `fact:<token> (changes\|relies)`, or `none` |
| `Milestone` | `<M-token>` [` — order <n>`] [` — approved YYYY-MM-DD`] [` — started YYYY-MM-DD`] |

Absent = not recorded; `none` = recorded empty. Item summary tables are not parsed (an optional
generated index goes between `<!-- coordgraph:index -->` markers). A milestone map table is wrapped
in `<!-- coordgraph:milestone-map -->` … `<!-- /coordgraph:milestone-map -->`; `lint` checks every
open item it names carries that row's milestone and the reverse. It is otherwise never parsed.

## References in `Depends on`

Scheme ID (`GAL-F0043`) as written, never home-prefixed; legacy as `<home>:K-NNN`. A bare `K-NNN`
parses as "same backlog" but is ambiguous to `wi`; do not write it. A ref is satisfied when the item
is open in some backlog (then it is a blocker) or a delivery-shaped history heading names it.
`lint` errors `unresolvable-dependency` on a ref that is neither: fix the ID, drop the ref, or add the
delivery heading. (Today the CLI's `lint` compares IDs as written; the registry-backed join, so a
dependency written under an old name matches a renamed item, is resolved at read time by the views.)

## Delivery-heading convention

A history file is `<component>/docs/HISTORY.md` (root `docs/HISTORY.md` for `cpg`,
`claude/<agent>/kaizen/history.md` for an agent). A heading **delivers** an ID when:

- it is a Markdown heading of level 2-4 (`##`-`####`), outside a code fence, and
- the ID, optionally followed by `/`-joined sibling IDs, is **immediately followed by whitespace and
  one verb of** `closed` · `delivered` · `fixed` · `shipped` · `done` (case-insensitive; not a hyphenated word like `fixed-up`, nor `done criteria`/`checklist`), or
- the heading contains `✅` and the ID anywhere.

Write: `## 2026-11-20 — GAL-F0042 delivered: <title>`; `### K-070/K-071 delivered`;
`## … — falkor-chat:K-065/GAL-F0043 closed`. Scheme IDs and legacy IDs both count; mixed
slash-joined runs are fine. **Strict, by design:** only the token run directly before the verb
counts. `K-070, K-071 delivered`, `K-070,K-071 delivered`, `K-070 and K-071 delivered`,
`K-070 + K-071 delivered`, `K-070 / K-071 delivered` deliver **only** the last ID (or none — a comma
list matches nothing); the earlier IDs silently stay undelivered. `K-070 (part 1) delivered` and
`K-067: … (mechanism delivered; proof outstanding)` deliver nothing, and a body line never counts.
A scheme ID is searched in every history file; a legacy ID only in its owning backlog's. Check a
change with `coordgraph history-coverage -v` (match rate per history file, unmatched IDs listed).
Existing deliveries are not backfilled; fix one when `lint` first flags a dependency on it.

## Milestone `started` convention

When the first item of milestone `Mx` turns `🟡`, append `— started YYYY-MM-DD` to `Milestone` on
**every** open item of `Mx` in the same edit. The marker survives delivery, so a half-delivered
milestone still counts as started (`propose`, never inserts a milestone ahead of a started one).
`lint` errors `milestone-started` if items of one milestone disagree on `started`, or a `🟡` item's
milestone lacks it.

## Fact tokens

`<repo-root path>#<symbol-or-anchor>` (`falkor-chat/server/falkorchat/repository.py#list_documents`,
`falkor-chat/docs/QUERIES.md#ingest-job-order`) or `mcp:<tool>.<field>`; compared lower-cased. Mode is
mandatory: `(changes)` if the item alters it, `(relies)` if it only depends on it. Two items overlap
(and `propose`, never schedules them in parallel) when they share a path/glob (conservatively), a `graph:` key,
or a fact token unless both sides are `(relies)`. Run `coordgraph facts --like <x>` first and reuse an
existing token; never invent a spelling variant. Touching an item means a substantive content edit;
filling all three facts (`Priority`, `Depends on`, `Touches`) on a new item is the duty, backfilling
old ones is not.

## `lint` codes (exit 1 on any error)

`status`, `delivered-in-backlog`, `duplicate-id`, `duplicate-ref`, `malformed-id`,
`unresolvable-dependency`, `milestone-started`, `milestone-map` (errors); `wilib-unavailable` (error
when the work-items scripts are expected, i.e. the canonical-ID checks did NOT run — a pass is then
not a pass; a warning only where they are absent); `parse` (warning: no `legacy_paths` scope, empty
`Depends on` token, a `Touches` entry without prefix or mode, …).
