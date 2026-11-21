# coordination-graph — Formats

> **Status:** active · **Owner:** `coder` · **Tracks:** —

> **How to read this document.** It is part of coordination-graph's live spec and states the data
> formats **as they are now**: the source-file grammars the parsers read, the graph the mirror
> writes, the key rules, the KB mapping, and the hook's state files. Behavior (sync, views, CLI,
> guards, hook, KB engine) is in `coordination-graph/docs/DESIGN.md`, read first. Change history is
> in `coordination-graph/docs/HISTORY.md`. Section numbers are stable and cited from the code's
> and the context files' pointers.

## Index

| § | Topic |
|---|---|
| 1 | Header block |
| 2 | Ledger |
| 3 | Review |
| 4 | Requirement |
| 5 | Backlog |
| 6 | KB file and KB registry |
| 7 | Graph: labels, properties, edges, indexes |
| 8 | Key and indexed-value hygiene |
| 9 | KB mapping nodes |
| 10 | Hook state files |

## 1. Header block

Parsed by `coordgraph/parse/header.py` for every `:Source` kind (stdlib only, over already-read
text; it re-implements `wi`'s own header reader and a test pins the two together).

- **Shape.** The line starting `> **Status:**` immediately under the H1 (one blank line allowed
  between them). Fields are ` · `-separated `**Label:** value` parts. Every label other than the
  ones below is tolerated and ignored. A file with no such line has no header block.
- **`docStatus`.** The canonical token is the first thing after `Status:`; free text after it is
  ignored. The closed set is `Interviewing`, `Ready for design`, `active`, `superseded`,
  `archived`, matched case-insensitively; a value outside the set keeps its leading word as
  written, so a mistyped status is visible in the graph.
- **`owner`, `tracks`.** The raw value of `Owner:` / `Tracks:`.
- **`itemId`.** The first backticked token of `Item:` (else the bare value; `—` and empty mean
  none), as written. An empty token or one over 512 UTF-8 bytes is not stored and adds the source
  warning `header Item: unusable (empty or over 512 bytes); itemId not stored`.
- **`aliases`.** The backticked tokens of `Aliases:`, else the whole value as one token (`A-1, B-2`
  unbackticked is the single alias `A-1, B-2`; `—` and empty mean none); `[]` for a header block without `Aliases:`; absent when
  the file has no header block.
- **Decoding.** UTF-8 with replacement, a leading BOM removed (so the header sits under the title).
  `sha256` and `bytes` of a source or KB file are over its **raw bytes**; `itemHash`, `rowHash`,
  `claim_hash` and `sectionHash` are over the decoded text (normalised cells for `rowHash`, normalised
  text for `claim_hash`, the stripped body for `sectionHash`).

## 2. Ledger

A header block plus pipe tables (outside fenced code). A table is a **unit table** when a column
header begins with `Unit` (case-insensitive, markup stripped); other tables are ignored. Columns
are mapped by header name, first match wins:

| Role | Header |
|---|---|
| unit | starts with `Unit` |
| owner | `Owner`; else `Agent` or `Agent (...)` when there is no `Owner` column |
| agentId | `Agent ID`, `Agent-ID`, `AgentID`, `Agent IDs` |
| status | `Status` |
| deliverable | starts with `Deliverable` |
| gateVerdict | starts with `Gate` |
| cost | `Cost` |
| itemId | `Item` |
| needs | `Needs` |

- **Rows.** An all-blank row is skipped. Cells beyond the header width add a warning (a stray pipe
  shifts later columns); missing trailing cells read empty. `tableIdx` and `rowIdx` are 1-based
  (table among every table of the file, row within its table).
- **`unitKey`.** The Unit cell with `*`, backticks and `~~` removed, cut at the first ` — `. A
  repeat within a ledger takes the suffix `~2`, `~3`, ... on the **key** (warning
  `duplicate unit <k> → <k>~n`); `unitKey` itself never carries the suffix.
- **`unitLabel`.** `unitKey` up to the first whitespace or `(`, trailing `:` `,` `;` trimmed; the
  whole key when that is empty (warning `unit label fell back to the key`).
- **`statusToken`.** One pure function over the Status cell: markup and pictographs stripped,
  case-folded, then the leading word (or two-word hyphen join, e.g. `in flight` to `in-flight`)
  matched against the vocabulary `queued`, `in-flight`, `delivered`, `gated`, `accepted`,
  `abandoned`, `paused`; the closed alias table `relayed` to `paused`, `done` to `delivered`,
  `superseded` to `abandoned`; anything else, and a ledger with no Status column, is `unknown`
  (warning `unit <k>: unknown status ...` / `no status`; a table without the column also warns
  `table <n>: no Status column`). The raw cell is kept as `statusText`.
- **Item cell.** Its first backticked token (else the value), normalised and stored as `itemId`; an
  empty value or `—` is not recorded; a token over 512 bytes adds a warning and is not stored.
- **Needs cell.** Empty or `—`: not recorded. `none`: `[]`. Otherwise comma-separated tokens kept
  raw, enclosing backticks dropped. An empty token between commas leaves `needs` absent and sets
  `needsError` (plus a source warning unless the ledger is archived).
- **Paused rows.** `question` and `askedOn` are read from the Deliverable cell: the quoted text (or
  the text before ` — relayed`, `⏸` dropped) and the date after `relayed` (else the first date in
  the cell); a missing date adds a warning.
- **Citations.** `cites` is the sorted set of backticked repo paths (`dir/.../file.ext`, anchor cut)
  in Deliverable and Gate. A backticked path under `docs/reviews/*.md` is a cited review, resolved
  against the covered review paths as written, else under the ledger's component directory; one that
  resolves to nothing warns `cited review ... not found`. `citedReviews` is the set of those; a
  cited review is also in `resolvedReviews` when the unit's `statusToken` is `accepted` or
  `abandoned`.
- **`rowHash`.** sha256 of the JSON of the row's whitespace-normalised cells.
- **Archived ledger** (`docStatus` exactly `archived`). Units are **minimal**: `key`, `unitKey`,
  `unitLabel`, `statusToken`, `ledgerPath`, `slug`, `component`, `tableIdx`, `rowIdx` and, when
  present, `itemId`. No cell text, `needs` or `rowHash` is mirrored; the reviews lists are still
  derived.
- **No units.** A ledger with table-like lines outside fences but no unit table carries the warning
  `no units parsed: <n> table-like line(s) but no pipe table with a Unit column (a table needs a
  `|---|` separator row under its header)`.

## 3. Review

Header block plus the latest verdict. The latest verdict is the **last statement in document
order** of the last `## Pass N` section, or of the whole file when it has none. A statement is the
word `verdict` (optionally prefixed `updated`/`overall`/`current`, optionally followed by a `(Pass
N)`-style aside) followed by a verdict phrase; the phrase may sit on the next line under a
`## Verdict` heading. Inside a pass section a statement marked with another pass is ignored; a
statement right after an opening quotation mark is another document's and is ignored.

- `latestVerdict`: the phrase's sentence, up to 300 characters; absent with the warning `no verdict
  found`.
- `latestPass`: the last `## Pass N` number; else the marker of the verdict statement; else 1.
- `verdictClass`: the earliest phrase in the verdict text, normalised to `needs-changes`, `approve`,
  `approve-with-suggestions`, `reject`, or `unknown`.

## 4. Requirement

The header block only; no children.

## 5. Backlog

A component `BACKLOG.md` or an agent `kaizen/plan.md`.

- **Item.** A `###` heading outside fenced code that either matches
  `### <ID> — <title>` (an em dash, U+2014) or is followed by a field block containing `Status`.
  `<ID>` is a legacy ID (`[A-Z]-\d{3,}`) or a scheme ID (`[A-Z]{2,6}` then one or more
  `-<I|S|T>\d{2,}` / `-<F|D|K>\d{4,}` segments). Any other `###` heading is a section heading, not an item.
  A `###` heading without a recognised ID but with a `Status` field is an **id-less item**: `ref`
  `<scope>:@<ord>`, no `itemId`, plus a warning.
- **Field block.** Consecutive `- **Label:** value` lines under the heading (one blank line allowed
  before the first); an indented non-blank line continues the previous field, joined by one
  space.
- **`ref`.** A scheme heading: the ID as written. A legacy heading: `<scope>:<ID>`; `<scope>` is the
  backlog's scope, or `unscoped`. `itemType` is the last scheme segment's letter, absent for legacy.
  A duplicate `ref` in one file fails the whole file (`BacklogParseError`).
- **Scope.** From `.work-items/project.json` `legacy_paths` (pattern to home; `*` is one path
  segment captured as `$1`); `{}` when the file or key is absent. A backlog no pattern matches has
  no `scope` and the warning `no legacy_paths entry gives this backlog a scope`.
- **`Status`.** `statusMarker` is the text before the first ` — `; the word is the first of
  `proposed`, `in-progress`, `deferred`, `rejected`, `delivered` found in it (the emoji, if any,
  must agree: `🔵` proposed, `🟡` in-progress, `⚪` deferred/rejected, `✅` delivered). `open` is
  false only for `delivered`; an unrecognised word is `open` true with a warning.
- **`Priority`.** The leading word before ` — ` or `(`: `high`, `medium`, `low`; else a warning.
- **`Depends on`.** Comma-split raw tokens, trimmed; `none` is `[]`; absent means not recorded.
  Tokens are stored as written, never qualified at parse.
- **`Touches`.** `;`-split entries; `none` is `[]`. Each entry should start `file:`, `graph:` or
  `fact:`; a `fact:` entry ends in `(changes)` or `(relies)`; otherwise a warning.
- **`Milestone`.** ` — `-separated: the token, then optional `order <n>`, `approved <date>`,
  `started <date>`; an unrecognised part warns.
- **`itemHash`.** sha256 of the item's lines (heading to the next heading of level 3 or above),
  right-trimmed and joined, stripped.
- **`facts`.** The `fact:` tokens of every `Touches` list are what `coordgraph facts` lists.

## 6. KB file and KB registry

- **Registry.** `coordination-graph/kb-files.txt`: repo-relative paths, one per line; blank lines
  and `#` lines skipped. A file is mirrored only when listed.
- **Sections.** Each `##` heading (outside fences) starts a section; text before the first `##` is
  not mirrored. `headingKey` is the lower-cased heading with punctuation removed and whitespace runs
  turned into `-`; a repeated heading takes `~n` and warns.
- **Modes.** `whole` (default): one claim per section, the body (an empty body warns `empty
  section`). `bullets`: set by `<!-- kb-split: bullets -->` on the heading line or alone on the
  first non-blank line under it; one claim per top-level bullet (`-`, `*`, `+`, `1.`, `1)`) with
  its continuation lines; non-bullet lines are not mirrored and warn. An unknown mode warns and
  reads as `whole`.
- **Claim.** `text` (the whole section body, or the bullet with its marker), `claim_hash` =
  sha256 of the text normalised the way falkor-chat does (whitespace collapsed, stripped,
  case-folded; comparable to `Document.textNormalizedHash`), `key` `whole` or `b-<12 hex of the
  hash>`. `sectionHash` is sha256 of the stripped body.
- **Document title.** The heading (`whole`), or `<file stem> — <bullet lead>` for a bullets claim,
  the bullet lead cut so the title stays within 200 characters.
- **Duplicates.** Two claims anywhere in the registry with the same normalised text are duplicates;
  the engine refuses to ingest either section.

## 7. Graph: labels, properties, edges, indexes

One graph, key `coordination` by default. A property whose value is absent is **not stored**
(absent is distinct from empty). `parse` (the parsers) decides every value; the store adds only
`syncedAt` and drops `None`s.

### 7.1 `:Source` and its kind label

Every covered file is one `:Source:<Kind>` node, `<Kind>` one of `Ledger`, `Backlog`, `Review`,
`Requirement`.

| Property | Meaning |
|---|---|
| `path` | repo-relative, UNIQUE |
| `kind` | `ledger` \| `backlog` \| `review` \| `requirement` |
| `component`, `slug` | per `coordination-graph/docs/DESIGN.md` §2 |
| `sha256`, `bytes` | of the raw file bytes |
| `parserVersion` | the `PARSER_VERSION` that wrote it |
| `warnings` | list of parser warnings, `[]` for a clean parse |
| `rowCount` | children written; `0` for a review or requirement |
| `syncedAt` | epoch ms of the write |
| `docStatus`, `owner`, `tracks`, `itemId`, `aliases` | header fields (§1), when recorded |

Ledger adds `citedReviews`, `resolvedReviews`. Review adds `latestVerdict` (when found),
`latestPass`, `verdictClass`. Backlog adds `scope` (when it has one). `mtimeNs` is a permitted
property that no shipped command writes.

### 7.2 `:Unit` (ledger child)

`(:Ledger)-[:HAS_UNIT]->(:Unit)`. Always: `key` (UNIQUE), `unitKey`, `unitLabel`, `statusToken`,
`ledgerPath`, `slug`, `component`, `tableIdx`, `rowIdx`; `itemId` when the Item cell has a token.
Non-archived ledgers add `unit` (the raw Unit cell), `cites`, `rowHash` and, when recorded,
`statusText`, `owner`, `agentId`, `deliverable`, `gateVerdict`, `cost`, `needs`, `needsError`,
`question`, `askedOn`.

### 7.3 `:Item` (backlog child)

`(:Backlog)-[:HAS_ITEM]->(:Item)`. Always: `ref` (UNIQUE), `backlogPath`, `ord` (1-based), `title`,
`open`, `itemHash`. When recorded: `itemId`, `scope`, `itemType`, `statusMarker`, `priority`,
`dependsOn`, `touches`, `milestone`, `milestoneOrder`, `milestoneApproved`, `milestoneStarted`.
Only the properties named here and in §7.1/§7.2 are written.

### 7.4 `:SyncState`

One node `{id: 'mirror'}`: `schemaVersion` (4, written by `bootstrap`), and `lastRebuildAt` (epoch
ms) and `parserVersion`, written by `rebuild`.

### 7.5 Indexes and constraints

Created by `bootstrap`, each UNIQUE constraint preceded by its supporting range index:

| Label | Property | Kind |
|---|---|---|
| `Source` | `path` | UNIQUE |
| `Unit` | `key` | UNIQUE |
| `Item` | `ref` | UNIQUE |
| `KbFile` | `path` | UNIQUE |
| `KbSection` | `key` | UNIQUE |
| `Ledger` | `slug` | range |
| `Backlog` | `scope` | range |
| `Unit` | `statusToken` | range |
| `Source` | `itemId` | range |
| `Unit` | `itemId` | range |

### 7.6 Replace statement

One literal statement per kind: delete the old source and its children, create the `:Source`, then
`UNWIND $rows` creating the children, all as a pure function of `($path, $meta, $rows)`. Atomic: a
failed statement leaves the previous subgraph. NUL bytes in any string are replaced by U+FFFD and
the source gains the warning `<n> NUL byte(s) replaced by U+FFFD`.

## 8. Key and indexed-value hygiene

A UNIQUE-indexed value over 4,096 bytes, or with leading or trailing whitespace at any length,
crashes the shared FalkorDB, and a whitespace-padded value silently misses an indexed equality
lookup. So:

- `safe_key(*parts)` builds every UNIQUE value: each part and the whole stripped; several parts
  join with `#` (`<path>#<unitKey>`); over 512 UTF-8 bytes becomes `h:<sha256>`. It raises
  `KeyHygieneError` if an edge whitespace survives.
- `norm_value(v)` strips every indexed non-unique value (`slug`, `scope`, `statusToken`, `itemId`).
- At the write boundary the store **asserts** a UNIQUE value is already `safe_key`-clean (it never
  rewrites one) and **normalises** the indexed non-unique ones; a non-unique indexed value must be
  non-empty and at most 512 bytes. A refusal is `UnsafeValue` and fails that source only.
- The same two functions apply to every lookup parameter the views send.

## 9. KB mapping nodes

Written into the mirror graph (not into falkor-chat).

- `:KbFile {path, sha256, syncedAt}` with `(:KbFile)-[:HAS_SECTION]->(:KbSection)`. Stamped only
  when none of the file's sections is pending, refused or failed.
- `:KbSection` keyed by `key` = `safe_key(<file path>, <headingKey>)`: `path`, `heading`, `ord`,
  `mode` (`whole`\|`bullets`), `state` (`pending`\|`synced`), `sectionHash`, `documentIds`,
  `claimHashes`, `claimTitles` (parallel lists, one entry per mirrored claim), `needsResplit`
  (a `whole` section that once mapped several documents), `syncedAt`.
- **Lease.** `leaseToken`, `leaseUntil` (epoch ms, TTL 60 s): the section's compare-and-set lock.
- **Pending.** While a changed section is awaiting embeds: `pendingSince`, `pendingSectionHash`,
  `pendingMode`, `pendingDocumentIds`, `pendingClaimHashes`, `pendingClaimTitles`,
  `pendingKeepHashes` (hashes of unchanged old claims that survive the swap).
- **Retry.** `retryCount`, `lastRetryAt` (epoch ms): the failed-document re-submissions since the
  last successful swap; cleared by the swap, kept by a give-up.

## 10. Hook state files

In `$COORDGRAPH_STATE_DIR` (default `coordination-graph/.state/`, gitignored):

| File | Content |
|---|---|
| `stamp` | mtime marker: files newer than it are the changed set |
| `failed.json` | `{repo-relative path: {sha256, error, attempts, nextRetryAt}}`; written atomically |
| `failed.pending` | one path per line, appended by the hook shell under `flock state.lock`, merged and truncated by Python under the same lock |
| `retry-due` | epoch seconds of the earliest due retry; absent when nothing waits |
| `disabled` | presence is the kill switch |
| `hook.log` | stdout/stderr of the hook's Python calls and detached children |
| `state.lock`, `retry.lock`, `kb-finish.lock` | `flock` files; `retry.lock` and `kb-finish.lock` are taken without waiting |
