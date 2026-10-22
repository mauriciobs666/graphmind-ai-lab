# Coordination graph — graph model, write path and operations

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** — · **Version:** 2

*Revision 2026-10-21 (v2):* folds in analyst Pass 1 (`claude/docs/reviews/coordination-graph.md`)
— M2 status normalization (§2.5), M3 unit lookup and review resolution (§2.2, §7.2(c), §7.3),
M6 first load (§5.5), m1, m2 (§4.1), m8 and M4's graph side (§6.4), n5 (§3.1); plus plan v2
Appendix B — `statusMarker`/`open` (§2.2, §7.5), planner-scope and overlap deferrals (§7.5),
`pendingDocumentIds` (§2.4, §6.4), Q1/Q2 answered (§9.3); plan Pass 2 P2-M3 — per-section lease
CAS and the cancel ordering (§2.4, §6.4 "Per-section lease"); Pass 3 P3-M1 — lease rules
L1 renew-before-call, L2 clean-up-own-POST, L3 swap order (§6.4).

Graph-side design note for `claude/docs/plans/coordination-graph.md` (architect), implementing
`claude/docs/requirements/coordination-graph.md`. Canonical for every Cypher/DDL detail the
architect's plan cites. Every claim is tagged **[live]** (probed against the running instance on
2026-10-20 — engine `falkordb/falkordb:v4.18.11`, Redis 8.6.3, `graph` module `41811`,
`falkordb-py` 1.6.2 from `cypher-mcp/.venv`), **[quirks]** (already live-verified in
`claude/graph-dba/falkordb-quirks.md`), **[docs]** or **[inferred]**. All probes ran in one
disposable graph, `scratch:graphdba-coordprobe` (§9 says whether it was deleted). v2's probes (2026-10-21)
ran only on a throwaway `falkordb/falkordb:v4.18.11` container on port 6391, never the shared instance.

## 1. Where the mirror lives

**A dedicated graph key, `coordination`** (overridable by `COORDGRAPH_KEY` / `--graph`, which the
rebuild parity test and the suite use to point at a scratch key).

Live keys today **[live]** (`GRAPH.LIST`): `cpg_falkorchat`, `reference`, `ws:acme`,
`ws:agent-team`, `ws:demo`, `ws:eval`, `ws:k069probe`, `ws:nlq-eval`, `ws:test`. Naming
conventions in use: `ws:<id>` is falkor-chat's workspace namespace (its server builds keys as
`f"ws:{ws}"`, `falkor-chat/server/falkorchat/config.py:268`, and its tests diff `GRAPH.LIST` for
`ws:` keys), `cpg_<component>` is the Joern loader's, and `reference` is falkor-chat's own
reference graph (`WorkflowDef`/`Step`/`Product`). A bare lowercase noun matches `reference`;
anything under `ws:` would masquerade as a falkor-chat workspace.

Why not inside `ws:agent-team`:

- **Different owner and lifecycle.** `ws:agent-team` is falkor-chat's schema (its bootstrap,
  20+ labels, vector/fulltext indexes, `seed_agent_team.sh`). Its documents are written only
  through falkor-chat's service layer; a second, direct-Cypher writer with its own labels inside
  that key couples the mirror to every falkor-chat reset/reseed and lets falkor-chat schema
  changes collide with ours.
- **Rebuild and blast radius.** The mirror must be droppable and rebuildable from files at will
  (FR-3/AC-2); `ws:agent-team` must not be (its KB documents carry embeddings and LLM-extracted
  entities that cost minutes of LM Studio time to regenerate). One key per lifecycle.
- **Nothing is gained by co-location.** Nothing here traverses into `ws:agent-team` (cross-graph
  edges do not exist in FalkorDB — **[quirks]** "Cross-graph edges silently no-op"), and both keys
  sit in the same Redis instance, so they share the same RDB snapshot instant anyway (§6).
- Single-shard-per-graph is moot: one non-clustered instance, and the mirror is ~1–2 MB (§8).

The KB-mirror sync state (FR-10) lives in this same `coordination` key, under labels the mirror
rebuild never touches (§2.4, §6.4).

## 2. Data model

### 2.1 Shape

```
(:Source:Ledger)      -[:HAS_UNIT]->  (:Unit)        one per non-archived ledger row
(:Source:Backlog)     -[:HAS_ITEM]->  (:Item)        one per backlog item
(:Source:Review)                                      status + latest verdict only, no children
(:Source:Requirement)                                 header only, no children
(:SyncState {id:'mirror'})                            singleton: schema/parser version, last full sync, last rebuild
(:KbFile) -[:HAS_SECTION]-> (:KbSection)              FR-10 sync state — NOT a :Source, never touched by mirror sync/prune/rebuild
```

**One `:Source` node per mirrored file**, carrying a second, kind label. Everything a file
contributes hangs off its `:Source` by a single, kind-specific, *intra-source* relationship — so
"the file's subgraph" is exactly `(s:Source {path})` plus its out-neighbours, and nothing else
in the graph points into it (§2.3). Labels `PascalCase`, relationships `UPPER_SNAKE`, properties
`camelCase` — the falkor-chat convention already used on this instance.

### 2.2 Properties

**`:Source` (all kinds)** — the per-source sync metadata:

| Property | Type | Notes |
|---|---|---|
| `path` | string | repo-root-relative, `/`-separated; **UNIQUE** (§4). Always ≪ 4096 bytes. |
| `kind` | string | `ledger` · `review` · `requirement` · `backlog` |
| `component` | string | first path segment (`claude` for agent plans). Repo-root `docs/`: `cpg` for `docs/BACKLOG.md` only (its header says it is `cpg`'s); `root` for everything under `docs/{plans,reviews,requirements}/`, which mixes cross-cutting and model-bench/salesperson work (root `AGENTS.md`) — never a guessed owner. Views print the path beside it. |
| `slug` | string | topic slug: basename minus `.md` minus a closed-set role suffix (`-coordination`, `-ml`, `-graph`, `-rca`, `-impl`, `-report`) — the join key for "picked up" (§2.3) |
| `docStatus` | string | the canonical `Status:` token from the header block; **absent** when the file has none (`kaizen/plan.md`, `BACKLOG.md`) |
| `owner`, `tracks` | string | header block fields, raw, when present |
| `sha256` | string | hex sha256 of the file's **raw bytes** at parse time — the FR-5 file-level drift key |
| `bytes`, `mtimeNs` | int | for the hook's mtime prefilter and diagnostics only; never a correctness input |
| `syncedAt` | int | epoch ms of the committing write (client clock) |
| `parserVersion` | int | bump forces a re-sync of every source (`sync --all` compares it) |
| `warnings` | list<string> | parse warnings, `[]` when clean — e.g. `table 3: no Status column`, `duplicate unit U32 → U32~2`, `item without id`. Stored list, never absent, so "clean" ≠ "not parsed". |
| `rowCount` | int | children written — parity input (§5) |

**`:Ledger`** adds two lists, written on **every** ledger, archived or not, always as a list
(`[]` when empty, never absent — so `IN` never meets a `null`):

| Property | Notes |
|---|---|
| `citedReviews` | sorted, distinct repo-root paths of covered review files cited in backticks in any row's `Deliverable` or `Gate → verdict` cell. A cell path that does not exist from the repo root is retried under the ledger's component directory (`docs/reviews/x.md` in a `falkor-chat/` ledger → `falkor-chat/docs/reviews/x.md`); a path that resolves to no review file on disk is dropped with a source warning. |
| `resolvedReviews` | the subset cited by a row whose `statusToken` (§2.5) is `accepted` or `abandoned` |

An **archived ledger is the same node with `docStatus:'archived'`, these two lists, and no
`:Unit` children** — that *is* the "stub" the framing asks for, with no separate label to keep in
step, and it keeps exactly the review references §7.2(c) needs. Both lists are a function of the
file, so rebuild and parity (§5) cover them like any other `:Source` property. **`:Review`** adds `latestVerdict` (raw verdict text of the last `## Pass N`, or the
single verdict), `latestPass` (int, 1 when unnumbered), `verdictClass` (`needs-changes` ·
`approve` · `approve-with-suggestions` · `reject` · `unknown` — the parser's normalization of the
spellings the plan §2.1 lists). **`:Backlog`** adds `scope` (`falkor-chat`, `cpg`, `cobb`, …) and
`scopeKind` (`component` | `agent`). **`:Requirement`** adds nothing.

**`:Unit`** (ledger row):

| Property | Notes |
|---|---|
| `key` | `<ledgerPath>#<unitKey>` — **UNIQUE**. `unitKey` is the Unit cell with Markdown emphasis/backticks stripped and cut at the first ` — `; the **n-th repeat in the same file gets `~n`** (`U32~2`). Unit labels are *not* unique per file today **[live]** — 8 ledgers repeat one (`U32`, `U3b`, `U4`, `U11`, `U1`, `Close-out`, …). The CLI caps `key` at 512 bytes (longer → `h:<sha256>`), because a >4096-byte value in a UNIQUE property crashes the instance **[quirks]**. |
| `unit` | the Unit cell as displayed |
| `unitKey` | `unitKey` as defined above, **without** the `~n` suffix — the FR-8 lookup value (§7.3), so a lookup of `U32` returns every repeat. Not indexed (filtered after the indexed ledger anchor). |
| `ledgerPath`, `slug`, `component` | denormalized from the parent — lets FR-8 anchor on an index without a hop |
| `tableIdx`, `rowIdx` | position (one file holds 30 tables) — display order only, **not** identity |
| `statusText` | the raw `Status` cell, verbatim — what the views display |
| `statusToken` | the normalized status (§2.5): one of teco's 7 values or `unknown`; **always present**, indexed |
| `owner`, `agentId`, `deliverable`, `gateVerdict`, `cost` | cells mapped **by header name** (header variants: `Owner`/`Agent`/`Agent (scope)`, `Unit`/`Unit(s)`, with/without `Cost`); a column the table lacks is **absent**, not `''` |
| `question`, `askedOn` | `statusToken = 'paused'` rows only: the open question and relay date parsed out of `Deliverable` (`askedOn` as `YYYY-MM-DD` string; absent + a warning if unparseable) |
| `cites` | list<string> of repo paths found in backticks in `Deliverable`/`Gate → verdict` — query-time join target for reviews (§2.3) |
| `rowHash` | sha256 of the row's normalized cell list — the FR-5 **row-level** drift key |

Cell text runs up to 5,291 bytes **[live]**; that is fine on unconstrained properties (safe to
≥1 MB **[quirks]**) — **never put a cell on an indexed-UNIQUE property**.

**`:Item`** (backlog item; FR-14 fields modelled as *may be absent*):

| Property | Notes |
|---|---|
| `ref` | `<scope>:<itemId>` — **UNIQUE**, e.g. `falkor-chat:K-065`, `cobb:K-058`, `cpg:C-310`. IDs collide across backlogs today (13 agent plans all use `K-0NN`), so only the qualified form is a key. An id-less item (today's `mcp-monitor`/`opencode` bullets) gets `ref:'<scope>:@<ord>'` plus a source warning — positional, so it is honest about being unstable until cobb's common format gives it an id. |
| `scope`, `itemId`, `backlogPath`, `ord` | `ord` = position in the file |
| `title`, `statusMarker`, `open` | `statusMarker` = the item's `Status` marker as written (`🔵 proposed` · `🟡 in-progress` · `⚪ deferred` · `⚪ rejected` · `✅ delivered`, plan §3.7 / FR-14 format). `open` = **every item still in the backlog except `✅ delivered`** — so `🔵`, `🟡`, `⚪ deferred` and `⚪ rejected` are all `open = true`; the planner classifies them from `statusMarker` (plan §3.7 step 1), and the "planned" view shows the marker beside each. |
| `priority`, `milestone` | absent = not recorded |
| `dependsOn` | list<string> of **qualified** refs (the parser qualifies a bare `K-065` with the item's own scope). **Absent = "not recorded"; `[]` = "recorded: none".** FalkorDB stores an empty list distinctly from an absent property — `keys(n)` shows it, `size()` is 0, `IS NULL` is false **[live]** — while a *null* is never stored at all **[quirks]**, so the distinction survives only if the parser writes `[]` deliberately and omits the key otherwise. |
| `touches` | list<string>, each typed `file:<path-or-glob>` · `graph:<key>` · `fact:<token>`; same absent-vs-`[]` rule |
| `itemHash` | sha256 of the item's normalized source text — row-level drift key |

### 2.3 Cross-source references: stored strings, resolved at query time

**Decision: no relationship ever crosses a source boundary.** Every cross-source reference is a
string property *owned by the referring row* and joined at read time against an indexed key:

| Reference | Stored on the referrer as | Resolved by joining |
|---|---|---|
| item → item (same or other backlog) | `Item.dependsOn` (`'falkor-chat:K-065'`) | `Item.ref` (UNIQUE index) |
| ledger unit → review doc | `Unit.cites` (paths, display) | `Source.path` (UNIQUE index) |
| ledger → review it closed | `Ledger.citedReviews` / `resolvedReviews` (§2.2) | `Review.path` by list membership (§7.2(c)) |
| requirement → "picked up" | nothing stored — derived | `Ledger.slug = Requirement.slug` (index on `slug`) |

Why not edges re-linked on sync: replacing source B's subgraph `DETACH DELETE`s B's nodes, which
silently deletes every edge *another* source A pointed into B — A's fact disappears though A's
file never changed, and nothing records that it ever existed. Re-linking is possible (after
recreating B's items, re-`MATCH` every referrer whose list contains a recreated ref and rebuild
its edge, inside the same statement), but it makes every sync's correctness depend on every
other source's stored state, and a referrer to a ref that is *temporarily* absent loses its edge
for good. With strings:

- **Replacing one source can never alter another source's facts** — each source's subgraph is a
  pure function of its own file, which is exactly what FR-3's "rebuildable from files alone" and
  §5's parity proof need.
- **A dangling reference is visible, not lost**: a `dependsOn` ref with no matching `:Item` reads
  back as *unresolved* (§7.4) — the proposal must name it, never drop it.
- Cost is nil at this size: the join is an index seek per ref over ~500 items (§7.5).

**One semantic gap the graph cannot close (answered by plan §3.7 step 2, §9.3 Q1):** backlogs are forward-only, so a delivered
item *leaves* its `BACKLOG.md`. An unresolved `dependsOn` ref therefore means either "delivered"
(fine — the dependency is satisfied) or "never existed / typo" (a defect), and history is
deliberately not mirrored (FR-1). The graph can only report "not in any open backlog".

### 2.4 KB-mirror sync state (FR-10)

```
(:KbFile {path UNIQUE, sha256, syncedAt})
  -[:HAS_SECTION]->
(:KbSection {key UNIQUE = '<path>#<headingKey>', path, heading, ord,
             sectionHash, state: 'synced'|'pending', pendingSince,
             documentIds: [..], claimTitles: [..], claimHashes: [..], syncedAt,
             pendingDocumentIds: [..], pendingSectionHash, pendingClaimHashes: [..],
             leaseToken, leaseUntil})
```

While `state = 'pending'`, `documentIds`/`claimHashes`/`sectionHash` still describe the **old,
still-served** claim(s), and the `pending*` properties describe the new document(s) submitted but
not yet retrievable — so a crash, a rollback or a `kb-finish` run always knows both sets: which to
delete on success and which to delete on failure. On `synced` the `pending*` properties are
removed (`SET ks.pendingDocumentIds = null` removes the property **[live, throwaway container,
2026-10-22]**); they are
never present on a `synced` section.

`leaseToken` (random per acquire, e.g. a uuid4 hex) and `leaseUntil` (epoch ms, **FalkorDB
server clock** via `timestamp()`, so processes never compare their own clocks) are the
per-section lease of plan §3.5 — both absent when the section is free. Statements in §6.4.

`claimHashes[i]` is `sha256(normalize_text(claimText))` using falkor-chat's own normalizer
(whitespace-collapse + strip + `casefold`, `falkor-chat/server/falkorchat/update_detection.py:74`),
so it is directly comparable to `Document.textNormalizedHash` in `ws:agent-team` **[live]: the
property exists on every `Document`** — that is what makes `kb-reconcile` (§6.4) a pure hash
comparison. Unlike the mirror, these nodes are **not** a function of the file alone (the
`documentId`s come back from falkor-chat), so they are written by per-section `MERGE … SET`, never
by the replace-from-parse statement, and the mirror rebuild never touches `:KbFile`/`:KbSection`.

### 2.5 Status normalization (`Unit.statusToken`)

Ledger `Status` cells are free text — 362 distinct values over 74 ledgers (review Appendix B):
`**in-flight** — dispatched …`, `**accepted — APPROVE WITH SUGGESTIONS**`, `✅ done 2026-07-29 — …`,
`**relayed**`, `—`. Every view filters on `statusToken`, never on the raw cell. The parser maps a
cell in this order, as one pure function with its own fixture table:

1. Drop backticks, `*`, `_`, and emoji/pictographs (`✅` …); `strip()`; casefold.
2. Take the leading word, letters and hyphens only (`in-flight`); also try the two-word form
   (`in flight` → `in-flight`).
3. **Exact match** to teco's vocabulary (`claude/teco/teco.md`, ledger convention): `queued` ·
   `in-flight` · `delivered` · `gated` · `accepted` · `abandoned` · `paused`.
4. Else a **closed alias table**, a code constant, each entry justified by a real cell:
   `relayed` → `paused` (teco's "question relayed to the stakeholder"),
   `done` → `delivered`, `superseded` → `abandoned`.
5. Else **`unknown`**, plus a source warning naming the row and the raw text.

Deliberately **not** aliased: `blocked` (the one real cell is blocked on an architect fix, not on
the stakeholder, so `paused` would put it in the wrong view), `teco-verified`,
`accepted-checkpoint`, verdict-shaped cells (`needs changes …`, `approve with suggestions …`,
`pass with defects …`) — those read `unknown` with a warning, and a human fixes the cell or
extends the alias table with a fixture. Offline dry run (2026-10-21, no database) over today's
74 ledgers / 911 status cells with exactly this function: `accepted` 425 · `delivered` 357 ·
`in-flight` 37 · `gated` 26 · `queued` 17 · `paused` 5 · `abandoned` 3 · **`unknown` 41**. All 5
`paused` are `**relayed**` — no ledger holds a bare `paused` cell today, so the v1 exact-match
query would have found **none** of them. All 41 `unknown` are the forms listed above and all sit
in archived ledgers, which the views do not show. The parser
fixtures (plan §5) must include every raw variant quoted in this section, including the
`**paused** — …`, `**relayed**`, `✅ done …`, `—` and `blocked (…)` forms.

## 3. Write path & atomicity

### 3.1 Coverage rule: every covered file gets a `:Source`; children only while live

Every file matched by the coverage globs — `**/docs/plans/*-coordination.md`,
`**/docs/reviews/*.md`, `**/docs/requirements/*.md`, every `**/docs/BACKLOG.md`, every
`claude/*/kaizen/plan.md` — gets a `:Source` node with its `sha256`, **whatever its status**
(~300 files today, ~300 small nodes). One glob form everywhere: `**` with Python
`glob(..., recursive=True)` semantics (so it also matches the repo-root `docs/` tree), excluding
`deprecated/**` and any path with an `archive/` segment — the same rule as plan §3.3. Children are
written only while the file is live: units for a non-archived ledger, items for a backlog; a
review/requirement carries its few fields on the `:Source` itself and the *views* filter by
`docStatus`. An archived ledger's rows are still **parsed** (they feed `citedReviews` /
`resolvedReviews`, §2.2) but not written. This keeps FR-5 coverage a set
comparison — *glob set on disk* vs *`:Source.path` set in the graph* — with no per-kind
exceptions, and gives the archived-ledger "stub" for free.

### 3.2 "Replace one source file's subgraph" — one statement per kind

The CLI holds **one literal statement per kind** (labels cannot be parameters; they are code
constants, never spliced from input). Ledger form — the backlog form is identical with
`:Source:Backlog` / `HAS_ITEM` / `:Item`, and the review/requirement form stops after `SET s = $meta`:

```cypher
OPTIONAL MATCH (old:Source {path: $path})
OPTIONAL MATCH (old)-[]->(child)
WITH old, collect(child) AS children
FOREACH (c IN children | DETACH DELETE c)
FOREACH (o IN CASE WHEN old IS NULL THEN [] ELSE [old] END | DETACH DELETE o)
WITH count(*) AS _
CREATE (s:Source:Ledger)
SET s = $meta
WITH s
UNWIND $rows AS row
CREATE (s)-[:HAS_UNIT]->(u:Unit)
SET u = row
RETURN count(u) AS written
```

- `$meta` is the `:Source` property map (incl. `path`, `sha256`, `syncedAt`, `warnings`,
  `rowCount`); `$rows` the list of child property maps. **A `null` map value is simply not
  stored [live]** — so the parser sends `null` for "column absent / fact not recorded" and `[]`
  only for "recorded: none" (§2.2).
- Both `OPTIONAL MATCH` anchors plan as `Node By Index Scan | (old:Source)` **[live, EXPLAIN]**,
  for `$path` and for `$meta.path` alike.
- **Zero rows is safe**: the `:Source` is created *before* the `UNWIND`, so an archived ledger or
  an empty backlog still gets its node — the empty-`UNWIND` collapse **[quirks]** only drops the
  `RETURN` row (`written` reads 0) **[live]**: replacing with `$rows = []` left the `:Source` with
  the new `sha256` and no children.
- **Deleting and re-creating the same UNIQUE keys in one statement works [live]** — no constraint
  violation, `Nodes deleted: 6 / Nodes created: 6`.
- A file that disappeared (rename, delete) uses a delete-only statement:
  `MATCH (s:Source {path:$path}) OPTIONAL MATCH (s)-[]->(c) WITH s, collect(c) AS cs FOREACH (c IN cs | DETACH DELETE c) DELETE s`.

The quirks file's line *"there is no clear-then-write, no delete-and-recreate … in a single
call"* (Concurrency/Cypher section) is **too strong**: what FalkorDB refuses is a
*multi-statement* query; a delete-and-recreate fits in one statement using `WITH` as the clause
bridge (its own next bullet shows the chaining). Verified live above — I will fold the correction
into `falkordb-quirks.md`.

### 3.3 Atomicity and isolation

| Property | Result |
|---|---|
| **Atomic** — all-or-nothing | **[live]** A replace whose `$rows` held a duplicate `key` failed with `unique constraint violation on node of type Unit`, and the graph still held the **previous** subgraph untouched (old `sha256`, both old units) — the earlier `DETACH DELETE`s in the same statement were rolled back too. Consequence: a parser bug that emits a duplicate key can never leave a file half-mirrored; the CLI reports the error and the drift check keeps flagging that file. |
| **Isolated from `GRAPH.RO_QUERY` readers** | **[live]** 3 reader threads polling `OPTIONAL MATCH (s:Source {path}) OPTIONAL MATCH (s)-[:HAS_ROW]->(u) RETURN s.sha256, count(u), count(DISTINCT u.status)` while two writers alternately replaced that one source with 30,000 and 20,000 rows (0.38–0.61 s per write): **104 observations, every one a complete state** (`sha A ↔ 30000 rows, 1 status`, `sha B ↔ 20000`, or the prior state) — never an absent `:Source`, a partial count or mixed statuses. Matches **[quirks]/[docs]** (docs.falkordb.com/design/concurrency: writes are atomic, readers never see partial writes). |
| **Concurrent writers** | **[live]** The same run: the two writers' 8 replaces serialized (per-graph write queue **[quirks]**); final state was exactly one `:Source`, 30,000 units, no duplicates. **Last committed statement wins.** |

**The one race left is a stale parse winning**: session A parses file v1, session B parses v2
and commits, then A commits v1. Both statements are individually atomic, so the graph ends at v1
while the file is v2. Close it client-side, not with a lock: **after its write, the CLI re-hashes
the file; if the hash differs from the `sha256` it just wrote, it re-parses and writes again**
(bounded, e.g. 3 rounds, then report). Under per-graph write serialization this converges: the
last writer to commit always re-checks against the file's final content. FR-5's read-time drift
check (§7.1) remains the backstop if a process dies between write and re-check.

**Retries are harmless**: the statement is a pure function of `($path, $meta, $rows)`, so
re-applying it is idempotent — relevant because a client-side socket timeout abandons the answer,
not the write **[quirks]** (`falkordb-py` disables retry anyway **[quirks]**). Writes ignore
`TIMEOUT` **[quirks]**; at these sizes that is moot (§8).

**Parameter size.** `falkordb-py` inlines parameters into a `CYPHER k=v …` preamble, so the bound
is Redis's bulk-string limit (`proto-max-bulk-len`, 512 MB default **[docs]**, not checked live),
not a FalkorDB parameter cap. **[live]** a 4.1 MB `$rows` (2,000 rows × 2 KB) committed in 55 ms;
the largest mirrored file is cobb's `kaizen/plan.md` at 81 KB. No chunking is needed per file;
chunk only if one file's rows ever exceed ~10 MB.

**Write cost.** Per-file statements are milliseconds (§8), and each write holds only this graph's
write queue — other keys (`ws:agent-team`) are unaffected **[inferred** from per-graph
serialization; not measured cross-key**]**. `sync --all` is ~300 statements; run them as
separate calls (sync unit = one file, so one bad file cannot block the rest).

## 4. Indexes, constraints & bootstrap

### 4.1 Key hygiene — mandatory, because a bad key kills the whole instance

Two live-verified ways a value in a `UNIQUE`-constrained property **segfaults the entire shared
FalkorDB** (every graph, plus `ws:agent-team`'s server, which answers 503 until FalkorDB is back):

1. a value **> 4,096 bytes** **[quirks]**;
2. a value with **leading or trailing whitespace, at any length** — `'set '`, `' a'`, `'a\t'`,
   `'a\n'`, `'a\r'`, `'a\v'`, `'a\f'` **[live, throwaway containers — found by this consult, see
   §9.1]**. The crashing set is exactly C `isspace()`; Unicode spaces (NBSP, em space, ideographic
   space, zero-width space) are safe **[live, 2026-10-21]**. An exact-match index also silently
   *misses* such a value on lookup (0 rows via `Node By Index Scan`, 1 row by scan), **constraint
   or not**.

Two functions, applied before any write **and** to every lookup parameter:

- **`safe_key()`** — every UNIQUE value (`Source.path`, `Unit.key`, `Item.ref`, `KbFile.path`,
  `KbSection.key`): `strip()` each component (and the whole), then if the UTF-8 length exceeds
  **512 bytes** replace it by `'h:' + sha256(value)`; finally **assert** that neither end
  `isspace()` and the length is ≤ 512 bytes, and raise otherwise (defence in depth — the failure
  mode is not a query error, it is an outage). Python's `strip()`/`isspace()` cover a strict
  superset of the crashing set, so no NBSP special case is needed.
- **`norm_value()`** — every value on an **indexed but not unique** property (`Ledger.slug`,
  `Backlog.scope`, `Unit.statusToken`) and the matching lookup parameters (`$slug`, `$scope`,
  `$unit` for §7.3): the same strip-and-assert, no hashing (these are short by construction). No
  crash risk without a constraint, but an unstripped value is an index lookup that silently
  returns nothing.

Test both with the exact crashing inputs above as fixtures (against an ephemeral container,
§5.4), and keep long free text (cells, titles, headings) **off** indexed properties. The design's
keys are path- and id-shaped, so collisions from stripping are not a practical concern; the
parser's `~n` duplicate suffix is applied *after* stripping.

### 4.2 Schema

| Kind | Label.property | Why |
|---|---|---|
| RANGE + **UNIQUE** | `Source.path` | replace anchor; join target for `Unit.cites` |
| RANGE + **UNIQUE** | `Unit.key` | integrity of the row identity (a parser bug fails the statement atomically instead of duplicating) |
| RANGE + **UNIQUE** | `Item.ref` | integrity; dependency resolution seek (`Node By Index Scan \| (d:Item)` **[live]**) |
| RANGE + **UNIQUE** | `KbFile.path`, `KbSection.key` | KB sync state is `MERGE`d per section — `MERGE` needs a constraint behind it |
| RANGE | `Ledger.slug` | FR-8 anchor and the "picked up" anti-join (`Node By Index Scan \| (l:Ledger)` **[live]**) |
| RANGE | `Backlog.scope` | FR-12/FR-13 anchor (`Node By Index Scan \| (b:Backlog)` **[live]**) |
| RANGE | `Unit.statusToken` | paused-questions view anchor (`Node By Index Scan \| (u:Unit)` **[live]**, re-checked on v2's query) — the closed-set token, never the raw `statusText` |

Indexes on a kind label work on the multi-labelled `:Source:Ledger` nodes **[live]**. No full-text
or vector index: nothing here is searched by meaning. `(:SyncState {id:'mirror'})` is a singleton
written by one `MERGE … SET` statement — per-graph write serialization makes that race-free
without a constraint **[quirks]**.

### 4.3 Idempotent bootstrap (`coordgraph bootstrap`, also run at the start of `sync`/`rebuild`)

Re-running DDL is **not** a no-op on this build: `CREATE INDEX` on an indexed property errors
*"Attribute 'path' is already indexed"* and `GRAPH.CONSTRAINT CREATE` on an existing one errors
*"Constraint already exists"* **[live]**. So bootstrap is read-then-create:

```text
1. existing_idx  = GRAPH.QUERY  CALL db.indexes()     YIELD label, properties RETURN label, properties
   existing_cons = GRAPH.QUERY  CALL db.constraints() YIELD label, properties, status RETURN label, properties, status
   (use GRAPH.QUERY, not RO_QUERY, on first run: RO_QUERY on an absent key errors "empty key" [quirks];
    GRAPH.QUERY materializes the key, which is what bootstrap wants)
2. for each missing RANGE index:   CREATE INDEX FOR (n:<Label>) ON (n.<prop>)          -- index FIRST
3. for each missing constraint:    GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE <Label> PROPERTIES 1 <prop>
                                   -- NODE, not LABEL; needs the index from step 2 [quirks]
4. poll db.constraints() until every status = 'OPERATIONAL' (returns PENDING first [live]);
   fail loudly on 'FAILED' or after ~5 s. Guard the poll against an EMPTY list reading as
   "all operational" (Python's all([]) is True).
5. MERGE (m:SyncState {id:'mirror'}) SET m.schemaVersion = $v
```

No write may run before step 4 completes — a write against a `PENDING` constraint is not
guaranteed to be checked.

## 5. Rebuild & parity proof

### 5.1 Rebuild = unconditional upsert of every covered file + prune — not drop-and-reload

The architect's table says `rebuild` "drops the mirror and reloads it". Push back: **don't
`GRAPH.DELETE`.** Because each source's subgraph is a pure function of its file (§2.3) and the
replace statement removes every node the previous version of that file contributed (§3.2), an
unconditional replace of every covered file *is* a rebuild from files alone. Dropping the key
instead would:

- open a window where every reader gets `ERR Invalid graph operation on empty key` and FR-4/AC-3
  fail by construction;
- destroy the `:KbFile`/`:KbSection` state, which is **not** derivable from files (§2.4, §6.4);
- route through the destructive-ops hook (`GRAPH.DELETE` escalates to a human every time).

Procedure (`coordgraph rebuild`):

1. bootstrap (§4.3);
2. enumerate covered files (§3.1 globs); for each, parse and run the kind's replace statement
   **regardless of stored `sha256`/`parserVersion`** (that is the only difference from
   `sync --all`);
3. prune sources no longer on disk, mirror kinds only (`:KbFile` is not a `:Source`):
   ```cypher
   MATCH (s:Source) WHERE NOT s.path IN $paths
   OPTIONAL MATCH (s)-[]->(c)
   WITH s, collect(c) AS cs
   FOREACH (x IN cs | DETACH DELETE x)
   DELETE s
   ```
4. orphan sweep — must always delete 0; non-zero is a bug signal to report, not to hide:
   `MATCH (u:Unit) OPTIONAL MATCH (s:Source)-[:HAS_UNIT]->(u) WITH u, s WHERE s IS NULL DELETE u RETURN count(u)`
   (same for `:Item`/`HAS_ITEM`; the `OPTIONAL MATCH … IS NULL` anti-join is the working form
   **[quirks]** — `EXISTS {}`/`exists()` are unusable);
5. `MERGE (m:SyncState {id:'mirror'}) SET m.lastRebuildAt = $now, m.parserVersion = $pv`;
6. run the parity check (§5.2) and print its verdict.

Measured **[live, throwaway container]**: a full load of the real repo — 301 per-file statements
(72 ledgers, 168 reviews, 43 requirements, 18 backlogs) — took **0.31–0.35 s** end to end;
statement median 0.6 ms, p95 1.1–1.3 ms, max 11 ms.

### 5.2 Proving "nothing missing" (AC-2)

The check compares a **fresh parse** against a **graph export**, never the graph against itself:

- **Expected** — parse every covered file → per path: `kind`, `docStatus`, `sha256`, `rowCount`,
  and the set of `(childKey, childHash)` where `childHash` is computed by the parser from the
  canonical property map it *would* write.
- **Actual** — one read, ~300 rows (well under `RESULTSET_SIZE` 10000 **[quirks]**; assert the
  row count equals the source count, since a capped reply looks complete):
  ```cypher
  MATCH (s:Source)
  OPTIONAL MATCH (s)-[]->(c)
  RETURN s.path AS path, properties(s) AS meta, collect(properties(c)) AS kids
  ```
  (`properties()` works on this build **[live]**.) The checker **recomputes** each child's hash
  from the returned property map with the same canonicaliser — it does not trust the stored
  `rowHash`/`itemHash`, so a write-side mapping bug (a column landing in the wrong property, a
  `null` dropped) is caught.
- **Verdict** — path sets equal, and per path `sha256`, `rowCount` and the child `(key, hash)`
  sets equal. Report every difference by path and key; pass only on zero.

**AC-2 as a test**: `rebuild --graph <empty scratch key>` then parity of that key against the
fresh parse **and** against the live `coordination` export (equal whenever no file changed in
between). The check must be able to fail: a mutation test edits one property in the scratch key
and asserts the parity check reports exactly that key.

### 5.3 `sync --all` vs `rebuild`

`sync --all` = steps 1–5 but skipping files whose stored `sha256` and `parserVersion` both match —
the routine repair after a crash (§6.2). `rebuild` = the unconditional form, for parser changes,
suspected corruption, and AC-2.

### 5.4 Where the tests run

**Never against the shared `falkordb-dev`.** This consult crashed it once with a single mirror
write (§9.1). The CLI's suite should run against an ephemeral `falkordb/falkordb:v4.18.11`
container on a non-default port (the way the crash was isolated here: `docker run -d -p 6390:6379
falkordb/falkordb:v4.18.11`, no volume, removed after) — that plumbing is `devops`'s to provide
(a fixture/script), and it also removes any need for `GRAPH.DELETE` in tests.

### 5.5 First load into the shared instance (owner: `graph-dba`)

The first `bootstrap` + `rebuild` against `falkordb-dev` is the riskiest write this design ever
makes: it is the first time real repo text reaches UNIQUE constraints on the instance every
session shares, which is exactly how the 2026-10-20 crash happened (§9.1). Run it as its own step,
after the CLI and its suite are green and **before** the post-edit hook goes live. Every gate
below must pass before the next one starts; any failure stops the procedure.

1. **Offline key validation — no database connection.** `coordgraph rebuild --dry-run` (or an
   equivalent `validate` subcommand) parses every covered file (§3.1) and computes every property
   map it would write, then asserts:
   - every UNIQUE value (§4.1 list) passes `safe_key()`'s assertions — no `isspace()` at either
     end, ≤ 512 UTF-8 bytes — and every indexed value passes `norm_value()`'s;
   - **no duplicates across the whole load** for each UNIQUE property (`Source.path`, `Unit.key`,
     `Item.ref`, `KbFile.path`, `KbSection.key`) — a cross-file duplicate would not crash, but it
     fails that file's statement atomically and leaves it unmirrored;
   - no cell, title or heading text sits on an indexed property;
   - each `$meta`/`$rows` parameter is under ~10 MB (§3.3).
   It prints the per-kind source counts, the `statusToken` tally with every `unknown` raw value
   (§2.5), and all source warnings. Exit non-zero on any assertion. Record the
   `(path, sha256)` manifest it parsed.
2. **Rehearsal on a throwaway container.** Same image tag (`falkordb/falkordb:v4.18.11`),
   non-default port, no volume (§5.4). Run `bootstrap`, `rebuild`, then the §5.2 parity check;
   run `rebuild` a **second** time and re-check parity (idempotence: identical node counts, zero
   diffs); run each §7 view once. Pass = zero parity diffs on both runs, the container still
   answers `PING` (no crash), and `INFO memory` `used_memory` growth within §8's range. Then stop
   the container.
3. **Freshness check.** Re-hash the covered files and compare with step 1's manifest. Any file
   that changed since then goes through step 1 again (only the changed files) before the live load.
4. **Pre-load snapshot and baseline on the shared instance** (read-only plus one `BGSAVE`):
   `GRAPH.LIST` — record the key list, and **if `coordination` already exists, stop and ask**
   (never delete it as part of this step); `INFO memory`; then `BGSAVE` and wait until `LASTSAVE`
   advances. This makes the restore point *now*, so a crash during the load loses nothing other
   sessions wrote before it, instead of up to 60 minutes (§6.1). Announce the window to whoever
   is working (a crash takes `ws:agent-team`'s server down with it).
5. **Live load.** `coordgraph bootstrap` then `coordgraph rebuild` with
   `COORDGRAPH_KEY=coordination`, then the parity check. Immediately after: `PING`, `GRAPH.LIST`
   (every baseline key still present, plus `coordination`), `ws:agent-team` `/health` 200,
   `INFO memory` delta. Record the parity verdict, the source/unit/item counts and the memory delta
   in the ledger row for this step.
6. **Only then** may the stakeholder register the hook (plan's settings step).

**If something fails:**

- Before step 5: nothing touched the shared instance; fix the parser and restart at step 1.
- During or after step 5 without a crash: the mirror is file-derived and every per-file write is
  atomic (§3.3), so a partial load is just an incomplete mirror — fix and re-run `rebuild`, which
  converges. Never drop the key to "start clean"; that is a `GRAPH.DELETE` and escalates to a human
  (§5.1).
- If the instance crashed: bring it back with the repo's additive bring-up
  (`claude/teco/hooks/ensure-environment.sh`), confirm the RDB loaded is step 4's (`LASTSAVE`), and
  report the window between step 4 and the crash as possibly lost for other sessions. Then
  reproduce on a throwaway container before any retry; do not retry on the shared instance.

## 6. Durability

### 6.1 Settings — verified

**[live]** `CONFIG GET appendonly` → `no`; `CONFIG GET save` → `3600 1 300 100 60 10000`;
`dir` `/var/lib/falkordb/data` (the `falkordb-data` volume, `falkor-chat/scripts/start_falkordb.sh`);
`maxmemory 0`, `maxmemory-policy noeviction`. Each graph write query counts as **one** Redis change
(`rdb_changes_since_last_save` went +1 per `GRAPH.QUERY` write **[live]**). So the loss window is:
≤ 60 s once 10,000 writes accumulate, ≤ 5 min at ≥ 100, else **≤ 60 min** — and with this
mirror's write rate (one statement per edited file) the realistic window is the 60-minute one.
After a long idle period the first write triggers a save almost immediately (≥ 3600 s since the
last save and ≥ 1 change) **[live]**: the instance had last saved ~4.8 h earlier and saved within
~85 s of this consult's first scratch write. The container runs `--rm`, so a crash also removes
the container and its logs; data survives in the volume.

This was demonstrated, not just reasoned: the 2026-10-20 crash (§9.1) restarted the instance from
the RDB taken ~6.5 min before it (`keys loaded: 50`, ~1.8 s load), and every write in that window
was lost — it took two of my scratch graphs with it, while `ws:agent-team` (367 `Document`s) and
the other keys came back intact.

### 6.2 Is RDB-only acceptable because files are authoritative? — Yes, with one condition

The requirements' reasoning holds **for the mirror**: no mirror fact is graph-only. Every `:Source`,
`:Unit` and `:Item` property is a function of a file (plus `syncedAt`/`warnings`, which are
metadata about the sync, regenerated by it). An RDB rollback therefore restores an *older but
internally consistent* mirror — consistent because every replace is atomic (§3.3) and an RDB is a
fork-time point-in-time image of the whole keyspace **[docs; inferred for module keys** —
FalkorDB serializes graphs in the fork child like any key**]**.

**The condition:** detecting "the mirror is behind" must not depend on the stored state alone.
After a rollback, three kinds of divergence exist, and only a **set comparison** catches all of
them — which is why §3.1 gives every covered file a `:Source`:

| After rollback | Detected by |
|---|---|
| file edited after the snapshot | stored `sha256` ≠ sha256 of the file on disk |
| file **created** after the snapshot | path on disk, no `:Source` |
| file **deleted/renamed** after the snapshot | `:Source` with no file on disk |

Every read runs this check (§7.1: ~300 file hashes, a few MB of text, milliseconds), so the first
answer after a restart already reports the drift (FR-5) and points to `sync --all`, which repairs
it (§5.3). Nothing needs to notice the crash itself. Because `syncedAt` rolls back with the data,
it stays *truthful* — it reports when the data actually shown was synced.

What RDB-only does cost: **FR-4 ("same moment") is broken by a crash** until the next sync — the
graph is knowingly behind the file during that window. FR-5 makes it visible, which is what the
requirements chose ("told, with sync time" over self-healing). If the stakeholder wants the window
closed rather than flagged, the fix is `appendonly yes` / `appendfsync everysec` (≤ 1 s loss) — a
`devops`/stakeholder call, not required by this design (§9 Q3).

### 6.3 Graph-only facts in this design

| Fact | Graph-only? | After a rollback |
|---|---|---|
| ledger/backlog/review/requirement content | no — files | re-synced from files |
| `syncedAt`, `warnings`, `rowCount` | metadata | regenerated; old values stay truthful |
| `SyncState.lastRebuildAt` | yes, informational | stale by ≤ 60 min; harmless |
| **`KbSection.documentIds` / `claimHashes`** | **yes** — ids are minted by falkor-chat | rolls back **with** the documents (§6.4) |

### 6.4 Where the KB claim mapping should live — the coordination graph, plus a reconcile

**Agree with moving the live mapping into the graph** (as `:KbFile`/`:KbSection`, §2.4), for the
architect's reason and one more:

- **Same snapshot instant.** `ws:agent-team` and `coordination` are keys of one Redis instance, so
  one RDB covers both at one instant. After a rollback, KB documents ingested after the snapshot
  are gone **and** so are their mapping entries; the mapping's `sectionHash` is old, the KB file's
  current section hash differs, `kb-sync` sees drift and re-ingests (new `documentId`s). The
  system converges with no manual step.
- **The git manifest fails silently in exactly that case.** After a rollback, a git-tracked
  manifest still says "section X → ids {a,b}, hash = current": the sync believes X is in sync,
  while ids a,b no longer exist — the claim is missing from search with nothing flagging it, the
  K-058 failure mode FR-10 forbids. (Git does not help in the opposite disaster either: if the
  volume is lost, the manifest's ids point at nothing.)

Two windows remain, so the mapping must be **reconciled**, not trusted:

1. *Between* falkor-chat's commit of a document and our `KbSection` write (two commands, two
   keys) — a snapshot can fall between them. Mitigate with a two-phase marker, append-before-delete
   (plan §3.5):
   - **before** calling falkor-chat: `SET ks.state='pending', ks.pendingSince=$now,
     ks.pendingSectionHash=$h`;
   - **after** `POST /documents` returns: `SET ks.pendingDocumentIds=$newIds,
     ks.pendingClaimHashes=$newHashes` (old `documentIds` untouched — the old claim is still
     served);
   - **swap** (in `kb-sync`, or later in `kb-finish`), once the §6.4 reconcile query shows every
     new chunk embedded — **in rule L3's order** (below): renew, delete the old ids in
     falkor-chat, then one fenced statement
     `SET ks.documentIds=ks.pendingDocumentIds, ks.claimHashes=ks.pendingClaimHashes,
     ks.sectionHash=ks.pendingSectionHash, ks.state='synced', ks.syncedAt=$now,
     ks.pendingDocumentIds=null, ks.pendingClaimHashes=null, ks.pendingSectionHash=null,
     ks.pendingSince=null`;
   - **give up** (embed failed or cap passed): renew (L1), delete the *pending* ids in falkor-chat, then clear
     the `pending*` properties and set `state='synced'` with the **old** `sectionHash` kept — the
     old ids stay mapped and served, `kb-status` flags the section stale (file hash ≠
     `sectionHash`), and the next `kb-sync` retries.
   After a rollback, a `pending` section's `pendingDocumentIds` may name documents that no longer
   exist, or exist but were never recorded (snapshot between `POST` and the second `SET`); the
   reconcile handles the first by id, the second via the `kb-mirror` identity (plan §3.5).
2. falkor-chat's ingestion is asynchronous (chunk embedding, extraction jobs); whether a
   rolled-back `processing` document resumes is falkor-chat's behaviour — **[not verified]**.

**Per-section lease (plan §3.5, P2-M3).** Every step that changes a `:KbSection`'s mapping —
`kb-sync`, `retry-failed`, each `kb-finish` poll-and-swap — runs between an acquire and a release.
Acquire is one statement, so the per-graph write serialization **[quirks]** makes it a
compare-and-set; it also creates the section on first sight (the `MERGE` is backed by the
`KbSection.key` UNIQUE constraint, §4.2, and `$key` goes through `safe_key()`, §4.1):

```cypher
// acquire — 1 row = lease held by $token; 0 rows = held by someone else (report pending, retry in 30 s)
MERGE (ks:KbSection {key: $key})
ON CREATE SET ks.path = $path, ks.heading = $heading, ks.state = 'pending',
              ks.pendingSince = timestamp()
WITH ks
WHERE ks.leaseUntil IS NULL OR ks.leaseUntil < timestamp()
SET ks.leaseToken = $token, ks.leaseUntil = timestamp() + $ttlMs     // $ttlMs = 60000
RETURN ks.leaseToken AS token, ks.leaseUntil AS leaseUntil

// release — clears only our own lease; 0 = it had expired and was taken over
MATCH (ks:KbSection {key: $key})
WHERE ks.leaseToken = $token
SET ks.leaseToken = null, ks.leaseUntil = null
RETURN count(ks) AS released
```

**Fencing:** an expired holder must not write over the new holder, so **every** mapping write
in the flow above (the `pending` marker, the pending-ids `SET`, the swap, the give-up, the cancel
clear) is anchored the same way — `MATCH (ks:KbSection {key: $key}) WHERE ks.leaseToken = $token
SET …` — and a 0-row result means "lease lost". The release can be folded into the swap/give-up
statement itself (`… ks.leaseToken = null, ks.leaseUntil = null`) so the final write and the
release are atomic. A step that might outlive 60 s (an embed poll) re-acquires per poll iteration
rather than holding a long lease, matching plan §3.5's `kb-finish` loop.

Fencing protects the graph; the three rules below protect **falkor-chat**, whose side effects
(a `POST`, a `DELETE`) the lease cannot roll back (review P3-M1):

- **L1 — renew before every falkor-chat call.** Immediately before each `POST /documents`, each
  delete of old ids (swap), each delete of pending ids (give-up, cancel), run the fenced renew:
  ```cypher
  MATCH (ks:KbSection {key: $key})
  WHERE ks.leaseToken = $token
  SET ks.leaseUntil = timestamp() + $ttlMs
  RETURN count(ks) AS renewed
  ```
  `renewed = 0` → **stop before calling**, report the section pending. Matching on the token
  alone is correct: a lease that expired but was not taken over still carries our token and nobody
  else can hold it; a takeover replaces the token. The `pending` marker write that precedes the
  `POST` may itself carry the renew (`SET …, ks.leaseUntil = timestamp() + $ttlMs`) instead of a
  separate statement. Every falkor-chat call made under a renewed lease uses an HTTP timeout well
  under the TTL (e.g. 10 s against 60 s), so the call cannot outlive the lease it was made under.
- **L2 — clean up your own POST.** If any fenced write *after* a `POST` returns 0 rows (the
  pending-ids `SET` above all), `DELETE` the document ids that `POST` returned (a 404 counts as
  done) and only then report the section pending. Those ids were never recorded, so no other
  holder can know or serve them; deleting them is always safe and is the only way they do not end
  up current and unmapped. (A process that dies between the `POST` and that write leaves the same
  orphan; `kb-reconcile` finds it through the `kb-mirror` identity, plan §3.5.)
- **L3 — swap order.** Renew (L1) → delete the old ids in falkor-chat → the fenced swap `SET`
  (with the release folded in). Old ids are deleted only under a lease renewed just before, so an
  expired holder can never delete documents a new holder is serving; if the swap `SET` itself then
  returns 0 rows (lease lost during the deletes), stop and report pending — the new holder sees
  the old ids 404 and the pending ids still recorded, and completes the swap.

**Lease-expiry test** (for plan §5): TTL 1 s with the embed delayed past it — the stale holder's
fenced write returns 0, its submitted document is deleted (L2), and the end state is exactly one
`kb-mirror` document per section.

Verified **[live, throwaway `falkordb/falkordb:v4.18.11` container, 2026-10-22]**: 16 threads
racing the acquire on a fresh key, 5 trials → exactly 1 winner each, 1 node each, stored token =
winner's; acquire while held → 0 rows; release with a wrong token → 0, with the right token → 1
and both properties gone from `keys()`; acquire after release → 1 row; a 50 ms lease is taken over
after expiry, and the stale holder's release then returns 0.

**Cancel path — a new edit to a section that is still `pending`** (plan §3.5), under the lease,
in this order:

1. renew (L1), then `DELETE` every id in `pendingDocumentIds` through falkor-chat (a 404 counts
   as done);
2. one fenced statement clearing `pendingDocumentIds`, `pendingClaimHashes`, `pendingSectionHash`
   (state stays `pending`; `documentIds`/`sectionHash` — the old served claim — untouched);
3. submit the new text through the normal flow above (new `pendingSectionHash`, renew, `POST`,
   record the new pending ids — L1 and L2 apply).

Deleting before clearing is the safe order: if the process dies after step 1, the ids are still
recorded and the next run's step 1 re-deletes them idempotently; clearing first would orphan a
submitted document — current and searchable, with no mapping. If step 1 fails part-way, stop
with the fields intact and report the section pending.

`kb-reconcile` (run by `sync --all` and on any `pending` section) reads `ws:agent-team`
**directly through `falkordb-py`, read-only** — falkor-chat's REST surface has no chunk/embedding
state (review M4) — so this query is a coupling to falkor-chat's schema
(`Document.documentId`/`textNormalizedHash`/`currentVersion`, `HAS_CHUNK`, `Chunk.embedding`,
`falkor-chat/docs/DESIGN.md` and `repository.py`) that needs a contract test against falkor-chat's
own bootstrap:

```cypher
UNWIND $ids AS id
OPTIONAL MATCH (d:Document {documentId: id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
RETURN id, d.textNormalizedHash AS hash, d.status AS status, d.currentVersion AS current,
       count(c) AS chunks, count(c.embedding) AS embedded
```

Plan `Node By Index Scan | (d:Document)` → `Optional Conditional Traverse`; a missing id returns
`chunks 0` with null fields, and `count(c.embedding)` counts only chunks whose `vecf32` is set
**[live, throwaway container, 2026-10-21; v1's index seek also [live, EXPLAIN] on the shared
instance]**. The same query is the "embed landed" poll for `kb-sync` (`embedded = chunks > 0`).
Compare with `claimHashes`: missing → re-ingest; hash mismatch → re-ingest and delete the stale
id; **`current = false` → report, do not re-ingest** (review m8): falkor-chat auto-supersedes a
document whose normalized text equals a newer one's, so a non-current mapped id means two KB
headings hold the same claim text, and re-ingesting would just supersede the other one in turn.
`kb-sync` should also refuse to ingest a section whose `claimHash` already belongs to another
section, with a warning naming both headings. **Orphans** — documents falkor-chat holds that no section maps — cannot be
found today: **[live]** the 333 KB claims are `INGESTED_BY` agent `cobb` with `sourceKind:'agent'`,
indistinguishable from cobb's own raw-capture documents. Making them findable needs the KB sync
to ingest under a dedicated producer identity — adopted as the seeded `kb-mirror` agent, plan §3.5 (§9.3 Q2).

Keep `claude/cobb/scripts/kb-claim-manifest.json` as a **historical record** of the K-030
migration, not as live state; if a git-visible copy is wanted for review, have `kb-status` export
one on demand rather than syncing two copies.

## 7. Read queries

All reads go through `GRAPH.RO_QUERY` (the `cypher` MCP tool is fine for ad-hoc agent reads; it
is read-unrestricted). Columns are always aliased (`falkordb-py` names an unaliased column by its
expression text **[quirks]**). Every plan below was checked with `GRAPH.EXPLAIN` and
`GRAPH.PROFILE` against the real repo text loaded into a throwaway container (§8) **[live]**;
`PROFILE` executes, so it is used only on throwaway data.

### 7.1 Sync header and drift (every read)

```cypher
MATCH (s:Source)
RETURN s.path AS path, s.kind AS kind, s.sha256 AS sha256, s.syncedAt AS syncedAt,
       s.parserVersion AS parserVersion, s.warnings AS warnings
```
The CLI hashes every covered file and reports: last sync time of each source the answer uses,
the newest/oldest `syncedAt`, and the three divergence sets of §6.2. For each mismatched file it
re-parses and diffs rows against
```cypher
MATCH (s:Source {path: $path})
OPTIONAL MATCH (s)-[]->(c)
RETURN s.sha256 AS fileSha, s.syncedAt AS syncedAt, collect(properties(c)) AS kids
```
by child key: *changed* (hash differs), *added in file*, *gone from file* — and marks those rows in
the answer (AC-4/AC-10). Plan: `Node By Index Scan | (s:Source)` → `Optional Conditional
Traverse`.

### 7.2 In-flight view (FR-6, FR-9) — four reads

```cypher
// (a) units of every non-archived coordination
MATCH (l:Ledger)-[:HAS_UNIT]->(u:Unit)
WHERE coalesce(l.docStatus, '') <> 'archived'
RETURN l.component AS component, l.path AS path, l.syncedAt AS syncedAt,
       u.unit AS unit, u.owner AS owner, u.statusToken AS status, u.statusText AS statusText,
       u.agentId AS agentId, u.deliverable AS deliverable
ORDER BY path, u.tableIdx, u.rowIdx

// (b) questions waiting on the stakeholder
MATCH (u:Unit {statusToken: 'paused'})
MATCH (l:Ledger {path: u.ledgerPath})
WHERE coalesce(l.docStatus, '') <> 'archived'
RETURN l.component AS component, l.path AS path, u.unit AS unit, u.statusText AS statusText,
       u.question AS question, u.askedOn AS askedOn, l.syncedAt AS syncedAt
ORDER BY askedOn

// (c) open "needs changes" verdicts, minus those closed in a ledger (plan §3.4a(c))
MATCH (r:Review)
WHERE coalesce(r.docStatus, '') <> 'archived' AND r.verdictClass = 'needs-changes'
OPTIONAL MATCH (l:Ledger)
WHERE r.path IN l.resolvedReviews
   OR (l.docStatus = 'archived' AND r.path IN l.citedReviews)
WITH r, collect(l.path) AS closedBy
WHERE size(closedBy) = 0
RETURN r.component AS component, r.path AS path, r.latestPass AS pass,
       r.latestVerdict AS verdict, r.syncedAt AS syncedAt

// (d) requirements not yet picked up
MATCH (q:Requirement)
WHERE q.docStatus IN ['Interviewing', 'Ready for design']
OPTIONAL MATCH (l:Ledger {slug: q.slug})
WITH q, l WHERE l IS NULL
RETURN q.component AS component, q.path AS path, q.docStatus AS status, q.syncedAt AS syncedAt
```

- `coalesce(…, '') <> 'archived'`, never a bare `<>`: a ledger with no `Status:` header has an
  absent `docStatus`, and `null <> 'archived'` is `null` — the row would silently vanish
  **[quirks]** cross-type/NULL comparison. AC-7 holds by construction: the filter is on the
  document's `Status:`, never on the units' states.
- (d) counts a requirement as picked up by **any** ledger with the same slug, archived or not
  (an archived ledger is still a `:Ledger` node, §2.2) — that is what makes FR-9's "leaves once
  picked up by a coordination or archived" answerable.
- (b) filters on the normalized `statusToken` (§2.5), so `**relayed**` and `**paused** — …` cells
  are found; the raw cell is returned as `statusText` for display.
- (c) implements plan §3.4a(c): a needs-changes review is **resolved** when a ledger row citing it
  is `accepted`/`abandoned` (`resolvedReviews`), or when any ledger citing it is archived
  (`citedReviews` on the archived stub, §2.2). It is an anti-join by `OPTIONAL MATCH` +
  `collect` + `size() = 0` — the `exists()`/`EXISTS {}` forms are unusable on this build
  **[quirks]**. `docStatus = 'archived'` needs no `coalesce` here: a ledger without a header reads
  `null`, which correctly does not count as archived. Neither list is ever absent (§2.2), so `IN`
  never sees `null`.
- Plans **[live]**: (a) `Node By Label Scan | (l:Ledger)` (72) → `Filter` (2) → traverse (8 units);
  (b) `Node By Index Scan | (u:Unit)` on `statusToken` + `Value Hash Join` with the ledger scan
  (re-checked for v2 on a throwaway container); (c) label scan of 168 reviews → `Filter` (~10 needs-changes)
  × `Apply`/`Optional` label scan of ~74 ledgers with a list-membership `Filter` — ~750 records,
  no index applies to list membership and none is needed at this size; v2 form verified on a
  throwaway container against a six-review fixture covering every branch (resolved by an
  accepted row, cited only by an archived ledger, cited by an in-flight row, uncited, review
  itself archived, not needs-changes → exactly the in-flight-cited and uncited reviews returned);
  (d) label scan of 43 requirements + per-row `Node By Index Scan | (l:Ledger)` on `slug`. The v1
  operators measured ≤ 0.07 ms each on the real text.
- What survives (c)'s rule is still only as good as `verdictClass` and the ledgers' citations;
  plan §6 R3 triages the remainder once.

### 7.3 Single-unit lookup (FR-8, AC-6)

```cypher
MATCH (l:Ledger {slug: $slug})-[:HAS_UNIT]->(u:Unit)
WHERE u.unitKey = $unit
RETURN l.path AS path, l.sha256 AS fileSha, l.syncedAt AS syncedAt, u.key AS key,
       u.statusToken AS status, u.statusText AS statusText, u.agentId AS agentId,
       u.deliverable AS deliverable, u.question AS question, u.askedOn AS askedOn,
       u.rowHash AS rowHash
ORDER BY path, u.tableIdx, u.rowIdx
```
`$slug` and `$unit` go through `norm_value()` (§4.1), and the CLI derives `$unit` with the same
cleaning as the stored `unitKey` (emphasis/backticks stripped, cut at the first ` — `). `unitKey`
is a stored property (§2.2) and carries no `~n` suffix, so a repeated label returns **every**
repeat, each with its own `key` — the CLI shows them all, in table order, rather than guessing.
Plan: `Node By Index Scan | (l:Ledger)` → `Conditional Traverse` → `Filter` **[live, v2 form on a
throwaway container: `U32` returned both `#U32` and `#U32~2`; `U1` returned one row]**. Two ledgers
in different components could share a slug — the CLI shows both, with their paths. The CLI hashes
just the matched file(s) for the drift flag.

### 7.4 What's planned (FR-12, AC-9)

```cypher
MATCH (b:Backlog {scope: $scope})-[:HAS_ITEM]->(i:Item)
WHERE i.open = true
RETURN b.path AS path, b.syncedAt AS syncedAt, i.ref AS ref, i.statusMarker AS marker,
       i.title AS title, i.priority AS priority, i.milestone AS milestone
ORDER BY i.ord
```
`Node By Index Scan | (b:Backlog)` → traverse; 156 rows for `cobb` in ~0.2 ms total **[live]**.
AC-9's "none of them in the in-flight view" holds structurally: the in-flight reads never touch
`:Item`.

### 7.5 Milestone-proposal facts (FR-13, FR-14)

```cypher
MATCH (b:Backlog {scope: $scope})-[:HAS_ITEM]->(i:Item)
WHERE i.open = true
UNWIND (CASE WHEN size(coalesce(i.dependsOn, [])) = 0 THEN [null] ELSE i.dependsOn END) AS dep
OPTIONAL MATCH (d:Item {ref: dep})
WITH b, i, collect(CASE WHEN d IS NULL THEN null
                   ELSE {ref: d.ref, scope: d.scope, open: d.open, backlogPath: d.backlogPath} END) AS resolved
RETURN b.syncedAt AS syncedAt, i.ref AS ref, i.statusMarker AS marker, i.title AS title,
       i.priority AS priority, i.milestone AS milestone, i.dependsOn AS dependsOn, i.touches AS touches, resolved,
       [r IN coalesce(i.dependsOn, []) WHERE NOT r IN [x IN resolved | x.ref]] AS unresolved,
       [f IN ['dependsOn', 'priority', 'touches'] WHERE i[f] IS NULL] AS missing
ORDER BY i.ord
```
- The `[null]` guard is the empty-`UNWIND` collapse fix **[quirks]** — without it every item with
  no dependencies would vanish from the result. The `CASE … null` inside `collect` avoids the
  one-map-of-nulls artifact of collecting a map over an empty `OPTIONAL MATCH` **[quirks]**.
- **Tuning that matters [live]:** the first draft, `OPTIONAL MATCH (d:Item) WHERE d.ref IN
  coalesce(i.dependsOn, [])`, planned as a **`Node By Label Scan | (d:Item)` per item — 104,676
  records, 13 ms**; the `UNWIND` + `{ref: dep}` form plans `Node By Index Scan | (d:Item)` —
  **107 records, 0.2 ms**. Same answer; ~100× less work, and it is the shape that stays flat as
  backlogs grow.
- `missing` lists exactly the FR-14 facts not recorded (absent property), so `[]` recorded as "no
  dependencies" counts as recorded (§2.2) — AC-12's "not plannable yet, with what's missing".
- `resolved[].open = true` with a different `scope` is an **external blocker** (AC-13);
  `unresolved` is "not in any open backlog"; plan §3.7 step 2 decides what that means (§9.3 Q1).
- "Unblocks the most items" counts **same-backlog dependents only** (plan §3.7, review m7), so
  `propose` needs no extra query: the dependents are the inverse of the `dependsOn` lists already
  returned above, computed client-side. The cross-backlog form below is **display-only** (an
  "also waited on by" line in the proposal), never a ranking input:
  ```cypher
  MATCH (o:Item)
  WHERE o.open = true AND o.dependsOn IS NOT NULL
  UNWIND o.dependsOn AS dep
  WITH dep, collect(o.ref) AS dependents
  WHERE dep STARTS WITH ($scope + ':')
  RETURN dep, dependents
  ```
  (parenthesised concatenated prefix **[quirks]**). A label scan of ~700 items, 0.3 ms **[live]**;
  list membership cannot be indexed, and at this size it does not need to be.
- **Ordering, overlap and grouping are computed client-side**, from these facts: a topological
  sort with priority tie-break and the overlap test — **its rules are plan §3.7 step 4's**, not
  this note's (in particular glob vs glob is the conservative literal-prefix rule there: `fnmatch`
  of one glob against another is not an intersection test) — over ≤ ~150 items is trivial in Python and keeps the explanation of *which criterion
  drove each placement* (FR-13) in the code that made the decision. No `algo.*` procedure fits
  better.

### 7.6 Cost summary

Every view is one to four `RO_QUERY`s whose operators each run in ≤ 0.4 ms on the real data
volume; the dominant cost of any answer is the CLI hashing ~300 files for the drift header (a few
MB of text — milliseconds), not the graph. Reads do not serialize against each other **[quirks]**; how long a
read can wait behind a commit is not measured here **[inferred]** negligible at these write sizes.

## 8. Memory footprint

Measured **[live, throwaway container, `INFO memory` `used_memory` delta]** by loading the real repo
text (cells, review verdicts, full item bodies) with a crude stand-in parser:

| Load | Nodes | `used_memory` delta |
|---|---|---|
| Design scope: all 301 `:Source`s, units of non-archived ledgers only (8), 671 item candidates incl. full body text | ~980 | **2.9 MB** (incl. index/schema overhead) |
| Worst case: units of **all** 72 ledgers (945 rows, cells up to 5.3 KB) | ~1,900 | +5.2 MB more |

(`GRAPH.MEMORY USAGE` rounds the first case to `total_graph_sz_mb 1`; it is known to under-report
index memory **[quirks]**, hence `INFO` deltas.) Projection: **~3–5 MB steady state, < 10 MB even
if every ledger went live at once** — against 376 MB in use today (dominated by `cpg_falkorchat`,
255 MB **[live]**), i.e. ~1–2 %. The KB state adds ~400 small `:KbSection` nodes (< 1 MB). No
sizing action needed; `maxmemory 0`/`noeviction` stays correct for a store that must never evict
its own keys. Growth is bounded by the forward-only backlog rule and by archived ledgers dropping
their units.

## 9. Verified vs inferred; incident; open questions

### 9.1 Incident during this consult — the shared FalkorDB was crashed once

At **2026-10-20 ~21:00 -03** a worst-case load probe (real ledger text into a scratch graph on the
shared `falkordb-dev`) segfaulted the instance. Root cause, isolated afterwards in throwaway
containers only: a Unit key cut from a Markdown cell at a delimiter ended in a **space**, and a
whitespace-edged value in a `UNIQUE` property crashes `EnforceUniqueEntity` (§4.1). Impact:

- the `--rm` container exited and vanished; restored with the repo's own additive bring-up,
  `claude/teco/hooks/ensure-environment.sh` (it ran `falkor-chat/scripts/start_falkordb.sh -d`);
  `PONG` and `ws:agent-team` `/health` 200 afterwards; all 9 pre-existing graph keys back,
  `ws:agent-team` at 367 `Document`s as before;
- the restart loaded the RDB taken at **20:54:18** — every write between then and ~21:00 was lost.
  At 20:59:09 `rdb_changes_since_last_save` was 25, all of which I can attribute to my own probe
  writes; **I cannot rule out another session's writes between 20:59 and the crash.** Anyone who
  wrote to FalkorDB in that window should re-check.
- New quirk recorded in `claude/graph-dba/falkordb-quirks.md` (Indexing section). It also bears on
  falkor-chat: any `UNIQUE` property fed with caller-supplied strings needs the same strip guard as
  the 4,096-byte one (K-049) — worth a falkor-chat backlog item (not filed by me).

### 9.2 Claim ledger

| Claim | Status |
|---|---|
| engine v4.18.11 / module `41811`, Redis 8.6.3, `vectorset` loaded | **[live]** `MODULE LIST`, `INFO server` |
| `appendonly no`, `save 3600 1 300 100 60 10000`, `noeviction`, 376 MB used | **[live]** `CONFIG GET`, `INFO memory` |
| one graph write = one RDB change; first write after > 1 h idle triggers a save | **[live]** `INFO persistence` deltas |
| single-statement delete+recreate works, incl. same UNIQUE keys; empty `$rows` keeps the `:Source` | **[live]** |
| replace is atomic — a constraint violation rolls back the deletes too | **[live]** |
| readers never saw a partial replace; concurrent writers serialize, last commit wins | **[live]** 104 reads / 8 writes; consistent with **[docs]** concurrency page — a strong sample, not a proof |
| 4.1 MB parameter list is fine | **[live]**; the 512 MB bulk limit is **[docs]**, not probed |
| `null` map values not stored; `[]` stored distinctly from absent | **[live]** |
| re-running `CREATE INDEX` / `GRAPH.CONSTRAINT CREATE` errors | **[live]** |
| whitespace-edged UNIQUE value crashes the instance; index lookup misses it | **[live]**, throwaway containers |
| every plan in §7 and the 100× dependency-seek fix | **[live]** EXPLAIN + PROFILE on real text, throwaway container |
| memory 2.9 MB design scope, +5.2 MB worst case | **[live]** `INFO memory` deltas, throwaway container |
| RDB fork snapshot is point-in-time across all keys incl. graph keys | **[docs]** for Redis; **[inferred]** for module keys |
| falkor-chat resumes or abandons a rolled-back `processing` document | **not verified** |
| KB claims are indistinguishable from cobb's raw captures in `ws:agent-team` | **[live]** 333 docs `INGESTED_BY` `cobb`, `sourceKind:'agent'` |
| `ws:` prefix is falkor-chat's workspace namespace | **[live]** code read (`config.py:268`, tests diff `GRAPH.LIST`) |
| v2 queries §7.2(b), §7.2(c), §7.3 return the intended rows and plan as stated | **[live]** throwaway container (port 6391, no volume), 2026-10-21 |
| UNIQUE crash set is exactly C `isspace()` (`\n \r \v \f` too); NBSP/Unicode spaces safe | **[live]** throwaway containers, 2026-10-21 |
| `count(c.embedding)` counts only chunks with a `vecf32` set; reconcile query plans an index seek | **[live]** throwaway container, 2026-10-21 |
| per-section lease CAS acquire/release is race-safe; `SET prop = null` removes the property; `timestamp()` works | **[live]** throwaway container, 2026-10-22 |
| fenced renew (L1): extends in time, extends an expired-but-untaken lease, returns 0 after a takeover; a renewed lease blocks a competing acquire | **[live]** throwaway container, 2026-10-22 |
| status normalization tally (§2.5) | offline parse of the repo, no database, 2026-10-21 |
| falkor-chat's `HAS_CHUNK`/`Chunk.embedding`/`currentVersion` schema | code read (`DESIGN.md` l.204, `repository.py`) — needs the §6.4 contract test |

Scratch data: `scratch:graphdba-coordprobe` was deleted with `GRAPH.DELETE` (confirmed absent from
`GRAPH.LIST`); `scratch:graphdba-coordprobe2` and `scratch:graphdba-coordreal` were never persisted
(lost in the crash). The throwaway containers (`graphdba-crashprobe`, `graphdba-memprobe`, port
6390, no volumes) are removed. The shared instance holds nothing of mine.

### 9.3 Open questions (scope-affecting — not decided here)

- **Q1 — answered** by plan §3.7 step 2: an unresolved `dependsOn` ref counts as delivered only
  on a delivery-shaped match in the referenced backlog's `HISTORY.md`; no match → not plannable.
- **Q2 — answered** by plan §3.5: KB claims are ingested under a dedicated, seeded `kb-mirror`
  producer identity (seeded in plan S8a), which makes orphaned claim documents findable (§6.4).
- **Q3 — AOF.** RDB-only is sound for the mirror *because* drift is flagged (FR-5); it does not
  make FR-4 hold across a crash. If the stakeholder wants that, `appendonly yes` +
  `appendfsync everysec` closes the window to ~1 s at negligible cost at this write rate — a
  stakeholder/`devops` decision, not a design need.
- **Q4 — answered** by plan §3.4a(c) (resolved when a citing ledger row is `accepted`/`abandoned`
  or the citing ledger is archived); modelled in §2.2 and queried in §7.2(c).
- **Q5 — "picked up" is a global slug match.** Root `AGENTS.md` rule 3 forbids reusing a slug for
  a different topic, so a same-slug ledger in another component is assumed to be the same topic;
  if that is ever false, a requirement silently leaves the view. Low risk; flagged.
