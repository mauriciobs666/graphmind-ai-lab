# coordination-graph — Design

> **Status:** active · **Owner:** `coder` · **Tracks:** —

> **How to read this document.** It is part of coordination-graph's live spec and states the
> component **as it is now**: what is mirrored, how sources are discovered and identified, the sync
> and parity semantics, the CLI, the views and `propose`, the shared-instance safety model, the
> post-edit hook, the KB mirror and lint. Data formats (source grammars, the graph schema, key rules,
> KB mapping nodes, hook state files) are in `coordination-graph/docs/FORMATS.md`, read by lookup.
> Change history is in `coordination-graph/docs/HISTORY.md`. Section numbers are stable and cited
> from `coordination-graph/AGENTS.md`.

## Index

| § | Topic |
|---|---|
| 1 | Purpose, truth model, what is mirrored |
| 2 | Discovery, component, slug and scope |
| 3 | Work-item identity (read side) |
| 4 | Sync, rebuild, parity, drift, the sync header |
| 5 | CLI: global rules, exit codes, each command |
| 6 | Views and `propose` |
| 7 | The shared-instance safety model |
| 8 | The post-edit hook |
| 9 | The KB mirror |
| 10 | Lint, dry-run, migrate, history |
| 11 | Configuration |
| 12 | Surface index |

## 1. Purpose, truth model, what is mirrored

`coordgraph` mirrors the repo's coordination files into a FalkorDB graph and answers read views over
it: what is in flight, what waits on the stakeholder, one unit's status, what is planned, a
milestone proposal. It also mirrors registered knowledge-base files into falkor-chat's
`ws:agent-team` workspace.

- **The files are the truth; the graph is a read-only mirror**, never written by hand. A view
  that disagrees with a file is stale: the file wins.
- **Sources mirrored into the graph** (four kinds): teco coordination ledgers (`ledger`), reviews
  (`review`), requirements (`requirement`), backlogs (`backlog`: component `BACKLOG.md` files and
  agent `kaizen/plan.md`). History files are never mirrored; they are read from disk by `lint`,
  `history-coverage` and `propose`.
- **Sources mirrored into falkor-chat** (not the coordination graph): the KB files listed in
  `coordination-graph/kb-files.txt` (§9). Only their mapping lives in the graph (FORMATS §9).
- **A source's subgraph is a pure function of its file.** Nothing stored depends on the work-item
  registry, the clock or another file (a ledger's cited-review resolution reads the set of covered
  review paths, which is also a function of the tree).
- **Not features:** no write path to the coordination files, no scheduling that takes effect (a
  proposal is text), no stored identity resolution, no mirror of history files or the work-item
  registry.
- Python 3.12, one runtime dependency (`falkordb`, imported lazily; offline commands never load it).

## 2. Discovery, component, slug and scope

**Coverage** (`coordgraph/discover.py`): one glob set, `**` also matching zero directories, hidden
files and directories never matched:

| Path shape | Kind |
|---|---|
| `**/docs/plans/*-coordination.md` | ledger |
| `**/docs/reviews/*.md` | review |
| `**/docs/requirements/*.md` | requirement |
| `**/docs/BACKLOG.md` | backlog |
| `claude/*/kaizen/plan.md` | backlog |

Excluded: `deprecated/**` at the repo root and any path with an `archive/` segment. `.git`,
`node_modules` and `.venv` are pruned while walking. The hook's shell re-implements the same shapes
and a test ties the two together.

- **`slug`.** The file name minus `.md` minus one role suffix the **kind** can carry: ledger
  `-coordination`; review `-ml`, `-rca`, `-impl`; requirement and backlog none. So
  `requirements/coordination-graph.md` has slug `coordination-graph`, the join key with its ledger.
- **`component`.** The first path segment, except the repo-root `docs/` tree: `docs/BACKLOG.md` is
  `cpg`, anything else under root `docs/` is `root`.
- **`scope`** (backlogs only). From `.work-items/project.json` `legacy_paths` (FORMATS §5); the
  semantics are those of `wilib.check._home_of`, re-implemented in `coordgraph/scope.py` so the sync
  path stays free of `wilib`; a test pins the two together.
- **Archived** means the header `Status:` token is exactly `archived`. Archived ledgers keep minimal
  units (FORMATS §2); archived sources stay in the graph and are filtered by the views.

## 3. Work-item identity (read side)

Every ID in a source file is stored **as written**. What a stored string *denotes* is decided at
**read time**, in `coordgraph/identity.py`, the one module that may import `wilib` (lazily, from
`COORDGRAPH_WI_ROOT`, default `<repo>/skills/work-items/scripts`). **The sync path never imports
`wilib` or `identity`** (`sync`, `rebuild`, `kbsync`, `commands_kb`, `commands_hook`,
`hookstate`): the hook stays offline and a bad `COORDGRAPH_WI_ROOT` cannot affect freshness. A test
enforces it. Readers of identity: `lint`, `history-coverage`, the `Registry:` line of `drift` and
`query`, the views, `propose`.

**Shape routing** (`coordgraph/shapes.py`; shapes only pick a resolver, they never validate):

| Shape | Route |
|---|---|
| `[A-Z]-\d+` | `legacy-bare`: qualified with the referrer's scope (`<scope>:<token>`) when it has one |
| `<home>:[A-Z]-\d+` | `legacy-qualified`: as written |
| `[A-Z]{2,6}` then `-<letter><digits>`, `-FR<n>[a-z]`, `-AC<n>[a-z]` segments | `scheme`: never scope-qualified |
| in a `Needs` cell only, `<slug>:<unit>` | `unit` |
| anything else | unrecognised (`unknown`) |

**Resolution.** `Identity.resolve(token, scope)` returns a state and a candidate set. States:
`resolved`, `unminted`, `ambiguous`, `unknown`, `retired`, `unavailable` (registry or `.work-items/`
unreadable, or `wilib` not importable; this never raises). `candidates` are the strings under which
that identity may be stored: the item's `id`, `uid`, aliases, its legacy `home:name` when minted, and
the token as written, except that a bare legacy token the referrer's scope qualified is denoted by
the **qualified** name alone (the bare string would also match another home's item). Two references
denote one item iff their candidate sets intersect. Without a registry, candidates are the token as
written (qualified when a scope applies).

**Joins are `IN $candidates` on indexed raw strings** (`Item.ref`, `Source.itemId`, `Unit.itemId`),
so a rename after `wi move` joins without any stored string changing. A stored **bare** legacy
string no longer joins a qualified reference: legacy IDs are written qualified or as the scheme ID.

**The sync header's `Registry:` line** reports the registry's item count, the last event date and
the unflushed journal events, or `Registry: unavailable — identities compared as written` (§4.6).

**Need resolution** (`identity.resolve_need`, a library contract over fetched rows; no CLI command
calls it). A `Needs` token is `met`, `unmet` or `unresolvable`; a need never copies status:

- a task ID: units whose `itemId` is a candidate; exactly one unit: `accepted` is met, `unknown` is
  unresolvable (the reason carries the raw cell), anything else unmet; several units are
  unresolvable (`ambiguous`);
- a feature ID: ledgers whose `Source.itemId` is a candidate; met iff every such ledger is
  `archived`, or is not archived and has at least one unit with all units `accepted`; none carrying
  it is unresolvable;
- `<ledger-slug>:<unit>`: `<ledger-slug>` is the **graph's per-kind slug** (§2: `connected-code-graph`,
  never the basename `connected-code-graph-coordination`); a bare token equal to a unit label of the
  referring ledger is a local unit;
- an unresolved token (unminted, unknown, ambiguous, unavailable) is unresolvable with the reason.

**One unit-lookup rule** (`identity.select_units`, shared by need resolution and the `unit` view): a
trailing `~n` addresses one repeat by its `key`; otherwise rows whose `unitKey` equals the reference
win (every repeat of that key); only when none exists do `unitLabel` matches count. Neither stored
property carries the `~n` suffix, so a trailing `~<digits>` is removed before the query (a key that
itself ends in `~n` is matched exactly first).

A unit's `Needs` cell as a whole: absent or malformed is `not-recorded` (never ready); any unmet need
blocks; otherwise any unresolvable need is `unresolvable`; else `ready`.

## 4. Sync, rebuild, parity, drift, the sync header

### 4.1 Per-source replace

A sync is: parse the file, then one atomic statement replacing that source's subgraph (FORMATS §7.6),
then re-hash the file; if it moved, re-parse and write again, up to 3 rounds (`ConvergenceError`
otherwise). This closes the stale-parse race without a lock. A failed replace leaves the previous
subgraph. A failure on one path never stops the others. A backlog with a duplicate item `ref`
fails to parse and its source is reported `failed`; a cross-file duplicate `Item.ref` is a unique
constraint violation named with both paths (a `wi move` may be in progress).

### 4.2 `PARSER_VERSION`

`coordgraph/parse/source.py` `PARSER_VERSION` is 2. A source is skipped when its stored `sha256` and
`parserVersion` both equal the file's and the current version. Bumping it makes the next `sync
--all` rewrite every source and makes `drift` show `stale-parser`.

### 4.3 `sync`

Each named path (or every covered path for `--changed` / `--all`) yields one outcome:
`synced` (with `<n> rounds` when more than one), `unchanged` (hash and version equal),
`deleted` (a path no longer covered whose file is gone and which the graph holds), `failed`
(parse error, refused value, I/O error) or `skipped` (`not a covered file`). `--all` adds a prune:
every stored source that is not a covered file on disk is deleted.

### 4.4 `rebuild`

Bootstrap (§5), then an unconditional replace of every covered source, prune, an **orphan sweep**
(`Unit` and `Item` nodes no `:Source` owns, deleted and reported as a bug signal), `SyncState`
stamped (`lastRebuildAt`, `parserVersion`), then parity.

### 4.5 Parity

A fresh parse of every covered file is put in the form the store writes (NULs scrubbed, `None`s
dropped, keys asserted) and compared with one export of the whole graph, **never the graph with
itself**. Compared per source: kind, `docStatus`, `sha256`, `rowCount`, labels, the property hash
(volatile `syncedAt`/`mtimeNs` excluded), and per child its key, property hash, relationship type and
labels. Reported differences: `missing in graph`, `in graph, not on disk`, `does not parse`, a
differing property, a child missing / extra / differing. An export whose row count differs from the
source count is an error (a capped reply looks complete). `rebuild` exits 0 only when parity holds,
no source failed and no orphan was swept.

### 4.6 Drift and the sync header

`drift.check` compares each covered file with its stored `sha256` and `parserVersion`, never repairs:
`changed`, `unsynced` (file not in graph), `gone` (in graph, file missing), `stale-parser` (same
bytes, older `parserVersion`). A `changed` source is re-parsed and diffed against its stored children
by row key (`changed` / `added in file` / `removed from file`); an unparseable one carries `does not
parse`. Every read subcommand prints the header first:

```
Synced: <oldest>Z – <newest>Z (<n> sources) · Hook: registered|not registered|disabled
Registry: <n> items · last event <date> · <m> unflushed journal events
Drift: none
```

- `Synced`: oldest and newest `syncedAt` over the covered sources; `never` when a source has none.
- `Hook`: `disabled` when the kill switch is on (§8.6); else `registered` only when the project's
  `.claude/settings.json` / `settings.local.json` register the hook (below); else `not registered`,
  including for unreadable or garbled settings. It never over-claims. Rules (`coordgraph/hookreg.py`):
  - both `PostToolUse` and `PostToolUseFailure` need a registration whose matcher covers Write, Edit
    and Bash (tokens unioned across entries and across the two files). An empty or `*` matcher covers
    all; a name list of letters, digits, `_`, `-` split on `|` or `,` (spaces tolerated) is exact;
    anything else is a regular expression judged as a full match of each tool name (an invalid one
    covers nothing);
  - the handler must be `type: command` (or no `type`) and carry **no `if` key**;
  - its command must name `coordination-graph/hooks/post-tool-sync.sh` as the executable, or as the
    argument of `bash`/`sh`, by an **absolute path of this checkout** or a `$CLAUDE_PROJECT_DIR` form
    (`$CLAUDE_PROJECT_DIR` and `${CLAUDE_PROJECT_DIR}` are expanded). A relative path is unproven (it
    resolves against the session cwd) and does not count; nor do a `.bak` suffix, `echo <path>`,
    `bash -c`, env/`timeout` prefixes, another checkout or a `..` detour;
  - the script file must exist, and in the direct form must be executable (the `bash`/`sh <script>`
    form needs it to exist only);
  - `disableAllHooks: true` gives `not registered`; the local file overrides the project file when it
    sets the key;
  - user-scope (`~/.claude/settings.json`) and managed settings are not readable from the project and
    are not judged, so `registered` is a static reading of the project files only and does not prove
    the hook fires in a given context.
  The executable statement of the cases is the golden table
  `coordination-graph/tests/fixtures/hook-settings-cases.json` (57 cases, each with settings, the
  expected state and the reason), exercised by `coordination-graph/tests/test_hookreg.py`.
- `Drift: <n> source(s)` is followed, per source, by its state, path and `→ file wins; run coordgraph
  sync` (plus the note), and per row `⚠ drift  <key>  <what>`. A `⚠ drift` row marks a view row whose
  file moved since the sync: **the graph is stale there; read the file and quote it.**

## 5. CLI

`coordgraph <sub>` abbreviates `coordination-graph/coordgraph.sh <sub>` (it exits 1 naming `setup.sh`
when there is no venv; `python -m coordgraph` works from the component's venv). Subcommands, as
`--help` lists them:

**Offline** (no database):

| Command | Behavior |
|---|---|
| `lint [--count] [PATH...]` | §10.1. `--count` prints items per backlog file, then `total`; a file that does not parse counts -1 and the exit is 1 |
| `migrate (--dry-run \| --apply) [--titles-reviewed] [PATH...]` | §10.3 |
| `facts [--like TEXT]` | `fact:` tokens of every `Touches` list, as `token<TAB>mode<TAB>ref<TAB>path`, sorted, de-duplicated |
| `history-coverage [-v]` | per history file: tokens mentioned, still open, matched, unmatched, rate; `-v` lists the unmatched |
| `rebuild --dry-run [--manifest FILE]` | §10.2 |

**Store-backed.** Each takes `--graph KEY` (default `COORDGRAPH_KEY`, itself default `coordination`)
and `--confirm GRAPHKEY` (§7). `bootstrap`, `sync`, `rebuild`, `drift`, `query`, `kb-*`, `hook-sync`,
`retry-failed` take both explicitly; the views and `propose` take `--confirm` and `--graph` too.

| Command | Behavior |
|---|---|
| `bootstrap` | create the indexes and UNIQUE constraints of FORMATS §7.5 that are missing, poll to OPERATIONAL (5 s), then `SyncState` (`schemaVersion`); idempotent; prints `created ...` lines and `bootstrap: done` or `nothing to do` |
| `sync [--changed \| --all] [-v] [PATH...]` | §4.3; prints each outcome except `unchanged` (`-v` includes it); a summary line on stderr |
| `rebuild` | §4.4; prints the outcomes `failed`/`deleted`, the per-status counts, orphans and the parity verdict |
| `drift [PATH...]` | prints the header; exit 1 when any covered file differs |
| `query CYPHER [--covers PATH...]` | `GRAPH.RO_QUERY` of the given Cypher under the header (drift limited to `--covers` when given), then a tab-separated header row and rows; a write is refused by the read-only call |
| `inflight [--waiting]`, `unit SLUG_OR_ID [UNIT]`, `planned COMPONENT_OR_AGENT [--all]`, `propose COMPONENT_OR_AGENT [--per-milestone N]` | §6 |
| `kb-sync [-v] [--detach-finish] [PATH...]`, `kb-status [--ids ID...]`, `kb-reconcile`, `kb-finish [--once]` | §9 |
| `hook-sync [--first-run [--stamp-at EPOCH]] [PATH...]`, `retry-failed` | §8 (called by the hook script, not by hand) |

PATH arguments may be absolute (made repo-relative) or repo-relative.

**Exit codes.**

| Command | 0 | 1 | 2 |
|---|---|---|---|
| `lint` | no error finding (warnings do not fail) | an error finding | |
| `migrate` | ok | a failed write | bad usage (no mode, `--apply` without exactly one covered backlog PATH, a missing file, or titles needing review) |
| `facts`, `history-coverage` | always | | |
| `bootstrap`, `sync`, `rebuild`, `propose` | ok | refused or unreachable endpoint, a failed source, failed parity, no synced backlog for `propose`, `propose --per-milestone` below 1 | |
| `rebuild --dry-run` | no violation | a violation | |
| `drift` | no drift | drift, or a failure | |
| `query` | ok | failure | |
| `inflight` | ok | failure | |
| `unit`, `planned` | something found | nothing found, or a failure | |
| `kb-sync`, `kb-finish`, `kb-reconcile` | settled | the command could not run | a section pending, refused or failed (`kb-reconcile`: any result to review) |
| `kb-status` | nothing flagged (`coarse`/`degraded` are informational) | could not run | anything flagged |
| `hook-sync` | synced, or nothing to do | | a file failed or is pending, or the guard refused |
| `retry-failed` | always | | |

A failure prints one line to stderr (`<command>: <reason>`), never a traceback; `inflight`, `unit` and `planned` over an
absent mirror add `(no synced mirror: run `coordgraph bootstrap` and `coordgraph sync`)`; `propose`, `drift` and
`query` print the server's own error.

## 6. Views and `propose`

Every view statement, the header's drift reads included, runs through `GRAPH.RO_QUERY`, so a view can never write (an absent key errors instead of being created); every answer prints
the sync header (§4.6) first, with `⚠ drift` on each row whose file row no longer matches the graph.
Identity is resolved at read time (§3); the slug join is never skipped.

### 6.1 `inflight [--waiting]`

Sections, in order: **Units in open coordinations** (units of every ledger not headed `archived`,
grouped by ledger, in table order: unit, owner, status token with the raw cell beside it when it
adds anything, agent ID, deliverable clipped to 140 characters); **Waiting on the stakeholder**
(units with `statusToken` `paused`, from non-archived ledgers, ordered by `askedOn`: component,
path, unit, `asked <date>` or `asked unknown`, the question); **Open "needs changes" verdicts**
(non-archived reviews with `verdictClass` `needs-changes` that no ledger closes: a ledger closes one
through `resolvedReviews`, an archived ledger through `citedReviews`); **Requirements not yet picked
up** (requirements with `docStatus` `Interviewing` or `Ready for design` with no ledger sharing their
slug and none whose `Source.itemId` is a candidate of the requirement's `Item:`, resolved in the
requirement's component scope). `--waiting` prints only the stakeholder section.

### 6.2 `unit SLUG_OR_ID [UNIT]`

Finds units by the ledger slug (always) united by `key` with, when the argument has an ID shape, the
ledgers whose `Source.itemId` is a candidate and the units whose `itemId` is a candidate (task ID).
`UNIT`, a key or label (`U3`, `U32~2`), filters by the rule of §3. Output per ledger path: its
`docStatus`, then per unit its key, status (raw cell beside the token when it adds anything), agent,
deliverable clipped to 600 characters, and for a paused unit the question and `asked` date. An
archived ledger's units show only key, label and status. More than one hit ends with `<n> rows
match; cite one by its key or by task ID`. An ID argument that does not resolve prints a note with
its state and reason and never guesses. No hit: `no unit matches ...`, exit 1.

### 6.3 `planned COMPONENT_OR_AGENT [--all]`

The open items (`open` true) of every backlog with that `scope`, in file order: ref, status marker,
priority, milestone, title, path. Deferred and rejected items (marker `⚪`, or the word deferred or
rejected) are parked and listed in a separate count, in full only with `--all`. An unknown scope
prints `no backlog with scope ...; known scopes: ...`, exit 1.

### 6.4 `propose COMPONENT_OR_AGENT [--per-milestone N]`

A **read-only, pure planner** (`coordgraph/propose.py`; the adapter in
`coordgraph/commands_propose.py` issues `RO_QUERY` reads and file reads only). `propose(items,
resolver, per_milestone)` has no store, network, clock or file access, so one input always yields one
output; a proposal never takes effect. `N` defaults to 6 and must be at least 1.

- **Input.** The open items of the scope's backlog (`Priority`, `Depends on`, `Touches`,
  `Milestone`, status) and, per distinct `Depends on` token, a resolution: the candidates (§3),
  whether it is open in another backlog (from the graph), or delivered (a delivery-shaped heading in
  the owning history file, §10.4), else a reason it is unknown.
- **Classification.** An item with a `Milestone` is **fixed** and placed as recorded (ordered by its
  `order`, then file order). Free items: `deferred`/`rejected` are listed apart; `in-progress` are
  **carried**; the rest are plannable only when `Priority`, `Depends on` and `Touches` are all
  recorded and no dependency is unknown; otherwise **not plannable**, naming what is missing or the
  unknown token with its reason.
- **Dependencies.** A token resolving to an open item of the same backlog is internal; open in
  another backlog is an **external blocker** (the item is **held**); delivered is satisfied; a
  dependency cycle (including a self-dependency) makes its members not plannable. An item waiting on
  a deferred, held or not-plannable item is held.
- **Ranking.** By effective priority (the maximum of its own and its transitive dependents'), then
  transitive-dependent count, own priority, file order.
- **Waves and milestones.** Items are scheduled in waves: an item enters a wave once every internal
  dependency in the schedule sits in an earlier wave (a carried dependency forces wave 2 or later)
  and its `Touches` do not overlap a carried item (wave 1) or an item already in that wave.
  Overlap is: a `file:` entry covering another (path, directory or glob; glob against glob is
  conservative), or the same `fact:` token unless both sides are `(relies)`. A **proposed**
  milestone holds `N` items; an **inserted** milestone `ahead-of-M<n>` is offered before each fixed
  numbered milestone that has not started (no member in progress or with `started`), made of
  high-effective-priority items whose internal dependencies are delivered, in an earlier fixed
  milestone or in the same gap, capped at `N`. Fixed milestones order by `M<n>[.<m>]`, `ahead-of-Mx`
  just before `Mx`, any other token by first appearance after the numbered ones (with a warning).
- **Output.** Milestones (`approved, as recorded` / `inserted` / `proposed`), each placement with one
  explaining criterion: `(a) after <dep>` / `(a) unblocks <ids>`, `(b) overlap with <ref> on <kind>
  <value>`, `(c) priority <p>` or `(c) milestone capacity`; then carried, held (with the blocker),
  deferred, not plannable, and fixed items whose recorded placement cannot hold (`impossible`, with
  causes); then warnings. `nothing to propose` when empty. Exit 1 with `propose: no synced backlog
  for <scope>` when the graph has no backlog of that scope.

## 7. The shared-instance safety model

The mirror is written by agents and a hook, and the shared FalkorDB (`falkordb-dev`, port 6379) also
holds other components' graphs. Two exact-complement modes, chosen per command by one choke point
(`store.open_graph`):

- **Ephemeral (default).** Allowed only when the host is loopback (`localhost`, `127.0.0.1`, `::1`),
  the port is **not** a shared one (6379, 6390, 3000) and `COORDGRAPH_EPHEMERAL_PORT` names exactly
  this port. The KB endpoints must be loopback and **not** port 8200 (the real falkor-chat). There is
  no override.
- **Shared (explicit).** Only with a confirmation equal to the graph key: loopback host; a shared
  port; the graph key is `coordination` (the only key shared mode may touch); the confirmation equals
  the key exactly; `COORDGRAPH_EPHEMERAL_PORT` unset (both at once is a mistake, not a mode). KB
  endpoints must be loopback **on port 8200**; any other URL refuses. `COORDGRAPH_KB_URL` defaults to
  `http://localhost:8200`, so a shared run needs nothing exported.
- **Confirmation sources.** The flag `--confirm GRAPHKEY` always wins. The standing env
  `COORDGRAPH_SHARED_CONFIRM` is honoured by the **routine** commands only: `sync`, `drift`, `query`,
  the views, `propose`, `kb-sync`, `kb-status`, `kb-finish`, `hook-sync`, `retry-failed`.
  **Flag-only** commands: `bootstrap`, `rebuild`, `kb-reconcile`; with the env set and no flag they
  refuse rather than fall back to ephemeral. Any value of `--confirm`, even blank, selects shared
  mode, so a wrong confirmation refuses instead of silently falling back. `kb-sync --detach-finish`
  hands `--confirm` and `--graph` to the child.
- **Refusal** is raised before any connection and reported as one stderr line; the hook reports it as
  a refusal and never queues it (§8.4).
- **Offline commands** (`lint`, `migrate`, `facts`, `history-coverage`, `rebuild --dry-run`) open no
  connection and need no confirmation.
- **`migrate --apply` mints permanent IDs** through `wi new`; it is never to be run against the real
  files or registry outside an owner-approved migration (§10.3).

## 8. The post-edit hook

`coordination-graph/hooks/post-tool-sync.sh` is a `PostToolUse` and `PostToolUseFailure` handler
(matcher `Write|Edit|Bash`) that mirrors a changed coordination or KB file before the agent reads the
tool result. **Registration is a stakeholder action**: the settings block is in
`coordination-graph/docs/hook-settings.md`, and nothing fires the script until it is applied. **Where
Claude Code fires it (which tool calls, which contexts) is unverified until the S0 probe (`claude/cobb/scripts/hook-probe.md`) has run**; the
`Hook:` header state is a static reading (§4.6) and does not prove it fires.

### 8.1 Fast path

The script needs bash 5 (`EPOCHSECONDS`, `EPOCHREALTIME`, `${v,,}`). The project root is
`$CLAUDE_PROJECT_DIR`, never the payload's `cwd`. The payload is read with `cat`. With the stamp
present, a call that changed nothing runs one pruned `find -newer .state/stamp` over the covered
shapes plus a check of the KB registry's files, and exits 0: no Python, no database. The `find`
globs are re-filtered by a regular expression equal to the coverage shapes of §2 and a test ties it to
`discover.py`. The 50 ms target is a recorded measurement, not a gate: the tests pin the structure (no
CLI call) and relative bounds only.

### 8.2 A changed mirrored file

The changed files run `hook-sync --confirm coordination PATH...` in the foreground under `timeout 50`
(a positive integer in `COORDGRAPH_HOOK_TIMEOUT_S` replaces 50; anything else means 50). `hook-sync`:

1. the pure guard first (§7): a refusal sends nothing to any listener;
2. a RESP `PING` of FalkorDB (1 s); down marks the files failed (the first run is silent instead);
3. coordination files: `sync_paths` (§4.3); a file whose hash equals the graph's is a no-op;
4. KB files that differ from `:KbFile.sha256`: falkor-chat `/health` and the embedder `<EMBED_URL>/models`
   (2 s each; a down dependency marks the files it serves failed, with no sync attempt), then the KB
   engine (§9); KB sync waits at most `COORDGRAPH_KB_BUDGET_S`; a section still embedding is
   *pending* and a detached `kb-finish` completes it.

### 8.3 Exit 2 and the relay

`hook-sync` exits 2 (stderr names each file and the reason, then "this update is not complete — say
so; it is retried automatically, or run `coordgraph sync <file>` once <dependency> is back") whenever
an edited mirrored file could not be synced or is pending. **Every such edit warns again**, with no
throttle; a call that edited no mirrored file never warns or waits. The shell relays exit 2 to the
agent. An exit other than 0 or 2 from `hook-sync` (venv missing, exception, killed by the timeout) is
the hook's own fault: the shell appends all the files to `failed.pending`, sets `retry-due` to 60 s
ahead, and exits 2 naming up to 20 files (then `and N more`). The stamp advances to the call's start
(`max(stamp, start)`, never "now"), never held back by a failure.

### 8.4 Guard refusal

A refusal is not an outage: `hook-sync` exits 2 with `refused by the guard: <reason>` (also written to
`hook.log`), queues nothing and promises no retry (for example `COORDGRAPH_EPHEMERAL_PORT` exported in
the hook's environment). A first run's refusal sets `retry-due` 15 minutes ahead: until then edits are
neither mirrored nor warned about, and a configuration fix takes up to 15 minutes to take effect
(deleting `.state/retry-due` retries at once). Edits refused meanwhile are healed by `coordgraph sync
<file>` or `sync --all`. `retry-failed` on a refusal leaves the failure state, paces the next look 60 s
ahead and exits 0.

### 8.5 Failure state and retry

State files are in FORMATS §10. Backoff after the n-th failure of the same content is 1, 2, 4, 8, then
15 minutes (the 5th and later attempts wait 15); a new edit (a different `sha256`) restarts it. With no
changed file in the call and `retry-due` passed, the shell first bumps `retry-due` by 60 s, then
launches `retry-failed --confirm coordination` detached (`setsid`, all three streams redirected), so
a crashing retry costs one fork a minute. `retry-failed` (one instance, `retry.lock`) merges
`failed.pending`, retries every due entry, drops an entry whose file is gone or whose hash equals the
graph's (it never deletes a source: `sync --all` prunes), and leaves a KB section still inside its
15-minute `kb-finish` cap to `kb-finish`; it prints one line per failed-again file and a summary and
exits 0. A Write/Edit payload whose `file_path` is in the failure state re-enters the sync even when
the stamp is past the file's mtime.

**No stamp** (fresh clone, `.state` cleaned): if `retry-due` is in the future the script exits 0; else
one `hook-sync --first-run` hash-compares every discovered file and every KB registry file and creates
the stamp at the call's start. FalkorDB down then is silent, leaves the stamp absent and sets
`retry-due` 60 s ahead.

### 8.6 Kill switch, authorization, scope

- **Kill switch.** `COORDGRAPH_HOOK_DISABLED` = `1`, `true`, `yes` or `on` (case-insensitive,
  whitespace ignored; anything else leaves the hook on), or a file `.state/disabled`: exit 0 at once;
  the header reads `Hook: disabled`.
- **Authorization.** The script passes the literal `--confirm coordination` itself on `hook-sync` and
  `retry-failed`; it never reads the key from the environment, unsets any ambient
  `COORDGRAPH_SHARED_CONFIRM` and exports none. The guards of §7 are unchanged, so the hook acts only
  on the shared instance and the real falkor-chat.
- **Not covered, by construction:** an edit made outside Claude Code (the sync header flags it), a file
  deletion (healed by `sync --all`), a worktree edit before it is merged.
- **Test seams** (the script only; never set in production): `COORDGRAPH_HOOK_CLI` (entry point) and
  `COORDGRAPH_HOOK_TIMEOUT_S`.

## 9. The KB mirror

Each registered KB file (FORMATS §6) is mirrored into `ws:agent-team` as one falkor-chat document per
claim: per `##` section in mode `whole`, per top-level bullet in mode `bullets`. Documents are
written with producer `kb-mirror`.

- **Write path is falkor-chat's REST; the observation path is a read-only graph read.** Writes are
  `POST /documents` (`sourceFormat` markdown), `DELETE /documents/{id}` (404 counts as done) and `GET
  /health`, stdlib HTTP with the timeout `COORDGRAPH_KB_HTTP_TIMEOUT_S` (default 10, must be above 0
  and below 20, a third of the lease TTL). Embedding landed, current version and normalised hash are
  read with `RO_QUERY` on `COORDGRAPH_KB_GRAPH` (`ws:agent-team`) through the same FalkorDB
  connection; that read is a coupling to falkor-chat's `Document`/`Chunk` schema.
- **Section is the concurrency unit.** Every mapping change runs between a lease acquire and a release
  (a compare-and-set on `KbSection` with a token and a 60 s `leaseUntil`); every mapping write is
  fenced on the holder's token; the lease is renewed before every falkor-chat call and each DELETE.
  A lease held by another actor reports `pending`; a lost lease means: stop, make no further
  falkor-chat call for that section, report it pending. A POST whose record write loses the lease is
  deleted by its owner.
- **Append before delete.** A changed section writes the pending marker, POSTs its new claims (only
  those whose normalised hash is not already mapped; unchanged claims are kept, not re-posted),
  records them as pending, waits up to the budget (`COORDGRAPH_KB_BUDGET_S`, default 10 s, polling
  every 0.5 s) for every new chunk to carry an embedding, then deletes the old ids under a freshly
  renewed lease and swaps the mapping. Over budget the section stays `pending`: the old claim is still
  served and `kb-status` flags it stale. An edit during a pending submission deletes that submission
  first; the same text already in flight is left to `kb-finish`.
- **Receipt check.** falkor-chat auto-supersedes a current document with the same normalised text. A
  receipt reporting `autoSuperseded` is accepted only when the victim is this section's own old
  document, a vanished section's, one whose section's file text no longer carries that claim, or an
  unmapped `kb-mirror` document of this very text (the twin of our own timed-out POST, which is then
  deleted). Any other victim aborts the section: delete everything this run submitted, keep the old
  mapping, report `failed`.
- **Refusals.** A section is `refused` when two of its claims, or a claim and another section's, have
  the same normalised text (dedupe in the file).
- **Removed section or file.** A heading gone from a registered file has its documents deleted and its
  mapping dropped (`removed`). A file dropped from the registry first is not cleaned up by the engine.
- **Section outcomes:** `synced`, `unchanged`, `degraded`, `pending`, `refused`, `failed`, `removed`.
  Incomplete (exit 2) are `pending`, `refused`, `failed`.
- **Failed documents.** falkor-chat marks a document `failed` terminally. If its chunks are all
  embedded it is still searchable (extraction is enrichment) and re-posting it would retire it at
  once: it is **`degraded`** (reported, never re-posted). An unsearchable one is **`failed-doc`**: an
  otherwise unchanged section re-posts it, bounded by `COORDGRAPH_KB_RETRY_MAX` attempts per section
  (default 3, until an edit or a success resets the count), a doubling backoff starting at
  `COORDGRAPH_KB_RETRY_BACKOFF_S` (default 1800: 30 min, then 60 min) and `COORDGRAPH_KB_RETRY_RUN_CAP`
  sections per run (default 4). The hold reason (`retry limit reached`, `retry backoff: next attempt in
  <n> s`, `per-run retry cap reached`) is the `failed` outcome's detail.
- **`kb-sync [PATH...]`** syncs every registered file (or the named ones; a name not in the registry is
  `failed`: `not a registered KB file`), section by section; an unreadable file is a `failed` outcome,
  not an abort. When incomplete, stderr says so and that `kb-finish` or the next `kb-sync` completes
  it. `--detach-finish` launches a detached `kb-finish` when any section is pending.
- **`kb-finish [--once]`** completes or gives up each `pending` section under its own lease: all new
  documents embedded swaps (old ids not kept are deleted first); a failed or vanished new document, or
  a section pending longer than 15 minutes, deletes the submission and gives up, keeping the old
  mapping and the old `sectionHash` (so `kb-status` keeps flagging it stale). Without `--once` it
  loops (5 s between passes) until none is pending. One instance at a time (`kb-finish.lock`); a
  second exits 0 saying so.
- **`kb-reconcile`** (flag-only) runs `kb-finish --once`, then: a section with no mapping whose claims
  all have a current document of the same hash is **adopted** without re-embedding (a marker-stripped
  match is adopted and re-ingested by the next `kb-sync`); a synced section whose mapped id is missing
  or hash-mismatched is **re-ingested**; a mapped document with `currentVersion` false is reported
  `non-current` (never re-ingested); a `kb-mirror` document no section maps is an `orphan`.
- **`kb-status [--ids ID...]`** prints `KB mirror: last sync <time>` and one line per finding:
  `stale` (file text no longer matches the synced hash), `pending`, `unmirrored` (no document yet),
  `coarse` (one document for a section that once mapped several, informational), `degraded`
  (informational), `failed-doc`, `duplicate` (the same claim text under several headings),
  `unregistered` (a mapped section whose heading or file is not in the registry) and `unmapped-id`
  (an `--ids` value no section maps). With `--ids`, `stale`, `pending`, `coarse` and the failed
  kinds cover only the sections holding one of them.
- **Ephemeral rehearsal.** `kb-*` refuse a non-loopback `COORDGRAPH_KB_URL` and port 8200 in
  ephemeral mode; a rehearsal points `COORDGRAPH_KB_URL` at a scratch falkor-chat and
  `COORDGRAPH_FALKORDB_PORT` / `COORDGRAPH_EPHEMERAL_PORT` at the ephemeral container.

## 10. Lint, dry-run, migrate, history

### 10.1 `lint`

Backlog and ledger checks that are errors in the file, found when written. `lint` reads identity
(§3) and `wilib.rules` (the canonical ID grammar). Findings print as
`<path>[:line]: <severity>: <code>: <message>`; the tally goes to stderr. Codes:

| Code | Severity | Rule |
|---|---|---|
| `wilib-unavailable` | error when `wilib` should be importable (the root named in `COORDGRAPH_WI_ROOT`, or the default root exists), else warn | canonical checks were not run |
| `duplicate-id` | error | a backlog file that fails to parse for a duplicate item |
| `parse` | warn | a parser warning not already reported by a rule below |
| `status` | error | an item with no recognised `Status` |
| `delivered-in-backlog` | error | a delivered item (delivered items leave the backlog) |
| `duplicate-ref` | error | a scheme `ref` headed in two backlogs |
| `malformed-id` | error | an item ID or scheme `Depends on` token failing `wilib.rules` |
| `unresolvable-dependency` | error | an unrecognised token, or one that resolves to no open item and no delivery-shaped history heading |
| `milestone-started` | error | items of one milestone disagree on `started`, or an in-progress item's milestone has none |
| `milestone-map` | error | a milestone-map row (between `<!-- coordgraph:milestone-map -->` markers) and an item's `Milestone` disagree |
| `unknown-status` | error | a unit in a ledger not headed `archived` whose `statusToken` is `unknown` |
| `no-units` | warn | a ledger whose table did not parse (FORMATS §2) |

### 10.2 `rebuild --dry-run`

Offline validation, no database and no `falkordb` import: parses every covered file and the KB
registry into the maps the store would write and asserts: every UNIQUE value passes `safe_key` and
every indexed one `norm_value`, no duplicate `Source.path`, `Unit.key`, `Item.ref`, `KbFile.path` or
`KbSection.key` across the whole load, the property sets of FORMATS §7 (required, unknown, shape),
derived values (`slug`, `unitLabel`, `statusToken`, `ref` form, denormalised fields, `rowCount`), the
label coverage probe, and each `$meta`/`$rows` parameter under 10 MB. It prints source counts per
kind, the `statusToken` tally, each `unknown` value, parser warnings, the manifest count and each
violation (`ERROR <code> <where> <message>`), ending `dry-run: ok` or `<n> violation(s)`. `--manifest
FILE` writes `<sha256>  <path>` lines.

### 10.3 `migrate`

The one-time backlog layout reformat. `--dry-run` reports and diffs and allocates nothing (it renders
placeholders and the exact `wi new` commands). `--apply` takes exactly one covered backlog PATH,
mints scheme IDs through `wi new` (never a legacy `K-NNN`/`C-NNN`), and rewrites that one file
atomically; titles the dry run flagged need `--titles-reviewed`. It brings parking-lot bullets and
id-less headings to `###` items with a `Status` field and a minted ID, keeps existing IDs, moves a
heading's trailing `(<emoji> <word> ...)` into `Status`, folds an agent plan's `| ID | Added |
Priority | Status | Summary |` table into its items, and wraps a milestone map table in the markers;
`Depends on` and `Touches` are never filled.

### 10.4 History files

`history-coverage` and `propose` read each backlog's owning history file (`docs/BACKLOG.md` to
`docs/HISTORY.md`, `kaizen/plan.md` to `kaizen/history.md`). A **delivery-shaped heading** (`##` to
`####`, outside fences) is one where an ID, or `/`-joined sibling IDs, is immediately followed by
`closed`, `delivered`, `fixed`, `shipped` or `done` (any case, not part of a hyphenated word or a noun
phrase such as `done criteria`), or one carrying both `✅` and an ID. A comma list does not match:
only the token run directly preceding the verb counts. A scheme ID is searched in every history
file, a legacy ID in the owning backlog's. A heading delivers a dependency when any of its tokens'
candidates intersect the dependency's.

## 11. Configuration

All values are `COORDGRAPH_*`, read in `coordgraph/config.py`; tests set every one.

| Variable | Default / rule |
|---|---|
| `REPO_ROOT` | else `CLAUDE_PROJECT_DIR`, else the package's repo |
| `STATE_DIR` | `<repo>/coordination-graph/.state` |
| `FALKORDB_HOST`, `FALKORDB_PORT` | `localhost`, 6379 |
| `EPHEMERAL_PORT` | unset; names the ephemeral port (§7) |
| `SHARED_CONFIRM` | unset; the standing confirmation (§7) |
| `KEY` | `coordination` |
| `KB_URL` | `http://localhost:8200` |
| `KB_GRAPH` | `ws:agent-team` |
| `EMBED_URL` | else `FALKORCHAT_EMBEDDING_BASE_URL`, else `http://localhost:1234/v1` |
| `KB_BUDGET_S` | 10 |
| `KB_HTTP_TIMEOUT_S` | 10; must be above 0 and below 20, else a configuration error at start-up |
| `KB_RETRY_MAX`, `KB_RETRY_BACKOFF_S`, `KB_RETRY_RUN_CAP` | 3, 1800, 4 |
| `HOOK_DISABLED` | the kill switch values of §8.6 |
| `WI_ROOT` | where `wilib` is imported from (read side only) |

Script-only test seams: `COORDGRAPH_HOOK_CLI`, `COORDGRAPH_HOOK_TIMEOUT_S`. The live tests also read
`COORDGRAPH_FALKOR_CHAT_DIR` and `COORDGRAPH_FC_VENV` (scratch falkor-chat location).

## 12. Surface index

| Path | Contract |
|---|---|
| `coordgraph/cli.py`, `commands_*.py` | §5, §6, §7, §8, §9 |
| `coordgraph/discover.py`, `scope.py`, `shapes.py`, `keys.py` | §2, §3, FORMATS §8 |
| `coordgraph/parse/*` | FORMATS §1 to §6 |
| `coordgraph/store.py` | §7, FORMATS §7 |
| `coordgraph/sync.py`, `drift.py` | §4 |
| `coordgraph/identity.py` | §3 |
| `coordgraph/views.py` | §6.1 to §6.3 |
| `coordgraph/propose.py` | §6.4 |
| `coordgraph/kbsync.py` | §9, FORMATS §9 |
| `coordgraph/hookstate.py`, `hookreg.py`, `hooks/post-tool-sync.sh` | §8, §4.6, FORMATS §10 |
| `coordgraph/lint.py`, `dryrun.py`, `migrate.py`, `history.py` | §10 |
| `coordgraph/config.py` | §11 |
| `coordination-graph/kb-files.txt` | FORMATS §6 |
| `coordination-graph/docs/hook-settings.md` | the settings block the stakeholder applies (§8) |
