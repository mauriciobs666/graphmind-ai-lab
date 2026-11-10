# Work-item numbering and terminology — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `docs/requirements/work-item-numbering.md` · **Reviews:** `docs/reviews/work-item-numbering.md`

## 1. Goal & scope

Build the standard of `docs/requirements/work-item-numbering.md` (FR-1..FR-25, AC-1..AC-17): one **skill** (`skills/work-items/`) carrying the rules and one stdlib-Python **CLI** (`wi`) carrying allocation, lookup, rename/alias, legacy minting, header sync, rebuild and checks, over a per-project **registry** (an append-only event log).

Out of scope: rewriting existing documents or bulk-migrating `K-`/`C-` items; status/priority rules; legacy-project engagement; renaming files; the consumers' own plans (cross-initiative coordination, team-engagement) — only the contract they cite is fixed here (3.9); root `AGENTS.md` conventions other than the header fields (section 6, applied at feature close).

## 2. Context & findings

- Legacy namespaces: `K-` is issued by 14 homes (the 13 `claude/<agent>/kaizen/plan.md` files plus `falkor-chat/docs/BACKLOG.md`); `C-` by one (root `docs/BACKLOG.md`, component `cpg`). The `mcp-monitor`/`model-bench` backlogs only cite falkor-chat items and are not homes. These 14 drive ambiguity (3.6). `K-004` is defined in 6 agent plans.
- Existing documents carry letter-suffixed numbers (`FR-2a`, `AC-21a`); no multi-letter suffix occurs (grep).
- Skills reach every repo on this machine through per-skill symlinks `~/.claude/skills/<name>` → `skills/<name>` (`skills/README.md`, Deployment) and may carry `scripts/` (precedent `skills/joern-cpg/scripts/`); `Path(__file__).resolve()` leads back to the team repo, so one skill directory is the single shared copy (team-engagement FR-6, FR-7, AC-4).
- `git rev-parse --git-common-dir` is the same directory from every worktree and branch of one clone.
- `claude/coder/hooks/guard-coder-broad-write.sh` denies implementers `skills/README.md`, `skills/agent-maintenance/*`, `skills/agent-standards/*`, `claude/*/*.md`, `docs/{plans,reviews,requirements}/*`, not `skills/work-items/*`; `cobb` owns those prose files.
- Header block (root `AGENTS.md`): three fields, then optional `Version/Supersedes/Superseded by/Extends/Extended by/Last updated/Reviews`; header edits on `archived` documents are permitted as metadata.
- CPG: considered, not relevant — a new standalone script package and docs; no existing code changes, so there is no call graph to query.
- No step needs `data-scientist` (no ML method) or `graph-dba` (no graph-backed registry, 3.1). `security-expert` is not needed: the tool writes only `.work-items/`, the journal, family documents' header lines, and the skill's `keys.json`.

## 3. Design & rationale

### 3.1 Registry (FR-5, FR-6, FR-7, FR-8, FR-20)

An append-only JSONL event log in two copies, the journal authoritative:
- **Journal** `<git-common-dir>/work-items/<KEY>/journal.jsonl` (absolute path: the common dir is relative `.git` from the main tree, absolute from a linked worktree), shared by every branch and worktree of the clone. Every allocation and mutation appends here under one `flock`, so branch B sees every number branch A took before A merges (FR-5, AC-2).
- **Tracked** `.work-items/registry.jsonl` (`.gitattributes`: `merge=union`), written **only by `wi flush`**: it appends every journal event not already present, **and removes duplicate `evid` lines (keep first, order otherwise unchanged)**, under the lock. Allocation never touches it: a per-allocation write dirties the tree and makes `git merge` abort. Flush copies all journal events, including other branches', so an abandoned branch's numbers reach shared history (FR-6) and a document citing a journal-only ID cannot merge to a registry lacking its `new`. Two branches flushing the same events in different orders make union merges duplicate lines (verified: `e0 a1 b1 a2 b1 a1 a2 b2` → 5 unique); the next flush after the merge dedups (`e0 a1 b1 a2 b2`). `check` lists duplicate lines.
- Not a git repo (empty green-field repo before `git init`): the tracked file alone, written directly under the same lock, whose file is `.work-items/.lock`; `init` writes `.work-items/.gitignore` containing `.lock` so it is never committed (flush reads event files only).

**State** = fold of the union, deduplicated by `evid`. Order: all `new` events first, then others by `(ts, evid)`; the writer stamps `ts = max(now_ns, last_ts + 1)` with `last_ts` the maximum over the folded union, so a backward clock step (WSL2) cannot order a `move` before its `new`. **Reading:** records end with `\n`, the writer first adds a missing trailing newline, and the reader loops `json.JSONDecoder().raw_decode` per line so a glued `{…}{…}` line from a union merge is read as two events. Allocation reads only these files (FR-7, AC-5); no counter cache.

**Journal loss / fresh clone.** Tracked events missing from the journal are appended to it on open. When the journal is missing or empty, reseed also folds `git show <ref>:.work-items/registry.jsonl` for every local branch and `refs/remotes/*` ref (once, on recovery). What remains lost is events never flushed and never pushed; `check` reports any `Item:`/`Aliases:` ID with no `new` event, which catches a forgotten flush before merge.

**Rule** (in `SKILL.md` and `claude/AGENTS.md`'s git-commit section, so it binds every committing role): run `wi flush` and include `.work-items/registry.jsonl` in the path-limited commit of the unit that allocated.

Rejected: tracked file only (two branches pick the same number); FalkorDB/Redis `INCR` (service dependency per allocation, unreachable from a green-field repo whose graph tooling is on demand — team-engagement FR-2b — and still needs a file copy for FR-20; the coordination graph can mirror the JSONL later); an MCP tool (Claude-only wiring, new surface). Deciding trade-off: an offline, any-tool CLI with git-native durability. Residuals: separate clones collide undetected until `check` sees both events (agents use `git worktree`, not scratch clones); a GitHub web-UI merge ignores `merge=union` (inferred, no such flow here).

### 3.2 Events and identity

Line: `{"evid", "ts", "ev", …}`. Events: `new` (id, type, parent uid, home, optional slug/title, optional `legacy:{home,name}`); `move` (uid, new parent uid or null, `moves:[{uid, old, new}]` covering the item and its whole subtree, optional `retype` for kaizen→feature); `link` (rel, from, to or `to:null`); `set` (uid, attr ∈ milestone, home, touched, slug, where, title); `retire` (uid). No event deletes a number.

An item's **uid** is its birth ID. Current ID = last `move` (or birth); every old ID and legacy qualified name is an alias to the uid. A `move` materializes the subtree's old→new IDs, so each descendant's old ID is an explicit alias and local numbers are kept (FR-24). **FR/AC are not registry rows**: `<feature-id>-FR<n>` / `-AC<n>` resolve by splitting the suffix, resolving the feature part through any alias and re-appending it (FR-10, FR-24, AC-17); `check` verifies the number exists in the feature's requirements document (found by slug), reading `FR-<n>[a-z]?`/`AC-<n>[a-z]?` labels.

Counters (derived by fold): `I` from 03 (`init` seeds I01 Defects, I02 Kaizen; FR-14); `F`, `D`, `K` per `(project, type)`; `S`, `T` per `(parent uid, type)` counting every number ever issued under that parent, including ones that moved out (FR-6; a move-in takes the next free number, FR-22). Widths (FR-21): 2 for I/S/T, 4 for F/D/K, growing on overflow; the parse regex is `\d{w,}`. FR/AC segments are `(FR|AC)\d+[a-z]?`, so `FR-2a` is cited unchanged (FR-10).

### 3.3 Grammar and legality (FR-2, FR-3, FR-25)

`wilib/rules.py` holds `ALLOWED_PARENT = {I: {project}, F: {project, I>=03}, S: {F}, T: {S, F, D, K}, D: {I01}, K: {I02}, FR: {F}, AC: {F}}`, the parser and the formatter. Any other pair raises `IllegalParent` (AC-16).

### 3.4 Moves, promotion, retirement (FR-4, FR-13, FR-22, FR-24)

`wi move <id> <newparent|none>` and `wi promote <K-id> --to F [--parent INITIATIVE]` allocate the new segment (a feature number for a promoted kaizen item; the next local number for a moved story/task; a feature moving initiative keeps its feature segment), validate against 3.3 and emit one `move` covering the subtree. Promoted tasks land directly under the feature; the promoted feature keeps the kaizen item's slug, and `wi set <id> slug=S` supplies one (no `--slug` on `promote`). `wi retire <id>` abandons; the ID still resolves (AC-4).

### 3.5 Links and attributes (FR-11, FR-15, FR-16, FR-19)

`wi new D` places the defect under `I01`. Relations are closed — `concerns`, `fixed-by`, `relates` — and recorded from the item they are set on; none alters an ID (AC-8). **`concerns` is functional**: a defect concerns at most one feature/story/task, `wi link <D> concerns <id>` replaces it and `wi link <D> concerns --clear` empties it (event `link` with `to:null`, last-wins per `(rel, from)`, as committed). **`fixed-by` and `relates` are multi-valued** (a defect can have several fixes; FR-11 only requires the fix to be linked back): `wi link <id> <rel> <target>` adds, `--remove` removes, via an `op` field (`set` default for `concerns`, `add`, `remove`; `set`/`clear` on a multi-valued relation or `add` on `concerns` is refused). The fold keeps `rel → ordered set of uids`. `resolve --json` `links` is uniformly `{rel: [ids]}` (a one-element list for `concerns`), so a consumer never branches on the relation; the key set is unchanged, only the value shape moves from the committed `{rel: id}`. Milestone, home (one, validated against `project.json` `components`), touched components (a list), slug and `where` (FR-8's "where it lives") are `wi set <id> attr=value` attributes (AC-11).

### 3.6 Legacy items (FR-12, FR-20)

`wi mint-legacy <home>:<name> --type K|F|D [--parent ID] [--title T]` runs inside the allocation lock: an existing qualified alias returns the existing ID (idempotent, also concurrently); otherwise it allocates and records `legacy:{home,name}`. The caller supplies `--type` (kaizen entries → `K`, backlog entries → `F`, defects → `D`). Triggers are observable events: an item under the scheme links/cites it, or work on it is opened (`tico`/`teco` open a document or ledger for it). **Bare names** resolve from `project.json` `legacy_namespaces` (`"K-"`: the 14 homes, `"C-"`: `["cpg"]`) — a prefix with one home resolves to that home's item if minted; a prefix with several returns `ambiguous` with every candidate `home:name` and the IDs of the minted ones, never a guess. Exit codes: 0 resolved, 3 ambiguous, 4 unknown, 5 known legacy not minted. A new agent kaizen plan or backlog must add its home (`check` enforces, 3.8).

### 3.7 Family headers, slug lookup, rebuild (FR-20, FR-23)

Two optional header fields: `**Item:** \`GAL-I03-F0042\`` and `**Aliases:** \`old-id\`, \`architect:K-004\``. Header writing is a **second phase, separate from allocation**: `wi new F --slug S` allocates and prints the ID (reading no document); `wi headers sync <id>` then writes both fields into each family document, and `--sync` on `new`/`move`/`promote` runs it after the lock is released (committed code leaves `--sync` a stub that raises; step 6b implements it for all three). Family = files named `<slug>(\d*)(-<role>)?.md` under `docs/<kind>/` and `*/docs/<kind>/` (kind ∈ requirements, plans, reviews, test-plans, test-reports, manuals), excluding any file that also matches a longer registered slug's family pattern (`x2-coordination.md` with `x2` registered is not `x`'s). It is a CLI Bash action, since implementer write guards deny `docs/*` edits. `wi resolve` maps slug↔ID both ways (AC-14). `tico` calls `wi new F --slug S --sync` when the requirements document is created.

**Rebuild** (`wi rebuild --from-headers <roots>`; AC-12): one `new` per distinct `Item:` ID (type and parent from the path, slug from the filename), one alias per `Aliases:` entry; counters follow because the fold takes the maximum over every number seen, aliases included. Dry-run by default, `--write` replaces. Limit: stories, tasks and undocumented deleted IDs are not recoverable from headers; git history of the registry is the backup.

### 3.8 Check (FR-9)

`wi check [--set file.json] [paths]` reports:
- no ID: a `requirements/` document absent from `.work-items/baseline.txt` (existing documents grandfathered at `init`) and lacking `Item:`; a `--set` entry without ID.
- family: every family document of a registered slug whose `Item:` differs from the registry ID.
- malformed ID: a token beginning with a claimed project key and a hyphen that fails the grammar; **scanned only in header lines (`Item:`, `Tracks:`, `Aliases:`) and `--set`**, so prose placeholders such as `GAL-F<4 digits>` and legacy ids (`K-030`, `C-1001`, which start with no claimed key) never fire.
- dangling: such a scheme-shaped ID with no `new` event or alias (also how a forgotten flush shows).
- duplicate: two distinct `new` events with one ID, or one `Item:` shared by two slugs; duplicate registry lines (informational).
- `Item:`/slug mismatch; parent violating 3.3; FR/AC cited but absent.
- namespace: a file that **defines** legacy items but whose home is not in `legacy_namespaces`. Candidates: `docs/BACKLOG.md`, `*/docs/BACKLOG.md`, `claude/*/kaizen/plan.md` (the root file is matched by its own pattern: `*/docs/BACKLOG.md` does not match it). "Defines" = a line starting `#{2,4} K-nnn`, `- **K-nnn`, or a table row `| K-nnn` (same for `C-`); a mere mention does not. The home comes from a path→home map in `project.json` (`docs/BACKLOG.md` → `cpg`; `claude/<a>/kaizen/plan.md` → `<a>`; `<c>/docs/BACKLOG.md` → `<c>`). Dry run over this repo: exactly 15 files define (13 agent plans, `falkor-chat`, root `docs/BACKLOG.md`).

Informational: journal events not yet flushed. Ledgers are not scanned; units are not in the scheme (3.9).

### 3.9 Shared keys and consumer contract (FR-17, FR-18)

`skills/work-items/keys.json`: `[{"key","repo","claimed","feeds_pool"}]`, one copy in the team repo, found by `Path(__file__).resolve()`. A key matches `[A-Z]{2,6}` and is refused if listed, or equal to a declared legacy prefix or to `FR`/`AC`/`TP`/`R`. `wi key suggest "<name>"` proposes; `wi key claim KEY --repo NAME` (flock-protected) runs **only after the stakeholder agrees** (`tico` prompt) and tells the caller to commit the change in the team repo. A project marked not feeding the pool (FR-5b) may skip listing. `wi init --key KEY --components a,b,c` writes `.work-items/{project.json,registry.jsonl,baseline.txt}`, the `.gitattributes` line, and seeds `I01`/`I02` (AC-10).

**Contract** (pinned by a test, step 4): `wi resolve <name> --json` returns exactly `{id, uid, type, parent, aliases[], slug, home, touched[], milestone, where, links, retired, matched_as, input}`; `matched_as` ∈ `current|alias|legacy|bare|slug`, and for a non-current input `id` is the current ID and text output says "alias of …" (FR-20). The registry holds identity, never status.
- **Cross-initiative coordination** (`docs/requirements/cross-initiative-coordination.md`): ledger units are **not** forced into the scheme — a `T` per ledger row would bind `teco`'s ledger format with no owning step. A unit can be made a task (`GAL-F0042-T03`; FR-25 allows a task under a feature) and cited by ID, and a feature's coordination ledger is a family document carrying `Item:`. Whether `needs` facts cite scheme IDs and its parser takes them as opaque strings (unread) belongs to that feature's plan (step 13).
- **Team-engagement** (`claude/docs/requirements/team-engagement-in-other-repos.md`): the symlink serves an empty repo (FR-6, AC-4); project data is `.work-items/` only (FR-7, AC-7); `project.json.key` can attribute learnings (FR-5a). Conflict: its AC-1 says a kickoff ends with exactly a requirements document and `AGENTS.md`, but FR-23 here needs `git init` and `.work-items/` at kickoff. Routed to its owner (step 13): the kickoff set gains `.work-items/` and `.gitattributes`.

### 3.10 Skill vs tooling

A skill, as the requirement suggested: it alone reaches an empty repo with no setup and can carry scripts. `SKILL.md` carries the glossary (FR-1, "epic" an alias), one ID example per type, the FR-25 table, citing form, which `wi` command each trigger calls (including the prompt-side "work first opened"), the legacy and key protocols and the operating rules (flush with the allocating unit; worktrees, not clones; a new `K-`/`C-` home goes into `legacy_namespaces`); `scripts/` carries every enforcement. The FR-25 table, the one duplicated fact, is guarded by a drift test.

## 4. Step-by-step implementation

Layout: `skills/work-items/{SKILL.md, keys.json, references/glossary.md, scripts/wi.py, scripts/wilib/{rules,events,store,project,alloc,resolve,move,legacy,links,headers,rebuild,check,keys,cli}.py, tests/}`. Python 3.12, stdlib only. Tests run with `skills/work-items/` as working directory. Each step ≤ 5 files; steps 1–8 are ordered by fixture dependence. The ACs column is the AC→step map; section 5 tags each test with its ACs.

| # | Owner | Files | Done when | ACs |
|---|---|---|---|---|
| 1 | `tdd-engineer` | `wilib/rules.py`, `wilib/events.py` (schema, tolerant reader, fold, ordering), `tests/test_rules.py`, `tests/test_fold.py` | parse/format incl. widths and `FR2a`/`AC21a`; matrix; fold from a hand-written event list survives a glued line and a `move` stamped before its `new` | AC-1, AC-13, AC-16 |
| 2 | `tdd-engineer` | `wilib/store.py` (journal, tracked copy, `flock`, `ts`, reseed incl. refs, `flush` with dedup, non-git mode), `wilib/project.py` (`wi init` minus key claim), `tests/test_store.py`, `tests/test_init.py` | `init` in a temp git repo; flush idempotent and dedups; reseed from a ref after journal deletion; non-git mode | AC-5, AC-10 |
| 3 | `tdd-engineer` | `wilib/alloc.py` (`new`, `retire`), `wilib/cli.py`, `scripts/wi.py`, `tests/test_alloc.py`, `tests/test_concurrency.py` | correct IDs and legality by allocation; process concurrency; worktree/flush/merge scenarios | AC-1, AC-2, AC-4, AC-5, AC-13 |
| 4 | `tdd-engineer` | `wilib/resolve.py`, `wilib/move.py`, `tests/test_resolve.py`, `tests/test_move.py` | subtree renames with aliases across two moves; promotion keeps tasks; FR/AC suffix resolution; JSON keys pinned | AC-3, AC-9, AC-17 |
| 5 | `tdd-engineer` | `wilib/legacy.py`, `tests/test_legacy.py`, namespace loading in `project.py` | idempotent, concurrent-safe mint; qualified, bare-ambiguous, bare-resolving, known-not-minted | AC-6 |
| 6 | `tdd-engineer` | `wilib/links.py` (`link`, `set`, closed relations, home check against `components`), `wilib/events.py` (`op` on `link`; fold to `rel → ordered set`), `wilib/resolve.py` (`links` as `{rel: [ids]}`), `wilib/cli.py` (`link`, `set` subcommands), `tests/test_links.py` (existing `test_fold`/`test_resolve` link assertions updated with it) | `concerns` replaced/cleared with the ID constant; two `fixed-by` targets add and remove; milestone/home/touched/slug/where set; unknown home refused; JSON key set still pinned | AC-8, AC-11 |
| 6b | `tdd-engineer` | `wilib/headers.py` (`headers sync`, family glob), `wilib/cli.py` (`--sync` wired on `new`/`move`/`promote`, `--home` validated on `new` and accepted on `mint-legacy`), `wilib/legacy.py` (`home` parameter), `wilib/project.py` (`init` writes `.work-items/.gitignore`), `tests/test_headers.py` | slug↔ID; headers written and updated after a move; `--sync` works on all three; `mint-legacy --home` sets the item's component home (default as `new`); non-git `.lock` ignored | AC-14, AC-11 |
| 7 | `tdd-engineer` | `wilib/rebuild.py`, `wilib/check.py`, `tests/test_rebuild.py`, `tests/test_check.py` | rebuild with both copies deleted; check reports every rule of 3.8 | AC-7, AC-12, AC-14 |
| 8 | `tdd-engineer` | `wilib/keys.py`, `keys.json` (`[]`), `wi key` in `cli.py`, `tests/test_keys.py` | used/reserved keys refused; two concurrent claims → one wins | AC-10 |
| 9 | `cobb` (drift test by `tdd-engineer`) | `skills/work-items/SKILL.md`, `references/glossary.md`, `skills/README.md` (catalog row), `tests/test_skill_drift.py` | frontmatter valid (name = folder, description ≤ 1024); each level/type defined once, "epic" an alias; `SKILL.md` table equals a literal FR-25 matrix | AC-15 |
| 10 | `cobb` | `claude/tico/tico.md`, `claude/teco/teco.md`, `claude/architect/architect.md`, `skills/agent-maintenance/SKILL.md`, `claude/AGENTS.md` | `tico`: key proposal and stakeholder agreement at a green-field first session, `wi new F --slug S --sync` at requirements creation; `tico`/`teco`: mint legacy on first cite/open; `architect`: cite by ID; kaizen entries via `wi new K`; the flush rule in `claude/AGENTS.md`'s git-commit section (all roles); `audit-team.sh` green. Prompt-side ACs verified by `cobb` review | AC-6 (opened trigger), AC-10 (agreement) |
| 11 | `teco` (delegated writer) | root `AGENTS.md` header block only: add optional `Item:`, `Aliases:` | the convention permits the fields before step 12 writes them; other root conventions wait for feature close | — |
| 12 | `coder`; header edits through the CLI (Bash) | `wi init --key GAL` + first claim; `.work-items/project.json` (components, namespaces, path→home map); `.gitattributes`; `wi new F --slug work-item-numbering --sync` | `wi check` clean on the repo; slug↔ID resolves; bare `C-310` after mint resolves, bare `K-004` exits 3 | AC-6, AC-10, AC-14 |
| 13 | `teco` | coordination-ledger follow-ups | (a) `tico` amends team-engagement AC-1 (kickoff set gains `.work-items/`, `.gitattributes`); (b) the cross-initiative plan author decides on scheme IDs in `needs` | seam |

Machine step (human/`devops`): `ln -s <repo>/skills/work-items ~/.claude/skills/work-items`.

Step 12 values (editable in `project.json`): components `falkor-chat, salesperson, opencode, cpg, cypher-mcp, claude, kiro, mcp-monitor, model-bench, decision-models, skills, repo` (`repo` = cross-component root documents); `K-` homes and path→home map per 3.8.

## 5. Test strategy

`tdd-engineer` drives red→green and mutation-tests the allocator, legality table and fold ordering. Integration tests run `wi.py` as a subprocess in a temp git repo; concurrency uses real processes and real `git worktree`. Ordered behaviors:
1. Parse/format `S100`, `F10000`, `AC21a`, `FR2a`; a malformed string is refused. [AC-1, AC-13, AC-17]
2. Legality: the expected matrix is written out literally from FR-25's text, not iterated from `ALLOWED_PARENT`. [AC-16]
3. Fold: counters per project/type and per parent, never reduced by a move-out; two features' stories both start at `S01`. [AC-16]
4. Fold robustness: glued line read as two events; `move` before its `new` still applies; backward clock cannot reorder. [AC-2]
5. Allocation behavior: create `I03`, a feature, a story, a task and assert `GAL-I03-F####`, `…-S01`, `…-T01`, plus a feature under no initiative; allocate 100 stories under one feature and assert the 100th is `S100` with the first 99 unchanged; 8 processes × 25 allocations are 200 distinct IDs (a removed `flock` reddens it); two tasks under one story on two worktrees are distinct. [AC-1, AC-2, AC-13]
6. Branches: both worktrees branch from one commit before either allocates, allocate, flush, merge — IDs distinct (a tracked-file-only mutant reddens it); **differing-order flushes** (A: `a1 b1 a2`, B: `a1 a2 b2`) merge, one more flush runs, and `sort registry | uniq -d` is empty with unique lines equal to the `evid` count; `git status` is clean after `new`. [AC-2]
7. AC-5: run plain `new` and `new F --slug S` (no `--sync`) in a tree whose `docs/` holds a document with `Item: GAL-F0500`, a backlog with `GAL-F9999` and an unreadable file; the result ignores them and a patched `open`/`os.scandir` recorder shows no path outside `.work-items/`, the journal and `project.json` was opened. With `--sync`, the header phase opens only files matching the family pattern. [AC-5]
8. Retired item resolves; its number is not reissued. [AC-4]
9. Move a feature twice: old IDs and a descendant's old ID resolve; a story move takes the next local number, moving back another; `AC21a` citations resolve after the move; `resolve <old>` reports `matched_as: alias` and the current ID. [AC-3, AC-17]
10. Promote a kaizen item with tasks: tasks land under the feature; old `K` and task IDs resolve. [AC-9]
11. Legacy: mint twice → one ID; two homes with `K-004` → two IDs; bare `K-004` exits 3 listing both; bare `C-310` exits 0 after mint and 5 before; concurrent double mint → one ID; `git status` clean afterwards. [AC-6]
12. Mint on A, mint again on B, flush on B, merge only B: the `new` is in main's registry. [AC-6, AC-2]
13. Defect `concerns` set, changed to another target type, cleared: ID constant; two `fixed-by` targets added and one removed; `add` on `concerns` refused; milestone/home change renames nothing; unknown home refused (on `set`, `new --home`, `mint-legacy --home`). [AC-8, AC-11]
14. `resolve --json` has exactly the 3.9 key set, with and without attributes. [AC-3]
15. Slug↔ID both ways; headers written; `Aliases:` updated after a move; with `x` and `x2` registered, `x2-coordination.md` is stamped with `x2`'s ID only; a plan created after `new F` is reported by `check` until synced. [AC-14]
16. Rebuild: delete both copies (assert the journal is gone), rebuild from headers, header-recorded aliases resolve. [AC-12]
17. Check: no-ID/malformed/duplicate set all reported; a prose placeholder and a `Tracks: C-1001, K-030` header not reported; a planted citing-only backlog does not fire; a planted root-level definer with its home missing fires; the dry run over this repo is clean once step 12 lands. [AC-7]
18. Keys: used and reserved refused; concurrent claims; `init` seeds `I01`/`I02`; journal reseed from a ref. [AC-10]
19. `SKILL.md` matches the literal matrix; glossary lists "epic" as an alias. [AC-15]

## 6. Risks, open questions, documentation impact

**Risks.** (1) Separate clones collide undetected until `check`; use worktrees. (2) `key claim` writes into the team repo's working tree from another project's session and needs a commit; a lost claim re-opens the key. (3) Two moves of one item on two branches resolve by `(ts, evid)`; `check` flags an item moved on both sides. (4) Rebuild is partial (3.7). (5) A forgotten `wi flush` leaves IDs journal-only; `check` at the commit step catches it only if run (prompt discipline, step 10). (6) A wrong legacy `--type` is reversible for `K→F` only; otherwise `retire` and create anew. (7) GitHub web-UI merges ignore `merge=union` (untested).

**Open (defaulted).** Component list including `repo`; closed relation vocabulary.

**Documentation impact** (step 11 lands the header fields; the rest at feature close under the root live-spec rule): root `AGENTS.md` filename-grammar/collision rule 2 (family documents carry `Item:`; filenames carry no ID), the "`K-` numbered" backlog line (new items take scheme IDs; legacy aliased), the Structure `skills/` bullet, a note on `.work-items/` (tracked data, `merge=union`, commit with `wi flush`); `claude/README.md` and `claude/docs/manuals/agent-team.md`; each touched agent's `kaizen/history.md` via `cobb`; component backlogs' numbering sentences (`falkor-chat/docs/BACKLOG.md` and others); root `REVIEW.md` (optional `Item:` lint); the `HISTORY.md` entry last.
