# work-items — Live Spec

> **Status:** active · **Owner:** `coder` · **Tracks:** GAL-F0001

Current, externally observable behavior of the `work-items` skill and its `wi` CLI
(`skills/work-items/scripts/wi.py`, library `scripts/wilib/`, stdlib Python). Requirements:
`docs/requirements/work-item-numbering.md` (FR/AC); design rationale:
`docs/plans/work-item-numbering.md` (cited as "plan 3.x"). Agent-facing usage rules live in
`skills/work-items/SKILL.md`.

## 1. ID grammar

- An ID is `<KEY>-<segment>(-<segment>)*`. `KEY` matches `[A-Z]{2,6}`; `FR`, `AC`, `TP`, `R` are reserved and never a key.
- Segment letters: `I` initiative, `F` feature, `S` story, `T` task, `D` defect, `K` kaizen; `FR`/`AC` requirement / acceptance criterion (path descendants of a feature only).
- Minimum zero-padded widths: I, S, T = 2; F, D, K = 4; widths grow on overflow, never renaming. FR/AC are unpadded and may carry one lowercase letter suffix (`FR2a`). Numbers start at 1.
- Parsing is canonical: a non-canonical segment (wrong width, leading zeros, unknown letter, suffix on a non-FR/AC type) is malformed.
- Counters: I, F, D, K are counted per project; S, T per (parent, type). `I01` is Defects and `I02` Kaizen (seeded by `init`); `wi new I` starts at `I03`. Counters follow the highest number ever seen, aliases included; no number is reissued (retired, moved-away and renamed items keep theirs). FR/AC are not registry rows and are never allocated.

## 2. Parent matrix (FR-25)

| Type | Allowed parent |
|---|---|
| I | the project |
| F | the project, or an initiative numbered `I03` or higher (never `I01`/`I02`) |
| S | a feature |
| T | a story, feature, defect or kaizen item |
| D | `I01` only (implicit when `--parent` is omitted) |
| K | `I02` only (implicit when `--parent` is omitted) |
| FR, AC | a feature |

Anything else is refused (exit 1), for `new`, `move`, `promote` and every descendant of a moved subtree.
A story or task always needs an explicit parent (`wi new S` with none is refused).

## 3. CLI contract

`wi <cmd>`; the project root is found by walking up from the current directory to
`.work-items/project.json` (`init` and `key` do not need one). Results go to stdout, errors to
stderr as `wi: <message>`, warnings and header-update notes to stderr.

**Exit codes.** 0 success. 1 refused (illegal parent, malformed ID, unknown item/parent/home,
duplicate slug, bad key, bad arguments to a subcommand's logic, corrupt registry or `keys.json`, `check` found a
non-informational finding, `rebuild` guard). 2 usage error (argparse: unknown subcommand,
bad choice, missing argument). 3 ambiguous name (`resolve`, `link`, `set`). 4 unknown name
(`resolve`, and `link`/`set`, which resolve their item and target through `resolve`). 5 known legacy name not
minted (same commands as 3/4). Commands that look up an item by ID only (`retire`, `move`,
`promote`, `headers sync`) report an unknown item with exit 1.

| Command | Behavior |
|---|---|
| `init --key KEY [--components a b,c]` | Creates `.work-items/` in the current directory (see 7); needs no git repo. Idempotent for the same key; a different key already present is refused. Refuses malformed and reserved keys. Does not consult `keys.json`. Seeds `I01` "Defects" and `I02` "Kaizen" (home `repo`) once, then flushes. |
| `new TYPE [--parent ID] [--slug S] [--title T] [--home C] [--sync]` | Allocates and prints the ID. `--home` must be a declared component (or `repo`; default `repo`). A slug already held by any item (retired included) is refused. `--sync` then runs `headers sync` on the new item (5). |
| `retire ID` | Marks an item abandoned and prints its ID. The ID keeps resolving; a retired item stays a valid parent, move target and link target. |
| `resolve NAME [--json]` | See 4. |
| `move ID NEWPARENT [--sync]` | Renames the item and its whole subtree under `NEWPARENT` (`none`, `project` or empty = the project); prints the item's new ID. Stories/tasks take the next free local number under the new parent; I/F/D/K keep their own segment; descendants keep their local numbers. Every old ID becomes an alias. Moving to the same place is refused. |
| `promote ID --to F [--parent INITIATIVE] [--sync]` | Turns a kaizen item into a feature (new `F` number, optional initiative parent, default none), keeps its slug, subtree renamed, prints the new ID. Only `K` items. |
| `headers sync [ID]` | Writes `Item:`/`Aliases:` into the family documents of `ID`'s subtree (default: every item with a slug). Prints `wi: header updated: <path>` to stderr per changed file. |
| `mint-legacy HOME:NAME --type K\|F\|D [--parent ID] [--title T] [--home C]` | Prints the scheme ID of a legacy item, minting it on first call (idempotent). `HOME` must be listed under the name's prefix in `legacy_namespaces` (6). The item's home defaults to the legacy home when that is a declared component; an explicit `--home` that is neither a declared component nor `repo` is refused (exit 1, no ID allocated), and an idempotent re-mint ignores `--home`. |
| `link ID REL [TARGET] [--remove\|--clear]` | `concerns` is functional: `TARGET` replaces, `--clear` empties. `fixed-by` and `relates` are multi-valued: `TARGET` adds, `--remove` removes. Other combinations are refused. Both ends must be registry items (an FR/AC citation is refused). No output. |
| `set ID ATTR=VALUE` | `ATTR` is one of `milestone`, `home`, `touched`, `slug`, `where`. `home` must be a declared component or `repo`; `touched` is a comma-separated list of declared components; `slug` must be free. `title` is schema-valid in the event log but refused by `set` (a title is given only via `new --title` / `mint-legacy --title`). No output. Never changes an ID. |
| `flush` | Copies journal events missing from the tracked registry into it and drops duplicate lines (8). |
| `check [--set FILE] [PATH...]` | See 9. Findings to stdout; exit 1 if any finding is not informational. |
| `rebuild [--from-headers ROOT...] [--floor SPEC]... [--accept-reissue] [--write]` | See 8. Dry run by default. |
| `key suggest NAME` | Prints the first free acceptable key derived from the name; writes nothing. |
| `key claim KEY --repo R [--no-pool]` | Appends the key to `keys.json` and prints `claimed KEY; commit <path> in the team repo`; refuses reserved, malformed or already claimed keys. Run only after the stakeholder agrees. |

## 4. `resolve` contract

A name resolves, in order, as: a current ID or any former ID (alias); an FR/AC citation
(`<feature path>-FR<n>[a-z]`, the feature part resolved through any alias, then re-appended;
only under a feature); a slug; a legacy name (6). Text output is the current ID; a non-current
input adds `  [<input>: alias of <id>]` (or `  [slug <input>]`). An ambiguous slug exits 3.

`--json` prints exactly these 14 keys: `id`, `uid`, `type`, `parent` (parent's current ID or null),
`aliases` (former IDs, oldest first; for a legacy item the legacy name is not among them),
`slug`, `home`, `touched` (list), `milestone`, `where`, `links`, `retired`, `matched_as`,
`input`.

- `uid` is the item's birth ID and stays fixed across moves; `id` is the current one.
- `matched_as` is one of `current`, `alias`, `legacy` (qualified `home:name`), `bare` (unqualified legacy name), `slug`.
- `links` is `{rel: [current IDs]}`, uniformly a list (one element for `concerns`); relations with no targets are absent, so an unlinked item has `{}`.
- An FR/AC citation returns the feature's record with `id`/`uid` carrying the `-FRn`/`-ACn` suffix, `type` `FR`/`AC`, `parent` the feature's ID, aliases suffixed, `links` `{}`, `matched_as` `current` or `alias`. Resolution does not check that the FR/AC exists in a document (that is `check`).
- The registry holds identity, never status; `retired` is the only lifecycle marker.

## 5. Headers and family documents

Two optional fields in a family document's one-line header block (the `> **Status:**` line
immediately under the H1, a blank line allowed): `**Item:** \`<id>\`` and, when the item has
former IDs or a legacy name, `**Aliases:** \`a\`, \`home:name\``. `headers sync` replaces only these
two fields in that line, appends them last, and leaves the body and other fields verbatim; a
file without such a header block is left untouched.

A family is the files named `<slug>(\d*)(-<role>)?.md` under `docs/<kind>/` and
`*/docs/<kind>/`, kind in `requirements`, `plans`, `reviews`, `test-plans`, `test-reports`,
`manuals`. A file that also matches a longer registered slug's pattern belongs to that slug.
Header writing is a second phase after allocation: allocation never opens a document.
Rationale: plan 3.7.

## 6. Legacy items

`project.json` `legacy_namespaces` maps a prefix (`K-`, `C-`) to the homes that define items
under it. A qualified name `home:name` resolves only for a declared home. A bare name resolves
(`matched_as` `bare`) when its prefix has exactly one home and the item is minted; with several
homes it exits 3, listing every `home:name` and its minted ID; a declared but unminted name
exits 5 with the `mint-legacy` command; a name with no declared prefix/home exits 4.
Ambiguity comes from the declared list, never from what has been minted. A minted item records
`legacy: {home, name}`, listed in `Aliases:` as `home:name`. Plan 3.6.

## 7. Project files

`.work-items/` (created by `init`, in the repo root):

- `project.json`: `key`; `components` (fixed list, including `repo`, validated for `--home`, `home`, `touched`); `legacy_namespaces`; `legacy_paths` (pattern to home, `*` captured as `$1`, read by `check`; `init` does not write it); `rebuilt` (marker, 8).
- `registry.jsonl`: the tracked event log (8).
- `baseline.txt`: the requirements documents that existed at `init`, exempt from the no-ID finding.
- `.gitignore`: `.lock` and `*.tmp`.
- repo-root `.gitattributes` gains `.work-items/registry.jsonl merge=union` (added once).

## 8. Registry, journal and rebuild

Events (one JSON line each, `evid`, integer `ts`, `ev`): `new` (id, type, parent uid, home,
optional slug/title/legacy), `move` (uid, parent, `moves[{uid, old, new}]` for the whole
subtree, optional `retype`), `link` (rel, from, to, `op` set/add/remove), `set` (uid, attr,
value), `retire` (uid). An item's uid is its birth ID. State is the fold of the deduplicated
union: all `new` events first, then the rest by `(ts, evid)`; `ts` never falls at or below the
folded maximum. Duplicate `new` for one ID: first wins. The reader tolerates a missing final
newline, blank lines and several objects glued on one line; any other malformed line is an
error naming the line (exit 1). Plan 3.1/3.2.

- **Journal**: `<git-common-dir>/work-items/<KEY>/journal.jsonl`, shared by all branches and worktrees of a clone; every mutation appends here under one `flock`. **Tracked** copy: `.work-items/registry.jsonl`, written only by `flush`. Outside a git repo there is no journal: the tracked file is written directly under `.work-items/.lock`.
- **`flush`** appends every journal event the tracked file lacks (other branches' included) and removes duplicate `evid` lines, keeping the first; idempotent, no rewrite when unchanged. Commit the registry with the unit that allocated.
- **Reseed**: tracked events absent from the journal are copied into it on every open. When the journal is missing or empty, the `registry.jsonl` of every local and remote-tracking ref is also folded in. Events never flushed and never pushed are not recoverable.
- **`rebuild`** recovers a registry that no ref carries. Sources in order: git history of `registry.jsonl` on any ref (all committed versions, unioned by `evid`); `Item:`/`Aliases:` headers under the `--from-headers` roots, for items absent from history (one `new` at the oldest alias, one `move` per alias step; stories, tasks, titles, links, attributes and retirements are not recoverable this way); `--floor TYPE=N` or `--floor TYPE:<parent>=N` (S/T need a parent; I/F/D/K must not have one), which adds a retired sentinel so the counter starts above N. Default is a dry run listing what would be written. `--write` is refused when either on-disk copy (tracked `registry.jsonl` or journal) holds events (ref tips do not count; history is applied over the events the reseed recovered from tips), and, when history yielded nothing, without `--accept-reissue` (an ID that reached no commit, header or floor can be reissued). A successful write records a `rebuilt` marker in `project.json`. Plan 3.7.

## 9. `check`

Read-only for documents (like any read, it may copy tracked events into the journal). `PATH`
arguments limit the document scan; `--set FILE` takes a JSON list (or `{items: [...]}`) of
`{id: ...}` entries. Malformed-ID scanning covers only the `Item:`, `Tracks:` and `Aliases:`
header fields and `--set` IDs, and only tokens starting with the project key. Output line:
`[info ]<kind>: [<path>: ]<detail>`.

| Kind | Meaning |
|---|---|
| `duplicate` | two distinct `new` events for one ID; two items with one slug; one `Item:` shared by two slugs; one ID on two `--set` entries |
| `parent` | an item whose ID violates the parent matrix, or whose recorded parent differs from its ID path; an illegal ID in a header or `--set` |
| `malformed` | a key-prefixed header/`--set` token failing the grammar |
| `dangling` | a scheme ID with no `new` event or alias (also a forgotten flush); an FR/AC under no feature |
| `absent` | an FR/AC cited but not defined in the feature's requirements document(s) (checked only when such a document exists) |
| `no-id` | a requirements document outside `baseline.txt` without `Item:`; a `--set` entry without an ID |
| `family-no-item` | a family document of a registered slug lacking `Item:` |
| `family-mismatch` | a family document whose `Item:` differs from the registry ID |
| `item-slug` | an `Item:` that belongs to a different slug than the file's |
| `namespace` | a candidate file (`docs/BACKLOG.md`, `*/docs/BACKLOG.md`, `claude/*/kaizen/plan.md`) defining `K-`/`C-` items (a line starting `## `, `### `, `#### `, `- **` or `| ` followed by `K-`/`C-` and digits; a plain `- K-001` bullet or a mere mention does not count) with no `legacy_paths` home, or a home absent from `legacy_namespaces` |

Informational (printed with an `info ` prefix, never fail the exit code): `unflushed` (journal
events not in the tracked registry), `duplicate-lines` (duplicate registry lines),
`legacy-id` (a non-key ID in `--set`), `rebuilt` (the marker, until removed from
`project.json`). Plan 3.8.

## 10. Shared keys

`skills/work-items/keys.json` is a JSON list of `{key, repo, claimed, feeds_pool}`; the path is
resolved from the installed script's real path, overridable by `WI_KEYS_FILE`. `claim` takes an
exclusive `flock` on the file, refuses a duplicate, and requires the file to exist. `suggest` and `claim` alike
exit 1 with a `wi: ...` message naming the key list when it is missing, invalid JSON or not a list. Candidates
derive from the name's initials, then letter prefixes; a reserved, malformed or listed
candidate is skipped. Plan 3.9.

## 11. Residual risks

- Separate clones of one repo allocate independently and collide until `check` sees both events; use worktrees.
- A forgotten `wi flush` leaves IDs journal-only; `check` reports `unflushed` only when run.
- A header-only `rebuild` after losing every copy can reissue IDs that reached no commit, header or floor (marked by `rebuilt`).
- A key claimed from another project's session edits the team repo's working tree and is lost unless committed there.
- A web-UI merge ignores `merge=union`; `check` and `flush` repair duplicates.
- Two moves of one item on different branches resolve by `(ts, evid)`.
- A bare `K-` name resolves only through its declared homes; a name matching no declared prefix and home is unknown.
