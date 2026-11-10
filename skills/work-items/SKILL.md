---
name: work-items
description: >-
  Standard IDs and terms for work items (initiative, feature, story, task, defect, kaizen,
  requirement FR, acceptance criterion AC) and the `wi` CLI that allocates, resolves, moves,
  promotes and checks them over a per-project registry (.work-items/). Use when creating or
  citing a feature, story, task, defect or kaizen item, when a requirements document is created
  (header Item:), when an old K-/C- backlog or kaizen item is first cited or opened (mint a
  legacy ID), when moving or promoting an item, when choosing a project key in a new repo, or
  before committing after any allocation (wi flush). Never invent or hand-number an ID.
---

# work-items — IDs, terms and the `wi` CLI

Run `python <this-skill-dir>/scripts/wi.py <cmd>` (alias it `wi`) from the project root; stdlib
Python 3.12, no setup beyond `wi init`. Glossary: `references/glossary.md` (initiative, feature,
story, task; requirement, defect, kaizen; "epic" is an alias of feature, never used in IDs).

## ID shape

`<KEY>-<segment>(-<segment>)*`, hyphen-separated; `KEY` is 2-6 capitals (one per repo).
Segment = letter + zero-padded number, growing a digit on overflow (nothing is renamed):
`I` initiative 2 digits, `F` feature 4, `D` defect 4, `K` kaizen 4, `S` story 2, `T` task 2;
`FR`/`AC` unpadded, counted per feature. `I`, `F`, `D`, `K` numbers are project-wide; `S`/`T`
restart at 01 under their parent. A missing level is skipped (a feature under no initiative).

Examples: `GAL-I03`, `GAL-I03-F0042`, `GAL-F0042` (no initiative), `GAL-I03-F0042-S01-T02`,
`GAL-I01-D0007` (defect), `GAL-I02-K0031` (kaizen), `GAL-I03-F0042-FR2`, `GAL-I03-F0042-AC3`.
`I01` = Defects and `I02` = Kaizen (reserved, created by `wi init`).

## Allowed parents (FR-25; anything else is refused)

| Child | Allowed parent |
|---|---|
| I | project |
| F | project, I>=03 |
| S | F |
| T | S, F, D, K |
| D | I01 |
| K | I02 |
| FR, AC | F |

A defect's feature, story or task is a link (`wi link`), not a parent.

## Operations

- **Allocate:** `wi new F --slug S [--parent I03] [--title T] [--home COMPONENT] [--sync]`;
  `wi new S --parent <feature>`; `T` under S/F/D/K; `D` and `K` default to `I01` / `I02`
  (`wi new -h`). Prints the ID. Never read backlogs to number.
- **Family headers:** `wi headers sync [<id>]` (or `--sync` on `new`/`move`/`promote`) writes
  `Item:` / `Aliases:` into each family document; do it when a requirements document is created
  (`wi new F --slug S --sync`).
- **Cite:** always the full ID, or `` `ID` `` in headers; `wi resolve <name> [--json]` maps any ID,
  alias, slug or FR/AC citation to the current ID and reports renames.
- **Move:** `wi move <id> <newparent|none>` renames the item and its whole subtree; old IDs stay
  aliases. **Promote:** `wi promote <K-id> --to F [--parent INITIATIVE]` (set a slug first with
  `wi set <id> slug=S`). **Retire:** `wi retire <id>`; the number is never reused.
- **Link / attributes:** `wi link <id> concerns|fixed-by|relates <target>`; `wi set <id>
  milestone|home|touched|slug|where=...`.
- **Legacy `K-`/`C-` items:** on first cite by a scheme item, or when work on one is first
  opened, `wi mint-legacy <home>:<name> --type K|F|D` (idempotent; e.g. `architect:K-004`).
  A bare `K-004` is always ambiguous (exit 3, lists the declared homes): write `<home>:K-004`. Only `C-310` (one declared home) resolves bare, exit 5 if unminted. Never edit the old document.
- **Check:** `wi check [--set file.json] [paths]` before closing a feature, and after a merge
  or a rebuild; it reports missing, malformed, dangling, duplicate and mismatched IDs.
- **Project key (new repo):** propose with `wi key suggest "<name>"`; run `wi key claim KEY
  --repo NAME` and `wi init --key KEY --components a,b,c` **only after the stakeholder agrees**.
  Commit the `keys.json` change in the team repo.

## Flush rule (binds every committing role)

After any allocation or mutation run `wi flush` and include `.work-items/registry.jsonl` in the
path-limited commit of the unit that allocated. Run it only after a merge has completed, never
mid-merge. Allocation itself writes only the shared journal, never the tracked file.

## Residual risks

- Use `git worktree`, not scratch clones: separate clones collide undetected until `check`
  sees both events.
- A GitHub web-UI merge ignores `.gitattributes` `merge=union`; flush again after pulling.
- A forgotten flush leaves dangling IDs in documents; `wi check` reports them before merge.
- Lost registry: `wi rebuild` (git history first, then headers) cannot recover an ID that
  reached no commit, header or floor; commit early. Caveats: plan
  `docs/plans/work-item-numbering.md` §3.7.
