# coordination-graph

`coordgraph` mirrors the repo's coordination files (teco ledgers, reviews, requirements,
component `BACKLOG.md` files, agent `kaizen/plan.md`) into a FalkorDB graph and answers read views
over it: what is in flight, what waits on the stakeholder, one unit's status, what is planned, a
milestone proposal. **The files are the truth; the graph is a read-only mirror** and is never
written by hand. It also mirrors registered knowledge-base files into falkor-chat's
`ws:agent-team` workspace. Python 3.12, one runtime dependency (`falkordb`).

Design: `claude/docs/plans/coordination-graph.md`, `claude/docs/plans/coordination-graph3.md`.
Delivery record: `docs/HISTORY.md`. How agents use it: `skills/coordination-graph/SKILL.md`.
Context for agents working on this code: `AGENTS.md`.

## Setup

```bash
coordination-graph/setup.sh            # creates coordination-graph/.venv, installs the package + pytest
coordination-graph/setup.sh --recreate # rebuild the venv from scratch
```

`setup.sh` needs Python >= 3.12 (`PYTHON=` overrides the interpreter). The venv is this
component's own. Run the CLI from anywhere with `coordination-graph/coordgraph.sh <subcommand>`
(it exits 1 naming `setup.sh` when there is no venv). Below, `coordgraph <sub>` means exactly that.

## Tests

Run with **this directory as the working directory**; from the repo root pytest would use the
monorepo as rootdir and ignore this component's `testpaths`.

```bash
cd coordination-graph
.venv/bin/python -m pytest -q          # offline (default: -m 'not live')
.venv/bin/python -m pytest -q -m live  # needs Docker; see below
```

**Live tests run against an ephemeral container only**, never the shared `falkordb-dev`.
`tests/fixtures/live.py` starts `falkordb/falkordb:v4.18.11` on a free 127.0.0.1 port with no
volume, and `store.connect` (the default path) refuses any endpoint that is not loopback, is on a shared port
(6379, 6390, 3000), or whose port is not named in `COORDGRAPH_EPHEMERAL_PORT`. The KB live tests
also start a scratch falkor-chat (needs `falkor-chat/server/.venv` and `redis-cli`; paths
overridable with `COORDGRAPH_FALKOR_CHAT_DIR` / `COORDGRAPH_FC_VENV`) with a fake embedding
endpoint, never the real `ws:agent-team` or LM Studio.

## Command surface

Every store-backed command takes `--graph KEY` (default `COORDGRAPH_KEY`, itself default
`coordination`). Verified with `--help`.

**Offline (no database):**

| Command | What it does |
|---|---|
| `lint [--count] [PATH...]` | Check backlog files; `--count` prints items per file instead. |
| `migrate (--dry-run \| --apply) [--titles-reviewed] [PATH...]` | One-time backlog layout reformat. `--dry-run` reports and diffs and allocates nothing. `--apply` allocates IDs via `wi new` and rewrites exactly one file (one PATH). |
| `facts` | List the fact tokens used in `Touches`. |
| `history-coverage [-v]` | Delivery-heading match rate over every history file. |
| `rebuild --dry-run [--manifest FILE]` | Offline validation (keys, cross-load duplicates, parameter size, status tally); no connection. `--manifest` writes the `(path, sha256)` list. |

**Store (ephemeral FalkorDB by default; the shared instance only with `--confirm`, see below):**

| Command | What it does |
|---|---|
| `bootstrap` | Create indexes and UNIQUE constraints; idempotent. |
| `sync [--changed \| --all] [-v] [PATH...]` | Parse and replace the named, changed (hash moved) or all covered files; `--all` also prunes sources no longer on disk. |
| `rebuild` | Unconditional per-file replace, prune, orphan sweep, parity check. |
| `drift [PATH...]` | Print the sync header; exit 1 when any covered file differs from the graph. |
| `query CYPHER [--covers PATH...]` | Read-only Cypher (`GRAPH.RO_QUERY`) under the sync header; `--covers` limits the drift check to those files. |
| `inflight [--waiting]` | Open coordinations, questions waiting, verdicts, requirements; `--waiting` is the stakeholder questions only. |
| `unit SLUG_OR_ID [UNIT]` | One unit's (or every unit's) status, agent, deliverable, question, by ledger slug or work-item ID; UNIT is a key or label (`U3`, `U32~2`). |
| `planned COMPONENT_OR_AGENT [--all]` | Open backlog items of one component or agent; `--all` lists deferred/rejected in full. |
| `propose COMPONENT_OR_AGENT [--per-milestone N]` | Read-only milestone proposal (default 6 items per milestone); never takes effect on its own. |

**KB mirror (ephemeral FalkorDB and a loopback falkor-chat; shared mode pairs with the real one):**

| Command | What it does |
|---|---|
| `kb-sync [-v] [--detach-finish] [PATH...]` | Mirror the registered KB files (`kb-files.txt`) into `ws:agent-team`, section by section. `--detach-finish` launches a detached `kb-finish` when a section is pending. |
| `kb-status [--ids ID...]` | Last sync time; stale, pending, unmirrored and duplicate claims. |
| `kb-reconcile` | Check every mapped KB document against `ws:agent-team`. |
| `kb-finish [--once]` | Complete or give up pending sections; loops until settled unless `--once`; one instance at a time (`kb-finish.lock`). |

## Reading the sync header

Every read view (`drift`, `query`, `inflight`, `unit`, `planned`, `propose`) prints it first:

```
Synced: <oldest>Z – <newest>Z (<n> sources) · Hook: active|disabled
Registry: <n> items · last event <date> · <m> unflushed journal events
Drift: none
```

- `Synced` is the oldest and newest sync time over the covered sources. `never` means a source
  has no sync time. `Hook` is `disabled` when `COORDGRAPH_HOOK_DISABLED` is `1/true/yes/on` or
  `<state dir>/disabled` exists; the hook itself is not built yet (see "Not yet").
- `Registry:` is the work-item registry read at query time; `Registry: unavailable -- identities
  compared as written` means identities were not resolved.
- `Drift: <n> source(s)` is followed by one line per source: state `changed`, `unsynced`, `gone`
  or `stale-parser`, the path, and `-> file wins; run coordgraph sync`. A `⚠ drift` row marks a
  view row whose file moved since the sync. **The graph is stale there: read the file and
  quote it.**

## Rebuild and parity

`coordgraph rebuild` bootstraps, force-replaces every covered source, prunes the ones gone,
deletes orphan `Unit`/`Item` nodes (reported as a bug signal), stamps `SyncState`, then compares
a fresh parse of the files against an export of the graph (never the graph with itself). It prints
the per-status counts, any orphans and the parity verdict; exit 0 only when parity holds with no
failed source and no orphans. Run `rebuild --dry-run` first: it validates offline.

## The shared instance

Every store-backed command (`bootstrap`, `sync`, `rebuild`, `drift`, `query`, `inflight`, `unit`,
`planned`, `propose`, `kb-*`) takes `--confirm GRAPHKEY`. Without it the ephemeral guard applies
unchanged. With it the command connects through `store.connect_shared`, which requires: loopback host,
a shared port (6379/6390/3000), the graph key to be the coordination mirror's (`coordination`),
`--confirm` to equal that key exactly, and `COORDGRAPH_EPHEMERAL_PORT` to be unset. Anything else
refuses before connecting, so ephemeral and shared are exact complements.

`COORDGRAPH_SHARED_CONFIRM=<key>` is the standing form of the same confirmation, honoured by the
routine commands only (`sync`, `drift`, `query`, the views, `propose`, `kb-sync`, `kb-status`,
`kb-finish`; the flag wins when both are given). `bootstrap`, `rebuild` and `kb-reconcile` are
flag-only and refuse the env alone. `kb-sync --detach-finish` hands `--confirm` and `--graph` on to the
child.

The stored `parserVersion` is 2 (`coordgraph/parse/source.py`): a graph stamped 1 re-syncs on the next
`sync` and shows `stale-parser` in `drift`.

## KB mirror and the loopback rule

An ephemeral mirror's falkor-chat must be ephemeral too. `kb-*` commands
refuse a non-loopback `COORDGRAPH_KB_URL`, and refuse port **8200** (the real falkor-chat, and the
default when the variable is unset). For a rehearsal, start a scratch falkor-chat (the live
fixture does this) and point `COORDGRAPH_KB_URL` at its loopback URL, with
`COORDGRAPH_FALKORDB_PORT` and `COORDGRAPH_EPHEMERAL_PORT` set to the ephemeral container's port.
In shared mode the rule inverts: `COORDGRAPH_KB_URL=http://localhost:8200` (the real falkor-chat,
which fronts `falkordb-dev` on 6379) must be set explicitly, and any other URL refuses.
Writes use producer `kb-mirror`.

## Configuration

All `COORDGRAPH_*`, read in `coordgraph/config.py`: `REPO_ROOT` (else `CLAUDE_PROJECT_DIR`, else
the package's repo), `STATE_DIR` (default `coordination-graph/.state`), `FALKORDB_HOST` /
`FALKORDB_PORT` (default `localhost` / 6379), `EPHEMERAL_PORT`, `SHARED_CONFIRM`, `KEY`, `KB_URL`, `KB_GRAPH`
(default `ws:agent-team`), `EMBED_URL`, `KB_BUDGET_S`, `KB_HTTP_TIMEOUT_S` (must be < 20),
`HOOK_DISABLED`, and `WI_ROOT` (where `wilib` is imported from, read side only).

## Exit codes

- `lint`: 0 clean, 1 on an error finding (warnings do not fail). `migrate`: 0 ok, 1 a failed
  write, 2 bad usage (`--apply` without exactly one covered backlog PATH, or titles needing review).
- `bootstrap`, `sync`, `rebuild`, `drift`, `query`, `propose`: 0 ok, 1 failure (refused or
  unreachable endpoint, a failed source, failed parity; `drift` also 1 when it found drift).
- `unit` and `planned`: 1 also when the lookup finds nothing.
- `kb-*`: 0 settled, 2 not complete (a section pending, refused or failed; for `kb-status`
  anything flagged), 1 the command could not run.

## Not yet

The post-edit hook (`hooks/post-tool-sync.sh`, `hook-sync`, `retry-failed`), its contract, kill
switch and failure/backoff behaviour, the S0 probe result for FR-4, and the first load into the shared
instance (the path exists; the load has not been run) are plan steps S8 and S8a (`claude/docs/plans/coordination-graph.md`, §3.2). They are
documented here when built; nothing about them is stated now.
