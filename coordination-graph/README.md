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
| `lint [--count] [PATH...]` | Check backlog files and report every `unknown` unit status in a non-archived ledger; `--count` prints backlog items per file instead. |
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
| `kb-sync [-v] [--detach-finish] [PATH...]` | Mirror the registered KB files (`kb-files.txt`) into `ws:agent-team`, section by section. A mapped document that ended `failed` in falkor-chat is re-submitted only when it is not searchable (a `processing` one is left alone; an embedded one is reported `degraded`, never re-posted: re-posting retires the searchable copy at once). Retries are bounded per section (`RETRY_MAX`, doubling `RETRY_BACKOFF_S`) and per run (`RETRY_RUN_CAP`). `--detach-finish` launches a detached `kb-finish` when a section is pending. |
| `kb-status [--ids ID...]` | Last sync time; stale, pending, unmirrored and duplicate claims; a `failed` document as `degraded` (still searchable, informational) or `failed-doc` (not searchable). |
| `kb-reconcile` | Check every mapped KB document against `ws:agent-team`; a missing or hash-mismatched document is re-ingested, a `failed` one follows the `kb-sync` retry rule. |
| `kb-finish [--once]` | Complete or give up pending sections; loops until settled unless `--once`; one instance at a time (`kb-finish.lock`). |

**Hook (called by `hooks/post-tool-sync.sh`, not by hand; both routine, both take `--graph`/`--confirm`):**

| Command | What it does |
|---|---|
| `hook-sync [--first-run [--stamp-at EPOCH]] [PATH...]` | Sync the named changed files (coordination files and registered KB files) in the foreground; exit 2 when any file failed or is pending. `--first-run` takes no PATH: hash-compare every discovered file and create `.state/stamp`. |
| `retry-failed` | Retry every due entry of `.state/failed.json`; one instance at a time (`retry.lock`), meant to run detached. |

## Reading the sync header

Every read view (`drift`, `query`, `inflight`, `unit`, `planned`, `propose`) prints it first:

```
Synced: <oldest>Z – <newest>Z (<n> sources) · Hook: registered|not registered|disabled
Registry: <n> items · last event <date> · <m> unflushed journal events
Drift: none
```

- `Synced` is the oldest and newest sync time over the covered sources. `never` means a source
  has no sync time. `Hook` is `disabled` when `COORDGRAPH_HOOK_DISABLED` is `1/true/yes/on` or
  `<state dir>/disabled` exists (the kill switch, see "The post-edit hook"); else `registered` when the
  project's `.claude/settings.json`/`settings.local.json` run `coordination-graph/hooks/post-tool-sync.sh`
  for both `PostToolUse` and `PostToolUseFailure` with a matcher covering Write, Edit and Bash
  (`coordgraph/hookreg.py`); else `not registered` (also for unreadable settings). A static reading: it
  does not prove the hook fires in a given context.
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
`planned`, `propose`, `kb-*`, `hook-sync`, `retry-failed`) takes `--confirm GRAPHKEY`. Without it the ephemeral guard applies
unchanged. With it the command connects through `store.connect_shared`, which requires: loopback host,
a shared port (6379/6390/3000), the graph key to be the coordination mirror's (`coordination`),
`--confirm` to equal that key exactly, and `COORDGRAPH_EPHEMERAL_PORT` to be unset. Anything else
refuses before connecting, so ephemeral and shared are exact complements.

`COORDGRAPH_SHARED_CONFIRM=<key>` is the standing form of the same confirmation, honoured by the
routine commands only (`sync`, `drift`, `query`, the views, `propose`, `kb-sync`, `kb-status`,
`kb-finish`, `hook-sync`, `retry-failed`; the flag wins when both are given). `bootstrap`, `rebuild` and `kb-reconcile` are
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
In shared mode the rule inverts: `COORDGRAPH_KB_URL` defaults to `http://localhost:8200` (the real
falkor-chat, which fronts `falkordb-dev` on 6379), so nothing needs exporting, and any other URL refuses.
Writes use producer `kb-mirror`.

## Configuration

All `COORDGRAPH_*`, read in `coordgraph/config.py`: `REPO_ROOT` (else `CLAUDE_PROJECT_DIR`, else
the package's repo), `STATE_DIR` (default `coordination-graph/.state`), `FALKORDB_HOST` /
`FALKORDB_PORT` (default `localhost` / 6379), `EPHEMERAL_PORT`, `SHARED_CONFIRM`, `KEY`, `KB_URL`, `KB_GRAPH`
(default `ws:agent-team`), `EMBED_URL`, `KB_BUDGET_S`, `KB_HTTP_TIMEOUT_S` (must be < 20), `KB_RETRY_MAX` (3), `KB_RETRY_BACKOFF_S` (1800, doubling), `KB_RETRY_RUN_CAP` (4),
`HOOK_DISABLED`, and `WI_ROOT` (where `wilib` is imported from, read side only).

## Exit codes

- `lint`: 0 clean, 1 on an error finding (warnings do not fail). `migrate`: 0 ok, 1 a failed
  write, 2 bad usage (`--apply` without exactly one covered backlog PATH, or titles needing review).
- `bootstrap`, `sync`, `rebuild`, `drift`, `query`, `propose`: 0 ok, 1 failure (refused or
  unreachable endpoint, a failed source, failed parity; `drift` also 1 when it found drift).
- `unit` and `planned`: 1 also when the lookup finds nothing.
- `kb-*`: 0 settled, 2 not complete (a section pending, refused or failed; for `kb-status`
  anything flagged), 1 the command could not run.
- `hook-sync`: 0 synced or nothing to do, 2 a file failed or is pending; `retry-failed`: 0.

## The post-edit hook

`hooks/post-tool-sync.sh` is the `PostToolUse` and `PostToolUseFailure` handler (matcher
`Write|Edit|Bash`) that mirrors a coordination file into the graph before the agent reads the tool
result. **It is built but not registered by this component**: the settings block is applied by the
stakeholder (plan S8, `claude/docs/plans/coordination-graph.md` §3.2), and until it is, nothing
fires it. Where it fires (S0 probe, FR-4) is not recorded here.

- **Fast path.** A call that changed nothing runs one pruned `find -newer .state/stamp` over the
  repo (plus the KB registry) and exits 0: no Python, no database. The script needs **bash 5**
  (`EPOCHSECONDS`, `EPOCHREALTIME`, `${v,,}`). **Measured 2026-11-21 to process exit** (python
  `subprocess`, 20 calls per row, this repo as root: 4,467 files after pruning, a scratch state dir,
  the CLI stubbed, other work running on the machine), by size of the `PostToolUse` payload on
  stdin: 1 KB median 12.9 ms, 30 KB 12.6, 300 KB 15.9, 2 MB 32.7. The payload is read with `cat`:
  the earlier `read -r -d ''` took a pipe a byte at a time and cost 12.0, 16.1, 95.9 and 535.1 ms
  at the same sizes. The tests pin the structure (no CLI call) and relative bounds, never a wall time, which CPU
  load inflates (a best-of-20 absolute 50 ms still failed under 3x oversubscription): the no-change
  call's best of 20 within 6x the same-loop best of a kill-switched call plus 30 ms, and a 300 KB
  payload within 8x the best small call plus 20 ms. The 50 ms target is a recorded measurement.
  If a busier tree misses 50 ms, the
  plan's recorded fallback is a `find` restricted to the directories that hold sources (§R2).
- **A changed mirrored file** (coordination glob set or KB registry) runs `hook-sync` in the
  foreground under `timeout 50`. It pings FalkorDB (1 s), and for a changed KB file also falkor-chat
  `/health` and the embedder `/models` (2 s each); a down dependency marks the files it serves
  failed with no sync attempt. A file whose hash already equals the graph's is a no-op, not a
  warning. KB sync waits at most `COORDGRAPH_KB_BUDGET_S`; a section still embedding is *pending*
  and a detached `kb-finish` completes it.
- **Exit 2** (stderr names each file, the reason, "this update is not complete — say so") whenever
  an edited mirrored file could not be synced or is pending; **every such edit warns again**, no
  throttle. A call that edited no mirrored file never warns or waits. An exit other than 0/2 from
  `hook-sync` (venv missing, exception, killed by the timeout) is the hook's own fault: the shell
  exits 2 naming up to 20 files (then "and N more") and queues them all in `failed.pending`.
  **A guard refusal is not an outage:** `hook-sync` exits 2 with "refused by the guard: <reason>"
  (also written to `hook.log`), queues nothing and promises no retry (for instance
  `COORDGRAPH_EPHEMERAL_PORT` exported in the hook's environment); a first run's refusal sets `retry-due` 15
  minutes ahead: until then edits are neither mirrored nor warned about, and a configuration fix
  takes up to 15 minutes to take effect (delete `.state/retry-due` to retry at once). Edits refused meanwhile are healed by `coordgraph sync <file>` or `sync --all`.
  The guard runs before the FalkorDB ping, so a refused run sends nothing to the listener.
- **Failure state** in `$COORDGRAPH_STATE_DIR` (default `coordination-graph/.state/`):
  `failed.json` (per repo-relative path: `sha256`, `error`, `attempts`, `nextRetryAt`), `failed.pending`
  (the shell's lock-guarded append), `retry-due` (epoch seconds), `stamp`, `hook.log`, the locks.
  Backoff after the n-th failure is 1, 2, 4, 8, then 15 minutes; a new edit (different sha256)
  restarts it. With no edit in the call and `retry-due` passed, the shell bumps `retry-due` by 60 s
  *first*, then launches `retry-failed` detached (all three streams redirected, `setsid`), so a
  crashing retry costs one fork a minute. `retry-failed` drops an entry whose file is gone or whose
  hash already equals the graph's (it never deletes a source: `sync --all` prunes) and leaves a KB
  section inside its 15-minute `kb-finish` cap to `kb-finish`.
- **No stamp** (fresh clone): one `hook-sync --first-run` hash-compares every discovered file and
  creates the stamp at the call's start. FalkorDB down then is silent, leaves the stamp absent and
  waits for `retry-due` (+60 s) before trying again. A Write/Edit payload naming a file that is in
  the failure state re-enters the sync even when the stamp is past its mtime.
- **Kill switch.** `COORDGRAPH_HOOK_DISABLED` = `1`/`true`/`yes`/`on`, or a file `.state/disabled`:
  exit 0 at once. The sync header then reads `Hook: disabled`.
- **Authorization.** The shared instance needs `--confirm coordination`. The script passes that
  literal itself on `hook-sync` and `retry-failed`; it never reads the key from the environment,
  unsets any ambient `COORDGRAPH_SHARED_CONFIRM` and exports none. The guards are unchanged:
  loopback, shared port, key exactly `coordination`, `COORDGRAPH_EPHEMERAL_PORT` unset, KB URL the
  real falkor-chat.
- **Not covered, by construction:** an edit outside Claude Code (the sync header flags it), a file
  deletion, a worktree edit before it is merged (the root is `$CLAUDE_PROJECT_DIR`, never the
  payload `cwd`).
- **Test seams** (the script only; **never set in production**): `COORDGRAPH_HOOK_CLI` (entry
  point) and `COORDGRAPH_HOOK_TIMEOUT_S` (the 50 s bound; anything but a positive integer means
  50). Live tests use a shim that drops the
  `--confirm coordination` pair, because the guard refuses it on an ephemeral port.

The first load into the shared instance is done (S8a step 5: 548 sources, parity ok); only the KB
sync is still open. The hook's go-live (the stakeholder registers it, then the stamp is initialised)
is not part of this component's code and has not been run here.
