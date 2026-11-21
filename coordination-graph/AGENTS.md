# coordination-graph — context for AI agents working here

Python 3.12 CLI (`coordgraph`) that mirrors the repo's coordination files (teco ledgers, reviews,
requirements, `BACKLOG.md` files, agent `kaizen/plan.md`) into FalkorDB and answers read views over
the mirror. **Files are the truth; the graph is a read-only mirror.** Run instructions and the
command surface: `README.md`. Design: `claude/docs/plans/coordination-graph.md` and
`claude/docs/plans/coordination-graph3.md`; delivery record: `docs/HISTORY.md`. Skill for
consumers: `skills/coordination-graph/SKILL.md`.

## What is true now

- **Built:** every subcommand in `README.md`, the shared-instance path (`--confirm`) and the
  post-edit hook (`hooks/post-tool-sync.sh`, `hook-sync`, `retry-failed`). **Loaded:** the shared
  mirror (S8a step 5, parity ok). **Not done:** registering the hook (the stakeholder applies
  `cobb`'s settings block) and the KB sync.
- **Tests run with this directory as the working directory**, never from the repo root (rootdir
  and `testpaths` would be wrong). Default run is offline (`-m 'not live'`); `live` needs Docker.

## What will bite

- **Ephemeral by default.** `store.connect` refuses anything but loopback, a non-shared port
  (6379/6390/3000) named in `COORDGRAPH_EPHEMERAL_PORT`; the KB path refuses falkor-chat on 8200.
  Tests use that path, the container in `tests/fixtures/live.py`, or fakes; never `falkordb-dev`.
- **Shared mode is a separate, explicit path** (`store.connect_shared`, via `open_graph`):
  `--confirm <key>` on a store-backed command, key = `coordination` and equal to the confirm,
  shared port, loopback, `COORDGRAPH_EPHEMERAL_PORT` unset. `COORDGRAPH_SHARED_CONFIRM` is the
  standing env form for routine commands only (`hook-sync` and `retry-failed` are routine);
  `bootstrap`, `rebuild`, `kb-reconcile` need the flag. Shared KB accepts only 8200, which is the
  `COORDGRAPH_KB_URL` default. Never weaken either guard (`tests/test_shared.py`).
- **The hook hard-codes `--confirm coordination`** (a literal, never env), so the guard refuses it
  on an ephemeral port: live tests use a shim that drops that pair. A guard refusal is reported
  as a refusal, never queued. Fast path: bash 5 only, payload read with `cat`; its shell globs
  are tied to `discover.py` by a test. `COORDGRAPH_HOOK_CLI`/`_TIMEOUT_S`: test seams, never set.
- **The sync path never imports `wilib`** (or `identity`): `sync`, `rebuild`, `kbsync`,
  `commands_kb` and `commands_hook` stay offline: a source's subgraph is a pure function of its
  file. Identity is resolved at read time only (`lint`, `history-coverage`, `drift`/`query` header, views,
  `propose`). A test enforces it.
- **Never run `migrate --apply` on real files or the real registry**: it allocates permanent
  IDs via `wi new`; use `--dry-run` or a scratch root and registry.
- **Per-source atomic replace**: one statement per source; a failed replace leaves the previous
  subgraph, and one is skipped when sha256 and `parserVersion` match. `rebuild` forces every
  source, prunes, sweeps orphans and proves parity against a fresh parse, never the graph itself.
- **The file wins** over a `⚠ drift` row or `Drift:` source.
- **KB retry**: `kb-sync` re-posts a `failed` document only if unsearchable (the POST supersedes the old
  same-text copy); an embedded one is `degraded`. Bounded by `COORDGRAPH_KB_RETRY_*`.
- **Fixtures live in `tmp_path`**, never under a `docs/plans/`-shaped path.
- **Do not edit the root `AGENTS.md`**; `teco` owns it.

## Conventions

- One runtime dependency (`falkordb`), pinned in `pyproject.toml`; parsers, discovery, keys and
  config are stdlib only. The venv is this component's own (`setup.sh`), not shared.
- Configuration is env-overridable through `coordgraph/config.py` (`COORDGRAPH_*`); tests set every value.
- Docs follow the root `AGENTS.md` module convention; this file is rewritten in place.
