# coordination-graph — context for AI agents working here

Python 3.12 CLI (`coordgraph`) that mirrors the repo's coordination files (teco ledgers, reviews,
requirements, `BACKLOG.md` files, agent `kaizen/plan.md`) into FalkorDB and answers read views over
the mirror. **Files are the truth; the graph is a read-only mirror.** Run instructions and the
command surface: `README.md`. Design: `claude/docs/plans/coordination-graph.md` and
`claude/docs/plans/coordination-graph3.md`; delivery record: `docs/HISTORY.md`. Skill for
consumers: `skills/coordination-graph/SKILL.md`.

## What is true now

- **Built:** every subcommand in `README.md`, the shared-instance path (`--confirm`) and the
  post-edit hook (`hooks/post-tool-sync.sh`, `hook-sync`, `retry-failed`). **Not done:** registering
  the hook (the stakeholder applies `cobb`'s settings block) and the first load itself (S8a).
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
  `bootstrap`, `rebuild`, `kb-reconcile` need the flag. Shared KB needs
  `COORDGRAPH_KB_URL=http://localhost:8200` explicit. Do not weaken either guard;
  `tests/test_shared.py` pins both. `PARSER_VERSION` is 2.
- **The hook script hard-codes `--confirm coordination`** (a literal, never env) and exports no
  confirmation, so the guard refuses it on an ephemeral port: live tests use a shim that drops
  that pair; `tests/test_hook_script.py` pins the literal. Its fast path is bash only, < 50 ms.
  `COORDGRAPH_HOOK_CLI`/`_TIMEOUT_S` are test seams.
- **The sync path never imports `wilib`** (or `identity`): `sync`, `rebuild`, `kbsync`,
  `commands_kb` and `commands_hook` stay offline: a source's subgraph is a pure function of its
  file. Identity is resolved at read time only (`lint`, `history-coverage`, `drift`/`query` header, views,
  `propose`). A test enforces the import rule; keep it passing.
- **Never run `migrate --apply` on real files or the real registry**: it allocates permanent
  IDs via `wi new`. Use `--dry-run`, or a scratch `COORDGRAPH_REPO_ROOT` and registry.
- **Per-source atomic replace**: one statement per source; a failed replace leaves the previous
  subgraph, and one is skipped when sha256 and `parserVersion` match. `rebuild` forces every
  source, prunes, sweeps orphans and proves parity against a fresh parse, never the graph itself.
- **The file wins.** A `⚠ drift` row or `Drift:` source means the graph lags; quote the file.
- **Fixtures are built in `tmp_path`**, never checked in under a `docs/plans/`-shaped path
  (discovery would treat them as sources).
- **Do not edit the root `AGENTS.md` from here**; `teco` owns it under a net-zero word budget.

## Conventions

- One runtime dependency (`falkordb`), pinned in `pyproject.toml`; parsers, discovery, keys and
  config are stdlib only. The venv is this component's own (`setup.sh`), not shared.
- Configuration is env-overridable through `coordgraph/config.py` (`COORDGRAPH_*`); tests set
  every value and never rely on the ambient environment.
- Docs follow the module convention in the root `AGENTS.md`; this file is rewritten, not appended
  to, and `docs/HISTORY.md` gets a dated entry per delivered step.
