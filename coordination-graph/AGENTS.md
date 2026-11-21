# coordination-graph — context for AI agents working here

Python 3.12 CLI (`coordgraph`) that mirrors the repo's coordination files (teco ledgers, reviews,
requirements, `BACKLOG.md` files, agent `kaizen/plan.md`) into FalkorDB and answers read views over
the mirror. **Files are the truth; the graph is a read-only mirror.** Run instructions and the
command surface: `README.md`. Design: `claude/docs/plans/coordination-graph.md` and
`claude/docs/plans/coordination-graph3.md`; delivery record: `docs/HISTORY.md`. Skill for
consumers: `skills/coordination-graph/SKILL.md`.

## What is true now

- **Built:** every subcommand in `README.md`, and the shared-instance path (`--confirm`). **Not
  built:** the post-edit hook (plan S8) and the first load itself (S8a). Do not document or invent
  hook behaviour; it lives in the plan's S8 row and §3.2 until it ships.
- **Tests run with this directory as the working directory**, never from the repo root (rootdir
  and `testpaths` would be wrong). Default run is offline (`-m 'not live'`); `live` needs Docker.

## What will bite

- **Ephemeral by default.** `store.connect` refuses anything but loopback, a non-shared port
  (6379/6390/3000), and a port named in `COORDGRAPH_EPHEMERAL_PORT`; the KB path refuses
  falkor-chat on 8200. Tests use only that path, the throwaway container in
  `tests/fixtures/live.py`, or fakes; never the shared `falkordb-dev`.
- **Shared mode is a separate, explicit path** (`store.connect_shared`, via `open_graph`):
  `--confirm <key>` on a store-backed command, key = `coordination` and equal to the confirm,
  shared port, loopback, `COORDGRAPH_EPHEMERAL_PORT` unset. `COORDGRAPH_SHARED_CONFIRM` is the
  standing env form for routine commands only; `bootstrap`, `rebuild`, `kb-reconcile` need the
  flag. Shared KB needs `COORDGRAPH_KB_URL=http://localhost:8200` explicit. Do not weaken either
  guard; `tests/test_shared.py` pins both. `PARSER_VERSION` is 2.
- **The sync path never imports `wilib`** (or `identity`): `sync`, `rebuild`, `kbsync` and
  `commands_kb` stay offline so a source's subgraph is a pure function of its own file. Identity
  is resolved at read time only (`lint`, `history-coverage`, `drift`/`query` header, views,
  `propose`). A test enforces the import rule; keep it passing.
- **Never run `migrate --apply` on real files or the real registry** to try something: it
  allocates permanent IDs via `wi new` and rewrites one backlog. Use `--dry-run`, or a scratch
  repo root with `COORDGRAPH_REPO_ROOT` and a scratch work-items registry.
- **Per-source atomic replace.** One source is one statement (delete old subgraph, create new);
  a failed replace leaves the previous subgraph. A replace is skipped when the stored sha256 and
  `parserVersion` match. `rebuild` forces every source, prunes the gone ones, sweeps orphans and
  proves parity against a fresh parse, never the graph against itself.
- **The file wins.** A `⚠ drift` row or `Drift:` source means the graph lags; quote the file.
- **Fixtures are built in `tmp_path`**, never checked in under a `docs/plans/`-shaped path
  (write-guard deny-lists and discovery would treat them as sources).
- **Do not edit the root `AGENTS.md` from here**; `teco` owns it under a net-zero word budget.

## Conventions

- One runtime dependency (`falkordb`), pinned in `pyproject.toml`; parsers, discovery, keys and
  config are stdlib only. The venv is this component's own (`setup.sh`), not shared.
- All configuration is env-overridable through `coordgraph/config.py` (`COORDGRAPH_*`); tests set
  every value to fixture ones and never rely on the ambient environment.
- Docs follow the module convention in the root `AGENTS.md`; this file is rewritten, not appended
  to, and `docs/HISTORY.md` gets a dated entry per delivered step.
