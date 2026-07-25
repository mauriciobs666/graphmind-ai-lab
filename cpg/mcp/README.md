# `cpg` MCP server

A small stdio MCP server exposing **one** tool, `mcp__cpg__query`, that runs read-only OpenCypher
against a named FalkorDB graph — typically a loaded Joern Code Property Graph. It replaces
hand-assembled `redis-cli GRAPH.QUERY` command lines on the CPG **read** path for the
`cpg-analysis` skill's consumers (`analyst`, `architect`, `qa-engineer`).

Design and rationale: [`../../docs/plans/cpg-query-access.md`](../../docs/plans/cpg-query-access.md).
CPG schema: [`../../skills/joern-cpg/references/cpg-model.md`](../../skills/joern-cpg/references/cpg-model.md).

> **Status:** the environment (this file's setup section) is in place. The server itself
> (`server.py`, `run.sh`, `tests/`) and the Claude Code wiring (`.mcp.json`) land in the following
> implementation steps; the sections marked _(pending)_ below are filled in then.

---

## Setup

Requires **Python ≥ 3.12** (3.12.3 on this box) and network access for `pip`. There is no `uv` and
no `pipx` here — this is a plain `venv`, the same choice `falkor-chat/server` makes.

```bash
cpg/mcp/setup.sh              # create cpg/mcp/.venv + install requirements-dev.txt
cpg/mcp/setup.sh --recreate   # rebuild the venv from scratch
cpg/mcp/setup.sh --help
```

`setup.sh` is idempotent — re-running is safe and fast. It ends by importing the runtime
dependencies, so a clean exit also proves the venv is usable.

The venv is **dedicated**: it is not shared with `falkor-chat/server/.venv`, which pins a chat
application's dependency set. It is untracked (the repo-root `.gitignore` already ignores
`.venv`) — clone the repo, run `setup.sh`, done.

### Dependencies

| File | Contents | Why these bounds |
|---|---|---|
| `requirements.txt` | `mcp>=1.28,<1.29` · `falkordb>=1.6,<1.7` | Mirrors the live-verified pins in `falkor-chat/server/pyproject.toml`, the in-repo precedent for this stack. |
| `requirements-dev.txt` | `-r requirements.txt` · `pytest>=9.1,<10` | Requirements files have no "extras"; this is the equivalent of that pyproject's `dev` optional-dependency group. |

`cpg/mcp` is a script run by path, not an installable package — hence two plain requirements files
rather than a `pyproject.toml`. If `cpg/` ever grows into a package, the migration is mechanical.

## Smoke check

Nothing else in this repo will tell you when this component breaks — there is no root-level test
runner — so run the smoke check after a dependency change, a Python upgrade, or a fresh clone:

```bash
cpg/mcp/.venv/bin/python -c "import mcp.server.fastmcp, falkordb"   # dependencies import (exit 0)
cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q                           # offline: contract + formatting + errors
cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q -m live                   # requires FalkorDB up on :6379
```

The first command is available now; the two `pytest` commands become available with the server's
test module.

## The tool _(pending — filled in with the server)_

Server name `cpg` · tool name `query` · callable name `mcp__cpg__query`. Two parameters, `graph`
(the FalkorDB graph key, always caller-supplied) and `cypher` (the query text, verbatim,
multi-line allowed). Semantics — read-only (`GRAPH.RO_QUERY`), `EXPLAIN`-only plan access with
`PROFILE` refused, display-only truncation and its env-var caps, the query timeout — are
documented here once the server exists.

## Running and debugging _(pending)_

Launcher (`run.sh`), env vars and defaults (`FALKORDB_HOST`, `FALKORDB_PORT`, the cap/timeout
vars), how to restart a dead server (stdio servers are **not** auto-reconnected), and how to wire
it into Claude Code via the repo-root `.mcp.json`.

## Wiring it elsewhere _(pending)_

The local-scope registration fallback (written with a `<repo-root>` placeholder — never a concrete
machine path, which `claude/scripts/audit-team.sh` fails on), and the note that other harnesses
(OpenCode, Kiro) do not read `.mcp.json` but can launch the same stdio command.
