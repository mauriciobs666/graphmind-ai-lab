# `cpg` MCP server

A small stdio MCP server exposing **one** tool, `mcp__cpg__query`, that runs read-only OpenCypher
against a named FalkorDB graph — typically a loaded Joern Code Property Graph. It replaces
hand-assembled `redis-cli GRAPH.QUERY` command lines on the CPG **read** path for the
`cpg-analysis` skill's consumers (`analyst`, `architect`, `qa-engineer`).

Design and rationale: [`../../docs/plans/cpg-query-access.md`](../../docs/plans/cpg-query-access.md).
CPG schema: [`../../skills/joern-cpg/references/cpg-model.md`](../../skills/joern-cpg/references/cpg-model.md).

---

## Quick start

```bash
cpg/mcp/setup.sh                          # once per clone: create the venv
./falkor-chat/scripts/start_falkordb.sh -d   # FalkorDB must be up to answer queries
cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q    # smoke check
```

In Claude Code the server is wired by the repo-root [`.mcp.json`](../../.mcp.json) and needs no
manual launch — see [Running and debugging](#running-and-debugging). Everywhere else, the server
is the single command `cpg/mcp/run.sh`, speaking MCP over stdio.

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

The `-m live` run needs FalkorDB up; the default run is offline and stays green regardless
(`pytest.ini` **deselects** the `live` marker rather than skipping on reachability, so a running
database cannot silently change what the default command covers).

## The tool

Server name `cpg` · tool name `query` · callable name **`mcp__cpg__query`**. Exactly one tool,
exactly two parameters — that is the requirement (FR-2) the component exists to satisfy, which is
why every other knob is an environment variable rather than a third parameter.

| Parameter | Type | Meaning |
|---|---|---|
| `graph` | `str` | The FalkorDB graph key. **Always caller-supplied** — never defaulted, never inferred from context. CPGs are conventionally `cpg_<component>`. |
| `cypher` | `str` | The query text, sent **verbatim**. Multi-line is fine; there is no shell layer, so no quoting or escaping. |

Both are required. There is deliberately no `params` argument: Cypher parameters would be a third
parameter, so recipes substitute literals into the query text (the same rule the `redis-cli` path
always had).

### Read-only — and why that is a design guarantee, not a convention

The plain path is **`GRAPH.RO_QUERY`**, which buys two distinct things:

1. **Writes are rejected server-side.** The tool can name *any* graph on the instance — including
   `falkor-chat`'s `ws:*` and `reference` — so server-side rejection, not a client-side check, is
   what bounds the blast radius.
2. **A typo'd graph name cannot create a graph.** Plain `GRAPH.QUERY` against a non-existent key
   **materialises** it; `GRAPH.RO_QUERY` does not. The read path is safe *by construction*.

This is a property of this tool only. FalkorDB itself remains open on `:6379` with no auth,
`redis-cli` is unrestricted, and the `joern` load/write path is untouched.

### `EXPLAIN` yes, `PROFILE` no

Prefix the `cypher` text with `EXPLAIN` to get a query plan instead of results. `PROFILE` is
**refused before any server call**, with a message pointing at the fallback.

> **This diverges from raw FalkorDB, deliberately and importantly.** FalkorDB *silently ignores* an
> `EXPLAIN` or `PROFILE` prefix and **executes the query for real** — plans come only from the
> separate `GRAPH.EXPLAIN` / `GRAPH.PROFILE` commands. So a query copied verbatim out of this tool
> into `redis-cli` with an `EXPLAIN` prefix will **run**, not explain.

`PROFILE` is refused because `GRAPH.PROFILE` executes the query **including writes** (FalkorDB's
own docs; reproduced live in this repo — a `GRAPH.PROFILE … DELETE n` really deleted the node).
That is incompatible with the tool's `readOnlyHint`. For measured profiling, use `redis-cli`
directly — `graph-dba`'s territory:

```bash
redis-cli -p 6379 GRAPH.PROFILE <graph> '<cypher>' --no-raw
```

The directive sniff is **comment-blind**: `/* hi */ PROFILE …` and `// hi⏎PROFILE …` are both
classified as `PROFILE` and refused. (Live-verified: FalkorDB accepts those spellings and returns
results, so a naive `startswith` check would hand back results to a caller that asked for a
profile — a wrong answer rather than an error.) On the `EXPLAIN` path the graph's existence is
checked via `GRAPH.LIST` first, because `GRAPH.EXPLAIN` is *not* a read-only command and would
otherwise materialise a key for a mistyped name.

### Graph discovery

There is no `list_graphs` tool (FR-2: one tool). To discover graph names, either query a name that
does not exist — the error lists every loaded graph — or use `redis-cli -p 6379 GRAPH.LIST`.

### Result format and truncation

Results come back as plain text, not JSON (JSON roughly doubles the token cost and adds nothing an
agent needs):

```
graph=cpg_falkorchat · rows=2 · 12.3ms
caller | file | line
falkorchat/api.py:<module>.build_router.post_message | falkorchat/api.py | 139
falkorchat/mcp.py:<module>.send_message | falkorchat/mcp.py | 53
```

Newlines and tabs inside a cell are escaped so one row is always one line. Pipes inside values are
**not** escaped — return a single column when a value must be copied out exactly. An empty result
still prints the stats line and column names plus `(no rows)`, so "ran, matched nothing" is
distinguishable from "failed".

> **Truncation is display-only.** The full result set is materialised before formatting, so the
> caps below bound the *rendering*, not the query — memory and latency are bounded by the Cypher
> you wrote, and the `rows=` figure is always the **true** total. Do not read the caps as a safety
> limit: `MATCH (n) RETURN n` on a CPG still fetches tens of thousands of rows.

When anything is cut, a notice line is emitted **twice** — as the first *and* the last line of the
payload, byte-identical — naming which cap bound, how many rows are shown of how many, and warning
that the shown rows are an arbitrary sample unless the query has `ORDER BY`. The duplication is
deliberate: it survives any tail-side clipping by the harness, which is exactly the run where the
notice matters most. Tools diffing this output against `redis-cli` should ignore lines beginning
`… truncated:`.

### Environment variables

All are read once at import. `FALKORDB_HOST`/`FALKORDB_PORT` are set for Claude Code in
`.mcp.json`; the rest are for ad-hoc runs. A malformed or non-positive value is reported on stderr
and the default is used, rather than taking the server down.

| Variable | Default | Meaning |
|---|---|---|
| `FALKORDB_HOST` | `127.0.0.1` | Where FalkorDB listens. |
| `FALKORDB_PORT` | `6379` | Port. |
| `CPG_MCP_MAX_ROWS` | `200` | Rows rendered before the row cap binds. |
| `CPG_MCP_MAX_CELL` | `300` | Chars per cell; a cut appends `…(+N chars)`. |
| `CPG_MCP_MAX_CHARS` | `30000` | Total payload chars; whole rows are dropped from the tail (never a partial row) until it fits. |
| `CPG_MCP_TIMEOUT_MS` | `30000` | Server-side query timeout, passed to `ro_query`. Deliberately below the 60 s `.mcp.json` wall so the *server*, not the harness, produces the error message. **Does not apply to the `EXPLAIN` path** — `explain()` takes no timeout argument; planning does not execute the traversal, and the 60 s wall remains the backstop. |

**On raising `CPG_MCP_MAX_CHARS`.** The server declares
`_meta["anthropic/maxResultSizeChars"] = min(2 × CPG_MCP_MAX_CHARS, 500000)`, so Claude Code's
persist-to-disk threshold scales with the cap and the two can never disagree (without it, Claude
Code estimates a *token* budget and, above it, replaces the result with a file reference — which
would swallow the truncation notice). Raising the cap therefore stays free of disk substitution
until `2 × cap` hits the 500 000-char ceiling. But Claude Code still **warns** above roughly
10 000 tokens (~25 000–35 000 chars), and every char is context an agent pays for. Raise it for a
specific investigation, not by default.

## Running and debugging

`cpg/mcp/run.sh` is the entire launch surface. It resolves the interpreter relative to **its own**
location, so the working directory the harness starts in is irrelevant, and it fails loudly on
stderr (pointing at `setup.sh`) if the venv is missing. It never writes to stdout — the stdio
transport owns it; diagnostics go to stderr, which the harness surfaces in its MCP log.

In Claude Code the server is configured at **project scope** by the repo-root `.mcp.json`:

```json
{
  "mcpServers": {
    "cpg": {
      "command": "bash",
      "args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh\""],
      "env": { "FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "6379" },
      "timeout": 60000
    }
  }
}
```

Two non-obvious details in that shape:

- **The `bash -c` wrapper is what makes the path portable without leaking one.** A cwd-relative
  path breaks whenever a session starts inside `falkor-chat/` or `salesperson/` — the normal way to
  work in this monorepo — and an absolute path would leak the maintainer's home directory into a
  tracked file, which `claude/scripts/audit-team.sh` check 7 fails the repo on. Claude Code expands
  only `${VAR}` and `${VAR:-default}` in a config file, so the **unbraced** `$CLAUDE_PROJECT_DIR`
  passes through untouched and **bash** expands it from the spawned server's environment, where
  Claude Code does set it. `${CLAUDE_PROJECT_DIR}` in this file would *not* work.
- **`"timeout": 60000`** caps a runaway tool call at 60 s. The default is effectively ~28 hours.

`.claude/settings.json` carries `"enabledMcpjsonServers": ["cpg"]`, approving this server **by
name** rather than blanket-enabling every project server. A one-time interactive trust prompt on
first run is still expected; `claude mcp reset-project-choices` resets that answer. Note the
consequence for automation: a headless (`claude -p`) run in a workspace that has not been approved
silently has **no** `cpg` server at all.

> **Starting a session in a subdirectory needs its own approval** (verified 2026-07-25).
> `.mcp.json` discovery walks up to the repo root, so `claude mcp list` run from `falkor-chat/`
> *does* find the `cpg` server — but the project-approval state is keyed on the session's working
> directory, and `falkor-chat/` carries its own `.claude/` settings dir, so the repo-root
> `enabledMcpjsonServers` does not reach it. The result there is `⏸ Pending approval` until it is
> approved once interactively from that directory. This is an approval-scoping behaviour, not a
> path-expansion failure.

### Checking and restarting

```bash
claude mcp list          # cpg — connected / ⏸ Pending approval / failed
```

In-session, `/mcp` lists the server and its tool count (**1**). **Stdio servers are not
auto-reconnected** by Claude Code if the process dies mid-session (only HTTP/SSE are) — the server
is written to never raise out of the tool body precisely because of this. To recover: use `/mcp` →
reconnect, or restart the session. Editing `.mcp.json` also requires a session restart to take
effect.

To debug the server outside a harness, run it by hand and speak MCP at it on stdin — or just call
the tool body directly, which needs no protocol plumbing:

```bash
cpg/mcp/.venv/bin/python -c "
import sys; sys.path.insert(0, 'cpg/mcp')
import server; print(server.run_query('cpg_falkorchat', 'MATCH (m:METHOD) RETURN count(m)'))"
```

### When the tool is unavailable

The `redis-cli` path is the documented fallback, and remains the **only** path outside Claude Code:

```bash
redis-cli -p 6379 GRAPH.QUERY <graph> '<cypher>' --no-raw
```

If FalkorDB itself is down, start it with `./falkor-chat/scripts/start_falkordb.sh -d` — that one
container is shared with `falkor-chat` and `salesperson`, so never `docker rm -f` it or remove the
`falkordb-data` volume to fix an MCP problem. If a graph is missing entirely, building and loading
a CPG is the `joern` agent's job (the `joern-cpg` pipeline); this tool only queries.

## Wiring it elsewhere

**Claude Code, local scope (fallback).** If the project-scoped `.mcp.json` above does not connect —
notably if the `$CLAUDE_PROJECT_DIR` expansion ever fails — register the server per-machine
instead. This writes to `~/.claude.json`, which is untracked, so a concrete absolute path is fine
there and is *never* committed:

```bash
claude mcp add --scope local cpg -- <repo-root>/cpg/mcp/run.sh
```

Substitute your own checkout path for `<repo-root>`. Do **not** write that path back into
`.mcp.json` or any other tracked file — `claude/scripts/audit-team.sh` check 7 greps every tracked
file for a home path and fails the audit on a hit.

**OpenCode and Kiro.** Neither reads `.mcp.json` — OpenCode configures servers under its own
`opencode.json` `mcp` key, Kiro under `~/.kiro/settings/mcp.json`, and neither is wired in this
repo today (backlog **C-310**). The *command* ports unchanged — it is the same stdio process,
`cpg/mcp/run.sh`, with the same two env vars — but the config file, the tool-naming scheme and the
approval model do not. Until that wiring exists, the `cpg-analysis` skill keeps its `redis-cli`
fallback for exactly this reason: the skill is shared with all three harnesses, but the MCP tool
reaches only one.
