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
cpg/mcp/build.sh                             # once per clone (or after any code change):
                                             #   build the container images — THE supported path
./falkor-chat/scripts/start_falkordb.sh -d   # FalkorDB must be up to answer queries
cpg/mcp/setup.sh                             # the host venv: the test loop and the fallback
cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q    # smoke check
```

**The server runs in a container.** In Claude Code it is wired by the repo-root
[`.mcp.json`](../../.mcp.json), which launches `cpg/mcp/docker-run.sh`, and needs no manual start —
see [Running in a container](#running-in-a-container). `cpg/mcp/build.sh` is the **supported**
fresh-clone step, not an optional nicety: the launch wrapper *will* build on a miss, but that
self-heal is a safety net whose success on a slow link is not guaranteed.

The **host venv path is retained** (`setup.sh` + `run.sh`) and is deliberately not going away: it is
the fast regression loop and the fallback if the image or the Docker daemon is broken. Two paths, two
jobs — see [Two launch paths](#two-launch-paths).

## Setup

There are two setups, because there are two launch paths (see
[Two launch paths](#two-launch-paths)):

| Path | Setup | Needs |
|---|---|---|
| **Container** (what `.mcp.json` uses) | `cpg/mcp/build.sh` | Docker |
| **Host venv** (test loop + fallback) | `cpg/mcp/setup.sh` | Python ≥ 3.12 |

### Host venv

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

`tests/test_build_inputs.py` is the one **host-only** module: it exercises
`build.sh --verify-inputs` against a throwaway copy of `cpg/mcp/` in `tmp_path` (no Docker, no image
built, the tracked tree never touched), because that check is the only thing enforcing
"[the hash covers every `COPY`](#the-image-tag-is-a-content-hash)" and a false
*pass* there is silent. It is **not collected inside the image** — `conftest.py` drops it when
`build.sh` is absent, because `.dockerignore` deliberately keeps the build tooling out of the build
context — so the in-container gate's expected counts are unchanged by it.

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

> **Truncation is display-only — with one caveat below this tool's own control.** The full result
> set this tool sees is materialised before formatting, so the caps below bound the *rendering*,
> not the query as this tool executes it, and this tool's own row-cap/char-cap accounting is
> exact. Do not read the caps as a safety limit: `MATCH (n) RETURN n` on a CPG still fetches tens
> of thousands of rows. **But** FalkorDB itself enforces a server-side `RESULTSET_SIZE` (default
> 10000, `GRAPH.CONFIG GET RESULTSET_SIZE`) beneath this tool, silently, even against an explicit
> larger `LIMIT` — verified 2026-07-30: a graph with 110k+ matching rows reports `rows=10000` with
> no indication that figure is itself a cap rather than the true count. Below 10,000 true rows the
> `rows=` figure is exact; at or above it, treat it as "at least this many," not "exactly this
> many," and re-query with an explicit narrowing predicate to get the real total.

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

> **`FALKORDB_HOST` has two different correct values, depending on the path.** `server.py`'s own
> default is `127.0.0.1`, which is right on the host venv path. Inside a container `127.0.0.1` means
> *the container itself*, so the **image** sets `ENV FALKORDB_HOST=host.docker.internal` and
> `docker-run.sh` supplies the matching `--add-host=host.docker.internal:host-gateway`. `.mcp.json`
> also states `host.docker.internal` explicitly — the redundancy is deliberate, so a reader of the
> config sees the truth and has an override point. `FALKORDB_PORT` means **the host's published
> port** on the container path.

| Variable | Default | Meaning |
|---|---|---|
| `FALKORDB_HOST` | `127.0.0.1` (`server.py`) / **`host.docker.internal`** (image `ENV` + `.mcp.json`) | Where FalkorDB listens. See the note above — the container cannot use `127.0.0.1`. |
| `FALKORDB_PORT` | `6379` | Port. On the container path this is the **host's published** port. |
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

## Two launch paths

| Path | Command | Job | Needs |
|---|---|---|---|
| **Container** — the default; what `.mcp.json` names | `cpg/mcp/docker-run.sh` | The real launch surface. | Docker |
| **Host venv** — retained | `cpg/mcp/run.sh` | The fast test loop, the fallback when the image or daemon is broken, and what ports to a Docker-less host. | Python ≥ 3.12 |

Both install from the same `requirements*.txt`, and the
[in-container test gate](#the-in-container-test-gate) is the control that keeps them from drifting.
Both resolve everything from their **own** location, so the working directory the harness starts in
is irrelevant, and neither writes to stdout — the stdio transport owns it; diagnostics go to stderr,
which the harness surfaces in its MCP log.

Rolling back to the venv path is two lines in `.mcp.json` (`docker-run.sh` → `run.sh`,
`host.docker.internal` → `127.0.0.1`) plus a session restart. Nothing else has to change.

## Running and debugging

In Claude Code the server is configured at **project scope** by the repo-root `.mcp.json`:

```json
{
  "mcpServers": {
    "cpg": {
      "command": "bash",
      "args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/cpg/mcp/docker-run.sh\""],
      "env": { "FALKORDB_HOST": "host.docker.internal", "FALKORDB_PORT": "6379" },
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
- **`"timeout": 60000`** caps a runaway **tool call** at 60 s. The default is effectively ~28 hours.
  It is **not** startup headroom — see the next bullet.
- **`MCP_TIMEOUT` is the startup budget, and it is a different thing.** `"timeout"` above bounds one
  tool call; `MCP_TIMEOUT` (an **environment variable**, not a config key, default **30000 ms**)
  bounds how long the server has to *connect*. It is not set anywhere in this repo, so the 30 s
  default applies. Raise it for a slow first container build: `MCP_TIMEOUT=60000 claude`.
  **Verified live on 2026-07-26** (Claude Code 2.1.220), because the official env-var reference table
  and its prose pages disagree about which variable owns which default:

  ```
  claude mcp list                        # cpg — ✔ Connected
  MCP_TIMEOUT=1 claude mcp list          # cpg — ✘ Failed: connection timed out after 1ms
  MCP_CONNECT_TIMEOUT_MS=1 claude mcp list   # cpg — ✔ Connected  (does NOT bind here)
  ```

  So `MCP_TIMEOUT` is the knob to raise. `MCP_CONNECT_TIMEOUT_MS` (5 s) applies only under
  `MCP_CONNECTION_NONBLOCKING=0` or a server with `alwaysLoad: true`; `cpg` uses neither. Since
  v2.1.142 MCP startup is **non-blocking** by default, so a slow connect delays tool availability
  rather than stalling the session, and a failed connect is reported to the model.

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

## Running in a container

Design and the measurements behind every choice below:
[`../../docs/plans/cpg-mcp-containerization.md`](../../docs/plans/cpg-mcp-containerization.md).

```bash
cpg/mcp/build.sh                  # runtime + test images, plus the :dev/:test aliases
cpg/mcp/build.sh --runtime-only   # just the launch image — what docker-run.sh calls on a miss
cpg/mcp/build.sh --no-cache       # force a rebuild of an existing tag
cpg/mcp/build.sh --verify-inputs  # check image-tag.sh covers every Dockerfile COPY
cpg/mcp/build.sh --help
```

`build.sh` writes **everything to stderr** and reads nothing from stdin — `docker-run.sh` may call it
while stdin is the live MCP pipe, where a stray stdout byte corrupts the protocol and a byte *read*
from stdin is a byte the server never sees.

### The image tag is a content hash

There is no mutable `:latest`-style launch tag. The tag **is** the content of the build inputs:

| Tag | Meaning |
|---|---|
| `cpg-mcp:<hash12>` | The launch image. The only tag `docker-run.sh` ever names. |
| `cpg-mcp:test-<hash12>` | The test-gate image, at the **same** hash. |
| `cpg-mcp:dev`, `cpg-mcp:test` | Moving aliases for humans and ad-hoc `docker run`. Re-pointed by every `build.sh`, never used by the launch path. |

`<hash12>` is a SHA-256 over the Dockerfile, `.dockerignore`, both `requirements*.txt`, `server.py`,
`pytest.ini` and **every file under `tests/`** — contents and *relative* paths only, so the value is
identical on every machine and no absolute path can leak into a tracked file. On every launch,
`docker-run.sh` does one `docker image inspect` (~0.05 s, purely local, **no registry contact ever**):

- **hit** → run it. Because the tag is a function of the bytes, a hit *proves* the image was built
  from exactly the code on disk. Staleness is not merely unlikely, it is unrepresentable.
- **miss** → build, then run. A miss is the only thing that triggers a build.

Adding a file under `tests/` — of any extension — changes the hash automatically, because the
enumeration *walks* the directory rather than globbing `*.py`. `build.sh --verify-inputs` fails the
build if the Dockerfile ever `COPY`s something the hash does not cover, so the two cannot drift — it
runs implicitly before every build, it joins `\`-continued `COPY` lines before parsing (a
line-oriented parse answered "OK" for those, which is a false pass in the one direction that costs
correctness), and it is itself regression-tested by `tests/test_build_inputs.py`. `COPY --from=<stage>`
is skipped: those sources come from another build stage, not from the build context.

> **"Immutable" means with respect to the repo bytes only.** The base image (`python:3.12-slim`, a
> moving tag) and pip's resolution of the version *ranges* in `requirements.txt` are deliberately
> **outside** the hash. Consequence: once an image exists, nothing ever refreshes its base — a hash
> hit will keep serving an image built on a since-patched base indefinitely. That refresh is a manual
> housekeeping act, on purpose, because doing it automatically would put a network pull back on the
> launch path.

### Why `host.docker.internal` and not `127.0.0.1`

FalkorDB runs in its **own** container (`falkordb-dev`, started by
`falkor-chat/scripts/start_falkordb.sh`, shared with `falkor-chat` and `salesperson`) and publishes
`6379` on the host. From inside the `cpg` container `127.0.0.1` would mean *that container*, so
`docker-run.sh` passes `--add-host=host.docker.internal:host-gateway` and the image defaults
`FALKORDB_HOST=host.docker.internal`. This rides the **already-published host port** that
`redis-cli`, `falkor-chat` and `salesperson` all use, so it adds no new coupling — it inherits an
existing one. `--network host` was rejected (whole-host network namespace for one outbound TCP
connection, and it behaves differently under Docker Desktop); a shared user-defined network was
rejected because reaching it means either re-creating the shared FalkorDB container or a manual,
non-persistent `docker network connect`.

### Launch flags, and why each one is there

`docker run -i --rm --init --label cpg-mcp=1 --pull=never --read-only --tmpfs /tmp
--add-host=host.docker.internal:host-gateway`

| Flag | Why |
|---|---|
| `--init` | **Required, not decorative.** PID-1 `python` **ignores `SIGTERM`** (measured: still running a minute later). tini forwards it, so the container exits `143` instead of surviving the harness's shutdown sequence. |
| `--label cpg-mcp=1` | The only handle for finding a leaked container: `docker ps -a --filter label=cpg-mcp=1`. |
| `--rm` | Reaps on normal exit and on death of the `docker run` CLI (both measured). |
| `--pull=never` | The image is local-only, so a missing tag must say `No such image` rather than docker's misleading `pull access denied … may require 'docker login'`. |
| `--read-only --tmpfs /tmp` | Least privilege. Adopted only after probing every tool-body path under it (query, `EXPLAIN`, unknown graph, `PROFILE` refusal, invalid Cypher) with no filesystem error. **First thing to drop** if the server ever fails at session start with a read-only/permission error. |
| no `--name` | A fixed name would collide the moment two sessions run concurrently — which this repo encourages — turning benign duplication into a hard failure at session start. |

Env vars are forwarded **only when actually set**. The bare `-e VAR` form is deliberately *not* used
unconditionally: with `VAR` unset in the caller's environment it does not fall through to the image
default, it **deletes** the variable in the container (measured on Docker 29.6.1), which would leave
`server.py` on its `127.0.0.1` default — the container talking to itself.

### Escape hatches

| Variable | Effect |
|---|---|
| `CPG_MCP_NO_AUTOBUILD=1` | Never build in the launch path; fail with a curated "run `cpg/mcp/build.sh`" instead. |
| `CPG_MCP_IMAGE=<ref>` | Run this exact image, **bypassing the hash gate entirely** — "the caller knows what they are running". Nothing is built for it; if it is absent you get a curated message saying so. |
| `CPG_MCP_IMAGE_REPO` | Repository name instead of `cpg-mcp`. |
| `CPG_MCP_NO_PULL=1` | `build.sh` skips the base-image `docker pull`. |
| `MCP_TIMEOUT=60000` | Raise Claude Code's 30 s **startup** budget for a slow first build. |

### Measured on this box (2026-07-26, Docker 29.6.1, Claude Code 2.1.220)

- **Connect cost through the wrapper, spawn → `initialize` + `tools/list`: median 1.47 s** over 7
  runs (range 1.40–1.58) — **4.9 % of the 30 s `MCP_TIMEOUT` budget**, a ~20× margin. The host venv
  path is ~0.37 s; that ~1 s difference is per *session*, not per query.
- **The launch path is fully offline.** Run inside a network namespace with no connectivity and no
  DNS, the full handshake still succeeded and returned real rows: the gate is a local daemon call, and
  the container's own networking is set up by the daemon, not by the client. **A *build*, by contrast,
  needs the network** unless `python:3.12-slim` is already in the local **image store** — a warm,
  fully-cached `docker build` still makes a Docker Hub `load metadata` round trip, which is why
  `build.sh` pulls the base explicitly and why the launch path does not build unless it must.

### Housekeeping

```bash
docker image ls --filter label=cpg-mcp=1                     # what has accumulated
docker pull python:3.12-slim && cpg/mcp/build.sh --no-cache   # rebuild on a refreshed base image
```

Every distinct input state leaves an image (~150 MB, almost entirely shared base layers, so the
marginal cost of each is small). Old hash tags accumulate and pruning is left to a human with that
listing in front of them: nothing in the launch path ever removes an image, because a wrapper on the
MCP startup path must not be able to destroy state.

`docker ps -a --filter label=cpg-mcp=1` finds containers. The container is **session-lifetime**, so
with N sessions open expect N in `Up` — those are live servers, not orphans. An orphan is an entry in
`Exited`/`Created`, or an `Up` count exceeding the number of open sessions. **Never `docker stop` a
labelled container while any session is open**: there is no way to tell from outside which session
owns which container, and stopping the wrong one removes a live agent's CPG read path until that
session restarts (stdio servers are not auto-reconnected).

### The in-container test gate

The host venv suite stays the primary regression signal — it is ~0.5 s and needs no Docker. The
container gate proves *the image*: its interpreter, its resolved dependencies and its network path.
It runs the same suite **minus `tests/test_build_inputs.py`**, which tests the build tooling the image
deliberately does not contain and is therefore not collected there — so the counts below are the
host counts minus that module, not a different suite.

```bash
cpg/mcp/build.sh                                    # precondition: both targets, immediately first
docker run --rm cpg-mcp:test python -m pytest tests -q                    # 53 passed, 7 deselected
docker run --rm --add-host=host.docker.internal:host-gateway \
  cpg-mcp:test python -m pytest tests -q -m live                          # 7 passed, 53 deselected
redis-cli -p 6379 GRAPH.LIST                        # done-condition: no _cpg_mcp_selftest_* residue
```

Run it after a Dockerfile, `requirements*.txt` or base-image change. Because the test tag shares the
runtime tag's content hash, a stale gate image cannot be reached by accident — if
`cpg-mcp:test-<hash>` does not exist for the current hash, the command simply does not resolve.

> **The live gate's scratch-graph name is unique per run** (fixed by C-321). Earlier, inside a
> container `os.getpid()` was **1**, so the scratch-graph name collapsed to the constant
> `_cpg_mcp_selftest_1` on the **shared** FalkorDB, and concurrent runs would corrupt each other.
> That collision is **fixed**: the name now derives from `uuid4().hex[:8]` instead of `getpid()`.

### Container debug recipe

The container twin of the venv recipe below — it overrides the image `CMD`, so no MCP plumbing is
involved:

```bash
docker run --rm --add-host=host.docker.internal:host-gateway cpg-mcp:dev \
  python -c "import server; print(server.run_query('cpg_falkorchat','MATCH (m:METHOD) RETURN count(m) AS n'))"
```

To exercise the **full protocol** through the real wrapper, pipe JSON-RPC at it. Note the trailing
`ping`: on `mcp 1.28.x` EOF on stdin tears the session down before the last reply flushes, so a
throwaway trailing message is what makes the reply you care about appear. **The throwaway's own reply
is expected to be missing, and sometimes more than one trailing reply is lost** — assert only on the
substantive ids, never on the padding.

```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
 '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"query","arguments":{"graph":"cpg_falkorchat","cypher":"MATCH (m:METHOD) RETURN count(m) AS n"}}}' \
 '{"jsonrpc":"2.0","id":4,"method":"ping"}' \
 | cpg/mcp/docker-run.sh 2>/tmp/cpg-mcp.err
# expect replies for ids 1, 2, 3 — every line valid JSON, nothing but JSON on stdout,
# all diagnostics in /tmp/cpg-mcp.err. Id 4's reply is expected to be absent.
```

### Container troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FalkorDB unreachable at 127.0.0.1:6379` from the container | `FALKORDB_HOST` reached the container as `127.0.0.1`, which means the container itself. Use `host.docker.internal` (what `.mcp.json` and the image default set). |
| `FalkorDB unreachable at host.docker.internal:6379` | FalkorDB is down (`./falkor-chat/scripts/start_falkordb.sh -d`), **or** the session is on a different Docker context (`docker context ls`) where `falkordb-dev` does not exist. |
| A query **hangs** until the 60 s tool timeout instead of failing fast | A host firewall that **DROP**s traffic from `docker0` (rather than rejecting it) turns a fast `ECONNREFUSED` into a hang, so the curated "unreachable" message never appears. |
| It worked, then stopped after FalkorDB was "hardened" | The container path depends on FalkorDB publishing on **`0.0.0.0`**, not just "to the host". `start_falkordb.sh` uses `-p "${FALKORDB_PORT}:6379"` (all interfaces). If that ever becomes `-p 127.0.0.1:6379:6379`, the container path needs `--network host` or a user-defined network instead. |
| `No such image: cpg-mcp:<tag>` | `--pull=never` doing its job on a local-only name. Run `cpg/mcp/build.sh`. |
| Build fails at `[internal] load metadata` | Offline with `python:3.12-slim` absent from the local image store. `docker pull python:3.12-slim` while connected, or fall back to `cpg/mcp/run.sh`. |
| Server fails at session start with a read-only/permission error | Drop `--read-only --tmpfs /tmp` from `docker-run.sh` and re-run the protocol probe above. |
| `docker not on PATH` / `Docker daemon not reachable` | Curated messages, both pointing at `cpg/mcp/run.sh`. |

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
the tool body directly, which needs no protocol plumbing. **Host-venv version** (the container twin
is in [Container debug recipe](#container-debug-recipe)):

```bash
cpg/mcp/.venv/bin/python -c "
import sys; sys.path.insert(0, 'cpg/mcp')
import server; print(server.run_query('cpg_falkorchat', 'MATCH (m:METHOD) RETURN count(m)'))"
```

### When the tool is unavailable

**First fallback: the host venv path.** A broken image, a dead Docker daemon or a wrong Docker
context costs two lines in `.mcp.json`, not the capability — which is exactly why `setup.sh`/`run.sh`
were kept:

```bash
cpg/mcp/setup.sh    # if the venv is not there yet
# then in .mcp.json:  docker-run.sh -> run.sh  and  host.docker.internal -> 127.0.0.1
#                     …and restart the session
```

**Second fallback: `redis-cli`**, which remains the **only** path outside Claude Code (OpenCode and
Kiro are not wired — backlog C-310):

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
claude mcp add --scope local cpg -- <repo-root>/cpg/mcp/docker-run.sh
```

Substitute your own checkout path for `<repo-root>`. Do **not** write that path back into
`.mcp.json` or any other tracked file — `claude/scripts/audit-team.sh` check 7 greps every tracked
file for a home path and fails the audit on a hit.

> **Use `docker-run.sh` here, matching the project scope.** This recipe exists for
> `$CLAUDE_PROJECT_DIR` expansion failure, which is orthogonal to container-vs-venv — so naming
> `run.sh` would quietly wire the local scope to a *different launch path* than the project scope.
> `run.sh` is the deliberate substitution only when you actually want the Docker-less variant (no
> Docker on the host, or the image is broken); remember to set `FALKORDB_HOST=127.0.0.1` if you do.

**OpenCode and Kiro.** Neither reads `.mcp.json` — OpenCode configures servers under its own
`opencode.json` `mcp` key, Kiro under `~/.kiro/settings/mcp.json`, and neither is wired in this
repo today (backlog **C-310**). The *command* ports unchanged — it is the same stdio process,
now `cpg/mcp/docker-run.sh`, with the same two env vars — but the config file, the tool-naming scheme
and the approval model do not. Containerizing is mildly **helpful** to that wiring: the launch surface
is still a single command (a script ports; a JSON `args` array does not), and "is there a working
Python 3.12 venv at that path" becomes the easier-to-check "is there a Docker daemon". It also adds
Docker as a prerequisite, and note that `MCP_TIMEOUT` is a *Claude-Code* knob — OpenCode and Kiro have
their own startup budgets, which C-310 must establish separately. `run.sh` remains what ports to a
Docker-less host. Until that wiring exists, the `cpg-analysis` skill keeps its `redis-cli` fallback
for exactly this reason: the skill is shared with all three harnesses, but the MCP tool reaches only
one.
