# CPG query access — implementation plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** C-301…C-307 (M3) ·
> **approved (re-gate 2026-07-25: approve with suggestions, 0 blockers) → in implementation**.
>
> Design for **CPG query access** (MCP tool replaces `redis-cli GRAPH.QUERY` on the CPG **read**
> path). Requirements: [`../requirements/cpg-query-access.md`](../requirements/cpg-query-access.md).
> Coordination: [`cpg-query-access-coordination.md`](./cpg-query-access-coordination.md).
> Review that gated this document: [`../reviews/cpg-query-access.md`](../reviews/cpg-query-access.md).
> Author: `architect`. v1 2026-07-24 · v2 2026-07-25 (rework against the analyst review +
> stakeholder decisions D1–D4) · v2.1 2026-07-25 (re-gate follow-up: N-1, N-2, n-3 — §10) ·
> **v2.2 2026-07-25 (audit-trail correction: re-gate finding n-4 recorded as accepted and
> implemented — §4.4, S2, §10; no design change)**.
> **S8 must be read at v2.1 and S2 at v2.2**: S8's destructive procedure and §4.4's
> truncation/directive spec changed after the re-gate, and §4.4/S2 gained the server-level
> `instructions=` string (n-4) at v2.2.

**Reading order for an implementer:** §4 (the frozen contract) → §5 (your step) → §7 (how it is
verified). §10 is the rework log for the reviewer, not for you.

**Path convention in this document:** `<repo-root>` stands for the absolute path of this
repository. No absolute machine path appears in this file, in `.mcp.json`, or in any other tracked
artifact this plan produces — `claude/scripts/audit-team.sh` check 7 greps every tracked file for
the maintainer's home path and username and fails the audit on a hit (§4.3, R-9).

---

## 1. Goal & scope

Give `analyst` / `architect` / `qa-engineer` a **single MCP tool** that runs Cypher against a
named FalkorDB graph, so CPG questions stop being hand-assembled `redis-cli` command lines.
One tool, exactly two parameters (`graph`, `cypher`) — FR-1…FR-5. The `cpg-analysis` skill is
re-pointed at the tool; `redis-cli` survives as a documented fallback. `joern-cpg-pipeline.md`
FR-9 is reversed in the same change (FR-6 / AC-4).

**Out of scope** (unchanged from requirements): auth / grants; `falkor-chat` + `salesperson`
shell scripts; the `joern` write/load path; reshaping the four recipes into per-recipe tools;
the `MAX_ARG_STRLEN` loader bug (C-101). **Also out of scope by this design:**

- MCP wiring for OpenCode and Kiro (§4.3, R-7) — deferred as **C-310**, but the *knowledge* gap is
  closed in S7 and the `redis-cli` fallback is retained precisely because of it.
- A bounded transitive **upward** call-closure query — deferred as **C-308**, owner `graph-dba`
  (decision D3). AC-1 is demonstrated with the direct-caller question.
- Measured profiling through the tool — `PROFILE` is **removed** (decision D4); `graph-dba` keeps
  `GRAPH.PROFILE` via `redis-cli`.
- Resolving `joern-cpg-pipeline.md` OQ2 (component structure) — this change puts the first code
  under `cpg/`, which informs OQ2 without closing it (R-8).

### 1.1 Stakeholder decisions this plan is built on (ruled 2026-07-25, do not re-open)

| ID | Decision | Where it lands |
|---|---|---|
| **D1** | A destructive rebuild of `cpg_falkorchat` is **approved** (the graph's current data does not matter). Rebuild from `falkor-chat/server/{falkorchat,tests}` **including tests**, then **record the fresh counts as the new baseline**. The M2 figures (79,581 / 522,182 / 21 / 39) are *not* reproducible — the source has moved 8 commits — and are formally superseded. | §2.3, S8, §7.2 AC-3 |
| **D2** | AC-3's stale "30 untested methods" is corrected to "39 rows / 32 distinct names", and in practice superseded by the fresh baseline. Requirements edit — **not** the architect's, and **not** teco's. | S6 |
| **D3** | AC-1 is demonstrated with the **direct-caller** question. This feature changes *how Cypher is transmitted*, not how powerful Cypher is. The transitive upward-closure query is **deferred**, not discarded → backlog **C-308**, owner `graph-dba`. | §7.2 AC-1, S6, S10 |
| **D4** | **`EXPLAIN`-only. `PROFILE` is removed from the tool.** `GRAPH.PROFILE` executes writes (reproduced live by teco: `GRAPH.PROFILE _teco_probe "MATCH (n:T) DELETE n"` really deleted the node). The read-only guarantee wins. | §4.4 D5 |

---

## 2. Context & findings

Everything below was verified in this environment (2026-07-24, re-verified 2026-07-25 where
marked) unless flagged *inferred*.

### 2.1 Environment

| Fact | Evidence |
|---|---|
| FalkorDB up, container `falkordb-dev`, `falkordb/falkordb:v4.18.11`, `:6379` | teco, re-confirmed by live queries below |
| `GRAPH.LIST` → `cpg_falkorchat`, `cpg_salesperson`, `ws:acme`, `reference`, `ws:test` | live |
| **No MCP server configured anywhere**: empty `mcpServers` globally and per-project, no `.mcp.json` | teco |
| **No Node.js in WSL.** `node` not found; `npx` resolves to the Windows install via interop | `command -v node` → not found |
| Python 3.12.3 system; **no `uv`, no `pipx`** | `which uv uvx pipx` → empty |
| The repo already ships an MCP server built on the **official Python SDK** (`mcp.server.fastmcp.FastMCP`) — `falkor-chat/server/falkorchat/mcp.py`, pinned `mcp>=1.28,<1.29`, `falkordb>=1.6,<1.7` in `falkor-chat/server/pyproject.toml` | file read |
| That venv (`falkor-chat/server/.venv`) has `mcp 1.28.1`, `FalkorDB 1.6.1`, `redis 8.0.1`, `pytest 9.1.1` | `pip list` |
| Import cost of `falkordb` + `mcp.server.fastmcp`: **~1.5 s cold** (one-off per session, not per query). A `redis-cli` round trip is ~4 ms | `time python -c "import ..."` |

### 2.2 FalkorDB / client behaviour (live-probed against `cpg_falkorchat`)

- `falkordb-py` exposes `Graph.query`, `Graph.ro_query`, `Graph.explain`, `Graph.profile`.
  `ro_query` **rejects writes** server-side: `CREATE (:Foo)` → `ResponseError: graph.RO_QUERY is
  to be executed only on read-only queries`. This is a free read-only mode, no extra code.
- Verified signatures in `falkordb` 1.6.x (analyst, in the pinned venv):
  `ro_query(self, q, params=None, timeout: Optional[int] = None)`; `explain(self, query,
  params=None)` and `profile(self, query, params=None)` take **no timeout**.
- **`EXPLAIN` / `PROFILE` as a Cypher prefix is silently ignored by FalkorDB and the query is
  executed for real.** Verified: `GRAPH.QUERY cpg_falkorchat "EXPLAIN MATCH (m:METHOD) RETURN
  count(m)"` returned `747`, not a plan (analyst reproduced it). Plans come only from the separate
  `GRAPH.EXPLAIN` / `GRAPH.PROFILE` commands. A naive pass-through tool would therefore turn "let
  me just explain this" into "run the heavy traversal". This drives D5 (§4.4).
- **`GRAPH.PROFILE` executes the query, including writes** — FalkorDB docs, verbatim: *"Unlike
  `GRAPH.EXPLAIN`, `GRAPH.PROFILE` actually executes the query including any write operations
  (CREATE, DELETE, SET)"*. Reproduced live by teco on a throwaway graph. This is why D4 removes it.
- **Empty-key quirk** (`claude/graph-dba/falkordb-quirks.md:159-165`, verified): `GRAPH.QUERY`
  against a non-existent graph **materialises the key**; `GRAPH.RO_QUERY` does **not**.
  `GRAPH.EXPLAIN` is not a read-only command, so it is assumed to materialise the key too
  (*inferred* — deliberately not probed, since probing creates the key). §4.4 guards it.
- Result shape: `QueryResult.header` = `[[type, name], …]`, `result_set` = list of rows;
  cells are scalars, `None`, `falkordb.node.Node`, `falkordb.edge.Edge`. `str(Node)` renders
  `(:CpgNode:METHOD{CODE:"…",FULL_NAME:"…",…})` — can be very large (CPG `CODE` properties).
- Error strings the tool must classify:
  - bad Cypher → `redis.exceptions.ResponseError: errMsg: Invalid input 'R': expected … line: 1,
    column: 17 … errCtx: MATCH (m:METHOD RETURN m` (already excellent, pass it through)
  - missing graph → `ResponseError: Invalid graph operation on empty key`
  - write via RO → see above.

### 2.3 The CPG substrate: what is on disk, and what the rebuild can and cannot deliver

| | M2 baseline (HISTORY, 2026-07-19) | Live graph today |
|---|---|---|
| nodes / edges | 79,581 / 522,182 | **29,447 / 185,517** |
| `FILENAME` shape | `falkorchat/services.py`, `tests/…` | **`services.py`, `api.py`** (no dir prefix) |
| methods under `tests/` | 336 test entrypoints | **0** — 16 files, package only |
| `post_message` callers | **21** (2 prod + 19 tests) | **2** |

The live graph was built from `falkor-chat/server/falkorchat` alone, so tests are absent and paths
are unprefixed — which is exactly why **test-gap analysis is unverifiable on it today**. The M2
reload artifacts (`cpg/.cpg-artifacts/{cpg.bin,load.cypher,load_stdin.sh}`) no longer exist; `cpg/`
contains only `.gitignore`.

**The M2 numbers are gone for good, and that is not a defect of this feature.**
`git log --oneline --since=2026-07-18 -- falkor-chat/server` returns **8 commits** (verified
2026-07-25: `4f69a16`, `788e5bf`, `efdeeb3`, `670474a`, `98a3cc8`, `2ee6eba`, `1dd48a0`,
`2e1dad8`), introducing whole new modules (`guards.py`, `proof_defs.py`) and new test modules
(`test_process_flow.py`, `test_process_input.py`). A CPG built today is a CPG of **different
source**: node/edge counts, the `tests/` entrypoint count and the caller count will all
legitimately differ. Per **D1**, S8 rebuilds cleanly and the resulting counts become **the new
recorded baseline**; AC-3 is satisfied by **tool ≡ `redis-cli` equivalence** on that graph plus the
freshly recorded numbers (§7.2).

**Source-root hazard (verified 2026-07-25, and the reason S8 stages a copy).**
`falkor-chat/server/` contains, besides `falkorchat/` and `tests/`: `.venv/`, `.pytest_cache/`,
`.ruff_cache/`, `falkorchat.egg-info/`. First-party `.py` files: **41**. Files under `.venv/`:
**1,808**. `skills/joern-cpg/scripts/build-cpg.sh` passes the source path to `joern-parse`
verbatim with **no exclusion mechanism**, so parsing `falkor-chat/server` directly would drown the
CPG in third-party dependency code. S8 therefore parses a staged copy of exactly
`{falkorchat, tests}` — which also preserves the `falkorchat/…` + `tests/…` `FILENAME` prefixes
that every recipe's `STARTS WITH 'tests/'` filter depends on.

### 2.4 Claude Code mechanics (verified against `code.claude.com/docs/en/mcp` and `/en/skills`; analyst re-verified 2026-07-24)

These facts are durable and non-obvious. They do **not** stay buried in this plan — S7 folds them
into `skills/agent-standards/claude-code.md` §MCP, which is their canonical home.

- **Scopes:** `local` (`~/.claude.json`, private) → `project` (`.mcp.json` at repo root, version
  controlled) → `user` (`~/.claude.json`, all projects). Project-scoped servers **require a
  one-time interactive approval**; `enabledMcpjsonServers: ["cpg"]` in `.claude/settings.json`
  pre-approves by name, but only in a **trusted** workspace. Reset with
  `claude mcp reset-project-choices`. Consequence: a **headless** (`claude -p`) run in an
  un-approved workspace silently has no `cpg` server at all.
- `.mcp.json` shape: `{"mcpServers": {"<name>": {"command": …, "args": […], "env": {…},
  "timeout": <ms>}}}`. The **only** expansion syntax Claude Code performs is `${VAR}` and
  `${VAR:-default}` — a bare `$VAR` is passed through untouched.
- **`CLAUDE_PROJECT_DIR` is set in the spawned server's environment, not in Claude Code's own
  config-expansion environment.** So `${CLAUDE_PROJECT_DIR}` inside `.mcp.json` is useless, but an
  unbraced `$CLAUDE_PROJECT_DIR` handed to `bash -c` **is** expanded by the shell from the server
  env. That is the mechanism §4.3 uses.
- **Tool naming:** `mcp__<server>__<tool>`. That exact string is what goes in permission rules, a
  skill's `allowed-tools`, a subagent's `tools` field, and hook matchers.
- **`allowed-tools` in `SKILL.md` does not gate anything** — it *pre-approves* the listed tools
  for the turn that invokes the skill.
- **`tools:` in subagent frontmatter *is* an allowlist.** `analyst` and `architect` both declare
  `tools: Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent` — **they will not see
  the new MCP tool unless it is added to those lists.** `qa-engineer` and `graph-dba` omit
  `tools:` and inherit everything. This is the single easiest way to ship a feature that silently
  doesn't work for two of its three consumers.
- Subagent frontmatter also accepts `mcpServers` (`skills/agent-standards/claude-code.md:38`) —
  but the same file records at `:110-111` that **`skills`/`mcpServers` frontmatter is ignored for
  teammates**, and at `:46` that **plugin subagents ignore it**. See §4.3 for why it is rejected.
- **Stdio servers are not auto-reconnected** if the process dies mid-session (HTTP/SSE are). The
  server must be crash-proof; recovery is `/mcp` reconnect or a new session.
- Per-server `"timeout"` in `.mcp.json` is a hard wall-clock cap per tool call (default is
  effectively ~28 h). Set it.
- There is **no per-server tool-filtering mechanism** in Claude Code (analyst re-verified). A
  bought server's tool set cannot be reduced. This is load-bearing for §3.

### 2.5 Current surface to be changed

- `skills/cpg-analysis/SKILL.md` — frontmatter `description` (line 5 says "redis-cli
  GRAPH.QUERY"), `allowed-tools` (line 15), §1 "Connect and run a query" (lines 31–57: the bash
  block, the `GRAPH.LIST` line, the `GRAPH.EXPLAIN`/`GRAPH.PROFILE` bullet), §3 note about
  `$fn`/`$full` not being bound parameters (lines 88–95).
- `skills/cpg-analysis/references/impact-analysis.md:11-13` — the only recipe carrying a
  `redis-cli` invocation. `rca.md`, `code-review.md`, `test-gap.md` carry **no** connection
  framing (verified by grep; analyst spot-checked) — they need no edit.
- `skills/joern-cpg/references/cpg-model.md:72` mentions `GRAPH.QUERY` only as a property of the
  transformer's output — **leave it**, it is about the producer path.
- `claude/README.md` — rows **9 (`architect`)**, **16 (`qa-engineer`)** and **17 (`analyst`)** each
  carry a `cpg-analysis` clause and all three need updating (verified 2026-07-25; this corrects an
  earlier claim in the coordination doc that they did not). Row **12 (`graph-dba`)** does *not*
  mention `cpg-analysis`; its `GRAPH.PROFILE` tuning clause stays accurate and, under D4, is now
  the *only* profiling path — **no change needed**.
- root `AGENTS.md` has a `cpg-analysis` bullet (line 74) and **no `cpg/` entry in Structure**;
  `skills/README.md` catalog row says "(redis-cli GRAPH.QUERY)".
- `docs/requirements/joern-cpg-pipeline.md` FR-9 (line 76) + decision log — the AC-4 reversal.
- `claude/tico/kaizen/inbox.md:19` already flags the FR-9-vs-MCP contradiction — S6 resolves it,
  so the note must be closed out (S5).
- `skills/agent-standards/claude-code.md:168` — the `## MCP` section is **three lines of prose**
  with none of §2.4 in it, and there is **no OpenCode MCP section** anywhere.
- `claude/{analyst,architect,qa-engineer}/kaizen/history.md` — a dated entry is due for each agent
  whose source or catalog entry changes (`claude/AGENTS.md:35`, verified).

---

## 3. Build vs. buy

### 3.1 What exists (verified 2026-07-24, analyst re-verified against the npm registry)

**Official `FalkorDB/FalkorDB-MCPServer`** — npm `@falkordb/mcpserver`, **v1.3.0 published
2026-07-01**, MIT, `engines.node >= 18`, bin `falkordb-mcp`, deps `@modelcontextprotocol/sdk
^1.17.0`, `falkordb ^6.3.0`, `zod ^4.3.6`. stdio + streamable-HTTP.

Tools exposed (README, verbatim): `query_graph` *(query, graph, readOnly?, params?)*,
`query_graph_readonly` *(query, graph, params?)*, `list_graphs` *(none)*, `get_graph_schema`,
`get_node_schema`, `get_relationship_schema`, **`delete_graph` *(graph)***.

**Others considered:** `SecKatie/FalkorDB-MCPServer` (a fork of the same, same shape);
`mcp-neo4j-cypher` (two tools, but speaks **Bolt** — FalkorDB v4 does not, so it cannot connect);
generic Redis MCP servers (dozens of key/value tools, no `GRAPH.*` shaping).

### 3.2 Does buying satisfy FR-2?

**No, explicitly.** FR-2 is "**a single tool** taking exactly **two parameters**". The official
server exposes **7 tools**, its query tool takes **4 parameters**, and there is **no documented
way to disable or filter the tool set** (§2.4, re-verified). Permission `deny` rules can block a
call, but the tool still occupies the model's tool list — the surface is unchanged, so the
requirement is still violated. It also ships `delete_graph` into every agent's tool list on an
instance that holds `falkor-chat`'s `ws:*` and `reference` graphs, and offers **no plan capability**.

Adoption cost on top of the FR-2 violation:
1. **Install Node ≥18 inside WSL** (none today) — a new toolchain in a Python + shell repo. Using
   the Windows `npx` on the PATH would run the server as a Windows process talking to a WSL-hosted
   Docker container over interop — the fragility class the `severino`/LM-Studio notes document.
2. A `falkordb@^6.3.0` JS client against server v4.18.11 — an unpinned major we have not verified.
3. No control over result formatting or error text, so the "no graph → route to the `joern`
   agent" affordance the skill documents cannot be implemented.

### 3.3 Recommendation — **build**, and keep it tiny

A purpose-built stdio server: **~150 lines of Python**, one tool, two parameters, on dependencies
the repo has already pinned and live-verified (`mcp 1.28.x`, `falkordb 1.6.x`) with an in-repo
precedent for the exact API (`falkor-chat/server/falkorchat/mcp.py`). The requirement that decides
it is FR-2, which no off-the-shelf server meets; everything else (no Node, control over
errors/formatting/truncation, EXPLAIN routing) reinforces it.

**Trade-off accepted — the honest artifact count.** This is not "150 lines". The maintained set is
`cpg/mcp/{server.py, run.sh, setup.sh, requirements.txt, requirements-dev.txt, README.md,
tests/test_server.py}` + a venv lifecycle + a tracked `.mcp.json` + a `.claude/settings.json` key +
two agent frontmatters — in a repo with **no root-level test runner**. Mitigation (m-6): the
component ships **one smoke command**, referenced from `cpg/mcp/README.md` *and* root `AGENTS.md`'s
key-commands section, so a break is detectable without waiting for an agent's failed query:

```bash
cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q            # offline: contract + formatting + errors
cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q -m live    # requires FalkorDB up
```

We also forgo the official server's schema-discovery tools — which this component does not need,
because the CPG schema is fixed and documented once in `skills/joern-cpg/references/cpg-model.md`.

**Reversal trigger** (record it in HISTORY): if a future need arises for multi-tool graph access
(schema discovery, write paths, non-CPG graphs for other agents), revisit — at that point FR-2 no
longer binds and the official server becomes the cheaper answer.

---

## 4. Design

### 4.1 Where the artifact lives (D-loc)

```
cpg/
  .gitignore                 (exists — ignores .cpg-artifacts/; gains .venv/)
  mcp/
    server.py                the FastMCP stdio server
    run.sh                   launcher: resolves its own venv, execs server.py
    requirements.txt         mcp>=1.28,<1.29 · falkordb>=1.6,<1.7
    requirements-dev.txt     -r requirements.txt · pytest>=9.1,<10
    setup.sh                 idempotent: create .venv + pip install both files
    README.md                what it is, how to run/debug/restart, wiring other harnesses
    tests/test_server.py     unit tests (no FalkorDB) + `live`-marked integration tests
    .venv/                   NOT committed
```

**Why `cpg/`:** the directory already exists as the component's local home, it is outside
`skills/` (which is symlinked whole into three harnesses' config dirs — shipping a server and a
venv through that symlink is wrong), and `.mcp.json` can reference a stable repo-relative path.
This makes `cpg/` the component's code root for the first time, which is **evidence for**, not a
resolution of, `joern-cpg-pipeline.md` **OQ2** (R-8); the component's docs stay at repo-root `docs/`.

**Rejected:** `skills/cpg-analysis/server/` (couples harness config to the shared-skills symlink;
a skill package is documentation + copyable scripts, not a long-running process);
`falkor-chat/server/` (wrong component — the CPG component is deliberately distinct, decided
2026-07-12).

**Two requirements files, not `pyproject.toml` extras** (n-1): the review is right that
`requirements.txt` has no extras concept. `falkor-chat/server`'s `pyproject.toml` exists because it
is an installable package with a test suite; `cpg/mcp` is a single script run by path. Two plain
files keep `setup.sh` a two-line pip invocation. If `cpg/` ever grows into a package, migrating to
`pyproject.toml` is mechanical.

### 4.2 Runtime & launch (D-run)

- Dedicated venv at `cpg/mcp/.venv`, created by `setup.sh`. Do **not** reuse
  `falkor-chat/server/.venv` — that pins a chat application's dependency set; coupling them means
  a `falkor-chat` bump can break CPG access.
- `run.sh` resolves the interpreter relative to its **own** location
  (`HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`, then `exec "$HERE/.venv/bin/python"
  "$HERE/server.py"`), so only one path — `run.sh` itself — appears in `.mcp.json`. It fails
  loudly on stderr with a "run cpg/mcp/setup.sh" message if the venv is missing.
- **stdio** transport (`mcp.run(transport="stdio")`), one process per Claude Code session.
  Rejected HTTP: it would need a supervised long-lived process (systemd/compose) for no benefit
  at this scale — but note it is the escape hatch if session-scoped startup ever becomes painful
  (R-4).

### 4.3 Configuration scope & mechanism (D-cfg)

**Project scope, `.mcp.json` at the repo root, checked in — with no absolute path:**

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

- **Why this form is primary (B-2).** A cwd-relative path breaks whenever a session starts inside
  `falkor-chat/` or `salesperson/` — the normal way to work in this monorepo. An absolute path
  fixes that but **leaks the maintainer's home directory into a tracked file**, which
  `audit-team.sh` check 7 fails on (§4 preamble, R-9). The `bash -c` form is portable *and*
  audit-clean: Claude Code expands only `${VAR}` / `${VAR:-default}`, so the unbraced
  `$CLAUDE_PROJECT_DIR` passes through untouched and **bash** expands it from the spawned server's
  environment, where Claude Code does set it (§2.4, docs-verified by the analyst 2026-07-24).
- **If it does not connect** (S3 must actually check, from a subdirectory session): the fallback is
  **not** an absolute path in a tracked file. It is a **local-scope** registration —
  `claude mcp add --scope local cpg -- <repo-root>/cpg/mcp/run.sh` — which writes to
  `~/.claude.json` (untracked, per-machine, no audit surface). `cpg/mcp/README.md` documents this
  with a `<repo-root>` placeholder; the concrete path is never committed.
- `"timeout": 60000` caps a runaway tool call at 60 s (default is ~28 h).
- Add `"enabledMcpjsonServers": ["cpg"]` to `.claude/settings.json` so the approval is by name
  rather than a blanket `enableAllProjectMcpServers`. A **one-time interactive approval / trust
  dialog** is still expected on first run — an unavoidable human step (R-5).
- **Rejected — user scope** (`~/.claude.json`): invisible to the repo, un-reviewable, and it loads
  the CPG tool in every unrelated project.
- **Rejected — per-subagent `mcpServers:` frontmatter** (m-5). It looks cheaper (scope the server
  to the three consumers instead of every session), but: (a)
  `skills/agent-standards/claude-code.md:110-111` records that `skills`/`mcpServers` frontmatter is
  **ignored for teammates**, and `:46` that **plugin subagents ignore it** — so it would not
  reliably reach the consumers at all; (b) it spawns a server process per subagent invocation
  instead of one per session; (c) it moves the wiring into three agent files instead of one
  reviewable `.mcp.json`; (d) it does **not** remove the `tools:` allowlist problem (R-3), which is
  the actual delivery risk. Project scope + two allowlist edits is both cheaper and more visible.

**What this does *not* cover:** OpenCode and Kiro. The repo's skills are shared with both via
whole-directory symlinks, but **MCP wiring does not port** — OpenCode configures servers under its
own `opencode.json` `mcp` key and Kiro under `~/.kiro/settings/mcp.json`; neither reads
`.mcp.json`, and neither has any MCP server configured in this repo today. Therefore:
**`skills/cpg-analysis/SKILL.md` must keep the `redis-cli` path as a documented fallback**, not
merely for the "MCP is down" case but because it is the *only* path in two of the three harnesses
the skill is deployed to (R-7). The knowledge gap is closed in S7; actually wiring the other two
harnesses is deferred as **C-310**.

### 4.4 Tool contract (D-tool) — **frozen; implement exactly this**

**Server name** `cpg` · **tool name** `query` · **callable name `mcp__cpg__query`**.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `graph` | `str` | yes | FalkorDB graph key, caller-supplied (FR-4). Never defaulted, never inferred. |
| `cypher` | `str` | yes | Cypher text, verbatim, multi-line allowed (FR-3). |

**Registration — the exact decorator (M-3, empirically established by the analyst on the pinned
`mcp 1.28.1`):**

```python
@mcp.tool(
    name="query",
    description=TOOL_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
    structured_output=False,          # ← required; see below
    meta={"anthropic/maxResultSizeChars": MAX_RESULT_SIZE_CHARS},  # ← N-2; see Truncation
)
def query(graph: str, cypher: str) -> str: ...
```

`-> str` alone does **not** produce unstructured content on this SDK. The analyst built the exact
tool in `falkor-chat/server/.venv` and got `outputSchema: {'properties': {'result': {'type':
'string'}}, 'required': ['result']}` — and with an `outputSchema` present FastMCP returns the
payload **twice** (text content *and* `structuredContent`), so any char-capped result would arrive
at ~2× its budgeted size — defeating the very caps below. `structured_output=False` yields
`outputSchema: None` (verified on the same SDK). S2 asserts this in a unit test.

`readOnlyHint=True` is an **honest** declaration under this design: the plain path is
`GRAPH.RO_QUERY`, the `EXPLAIN` path is plan-only, and `PROFILE` is refused (below). No code path
can mutate a graph.

**Server instructions** (n-4, added v2.2 to match what shipped): the server is constructed as
`FastMCP(name="cpg", instructions=SERVER_INSTRUCTIONS)`. This is a *different* string from the tool
description below and does a different job: Claude Code's **tool search is on by default and MCP
tools are deferred**, so in a cold session — AC-1's exact scenario — the server `instructions` is
what the model reads to decide whether to search for this server's tools at all; it is returned by
the `initialize` handshake and truncated at **2 KB**. Keep it short, and keep it about *when to
reach for the tool* (call-graph / data-flow / impact-analysis / test-gap questions answered without
reading files), not about the tool's mechanics — those belong to the description.

**Tool description** (the model's routing contract — `cobb` finalises wording in S5, ~350 chars):
run a read-only OpenCypher query against a named FalkorDB graph, typically a loaded Joern CPG;
`graph` is the graph key, `cypher` is the query text; prefix `EXPLAIN` for a query plan (`PROFILE`
is not supported — it executes); FalkorDB is OpenCypher — **no APOC, no GDS**; schema for CPG
graphs is `skills/joern-cpg/references/cpg-model.md`.

#### D4a — execution mode: `GRAPH.RO_QUERY`, and *why* it is what makes this safe

The tool calls `Graph.ro_query`. Two distinct guarantees follow, and **both must be stated in the
docs** because the review found neither was:

1. **Writes are rejected server-side** with a clear message. The tool can name *any* graph on the
   instance — `ws:acme`, `ws:test`, `reference` — so this is what bounds the blast radius.
2. **A typo'd graph name cannot materialise a key.** `GRAPH.QUERY` on a non-existent graph creates
   it; `GRAPH.RO_QUERY` does not (§2.2). The read path is therefore safe *by construction*, not by
   validation. This is a design reason, not an accident.

This does **not** re-open the withdrawn auth / read-only-enforcement scope: FalkorDB stays open on
`:6379` with no auth, `redis-cli` remains unrestricted, and the `joern` load path is untouched. It
is a property of *this read-path tool*. Reversal is a one-word change (`ro_query` → `query`).

#### D5 — `EXPLAIN` only; `PROFILE` is refused (decision D4)

FR-2 forbids a second tool and a third parameter, so plan access is signalled by a **directive
prefix in the `cypher` text**. Three cases, and the third is not optional:

| Input (normalised per **D5a**, case-insensitive) | Behaviour |
|---|---|
| `EXPLAIN <cypher>` | Strip the keyword. **First confirm the graph exists** via an internal `GRAPH.LIST` (see below), then call `Graph.explain()` and return the rendered plan. |
| `PROFILE <cypher>` | **Refuse before touching the server.** Return: `PROFILE is not available through this tool: GRAPH.PROFILE executes the query, including writes. Use EXPLAIN for the plan. For measured profiling use the fallback: redis-cli -p 6379 GRAPH.PROFILE <graph> '<cypher>' --no-raw.` |
| anything else | `Graph.ro_query(cypher, timeout=CPG_MCP_TIMEOUT_MS)`. |

- **The `PROFILE` refusal is load-bearing, not cosmetic.** Without it, `PROFILE MATCH …` falls
  through to `ro_query`, FalkorDB *silently ignores the prefix* (§2.2) and the agent gets results
  where it asked for a profile — a wrong answer, not an error.

- **Why the `GRAPH.LIST` pre-check on the EXPLAIN path.** `GRAPH.EXPLAIN` is not a read-only
  command, so a typo'd graph name is assumed to materialise a junk key (§2.2, inferred). One extra
  round-trip on the plan path only — and the tool already needs `GRAPH.LIST` for the "graph does
  not exist" error message, so it is the same code. If the graph is absent, return that error
  instead of calling `explain()`.
- **No query timeout on the plan path** (M-6): `explain()` takes no `timeout` argument (§2.2).
  With `PROFILE` removed this is nearly moot — planning does not execute the traversal — and the
  `.mcp.json` 60 s wall remains the backstop. Say so in `cpg/mcp/README.md` rather than leaving it
  implicit.
- **Document the divergence loudly** in both the tool description and SKILL.md §1: raw FalkorDB
  *ignores* an `EXPLAIN` prefix and executes the query; the tool deliberately does not. A query
  copied verbatim from the tool to `redis-cli` with that prefix will **execute**.
- *Rejected:* keeping `PROFILE` behind an explain-first write-operator sniff (more code, more
  failure modes, and it still leaves `readOnlyHint` arguable — D4 rules against it); dropping plan
  access entirely (a regression vs `redis-cli` for exactly the heavy traversals that need it); a
  `mode` parameter (violates FR-2).

##### D5a — how the directive is detected (**n-3**: the sniff must be comment-blind)

A naive `cypher.strip().upper().startswith("PROFILE")` is **not** sufficient. Verified live by the
analyst (read-only, v4.18.11): both `/* hi */ PROFILE MATCH (m:METHOD) RETURN count(m)` and
`// hi⏎PROFILE MATCH …` are accepted by `GRAPH.RO_QUERY` and return **747 — results**. A
comment-prefixed directive would therefore classify as "plain", fall through to `ro_query`, and
hand the caller results where it asked for a profile or a plan: exactly the wrong-answer class D5
exists to prevent. `split_directive()` must implement this normalisation **exactly**:

1. Work on a scanning cursor over the **original** string; consume a prefix only.
2. Loop until nothing more is consumed:
   1. consume leading whitespace (`str.lstrip()` semantics — space, `\t`, `\r`, `\n`, `\f`, `\v`);
   2. if the remainder starts with `//`, consume through the next `\n` inclusive (or to end of
      string if there is none);
   3. if the remainder starts with `/*`, consume through the next `*/` inclusive. **If there is no
      closing `*/`, consume nothing and stop** — the statement is malformed; classify it `"query"`
      and let FalkorDB return its own error verbatim. Comments do not nest in Cypher: match the
      *first* `*/`, non-greedy.
3. On the remainder, match `^(EXPLAIN|PROFILE)\b` **case-insensitively** — `\b` (or an explicit
   "next char is end-of-string or not `[A-Za-z0-9_]`") is required so `EXPLAIN_ME`, `PROFILER` and
   `PROFILEDATA` do **not** match, while `EXPLAIN(`, `EXPLAIN\n` and `explain\tMATCH …` do.
4. Return `("explain" | "profile" | "query", cypher_to_send)`.

Two rules about what is *sent*, and they differ by path:

- **Plain path — send the caller's string byte-for-byte.** The normalisation above is used for
  **classification only**; it never rewrites the query. FR-3 promises verbatim transmission, and
  comment-stripping the payload would break `CONTAINS '// x'`-style literals. (The scan stops at
  the first non-whitespace, non-comment character, so comment markers *inside* a string literal
  later in the query are never seen — no lexer is needed.)
- **`EXPLAIN` path — send the text following the matched keyword**, left-stripped. The consumed
  comment/whitespace prefix is discarded (it is leading trivia; dropping it cannot change the
  plan). The `PROFILE` path sends nothing at all.

#### Result rendering

Plain text, not JSON — CPG columns are long strings and JSON roughly doubles the token cost while
adding nothing an agent needs.

```
graph=cpg_falkorchat · rows=2 · 12.3ms
caller | file | line
falkorchat/api.py:<module>.build_router.post_message | falkorchat/api.py | 139
falkorchat/mcp.py:<module>.send_message | falkorchat/mcp.py | 53
```

- Header line from `QueryResult.header` column names, joined by ` | `.
- Cell rendering: `None` → `null`; `str` verbatim; `Node`/`Edge` → `str(value)`; list/map →
  `repr`. **Newlines and tabs inside a cell are escaped** (`\n` → `\\n`) so one row is always one
  line. Pipes inside values are *not* escaped — documented; return a single column when a value
  must be copied exactly.
- Empty result → the same first line with `rows=0` plus `(no rows)` and the column names, so the
  agent can tell "query ran, matched nothing" from "query failed".
- Non-`RETURN` statements (none expected under RO) → report the stats line only.

#### Truncation (m-1 — fully specified; **N-2**: sized and placed against Claude Code's own output limits)

Caps, all env-overridable, since FR-2 forbids a `limit` parameter: `CPG_MCP_MAX_ROWS` (default
**200**), `CPG_MCP_MAX_CELL` chars (default **300**), `CPG_MCP_MAX_CHARS` total (default
**30000** — see *Why 30 000* below; this was 60 000 in v2 and was lowered by N-2).

- **Truncation is display-only.** The client materialises the full result set before formatting,
  so memory and latency are bounded by the *query*, not by the caps; the reported row count is
  always exact. Say this in `cpg/mcp/README.md` so nobody reads the caps as a safety limit on the
  query.
- **Row cap binding** → keep the first `CPG_MCP_MAX_ROWS` rows and use exactly this notice:
  `… truncated: showing 200 of 79581 rows (row cap) — results are unordered unless the query has
  ORDER BY; narrow with LIMIT, a projection, or an aggregate.`
- **Char cap binding** → after row and cell capping, if the rendered body still exceeds
  `CPG_MCP_MAX_CHARS`, **drop whole rows from the tail** (never a partial row) until it fits, and
  use the same notice with the char cap named:
  `… truncated: showing 87 of 79581 rows (char cap 30000) — results are unordered unless the query
  has ORDER BY; narrow with LIMIT, a projection, or an aggregate.`
- **Whenever a truncation notice exists it is emitted TWICE — as the first line of the payload and
  again as the last line**, byte-identical, above the stats line and below the last row:

  ```
  … truncated: showing 200 of 79581 rows (row cap) — results are unordered unless …
  graph=cpg_falkorchat · rows=79581 · 812.4ms
  caller | file | line
  …200 rows…
  … truncated: showing 200 of 79581 rows (row cap) — results are unordered unless …
  ```

  When nothing is truncated, neither line appears (an untruncated payload gains nothing). The
  notice counts toward `CPG_MCP_MAX_CHARS`: reserve `2 × len(notice)` before deciding how many
  rows fit, so adding the notice can never push the payload back over the cap.
- A cell cut appends `…(+N chars)`.
- The "unordered" clause is not padding: the first 200 rows of an unordered result set are
  arbitrary, and an agent that reads "showing 200 of 79581" may otherwise draw a conclusion from a
  non-deterministic sample.
- The notice lines are **display metadata, not data**: S9's equivalence comparison (§7.2 AC-3) and
  any diff against `redis-cli` output must ignore lines beginning with `… truncated:`.

Rationale for having caps at all: `MATCH (n) RETURN n` on a CPG is tens of thousands of rows with
multi-KB `CODE` properties — one such call would blow the context.

##### Why the notice is duplicated, why 30 000, and why `_meta` — the harness layer (N-2)

Claude Code applies **its own** limit on top of ours, and v2 collided with it. Verbatim from
`code.claude.com/docs/en/mcp` § *MCP output limits and warnings* (re-fetched 2026-07-25): a warning
is displayed *"when any MCP tool output exceeds 10,000 tokens"*; *"the default maximum is 25,000
tokens"* (`MAX_MCP_OUTPUT_TOKENS`); and *"Without the annotation, results that exceed the default
threshold are **persisted to disk and replaced with a file reference in the conversation**."* At
2.5–3.5 chars/token for identifier-dense CPG text, a 60 000-char payload is ~17–24 k tokens —
always over the warning threshold and plausibly at the substitution threshold. The failure is
perverse: the run that binds the char cap is exactly the run whose honest notice matters, and in v2
that notice was the payload's **last** line — the first thing lost to a tail-side cut or a
file-reference substitution. The agent would see a truncated table and no sign that it was
truncated. All three of the reviewer's remedies are adopted, each doing a different job:

1. **Notice first *and* last (the durable one).** Harness-independent: it survives any tail-side
   clipping, a future limit change, a file-reference preview, or a different harness entirely
   (`_meta` and token limits are Claude-Code-specific; this is not). Cost: one duplicated line.
2. **Default lowered 60 000 → 30 000 (removes the collision at the source).** 30 000 chars is
   ≤ 15 k tokens even at a pessimistic 2 chars/token — comfortably under the 25 k substitution
   limit, and usually under the 10 k warning. It is **nearly free in practice**: for the recipes'
   typical projections (`FULL_NAME | FILENAME | LINE_NUMBER` ≈ 100–150 chars/row) the 200-row cap
   binds first at ~20–30 k chars, so the char cap only bites on wide `CODE` projections — precisely
   the payloads that should be cut. Rejected: keeping 60 000 and relying on remedy 1 alone (it
   makes truncation *visible* but still ships a file-reference-sized payload, defeating the token
   economy that motivates caps at all).
3. **`_meta["anthropic/maxResultSizeChars"]` (pins the unit).** Docs: *"Tools that set
   `anthropic/maxResultSizeChars` use that value instead for text content, regardless of what
   `MAX_MCP_OUTPUT_TOKENS` is set to"*, hard ceiling **500 000** chars. Declaring it converts the
   persist-to-disk threshold from a **token estimate we cannot compute** into a **char budget in
   the same unit as our own cap**, and makes the behaviour independent of the user's environment.
   Set it to `MAX_RESULT_SIZE_CHARS = min(2 * CPG_MCP_MAX_CHARS, 500_000)` computed at import from
   the same env var, so raising `CPG_MCP_MAX_CHARS` raises the declared threshold in lockstep and
   the two can never disagree; the 2× headroom covers the stats line, header and both notice lines.
   Note this is a *pin*, not a raise — the tool never emits more than ~30 000 chars anyway.

**Verified on the pinned SDK** (`mcp 1.28.1` in `falkor-chat/server/.venv`, 2026-07-25):
`FastMCP.tool()` accepts `meta: dict[str, Any] | None`, `mcp.types.Tool` carries a `meta` field, and
a probe registration emitted `"_meta": {"anthropic/maxResultSizeChars": 60000}` in the `tools/list`
entry with **no** `outputSchema` (`structured_output=False` still holds). So this is one kwarg, not
a new mechanism. Risk if the key is ever renamed or ignored: unknown `_meta` keys are reserved for
exactly this by the MCP spec and are silently ignored by other harnesses — remedies 1 and 2 hold
regardless, which is why `_meta` is the *third* line of defence and not the only one.

Record the chosen combination in `cpg/mcp/README.md` next to the "truncation is display-only" note,
with the honest bounds on the env override: because the annotation scales with the cap, raising
`CPG_MCP_MAX_CHARS` stays free of disk-substitution until `2 × cap` hits the 500 000-char ceiling
(i.e. a cap of 250 000) — but the 10 k-token **warning** returns above roughly 25 000–35 000 chars,
and every char is context an agent pays for. Raise it for a specific investigation, not by default.

#### Query timeout

`CPG_MCP_TIMEOUT_MS`, default **30000**, passed as `ro_query(..., timeout=…)`; strictly below the
60 s `.mcp.json` wall-clock cap so the server, not the harness, produces the error message. Does
not apply to the `EXPLAIN` path (above).

#### Error behaviour

Every failure returns a *curated, actionable* message. FastMCP wraps a raised exception as
`ToolError("Error executing tool query: <message>")`, so the text reaches the agent; the server
must never let an exception escape any other way (a crashed stdio server is **not** reconnected —
§2.4, R-4).

| Condition | Detection | Message (shape) |
|---|---|---|
| FalkorDB unreachable | `redis.exceptions.ConnectionError` / `TimeoutError` | `FalkorDB unreachable at <host>:<port>. Start it (falkor-chat/scripts/start_falkordb.sh, or docker start falkordb-dev) and retry.` |
| Graph does not exist | `ResponseError` containing `empty key`, **or** the EXPLAIN pre-check | `Graph '<graph>' does not exist. Loaded graphs: <GRAPH.LIST>. If no CPG is loaded, building and loading one is the joern agent's job (joern-cpg pipeline) — this tool only queries.` |
| `PROFILE` requested | directive sniff, before any server call | the D5 refusal text above |
| Cypher syntax/semantic error | other `ResponseError` | FalkorDB's message verbatim (it carries line/column/context) + one line: `FalkorDB is OpenCypher (no APOC/GDS); CPG property keys are UPPER_CASE — see skills/joern-cpg/references/cpg-model.md.` |
| Write attempted | `ResponseError` mentioning `RO_QUERY` | `This tool is read-only (GRAPH.RO_QUERY). Loading/writing a CPG goes through the joern pipeline, or redis-cli for ad-hoc writes.` |
| Query timeout | `ResponseError` mentioning timeout | `Query exceeded <N> ms. Bound variable-length traversals (*1..N), add LIMIT, or prefix EXPLAIN to inspect the plan first.` |
| Anything else | `Exception` | `Unexpected error: <type>: <message>` — never a traceback, never a crash. |

The graph-list-on-missing-graph affordance implements the skill's documented "no graph → route to
`joern`" path **inside the error message**; it is not a second tool (no `list_graphs` is
advertised) and therefore does not touch FR-2. It is also the only in-tool graph *discovery*
mechanism — SKILL.md must say so explicitly (m-2, S4).

**Connection handling.** One module-level `FalkorDB(host, port)` client, created lazily on first
call and reused (redis-py pools and auto-reconnects). Env: `FALKORDB_HOST` (default `127.0.0.1`),
`FALKORDB_PORT` (default `6379`). Module import must not connect.

### 4.5 Skill and agent-surface changes (D-surface)

`skills/cpg-analysis/SKILL.md`:
- **frontmatter `description`** — replace "(redis-cli GRAPH.QUERY)" with the MCP tool as the
  path; keep ≤1024 chars; keep the "graph name is caller-supplied" and "building/loading routes
  to `joern`" clauses.
- **frontmatter `allowed-tools`** — `mcp__cpg__query, Bash, Read`. `Bash` **stays**: it is the
  fallback path and other harnesses have nothing else. (Reminder for cobb: this field
  pre-approves, it does not restrict.)
- **§1 "Connect and run a query"** — rewritten around the tool: two parameters, no shell quoting,
  multi-line accepted verbatim, graph name still caller-supplied and never hardcoded. Keep the
  read-only rule and the "no graph / FalkorDB down → route to the `joern` agent" paragraph (now
  also surfaced by the tool's error text), and add:
  - **(a) graph discovery, explicitly** (m-2): graph names come from the caller; to discover them,
    use the `redis-cli GRAPH.LIST` fallback or read the tool's not-found error, which lists the
    loaded graphs. There is deliberately no `list_graphs` tool (FR-2).
  - **(b)** the `EXPLAIN` prefix convention, the divergence from raw FalkorDB, and that **`PROFILE`
    is not available through the tool** — it is a `redis-cli` operation, `graph-dba`'s territory.
  - **(c)** truncation defaults, that truncation is display-only, and how to narrow.
  - **(d)** a short **fallback** block retaining today's `redis-cli … GRAPH.QUERY … --no-raw`
    snippet, labelled for "outside Claude Code (OpenCode/Kiro), or when the tool is unavailable".
- **§3 `$fn`/`$full` note** — keep the literal-substitution rule and update the reason: neither
  `redis-cli` **nor this tool** binds Cypher parameters (a `params` argument would be a third
  parameter, which FR-2 forbids).
- **§2 gotchas, §4 navigation, and the Python-only coverage boundary** — unchanged.

`skills/cpg-analysis/references/impact-analysis.md` — replace the `redis-cli` invocation in the
preamble (lines 11–13) with "pass the graph key and the Cypher below as the tool's two
parameters". The "iterate Q1 by name" note for transitive upstream (lines ~76–80) **stays as is**;
replacing it with a single-query closure is deferred to **C-308** (D3). The other three recipes
need no edit (verified: no connection framing).

Agent surfaces:
- **`claude/analyst/analyst.md` and `claude/architect/architect.md`: add `mcp__cpg__query` to the
  `tools:` allowlist.** Without this the tool is invisible to them (§2.4, R-3). `qa-engineer` and
  `graph-dba` inherit it automatically.
- Descriptions: the three consumers' CPG lines say "uses the `cpg-analysis` skill" — that stays
  true and needs no rewording. **Do not** grow descriptions for its own sake.
- `claude/README.md` rows **9 / 16 / 17** (`architect`, `qa-engineer`, `analyst`): mention the MCP
  tool where the row already mentions the skill. Row 12 (`graph-dba`) needs **no** change (§2.5).
- `claude/{analyst,architect,qa-engineer}/kaizen/history.md`: a dated entry each, per
  `claude/AGENTS.md:35` (M-4).
- `claude/tico/kaizen/inbox.md:19`: close out the FR-9-contradiction note, pointing at S6's edit.

---

## 5. Implementation steps

Each step is independently checkable. **Ownership constraint:** `teco` cannot own any step —
its `Write`/`Edit` is harness-restricted to `docs/plans/`, so every documentation edit outside that
directory is assigned to an agent that can actually write it. `cobb` owns agent/skill/prompt
surfaces; `coder` owns the module docs (`docs/requirements/`, `docs/BACKLOG.md`, `docs/HISTORY.md`)
— either may be swapped for the other, but **never** back to `teco`.

| # | Step | Owner | Depends on |
|---|---|---|---|
| S1 | venv + deps + `setup.sh` + smoke command | `devops` | — |
| S2 | `server.py` + `run.sh` + unit tests | `coder` | S1 |
| S3 | `.mcp.json` + settings + live connect (**one human approval**) | `devops` | S2 |
| S4 | Skill surface (`SKILL.md`, impact recipe, `skills/README.md`) | `cobb` | plan (contract frozen) |
| S5 | Agent wiring (`tools:`, catalogs, root `AGENTS.md`, kaizen histories, tico inbox) | `cobb` | plan; inbox close-out after S6 |
| S6 | Requirements reconciliation: FR-9 reversal (AC-4) + AC-1/AC-3 corrections (D2, D3) | `coder` | plan |
| S7 | Knowledge capture into `skills/agent-standards/` (Claude Code §MCP + OpenCode subsection) | `cobb` | plan; confirm against S3's result |
| S8 | CPG rebuild + **record the fresh baseline** (⚠ destructive, approved by D1) | `joern` | — |
| S9 | Live acceptance AC-1…AC-4 + test plan & report | `qa-engineer` | S3, S4, S5, S6, S8 |
| S10 | BACKLOG (M3 + C-301…C-310) + HISTORY | `coder` | S9 |

**Parallelism:** S4–S7 (docs) run in parallel with S1–S3 (code) — the tool contract in §4.4 is
frozen by this plan, so no doc edit waits on code. S8 is independent of everything and should
start early (longest latency: a Joern parse + load). S9 joins all of them.

**Audit hygiene, for every step that writes a tracked file (B-2, R-9):** `audit-team.sh` already
returns `RESULT: FAIL` on **pre-existing** home-path/username leaks (`.claude/settings.json`, two
kaizen inboxes, an M2 plan doc — confirmed by teco 2026-07-25). The done-condition everywhere is
therefore **"no *new* failures"**, measured as a diff, not a pass:

```bash
bash claude/scripts/audit-team.sh > "$TMP/audit-before.txt" 2>&1 || true   # before your edits
bash claude/scripts/audit-team.sh > "$TMP/audit-after.txt"  2>&1 || true   # after
diff "$TMP/audit-before.txt" "$TMP/audit-after.txt"                        # must show no new FAIL lines
```

Keep both files out of the repo. Fixing the pre-existing leaks is **C-309**, not this feature.

---

### S1 — venv, dependencies, setup script · `devops`

Create `cpg/mcp/requirements.txt` (`mcp>=1.28,<1.29`, `falkordb>=1.6,<1.7`) and
`cpg/mcp/requirements-dev.txt` (`-r requirements.txt`, `pytest>=9.1,<10`), mirroring the bounds
live-verified in `falkor-chat/server/pyproject.toml`. Write `cpg/mcp/setup.sh` (idempotent:
`python3 -m venv .venv` + `pip install -r requirements-dev.txt`). Append `.venv/` to
`cpg/.gitignore`. Write the `cpg/mcp/README.md` skeleton (S3 fills it in).

**Done when:** `cpg/mcp/setup.sh` run twice in a row succeeds; `cpg/mcp/.venv/bin/python -c "import
mcp.server.fastmcp, falkordb"` exits 0; no new `audit-team.sh` failures.

### S2 — the server · `coder`

Implement `cpg/mcp/server.py` to the contract in §4.4 — **exactly** as specified, including
`structured_output=False`, the server-level `instructions=` string (n-4, §4.4), the `PROFILE`
refusal, and the `GRAPH.LIST` pre-check on the `EXPLAIN`
path — plus `cpg/mcp/run.sh`. Keep the pure logic in importable, side-effect-free functions so it
is unit-testable without FalkorDB:

```python
def split_directive(cypher: str) -> tuple[str, str]     # ("query"|"explain"|"profile", cypher)
def render_cell(value: object, max_chars: int) -> str
def format_result(graph: str, header: list, rows: list, elapsed_ms: float,
                  max_rows: int, max_cell: int, max_chars: int) -> str
def explain_error(exc: Exception, graph: str, host: str, port: int,
                  graphs: list[str] | None) -> str      # returns the curated message
def query(graph: str, cypher: str) -> str               # the @mcp.tool()
```

Constraints: never raise outside the curated path; **no `print()` to stdout** (stdio transport owns
it — log to stderr only); no hardcoded graph name anywhere (FR-4); module import must not connect.

Unit tests in `cpg/mcp/tests/test_server.py` (no FalkorDB):

1. **Tool contract:** `list_tools()` returns **exactly one** tool named `query`; `inputSchema` has
   exactly the required properties `{graph, cypher}`; **`outputSchema is None`** (M-3 — this is the
   regression guard for the double-payload bug); **`meta == {"anthropic/maxResultSizeChars":
   60000}`** at default settings, i.e. `2 × CPG_MCP_MAX_CHARS` (N-2).
2. Directive splitting **per D5a**: `EXPLAIN `, `explain\n`, leading whitespace, a query merely
   *containing* the word, no directive; `PROFILE ` / `profile\t` → the refusal path; and the
   **comment-blind** cases (n-3), which are the regression guard for a live-verified hole:
   `/* hi */ PROFILE MATCH …` → profile; `// hi\nPROFILE MATCH …` → profile;
   `  /* a */\n// b\n  explain match (n) return n` → explain **with `cypher_to_send ==
   "match (n) return n"`**; `/* unterminated PROFILE MATCH …` (no closing `*/`) → **plain**, sent
   verbatim; `EXPLAIN_ME …` / `PROFILER …` → **plain** (the `\b` boundary);
   `MATCH (n) WHERE n.CODE CONTAINS '// PROFILE' RETURN n` → **plain**, and the returned string is
   byte-identical to the input.
3. `PROFILE` refusal happens **before** any client call (assert the fake client saw nothing) —
   including for the comment-prefixed spellings above.
4. Cell rendering: `None`, long string, embedded newline/tab, a `Node`-like stub.
5. Table formatting: header, empty result, **row-cap** truncation notice with exact counts,
   **char-cap** truncation dropping whole rows from the tail with the char-cap notice, and the
   "unordered" clause present in both. Plus (N-2): the default `CPG_MCP_MAX_CHARS` is **30000**
   (guard against a silent bump back); a truncated payload's **first and last lines are the
   byte-identical notice** and the notice appears exactly twice; an untruncated payload contains
   **no** notice line; and the notice-bearing payload still measures ≤ `CPG_MCP_MAX_CHARS`.
6. Error classification for each row of the §4.4 table using synthetic
   `ResponseError`/`ConnectionError` instances.
7. EXPLAIN pre-check: with a stub whose `GRAPH.LIST` omits the name, `explain()` is **never
   called** and the not-found message is returned.

Live-marked tests (`@pytest.mark.live`, deselected by default — same convention as
`falkor-chat/server/pyproject.toml`): a real 2-column query, a multi-line query, a missing graph, a
syntax error, a write rejection, an `EXPLAIN` returning plan text.

**Done when:** `cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q` is green; `-m live` is green against the
live FalkorDB; and a manual stdio smoke test (`initialize` + `tools/list` + `tools/call` round
trip) lists **exactly one tool** named `query` with **exactly two** required parameters and **no**
`outputSchema`, and the `initialize` response carries a **non-empty `instructions`** string under
2 KB (n-4).

> **Delivered 2026-07-25 (v2.2 audit correction).** All of the above is implemented in
> `cpg/mcp/server.py`, including `instructions=` — `SERVER_INSTRUCTIONS` (408 chars) at
> `server.py:103`, passed at `server.py:399`. Verified live by `architect`: an `initialize`
> handshake over `cpg/mcp/run.sh` returns it verbatim, and Claude Code surfaces it in-session as
> the "MCP Server Instructions" block. See §10 v2.1 → v2.2.

### S3 — wire it into Claude Code · `devops` (needs one human approval)

Write `.mcp.json` per §4.3 — the `bash -c "$CLAUDE_PROJECT_DIR/…"` form, **no absolute path**. Add
`"enabledMcpjsonServers": ["cpg"]` to `.claude/settings.json`. Complete `cpg/mcp/README.md`: what
it is, the two parameters, env vars and defaults, `setup.sh`, the **smoke command** (§3.3), how to
restart (`/mcp` → reconnect, or restart the session — stdio servers are not auto-reconnected), the
read-only + truncation (display-only) + `EXPLAIN`-only semantics with the `redis-cli GRAPH.PROFILE`
pointer, the local-scope fallback written with a `<repo-root>` placeholder, and a short "wiring it
elsewhere" note (the command line is identical: run `cpg/mcp/run.sh` over stdio).

**Done when:** `claude mcp list` shows `cpg` **connected** (not `⏸ Pending approval`); `/mcp` shows
a tool count of **1**; a real session run of `mcp__cpg__query(graph="cpg_falkorchat",
cypher="MATCH (m:METHOD) RETURN count(m)")` returns a formatted count; **the same works from a
session started in a subdirectory** (e.g. `falkor-chat/`) — this is the check that validates the
`$CLAUDE_PROJECT_DIR` form, and if it fails, switch to the local-scope fallback and record that in
`cpg/mcp/README.md`; and no new `audit-team.sh` failures (specifically: `git grep` for the home
path finds nothing new).

### S4 — skill surface · `cobb`

Apply §4.5 to `skills/cpg-analysis/SKILL.md` and `references/impact-analysis.md`. Update
`skills/README.md`'s `cpg-analysis` catalog row (drop "(redis-cli GRAPH.QUERY)" → MCP tool; note
the `redis-cli` fallback and that MCP wiring is Claude-Code-only today).

**Portability spot-check (m-3):** the skill is symlinked into three harnesses, and
`allowed-tools` now names a Claude-only tool. Load `cpg-analysis` once under OpenCode and confirm
the unknown `mcp__cpg__query` entry is **ignored, not rejected**; record the result in one line in
`skills/README.md` (or in `agent-standards`, alongside S7). If OpenCode cannot be exercised in this
environment, record it explicitly as **unverified** with that reason and fold it into **C-310** —
do not leave it silently untested.

**Done when:** no stale `redis-cli`-as-*the*-path claim remains in `skills/cpg-analysis/`
(`grep -rn "redis-cli" skills/cpg-analysis/` returns only the explicitly-labelled fallback block
and the `GRAPH.PROFILE`/`GRAPH.LIST` pointers); frontmatter `description` ≤1024 chars;
`allowed-tools` lists `mcp__cpg__query`; SKILL.md §1 states graph discovery (m-2) and the
`PROFILE`-is-unavailable rule; the skill passes cobb's single-artifact prompt-quality lint
(`agent-maintenance` §7); OpenCode spot-check recorded (verified or explicitly unverified).

### S5 — agent wiring & catalogs · `cobb`

- Add `mcp__cpg__query` to the `tools:` allowlists of `claude/analyst/analyst.md` and
  `claude/architect/architect.md`.
- `claude/README.md` rows **9 / 16 / 17** — mention the MCP tool in the existing `cpg-analysis`
  clause. Row 12 (`graph-dba`) unchanged.
- root `AGENTS.md` — add a **`cpg/` entry to the Structure section** (component code home: the MCP
  query server; artifacts gitignored); note in the `cpg-analysis` bullet that the read path is the
  `mcp__cpg__query` tool with `redis-cli` as fallback and that `.mcp.json` is the repo's first MCP
  wiring, Claude-Code-only; add the **smoke command** (§3.3) to the key-commands section (m-6).
- `claude/{analyst,architect,qa-engineer}/kaizen/history.md` — a dated entry each (M-4): what
  changed (`tools:` gains `mcp__cpg__query` for the first two; catalog row for all three) and the
  `C-304` reference.
- `claude/tico/kaizen/inbox.md:19` — close out the FR-9-contradiction note with a pointer to S6's
  edit (or, if cobb prefers to batch it, leave it explicitly marked "resolved by C-305, pending
  distillation"). **Do not leave it silently open** — it is the exact note AC-4 resolves.

**Done when:** `claude/AGENTS.md:35`'s "in the same change" rule is satisfied for all three agents
(source/catalog + kaizen history); catalogs, root `AGENTS.md` and the skill agree; no new
`audit-team.sh` failures. The *live* proof that the allowlist edit worked (a cold `analyst` and a
cold `architect` each calling the tool) is **not** part of this step's done-condition — it needs
S3, so it is verified in S9 (m-4).

### S6 — requirements reconciliation · `coder`

`docs/requirements/joern-cpg-pipeline.md` (AC-4):
- Rewrite **FR-9** to record the reversal — agents access the loaded CPG through the `cpg-analysis`
  skill, which now queries FalkorDB via the **`mcp__cpg__query` MCP tool** (`redis-cli GRAPH.QUERY`
  retained as fallback and as the only path outside Claude Code), superseded by
  [`./cpg-query-access.md`](../requirements/cpg-query-access.md) FR-1/FR-2. *(Sibling link form —
  n-2: from inside `docs/requirements/`, write `./cpg-query-access.md`.)*
- Add a dated decision-log entry: "2026-07-… — access mechanism reversed: MCP tool over
  `redis-cli`; see cpg-query-access.md". Update the doc's status line.

`docs/requirements/cpg-query-access.md` — three edits, all stakeholder-ruled:
- **AC-3 (D1 + D2):** the M2 figures are not reproducible (the source moved 8 commits, §2.3).
  Restate AC-3 as: *the M2 acceptance queries, re-run through the tool against a freshly built
  `cpg_falkorchat`, return **byte-identical value sets** to the same queries run through
  `redis-cli GRAPH.QUERY` on the same graph; the resulting counts are recorded as the new
  baseline.* Correct the stale "30 untested methods" to "39 rows / 32 distinct method names" where
  the historical figure is cited, and mark it superseded by the fresh baseline.
- **AC-1 (D3):** the demonstrated question is **direct** callers of `post_message`, not transitive
  — this feature changes how Cypher is transmitted, not how powerful Cypher is. Note that a bounded
  transitive upward-closure query is tracked as **C-308** (`graph-dba`).
- Set the document's status on delivery.

**Done when:** a reader of either requirements doc reaches the same conclusion about the access
mechanism; `grep -n "redis-cli" docs/requirements/joern-cpg-pipeline.md` shows only
historical/fallback framing; `grep -n "30 untested" docs/requirements/` is empty; AC-1 and AC-3
match what S9 actually verifies; no new `audit-team.sh` failures.

### S7 — knowledge capture into `skills/agent-standards/` · `cobb`

`skills/agent-standards/claude-code.md` §MCP (line 168) is three lines of prose and is the declared
authoring source for exactly these perishable specifics (M-5). When this plan is archived the facts
in §2.4 go with it unless they are moved. Fold in: scopes + approval/trust dialog (including that a
headless run in an un-approved workspace silently has **no** server), `.mcp.json` shape, the
`${VAR}`-only expansion rule and the `CLAUDE_PROJECT_DIR`-in-the-server's-env consequence with the
`bash -c` idiom, per-server `timeout` semantics, `mcp__<server>__<tool>` naming and where that
string is used, the subagent `tools:`-allowlist interaction, the `mcpServers:` frontmatter caveats
(`:38`, `:46`, `:110-111`), stdio servers not auto-reconnecting, and the **verified absence of any
per-server tool-filtering mechanism**.

Add a short **OpenCode MCP** subsection (the gap teco's inventory found): OpenCode configures
servers under `opencode.json`'s `mcp` key, this repo wires none, and MCP config does **not** port
through the `skills/` symlink even though `SKILL.md` does. Fold in S4's OpenCode spot-check result
if it exists.

**Done when:** a reader wiring a second MCP server in this repo needs neither this plan nor the
official docs for any of the above; the file passes cobb's lint; cobb's own `kaizen/history.md`
records the `agent-standards` change (per `claude/AGENTS.md`); no new `audit-team.sh` failures.

### S8 — rebuild the CPG and record the fresh baseline · `joern` (⚠ destructive — **approved by D1**)

The live graph is a package-only build with **zero** test methods, which is why test-gap analysis
is unverifiable today (§2.3). D1 authorises a clean, destructive rebuild. **Stage the source first**
— parsing `falkor-chat/server` directly would pull in `.venv` (1,808 third-party `.py` files vs 41
first-party) and `.pytest_cache`/`.ruff_cache`/`*.egg-info`, and `build-cpg.sh` has no exclusion
mechanism (§2.3):

```bash
# from <repo-root>; SRC is scratch, outside the repo
SRC=/tmp/cpg-src/falkor-chat-server
rm -rf "$SRC"; mkdir -p "$SRC"
cp -r falkor-chat/server/falkorchat "$SRC"/
cp -r falkor-chat/server/tests      "$SRC"/
find "$SRC" -name __pycache__ -type d -prune -exec rm -rf {} +

# delete explicitly (NOT via --reset — see below), then load into the empty key
redis-cli -p 6379 GRAPH.LIST                    # snapshot: five graphs expected
redis-cli -p 6379 GRAPH.DELETE cpg_falkorchat   # ← trips the guard; approve HERE, deliberately

skills/joern-cpg/scripts/pipeline.sh "$SRC" \
  --graph cpg_falkorchat --language pythonsrc \
  --workdir /tmp/cpg-work/falkorchat --load
```

Staging exactly `{falkorchat, tests}` also reproduces the M2 **`FILENAME` shape** (`falkorchat/…`,
`tests/…`), which every recipe's `STARTS WITH 'tests/'` filter depends on.

**⚠ Do NOT use `--reset`, and do not "simplify" this back to `--reset` later (N-1).** The reason is
not style, it is that the safety prompt would **not fire**. `claude/scripts/guard-destructive-ops.sh`
(lines 34–58) reads `.tool_input.command` and pattern-matches **the Bash command string** —
`grep -qiE "…|GRAPH\.DELETE([^[:alnum:]]|$)"`. `pipeline.sh --reset` runs its `redis-cli … GRAPH.DELETE`
**inside the script** (lines 66–72), so the string never appears in the command the hook sees: no
prompt, no human, the wipe proceeds unattended. A mistyped `--graph cpg_salesperson` would be
executed silently, with "the other four graphs untouched" left as a *post-hoc* done-condition
instead of a gate. Running `GRAPH.DELETE` as its own Bash command puts the graph name **in the text
the human approves**, which is the whole point of the guard. **Run that `GRAPH.DELETE` as its own
Bash call**, not pasted inside a larger block — the approval prompt should show one command, not a
script. Generalisation worth remembering, and filed as backlog **C-311** (S10, `cobb`/`devops`):
**the guard is blind to any destructive command wrapped inside a script**, so every future wrapper
needs either an explicit pre-step like this one or a hook that also matches the script name.

**One property this ordering gives up, and why that is acceptable.** `pipeline.sh` issues its reset
only when **both** `--reset` and `--load` are set **and** only after the export-non-empty assertion
has passed (lines 58–72) — i.e. a failed parse cannot delete the graph. Deleting first forfeits
that: if the subsequent parse fails, `cpg_falkorchat` is gone. Accepted, because **D1** rules that
the current graph's data does not matter (it is a package-only build with zero test methods, §2.3)
and the load artifacts are re-derivable. If you want both properties, run the pipeline **once
without `--load`** first (parse → export → `.cypher` artifact, touching FalkorDB not at all), then
`GRAPH.DELETE`, then load the already-built export without re-parsing:
`python3 skills/joern-cpg/scripts/cpg-to-falkordb.py /tmp/cpg-work/falkorchat/export -o
/tmp/cpg-work/falkorchat/load.cypher --graph cpg_falkorchat --load` — the same invocation
`pipeline.sh` makes in its transform step, so this is not a new code path. Note that this variant
skips `pipeline.sh`'s own post-load count verification: run B1 below by hand instead.

Approve the deletion **only** for `cpg_falkorchat`. The other four graphs (`cpg_salesperson`,
`ws:acme`, `ws:test`, `reference`) must be untouched — the `GRAPH.LIST` snapshot above is what you
compare against. `pipeline.sh` uses a single-socket loader ("no per-statement redis-cli"), so the
C-101 `MAX_ARG_STRLEN` scar should not recur; if it does, C-101 becomes a blocker for this step
only, and S9 still has a graph to run the equivalence proof against (AC-3's equivalence half is
graph-independent).

**Record the fresh baseline** — run each of these once the load verifies, and write the numbers
into the S9 test plan (they become the recorded baseline that supersedes the M2 figures):

| # | Query | What it anchors |
|---|---|---|
| B1 | `MATCH (n) RETURN count(n)` and the edge equivalent | build size |
| B2 | `MATCH (m:METHOD) WHERE m.FILENAME STARTS WITH 'tests/' RETURN count(m)` | **must be > 0** — this is the point of including tests |
| B3 | `MATCH (caller:METHOD)-[:CONTAINS]->(c:CALL {NAME:'post_message'}) RETURN DISTINCT caller.FULL_NAME, caller.FILENAME, caller.LINE_NUMBER ORDER BY caller.FILENAME, caller.LINE_NUMBER` | AC-1/AC-3 direct callers |
| B4 | the L1/L2/L3 closure query from `skills/cpg-analysis/references/test-gap.md`, plus its `count(DISTINCT g.NAME)` variant | AC-3 test-gap |

**Done when:** `pipeline.sh` exits 0 and its own post-load count verification passes; B2 > 0;
B1–B4 numbers are recorded and handed to `qa-engineer`; `GRAPH.LIST` still contains the other four
graphs; and the reload artifacts (`cpg.bin`, `load.cypher`) are copied to `cpg/.cpg-artifacts/`
(gitignored) so a re-load is cheap.

**Explicitly not a done-condition:** matching 79,581 / 522,182 / 336 / 21. Those describe source
that no longer exists (§2.3, D1). An implementer who "iterates until the numbers match" is chasing
a ghost.

### S9 — live acceptance · `qa-engineer`

Execute §7 and write `docs/archive/test-reports/cpg-query-access-report.md`, test plan first per the
agent's own convention (`docs/archive/test-plans/cpg-query-access.md`). The test plan carries S8's B1–B4
numbers as **the recorded baseline**.

**Done when:** AC-1…AC-4 each have a recorded pass/fail with evidence; the S5 allowlist edit is
proven live for both `analyst` and `architect` (m-4); the fresh baseline is written down; any
defect is filed.

### S10 — backlog & history · `coder`

`docs/BACKLOG.md` — new milestone **M3 — CPG query access (MCP)**:

| ID | Item | Owner |
|---|---|---|
| C-301 | MCP server (`cpg/mcp/`): venv, `server.py`, `run.sh`, tests, smoke command | devops/coder |
| C-302 | `.mcp.json` + `.claude/settings.json` wiring, live connect | devops |
| C-303 | `cpg-analysis` skill surface + catalog row | cobb |
| C-304 | Agent `tools:` wiring, `claude/README.md` rows 9/16/17, root `AGENTS.md`, kaizen histories, tico inbox close-out | cobb |
| C-305 | Requirements: `joern-cpg-pipeline.md` FR-9 reversal + `cpg-query-access.md` AC-1/AC-3 corrections | coder |
| C-306 | CPG rebuild + fresh baseline + live acceptance | joern / qa-engineer |
| C-307 | `skills/agent-standards/` Claude Code §MCP + OpenCode MCP subsection | cobb |

Plus three **follow-ups (not delivered by M3)**:

| ID | Item | Owner |
|---|---|---|
| C-308 | Bounded transitive **upward** name-closure query for `impact-analysis.md` (the L1/L2/L3 shape `test-gap.md` uses downward, with the `WITH`-splitting idiom and the name-collision caveat), live-verified. A naive composition returned **0 rows** on the live graph — real work, not a copy-paste. Deferred by **D3**. | `graph-dba` |
| C-309 | Genericize the pre-existing home-path/username leaks that make `audit-team.sh` return `RESULT: FAIL` (`.claude/settings.json`, `claude/devops/kaizen/inbox.md`, `claude/joern/kaizen/inbox.md`, `docs/plans/m2-cpg-analysis-skill.md`), then restore "audit passes" as a usable gate. | `cobb` / `devops` |
| C-310 | OpenCode + Kiro MCP wiring for the `cpg` server (R-7), including the `allowed-tools` portability result from S4. | `cobb` / `devops` |
| C-311 | **`guard-destructive-ops.sh` is blind to destructive commands wrapped in scripts** (found in the re-gate as N-1: `pipeline.sh --reset` deletes a graph with no prompt, because the guard matches the Bash command string). Decide the remedy — match known wrapper invocations (`pipeline.sh .* --reset`), or make wrappers require an out-of-band delete as S8 now does — and apply it to any other wrapper in `skills/*/scripts/`. | `cobb` / `devops` |

`docs/HISTORY.md` — dated entry above the M2 one: what shipped, the tool name, the ACs verified
**with the fresh baseline numbers**, the build-vs-buy decision and its reversal trigger in one
line, and the known limits (Claude-Code-only wiring; read-only; `EXPLAIN`-only, no `PROFILE`;
truncation defaults). Record explicitly that **the M2 CPG figures are superseded and why** — they
are a property of a specific build of a moving source tree, not of the access mechanism. That
sentence is what stops the next reader from re-deriving §2.3 from scratch.

**Done when:** BACKLOG and HISTORY agree with what is on disk; every doc row in §6 is ticked; no
new `audit-team.sh` failures.

---

## 6. Documentation impact (each row owned by a step)

| Doc | Change | Step / owner |
|---|---|---|
| `docs/requirements/joern-cpg-pipeline.md` | FR-9 reversed + decision-log entry + status (**AC-4**) | S6 · `coder` |
| `docs/requirements/cpg-query-access.md` | AC-3 restated (equivalence + fresh baseline, D1/D2); AC-1 → direct callers (D3); status on delivery | S6 · `coder` |
| `skills/cpg-analysis/SKILL.md` | frontmatter `description` + `allowed-tools`; §1 rewritten (tool, discovery, `EXPLAIN`-only, truncation, fallback block); §3 parameter note | S4 · `cobb` |
| `skills/cpg-analysis/references/impact-analysis.md` | preamble `redis-cli` invocation → tool parameters; "iterate Q1" note **kept** (C-308) | S4 · `cobb` |
| `skills/cpg-analysis/references/{rca,code-review,test-gap}.md` | **no change** (verified: no connection framing) | — |
| `skills/joern-cpg/references/cpg-model.md` | **no change** (its `GRAPH.QUERY` mention is producer-side); remains the single schema source | — |
| `skills/README.md` | `cpg-analysis` catalog row: MCP tool as the path, `redis-cli` fallback, Claude-Code-only wiring, OpenCode spot-check result | S4 · `cobb` |
| `skills/agent-standards/claude-code.md` | §MCP fleshed out from §2.4 (**M-5**) | S7 · `cobb` |
| `skills/agent-standards/` (OpenCode surface) | new short **OpenCode MCP** subsection (**M-5**) | S7 · `cobb` |
| root `AGENTS.md` | new `cpg/` Structure entry; `cpg-analysis` bullet; first-MCP-wiring note; smoke command in key commands (**m-6**) | S5 · `cobb` |
| `claude/README.md` | rows **9 / 16 / 17** mention the tool; row 12 unchanged | S5 · `cobb` |
| `claude/analyst/analyst.md`, `claude/architect/architect.md` | `tools:` allowlist gains `mcp__cpg__query` | S5 · `cobb` |
| `claude/{analyst,architect,qa-engineer}/kaizen/history.md` | dated entry each (**M-4**, `claude/AGENTS.md:35`) | S5 · `cobb` |
| `claude/tico/kaizen/inbox.md:19` | FR-9-contradiction note closed out / marked resolved by C-305 (**M-4**) | S5 · `cobb` |
| `claude/cobb/kaizen/history.md` | entry for the `agent-standards` change | S7 · `cobb` |
| `cpg/mcp/README.md` | **new** — run/debug/restart, env vars, semantics (read-only, `EXPLAIN`-only, display-only truncation), local-scope fallback with `<repo-root>` placeholder, smoke command, other-harness wiring | S1 skeleton / S3 · `devops` |
| `.mcp.json`, `.claude/settings.json` | **new / edited** — no absolute paths | S3 · `devops` |
| `docs/BACKLOG.md` | M3 + C-301…C-307, follow-ups C-308/C-309/C-310/**C-311** | S10 · `coder` |
| `docs/HISTORY.md` | dated delivery entry incl. the superseded-M2-figures note | S10 · `coder` |
| `docs/archive/test-plans/cpg-query-access.md`, `docs/archive/test-reports/cpg-query-access-report.md` | **new** — plan (carries the fresh baseline) + report | S9 · `qa-engineer` |
| `docs/plans/cpg-query-access-coordination.md` | unit table / log kept current | `teco`, continuous (plans dir — within teco's write scope) |

---

## 7. Test strategy

### 7.1 Unit (S2, no FalkorDB)

Covered in S2's list: tool-contract introspection (**including `outputSchema is None`**), directive
splitting **and the `PROFILE` refusal happening before any server call**, cell rendering, table
formatting with **both** truncation paths, the EXPLAIN graph-existence pre-check, and error
classification. These are the parts most likely to rot and the only parts worth mocking. Everything
else is integration — do not mock FalkorDB.

### 7.2 Live acceptance (S9) — against the freshly built `cpg_falkorchat`

**Precondition check, before anything else** (M-2): in the same workspace that will run the tests,
`claude mcp list` must show `cpg` **connected**. Project-scoped servers stay at `⏸ Pending
approval` until the workspace is trusted interactively, and a headless `claude -p` run in an
un-approved workspace silently has no `cpg` server — the ACs would then fail for a reason that has
nothing to do with this feature.

**AC-1 — cold session, one graph-query tool call.** The question is the **direct**-caller
formulation (D3):

```
claude -p "In the loaded CPG graph cpg_falkorchat, who calls post_message?" \
       --output-format stream-json
```

from the repo root, in a session with no prior context. Assert on the transcript:
- **exactly one** tool call with `name == "mcp__cpg__query"`, and its `input` keys are exactly
  `{graph, cypher}`;
- **zero** `Bash` events;
- no shell quoting or escaping anywhere in the transcript.

Do **not** assert "exactly one `tool_use` event" (M-2): a cold session legitimately emits a `Skill`
invocation and `Read`s of `SKILL.md` and the recipe — `cpg-analysis` is explicitly designed to
require that. Those are expected and must not fail the criterion.

Repeat once via the `analyst` subagent and once via `architect` — the two with `tools:` allowlists.
This is the live proof that S5 worked (R-3, m-4).

**AC-2 — multi-line verbatim.** Send the direct-callers query as a single line and again with
newlines and indentation between clauses; assert byte-identical result bodies. Also send a query
containing `'`, `"` and `$` inside string literals — none of which needs escaping through the tool.

**AC-3 — equivalence, plus the fresh baseline (D1).** AC-3 has two halves, and the first is the one
that actually tests *this* feature:

1. **Equivalence (graph-independent, the real proof).** For each of B3 (direct callers), B4 (the
   test-gap L1/L2/L3 closure) and B4's `count(DISTINCT g.NAME)` variant, run the query through
   `mcp__cpg__query` **and** through `redis-cli -p 6379 GRAPH.QUERY cpg_falkorchat '<cypher>'
   --no-raw`, then diff the **value sets** (order-insensitive where the query has no `ORDER BY`;
   byte-identical where it does). They must be identical. Any difference is a defect in the tool's
   rendering, which is exactly what this criterion exists to catch.
2. **Fresh baseline.** Record B1–B4's actual numbers from S8 in the test plan and report, and state
   plainly that they **supersede** the M2 figures (79,581 / 522,182 / 336 / 21 / 39 / 32), which
   describe source that has since moved 8 commits (§2.3).

Anchor spot-checks from the recipe still apply as sanity signals rather than fixed numbers: the
test-gap result should flag production methods with no test path and **not** flag ones that have
one. Because the source changed, the specific names from M2 (`Services.ping`, `_safe_respond`,
`_safe_run_workflow` flagged; `_serialize_opaque` not) are **indicative, not required** — if they
differ, confirm against the source rather than treating it as a failure.

Note the row cap is 200; if B4 exceeds it, the truncation notice must appear and the equivalence
check must be run on a narrowed (`LIMIT`-free but projected or ordered) form so both paths return
the full set.

**AC-4 — no contradiction.** `grep -n "redis-cli\|MCP" docs/requirements/joern-cpg-pipeline.md
docs/requirements/cpg-query-access.md skills/cpg-analysis/SKILL.md` and read every hit: one
mechanism, one fallback, no disagreement. Also confirm `claude/tico/kaizen/inbox.md:19` no longer
reads as an open contradiction (S5).

### 7.3 Robustness pass (S9, beyond the ACs)

| Check | Expectation |
|---|---|
| Nonexistent graph name (plain query) | Error names the graph, lists loaded graphs, routes to `joern`; **`GRAPH.LIST` is unchanged afterwards** (the `RO_QUERY` empty-key guarantee, §4.4 D4a) |
| Nonexistent graph name with `EXPLAIN ` prefix | Same error, and **`GRAPH.LIST` is still unchanged** — this is the pre-check doing its job (B-1) |
| `PROFILE MATCH …` | The refusal message, pointing at `redis-cli GRAPH.PROFILE`; **no results returned**, no server call made |
| `/* c */ PROFILE MATCH …` and `// c⏎PROFILE MATCH …` (n-3) | The **same** refusal — not results. Live-verified that raw `GRAPH.RO_QUERY` accepts both and returns rows, so this is the sniff (D5a) doing its job |
| Syntax error | FalkorDB's line/column message, verbatim, plus the schema pointer |
| `CREATE (:X)` | Read-only rejection message |
| `MATCH (n) RETURN n` (tens of thousands of nodes) | Truncated with the exact notice (row cap **or** char cap, whichever binds), the "unordered" clause present, session survives, latency sane |
| A query that **binds the char cap** (e.g. `MATCH (m:METHOD) RETURN m.CODE`) (N-2) | The result arrives **in the conversation with its notice intact as both the first and the last line** — *not* replaced by a file reference, and ideally with no 10 k-token warning. If a file reference appears, `_meta`/the 30 000 default is not doing its job — file it as a defect, do not raise the cap |
| `EXPLAIN MATCH …` on a real graph | Plan text, not results |
| FalkorDB stopped (`docker stop falkordb-dev`), one call, then started again | Actionable "unreachable" message; **the next call after restart succeeds without restarting the session** (connection-pool recovery) |
| Server killed mid-session | Documented recovery path works (`/mcp` reconnect or new session) — a known limitation, not a defect |
| Deep traversal `*1..12` | Times out at 30 s with the bounded-traversal hint, not at the 60 s harness wall |

---

## 8. Risks & open questions

**R-1 — requirements corrections (ruled, now owned).** AC-3's "30 untested methods" is stale (39
rows / 32 distinct names) and AC-1's "transitively" overreaches for this feature. Both are ruled by
D2/D3 and assigned to **S6 · `coder`**. The architect does not edit requirements, and `teco` cannot
(harness-restricted to `docs/plans/`).

**R-2 — the rebuilt CPG is a CPG of different source, by design.** D1 accepts this. The residual
risks are: the C-101 loader scar (believed fixed — single-socket loader); ordinary Joern
non-determinism; and **skipping the staging step**, which would pull `.venv`'s 1,808 files into the
graph and produce a large, slow, useless CPG. S8's done-condition (B2 > 0, other graphs untouched)
catches the first two; the staged `cp` prevents the third. AC-3's equivalence half does not depend
on any of it.

**R-3 — the two consumers with `tools:` allowlists.** `analyst` and `architect` will silently not
see `mcp__cpg__query` unless S5 edits their frontmatter. Mitigated by S5 and, decisively, by the
per-agent AC-1 checks in S9 (§7.2) — the *live* check is deliberately not in S5's done-condition,
because it needs S3 (m-4).

**R-4 — stdio server lifecycle.** One process per session; **not auto-reconnected** if it dies.
Mitigations: never let an exception escape the tool; nothing on stdout except protocol traffic;
`.mcp.json` `"timeout": 60000`; a query timeout below it; recovery documented in
`cpg/mcp/README.md`. Escape hatch if crashes are ever observed: HTTP transport with a supervised
process.

**R-5 — first-run friction (human step).** Project-scoped servers need an interactive approval /
workspace-trust acceptance; `enabledMcpjsonServers` only helps in a trusted folder. The user must
run `claude` interactively once and approve `cpg`. `claude mcp reset-project-choices` undoes a
mistaken rejection. Budget one human interaction in S3 — and note it also gates S9's headless runs.

**R-6 — could this be *worse* than `redis-cli`?** Honest accounting:
- Latency: `redis-cli` is ~4 ms/call; the server adds ~1.5 s **once** per session, then is faster
  per query (no process spawn). FR-5's process-overhead argument is real but small — **the actual
  win is FR-3 (no shell quoting) and the removal of connection rediscovery.** Do not oversell the
  performance framing in the docs.
- **Losing `PROFILE` is a real regression** versus `redis-cli`, accepted deliberately (D4): a
  read-only guarantee that a single directive can defeat is not a guarantee. `graph-dba` keeps
  `GRAPH.PROFILE` via `redis-cli`, and `claude/README.md` row 12 already names that as its tuning
  path. `EXPLAIN` — the planning tool an agent actually needs before a heavy traversal — survives.
- Truncation is a new failure mode `redis-cli` did not have (it just floods). The explicit
  truncation notice with the true row count and the "unordered" caveat is the mitigation; caps are
  env-tunable; truncation is display-only.
- The `EXPLAIN` prefix **diverges from raw FalkorDB behaviour** (§2.2). Documented in both the tool
  description and SKILL.md; a query copied to `redis-cli` with that prefix will execute.
- The tool can name **any** graph on the instance (`ws:acme`, `reference`, …). `GRAPH.RO_QUERY`
  bounds this to reads *and* prevents empty-key creation; no graph-name allowlist is imposed,
  because FR-4 requires caller-supplied names and a `cpg_` prefix rule would break on the first
  rename.

**R-7 — portability.** MCP wiring covers **Claude Code only**. OpenCode and Kiro read the same
skills via symlink but have no MCP configuration in this repo; for them the skill's `redis-cli`
fallback is the *only* path. Stated in `SKILL.md`, `skills/README.md`, `cpg/mcp/README.md` and root
`AGENTS.md` (S4/S5); the durable facts land in `agent-standards` (S7); actual wiring is **C-310**;
the `allowed-tools` cross-harness spot-check is S4's (m-3).

**R-8 — OQ2 (component structure) remains open.** This change creates `cpg/mcp/` as the
component's first code while its docs stay at repo-root `docs/`. A deliberate half-step: if the
component grows, the natural move is `cpg/docs/` + `cpg/mcp/`, and this plan does not pre-empt it.

**R-9 — the audit gate is already red.** `claude/scripts/audit-team.sh` returns `RESULT: FAIL`
today on pre-existing home-path/username leaks in files this feature does not touch (confirmed by
teco, 2026-07-25). Every step's done-condition is therefore **"no new failures"**, measured by the
before/after diff in §5. Do **not** add a hit: no absolute path goes into `.mcp.json`,
`cpg/mcp/README.md`, or any doc this plan produces. Cleaning up the existing leaks is **C-309**.

**R-10 — no regression signal.** The repo has no root-level test runner, so nothing will report
that this component broke except an agent's failed query. Mitigated by the one-command smoke check
(§3.3) referenced from `cpg/mcp/README.md` and root `AGENTS.md` key commands (m-6). This is a
mitigation, not a fix; a real fix is a repo-level CI entry point, which is out of scope here.

### Open questions

**None blocking.** D1–D4 closed the four that were open at v1. Two non-blocking notes for the
stakeholder, recorded so they are not lost:

1. **C-308** (transitive upward-closure query) is deferred, not discarded. Until it exists, "who
   depends on X, transitively" remains an iterative, multi-call analysis — the honest state today,
   and unrelated to the transport this feature changes.
2. **C-309** (the pre-existing `audit-team.sh` leaks) means the repo's own coherence gate cannot be
   used as a binary pass/fail by any agent until it is cleaned. Worth scheduling soon; every plan
   that touches tracked files pays the "diff the audit output" tax in the meantime.

---

## 9. Ready to implement

Contract frozen: **one tool `mcp__cpg__query(graph: str, cypher: str) -> str`**, registered with
`structured_output=False` and `readOnlyHint=True`; stdio Python server at `cpg/mcp/`;
project-scoped `.mcp.json` using `bash -c "$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh"` (no absolute
paths); read-only via `GRAPH.RO_QUERY` — which is also what prevents a typo'd graph name from
materialising a key; **`EXPLAIN` only, `PROFILE` refused**, with a **comment-blind** directive sniff
(§4.4 D5a); 200-row / 300-char / **30 000**-char display-only truncation whose honest,
order-caveated notice is emitted **first and last** and whose ceiling is pinned by
`_meta["anthropic/maxResultSizeChars"]`; curated errors that route "no graph" to the `joern` agent
and double as the only graph-discovery affordance.

Docs (S4–S7) can start immediately and in parallel with the code (S1–S3). S8 should start early.
S9 joins all of them and gates delivery. S10 closes the books. No step is owned by `teco`.

---

## 10. Rework log

### v1 → v2

Against [`../reviews/cpg-query-access.md`](../reviews/cpg-query-access.md) (analyst, 2026-07-24)
and teco's coordination corrections (2026-07-25). Every finding is resolved; nothing is silently
dropped.

| ID | Sev | Resolution |
|---|---|---|
| **B-1** | Blocker | **Fixed via D4.** `PROFILE` removed from the tool and *actively refused* before any server call (a passive drop would have let `PROFILE …` fall through to `ro_query`, where FalkorDB ignores the prefix and returns results — a wrong answer). `readOnlyHint=True` is now honest. Added the two statements the review found missing: `GRAPH.RO_QUERY` is *why* the main path is safe **and** why a typo'd graph name cannot materialise a key (§4.4 D4a), plus a `GRAPH.LIST` pre-check so the `EXPLAIN` path cannot materialise one either. Robustness rows added for both (§7.3). |
| **B-2** | Blocker | **Fixed.** The `bash -c "$CLAUDE_PROJECT_DIR/…"` form is now **primary** (§4.3); the absolute-path fallback is replaced by a **local-scope** `claude mcp add --scope local` registration (untracked), documented with a `<repo-root>` placeholder. All absolute paths scrubbed from this document. Done-conditions restated repo-wide as **"no new `audit-team.sh` failures"** with a concrete before/after diff recipe (§5), since the audit is already red on pre-existing leaks (teco-confirmed). Cleanup filed as **C-309**. |
| **B-3** | Blocker | **Superseded by D1.** The rebuild is approved and destructive-deletion authorised, but the M2 numbers are formally **unreachable** (8 commits, verified 2026-07-25) and are no longer a done-condition — S8 says so explicitly. S8 now stages a clean source copy (new finding: `falkor-chat/server/.venv` holds 1,808 `.py` files vs 41 first-party, and `build-cpg.sh` has no exclusion mechanism), records B1–B4 as **the new baseline**, and AC-3 becomes **equivalence + fresh baseline** (§7.2). The pinned-worktree option A′ is dropped (one-line mention only, below). |
| **M-1** | Major | **Superseded by D3.** AC-1 is demonstrated with the **direct**-caller question; the transitive upward-closure query is a first-class deferred backlog item **C-308** owned by `graph-dba`, with the "returned 0 rows on a naive composition" warning carried into the item. The AC-1 wording change is a requirements edit owned by **S6 · `coder`**, not left to an implementer. |
| **M-2** | Major | **Fixed.** §7.2 now asserts *exactly one **graph-query** tool call* (`mcp__cpg__query`, keys exactly `{graph, cypher}`), **zero** `Bash` events, no shell quoting — and explicitly permits the `Skill`/`Read` events a cold `cpg-analysis` session must emit. Added the `claude mcp list` **connected** precondition before any headless run. |
| **M-3** | Major | **Fixed, verbatim from the analyst's empirical result.** `@mcp.tool(..., structured_output=False)` is now in the frozen contract with the measured evidence (`outputSchema {result: string}` → double payload on `mcp 1.28.1`), plus an S2 unit assertion that `outputSchema is None` as the regression guard. |
| **M-4** | Major | **Fixed.** S5 now owns dated `kaizen/history.md` entries for `analyst`, `architect` **and** `qa-engineer` (catalog row change also counts, per `claude/AGENTS.md:35`), and the close-out of `claude/tico/kaizen/inbox.md:19`. S7 owns cobb's own kaizen entry. All four appear as §6 rows. `claude/AGENTS.md`'s roster is capability-level and needs no line — decided, not skipped. |
| **M-5** | Major | **Fixed.** New **S7 · `cobb`**: fold §2.4 into `skills/agent-standards/claude-code.md` §MCP (currently three lines of prose) and add the missing **OpenCode MCP** subsection, which is also R-7's remediation. Both are §6 rows. |
| **M-6** | Major | **Fixed / mostly moot under D4.** §4.4 now states outright that `explain()` takes no `timeout`, that with `PROFILE` gone this is nearly moot (planning does not execute), and that the `.mcp.json` 60 s wall is the backstop. §7.3's timeout row is scoped to the plain path. |
| **m-1** | Minor | **Fixed.** Char-cap behaviour fully specified (drop whole rows from the tail, same notice naming the char cap and both counts); the "results are unordered unless the query has ORDER BY" clause added to the notice text; truncation documented as **display-only**. |
| **m-2** | Minor | **Fixed.** S4 must add an explicit graph-discovery sentence to SKILL.md §1 (names come from the caller; discover via the `redis-cli GRAPH.LIST` fallback or the tool's not-found error) and it is in S4's done-condition. |
| **m-3** | Minor | **Fixed.** S4 carries an OpenCode load spot-check for the unknown `allowed-tools` entry, with an explicit "record as unverified + fold into C-310" branch if OpenCode cannot be exercised here — no silent gap either way. |
| **m-4** | Minor | **Fixed.** Dependencies corrected to `S9 ← S3, S4, S5, S6, S8`; S5's live spot-check moved out of its done-condition into S9, where S3 exists. |
| **m-5** | Minor | **Fixed (rejected with reasons).** §4.3 now weighs per-subagent `mcpServers:` frontmatter and rejects it on four grounds, the decisive one being `agent-standards/claude-code.md:110-111` — the field is **ignored for teammates** — so it would not reach the consumers at all. |
| **m-6** | Minor | **Fixed.** §3.3 states the honest artifact set (not "~150 lines") and adds a one-command smoke check, referenced from `cpg/mcp/README.md` (S3) and root `AGENTS.md` key commands (S5). Tracked as R-10. |
| **n-1** | Nit | **Fixed.** `requirements.txt` + `requirements-dev.txt`, with a one-line rationale for not using `pyproject.toml` extras (`cpg/mcp` is a script run by path, not an installable package). |
| **n-2** | Nit | **Fixed.** S6 specifies the sibling link form `./cpg-query-access.md`. |
| **A′** | Reviewer alternative | **Dropped per D1.** A CPG built from a worktree pinned at `4f69a16` into `cpg_falkorchat_m2` would reproduce the M2 figures without any `GRAPH.DELETE`, and FR-4 makes the separate graph key free — but D1 authorises the destructive rebuild and rules that the fresh numbers *are* the baseline, so the extra build buys a number nobody will use again. Recorded here in case a future reader wants the M2 graph back. |
| **T-1** | teco (hard) | **Fixed.** No step is owned by `teco` (its `Write`/`Edit` is restricted to `docs/plans/`). S6 and S10 → `coder`; S5 and S7 → `cobb`; the ownership rule is stated at the top of §5 and every §6 row names its owner. Only the coordination doc (inside `docs/plans/`) remains teco's. |
| **T-2** | teco | **Fixed.** See B-2 — "no *new* audit failures", `$CLAUDE_PROJECT_DIR` form, absolute paths scrubbed from this document. |
| **T-3** | teco | **Fixed.** See M-3. |
| **T-4** | teco | **Fixed.** kaizen histories + tico inbox → S5; `agent-standards` §MCP + OpenCode gap → S7; R-7's remaining wiring work → **C-310**. Nothing is scoped out without an owning backlog item. |
| **T-5** | teco | **Fixed.** §2.5 and §4.5 now state that `claude/README.md` rows **9, 16, 17** each carry a `cpg-analysis` clause and all three need updating; row 12 (`graph-dba`) does not mention the skill and needs no change — its `GRAPH.PROFILE` clause is, under D4, now the *only* profiling path. |

### v2 → v2.1 — re-gate follow-up (analyst, 2026-07-25: *approve with suggestions*, 0 blockers)

| ID | Sev | Resolution |
|---|---|---|
| **N-1** | Major | **2026-07-25 — Fixed in S8.** `--reset` dropped: the `guard-destructive-ops.sh` prompt S8 leaned on **cannot fire**, because the guard matches the Bash *command string* and `pipeline.sh` runs `GRAPH.DELETE` internally (re-verified in the script: guard lines 34–58, `pipeline.sh` lines 66–72). S8 now runs a `GRAPH.LIST` snapshot, an explicit `redis-cli GRAPH.DELETE cpg_falkorchat` (which does trip the guard, with the graph name in the text the human approves), then `pipeline.sh … --load` **without** `--reset`; the rationale is written into the step with a ⚠ "do not simplify this back" warning. Also recorded: the property this forfeits (`pipeline.sh` deletes only *after* the export-non-empty assertion), why D1 makes that acceptable, and the parse-first variant that keeps both. The generalisation — the guard is blind to any destructive command wrapped in a script — is filed as new backlog item **C-311** in S10. |
| **N-2** | Major | **2026-07-25 — Fixed in §4.4 (Truncation) + the frozen decorator + S2 tests + §7.3; all three remedies adopted, each with a distinct job.** Claude Code warns above 10 k tokens, defaults to a 25 k-token limit, and **persists over-threshold results to disk, replacing them with a file reference** — so v2's 60 000-char payload (~17–24 k tokens) risked losing its truncation notice, which was the payload's *last* line, on exactly the runs that were truncated. (1) The notice is now emitted **first and last**, byte-identical — the only harness-independent remedy, and the one that survives a future limit change or a different harness. (2) `CPG_MCP_MAX_CHARS` default **60 000 → 30 000** — removes the collision at the source (≤ 15 k tokens even at 2 chars/token) and is nearly free, since the 200-row cap binds first for the recipes' typical projections. (3) `_meta["anthropic/maxResultSizeChars"] = min(2 × CPG_MCP_MAX_CHARS, 500 000)` — pins the persist-to-disk threshold in **chars**, the same unit as our cap, independent of `MAX_MCP_OUTPUT_TOKENS`; verified live on the pinned `mcp 1.28.1` that `@mcp.tool(meta=…)` serialises as `_meta` in `tools/list` with `outputSchema` still `None`. *Rejected:* remedy 1 alone (keeps shipping a file-reference-sized payload, defeating the token economy) and remedy 3 alone (it *raises* a ceiling we do not want to reach, and is the most version-sensitive of the three). New §7.3 row asserts a char-cap-binding result arrives with its notice intact and is **not** replaced by a file reference. |
| **n-3** | Minor | **2026-07-25 — Fixed as §4.4 D5a.** The directive sniff is now specified as a comment-blind prefix scan — loop over leading whitespace, `//`-to-EOL and `/* … */` blocks (unterminated `/*` ⇒ classify plain and let FalkorDB error), then `^(EXPLAIN\|PROFILE)\b` case-insensitively — with the two rules about what is *sent*: the plain path transmits the caller's string **byte-for-byte** (classification only, FR-3), the `EXPLAIN` path sends the text after the keyword. Six named cases added to S2's test 2/3 (both comment forms, the `\b` boundary, the unterminated block, the string-literal decoy) and one §7.3 live row, since the analyst verified that raw `GRAPH.RO_QUERY` executes `/* hi */ PROFILE …` and returns rows. |

### v2.1 → v2.2 — audit-trail correction (`architect`, 2026-07-25)

No design change. The v2.1 log above closed **N-1, N-2, n-3** and left the re-gate's remaining
minor findings unrecorded; `coder` nevertheless implemented **n-4** in S2, so the plan's own audit
trail disagreed with the shipped code (flagged by `teco` and `devops`). This entry closes that gap
for n-4 only.

| ID | Sev | Resolution |
|---|---|---|
| **n-4** | Minor | **2026-07-25 — Accepted and implemented (was: unrecorded).** The `instructions=` half of the finding shipped in S2 and is **live**: `FastMCP(name="cpg", instructions=SERVER_INSTRUCTIONS)` at `cpg/mcp/server.py:399`, the 408-char string at `server.py:103`. Verified by `architect` 2026-07-25 by driving an `initialize` handshake over `cpg/mcp/run.sh` — the response carries the string verbatim (408 chars, well under the 2 KB truncation) — and independently by observing Claude Code inject it in-session as the **"MCP Server Instructions"** block; `teco` and `devops` verified it separately. Judged **adequate**: it names the tool, the graph-key convention (`cpg_<component>`), the four question classes tool search must match on (call-graph / data-flow / impact-analysis / test-gap), the discriminator that matters in a cold session (*"without reading files"*), and the FR-4 caller-supplies-the-graph rule plus the not-found discovery affordance — and it does **not** duplicate the tool description's mechanics (EXPLAIN/PROFILE/no-APOC), which is the right split, since the description is only loaded once the tool is. Frozen into §4.4 and S2 retroactively. **The `"alwaysLoad": true` half is deliberately not adopted:** it loads the tool set upfront at the cost of **blocking session startup until the server connects** (cobb, docs-verified 2026-07-25), which is a bad trade for a one-tool server whose discovery path is now demonstrably working. Reversal trigger: if S9's AC-1 shows a cold session failing to find the tool, add `"alwaysLoad": true` to `.mcp.json` — a one-key change. **Two loose ends, owned elsewhere, not fixed here:** (a) the finding's third knock-on — `ToolSearch` must join §7.2 AC-1's permitted-extras list (`Skill`, `Read`) or a *good* run fails the criterion — is already carried to `qa-engineer` in the coordination doc and stays S9's to land; (b) *recommended*, not required: no test pins the string, so `cpg/mcp/tests/test_server.py` should gain a one-line guard that `mcp.instructions` is non-empty and ≤ 2000 chars, next to the existing tool-contract assertions — cheap insurance for the one string a cold session depends on and the only part of the tool contract currently unguarded. |

**Still unrecorded by design, not by oversight:** re-gate findings **n-5** (headless permission for
`mcp__cpg__query`), **n-6** (AC-3 compares two renderings) and **n-7** (S6 decision-log entry) plus
nits **nn-1/nn-2** are all owned by S3/S6/S9, not by this document's design. n-7 is already
delivered (U5, `teco`-verified). n-5 and n-6 remain the review's suggested fixes for S3 and S9
respectively and were **not** re-checked in this pass. They are listed here only so a reader can
tell "handled elsewhere" from "dropped".

**Unchanged from v1 and endorsed by the review** (kept deliberately, not by inertia): the
build-vs-buy decision and its reversal trigger (§3); §2.4's Claude Code mechanics — now also
promoted to `agent-standards` rather than left to be archived with this plan; the §2.2 `EXPLAIN`
finding and the instinct to make the divergence loud in two places; the row-by-row §6 doc table
including its "no change, and here is why" rows; and R-6's honest accounting, now extended with the
`PROFILE` regression.
