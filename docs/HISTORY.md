# Change History — CPG code-graph component

> Dated log of actual changes to the repo-root **CPG / code-graph** component (Joern → FalkorDB).
> Most recent first. Forward-looking work lives in [`BACKLOG.md`](./BACKLOG.md); requirements in
> [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md) and, for the read
> path, [`requirements/cpg-query-access.md`](./requirements/cpg-query-access.md).

## 2026-07-26 — `--verify-inputs` no longer answers "OK" for a line-continued `COPY` (C-320 review follow-up) ✅

The `analyst` review of the delivered C-320 change (Part III of
`docs/reviews/cpg-mcp-containerization.md`, *approve with suggestions*, no blocker) found one must-fix.
This is that fix plus the doc corrections; no design decision moved.

- **The defect (M-7).** `build.sh --verify-inputs` parsed the Dockerfile **line by line**, so a
  `\`-continued `COPY` was invisible to it and the check answered *"--verify-inputs OK"*. Reproduced
  against the shipped script: appending `COPY requirements.txt \` / `     setup.sh /app/` to the
  Dockerfile passed with exit 0, while the single-line control `COPY setup.sh setup.sh` correctly
  failed. That is the **one direction that costs correctness**: the file lands in the image, the
  content hash does not move, `docker image inspect` hits, and the launch path serves an image without
  the change — the exact failure the hash exists to make unrepresentable. `--verify-inputs` is the only
  mechanism enforcing that invariant, which `Dockerfile`, `image-tag.sh` and `README.md` all state as
  absolute. Nothing was wrong in the committed tree (no continued `COPY` in it); it was a trap for the
  next editor.
- **The fix** — the `awk` parse now joins continuations (and drops comment lines, which Docker also
  permits *inside* a continuation) before any rule looks at a `COPY`. Same pass: `COPY --from=<stage>`
  is skipped, because its sources come from another build stage and were being misreported as missing
  build-context files — a wrong diagnostic on the natural next edit to a multi-stage Dockerfile.
- **The regression is checked in, because a silent false pass is undetectable without one.**
  `cpg/mcp/tests/test_build_inputs.py` (9 cases) runs `--verify-inputs` against a throwaway copy of
  `cpg/mcp/` in pytest's `tmp_path`: the unmodified tree passes and writes nothing to stdout; the two
  continued-`COPY` forms and the single-line control all fail with the offending operand named; a
  *covered* continued `COPY` still passes; `COPY --from=` is accepted; and the directory rule (M-4)
  keeps its cover. Proven to catch the bug: run against the pre-fix `build.sh`, exactly the two M-7
  cases and the `--from=` case fail. It needs no Docker and never touches the tracked tree.
  It lives in `tests/` — the component's only automatically-run signal — but is the one **host-only**
  module: `.dockerignore` deliberately keeps the build tooling out of the build context, so
  `conftest.py` does not collect it when `build.sh` is absent. Not collecting (rather than skipping)
  keeps the in-image gate's counts **exactly** what they were, so a real regression there still shows
  as a diff. Host: **62 passed, 7 deselected** / **7 passed, 62 deselected**. In-image, unchanged:
  **53 passed, 7 deselected** / **7 passed, 53 deselected**.
- **Doc corrections** — `docs/BACKLOG.md`'s C-320 entry claimed "no registry contact" unqualified
  (true on a hit; a miss builds and does pull); `docs/test-plans/cpg-query-access.md`'s environment
  table still recorded the wiring as `run.sh`; and the plan's §12 M-4 row claimed the hash walk applies
  `.dockerignore`'s exclusions when it applies two of the three (`.pytest_cache` is not excluded —
  safe direction, filed on C-321).
- **Deferred by stakeholder decision, recorded on C-321 (M-8).** The autobuild calls
  `build.sh --runtime-only` without `CPG_MCP_NO_PULL`, so a hash miss puts an unbounded Docker Hub
  pull inside the 30 s MCP startup budget — and because the hash covers `tests/`, `pytest.ini` and
  `requirements-dev.txt` while the *runtime* stage COPYs none of them, a test-only edit forces a
  rebuild of a byte-identical runtime image. C-321 already edits `tests/test_server.py`, so it will
  trigger exactly this; the finding, the reason it belongs there and the cheapest fix
  (`CPG_MCP_NO_PULL=1` on the autobuild call) are now on that entry, together with the review's other
  one-line `image-tag.sh` minors. **Not implemented here.**
- **Verified** — the images were rebuilt at the new hash (`ba910c48571d` → `3f825c8afe4f`; the tag
  moves only because of the two `tests/` changes — `build.sh` is not a hash input), so the wired path
  does not pay for a build at the next session start. Full protocol handshake through
  `cpg/mcp/docker-run.sh`: ids 1–3 answered, `MATCH (m:METHOD) RETURN count(m)` on `cpg_falkorchat` →
  **1968**, stdout pure JSON. `docker ps -a --filter label=cpg-mcp=1` shows one `Up` container (this
  session's own server, on the pre-change tag) and **no `Exited`/`Created`** entry, i.e. no orphan.
  `GRAPH.LIST` still the same five graphs, no `_cpg_mcp_selftest_*` residue; `falkordb-dev` and
  `falkordb-data` untouched. `claude/scripts/audit-team.sh`: **no new failures** — the same two
  pre-existing C-309a leaks, in none of the files this change touches (the new file is untracked, so
  it was grepped directly for all five personal identifiers: clean).

## 2026-07-26 — The `cpg` MCP server is containerized (C-320) ✅

A clone now needs **Docker**, not a correctly built local Python 3.12 venv, to answer CPG queries.
The tool contract did not change — one tool, two parameters, read-only, same output format — and
`server.py` was not touched. `.mcp.json` changed by exactly two lines.

- **What shipped** — `cpg/mcp/Dockerfile` (multi-stage: `runtime` carries `server.py` and runtime
  deps only, `test` adds pytest and the suite; non-root `appuser`; `python:3.12-slim` following
  `falkor-chat/Dockerfile`; **no `EXPOSE` and no `HEALTHCHECK`** because this is a one-shot stdio
  process, not a service, and for the same reason **no Compose service** — `falkor-chat/compose.yaml`
  already defines a `falkordb` service that would bind a *second* engine on `:6379` over the same
  volume). Plus `.dockerignore`, `image-tag.sh` (sourced), `build.sh`, `docker-run.sh`.
- **The launch gate is a content hash, and that is the load-bearing decision.** `cpg-mcp:<hash12>` is
  a SHA-256 over every build input; `docker-run.sh` does one `docker image inspect` (~0.05 s, purely
  local) and builds **only on a miss**. The first design had the wrapper run a cached `docker build`
  on every launch; measurement killed it — **a warm, fully-cached BuildKit build still makes a Docker
  Hub `load metadata` round trip every single time** (0.5 s, essentially the whole build cost) unless
  the base image is in the local **image store**, which a BuildKit build does *not* populate. That
  would have made every session start depend on Hub reachability, a straight regression against the
  venv path, which needs no network at all. Verified end-to-end: in a network namespace with no
  connectivity and no DNS, the full handshake still returned real rows. Because the tag *is* the
  bytes, "missing" and "stale" become the same question, and two concurrent sessions can never
  clobber each other's image.
- **Networking** — default bridge + `--add-host=host.docker.internal:host-gateway`, riding the host
  port `falkordb-dev` already publishes. **The shared FalkorDB container and the `falkordb-data`
  volume were not touched, restarted or reconfigured** (`StartedAt` and `RestartCount 0` unchanged
  throughout). `--network host` was rejected as maximal privilege for one outbound connection, and
  behaves differently under Docker Desktop; a shared user-defined network was rejected because it
  needs either re-creating the shared container (`falkor-chat` + `salesperson` depend on it) or a
  manual, non-persistent `docker network connect`.
- **Lifecycle, measured** — `--init` is *required*, not defensive: PID-1 `python` **ignores
  `SIGTERM`** (still running a minute later), so without tini the harness's shutdown sequence cannot
  stop it. `--label cpg-mcp=1` makes any leak findable, `--rm` reaps, and **no `--name`** because a
  fixed name would collide across the concurrent sessions this repo encourages. `--read-only
  --tmpfs /tmp` was adopted only after probing every tool-body path under it.
- **Two implementation-time finds, both fixed here.** Docker's bare `-e VAR` form does **not** fall
  through to the image's `ENV` when the variable is unset in the caller's environment — it **deletes**
  it in the container, which silently left `server.py` on its `127.0.0.1` default, i.e. the container
  talking to itself. Env vars are now forwarded only when actually set. And `CPG_MCP_IMAGE`, which is
  documented to *bypass* the hash gate, still fell into the autobuild branch on a miss and then failed
  with docker's bare `No such image`; it now short-circuits with a curated message.
- **The host venv path is retained** (`setup.sh`, `run.sh`, `.venv`) and re-documented as (a) the fast
  regression loop and (b) the fallback. Both regression commands are unchanged and still green:
  `cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q` → **53 passed, 7 deselected**; `-q -m live` → **7
  passed, 53 deselected**. The same suite **inside the image** gives byte-identical counts, which is
  the control against the two paths drifting. Rollback is those two `.mcp.json` lines plus a restart.
- **Measured** — connect through the wrapper, spawn → `initialize` + `tools/list`: **median 1.47 s**
  over 7 runs (1.40–1.58), i.e. **4.9 % of the 30 s startup budget**. That budget was *verified*, not
  assumed, closing an ambiguity between the official env-var table and its prose: `MCP_TIMEOUT=1
  claude mcp list` → *"connection timed out after 1ms"*, while `MCP_CONNECT_TIMEOUT_MS=1` still
  connected. **`MCP_TIMEOUT` is the startup knob**; `.mcp.json`'s `"timeout": 60000` is the
  per-tool-call wall.
- **Design & review** — `docs/plans/cpg-mcp-containerization.md` (v3) and
  `docs/reviews/cpg-mcp-containerization.md` (two `analyst` passes: *needs changes* on v1, then
  *approve with suggestions* on v2). Backlog: **C-320** ✅, new **C-321** (the live suite's
  `os.getpid()`-derived scratch-graph name collapses to the constant `_cpg_mcp_selftest_1` inside a
  container — test code, so out of scope here and worked around by documentation plus a residue
  check). **C-310 is not absorbed**; no OpenCode/Kiro config was written.

## 2026-07-25 — M3: CPG query access — the MCP read path ✅

Asking the code graph a question is now **one tool call**, not a hand-assembled shell command.
`mcp__cpg__query(graph, cypher)` replaces `redis-cli GRAPH.QUERY` on the CPG **read** path:
the graph key and the Cypher text are parameters, so nothing has to survive a shell layer.

- **`cpg` MCP server** (`cpg/mcp/`) — a Python **FastMCP** stdio server exposing **exactly one**
  read-only tool over `GRAPH.RO_QUERY`, with `setup.sh`, `run.sh`, a README and a pytest suite
  (**53 offline / 7 live** — the component's only regression signal). Semantics: read-only;
  **`EXPLAIN`-only, `PROFILE` removed** (decision D4 — `GRAPH.PROFILE` *executes* the query
  including writes, so routing to it from a `readOnlyHint=True` tool was a read-only hole;
  `graph-dba` keeps `PROFILE` via `redis-cli`); the `PROFILE` refusal is comment-blind, because
  `/* c */ PROFILE …` through raw `GRAPH.RO_QUERY` really does return results; a typo'd graph name
  returns a curated not-found listing the loaded graphs and **does not materialise an empty key**
  (closing the known FalkorDB quirk); truncation is **display-only** (200 rows / 300-char cells /
  30,000 chars) with the notice repeated as the first *and* last line.
- **Wiring** — repo-root `.mcp.json` (`bash -c 'exec "$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh"'`, no
  absolute paths) plus `enabledMcpjsonServers` in `.claude/settings.json`. This is the repo's
  **first MCP wiring, and it is Claude-Code-only** — OpenCode and Kiro configure MCP through their
  own files and neither is wired (backlog **C-310**), so `redis-cli GRAPH.QUERY` remains their only
  path and stays documented as the fallback everywhere.
- **Consumers** — `mcp__cpg__query` added to the `analyst` and `architect` `tools:` allowlists
  (without which the tool is invisible to them; `qa-engineer` declares none and inherits) and to
  `skills/cpg-analysis/SKILL.md` `allowed-tools`, with §1 rewritten around the tool.
  `skills/agent-standards/claude-code.md` §MCP was rewritten and an **OpenCode MCP** section added,
  recording the divergences and the cross-tool rule that **MCP wiring does not port**.
- **`joern-cpg-pipeline.md` FR-9 reversed** — it had chosen `redis-cli` *"over MCP tool"*; it now
  routes through `mcp__cpg__query` and points at `docs/requirements/cpg-query-access.md`, with
  `redis-cli` as the documented fallback (**AC-4**).
- **Build, not buy** — the official `@falkordb/mcpserver` v1.3.0 exposes 7 tools including
  `delete_graph` with no tool filtering (a flat FR-2 violation) and needs Node ≥18, absent on the
  Linux side; **reversal trigger:** an upstream server that can be filtered to one read-only tool.
- **CPG rebuilt** (stakeholder-authorised destructive rebuild, decision D1) from
  `falkor-chat/server/{falkorchat,tests}`. **New baseline for `cpg_falkorchat`: 110,048 nodes ·
  734,929 edges · 1,968 METHODs · 1,019 test-file METHODs (512 `test_*`) · direct callers of
  `post_message` = 21 · test-gap = 50 rows / 43 distinct names** (the pair does not collapse to one
  number).
  ⚠ **These figures supersede the M2 numbers below** (79,581 nodes / 522,182 edges; test-gap 39
  rows / 32 distinct names). Those describe a specific build of a *moving* source tree — 8 commits
  have landed in `falkor-chat/server` since — not a property of the access mechanism. They are not
  a target and must not be iterated toward.
  The M2 entry stays as written; it was true when written.
- **Acceptance: PASS WITH DEFECTS** (`docs/test-reports/cpg-query-access-report.md`, 23 cases,
  22 pass / 1 fail). **AC-1** (one tool call, zero shell quoting; 1 tool / 2 parameters at protocol
  level), **AC-2** (multi-line ≡ single-line, byte-identical row bodies) and **AC-4** pass.
  The one failing case (TP-010) was **DEF-1**, a conflict between two approved specs — AC-3's
  *"byte-identical value sets"* vs plan §4.4's `repr` rendering for list/map cells, which cannot
  both hold for any query projecting a non-scalar. 5 of 6 tool-vs-`redis-cli` pairs were
  byte-identical; the sixth (RCA data-flow, projecting `labels()`) returned the same 44 rows in the
  same order with identical values and differed only in list syntax.
- **DEF-1 ruled the same day (stakeholder decision D5, Option A) → C-313 closed.** **AC-3 is
  narrowed to values + row counts + ordering**, excluding the display rendering of non-scalar cells,
  with plan §4.4 named as the authority for how a cell is rendered — a **specification
  reconciliation, not a code fix**: the alternative (re-rendering lists `redis-cli`-style) was
  rejected and **no source changed**. **AC-3 passes** under the reconciled wording, so
  **AC-1…AC-4 are all met**. The test report keeps its original results and verdict as the dated
  execution record, with the ruling appended as an addendum. DEF-2/DEF-3/DEF-5 remain low-severity
  cleanups (C-314/C-315/C-316).
- **Known limits:** Claude-Code-only wiring; read-only; `EXPLAIN`-only; display-only truncation;
  non-scalar cell rendering diverges from `redis-cli`; the transitive upward call-closure query is
  deferred to **C-308** (D3 — this feature changed how Cypher is *transmitted*, not how powerful it
  is). Also learned, and bigger than this feature: `FILENAME` is **relative to the Joern parse
  root**, so the parse root alone silently decides whether every `STARTS WITH 'tests/'` recipe
  filter works — and the failure is invisible in node/edge counts. That, not the missing test
  sources, is why the pre-rebuild graph was useless; a post-load check is filed as **C-312**.

Delivers M3 (FR-1…FR-6 / AC-1…AC-4 of `docs/requirements/cpg-query-access.md`, superseding FR-9 of
`joern-cpg-pipeline.md`) — items **C-301…C-307**, follow-ups **C-308…C-319** in
[`BACKLOG.md`](./BACKLOG.md). Consumer skill was M2 (2026-07-19); producer pipeline M1 (2026-07-17).

## 2026-07-19 — M2: CPG consumer skill (`cpg-analysis`) ✅

The **consumer** side of the component: one `cpg-analysis` skill teaches the agent team to
query a loaded CPG in FalkorDB with Cypher (`redis-cli GRAPH.QUERY`), closing the M2 gap.

- **`cpg-analysis` skill** (`skills/cpg-analysis/`) — lean `SKILL.md` core (connection idiom,
  silent-failure gotchas, shared traversal idioms: `CONTAINS`→`CALL`, `REACHING_DEF`,
  interprocedural bridge) plus four on-demand `references/` recipes: **impact-analysis**
  (callers/callees + transitive reach), **rca** (data-flow slice + cross-file symbol def/ref),
  **code-review** (taint to risky sinks), **test-gap** (production methods outside the
  test-reach closure). Cites the single canonical schema
  `skills/joern-cpg/references/cpg-model.md` (FR-14) — no duplicated schema; C-201 added a
  "Consumer-query facts" section there.
- **Consumers wired** (C-207): CPG-capability lines added to the `analyst`, `architect`, and
  `qa-engineer` routing descriptions (skill owned by `graph-dba`).
- **Satisfies FR-9…FR-14 / AC-2…AC-8.** Live-verified against `cpg_falkorchat` (79,581 nodes /
  522,182 edges — a Python CPG of `falkor-chat/server/{falkorchat,tests}` via `pysrc2cpg`):
  AC-2 callers=21; AC-3 transitive reach; AC-4 `REACHING_DEF` backward slice; AC-5
  `hybrid_search` cross-file def/ref; **AC-6 independent cold invocation by `analyst` passed on
  all four recipes** (correct results without hand-knowing the schema); AC-7 taint both
  directions (clean=none is a true clean with a documented coverage caveat); AC-8 test-gap =
  **39 untested-method sites / 32 distinct names**.
- **Reviews:** plan Gate-1 (`docs/reviews/m2-cpg-analysis.md`) and skill Gate-2a
  (`docs/reviews/m2-cpg-analysis-skill.md`) both **approve with suggestions**; cobb standards
  Gate-2b **accept**. All suggestions folded in.
- **Known limits:** verification is **Python-only** (JS/TS frontends not exercised);
  `REACHING_DEF` is intraprocedural in this CPG; deep interprocedural taint routes to the
  `joern` agent's `reachableBy`.

Delivers M2 (FR-9…FR-14 / AC-2…AC-8). Producer pipeline was M1 (2026-07-17).

## 2026-07-17 — M1: Producer pipeline (CPG build → FalkorDB load) ✅

First milestone: the **producer** side of the component — turn any source repository into a Code
Property Graph and materialize it in FalkorDB so the code graph is traversable with Cypher.
Delivered as commit `b2b9a6e` and **live-load verified**.

- **`joern` agent** (`claude/joern/`) — CPG specialist that operates the Joern toolset in the local
  Linux environment: builds CPGs with `joern-parse`, queries via the REPL/CPGQL (AST·CFG·CDG·DDG·PDG,
  call graphs, data-flow & taint), exports (neo4jcsv), transforms to FalkorDB-dialect Cypher, and
  ingests end-to-end.
- **`joern-cpg` skill** (`skills/joern-cpg/`) — the scripts and contract the agent drives:
  `pipeline.sh` (build → export → transform → optional load), the CPG→FalkorDB model (shared
  `:CpgNode` label + `CpgNode(id)` index, UPPER_CASE property keys, real booleans), and a CPGQL
  cheat-sheet. Schema/model reference: `skills/joern-cpg/references/cpg-model.md`.
- **Satisfies FR-1** (extract a CPG and load it into FalkorDB) and **AC-1** (a run yields a
  queryable CPG in FalkorDB). Verified against `falkordb v4.18.11`, Joern v4.0.579, JDK 21.

Consumer-side querying (letting `analyst`/`architect`/`qa-engineer` use the loaded CPG) is the next
milestone — **M2**, tracked in [`BACKLOG.md`](./BACKLOG.md) (C-200…C-208).
