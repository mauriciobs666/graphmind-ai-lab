# falkor-chat — agent working context

## Project in one sentence

A hybrid chat system (humans + AI) where **FalkorDB is the single store for everything**:
chat history, workspace data, reference data, workflow definitions and execution traces.

---

## Decisions locked in — do not reopen without strong cause

The single authoritative decision register is `falkor-chat/docs/DESIGN.md` §1 (§1.1 top-level axes, §1.2
detailed register, §1.3 M2 stack) — do not reopen any row there without strong cause. This file
carries no copy of it; follow the link.

---

## Live-verified FalkorDB facts

General engine/dialect quirks (vector index DDL, index-before-constraint ordering, the `exists()`
pattern bug, empty-`UNWIND` row collapse, `TIMEOUT` behavior, `OR`-as-scan-anchor, composite
keyset-predicate planning, etc.) live in the `graph-dba` agent's knowledge base,
**`claude/graph-dba/falkordb-quirks.md`**. This project's specific applications of those facts
(member resolution, the mention write-block guard, keyset formulation, TIMEOUT posture) are
annotated inline at the relevant query in `falkor-chat/docs/QUERIES.md` (§2, §4, §9.1) and `falkor-chat/docs/DESIGN.md`
§10 — not restated here.

---

## Graph topology

```
identity          — global user identity, auth (read-mostly, replicated)
reference         — WorkflowDef templates, ontology, tool registry (read-mostly, replicated)
ws:{workspaceId}  — per-workspace hot path: chat, embeddings, workflow runs
```

Edges cannot cross graphs. Cross-graph references use property keys or materialized snapshots.

---

## Schema conventions

- Labels: `PascalCase` — `User`, `Channel`, `Thread`, `Message`, `Agent`, `ReadCursor`
- Relationship types: `UPPER_SNAKE` — `POSTED_BY`, `REPLY_TO`, `HAS_THREAD`, `NEXT`, `MENTIONS_MEMBER`, `HAS_CURSOR`
- Properties: `camelCase` — `userId`, `createdAt`, `embedding`
- Graph keys: `ws:{workspaceId}`, `reference`, `identity`
- Every entity node has a stable `{label}Id` property, a range index, and a uniqueness constraint
- Every `MERGE` must be backed by a uniqueness constraint — no exceptions

---

## Message write paths (two variants — keep them separate)

The exact, verified Cypher lives in **one place — `falkor-chat/docs/QUERIES.md` §4** (single source of
truth); the invariants that govern it live in **`falkor-chat/docs/DESIGN.md` §5.3/§9** and the `services.py`/
`repository.py` docstrings — do not copy either into this file. Two things to hold onto: *first
message in a thread* and *subsequent message* are separate self-guarding write paths, never a
conditional MERGE, and every write returns a status row the service dispatches on (`dupMsg` =
idempotent retry, `hadHead` = lost the first-post race). See DESIGN §5.3 for the rest.

---

## Key scripts

| Script | Purpose |
|---|---|
| `./scripts/start_falkordb.sh` | Starts FalkorDB in Docker (foreground; `-d`/`--detach` for headless). Data lives in the `falkordb-data` volume. |
| `./scripts/start_server.sh` | One-shot: starts FalkorDB, bootstraps schema, seeds demo agent + workflows, starts uvicorn. Runtime env vars are documented in the script's own header comment. |
| `./scripts/bootstrap_schema.sh <wsId> …` | Creates all indexes + constraints for `reference` + workspace(s). Idempotent. **Always touches `reference` too, even when called with only a probe workspace ID** — `bootstrap_reference` (`scripts/bootstrap_schema.sh:37-70`) runs unconditionally before the per-workspace loop (`:248`), but it is exclusively `CREATE INDEX`/`GRAPH.CONSTRAINT CREATE` — no `MERGE`/`CREATE (n)`/`DELETE` — so bootstrapping a throwaway probe workspace is safe for `reference`'s *data*. |
| `./scripts/test_queries.sh` | End-to-end test suite against the live instance; must pass before any schema change is committed. **⚠️ Deletes the `reference` graph at teardown** (`GRAPH.DELETE`, not just its node data — indexes/constraints go too), wiping both published workflow defs (workspace snapshots survive, so `@mention`-to-start silently breaks). Re-run the full `bootstrap_schema.sh <wsId>` → `seed_demo.sh <wsId>` → `seed_workflows.sh <wsId>` sequence afterward — `seed_workflows.sh` alone is not enough once `reference`'s schema is gone too — or check first with `verify_workflows.sh <wsId>`. |
| `./scripts/backfill_thread_ids.sh <wsId> …` | One-off: stamps `Message.threadId` on pre-K-007 messages. Idempotent; run once per existing workspace. |
| `./scripts/load_test.sh` | Load-tests the REST append path, profiles the four hot reads, and measures per-workspace RAM delta against a throwaway `ws:load`. Env: `LOAD_MESSAGES`/`LOAD_WORKERS`/`SERVER_PORT`. |
| `./scripts/seed_demo.sh [<wsId>]` | Registers the demo AI **Agent** plus a demo `Channel`/`Thread`/`MEMBER_OF` edges, so a human can `@mention` the agent in the web UI. Idempotent; run after `bootstrap_schema.sh`. |
| `./scripts/seed_workflows.sh [<wsId>]` | Publishes + materializes the two proof workflow defs (`triage@v1`, `access-request@v1`) into `reference` + `ws:<id>`. Run after `bootstrap_schema.sh` + `seed_demo.sh`. **⚠️ Topology-immutable, not update (K-034)**: re-running after editing a def's steps/transitions/start now **fails loudly** (`409`) on the *materialize* half if the workspace snapshot predates the edit — publish a new version instead of editing this one. Property-only edits (name, step config, guard text) still silently no-op. `reference`/`ws:<id>` can still drift out of sync independently (split-brain — the workspace snapshot is what actually executes) whenever one side was never re-published/re-materialized at all. Use `verify_workflows.sh` (or the `/diff` route) to detect drift rather than reasoning about it. |
| `./scripts/verify_workflows.sh [<wsId>]` | Read-only check that `reference` and `ws:<id>` agree on both seeded defs — right version, right content, exactly one start key. Exit `0` = in sync, `1` = missing/divergent (prints what and the fix command). Never re-seeds; works with no uvicorn running. Before K-005 (fixed 2026-08-25), a full `reference` `GRAPH.DELETE` (as `test_queries.sh`'s teardown performs) made this report the `ws:<id>` snapshot itself missing too; that false negative is fixed — a report after this date reflects real state. |

Bootstrap takes an optional `EMBEDDING_DIM` env var (default `1536`). Set it to match the
embedding model before creating a workspace — **for this system that means `EMBEDDING_DIM=1024`**
(DESIGN §1.3), so `bootstrap_schema.sh`'s default is wrong for every new workspace here and must be
overridden explicitly. `start_server.sh` already passes 1024.

**`FALKORCHAT_WORKFLOW_ENABLED=1` alone is not enough to run a workflow** — the executor/trigger
are wired only *inside* the `FALKORCHAT_ENABLE_AGENT` branch of `_build_default_app()`; without
`ENABLE_AGENT` also set, `POST /workflow-runs` 503s (`WorkflowEngineDisabledError`) even though the
flags read as independent. Both flags are needed to exercise a workflow end-to-end, including the
LLM-free `access-request@v1` proof flow.

**A default (offline) `pytest` run wipes the `reference` graph at teardown (see `test_queries.sh`
above); `pytest -m live` does not** — the live-only marker deselects every offline test, so the
`wf_repo` fixture that clears `reference` never runs, and the live test seeds its own throwaway
`ws:live` instead. The re-seed obligation (`seed_workflows.sh`) attaches to a **default** `pytest`
run, not to a `-m live`-only one.

**Model/provider configuration is two hand-edited files, not env vars.** Every LLM/
embedding consumer resolves through `falkorchat.modelconfig.ModelGateway` — the pristine, shared
`FALKORCHAT_OPENCODE_CONFIG` (providers only; no product default, `scripts/start_server.sh` sets
the dev convenience default) and falkor-chat's own overlay, `FALKORCHAT_MODEL_CONFIG` (per-kind
defaults/timeouts, per-model settings; defaults to the shipped `config/models.json`). The four
legacy per-provider/per-model env vars are gone — `config.assert_no_legacy_model_env()` (see
`config.LEGACY_MODEL_ENV_VARS`) refuses to start if any is still set. Also: a ref with no `/`
resolves as a **role** to an ordered fallback chain; a per-workspace override is a **hard cap**
that wins over every consumer's own choice, reaching all four kinds including the `guard` judge;
and publishing a workflow def that names an unresolvable model or role fails at publish time
(400) instead of first use. See `falkor-chat/docs/plans/llm-provider-config.md` §4/§7 for the
design, `falkor-chat/docs/SERVER.md` §1.8 for the shipped seam, and
`config/opencode.example.json` / `config/models.json` for the shipped shapes.

**`node` is not on `PATH` on the usual dev box (WSL2).** The web unit tests are bare-`node`
scripts (`node web/tests/run-select.test.js`), so find a working interpreter before assuming the
suite is unrunnable: a Playwright-bundled Node binary, or the Windows `node.exe` reachable from
WSL, both work.

### Probing shared graph state without mutating it

- **A workflow-def *publish* has no graph seam** — `Repository._reference()`
  (`server/falkorchat/repository.py:156-158`) always resolves to `db.reference_graph()`
  (`server/falkorchat/db.py:87-94`), a hardcoded `select_graph("reference")` with no
  parameter/env override, so `publish_def` can only ever write the global graph. The isolatable
  equivalent is the **snapshot** side: `materialize_snapshot` (`repository.py:1669`) formats the
  same `_PUBLISH_CYPHER` constant (`:992`) against `self._graph(ws)` instead, and
  `get_snapshot` (`repository.py:1702`), via `_read_subgraph` (`:1031`), read it back with the
  same `_READ_META_CYPHER` (`:1016`). Any engine-semantics probe about publish/read behaviour can
  therefore run byte-identically against a throwaway `ws:<probe>` graph and be torn down with
  `GRAPH.DELETE`, instead of touching `reference`.
- **`server/tests/test_services.py` is the review-safe pytest subset** — it builds
  `Services(FakeRepo())` (a module-local fake) and requests no `conftest.py` fixture that reaches
  a real `conn`/`wf_repo`, so it (and any `test_api.py` node that builds its own throwaway
  `FastAPI` app rather than requesting `wf_client`) can run against a live shared instance with
  zero risk to `reference` or any `ws:<id>`. Pair with `./scripts/verify_workflows.sh <wsId>`
  (read-only, `GRAPH.RO_QUERY` only) for a before/after check that a review didn't disturb shared
  state.

### M1 server (`server/`)

The M1 app (FastAPI REST + MCP Streamable-HTTP + static web UI on one process) lives in `server/`
(and `web/`). No `uv` on the box — use a `venv`.

```bash
cd server
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # first time
.venv/bin/python -m pytest -q                                # needs FalkorDB up; network-free
.venv/bin/python -m pytest -m live -s                        # opt-in live e2e — needs LM Studio too
.venv/bin/uvicorn falkorchat.app:app                         # web UI + REST under /, MCP at /mcp
```

Application architecture (layering, front doors, REST/MCP surface, layout) is
`falkor-chat/docs/SERVER.md` §1–§2 — not restated here. **Testing hazards specific to this suite**
(the pytest-side destructive-reference-graph gotcha, skip-count reading, `ws:test`'s fixed dim-4
vector index, ruff not being a wired gate) are `falkor-chat/docs/SERVER.md` §1.7. Model-output parse tolerance (`llm.py`)
and the executor/workflow-def invariants are documented at their own definitions
(`llm.py` docstrings; `services._validate_def_spec`, `executor.py`) and in `falkor-chat/docs/DESIGN.md` §6 —
read the code, don't look for a copy here.

---

## Key documents

| File | Contents |
|---|---|
| `falkor-chat/docs/DESIGN.md` | The **graph** blueprint — topology, data model, indexes, capacity, ops, roadmap. The *why*; queries live in QUERIES.md, DDL in `bootstrap_schema.sh`. Stops at the graph. |
| `falkor-chat/docs/SERVER.md` | The **server process** — layering, auth/tenancy seam, REST + MCP front doors, `server/` layout, testing hazards, model-resolution seam. (DESIGN §14 redirects here.) |
| `falkor-chat/docs/test-reports/capacity-report.md` | All capacity measurements — per-message RAM, append throughput, hot-read plans, shard packing. DESIGN §11 keeps only the design-shaping numbers. |
| `falkor-chat/docs/QUERIES.md` | Canonical query library, verified against the live instance — source of truth for queries. |
| `falkor-chat/docs/BACKLOG.md` | Forward-looking backlog: open K-numbered items and the milestones still open. A delivered item is not kept there — its record is `HISTORY.md`. |
| `falkor-chat/docs/HISTORY.md` | Dated change log, most recent first — one entry per delivered change. |
| `falkor-chat/docs/archive/` | Frozen plans/test-plans/test-reports from closed milestones — **read-only history of the previous convention, not a destination.** Nothing moves here and nothing is un-archived; a document that freezes stays in place with `Status: archived` (root `AGENTS.md`). |
| `scripts/bootstrap_schema.sh` | Source of truth for executable DDL — indexes, constraints, full-text/vector. |
| `claude/graph-dba/falkordb-quirks.md` | General FalkorDB engine/dialect facts for this lab's pinned build — not project-specific. |
| `falkor-chat/docs/archive/plans/m1-chat-mcp.md` | K-002 plan: MCP transport + mentions + read-cursors. |
| `falkor-chat/docs/archive/plans/m2-groundwork.md` · `falkor-chat/docs/archive/plans/m2-groundwork-queries.md` | K-007 plan + verified-query deliverable: v2 write paths, keyset cursors, threadId denorm, TIMEOUT/RAM findings. |

---

## Rules for future work

1. **Always parameterise Cypher.** Never interpolate variables into query strings.
2. **Verify dialect before assuming.** This is FalkorDB OpenCypher, not Neo4j. No APOC, no GDS, no `PROFILE` keyword prefix. Check `CALL dbms.procedures()` when unsure.
3. **Profile before tuning.** Use `GRAPH.PROFILE` to confirm an index is actually hit before declaring a query fast. Look for `Node By Index Scan`, not `NodeByLabelScan`.
4. **All writes that touch HEAD/TAIL must be a single `GRAPH.QUERY`** — atomicity is per-query.
5. **Test suite must stay green.** The full suite (`./scripts/test_queries.sh`) must pass before any schema or query change is committed.
6. **RAM is the binding constraint.** Any new node type, index, or vector dimension affects per-workspace RAM. Call it out.
7. **One graph per workspace.** Never add a `workspaceId` property to filter inside a shared graph.
8. **`ctx`, `input`, `output` on workflow nodes are serialised strings.** Do not design queries that filter inside them.
