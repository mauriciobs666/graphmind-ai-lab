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
| `./scripts/bootstrap_schema.sh <wsId> …` | Creates all indexes + constraints for `reference` + workspace(s). Idempotent. |
| `./scripts/test_queries.sh` | End-to-end test suite against the live instance; must pass before any schema change is committed. **⚠️ Deletes the `reference` graph at teardown**, wiping both published workflow defs (workspace snapshots survive, so `@mention`-to-start silently breaks). Re-run `seed_workflows.sh <wsId>` afterward, or check first with `verify_workflows.sh <wsId>`. |
| `./scripts/backfill_thread_ids.sh <wsId> …` | One-off: stamps `Message.threadId` on pre-K-007 messages. Idempotent; run once per existing workspace. |
| `./scripts/load_test.sh` | Load-tests the REST append path, profiles the four hot reads, and measures per-workspace RAM delta against a throwaway `ws:load`. Env: `LOAD_MESSAGES`/`LOAD_WORKERS`/`SERVER_PORT`. |
| `./scripts/seed_demo.sh [<wsId>]` | Registers the demo AI **Agent** plus a demo `Channel`/`Thread`/`MEMBER_OF` edges, so a human can `@mention` the agent in the web UI. Idempotent; run after `bootstrap_schema.sh`. |
| `./scripts/seed_workflows.sh [<wsId>]` | Publishes + materializes the two proof workflow defs (`triage@v1`, `access-request@v1`) into `reference` + `ws:<id>`. Run after `bootstrap_schema.sh` + `seed_demo.sh`. **⚠️ Topology-immutable, not update (K-034)**: re-running after editing a def's steps/transitions/start now **fails loudly** (`409`) on the *materialize* half if the workspace snapshot predates the edit — publish a new version instead of editing this one. Property-only edits (name, step config, guard text) still silently no-op. `reference`/`ws:<id>` can still drift out of sync independently (split-brain — the workspace snapshot is what actually executes) whenever one side was never re-published/re-materialized at all. Use `verify_workflows.sh` (or the `/diff` route) to detect drift rather than reasoning about it. |
| `./scripts/verify_workflows.sh [<wsId>]` | Read-only check that `reference` and `ws:<id>` agree on both seeded defs — right version, right content, exactly one start key. Exit `0` = in sync, `1` = missing/divergent (prints what and the fix command). Never re-seeds; works with no uvicorn running. |

Bootstrap takes an optional `EMBEDDING_DIM` env var (default `1536`). Set it to match the
embedding model before creating a workspace.

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

Application architecture (layering, front doors, REST/MCP surface, layout) is `falkor-chat/docs/DESIGN.md`
§14–§15 — not restated here. **Testing hazards specific to this suite** (the pytest-side
destructive-reference-graph gotcha, skip-count reading, `ws:test`'s fixed dim-4 vector index,
ruff not being a wired gate) are `falkor-chat/docs/DESIGN.md` §14.7. Model-output parse tolerance (`llm.py`)
and the executor/workflow-def invariants are documented at their own definitions
(`llm.py` docstrings; `services._validate_def_spec`, `executor.py`) and in `falkor-chat/docs/DESIGN.md` §6 —
read the code, don't look for a copy here.

---

## Key documents

| File | Contents |
|---|---|
| `falkor-chat/docs/DESIGN.md` | Full blueprint — graph topology, data model, indexes, ops, roadmap, §14–§15 M1 app + MCP. The *why*; queries live in QUERIES.md, DDL in `bootstrap_schema.sh`. |
| `falkor-chat/docs/QUERIES.md` | Canonical query library, verified against the live instance — source of truth for queries. |
| `falkor-chat/docs/BACKLOG.md` | Forward-looking backlog: K-numbered items, milestone map, sequencing. |
| `falkor-chat/docs/HISTORY.md` | Dated change log, most recent first — one entry per delivered change. |
| `falkor-chat/docs/archive/` | Frozen plans/test-plans/test-reports from closed milestones — **read-only history of the previous convention, not a destination.** Nothing moves here any more and nothing is un-archived; a document that freezes now stays in place and gets `Status: archived` in its header (root `AGENTS.md`). |
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
