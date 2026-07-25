# falkor-chat — agent working context

## Project in one sentence

A hybrid chat system (humans + AI) where **FalkorDB is the single store for everything**:
chat history, workspace data, reference data, workflow definitions and execution traces.

---

## Decisions locked in — do not reopen without strong cause

The single authoritative decision register is `docs/DESIGN.md` §1 (§1.1 top-level axes, §1.2
detailed register, §1.3 M2 stack) — do not reopen any row there without strong cause. This file
carries no copy of it; follow the link.

---

## Live-verified FalkorDB facts

General engine/dialect quirks (vector index DDL, index-before-constraint ordering, the `exists()`
pattern bug, empty-`UNWIND` row collapse, `TIMEOUT` behavior, `OR`-as-scan-anchor, composite
keyset-predicate planning, etc.) live in the `graph-dba` agent's knowledge base,
**`claude/graph-dba/falkordb-quirks.md`**. This project's specific applications of those facts
(member resolution, the mention write-block guard, keyset formulation, TIMEOUT posture) are
annotated inline at the relevant query in `docs/QUERIES.md` (§2, §4, §9.1) and `docs/DESIGN.md`
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

The exact, verified Cypher lives in **one place — `docs/QUERIES.md` §4** (single source of
truth); the invariants that govern it live in **`docs/DESIGN.md` §5.3/§9** and the `services.py`/
`repository.py` docstrings — do not copy either into this file. Two things to hold onto: *first
message in a thread* and *subsequent message* are separate self-guarding write paths, never a
conditional MERGE, and every write returns a status row the service dispatches on (`dupMsg` =
idempotent retry, `hadHead` = lost the first-post race). See DESIGN §5.3 for the rest.

---

## Key scripts

| Script | Purpose |
|---|---|
| `./scripts/start_falkordb.sh` | Start FalkorDB in Docker (foreground; `-d`/`--detach` for headless). Data in `falkordb-data` volume. |
| `./scripts/start_server.sh` | One-shot: start FalkorDB, bootstrap schema, seed demo agent + workflows, start uvicorn. Every runtime env var (`FALKORCHAT_ENABLE_AGENT`, `EMBEDDING_DIM`, `FALKORCHAT_WORKFLOW_ENABLED`, etc.) is documented in the script's own header comment — read there, not here. |
| `./scripts/bootstrap_schema.sh <wsId> …` | Create all indexes + constraints for `reference` + workspace(s). Idempotent. |
| `./scripts/test_queries.sh` | End-to-end test suite against the live instance. Must pass before any schema change is committed. **⚠️ Deletes the global `reference` graph at teardown** — that wipes **both** published defs, `triage@v1` and `access-request@v1` (the `ws:<id>` snapshots survive), so `@mention`-to-start silently no-ops afterwards. **Re-run `./scripts/seed_workflows.sh <wsId>` after this suite** before exercising a workflow flow. (`start_server.sh` self-heals — it seeds on every start.) **To check rather than assume, run `./scripts/verify_workflows.sh <wsId>`** — it exits 1 and names the missing def. |
| `./scripts/backfill_thread_ids.sh <wsId> …` | One-off: stamp `Message.threadId` on pre-K-007 messages (QUERIES.md §4.x). Idempotent; run once per existing workspace after deploying the v2 write paths. |
| `./scripts/load_test.sh` | K-011 M1 DoD closeout harness: load-tests the REST append path (`scripts/load_append.py`), `GRAPH.PROFILE`s the four hot reads, and captures a per-workspace RAM delta — all against an isolated throwaway `ws:load` (torn down at the end unless `KEEP_WS=1`). Results folded into DESIGN §11.1–§11.2. Env: `LOAD_MESSAGES`/`LOAD_WORKERS`/`SERVER_PORT`. Needs FalkorDB up + the `server/.venv`. |
| `./scripts/seed_demo.sh [<wsId>]` | K-014 M2 demo seed: registers the AI **Agent** (`FALKORCHAT_AGENT_ID`, default `assistant`) + a demo `Channel`/`Thread` (fixed ids → MERGE, backed by the uniqueness constraints) + `MEMBER_OF` edges, so a human can open the web UI and `@mention` the agent. Idempotent. `start_server.sh` runs it automatically. Run `bootstrap_schema.sh` first. |
| `./scripts/seed_workflows.sh [<wsId>]` | Publishes + materializes **two** proof workflow defs into `reference` + `ws:<id>`, looping over both: **(1) `triage@v1`** (K-022 U13, kind `conversation`, intake→research→answer `type:'agent'` steps per `docs/archive/plans/m3-executor.md` §8; **def content inline in the script**, key/version must match `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION`) and **(2) `access-request@v1`** (K-024 U4, kind `process`, the LLM-free proof flow of `docs/archive/plans/m3-process-flow.md` §4 — submit→route→approval→provision→activate\|rejected over `human`/`decision`/`wait` steps and six `cmp` guards; **def content imported from `server/falkorchat/proof_defs.py` (`ACCESS_REQUEST_DEF`)**, the same constant the offline acceptance test `server/tests/test_process_flow.py` drives, so seed and test cannot drift. Started over REST, not by `@mention` ⇒ **no config var refers to it**; the script's local `FALKORCHAT_PROCESS_DEF_KEY`/`_VERSION` overrides would seed a def nothing else points at). The two def-source conventions are deliberate for this slice — converging them is proposed K-029. Wraps a Python one-shot over the **service layer** (`publish_workflow_def`+`materialize_def` — real validation/start-key derivation/publish invariants, not raw Cypher). Additive-only, idempotent (MERGE on the fixed `key`/`version`); a clean re-run prints `already present — no-op` for **both** defs. Run **after** `bootstrap_schema.sh` + `seed_demo.sh`. `start_server.sh` runs it when `FALKORCHAT_WORKFLOW_ENABLED` is on (its default there). **Re-run it after `test_queries.sh` or a `server` pytest run** — but for *different* reasons, and only one of them empties the graph. `test_queries.sh` deletes `reference` at **teardown**, taking both defs with it. `server/tests`' `wf_repo` fixture wipes `reference` at fixture **setup**, once per workflow test, so a finished pytest session *leaves the last workflow test's defs behind* — meaning `already present — no-op` after a pytest run may be reporting a **test's** publish, not a real seed, while `ws:<id>` still holds the older snapshot the executor actually drives. (The acceptance test `test_process_flow.py` publishes under the test-only version `access-request@v1-test` precisely so it cannot collide with the production pair; anything else published by a test can.) **⚠️ Published defs are effectively IMMUTABLE — "idempotent" means *create-only*, not *update* — for both defs alike.** `repository._PUBLISH_CYPHER` is `MERGE (st:Step …) ON CREATE SET st.config` (same shape for `d.name`/`d.kind` and `rel.guard`, and `materialize_snapshot` reuses it), so **editing a `systemPrompt`/guard in this script — or a step config/guard in `proof_defs.py` — and re-running changes nothing live**: the run prints a clean `already present — no-op` while the old config stays. Worse, `reference` (def) and `ws:<id>` (snapshot) go stale **independently**: `test_queries.sh` and `server/tests`' `wf_repo` fixture each clear `reference` — by the two *different* mechanisms described above — but neither touches `ws:<id>`, so a naive re-seed republishes the *new* def while the workspace keeps the *old* snapshot — a silent split-brain, and the snapshot is what the executor drives. Landing a def edit therefore requires an explicit act (delete the def + snapshot subgraphs and republish, or a `key`/`version` bump — for `triage` kept in sync with `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION`, note `start_server.sh` neither forwards nor exports those two vars today, so a version bump also needs a script change; for `access-request` kept in sync with `proof_defs.py` and its acceptance test). Deleting a snapshot breaks live `WorkflowRun`s that point at it via `OF_DEF`/`AT_STEP` — a destructive shared-state op, not a routine re-seed. **And here is how you now DETECT all of that (K-031)** instead of reasoning about it: `GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff` (one call → `inSync` + an enumerated difference list), the two structure reads `GET /workflow-defs/{key}/versions/{version}` and `GET /workspaces/{ws}/snapshots/{key}/versions/{version}`, or **`./scripts/verify_workflows.sh <wsId>`** for both seeded defs at once with no server running. Note the diff is *version-qualified* — it answers "same version, different content", never "wrong version"; `verify_workflows.sh` covers the version case because it reads the expected version from `config`/`proof_defs`. |
| `./scripts/verify_workflows.sh [<wsId>]` | K-031 read-only check that `reference` and `ws:<id>` still agree about the two seeded defs (`triage@v1` from `config.TRIGGER_DEF_KEY`/`_VERSION`, `access-request@v1` from `proof_defs.ACCESS_REQUEST_DEF` — the same sources `seed_workflows.sh` publishes from, so the check cannot drift from the seed). Per def it verifies: published in `reference` **at the expected version** (the version-staleness check the `/diff` route is structurally unable to do), materialized in `ws:<id>` at that version, `inSync` per the comparator (printing every difference), and exactly **one** start key (a `startKeys` list means a root grew a second `START` edge — see K-034). Exit **0** = all green, **1** = anything missing or divergent, with the `seed_workflows.sh` command printed. Drives the **service layer** via a Python one-shot, so it works with **no uvicorn running** — which is exactly when it is most needed (right after `test_queries.sh` or a pytest run). **⚠️ Strictly read-only** — it must never gain a "let me just re-seed that for you" fallback: a create-only re-publish cannot overwrite a divergent def, and deleting a snapshot breaks live `WorkflowRun`s. Needs FalkorDB up + the `server/.venv`. |

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

Application architecture (layering, front doors, REST/MCP surface, layout) is `docs/DESIGN.md`
§14–§15 — not restated here. **Testing hazards specific to this suite** (the pytest-side
destructive-reference-graph gotcha, skip-count reading, `ws:test`'s fixed dim-4 vector index,
ruff not being a wired gate) are `docs/DESIGN.md` §14.7. Model-output parse tolerance (`llm.py`)
and the executor/workflow-def invariants are documented at their own definitions
(`llm.py` docstrings; `services._validate_def_spec`, `executor.py`) and in `docs/DESIGN.md` §6 —
read the code, don't look for a copy here.

---

## Key documents

| File | Contents |
|---|---|
| `docs/DESIGN.md` | Full blueprint: graph topology, data model, indexes, ops, roadmap, §14–§15 M1 app + MCP. The *why*; not a query/DDL copy — §5.3/§8 point to QUERIES.md, §7 points to `bootstrap_schema.sh`. |
| `docs/QUERIES.md` | Canonical query library — all verified against the live instance (source of truth for **queries**) |
| `docs/BACKLOG.md` | Forward-looking backlog: K-numbered items, milestone map, sequencing (formerly `kaizen/plan.md`) |
| `docs/HISTORY.md` | Dated change log, most recent first — every delivered change gets an entry (formerly `kaizen/history.md`) |
| `docs/archive/` | Frozen plans/test-plans/test-reports of closed milestones (same subdir names as the active dirs); a doc moves here when its milestone closes, inbound links fixed in the same change |
| `scripts/bootstrap_schema.sh` | Source of truth for **executable DDL** (indexes + constraints + full-text/vector); DESIGN §7 describes it, doesn't duplicate it |
| `claude/graph-dba/falkordb-quirks.md` | General FalkorDB engine/dialect facts verified against this lab's pinned build — not project-specific; this project's applications of those facts live inline in QUERIES.md/DESIGN.md |
| `docs/archive/plans/m1-chat-mcp.md` | K-002 plan: MCP transport + mentions + read-cursors |
| `docs/archive/plans/m2-groundwork.md` · `docs/archive/plans/m2-groundwork-queries.md` | K-007 plan + graph-dba verified-query deliverable: v2 write paths, keyset cursors, threadId denorm, TIMEOUT/RAM findings |

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
