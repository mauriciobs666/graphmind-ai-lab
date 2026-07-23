# falkor-chat — agent working context

## Project in one sentence

A hybrid chat system (humans + AI) where **FalkorDB is the single store for everything**:
chat history, workspace data, reference data, workflow definitions and execution traces.

---

## Decisions locked in — do not reopen without strong cause

> Rationale lives once, in `docs/DESIGN.md` §1 (the authoritative register). This is the quick
> do-not-reopen index — follow the link for the *why*.

| Decision | Home |
|---|---|
| FalkorDB is the single store (no secondary store) | DESIGN §1.2 → §2 |
| One graph per workspace (`ws:{id}`) | DESIGN §1.1 (Tenancy) / §3 |
| Thread-scoped `NEXT` linked list | DESIGN §1.2 → §5.2 |
| No DayBucket | DESIGN §1.2 |
| `Thread` owns `HEAD`+`TAIL` pointers | DESIGN §1.2 → §5.2 |
| `Message.role` inline + derived server-side | DESIGN §1.2 → §5.1 |
| `coalesce` member identity | DESIGN §1.2 → QUERIES §2 |
| Vector indexes via DDL, not a procedure | DESIGN §1.2 → §7.1 |
| Index before constraint, always | DESIGN §1.2 → §7.1 |
| `Message.embedding` inline `vecf32` | DESIGN §1.2 → §5.2 |
| Vector score is cosine distance (ASC) | DESIGN §1.2 → §8 |
| `status` as property, not label | DESIGN §1.2 → §6.2 |
| Flat `ctx`/`input`/`output` | DESIGN §1.2 → §6.2 |
| `Message.threadId` denorm, unindexed | DESIGN §1.2 → §5.1 |
| Guarded-CREATE write paths + status row | DESIGN §1.2 → §5.3/§9 |
| Composite `(createdAt, msgId)` keyset cursor | DESIGN §1.2 → QUERIES §9 |
| Member ids namespace-unique across `User`/`Agent` | DESIGN §1.2 → QUERIES §2/§7 |
| Identity graph is authoritative (standalone) | DESIGN §1.2 → §3 |

---

## Live-verified FalkorDB facts (falkordb/falkordb:v4.18.11, Redis 8.6.3, module 41811)

General engine/dialect quirks verified against this build (vector index DDL, index-before-constraint
ordering, the `exists()` pattern bug, empty-`UNWIND` row collapse, `TIMEOUT` behavior, `OR`-as-scan-anchor,
etc.) now live in the `graph-dba` agent's knowledge base, **`claude/graph-dba/falkordb-quirks.md`** —
check there first. What's below is specific to this project's schema/queries:

- **`repository.thread_has_head`/`thread_exists`** exist specifically to route around graph-dba's
  `exists()`-pattern-bug finding — they use `OPTIONAL MATCH (n)-[:REL]->(x) RETURN x IS NOT NULL`,
  never a pattern-`exists()` check.
- **The mention write-block's empty-`UNWIND` guard is load-bearing for the write itself**, not just
  the mentions (see `QUERIES.md` §4 mentions note): `UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE
  $mentions END) AS mid` + a `FOREACH` that never filters. A bare `UNWIND []` would collapse the whole
  row stream before that `FOREACH`, silently dropping the message write, not just the
  `MENTIONS_MEMBER` edges.
- **Member resolution (`userId`/`agentId`) is the concrete case of graph-dba's `OR`-scan-anchor
  quirk** — two `OPTIONAL MATCH (u:User {userId:mid})` / `(a:Agent {agentId:mid})` + `coalesce(u,a)`
  for anchored lookups (`labels(coalesce(u,a))[0]` gives the member kind). The `OR` form is fine only
  in mention-flag and cursor reads, where `n` is already bound by a traversal/indexed anchor.
- **Formulation-A composite keyset predicate** (`m.createdAt > $since OR (m.createdAt = $since
  AND m.msgId > $sinceMsgId)`) still plans as a bare `Node By Index Scan` on `Message.createdAt`
  with no residual Filter — **re-profile on engine upgrades** (edge build; formulation B in
  QUERIES.md §9.1 is the documented fallback).
- **`TIMEOUT` default (1000ms) was reviewed for M2 (K-007) and kept as the deployment default** —
  writes ignore it entirely regardless (graph-dba finding); GraphRAG reads pass a per-query client
  `timeout=` override instead (DESIGN §10 posture).

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
truth). Do not copy query bodies here or into `DESIGN.md`; link to QUERIES.md instead — the
duplication is what lets the copies drift. The invariants that govern those queries (v2, K-007):

- **Two separate write paths, never a conditional MERGE:** *first message in a thread*
  (creates `HEAD` + `TAIL`) vs. *subsequent message* (moves `TAIL` forward via `NEXT`). Each is
  **self-guarding**: the write happens inside a `FOREACH (… IN CASE WHEN ok THEN [1] ELSE []
  END | …)` guard *per path* — a guarded `CREATE`, **no MERGE on Message** (the constraint
  stays as the concurrency backstop). The service picks the initial variant by checking for a
  `HEAD`, then dispatches on the returned status row.
- **Status-row contract:** both paths always return `(written, hadHead, dupMsg, authorFound)`
  when their anchor matches. **Zero rows = the anchor missed only** (first: thread missing →
  404; subsequent: no TAIL → retry as first). `dupMsg=true` = **idempotent success** (a retry
  replay of our own server-minted msgId — trusted without a payload check; add a checksum if
  msgIds ever become client-supplied). `hadHead=true` = lost the first-post race → re-dispatch
  as subsequent. `authorFound=false` = unknown member, nothing written. The dispatch loop is
  bounded at 4 attempts (tripwire — ping-pong is impossible by contract).
- **Each write is a single `GRAPH.QUERY`** — atomicity is per-query; the HEAD/TAIL relink must
  not be split across queries.
- **`role` is derived, never trusted:** the service resolves the author's label
  (`User → user`, `Agent → assistant`) via the §2 member-kind lookup — Agents author
  first-class. Author resolution in the write is label-specific (two indexed `OPTIONAL
  MATCH`es + `coalesce`), closing the old `All Node Scan`/silent-Agent-no-op defect.
- **Every message records its author** with `(m)-[:POSTED_BY]->(author)`. The canonical
  thread-read path (`QUERIES.md` §4) *requires* that edge — a message written without it is
  invisible to thread reads.
- **Participant mentions ride inside the same write query.** Mention resolution runs before the
  guard; the nested `FOREACH` creates `(m)-[:MENTIONS_MEMBER]->(member)` edges *inside* it —
  never a follow-up query (atomicity rule). The empty-`UNWIND` `CASE` guard is now
  **load-bearing for the write itself** (a bare `UNWIND []` collapses the stream before the
  FOREACH). `MENTIONS_MEMBER` (participants) is **distinct from** `MENTIONS`→`Entity` (GraphRAG
  co-occurrence, §6) — do not conflate them. `$mentions = []` is a verified no-op.
- **Every `MERGE` is backed by a uniqueness constraint** (`ReadCursor.cursorId`; `ensure_user`/
  `ensure_agent`). Channel/thread creates are plain `CREATE` (server-minted ids —
  **non-idempotent**, a retried create mints a new id).
- **The service owns the timestamps:** message `createdAt` comes from a lock-guarded monotonic
  per-process clock (`max(clock, last+1)`) — same-ms ties are impossible at the source.
- **Since-reads (§9.1/§9.2) are chronological in the `(createdAt, msgId)` total order; cursors
  advance to what was delivered.** Reader mentions are carried by the `isMention` flag, never a
  mention-first sort — a resorted page + `LIMIT` breaks the contiguous-prefix invariant.
  Cursor-driven reads use the composite keyset + the composite `ReadCursor` pair (advanced to
  the newest *returned* `(createdAt, msgId)`, never the server clock) and never skip or
  re-deliver, even across millisecond ties. Explicit-`since` reads keep plain `>` semantics
  (may re-deliver/skip within that exact millisecond — documented, OQ3).

---

## Key scripts

| Script | Purpose |
|---|---|
| `./scripts/start_falkordb.sh` | Start FalkorDB in Docker (foreground; `-d`/`--detach` for headless). Data in `falkordb-data` volume. |
| `./scripts/bootstrap_schema.sh <wsId> …` | Create all indexes + constraints for `reference` + workspace(s). Idempotent. |
| `./scripts/test_queries.sh` | End-to-end test suite against the live instance. Must pass before any schema change is committed. **⚠️ Deletes the global `reference` graph at teardown** — that wipes **both** published defs, `triage@v1` and `access-request@v1` (the `ws:<id>` snapshots survive), so `@mention`-to-start silently no-ops afterwards. **Re-run `./scripts/seed_workflows.sh <wsId>` after this suite** before exercising a workflow flow. (`start_server.sh` self-heals — it seeds on every start.) |
| `./scripts/backfill_thread_ids.sh <wsId> …` | One-off: stamp `Message.threadId` on pre-K-007 messages (QUERIES.md §4.x). Idempotent; run once per existing workspace after deploying the v2 write paths. |
| `./scripts/load_test.sh` | K-011 M1 DoD closeout harness: load-tests the REST append path (`scripts/load_append.py`), `GRAPH.PROFILE`s the four hot reads, and captures a per-workspace RAM delta — all against an isolated throwaway `ws:load` (torn down at the end unless `KEEP_WS=1`). Results folded into DESIGN §11.1–§11.2. Env: `LOAD_MESSAGES`/`LOAD_WORKERS`/`SERVER_PORT`. Needs FalkorDB up + the `server/.venv`. |
| `./scripts/seed_demo.sh [<wsId>]` | K-014 M2 demo seed: registers the AI **Agent** (`FALKORCHAT_AGENT_ID`, default `assistant`) + a demo `Channel`/`Thread` (fixed ids → MERGE, backed by the uniqueness constraints) + `MEMBER_OF` edges, so a human can open the web UI and `@mention` the agent. Idempotent. `start_server.sh` runs it automatically. Run `bootstrap_schema.sh` first. |
| `./scripts/seed_workflows.sh [<wsId>]` | Publishes + materializes **two** proof workflow defs into `reference` + `ws:<id>`, looping over both: **(1) `triage@v1`** (K-022 U13, kind `conversation`, intake→research→answer `type:'agent'` steps per `docs/archive/plans/m3-executor.md` §8; **def content inline in the script**, key/version must match `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION`) and **(2) `access-request@v1`** (K-024 U4, kind `process`, the LLM-free proof flow of `docs/archive/plans/m3-process-flow.md` §4 — submit→route→approval→provision→activate\|rejected over `human`/`decision`/`wait` steps and six `cmp` guards; **def content imported from `server/falkorchat/proof_defs.py` (`ACCESS_REQUEST_DEF`)**, the same constant the offline acceptance test `server/tests/test_process_flow.py` drives, so seed and test cannot drift. Started over REST, not by `@mention` ⇒ **no config var refers to it**; the script's local `FALKORCHAT_PROCESS_DEF_KEY`/`_VERSION` overrides would seed a def nothing else points at). The two def-source conventions are deliberate for this slice — converging them is proposed K-029. Wraps a Python one-shot over the **service layer** (`publish_workflow_def`+`materialize_def` — real validation/start-key derivation/publish invariants, not raw Cypher). Additive-only, idempotent (MERGE on the fixed `key`/`version`); a clean re-run prints `already present — no-op` for **both** defs. Run **after** `bootstrap_schema.sh` + `seed_demo.sh`. `start_server.sh` runs it when `FALKORCHAT_WORKFLOW_ENABLED` is on (its default there). **Re-run it after `test_queries.sh` or a `server` pytest run** — but for *different* reasons, and only one of them empties the graph. `test_queries.sh` deletes `reference` at **teardown**, taking both defs with it. `server/tests`' `wf_repo` fixture wipes `reference` at fixture **setup**, once per workflow test, so a finished pytest session *leaves the last workflow test's defs behind* — meaning `already present — no-op` after a pytest run may be reporting a **test's** publish, not a real seed, while `ws:<id>` still holds the older snapshot the executor actually drives. (The acceptance test `test_process_flow.py` publishes under the test-only version `access-request@v1-test` precisely so it cannot collide with the production pair; anything else published by a test can.) **⚠️ Published defs are effectively IMMUTABLE — "idempotent" means *create-only*, not *update* — for both defs alike.** `repository._PUBLISH_CYPHER` is `MERGE (st:Step …) ON CREATE SET st.config` (same shape for `d.name`/`d.kind` and `rel.guard`, and `materialize_snapshot` reuses it), so **editing a `systemPrompt`/guard in this script — or a step config/guard in `proof_defs.py` — and re-running changes nothing live**: the run prints a clean `already present — no-op` while the old config stays. Worse, `reference` (def) and `ws:<id>` (snapshot) go stale **independently**: `test_queries.sh` and `server/tests`' `wf_repo` fixture each clear `reference` — by the two *different* mechanisms described above — but neither touches `ws:<id>`, so a naive re-seed republishes the *new* def while the workspace keeps the *old* snapshot — a silent split-brain, and the snapshot is what the executor drives. Landing a def edit therefore requires an explicit act (delete the def + snapshot subgraphs and republish, or a `key`/`version` bump — for `triage` kept in sync with `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION`, note `start_server.sh` neither forwards nor exports those two vars today, so a version bump also needs a script change; for `access-request` kept in sync with `proof_defs.py` and its acceptance test). Deleting a snapshot breaks live `WorkflowRun`s that point at it via `OF_DEF`/`AT_STEP` — a destructive shared-state op, not a routine re-seed. |

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

- **Layering (locked):** `api.py` (REST) and `mcp.py` (MCP) are thin adapters over `services.py`;
  all Cypher lives in `repository.py` (1:1 with `QUERIES.md`); the tenant seam is `config.get_context`.
- **Front doors on one process:** `app.py` mounts REST + MCP, and serves the repo-root `web/`
  (`index.html` + `app.js`) as static files at `/`. The static mount is registered **last** — `/`
  is a catch-all that must sit behind the REST routes and the `/mcp` mount. Same-origin ⇒ no CORS.
- **Full-text search:** `GET /search?q=` → `services.search_messages` → `repository.search_messages`
  (`QUERIES.md` §5, workspace-wide — the channel-scoping MATCH is omitted).
- Repository/services tests run against the isolated `ws:test` graph (same approach as
  `test_queries.sh`); the `conftest` fixture bootstraps schema + wipes node data per test.
  **`ws:test` vector indexes are dim 4** (`conftest.TEST_EMBEDDING_DIM`) — never use it for a
  real-embedder test: a wrong-dim `vecf32` write is silently accepted and then drops out of ANN,
  so the retrieval passes while finding nothing.
- MCP is tested in-memory (`mcp.call_tool` / `list_tools`) — no HTTP server needed.
- **The `live` pytest marker (K-022 U14):** tests needing a **real LM Studio** are marked
  `@pytest.mark.live` and **deselected by default** (`addopts = -ra -m "not live"` in
  `server/pyproject.toml`), so the standard `pytest` baseline stays network-free and fast *even
  when LM Studio is running* — a reachability-skip alone would not do that. Opt in with
  `pytest -m live` (a command-line `-m` overrides the addopts one). Live tests still skip with a
  reason when a dep is unreachable, so they never fail for environmental reasons.
  `tests/test_workflow_live.py` is the triage-flow e2e (AC-1…AC-4): it needs FalkorDB **and** LM
  Studio (chat + embedding model) at `:1234`, and builds its own throwaway **`ws:live`** graph
  bootstrapped at the **probed** live embedding dim (never hardcoded 1024 — the loaded model
  decides), seeding the real def via `scripts/seed_workflows.sh` rather than a copy that could
  drift. `KEEP_WS=1` keeps the graph for post-mortem inspection.
- **Live AI agent loop (K-014, gated):** `app.py` builds the module-level `app` via
  `_build_default_app()`, which wires the real `LMStudioEmbedder` + `EmbeddingWorker` +
  `LMStudioLLM` + `AgentResponder` **only when `FALKORCHAT_ENABLE_AGENT` is truthy** — off by
  default so imports and the pytest baseline stay network-free. The served app must also run at
  the workspace's embedding dimension (`FALKORCHAT_EMBEDDING_DIM=1024` for `ws:acme`) or embeddings
  silently drop out of ANN. `scripts/start_server.sh` sets both, seeds the demo, and starts uvicorn;
  `server/.env.example` documents every runtime env var. `@mention`-ing `FALKORCHAT_AGENT_ID` (default
  `assistant`) triggers a retrieval-grounded reply posted as the Agent (role `assistant`) with an
  `EMITTED` provenance edge. **Channel scoping is workspace-wide for M2-green** (`responder` passes
  `channel_id=None`; a thread→channel read isn't in QUERIES.md yet) — K-015 follow-up.
- **Since-read `displayName` (K-014):** `read_thread_since`/`read_ws_since` (QUERIES.md §9.1/§9.2)
  carry `author.displayName` so the polling web client shows member names, not raw ids; clients
  tolerate `null`.
- **Executor / workflow-def invariants (K-024 U2, `docs/archive/plans/m3-process-flow.md` §3.3):**
  - A `human` or `wait` step **must** declare `config.waitsForHuman: true` — enforced at
    **publish** (`services._validate_def_spec`, `WorkflowDefSpecError`). Without it the step never
    reaches the executor's OUTCOME B park and self-loops until the step budget fails the run.
  - A `cmp`-family transition guard (`kind` ∈ `cmp|all|any|not`) is **structurally validated at
    publish** (`guards.validate_cmp` → `WorkflowConfigError`): a typo'd `op` is an authoring error
    at seed time, not a live run that parks forever. Path roots are **strict at publish and total
    at drive** (an unwhitelisted root is rejected on publish, but only "missing" ⇒ `False` when a
    run evaluates it). A guard with **no `kind`** (e.g. `{"expr":"x>0"}`) or one that does not
    normalize to a dict is *not a declaration this validator owns* and publishes unchanged.
  - A def **must declare ≥ 1 transition** — enforced at publish (K-024 U4b, O-6). `_PUBLISH_CYPHER`
    ends in `UNWIND $transitions`, which collapses the row stream *after* the def, its `Step`s and
    the `START` edge are MERGEd; `publish_def` then indexes `result_set[0]` ⇒ `IndexError` ⇒ 500,
    and because publish is `MERGE … ON CREATE SET` the corrected retry on the same `(key, version)`
    is a silent no-op on the half-written def — that version is **unrepairable**. A terminal
    outcome is a step with no *outgoing* transition, never a def with none.
  - All three run **last** in `_validate_def_spec`, after the key-uniqueness / start-count /
    dangling-endpoint checks, so an older check keeps failing for its own reason. `config`/`guard`
    are **normalized first** (`_normalize_opaque`): the REST front door types them `str` while
    service/MCP callers pass dicts, and a validator that skipped strings would let every
    REST-published def escape both invariants silently.
  - A `decision` step has **no side effect** — its semantics are entirely its outgoing guards; with
    no outgoing transitions it is a terminal outcome node (run ends `done`).
  - `wait` is **signal-driven, not timer-driven** (there is no scheduler — proposed K-028), and is
    mechanically identical to `human` to the engine; only the `awaiting.kind` string differs.
  - ⚠️ **Not enforced (n-3):** a `decision` step whose outgoing transitions are *all* conditional
    and which does not declare `waitsForHuman` **self-loops to budget exhaustion**. The symmetric
    invariant would retro-reject existing fixtures; filed as a proposed hardening with K-029.
  - `prompt` / `tool` / `message` (and any unknown type) **raise** `NotImplementedError` from
    `executor._execute_step` — the documented typed-handler seam (D-E). The M-1 fault net stamps
    `fail_run` and re-raises. `agent` **without a wired LLM** deliberately keeps returning the empty
    stub: it is the affordance the whole offline executor test estate rests on.
  - `_drive_loop` is **SHA-locked** (`71055f756280`, see `docs/archive/plans/m3-process-flow.md` §3.1) —
    `_execute_step`, `_select_transition`, `_trace_step` and `resume` sit **outside** the lock.

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
