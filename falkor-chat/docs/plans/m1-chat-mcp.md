# Plan — M1 design change: add a Chat MCP transport (mentions + read-cursors)

> Status: ready to implement · Author: architect · Date: 2026-06-21
> Supersedes nothing; **extends** `docs/DESIGN.md §14` (M1 — Chat core).
> Inspired by `../../kiro/DESIGN.md` ("Chat MCP Server"). Decisions taken with the user:
> **A** (MCP as an additive second transport on the shared service layer), **2a** (chat-only —
> no Tasks/agent-presence/`get_room_state`; coordination is deferred to M3), **Streamable-HTTP**
> (MCP mounted on the M1 FastAPI app), and **incorporate `mentions` + read-cursors into M1 now**.

---

## 1. Goal & scope

**Goal.** Give AI agents a first-class, tool-call front door to falkor-chat's chat data —
`send_message` and `read_messages` exposed over **MCP (Streamable-HTTP)**, mounted on the same
FastAPI process and sitting on the *same* `services.py`/`repository.py` layer as the M1 REST API.
Fold the two capabilities those tools require — **@mentions of participants** and **per-agent
read-cursors** — into the M1 schema and query library.

**In scope**
- A new MCP server surface (`server/falkorchat/mcp.py`) mounted on the M1 FastAPI app.
- MCP tools `send_message` and `read_messages` (+ a thin `create_thread` so an agent client is
  self-sufficient — see §6).
- Schema additions: a participant-mention edge and a read-cursor node, with indexes/constraints.
- Query-library additions, and an extension of the two §4 message write paths to record mentions
  atomically.
- Service- and repository-layer methods backing the tools.

**Out of scope (explicitly)**
- `get_room_state`, Tasks, agent presence/status, the task state machine (kiro's coordination
  layer). Per decision **2a**, coordination lands later as an **M3 `WorkflowDef` of `kind:'process'`**,
  reusing the workflow engine — *not* a parallel flat `Task` node. See §10.
- Real auth. MCP keeps M1's single hardcoded tenant; the actor seam is §14.3's `get_context`.
- stdio transport. Streamable-HTTP only for now (stdio can be added later as a second entry point).
- Real-time push (deferred to M2, same as REST).

---

## 2. Context & findings (what's already there)

- **M0 complete, no app code yet.** Only `docs/`, `scripts/`, `kaizen/`. The `server/` + `web/`
  trees in `DESIGN.md §14.5` are still proposals. This change lands *before* any M1 code, so there
  is **no migration** — schema additions just join `bootstrap_schema.sh` before the first build.
- **The transport seam already anticipates this.** `DESIGN.md §14.2`: *"`services.py` owns the
  invariants … `api.py` is the only layer that changes if the transport is ever revisited … a
  non-browser consumer can be bolted onto the same `Service`."* An MCP tool handler is exactly that
  — another caller of `services.post_message(...)`. **No AGENTS.md locked decision is reopened.**
- **`MENTIONS` is already taken.** `QUERIES.md §6` uses `(:Message)-[:MENTIONS]->(:Entity)` for
  GraphRAG entity co-occurrence (`QUERIES.md:225`). Participant mentions are a *different* concept
  and **must use a different relationship type** (this plan: `MENTIONS_MEMBER`).
- **Member identity is split.** `User` has `userId`, `Agent` has `agentId`; channel membership uses
  `coalesce(u.userId, u.agentId)` (`QUERIES.md:53`). The §4 write paths match the author with
  `MATCH (author {userId: $authorId})` and a comment "works for User or Agent" — but an `Agent`
  node has no `userId`, so that pattern only matches `User` authors today. The mention-matching
  query must handle both labels (see §5.3 and the open question in §10).
- **Write atomicity is locked.** AGENTS.md: *"All writes that touch HEAD/TAIL must be a single
  `GRAPH.QUERY`."* Therefore mention edges are written **inside** the existing §4 write paths, not
  in a follow-up query.
- **Query authoring is graph-dba's job.** K-001 (`list_channels`) set the precedent: new Cypher is
  authored + `GRAPH.PROFILE`-verified + asserted in `test_queries.sh` by graph-dba *before* the
  repository method is built (`kaizen/history.md` 2026-06-11). This plan follows that order.
- **MCP-on-FastAPI is verified viable.** The official MCP Python SDK / FastMCP mounts a
  Streamable-HTTP server into an existing FastAPI app via `app.mount(...)`, **but** the MCP app's
  lifespan must be passed to FastAPI (`FastAPI(lifespan=mcp_app.lifespan)`) or the session manager
  never initialises. (python-sdk issue #1367; FastMCP HTTP deployment docs.)

---

## 3. Design & rationale

### 3.1 The shape (one process, two front doors)

```
browser ── REST/JSON ──┐
                       ├─▶ services.py ─▶ repository.py ─▶ FalkorDB
agents  ── MCP/HTTP ───┘   (invariants live here; both front doors call the SAME methods)
```

`mcp.py` is a *peer of `api.py`* — a thin transport adapter that translates MCP tool calls into
the same service methods the REST router calls. Both are mounted on one FastAPI/uvicorn process.

**Why MCP, not "agents call the REST API":** an LLM agent's native interface is tool calls; making
every agent hand-roll an HTTP client is the same "bridge tax" §14.1 rejected gRPC-Web for. MCP is
the idiomatic agent front door, and Streamable-HTTP lets it co-host with the REST server (no second
process to run, no schema change). *Rejected:* stdio (kiro's choice) — fine for a single local
client, but the kiro design has *multiple* agents sharing one chat; a co-hosted HTTP server is the
natural fit and reuses the M1 server you're already building.

### 3.2 Participant mentions — `MENTIONS_MEMBER`

- New edge: `(:Message)-[:MENTIONS_MEMBER]->(member)` where `member` is a `User` **or** `Agent`.
- Distinct from the existing `(:Message)-[:MENTIONS]->(:Entity)` (GraphRAG). Do **not** reuse it.
- Written **inside** both §4 write paths (atomicity rule), one edge per `mentions[]` entry.
- *Rejected:* an inline `mentions` string-list property on `Message`. FalkorDB stores scalar lists,
  so it's *possible*, but an edge is queryable both ways ("who was mentioned" and "who mentioned
  me"), which is exactly what `read_messages`' mention-prioritisation and any future
  notification/`get_room_state` needs. The edge also matches the schema's relationship grain.

### 3.3 Read-cursors — `ReadCursor` node

Kiro's `read_messages(... since?)` means "messages since I last read." falkor-chat has no per-agent
read state. Model it as a constraint-backed node (every `MERGE` needs a uniqueness constraint —
AGENTS.md):

```
(:ReadCursor {cursorId, memberId, threadId, lastReadAt})
(member)-[:HAS_CURSOR]->(:ReadCursor)
cursorId = "{memberId}:{threadId}"   // deterministic → MERGE-able, unique
```

- `lastReadAt` (a timestamp) is the cursor — `read_messages` returns `m.createdAt > lastReadAt`.
  No edge to the last-read `Message` (avoids relink churn; a timestamp compare against the indexed
  `Message.createdAt` is enough).
- *Rejected:* a `LAST_READ` edge from member→Message. It re-links on every read (churn) and forces
  a Message→Thread back-walk to know which thread it belongs to; a `threadId` property on a cursor
  node is cheaper and directly index-addressable.
- **Cursor advance makes `read_messages` a write.** Two modes, made explicit in the tool:
  - explicit `since=<ts>` → pure read (`GRAPH.RO_QUERY`), cursor untouched;
  - `advance=true` (default for the MCP tool) → MERGE/SET the cursor (`GRAPH.QUERY`, RW).
- *Room-wide vs thread-scoped:* with `re`/`thread_id` → scoped to that thread. Without it →
  workspace-wide "since" read anchored on the `Message.createdAt` index (see §5.3 and the TIMEOUT
  risk in §10).

### 3.4 Tool → service → query mapping

| MCP tool | kiro signature | Service method | Query (QUERIES.md) |
|---|---|---|---|
| `send_message` | `send_message(from, body, mentions[], re)` | `post_message(ctx, thread_id, text, mentions)` | §4 first/subsequent **(extended with mentions)** |
| `read_messages` | `read_messages(agent_id, since?, limit?, re?)` | `read_messages(ctx, thread_id?, since?, limit, advance)` | new §9 (thread-scoped + room-wide variants) |
| `create_thread` *(added)* | — | `create_thread(ctx, channel_id, title)` | §3 create a thread (existing) |

Mapping notes:
- `from` / `agent_id` → the **actor**, from the `get_context` seam (§14.3), not trusted from the
  client in a future-auth world. For M1's single tenant it is the configured actor. See §10.
- `re` → `thread_id`. The thread must already exist; `create_thread` is exposed so an agent can make
  one without dropping to REST. (Channel creation stays REST-only for M1 — agents post into existing
  channels.)
- `mentions[]` → list of `memberId`s (each a `userId` or `agentId`).

---

## 4. Schema deltas (bootstrap)

Add to `scripts/bootstrap_schema.sh` `bootstrap_workspace()` — **index before constraint**, per the
live-verified ordering rule:

```bash
# ── identity anchors (add alongside the existing ones) ──
echo "[index] ReadCursor.cursorId"
gquery "$g" "CREATE INDEX FOR (n:ReadCursor) ON (n.cursorId)"

# ── uniqueness constraints (add alongside the existing ones) ──
echo "[constraint] ReadCursor unique {cursorId}"
gconstraint "$g" UNIQUE NODE ReadCursor PROPERTIES 1 cursorId
```

- No index is needed for the `MENTIONS_MEMBER` edge itself (FalkorDB has no edge-property index
  here; traversal from an indexed `Message`/member anchor is enough).
- **RAM impact (AGENTS.md rule #6):** one extra `ReadCursor` node per *(member, thread)* the member
  has read, one `MENTIONS_MEMBER` edge per mention, and one new range index + one constraint per
  workspace. Cursors are the growth term — bound it (e.g. only threads an agent actually reads) and
  call it out in the §11 capacity sketch. No new vector dimension, so no embedding-RAM change.

---

## 5. Query-library changes (`docs/QUERIES.md`) — graph-dba authors & verifies

> These are **candidate** queries expressing intent + constraints. graph-dba authors the final
> Cypher, confirms `Node By Index Scan` via `GRAPH.PROFILE`, and adds assertions to
> `test_queries.sh` **before** the repository methods are built — mirroring the K-001 workflow.

### 5.1 Extend §4 write paths with mentions (single GRAPH.QUERY — atomicity)

Append to **both** "first message" and "subsequent message" write paths, after `(m)` is created:

```cypher
// ... existing write path creates (m), HEAD/TAIL/NEXT, POSTED_BY, sets t.updatedAt ...
WITH m
UNWIND $mentions AS mid                       // $mentions defaults to []
MATCH (mem) WHERE mem.userId = mid OR mem.agentId = mid
CREATE (m)-[:MENTIONS_MEMBER]->(mem)
```

graph-dba to confirm: with `$mentions = []` the `UNWIND` is a no-op and the write path is unchanged
(regression-safe for non-mention posts); and that the `userId OR agentId` match is acceptable
(see the open question in §10 — may become two indexed matches or a typed `[{id,kind}]` param).

### 5.2 New §9.1 — read a thread since a cursor/timestamp (thread-scoped)

```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
WHERE m.createdAt > $since                     // $since = cursor.lastReadAt, or 0 for "all"
MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me {userId: $meId})   // mention flag for prioritisation
WITH m, author, count(me) > 0 AS isMention
RETURN m.msgId, m.text, m.role, m.createdAt,
       author.userId, labels(author) AS authorType, isMention
ORDER BY isMention DESC, m.createdAt
LIMIT $limit
```

### 5.3 New §9.2 — read workspace-wide since a timestamp (room-wide, no `re`)

```cypher
MATCH (m:Message)
WHERE m.createdAt > $since                      // anchors on Message.createdAt index
MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me {userId: $meId})
WITH m, author, count(me) > 0 AS isMention
RETURN m.msgId, m.text, m.role, m.createdAt,
       author.userId, labels(author) AS authorType, isMention
ORDER BY isMention DESC, m.createdAt
LIMIT $limit
```

graph-dba: confirm this is a `Node By Index Scan` on `Message.createdAt` (not `NodeByLabelScan`),
and watch the default `TIMEOUT 1000ms` on large workspaces (§10 risk).

### 5.4 New §9.3 — advance a read-cursor (RW; only in `advance` mode)

```cypher
MATCH (mem) WHERE mem.userId = $meId OR mem.agentId = $meId
MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId: $cursorId})
ON CREATE SET rc.memberId = $meId, rc.threadId = $threadId
SET rc.lastReadAt = CASE WHEN $now > coalesce(rc.lastReadAt, 0) THEN $now
                        ELSE rc.lastReadAt END   // monotonic — never moves backward
RETURN rc.lastReadAt
```

`cursorId = "{meId}:{threadId}"` (per-thread only — Q#3 resolved; no room-wide cursor). MERGE is
backed by the new `ReadCursor.cursorId` constraint (§4).

**Monotonic-advance guard (design decision, not left to the implementer).** The cursor must never
move backward, so re-reading with a stale `$now` is a no-op. The guard is expressed in the query
above via the `CASE` on `lastReadAt`; the **service** (`read_messages`, `advance=true`) is the
authoritative owner of `$now` (server clock, not client-supplied) and remains free to short-circuit
before the write. graph-dba to confirm the `CASE`/`coalesce` form runs on this build; if not, move
the guard entirely into `services.read_messages` (read cursor → compare → conditionally issue §9.3).
Either way, the §8 "re-advancing to an older ts doesn't move it backward" assertion is the contract.

### 5.5 New §9.4 — read a cursor (to compute `since` when not supplied)

Intent only — graph-dba authors the final Cypher (the earlier draft used invalid `MATCH … AS rc`
aliasing). Match a single `ReadCursor` by its unique `cursorId` and return `lastReadAt`; when no
cursor exists yet, the service treats `since` as `0`/epoch (§8 edge case). Backed by the
`ReadCursor.cursorId` index (§4), so this is a single indexed point-lookup.

---

## 6. Application-layer changes (`server/`)

Extends the `DESIGN.md §14.5` layout. New file in **bold**.

```
server/falkorchat/
  config.py        WS_ID, USER_ID  (+ MCP actor mapping — see §10)
  db.py            falkordb-py conn, select_graph (unchanged)
  repository.py    + post_message(mentions=…), read_thread_since, read_ws_since,
                   advance_cursor, get_cursor, create_thread
  services.py      + post_message(mentions), read_messages(thread_id?, since?, advance),
                   create_thread; mention validation; cursorId construction
  schemas.py       + MentionList, ReadMessagesResult, etc.
  api.py           REST router (unchanged by this plan; mentions optional on POST message)
  mcp.py           ← NEW: FastMCP server; tools send_message, read_messages, create_thread
  app.py           build FastAPI, mount REST router + MCP app, wire MCP lifespan
  tests/           + test_mcp.py; extend test_repository/test_services/test_api
  pyproject.toml   + mcp (the official Python SDK / fastmcp)
```

### `mcp.py` (sketch — transport adapter only, no business logic)

```python
from mcp.server.fastmcp import FastMCP
from .services import Services
from .api import get_context           # the §14.3 seam, reused

mcp = FastMCP("falkor-chat")

@mcp.tool()
def send_message(body: str, re: str, mentions: list[str] = [], frm: str | None = None) -> dict:
    ctx = get_context()                # actor from the seam (frm reserved for future auth)
    return svc.post_message(ctx, thread_id=re, text=body, mentions=mentions)

@mcp.tool()
def read_messages(re: str | None = None, since: str | None = None,
                  limit: int = 50, advance: bool = True) -> list[dict]:
    ctx = get_context()
    return svc.read_messages(ctx, thread_id=re, since=since, limit=limit, advance=advance)

@mcp.tool()
def create_thread(channel_id: str, title: str) -> dict:
    ctx = get_context()
    return svc.create_thread(ctx, channel_id=channel_id, title=title)
```

### `app.py` (the mount — note the lifespan gotcha from §2)

```python
mcp_app = mcp.streamable_http_app()
app = FastAPI(lifespan=mcp_app.lifespan)   # MUST forward MCP lifespan or session mgr won't init
app.include_router(rest_router)
app.mount("/mcp", mcp_app)                 # agents connect at /mcp
```

`services.py` keeps **all** invariants: it picks the first-vs-subsequent §4 write variant (now also
passing `mentions`), builds `cursorId`, decides RO vs RW for `read_messages`, and validates that each
mention resolves to a known member. `mcp.py` and `api.py` stay dumb adapters.

---

## 7. Step-by-step implementation (sequenced; tree stays buildable)

1. **Schema (graph-dba).** Add the `ReadCursor` index + constraint to `bootstrap_schema.sh` (§4).
   Re-run bootstrap on `ws:test`. *Done when:* `CALL db.indexes()`/`db.constraints()` show them.
2. **Queries (graph-dba).** Author + `GRAPH.PROFILE`-verify the §5 queries (mention-extended write
   paths, §9.1–9.4); add them to `QUERIES.md` (new §9, edits to §4); add assertions to
   `test_queries.sh`. *Done when:* suite green at a new baseline (was 67/67) and PROFILE shows index
   scans for §9.2. **This is the prerequisite gate — repository work starts only after it lands.**
3. **Repository (coder/tdd-engineer).** Add the new methods over `falkordb-py`, each 1:1 with a §9
   query / extended §4 write; integration tests against an isolated `ws:test` graph (the
   `test_queries.sh` approach). *Done when:* method tests green; mentions persist; cursor advances.
4. **Services.** Append-variant dispatch already exists in the M1 plan; extend `post_message` to
   thread `mentions` through, add `read_messages` (RO/RW dispatch on `advance`), `create_thread`,
   `cursorId` construction, mention validation. Fake-repo unit tests + a few live checks.
5. **MCP transport.** Add `mcp.py` + the `app.py` mount with the lifespan wiring. *Done when:* an
   MCP client (or `mcp` SDK test client) lists the three tools and a `send_message`→`read_messages`
   round-trip returns the posted message with its mention flag.
6. **REST parity (small).** Allow optional `mentions` on `POST /threads/{tid}/messages` so both
   front doors are symmetric. *Done when:* `TestClient` posts with mentions and reads them back.
7. **Docs (same change, per repo rule).** Update `DESIGN.md` (§14.4 surface table + a new §15 "MCP
   transport" or §14.x), `QUERIES.md` (done in step 2), `AGENTS.md` (message write-path invariants
   now include `MENTIONS_MEMBER`; new key facts), `README.md` (roadmap/layout), and
   `kaizen/{plan,history}.md`.

---

## 8. Test strategy

- **Query suite (`test_queries.sh`, live FalkorDB) — first, TDD gate.** New assertions: a message
  posted with `$mentions=[a,b]` has exactly two `MENTIONS_MEMBER` edges; a post with `$mentions=[]`
  is byte-identical in effect to today's write path (regression guard); §9.2 returns only messages
  after `$since`; cursor advance is idempotent (re-advancing to an older ts doesn't move it
  backward — service-enforced); the §8 `assert_index_scan` pair for §9.2 on `Message.createdAt`.
- **Repository (integration, `ws:test`).** Each new method against the live graph; mention edges,
  cursor MERGE/SET, RO vs RW paths.
- **Services (unit + a few live).** Mention validation (unknown member → error, not a dangling
  edge), RO/RW dispatch on `advance`, `cursorId` construction, first-vs-subsequent dispatch still
  correct when mentions are present.
- **MCP (`mcp` SDK in-memory client).** Tool discovery; `send_message`→`read_messages` round-trip;
  mention-prioritised ordering (`isMention DESC`); `create_thread` then post into it.
- **API (`TestClient`).** REST mention parity.
- **Edge cases:** empty `mentions`; mention of a non-member; `read_messages` with no cursor yet
  (treat `since` as 0/epoch); room-wide vs thread-scoped; self-mention; duplicate mentions in one
  call (dedupe to one edge).

---

## 9. Risks & open questions

**Risks (with mitigations)**
- **`MENTIONS` name collision** with the GraphRAG entity edge → mitigated by the distinct
  `MENTIONS_MEMBER` type. Implementer must not reuse `MENTIONS`.
- **Atomicity regression.** Mentions must ride inside the single §4 write `GRAPH.QUERY`; a second
  query would violate the locked HEAD/TAIL atomicity rule. Covered by the `$mentions=[]`
  no-op-equivalence test.
- **`read_messages` is RW in `advance` mode** → cannot use `GRAPH.RO_QUERY`/replicas when advancing.
  Document it; offer explicit `since` for pure reads. Revisit for M4 replica routing.
- **TIMEOUT 1000ms** (live default) on room-wide §9.2 over a large workspace → keep `limit` modest,
  PROFILE it, and note in §11 capacity sketch; consider a bounded `since` window.
- **RAM growth from cursors** (one node per member×thread read) → bound and document (rule #6).
- **MCP lifespan wiring** (python-sdk #1367) → the `FastAPI(lifespan=mcp_app.lifespan)` pattern is
  mandatory; an integration test that actually calls a tool will catch a missing session manager.

**Resolved decisions (locked 2026-06-21)**
1. **MCP actor identity — RESOLVED: option (a).** Ignore client-supplied `from`; the actor is
   `get_context()`'s configured actor (M1-consistent). Real per-client identity arrives with the
   §14.3 auth replacement. The `frm` tool param is reserved/ignored for now.
3. **Room-wide cursor semantics — RESOLVED: per-thread cursors only.** No `"{meId}:*"` room cursor
   in M1; `read_messages` without `re` requires an explicit `since` (or defaults to epoch/0).

**Open questions still live**
2. **Member-match in the mention/cursor queries.** `WHERE mem.userId = mid OR mem.agentId = mid`
   may not be index-friendly. Alternatives graph-dba should weigh: pass typed mentions
   `[{id, kind}]` and do two label-specific indexed matches, or `UNION`. **Decide during step 2
   (graph-dba), guided by `GRAPH.PROFILE`.**
4. **Expose `create_channel` over MCP too?** Out of scope here (agents post into existing channels);
   flag if the kiro flow needs agents to create channels autonomously.

---

## 10. Why coordination is NOT here (the 2a decision, recorded)

Kiro's `get_room_state`/Tasks/agent-presence were evaluated and deferred. falkor-chat's **M3
workflow engine** (`DESIGN.md §6`) is the stronger home for coordination: process-as-data
(`Step`+`TRANSITION` guards, versioned, snapshotted), a queryable `StepRun` audit trail, atomic
status transitions, and causal chat linkage (`TRIGGERED_BY`/`EMITTED`) — versus kiro's flat `Task`
node whose history is split across YAML+git+chat. Building a flat `Task` now would duplicate M3 and
require a later migration, fighting the "single store" philosophy. **Decision: coordination lands as
an M3 `WorkflowDef` of `kind:'process'` ("task lifecycle"); until then, kiro keeps task state in its
own `meta/*.yaml`+git as its design already prescribes.** Agent *presence* (status/current_task) is
not a workflow concept and is left out of scope entirely for now.

---

## Ready to implement

Build order, gated on the schema/query work first (K-001 precedent):

1. **graph-dba** — add `ReadCursor` index+constraint to `bootstrap_schema.sh`; author & PROFILE the
   §5 queries; extend `QUERIES.md` (§4 mentions, new §9) and `test_queries.sh`. **Gate: suite green
   + index scans confirmed** before anything below.
2. **coder/tdd-engineer** — repository methods → services (mention threading, RO/RW read dispatch,
   cursors, `create_thread`) → `mcp.py` + `app.py` mount (lifespan!) → REST mention parity.
3. **Docs in the same change** — `DESIGN.md`, `AGENTS.md` (write-path invariant now includes
   `MENTIONS_MEMBER`), `README.md`, `kaizen/{plan,history}.md`.

Decisions are locked (Q#1 MCP actor = `get_context()`; Q#3 per-thread cursors only). The only call
left is **Q#2 (member-match index strategy)** — graph-dba's, made against a real `GRAPH.PROFILE`
during step 1. Q#4 (`create_channel` over MCP) stays deferred.

---

## Appendix A — MCP client connection contract (consumer side)

How an agent connects to this server once it's running. falkor-chat is the **backend**; this is the
config a consumer (a Kiro/Claude/OpenCode agent) puts in its own MCP-server list. It differs from
`../../kiro/DESIGN.md`'s block, which assumed **stdio** — we chose **Streamable-HTTP**, so the
contract is a URL, not a subprocess command.

**Endpoint:** `http://<host>:<port>/mcp` (the M1 FastAPI server; default `http://localhost:8000/mcp`
once `app.mount("/mcp", mcp_app)` is in place — confirm the port the uvicorn run command uses).

**Transport:** `streamable-http` (a single running server many agents share; no per-client process).

**Tools exposed:** `send_message`, `read_messages`, `create_thread` (signatures in §3.4 / §6).

**Example client config (HTTP form):**
```json
{
  "mcpServers": {
    "falkor-chat": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```
> Compare kiro's stdio block (`command: python3, args: [-m, chat_mcp]`): with HTTP there is no
> `command`/`args`/`env` — the server runs independently (it's the M1 app) and clients just point at
> the URL. The exact config key shape varies per harness (Claude Code / Kiro / OpenCode) — verify
> against each tool's MCP-server schema; the **transport + URL** are the portable part.

**Actor identity (M1):** the server ignores any client-supplied `from`; every call is attributed to
the single configured actor via the §14.3 `get_context()` seam (resolved Q#1). When real auth lands,
per-client identity (token → user + workspace) replaces only `get_context` — no client-config change
beyond adding credentials. **Until then this endpoint is unauthenticated; bind it to localhost / a
trusted network only.**

**System-prompt pattern for a consuming agent** (adapted from kiro §"Kiro Agent Config"):
> At the start of every turn, call `read_messages` to catch up. Use `send_message` to speak; use
> `mentions` to address specific members. Post into an existing thread via its `re`/`thread_id`, or
> `create_thread` first.

---

## Appendix B — ADR: coordination (Tasks/presence) is deferred to M3, not modelled now

**Status:** Accepted (2026-06-21) · **Context:** decision 2a, recorded here in full because the
reasoning is the main thing future-you will want when M3 arrives.

**Decision.** falkor-chat does **not** adopt kiro's coordination layer (`Task`, agent presence,
`get_room_state`, the task state machine) in the MCP work. Coordination lands later as an **M3
`WorkflowDef` of `kind:'process'`** ("task lifecycle"), reusing the workflow engine (`DESIGN.md §6`).
Until then, a consuming agent team keeps task state in its **own** store (kiro's `meta/*.yaml` + git,
as kiro's design already prescribes).

**Why M3's engine is the stronger home for coordination than a flat `Task` node:**

| Aspect | kiro flat `Task` | falkor-chat M3 engine | Why M3 wins |
|---|---|---|---|
| Process representation | FSM hardcoded in prompts/diagram | `Step` + `TRANSITION {on,guard,order}`, versioned, snapshotted per run | change the process as data, not code; run versions concurrently; old runs keep their definition |
| Auditability | only current `status` + `updated_at` | `StepRun` chain (`NEXT`) — immutable, queryable trace of what ran, when, in/out | "how did this reach done?" is one query, not archaeology across yaml+git+chat |
| Branching | free-text chat ("owner signals done") | `TRANSITION.guard` over run `ctx`; `decision`/`wait`/`human` steps | enforce "done only when all reviewers approved" instead of judgment calls |
| Chat linkage | `Message-[:ABOUT]->Task` (topical tag) | `WorkflowRun-[:TRIGGERED_BY]->Message`, `StepRun-[:EMITTED]->Message` | causal, bidirectional provenance |
| Concurrency/claim | kiro's own *open question* (yaml lock? `claim_task`?) | status-as-indexed-property + per-`GRAPH.QUERY` atomicity | the engine arbitrates transitions atomically |
| Source of truth | state split across yaml + git + FalkorDB, synced by convention | execution state in the graph beside the chat | no split-brain; matches the "single store" philosophy |
| Generality | only a dev-team task shape | one model spans `kind: conversation \| process` | same engine for agent flows *and* business processes |

**What kiro's lightweight model is genuinely better at** (so we keep it for the interim): nothing to
build (yaml + prompts ship today), human-legible git-diffable state, and a deliberately loose
self-organizing flow. Those are real wins for a single agent team *right now* — which is exactly why
the interim answer is "keep it in kiro's repo," not "build a weak Task node here."

**Consequences.** (1) `get_room_state` is **not** an MCP tool in this work. (2) Agent *presence*
(`status`/`current_task`) is out of scope entirely — it isn't a workflow concept and M3 doesn't model
it; revisit as a separate small concern if needed. (3) When M3 builds the `kind:'process'` task
lifecycle, a `get_room_state`-equivalent becomes a read over `WorkflowRun`/`StepRun` — no migration,
because no parallel `Task` node was ever created. **This is the payoff of deferring: we avoid building
something we'd have to migrate away from.**

**Rejected alternative — build a flat `Task` node now (2b-lite).** Gives kiro's weaker model *and*
incurs a later reconcile/migrate into M3 (duplication + drift) — the precise failure mode the
single-store philosophy exists to prevent.
</content>
</invoke>
