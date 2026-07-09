# K-013 — M2 AI agent participant with `EMITTED` provenance: verified query changes (graph-dba deliverable)

**Date:** 2026-07-08 · **Author:** graph-dba
**Verified against:** `falkordb/falkordb:edge` (module `999999`), Redis 8.2.2, container `falkordb-dev`,
throwaway graph `ws:emitverify` (deleted after verification) + the suite's isolated `ws:test`.

Every query below was **executed against the live instance**; evidence excerpts are pasted verbatim.
This is the graph-dba gate for K-013: it defines the `EMITTED` edge, verifies the write/read, and
raises the shell suite. The canonical query bodies now live in **`docs/QUERIES.md` §10** (single
source of truth); the bodies reproduced here are the point-in-time verification record. **This gate
does not write the Python responder** — that is the tdd-engineer step (handoff at the end).

---

## 1. The `EMITTED` edge shape (decided)

| Aspect | Decision | Why |
|---|---|---|
| **Direction / endpoints** | `(answer:Message)-[:EMITTED]->(seed:Message)`, answer → seed | The hot query is "given an agent answer, what provenance did it cite?" — anchor on the answer's `msgId` (indexed point lookup) and expand outward to ≤k seeds. Matches the existing `REPLY_TO` convention (subject message → referenced message); `MENTIONS_MEMBER` and `MENTIONS` also fan out *from* the message. |
| **Properties** | `score` (float, §6 cosine distance of the seed at answer time) + `rank` (int, 0-based position in the ranked seed list) | Point-in-time snapshot. Retrieval is non-deterministic as the graph grows, so the score/rank *at answer time* is the real provenance value — you can rank an answer's citations without re-running retrieval. Kept lean: no edge `createdAt` (the answer's own `createdAt` is one hop away and shared by all its edges). |
| **Index / constraint** | **None** | Endpoints are `Message` nodes already carrying the `msgId` range index + uniqueness constraint. FalkorDB traverses the typed `EMITTED` edge from the anchored answer via its adjacency matrix — no relationship-property index needed. Uniqueness is guaranteed structurally by the write (see §2), so no relationship constraint (which would also require a supporting index). |
| **Distinctness** | `EMITTED` is a **third, new** edge type | Not to be conflated with `MENTIONS_MEMBER`→`User`/`Agent` (participants, §4) or `MENTIONS`→`Entity` (GraphRAG co-occurrence, §6). |

Reverse query ("which answers cited this seed?") is the same edge type traversed inbound, anchored
on the seed's `msgId` — also verified (§10.3 / item 5 below).

---

## 2. Atomicity decision: `EMITTED` rides **inside** the §4 write

**Decision: the `EMITTED` edges are created in the same `GRAPH.QUERY` as the answer's §4 write,
inside the guarded `FOREACH`, exactly like `MENTIONS_MEMBER`** — *not* as a follow-up `link_emitted`
query.

Justification against the repo's atomicity rule and the `dupMsg` idempotency contract:

- **The atomicity rule is satisfied, not stretched.** The rule is "the HEAD/TAIL relink must be one
  `GRAPH.QUERY`". `EMITTED` riding inside the same guarded `FOREACH` is strictly compatible and
  mirrors the already-proven `MENTIONS_MEMBER` idiom.
- **Exactly-once provenance, gated by the same `dupMsg` check.** A retried responder replays the
  same server-minted `msgId` → the guard sees `dup IS NOT NULL` → `ok=false` → the entire `FOREACH`
  (message `CREATE` + HEAD/TAIL + `POSTED_BY` + `MENTIONS_MEMBER` + `EMITTED`) is skipped.
  Provenance is therefore written **exactly once**, atomically with the message. Verified: replaying
  the `ag1` write left the `EMITTED` count at 2, not 4 (item 4).
- **No torn write.** A follow-up `link_emitted` write would (a) double-write on retry unless
  separately made idempotent (a relationship `MERGE` or existence guard), and (b) leave an "answer
  with no provenance" state if the responder crashed between the two queries. Riding inside the guard
  makes "message + provenance" one all-or-nothing unit and needs **no separate idempotency
  mechanism**. This is why `EMITTED` needs no relationship constraint.
- **The "message readable before provenance" tolerance is not needed** here (unlike §6 embeddings,
  which are legitimately async). Because provenance is cheap and comes from the same retrieval call
  that produced the answer, there is no reason to decouple it — and decoupling would only reintroduce
  the torn-write risk.

Consequence for modeling: the answer write is the §4 **subsequent** path plus one added guarded
`UNWIND`/`FOREACH` block for seeds. `$seedIds = []` is a verified no-op (the `CASE` guard keeps the
write alive), so the same query shape serves non-provenance writes too.

---

## 3. Build quirks that shaped the query (live-verified, new)

- **A map-projection cannot be a `CREATE` endpoint.** `CREATE (m)-[:EMITTED {...}]->(rec.node)`
  where `rec` is a map with a `node` field **errors** (`Invalid input '.': expected a label, '{', a
  parameter or ')'`). The endpoint of a `CREATE` relationship must be a **bound node variable**.
  → Seeds are collected as **nodes** (`collect(DISTINCT s)`), and per-edge props are pulled from
  **map parameters keyed by the node's own `msgId`**: `$scoreBy[seed.msgId]`, `$rankBy[seed.msgId]`.
- **Dynamic map-parameter indexing by a node property works** (`$scoreBy[seed.msgId]` inside a
  `FOREACH`) — this is what makes per-edge properties possible without a map-list.
- **Two sequential guarded `UNWIND`s** (mentions, then seeds) each `collect` back to one row before
  the next expands — **no row multiplication**. Verified (the write reports the exact expected
  relationship counts).

These are folded into `claude/graph-dba/falkordb-quirks.md`.

---

## 4. Verified write — evidence

Scratch structure: `Channel ch1 → Thread th1 (HEAD s1 → s2 → s3 TAIL)`, `User u1`, `Agent bot1`.
Agent `bot1` answers into `th1` (subsequent path), `mentions=['u1']`, cites `s1,s2,s3` + unknown
`zz`.

```
CYPHER mentions=['u1'] seedIds=['s1','s2','s3','zz']
       scoreBy={s1:0.0,s2:0.006,s3:0.5} rankBy={s1:0,s2:1,s3:2}
=> status: w=true auth=true
   Nodes created: 1   Relationships created: 7   Relationships deleted: 1
   (7 = NEXT + TAIL + POSTED_BY + MENTIONS_MEMBER + EMITTED×3)
```

Provenance read back (forward, ordered by rank):

```
assistant  s1  seed one    0      0
assistant  s2  seed two    0.006  1
assistant  s3  seed three  0.5    2
```

- Role reads **`assistant`** and author is an **`Agent`** — the K-007 authorship invariant holds.
- Unknown seed `zz` skipped → **exactly 3** `EMITTED` edges (`collect(DISTINCT s)` drops the null).
- Each edge carries its `score` + `rank`.

**Empty-seed no-op** (a normal message): `seedIds=[]` → `w=true`, `Relationships created: 3`
(NEXT + TAIL + POSTED_BY), **0** `EMITTED`. Guard load-bearing, confirmed.

**`dupMsg` exactly-once**: replaying the `ag1`/answer write → `w=false dup=true`, and the `EMITTED`
count stayed **2** (not 4). Provenance written exactly once.

**First-path symmetry**: the same seed block folded into the §4 **first-message** write on a fresh
thread → `w=true hh=false auth=true`, `Relationships created: 5` (HEAD + TAIL + POSTED_BY +
EMITTED×2). The realistic responder path is always subsequent, but both paths carry the block.

---

## 5. Verified reads — evidence

**Forward (hot path), `GRAPH.PROFILE`:**

```
Results
  Sort
    Project
      Conditional Traverse | (a)-[e:EMITTED]->(s:Message)
        Node By Index Scan | (a:Message)            # anchored on Message.msgId — no label scan
```

**Reverse** (`MATCH (a:Message)-[e:EMITTED]->(s:Message {msgId:'s1'})`): returns `ans1, assistant,
score 0, rank 0` — anchored on the seed's `msgId`, `EMITTED` traversed inbound.

**Thread-read integrity**: the canonical §4 thread read of `th1` (which now contains the answer plus
its intra-thread `EMITTED` edges) returned 5 rows in chronological order, the answer with
`role=assistant`/`[Agent]`, **no duplication** — the typed `NEXT` walk does not follow `EMITTED`.

---

## 6. RAM (repo rule 6, mandatory)

`EMITTED` is a **new relationship type** (one GraphBLAS adjacency matrix) plus one edge per cited
seed, each carrying two scalar properties (`score` float + `rank` int).

- Per answer: 1 answer `Message` (already budgeted by §4/§6) + **k `EMITTED` edges** where k = the
  seed count (typically 3–10). An edge + its two scalar props is on the order of tens of bytes;
  at k≈10 that is a few hundred bytes **per answer message**, not per message overall.
- This is **negligible next to the §6 vector line** (~12.5 KB/message at 1024 dims — K-008 item 7).
  Only answer messages carry `EMITTED` edges, and answers are a small fraction of traffic.
- One new relationship-type matrix per workspace; **no new index, no new constraint** → no
  index/constraint RAM. Net: a rounding error against the embedding budget. No change to the DESIGN
  §11 per-workspace sizing rule of thumb.

---

## 7. Test suite raised: 135 → 149, fully green

`scripts/test_queries.sh` gained a `▶ §EMITTED agent answer provenance (K-013)` block (runs on the
isolated `ws:test`). 14 new assertions:

```
✓ §EMITTED agent answer ag1 commits (agent-authored)
✓ §EMITTED ag1 role derived assistant (K-007 invariant)
✓ §EMITTED ag1 author resolves to Agent
✓ §EMITTED provenance read cites m1
✓ §EMITTED provenance read cites m2
✓ §EMITTED ag1 has exactly 2 provenance edges (unknown seed skipped)
✓ §EMITTED edge carries rank + score props
✓ §EMITTED reverse read: m1 cited by ag1
✓ §EMITTED empty seedIds=[] still commits the write
✓ §EMITTED empty seedIds=[] creates zero provenance edges
✓ §EMITTED dupMsg replay no-ops (idempotent)
✓ §EMITTED provenance written exactly once (still 2 after replay)
✓ §EMITTED provenance read uses Message.msgId index          # assert_index_scan = 2 assertions
✓ §EMITTED provenance read uses Message.msgId index (no label scan)
```

Final: **`Results: 149/149 passed`**.

---

## 8. Handoff to tdd-engineer (Python impl — `server/`)

Thin adapter over the verified queries. Layering is locked: Cypher in `repository.py` (1:1 with
`QUERIES.md` §10), orchestration in `services.py`, tenant seam via `config.get_context`.

1. **`repository.post_agent_answer(...)` — 1:1 with `QUERIES.md` §10.1.** A provenance-carrying
   variant of the existing agent-authored §4 subsequent write. Suggested signature:
   ```python
   def post_agent_answer(
       g, *, thread_id: str, msg_id: str, author_id: str, text: str,
       role: str = "assistant", created_at: int,
       mentions: list[str] | None = None,
       seeds: list[SeedProvenance],   # ordered by rank; SeedProvenance = (msg_id, score)
   ) -> WriteStatus: ...
   ```
   - Build the three seed params **from the ordered `seeds` list** so rank is positional:
     `seed_ids = [s.msg_id for s in seeds]`,
     `score_by = {s.msg_id: s.score for s in seeds}`,
     `rank_by  = {s.msg_id: i for i, s in enumerate(seeds)}`.
   - Pass `mentions or []` and `seeds or []`; **`seed_ids == []` is a verified no-op** — the query
     still commits the message. Do not branch the query on empty seeds.
   - Returns the §4 status row `(written, hadHead, dupMsg, authorFound)` — **reuse the existing §4
     dispatch loop verbatim** (zero rows → retry as first-path; `dupMsg=true` → idempotent success;
     `hadHead=true` → re-dispatch; `authorFound=false` → 4xx). Provenance needs no extra dispatch
     logic — it is written or skipped atomically with the message.
   - `role` is still **derived, not trusted** — resolve the author's label via the §2 member-kind
     lookup (`Agent → assistant`); do not let the caller pass an arbitrary role.
   - Write path — **not** routable to a replica.
   - *(Optional, for symmetry)* a first-path variant if the responder ever answers into an empty
     thread; realistically the trigger message is always the HEAD, so subsequent-path suffices.

2. **`repository.read_provenance(msg_id)` — 1:1 with `QUERIES.md` §10.2 (forward).** Returns rows of
   `(seedMsgId, text, role, score, rank)` ordered by `rank`. Read path — `g.ro_query(...)`.

3. **`repository.read_citing_answers(seed_msg_id)` — 1:1 with `QUERIES.md` §10.3 (reverse).** Returns
   `(answerMsgId, role, createdAt, score, rank)`. Read path — `g.ro_query(...)`. (Wire only if the
   API/UI needs an attribution/impact view; the edge supports it regardless.)

4. **Responder orchestration (services.py)** — out of the graph-dba gate, but the shape the queries
   assume: on a trigger (agent `@mention`, or a new question in a channel the agent belongs to) →
   run K-008 `hybrid_search` → the ranked seed rows *are* the `seeds` list (their `score` is the §6
   cosine distance; rank = their ASC order) → call the LLM with the seed context → `post_agent_answer`
   with those seeds. The score stored on `EMITTED` is the retrieval score, passed straight through —
   do not recompute or invert it.

5. **Tests.** Add `repository`/`services` tests on the isolated `ws:test` (conftest bootstraps
   schema + wipes per test): `post_agent_answer` writes `EMITTED` with correct rank/score, unknown
   seeds are dropped, `seeds=[]` commits with zero edges, a `dupMsg` retry leaves provenance at the
   original count, and `read_provenance`/`read_citing_answers` round-trip. The shell suite already
   covers the raw query surface.

---

## 9. Open questions to surface before impl

1. **Seed identity = message `msgId`.** This gate models provenance as answer→**seed message**. If a
   later milestone adds `Chunk`/`Document` retrieval (the bootstrapped-but-unexercised
   `Chunk.embedding` index), `EMITTED` will need to also point at `Chunk` nodes — either a second
   relationship or a `Message|Chunk` endpoint union. Out of K-013 scope; flag when chunk retrieval
   lands. The current edge is `Message`→`Message` only.
2. **Trigger policy is a service decision.** "Agent `@mention`" vs. "new question in a joined
   channel" — the detection/eligibility rule lives in `services.py`, not the graph. Confirm the
   trigger set before wiring the responder.
3. **Where the answer's own embedding is set.** The agent answer is a normal `Message`; if answers
   should themselves be retrievable via §6, the embedding worker must embed them too (async, §6 SET).
   Sequencing is the same open question as K-008 item 2.
