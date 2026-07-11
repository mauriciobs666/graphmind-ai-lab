# K-007 — M2 groundwork: implementation plan

**Date:** 2026-07-04 · **Author:** architect
**Input:** `docs/archive/plans/m2-groundwork-queries.md` (graph-dba deliverable — all query bodies in
this plan are **verbatim from that document**; they are live-verified ground truth and must be
copied, not re-derived). Referenced below as **[DBA §item-N]**.

**Goal.** Land the six verified K-007 items in code and canonical docs: agent authorship,
self-guarding write paths v2 (retry idempotency + first-post race), `Message.threadId` denorm,
millisecond-tie-safe reads/cursors, TIMEOUT posture, and the 1024-dim RAM line — plus the two
server fold-ins. Out of scope: GraphRAG reads (K-008), reply support in the server API, any
auth/tenancy change, changing the bootstrap `EMBEDDING_DIM` default.

**Baselines going in:** server pytest **75 passed**; `./scripts/test_queries.sh` **92/92**.
**Expected baselines going out:** pytest **95 passed**; query suite **115/115** (enumerations in
§7 are authoritative — if a count lands differently, the enumerated behaviors are the contract
and the docs record the actual printed totals).

---

## 1. Decisions (incl. resolution of the deliverable's open questions)

| # | Question / decision point | Resolution |
|---|---|---|
| OQ1 | DESIGN §9 "idempotent via MERGE" claim; §5.1 role values | **Fix both docs** (step 10): §9 rewritten around the status-row contract; §5.1 becomes `'user' \| 'assistant'` (drop `'human'` and the unused `'system'`). User-confirmed. |
| OQ2 | Should `dupMsg=true` verify payload equality? | **No — trust `msgId`.** It is server-minted (`uuid4().hex`, never client-supplied), so a duplicate can only be a replay of our own write. Record in QUERIES.md §4: *if msgIds ever become client-supplied, add a payload checksum before honoring `dupMsg` as idempotent success.* |
| OQ3 | REST `?since=` — document plain `>` or extend the API with `(since, sinceMsgId)`? | **Document plain `>`; do not extend the API in K-007.** The composite pair is internal to the cursor path. Explicit-`since` reads may re-deliver or skip within that exact millisecond — documented in QUERIES.md §9 and the route docstring. With the monotonic service clock, same-process ties no longer occur; agents that need lossless catch-up use cursor mode. Extending the surface is deferred until a client needs it. |
| OQ4 | `REPLY_TO` inside the guarded FOREACH — reasoned, not verified | **Verify it live now, in `test_queries.sh`** (step 3 rewrites the m4 reply write as a v2 first/subsequent body with the `REPLY_TO` CREATE inside the guarded FOREACH, then asserts the edge exists). **Do not add reply params to `repository.py`** — the server has no reply write surface today; the verified body waits in QUERIES.md §4 for whenever reply support lands. |
| OQ5 | TIMEOUT strategy | **Keep legacy `TIMEOUT=1000` as the deployment default; docs only in K-007.** Per-query client overrides (`g.ro_query(q, params=…, timeout=…)` — pass-through verified) are the documented pattern for future GraphRAG/§6/§8 reads; no code is added now because no such read exists in the server yet (dead code otherwise). DESIGN §10 gains the ops note: **writes ignore TIMEOUT on this build** — bounded batches (≤ a few hundred `UNWIND` rows) and the existing API input caps are the only write-path protection; `TIMEOUT_DEFAULT`/`TIMEOUT_MAX` is the future hard-ceiling switch (mutually exclusive with legacy `TIMEOUT`, change deliberately in one step). |
| OQ6 | File upstream FalkorDB issues (`GRAPH.MEMORY USAGE` vector under-report; one-shot instant-timeout anomaly) | **Punted to the user** — recommended, but not part of this implementation. Both are recorded as caveats in DESIGN §11 / AGENTS.md so they aren't lost. |
| — | Where does the `threadId` backfill live? | **A new ops script `scripts/backfill_thread_ids.sh`** (same shape as `bootstrap_schema.sh`), not a repository method — the server never calls it. Idempotent; run once per existing workspace after deploy. `ws:acme` currently holds 0 messages (checked live), so the dev backfill is a no-op — the script exists for real/imported data. |
| — | Which reads gain `threadId`? | §9.1, §9.2, §5 search, and `get_message` (§4). **Not** the §4 full-thread read (`read_thread`) — its caller supplied the threadId; adding it there is pure noise. §9.1 gains it (beyond the dba's §9.2 verification) so both since-read variants keep one row shape; it is a plain property read on an already-matched node — no verification risk. |
| — | Keyset formulation | **A** — `WHERE m.createdAt > $since OR (m.createdAt = $since AND m.msgId > $sinceMsgId)` (mirrors the ORDER BY 1:1). B is the documented fallback; re-profile on engine upgrades [DBA §item-4]. |
| — | Plain-`>` vs composite in one repository method | One method, two predicate forms selected by `since_msg_id: str \| None` — `None` → plain `m.createdAt > $since` (explicit-since semantics, current baseline query); a string (possibly `""`) → composite keyset. **Both** forms get `ORDER BY m.createdAt, m.msgId` (deterministic order is correct for both; live-verified). QUERIES.md §9.1/§9.2 v2 document the two forms side by side. |
| — | Monotonic clock scope | Applied to **message `createdAt` minting only** (`post_message`) via a lock-guarded `max(clock(), last+1)` in `Services`. Channel/thread `createdAt` ties are harmless and keep the plain clock. Lock because FastAPI runs sync endpoints on a threadpool. |
| — | `existing_members` vs. role lookup | **Replaced** by one method `resolve_member_kinds` (the [DBA §item-1] verified `labels(coalesce(u,a))[0]` query) — one round trip covers author validation, mention validation, *and* role derivation. |
| — | `EMBEDDING_DIM` default | **Stays 1536** in `bootstrap_schema.sh` (the embedding model is still a DESIGN §13 open question and a vector index dimension cannot be altered in place). The K-007 costing is *at 1024* — DESIGN §11 and the script comment say so explicitly; set `EMBEDDING_DIM=1024` when creating M2 workspaces for that model class. |
| — | Dispatch-loop bound | The first↔subsequent re-dispatch loop is bounded at **4 attempts** then raises `RuntimeError`. Ping-pong is impossible by contract (a `hadHead=true` thread always has a TAIL — HEAD and TAIL are created atomically), so the bound is a pure tripwire. |

---

## 2. Repository changes (`server/falkorchat/repository.py`)

All query bodies 1:1 with the updated QUERIES.md (step 3/6 updates the doc first).

### 2.1 New result type + v2 write paths (replaces current §4 methods)

```python
@dataclass(frozen=True)
class MessageWriteStatus:
    written: bool        # committed this call
    had_head: bool       # first path only: lost the first-post race
    dup_msg: bool        # msgId already exists (retry replay)
    author_found: bool   # authorId resolved to a User or Agent

def post_first_message(self, ws, *, thread_id, msg_id, author_id, text, role,
                       created_at, mentions=None) -> MessageWriteStatus | None: ...
def post_subsequent_message(self, ws, *, thread_id, msg_id, author_id, text, role,
                            created_at, mentions=None) -> MessageWriteStatus | None: ...
```

- Query bodies: **verbatim** [DBA §item-2] "§4 v2 — post the first message" / "post a subsequent
  message". They write `threadId: $threadId` inline (item 3) and carry mention resolution
  *before* the guard with the nested-`FOREACH` edge creation *inside* it.
- Return mapping: one status row → `MessageWriteStatus(*row)`; **zero rows → `None`**
  (first path: thread anchor missing; subsequent: no TAIL). The old `_assert_written` raise and
  the `_MENTIONS_BLOCK` constant are **deleted** — the "empty result ⇒ raise" doctrine now
  applies only to the anchor, and the service owns the dispatch (see §3).
- Keep the docstring notes: the `UNWIND (CASE WHEN $mentions = [] …)` guard is now
  **load-bearing for the write itself** (a bare empty UNWIND collapses the stream before the
  FOREACH); `DELETE` inside FOREACH and nested FOREACH are live-verified.

### 2.2 `resolve_member_kinds` (replaces `existing_members`)

```python
def resolve_member_kinds(self, ws, *, ids: list[str]) -> dict[str, str | None]:
    """id -> 'User' | 'Agent' | None. QUERIES.md §2 (member-kind lookup)."""
```

Query: verbatim [DBA §item-1] role-derivation lookup (`UNWIND $ids … RETURN id, CASE … labels(
coalesce(u, a))[0] …`). Empty `ids` short-circuits to `{}` without a query (as today).
Remove `existing_members`; update its two call sites' semantics in services (§3).

### 2.3 Keyset since-reads (§9.1/§9.2 v2)

```python
def read_thread_since(self, ws, *, thread_id, me_id, since: int,
                      since_msg_id: str | None = None, limit: int = 50) -> list[dict]: ...
def read_ws_since(self, ws, *, me_id, since: int,
                  since_msg_id: str | None = None, limit: int = 50) -> list[dict]: ...
```

- `since_msg_id is None` → predicate `m.createdAt > $since` (unchanged baseline);
  else → formulation-A composite `m.createdAt > $since OR (m.createdAt = $since AND
  m.msgId > $sinceMsgId)` [DBA §item-4].
- **Both** forms: `ORDER BY m.createdAt, m.msgId` and `RETURN … m.threadId AS threadId …`
  added; `_since_row` gains the `threadId` key. Everything else (POSTED_BY, `isMention`
  OPTIONAL MATCH, `coalesce` author id) unchanged.

### 2.4 Composite cursor (§9.3/§9.4 v2)

```python
def advance_cursor(self, ws, *, me_id, thread_id, cursor_id,
                   now: int, now_msg_id: str) -> tuple[int, str | None] | None: ...
def get_cursor(self, ws, *, cursor_id) -> tuple[int, str | None] | None: ...
```

- `advance_cursor` body: **verbatim** [DBA §item-4] §9.3 v2 (composite monotonic guard computed
  once in a `WITH`, both `SET`s CASE on `adv`; `coalesce(rc.lastReadMsgId, '')` covers pre-K-007
  cursors — **no cursor backfill needed**). Returns the `(lastReadAt, lastReadMsgId)` row, or
  `None` when the member anchor misses (as today).
- `get_cursor`: `RETURN rc.lastReadAt, rc.lastReadMsgId` → tuple, or `None` when absent.
- **No schema change**: `lastReadMsgId` is a plain property; the `cursorId` index + constraint
  already back the MERGE. `bootstrap_schema.sh` is untouched except the `EMBEDDING_DIM` comment.

### 2.5 `threadId` on point reads; §3 creates

- `search_messages`: add `m.threadId AS threadId` to RETURN and the row dict (§5 v2).
- `get_message`: add `m.threadId AS threadId` to RETURN and the row dict.
- `create_channel`: `MERGE … ON CREATE SET …` → `CREATE (c:Channel {channelId:…, name:…,
  createdAt:…}) RETURN c` (fold-in 2 — ids are freshly minted uuids, the MERGE can never match;
  the uniqueness constraint stays as backstop).
- `create_thread`: `MATCH (c:Channel {channelId:$channelId}) CREATE (t:Thread {…})
  CREATE (c)-[:HAS_THREAD]->(t) RETURN t` — and **raise on empty result** (channel anchor
  missing ⇒ nothing written; the service pre-validates, so this is a tripwire like the old
  `_assert_written`). Documented consequence: create endpoints are **non-idempotent** — a
  retried create mints a new id (QUERIES.md §3 note + DESIGN §9 row).

Unchanged: `ping`, `list_channels`, `list_threads`, `thread_has_head`, `ensure_user`,
`ensure_agent` (MERGE is correct there — externally-supplied ids), `read_thread`,
`thread_exists`, `channel_exists`.

---

## 3. Service changes (`server/falkorchat/services.py`)

### 3.1 Monotonic clock

```python
def __init__(self, repo, *, clock=_default_clock, id_gen=_default_id):
    ...
    self._ts_lock = threading.Lock()
    self._last_ts = 0

def _next_ts(self) -> int:
    """Monotonic per-process ms clock — makes same-ms message ties impossible (K-007 item 4a)."""
    with self._ts_lock:
        ts = max(self._clock(), self._last_ts + 1)
        self._last_ts = ts
        return ts
```

Used **only** for message `createdAt` in `post_message`. Injected-clock tests keep working
(a fixed injected clock now yields strictly increasing stamps across posts — that is the point).

### 3.2 `post_message` — role derivation + status-row dispatch

Replace the body after `thread_exists` with:

```python
wanted = _dedup(list(mentions or []))
kinds = self._repo.resolve_member_kinds(ctx.ws, ids=[ctx.actor, *wanted])
actor_kind = kinds.get(ctx.actor)
if actor_kind is None:
    raise UnknownActorError(ctx.actor)
role = "user" if actor_kind == "User" else "assistant"   # confirmed mapping
unknown = [m for m in wanted if kinds.get(m) is None]
if unknown:
    raise UnknownMemberError(unknown)

msg_id, now = self._id(), self._next_ts()
use_first = not self._repo.thread_has_head(ctx.ws, thread_id=thread_id)
for _attempt in range(4):
    write = self._repo.post_first_message if use_first else self._repo.post_subsequent_message
    st = write(ctx.ws, thread_id=thread_id, msg_id=msg_id, author_id=ctx.actor,
               text=text, role=role, created_at=now, mentions=wanted)
    if st is None:
        if use_first:                       # thread anchor vanished (TOCTOU)
            raise ThreadNotFoundError(thread_id)
        use_first = True                    # no TAIL yet — retry as first-post
        continue
    if st.written or st.dup_msg:            # dup_msg = idempotent success (OQ2)
        break
    if not st.author_found:                 # belt-and-suspenders vs the pre-check
        raise UnknownActorError(ctx.actor)
    if st.had_head:                         # lost the first-post race
        use_first = False
        continue
    raise RuntimeError(f"unexpected write status {st!r} (thread={thread_id!r})")
else:
    raise RuntimeError(f"message write dispatch did not converge (thread={thread_id!r}, msg={msg_id!r})")
return {"msgId": msg_id, "threadId": thread_id, "authorId": ctx.actor,
        "text": text, "role": role, "createdAt": now, "mentions": wanted}
```

The docstring's "actor is a User in M1" caveat is replaced by: role is derived from the actor's
node label (`User → user`, `Agent → assistant`) — agents can now author.

### 3.3 `read_messages` — composite cursor wiring

Thread mode changes only:

```python
cursor_id = f"{ctx.actor}:{thread_id}"
if explicit_since:
    rows = self._repo.read_thread_since(ctx.ws, thread_id=thread_id, me_id=ctx.actor,
                                        since=since, since_msg_id=None, limit=limit)  # plain >
    return rows
pair = self._repo.get_cursor(ctx.ws, cursor_id=cursor_id)
eff_since, eff_msg = pair if pair is not None else (0, None)
rows = self._repo.read_thread_since(ctx.ws, thread_id=thread_id, me_id=ctx.actor,
                                    since=eff_since or 0, since_msg_id=eff_msg or "", limit=limit)
if advance and rows:
    last = rows[-1]     # rows are ORDER BY (createdAt, msgId): last row IS the max pair
    self._repo.advance_cursor(ctx.ws, me_id=ctx.actor, thread_id=thread_id,
                              cursor_id=cursor_id, now=last["createdAt"],
                              now_msg_id=last["msgId"])
return rows
```

Room-wide mode: unchanged semantics (`since_msg_id=None`, plain `>`). Docstring documents the
OQ3 contract: cursor-driven reads never skip/re-deliver; explicit `since` may, within that exact
millisecond.

`ensure_actor` is unchanged (still projects a `User` — the configured M1 actor). An **Agent**
actor is exercised in tests by seeding `ensure_agent` and a `CallContext` pointing at it; real
agent auth is out of scope.

---

## 4. Transport / schema / config surfaces

- **`api.py`** — no route changes. Responses (plain dicts) gain `threadId` automatically in
  `/threads/{id}/messages?since=…`, `/search`, `/messages/{id}`. Two comment fixes:
  the `/messages/{msg_id}` comment ("Message has no threadId") is stale — the route stays flat
  (msgId is workspace-unique) but the body now carries `threadId`; the `read_thread` route
  comment gains the explicit-`since` plain-`>` note (OQ3).
- **`mcp.py`** — no signature changes; `read_messages`/`search_messages` rows gain `threadId`
  through the shared service. Touch only the `read_messages` docstring (cursor is now
  tie-safe; explicit `since` is plain `>`).
- **`schemas.py`** — unchanged (request models only; responses are plain dicts).
- **`config.py`** — unchanged.
- **`db.py`** (fold-in 1) — late-bind config:

  ```python
  def connect(host: str | None = None, port: int | None = None) -> FalkorDB:
      return FalkorDB(host=host if host is not None else config.FALKORDB_HOST,
                      port=port if port is not None else config.FALKORDB_PORT)
  ```

- **`app.py`** — unchanged.

---

## 5. Ops: backfill script + bootstrap comment

**New `scripts/backfill_thread_ids.sh <workspaceId> [...]`** — same conventions as
`bootstrap_schema.sh` (env `FALKORDB_HOST`/`FALKORDB_PORT`, redis-cli). Per workspace, run the
workspace-wide backfill (verbatim [DBA §item-3], workspace-wide variant):

```cypher
MATCH (t:Thread)-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
WHERE m.threadId IS NULL
SET m.threadId = t.threadId
RETURN count(m) AS backfilled
```

Print the count; idempotent (second run returns 0). Header comment must carry: (a) the
**per-thread batched variant** (add `{threadId: $threadId}` to the Thread anchor) for large
workspaces — *writes cannot be killed by TIMEOUT on this build, so bound the work yourself*;
(b) the orphan caveat — messages unreachable from a HEAD (residue of the pre-v2 defects) are
not backfilled and are already invisible to thread reads. Run order: deploy v2 → run backfill
per existing workspace (`ws:acme` has 0 messages today — a no-op, but run it anyway to prove
the script). Old rows return `threadId: null` in §9.2/§5 until backfilled — acceptable transient.

**`bootstrap_schema.sh`** — comment-only change at the `EMBEDDING_DIM` block: default stays
1536; note the M2 costing baseline is 1024 dims (~12.4 KB/msg) and the dim must be chosen per
model **before** workspace creation (vector index dimension is fixed at creation).

---

## 6. Canonical doc updates (first-class steps — QUERIES.md owns all query bodies)

### 6.1 `docs/QUERIES.md`

- **§2** — add the member-kind lookup query ([DBA §item-1]) with the `User→user / Agent→assistant
  / None→reject` mapping note (mapping itself is service-side).
- **§3** — "Create a channel" / "Create a thread" switch to the `CREATE` forms (§2.5), with the
  note: ids are server-minted so MERGE could never match; constraints backstop; **creates are
  non-idempotent** (a retried create mints a new id).
- **§4 — v2 replaces the current §4 write paths entirely.** Both v2 bodies verbatim; the
  status-row contract table ([DBA §item-2]); the three "notes that must survive": load-bearing
  empty-`UNWIND` guard, `DELETE`-inside-FOREACH + nested FOREACH verified, and the **reply
  add-on** shown *inside* the guarded FOREACH (now live-verified by the updated
  `test_queries.sh` — see step 3). Keep the mentions explainer (MENTIONS_MEMBER vs MENTIONS,
  dedup/unknown-skip). Add the OQ2 note (trust server-minted msgId; checksum if ever
  client-supplied). Note `threadId` is written inline, **deliberately unindexed** (navigation
  metadata; §9.1 remains the canonical walk; saves RAM/write cost).
- **§4** — `get_message` RETURN gains `m.threadId`.
- **New §4.x "Backfill `threadId` (one-off)"** — per-thread + workspace-wide variants, verified
  idempotent, orphan caveat, pointer to `scripts/backfill_thread_ids.sh`.
- **§5** — RETURN gains `m.threadId`.
- **§9 intro** — ordering note extended: deterministic total order is `(createdAt, msgId)`;
  tie-break is lexical msgId (within one ms, delivery order is id order, not arrival order —
  acceptable across writers; mint k-sortable ids if human-facing tie order ever matters — never
  re-sort pages). Cursor advances to the newest **returned** `(createdAt, msgId)` pair.
- **§9.1 / §9.2 v2** — composite-keyset form (formulation A) + plain-`>` form side by side,
  `ORDER BY m.createdAt, m.msgId` on both, `m.threadId` in RETURN, the `$sinceMsgId = ''` base
  convention, formulation B recorded as the fallback with the re-profile-on-upgrade note, and
  the OQ3 semantics sentence (explicit `since` may re-deliver/skip within that millisecond;
  cursor-driven reads never do).
- **§9.3 v2** — composite monotonic advance verbatim, with the five verified scenarios and the
  "no schema change; `coalesce(…, '')` covers old cursors" note.
- **§9.4** — returns the `(lastReadAt, lastReadMsgId)` pair.

### 6.2 `docs/DESIGN.md` (link to QUERIES.md — never copy query bodies)

- **§5.1** — `Message.role` → `'user' | 'assistant'`; add `threadId` to the Message property
  list (denormalized, unindexed, navigation metadata).
- **§9 write-path table** — replace the "Post message" row: guarded `CREATE` inside
  FOREACH+CASE guards (never a conditional MERGE of the two paths), always returns a status
  row, **retry-idempotent via the `dupMsg` status** (the old "idempotent via unique constraint"
  claim was falsified — [DBA §item-2] evidence), constraint rollback as the concurrency
  backstop. Add a row/note: channel/thread creates are plain `CREATE` — non-idempotent.
- **§10 ops** — TIMEOUT posture per OQ5 (keep legacy `TIMEOUT=1000`; per-query client override
  for GraphRAG reads; **writes ignore TIMEOUT — bounded batches are the only write protection**;
  `TIMEOUT_DEFAULT`/`TIMEOUT_MAX` mutually exclusive with legacy knob).
- **§11** — replace the sketch with the empirical 1024-dim line ([DBA §item-6]): the
  per-message component table, **12,387 B/message observed ≈ 12.5 KB ⇒ ~1.25 GB per
  100k-message workspace** (vs ~17–18 KB extrapolated at 1536), `threadId` ≈ 50–60 B noise,
  `lastReadMsgId` negligible, ingestion datapoint ~1,178 msg/s at 256-row batches, and the
  measurement caveat: `GRAPH.MEMORY USAGE` under-reports vector-index memory on this build —
  **size from `INFO memory` deltas** until fixed upstream.
- **§12** — append to the M2 line: groundwork (K-007) landed — agent authorship, guarded write
  paths, keyset cursors, threadId denorm.

### 6.3 `AGENTS.md` (falkor-chat) + baseline citations

- **Locked-decisions table, add:** `Message.threadId` denormalized inline, unindexed ·
  guarded-CREATE write paths (FOREACH+CASE per path) with an always-returned status row ·
  `Message.role` values `user`/`assistant` derived server-side from the author label ·
  composite `(createdAt, msgId)` keyset for cursor reads (`ReadCursor.lastReadMsgId`).
- **Live-verified facts, add:** writes ignore `TIMEOUT` (reads enforce it batch-granularly;
  client `timeout=` pass-through works and is uncapped while `TIMEOUT_MAX=0`) ·
  `GRAPH.MEMORY USAGE` under-reports vector-index memory (use `INFO memory` deltas) ·
  `labels(coalesce(u,a))[0]` subscripting works · `DELETE` inside `FOREACH` and nested
  `FOREACH` work · formulation-A composite OR predicate still plans as a bare
  `Node By Index Scan` (re-profile on engine upgrades).
- **"Message write paths" section rewrite:** status-row contract (zero rows = thread-anchor
  missing only; `dupMsg` = idempotent success; `hadHead` = re-dispatch), guarded CREATE (no
  MERGE on Message — constraint stays as backstop), mentions ride inside the guard, `role`
  derived not trusted, empty-`UNWIND` guard now load-bearing for the write itself.
- **Baselines:** rule 5 and the scripts table `92` → `115`; M1-server section `75 passed` →
  `95 passed`.
- **Root `/AGENTS.md`** and **`falkor-chat/README.md`**: update the `92/92` expected output
  (README line ~122) and `# 75 passed` (README ~220, root AGENTS.md key-commands block) to the
  new totals. (README §168's "92/92" is M0 milestone history — leave it.) Sweep with
  `grep -rn "92/92\|92-assertion\|75 passed"` before finishing.

---

## 7. Test strategy

### 7.1 `scripts/test_queries.sh` — 92 → **115** (+23)

Rewrites (no count change unless noted): §3 channel/thread creates → CREATE forms; §4 writes
m1–m4 → v2 bodies (m4 = the reply write with `REPLY_TO` **inside the guarded FOREACH** — this
*is* the OQ4 live verification); §7 agent post → v2; §9 mention writes mn1–mn3 → v2; §9.3
cursor asserts → composite form.

Net-new assertions (+23):

| Area | New asserts |
|---|---|
| §3 constraint checks after CREATE switch (dup `channelId`, dup `threadId` blocked) | +2 |
| §4 first-post: status row + node-exists become two asserts; m4 `REPLY_TO` edge exists | +2 |
| §4 guard block: replay of a subsequent write → `dupMsg=true`; NEXT count unchanged/no self-loop; node count unchanged; late first-post → `hadHead=true`; HEAD count still 1; unknown author → `authorFound=false`; its message absent; `threadId` present on m1 | +8 |
| §4 profile: v2 subsequent write — `assert_not_contains "All Node Scan"` | +1 |
| §4.x backfill: run 1 counts, run 2 idempotent (0) | +2 |
| §9 keyset: seed a same-`createdAt` tie; page 1 at the boundary; page 2 with `(since, sinceMsgId)` returns the tied sibling; nothing skipped (tie-skip regression) | +3 |
| §9.2 RETURN carries `threadId` | +1 |
| §5 RETURN carries `threadId` | +1 |
| §9.3 five-scenario composite coverage (tie-larger advances / tie-smaller refused add 2); §9.4 returns the pair | +3 |

### 7.2 Server pytest — 75 → **95** (+20)

**`test_repository.py` 30 → 39.** Updated in place: the two unknown-author tests (now assert
`author_found=False` status + nothing written, not `RuntimeError`); the two `existing_members`
tests → `resolve_member_kinds` dict shape; the four cursor tests → new signatures/pair returns.
New (+9), integration against live `ws:test`:

1. **Retry replay** (defect A regression): exact replay of a subsequent write returns
   `dup_msg=True`; chain, NEXT count, and POSTED_BY count unchanged; no self-loop.
2. Replay of a first write → `dup_msg=True, had_head=True`; single HEAD.
3. **Two-HEAD refusal** (defect B): first-path on a headed thread (fresh msgId) →
   `written=False, had_head=True`; no node created.
4. Subsequent on a TAIL-less thread → returns `None` (dispatch signal).
5. **Agent authorship** (item 1 regression): agent-authored subsequent write commits;
   `POSTED_BY` targets the Agent; role stored as passed.
6. `threadId` stamped by both write paths and returned by `read_ws_since` + `search_messages`.
7. **Tie-skip regression** (defect item 4): two messages with equal `createdAt`; composite
   keyset paging (`limit` at the boundary) delivers all rows; plain-`>` explicit read still
   excludes the boundary row (`since_msg_id=None` semantics).
8. Composite cursor: tie-larger advances, tie-smaller replay refused, backward refused.
9. `get_cursor` returns the `(lastReadAt, lastReadMsgId)` pair; pre-K-007 cursor (no msgId
   property) reads back `(ts, None)`.

**`test_services.py` 17 → 24.** `FakeRepo` grows `resolve_member_kinds`, status-returning
post methods, and pair-shaped cursors; existing tests updated accordingly. New (+7), unit:
role `assistant` for an Agent actor · `dup_msg` → idempotent success (returns the message
dict) · `had_head` → re-dispatched as subsequent, exactly one write committed · subsequent
`None` → re-dispatched as first · dispatch loop bounded → `RuntimeError` · monotonic clock:
fixed injected clock, two posts → strictly increasing `createdAt` · cursor advance receives
the last returned row's `(createdAt, msgId)` and the stored pair is fed back as the composite
`since`.

**`test_services_live.py` 3 → 5.** New: **concurrency hammer** — one shared `Services`, a
fresh thread, ~8 `ThreadPoolExecutor` workers racing `post_message` (redis-py connections are
pool-backed and thread-safe); assert exactly 1 HEAD, 1 TAIL, contiguous chain of 8, no errors
(mirrors [DBA §item-2] hammer, scaled for CI). New: agent-actor end-to-end — seed
`ensure_agent("a1")`, post with `CallContext(actor="a1")`, read back `role == "assistant"`.

**`test_api.py` 14 → 15.** New: `threadId` present in `?since=` read rows, `/search` rows,
and `GET /messages/{id}`.

**`test_mcp.py` 7 → 8.** New: `read_messages` rows carry `threadId`.

**`test_app.py` 4** — unchanged.

TDD note (repo convention): for steps 4–6 write the defect-regression tests first against the
old code where they reproduce (retry replay and tie-skip both reproduce today), watch them
fail, then land the v2 queries/service logic.

---

## 8. Ordered steps (green-suite checkpoint after every step)

1. **Baseline check** — FalkorDB up; run pytest (75) and `./scripts/test_queries.sh` (92/92).
2. **`db.py` late-bind fold-in** (§4). Checkpoint: pytest 75.
3. **QUERIES.md §4 v2 + §4.x backfill + `test_queries.sh` §4/§7/§9-mention rewrite** (§6.1,
   §7.1 rows 2–6) — includes the **OQ4 `REPLY_TO`-inside-FOREACH live verification**.
   Checkpoint: query suite green at its intermediate count.
4. **`repository.py` write paths v2 + `resolve_member_kinds`** (§2.1–§2.2) with
   `test_repository.py` updates + new tests 1–6 (red → green). Checkpoint: pytest green.
5. **`services.py` role derivation + status dispatch + monotonic clock** (§3.1–§3.2) with
   `test_services.py` updates/new + live hammer + agent-actor test. Checkpoint: pytest green.
   **← riskiest step** (see §9).
6. **Keyset reads + composite cursor** — QUERIES.md §9 v2 + `test_queries.sh` §9 asserts;
   `repository.py` §2.3–§2.4; `services.py` §3.3; repository tests 7–9 + services cursor
   tests (tie-skip regression red-first). Checkpoint: both suites green.
7. **`threadId` surfacing** — QUERIES.md §5/§4-get_message; `repository.search_messages`/
   `get_message`; api/mcp test additions + comment fixes (§4). Checkpoint: both suites green.
8. **§3 MERGE→CREATE** — QUERIES.md §3 + `test_queries.sh` §3 + `repository.create_channel`/
   `create_thread` (+ empty-result raise). Checkpoint: both suites green.
9. **Backfill ops** — `scripts/backfill_thread_ids.sh`; run against `ws:acme` (no-op today);
   `test_queries.sh` backfill asserts if not already landed in step 3. Checkpoint: query suite.
10. **Doc sweep** — DESIGN.md §5.1/§9/§10/§11/§12; AGENTS.md (falkor-chat) decisions/facts/
    write-path section/baselines; root AGENTS.md + README baselines; `bootstrap_schema.sh`
    comment (§5, §6.2–§6.3).
11. **Final full run** — pytest **95**, `./scripts/test_queries.sh` **115/115**; pin the actual
    printed totals in every doc touched in step 10.

---

## 9. Risks & mitigations

- **Riskiest step: 5 (service dispatch over v2 status rows)** — it rewrites the heart of the
  write path. Mitigation: the queries themselves are pre-verified verbatim (including the
  16-thread hammer and all-or-nothing constraint rollback [DBA §item-2]); the dispatch loop is
  a direct transcription of the verified contract table; defect regressions are written
  red-first against the reproducing old code; the loop is bounded (tripwire, not correctness);
  green checkpoints on both sides.
- **`test_queries.sh` assertion-count drift** — the enumerated list (§7.1) is authoritative;
  the docs record whatever total the final script prints.
- **Composite predicate on future engine builds** — formulation A folds into the index scan
  *today*; B is the documented fallback; AGENTS.md carries the re-profile-on-upgrade note.
- **Unbackfilled rows** return `threadId: null` in §9.2/§5 until the script runs — transient,
  documented; clients must tolerate null. Orphan messages (pre-v2 residue) stay unbackfilled
  and invisible — by design.
- **Writes are unkillable by TIMEOUT** — carried as a standing constraint (bounded batches,
  input caps stay); no new unbounded write is introduced by this plan.
- **Lexical msgId tie order** — within one ms, delivery order is id order, not arrival order;
  acceptable across writers (monotonic clock removes same-process ties). Documented; UUIDv7/ULID
  is the future fix if it ever matters.

---

## 10. Ready to implement

Execute steps 1–11 in order; every query body comes verbatim from
`docs/archive/plans/m2-groundwork-queries.md` (via the updated `QUERIES.md`, which this plan makes
canonical). All design decisions are made (§1) — including all six graph-dba open questions
(OQ1 doc fixes in step 10; OQ2 trust-msgId; OQ3 plain-`>` REST, no API extension; OQ4 verified
live in step 3, repository fold-in deferred; OQ5 docs-only TIMEOUT posture; OQ6 punted to the
user as recommended upstream filings). Signatures, dispatch pseudocode, doc deltas, and the
full test enumeration are in §§2–7. Target baselines: **pytest 95 · query suite 115/115**.
