"""Data-access layer — the only place Cypher lives (DESIGN §14.2).

Each method maps 1:1 to a verified query in `docs/QUERIES.md`, always
parameterised, using `ro_query` for reads and `query` for writes, scoped to the
per-workspace graph via `db.workspace_graph`. No business logic lives here —
that is `services.py`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from falkordb import FalkorDB

from . import db


class MemberIdCollisionError(Exception):
    """Member id already held by the other label (QUERIES.md §2/§7 v2, DEF-1).

    Member ids are namespace-unique across `User`/`Agent` (locked rule): every
    `coalesce(u, a)` lookup assumes one id resolves to one member, so a guarded
    ensure refuses to create a node whose id the other label already holds.
    Raised by `ensure_user`/`ensure_agent` on the `collided` status states —
    both the refusal (other label holds the id) and the pre-guard corruption
    alarm (both labels hold it). Re-exported by `services` as part of the
    service error surface (defined here to avoid a repository→services cycle).
    """


@dataclass(frozen=True)
class MessageWriteStatus:
    """Status row returned by the §4 v2 write paths (QUERIES.md §4 contract).

    ``None`` from the write methods means the *anchor* missed (first path:
    thread missing; subsequent path: no TAIL yet) — everything else comes back
    as a status row the service dispatches on.
    """

    written: bool        # committed this call
    had_head: bool       # first path only: lost the first-post race
    dup_msg: bool        # msgId already exists (retry replay → idempotent success)
    author_found: bool   # authorId resolved to a User or Agent


class Repository:
    """Cypher access over a FalkorDB connection.

    Every method takes the workspace id `ws` (graph key ``ws:{ws}``) so the
    single-tenant seam stays in `config`/`api`, never hardcoded here.
    """

    def __init__(self, conn: FalkorDB | db.LazyFalkorDB) -> None:
        self._conn = conn

    def _graph(self, ws: str):
        return db.workspace_graph(self._conn, ws)

    def ping(self, ws: str) -> bool:
        """Liveness probe — a trivial read against the workspace graph."""
        res = self._graph(ws).ro_query("RETURN 1")
        return bool(res.result_set and res.result_set[0][0] == 1)

    # ── §3 Channels ────────────────────────────────────────────────────────────

    def create_channel(
        self, ws: str, *, channel_id: str, name: str, created_at: int
    ) -> None:
        """Create a channel (plain CREATE). QUERIES.md §3.

        Ids are server-minted uuids, so a MERGE could never match — this is a
        CREATE with the uniqueness constraint as backstop. Consequence: creates
        are non-idempotent (a retried create mints a new id).
        """
        self._graph(ws).query(
            "CREATE (c:Channel {channelId: $channelId, name: $name, "
            "createdAt: $createdAt}) "
            "RETURN c",
            {"channelId": channel_id, "name": name, "createdAt": created_at},
        )

    def list_channels(self, ws: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """List channels newest-first (index-anchored). QUERIES.md §3."""
        res = self._graph(ws).ro_query(
            "MATCH (c:Channel) WHERE c.channelId > '' "
            "RETURN c.channelId AS channelId, c.name AS name, c.createdAt AS createdAt "
            "ORDER BY c.createdAt DESC LIMIT $limit",
            {"limit": limit},
        )
        return [
            {"channelId": row[0], "name": row[1], "createdAt": row[2]}
            for row in res.result_set
        ]

    # ── §3 Threads ─────────────────────────────────────────────────────────────

    def create_thread(
        self, ws: str, *, channel_id: str, thread_id: str, title: str, created_at: int
    ) -> None:
        """Create a thread under a channel (plain CREATE). QUERIES.md §3.

        Non-idempotent like `create_channel` (server-minted ids; constraint as
        backstop). Raises when the channel anchor is missing — the whole query
        no-ops then, and a silent no-op would lose the thread. The service
        pre-validates, so the raise is a tripwire.
        """
        res = self._graph(ws).query(
            "MATCH (c:Channel {channelId: $channelId}) "
            "CREATE (t:Thread {threadId: $threadId, title: $title, "
            "createdAt: $createdAt, updatedAt: $createdAt}) "
            "CREATE (c)-[:HAS_THREAD]->(t) "
            "RETURN t",
            {
                "channelId": channel_id,
                "threadId": thread_id,
                "title": title,
                "createdAt": created_at,
            },
        )
        if not res.result_set:
            raise RuntimeError(
                f"thread create was a no-op — channel not found "
                f"(channel={channel_id!r}, thread={thread_id!r})"
            )

    def list_threads(
        self, ws: str, *, channel_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List a channel's threads, most-recently-updated first. QUERIES.md §3."""
        res = self._graph(ws).ro_query(
            "MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t:Thread) "
            "RETURN t.threadId AS threadId, t.title AS title, t.updatedAt AS updatedAt "
            "ORDER BY t.updatedAt DESC LIMIT $limit",
            {"channelId": channel_id, "limit": limit},
        )
        return [
            {"threadId": row[0], "title": row[1], "updatedAt": row[2]}
            for row in res.result_set
        ]

    def thread_has_head(self, ws: str, *, thread_id: str) -> bool:
        """True once a thread has its first message (a HEAD pointer).

        The service uses this to pick the first-vs-subsequent §4 write variant.
        """
        # NOTE (live gotcha): `exists((t)-[:HEAD]->())` returns true even when no
        # HEAD edge exists on this build; `count{ }` isn't supported. An
        # OPTIONAL MATCH + `IS NOT NULL` is the reliable existence check here.
        res = self._graph(ws).ro_query(
            "MATCH (t:Thread {threadId: $threadId}) "
            "OPTIONAL MATCH (t)-[:HEAD]->(h) "
            "RETURN h IS NOT NULL AS hasHead",
            {"threadId": thread_id},
        )
        return bool(res.result_set and res.result_set[0][0])

    # ── §2/§7 Members (author + mention targets) ────────────────────────────────

    # Both ensures are v2 guarded CREATEs (QUERIES.md §2/§7, DEF-1): member ids
    # are namespace-unique across User/Agent (locked rule). Status-row contract
    # (exactly one row, always — no anchor MATCH, so zero-row is impossible):
    #   created          → fresh node written (success)
    #   existed          → id already on the same label, nothing written
    #                      (idempotent success — re-ensure never updates props)
    #   collided         → id held by the OTHER label, nothing written → raise
    #   existed+collided → both labels hold the id (pre-guard corruption) → alarm
    # Idempotency comes from the status logic (MERGE-in-FOREACH is non-standard);
    # the same-label uniqueness constraints stay as the concurrency backstop.
    # The residual cross-label race is one-query wide and documented — not closed.

    @staticmethod
    def _check_ensure_status(res, *, member_id: str, wanted: str, other: str) -> None:
        created, existed, collided = (bool(v) for v in res.result_set[0])
        if not collided:
            return  # created or existed — both are quiet successes
        if existed:
            raise MemberIdCollisionError(
                f"member-id namespace corrupted: {member_id!r} exists as both a "
                f"User and an Agent — nothing written, manual repair required"
            )
        article = {"User": "a", "Agent": "an"}
        raise MemberIdCollisionError(
            f"member id {member_id!r} is already held by {article[other]} {other} — "
            f"refusing to create {article[wanted]} {wanted} "
            f"(member ids are namespace-unique across User/Agent)"
        )

    def ensure_user(
        self, ws: str, *, user_id: str, display_name: str | None = None,
        email: str | None = None,
    ) -> None:
        """Project a User into the workspace graph (guarded ensure). QUERIES.md §2.

        Idempotent on the same label; raises `MemberIdCollisionError` when the
        id is held by an Agent (or by both labels — corruption alarm).
        """
        res = self._graph(ws).query(
            "OPTIONAL MATCH (u:User  {userId:  $userId}) "
            "OPTIONAL MATCH (a:Agent {agentId: $userId}) "
            "WITH u, a, (u IS NULL AND a IS NULL) AS ok "
            "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
            "  CREATE (:User {userId: $userId, displayName: $displayName, email: $email}) "
            ") "
            "RETURN ok            AS created, "
            "       u IS NOT NULL AS existed, "
            "       a IS NOT NULL AS collided",
            {"userId": user_id, "displayName": display_name, "email": email},
        )
        self._check_ensure_status(res, member_id=user_id, wanted="User", other="Agent")

    def ensure_agent(
        self, ws: str, *, agent_id: str, name: str | None = None,
        model: str | None = None, created_at: int | None = None,
    ) -> None:
        """Register an Agent in the workspace graph (guarded ensure). QUERIES.md §7.

        Mirror of `ensure_user` — same contract with the labels swapped.
        """
        res = self._graph(ws).query(
            "OPTIONAL MATCH (a:Agent {agentId: $agentId}) "
            "OPTIONAL MATCH (u:User  {userId:  $agentId}) "
            "WITH a, u, (a IS NULL AND u IS NULL) AS ok "
            "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
            "  CREATE (:Agent {agentId: $agentId, name: $name, model: $model, "
            "createdAt: $createdAt}) "
            ") "
            "RETURN ok            AS created, "
            "       a IS NOT NULL AS existed, "
            "       u IS NOT NULL AS collided",
            {"agentId": agent_id, "name": name, "model": model, "createdAt": created_at},
        )
        self._check_ensure_status(res, member_id=agent_id, wanted="Agent", other="User")

    # ── §4 Messages (v2 self-guarding write paths) ──────────────────────────────
    #
    # Both paths guard the write inside their single GRAPH.QUERY via
    # `FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | …)` — a guard per path,
    # never a conditional merge of the two paths (locked decision). The message
    # is a guarded CREATE (no MERGE); the `Message.msgId` uniqueness constraint
    # stays as the concurrency backstop. Notes that must survive edits:
    #   * The `UNWIND (CASE WHEN $mentions = [] …)` guard is LOAD-BEARING for the
    #     write itself — a bare empty UNWIND collapses the row stream before the
    #     FOREACH and the whole write silently no-ops (live-verified).
    #   * `DELETE` inside FOREACH (TAIL relink) and nested FOREACH (mentions) are
    #     live-verified on this build.
    #   * Zero rows back = the *anchor* missed (thread / TAIL) → return None; the
    #     service owns the dispatch (QUERIES.md §4 status-row contract).

    @staticmethod
    def _write_status(res) -> MessageWriteStatus | None:
        if not res.result_set:
            return None
        return MessageWriteStatus(*(bool(v) for v in res.result_set[0]))

    def post_first_message(
        self, ws: str, *, thread_id: str, msg_id: str, author_id: str,
        text: str, role: str, created_at: int, mentions: list[str] | None = None,
    ) -> MessageWriteStatus | None:
        """First message in a thread — creates HEAD + TAIL. QUERIES.md §4 v2.

        Mentions ride inside this single GRAPH.QUERY (atomicity rule).
        Returns the status row, or None when the thread anchor is missing.
        """
        res = self._graph(ws).query(
            "MATCH (t:Thread {threadId: $threadId}) "
            "OPTIONAL MATCH (t)-[:HEAD]->(h) "
            "OPTIONAL MATCH (dup:Message {msgId: $msgId}) "
            "OPTIONAL MATCH (ua:User  {userId:  $authorId}) "
            "OPTIONAL MATCH (aa:Agent {agentId: $authorId}) "
            "WITH t, h, dup, coalesce(ua, aa) AS author "
            "UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid "
            "OPTIONAL MATCH (mu:User  {userId:  mid}) "
            "OPTIONAL MATCH (ma:Agent {agentId: mid}) "
            "WITH t, h, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems "
            "WITH t, h, dup, author, mems, "
            "     (h IS NULL AND dup IS NULL AND author IS NOT NULL) AS ok "
            "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
            "  CREATE (m:Message {msgId: $msgId, text: $text, role: $role, "
            "                     createdAt: $createdAt, threadId: $threadId}) "
            "  CREATE (t)-[:HEAD]->(m) "
            "  CREATE (t)-[:TAIL]->(m) "
            "  CREATE (m)-[:POSTED_BY]->(author) "
            "  SET t.updatedAt = $createdAt "
            "  FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) "
            ") "
            "RETURN ok                 AS written, "
            "       h    IS NOT NULL   AS hadHead, "
            "       dup  IS NOT NULL   AS dupMsg, "
            "       author IS NOT NULL AS authorFound",
            {
                "threadId": thread_id, "msgId": msg_id, "authorId": author_id,
                "text": text, "role": role, "createdAt": created_at,
                "mentions": list(mentions or []),
            },
        )
        return self._write_status(res)

    def post_subsequent_message(
        self, ws: str, *, thread_id: str, msg_id: str, author_id: str,
        text: str, role: str, created_at: int, mentions: list[str] | None = None,
    ) -> MessageWriteStatus | None:
        """Append a message — moves TAIL forward via NEXT. QUERIES.md §4 v2.

        Mentions ride inside this single GRAPH.QUERY (atomicity rule).
        Returns the status row, or None when the thread has no TAIL yet.
        """
        res = self._graph(ws).query(
            "MATCH (t:Thread {threadId: $threadId})-[tailRel:TAIL]->(prev:Message) "
            "OPTIONAL MATCH (dup:Message {msgId: $msgId}) "
            "OPTIONAL MATCH (ua:User  {userId:  $authorId}) "
            "OPTIONAL MATCH (aa:Agent {agentId: $authorId}) "
            "WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author "
            "UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid "
            "OPTIONAL MATCH (mu:User  {userId:  mid}) "
            "OPTIONAL MATCH (ma:Agent {agentId: mid}) "
            "WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems "
            "WITH t, tailRel, prev, dup, author, mems, "
            "     (dup IS NULL AND author IS NOT NULL) AS ok "
            "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
            "  CREATE (m:Message {msgId: $msgId, text: $text, role: $role, "
            "                     createdAt: $createdAt, threadId: $threadId}) "
            "  CREATE (prev)-[:NEXT]->(m) "
            "  DELETE tailRel "
            "  CREATE (t)-[:TAIL]->(m) "
            "  CREATE (m)-[:POSTED_BY]->(author) "
            "  SET t.updatedAt = $createdAt "
            "  FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) "
            ") "
            "RETURN ok                 AS written, "
            "       false              AS hadHead, "
            "       dup  IS NOT NULL   AS dupMsg, "
            "       author IS NOT NULL AS authorFound",
            {
                "threadId": thread_id, "msgId": msg_id, "authorId": author_id,
                "text": text, "role": role, "createdAt": created_at,
                "mentions": list(mentions or []),
            },
        )
        return self._write_status(res)

    def read_thread(self, ws: str, *, thread_id: str) -> list[dict[str, Any]]:
        """Read a full thread in order. QUERIES.md §4."""
        res = self._graph(ws).ro_query(
            "MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message) "
            "MATCH (first)-[:NEXT*0..]->(m:Message) "
            "MATCH (m)-[:POSTED_BY]->(author) "
            "RETURN m.msgId AS msgId, m.text AS text, m.role AS role, "
            "m.createdAt AS createdAt, "
            "coalesce(author.userId, author.agentId) AS authorId, "
            "author.displayName AS displayName, labels(author) AS authorType "
            "ORDER BY m.createdAt",
            {"threadId": thread_id},
        )
        return [
            {
                "msgId": row[0], "text": row[1], "role": row[2], "createdAt": row[3],
                "authorId": row[4], "displayName": row[5], "authorType": row[6],
            }
            for row in res.result_set
        ]

    # ── §5 Full-text search ─────────────────────────────────────────────────────

    def search_messages(
        self, ws: str, *, query: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Workspace-wide full-text keyword search over message text. QUERIES.md §5.

        Anchored on the `Message` full-text index; the channel-scoping MATCH from
        §5 is omitted for the workspace-wide search surface (DESIGN §14.4).
        """
        res = self._graph(ws).ro_query(
            "CALL db.idx.fulltext.queryNodes('Message', $query) "
            "YIELD node AS m, score "
            "RETURN m.msgId AS msgId, m.threadId AS threadId, m.text AS text, "
            "m.createdAt AS createdAt, score "
            "ORDER BY score DESC LIMIT $limit",
            {"query": query, "limit": limit},
        )
        return [
            {
                "msgId": row[0], "threadId": row[1], "text": row[2],
                "createdAt": row[3], "score": row[4],
            }
            for row in res.result_set
        ]

    # ── §9 Read-cursors & since-reads ───────────────────────────────────────────

    @staticmethod
    def _since_row(row: list[Any]) -> dict[str, Any]:
        return {
            "msgId": row[0], "text": row[1], "role": row[2], "createdAt": row[3],
            "authorId": row[4], "authorType": row[5], "isMention": bool(row[6]),
            "threadId": row[7],
        }

    # Two predicate forms, one method (QUERIES.md §9.1/§9.2 v2):
    #   * `since_msg_id is None` → plain `m.createdAt > $since` — explicit-since
    #     semantics (may re-deliver or skip within that exact millisecond).
    #   * a string (possibly "") → formulation-A composite keyset, which mirrors
    #     the ORDER BY 1:1 and never skips a millisecond-tied sibling.
    # Both forms share the deterministic total order (createdAt, msgId).
    _SINCE_PLAIN = "m.createdAt > $since "
    _SINCE_KEYSET = (
        "m.createdAt > $since "
        "OR (m.createdAt = $since AND m.msgId > $sinceMsgId) "
    )

    def read_thread_since(
        self, ws: str, *, thread_id: str, me_id: str, since: int,
        since_msg_id: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Thread-scoped since-read; chronological, reader-mentions flagged. §9.1.

        Chronological `(createdAt, msgId)` order is the cursor-pagination
        invariant: a truncated page must be the earliest rows so advancing the
        cursor to the last delivered pair never skips anything (mention-first
        sorting + LIMIT loses messages). `isMention` stays a flag for the client.
        """
        where = self._SINCE_PLAIN if since_msg_id is None else self._SINCE_KEYSET
        res = self._graph(ws).ro_query(
            "MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message) "
            "MATCH (first)-[:NEXT*0..]->(m:Message) "
            "WHERE " + where +
            "MATCH (m)-[:POSTED_BY]->(author) "
            "OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) "
            "       WHERE me.userId = $meId OR me.agentId = $meId "
            "WITH m, author, count(me) > 0 AS isMention "
            "RETURN m.msgId, m.text, m.role, m.createdAt, "
            "coalesce(author.userId, author.agentId) AS authorId, "
            "labels(author) AS authorType, isMention, m.threadId AS threadId "
            "ORDER BY m.createdAt, m.msgId "
            "LIMIT $limit",
            {
                "threadId": thread_id, "meId": me_id, "since": since,
                "sinceMsgId": since_msg_id or "", "limit": limit,
            },
        )
        return [self._since_row(row) for row in res.result_set]

    def read_ws_since(
        self, ws: str, *, me_id: str, since: int,
        since_msg_id: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Workspace-wide since-read across all threads. §9.2.

        Anchors on the `Message.createdAt` index (formulation A folds the
        composite OR into the same index scan — re-profile on engine upgrades).
        Chronological (see `read_thread_since`); reader-mentions flagged.
        """
        where = self._SINCE_PLAIN if since_msg_id is None else self._SINCE_KEYSET
        res = self._graph(ws).ro_query(
            "MATCH (m:Message) "
            "WHERE " + where +
            "MATCH (m)-[:POSTED_BY]->(author) "
            "OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) "
            "       WHERE me.userId = $meId OR me.agentId = $meId "
            "WITH m, author, count(me) > 0 AS isMention "
            "RETURN m.msgId, m.text, m.role, m.createdAt, "
            "coalesce(author.userId, author.agentId) AS authorId, "
            "labels(author) AS authorType, isMention, m.threadId AS threadId "
            "ORDER BY m.createdAt, m.msgId "
            "LIMIT $limit",
            {
                "meId": me_id, "since": since,
                "sinceMsgId": since_msg_id or "", "limit": limit,
            },
        )
        return [self._since_row(row) for row in res.result_set]

    def advance_cursor(
        self, ws: str, *, me_id: str, thread_id: str, cursor_id: str,
        now: int, now_msg_id: str,
    ) -> tuple[int, str | None] | None:
        """MERGE/advance a read-cursor with the composite monotonic guard. §9.3 v2.

        The guard is computed once in a `WITH` so both `SET`s see the pre-write
        state; `(now, nowMsgId)` advances only when strictly ahead of the stored
        pair — a stale or tie-smaller replay never moves the cursor backward.
        `coalesce(rc.lastReadMsgId, '')` covers pre-K-007 cursors (no backfill
        needed). When the member node doesn't exist the anchor MATCH yields no
        rows — the advance is a no-op returning None (never an IndexError).
        """
        res = self._graph(ws).query(
            "MATCH (mem) WHERE mem.userId = $meId OR mem.agentId = $meId "
            "MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId: $cursorId}) "
            "ON CREATE SET rc.memberId = $meId, rc.threadId = $threadId "
            "WITH rc, ($now > coalesce(rc.lastReadAt, 0) "
            "      OR ($now = coalesce(rc.lastReadAt, 0) "
            "          AND $nowMsgId > coalesce(rc.lastReadMsgId, ''))) AS adv "
            "SET rc.lastReadAt    = CASE WHEN adv THEN $now      ELSE rc.lastReadAt    END, "
            "    rc.lastReadMsgId = CASE WHEN adv THEN $nowMsgId ELSE rc.lastReadMsgId END "
            "RETURN rc.lastReadAt, rc.lastReadMsgId",
            {
                "meId": me_id, "threadId": thread_id, "cursorId": cursor_id,
                "now": now, "nowMsgId": now_msg_id,
            },
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return (row[0], row[1])

    def get_cursor(self, ws: str, *, cursor_id: str) -> tuple[int, str | None] | None:
        """Read a cursor's `(lastReadAt, lastReadMsgId)` pair, or None. §9.4.

        Pre-K-007 cursors have no `lastReadMsgId` — the pair comes back as
        `(ts, None)` and the service maps it to the `''` base convention.
        """
        res = self._graph(ws).ro_query(
            "MATCH (rc:ReadCursor {cursorId: $cursorId}) "
            "RETURN rc.lastReadAt, rc.lastReadMsgId",
            {"cursorId": cursor_id},
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return (row[0], row[1])

    def get_message(self, ws: str, *, msg_id: str) -> dict[str, Any] | None:
        """Fetch a single message with author + quoted-id, or None. QUERIES.md §4."""
        res = self._graph(ws).ro_query(
            "MATCH (m:Message {msgId: $msgId}) "
            "OPTIONAL MATCH (m)-[:POSTED_BY]->(author) "
            "OPTIONAL MATCH (m)-[:REPLY_TO]->(quoted:Message) "
            "RETURN m.msgId, m.text, m.role, m.createdAt, "
            "coalesce(author.userId, author.agentId) AS authorId, "
            "labels(author) AS authorType, m.threadId AS threadId, "
            "quoted.msgId AS quotedId",
            {"msgId": msg_id},
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {
            "msgId": row[0], "text": row[1], "role": row[2], "createdAt": row[3],
            "authorId": row[4], "authorType": row[5], "threadId": row[6],
            "quotedId": row[7],
        }

    # ── validation reads (used by services) ─────────────────────────────────────

    def thread_exists(self, ws: str, *, thread_id: str) -> bool:
        """True if the thread exists in the workspace."""
        res = self._graph(ws).ro_query(
            "OPTIONAL MATCH (t:Thread {threadId: $threadId}) RETURN t IS NOT NULL",
            {"threadId": thread_id},
        )
        return bool(res.result_set and res.result_set[0][0])

    def channel_exists(self, ws: str, *, channel_id: str) -> bool:
        """True if the channel exists in the workspace."""
        res = self._graph(ws).ro_query(
            "OPTIONAL MATCH (c:Channel {channelId: $channelId}) "
            "RETURN c IS NOT NULL",
            {"channelId": channel_id},
        )
        return bool(res.result_set and res.result_set[0][0])

    def resolve_member_kinds(self, ws: str, *, ids: list[str]) -> dict[str, str | None]:
        """id -> 'User' | 'Agent' | None. QUERIES.md §2 (member-kind lookup).

        One round trip covers author validation, mention validation, and role
        derivation (`User → user`, `Agent → assistant` — mapping is service-side).
        """
        if not ids:
            return {}
        res = self._graph(ws).ro_query(
            "UNWIND $ids AS id "
            "OPTIONAL MATCH (u:User  {userId:  id}) "
            "OPTIONAL MATCH (a:Agent {agentId: id}) "
            "RETURN id, CASE WHEN coalesce(u, a) IS NULL THEN null "
            "                ELSE labels(coalesce(u, a))[0] END AS kind",
            {"ids": list(ids)},
        )
        return {row[0]: row[1] for row in res.result_set}
