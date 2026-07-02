"""Data-access layer — the only place Cypher lives (DESIGN §14.2).

Each method maps 1:1 to a verified query in `docs/QUERIES.md`, always
parameterised, using `ro_query` for reads and `query` for writes, scoped to the
per-workspace graph via `db.workspace_graph`. No business logic lives here —
that is `services.py`'s job.
"""

from __future__ import annotations

from typing import Any

from falkordb import FalkorDB

from . import db


class Repository:
    """Cypher access over a FalkorDB connection.

    Every method takes the workspace id `ws` (graph key ``ws:{ws}``) so the
    single-tenant seam stays in `config`/`api`, never hardcoded here.
    """

    def __init__(self, conn: FalkorDB) -> None:
        self._conn = conn

    def _graph(self, ws: str):
        return db.workspace_graph(self._conn, ws)

    # ── §3 Channels ────────────────────────────────────────────────────────────

    def create_channel(
        self, ws: str, *, channel_id: str, name: str, created_at: int
    ) -> None:
        """Create a channel (idempotent MERGE). QUERIES.md §3."""
        self._graph(ws).query(
            "MERGE (c:Channel {channelId: $channelId}) "
            "ON CREATE SET c.name = $name, c.createdAt = $createdAt "
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
        """Create a thread under a channel (idempotent). QUERIES.md §3."""
        self._graph(ws).query(
            "MATCH (c:Channel {channelId: $channelId}) "
            "MERGE (t:Thread {threadId: $threadId}) "
            "ON CREATE SET t.title = $title, t.createdAt = $createdAt, "
            "t.updatedAt = $createdAt "
            "MERGE (c)-[:HAS_THREAD]->(t) "
            "RETURN t",
            {
                "channelId": channel_id,
                "threadId": thread_id,
                "title": title,
                "createdAt": created_at,
            },
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

    def ensure_user(
        self, ws: str, *, user_id: str, display_name: str | None = None,
        email: str | None = None,
    ) -> None:
        """Project a User into the workspace graph (idempotent). QUERIES.md §2."""
        self._graph(ws).query(
            "MERGE (u:User {userId: $userId}) "
            "ON CREATE SET u.displayName = $displayName, u.email = $email "
            "RETURN u",
            {"userId": user_id, "displayName": display_name, "email": email},
        )

    def ensure_agent(
        self, ws: str, *, agent_id: str, name: str | None = None,
        model: str | None = None, created_at: int | None = None,
    ) -> None:
        """Register an Agent in the workspace graph (idempotent). QUERIES.md §7."""
        self._graph(ws).query(
            "MERGE (a:Agent {agentId: $agentId}) "
            "ON CREATE SET a.name = $name, a.model = $model, a.createdAt = $createdAt "
            "RETURN a",
            {"agentId": agent_id, "name": name, "model": model, "createdAt": created_at},
        )

    # ── §4 Messages ─────────────────────────────────────────────────────────────

    # The atomic participant-mention block appended to BOTH write paths. The CASE
    # guard is required — a bare `UNWIND []` collapses the row stream and `RETURN m`
    # comes back empty (live-verified, AGENTS.md). `$mentions = []` is a true no-op.
    _MENTIONS_BLOCK = (
        "WITH m "
        "UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid "
        "OPTIONAL MATCH (u:User  {userId:  mid}) "
        "OPTIONAL MATCH (a:Agent {agentId: mid}) "
        "WITH m, collect(DISTINCT coalesce(u, a)) AS mems "
        "FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) "
        "RETURN m"
    )

    @staticmethod
    def _assert_written(res, *, thread_id: str, author_id: str) -> None:
        """Fail loudly when a §4 write query matched nothing.

        The write anchors on MATCH (thread / author / TAIL); if any anchor is
        missing the whole query no-ops with zero rows — silent message loss.
        The service validates first, so reaching this means a broken invariant
        (or a lost race on the thread pointers).
        """
        if not res.result_set:
            raise RuntimeError(
                "message write was a no-op — thread, author, or TAIL not found "
                f"(thread={thread_id!r}, author={author_id!r})"
            )

    def post_first_message(
        self, ws: str, *, thread_id: str, msg_id: str, author_id: str,
        text: str, role: str, created_at: int, mentions: list[str] | None = None,
    ) -> None:
        """First message in a thread — creates HEAD + TAIL. QUERIES.md §4.

        Mentions ride inside this single GRAPH.QUERY (atomicity rule).
        """
        res = self._graph(ws).query(
            "MATCH (t:Thread {threadId: $threadId}) "
            "MATCH (author {userId: $authorId}) "
            "MERGE (m:Message {msgId: $msgId}) "
            "ON CREATE SET m.text = $text, m.role = $role, m.createdAt = $createdAt "
            "CREATE (t)-[:HEAD]->(m) "
            "CREATE (t)-[:TAIL]->(m) "
            "CREATE (m)-[:POSTED_BY]->(author) "
            "SET t.updatedAt = $createdAt " + self._MENTIONS_BLOCK,
            {
                "threadId": thread_id, "msgId": msg_id, "authorId": author_id,
                "text": text, "role": role, "createdAt": created_at,
                "mentions": list(mentions or []),
            },
        )
        self._assert_written(res, thread_id=thread_id, author_id=author_id)

    def post_subsequent_message(
        self, ws: str, *, thread_id: str, msg_id: str, author_id: str,
        text: str, role: str, created_at: int, mentions: list[str] | None = None,
    ) -> None:
        """Append a message — moves TAIL forward via NEXT. QUERIES.md §4.

        Mentions ride inside this single GRAPH.QUERY (atomicity rule).
        """
        res = self._graph(ws).query(
            "MATCH (t:Thread {threadId: $threadId})-[tailRel:TAIL]->(prev:Message) "
            "MATCH (author {userId: $authorId}) "
            "MERGE (m:Message {msgId: $msgId}) "
            "ON CREATE SET m.text = $text, m.role = $role, m.createdAt = $createdAt "
            "CREATE (prev)-[:NEXT]->(m) "
            "DELETE tailRel "
            "CREATE (t)-[:TAIL]->(m) "
            "CREATE (m)-[:POSTED_BY]->(author) "
            "SET t.updatedAt = $createdAt " + self._MENTIONS_BLOCK,
            {
                "threadId": thread_id, "msgId": msg_id, "authorId": author_id,
                "text": text, "role": role, "createdAt": created_at,
                "mentions": list(mentions or []),
            },
        )
        self._assert_written(res, thread_id=thread_id, author_id=author_id)

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
            "RETURN m.msgId AS msgId, m.text AS text, "
            "m.createdAt AS createdAt, score "
            "ORDER BY score DESC LIMIT $limit",
            {"query": query, "limit": limit},
        )
        return [
            {"msgId": row[0], "text": row[1], "createdAt": row[2], "score": row[3]}
            for row in res.result_set
        ]

    # ── §9 Read-cursors & since-reads ───────────────────────────────────────────

    @staticmethod
    def _since_row(row: list[Any]) -> dict[str, Any]:
        return {
            "msgId": row[0], "text": row[1], "role": row[2], "createdAt": row[3],
            "authorId": row[4], "authorType": row[5], "isMention": bool(row[6]),
        }

    def read_thread_since(
        self, ws: str, *, thread_id: str, me_id: str, since: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Thread-scoped since-read; chronological, reader-mentions flagged. §9.1.

        Chronological order is the cursor-pagination invariant: a truncated
        page must be the earliest rows so advancing the cursor to the last
        delivered `createdAt` never skips anything (mention-first sorting +
        LIMIT loses messages). `isMention` stays a flag for the client.
        """
        res = self._graph(ws).ro_query(
            "MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message) "
            "MATCH (first)-[:NEXT*0..]->(m:Message) "
            "WHERE m.createdAt > $since "
            "MATCH (m)-[:POSTED_BY]->(author) "
            "OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) "
            "       WHERE me.userId = $meId OR me.agentId = $meId "
            "WITH m, author, count(me) > 0 AS isMention "
            "RETURN m.msgId, m.text, m.role, m.createdAt, "
            "coalesce(author.userId, author.agentId) AS authorId, "
            "labels(author) AS authorType, isMention "
            "ORDER BY m.createdAt "
            "LIMIT $limit",
            {"threadId": thread_id, "meId": me_id, "since": since, "limit": limit},
        )
        return [self._since_row(row) for row in res.result_set]

    def read_ws_since(
        self, ws: str, *, me_id: str, since: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Workspace-wide since-read across all threads. §9.2.

        Anchors on the `Message.createdAt` index. Chronological (see
        `read_thread_since`); reader-mentions flagged via `isMention`.
        """
        res = self._graph(ws).ro_query(
            "MATCH (m:Message) WHERE m.createdAt > $since "
            "MATCH (m)-[:POSTED_BY]->(author) "
            "OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) "
            "       WHERE me.userId = $meId OR me.agentId = $meId "
            "WITH m, author, count(me) > 0 AS isMention "
            "RETURN m.msgId, m.text, m.role, m.createdAt, "
            "coalesce(author.userId, author.agentId) AS authorId, "
            "labels(author) AS authorType, isMention "
            "ORDER BY m.createdAt "
            "LIMIT $limit",
            {"meId": me_id, "since": since, "limit": limit},
        )
        return [self._since_row(row) for row in res.result_set]

    def advance_cursor(
        self, ws: str, *, me_id: str, thread_id: str, cursor_id: str, now: int
    ) -> int | None:
        """MERGE/advance a read-cursor monotonically; returns lastReadAt. §9.3.

        The CASE guard makes advancing never move the cursor backward
        (live-verified on this build). When the member node doesn't exist the
        anchor MATCH yields no rows — the advance is a no-op returning None
        (never an IndexError).
        """
        res = self._graph(ws).query(
            "MATCH (mem) WHERE mem.userId = $meId OR mem.agentId = $meId "
            "MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId: $cursorId}) "
            "ON CREATE SET rc.memberId = $meId, rc.threadId = $threadId "
            "SET rc.lastReadAt = CASE WHEN $now > coalesce(rc.lastReadAt, 0) "
            "THEN $now ELSE rc.lastReadAt END "
            "RETURN rc.lastReadAt",
            {"meId": me_id, "threadId": thread_id, "cursorId": cursor_id, "now": now},
        )
        return res.result_set[0][0] if res.result_set else None

    def get_cursor(self, ws: str, *, cursor_id: str) -> int | None:
        """Read a cursor's lastReadAt, or None if it doesn't exist yet. §9.4."""
        res = self._graph(ws).ro_query(
            "MATCH (rc:ReadCursor {cursorId: $cursorId}) RETURN rc.lastReadAt",
            {"cursorId": cursor_id},
        )
        return res.result_set[0][0] if res.result_set else None

    def get_message(self, ws: str, *, msg_id: str) -> dict[str, Any] | None:
        """Fetch a single message with author + quoted-id, or None. QUERIES.md §4."""
        res = self._graph(ws).ro_query(
            "MATCH (m:Message {msgId: $msgId}) "
            "OPTIONAL MATCH (m)-[:POSTED_BY]->(author) "
            "OPTIONAL MATCH (m)-[:REPLY_TO]->(quoted:Message) "
            "RETURN m.msgId, m.text, m.role, m.createdAt, "
            "coalesce(author.userId, author.agentId) AS authorId, "
            "labels(author) AS authorType, quoted.msgId AS quotedId",
            {"msgId": msg_id},
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {
            "msgId": row[0], "text": row[1], "role": row[2], "createdAt": row[3],
            "authorId": row[4], "authorType": row[5], "quotedId": row[6],
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

    def existing_members(self, ws: str, *, ids: list[str]) -> set[str]:
        """Subset of `ids` that resolve to a User or Agent in the workspace.

        Used by the service to reject mentions of non-members before writing.
        """
        if not ids:
            return set()
        res = self._graph(ws).ro_query(
            "UNWIND $ids AS mid "
            "OPTIONAL MATCH (u:User  {userId:  mid}) "
            "OPTIONAL MATCH (a:Agent {agentId: mid}) "
            "WITH mid, coalesce(u, a) AS m WHERE m IS NOT NULL "
            "RETURN collect(DISTINCT mid) AS ids",
            {"ids": list(ids)},
        )
        return set(res.result_set[0][0]) if res.result_set else set()
