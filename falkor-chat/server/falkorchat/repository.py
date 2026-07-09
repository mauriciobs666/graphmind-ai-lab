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

from . import config, db


class EmbeddingDimensionError(Exception):
    """Embedding length does not match the workspace's vector-index dimension.

    Guards the silent-corruption quirk (K-008, `docs/plans/m2-graphrag.md` item 2):
    a wrong-dimension `vecf32` write is **silently accepted** at `SET` (no engine
    error) but the node then **falls out of the ANN index** — permanently
    unretrievable by `hybrid_search`, with no way to detect it after the fact
    (no `vec.dimension()` on this build). `set_embedding` validates length
    client-side and raises this *before* the write. Re-exported by `services`.
    """


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

    # ── §10 Agent answer provenance — EMITTED (K-013) ───────────────────────────
    #
    # The AI responder posts its answer as the Agent (role derived `assistant`)
    # via the §4 write path, and records the §6 retrieval seeds that grounded it
    # as `(answer)-[:EMITTED {score, rank}]->(seed)` provenance edges written
    # INSIDE the same GRAPH.QUERY (atomicity — QUERIES.md §10). Both write paths
    # carry the seed block so the service can reuse the §4 dispatch loop verbatim
    # (subsequent → on None retry as first). `seeds` is ordered by rank; the three
    # §10.1 map params are derived positionally. `seedIds == []` is a verified
    # no-op (the CASE guard keeps the write alive), so the same shape serves
    # non-provenance writes. Build quirk: an EMITTED endpoint must be a bound node
    # (a map-projection cannot be a CREATE endpoint), so seeds are collected as
    # nodes and per-edge props come from map params keyed by the node's own msgId.

    @staticmethod
    def _seed_params(
        seeds: list[tuple[str, float]] | None,
    ) -> tuple[list[str], dict[str, float], dict[str, int]]:
        """Derive the three §10.1 seed params from the ranked `seeds` list.

        `seeds = [(msgId, score)]` in rank order → `seed_ids` (rank = position),
        `score_by = {msgId: score}`, `rank_by = {msgId: index}`.
        """
        ordered = list(seeds or [])
        seed_ids = [s[0] for s in ordered]
        score_by = {s[0]: s[1] for s in ordered}
        rank_by = {s[0]: i for i, s in enumerate(ordered)}
        return seed_ids, score_by, rank_by

    def post_agent_answer(
        self, ws: str, *, thread_id: str, msg_id: str, author_id: str,
        text: str, role: str, created_at: int,
        mentions: list[str] | None = None,
        seeds: list[tuple[str, float]] | None = None,
    ) -> MessageWriteStatus | None:
        """Agent answer with EMITTED provenance (subsequent path). QUERIES.md §10.1.

        The §4 subsequent write plus one added guarded UNWIND/FOREACH block that
        resolves seed msgIds to bound `Message` nodes and creates the `EMITTED`
        edges inside the same guard. Same status-row contract as §4 (None = no
        TAIL → retry as first). Mentions + seeds both ride inside the single
        GRAPH.QUERY (atomicity rule).
        """
        seed_ids, score_by, rank_by = self._seed_params(seeds)
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
            "UNWIND (CASE WHEN $seedIds = [] THEN [null] ELSE $seedIds END) AS sid "
            "OPTIONAL MATCH (s:Message {msgId: sid}) "
            "WITH t, tailRel, prev, dup, author, mems, collect(DISTINCT s) AS seeds "
            "WITH t, tailRel, prev, dup, author, mems, seeds, "
            "     (dup IS NULL AND author IS NOT NULL) AS ok "
            "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
            "  CREATE (m:Message {msgId: $msgId, text: $text, role: $role, "
            "                     createdAt: $createdAt, threadId: $threadId}) "
            "  CREATE (prev)-[:NEXT]->(m) "
            "  DELETE tailRel "
            "  CREATE (t)-[:TAIL]->(m) "
            "  CREATE (m)-[:POSTED_BY]->(author) "
            "  SET t.updatedAt = $createdAt "
            "  FOREACH (mem  IN mems  | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) "
            "  FOREACH (seed IN seeds | CREATE (m)-[:EMITTED "
            "    {score: $scoreBy[seed.msgId], rank: $rankBy[seed.msgId]}]->(seed)) "
            ") "
            "RETURN ok                 AS written, "
            "       false              AS hadHead, "
            "       dup  IS NOT NULL   AS dupMsg, "
            "       author IS NOT NULL AS authorFound",
            {
                "threadId": thread_id, "msgId": msg_id, "authorId": author_id,
                "text": text, "role": role, "createdAt": created_at,
                "mentions": list(mentions or []),
                "seedIds": seed_ids, "scoreBy": score_by, "rankBy": rank_by,
            },
        )
        return self._write_status(res)

    def post_agent_answer_first(
        self, ws: str, *, thread_id: str, msg_id: str, author_id: str,
        text: str, role: str, created_at: int,
        mentions: list[str] | None = None,
        seeds: list[tuple[str, float]] | None = None,
    ) -> MessageWriteStatus | None:
        """First-path agent answer with EMITTED provenance. QUERIES.md §10.1 note.

        The §4 first-message write (creates HEAD + TAIL) with the same seed block
        folded in — the defensive fallback the dispatch loop needs when the agent
        answers into a headless thread. Realistically the trigger message is the
        HEAD, so the subsequent path is the live one; both carry the block.
        """
        seed_ids, score_by, rank_by = self._seed_params(seeds)
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
            "UNWIND (CASE WHEN $seedIds = [] THEN [null] ELSE $seedIds END) AS sid "
            "OPTIONAL MATCH (s:Message {msgId: sid}) "
            "WITH t, h, dup, author, mems, collect(DISTINCT s) AS seeds "
            "WITH t, h, dup, author, mems, seeds, "
            "     (h IS NULL AND dup IS NULL AND author IS NOT NULL) AS ok "
            "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
            "  CREATE (m:Message {msgId: $msgId, text: $text, role: $role, "
            "                     createdAt: $createdAt, threadId: $threadId}) "
            "  CREATE (t)-[:HEAD]->(m) "
            "  CREATE (t)-[:TAIL]->(m) "
            "  CREATE (m)-[:POSTED_BY]->(author) "
            "  SET t.updatedAt = $createdAt "
            "  FOREACH (mem  IN mems  | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) "
            "  FOREACH (seed IN seeds | CREATE (m)-[:EMITTED "
            "    {score: $scoreBy[seed.msgId], rank: $rankBy[seed.msgId]}]->(seed)) "
            ") "
            "RETURN ok                 AS written, "
            "       h    IS NOT NULL   AS hadHead, "
            "       dup  IS NOT NULL   AS dupMsg, "
            "       author IS NOT NULL AS authorFound",
            {
                "threadId": thread_id, "msgId": msg_id, "authorId": author_id,
                "text": text, "role": role, "createdAt": created_at,
                "mentions": list(mentions or []),
                "seedIds": seed_ids, "scoreBy": score_by, "rankBy": rank_by,
            },
        )
        return self._write_status(res)

    def read_provenance(self, ws: str, *, msg_id: str) -> list[dict[str, Any]]:
        """An answer's cited seeds, ordered by rank (forward). QUERIES.md §10.2.

        Read path (`ro_query`). Anchored on the answer's `msgId` index, traverses
        `EMITTED` outward. Rows `{seedMsgId, text, role, score, rank}`.
        """
        res = self._graph(ws).ro_query(
            "MATCH (a:Message {msgId: $msgId})-[e:EMITTED]->(s:Message) "
            "RETURN s.msgId, s.text, s.role, e.score, e.rank "
            "ORDER BY e.rank",
            {"msgId": msg_id},
        )
        return [
            {
                "seedMsgId": row[0], "text": row[1], "role": row[2],
                "score": row[3], "rank": row[4],
            }
            for row in res.result_set
        ]

    def read_citing_answers(
        self, ws: str, *, seed_msg_id: str
    ) -> list[dict[str, Any]]:
        """Answers that cited a seed, newest-first (reverse). QUERIES.md §10.3.

        Read path (`ro_query`). Anchored on the seed's `msgId` index, traverses
        `EMITTED` inbound. Rows `{answerMsgId, role, createdAt, score, rank}`.
        """
        res = self._graph(ws).ro_query(
            "MATCH (a:Message)-[e:EMITTED]->(s:Message {msgId: $seedMsgId}) "
            "RETURN a.msgId, a.role, a.createdAt, e.score, e.rank "
            "ORDER BY a.createdAt DESC",
            {"seedMsgId": seed_msg_id},
        )
        return [
            {
                "answerMsgId": row[0], "role": row[1], "createdAt": row[2],
                "score": row[3], "rank": row[4],
            }
            for row in res.result_set
        ]

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

    # ── §6 GraphRAG hybrid retrieval ────────────────────────────────────────────

    def set_embedding(
        self, ws: str, *, msg_id: str, embedding: list[float],
        expected_dim: int | None = None,
    ) -> bool:
        """Set a message's embedding (async, after posting). QUERIES.md §6. Write path.

        Decoupled from the message write path — the message is readable before the
        embedding lands (DESIGN §9); this runs once per message from the embedding
        worker. `vecf32(...)` wraps the `$embedding` list parameter in the query.

        **Validates `len(embedding) == expected_dim` client-side before the SET**
        (default `config.EMBEDDING_DIM`). This is not optional: a wrong-dimension
        `vecf32` write is silently accepted by the engine but the node then drops
        out of the ANN index permanently (item-2 quirk) — raise
        `EmbeddingDimensionError` rather than let the message vanish.

        Returns True if the message existed and the embedding committed, False if
        no message matched `$msgId` (a silent no-op the caller may want to notice).
        """
        dim = config.EMBEDDING_DIM if expected_dim is None else expected_dim
        if len(embedding) != dim:
            raise EmbeddingDimensionError(
                f"embedding has {len(embedding)} dimensions, expected {dim} "
                f"(msgId={msg_id!r}) — a wrong-dimension vecf32 write is silently "
                f"accepted but drops the node out of the ANN index; refusing to write"
            )
        res = self._graph(ws).query(
            "MATCH (m:Message {msgId: $msgId}) SET m.embedding = vecf32($embedding)",
            {"msgId": msg_id, "embedding": list(embedding)},
        )
        return res.properties_set > 0

    def hybrid_search(
        self, ws: str, *, q_vec: list[float], k: int = 10, limit: int = 10,
        channel_id: str | None = None, timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """GraphRAG hybrid retrieval: vector ANN seed + scope traversal. QUERIES.md §6.

        Read path (routes via `ro_query`). Two variants mirror §6: channel-scoped
        (`channel_id` given → keep the `MATCH (c:Channel …)` line) vs workspace-wide
        (`channel_id` omitted → drop it). `vecf32(...)` wraps the `$qVec` list
        parameter, same as the SET.

        `score` is **cosine distance** (0 = identical); rows come back already
        `ORDER BY score ASC` (most similar first) — do not re-sort. `relatedContext`
        is `[]` in M2: the Entity co-occurrence expansion (`MENTIONS`→`Entity`, §6 —
        distinct from `MENTIONS_MEMBER`) is present but dormant until an extraction
        pipeline lands, so it no-ops cleanly. Passed through unchanged.

        `timeout` (ms) is a per-query client override for long GraphRAG reads
        (DESIGN §10 / K-007 posture) — the service supplies it, not this layer.
        ANN recall is approximate: kNN may return fewer than `k` — never treat
        "returns exactly k" as an invariant.
        """
        scoped = channel_id is not None
        cypher = (
            "CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) "
            "YIELD node AS seed, score "
            "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "
            + (
                "MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t) "
                if scoped else ""
            )
            + "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
            "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "
            "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, "
            "score, [m IN expanded | m.text] AS relatedContext "
            "ORDER BY score ASC "
            "LIMIT $limit"
        )
        params: dict[str, Any] = {"qVec": list(q_vec), "k": k, "limit": limit}
        if scoped:
            params["channelId"] = channel_id
        res = self._graph(ws).ro_query(cypher, params, timeout=timeout)
        return [
            {
                "msgId": row[0], "text": row[1], "role": row[2],
                "score": row[3], "relatedContext": row[4],
            }
            for row in res.result_set
        ]

    # ── §9 Read-cursors & since-reads ───────────────────────────────────────────

    @staticmethod
    def _since_row(row: list[Any]) -> dict[str, Any]:
        return {
            "msgId": row[0], "text": row[1], "role": row[2], "createdAt": row[3],
            "authorId": row[4], "authorType": row[5], "isMention": bool(row[6]),
            "threadId": row[7], "displayName": row[8],
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
            "labels(author) AS authorType, isMention, m.threadId AS threadId, "
            "author.displayName AS displayName "
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
            "labels(author) AS authorType, isMention, m.threadId AS threadId, "
            "author.displayName AS displayName "
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
