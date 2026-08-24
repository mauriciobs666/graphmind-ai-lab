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
from redis.exceptions import ResponseError

from . import config, db


class EmbeddingDimensionError(Exception):
    """Embedding length does not match the workspace's vector-index dimension.

    Guards the silent-corruption quirk (K-008, `docs/archive/plans/m2-graphrag.md` item 2):
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


class WorkflowDefNotFoundError(Exception):
    """A workflow def version was not found in the `reference` graph (M3 §11).

    Raised by the service `materialize_def` when the `(key, version)` to
    materialize has never been published — nothing is written to the workspace.
    Re-exported by `services` (defined here to avoid a repository→services cycle).
    """


class WorkflowDefSpecError(Exception):
    """A workflow-def spec failed validation before any write (M3 §11 / plan §B5).

    Raised by the service `publish_workflow_def` when the spec is malformed —
    unknown `kind`/step `type`, duplicate step keys, a `start_key` that resolves
    to no declared step, or a transition endpoint that is not a declared step.
    Validation runs *before* the repository write, so an invalid spec writes
    nothing. Re-exported by `services`.
    """


class WorkflowDefConflictError(Exception):
    """A re-publish or re-materialize would change the topology of an existing
    `(key, version)` (M3 §11 / K-034).

    Raised by `services.publish_workflow_def` (against `reference`) and
    `services.materialize_def` (against `ctx.ws`'s snapshot) when the incoming
    step-key set, transition-identity set `(from,to,on,order)`, or start key
    differs from what's already stored — before any repository write. A
    property-only difference (`name`, `kind`, a step's `type`/`config`, a
    transition's `guard`) is NOT a conflict; that stays create-only-on-properties
    (K-031-pinned). Nothing is written when this is raised. Re-exported by
    `services`.

    Both `publish_workflow_def` and `materialize_def` read-then-decide-then-write
    with no atomicity between the read and the write (the same residual TOCTOU
    shape as any check-then-act over two round trips): two concurrent callers
    racing a **brand-new** `(key, version)` with differing content can both
    observe "nothing stored yet" and both proceed to write. This is a
    pre-existing, out-of-scope hazard (unrelated to and not worsened by this
    gate) — for an *existing* `(key, version)`, a topology-differing candidate
    is always rejected regardless of read timing.
    """


class WorkflowRunNotFoundError(Exception):
    """A workflow run (or its current-position anchor) was not found (M3 §12).

    Raised by the service/executor when a run state-move query returns zero rows
    for a reason other than a guarded CAS miss — the run id does not exist, or it
    has no `AT_STEP` (already terminal) when an advance was attempted. Re-exported
    by `services` (defined here to avoid a repository→services cycle).
    """


class WorkflowRunNotWaitingError(Exception):
    """Input was submitted to a run that is not parked (`waiting`) (K-024 D-G).

    The §12.13 resume-with-ctx CAS returns zero rows for **two** reasons — the run
    does not exist, or it exists but is not `waiting` — and the query cannot tell
    them apart. The service splits them with a prior `get_run`: absent →
    `WorkflowRunNotFoundError` (404); present-but-not-waiting (or the CAS lost to a
    concurrent submitter) → this (409). Re-exported by `services`.
    """


class WorkflowInputRejectedError(Exception):
    """Submitted run input (or a start `ctx`) failed validation (K-024 D-H/M-2).

    Raised **before anything is written and before any step budget is consumed**:
    a reserved key (`threadId`/`error`), a key the parked step does not declare, a
    value outside the step's `config.expects` list, or a merged ctx over the size
    bound. 400. Re-exported by `services`.
    """


class StepBudgetExceededError(Exception):
    """A run exceeded its `maxSteps` budget (M3 §7 / §12.5).

    The executor's runaway guard: `record_step_and_advance` bumps `stepCount`, and
    when it passes the run's `maxSteps` the executor fails the run and surfaces this
    (the run is stamped `failed` via `fail_run`). Re-exported by `services`.
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


@dataclass(frozen=True)
class DocumentWriteStatus:
    """Status row returned by `create_document` (§14.1 — K-050 M5 Stage 1).

    Mirrors `MessageWriteStatus`'s "always return a status row, let the
    service decide what to do with it" shape. ``written`` is False only when
    ``ingestor_found`` is False — the actor guard is the only thing that can
    make this write a no-op (no HEAD/TAIL race here, §2.4).
    """

    written: bool
    ingestor_found: bool


class Repository:
    """Cypher access over a FalkorDB connection.

    Every method takes the workspace id `ws` (graph key ``ws:{ws}``) so the
    single-tenant seam stays in `config`/`api`, never hardcoded here.
    """

    def __init__(self, conn: FalkorDB | db.LazyFalkorDB) -> None:
        self._conn = conn

    def _graph(self, ws: str):
        return db.workspace_graph(self._conn, ws)

    def _reference(self):
        """Select the global `reference` graph (workflow def templates, §11)."""
        return db.reference_graph(self._conn)

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

    def read_index_dimension(
        self, ws: str, *, label: str, prop: str = "embedding"
    ) -> int | None:
        """Introspect `ws:{ws}`'s vector-index dimension for `label.prop` (FR-19,
        `-graph.md` §3.2). QUERIES.md §6 (new subsection).

        Returns the configured dimension, or `None` for every "no vector index"
        edge case — the four `-graph.md` §3.2 edge behaviours collapse to exactly
        two outcomes here:

        - **a real int** — the vector index exists (dimension is index metadata,
          not data, so this is `True` even with zero vectors written).
        - **`None`** — covers all three "there is no vector index to compare
          against" cases, deliberately not distinguished further (the caller's
          only correct response to any of them is "refuse"):
            1. the label has only non-vector (e.g. `RANGE`) indexes → one row,
               `dim` column `NULL`;
            2. the label is unknown to this graph → zero rows;
            3. the graph key `ws:{ws}` does not exist at all → FalkorDB raises
               `ERR Invalid graph operation on empty key` rather than returning
               rows — caught here and folded into the same `None` outcome.

        Routed via `ro_query` (never a write) so a nonexistent graph key is never
        implicitly created as a side effect of this check — `ro_query`/`GRAPH.RO_QUERY`
        raising the "empty key" error (rather than silently creating the graph) is
        exactly what makes case 3 distinguishable from case 2 at all, and is the same
        behaviour `services._read_or_absent` already relies on for the reference/
        snapshot "cold graph key" probe.
        """
        try:
            res = self._graph(ws).ro_query(
                "CALL db.indexes() YIELD label, types, options "
                "WHERE label = $label AND types[$prop] = ['VECTOR'] "
                "RETURN options[$prop].dimension AS dim",
                {"label": label, "prop": prop},
            )
        except ResponseError as exc:
            if "empty key" in str(exc):
                return None
            raise
        if not res.result_set:
            return None
        return res.result_set[0][0]

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

    # ── §14 Documents & Chunks (K-050 M5 Stage 1) ────────────────────────────────

    def create_document(
        self, ws: str, *, document_id: str, title: str, text: str,
        source_format: str, ingested_by: str, created_at: int,
        chunks: list[dict[str, Any]],
    ) -> DocumentWriteStatus:
        """Create a `Document` + all its `Chunk`s + `HAS_CHUNK` edges. QUERIES.md §14.1.

        One `GRAPH.QUERY`, per `document-ingestion-graph.md` §2.1: the actor
        (`ingested_by`) is resolved against `User`/`Agent` the same
        `coalesce(u, a)` way `post_first_message` resolves an author, and
        `Document.sourceKind` is derived server-side from *which* label
        resolved — never trusted from the caller (same posture as
        `Message.role`). Plain guarded `CREATE`, no `dup` guard: a retried
        call mints a second `Document` (§2.4's deliberate non-idempotent
        posture, matching the channel/thread-creation precedent). `chunks` is
        a list of `{chunkId, text, seq}` maps, pre-split app-side
        (`chunking.split_into_chunks`).
        """
        res = self._graph(ws).query(
            "OPTIONAL MATCH (u:User  {userId:  $ingestedBy}) "
            "OPTIONAL MATCH (a:Agent {agentId: $ingestedBy}) "
            "WITH u, a, coalesce(u, a) AS ingestor, (coalesce(u, a) IS NOT NULL) AS ok "
            "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
            "  CREATE (d:Document {"
            "    documentId: $documentId, title: $title, text: $text, "
            "    sourceFormat: $sourceFormat, "
            "    sourceKind: CASE WHEN u IS NOT NULL THEN 'document' ELSE 'agent' END, "
            "    status: 'processing', createdAt: $createdAt"
            "  }) "
            "  CREATE (d)-[:INGESTED_BY]->(ingestor) "
            "  FOREACH (ch IN $chunks | "
            "    CREATE (d)-[:HAS_CHUNK]->(:Chunk {"
            "      chunkId: ch.chunkId, text: ch.text, seq: ch.seq, documentId: $documentId"
            "    })"
            "  )"
            ") "
            "RETURN ok AS written, ingestor IS NOT NULL AS ingestorFound",
            {
                "documentId": document_id, "title": title, "text": text,
                "sourceFormat": source_format, "ingestedBy": ingested_by,
                "createdAt": created_at, "chunks": chunks,
            },
        )
        row = res.result_set[0]
        return DocumentWriteStatus(written=bool(row[0]), ingestor_found=bool(row[1]))

    def get_document(self, ws: str, *, document_id: str) -> dict[str, Any] | None:
        """Fetch a document (verbatim `text`, AC-9) + its ingesting actor. §14.2."""
        res = self._graph(ws).ro_query(
            "MATCH (d:Document {documentId: $documentId}) "
            "OPTIONAL MATCH (d)-[:INGESTED_BY]->(actor) "
            "RETURN d.documentId AS documentId, d.title AS title, d.text AS text, "
            "d.sourceFormat AS sourceFormat, d.sourceKind AS sourceKind, "
            "d.status AS status, d.createdAt AS createdAt, "
            "labels(actor)[0] AS ingestedByKind, "
            "coalesce(actor.userId, actor.agentId) AS ingestedById",
            {"documentId": document_id},
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {
            "documentId": row[0], "title": row[1], "text": row[2],
            "sourceFormat": row[3], "sourceKind": row[4], "status": row[5],
            "createdAt": row[6], "ingestedByKind": row[7], "ingestedById": row[8],
        }

    def list_document_chunks(
        self, ws: str, *, document_id: str
    ) -> list[dict[str, Any]]:
        """List a document's chunks in order (`chunkId` + `text` only).

        K-050 M5 Stage 2: the internal seam `services.ingest_document`'s caller
        (`api.py`/`mcp.py`) uses to schedule background chunk embedding right
        after a document is written — not part of the public MCP/REST surface
        (mirrors `posted["text"]` riding `post_message`'s own return for the
        same purpose at message scale; a document's chunks are read back
        instead of echoed in the receipt to keep `ingest_document`'s response
        at the documented `{documentId, chunkCount, status}` shape, QUERIES.md §14.4,
        rather than doubling a up-to-500,000-char payload back into the
        response body). Always traversed from an already-anchored `Document`,
        never independently scanned — no new index needed, same posture as
        `ABOUT`/`RELATES_TO` (`document-ingestion-graph.md` §2.2).
        """
        res = self._graph(ws).ro_query(
            "MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN c.chunkId AS chunkId, c.text AS text "
            "ORDER BY c.seq",
            {"documentId": document_id},
        )
        return [{"chunkId": row[0], "text": row[1]} for row in res.result_set]

    def set_chunk_embedding(
        self, ws: str, *, chunk_id: str, embedding: list[float],
        expected_dim: int | None = None,
    ) -> bool:
        """Set a chunk's embedding (async, after ingestion). QUERIES.md §6.

        K-050 M5 Stage 2: mirrors `set_embedding` exactly, `Chunk` in place of
        `Message` — the same decoupled-from-the-write-path posture (a chunk is
        readable before its embedding lands, plan §4 Stage 2) and the same
        pre-write dimension validation (a wrong-dimension `vecf32` write is
        silently accepted by the engine but the node then drops out of the ANN
        index permanently — item-2 quirk, applies identically to `Chunk`).

        Returns True if the chunk existed and the embedding committed, False if
        no chunk matched `$chunkId`.
        """
        dim = config.EMBEDDING_DIM if expected_dim is None else expected_dim
        if len(embedding) != dim:
            raise EmbeddingDimensionError(
                f"embedding has {len(embedding)} dimensions, expected {dim} "
                f"(chunkId={chunk_id!r}) — a wrong-dimension vecf32 write is "
                f"silently accepted but drops the node out of the ANN index; "
                f"refusing to write"
            )
        res = self._graph(ws).query(
            "MATCH (c:Chunk {chunkId: $chunkId}) SET c.embedding = vecf32($embedding)",
            {"chunkId": chunk_id, "embedding": list(embedding)},
        )
        return res.properties_set > 0

    def search_chunks(
        self, ws: str, *, q_vec: list[float], k: int = 10, limit: int = 10,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """`Chunk`-only ANN retrieval — the standalone KB search surface (FR-3,
        plan §3.7). K-050 M5 Stage 2. Read path (`ro_query`).

        Mirrors `hybrid_search`'s `CALL db.idx.vector.queryNodes(...) YIELD ...`
        shape, `Chunk` in place of `Message` — but with no scope traversal (a
        chunk has no Thread/Channel to scope by, unlike a message) and no
        Entity co-occurrence expansion (`ABOUT` is dormant until extraction,
        Stage 3 — an `OPTIONAL MATCH` on it here would just no-op today, so it
        is deferred rather than added speculatively). `documentId` rides
        denormalized on `Chunk` itself (§3.2), so the source document is
        reported with no extra hop.

        `score` is cosine distance (0 = identical) — rows come back already
        `ORDER BY score ASC` (most similar first), same convention as
        `hybrid_search`; do not re-sort. `timeout` (ms) is a per-query client
        override, same posture as `hybrid_search`'s.
        """
        res = self._graph(ws).ro_query(
            "CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec)) "
            "YIELD node AS seed, score "
            "RETURN seed.chunkId AS chunkId, seed.text AS text, "
            "seed.documentId AS documentId, seed.seq AS seq, score "
            "ORDER BY score ASC "
            "LIMIT $limit",
            {"qVec": list(q_vec), "k": k, "limit": limit},
            timeout=timeout,
        )
        return [
            {
                "chunkId": row[0], "text": row[1], "documentId": row[2],
                "seq": row[3], "score": row[4],
            }
            for row in res.result_set
        ]

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

    def list_thread_participants(
        self, ws: str, *, thread_id: str
    ) -> list[dict[str, Any]]:
        """A thread's participants = its parent channel's roster (RO). QUERIES.md
        §2 "List thread participants" (K-036 — web-api-coverage FR-8).

        Anchors on `Node By Index Scan | (t:Thread)` (`Thread.threadId`), one
        backward `HAS_THREAD` hop to the channel, one forward `MEMBER_OF` hop
        per member — no label scan (live-verified `GRAPH.PROFILE`, QUERIES.md
        §2). `type` is `labels(u)` (mirrors `read_thread`'s `authorType`
        convention — the kind-string derivation is the caller's job, same as
        everywhere else this repository returns raw `labels()`). Empty list
        when the thread's channel has zero members — not an error (the
        demo-seed-timing edge case QUERIES.md §2 calls out).
        """
        res = self._graph(ws).ro_query(
            "MATCH (c:Channel)-[:HAS_THREAD]->(t:Thread {threadId: $threadId}) "
            "MATCH (u)-[:MEMBER_OF]->(c) "
            "RETURN coalesce(u.userId, u.agentId) AS memberId, "
            "       u.displayName                 AS displayName, "
            "       labels(u)                     AS type "
            "ORDER BY u.displayName",
            {"threadId": thread_id},
        )
        return [
            {"memberId": row[0], "displayName": row[1], "type": row[2]}
            for row in res.result_set
        ]

    # ── §11 Workflow definitions & snapshots (M3 Slice 1) ────────────────────────
    #
    # Definitions live canonically in the GLOBAL `reference` graph as versioned
    # WorkflowDef templates; materializing copies a def@version into a
    # workspace as a local WorkflowDefSnapshot subgraph. Reference-scoped methods
    # take NO `ws` (plan F3); only materialization consumes a workspace. Every
    # node MERGE is backed by a UNIQUE constraint (`WorkflowDef {key,version}` /
    # `WorkflowDefSnapshot {key,version}` / `Step {stepUid}`). `config`/`guard`
    # are opaque strings, stored and returned verbatim (rule 8). Each method maps
    # 1:1 to a verified QUERIES.md §11 query.

    # Publish/materialize share the same MERGE shape (§11.1 / §11.4), differing
    # only in the root label and the target graph. Create-only on properties,
    # topology-enforced by services (K-034) — see this class's docstrings, not
    # "idempotent" unconditionally. The two sequential UNWIND blocks each
    # collapse via `count(...)` so the RETURN is one status row
    # (a naive shape row-multiplies steps × transitions).
    _PUBLISH_CYPHER = (
        "MERGE (d:{label} {{key: $key, version: $version}}) "
        "  ON CREATE SET d.name = $name, d.kind = $kind "
        "WITH d "
        "UNWIND $steps AS s "
        "  MERGE (st:Step {{stepUid: $key + ':' + $version + ':' + s.key}}) "
        "    ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config "
        "  MERGE (d)-[:HAS_STEP]->(st) "
        "WITH d, count(st) AS stepCount "
        "MATCH (start:Step {{stepUid: $key + ':' + $version + ':' + $startKey}}) "
        "MERGE (d)-[:START]->(start) "
        "WITH d, stepCount "
        "UNWIND $transitions AS tr "
        "  MATCH (from:Step {{stepUid: $key + ':' + $version + ':' + tr.from}}) "
        "  MATCH (to:Step   {{stepUid: $key + ':' + $version + ':' + tr.to}}) "
        "  MERGE (from)-[rel:TRANSITION {{on: tr.on, order: tr.order}}]->(to) "
        "    ON CREATE SET rel.guard = tr.guard "
        "WITH d, stepCount, count(rel) AS transitionCount "
        "RETURN d.key AS key, d.version AS version, stepCount, transitionCount"
    )

    # Subgraph read = two focused, index-anchored queries (§11.2 / §11.5): meta +
    # steps (via HAS_STEP), then transitions. No `length(path)` ordering (F6) —
    # the app reconstructs order from `TRANSITION.order`/topology.
    _READ_META_CYPHER = (
        "MATCH (d:{label} {{key: $key, version: $version}}) "
        "OPTIONAL MATCH (d)-[:START]->(start:Step) "
        "OPTIONAL MATCH (d)-[:HAS_STEP]->(s:Step) "
        "RETURN d.name AS name, d.kind AS kind, start.key AS startKey, "
        "       collect(DISTINCT {{key: s.key, type: s.type, config: s.config}}) AS steps"
    )
    _READ_TRANSITIONS_CYPHER = (
        "MATCH (d:{label} {{key: $key, version: $version}})"
        "-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step) "
        "RETURN collect({{from: from.key, to: to.key, on: tr.on, "
        "                 guard: tr.guard, order: tr.order}}) AS transitions"
    )

    @staticmethod
    def _read_subgraph(graph, *, label: str, key: str, version: str):
        """Read a def/snapshot subgraph (shared by def-read and snapshot-read).

        `None` when the root node is absent (the meta `MATCH` yields no rows).
        Otherwise `{name, kind, start_key, steps, transitions}`.
        """
        meta = graph.ro_query(
            Repository._READ_META_CYPHER.format(label=label),
            {"key": key, "version": version},
        )
        if not meta.result_set:
            return None
        name, kind, start_key, steps = meta.result_set[0]
        trs = graph.ro_query(
            Repository._READ_TRANSITIONS_CYPHER.format(label=label),
            {"key": key, "version": version},
        )
        transitions = trs.result_set[0][0] if trs.result_set else []
        return {
            "name": name, "kind": kind, "start_key": start_key,
            "steps": list(steps), "transitions": list(transitions),
        }

    @staticmethod
    def _read_structure(graph, *, label: str, key: str, version: str):
        """Observability read of a def/snapshot subgraph — **all** meta rows (K-031).

        Runs the *same* `_READ_META_CYPHER`/`_READ_TRANSITIONS_CYPHER` constants
        as `_read_subgraph` (§11.2 / §11.5) — no new or modified Cypher.

        **Why this duplicates `_read_subgraph`** (~12 lines, deliberate): that
        helper feeds `materialize_def` and the SHA-locked executor
        (`get_snapshot` → `executor._drive_loop`) and must not move. This one
        serves the read-only structure/diff endpoints, which need a different
        shape and a different row discipline.

        **Why it reads every row, not `result_set[0]`.** `_READ_META_CYPHER`
        returns `start.key` as a non-aggregated grouping key beside
        `collect(DISTINCT …)`, so the one-row collapse QUERIES.md §11.2 documents
        holds only *because* `start.key` is constant across the fan-out. A root
        carrying more than one `START` edge breaks that premise and yields **one
        row per start key** (verified live, K-031 V-1: two `START` edges → two
        meta rows), of which `_read_subgraph` takes an arbitrary one. How a root
        acquires a second `START` edge is **K-034**'s (a re-publish is create-only
        on properties but *additive* on structure); detecting it is this method's.

        **Why the returned shape differs from §11.2's documented contract.**
        §11.2 documents a scalar `startKey`; this returns `start_keys:
        list[str]` — a deliberate, documented stretch of the "`repository.py` is
        a 1:1 mirror of QUERIES.md" rule (DESIGN §14.2), recorded from the query
        side in the **QUERIES.md §11.2 multi-`START` note** (mirrored at §11.5).

        `None` when the root node is absent. Otherwise
        `{name, kind, start_keys, steps, transitions}`.
        """
        meta = graph.ro_query(
            Repository._READ_META_CYPHER.format(label=label),
            {"key": key, "version": version},
        )
        if not meta.result_set:
            return None
        name, kind = meta.result_set[0][0], meta.result_set[0][1]
        start_keys: list[str] = []
        steps: list[dict[str, Any]] = []
        seen_steps: set[str] = set()
        for row in meta.result_set:
            if row[2] is not None and row[2] not in start_keys:
                start_keys.append(row[2])
            for step in row[3]:
                if step["key"] is not None and step["key"] not in seen_steps:
                    seen_steps.add(step["key"])
                    steps.append(dict(step))
        trs = graph.ro_query(
            Repository._READ_TRANSITIONS_CYPHER.format(label=label),
            {"key": key, "version": version},
        )
        transitions = trs.result_set[0][0] if trs.result_set else []
        return {
            "name": name, "kind": kind, "start_keys": start_keys,
            "steps": steps, "transitions": [dict(t) for t in transitions],
        }

    def publish_def(
        self, *, key: str, version: str, name: str, kind: str, start_key: str,
        steps: list[dict[str, Any]], transitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Publish a def version into the `reference` graph. §11.1.

        Write path. Every node MERGE is constraint-backed. **This method itself
        does not enforce structural immutability** — a differing payload for an
        existing `(key, version)` mints parallel `TRANSITION`/`START` structure
        (K-034; see `test_publish_def_direct_call_is_unsafe_mints_parallel_
        transition_on_retarget`). The guarantee is enforced by the caller:
        `services.publish_workflow_def` rejects a topology-differing re-publish
        (`WorkflowDefConflictError`, 409) *before* calling this method — same as
        `_validate_def_spec`, which is also run there, before this call, so the
        inner `MATCH (start …)`/`MATCH (from/to …)` always resolve for a valid
        spec. Returns `{key, version, stepCount, transitionCount}`.
        """
        res = self._reference().query(
            self._PUBLISH_CYPHER.format(label="WorkflowDef"),
            {
                "key": key, "version": version, "name": name, "kind": kind,
                "startKey": start_key, "steps": list(steps),
                "transitions": list(transitions),
            },
        )
        row = res.result_set[0]
        return {
            "key": row[0], "version": row[1],
            "stepCount": row[2], "transitionCount": row[3],
        }

    def read_def_subgraph(self, *, key: str, version: str) -> dict[str, Any] | None:
        """Read a def's full subgraph from `reference` — the materialize input. §11.2.

        `None` when the def version is absent. `{name, kind, start_key, steps,
        transitions}`; `steps`/`transitions` are unordered (F6). Read path.
        """
        return self._read_subgraph(
            self._reference(), label="WorkflowDef", key=key, version=version
        )

    def read_def_structure(self, *, key: str, version: str) -> dict[str, Any] | None:
        """Read a def's full structure from `reference` for observability. §11.2.

        Same query constants as `read_def_subgraph`, multi-`START`-aware shape
        (`start_keys: list[str]`) — see `_read_structure`. `None` when absent.
        Read path.
        """
        return self._read_structure(
            self._reference(), label="WorkflowDef", key=key, version=version
        )

    def get_def(self, *, key: str, version: str | None = None) -> dict[str, Any] | None:
        """Get a def's metadata (latest version if `version` is None). §11.3.

        `{key, version, name, kind}` or `None` if absent. Read path.
        """
        graph = self._reference()
        if version is None:
            res = graph.ro_query(
                "MATCH (d:WorkflowDef {key: $key}) "
                "RETURN d.key AS key, d.version AS version, d.name AS name, "
                "       d.kind AS kind "
                "ORDER BY d.version DESC LIMIT 1",
                {"key": key},
            )
        else:
            res = graph.ro_query(
                "MATCH (d:WorkflowDef {key: $key, version: $version}) "
                "RETURN d.key AS key, d.version AS version, d.name AS name, "
                "       d.kind AS kind",
                {"key": key, "version": version},
            )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {"key": row[0], "version": row[1], "name": row[2], "kind": row[3]}

    def list_defs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """List defs (every version), newest version first within each key. §11.3.

        Read path; anchored on the `WorkflowDef.key` index.
        """
        res = self._reference().ro_query(
            "MATCH (d:WorkflowDef) WHERE d.key > '' "
            "RETURN d.key AS key, d.version AS version, d.name AS name, "
            "       d.kind AS kind "
            "ORDER BY d.key, d.version DESC LIMIT $limit",
            {"limit": limit},
        )
        return [
            {"key": row[0], "version": row[1], "name": row[2], "kind": row[3]}
            for row in res.result_set
        ]

    # ── §12 Workflow execution — runs, step-runs & traces (M3 executor) ──────────
    #
    # The executor walks a materialized snapshot (§11) as a WorkflowRun that
    # records each executed step as a StepRun. All live workspace-local in
    # `ws:{id}`. Each method maps 1:1 to a verified QUERIES.md §12 query; every
    # state-move is a single GRAPH.QUERY (atomicity, rule 4). Status-move contract:
    # zero rows = the anchor missed (run gone / wrong current state / CAS did not
    # apply) → `None`; a row = the move committed. `ctx`/`input`/`output`/`payload`
    # are opaque flat strings, stored and returned verbatim (rule 8).

    def start_run(
        self, ws: str, *, run_id: str, def_key: str, def_version: str,
        started_at: int, trigger_msg_id: str, ctx: str, trace: bool, max_steps: int,
    ) -> dict[str, Any] | None:
        """Begin a run at the snapshot's START step. QUERIES.md §12.1.

        Creates the `WorkflowRun` (`running`, `stepCount:0`, `waitingThreadId:''`)
        with `OF_DEF`/`AT_STEP`/`TRIGGERED_BY` edges. Returns
        `{runId, startKey, status, stepCount}`, or `None` when the snapshot has no
        START or the trigger message is missing (both anchors must match).
        """
        res = self._graph(ws).query(
            "MATCH (snap:WorkflowDefSnapshot {key: $defKey, version: $defVersion})"
            "-[:START]->(start:Step) "
            "MATCH (trigger:Message {msgId: $triggerMsgId}) "
            "CREATE (r:WorkflowRun {runId: $runId, defKey: $defKey, "
            "                       defVersion: $defVersion, status: 'running', "
            "                       startedAt: $startedAt, ctx: $ctx, trace: $trace, "
            "                       maxSteps: $maxSteps, stepCount: 0, "
            "                       waitingThreadId: ''}) "
            "CREATE (r)-[:OF_DEF]->(snap) "
            "CREATE (r)-[:AT_STEP]->(start) "
            "CREATE (r)-[:TRIGGERED_BY]->(trigger) "
            "RETURN r.runId AS runId, start.key AS startKey, r.status AS status, "
            "       r.stepCount AS stepCount",
            {
                "runId": run_id, "defKey": def_key, "defVersion": def_version,
                "startedAt": started_at, "triggerMsgId": trigger_msg_id,
                "ctx": ctx, "trace": trace, "maxSteps": max_steps,
            },
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {
            "runId": row[0], "startKey": row[1], "status": row[2],
            "stepCount": row[3],
        }

    def start_run_untriggered(
        self, ws: str, *, run_id: str, def_key: str, def_version: str,
        started_at: int, ctx: str, trace: bool, max_steps: int,
    ) -> dict[str, Any] | None:
        """Begin a run with **no** chat trigger message. QUERIES.md §12.12.

        Parent: §12.1 (`start_run`), minus the `MATCH (trigger:Message …)` anchor
        and the `TRIGGERED_BY` edge — a `kind:'process'` run (K-024) is started from
        REST/API, so there is no `Message`, no `Thread` and nothing to link.
        Deliberately a **second, self-contained write path** rather than an
        `OPTIONAL MATCH` + `FOREACH` conditional inside §12.1 (the §4
        first/subsequent doctrine; it sidesteps the empty-row-collapse bug class).

        Single anchor ⇒ `None` = the snapshot has no `START` (or `(key, version)`
        does not exist) and **nothing is written**. `waitingThreadId` starts `''`
        and stays `''` — a process run has no thread, which is why §12.9's lookup
        must never be called with an empty `threadId` (plan F-5/F-6).
        """
        res = self._graph(ws).query(
            "MATCH (snap:WorkflowDefSnapshot {key: $defKey, version: $defVersion})"
            "-[:START]->(start:Step) "
            "CREATE (r:WorkflowRun {runId: $runId, defKey: $defKey, "
            "                       defVersion: $defVersion, status: 'running', "
            "                       startedAt: $startedAt, ctx: $ctx, trace: $trace, "
            "                       maxSteps: $maxSteps, stepCount: 0, "
            "                       waitingThreadId: ''}) "
            "CREATE (r)-[:OF_DEF]->(snap) "
            "CREATE (r)-[:AT_STEP]->(start) "
            "RETURN r.runId AS runId, start.key AS startKey, r.status AS status, "
            "       r.stepCount AS stepCount",
            {
                "runId": run_id, "defKey": def_key, "defVersion": def_version,
                "startedAt": started_at, "ctx": ctx, "trace": trace,
                "maxSteps": max_steps,
            },
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {
            "runId": row[0], "startKey": row[1], "status": row[2],
            "stepCount": row[3],
        }

    def record_step_and_advance(
        self, ws: str, *, run_id: str, step_run_id: str, step_status: str,
        started_at: int, ended_at: int, input: str, output: str, to_step_uid: str,
        resolved_model: str | None = None, model_source: str | None = None,
        model_fallback: bool | None = None,
    ) -> dict[str, Any] | None:
        """The M4 tail-anchored atomic advance. QUERIES.md §12.2.

        One query: records the just-executed step (`cur`, at `AT_STEP`) as a
        `StepRun`, appends it to the `NEXT` trail via the `LAST_STEP_RUN` tail
        pointer, moves the tail, relinks `AT_STEP` to `$toStepUid`, and bumps
        `stepCount` — all atomic. `to_step_uid` may equal the current step's uid
        (advance-to-self: the executor's self-loop / suspend / terminal record).
        Returns `{stepCount, stepRunId, ranStepKey}`, or `None` when the run is
        missing, already terminal (no `AT_STEP`), or `$toStepUid` is not a step in
        this workspace.

        K-042 Landing 2 (FR-8, `-graph.md` §1.4): `resolved_model`/`model_source`/
        `model_fallback` ride the same `CREATE` as three additional, nullable
        `StepRun` properties. A `None` param **omits** the property entirely
        ([verified] `-graph.md` §1.4/§8) — the non-LLM step types (`decision`/
        `human`/`wait`, the offline `agent`-without-LLM stub) pass all three as
        `None` and cost zero extra bytes, no branching needed.
        """
        res = self._graph(ws).query(
            "MATCH (r:WorkflowRun {runId: $runId})-[atRel:AT_STEP]->(cur:Step) "
            "MATCH (to:Step {stepUid: $toStepUid}) "
            "OPTIONAL MATCH (r)-[lastRel:LAST_STEP_RUN]->(prevSR:StepRun) "
            "CREATE (sr:StepRun {stepRunId: $stepRunId, stepKey: cur.key, "
            "                    status: $stepStatus, startedAt: $startedAt, "
            "                    endedAt: $endedAt, input: $input, output: $output, "
            "                    resolvedModel: $resolvedModel, "
            "                    modelSource: $modelSource, "
            "                    modelFallback: $modelFallback}) "
            "CREATE (r)-[:HAS_STEP_RUN]->(sr) "
            "CREATE (sr)-[:RAN]->(cur) "
            "FOREACH (p  IN CASE WHEN prevSR  IS NULL THEN [] ELSE [prevSR]  END | "
            "  CREATE (p)-[:NEXT]->(sr)) "
            "FOREACH (lr IN CASE WHEN lastRel IS NULL THEN [] ELSE [lastRel] END | "
            "  DELETE lr) "
            "CREATE (r)-[:LAST_STEP_RUN]->(sr) "
            "DELETE atRel "
            "CREATE (r)-[:AT_STEP]->(to) "
            "SET r.stepCount = r.stepCount + 1 "
            "RETURN r.stepCount AS stepCount, sr.stepRunId AS stepRunId, "
            "       cur.key AS ranStepKey",
            {
                "runId": run_id, "stepRunId": step_run_id, "stepStatus": step_status,
                "startedAt": started_at, "endedAt": ended_at, "input": input,
                "output": output, "toStepUid": to_step_uid,
                "resolvedModel": resolved_model, "modelSource": model_source,
                "modelFallback": model_fallback,
            },
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {"stepCount": row[0], "stepRunId": row[1], "ranStepKey": row[2]}

    def suspend_run(
        self, ws: str, *, run_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        """Guarded CAS `running → waiting`. QUERIES.md §12.3.

        Denorms `waitingThreadId` (so §12.9 finds the parked run index-anchored).
        The flip commits only if the run is currently `running`; a suspend of a
        non-running run matches the node but fails the `WHERE` → `None`.
        """
        res = self._graph(ws).query(
            "MATCH (r:WorkflowRun {runId: $runId}) "
            "WHERE r.status = 'running' "
            "SET r.status = 'waiting', r.waitingThreadId = $threadId "
            "RETURN r.runId AS runId, r.status AS status",
            {"runId": run_id, "threadId": thread_id},
        )
        return self._status_row(res)

    def resume_run(self, ws: str, *, run_id: str) -> dict[str, Any] | None:
        """Guarded CAS `waiting → running` (single-flight). QUERIES.md §12.4.

        Two concurrent replies both try to resume; per-query atomicity means only
        the one that observes `status = 'waiting'` flips it — the loser sees
        `running`, the `WHERE` fails, and gets `None` (no re-entry). Clears
        `waitingThreadId` so the run is no longer discoverable as parked.
        """
        res = self._graph(ws).query(
            "MATCH (r:WorkflowRun {runId: $runId}) "
            "WHERE r.status = 'waiting' "
            "SET r.status = 'running', r.waitingThreadId = '' "
            "RETURN r.runId AS runId, r.status AS status",
            {"runId": run_id},
        )
        return self._status_row(res)

    def resume_run_with_ctx(
        self, ws: str, *, run_id: str, ctx: str
    ) -> dict[str, Any] | None:
        """Guarded CAS `waiting → running` **that also writes `ctx`**. §12.13.

        Parent: §12.4 (`resume_run`), which stays in use unchanged for the
        chat/trigger resume path (no ctx is submitted there). This variant is §12.4
        plus one `SET` term and is the human/signal-input path for a `process` run
        (K-024 **D-F**): the submitted input, already validated and merged into the
        run ctx service-side, rides **inside** the CAS so the write and the flip
        cannot be split.

        Zero-row contract (live-verified, QUERIES.md §12.13 — the whole D-F argument
        rests on it): a run that is not `waiting` matches the node but fails the
        `WHERE` ⇒ **`None` and NOTHING is written — neither the status flip nor the
        ctx**. Only the CAS *winner's* ctx is ever persisted, so "which input
        advanced the run" and "which input is in `ctx`" can never disagree; the
        loser's input is rejected (409), never silently lost. A missing `runId` is
        likewise zero rows — the service splits the two cases with a prior
        `get_run` (absent → 404, present-but-not-waiting → 409).
        """
        res = self._graph(ws).query(
            "MATCH (r:WorkflowRun {runId: $runId}) "
            "WHERE r.status = 'waiting' "
            "SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx "
            "RETURN r.runId AS runId, r.status AS status",
            {"runId": run_id, "ctx": ctx},
        )
        return self._status_row(res)

    def complete_run(
        self, ws: str, *, run_id: str, ended_at: int
    ) -> dict[str, Any] | None:
        """Terminal `done` — clears `AT_STEP`. QUERIES.md §12.5.

        The audit trail survives via `HAS_STEP_RUN` + `NEXT` + `LAST_STEP_RUN`.
        `DELETE` of a null `AT_STEP` is a verified no-op (re-completing an
        already-terminal run does not error). `None` = run not found.
        """
        res = self._graph(ws).query(
            "MATCH (r:WorkflowRun {runId: $runId}) "
            "OPTIONAL MATCH (r)-[atRel:AT_STEP]->() "
            "DELETE atRel "
            "SET r.status = 'done', r.endedAt = $endedAt "
            "RETURN r.runId AS runId, r.status AS status",
            {"runId": run_id, "endedAt": ended_at},
        )
        return self._status_row(res)

    def fail_run(
        self, ws: str, *, run_id: str, ended_at: int, ctx: str
    ) -> dict[str, Any] | None:
        """Terminal `failed` — clears `AT_STEP`, stamps a `ctx` note. QUERIES.md §12.5.

        The executor calls this on `stepCount > maxSteps` (step-budget exceeded,
        §7), stamping a diagnostic note into `ctx`. `None` = run not found.
        """
        res = self._graph(ws).query(
            "MATCH (r:WorkflowRun {runId: $runId}) "
            "OPTIONAL MATCH (r)-[atRel:AT_STEP]->() "
            "DELETE atRel "
            "SET r.status = 'failed', r.endedAt = $endedAt, r.ctx = $ctx "
            "RETURN r.runId AS runId, r.status AS status",
            {"runId": run_id, "endedAt": ended_at, "ctx": ctx},
        )
        return self._status_row(res)

    def link_step_emission(
        self, ws: str, *, step_run_id: str, msg_id: str
    ) -> dict[str, Any] | None:
        """`StepRun -[:PRODUCED]-> Message` (D2). QUERIES.md §12.6.

        The second query of the deliberately two-step emission (post the message
        via the guarded §4 write, then link). `MERGE` on the relationship makes the
        link idempotent (a retry re-links exactly once). Distinct from K-013's
        `EMITTED` (§10). `None` when an endpoint is missing.
        """
        res = self._graph(ws).query(
            "MATCH (sr:StepRun {stepRunId: $stepRunId}) "
            "MATCH (m:Message  {msgId: $msgId}) "
            "MERGE (sr)-[:PRODUCED]->(m) "
            "RETURN sr.stepRunId AS stepRunId, m.msgId AS msgId",
            {"stepRunId": step_run_id, "msgId": msg_id},
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {"stepRunId": row[0], "msgId": row[1]}

    def get_run(self, ws: str, *, run_id: str) -> dict[str, Any] | None:
        """Read a run's state. QUERIES.md §12.7 (RO).

        `atStepKey` is `None` for terminal runs (`AT_STEP` cleared). `None` when
        the run does not exist.
        """
        res = self._graph(ws).ro_query(
            "MATCH (r:WorkflowRun {runId: $runId}) "
            "OPTIONAL MATCH (r)-[:AT_STEP]->(cur:Step) "
            "OPTIONAL MATCH (r)-[:OF_DEF]->(snap:WorkflowDefSnapshot) "
            "RETURN r.runId AS runId, r.status AS status, r.stepCount AS stepCount, "
            "       r.maxSteps AS maxSteps, r.trace AS trace, r.ctx AS ctx, "
            "       r.startedAt AS startedAt, r.endedAt AS endedAt, "
            "       r.waitingThreadId AS waitingThreadId, cur.key AS atStepKey, "
            "       snap.key AS defKey, snap.version AS defVersion",
            {"runId": run_id},
        )
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {
            "runId": row[0], "status": row[1], "stepCount": row[2],
            "maxSteps": row[3], "trace": row[4], "ctx": row[5],
            "startedAt": row[6], "endedAt": row[7], "waitingThreadId": row[8],
            "atStepKey": row[9], "defKey": row[10], "defVersion": row[11],
        }

    def read_step_runs(self, ws: str, *, run_id: str) -> list[dict[str, Any]]:
        """The NEXT-ordered audit trail. QUERIES.md §12.8 (RO).

        Finds the chain head via `OPTIONAL MATCH` + `IS NULL` (never the broken
        `exists()`-in-pattern check), then walks `NEXT*0..`, ordered by the
        executor's monotonic `startedAt` (coincides with NEXT order).

        K-042 Landing 2 (FR-8, `-graph.md` §1.7): projects `resolvedModel`/
        `modelSource`/`modelFallback` alongside the existing columns — `None` for a
        `StepRun` that never had them written (a non-LLM step, or a pre-Landing-2
        row; both read back identically, no backfill).
        """
        res = self._graph(ws).ro_query(
            "MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun) "
            "OPTIONAL MATCH (pv:StepRun)-[:NEXT]->(sr) "
            "WITH sr, pv WHERE pv IS NULL "
            "MATCH (sr)-[:NEXT*0..]->(x:StepRun) "
            "RETURN x.stepRunId AS stepRunId, x.stepKey AS stepKey, "
            "       x.status AS status, x.startedAt AS startedAt, "
            "       x.endedAt AS endedAt, x.input AS input, x.output AS output, "
            "       x.resolvedModel AS resolvedModel, "
            "       x.modelSource AS modelSource, "
            "       x.modelFallback AS modelFallback "
            "ORDER BY x.startedAt",
            {"runId": run_id},
        )
        return [
            {
                "stepRunId": row[0], "stepKey": row[1], "status": row[2],
                "startedAt": row[3], "endedAt": row[4], "input": row[5],
                "output": row[6], "resolvedModel": row[7], "modelSource": row[8],
                "modelFallback": row[9],
            }
            for row in res.result_set
        ]

    def find_waiting_run_for_thread(
        self, ws: str, *, thread_id: str
    ) -> dict[str, Any] | None:
        """The resume lookup, index-anchored on `WorkflowRun.status`. QUERIES.md §12.9 (RO).

        Anchors on the existing `status` index (point lookup on `'waiting'`) plus a
        residual filter on the denormed `waitingThreadId` — no new index. A thread
        holds at most one `waiting` run at a time. `None` when nothing is parked.
        """
        res = self._graph(ws).ro_query(
            "MATCH (r:WorkflowRun {status: 'waiting'}) "
            "WHERE r.waitingThreadId = $threadId "
            "RETURN r.runId AS runId, r.status AS status "
            "LIMIT 1",
            {"threadId": thread_id},
        )
        return self._status_row(res)

    def find_runs_for_thread(
        self, ws: str, *, thread_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Every run this thread has ever had, newest-first (RO). QUERIES.md
        §12.14 `find_runs_for_thread` (K-036 — web-api-coverage FR-2).

        The `r.startedAt >= 0` conjunct is functionally a no-op (`startedAt` is
        always non-negative) but is **load-bearing for the query plan** — see
        QUERIES.md §12.14's PROFILE finding: without it the anchor lands on
        `Node By Label Scan (m:Message)` (a workspace-wide scan) instead of the
        small `WorkflowRun` scan, because a `WHERE` predicate on one pattern
        variable pulls the label-scan anchor onto that variable regardless of
        relative cardinality. Paired with the `WorkflowRun.startedAt` range
        index (`bootstrap_schema.sh`), this anchors on
        `Node By Index Scan (r:WorkflowRun)`. Empty list for a thread with no
        runs (not an error).
        """
        res = self._graph(ws).ro_query(
            "MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(m:Message) "
            "WHERE r.startedAt >= 0 AND m.threadId = $threadId "
            "RETURN r.runId AS runId, r.status AS status, r.defKey AS defKey, "
            "       r.defVersion AS defVersion, r.startedAt AS startedAt, "
            "       r.endedAt AS endedAt "
            "ORDER BY r.startedAt DESC "
            "LIMIT $limit",
            {"threadId": thread_id, "limit": limit},
        )
        return [
            {
                "runId": row[0], "status": row[1], "defKey": row[2],
                "defVersion": row[3], "startedAt": row[4], "endedAt": row[5],
            }
            for row in res.result_set
        ]

    def find_due_wait_candidates(
        self, ws: str, *, limit: int
    ) -> list[dict[str, Any]]:
        """The K-028 sweep's read half — every parked `wait`/`human` candidate,
        due-agnostic. QUERIES.md §12.16 (RO).

        Anchors on the **existing** `WorkflowRun.status` value index (the same
        anchor §12.9 already uses, on the same "waiting set is tiny" cardinality
        argument) — no new index, no new `WorkflowRun` property. Dueness itself
        is derived app-side (`services._wait_due_at`) from data the existing
        suspend/record path already writes unconditionally on every park
        (`docs/plans/workflow-timers.md` §3.2); this query only reads. Returns
        every `waiting` run regardless of whether it is actually due, or even
        whether its parked step declares a timer at all — `sweep_due_workflow_
        runs` decides that app-side, never in Cypher (rule 8: never filter
        inside `config`).

        v3 (`docs/plans/workflow-timers.md` §3.4): `RETURN` gains `s.key AS
        stepKey` — an additive projection off the already-bound `s`, no new
        traversal — so the sweep can write the step-scoped `ctx.timerFired`
        marker (§3.3/§3.5) without a second read, and can also detect (§3.5 step
        5.1) whether a candidate has since moved to a *different* waiting step
        between this scan and the sweep's per-candidate act.
        """
        res = self._graph(ws).ro_query(
            "MATCH (r:WorkflowRun {status: 'waiting'})-[:AT_STEP]->(s:Step) "
            "OPTIONAL MATCH (r)-[:LAST_STEP_RUN]->(sr:StepRun) "
            "RETURN r.runId AS runId, s.key AS stepKey, s.type AS stepType, "
            "       s.config AS stepConfig, sr.startedAt AS parkedAt "
            "LIMIT $limit",
            {"limit": limit},
        )
        return [
            {
                "runId": row[0], "stepKey": row[1], "stepType": row[2],
                "stepConfig": row[3], "parkedAt": row[4],
            }
            for row in res.result_set
        ]

    def read_recent_post_success(
        self, ws: str, *, def_key: str, def_version: str, limit: int
    ) -> dict[str, int]:
        """Last-`limit` terminal-run post-success sample (RO). QUERIES.md §12.15
        (K-039 item 3).

        Feeds `Services.check_demo_readiness`'s `postSuccess` field: of the last
        `limit` **terminal** (`status IN ['done', 'failed']`) runs of
        `def_key`@`def_version`, ordered `startedAt DESC`, how many produced at
        least one reply (`StepRun -[:PRODUCED]-> Message`, D2, §12.6).
        `waiting`/`running` runs are excluded from the sample — they haven't
        reached a verdict yet. `sampleSize`/`postedCount` are both `0` for a
        fresh workspace or an unknown `def_key`/`def_version` — never an
        exception. **`postedCount` comes back from FalkorDB as a Python
        `float`**, never `NULL`/`None`, in both the empty and non-empty case
        (live-verified, QUERIES.md §12.15) — cast to `int` here so callers get
        two clean `int`s, not `"postedCount": 1.0` leaking into the JSON
        response.
        """
        res = self._graph(ws).ro_query(
            "MATCH (r:WorkflowRun) "
            "WHERE r.startedAt >= 0 "
            "  AND r.defKey = $defKey AND r.defVersion = $defVersion "
            "  AND r.status IN ['done', 'failed'] "
            "WITH r ORDER BY r.startedAt DESC LIMIT $limit "
            "OPTIONAL MATCH (r)-[:HAS_STEP_RUN]->(:StepRun)-[:PRODUCED]->(m:Message) "
            "WITH r, count(m) AS producedCount "
            "RETURN count(r) AS sampleSize, "
            "       sum(CASE WHEN producedCount > 0 THEN 1 ELSE 0 END) AS postedCount",
            {"defKey": def_key, "defVersion": def_version, "limit": limit},
        )
        row = res.result_set[0]
        return {"sampleSize": row[0], "postedCount": int(row[1])}

    def append_trace_event(
        self, ws: str, *, step_run_id: str, trace_id: str, seq: int, kind: str,
        at: int, payload: str,
    ) -> dict[str, Any] | None:
        """Write one debug trace record (FR-4). QUERIES.md §12.10.

        Called ONLY for a debug instance (`WorkflowRun.trace = true`) — the
        `GraphTracer`; the `NullTracer` issues no query so a lean run writes zero
        `TraceEvent` nodes. `payload` is a flat serialized string, length-capped by
        the caller (rule 6). `None` when the step-run anchor is missing.
        """
        res = self._graph(ws).query(
            "MATCH (sr:StepRun {stepRunId: $stepRunId}) "
            "CREATE (te:TraceEvent {traceId: $traceId, seq: $seq, kind: $kind, "
            "                       at: $at, payload: $payload}) "
            "CREATE (sr)-[:TRACED]->(te) "
            "RETURN te.traceId AS traceId",
            {
                "stepRunId": step_run_id, "traceId": trace_id, "seq": seq,
                "kind": kind, "at": at, "payload": payload,
            },
        )
        if not res.result_set:
            return None
        return {"traceId": res.result_set[0][0]}

    def read_trace(self, ws: str, *, run_id: str) -> list[dict[str, Any]]:
        """Reconstruct a run's execution (debug). QUERIES.md §12.11 (RO).

        Traverses run → step-runs → trace events, ordered by
        `(StepRun.startedAt, TraceEvent.seq)` — the full cross-step reconstruction.
        A non-debug run has zero `TRACED` edges → `[]` (AC-5's negative half).
        """
        res = self._graph(ws).ro_query(
            "MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun)"
            "-[:TRACED]->(te:TraceEvent) "
            "RETURN sr.stepRunId AS stepRunId, sr.stepKey AS stepKey, "
            "       te.traceId AS traceId, te.seq AS seq, te.kind AS kind, "
            "       te.at AS at, te.payload AS payload "
            "ORDER BY sr.startedAt, te.seq",
            {"runId": run_id},
        )
        return [
            {
                "stepRunId": row[0], "stepKey": row[1], "traceId": row[2],
                "seq": row[3], "kind": row[4], "at": row[5], "payload": row[6],
            }
            for row in res.result_set
        ]

    # ── K-042 Landing 2 U9 (FR-16/FR-17): workspace model overrides ─────────────
    #
    # A distinct topic from workflow execution above — workspace-level configuration,
    # not a run/step-run/trace. `docs/plans/llm-provider-config-graph.md` §2.2/§2.4/
    # §2.5/§6.1 owns the design; QUERIES.md §13 documents the same Cypher.

    def write_model_overrides(
        self, ws: str, *, agent: str | None = None, guard: str | None = None,
        embedding: str | None = None, responder: str | None = None,
        at: int, by: str,
    ) -> dict[str, Any]:
        """Write the workspace's per-kind model-override singleton (FR-16/FR-17,
        `-graph.md` §2.2/§2.4). `docs/QUERIES.md` §13.1.

        Constraint-backed `MERGE` on `WorkspaceConfig.workspaceConfigId` (the
        `bootstrap_schema.sh` `UNIQUE` constraint — the standing "every `MERGE` must
        be backed by a uniqueness constraint" rule) — safe under concurrent writers.
        Each of `agent`/`guard`/`embedding`/`responder` is a `"<provider>/<model-id>"`
        ref, a role name, or `None`. A `None` in `SET` **clears** that kind's
        override — unlike `CREATE`, where a `None` param merely omits the key — so
        "never set" and "explicitly cleared" land on the identical representation
        (the property absent), which is also what §2.5's read treats as "no
        override at that kind". **Callers must never pass `''`** — an empty string
        is a value ("a model literally named empty"), never "no override"; that
        representation is reserved for `None` alone (`-graph.md` §2.4's explicit
        warning).

        Property-name crosswalk against this repo's own `kind` strings is NOT this
        method's concern — it only knows the graph's four property names
        (`agentModelOverride`/`guardModelOverride`/`embeddingModelOverride`/
        `responderModelOverride`). The kind↔property mapping (which is **not** a
        1:1 name match — `-graph.md` §8.4) lives in `modelconfig._KIND_TO_OVERRIDE_KEY`,
        the resolver's own concern.

        Returns the four values as written (post-`SET`, so a caller can confirm a
        clear actually landed).
        """
        res = self._graph(ws).query(
            "MERGE (c:WorkspaceConfig {workspaceConfigId: 'default'}) "
            "SET c.agentModelOverride     = $agent, "
            "    c.guardModelOverride     = $guard, "
            "    c.embeddingModelOverride = $embedding, "
            "    c.responderModelOverride = $responder, "
            "    c.modelOverrideUpdatedAt = $at, "
            "    c.modelOverrideUpdatedBy = $by "
            "RETURN c.agentModelOverride AS agent, c.guardModelOverride AS guard, "
            "       c.embeddingModelOverride AS embedding, "
            "       c.responderModelOverride AS responder",
            {
                "agent": agent, "guard": guard, "embedding": embedding,
                "responder": responder, "at": at, "by": by,
            },
        )
        row = res.result_set[0]
        return {"agent": row[0], "guard": row[1], "embedding": row[2], "responder": row[3]}

    def read_model_overrides(self, ws: str) -> dict[str, str | None]:
        """Read the workspace's per-kind model overrides (FR-16/FR-17, `-graph.md`
        §2.5/§6.1). `docs/QUERIES.md` §13.2 (RO).

        Returns `{agentModel, guardModel, embeddingModel, responderModel}` — the
        exact shape `-graph.md` §6.1 names. **Zero rows (the `WorkspaceConfig` node
        was never written) and a one-row all-`NULL` result (written, but this
        particular kind was never set) are the SAME answer — "no override at that
        kind" — and both are returned as the identical all-`None` dict here, one
        code path, per §2.5's own live-verified note.** Every existing workspace is
        the zero-row case, so this must stay the cheap, silent, non-exceptional
        default, never an error.
        """
        res = self._graph(ws).ro_query(
            "MATCH (c:WorkspaceConfig {workspaceConfigId: 'default'}) "
            "RETURN c.agentModelOverride     AS agentModel, "
            "       c.guardModelOverride     AS guardModel, "
            "       c.embeddingModelOverride AS embeddingModel, "
            "       c.responderModelOverride AS responderModel"
        )
        if not res.result_set:
            return {
                "agentModel": None, "guardModel": None,
                "embeddingModel": None, "responderModel": None,
            }
        row = res.result_set[0]
        return {
            "agentModel": row[0], "guardModel": row[1],
            "embeddingModel": row[2], "responderModel": row[3],
        }

    @staticmethod
    def _status_row(res) -> dict[str, Any] | None:
        """`{runId, status}` for the §12 state-move CAS queries, or `None` (zero rows)."""
        if not res.result_set:
            return None
        row = res.result_set[0]
        return {"runId": row[0], "status": row[1]}

    def materialize_snapshot(
        self, ws: str, *, key: str, version: str, name: str, kind: str,
        start_key: str, steps: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Materialize a def@version into `ws:{id}` as a local snapshot. §11.4.

        Write path — the workspace half of the two-phase materialize (the
        `name/kind/start_key/steps/transitions` come from `read_def_subgraph`).
        Same MERGE shape as `publish_def` with the `WorkflowDefSnapshot` root
        label, scoped to the workspace graph; produces a subgraph structurally
        identical to the reference def. **This method itself does not enforce
        structural immutability** — a differing payload for an existing
        `(key, version)` mints parallel `TRANSITION`/`START` structure in the
        snapshot (K-034), the same hazard `publish_def` carries. The guarantee is
        enforced by the caller: `services.materialize_def` rejects a topology-
        differing re-materialize (`WorkflowDefConflictError`, 409) *before*
        calling this method. Returns `{key, version, stepCount, transitionCount}`.
        """
        res = self._graph(ws).query(
            self._PUBLISH_CYPHER.format(label="WorkflowDefSnapshot"),
            {
                "key": key, "version": version, "name": name, "kind": kind,
                "startKey": start_key, "steps": list(steps),
                "transitions": list(transitions),
            },
        )
        row = res.result_set[0]
        return {
            "key": row[0], "version": row[1],
            "stepCount": row[2], "transitionCount": row[3],
        }

    def get_snapshot(self, ws: str, *, key: str, version: str) -> dict[str, Any] | None:
        """Read a snapshot's full subgraph from `ws:{id}`. §11.5.

        Mirror of `read_def_subgraph` with the `WorkflowDefSnapshot` root; `None`
        when absent. `{name, kind, start_key, steps, transitions}`. Read path.
        """
        return self._read_subgraph(
            self._graph(ws), label="WorkflowDefSnapshot", key=key, version=version
        )

    def read_snapshot_structure(
        self, ws: str, *, key: str, version: str
    ) -> dict[str, Any] | None:
        """Read a snapshot's full structure from `ws:{id}` for observability. §11.5.

        Mirror of `read_def_structure` with the `WorkflowDefSnapshot` root; same
        multi-`START`-aware shape (`start_keys: list[str]`) — see
        `_read_structure`. `None` when absent. Read path.
        """
        return self._read_structure(
            self._graph(ws), label="WorkflowDefSnapshot", key=key, version=version
        )

    def list_snapshots(self, ws: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """List a workspace's snapshots, newest version first per key. §11.6.

        Read path; anchored on the `WorkflowDefSnapshot.key` index.
        """
        res = self._graph(ws).ro_query(
            "MATCH (snap:WorkflowDefSnapshot) WHERE snap.key > '' "
            "RETURN snap.key AS key, snap.version AS version, snap.name AS name, "
            "       snap.kind AS kind "
            "ORDER BY snap.key, snap.version DESC LIMIT $limit",
            {"limit": limit},
        )
        return [
            {"key": row[0], "version": row[1], "name": row[2], "kind": row[3]}
            for row in res.result_set
        ]
