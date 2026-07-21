"""Domain logic — the invariants live here (DESIGN §14.2).

`services.py` is the only layer that:
  * generates ids and timestamps (server clock — never client-supplied),
  * picks the first-vs-subsequent §4 message write variant,
  * validates that mentions resolve to known members before writing,
  * constructs `cursorId` and decides read-only vs read-write for `read_messages`.

Both front doors (`api.py` REST, `mcp.py` MCP tools) are thin adapters over
these methods; they carry no business logic.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from redis.exceptions import ResponseError

from .config import CallContext
from .guards import CMP_KINDS, validate_cmp

# `MemberIdCollisionError`/`EmbeddingDimensionError`/`WorkflowDef*Error` are
# re-exported (redundant-alias idiom) as part of the service error surface: the
# repository owns them (the §2/§7 status-row contract, the §6 embedding-write
# validation, the §11 workflow error types); they live there only to avoid an
# import cycle.
from .repository import EmbeddingDimensionError as EmbeddingDimensionError
from .repository import MemberIdCollisionError as MemberIdCollisionError
from .repository import Repository
from .repository import StepBudgetExceededError as StepBudgetExceededError
from .repository import WorkflowDefNotFoundError as WorkflowDefNotFoundError
from .repository import WorkflowDefSpecError as WorkflowDefSpecError
from .repository import WorkflowRunNotFoundError as WorkflowRunNotFoundError

# ── §11 workflow spec whitelists (plan §B5 / DESIGN §6.1) ───────────────────────
WORKFLOW_KINDS: frozenset[str] = frozenset({"conversation", "process"})
# `agent` is the M3 LLM-native node kind (§3) — a plain-language system prompt the
# model runs as a bounded, tool-scoped agent. `type` stays opaque in-graph (rule 8);
# this whitelist only gates what a def may declare at publish time.
STEP_TYPES: frozenset[str] = frozenset(
    {"prompt", "tool", "decision", "human", "message", "wait", "agent"}
)
# Step types that park the run pending an outside actor (a person for `human`, a
# signalling system for `wait` — mechanically identical to the engine, K-024 D-C). Both
# MUST declare `config.waitsForHuman: true`: the executor's OUTCOME B keys on exactly
# that flag, so a parking step without it self-loops (OUTCOME C) until the step budget
# kills the run — a silent, expensive footgun best caught at authoring time.
WAITING_STEP_TYPES: frozenset[str] = frozenset({"human", "wait"})

# ── GraphRAG read posture (K-007 TIMEOUT / DESIGN §10) ──────────────────────────
# The FalkorDB global TIMEOUT default is 1000 ms and writes ignore it; GraphRAG
# reads (ANN seed + traversal) can legitimately run longer, so they pass a single
# per-query client `timeout=` override here rather than ad-hoc per call. Uncapped
# while the deployment keeps `TIMEOUT_MAX=0`.
RAG_QUERY_TIMEOUT_MS = 5000

# ── errors ─────────────────────────────────────────────────────────────────────


class ServiceError(Exception):
    """Base class for service-layer validation errors."""


class ChannelNotFoundError(ServiceError):
    pass


class ThreadNotFoundError(ServiceError):
    pass


class UnknownMemberError(ServiceError):
    """Raised when a mention does not resolve to a known member."""


class UnknownActorError(ServiceError):
    """Raised when the context actor does not resolve to a known member.

    Guards the silent-no-op failure mode: the §4 write queries anchor on
    `MATCH (author …)`, and a missing author makes the whole write a no-op
    while the transport would still report success.
    """


class InvalidSearchQueryError(ServiceError):
    """Raised when the full-text query is rejected by RediSearch syntax."""


def _default_id() -> str:
    return uuid.uuid4().hex


def _default_clock() -> int:
    """Server clock in milliseconds since the epoch."""
    return int(time.time() * 1000)


def _dedup(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    return list(dict.fromkeys(items))


def _serialize_opaque(value: Any) -> str:
    """Serialize a `config`/`guard` value to the opaque string stored in-graph.

    §11 rule 8: `Step.config` and `TRANSITION.guard` are flat serialized strings
    stored verbatim and never queried inside. `None`/missing → `""`; an existing
    string passes through unchanged (already-serialized); anything else is
    compact JSON (stable key order) so re-publishing the same spec is a no-op.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _normalize_opaque(value: Any) -> Any:
    """The inverse of `_serialize_opaque`, for validating a spec before it is written.

    `_validate_def_spec` sees `config`/`guard` **heterogeneously typed** (M-7), because
    serialization to opaque strings happens *after* it, in `publish_workflow_def`: the
    REST front door types both as `str` (`schemas.py`), while service-layer and MCP
    callers hand over dicts. Validating without normalizing would either blow up on a
    string (`AttributeError` → a 500 on `POST /workflow-defs`) or — worse — skip every
    string-shaped value, letting **every REST-published def escape the invariants**
    silently. So: a `str` is parsed as JSON when it can be, and returned unchanged when
    it cannot (an opaque `"raw-string"` config stays exactly that); anything else is
    already parsed and passes through.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


class Services:
    def __init__(
        self,
        repo: Repository,
        *,
        clock: Callable[[], int] = _default_clock,
        id_gen: Callable[[], str] = _default_id,
        executor: Any = None,
    ) -> None:
        self._repo = repo
        self._clock = clock
        self._id = id_gen
        self._ts_lock = threading.Lock()
        self._last_ts = 0
        # The workflow-run engine (M3 §12). Injected — and, because the executor
        # holds a back-reference to these services (post/retrieve seams), it can be
        # bound late via `set_executor` when the app wires both (avoids a
        # construction cycle). Off by default so the M1/M2 surface is untouched.
        self._executor = executor

    def set_executor(self, executor: Any) -> None:
        """Late-bind the workflow executor (Phase-4 app wiring — see `__init__`)."""
        self._executor = executor

    def _next_ts(self) -> int:
        """Monotonic per-process ms clock — makes same-ms message ties impossible
        (K-007 item 4a). Used only for message `createdAt`; channel/thread stamps
        keep the plain clock (ties there are harmless). Lock-guarded because
        FastAPI runs sync endpoints on a threadpool."""
        with self._ts_lock:
            ts = max(self._clock(), self._last_ts + 1)
            self._last_ts = ts
            return ts

    # ── health ──────────────────────────────────────────────────────────────────

    def ping(self, ctx: CallContext) -> bool:
        """True when the workspace graph answers a trivial read."""
        return self._repo.ping(ctx.ws)

    # ── members ─────────────────────────────────────────────────────────────────

    def ensure_actor(self, ctx: CallContext) -> None:
        """Project the context actor into the workspace as a `User` (idempotent).

        Called at app startup so the configured actor exists before the first
        write — the §4 write paths refuse an unknown author. The configured
        actor is projected as a `User`; Agent actors (seeded via
        `repo.ensure_agent`) post with role `assistant` — real per-client agent
        auth is still to come.
        """
        self._repo.ensure_user(ctx.ws, user_id=ctx.actor)

    # ── channels ────────────────────────────────────────────────────────────────

    def create_channel(self, ctx: CallContext, *, name: str) -> dict[str, Any]:
        channel_id = self._id()
        now = self._clock()
        self._repo.create_channel(
            ctx.ws, channel_id=channel_id, name=name, created_at=now
        )
        return {"channelId": channel_id, "name": name, "createdAt": now}

    def list_channels(self, ctx: CallContext, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._repo.list_channels(ctx.ws, limit=limit)

    # ── threads ─────────────────────────────────────────────────────────────────

    def create_thread(
        self, ctx: CallContext, *, channel_id: str, title: str
    ) -> dict[str, Any]:
        if not self._repo.channel_exists(ctx.ws, channel_id=channel_id):
            raise ChannelNotFoundError(channel_id)
        thread_id = self._id()
        now = self._clock()
        self._repo.create_thread(
            ctx.ws, channel_id=channel_id, thread_id=thread_id,
            title=title, created_at=now,
        )
        return {
            "threadId": thread_id, "channelId": channel_id,
            "title": title, "createdAt": now,
        }

    def list_threads(
        self, ctx: CallContext, *, channel_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._repo.list_threads(ctx.ws, channel_id=channel_id, limit=limit)

    # ── messages ────────────────────────────────────────────────────────────────

    def _validate_and_derive_role(
        self, ctx: CallContext, *, thread_id: str, mentions: list[str] | None,
    ) -> tuple[list[str], str]:
        """Shared §4 pre-write validation: thread exists, actor + mentions known.

        Returns `(wanted_mentions, role)` — `role` derived from the actor's node
        label (`User → user`, `Agent → assistant`; agents author first-class).
        Raises the same errors as the write paths would silently no-op on.
        """
        if not self._repo.thread_exists(ctx.ws, thread_id=thread_id):
            raise ThreadNotFoundError(thread_id)

        wanted = _dedup(list(mentions or []))
        # One member-kind lookup covers the author and every mention. The author
        # check is load-bearing: an unknown author makes the v2 write refuse
        # (authorFound=false) — validate before writing.
        kinds = self._repo.resolve_member_kinds(ctx.ws, ids=[ctx.actor, *wanted])
        actor_kind = kinds.get(ctx.actor)
        if actor_kind is None:
            raise UnknownActorError(ctx.actor)
        role = "user" if actor_kind == "User" else "assistant"
        unknown = [m for m in wanted if kinds.get(m) is None]
        if unknown:
            raise UnknownMemberError(unknown)
        return wanted, role

    def _dispatch_write(
        self, ctx: CallContext, *, thread_id: str, msg_id: str,
        first_write: Callable[..., Any], subsequent_write: Callable[..., Any],
        write_kwargs: dict[str, Any],
    ) -> None:
        """Run the §4 v2 first/subsequent dispatch loop (QUERIES.md §4 contract).

        Shared by `post_message` and `post_agent_answer` — the only difference is
        which write-path pair is passed in (plain §4 vs the §10 EMITTED-carrying
        variants) and the extra `write_kwargs` (e.g. `seeds`). Dispatch:
        `dupMsg` = idempotent retry success; `hadHead` = lost the first-post race
        → re-dispatch as subsequent; subsequent with no TAIL → `None` → re-dispatch
        as first. The loop bound is a tripwire — ping-pong is impossible by
        contract (a headed thread always has a TAIL).
        """
        use_first = not self._repo.thread_has_head(ctx.ws, thread_id=thread_id)
        for _attempt in range(4):
            write = first_write if use_first else subsequent_write
            st = write(ctx.ws, thread_id=thread_id, msg_id=msg_id, **write_kwargs)
            if st is None:
                if use_first:                    # thread anchor vanished (TOCTOU)
                    raise ThreadNotFoundError(thread_id)
                use_first = True                 # no TAIL yet — retry as first-post
                continue
            if st.written or st.dup_msg:         # dup_msg = idempotent success (OQ2)
                return
            if not st.author_found:              # belt-and-suspenders vs the pre-check
                raise UnknownActorError(ctx.actor)
            if st.had_head:                      # lost the first-post race
                use_first = False
                continue
            raise RuntimeError(f"unexpected write status {st!r} (thread={thread_id!r})")
        raise RuntimeError(
            "message write dispatch did not converge "
            f"(thread={thread_id!r}, msg={msg_id!r})"
        )

    def post_message(
        self, ctx: CallContext, *, thread_id: str, text: str,
        mentions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Post a message into an existing thread.

        Validates the actor and mentions, derives `role` from the actor's node
        label (`User → user`, `Agent → assistant` — agents can author), then
        dispatches on the §4 v2 status row via `_dispatch_write`.
        """
        wanted, role = self._validate_and_derive_role(
            ctx, thread_id=thread_id, mentions=mentions
        )
        msg_id, now = self._id(), self._next_ts()
        self._dispatch_write(
            ctx, thread_id=thread_id, msg_id=msg_id,
            first_write=self._repo.post_first_message,
            subsequent_write=self._repo.post_subsequent_message,
            write_kwargs={
                "author_id": ctx.actor, "text": text, "role": role,
                "created_at": now, "mentions": wanted,
            },
        )
        return {
            "msgId": msg_id, "threadId": thread_id, "authorId": ctx.actor,
            "text": text, "role": role, "createdAt": now, "mentions": wanted,
        }

    def post_agent_answer(
        self, ctx: CallContext, *, thread_id: str, text: str,
        mentions: list[str] | None = None,
        seeds: list[tuple[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Post an agent-authored answer with §10 `EMITTED` provenance (K-013).

        `ctx.actor` is the answering Agent — the responder swaps the actor to the
        agent id so `role` derives to `assistant` here exactly like `post_message`
        (never trusted from the caller). Same §4 dispatch (`_dispatch_write`) over
        the §10.1 EMITTED-carrying write paths; `seeds` (`[(msgId, score)]` in rank
        order) ride inside the single GRAPH.QUERY (atomicity). `seeds=[]` is a
        verified no-op — the message still commits.
        """
        wanted, role = self._validate_and_derive_role(
            ctx, thread_id=thread_id, mentions=mentions
        )
        ordered_seeds = list(seeds or [])
        msg_id, now = self._id(), self._next_ts()
        self._dispatch_write(
            ctx, thread_id=thread_id, msg_id=msg_id,
            first_write=self._repo.post_agent_answer_first,
            subsequent_write=self._repo.post_agent_answer,
            write_kwargs={
                "author_id": ctx.actor, "text": text, "role": role,
                "created_at": now, "mentions": wanted, "seeds": ordered_seeds,
            },
        )
        return {
            "msgId": msg_id, "threadId": thread_id, "authorId": ctx.actor,
            "text": text, "role": role, "createdAt": now, "mentions": wanted,
            "seeds": ordered_seeds,
        }

    def read_messages(
        self, ctx: CallContext, *, thread_id: str | None = None,
        since: int | None = None, limit: int = 50, advance: bool = True,
    ) -> list[dict[str, Any]]:
        """Read messages since a cursor/timestamp.

        Modes:
          * explicit ``since`` → pure read with plain ``>`` timestamp semantics;
            the cursor is never touched. May re-deliver or skip messages within
            that exact millisecond (OQ3 contract) — agents that need lossless
            catch-up use cursor mode.
          * no ``since`` + ``thread_id`` → read from the member's per-thread
            composite cursor ``(lastReadAt, lastReadMsgId)`` (or the epoch base
            ``(0, '')``), then, when ``advance`` is set, move the cursor forward
            to the newest ``(createdAt, msgId)`` pair actually delivered (a
            write). Never the server clock — that would permanently skip rows a
            ``limit`` truncated. An empty page advances nothing. Cursor-driven
            reads never skip or re-deliver, even across millisecond ties.
          * no ``since`` + no ``thread_id`` → room-wide read from epoch 0. There
            is no room-wide cursor in M1, so nothing is advanced.
        """
        explicit_since = since is not None

        if thread_id is not None:
            cursor_id = f"{ctx.actor}:{thread_id}"
            if explicit_since:
                return self._repo.read_thread_since(
                    ctx.ws, thread_id=thread_id, me_id=ctx.actor,
                    since=since, since_msg_id=None, limit=limit,  # plain `>`
                )
            pair = self._repo.get_cursor(ctx.ws, cursor_id=cursor_id)
            eff_since, eff_msg = pair if pair is not None else (0, None)
            rows = self._repo.read_thread_since(
                ctx.ws, thread_id=thread_id, me_id=ctx.actor,
                since=eff_since or 0, since_msg_id=eff_msg or "", limit=limit,
            )
            if advance and rows:
                last = rows[-1]  # rows are ORDER BY (createdAt, msgId) — the max pair
                self._repo.advance_cursor(
                    ctx.ws, me_id=ctx.actor, thread_id=thread_id,
                    cursor_id=cursor_id,
                    now=last["createdAt"], now_msg_id=last["msgId"],
                )
            return rows

        # room-wide: no cursor, defaults to epoch 0, never advances (plain `>`)
        eff_since = since if explicit_since else 0
        return self._repo.read_ws_since(
            ctx.ws, me_id=ctx.actor, since=eff_since, since_msg_id=None, limit=limit
        )

    def search_messages(
        self, ctx: CallContext, *, query: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Workspace-wide full-text keyword search. QUERIES.md §5.

        RediSearch parses the query string; its syntax errors (unbalanced
        quotes, stray operators) are a caller problem, not a server fault.
        """
        try:
            return self._repo.search_messages(ctx.ws, query=query, limit=limit)
        except ResponseError as exc:
            raise InvalidSearchQueryError(str(exc)) from exc

    def hybrid_search(
        self, ctx: CallContext, *, q_vec: list[float], k: int = 10,
        limit: int = 10, channel_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """GraphRAG hybrid retrieval (QUERIES.md §6): vector ANN + scope traversal.

        Passes the single service-layer `RAG_QUERY_TIMEOUT_MS` override on the RO
        query (DESIGN §10). Rows come back already ordered by cosine distance ASC
        (most similar first) — not re-sorted here. `score` is a distance, not a
        similarity; a caller that wants similarity derives `1 - score` client-side.
        `relatedContext` is `[]` in M2 (Entity layer dormant) and passed through.
        """
        return self._repo.hybrid_search(
            ctx.ws, q_vec=q_vec, k=k, limit=limit, channel_id=channel_id,
            timeout=RAG_QUERY_TIMEOUT_MS,
        )

    # ── reads (thin passthroughs) ───────────────────────────────────────────────

    def read_thread(self, ctx: CallContext, *, thread_id: str) -> list[dict[str, Any]]:
        return self._repo.read_thread(ctx.ws, thread_id=thread_id)

    def get_message(self, ctx: CallContext, *, msg_id: str) -> dict[str, Any] | None:
        return self._repo.get_message(ctx.ws, msg_id=msg_id)

    # ── §11 Workflow definitions & snapshots (M3 Slice 1) ────────────────────────
    #
    # Def authoring/reading is GLOBAL (the `reference` graph; repo methods omit
    # `ws`, plan F3); only materialization + snapshot reads consume `ctx.ws`.
    # `CallContext`/`config.get_context` are unchanged.

    @staticmethod
    def _validate_def_spec(
        *, kind: str, steps: list[dict[str, Any]], transitions: list[dict[str, Any]],
    ) -> str:
        """Validate a def spec BEFORE any write; return the derived `start_key`.

        Raises `WorkflowDefSpecError` (nothing written) when: `kind` is not in
        `WORKFLOW_KINDS`; a step `type` is not in `STEP_TYPES`; step keys are not
        unique; not exactly one step is marked `start: True`; or a transition
        `from`/`to` references a key that is not a declared step. This is the
        service invariant that lets the repository's inner `MATCH (start/from/to
        …)` always resolve for a valid spec (QUERIES.md §11 note).

        Two further invariants (K-024 U2) run **last**, after all of the above:
        a `human`/`wait` step must declare `config.waitsForHuman: true`, and a
        `cmp`-family transition guard must be structurally sound
        (`guards.validate_cmp` → `WorkflowConfigError`). Running them last is
        load-bearing: an older check must keep failing for its **own** reason, so a
        new invariant can never mask — or make vacuous a test of — a pre-existing one.
        """
        if kind not in WORKFLOW_KINDS:
            raise WorkflowDefSpecError(
                f"invalid workflow kind {kind!r} — must be one of "
                f"{sorted(WORKFLOW_KINDS)}"
            )

        keys: list[str] = []
        start_keys: list[str] = []
        for step in steps:
            skey = step["key"]
            keys.append(skey)
            stype = step.get("type")
            if stype not in STEP_TYPES:
                raise WorkflowDefSpecError(
                    f"invalid step type {stype!r} for step {skey!r} — must be one "
                    f"of {sorted(STEP_TYPES)}"
                )
            if step.get("start"):
                start_keys.append(skey)

        declared = set(keys)
        if len(declared) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise WorkflowDefSpecError(
                f"duplicate step key(s) {dupes} — step keys must be unique within a def"
            )
        if len(start_keys) != 1:
            raise WorkflowDefSpecError(
                f"a def must declare exactly one start step (start: true); "
                f"found {len(start_keys)} ({start_keys})"
            )

        for tr in transitions:
            for endpoint in ("from", "to"):
                if tr[endpoint] not in declared:
                    raise WorkflowDefSpecError(
                        f"transition {endpoint} {tr[endpoint]!r} is not a declared "
                        f"step key {sorted(declared)}"
                    )

        # ── K-024 U2 invariants — deliberately LAST (see the docstring) ─────────
        for step in steps:
            if step.get("type") not in WAITING_STEP_TYPES:
                continue
            cfg = _normalize_opaque(step.get("config"))
            if not isinstance(cfg, dict) or not cfg.get("waitsForHuman"):
                raise WorkflowDefSpecError(
                    f"step {step['key']!r} of type {step['type']!r} must declare "
                    f"config.waitsForHuman: true — a parking step without it "
                    f"self-loops until the step budget fails the run"
                )

        for tr in transitions:
            guard = _normalize_opaque(tr.get("guard"))
            # A guard that does not normalize to a dict, or that carries no `kind`, is
            # **not a declaration this validator owns** — `{"expr":"x>0"}` and an opaque
            # `"raw-string"` publish exactly as before. Only the cmp family is validated
            # here; `{"kind":"llm"}`/`{"kind":"expr"}` keep their drive-time semantics.
            if isinstance(guard, dict) and guard.get("kind") in CMP_KINDS:
                validate_cmp(guard)

        return start_keys[0]

    def publish_workflow_def(
        self, ctx: CallContext, *, key: str, version: str, name: str, kind: str,
        steps: list[dict[str, Any]], transitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate + publish a def version into the global `reference` graph. §11.1.

        The spec is validated first (`_validate_def_spec`) — on any violation
        nothing is written. `config`/`guard` are serialized to opaque strings.
        A step declares itself the start via `start: True` (exactly one required);
        that step's key becomes the repository `start_key`. Global write: no `ws`.
        """
        start_key = self._validate_def_spec(
            kind=kind, steps=steps, transitions=transitions
        )
        repo_steps = [
            {
                "key": s["key"], "type": s["type"],
                "config": _serialize_opaque(s.get("config")),
            }
            for s in steps
        ]
        repo_transitions = [
            {
                "from": tr["from"], "to": tr["to"], "on": tr["on"],
                "order": tr["order"], "guard": _serialize_opaque(tr.get("guard")),
            }
            for tr in transitions
        ]
        return self._repo.publish_def(
            key=key, version=version, name=name, kind=kind, start_key=start_key,
            steps=repo_steps, transitions=repo_transitions,
        )

    def materialize_def(
        self, ctx: CallContext, *, key: str, version: str
    ) -> dict[str, Any]:
        """Materialize a def@version from `reference` into `ctx.ws`. §11.4.

        Two-phase (plan F4, non-atomic across the graph boundary but retry-safe):
        read the def subgraph from the global `reference` graph, then write the
        snapshot into the workspace. Raises `WorkflowDefNotFoundError` when the
        def version was never published — nothing is written. Idempotent (the
        workspace MERGE no-ops on re-materialize).
        """
        sub = self._repo.read_def_subgraph(key=key, version=version)
        if sub is None:
            raise WorkflowDefNotFoundError(
                f"workflow def {key!r} version {version!r} not found in `reference` "
                f"— publish it before materializing"
            )
        return self._repo.materialize_snapshot(
            ctx.ws, key=key, version=version,
            name=sub["name"], kind=sub["kind"], start_key=sub["start_key"],
            steps=sub["steps"], transitions=sub["transitions"],
        )

    def get_workflow_def(
        self, ctx: CallContext, *, key: str, version: str | None = None
    ) -> dict[str, Any] | None:
        """Get a def's metadata (latest if `version` None). Global read. §11.3."""
        return self._repo.get_def(key=key, version=version)

    def list_workflow_defs(
        self, ctx: CallContext, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List published defs (global read). §11.3."""
        return self._repo.list_defs(limit=limit)

    def get_snapshot(
        self, ctx: CallContext, *, key: str, version: str
    ) -> dict[str, Any] | None:
        """Read a materialized snapshot subgraph from `ctx.ws`. §11.5."""
        return self._repo.get_snapshot(ctx.ws, key=key, version=version)

    def list_snapshots(
        self, ctx: CallContext, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List the workspace's materialized snapshots. §11.6."""
        return self._repo.list_snapshots(ctx.ws, limit=limit)

    # ── §12 Workflow execution — runs, step-runs & traces (M3 executor) ──────────
    #
    # The service mints the run id + start timestamp (server clock — never
    # client-supplied), resolves the trigger message's thread into the run `ctx`
    # (so a suspend can denorm it for the resume lookup, §2.4), starts the run
    # (repository, §12.1), then hands off to the injected executor which drives the
    # §2.1 loop. Reads are thin, `ctx.ws`-scoped pass-throughs. All Cypher lives in
    # `repository.py`; the engine logic lives in `executor.py`.

    def _require_executor(self):
        if self._executor is None:
            raise RuntimeError(
                "workflow executor is not wired — enable the workflow engine "
                "(app._build_default_app) before starting/resuming runs"
            )
        return self._executor

    def start_workflow_run(
        self, ctx: CallContext, *, def_key: str, version: str,
        trigger_msg_id: str, trace: bool = False,
    ) -> dict[str, Any]:
        """Start a run for a materialized def snapshot and drive it (FR-7/AC-1).

        Mints the run id + start clock, resolves the trigger message's thread into
        the initial `ctx` (`{"threadId": …}` — the resume denorm anchor, §2.4),
        starts the run at the snapshot's START step with the executor's step budget,
        then drives the §2.1 loop out-of-band-ready (synchronous here; a background
        task in the trigger wiring). Raises `WorkflowRunNotFoundError` when the
        snapshot or trigger message is missing (nothing is started).
        """
        executor = self._require_executor()
        msg = self._repo.get_message(ctx.ws, msg_id=trigger_msg_id)
        thread_id = msg["threadId"] if msg else ""
        run_id = self._id()
        started_at = self._clock()
        run_ctx = json.dumps(
            {"threadId": thread_id}, separators=(",", ":"), sort_keys=True
        )
        started = self._repo.start_run(
            ctx.ws, run_id=run_id, def_key=def_key, def_version=version,
            started_at=started_at, trigger_msg_id=trigger_msg_id, ctx=run_ctx,
            trace=trace, max_steps=executor.step_budget,
        )
        if started is None:
            raise WorkflowRunNotFoundError(
                f"cannot start run: snapshot {def_key!r}@{version!r} has no START "
                f"or trigger message {trigger_msg_id!r} is missing in this workspace"
            )
        status = executor.run(ctx, run_id=run_id)
        return {
            "runId": run_id, "status": status, "defKey": def_key,
            "defVersion": version, "trace": trace,
        }

    def resume_workflow_run(
        self, ctx: CallContext, *, run_id: str
    ) -> dict[str, Any]:
        """Resume a parked run on a human reply (§2.4/§6).

        Delegates to the executor's single-flight `waiting→running` CAS + drive;
        `status` is `None` when the CAS did not apply (the run was not waiting, or a
        concurrent reply already resumed it) — the caller treats that as a no-op.
        """
        executor = self._require_executor()
        status = executor.resume(ctx, run_id=run_id)
        return {"runId": run_id, "status": status}

    def link_step_emission(
        self, ctx: CallContext, *, step_run_id: str, msg_id: str
    ) -> dict[str, Any] | None:
        """Link `StepRun -[:PRODUCED]-> Message` (D2). QUERIES.md §12.6.

        The second query of the deliberately two-step emission (§3/§9): the message is
        posted via the guarded §4 write (`post_agent_answer`), then the emission is linked
        here. `PRODUCED` is distinct from K-013's `EMITTED` (§10). `None` when an endpoint
        is missing — a diagnosable, retry-able gap, not a torn thread. The `post_message`
        node tool (`tools.py`) drives this after posting.
        """
        return self._repo.link_step_emission(
            ctx.ws, step_run_id=step_run_id, msg_id=msg_id
        )

    def get_workflow_run(
        self, ctx: CallContext, *, run_id: str
    ) -> dict[str, Any] | None:
        """Read a run's state (RO pass-through). QUERIES.md §12.7."""
        return self._repo.get_run(ctx.ws, run_id=run_id)

    def read_workflow_step_runs(
        self, ctx: CallContext, *, run_id: str
    ) -> list[dict[str, Any]]:
        """The NEXT-ordered audit trail (RO pass-through). QUERIES.md §12.8."""
        return self._repo.read_step_runs(ctx.ws, run_id=run_id)

    def read_workflow_trace(
        self, ctx: CallContext, *, run_id: str
    ) -> list[dict[str, Any]]:
        """A debug run's reconstruction (RO pass-through). QUERIES.md §12.11."""
        return self._repo.read_trace(ctx.ws, run_id=run_id)

    def find_waiting_run_for_thread(
        self, ctx: CallContext, *, thread_id: str
    ) -> dict[str, Any] | None:
        """The resume lookup: the thread's parked (`waiting`) run, if any (RO). §12.9.

        Index-anchored on `WorkflowRun.status` + the denormed `waitingThreadId` (no new
        index). The trigger (§6) uses this to route a human reply to a waiting run before
        it considers @mention-to-start. `None` when nothing is parked in this thread.
        """
        return self._repo.find_waiting_run_for_thread(ctx.ws, thread_id=thread_id)
