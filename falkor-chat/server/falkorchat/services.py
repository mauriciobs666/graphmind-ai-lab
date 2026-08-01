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
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from redis.exceptions import ResponseError

from . import config, proof_defs
from .config import CallContext
from .guards import CMP_KINDS, WorkflowConfigError, validate_cmp

# `MemberIdCollisionError`/`EmbeddingDimensionError`/`WorkflowDef*Error` are
# re-exported (redundant-alias idiom) as part of the service error surface: the
# repository owns them (the §2/§7 status-row contract, the §6 embedding-write
# validation, the §11 workflow error types); they live there only to avoid an
# import cycle.
from .repository import EmbeddingDimensionError as EmbeddingDimensionError
from .repository import MemberIdCollisionError as MemberIdCollisionError
from .repository import Repository
from .repository import StepBudgetExceededError as StepBudgetExceededError
from .repository import WorkflowDefConflictError as WorkflowDefConflictError
from .repository import WorkflowDefNotFoundError as WorkflowDefNotFoundError
from .repository import WorkflowDefSpecError as WorkflowDefSpecError
from .repository import WorkflowInputRejectedError as WorkflowInputRejectedError
from .repository import WorkflowRunNotFoundError as WorkflowRunNotFoundError
from .repository import WorkflowRunNotWaitingError as WorkflowRunNotWaitingError

# `MAX_CONFIG_LEN` is the opaque-payload bound declared once at the REST boundary;
# the service reuses that single number for the bound pydantic structurally CANNOT
# enforce — the size of the **merged** run ctx (plan m-5). `schemas.py` is a leaf
# module (pydantic only), so this import adds no cycle and no layering inversion:
# the constant flows boundary → service, never logic service → boundary.
from .schemas import MAX_CONFIG_LEN, MAX_DIFF_PREVIEW

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

# ── §12 run-ctx keys the engine owns — never caller-supplied (K-024 M-2/F-6) ────
# `threadId` is the resume denorm anchor: `_drive_loop`'s suspend copies it into
# `WorkflowRun.waitingThreadId`, and `trigger.py` step 2 resumes ANY waiting run
# whose `waitingThreadId` matches a posted message's thread — before it even looks
# at mentions. A caller-set `threadId` would therefore park a process run against a
# live chat thread and let the next ordinary human message drive it one step with
# no input and no guard data. `error` is the diagnostic note `fail_run` stamps.
# Enforced in the SERVICE (both the start ctx and submitted input), not only in
# `schemas.py`: MCP tools and direct service callers never see a pydantic model.
RESERVED_CTX_KEYS: frozenset[str] = frozenset({"threadId", "error"})

# Statuses a run may legitimately hold after the executor's M-1 fault net has
# stamped it (K-024 D-G / m-12). Anything else — notably a still-`running` zombie —
# means the fault escaped before `fail_run` landed, and the service re-raises so the
# caller gets a 500 rather than a success-shaped envelope describing a broken run.
TERMINAL_OR_PARKED_STATUSES: frozenset[str] = frozenset({"failed", "done", "waiting"})


class WorkflowEngineDisabledError(RuntimeError):
    """The workflow executor is not wired into this app (K-024 D-G, folds OQ-1).

    A **named** `RuntimeError` subclass on purpose: it maps cleanly to 503 in
    `app._register_error_handlers` without a blanket `RuntimeError` handler that
    would mask genuine bugs, and any existing `pytest.raises(RuntimeError)` stays
    green. Lives here, not in `repository.py`: wiring the engine is a service-layer
    concern and no repository code can raise it.
    """

# ── GraphRAG read posture (K-007 TIMEOUT / DESIGN §10) ──────────────────────────
# The FalkorDB global TIMEOUT default is 1000 ms and writes ignore it; GraphRAG
# reads (ANN seed + traversal) can legitimately run longer, so they pass a single
# per-query client `timeout=` override here rather than ad-hoc per call. Uncapped
# while the deployment keeps `TIMEOUT_MAX=0`.
RAG_QUERY_TIMEOUT_MS = 5000

# ── K-039 item 3: readiness "recent triage post-success" sample size ────────────
# Last-N terminal runs of the `@mention`-triggered def sampled by
# `check_demo_readiness`'s `postSuccess` field (repository.read_recent_post_success,
# QUERIES.md §12.15). A plain module constant, mirroring RAG_QUERY_TIMEOUT_MS above
# — nobody has asked to tune this per-deployment.
POST_SUCCESS_SAMPLE_SIZE = 20

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


def _load_json_dict(raw: Any) -> dict[str, Any]:
    """Deserialize an opaque stored `ctx` into a dict; `{}` for anything unusable.

    A run's `ctx` is a flat serialized string (rule 8). Never raise here — a run
    whose ctx got corrupted must still be advanceable rather than permanently 500.
    """
    parsed = _normalize_opaque(raw) if raw else None
    return parsed if isinstance(parsed, dict) else {}


def _str_values(value: Any) -> list[str]:
    """A defensive list-of-strings view over author-supplied config data."""
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


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


# ── §11 def/snapshot structure canonicalization + diff (K-031) ──────────────────
# Ordering and comparison live HERE, not in `repository.py`: the repository stays
# a 1:1 mirror of QUERIES.md, and §11.2's steps/transitions are unordered at the
# source by design (F6 — "the app reconstructs order"). Deterministic ordering is
# also what makes the structure endpoints diffable by hand (`curl … | jq`).


def _canonical_structure(
    raw: dict[str, Any], *, source: str, key: str, version: str
) -> dict[str, Any]:
    """Canonical, camelCase view of a repository structure read.

    Steps sorted by `key`; transitions by `(from, order, to, on)`; `startKeys`
    lexicographically, with `startKey` = `startKeys[0]` **after** sorting — the
    meta query has no `ORDER BY`, so an unsorted list would make `startKey`
    nondeterministic between two calls and could report a false divergence on
    list order alone (the exact failure the server-side diff exists to prevent).

    `stepCount`/`transitionCount` count what is **stored**. The identically-named
    fields on the publish/materialize receipt count what was **submitted**
    (`_PUBLISH_CYPHER`'s `count(st)`/`count(rel)` sit immediately after the
    `UNWIND`s). A divergence between the two is a **signal, not a bug** — see
    K-034.

    `config`/`guard` pass through verbatim (rule 8).
    """
    steps = sorted(
        (
            {"key": s["key"], "type": s["type"], "config": s["config"]}
            for s in raw["steps"]
        ),
        key=lambda s: s["key"],
    )
    transitions = sorted(
        (
            {
                "from": t["from"], "to": t["to"], "on": t["on"],
                "order": t["order"], "guard": t["guard"],
            }
            for t in raw["transitions"]
        ),
        key=lambda t: (t["from"], t["order"], t["to"], t["on"]),
    )
    start_keys = sorted(raw.get("start_keys") or [])
    out: dict[str, Any] = {
        "source": source, "key": key, "version": version,
        "name": raw["name"], "kind": raw["kind"],
        "startKey": start_keys[0] if start_keys else None,
    }
    # Omitted entirely for the ordinary single-`START` case; the routes declare
    # `response_model_exclude_unset=True` so "absent" survives serialization.
    if len(start_keys) > 1:
        out["startKeys"] = start_keys
    out["stepCount"] = len(steps)
    out["transitionCount"] = len(transitions)
    out["steps"] = steps
    out["transitions"] = transitions
    return out


def _diff_preview(value: Any) -> str | None:
    """A bounded preview of a difference value (never the payload)."""
    if value is None:
        return None
    text = ",".join(value) if isinstance(value, list) else str(value)
    if len(text) > MAX_DIFF_PREVIEW:
        return text[:MAX_DIFF_PREVIEW] + "…"
    return text


def _transition_path(tr: dict[str, Any]) -> str:
    return f"transitions[{tr['from']}->{tr['to']}@{tr['on']}#{tr['order']}]"


def _diff_structures(
    def_s: dict[str, Any], snap_s: dict[str, Any]
) -> list[dict[str, Any]]:
    """Enumerate the differences between two canonical structures.

    Pure function. Identity rules come from `_PUBLISH_CYPHER`'s `MERGE` keys, not
    from intuition: a **step**'s identity is its `key`; a **transition**'s is the
    4-tuple `(from, to, on, order)` and its only comparable payload is `guard`.
    (A client keying on `(from, to)` would mis-report an added parallel edge as a
    modified one — which is why this comparator is server-side.)

    `config`/`guard` are compared by **exact string equality**, never normalized
    JSON (rule 8 — the bytes in the graph are the contract).

    Path grammar: `meta.<field>` · `steps[<key>]` (presence) ·
    `steps[<key>].<type|config>` · `transitions[<from>-><to>@<on>#<order>]`
    (presence) · `transitions[…].guard`. Presence uses `"present"`/`"absent"`.
    """
    diffs: list[dict[str, Any]] = []

    for field in ("name", "kind", "startKey", "startKeys"):
        left, right = def_s.get(field), snap_s.get(field)
        if left != right:
            diffs.append({
                "path": f"meta.{field}",
                "def": _diff_preview(left),
                "snapshot": _diff_preview(right),
            })

    def_steps = {s["key"]: s for s in def_s["steps"]}
    snap_steps = {s["key"]: s for s in snap_s["steps"]}
    for step_key in sorted(set(def_steps) | set(snap_steps)):
        left, right = def_steps.get(step_key), snap_steps.get(step_key)
        if left is None or right is None:
            diffs.append({
                "path": f"steps[{step_key}]",
                "def": "present" if left is not None else "absent",
                "snapshot": "present" if right is not None else "absent",
            })
            continue
        for field in ("type", "config"):
            if left[field] != right[field]:
                diffs.append({
                    "path": f"steps[{step_key}].{field}",
                    "def": _diff_preview(left[field]),
                    "snapshot": _diff_preview(right[field]),
                })

    def _identity(tr: dict[str, Any]) -> tuple[str, str, str, int]:
        return (tr["from"], tr["to"], tr["on"], tr["order"])

    def_trs = {_identity(t): t for t in def_s["transitions"]}
    snap_trs = {_identity(t): t for t in snap_s["transitions"]}
    for ident in sorted(
        set(def_trs) | set(snap_trs), key=lambda i: (i[0], i[3], i[1], i[2])
    ):
        left, right = def_trs.get(ident), snap_trs.get(ident)
        if left is None or right is None:
            diffs.append({
                "path": _transition_path(left or right),
                "def": "present" if left is not None else "absent",
                "snapshot": "present" if right is not None else "absent",
            })
            continue
        if left["guard"] != right["guard"]:
            diffs.append({
                "path": f"{_transition_path(left)}.guard",
                "def": _diff_preview(left["guard"]),
                "snapshot": _diff_preview(right["guard"]),
            })

    return diffs


def _structural_diffs(diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter `_diff_structures`' output to topology-changing entries only (K-034).

    `meta.startKey`/`meta.startKeys` and a bare `steps[<key>]`/`transitions[...]`
    presence row are structural — exactly what `_PUBLISH_CYPHER`'s `MERGE`
    patterns can mint as *parallel* structure on a differing re-publish. `meta.name`/
    `meta.kind` and any `.type`/`.config`/`.guard`-suffixed row are property-only —
    `ON CREATE SET` already makes those safely create-only; re-publish stays a
    silent no-op on them (K-031-pinned, unchanged by this filter).
    """
    return [
        d for d in diffs
        if d["path"] in ("meta.startKey", "meta.startKeys")
        or (d["path"].startswith("steps[") and not d["path"].endswith((".type", ".config")))
        or (d["path"].startswith("transitions[") and not d["path"].endswith(".guard"))
    ]


def _check_no_structural_conflict(
    *, existing_raw: dict[str, Any] | None, candidate_raw: dict[str, Any],
    key: str, version: str, resource: str,
) -> None:
    """Raise `WorkflowDefConflictError` if `candidate_raw`'s topology differs from
    what's already stored at `(key, version)`. No-op when nothing is stored yet
    (`existing_raw is None`) — a first-time publish/materialize is unaffected.
    `resource` is "workflow def" or "workspace snapshot", for the message only.
    """
    if existing_raw is None:
        return
    existing = _canonical_structure(existing_raw, source="existing", key=key, version=version)
    candidate = _canonical_structure(candidate_raw, source="candidate", key=key, version=version)
    diffs = _structural_diffs(_diff_structures(existing, candidate))
    if diffs:
        paths = ", ".join(d["path"] for d in diffs)
        raise WorkflowDefConflictError(
            f"{resource} {key!r} version {version!r} is already published with a "
            f"different topology ({len(diffs)} difference(s): {paths}) — a published "
            f"version's structure is immutable; publish a new version instead of "
            f"editing this one, or inspect the mismatch with "
            f"GET /workflow-defs/{key}/versions/{version}"
        )


# ── FR-10 workspace readiness (web-api-coverage plan §3.1c / U2) ────────────────
# `check_demo_readiness` is the HTTP form of `scripts/verify_workflows.sh`: same
# expected pairs, same `diff_def_snapshot` + structure-read composition, same
# "cold graph key / absent def" tolerance, same problem-string wording — so the
# script and the endpoint can never disagree about what "ready" means.

# The exact pair `scripts/seed_workflows.sh` publishes/materializes, and the pair
# `verify_workflows.sh` checks. Importing it here (rather than each declaring its
# own list) is what keeps the two checks from drifting apart silently.
DEMO_EXPECTED_DEFS: tuple[tuple[str, str], ...] = (
    (config.TRIGGER_DEF_KEY, config.TRIGGER_DEF_VERSION),
    (proof_defs.ACCESS_REQUEST_DEF["key"], proof_defs.ACCESS_REQUEST_DEF["version"]),
)

_ABSENT_DIFF: dict[str, Any] = {
    "defPresent": False, "snapshotPresent": False, "inSync": False,
    "differences": [], "differenceCount": 0,
}


def _read_or_absent(fn: Callable[[], Any], absent: Any = None) -> Any:
    """Read-only probe: a cold graph key or an absent def is 'nothing there'.

    Mirrors `scripts/verify_workflows.sh`'s own `read()` helper byte-for-byte —
    a `WorkflowDefNotFoundError` (both sides absent) or a FalkorDB "empty key"
    `ResponseError` (the `reference`/`ws:{id}` graph key was never created) both
    mean "nothing there yet", not a fault worth a 500.
    """
    try:
        return fn()
    except WorkflowDefNotFoundError:
        return absent
    except ResponseError as exc:
        if "empty key" in str(exc):
            return absent
        raise


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

    def list_thread_participants(
        self, ctx: CallContext, *, thread_id: str
    ) -> list[dict[str, Any]]:
        """A thread's participants = its parent channel's roster (RO). QUERIES.md
        §2 "List thread participants" (K-036 — web-api-coverage FR-8).

        Validates the thread exists first, raising `ThreadNotFoundError`
        exactly like `_validate_and_derive_role` does before a §4 write —
        reused here rather than inventing a second idiom. Normalizes the
        repository's raw `type` (`labels(u)`, a list) down to a single `kind`
        string (`type[0]`) — the same `User`/`Agent` derivation
        `resolve_member_kinds` already does in Cypher for its own case
        (QUERIES.md §2), done here in Python because this read's repository
        method stays a literal mirror of its query's column names.
        """
        if not self._repo.thread_exists(ctx.ws, thread_id=thread_id):
            raise ThreadNotFoundError(thread_id)
        rows = self._repo.list_thread_participants(ctx.ws, thread_id=thread_id)
        return [
            {
                "memberId": row["memberId"],
                "displayName": row["displayName"],
                "kind": row["type"][0] if row["type"] else None,
            }
            for row in rows
        ]

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

        Three further invariants run **last**, after all of the above: a `human`/
        `wait` step must declare `config.waitsForHuman: true` (K-024 U2); a
        `cmp`-family transition guard must be structurally sound (K-024 U2,
        `guards.validate_cmp` → `WorkflowConfigError`); and a def must carry **at
        least one transition** (K-024 U4b, O-6). Running them last is
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

        # ── K-024 U4b (O-6) — also LAST, for the same reason ───────────────────
        # `repository._PUBLISH_CYPHER` ends in `UNWIND $transitions`, which collapses
        # the row stream to zero rows AFTER the `WorkflowDef`, its `Step`s and the
        # `START` edge have already been MERGEd. `publish_def` then indexes
        # `result_set[0]` ⇒ `IndexError` ⇒ 500 — and because publish is
        # `MERGE … ON CREATE SET`, re-publishing the corrected spec on the same
        # `(key, version)` is a silent no-op on the half-written def: the version is
        # permanently wrong and unrepairable. `POST /workflow-defs` accepts
        # `"transitions": []` (schemas.py defaults it), so this is reachable from the
        # public route. Rejecting here turns an unrepairable poisoning into a clean
        # 400 with nothing written. A terminal step is one with no *outgoing*
        # transition — never a def with no transitions at all.
        if not transitions:
            raise WorkflowDefSpecError(
                "a def must declare at least one transition — a zero-transition "
                "publish partially writes the def and then fails (see O-6); model a "
                "terminal outcome as a step with no outgoing transition instead"
            )

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

        **Every def must carry at least one transition** (O-6): the publish query
        ends in `UNWIND $transitions`, so an empty list would half-write the def and
        then fail — and since publish is `MERGE … ON CREATE SET`, that `(key,
        version)` could never be repaired by re-publishing. A terminal outcome is a
        step with no *outgoing* transition, not a def with none.

        **Topology-immutable per version (K-034).** After spec validation, a
        second read (`read_def_structure`) checks whether `(key, version)` is
        already published; if so, `_check_no_structural_conflict` rejects
        (`WorkflowDefConflictError`, 409, nothing written) a candidate whose step-
        key set, transition-identity set `(from,to,on,order)`, or start key differs
        from what's stored. A property-only difference (`name`, `kind`, a step's
        `type`/`config`, a transition's `guard`) is not a conflict — that stays
        create-only-on-properties, unchanged (K-031-pinned). This read-then-write
        is not atomic with the repository write below — see
        `WorkflowDefConflictError`'s docstring for the residual TOCTOU shape.
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
        existing_raw = self._repo.read_def_structure(key=key, version=version)
        _check_no_structural_conflict(
            existing_raw=existing_raw,
            candidate_raw={
                "name": name, "kind": kind, "start_keys": [start_key],
                "steps": repo_steps, "transitions": repo_transitions,
            },
            key=key, version=version, resource="workflow def",
        )
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
        def version was never published — nothing is written.

        **Topology-immutable per `(key, version)` against the workspace snapshot
        (K-034).** A third read (`read_snapshot_structure`) checks whether `ctx.ws`
        already carries a snapshot for this `(key, version)`; if so,
        `_check_no_structural_conflict` rejects (`WorkflowDefConflictError`, 409,
        nothing written) a candidate whose topology differs from what's stored.
        Property-only differences stay a silent no-op (unchanged `MERGE … ON
        CREATE SET` behavior). This read-then-write is not atomic with the
        repository write below — see `WorkflowDefConflictError`'s docstring for
        the residual TOCTOU shape (it applies to this method exactly as it does to
        `publish_workflow_def`, not only the reference-graph side).
        """
        sub = self._repo.read_def_subgraph(key=key, version=version)
        if sub is None:
            raise WorkflowDefNotFoundError(
                f"workflow def {key!r} version {version!r} not found in `reference` "
                f"— publish it before materializing"
            )
        existing_raw = self._repo.read_snapshot_structure(ctx.ws, key=key, version=version)
        _check_no_structural_conflict(
            existing_raw=existing_raw,
            candidate_raw={
                "name": sub["name"], "kind": sub["kind"],
                "start_keys": [sub["start_key"]],
                "steps": sub["steps"], "transitions": sub["transitions"],
            },
            key=key, version=version, resource="workspace snapshot",
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

    # ── §11 structure reads + def↔snapshot diff (K-031 observability) ────────
    # Black-box answers to three questions that otherwise need raw Cypher: is
    # what I think is published actually published; is the workspace running the
    # same thing; have `reference` and `ws:{id}` gone stale independently.
    # Read-only — this surface makes the current publish semantics *observable*,
    # it never changes or repairs them.

    def get_workflow_def_structure(
        self, ctx: CallContext, *, key: str, version: str
    ) -> dict[str, Any]:
        """Read a published def's full structure. Global read (no `ctx.ws`). §11.2.

        Raises `WorkflowDefNotFoundError` when the version was never published,
        consistent with `materialize_def`.
        """
        raw = self._repo.read_def_structure(key=key, version=version)
        if raw is None:
            raise WorkflowDefNotFoundError(
                f"workflow def {key!r} version {version!r} not found in `reference` "
                f"— publish it with POST /workflow-defs"
            )
        return _canonical_structure(
            raw, source="reference", key=key, version=version
        )

    def get_snapshot_structure(
        self, ctx: CallContext, *, key: str, version: str
    ) -> dict[str, Any] | None:
        """Read a materialized snapshot's full structure from `ctx.ws`. §11.5.

        `None` when absent (the route 404s), matching `get_workflow_def`'s
        passthrough style.
        """
        raw = self._repo.read_snapshot_structure(ctx.ws, key=key, version=version)
        if raw is None:
            return None
        return _canonical_structure(
            raw, source="workspace", key=key, version=version
        )

    def diff_def_snapshot(
        self, ctx: CallContext, *, key: str, version: str
    ) -> dict[str, Any]:
        """Compare `reference`'s def against `ctx.ws`'s snapshot at one version.

        `def` = `reference` (the *intended* truth), `snapshot` = `ws:{id}` (the
        **operational** truth — the snapshot is what the executor drives).

        One side missing is a first-class **200** carrying the presence flags: it
        is the documented trap after a `pytest`/`test_queries.sh` run wipes
        `reference` while `ws:{id}` survives, and a 404 there would push the
        operator straight back to raw Cypher. Both sides missing → 404.

        **Version-qualified**: this answers "same version, different content",
        never "wrong version". To detect a stale *version*, compare
        `GET /workflow-defs` against `GET /workspaces/{ws}/snapshots` first.
        """
        def_s = None
        raw_def = self._repo.read_def_structure(key=key, version=version)
        if raw_def is not None:
            def_s = _canonical_structure(
                raw_def, source="reference", key=key, version=version
            )
        snap_s = None
        raw_snap = self._repo.read_snapshot_structure(
            ctx.ws, key=key, version=version
        )
        if raw_snap is not None:
            snap_s = _canonical_structure(
                raw_snap, source="workspace", key=key, version=version
            )
        if def_s is None and snap_s is None:
            raise WorkflowDefNotFoundError(
                f"workflow def {key!r} version {version!r} is present neither in "
                f"`reference` nor in this workspace"
            )
        both = def_s is not None and snap_s is not None
        differences = _diff_structures(def_s, snap_s) if both else []
        return {
            "key": key,
            "version": version,
            "defPresent": def_s is not None,
            "snapshotPresent": snap_s is not None,
            "inSync": both and not differences,
            "differences": differences,
            "differenceCount": len(differences),
        }

    def check_demo_readiness(self, ctx: CallContext) -> dict[str, Any]:
        """Is `ctx.ws` ready to demo? The HTTP form of `verify_workflows.sh` (FR-10).

        For each `DEMO_EXPECTED_DEFS` pair: `diff_def_snapshot` gives presence +
        sync; `get_workflow_def_structure`/`get_snapshot_structure` add the
        Finding-3 multi-`START` tripwire (`"startKeys" in structure` — see
        K-034). Every read is wrapped in `_read_or_absent`, exactly like the
        script's `read()` helper, so a cold graph key reads as "absent", not as
        a 500. `ready` is `True` only when every def is fully present, in sync,
        and problem-free. `problems` reuses the script's own wording verbatim —
        this endpoint and the script must never disagree about what "ready"
        means.

        **`postSuccess` (K-039 item 3)** is a separate, purely informational
        field — a lagging, production-data signal ("of the last N triage runs,
        how many actually posted a reply") that is deliberately **not** folded
        into `ready` (plan `docs/plans/mention-reply-delivery.md` §3.3: `ready`
        stays a deterministic, configuration-only signal; a model-behavior
        metric would make it flip on LLM mood instead). `rate` is `None` when
        `sampleSize == 0` — a fresh workspace with no triage runs yet is
        "no data," not "0% healthy".
        """
        results: list[dict[str, Any]] = []
        ready = True
        for key, version in DEMO_EXPECTED_DEFS:
            label = f"{key}@{version}"
            diff = _read_or_absent(
                lambda k=key, v=version: self.diff_def_snapshot(ctx, key=k, version=v),
                absent=_ABSENT_DIFF,
            )
            def_present = diff["defPresent"]
            snap_present = diff["snapshotPresent"]
            in_sync = diff["inSync"]

            problems: list[str] = []
            if not def_present:
                problems.append(
                    f"{label}: not published in `reference` at this version"
                )
            if not snap_present:
                problems.append(
                    f"{label}: not materialized into ws:{ctx.ws} at this version"
                )
            if def_present and snap_present and not in_sync:
                problems.append(
                    f"{label}: reference def and ws:{ctx.ws} snapshot diverge "
                    f"({diff['differenceCount']} differences)"
                )

            sides = (
                ("reference def",
                 lambda k=key, v=version:
                     self.get_workflow_def_structure(ctx, key=k, version=v)),
                (f"ws:{ctx.ws} snapshot",
                 lambda k=key, v=version:
                     self.get_snapshot_structure(ctx, key=k, version=v)),
            )
            for side, reader in sides:
                structure = _read_or_absent(reader)
                if structure and "startKeys" in structure:
                    starts = structure["startKeys"]
                    problems.append(
                        f"{label}: {side} has {len(starts)} START edges "
                        f"({', '.join(starts)}) — see K-034"
                    )

            this_ready = def_present and snap_present and in_sync and not problems
            ready = ready and this_ready
            results.append({
                "key": key,
                "version": version,
                "defPresent": def_present,
                "snapshotPresent": snap_present,
                "inSync": in_sync,
                "problems": problems,
            })

        post_success = self._repo.read_recent_post_success(
            ctx.ws, def_key=config.TRIGGER_DEF_KEY,
            def_version=config.TRIGGER_DEF_VERSION, limit=POST_SUCCESS_SAMPLE_SIZE,
        )
        sample_size = post_success["sampleSize"]
        posted_count = post_success["postedCount"]
        return {
            "ready": ready,
            "defs": results,
            "postSuccess": {
                "defKey": config.TRIGGER_DEF_KEY,
                "defVersion": config.TRIGGER_DEF_VERSION,
                "sampleSize": sample_size,
                "postedCount": posted_count,
                "rate": (posted_count / sample_size) if sample_size else None,
                "status": (
                    "no-data" if sample_size == 0
                    else "ok" if posted_count == sample_size
                    else "degraded"
                ),
            },
        }

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
            raise WorkflowEngineDisabledError(
                "workflow executor is not wired — enable the workflow engine "
                "(app._build_default_app) before starting/resuming runs"
            )
        return self._executor

    def start_workflow_run(
        self, ctx: CallContext, *, def_key: str, version: str,
        trigger_msg_id: str | None = None, run_ctx: dict[str, Any] | None = None,
        trace: bool = False, max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Start a run for a materialized def snapshot and drive it (FR-7/AC-1).

        Two self-contained start paths — the §4 first/subsequent doctrine, never a
        conditional write (K-024 D-B / plan F-2):

          * **`trigger_msg_id` given** — the chat path, byte-identical to before:
            `repo.start_run` (§12.1) with the `TRIGGERED_BY` edge, and the trigger
            message's thread seeded into the run `ctx` (`{"threadId": …}`, the
            §2.4 resume denorm anchor).
          * **`trigger_msg_id is None`** — the process path (§12.12): no `Message`,
            no `Thread`, no `TRIGGERED_BY`. The initial ctx is the caller's
            `run_ctx` (default `{}`).

        `max_steps` lets a `process` def declare its own budget (D-H part c —
        `access-request@v1` passes 24); omitted ⇒ the executor's global default.
        Raises `WorkflowInputRejectedError` on a reserved ctx key (M-2, below),
        `WorkflowRunNotFoundError` when the snapshot (or trigger message) anchor
        misses — nothing is started in either case. A fault *during the drive* is
        caught per D-G: the run is already correctly terminal in the graph, so the
        caller gets `{"status": "failed", "error": …}`, not a traceback.
        """
        executor = self._require_executor()
        run_id = self._id()
        started_at = self._clock()
        budget = executor.step_budget if max_steps is None else max_steps

        if trigger_msg_id is not None:
            msg = self._repo.get_message(ctx.ws, msg_id=trigger_msg_id)
            thread_id = msg["threadId"] if msg else ""
            initial_ctx = json.dumps(
                {"threadId": thread_id}, separators=(",", ":"), sort_keys=True
            )
            started = self._repo.start_run(
                ctx.ws, run_id=run_id, def_key=def_key, def_version=version,
                started_at=started_at, trigger_msg_id=trigger_msg_id,
                ctx=initial_ctx, trace=trace, max_steps=budget,
            )
        else:
            # M-2/F-6: reject engine-owned keys BEFORE anything is written. A
            # caller-set `threadId` would park this run against a real chat thread
            # and let `trigger.py` step 2 advance it on the next ordinary message.
            self._reject_reserved_keys(run_ctx or {}, where="run ctx")
            initial_ctx = self._dump_ctx(run_ctx or {})
            started = self._repo.start_run_untriggered(
                ctx.ws, run_id=run_id, def_key=def_key, def_version=version,
                started_at=started_at, ctx=initial_ctx, trace=trace,
                max_steps=budget,
            )

        if started is None:
            raise WorkflowRunNotFoundError(
                f"cannot start run: snapshot {def_key!r}@{version!r} has no START "
                f"or trigger message {trigger_msg_id!r} is missing in this workspace"
            )
        status, error, _fault_ctx = self._drive_or_fault(
            ctx, run_id=run_id, drive=lambda: executor.run(ctx, run_id=run_id)
        )
        out = {
            "runId": run_id, "status": status, "defKey": def_key,
            "defVersion": version, "trace": trace,
        }
        if error is not None:
            out["error"] = error
        return out

    def submit_workflow_input(
        self, ctx: CallContext, *, run_id: str, input: dict[str, Any],
    ) -> dict[str, Any]:
        """Advance a parked (`waiting`) run with human / external-signal input.

        The non-chat half of D-B: a `human` or `wait` step parks because its
        outgoing guards read data that is not in `ctx` yet; this supplies that data
        and lets the same step re-execute against it.

        Order is load-bearing:
          1. `get_run` → absent ⇒ `WorkflowRunNotFoundError` (404).
          2. not `waiting` ⇒ `WorkflowRunNotWaitingError` (409) — nothing to unblock.
          3. **Validate before touching anything (D-H):** an **empty** `input` (the
             mistake a UI is most likely to emit — a submit with nothing filled in),
             then reserved keys, then the
             parked step's own declarations (`config.fields` / `config.signal` /
             `config.expects`), resolved from `run["atStepKey"]` + `get_snapshot`
             — two existing RO reads, **no new query**. A rejected value is a free
             400: nothing written, no step budget consumed, so ordinary human
             mistakes are unbounded and cost nothing.
          4. Merge **flat** into the run ctx (so guards read `ctx.decision`, not
             `ctx.input.decision`) and bound the **merged** size here — pydantic
             only ever sees the submitted input, never the ctx it merges into (m-5).
          5. **One query (D-F):** the merged ctx rides inside the resume CAS
             (§12.13). Zero rows ⇒ the run stopped being `waiting` under us (a
             concurrent submitter won) ⇒ 409 with **nothing written** — the input is
             visibly rejected, never silently lost, and never a wrong branch.

        A drive fault after that point is D-G's failed envelope, not a 500.
        """
        executor = self._require_executor()
        run = self._repo.get_run(ctx.ws, run_id=run_id)
        if run is None:
            raise WorkflowRunNotFoundError(
                f"workflow run {run_id!r} not found in this workspace"
            )
        if run.get("status") != "waiting":
            raise WorkflowRunNotWaitingError(
                f"workflow run {run_id!r} is {run.get('status')!r}, not 'waiting' — "
                f"there is nothing to unblock"
            )

        if not input:
            # m-A: `{}` passes every other rule (no reserved key, no undeclared key),
            # merges to a no-op, wins the resume CAS and re-executes the parked step
            # against unchanged ctx — no guard fires, the step re-parks, and one of
            # the run's steps is gone. D-H's "mistakes are free" has to cover this
            # one too, and at the service layer so MCP (OQ-2) inherits it.
            raise WorkflowInputRejectedError(
                "no input submitted — an empty input cannot advance a parked run "
                "and would consume a step of the run's budget"
            )

        self._reject_reserved_keys(input, where="input")
        self._validate_against_parked_step(ctx, run, input)

        merged = _load_json_dict(run.get("ctx"))
        merged.update(input)
        merged_json = self._dump_ctx(merged)
        if len(merged_json) > MAX_CONFIG_LEN:
            raise WorkflowInputRejectedError(
                f"merged run ctx would be {len(merged_json)} characters, over the "
                f"{MAX_CONFIG_LEN}-character bound"
            )

        status, error, fault_ctx = self._drive_or_fault(
            ctx, run_id=run_id,
            drive=lambda: executor.resume(
                ctx, run_id=run_id, run_ctx_json=merged_json
            ),
        )
        if status is None:
            # The CAS found the run no longer `waiting` — neither the flip nor the
            # ctx was written (§12.13's live-verified zero-row contract).
            raise WorkflowRunNotWaitingError(
                f"workflow run {run_id!r} was resumed concurrently — the input was "
                f"not applied; re-read the run and retry"
            )
        # On the clean path the merged ctx IS the graph's ctx (the CAS wrote exactly
        # it). On the fault path the engine's net has since rewritten `ctx` with its
        # diagnostic note, so the envelope reports **that** — status and ctx from one
        # post-fault observation, never a re-read status beside a hoped-for ctx.
        out = {
            "runId": run_id, "status": status,
            "ctx": merged if fault_ctx is None else fault_ctx,
        }
        if error is not None:
            out["error"] = error
        return out

    # ── §12 input helpers (K-024 D-G / D-H / M-2) ────────────────────────────

    @staticmethod
    def _reject_reserved_keys(payload: dict[str, Any], *, where: str) -> None:
        """M-2/F-6: engine-owned ctx keys are never caller-supplied."""
        offending = sorted(RESERVED_CTX_KEYS & set(payload))
        if offending:
            raise WorkflowInputRejectedError(
                f"reserved key(s) {offending} may not be set in the {where} — "
                f"they are owned by the engine (see RESERVED_CTX_KEYS)"
            )

    @staticmethod
    def _dump_ctx(value: dict[str, Any]) -> str:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    def _validate_against_parked_step(
        self, ctx: CallContext, run: dict[str, Any], input: dict[str, Any]
    ) -> None:
        """D-H: check submitted input against the parked step's own declarations.

        `suspend_run` does not clear `AT_STEP`, so a parked run always knows which
        step it sits on — `run["atStepKey"]` plus `get_snapshot` (§11.5) resolves the
        step's `config` with no new query.

        **Permissive fallback, deliberately:** a step that declares neither `fields`
        (`human`) nor `signal` (`wait`) accepts any non-reserved key. That is what
        makes the invariant non-retroactive — no existing def or fixture can start
        failing because it never declared a field list. An unresolvable parked step
        (snapshot or step gone, opaque config) degrades the same way: a workspace
        with a deleted snapshot must not turn every input into an engine error.
        """
        step_key = run.get("atStepKey")
        if not step_key:
            return
        snapshot = self._repo.get_snapshot(
            ctx.ws, key=run.get("defKey"), version=run.get("defVersion")
        )
        if snapshot is None:
            return
        step = next(
            (s for s in snapshot.get("steps", []) if s.get("key") == step_key), None
        )
        if step is None:
            return
        config = _normalize_opaque(step.get("config"))
        if not isinstance(config, dict):
            return

        # An explicitly-declared EMPTY `fields` list is still a declaration (accepts
        # nothing); only an ABSENT declaration triggers the permissive fallback.
        accepted: set[str] | None = None
        if step.get("type") == "human" and config.get("fields") is not None:
            accepted = set(_str_values(config.get("fields")))
        elif step.get("type") == "wait" and config.get("signal"):
            accepted = {str(config["signal"])}

        if accepted is not None:
            undeclared = sorted(set(input) - accepted)
            if undeclared:
                raise WorkflowInputRejectedError(
                    f"key(s) {undeclared} are not declared by the parked step "
                    f"{step_key!r} (accepts {sorted(accepted)})"
                )

        expects = config.get("expects")
        if isinstance(expects, dict):
            for field, allowed in expects.items():
                # A non-list `allowed` is not an allowed-value declaration — skip it
                # rather than invent a rule §3.4 does not state.
                if field not in input or not isinstance(allowed, list):
                    continue
                if input[field] not in allowed:
                    raise WorkflowInputRejectedError(
                        f"value {input[field]!r} for {field!r} is not one of "
                        f"{allowed} declared by step {step_key!r}"
                    )

    def _drive_or_fault(
        self, ctx: CallContext, *, run_id: str, drive: Callable[[], str | None],
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        """Run `drive`, converting a drive-time engine fault into D-G's envelope.

        Returns `(status, error, fault_ctx)`. On the clean path `error` and
        `fault_ctx` are `None`; on the fault path **both** `status` and `fault_ctx`
        come from the *same* post-fault `get_run` re-read, so every field of the
        envelope reports graph truth from one observation. Reporting a re-read
        status beside the caller's submitted ctx would only half-apply m-12's rule,
        and the two could then disagree in exactly the situation where a reader most
        needs them consistent — the failed envelope must carry the engine's own
        diagnostic `ctx` note, not what the caller hoped happened.

        The executor's M-1 fault net has already `fail_run`-stamped the run before
        re-raising, so the run **is** terminal and correct in the graph: a 500
        traceback would misreport a correctly-recorded terminal run as a server bug
        and break exactly the audit property DESIGN §6.3 exists to prove.

        Two deliberate limits:
          * Only `NotImplementedError` (the typed-handler seam) and
            `WorkflowConfigError` (a malformed guard reaching evaluation) are caught,
            and only out of the drive call. Faults raised *before* anything is
            written keep their own status codes. **Budget exhaustion is not here and
            must never be** — `_fail_budget` *returns* `"failed"` through the normal
            path and raises nothing (plan m-11).
          * The reported status comes from that re-read, never a guess. If the graph
            says the run is still `running` (or the run has vanished), the fault
            escaped before `fail_run` landed — re-raise, because a 500 is a better
            answer than a success envelope describing a zombie run (plan m-12).

        **The fault is logged with its stack before the envelope is built** (U4b M-A).
        Swallowing the exception here also swallows it for the *chat* start path:
        `start_workflow_run` is what `trigger.py` step 3 (@mention-to-start) calls, and
        `api._safe_run_workflow`'s `logging.exception` only ever sees what propagates
        out of it. Without this line a live `triage@v1` run that dies on an unwired
        judge leaves no log entry anywhere — only a `failed` run someone has to go
        looking for. The envelope itself is D-G verbatim; only the trace is restored.
        """
        try:
            return drive(), None, None
        except (NotImplementedError, WorkflowConfigError) as exc:
            logging.getLogger(__name__).exception(
                "workflow drive fault for run %s", run_id
            )
            run = self._repo.get_run(ctx.ws, run_id=run_id)
            status = run.get("status") if run else None
            if status not in TERMINAL_OR_PARKED_STATUSES:
                raise
            return (
                status,
                f"{type(exc).__name__}: {exc}",
                _load_json_dict(run.get("ctx")),
            )

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

        **An empty `thread_id` short-circuits to `None` without touching the graph**
        (plan F-5): a `kind:'process'` run parks with `waitingThreadId = ''` because
        it has no thread, so an empty-string lookup would match every parked process
        run and hand one of them a chat reply. No caller passes `''` today — this
        keeps it that way defensively, in the service rather than in the Cypher.
        """
        if not thread_id:
            return None
        return self._repo.find_waiting_run_for_thread(ctx.ws, thread_id=thread_id)

    def list_workflow_runs_for_thread(
        self, ctx: CallContext, *, thread_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Every run this thread has ever had, newest-first (RO). QUERIES.md
        §12.14 (K-036 — web-api-coverage FR-2).

        Validates the thread exists first, raising `ThreadNotFoundError`
        exactly like `_validate_and_derive_role` does before a §4 write —
        reused here rather than inventing a second idiom. Drives the browser's
        inline run cue (polled alongside the existing message catch-up loop,
        plan §3.2) — unlike `find_waiting_run_for_thread` (the resume lookup,
        one *parked* run at most), this returns the thread's full run history.
        """
        if not self._repo.thread_exists(ctx.ws, thread_id=thread_id):
            raise ThreadNotFoundError(thread_id)
        return self._repo.find_runs_for_thread(
            ctx.ws, thread_id=thread_id, limit=limit
        )
