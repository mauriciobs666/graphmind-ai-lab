"""The salesperson storefront's core — participant registry, join, token
verification and the per-participant turn-state map.

`docs/plans/salesperson-ui.md` S6 (§4.3 identity & isolation, §4.10 the join-time
profile write). The `/shop/api` router that fronts this lives in
`storefront_api.py` (S8); state/reset/catalog (S7) and the turn executor (S9)
extend this module.

**No Cypher lives here** (`falkor-chat/AGENTS.md` rule 1, `docs/SERVER.md` §1.2):
every graph touch goes through a `Repository`/`Services` method delivered by S4.

The one invariant this module exists to hold
--------------------------------------------
**The graph is the participant registry. The in-process map is a cache.**
`resolve_token` re-reads `User.tokenHash` from the workspace on *every* call and
never consults the cache, which buys two properties the demo depends on:

1. **Restart survival.** A single file write under `falkor-chat/` restarts
   uvicorn under `--reload`; with an authoritative in-process map that restart
   would invalidate every token and bounce every participant to a fresh
   `participantId` — losing their cart and order, not just their session, because
   `customerId == participantId` (§4.3). With the graph authoritative, a restart
   is invisible.
2. **A deleted participant stops resolving immediately.** "Reset everyone"
   deletes participant `User` nodes; a cache-first `resolve_token` would keep
   authenticating them out of stale memory until the process was restarted.

Both properties are pinned by tests in `tests/test_storefront.py` that go red
when `resolve_token` is made to answer from the cache — that mutation is the
review this module was written against, not a hypothetical.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from . import config
from .config import CallContext

# `secrets.token_urlsafe(32)` — 32 bytes of entropy rendered as 43 url-safe
# base64 characters (§4.3). The alphabet is `[A-Za-z0-9_-]`, so it contains no
# `.` and can never be confused with the `<participantId>.<token>` separator.
TOKEN_BYTES = 32

# `participantId = "p-" + uuid4().hex` (§4.3). Server-minted and unguessable:
# no route accepts one from a client, and `join` never derives it from
# participant input.
PARTICIPANT_ID_PREFIX = "p-"
CHANNEL_ID_PREFIX = "ch-"
THREAD_ID_PREFIX = "th-"
THREAD_TITLE = "Chat"

_BEARER_SCHEME = "bearer"
_CREDENTIAL_SEPARATOR = "."

# The three turn states `GET /shop/api/state` reports (§4.4 measure 1). `queued`
# carries a position; the other two are always position 0.
TURN_IDLE = "idle"
TURN_QUEUED = "queued"
TURN_THINKING = "thinking"


class StorefrontError(RuntimeError):
    """Base for the storefront's own refusals (mapped to HTTP by S8)."""


class DemoNotSeededError(StorefrontError):
    """The demo `Agent` is absent from the workspace, so `ensure_participant`
    wrote nothing at all (graph note §3, row 5 of the status table).

    Maps to `503`, naming `seed_demo.sh`. This is §4.9's readiness preflight
    failing *late* — the preflight should have caught it at boot, so a
    participant seeing this means the deployment came up mis-seeded.
    """


def _default_clock() -> int:
    """Server clock in milliseconds since the epoch (matches `services`)."""
    return int(time.time() * 1000)


def _default_participant_id() -> str:
    return PARTICIPANT_ID_PREFIX + uuid.uuid4().hex


def hash_token(token: str) -> str:
    """`sha256` hex of a participant token — the only form ever stored (§4.3).

    The raw token exists in exactly two places: the participant's browser, and
    the `ParticipantRecord` `join` hands back once. `User.tokenHash` holds this.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_bearer(bearer: str | None) -> tuple[str, str] | None:
    """Split `[Bearer ]<participantId>.<token>` into its two halves.

    `None` — never an exception — for every malformed shape: absent, empty,
    whitespace, a scheme that is not `Bearer`, no separator, an empty id half, an
    empty token half. The caller cannot distinguish "malformed" from "wrong", and
    that is deliberate: both are the same `401` and neither may leak which.

    Accepts the raw `Authorization` header value *or* the bare credential, so S8
    can hand this whatever FastAPI gave it. The scheme match is case-insensitive
    per RFC 7235; the credential is not touched.
    """
    if not bearer:
        return None
    candidate = bearer.strip()
    if not candidate:
        return None
    scheme, sep, rest = candidate.partition(" ")
    if sep:
        if scheme.lower() != _BEARER_SCHEME:
            return None
        candidate = rest.strip()
    participant_id, sep, token = candidate.partition(_CREDENTIAL_SEPARATOR)
    if not sep or not participant_id or not token:
        return None
    return participant_id, token


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
    """One participant's server-resolved scope.

    Everything a storefront route needs to act on someone's behalf, and nothing a
    client may name: `channelId`/`threadId` are resolved here from the token, so
    no route has to accept them (§4.3).

    `token` is the **raw credential**, populated only on the mint path (`join`)
    and only so the caller can hand it to the participant once. It is never
    populated by `resolve_token`, never read back from the graph (only its
    `sha256` is stored), and never kept in the registry cache — so a record that
    came from a graph read always carries `token is None`.
    """

    participant_id: str
    display_name: str
    language: str
    channel_id: str
    thread_id: str
    joined_at: int
    token: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ParticipantRecord:
        """Build from `repository.get_participant_record`'s projection."""
        return cls(
            participant_id=row["participantId"],
            display_name=row["displayName"],
            language=row["language"],
            channel_id=row["channelId"],
            thread_id=row["threadId"],
            joined_at=row["joinedAt"],
        )

    def without_token(self) -> ParticipantRecord:
        """The cacheable form — identical but carrying no raw credential."""
        return self if self.token is None else replace(self, token=None)


@dataclass(frozen=True, slots=True)
class TurnState:
    """One participant's agent-turn state, as `GET /shop/api/state` reports it."""

    state: str = TURN_IDLE
    queue_position: int = 0

    @property
    def in_flight(self) -> bool:
        """True while a turn is queued or running — the `409 TurnInProgress`
        gate (§4.4 measure 1a). Refusing a second post *before* the message
        write is the point: `trigger.maybe_trigger` resumes only a `waiting`
        run, so a message posted while the first turn is still `running` starts
        a **second** `WorkflowRun` on the same thread.
        """
        return self.state != TURN_IDLE

    def as_payload(self) -> dict[str, Any]:
        return {"state": self.state, "queuePosition": self.queue_position}


IDLE_TURN = TurnState()


class Storefront:
    """The storefront's participant registry and turn-state map.

    One instance per process, built by `create_app` (S8) and shared by every
    `/shop/api` route. All of its mutable state — the record cache and the turn
    map — is per-instance and lock-guarded, never module-global: FastAPI runs
    sync endpoints on a threadpool and S9 adds a `ThreadPoolExecutor` on top, so
    both maps are touched concurrently. Keeping them per-instance is also what
    makes the restart-survival test in `tests/test_storefront.py` mean anything
    — a second `Storefront` shares nothing with the first but the graph.
    """

    def __init__(
        self,
        services: Any,
        *,
        presenter_key: str,
        turn_workers: int,
        quiesce_s: float,
        ws: str | None = None,
        agent_id: str | None = None,
        locales: tuple[str, ...] | None = None,
        clock: Callable[[], int] = _default_clock,
        id_gen: Callable[[], str] = _default_participant_id,
    ) -> None:
        """`services` is the app's `Services`; the rest is configuration.

        `presenter_key`/`turn_workers`/`quiesce_s` are the plan's constructor
        contract (S6) and come from `config.STOREFRONT_*`. `ws`/`agent_id`/
        `locales`/`clock`/`id_gen` default to the same config constants the
        production wiring uses and exist so the suite can drive this against
        `ws:test` with a pinned clock — the same injection seam `Services`
        itself has. **`ws` is not a client-facing knob**: §4.9 collapsed the
        storefront's workspace onto `config.WS_ID` precisely so there is no
        second value to get wrong.
        """
        self._services = services
        # The repository is reached through `Services`, which owns it. S4 put the
        # participant registry on `Repository` (nine methods) and gave `Services`
        # only the two order wrappers the storefront also needs, so there is no
        # public service-level accessor for the other seven. Read once here, at
        # construction, rather than reaching through `services` at each call
        # site: one documented coupling instead of seven.
        self._repo = services._repo  # noqa: SLF001 — see above
        self._presenter_key = presenter_key
        self._turn_workers = turn_workers
        self._quiesce_s = quiesce_s
        self._ws = config.WS_ID if ws is None else ws
        self._agent_id = config.AGENT_ID if agent_id is None else agent_id
        self._locales = config.STOREFRONT_LOCALES if locales is None else locales
        self._clock = clock
        self._id = id_gen
        # The read-through cache (§4.3). Keyed by `participantId`, holds
        # token-free records, and is **never** consulted by `resolve_token`.
        self._records: dict[str, ParticipantRecord] = {}
        self._records_lock = threading.Lock()
        # The turn-state map (§4.4 measure 1). Absent key == idle.
        self._turns: dict[str, TurnState] = {}
        self._turns_lock = threading.Lock()

    # ── configuration readers (S7/S9/S10 wiring) ────────────────────────────

    @property
    def ws(self) -> str:
        return self._ws

    @property
    def locales(self) -> tuple[str, ...]:
        return self._locales

    @property
    def turn_workers(self) -> int:
        return self._turn_workers

    @property
    def quiesce_s(self) -> float:
        return self._quiesce_s

    @property
    def presenter_configured(self) -> bool:
        """Whether a presenter key is set at all.

        S10's login path must check this **before** comparing a submitted key:
        `hmac.compare_digest("", "")` is `True`, so an unconfigured deployment
        would otherwise hand the reset-everyone button to whoever posts an empty
        key first.
        """
        return bool(self._presenter_key)

    def context_for(self, participant_id: str) -> CallContext:
        """The `CallContext` every storefront route builds (§4.3).

        `actor` is the participant id, which is also their `customerId` — that
        identity is what makes cart/order/profile isolation structural rather
        than filtered.
        """
        return CallContext(ws=self._ws, actor=participant_id)

    # ── join (§4.3 provisioning + §4.10 the profile name) ───────────────────

    def join(self, display_name: str, language: str) -> ParticipantRecord:
        """Provision one participant and mint their credential.

        Two writes, deliberately (graph note §3.1): `ensure_participant` — the
        whole `User`+`Channel`+`Thread`+`MEMBER_OF` join in **one** atomic query,
        so no crash can leave a `Channel` without the `participantId` marker both
        resets scope on — and then `services.save_profile(name=display_name)`,
        which creates the `Customer` anchor eagerly (§4.10). A crash between them
        leaves a participant whose profile name is unset, which the next
        `save_profile` fixes and the UI already renders as an em-dash.

        Returns the record **with** its raw `token`; that is the only time the
        token exists outside the participant's browser.

        Bounds and the locale enum are **not** checked here: `POST
        /shop/api/session`'s Pydantic model owns them and answers `422` (§5.2,
        §5.3 C11). A second, differently-typed rejection path in this layer is
        exactly the "two places that can disagree" shape §4.9 rules out.
        """
        participant_id = self._id()
        token = secrets.token_urlsafe(TOKEN_BYTES)
        token_hash = hash_token(token)
        now = self._clock()

        status = self._repo.ensure_participant(
            self._ws,
            participant_id=participant_id,
            display_name=display_name,
            token_hash=token_hash,
            language=language,
            channel_id=CHANNEL_ID_PREFIX + participant_id,
            thread_id=THREAD_ID_PREFIX + participant_id,
            thread_title=THREAD_TITLE,
            agent_id=self._agent_id,
            now=now,
        )
        if status["agentMissing"]:
            raise DemoNotSeededError(
                f"the demo agent {self._agent_id!r} is not registered in "
                f"ws:{self._ws} — nothing was written. Seed it with "
                f"./scripts/seed_demo.sh {self._ws}"
            )

        if status["created"]:
            record = ParticipantRecord(
                participant_id=participant_id,
                display_name=display_name,
                language=language,
                channel_id=status["channelId"],
                thread_id=status["threadId"],
                joined_at=now,
                token=token,
            )
        else:
            # A replay: the id was already a participant. Unreachable in
            # production — `participantId` is a server-minted uuid4 that no
            # client can supply — and reachable only from a caller that pins
            # `id_gen`. It is handled rather than raised because
            # `ensure_participant` is idempotent by design and **is not a
            # token-rotation path**: it returned the *stored* ids and did not
            # write the fresh hash, so the token minted above would resolve to
            # `None`. Writing it through here keeps the one contract callers
            # depend on — *the token `join` returns always resolves* — and
            # provisioning stays idempotent either way: no second `User`,
            # `Channel` or `Thread` is created, and the original `joinedAt`
            # (which the row below carries) is not rewritten.
            row = self._repo.set_participant_record(
                self._ws,
                participant_id=participant_id,
                display_name=display_name,
                token_hash=token_hash,
                language=language,
            )
            record = replace(ParticipantRecord.from_row(row), token=token)

        # §4.10: the display name reaches the profile immediately, so the profile
        # panel never shows an em-dash for a name the participant typed thirty
        # seconds earlier. Existing service call, no new Cypher.
        self._services.save_profile(
            self.context_for(participant_id), name=display_name
        )

        self._cache_put(record)
        return record

    # ── token verification (§4.3 — the graph answers, always) ───────────────

    def resolve_token(self, bearer: str | None) -> ParticipantRecord | None:
        """Resolve `Bearer <participantId>.<token>` to a participant, or `None`.

        **Re-reads the graph on every call.** The registry cache is not
        consulted here and must never be: it is what would make a deleted
        participant keep resolving and a restarted process stop resolving anyone
        (see this module's docstring).

        `None` — never an exception, never a partial answer — for every failure:
        an absent, malformed or wrong-scheme header; an unknown participant id;
        a `User` that is not a participant (no `tokenHash`, e.g. `seed_demo.sh`'s
        `u1` or the lifespan's `config.USER_ID` node); a participant deleted by
        either reset; and a valid id carrying the wrong token — including another
        participant's token. The caller maps all of them to one `401`.

        The hash comparison is `hmac.compare_digest`, so a wrong token costs the
        same time whatever prefix it shares with the right one.
        """
        parsed = parse_bearer(bearer)
        if parsed is None:
            return None
        participant_id, token = parsed

        row = self._repo.get_participant_record(
            self._ws, participant_id=participant_id
        )
        if row is None:
            # Unknown, or no longer a participant. Evict any cached record: the
            # cache must not outlive the registry entry it mirrors.
            self._cache_drop(participant_id)
            return None

        stored_hash = row.get("tokenHash")
        if not isinstance(stored_hash, str):
            return None
        if not hmac.compare_digest(stored_hash, hash_token(token)):
            return None

        record = ParticipantRecord.from_row(row)
        # **This write is load-bearing — do not delete it as "the cache is never
        # read here anyway".** It is the *refresh* half of the read-through
        # cache, and the only one there is: `lookup` populates on a miss but
        # never re-reads a hit, so without this line a record that changed in
        # the graph after a participant's first `lookup` would be served stale
        # to `lookup`'s callers indefinitely, while `resolve_token` itself kept
        # returning the current one. Every authenticated request refreshes the
        # entry as a side effect of the read it already performed, at no extra
        # query. Pinned by
        # `test_resolving_refreshes_the_cache_so_lookup_never_serves_a_stale_record`
        # (review `docs/reviews/salesperson-ui-impl.md` Pass 6, S6-1 — deleting
        # this line passed all 2439 tests before that test existed).
        self._cache_put(record)
        return record

    # ── the registry cache (read-through, never an auth path) ───────────────

    def lookup(self, participant_id: str) -> ParticipantRecord | None:
        """A participant's record by id — cache first, graph on miss.

        **Not an authentication path.** It answers "who is `p-…`", not "is this
        credential valid", and the caller must already have resolved that id
        from a token (or from the presenter roster). Only `resolve_token`
        decides whether a credential is good, and it never comes through here.

        This is what the cache is *for*: a worker thread holding a
        `participantId` (S9's turn queue, S7's post-reset profile re-write) needs
        `displayName`/`threadId`/`language` without a graph round-trip per call.
        """
        with self._records_lock:
            cached = self._records.get(participant_id)
        if cached is not None:
            return cached
        row = self._repo.get_participant_record(
            self._ws, participant_id=participant_id
        )
        if row is None:
            return None
        record = ParticipantRecord.from_row(row)
        self._cache_put(record)
        return record

    def cached_ids(self) -> frozenset[str]:
        """The ids currently cached — diagnostics and tests only. Never a
        roster: the roster is `repository.list_participants`, which reads the
        graph (S10)."""
        with self._records_lock:
            return frozenset(self._records)

    def forget(self, participant_id: str) -> None:
        """Drop one participant from the cache — for the reset paths (S7/S10),
        which delete registry entries out from under it. Not required for
        correctness of `resolve_token` (that re-reads regardless); it keeps the
        cache from holding records for participants that no longer exist."""
        self._cache_drop(participant_id)

    def forget_all(self) -> None:
        """Drop every cached record — "reset everyone" (S10)."""
        with self._records_lock:
            self._records.clear()

    def _cache_put(self, record: ParticipantRecord) -> None:
        with self._records_lock:
            self._records[record.participant_id] = record.without_token()

    def _cache_drop(self, participant_id: str) -> None:
        with self._records_lock:
            self._records.pop(participant_id, None)

    # ── the turn-state map (§4.4 measure 1) ─────────────────────────────────

    def turn_state(self, participant_id: str) -> TurnState:
        """This participant's turn state — `idle` when they have none.

        In-process by design: a turn is bound to the worker driving it in *this*
        process, so unlike the registry there is nothing durable to read. A
        restart drops every in-flight turn, which is correct — the workers that
        were driving them are gone too.
        """
        with self._turns_lock:
            return self._turns.get(participant_id, IDLE_TURN)

    def set_turn_state(
        self, participant_id: str, state: str, *, queue_position: int = 0
    ) -> TurnState:
        """Record a participant's turn state; `idle` clears the entry.

        S9 drives this from `enqueue_turn` and the worker; S8 reads it for
        `GET /shop/api/state` and for the `409 TurnInProgress` gate.
        """
        if state == TURN_IDLE:
            self.clear_turn(participant_id)
            return IDLE_TURN
        turn = TurnState(state=state, queue_position=queue_position)
        with self._turns_lock:
            self._turns[participant_id] = turn
        return turn

    def clear_turn(self, participant_id: str) -> None:
        with self._turns_lock:
            self._turns.pop(participant_id, None)

    def clear_all_turns(self) -> None:
        """Drop every turn entry — the reset paths, after quiesce (S7/S10)."""
        with self._turns_lock:
            self._turns.clear()

    def turn_in_flight(self, participant_id: str) -> bool:
        """Whether this participant already has a turn queued or running."""
        return self.turn_state(participant_id).in_flight
