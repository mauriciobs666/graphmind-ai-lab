"""The salesperson storefront's core — participant registry, join, token
verification and the per-participant turn-state map.

`docs/plans/salesperson-ui.md` S6 (§4.3 identity & isolation, §4.10 the join-time
profile write) and S7 (§4.7 the product-image manifest, §4.8 the two resets and
their quiesce, §5.2's `GET /shop/api/state` and `GET /shop/api/catalog`). The
`/shop/api` router that fronts this lives in `storefront_api.py` (S8); the turn
executor (S9) and the presenter surface (S10) extend this module further.

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
from pathlib import Path
from typing import Any

from redis import exceptions as redis_exceptions

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

# `list_catalog`'s **explicit** row bound (S7). `services.filter_products`
# defaults `limit=20`, which is correct for the seeded 15-product catalog and
# silently wrong at 21 — a truncated catalog with no error anywhere. The bound
# is kept rather than removed (the repository query needs one) but raised far
# above any plausible demo catalog, so it is a ceiling, not a page size.
CATALOG_LIMIT = 500

# §4.7: the manifest is built from the **served** directory
# (`<FALKORCHAT_STOREFRONT_DIR>/products/`), never the source tree — ship
# `dist/` alone and a source-tree manifest would be empty, every `imageUrl`
# `null`, and AC-11 would still pass because its negative branch masks the
# total failure of its positive one.
PRODUCTS_SUBDIR = "products"

# Accepted image extensions, **first match in this order** (§4.7).
IMAGE_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png")

# `imageUrl` is served from the SPA mount, matching Vite's `base: "/shop/"`.
IMAGE_URL_PREFIX = "/shop/products/"

# How often the reset waits on the turn map while quiescing. Small enough that
# a test can drive the whole wait, irrelevant to the 30 s production bound.
QUIESCE_POLL_S = 0.02


class StorefrontError(RuntimeError):
    """Base for the storefront's own refusals (mapped to HTTP by S8)."""


class DemoNotSeededError(StorefrontError):
    """The demo `Agent` is absent from the workspace, so `ensure_participant`
    wrote nothing at all (graph note §3, row 5 of the status table).

    Maps to `503`, naming `seed_demo.sh`. This is §4.9's readiness preflight
    failing *late* — the preflight should have caught it at boot, so a
    participant seeing this means the deployment came up mis-seeded.
    """


class QuiesceTimeoutError(StorefrontError):
    """A reset gave up waiting for that participant's turn to finish, and
    **changed nothing** (§4.8, graph note §7.1).

    Maps to `503`. This is the *only* reset failure that means "nothing
    changed" — a FalkorDB socket timeout is `ResetStateUnknownError` below, and
    conflating the two is the F8 defect this pair exists to prevent.
    """


class UnknownParticipantError(StorefrontError):
    """`repository.reset_participant` returned **zero rows**: the id is not a
    participant, or was already deleted (graph note §12's anomaly contract).

    Maps to the route's existing not-a-participant handling (`404`/`401`). Not
    an anomaly — indistinguishable from an already-deleted participant.
    """


class UnscopedParticipantError(StorefrontError):
    """`repository.reset_participant` returned `scoped=false` — the participant
    resolved but their own `Channel` did not, so the reset was a **guaranteed
    no-op** (graph note §4's G2, §12's anomaly contract).

    Maps to **`409`**, body carrying `code`, **never `200`**: nothing was reset
    and nothing will be until the graph is repaired.
    """

    code = "unscoped_participant"


class ResetStateUnknownError(StorefrontError):
    """The reset crossed `FALKORDB_SOCKET_TIMEOUT` on the way to FalkorDB, so
    **the delete may well have committed** (§4.8 F8, `docs/QUERIES.md` §18.7).

    Maps to **`504`**, never the quiesce `503`: the participant-facing meaning
    is *unknown*, never "nothing changed". `state` carries a fresh re-read of
    the graph when one was obtainable and is `None` when it was not — the
    re-read is another query against the same graph, and the stalled write that
    produced the first timeout is precisely what stalls it for a second
    `FALKORDB_SOCKET_TIMEOUT`. **A second timeout must not escape as a `500`**:
    the response is still `504`, simply with no state body. The state block is
    a courtesy the response carries when it can, not the contract.
    """

    code = "reset_state_unknown"

    def __init__(self, participant_id: str, *, state: dict[str, Any] | None) -> None:
        super().__init__(
            f"the reset of {participant_id!r} timed out on the way to FalkorDB "
            f"and may have committed"
        )
        self.participant_id = participant_id
        self.state = state


class UnknownOrderError(StorefrontError):
    """The order does not exist, or belongs to another participant — the two are
    deliberately indistinguishable (`services.order_belongs_to_customer`, graph
    note §10.2).

    Maps to **`404`**. §5.3 C10: an ordinary stale-button outcome, never an auth
    failure — the client must not clear a credential over it.
    """


class OrderTransitionRefusedError(StorefrontError):
    """The order is the participant's own, but its current status does not match
    the transition's guard — a stale, duplicate or out-of-order button press
    (`services.advance_order` returning `None`).

    Maps to **`409`**, carrying the order's current status so the client can
    repaint. Also §5.3 C10: never an auth failure.
    """

    def __init__(self, order_id: str, transition: str, status: str | None) -> None:
        super().__init__(
            f"order {order_id!r} cannot {transition} from status {status!r}"
        )
        self.order_id = order_id
        self.transition = transition
        self.status = status


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
        storefront_dir: str | Path | None = None,
        clock: Callable[[], int] = _default_clock,
        id_gen: Callable[[], str] = _default_participant_id,
    ) -> None:
        """`services` is the app's `Services`; the rest is configuration.

        `presenter_key`/`turn_workers`/`quiesce_s` are the plan's constructor
        contract (S6) and come from `config.STOREFRONT_*`. `ws`/`agent_id`/
        `locales`/`storefront_dir`/`clock`/`id_gen` default to the same config
        constants the production wiring uses and exist so the suite can drive
        this against `ws:test` with a pinned clock — the same injection seam
        `Services` itself has. **`ws` is not a client-facing knob**: §4.9
        collapsed the storefront's workspace onto `config.WS_ID` precisely so
        there is no second value to get wrong.

        `storefront_dir` (S7) is the **served** SPA build directory — the root
        of the product-image manifest, `<dir>/products/` (§4.7). `None` (the
        default when `FALKORCHAT_STOREFRONT_DIR` is unset) yields an empty
        manifest and therefore `imageUrl: null` on every catalog row, which is
        the correct answer for a deployment that serves no assets.
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
        directory = config.STOREFRONT_DIR if storefront_dir is None else storefront_dir
        self._storefront_dir = None if directory is None else Path(directory)
        self._clock = clock
        self._id = id_gen
        # The read-through cache (§4.3). Keyed by `participantId`, holds
        # token-free records, and is **never** consulted by `resolve_token`.
        self._records: dict[str, ParticipantRecord] = {}
        self._records_lock = threading.Lock()
        # The turn-state map (§4.4 measure 1). Absent key == idle.
        self._turns: dict[str, TurnState] = {}
        self._turns_lock = threading.Lock()
        # The product-image manifest (§4.7), built from the served directory
        # **once** — `None` until then. See `build_image_manifest`.
        self._image_manifest: dict[str, str] | None = None

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
    def storefront_dir(self) -> Path | None:
        """The served SPA build directory, or `None` when none is configured."""
        return self._storefront_dir

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

    # ── participant state (§5.2 `GET /shop/api/state`) ──────────────────────

    def get_state(self, ctx: CallContext) -> dict[str, Any]:
        """Everything the storefront repaints on a 2 s poll, in one place (S7).

        Four blocks, three of them repository reads through `Services` and the
        fourth from this process's own turn map:

        * `profile` — `services.get_profile`, always both fields (`name` and
          `deliveryAddress` are `None` before the participant supplies them).
        * `cart` — `services.get_cart`, lines priced live from `reference`.
        * `order` — **`services.get_current_order`**, the most recently *placed*
          order whatever its status, or `None`. It is a repository read
          (`docs/QUERIES.md` §18.8), deliberately **not** composed here from
          cart/profile parts: "current" is a graph question (`placedAt DESC`,
          ties by `orderId DESC`) and a storefront-side reconstruction would
          have to re-answer it on every poll and could disagree with the order
          route's own view.
        * `turn` — this participant's entry in the in-process turn map.

        `ctx.actor` is the participant id and also their `customerId`, so all
        three graph reads are scoped structurally rather than by a filter
        anyone could forget (§4.3).
        """
        return {
            "profile": self._services.get_profile(ctx),
            "cart": self._services.get_cart(ctx),
            "order": self._services.get_current_order(ctx),
            "turn": self.turn_state(ctx.actor).as_payload(),
        }

    # ── catalog + the product-image manifest (§4.7) ─────────────────────────

    @property
    def _catalog_ctx(self) -> CallContext:
        """The `CallContext` the catalog reads take.

        The catalog is **global `reference` data** — `repository.filter_products`
        and `repository.lookup_product` take no `ws` and no customer at all, and
        `services` accepts a `ctx` for interface parity without reading either
        field. Naming the demo `Agent` as the actor keeps that honest: no
        participant identity is invented for a read that has nothing to do with
        one, and no route can leak one participant's scope into another's
        catalog.
        """
        return CallContext(ws=self._ws, actor=self._agent_id)

    def _catalog_rows(self) -> list[dict[str, Any]]:
        """The whole catalog as `{productId, name, category, price}` rows.

        **Two calls, and the second one is a plan defect worked around here
        rather than fixed.** `services.filter_products` is the delivered catalog
        list (S2/S4, `docs/QUERIES.md` §15.2) and its projection is
        `{name, category, price}` — **it does not return `productId`**, while
        §5.2's `GET /shop/api/catalog` contract and §4.7's whole image design are
        keyed on exactly that field. So each row's id is resolved with a second,
        index-anchored point read (`services.lookup_product`, which *does*
        project `productId`). The alternatives were both worse from inside S7's
        two files: re-deriving the slug from the name would duplicate
        `scripts/seed_catalog.sh`'s `_slugify` in the serving path and fail
        silently the day either copy changed, and widening
        `repository.filter_products`'s projection edits a delivered step's file
        **and** changes what `tools.FilterProductsTool` hands the LLM. The cost
        is `1 + n` indexed reads of a static, 15-row global catalog on a route
        the client fetches once per session; the fix is one line in
        `repository.filter_products` if that is ever judged worth the two-file
        reach.

        `limit=CATALOG_LIMIT` is the **explicit** bound: the delivered default is
        `20`, right for 15 products and silently truncating at 21.

        A row whose name no longer resolves is dropped — reachable only by a
        catalog re-seed landing between the two reads, which is why it is
        silent rather than raised (the same posture `services._priced_cart_lines`
        takes for a cart line whose product vanished).
        """
        ctx = self._catalog_ctx
        rows = self._services.filter_products(
            ctx, category=None, min_price=None, max_price=None,
            limit=CATALOG_LIMIT,
        )
        resolved: list[dict[str, Any]] = []
        for row in rows:
            product = self._services.lookup_product(ctx, name=row["name"])
            if product is None:
                continue
            resolved.append({
                "productId": product["productId"], "name": row["name"],
                "category": row["category"], "price": row["price"],
            })
        return resolved

    def build_image_manifest(self) -> dict[str, str]:
        """`{productId: "/shop/products/<productId>.<ext>"}` from the **served**
        directory (§4.7), and store it on this instance.

        Lists `<storefront_dir>/products/`, keeps only `IMAGE_EXTENSIONS`, and
        **intersects the basenames with the catalog's `productId`s** — an asset
        with no product never becomes a URL, and a product with no asset never
        gets one. Extension precedence is `IMAGE_EXTENSIONS`'s own order, first
        match wins; the stored URL carries the file's real name, so an
        upper-case suffix on disk still resolves.

        Built at **startup only** (S8 calls this from the app's lifespan;
        `list_catalog` builds it once on first use if nobody did), so dropping
        an asset in later needs a restart — §4.7's stated operational note, not
        an oversight.

        An unset `FALKORCHAT_STOREFRONT_DIR`, a missing `products/`
        subdirectory, or an empty one all yield `{}` — every `imageUrl` is then
        `null` and the client renders its text-only card variant. That is the
        failure §4.7 calls out as invisible to AC-11, which is why S7's
        done-condition asserts a **non-empty** manifest against a real asset
        directory rather than merely a well-formed one.
        """
        manifest: dict[str, str] = {}
        directory = (
            None if self._storefront_dir is None
            else self._storefront_dir / PRODUCTS_SUBDIR
        )
        if directory is not None and directory.is_dir():
            by_id: dict[str, dict[str, str]] = {}
            for entry in directory.iterdir():
                if not entry.is_file():
                    continue
                suffix = entry.suffix.lower()
                if suffix not in IMAGE_EXTENSIONS:
                    continue
                by_id.setdefault(entry.stem, {}).setdefault(suffix, entry.name)
            for row in self._catalog_rows():
                available = by_id.get(row["productId"])
                if not available:
                    continue
                for extension in IMAGE_EXTENSIONS:
                    filename = available.get(extension)
                    if filename is not None:
                        manifest[row["productId"]] = IMAGE_URL_PREFIX + filename
                        break
        self._image_manifest = manifest
        return manifest

    def list_catalog(self) -> list[dict[str, Any]]:
        """The whole catalog with `imageUrl` attached (§5.2's `GET
        /shop/api/catalog`) — `"/shop/products/<productId>.<ext>"` when an asset
        was found for that product, `None` when there is none.

        Row order is `services.filter_products`'s own (`price ASC`).
        """
        if self._image_manifest is None:
            self.build_image_manifest()
        manifest = self._image_manifest or {}
        return [
            {**row, "imageUrl": manifest.get(row["productId"])}
            for row in self._catalog_rows()
        ]

    # ── the order lifecycle, gated on ownership (§4.6 / §5.2) ───────────────

    def advance_own_order(
        self, ctx: CallContext, *, order_id: str, transition: str
    ) -> dict[str, Any]:
        """Drive one lifecycle transition on **this participant's own** order.

        `services.order_belongs_to_customer` first, always
        (`docs/QUERIES.md` §18.9): `services.advance_order`'s guarded CAS is
        keyed on `orderId` alone, so without this gate anyone who learned
        another participant's `orderId` could cancel their order. No storefront
        route puts an `orderId` in a request body (§5.2), which makes the gate
        defence in depth — and the only thing standing between the two.

        Raises `UnknownOrderError` (`404`) when the order is unknown *or* is
        someone else's — the two are one answer by construction, and neither may
        be distinguishable from the other. Raises
        `OrderTransitionRefusedError` (`409`, carrying the current status) when
        the CAS guard does not match: a stale or duplicate button press. Both
        are ordinary stale-button outcomes, **never** auth failures (§5.3 C10).

        `services.advance_order`'s own `UnknownOrderTransitionError` is
        deliberately **not** caught: S8's Pydantic enum answers `422` before the
        call, so reaching it means a caller bypassed the model — a bug, not a
        runtime race.
        """
        ownership = self._services.order_belongs_to_customer(ctx, order_id=order_id)
        if not ownership["owned"]:
            raise UnknownOrderError(
                f"order {order_id!r} is not an order of {ctx.actor!r}"
            )
        result = self._services.advance_order(
            ctx, order_id=order_id, transition=transition
        )
        if result is None:
            raise OrderTransitionRefusedError(
                order_id, transition, ownership["status"]
            )
        return {"orderId": result["orderId"], "status": result["status"]}

    # ── "reset mine" (§4.8, graph note §4/§7/§12) ───────────────────────────

    def _await_quiesce(self, participant_id: str) -> bool:
        """Wait, bounded by `quiesce_s`, for this participant to have no turn in
        flight. `True` when they are idle, `False` on timeout.

        **Quiesce → delete, and the order is not interchangeable** (graph note
        §7.3). Deleting first and draining after produces a turn that consumes
        an LLM call and writes nothing: `post_message` raises
        `ThreadNotFoundError` against the vanished thread while
        `record_step_and_advance`/`append_trace_event` are anchored on deleted
        nodes and silently no-op.

        §4.8 also has this path *cancel* the participant's queued turn to
        shorten the wait. **S7 does not cancel, and the wait is not weakened by
        that**: the queue lives in S9's executor, which does not exist yet, and
        a queued turn still reaches a worker, completes, and clears its entry
        here — so waiting subsumes cancelling for correctness and differs only
        in latency. Dropping the turn-map entry as a stand-in would be actively
        wrong: it would report idle while the job was still queued, and the
        delete would then race exactly the turn this waits for. When S9 lands
        the queue, cancellation belongs *there*, in front of this wait, never in
        place of it.
        """
        deadline = time.monotonic() + self._quiesce_s
        while self.turn_in_flight(participant_id):
            if time.monotonic() >= deadline:
                return False
            time.sleep(QUIESCE_POLL_S)
        return True

    def reset_participant(self, participant: ParticipantRecord) -> dict[str, Any]:
        """"Reset mine" — quiesce, then one atomic delete (§4.8, graph note §4).

        The participant's **identity survives**: their `User` (token,
        `displayName`, `language`) and `Channel` stay, a fresh `Thread` is
        minted and `User.threadId` repointed, and everything else of theirs goes
        — transcript, runs, cursors, `Customer`/`Cart`/`Order`. Their token
        keeps working, which is why the client returns to a language step rather
        than the join screen.

        **Takes the authenticated `ParticipantRecord`, not a bare `ctx`**, and
        that is the whole reason no graph read is needed for the profile
        re-write below: S8 has just resolved this record *from the graph* on
        this very request (`resolve_token` re-reads every time), and
        `displayName`/`language` are exactly the fields the reset does not
        touch. Reading them back afterwards would cost a query for the same
        answer, and reading them from the registry cache would be worse — the
        cached `threadId` is stale the instant this returns.

        Returns `{"threadId": …, "language": …}` — §5.2's `200` body. Raises,
        for each of the four ways this can end other than success:

        * `QuiesceTimeoutError` → `503`, **nothing changed**.
        * `ResetStateUnknownError` → `504`, *unknown* — see F8 below.
        * `UnknownParticipantError` → `404`/`401`, zero rows.
        * `UnscopedParticipantError` → `409`, a guaranteed no-op.

        A `Thread` UNIQUE violation (`redis.exceptions.ResponseError`) from the
        duplicate-marker fail-safe is **not** caught: it propagates as a `5xx`
        and is never retried — a retry re-raises forever and the graph needs
        repair (graph note §4/§12).

        **F8 — a socket timeout means *unknown*, never "nothing changed"**
        (§4.8, `docs/QUERIES.md` §18.7). The module's `TIMEOUT` applies to reads
        only, so a slow reset is never truncated server-side; if one crosses
        `FALKORDB_SOCKET_TIMEOUT` the client raises while **the server commits
        the delete**. So a `redis.exceptions.TimeoutError` here never maps to
        the quiesce `503`: it becomes `ResetStateUnknownError`, carrying a fresh
        re-read of state — and carrying `None` when that re-read *also* times
        out, which is the likelier fault, not the exotic one, because FalkorDB
        serialises writes per graph and the stalled reset is precisely what
        stalls the re-read.

        **The profile name is re-written afterwards, and it is not cosmetic.**
        The `Customer` node goes with the reset while `User.displayName`
        survives, so without this call the profile panel shows an em-dash for a
        name the participant typed on the join screen and never withdrew
        (§2.4's FR-10 parity bar, graph note §12 item 1). Existing
        `services.save_profile` call, no new Cypher.
        """
        participant_id = participant.participant_id
        ctx = self.context_for(participant_id)
        if not self._await_quiesce(participant_id):
            raise QuiesceTimeoutError(
                f"a turn for {participant_id!r} did not finish within "
                f"{self._quiesce_s}s — nothing was reset"
            )
        try:
            status = self._repo.reset_participant(
                self._ws,
                participant_id=participant_id,
                new_thread_id=THREAD_ID_PREFIX + uuid.uuid4().hex,
                thread_title=THREAD_TITLE,
                now=self._clock(),
            )
        except redis_exceptions.TimeoutError as exc:
            raise self._reset_state_unknown(ctx, participant_id) from exc

        if status is None:
            self._cache_drop(participant_id)
            raise UnknownParticipantError(
                f"{participant_id!r} is not a participant of ws:{self._ws}"
            )
        if not status["scoped"]:
            raise UnscopedParticipantError(
                f"{participant_id!r} has no owned channel — nothing was reset"
            )

        self._services.save_profile(ctx, name=participant.display_name)
        self._cache_put(replace(participant, thread_id=status["threadId"]))
        return {"threadId": status["threadId"], "language": participant.language}

    def _reset_state_unknown(
        self, ctx: CallContext, participant_id: str
    ) -> ResetStateUnknownError:
        """Build F8's `504` after a reset timed out on the way to FalkorDB.

        Drops the cached record first — the delete may have committed, which
        makes the cached `threadId` wrong — then re-reads state so the response
        can report what the graph actually holds. A second `TimeoutError` from
        that re-read is swallowed into `state=None`: still a `504`, never a
        `500`, and never "nothing changed".
        """
        self._cache_drop(participant_id)
        try:
            state: dict[str, Any] | None = self.get_state(ctx)
        except redis_exceptions.TimeoutError:
            state = None
        return ResetStateUnknownError(participant_id, state=state)
