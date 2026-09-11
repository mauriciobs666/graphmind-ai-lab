"""The salesperson storefront's core — participant registry, join, token
verification and the per-participant turn-state map.

`docs/plans/salesperson-ui.md` S6 (§4.3 identity & isolation, §4.10 the join-time
profile write), S7 (§4.7 the product-image manifest, §4.8 the two resets and
their quiesce, §5.2's `GET /shop/api/state` and `GET /shop/api/catalog`) and S9
(§4.4 measure 1's bounded turn executor and its queue-position accounting). The
`/shop/api` router that fronts this lives in `storefront_api.py` (S8); the
presenter surface (S10) extends this module further.

**No Cypher lives here** (`falkor-chat/AGENTS.md` rule 1, `docs/SERVER.md` §1.2):
every graph touch goes through a `Repository`/`Services` method delivered by S4.

The one invariant this module exists to hold
--------------------------------------------
**The graph is the sole participant registry — nothing in this process holds
a second copy of it.** `resolve_token` re-reads `User.tokenHash` from the
workspace on *every* call, which buys two properties the demo depends on:

1. **Restart survival.** A single file write under `falkor-chat/` restarts
   uvicorn under `--reload`; with an authoritative in-process map that restart
   would invalidate every token and bounce every participant to a fresh
   `participantId` — losing their cart and order, not just their session, because
   `customerId == participantId` (§4.3). With the graph authoritative, a restart
   is invisible.
2. **A deleted participant stops resolving immediately.** "Reset everyone"
   deletes participant `User` nodes; with no in-process record surviving that
   delete, the very next `resolve_token` call sees the graph's current state
   and returns `None` — there is nothing stale to keep authenticating from.

Both properties are pinned by tests in `tests/test_storefront.py` that go red
when `resolve_token` is made to answer from an in-process map instead of
re-reading the graph — that mutation is the review this module was written
against, not a hypothetical.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from redis import exceptions as redis_exceptions

from . import config
from .config import CallContext

_log = logging.getLogger(__name__)

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

# The thread-name prefix every turn-executor worker carries (§4.4 measure 1).
# Named rather than left as `ThreadPoolExecutor-N` because telling a turn thread
# apart from anyio's request threadpool in a stack dump *is* the measure: the
# whole point of the split is that agent turns are not on the limiter the poll
# reads share.
TURN_THREAD_PREFIX = "storefront-turn"


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
    the response is still `504`, with `state` present and `null` — the key is
    never dropped, only its value, matching `participants` on the reset-all
    route's own second-timeout case. The state block is a courtesy the
    response carries when it can, not the contract.
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
    `sha256` is stored) — so a record that came from a graph read always
    carries `token is None`.
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


@dataclass(frozen=True, slots=True)
class TurnBooking:
    """One accepted turn's **ownership token** and its place in the **arrival
    order** — one value doing both jobs, because both answer the same question
    of a map that holds one slot per participant: *which* turn is this?

    `ordinal` comes from `Storefront`'s own strictly monotonic counter, taken
    under the turn lock, never reset and never reused (`reserve_turn`). Both
    jobs need exactly that. As an ordering key it decides §5.2's queue
    position; as an ownership token it is what every map write is checked
    against — so a value two live bookings can share mis-places the line *and*
    lets a worker clear a slot that is not its own, which is
    `docs/reviews/salesperson-ui-impl.md` `## Pass 17`, P17-1 in a second
    spelling. **`len(self._turns)` is a plausible reading of "arrival ordinal"
    and is wrong for both**: it restarts after `clear_all_turns()`, so a live
    booking and a fresh one can hold the same value (`## Pass 18`, P18-2).

    A value class rather than a bare `int` for one reason worth stating: the
    first booking a process ever makes has ordinal `0`, and `if not booking`
    would drop it.
    """

    ordinal: int


@dataclass(frozen=True, slots=True)
class TurnState:
    """One participant's agent-turn state, as `GET /shop/api/state` reports it.

    **It carries no queue position, deliberately.** §5.2 defines that number as
    the participant's index in the waiting line, which is a property of the
    *map* rather than of one entry, so `Storefront.turn_payload` derives it on
    every read and nothing stores it. S9a stored it here, taken once at
    booking — `len(self._turns)`, *how many accepted turns were unfinished
    when this one arrived*. That counts the **running** turn as a place in the
    line, which §5.2 excludes, so it was wrong at **every** `turn_workers`
    rather than at some of them: at `turn_workers=1` three arrivals read
    `[0, 1, 2]` where the definition gives `[0, 0, 1]`, and at the delivered
    default of 4 — where it is loudest — a fifth arrival first in line was
    told `4`. It never counted down either. The two numbers do coincide while
    **nothing** is running — `turn_payload` then counts the same earlier
    `TURN_QUEUED` entries `len(self._turns)` did — but that is a transient the
    first worker ends, not a `turn_workers` at which the stored number was
    right (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`, P17-2,
    reproduced; `## Pass 20`, P20-3 for the "right at `turn_workers=1`" claim
    this replaces; `## Pass 21`, P21-7 for the bound on *when* they diverge).

    **It does carry the turn's `Future`, and that placement is the plan's, not
    a preference** — §5.1's S9 row: the booking token "is also the handle S9b's
    cancellation hangs its `Future` on — **one map, not two**"
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`, P17-9). A second
    `participantId → Future` dict beside `_turns` would need its own lock, its
    own ownership rule and its own clear on all four of this map's exits
    (`release_turn`, the `finally`, `clear_all_turns`, reset-mine's cancel),
    and every one of those is a place the two could disagree — the disagreement
    being precisely "the map says idle while a job is still queued", which is
    what §4.8 forbids. Here the future cannot outlive its entry: it arrives on
    the entry and leaves with it.

    `None` is not "no turn" — it is **not cancellable through this map**, and
    the two are different. An entry is created by `reserve_turn`, *before* the
    message write and therefore before anything has been submitted, so a
    `queued` entry with `future=None` is the ordinary state of a post still
    inside `services.post_message`. `enqueue_turn` attaches the future after
    `submit` returns. Nothing outside this module reads the field, and
    `as_payload` deliberately does not carry it: it is a process-local handle,
    not part of §5.2's `turn` block.
    """

    state: str = TURN_IDLE
    booking: TurnBooking | None = None
    future: Future[None] | None = None

    @property
    def in_flight(self) -> bool:
        """True while a turn is queued or running — the `409 TurnInProgress`
        gate (§4.4 measure 1a). Refusing a second post *before* the message
        write is the point: `trigger.maybe_trigger` resumes only a `waiting`
        run, so a message posted while the first turn is still `running` starts
        a **second** `WorkflowRun` on the same thread.
        """
        return self.state != TURN_IDLE

    def as_payload(self, queue_position: int) -> dict[str, Any]:
        """§5.2's `turn` block, minus `lastTurn`. `queue_position` is handed in
        by `Storefront.turn_payload`, the only scope that can see the rest of
        the line — an entry cannot know its own place in it. `lastTurn` is
        merged in by that same caller, from a latch this class does not hold —
        an entry cannot report on a turn that already deleted it, either.
        """
        return {"state": self.state, "queuePosition": queue_position}


IDLE_TURN = TurnState()


class Storefront:
    """The storefront's participant registry and turn-state map.

    One instance per process, built by `create_app` (S8) and shared by every
    `/shop/api` route. Its mutable state — the turn-state map — is per-instance
    and lock-guarded, never module-global: FastAPI runs sync endpoints on a
    threadpool and the turn executor below runs on top of that, so the map is
    touched concurrently. Participant identity itself is not held as mutable
    state here at all — every resolution reads straight through to the graph —
    which is what makes the restart-survival test in `tests/test_storefront.py`
    mean anything: a second `Storefront` shares nothing with the first but the
    graph.
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
        trigger: Any | None = None,
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

        `trigger` is the app's `WorkflowTrigger` (§4.4 measure 1), and it is
        **the turn worker's only collaborator** — the storefront reaches the
        workflow layer through `trigger.maybe_trigger` and never through its own
        `self._services`. `None` — the default, and what an app built without
        `FALKORCHAT_WORKFLOW_ENABLED` gets — makes a turn a no-op that still
        occupies its queue slot: the `409` gate, the queue accounting and the
        quiesce wait are properties of the *post*, not of the engine, so they
        must not switch off with it.
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
        # The turn-state map (§4.4 measure 1). Absent key == idle.
        self._turns: dict[str, TurnState] = {}
        self._turns_lock = threading.Lock()
        # Where every booking's arrival ordinal comes from: strictly
        # monotonic, never reset, never reused, advanced under `_turns_lock`
        # (§5.1's S9 row). **Per-`Storefront`, never module-level** — the rule
        # this class's docstring already gives for every other piece of its
        # mutable state, and what keeps a second `Storefront` sharing nothing
        # with the first but the graph.
        self._turn_ordinals = itertools.count()
        # §4.4 measure 1's bounded turn executor, and the trigger its workers
        # drive. Constructed eagerly rather than on first use because
        # `ThreadPoolExecutor` starts no thread until the first `submit`, so a
        # `Storefront` that never runs a turn costs one object and no thread.
        self._trigger = trigger
        self._executor = ThreadPoolExecutor(
            max_workers=turn_workers, thread_name_prefix=TURN_THREAD_PREFIX
        )
        # Executor **lifecycle** — is the pool still accepting work — set
        # once by `shutdown_turns()` and never cleared. Deliberately a plain
        # attribute under **no lock**, because monotonic `False → True` makes
        # a stale `True` unreachable and leaves only a stale `False`, which
        # lands on the safe side either way: the submit is attempted, and it
        # either succeeds (the executor has not stopped yet, and
        # `shutdown(wait=True)` drains the turn) or is refused before it
        # queues anything (`concurrent/futures/thread.py:170`), costing a
        # leaked booking rather than the orphaned live turn `enqueue_turn`'s
        # docstring rules out. This is **not** S10's reset-all stop-intake
        # gate, which goes up and comes back down; one attribute doing both
        # jobs would make reset-all refuse turns permanently
        # (`docs/plans/salesperson-ui.md` §5.1's S9 row).
        self._turns_shutdown = False
        # The dead-turn latch (§5.2 *The dead-turn signal*, §5.1's S9 row).
        # A **separate** per-participant set — deliberately not a field on
        # `TurnState` — because `set_turn_state(idle)` deletes the `_turns`
        # entry by delivered design, which would wipe the signal at the exact
        # instant it is earned. Membership means "that participant's last
        # completed turn died without a reply"; composed into `turn_payload`
        # alongside (not inside) the `_turns` lookup. Its own lock rather than
        # `_turns_lock`, since it is written from the worker's failure-isolation
        # block (`_run_turn`, holding no turn-map lock at that point) and read
        # by every poll.
        self._last_turn_failed: set[str] = set()
        self._last_turn_failed_lock = threading.Lock()
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

        return record

    # ── token verification (§4.3 — the graph answers, always) ───────────────

    def resolve_token(self, bearer: str | None) -> ParticipantRecord | None:
        """Resolve `Bearer <participantId>.<token>` to a participant, or `None`.

        **Re-reads the graph on every call, full stop** — no caller, including
        `resolve_token` itself, is ever handed a record older than the read it
        just performed.

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
            return None

        stored_hash = row.get("tokenHash")
        if not isinstance(stored_hash, str):
            return None
        if not hmac.compare_digest(stored_hash, hash_token(token)):
            return None

        record = ParticipantRecord.from_row(row)
        return record

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

    def turn_payload(self, participant_id: str) -> dict[str, Any]:
        """§5.2's `turn` block for one participant — **derived on read, never
        stored**.

        `queuePosition` is the 0-based index in the waiting line: how many
        *other* accepted turns are ahead of theirs, and the line is ordered by
        booking. It is `0` for `idle` and for `thinking`, where it is a
        constant rather than a position — an idle participant is in no line and
        a running turn has already left it — so only `queued` entries booked
        **earlier** than this one are counted. `0` on a `queued` turn is
        ordinary and load-bearing: *first in line*.

        Two properties follow, and they are why §5.2 requires the derivation
        rather than a number taken at booking. It is **correct at every
        `turn_workers`** — a `thinking` turn occupies a worker, not a place in
        line, so a fifth arrival behind four *running* turns reads `0` and not
        `4` — and it **counts down** as the queue drains, which is the whole
        difference between a queue position and an indefinite spinner wearing a
        number (§4.4 measure 1). The cost is one scan of a map bounded by the
        participant count, once per poll.

        One accepted under-count, stated rather than discovered: reset-all's
        `clear_all_turns()` empties the map under workers that are still
        running, so for at most one turn's duration a fresh arrival can read
        `queued`/`0` while every worker is busy (§5.2 *One bound*).

        **`lastTurn` is composed in here too, and it is a separate read** —
        `self._last_turn_failed`, guarded by its own lock, never the `_turns`
        entry above. The two are independent questions (*is a turn running
        now* versus *did the last one die without a reply*), and composing
        them here rather than inside `TurnState.as_payload` is what keeps the
        latch out of the map entry `set_turn_state(idle)` deletes.
        """
        with self._turns_lock:
            turn = self._turns.get(participant_id)
            if turn is None:
                payload = IDLE_TURN.as_payload(0)
            elif turn.state != TURN_QUEUED:
                payload = turn.as_payload(0)
            else:
                ordinal = turn.booking.ordinal
                ahead = sum(
                    1
                    for other in self._turns.values()
                    if other.state == TURN_QUEUED and other.booking.ordinal < ordinal
                )
                payload = turn.as_payload(ahead)
        payload["lastTurn"] = self._last_turn(participant_id)
        return payload

    def _last_turn(self, participant_id: str) -> str | None:
        """§5.2's `lastTurn`: `"failed"` while the latch is set, `None`
        otherwise. Read-only; see `_mark_turn_failed`/`_clear_turn_failed`.
        """
        with self._last_turn_failed_lock:
            return "failed" if participant_id in self._last_turn_failed else None

    def _mark_turn_failed(self, participant_id: str) -> None:
        """Set the dead-turn latch. Called only from `_run_turn`'s own
        failure-isolation block, on the same turn whose exception it logs —
        never from the request thread.
        """
        with self._last_turn_failed_lock:
            self._last_turn_failed.add(participant_id)

    def _clear_turn_failed(self, participant_id: str) -> None:
        """Clear the dead-turn latch. `set.discard` — a no-op when it was
        already clear, which is the ordinary case (most turns do not fail).
        """
        with self._last_turn_failed_lock:
            self._last_turn_failed.discard(participant_id)

    def reserve_turn(self, participant_id: str) -> TurnBooking | None:
        """**The `409` check and the booking as one indivisible step** (§4.4
        measure 1a): the new booking, or `None` when that participant already
        has a turn in flight.

        `None` **is** the `409 TurnInProgress` — the route raises it, still
        before the message write. What this replaces is a check on the request
        thread followed by a booking with a FalkorDB round trip in between, so
        two posts from one participant could both pass the check and both book.
        The map holds one slot per participant, so the first worker's clear
        then erased the *second*, still-running turn's entry and
        `turn_in_flight` reported idle under a live turn — the state
        `_await_quiesce` exists to make impossible
        (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`, P17-1, reproduced).

        **Every path that then fails to reach a worker must `release_turn`** —
        `services.post_message` raising, or a refused `submit`. That is not
        tidiness. A leaked reservation `409`-locks that participant for the
        life of the process and turns every reset-mine of theirs into a `503`
        (P17-3); and it makes §5.3's `504 post_state_unknown` reconciliation
        decide **wrongly** rather than merely lose information, since
        `turn.state !== 'idle'` parks that client in *wait, as normal* forever
        for a turn nobody will run (`## Pass 18`, question 2).
        """
        with self._turns_lock:
            if self._turns.get(participant_id, IDLE_TURN).in_flight:
                return None
            booking = TurnBooking(ordinal=next(self._turn_ordinals))
            self._turns[participant_id] = TurnState(
                state=TURN_QUEUED, booking=booking
            )
            return booking

    def release_turn(self, participant_id: str, booking: TurnBooking) -> bool:
        """Clear that participant's slot **only if `booking` still owns it**;
        `True` when it did and the entry is gone.

        Two **roles**, one operation — and more call sites than roles: the
        request thread undoing a reservation that never reached a worker
        (a failed `services.post_message` in `storefront_api.py`, and
        `enqueue_turn`'s pre-submit shutdown check), and the worker clearing
        its own slot in `_run_turn`'s `finally` (which `set_turn_state`'s
        `TURN_IDLE` branch delegates to). The ownership
        condition is on **both**, not on the worker alone — between a
        reservation and a failed write, a `clear_all_turns()` plus a second-tab
        post can install a different booking in that slot, and an unconditional
        release would delete it. That is the very defect the token exists to
        prevent, on the one path a worker-only rule does not cover
        (`docs/reviews/salesperson-ui-impl.md` `## Pass 18`, P18-3).
        """
        with self._turns_lock:
            current = self._turns.get(participant_id)
            if current is None or current.booking != booking:
                return False
            del self._turns[participant_id]
            return True

    def set_turn_state(
        self, participant_id: str, state: str, *, booking: TurnBooking
    ) -> bool:
        """Move `booking`'s turn to `state`; `idle` clears the entry. `True`
        when the write took effect, `False` when that booking no longer owns
        the slot.

        **`booking` is required, and this write is conditional on it**, exactly
        as the release and the `finally` clear are. The reservation closes the
        admission window; the token closes the ownership one; neither subsumes
        the other (§5.1's S9 row). `clear_all_turns()` empties the map under
        workers that are still running, so an unconditional `thinking` flip
        would move a **later** booking's entry to `thinking` when the wiped
        turn's work item finally reached a worker.

        There is deliberately **no unconditional single-slot write left on this
        class**: an entry is created by `reserve_turn`, changed here and by
        `_attach_turn_future`, removed by `release_turn`, and dropped wholesale
        by the reset paths' `clear_all_turns()`.

        **The flip is a `replace`, not a fresh `TurnState`, so it carries the
        entry's `future` across** — this is the only state change an entry
        undergoes while it lives, and rebuilding the entry here would drop the
        handle on exactly the ordering where the worker started before
        `enqueue_turn` got to attach it. Keeping it costs nothing and makes the
        field mean one thing (*this booking's future, for as long as the entry
        lives*) instead of two. It does not make a running turn cancellable:
        `Future.cancel()` answers `False` once the work item is running, which
        is what sends reset-mine to `_await_quiesce`.
        """
        if state == TURN_IDLE:
            return self.release_turn(participant_id, booking)
        with self._turns_lock:
            current = self._turns.get(participant_id)
            if current is None or current.booking != booking:
                return False
            self._turns[participant_id] = replace(current, state=state)
            return True

    def _attach_turn_future(
        self, participant_id: str, booking: TurnBooking, future: Future[None]
    ) -> bool:
        """Hang `booking`'s submitted `Future` on its map entry; `True` when the
        write took effect (§5.1's S9 row — *one map, not two*).

        Ownership-checked like every other write on this class, and for the
        sharper of the two reasons: by the time `submit` has returned, the
        worker may already have run the whole turn and cleared the slot in
        `_run_turn`'s `finally`. An unconditional write here would **resurrect**
        that entry — a `queued`/`thinking` turn nobody will ever clear, which
        `409`-locks the participant for the life of the process and answers
        every reset-mine of theirs `503`. That is P17-3's shape through a door
        the release rule does not cover, since nothing was refused here.

        Losing the write is harmless in both directions it can be lost. The turn
        finished (no entry, or a later booking's) — nothing to cancel. Or
        `clear_all_turns()` wiped the slot and a fresh post replaced it — the
        orphaned turn is not this participant's current one and must not be
        reachable through their slot.
        """
        with self._turns_lock:
            current = self._turns.get(participant_id)
            if current is None or current.booking != booking:
                return False
            self._turns[participant_id] = replace(current, future=future)
            return True

    def clear_all_turns(self) -> None:
        """Drop every turn entry — the reset paths, after quiesce (S7/S10).

        **Also drops every dead-turn latch** (§5.1's S9 row): reset-everyone
        deletes every participant's transcript, so the notice a latch refers
        to no longer names anything that survives — a stale `lastTurn:
        "failed"` after a reset-everyone would point at a turn nobody can see.
        """
        with self._turns_lock:
            self._turns.clear()
        with self._last_turn_failed_lock:
            self._last_turn_failed.clear()

    def turn_in_flight(self, participant_id: str) -> bool:
        """Whether this participant already has a turn queued or running."""
        return self.turn_state(participant_id).in_flight

    # ── the turn queue (§4.4 measure 1) ─────────────────────────────────────

    def enqueue_turn(
        self,
        ctx: CallContext,
        participant: ParticipantRecord,
        posted: dict[str, Any],
        booking: TurnBooking,
    ) -> Future[None]:
        """Hand `booking`'s already-reserved turn to the executor.

        **This method submits; it does not run the turn.** The request thread
        does four things and then answers — `reserve_turn` (the `409`
        single-flight check and the booking as one atomic step, in the route,
        *before* the message write), `services.post_message`, this submit, and
        the response, **releasing the reservation only where nothing was
        queued — which is decided by a place in the sequence, before the
        submit, never inside its `except`** (spelled out below). Everything
        after the submit is `_run_turn`, on a worker.
        `docs/plans/salesperson-ui.md` §5.1's S9 row decides that placement
        rather than leaving it open, for three reasons that are not
        preferences:

        1. `services.start_workflow_run` is **synchronous and drives the whole
           run** — up to eight chat completions against a 180 s agent timeout.
           On the request thread that *is* the turn, and `POST
           /shop/api/messages` becomes the slowest route in the system, which is
           the exact outcome §4.4 measure 1 exists to prevent: measure 1 swaps
           *which scheduler* runs the turn, not whether it is scheduled.
        2. It is the platform's delivered posture on both existing transports —
           `background._safe_run_workflow` is failure-isolated and off-band, on
           `BackgroundTasks` in `api.py` and on a thread in `mcp.py`. The
           storefront replaces `BackgroundTasks` with this bounded executor and
           inherits the rest.
        3. `participant` is **handed in** precisely so the worker never resolves
           a `ParticipantRecord` of its own: the request thread has already
           re-read it from the graph in `resolve_token`, and it is where
           `run_ctx`'s `language` comes from (§4.5).

        **No queue position is written here, or anywhere.** The entry already
        exists — `reserve_turn` created it as `queued`, before the message
        write, which is what makes a turn impossible to run while the map says
        idle (the ordering `_await_quiesce` depends on, §4.8). Its place in the
        line is §5.2's, derived on every read by `turn_payload`; `_run_turn`
        flips the entry to `thinking` when a worker picks it up, and a
        `thinking` turn is in no line at all.

        **The turn lock is not held across `executor.submit(...)`, and that is
        a rule rather than an accident** (§5.1's S9 row, which says *do not
        delete it on finding a mechanism that does not hold — that check has
        been run*). It rests on there being nothing to buy: §5.2 defines the
        line by **booking** order, so forcing submit order to match it buys an
        ordering nobody reads, at the price of an application lock underneath
        two `concurrent.futures` internals. It is deliberately **not** a
        deadlock claim — `_python_exit` joins every worker *outside*
        `_global_shutdown_lock` (`/usr/lib/python3.12/concurrent/futures/
        thread.py:23-31`, pinned 3.12.3), so no cycle can form
        (`docs/reviews/salesperson-ui-impl.md` `## Pass 18`, P18-1). The
        exit-time fact that *is* true is a cost, not a hang: those joins mean a
        worker blocked on the turn lock delays process exit for as long as the
        lock is held, which argues for holding it briefly. Beneath that, an
        application lock held across two `concurrent.futures` internals is a
        lock-ordering hazard whose present benignity is an implementation
        detail rather than a contract. Two turns booked microseconds apart may
        therefore reach workers in the other order, which §5.2's definition
        accommodates by construction. Reversal trigger, narrow — because only
        one of the three legs is contingent: only if §5.2 stops defining the
        line by booking order, and the answer then is a change to how work
        reaches the pool, not this lock. **Nothing else reopens it**: the
        exit-cost and lock-ordering legs do not depend on §5.2 at all.

        **Where a refusal releases the booking is a place in the sequence, not
        a property of the exception.** This method reads `_turns_shutdown`
        **before** it calls `submit`; on a set flag it releases the booking —
        ownership-checked like every other map write, so it cannot delete a
        booking that replaced this one in the meantime — and raises
        `RuntimeError`, having submitted nothing. That closes P17-3's shape:
        a booking left behind by a post that raced `shutdown_turns()` would
        `409`-refuse that participant for the life of the process and answer
        every reset-mine of theirs `503 quiesce_timeout`
        (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`, P17-3). **Past
        that check the `except` around `submit` logs at `ERROR` — naming the
        participant and the booking's ordinal — and re-raises, releasing
        nothing.**

        **Why, read out of `submit` itself** (CPython 3.12.3,
        `/usr/lib/python3.12/concurrent/futures/thread.py`, read in the change
        that wrote this): `submit` (`:164-180`) does `self._work_queue.put(w)`
        at `:178` and only *then* `self._adjust_thread_count()` at `:179`,
        whose `t.start()` (`:202`) is where `RuntimeError("can't start new
        thread")` originates; `_worker`'s loop (`:69-95`) pulls from that same
        shared queue and never checks who put an item on it. So an exhaustion
        refusal raises with the work item **already queued**, and any worker —
        one that exists now, or one a later `submit` starts — will run it.
        Releasing the booking there manufactures a live turn that
        `turn_in_flight` reports as `False`: §4.4 measure 1a's corrupted
        invariant, the one `_await_quiesce` exists to make impossible, through
        a third door (`## Pass 20`, P20-1). Reading the flag *inside* the
        `except` would get the direction right and still be strictly weaker —
        a `shutdown_turns()` landing between the flag write and the executor
        actually stopping leaves a submit that queues its item, fails, sees a
        set flag and releases.

        **What that accepts, derived from `submit` rather than listed.**
        Everything before the queue put is three guarded raises —
        `BrokenThreadPool` (`:167`), the executor's own `_shutdown` (`:170`),
        the interpreter's global `_shutdown` (`:172-173`) — plus the two
        object constructions at `:175-176`, which fail only on `MemoryError`.
        Those are the refusals that genuinely queue nothing, and wherever the
        flag was not already set this design **leaks** a booking on them
        instead of releasing it. That is the intended trade: a leaked booking
        is P17-3 — minor, one participant, self-limited — while an orphaned
        live turn breaks measure 1a's invariant for everyone `_await_quiesce`
        serves. The residue is smaller than P17-3's own statement of it:
        `_broken` is written in exactly one place, `_initializer_failed`
        (`:206-208`), reached only from `_worker` (`:77`) when an `initializer`
        raises, and this executor is constructed with `max_workers` and
        `thread_name_prefix` only (see `__init__`), so `BrokenThreadPool` is
        unreachable here. *(Inference rather than observation, and marked as
        such: reading `:172-173` as "the interpreter is exiting" rests on
        `_python_exit` being the writer that matters, not on an enumeration of
        every writer of that module global. It changes the size of the
        residue, never the direction of the trade.)*

        Discriminating on the exception type is not an option in any case:
        both shapes are a bare `RuntimeError` (`:170` and `:202`), and the only
        thing that differs is CPython's message string.

        *(Tombstone, 2026-09-08 — plan v1.30. Through v1.29 this paragraph said
        a refused `submit` releases the reservation on **any** exception, and
        named thread exhaustion as one of the two shapes it repaired. The rule
        was and is right — a reservation that reaches no worker must not
        survive — and the mechanism printed beside it was false on that second
        shape, which queues before it fails. The mechanism was replaced rather
        than the rule weakened: the release moved ahead of `submit`, where
        "nothing was queued" is knowable. Do not restore the unconditional form
        on rediscovering that a leaked booking is bad — it is, and it is the
        lesser of the two.)*

        **The `Future` is retained on the booking's map entry before it is
        returned** (S9b), which is what gives reset-mine's cancellation
        something to cancel — P17-9's gap, closed on the side the plan chose:
        the handle hangs off the booking token, one map and not two
        (§5.1's S9 row; `TurnState.future` for why). The attach is
        ownership-checked and may legitimately find nothing, which is
        `_attach_turn_future`'s own paragraph.

        Returns the `Future` so a caller can wait on the turn. Nothing in the
        request path does — the response is sent without it, which is the
        point — and the map **entry** is still what the `409` gate and both
        quiesce drains read: reset-mine reaches the future *through* the entry,
        never instead of it.
        """
        if self._turns_shutdown:
            self.release_turn(participant.participant_id, booking)
            raise RuntimeError(
                "cannot schedule new turns after shutdown_turns()"
            )
        # **Cleared here — after the shutdown check, before `submit` — and
        # only here** (§5.1's S9 row, carried forward from
        # `docs/reviews/salesperson-ui-impl.md` `## Pass 20`). `reserve_turn`
        # is too early: a booking whose `services.post_message` then raises
        # never reaches this method at all, and clearing at reservation would
        # drop the prior failure's notice from under a post that itself
        # failed to queue anything. The pre-submit `_turns_shutdown` branch
        # above raises before this line, so a refusal there leaves the latch
        # untouched too.
        #
        # **Strictly before `submit`, never after it returns — measured, not
        # assumed.** This clear and this turn's own possible failure share one
        # participant's latch, and `submit` starts the worker asynchronously:
        # a clear placed after a successful `submit` races the very turn it
        # just queued, and the worker can win. Reproduced in this revision —
        # of 200 single-turn runs with the clear placed after `submit`, 8
        # ended with `lastTurn: None` on a turn that had in fact just failed,
        # because `_run_turn` reached its `except` and called
        # `_mark_turn_failed` on its worker thread before the request thread
        # came back from `submit` to call this line. Placed before `submit`,
        # the clear always happens-before anything the worker for *this*
        # booking can do, so it can no longer race that worker's own mark.
        #
        # **The residual, named rather than hidden**: on the rare `submit`
        # refusal that queues nothing at all (the three guarded raises this
        # method's own docstring derives from `thread.py:164-180` —
        # `BrokenThreadPool`, either `_shutdown`, or a `MemoryError` on the two
        # allocations) the latch has already been cleared for a turn that
        # never ran. Moving the clear after `submit` to close this would
        # reopen the race measured above on the ordinary path, which is the
        # trade this file already makes elsewhere for the same three raises
        # (`release_turn`'s own booking-leak residual, below): the two shapes
        # cannot both be avoided.
        self._clear_turn_failed(participant.participant_id)
        try:
            future = self._executor.submit(
                self._run_turn, ctx, participant, posted, booking
            )
        except BaseException:
            _log.exception(
                "storefront turn submit refused, booking left standing "
                "(participantId=%s, ordinal=%s)",
                participant.participant_id, booking.ordinal,
            )
            raise
        self._attach_turn_future(participant.participant_id, booking, future)
        return future

    def _run_turn(
        self,
        ctx: CallContext,
        participant: ParticipantRecord,
        posted: dict[str, Any],
        booking: TurnBooking,
    ) -> None:
        """One agent turn, on a turn-executor worker.

        **Failure-isolated, exactly like `background._safe_run_workflow`**: an
        LLM outage, the 180 s agent timeout, a `WorkflowEngineDisabledError`
        from an unwired executor — all of it is logged here and none of it
        propagates, because there is no longer a response to propagate into.
        The `200` was sent on the request thread.

        The workflow layer is reached **through the trigger**, never through
        `self._services`: `trigger.maybe_trigger` applies the §6 ordered rule
        (resume a waiting run / start one / fall through), so the storefront
        does not re-implement a policy the platform owns and its own service
        surface stays exactly what S8 measured.

        `run_ctx={"language": …}` is §4.5's carrier — the participant's chosen
        language rides in the run ctx, which `executor._assemble_messages`
        replays as the `CONTEXT:` block on every LLM iteration, so it survives
        the whole conversation rather than only its first turn.

        The `finally` releases the map entry **when this booking still owns
        it**, which is what re-opens the composer and releases the `409` gate;
        the paragraph below is why that condition is there. **A turn that dies
        here sets the dead-turn latch first** — `_mark_turn_failed`, in this
        same `except`, the one place §5.2's `lastTurn` is written — so it stays
        distinguishable from one that completed even after the `finally` below
        deletes the entry the exception happened on (§5.1's S9 row). A
        `self._trigger is None` no-op turn is not a failure and does not touch
        the latch: nothing ran, so nothing died.

        **Both map writes name this booking and take effect only while it still
        owns the slot** — the `thinking` flip as much as the `finally`.
        `clear_all_turns()` empties the map under workers that are still
        running, so an unconditional clear would delete a *later* booking's
        entry and an unconditional flip would move it back to `thinking`
        (`docs/reviews/salesperson-ui-impl.md` `## Pass 17`, P17-1;
        `## Pass 18`, question 2). Losing the slot does not abandon the work:
        the turn was accepted, so it runs to completion and only its
        bookkeeping is skipped.
        """
        participant_id = participant.participant_id
        try:
            self.set_turn_state(participant_id, TURN_THINKING, booking=booking)
            if self._trigger is None:
                return
            self._trigger.maybe_trigger(
                ctx,
                thread_id=posted["threadId"],
                msg_id=posted["msgId"],
                text=posted["text"],
                role=posted["role"],
                mentions=posted.get("mentions", []),
                run_ctx={"language": participant.language},
            )
        except Exception:  # noqa: BLE001 — turn isolation: log, never propagate
            _log.exception(
                "storefront turn failed (participantId=%s, msgId=%s)",
                participant_id, posted.get("msgId"),
            )
            self._mark_turn_failed(participant_id)
        finally:
            self.release_turn(participant_id, booking)

    def shutdown_turns(self) -> None:
        """Stop accepting turns and **drain** the ones already accepted.

        Called from `create_app`'s lifespan after `yield`. `wait=True` with no
        `cancel_futures` is the whole contract: a queued turn has a message
        written for it in the transcript, so dropping it **here** is the
        "message with no reply" §4.4 measure 1a refuses to create, one layer
        down. Idempotent — a second call on an already-shut-down executor
        returns immediately.

        **The rule is about this path, not about cancelling as such**, and S9b
        is why the distinction has to be written down. `_cancel_queued_turn`
        drops a queued turn on purpose, and it does not contradict the sentence
        above: it runs only inside reset-mine, where the participant has asked
        for that message and its whole transcript to be deleted, so the reply
        that will never be written has nothing left to be missing from. A
        shutdown asked for no such thing — every participant's transcript
        survives it — which is why the same act is right there and wrong here.

        **`_turns_shutdown` is set before the executor is told to stop, and
        that order is the contract.** `enqueue_turn` reads the flag *before*
        it calls `submit`, so a post racing this call is refused without ever
        entering `submit` rather than from inside it — which is what keeps the
        refusal on the side where nothing was queued (§5.1's S9 row, and
        `enqueue_turn`'s own docstring for why the distinction is load-bearing).
        """
        self._turns_shutdown = True
        self._executor.shutdown(wait=True)

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
        * `turn` — this participant's entry in the in-process turn map,
          with §5.2's `queuePosition` **derived here** from the rest of the map
          rather than read off the entry, and `lastTurn: 'failed' | null` — the
          dead-turn latch, composed *alongside* the entry rather than read off
          it, since it must survive the very `set_turn_state(idle)` call that
          deletes that entry (`turn_payload`).

        `ctx.actor` is the participant id and also their `customerId`, so all
        three graph reads are scoped structurally rather than by a filter
        anyone could forget (§4.3).
        """
        return {
            "profile": self._services.get_profile(ctx),
            "cart": self._services.get_cart(ctx),
            "order": self._services.get_current_order(ctx),
            "turn": self.turn_payload(ctx.actor),
        }

    # ── catalog + the product-image manifest (§4.7) ─────────────────────────

    @property
    def _catalog_ctx(self) -> CallContext:
        """The `CallContext` the catalog reads take.

        The catalog is **global `reference` data** — `repository.filter_products`
        takes no `ws` and no customer at all, and `services` accepts a `ctx` for
        interface parity without reading either field. Naming the demo `Agent`
        as the actor keeps that honest: no participant identity is invented for
        a read that has nothing to do with one, and no route can leak one
        participant's scope into another's catalog.
        """
        return CallContext(ws=self._ws, actor=self._agent_id)

    def _catalog_rows(self) -> list[dict[str, Any]]:
        """The whole catalog as `{productId, name, category, price}` rows —
        **one read**.

        `services.filter_products` (S2/S4, `docs/QUERIES.md` §15.2) projects
        exactly the row §5.2's `GET /shop/api/catalog` contract and §4.7's
        image manifest are keyed on, so this method is the pass-through it
        looks like.

        **S7 shipped a `1 + n` here and S7c removed it.** At S7 the delivered
        projection was `{name, category, price}` with no `productId`, so each
        row's id came from a second, index-anchored `services.lookup_product`
        point read, and a row whose name no longer resolved was silently
        dropped — the catalog route's only unbounded failure path. S7c widened
        `repository.filter_products`'s projection instead (the identical
        additive change K-053 made to `lookup_product`), which deletes both the
        second read and the silent drop. The two halves are bound by one test
        — `test_the_catalog_is_read_once_not_once_per_product`, which patches
        `services.lookup_product` to raise — so neither can be reverted alone
        (`docs/plans/salesperson-ui.md` §5.1 S7c).

        `limit=CATALOG_LIMIT` is the **explicit** bound: the delivered default is
        `20`, right for 15 products and silently truncating at 21.
        """
        return self._services.filter_products(
            self._catalog_ctx, category=None, min_price=None, max_price=None,
            limit=CATALOG_LIMIT,
        )

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

    def _cancel_queued_turn(self, participant_id: str) -> bool:
        """Drop this participant's turn **if it is still sitting in the
        executor's queue**; `True` only when a turn was actually cancelled *and*
        its map entry cleared (§4.8, S9b).

        **This runs in front of `_await_quiesce`, never in place of it**
        (§5.1's S9 row; `docs/reviews/salesperson-ui-impl.md` `## Pass 7`,
        Ruling 3). It buys **availability, not correctness**: the wait already
        makes the result right, because a queued turn reaches a worker,
        completes and clears its own entry. What the wait cannot do is finish
        inside `quiesce_s` — 30 s against a 180 s agent timeout — so a slow
        turn turns reset-mine into a `503` that resets nothing, exactly where
        dropping work the LLM has not started yet would have let it succeed.

        **The order is cancel-then-clear, and it is the whole design.**
        `Future.cancel()` answers `False` once the work item is running, so a
        `True` from it is the one moment at which the turn is provably never
        going to run — and only then may its slot go. Clearing first would
        report idle while the job was still queued and let the delete race the
        very turn quiesce exists to prevent (§4.8; the `_await_quiesce`
        docstring). Losing that race is ordinary and needs no repair: a turn
        that started between the read below and the `cancel()` keeps its entry
        and falls through to the wait.

        **What falls through to the wait, stated because the `503` contract is
        exactly its complement.** A turn already on a worker (`cancel()` →
        `False`) does not reach cancellation at all. A turn *reserved but not
        yet submitted* — the participant's own second tab still inside
        `services.post_message` — has no future yet (`TurnState.future`) and so
        is not reachable either. The third case is different in kind, not just
        in name: a booking that lost its slot to `clear_all_turns()` plus a
        fresh post *is* cancelled — its future really is `PENDING` and
        `cancel()` really answers `True` — but `release_turn`'s ownership check
        then declines to delete the **live** entry that replaced it, so the
        clear never reaches the map. All three leave the participant's current
        turn queued, which is what sends them to the wait below.

        **Nothing is rolled back if the reset then fails.** A cancelled turn
        stays cancelled through a `504`, and the participant is left with a
        message and no reply. That is the same price §4.8 already accepts for
        the turn whose transcript the reset deletes, and it is not new
        exposure: reset-mine is the participant asking for that transcript to
        go.

        The lock is dropped before `cancel()` rather than held across it, for
        the reason `enqueue_turn` gives for `submit`: an application lock held
        underneath a `concurrent.futures` internal is a lock-ordering hazard
        whose benignity is an implementation detail. Every write that follows
        is ownership-checked, so the window buys nothing but a `False`.
        """
        with self._turns_lock:
            current = self._turns.get(participant_id)
            if current is None or current.future is None:
                return False
            future, booking = current.future, current.booking
        if not future.cancel():
            return False
        return self.release_turn(participant_id, booking)

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
        shorten the wait. **`_cancel_queued_turn` now does that, in front of
        this wait and never in place of it** (S9b) — read that method for what
        it covers. The wait is **not weakened by it and not conditional on it**:
        everything the cancel does not reach still arrives here, and a queued
        turn that reaches a worker completes and clears its own entry, so
        waiting subsumes cancelling for correctness and differs only in latency.
        Dropping the turn-map entry as a stand-in would be actively wrong: it
        would report idle while the job was still queued, and the delete would
        then race exactly the turn this waits for. So the cancel clears the
        entry only *after* `Future.cancel()` has answered `True`, since a future
        already running cannot be cancelled and must fall through to this wait
        (§4.8; `docs/reviews/salesperson-ui-impl.md` `## Pass 7`, Ruling 3).

        **What this wait now has to wait for.** Until the turn executor landed,
        nothing populated the turn map, so every drain passed on its first
        check; a turn is real work on a worker now, and this deadline is what
        bounds it. With the cancel in front, what remains for this deadline is
        the work that was **not cancellable at the instant reset-mine asked** —
        a turn already on a worker, and a turn reserved but not yet submitted —
        which is also the exact reach of the `503`.
        """
        deadline = time.monotonic() + self._quiesce_s
        while self.turn_in_flight(participant_id):
            if time.monotonic() >= deadline:
                return False
            time.sleep(QUIESCE_POLL_S)
        return True

    def reset_participant(self, participant: ParticipantRecord) -> dict[str, Any]:
        """"Reset mine" — cancel what is only queued, quiesce the rest, then one
        atomic delete (§4.8, graph note §4).

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
        answer this record already carries.

        Returns `{"threadId": …, "language": …}` — §5.2's `200` body. Raises,
        for each of the four ways this can end other than success:

        * `QuiesceTimeoutError` → `503`, **nothing in the graph changed**.
        * `ResetStateUnknownError` → `504`, *unknown* — see F8 below.
        * `UnknownParticipantError` → `404`/`401`, zero rows.
        * `UnscopedParticipantError` → `409`, a guaranteed no-op.

        **The `503` is "nothing was reset", and since S9b that is stated of the
        graph rather than of the process.** `_cancel_queued_turn` runs first, so
        on the ordinary path a `503` now means the turn was **not cancellable**
        — already on a worker, or reserved by a second tab still inside its
        message write — and that path cancels nothing. The one arrangement
        where a `503` follows a successful cancel is the orphan: a booking
        stranded by `clear_all_turns()` is cancelled, its slot is *not* cleared
        (it belongs to the fresh post that replaced it), and the wait then times
        out on that live turn. The graph is still untouched, which is what the
        contract and §5.3 C9's "retry is safe" rest on; what the participant
        loses is a reply to a message that — no reset having happened — stays in
        their transcript.

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
        # **In front of the wait, never in place of it** (§4.8, S9b). Its return
        # value is deliberately unread: a cancel that succeeded leaves the slot
        # empty and the wait below passes on its first check, and one that did
        # not is precisely what the wait is for. Branching on it would give this
        # path two shapes where it has one.
        self._cancel_queued_turn(participant_id)
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
            raise UnknownParticipantError(
                f"{participant_id!r} is not a participant of ws:{self._ws}"
            )
        if not status["scoped"]:
            raise UnscopedParticipantError(
                f"{participant_id!r} has no owned channel — nothing was reset"
            )

        # **Cleared here, on the success path only** (§5.1's S9 row) — the
        # delete above just committed, and the dead-turn latch refers to a
        # transcript that is now gone. Every earlier `raise` in this method
        # (quiesce timeout, F8's `504`, unknown/unscoped participant) leaves
        # this line unreached, so a `503`/`504` that reset nothing does not
        # erase the notice either.
        self._clear_turn_failed(participant_id)
        self._services.save_profile(ctx, name=participant.display_name)
        return {"threadId": status["threadId"], "language": participant.language}

    def _reset_state_unknown(
        self, ctx: CallContext, participant_id: str
    ) -> ResetStateUnknownError:
        """Build F8's `504` after a reset timed out on the way to FalkorDB.

        Re-reads state so the response can report what the graph actually
        holds — the delete may have committed. A second `TimeoutError` from
        that re-read, **or a `RuntimeError` out of `get_state`**, is swallowed
        into `state=None`: still a `504`, never a `500`, and never "nothing
        changed".

        **This is `get_state`'s other call site**, not just
        `GET /shop/api/state`'s handler body (`storefront_api.py:1103`) — a
        gap `## Pass 22` (P22-4) found the review itself had missed too. The
        two exceptions caught below are exactly the ones this contract has to
        survive: `TimeoutError` is F8's own premise (a stalled reset stalls
        the re-read the same way), and `RuntimeError` is the one class this
        module's raises-guard (`tests/test_storefront_api.py`'s
        `_raise_sites` assertion) polices as a defensive reflex elsewhere in
        it — catching it here means a future `raise RuntimeError` reached
        through `get_state` still answers F8's `504` rather than turning it
        into a bare `500`. Nothing wider: any other exception out of
        `get_state` is a bug this method has no reason to hide behind
        "unknown".
        """
        try:
            state: dict[str, Any] | None = self.get_state(ctx)
        except (redis_exceptions.TimeoutError, RuntimeError):
            state = None
        return ResetStateUnknownError(participant_id, state=state)
