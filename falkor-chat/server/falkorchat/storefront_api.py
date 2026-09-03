"""The storefront's HTTP door — the `/shop/api` router and its error map (S8).

`docs/plans/salesperson-ui.md` §5.2 (the eleven-route surface), §5.3 (the
route-class table and the `(route, response)` completeness table it feeds),
§4.7 (the product-image manifest), §4.9 (the deployment with no unauthenticated
read path). The storefront's *core* — the participant registry, join, token
verification, state, reset, catalog and the order gate — is `storefront.py`;
this module only turns it into HTTP.

The one thing this module exists to make decidable
--------------------------------------------------
**The error map is total by type**, so the set of responses the server can
produce is bounded by construction rather than by anybody having enumerated it
(§5.3's C13 residual paragraph). Two machine-readable constants carry that:

* `ROUTE_CLASSES` — §5.3's route-class table, keyed on `(METHOD, path)` because
  `/shop/api/messages` is a `reads-only` route under `GET` and a `writes` route
  under `POST`. It is the **input** to the gate, not a nicety: without it the
  `{handlers} × {routes}` cross product is nonsense on the two routes that
  issue no query at all.
* `CROSS_CUTTING_HANDLERS` / `ENVELOPE_HANDLERS` — every exception handler this
  module registers, split by whether it *produces* one of §5.3's three
  cross-cutting responses or merely re-shapes a response a route already
  declares. `tests/test_storefront_api.py` asserts that the handlers actually
  registered on the app are exactly these two sets, so a handler added without
  a classification reddens rather than passing unnoticed.

`cross_cutting_response()` is the **single seam** both the live handler and the
gate read, so the gate cannot agree with a handler that has drifted from it —
and the per-route contract tests then assert the same answers by *execution*,
which is what keeps the seam itself honest.

What this module deliberately does not hold
-------------------------------------------
No Cypher (`falkor-chat/AGENTS.md` rule 1) and no participant resolution of its
own: authentication is `Storefront.resolve_token`, which re-reads the graph on
every call. In particular this module **never calls `.lookup(`** — the
read-through cache returns the identical `ParticipantRecord`, so a router
authenticating through it would be indistinguishable from one authenticating
against the graph (`docs/reviews/salesperson-ui-impl.md` `## Pass 6`). S9
removes that cache outright, at which point the rule holds structurally.

Steps that extend this module
-----------------------------
* **S9** — `POST /shop/api/messages` gains the turn enqueue behind its write.
* **S10** — the three presenter operations move onto `Storefront` itself
  (`presenter_login`, `list_participants`, `reset_all`), together with the
  login rate-limiter and reset-everyone's stop-intake-then-drain quiesce. They
  are implemented here, against the delivered core, only because S8's gate is
  evaluated over **all eleven** routes and cannot be run on a partial surface;
  see `_STEP_10_INTERIM` below for every line S10 takes with it.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import threading
import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from redis import exceptions as redis_exceptions

from . import config
from .db import FalkorDBUnreachableError
from .storefront import (
    QUIESCE_POLL_S,
    OrderTransitionRefusedError,
    ParticipantRecord,
    QuiesceTimeoutError,
    ResetStateUnknownError,
    Storefront,
    UnknownOrderError,
    UnknownParticipantError,
    UnscopedParticipantError,
    parse_bearer,
)

_log = logging.getLogger(__name__)

# Where `create_app` mounts this router, and where it mounts the SPA build.
# `ROUTE_CLASSES` below spells the *full* paths, so a change to either that is
# not made here fails the route-table assertion rather than silently
# re-classifying every route (`_route_paths` is prefix-blind unless threaded —
# `docs/reviews/salesperson-ui-impl.md` `## Pass 5`).
API_PREFIX = "/shop/api"
SHOP_MOUNT = "/shop"

# The presenter credential's id half: `Authorization: Bearer presenter.<token>`
# (§5.3's credentials table). A participant id is `p-<uuid4hex>`, which can
# never collide with it.
PRESENTER_PRINCIPAL = "presenter"

# ── §5.3's route classes ─────────────────────────────────────────────────────
#
# "Which of the three cross-cutting responses a given route can produce is
# decided by its class." These three strings are that table.
WRITES = "writes"
READS_ONLY = "reads-only"
NO_GRAPH = "no graph access"

# `(METHOD, path)` -> `(class, op token)`. The op token is the `<op>` in
# §5.3's `504 <op>_state_unknown` and is `None` for every non-writing route.
#
# **Keyed on the method as well as the path, and that is load-bearing:**
# `/shop/api/messages` is `reads-only` under `GET` (its read passes an explicit
# `since`, so it never advances a cursor and therefore never writes) and
# `writes` under `POST`. A path-keyed table would give one of the two the other
# one's cross-cutting row.
ROUTE_CLASSES: dict[tuple[str, str], tuple[str, str | None]] = {
    ("GET", f"{API_PREFIX}/health"): (NO_GRAPH, None),
    ("POST", f"{API_PREFIX}/session"): (WRITES, "join"),
    ("GET", f"{API_PREFIX}/state"): (READS_ONLY, None),
    ("GET", f"{API_PREFIX}/messages"): (READS_ONLY, None),
    ("POST", f"{API_PREFIX}/messages"): (WRITES, "post"),
    ("GET", f"{API_PREFIX}/catalog"): (READS_ONLY, None),
    ("POST", f"{API_PREFIX}/order/advance"): (WRITES, "order"),
    ("POST", f"{API_PREFIX}/reset"): (WRITES, "reset"),
    ("POST", f"{API_PREFIX}/presenter/session"): (NO_GRAPH, None),
    ("GET", f"{API_PREFIX}/presenter/participants"): (READS_ONLY, None),
    ("POST", f"{API_PREFIX}/presenter/reset-all"): (WRITES, "reset"),
}

# The two handler tokens `cross_cutting_response` dispatches on. Not the
# response tokens — `graph_timeout` becomes `graph_read_timeout` or
# `<op>_state_unknown` depending on the route's class, which is the whole point.
_GRAPH_UNAVAILABLE = "graph_unavailable"
_GRAPH_TIMEOUT = "graph_timeout"

# Every exception type this module maps to one of §5.3's **three cross-cutting
# responses**, with the handler token that decides which. The gate crosses this
# with `ROUTE_CLASSES` and asserts the result against §5.3's completeness table.
CROSS_CUTTING_HANDLERS: dict[type[BaseException], str] = {
    # Nothing was sent — the precedent is `api.py:63` / `app.py:345`.
    FalkorDBUnreachableError: _GRAPH_UNAVAILABLE,
    redis_exceptions.ConnectionError: _GRAPH_UNAVAILABLE,
    # A *query-time* timeout: `reads-only` ⇒ nothing changed; `writes` ⇒ the
    # write may have committed and the participant-facing meaning is *unknown*
    # (§4.8 F8).
    redis_exceptions.TimeoutError: _GRAPH_TIMEOUT,
}


def cross_cutting_response(
    handler_token: str, method: str, path: str
) -> tuple[int, str] | None:
    """`(status, error token)` for one cross-cutting handler on one route, or
    `None` when that route's class permits none of the three.

    **The single seam the live handler and the gate both read.** Written as one
    function rather than as a table in the tests and a branch in the handler,
    because two spellings of one rule is exactly the shape §5.3 keeps finding
    defects in — and because a gate that agrees with a handler it does not
    share code with is a gate that cannot see the handler drift. What the gate
    therefore cannot check is this function itself, which is why the per-route
    contract tests assert the same answers by *execution* against a stubbed
    repository on every one of the nine graph-touching routes.
    """
    klass, op = ROUTE_CLASSES[(method, path)]
    if klass == NO_GRAPH:
        # §5.3: no query is issued, so no typed handler can fire and the route
        # takes none of the three rows. A handler firing here is a defect.
        return None
    if handler_token == _GRAPH_UNAVAILABLE:
        return 503, _GRAPH_UNAVAILABLE
    if klass == READS_ONLY:
        return 503, "graph_read_timeout"
    return 504, f"{op}_state_unknown"


# ── size bounds (§5.2, mirroring `schemas.py`'s rule-6 posture) ──────────────
MAX_DISPLAY_NAME_LEN = 60
MAX_LANGUAGE_LEN = 32
MAX_MESSAGE_TEXT_LEN = 2000
MAX_MESSAGE_PAGE = 200
DEFAULT_MESSAGE_PAGE = 50
MAX_PRESENTER_KEY_LEN = 200

# The join response's `welcome` line (§5.2). Server-side because the join
# response is minted before the SPA knows the participant's language is
# accepted; S12c/S13 own every other string the participant reads.
WELCOME: dict[str, str] = {
    "en": "Welcome to the store, {name}.",
    "pt-BR": "Bem-vindo à loja, {name}.",
    "es": "Bienvenido a la tienda, {name}.",
}
WELCOME_FALLBACK = WELCOME["en"]


def _welcome(display_name: str, language: str) -> str:
    return WELCOME.get(language, WELCOME_FALLBACK).format(name=display_name)


def _nonblank(value: str) -> str:
    """Reject a whitespace-only string at the boundary.

    `Field(min_length=1)` accepts `"   "` (verified, `python-web-quirks`
    SKILL.md), and a blank display name or a blank presenter key is the most
    ordinary human mistake in both flows (§5.3 C11 files the blank key as
    *user-supplied*, to be shown next to the box). Raising here keeps it a
    `422 validation_failed` carrying that field's own name rather than a
    server-side surprise later.
    """
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class JoinIn(BaseModel):
    """`POST /shop/api/session` (§5.2).

    **Field order is part of the contract.** §5.3 C11 pins the `422` selection
    rule to the *first* entry of `RequestValidationError.errors()`, which for a
    single request model is declaration order — so a request violating both
    bounds reports `displayName`, deterministically, and the client's
    highlighting cannot depend on which field the user fixed last.

    `language` is bounded for length here and checked against the deployment's
    own `Storefront.locales` in the route, because the locale set is per-
    instance configuration (`FALKORCHAT_STOREFRONT_LOCALES`) and not a class
    constant. That check raises the same `RequestValidationError` with the same
    `loc`, so both violations are one response shape and one code path.
    """

    displayName: str = Field(min_length=1, max_length=MAX_DISPLAY_NAME_LEN)
    language: str = Field(min_length=1, max_length=MAX_LANGUAGE_LEN)

    _check_display_name = field_validator("displayName")(_nonblank)


class PostMessageIn(BaseModel):
    """`POST /shop/api/messages` (§5.2) — `text ≤ 2000`."""

    text: str = Field(min_length=1, max_length=MAX_MESSAGE_TEXT_LEN)

    _check_text = field_validator("text")(_nonblank)


class AdvanceOrderIn(BaseModel):
    """`POST /shop/api/order/advance` (§5.2).

    No `orderId`: §5.2's invariant is that no route accepts `ws`, `threadId`,
    `customerId` or `orderId` from the client, and §4.6 takes the order id from
    server-side state — which is what makes `404`/`409` mean *stale*, never
    *someone else's* (§5.3 C10).
    """

    transition: Literal["fulfill", "deliver", "cancel"]


class PresenterSessionIn(BaseModel):
    """`POST /shop/api/presenter/session` (§5.2) — the key, exchanged for a token."""

    key: str = Field(min_length=1, max_length=MAX_PRESENTER_KEY_LEN)

    _check_key = field_validator("key")(_nonblank)


class StorefrontHTTPError(HTTPException):
    """A refusal a **route** produces, carrying the storefront's stable envelope.

    Deliberately a subclass of `HTTPException` rather than a re-registration of
    the base handler: Starlette resolves a handler by walking the exception's
    MRO and taking the first match, so this claims its own envelope without
    changing what a plain `HTTPException` does — which matters because the
    storefront deployment's bare `GET /health` raises one (`app.py`, §4.9
    move 1) and is not part of this contract.

    `error` is the machine-readable token §5.3's rules dispatch on (C6b reads
    the *body*, not the `409`); `extra` carries the per-response fields §5.2
    names — the order's current `status`, a reset's re-read `state`.
    """

    def __init__(
        self, status_code: int, error: str, detail: str, **extra: Any
    ) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error = error
        self.extra = extra


class StorefrontPreflightError(RuntimeError):
    """§4.9's readiness preflight refused to start the storefront.

    Raised from the lifespan **before** `yield`, so a mis-seeded deployment
    fails at boot naming the fix command, rather than coming up "green but
    dead" and 500-ing the first participant who types.
    """


# ── the error map, total by type (§5.3) ──────────────────────────────────────


async def _handle_storefront_http_error(_request, exc: StorefrontHTTPError):  # noqa: ANN001
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error, "detail": exc.detail, **exc.extra},
    )


async def _handle_validation_error(_request, exc: RequestValidationError):  # noqa: ANN001
    """§5.3 C11: the storefront's own stable `{error, field}`, never FastAPI's
    `loc` shape.

    The selection rule is pinned by the plan and is not an implementation
    detail: the **first** entry of `exc.errors()` — declaration order for a
    single request model — and `field` is the **last element** of its `loc`, so
    the client sees `displayName`, never `body.displayName`. A response
    therefore never carries two field names, and the client's own bounds keep
    the common case from round-tripping at all.
    """
    errors = exc.errors()
    field = ""
    if errors:
        loc = errors[0].get("loc") or ()
        if loc:
            field = str(loc[-1])
    return JSONResponse(
        status_code=422, content={"error": "validation_failed", "field": field}
    )


def _cross_cutting_json(request, handler_token: str) -> JSONResponse:  # noqa: ANN001
    answer = cross_cutting_response(handler_token, request.method, request.url.path)
    if answer is None:
        # Unreachable by construction: a `no graph access` route issues no
        # query, and the route-table assertion keeps `ROUTE_CLASSES` exactly the
        # registered route set. Reaching it means one of those two is false, so
        # say so loudly and answer the conservative code — *unknown*, never
        # "nothing changed".
        _log.error(
            "a typed graph handler fired on %s %s, which §5.3 classifies as "
            "issuing no query at all — this is a defect, not a row",
            request.method, request.url.path,
        )
        return JSONResponse(
            status_code=504,
            content={"error": "state_unknown", "detail": "unclassified route"},
        )
    status, token = answer
    return JSONResponse(status_code=status, content={"error": token})


async def _handle_graph_unavailable(request, _exc):  # noqa: ANN001
    return _cross_cutting_json(request, _GRAPH_UNAVAILABLE)


async def _handle_graph_timeout(request, _exc):  # noqa: ANN001
    return _cross_cutting_json(request, _GRAPH_TIMEOUT)


# Handlers that re-shape a response the route itself already declares, rather
# than producing one of §5.3's three cross-cutting rows. They contribute
# nothing to the `{handlers} × {routes}` cross product; what covers them is the
# declaration half plus the per-route contract tests, which prove each declared
# entry is actually producible.
ENVELOPE_HANDLERS: frozenset[type[BaseException]] = frozenset(
    {StorefrontHTTPError, RequestValidationError}
)


def register_storefront_error_handlers(app: FastAPI) -> None:
    """Register the storefront's error map on `app` (§5.1 S8).

    Typed throughout, in this codebase's own stated idiom — "without a blanket
    handler masking real bugs" (`app.py`). Registered on the **storefront**
    deployment only: the `503 graph_unavailable` half would be an improvement
    everywhere, but the `504 <op>_state_unknown` half is meaningless without
    `ROUTE_CLASSES`, and widening the map to the legacy surface is not S8's.
    """
    app.add_exception_handler(StorefrontHTTPError, _handle_storefront_http_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(FalkorDBUnreachableError, _handle_graph_unavailable)
    app.add_exception_handler(
        redis_exceptions.ConnectionError, _handle_graph_unavailable
    )
    app.add_exception_handler(redis_exceptions.TimeoutError, _handle_graph_timeout)


# ── the presenter session store (S10 interim — see the module docstring) ─────

_STEP_10_INTERIM = """
Everything below that S10 takes with it, listed once so the move is mechanical:

* `_PresenterSessions` and the `presenter/session` route body  -> `Storefront.presenter_login`
* `presenter/participants`' `_repo.list_participants` call -> `Storefront.list_participants`
* the `presenter/reset-all` route body                          -> `Storefront.reset_all`
* the three private reads in `build_storefront_router`          -> deleted with them

S8 implements them here because its gate is evaluated over **all eleven**
routes (`{declared} ∪ {handler-produced} == §5.3's table`, read off
`app.routes`) and cannot be run on a partial surface, and because §5.3's
negative assertion names `POST /shop/api/presenter/session` answering
`200`/`403`/`422` by name. What S8 deliberately does **not** build, because it
is S10's own content: the login's fixed per-attempt delay and observational
attempt counter, and reset-everyone's **stop-intake** flag (which is
`Storefront` state, and `storefront.py` is not S8's file). The drain below is
the exact parallel of S7 shipping reset-mine's wait without S9's cancellation.
"""


class _PresenterSessions:
    """The presenter tokens minted in this process.

    In-process by design and unrelated to the graph: the presenter is not a
    `User` (§4.3), so there is nothing durable to read. A restart invalidates
    them, which is correct — and `reset-all` deliberately does **not** touch
    them (§4.8: the presenter keeps driving the demo through the sweep).
    """

    def __init__(self) -> None:
        self._tokens: set[str] = set()
        self._lock = threading.Lock()

    def mint(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._tokens.add(token)
        return token

    def verify(self, token: str) -> bool:
        with self._lock:
            candidates = tuple(self._tokens)
        # `compare_digest` per candidate rather than a set membership test, so
        # a wrong token costs the same time whatever prefix it shares with a
        # live one — the same posture `resolve_token` takes.
        return any(hmac.compare_digest(known, token) for known in candidates)


# ── the router ───────────────────────────────────────────────────────────────


def build_storefront_router(shop: Storefront) -> APIRouter:
    """The eleven `/shop/api` routes (§5.2), to be included at `API_PREFIX`.

    Every route carries a `responses={…}` declaration naming **its own**
    returns — its status codes and their error tokens, and *not* the three
    cross-cutting ones, which come from the handler set. FastAPI keeps them on
    the route object, so the second half of S8's gate reads them back off
    `app.routes` instead of reading this file.
    """
    # One documented coupling instead of three scattered ones, the same posture
    # `Storefront.__init__` takes towards `Services._repo` and for the same
    # reason: `Storefront` exposes no accessor for these three, and S9/S10
    # delete every use of them (`_STEP_10_INTERIM`). Read once, here.
    repo = shop._repo  # noqa: SLF001 — see above
    services = shop._services  # noqa: SLF001 — the post/read/order service calls
    agent_id = shop._agent_id  # noqa: SLF001 — the mention every post carries
    presenter_key = shop._presenter_key  # noqa: SLF001 — moves to S10's login

    router = APIRouter()
    presenter_sessions = _PresenterSessions()

    # ── the two credential dependencies (§5.3's credentials table) ───────────

    def get_participant(
        authorization: str | None = Header(default=None),
    ) -> ParticipantRecord:
        """Resolve `Bearer <participantId>.<token>` against the **graph**.

        Never `.lookup(`: the read-through cache returns the identical record,
        so authenticating through it would be indistinguishable from
        authenticating against the registry — and a deleted participant would
        keep resolving out of stale memory until the process restarted.

        A *presenter* token presented here parses as participant id
        `"presenter"`, which no `User` carries, so it resolves to `None` and
        answers the same `401` as any other bad credential — the auth matrix's
        second row, held structurally rather than by a special case.
        """
        record = shop.resolve_token(authorization)
        if record is None:
            raise StorefrontHTTPError(
                401, "invalid_token", "no valid participant credential"
            )
        return record

    def get_presenter(authorization: str | None = Header(default=None)) -> None:
        """Verify `Bearer presenter.<presenterToken>`.

        §5.3 keeps two responses apart here, and the split is the contract:
        **`403` wrong credential type** when the request carried a participant
        token (or anything else that is not the presenter principal), and
        **`401` presenter session gone** when it carried no credential or a
        presenter token this process never minted. C2 clears the presenter
        credential on both; the client cannot act on the difference, but the
        auth matrix can.
        """
        parsed = parse_bearer(authorization)
        if parsed is None:
            raise StorefrontHTTPError(
                401, "presenter_session_gone", "no presenter credential"
            )
        principal, token = parsed
        if principal != PRESENTER_PRINCIPAL:
            raise StorefrontHTTPError(
                403, "wrong_credential_type", "not a presenter credential"
            )
        if not presenter_sessions.verify(token):
            raise StorefrontHTTPError(
                401, "presenter_session_gone", "presenter session not recognised"
            )

    # ── no graph access (§5.3) ──────────────────────────────────────────────

    @router.get(
        "/health",
        responses={
            200: {
                "description": "liveness + the locale list S12c's chooser reads",
                "x-storefront-tokens": ["ok"],
            }
        },
    )
    def health() -> dict[str, Any]:
        """§5.3: deliberately **does not touch the graph**, unlike the
        platform's `/health`.

        Its two consumers are a liveness probe and S12c's locale chooser, which
        must render on the join screen before anything else works; coupling it
        to the graph would hide the language list during an outage and buy
        nothing, because the storefront deployment already carries the
        graph-pinging liveness at the bare `GET /health` (§4.9 move 1).

        `storefrontEnabled` reports **this app's own wiring**, not
        `config.STOREFRONT_ENABLED`. Reading the module constant here would
        repeat exactly the defect §4.9 rules out for the route-table assertion:
        `config.py` resolves every flag at *import* time, so every test that
        builds the app through `create_app(storefront=True, dev_surface=False)`
        would be told the storefront is disabled by the storefront's own route.
        Reaching this route *is* the answer, and in production both derive from
        the one flag anyway (`_build_default_app`).
        """
        return {
            "status": "ok",
            "storefrontEnabled": True,
            "locales": list(shop.locales),
        }

    @router.post(
        "/presenter/session",
        responses={
            200: {
                "description": "the presenter credential",
                "x-storefront-tokens": ["ok"],
            },
            403: {
                "description": "wrong key — or no key configured at all",
                "x-storefront-tokens": ["bad_presenter_key"],
            },
            422: {
                "description": "missing or blank `key` (C11, user-supplied)",
                "x-storefront-tokens": ["validation_failed"],
            },
        },
    )
    def presenter_session(body: PresenterSessionIn) -> dict[str, Any]:
        """Key -> token, **entirely in-process** (§5.3's `no graph access`).

        `presenter_configured` is checked *before* any comparison, because
        `hmac.compare_digest("", "")` is `True` — an unconfigured deployment
        would otherwise hand the reset-everyone button to whoever posts an
        empty key first. A deployment with no key configured therefore answers
        the same `403` as a wrong key, deliberately: the two meanings differ,
        but telling the LAN that no key is configured is worse than telling it
        the key was wrong, and the client's action is identical (§5.3 C2). The
        operator's signal is this log line, not the response.
        """
        if not shop.presenter_configured:
            _log.warning(
                "presenter login refused: FALKORCHAT_STOREFRONT_PRESENTER_KEY "
                "is not set, so no key can ever authenticate"
            )
            raise StorefrontHTTPError(403, "bad_presenter_key", "presenter key rejected")
        if not hmac.compare_digest(presenter_key, body.key):
            _log.warning("presenter login refused: wrong key")
            raise StorefrontHTTPError(403, "bad_presenter_key", "presenter key rejected")
        return {"token": presenter_sessions.mint()}

    # ── the participant surface ─────────────────────────────────────────────

    @router.post(
        "/session",
        responses={
            200: {
                "description": "the participant credential, minted once",
                "x-storefront-tokens": ["ok"],
            },
            422: {
                "description": "`displayName` bounds (user-supplied) or "
                               "`language` not in `locales` (UI-supplied) — C11",
                "x-storefront-tokens": ["validation_failed"],
            },
        },
    )
    def join(body: JoinIn) -> dict[str, Any]:
        """Provision a participant and mint their credential (§4.3, §4.10).

        **Not idempotent**, decided rather than engineered away (§5.3 C4, R12):
        a lost response leaves a `User` with a `tokenHash` nobody holds. The
        alternative — a client-supplied idempotency nonce — would reopen
        delivered S6 to close a window that needs a FalkorDB socket timeout
        during the one write a participant makes before they hold any state.

        **`DemoNotSeededError` is deliberately not caught, and it is not a
        hole.** `storefront.py` documents it as a `503`, but §5.3's completeness
        table has no such row for this route — and after `storefront_preflight`
        has run, the condition it reports is structurally unreachable: the
        preflight asks the identical `resolve_member_kinds` question at boot,
        and the demo `Agent` is a **positive survivor of both resets**
        (`repository.reset_all_participants`), so nothing the storefront does
        can take it away afterwards. If it ever escaped it would propagate as a
        `5xx`, which is §5.2's own stance on an unmapped graph error. Recorded
        as a plan gap rather than papered over: adding a row is the plan's call,
        not this step's.
        """
        if body.language not in shop.locales:
            # The same `RequestValidationError` Pydantic would have raised, so
            # this is one response shape and one code path — and `loc`'s last
            # element is `language`, which is the field C11 branches on.
            raise RequestValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ("body", "language"),
                        "msg": f"language must be one of {list(shop.locales)}",
                        "input": body.language,
                    }
                ]
            )
        record = shop.join(body.displayName, body.language)
        return {
            "participantId": record.participant_id,
            "token": record.token,
            "displayName": record.display_name,
            "language": record.language,
            "welcome": _welcome(record.display_name, record.language),
        }

    @router.get(
        "/state",
        responses={
            200: {
                "description": "profile, cart, order and turn — the 2 s poll",
                "x-storefront-tokens": ["ok"],
            },
            401: {
                "description": "credential rejected (C3)",
                "x-storefront-tokens": ["invalid_token"],
            },
        },
    )
    def state(who: ParticipantRecord = Depends(get_participant)) -> dict[str, Any]:
        return shop.get_state(shop.context_for(who.participant_id))

    @router.get(
        "/messages",
        responses={
            200: {
                "description": "the participant's own thread, server-resolved",
                "x-storefront-tokens": ["ok"],
            },
            401: {
                "description": "credential rejected (C3)",
                "x-storefront-tokens": ["invalid_token"],
            },
            422: {
                "description": "`since`/`limit` out of range (C11, UI-supplied)",
                "x-storefront-tokens": ["validation_failed"],
            },
        },
    )
    def messages(
        who: ParticipantRecord = Depends(get_participant),
        since: int = Query(0, ge=0),
        limit: int = Query(DEFAULT_MESSAGE_PAGE, ge=1, le=MAX_MESSAGE_PAGE),
    ) -> list[dict[str, Any]]:
        """The participant's transcript since `since` — **a pure read**.

        `since` is always passed explicitly, and that is what puts this route in
        §5.3's `reads-only` class rather than in `writes`: `services.read_messages`
        with no `since` takes its cursor path, which *advances the cursor* — a
        write, on a route polled every 2 s, whose `503`/`504` meaning would then
        be wrong. The thread id comes from the resolved record, never from the
        client (§5.2).
        """
        return services.read_messages(
            shop.context_for(who.participant_id),
            thread_id=who.thread_id, since=since, limit=limit,
        )

    @router.post(
        "/messages",
        responses={
            200: {
                "description": "the posted row",
                "x-storefront-tokens": ["ok"],
            },
            401: {
                "description": "credential rejected (C3)",
                "x-storefront-tokens": ["invalid_token"],
            },
            409: {
                "description": "that participant already has a turn in flight — "
                               "**nothing written** (§4.4 measure 1a, C6a)",
                "x-storefront-tokens": ["turn_in_progress"],
            },
            422: {
                "description": "`text` bounds (C11, user-supplied)",
                "x-storefront-tokens": ["validation_failed"],
            },
        },
    )
    def post_message(
        body: PostMessageIn, who: ParticipantRecord = Depends(get_participant)
    ) -> dict[str, Any]:
        """Post into the participant's own thread, refusing **before** the write
        when a turn is already in flight (§4.4 measure 1a).

        The refusal is pre-write and server-side on purpose: `trigger` resumes
        only a `waiting` run, so a message posted while the first turn is still
        running starts a *second* `WorkflowRun` on the same thread — and a
        written message with no reply sits in the transcript forever.

        S9 adds the turn enqueue behind this write; the `409` gate and the
        `mentions` are here because both are properties of the *post*, not of
        the queue.
        """
        if shop.turn_in_flight(who.participant_id):
            raise StorefrontHTTPError(
                409, "turn_in_progress", "a turn is already in flight"
            )
        return services.post_message(
            shop.context_for(who.participant_id),
            thread_id=who.thread_id, text=body.text, mentions=[agent_id],
        )

    @router.get(
        "/catalog",
        responses={
            200: {
                "description": "every product with `imageUrl` or `null` (§4.7)",
                "x-storefront-tokens": ["ok"],
            },
            401: {
                "description": "credential rejected (C3)",
                "x-storefront-tokens": ["invalid_token"],
            },
        },
    )
    def catalog(
        _who: ParticipantRecord = Depends(get_participant),
    ) -> list[dict[str, Any]]:
        return shop.list_catalog()

    @router.post(
        "/order/advance",
        responses={
            200: {
                "description": "`{orderId, status}` after the transition",
                "x-storefront-tokens": ["ok"],
            },
            401: {
                "description": "credential rejected (C3)",
                "x-storefront-tokens": ["invalid_token"],
            },
            404: {
                "description": "no current order of theirs — an ordinary stale "
                               "button, never an auth failure (C10)",
                "x-storefront-tokens": ["no_current_order"],
            },
            409: {
                "description": "the CAS guard did not match — the order already "
                               "moved on (C10)",
                "x-storefront-tokens": ["order_transition_refused"],
            },
            422: {
                "description": "unknown `transition` (C11, UI-supplied)",
                "x-storefront-tokens": ["validation_failed"],
            },
        },
    )
    def advance_order(
        body: AdvanceOrderIn, who: ParticipantRecord = Depends(get_participant)
    ) -> dict[str, Any]:
        """Drive one lifecycle transition on the participant's **own** order.

        The order id comes from server-side state (§4.6), never from the body —
        which is what makes both refusals below mean *stale*, never *someone
        else's*. `advance_own_order` re-checks ownership anyway
        (`services.order_belongs_to_customer` before the CAS), so the guarantee
        is structural at two layers rather than by this route's discipline.
        """
        ctx = shop.context_for(who.participant_id)
        current = services.get_current_order(ctx)
        if current is None:
            raise StorefrontHTTPError(
                404, "no_current_order", "no current order for this participant"
            )
        try:
            return shop.advance_own_order(
                ctx, order_id=current["orderId"], transition=body.transition
            )
        except UnknownOrderError as exc:
            # The order was theirs a moment ago and is not now — a reset or a
            # racing sweep. Same answer as "no order of theirs": §4.6 makes the
            # two indistinguishable by construction.
            raise StorefrontHTTPError(
                404, "no_current_order", str(exc)
            ) from exc
        except OrderTransitionRefusedError as exc:
            raise StorefrontHTTPError(
                409, "order_transition_refused", str(exc), status=exc.status
            ) from exc

    @router.post(
        "/reset",
        responses={
            200: {
                "description": "`{threadId, language}` — the token survives, so "
                               "the client returns to the language step (C7)",
                "x-storefront-tokens": ["ok"],
            },
            401: {
                "description": "credential rejected (C3)",
                "x-storefront-tokens": ["invalid_token"],
            },
            404: {
                "description": "zero rows — not a participant, or already "
                               "deleted (C3)",
                "x-storefront-tokens": ["unknown_participant"],
            },
            409: {
                "description": "`scoped=false` — nothing was reset and nothing "
                               "will be until the graph is repaired (C6b)",
                "x-storefront-tokens": ["unscoped_participant"],
            },
            503: {
                "description": "quiesce timeout — **nothing changed** (C9)",
                "x-storefront-tokens": ["quiesce_timeout"],
            },
            504: {
                "description": "the delete may have committed (§4.8 F8, C4). "
                               "**Produced by the route**, not by the typed "
                               "handler, which must not pre-empt it",
                "x-storefront-tokens": ["reset_state_unknown"],
            },
            500: {
                "description": "a `Thread` UNIQUE violation (or any other "
                               "unmapped graph error) propagates as `5xx` and "
                               "is **never retried** (§5.2, C12)",
                "x-storefront-tokens": ["unhandled"],
            },
        },
    )
    def reset(who: ParticipantRecord = Depends(get_participant)) -> dict[str, Any]:
        """"Reset mine" — quiesce, then one atomic delete (§4.8).

        Takes the authenticated record rather than a bare id, so the profile
        re-write costs no second read: `resolve_token` has just read
        `displayName`/`language` from the graph on this same request, and those
        are exactly the fields the reset does not touch.
        """
        try:
            return shop.reset_participant(who)
        except QuiesceTimeoutError as exc:
            raise StorefrontHTTPError(503, "quiesce_timeout", str(exc)) from exc
        except UnscopedParticipantError as exc:
            raise StorefrontHTTPError(
                409, UnscopedParticipantError.code, str(exc)
            ) from exc
        except UnknownParticipantError as exc:
            raise StorefrontHTTPError(404, "unknown_participant", str(exc)) from exc
        except ResetStateUnknownError as exc:
            raise StorefrontHTTPError(
                504, ResetStateUnknownError.code, str(exc), state=exc.state
            ) from exc

    # ── the presenter surface (S10 interim — `_STEP_10_INTERIM`) ────────────

    @router.get(
        "/presenter/participants",
        responses={
            200: {
                "description": "the roster — exactly §5.2's four keys",
                "x-storefront-tokens": ["ok"],
            },
            401: {
                "description": "presenter session gone (C2)",
                "x-storefront-tokens": ["presenter_session_gone"],
            },
            403: {
                "description": "wrong credential type — a participant token "
                               "(the auth matrix, C2)",
                "x-storefront-tokens": ["wrong_credential_type"],
            },
        },
        dependencies=[Depends(get_presenter)],
    )
    def presenter_participants() -> list[dict[str, Any]]:
        """§5.2's four keys and nothing more.

        `list_participants` projects six — `channelId` and `threadId` are
        server-side ids no client needs (§4.3), so they are dropped here rather
        than narrowed in the query, which both resets also anchor on. **No
        activity stats**: composing them per participant would be ~150 extra
        graph queries per presenter poll at 50 participants (S10).
        """
        return [
            {
                "participantId": row["participantId"],
                "displayName": row["displayName"],
                "language": row["language"],
                "joinedAt": row["joinedAt"],
            }
            for row in repo.list_participants(shop.ws)
        ]

    @router.post(
        "/presenter/reset-all",
        responses={
            200: {
                "description": "`{clearedParticipants}` — plus `incomplete` and "
                               "`unresolved` when the graph left some "
                               "participant unresolvable (§5.2)",
                "x-storefront-tokens": ["ok", "incomplete"],
            },
            401: {
                "description": "presenter session gone (C2)",
                "x-storefront-tokens": ["presenter_session_gone"],
            },
            403: {
                "description": "wrong credential type (C2)",
                "x-storefront-tokens": ["wrong_credential_type"],
            },
            503: {
                "description": "quiesce timeout — **nothing changed** (C9)",
                "x-storefront-tokens": ["quiesce_timeout"],
            },
            504: {
                "description": "the sweep may have committed (§4.8 F8, C4+C5). "
                               "**Produced by the route**",
                "x-storefront-tokens": ["reset_state_unknown"],
            },
            500: {
                "description": "an unmapped graph error propagates as `5xx` and "
                               "is never retried (§5.2, C12) — this query "
                               "re-mints nothing, so it cannot raise the "
                               "`Thread` UNIQUE violation reset-mine can",
                "x-storefront-tokens": ["unhandled"],
            },
        },
        dependencies=[Depends(get_presenter)],
    )
    def presenter_reset_all() -> dict[str, Any]:
        """"Reset everyone" — drain, then one atomic sweep (§4.8).

        Every participant token is invalidated; the presenter's own is not, so
        they keep driving the demo through the reset. Their *participant* poll
        starts `401`-ing within one tick, which is §5.3 C3's headline scenario
        and C5's evidence.

        **What S10 adds and this does not have: the stop-intake flag.** §4.8's
        reset-everyone stops intake first, then drains, then deletes; the flag
        is `Storefront` state and `storefront.py` is not S8's file. The drain
        below is therefore the exact parallel of S7 shipping reset-mine's wait
        without S9's cancellation — it waits for what is in flight, which
        subsumes stopping intake for *correctness* of the sweep and differs
        only in whether a post that lands mid-drain extends the wait.
        """
        roster = repo.list_participants(shop.ws)
        deadline = time.monotonic() + shop.quiesce_s
        while any(shop.turn_in_flight(row["participantId"]) for row in roster):
            if time.monotonic() >= deadline:
                raise StorefrontHTTPError(
                    503, "quiesce_timeout",
                    f"a turn did not finish within {shop.quiesce_s}s — "
                    "nothing was reset",
                )
            time.sleep(QUIESCE_POLL_S)

        try:
            status = repo.reset_all_participants(shop.ws)
        except redis_exceptions.TimeoutError as exc:
            # F8, both orderings: the sweep may have committed, so this is
            # *unknown*, never the quiesce `503`. The re-read is another query
            # against the same graph and is the *likelier* second fault — a
            # second timeout still answers `504`, simply with no roster.
            shop.forget_all()
            shop.clear_all_turns()
            try:
                unresolved: list[dict[str, Any]] | None = [
                    {
                        "participantId": row["participantId"],
                        "displayName": row["displayName"],
                        "language": row["language"],
                        "joinedAt": row["joinedAt"],
                    }
                    for row in repo.list_participants(shop.ws)
                ]
            except redis_exceptions.TimeoutError:
                unresolved = None
            raise StorefrontHTTPError(
                504, ResetStateUnknownError.code,
                "the sweep timed out on the way to FalkorDB and may have "
                "committed",
                participants=unresolved,
            ) from exc

        shop.forget_all()
        shop.clear_all_turns()
        body: dict[str, Any] = {"clearedParticipants": status["userCount"]}
        if status["unscopedCount"]:
            # Not an error — the sweep did everything it could — but it must
            # not read as clean (§5.2). `unscopedCount == 0` returns no
            # `incomplete` field **at all**, never `incomplete: false`.
            body["incomplete"] = True
            body["unresolved"] = list(status["unscopedIds"])
        return body

    return router


# ── §4.9's readiness preflight ───────────────────────────────────────────────


def storefront_preflight(shop: Storefront) -> dict[str, Any]:
    """Refuse to start a mis-seeded storefront, naming the fix command.

    Three conditions, all of which produce a demo that comes up "green but
    dead" when they are false: the demo `Agent` resolves in `ws:{WS_ID}` (every
    storefront post mentions it, and an unresolvable mention raises *before any
    write*, so every participant's first message would 500); the trigger def's
    snapshot is materialised into that same workspace (no snapshot, no agent
    turn); and the catalog is non-empty.

    The image manifest is built here too (§4.7's "at startup only") but is
    **not** a preflight condition: an empty manifest is a legitimate deployment
    — the text-only card variant — so the count is logged and startup
    continues. Building it here rather than leaving it to `list_catalog` is
    what keeps the first participant's catalog fetch from listing a directory.
    """
    ws = shop.ws
    problems: list[str] = []

    # `resolve_member_kinds` is the same lookup `services.post_message` runs on
    # every post, so this preflight asks the exact question the demo will.
    kinds = shop._repo.resolve_member_kinds(  # noqa: SLF001 — see the router
        ws, ids=[shop._agent_id]  # noqa: SLF001
    )
    if kinds.get(shop._agent_id) != "Agent":  # noqa: SLF001
        problems.append(
            f"the demo agent {shop._agent_id!r} is not registered in "  # noqa: SLF001
            f"ws:{ws} — run ./scripts/seed_demo.sh {ws}"
        )

    ctx = shop.context_for(shop._agent_id)  # noqa: SLF001
    key, version = config.TRIGGER_DEF_KEY, config.TRIGGER_DEF_VERSION
    if shop._services.get_snapshot(ctx, key=key, version=version) is None:  # noqa: SLF001
        problems.append(
            f"{key}@{version} is not materialized into ws:{ws} — run "
            f"./scripts/seed_salesperson.sh {ws} (then ./scripts/verify_salesperson.sh {ws})"
        )

    catalog = shop.list_catalog()
    if not catalog:
        problems.append(
            "the product catalog is empty — run ./scripts/seed_catalog.sh "
            "(then ./scripts/verify_catalog.sh)"
        )

    if problems:
        raise StorefrontPreflightError(
            "the storefront refuses to start:\n  - " + "\n  - ".join(problems)
        )

    manifest = shop.build_image_manifest()
    _log.info(
        "storefront preflight ok: ws=%s agent=%s def=%s@%s products=%d images=%d",
        ws, shop._agent_id, key, version, len(catalog), len(manifest),  # noqa: SLF001
    )
    return {"products": len(catalog), "images": len(manifest)}
