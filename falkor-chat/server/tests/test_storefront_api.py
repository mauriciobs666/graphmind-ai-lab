"""The `/shop/api` contract, and the gate that makes it decidable (S8).

`docs/plans/salesperson-ui.md` §5.2 (the eleven-route surface), §5.3 (the
route-class table and the `(route, response)` completeness table), §6.2. Live
integration tests against `ws:test`, the same posture `test_storefront.py`
takes, except for the error-injection block which needs a repository that
raises and therefore builds its own.

**The gate is the point of this file, and it has two halves** (§5.1's S8 row):

*(i) Handler half* — `{registered handlers} × {routes that route class permits
them on} ⊆ §5.3's table`. The class filter is not a refinement, it *is* the
gate: without it the cross product is nonsense on the two routes that issue no
query at all, and the symmetric side falsely fails on exactly those two.

*(ii) Declaration half* — `{declared in each route's responses={…}} ∪
{handler-produced} == §5.3's table`, read back off `app.routes`.

**Both halves are demonstrated to fail**, in both directions, by
`test_the_gate_*` below: a handler with no row, an unclassified handler, a
route with no declaration, a declared response nobody produces, a table row
nobody declares, and a route reclassified out of `no graph access`. A gate that
cannot be shown to fail is not a gate.

**Three things Pass 10 established the two halves do not close**
(`docs/reviews/salesperson-ui-impl.md`), each repaired here rather than noted:

1. **The handler half's *input* was a delta, not a set** (P10-1). It subtracted
   a baseline app's handlers and so saw 5 of the 17 the storefront app carries.
   `registered_handlers` now enumerates `app.exception_handlers` whole, and
   `_assert_handler_ownership` checks that each of the four classifications is
   a true claim about who registered the handler — which is the part the delta
   was really buying.
2. **`RequestValidationError` is an envelope handler whose route set is
   derivable** (P10-3), so it is derived — `validating_routes` — and fed into
   the produced set. A route that gains a query parameter now reddens instead
   of producing an undeclared `422` in silence.
3. **"Every declared entry is proved producible" was a convention, not a
   mechanism** (P10-5). Every response every `TestClient` in this file receives
   is recorded, and the two tests at the end assert the observed set and the
   table agree in both directions. The `⊆` direction is what catches an unruled
   response *arriving from the server* on a route nobody wrote a test for; the
   `⊇` direction is what catches a declared row nothing produces.

**What still does not close, stated because the residue ships** (§5.3 C13's own
scoring): a route that `return`s a `JSONResponse(status_code=…)` never raises,
so it is invisible to the handler set — the observation check above narrows
even that, but only for a response some test happens to provoke.
"""

from __future__ import annotations

import ast
import hmac
import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from redis import exceptions as redis_exceptions
from test_app import _FASTAPI_BUILTIN_PATHS, _route_entries

from falkorchat import config, db, storefront, storefront_api
from falkorchat.app import create_app
from falkorchat.config import CallContext
from falkorchat.services import SearchNotAvailableError, ServiceError, Services
from falkorchat.storefront_api import (
    API_PREFIX,
    CROSS_CUTTING_HANDLERS,
    ENVELOPE_HANDLERS,
    INHERITED_HANDLERS,
    RESHAPED_HANDLERS,
    ROUTE_CLASSES,
    SERVICE_ERROR_RESPONSES,
    SERVICE_ERROR_ROUTES,
    SERVICE_ERRORS_UNREACHABLE,
    StorefrontPreflightError,
    cross_cutting_response,
    service_error_response,
)

WS = "test"
AGENT = "assistant"
PRESENTER_KEY = "presenter-secret"
# Short enough that the quiesce-timeout branch is a test rather than a wait.
QUIESCE_S = 0.15

CTX = lambda: CallContext(ws=WS, actor="u1")  # noqa: E731


# ═══════════════════════════════════════════════════════════════════════════
# §5.3's completeness table, transcribed
# ═══════════════════════════════════════════════════════════════════════════
#
# Key: `(METHOD, path)`. Value: every `(status, error token)` the server can
# produce there — a route's own returns **plus** whatever S8's typed handlers
# can produce on it, which is §5.3's stated generation rule. The rows that come
# from the handler set are marked `[X]`; everything else is the route's own.
#
# Three deliberate departures from §5.3's literal text, each one a *narrowing*
# of what the table can express rather than a change to what it says:
#
# 1. **The two `422` rows on `POST /shop/api/session`** (`displayName`,
#    user-supplied; `language`, UI-supplied) collapse to one entry. The
#    discriminator is the `field` in the response body, which is C11's client-
#    side key and is not expressible in a `responses={…}` declaration (FastAPI
#    keys those by status). Both are asserted separately, by execution, in
#    `test_join_reports_the_first_violation_by_declaration_order` and
#    `test_an_unknown_language_is_a_422_naming_language`.
# 2. **`5xx` is written `500`.** §5.3's `Thread` UNIQUE row is an *unhandled*
#    propagation, which is what Starlette turns into a 500; the token
#    `"unhandled"` names it. Producibility is asserted in
#    `test_an_unmapped_graph_error_propagates_as_5xx_and_is_never_retried`.
# 3. **`200 clean` and `200 + incomplete` on reset-all** are two rows, kept as
#    two, because §5.2 makes the difference load-bearing (`unscopedCount == 0`
#    returns no `incomplete` field **at all**, never `incomplete: false`).
TABLE: dict[tuple[str, str], set[tuple[int, str]]] = {
    # class `no graph access` — its whole response set is this one row (§5.3)
    ("GET", f"{API_PREFIX}/health"): {(200, "ok")},
    ("POST", f"{API_PREFIX}/session"): {
        (200, "ok"),
        (422, "validation_failed"),
        # §5.3 (plan v1.20): the demo `Agent` is gone, so `ensure_participant`
        # wrote nothing — C9's fourth source, not a new rule. Produced by the
        # route (`except DemoNotSeededError`), not by a handler.
        (503, "demo_not_seeded"),
        (503, "graph_unavailable"),          # [X]
        (504, "join_state_unknown"),         # [X] no re-read is possible (C4)
    },
    ("GET", f"{API_PREFIX}/state"): {
        (200, "ok"),
        (401, "invalid_token"),
        (503, "graph_unavailable"),          # [X]
        (503, "graph_read_timeout"),         # [X]
    },
    ("GET", f"{API_PREFIX}/messages"): {
        (200, "ok"),
        (401, "invalid_token"),
        (422, "validation_failed"),
        (503, "graph_unavailable"),          # [X]
        (503, "graph_read_timeout"),         # [X]
    },
    ("POST", f"{API_PREFIX}/messages"): {
        (200, "ok"),
        (401, "invalid_token"),
        (409, "turn_in_progress"),
        (422, "validation_failed"),
        # [X] `UnknownMemberError` — the demo `Agent` named in `mentions` is
        # gone, and `_validate_and_derive_role` raises **before any write**, so
        # this is the same condition, the same token and the same C9 rule as
        # join's row above, arriving one route over. **The plan owes this row**;
        # it is the only `(route, response)` pair S8b adds that v1.20 does not
        # already carry. `ThreadNotFoundError`/`UnknownActorError` from the same
        # handler land on `(401, invalid_token)`, which is already a row.
        (503, "demo_not_seeded"),
        (503, "graph_unavailable"),          # [X]
        (504, "post_state_unknown"),         # [X] written but never enqueued
    },
    ("GET", f"{API_PREFIX}/catalog"): {
        (200, "ok"),
        (401, "invalid_token"),
        (503, "graph_unavailable"),          # [X]
        (503, "graph_read_timeout"),         # [X]
    },
    ("POST", f"{API_PREFIX}/order/advance"): {
        (200, "ok"),
        (401, "invalid_token"),
        (404, "no_current_order"),
        (409, "order_transition_refused"),
        (422, "validation_failed"),
        (503, "graph_unavailable"),          # [X]
        (504, "order_state_unknown"),        # [X]
    },
    ("POST", f"{API_PREFIX}/reset"): {
        (200, "ok"),
        (401, "invalid_token"),
        (404, "unknown_participant"),
        (409, "unscoped_participant"),
        (503, "quiesce_timeout"),
        # Producer is the **route**, not the typed handler (§4.8 F8) — the
        # handler is the backstop for the other seven graph-touching routes and
        # must not pre-empt it. Both reach the same row.
        (504, "reset_state_unknown"),
        (500, "unhandled"),
        (503, "graph_unavailable"),          # [X]
    },
    # class `no graph access` — exactly these three rows (§5.3)
    ("POST", f"{API_PREFIX}/presenter/session"): {
        (200, "ok"),
        (403, "bad_presenter_key"),
        (422, "validation_failed"),
    },
    ("GET", f"{API_PREFIX}/presenter/participants"): {
        (200, "ok"),
        (401, "presenter_session_gone"),
        (403, "wrong_credential_type"),
        (503, "graph_unavailable"),          # [X]
        (503, "graph_read_timeout"),         # [X]
    },
    ("POST", f"{API_PREFIX}/presenter/reset-all"): {
        (200, "ok"),
        (200, "incomplete"),
        (401, "presenter_session_gone"),
        (403, "wrong_credential_type"),
        (503, "quiesce_timeout"),
        (504, "reset_state_unknown"),
        (500, "unhandled"),
        (503, "graph_unavailable"),          # [X]
    },
}

FLAT_TABLE = {
    (method, path, status, token)
    for (method, path), rows in TABLE.items()
    for status, token in rows
}


# ═══════════════════════════════════════════════════════════════════════════
# The third half of the gate: every row proved producible, by observation
# ═══════════════════════════════════════════════════════════════════════════
#
# The two halves compare **declarations** — one against the table, one against
# the handler set. Neither can tell whether a declared row is producible at
# all, and the file's own claim that "every declared entry is proved producible
# by a contract test below" was a *convention*: nothing linked a row to a test,
# and two declared rows had no producer at all
# (`docs/reviews/salesperson-ui-impl.md` `## Pass 10`, P10-5, mutations M-D and
# M-V, both survived).
#
# The link is mechanical here, and it is made by **observation rather than by
# tagging**: every response every `TestClient` in this file receives is
# recorded, and the two tests at the end of the file assert that the observed
# set and §5.3's table agree. A tag is a claim about what a test proves and can
# be wrong; a recorded response is what the server actually said.
_OBSERVED: set[tuple[str, str, int, str]] = set()


def _observed_token(response) -> str:
    """The `(status, token)` key §5.3 uses, read off a real response.

    `200` carries no `error` field, so it is `"ok"` — except reset-all's
    `incomplete` body, which §5.2 makes a **different row** from its clean one.
    A body with no JSON at all is `"unhandled"`, which is the token §5.3 gives
    the `5xx` propagation row and is exactly what a bare `500` looks like.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code < 300:
        return "incomplete" if isinstance(body, dict) and body.get("incomplete") else "ok"
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return body["error"]
    return "unhandled"


class _RecordingTestClient(TestClient):
    """A `TestClient` that records what it saw, and changes nothing else.

    Subclassed and rebound over the imported name below, so every construction
    site in this file is covered without editing any of them — including the
    ones a later step adds.
    """

    def request(self, method, url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        response = super().request(method, url, *args, **kwargs)
        key = (str(method).upper(), str(url).split("?")[0])
        if key in ROUTE_CLASSES:
            _OBSERVED.add((*key, response.status_code, _observed_token(response)))
        return response


TestClient = _RecordingTestClient  # noqa: F811 — see the class docstring


# ═══════════════════════════════════════════════════════════════════════════
# The gate
# ═══════════════════════════════════════════════════════════════════════════


def registered_handlers(app) -> set:
    """**Every** exception handler on the app object — the whole set.

    This used to be a *difference* against `create_app(dev_surface=False)`, on
    the reasoning that a baseline diff finds a handler nobody remembered. It
    does — but `create_app` builds one app, so the eleven legacy workflow
    handlers, the `ServiceError` handler and three framework defaults are on
    the storefront deployment too, and subtracting them showed the gate **5**
    of **17**. One of the twelve it subtracted provably fires: the delivered
    app answered `POST /shop/api/messages` with
    `404 {"error":"ThreadNotFoundError"}` — a `(route, response)` pair invisible
    to both halves of the gate and to the AST refusal check
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 10`, P10-1).

    §5.1 S8 asks for "the handlers **actually registered on the app object**",
    which is this. What the baseline diff was really buying — noticing an
    *override*, which changes a handler's value and not its key — is bought
    instead by `_assert_handler_ownership` below, on the axis that actually
    matters here: who registered it.
    """
    return set(app.exception_handlers)


def _assert_handler_ownership(app) -> None:
    """A classification is a claim about **who registered** the handler; check it.

    Without this, the four buckets are keyed on the exception type alone, so
    swapping a storefront handler onto an inherited key — or letting an
    inherited handler answer where the storefront promised its own envelope —
    changes only the value and passes. Both directions are asserted.
    """
    own = (
        set(CROSS_CUTTING_HANDLERS)
        | set(ENVELOPE_HANDLERS)
        | set(RESHAPED_HANDLERS)
    )
    module = storefront_api.__name__
    for exc in own:
        handler = app.exception_handlers[exc]
        if getattr(handler, "__module__", "") != module:
            raise AssertionError(
                f"{exc.__name__} is classified as the storefront's own but is "
                f"handled by {getattr(handler, '__module__', '?')}"
            )
    for exc in INHERITED_HANDLERS:
        handler = app.exception_handlers[exc]
        if getattr(handler, "__module__", "") == module:
            raise AssertionError(
                f"{exc.__name__} is classified as inherited — 'produces no "
                "row on a /shop/api route' — but the storefront registered it"
            )


def storefront_routes(app) -> dict[tuple[str, str], dict]:
    """`(METHOD, path) -> responses={…}` for the storefront's own routes.

    Reads through `test_app._route_entries`, which threads `include_router`'s
    prefix through the walk — `/shop/api/...` is two levels of prefix and is
    exactly the case a naive walk reports as `/state`
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 5`).
    """
    found: dict[tuple[str, str], dict] = {}
    for entry in _route_entries(app):
        if entry.path in _FASTAPI_BUILTIN_PATHS:
            continue
        if not entry.path.startswith(API_PREFIX):
            continue
        for method in entry.methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            found[(method, entry.path)] = entry.responses
    return found


def declared_pairs(app) -> set[tuple[str, str, int, str]]:
    """Every `(method, path, status, token)` the routes declare for themselves."""
    pairs: set[tuple[str, str, int, str]] = set()
    for (method, path), responses in storefront_routes(app).items():
        if not responses:
            raise AssertionError(
                f"{method} {path} carries no `responses={{…}}` declaration — "
                "every one of the eleven routes must name its own returns, so "
                "an omission fails loudly rather than silently shrinking the set"
            )
        for status, spec in responses.items():
            tokens = spec.get("x-storefront-tokens")
            if not tokens:
                raise AssertionError(
                    f"{method} {path} declares {status} with no "
                    "`x-storefront-tokens` — the status alone cannot separate "
                    "`503 quiesce_timeout` from `503 graph_unavailable`"
                )
            for token in tokens:
                pairs.add((method, path, status, token))
    return pairs


def validating_routes(app) -> set[tuple[str, str]]:
    """The routes that can produce a `422`, **derived from the route objects**.

    `RequestValidationError` is an envelope handler, so it contributes nothing
    to the `{handlers} × {routes}` cross product and the plan leaves it to "the
    declaration half plus the per-route contract tests". For
    `StorefrontHTTPError` that holds. For this one it does not: the framework
    raises it, not a route body, so no AST check sees it, and the declaration
    half then compares two hand-written enumerations that can be wrong together
    — a route gaining a query parameter produces an undeclared, untabled `422`
    and nothing reddens (`docs/reviews/salesperson-ui-impl.md` `## Pass 10`,
    P10-3, mutation M-C).

    It is the one handler whose route set is **mechanically derivable**: FastAPI
    validates a request iff the route has a body model or any query/path
    parameter, its own or a dependency's. So it is derived here and fed into the
    gate's produced set like a cross-cutting handler, and the gate's symmetric
    side then reddens on M-C instead of shrugging.
    """
    found: set[tuple[str, str]] = set()
    for path, route in _raw_routes(app):
        if not path.startswith(API_PREFIX):
            continue
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        if getattr(route, "body_field", None) is None and not _takes_params(dependant):
            continue
        for method in getattr(route, "methods", None) or ():
            if method not in {"HEAD", "OPTIONS"}:
                found.add((method, path))
    return found


def _takes_params(dependant) -> bool:
    """Query or path parameters on this dependant or any sub-dependency.

    Header parameters are deliberately excluded: the only one the storefront
    declares is `Authorization: str | None`, which cannot fail validation — a
    bad credential is `get_participant`'s `401`, not a `422`.
    """
    if dependant.query_params or dependant.path_params:
        return True
    return any(_takes_params(sub) for sub in dependant.dependencies)


def validation_pairs(app) -> set[tuple[str, str, int, str]]:
    """`validating_routes` as `(method, path, 422, "validation_failed")` rows."""
    return {(method, path, 422, "validation_failed")
            for method, path in validating_routes(app)}


def service_error_pairs(app) -> set[tuple[str, str, int, str]]:
    """The re-shaping `ServiceError` handler × the routes it can fire on.

    `SERVICE_ERROR_ROUTES` is a measurement, not a reading — see
    `test_only_post_messages_can_raise_a_service_error`, which arms each fault
    in turn and drives all eleven routes.
    """
    registered = set(storefront_routes(app))
    unregistered = SERVICE_ERROR_ROUTES - registered
    if unregistered:
        raise AssertionError(
            f"SERVICE_ERROR_ROUTES names routes this app does not carry: "
            f"{sorted(unregistered)}"
        )
    return {
        (method, path, status, token)
        for method, path in SERVICE_ERROR_ROUTES
        for status, token in set(SERVICE_ERROR_RESPONSES.values())
    }


def handler_produced_pairs(app) -> set[tuple[str, str, int, str]]:
    """The handler half's cross product: registered cross-cutting handlers ×
    the routes their class permits them on.

    Raises before computing anything if a handler is registered that carries no
    classification at all — *a handler with no row fails the step*, and this is
    where that is enforced, over the **whole** handler set.
    """
    registered = registered_handlers(app)
    classified = (
        set(CROSS_CUTTING_HANDLERS)
        | set(ENVELOPE_HANDLERS)
        | set(RESHAPED_HANDLERS)
        | set(INHERITED_HANDLERS)
    )
    unclassified = registered - classified
    if unclassified:
        raise AssertionError(
            "handler(s) registered on the storefront app with no classification "
            f"in `storefront_api`: {sorted(e.__name__ for e in unclassified)} — "
            "a handler must declare whether it produces one of §5.3's three "
            "cross-cutting responses, re-shapes a declared one, re-shapes an "
            "inherited one on /shop/api, or produces no row at all"
        )
    missing = classified - registered
    if missing:
        raise AssertionError(
            f"classified but not registered: {sorted(e.__name__ for e in missing)}"
        )
    _assert_handler_ownership(app)

    pairs: set[tuple[str, str, int, str]] = set()
    for handler_token in set(CROSS_CUTTING_HANDLERS.values()):
        for method, path in storefront_routes(app):
            # A `KeyError` here is the route-table assertion firing from the
            # inside: an unclassified route cannot be crossed with anything.
            answer = cross_cutting_response(handler_token, method, path)
            if answer is not None:
                pairs.add((method, path, *answer))
    return pairs


def evaluate_gate(app) -> None:
    """Run both halves of S8's gate against `app`. Raises `AssertionError`."""
    routes = set(storefront_routes(app))
    classified = set(ROUTE_CLASSES)
    if routes != classified:
        raise AssertionError(
            "the route table and §5.3's route-class table disagree — "
            f"registered but unclassified: {sorted(routes - classified)}; "
            f"classified but unregistered: {sorted(classified - routes)}"
        )

    declared = declared_pairs(app)
    produced = (
        handler_produced_pairs(app)
        | service_error_pairs(app)
        | validation_pairs(app)
    )

    # (i.a) the `422` half, both directions. Derived from the route objects, so
    # a route that *gains* a validating parameter and a route that declares a
    # `422` it cannot produce both fail here rather than agreeing by omission.
    derived_422 = validation_pairs(app)
    declared_422 = {row for row in declared if row[2] == 422}
    if derived_422 != declared_422:
        raise AssertionError(
            "the routes FastAPI will validate and the routes declaring a `422` "
            f"disagree — validating but undeclared: "
            f"{sorted(derived_422 - declared_422)}; declared but not "
            f"validating: {sorted(declared_422 - derived_422)}"
        )

    # (i) handler half
    orphan_handler_rows = produced - FLAT_TABLE
    if orphan_handler_rows:
        raise AssertionError(
            "handler-produced (route, response) pairs with no row in §5.3's "
            f"completeness table: {sorted(orphan_handler_rows)}"
        )

    # (ii) declaration half — and its symmetric side, which is what catches a
    # table row nobody produces.
    covered = declared | produced
    undeclared = FLAT_TABLE - covered
    if undeclared:
        raise AssertionError(
            "rows in §5.3's completeness table that nothing on this app "
            f"produces or declares: {sorted(undeclared)}"
        )
    extra = covered - FLAT_TABLE
    if extra:
        raise AssertionError(
            "responses this app declares or produces that §5.3's completeness "
            f"table does not carry: {sorted(extra)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════
#
# `_catalog_rows` / `_seed_catalog` / `_ticking_clock` are deliberate local
# copies of `test_storefront.py`'s (S7). S9 rewrites that file — it deletes the
# record cache and every test of it — so importing its private helpers across
# steps would couple this file's survival to an edit it does not control.


def _catalog_rows(n: int) -> list[dict]:
    return [
        {
            "productId": f"widget-{i:03d}", "name": f"Widget {i:03d}",
            "nameNormalized": f"widget {i:03d}", "category": "Accessories",
            "categoryNormalized": "accessories", "price": float(10 + i),
        }
        for i in range(1, n + 1)
    ]


def _seed_catalog(conn, rows):
    db.reference_graph(conn).query(
        "UNWIND $rows AS row "
        "CREATE (:Product {productId: row.productId, name: row.name, "
        "                  nameNormalized: row.nameNormalized, "
        "                  category: row.category, "
        "                  categoryNormalized: row.categoryNormalized, "
        "                  price: row.price})",
        {"rows": rows},
    )
    return rows


def _ticking_clock(start: int = 1_700_000_000_000):
    """Strictly increasing ms — `Order.placedAt` ties break by `orderId DESC`,
    so two orders in one millisecond make "current order" a coin flip."""
    counter = iter(range(start, start + 1_000_000))
    return lambda: next(counter)


_SNAPSHOT_STEPS = [
    {"key": "intake", "type": "agent", "name": "Intake", "config": "{}",
     "waitsForHuman": True, "order": 0},
    {"key": "answer", "type": "agent", "name": "Answer", "config": "{}",
     "waitsForHuman": False, "order": 1},
]
_SNAPSHOT_TRANSITIONS = [
    {"from": "intake", "to": "answer", "on": "ready", "guard": "", "order": 0},
]


def _seed_preflight(repo, conn, *, products: int = 3) -> None:
    """The three things §4.9's readiness preflight refuses to start without."""
    repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    repo.materialize_snapshot(
        WS, key=config.TRIGGER_DEF_KEY, version=config.TRIGGER_DEF_VERSION,
        name="Trigger def", kind="conversation", start_key="intake",
        steps=_SNAPSHOT_STEPS, transitions=_SNAPSHOT_TRANSITIONS,
    )
    if products:
        _seed_catalog(conn, _catalog_rows(products))


@pytest.fixture()
def storefront_config(monkeypatch, tmp_path):
    """`config` pinned the way `start_demo.sh` will pin it, minus the waits."""
    monkeypatch.setattr(config, "STOREFRONT_PRESENTER_KEY", PRESENTER_KEY)
    monkeypatch.setattr(config, "STOREFRONT_QUIESCE_S", QUIESCE_S)
    monkeypatch.setattr(config, "STOREFRONT_TURN_WORKERS", 2)
    # Pointed at a real but empty directory rather than left unset: an unset
    # `config.STOREFRONT_DIR` would let a `create_app` that reads config instead
    # of forwarding its parameter pass the image-wiring test with `null` URLs.
    empty = tmp_path / "config-default"
    (empty / "products").mkdir(parents=True)
    monkeypatch.setattr(config, "STOREFRONT_DIR", str(empty))
    return empty


@pytest.fixture()
def seeded(conn, wf_repo, storefront_config):
    """`ws:test` + `reference` wiped, then seeded for the preflight.

    `wf_repo` wipes `reference` on *setup* only, so the teardown here leaves it
    empty — `seed_catalog.sh` MERGEs by `productId`, and a stray `widget-…`
    would survive the re-seed a default `pytest` run already obliges and then
    make `verify_catalog.sh` report a mismatch to whoever ran it next.
    """
    _seed_preflight(wf_repo, conn)
    yield wf_repo
    db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")


def _build_app(repo, *, storefront_dir=None):
    return create_app(
        Services(repo, clock=_ticking_clock()),
        context_provider=CTX, mount_mcp=False, dev_surface=False,
        storefront=True, storefront_dir=storefront_dir,
    )


@pytest.fixture()
def client(seeded):
    """A `TestClient` over the storefront app, lifespan run — so the readiness
    preflight and the startup image-manifest build execute on every test here."""
    with TestClient(_build_app(seeded)) as c:
        yield c


def _join(client, name="Ada", language="en") -> dict:
    response = client.post(f"{API_PREFIX}/session",
                           json={"displayName": name, "language": language})
    assert response.status_code == 200, response.text
    return response.json()


def _bearer(session: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {session['participantId']}.{session['token']}"}


def _presenter(client, key: str = PRESENTER_KEY) -> dict[str, str]:
    response = client.post(f"{API_PREFIX}/presenter/session", json={"key": key})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer presenter.{response.json()['token']}"}


# Every route, with a request that is valid apart from its credential — so an
# auth assertion is never confounded by a 422 from a missing body. Asserted
# against `ROUTE_CLASSES` below, so a twelfth route cannot be added without one.
CALLS: dict[tuple[str, str], dict] = {
    ("GET", f"{API_PREFIX}/health"): {},
    ("POST", f"{API_PREFIX}/session"): {
        "json": {"displayName": "Zoe", "language": "en"}
    },
    ("GET", f"{API_PREFIX}/state"): {},
    ("GET", f"{API_PREFIX}/messages"): {"params": {"since": 0, "limit": 10}},
    ("POST", f"{API_PREFIX}/messages"): {"json": {"text": "hello"}},
    ("GET", f"{API_PREFIX}/catalog"): {},
    ("POST", f"{API_PREFIX}/order/advance"): {"json": {"transition": "cancel"}},
    ("POST", f"{API_PREFIX}/reset"): {},
    ("POST", f"{API_PREFIX}/presenter/session"): {"json": {"key": PRESENTER_KEY}},
    ("GET", f"{API_PREFIX}/presenter/participants"): {},
    ("POST", f"{API_PREFIX}/presenter/reset-all"): {},
}

# The routes that carry a participant credential, and the ones that carry the
# presenter's (§5.3's credentials table, "Sent on" rows).
PARTICIPANT_ROUTES = frozenset({
    ("GET", f"{API_PREFIX}/state"),
    ("GET", f"{API_PREFIX}/messages"),
    ("POST", f"{API_PREFIX}/messages"),
    ("GET", f"{API_PREFIX}/catalog"),
    ("POST", f"{API_PREFIX}/order/advance"),
    ("POST", f"{API_PREFIX}/reset"),
})
PRESENTER_ROUTES = frozenset({
    ("GET", f"{API_PREFIX}/presenter/participants"),
    ("POST", f"{API_PREFIX}/presenter/reset-all"),
})
OPEN_ROUTES = frozenset({
    ("GET", f"{API_PREFIX}/health"),
    ("POST", f"{API_PREFIX}/session"),
    ("POST", f"{API_PREFIX}/presenter/session"),
})


def _call(client, method, path, headers=None):
    return client.request(method, path, headers=headers or {}, **CALLS[(method, path)])


# ═══════════════════════════════════════════════════════════════════════════
# The gate — and the demonstration that both halves fail when they should
# ═══════════════════════════════════════════════════════════════════════════
#
# Every test in this block builds the app with the default (deferred,
# network-free) services: the gate reads the route table and the handler map,
# and touches no graph at all.


def _gate_app():
    return create_app(dev_surface=False, storefront=True)


def _raw_routes(app):
    """The live route objects, with their accumulated prefix.

    Exists **only** so the mutation tests below can break a real declaration;
    the gate itself reads through `test_app._route_entries`, which copies. Kept
    separate deliberately — a harness that can mutate what the assertion reads
    is a harness that can also silently repair it.
    """
    found = []

    def walk(routes, prefix=""):
        for route in routes:
            nested = getattr(getattr(route, "original_router", None), "routes", None)
            if nested is not None:
                own = getattr(getattr(route, "include_context", None), "prefix", "")
                walk(nested, prefix + (own or ""))
                continue
            found.append((prefix + getattr(route, "path", ""), route))

    walk(app.routes)
    return found


def _route_object(app, method, path):
    matches = [
        route
        for route_path, route in _raw_routes(app)
        if route_path == path and method in (getattr(route, "methods", None) or ())
    ]
    assert len(matches) == 1, f"expected one route at {method} {path}, got {matches}"
    return matches[0]


def test_the_storefront_registers_exactly_the_eleven_classified_routes():
    """The control for everything below: the route table this gate is evaluated
    over is the one §5.3 classifies, at the paths §5.2 names — not a subset the
    walk happened to find, and not at some other prefix.
    """
    assert set(storefront_routes(_gate_app())) == set(ROUTE_CLASSES)
    assert len(ROUTE_CLASSES) == 11
    # §5.3's own counts: five `writes`, four `reads-only`, two `no graph access`
    classes = [klass for klass, _ in ROUTE_CLASSES.values()]
    assert classes.count(storefront_api.WRITES) == 5
    assert classes.count(storefront_api.READS_ONLY) == 4
    assert classes.count(storefront_api.NO_GRAPH) == 2
    # and the request table this file drives them with covers all of them
    assert set(CALLS) == set(ROUTE_CLASSES)
    assert (
        PARTICIPANT_ROUTES | PRESENTER_ROUTES | OPEN_ROUTES == set(ROUTE_CLASSES)
    )


def test_the_gate_passes_on_the_delivered_app():
    """Both halves, against the app `create_app` actually builds."""
    evaluate_gate(_gate_app())


def test_the_two_no_graph_routes_take_none_of_the_three_cross_cutting_rows():
    """§5.3's classification, read off the seam the live handler uses.

    Asserted here as well as by execution below because this is the input the
    whole gate is computed from: without it the cross product is nonsense on
    these two routes and the symmetric side falsely fails on them.
    """
    for method, path in ROUTE_CLASSES:
        klass, _ = ROUTE_CLASSES[(method, path)]
        answers = {
            cross_cutting_response(token, method, path)
            for token in set(CROSS_CUTTING_HANDLERS.values())
        }
        if klass == storefront_api.NO_GRAPH:
            assert answers == {None}, (method, path, answers)
        else:
            assert None not in answers, (method, path, answers)


def test_every_writing_route_gets_its_own_named_504_op_token():
    """§5.3's cross-cutting table is **one `504` row per writing route**, not one
    row spanning five — because C4's action (which endpoint to re-read) differs
    per route, and on join it is not a re-read at all."""
    ops = {
        path: cross_cutting_response("graph_timeout", method, path)
        for (method, path), (klass, _) in ROUTE_CLASSES.items()
        if klass == storefront_api.WRITES
    }
    assert ops == {
        f"{API_PREFIX}/session": (504, "join_state_unknown"),
        f"{API_PREFIX}/messages": (504, "post_state_unknown"),
        f"{API_PREFIX}/order/advance": (504, "order_state_unknown"),
        f"{API_PREFIX}/reset": (504, "reset_state_unknown"),
        f"{API_PREFIX}/presenter/reset-all": (504, "reset_state_unknown"),
    }
    # and no `reads-only` route ever answers a 504 — a read that times out
    # changed nothing, by definition
    reads = {
        cross_cutting_response("graph_timeout", method, path)
        for (method, path), (klass, _) in ROUTE_CLASSES.items()
        if klass == storefront_api.READS_ONLY
    }
    assert reads == {(503, "graph_read_timeout")}


# ── the gate fails when it should: the handler half ──────────────────────────


def test_the_gate_fails_when_a_handler_is_registered_with_no_classification():
    """*A handler with no row fails the step* — the enumeration side.

    The mutation is the realistic one: someone adds a typed handler to
    `register_storefront_error_handlers` and does not touch
    `CROSS_CUTTING_HANDLERS`. The baseline diff sees it whether or not anybody
    remembered it existed.
    """
    app = _gate_app()
    evaluate_gate(app)  # green before

    class _Mutant(Exception):
        pass

    async def _handle(_request, _exc):
        raise NotImplementedError  # pragma: no cover — never invoked

    app.add_exception_handler(_Mutant, _handle)

    with pytest.raises(AssertionError, match="no classification"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_classified_handler_is_not_registered(monkeypatch):
    """The other direction of the same check: a handler named in the map but
    never wired. Without it, the classification could drift into fiction and the
    cross product would keep producing rows nothing can raise."""
    app = _gate_app()
    evaluate_gate(app)

    class _Ghost(Exception):
        pass

    monkeypatch.setitem(CROSS_CUTTING_HANDLERS, _Ghost, "graph_unavailable")

    with pytest.raises(AssertionError, match="classified but not registered"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_handler_can_fire_on_a_route_with_no_such_row(
    monkeypatch,
):
    """*A handler with no table row reddens it* — the cross-product side, and
    the demonstration that the class filter **is** the gate.

    The mutation re-classifies `GET /shop/api/health` out of `no graph access`,
    which is exactly what a future edit that couples it to the graph would do.
    §5.3 gives that route one row and one only, so both cross-cutting handlers
    immediately produce pairs the table does not carry.
    """
    app = _gate_app()
    evaluate_gate(app)

    monkeypatch.setitem(
        ROUTE_CLASSES,
        ("GET", f"{API_PREFIX}/health"),
        (storefront_api.READS_ONLY, None),
    )

    with pytest.raises(AssertionError, match="no row in §5.3's completeness table"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_route_is_registered_with_no_class(monkeypatch):
    """A twelfth route added to the router without a `ROUTE_CLASSES` entry.

    This is the failure that makes the handler half *computable at all*: an
    unclassified route cannot be crossed with anything, so it would otherwise
    contribute silently nothing rather than loudly failing.
    """
    app = _gate_app()
    evaluate_gate(app)

    from fastapi import APIRouter

    extra = APIRouter()

    @extra.get("/twelfth", responses={200: {"x-storefront-tokens": ["ok"]}})
    def twelfth():  # pragma: no cover — only its registration matters
        return {}

    app.include_router(extra, prefix=API_PREFIX)

    with pytest.raises(AssertionError, match="registered but unclassified"):
        evaluate_gate(app)


# ── the gate fails when it should: the declaration half ──────────────────────


def test_the_gate_fails_when_a_route_carries_no_responses_declaration():
    """*Every route must carry a declaration, so an omission fails loudly
    rather than silently shrinking the set.*"""
    app = _gate_app()
    evaluate_gate(app)

    _route_object(app, "GET", f"{API_PREFIX}/catalog").responses = {}

    with pytest.raises(AssertionError, match="carries no `responses="):
        evaluate_gate(app)


def test_the_gate_fails_when_a_declared_response_carries_no_token():
    """A declaration keyed on the status alone cannot separate
    `503 quiesce_timeout` from `503 graph_unavailable`, which is the pair §5.3
    keys the whole table on `(route, response)` to keep apart."""
    app = _gate_app()
    route = _route_object(app, "POST", f"{API_PREFIX}/reset")
    route.responses = {**route.responses, 503: {"description": "quiesce"}}

    with pytest.raises(AssertionError, match="no `x-storefront-tokens`"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_table_row_has_no_producer():
    """The symmetric side: drop one declared response and the row it covered is
    left with nothing that produces or declares it."""
    app = _gate_app()
    route = _route_object(app, "POST", f"{API_PREFIX}/messages")
    assert 409 in route.responses
    route.responses = {k: v for k, v in route.responses.items() if k != 409}

    with pytest.raises(AssertionError, match="nothing on this app"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_route_declares_a_response_nobody_carries():
    """A declaration is itself an enumeration that can be wrong in **both**
    directions — this is the other one."""
    app = _gate_app()
    route = _route_object(app, "GET", f"{API_PREFIX}/state")
    route.responses = {
        **route.responses,
        418: {"description": "teapot", "x-storefront-tokens": ["teapot"]},
    }

    with pytest.raises(AssertionError, match="table does not carry"):
        evaluate_gate(app)


# ── the gate's *input set*: every handler on the app, classified ─────────────


def test_the_gate_sees_every_handler_the_app_carries():
    """The control for the handler half, and the fix for P10-1.

    The count is asserted, not just the partition: the delivered gate computed
    a difference against a baseline app and therefore saw **5** of these — the
    twelve it subtracted included `ServiceError`, which is live on `/shop/api`
    because the storefront calls the same `services` layer.
    """
    app = _gate_app()
    registered = registered_handlers(app)
    assert len(registered) == 17, sorted(e.__name__ for e in registered)
    assert registered == (
        set(CROSS_CUTTING_HANDLERS)
        | set(ENVELOPE_HANDLERS)
        | set(RESHAPED_HANDLERS)
        | set(INHERITED_HANDLERS)
    )
    # the four buckets are a partition, not merely a cover — a handler with two
    # classifications is two claims about the same response
    buckets = [
        set(CROSS_CUTTING_HANDLERS), set(ENVELOPE_HANDLERS),
        set(RESHAPED_HANDLERS), set(INHERITED_HANDLERS),
    ]
    assert sum(len(bucket) for bucket in buckets) == len(registered)
    _assert_handler_ownership(app)


def test_the_gate_fails_when_an_inherited_handler_stops_being_inherited():
    """An `INHERITED_HANDLERS` entry says *this one produces no row*. Registering
    a storefront handler on that key makes the claim false while leaving the key
    set — and therefore a key-only classification — unchanged."""
    app = _gate_app()
    evaluate_gate(app)

    async def _handle(_request, _exc):  # pragma: no cover — never invoked
        raise NotImplementedError

    _handle.__module__ = storefront_api.__name__
    app.add_exception_handler(SearchNotAvailableError, _handle)

    with pytest.raises(AssertionError, match="but the storefront registered it"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_storefront_handler_is_left_to_the_inherited_one():
    """The other direction, and the exact regression P10-1 reported: dropping the
    storefront's `ServiceError` re-shaper leaves `app.py`'s handler answering
    `{"error": "<Python class name>"}` on `/shop/api` — with the key set, and so
    a key-only classification, still unchanged."""
    app = _gate_app()
    evaluate_gate(app)

    async def _inherited(_request, _exc):  # pragma: no cover — never invoked
        raise NotImplementedError

    _inherited.__module__ = "falkorchat.app"
    app.add_exception_handler(ServiceError, _inherited)

    with pytest.raises(AssertionError, match="classified as the storefront's own"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_route_gains_a_parameter_fastapi_will_validate(
    monkeypatch,
):
    """P10-3's mutation M-C, which survived the delivered gate: give
    `GET /shop/api/catalog` a query parameter and it produces a `422` that is
    undeclared and untabled. Derived from the route object, so this reddens."""
    app = _gate_app()
    evaluate_gate(app)

    catalog = _route_object(app, "GET", f"{API_PREFIX}/catalog")
    borrowed = _route_object(app, "GET", f"{API_PREFIX}/messages")
    monkeypatch.setattr(
        catalog.dependant, "query_params", list(borrowed.dependant.query_params)
    )

    with pytest.raises(AssertionError, match="validating but undeclared"):
        evaluate_gate(app)


def test_the_gate_fails_when_a_route_declares_a_422_it_cannot_produce():
    """The other direction of the same derivation — a `422` on a route that
    takes no body, query or path parameter and so can never raise one."""
    app = _gate_app()
    route = _route_object(app, "GET", f"{API_PREFIX}/catalog")
    route.responses = {
        **route.responses,
        422: {"description": "invented", "x-storefront-tokens": ["validation_failed"]},
    }

    with pytest.raises(AssertionError, match="declared but not validating"):
        evaluate_gate(app)


def test_the_derived_422_routes_are_the_five_that_declare_one():
    """The derivation's own control: it reproduces the declared set exactly, in
    both directions, on the delivered app — so the test above is measuring the
    mutation and not a pre-existing disagreement."""
    app = _gate_app()
    assert validating_routes(app) == {
        ("POST", f"{API_PREFIX}/session"),
        ("GET", f"{API_PREFIX}/messages"),
        ("POST", f"{API_PREFIX}/messages"),
        ("POST", f"{API_PREFIX}/order/advance"),
        ("POST", f"{API_PREFIX}/presenter/session"),
    }


def test_the_credential_route_sets_are_the_ones_the_dependencies_declare():
    """`PARTICIPANT_ROUTES`/`PRESENTER_ROUTES` are hand-written and the AST
    refusal check attributes `get_participant`'s `401` through them — so a route
    that *gains* `Depends(get_participant)` while staying in `OPEN_ROUTES` would
    be attributed nothing (`docs/reviews/salesperson-ui-impl.md` `## Pass 10`,
    P10-11). Derived from each route's own dependant instead, and the literals
    are asserted against the derivation rather than trusted."""
    app = _gate_app()
    derived: dict[str, set[tuple[str, str]]] = {"participant": set(), "presenter": set()}
    for path, route in _raw_routes(app):
        if not path.startswith(API_PREFIX):
            continue
        names = _dependency_names(route.dependant)
        for method in getattr(route, "methods", None) or ():
            if method in {"HEAD", "OPTIONS"}:
                continue
            if "get_participant" in names:
                derived["participant"].add((method, path))
            if "get_presenter" in names:
                derived["presenter"].add((method, path))

    assert derived["participant"] == set(PARTICIPANT_ROUTES)
    assert derived["presenter"] == set(PRESENTER_ROUTES)
    # and the two are disjoint — §5.3 C1's route→credential bijection, which is
    # what licenses C2/C3 being stated by route rather than by credential
    assert not derived["participant"] & derived["presenter"]


def _dependency_names(dependant) -> set[str]:
    """Every dependency callable's name on a route, recursively."""
    names = {getattr(dependant.call, "__name__", "")}
    for sub in dependant.dependencies:
        names |= _dependency_names(sub)
    return names


# ═══════════════════════════════════════════════════════════════════════════
# Contract tests — every declared response, proved producible
# ═══════════════════════════════════════════════════════════════════════════
#
# The declaration half asserts that a route *names* its returns; these assert
# that it can actually *make* them. That pairing is what keeps the two from
# agreeing by omission — a declared response nobody produces fails here, and a
# produced response nobody declared fails the gate.


def _services_on(repo) -> Services:
    return Services(repo, clock=_ticking_clock())


def _place_order(repo, participant_id: str, product: str = "Widget 001") -> str:
    """Give a participant a real, placed order (the §16 cart → order path)."""
    services = _services_on(repo)
    ctx = CallContext(ws=WS, actor=participant_id)
    services.add_cart_item(ctx, product_name=product, quantity=1)
    order = services.place_order(ctx)
    assert order is not None
    return order["orderId"]


def _unscope(conn, channel_id: str) -> None:
    """Strip a `Channel`'s `participantId` marker — the one graph state that
    makes a reset report `scoped=false` (§5.2's `409 unscoped_participant`).

    Done in the graph rather than with a stub repository deliberately: this is
    the branch `docs/plans/salesperson-ui-graph.md` calls unreachable on a
    healthy graph, and the assertion is worth more if the graph really is in
    that state than if a fake said so.
    """
    db.workspace_graph(conn, WS).query(
        "MATCH (c:Channel {channelId: $cid}) SET c.participantId = null",
        {"cid": channel_id},
    )


# ── GET /shop/api/health — `no graph access` ─────────────────────────────────


def test_health_reports_status_enabled_and_the_locale_list(client):
    body = client.get(f"{API_PREFIX}/health").json()
    assert body == {
        "status": "ok",
        "storefrontEnabled": True,
        "locales": list(config.STOREFRONT_LOCALES),
    }


def test_health_needs_no_credential_at_all(client):
    assert client.get(f"{API_PREFIX}/health").status_code == 200


# ── POST /shop/api/session ───────────────────────────────────────────────────


def test_join_mints_a_credential_and_returns_the_session_body(client, conn):
    body = _join(client, "Ada", "pt-BR")

    assert body["participantId"].startswith("p-")
    assert body["displayName"] == "Ada"
    assert body["language"] == "pt-BR"
    assert "Ada" in body["welcome"]
    # the token works, which is the only thing about it a client can check
    assert client.get(f"{API_PREFIX}/state", headers=_bearer(body)).status_code == 200
    # ...and only its hash reached the graph
    rows = db.workspace_graph(conn, WS).ro_query(
        "MATCH (n) WHERE any(k IN keys(n) WHERE n[k] = $v) RETURN count(n)",
        {"v": body["token"]},
    ).result_set
    assert rows[0][0] == 0


def test_join_writes_the_display_name_into_the_profile(client):
    """§4.10: the profile panel never shows an em-dash for a name the
    participant typed on the join screen."""
    session = _join(client, "Ada", "en")
    state = client.get(f"{API_PREFIX}/state", headers=_bearer(session)).json()
    assert state["profile"] == {"name": "Ada", "deliveryAddress": None}


def test_join_reports_the_first_violation_by_declaration_order(client):
    """§5.3 C11's selection rule, pinned: the **first** entry of `errors()` —
    declaration order for a single request model — so a request violating both
    bounds reports `displayName`, never `language`, and never both.

    The mutation this kills: taking `errors()[-1]`, or serialising FastAPI's
    `loc` array. Both pass a test that only asserts `status_code == 422`.

    **Both violations must be *Pydantic* violations**, and getting that wrong is
    how the first version of this test let `errors()[-1]` live: an over-long
    `displayName` with an unknown-but-well-formed `language` produces exactly
    **one** error, because the locale check is the route's own and never runs
    after the model has already refused. With one entry, `[0]` and `[-1]` are
    the same element and the rule under test is not exercised at all. Two
    length violations put two entries in the list, in declaration order.
    """
    response = client.post(
        f"{API_PREFIX}/session",
        json={"displayName": "x" * 61, "language": "y" * 33},
    )
    assert response.status_code == 422
    assert response.json() == {"error": "validation_failed", "field": "displayName"}


def test_a_configured_locale_with_no_greeting_gets_the_english_line(seeded, monkeypatch):
    """§5.2's `welcome` fallback, which had no test — M-O (`WELCOME.get(...)` →
    `WELCOME[...]`) left the whole file green.

    `WELCOME` covers exactly `config.STOREFRONT_LOCALES`'s default, so the
    fallback is unreachable through the default deployment and reachable only
    through `FALKORCHAT_STOREFRONT_LOCALES` — a real operator knob, not a
    hypothetical. Without the fallback a deployment that adds a locale answers
    `500` on the first join in it.
    """
    monkeypatch.setattr(config, "STOREFRONT_LOCALES", ("en", "de"))
    with TestClient(_build_app(seeded)) as client:
        assert client.get(f"{API_PREFIX}/health").json()["locales"] == ["en", "de"]
        body = _join(client, "Ada", "de")

    assert body["language"] == "de"
    assert body["welcome"] == storefront_api.WELCOME["en"].format(name="Ada")
    # the control: a locale that *is* in the table does not take the fallback
    assert storefront_api.WELCOME["pt-BR"] != storefront_api.WELCOME_FALLBACK


def test_a_blank_display_name_is_a_422_not_a_participant_named_nothing(client):
    """`Field(min_length=1)` accepts `"   "` (`python-web-quirks`), and a blank
    name is the most ordinary mistake on the join screen."""
    response = client.post(
        f"{API_PREFIX}/session", json={"displayName": "   ", "language": "en"}
    )
    assert response.status_code == 422
    assert response.json()["field"] == "displayName"


def test_an_unknown_language_is_a_422_naming_language(client):
    """The other §5.3 row on this route — **UI-supplied**, per C11, because the
    chooser is S12c's bundle list and a server `locales` narrower than the
    bundles makes this reachable by demo bring-up config drift."""
    response = client.post(
        f"{API_PREFIX}/session", json={"displayName": "Ada", "language": "klingon"}
    )
    assert response.status_code == 422
    assert response.json() == {"error": "validation_failed", "field": "language"}


def test_a_rejected_join_writes_nothing(client, conn):
    """The `422` is pre-write, which is what makes C11's "keep the input and
    retry" safe: a rejected join leaves no half-provisioned participant."""
    before = db.workspace_graph(conn, WS).ro_query(
        "MATCH (u:User) WHERE u.tokenHash IS NOT NULL RETURN count(u)"
    ).result_set[0][0]
    client.post(f"{API_PREFIX}/session",
                json={"displayName": "", "language": "en"})
    after = db.workspace_graph(conn, WS).ro_query(
        "MATCH (u:User) WHERE u.tokenHash IS NOT NULL RETURN count(u)"
    ).result_set[0][0]
    assert (before, after) == (0, 0)


# ── GET /shop/api/state ──────────────────────────────────────────────────────


def test_state_carries_profile_cart_order_and_turn(client, seeded):
    session = _join(client, "Ada", "en")
    _place_order(seeded, session["participantId"])

    state = client.get(f"{API_PREFIX}/state", headers=_bearer(session)).json()

    assert set(state) == {"profile", "cart", "order", "turn"}
    assert state["profile"]["name"] == "Ada"
    assert state["order"]["status"] == "placed"
    assert state["turn"] == {"state": "idle", "queuePosition": 0}


def test_state_without_a_credential_is_401(client):
    response = client.get(f"{API_PREFIX}/state")
    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"


# ── GET/POST /shop/api/messages ──────────────────────────────────────────────


def test_a_posted_message_comes_back_on_the_participants_own_thread(client):
    session = _join(client, "Ada", "en")
    headers = _bearer(session)

    posted = client.post(f"{API_PREFIX}/messages", headers=headers,
                         json={"text": "hello"})
    assert posted.status_code == 200, posted.text
    assert posted.json()["text"] == "hello"
    # every storefront post mentions the demo agent (§4.9) — an unresolvable
    # mention would have raised *before* the write, so this is also the
    # assertion that the preflight's agent check is about the right thing
    assert posted.json()["mentions"] == [AGENT]

    rows = client.get(f"{API_PREFIX}/messages", headers=headers,
                      params={"since": 0, "limit": 10}).json()
    assert [row["text"] for row in rows] == ["hello"]


def test_reading_messages_never_advances_a_cursor(client, conn):
    """This is what puts `GET /shop/api/messages` in the `reads-only` class.

    `services.read_messages` **writes** when it is called without an explicit
    `since` — it advances the member's per-thread `ReadCursor`. On a route
    polled every 2 s that would make `503 graph_read_timeout` ("nothing
    changed") a false statement, and the route would need a `504` row it does
    not have. The route therefore always passes `since`; this asserts the
    consequence in the graph, where a class error is visible.
    """
    session = _join(client, "Ada", "en")
    headers = _bearer(session)
    client.post(f"{API_PREFIX}/messages", headers=headers, json={"text": "hello"})

    for _ in range(3):
        assert client.get(f"{API_PREFIX}/messages", headers=headers).status_code == 200

    cursors = db.workspace_graph(conn, WS).ro_query(
        "MATCH (c:ReadCursor) RETURN count(c)"
    ).result_set[0][0]
    assert cursors == 0


def test_messages_since_filters_by_timestamp(client):
    session = _join(client, "Ada", "en")
    headers = _bearer(session)
    first = client.post(f"{API_PREFIX}/messages", headers=headers,
                        json={"text": "one"}).json()
    client.post(f"{API_PREFIX}/messages", headers=headers, json={"text": "two"})

    rows = client.get(f"{API_PREFIX}/messages", headers=headers,
                      params={"since": first["createdAt"]}).json()
    assert [row["text"] for row in rows] == ["two"]


def test_an_out_of_range_limit_is_a_422_naming_limit(client):
    """§5.3 C11 · UI-supplied: there is no field to blame and no retry, because
    a retry resends the same invalid value."""
    session = _join(client, "Ada", "en")
    response = client.get(f"{API_PREFIX}/messages", headers=_bearer(session),
                          params={"limit": 201})
    assert response.status_code == 422
    assert response.json() == {"error": "validation_failed", "field": "limit"}


def test_an_oversized_message_is_a_422_naming_text(client):
    session = _join(client, "Ada", "en")
    response = client.post(f"{API_PREFIX}/messages", headers=_bearer(session),
                           json={"text": "x" * 2001})
    assert response.status_code == 422
    assert response.json() == {"error": "validation_failed", "field": "text"}


def test_a_second_post_while_a_turn_is_in_flight_is_409_with_nothing_written(
    client, conn
):
    """§4.4 measure 1a: the refusal is server-side and **pre-write**.

    A written message with no reply sits in the transcript forever, and a
    second post while the first turn runs starts a *second* `WorkflowRun` on
    the same thread — `trigger.maybe_trigger` resumes only a `waiting` run.
    The zero-message assertion is the half that matters: a `409` returned
    *after* the write would pass a status-only test.
    """
    session = _join(client, "Ada", "en")
    headers = _bearer(session)
    client.app.state.storefront.set_turn_state(session["participantId"], "thinking")

    response = client.post(f"{API_PREFIX}/messages", headers=headers,
                           json={"text": "hello"})

    assert response.status_code == 409
    assert response.json()["error"] == "turn_in_progress"
    assert db.workspace_graph(conn, WS).ro_query(
        "MATCH (m:Message) RETURN count(m)"
    ).result_set[0][0] == 0
    # and the state route reports the turn the client's C6a branch keys on
    state = client.get(f"{API_PREFIX}/state", headers=headers).json()
    assert state["turn"]["state"] == "thinking"


# ── GET /shop/api/catalog ────────────────────────────────────────────────────


def test_the_catalog_lists_every_product_with_an_image_url_field(client):
    session = _join(client, "Ada", "en")
    rows = client.get(f"{API_PREFIX}/catalog", headers=_bearer(session)).json()
    assert len(rows) == 3
    assert set(rows[0]) == {"productId", "name", "category", "price", "imageUrl"}
    # no asset directory in this fixture, so every URL is `null` — the
    # text-only card variant, which §4.7 calls a legitimate deployment
    assert {row["imageUrl"] for row in rows} == {None}


def test_the_catalog_needs_a_credential(client):
    assert client.get(f"{API_PREFIX}/catalog").status_code == 401


# ── POST /shop/api/order/advance ─────────────────────────────────────────────


def test_advancing_an_order_returns_the_new_status(client, seeded):
    session = _join(client, "Ada", "en")
    order_id = _place_order(seeded, session["participantId"])

    response = client.post(f"{API_PREFIX}/order/advance", headers=_bearer(session),
                           json={"transition": "fulfill"})

    assert response.status_code == 200
    assert response.json() == {"orderId": order_id, "status": "fulfilled"}


def test_advancing_with_no_order_is_404_and_not_an_auth_failure(client):
    """§5.3 C10: an ordinary stale-button outcome. Routing this through C3
    would log a participant out for pressing a stale `cancel`."""
    session = _join(client, "Ada", "en")
    response = client.post(f"{API_PREFIX}/order/advance", headers=_bearer(session),
                           json={"transition": "cancel"})
    assert response.status_code == 404
    assert response.json()["error"] == "no_current_order"


def test_a_stale_transition_is_409_carrying_the_current_status(client, seeded):
    """The CAS guard did not match — the order already moved on. The body
    carries the current status so the client can repaint rather than guess."""
    session = _join(client, "Ada", "en")
    _place_order(seeded, session["participantId"])
    headers = _bearer(session)
    assert client.post(f"{API_PREFIX}/order/advance", headers=headers,
                       json={"transition": "cancel"}).status_code == 200

    response = client.post(f"{API_PREFIX}/order/advance", headers=headers,
                           json={"transition": "cancel"})

    assert response.status_code == 409
    assert response.json()["error"] == "order_transition_refused"
    assert response.json()["status"] == "cancelled"


def test_an_unknown_transition_is_a_422_naming_transition(client):
    session = _join(client, "Ada", "en")
    response = client.post(f"{API_PREFIX}/order/advance", headers=_bearer(session),
                           json={"transition": "teleport"})
    assert response.status_code == 422
    assert response.json() == {"error": "validation_failed", "field": "transition"}


# ── POST /shop/api/reset ─────────────────────────────────────────────────────


def test_reset_mine_keeps_the_credential_and_returns_the_language_step(
    client, seeded, conn
):
    """§4.8: the participant's identity survives, so the client returns to the
    language step rather than the join screen (C7)."""
    session = _join(client, "Ada", "pt-BR")
    headers = _bearer(session)
    client.post(f"{API_PREFIX}/messages", headers=headers, json={"text": "hello"})
    _place_order(seeded, session["participantId"])

    response = client.post(f"{API_PREFIX}/reset", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "pt-BR"
    assert body["threadId"] != f"th-{session['participantId']}"
    # the same token still works — that is the whole point of this reset
    state = client.get(f"{API_PREFIX}/state", headers=headers).json()
    # §4.8's operative post-reset fact: the name is re-written, the address is
    # not (a `None` address is what tells a re-write from a survivor)
    assert state["profile"] == {"name": "Ada", "deliveryAddress": None}
    assert state["order"] is None
    assert db.workspace_graph(conn, WS).ro_query(
        "MATCH (m:Message) RETURN count(m)"
    ).result_set[0][0] == 0


def test_reset_without_a_credential_is_401(client):
    assert client.post(f"{API_PREFIX}/reset").status_code == 401


def test_reset_of_an_unscoped_participant_is_409_and_never_a_200(client, conn):
    """§5.2: `scoped=false` means nothing was reset and nothing will be until
    the graph is repaired. **Never `200`** — C6b's alarm, dispatched on the
    body rather than on the `409` it shares with `turn_in_progress`."""
    session = _join(client, "Ada", "en")
    headers = _bearer(session)
    client.post(f"{API_PREFIX}/messages", headers=headers, json={"text": "hello"})
    _unscope(conn, f"ch-{session['participantId']}")

    response = client.post(f"{API_PREFIX}/reset", headers=headers)

    assert response.status_code == 409
    assert response.json()["error"] == "unscoped_participant"
    # a guaranteed no-op: the transcript is still there
    assert db.workspace_graph(conn, WS).ro_query(
        "MATCH (m:Message) RETURN count(m)"
    ).result_set[0][0] == 1


def test_reset_gives_up_on_a_turn_that_never_finishes_and_changes_nothing(
    client, conn
):
    """The quiesce `503` (§4.8) — the **only** reset failure that means
    "nothing changed"; a FalkorDB socket timeout is the `504` below, and
    conflating the two is the F8 defect this pair exists to prevent."""
    session = _join(client, "Ada", "en")
    headers = _bearer(session)
    client.post(f"{API_PREFIX}/messages", headers=headers, json={"text": "hello"})
    client.app.state.storefront.set_turn_state(session["participantId"], "thinking")

    response = client.post(f"{API_PREFIX}/reset", headers=headers)

    assert response.status_code == 503
    assert response.json()["error"] == "quiesce_timeout"
    assert db.workspace_graph(conn, WS).ro_query(
        "MATCH (m:Message) RETURN count(m)"
    ).result_set[0][0] == 1


# ── the presenter surface ────────────────────────────────────────────────────


def test_the_presenter_key_is_exchanged_for_a_token(client):
    response = client.post(f"{API_PREFIX}/presenter/session",
                           json={"key": PRESENTER_KEY})
    assert response.status_code == 200
    assert response.json()["token"]


def test_a_wrong_presenter_key_is_403(client):
    response = client.post(f"{API_PREFIX}/presenter/session", json={"key": "nope"})
    assert response.status_code == 403
    assert response.json()["error"] == "bad_presenter_key"


def test_a_blank_presenter_key_is_a_422_naming_key(client):
    """§5.3 C11 files this as **user-supplied** — a human pressing Enter on an
    empty box, the most ordinary mistake in the presenter flow."""
    response = client.post(f"{API_PREFIX}/presenter/session", json={"key": "  "})
    assert response.status_code == 422
    assert response.json() == {"error": "validation_failed", "field": "key"}


class _CompareDigestSpy:
    """Stands in for the `hmac` module inside `storefront_api`, recording every
    `compare_digest` call and forwarding to the real one.

    A spy at the **comparison seam** rather than an assertion on the status
    code, because the status code cannot tell the two implementations apart:
    with no key configured, `compare_digest("", "anything")` is already `False`,
    so a login that skipped the `presenter_configured` guard entirely still
    answers `403` and a status-only test stays green. What separates them is
    whether the comparison happens at all.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def compare_digest(self, left, right):
        self.calls.append((left, right))
        return hmac.compare_digest(left, right)


def test_an_unset_presenter_key_never_reaches_the_comparison(seeded, monkeypatch):
    """`hmac.compare_digest("", "")` is `True`, so a deployment with no key
    configured would otherwise hand the reset-everyone button to whoever posts
    an empty key first. `presenter_configured` is checked **before** any
    comparison, and this asserts that order directly.

    The empty-key request cannot prove it on its own: the model's `422` refuses
    a blank key before the route runs, so the guard's own scenario is
    unreachable through the wire. That makes the guard defence-in-depth rather
    than dead — the thing standing behind it is a Pydantic bound one edit away
    from being relaxed — which is exactly why the order is pinned at the seam.
    """
    monkeypatch.setattr(config, "STOREFRONT_PRESENTER_KEY", "")
    spy = _CompareDigestSpy()
    monkeypatch.setattr(storefront_api, "hmac", spy)

    with TestClient(_build_app(seeded)) as unkeyed:
        # a blank key is refused by the model before it reaches the route
        assert unkeyed.post(f"{API_PREFIX}/presenter/session",
                            json={"key": " "}).status_code == 422
        response = unkeyed.post(f"{API_PREFIX}/presenter/session",
                                json={"key": "anything"})

    assert response.status_code == 403
    assert response.json()["error"] == "bad_presenter_key"
    assert spy.calls == []


def test_a_configured_presenter_key_is_compared_once_in_constant_time(
    seeded, monkeypatch
):
    """The positive control for the assertion above — without it, a login that
    never compared anything at all would pass it — and S6's constant-time
    tripwire extended to the second `compare_digest` site in this codebase."""
    spy = _CompareDigestSpy()
    monkeypatch.setattr(storefront_api, "hmac", spy)

    with TestClient(_build_app(seeded)) as client:
        assert client.post(f"{API_PREFIX}/presenter/session",
                           json={"key": PRESENTER_KEY}).status_code == 200

    assert spy.calls == [(PRESENTER_KEY, PRESENTER_KEY)]


def test_the_roster_carries_exactly_the_four_keys_and_no_non_participants(client):
    """§5.2's four keys and nothing more: `channelId`/`threadId` are server-side
    ids no client needs, and the lifespan's own `config.USER_ID` node carries no
    `tokenHash`, so it never appears."""
    ada = _join(client, "Ada", "en")
    bob = _join(client, "Bob", "es")

    rows = client.get(f"{API_PREFIX}/presenter/participants",
                      headers=_presenter(client)).json()

    assert [row["displayName"] for row in rows] == ["Ada", "Bob"]
    assert {row["participantId"] for row in rows} == {
        ada["participantId"], bob["participantId"]
    }
    for row in rows:
        assert set(row) == {"participantId", "displayName", "language", "joinedAt"}


def test_reset_everyone_invalidates_participant_tokens_but_not_the_presenters(
    client, conn
):
    """§4.8's asymmetry, and §5.3 C3/C5's whole basis: the presenter keeps
    driving the demo through the sweep while every participant is bounced."""
    ada = _join(client, "Ada", "en")
    _join(client, "Bob", "es")
    presenter = _presenter(client)
    client.post(f"{API_PREFIX}/messages", headers=_bearer(ada), json={"text": "hi"})

    response = client.post(f"{API_PREFIX}/presenter/reset-all", headers=presenter)

    assert response.status_code == 200
    assert response.json() == {"clearedParticipants": 2}
    # `unscopedCount == 0` returns **no `incomplete` field at all** (§5.2)
    assert "incomplete" not in response.json()
    # the participant credential is dead...
    assert client.get(f"{API_PREFIX}/state", headers=_bearer(ada)).status_code == 401
    # ...and the presenter's is not
    assert client.get(f"{API_PREFIX}/presenter/participants",
                      headers=presenter).json() == []
    assert db.workspace_graph(conn, WS).ro_query(
        "MATCH (m:Message) RETURN count(m)"
    ).result_set[0][0] == 0


def test_reset_everyone_reports_incomplete_when_a_participant_is_unresolvable(
    client, conn
):
    """Not an error — the sweep did everything it could — but it must not read
    as clean (§5.2)."""
    ada = _join(client, "Ada", "en")
    _join(client, "Bob", "es")
    _unscope(conn, f"ch-{ada['participantId']}")

    body = client.post(f"{API_PREFIX}/presenter/reset-all",
                       headers=_presenter(client)).json()

    assert body["incomplete"] is True
    assert body["unresolved"] == [ada["participantId"]]


def test_reset_everyone_gives_up_on_a_turn_that_never_finishes(client, conn):
    ada = _join(client, "Ada", "en")
    presenter = _presenter(client)
    client.post(f"{API_PREFIX}/messages", headers=_bearer(ada), json={"text": "hi"})
    client.app.state.storefront.set_turn_state(ada["participantId"], "thinking")

    response = client.post(f"{API_PREFIX}/presenter/reset-all", headers=presenter)

    assert response.status_code == 503
    assert response.json()["error"] == "quiesce_timeout"
    assert db.workspace_graph(conn, WS).ro_query(
        "MATCH (u:User) WHERE u.tokenHash IS NOT NULL RETURN count(u)"
    ).result_set[0][0] == 1


# ═══════════════════════════════════════════════════════════════════════════
# The auth matrix and the cross-participant probe (§6.2)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("route", sorted(ROUTE_CLASSES))
def test_auth_matrix_no_credential(client, route):
    """Every participant and presenter route refuses an unauthenticated call;
    the three open routes answer normally."""
    response = _call(client, *route)
    if route in OPEN_ROUTES:
        assert response.status_code not in (401, 403), response.text
    else:
        assert response.status_code == 401, response.text


@pytest.mark.parametrize("route", sorted(ROUTE_CLASSES))
def test_auth_matrix_participant_credential(client, route):
    """A participant token is accepted on its own routes and **refused on every
    presenter route** — `403`, because the credential type is wrong rather than
    the session gone (§5.3 C2 keeps the two apart)."""
    headers = _bearer(_join(client, "Ada", "en"))
    response = _call(client, *route, headers=headers)
    if route in PRESENTER_ROUTES:
        assert response.status_code == 403, response.text
        assert response.json()["error"] == "wrong_credential_type"
    else:
        assert response.status_code not in (401, 403), response.text


@pytest.mark.parametrize("route", sorted(ROUTE_CLASSES))
def test_auth_matrix_presenter_credential(client, route):
    """And the other way round. A presenter token parses as participant id
    `"presenter"`, which no `User` carries, so participant routes answer the
    same `401` as any other bad credential — structurally, not by a special
    case that a future edit could drop."""
    headers = _presenter(client)
    response = _call(client, *route, headers=headers)
    if route in PARTICIPANT_ROUTES:
        assert response.status_code == 401, response.text
        assert response.json()["error"] == "invalid_token"
    else:
        assert response.status_code not in (401, 403), response.text


def test_a_participant_token_reaches_no_other_participants_data(client, seeded):
    """The cross-participant probe (§6.2): with A holding cart items, messages
    and an order, **every** route called with B's token returns only B's data.

    Isolation here is structural rather than filtered — `customerId ==
    participantId` and every id is resolved from the token — so the value of
    this test is that it walks the whole surface rather than one route.
    """
    ada = _join(client, "Ada", "en")
    bob = _join(client, "Bob", "es")
    ada_h, bob_h = _bearer(ada), _bearer(bob)

    client.post(f"{API_PREFIX}/messages", headers=ada_h, json={"text": "ada speaks"})
    _place_order(seeded, ada["participantId"])
    services = _services_on(seeded)
    ada_ctx = CallContext(ws=WS, actor=ada["participantId"])
    services.add_cart_item(ada_ctx, product_name="Widget 002", quantity=3)

    state = client.get(f"{API_PREFIX}/state", headers=bob_h).json()
    assert state["profile"]["name"] == "Bob"
    assert state["cart"]["items"] == []
    assert state["order"] is None

    assert client.get(f"{API_PREFIX}/messages", headers=bob_h).json() == []

    # B cannot touch A's order, and the refusal is the same `404` a participant
    # with no order at all gets — "unknown" and "someone else's" are one answer
    # by construction (§4.6), so nothing leaks by status code either.
    advance = client.post(f"{API_PREFIX}/order/advance", headers=bob_h,
                          json={"transition": "cancel"})
    assert advance.status_code == 404

    # B's reset leaves A's transcript, cart and order untouched
    assert client.post(f"{API_PREFIX}/reset", headers=bob_h).status_code == 200
    ada_state = client.get(f"{API_PREFIX}/state", headers=ada_h).json()
    assert ada_state["order"]["status"] == "placed"
    assert len(ada_state["cart"]["items"]) == 1
    assert [m["text"] for m in
            client.get(f"{API_PREFIX}/messages", headers=ada_h).json()] == ["ada speaks"]

    # the catalog is global `reference` data and is deliberately identical
    assert (client.get(f"{API_PREFIX}/catalog", headers=bob_h).json()
            == client.get(f"{API_PREFIX}/catalog", headers=ada_h).json())


# ═══════════════════════════════════════════════════════════════════════════
# The error map, asserted by execution
# ═══════════════════════════════════════════════════════════════════════════


class _RaisingRepo:
    """A `Repository` stand-in that raises from **every** call, recording the
    method names it was asked for.

    The recording half is what turns "the two `no graph access` routes still
    answer normally" from a positive claim into a negative one: they do not
    merely survive a broken graph, they never ask it anything.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.calls: list[str] = []

    def __getattr__(self, name):
        def call(*_args, **_kwargs):
            self.calls.append(name)
            raise self._exc

        return call


class _FailingMethodRepo:
    """A real `Repository` with exactly one method replaced by a raise.

    Counts the calls, because §4.8's premise — *nothing beneath the browser
    retries a reset* — is only enforceable at this layer if the application
    layer is known not to re-issue. (The library layer is beyond this test's
    reach by construction, which is why the plan writes the premise down rather
    than only asserting it.)
    """

    def __init__(self, inner, method: str, exc: BaseException) -> None:
        self._inner = inner
        self._method = method
        self._exc = exc
        self.calls = 0

    def __getattr__(self, name):
        if name == self._method:
            def call(*_args, **_kwargs):
                self.calls += 1
                raise self._exc

            return call
        return getattr(self._inner, name)


class _FailingAfterNCalls:
    """A real repository whose named method raises only **after** `after` calls.

    `_FailingMethodRepo` fails one method for every call, which cannot reach a
    branch that is itself a *retry* of that method: failing
    `list_participants` there breaks reset-all's pre-drain roster read and the
    request never gets as far as the re-read
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 10`, P10-4 — the third time
    in this coordination that an over-general stub silently skipped the rule its
    test named).
    """

    def __init__(self, inner, method: str, exc: BaseException, *, after: int) -> None:
        self._inner, self._method, self._exc, self._after = inner, method, exc, after
        self.calls = 0

    def __getattr__(self, name):
        if name == self._method:
            def call(*args, **kwargs):
                self.calls += 1
                if self.calls > self._after:
                    raise self._exc
                return getattr(self._inner, self._method)(*args, **kwargs)

            return call
        return getattr(self._inner, name)


def _broken_client(exc: BaseException, storefront_config):
    """A storefront over a repository that raises `exc` from everything.

    Built **without** entering the lifespan: `services.ensure_actor` is the
    first thing it does and would raise, and the point here is the request
    path, not startup. `raise_server_exceptions=False` so an unmapped error
    surfaces as the `500` a real client would see rather than as a test error —
    which is what lets "no route anywhere answers a bare `500`" be an assertion.
    """
    repo = _RaisingRepo(exc)
    app = create_app(
        Services(repo), context_provider=CTX, mount_mcp=False,
        dev_surface=False, storefront=True,
    )
    return repo, TestClient(app, raise_server_exceptions=False)


# What each route answers when every graph call raises. Written as literals
# rather than derived from `cross_cutting_response`, deliberately: the gate
# already reads that seam, so a test that read it too would agree with a broken
# seam. These are the answers §5.3's table asks for, spelled out.
#
# **Why a writing route answers `504` even when the failing query was a read
# that ran before the write** (P10-9). Under `_RaisingRepo` *every* call raises,
# so the one that actually fails on a writing route is usually the earliest —
# `get_participant`'s `resolve_token`, or reset-all's pre-drain roster read —
# at which point nothing was attempted and "nothing changed" would be the
# stronger, truer report. `504 <op>_state_unknown` is nonetheless right, and
# right by §5.3's own class map rather than by accident: the class is a
# property of the **route**, not of which query inside it failed, and C4's
# action for a `504` is a safe re-read whose answer on this branch is simply
# "unchanged". Getting it the other way round is the one that harms — a write
# that did commit reported as *nothing changed* is F8's defect exactly. The
# conservative direction is the deliberate one.
_UNAVAILABLE = {route: (503, "graph_unavailable") for route in ROUTE_CLASSES}
_UNAVAILABLE[("GET", f"{API_PREFIX}/health")] = (200, None)
_UNAVAILABLE[("POST", f"{API_PREFIX}/presenter/session")] = (200, None)

_TIMEOUT = {
    ("GET", f"{API_PREFIX}/health"): (200, None),
    ("POST", f"{API_PREFIX}/presenter/session"): (200, None),
    ("POST", f"{API_PREFIX}/session"): (504, "join_state_unknown"),
    ("GET", f"{API_PREFIX}/state"): (503, "graph_read_timeout"),
    ("GET", f"{API_PREFIX}/messages"): (503, "graph_read_timeout"),
    ("POST", f"{API_PREFIX}/messages"): (504, "post_state_unknown"),
    ("GET", f"{API_PREFIX}/catalog"): (503, "graph_read_timeout"),
    ("POST", f"{API_PREFIX}/order/advance"): (504, "order_state_unknown"),
    ("POST", f"{API_PREFIX}/reset"): (504, "reset_state_unknown"),
    ("GET", f"{API_PREFIX}/presenter/participants"): (503, "graph_read_timeout"),
    ("POST", f"{API_PREFIX}/presenter/reset-all"): (504, "reset_state_unknown"),
}


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (redis_exceptions.ConnectionError("refused"), _UNAVAILABLE),
        (db.FalkorDBUnreachableError("unreachable"), _UNAVAILABLE),
        (redis_exceptions.TimeoutError("timed out"), _TIMEOUT),
    ],
    ids=["redis-ConnectionError", "FalkorDBUnreachableError", "redis-TimeoutError"],
)
def test_every_route_answers_its_own_cross_cutting_response_and_never_a_bare_500(
    storefront_config, exc, expected
):
    """The whole error map, **asserted by execution rather than by inspection**.

    Today `_register_error_handlers` maps `ServiceError` and the workflow errors
    only, and `FalkorDBUnreachableError` has no handler at all — so before S8 a
    query-time `redis.TimeoutError` escaped as a bare `500` on `/state` and
    `/messages`, for every polling participant at once, in exactly the scenario
    §4.8's F8 exists for. This sweeps all eleven routes rather than the three
    the plan names, because a map that is total *by type* should have nothing
    to hide on the other eight.
    """
    repo, client = _broken_client(exc, storefront_config)
    presenter = _presenter(client)
    participant = {"Authorization": "Bearer p-nobody.some-token"}

    for route in sorted(ROUTE_CLASSES):
        headers = presenter if route in PRESENTER_ROUTES else participant
        response = _call(client, *route, headers=headers)
        status, token = expected[route]
        assert response.status_code == status, f"{route} -> {response.text}"
        assert response.status_code != 500, f"{route} answered a bare 500"
        if token is not None:
            assert response.json()["error"] == token, f"{route} -> {response.text}"


def test_a_typed_handler_on_an_unclassified_route_is_loud_and_conservative():
    """The `no graph access` / unclassified branch of `_cross_cutting_json`.

    Unreachable by construction — the route-table assertion keeps
    `ROUTE_CLASSES` exactly the registered set — but it is reached *from inside
    an exception handler*, where the delivered code raised `KeyError` on an
    unclassified path and Starlette turned that into a `500` with none of the
    log line the branch exists for (P10-8). `cross_cutting_response` still
    raises, deliberately: the gate reads that seam and the `KeyError` is the
    route-table assertion firing from the inside.
    """
    with pytest.raises(KeyError):
        cross_cutting_response("graph_timeout", "GET", f"{API_PREFIX}/nowhere")

    class _Url:
        path = f"{API_PREFIX}/nowhere"

    class _Request:
        method = "GET"
        url = _Url()

    response = storefront_api._cross_cutting_json(_Request(), "graph_timeout")
    assert response.status_code == 504
    assert b"state_unknown" in response.body

    # and the classified `no graph access` route takes the same branch, which is
    # the case that is *supposed* to be unreachable rather than merely absent
    class _Health(_Request):
        url = type("U", (), {"path": f"{API_PREFIX}/health"})()

    assert storefront_api._cross_cutting_json(_Health(), "graph_timeout").status_code == 504


def test_presenter_token_verification_compares_every_candidate_in_constant_time(
    monkeypatch,
):
    """M-P: replacing the `compare_digest` loop with `token in candidates` left
    the file green, so nothing pinned the property the docstring claims.

    What is pinned is what is true — **per comparison**, not per call: every
    candidate token reaches `hmac.compare_digest`, and a wrong token that shares
    a long prefix with a live one is rejected by the same call the right one
    would be accepted by. `any()` short-circuits, so a valid token costs fewer
    comparisons; that is stated in the docstring rather than asserted away.
    """
    sessions = storefront_api._PresenterSessions()
    minted = sessions.mint()
    seen: list[tuple[str, str]] = []

    class _Hmac:
        @staticmethod
        def compare_digest(a, b):  # noqa: ANN001
            seen.append((a, b))
            return hmac.compare_digest(a, b)

    monkeypatch.setattr(storefront_api, "hmac", _Hmac)

    assert sessions.verify(minted) is True
    assert seen == [(minted, minted)]

    seen.clear()
    near_miss = minted[:-1] + ("A" if minted[-1] != "A" else "B")
    assert sessions.verify(near_miss) is False
    # the near miss went through the same comparison, not a membership test
    assert seen == [(minted, near_miss)]


def test_the_two_no_graph_routes_issue_no_query_at_all(storefront_config):
    """§5.3's classification asserted **negatively**, which is the only way it
    can be asserted at all: with the repository raising on *any* call, both
    routes still answer their normal `200`/`403`/`422` — and the call log is
    empty, so this is "no query was issued", not "the query happened to work"."""
    repo, client = _broken_client(redis_exceptions.TimeoutError("nope"), storefront_config)

    assert client.get(f"{API_PREFIX}/health").status_code == 200
    assert client.post(f"{API_PREFIX}/presenter/session",
                       json={"key": PRESENTER_KEY}).status_code == 200
    assert client.post(f"{API_PREFIX}/presenter/session",
                       json={"key": "wrong"}).status_code == 403
    assert client.post(f"{API_PREFIX}/presenter/session",
                       json={"key": " "}).status_code == 422

    assert repo.calls == []

    # The positive control: the same repository, one route class over, does
    # reach the graph — without it an always-empty call log would pass.
    client.get(f"{API_PREFIX}/state", headers={"Authorization": "Bearer p-x.y"})
    assert repo.calls == ["get_participant_record"]


def test_a_reset_that_times_out_is_504_unknown_and_is_never_retried(seeded, conn):
    """§4.8 F8: the delete may have committed, so the participant-facing meaning
    is *unknown* — never the quiesce `503`, and never "nothing changed".

    The call count is the other half: the **application** layer does not retry.
    A retry here would re-issue the same `$newThreadId` and surface as a
    `Thread` UNIQUE violation — this plan's own "the graph needs repair" signal
    — raised by a benign, already-committed reset.
    """
    repo = _FailingMethodRepo(
        seeded, "reset_participant", redis_exceptions.TimeoutError("timed out")
    )
    with TestClient(_build_app(repo)) as client:
        session = _join(client, "Ada", "en")
        response = client.post(f"{API_PREFIX}/reset", headers=_bearer(session))

    assert response.status_code == 504
    body = response.json()
    assert body["error"] == "reset_state_unknown"
    # the courtesy re-read succeeded here, so the response reports what the
    # graph actually holds rather than claiming nothing changed
    assert body["state"]["profile"]["name"] == "Ada"
    assert repo.calls == 1


def test_a_reset_all_that_times_out_is_504_unknown_and_is_never_retried(seeded):
    """The same contract on the presenter's sweep, whose C4 re-read is the
    **roster** rather than `/state`: if the delete committed, the participant
    credential is already dead, so `/state` would answer `401` rather than
    state, while the roster is what the surviving presenter credential can
    still reach — and is the thing that actually says whether the sweep
    happened."""
    repo = _FailingMethodRepo(
        seeded, "reset_all_participants", redis_exceptions.TimeoutError("timed out")
    )
    with TestClient(_build_app(repo)) as client:
        _join(client, "Ada", "en")
        response = client.post(f"{API_PREFIX}/presenter/reset-all",
                               headers=_presenter(client))

    assert response.status_code == 504
    body = response.json()
    assert body["error"] == "reset_state_unknown"
    assert [row["displayName"] for row in body["participants"]] == ["Ada"]
    assert repo.calls == 1


def test_a_reset_all_whose_re_read_also_times_out_is_still_504_with_no_roster(
    seeded,
):
    """§4.8 F8's **second** ordering, which S8 shipped as code with no test.

    The re-read is another query against the same graph, and the stalled write
    that produced the first timeout is precisely what stalls it — so a second
    `TimeoutError` is the *likelier* second fault, not the exotic one. It must
    still answer `504`: simply with no roster. A `500` here would report a
    *possibly committed* sweep as a server fault, which is the exact
    misattribution F8 exists to prevent.

    **`list_participants` fails only on its second call**, because reset-all
    reads the roster once before draining and once as the re-read; a stub that
    failed it from the first call would break the pre-drain read and never reach
    this branch at all (P10-4).
    """
    timeout = redis_exceptions.TimeoutError("timed out")
    roster = _FailingAfterNCalls(seeded, "list_participants", timeout, after=1)
    repo = _FailingMethodRepo(roster, "reset_all_participants", timeout)
    with TestClient(_build_app(repo)) as client:
        _join(client, "Ada", "en")
        response = client.post(
            f"{API_PREFIX}/presenter/reset-all", headers=_presenter(client)
        )

    assert response.status_code == 504, response.text
    body = response.json()
    assert body["error"] == "reset_state_unknown"
    # the state block is a courtesy the response carries when it can, never the
    # contract — so it is present and empty, not absent-and-therefore-clean
    assert body["participants"] is None
    # and the application layer still did not retry either query
    assert repo.calls == 1
    assert roster.calls == 2


@pytest.mark.parametrize(
    ("method_name", "path"),
    [
        ("reset_participant", f"{API_PREFIX}/reset"),
        ("reset_all_participants", f"{API_PREFIX}/presenter/reset-all"),
    ],
)
def test_an_unmapped_graph_error_propagates_as_5xx_and_is_never_retried(
    seeded, method_name, path
):
    """§5.2: a `Thread` UNIQUE violation propagates as `5xx` and is **never
    retried** — a retry re-raises forever and the graph needs repair.

    Asserted on both reset routes because §5.3 gives both a `5xx` row, even
    though only reset-mine re-mints a thread and can therefore actually raise
    the UNIQUE violation; the row on reset-all is about any unmapped graph
    error, which is what this drives.
    """
    repo = _FailingMethodRepo(
        seeded, method_name,
        redis_exceptions.ResponseError(
            "unique constraint violation on node of type Thread"
        ),
    )
    app = _build_app(repo)
    with TestClient(app, raise_server_exceptions=False) as client:
        session = _join(client, "Ada", "en")
        headers = (
            _presenter(client) if "presenter" in path else _bearer(session)
        )
        response = client.post(path, headers=headers)

    assert response.status_code == 500
    assert repo.calls == 1


# ═══════════════════════════════════════════════════════════════════════════
# The service layer's own refusals (P10-1) and the two family guards
# ═══════════════════════════════════════════════════════════════════════════
#
# `create_app` builds one app, so `app.py`'s `ServiceError` handler is on the
# storefront deployment — and the storefront calls the same `services` layer
# the legacy surface does. Left alone it answers `{"error": "<Python class
# name>"}`, which is not a contract: §5.3's rules dispatch on a plan token.
# Everything below is about that seam.


def _delete_demo_agent(conn) -> None:
    """Delete the demo `Agent` out of band, **after** the preflight has passed.

    §4.9's preflight is a boot-time check, not an invariant — which is the whole
    of P10-2: S8 argued `DemoNotSeededError` unreachable *because* the preflight
    asks the identical question, and the argument holds only for the boot-time
    snapshot. Done in the graph rather than with a stub for the same reason
    `_unscope` is: the state is real, an operator can produce it, and the demo
    `Agent` is not something the storefront itself can protect.
    """
    db.workspace_graph(conn, WS).query(
        "MATCH (a:Agent {agentId: $id}) DETACH DELETE a", {"id": AGENT}
    )


def _delete_thread(conn, participant_id: str) -> None:
    """Delete a participant's `Thread` while their `User` survives — the reset
    window `Storefront._await_quiesce`'s docstring names, and the state that
    produced P10-1's `404 {"error":"ThreadNotFoundError"}`.

    `resolve_token` still resolves (it reads the `User`), so the request reaches
    the route and `services._validate_and_derive_role` is the first thing to
    notice. Real graph state again, not a stub."""
    db.workspace_graph(conn, WS).query(
        "MATCH (t:Thread {threadId: $id}) DETACH DELETE t",
        {"id": f"th-{participant_id}"},
    )


def test_join_with_the_demo_agent_gone_is_503_and_never_a_bare_500(client, conn):
    """P10-2, reproduced and closed. Before this, the same request answered
    `500 Internal Server Error` — in plain text, not even JSON — because
    `DemoNotSeededError` was the one `StorefrontError` of seven with no mapping,
    against S8's own done-condition that *no route anywhere answers a bare
    `500`*."""
    _delete_demo_agent(conn)

    response = client.post(
        f"{API_PREFIX}/session", json={"displayName": "Ada", "language": "en"}
    )

    assert response.status_code == 503, response.text
    assert response.json()["error"] == "demo_not_seeded"
    # C9's "nothing changed" is the reason it is a `503` and not a `504`:
    # `ensure_participant` reports `agentMissing` having written nothing
    assert "seed_demo.sh" in response.json()["detail"]


def test_a_post_into_a_swept_thread_is_401_not_a_python_class_name(client, conn):
    """**The response that started P10-1.**

    A participant whose `Thread` was swept out from under them posts. The
    delivered app answered `404 {"error":"ThreadNotFoundError"}` — undeclared,
    untabled, and invisible to both halves of the gate *and* to the AST refusal
    check, because it came from an app-wide handler the gate subtracted.

    `401 invalid_token` is the honest answer and not merely a tabled one: their
    credential names nothing live, and `resolve_token` — which re-reads the
    graph on every call — answers `401` on their very next request anyway. So
    this converges the race with the steady state rather than inventing a third
    outcome, and C3's action (clear the credential, rejoin) is right for both.
    """
    session = _join(client, "Ada", "en")
    _delete_thread(conn, session["participantId"])

    response = client.post(
        f"{API_PREFIX}/messages", headers=_bearer(session), json={"text": "hello"}
    )

    assert response.status_code == 401, response.text
    body = response.json()
    assert body["error"] == "invalid_token"
    assert "ThreadNotFoundError" not in response.text


def test_a_post_with_the_demo_agent_gone_is_503_demo_not_seeded(client, conn):
    """The same operator error as join's, one route over: every storefront post
    carries `mentions=[agent_id]`, and `_validate_and_derive_role` raises
    `UnknownMemberError` **before any write** — so it is the same token and the
    same C9 rule, not a new one.

    This is the `(route, response)` pair S8b adds that §5.3 does not yet
    carry."""
    session = _join(client, "Ada", "en")
    _delete_demo_agent(conn)

    response = client.post(
        f"{API_PREFIX}/messages", headers=_bearer(session), json={"text": "hello"}
    )

    assert response.status_code == 503, response.text
    assert response.json()["error"] == "demo_not_seeded"
    assert "UnknownMemberError" not in response.text


class _PatchedMethodRepo:
    """A real repository with one method answering a fixed value.

    The narrow sibling of `_FailingMethodRepo`: some declared rows are reached
    not by a method *raising* but by one returning the graph's own "no rows"
    answer, which a healthy graph will not produce on demand.
    """

    def __init__(self, inner, method: str, result) -> None:  # noqa: ANN001
        self._inner, self._method, self._result = inner, method, result
        self.calls = 0

    def __getattr__(self, name):
        if name == self._method:
            def call(*_args, **_kwargs):
                self.calls += 1
                return self._result

            return call
        return getattr(self._inner, name)


def test_resetting_a_participant_the_graph_no_longer_has_is_404(seeded):
    """P10-5 / M-D: `(404, unknown_participant)` is declared on this route and
    sits in §5.3's table, and **nothing produced it** — deleting the route's
    `except UnknownParticipantError` left all 99 tests green.

    The zero-row contract is `repository.reset_participant` returning `None`
    (graph note §12's anomaly contract), which is what this pins. It is
    indistinguishable from an already-deleted participant, by design — which is
    why C3 routes it the same way as the `401`."""
    repo = _PatchedMethodRepo(seeded, "reset_participant", None)
    with TestClient(_build_app(repo)) as client:
        session = _join(client, "Ada", "en")
        response = client.post(f"{API_PREFIX}/reset", headers=_bearer(session))

    assert response.status_code == 404, response.text
    assert response.json()["error"] == "unknown_participant"
    assert repo.calls == 1


def test_an_order_that_stops_being_theirs_mid_transition_is_404(seeded):
    """P10-5 / M-V: `advance_order`'s `except UnknownOrderError` had no producer
    either — and that escape is a `StorefrontError`, so deleting the branch
    turned it into a **bare `500`**, P10-2's family one route over.

    The state is a race the plan names: the participant held an order when
    `get_current_order` answered, and `order_belongs_to_customer` says it is not
    theirs by the time the CAS is attempted — a reset or a racing sweep landed
    in between. §4.6 makes that indistinguishable from "no order of theirs", so
    both are the same `404` (C10: an ordinary stale button, never an auth
    failure)."""
    repo = _PatchedMethodRepo(
        seeded, "order_belongs_to_customer", {"owned": False, "status": None}
    )
    with TestClient(_build_app(repo)) as client:
        session = _join(client, "Ada", "en")
        _place_order(seeded, session["participantId"])
        response = client.post(
            f"{API_PREFIX}/order/advance",
            headers=_bearer(session), json={"transition": "fulfill"},
        )

    assert response.status_code == 404, response.text
    assert response.json()["error"] == "no_current_order"
    # the branch under test is the `except`, not the `current is None` guard —
    # so the ownership check must actually have been reached
    assert repo.calls == 1


def test_the_service_error_map_is_read_off_the_one_seam():
    """`service_error_response` is the seam the live handler and the gate share,
    in the same sense `cross_cutting_response` is — asserted directly because
    `UnknownActorError` has no graph state that produces it without also
    failing `resolve_token` first, so it is unreachable through the wire."""
    assert service_error_response(storefront_api.ThreadNotFoundError("t")) == (
        401, "invalid_token",
    )
    assert service_error_response(storefront_api.UnknownActorError("a")) == (
        401, "invalid_token",
    )
    assert service_error_response(storefront_api.UnknownMemberError(["x"])) == (
        503, "demo_not_seeded",
    )
    # ...and a subclass nobody mapped gets no invented answer
    assert service_error_response(storefront_api.MatchNotFoundError("m")) is None


class _ArmedRepo:
    """A real repository whose service-layer pre-write check fails **once armed**.

    Armed after startup deliberately: the preflight uses the same
    `resolve_member_kinds` lookup, so a repo that failed from construction would
    never get past the lifespan and the sweep would be measuring startup rather
    than the request path.
    """

    def __init__(self, inner, mode: str) -> None:
        self._inner, self._mode, self.armed = inner, mode, False

    def __getattr__(self, name):
        if self.armed and self._mode == "thread" and name == "thread_exists":
            return lambda *_a, **_k: False
        if self.armed and self._mode == "actor" and name == "resolve_member_kinds":
            return lambda *_a, **_k: {}
        if self.armed and self._mode == "member" and name == "resolve_member_kinds":
            inner = self._inner.resolve_member_kinds
            return lambda ws, *, ids: {
                key: kind for key, kind in inner(ws, ids=ids).items() if key != AGENT
            }
        return getattr(self._inner, name)


# What every route answers when the service layer refuses. Literals, for the
# same reason `_UNAVAILABLE`/`_TIMEOUT` are: this file's expectations must not
# be readable off the seam the gate reads, or a broken seam would satisfy both.
_SERVICE_HEALTHY: dict[tuple[str, str], tuple[int, str]] = {
    route: (200, "ok") for route in ROUTE_CLASSES
}
_SERVICE_HEALTHY[("POST", f"{API_PREFIX}/order/advance")] = (404, "no_current_order")

_SERVICE_REFUSED = {
    "thread": {**_SERVICE_HEALTHY,
               ("POST", f"{API_PREFIX}/messages"): (401, "invalid_token")},
    "actor": {**_SERVICE_HEALTHY,
              ("POST", f"{API_PREFIX}/messages"): (401, "invalid_token")},
    "member": {**_SERVICE_HEALTHY,
               ("POST", f"{API_PREFIX}/messages"): (503, "demo_not_seeded")},
}


@pytest.mark.parametrize("mode", ["thread", "actor", "member"])
def test_only_post_messages_can_raise_a_service_error(seeded, mode):
    """**`SERVICE_ERROR_ROUTES` is this measurement, not a hand-list.**

    Each of the three faults the storefront's service calls can hit is armed in
    turn and all eleven routes are driven. Exactly one route's answer moves —
    `POST /shop/api/messages`, the only route whose call reaches
    `services._validate_and_derive_role`; every other route's service calls are
    thin reads and writes over the repository, which raises no `ServiceError`.

    The sweep is also the standing guard on P10-1: **every** response it sees
    must be a row of §5.3's table, so an escape to `app.py`'s inherited handler
    shows up as a Python class name where a plan token belongs — on any route,
    including one added later.
    """
    repo = _ArmedRepo(seeded, mode)
    moved: set[tuple[str, str]] = set()
    with TestClient(_build_app(repo), raise_server_exceptions=False) as client:
        for route in sorted(ROUTE_CLASSES):
            session = _join(client, "Ada", "en")
            headers = _presenter(client) if route in PRESENTER_ROUTES else _bearer(session)
            repo.armed = True
            try:
                response = _call(client, *route, headers=headers)
            finally:
                repo.armed = False

            token = _observed_token(response)
            assert (response.status_code, token) == _SERVICE_REFUSED[mode][route], (
                f"{route} -> {response.status_code} {response.text}"
            )
            assert (response.status_code, token) in TABLE[route], (
                f"{route} answered {(response.status_code, token)}, which §5.3's "
                "completeness table does not carry — an unruled (route, "
                "response) arriving from the server"
            )
            if (response.status_code, token) != _SERVICE_HEALTHY[route]:
                moved.add(route)

    assert moved == set(SERVICE_ERROR_ROUTES)


def _subclasses(root: type) -> set[type]:
    found = set()
    for sub in root.__subclasses__():
        found.add(sub)
        found |= _subclasses(sub)
    return found


def _caught_names(source: str) -> set[str]:
    """Every exception name an `except` clause in `source` catches.

    Parsed, not grepped, for the reason the `.lookup(` tripwire is: prose that
    quotes a name even to disown it is invisible to an AST walk and is not to a
    substring search — and this file's own docstrings name most of the family.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        clauses = (
            node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        )
        for clause in clauses:
            if isinstance(clause, ast.Name):
                names.add(clause.id)
            elif isinstance(clause, ast.Attribute):
                names.add(clause.attr)
    return names


def test_every_storefront_error_subclass_is_mapped_to_a_response():
    """**The structural close of P10-2**, so the eighth subclass cannot repeat it.

    `DemoNotSeededError` was not a slip of attention — it was a family with no
    membership check, and one member with a docstring saying `503` that nothing
    honoured. A subclass is mapped when a route catches it or a classified
    handler answers it; anything else is a bare `500` waiting for the graph to
    reach the state its own docstring describes.
    """
    source = Path(storefront_api.__file__).read_text(encoding="utf-8")
    caught = _caught_names(source)
    handled = {
        klass.__name__
        for klass in set(CROSS_CUTTING_HANDLERS) | set(RESHAPED_HANDLERS)
    }
    family = {klass.__name__ for klass in _subclasses(storefront.StorefrontError)}
    assert family, "the family is read off the live class tree, and it is empty"

    assert family - caught - handled == set()

    # the control: the reader really does find `except` clauses, and really does
    # report a member nothing catches
    assert "QuiesceTimeoutError" in caught
    assert {"NeverCaughtError"} - caught - handled == {"NeverCaughtError"}


def test_every_service_error_subclass_is_mapped_or_declared_unreachable():
    """The same guard on the family that actually crosses the layer boundary.

    `SERVICE_ERROR_RESPONSES` and `SERVICE_ERRORS_UNREACHABLE` must **partition**
    the subclasses of `ServiceError` — read off the live class tree, so a
    subclass added in `services.py` lands in neither and reddens here rather
    than reaching a participant as a Python class name.
    """
    family = _subclasses(ServiceError)
    mapped, unreachable = set(SERVICE_ERROR_RESPONSES), set(SERVICE_ERRORS_UNREACHABLE)
    assert len(family) == 10, sorted(k.__name__ for k in family)
    assert family - (mapped | unreachable) == set()
    assert not mapped & unreachable
    assert (mapped | unreachable) - family == set()
    # every "unreachable" claim carries its reason, not just a membership
    assert all(reason.strip() for reason in SERVICE_ERRORS_UNREACHABLE.values())
    # the control: a subclass in neither bucket is reported
    assert (family | {KeyError}) - (mapped | unreachable) == {KeyError}


def test_every_inherited_handler_states_why_it_produces_no_row():
    """`INHERITED_HANDLERS` is an *exclusion rule*, and an exclusion rule that
    says only "excluded" is the thing P10-1 was. Each entry carries the reason;
    the sweep above is what checks the reasons are true."""
    assert all(reason.strip() for reason in INHERITED_HANDLERS.values())
    assert ServiceError not in INHERITED_HANDLERS
    assert ServiceError in RESHAPED_HANDLERS


# ═══════════════════════════════════════════════════════════════════════════
# The image wiring (§4.7) and the readiness preflight (§4.9)
# ═══════════════════════════════════════════════════════════════════════════


def _assets(root: Path, extension: str, ids=("widget-001", "widget-002", "widget-003")):
    products = root / "products"
    products.mkdir(parents=True, exist_ok=True)
    for product_id in ids:
        (products / f"{product_id}{extension}").write_bytes(b"\x00")
    return root


def test_create_app_forwards_one_directory_to_both_the_manifest_and_the_mount(
    seeded, tmp_path, monkeypatch
):
    """S8's own done-condition, built so that missing it goes **red** — which
    the obvious version of this test does not.

    `config.STOREFRONT_DIR` is pointed at a **different, also-populated**
    directory whose assets carry a different extension, so the two trees
    produce different `imageUrl`s. That separates the three ways to get this
    wrong, where a single tmp directory against an unset config catches only
    the last: a `create_app` that reads config instead of forwarding fails with
    **wrong** URLs; one that forwards only to the mount fails the same way
    (the `Storefront` falls back to `config.STOREFRONT_DIR`, which S7 shipped);
    one that forwards only to the `Storefront` serves 404s from the mount; and
    an unset config would have hidden the first two behind `null`.
    """
    _assets(tmp_path / "config-tree", ".png")
    monkeypatch.setattr(config, "STOREFRONT_DIR", str(tmp_path / "config-tree"))
    served = _assets(tmp_path / "served", ".webp")

    app = _build_app(seeded, storefront_dir=served)
    with TestClient(app) as client:
        # §4.7's "built at startup only": the manifest is already populated
        # before the first request, so no participant's catalog fetch lists a
        # directory. `list_catalog`'s lazy build would leave this at 0.
        assert app.state.storefront_preflight["images"] == 3

        session = _join(client, "Ada", "en")
        rows = client.get(f"{API_PREFIX}/catalog", headers=_bearer(session)).json()
        urls = {row["productId"]: row["imageUrl"] for row in rows}

        assert urls == {
            "widget-001": "/shop/products/widget-001.webp",
            "widget-002": "/shop/products/widget-002.webp",
            "widget-003": "/shop/products/widget-003.webp",
        }
        # ...and the mount serves that same tree, not the config one
        assert client.get("/shop/products/widget-001.webp").status_code == 200
        assert client.get("/shop/products/widget-001.png").status_code == 404


def test_an_empty_asset_directory_is_a_legitimate_deployment(seeded, tmp_path):
    """§4.7: the text-only card variant. An empty manifest logs a count and
    starts — it is explicitly **not** a preflight condition."""
    served = tmp_path / "bare"
    (served / "products").mkdir(parents=True)
    app = _build_app(seeded, storefront_dir=served)
    with TestClient(app) as client:
        assert app.state.storefront_preflight["images"] == 0
        session = _join(client, "Ada", "en")
        rows = client.get(f"{API_PREFIX}/catalog", headers=_bearer(session)).json()
        assert {row["imageUrl"] for row in rows} == {None}


def test_the_preflight_refuses_to_start_without_the_demo_agent(conn, wf_repo,
                                                               storefront_config):
    """§4.9: a mis-seeded demo can no longer come up "green but dead".

    Without the agent, every participant's first message would 500 —
    `services._validate_and_derive_role` raises `UnknownMemberError` on an
    unresolvable mention *before any write* — while the bring-up script's "a
    reachable `/shop` with a working join" done-condition was met.
    """
    wf_repo.materialize_snapshot(
        WS, key=config.TRIGGER_DEF_KEY, version=config.TRIGGER_DEF_VERSION,
        name="Trigger def", kind="conversation", start_key="intake",
        steps=_SNAPSHOT_STEPS, transitions=_SNAPSHOT_TRANSITIONS,
    )
    _seed_catalog(conn, _catalog_rows(2))
    try:
        with pytest.raises(StorefrontPreflightError, match="seed_demo.sh"):
            with TestClient(_build_app(wf_repo)):
                pass  # pragma: no cover — startup raises
    finally:
        db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")


def test_the_preflight_refuses_to_start_without_the_workflow_snapshot(
    conn, wf_repo, storefront_config
):
    wf_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    _seed_catalog(conn, _catalog_rows(2))
    try:
        with pytest.raises(StorefrontPreflightError, match="seed_salesperson.sh"):
            with TestClient(_build_app(wf_repo)):
                pass  # pragma: no cover
    finally:
        db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")


def test_the_preflight_refuses_to_start_on_an_empty_catalog(conn, wf_repo,
                                                            storefront_config):
    wf_repo.ensure_agent(WS, agent_id=AGENT, name="Demo agent", created_at=90)
    wf_repo.materialize_snapshot(
        WS, key=config.TRIGGER_DEF_KEY, version=config.TRIGGER_DEF_VERSION,
        name="Trigger def", kind="conversation", start_key="intake",
        steps=_SNAPSHOT_STEPS, transitions=_SNAPSHOT_TRANSITIONS,
    )
    with pytest.raises(StorefrontPreflightError, match="seed_catalog.sh"):
        with TestClient(_build_app(wf_repo)):
            pass  # pragma: no cover


def test_the_preflight_passes_on_a_correctly_seeded_workspace(client):
    """The positive control for the three refusals above: without it, a
    preflight that raised unconditionally would pass all of them."""
    assert client.app.state.storefront_preflight["products"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# Source tripwires carried from S6's gate
# ═══════════════════════════════════════════════════════════════════════════


def test_the_router_never_authenticates_through_the_record_cache():
    """`lookup` and `resolve_token` return the identical `ParticipantRecord`,
    so a router authenticating through the read-through cache would be
    indistinguishable from one authenticating against the graph — and a deleted
    participant would keep resolving out of stale memory until the process
    restarted (`docs/reviews/salesperson-ui-impl.md` `## Pass 6`).

    **This tripwire goes vacuous at S9, and that is the correct end state**
    (plan v1.19's S9 row): S9 removes the record cache whole, at which point
    the rule it enforces holds structurally rather than by assertion. It is
    written anyway because S8 is where it does its work.

    **Checked on the parsed module, not on a `".lookup(" not in source` grep**,
    and the substitution is a strict improvement rather than a liberty: the
    grep trips on this very docstring — the same "a section quotes the wrong
    spelling even to disown it" failure the plan's own `FALKORCHAT_…PRESENTER…`
    check is worded around — while an AST walk sees calls and cannot be
    defeated by whitespace either. `getattr(x, "lookup")` is covered too,
    because that is the way round the first check that costs nothing to write.
    The checker is proved non-vacuous by
    `test_the_lookup_tripwire_catches_a_router_that_does_call_lookup`.
    """
    assert _lookup_call_sites(Path(storefront_api.__file__).read_text("utf-8")) == []
    # the positive control — the router does authenticate, through the graph
    assert "shop.resolve_token(" in Path(storefront_api.__file__).read_text("utf-8")


def _lookup_call_sites(source: str) -> list[int]:
    """Line numbers of every `.lookup(...)` call, and every `getattr(_,
    "lookup")`, in `source`. Prose and comments are invisible to it."""
    tree = ast.parse(source)
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "lookup":
            hits.append(node.lineno)
        if (
            isinstance(func, ast.Name)
            and func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "lookup"
        ):
            hits.append(node.lineno)
    return hits


def test_the_lookup_tripwire_catches_a_router_that_does_call_lookup():
    """The tripwire's own control. A source check that cannot be shown to fire
    is a comment with a `def` in front of it."""
    assert _lookup_call_sites("record = shop.lookup(participant_id)") == [1]
    assert _lookup_call_sites('f = getattr(shop, "lookup")\n') == [1]
    # ...and it is not merely matching the word anywhere
    assert _lookup_call_sites("# never call .lookup( here\nx = 1\n") == []


def test_the_router_is_the_only_thing_between_a_token_and_the_graph():
    """`get_participant` resolves through `Storefront.resolve_token` and
    nothing else: no second resolution path can drift from it."""
    source = inspect.getsource(storefront_api.build_storefront_router)
    assert source.count("resolve_token(") == 1


def _raised_refusals() -> dict[str, set[tuple[int, str]]]:
    """`{route function name: {(status, token)}}` — every `StorefrontHTTPError`
    the router can raise, read off the parsed source.

    Both argument shapes the router uses are resolved: a literal token, and
    `<StorefrontError subclass>.code`, whose value is the plan's own name for
    that response (`unscoped_participant`, `reset_state_unknown`).
    """
    source = Path(storefront_api.__file__).read_text(encoding="utf-8")
    builder = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_storefront_router"
    )
    found: dict[str, set[tuple[int, str]]] = {}
    for function in builder.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        raised: set[tuple[int, str]] = set()
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "StorefrontHTTPError"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
            ):
                continue
            token = node.args[1]
            if isinstance(token, ast.Constant):
                raised.add((node.args[0].value, token.value))
            elif isinstance(token, ast.Attribute) and token.attr == "code":
                raised.add((
                    node.args[0].value,
                    getattr(storefront, token.value.id).code,
                ))
            else:  # pragma: no cover — a shape this reader cannot resolve
                raise AssertionError(
                    f"{function.name} raises StorefrontHTTPError with a token "
                    f"this reader cannot resolve: {ast.dump(token)}"
                )
        found[function.name] = raised
    return found


def test_no_route_can_raise_a_refusal_it_does_not_declare():
    """**The residue the two halves of the gate leave open, closed.**

    Neither half sees a response a route *raises* but never *declares*: the
    declaration half compares declarations against the table, and the handler
    half only crosses the three cross-cutting handlers. A route that raised an
    undeclared `410` would sail through both and reach the client as exactly
    the *unruled response* §5.3 spent eight passes closing — the one C13 exists
    to shout about, arriving from the server rather than despite it.

    So every `StorefrontHTTPError` the router can raise is read off the parsed
    source and checked against that route's own `responses={…}`. The two
    credential dependencies are attributed to the routes that carry them,
    because that is where their refusals actually surface.
    """
    app = _gate_app()
    endpoints = {
        (method, path): route.endpoint.__name__
        for path, route in _raw_routes(app)
        if path.startswith(API_PREFIX)
        for method in (getattr(route, "methods", None) or ())
        if method not in {"HEAD", "OPTIONS"}
    }
    declared = declared_pairs(app)
    raised = _raised_refusals()

    # the positive control: the reader really did find the refusals, and
    # resolved both argument shapes
    assert (409, "unscoped_participant") in raised["reset"]
    assert (504, "reset_state_unknown") in raised["reset"]
    assert (404, "no_current_order") in raised["advance_order"]
    assert raised["get_participant"] == {(401, "invalid_token")}

    for (method, path), name in endpoints.items():
        own = set(raised.get(name, set()))
        if (method, path) in PARTICIPANT_ROUTES:
            own |= raised["get_participant"]
        if (method, path) in PRESENTER_ROUTES:
            own |= raised["get_presenter"]
        route_declared = {
            (status, token)
            for m, p, status, token in declared
            if (m, p) == (method, path)
        }
        assert own <= route_declared, (
            f"{method} {path} can raise {sorted(own - route_declared)}, which "
            "it does not declare — an unruled response reaching the client "
            "from the server side, which neither half of the gate sees"
        )


# ═══════════════════════════════════════════════════════════════════════════
# The observed-response coverage check — last in the file, deliberately
# ═══════════════════════════════════════════════════════════════════════════


def test_no_response_this_file_observed_is_missing_from_the_table():
    """`{observed} ⊆ §5.3's table`, over every response every test above saw.

    This is the direction that catches an **unruled response arriving from the
    server** — P10-1's `404 {"error":"ThreadNotFoundError"}` is a member of
    `{observed}` and not of the table, so it fails here even on a route nobody
    thought to write a contract test for. Safe under any `-k` or `-x` subset:
    it only ever judges responses that were actually produced.
    """
    orphans = _OBSERVED - FLAT_TABLE
    assert orphans == set(), (
        f"responses observed in this file with no row in §5.3's completeness "
        f"table: {sorted(orphans)}"
    )


def test_every_row_of_the_table_was_produced_by_execution(request):
    """`§5.3's table ⊆ {observed}` — *every declared entry is proved producible*,
    which the file claimed in prose and did not check.

    Two declared rows had no producer at all: deleting `reset`'s
    `except UnknownParticipantError` and `advance_order`'s
    `except UnknownOrderError` both left the suite green (M-D, M-V). Both now
    fail here, and so does the next one — no tagging to keep in step, because
    the evidence is the response the server actually sent.

    Skipped under a `-k` filter, where a subset of the file cannot cover the
    whole table by construction; the full-suite run is where it does its work,
    and the `⊆` direction above holds unconditionally either way.
    """
    if request.config.option.keyword:
        pytest.skip("a -k subset cannot cover the whole table by construction")
    missing = FLAT_TABLE - _OBSERVED
    assert missing == set(), (
        f"rows of §5.3's completeness table that no test in this file ever "
        f"provoked: {sorted(missing)} — declared, tabled, and not proved "
        "producible"
    )
