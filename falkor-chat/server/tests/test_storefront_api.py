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
import sys
from pathlib import Path
from typing import get_args

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis import exceptions as redis_exceptions
from test_app import _FASTAPI_BUILTIN_PATHS, _route_entries

from falkorchat import config, db, storefront, storefront_api
from falkorchat import repository as repository_module
from falkorchat import services as services_module
from falkorchat.app import _register_error_handlers, create_app
from falkorchat.config import CallContext
from falkorchat.services import (
    SearchNotAvailableError,
    ServiceError,
    Services,
    ThreadNotFoundError,
)
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

    The delivered gate computed a difference against a baseline app and
    therefore saw **5** of these — the twelve it subtracted included
    `ServiceError`, which is live on `/shop/api` because the storefront calls
    the same `services` layer.

    **The literal count is deliberately not asserted** (P11-7): the equality
    below already catches everything a `len(...) == 17` would, and §4.9 decided
    the enumerated-vs-derived trade the other way for `_FASTAPI_BUILTIN_PATHS`
    — "a framework upgrade must not red-fail an assertion about this app". The
    failure message is no worse for it.
    """
    app = _gate_app()
    registered = registered_handlers(app)
    assert registered == (
        set(CROSS_CUTTING_HANDLERS)
        | set(ENVELOPE_HANDLERS)
        | set(RESHAPED_HANDLERS)
        | set(INHERITED_HANDLERS)
    ), sorted(e.__name__ for e in registered)
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


def test_the_gate_fails_when_a_dependency_gains_a_parameter_fastapi_will_validate(
    monkeypatch,
):
    """The same mutation one level down — **and the control `_takes_params`'
    recursion never had** (P11-8).

    FastAPI validates a request iff the route *or any of its dependencies*
    declares a body, query or path parameter, so `_takes_params` recurses. No
    delivered route reaches that arm — the only dependency parameter the
    storefront declares is `Authorization`, a header, which is deliberately
    excluded — so replacing the recursive arm with `False` (mutation N-J) left
    the file green and the derivation's own control could not tell.
    """
    app = _gate_app()
    evaluate_gate(app)

    catalog = _route_object(app, "GET", f"{API_PREFIX}/catalog")
    borrowed = _route_object(app, "GET", f"{API_PREFIX}/messages")
    assert not _takes_params(catalog.dependant)

    monkeypatch.setattr(
        catalog.dependant,
        "dependencies",
        [*catalog.dependant.dependencies, borrowed.dependant],
    )
    # the recursion, exercised: the parameter is on a sub-dependency, not here
    assert not catalog.dependant.query_params and not catalog.dependant.path_params
    assert _takes_params(catalog.dependant)

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


def test_the_service_error_re_shaper_answers_only_on_shop_api():
    """**P11-3** — the re-shaper's `/shop/api` path scope, pinned by execution.

    `if request.url.path.startswith(API_PREFIX)` → `if True` survived the whole
    file (mutation N-A), because no delivered app carries both surfaces:
    `create_app` refuses `storefront and dev_surface` together and registers
    this handler only `if shop is not None`, so *absence* — not the path check
    — is what keeps the legacy envelope byte-identical there.

    The scope is nonetheless a real property of the handler, so it is asserted
    against an app built to carry both: the same `ThreadNotFoundError`, raised
    on each side, must answer the storefront's `401 invalid_token` inside
    `/shop/api` and the incumbent's `404 {"error": "<class name>"}` outside it.
    """
    app = FastAPI()
    _register_error_handlers(app)
    storefront_api.register_storefront_error_handlers(app)

    @app.get("/legacy/thing")
    def _legacy() -> dict:  # pragma: no cover — raises on every call
        raise ThreadNotFoundError("t1")

    @app.get(f"{API_PREFIX}/thing")
    def _shop() -> dict:  # pragma: no cover — raises on every call
        raise ThreadNotFoundError("t1")

    with TestClient(app) as client:
        outside = client.get("/legacy/thing")
        inside = client.get(f"{API_PREFIX}/thing")

    assert (inside.status_code, inside.json()["error"]) == (401, "invalid_token")
    assert (outside.status_code, outside.json()["error"]) == (404, "ThreadNotFoundError")


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

    **Two tokens, because "every candidate" over a set of size one is not a
    measurement of anything** (P11-4). With one minted token, replacing the
    loop with `compare_digest(candidates[0], token)` — the mutation this test
    names — leaves it green, and dies only incidentally in the `ServiceError`
    sweep, whose helper happens to log in twice. The property is real and about
    to move: a presenter who logs in twice must not be locked out of the first
    session, and S10 relocates this code to `Storefront.presenter_login`.

    `self._tokens` is a `set`, so candidate order is arbitrary. The assertions
    are written not to depend on it: each token verifies, and the *rejection*
    of a non-member is what proves every candidate was compared, since that is
    the path `any()` cannot short-circuit.
    """
    sessions = storefront_api._PresenterSessions()
    first, second = sessions.mint(), sessions.mint()
    assert first != second
    seen: list[tuple[str, str]] = []

    class _Hmac:
        @staticmethod
        def compare_digest(a, b):  # noqa: ANN001
            seen.append((a, b))
            return hmac.compare_digest(a, b)

    monkeypatch.setattr(storefront_api, "hmac", _Hmac)

    # both live sessions verify, whichever order the set yields them in
    for token in (first, second):
        seen.clear()
        assert sessions.verify(token) is True
        # every comparison made was against a live candidate, and the last one
        # is the match `any()` stopped on
        assert {known for known, _ in seen} <= {first, second}
        assert seen[-1] == (token, token)

    seen.clear()
    near_miss = first[:-1] + ("A" if first[-1] != "A" else "B")
    assert sessions.verify(near_miss) is False
    # the near miss went through the same comparison against **both** candidates,
    # not a membership test and not just the first one
    assert {known for known, _ in seen} == {first, second}
    assert {presented for _, presented in seen} == {near_miss}


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


def test_a_reset_all_whose_pre_drain_roster_read_times_out_never_enters_the_sweep(
    seeded,
):
    """**P10-9's third pass** — the pre-drain roster read sits *outside* the
    `try`, and this is the branch that says what that means.

    The two branches are distinguishable from the wire, which is what makes
    this an assertion rather than a restatement of the code: the `except` arm
    answers `504` **with** a `participants` key (the roster it re-read, or
    `null` when that re-read timed out too), while a failure before the `try`
    reaches the typed handler, which answers `{"error": …}` and nothing else.
    So moving the read inside the `try` reddens here.

    Why it must not move is at the call site: the arm's `forget_all()` +
    `clear_all_turns()` are correct only after a sweep that may have committed,
    and running them before the drain would discard turn state that was never
    waited on. Why `504` is nonetheless honest on a read that changed nothing
    is the `_TIMEOUT` note above — the conservative of the two responses §5.3
    gives a `writes` route, never the harmful direction.
    """
    repo = _FailingMethodRepo(
        seeded, "list_participants", redis_exceptions.TimeoutError("timed out")
    )
    with TestClient(_build_app(repo)) as client:
        _join(client, "Ada", "en")
        response = client.post(
            f"{API_PREFIX}/presenter/reset-all", headers=_presenter(client)
        )

    assert response.status_code == 504, response.text
    body = response.json()
    assert body["error"] == "reset_state_unknown"
    # the discriminator: the typed handler carries no roster, because there was
    # no sweep to report on
    assert "participants" not in body
    # the drain was never entered and the sweep was never issued — one attempt
    # at one read, and no retry
    assert repo.calls == 1


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
    """Every subclass of `root` **defined in the `falkorchat` package**.

    The package filter is what makes the family a property of the shipped code
    rather than of what has been imported: a test that mints a synthetic
    subclass to exercise a mapping would otherwise enter the live class tree
    and make the two "the family is exactly ten" assertions depend on garbage
    collection reclaiming it in time (`docs/reviews/salesperson-ui-impl.md`
    `## Pass 12`, P12-4). It costs nothing real — every production subclass of
    both families is in this package by construction, and one added outside it
    would be a defect of a different kind.
    """
    found = set()
    for sub in root.__subclasses__():
        if sub.__module__.partition(".")[0] == "falkorchat":
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


def test_the_only_behavioural_unreachability_claim_is_pinned_to_its_producer():
    """**P11-6** — six of the seven "unreachable" reasons say *no storefront
    route calls that layer*, which
    `test_the_routers_service_layer_reach_is_exactly_what_the_exemptions_assume`
    now checks. The seventh is different in kind: it argues that
    `UnknownOrderTransitionError` cannot be reached **because a `422` answers
    first**, which is a claim about two declarations agreeing — and nothing
    held them together. Widening `AdvanceOrderIn.transition`'s `Literal` by one
    string left the file green (mutation N-H) and would put an unmapped
    `ServiceError` on `POST /shop/api/order/advance`: `400 {"error":
    "UnknownOrderTransitionError"}`, a Python class name where §5.3 expects a
    plan token.
    """
    literal = storefront_api.AdvanceOrderIn.model_fields["transition"].annotation
    assert set(get_args(literal)) == set(Services._ORDER_TRANSITIONS)  # noqa: SLF001


def test_the_service_error_map_resolves_through_the_class_tree():
    """**P11-10** — `service_error_response` walks `type(exc).__mro__`, and
    replacing that with an exact-type lookup left the file green (mutation
    N-B). Unexercised, the walk also quietly weakened its neighbour: a future
    subclass of `ThreadNotFoundError` placed in `SERVICE_ERRORS_UNREACHABLE`
    would still be *mapped*, so the partition guard would not mean what it says.

    Both halves are pinned here rather than the walk being dropped, because the
    walk is the fail-safe direction: an unclassified subclass answers §5.3's
    token instead of a Python class name, which is the whole point of the
    re-shaper. What it must not do is contradict the partition.
    """
    # the walk, exercised — a subclass inherits its parent's mapping
    derived = type("_DerivedThreadNotFound", (ThreadNotFoundError,), {})
    assert service_error_response(derived("gone")) == (401, "invalid_token")

    # ...and minting it did not disturb the family the two partition tests
    # assert is exactly ten. This used to need `del` + `gc.collect()` and so
    # rested on reclaim timing; `_subclasses` now filters on the defining
    # package, which holds while the subclass is still alive (P12-4)
    assert derived in ThreadNotFoundError.__subclasses__()
    assert len(_subclasses(ServiceError)) == 10
    assert derived not in _subclasses(ServiceError)

    # ...and it agrees with the partition: nothing declared unreachable
    # inherits a mapping through that same walk
    for klass in SERVICE_ERRORS_UNREACHABLE:
        assert not set(klass.__mro__) & set(SERVICE_ERROR_RESPONSES), (
            f"{klass.__name__} is declared unreachable but the MRO walk maps it"
        )


def test_every_inherited_handler_states_why_it_produces_no_row():
    """`INHERITED_HANDLERS` is an *exclusion rule*, and an exclusion rule that
    says only "excluded" is the thing P10-1 was. Each entry carries the reason.

    **What checks the reasons are true is not the sweep** — this docstring said
    it was, and the sweep arms three `ServiceError` faults and no workflow
    fault, so it checked these eleven reasons in an empty intersection (P11-1).
    Every excuse here has one of two shapes, and each has its own mechanism
    below: *"no storefront route calls layer X"* is
    `test_the_routers_service_layer_reach_is_exactly_what_the_exemptions_assume`,
    and *"no storefront route raises it"* is
    `test_the_raises_a_route_can_reach_are_exactly_what_the_exemptions_assume`.
    """
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


def _router_source() -> str:
    return Path(storefront_api.__file__).read_text(encoding="utf-8")


def _parse_router(source: str) -> ast.FunctionDef:
    """`build_storefront_router`'s node, out of `source`.

    Takes the source rather than reading it, so every reader below can be run
    against a synthetic snippet as its own positive control — the thing
    `_raised_refusals` had no way to do.
    """
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_storefront_router"
    )


def _storefront_source() -> str:
    return Path(storefront.__file__).read_text(encoding="utf-8")


def _services_source() -> str:
    return Path(services_module.__file__).read_text(encoding="utf-8")


def _repository_source() -> str:
    return Path(repository_module.__file__).read_text(encoding="utf-8")


def _attribute_targets(node, prefixes: set[str]) -> set[str]:
    """The `<prefix>.<name>` accesses anywhere under `node`.

    The prefix is matched as **unparsed source**, so `services`,
    `shop._services` and `self._services` are one mechanism rather than three
    special cases — which is the thing the first version of this reader got
    wrong: it matched `ast.Attribute` whose `.value` was a bare `ast.Name`, and
    `shop._services.<name>` is an Attribute on an Attribute
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 12`, P12-1).

    Parsed, not grepped, for the reason `_caught_names` is: `advance_order`'s
    docstring names `services.order_belongs_to_customer` in prose, which a
    substring search reports as a call and an AST walk does not see.
    """
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute) and ast.unparse(child.value) in prefixes
    }


# The `ast` node types that bind a **local name to a value expression written
# at the binding site**, and are therefore the ones an alias can be spelled
# with. `_bindings` handles exactly these and nothing else, and
# `test_the_alias_reader_covers_every_binding_form_the_grammar_has` holds this
# set plus `_NON_ALIAS_BINDING_NODES` against the grammar's own enumeration —
# so this is a closed set by construction rather than by anyone's memory.
#
# `ast.MatchAs` is here rather than `ast.Match` because the *pattern* is what
# binds; the value it binds is the enclosing `match` subject, which the pattern
# node does not carry, so `_bindings` reaches it through `ast.Match`.
_ALIAS_BINDING_NODES: frozenset[type] = frozenset({
    ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor,
    ast.comprehension, ast.withitem, ast.MatchAs,
})

# Every other name-binding node in the grammar, with the reason it cannot make
# a second name for an object already named by a prefix. These are **not**
# oversights being tolerated: each one binds something that is provably not the
# value on its right-hand side, or is bound by a caller this reader cannot see.
_NON_ALIAS_BINDING_NODES: dict[type, str] = {
    ast.AugAssign: (
        "binds `target OP value`, never `value` — the object bound is whatever "
        "the operator returns, so it cannot be a second name for the operand"
    ),
    ast.Delete: "un-binds a name; it introduces none",
    ast.FunctionDef: "binds a function object to its own `def` name",
    ast.AsyncFunctionDef: "binds a coroutine function to its own `def` name",
    ast.ClassDef: "binds a class object to its own `class` name",
    ast.Import: "binds a module object, never an attribute of `self`",
    ast.ImportFrom: "binds a module attribute resolved at import time",
    ast.alias: "the `as` half of an import — same object, a different name",
    ast.ExceptHandler: "binds the caught exception, and is deleted at block exit",
    ast.Global: "declares a scope for a name; binds no value",
    ast.Nonlocal: "declares a scope for a name; binds no value",
    ast.arg: (
        "a parameter, bound by the **caller** — this is precisely the "
        "`passed to a helper` shape the reach statement names as outside the "
        "walk. Closing it needs interprocedural analysis, not another node type"
    ),
    ast.keyword: "a call-site keyword name; it binds nothing in the caller's scope",
    ast.MatchStar: "binds a `list` of the subject's unmatched items, not the subject",
    ast.MatchMapping: "`rest` binds a `dict` of unmatched keys, not the subject",
    ast.TypeVar: "a type parameter, in the type namespace",
    ast.ParamSpec: "a type parameter, in the type namespace",
    ast.TypeVarTuple: "a type parameter, in the type namespace",
    ast.TypeAlias: (
        "binds a `TypeAliasType` whose value is lazily evaluated in the type "
        "namespace; a `type` statement cannot name an attribute of `self`"
    ),
}


def _bind_pairs(target, value) -> list[tuple[str, str]]:
    """`(name, unparsed value)` for one target/value pair, destructuring.

    A bare `ast.Name` target is the whole of it in the common case. The
    sequence case (`svc, _ = self._services, None`) is paired **positionally,
    and only when the pairing is decidable without running anything**: both
    sides literal sequences, equal length, no `*` on either side. Anything
    else — a `Starred` target, a call on the right — contributes nothing,
    which is a miss the reach statement names rather than a silent one.
    """
    if isinstance(target, ast.Name):
        return [(target.id, ast.unparse(value))]
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(
            isinstance(elt, ast.Starred) for elt in [*target.elts, *value.elts]
        )
    ):
        found: list[tuple[str, str]] = []
        for sub_target, sub_value in zip(target.elts, value.elts):
            found += _bind_pairs(sub_target, sub_value)
        return found
    return []


def _bindings(node) -> list[tuple[str, str]]:
    """Every `(local name, unparsed value)` bound under `node`.

    One branch per member of `_ALIAS_BINDING_NODES`. The reader this replaced
    had exactly one — `ast.Assign` — while the sentence above it said "any
    local name transitively bound to it", and `svc: object = self._services`,
    the spelling this tripwire exists to redden on plus a type annotation,
    survived the whole file at 183 passed
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 14`, P14-2). Annotated
    local assignment is a house idiom in the two files S9 edits, so that was
    not an exotic spelling; it was the ordinary one.

    `with <expr> as x` is read as a binding of `<expr>` although what is
    actually bound is `<expr>.__enter__()`. That over-approximates on purpose:
    an over-approximation here can only make the guard fire on a call the
    request cannot reach — a false **red**, which is loud and cheap — while the
    under-approximation is the defect this file has now produced fourteen
    times (`## Pass 15`).

    A `for`/comprehension target is read only over a **literal** sequence
    (`for svc in (self._services,)`), because that is the only iterable whose
    elements are visible to a reader that runs nothing.

    **This is the target axis only.** What each pair's *value* half means is
    decided by `_alias_prefixes`, and it is `ast.unparse(value) in prefixes` —
    exact source-text identity. Every node type here is read at that
    granularity, so widening this dict does not widen what counts as naming
    the object (`## Pass 15`, P15-1).
    """
    found: list[tuple[str, str]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                found += _bind_pairs(target, child.value)
        elif isinstance(child, (ast.AnnAssign, ast.NamedExpr)):
            if child.value is not None:
                found += _bind_pairs(child.target, child.value)
        elif isinstance(child, (ast.For, ast.AsyncFor, ast.comprehension)):
            if isinstance(child.iter, (ast.Tuple, ast.List, ast.Set)):
                for element in child.iter.elts:
                    found += _bind_pairs(child.target, element)
        elif isinstance(child, ast.withitem):
            if child.optional_vars is not None:
                found += _bind_pairs(child.optional_vars, child.context_expr)
        elif isinstance(child, ast.Match):
            for case in child.cases:
                if isinstance(case.pattern, ast.MatchAs) and case.pattern.name:
                    found.append((case.pattern.name, ast.unparse(child.subject)))
    return found


def _alias_prefixes(node, seeds: set[str]) -> set[str]:
    """`seeds`, closed over every local name bound to one under `node`.

    A reader that matches a list of spellings guards against the spellings it
    listed. `svc = self._services` followed by `svc.start_workflow_run(ctx)` —
    the shape this tripwire exists to redden on, written in two lines instead
    of one — walked straight past a reader holding the three literal prefixes,
    as did a second router alias
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 13`, P13-1).

    So the prefixes are **derived, not listed**: any name bound to something
    that is already a prefix becomes one, to a fixpoint, which also resolves an
    alias of an alias. The one hand-written prefix left is the attribute the
    object is *reached* by (`shop._services`, `self._services`), which is a
    property of the class rather than of a body someone is editing.

    **What counts as "bound" is `_bindings`, i.e. `_ALIAS_BINDING_NODES`** —
    eight node types, held against the grammar's own list of name-binding nodes
    by `test_the_alias_reader_covers_every_binding_form_the_grammar_has`. That
    test is the enumeration, in the file: deriving the prefixes fixed the
    spelling problem and left a node-type problem exactly one shape smaller
    (`## Pass 14`, P14-2), and only an enumeration closes that.

    **And that closes the *target* axis only.** The line below —
    `if value in prefixes` — is the *value* axis, and it is **exact
    source-text identity** against a prefix already derived, nothing more. So
    a name is added when what it is bound to is spelled exactly like a prefix,
    and not when the reader would have to work out that it denotes the same
    object. The consequence worth naming, because it is a **documented stop
    rather than a bug**: seeded on `self`, this closure adds `me` from
    `me = self` — seeded on `self._services`, it does not, so
    `me._services.<name>` is invisible on the collaborator legs while
    `me.<method>` is followed on the frontier legs. Measured on the delivered
    suite: `me = self` / `me._services.start_workflow_run(...)` injected on the
    router-reached `Storefront.join` survives at **185 passed**, where
    `svc = self._services` on the same point is **1 failed / 184**
    (`## Pass 15`, P15-1). Conditional expressions, container round-trips and
    non-literal iterables stop the same way, for the same reason. Closing that
    axis is alias analysis rather than another node type, and it is **not**
    closed here by decision (`docs/plans/salesperson-ui-coordination.md`,
    "STOPPED — the stopping rule fired").
    """
    prefixes = set(seeds)
    bindings = _bindings(node)
    grew = True
    while grew:
        grew = False
        for name, value in bindings:
            if value in prefixes and name not in prefixes:
                prefixes.add(name)
                grew = True
    return prefixes


def _class_methods(source: str, class_name: str) -> dict[str, ast.AST]:
    """`{method name: node}` for one class defined in `source`."""
    klass = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node
        for node in klass.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _storefront_reach(api_source: str, storefront_source: str) -> set[str]:
    """The `Storefront` methods a `/shop/api` route reaches, transitively.

    The router's own frontier is `shop.<method>`, alias-resolved; each reached
    method's frontier is `self.<method>`, alias-resolved, intersected with the
    class's own methods. Walked to a fixpoint because `list_catalog` reaches
    `filter_products` through `_catalog_rows`, two hops down, and a guard that
    stopped at one would have to argue why one hop is the boundary.
    """
    router = _parse_router(api_source)
    methods = _class_methods(storefront_source, "Storefront")
    frontier = _attribute_targets(router, _alias_prefixes(router, {"shop"}))
    frontier &= methods.keys()
    walked: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in walked:
            continue
        walked.add(name)
        body = methods[name]
        frontier |= (
            _attribute_targets(body, _alias_prefixes(body, {"self"})) & methods.keys()
        ) - walked
    return walked


def _collaborator_reach(
    api_source: str, storefront_source: str, router_attr: str, self_attr: str,
) -> set[str]:
    """The methods of one injected collaborator a `/shop/api` route reaches.

    A route reaches a collaborator two ways, and the exemptions in
    `INHERITED_HANDLERS` are claims about **both**: directly, as
    `<router_attr>.<name>` in the router body, and one or more hops down
    through `shop.<method>`, where `Storefront` calls `<self_attr>.<name>` on
    the router's behalf. Reading only the first is what let a call written in
    `storefront.py` — the half of the surface a route reaches through
    `shop.<method>` — stay invisible (`## Pass 12`, P12-1).

    **One function for both collaborators**, because the `Services` leg and the
    `Repository` leg are the same walk one attribute over, and reading the
    second at the router only is how `Repository.ensure_participant` — reached
    from `Storefront.join`, raising an exception classified nowhere — stayed
    outside every scope (`docs/reviews/salesperson-ui-impl.md` `## Pass 14`,
    P14-1). It also means S10's move of the two `repo.<name>` calls onto
    `Storefront` re-points this leg instead of emptying it.

    **Every prefix on every leg is alias-resolved** (P13-1): the two that reach
    the collaborator, and the two `_storefront_reach` walks `Storefront` with.

    **Where it stops**, said as a property of the walk rather than as a claim
    about the code: the collaborator object is followed only through attribute
    access on a name `_bindings` can bind **to an expression spelled exactly
    like a prefix already derived**. So a call made by handing the object
    somewhere else — passed to a helper, returned, stored on an attribute — is
    outside it, and so is `getattr(svc, "start_workflow_run")`; and so, on
    these two legs specifically, is an alias of the **receiver**
    (`me = self` then `me._services.<name>`), whose value text `self` is not
    the prefix `self._services`. `_storefront_reach`, seeded on `shop`/`self`,
    does follow that one — the asymmetry is documented and measured, not
    accidental (`## Pass 15`, P15-1).
    """
    router = _parse_router(api_source)
    methods = _class_methods(storefront_source, "Storefront")
    reached = _attribute_targets(router, _alias_prefixes(router, {router_attr}))
    for name in _storefront_reach(api_source, storefront_source):
        body = methods[name]
        reached |= _attribute_targets(body, _alias_prefixes(body, {self_attr}))
    return reached


def _service_layer_reach(api_source: str, storefront_source: str) -> set[str]:
    """The `Services` methods a `/shop/api` route reaches. `storefront.py` is
    read here and never written."""
    return _collaborator_reach(
        api_source, storefront_source, "shop._services", "self._services"
    )


def _repository_reach(api_source: str, storefront_source: str) -> set[str]:
    """The `Repository` methods a `/shop/api` route reaches.

    Two legs, exactly as the service one has: `repo.<name>` in the router
    (`repo = shop._repo`, which S10 moves onto `Storefront`) and
    `self._repo.<name>` in the `Storefront` methods the router reaches — which
    is where `join` provisions the participant.
    """
    return _collaborator_reach(
        api_source, storefront_source, "shop._repo", "self._repo"
    )


def _reached_methods(source: str, class_name: str, seeds) -> set[str]:
    """`seeds`, closed over the `self.<name>` calls those methods make.

    The frontier idiom `_storefront_reach` uses, applied inside a collaborator.
    Without it the raise walk stopped one hop earlier than the sentence
    governing it: `Services._refuse_retired_name()` raising a bare
    `HTTPException(410)` and called from the reached `save_profile` **survived
    at 183 passed** and answered `410 {"detail":"gone"}` from
    `POST /shop/api/session`, as did the same raise in
    `Repository.ensure_participant` reached from `Storefront.join`
    (`docs/reviews/salesperson-ui-impl.md` `## Pass 14`, P14-1). A raise one
    helper call out of a *route body* has been in scope since P12-2; this is
    the identical argument one class over.

    `self` is alias-resolved here for the same reason it is everywhere else in
    this file — a walk that resolves aliases on three legs and not the fourth
    is the defect one field over, and this chain has produced that twice.

    **On one axis the fourth leg genuinely is different, and it is written
    down rather than left to be discovered** (`## Pass 15`, P15-1). Seeded on
    `self`, this walk and `_storefront_reach` follow an alias of the
    *receiver*: `me = self` then `me.<name>` is seen. `_collaborator_reach` is
    seeded on `self._services`/`self._repo`, and `_alias_prefixes` compares
    **source text**, so `me = self` adds nothing there and
    `me._services.<name>` is missed — measured, at **185 passed** where the
    equivalent `svc = self._services` is **1 failed / 184**. That is a
    documented stop under a stakeholder decision not to widen the reader
    further, not an oversight
    (`docs/plans/salesperson-ui-coordination.md`, "STOPPED — the stopping rule
    fired").
    """
    methods = _class_methods(source, class_name)
    missing = set(seeds) - methods.keys()
    assert not missing, f"`{class_name}` has no method(s) {sorted(missing)}"
    frontier, walked = set(seeds), set()
    while frontier:
        name = frontier.pop()
        if name in walked:
            continue
        walked.add(name)
        body = methods[name]
        frontier |= (
            _attribute_targets(body, _alias_prefixes(body, {"self"})) & methods.keys()
        ) - walked
    return walked


def _raises_of(source: str, class_name: str, method_names) -> set[str]:
    """The exception classes a `raise` names in the given methods of one class,
    **and in the sibling methods those call** (`_reached_methods`).

    Every seed name is resolved, so a method renamed out from under the caller
    reddens here rather than silently contributing nothing to the union.

    The reached methods are handed to `_raised_class_names` as **one** synthetic
    module rather than one at a time, which is what lets `raise self._mk(...)`
    resolve through `_mk`'s `return`s the way it already does in the two
    storefront modules. Read one method at a time the factory is outside the
    walk, so the raise resolves to the *method name* — loud, because a foreign
    name reddens the set equality, but loud with the wrong name, and the two
    kinds of scope in this file would then resolve the same source differently.
    The scope is still exactly the reached methods: nothing outside them is in
    the module, so this widens what is *resolved*, never what is *read*.
    """
    methods = _class_methods(source, class_name)
    reached = _reached_methods(source, class_name, method_names)
    scope = ast.Module(
        body=[methods[name] for name in sorted(reached)], type_ignores=[]
    )
    return _raised_class_names(scope)


def _named_class(expr) -> str:
    """The class `Foo(...)`, `Foo` and `mod.Foo(...)` all name: `"Foo"`."""
    if isinstance(expr, ast.Call):
        expr = expr.func
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    raise AssertionError(  # pragma: no cover — a shape it cannot resolve
        f"unresolvable class expression: {ast.dump(expr)}"
    )


def _raised_class_names(node) -> set[str]:
    """Every exception class a `raise` under `node` names.

    `raise Foo(...)`, `raise Foo` and `raise mod.Foo(...)` all resolve to
    `"Foo"`; a bare `raise` (re-raise) names nothing and is skipped. Takes a
    node rather than a source string so the one reader serves every scope —
    the router node, and either storefront module whole (P12-2).

    **A `raise <factory>(...)` resolves through the factory.** `storefront.py`
    has one — `raise self._reset_state_unknown(ctx, pid)`, a method that builds
    a `ResetStateUnknownError` — and a reader that stopped at the attribute
    would put a *method name* into a set that is supposed to hold exception
    classes, and would be blind to a factory that returned an `HTTPException`
    one call away from the `raise`. A name counts as a factory when the walked
    tree defines it as a **function**; a class it defines is not one.
    """
    root = ast.parse(node) if isinstance(node, str) else node
    factories = {
        child.name: child
        for child in ast.walk(root)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names: set[str] = set()
    for child in ast.walk(root):
        if not isinstance(child, ast.Raise) or child.exc is None:
            continue
        raised = _named_class(child.exc)
        if raised in factories:
            resolved = {
                _named_class(returned.value)
                for returned in ast.walk(factories[raised])
                if isinstance(returned, ast.Return) and returned.value is not None
            }
            # the one shape that would otherwise resolve to nothing at all,
            # where every other unresolvable shape is loud (P13-4): a factory
            # whose `return`s carry no value contributes `set()` and the raise
            # vanishes. It cannot actually raise anything, so this is a reader
            # invariant rather than a source claim — stated as one.
            assert resolved, (
                f"`raise {raised}(...)` resolved to nothing: the factory has no "
                "`return <expr>` this reader can name"
            )
            names |= resolved
        else:
            names.add(raised)
    return names


# Every `Services` method a `/shop/api` route can reach **today**, spelled out
# so that the step which adds one has to come back here. Three of them are
# direct — `services.<name>` in the router — and the other six arrive through
# `shop.<method>`: `get_state` → `get_cart`/`get_current_order`/`get_profile`,
# `advance_own_order` → `advance_order`/`order_belongs_to_customer`,
# `join`/`reset_participant` → `save_profile`, and `list_catalog` →
# `_catalog_rows` → `filter_products`, which is two hops down.
#
# Deliberately *not* forward-looking — and **not expected to move at S9**.
# `docs/plans/salesperson-ui.md` v1.25 places the trigger on the turn worker,
# reaching the workflow layer through `trigger.maybe_trigger`, whose own
# `start_workflow_run` call site (`trigger.py:82`) is outside all four scopes
# these two guards walk. So this set is a **service-surface tripwire**: red
# means a `Services` call was acquired through the storefront's own
# `self._services` rather than through the trigger — a stop-and-re-decide, not
# a mechanical bump — and S9's done-condition is that it stays **green**. A set
# written to accommodate a forecast would be a guard that cannot fail at the
# one moment it is worth something (`docs/reviews/salesperson-ui-impl.md`
# `## Pass 11`, P11-1); a reader that sees only one spelling of the
# acquisition is the other half of the same failure (`## Pass 12`, P12-1).
#
# **Which `INHERITED_HANDLERS` excuses S9 falsifies is not stated here, and
# that is deliberate.** The version of this comment that named three of them
# named one the row does not carry and dropped one it does (`## Pass 14`,
# P14-3). The mapping is settled in the plan — and under v1.25 **no walk in
# this file can see it**: `_raises_of` over the closed reach reports none of
# the three, because the trigger's call site is outside every scope. The
# evidence is §5.1's S9 row's armed-fault measurement at the response
# boundary, not anything measured here (`## Pass 15`, P15-2).
SERVICE_LAYER_REACH_TODAY = frozenset({
    # direct, in the router body
    "read_messages", "post_message", "get_current_order",
    # through `shop.<method>` → `self._services.<name>`
    "get_cart", "get_profile", "save_profile",
    "advance_order", "order_belongs_to_customer", "filter_products",
})


def test_the_routers_service_layer_reach_is_exactly_what_the_exemptions_assume():
    """**The mechanism `INHERITED_HANDLERS`' eleven excuses never had** (P11-1),
    reading the whole reach those excuses claim (P12-1).

    Every one of those excuses has the form *"no storefront route calls layer
    X"*, and the only test on them asserted that the reason strings are
    non-empty. Its docstring credited "the sweep above", which arms three
    `ServiceError` faults and no workflow fault at all — so it checked eleven
    reasons in an empty intersection, and mutation N-F (a route raising
    `WorkflowEngineDisabledError`) answered `503 {"error":"<class name>"}`
    through it, colliding in status with `demo_not_seeded` on the token C9
    dispatches on.

    **"Calls layer X" is reachability, not one spelling.** The first version of
    this guard pinned `services.<name>` in the router and nothing else, so it
    reddened on `services.start_workflow_run(...)` and stayed green on the two
    other ways the same acquisition is written:
    `shop._services.start_workflow_run(...)` in the router (an Attribute on an
    Attribute, a spelling `storefront_api.py` already uses), and
    `self._services.start_workflow_run(...)` inside the `Storefront` method the
    router calls, which is not in the parsed file at all. The router's true
    reach is **nine** methods; that guard measured **three**, and six of the
    nine were invisible to it.

    So the reader takes the union: both direct spellings in the router, plus
    `self._services.<name>` in every `Storefront` method the router reaches,
    transitively. `storefront.py` is read here and never written.

    **And a local alias is one of those paths** (P13-1). Listing three
    spellings guarded against those three: `svc = self._services` /
    `svc.start_workflow_run(ctx)` — the same acquisition plus one line —
    survived the whole file at 183 passed, and so did a second router alias,
    `svc2 = shop._services`. The direct spelling on the same injection point
    reddened, so the difference was purely the binding. Every prefix on every
    leg is therefore **derived** from the file's own bindings to a fixpoint
    rather than written down, which is also what turns the old hand-written
    `services == shop._services` control into a structural one: a rename is now
    followed instead of blinding the walk.

    **"Its own bindings" is `_bindings`, which is eight `ast` node types**
    (P14-2). Deriving the prefixes over `ast.Assign` alone made the same
    mistake one shape smaller: `svc: object = self._services` — the identical
    two lines with a type annotation, which is a house idiom in the two files
    S9 edits — survived the whole file at 183 passed, while the same lines
    without `: object` were `1 failed`. Which node types those are, and why
    every other name-binding node in the grammar cannot carry an alias, is
    `test_the_alias_reader_covers_every_binding_form_the_grammar_has`, which
    takes its enumeration from `ast` rather than from a list here.

    **All of that is the *target* axis, and the reader has two** (`## Pass 15`,
    P15-1). The value half is `_alias_prefixes`' `value in prefixes` — exact
    source-text identity — so what reddens here is an acquisition written on
    one of the derived prefixes, or on a local name bound to an expression
    spelled **exactly** like one. An alias of the *receiver* does not:
    `me = self` then `me._services.start_workflow_run(...)`, injected on the
    router-reached `Storefront.join`, survives the two-file suite at **185
    passed**, where `svc = self._services` on the same point is
    **1 failed / 184**. A conditional expression, a container round-trip and a
    non-literal iterable stop for the same reason. Those are **documented
    stops, not open defects**: closing them is alias analysis rather than one
    more node type, and the decision to narrow the sentence instead of widening
    the reader is recorded in `docs/plans/salesperson-ui-coordination.md`,
    "STOPPED — the stopping rule fired". Under v1.25 that is enough, because
    what this guard is for is narrower than the shapes it misses — a
    **service-surface tripwire** on the storefront's own `self._services`.
    """
    api, sf = _router_source(), _storefront_source()

    # the control on the direct half, structural rather than hand-written: the
    # walk must have *derived* `services` from the router's own binding. A
    # reader that lost the fixpoint reddens here on the delivered file, not
    # only on a stub.
    assert "services" in _alias_prefixes(_parse_router(api), {"shop._services"})

    assert _service_layer_reach(api, sf) == set(SERVICE_LAYER_REACH_TODAY)

    # the controls on the reader: the five spellings it was widened for are
    # resolved — the three direct ones, and the two aliased ones it still
    # walked past (P13-B, P13-C)
    api_stub = (
        "def build_storefront_router(shop):\n"
        "    services = shop._services\n"
        "    def post(body):\n"
        "        return {}\n"
    )
    sf_stub = "class Storefront:\n    def unrelated(self):\n        return None\n"

    def reach(router_line, storefront_class=sf_stub):
        return _service_layer_reach(
            api_stub.replace("return {}", router_line), storefront_class
        )

    assert reach("return services.start_workflow_run(None)") == {"start_workflow_run"}
    assert reach("return shop._services.start_workflow_run(None)") == {
        "start_workflow_run"
    }
    assert reach(
        "return shop.enqueue_turn(None)",
        "class Storefront:\n"
        "    def enqueue_turn(self, ctx):\n"
        "        return self._services.start_workflow_run(ctx)\n",
    ) == {"start_workflow_run"}

    # P13-B — the alias inside the `Storefront` method: the same acquisition
    # written in two lines instead of one
    assert reach(
        "return shop.enqueue_turn(None)",
        "class Storefront:\n"
        "    def enqueue_turn(self, ctx):\n"
        "        svc = self._services\n"
        "        return svc.start_workflow_run(ctx)\n",
    ) == {"start_workflow_run"}

    # P13-C — a *second* router alias, which the old binding control could not
    # see because it checked a rename of the existing one, not an added one
    assert _service_layer_reach(
        api_stub.replace(
            "    def post(body):\n        return {}\n",
            "    svc2 = shop._services\n"
            "    def post(body):\n"
            "        return svc2.start_workflow_run(None)\n",
        ),
        sf_stub,
    ) == {"start_workflow_run"}

    # ...and an alias of an alias, which is why the derivation is a fixpoint
    assert reach(
        "return shop.enqueue_turn(None)",
        "class Storefront:\n"
        "    def enqueue_turn(self, ctx):\n"
        "        svc = self._services\n"
        "        also = svc\n"
        "        return also.start_workflow_run(ctx)\n",
    ) == {"start_workflow_run"}


# One snippet per member of `_ALIAS_BINDING_NODES`: the body of a `Storefront`
# method the router reaches, writing a `Services` call through a local name
# bound in that form. Every one must resolve to `{"start_workflow_run"}`.
#
# This is the enumeration `## Pass 14` asked to be shipped in the file rather
# than run once in a transcript — *"enumerate every syntactic way to write the
# thing the sentence names, run the delivered reader over each, and list the
# misses"*. What keeps it from being circular is the assertion in the test
# below, which takes the node types from the **grammar** and not from this
# dict, so a binding form nobody classified fails rather than passing silently.
#
# **It enumerates one of the reader's two axes, and says so rather than
# implying otherwise** (`## Pass 15`, P15-1). Every snippet holds the *value*
# fixed at the single spelling `self._services` and varies only the *target*
# node type. Run over the value axis the same enumeration returns misses —
# an alias of the receiver, a conditional expression, a container round-trip,
# a non-literal iterable — three of which survive the delivered suite. Those
# are documented stops, listed in `_alias_prefixes`' docstring and in
# `storefront_api.py`'s guard-reach block. So this dict is a complete
# enumeration **of the target axis**, not of the mechanism.
_ALIAS_FORM_SNIPPETS: dict[type, str] = {
    ast.Assign: (
        "    def enqueue_turn(self, ctx):\n"
        "        svc = self._services\n"
        "        return svc.start_workflow_run(ctx)\n"
    ),
    ast.AnnAssign: (
        "    def enqueue_turn(self, ctx):\n"
        "        svc: object = self._services\n"
        "        return svc.start_workflow_run(ctx)\n"
    ),
    ast.NamedExpr: (
        "    def enqueue_turn(self, ctx):\n"
        "        if (svc := self._services):\n"
        "            return svc.start_workflow_run(ctx)\n"
    ),
    ast.For: (
        "    def enqueue_turn(self, ctx):\n"
        "        for svc in (self._services,):\n"
        "            return svc.start_workflow_run(ctx)\n"
    ),
    ast.AsyncFor: (
        "    async def enqueue_turn(self, ctx):\n"
        "        async for svc in [self._services]:\n"
        "            return svc.start_workflow_run(ctx)\n"
    ),
    ast.comprehension: (
        "    def enqueue_turn(self, ctx):\n"
        "        return [svc.start_workflow_run(ctx) for svc in (self._services,)]\n"
    ),
    ast.withitem: (
        "    def enqueue_turn(self, ctx):\n"
        "        with self._services as svc:\n"
        "            return svc.start_workflow_run(ctx)\n"
    ),
    ast.MatchAs: (
        "    def enqueue_turn(self, ctx):\n"
        "        match self._services:\n"
        "            case svc:\n"
        "                return svc.start_workflow_run(ctx)\n"
    ),
}


def test_the_alias_reader_covers_every_binding_form_the_grammar_has():
    """**The enumeration, in the file** (`## Pass 14`, P14-2 and its closing
    condition).

    This artifact has produced one defect fourteen times — *a stated rule
    broader than the reach the mechanism implements* — and six review passes
    each found it inside the fix for the previous one. Every fix widened what
    the reader looked at: three prefix spellings, then two files, then a
    fixpoint over `ast.Assign`, then this enumeration. Each was correct and
    each left the next instance one spelling away, because the sentence above
    the reader kept naming a **semantic** scope ("any local name transitively
    bound to it") that no finite reader implements, while the body walked a
    **syntactic** one.

    P14-2 is that gap at its smallest: `svc: object = self._services` — the
    acquisition this guard trips on plus a type annotation, and a house idiom
    (**68** annotated local assignments *inside function bodies* in this
    package, **14** of them in `services.py`, 5 in `storefront_api.py`, 3 in
    `storefront.py`) — survived the whole file at 183 passed, while the same
    two lines without `: object` were `1 failed`. That one definition is
    stated because counting at *all* scopes yields different figures, and the
    earlier version of this sentence quoted a package total from one
    definition beside a per-file count from the other (`## Pass 15`, P15-4).

    **What this test closes is the target axis, and only that** (`## Pass 15`,
    P15-1). It enumerates which node types can bind a name; which *values*
    count as naming the object is `_alias_prefixes`' exact-source-text
    comparison, documented there and in `storefront_api.py`'s guard-reach
    block, and deliberately left open. The fourteenth instance was found in
    the gap between the two axes, and the answer adopted was to narrow the
    sentences to the reader rather than widen the reader again
    (`docs/plans/salesperson-ui-coordination.md`, "STOPPED — the stopping rule
    fired").

    So the sentence now names node types, and this test is what makes that a
    closed statement instead of a shorter one. It takes the binding forms from
    the **grammar**: every `ast` node class carrying a target-shaped field
    (`target`, `targets`, `optional_vars`) or a name-shaped one (`name`,
    `names`, `asname`, `arg`, `rest`) is a place the language can introduce a
    name, and each must be classified — walked by `_bindings`, or excluded with
    a written reason that says what it binds instead. A Python version that
    adds a binding form reddens here rather than opening another instance in
    silence — **so long as it carries one of the eight `_fields` names the set
    below keys on**; a node spelled `_fields = ('var', 'value')` would open a
    hole quietly, and none in Python 3.12 is (`## Pass 15`, P15-5).

    The exclusions are not a shorter promise. `ast.arg` is the one that matters
    and it is the one the reach statement already names as outside the walk: a
    parameter is bound by the **caller**, so `_go(self._services)` needs
    interprocedural analysis rather than another node type, and no enumeration
    of binding forms will ever reach it. Saying that here, next to the forms
    that *are* covered, is the difference between a documented non-reach and
    the defect this chain keeps producing.
    """
    binding_nodes = {
        obj
        for obj in vars(ast).values()
        if isinstance(obj, type)
        and issubclass(obj, ast.AST)
        and set(getattr(obj, "_fields", ()))
        & {"target", "targets", "optional_vars", "name", "names", "asname",
           "arg", "rest"}
    }
    classified = _ALIAS_BINDING_NODES | set(_NON_ALIAS_BINDING_NODES)

    assert not (_ALIAS_BINDING_NODES & set(_NON_ALIAS_BINDING_NODES))
    assert classified == binding_nodes, (
        "the grammar's name-binding nodes and this file's classification of "
        "them have diverged: unclassified "
        f"{sorted(k.__name__ for k in binding_nodes - classified)}, "
        "classified-but-gone "
        f"{sorted(k.__name__ for k in classified - binding_nodes)}. Every one "
        "is either a form an alias can be written in — walk it in `_bindings` "
        "— or one that provably binds something else, which is a reason, not "
        "an omission"
    )
    assert all(reason.strip() for reason in _NON_ALIAS_BINDING_NODES.values())

    # ...and every walked form is exercised against the **delivered** reader,
    # on the leg the tripwire watches: `shop.enqueue_turn(...)` in the router,
    # `<local>.start_workflow_run(...)` in `storefront.py`. The value half is
    # held fixed at `self._services` throughout: this is the target axis.
    api_stub = (
        "def build_storefront_router(shop):\n"
        "    def post(body):\n"
        "        return shop.enqueue_turn(None)\n"
    )
    assert set(_ALIAS_FORM_SNIPPETS) == _ALIAS_BINDING_NODES
    for node_type, body in _ALIAS_FORM_SNIPPETS.items():
        storefront_stub = "class Storefront:\n" + body
        # the snippet really is written in the form it is filed under, so a
        # control cannot pass by exercising a different node type
        assert any(
            isinstance(node, node_type)
            for node in ast.walk(ast.parse(storefront_stub))
        ), f"the {node_type.__name__} snippet does not contain one"
        assert _service_layer_reach(api_stub, storefront_stub) == {
            "start_workflow_run"
        }, f"{node_type.__name__} binds a local the reader cannot follow"

    # the negative half of the same claim: with the binding removed and the
    # call left in place, the reader reports nothing — so the assertions above
    # measure the binding and not the presence of the call
    assert _service_layer_reach(
        api_stub,
        "class Storefront:\n"
        "    def enqueue_turn(self, ctx):\n"
        "        return svc.start_workflow_run(ctx)\n",
    ) == set()

    # sequence destructuring, which is `ast.Assign` in a shape `_bind_pairs`
    # has to pair positionally rather than by unparsing one value
    assert _service_layer_reach(
        api_stub,
        "class Storefront:\n"
        "    def enqueue_turn(self, ctx):\n"
        "        svc, _ = self._services, None\n"
        "        return svc.start_workflow_run(ctx)\n",
    ) == {"start_workflow_run"}
    # ...and the shape it declines to pair, because pairing it would need the
    # value: a starred target. Declined loudly here rather than guessed.
    assert _service_layer_reach(
        api_stub,
        "class Storefront:\n"
        "    def enqueue_turn(self, ctx):\n"
        "        svc, *_ = self._services, None\n"
        "        return svc.start_workflow_run(ctx)\n",
    ) == set()

    # the three excluded forms an editor is most likely to reach for, each
    # measured rather than asserted in prose: what they bind is not the
    # service object, so the reader reporting nothing is correct, not a miss
    for body in (
        # `ast.arg` — the documented `passed to a helper` non-reach
        "    def enqueue_turn(self, ctx):\n"
        "        return _go(self._services, ctx)\n",
        # `ast.ExceptHandler` — binds the caught exception
        "    def enqueue_turn(self, ctx):\n"
        "        try:\n"
        "            return self._services\n"
        "        except Exception as svc:\n"
        "            return svc.start_workflow_run(ctx)\n",
        # `ast.AugAssign` — binds `svc + self._services`, not `self._services`
        "    def enqueue_turn(self, ctx):\n"
        "        svc = 0\n"
        "        svc += self._services\n"
        "        return svc.start_workflow_run(ctx)\n",
    ):
        assert _service_layer_reach(api_stub, "class Storefront:\n" + body) == set()


# Every exception class a `raise` in `storefront.py` names **today**, spelled
# out for the same reason `SERVICE_LAYER_REACH_TODAY` is: a `/shop/api` route
# executes this module, so a `raise` added to it has to come back here.
#
# All seven are `StorefrontError` subclasses, and that is what makes
# `INHERITED_HANDLERS`' excuses true of this file rather than merely stated for
# it: `test_every_storefront_error_subclass_is_mapped_to_a_response` already
# holds that every member of that family is caught by a route or answered by a
# classified handler, so none of them can arrive at a handler this table
# excuses — and none of them is a bare `HTTPException`.
#
# `ResetStateUnknownError` is here because the reader resolves the factory:
# `reset_participant` writes `raise self._reset_state_unknown(...)`, and the
# name in that `raise` is a method.
STOREFRONT_RAISES_TODAY = frozenset({
    "DemoNotSeededError", "QuiesceTimeoutError", "UnknownParticipantError",
    "UnscopedParticipantError", "ResetStateUnknownError",
    "UnknownOrderError", "OrderTransitionRefusedError",
})


# Everything the *reached* methods of the two shared collaborators raise
# (P13-2), where "reached" is `_reached_methods`: the methods the two guards
# above name, closed over the `self.<name>` calls those make (P14-1). Not whole
# modules: `services.py` and `repository.py` are shared with the legacy
# surface, so only the methods a `/shop/api` request actually executes are read.
#
# **Five, not one**, and that is the measurement P14-1 turned on: the nine
# reached `Services` methods delegate to four siblings on the request path
# today, and a walk that stopped at the nine reported one class while a route
# could reach five. `ThreadNotFoundError`, `UnknownActorError` and
# `UnknownMemberError` (from `_validate_and_derive_role` / `_dispatch_write`)
# are `SERVICE_ERROR_RESPONSES` rows — mapped, not excused.
# `UnknownOrderTransitionError` is `services.advance_order`'s guard on an
# unknown transition string, already in `SERVICE_ERRORS_UNREACHABLE` with a
# behavioural reason (§5.3 C11: `AdvanceOrderIn.transition` is a `Literal` of
# exactly the three it accepts, so `422 validation_failed` answers first) that
# has its own producer test. So four of the five are exempted here by a
# mechanism rather than by appearing in a list; `RuntimeError` is not a
# `ServiceError` at all and takes a written reason in `NON_FAMILY_RAISES`.
SERVICE_RAISES_TODAY = frozenset({
    "RuntimeError", "ThreadNotFoundError", "UnknownActorError",
    "UnknownMemberError", "UnknownOrderTransitionError",
})

# `Repository.ensure_participant`'s namespace refusal, reached from
# `Storefront.join` on `POST /shop/api/session`. It was in no table anywhere
# while the repository leg was read at the router's two calls only (P14-1); the
# reason it is classified rather than handled is in `NON_FAMILY_RAISES`.
REPOSITORY_RAISES_TODAY = frozenset({"MemberIdCollisionError"})

# The raises the four scopes report that belong to **neither** exception
# family, each with the reason it is not a `(route, response)` pair.
#
# The three `*_RAISES_TODAY` constants are allowlists, and extending one is the
# cheapest way to silence the guard that reads it — which is what P13-3's
# cross-check closed for `storefront.py` and what the same commit left open on
# the two constants it introduced (`## Pass 14`, P14-4). The close is one
# mechanism for all three legs rather than three: a name in a family is
# exempted by that family's own partition test, and a name outside every family
# has no such argument and must carry its own reason **here**, asserted as an
# equality so a reason with no raise behind it reddens too.
#
# Both entries are internal-invariant alarms, not participant-facing outcomes.
# Answering either with a plan token would tell a participant that something
# they can act on went wrong, and would take the alarm away from the operator,
# who is the only party that can do anything about it. This is `app.py`'s own
# posture at startup, where `MemberIdCollisionError` is left to abort loudly
# "instead of silently shadowing it (DEF-1)".
NON_FAMILY_RAISES: dict[str, str] = {
    "RuntimeError": (
        "`services._dispatch_write`'s two invariant alarms — an unrecognised "
        "write-status row, and a retry loop that did not converge. Neither is "
        "a state a request can put the write path into: both mean the "
        "repository returned a row shape the service layer's own contract "
        "rules out, which is a defect report, not a response"
    ),
    "MemberIdCollisionError": (
        "`repository.ensure_participant`'s refusal to provision a participant "
        "over an id the `Agent`/`User` namespace already holds. **No request "
        "can reach it**: the id is `_default_participant_id()`, `'p-' + "
        "uuid4().hex`, minted server-side with no client input, and no caller "
        "pins `id_gen` (`test_create_app_never_pins_the_participant_id_"
        "generator`, `tests/test_app.py`), so reaching any of its three "
        "branches needs a pre-existing `Agent`, or a `User` with no "
        "`tokenHash`, at that exact freshly-minted uuid4. Measured rather "
        "than argued: raised through `POST /shop/api/session` it answers a "
        "bare `500 text/plain 'Internal Server Error'`, which is the right "
        "answer for a message whose own text says `manual repair required` "
        "and the wrong one for anything a participant could cause"
    ),
}


def test_the_raises_a_route_can_reach_are_exactly_what_the_exemptions_assume():
    """**P11-2, and P11-1's other half.**

    `ENVELOPE_HANDLERS` are the two handlers that re-shape a response the route
    itself declares; every other handler on the app either produces one of the
    three cross-cutting rows or is excused by `INHERITED_HANDLERS`. So a route
    body that raises anything else is an unruled `(route, response)` by
    construction, and neither half of the gate can see it: the declaration half
    compares declarations against the table, and the handler half only crosses
    the cross-cutting three.

    Two mutations proved the hole, both green through the whole file:
    **N-E** — `raise HTTPException(status_code=410, detail="gone")` in `join`,
    which answered `410 {"detail":"gone"}` with no `error` token at all
    (`HTTPException` is imported here, and `_raised_refusals` collects only
    `StorefrontHTTPError` calls, so it sees nothing). **N-F/N-F2** — a route
    raising `WorkflowEngineDisabledError` / `SearchNotAvailableError`, which
    reach the inherited handlers `INHERITED_HANDLERS` excuses.

    **The scope is both storefront modules, whole** (P12-2). The reason string
    says "*no `/shop/api` route raises a bare `HTTPException`*", which is a
    claim about what a request can reach, not about lexical position — and a
    walk of `build_storefront_router` left the escape one call out: mutation
    N-M, a module-level `_refuse_retired_name()` in *this* file raising
    `HTTPException(410)` and called from `join`, survived and answered
    `410 {"detail":"gone"}` on the wire.

    Widening to `storefront_api.py` whole closes that one and no more. The same
    helper written in `storefront.py` (**N-M3**), and the `raise` put straight
    into a `Storefront` method a route calls (**N-M2**), both survive a
    whole-module walk of this file, and both answer the identical
    `410 {"detail":"gone"}` from `POST /shop/api/session`: a route executes
    `shop.<method>` exactly as it executes a local helper.

    So the unit is the module for the storefront's own two files, which are
    read whole and not filtered by reachability: both exist only to serve
    `/shop/api`, so a `raise` anywhere in either is one the next edit can put on
    a request path, and over-approximating costs a named line in a list while
    under-approximating is the same defect a second time.

    **The walk does not stop there, because the request does not** (P13-2).
    This docstring used to say it stopped at `services.py`, "which the reach
    guard above covers instead" — false in exactly the way this guard keeps
    being false: that guard measures **which** `Services` methods a route
    reaches, never **what they raise**. A bare `HTTPException(410)` on a dead
    branch of `services.save_profile` — one of the nine names it itself lists,
    reached from `Storefront.join` — survived the whole file at 183 passed and
    answered the identical `410 {"detail":"gone"}` from
    `POST /shop/api/session`. So the two guards are **composed**.

    **And the composition is closed under `self.<name>`, in both directions**
    (`## Pass 14`, P14-1). Composing at the reach guard's own frontier left
    the walk exactly one hop short of its sentence, twice:
    `Services._refuse_retired_name()` raising `HTTPException(410)` and called
    from the reached `save_profile`, and the same raise in
    `Repository.ensure_participant` reached from `Storefront.join`, each
    **survived at 183 passed** and each answered `410 {"detail":"gone"}` from
    `POST /shop/api/session`. Both legs are therefore seeded with the reached
    set and closed over `self.<name>` to a fixpoint (`_reached_methods`), and
    the repository leg reads `self._repo.<name>` in the reached `Storefront`
    methods as well as `repo.<name>` in the router (`_repository_reach`). The
    numbers moved with the reach and are the point of the fix: `Services` 9
    methods to 13 and one raised class to five, `Repository` 2 methods to 7 and
    zero raised classes to one — `MemberIdCollisionError`, which until then
    appeared in no §5.2/§5.3 row, no `SERVICE_ERROR_RESPONSES` entry and no
    `INHERITED_HANDLERS` excuse.

    Neither collaborator is read whole — both are shared with the legacy
    surface, and only the methods a request executes are on this path.

    Composition is the point, not economy: the closure runs from each seed
    over its own `self.<name>` callees, so a class raised in a helper a reached
    method only *calls* is inside the walk rather than a prose argument.

    **This walk reports nothing new at S9, and that is stated rather than
    forecast away.** Under `docs/plans/salesperson-ui.md` v1.25 the trigger
    runs on the turn worker and reaches the workflow layer through
    `trigger.maybe_trigger`, whose `start_workflow_run` call site
    (`trigger.py:82`) is outside all four scopes — so `start_workflow_run`
    never enters the seed set and none of the three workflow classes enters
    this one. S9's evidence is §5.1's S9 row's armed-fault measurement at the
    response boundary, not anything this test can see. The version of this
    paragraph that predicted otherwise was written to fix a docstring making a
    claim about the plan the plan contradicts, and reproduced it
    (`## Pass 14`, P14-3; `## Pass 15`, P15-2).

    **Where it does stop, stated as node types rather than as intent:** at
    calls that are not `self.<name>` on the walked class — `Services` into
    `Repository`, `Repository` into redis. None of that is walked, and none of
    it is excused here either **except `WorkflowConfigError`**, whose raise
    sites are `guards.py`'s fourteen, reached through
    `Services` → `self._executor` → `executor` → `guards` — squarely this
    unwalked region — and which no mechanism in either file checks
    (`## Pass 15`, P15-3). For the rest: the graph faults are S8's typed
    handlers' own rows, and the `ServiceError` family is covered by the
    `SERVICE_ERROR_RESPONSES` / `SERVICE_ERRORS_UNREACHABLE` partition, which
    `WorkflowConfigError` is outside on both counts. A `raise` a route can
    reach from outside all four scopes is not something this test can see, and
    no reason string may claim otherwise without saying that it is prose.

    Four scopes are pinned. Inside the router: exactly the two envelope
    classes. `storefront_api.py` whole: those two, plus the three raises that
    happen at wiring or boot time and can never be on a request path.
    `storefront.py` whole: `STOREFRONT_RAISES_TODAY`, seven `StorefrontError`
    subclasses and nothing else. The reached-and-closed methods of the two
    collaborators: `SERVICE_RAISES_TODAY` and `REPOSITORY_RAISES_TODAY`, whose
    members outside both families carry a written reason in
    `NON_FAMILY_RAISES` — asserted as an equality, so neither an extended
    allowlist nor a stale reason passes (P14-4).

    This is what makes `INHERITED_HANDLERS[StarletteHTTPException]`'s reason
    true rather than merely stated.
    """
    source = _router_source()
    envelope = {klass.__name__ for klass in ENVELOPE_HANDLERS}
    assert envelope == {"StorefrontHTTPError", "RequestValidationError"}

    assert _raised_class_names(_parse_router(source)) == envelope

    # Module-wide, and the three extras are named rather than tolerated:
    # `register_storefront_error_handlers`'s `RuntimeError` refuses a missing
    # incumbent at wiring time and `storefront_preflight` refuses a mis-seeded
    # workspace at boot — neither is on a request path at all — while
    # `_nonblank`'s `ValueError` is, but is raised inside a pydantic
    # `field_validator`, which absorbs it and re-emits it as the
    # `RequestValidationError` already in `envelope`. So none of the three can
    # reach an exception handler under its own name.
    assert _raised_class_names(source) == envelope | {
        "RuntimeError", "StorefrontPreflightError", "ValueError",
    }

    # ...and `storefront.py`, whose raises a route reaches through
    # `shop.<method>` exactly as it reaches this file's own helpers
    storefront_raises = _raised_class_names(_storefront_source())
    assert storefront_raises == set(STOREFRONT_RAISES_TODAY)

    # ...and the two shared collaborators, at the methods a request reaches —
    # `services.py` is on the request path and nothing here read it (P13-2) —
    # each seeded with that collaborator's reach and closed over `self.<name>`
    # by `_raises_of` (P14-1)
    service_raises = _raises_of(
        _services_source(), "Services", SERVICE_LAYER_REACH_TODAY
    )
    assert service_raises == set(SERVICE_RAISES_TODAY)

    repo_reach = _repository_reach(source, _storefront_source())
    assert repo_reach, (
        "no `/shop/api` route reaches `Repository` at all any more — S10 moves "
        "the router's two `repo.<name>` calls onto `Storefront`, which this "
        "reader follows, so an empty reach means the leg lost its seed rather "
        "than that the calls moved"
    )
    repo_raises = _raises_of(_repository_source(), "Repository", repo_reach)
    assert repo_raises == set(REPOSITORY_RAISES_TODAY)

    # P13-3, applied to all three allowlists by one mechanism instead of one
    # (P14-4). Each leg's exemption argument is "every one of these is a member
    # of a family whose members are all already classified"; the family halves
    # have their own partition tests, the subset halves had nothing but the
    # `storefront.py` one. Extending an allowlist is the cheapest way to
    # silence the three assertions above, and only `HTTPException` is
    # separately fenced.
    storefront_family = {
        klass.__name__ for klass in _subclasses(storefront.StorefrontError)
    }
    service_family = {klass.__name__ for klass in _subclasses(ServiceError)}
    assert set(STOREFRONT_RAISES_TODAY) <= storefront_family
    # ...and an **equality**, not a subset, over all four scopes at once: a
    # raise outside both families has to carry its own reason, and a reason
    # left behind by a raise that is gone reddens the same assertion.
    assert (
        (storefront_raises | service_raises | repo_raises)
        - storefront_family - service_family
    ) == set(NON_FAMILY_RAISES)
    assert all(reason.strip() for reason in NON_FAMILY_RAISES.values())

    # the bare `HTTPException` half, named separately because it is the one the
    # reason string cites and the one a reflex reaches for — over all four
    # scopes, since `410 {"detail":"gone"}` reaches the participant identically
    # from any of them (N-M, N-M2, N-M3, P13-A)
    assert "HTTPException" not in (
        _raised_class_names(source)
        | storefront_raises
        | service_raises
        | repo_raises
    )

    # the controls: the reader resolves both mutation shapes it is shown
    assert _raised_class_names(
        "def build_storefront_router(shop):\n"
        "    def join(body):\n"
        "        raise HTTPException(status_code=410, detail='gone')\n"
    ) == {"HTTPException"}
    assert _raised_class_names(
        "def build_storefront_router(shop):\n"
        "    def join(body):\n"
        "        raise WorkflowEngineDisabledError('engine off')\n"
    ) == {"WorkflowEngineDisabledError"}

    # and N-M's shape, which is the whole reason the scope widened: the raise
    # is module-level, so the router-scoped walk reports nothing about it
    n_m = (
        "def _refuse_retired_name(name):\n"
        "    raise HTTPException(status_code=410, detail='gone')\n"
        "def build_storefront_router(shop):\n"
        "    def join(body):\n"
        "        return _refuse_retired_name(body.displayName)\n"
    )
    assert _raised_class_names(n_m) == {"HTTPException"}
    assert _raised_class_names(_parse_router(n_m)) == set()

    # the two shapes a walk of `storefront_api.py` alone still could not see,
    # both of them measured survivors of this file (N-M2, N-M3)
    assert _raised_class_names(
        "class Storefront:\n"
        "    def join(self, display_name):\n"
        "        raise HTTPException(status_code=410, detail='gone')\n"
    ) == {"HTTPException"}
    assert _raised_class_names(
        "def _refuse_retired_name(name):\n"
        "    raise HTTPException(status_code=410, detail='gone')\n"
        "class Storefront:\n"
        "    def join(self, display_name):\n"
        "        return _refuse_retired_name(display_name)\n"
    ) == {"HTTPException"}

    # ...and the factory resolution, which keeps a method name out of a set of
    # exception classes and, more to the point, is not blind to a factory that
    # builds the `HTTPException` one call away from the `raise`
    factory = (
        "class Storefront:\n"
        "    def reset_participant(self):\n"
        "        raise self._reset_state_unknown()\n"
        "    def _reset_state_unknown(self):\n"
        "        return {}\n"
    )
    assert _raised_class_names(
        factory.replace("return {}", "return ResetStateUnknownError('p-1')")
    ) == {"ResetStateUnknownError"}
    assert _raised_class_names(
        factory.replace("return {}", "return HTTPException(status_code=410)")
    ) == {"HTTPException"}

    # P13-A's shape — a raise in a *reached* collaborator method, which no
    # scope before P13-2 read; and the control that `_raises_of` really is
    # selective, so the composed leg cannot silently read the wrong methods
    collaborator = (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        raise HTTPException(status_code=410, detail='gone')\n"
        "    def unreached(self, ctx):\n"
        "        raise SomethingElse()\n"
    )
    assert _raises_of(collaborator, "Services", {"save_profile"}) == {"HTTPException"}
    assert _raises_of(collaborator, "Services", {"unreached"}) == {"SomethingElse"}
    with pytest.raises(AssertionError, match="has no method"):
        _raises_of(collaborator, "Services", {"renamed_away"})

    # P14-M1's shape — the raise in a **sibling** of a reached method, which
    # the composed walk missed while it stopped at the reach guard's frontier.
    # The alias control on the same leg is the second one: `self` is resolved
    # here exactly as it is on the four legs of the reach guard.
    sibling = (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return self._refuse_retired_name(ctx)\n"
        "    def _refuse_retired_name(self, ctx):\n"
        "        raise HTTPException(status_code=410, detail='gone')\n"
        "    def unreached(self, ctx):\n"
        "        raise SomethingElse()\n"
    )
    assert _raises_of(sibling, "Services", {"save_profile"}) == {"HTTPException"}
    assert _raises_of(
        sibling.replace(
            "        return self._refuse_retired_name(ctx)\n",
            "        me = self\n        return me._refuse_retired_name(ctx)\n",
        ),
        "Services", {"save_profile"},
    ) == {"HTTPException"}
    # ...and it is still selective: the sibling `unreached` raises are not
    # swept in just because the class defines them
    assert "SomethingElse" not in _raises_of(sibling, "Services", {"save_profile"})

    # `raise self.<factory>()` across a sibling, which is the shape the
    # enumeration probe caught: read one method at a time this resolved to
    # `{"_mk"}` — the method's own name — while the identical source read
    # whole-module in `storefront.py` resolves to the class. The two scopes
    # now agree.
    assert _raises_of(
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        raise self._mk()\n"
        "    def _mk(self):\n"
        "        return HTTPException(status_code=410)\n",
        "Services", {"save_profile"},
    ) == {"HTTPException"}

    # P14-M2's shape — the same raise in a `Repository` method reached from a
    # `Storefront` method rather than from the router, which is the leg
    # `_repository_reach` added. Both halves are controlled: the reader finds
    # the method, and finding it is what reports the raise.
    repo_stub = (
        "class Repository:\n"
        "    def ensure_participant(self, ws):\n"
        "        raise HTTPException(status_code=410, detail='gone')\n"
        "    def never_called(self, ws):\n"
        "        raise SomethingElse()\n"
    )
    api_repo_stub = (
        "def build_storefront_router(shop):\n"
        "    def join(body):\n"
        "        return shop.join(body)\n"
    )
    sf_repo_stub = (
        "class Storefront:\n"
        "    def join(self, name):\n"
        "        return self._repo.ensure_participant(self._ws)\n"
    )
    assert _repository_reach(api_repo_stub, sf_repo_stub) == {"ensure_participant"}
    assert _raises_of(
        repo_stub, "Repository",
        _repository_reach(api_repo_stub, sf_repo_stub),
    ) == {"HTTPException"}
    # the router leg of the same reader, which is what S10 empties
    assert _repository_reach(
        api_repo_stub.replace(
            "        return shop.join(body)\n",
            "        repo = shop._repo\n"
            "        return repo.list_participants()\n",
        ),
        "class Storefront:\n    def unrelated(self):\n        return None\n",
    ) == {"list_participants"}


# Every syntactic route from a **reached** collaborator method to a `raise`,
# and every one the walk stops at, as `{case: (source, reaches the raise?)}`.
# The sibling of `_ALIAS_FORM_SNIPPETS`: that one enumerates the ways a call
# can be spelled, this one the ways a raise can be reached, and both exist
# because the defect this artifact keeps producing is a sentence wider than a
# walk, which only an enumeration closes (`## Pass 14`, the closing condition).
_RAISE_ROUTES: dict[str, tuple[str, bool]] = {
    "in the seed method itself": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        raise HTTPException(410)\n", True),
    "in a nested function of the seed": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        def inner():\n"
        "            raise HTTPException(410)\n"
        "        return inner\n", True),
    "via `self.<sibling>()`": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return self._h(ctx)\n"
        "    def _h(self, ctx):\n"
        "        raise HTTPException(410)\n", True),
    "via an alias of `self`": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        me = self\n"
        "        return me._h(ctx)\n"
        "    def _h(self, ctx):\n"
        "        raise HTTPException(410)\n", True),
    "via an annotated alias of `self`": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        me: object = self\n"
        "        return me._h(ctx)\n"
        "    def _h(self, ctx):\n"
        "        raise HTTPException(410)\n", True),
    "two siblings deep, which is why the closure is a fixpoint": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return self._a(ctx)\n"
        "    def _a(self, ctx):\n"
        "        return self._b(ctx)\n"
        "    def _b(self, ctx):\n"
        "        raise HTTPException(410)\n", True),
    "mutually recursive siblings, which the fixpoint must terminate on": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return self._a(ctx)\n"
        "    def _a(self, ctx):\n"
        "        return self._b(ctx)\n"
        "    def _b(self, ctx):\n"
        "        raise HTTPException(410)\n"
        "        return self._a(ctx)\n", True),
    "a `staticmethod` sibling called as `self.<name>`": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return self._h()\n"
        "    @staticmethod\n"
        "    def _h():\n"
        "        raise HTTPException(410)\n", True),
    "`raise self.<factory>()`, resolved through the factory's returns": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        raise self._mk()\n"
        "    def _mk(self):\n"
        "        return HTTPException(410)\n", True),
    # ...and the stops, each one a sentence in the comment above
    # `INHERITED_HANDLERS` rather than an omission
    "STOP — a bare re-raise names no class, by design": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        try:\n"
        "            pass\n"
        "        except Exception:\n"
        "            raise\n", False),
    "STOP — a module-level helper of the collaborator file": (
        "def _h():\n"
        "    raise HTTPException(410)\n"
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return _h()\n", False),
    "STOP — another object the collaborator calls (`self._repo.x()`)": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return self._repo.x(ctx)\n", False),
    "STOP — a sibling nothing reached calls": (
        "class Services:\n"
        "    def save_profile(self, ctx):\n"
        "        return None\n"
        "    def _other(self):\n"
        "        raise HTTPException(410)\n", False),
}


def test_the_raise_walk_reaches_every_route_from_a_reached_method_it_claims():
    """**The other half of the enumeration** (`## Pass 14`, the closing
    condition), for the second sentence.

    `_ALIAS_FORM_SNIPPETS` enumerates the ways the *call* can be spelled;
    this enumerates the ways the *raise* can be reached from a method a route
    runs. Both sentences above `INHERITED_HANDLERS` are of the shape that has
    failed here five times — a scope stated more widely than the walk — and the
    only thing that closes one is a list of the forms it covers, run against
    the delivered reader, with the stops written down beside them rather than
    discovered by the next gate.

    The stops are the load-bearing half. A module-level helper in `services.py`
    is *not* read, where the same helper in `storefront.py` is — because those
    two files are read whole and the collaborators are read at the reached
    methods only, both files being shared with the legacy surface. That
    asymmetry is a decision (`## Pass 13`, P13-2's closure (a)), and this is
    where it is visible as a measurement instead of as a paragraph.
    """
    for label, (source, reaches) in _RAISE_ROUTES.items():
        found = _raises_of(source, "Services", {"save_profile"})
        assert ("HTTPException" in found) is reaches, (
            f"{label}: the walk {'missed' if reaches else 'reached'} a raise "
            f"it says it {'reaches' if reaches else 'stops before'} — {found}"
        )


def _raised_refusals() -> dict[str, set[tuple[int, str]]]:
    """`{route function name: {(status, token)}}` — every `StorefrontHTTPError`
    the router can raise, read off the parsed source.

    Both argument shapes the router uses are resolved: a literal token, and
    `<StorefrontError subclass>.code`, whose value is the plan's own name for
    that response (`unscoped_participant`, `reset_state_unknown`).

    It collects `StorefrontHTTPError` **only**, which is what it is for and is
    also what made it unable to see a bare `HTTPException` (P11-2). That gap is
    closed by `test_the_raises_a_route_can_reach_are_exactly_what_the_exemptions_assume`
    above, not here.
    """
    builder = _parse_router(_router_source())
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

    Skipped whenever the caller ran a subset of this module, which cannot cover
    the whole table by construction; the full run is where it does its work, and
    the `⊆` direction above holds unconditionally either way.

    **The predicate is what was collected, not how it was selected** (P11-9).
    Keying on `-k` alone let a node-id selection through: `pytest <file>::<this
    test>` — which is exactly what `pytest --lf` re-runs after any failure —
    leaves `config.option.keyword` empty, so this ran against an almost-empty
    `_OBSERVED` and reported all 57 rows as unproducible. Every selector
    (`-k`, a node id, `--lf`, `--deselect`) shows up the same way here: a test
    function of this module that pytest did not collect.

    **Marker deselection is the one selector that must not count** (P12-3).
    `-m "not live"` is in this project's `addopts`, so the first
    `@pytest.mark.live` test added to this module would deselect *itself* on
    every default run and turn this check off permanently — visibly, since the
    skip reason prints, but off. Zero such tests exist today, which makes it
    latent rather than live. So a `live`-marked function is not part of
    `defined`, and the predicate reads *every offline test of this module was
    collected* — exactly what covering the table requires. Under `-m live` this
    test is itself deselected and never asks the question; under a marker-free
    run the live ones collect, are absent from `defined`, and are simply extra.
    """
    module = sys.modules[__name__]
    defined = {
        name
        for name in dir(module)
        if name.startswith("test_")
        and not any(
            mark.name == "live"
            for mark in getattr(getattr(module, name), "pytestmark", ())
        )
    }
    collected = {
        getattr(item, "originalname", None) or item.name.split("[")[0]
        for item in request.session.items
        if getattr(item, "module", None) is module
    }
    uncollected = defined - collected
    if uncollected:
        pytest.skip(
            f"{len(uncollected)} of this module's tests were not "
            "collected, so the run cannot cover the whole table"
        )
    missing = FLAT_TABLE - _OBSERVED
    assert missing == set(), (
        f"rows of §5.3's completeness table that no test in this file ever "
        f"provoked: {sorted(missing)} — declared, tabled, and not proved "
        "producible"
    )
