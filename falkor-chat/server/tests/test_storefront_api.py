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

**What neither half closes, stated because the residue ships** (§5.3 C13's own
scoring): a route that `return`s a `JSONResponse(status_code=…)` never raises,
so it is invisible to the handler set; and a declaration is an enumeration that
can be wrong in both directions. What narrows it here is that every declared
entry is proved *producible* by a contract test below, so the declaration and
the implementation disagree loudly instead of agreeing by omission.
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
from falkorchat.services import Services
from falkorchat.storefront_api import (
    API_PREFIX,
    CROSS_CUTTING_HANDLERS,
    ENVELOPE_HANDLERS,
    ROUTE_CLASSES,
    StorefrontPreflightError,
    cross_cutting_response,
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
# The gate
# ═══════════════════════════════════════════════════════════════════════════


def _handler_signature(app) -> dict:
    """`{exception type: handler qualname}` for every handler on `app`.

    Keyed on the qualname rather than the function object because two apps
    built by two `create_app` calls hold two distinct closures for the same
    registration — while an *override* of a handler FastAPI installs by default
    (`RequestValidationError`) changes the value and not the key, and would be
    invisible to a key-only diff.
    """
    return {
        exc: getattr(handler, "__qualname__", repr(handler))
        for exc, handler in app.exception_handlers.items()
    }


def registered_storefront_handlers(app) -> set:
    """The handlers this app carries that a non-storefront app does not.

    Computed as a difference against a real baseline app rather than compared
    against a hand-listed set, so a handler added anywhere in the storefront
    branch — including one that shadows a FastAPI default — shows up here
    whether or not anybody remembered to classify it.
    """
    baseline = _handler_signature(create_app(dev_surface=False))
    return {
        exc
        for exc, qualname in _handler_signature(app).items()
        if baseline.get(exc) != qualname
    }


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


def handler_produced_pairs(app) -> set[tuple[str, str, int, str]]:
    """The handler half's cross product: registered cross-cutting handlers ×
    the routes their class permits them on.

    Raises before computing anything if a handler is registered that is neither
    a cross-cutting producer nor an envelope — *a handler with no row fails the
    step*, and this is where that is enforced.
    """
    registered = registered_storefront_handlers(app)
    classified = set(CROSS_CUTTING_HANDLERS) | set(ENVELOPE_HANDLERS)
    unclassified = registered - classified
    if unclassified:
        raise AssertionError(
            "handler(s) registered on the storefront app with no classification "
            f"in `storefront_api`: {sorted(e.__name__ for e in unclassified)} — "
            "a handler must declare whether it produces one of §5.3's three "
            "cross-cutting responses or merely re-shapes a declared one"
        )
    missing = classified - registered
    if missing:
        raise AssertionError(
            f"classified but not registered: {sorted(e.__name__ for e in missing)}"
        )

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
    produced = handler_produced_pairs(app)

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
