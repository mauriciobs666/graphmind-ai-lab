"""`packs/tool-caller-shop-assistant/tools/sim.py` — S5 spec §3.3/§4 Step 1.

Loads the pack's tool module the same way `modelbench.packs.Pack.load_tool_module` does
(`importlib.util.spec_from_file_location`), not via a package import — the module lives under
`packs/`, not `modelbench/`, and is never imported as part of this package's own import graph
(the AST allowlist exists precisely because a pack module is loaded this way at run time).

Three groups of tests, per the S5 spec's Step 1:

(a) each tool's ordinary behaviour against a small in-memory catalog fixture;
(b) E1 — the dispatch-totality property test: every tool in the real `schemas()` crossed with an
    adversarial argument set, asserting zero raises and exactly one `DispatchRecord` per call;
(c) the `boundaryRule` extension round-trips off the real `tools/schemas.json`, for one
    $/cents-shaped parameter (`maxPrice`) and one inclusive/exclusive-bound-shaped parameter
    (`minPrice`).

Plus the AST import allowlist, run directly against `tools/sim.py` per the spec's own "Done when"
clause (Step 1): `validate_pack`'s full pass needs `pack.json`/`conversations.jsonl`, neither of
which exists yet (Step 2/S6), so this calls `modelbench.packs._tool_import_problems` — the one
axis that only needs a pack root and does not touch the manifest — directly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import pytest

from modelbench import packs
from modelbench.tooling import DispatchRecord

PACK_ROOT = Path(__file__).parent.parent / "packs" / "tool-caller-shop-assistant"
SIM_PATH = PACK_ROOT / "tools" / "sim.py"
SCHEMAS_PATH = PACK_ROOT / "tools" / "schemas.json"
CATALOG_PATH = PACK_ROOT / "catalog.json"


def _load_sim_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("modelbench_test_tool_caller_sim", SIM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sim() -> ModuleType:
    return _load_sim_module()


# --------------------------------------------------------------------------------------------
# AST import allowlist — Step 1's own "Done when" clause
# --------------------------------------------------------------------------------------------


def test_sim_module_imports_nothing_beyond_stdlib_and_modelbench_tooling() -> None:
    """`validate_pack`'s AST-allowlist axis, run directly against `tools/sim.py` — the pack has no
    `pack.json` yet (Step 2), so the full `validate_pack` pass cannot run; this calls the one axis
    that only needs a pack root."""
    fake_pack = packs.Pack(
        packId="tool-caller-shop-assistant",
        packVersion="0.0.0-step1",
        role="tool-caller",
        contentHash="",
        manifest={},
        root=PACK_ROOT,
    )
    assert packs._tool_import_problems(fake_pack) == []


# --------------------------------------------------------------------------------------------
# (a) ordinary tool behaviour against a small in-memory catalog fixture
# --------------------------------------------------------------------------------------------

_FIXTURE_SCHEMAS: list[dict[str, Any]] = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))

_FIXTURE_CATALOG: dict[str, dict[str, Any]] = {
    "Widget": {
        "productId": "widget", "name": "Widget", "category": "Gadgets", "price": 9.99, "stock": 5,
    },
    "Gizmo": {
        "productId": "gizmo", "name": "Gizmo", "category": "Gadgets", "price": 49.99, "stock": 2,
    },
    "Sprocket": {
        "productId": "sprocket", "name": "Sprocket", "category": "Parts", "price": 3.50, "stock": 0,
    },
}


def _fresh_env(sim: ModuleType):
    return sim.ShopEnvironment(catalog=dict(_FIXTURE_CATALOG), schemas=_FIXTURE_SCHEMAS)


def test_lookup_product_fact_found(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("lookup_product_fact", {"name": "Widget"})
    assert result == {"found": True, "category": "Gadgets", "price": 9.99}


def test_lookup_product_fact_not_found(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("lookup_product_fact", {"name": "Nonexistent Product"})
    assert result == {"found": False}


def test_filter_products_by_category(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("filter_products", {"category": "Gadgets"})
    assert {item["name"] for item in result["items"]} == {"Widget", "Gizmo"}


def test_filter_products_by_price_range_is_inclusive_at_both_ends(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("filter_products", {"minPrice": 3.50, "maxPrice": 9.99})
    assert {item["name"] for item in result["items"]} == {"Widget", "Sprocket"}


def test_filter_products_no_match_reports_a_finding(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("filter_products", {"category": "Nope"})
    assert result == {"items": [], "finding": "no matching products found"}


def test_add_to_cart_then_view_cart_computes_live_total(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    env.dispatch("add_to_cart", {"productName": "Widget", "quantity": 2})
    result = env.dispatch("view_cart", {})
    assert result == {"items": [{"name": "Widget", "quantity": 2, "price": 9.99}], "total": 19.98}


def test_add_to_cart_accumulates_quantity_across_calls(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    env.dispatch("add_to_cart", {"productName": "Widget", "quantity": 1})
    result = env.dispatch("add_to_cart", {"productName": "Widget", "quantity": 2})
    assert result == {"found": True, "productName": "Widget", "quantity": 3}


def test_add_to_cart_defaults_quantity_to_one_when_omitted(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("add_to_cart", {"productName": "Widget"})
    assert result == {"found": True, "productName": "Widget", "quantity": 1}


def test_add_to_cart_unknown_product_abstains(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("add_to_cart", {"productName": "Nonexistent"})
    assert result == {"found": False}


def test_add_to_cart_rejects_exceeding_stock(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("add_to_cart", {"productName": "Gizmo", "quantity": 5})
    assert result == {"error": "insufficient-stock", "available": 2}


def test_add_to_cart_zero_stock_product_always_exceeds(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("add_to_cart", {"productName": "Sprocket", "quantity": 1})
    assert result == {"error": "insufficient-stock", "available": 0}


def test_remove_from_cart_partial_quantity(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    env.dispatch("add_to_cart", {"productName": "Widget", "quantity": 3})
    result = env.dispatch("remove_from_cart", {"productName": "Widget", "quantity": 1})
    assert result == {"found": True, "removed": True, "productName": "Widget", "quantityRemoved": 1}
    assert env.dispatch("view_cart", {})["items"][0]["quantity"] == 2


def test_remove_from_cart_whole_line_when_quantity_omitted(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    env.dispatch("add_to_cart", {"productName": "Widget", "quantity": 3})
    result = env.dispatch("remove_from_cart", {"productName": "Widget"})
    assert result == {"found": True, "removed": True, "productName": "Widget", "quantityRemoved": 3}
    assert env.dispatch("view_cart", {})["items"] == []


def test_remove_from_cart_known_product_never_added_is_found_but_not_removed(
    sim: ModuleType,
) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("remove_from_cart", {"productName": "Widget"})
    assert result == {"found": True, "removed": False, "productName": "Widget"}


def test_remove_from_cart_unknown_product_abstains(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("remove_from_cart", {"productName": "Nonexistent"})
    assert result == {"found": False}


def test_clear_cart_empties_it(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    env.dispatch("add_to_cart", {"productName": "Widget", "quantity": 1})
    result = env.dispatch("clear_cart", {})
    assert result == {"cleared": True}
    assert env.dispatch("view_cart", {}) == {"items": [], "total": 0.0}


def test_place_order_on_empty_cart_returns_an_explanatory_string(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("place_order", {})
    assert result == "The cart is empty — add an item before placing an order."


def test_place_order_moves_cart_contents_into_an_order_and_clears_the_cart(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    env.dispatch("add_to_cart", {"productName": "Widget", "quantity": 2})
    order = env.dispatch("place_order", {})
    assert order["total"] == 19.98
    assert order["lines"] == [{"name": "Widget", "quantity": 2, "price": 9.99, "lineTotal": 19.98}]
    assert env.dispatch("view_cart", {}) == {"items": [], "total": 0.0}
    assert env.state()["orders"] == [order]


def test_dispatch_records_exactly_one_dispatch_record_per_call(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    assert env.trace() == []
    env.dispatch("view_cart", {})
    assert len(env.trace()) == 1
    record = env.trace()[0]
    assert isinstance(record, DispatchRecord)
    assert record.name == "view_cart"
    env.dispatch("view_cart", {})
    assert len(env.trace()) == 2


def test_dispatch_unknown_tool_name_returns_an_error_rather_than_raising(sim: ModuleType) -> None:
    env = _fresh_env(sim)
    result = env.dispatch("not_a_real_tool", {})
    assert result == {"error": "unknown-tool", "name": "not_a_real_tool"}
    assert len(env.trace()) == 1


# --------------------------------------------------------------------------------------------
# (b) E1 — the dispatch-totality property test
# --------------------------------------------------------------------------------------------

#: JSON-representable stand-ins for a value whose type contradicts a declared JSON-Schema
#: primitive (the "wrong type per declared field" adversarial case) — deliberately JSON-shaped,
#: since a real model's tool-call arguments are JSON-parsed data, never a raw Python object.
_WRONG_TYPE_VALUE_BY_DECLARED_TYPE: dict[str, Any] = {
    "string": 12345,
    "integer": "not-an-integer",
    "number": "not-a-number",
}

#: One legitimate value per property NAME this pack's schemas declare, shared across every tool
#: that names it (`productName` on both `add_to_cart`/`remove_from_cart`, etc.) — "Wireless
#: Earbuds" is a real row of the shipped `catalog.json`. This is what lets a field-specific
#: adversarial case reach the field it actually targets on a tool with a *gated* second field:
#: `add_to_cart`/`remove_from_cart` both read `productName` first and return early
#: (`{"found": False}`) whenever it is absent, so a case that means to test `quantity` alone (e.g.
#: `{"quantity": "not-an-integer"}`) never reaches `quantity`'s own guard at all — it dead-ends at
#: the `productName` check first. Coordinator-found gap (independent mutation probe: removing
#: `_add_to_cart`'s `_as_int` guard on `quantity` did not redden this file). The fix generalises
#: rather than special-cases the two gated tools: every field-specific case below is layered onto
#: a fully valid call (`_valid_base_arguments`), so a case targeting field X is always reachable
#: regardless of what any other field's own logic gates on — current gating shapes and any this
#: pack's tools grow later alike.
_VALID_VALUE_BY_PROPERTY_NAME: dict[str, Any] = {
    "name": "Wireless Earbuds",
    "productName": "Wireless Earbuds",
    "quantity": 1,
    "category": "Electronics",
    "minPrice": 0,
    "maxPrice": 1000,
}

#: The six field-value adversarial cases (S5 spec §4 Step 1(b)), applied per declared field —
#: independent of "wrong type per declared field", which is generated separately below since it
#: depends on each field's own declared JSON-Schema type.
_GENERIC_ADVERSARIAL_VALUES: list[tuple[str, Any]] = [
    ("empty string", ""),
    ("zero", 0),
    ("negative one", -1),
    ("large int", 10**18),
    ("unicode", "国際化テスト🌟"),
    ("nested object", {"nested": "object"}),
]


def _valid_base_arguments(props: Mapping[str, Any]) -> dict[str, Any]:
    """A fully valid call for `props` (every declared property present, legitimate value) — the
    base a field-specific adversarial case is layered onto, so overriding one field never starves
    another field the handler's own logic gates on."""
    return {
        name: _VALID_VALUE_BY_PROPERTY_NAME[name]
        for name in props
        if name in _VALID_VALUE_BY_PROPERTY_NAME
    }


def _adversarial_argument_dicts(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The S5 spec §4 Step 1(b) adversarial set, generated from one tool's own declared
    `parameters.properties` rather than hand-typed per tool — so a schema change (a field added or
    retyped) changes what this generates without anyone touching this function.

    Every property gets its own full sweep (the six generic values plus, where the property's
    declared type has one, a wrong-type value), each layered onto `_valid_base_arguments` so a
    case aimed at one field cannot dead-end at another field's own early return."""
    props: dict[str, Any] = schema["function"]["parameters"].get("properties") or {}

    if not props:
        # A zero-property tool (view_cart, clear_cart, place_order) has no field to target — still
        # probe it with junk arguments under a stand-in key, since a model may pass some anyway.
        cases: list[tuple[str, dict[str, Any]]] = [
            ("absent key", {}),
            ("empty object", {}),
            ("extra key", {"unexpectedExtraKey": True}),
        ]
        cases.extend((label, {"arg": value}) for label, value in _GENERIC_ADVERSARIAL_VALUES)
        return cases

    valid_base = _valid_base_arguments(props)
    cases = [
        ("absent key", {}),
        ("empty object", {}),
        ("extra key", {**valid_base, "unexpectedExtraKey": True}),
    ]
    for prop_name, prop_spec in props.items():
        for label, value in _GENERIC_ADVERSARIAL_VALUES:
            cases.append((f"{label} for {prop_name!r}", {**valid_base, prop_name: value}))
        declared_type = prop_spec.get("type")
        wrong_value = _WRONG_TYPE_VALUE_BY_DECLARED_TYPE.get(declared_type)
        if wrong_value is not None:
            cases.append((f"wrong type for {prop_name!r}", {**valid_base, prop_name: wrong_value}))
    return cases


def test_dispatch_never_raises_across_every_tool_and_adversarial_argument_shape(
    sim: ModuleType,
) -> None:
    """E1 — the totality claim: crossing every tool in the real `schemas()` with the adversarial
    argument set above must produce zero raises and exactly one new `DispatchRecord` per call.
    Deliberately one test function with no per-case `try/except`: one raise anywhere fails this
    whole test, per the spec's own framing — a caught-and-swallowed exception here would defeat
    the totality claim as surely as an uncaught one."""
    env = sim.build_environment()
    total_cases = 0
    for schema in env.schemas():
        name = schema["function"]["name"]
        for label, args in _adversarial_argument_dicts(schema):
            before = len(env.trace())
            env.dispatch(name, args)
            after = len(env.trace())
            assert after == before + 1, (
                f"{name} ({label}): expected exactly one new DispatchRecord, got {after - before}"
            )
            total_cases += 1
    # Sanity on the generator itself, not just on dispatch: this run must have actually exercised
    # every one of the pack's seven tools, not silently iterated zero schemas.
    assert {schema["function"]["name"] for schema in env.schemas()} == {
        "lookup_product_fact",
        "filter_products",
        "view_cart",
        "add_to_cart",
        "remove_from_cart",
        "clear_cart",
        "place_order",
    }
    assert total_cases >= 7 * 9  # every tool contributed at least the 9 field-agnostic cases


# --------------------------------------------------------------------------------------------
# (c) boundaryRule round-trip off the real tools/schemas.json
# --------------------------------------------------------------------------------------------


def _real_filter_products_property(name: str) -> dict[str, Any]:
    schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    filter_products = next(s for s in schemas if s["function"]["name"] == "filter_products")
    return filter_products["function"]["parameters"]["properties"][name]


def test_max_price_boundary_rule_is_dollars_vs_cents_shaped() -> None:
    """`maxPrice` is the $/cents-shaped parameter: a model could confuse a dollar amount like
    $49.99 with its cents-shaped literal 4999, or with a rounded-up 5000."""
    prop = _real_filter_products_property("maxPrice")
    assert prop["type"] == "number"
    boundary_rule = prop["boundaryRule"]
    assert boundary_rule["inclusive"] is True
    assert 4999 in boundary_rule["confusedWith"]


def test_min_price_boundary_rule_is_inclusive_exclusive_bound_shaped() -> None:
    """`minPrice` is the inclusive/exclusive-bound-shaped parameter: the schema documents `minPrice`
    as inclusive, so a model that means "at least $10.00" but writes an exclusive-style $10.01
    boundary is the confusion this rule names."""
    prop = _real_filter_products_property("minPrice")
    assert prop["type"] == "number"
    boundary_rule = prop["boundaryRule"]
    assert boundary_rule["inclusive"] is True
    assert 10.01 in boundary_rule["confusedWith"]


def test_boundary_rule_is_absent_from_a_non_boundary_parameter() -> None:
    """`category` (a plain string filter, no numeric boundary) carries no `boundaryRule` at all —
    confirms the extension is per-parameter, not a blanket schema-level annotation."""
    prop = _real_filter_products_property("category")
    assert "boundaryRule" not in prop
