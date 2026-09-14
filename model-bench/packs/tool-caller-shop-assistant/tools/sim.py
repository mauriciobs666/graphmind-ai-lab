"""`tool-caller-shop-assistant`'s simulated storefront (S5 spec §3.3).

Seven tools transcribed from `falkorchat/tools.py`'s catalog/cart/order cluster:
`lookup_product_fact`, `filter_products`, `view_cart`, `add_to_cart`, `remove_from_cart`,
`clear_cart`, `place_order`. `build_environment()` is the pack's `tools.entrypoint` (`pack.json`),
returning a `ShopEnvironment` that structurally satisfies `modelbench.tooling.ToolEnvironment` —
no subclassing, no import beyond stdlib and `modelbench.tooling` (`validate_pack`'s AST import
allowlist, `modelbench/packs.py`).

**The dispatch-totality contract is the load-bearing rule** (S5 spec §3.3, the dispatch-failure
note §4(a)): `dispatch(name, arguments)` never raises on anything the model can produce. Every
input-shaped problem — an unknown tool name, a missing/extra/wrong-typed argument, a product name
not in the catalog, a non-positive or catalog-exceeding quantity, removing a line not in the cart —
is a returned `{"error": "<code>", ...}` value, recorded as an ordinary `DispatchRecord` like any
other call. `raise` is reserved for a condition independent of the model's own arguments (there is
none in this module — a corrupted catalog would be the only candidate, and this module does not
load one lazily per call). There is deliberately **no** blanket `try/except` around dispatch: each
handler below guards its own fields explicitly (`_as_str`/`_as_number`/`_as_int`, all total
functions that return `None` rather than raise on an unexpected type), because a catch-all would
make the totality claim untestable — a removed guard would silently fall through to the catch-all
instead of reddening `tests/test_tools_sim.py`'s adversarial property test.

The sim does **not** schema-validate arguments before executing (§3.3's corollary): a wrong-typed
argument still reaches a handler's own logic and is coerced or rejected there, by name, never
rejected pre-dispatch — pre-dispatch rejection would make the call undispatchable and destroy
FR-8(d)'s own boundary-argument measurement.

Per-conversation state (a cart, a placed-order list) is fresh per `ShopEnvironment` instance;
`runner._drive_conversations` already builds one environment per script, so no cross-conversation
reset is this module's own concern.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from modelbench.tooling import DispatchRecord

#: Wrong-type stand-ins for each JSON-Schema primitive this pack's tools declare (S5 spec §3.3's
#: "wrong type per declared field" adversarial case) — deliberately JSON-representable, since a
#: real model's tool-call arguments are JSON-parsed data, never a raw Python object.
_WRONG_TYPE_VALUE_BY_DECLARED_TYPE: dict[str, Any] = {
    "string": 12345,
    "integer": "not-an-integer",
    "number": "not-a-number",
}


def _as_str(value: Any) -> str | None:
    """`value` if it is a real `str`, else `None` — never raises, whatever `value` is (including
    an unhashable dict/list, which a bare `catalog.get(value)` would blow up on)."""
    return value if isinstance(value, str) else None


def _as_number(value: Any) -> float | int | None:
    """`value` if it is a real `int`/`float` (excluding `bool`, which is a `int` subclass in
    Python and not what a JSON Schema `number`/`integer` author means), else `None`."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ShopEnvironment:
    """A simulated storefront: a product catalog, a cart (`{name: quantity}`), and a placed-order
    list. `catalog`/`schemas` are injected so tests can build one against a small hand-written
    fixture catalog without touching the real `catalog.json`/`schemas.json` (§3.3, `tests/
    test_tools_sim.py`'s "(a) each tool's ordinary behaviour" cases); `build_environment()` below
    is what wires the real pack files for an actual run.
    """

    def __init__(
        self, catalog: Mapping[str, Mapping[str, Any]], schemas: list[Mapping[str, Any]]
    ) -> None:
        self._catalog: dict[str, Mapping[str, Any]] = dict(catalog)
        self._schemas: list[Mapping[str, Any]] = list(schemas)
        self._cart: dict[str, int] = {}
        self._orders: list[dict[str, Any]] = []
        self._trace: list[DispatchRecord] = []
        self._handlers = {
            "lookup_product_fact": self._lookup_product_fact,
            "filter_products": self._filter_products,
            "view_cart": self._view_cart,
            "add_to_cart": self._add_to_cart,
            "remove_from_cart": self._remove_from_cart,
            "clear_cart": self._clear_cart,
            "place_order": self._place_order,
        }

    # ---- ToolEnvironment protocol ----------------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._schemas]

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        args = dict(arguments) if isinstance(arguments, Mapping) else {}
        safe_name = name if isinstance(name, str) else str(name)
        handler = self._handlers.get(name)
        result: Any
        if handler is None:
            result = {"error": "unknown-tool", "name": safe_name}
        else:
            result = handler(args)
        self._trace.append(
            DispatchRecord(
                name=safe_name,
                rawArguments=args,
                parsedArguments=args,
                returnValue=result,
                timestamp=_utc_timestamp(),
            )
        )
        return result

    def trace(self) -> list[DispatchRecord]:
        return list(self._trace)

    def state(self) -> dict[str, Any]:
        return {"cart": dict(self._cart), "orders": [dict(o) for o in self._orders]}

    # ---- tool implementations ---------------------------------------------------------------

    def _lookup_product_fact(self, args: dict[str, Any]) -> dict[str, Any]:
        name = _as_str(args.get("name"))
        product = self._catalog.get(name) if name is not None else None
        if product is None:
            return {"found": False}
        return {"found": True, "category": product["category"], "price": product["price"]}

    def _filter_products(self, args: dict[str, Any]) -> dict[str, Any]:
        category = _as_str(args.get("category"))
        min_price = _as_number(args.get("minPrice"))
        max_price = _as_number(args.get("maxPrice"))
        items = []
        for product in self._catalog.values():
            if category is not None and product["category"] != category:
                continue
            if min_price is not None and product["price"] < min_price:
                continue
            if max_price is not None and product["price"] > max_price:
                continue
            items.append({
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
            })
        if not items:
            return {"items": [], "finding": "no matching products found"}
        return {"items": items}

    def _view_cart(self, args: dict[str, Any]) -> dict[str, Any]:
        items = []
        total = 0.0
        for name, quantity in self._cart.items():
            product = self._catalog.get(name)
            price = product["price"] if product is not None else 0.0
            items.append({"name": name, "quantity": quantity, "price": price})
            total += price * quantity
        return {"items": items, "total": round(total, 2)}

    def _add_to_cart(self, args: dict[str, Any]) -> dict[str, Any]:
        name = _as_str(args.get("productName"))
        product = self._catalog.get(name) if name is not None else None
        if product is None:
            return {"found": False}

        quantity_raw = args.get("quantity")
        if quantity_raw is None:
            quantity = 1
        else:
            quantity = _as_int(quantity_raw)
            if quantity is None:
                return {"error": "wrong-type", "field": "quantity", "expected": "integer"}
        if quantity <= 0:
            return {"error": "invalid-quantity", "quantity": quantity}

        current = self._cart.get(name, 0)
        stock = product.get("stock")
        if isinstance(stock, int) and current + quantity > stock:
            return {"error": "insufficient-stock", "available": max(stock - current, 0)}

        self._cart[name] = current + quantity
        return {"found": True, "productName": name, "quantity": self._cart[name]}

    def _remove_from_cart(self, args: dict[str, Any]) -> dict[str, Any]:
        name = _as_str(args.get("productName"))
        product = self._catalog.get(name) if name is not None else None
        if product is None:
            return {"found": False}

        quantity_raw = args.get("quantity")
        if quantity_raw is None:
            quantity: int | None = None  # remove the whole line
        else:
            quantity = _as_int(quantity_raw)
            if quantity is None:
                return {"error": "wrong-type", "field": "quantity", "expected": "integer"}
            if quantity <= 0:
                return {"error": "invalid-quantity", "quantity": quantity}

        current = self._cart.get(name, 0)
        if current <= 0:
            return {"found": True, "removed": False, "productName": name}

        if quantity is None or quantity >= current:
            del self._cart[name]
            removed = current
        else:
            self._cart[name] = current - quantity
            removed = quantity
        return {"found": True, "removed": True, "productName": name, "quantityRemoved": removed}

    def _clear_cart(self, args: dict[str, Any]) -> dict[str, Any]:
        self._cart.clear()
        return {"cleared": True}

    def _place_order(self, args: dict[str, Any]) -> Any:
        if not self._cart:
            return "The cart is empty — add an item before placing an order."
        lines = []
        total = 0.0
        for name, quantity in self._cart.items():
            product = self._catalog.get(name)
            price = product["price"] if product is not None else 0.0
            line_total = round(price * quantity, 2)
            total += line_total
            lines.append({
                "name": name, "quantity": quantity, "price": price, "lineTotal": line_total,
            })
        order = {
            "orderId": f"order-{len(self._orders) + 1}",
            "lines": lines,
            "total": round(total, 2),
        }
        self._orders.append(order)
        self._cart.clear()
        return order


def _load_catalog(pack_root: Path) -> dict[str, Mapping[str, Any]]:
    raw = json.loads((pack_root / "catalog.json").read_text(encoding="utf-8"))
    return {product["name"]: product for product in raw}


def _load_schemas(pack_root: Path) -> list[Mapping[str, Any]]:
    return json.loads((pack_root / "tools" / "schemas.json").read_text(encoding="utf-8"))


def build_environment() -> ShopEnvironment:
    """`pack.json`'s `tools.entrypoint` (§3.2) — loads the real `catalog.json`/`tools/schemas.json`
    shipped beside this file."""
    pack_root = Path(__file__).resolve().parent.parent
    return ShopEnvironment(catalog=_load_catalog(pack_root), schemas=_load_schemas(pack_root))
