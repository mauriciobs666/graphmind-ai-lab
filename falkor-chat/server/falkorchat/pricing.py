"""Pure line-item total arithmetic (K-053 M6, `docs/plans/workflow-cart-and-totals.md` §3.1).

No I/O — `compute_line_total` is the only public entry point, trivially
unit-testable and, by construction, never reachable from an LLM client (that
is the whole FR-8 argument the owning plan makes: a tool's `Tool.run()` body
calls this directly, in plain Python, so the arithmetic itself is never the
reason for, or the mechanism of, a model completion).

Mirrors `chunking.split_into_chunks`'s "pure function, no I/O" precedent
(`docs/plans/document-ingestion.md` §3.2).
"""

from __future__ import annotations

from typing import Any


def compute_line_total(items: list[dict[str, Any]]) -> float:
    """Sum ``price * quantity`` across `items`.

    Deterministic: identical inputs always produce an identical output (AC-4)
    — no randomness, no clock, no I/O. Each item is expected to carry
    ``"price"`` (a number) and ``"quantity"`` (a number); a malformed line —
    missing either key, or either value not numeric (``bool`` is rejected too,
    since a bool being an `int` subclass would otherwise silently pass) — is
    **skipped, never raised on**. This mirrors the guard family's totality/
    bias-to-decline discipline (`guards.py`'s `_order` wrapper: a comparison
    that cannot decide declines rather than crashing) applied to arithmetic:
    one bad row must never fail a whole cart/order total.
    """
    total = 0.0
    for item in items:
        price = item.get("price")
        quantity = item.get("quantity")
        if not _is_number(price) or not _is_number(quantity):
            continue
        total += price * quantity
    return total


def _is_number(value: Any) -> bool:
    """True for `int`/`float`, excluding `bool` (an `int` subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
