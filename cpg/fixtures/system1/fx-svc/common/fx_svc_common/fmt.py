"""Formatting helpers (version 2.1)."""
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal


def fmt_money(amount, *, locale):
    sep = "," if locale.startswith("pt") else "."
    return f"{Decimal(amount):.2f}".replace(".", sep)


def round_money(amount, *, mode=ROUND_HALF_EVEN):
    return Decimal(amount).quantize(Decimal("0.01"), rounding=mode)


def fmt_date(ts):
    return datetime.fromtimestamp(ts).date().isoformat()
