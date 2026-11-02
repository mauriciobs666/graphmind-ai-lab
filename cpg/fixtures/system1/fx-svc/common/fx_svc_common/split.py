"""Splitting helpers (version 2.1)."""


def split_money(amount, parts, *, remainder="last"):
    cents = int(round(amount * 100))
    share, extra = divmod(cents, parts)
    shares = [share] * parts
    shares[-1 if remainder == "last" else 0] += extra
    return [s / 100 for s in shares]
