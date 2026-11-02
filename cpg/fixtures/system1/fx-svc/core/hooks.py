"""Pricing hooks."""
_HOOKS = {"promo": 0.9}


def find_hook(name):
    return _HOOKS.get(name, 1)
