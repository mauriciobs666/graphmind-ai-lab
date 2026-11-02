"""Eligibility and tier rules."""
from fx_svc.db.orders import connect, load_rate
from fx_svc.state.cache import OrderCache

LIMIT = 1000
_CACHE = OrderCache()


class Rules:
    def fee(self, x):
        return x * 0.02


class PremiumRules(Rules):
    def fee(self, x):
        return x * 0.01


def audit(o):
    return ("audit", o.id)


def is_member(item_id):
    conn = connect()
    return conn is not None and item_id % 2 == 0


def tier_of(o):
    return "gold" if load_rate(o.item_id) < 0.5 else "basic"


def eligible(o, hooks=None):
    if o.total > LIMIT:
        audit(o)
    tier = tier_of(o)
    hook = hooks.find_hook(o.kind) if hooks is not None else 1
    cached = _CACHE.get(o.id)
    return tier != "basic" and cached is None and hook >= 1
