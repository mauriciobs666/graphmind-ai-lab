# Pricing flow.
"""Price computation and order finalization."""
from typing import cast

from fx_svc.bus.producer import publish_created
from fx_svc.core.rules import PremiumRules, Rules, eligible, is_member
from fx_svc.db.orders import load_rate


def base_price(item_id, qty, hooks=None):
    rate = load_rate(item_id)
    member_cut = 0.1 * is_member(item_id)
    hook_factor = hooks.find_hook(item_id) if hooks is not None else 1
    return rate * qty * (1 - member_cut) * hook_factor


def apply_discount(item_id, qty, rate):
    return base_price(item_id, qty) * (1 - rate)


def price(item, qty):
    return item.unit_price * qty


def price_with(x):
    rules = cast(Rules, PremiumRules())
    return rules.fee(x)


def notify(o, bus):
    publish_created(o)
    bus.publish_alert(o, "created")


def finalize(o, bus):
    if eligible(o):
        notify(o, bus)
