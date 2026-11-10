"""``link`` and ``set`` (plan 3.5; FR-11, FR-15, FR-16, FR-19; AC-8, AC-11).

Relations are closed and recorded from the item they are set on; none alters an
ID.  ``concerns`` is functional (op ``set``, ``to: null`` clears); ``fixed-by`` and
``relates`` are multi-valued (op ``add`` / ``remove``).  Attributes never touch IDs.
"""
from . import alloc, events, project, resolve
from .store import Store

ATTRS = ("milestone", "home", "touched", "slug", "where")


class LinkError(alloc.AllocError):
    """The relation, op, attribute or value is not acceptable."""


def _uid(st, name):
    uid = resolve.resolve(st, name)["uid"]
    if uid not in st.items:     # an FR/AC citation is not a registry row
        raise LinkError(f"{name!r} is not a registry item")
    return uid


def link(root, from_name, rel, target=None, op=None, clear=False, store=None):
    """Record a relation event; ``op`` defaults to set (concerns) / add (others)."""
    if rel not in events.RELATIONS:
        raise LinkError(f"unknown relation {rel!r}: one of {', '.join(events.RELATIONS)}")
    functional = rel in events.FUNCTIONAL_RELATIONS
    if clear:
        if not functional:
            raise LinkError(f"{rel} is multi-valued: remove a target instead of clearing")
        if target is not None:
            raise LinkError("--clear takes no target")
        op = "set"
    op = op or ("set" if functional else "add")
    if (op == "set") != functional:
        raise LinkError(f"op {op!r} is not valid for {rel}: use "
                        + ("set or --clear" if functional else "add or remove"))
    if target is None and not clear:
        raise LinkError(f"{rel} needs a target")
    store = store or Store(root, alloc.load_key(root))
    with store.transaction() as tx:
        frm = _uid(tx.state, from_name)
        to = None if clear else _uid(tx.state, target)
        tx.append({"ev": "link", "rel": rel, "from": frm, "to": to, "op": op})
    return frm


def set_attr(root, name, attr, value, store=None):
    """Set one attribute; ``home``/``touched`` are validated against the components."""
    if attr not in ATTRS:
        raise LinkError(f"unknown attribute {attr!r}: one of {', '.join(ATTRS)}")
    comps = project.load_components(root)
    allowed = comps + [alloc.DEFAULT_HOME]
    if attr == "touched":
        value = [c for c in (value if isinstance(value, list) else str(value).split(",")) if c]
        bad = [c for c in value if c not in comps]
    elif attr == "home":
        bad = [] if value in allowed else [value]
    else:
        bad = []
    if bad:
        raise LinkError(f"unknown component {bad[0]!r} for {attr}: one of {', '.join(allowed)}")
    store = store or Store(root, alloc.load_key(root))
    with store.transaction() as tx:
        uid = _uid(tx.state, name)
        if attr == "slug" and value:
            alloc.check_slug_free(tx.state, value, uid, LinkError)
        tx.append({"ev": "set", "uid": uid, "attr": attr, "value": value})
    return uid
