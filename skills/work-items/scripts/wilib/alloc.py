"""Allocation: ``new`` and ``retire`` (plan 3.2-3.5).

Allocation happens inside one ``Store.transaction()`` (the flock) and reads only
the registry files (FR-7, AC-5); family-document headers are a separate second
phase (``--sync``, step 6).
"""
import json
from pathlib import Path

from . import project, rules
from .store import Store

DEFAULT_HOME = "repo"


class AllocError(RuntimeError):
    """The request cannot be honoured (unknown parent/item, bad type)."""


def load_key(root):
    path = Path(root) / ".work-items" / "project.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["key"]
    except FileNotFoundError:
        raise AllocError(f"{root} has no .work-items/project.json: run init first") from None


def _implicit_parent(key, type_):
    """D lives under I01 (Defects), K under I02 (Kaizen) (FR-14)."""
    num = {"D": rules.DEFECTS_INITIATIVE, "K": rules.KAIZEN_INITIATIVE}.get(type_)
    if num is None:
        return None
    return rules.format_id(key, [rules.Segment("I", num)])


def unknown_item(name, error=None):
    """Error for an item name that is not an ID or alias, pointing legacy names at resolve."""
    hint = f": resolve it first (wi resolve {name}) and use its ID" if ":" in name else ""
    return (error or AllocError)(f"unknown item {name!r}{hint}")


def check_slug_free(st, slug, uid=None, error=None):
    """FR-8/AC-14: a slug names one item.  Retired items keep theirs (their documents
    remain on disk and the family glob would still attribute them)."""
    for item in st.items.values():
        if item.slug == slug and item.uid != uid:
            raise (error or AllocError)(f"slug {slug!r} is already held by {item.id}")


def _segments(item_id):
    return list(rules.parse(item_id).segments)


def allocate(tx, key, type_, parent=None, slug=None, title=None, home=None, legacy=None):
    """Allocate and record one item inside an open transaction; return its ID."""
    if type_ not in rules.WIDTH or type_ in ("FR", "AC"):
        raise AllocError(f"cannot allocate type {type_!r}")
    parent = parent or _implicit_parent(key, type_)
    st = tx.state
    if slug:
        check_slug_free(st, slug)
    if parent is None:
        parent_item, parent_segs = None, []
        rules.validate_parent(type_)
    else:
        uid = st.resolve_uid(parent)
        if uid is None:
            raise AllocError(f"unknown parent {parent!r}")
        parent_item = st.items[uid]
        parent_segs = _segments(parent_item.id)
        last = parent_segs[-1]
        rules.validate_parent(type_, last.type, last.num)
    number = st.next_number(type_, parent_item.uid if parent_item else None)
    id_ = rules.format_id(key, parent_segs + [rules.Segment(type_, number)])
    ev = {"ev": "new", "id": id_, "type": type_,
          "parent": parent_item.uid if parent_item else None,
          "home": home or DEFAULT_HOME}
    if slug:
        ev["slug"] = slug
    if title:
        ev["title"] = title
    if legacy:
        ev["legacy"] = legacy
    tx.append(ev)
    return id_


def new(root, type_, parent=None, slug=None, title=None, home=None, store=None):
    """Allocate and record one item; return its ID.  Raises IllegalParent."""
    key = load_key(root)
    if home is not None and home != DEFAULT_HOME and home not in project.load_components(root):
        raise AllocError(f"unknown home {home!r}: not a declared component of this project")
    store = store or Store(root, key)
    with store.transaction() as tx:
        return allocate(tx, key, type_, parent, slug, title, home)


def retire(root, id_, store=None):
    """Abandon an item: the ID keeps resolving and its number is never reissued."""
    store = store or Store(root, load_key(root))
    with store.transaction() as tx:
        uid = tx.state.resolve_uid(id_)
        if uid is None:
            raise unknown_item(id_)
        tx.append({"ev": "retire", "uid": uid})
    return id_
