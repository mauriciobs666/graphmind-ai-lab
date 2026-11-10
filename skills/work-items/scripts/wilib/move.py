"""``move`` and ``promote`` (plan 3.4; FR-4, FR-13, FR-22, FR-24).

One ``move`` event covers the item and its whole subtree: ``moves`` lists
``{uid, old, new}`` for each, so every descendant's old ID is an explicit alias.
Descendants keep their local numbers (FR-24); only the item itself takes a new
local number, and only for a story/task (FR-22).  F/I/D/K keep their own segment.
"""
from . import alloc, rules
from .store import Store

_PROJECT = {"none", "project", ""}


class MoveError(alloc.AllocError):
    """The request cannot be honoured (unknown item, no-op, wrong type)."""


def _subtree(st, root_uid):
    out, todo = [], [root_uid]
    while todo:
        uid = todo.pop()
        out.append(st.items[uid])
        todo += sorted(i.uid for i in st.items.values() if i.parent == uid)
    return out


def _lookup(st, name, what):
    uid = st.resolve_uid(name)
    if uid is None:
        raise MoveError(f"unknown {what} {name!r}")
    return st.items[uid]


def _rename(tx, item, parent_item, new_segment, retype=None):
    """Validate and emit the ``move`` event; return the item's new ID."""
    st = tx.state
    subtree = _subtree(st, item.uid)
    old_root = item.id
    parent_segs = list(rules.parse(parent_item.id).segments) if parent_item else []
    project = rules.parse(old_root).project
    new_root = rules.format_id(project, parent_segs + [new_segment])
    if new_root == old_root:
        raise MoveError(f"{old_root} is already there")
    moves = []
    for sub in subtree:
        assert sub.id.startswith(old_root), (sub.id, old_root)
        new_id = new_root + sub.id[len(old_root):]
        rules.validate_path(rules.parse(new_id))    # FR-25 for the item and every descendant
        moves.append({"uid": sub.uid, "old": sub.id, "new": new_id})
    ev = {"ev": "move", "uid": item.uid,
          "parent": parent_item.uid if parent_item else None, "moves": moves}
    if retype:
        ev["retype"] = retype
    tx.append(ev)
    return new_root


def _parent(st, name):
    return None if name in _PROJECT else _lookup(st, name, "parent")


def move(root, id_, new_parent, store=None):
    """Rename ``id_`` and its subtree under ``new_parent`` ('none' = the project)."""
    store = store or Store(root, alloc.load_key(root))
    with store.transaction() as tx:
        st = tx.state
        item = _lookup(st, id_, "item")
        parent_item = _parent(st, new_parent)
        own = rules.parse(item.id).segments[-1]
        if own.type in rules.LOCAL_TYPES_IDS:       # S/T: next free local number (FR-22)
            if parent_item is None:
                rules.validate_parent(own.type)     # raises IllegalParent
            seg = rules.Segment(own.type, st.next_number(own.type, parent_item.uid))
        else:                                       # I/F/D/K keep their segment
            seg = own
        return _rename(tx, item, parent_item, seg)


def promote(root, id_, to, parent=None, store=None):
    """Promote a kaizen item to a feature under ``parent`` (initiative or none)."""
    store = store or Store(root, alloc.load_key(root))
    with store.transaction() as tx:
        st = tx.state
        item = _lookup(st, id_, "item")
        if to != "F" or item.type != "K":
            raise MoveError("promote turns a kaizen item (K) into a feature (F)")
        parent_item = _parent(st, parent or "none")
        seg = rules.Segment("F", st.next_number("F"))
        return _rename(tx, item, parent_item, seg, retype="F")
