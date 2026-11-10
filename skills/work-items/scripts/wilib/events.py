"""Event schema, tolerant reader, deterministic fold and counters (FR-5..FR-8).

The registry is an append-only JSONL log.  Each line is one event
``{"evid", "ts", "ev", ...}``:

* ``new``    id, type, parent (uid or null), home; optional slug, title,
             ``legacy: {home, name}``
* ``move``   uid, parent (new parent uid or null), ``moves: [{uid, old, new}]``
             covering the item and its whole subtree; optional ``retype``
* ``link``   rel (concerns | fixed-by | relates), from, to (uid), ``op``:
             ``set`` (default; functional ``concerns`` only, ``to: null`` clears) |
             ``add`` | ``remove`` (multi-valued ``fixed-by``/``relates``)
* ``set``    uid, attr (milestone | home | touched | slug | where | title), value
* ``retire`` uid

An item's uid is its birth ID.  State is the fold of the deduplicated union:
all ``new`` events first, then the rest by ``(ts, evid)``, so a ``move``
stamped before its ``new`` (backward clock step) still applies.  No event
deletes a number: counters follow the maximum number ever seen, aliases
included, and are never reduced by a move.
"""
import json

from . import rules

EV_KINDS = ("new", "move", "link", "set", "retire")
RELATIONS = ("concerns", "fixed-by", "relates")
FUNCTIONAL_RELATIONS = ("concerns",)
LINK_OPS = ("set", "add", "remove")
SET_ATTRS = ("milestone", "home", "touched", "slug", "where", "title")
GLOBAL_TYPES = ("I", "F", "D", "K")     # counted per project
LOCAL_TYPES = ("S", "T")                # counted per (parent uid, type)
_COUNTER_FLOOR = {"I": 2}               # I01/I02 are reserved: first free is I03


class EventError(ValueError):
    """A line or record is not a valid registry event."""


# ------------------------------------------------------------------ schema
def _need(ev, key, kind):
    if key not in ev:
        raise EventError(f"{kind} event lacks {key!r}")
    return ev[key]


def _need_str(ev, key, kind):
    v = _need(ev, key, kind)
    if not isinstance(v, str) or not v:
        raise EventError(f"{kind} event {key!r} must be a non-empty string")
    return v


def _check_id(text, what):
    try:
        rules.parse(text)
    except rules.MalformedId as exc:
        raise EventError(f"{what}: {exc}") from None


def validate_event(ev):
    """Raise EventError unless ``ev`` is a well-formed event; return it."""
    if not isinstance(ev, dict):
        raise EventError("event is not a JSON object")
    _need_str(ev, "evid", "any")
    ts = _need(ev, "ts", "any")
    if isinstance(ts, bool) or not isinstance(ts, int):
        raise EventError("ts must be an integer")
    kind = ev.get("ev")
    if kind not in EV_KINDS:
        raise EventError(f"unknown event kind {kind!r}")
    if kind == "new":
        _check_id(_need_str(ev, "id", kind), "new.id")
        _need_str(ev, "type", kind)
        parent = _need(ev, "parent", kind)
        if parent is not None and not isinstance(parent, str):
            raise EventError("new.parent must be a uid or null")
        _need_str(ev, "home", kind)
        legacy = ev.get("legacy")
        if legacy is not None and not (
                isinstance(legacy, dict) and isinstance(legacy.get("home"), str)
                and isinstance(legacy.get("name"), str)):
            raise EventError("new.legacy must be {home, name}")
    elif kind == "move":
        _need_str(ev, "uid", kind)
        parent = _need(ev, "parent", kind)
        if parent is not None and not isinstance(parent, str):
            raise EventError("move.parent must be a uid or null")
        moves = _need(ev, "moves", kind)
        if not isinstance(moves, list) or not moves:
            raise EventError("move.moves must be a non-empty list")
        for m in moves:
            if not isinstance(m, dict) or not all(isinstance(m.get(k), str) for k in ("uid", "old", "new")):
                raise EventError("each move entry needs string uid, old, new")
            _check_id(m["old"], "move.old")
            _check_id(m["new"], "move.new")
    elif kind == "link":
        if _need(ev, "rel", kind) not in RELATIONS:
            raise EventError(f"link.rel must be one of {RELATIONS}")
        _need_str(ev, "from", kind)
        to = _need(ev, "to", kind)
        if to is not None and not isinstance(to, str):
            raise EventError("link.to must be a uid or null")
        op = ev.get("op", "set")
        if op not in LINK_OPS:
            raise EventError(f"link.op must be one of {LINK_OPS}")
        if (op == "set") != (ev["rel"] in FUNCTIONAL_RELATIONS):
            want = "set" if ev["rel"] in FUNCTIONAL_RELATIONS else "add or remove"
            raise EventError(f"link.op {op!r} is not valid for {ev['rel']!r}: use {want}")
        if to is None and op != "set":
            raise EventError("link.to may be null only with op set")
    elif kind == "set":
        _need_str(ev, "uid", kind)
        if _need(ev, "attr", kind) not in SET_ATTRS:
            raise EventError(f"set.attr must be one of {SET_ATTRS}")
        _need(ev, "value", kind)
    else:  # retire
        _need_str(ev, "uid", kind)
    return ev


# ------------------------------------------------------------------ reader
def dump_event(ev):
    """One event as one newline-terminated JSON line."""
    return json.dumps(ev, separators=(",", ":"), ensure_ascii=False) + "\n"


def read_events(text):
    """Parse registry text into validated events.

    Tolerant of what a union merge produces: a missing final newline, blank
    lines, and several objects glued on one line (``{..}{..}``), which are read
    one by one with ``raw_decode``.  Anything else raises EventError naming the
    1-based line, so corruption is never silently dropped.
    """
    decoder = json.JSONDecoder()
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        pos, end = 0, len(line)
        while True:
            while pos < end and line[pos].isspace():
                pos += 1
            if pos >= end:
                break
            try:
                obj, pos = decoder.raw_decode(line, pos)
                out.append(validate_event(obj))
            except (json.JSONDecodeError, EventError) as exc:
                raise EventError(f"line {lineno}: {exc}") from None
    return out


def stamp_ts(last_ts, now_ns):
    """Next event timestamp: never at or below the folded maximum."""
    return max(now_ns, last_ts + 1)


# ------------------------------------------------------------------ fold
class Item:
    def __init__(self, uid, id_, type_, parent, home):
        self.uid, self.id, self.type, self.parent, self.home = uid, id_, type_, parent, home
        self.slug = self.title = self.milestone = self.where = None
        self.touched = []
        self.aliases = []       # every former ID, oldest first
        self.legacy = None
        self.retired = False


class State:
    def __init__(self):
        self.items = {}         # uid -> Item
        self.by_id = {}         # any ID ever held (birth, old, current) -> uid
        self.legacy = {}        # (home, name) -> uid
        self.links = {}         # (rel, from uid) -> ordered list of to uids; empty ones are absent
        self.orphans = []       # events naming a uid with no `new` (forgotten flush)
        self.last_ts = 0
        self._global = {}       # type -> max number seen (aliases included)
        self._local = {}        # (parent uid, type) -> max local number ever issued

    def resolve_uid(self, id_):
        return self.by_id.get(id_)

    def next_number(self, type_, parent_uid=None):
        """Next free number: per project for I/F/D/K, per parent for S/T."""
        if type_ in LOCAL_TYPES:
            if parent_uid is None:
                raise ValueError(f"{type_} is counted per parent: parent_uid required")
            return self._local.get((parent_uid, type_), 0) + 1
        if type_ in GLOBAL_TYPES:
            return max(self._global.get(type_, 0), _COUNTER_FLOOR.get(type_, 0)) + 1
        raise ValueError(f"{type_} has no registry counter")

    def _see_id(self, id_):
        for seg in rules.parse(id_).segments:
            if seg.type in GLOBAL_TYPES:
                self._global[seg.type] = max(self._global.get(seg.type, 0), seg.num)

    def _see_local(self, parent_uid, id_):
        last = rules.parse(id_).segments[-1]
        if last.type in LOCAL_TYPES and parent_uid is not None:
            key = (parent_uid, last.type)
            self._local[key] = max(self._local.get(key, 0), last.num)


def fold(events):
    st = State()
    seen, uniq = set(), []
    for ev in events:
        if ev["evid"] not in seen:
            seen.add(ev["evid"])
            uniq.append(ev)
    # all `new` first, then the rest; each group by (ts, evid)
    uniq.sort(key=lambda e: (e["ev"] != "new", e["ts"], e["evid"]))
    for ev in uniq:
        st.last_ts = max(st.last_ts, ev["ts"])
        _APPLY[ev["ev"]](st, ev)
    return st


def _apply_new(st, ev):
    uid = ev["id"]
    if uid in st.items:     # two `new` with one ID: first wins; `check` reports it
        return
    item = Item(uid, ev["id"], ev["type"], ev["parent"], ev["home"])
    item.slug, item.title = ev.get("slug"), ev.get("title")
    item.legacy = ev.get("legacy")
    st.items[uid] = item
    st.by_id[uid] = uid
    if item.legacy:
        st.legacy[(item.legacy["home"], item.legacy["name"])] = uid
    st._see_id(ev["id"])
    st._see_local(ev["parent"], ev["id"])


def _apply_move(st, ev):
    root = st.items.get(ev["uid"])
    if root is None:
        st.orphans.append(ev)
        return
    for m in ev["moves"]:
        st._see_id(m["old"])    # numbers count even when the item's `new` is unseen
        st._see_id(m["new"])
        item = st.items.get(m["uid"])
        if item is None:
            continue
        for former in (m["old"], item.id):
            if former != m["new"] and former not in item.aliases:
                item.aliases.append(former)
        item.id = m["new"]
        if m["new"] in item.aliases:
            item.aliases.remove(m["new"])
        st.by_id[m["old"]] = item.uid
        st.by_id[m["new"]] = item.uid
    root.parent = ev["parent"]
    if ev.get("retype"):
        root.type = ev["retype"]
    new_id = next((m["new"] for m in ev["moves"] if m["uid"] == root.uid), root.id)
    st._see_local(root.parent, new_id)


def _apply_link(st, ev):
    key = (ev["rel"], ev["from"])
    op, to = ev.get("op", "set"), ev["to"]
    if op == "set":
        targets = [] if to is None else [to]
    else:
        targets = [t for t in st.links.get(key, []) if t != to]
        if op == "add":
            targets.append(to)
    if targets:
        st.links[key] = targets
    else:
        st.links.pop(key, None)


def _apply_set(st, ev):
    item = st.items.get(ev["uid"])
    if item is None:
        st.orphans.append(ev)
        return
    setattr(item, ev["attr"], ev["value"])


def _apply_retire(st, ev):
    item = st.items.get(ev["uid"])
    if item is None:
        st.orphans.append(ev)
        return
    item.retired = True


_APPLY = {"new": _apply_new, "move": _apply_move, "link": _apply_link,
          "set": _apply_set, "retire": _apply_retire}
