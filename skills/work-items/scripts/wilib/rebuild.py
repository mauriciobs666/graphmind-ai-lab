"""``wi rebuild``: recover a registry that no ref tip carries (plan 3.7, AC-12).

Sources, in order: (1) the git history of ``.work-items/registry.jsonl`` on any ref,
unioned by ``evid`` -- everything ever committed comes back whole; (2) ``Item:`` /
``Aliases:`` headers for items absent from history (one ``new`` per birth ID, one
``move`` per alias step); (3) ``--floor`` retired sentinels so a counter starts above
a number.  Without history only the header fallback is possible, and ``--write`` then
needs ``--accept-reissue``: an ID that reached no commit, header or floor can be reissued.
"""
import json
import re
from pathlib import Path

from . import alloc, events, headers, rules
from .store import REGISTRY_REL, Store, _git

ROLES = ("coordination", "ml", "graph", "rca", "impl", "report")
NOT_RECOVERABLE = ("not recoverable from headers: stories, tasks, titles, links, attributes, "
                   "retirements and items with no document; restore those from the registry's "
                   "git history")
_PART = re.compile(r"\*\*([^:*]+):\*\*\s*(.*)\Z")
_TICKS = re.compile(r"`([^`]+)`")
_SEEDS = ((1, "Defects"), (2, "Kaizen"))


REISSUE_RISK = ("no registry history was found in git: a header-only rebuild can REISSUE an ID "
                "that reached no commit, header or floor (undocumented or promoted features, "
                "story/task counters, kaizen numbers, legacy mappings, so a later mint-legacy "
                "would mint a second ID); pass --accept-reissue to proceed, and --floor to "
                "start counters above the numbers you know were used")
_FLOOR = re.compile(r"([A-Z]+)(?::([^=]+))?=(\d+)\Z")


class RebuildError(alloc.AllocError):
    """Rebuild refused (registry is not lost, no history, bad floor)."""


def history_events(root):
    """``(events, versions, unreadable)``: every committed version of the tracked registry
    on any ref, oldest first, unioned by ``evid`` (first occurrence kept)."""
    rel = f"./{REGISTRY_REL.as_posix()}"
    log = _git(root, "log", "--all", "--full-history", "--format=%H", "--", rel) or ""
    union, seen, versions, unreadable = [], set(), 0, 0
    for commit in reversed(log.split()):
        text = _git(root, "show", f"{commit}:{rel}")
        if text is None:            # a commit that deleted the file
            continue
        try:
            evs = events.read_events(text)
        except events.EventError:
            unreadable += 1         # reported, never silent
            continue
        versions += 1
        for ev in evs:
            if ev["evid"] not in seen:
                seen.add(ev["evid"])
                union.append(ev)
    return union, versions, unreadable


def iter_docs(root):
    """Every document of a family kind under ``root``, sorted."""
    root = Path(root)
    found = set()
    for kind in headers.KINDS:
        found |= set(root.glob(f"docs/{kind}/*.md")) | set(root.glob(f"*/docs/{kind}/*.md"))
    return sorted(p for p in found if p.is_file())


def header_fields(path):
    """``{name: raw value}`` of the one-line header block, or None without a header block."""
    lines = Path(path).read_bytes().decode("utf-8", "replace").splitlines()
    i = headers.header_index(lines)
    if i is None:
        return None
    out = {}
    for part in lines[i].removeprefix("> ").split(headers.SEP):
        m = _PART.match(part.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def stem_slug(name):
    """Slug a family file name stands for: extension and a closed-set ``-role`` dropped."""
    stem = name.removesuffix(".md")
    for role in ROLES:
        if stem.endswith("-" + role):
            return stem[: -len(role) - 1]
    return stem


def tokens(value):
    return _TICKS.findall(value) or ([value] if value and value != "—" else [])


def _legal(id_):
    try:
        rules.validate_path(rules.parse(id_))
    except (rules.MalformedId, rules.IllegalParent):
        return False
    return True


def collect(roots):
    """``(items, notes)``: one record per distinct legal ``Item:`` ID."""
    found, notes = {}, []
    for root in roots:
        root = Path(root)
        for path in iter_docs(root):
            fields = header_fields(path)
            toks = tokens((fields or {}).get("Item", ""))
            if not toks:
                continue
            if not _legal(toks[0]):
                notes.append(f"{path}: Item {toks[0]!r} is not a legal ID; skipped")
                continue
            rel = path.relative_to(root).as_posix()
            found.setdefault(toks[0], []).append((rel, fields))
    items = []
    for id_, docs in sorted(found.items()):
        aliases, legacy = [], None
        for _, fields in docs:      # the doc with the most aliases wins; ties keep the first
            ids = [a for a in tokens(fields.get("Aliases", "")) if ":" not in a]
            if len(ids) > len(aliases):
                aliases = ids
            for a in tokens(fields.get("Aliases", "")):
                if ":" in a and legacy is None:
                    legacy = dict(zip(("home", "name"), a.split(":", 1)))
        slugs = sorted({stem_slug(Path(rel).name) for rel, _ in docs}, key=len)
        if any(not s.startswith(slugs[0]) for s in slugs):
            notes.append(f"{id_}: documents disagree on the slug {slugs}; kept {slugs[0]!r}")
        rel = docs[0][0]
        home = rel.split("/")[0] if not rel.startswith("docs/") else "repo"
        items.append({"id": id_, "aliases": [a for a in aliases if _legal(a)], "legacy": legacy,
                      "slug": slugs[0], "home": home})
    return items, notes


def build_events(items, key, known=None):
    """Event dicts (no evid/ts): ``new`` for each birth ID, ``move`` per alias step.
    ``known`` is the state already recovered from history: its seeds are not repeated and
    its items are valid parents."""
    birth_of = {}
    for it in items:
        chain = it["aliases"] + [it["id"]]
        for c in chain:
            birth_of[c] = chain[0]

    def parent_uid(id_):
        segs = rules.parse(id_).segments
        if len(segs) < 2:
            return None
        pid = rules.format_id(key, segs[:-1])
        return birth_of.get(pid) or (known.by_id.get(pid) if known else None) or pid

    out = []
    for num, title in _SEEDS:
        sid = rules.format_id(key, [rules.Segment("I", num)])
        if sid not in birth_of and not (known and sid in known.by_id):
            out.append({"ev": "new", "id": sid, "type": "I", "parent": None,
                        "home": alloc.DEFAULT_HOME, "title": title})
    for it in items:
        chain = it["aliases"] + [it["id"]]
        birth = chain[0]
        new = {"ev": "new", "id": birth, "type": rules.parse(birth).segments[-1].type,
               "parent": parent_uid(birth), "home": it["home"], "slug": it["slug"]}
        if it["legacy"]:
            new["legacy"] = it["legacy"]
        out.append(new)
        for old, nxt in zip(chain, chain[1:]):
            mv = {"ev": "move", "uid": birth, "parent": parent_uid(nxt),
                  "moves": [{"uid": birth, "old": old, "new": nxt}]}
            if rules.parse(old).segments[-1].type != rules.parse(nxt).segments[-1].type:
                mv["retype"] = rules.parse(nxt).segments[-1].type
            out.append(mv)
    return out


def _stamped(evs, base=0):
    return [{"evid": f"tmp{base + n}", "ts": base + n + 1, **e} for n, e in enumerate(evs)]


def parse_floor(spec):
    m = _FLOOR.match(spec)
    if not m or int(m.group(3)) < 1:
        raise RebuildError(f"bad --floor {spec!r}: expected TYPE=N or TYPE:<parent>=N with N >= 1")
    type_, parent, num = m.group(1), m.group(2), int(m.group(3))
    if type_ in events.LOCAL_TYPES and not parent:
        raise RebuildError(f"--floor {spec}: {type_} is counted per parent: give a parent "
                           f"({type_}:<parent>=N)")
    if type_ in events.GLOBAL_TYPES and parent:
        raise RebuildError(f"--floor {spec}: {type_} is counted per project: no parent allowed")
    if type_ not in events.GLOBAL_TYPES + events.LOCAL_TYPES:
        raise RebuildError(f"bad --floor {spec!r}: {type_} has no counter")
    return type_, parent, num


def floor_events(floors, st, key):
    """Retired sentinel ``new`` + ``retire`` per floor, so the counter starts above it."""
    out = []
    for spec in floors:
        type_, parent, num = parse_floor(spec)
        if parent:
            uid = st.resolve_uid(parent)
            if uid is None:
                raise RebuildError(f"--floor {spec}: unknown parent {parent!r}")
            parent_item = st.items[uid]
        else:
            implicit = alloc._implicit_parent(key, type_)
            parent_item = st.items[st.resolve_uid(implicit)] if implicit else None
        segs = list(rules.parse(parent_item.id).segments) if parent_item else []
        id_ = rules.format_id(key, segs + [rules.Segment(type_, num)])
        try:
            rules.validate_path(rules.parse(id_))
        except rules.IllegalParent as exc:
            raise RebuildError(f"--floor {spec}: {exc}") from None
        if st.resolve_uid(id_) or any(e.get("id") == id_ for e in out):
            continue                    # that number is already taken
        out.append({"ev": "new", "id": id_, "type": type_,
                    "parent": parent_item.uid if parent_item else None,
                    "home": alloc.DEFAULT_HOME, "title": "floor sentinel (rebuild)"})
        out.append({"ev": "retire", "uid": id_})
    return out


def _known(item, st):
    return any(c in st.by_id for c in item["aliases"] + [item["id"]])


def rebuild(root, roots=(), write=False, floors=(), accept_reissue=False, store=None):
    """Plan (and with ``write`` apply) the rebuild; return report lines."""
    root = Path(root)
    key = alloc.load_key(root)
    hist, versions, unreadable = history_events(root)
    hist_state = events.fold(hist)
    items, notes = collect([root / r for r in roots])
    items = [i for i in items if not _known(i, hist_state)]
    evs = build_events(items, key, hist_state)
    evs += floor_events(floors, events.fold(hist + _stamped(evs)), key)
    lines = [f"history: {len(hist)} events from {versions} committed version(s)"]
    if unreadable:
        lines.append(f"note: {unreadable} committed version(s) were unreadable and skipped")
    lines += [f"item {i['id']} slug={i['slug']} aliases={','.join(i['aliases']) or '-'}"
              for i in items]
    lines += [f"note: {n}" for n in notes]
    if not hist:
        lines.append(f"warning: {REISSUE_RISK}")
    lines.append(NOT_RECOVERABLE)
    if not write:
        return ["dry run: nothing written (use --write)"] + lines
    store = store or Store(root, key)
    with store.transaction() as tx:
        if not tx.lost:     # the on-disk copies, not the state: opening reseeded from ref tips
            raise RebuildError(
                f"the registry is not lost ({len(tx.state.items)} items present): refusing to "
                "rebuild over it; delete both copies first if that is really intended")
        if not hist and not accept_reissue:
            raise RebuildError(REISSUE_RISK)
        have = tx.evids                 # a stale ref tip's events were already reseeded
        for ev in hist:
            if ev["evid"] not in have:
                tx.append(ev)           # keeps its evid and ts
        for ev in evs:
            tx.append(ev)
    store.flush()
    _record_marker(root, {"history_events": len(hist), "header_items": len(items),
                          "floors": list(floors), "reissue_accepted": bool(accept_reissue)})
    return [f"wrote {len(hist) + len(evs)} events"] + lines


def _record_marker(root, marker):
    path = Path(root) / ".work-items" / "project.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["rebuilt"] = marker
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
