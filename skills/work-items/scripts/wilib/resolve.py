"""Name resolution (plan 3.6/3.9; FR-10, FR-20, FR-24, AC-3, AC-17).

``resolve(state, name)`` maps any name to an item record: a current ID, an alias
(any former ID), a FR/AC path citation (suffix split, feature part resolved
through any alias, suffix re-appended: FR/AC are not registry rows), a slug, or
-- through ``FALLBACKS`` hooks that step 5 fills -- a legacy qualified/bare name.
"""
from . import rules

JSON_KEYS = ("id", "uid", "type", "parent", "aliases", "slug", "home", "touched",
             "milestone", "where", "links", "retired", "matched_as", "input")
MATCHED_AS = ("current", "alias", "legacy", "bare", "slug")

# Hooks ``fn(state, name) -> (uid, matched_as) | None``, tried after IDs and slugs.
FALLBACKS = []


class ResolveError(LookupError):
    """``code`` follows the plan's exit codes: 3 ambiguous, 4 unknown."""

    def __init__(self, message, code=4):
        super().__init__(message)
        self.code = code


def _links(st, item):
    """``{rel: [current IDs]}`` for the relations set on ``item``, targets by uid."""
    out = {}
    for (rel, frm), targets in sorted(st.links.items()):
        ids = [st.items[t].id for t in targets if t in st.items]
        if frm == item.uid and ids:
            out[rel] = ids
    return out


def _record(st, item, input_, matched_as):
    parent = st.items.get(item.parent) if item.parent else None
    return {
        "id": item.id, "uid": item.uid, "type": item.type,
        "parent": parent.id if parent else None,
        "aliases": list(item.aliases), "slug": item.slug, "home": item.home,
        "touched": list(item.touched or []), "milestone": item.milestone,
        "where": item.where,
        "links": _links(st, item),
        "retired": item.retired, "matched_as": matched_as, "input": input_,
    }


def _by_id(st, name):
    uid = st.resolve_uid(name)
    if uid is None:
        return None
    item = st.items[uid]
    return _record(st, item, name, "current" if name == item.id else "alias")


def _requirement(st, name):
    """``<feature path>-FR<n>[a-z]`` / ``-AC<n>[a-z]``: resolve the feature, re-append."""
    try:
        parsed = rules.parse(name)
    except rules.MalformedId:
        return None
    if len(parsed.segments) < 2 or parsed.segments[-1].type not in ("FR", "AC"):
        return None
    tail = parsed.segments[-1]
    feature = _by_id(st, rules.format_id(parsed.project, parsed.segments[:-1]))
    if feature is None or feature["type"] != "F":    # FR-25: FR/AC have a feature, nothing else
        return None
    suffix = "-" + rules.format_segment(tail)
    rec = dict(feature)
    rec.update(
        id=feature["id"] + suffix, uid=feature["uid"] + suffix, type=tail.type,
        parent=feature["id"], aliases=[a + suffix for a in feature["aliases"]],
        links={}, input=name,
        matched_as="current" if feature["matched_as"] == "current" else "alias")
    return rec


def _by_slug(st, name):
    hits = [i for i in st.items.values() if i.slug == name]
    if len(hits) > 1:
        raise ResolveError(
            f"slug {name!r} is ambiguous: " + ", ".join(sorted(i.id for i in hits)), 3)
    return _record(st, hits[0], name, "slug") if hits else None


def resolve(st, name):
    for finder in (_by_id, _requirement, _by_slug):
        rec = finder(st, name)
        if rec is not None:
            return rec
    for hook in FALLBACKS:
        hit = hook(st, name)
        if hit is not None:
            uid, how = hit
            return _record(st, st.items[uid], name, how)
    raise ResolveError(f"unknown name {name!r}")


def describe(rec):
    """One text line; a non-current input says what it is an alias of."""
    if rec["matched_as"] == "current":
        return rec["id"]
    if rec["matched_as"] == "slug":
        return f"{rec['id']}  [slug {rec['input']}]"
    return f"{rec['id']}  [{rec['input']}: alias of {rec['id']}]"
