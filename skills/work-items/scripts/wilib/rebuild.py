"""``wi rebuild --from-headers``: recover the alias table from ``Item:``/``Aliases:`` headers.

For the case where BOTH registry copies (journal and tracked file) are lost (plan 3.7,
AC-12).  One ``new`` per distinct ``Item:`` ID, then one ``move`` per ``Aliases:`` entry,
so every header-recorded ID resolves as before and the counters follow from the fold.
Not recoverable from headers: stories, tasks, titles, links, attributes, retirements and
any item with no document; the git history of the registry is the backup.
"""
import re
from pathlib import Path

from . import alloc, headers, rules
from .store import Store

ROLES = ("coordination", "ml", "graph", "rca", "impl", "report")
NOT_RECOVERABLE = ("not recoverable from headers: stories, tasks, titles, links, attributes, "
                   "retirements and items with no document; restore those from the registry's "
                   "git history")
_PART = re.compile(r"\*\*([^:*]+):\*\*\s*(.*)\Z")
_TICKS = re.compile(r"`([^`]+)`")
_SEEDS = ((1, "Defects"), (2, "Kaizen"))


class RebuildError(alloc.AllocError):
    """Rebuild refused (registry is not lost)."""


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
    i = headers._header_index(lines)
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


def _tokens(value):
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
            toks = _tokens((fields or {}).get("Item", ""))
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
            ids = [a for a in _tokens(fields.get("Aliases", "")) if ":" not in a]
            if len(ids) > len(aliases):
                aliases = ids
            for a in _tokens(fields.get("Aliases", "")):
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


def build_events(items, key):
    """Event dicts (no evid/ts): ``new`` for each birth ID, ``move`` per alias step."""
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
        return birth_of.get(pid, pid)

    out = []
    for num, title in _SEEDS:
        sid = rules.format_id(key, [rules.Segment("I", num)])
        if sid not in birth_of:
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


def rebuild(root, roots, write=False, store=None):
    """Plan (and with ``write`` apply) the rebuild; return report lines."""
    root = Path(root)
    key = alloc.load_key(root)
    items, notes = collect([root / r for r in roots])
    evs = build_events(items, key)
    lines = [f"item {i['id']} slug={i['slug']} aliases={','.join(i['aliases']) or '-'}"
             for i in items]
    lines += [f"note: {n}" for n in notes] + [NOT_RECOVERABLE]
    if not write:
        return ["dry run: nothing written (use --write)"] + lines
    store = store or Store(root, key)
    with store.transaction() as tx:
        if tx.state.items:
            raise RebuildError(
                f"the registry is not lost ({len(tx.state.items)} items present): refusing to "
                "rebuild over it; delete both copies first if that is really intended")
        for ev in evs:
            tx.append(ev)
    store.flush()
    return [f"wrote {len(evs)} events"] + lines
