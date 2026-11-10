"""``headers sync``: write ``Item:`` / ``Aliases:`` into family documents (plan 3.7).

Header writing is the *second phase*, run after the allocation lock is released.
A family is the files named ``<slug>(\\d*)(-<role>)?.md`` under ``docs/<kind>/`` and
``*/docs/<kind>/``; a file that also matches a longer registered slug's pattern
belongs to that slug.  Only the one-line header block is edited, never the body.
"""
import re
from pathlib import Path

from . import alloc
from .store import Store

KINDS = ("requirements", "plans", "reviews", "test-plans", "test-reports", "manuals")
SEP = " · "
_FIELD = re.compile(r"\*\*(Item|Aliases):\*\*")


def _pattern(slug):
    return re.compile(re.escape(slug) + r"\d*(-[a-z]+)?\.md\Z")


def _owner(name, slugs):
    """Longest registered slug whose family pattern matches ``name``."""
    hits = [s for s in slugs if _pattern(s).match(name)]
    return max(hits, key=len) if hits else None


def _family_files(root, slug, slugs):
    """Family documents of ``slug``; only names matching its pattern are ever opened."""
    for kind in KINDS:
        for p in sorted([*root.glob(f"docs/{kind}/*.md"), *root.glob(f"*/docs/{kind}/*.md")]):
            if p.is_file() and _owner(p.name, slugs) == slug:
                yield p


def _alias_list(item):
    names = list(item.aliases)
    if item.legacy:
        names.append(f"{item.legacy['home']}:{item.legacy['name']}")
    return names


def _fields(item):
    out = [f"**Item:** `{item.id}`"]
    aliases = _alias_list(item)
    if aliases:
        out.append("**Aliases:** " + ", ".join(f"`{a}`" for a in aliases))
    return out


def _header_index(lines):
    """Index of the one-line header block: immediately under the H1, blank allowed."""
    if not lines or not lines[0].startswith("# "):
        return None
    i = 1
    if i < len(lines) and not lines[i].strip():
        i += 1
    return i if i < len(lines) and lines[i].startswith("> **Status:**") else None


def _rewrite(line, wanted):
    """``line`` with its Item/Aliases fields replaced by ``wanted``, other fields verbatim."""
    ending = line[len(line.rstrip("\r\n")):]
    parts = line[:len(line) - len(ending)].split(SEP)
    kept = [p for p in parts if not _FIELD.match(p.strip().removeprefix("> "))]
    return SEP.join(kept + wanted) + ending


def _sync_file(path, wanted):
    text = path.read_bytes().decode("utf-8")
    lines = text.splitlines(keepends=True)
    i = _header_index(lines)
    if i is None:
        return False
    new = _rewrite(lines[i], wanted)
    if new == lines[i]:
        return False
    lines[i] = new
    path.write_bytes("".join(lines).encode("utf-8"))
    return True


def _targets(st, id_):
    if id_ is None:
        return [i for i in st.items.values() if i.slug]
    uid = st.resolve_uid(id_)
    if uid is None:
        raise alloc.unknown_item(id_)
    out, todo = [], [uid]
    while todo:
        item = st.items[todo.pop()]
        out.append(item)
        todo += [i.uid for i in st.items.values() if i.parent == item.uid]
    return [i for i in out if i.slug]


def sync(root, id_=None, store=None):
    """Write headers for ``id_``'s subtree (default: every slugged item); return changed paths."""
    root = Path(root)
    st = (store or Store(root, alloc.load_key(root))).read_state()
    slugs = {i.slug for i in st.items.values() if i.slug}
    changed = []
    for item in _targets(st, id_):
        wanted = _fields(item)
        for path in _family_files(root, item.slug, slugs):
            if _sync_file(path, wanted):
                changed.append(path)
    return changed
