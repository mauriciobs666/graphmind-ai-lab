"""``wi check``: every rule of plan 3.8 (FR-9, FR-20; AC-7, AC-14).

Read-only.  Reports registry problems (duplicate ``new`` events, illegal parents),
header problems of family documents (no ID, mismatch, malformed, dangling, duplicate,
FR/AC absent), ``--set`` entries, and legacy-namespace definers.  The malformed scan
reads only the header fields ``Item:``/``Tracks:``/``Aliases:`` and ``--set`` ids, so
prose placeholders and legacy ids (``K-030``) never fire.  Unflushed journal events and
duplicate registry lines are informational.
"""
import fnmatch
import json
import re
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

from . import alloc, events, headers, project, rebuild, rules
from .store import Store, read_text

Finding = namedtuple("Finding", "kind path detail informational", defaults=(False,))
NAMESPACE_CANDIDATES = ("docs/BACKLOG.md", "*/docs/BACKLOG.md", "claude/*/kaizen/plan.md")
_DEFINER = re.compile(r"^(?:#{2,4} |- \*\*|\| )([KC]-)\d+")
_HEADER_FIELDS = ("Item", "Tracks", "Aliases")


class CheckError(alloc.AllocError):
    """The check could not run (bad ``--set`` file)."""


def format_finding(f):
    where = f"{f.path}: " if f.path else ""
    return f"{'info ' if f.informational else ''}{f.kind}: {where}{f.detail}"


def _in_scope(path, root, paths):
    if not paths:
        return True
    return any(path == (root / p) or (root / p) in path.parents for p in paths)


# ------------------------------------------------------------------ registry
def _registry_findings(st, union, store):
    out = []
    ids = defaultdict(set)
    for ev in union:
        if ev["ev"] == "new":
            ids[ev["id"]].add(ev["evid"])
    for id_, evids in sorted(ids.items()):
        if len(evids) > 1:
            out.append(Finding("duplicate", "", f"{id_}: {len(evids)} distinct new events"))
    for item in sorted(st.items.values(), key=lambda i: i.id):
        try:
            segs = rules.parse(item.id).segments
            rules.validate_path(rules.parse(item.id))
        except (rules.MalformedId, rules.IllegalParent) as exc:
            out.append(Finding("parent", "", f"{item.id}: {exc}"))
            continue
        want = rules.format_id(rules.parse(item.id).project, segs[:-1]) if len(segs) > 1 else None
        parent = st.items.get(item.parent) if item.parent else None
        have = parent.id if parent else None
        if have != want:
            out.append(Finding("parent", "", f"{item.id}: parent is {have}, its path says {want}"))
    tracked = store.read_tracked()
    dup = sum(n - 1 for n in Counter(e["evid"] for e in tracked).values())
    if dup:
        out.append(Finding("duplicate-lines", str(store.registry_path.name),
                           f"{dup} duplicate registry line(s); the next flush removes them", True))
    if store.journal_path is not None:
        have = {e["evid"] for e in tracked}
        loose = [e for e in store.read_journal() if e["evid"] not in have]
        if loose:
            out.append(Finding("unflushed", "", f"{len(loose)} journal event(s) not in the "
                               "tracked registry: run wi flush", True))
    return out


# ------------------------------------------------------------------ ids
def _requirement_text(root, st, slugs, feature_uid):
    """Text of the feature's requirements documents, or None when it has none."""
    slug = st.items[feature_uid].slug
    if not slug:
        return None
    docs = [p for p in headers.family_files(root, slug, slugs) if p.parent.name == "requirements"]
    return "\n".join(p.read_text(encoding="utf-8") for p in docs) if docs else None


def _classify(token, root, st, slugs):
    """Finding ``(kind, detail)`` for one scheme-shaped token, or None when it is fine."""
    try:
        parsed = rules.parse(token)
        rules.validate_path(parsed)
    except rules.MalformedId as exc:
        return "malformed", f"{token}: {exc}"
    except rules.IllegalParent as exc:
        return "parent", f"{token}: {exc}"
    tail = parsed.segments[-1]
    if len(parsed.segments) > 1 and tail.type in ("FR", "AC"):
        feature = st.by_id.get(rules.format_id(parsed.project, parsed.segments[:-1]))
        if feature is None:
            return "dangling", f"{token}: no such feature"
        text = _requirement_text(root, st, slugs, feature)
        label = f"{tail.type}-{tail.num}{tail.suffix}"
        if text is not None and not re.search(
                rf"^[\s\-*|#]*\**{label}(?![A-Za-z0-9])", text, re.M):
            return "absent", f"{token}: {label} is not defined in the requirements document"
        return None
    if token not in st.by_id:
        return "dangling", f"{token}: no new event or alias (forgotten flush?)"
    return None


def _token_re(key):
    return re.compile(rf"(?<![\w-]){re.escape(key)}-[\w-]*")


# ------------------------------------------------------------------ documents
def _document_findings(root, st, slugs, key, baseline, paths):
    out = []
    shared = defaultdict(set)
    token_re = _token_re(key)
    by_slug = defaultdict(list)
    for item in sorted(st.items.values(), key=lambda i: i.id):
        if item.slug:
            by_slug[item.slug].append(item)
    for path in rebuild.iter_docs(root):
        if not _in_scope(path, root, paths):
            continue
        rel = path.relative_to(root).as_posix()
        fields = rebuild.header_fields(path) or {}
        item_toks = rebuild.tokens(fields.get("Item", ""))
        item_tok = item_toks[0] if item_toks else None
        for name in _HEADER_FIELDS:
            for token in token_re.findall(fields.get(name, "")):
                hit = _classify(token, root, st, slugs)
                if hit:
                    out.append(Finding(hit[0], rel, f"{name}: {hit[1]}"))
        owner = headers.owner(path.name, slugs)
        if path.parent.name == "requirements" and rel not in baseline and item_tok is None:
            out.append(Finding("no-id", rel, "requirements document has no Item: header"))
        elif owner and item_tok is None:
            out.append(Finding("family-no-item", rel, f"family of {owner!r} lacks Item:"))
        elif owner:
            items = by_slug[owner]
            if len(items) > 1:
                out.append(Finding("duplicate", rel, f"slug {owner!r} is held by "
                                   + ", ".join(i.id for i in items)))
            uid = st.by_id.get(item_tok)
            if uid is not None and st.items[uid].slug != owner:
                out.append(Finding("item-slug", rel, f"Item {item_tok} belongs to slug "
                                   f"{st.items[uid].slug!r}, not {owner!r}"))
            elif item_tok != items[0].id:
                out.append(Finding("family-mismatch", rel,
                                   f"Item {item_tok} differs from registry ID {items[0].id}"))
        if item_tok:
            shared[item_tok].add(owner or rebuild.stem_slug(path.name))
    for tok, names in sorted(shared.items()):
        if len(names) > 1:
            out.append(Finding("duplicate", "", f"Item {tok} is shared by slugs {sorted(names)}"))
    return out


# ------------------------------------------------------------------ --set
def _set_findings(set_file, root, st, slugs, key):
    try:
        data = json.loads(Path(set_file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckError(f"cannot read --set file {set_file}: {exc}") from None
    entries = data.get("items") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise CheckError("--set file must be a JSON list of {id: ...} entries")
    out, seen = [], defaultdict(list)
    for n, entry in enumerate(entries):
        where = f"{Path(set_file).name}#{n}"
        id_ = entry.get("id") if isinstance(entry, dict) else None
        if not id_ or not isinstance(id_, str):
            out.append(Finding("no-id", where, "entry has no ID"))
            continue
        seen[id_].append(where)
        if _token_re(key).match(id_):
            hit = _classify(id_, root, st, slugs)
        else:       # a legacy id (K-030) is not in the scheme: never an error, but never silent
            out.append(Finding("legacy-id", where, f"{id_} is not a {key} ID; not checked", True))
            continue
        if hit:
            out.append(Finding(hit[0], where, hit[1]))
    for id_, wheres in sorted(seen.items()):
        if len(wheres) > 1:
            out.append(Finding("duplicate", wheres[0], f"{id_} is shared by entries {wheres}"))
    return out


# ------------------------------------------------------------------ namespaces
def _home_of(rel, mapping):
    for pattern, home in mapping.items():
        rx = "".join("([^/]+)" if c == "*" else re.escape(c) for c in pattern)
        m = re.fullmatch(rx, rel)
        if m:
            return home.replace("$1", m.group(1)) if m.groups() else home
    return None


def _namespace_findings(root, paths):
    cfg = json.loads((root / ".work-items" / "project.json").read_text(encoding="utf-8"))
    mapping = cfg.get("legacy_paths") or {}
    namespaces = project.load_namespaces(root)
    out = []
    cands = sorted({p for pat in NAMESPACE_CANDIDATES for p in root.glob(pat) if p.is_file()
                    and fnmatch.fnmatch(p.relative_to(root).as_posix(), pat)})
    for path in cands:
        if not _in_scope(path, root, paths):
            continue
        rel = path.relative_to(root).as_posix()
        prefixes = {m.group(1) for line in path.read_text(encoding="utf-8").splitlines()
                    if (m := _DEFINER.match(line))}
        home = _home_of(rel, mapping)
        for prefix in sorted(prefixes):
            if home is None:
                out.append(Finding("namespace", rel, f"defines {prefix} items but has no "
                                   "legacy_paths entry giving its home"))
            elif home not in namespaces.get(prefix, []):
                out.append(Finding("namespace", rel, f"defines {prefix} items but home {home!r} "
                                   "is not in legacy_namespaces"))
    return out


# ------------------------------------------------------------------ rebuilt
def _rebuilt_findings(root):
    cfg = json.loads((root / ".work-items" / "project.json").read_text(encoding="utf-8"))
    marker = cfg.get("rebuilt")
    if not marker:
        return []
    return [Finding("rebuilt", ".work-items/project.json",
                    f"the registry was rebuilt ({json.dumps(marker, sort_keys=True)}); IDs that "
                    "reached no commit, header or floor may have been reissued; remove the "
                    "'rebuilt' key once reviewed", True)]


# ------------------------------------------------------------------ entry
def check(root, set_file=None, paths=None, store=None):
    """Run every rule; return findings (informational ones have ``informational=True``)."""
    root = Path(root)
    key = alloc.load_key(root)
    store = store or Store(root, key)
    union = store.read_union()
    st = events.fold(union)
    slugs = {i.slug for i in st.items.values() if i.slug}
    baseline = set(read_text(root / ".work-items" / "baseline.txt").split())
    paths = list(paths or [])
    out = _registry_findings(st, union, store)
    out += _document_findings(root, st, slugs, key, baseline, paths)
    if set_file:
        out += _set_findings(set_file, root, st, slugs, key)
    out += _namespace_findings(root, paths)
    out += _rebuilt_findings(root)
    return out
