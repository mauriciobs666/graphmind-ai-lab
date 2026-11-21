"""`rebuild --dry-run`: the offline key and property validation of graph note v4 section 5.5 step 1.

Parses every covered file, builds every property map the store would write, and asserts the
invariants the first load into a shared FalkorDB must not break. **No database connection and no
`falkordb` import**: `build_load` reads files, `validate` is a pure function over the maps, so a
test can feed it hand-built loads (a crashing key, a cross-file duplicate).

Asserted, each as a named `Violation` (code, where, message):

* every UNIQUE value passes `safe_key()` (no `isspace()` edge, <= 512 UTF-8 bytes) and every indexed
  non-unique value passes `norm_value()`, is <= 512 bytes, non-empty and text-free;
* no duplicate across the whole load for `Source.path`, `Unit.key`, `Item.ref`, `KbFile.path`,
  `KbSection.key` (a scheme `Item.ref` in two files is named with both paths);
* property sets: `warnings`/`rowCount` on every `:Source`; the label, slug, `statusText`/
  `statusToken` and minimal-unit rules; list/int shapes; only properties the note defines;
* the label coverage probe; each `$meta`/`$rows` parameter under ~10 MB.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .discover import discover, kb_files
from .keys import MAX_KEY_BYTES, KeyHygieneError, norm_value, safe_key
from .parse.backlog import BacklogParseError, parse_backlog
from .parse.kb import parse_kb
from .parse.ledger import VOCABULARY, parse_ledger
from .parse.requirements import parse_requirements
from .parse.review import CLASSES, parse_review

PARAM_LIMIT = 10 * 1024 * 1024
ITEM_TYPES = frozenset("ISTFDK")
STATUS_TOKENS = frozenset(VOCABULARY) | {"unknown"}

# The property sets of graph note v4 section 2.2. A property outside them is a mismatch to report
# (parser or note), never to invent.
SOURCE_PROPS = frozenset({
    "path", "kind", "component", "slug", "docStatus", "owner", "tracks", "itemId", "aliases",
    "sha256", "bytes", "mtimeNs", "syncedAt", "parserVersion", "warnings", "rowCount",
})
SOURCE_EXTRA = {
    "ledger": {"citedReviews", "resolvedReviews"},
    "review": {"latestVerdict", "latestPass", "verdictClass"},
    "backlog": {"scope"},
    "requirement": set(),
}
SOURCE_REQUIRED = frozenset({"path", "kind", "component", "slug", "sha256", "bytes", "parserVersion",
                             "warnings", "rowCount"})
SOURCE_REQUIRED_EXTRA = {"ledger": {"citedReviews", "resolvedReviews"}, "review": {"latestPass", "verdictClass"},
                         "backlog": set(), "requirement": set()}
UNIT_MINIMAL = frozenset({"key", "unitKey", "unitLabel", "statusToken", "itemId", "ledgerPath", "slug",
                          "component", "tableIdx", "rowIdx"})
UNIT_FULL = UNIT_MINIMAL | {"unit", "needs", "needsError", "statusText", "owner", "agentId", "deliverable",
                            "gateVerdict", "cost", "question", "askedOn", "cites", "rowHash"}
UNIT_REQUIRED_MINIMAL = frozenset({"key", "unitKey", "unitLabel", "statusToken", "ledgerPath", "slug",
                                   "component", "tableIdx", "rowIdx"})
UNIT_REQUIRED_FULL = UNIT_REQUIRED_MINIMAL | {"unit", "cites", "rowHash"}
ITEM_PROPS = frozenset({"ref", "scope", "itemId", "backlogPath", "ord", "itemType", "title", "statusMarker",
                        "open", "priority", "milestone", "milestoneOrder", "milestoneApproved",
                        "milestoneStarted", "dependsOn", "touches", "itemHash"})
ITEM_REQUIRED = frozenset({"ref", "backlogPath", "ord", "title", "open", "itemHash"})
LIST_PROPS = frozenset({"aliases", "warnings", "citedReviews", "resolvedReviews", "needs", "cites",
                        "dependsOn", "touches"})
INT_PROPS = frozenset({"bytes", "parserVersion", "rowCount", "latestPass", "tableIdx", "rowIdx", "ord", "milestoneOrder"})

_ROLES = {"ledger": ("coordination",), "review": ("ml", "rca", "impl"), "requirement": (), "backlog": ()}
_NO_STATUS_TABLE = re.compile(r"^table (\d+): no Status column$")


@dataclass(frozen=True)
class Violation:
    code: str
    where: str
    message: str

    def render(self) -> str:
        return f"{self.code}\t{self.where}\t{self.message}"


@dataclass
class SourceLoad:
    kind: str
    path: str
    meta: dict
    rows: list[dict] = field(default_factory=list)


@dataclass
class KbLoad:
    path: str
    sha256: str
    section_keys: list[str]


@dataclass
class Load:
    sources: list[SourceLoad] = field(default_factory=list)
    kb: list[KbLoad] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)  # parse-time problems


@dataclass
class Report:
    kind_counts: Counter = field(default_factory=Counter)
    status_tally: Counter = field(default_factory=Counter)
    unknown_values: list[tuple[str, str, str | None]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    manifest: list[tuple[str, str]] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def render(self) -> str:
        out = ["sources: " + ", ".join(f"{k} {n}" for k, n in sorted(self.kind_counts.items())),
               "statusToken tally: " + ", ".join(f"{k} {n}" for k, n in sorted(self.status_tally.items()))]
        out += [f"unknown\t{p}\t{k}\t{raw!r}" for p, k, raw in self.unknown_values]
        out += [f"warning\t{p}\t{w}" for p, w in self.warnings]
        out += [f"manifest: {len(self.manifest)} (path, sha256) pairs"]
        out += [f"ERROR {v.render()}" for v in self.violations]
        out.append("dry-run: " + ("ok" if self.ok else f"{len(self.violations)} violation(s)"))
        return "\n".join(out)


# --- loading (reads files; no database) ---------------------------------------------------------

def build_load(root: str | Path, legacy_paths: Mapping[str, str] | None = None) -> Load:
    """Parse every covered file and the KB registry into the maps the store would write."""
    root = Path(root)
    found = discover(root)
    reviews = {f.path for f in found if f.kind == "review"}
    load = Load()
    for f in found:
        raw = (root / f.path).read_bytes()
        try:
            if f.kind == "ledger":
                doc = parse_ledger(f.path, raw, reviews)
                rows = doc.unit_props()
            elif f.kind == "backlog":
                doc = parse_backlog(f.path, raw, legacy_paths)
                rows = doc.item_props()
            elif f.kind == "review":
                doc, rows = parse_review(f.path, raw), []
            else:
                doc, rows = parse_requirements(f.path, raw), []
        except BacklogParseError as exc:
            load.violations.append(Violation("parse-error", f.path, str(exc)))
            continue
        load.sources.append(SourceLoad(f.kind, f.path, doc.source_props(), rows))
    for rel in kb_files(root):
        file = root / rel
        if not file.is_file():
            load.violations.append(Violation("kb-file-missing", rel, "listed in the KB registry, not on disk"))
            continue
        kb = parse_kb(rel, file.read_bytes())
        keys = []
        for s in kb.sections:
            try:
                keys.append(safe_key(rel, s.heading_key))
            except KeyHygieneError as exc:
                load.violations.append(Violation("unique-key-unsafe", f"{rel}#{s.heading_key}", str(exc)))
        load.kb.append(KbLoad(rel, kb.sha256, keys))
    return load


# --- independent derivations ---------------------------------------------------------------------

def expected_slug(path: str, kind: str) -> str:
    """Per-kind slug (note section 2.2), derived here without `discover.slug_of`."""
    stem = path.rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.endswith(".md") else stem
    for role in _ROLES.get(kind, ()):
        if stem.endswith("-" + role):
            return stem[: -len(role) - 1]
    return stem


def expected_label(unit_key: str) -> str:
    """plan3 section 3.4 `Unit label`: up to the first whitespace or `(`, trailing `:,;` trimmed;
    the whole key when that is empty (so empty only for an empty key)."""
    if not unit_key:
        return ""
    m = re.match(r"[^\s(]+", unit_key)
    return (m.group(0).rstrip(":,;") if m else "") or unit_key


# --- validation (pure) -----------------------------------------------------------------------

class _Checker:
    def __init__(self) -> None:
        self.out: list[Violation] = []

    def add(self, code: str, where: str, message: str) -> None:
        self.out.append(Violation(code, where, message))

    def unique(self, value, where: str, what: str) -> None:
        if not isinstance(value, str):
            self.add("unique-key-unsafe", where, f"{what} is not a string: {value!r}")
            return
        try:
            ok = safe_key(value) == value
        except KeyHygieneError:
            ok = False
        if not ok:
            self.add("unique-key-unsafe", where,
                     f"{what} would crash or miss in a UNIQUE index (whitespace edge or > {MAX_KEY_BYTES} "
                     f"bytes): {value!r}")

    def indexed(self, value, where: str, what: str) -> None:
        if not isinstance(value, str):
            self.add("indexed-value-bad", where, f"{what} is not a string: {value!r}")
            return
        if value == "":
            self.add("indexed-value-bad", where, f"{what} is empty")
        elif norm_value(value) != value:
            self.add("indexed-value-bad", where, f"{what} has a whitespace edge: {value!r}")
        elif len(value.encode("utf-8")) > MAX_KEY_BYTES:
            self.add("indexed-value-bad", where, f"{what} is over {MAX_KEY_BYTES} bytes")
        elif any(c.isspace() for c in value) and what not in ("Unit.statusToken",):
            self.add("text-on-indexed-property", where, f"{what} holds text, not a token: {value!r}")
        elif what == "Unit.statusToken" and value not in STATUS_TOKENS:
            self.add("text-on-indexed-property", where, f"{what} is outside the closed set: {value!r}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_shapes(c: _Checker, props: Mapping, where: str) -> None:
    for name, value in props.items():
        if name in LIST_PROPS and not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            c.add("bad-shape", where, f"{name} must be a list of strings, got {value!r}")
        if name in INT_PROPS and not _is_int(value):
            c.add("bad-shape", where, f"{name} must be an int, got {value!r}")
        if name == "open" and not isinstance(value, bool):
            c.add("bad-shape", where, f"open must be a bool, got {value!r}")
        if value is None:
            c.add("bad-shape", where, f"{name} is None (a null is omitted, never written)")


def _check_source(c: _Checker, s: SourceLoad) -> None:
    where = s.path
    m = s.meta
    if s.kind not in _ROLES:
        c.add("bad-kind", where, f"unknown source kind {s.kind!r}")
        return
    for name in sorted(SOURCE_REQUIRED | SOURCE_REQUIRED_EXTRA[s.kind]):
        if name not in m:
            c.add("missing-property", where, f"{name} missing on :Source:{s.kind.title()}"
                  + (" (the parser must always supply it)" if name in ("warnings", "rowCount") else ""))
    allowed = SOURCE_PROPS | SOURCE_EXTRA[s.kind]
    for name in sorted(set(m) - allowed):
        c.add("unknown-property", where, f"{name} is not a note v4 property of :Source:{s.kind.title()}")
    _check_shapes(c, m, where)
    if m.get("path") != s.path or m.get("kind") != s.kind:
        c.add("meta-mismatch", where, f"meta path/kind {m.get('path')!r}/{m.get('kind')!r} differ from the source")
    if m.get("slug") != expected_slug(s.path, s.kind):
        c.add("slug-wrong", where, f"slug {m.get('slug')!r}, expected {expected_slug(s.path, s.kind)!r}")
    if _is_int(m.get("rowCount")) and m["rowCount"] != len(s.rows):
        c.add("rowcount-mismatch", where, f"rowCount {m['rowCount']} but {len(s.rows)} children")
    c.unique(m.get("path"), where, "Source.path")
    if "itemId" in m:
        c.indexed(m["itemId"], where, "Source.itemId")
    if s.kind == "ledger":
        c.indexed(m.get("slug"), where, "Ledger.slug")
    if s.kind == "backlog" and "scope" in m:
        c.indexed(m["scope"], where, "Backlog.scope")
    if s.kind == "review" and m.get("verdictClass") not in CLASSES:
        c.add("bad-shape", where, f"verdictClass {m.get('verdictClass')!r} outside {CLASSES}")
    if s.kind not in ("ledger", "backlog") and s.rows:
        c.add("bad-shape", where, f"a {s.kind} has no children, got {len(s.rows)}")


def _check_ledger(c: _Checker, s: SourceLoad) -> None:
    archived = s.meta.get("docStatus") == "archived"
    no_status = {int(g.group(1)) for w in s.meta.get("warnings", []) if isinstance(w, str)
                 and (g := _NO_STATUS_TABLE.match(w))}
    required = UNIT_REQUIRED_MINIMAL if archived else UNIT_REQUIRED_FULL
    allowed = UNIT_MINIMAL if archived else UNIT_FULL
    for u in s.rows:
        where = f"{s.path} {u.get('key')!r}"
        for name in sorted(required - set(u)):
            c.add("missing-property", where, f"{name} missing on :Unit")
        for name in sorted(set(u) - allowed):
            code = "minimal-unit-extra" if archived and name in UNIT_FULL else "unknown-property"
            c.add(code, where, f"{name} must not be written on this :Unit")
        _check_shapes(c, u, where)
        c.unique(u.get("key"), where, "Unit.key")
        if "itemId" in u:
            c.indexed(u["itemId"], where, "Unit.itemId")
        c.indexed(u.get("statusToken"), where, "Unit.statusToken")
        if "needs" in u and "needsError" in u:
            c.add("bad-shape", where, "needs and needsError are both present")
        if (u.get("ledgerPath"), u.get("slug"), u.get("component")) != (s.path, s.meta.get("slug"),
                                                                        s.meta.get("component")):
            c.add("denormalised-mismatch", where, "ledgerPath/slug/component differ from the parent ledger")
        key = u.get("unitKey")
        label = u.get("unitLabel")
        if isinstance(key, str) and isinstance(label, str):
            if label != expected_label(key):  # empty exactly when the key is, by the derivation
                c.add("unit-label-wrong", where, f"unitLabel {label!r}, expected {expected_label(key)!r}")
        if u.get("tableIdx") in no_status:
            if "statusText" in u:
                c.add("status-text-present", where, f"statusText {u['statusText']!r} on a row of a table "
                      "with no Status column (must be absent, never '')")
            if u.get("statusToken") != "unknown":
                c.add("status-token-wrong", where, "a no-Status-column row must read unknown")
    _label_probe(c, s)


def _label_probe(c: _Checker, s: SourceLoad) -> None:
    """Every unit is returned by its label under the note 7.3 predicate, with every holder of it."""
    rows = [u for u in s.rows if isinstance(u.get("unitKey"), str) and isinstance(u.get("unitLabel"), str)]
    for u in rows:
        want = expected_label(u["unitKey"])
        if not want:
            continue
        got = {r["key"] for r in rows if r["unitKey"] == want or r["unitLabel"] == want}
        holders = {r["key"] for r in rows if expected_label(r["unitKey"]) == want}
        if u["key"] not in got or not holders <= got:
            c.add("label-probe", f"{s.path} {u['key']!r}", f"label {want!r} does not return all its holders")


def _check_backlog(c: _Checker, s: SourceLoad) -> None:
    scope = s.meta.get("scope")
    for it in s.rows:
        where = f"{s.path} {it.get('ref')!r}"
        for name in sorted(ITEM_REQUIRED - set(it)):
            c.add("missing-property", where, f"{name} missing on :Item")
        for name in sorted(set(it) - ITEM_PROPS):
            c.add("unknown-property", where, f"{name} is not a note v4 property of :Item")
        _check_shapes(c, it, where)
        c.unique(it.get("ref"), where, "Item.ref")
        if it.get("backlogPath") != s.path:
            c.add("denormalised-mismatch", where, "backlogPath differs from the parent backlog")
        if "itemType" in it and it["itemType"] not in ITEM_TYPES:
            c.add("bad-shape", where, f"itemType {it['itemType']!r} not in I S T F D K")
        ref, iid = it.get("ref"), it.get("itemId")
        if isinstance(ref, str) and isinstance(iid, str):
            legacy = len(iid.split("-")[0]) == 1  # `K-065`; a scheme ID starts with a 2-6 letter project key
            expected = f"{scope or 'unscoped'}:{iid}" if legacy else iid
            if ref != expected:
                c.add("ref-form", where, f"ref {ref!r}, expected {expected!r} "
                      f"({'legacy: <scope>:<id>' if legacy else 'scheme: the ID as written'})")
            if legacy == ("itemType" in it):
                c.add("ref-form", where, "itemType is present exactly for a scheme ID")


def _duplicates(c: _Checker, load: Load) -> None:
    paths: dict[str, str] = {}
    units: dict[str, str] = {}
    items: dict[str, tuple[str, object]] = {}
    for s in load.sources:
        if s.path in paths:
            c.add("duplicate-unique", s.path, f"Source.path {s.path!r} loaded twice")
        paths[s.path] = s.path
        for u in s.rows if s.kind == "ledger" else []:
            k = u.get("key")
            if k in units:
                c.add("duplicate-unique", f"{s.path} {k!r}",
                      f"Unit.key {k!r} in both {units[k]} and {s.path}")
            units.setdefault(k, s.path)
        for it in s.rows if s.kind == "backlog" else []:
            ref = it.get("ref")
            if ref in items:
                other_path, other_ord = items[ref]
                kind = "scheme ref" if ":" not in str(ref) else "ref"
                hint = "; a wi move may be in progress" if other_path != s.path else ""
                c.add("duplicate-unique", f"{s.path} {ref!r}",
                      f"{kind} {ref} headed in both {other_path} (ord {other_ord}) and {s.path} "
                      f"(ord {it.get('ord')}){hint}")
            items.setdefault(ref, (s.path, it.get("ord")))
    kb_paths: set[str] = set()
    sections: dict[str, str] = {}
    for kb in load.kb:
        c.unique(kb.path, kb.path, "KbFile.path")
        if kb.path in kb_paths:
            c.add("duplicate-unique", kb.path, f"KbFile.path {kb.path!r} listed twice")
        kb_paths.add(kb.path)
        for k in kb.section_keys:
            c.unique(k, kb.path, "KbSection.key")
            if k in sections:
                c.add("duplicate-unique", kb.path, f"KbSection.key {k!r} in both {sections[k]} and {kb.path}")
            sections.setdefault(k, kb.path)


def _param_sizes(c: _Checker, s: SourceLoad) -> None:
    for name, value in (("$meta", s.meta), ("$rows", s.rows)):
        size = len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        if size > PARAM_LIMIT:
            c.add("parameter-too-large", s.path, f"{name} is {size} bytes, over {PARAM_LIMIT}")


def validate(load: Load) -> Report:
    """Every note section 5.5 step 1 assertion over `load`; never touches a database."""
    c = _Checker()
    c.out.extend(load.violations)
    report = Report()
    for s in load.sources:
        report.kind_counts[s.kind] += 1
        report.manifest.append((s.path, str(s.meta.get("sha256"))))
        report.warnings += [(s.path, w) for w in s.meta.get("warnings", []) if isinstance(w, str)]
        _check_source(c, s)
        if s.kind == "ledger":
            _check_ledger(c, s)
            for u in s.rows:
                report.status_tally[str(u.get("statusToken"))] += 1
                if u.get("statusToken") == "unknown":
                    report.unknown_values.append((s.path, str(u.get("key")), u.get("statusText")))
        elif s.kind == "backlog":
            _check_backlog(c, s)
        _param_sizes(c, s)
    for kb in load.kb:
        report.manifest.append((kb.path, kb.sha256))
    _duplicates(c, load)
    report.manifest.sort()
    report.violations = c.out
    return report


def rebuild_dry_run(root: str | Path, legacy_paths: Mapping[str, str] | None = None) -> Report:
    """`coordgraph rebuild --dry-run`: parse everything under `root`, validate, no database."""
    return validate(build_load(root, legacy_paths))
