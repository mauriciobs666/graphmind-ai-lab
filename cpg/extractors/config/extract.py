"""Config extractor entry point (C-1001 P1-11, F-6): files in, ``configEntry`` and ``launchFact`` records out.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-11; spec: `-bundle-spec3.md` §1.6, §1.7, §5.17.

``extract_config(repo_id, files, key=...)`` takes the repo's config files as ``(path, bytes)`` pairs, dispatches each by
format (``formats.FORMATS``), and returns the records of the whole repository, the ``File.role`` of every file it read and a
report of counters. Rules decided elsewhere, by import:

- ``bindValue``, ``bindParts``, ``launchFact.default``/``defaultHash``, ``command``/``args`` and ``valueHash`` come only from
  ``extractors.common.writer_rules`` (P1-07); this module supplies the *inputs*: the entry's binding role (E3), the alias
  anchor (W3), the declared Compose services (W2) and the text. ``ConfigResult.header_fields`` is the version rule's
  (`-bundle-spec5.md` §1): ``"1.1"`` and ``bindKeyCheck`` exactly when some entry carries ``bindParts``, else ``{}``.
- A file that is not UTF-8 refuses the whole unit (``SourceRefused``, index D12). Text with no UTF-8 encoding inside a
  readable file (a YAML ``"\\uD800"``) drops that entry or fact, counted in ``unencodableEntries``.
- The text of a value never leaves this module except as a gated field and a hash: the report holds counts and paths only,
  and no exception message carries file text.
"""
from __future__ import annotations

import posixpath
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ccg import ids
from extractors.common import writer_rules as wr
from extractors.common.errors import SourceRefused

from . import formats as F
from . import launch as L

MAX_FACTS = 5_000
_VAR = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def file_role(path: str) -> str:
    """``File.role`` of a config file (§5.2): ``example`` for a basename matching ``*.example*``, ``*.sample*``, ``example*`` or
    ``sample*``, or a file under an ``examples/``/``samples/`` directory; else ``primary``."""
    base = path.rsplit("/", 1)[-1].lower()
    parts = path.lower().split("/")
    if (re.search(r"\.example", base) or re.search(r"\.sample", base) or base.startswith("example")
            or base.startswith("sample") or "examples" in parts[:-1] or "samples" in parts[:-1]):
        return "example"
    return "primary"


@dataclass
class ConfigResult:
    records: list[dict] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)            # path -> File.role, for every file that yielded records
    declared_services: frozenset[str] = frozenset()
    report: dict = field(default_factory=dict)
    header_fields: dict = field(default_factory=dict)              # header fields to merge when these records go in one bundle


def _clean_path(path: object) -> str:
    if not (isinstance(path, str) and path and not path.startswith("/") and "\\" not in path and "\x00" not in path
            and posixpath.normpath(path) == path and not path.startswith("../") and path != ".."):
        raise SourceRefused("a config file name is not a repository-relative POSIX path")
    return path


def _decode(path: str, raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise SourceRefused(f"{path!r} is not UTF-8 ({e.reason} at byte {e.start}); the unit cannot be normalized") from None


def extract_config(repo_id: str, files: Mapping[str, bytes] | Iterable[tuple[str, bytes]], *, key: bytes,
                   extra_declared_services: Iterable[str] = ()) -> ConfigResult:
    """Records of every config file in ``files``. ``key`` is the system's HMAC key (the caller reads it through
    ``writer_rules.system_key``; tests pass a test key). ``extra_declared_services`` adds service names the caller knows from
    Compose files outside ``files`` (W2); the Compose files in ``files`` always count."""
    items = sorted(files.items() if isinstance(files, Mapping) else files)
    report: Counter = Counter()
    skipped: list[dict] = []
    texts: dict[str, str] = {}
    rows: dict[str, F.FormatRow] = {}
    sizes: dict[str, int] = {}
    for path, raw in items:
        _clean_path(path)
        row = F.classify(path)
        if row is None:
            report["ignoredFiles"] += 1
            continue
        texts[path] = _decode(path, raw)                          # D12: a non-UTF-8 file refuses the unit, before any output
        rows[path], sizes[path] = row, len(raw)
    trees: dict[str, F.Node] = {}
    for path, row in rows.items():
        if sizes[path] > F.MAX_BYTES:
            skipped.append({"path": path, "reason": "oversize"})
            continue
        try:
            trees[path] = row.parse(texts[path])
        except F.FormatError as e:
            skipped.append({"path": path, "reason": str(e)})
        except MemoryError:                                       # one file never aborts the unit
            skipped.append({"path": path, "reason": "memory"})
            report["memoryErrors"] += 1
    roles = {path: file_role(path) for path in trees}
    declared = {F.norm_name(s).lower() for s in extra_declared_services}
    for path, tree in trees.items():
        if rows[path].name == "compose" and roles[path] == "primary":      # an example file declares nothing (W2)
            declared |= F.compose_declared_services(tree)
    result = ConfigResult(declared_services=frozenset(declared))
    for path in sorted(trees):
        result.roles[path] = roles[path]
        mark = len(result.records)
        try:
            _file_records(repo_id, path, rows[path].name, trees[path], texts[path], key, declared, result, report)
        except MemoryError:
            del result.records[mark:]
            del result.roles[path]
            skipped.append({"path": path, "reason": "memory"})
            report["memoryErrors"] += 1
    result.header_fields = wr.bind_header_fields(key, result.records)
    report["files"] = len(trees)
    report["skippedFiles"] = len(skipped)
    result.report = {**{k: report[k] for k in ("files", "ignoredFiles", "skippedFiles", "configEntries", "launchFacts",
                                                "unencodableEntries", "droppedEntries", "truncatedFiles", "memoryErrors")},
                     "skipped": skipped}
    return result


def _file_records(repo_id, path, fmt, tree, text, key, declared, result, report) -> None:
    example = result.roles[path] == "example"
    entries: dict[str, dict] = {}
    drafts: list[L.Draft] = []
    walk = F.Walk(tree)
    count = 0
    for leaf in walk:
        count += 1
        if count > F.MAX_ENTRIES:
            walk.truncated = True
            break
        kpath = F.render_key_path(leaf.segs)
        if (any(isinstance(s, str) and len(s) > F.MAX_KEY_CHARS for s in leaf.segs) or len(kpath) > 2 * F.MAX_KEY_CHARS):
            report["droppedEntries"] += 1
            continue
        if not wr.encodable(kpath) or not wr.encodable(leaf.text):
            report["unencodableEntries"] += 1                     # the whole entry goes: valueHash is required, never hashed
            continue
        rid = ids.config_entry_id(repo_id, path, kpath)
        anchor = F.render_key_path(leaf.anchor_segs) if leaf.anchor_segs else None
        rec = {"type": "configEntry", "id": rid, "repoId": repo_id, "path": path, "line": leaf.line, "format": fmt,
               "keyPath": kpath, "example": example}
        rec.update(wr.value_hash_fields(key, rid, leaf.text))     # §1.7.5: domain is the entry's own id
        gate = dict(binding_role=F.binding_role(fmt, leaf.segs), declared_services=declared, via_alias=leaf.via_alias,
                    anchor_key_path=anchor)
        rec.update(wr.bind_fields(kpath, leaf.text, **gate))
        rec.update(wr.bind_parts_fields(key, kpath, leaf.text, **gate))      # spec5 §6.1: E1-E4; the keyed parts of a URL value
        entries[kpath] = rec                                      # a repeated keyPath: the last occurrence wins
        if not leaf.via_alias:
            drafts += L.env_defaults(leaf.text, leaf.line)
    if walk.truncated:
        report["truncatedFiles"] += 1
    drafts += _format_drafts(fmt, tree, text)
    result.records += entries.values()
    report["configEntries"] += len(entries)
    result.records += _launch_facts(repo_id, path, drafts, key, declared, report)


def _format_drafts(fmt: str, tree: F.Node, text: str) -> list[L.Draft]:
    if fmt == "dockerfile":
        return L.dockerfile_drafts(text)
    if fmt == "compose":
        return L.compose_drafts(tree)
    if fmt in ("mcp-json", "kiro-agent"):
        return L.mcp_drafts(tree)
    return []


def _launch_facts(repo_id: str, path: str, drafts: list[L.Draft], key: bytes, declared, report) -> list[dict]:
    keep: list[L.Draft] = []
    for d in drafts:
        texts = [*(d.argv or ()), *(t for t in (d.default_text, d.mount_path, d.service) if t is not None)]
        if all(wr.encodable(t) for t in texts):
            keep.append(d)
        else:
            report["unencodableEntries"] += 1
    out: list[dict] = []
    for kind in ("command", "port", "env-default", "mount"):
        group = sorted((d for d in keep if d.kind == kind), key=lambda d: d.line)      # stable: source order within a line
        if len(group) > MAX_FACTS:
            group = group[:MAX_FACTS]
            report["truncatedFiles"] += 1
        for ordinal, d in enumerate(group):
            out.append(_launch_fact(repo_id, path, d, ordinal, key, declared))
    report["launchFacts"] += len(out)
    return out


def _launch_fact(repo_id: str, path: str, d: L.Draft, ordinal: int, key: bytes, declared) -> dict:
    lid = ids.launch_fact_id(repo_id, path, d.kind, ordinal)
    rec = {"type": "launchFact", "id": lid, "path": path, "line": d.line, "ordinal": ordinal, "kind": d.kind,
           "origin": d.origin}
    if d.service is not None:
        rec["service"] = d.service
    if d.kind == "command":
        rec.update(wr.launch_argv_fields(list(d.argv), program_known=d.program_known))
    elif d.kind == "port":
        rec["port"], rec["protocol"] = d.port, d.protocol
    elif d.kind == "mount":
        rec["mountPath"] = d.mount_path
    else:
        rec["envVar"] = d.env_var
        rec.update(wr.default_fields(key, lid, d.env_var, d.default_text, binding_role=F.has_role_token(d.env_var),
                                     declared_services=declared))
    return rec
