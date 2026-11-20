"""`coordgraph lint`: backlog checks that are errors in the file, found when written.

Read-side only: `lint` may import `wilib` (the canonical ID grammar, `wilib.rules`), the sync
path never does. Reference resolution is an injected `Candidates` callable (default: no
registry, `shapes.offline_candidates`) so S1d's registry-backed one plugs in unchanged.
"""
from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import package_repo_root
from .discover import discover
from .history import HistoryIndex, find_id_tokens
from .parse.backlog import BacklogDoc, BacklogParseError, Item, parse_backlog
from .scope import load_legacy_paths
from .shapes import Candidates, Reason, classify, offline_candidates

MAP_OPEN = "<!-- coordgraph:milestone-map -->"
MAP_CLOSE = "<!-- /coordgraph:milestone-map -->"
_MILESTONE_TOKEN = re.compile(r"\b(?:ahead-of-)?M\d+(?:\.\d+)?\b")
# Parser warnings that lint reports itself as errors, so they are not repeated.
_COVERED = ("no Status field", "unrecognised Status", "Status emoji does not match")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int | None
    severity: str  # "error" | "warn"
    code: str
    message: str

    def render(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.severity}: {self.code}: {self.message}"


def default_wi_root() -> Path:
    return Path(os.environ.get("COORDGRAPH_WI_ROOT") or package_repo_root() / "skills" / "work-items" / "scripts")


def load_wilib_rules():
    """`wilib.rules` or None when `wilib` cannot be imported (lint then skips canonical checks)."""
    root = str(default_wi_root())
    if root not in sys.path:
        sys.path.append(root)
    try:
        from wilib import rules  # noqa: PLC0415 - read side only, deliberately lazy
    except ImportError:
        return None
    return rules


def _canonical_problem(rules, token: str) -> str | None:
    try:
        rules.validate_path(rules.parse(token))
    except (rules.MalformedId, rules.IllegalParent) as exc:
        return str(exc)
    return None


def lint_repo(
    root: str | Path,
    paths: Iterable[str] | None = None,
    candidates: Candidates = offline_candidates,
    rules=None,
    reason: Reason | None = None,
) -> list[Finding]:
    """Lint every covered backlog (or `paths`); `rules` defaults to the real `wilib.rules`."""
    root = Path(root)
    rules = load_wilib_rules() if rules is None else rules
    legacy_paths = load_legacy_paths(root)
    backlogs = [f.path for f in discover(root) if f.kind == "backlog"]
    chosen = backlogs if paths is None else [p for p in backlogs if p in set(paths)]
    findings: list[Finding] = []
    if rules is None:
        findings.append(Finding(".", None, "warn", "wilib-unavailable",
                                "wilib not importable: canonical ID checks skipped"))
    docs: dict[str, BacklogDoc] = {}
    for path in backlogs:
        try:
            docs[path] = parse_backlog(path, (root / path).read_bytes(), legacy_paths)
        except BacklogParseError as exc:
            if path in chosen:
                findings.append(Finding(path, None, "error", "duplicate-id", str(exc)))
    history = HistoryIndex.from_repo(root, backlogs, legacy_paths)
    known_refs = {i.ref for d in docs.values() for i in d.items}
    owners: dict[str, str] = {}
    for path in chosen:
        doc = docs.get(path)
        if doc is None:
            continue
        text = (root / path).read_text(encoding="utf-8", errors="replace")
        for w in doc.warnings:
            if not any(c in w for c in _COVERED):
                findings.append(Finding(path, None, "warn", "parse", w))
        for item in doc.items:
            findings.extend(_item_findings(item, rules, history, known_refs, candidates, owners, reason))
        findings.extend(_milestone_findings(doc))
        findings.extend(_map_findings(doc, text))
    return findings


def _item_findings(item: Item, rules, history: HistoryIndex, known_refs: set[str],
                   candidates: Candidates, owners: dict[str, str],
                   reason: Reason | None = None) -> list[Finding]:
    p, ln = item.backlog_path, item.line
    out: list[Finding] = []
    label = item.item_id or item.ref
    if item.status is None:
        out.append(Finding(p, ln, "error", "status", f"{label}: no recognised Status field"))
    elif item.status == "delivered":
        out.append(Finding(p, ln, "error", "delivered-in-backlog",
                           f"{label}: delivered items leave the backlog (record is HISTORY.md)"))
    if item.item_id and classify(item.item_id) == "scheme":
        if item.ref in owners and owners[item.ref] != p:
            out.append(Finding(p, ln, "error", "duplicate-ref",
                               f"{item.ref} is headed in {owners[item.ref]} and {p} "
                               "(a `wi move` may be in progress)"))
        owners.setdefault(item.ref, p)
        if rules is not None and (problem := _canonical_problem(rules, item.item_id)):
            out.append(Finding(p, ln, "error", "malformed-id", f"{item.item_id}: {problem}"))
    for token in item.depends_on or []:
        kind = classify(token)
        if kind is None:
            out.append(Finding(p, ln, "error", "unresolvable-dependency",
                               f"{label}: unrecognised Depends on token {token!r}"))
            continue
        if kind == "scheme" and rules is not None and (problem := _canonical_problem(rules, token)):
            out.append(Finding(p, ln, "error", "malformed-id", f"{label}: Depends on {token}: {problem}"))
            continue
        scope = item.scope or "unscoped"
        if candidates(token, scope) & known_refs:
            continue
        if history.delivered(token, scope, candidates) is not None:
            continue
        why = reason(token, scope) if reason is not None else None
        out.append(Finding(
            p, ln, "error", "unresolvable-dependency",
            f"{label}: unknown dependency {token}" + (f" ({why})" if why else "")
            + " - if it was delivered, drop the ref or add a "
            "`<ID> delivered` history heading; otherwise fix the id"))
    return out


def _milestone_findings(doc: BacklogDoc) -> list[Finding]:
    out: list[Finding] = []
    by_milestone: dict[str, list[Item]] = {}
    for item in doc.items:
        if item.open and item.milestone:
            by_milestone.setdefault(item.milestone, []).append(item)
    for token, items in by_milestone.items():
        started = {i.milestone_started for i in items}
        if len(started) > 1:
            out.append(Finding(doc.path, items[0].line, "error", "milestone-started",
                               f"items of milestone {token} disagree on `started`: "
                               f"{sorted(s or 'absent' for s in started)}"))
        for i in items:
            if i.status == "in-progress" and i.milestone_started is None:
                out.append(Finding(doc.path, i.line, "error", "milestone-started",
                                   f"{i.item_id or i.ref} is in progress but milestone {token} has no `started`"))
    return out


def _map_findings(doc: BacklogDoc, text: str) -> list[Finding]:
    if MAP_OPEN not in text or MAP_CLOSE not in text:
        return []
    block = text.split(MAP_OPEN, 1)[1].split(MAP_CLOSE, 1)[0]
    by_id = {i.item_id: i for i in doc.items if i.item_id}
    rows: dict[str, set[str]] = {}
    for line in block.splitlines():
        if not line.lstrip().startswith("|") or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = _MILESTONE_TOKEN.search(cells[0]) if cells else None
        if m:
            rows.setdefault(m.group(0), set()).update(find_id_tokens(" ".join(cells[1:])))
    out: list[Finding] = []
    for token, ids in rows.items():
        for ident in sorted(ids):
            item = by_id.get(ident)
            if item is not None and item.open and item.milestone != token:
                out.append(Finding(doc.path, item.line, "error", "milestone-map",
                                   f"map row {token} names open {ident} whose Milestone is "
                                   f"{item.milestone or 'absent'}"))
    for item in doc.items:
        if item.open and item.milestone in rows and item.item_id not in rows[item.milestone]:
            out.append(Finding(doc.path, item.line, "error", "milestone-map",
                               f"{item.item_id or item.ref} has Milestone {item.milestone} but is "
                               "absent from that map row"))
    return out


def item_counts(root: str | Path, paths: Iterable[str] | None = None) -> dict[str, int]:
    """Items per backlog file (`lint --count`); a file that does not parse counts as -1."""
    root = Path(root)
    legacy_paths = load_legacy_paths(root)
    out: dict[str, int] = {}
    for f in discover(root):
        if f.kind != "backlog" or (paths is not None and f.path not in set(paths)):
            continue
        try:
            out[f.path] = len(parse_backlog(f.path, (root / f.path).read_bytes(), legacy_paths).items)
        except BacklogParseError:
            out[f.path] = -1
    return out


