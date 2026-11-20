"""Which files the mirror covers (plan section 3.3 step 1; graph note section 3.1).

One glob set, Python `glob(..., recursive=True)` semantics (`**` also matches zero
directories, so the repo-root `docs/` tree is covered; hidden directories and files are
never matched):

    **/docs/plans/*-coordination.md    ledger
    **/docs/reviews/*.md               review
    **/docs/requirements/*.md          requirement
    **/docs/BACKLOG.md                 backlog
    claude/*/kaizen/plan.md            backlog (agent plan)

plus the KB registry (`coordination-graph/kb-files.txt`). Excluded: `deprecated/**` at the
repo root and any path with an `archive/` segment. `.git`, `node_modules` and `.venv` are
pruned while walking, a pure speed measure shared with the hook's `find`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

KINDS = ("ledger", "review", "requirement", "backlog")
ROLES = ("coordination", "ml", "graph", "rca", "impl", "report")
PRUNE_DIRS = frozenset({".git", "node_modules", ".venv"})
EXCLUDED_SEGMENT = "archive"
EXCLUDED_ROOT = "deprecated"
KB_REGISTRY = "coordination-graph/kb-files.txt"


@dataclass(frozen=True, order=True)
class Found:
    path: str  # repo-root-relative, `/`-separated
    kind: str  # one of KINDS


def kind_of(path: str) -> str | None:
    """The source kind of a repo-relative path, or None when no coverage glob matches."""
    parts = PurePosixPath(path).parts
    if not parts or any(p.startswith(".") for p in parts):
        return None
    if parts[0] == EXCLUDED_ROOT or EXCLUDED_SEGMENT in parts[:-1]:
        return None
    if PRUNE_DIRS.intersection(parts[:-1]):
        return None
    name = parts[-1]
    parent = parts[-3:-1]
    if parent == ("docs", "plans") and name.endswith("-coordination.md"):
        return "ledger"
    if parent == ("docs", "reviews") and name.endswith(".md"):
        return "review"
    if parent == ("docs", "requirements") and name.endswith(".md"):
        return "requirement"
    if name == "BACKLOG.md" and len(parts) >= 2 and parts[-2] == "docs":
        return "backlog"
    if len(parts) == 4 and parts[0] == "claude" and parts[2] == "kaizen" and name == "plan.md":
        return "backlog"
    return None


def discover(root: str | Path) -> list[Found]:
    """Every covered coordination file under `root`, sorted by path."""
    root = Path(root)
    found: list[Found] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in PRUNE_DIRS and not d.startswith(".") and d != EXCLUDED_SEGMENT
            and not (rel_dir == "" and d == EXCLUDED_ROOT)
        )
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            kind = kind_of(rel)
            if kind is not None and (Path(dirpath) / name).is_file():
                found.append(Found(rel, kind))
    return sorted(found)


def kb_files(root: str | Path) -> list[str]:
    """The KB registry: repo-relative paths, one per line; blank lines and `#` comments skipped."""
    registry = Path(root) / KB_REGISTRY
    if not registry.is_file():
        return []
    out = []
    for line in registry.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


# Which closed-set role suffixes a kind's file name can carry (root AGENTS.md filename grammar):
# a ledger is `<slug>-coordination`, a review may be `-ml`/`-rca`/`-impl`, a requirement never has
# one. Stripping by kind keeps `requirements/coordination-graph.md` (a feature whose name ends in
# `-graph`) the slug `coordination-graph`, the join key with its ledger.
_KIND_ROLES = {
    "ledger": ("coordination",),
    "review": ("ml", "rca", "impl"),
    "requirement": (),
    "backlog": (),
}


def slug_of(path: str, kind: str | None = None) -> str:
    """Topic slug: the basename minus `.md` minus one role suffix the kind can carry.

    `kind` defaults to the path's covered kind; a path no coverage glob matches may carry
    any role of the closed set.
    """
    kind = kind or kind_of(path)
    roles = _KIND_ROLES.get(kind, ROLES) if kind else ROLES
    stem = PurePosixPath(path).name.removesuffix(".md")
    for role in roles:
        if stem.endswith("-" + role):
            return stem[: -len(role) - 1]
    return stem


def component_of(path: str) -> str:
    """The component a source belongs to (plan section 3.4a).

    First path segment, except the repo-root `docs/` tree: `docs/BACKLOG.md` -> `cpg`
    (root AGENTS.md), everything else under root `docs/` -> `root`.
    """
    parts = PurePosixPath(path).parts
    if len(parts) >= 2 and parts[0] == "docs":
        return "cpg" if parts == ("docs", "BACKLOG.md") else "root"
    return parts[0] if parts else "root"
