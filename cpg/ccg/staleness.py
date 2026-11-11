"""Staleness: the closed ``Unit.failReason`` vocabulary (``-p1-graph.md`` §1 G4).

Created by P1-18 with the constant; P1-20 adds ``unit_staleness`` and ``local_head``, the one staleness module.
A ``failReason`` is **value-free**: one code of :data:`FAIL_REASONS`, never exception text, source text or a configuration value.
A new code is an edit of this constant and of the G4 row.
"""
from __future__ import annotations

import re
import subprocess
from typing import NamedTuple, Protocol

from ccg import gitsafe

#: What every report that shows a head says about it: P1 never fetches (index §6; the manifest loader rejects a ``remote`` key).
HEAD_NOTE = "head is the local clone's"
_COMMIT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
MODES = ("files", "commits")

FAIL_REASONS: tuple[str, ...] = (
    "manifest:never-built",
    "build:shallow-clone",
    "extract:failed",
    "extract:timeout",
    "bundle:invalid",
    "bundle:schema-minor",
    "bundle:bind-key-check",
    "bundle:audit-missing",
    "bundle:unindexable",
)


class _Built(Protocol):
    rootPath: str
    builtCommit: str


class Staleness(NamedTuple):
    """``stale`` is True iff the unit's built commit is behind the head by the ``basis`` rule. A count is None when it was not
    computed (``basis`` is the other rule) or could not be (``basis`` ``"unknown-commit"``: the local clone lacks the built commit,
    which is stale by definition). Never carries a path or a git message."""
    stale: bool
    built_commit: str
    head: str
    basis: str
    commits_behind: int | None = None
    files_changed: int | None = None


def _check(kind: str, commit) -> str:
    if not isinstance(commit, str) or not _COMMIT_ID.fullmatch(commit):
        raise ValueError(f"{kind} must be a full lowercase object id")
    return commit


def local_head(repo) -> str:
    """The full commit id of ``repo``'s checked-out ``HEAD``, read from the local clone (no fetch, no network)."""
    out = gitsafe.git(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    return _check("head", out)


def unit_staleness(unit: _Built, head: str, *, repo, mode: str = "files") -> Staleness:
    """Is ``unit`` (anything with ``rootPath`` and ``builtCommit``) behind ``head``, the **local** clone's head (``HEAD_NOTE``)?

    ``mode="files"`` (the default): stale iff a file under the unit's root differs between its built commit and ``head`` (a commit that
    touches only other directories leaves the unit fresh). ``mode="commits"``: stale iff ``head`` has a commit the built commit lacks.
    The same commit is fresh in both. No remote is contacted; ``repo`` is the local clone."""
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}")
    built, head = _check("built commit", unit.builtCommit), _check("head", head)
    if built == head:
        return Staleness(False, built, head, mode, 0, 0)
    try:
        if mode == "commits":
            n = int(gitsafe.git(repo, "rev-list", "--count", f"{built}..{head}").decode().strip())
            return Staleness(n > 0, built, head, mode, commits_behind=n)
        paths = () if unit.rootPath == "." else (unit.rootPath,)
        out = gitsafe.git(repo, "diff", "--name-only", "-z", "--no-renames", built, head, paths=paths)
        n = sum(1 for f in out.split(b"\0") if f)
        return Staleness(n > 0, built, head, mode, files_changed=n)
    except (subprocess.CalledProcessError, ValueError):               # the clone has no such object: stale, and nothing more is known
        return Staleness(True, built, head, "unknown-commit")
