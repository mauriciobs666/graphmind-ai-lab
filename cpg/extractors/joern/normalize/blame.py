"""History for ``lastChangedAt``/``lastChangedCommit`` (bundle spec3 §5.3, BF12): ``git blame --porcelain`` of the source
repository at the built commit.

The normalizer runs on the host, outside the sandbox, and a staged tree has no ``.git``: history is read from the
**source repository**, never from the stage. Ported from ``cpg/spikes/vertical-slice/normalize.py`` (``blame``,
``last_changed``) with the spec4 §2 fallback factored out: a symbol whose span plus ``ownedRegions`` holds no blamed line
(an empty file's ``<module>``) takes the committer date and id of the last commit up to the built commit that touched the
path (``git log -1``); ``header.history`` stays ``available``.
"""
from __future__ import annotations

import re
import subprocess  # noqa: F401  (tests patch ``blame.subprocess.run``: the one ``subprocess`` module gitsafe runs through)
from datetime import datetime, timezone
from pathlib import Path

from ccg import gitsafe

COMMIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")           # a SHA-1 or a SHA-256 object id, lowercase
_HDR = re.compile(r"^(" + COMMIT_ID.pattern + r") \d+ (\d+)(?: \d+)?$")
GIT_TIMEOUT = 300                                                   # seconds; a hostile or huge history must not hang a build


class HistoryError(RuntimeError):
    """The source repository cannot give the history a bundle needs (a shallow clone, spec3 §2.2, index D1; a path with no
    history at the built commit; a git call that timed out)."""


def is_commit_id(commit) -> bool:
    return isinstance(commit, str) and COMMIT_ID.fullmatch(commit) is not None


_env = gitsafe.environment      # kept for callers and tests that read the git environment through this module


def git(repo, *args: str) -> bytes:
    """Stdout of ``git`` run in ``repo`` through ``ccg.gitsafe`` (hardened environment and configuration, literal pathspecs,
    a timeout, an output cap; the source repository is not trusted, spec3 §8.5). ``CalledProcessError`` on a git failure,
    ``HistoryError`` on a timeout. (The repository's own ``blame.ignoreRevsFile`` is cleared by ``blame`` with
    ``--ignore-revs-file=``: ``-c blame.ignoreRevsFile=`` does not reset it.)"""
    try:
        return gitsafe.git(repo, *args, timeout=GIT_TIMEOUT)
    except gitsafe.GitTimeout as e:
        raise HistoryError(f"git {args[0]} in {repo} timed out after {GIT_TIMEOUT} s") from e


def _check(commit: str) -> None:
    if not is_commit_id(commit):
        raise HistoryError(f"{commit!r} is not a commit id (40 or 64 lowercase hex digits)")


def require_full_history(repo: str | Path) -> None:
    """Refuse a shallow clone: ``header.history`` may only be ``available`` and a shallow blame is silently wrong."""
    if git(repo, "rev-parse", "--is-shallow-repository").decode().strip() == "true":
        raise HistoryError(f"{repo} is a shallow clone: lastChanged* would be blamed to the shallow boundary; "
                           "fetch the full history (index D1)")


def parse_porcelain(text: str):
    """``({line: commit}, {commit: committer epoch})`` of ``git blame --porcelain`` output. A commit's later lines come as
    abbreviated blocks (the header and the tab line only)."""
    lines = text.split("\n")
    times: dict[str, int] = {}
    per_line: dict[int, str] = {}
    i, cur = 0, None
    while i < len(lines):
        m = _HDR.match(lines[i])
        if m:
            cur = m.group(1)
            per_line[int(m.group(2))] = cur
            i += 1
            while i < len(lines) and not lines[i].startswith("\t"):
                if lines[i].startswith("committer-time "):
                    times[cur] = int(lines[i].split()[1])
                i += 1
        i += 1
    return per_line, times


def blame(repo: Path, commit: str, repo_path: str):
    """Per-line ``({line: commit}, {commit: committer epoch})`` for a file at ``commit``."""
    _check(commit)
    return parse_porcelain(git(repo, "blame", "--porcelain", "--no-textconv", "--ignore-revs-file=", commit, "--", repo_path).decode("utf-8", "surrogateescape"))


def _stamp(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def last_changed(per_line, times, regions):
    """The latest committer time over the blamed lines of ``regions`` (inclusive ``(first, last)`` pairs: a symbol's span
    plus its ``ownedRegions``); a tie takes the lower commit id. ``ValueError`` when no line of them is blamed."""
    best = None
    top = max(per_line, default=0)
    for a, b in regions:
        for ln in range(max(a, 1), min(b, top) + 1):               # bounded by the blamed lines, whatever a region says
            c = per_line.get(ln)
            if c is None:
                continue
            key = (-times[c], c)   # latest time; ties -> the lower commit id
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError("no blame for span")
    return _stamp(-best[0]), best[1]


def path_last_commit(repo: Path, commit: str, repo_path: str):
    """The committer date and id of the last commit up to ``commit`` that touched ``repo_path`` (spec4 §2 fallback)."""
    _check(commit)
    out = git(repo, "log", "-1", "--no-show-signature", "--format=%ct %H", commit, "--", repo_path).decode("ascii", "replace").split()
    if len(out) != 2:
        raise HistoryError(f"{repo_path!r} has no history at {commit[:12]}: no commit up to it touched that path")
    return _stamp(int(out[0])), out[1]
