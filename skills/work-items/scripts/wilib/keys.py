"""Shared project keys (plan 3.9, FR-17/FR-18): ``keys.json`` in the team repo.

``suggest`` is read-only.  ``claim`` is the only writer and touches nothing but the key
list; the stakeholder's agreement that must precede it is a prompt-side rule.  ``init``
neither claims nor requires a claim (a project not feeding the pool may skip listing).
"""
import fcntl
import json
import os
import re
import time
from pathlib import Path

from . import rules

RESERVED = frozenset({"FR", "AC", "TP", "R"})
ENV_PATH = "WI_KEYS_FILE"                   # test / override hook


class KeyRefused(RuntimeError):
    """The key is malformed, reserved, or already claimed."""


def keys_path():
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override)
    # wilib/keys.py -> wilib -> scripts -> <skill dir>; resolve() follows the skill symlink
    return Path(__file__).resolve().parents[2] / "keys.json"


def _load(text, path):
    try:
        entries = json.loads(text) if text.strip() else []
    except json.JSONDecodeError as exc:
        raise KeyRefused(f"{path} is not valid JSON ({exc}); fix the key list") from None
    if not isinstance(entries, list):
        raise KeyRefused(f"{path} must hold a JSON list of key entries")
    return entries


def check_key(key, entries):
    """Raise KeyRefused unless ``key`` may be claimed given the current ``entries``."""
    if key in RESERVED:
        raise KeyRefused(f"{key} is reserved (FR/AC/TP/R)")
    if not rules.PROJECT_RE.fullmatch(key):
        raise KeyRefused(f"malformed key {key!r}: must match [A-Z]{{2,6}}")
    if any(e.get("key") == key for e in entries):
        raise KeyRefused(f"{key} is already claimed")


def _candidates(name):
    words = re.findall(r"[A-Za-z]+", name)
    letters = "".join(words).upper()
    initials = "".join(w[0] for w in words).upper()
    seen = []

    def add(c):
        if c not in seen:
            seen.append(c)
    add(initials[:6])
    for n in (3, 4, 5, 6, 2):
        add(letters[:n])
    for head in (initials[:2], letters[:2]):
        for ch in letters:
            add(head + ch)
    return [c for c in seen if rules.PROJECT_RE.fullmatch(c)]


def suggest(name, path=None):
    """Propose the first acceptable key derived from ``name``; never writes."""
    path = Path(path) if path else keys_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise KeyRefused(f"key list {path} not found: the skill ships keys.json as []") from None
    entries = _load(text, path)
    for cand in _candidates(name):
        try:
            check_key(cand, entries)
        except KeyRefused:
            continue
        return cand
    raise KeyRefused(f"no acceptable key can be derived from {name!r}")


def _after_read():
    """Seam between the read-check and the write (tests widen the race window here)."""


def claim(key, repo, path=None, feeds_pool=True):
    """Append the key under an exclusive flock on the list itself; refuses a duplicate."""
    path = Path(path) if path else keys_path()
    try:
        fd = os.open(path, os.O_RDWR)   # the list must exist: the skill ships it as []
    except FileNotFoundError:
        raise KeyRefused(f"key list {path} not found: the skill ships keys.json as []") from None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        entries = _load(os.pread(fd, 1 << 24, 0).decode("utf-8"), path)
        check_key(key, entries)
        _after_read()
        entries.append({"key": key, "repo": repo, "claimed": time.strftime("%Y-%m-%d"),
                        "feeds_pool": bool(feeds_pool)})
        os.ftruncate(fd, 0)
        os.pwrite(fd, (json.dumps(entries, indent=2) + "\n").encode("utf-8"), 0)
        os.fsync(fd)
    finally:
        os.close(fd)                    # releases the flock
    return path
