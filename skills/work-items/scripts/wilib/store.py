"""Journal + tracked registry, flock, seq-safe timestamps, reseed, flush (plan 3.1).

Two copies of one append-only event log, the journal authoritative:

* journal  ``<git-common-dir>/work-items/<KEY>/journal.jsonl`` -- shared by every
  branch and worktree of the clone; every allocation appends here under one flock;
* tracked  ``<root>/.work-items/registry.jsonl`` -- written only by ``flush`` (and
  directly, under the same lock, when the project is not a git repo).

State is the fold of the union of both, deduplicated by ``evid``.  Reads touch only
these two files (plus ``git`` for the reseed): never a family document (AC-5).
"""
import fcntl
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import events

REGISTRY_REL = Path(".work-items") / "registry.jsonl"


def _git(root, *args):
    """Run git in ``root``; stdout on success, None on any failure."""
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout if r.returncode == 0 else None


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _append(path, evs):
    """Append events, first adding a missing trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab+") as fh:
        size = fh.seek(0, os.SEEK_END)
        if size:
            fh.seek(size - 1)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write("".join(events.dump_event(e) for e in evs).encode("utf-8"))


def _dedup(evs):
    seen, out = set(), []
    for e in evs:
        if e["evid"] not in seen:
            seen.add(e["evid"])
            out.append(e)
    return out


class Transaction:
    """Handle valid only inside ``Store.transaction()``: state + stamped append."""

    def __init__(self, store, evs):
        self._store, self._events, self._state = store, evs, None

    @property
    def state(self):
        if self._state is None:
            self._state = events.fold(self._events)
        return self._state

    def append(self, ev):
        ev = {"evid": uuid.uuid4().hex[:16],
              "ts": events.stamp_ts(self.state.last_ts, self._store._clock()), **ev}
        events.validate_event(ev)
        _append(self._store._write_path, [ev])
        self._events.append(ev)
        self._state = None
        return ev


class Store:
    def __init__(self, root, key, clock=time.time_ns):
        self.root, self.key, self._clock = Path(root), key, clock
        self.registry_path = self.root / REGISTRY_REL
        common = _git(self.root, "rev-parse", "--git-common-dir")
        if common is None:
            self.journal_path = None
            self._lock_path = self.root / ".work-items" / ".lock"
        else:
            jdir = (self.root / common.strip()).resolve() / "work-items" / key
            self.journal_path = jdir / "journal.jsonl"
            self._lock_path = jdir / "lock"

    @property
    def _write_path(self):
        return self.journal_path or self.registry_path

    # -------------------------------------------------------------- locking
    @contextmanager
    def _locked(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)        # closing releases the flock

    # -------------------------------------------------------------- reading
    def _ref_events(self):
        refs = _git(self.root, "for-each-ref", "--format=%(refname)",
                    "refs/heads", "refs/remotes") or ""
        out = []
        for ref in refs.split():
            text = _git(self.root, "show", f"{ref}:./{REGISTRY_REL.as_posix()}")
            if text:
                try:
                    out += events.read_events(text)
                except events.EventError:
                    continue        # an unreadable ref copy must not block recovery
        return out

    def _reconcile(self):
        """Return the deduplicated union, first copying into the journal every
        tracked event it lacks (and, on a missing/empty journal, every ref's)."""
        tracked = events.read_events(_read(self.registry_path))
        if self.journal_path is None:
            return _dedup(tracked)
        journal = events.read_events(_read(self.journal_path))
        incoming = list(tracked)
        if not journal:
            incoming += self._ref_events()
        have = {e["evid"] for e in journal}
        missing = _dedup([e for e in incoming if e["evid"] not in have])
        if missing:
            _append(self.journal_path, missing)
            journal += missing
        return _dedup(journal + tracked)

    @contextmanager
    def transaction(self):
        with self._locked():
            yield Transaction(self, self._reconcile())

    def read_state(self):
        with self._locked():
            return events.fold(self._reconcile())

    # -------------------------------------------------------------- flush
    def flush(self):
        """Append missing journal events to the tracked file, then drop duplicate
        ``evid`` lines (keep first).  Idempotent; a no-op rewrite is skipped."""
        with self._locked():
            union = self._reconcile()
            tracked = _dedup(events.read_events(_read(self.registry_path)))
            have = {e["evid"] for e in tracked}
            want = tracked + [e for e in union if e["evid"] not in have]
            text = "".join(events.dump_event(e) for e in want)
            if text == _read(self.registry_path):
                return
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.registry_path.with_name(self.registry_path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.registry_path)
