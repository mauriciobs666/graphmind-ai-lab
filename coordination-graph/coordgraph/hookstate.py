"""The hook's on-disk state (plan section 3.2): `$COORDGRAPH_STATE_DIR` and the files in it.

`failed.json` is `{repo-relative path: {sha256, error, attempts, nextRetryAt}}`; `failed.pending` holds one
path per line, appended by the shell with no Python (under `flock state.lock`) and merged, then truncated, by
Python under the same lock, so no append can land between the read and the truncate; `retry-due` is the epoch
second of the earliest due retry (absent when nothing is waiting), testable by the shell without Python;
`stamp` is the mtime marker the shell's `find -newer` reads. This module imports nothing outside the standard
library: it is on the sync path (never `wilib`, never `identity`).
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

BACKOFF_MINUTES = (1, 2, 4, 8, 15)  # the 5th and every later attempt wait the 15-minute cap


def backoff_s(attempts: int) -> int:
    """Seconds to wait after the `attempts`-th failure (1-based)."""
    return 60 * BACKOFF_MINUTES[min(max(attempts, 1), len(BACKOFF_MINUTES)) - 1]


def relative(root: Path, raw: str) -> str | None:
    """`raw` as a repo-relative `/` path, or None when it points outside `root`."""
    p = Path(raw)
    if not p.is_absolute():
        return p.as_posix().removeprefix("./")
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


class HookState:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.stamp = state_dir / "stamp"
        self.failed_json = state_dir / "failed.json"
        self.pending = state_dir / "failed.pending"
        self.retry_due = state_dir / "retry-due"
        self.state_lock = state_dir / "state.lock"
        self.retry_lock = state_dir / "retry.lock"

    def ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """`state.lock`, exclusive: guards `failed.json` read-modify-write and the pending merge (milliseconds)."""
        self.ensure()
        with open(self.state_lock, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            yield

    @contextmanager
    def retry_singleton(self) -> Iterator[bool]:
        """`retry.lock` without waiting: yields False when another `retry-failed` holds it."""
        self.ensure()
        with open(self.retry_lock, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield True

    # -- under `locked()` ------------------------------------------------------------------------
    def load_failed(self) -> dict[str, dict]:
        try:
            data = json.loads(self.failed_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {p: e for p, e in data.items() if isinstance(e, dict)} if isinstance(data, dict) else {}

    def take_pending(self) -> list[str]:
        """The paths the shell appended; the file is truncated in the same locked section."""
        try:
            lines = self.pending.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        self.pending.write_text("", encoding="utf-8")
        return sorted({ln.strip() for ln in lines if ln.strip()})

    def save_failed(self, failed: dict[str, dict]) -> None:
        tmp = self.failed_json.with_name("failed.json.tmp")
        tmp.write_text(json.dumps(failed, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.failed_json)
        due = [e["nextRetryAt"] for e in failed.values() if "nextRetryAt" in e]
        if due:
            self.write_retry_due(math.ceil(min(due)))
        else:
            self.retry_due.unlink(missing_ok=True)

    # -- anywhere ----------------------------------------------------------------------------------
    def write_retry_due(self, epoch_s: int) -> None:
        self.ensure()
        self.retry_due.write_text(f"{int(epoch_s)}\n", encoding="utf-8")

    def advance_stamp(self, started: float) -> None:
        """`max(stamp, started)`, never "now": a write landing during the sync keeps a newer mtime."""
        self.ensure()
        if not self.stamp.exists() or self.stamp.stat().st_mtime < started:
            self.stamp.touch()
            os.utime(self.stamp, (started, started))
