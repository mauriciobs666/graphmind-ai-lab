"""The broker's audit writer (C-1001 P1-29). Spec: ``docs/plans/connected-code-graph-egress-spec3.md`` section 5.1-5.2.

One writer per file (I6): only the broker writes ``<auditDir>/egress.jsonl``, and inside the broker every record goes
through one `AuditWriter`. Records are append-only JSONL, one ``write(2)`` of the whole line under an exclusive
``flock``, followed by ``fsync``; each is validated against ``egress-audit-v1.schema.json`` before it is written. A
record that cannot be validated, written or made durable latches the writer: every later call raises
`AuditUnwritable` until the process restarts (section 2.5).
"""
from __future__ import annotations

import fcntl
import functools
import json
import os
import stat
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import jsonschema

SCHEMA = Path(__file__).resolve().parent.parent / "schemas" / "egress-audit-v1.schema.json"
# `<directory of the ccg package>/../.audit` (section 5.1): the code's own location, never the working directory.
DEFAULT_AUDIT_DIR = Path(__file__).resolve().parent.parent.parent / ".audit"
AUDIT_FILE = "egress.jsonl"


class AuditUnwritable(Exception):
    """A record could not be validated, written or made durable; the writer is latched."""


class PreconditionError(Exception):
    """A start precondition failed (lock held, live socket, directory or file check): the broker exits 69."""


@functools.lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def validate_record(record: Any) -> None:
    """Raise `jsonschema.ValidationError` unless ``record`` is a valid broker record."""
    errors = sorted(_validator().iter_errors(record), key=lambda e: list(map(str, e.absolute_path)))
    if errors:
        raise jsonschema.ValidationError(errors[0].message)


def now_ts(now: Callable[[], datetime] | None = None) -> str:
    t = (now or (lambda: datetime.now(timezone.utc)))()
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


# ------------------------------------------------------------------ directory checks


def _ancestors_ok(real: Path, uid: int, *, allow_sticky: bool = False,
                  stat_fn: Callable[[Path], os.stat_result] = os.stat) -> str | None:
    """None, or why an ancestor of ``real`` is group- or world-writable (section 2.0: no exception). With
    ``allow_sticky`` (tests only; the broker process never passes it) a sticky directory owned by root or the user,
    such as ``/tmp``, is accepted: nobody else can rename or remove our entry in it."""
    for anc in real.parents:
        st = stat_fn(anc)
        if st.st_mode & 0o022 and not (allow_sticky and st.st_mode & stat.S_ISVTX and st.st_uid in (0, uid)):
            return f"ancestor {anc} is group- or world-writable ({stat.S_IMODE(st.st_mode):04o})"
    return None


def require_dir(path: Path, *, exact_mode: int | None, create: bool, check_ancestors: bool,
                allow_sticky: bool = False, stat_fn: Callable[[Path], os.stat_result] = os.stat) -> Path:
    """A plain directory (not a symlink) owned by this uid; ``exact_mode`` requires that mode, otherwise only that it
    is not group- or world-writable. Created ``0700`` if absent and ``create``. `PreconditionError` otherwise."""
    uid = os.getuid()
    if create and not os.path.lexists(path):
        os.makedirs(path.parent, mode=0o700, exist_ok=True)
        os.mkdir(path, 0o700)
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise PreconditionError(f"{path}: {exc.strerror}") from None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise PreconditionError(f"{path} is not a plain directory (a symlink or not a directory)")
    if st.st_uid != uid:
        raise PreconditionError(f"{path} is owned by uid {st.st_uid}, not {uid}")
    mode = stat.S_IMODE(st.st_mode)
    if exact_mode is not None and mode != exact_mode:
        raise PreconditionError(f"{path} has mode {mode:04o}, expected {exact_mode:04o}")
    if exact_mode is None and mode & 0o022:
        raise PreconditionError(f"{path} is group- or world-writable ({mode:04o})")
    if check_ancestors:
        why = _ancestors_ok(Path(os.path.realpath(path)), uid, allow_sticky=allow_sticky, stat_fn=stat_fn)
        if why:
            raise PreconditionError(f"{path}: {why}")
    return path


# ------------------------------------------------------------------ the writer


class AuditWriter:
    def __init__(self, audit_dir: Path | str, *, now: Callable[[], datetime] | None = None,
                 log: Callable[[str], None] | None = None, write: Callable[[int, bytes], int] = os.write,
                 fsync: Callable[[int], None] = os.fsync):
        self.audit_dir = Path(os.path.abspath(audit_dir))
        self._now, self._write, self._fsync = now, write, fsync
        self._log = log or (lambda m: print(f"ccg-broker: {m}", file=sys.stderr, flush=True))
        self._lock = threading.Lock()
        self.latched = False
        require_dir(self.audit_dir, exact_mode=None, create=True, check_ancestors=False)
        self.path = self.audit_dir / AUDIT_FILE
        existed = os.path.lexists(self.path)
        # O_NONBLOCK: a FIFO planted at the path fails the open (ENXIO) instead of hanging the start.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        try:
            self._fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise PreconditionError(f"{self.path}: cannot open ({exc.strerror})") from None
        try:
            st = os.fstat(self._fd)
            if not stat.S_ISREG(st.st_mode):
                raise PreconditionError(f"{self.path} is not a regular file")
            if st.st_uid != os.getuid():
                raise PreconditionError(f"{self.path} is owned by uid {st.st_uid}")
            if st.st_mode & 0o077:
                raise PreconditionError(f"{self.path} has group/other permission bits ({stat.S_IMODE(st.st_mode):04o})")
            if st.st_nlink != 1:
                raise PreconditionError(f"{self.path} has {st.st_nlink} hard links")
            if not existed:
                dfd = os.open(self.audit_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            if st.st_size and self._last_byte() != b"\n":
                self._append_raw(b"\n")
                self._log("audit file did not end in a newline; appended one")
        except BaseException:
            os.close(self._fd)
            raise

    def _last_byte(self) -> bytes:
        rfd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            size = os.fstat(rfd).st_size
            return os.pread(rfd, 1, size - 1)
        finally:
            os.close(rfd)

    def _append_raw(self, data: bytes) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            if self._write(self._fd, data) != len(data):
                raise OSError("short write")
            self._fsync(self._fd)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def record(self, record_type: str, **fields: Any) -> dict:
        """A record with the common fields filled in (not yet validated or written)."""
        return {"schemaVersion": 1, "recordType": record_type, "auditId": str(uuid.uuid4()),
                "ts": now_ts(self._now), **fields}

    def append(self, record: dict) -> str:
        """Validate, write as one line and fsync; returns the ``auditId``. Latches on any failure."""
        with self._lock:
            if self.latched:
                raise AuditUnwritable("the audit writer is latched")
            try:
                validate_record(record)
                # ensure_ascii: still valid UTF-8 JSON, and U+2028/U+0085 never reach a line-splitting reader raw
                line = json.dumps(record, allow_nan=False, ensure_ascii=True, separators=(",", ":")) + "\n"
                self._append_raw(line.encode("ascii"))
            except Exception as exc:  # validation failure is a broker defect; an I/O failure is the disk's
                self.latched = True
                why = f"{type(exc).__name__}: {str(exc)[:160]}"  # a validation message quotes the offending value
                self._log(f"audit write failed, latching closed: {why}")
                raise AuditUnwritable(why) from None
            return record["auditId"]

    def latch(self) -> None:
        with self._lock:
            self.latched = True

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1


def read_records(path: Path | str) -> Iterator[dict]:
    """Every record of an ``egress.jsonl``, each validated; a trailing partial line (no newline) is skipped."""
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    complete = lines[:-1]  # the last element is the (possibly empty) text after the final newline
    for n, raw in enumerate(complete, 1):
        if not raw:
            continue  # the blank line a newline repair leaves
        rec = json.loads(raw)
        try:
            validate_record(rec)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"{path}:{n}: invalid record: {exc.message}") from None
        yield rec
