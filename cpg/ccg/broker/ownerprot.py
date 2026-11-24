"""Owner protection of an operator-installed file (``docs/plans/connected-code-graph-egress-spec5.md`` section 1).

A file passes only when it is opened ``O_NOFOLLOW``, ``fstat`` on that descriptor shows a regular file, its owner uid is
in the injected ``trusted_uids`` (production ``{0}``), ``mode & 0o022 == 0``, and every ancestor, walked by descriptor
(``dir_fd``), is a non-symlink directory owned by a trusted uid and not group/other-writable. The bytes come from that
same descriptor, so a path swapped after the check changes nothing the caller parses. Nothing is cached: the broker
checks once, at start.

The broker never imports ``llmgateway``, so this is its own implementation; ``cpg/tests/test_broker_ownerprot.py`` holds
it to parity with ``llmgateway.ownercheck.read_owner_file`` on the same planted files.
"""
from __future__ import annotations

import os
import stat
from collections.abc import Collection
from dataclasses import dataclass

from ccg.broker.audit import PreconditionError

MAX_BYTES = 1 << 20

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


@dataclass(frozen=True)
class OwnerRead:
    """Outcome of one check-and-read; ``data`` is empty unless ``ok``."""

    ok: bool
    reason: str = ""
    data: bytes = b""


def _refuse(reason: str) -> OwnerRead:
    return OwnerRead(False, reason)


def _components(path: str) -> "list[str] | None":
    if "\0" in path or not path.startswith("/"):
        return None
    parts = [p for p in path.split("/") if p not in ("", ".")]
    return None if ".." in parts else parts


def _problem(st: os.stat_result, trusted: Collection[int], label: str) -> "str | None":
    if st.st_uid not in trusted:
        return f"{label} owned by untrusted uid {st.st_uid}"
    if st.st_mode & 0o022:
        return f"{label} is group- or other-writable"
    return None


def read_owner_file(path: "str | os.PathLike[str]", trusted_uids: Collection[int], *,
                    anchor: "str | os.PathLike[str] | None" = None, max_bytes: int = MAX_BYTES) -> OwnerRead:
    """Check ``path`` and, when protected, return its bytes. ``anchor`` is a test seam: the ancestor walk starts at that
    directory (itself opened ``O_NOFOLLOW`` and checked) instead of ``/``; production passes none."""
    parts = _components(os.fspath(path))
    if parts is None or not parts:
        return _refuse("path must be absolute, without '..'")
    base: list[str] = []
    if anchor is not None:
        anchor_parts = _components(os.fspath(anchor))
        if anchor_parts is None or parts[: len(anchor_parts)] != anchor_parts:
            return _refuse("path is not under the anchor")
        base = anchor_parts
    rest = parts[len(base):]
    if not rest:
        return _refuse("path names a directory")
    fds: list[int] = []
    try:
        try:
            fd = os.open("/" + "/".join(base), _DIR_FLAGS)
        except OSError as exc:
            return _refuse(f"cannot open ancestor: {exc.strerror}")
        fds.append(fd)
        failure = _problem(os.fstat(fd), trusted_uids, "ancestor /" + "/".join(base)) or ""
        for name in rest[:-1]:
            try:
                fd = os.open(name, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                return _refuse(f"ancestor {name!r} is a symlink or not a directory ({exc.strerror})")
            fds.append(fd)
            failure = failure or _problem(os.fstat(fd), trusted_uids, f"ancestor {name!r}") or ""
        try:
            ffd = os.open(rest[-1], _FILE_FLAGS, dir_fd=fd)
        except FileNotFoundError:
            return _refuse("file does not exist")
        except OSError as exc:
            return _refuse(f"cannot open file (symlink?): {exc.strerror}")
        fds.append(ffd)
        st = os.fstat(ffd)
        if not stat.S_ISREG(st.st_mode):
            return _refuse("not a regular file")
        failure = failure or _problem(st, trusted_uids, "file") or ""
        if st.st_size > max_bytes:
            return _refuse("file too large")
        data = b""
        while len(data) <= max_bytes:
            chunk = os.read(ffd, 65536)
            if not chunk:
                break
            data += chunk
        if len(data) > max_bytes:
            return _refuse("file too large")
        return OwnerRead(False, failure) if failure else OwnerRead(True, data=data)
    finally:
        for f in fds:
            os.close(f)


def require_owner_file(path: "str | os.PathLike[str]", trusted_uids: Collection[int], **kw) -> bytes:
    """The verified bytes of ``path``, or `PreconditionError` (exit 69) naming the file and the failed condition."""
    res = read_owner_file(path, trusted_uids, **kw)
    if not res.ok:
        raise PreconditionError(f"{os.fspath(path)}: not owner-protected: {res.reason}")
    return res.data
