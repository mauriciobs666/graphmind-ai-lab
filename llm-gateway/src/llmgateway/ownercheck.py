"""Owner-protection check for the gateway configuration (docs/plans/llm-gateway.md §3.5).

A file is *owner-protected* when it is opened `O_NOFOLLOW`, inspected with `fstat` on the
descriptor and read from that same descriptor, its owner uid is in the injected `trusted_uids`
(production `{0}`), `mode & 0o022 == 0`, and every ancestor directory is trusted-owned, not
group/other-writable and not a symlink. The whole walk goes by descriptor (`dir_fd`), so a path
swapped after the check cannot change what was checked. Nothing is cached here: callers re-run
the check on every call.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Collection
from dataclasses import dataclass

MAX_BYTES = 1 << 20

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


@dataclass(frozen=True, slots=True)
class OwnerRead:
    """Outcome of one check-and-read. `data` is empty and `key` is None unless `ok`."""

    ok: bool
    reason: str = ""
    data: bytes = b""
    # Bytes of a regular file reached by a symlink-free chain that failed only the uid/mode
    # checks. NEVER trust it: it exists so the loader can read a failing file's non-owner parts
    # (§3.5: the unprotected file's other sections still apply). Empty when `ok`.
    untrusted_data: bytes = b""
    # (st_dev, st_ino, st_mtime_ns, st_size) of the descriptor that was read (§3.5 parse-cache key).
    key: tuple[int, int, int, int] | None = None


def _refuse(reason: str) -> OwnerRead:
    return OwnerRead(ok=False, reason=reason)


def _components(path: str) -> list[str] | None:
    if "\0" in path or not path.startswith("/"):
        return None
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if ".." in parts:
        return None
    return parts


def _dir_problem(st: os.stat_result, trusted_uids: Collection[int], label: str) -> str | None:
    if st.st_uid not in trusted_uids:
        return f"{label} owned by untrusted uid {st.st_uid}"
    if st.st_mode & 0o022:
        return f"{label} is group- or other-writable"
    return None


def read_owner_file(
    path: str | os.PathLike[str],
    trusted_uids: Collection[int],
    *,
    anchor: str | os.PathLike[str] | None = None,
    max_bytes: int = MAX_BYTES,
) -> OwnerRead:
    """Check `path` and, if protected, return its bytes.

    A uid/mode failure is reported only after the file was read through descriptors, so the
    unprotected remainder is available as `untrusted_data`; a symlink, a missing or non-regular
    file yields nothing.

    `anchor` is a test seam: when given, the ancestor walk starts at that directory (which is
    itself opened `O_NOFOLLOW` and checked) instead of `/`, so a test can build a trusted chain
    under a temp dir whose own ancestors (`/tmp`, ...) are not trusted. Production passes none.
    """
    parts = _components(os.fspath(path))
    if parts is None or not parts:
        return _refuse("path must be absolute, without '..'")
    base_parts: list[str] = []
    if anchor is not None:
        anchor_parts = _components(os.fspath(anchor))
        if anchor_parts is None or parts[: len(anchor_parts)] != anchor_parts:
            return _refuse("path is not under the anchor")
        base_parts = anchor_parts
    rest = parts[len(base_parts) :]
    if not rest:
        return _refuse("path names a directory")

    fds: list[int] = []
    failure = ""
    try:
        try:
            fd = os.open("/" + "/".join(base_parts), _DIR_FLAGS)
        except OSError as exc:
            return _refuse(f"cannot open ancestor: {exc.strerror}")
        fds.append(fd)
        failure = (
            failure
            or _dir_problem(os.fstat(fd), trusted_uids, "ancestor /" + "/".join(base_parts))
            or ""
        )
        for name in rest[:-1]:
            try:
                fd = os.open(name, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                return _refuse(
                    f"ancestor {name!r} is a symlink or not a directory ({exc.strerror})"
                )
            fds.append(fd)
            problem = _dir_problem(os.fstat(fd), trusted_uids, f"ancestor {name!r}")
            failure = failure or problem or ""
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
        failure = failure or _dir_problem(st, trusted_uids, "file") or ""
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
        if failure:
            return OwnerRead(ok=False, reason=failure, untrusted_data=data)
        return OwnerRead(
            ok=True,
            data=data,
            key=(st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size),
        )
    finally:
        for f in fds:
            os.close(f)
