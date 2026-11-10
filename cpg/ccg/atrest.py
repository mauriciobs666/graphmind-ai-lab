"""Bundle-at-rest and key custody (C-1001 P1-02, F-1).

The decision, its locations, modes, retention rule and reasons are in `cpg/infra/README.md`, section
"Bundle-at-rest"; this module enforces it and does not restate it.

Every accessor is pinned to exactly one area: bundle functions to the work area (``root=``, default
``WORK_ROOT``), the HMAC key to the key area (``keys_root=``, default ``KEY_ROOT``), ACL credentials to
``credentials_dir()``. The generic ``ensure_private`` / ``read_private`` / ``write_private`` take the area
explicitly; there is no "either area" default. Inside an area:

- the path must lie strictly below the area root, with no lexical ``..`` component;
- no component below the area root may be a symlink (checked with ``lstat`` on the unresolved path), so an
  in-area link, a link out, and a dangling link are all refused;
- every existing component must be owned by the current user and carry no group/other bits; an exposed file
  raises and is never re-moded;
- reads walk the chain by fd (each directory ``O_DIRECTORY|O_NOFOLLOW`` relative to its parent, the leaf
  ``O_NOFOLLOW``) and re-check owner, type and mode with ``fstat``, so a file or parent directory swapped for a
  link after the name check is never followed (writes check the chain, then create);
- created directories are 0700 and files 0600, independent of the umask, and writes are atomic.

A key area (or ``$CCG_CREDENTIALS_DIR``) inside the repository is refused. ``hmac_key`` is the single place that
refuses the published ``-bundle-spec5.md`` §9 test keys, on generation and on every load, with no bypass.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Iterable

WORK_ROOT = Path(__file__).resolve().parent.parent / ".ccg-work"
REPO_ROOT = WORK_ROOT.parent.parent
KEY_ROOT = Path.home() / ".config" / "ccg" / "keys"
BUNDLE_SUFFIX = ".bundle.jsonl"
KEY_BYTES = 32
BIND_KEYCHECK_TAG = "ccg.bindParts.keycheck.v1"
# bindKeyCheck of the two published -bundle-spec5.md §9 test keys: EH-30 (bytes 00..1f) and §9.4's K2
# (bytes 20..3f). A key with either check is public knowledge, so every hash under it is open to a dictionary
# attack; it never loads (security advisory S5-7). Tests read the expected values from the spec text.
PUBLISHED_TEST_KEYCHECKS = frozenset({
    "aca11439350046b0f12310cffc38748816b211bcbe3cb4bfc1b783d8f98f090a",
    "4d6dd0974c662bfb2eafcf965b6a69b09f6689dc8cd85e85368fc44ed02414ac",
})

# A path guard only: the authoritative system-name whitelist is `docs/plans/connected-code-graph-p1-graph.md` §7
# (stricter; every name it admits passes here). This only guarantees a name is one safe path component.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class AtRestError(PermissionError):
    """A path is outside its area, a symlink, exposed, foreign-owned or malformed."""


# ------------------------------------------------------------------ areas

def _key_root(keys_root: str | os.PathLike | None) -> Path:
    k = Path(keys_root) if keys_root is not None else KEY_ROOT
    rk, rr = Path(os.path.realpath(k)), Path(os.path.realpath(REPO_ROOT))
    if rk == rr or rr in rk.parents:
        raise AtRestError(f"key area {k} is inside the repository {rr}; keys live outside it (-bundle-spec3.md §5.17)")
    return k


def _work_root(root: str | os.PathLike | None) -> Path:
    return Path(root) if root is not None else WORK_ROOT


def credentials_dir(keys_root: str | os.PathLike | None = None) -> Path:
    """``$CCG_CREDENTIALS_DIR`` if set, else the key area; refused if inside the repository."""
    env = os.environ.get("CCG_CREDENTIALS_DIR")
    return _key_root(env) if env else _key_root(keys_root)


def _name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise AtRestError(f"{kind} name {value!r} is not one safe path component ([A-Za-z0-9][A-Za-z0-9_-]{{0,63}})")
    return value


# ------------------------------------------------------------------ checks

def _rel_parts(path: str | os.PathLike, area: Path) -> tuple[Path, tuple[str, ...]]:
    """(resolved area root, the path's components below it), lexically: no ``..``, strictly below the root."""
    raw = os.fspath(path)
    if "\0" in raw:
        raise AtRestError("a path with a NUL byte")
    if ".." in Path(raw).parts:
        raise AtRestError(f"{raw}: a lexical '..' component is refused")
    lex = Path(os.path.abspath(raw))
    a_lex = Path(os.path.abspath(area))
    a_real = Path(os.path.realpath(area))
    for base in (a_lex, a_real):
        try:
            rel = lex.relative_to(base)
        except ValueError:
            continue
        if rel.parts:
            return a_real, rel.parts
    raise AtRestError(f"{raw} is outside its area {a_lex}")


def _check_stat(name: Path, st: os.stat_result) -> None:
    uid = os.getuid()
    if st.st_uid != uid:
        raise AtRestError(f"{name} is owned by uid {st.st_uid}, not {uid}")
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise AtRestError(f"{name} has group/other permission bits ({stat.S_IMODE(st.st_mode):04o}); "
                          "it may have been exposed, so it is refused rather than fixed")


def _check_chain(a_real: Path, parts: tuple[str, ...], check_root: bool) -> Path:
    if check_root:
        _check_stat(a_real, os.lstat(a_real))
    cur = a_real
    for part in parts:
        cur = cur / part
        if not os.path.lexists(cur):
            continue
        st = os.lstat(cur)
        if stat.S_ISLNK(st.st_mode):
            raise AtRestError(f"{cur} is a symlink; no link is followed inside a protected area")
        _check_stat(cur, st)
    return cur


def ensure_private(path: str | os.PathLike, area: str | os.PathLike) -> Path:
    """Refuse ``path`` unless it lies strictly below ``area`` with no ``..`` and no symlink on the way, and every
    existing component is the user's own with no group/other bits. Returns the path below the resolved area.
    Never changes a mode."""
    area = Path(area)
    a_real, parts = _rel_parts(path, area)
    return _check_chain(a_real, parts, check_root=not _is_work(area))


def _is_work(area: Path) -> bool:
    """A work area is a directory named ``.ccg-work`` (its root may be 0755, as the real one is); any other area
    is a key/credentials area, whose root must itself be private."""
    return Path(area).name == WORK_ROOT.name


def _mkdir_private(d: Path) -> None:
    try:
        os.mkdir(d, 0o700)
        os.chmod(d, 0o700)                          # umask-independent
    except FileExistsError:
        pass


def _private_dir(d: Path, area: Path, is_key: bool) -> Path:
    """Create directory ``d`` (and its parents up to the area root) at 0700, refusing any symlink, then check it.
    A key area's root is created 0700 if absent (its parents, e.g. ~/.config/ccg, at 0700 too)."""
    if is_key and not os.path.lexists(area):
        os.makedirs(area.parent, mode=0o700, exist_ok=True)
        _mkdir_private(area)
    a_real = Path(os.path.realpath(area))
    if Path(os.path.abspath(d)) in (Path(os.path.abspath(area)), a_real):
        parts: tuple[str, ...] = ()
    else:
        a_real, parts = _rel_parts(d, area)
    cur = a_real
    for part in parts:
        cur = cur / part
        if os.path.lexists(cur) and stat.S_ISLNK(os.lstat(cur).st_mode):
            raise AtRestError(f"{cur} is a symlink; no link is followed inside a protected area")
        _mkdir_private(cur)
    return _check_chain(a_real, parts, check_root=is_key)


def write_private(path: str | os.PathLike, data: bytes, area: str | os.PathLike) -> Path:
    """Write ``data`` atomically to ``path`` (below ``area``) as a 0600 file; parents are created 0700."""
    area = Path(area)
    a_real, parts = _rel_parts(path, area)                  # refuse before creating anything
    is_key = not _is_work(area)
    _private_dir(a_real.joinpath(*parts[:-1]) if parts[:-1] else a_real, area, is_key)
    target = _check_chain(a_real, parts, check_root=is_key)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    _write_new(tmp, data)
    try:
        os.replace(tmp, target)                             # rename never follows a link at the target
    except BaseException:
        _unlink_quiet(tmp)
        raise
    return target


def _write_new(p: Path, data: bytes) -> None:
    """Create ``p`` (must not exist) 0600 with ``data``, fsync'd; removes it again on any failure."""
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        _unlink_quiet(p)
        raise
    os.close(fd)


def _unlink_quiet(p: Path) -> None:
    try:
        os.unlink(p)
    except FileNotFoundError:
        pass


def read_private(path: str | os.PathLike, area: str | os.PathLike) -> bytes:
    """Read a file below ``area``. The name is checked first; then the open walks the chain itself: the area root,
    then each directory with ``O_DIRECTORY|O_NOFOLLOW`` relative to its parent's fd (owner and mode re-checked on
    each fd), then the leaf with ``O_NOFOLLOW``, its fd checked for owner, regular-file type and mode. A file or a
    parent directory swapped for a link after the name check is therefore never followed."""
    p = ensure_private(path, area)
    a_real, parts = _rel_parts(path, Path(area))
    fds: list[int] = []
    try:
        try:
            fds.append(os.open(a_real, os.O_RDONLY | os.O_DIRECTORY))
            for part in parts[:-1]:
                fds.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fds[-1]))
                _check_stat(p, os.fstat(fds[-1]))
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fds[-1])
        except OSError as e:
            raise AtRestError(f"{p}: cannot open without following links ({e.strerror})") from None
        fds.append(fd)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise AtRestError(f"{p} is not a regular file")
        _check_stat(p, st)
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        for d in reversed(fds):
            os.close(d)


# ------------------------------------------------------------------ bundles (work area)

def bundle_dir(system: str, root: str | os.PathLike | None = None) -> Path:
    """``<work>/bundles/<system>/``, created 0700."""
    r = _work_root(root)
    return _private_dir(r / "bundles" / _name("system", system), r, is_key=False)


def bundle_path(system: str, unit: str, build: str, root: str | os.PathLike | None = None) -> Path:
    """Where the bundle of ``unit`` from build ``build`` lives (the directory is created, the file is not)."""
    return bundle_dir(system, root) / f"{_name('unit', unit)}.{_name('build', build)}{BUNDLE_SUFFIX}"


def write_bundle(system: str, unit: str, build: str, records: Iterable[dict],
                 root: str | os.PathLike | None = None) -> Path:
    """Validate and write one bundle (``ccg.bundle.dumps``: §8.3 order, schema) as a 0600 file."""
    from ccg import bundle                                  # deferred: jsonschema is only needed to write

    data = bundle.dumps(records)
    return write_private(bundle_path(system, unit, build, root), data, area=_work_root(root))


def read_bundle(system: str, unit: str, build: str, root: str | os.PathLike | None = None) -> bytes:
    """The bytes of one stored bundle, read under the work-area checks (never through a link)."""
    return read_private(bundle_path(system, unit, build, root), area=_work_root(root))


def prune_unit(system: str, unit: str, keep_build: str, root: str | os.PathLike | None = None) -> list[Path]:
    """Retention: delete every completed bundle of ``unit`` except build ``keep_build`` (the last good one).

    An in-progress write (a ``.tmp`` name) is never touched. Returns the removed paths. Refuses if
    ``keep_build``'s bundle does not exist."""
    r = _work_root(root)
    d = bundle_dir(system, r)
    keep = bundle_path(system, unit, keep_build, r)
    ensure_private(keep, r)
    if not keep.is_file() or keep.is_symlink():
        raise AtRestError(f"no bundle {keep.name} to keep; nothing pruned")
    prefix = f"{_name('unit', unit)}."
    removed = []
    for f in sorted(d.iterdir()):
        n = f.name
        if not (n.startswith(prefix) and n.endswith(BUNDLE_SUFFIX)) or f == keep:
            continue
        if not _NAME_RE.fullmatch(n[len(prefix):-len(BUNDLE_SUFFIX)]):
            continue                                        # not <unit>.<build>: never this unit's to delete
        ensure_private(f, r)
        f.unlink()                                          # unlinks a link itself, never its target
        removed.append(f)
    return removed


# ------------------------------------------------------------------ the HMAC key (key area)

def key_path(system: str, keys_root: str | os.PathLike | None = None) -> Path:
    """``~/.config/ccg/keys/<system>.hmac``, the single accessor of the key's location (nothing is created)."""
    return _key_root(keys_root) / f"{_name('system', system)}.hmac"


def bind_key_check(key: bytes) -> str:
    """``header.bindKeyCheck`` = ``entry_hmac(key, BIND_KEYCHECK_TAG, "")`` (-bundle-spec5.md §2)."""
    return hmac.new(key, BIND_KEYCHECK_TAG.encode() + b"\x00", hashlib.sha256).hexdigest()


def _refuse_test_key(key: bytes, where: Path) -> None:
    check = bind_key_check(key)
    if any(hmac.compare_digest(check, c) for c in PUBLISHED_TEST_KEYCHECKS):
        raise AtRestError(f"{where}: refusing a published test key of -bundle-spec5.md §9; it is never a system key")


def hmac_key(system: str, keys_root: str | os.PathLike | None = None) -> bytes:
    """The system's HMAC key: generated once with ``secrets.token_bytes(32)`` (the OS CSPRNG), then read back.

    The first write is atomic (a complete 0600 temp file, then ``os.link`` onto the name, which fails if another
    call won the race; the loser reads the winner's key). A published test key is refused when generated (it is
    then never written) and on every load. Never printed or logged. Rotation: delete the file by hand, then run a
    full rebuild, because the key changes every ``valueHash``."""
    k = _key_root(keys_root)
    p = key_path(system, k)
    _private_dir(k, k, is_key=True)
    target = ensure_private(p, k)
    if not os.path.lexists(target):
        new = secrets.token_bytes(KEY_BYTES)
        _refuse_test_key(new, target)
        tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
        _write_new(tmp, new.hex().encode() + b"\n")
        try:
            os.link(tmp, target)
        except FileExistsError:
            pass                                            # a concurrent first call won; read its key below
        finally:
            _unlink_quiet(tmp)
    text = read_private(target, k).decode("ascii", "replace").strip()
    if not re.fullmatch(rf"[0-9a-f]{{{2 * KEY_BYTES}}}", text):
        raise AtRestError(f"{p} is malformed (expected {2 * KEY_BYTES} lowercase hex characters)")
    key = bytes.fromhex(text)
    _refuse_test_key(key, p)
    return key


# ------------------------------------------------------------------ ACL credentials (credentials dir)

def acl_path(system: str, keys_root: str | os.PathLike | None = None) -> Path:
    """``<credentials dir>/<system>.acl``: the four ACL users' passwords, one line per user."""
    return credentials_dir(keys_root) / f"{_name('system', system)}.acl"


def write_acl(system: str, data: bytes, keys_root: str | os.PathLike | None = None) -> Path:
    """Write the ACL file (P1-19's ``create_users``), 0600, in the credentials dir, which is created 0700."""
    d = credentials_dir(keys_root)
    _private_dir(d, d, is_key=True)
    return write_private(acl_path(system, keys_root), data, area=d)


def read_acl(system: str, keys_root: str | os.PathLike | None = None) -> bytes:
    """Read the ACL file (the clients), under the credentials-dir checks."""
    return read_private(acl_path(system, keys_root), area=credentials_dir(keys_root))
