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
import json
import os
import re
import secrets
import stat
import time
from pathlib import Path
from typing import Iterable

WORK_ROOT = Path(__file__).resolve().parent.parent / ".ccg-work"
REPO_ROOT = WORK_ROOT.parent.parent
KEY_ROOT = Path.home() / ".config" / "ccg" / "keys"
BUNDLE_SUFFIX = ".bundle.jsonl"
STATE_SUFFIX = ".state.json"
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


class RealCredentialsDirTouched(AtRestError):
    """Test mode (``CCG_FORBID_REAL_CREDENTIALS=1``) reached the real ``~/.config/ccg``."""


#: ``tests/conftest.py`` sets this in the environment (child processes inherit it, ``importlib.reload`` keeps it): while it is ``1`` every
#: key-area / credentials operation refuses a path under the real ``~/.config/ccg`` (``$HOME``'s and the password database's). Unset: no effect.
FORBID_REAL_ENV = "CCG_FORBID_REAL_CREDENTIALS"


def _real_credential_roots() -> set[Path]:
    import pwd
    homes = {Path(os.path.expanduser("~")), Path(pwd.getpwuid(os.getuid()).pw_dir)}
    roots = {h / ".config" / "ccg" for h in homes}
    return roots | {Path(os.path.realpath(r)) for r in roots}


def assert_not_real_credentials(path: str | os.PathLike) -> None:
    """The one test-mode choke point: raises :class:`RealCredentialsDirTouched` for ``path`` (lexically or by realpath) at or under the real
    ``~/.config/ccg`` when ``CCG_FORBID_REAL_CREDENTIALS=1``; a no-op otherwise. Pure: it touches nothing."""
    if os.environ.get(FORBID_REAL_ENV) != "1":
        return
    raw = os.path.abspath(os.fspath(path))
    cands = {Path(raw), Path(os.path.realpath(raw))}
    roots = _real_credential_roots()
    if any(c == r or r in c.parents for c in cands for r in roots):
        raise RealCredentialsDirTouched(f"test mode: {raw} is under the real ~/.config/ccg; give the test its own keys_root/CCG_CREDENTIALS_DIR")


# ------------------------------------------------------------------ areas

def _key_root(keys_root: str | os.PathLike | None) -> Path:
    k = Path(keys_root) if keys_root is not None else KEY_ROOT
    assert_not_real_credentials(k)
    rk, rr = Path(os.path.realpath(k)), Path(os.path.realpath(REPO_ROOT))
    if rk == rr or rr in rk.parents:
        raise AtRestError(f"key area {k} is inside the repository {rr}; keys live outside it (-bundle-spec3.md §5.17)")
    return k


def _work_root(root: str | os.PathLike | None) -> Path:
    return Path(root) if root is not None else WORK_ROOT


def credentials_dir(keys_root: str | os.PathLike | None = None) -> Path:
    """``$CCG_CREDENTIALS_DIR`` if set, else the key area; refused if inside the repository. In test mode (``CCG_FORBID_REAL_CREDENTIALS=1``)
    an explicit ``keys_root`` that names another directory than the variable is an error, never silently overridden."""
    env = os.environ.get("CCG_CREDENTIALS_DIR")
    if env and keys_root is not None and os.environ.get(FORBID_REAL_ENV) == "1" \
            and os.path.realpath(env) != os.path.realpath(keys_root):
        raise AtRestError("test mode: keys_root and CCG_CREDENTIALS_DIR name different directories")
    return _key_root(env) if env else _key_root(keys_root)


def _name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise AtRestError(f"{kind} name {value!r} is not one safe path component ([A-Za-z0-9][A-Za-z0-9_-]{{0,63}})")
    return value


# ------------------------------------------------------------------ checks

def _rel_parts(path: str | os.PathLike, area: Path) -> tuple[Path, tuple[str, ...]]:
    """(resolved area root, the path's components below it), lexically: no ``..``, strictly below the root."""
    raw = os.fspath(path)
    assert_not_real_credentials(area)
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
    assert_not_real_credentials(area)
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
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    _write_new(tmp, data)
    try:
        os.replace(tmp, target)                             # rename never follows a link at the target
    except BaseException:
        _unlink_quiet(tmp)
        raise
    return target


def _write_new(p: Path, data: bytes) -> None:
    """Create ``p`` (must not exist) 0600 with ``data``, fsync'd; removes it again on any failure."""
    assert_not_real_credentials(p)
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


def read_private(path: str | os.PathLike, area: str | os.PathLike, single_link: bool = False) -> bytes:
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
        if single_link and st.st_nlink != 1:
            raise AtRestError(f"{p} has {st.st_nlink} hard links; a credentials file has exactly one")
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


# --- per-unit carried state (work area): the refresh's marker of a unit whose last refresh failed

_STATE_STATUSES = ("fresh", "stale-kept")
_STATE_KEYS = frozenset({"status", "builtCommit", "failedAt", "failReason"})
_STAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_REASON_RE = re.compile(r"[a-z]+:[a-z][a-z0-9-]*")


def unit_state_path(system: str, unit: str, root: str | os.PathLike | None = None) -> Path:
    """Where the carried state of ``unit`` lives, beside its bundles (the directory is created, the file is not)."""
    return bundle_dir(system, root) / f"{_name('unit', unit)}{STATE_SUFFIX}"


def _check_state(state: object) -> dict:
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise AtRestError("unit state: not exactly status, builtCommit, failedAt, failReason")
    status, commit, at, reason = (state[k] for k in ("status", "builtCommit", "failedAt", "failReason"))
    if status not in _STATE_STATUSES or not isinstance(commit, str) or not _NAME_RE.fullmatch(commit):
        raise AtRestError("unit state: bad status or builtCommit")
    marked = (at, reason) != (None, None)
    if marked != (status == "stale-kept") or (marked and not (isinstance(at, str) and _STAMP_RE.fullmatch(at)
                                                              and isinstance(reason, str) and _REASON_RE.fullmatch(reason))):
        raise AtRestError("unit state: the failure marker does not match the status")
    return dict(state)


def write_unit_state(system: str, unit: str, status: str, built_commit: str, failed_at: str | None = None,
                     fail_reason: str | None = None, root: str | os.PathLike | None = None) -> Path:
    """Record ``unit``'s status after a refresh (``fresh`` or ``stale-kept``; the latter carries ``failedAt`` and one failure code), atomically,
    0600. The record is value-free by construction: four closed fields, no text from a repository. Overwrites the unit's previous record."""
    state = _check_state({"status": status, "builtCommit": built_commit, "failedAt": failed_at, "failReason": fail_reason})
    data = (json.dumps(state, sort_keys=True) + "\n").encode("ascii")
    return write_private(unit_state_path(system, unit, root), data, area=_work_root(root))


def read_unit_state(system: str, unit: str, root: str | os.PathLike | None = None) -> dict | None:
    """The record :func:`write_unit_state` left, or None when the unit has none. A record that is a link, not a regular file, not
    parseable or not exactly the four closed fields raises :class:`AtRestError` (the caller decides what a bad record means)."""
    p = unit_state_path(system, unit, root)
    if not os.path.lexists(p):
        return None
    try:
        return _check_state(json.loads(read_private(p, area=_work_root(root))))
    except (ValueError, UnicodeDecodeError):
        raise AtRestError("unit state: not a JSON record") from None


def clear_unit_state(system: str, unit: str, root: str | os.PathLike | None = None) -> bool:
    """Remove ``unit``'s record (a link is unlinked itself, never followed); True if one existed."""
    p = unit_state_path(system, unit, root)
    if not os.path.lexists(p):
        return False
    ensure_private(p, _work_root(root))
    p.unlink()
    return True


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


def prune_orphans(system: str, live_units: Iterable[str], root: str | os.PathLike | None = None) -> list[Path]:
    """Retention for units that no longer exist: delete every completed bundle ``<unit>.<build>.bundle.jsonl`` whose ``<unit>`` is not in
    ``live_units``. An in-progress write (a ``.tmp`` name) and any other name is never touched; a link is unlinked itself, never followed
    (``ensure_private`` refuses it first, so the call raises :class:`AtRestError` on one). Returns the removed paths."""
    r = _work_root(root)
    d = bundle_dir(system, r)
    live = frozenset(live_units)
    removed = []
    for f in sorted(d.iterdir()):
        n = f.name
        if n.endswith(STATE_SUFFIX):                        # a unit's carried state goes with the unit
            if _NAME_RE.fullmatch(n[:-len(STATE_SUFFIX)]) and n[:-len(STATE_SUFFIX)] not in live:
                ensure_private(f, r)
                f.unlink()
                removed.append(f)
            continue
        if not n.endswith(BUNDLE_SUFFIX):
            continue
        unit, _, build = n[:-len(BUNDLE_SUFFIX)].partition(".")
        if unit in live or not _NAME_RE.fullmatch(unit) or not _NAME_RE.fullmatch(build):
            continue                                        # not <unit>.<build>: never ours to delete
        ensure_private(f, r)
        f.unlink()
        removed.append(f)
    return removed


#: The temp name of an interrupted bundle write: ``.<unit>.<build>.bundle.jsonl.<hex>.tmp``. Both writers (``write_private``, the one every
#: build goes through, and ``ccg.bundle.write_bundle``) draw 8 random bytes (16 hex digits); 8 hex digits are still accepted, the name an
#: older ``write_private`` left behind.
_STALE_TMP_RE = re.compile(
    rf"\.{_NAME_RE.pattern}\.(?:{_NAME_RE.pattern}{re.escape(BUNDLE_SUFFIX)}|{re.escape(STATE_SUFFIX[1:])})\.(?:[0-9a-f]{{8}}|[0-9a-f]{{16}})\.tmp")


def sweep_stale_tmp(system: str, older_than_s: float = 3600, root: str | os.PathLike | None = None,
                    now: float | None = None) -> list[Path]:
    """Remove the temp files a crashed bundle write left in ``system``'s bundle directory (and only there); returns the removed paths.

    A file goes iff its name is exactly ``.<unit>.<build>.bundle.jsonl.<8 or 16 hex>.tmp``, it is a regular file (a link, a directory or
    a device with that name is left alone) owned by this user, and its mtime is at least ``older_than_s`` seconds before ``now`` (a write
    in progress is younger). Nothing else is touched: no completed bundle, no other directory, no file of another name. The directory is
    reached through ``bundle_dir`` (lexical checks, no link below the area root) and then by file descriptor, ``O_NOFOLLOW``, and every
    unlink is relative to that descriptor, so a directory swapped for a link after the check is not followed."""
    if isinstance(older_than_s, bool) or not isinstance(older_than_s, (int, float)) or older_than_s < 0:
        raise ValueError("older_than_s is a non-negative number of seconds")
    r = _work_root(root)
    d = bundle_dir(system, r)
    cutoff = (time.time() if now is None else now) - older_than_s
    removed: list[Path] = []
    try:
        fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        raise AtRestError(f"{d}: cannot open without following links ({e.strerror})") from None
    try:
        _check_stat(d, os.fstat(fd))
        with os.scandir(fd) as it:
            names = sorted(e.name for e in it)
        for name in names:
            if not _STALE_TMP_RE.fullmatch(name):
                continue
            try:
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mtime > cutoff:
                continue
            try:
                os.unlink(name, dir_fd=fd)
            except FileNotFoundError:
                continue
            removed.append(d / name)
    finally:
        os.close(fd)
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


def prepare_acl_target(system: str, keys_root: str | os.PathLike | None = None) -> Path:
    """Everything ``write_acl`` would refuse, checked WITHOUT writing (the credentials dir is created 0700 as ``write_acl`` would): not in the
    repository, no link on the way, own and private, the existing file (if any) not a link, the directory writable. ``create_users`` calls it
    before the first ``ACL SETUSER``, so a bad directory cannot reset live credentials it then cannot store."""
    d = credentials_dir(keys_root)
    _private_dir(d, d, is_key=True)
    p = acl_path(system, keys_root)
    a_real, parts = _rel_parts(p, d)
    _check_chain(a_real, parts, check_root=True)
    if not os.access(d, os.W_OK | os.X_OK):
        raise AtRestError(f"{d} is not writable")
    return p


def write_acl(system: str, data: bytes, keys_root: str | os.PathLike | None = None) -> Path:
    """Write the ACL file (P1-19's ``create_users``), 0600, in the credentials dir, which is created 0700."""
    d = credentials_dir(keys_root)
    _private_dir(d, d, is_key=True)
    return write_private(acl_path(system, keys_root), data, area=d)


def remove_acl(system: str, keys_root: str | os.PathLike | None = None) -> bool:
    """Remove ``system``'s ACL file (a regular file below the credentials dir; never a link). True if one was removed."""
    d = credentials_dir(keys_root)
    p = ensure_private(acl_path(system, keys_root), d)
    try:
        os.unlink(p)
    except FileNotFoundError:
        return False
    return True


_ACL_TMP_RE = re.compile(rf"\.{_NAME_RE.pattern}\.acl\.(?:[0-9a-f]{{8}}|[0-9a-f]{{16}})\.tmp")


def sweep_stale_acl_tmp(older_than_s: float = 3600, keys_root: str | os.PathLike | None = None, now: float | None = None) -> list[Path]:
    """Remove the temp files a killed ``write_acl`` left in the credentials dir (``.<system>.acl.<8 hex>.tmp``, 0600, holding passwords):
    regular files of this user at least ``older_than_s`` old, nothing else, each unlink relative to the directory descriptor."""
    d = credentials_dir(keys_root)
    if not os.path.lexists(d):
        return []
    cutoff = (time.time() if now is None else now) - older_than_s
    try:
        fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        raise AtRestError(f"{d}: cannot open without following links ({e.strerror})") from None
    removed: list[Path] = []
    try:
        _check_stat(d, os.fstat(fd))
        with os.scandir(fd) as it:
            names = sorted(e.name for e in it)
        for name in names:
            if not _ACL_TMP_RE.fullmatch(name):
                continue
            try:
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mtime > cutoff:
                    continue
                os.unlink(name, dir_fd=fd)
            except FileNotFoundError:
                continue
            removed.append(d / name)
    finally:
        os.close(fd)
    return removed


def read_acl(system: str, keys_root: str | os.PathLike | None = None) -> bytes:
    """Read the ACL file (the clients), under the credentials-dir checks."""
    return read_private(acl_path(system, keys_root), area=credentials_dir(keys_root), single_link=True)
