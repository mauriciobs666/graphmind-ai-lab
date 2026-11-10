"""Legacy items: ``mint-legacy`` and qualified/bare name resolution (plan 3.6).

A legacy item is an old per-home name (``architect:K-004``) that gets one scheme
ID the first time work touches it.  Minting is allocation only: it reads and
writes the registry/journal inside the allocation lock and never opens a document.

Ambiguity comes from the **declared** namespace list (``project.json``
``legacy_namespaces``: prefix -> homes), never from what has been minted, so a
bare name's meaning cannot change when a second home is minted later.
"""
import re

from . import alloc, project, resolve
from .store import Store

EXIT_AMBIGUOUS, EXIT_UNKNOWN, EXIT_NOT_MINTED = 3, 4, 5
_PREFIX = re.compile(r"^[A-Za-z]+-")
_namespaces = {}


def _split(spec):
    """``home:name`` -> (home, name); raises AllocError otherwise."""
    home, sep, name = spec.partition(":")
    if not (sep and home and name):
        raise alloc.AllocError(f"legacy name must be <home>:<name>, got {spec!r}")
    return home, name


def _prefix(name):
    m = _PREFIX.match(name)
    return m.group(0) if m else None


def mint(root, spec, type_, parent=None, title=None, home=None, store=None):
    """Return the scheme ID of legacy item ``spec``, minting it once (idempotent).

    ``home`` is the component the new item lives in (FR-19); it defaults to the
    legacy home when that is a declared component, else the allocator default.
    """
    legacy_home, name = _split(spec)
    if legacy_home not in project.load_namespaces(root).get(_prefix(name), []):
        raise alloc.AllocError(
            f"{legacy_home!r} is not a declared legacy home for {name!r}: add it to "
            "project.json legacy_namespaces (citing-only homes are not homes)")
    key = alloc.load_key(root)
    if home is None and legacy_home in project.load_components(root):
        home = legacy_home
    store = store or Store(root, key)
    with store.transaction() as tx:
        uid = tx.state.legacy.get((legacy_home, name))
        if uid is not None:
            return tx.state.items[uid].id
        return alloc.allocate(tx, key, type_, parent=parent, title=title, home=home,
                              legacy={"home": legacy_home, "name": name})


def _minted(st, home, name):
    uid = st.legacy.get((home, name))
    return st.items[uid].id if uid else None


def fallback(st, name):
    """``resolve.FALLBACKS`` hook: qualified ``home:name`` or bare ``name``."""
    if ":" in name:
        home, _, old = name.partition(":")
        if home not in _namespaces.get(_prefix(old), []):
            return None
        uid = st.legacy.get((home, old))
        if uid is None:
            raise resolve.ResolveError(
                f"{name!r} is a known legacy item but not minted: "
                f"wi mint-legacy {name} --type K|F|D", EXIT_NOT_MINTED)
        return uid, "legacy"
    homes = _namespaces.get(_prefix(name), [])
    if not homes:
        return None
    if len(homes) == 1:
        uid = st.legacy.get((homes[0], name))
        if uid is None:
            raise resolve.ResolveError(
                f"{name!r} is a known legacy item ({homes[0]}:{name}) but not minted",
                EXIT_NOT_MINTED)
        return uid, "bare"
    cands = [f"{h}:{name}" + (f" -> {_minted(st, h, name)}" if _minted(st, h, name) else " (not minted)")
             for h in homes]
    raise resolve.ResolveError(
        f"{name!r} is ambiguous, defined by several homes: " + "; ".join(cands), EXIT_AMBIGUOUS)


def install(namespaces):
    """Register the fallback (once) with the declared namespaces."""
    _namespaces.clear()
    _namespaces.update(namespaces)
    if fallback not in resolve.FALLBACKS:
        resolve.FALLBACKS.append(fallback)
