"""The one predicate for a string the store's index cannot hold, and the one list of indexed properties (``-p1-linking.md`` P1-18b).

Engine facts, falkordb v4.18.11 (live-observed; ``tests/test_fuzz_idhash_live.py``, ``tests/test_indexed_live.py``):

* an indexed string over :data:`INDEXED_VALUE_MAX_BYTES` UTF-8 bytes is stored but NOT indexed: an equality anchor on it silently returns nothing, and a
  ``UNIQUE`` constraint created after the load over one above 5,000 bytes crashes the server;
* the index trims :data:`INDEX_TRIMMED_CHARS` off both ends of a STORED value, not off a lookup value: ``= ' x '`` finds nothing and ``= 'x'`` also
  returns the nodes holding ``' x '``;
* an empty string is stored but an equality on it matches nothing (live grid, P1-18b);
* text that is not valid Unicode (a lone surrogate) cannot be sent at all.

Two callers, one definition: the assembler (``ccg.assemble.rows.classify_bundle``) refuses the ONE unit whose bundle holds such a value
(``bundle:unindexable``), and the loader's preflight (``ccg.store.loader``) refuses the row as the fail-closed backstop. A result carries a code, a size
and a limit, never the value.

The property list is read from ``ccg/store/ddl_v1.cypher`` (:func:`ccg.store.parse_ddl`), never typed twice; :data:`BUNDLE_SOURCES` says which record
field becomes which indexed property, and an import-time check makes a DDL index with no source and no declared exemption impossible.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple

from ccg.store import parse_ddl

#: An indexed string holds at most this many UTF-8 bytes (bytes, not characters).
INDEXED_VALUE_MAX_BYTES = 4096
#: The six ASCII characters the index trims off both ends of a stored value (not NBSP, nor any other Unicode space).
INDEX_TRIMMED_CHARS = frozenset(" \t\n\v\f\r")
#: ``(label, property)`` of every DDL index.
INDEXED_PROPS = frozenset((s[1], s[2]) for s in parse_ddl() if s[0] == "index")
_BY_LABEL: dict[str, tuple[str, ...]] = {}
for _label, _prop in sorted(INDEXED_PROPS):
    _BY_LABEL[_label] = _BY_LABEL.get(_label, ()) + (_prop,)

UNICODE, EMPTY, WHITESPACE, TOO_LONG, NOT_STR = "unicode", "empty", "whitespace", "too-long", "not-a-string"


class Unindexable(NamedTuple):
    """Why a string cannot be indexed: ``reason`` is one of :data:`UNICODE`, :data:`EMPTY`, :data:`WHITESPACE`, :data:`TOO_LONG`, :data:`NOT_STR`; ``size`` its UTF-8 bytes (0 when not
    valid Unicode or not a string); ``limit`` the bound. Carries no text of the value."""
    reason: str
    size: int
    limit: int = INDEXED_VALUE_MAX_BYTES


def check(value: Any) -> Unindexable | None:
    """``None`` when ``value`` is absent (``None``: another gate's business) or a string the index holds faithfully, else why it cannot be indexed
    (a value that is not a string at all, then the first rule that fires: not valid Unicode, then empty, then ASCII whitespace at an end, then over the
    byte bound). The bundle schema types every mapped field as a string; this is the refusal if one ever is not, never a silent pass."""
    if value is None:
        return None
    if not isinstance(value, str):
        return Unindexable(NOT_STR, 0)
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return Unindexable(UNICODE, 0)
    if value == "":
        return Unindexable(EMPTY, 0)
    if value[:1] in INDEX_TRIMMED_CHARS or value[-1:] in INDEX_TRIMMED_CHARS:
        return Unindexable(WHITESPACE, size)
    if size > INDEXED_VALUE_MAX_BYTES:
        return Unindexable(TOO_LONG, size)
    return None


def props_of(label: str) -> tuple[str, ...]:
    """The indexed properties of ``label`` (DDL order is not kept: sorted)."""
    return _BY_LABEL.get(label, ())


# -----------------------------------------------------------------------------------------------------------------------------------
# which bundle field becomes which indexed property (``ccg.assemble.rows._record`` is the other half of this table)
# -----------------------------------------------------------------------------------------------------------------------------------
#: record type -> ``(label, property, record field)``; ``header`` is the header record, whose ``unit.repoId`` and ``skippedFiles[].path`` are special-cased.
BUNDLE_SOURCES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "header": (("Unit", "unitId", "unitId"),),
    "file": (("File", "fileId", "id"), ("File", "path", "path")),
    "symbol": (("Symbol", "symbolId", "id"), ("Symbol", "nameNorm", "nameNorm"), ("Symbol", "qname", "qname")),
    "param": (("Param", "pointId", "id"),),
    "ret": (("Ret", "pointId", "id"),),
    "cond": (("Cond", "pointId", "id"),),
    "callsite": (("CallSite", "callSiteId", "id"),),
    "dependency": (("Dependency", "depId", "depId"),),
    "configEntry": (("ConfigEntry", "entryId", "id"), ("ConfigEntry", "path", "path")),
    "launchFact": (("ConfigEntry", "entryId", "id"), ("ConfigEntry", "path", "path")),
}
#: header fields outside ``BUNDLE_SOURCES``' flat shape: ``unit.repoId`` (-> ``Repo.repoId``, ``SkippedFile.repoId``) and ``skippedFiles[].path``.
HEADER_REPO = ("Repo", "repoId")
HEADER_SKIPPED = ("SkippedFile", "path")
HEADER_SKIPPED_REPO = ("SkippedFile", "repoId")
#: indexed properties a bundle does not carry as a field: ids formed from hashes by the assembler or the linker (``Name.nameKey``: the hash of the
#: callee name, so its source ``calleeName`` is only checked for being valid Unicode, which ``ids.name_key`` needs), and the linker's / pack runtime's
#: ``Link.linkId`` and ``ContractPoint.cpId``/``keyNorm`` (the pack's key, raw; checked by the loader's backstop only).
NOT_IN_A_BUNDLE = frozenset({("Name", "nameKey"), ("Link", "linkId"), ("ContractPoint", "cpId"), ("ContractPoint", "keyNorm")})

_mapped = {(lbl, prop) for rows in BUNDLE_SOURCES.values() for lbl, prop, _ in rows} | {HEADER_REPO, HEADER_SKIPPED, HEADER_SKIPPED_REPO}
if _mapped | NOT_IN_A_BUNDLE != INDEXED_PROPS or _mapped & NOT_IN_A_BUNDLE:
    raise ImportError("ccg.indexed: the DDL's indexed properties and the bundle sources disagree: "
                      f"unmapped {sorted(INDEXED_PROPS - _mapped - NOT_IN_A_BUNDLE)}, not indexed {sorted((_mapped | NOT_IN_A_BUNDLE) - INDEXED_PROPS)}")


def bundle_violations(records: Sequence[Mapping]) -> Iterator[tuple[str, str, int, Unindexable]]:
    """Every ``(label, property, record index, verdict)`` in ``records`` (a bundle, header first) whose value the index cannot hold. Value-free."""
    for i, r in enumerate(records):
        if not isinstance(r, Mapping):
            continue
        t = r.get("type")
        for label, prop, fld in BUNDLE_SOURCES.get(t, ()):
            if (v := check(r.get(fld))) is not None:
                yield label, prop, i, v
        if t == "header":
            unit = r.get("unit")
            if isinstance(unit, Mapping):
                for (label, prop) in (HEADER_REPO, HEADER_SKIPPED_REPO):
                    if (v := check(unit.get("repoId"))) is not None:
                        yield label, prop, i, v
            for sk in r.get("skippedFiles") or ():
                if isinstance(sk, Mapping) and (v := check(sk.get("path"))) is not None:
                    yield HEADER_SKIPPED[0], HEADER_SKIPPED[1], i, v
        elif t == "callsite":
            # Name.nameKey = ids.name_key(lang, calleeName) needs valid Unicode, and Name.name is stored exactly as given while ``RESOLVES_TO`` joins it
            # to the symbols' raw names: an edge-whitespace callee would be stored padded (and miss its symbol), so it is refused, never stripped.
            # An empty or long callee name is not refused (Name.name is not indexed, the key is a hash).
            if (v := check(r.get("calleeName"))) is not None and v.reason in (UNICODE, WHITESPACE, NOT_STR):
                yield "Name", "nameKey", i, v


def first_bundle_violation(records: Sequence[Mapping]) -> tuple[str, str, int, Unindexable] | None:
    return next(bundle_violations(records), None)
