"""Canonical tuples and ids — the one place hashing is defined.

Spec: ``docs/plans/connected-code-graph-bundle-spec.md`` §3.1 (the hash) and §3.2 (the tuples);
P0 plan ``docs/plans/connected-code-graph-p0.md`` §2 D10.

    id(tuple) = lowercase-hex(SHA-256(canonical-json(normalize(tuple))))[0:32]
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence

ID_HEX_LENGTH = 32

Element = str | int | Sequence["Element"]


def _token(value: str) -> str:
    """NFC first, then strip the characters ``str.isspace()`` is true for (§3.1)."""
    return unicodedata.normalize("NFC", value).strip()


def _normalize(value: Element) -> Element:
    """NFC then strip every string, recursively; integers unchanged (§3.1)."""
    if isinstance(value, str):
        return _token(value)
    # bool is an int subclass, so it must be refused before the int branch.
    if isinstance(value, bool):
        raise TypeError("booleans never appear in an id tuple")
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    raise TypeError(f"id tuple elements are str or int, got {type(value).__name__}")


def canonical_bytes(tup: Sequence[Element]) -> bytes:
    """The canonical JSON bytes of the normalized tuple (§3.1).

    Element types are checked per position by the constructors, not here: a raw tuple such as
    GV-18's (a string where §3.2 wants an integer) must stay hashable.
    """
    if not isinstance(tup, (list, tuple)):
        raise TypeError(f"an id tuple is a list or tuple, got {type(tup).__name__}")
    if not tup:
        raise ValueError("an id tuple has at least its record-type tag")
    if not isinstance(tup[0], str):
        raise TypeError("an id tuple's first element is its record-type tag, a str")
    return json.dumps(
        _normalize(list(tup)), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def tuple_id(tup: Sequence[Element]) -> str:
    """``h(tuple)``: the 32-hex id of a canonical tuple whose first element is its tag."""
    return hashlib.sha256(canonical_bytes(tup)).hexdigest()[:ID_HEX_LENGTH]


# --- §3.2 canonical tuples -----------------------------------------------------------------
# Each constructor supplies the record-type tag (BF1) and the element order the spec fixes.
# Inputs are passed raw; normalization is the hash's job, so a caller never pre-strips.
# Each constructor checks every position's type: a string where an integer belongs (or the
# reverse) would otherwise hash to a different, valid-looking id (review S09 M1).

#: Languages whose function-like symbols carry a `signature` 5th element (§3.2, §4.3).
SIGNATURE_LANGS = frozenset({"java", "kotlin", "csharp", "cpp"})
#: `pointId` kinds (§3.2). A call site's point id is its `callSiteId`, not a `point`.
POINT_KINDS = frozenset({"param", "ret", "cond"})


def _strs(*values: object) -> None:
    for v in values:
        if not isinstance(v, str):
            raise TypeError(f"expected str, got {type(v).__name__}: {v!r}")


def _ints(*values: object) -> None:
    for v in values:
        # `type(v) is int` also refuses bool, an int subclass.
        if type(v) is not int:
            raise TypeError(f"expected int, got {type(v).__name__}: {v!r}")


def unit_id(repo_id: str, root_path: str, lang: str) -> str:
    _strs(repo_id, root_path, lang)
    return tuple_id(["unit", repo_id, root_path, lang])


def file_id(repo_id: str, path: str) -> str:
    _strs(repo_id, path)
    return tuple_id(["file", repo_id, path])


def symbol_id(lang: str, repo_id: str, qname: str, signature: str | None = None) -> str:
    """``symbolId``. Pass ``signature`` only for a function-like symbol in a §4.3 language;
    a type, ``<module>`` or ``<anon>#N`` scope passes none and gets 4 elements."""
    _strs(lang, repo_id, qname)
    tup: list[Element] = ["symbol", lang, repo_id, qname]
    if signature is not None:
        _strs(signature)
        if _token(lang) not in SIGNATURE_LANGS:
            raise ValueError(
                f"a signature is carried only for {sorted(SIGNATURE_LANGS)}, not {lang!r}"
            )
        if not _token(signature):
            raise ValueError("an empty signature is never valid; §4.3's minimum is '()'")
        tup.append(signature)
    return tuple_id(tup)


def callsite_id(sym_id: str, callee_name: str, ordinal: int) -> str:
    _strs(sym_id, callee_name)
    _ints(ordinal)
    return tuple_id(["callsite", sym_id, callee_name, ordinal])


def point_id(sym_id: str, kind: str, n: int) -> str:
    _strs(sym_id, kind)
    _ints(n)
    if _token(kind) not in POINT_KINDS:
        raise ValueError(f"point kind must be one of {sorted(POINT_KINDS)}, not {kind!r}")
    return tuple_id(["point", sym_id, kind, n])


def constant_id(lang: str, repo_id: str, qname: str) -> str:
    _strs(lang, repo_id, qname)
    return tuple_id(["constant", lang, repo_id, qname])


def config_entry_id(repo_id: str, path: str, key_path: str) -> str:
    _strs(repo_id, path, key_path)
    return tuple_id(["configEntry", repo_id, path, key_path])


def launch_fact_id(repo_id: str, path: str, kind: str, ordinal: int) -> str:
    _strs(repo_id, path, kind)
    _ints(ordinal)
    return tuple_id(["launchFact", repo_id, path, kind, ordinal])


def schema_entry_id(repo_id: str, path: str, object_key: str) -> str:
    _strs(repo_id, path, object_key)
    return tuple_id(["schemaEntry", repo_id, path, object_key])


def dep_id(ecosystem: str, package: str) -> str:
    """``depId``, the repository-neutral node id. ``package`` must already carry §5.15's
    ecosystem normalization."""
    _strs(ecosystem, package)
    return tuple_id(["dependency", ecosystem, package])


def dependency_record_id(ecosystem: str, package: str, manifest_path: str, line: int) -> str:
    """The §5.15 ``dependency`` *record* id: ``depId``'s tuple extended by manifest and line,
    so one manifest may declare a package twice (runtime and dev). Sorts and dedups (§8.3)."""
    _strs(ecosystem, package, manifest_path)
    _ints(line)
    return tuple_id(["dependency", ecosystem, package, manifest_path, line])


def name_key(lang: str, callee_name: str) -> str:
    """``nameKey`` (assembly id, BF14)."""
    _strs(lang, callee_name)
    return tuple_id(["name", lang, callee_name])


def contract_point_id(kind: str, key_norm: str) -> str:
    """``cpId`` (assembly id). ``key_norm`` must already be the §6 ``keyNorm``."""
    _strs(kind, key_norm)
    return tuple_id(["contractPoint", kind, key_norm])


def link_id(kind: str, key_norm: str, provider_anchor_id: str, consumer_anchor_id: str) -> str:
    """``linkId`` (assembly id)."""
    _strs(kind, key_norm, provider_anchor_id, consumer_anchor_id)
    return tuple_id(["link", kind, key_norm, provider_anchor_id, consumer_anchor_id])


def sighting_id(repo_id: str, path: str, commit: str, fingerprint: str) -> str:
    """``sightingId`` — reserved for P5 (§5.19)."""
    _strs(repo_id, path, commit, fingerprint)
    return tuple_id(["secretSighting", repo_id, path, commit, fingerprint])


# --- §5 record ids with no node (sorting and deduplication only) -------------------------

def value_set_id(cs_id: str, arg_path: str) -> str:
    _strs(cs_id, arg_path)
    return tuple_id(["valueSet", cs_id, arg_path])


def data_id(use: str, def_: str, use_path: str, def_path: str) -> str:
    """``data`` id. ``use_path`` is the record's ``toPath``; ``def_path`` is its ``fromPath``."""
    _strs(use, def_, use_path, def_path)
    return tuple_id(["data", use, def_, use_path, def_path])


def controlled_by_id(point: str, cond: str) -> str:
    _strs(point, cond)
    return tuple_id(["controlledBy", point, cond])


def decorator_id(target_id: str, ordinal: int) -> str:
    _strs(target_id)
    _ints(ordinal)
    return tuple_id(["decorator", target_id, ordinal])


def field_access_id(
    pt_id: str, path: str, mode: str, receiver_type: str, field: str
) -> str:
    """``fieldAccess`` id. ``path`` is in the tuple (analyst F1; GV-20/GV-21)."""
    _strs(pt_id, path, mode, receiver_type, field)
    return tuple_id(["fieldAccess", pt_id, path, mode, receiver_type, field])


def import_id(f_id: str, ordinal: int) -> str:
    _strs(f_id)
    _ints(ordinal)
    return tuple_id(["import", f_id, ordinal])


def test_marker_id(sym_id: str, framework: str) -> str:
    _strs(sym_id, framework)
    return tuple_id(["testMarker", sym_id, framework])


# A constructor, not a test: keep pytest from collecting it wherever it is imported by name.
test_marker_id.__test__ = False  # type: ignore[attr-defined]
