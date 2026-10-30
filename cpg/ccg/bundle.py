"""Fact-bundle JSONL: read, validate and write with deterministic order.

Spec: ``docs/plans/connected-code-graph-bundle-spec.md`` v1.3 — §2 (envelope and header), §5
(records), §7 (reason codes), §8 (determinism). The record shapes live in
``ccg/schemas/bundle-v1.schema.json``, the contract; this module adds what JSON Schema cannot
say: line framing and canonical bytes (§8.1), record order and duplicate ids (§8.3), id
recomputation through ``ccg.ids`` (§3.2), in-bundle references (§5.0), and the cross-record
rules of §2.2 and §5.

Public API: ``loads``/``read_bundle`` (parse + validate), ``validate`` (records in file
order), ``dumps``/``write_bundle`` (sort + validate + serialize), ``canonical_line``.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import unicodedata
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import jsonschema

from ccg import ids

SCHEMA_PATH = Path(__file__).parent / "schemas" / "bundle-v1.schema.json"

#: §8.3: the header, then the 19 record types in file order.
RECORD_TYPES: tuple[str, ...] = (
    "header", "file", "symbol", "param", "ret", "cond", "callsite", "valueSet", "constant",
    "data", "controlledBy", "decorator", "fieldAccess", "import", "dependency", "testMarker",
    "configEntry", "launchFact", "schemaEntry", "secretSighting",
)
_ORDER = {t: i for i, t in enumerate(RECORD_TYPES)}


class BundleError(ValueError):
    """A bundle (or a record in it) violates the bundle-v1 spec."""


@cache
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@cache
def _validator() -> jsonschema.protocols.Validator:
    schema = load_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


# --- serialization (§8.1) -------------------------------------------------------------------

def _canon(value: object) -> bytes:
    """§3.1's canonical JSON: sorted keys, ``,``/``:`` with no spaces, non-ASCII literal."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")


def canonical_line(record: dict) -> bytes:
    """One record's line (§8.1): its canonical JSON plus ``\\n``."""
    return _canon(record) + b"\n"


def _sort_key(record: dict) -> tuple[int, str]:
    t = record.get("type")
    if t not in _ORDER:
        raise BundleError(f"unknown record type {t!r}")
    return (_ORDER[t], record.get("id", ""))


def dumps(records: Iterable[dict]) -> bytes:
    """Sort ``records`` into §8.3 order, validate them, and return the bundle's bytes."""
    recs = list(records)
    for r in recs:
        if isinstance(r, dict) and r.get("type") == "secretSighting":
            raise BundleError("v1 writers must not emit secretSighting (§5.19)")
    ordered = sorted(recs, key=_sort_key)
    validate(ordered)
    return b"".join(canonical_line(r) for r in ordered)


def write_bundle(path: str | os.PathLike, records: Iterable[dict]) -> None:
    """Write a bundle file atomically (temp file + rename), deterministic bytes."""
    data = dumps(records)
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


# --- parsing --------------------------------------------------------------------------------

def _no_dup_keys(pairs: list[tuple[str, object]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise BundleError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _no_float(text: str) -> object:
    raise BundleError(f"numbers are integers only (§8.1), got {text}")


def loads(data: bytes) -> list[dict]:
    """Parse and validate a bundle's bytes; return its records in file order."""
    if data.startswith(b"\xef\xbb\xbf"):
        raise BundleError("a bundle has no BOM (§8.1)")
    if b"\r" in data:
        raise BundleError("a bundle has no \\r (§8.1)")
    if not data:
        raise BundleError("empty bundle: line 1 must be the header (§2.1)")
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise BundleError("a bundle ends with exactly one \\n (§8.1)")
    records = []
    for n, raw in enumerate(data[:-1].split(b"\n"), start=1):
        try:
            rec = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_no_dup_keys,
                parse_float=_no_float,
                parse_constant=_no_float,
            )
        except BundleError as e:
            raise BundleError(f"line {n}: {e}") from None
        except ValueError as e:  # UnicodeDecodeError and JSONDecodeError
            raise BundleError(f"line {n}: not UTF-8 JSON: {e}") from None
        if not isinstance(rec, dict):
            raise BundleError(f"line {n}: a record is a JSON object")
        if canonical_line(rec) != raw + b"\n":
            raise BundleError(f"line {n}: not the canonical JSON of its record (§8.1)")
        records.append(rec)
    validate(records)
    return records


def read_bundle(path: str | os.PathLike) -> list[dict]:
    return loads(Path(path).read_bytes())


# --- validation -----------------------------------------------------------------------------

def validate(records: list[dict]) -> None:
    """Validate records given in file order. Raises ``BundleError`` on the first violation."""
    if not records or not isinstance(records[0], dict) or records[0].get("type") != "header":
        raise BundleError("line 1 must be the header record (§2.1)")
    for n, rec in enumerate(records, start=1):
        if n > 1 and isinstance(rec, dict) and rec.get("type") == "header":
            raise BundleError(f"line {n}: only line 1 is a header (§2.1)")
        _check_values(n, rec)
        _check_schema(n, rec)
    _check_order(records)
    _Semantics(records).check()


def _check_schema(n: int, rec: object) -> None:
    err = jsonschema.exceptions.best_match(_validator().iter_errors(rec))
    if err is not None:
        where = "/".join(str(p) for p in err.absolute_path) or "<record>"
        raise BundleError(f"line {n}: schema: {where}: {err.message}")


#: §8.1: identity fields are stripped as well as NFC (every ``*Id`` key counts as one).
_IDENTITY_KEYS = frozenset({"id", "qname", "path", "keyPath", "name", "calleeName"})


def _check_values(n: int, value: object, key: str = "") -> None:
    """§8.1: integers only (JSON Schema's ``integer`` admits ``1.0``), every string NFC,
    identity strings stripped."""
    if isinstance(value, float):
        raise BundleError(f"line {n}: {key}: numbers are integers only (§8.1), got {value!r}")
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise BundleError(f"line {n}: {key}: string is not NFC (§8.1)")
        if (key in _IDENTITY_KEYS or key.endswith("Id")) and value.strip() != value:
            raise BundleError(f"line {n}: {key}: identity field must be stripped (§8.1)")
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_values(n, v, k)
    elif isinstance(value, list):
        for v in value:
            _check_values(n, v, key)


def _check_order(records: list[dict]) -> None:
    """§8.3: the fixed type order, each block by id; equal ids within a type are an error."""
    prev = None
    for n, rec in enumerate(records[1:], start=2):
        key = _sort_key(rec)
        if prev is not None:
            if key == prev:
                raise BundleError(f"line {n}: duplicate {rec['type']} id {rec['id']} (§8.3)")
            if key < prev:
                raise BundleError(f"line {n}: record out of §8.3 order")
        prev = key


# §5.1 access paths, parsed for the cross-record rules.
_IDENT = r"(?:[^\W0-9][\w$]*|\$[\w$]*)"
# Alternation order matters: `recv`/`ret` are tried before `r`.
_HEAD = re.compile(r"a(?:0|[1-9][0-9]*)|p(?:0|[1-9][0-9]*)|recv|ret|r|kw\." + _IDENT)
_FIELD = re.compile(r"\." + _IDENT + r'|\[(?P<q>"(?:[^"\\]|\\.)*")\]')


def _path_head(n: int, path: str, what: str) -> str:
    """The head token of an access path (``a0``, ``p1``, ``recv``, ``kw.x``, ``ret``, ``r``,
    or ``""``), after checking that every quoted field has §5.1's one encoding."""
    if path == "":
        return ""
    m = _HEAD.match(path)
    if not m:
        raise BundleError(f"line {n}: {what}: not an access path: {path!r}")
    head, pos = m.group(0), m.end()
    while pos < len(path):
        f = _FIELD.match(path, pos)
        if not f:
            raise BundleError(f"line {n}: {what}: bad field segment in {path!r}")
        if f.group("q") is not None:
            text = f.group("q")
            key = json.loads(text)
            if re.fullmatch(_IDENT, key) or json.dumps(key, ensure_ascii=False) != text:
                raise BundleError(
                    f"line {n}: {what}: field {text} must be quoted only when it is not an "
                    "identifier, with canonical JSON escaping (§5.1)")
        pos = f.end()
    return head


def _use_kind(head: str) -> str:
    """§5.10: the point kind a value lands on, from a use path's head."""
    return {"": "cond", "ret": "ret"}.get(head, "callsite")


def _def_kind(head: str) -> tuple[str, int | None]:
    """§5.10: the point kind (and param index) a value comes from, from a def path's head."""
    if head == "r":
        return "callsite", None
    if head == "recv":
        return "param", -1
    return "param", int(head[1:])


def _strictly_sorted(keys: list) -> bool:
    return all(a < b for a, b in zip(keys, keys[1:]))


def _pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _bind_ip_literal(value: str) -> str | None:
    """The part of a ``bindValue`` the schema admitted by IPv6 *shape* only: a bare value with
    a ``:`` (ports and DNS names have none), or a URL's ``[…]`` host. ``None`` otherwise."""
    if "://" in value:
        host = value.split("://", 1)[1]
        return host[1:host.index("]")] if host.startswith("[") else None
    return value if ":" in value else None


def _is_ipv6(text: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(text), ipaddress.IPv6Address)
    except ValueError:
        return False


_POINT_TYPES = ("param", "ret", "cond", "callsite")


class _Semantics:
    """The cross-record rules of §2.2, §3.2 and §5 over one ordered, schema-valid bundle."""

    def __init__(self, records: list[dict]):
        self.header = records[0]
        self.lines = list(enumerate(records, start=1))
        self.by_type: dict[str, dict[str, tuple[int, dict]]] = {t: {} for t in RECORD_TYPES}
        for n, r in self.lines[1:]:
            self.by_type[r["type"]][r["id"]] = (n, r)
        unit = self.header["unit"]
        self.repo, self.root = unit["repoId"], unit["rootPath"]
        self.vendored = set(self.header["vendoredDirs"])

    def check(self) -> None:
        self._header()
        # Pass 1: each record's own repoId and id, so a bad id is reported as itself and not as a
        # dangling reference from some later-checked record.
        for n, r in self.lines[1:]:
            if "repoId" in r:
                self._same_repo(n, r)
            fn = self._ID_FNS.get(r["type"])
            if fn:
                self._id(n, r, lambda: fn(self, r))
        # Pass 2: references and the cross-record rules.
        for n, r in self.lines[1:]:
            getattr(self, "_" + r["type"])(n, r)
            for field in ("path", "manifestPath", "resolvedPath"):
                if r["type"] != "secretSighting" and field in r and self._is_vendored(r[field]):
                    raise BundleError(
                        f"line {n}: {field} {r[field]!r} is vendored: a vendored file yields "
                        "no record (§5.0)")
        self._ordinals("import", lambda r: r["fileId"],
                       lambda key, k: ids.import_id(key, k))
        self._ordinals("launchFact", lambda r: (r["path"], r["kind"]),
                       lambda key, k: ids.launch_fact_id(self.repo, key[0], key[1], k))

    #: §3.2/§5 tuples, through ``ccg.ids``. ``constant`` needs its file's lang (pass 2);
    #: ``import``/``launchFact`` hold an ordinal no field carries (``_ordinals``).
    _ID_FNS = {
        "file": lambda s, r: ids.file_id(r["repoId"], r["path"]),
        "symbol": lambda s, r: ids.symbol_id(r["lang"], r["repoId"], r["qname"],
                                             r.get("signature")),
        "param": lambda s, r: ids.point_id(r["symbolId"], "param", r["index"]),
        "ret": lambda s, r: ids.point_id(r["symbolId"], "ret", r["ordinal"]),
        "cond": lambda s, r: ids.point_id(r["symbolId"], "cond", r["ordinal"]),
        "callsite": lambda s, r: ids.callsite_id(r["symbolId"], r["calleeName"], r["ordinal"]),
        "valueSet": lambda s, r: ids.value_set_id(r["callSiteId"], r["arg"]),
        "data": lambda s, r: ids.data_id(r["use"], r["def"], r["toPath"], r["fromPath"]),
        "controlledBy": lambda s, r: ids.controlled_by_id(r["point"], r["cond"]),
        "decorator": lambda s, r: ids.decorator_id(r["targetId"], r["ordinal"]),
        "fieldAccess": lambda s, r: ids.field_access_id(r["pointId"], r["path"], r["mode"],
                                                        r["receiverType"], r["field"]),
        "dependency": lambda s, r: ids.dependency_record_id(r["ecosystem"], r["package"],
                                                            r["manifestPath"], r["line"]),
        "testMarker": lambda s, r: ids.test_marker_id(r["symbolId"], r["framework"]),
        "configEntry": lambda s, r: ids.config_entry_id(r["repoId"], r["path"], r["keyPath"]),
        "schemaEntry": lambda s, r: ids.schema_entry_id(s.repo, r["path"], r["objectKey"]),
        "secretSighting": lambda s, r: ids.sighting_id(s.repo, r["path"], r["commit"],
                                                       r["fingerprint"]),
    }

    # -- helpers --
    def _id(self, n: int, r: dict, expected_fn) -> None:
        try:
            expected = expected_fn()
        except (TypeError, ValueError) as e:
            raise BundleError(f"line {n}: {r['type']} id cannot be computed: {e}") from None
        if r["id"] != expected:
            raise BundleError(
                f"line {n}: {r['type']} id {r['id']} does not match its fields "
                f"(expected {expected}, §3.2)")

    def _ref(self, n: int, r: dict, field: str, *types: str) -> dict:
        target = r[field]
        for t in types:
            if target in self.by_type[t]:
                return self.by_type[t][target][1]
        raise BundleError(
            f"line {n}: {field} {target} names no {'/'.join(types)} record in this bundle (§5.0)")

    def _same_repo(self, n: int, r: dict) -> None:
        if r["repoId"] != self.repo:
            raise BundleError(f"line {n}: repoId {r['repoId']!r} is not the unit's {self.repo!r}")

    def _point(self, n: int, r: dict, field: str, kind: str, index: int | None = None) -> dict:
        p = self._ref(n, r, field, *_POINT_TYPES)
        if p["type"] != kind or (index is not None and p["index"] != index):
            want = kind if index is None else f"{kind} index {index}"
            raise BundleError(
                f"line {n}: {field} is a {p['type']}, but its path names a {want} (§5.1, §5.10)")
        return p

    def _sorted_values(self, n: int, values: list, what: str) -> None:
        if not _strictly_sorted([_canon(v) for v in values]):
            raise BundleError(f"line {n}: {what} must be sorted by canonical JSON, no repeats "
                              "(§5.8)")

    def _is_vendored(self, path: str) -> bool:
        if not self.vendored:
            return False
        rel = path
        if self.root != "." and path.startswith(self.root + "/"):
            rel = path[len(self.root) + 1:]
        return any(seg in self.vendored for seg in rel.split("/")[:-1])

    def _ordinals(self, rtype: str, group_key, id_fn) -> None:
        """Ids whose tuple holds a source-order ordinal no field carries (§5.14, §5.18): each
        group's ids must be exactly ordinals 0..k-1, in non-decreasing line order."""
        groups: dict = {}
        for i, (n, r) in self.by_type[rtype].items():
            groups.setdefault(group_key(r), []).append((n, r))
        for key, members in groups.items():
            by_id = {id_fn(key, k): k for k in range(len(members))}
            ranked = []
            for n, r in members:
                if r["id"] not in by_id:
                    raise BundleError(
                        f"line {n}: {rtype} id {r['id']} is not ordinal 0..{len(members) - 1} "
                        "of its group (§3.2)")
                ranked.append((by_id[r["id"]], r["line"], n))
            ranked.sort()
            for (_, a, _), (_, b, n) in zip(ranked, ranked[1:]):
                if b < a:
                    raise BundleError(f"line {n}: {rtype} ordinals must follow source order")

    # -- header (§2.2) --
    def _header(self) -> None:
        h = self.header
        u = h["unit"]
        expected = ids.unit_id(u["repoId"], u["rootPath"], u["lang"])
        if h["unitId"] != expected:
            raise BundleError(f"line 1: unitId does not match unit (expected {expected})")
        actual = {t: len(self.by_type[t]) for t in RECORD_TYPES[1:]}
        if h["counts"] != actual:
            diff = {t: (h["counts"][t], actual[t]) for t in actual if h["counts"][t] != actual[t]}
            raise BundleError(f"line 1: counts do not match the records (header, actual): {diff}")
        if h["vendoredDirs"] != sorted(h["vendoredDirs"]):
            raise BundleError("line 1: vendoredDirs must be sorted (§2.2)")
        if h["history"] != "available":
            raise BundleError("line 1: history must be 'available' in P0 (§2.2)")

    # -- records (§5) --
    def _file(self, n, r):
        if r["unitId"] != self.header["unitId"]:
            raise BundleError(f"line {n}: unitId is not the header's")

    def _symbol(self, n, r):
        f = self._ref(n, r, "fileId", "file")
        if f["path"] != r["path"]:
            raise BundleError(f"line {n}: path {r['path']!r} is not its file's {f['path']!r}")
        if f["lang"] != r["lang"]:
            raise BundleError(f"line {n}: lang {r['lang']!r} is not its file's {f['lang']!r}")
        if not r["qname"].startswith(r["path"] + "::"):
            raise BundleError(f"line {n}: qname must start with path + '::' (§4.1)")
        if "parentId" in r:
            self._ref(n, r, "parentId", "symbol")
        if r["lineStart"] > r["lineEnd"]:
            raise BundleError(f"line {n}: lineStart is after lineEnd")
        regions = r.get("ownedRegions", [])
        if not _strictly_sorted(regions) or any(a > b for a, b in regions):
            raise BundleError(f"line {n}: ownedRegions must be sorted [start, end] pairs (§5.3)")
        if "lastChangedAt" not in r:
            raise BundleError(f"line {n}: lastChangedAt is required when history is "
                              "available (§5.3)")

    def _param(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _ret(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _cond(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _callsite(self, n, r):
        self._ref(n, r, "symbolId", "symbol")
        targets = r["targets"]
        if not _strictly_sorted([t["symbolId"] for t in targets]):
            raise BundleError(f"line {n}: targets must be sorted by symbolId (§5.6)")
        # Not checked: a target's path against vendoredDirs (§5.6 "never vendored code").
        # §5.0's test is relative to the *defining* unit's rootPath (P3-4), which a targets
        # entry does not carry; one path can be vendored for unit "." and first-party for a
        # unit rooted at third_party/ourlib/. An in-bundle target is covered by its own
        # symbol/file record's path check; an out-of-bundle one is the assembler's to judge.
        for t in targets:
            hit = self.by_type["symbol"].get(t["symbolId"])
            if hit and (hit[1]["qname"], hit[1].get("signature")) != (t["qname"],
                                                                      t.get("signature")):
                raise BundleError(f"line {n}: targets entry disagrees with symbol "
                                  f"{t['symbolId']}")

    def _valueSet(self, n, r):
        self._ref(n, r, "callSiteId", "callsite")
        _path_head(n, r["arg"], "arg")
        self._sorted_values(n, r["values"], "values")

    def _constant(self, n, r):
        f = self._ref(n, r, "fileId", "file")
        self._sorted_values(n, r["values"], "values")
        self._id(n, r, lambda: ids.constant_id(f["lang"], self.repo, r["qname"]))

    def _data(self, n, r):
        use = self._point(n, r, "use", _use_kind(_path_head(n, r["toPath"], "toPath")))
        kind, index = _def_kind(_path_head(n, r["fromPath"], "fromPath"))
        def_ = self._point(n, r, "def", kind, index)
        if use["symbolId"] != def_["symbolId"]:
            raise BundleError(f"line {n}: use and def must be in the same symbol (§5.10)")

    def _controlledBy(self, n, r):
        point = self._ref(n, r, "point", *_POINT_TYPES)
        cond = self._point(n, r, "cond", "cond")
        if point["symbolId"] != cond["symbolId"]:
            raise BundleError(f"line {n}: point and cond must be in the same symbol (§5.11)")

    def _decorator(self, n, r):
        self._ref(n, r, "targetId", "symbol")
        args = r["args"]
        if not _strictly_sorted([a["arg"] for a in args]):
            raise BundleError(f"line {n}: args must be sorted by arg, no repeats (§5.12)")
        for a in args:
            _path_head(n, a["arg"], "args.arg")
            self._sorted_values(n, a["values"], "args.values")

    def _fieldAccess(self, n, r):
        head = _path_head(n, r["path"], "path")
        if r["mode"] == "read":
            self._point(n, r, "pointId", _use_kind(head))
        else:
            self._point(n, r, "pointId", *_def_kind(head))

    def _import(self, n, r):
        self._ref(n, r, "fileId", "file")
        if "symbolId" in r:
            self._ref(n, r, "symbolId", "symbol")
        keys = [(x["name"], 0, "") if "alias" not in x else (x["name"], 1, x["alias"])
                for x in r["names"]]
        if not _strictly_sorted(keys):
            raise BundleError(f"line {n}: names must be sorted by (name, alias), an absent "
                              "alias first (§5.14)")

    def _dependency(self, n, r):
        eco, pkg = r["ecosystem"], r["package"]
        if (eco == "PyPI" and pkg != _pep503(pkg)) or (eco == "NuGet" and pkg != pkg.lower()):
            raise BundleError(f"line {n}: package {pkg!r} is not {eco}-normalized (§5.15)")
        expected = ids.dep_id(eco, pkg)
        if r["depId"] != expected:
            raise BundleError(f"line {n}: depId does not match (expected {expected})")

    def _testMarker(self, n, r):
        self._ref(n, r, "symbolId", "symbol")

    def _configEntry(self, n, r):
        ip = _bind_ip_literal(r.get("bindValue", ""))
        if ip is not None and not _is_ipv6(ip):
            raise BundleError(f"line {n}: bindValue {r['bindValue']!r}: {ip!r} is not an IP "
                              "literal (§5.17 (a))")
        f = self.by_type["file"].get(ids.file_id(self.repo, r["path"]))
        if f and r["example"] != (f[1]["role"] == "example"):
            raise BundleError(f"line {n}: example must be copied from its file's role (§5.17)")

    def _launchFact(self, n, r):
        pass  # id checked by _ordinals

    def _schemaEntry(self, n, r):
        pass  # id only (pass 1)

    def _secretSighting(self, n, r):
        pass  # reserved (P5): shape and id only; P5 fixes its enclosing end's reference rule
