"""Tests for ``ccg.bundle`` and ``ccg/schemas/bundle-v1.schema.json``.

Spec: ``docs/plans/connected-code-graph-bundle-spec3.md`` v1.1 — §2 (envelope, header), §3 (ids),
§5 (records), §7 (reason codes), §8 (determinism). P0 plan
``docs/plans/connected-code-graph-p0.md`` §9.1, row ``ccg/bundle.py + bundle-v1 (S10)``.

Fixture ids are built with ``ccg.ids`` (the S09 constructors, pinned by the §3.6 golden vectors);
the expected *field sets and tokens* of the schema are read from the spec text itself, never
from the schema or ``ccg.bundle``.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from ccg import bundle, ids

REPO = "graphmind-ai-lab"
ROOT = "salesperson"
LANG = "javascript"
UNIT_ID = ids.unit_id(REPO, ROOT, LANG)
H64 = hashlib.sha256(b"x").hexdigest()
COMMIT = "1" * 40
TREE = "2" * 40
AUDIT = "0f8fad5b-d9cb-469f-a165-70867728950e"


def zero_counts() -> dict[str, int]:
    return {t: 0 for t in bundle.RECORD_TYPES[1:]}


def header(**over) -> dict:
    h = {
        "type": "header",
        "bundleSchemaVersion": "1.0",
        "unit": {"repoId": REPO, "rootPath": ROOT, "lang": LANG},
        "unitId": UNIT_ID,
        "builtCommit": COMMIT,
        "builtTree": TREE,
        "builtAt": "2026-10-29T12:00:00Z",
        "tools": {
            "joern": "4.0.579",
            "frontend": "jssrc2cpg",
            "frontendVersion": "4.0.579",
            "normalizer": "normalize.py",
            "normalizerVersion": "0.0.1",
            "extractScript": "extract.sc",
            "extractScriptHash": H64,
        },
        "sandboxAuditIds": [AUDIT],
        "egressMode": "offline",
        "flowProjected": True,
        "fieldDepthCap": 3,
        "vendoredDirs": [],
        "history": "available",
        "counts": zero_counts(),
    }
    h.update(over)
    return h


def to_bytes(records: list[dict]) -> bytes:
    """Serialize records exactly as §8.1 says, independently of ``ccg.bundle``."""
    return b"".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for r in records
    )


# --- 1. a minimal valid bundle validates -------------------------------------------------

def test_minimal_header_only_bundle_validates():
    """A header with every count 0 is a valid v1 bundle: spec v1.3 states no minimum content
    (§2.2 ``counts`` is a truncation check and admits zero). This is risk R11's shape — an
    empty CPG from a ``joern-parse`` that exited 0 — so the guard is S11's content check, not
    the schema. Pinned here so a spec successor that adds a rule turns this test red."""
    records = bundle.loads(to_bytes([header()]))
    assert records == [header()]


def test_minimal_bundle_writes_byte_identically_to_spec_serialization(tmp_path):
    out = tmp_path / f"{UNIT_ID}.bundle.jsonl"
    bundle.write_bundle(out, [header()])
    assert out.read_bytes() == to_bytes([header()])
    assert bundle.read_bundle(out) == [header()]


# --- 2. the header must come first (§2.1) -------------------------------------------------

def test_empty_bundle_is_rejected():
    with pytest.raises(bundle.BundleError, match="header"):
        bundle.loads(b"")


def test_bundle_without_header_first_is_rejected():
    rec = file_rec()
    h = header(counts={**zero_counts(), "file": 1})
    with pytest.raises(bundle.BundleError, match="header"):
        bundle.validate([rec, h])


def test_second_header_is_rejected():
    with pytest.raises(bundle.BundleError, match="header"):
        bundle.validate([header(), header()])


# --- record builders ------------------------------------------------------------------------

P_CLIENT = "salesperson/src/api/client.ts"
F_CLIENT = ids.file_id(REPO, P_CLIENT)


def file_rec(**over) -> dict:
    r = {
        "type": "file",
        "id": F_CLIENT,
        "repoId": REPO,
        "unitId": UNIT_ID,
        "path": P_CLIENT,
        "lang": LANG,
        "kind": "code",
        "role": "primary",
        "contentHash": H64,
        "lineCount": 140,
    }
    r.update(over)
    return r


P_ERRORS = "salesperson/src/api/errors.ts"
P_PKG = "salesperson/package.json"
P_SQL = "salesperson/schema.sql"
F_PKG = ids.file_id(REPO, P_PKG)
S_MOD = ids.symbol_id(LANG, REPO, P_CLIENT + "::<module>")
S_API = ids.symbol_id(LANG, REPO, P_CLIENT + "::apiFetch")
S_URL = ids.symbol_id(LANG, REPO, P_CLIENT + "::buildUrl")
PARAM_API = ids.point_id(S_API, "param", 0)
PARAM_URL = ids.point_id(S_URL, "param", 0)
RET_URL = ids.point_id(S_URL, "ret", 0)
COND_API = ids.point_id(S_API, "cond", 0)
CS_FETCH = ids.callsite_id(S_API, "fetch", 0)
CS_URL = ids.callsite_id(S_API, "buildUrl", 0)


def symbol_rec(sid, qname, kind, name, start, end, parent=None, **over) -> dict:
    r = {
        "type": "symbol", "id": sid, "lang": LANG, "repoId": REPO, "qname": qname,
        "fileId": F_CLIENT, "path": P_CLIENT, "kind": kind, "name": name,
        "nameNorm": name.casefold(), "lineStart": start, "lineEnd": end,
        "fingerprint": H64, "complexity": 1,
        "lastChangedAt": "2026-10-01T08:00:00Z", "lastChangedCommit": COMMIT,
    }
    if parent:
        r["parentId"] = parent
    r.update(over)
    return r


def rich_records() -> list[dict]:
    """One or more valid records of every non-reserved type, cross-referenced (§5)."""
    recs = [
        file_rec(),
        file_rec(id=F_PKG, path=P_PKG, lang="config", kind="config", lineCount=30),
        symbol_rec(S_MOD, P_CLIENT + "::<module>", "module", "<module>", 1, 140),
        symbol_rec(S_API, P_CLIENT + "::apiFetch", "function", "apiFetch", 107, 130, S_MOD,
                   complexity=3),
        symbol_rec(S_URL, P_CLIENT + "::buildUrl", "function", "buildUrl", 73, 89, S_MOD),
        {"type": "param", "id": PARAM_API, "symbolId": S_API, "index": 0, "name": "opts",
         "line": 107},
        {"type": "param", "id": PARAM_URL, "symbolId": S_URL, "index": 0, "name": "path",
         "line": 73},
        {"type": "ret", "id": RET_URL, "symbolId": S_URL, "ordinal": 0, "line": 88},
        {"type": "cond", "id": COND_API, "symbolId": S_API, "ordinal": 0, "line": 115,
         "code": "!res.ok"},
        {"type": "callsite", "id": CS_FETCH, "symbolId": S_API, "calleeName": "fetch",
         "ordinal": 0, "callKind": "call", "line": 112, "resolution": "external",
         "targets": [], "externalTarget": "fetch", "status": "possible",
         "reason": "name-match"},
        {"type": "callsite", "id": CS_URL, "symbolId": S_API, "calleeName": "buildUrl",
         "ordinal": 0, "callKind": "call", "line": 110, "resolution": "resolved",
         "targets": [{"symbolId": S_URL, "qname": P_CLIENT + "::buildUrl"}],
         "status": "confirmed", "reason": "resolved-call"},
        {"type": "valueSet", "id": ids.value_set_id(CS_FETCH, "a1.method"),
         "callSiteId": CS_FETCH, "arg": "a1.method",
         "values": [{"const": "GET"}, {"unknown": "point-derived"}]},
        {"type": "constant", "id": ids.constant_id(LANG, REPO, P_CLIENT + "::API_PREFIX"),
         "qname": P_CLIENT + "::API_PREFIX", "fileId": F_CLIENT, "line": 8,
         "values": [{"template": [{"const": "/shop"}, {"ref": P_CLIENT + "::BASE"}]}]},
        {"type": "data", "id": ids.data_id(CS_FETCH, CS_URL, "a0", "r"), "use": CS_FETCH,
         "def": CS_URL, "toPath": "a0", "fromPath": "r", "status": "confirmed",
         "reason": "intra-flow"},
        {"type": "data", "id": ids.data_id(CS_URL, PARAM_API, "a0", "p0.path"),
         "use": CS_URL, "def": PARAM_API, "toPath": "a0", "fromPath": "p0.path",
         "status": "confirmed", "reason": "intra-flow"},
        {"type": "data", "id": ids.data_id(RET_URL, PARAM_URL, "ret", "p0"), "use": RET_URL,
         "def": PARAM_URL, "toPath": "ret", "fromPath": "p0", "status": "confirmed",
         "reason": "intra-flow"},
        {"type": "data", "id": ids.data_id(COND_API, PARAM_API, "", "p0"), "use": COND_API,
         "def": PARAM_API, "toPath": "", "fromPath": "p0", "status": "confirmed",
         "reason": "intra-flow"},
        {"type": "controlledBy", "id": ids.controlled_by_id(CS_FETCH, COND_API),
         "point": CS_FETCH, "cond": COND_API},
        {"type": "decorator", "id": ids.decorator_id(S_API, 0), "targetId": S_API,
         "ordinal": 0, "form": "decorator", "name": "memo", "line": 106,
         "args": [{"arg": "a0", "values": [{"const": "x"}]}]},
        {"type": "fieldAccess",
         "id": ids.field_access_id(CS_FETCH, "a0", "read", "?", "url"),
         "pointId": CS_FETCH, "path": "a0", "mode": "read", "receiverType": "?",
         "field": "url", "line": 112, "status": "possible", "reason": "field-name-match"},
        {"type": "import", "id": ids.import_id(F_CLIENT, 0), "fileId": F_CLIENT, "line": 1,
         "ordinal": 0, "module": "./errors", "names": [{"name": "ApiError"}, {"alias": "E", "name": "Err"}],
         "resolvedPath": P_ERRORS, "dynamic": False},
        {"type": "dependency",
         "id": ids.dependency_record_id("npm", "react", P_PKG, 12),
         "depId": ids.dep_id("npm", "react"), "ecosystem": "npm", "package": "react",
         "version": "^18.3.1", "scope": "runtime", "manifestPath": P_PKG, "line": 12},
        {"type": "testMarker", "id": ids.test_marker_id(S_API, "vitest"), "symbolId": S_API,
         "framework": "vitest", "kindHint": "unit", "evidence": "name"},
        {"type": "configEntry", "id": ids.config_entry_id(REPO, P_PKG, "server.port"),
         "repoId": REPO, "path": P_PKG, "line": 6, "format": "json",
         "keyPath": "server.port", "valueHash": H64, "bindValue": "5173", "example": False},
        {"type": "launchFact", "id": ids.launch_fact_id(REPO, P_PKG, "command", 0),
         "path": P_PKG, "line": 7, "ordinal": 0, "kind": "command", "command": "vite",
         "args": ["--host", "0.0.0.0"], "origin": "explicit"},
        {"type": "schemaEntry", "id": ids.schema_entry_id(REPO, P_SQL, "sql:table:orders"),
         "path": P_SQL, "line": 1, "objectKey": "sql:table:orders", "op": "create"},
    ]
    return recs


def counted(records: list[dict], **hdr) -> list[dict]:
    counts = zero_counts()
    for r in records:
        counts[r["type"]] += 1
    return [header(counts=counts, **hdr), *records]


def ordered(records: list[dict]) -> list[dict]:
    """§8.3 order, computed independently: the spec's type list, then id."""
    return sorted(records, key=lambda r: (SPEC_ORDER.index(r["type"]), r.get("id", "")))


def rich_bundle() -> list[dict]:
    return ordered(counted(rich_records()))


def test_rich_bundle_covers_every_non_reserved_record_type():
    assert {r["type"] for r in rich_bundle()} == set(SPEC_ORDER) - {"secretSighting"}


def test_rich_bundle_validates():
    bundle.validate(rich_bundle())
    assert bundle.loads(to_bytes(rich_bundle())) == rich_bundle()


# --- 5. shuffled records write byte-identically; 6. read -> write round-trips --------------

@pytest.mark.parametrize("seed", range(5))
def test_shuffled_records_write_byte_identical(seed, tmp_path):
    import random

    recs = rich_bundle()
    random.Random(seed).shuffle(recs)
    out = tmp_path / "b.jsonl"
    bundle.write_bundle(out, recs)
    assert out.read_bytes() == to_bytes(rich_bundle())


def test_read_then_write_round_trips_byte_identically(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_bytes(to_bytes(rich_bundle()))
    dst = tmp_path / "out.jsonl"
    bundle.write_bundle(dst, bundle.read_bundle(src))
    assert dst.read_bytes() == src.read_bytes()


def test_write_leaves_no_temp_file(tmp_path):
    bundle.write_bundle(tmp_path / "b.jsonl", rich_bundle())
    assert [p.name for p in tmp_path.iterdir()] == ["b.jsonl"]


# --- spec text: tables parsed from the spec document --------------------------------------

import re  # noqa: E402
from pathlib import Path  # noqa: E402

_PLANS = Path(__file__).resolve().parents[2] / "docs/plans"
#: One source: spec3 v1.1, the consolidation of the bundle spec and spec2.
SPEC_TEXT = (_PLANS / "connected-code-graph-bundle-spec3.md").read_text(encoding="utf-8")
#: spec4 amends spec3 (read together, spec4 §0): its one schema change is header.skippedFiles.
SPEC4_TEXT = (_PLANS / "connected-code-graph-bundle-spec4.md").read_text(encoding="utf-8")


#: spec5 extends spec3/spec4 (F-7): `configEntry.bindParts` (§5.1), `header.bindKeyCheck` (§5.2), `bundleSchemaVersion` in {"1.0","1.1"} (§1).
SPEC5_TEXT = (_PLANS / "connected-code-graph-bundle-spec5.md").read_text(encoding="utf-8")


def _section(heading_re: str, text: str | None = None) -> str:
    """The text of the first section of ``text`` (default: the spec) whose heading matches."""
    text = SPEC_TEXT if text is None else text
    m = re.search(rf"^(#+) {heading_re}.*$", text, re.M)
    assert m, heading_re
    level = len(m.group(1))
    rest = text[m.end():]
    end = re.search(rf"^#{{1,{level}}} ", rest, re.M)
    return rest[: end.start()] if end else rest


def _table_rows(text: str) -> list[list[str]]:
    """The body rows of the **first** table in ``text`` (a section may hold several, e.g.
    BS2 §1.6 holds the field table and, in §1.6.1, the AV table)."""
    rows, started = [], False
    for line in text.splitlines():
        if line.startswith("|"):
            started = True
            if not re.match(r"^\|[-| ]+\|$", line):
                rows.append([c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]])
        elif started:
            break
    return rows[1:]  # drop the column-header row


def _ticks(cell: str) -> list[str]:
    return [t.strip('"') for t in re.findall(r"`([^`]+)`", cell)]


# §8.3's list, read from the spec.
SPEC_ORDER = ["header"] + _ticks(
    re.search(r"each block sorted by `id`.*?\n\n(.*?)\n\n", _section(r"8\.3 "), re.S).group(1)
)
# §3.3's lang tokens.
SPEC_LANGS = [_ticks(r[1])[0] for r in _table_rows(_section(r"3\.3 "))]

# Table sections: record type -> heading regex (spec), or (heading regex, SPEC_TEXT) where
# BS2 replaces the table: §1.5 replaces spec §5.14's, §1.6 spec §5.18's launchFact list.
TABLE_SECTIONS = {
    "header": r"2\.2 ", "file": r"5\.2 ", "symbol": r"5\.3 ", "param": r"5\.4 ",
    "callsite": r"5\.6 ", "valueSet": r"5\.8 ", "constant": r"5\.9 ", "data": r"5\.10 ",
    "decorator": r"5\.12 ", "fieldAccess": r"5\.13 ", "import": (r"1\.5 ", SPEC_TEXT),
    "dependency": r"5\.15 ", "configEntry": r"5\.17 ",
    "launchFact": (r"1\.6 ", SPEC_TEXT),
}


def _table_section(rtype: str) -> str:
    where = TABLE_SECTIONS[rtype]
    return _section(*where) if isinstance(where, tuple) else _section(where)

# Record types whose fields the spec gives in prose, transcribed here with citations.
# (type -> (section regex, required fields, optional fields, {field: enum tokens}))
PROSE_RECORDS = {
    # §5.5: "Fields: `symbolId`, `ordinal`, `line`."
    "ret": (r"5\.5 ", ["symbolId", "ordinal", "line"], [], {}),
    # §5.5: "Fields: `symbolId`, `ordinal`, `line`, `code`."
    "cond": (r"5\.5 ", ["symbolId", "ordinal", "line", "code"], [], {}),
    # BS2 §1.4 replaces §5.11's field sentence: "`point` is any point …, `cond` is a `cond`
    # point, and `point ≠ cond`."
    "controlledBy": ((r"1\.4 ", SPEC_TEXT), ["point", "cond"], [], {}),
    # §5.16: symbolId; framework (open list, "…"); kindHint ∈ {…}; evidence ∈ {…}.
    "testMarker": (r"5\.16 ", ["symbolId", "framework", "kindHint", "evidence"], [],
                   {"kindHint": ["unit", "integration", "e2e", "unknown"],
                    "evidence": ["decorator", "name", "path", "base-class", "config"]}),
    # §5.18 schemaEntry: path, line, objectKey, migration ("when there is one"), op ∈ {…}.
    "schemaEntry": (r"5\.18 ", ["path", "line", "objectKey", "op"], ["migration"],
                    {"op": ["create", "alter", "drop", "rename"]}),
    # §5.19: fingerprint; path, line, commit, committedAt; removedInCommit (optional);
    # presentAtHead; `symbolId` or `entryId`.
    "secretSighting": (r"5\.19 ", ["fingerprint", "path", "line", "commit", "committedAt",
                                   "presentAtHead"],
                       ["removedInCommit", "symbolId", "entryId"], {}),
}


def _spec_fields(rtype: str) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """(required, optional, enums) for one record type, from the spec text."""
    if rtype in PROSE_RECORDS:
        sec, req, opt, enums = PROSE_RECORDS[rtype]
        return set(req) | {"type", "id"}, set(opt), enums
    req, opt, enums = set(), set(), {}
    for cells in _table_rows(_table_section(rtype)):
        names = _ticks(cells[0])
        (req if cells[2].startswith("yes") else opt).update(names)
        tcell = cells[1]
        if "\\|" in tcell:
            for n in names:
                enums[n] = _ticks(tcell)
        elif tcell.startswith("OSV ecosystem name"):
            enums[names[0]] = _ticks(tcell)
        elif tcell == "§3.3 token":
            enums[names[0]] = SPEC_LANGS
    if rtype == "header":  # spec4 §1 adds one row to spec3 §2.2's table (the v1 freeze text); spec5 §5.2 another
        for cells in _table_rows(_section(r"1\. Rule 4", SPEC4_TEXT)) + _table_rows(_section(r"5\.2 ", SPEC5_TEXT)):
            (req if cells[2].startswith("yes") else opt).update(_ticks(cells[0]))
    else:
        req |= {"type", "id"}  # §5.0: every record carries both
        if rtype == "configEntry":  # spec5 §5.1 adds one optional row
            for cells in _table_rows(_section(r"5\.1 ", SPEC5_TEXT)):
                (req if cells[2].startswith("yes") else opt).update(_ticks(cells[0]))
    return req, opt, enums


def _prose_section(sec) -> str:
    return _section(*sec) if isinstance(sec, tuple) else _section(sec)


def test_prose_transcriptions_appear_in_their_spec_sections():
    for rtype, (sec, req, opt, enums) in PROSE_RECORDS.items():
        text = _prose_section(sec)
        for name in [*req, *opt, *[t for ts in enums.values() for t in ts]]:
            assert f"`{name}`" in text, (rtype, name)


def test_spec_order_is_header_plus_nineteen_types():
    assert len(SPEC_ORDER) == 20 and SPEC_ORDER[-1] == "secretSighting"
    assert set(TABLE_SECTIONS) | set(PROSE_RECORDS) == set(SPEC_ORDER)


def _resolve(node: dict) -> dict:
    schema = bundle.load_schema()
    while "$ref" in node:
        node = schema["$defs"][node["$ref"].split("/")[-1]]
    return node


def _def(rtype: str) -> dict:
    return bundle.load_schema()["$defs"][rtype]


def test_schema_record_type_enum_is_the_spec_order():
    assert bundle.load_schema()["properties"]["type"]["enum"] == SPEC_ORDER
    assert list(bundle.RECORD_TYPES) == SPEC_ORDER


@pytest.mark.parametrize("rtype", SPEC_ORDER)
def test_schema_fields_match_spec(rtype):
    req, opt, _ = _spec_fields(rtype)
    d = _def(rtype)
    assert d["additionalProperties"] is False
    assert set(d["properties"]) == req | opt
    assert set(d["required"]) == req


@pytest.mark.parametrize("rtype", SPEC_ORDER)
def test_schema_enums_match_spec(rtype):
    _, _, enums = _spec_fields(rtype)
    props = _def(rtype)["properties"]
    for field, tokens in enums.items():
        assert sorted(_resolve(props[field])["enum"]) == sorted(tokens), field


def test_header_constants_match_spec():
    rows = {(_ticks(c[0]) or [""])[0]: c for c in _table_rows(_section(r"2\.2 "))}
    props = _def("header")["properties"]
    assert _ticks(rows["bundleSchemaVersion"][1])[0] == "1.0"                 # spec3 §2.2's row: the v1.0 freeze text
    minors = re.search(r"`bundleSchemaVersion` is `\"(1\.\d)\"` or `\"(1\.\d)\"`", _section(r"1\. Version rule", SPEC5_TEXT))
    assert props["bundleSchemaVersion"]["enum"] == list(minors.groups()) == ["1.0", "1.1"]     # spec5 §1
    assert str(props["fieldDepthCap"]["const"]) == _ticks(rows["fieldDepthCap"][1])[0]
    assert set(_resolve(props["counts"])["required"]) == set(SPEC_ORDER[1:])


# §7.2's "On" column, mapped to the bundle record that carries each graph element.
ON_TO_RECORD = {"callsite": "callsite", "DATA": "data", "shell DATA": "data",
                "fieldAccess": "fieldAccess", "field READS": "fieldAccess"}


def _spec_bundle_reasons() -> dict[str, set[str]]:
    """§7.2 rows with Bundle = yes: code -> the record types its On column names."""
    out = {}
    for c in _table_rows(_section(r"7\.2 ")):
        if c[6] == "yes":
            cell = c[2].replace("`", "")
            on = {ON_TO_RECORD[k] for k in ON_TO_RECORD
                  if re.search(rf"\b{re.escape(k)}\b", cell)}
            out[_ticks(c[0])[0]] = on
    return out


def test_status_reasons_match_spec_registry():
    spec = _spec_bundle_reasons()
    assert len(spec) == 8
    got: dict[str, set[str]] = {}
    for rtype in SPEC_ORDER:
        reason = _def(rtype)["properties"].get("reason")
        for code in (_resolve(reason)["enum"] if reason else []):
            got.setdefault(code, set()).add(rtype)
    assert got == spec


def test_value_unknown_reasons_match_spec_7_3():
    spec = [_ticks(c[0])[0] for c in _table_rows(_section(r"7\.3 "))]
    unknown = bundle.load_schema()["$defs"]["elementLeaf"]["oneOf"]
    enums = [o["properties"]["unknown"]["enum"] for o in unknown if "unknown" in o["properties"]]
    assert enums == [spec] or sorted(enums[0]) == sorted(spec)


def test_status_tokens_match_spec_5_0():
    for rtype in ("callsite", "data", "fieldAccess"):
        assert sorted(_resolve(_def(rtype)["properties"]["status"])["enum"]) == [
            "confirmed", "possible"]


# --- 3. unknown record type or field rejected; 4. missing required field rejected --------

def _one_of_each() -> dict[str, dict]:
    out = {}
    for r in rich_bundle():
        out.setdefault(r["type"], r)
    out["secretSighting"] = sighting_rec()
    return out


def sighting_rec(**over) -> dict:
    r = {"type": "secretSighting",
         "id": ids.sighting_id(REPO, P_PKG, COMMIT, H64), "fingerprint": H64,
         "path": P_PKG, "line": 3, "commit": COMMIT, "committedAt": "2026-10-01T08:00:00Z",
         "presentAtHead": True, "entryId": ids.config_entry_id(REPO, P_PKG, "server.port")}
    r.update(over)
    return r


def test_unknown_record_type_is_rejected():
    recs = rich_bundle() + [{"type": "widget", "id": "0" * 32}]
    with pytest.raises(bundle.BundleError):
        bundle.validate(recs)


@pytest.mark.parametrize("rtype", SPEC_ORDER)
def test_unknown_field_is_rejected(rtype):
    rec = copy.deepcopy(_one_of_each()[rtype])
    rec["zzUnknown"] = 1
    with pytest.raises(bundle.BundleError, match="zzUnknown"):
        bundle._check_schema(1, rec)


@pytest.mark.parametrize("rtype", SPEC_ORDER)
def test_every_record_fixture_is_schema_valid(rtype):
    bundle._check_schema(1, _one_of_each()[rtype])


REQUIRED_CASES = [(t, f) for t in SPEC_ORDER for f in sorted(_spec_fields(t)[0])]


@pytest.mark.parametrize(("rtype", "field"), REQUIRED_CASES)
def test_missing_required_field_is_rejected(rtype, field):
    rec = copy.deepcopy(_one_of_each()[rtype])
    del rec[field]
    with pytest.raises(bundle.BundleError):
        bundle._check_schema(1, rec)


def test_secret_sighting_is_accepted_but_not_required():
    bundle.validate(rich_bundle())  # none present: not required
    recs = ordered(counted(rich_records() + [sighting_rec()]))
    assert bundle.loads(to_bytes(recs)) == recs


def test_writer_refuses_to_emit_secret_sighting(tmp_path):
    recs = counted(rich_records() + [sighting_rec()])
    with pytest.raises(bundle.BundleError, match="secretSighting"):
        bundle.write_bundle(tmp_path / "b.jsonl", recs)
    assert not list(tmp_path.iterdir())


# --- ids: recomputed through ccg.ids (§3.2); a record whose id disagrees is rejected -------

def reid(r: dict, lang: str = LANG) -> dict:
    """Recompute ``r['id']`` from its fields with the S09 constructors, per §5's tuples and
    BS2 §1.5/§1.6 (``import``/``launchFact`` carry their ``ordinal``)."""
    t, g = r["type"], r.get
    fn = {
        "file": lambda: ids.file_id(r["repoId"], r["path"]),
        "symbol": lambda: ids.symbol_id(r["lang"], r["repoId"], r["qname"], g("signature")),
        "param": lambda: ids.point_id(r["symbolId"], "param", r["index"]),
        "ret": lambda: ids.point_id(r["symbolId"], "ret", r["ordinal"]),
        "cond": lambda: ids.point_id(r["symbolId"], "cond", r["ordinal"]),
        "callsite": lambda: ids.callsite_id(r["symbolId"], r["calleeName"], r["ordinal"]),
        "valueSet": lambda: ids.value_set_id(r["callSiteId"], r["arg"]),
        "constant": lambda: ids.constant_id(lang, REPO, r["qname"]),
        "data": lambda: ids.data_id(r["use"], r["def"], r["toPath"], r["fromPath"]),
        "controlledBy": lambda: ids.controlled_by_id(r["point"], r["cond"]),
        "decorator": lambda: ids.decorator_id(r["targetId"], r["ordinal"]),
        "fieldAccess": lambda: ids.field_access_id(r["pointId"], r["path"], r["mode"],
                                                   r["receiverType"], r["field"]),
        "dependency": lambda: ids.dependency_record_id(r["ecosystem"], r["package"],
                                                       r["manifestPath"], r["line"]),
        "testMarker": lambda: ids.test_marker_id(r["symbolId"], r["framework"]),
        "configEntry": lambda: ids.config_entry_id(r["repoId"], r["path"], r["keyPath"]),
        "schemaEntry": lambda: ids.schema_entry_id(REPO, r["path"], r["objectKey"]),
        "import": lambda: ids.import_id(r["fileId"], r["ordinal"]),
        "launchFact": lambda: ids.launch_fact_id(REPO, r["path"], r["kind"], r["ordinal"]),
        "secretSighting": lambda: ids.sighting_id(REPO, r["path"], r["commit"],
                                                  r["fingerprint"]),
    }.get(t)
    if fn:
        r["id"] = fn()
    return r


def test_reid_reproduces_every_fixture_id():
    for r in rich_bundle()[1:] + [sighting_rec()]:
        assert reid(copy.deepcopy(r))["id"] == r["id"], r["type"]


def _replace(recs, rtype, n=0, **fields):
    """Copy of ``recs`` with the ``n``-th record of ``rtype`` updated (id left alone)."""
    out = copy.deepcopy(recs)
    want = [r for r in rich_records() + [sighting_rec()] if r["type"] == rtype][n]["id"]
    hit = next(r for r in out if r["type"] == rtype and r["id"] == want)
    hit.update(fields)
    return out, hit


@pytest.mark.parametrize("rtype", [t for t in SPEC_ORDER if t != "header"])
def test_record_whose_id_does_not_match_its_fields_is_rejected(rtype):
    recs = rich_bundle() if rtype != "secretSighting" else ordered(
        counted(rich_records() + [sighting_rec()]))
    bad, _ = _replace(recs, rtype, id="f" * 32)
    with pytest.raises(bundle.BundleError, match=r"\bid\b"):
        bundle.validate(ordered(bad))


# A field that feeds the tuple, changed while the id is kept: the id no longer matches.
ID_FIELD_CASES = [
    ("file", {"path": "salesperson/src/api/other.ts"}),
    ("symbol", {"qname": P_CLIENT + "::other", "name": "other", "nameNorm": "other"}),
    ("param", {"index": 1}),
    ("ret", {"ordinal": 1}),
    ("cond", {"ordinal": 1}),
    ("callsite", {"ordinal": 1}),
    ("valueSet", {"arg": "a2"}),
    ("constant", {"qname": P_CLIENT + "::OTHER"}),
    ("data", {"fromPath": "p0.query"}),
    ("decorator", {"ordinal": 1}),
    ("fieldAccess", {"field": "other"}),
    ("dependency", {"line": 13}),
    ("testMarker", {"framework": "jest"}),
    ("configEntry", {"keyPath": "server.host"}),
    ("schemaEntry", {"objectKey": "sql:table:items"}),
    ("import", {"fileId": F_PKG}),
    ("launchFact", {"path": "salesperson/Dockerfile"}),
]


@pytest.mark.parametrize(("rtype", "fields"), ID_FIELD_CASES, ids=[c[0] for c in ID_FIELD_CASES])
def test_id_is_recomputed_from_the_spec_tuple_fields(rtype, fields):
    bad, _ = _replace(rich_bundle(), rtype, **fields)
    with pytest.raises(bundle.BundleError, match=r"\bid\b"):
        bundle.validate(ordered(bad))


def _java_bundle(sym_id: str) -> list[dict]:
    path = "src/main/java/com/acme/fxlib/Pricing.java"
    qname = path + "::Pricing.quote"
    f = {**file_rec(), "id": ids.file_id(REPO, path), "path": path, "lang": "java",
         "unitId": ids.unit_id(REPO, ".", "java")}
    s = symbol_rec(sym_id, qname, "method", "quote", 3, 9, signature="(int,String)",
                   lang="java", fileId=f["id"], path=path)
    h = header(unit={"repoId": REPO, "rootPath": ".", "lang": "java"}, unitId=f["unitId"],
               counts={**zero_counts(), "file": 1, "symbol": 1})
    return [h, f, s]


def test_symbol_signature_feeds_the_id():
    qname = "src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote"
    bundle.validate(_java_bundle(ids.symbol_id("java", REPO, qname, "(int,String)")))
    with pytest.raises(bundle.BundleError, match=r"\bid\b"):
        bundle.validate(_java_bundle(ids.symbol_id("java", REPO, qname)))


def test_header_unit_id_is_recomputed():
    recs = rich_bundle()
    recs[0]["unitId"] = ids.unit_id(REPO, "falkor-chat", LANG)
    with pytest.raises(bundle.BundleError, match="unitId"):
        bundle.validate(recs)


def test_dependency_dep_id_is_recomputed():
    bad, _ = _replace(rich_bundle(), "dependency", depId=ids.dep_id("npm", "vue"))
    with pytest.raises(bundle.BundleError, match="depId"):
        bundle.validate(ordered(bad))


def test_import_ids_are_the_files_ordinals_in_line_order():
    recs = rich_bundle()
    bad, _ = _replace(recs, "import", id=ids.import_id(F_CLIENT, 1))  # field says ordinal 0
    with pytest.raises(bundle.BundleError, match=r"\bid\b"):
        bundle.validate(ordered(bad))
    bad, rec = _replace(recs, "import", ordinal=1)  # self-consistent, but no ordinal 0
    reid(rec)
    with pytest.raises(bundle.BundleError, match="ordinal"):
        bundle.validate(ordered(bad))
    second = {"type": "import", "id": ids.import_id(F_CLIENT, 1), "fileId": F_CLIENT,
              "line": 2, "ordinal": 1, "module": "./x", "names": [], "dynamic": False}
    bundle.validate(ordered(counted(rich_records() + [second])))
    second["line"] = 0 + 1  # ordinal 1 on a line before ordinal 0's? same line is allowed
    bundle.validate(ordered(counted(rich_records() + [second])))
    first = [r for r in rich_records() if r["type"] == "import"][0]
    swapped = [*(r for r in rich_records() if r["type"] != "import"),
               {**first, "line": 5}, {**second, "line": 2}]
    with pytest.raises(bundle.BundleError, match="source order"):
        bundle.validate(ordered(counted(swapped)))


def test_launch_fact_ids_are_ordinals_per_path_and_kind():
    bad, _ = _replace(rich_bundle(), "launchFact",
                      id=ids.launch_fact_id(REPO, P_PKG, "command", 1))
    with pytest.raises(bundle.BundleError, match=r"\bid\b"):
        bundle.validate(ordered(bad))
    bad, rec = _replace(rich_bundle(), "launchFact", ordinal=1)
    reid(rec)
    with pytest.raises(bundle.BundleError, match="ordinal"):
        bundle.validate(ordered(bad))
    port = {"type": "launchFact", "id": ids.launch_fact_id(REPO, P_PKG, "port", 0),
            "path": P_PKG, "line": 9, "ordinal": 0, "kind": "port", "port": 5173,
            "protocol": "tcp", "origin": "explicit"}
    bundle.validate(ordered(counted(rich_records() + [port])))


def test_two_field_accesses_differing_only_in_path_get_two_ids():
    """§11 (analyst F1): GV-20/GV-21's case — one point, one field, two paths."""
    fa = [r for r in rich_records() if r["type"] == "fieldAccess"][0]
    other = reid({**fa, "path": "recv"})
    assert other["id"] != fa["id"]
    bundle.validate(ordered(counted(rich_records() + [other])))


# --- order and duplicates (§8.3) ----------------------------------------------------------

def test_records_out_of_type_order_are_rejected():
    recs = rich_bundle()
    recs[1], recs[3] = recs[3], recs[1]  # a symbol before a file
    with pytest.raises(bundle.BundleError, match="order"):
        bundle.validate(recs)


def test_records_out_of_id_order_within_a_type_are_rejected():
    recs = rich_bundle()
    i = [n for n, r in enumerate(recs) if r["type"] == "symbol"]
    recs[i[0]], recs[i[1]] = recs[i[1]], recs[i[0]]
    with pytest.raises(bundle.BundleError, match="order"):
        bundle.validate(recs)
    with pytest.raises(bundle.BundleError, match="order"):
        bundle.loads(to_bytes(recs))


def test_duplicate_id_within_a_type_is_rejected_by_reader_and_writer(tmp_path):
    sym = [r for r in rich_records() if r["type"] == "symbol"][1]
    recs = counted(rich_records() + [{**sym, "complexity": 9}])
    with pytest.raises(bundle.BundleError, match="duplicate"):
        bundle.write_bundle(tmp_path / "b.jsonl", recs)
    assert not list(tmp_path.iterdir())
    with pytest.raises(bundle.BundleError, match="duplicate"):
        bundle.validate(ordered(recs))


# --- header rules (§2.2) ------------------------------------------------------------------

def test_counts_must_match_the_records():
    recs = rich_bundle()
    recs[0]["counts"]["symbol"] += 1
    with pytest.raises(bundle.BundleError, match="counts"):
        bundle.validate(recs)


def test_counts_must_match_on_a_header_only_bundle():
    with pytest.raises(bundle.BundleError, match="counts"):
        bundle.validate([header(counts={**zero_counts(), "file": 1})])


def test_vendored_dirs_must_be_sorted():
    with pytest.raises(bundle.BundleError, match="vendoredDirs"):
        bundle.validate([header(vendoredDirs=["vendor", "third_party"])])
    bundle.validate([header(vendoredDirs=["third_party", "vendor"])])


def test_history_must_be_available_in_p0():
    with pytest.raises(bundle.BundleError, match="history"):
        bundle.validate([header(history="unavailable")])


def test_symbol_needs_last_changed_when_history_is_available():
    recs = rich_bundle()
    sym = [r for r in recs if r["type"] == "symbol"][0]
    del sym["lastChangedAt"], sym["lastChangedCommit"]
    with pytest.raises(bundle.BundleError, match="lastChanged"):
        bundle.validate(recs)


# --- unit consistency and references (§5.0) ----------------------------------------------

def test_file_unit_id_must_be_the_headers():
    bad, _ = _replace(rich_bundle(), "file", unitId="e" * 32)
    with pytest.raises(bundle.BundleError, match="unitId"):
        bundle.validate(ordered(bad))


@pytest.mark.parametrize("rtype", ["file", "symbol", "configEntry"])
def test_repo_id_must_be_the_headers(rtype):
    bad, rec = _replace(rich_bundle(), rtype, repoId="other-repo")
    reid(rec)
    with pytest.raises(bundle.BundleError, match="repoId"):
        bundle.validate(ordered(bad))


GHOST = "d" * 32
REF_CASES = [
    ("symbol", 0, "fileId"), ("symbol", 1, "parentId"), ("param", 0, "symbolId"),
    ("ret", 0, "symbolId"), ("cond", 0, "symbolId"), ("callsite", 0, "symbolId"),
    ("valueSet", 0, "callSiteId"), ("constant", 0, "fileId"), ("data", 0, "use"),
    ("data", 0, "def"), ("controlledBy", 0, "point"), ("controlledBy", 0, "cond"),
    ("decorator", 0, "targetId"), ("fieldAccess", 0, "pointId"), ("import", 0, "fileId"),
    ("testMarker", 0, "symbolId"),
]


@pytest.mark.parametrize(("rtype", "n", "field"), REF_CASES,
                         ids=[f"{t}.{f}" for t, _, f in REF_CASES])
def test_dangling_in_bundle_reference_is_rejected(rtype, n, field):
    bad, rec = _replace(rich_bundle(), rtype, n, **{field: GHOST})
    reid(rec)
    with pytest.raises(bundle.BundleError, match=field):
        bundle.validate(ordered(bad))


def test_import_symbol_id_must_name_a_symbol():
    bad, _ = _replace(rich_bundle(), "import", symbolId=GHOST)
    with pytest.raises(bundle.BundleError, match="symbolId"):
        bundle.validate(ordered(bad))
    ok, _ = _replace(rich_bundle(), "import", symbolId=S_API)
    bundle.validate(ordered(ok))


def test_symbol_path_must_be_its_files():
    bad, rec = _replace(rich_bundle(), "symbol", 1, fileId=F_PKG)
    with pytest.raises(bundle.BundleError, match="path"):
        bundle.validate(ordered(bad))


def test_symbol_lang_must_be_its_files():
    bad, rec = _replace(rich_bundle(), "symbol", 1, lang="python")
    reid(rec)
    with pytest.raises(bundle.BundleError, match="lang"):
        bundle.validate(ordered(bad))


def test_constant_id_takes_lang_from_its_file():
    bad, rec = _replace(rich_bundle(), "constant")
    rec["id"] = ids.constant_id("python", REPO, rec["qname"])
    with pytest.raises(bundle.BundleError, match=r"\bid\b"):
        bundle.validate(ordered(bad))


def test_qname_must_start_with_path():
    bad, rec = _replace(rich_bundle(), "symbol", 1, qname="salesperson/x.ts::apiFetch")
    reid(rec)
    with pytest.raises(bundle.BundleError, match="qname"):
        bundle.validate(ordered(bad))


def test_symbol_span_and_owned_regions():
    bad, _ = _replace(rich_bundle(), "symbol", 1, lineStart=131)
    with pytest.raises(bundle.BundleError, match="lineStart"):
        bundle.validate(ordered(bad))
    bad, _ = _replace(rich_bundle(), "symbol", 1, ownedRegions=[[20, 22], [10, 12]])
    with pytest.raises(bundle.BundleError, match="ownedRegions"):
        bundle.validate(ordered(bad))
    bad, _ = _replace(rich_bundle(), "symbol", 1, ownedRegions=[[12, 10]])
    with pytest.raises(bundle.BundleError, match="ownedRegions"):
        bundle.validate(ordered(bad))
    ok, _ = _replace(rich_bundle(), "symbol", 1, ownedRegions=[[10, 12], [20, 22]])
    bundle.validate(ordered(ok))


def test_callsite_target_in_bundle_must_agree_with_its_symbol():
    recs = rich_bundle()
    bad, _ = _replace(recs, "callsite", 1,
                      targets=[{"symbolId": S_URL, "qname": P_CLIENT + "::wrong"}])
    with pytest.raises(bundle.BundleError, match="targets"):
        bundle.validate(ordered(bad))
    outside = [{"symbolId": GHOST, "qname": "falkor-chat/x.py::f"}]
    ok, _ = _replace(recs, "callsite", 1, targets=outside)  # another bundle: allowed
    bundle.validate(ordered(ok))


# --- data / controlledBy endpoints (§5.10, §5.11) ----------------------------------------

def _data(**f):
    return reid({"type": "data", "use": CS_FETCH, "def": CS_URL, "toPath": "a0",
                 "fromPath": "r", "status": "confirmed", "reason": "intra-flow", **f})


@pytest.mark.parametrize(("fields", "why"), [
    ({"def": PARAM_URL, "fromPath": "p0"}, "same symbol"),
    ({"use": PARAM_API, "toPath": "a0"}, "use"),          # a param is never a use end
    ({"use": CS_URL, "toPath": "ret"}, "use"),            # `ret` lands on a ret point
    ({"use": CS_URL, "toPath": ""}, "use"),               # `""` lands on a cond
    ({"def": COND_API, "fromPath": "r"}, "def"),          # a cond is never a def end
    ({"def": PARAM_API, "fromPath": "r"}, "def"),         # `r` is a call's result
    ({"def": PARAM_API, "fromPath": "p1"}, "def"),        # p<i> names the param's index
    ({"def": PARAM_API, "fromPath": "recv"}, "def"),      # recv is index -1
    ({"def": CS_URL, "fromPath": "p0"}, "def"),
], ids=lambda v: v if isinstance(v, str) else None)
def test_data_endpoint_kinds_follow_their_paths(fields, why):
    with pytest.raises(bundle.BundleError, match=why):
        bundle.validate(ordered(counted(rich_records() + [_data(**fields)])))


def test_controlled_by_cond_must_be_a_cond_in_the_same_symbol():
    cb = reid({"type": "controlledBy", "point": CS_FETCH, "cond": CS_URL})
    with pytest.raises(bundle.BundleError, match="cond"):
        bundle.validate(ordered(counted(rich_records() + [cb])))
    cond_url = reid({"type": "cond", "symbolId": S_URL, "ordinal": 0, "line": 80,
                     "code": "q"})
    cb = reid({"type": "controlledBy", "point": CS_FETCH, "cond": cond_url["id"]})
    with pytest.raises(bundle.BundleError, match="same symbol"):
        bundle.validate(ordered(counted(rich_records() + [cond_url, cb])))


def test_field_access_point_kind_follows_its_path():
    fa = reid({"type": "fieldAccess", "pointId": PARAM_API, "path": "a0", "mode": "read",
               "receiverType": "?", "field": "x", "line": 1, "status": "possible",
               "reason": "field-name-match"})
    with pytest.raises(bundle.BundleError, match="pointId"):
        bundle.validate(ordered(counted(rich_records() + [fa])))
    fa = reid({**fa, "path": "p0", "mode": "write"})
    bundle.validate(ordered(counted(rich_records() + [fa])))


# --- sorted arrays (§8.1) and one encoding per path (§5.1) --------------------------------

@pytest.mark.parametrize(("rtype", "fields"), [
    ("valueSet", {"values": [{"unknown": "point-derived"}, {"const": "GET"}]}),
    ("decorator", {"args": [{"arg": "a1", "values": [{"const": 1}]},
                            {"arg": "a0", "values": [{"const": 2}]}]}),
    ("decorator", {"args": [{"arg": "a0", "values": [{"const": 1}]},
                            {"arg": "a0", "values": [{"const": 2}]}]}),
    ("import", {"names": [{"name": "b"}, {"name": "a"}]}),
    ("import", {"names": [{"alias": "x", "name": "a"}, {"name": "a"}]}),
    ("constant", {"values": [{"const": "b"}, {"const": "a"}]}),
])
def test_arrays_the_spec_sorts_must_be_sorted(rtype, fields):
    bad, _ = _replace(rich_bundle(), rtype, **fields)
    with pytest.raises(bundle.BundleError, match="sorted"):
        bundle.validate(ordered(bad))


def test_callsite_targets_must_be_sorted_by_symbol_id():
    a, b = sorted([GHOST, "c" * 32])
    t = [{"symbolId": b, "qname": "x.py::b"}, {"symbolId": a, "qname": "x.py::a"}]
    bad, _ = _replace(rich_bundle(), "callsite", 1, targets=t)
    with pytest.raises(bundle.BundleError, match="sorted"):
        bundle.validate(ordered(bad))


def test_value_sets_sort_by_canonical_json_of_each_element():
    ok, _ = _replace(rich_bundle(), "valueSet",
                     values=[{"const": "a"}, {"const": 10}, {"const": 9}])
    # canonical JSON bytes: '"' (0x22) < '1' < '9', so "a" < 10 < 9 — not numeric order
    bundle.validate(ordered(ok))


@pytest.mark.parametrize(("arg", "ok"), [
    ('a1["content-type"]', True), ('kw.headers["a.b"]', True), ('a1[""]', True),
    ('a1["method"]', False),    # an identifier is written bare
    ('a1["caf\\u00e9"]', False),  # JSON escaping is the canonical one (literal non-ASCII)
])
def test_quoted_field_segments_have_one_encoding(arg, ok):
    recs, rec = _replace(rich_bundle(), "valueSet", arg=arg)
    reid(rec)
    if ok:
        bundle.validate(ordered(recs))
    else:
        with pytest.raises(bundle.BundleError, match="quot"):
            bundle.validate(ordered(recs))


# --- dependency normalization (§5.15) and configEntry.example (§5.17) --------------------

def test_pypi_package_must_be_pep503_normalized():
    bad, rec = _replace(rich_bundle(), "dependency", ecosystem="PyPI", package="Foo_Bar",
                        depId=ids.dep_id("PyPI", "Foo_Bar"))
    reid(rec)
    with pytest.raises(bundle.BundleError, match="package"):
        bundle.validate(ordered(bad))
    ok, rec = _replace(rich_bundle(), "dependency", ecosystem="PyPI", package="foo-bar",
                       depId=ids.dep_id("PyPI", "foo-bar"))
    reid(rec)
    bundle.validate(ordered(ok))


def test_nuget_package_must_be_lowercased():
    bad, rec = _replace(rich_bundle(), "dependency", ecosystem="NuGet", package="Newtonsoft.Json",
                        depId=ids.dep_id("NuGet", "Newtonsoft.Json"))
    reid(rec)
    with pytest.raises(bundle.BundleError, match="package"):
        bundle.validate(ordered(bad))


def test_config_entry_example_is_copied_from_its_files_role():
    bad, _ = _replace(rich_bundle(), "configEntry", example=True)
    with pytest.raises(bundle.BundleError, match="example"):
        bundle.validate(ordered(bad))


# --- vendored paths yield no record (§5.0) ------------------------------------------------

def test_record_under_a_vendored_dir_below_the_root_is_rejected():
    recs = rich_bundle()
    recs[0]["vendoredDirs"] = ["src"]
    with pytest.raises(bundle.BundleError, match="vendored"):
        bundle.validate(recs)


def test_unit_root_itself_never_makes_a_path_vendored():
    recs = rich_bundle()
    recs[0]["vendoredDirs"] = ["salesperson"]
    bundle.validate(recs)


def test_import_resolving_into_a_vendored_path_is_rejected():
    recs, _ = _replace(rich_bundle(), "import",
                       resolvedPath="salesperson/node_modules/x/index.js")
    recs[0]["vendoredDirs"] = ["node_modules"]
    with pytest.raises(bundle.BundleError, match="vendored"):
        bundle.validate(recs)


# --- strings: NFC everywhere, identity fields stripped (§8.1) -----------------------------

def test_non_nfc_string_is_rejected():
    bad, _ = _replace(rich_bundle(), "cond", code="café")
    with pytest.raises(bundle.BundleError, match="NFC"):
        bundle.validate(ordered(bad))


@pytest.mark.parametrize(("rtype", "field"), [("symbol", "name"), ("callsite", "calleeName"),
                                              ("param", "name")])
def test_identity_string_with_edge_whitespace_is_rejected(rtype, field):
    recs = rich_bundle()
    bad, rec = _replace(recs, rtype, 1 if rtype == "symbol" else 0)
    rec[field] = rec[field] + " "
    with pytest.raises(bundle.BundleError, match="strip"):
        bundle.validate(ordered(bad))


def test_free_text_keeps_its_whitespace():
    ok, _ = _replace(rich_bundle(), "valueSet", values=[{"const": " x"}])
    bundle.validate(ordered(ok))


def test_float_is_rejected_even_when_integral():
    bad, _ = _replace(rich_bundle(), "param", line=107.0)
    with pytest.raises(bundle.BundleError, match="integer"):
        bundle.validate(ordered(bad))


# --- line framing and canonical bytes (§8.1) ----------------------------------------------

def _good() -> bytes:
    return to_bytes(rich_bundle())


@pytest.mark.parametrize(("data", "why"), [
    (lambda: b"\xef\xbb\xbf" + _good(), "BOM"),
    (lambda: _good().replace(b"\n", b"\r\n"), r"\\r"),
    (lambda: _good()[:-1], "exactly one"),
    (lambda: _good() + b"\n", "exactly one"),
    (lambda: _good().replace(b'"line":107', b'"line":107.0', 1), "integer"),
    (lambda: _good().replace(b'"line":107', b'"line":NaN', 1), "integer"),
    (lambda: _good().replace(b'"index":0,', b'"index":0,"index":0,', 1), "duplicate key"),
    (lambda: _good().replace(b'","', b'", "', 1), "canonical"),
    (lambda: _good().replace(b'"type":"header",', b"", 1).replace(
        b'{"builtAt"', b'{"type":"header","builtAt"', 1), "canonical"),
    (lambda: to_bytes([header()]).replace(b"graphmind-ai-lab", "graphmind-ai-lab".encode(), 1)
     .replace(b'"python"', b'"python"') + b'["not an object"]\n', "object"),
    (lambda: b"not json\n", "JSON"),
    (lambda: b"\xff\n", "UTF-8"),
])
def test_non_canonical_bytes_are_rejected(data, why):
    with pytest.raises(bundle.BundleError, match=why):
        bundle.loads(data())


def test_escaped_non_ascii_is_not_canonical():
    recs, _ = _replace(rich_bundle(), "cond", code="café")
    good = to_bytes(ordered(recs))
    bundle.loads(good)
    with pytest.raises(bundle.BundleError, match="canonical"):
        bundle.loads(good.replace("café".encode(), b"caf\\u00e9"))


# --- schema conditionals (§4.3, §5.x) pinned on single records ----------------------------

def _sym(**f):
    return {**[r for r in rich_records() if r["type"] == "symbol"][1], **f}


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: _sym(lang="java", signature="(int)"), True),
    (lambda: _sym(lang="java"), False),                      # function-like java: required
    (lambda: _sym(lang="java", kind="type"), True),
    (lambda: _sym(lang="java", kind="type", signature="()"), False),
    (lambda: _sym(lang="java", kind="lambda", signature="()"), False),
    (lambda: _sym(signature="(x)"), False),                  # javascript: never
    (lambda: _sym(lang="config"), False),                    # config files: never symbols
    (lambda: _sym(kind="module"), False),                    # parentId absent on file syms
    (lambda: {k: v for k, v in _sym().items() if k != "lastChangedCommit"}, False),
    (lambda: _sym(ownedRegions=[]), False),
    (lambda: _sym(ownedRegions=[[1, 2, 3]]), False),
    (lambda: _sym(complexity=0), False),
    (lambda: _sym(lineStart=0), False),
    (lambda: _sym(contentHash=H64), False),
    (lambda: _sym(fingerprint=H64[:63] + "\n"), False),       # no trailing-newline slip
    (lambda: _sym(lastChangedAt="2026-10-01T08:00:00.5Z"), False),
    (lambda: _sym(lastChangedAt="2026-10-01T08:00:00+00:00"), False),
])
def test_symbol_schema_conditionals(rec, ok):
    check(rec(), ok)


def check(rec, ok):
    if ok:
        bundle._check_schema(1, rec)
    else:
        with pytest.raises(bundle.BundleError):
            bundle._check_schema(1, rec)


def _cs(n=0, **f):
    return {**[r for r in rich_records() if r["type"] == "callsite"][n], **f}


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: {k: v for k, v in _cs().items() if k != "externalTarget"}, False),
    (lambda: _cs(1, externalTarget="x"), False),             # only when external
    (lambda: _cs(1, targets=[]), False),                     # confirmed needs targets
    (lambda: _cs(1, targets=[], status="possible", reason="name-match"), True),
    (lambda: _cs(1, status="possible", reason="name-match"), False),
    (lambda: _cs(reason="prompt-mention"), True),
    (lambda: _cs(reason="resolved-call"), False),
    (lambda: _cs(status="confirmed"), False),
    (lambda: _cs(reason="intra-flow"), False),
    (lambda: _cs(1, targets=[{"symbolId": S_URL}]), False),
])
def test_callsite_schema_conditionals(rec, ok):
    check(rec(), ok)


def _vs(**f):
    return {**[r for r in rich_records() if r["type"] == "valueSet"][0], **f}


def _nest(depth):
    e = {"const": "x"}
    for _ in range(depth - 1):
        e = {"concat": [e]}
    return e


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: _vs(values=[]), False),
    (lambda: _vs(values=[{"const": i} for i in range(16)]), True),
    (lambda: _vs(values=[{"const": i} for i in range(17)]), False),
    (lambda: _vs(values=[_nest(4)]), True),
    (lambda: _vs(values=[_nest(5)]), False),
    (lambda: _vs(values=[{"template": []}]), False),
    (lambda: _vs(values=[{"const": None}]), False),
    (lambda: _vs(values=[{"const": 1.5}]), False),
    (lambda: _vs(values=[{"const": True}]), True),
    (lambda: _vs(values=[{"unknown": "made-up"}]), False),
    (lambda: _vs(values=[{"ref": "no-double-colon"}]), False),
    (lambda: _vs(values=[{"importRef": {"module": "m", "name": "n"}}]), True),
    (lambda: _vs(values=[{"const": "a", "ref": "a::b"}]), False),
    (lambda: _vs(arg="a1.b.c.d"), True),
    (lambda: _vs(arg="a1.b.c.d.e"), False),                  # fieldDepthCap 3
    (lambda: _vs(arg="kw.prefix"), True),
    (lambda: _vs(arg="recv"), True),
    (lambda: _vs(arg="a01"), False),
    (lambda: _vs(arg="p0"), False),                          # a defPath, not an argPath
    (lambda: _vs(arg="ret"), False),
    (lambda: _vs(arg="a1.[\"x-y\"]"), False),                # spec examples: no '.' before [
])
def test_value_set_schema_conditionals(rec, ok):
    check(rec(), ok)


def _d(n=0, **f):
    return {**[r for r in rich_records() if r["type"] == "data"][n], **f}


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: _d(fromPath=""), False),                        # no defPath is ""
    (lambda: _d(fromPath="ret"), False),
    (lambda: _d(toPath="r"), False),
    (lambda: _d(status="possible", reason="field-truncated"), True),
    (lambda: _d(status="possible", reason="intra-flow"), False),
    (lambda: _d(status="confirmed", reason="shell-dynamic"), False),
    (lambda: _d(reason="name-match"), False),
])
def test_data_schema_conditionals(rec, ok):
    check(rec(), ok)


def _fa(**f):
    return {**[r for r in rich_records() if r["type"] == "fieldAccess"][0], **f}


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: _fa(receiverType=P_CLIENT + "::Client", status="confirmed",
                 reason="resolved-receiver"), True),
    (lambda: _fa(status="confirmed", reason="resolved-receiver"), False),  # "?" receiver
    (lambda: _fa(receiverType=P_CLIENT + "::C"), False),     # resolved type: not name-match
    (lambda: _fa(reason="field-truncated"), True),
    (lambda: _fa(path="p0"), False),                         # read: not a defPath
    (lambda: _fa(path="ret.x"), True),
    (lambda: _fa(mode="write", path="r"), True),
    (lambda: _fa(mode="write", path="a0"), False),
    (lambda: _fa(receiverType="Client"), False),
])
def test_field_access_schema_conditionals(rec, ok):
    check(rec(), ok)


def _ce(**f):
    return {**[r for r in rich_records() if r["type"] == "configEntry"][0], **f}


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: _ce(bindValue="https://hooks.slack.com/services/T0/B0/XXXX"), False),
    (lambda: _ce(bindValue="http://u:p@localhost/"), False),
    (lambda: _ce(bindValue="http://localhost/?q=1"), False),
    (lambda: _ce(bindValue="http://localhost/#f"), False),
    (lambda: _ce(bindValue="http://h/" + "a" * 33), False),
    (lambda: _ce(bindValue="Hello world"), False),
    (lambda: _ce(bindValue="http://LOCALHOST/"), False),
    (lambda: _ce(keyPath="db.PassWord", bindValue="5432"), False),
    (lambda: _ce(keyPath="api.tokenUrl", bindValue="localhost"), False),
    (lambda: {k: v for k, v in _ce(keyPath="db.PassWord").items() if k != "bindValue"},
     True),                                                  # no bindValue: fine

    (lambda: _ce(keyPath="services.falkordb.ports[0]"), True),
    (lambda: _ce(keyPath='a["b.c"].d'), True),
    (lambda: _ce(keyPath="a..b"), False),
    (lambda: _ce(format="xml"), False),
])
def test_config_entry_schema_conditionals(rec, ok):
    check(rec(), ok)


def _lf(**f):
    return {**[r for r in rich_records() if r["type"] == "launchFact"][0], **f}


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: {k: v for k, v in _lf().items() if k != "args"}, False),
    (lambda: _lf(port=80), False),                           # a port field on a command
    (lambda: {**{k: v for k, v in _lf().items() if k not in ("command", "args")},
              "kind": "port", "port": 80, "protocol": "tcp"}, True),
    (lambda: {**{k: v for k, v in _lf().items() if k not in ("command", "args")},
              "kind": "port", "port": 80}, False),
    (lambda: {**{k: v for k, v in _lf().items() if k not in ("command", "args")},
              "kind": "env-default", "envVar": "PORT", "default": "8200", "defaultHash": H64,
              "origin": "env-default"}, True),
    (lambda: {**{k: v for k, v in _lf().items() if k not in ("command", "args")},
              "kind": "env-default", "envVar": "PORT", "defaultHash": H64,
              "origin": "env-default"}, True),                # default withheld (W4)
    (lambda: {**{k: v for k, v in _lf().items() if k not in ("command", "args")},
              "kind": "env-default", "envVar": "PORT", "default": "8200",
              "origin": "env-default"}, False),               # defaultHash is required
    (lambda: {k: v for k, v in _lf().items() if k != "ordinal"}, False),
    (lambda: {**{k: v for k, v in _lf().items() if k not in ("command", "args")},
              "kind": "mount", "mountPath": "/data", "service": "db",
              "origin": "compose"}, True),
])
def test_launch_fact_schema_conditionals(rec, ok):
    check(rec(), ok)


@pytest.mark.parametrize(("rec", "ok"), [
    (lambda: sighting_rec(symbolId=S_API), False),           # exactly one enclosing end
    (lambda: {k: v for k, v in sighting_rec().items() if k != "entryId"}, False),
    (lambda: sighting_rec(removedInCommit=TREE), True),
])
def test_secret_sighting_schema_conditionals(rec, ok):
    check(rec(), ok)


@pytest.mark.parametrize(("path", "ok"), [
    ("a/b.py", True), ("./a.py", False), ("/a.py", False), ("a/../b.py", False),
    ("a/./b.py", False), ("a/", False), ("a//b", False), (".", False), ("..", False),
    (".github/x.yml", True), ("a.py\n", False),
])
def test_path_form(path, ok):
    check(file_rec(path=path), ok)


@pytest.mark.parametrize(("over", "ok"), [
    ({"unit": {"repoId": REPO, "rootPath": ".", "lang": LANG}}, True),
    ({"unit": {"repoId": REPO, "rootPath": "salesperson/", "lang": LANG}}, False),
    ({"unit": {"repoId": "Graphmind", "rootPath": ROOT, "lang": LANG}}, False),
    ({"unit": {"repoId": "-x", "rootPath": ROOT, "lang": LANG}}, False),
    ({"builtAt": "2026-10-29T12:00:00.123Z"}, False),
    ({"sandboxAuditIds": []}, False),
    ({"sandboxAuditIds": [AUDIT.upper()]}, False),
    ({"sandboxAuditIds": ["0f8fad5b-d9cb-169f-a165-70867728950e"]}, False),  # not v4
    ({"bundleSchemaVersion": "1.1"}, True),     # spec5 §1 admits "1.1"; V1b (a "1.1" header with no bindParts) is bundle-level (BK-5)
    ({"bundleSchemaVersion": "1.2"}, False),
    ({"bundleSchemaVersion": "2.0"}, False),
    ({"fieldDepthCap": 4}, False),
    ({"builtCommit": "1" * 39}, False),
    ({"counts": {**zero_counts(), "widget": 0}}, False),
    ({"counts": {**zero_counts(), "file": -1}}, False),
    ({"tools": {**header()["tools"], "extractScriptHash": "abc"}}, False),
    ({"flowProjected": 1}, False),
])
def test_header_schema(over, ok):
    check(header(**over), ok)


def test_cond_code_is_collapsed_and_capped():
    c = [r for r in rich_records() if r["type"] == "cond"][0]
    check({**c, "code": "a" * 200}, True)
    check({**c, "code": "a" * 201}, False)
    check({**c, "code": "a  b"}, False)
    check({**c, "code": "a\nb"}, False)


def test_integer_fields_refuse_booleans():
    check({**[r for r in rich_records() if r["type"] == "param"][0], "index": True}, False)


# --- gaps found by mutation (S10 mutant table) --------------------------------------------

def test_bundle_with_no_header_at_all_is_rejected():
    with pytest.raises(bundle.BundleError, match="header"):
        bundle.validate([file_rec()])
    with pytest.raises(bundle.BundleError, match="header"):
        bundle.loads(to_bytes([file_rec()]))


def test_header_unit_id_is_recomputed_on_a_header_only_bundle():
    with pytest.raises(bundle.BundleError, match="unitId"):
        bundle.validate([header(unitId=ids.unit_id(REPO, "falkor-chat", LANG))])


NESTED_UNKNOWN = [
    ("header", lambda r: r["unit"]), ("header", lambda r: r["tools"]),
    ("callsite", lambda r: r["targets"][0]), ("import", lambda r: r["names"][0]),
    ("decorator", lambda r: r["args"][0]), ("decorator", lambda r: r["args"][0]["values"][0]),
    ("valueSet", lambda r: r["values"][0]),
]


@pytest.mark.parametrize(("rtype", "get"), NESTED_UNKNOWN,
                         ids=[f"{t}-{i}" for i, (t, _) in enumerate(NESTED_UNKNOWN)])
def test_unknown_nested_field_is_rejected(rtype, get):
    recs = {r["type"]: r for r in reversed(rich_bundle())}  # callsite: the one with targets
    rec = copy.deepcopy(recs[rtype] if rtype != "callsite" else _cs(1))
    get(rec)["zzUnknown"] = "x"
    with pytest.raises(bundle.BundleError):
        bundle._check_schema(1, rec)


def test_import_ref_element_rejects_unknown_key():
    check(_vs(values=[{"importRef": {"module": "m", "name": "n", "zz": "x"}}]), False)


def test_constant_id_uses_the_files_lang_not_the_units():
    """§3.3: a ``.c`` file in a ``cpp`` unit is ``c`` (GV-15's case); its constants too."""
    unit_id = ids.unit_id("fw", ".", "cpp")
    f = {**file_rec(), "id": ids.file_id("fw", "src/util.c"), "repoId": "fw",
         "unitId": unit_id, "path": "src/util.c", "lang": "c"}

    def bundle_with(const_id):
        c = {"type": "constant", "id": const_id, "qname": "src/util.c::CRC_POLY",
             "fileId": f["id"], "line": 3, "values": [{"const": 7}]}
        h = header(unit={"repoId": "fw", "rootPath": ".", "lang": "cpp"}, unitId=unit_id,
                   counts={**zero_counts(), "file": 1, "constant": 1})
        return [h, f, c]

    bundle.validate(bundle_with(ids.constant_id("c", "fw", "src/util.c::CRC_POLY")))
    with pytest.raises(bundle.BundleError, match=r"\bid\b"):
        bundle.validate(bundle_with(ids.constant_id("cpp", "fw", "src/util.c::CRC_POLY")))


# === S10 gate fix round (docs/reviews/connected-code-graph-p0-impl.md, ## S10) ============

# --- S10-M2: prose records, checked both ways ---------------------------------------------
# Each prose record's field text, cut from its spec section by the spec's own wording. A
# reworded section fails the `assert m` loudly rather than passing vacuously.
PROSE_FIELD_TEXT = {
    "ret": (r"5\.5 ", r"\*\*`ret`\.\*\*.*?Fields:(.*?)(?=\n- \*\*`cond`)"),
    "cond": (r"5\.5 ", r"\*\*`cond`\.\*\*.*?Fields:(.*)"),
    "controlledBy": ((r"1\.4 ", SPEC_TEXT), r"^(.*?) A `cond`→`cond` record"),
    "testMarker": (r"5\.16 ", r"Fields:(.*)"),
    "schemaEntry": (r"5\.18 ", r"\*\*`schemaEntry`\.\*\*.*?Fields:(.*?)Graph:"),
    "secretSighting": (r"5\.19 ", r"Fields \(.*?\):(.*?)\*\*v1 writers"),
}

# Backticked spans in a field text that are not fields, each with its reason.
_FRAMEWORKS = ["pytest", "unittest", "vitest", "jest", "playwright", "junit", "kotest",
               "xunit", "nunit", "go-test", "gtest", "bats"]
NOT_FIELDS = {
    "controlledBy": {
        "ret": "a point kind `point` may name (BS2 §1.4 'a call site, a `ret`, a `param` or a "
               "`cond`'), not a field",
        "param": "a point kind `point` may name (BS2 §1.4), not a field",
        "point ≠ cond": "BS2 §1.4's constraint between the two fields, not a field",
    },
    "testMarker": {
        fw: "an example in `framework`'s open list (§5.16 ends it with '…'): a free string, "
            "not a field or an enum token" for fw in _FRAMEWORKS
    },
    "secretSighting": {
        "findingId": "the assembly id the enclosing end feeds (§3.2), not a record field",
    },
}


def _prose_tokens(rtype: str) -> list[str]:
    sec, rx = PROSE_FIELD_TEXT[rtype]
    m = re.search(rx, _prose_section(sec).strip(), re.S)
    assert m, (rtype, rx)
    return [" ".join(t.split()) for t in re.findall(r"`([^`]+)`", m.group(1), re.S)]


def test_prose_field_text_regexes_cover_every_prose_record():
    assert set(PROSE_FIELD_TEXT) == set(PROSE_RECORDS)


@pytest.mark.parametrize("rtype", sorted(PROSE_RECORDS))
def test_every_backticked_token_in_a_prose_record_is_accounted_for(rtype):
    _, req, opt, enums = PROSE_RECORDS[rtype]
    known = set(req) | set(opt) | {t for ts in enums.values() for t in ts}
    excused = NOT_FIELDS.get(rtype, {})
    tokens = _prose_tokens(rtype)
    unaccounted = [t for t in tokens if t not in known and t not in excused]
    assert not unaccounted, f"{rtype}: spec names {unaccounted}, not transcribed nor excused"
    # and no stale excuse: each NOT_FIELDS entry still appears in the spec text
    assert set(excused) <= set(tokens), set(excused) - set(tokens)


# --- S10-M1: bindValue's IP literal, secret names, length cap ------------------------------

def _bind_bundle(value: str, key_path: str = "server.port") -> list[dict]:
    recs, rec = _replace(rich_bundle(), "configEntry", bindValue=value, keyPath=key_path)
    return ordered([r if r is not rec else reid(rec) for r in recs])


@pytest.mark.parametrize("value", [
    ":", "abc:", "a1b2c3d4e5:f6", "deadbeefcafe0123:9876543210abcdef",
    "1:2:3", "1::2::3", ":::", "12345::1", "1:2:3:4:5:6:7:8:9",
    "http://[abc:]/", "http://[1:2:3]:80/mcp", "http://[]/",
    # BS2 §1.7.2: RFC 5952 canonical text only — no '.', no '%', no leading zeros, '::' maximal
    "::ffff:192.0.2.1", "fe80::1%eth0", "0:0:0:0:0:0:0:1", "2001:0db8::1",
    "2001:db8:0:0:0:0:0:1", "2001:DB8::1", "http://[0:0:0:0:0:0:0:1]/",
])
def test_bind_value_colon_string_that_is_not_an_ip_is_rejected(value):
    with pytest.raises(bundle.BundleError, match="bindValue|schema"):
        bundle.validate(_bind_bundle(value))


@pytest.mark.parametrize("value", [
    "::1", "::", "fe80::1", "2001:db8::8a2e:370:7334", "1:2:3:4:5:6:7:8",
    "http://[::1]:8000/mcp", "http://[2001:db8::1]/", "[::1]:8000",
])
def test_bind_value_ipv6_literal_is_admitted(value):
    bundle.validate(_bind_bundle(value))


@pytest.mark.parametrize("value", [":", "abc:", "a1b2c3d4e5:f6",
                                   "deadbeefcafe0123:9876543210abcdef", "http://[abc:]/"])
def test_schema_alone_rejects_non_ip_colon_shapes(value):
    check(_ce(bindValue=value), False)


def _spec_secret_tokens() -> list[str]:
    """§1.7.3's 'the original 15' substring words."""
    text = _section(r"1\.7\.3 ")
    m = re.search(r"the original 15:(.*?);", text, re.S)
    assert m
    return _ticks(m.group(1))


def test_spec_lists_fifteen_secret_name_tokens():
    assert len(_spec_secret_tokens()) == 15


@pytest.mark.parametrize("token", _spec_secret_tokens())
def test_each_spec_secret_token_in_a_key_path_rejects_bind_value(token):
    for key_path in (f"svc.{token}", f"svc.My{token.upper()}x", f"{token}[0]"):
        with pytest.raises(bundle.BundleError):
            bundle.validate(_bind_bundle("5173", key_path))


def test_key_path_without_a_secret_token_keeps_bind_value():
    bundle.validate(_bind_bundle("5173", "server.port"))


def _long_dnsname(n: int) -> str:
    """A dnsname-shaped string of exactly ``n`` characters: seven 63-character labels, one
    of the remainder, then ``.com`` (valid only for 452 < n < 516)."""
    return ".".join(["a" * 63] * 7 + ["a" * (n - 7 * 64 - 4)]) + ".com"


def test_bind_value_length_cap_is_shorter_than_512_in_the_schema():
    """§1.7.2: 'fewer than 512 characters'. The grammar's own bounds (dnsname ≤ 253) make
    511 unreachable for a *valid* value, so the cap is pinned where it lives: the schema,
    on dnsname-shaped strings either side of it; ``bundle.py`` rejects both."""
    v511, v512 = _long_dnsname(511), _long_dnsname(512)
    assert (len(v511), len(v512)) == (511, 512)
    check(_ce(bindValue=v511), True)
    check(_ce(bindValue=v512), False)
    for v in (v511, v512):
        with pytest.raises(bundle.BundleError):
            bundle.validate(_bind_bundle(v))


# --- S10-m1: status x reason against §7.2's Status column ---------------------------------

def _spec_reason_status() -> dict[str, str]:
    return {_ticks(c[0])[0]: c[1] for c in _table_rows(_section(r"7\.2 ")) if c[6] == "yes"}


RESOLVED_TYPE = P_CLIENT + "::Client"


def _pairing_cases():
    """Every record with a reason x every code it admits x both statuses x the auxiliary
    field that §5.6/§5.13 tie the pairing to (callsite targets, fieldAccess receiverType)."""
    cases = []
    for code in _resolve(_def("callsite")["properties"]["reason"])["enum"]:
        for status in ("confirmed", "possible"):
            for targets in ([], [{"symbolId": S_URL, "qname": P_CLIENT + "::buildUrl"}]):
                cases.append(("callsite", code, status, {"targets": targets}))
    for code in _resolve(_def("data")["properties"]["reason"])["enum"]:
        for status in ("confirmed", "possible"):
            cases.append(("data", code, status, {}))
    for code in _resolve(_def("fieldAccess")["properties"]["reason"])["enum"]:
        for status in ("confirmed", "possible"):
            for rt in ("?", RESOLVED_TYPE):
                cases.append(("fieldAccess", code, status, {"receiverType": rt}))
    return cases


def _pairing_expected(rtype, code, status, aux) -> bool:
    if status != _spec_reason_status()[code]:          # §7.2 Status column
        return False
    if rtype == "callsite":                            # §5.6: confirmed iff targets non-empty
        return (status == "confirmed") == bool(aux["targets"])
    if rtype == "fieldAccess":                         # §5.13 status/reason row
        if code == "resolved-receiver":
            return aux["receiverType"] != "?"
        if code == "field-name-match":
            return aux["receiverType"] == "?"
    return True


@pytest.mark.parametrize(("rtype", "code", "status", "aux"), _pairing_cases(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_status_reason_pairing_matches_spec(rtype, code, status, aux):
    base = {"callsite": _cs(1), "data": _d(), "fieldAccess": _fa()}[rtype]
    check({**base, **aux, "status": status, "reason": code},
          _pairing_expected(rtype, code, status, aux))


def test_pairing_probe_has_an_accepted_case_per_code():
    accepted = {c for t, c, s, a in _pairing_cases() if _pairing_expected(t, c, s, a)}
    assert accepted == set(_spec_reason_status())


# --- S10-m2: four rules that held in code without a test ---------------------------------

IDENTITY_PROBES = [  # (record type, n, field) — one per _IDENTITY_KEYS entry, plus an *Id key
    ("symbol", 1, "id"), ("symbol", 1, "qname"), ("file", 0, "path"),
    ("configEntry", 0, "keyPath"), ("symbol", 1, "name"), ("callsite", 0, "calleeName"),
    ("param", 0, "symbolId"),
]


@pytest.mark.parametrize(("rtype", "n", "field"), IDENTITY_PROBES,
                         ids=[f"{t}.{f}" for t, _, f in IDENTITY_PROBES])
@pytest.mark.parametrize("edge", [" {}", "{} ", "\t{}"])
def test_every_identity_field_must_be_stripped(rtype, n, field, edge):
    bad, rec = _replace(rich_bundle(), rtype, n)
    rec[field] = edge.format(rec[field])
    with pytest.raises(bundle.BundleError, match="strip"):
        bundle.validate(ordered(bad))


def _java_call_bundle(target_sig: str) -> list[dict]:
    """A Java unit: `Pricing.quote(int,String)` and `Pricing.warm()` calling it (§4.3)."""
    path = "src/main/java/com/acme/fxlib/Pricing.java"
    unit = ids.unit_id(REPO, ".", "java")
    f = {**file_rec(), "id": ids.file_id(REPO, path), "path": path, "lang": "java",
         "unitId": unit}
    quote = symbol_rec(ids.symbol_id("java", REPO, path + "::Pricing.quote", "(int,String)"),
                       path + "::Pricing.quote", "method", "quote", 3, 9, lang="java",
                       signature="(int,String)", fileId=f["id"], path=path)
    warm = symbol_rec(ids.symbol_id("java", REPO, path + "::Pricing.warm", "()"),
                      path + "::Pricing.warm", "method", "warm", 10, 12, lang="java",
                      signature="()", fileId=f["id"], path=path)
    cs = reid({"type": "callsite", "symbolId": warm["id"], "calleeName": "quote",
               "ordinal": 0, "callKind": "call", "line": 11, "resolution": "resolved",
               "targets": [{"symbolId": quote["id"], "qname": quote["qname"],
                            "signature": target_sig}],
               "status": "confirmed", "reason": "resolved-call"})
    return ordered(counted([f, quote, warm, cs],
                           unit={"repoId": REPO, "rootPath": ".", "lang": "java"},
                           unitId=unit))


def test_overloaded_target_signature_must_agree_with_its_symbol():
    bundle.validate(_java_call_bundle("(int,String)"))
    with pytest.raises(bundle.BundleError, match="targets"):
        bundle.validate(_java_call_bundle("(int,String,Currency)"))


_LF_BASE = {
    "command": {"command": "vite", "args": []},
    "port": {"port": 80, "protocol": "tcp"},
    "env-default": {"envVar": "PORT", "default": "8200", "defaultHash": H64},
    "mount": {"mountPath": "/data"},
}
_LF_FOREIGN = {f: v for fs in _LF_BASE.values() for f, v in fs.items()}
LF_CASES = [(k, f) for k, own in _LF_BASE.items() for f in _LF_FOREIGN if f not in own]


@pytest.mark.parametrize(("kind", "foreign"), LF_CASES, ids=[f"{k}+{f}" for k, f in LF_CASES])
def test_launch_fact_kind_excludes_every_other_kinds_field(kind, foreign):
    rec = {"type": "launchFact", "id": "0" * 32, "path": P_PKG, "line": 1, "ordinal": 0,
           "kind": kind, "origin": "explicit", **_LF_BASE[kind]}
    check(rec, True)
    check({**rec, foreign: _LF_FOREIGN[foreign]}, False)


def test_vendored_name_as_a_files_basename_is_not_vendored():
    """§5.0 tests *directory* segments: a script literally named `vendor` is first-party."""
    path = "salesperson/scripts/vendor"
    f = file_rec(id=ids.file_id(REPO, path), path=path, lang="shell", kind="script")
    recs = ordered(counted(rich_records() + [f], vendoredDirs=["vendor"]))
    bundle.validate(recs)


# === BS2 v1.3 (docs/plans/connected-code-graph-bundle-spec3.md, §4.1's S10 items) ==========

import math  # noqa: E402
import shlex  # noqa: E402

# --- vector tables, parsed from BS2's text (§1.6.3: "never retyped into the test") ----------

def _vector_rows(prefix: str) -> dict[str, list[str]]:
    """Every ``| <prefix>-N | … |`` row of BS2, by id, as stripped cells."""
    out = {}
    for line in SPEC_TEXT.splitlines():
        m = re.match(rf"^\| ({prefix}-\d+) \|", line)
        if m:
            out[m.group(1)] = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
    return out


AV_ROWS = _vector_rows("AV")
BV_ROWS = _vector_rows("BV")


def _av(av_id: str) -> tuple[list[str], list[str]]:
    """(source argv, expected written argv). 'unchanged' means the input itself (§1.6.3).
    A parenthetical after the output is commentary and may itself hold backticks."""
    cells = AV_ROWS[av_id]
    src = shlex.split(re.search(r"`([^`]+)`", cells[1]).group(1))
    written = cells[2].split(" (")[0]
    expected = list(src) if written.startswith("unchanged") else _ticks(written)
    return src, expected


def test_av_and_bv_tables_are_complete():
    assert sorted(AV_ROWS, key=lambda k: int(k[3:])) == [f"AV-{i}" for i in range(1, 69)]
    assert sorted(BV_ROWS, key=lambda k: int(k[3:])) == [f"BV-{i}" for i in range(1, 29)]


@pytest.mark.parametrize("av_id", [f"AV-{i}" for i in range(1, 69)])
def test_av_vector_gate_output(av_id):
    src, expected = _av(av_id)
    assert bundle.gate_argv(src) == expected


def _lf_bundle(argv: list[str]) -> list[dict]:
    recs, _ = _replace(rich_bundle(), "launchFact", command=argv[0], args=argv[1:])
    return ordered(recs)


@pytest.mark.parametrize("av_id", [f"AV-{i}" for i in range(1, 69)])
def test_av_vector_validator_accepts_the_output_and_rejects_a_leak(av_id):
    src, expected = _av(av_id)
    bundle.validate(_lf_bundle(expected))
    if src != expected:
        with pytest.raises(bundle.BundleError, match="argv|command|args"):
            bundle.validate(_lf_bundle(src))


_FORMATS = {"mcp-json", "compose", "env", "json", "yaml", "toml"}


def _bv(bv_id: str) -> tuple[str, str, str, bool]:
    """(keyPath, format, value, binds) from a BV row. BV-20's value cell is the description
    '32 hex', made concrete here as 32 lowercase hex digits."""
    cells = BV_ROWS[bv_id]
    key_path = _ticks(cells[1])[0]
    fmt = re.search(r"\(([a-z-]+)\)", cells[1])
    fmt = fmt.group(1) if fmt and fmt.group(1) in _FORMATS else "json"
    vals = re.findall(r"`([^`]+)`", cells[2])
    if vals:
        value = vals[0]
    else:
        n = int(re.fullmatch(r"(\d+) hex", cells[2]).group(1))
        value = ("0123456789abcdef" * 4)[:n]
    return key_path, fmt, value, "bind" in cells[3]


@pytest.mark.parametrize("bv_id", [f"BV-{i}" for i in range(1, 29)])
def test_bv_vector_outcome(bv_id):
    key_path, fmt, value, binds = _bv(bv_id)
    recs, rec = _replace(rich_bundle(), "configEntry", keyPath=key_path, format=fmt,
                         bindValue=value)
    recs = ordered([r if r is not rec else reid(rec) for r in recs])
    if binds:
        bundle.validate(recs)
    else:
        with pytest.raises(bundle.BundleError):
            bundle.validate(recs)
    assert bundle.bind_value_allowed(key_path, value) is binds


# --- the 36 words (§1.7.3), parsed --------------------------------------------------------

def _spec2_words() -> tuple[list[str], list[str]]:
    text = _section(r"1\.7\.3 ", SPEC_TEXT)
    sub = text[text.index("as a substring of the folded segment"):text.index(
        "as a whole folded token")]
    whole = text[text.index("as a whole folded token"):text.index("The total stays")]
    return _ticks(sub), _ticks(whole)


SUBSTRING_WORDS, WHOLE_WORDS = _spec2_words()
WORDS = SUBSTRING_WORDS + WHOLE_WORDS


def test_spec2_lists_25_substring_and_11_whole_token_words():
    assert (len(SUBSTRING_WORDS), len(WHOLE_WORDS)) == (25, 11)
    assert list(bundle.SUBSTRING_WORDS) == SUBSTRING_WORDS
    assert list(bundle.WHOLE_TOKEN_WORDS) == WHOLE_WORDS


# §1.6.3's cases and suffixes; also the SA-6 probe's suffix axis (§4.1 item 3)
CASES = [lambda w: w.upper(), lambda w: w.lower(), lambda w: "my" + w[0].upper() + w[1:]]
SUFFIXES = ["", "2", "_2", "1"]


def _probe_words():
    return [c(w) + s for w in WORDS for c in CASES for s in SUFFIXES]


def test_sa6_name_probe_every_word_case_and_suffix_fails_the_name_rule():
    names = _probe_words()
    assert len(names) == 36 * 3 * 4
    assert [n for n in names if not bundle.name_fails(n)] == []
    for n in names[:: 7]:  # and so rejects a bindValue, through the record
        with pytest.raises(bundle.BundleError):
            bundle.validate(_bind_bundle("5173", "svc." + n))


@pytest.mark.parametrize(("name", "fails"), [
    ("port", False), ("path", False), ("ping", False), ("shopping", False),
    ("asphalt", False), ("HTTPServer", False), ("upward_limit", True), ("cfg", False),
    ("adminpw", True), ("ADMIN_PIN1", True), ("PIN1", True), ("GithubPat2", True),
    ("PRİVATE", True), ("rootpin", False), ("githubpat", False), ("keycloak", True),
])
def test_name_rule_cases_from_spec2_1_7_3(name, fails):
    assert bundle.name_fails(name) is fails


# --- the token screen (§1.7.1), examples parsed ------------------------------------------

def _t6_examples() -> tuple[list[str], list[str]]:
    text = _section(r"1\.7\.1 ", SPEC_TEXT)
    t6 = text[text.index("**T6"):text.index("**Scope of the screen")]
    fail_part, pass_part = t6.split("So these fail:")[1].split("pass.")[0].rsplit("\n\n", 1)
    fails = [t for t in _ticks(fail_part) if len(t) > 3]
    passes = [t for t in _ticks(pass_part) if len(t) > 3]
    return fails, passes


def test_t6_examples_from_the_spec():
    fails, passes = _t6_examples()
    assert len(fails) == 4 and len(passes) == 4
    assert [t for t in fails if not bundle.screen_fails(t)] == []
    assert [t for t in passes if bundle.screen_fails(t)] == []


@pytest.mark.parametrize(("text", "fails"), [
    ("abcdefgh12345678", True),                     # T1: 16, letter and digit
    ("abcdefgh1234567", False),
    ("abcdefghijklmnopqrstuvwx", True),             # T2: 24
    ("abcdefghijklmnopqrstuvw", False),
    ("0123456789ab", True),                         # T3: 12 hex
    ("0123456789a", False),
    ("x550e8400-e29b-41d4-a716-446655440000", True),  # T4 anywhere
    ("ghp_abcdefghijklmnop", True),                 # T5: 20 chars, prefix at start
    ("ghp_abcdefghijklmno", False),                 # 19
    ("token=ghp_abcdefghijklmnop", True),           # T5 after '='
    ("sk-SE", False), ("pkg", False), ("skus", False),
    ("x" * 23 + "-" + "y" * 23, False),             # runs, not whole strings (non-hex)
    ("a" * 23 + "-" + "b" * 23, True),              # ... but T6: 46 hex digits once joined
])
def test_token_screen_rules(text, fails):
    assert bundle.screen_fails(text) is fails


# --- the bindValue grammar (§1.7.2) --------------------------------------------------------

LABEL63 = "-".join(["x" * 20, "x" * 20, "x" * 21])  # passes the screen: no run >= 24, no hex


@pytest.mark.parametrize(("value", "ok"), [
    ("65535", True), ("65536", False), ("0", False), ("08000", False), ("99999", False),
    ("localhost", True), ("host.docker.internal", True), ("127.0.0.1", True),
    ("256.0.0.1", False), ("01.2.3.4", False), ("falkordb", True), ("123456", False),
    ("db.internal", True), ("db.c0m", False), ("-db", False), ("db-", False),
    ("localhost:8000", True), ("falkordb:6379", True), ("[::1]:8000", True),
    ("localhost:0", False), ("http://localhost:8000/mcp", True),
    ("http://localhost:8000/mcp/", True), ("redis://falkordb:6379", True),
    ("ftp://localhost/", False), ("http://localhost/a/b", False),
    ("http://LOCALHOST/", False), ("http://[::1]/", True), ("http://localhost:8000\\abc", False),
    ("Localhost", False), (LABEL63, True), (LABEL63 + "x", False),  # label <= 63
    ("a" * 24, False),                              # a label is screened (T2)
    (".".join([LABEL63] * 3 + ["x" * 20 + "-" + "x" * 20 + "-" + "x" * 15, "com"]),
     True),                                                          # 253 characters
    (".".join([LABEL63] * 3 + ["x" * 20 + "-" + "x" * 20 + "-" + "x" * 16, "com"]),
     False),                                                         # 254: over 253
])
def test_bind_value_grammar(value, ok):
    assert bundle.bind_value_allowed("server.port", value) is ok


def test_w1_argv_arrays_never_bind():
    for kp in ("services.app.command[2]", "mcpServers.gh.args[3]", "x.entrypoint[0]"):
        assert bundle.bind_value_allowed(kp, "localhost") is False
    for kp in ("services.app.command", "x.args.y", "x.arguments[0]"):
        assert bundle.bind_value_allowed(kp, "localhost") is True


# --- §1.6.3: the argv coverage probe, verbatim ---------------------------------------------

V = "hunter2"
PLACEMENTS = [  # §1.6.3 placements 1–8
    lambda W: ["--" + W, V], lambda W: ["--" + W + "=" + V], lambda W: ["-" + W, V],
    lambda W: ["-" + W + "=" + V], lambda W: ["-e", W + "=" + V], lambda W: [W + "=" + V],
    lambda W: ["--x." + W, V], lambda W: ["--x." + W + "=" + V],
]
SEPARATOR_PLACEMENTS = [  # §1.6.3 placements 9–11, the separator axis (v1.3)
    lambda W: [W + ":" + V], lambda W: ["--header", W + ":" + V],
    lambda W: ["--header=" + W + ":" + V],
]
PREFIXES = [[], ["--auth-enabled"], ["--log4jConfigurationFile1"], ["-Xmx2g"], ["-q7zz"]]


def _probe_counts() -> dict[str, int]:
    """§1.6.3's pass-condition formulas: check -> cell count, parsed from the text.

    Each formula reads ``a × b × … = **N**`` (or ``= N``); the product of its factors must equal
    the stated N, so a slipped factor or a slipped total in the spec is itself a failure.
    """
    text = _section(r"1\.6\.3 ")
    keys = {"grid": "the grid", "separator": "the separator axis",
            "allow-list": "**The allow-list prefix grid**", "transplant": "**BS-M1's transplant**"}
    found = {}
    for name, marker in keys.items():
        start = text.index(marker)
        m = re.search(r"((?:\d[\d,]*\s*×\s*)+\d[\d,]*)\s*=\s*\*{0,2}(\d[\d,]*)\*{0,2}",
                      text[start:].replace("\n", " "))
        assert m, name
        factors = [int(f.replace(",", "")) for f in re.split(r"\s*×\s*", m.group(1))]
        total = int(m.group(2).replace(",", ""))
        assert math.prod(factors) == total, (name, factors, total)
        found[name] = total
    return found


def _leaks(placements, needle=V) -> tuple[int, list]:
    cells, leaks = 0, []
    for W in _probe_words():
        for p in placements:
            for pre in PREFIXES:
                argv = ["x", *pre, *p(W)]
                out = bundle.gate_argv(argv)
                cells += 1
                if any(needle in el for el in out):
                    leaks.append(argv)
                # every written cell is itself valid: the validator's fixed point
                assert bundle.gate_argv(out) == out, (argv, out)
    return cells, leaks


def test_probe_grid_has_no_value_in_clear():
    cells, leaks = _leaks(PLACEMENTS)
    assert cells == _probe_counts()["grid"]
    assert leaks == []


def test_probe_separator_axis_has_no_value_in_clear():
    cells, leaks = _leaks(SEPARATOR_PLACEMENTS)
    assert cells == _probe_counts()["separator"]
    assert leaks == []


def _transplant_flag(key_path: str) -> str:
    """'the flag name taken from the keyPath's last segment (`command` for an index)'."""
    last = re.split(r"\.", re.sub(r"\[\d+\]$", "", key_path))[-1]
    return last


def test_probe_bs_m1_transplant_leaves_only_the_bare_positional_residual():
    expected_cells = _probe_counts()["transplant"]
    # §1.6.3 names the one residual: "only BV-11's `4821`, as a bare positional"; the leak count
    # (BV-11 under two prefixes) is observed behaviour, not stated by the spec.
    expected_leaks = 2
    cells, leaks = 0, []
    for i in range(6, 21):
        key_path, _, value, binds = _bv(f"BV-{i}")
        assert not binds
        W = _transplant_flag(key_path)
        for col, place in enumerate(PLACEMENTS + [None]):
            for pre in PREFIXES:
                if place is None:
                    tail = [value]
                else:
                    tail = [value if el == V else el.replace(V, value) for el in place(W)]
                out = bundle.gate_argv(["x", *pre, *tail])
                cells += 1
                if any(value in el for el in out):
                    leaks.append((f"BV-{i}", "positional" if place is None else col, pre))
    assert cells == expected_cells
    assert all(c == "positional" for _, c, _ in leaks), leaks
    assert len(leaks) == expected_leaks
    assert {b for b, _, _ in leaks} == {"BV-11"}
    assert _transplant_flag("services.app.command[2]") == "command"


# --- §1.6.1 rules the tables do not isolate -----------------------------------------------

def test_gate_keeps_sentinels_as_fixed_points():
    assert bundle.gate_argv(["x", "--<redacted>", "<redacted>"]) == [
        "x", "--<redacted>", "<redacted>"]
    assert bundle.gate_argv(["x", "-<redacted>", "8000"]) == ["x", "-<redacted>", "<redacted>"]
    assert bundle.gate_argv(["x", "<redacted>", "8000"]) == ["x", "<redacted>", "8000"]


def test_validator_rejects_clear_element_after_a_forcing_sentinel():
    with pytest.raises(bundle.BundleError, match="args"):
        bundle.validate(_lf_bundle(["x", "-<redacted>", "8000"]))
    bundle.validate(_lf_bundle(["x", "<redacted>", "8000"]))


def test_command_is_gated_too():
    assert bundle.gate_argv(["exec $CMD"]) == ["<redacted>"]
    assert bundle.gate_argv(["ghp_16C7e42F292c6912E7710c838347Ae178B4a"]) == ["<redacted>"]
    with pytest.raises(bundle.BundleError, match="command"):
        bundle.validate(_lf_bundle(["exec $CMD"]))


def test_flag_longer_than_128_characters_is_redacted():
    flag = "--" + "a-" * 64
    assert bundle.gate_argv(["x", flag, "v"])[1] == "--<redacted>"


# --- launchFact.default (§1.6) -------------------------------------------------------------

def _env(**f):
    r = {"type": "launchFact", "path": P_PKG, "line": 9, "ordinal": 0, "kind": "env-default",
         "envVar": "AGENT_TEAM_PORT", "defaultHash": H64, "origin": "env-default", **f}
    return reid(r)


@pytest.mark.parametrize(("fields", "ok"), [
    ({"default": "8200"}, True),
    ({}, True),                                          # withheld: defaultHash still there
    ({"default": "hunter"}, True),                       # single label: W2 is the writer's
    ({"default": "x7k2q9w4m1p8z3v6"}, False),            # fails the grammar's screen
    ({"default": "a b"}, False),
    ({"envVar": "ROOT_PASSWORD", "default": "8200"}, False),   # envVar fails the name rule
    ({"envVar": "ROOT_PASSWORD"}, True),                 # no default: envVar not judged
])
def test_launch_fact_default_only_when_gates_pass(fields, ok):
    recs = ordered(counted(rich_records() + [_env(**fields)]))
    if ok:
        bundle.validate(recs)
    else:
        with pytest.raises(bundle.BundleError, match="default"):
            bundle.validate(recs)


# --- §1.1 ASCII IDENT and the kw head ------------------------------------------------------

@pytest.mark.parametrize(("arg", "ok"), [
    ("a1.café", False), ('a1["café"]', True), ('kw["café"]', True), ('kw["prefix"]', False),
    # §1.1's own depth example kw.headers["a"]["b"]["c"] is non-canonical under its one-
    # encoding rule (`a` is an IDENT, so `.a`); non-IDENT keys show the same count.
    ("kw.prefix", True), ('kw.headers["a-1"]["b-1"]["c-1"]', True),
    ('kw.headers["a-1"]["b-1"]["c-1"]["d-1"]', False), ("kw.headers.a.b.c", True),
    ("kw.headers.a.b.c.d", False), ('kw.headers["a"]', False),
    ("a1.$x", True), ("a1._y", True), ("kw", False), ("kw.", False),
])
def test_access_path_ascii_ident_and_kw_head(arg, ok):
    recs, rec = _replace(rich_bundle(), "valueSet", arg=arg)
    reid(rec)
    if ok:
        bundle.validate(ordered(recs))
    else:
        with pytest.raises(bundle.BundleError):
            bundle.validate(ordered(recs))


# --- §1.2 nameNorm --------------------------------------------------------------------------

def test_name_norm_is_recomputed():
    bad, _ = _replace(rich_bundle(), "symbol", 1, nameNorm="apifetchx")
    with pytest.raises(bundle.BundleError, match="nameNorm"):
        bundle.validate(ordered(bad))


@pytest.mark.parametrize("name", ["\u01f0x", "\u0390", "Stra\u00dfe", "\u0130stanbul"])
def test_name_norm_formula_is_nfc_casefold_nfc_strip(name):
    """§1.2: for U+01F0 and U+0390, casefold of an NFC string is not NFC; the outer NFC is
    what keeps the record valid under §8.1."""
    import unicodedata as ud
    norm = ud.normalize("NFC", ud.normalize("NFC", name).casefold()).strip()
    qname = P_CLIENT + "::" + name
    sym = symbol_rec(ids.symbol_id(LANG, REPO, qname), qname, "function", name, 20, 30,
                     S_MOD, nameNorm=norm)
    bundle.validate(ordered(counted(rich_records() + [sym])))
    if norm != name.casefold():  # the plain-casefold reading is rejected where it differs
        bad = {**sym, "nameNorm": name.casefold()}
        with pytest.raises(bundle.BundleError):
            bundle.validate(ordered(counted(rich_records() + [bad])))


# --- §1.4 controlledBy ---------------------------------------------------------------------

def test_controlled_by_cond_to_cond_is_allowed_but_not_a_self_edge():
    cond2 = reid({"type": "cond", "symbolId": S_API, "ordinal": 1, "line": 116, "code": "x"})
    ok = reid({"type": "controlledBy", "point": cond2["id"], "cond": COND_API})
    bundle.validate(ordered(counted(rich_records() + [cond2, ok])))
    self_edge = reid({"type": "controlledBy", "point": COND_API, "cond": COND_API})
    with pytest.raises(bundle.BundleError, match="point"):
        bundle.validate(ordered(counted(rich_records() + [self_edge])))


# --- gaps found by the BS2 mutation run ----------------------------------------------------

@pytest.mark.parametrize(("argv", "written"), [
    # (v1): '[' and ']' only when the whole value is a hostport or url
    (["x", "--opt", "a[1]"], ["x", "--opt", "<redacted>"]),
    (["x", "--opt", "[abc]"], ["x", "--opt", "<redacted>"]),
    (["x", "--host", "[::1]:8000"], ["x", "--host", "[::1]:8000"]),
    (["x", "http://[::1]:8000/mcp"], ["x", "http://[::1]:8000/mcp"]),
])
def test_v1_brackets_only_in_a_hostport_or_url(argv, written):
    assert bundle.gate_argv(argv) == written


def test_class4_both_readings_must_pass():
    """§1.6.1 class 4: the attached reading (`-p` + `w.example.com`, a dnsname, passes (v6))
    and the long-option reading (`pw.example.com`, which fails on `pw`) must both pass."""
    assert bundle.gate_argv(["x", "-pw.example.com", "v"]) == ["x", "-<redacted>", "<redacted>"]
    assert bundle.gate_argv(["x", "-p8000"]) == ["x", "-p8000"]


@pytest.mark.parametrize("entry", ["-XX:+Token", "-XX:-PasswordFile", "-XX:+A1b2C3d4E5f6G7h8"])
def test_allow_listed_shape_must_itself_pass_screen_and_name_rule(entry):
    """§1.6.1 class 2: 'An entry is a flag only if it also passes the token screen, and every
    part passes §1.7.3. So `-XX:+Token` … [is] class 4 and get[s] `-<redacted>`'."""
    assert bundle.gate_argv(["x", entry, "v"]) == ["x", "-<redacted>", "<redacted>"]


@pytest.mark.parametrize(("value", "clear"), [
    ("app:hunter2@db", False),        # (v4) alone: no '://', and (v8)'s head `app` passes
    ("some-mcp@latest", True), ("@scope/pkg", True), ("git@host:org/repo", True),
])
def test_v4_no_colon_before_an_at(value, clear):
    assert (bundle.gate_argv(["x", "--opt", value])[2] == value) is clear


@pytest.mark.parametrize(("text", "fails"), [
    # T4 alone: the trailing `z` breaks T3's 12-hex run and T6's all-hex core, no T1/T2
    ("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaaz", True),
    ("aaaaaaaa-aaaa-aaaa-aaa-aaaaaaaaaaaaz", False),   # not 8-4-4-4-12
])
def test_t4_uuid_shape_decides_alone(text, fails):
    assert bundle.screen_fails(text) is fails


@pytest.mark.parametrize("path", ["kw", "kw.", 'kw["x"', "kwx"])
def test_kw_head_needs_its_field_in_bundle_py_alone(path):
    """BS2 §1.12: bundle.py is the reference validator in its own right, not only behind the
    schema (which also requires `kw` plus one field)."""
    with pytest.raises(bundle.BundleError):
        bundle._path_head(1, path, "arg")


# --- T6's separator class, one separator at a time (§1.7.1 T6, widened by P3-2) ------------
# Each chunked token has 4-character runs (no T1–T3), so only T6's join can fail it; each
# legitimate value has the same separator and stays under T6's thresholds once joined.
T6_SEPARATORS = [
    (".", "x7k2.q9w4.m1p8.z3v6", "falkorchat.app"),
    ("_", "x7k2_q9w4_m1p8_z3v6", "s3_bucket_v2"),
    (":", "x7k2:q9w4:m1p8:z3v6", "falkorchat.app:app"),
    ("+", "x7k2+q9w4+m1p8+z3v6", "c+plus+plus"),
    (",", "x7k2,q9w4,m1p8,z3v6", "a1,b2,c3"),
    ("-", "x7k2-q9w4-m1p8-z3v6", "release-2024-notes"),
]


def test_t6_separator_table_matches_the_spec_class():
    text = _section(r"1\.7\.1 ", SPEC_TEXT)
    removed = re.search(r"with (.*?) removed, leaves", text, re.S).group(1)
    assert sorted(_ticks(removed)) == sorted(s for s, _, _ in T6_SEPARATORS)


@pytest.mark.parametrize(("sep", "chunked", "legit"), T6_SEPARATORS,
                         ids=[repr(s) for s, _, _ in T6_SEPARATORS])
def test_t6_joins_across_each_separator(sep, chunked, legit):
    assert bundle.screen_fails(chunked) is True
    assert bundle.screen_fails(legit) is False
    for piece in chunked.split(sep):  # the pieces alone pass: T6's join is what decides
        assert bundle.screen_fails(piece) is False


def test_comma_chunked_token_is_a_reachable_argv_value_and_is_redacted():
    """`,` is in (v1)'s value class, so the comma-chunked token reaches the screen in argv."""
    assert bundle.gate_argv(["x", "--opt", "x7k2,q9w4,m1p8,z3v6"]) == ["x", "--opt", "<redacted>"]
    assert bundle.gate_argv(["x", "--opt", "a1,b2,c3"]) == ["x", "--opt", "a1,b2,c3"]


# --- §1.6.1's allow-list prefix grid (BS-P3-m2): every entry in front of the 11 placements --
# The closed shapes have no single string; these representatives were fixed by teco.
ALLOW_LIST_ENTRIES = ["-jar", "-cp", "-classpath", "-port", "-addr", "-host", "-listen",
                      "-bind", "-ea", "-da", "-server", "-client",
                      "-Xmx2g", "-Xms512m", "-Xss1k", "-XX:+UseG1GC", "-XX:-UseG1GC",
                      "-verbose:gc"]


def test_allow_list_entries_are_the_spec_representatives():
    """§1.6.3's allow-list prefix grid names its 18 representatives."""
    text = _section(r"1\.6\.3 ")
    bullet = text[text.index("**The allow-list prefix grid**"):]
    bullet = bullet[bullet.index("The entries are"):bullet.index("which stands for")]
    assert _ticks(bullet) == ALLOW_LIST_ENTRIES
    assert len(ALLOW_LIST_ENTRIES) == 18


def test_allow_list_entries_are_flags():
    for e in ALLOW_LIST_ENTRIES:
        assert bundle.gate_argv(["x", e]) == ["x", e], e


def test_allow_list_prefix_grid_has_no_value_in_clear():
    cells, leaks = 0, []
    for entry in ALLOW_LIST_ENTRIES:
        for W in _probe_words():
            for place in PLACEMENTS + SEPARATOR_PLACEMENTS:
                argv = ["x", entry, *place(W)]
                out = bundle.gate_argv(argv)
                cells += 1
                if any(V in el for el in out):
                    leaks.append(argv)
    assert cells == _probe_counts()["allow-list"]
    assert leaks == []


# === BS10-P2-M1: freeze-bearing lists, shapes and thresholds, probed both ways ===============

def _class2_text() -> str:
    text = _section(r"1\.6\.1 ", SPEC_TEXT)
    return text[text.index("**exact strings:**"):text.index("An entry is a flag only if")]


def _spec_allow_exact() -> list[str]:
    t = _class2_text()
    return _ticks(t[:t.index("**closed shapes:**")])


def _spec_allow_shapes() -> list[str]:
    t = _class2_text()
    return _ticks(t[t.index("**closed shapes:**"):])


def test_allow_list_exact_strings_are_the_spec_list():
    exact = _spec_allow_exact()
    assert len(exact) == 12
    assert set(exact) == set(bundle._ALLOW_EXACT)


@pytest.mark.parametrize("entry", _spec_allow_exact())
def test_each_exact_allow_list_entry_is_a_flag_and_its_near_misses_are_not(entry):
    assert bundle.gate_argv(["x", entry]) == ["x", entry]
    # an entry supplies its name as context exactly as `--NAME` does (§1.6.1 point 3)
    for v in ("8080", "app.jar", "a1,b2"):
        assert (bundle.gate_argv(["x", entry, v])[2]
                == bundle.gate_argv(["x", "--" + entry[1:], v])[2]), (entry, v)
    for miss in (entry + "x", entry.upper(), entry[:2].upper() + entry[2:]):
        if miss != entry and miss not in bundle._ALLOW_EXACT:
            assert bundle.gate_argv(["x", miss, "v"])[1] != miss, miss


@pytest.mark.parametrize("miss", ["-debug", "-verbose", "-Xdebug", "-version", "-help"])
def test_single_dash_words_outside_the_allow_list_are_not_flags(miss):
    assert bundle.gate_argv(["x", miss])[1] == "-<redacted>"


def _verbose_words() -> list[str]:
    return [m.group(1) for t in _spec_allow_shapes()
            if (m := re.fullmatch(r"-verbose:(\w+)", t))]


def _xmem_prefixes() -> list[str]:
    return [m.group(1) for t in _spec_allow_shapes()
            if (m := re.fullmatch(r"-(X\w\w)<n>", t))]


def test_closed_shapes_are_parsed_from_the_spec():
    assert _verbose_words() == ["gc", "class", "jni", "module"]
    assert _xmem_prefixes() == ["Xmx", "Xms", "Xss"]
    assert {"-XX:+Name", "-XX:-Name"} <= set(_spec_allow_shapes())


@pytest.mark.parametrize("word", ["gc", "class", "jni", "module"])
def test_each_verbose_word_is_a_flag(word):
    assert word in _verbose_words()
    assert bundle.gate_argv(["x", f"-verbose:{word}"]) == ["x", f"-verbose:{word}"]


@pytest.mark.parametrize("miss", ["-verbose:foo", "-verbose:GC", "-verbose:gcx", "-Verbose:gc"])
def test_verbose_near_misses_are_not_flags(miss):
    assert bundle.gate_argv(["x", miss])[1] == "-<redacted>"


@pytest.mark.parametrize("prefix", ["Xmx", "Xms", "Xss"])
def test_memory_shapes_admit_digits_and_one_unit_in_either_case(prefix):
    assert prefix in _xmem_prefixes()
    for n in ("2", "512", "2g", "2G", "512m", "512M", "1k", "1K"):
        assert bundle.gate_argv(["x", f"-{prefix}{n}"]) == ["x", f"-{prefix}{n}"], n
    for miss in (f"-{prefix}", f"-{prefix}2t", f"-{prefix}2gg", f"-{prefix.lower()}2g",
                 f"-{prefix}g"):
        assert bundle.gate_argv(["x", miss])[1] == "-<redacted>", miss


@pytest.mark.parametrize("sign", ["+", "-"])
def test_xx_shape_name_is_capitalised_and_case_sensitive(sign):
    assert bundle.gate_argv(["x", f"-XX:{sign}UseG1GC"]) == ["x", f"-XX:{sign}UseG1GC"]
    for miss in (f"-XX:{sign}useG1GC", f"-xx:{sign}UseG1GC", f"-Xx:{sign}UseG1GC",
                 f"-XX:{sign}1UseG1GC", f"-XX:{sign}"):
        assert bundle.gate_argv(["x", miss])[1] == "-<redacted>", miss


def _spec_t5_prefixes() -> list[str]:
    text = _section(r"1\.7\.1 ", SPEC_TEXT)
    return _ticks(text[text.index("case-sensitive prefixes:"):text.index("**T6")])


def test_t5_prefix_list_is_the_spec_list():
    prefixes = _spec_t5_prefixes()
    assert len(prefixes) == 23
    assert set(prefixes) == set(bundle._T5_PREFIXES)


@pytest.mark.parametrize("prefix", _spec_t5_prefixes())
def test_each_t5_prefix_fails_at_20_characters_only(prefix):
    """A letters-only filler (no hex letters, so T1/T3/T6 cannot fire; < 24, so not T2)."""
    filler20, filler19 = "q" * (20 - len(prefix)), "q" * (19 - len(prefix))
    assert bundle.screen_fails(prefix + filler20) is True
    assert bundle.screen_fails(prefix + filler19) is False      # T5 needs >= 20 characters
    assert bundle.screen_fails("q" * 20) is False               # the filler alone passes
    assert bundle.screen_fails("x=" + prefix + filler20) is True  # after '='
    assert bundle.screen_fails(prefix.swapcase() + filler20) is (prefix.swapcase() in
                                                                bundle._T5_PREFIXES)


def _core(s: str) -> str:
    return re.sub(r"[._:+,-]", "", s)


def _trans(core: str) -> int:
    return sum(1 for a, b in zip(core, core[1:]) if a.isdigit() != b.isdigit())


@pytest.mark.parametrize(("text", "transitions", "fails"), [
    ("qqq1-qqq1-qqq1-qqqq", 6, True),       # 16-character core, exactly 6 transitions
    ("qqq1-qqq1-qqqq-qqq1", 5, False),      # 16-character core, 5 transitions
])
def test_t6_transition_threshold_is_six(text, transitions, fails):
    assert (len(_core(text)), _trans(_core(text))) == (16, transitions)
    assert all(bundle.screen_fails(p) is False for p in text.split("-"))
    assert bundle.screen_fails(text) is fails


@pytest.mark.parametrize(("text", "fails"), [
    ("abcd-ef01-2345-6789", True),          # 16 hex once joined
    ("abcd-ef01-2345-678", False),          # 15 hex
])
def test_t6_hex_core_threshold_is_sixteen(text, fails):
    assert all(c in "0123456789abcdef" for c in _core(text))
    assert bundle.screen_fails(text) is fails


def _letters(n: int) -> str:
    """n characters, letters and dots only: passes the screen (non-hex letters, runs < 24)."""
    return (("q" * 10 + ".") * 12)[:n - 1] + "q"


@pytest.mark.parametrize(("n", "clear"), [(128, True), (129, False)])
def test_value_length_cap_is_128(n, clear):
    v = _letters(n)
    assert len(v) == n and not bundle.screen_fails(v)
    assert (bundle.gate_argv(["x", "--opt", v])[2] == v) is clear
    assert (bundle.gate_argv([v])[0] == v) is clear              # command: (v1) too


@pytest.mark.parametrize(("n", "clear"), [(128, True), (129, False)])
def test_flag_length_cap_is_128(n, clear):
    flag = "--" + _letters(n - 2)
    assert len(flag) == n and not bundle.screen_fails(flag)
    assert (bundle.gate_argv(["x", flag])[1] == flag) is clear


def test_empty_host_never_binds():
    assert bundle.bind_value_allowed("server.port", ":8080") is False
    assert bundle.bind_value_allowed("server.port", "localhost:8080") is True
    assert bundle.gate_argv(["x", "-p", ":8080"])[2] == "<redacted>"  # (v6) needs a hostport


def test_assignment_name_that_fails_only_the_screen_is_redacted():
    name = "x7k2q9w4m1p8z3v6"           # T1, but no name-rule word
    assert bundle.screen_fails(name) and not bundle.name_fails(name)
    assert bundle.gate_argv(["x", name + "=1"]) == ["x", "<redacted>"]
    assert bundle.gate_argv(["x", "mode=1"]) == ["x", "mode=1"]


def test_single_dash_element_is_screened_whole():
    """Both class-4 readings pass (`-h` + a per-label-clean dnsname; the long name has no
    word), so only the whole-element screen (T6 across the dots) decides."""
    el = "-hx7k2.q9w4.m1p8.z3v6.io"
    assert bundle.gate_argv(["x", "-h", el[2:]])[2] == el[2:]      # the attached reading
    assert not bundle.name_fails(el[1:])                             # the long reading
    assert bundle.gate_argv(["x", el]) == ["x", "-<redacted>"]


# === BS10-P2-m2: argv[0] supplies no context (today's reading, pinned) ====================

def test_argv0_supplies_no_context():
    """§1.6.1 leaves open whether argv[0] supplies context ('`args` is read left to right');
    the ambiguity is routed to -spec3. Today's reading: it does not, so `htpasswd` (which
    contains `pass`) does not redact `admin`. The positional `hunter2` is §1.6.1's named
    unflagged-positional residual."""
    assert bundle.gate_argv(["htpasswd", "admin", "hunter2"]) == ["htpasswd", "admin", "hunter2"]
    assert bundle.gate_argv(["x", "htpasswd", "admin"]) == ["x", "htpasswd", "<redacted>"]


# === spec4 §1: header.skippedFiles (inside bundleSchemaVersion "1.0") =========================

def _skipped_bundle(skipped, **hdr) -> list[dict]:
    """The rich bundle (unit rootPath ``salesperson``) with ``header.skippedFiles`` set."""
    return ordered(counted(rich_records(), skippedFiles=skipped, **hdr))


def _sk(path, reason="test") -> dict:
    return {"path": path, "reason": reason}


def _skipped_reasons_spec() -> list[str]:
    """spec4 §1's `reason` bullet: "closed enum `test` | `frontend-ignore`"."""
    sec = _section(r"1\. Rule 4", SPEC4_TEXT)
    m = re.search(r"\*\*`reason`:\*\* closed enum ([^.;]*)[.;]", sec)
    assert m
    return _ticks(m.group(1))


def _skipped_reason_enum() -> list[str]:
    items = _resolve(_def("header")["properties"]["skippedFiles"])["items"]
    return _resolve(items["properties"]["reason"])["enum"]


def test_spec4_skipped_reasons_are_the_two_codes():
    assert sorted(_skipped_reasons_spec()) == ["frontend-ignore", "test"]


def test_schema_skipped_file_reasons_match_spec4_registry():
    assert sorted(_skipped_reason_enum()) == sorted(_skipped_reasons_spec())


def test_skipped_files_reasons_stay_out_of_the_status_registry():
    """spec4 §1: "no `status`" - the codes are not record-level status reasons (§7.2)."""
    for rtype in SPEC_ORDER:
        reason = _def(rtype)["properties"].get("reason")
        codes = _resolve(reason)["enum"] if reason else []
        assert not set(codes) & set(_skipped_reasons_spec())


def test_skipped_files_is_optional_in_the_header_schema():
    assert "skippedFiles" not in _def("header")["required"]
    assert _def("header")["properties"]["bundleSchemaVersion"]["enum"] == ["1.0", "1.1"]     # spec4 kept "1.0"; spec5 §1 adds "1.1"


def test_bundle_without_skipped_files_is_valid():
    assert "skippedFiles" not in rich_bundle()[0]
    bundle.validate(rich_bundle())


def test_bundle_with_sorted_unique_skipped_files_is_valid():
    recs = _skipped_bundle([_sk("salesperson/a.test.ts"),
                            _sk("salesperson/src/i18n/x.ts", "frontend-ignore")])
    bundle.validate(recs)
    assert bundle.loads(to_bytes(recs)) == recs
    assert bundle.loads(bundle.dumps(recs)) == recs


def test_skipped_files_on_a_header_only_bundle_is_valid():
    bundle.validate([header(skippedFiles=[_sk("salesperson/a.test.ts")])])


def test_skipped_files_present_but_empty_is_rejected():
    with pytest.raises(bundle.BundleError, match="skippedFiles"):
        bundle.validate(_skipped_bundle([]))


@pytest.mark.parametrize("entry", [
    {"path": "salesperson/a.test.ts", "reason": "other"},
    {"path": "salesperson/a.test.ts", "reason": "vendored"},
    {"path": "salesperson/a.test.ts", "reason": ""},
    {"path": "salesperson/a.test.ts"},
    {"reason": "test"},
    {"path": "salesperson/a.test.ts", "reason": "test", "status": "possible"},
    {"path": 3, "reason": "test"},
    "salesperson/a.test.ts",
])
def test_skipped_file_entry_shape_is_closed(entry):
    with pytest.raises(bundle.BundleError, match="skippedFiles"):
        bundle.validate(_skipped_bundle([entry]))


@pytest.mark.parametrize(("path", "reason"), [("salesperson/a.test.ts", "test"),
                                              ("salesperson/a.ts", "frontend-ignore")])
def test_each_registry_reason_is_accepted(path, reason):
    bundle.validate(_skipped_bundle([_sk(path, reason)]))


def test_skipped_files_must_be_a_list():
    with pytest.raises(bundle.BundleError, match="skippedFiles"):
        bundle.validate([header(skippedFiles={"path": "salesperson/a.ts", "reason": "test"})])


def test_skipped_files_must_be_sorted_by_path():
    with pytest.raises(bundle.BundleError, match="skippedFiles.*sorted"):
        bundle.validate(_skipped_bundle([_sk("salesperson/b.test.ts"),
                                         _sk("salesperson/a.test.ts")]))


@pytest.mark.parametrize(("first", "second"), [
    ("salesperson/B.test.ts", "salesperson/a.test.ts"),   # byte order: uppercase first
    ("salesperson/a-b.test.ts", "salesperson/a/b.test.ts"),   # "-" (0x2d) < "/" (0x2f)
    ("salesperson/a.test.ts", "salesperson/a.test.tsx"),   # a prefix sorts first
])
def test_skipped_files_sort_is_byte_order_of_the_whole_path(first, second):
    bundle.validate(_skipped_bundle([_sk(first), _sk(second)]))
    with pytest.raises(bundle.BundleError, match="sorted"):
        bundle.validate(_skipped_bundle([_sk(second), _sk(first)]))


def test_skipped_files_paths_must_be_unique():
    with pytest.raises(bundle.BundleError, match="skippedFiles.*unique"):
        bundle.validate(_skipped_bundle([_sk("salesperson/a.test.ts"),
                                         _sk("salesperson/a.test.ts")]))


def test_skipped_files_uniqueness_is_by_path_not_by_entry():
    """A path listed under two reasons is a duplicate path (spec4 §1: "once, as `test`")."""
    with pytest.raises(bundle.BundleError, match="skippedFiles.*unique"):
        bundle.validate(_skipped_bundle([_sk("salesperson/a.ts", "frontend-ignore"),
                                         _sk("salesperson/a.ts", "test")]))


@pytest.mark.parametrize("path", [
    "/salesperson/a.test.ts", "salesperson//a.test.ts", "salesperson/", "./salesperson/a.ts",
    "salesperson/../a.ts", "salesperson/..", "",
])
def test_skipped_files_paths_must_be_repo_relative_posix(path):
    with pytest.raises(bundle.BundleError, match="skippedFiles"):
        bundle.validate(_skipped_bundle([_sk(path)]))


@pytest.mark.parametrize("path", [
    "other/a.test.ts", "salespersonX/a.test.ts", "salesperson", "a.test.ts",
])
def test_skipped_files_paths_must_be_under_root_path(path):
    with pytest.raises(bundle.BundleError, match="skippedFiles.*rootPath"):
        bundle.validate(_skipped_bundle([_sk(path)]))


def test_skipped_files_under_a_dot_root_may_be_any_path():
    recs = ordered(counted([], skippedFiles=[_sk("a.test.ts"), _sk("src/b.test.ts")],
                           unit={"repoId": REPO, "rootPath": ".", "lang": LANG},
                           unitId=ids.unit_id(REPO, ".", LANG)))
    bundle.validate(recs)


@pytest.mark.parametrize("path", [
    "salesperson/node_modules/x.js", "salesperson/src/node_modules/deep/x.js",
])
def test_skipped_files_must_not_be_under_vendored_dirs(path):
    with pytest.raises(bundle.BundleError, match="skippedFiles.*vendored"):
        bundle.validate(_skipped_bundle([_sk(path)], vendoredDirs=["node_modules"]))


def test_skipped_files_vendored_name_as_a_basename_is_not_vendored():
    bundle.validate(_skipped_bundle([_sk("salesperson/scripts/node_modules", "frontend-ignore")],
                                    vendoredDirs=["node_modules"]))


def test_skipped_file_must_have_no_file_record():
    assert any(r["type"] == "file" and r["path"] == P_CLIENT for r in rich_records())
    with pytest.raises(bundle.BundleError, match="skippedFiles.*file record"):
        bundle.validate(_skipped_bundle([_sk(P_CLIENT, "frontend-ignore")]))


def test_skipped_files_cannot_check_completeness():
    """spec4 §1: an absent source file that is not listed anywhere is not detectable."""
    bundle.validate(_skipped_bundle([_sk("salesperson/src/never-mentioned.test.ts")]))


def test_skipped_files_is_part_of_the_canonical_bytes():
    recs = _skipped_bundle([_sk("salesperson/a.test.ts")])
    line = bundle.canonical_line(recs[0])
    assert b'"skippedFiles":[{"path":"salesperson/a.test.ts","reason":"test"}]' in line


_S14A = Path(__file__).resolve().parents[1] / ".ccg-work/vertical-slice/s14a"


@pytest.mark.skipif(not _S14A.is_dir(), reason="stored s14a slice bundles absent")
def test_stored_s14a_slice_bundles_validate_with_no_skipped_files():
    """Positive control (spec4 §0): bundles written before the field stay valid, unchanged."""
    files = sorted(_S14A.glob("*.bundle.jsonl"))
    assert len(files) == 2
    for f in files:
        recs = bundle.read_bundle(f)
        assert "skippedFiles" not in recs[0]


# --- spec4 §1: the closed `test` pattern, decidable from one entry ----------------------------

def _spec4_test_pattern_text() -> str:
    m = re.search(r"\*\*`test` pattern.*?(?=\n- \*\*Precedence)", _section(r"1\. Rule 4", SPEC4_TEXT),
                  re.S)
    assert m
    return m.group(0)


def _spec_js_exts() -> list[str]:
    row = [r for r in _table_rows(_section(r"3\.3 ")) if _ticks(r[1]) == [LANG]][0]
    return [t.lstrip(".") for t in _ticks(row[2])]


def test_spec4_test_pattern_is_what_the_validator_implements():
    text = _spec4_test_pattern_text()
    for needle in ("*.test.<ext>", "*.spec.<ext>", "`e2e`", "`__tests__`", "JS/TS units only"):
        assert needle in text
    assert sorted(bundle.JS_TEST_EXTENSIONS) == sorted(_spec_js_exts())


@pytest.mark.parametrize("path", [
    "salesperson/a.test.ts", "salesperson/src/a.spec.tsx", "salesperson/a.test.mjs",
    "salesperson/.test.js", "salesperson/e2e/login.ts", "salesperson/src/__tests__/x.js",
    "salesperson/x/e2e/y/z.ts",
])
def test_frontend_ignore_on_a_test_pattern_path_is_rejected(path):
    """Precedence (spec4 §1): a file that is both is listed once, as `test`."""
    bundle.validate(_skipped_bundle([_sk(path, "test")]))
    with pytest.raises(bundle.BundleError, match="skippedFiles.*test"):
        bundle.validate(_skipped_bundle([_sk(path, "frontend-ignore")]))


@pytest.mark.parametrize("path", [
    "salesperson/a.ts", "salesperson/a.test.py", "salesperson/a.test", "salesperson/atest.ts",
    "salesperson/a.tests.ts", "salesperson/a.test.d.ts.map", "salesperson/e2e",
    "salesperson/e2e.ts", "salesperson/src/e2e-utils/a.ts", "salesperson/__tests__.js",
    "salesperson/src/tests/a.ts", "salesperson/a.test/b.ts",
])
def test_test_entry_that_does_not_match_the_pattern_is_rejected(path):
    bundle.validate(_skipped_bundle([_sk(path, "frontend-ignore")]))
    with pytest.raises(bundle.BundleError, match="skippedFiles.*test"):
        bundle.validate(_skipped_bundle([_sk(path, "test")]))


def test_test_pattern_directory_segments_are_below_root_path():
    """spec4 §1: "a directory segment below `rootPath`" - the root's own name does not count."""
    root = "e2e"
    recs = ordered(counted([], skippedFiles=[_sk("e2e/a.ts", "frontend-ignore")],
                           unit={"repoId": REPO, "rootPath": root, "lang": LANG},
                           unitId=ids.unit_id(REPO, root, LANG)))
    bundle.validate(recs)
    recs[0]["skippedFiles"] = [_sk("e2e/a.ts", "test")]
    with pytest.raises(bundle.BundleError, match="skippedFiles.*test"):
        bundle.validate(recs)


def _python_unit(skipped) -> list[dict]:
    return ordered(counted([], skippedFiles=skipped,
                           unit={"repoId": REPO, "rootPath": "svc", "lang": "python"},
                           unitId=ids.unit_id(REPO, "svc", "python")))


def test_a_non_js_ts_unit_has_no_test_pattern_so_no_test_entry():
    """spec4 §1: "any other language has none in v1" - so nothing there is `test`, and the
    JS/TS-only pattern never reclassifies a `frontend-ignore` entry."""
    bundle.validate(_python_unit([_sk("svc/test_a.py", "frontend-ignore"),
                                  _sk("svc/tests/a.test.ts", "frontend-ignore")]))
    with pytest.raises(bundle.BundleError, match="skippedFiles.*test"):
        bundle.validate(_python_unit([_sk("svc/a.test.ts", "test")]))
