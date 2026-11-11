"""P1-18b: one hostile or odd value fails ITS UNIT, not the system build (``-p1-linking.md`` P1-18b; ``ccg.indexed``).

Offline: the predicate, the property list, the assembler's per-bundle check (``bundle:unindexable``), the unit-only blast radius through
``build_rows`` + the loader (fake store) and through ``ccg.refresh`` on real git repositories, the loader's backstop, and the AST pins that
the loader and the assembler share ONE predicate. The engine half (a systematic grid against a throwaway server) is ``test_indexed_live.py``.
"""
from __future__ import annotations

import ast
import copy
import subprocess
import sys
from pathlib import Path

import pytest
from loader_helpers import FakeStore
from test_assemble_rows import HANDLER, SYSTEM, H, all_audit, build, file_rec, header, pack_dir, pack_def, prov, sym_rec, world
from test_refresh import SYSTEM as RF_SYSTEM
from test_refresh import ALL_UNITS, FILES, World, commit, make_target, uid

from ccg import indexed
from ccg import refresh as RF
from ccg.assemble import rows as R
from ccg.staleness import FAIL_REASONS
from ccg.store import loader as L
from ccg.store import parse_ddl

CPG = Path(__file__).resolve().parents[1]
WS = list(" \t\n\v\f\r")
CODE = "bundle:unindexable"


# ------------------------------------------------------------------------------------------------------------------ the predicate

@pytest.mark.parametrize("ch,width", [("k", 1), ("é", 2), ("中", 3), ("😀", 4)])
def test_the_bound_is_4096_utf8_bytes_not_characters(ch, width):
    ok = ch * (4096 // width) + "k" * (4096 % width)                             # exactly 4,096 bytes, multibyte up front
    assert len(ok.encode()) == 4096 and indexed.check(ok) is None
    over = ok + ch if width == 1 else ch * (4096 // width + 1)                   # 4,097 .. 4,100 bytes
    v = indexed.check(over)
    assert v is not None and (v.reason, v.limit) == (indexed.TOO_LONG, 4096) and v.size == len(over.encode()) > 4096
    assert indexed.check("k" * 4095) is None and indexed.check("k" * 4097).reason == indexed.TOO_LONG


@pytest.mark.parametrize("w", WS)
@pytest.mark.parametrize("shape", ["{w}x", "x{w}", "{w}x{w}", "{w}"])
def test_each_of_the_six_ascii_whitespace_chars_at_either_end_is_unindexable(w, shape):
    v = indexed.check(shape.format(w=w))
    assert v is not None and v.reason == indexed.WHITESPACE


@pytest.mark.parametrize("value", ["a b", "a\tb", "a\nb", "\u00a0x", "x\u2003", "\u3000x\u3000", "\u200bx", "x\u0085", "x\x1c", "x"])
def test_interior_ascii_whitespace_and_other_unicode_spaces_are_indexable(value):
    assert indexed.check(value) is None


def test_an_empty_string_is_unindexable_because_the_engine_never_matches_it():
    v = indexed.check("")
    assert v is not None and (v.reason, v.size) == (indexed.EMPTY, 0)


@pytest.mark.parametrize("value", ["a\ud800b", "\udfff", "x\ud800"])
def test_text_that_is_not_valid_unicode_is_unindexable_and_not_a_unicode_error(value):
    v = indexed.check(value)
    assert v is not None and v.reason == indexed.UNICODE and v.size == 0


@pytest.mark.parametrize("value", [5, 0, 1.5, True, b"x" * 9000, ["x"], {"k": " "}])
def test_a_non_string_value_is_refused_typed_and_value_free(value):
    v = indexed.check(value)
    assert v is not None and (v.reason, v.size) == (indexed.NOT_STR, 0) and repr(v) == "Unindexable(reason='not-a-string', size=0, limit=4096)"


def test_an_absent_value_is_not_the_predicates_business():
    assert indexed.check(None) is None


def test_the_verdict_never_carries_the_value():
    v = indexed.check("SECRET-" + "k" * 5000)
    assert "SECRET" not in repr(v) and set(v._fields) == {"reason", "size", "limit"}


# ------------------------------------------------------------------------------------------------------------------ the property list

def test_the_property_list_is_read_from_the_ddl_and_the_loader_uses_that_object():
    ddl = {(s[1], s[2]) for s in parse_ddl() if s[0] == "index"}
    assert indexed.INDEXED_PROPS == ddl and len(ddl) == 20
    assert L.INDEXED_PROPS is indexed.INDEXED_PROPS and L.INDEXED_VALUE_MAX_BYTES is indexed.INDEXED_VALUE_MAX_BYTES
    assert L.INDEX_TRIMMED_CHARS is indexed.INDEX_TRIMMED_CHARS and indexed.INDEX_TRIMMED_CHARS == frozenset(WS)
    for label, prop in ddl:
        assert prop in indexed.props_of(label)


def test_every_ddl_index_has_a_bundle_source_or_a_declared_exemption():
    mapped = {(lbl, p) for rows in indexed.BUNDLE_SOURCES.values() for lbl, p, _ in rows}
    mapped |= {indexed.HEADER_REPO, indexed.HEADER_SKIPPED, indexed.HEADER_SKIPPED_REPO}
    assert mapped | indexed.NOT_IN_A_BUNDLE == indexed.INDEXED_PROPS and not mapped & indexed.NOT_IN_A_BUNDLE


def test_a_new_ddl_index_nobody_mapped_fails_the_import_instead_of_going_unchecked(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(
        "import sys\nsys.path.insert(0, %r)\nimport ccg.store as S\nreal = S.parse_ddl\n"
        "S.parse_ddl = lambda t=None: real(t) + [('index', 'Symbol', 'brandNew', 'x')]\n"
        "try:\n    import ccg.indexed\nexcept ImportError as e:\n    print('REFUSED', e)\n" % str(CPG))
    out = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, timeout=60)
    assert "REFUSED" in out.stdout and "brandNew" in out.stdout, out.stdout + out.stderr


# ------------------------------------------------------------------------------------------------------------------ the assembler, per bundle

def _bundles():
    return copy.deepcopy(world())


def _plant(records, rtype, field, value, nth=0):
    hits = [r for r in records if r["type"] == rtype]
    hits[nth][field] = value
    return records


BAD = {"trailing-space": "svc/a.py ", "leading-tab": "\tsvc/a.py", "4100-bytes": "d/" + "p" * 4098, "lone-surrogate": "svc/a\ud800.py"}
#: every kind of value the gate must refuse in an indexed field: the four above, the empty string and a value that is not a string
BAD_KINDS = {**BAD, "empty": "", "not-a-string": 5, "list": ["svc/a.py"]}

#: THE expected map, a literal: (bundle record type, record field, label, indexed property). It is NOT read from ``indexed.BUNDLE_SOURCES`` (a check that
#: reads its expected rows from the table it checks cannot fail on a deleted row); the header's ``unit.repoId`` and ``skippedFiles[].path`` follow.
EXPECTED_SOURCES = [
    ("header", "unitId", "Unit", "unitId"),
    ("file", "id", "File", "fileId"), ("file", "path", "File", "path"),
    ("symbol", "id", "Symbol", "symbolId"), ("symbol", "nameNorm", "Symbol", "nameNorm"), ("symbol", "qname", "Symbol", "qname"),
    ("param", "id", "Param", "pointId"), ("ret", "id", "Ret", "pointId"), ("cond", "id", "Cond", "pointId"),
    ("callsite", "id", "CallSite", "callSiteId"), ("dependency", "depId", "Dependency", "depId"),
    ("configEntry", "id", "ConfigEntry", "entryId"), ("configEntry", "path", "ConfigEntry", "path"),
    ("launchFact", "id", "ConfigEntry", "entryId"), ("launchFact", "path", "ConfigEntry", "path"),
]


def test_the_bundle_source_table_is_exactly_the_literal_table():
    got = sorted((t, f, lbl, p) for t, rows in indexed.BUNDLE_SOURCES.items() for lbl, p, f in rows)
    assert got == sorted(EXPECTED_SOURCES) and len(EXPECTED_SOURCES) == 15
    assert (indexed.HEADER_REPO, indexed.HEADER_SKIPPED, indexed.HEADER_SKIPPED_REPO) == (("Repo", "repoId"), ("SkippedFile", "path"), ("SkippedFile", "repoId"))


def _record_branches():
    """record type -> ``(label, property, record field)`` triples that ``rows._record`` writes to an INDEXED property, found by walking its AST: the
    ``t == "<type>"`` branches, the ``b.node("<Label>", ...)`` calls in each, and the ``prop: r["field"]`` / ``r.get("field")`` entries of the dict literals
    (plus the comprehensions over the module's ``_SYMBOL_PROPS`` / ``_LAUNCH_PROPS`` tuples) of the same branch."""
    tree = ast.parse(Path(R.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_record")
    consts = {"_SYMBOL_PROPS": [(k, k) for k in R._SYMBOL_PROPS], "_LAUNCH_PROPS": list(R._LAUNCH_PROPS)}

    def field_of(v):
        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and v.value.id == "r" and isinstance(v.slice, ast.Constant):
            return v.slice.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "get" and isinstance(v.func.value, ast.Name) \
                and v.func.value.id == "r" and isinstance(v.args[0], ast.Constant):
            return v.args[0].value
        return None

    out: dict[str, set] = {}
    node = next(n for n in fn.body if isinstance(n, ast.If))
    while node is not None:
        rtype = node.test.comparators[0].value                                               # `t == "<rtype>"`
        labels, prop_field = set(), {}
        for n in (x for st in node.body for x in ast.walk(st)):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "node" and isinstance(n.args[0], ast.Constant):
                labels.add(n.args[0].value)
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if isinstance(k, ast.Constant) and field_of(v) is not None:
                        prop_field[k.value] = field_of(v)
            if isinstance(n, ast.DictComp):                                                  # {k: r.get(k) for k in _SYMBOL_PROPS} / {dst: r[src] for src, dst in _LAUNCH_PROPS}
                name = n.generators[0].iter.id
                for src, dst in consts[name]:
                    prop_field[dst] = src
        out[rtype] = {(lbl, p, prop_field[p]) for lbl in labels for p in indexed.props_of(lbl) if p in prop_field}
        node = node.orelse[0] if node.orelse and isinstance(node.orelse[0], ast.If) else None
    return out


def test_every_indexed_property_the_assembler_writes_from_a_record_is_in_the_literal_source_table():
    """The other half of the table, per record type: a ``b.node`` that writes an indexed property from a record field nobody listed (or listed under the
    wrong record type) fails here, and so does a listed row the builder no longer writes. ``Name.nameKey`` (a hash) and the contract/link ids are not fields."""
    written = {t: v for t, v in _record_branches().items() if v}
    expected: dict[str, set] = {}
    for t, f, lbl, p in EXPECTED_SOURCES:
        if t != "header":
            expected.setdefault(t, set()).add((lbl, p, f))
    assert written == expected


@pytest.mark.parametrize("rtype,field,label,prop", EXPECTED_SOURCES, ids=[f"{t}.{f}->{l}.{p}" for t, f, l, p in EXPECTED_SOURCES])
@pytest.mark.parametrize("kind", sorted(BAD_KINDS))
def test_classify_bundle_refuses_every_bad_kind_in_every_expected_source(tmp_path, rtype, field, label, prop, kind):
    """record type x indexed property x bad kind, from the LITERAL table: a mutant that deletes a row of ``BUNDLE_SOURCES`` (even one a sibling record type
    still maps to the same label.property) turns its cells red, because the world's record is planted and the gate must still refuse it."""
    b = _bundles()
    unit = b[1] if rtype in ("configEntry", "launchFact") else b[0]
    assert rtype in {r["type"] for r in unit}, f"the test world has no {rtype} record"
    _plant(unit, rtype, field, BAD_KINDS[kind])
    got = R.classify_bundle(unit, all_audit(tmp_path))
    assert got == CODE or (rtype == "header" and got == "bundle:invalid"), (rtype, field, kind, got)    # the header's own schema is the first gate for unitId


def test_a_world_with_no_bad_value_passes_every_expected_source(tmp_path):
    assert [R.classify_bundle(u, all_audit(tmp_path)) for u in _bundles()] == [None, None]


@pytest.mark.parametrize("kind", sorted(BAD_KINDS))
def test_a_bad_unit_repo_id_in_the_header_is_refused(tmp_path, kind):
    """``unit.repoId`` feeds ``Repo.repoId`` and ``SkippedFile.repoId``. The header schema pins it to ``[a-z0-9._-]+`` with no length bound, so a
    4,100-byte id passes the schema and only the indexed gate refuses it (``bundle:unindexable``); every other bad kind the schema refuses first (``bundle:invalid``)."""
    b = _bundles()
    b[0][0]["unit"]["repoId"] = BAD_KINDS[kind]
    got = R.classify_bundle(b[0], all_audit(tmp_path))
    assert got is not None and got in {CODE, "bundle:invalid"}
    if kind == "4100-bytes":
        b[0][0]["unit"]["repoId"] = "r" * 4100
        assert R.classify_bundle(b[0], all_audit(tmp_path)) == CODE


def test_a_bad_header_repo_id_is_reported_for_both_labels_that_carry_it(tmp_path):
    b = _bundles()
    b[0][0]["unit"]["repoId"] = "r" * 4100
    got = [(label, prop, i, v.reason) for label, prop, i, v in indexed.bundle_violations(b[0])]
    assert got == [("Repo", "repoId", 0, indexed.TOO_LONG), ("SkippedFile", "repoId", 0, indexed.TOO_LONG)]


def test_classify_bundle_refuses_a_bad_skipped_file_path_and_a_clean_world_passes(tmp_path):
    audit = all_audit(tmp_path)
    for k, v in BAD.items():
        w = world(skipped=[(v, "test")])
        assert R.classify_bundle(w[0], audit) == CODE, k
    assert [R.classify_bundle(u, audit) for u in world(skipped=[("svc/t_a.py", "test")])] == [None, None]


def test_a_lone_surrogate_callee_name_is_refused_because_its_name_key_cannot_be_formed(tmp_path):
    b = _bundles()
    _plant(b[0], "callsite", "calleeName", "f\ud800", nth=0)
    assert R.classify_bundle(b[0], all_audit(tmp_path)) == CODE


@pytest.mark.parametrize("name", [" handler ", "handler\t", "\nhandler", " "])
def test_a_callee_name_with_edge_whitespace_is_refused_not_stored_padded(tmp_path, name):
    """``Name.name`` is stored exactly as given and ``RESOLVES_TO`` joins it to the symbols' raw names, so a padded callee would store a Name that
    matches nothing (3 -> 2 edges on this world) and two spellings would share one ``nameKey``: refuse it at the gate, per unit, never strip."""
    c = _bundles()
    _plant(c[0], "callsite", "calleeName", name, nth=0)
    assert R.classify_bundle(c[0], all_audit(tmp_path)) == CODE
    _plant(c[0], "callsite", "calleeName", "handler", nth=0)
    assert R.classify_bundle(c[0], all_audit(tmp_path)) is None


def test_a_callee_name_that_is_empty_long_or_interior_padded_is_still_accepted_and_resolves_by_its_raw_name(tmp_path):
    for name in ("", "a b", "h" * 5000):
        c = _bundles()
        _plant(c[0], "callsite", "calleeName", name, nth=0)
        assert R.classify_bundle(c[0], all_audit(tmp_path)) is None, name[:5]
    rows = build(tmp_path)
    assert len(rows.edges[("Name", "RESOLVES_TO", "Symbol")]) == 3                      # the clean world: one edge per symbol the called names match


def test_a_non_indexed_value_may_be_hostile_and_the_bundle_is_accepted(tmp_path):
    b = _bundles()
    _plant(b[0], "symbol", "name", " x " + "n" * 9000)                      # Symbol.name has no index
    _plant(b[0], "cond", "code", "\n" + "c" * 9000 + "\n")
    assert R.classify_bundle(b[0], all_audit(tmp_path)) is None


def test_the_code_is_in_the_closed_set_and_carries_no_text_of_the_bundle(tmp_path):
    assert CODE in FAIL_REASONS
    b = _bundles()
    _plant(b[0], "file", "path", "SECRET-" + "k" * 5000)
    assert R.classify_bundle(b[0], all_audit(tmp_path)) == CODE
    with pytest.raises(R.BundleRefused) as e:
        build(tmp_path, bundles=[b[0], b[1]])
    assert e.value.code == CODE and "SECRET" not in str(e.value) and e.value.unit_id in str(e.value)


# ------------------------------------------------------------------------------------------------------------------ one unit, not the system

def _rows_with_one_bad_unit(tmp_path, bad_value, *, stale_kept: bool, system: str = SYSTEM):
    """U1 (the python unit) holds ``bad_value`` as a file path; U2 is a good config unit. The caller's gate (``classify_bundle``, as ``ccg.refresh``
    runs it) refuses U1, which becomes stale-kept (its last good bundle) or not-mapped; the system is built from the rest."""
    good, bad = _bundles(), _bundles()
    _plant(bad[0], "file", "path", bad_value)
    audit = all_audit(tmp_path)
    assert R.classify_bundle(bad[0], audit) == CODE and R.classify_bundle(bad[1], audit) is None
    u1 = good[0][0]["unitId"]
    if stale_kept:
        bundles, kw = good, {"stale": {u1: ("2026-11-02T01:02:03Z", CODE)}}
    else:
        nm = R.NotMappedUnit(u1, "repo1", "svc", "python", CODE)
        bundles, kw = [good[1]], {"not_mapped": [nm]}
    return build(tmp_path, bundles=bundles, system=system, **kw), bad_value, u1, bad


@pytest.mark.parametrize("stale_kept", [True, False], ids=["stale-kept", "not-mapped"])
@pytest.mark.parametrize("kind", sorted(BAD))
def test_a_bad_value_fails_only_its_unit_and_the_second_unit_still_loads(tmp_path, kind, stale_kept):
    rows, value, u1, bad = _rows_with_one_bad_unit(tmp_path, BAD[kind], stale_kept=stale_kept)
    unit = {u["unitId"]: u for u in rows.nodes["Unit"]}
    assert unit[u1]["buildStatus"] == ("stale-kept" if stale_kept else "not-mapped") and unit[u1]["failReason"] == CODE
    assert sum(1 for u in unit.values() if u["buildStatus"] == "fresh") == 1                     # U2
    assert any(f["path"] == "compose.yaml" for f in rows.nodes["File"])                         # U2's file is in the graph
    assert value not in repr(rows.nodes) and value not in repr(rows.edges)                      # the hostile text is nowhere in the rows
    store = FakeStore()
    report = L.load(store, rows, SYSTEM)                                                        # the loader's backstop does not fire: the system loads
    assert report.node_total > 0 and not store.ops("delete")


def test_a_hostile_value_that_skips_the_gate_is_refused_by_the_row_builder_and_then_by_the_loader(tmp_path, monkeypatch):
    """Two backstops behind the gate, neither of which alters the value: the row builder (an invariant) and the loader's preflight."""
    for value in ("p" * 4100, "svc/a.py ", "\tsvc/a.py"):
        b = _bundles()
        _plant(b[0], "file", "path", value)
        with monkeypatch.context() as mp:
            mp.setattr(R, "classify_bundle", lambda records, audit_dir: None)                 # the assembler's gate bypassed (a mutant)
            with pytest.raises(R.AssembleError, match="File.path: an unindexable value reached the row builder") as e:
                build(tmp_path, bundles=b)
        assert value[:8] not in str(e.value)
    rows = build(tmp_path)                                                                     # a row mutated after the build meets the loader
    for value, pat in (("svc/a.py ", "whitespace"), ("p" * 4100, r"4,100 UTF-8 bytes"), ("a\ud800b", "not valid Unicode")):
        rows.nodes["File"][0]["path"] = value
        store = FakeStore()
        with pytest.raises(L.LoadError, match=rf"File row 0: path .*{pat}") as e:
            L.load(store, rows, SYSTEM)
        assert store.calls == [] and "ppp" not in str(e.value) and "ud800" not in str(e.value) and "svc/a.py" not in str(e.value)


@pytest.mark.parametrize("value", [5, ["SECRET"], b"SECRET"])
def test_the_loader_refuses_a_non_string_indexed_value_without_naming_it(value):
    from loader_helpers import tiny_rows
    rows = tiny_rows(SYSTEM)
    rows.nodes["Symbol"][1]["qname"] = value
    store = FakeStore()
    with pytest.raises(L.LoadError, match=r"Symbol row 1: qname is not a string") as e:
        L.load(store, rows, SYSTEM)
    assert store.calls == [] and "SECRET" not in str(e.value)


@pytest.mark.parametrize("value,pat", [("", "empty string"), ("\tSECRET", "whitespace"), ("SECRET" * 900, "5,400 UTF-8 bytes"), ("SECRET\ud800", "not valid Unicode")])
def test_every_loader_refusal_names_label_row_property_and_size_and_never_the_value(value, pat):
    from loader_helpers import tiny_rows
    rows = tiny_rows(SYSTEM)
    rows.nodes["Symbol"][1]["qname"] = value
    store = FakeStore()
    with pytest.raises(L.LoadError, match=rf"Symbol row 1: qname .*{pat}") as e:
        L.load(store, rows, SYSTEM)
    assert store.calls == [] and "SECRET" not in str(e.value) and "ud800" not in str(e.value)


# ------------------------------------------------------------------------------------------------------------------ identity values are never altered

def test_a_path_that_differs_only_by_edge_whitespace_is_refused_not_stripped_into_a_collision(tmp_path):
    """Before P1-18b ``_clean`` stripped every string, so ``'svc/a.py '`` and ``'svc/a.py'`` became one ``File.path`` (two files, one lookup key)."""
    b = _bundles()
    b[0].append(file_rec("a2", "svc/a.py "))
    assert R.classify_bundle(b[0], all_audit(tmp_path)) == CODE
    with pytest.raises(R.BundleRefused) as e:
        build(tmp_path, bundles=b)
    assert e.value.code == CODE


def test_a_unicode_edged_value_the_engine_does_not_trim_is_stored_exactly_as_given(tmp_path):
    b = _bundles()
    b[0].append(file_rec("nb", "svc/a.py\u00a0"))
    b[0].append(file_rec("nb2", "\u3000svc/b.py"))
    rows = build(tmp_path, bundles=b)
    paths = {f["path"] for f in rows.nodes["File"]}
    assert {"svc/a.py", "svc/a.py\u00a0", "svc/b.py", "\u3000svc/b.py"} <= paths               # four distinct paths: nothing stripped, nothing merged


def test_the_row_builder_alters_no_string_value(tmp_path):
    b = _bundles()
    _plant(b[0], "cond", "code", "\u00a0x\n")                                                 # not indexed, not an identity: kept byte for byte
    rows = build(tmp_path, bundles=b)
    assert [c["code"] for c in rows.nodes["Cond"]] == ["\u00a0x\n"]


# ------------------------------------------------------------------------------------------------------------------ contract points from facts and links

def _fetch_world(tmp_path, literal, system=SYSTEM):
    from packrt_helpers import pack
    from ccg.linker import link
    from ccg.packrt import run_packs
    w = copy.deepcopy(world())
    w[0].append({"type": "valueSet", "id": H("vs", 1), "callSiteId": H("cs", "http"), "arg": "a0", "values": [{"const": literal}]})
    fp = pack("fetch")
    pf = run_packs([w[0]], [fp])                                                               # the real fetch pack over the unit's `fetch` call site
    pf = pf._replace(providers=[prov(HANDLER, "http", "GET " + literal)])
    lk = link(pf, w, packs=[fp])                                                               # the real linker
    return build(tmp_path, bundles=w, links=lk, facts=pf, packs=[fp], pack_dirs={}, system=system), lk


def test_a_key_over_4096_bytes_from_a_real_pack_and_linker_drops_its_contract_point_and_link_not_the_build(tmp_path):
    ok, lk_ok = _fetch_world(tmp_path, "/api/x")
    assert len(lk_ok.links) == 1 and ok.dropped.get("cp-unindexable") is None and len(ok.nodes["Link"]) == 1
    rows, lk = _fetch_world(tmp_path, "/" + "a" * 4100)
    assert len(lk.links) == 1 and max(len(l.key_norm or "") for l in lk.links) > 4096
    assert rows.dropped["cp-unindexable"] == 1 and rows.dropped["link-cp-unindexable"] == 1    # the provider's point and the link, each counted apart, value-free
    assert all(indexed.check(c["keyNorm"]) is None for c in rows.nodes.get("ContractPoint", []))
    assert "Link" not in rows.nodes and not rows.edges.get(("Link", "ON", "ContractPoint"))   # the Link and every edge of it went with its point
    assert not rows.edges.get(("Symbol", "PROVIDES", "ContractPoint"))
    assert any(s["symbolId"] == HANDLER for s in rows.nodes["Symbol"]) and any(c["callSiteId"] == H("cs", "http") for c in rows.nodes["CallSite"])
    assert "a" * 4000 not in repr(rows.nodes) + repr(rows.edges) + repr(rows.dropped)
    L.load(FakeStore(), rows, SYSTEM)                                                          # and the whole system loads


# ------------------------------------------------------------------------------------------------------------------ through ccg.refresh

def _refresh(world_, **kw):
    target, fakes = make_target()
    return RF.refresh(RF_SYSTEM, **world_.kw(target, **kw)), fakes


def _admit_hostile(monkeypatch, victim_root: str, value: str, *, only_commit: str | None = None):
    """``bundle.loads`` (the validator) admits ``value`` as the victim unit's file path: what it does for any >4,096-byte qname or path (the schema has no
    maxLength), so the assembler's gate is the one that must refuse it."""
    real = RF.bundle.loads

    def loads(data):
        records = real(data)
        if records[0]["unit"]["rootPath"] == victim_root and (only_commit is None or records[0]["builtCommit"] == only_commit):
            next(r for r in records if r["type"] == "file")["path"] = value
        return records

    monkeypatch.setattr(RF.bundle, "loads", loads)


@pytest.mark.parametrize("kind", sorted(BAD))
def test_through_refresh_a_hostile_path_fails_only_its_unit_and_the_system_swaps(tmp_path, monkeypatch, kind):
    w = World(tmp_path)
    _admit_hostile(monkeypatch, "bin/run.sh", BAD[kind])
    rep, fakes = _refresh(w)
    states = {u.root_path: (u.status, u.fail_reason) for u in rep.units}
    assert states["bin/run.sh"] == ("not-mapped", CODE) and states["conf/app.yaml"][0] == "fresh" and states["docs/note.md"][0] == "fresh"
    assert rep.swap.renamed and fakes.shared[f"ccg_{RF_SYSTEM}"] == "new"                      # the system is live
    assert BAD[kind] not in repr(fakes.store.queries())                                         # and the hostile text never reached the store


def test_through_refresh_a_unit_that_turns_unindexable_keeps_its_last_good_bundle(tmp_path, monkeypatch):
    w = World(tmp_path)
    first, _ = _refresh(w)
    c1 = first.heads["r1"]
    c2 = commit(w.repo(), {"bin/run.sh": "#!/bin/sh\necho two\n"})
    _admit_hostile(monkeypatch, "bin/run.sh", "p" * 4100, only_commit=c2)
    rep, fakes = _refresh(w)
    run = next(u for u in rep.units if u.root_path == "bin/run.sh")
    assert (run.status, run.built_commit, run.fail_reason) == ("stale-kept", c1, CODE) and rep.swap.renamed
    assert sum(u.status == "fresh" for u in rep.units) == 2


def _add_path(repo: Path, path: str, content=b"#!/bin/sh\necho x\n"):
    """Commit a file whose NAME the filesystem would refuse (too long), through the index: the refresh reads the commit's blobs."""
    env = {**__import__("os").environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    sha = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=content, capture_output=True, check=True, env=env).stdout.decode().strip()
    subprocess.run(["git", "-C", str(repo), "update-index", "--add", "--cacheinfo", f"100644,{sha},{path}"], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "p"], check=True, capture_output=True, env=env)


@pytest.mark.parametrize("name,path,code", [("trailing-space", "bin/odd ", "bundle:invalid"), ("4226-byte", "bin/" + "d" * 200 + "/" + ("e" * 200 + "/") * 20 + "x", "extract:failed")])
def test_real_git_paths_observed_through_refresh_fail_only_their_unit_at_an_earlier_gate(tmp_path, name, path, code):
    """Observed, not designed: the validator (``bundle:invalid``, §8.1 strips identity fields) and the extractor (``extract:failed``) refuse these two real
    shapes before the assembler's gate sees them. What matters here is the blast radius: one unit."""
    w = World(tmp_path)
    _add_path(w.repo(), path)
    rep, fakes = _refresh(w)
    odd = [u for u in rep.units if u.root_path == path]
    assert len(odd) == 1 and (odd[0].status, odd[0].fail_reason) == ("not-mapped", code)
    assert len(rep.fresh) == 3 and rep.swap.renamed


# ------------------------------------------------------------------------------------------------------------------ one predicate (AST)

def _tree(rel):
    return ast.parse((CPG / rel).read_text())


def _imports_indexed(tree):
    return any((isinstance(n, ast.ImportFrom) and n.module == "ccg.indexed")
               or (isinstance(n, ast.ImportFrom) and n.module == "ccg" and any(a.name == "indexed" for a in n.names))
               or (isinstance(n, ast.Import) and any(a.name == "ccg.indexed" for a in n.names)) for n in ast.walk(tree))


def _uses(tree, name):
    return any(isinstance(n, ast.Attribute) and n.attr == name and isinstance(n.value, ast.Name) and n.value.id == "indexed"
               for n in ast.walk(tree))


def test_loader_and_assembler_import_the_one_predicate_and_the_loader_calls_it():
    loader, rows = _tree("ccg/store/loader.py"), _tree("ccg/assemble/rows.py")
    assert _imports_indexed(loader) and _imports_indexed(rows)
    assert _uses(loader, "check") and _uses(loader, "props_of") and _uses(rows, "first_bundle_violation")


def _numbers_and_ws_sets(tree):
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool) and n.value in (4096, 4097, 5000):
            found.append(("bound", n.value, n.lineno))
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) >= 3 and set(n.value) <= set(" \t\n\v\f\r"):
            found.append(("ws", n.lineno))
    return found


def test_no_second_predicate_neither_module_holds_the_bound_the_whitespace_set_or_a_utf8_size_check():
    for rel in ("ccg/store/loader.py", "ccg/assemble/rows.py"):
        tree = _tree(rel)
        assert _numbers_and_ws_sets(tree) == [], rel
        src = (CPG / rel).read_text()
        assert "len(value.encode" not in src and ".encode(\"utf-8\")) >" not in src and "INDEX_TRIMMED_CHARS =" not in src and "_INDEXED_BY_LABEL" not in src, rel
        assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        assert not {"INDEXED_PROPS", "INDEXED_VALUE_MAX_BYTES", "INDEX_TRIMMED_CHARS"} & assigned, rel


def test_the_walk_finds_a_planted_second_predicate():
    planted = ast.parse("LIMIT = 4096\nWS = frozenset(' \\t\\n\\v\\f\\r')\nif len(v.encode('utf-8')) > LIMIT: pass\n")
    assert [k[0] for k in _numbers_and_ws_sets(planted)] == ["bound", "ws"]


def test_only_the_shared_module_defines_the_bound_across_the_package():
    offenders = []
    for p in sorted((CPG / "ccg").rglob("*.py")):
        rel = p.relative_to(CPG).as_posix()
        if rel == "ccg/indexed.py":
            continue
        for kind, *rest in _numbers_and_ws_sets(ast.parse(p.read_text())):
            if kind == "bound" and rest[0] == 4096 and rel.startswith(("ccg/store/", "ccg/assemble/")):
                offenders.append((rel, rest))
    assert offenders == []
