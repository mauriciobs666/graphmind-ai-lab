"""P1-19 dispatch 2: the id-hash fuzz (``-p1-linking.md`` P1-19 behaviour (4)). The pure half runs anywhere; the ``live`` half runs on a THROWAWAY instance only
(``infra/ccg-falkordb.sh --throwaway up``; ``CCG_FALKORDB_PORT`` + a scratch ``CCG_CONFIG_DIR``; ``CCG_CREDENTIALS_DIR`` is set to a tmp dir by every test).

What it establishes, with the engine observations (falkordb v4.18.11, 2026-11-11) that make the 4,096-byte bound a requirement and not a nicety:
* every id is the 32-hex hash of the canonical tuple, whatever the input (long, whitespace-edged, NFC/NFD, quotes, braces, ``$``, newlines, Cypher text);
* the raw text survives the loader byte for byte in an unindexed property and as an indexed one up to 4,096 UTF-8 bytes;
* an indexed value over 4,096 bytes is NOT indexed (an equality anchor returns nothing although the node exists), and a ``UNIQUE`` constraint over one
  SEGFAULTS the server (``EnforceUniqueEntity``; exit 139). The crash is observed by hand and recorded in ``claude/graph-dba/falkordb-quirks.md``; this
  suite never sends it, so it checks that the LOADER refuses such a value before the first write, and that the hash keeps ``nameKey`` far below the bound;
* a NUL in a parameter value is refused by the engine ("Failed to parse query parameter"), and a failed build leaves no build key."""
from __future__ import annotations

import ast
import os
import random
import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest
from loader_helpers import admin_password, skip_or_fail, throwaway_port, tiny_rows

from ccg import ids, sysname
from ccg.store import acl
from ccg.store import loader as L

HOST = "127.0.0.1"
HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")
BOUND = 4096


def _fuzz_strings() -> list[str]:
    fixed = [
        "x", " x", "x ", "\tx\n", "  padded  ", "\u00a0x\u2003", "\u200bx", "x\u200b",           # whitespace edges (NBSP and EM SPACE are ``isspace``; ZWSP is not)
        "a'b", 'a"b', "a\\b", "a\\", "'}) DETACH DELETE n //", "}) RETURN 1 //", "$param", "{a: 1}", "a`b", "a;b", "/* c */", "x -- y", "a\nb", "a\r\nb", "a\x0bb",
        "e\u0301", "\u00e9", "\u212b", "\u00c5", "A\u030a", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629", "\U0001f600x", "\ufeffx", "a\u2028b", "a\x01b", "a\x7fb",
        "k" * BOUND, "k" * (BOUND + 1), "\u00e9" * (BOUND // 2), "\u00e9" * (BOUND // 2 + 1), "w" * 100_000, "\u4e2d" * 1500,
    ]
    rng = random.Random(1190)
    alphabet = "abcXYZ019 \t\n'\"\\{}$`;:/.-_é中😀\u00a0\u0301"
    return fixed + ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 120))) for _ in range(300)]


FUZZ = _fuzz_strings()


# ------------------------------------------------------------------------------------------------ pure: the hash

@pytest.mark.parametrize("s", [pytest.param(s, id=f"{i}-{len(s.encode())}B") for i, s in enumerate(FUZZ)])
def test_every_name_key_is_32_lowercase_hex_whatever_the_input(s):
    k = ids.name_key("python", s)
    assert HEX32.match(k) and len(k.encode()) == 32 < BOUND
    assert ids.name_key("python", s) == k                                               # deterministic
    assert ids.name_key("python", unicodedata.normalize("NFD", s)) == k                 # NFC first
    assert ids.name_key("python", " \t" + s + "\n\u00a0") == k                          # whitespace edges are stripped
    assert ids.name_key("javascript", s) != k                                           # the language is in the tuple


def test_text_that_cannot_be_utf8_is_refused_loudly_not_hashed():
    with pytest.raises(UnicodeError):
        ids.name_key("python", "a\ud800b")


def test_a_name_over_4096_bytes_hashes_to_32_bytes_and_the_assembler_forms_name_keys_only_through_the_hash():
    long_name = "n" * (BOUND * 3)
    assert len(ids.name_key("python", long_name)) == 32
    rows_py = Path(__file__).resolve().parents[1] / "ccg" / "assemble" / "rows.py"
    tree = ast.parse(rows_py.read_text())
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "nameKey":
                    sites.append(ast.unparse(v))
    assert sites == ["nk"], sites                                                       # one site, and ``nk`` is assigned from ``ids.name_key(...)`` there
    assigns = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "nk" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].startswith("ids.name_key(")


# ------------------------------------------------------------------------------------------------ live

@pytest.fixture(autouse=True)
def _credentials_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(tmp_path / "cred"))


@pytest.fixture(scope="module")
def fz(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance: start one with cpg/infra/ccg-falkordb.sh --throwaway up and export CCG_FALKORDB_PORT and CCG_CONFIG_DIR")
    try:
        pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    admin = L.connect(HOST, port, username=None, password=pw)
    system = f"fz{os.getpid() % 1000}"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(tmp_path_factory.mktemp("ccg-credentials-fuzz")))
        users = acl.create_users(admin, system)
        build = L.connect(HOST, port, username=sysname.acl_user("build", system), password=acl.read_credentials(system)["build"])
    w = SimpleNamespace(port=port, admin=admin, system=system, build=build, n=0)
    try:
        yield w
    finally:
        for pattern in (f"ccg_{system}*", "p119b_fuzz*"):
            for key in list(admin.connection.scan_iter(match=pattern)):
                admin.connection.delete(key)
        for u in users:
            admin.connection.execute_command("ACL", "DELUSER", u)


def _rows(fz, **extra):
    fz.n += 1
    return tiny_rows(fz.system, tag=f"f{fz.n}")


def _alive(fz):
    assert fz.admin.connection.execute_command("INFO", "server")


def _name_rows(strings):
    """The assembler's Name rows: one per distinct hash (``b.node`` dedups by id), ``name`` carrying the raw text."""
    out: dict[str, dict] = {}
    for s in strings:
        out.setdefault(ids.name_key("python", s), {"nameKey": ids.name_key("python", s), "lang": "python", "name": s})
    return list(out.values())


def _with_names(rows, names):
    rows.nodes["Name"] = names
    rows.edges[("CallSite", "CALLS_NAME", "Name")] = [{"s": "CallSite1", "t": n["nameKey"]} for n in names]
    rows.edges[("Name", "RESOLVES_TO", "Symbol")] = [{"s": names[0]["nameKey"], "t": "Symbol1"}]
    return rows


def _loadable(s: str) -> bool:
    return "\x00" not in s


@pytest.mark.live
def test_the_fuzz_names_round_trip_through_the_loader_byte_for_byte_and_the_constraint_holds(fz):
    strings = [s for s in FUZZ if _loadable(s)]
    names = _name_rows(strings)
    assert len(names) < len(strings)                                                    # the NFC/NFD and whitespace-edge variants collapsed to one id each
    rows = _with_names(_rows(fz), names)
    report = L.load(L.LiveStore(fz.build, fz.system), rows, fz.system)
    g = fz.admin.select_graph(report.build_key)
    got = {r[0]: r[1] for r in g.ro_query("MATCH (n:Name) RETURN n.nameKey, n.name").result_set}
    assert got == {n["nameKey"]: n["name"] for n in names}                              # raw text intact: quotes, newlines, NBSP, emoji, 100,000 bytes, ...
    assert all(HEX32.match(k) for k in got)
    assert g.ro_query("MATCH (:CallSite)-[e:CALLS_NAME]->(:Name) RETURN count(e)").result_set == [[len(names)]]
    assert all(c[4] == "OPERATIONAL" for c in g.ro_query("CALL db.constraints()").result_set)
    assert g.ro_query("MATCH (n) WHERE n:Unit OR n:Symbol RETURN count(n)").result_set == [[4]]   # the injection-shaped names deleted nothing
    for s in ("'}) DETACH DELETE n //", "  padded  ", "a\nb", "w" * 100_000):            # a lookup by the hash finds exactly that text
        assert g.ro_query("MATCH (n:Name {nameKey: $k}) RETURN n.name", {"k": ids.name_key("python", s)}).result_set == [[s]]
    _alive(fz)
    fz.admin.connection.delete(report.build_key)


@pytest.mark.live
def test_a_nul_in_a_parameter_value_fails_the_build_loudly_and_leaves_no_build_key(fz):
    names = _name_rows(["ok", "a\x00b"])
    rows = _with_names(_rows(fz), names)
    with pytest.raises(Exception, match="(?i)parse|param|nul|null"):
        L.load(L.LiveStore(fz.build, fz.system), rows, fz.system)
    assert not fz.admin.connection.exists(sysname.build_key(fz.system, rows.snapshot["snapshotId"]))
    _alive(fz)


ODD_INDEXED = ["inner  space", "a'b\"c\\d", "line\nbreak", "{x: $y}", "\u00a0nbsp", "e\u0301", "😀", "'}) DETACH DELETE n //", "k" * BOUND, "\u00e9" * (BOUND // 2)]


@pytest.mark.live
def test_odd_indexed_values_up_to_4096_bytes_are_found_by_an_index_scan_equality(fz):
    rows = _rows(fz)
    rows.nodes["File"] += [{"fileId": f"FileOdd{i}", "path": p} for i, p in enumerate(ODD_INDEXED)]
    rows.nodes["Symbol"] += [{"symbolId": f"SymbolOdd{i}", "qname": p, "nameNorm": p} for i, p in enumerate(ODD_INDEXED)]
    report = L.load(L.LiveStore(fz.build, fz.system), rows, fz.system)
    g = fz.admin.select_graph(report.build_key)
    for i, p in enumerate(ODD_INDEXED):
        assert g.ro_query("MATCH (f:File) WHERE f.path = $p RETURN f.fileId", {"p": p}).result_set == [[f"FileOdd{i}"]], i
        assert g.ro_query("MATCH (s:Symbol) WHERE s.qname = $p RETURN s.symbolId", {"p": p}).result_set == [[f"SymbolOdd{i}"]], i
    plan = g.explain("MATCH (f:File) WHERE f.path = $p RETURN f.fileId", {"p": "x"})
    assert plan.collect_operations("Node By Index Scan") and not plan.collect_operations("Node By Label Scan")
    _alive(fz)
    fz.admin.connection.delete(report.build_key)


@pytest.mark.live
@pytest.mark.parametrize("value", ["  edged  ", "\tx", "x\n", " "])
def test_an_ascii_whitespace_edged_indexed_value_never_reaches_the_server(fz, value):
    rows = _rows(fz)
    rows.nodes["File"][0]["path"] = value
    bkey = sysname.build_key(fz.system, rows.snapshot["snapshotId"])
    with pytest.raises(L.LoadError, match=r"File row 0.*path.*whitespace"):
        L.load(L.LiveStore(fz.build, fz.system), rows, fz.system)
    assert not fz.admin.connection.exists(bkey)


@pytest.mark.live
def test_the_engine_index_trims_ascii_whitespace_so_an_equality_anchor_misses_or_matches_other_values(fz):
    """Observed on a scratch graph: the six ASCII whitespace characters are trimmed off the stored value in the index but not off the lookup value."""
    g = fz.admin.select_graph("p119b_fuzz_ws")
    g.query("CREATE INDEX FOR (f:File) ON (f.path)")
    stored = ["  edged  ", "edged", "\u00a0edged\u00a0"]
    for v in stored:
        g.query("CREATE (:File {path: $p})", {"p": v})
    count = lambda v: g.ro_query("MATCH (f:File) WHERE f.path = $p RETURN count(f)", {"p": v}).result_set[0][0]
    scan = lambda v: g.ro_query("MATCH (f:File) WHERE f.path + '' = $p RETURN count(f)", {"p": v}).result_set[0][0]
    assert (count("  edged  "), scan("  edged  ")) == (0, 1)                 # the exact value is MISSED by the index, found by a scan
    assert (count("edged"), scan("edged")) == (2, 1)                        # and the stripped one matches a node that does not hold it
    assert (count("\u00a0edged\u00a0"), scan("\u00a0edged\u00a0")) == (1, 1)    # NBSP is not trimmed: the control
    fz.admin.connection.delete("p119b_fuzz_ws")


@pytest.mark.live
@pytest.mark.parametrize("label,prop,value", [("Name", "nameKey", "k" * (BOUND + 1)), ("Name", "nameKey", "\u00e9" * (BOUND // 2 + 1)),
                                              ("Symbol", "symbolId", "k" * 100_000), ("Symbol", "qname", "q" * (BOUND + 1)), ("File", "path", "p" * (BOUND + 1))])
def test_an_indexed_value_over_4096_bytes_never_reaches_the_server_and_the_server_stays_up(fz, label, prop, value):
    """The unhashed ``nameKey`` of a bypass (and the silent-miss ones): refused by the loader before the build key exists; the instance is alive after."""
    rows = _rows(fz)
    rows.nodes[label][0][prop] = value
    bkey = sysname.build_key(fz.system, rows.snapshot["snapshotId"])
    with pytest.raises(L.LoadError, match=rf"{label} row 0.*{prop}.*4,096"):
        L.load(L.LiveStore(fz.build, fz.system), rows, fz.system)
    assert not fz.admin.connection.exists(bkey)
    _alive(fz)


@pytest.mark.live
def test_the_engine_does_not_index_a_value_over_4096_bytes_which_is_why_the_loader_refuses_it(fz):
    """Observed, on a scratch graph (index only, NO constraint: a constraint over such a value crashes the server and is not sent here)."""
    g = fz.admin.select_graph("p119b_fuzz_bound")
    g.query("CREATE INDEX FOR (n:Name) ON (n.nameKey)")
    for size in (BOUND, BOUND + 1):
        v = "k" * size
        g.query("CREATE (:Name {nameKey: $v, size: $n})", {"v": v, "n": size})
        hit = g.ro_query("MATCH (n:Name) WHERE n.nameKey = $v RETURN count(n)", {"v": v}).result_set[0][0]
        scan = g.ro_query("MATCH (n:Name) WHERE n.size = $n AND n.nameKey STARTS WITH 'kkk' RETURN count(n)", {"n": size}).result_set[0][0]
        assert scan == 1                                                                # the node exists in both cases
        assert hit == (1 if size == BOUND else 0), (size, hit)                          # ... but the indexed equality misses it past 4,096 bytes
    _alive(fz)
    fz.admin.connection.delete("p119b_fuzz_bound")
