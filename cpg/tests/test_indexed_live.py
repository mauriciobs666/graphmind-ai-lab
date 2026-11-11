"""P1-18b, live half: the systematic grid of ``ccg.indexed``'s verdict against the engine's behaviour, and a unit-only failure loaded for real.

THROWAWAY INSTANCES ONLY (``infra/ccg-falkordb.sh --throwaway up``; ``CCG_FALKORDB_PORT`` + a scratch ``CCG_CONFIG_DIR``; ``tests/conftest.py`` points
``CCG_CREDENTIALS_DIR`` at a tmp dir). ``loader_helpers.throwaway_port`` refuses 6379/6390 and any port not published by a ``ccg-falkordb.throwaway=1``
container. Grid keys are ``p118b_probe_*`` graphs, removed at the end.

The grid (falkordb v4.18.11): label x indexed property (the DDL's 20) x edge character (the six ASCII whitespace characters, five Unicode non-trimmed ones)
x position (start, end, both, only, interior) x byte length (4,094 .. 4,098 ASCII; a 2-, 3- and 4-byte character straddling 4,095 / 4,096 / 4,097,
at the start and at the end). For every value the engine is asked the one question the store exists to answer: ``WHERE n.prop = $v`` must return exactly
the nodes whose stored value is ``$v``. Soundness (asserted): no value the predicate accepts is mis-indexed. Conservatism (asserted): every value
the engine mis-indexes is flagged. The over-flag count (flagged, engine fine) is printed, not asserted: it is the price of a conservative rule."""
from __future__ import annotations

import os

import pytest
from loader_helpers import admin_password, skip_or_fail, throwaway_port
from test_fuzz_idhash_live import HOST, fz  # noqa: F401  (the module-scoped fixture: a throwaway, one system's ACL users)
from test_indexed import BAD, CODE, _fetch_world, _rows_with_one_bad_unit

from ccg import indexed
from ccg.store import loader as L
from ccg.store import parse_ddl

WS = list(" \t\n\v\f\r")
UNICODE_SPACES = [" ", " ", "　", "​", "\u0085"]
POSITIONS = ("start", "end", "both", "only", "interior")


def _edge(c: str, pos: str, token: str) -> str:
    return {"start": c + token, "end": token + c, "both": c + token + c, "only": c, "interior": token[:1] + c + token[1:]}[pos]


def grid() -> list[tuple[str, str]]:
    """``(value, family)`` pairs, all distinct."""
    out: list[tuple[str, str]] = []
    for ci, c in enumerate(WS + UNICODE_SPACES):
        for pos in POSITIONS:
            out.append((_edge(c, pos, f"tok{ci}"), f"edge:{c!r}:{pos}"))
        out.append((f"tok{ci}", "clean"))
    for n in (4094, 4095, 4096, 4097, 4098):
        out.append(("k" * n, f"ascii:{n}"))
    for ch in ("é", "中", "😀"):
        w = len(ch.encode())
        for b in (4095, 4096, 4097):
            out.append(("k" * (b - w) + ch, f"{w}byte:end:{b}"))
            out.append((ch + "k" * (b - w), f"{w}byte:start:{b}"))
        out.append((ch * (4096 // w), f"{w}byte:all:{4096 // w * w}"))
        out.append((ch * (4096 // w + 1), f"{w}byte:all:{(4096 // w + 1) * w}"))
    out.append(("", "empty"))
    seen: dict[str, str] = {}
    for v, fam in out:
        seen.setdefault(v, fam)
    return [(v, f) for v, f in seen.items()]


def test_the_grid_is_systematic_and_has_every_dimension():
    g = grid()
    fams = {f.split(":")[0] for _, f in g}
    assert {"edge", "clean", "ascii", "2byte", "3byte", "4byte", "empty"} <= fams
    assert len(g) > 90 and len({v for v, _ in g}) == len(g)
    sizes = {len(v.encode()) for v, _ in g}
    assert {4095, 4096, 4097} <= sizes
    assert {(l, p) for l, p in indexed.INDEXED_PROPS} == {(s[1], s[2]) for s in parse_ddl() if s[0] == "index"}


@pytest.fixture(scope="module")
def probe():
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance: start one with cpg/infra/ccg-falkordb.sh --throwaway up and export CCG_FALKORDB_PORT and CCG_CONFIG_DIR")
    admin = L.connect(HOST, port, username=None, password=admin_password())
    key = f"p118b_probe_{os.getpid()}"
    g = admin.select_graph(key)
    for s in parse_ddl():
        if s[0] == "index":
            g.query(s[3])                                           # the DDL's own index statements, on an empty key
    try:
        yield admin, g
    finally:
        for k in list(admin.connection.scan_iter(match="p118b_probe_*")):
            admin.connection.delete(k)


def _lookup(g, label: str, prop: str, v: str) -> list[str]:
    q = f"MATCH (n:{label}) WHERE n.{prop} = $v RETURN n.mark ORDER BY n.mark"
    return [r[0] for r in g.ro_query(q, {"v": v}, timeout=10000).result_set]


def _put(g, label: str, prop: str, v: str, mark: str) -> None:
    g.query(f"CREATE (:{label} {{{prop}: $v, mark: $m}})", {"v": v, "m": mark}, timeout=10000)


@pytest.mark.live
def test_the_index_is_used_for_the_equality_the_grid_asks(probe):
    _, g = probe
    plan = g.explain("MATCH (n:File) WHERE n.path = $v RETURN n.mark", {"v": "x"})
    assert plan.collect_operations("Node By Index Scan") and not plan.collect_operations("Node By Label Scan")


@pytest.mark.live
def test_the_predicate_is_sound_and_conservative_against_the_engine_over_the_grid(probe, capsys):
    admin, g = probe
    values = grid()
    flagged = [(v, f) for v, f in values if indexed.check(v) is not None]
    accepted = [(v, f) for v, f in values if indexed.check(v) is None]
    assert flagged and accepted
    cells = sorted(indexed.INDEXED_PROPS)
    assert len(cells) == 20
    unsound: list[tuple] = []                                           # accepted by the predicate, mis-indexed by the engine
    engine_bad: dict[str, int] = {}                                     # flagged reason -> the engine does mis-index it
    over_flagged: dict[str, int] = {}                                   # flagged reason -> the engine handles it fine
    checked = 0
    for label, prop in cells:
        # phase A: every accepted value together, so a false positive between neighbours would show: each equality returns exactly its own node
        marks = {}
        for i, (v, fam) in enumerate(accepted):
            marks[v] = f"a{i}"
            _put(g, label, prop, v, marks[v])
        for v, fam in accepted:
            checked += 1
            if _lookup(g, label, prop, v) != [marks[v]]:
                unsound.append((label, prop, fam, len(v.encode())))
        g.query(f"MATCH (n:{label}) DELETE n")
        # phase B: each flagged value alone beside its clean neighbour: does the engine mis-answer?
        for v, fam in flagged:
            token = v.strip(" \t\n\v\f\r")[:6] if fam.startswith("edge") else ""
            _put(g, label, prop, v, "bad")
            if token and fam.startswith("edge"):
                _put(g, label, prop, token, "clean")                    # the lookup of the clean text must not return the edged node as well
            ok_self = _lookup(g, label, prop, v) == ["bad"]
            ok_clean = (not token) or _lookup(g, label, prop, token) == (["clean"] if token != v else ["bad"])
            reason = indexed.check(v).reason
            checked += 1
            (over_flagged if (ok_self and ok_clean) else engine_bad).__setitem__(reason, (over_flagged if (ok_self and ok_clean) else engine_bad).get(reason, 0) + 1)
            g.query(f"MATCH (n:{label}) DELETE n")
        # a lone surrogate cannot be sent at all: the predicate flags it, the client refuses it, and the server stays up
        assert indexed.check("a\ud800b").reason == indexed.UNICODE
        with pytest.raises(Exception):
            g.query(f"CREATE (:{label} {{{prop}: $v}})", {"v": "a\ud800b"})
    with capsys.disabled():
        print(f"\n[p118b grid] cells={len(cells)} values={len(values)} (accepted {len(accepted)}, flagged {len(flagged)}) checks={checked}"
              f"\n[p118b grid] engine mis-indexes, by predicate reason: {dict(sorted(engine_bad.items()))}"
              f"\n[p118b grid] over-flagged (flagged, engine fine): {dict(sorted(over_flagged.items()))}")
    assert unsound == [], unsound[:5]                                   # soundness: every value the predicate accepts is indexed faithfully
    # the grid is not vacuous: the engine really mis-indexes both documented classes, and the predicate flagged every such value (a value the engine
    # mis-indexes yet the predicate accepted would be in ``unsound``: it sat in phase A and failed its own lookup)
    assert engine_bad.get(indexed.TOO_LONG, 0) > 0 and engine_bad.get(indexed.WHITESPACE, 0) > 0
    assert admin.connection.execute_command("INFO", "server")


@pytest.mark.live
@pytest.mark.parametrize("stale_kept", [True, False], ids=["stale-kept", "not-mapped"])
@pytest.mark.parametrize("kind", sorted(BAD))
def test_a_unit_with_a_hostile_value_fails_alone_and_the_system_loads_on_the_server(fz, tmp_path, kind, stale_kept):  # noqa: F811
    rows, value, u1, _ = _rows_with_one_bad_unit(tmp_path, BAD[kind], stale_kept=stale_kept, system=fz.system)
    report = L.load(L.LiveStore(fz.build, fz.system), rows, fz.system)
    g = fz.admin.select_graph(report.build_key)
    units = {r[0]: (r[1], r[2]) for r in g.ro_query("MATCH (u:Unit) RETURN u.unitId, u.buildStatus, u.failReason").result_set}
    assert units[u1] == ("stale-kept" if stale_kept else "not-mapped", CODE) and sorted(s for s, _ in units.values()).count("fresh") == 1
    assert g.ro_query("MATCH (f:File {path: 'compose.yaml'}) RETURN count(f)").result_set == [[1]]       # the second unit's file is findable by its index
    assert g.ro_query("MATCH (f:File) WHERE f.path STARTS WITH 'svc/a' RETURN count(f)").result_set == ([[1]] if stale_kept else [[0]])
    assert all(c[4] == "OPERATIONAL" for c in g.ro_query("CALL db.constraints()").result_set)
    fz.admin.connection.delete(report.build_key)


@pytest.mark.live
def test_a_key_over_4096_bytes_drops_its_contract_point_and_the_system_still_loads_on_the_server(fz, tmp_path):  # noqa: F811
    rows, _ = _fetch_world(tmp_path, "/" + "a" * 4100, system=fz.system)
    assert rows.dropped["cp-unindexable"] == 1 and rows.dropped["link-cp-unindexable"] == 1
    report = L.load(L.LiveStore(fz.build, fz.system), rows, fz.system)
    g = fz.admin.select_graph(report.build_key)
    assert g.ro_query("MATCH (c:ContractPoint) RETURN count(c)").result_set == [[1]]            # only the consumer's own point (a short key) is in the graph
    assert g.ro_query("MATCH (c:ContractPoint) WHERE size(c.keyNorm) > 4096 RETURN count(c)").result_set == [[0]]
    assert all(c[4] == "OPERATIONAL" for c in g.ro_query("CALL db.constraints()").result_set)
    fz.admin.connection.delete(report.build_key)
