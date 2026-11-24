"""CO-5: the per-kind miss breakdown. The report a builder reads carries STRATUM-level miss counts only; the fine breakdown (stratum x anchor type x impacted
type, counts) is held-out-derived and goes only to ``<custodian>/work/gate-miss-breakdown.jsonl``, never to stdout or the report. Offline, synthetic."""
import json

import gate_helpers as H
from gate_helpers import G
from test_gate_fail import hits, result
from test_gate_run import ScoreEnv


def rec(n, stratum, anchor, impacted):
    return H.item(n, anchor, impacted, stratum=stratum)


def world():
    sym_a, con = H.sym("a.py::f"), H.contract("http", "GET /x")
    recs = [rec(1, "http", con, H.sym("b.py::g")), rec(2, "http", con, H.cfg("c.yaml", "services.s")), rec(3, "scripts", sym_a, H.sym("d.sh::h")),
            rec(4, "scripts", sym_a, H.sym("e.sh::i")), rec(5, "mcp", con, H.sym("m.py::t"))]
    ids = [r["itemId"] for r in recs]
    rows = [(ids[0], "http", "hit", True, "A1"), (ids[1], "http", "miss", True, "A1"), (ids[2], "scripts", "miss", True, "A2"),
            (ids[3], "scripts", "miss", True, "A2"), (ids[4], "mcp", "hit", True, "A3")]
    return recs, result(rows)


def test_the_fine_breakdown_counts_items_hits_and_misses_per_stratum_anchor_type_and_impacted_type():
    recs, res = world()
    cells = {(c["stratum"], c["anchorType"], c["impactedType"]): (c["items"], c["hits"], c["misses"]) for c in G.miss_breakdown(recs, res)}
    assert cells == {("http", "contract", "symbol"): (1, 1, 0), ("http", "contract", "config"): (1, 0, 1), ("scripts", "symbol", "symbol"): (2, 0, 2),
                     ("mcp", "contract", "symbol"): (1, 1, 0)}


def test_the_fine_breakdown_carries_no_id_path_or_name():
    recs, res = world()
    text = json.dumps(G.miss_breakdown(recs, res))
    for r in recs:
        for secret in (r["itemId"], r["anchorId"], r["impactedId"], "b.py", "GET /x", "d.sh"):
            assert secret not in text


def test_a_withheld_or_unscored_item_is_not_in_the_breakdown():
    recs, res = world()
    res["items"][recs[4]["itemId"]]["outcome"] = "out-of-scope"
    assert ("mcp", "contract", "symbol") not in {(c["stratum"], c["anchorType"], c["impactedType"]) for c in G.miss_breakdown(recs, res)}


def test_the_builder_visible_report_has_stratum_level_miss_counts_and_no_fine_breakdown():
    recs, res = world()
    rep = G.build_report(res, G.decide_gate(res))
    assert rep["misses"]["byStratum"] == {"http": 1, "scripts": 2}
    text = json.dumps(rep)
    for word in ("anchorType", "impactedType", "breakdown"):
        assert word not in text


def test_score_writes_the_fine_breakdown_under_the_custodian_work_dir_and_never_prints_it(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    e.main()
    out = capsys.readouterr().out
    f = e.cust / "work" / "gate-miss-breakdown.jsonl"
    cells = [json.loads(l) for l in f.read_text().splitlines()]
    assert cells and all({"stratum", "anchorType", "impactedType", "items", "hits", "misses"} <= set(c) for c in cells)
    for word in ("anchorType", "impactedType", "gate-miss-breakdown"):
        assert word not in out
    assert "misses" in json.loads(out) and json.loads(out)["misses"]["byStratum"] == {}
