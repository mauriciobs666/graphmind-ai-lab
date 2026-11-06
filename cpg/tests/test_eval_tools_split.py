"""Tests for cpg/eval/tools/{capture,split_manifest}.py (S49 second half, P0-42b).

Run from cpg/:  .venv/bin/python -m pytest -q tests/test_eval_tools_split.py

Oracle: eval-protocol Appendix A (``capture.py``; the reference formulas are transcribed below, and the
document's rounded printed figures are pinned as literals), §13.6's fixture list (B1a, B1b, M2, M6, reserve,
m4, P2-1a-e, P2-3) for ``split_manifest.py``. Synthetic data only. ``CCG_TOOLS_DIR`` points the tests at a
mutated copy of the tools (the mutation harness).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from math import comb, exp, factorial, log, sqrt
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parent.parent
TOOLS = Path(os.environ.get("CCG_TOOLS_DIR", CPG / "eval" / "tools"))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(CPG) not in sys.path:
    sys.path.insert(0, str(CPG))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mk = _load("match_keys")
cap = _load("capture")
sm = _load("split_manifest")
STRATA = ("http", "mcp", "scripts", "wiring", "flow")


# =============================================================================================
# Appendix A reference (transcribed from the protocol document)
# =============================================================================================
Z = 1.959963984540054


def a_chapman(n1, n2, m):
    N = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    return N, var


def a_chao(r, f1, f2, t):
    return r + ((t - 1) / t) * f1 * (f1 - 1) / (2 * (f2 + 1))


def a_ci(r, N, var):
    f0 = N - r
    if f0 <= 0 or var <= 0:
        return r, r
    C = exp(Z * sqrt(log(1 + var / f0 ** 2)))
    return r + f0 / C, r + f0 * C


def a_nstar(n1, n2, m):
    r = n1 + n2 - m
    Nc, _ = a_chapman(n1, n2, m)
    return r, max(Nc, a_chao(r, n1 + n2 - 2 * m, m, 2))


def a_ptrigger(N, p1, p2, num=95, den=100):
    pa, pb, pc, pd = p1 * p2, p1 * (1 - p2), (1 - p1) * p2, (1 - p1) * (1 - p2)
    tot = 0.0
    for a in range(N + 1):
        for b in range(N - a + 1):
            for c in range(N - a - b + 1):
                d = N - a - b - c
                pr = factorial(N) / (factorial(a) * factorial(b) * factorial(c) * factorial(d)) * pa ** a * pb ** b * pc ** c * pd ** d
                r = a + b + c
                if r == 0:
                    tot += pr
                    continue
                _, Ns = a_nstar(a + b, a + c, a)
                if r * den < num * Ns:
                    tot += pr
    return tot


def hist2(n1, n2, m):
    return {"101": m, "100": n1 - m, "001": n2 - m}


THREE = {"111": 20, "110": 6, "101": 4, "011": 3, "100": 5, "010": 3, "001": 4}


def match_of(hists_by_stratum: dict, t=None) -> dict:
    """A scoped match.json. ``t`` defaults like match_keys: 3 where a KA2 bit appears, else 2 (override per stratum)."""
    recs = []
    for s, hist in hists_by_stratum.items():
        i = 0
        for h, n in hist.items():
            for _ in range(n):
                recs.append({"itemId": f"{s}-{h}-{i}", "derivedStratum": s, "history": h, "members": [], "inScope": True})
                i += 1
    tmap = {s: 3 if any(h[1] == "1" for h in hists_by_stratum.get(s, {})) else 2 for s in STRATA}
    tmap.update(t or {})
    return {"schema": "ccg-match-v1", "records": recs, "unclassified": 0, "stratumDisagreements": 0, "t": tmap}


# =============================================================================================
# capture.py (EP §6.4-§6.5, §13.5; Appendix A as the oracle)
# =============================================================================================

@pytest.mark.parametrize("n1,n2,m,nc,lo,hi,nh,comp", [
    (40, 35, 28, 49.9, 47.8, 58.0, 49.9, 0.941),
    (30, 25, 15, 49.4, 43.4, 66.2, 49.4, 0.810),
    (12, 10, 8, 14.9, 14.1, 20.5, 14.8, 0.940),
    (10, 9, 3, 26.5, 18.9, 53.5, 25.8, 0.604),
])
def test_capture_section_6_4_table_rows(n1, n2, m, nc, lo, hi, nh, comp):
    e = cap.estimate(hist2(n1, n2, m), 2)
    assert (e["n1"], e["n2"], e["m"], e["r"]) == (n1, n2, m, n1 + n2 - m)
    assert round(e["nChapman"], 1) == nc and round(e["nChao"], 1) == nh
    assert round(e["chapmanInterval"]["lo"], 1) == lo and round(e["chapmanInterval"]["hi"], 1) == hi
    assert round(e["completeness"], 3) == comp
    # and to 1e-9 against the transcribed Appendix formulas
    r = n1 + n2 - m
    Nc, var = a_chapman(n1, n2, m)
    Nh = a_chao(r, n1 + n2 - 2 * m, m, 2)
    clo, chi = a_ci(r, Nc, var)
    assert abs(e["nChapman"] - Nc) < 1e-9 and abs(e["var"] - var) < 1e-9 and abs(e["nChao"] - Nh) < 1e-9
    assert abs(e["chapmanInterval"]["lo"] - clo) < 1e-9 and abs(e["chapmanInterval"]["hi"] - chi) < 1e-9
    assert abs(e["completeness"] - r / max(Nc, Nh)) < 1e-9
    assert e["binds"] == ("chao" if Nh > Nc else "chapman")


def test_capture_third_pair_24_20_12_and_20_20_0():
    e = cap.estimate(hist2(24, 20, 12), 2)
    r, Ns = a_nstar(24, 20, 12)
    assert abs(e["completeness"] - r / Ns) < 1e-9
    z = cap.estimate(hist2(20, 20, 0), 2)           # m = 0: Chapman is defined, Lincoln-Petersen is not
    Nc, _ = a_chapman(20, 20, 0)
    assert abs(z["nChapman"] - Nc) < 1e-9 and Nc == 440.0


def test_capture_three_list_example():
    e = cap.estimate(THREE, 3)
    assert (e["r"], e["n1"], e["n2"], e["m"]) == (45, 41, 31, 27)
    assert (e["f1"], e["f2"], e["f3"]) == (12, 13, 20)
    assert round(e["nChapman"], 1) == 47.0 and round(e["nChao"], 1) == 48.1
    assert round(e["completeness"], 3) == 0.935
    assert e["binds"] == "chao" and e["chaoBinds"] is True
    assert e["perSession"] == {"KA": 35, "KA2": 32, "KB": 31}
    assert abs(e["nChao"] - a_chao(45, 12, 13, 3)) < 1e-9
    assert e["trigger"] is True                      # 0.935 < 0.95


def test_capture_degenerate_interval_n2_equals_m():
    e = cap.estimate(hist2(12, 8, 8), 2)             # n2 = m: one list contains the other
    r = 12
    assert e["chapmanInterval"] == {"lo": r, "hi": r, "degenerate": True}
    assert e["var"] == 0.0
    # the non-degenerate rows are not flagged
    assert cap.estimate(hist2(40, 35, 28), 2)["chapmanInterval"]["degenerate"] is False


def test_capture_trigger_threshold_is_095_not_090():
    # completeness 0.941 (row 1): below 0.95, above 0.90
    assert cap.estimate(hist2(40, 35, 28), 2)["trigger"] is True
    # n1=n2=m=20 -> r=20, N*=20 (f0=0): completeness 1.0
    assert cap.estimate(hist2(20, 20, 20), 2)["trigger"] is False
    # an exact boundary: completeness exactly 0.95 does not trigger (strictly below)
    e = cap.estimate({"101": 19, "100": 0, "001": 0}, 2)
    assert e["completeness"] == 1.0 and e["trigger"] is False


def test_capture_trigger_matches_appendix_decision_on_enumerated_outcomes():
    # every capture outcome of N=12 at p=(0.8,0.7): our trigger decision equals the Appendix decision
    N = 12
    for a in range(N + 1):
        for b in range(N - a + 1):
            for c in range(N - a - b + 1):
                if a + b + c == 0:
                    continue
                e = cap.estimate(hist2(a + b, a + c, a), 2)
                r, Ns = a_nstar(a + b, a + c, a)
                assert e["trigger"] == (r * 100 < 95 * Ns)
    assert round(a_ptrigger(20, 0.8, 0.7), 3) == 0.552          # §6.5 table row, at 0.95


def test_capture_heterogeneity_illustration_inputs_do_not_change_the_formulas():
    # expected-count mixture of Appendix A (not a sample): Chapman and Chao both report 0.975
    Ne, Nh_ = 48, 12
    n1 = Ne * .9 + Nh_ * .2
    n2 = Ne * .85 + Nh_ * .2
    m = Ne * .9 * .85 + Nh_ * .04
    r = n1 + n2 - m
    Nc, _ = cap.chapman(n1, n2, m)
    assert round(r / max(Nc, cap.chao_mh(r, n1 + n2 - 2 * m, m, 2)), 3) == 0.975


def test_capture_empty_stratum_and_tiny_stratum():
    out = cap.build(match_of({"http": hist2(40, 35, 28)}))
    s = out["strata"]
    assert s["mcp"]["r"] == 0 and s["mcp"]["completeness"] is None and s["mcp"]["trigger"] is True
    tiny = cap.estimate(hist2(4, 4, 3), 2)
    assert tiny["r"] == 5 and tiny["informative"] is False
    assert cap.estimate(hist2(40, 35, 28), 2)["informative"] is True


def test_capture_pooled_is_stratified_sum_ratio_without_interval():
    mt = match_of({"http": hist2(40, 35, 28), "mcp": hist2(10, 9, 3), "flow": hist2(30, 25, 15)})
    out = cap.build(mt)
    r = [47, 16, 40]
    Ns = [max(a_chapman(*x)[0], a_chao(rr, x[0] + x[1] - 2 * x[2], x[2], 2)) for rr, x in zip(r, [(40, 35, 28), (10, 9, 3), (30, 25, 15)])]
    assert abs(out["pooled"]["completeness"] - sum(r) / sum(Ns)) < 1e-9
    assert out["pooled"]["r"] == 103 and "interval" not in json.dumps(out["pooled"]).replace("no pooled interval", "")
    # pooled lists are NOT run through Chapman
    pr = cap.estimate({"101": 28 + 3 + 15, "100": 12 + 6 + 15, "001": 7 + 6 + 10}, 2)
    assert abs(pr["completeness"] - out["pooled"]["completeness"]) > 1e-6


def test_capture_history_bit_order_is_ka_ka2_kb():
    # KA2-only and KB-only items must land in P and C respectively (swapped order would swap them)
    e = cap.estimate({"010": 3, "001": 5, "011": 2}, 3)
    assert (e["n1"], e["n2"], e["m"]) == (5, 7, 2)
    assert e["perSession"] == {"KA": 0, "KA2": 5, "KB": 7}


def test_capture_t_comes_from_the_match_json_t_map():
    # histories 100/101/001 (10, 9, 3) hold no KA2 bit, yet t = 3 where match.json says KA2 ran (found nothing)
    two = cap.build(match_of({"http": hist2(10, 9, 3)}))["strata"]["http"]
    assert two["t"] == 2 and abs(two["nChao"] - a_chao(16, 13, 3, 2)) < 1e-9
    three = cap.build(match_of({"http": hist2(10, 9, 3)}, t={"http": 3}))["strata"]["http"]
    assert three["t"] == 3 and abs(three["nChao"] - 29.0) < 1e-9 and abs(three["nChao"] - a_chao(16, 13, 3, 3)) < 1e-9
    assert round(three["completeness"], 3) == 0.552 and round(two["completeness"], 3) == 0.604
    # the map wins over the KA2 bits, in both directions
    with pytest.raises(ValueError, match="KA2"):                     # t = 2 cannot coexist with a KA2 bit (N-3)
        cap.build(match_of({"flow": THREE}, t={"flow": 2}))
    legal = cap.build(match_of({"flow": hist2(5, 5, 3)}, t={"flow": 3}))["strata"]["flow"]   # KA2 ran, found nothing
    assert legal["t"] == 3
    assert cap.build(match_of({"flow": THREE}))["strata"]["flow"]["t"] == 3
    other = cap.build(match_of({"http": hist2(10, 9, 3)}, t={"mcp": 3}))["strata"]
    assert other["http"]["t"] == 2 and other["mcp"]["t"] == 3


def test_capture_refuses_a_missing_or_invalid_t_map():
    mt = match_of({"http": hist2(10, 9, 3)})
    del mt["t"]
    with pytest.raises(ValueError):
        cap.build(mt)
    for bad in ({"http": 2}, {**dict.fromkeys(STRATA, 2), "http": 4}):
        mt["t"] = bad
        with pytest.raises(ValueError):
            cap.build(mt)


def test_capture_out_of_scope_items_are_left_out_and_counted():
    mt = match_of({"flow": hist2(10, 9, 3)})
    for g in mt["records"][:2]:
        g["inScope"] = False
    out = cap.build(mt)
    assert out["strata"]["flow"]["r"] == 14 and out["outOfScope"] == 2


@pytest.mark.parametrize("bad", [None, "yes", 1, "missing"])
def test_capture_refuses_a_record_without_a_boolean_inscope(bad):
    mt = match_of({"flow": hist2(10, 9, 3)})
    if bad == "missing":
        del mt["records"][3]["inScope"]
    else:
        mt["records"][3]["inScope"] = bad
    with pytest.raises(ValueError):
        cap.build(mt)


def test_capture_trigger_boundary_exactly_095_does_not_fire():
    e = cap.estimate(hist2(62, 62, 48), 2)                       # r = 76, Chapman 80.0 exactly, Chao 79.857
    assert e["r"] == 76 and e["nChapman"] == 80.0 and e["nStar"] == 80.0 and e["completeness"] == 0.95
    assert e["trigger"] is False
    assert cap.estimate(hist2(62, 62, 47), 2)["trigger"] is True       # one fewer match: below 0.95


def test_capture_informative_threshold_is_r_at_least_8():
    assert cap.estimate({"101": 7}, 2)["informative"] is False
    assert cap.estimate({"101": 8}, 2)["informative"] is True


def test_capture_hops_are_counted_per_member_record():
    mt = match_of({"flow": {"101": 2}})
    mt["records"][0]["members"] = [{"hops": 1}, {"hops": 2}]
    mt["records"][1]["members"] = [{"hops": 1}]
    out = cap.build(mt)
    assert out["strata"]["flow"]["hops"] == {"1": 2, "2": 1}
    assert out["strata"]["http"]["hops"] == {}


def test_capture_rejects_bad_history_and_stratum():
    with pytest.raises(ValueError):
        cap.build(match_of({"http": {"10": 1}}))
    with pytest.raises(ValueError):
        cap.build(match_of({"nope": {"100": 1}}))


def test_capture_cli_roundtrip_and_determinism(tmp_path):
    mp = tmp_path / "match.json"
    mp.write_text(json.dumps(match_of({"http": hist2(40, 35, 28), "flow": THREE})), encoding="utf-8")
    outs = []
    for i in range(2):
        o = tmp_path / f"c{i}.json"
        r = subprocess.run([sys.executable, str(TOOLS / "capture.py"), "--match", str(mp), "--out", str(o)],
                           capture_output=True, text=True, cwd=CPG)
        assert r.returncode == 0, r.stderr
        outs.append(o.read_bytes())
    assert outs[0] == outs[1]
    d = json.loads(outs[0])
    assert round(d["strata"]["flow"]["nChao"], 1) == 48.1 and d["schema"] == "ccg-capture-v1"
    assert "http: r=47" in r.stdout and "itemId" not in r.stdout
    bad = subprocess.run([sys.executable, str(TOOLS / "capture.py"), "--match", str(tmp_path / "none.json"), "--out", str(tmp_path / "x")],
                         capture_output=True, text=True, cwd=CPG)
    assert bad.returncode == 4


# =============================================================================================
# split_manifest.py: synthetic union builders
# =============================================================================================
REPO = mk.REPO_ID


def sym(path, q):
    return {"type": "symbol", "repoId": REPO, "path": path, "qname": q}


def cfg(path, kp):
    return {"type": "config", "repoId": REPO, "path": path, "keyPath": kp}


def con(key, kind="mcp_tool"):
    return {"type": "contract", "kind": kind, "keyNorm": key}


def unit(root, lang="python"):
    return {"type": "unit", "repoId": REPO, "rootPath": root, "lang": lang}


def rec(anchor, impacted, stratum, counting=True, provider=None, serves=None, tag=""):
    aid, iid = mk.locator_id(anchor), mk.locator_id(impacted)
    item = mk._ids.tuple_id(["evalItem", aid, iid])
    member = {"writer": "KA", "strategy": "provider-first", "anchor": anchor, "impacted": impacted,
              "provider": provider, "serves": serves or [], "lines": [1], "chain": [],
              "original": {"schema": "ccg-key-v1", "anchor": anchor, "impacted": impacted, "tag": tag}}
    return {"itemId": item, "anchorId": aid, "impactedId": iid, "derivedStratum": stratum, "counting": counting,
            "inScope": True, "members": [member], "sessions": ["KA"], "history": "100"}


_n = [0]


def single(stratum, counting=True, tag=""):
    """One anchor, one item: its own cluster."""
    _n[0] += 1
    return rec(sym(f"f{_n[0]}.py", f"f{_n[0]}.py::f"), sym(f"g{_n[0]}.py", f"g{_n[0]}.py::g"), stratum, counting, tag=tag)


def commit(i: int) -> str:
    return f"{i:040x}"


def run_split(records, units=(), contact=(), commit_=None):
    cid = {mk.locator_id(a) if isinstance(a, dict) else a for a in contact}
    return sm.split(records, list(units), cid, commit_ or commit(1))


def assert_invariants(res, C=()):
    side, S, co = res["side"], res["S"], res["cluster_of"]
    for c in res["clusters"]:
        if c["anchors"] & set(C):
            assert side[c["id"]] == "dev"
        if side[c["id"]] in ("held", "reserve"):             # step 3b's invariant at the end
            for u in c["units"]:
                for a in S[u]:
                    k = co.get(a)
                    assert k is None or k == c["id"] or side[k] != "dev", "unit held with a served anchor in dev"


def bulk(n_by_stratum, counting=True):
    out = []
    for s, n in n_by_stratum.items():
        out += [single(s, counting) for _ in range(n)]
    return out


def sides_by_stratum(res):
    hs = {}
    for c in res["clusters"]:
        hs.setdefault(res["side"][c["id"]], []).append(c)
    return hs


# --- the seeded split ----------------------------------------------------------------------

def test_seed_rank_and_share_hand_computed():
    recs = bulk({"http": 10})
    ev = commit(7)
    res = run_split(recs, commit_=ev)
    assert res["seed"] == "ccg-p0f-split-v1|" + ev
    order = sorted(recs, key=lambda g: hashlib.sha256(f"ccg-p0f-split-v1|{ev}|{g['anchorId']}".encode()).hexdigest())
    # 10 items -> ceil(0.7*10) = 7 held; the first 7 by rank. (The minimum loop then tops the stratum up
    # to the rest because every stratum is short of 8 and H < 75; so check the SHARE step via the log.)
    moved = [l for l in res["log"] if l["rule"] == "minimum-stratum"]
    assert len(moved) == 1
    held_by_rank = [g["anchorId"] for g in order[:7]]
    assert moved[0]["cluster"] == order[7]["anchorId"]       # the lowest-rank dev cluster is the one moved in
    assert all(res["side"][res["cluster_of"][a]] == "held" for a in held_by_rank)
    assert res["report"]["Hs"]["http"] == 10      # the counting top-up then pulls the rest in (H < 75)


def test_share_targets_8_20_30_40_give_6_14_21_28():
    for n, t in [(8, 6), (20, 14), (30, 21), (40, 28)]:
        assert sm._ceil_share(n) == t
    assert sm._ceil_share(10) == 7 and sm._ceil_share(0) == 0


def test_share_step_alone_leaves_ceil_70_percent_when_minimums_hold():
    # 40 http items: 28 held by the share; H_http = 28 >= 8; H < 75 but no dev cluster is *needed* for strata,
    # so only the counting top-up moves clusters in. Check the share before it via the log.
    res = run_split(bulk({"http": 40}), commit_=commit(3))
    topup = [l for l in res["log"] if l["rule"] == "minimum-counting"]
    assert len(topup) == 12                                   # 40 - 28 dev clusters, all pulled in for H < 75
    assert res["report"]["H"] == 40 and res["report"]["trigger"] is True


def test_seed_prefix_is_part_of_the_rank():
    recs = bulk({"http": 30})
    a = run_split(recs, commit_=commit(5))
    order = [c["key"] for c in a["clusters"]]
    ev = commit(5)
    expect = sorted((g["anchorId"] for g in recs), key=lambda x: hashlib.sha256(f"ccg-p0f-split-v1|{ev}|{x}".encode()).hexdigest())
    assert order == expect
    other = sorted((g["anchorId"] for g in recs), key=lambda x: hashlib.sha256(f"{ev}|{x}".encode()).hexdigest())
    assert order != other


def test_hand_computed_assignment_30_anchors_five_strata_with_contact_and_deficit():
    recs = bulk({"http": 10, "mcp": 6, "scripts": 6, "wiring": 4, "flow": 4})
    contacted = [g["anchorId"] for g in recs if g["derivedStratum"] == "http"][:2]
    ev = commit(11)
    res = run_split(recs, contact=contacted, commit_=ev)
    assert_invariants(res, contacted)
    rep = res["report"]
    # share: http 8 non-contact -> 6; mcp 6 -> 5; scripts 6 -> 5; wiring 4 -> 3; flow 4 -> 3.
    # All are < 8 so step 4 pulls every eligible dev cluster in, wiring (3) before flow (3) by stratum order.
    # Hand trace: held 6/5/5/3/3 (http, mcp, scripts, wiring, flow); wiring->4, flow->4; then wiring and flow have
    # no dev cluster left (marked, in that order); mcp 5->6, scripts 5->6; http 6->7->8; mcp and scripts are marked.
    firsts = [l["stratum"] for l in res["log"] if l["rule"] == "minimum-stratum"]
    assert firsts == ["wiring", "flow", "mcp", "scripts", "http", "http"]
    held_items = {a for a in (g["anchorId"] for g in recs) if res["side"][res["cluster_of"][a]] == "held"}
    assert held_items == {g["anchorId"] for g in recs} - set(contacted)
    assert rep["Hs"] == {"http": 8, "mcp": 6, "scripts": 6, "wiring": 4, "flow": 4}
    assert rep["markedStrata"] == ["wiring", "flow", "mcp", "scripts"] and rep["trigger"] is True
    assert rep["devClusters"] == {"contacted": 2, "other": 0}
    # a byte-identical rerun
    again = run_split(recs, contact=contacted, commit_=ev)
    assert again["report"] == rep and again["log"] == res["log"]


def test_primary_stratum_tie_goes_to_the_earlier_stratum():
    u = unit("srv")
    t = con("t1")
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t]),
            rec(t, sym("srv/a.py", "srv/a.py::h"), "mcp", provider={"path": "srv/a.py"})]
    res = run_split(recs, units=[{"rootPath": "srv", "lang": "python"}])
    assert len(res["clusters"]) == 1 and res["clusters"][0]["primary"] == "mcp"       # mcp before wiring


# --- B1a: unit + probe ----------------------------------------------------------------------

def server(root, tools_items, probe_items=None, extra_wiring=None):
    """unit `root` serving tools {key: n_items}; returns (records, unit_anchor, {key: contract anchor})."""
    u = unit(root)
    tools = {k: con(k) for k in tools_items}
    recs = [rec(u, cfg(".mcp.json", f"mcpServers.{root}"), "wiring", serves=list(tools.values()))]
    for k, n in tools_items.items():
        for j in range(n):
            recs.append(rec(tools[k], sym(f"{root}/{k}.py", f"{root}/{k}.py::h{j}"), "mcp", provider={"path": f"{root}/{k}.py"}))
    return recs, u, tools


def test_b1a_unit_and_probe_share_a_cluster_and_a_side_and_contact_sends_both_to_dev():
    recs, u, tools = server("srv", {"p": 2})                  # probe p has its own items
    units = [{"rootPath": "srv", "lang": "python"}]
    recs += bulk({"mcp": 12, "wiring": 12})
    for ev in range(1, 12):
        res = run_split(recs, units, commit_=commit(ev))
        k = res["cluster_of"][mk.locator_id(u)]
        assert res["cluster_of"][mk.locator_id(tools["p"])] == k
        assert res["probe"][mk.locator_id(u)] == mk.locator_id(tools["p"])
        assert_invariants(res)
    for contact in (tools["p"], u):
        res = run_split(recs, units, contact=[contact], commit_=commit(2))
        k = res["cluster_of"][mk.locator_id(u)]
        assert res["side"][k] == "dev"
        assert [l for l in res["log"] if l["rule"] == "contact"][0]["cluster"] in res["clusters"][k]["anchors"]


def test_b1a_probe_with_no_items_of_its_own_is_still_clustered_with_the_unit():
    u = unit("srv")
    t1, t2 = con("t1"), con("t2")
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t1, t2])]
    recs += [rec(t2, sym("srv/b.py", f"srv/b.py::h{j}"), "mcp", provider={"path": "srv/b.py"}) for j in range(3)]
    res = run_split(recs, [{"rootPath": "srv", "lang": "python"}])
    # t1 has 0 items in U: it is the probe; it is an anchor of the unit's cluster though it holds no item
    assert res["probe"][mk.locator_id(u)] == mk.locator_id(t1)
    k = res["cluster_of"][mk.locator_id(u)]
    assert mk.locator_id(t1) in res["clusters"][k]["anchors"]
    res2 = run_split(recs, [{"rootPath": "srv", "lang": "python"}], contact=[t1])         # contacting the empty probe
    assert res2["side"][res2["cluster_of"][mk.locator_id(u)]] == "dev"


# --- B1b: fresh items (library half of regate-list) -----------------------------------------

def test_b1b_fresh_items_route_by_cluster_and_contact():
    recs = bulk({s: 40 for s in STRATA})
    c_anchor = recs[0]["anchorId"]
    res = run_split(recs, contact=[c_anchor], commit_=commit(4))
    held = next(g for g in recs if res["side"][res["cluster_of"][g["anchorId"]]] == "held")
    dev = next(g for g in recs if res["side"][res["cluster_of"][g["anchorId"]]] == "dev" and g["anchorId"] != c_anchor)
    fresh_dev = single("http")
    fresh_dev.update(anchorId=dev["anchorId"], itemId="fd")
    fresh_c = single("http")
    fresh_c.update(anchorId=c_anchor, itemId="fc")
    fresh_kept = single("http")
    fresh_kept.update(anchorId=held["anchorId"], itemId="fk")
    fresh_new = single("http")
    ghost = "f" * 32                                              # a C anchor that holds no item of U: still dev
    res_ghost = run_split(recs, contact=[ghost], commit_=commit(4))
    fresh_ghost = single("http")
    fresh_ghost.update(anchorId=ghost, itemId="fg")
    assert sm.route_fresh(res_ghost, [fresh_ghost])[0]["flag"] == "dev"
    flags = {r["itemId"]: r["flag"] for r in sm.route_fresh(res, [fresh_dev, fresh_c, fresh_kept, fresh_new])}
    assert flags["fd"] == "dev" and flags["fc"] == "dev" and flags["fk"] == "kept-cluster-fresh"
    assert flags[fresh_new["itemId"]] == "fresh-cluster"


# --- M2 / M6 / reserve ------------------------------------------------------------------------

def test_m2_flow_stuck_marked_then_other_stratum_topped_up_and_terminates():
    recs = bulk({"http": 20, "mcp": 12, "wiring": 12, "scripts": 10, "flow": 3})
    res = run_split(recs, commit_=commit(9))
    # flow: 3 held, no dev cluster has a flow item. scripts: ceil(7) = 7 held, 3 dev, H_scripts = 7 < 8.
    rep = res["report"]
    assert "flow" in rep["markedStrata"] and rep["trigger"] is True
    assert rep["Hs"]["flow"] == 3
    stratum_moves = [l for l in res["log"] if l["rule"] == "minimum-stratum"]
    assert [l["stratum"] for l in stratum_moves] == ["scripts"]          # flow was marked (no log), scripts moved
    assert rep["Hs"]["scripts"] >= 8
    assert_invariants(res)


def test_m6_all_items_at_least_75_but_counting_below_75_triggers():
    recs = bulk({"flow": 40}, counting=False) + bulk({"http": 15, "mcp": 15, "scripts": 15, "wiring": 15})
    res = run_split(recs, commit_=commit(6))
    rep = res["report"]
    assert rep["H_all"] == 88 and rep["H"] == 60 and rep["markedStrata"] == []
    assert rep["trigger"] is True and rep["reserveCounting"] == 0           # step 6 skipped
    # the same union with every item counting does not trigger
    ok = bulk({"flow": 40}) + bulk({"http": 15, "mcp": 15, "scripts": 15, "wiring": 15})
    assert run_split(ok, commit_=commit(6))["report"]["trigger"] is False


def test_trigger_when_a_stratum_is_marked_even_with_h_at_least_75():
    recs = bulk({"http": 80})                                              # other strata empty -> marked
    rep = run_split(recs, commit_=commit(2))["report"]
    assert rep["H"] >= 75 and rep["markedStrata"] == ["mcp", "scripts", "wiring", "flow"] and rep["trigger"] is True


def test_reserve_takes_35_counting_items_from_the_highest_ranks_and_keeps_minimums():
    recs = bulk({s: 40 for s in STRATA})
    ev = commit(21)
    res = run_split(recs, commit_=ev)
    rep = res["report"]
    assert rep["trigger"] is False and rep["reserveCounting"] == 35 and rep["reserveClusters"] == 35
    held_after = rep["H"]
    assert held_after == 105 and all(v >= 8 for v in rep["Hs"].values())
    # the reserve is the 35 highest-rank clusters among those held before the reserve step
    reserve = sorted((c for c in res["clusters"] if res["side"][c["id"]] == "reserve"), key=lambda c: c["rank"])
    others = [c for c in res["clusters"] if res["side"][c["id"]] == "held"]
    assert min(c["rank"] for c in reserve) > max(c["rank"] for c in others) or any(
        c["rank"] > reserve[0]["rank"] for c in others) is False


def test_reserve_never_takes_gate1_below_its_minimums():
    recs = bulk({"http": 60, "mcp": 20, "scripts": 20, "wiring": 20, "flow": 10})
    found = None
    for i in range(1, 400):
        res = run_split(recs, commit_=commit(i))
        reserve = [c for c in res["clusters"] if res["side"][c["id"]] == "reserve"]
        if res["report"]["trigger"] is False and any(c["primary"] == "flow" for c in res["clusters"] if res["side"][c["id"]] == "held"):
            top = sorted((c for c in res["clusters"] if res["side"][c["id"]] != "dev"), key=lambda c: c["rank"], reverse=True)[:25]
            if any(c["primary"] == "flow" for c in top):          # the descending walk meets a flow cluster early
                found = res
                break
    assert found is not None
    rep = found["report"]
    assert rep["Hs"]["flow"] == 8                                  # flow sat exactly at its minimum: never reserved
    assert rep["H"] == 75 and rep["reserveCounting"] == 17        # 92 held counting - 17 = 75, then every move would breach
    assert all(v >= 8 for v in rep["Hs"].values())
    assert not any(c["primary"] == "flow" for c in found["clusters"] if found["side"][c["id"]] == "reserve")


# --- P2-1: served anchors ------------------------------------------------------------------------

UNITS = [{"rootPath": "srv", "lang": "python"}]


def test_p2_1a_unit_with_a_dev_non_probe_tool_ends_in_dev_logged_served_anchor():
    recs, u, tools = server("srv", {"t1": 1, "t2": 1})
    recs += bulk({"mcp": 14, "wiring": 14})
    n_logged = 0
    for i in range(1, 40):
        res = run_split(recs, UNITS, contact=[tools["t2"]], commit_=commit(i))
        assert_invariants(res, [mk.locator_id(tools["t2"])])
        k = res["cluster_of"][mk.locator_id(u)]
        assert res["side"][k] == "dev"
        n_logged += any(l["rule"] == "served-anchor" for l in res["log"])
        assert res["report"]["servedAnchorMoves"] == sum(l["rule"] == "served-anchor" for l in res["log"])
    assert n_logged > 0                                            # the rule fired in some seeds (non-vacuous)


def test_p2_1a_with_t2_held_out_the_unit_follows_its_own_rank():
    recs, u, tools = server("srv", {"t1": 1, "t2": 1})
    recs += bulk({"mcp": 14, "wiring": 14})
    seen_held = 0
    for i in range(1, 40):
        res = run_split(recs, UNITS, commit_=commit(i))
        assert_invariants(res)
        k2 = res["cluster_of"][mk.locator_id(tools["t2"])]
        if res["side"][k2] == "held" and not any(l["cluster"] == res["clusters"][k2]["key"] for l in res["log"]):
            seen_held += 1                                          # T2 was never in dev, not even transiently
            assert not any(l["rule"] == "served-anchor" for l in res["log"])
    assert seen_held > 0


def test_p2_1b_single_tool_server_is_unaffected():
    recs, u, tools = server("srv", {"only": 2})
    recs += bulk({"mcp": 14, "wiring": 14})
    for i in range(1, 25):
        res = run_split(recs, UNITS, commit_=commit(i))
        assert not any(l["rule"] == "served-anchor" for l in res["log"])
        assert res["cluster_of"][mk.locator_id(u)] == res["cluster_of"][mk.locator_id(tools["only"])]


def test_p2_1c_step_4_respects_the_rule_and_marks_wiring():
    recs, u, tools = server("srv", {"t1": 1, "t2": 1})
    t2 = mk.locator_id(tools["t2"])
    res = run_split(recs, UNITS, contact=[t2], commit_=commit(5))
    uk = res["cluster_of"][mk.locator_id(u)]
    assert res["side"][uk] == "dev"                                 # T2 contacted -> u in dev ...
    assert "wiring" in res["report"]["markedStrata"]               # ... and step 4 cannot bring it back
    assert res["report"]["trigger"] is True
    assert not any(l["rule"].startswith("minimum") and l["cluster"] in res["clusters"][uk]["anchors"] for l in res["log"])
    assert_invariants(res, [t2])


def test_p2_1c_step_4_may_move_a_unit_once_its_served_tools_are_no_longer_in_dev():
    # no contact; make u's cluster dev through the share, with t2 *also* dev or held: when nothing served is in dev
    # the unit is eligible. Invariant check across seeds, and at least one seed where a unit is topped up.
    recs, u, tools = server("srv", {"t1": 1, "t2": 1})
    recs += bulk({"mcp": 10})
    topped = 0
    for i in range(1, 60):
        res = run_split(recs, UNITS, commit_=commit(i))
        assert_invariants(res)
        uc = res["clusters"][res["cluster_of"][mk.locator_id(u)]]
        topped += any(l["rule"].startswith("minimum") and l["cluster"] == uc["key"] for l in res["log"])
    assert topped > 0


def test_p2_1d_regate_moves_kept_and_reserve_units_without_counting_as_misses():
    t2 = con("t2")
    u, u2 = unit("srv"), unit("srv2")
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[con("t1"), t2]),
            rec(u2, cfg(".mcp.json", "mcpServers.srv2"), "wiring", serves=[con("t1b"), t2]),
            rec(t2, sym("srv/t2.py", "srv/t2.py::h"), "mcp", provider={"path": "srv/t2.py"}),
            ]                                                       # t1 and t1b hold no items: they are the probes
    units = [{"rootPath": "srv", "lang": "python"}, {"rootPath": "srv2", "lang": "python"}]
    res = run_split(recs, units)
    co, side = res["cluster_of"], res["side"]
    for c in res["clusters"]:
        side[c["id"]] = "held"
    side[co[mk.locator_id(u2)]] = "reserve"
    assert co[mk.locator_id(u)] != co[mk.locator_id(t2)] != co[mk.locator_id(u2)]
    moved = sm.regate_served(res, [mk.locator_id(t2)])             # a miss on T2 -> its cluster to dev
    assert side[co[mk.locator_id(t2)]] == "dev"
    assert side[co[mk.locator_id(u)]] == "dev" and side[co[mk.locator_id(u2)]] == "dev"
    assert moved == 2                                              # the two unit clusters, counted apart from the miss
    fresh = rec(u, cfg(".mcp.json", "mcpServers.srv.other"), "wiring")
    assert sm.route_fresh(res, [fresh])[0]["flag"] == "dev"


def test_p2_1e_probe_is_the_served_point_with_fewest_items_then_smaller_cpid():
    for counts, expect_zero in (((0, 3, 3), True),):
        recs, u, tools = server("srv", {"a": 0, "b": 3, "c": 3})
        res = run_split(recs, UNITS)
        assert res["probe"][mk.locator_id(u)] == mk.locator_id(tools["a"])
    recs, u, tools = server("srv", {"b": 3, "c": 3})
    res = run_split(recs, UNITS)
    smaller = min(mk.locator_id(tools["b"]), mk.locator_id(tools["c"]))
    assert res["probe"][mk.locator_id(u)] == smaller
    # fewest items beats a smaller cpId
    recs, u, tools = server("srv", {"b": 1, "c": 3})
    pick = run_split(recs, UNITS)["probe"][mk.locator_id(u)]
    assert pick == mk.locator_id(tools["b"])
    recs, u, tools = server("srv", {"b": 3, "c": 1})
    assert run_split(recs, UNITS)["probe"][mk.locator_id(u)] == mk.locator_id(tools["c"])


def test_served_set_comes_from_provider_path_and_from_serves():
    u = unit("srv")
    t_prov = con("viaprovider")                                    # served because provider.path is under srv
    t_serves = con("viaserves")                                    # served because it is in the unit's serves[]
    t_other = con("elsewhere")
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t_serves]),
            rec(t_prov, sym("srv/a.py", "srv/a.py::h"), "mcp", provider={"path": "srv/a.py"}),
            rec(t_serves, sym("lib/b.py", "lib/b.py::h"), "mcp", provider={"path": "lib/b.py"}),
            rec(t_other, sym("lib/c.py", "lib/c.py::h"), "mcp", provider={"path": "lib/c.py"})]
    S = sm.served_sets(recs, UNITS)[mk.locator_id(u)]
    assert S == {mk.locator_id(t_prov), mk.locator_id(t_serves)}


def test_served_anchor_fixpoint_repeats_until_nothing_changes():
    # u1 serves p2, the probe in u2's cluster; u2 goes to dev by contact on its other tool t2b; u1 must follow.
    u1, u2 = unit("s1"), unit("s2")
    y, p2, t2b = con("y"), con("p2"), con("t2b")
    recs = [rec(u1, cfg(".mcp.json", "mcpServers.s1"), "wiring", serves=[y, p2]),     # y: no items, u1's probe
            rec(u2, cfg(".mcp.json", "mcpServers.s2"), "wiring", serves=[p2, t2b]),   # p2: 1 item, u2's probe
            rec(p2, sym("s2/p.py", "s2/p.py::h"), "mcp", provider={"path": "s2/p.py"}),
            rec(t2b, sym("s2/t.py", "s2/t.py::h"), "mcp", provider={"path": "s2/t.py"}),
            rec(t2b, sym("s2/t.py", "s2/t.py::h2"), "mcp", provider={"path": "s2/t.py"})]
    recs += bulk({"mcp": 6, "wiring": 6})
    units = [{"rootPath": "s1", "lang": "python"}, {"rootPath": "s2", "lang": "python"}]
    chained = 0
    for i in range(1, 80):
        res = run_split(recs, units, contact=[t2b], commit_=commit(i))
        assert_invariants(res, [mk.locator_id(t2b)])
        k1, k2 = res["cluster_of"][mk.locator_id(u1)], res["cluster_of"][mk.locator_id(u2)]
        assert k1 != k2 and res["probe"][mk.locator_id(u1)] == mk.locator_id(y)
        assert res["side"][k2] == "dev"
        served = [l for l in res["log"] if l["rule"] == "served-anchor"]
        if len(served) == 2 and res["clusters"][k1]["rank"] < res["clusters"][k2]["rank"]:
            chained += 1                                           # u1 is visited before u2 moves: needs a second pass
    assert chained > 0


def test_per_server_cluster_alternative_is_not_what_split_does():
    # 6 tools on one server: they are independent clusters (the rejected alternative merges them into one)
    recs, u, tools = server("srv", {f"t{i}": 1 for i in range(6)})
    res = run_split(recs, UNITS, commit_=commit(3))
    ks = {res["cluster_of"][mk.locator_id(t)] for t in tools.values()}
    assert len(ks) >= 5                                            # at most the probe shares the unit's cluster


# --- outputs: files, manifest, verify ----------------------------------------------------------

def write_work(tmp, recs, units=UNITS, contact_ids=()):
    (tmp / "match.json").write_text(json.dumps({"schema": "ccg-match-v1", "records": recs, "t": dict.fromkeys(STRATA, 2)}), encoding="utf-8")
    (tmp / "context.json").write_text(json.dumps({"units": list(units)}), encoding="utf-8")
    (tmp / "contact.jsonl").write_text("".join(json.dumps({"anchorId": a, "locator": {}, "source": "S14"}) + "\n" for a in contact_ids), encoding="utf-8")


def cli(*args):
    return subprocess.run([sys.executable, str(TOOLS / "split_manifest.py"), *map(str, args)], capture_output=True, text=True, cwd=CPG)


def make_split(tmp, recs=None, contact_ids=(), ev=None):
    recs = recs or bulk({s: 40 for s in STRATA})
    write_work(tmp, recs, contact_ids=contact_ids)
    ev = ev or commit(21)
    r = cli("split", "--match", tmp / "match.json", "--context", tmp / "context.json", "--contact", tmp / "contact.jsonl",
            "--eval-commit", ev, "--heldout-dir", tmp / "E" / "heldout", "--dev-dir", tmp / "dev",
            "--report", tmp / "E" / "work" / "split.json")
    assert r.returncode == 0, r.stderr
    return r, ev, recs


def test_split_cli_files_counts_only_and_determinism(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    ra, ev, recs = make_split(a)
    rb, _, _ = make_split(b, recs=recs)
    assert "trigger: false" in ra.stdout and "reserve-counting: 35" in ra.stdout
    assert "f1.py" not in ra.stdout and "itemId" not in ra.stdout
    ho = a / "E" / "heldout"
    assert sorted(p.name for p in ho.iterdir()) == sorted([f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl", "exposure.jsonl"])
    assert (ho / "exposure.jsonl").read_bytes() == b""
    for p in ho.iterdir():
        assert (b / "E" / "heldout" / p.name).read_bytes() == p.read_bytes()      # byte-identical rerun
        data = p.read_bytes()
        assert data == b"" or data.endswith(b"\n")
        ids = [json.loads(l)["itemId"] for l in data.decode().splitlines()]
        assert ids == sorted(ids)
    total = sum(len((ho / f"ac5-{s}.jsonl").read_text().splitlines()) for s in STRATA) + len((ho / "ac5-reserve.jsonl").read_text().splitlines())
    dev_n = sum(len((a / "dev" / f"ac5-{s}.jsonl").read_text().splitlines()) for s in STRATA)
    assert total + dev_n == 200 and dev_n == 60
    first = json.loads((ho / "ac5-http.jsonl").read_text().splitlines()[0])
    assert first["originals"] and first["counting"] is True and first["derivedStratum"] == "http"


def test_manifest_and_verify_fail_on_a_one_byte_change(tmp_path):
    make_split(tmp_path)
    ev = commit(21)
    (tmp_path / "E" / "work").mkdir(parents=True, exist_ok=True)
    (tmp_path / "E" / "work" / "match.json").write_text("{}", encoding="utf-8")
    man = tmp_path / "heldout-manifest.json"
    ho = tmp_path / "E" / "heldout"
    r = cli("manifest", "--heldout-dir", ho, "--eval-commit", ev, "--protocol", "P@1.5+Q@1.4", "--contact", tmp_path / "contact.jsonl",
            "--out", man, "--raw-root", tmp_path / "E", "--raw", "work/match.json")
    assert r.returncode == 0, r.stderr
    d = json.loads(man.read_text())
    assert d["schema"] == "ccg-heldout-manifest-v1" and d["splitSeed"] == "ccg-p0f-split-v1|" + ev
    assert [f["name"] for f in d["files"]] == [f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl"]
    f0 = d["files"][0]
    assert set(f0) == {"name", "sha256", "bytes", "items", "countingItems", "anchors"}
    assert f0["items"] == f0["countingItems"] == f0["anchors"] and f0["bytes"] == (ho / "ac5-http.jsonl").stat().st_size
    assert d["contactAnchors"] == 0 and d["rawArtifacts"][0]["path"] == "work/match.json"
    assert "originals" not in man.read_text() and "f1.py" not in man.read_text()      # counts and hashes only
    ok = cli("--verify", man, "--heldout-dir", ho, "--raw-root", tmp_path / "E")
    assert ok.returncode == 0 and "mismatches: 0" in ok.stdout
    # one byte flipped in a held-out file: same length, so only the hash can notice
    p = ho / "ac5-mcp.jsonl"
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    p.write_bytes(bytes(raw))
    bad = cli("--verify", man, "--heldout-dir", ho, "--raw-root", tmp_path / "E")
    assert bad.returncode == 1 and "ac5-mcp.jsonl: sha256" in bad.stdout
    raw[len(raw) // 2] ^= 0x01
    p.write_bytes(bytes(raw))
    assert cli("--verify", man, "--heldout-dir", ho, "--raw-root", tmp_path / "E").returncode == 0
    # an appended line breaks sha256, bytes and items
    p.write_bytes(p.read_bytes() + b'{"x":1}\n')
    out = cli("--verify", man, "--heldout-dir", ho, "--raw-root", tmp_path / "E").stdout
    assert "bytes" in out and "items" in out
    p.unlink()
    assert "missing" in cli("--verify", man, "--heldout-dir", ho, "--raw-root", tmp_path / "E").stdout
    # a raw artifact changed
    (tmp_path / "E" / "work" / "match.json").write_text("{ }", encoding="utf-8")
    assert "work/match.json: sha256" in cli("--verify", man, "--heldout-dir", ho, "--raw-root", tmp_path / "E").stdout


def test_manifest_seed_is_checked(tmp_path):
    make_split(tmp_path)
    ho = tmp_path / "E" / "heldout"
    d = sm.build_manifest(ho, commit(21), "P", 0)
    assert sm.verify_manifest(d, ho) == []
    d["splitSeed"] = "ccg-p0f-split-v1|" + commit(22)
    assert sm.verify_manifest(d, ho) == ["splitSeed"]


# --- check-report (EP §9.4 v1.5, EP2 §1.4) ------------------------------------------------------

def held(qname="a.py::f", path="a.py", extra=None):
    o = {"schema": "ccg-key-v1", "anchor": sym(path, qname), "impacted": sym("b.py", "b.py::zzz_impacted"), "lines": [3]}
    o.update(extra or {})
    g = rec(o["anchor"], o["impacted"], "flow")
    g["originals"] = [o]
    return g


def hits(report, ho, dev=(), contact=()):
    return sm.check_report(report, list(ho), list(dev), list(contact))[0]


def test_leak_planted_heldout_qname_fails_even_when_shorter_than_8():
    ho = [held("a.py::f")]
    assert len("a.py::f") < 8                                          # v1.5: not bare (has '::'), so it stays a probe
    assert hits("the held-out item a.py::f was missed", ho)
    assert not hits("nothing to see here", ho)


def test_leak_p2_3_trailing_punctuation_cases():
    ho = [held("a.py::f")]
    for txt in ("held-out item a.py::f.", "a.py::f: missed", "a.py::f..", "./a.py::f", "see  a.py::f\n"):
        assert hits(txt, ho), txt
    for txt in ("only a.py::f.g here", "x/a.py::f", "a.py::fg", "a.py::f::g", "a.py::f.:g"):
        assert not hits(txt, ho), txt


def test_leak_pathline_boundaries():
    ho = [held("a.py::f", extra={"anchor": sym("a.py", "a.py::f")})]
    ho[0]["originals"][0]["impacted"] = sym("pkg/a.py", "pkg/a.py::q", ) | {"line": 12}
    assert hits("pkg/a.py:12.", ho) and not hits("pkg/a.py:123", ho)


def test_leak_m4_contacted_anchor_and_dev_f_g():
    ho = [held("a.py::f")]
    assert not hits("a.py::f.g only", ho, dev=[held("a.py::f.g")])          # dev holds only a.py::f.g
    assert hits("leak a.py::f here", ho, dev=[held("a.py::f.g")])           # containment does not remove the probe
    assert not hits("contacted a.py::f is public", ho, contact=[sym("a.py", "a.py::f")])
    assert hits("a.py::f", ho, contact=[sym("a.py", "a.py::f.g")])
    assert not hits("a.py::f", ho, dev=[held("a.py::f")])                    # exact equality with a dev value removes it


def test_leak_bare_probe_floor_is_one_named_constant():
    assert sm.MIN_BARE_PROBE_LEN == 8 and not hasattr(sm, "is_bare")
    bare = held("query")
    bare["originals"][0]["anchor"] = sym("a.py", "query")                  # qname without '::' -> bare
    assert not hits("calls query now", [bare])                              # 5 < 8, bare: dropped
    old = sm.MIN_BARE_PROBE_LEN
    try:
        sm.MIN_BARE_PROBE_LEN = 3
        assert hits("calls query now", [bare])                              # the constant reaches leak_check
        sm.MIN_BARE_PROBE_LEN = 6
        assert not hits("calls query now", [bare])
        sm.MIN_BARE_PROBE_LEN = 5
        assert hits("calls query now", [bare])                              # length 5 is not below a floor of 5
    finally:
        sm.MIN_BARE_PROBE_LEN = old
    assert mk._bare_short("query") and not hits("calls query now", [bare])
    long_bare = held("querytool_x")
    long_bare["originals"][0]["anchor"] = sym("a.py", "querytool_x")
    assert hits("calls querytool_x now", [long_bare])                        # >= 8 stays


def test_match_keys_leak_check_default_floor_is_8():
    t7 = held("abcdefg")
    t7["originals"][0]["anchor"] = sym("a.py", "abcdefg")                  # bare, 7 chars
    t8 = held("abcdefgh")
    t8["originals"][0]["anchor"] = sym("a.py", "abcdefgh")                 # bare, 8 chars
    assert not mk.leak_check("uses abcdefg here", [t7], [], [])[0]
    assert mk.leak_check("uses abcdefg here", [t7], [], [], bare_floor=7)[0]
    assert mk.leak_check("uses abcdefgh here", [t8], [], [])[0]


def test_leak_hex_ids_full_and_first_12_substring():
    ho = [held("a.py::f")]
    iid = ho[0]["itemId"]
    assert hits(f"id {iid} leaked", ho) and hits(f"id {iid[:12]} leaked", ho) and not hits(f"id {iid[:11]} x", ho)
    assert hits(f"aid {ho[0]['anchorId'][:12]}", ho) and hits(f"iid {ho[0]['impactedId'][:12]}", ho)


def test_leak_http_keynorm_normalized_and_pre8a_probes():
    o = {"schema": "ccg-key-v1", "anchor": con("GET /shop/api/products/x7k2q9w4m1p8z3v6", "http"), "impacted": sym("b.py", "b.py::zz_h"), "lines": [1]}
    g = rec(o["anchor"], o["impacted"], "http")
    g["originals"] = [o]
    pre = mk.normalize_http_key(o["anchor"]["keyNorm"], "provider", mask=False)
    assert hits(f"missed {pre.casefold()} here", [g])
    assert not hits("GET /shop/api/products/{}x", [g])
    assert not hits("unrelated", [g])


def heldout_dir_with(tmp, rows_by_file: dict) -> Path:
    d = tmp / "heldout"
    d.mkdir(exist_ok=True)
    for n in [f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl"]:
        (d / n).write_text("".join(json.dumps(r) + "\n" for r in rows_by_file.get(n, [])), encoding="utf-8")
    return d


def test_leak_output_is_by_hash_only(tmp_path):
    ho = held("a.py::f")
    d = heldout_dir_with(tmp_path, {"ac5-flow.jsonl": [ho]})
    (tmp_path / "r.md").write_text("held-out item a.py::f.\n", encoding="utf-8")
    r = cli("--check-report", tmp_path / "r.md", "--heldout-dir", d)
    assert r.returncode == 1 and "a.py" not in r.stdout and "hits: 1" in r.stdout
    h = [l for l in r.stdout.splitlines() if l.startswith("hit: ")][0][5:]
    assert h == hashlib.sha256(b"a.py::f").hexdigest()[:12]
    (tmp_path / "r.md").write_text("clean\n", encoding="utf-8")
    assert cli("check-report", "--report", tmp_path / "r.md", "--heldout-dir", d).returncode == 0
    (tmp_path / "raw.jsonl").write_text(json.dumps({"locator": sym("a.py", "a.py::f"), "source": "S14"}) + "\n", encoding="utf-8")
    assert cli("contact", "--in", tmp_path / "raw.jsonl", "--out", tmp_path / "contact.jsonl").returncode == 0
    (tmp_path / "r.md").write_text("held-out item a.py::f.\n", encoding="utf-8")
    assert cli("check-report", "--report", tmp_path / "r.md", "--heldout-dir", d, "--contact", tmp_path / "contact.jsonl").returncode == 0


def test_check_report_globs_the_directory_so_reserve_and_ac24_are_probed(tmp_path):
    res_item = held("r.py::reserve_only", path="r.py")
    q_item = held("q.py::ac24_only", path="q.py")
    d = heldout_dir_with(tmp_path, {"ac5-reserve.jsonl": [res_item]})
    rp = tmp_path / "r.md"
    rp.write_text("leak r.py::reserve_only here\n", encoding="utf-8")
    assert cli("check-report", "--report", rp, "--heldout-dir", d).returncode == 1
    rp.write_text("leak q.py::ac24_only here\n", encoding="utf-8")
    assert cli("check-report", "--report", rp, "--heldout-dir", d).returncode == 0
    (d / "ac24.jsonl").write_text(json.dumps(q_item) + "\n", encoding="utf-8")
    assert cli("check-report", "--report", rp, "--heldout-dir", d).returncode == 1


def test_check_report_refuses_an_explicit_list_that_omits_a_heldout_file(tmp_path):
    d = heldout_dir_with(tmp_path, {})
    rp = tmp_path / "r.md"
    rp.write_text("clean\n", encoding="utf-8")
    full = [d / f"ac5-{s}.jsonl" for s in STRATA] + [d / "ac5-reserve.jsonl"]
    assert cli("check-report", "--report", rp, "--heldout", *full).returncode == 0
    r = cli("check-report", "--report", rp, "--heldout", *full[:-1])           # reserve omitted
    assert r.returncode == 2 and "ac5-reserve.jsonl" in r.stderr
    assert cli("check-report", "--report", rp, "--heldout", *full[1:]).returncode == 2
    (d / "ac24.jsonl").write_text("", encoding="utf-8")                        # an existing ac24 must be listed too
    assert cli("check-report", "--report", rp, "--heldout", *full).returncode == 2
    assert cli("check-report", "--report", rp, "--heldout", *full, d / "ac24.jsonl").returncode == 0
    assert cli("check-report", "--report", rp).returncode == 2                 # neither a dir nor a list
    assert cli("check-report", "--report", rp, "--heldout-dir", d, "--heldout", *full).returncode == 2


def test_leak_agrees_with_match_keys_leak_check_where_the_floor_is_not_in_play():
    ho = [held("pkg/long_module.py::SomeClass.method")]
    dev = [held("pkg/other.py::Thing")]
    for txt in ("see pkg/long_module.py::someclass.method.", "pkg/long_module.py::SomeClass.method.x", "clean"):
        assert bool(sm.check_report(txt, ho, dev, [])[0]) == bool(mk.leak_check(txt, ho, dev, [])[0])


def test_end_to_end_split_then_planted_heldout_locator_is_caught(tmp_path):
    make_split(tmp_path)
    ho_dir = tmp_path / "E" / "heldout"
    row = json.loads((ho_dir / "ac5-http.jsonl").read_text().splitlines()[0])
    q = row["originals"][0]["anchor"]["qname"]
    rp = tmp_path / "report.md"
    rp.write_text("Counts only: 140 items.\n", encoding="utf-8")
    heldfiles = [str(ho_dir / f"ac5-{s}.jsonl") for s in STRATA] + [str(ho_dir / "ac5-reserve.jsonl")]
    devfiles = [str(tmp_path / "dev" / f"ac5-{s}.jsonl") for s in STRATA]
    assert cli("check-report", "--report", rp, "--heldout", *heldfiles, "--dev", *devfiles).returncode == 0
    rp.write_text(f"Counts only. Also {q}.\n", encoding="utf-8")
    r = cli("--check-report", rp, "--heldout", *heldfiles, "--dev", *devfiles)
    assert r.returncode == 1 and q not in r.stdout


def test_hs_counts_items_by_their_own_derived_stratum_not_the_clusters_primary():
    recs, u, tools = server("srv", {"p": 2})                      # cluster {u, p}: 1 wiring + 2 mcp items, primary mcp
    res = run_split(recs, UNITS, commit_=commit(1))
    assert len(res["clusters"]) == 1 and res["clusters"][0]["primary"] == "mcp"
    assert res["report"]["Hs"] == {"http": 0, "mcp": 2, "scripts": 0, "wiring": 1, "flow": 0}
    assert res["report"]["H_all"] == 3


# --- review fixes (Pass 8) ------------------------------------------------------------------------

def test_m1_contacted_item_less_served_anchor_sends_the_unit_to_dev():
    u = unit("srv")
    t1, t2 = con("t1"), con("t2")
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t1, t2])]    # t1, t2: no items anywhere
    assert min(mk.locator_id(t1), mk.locator_id(t2)) == mk.locator_id(t1)            # t1 is the probe
    base = run_split(recs, UNITS, commit_=commit(1))
    for contact, rule in ((t2, "served-anchor"), (t1, "contact")):
        res = run_split(recs, UNITS, contact=[contact], commit_=commit(1))
        k = res["cluster_of"][mk.locator_id(u)]
        assert res["side"][k] == "dev"
        assert [l["rule"] for l in res["log"] if l["rule"] in ("contact", "served-anchor")] == [rule]
        assert res["report"]["servedAnchorMoves"] == (1 if rule == "served-anchor" else 0)
    # step 4 must not bring it back either
    res = run_split(recs, UNITS, contact=[t2], commit_=commit(1))
    assert res["side"][res["cluster_of"][mk.locator_id(u)]] == "dev" and "wiring" in res["report"]["markedStrata"]
    # without contact the unit is held at share and stays held
    assert base["side"][base["cluster_of"][mk.locator_id(u)]] == "held"


def test_m1_regate_also_honours_a_contacted_item_less_served_anchor():
    u = unit("srv")
    t1, t2 = con("t1"), con("t2")
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t1, t2])]
    res = run_split(recs, UNITS, commit_=commit(1))
    res["C"].add(mk.locator_id(t2))
    assert sm.regate_served(res, []) == 1
    assert res["side"][res["cluster_of"][mk.locator_id(u)]] == "dev"


@pytest.mark.parametrize("bad", [None, "yes", "missing"])
def test_split_refuses_a_record_without_a_boolean_inscope(bad):
    recs = bulk({"http": 3})
    if bad == "missing":
        del recs[0]["inScope"]
    else:
        recs[0]["inScope"] = bad
    with pytest.raises(ValueError):
        run_split(recs)


def test_m3d_out_of_scope_records_appear_nowhere(tmp_path):
    recs = bulk({s: 40 for s in STRATA})
    ghost = single("flow")
    ghost["inScope"] = False
    recs.append(ghost)
    make_split(tmp_path, recs=recs)
    rep = json.loads((tmp_path / "E" / "work" / "split.json").read_text())
    assert rep["clusters"] == 200
    for f in list((tmp_path / "E" / "heldout").iterdir()) + list((tmp_path / "dev").iterdir()):
        assert ghost["itemId"] not in f.read_text()
    res = run_split(recs, commit_=commit(21))
    assert ghost["anchorId"] not in res["cluster_of"]


def dev_after_share(recs, ev):
    """Independent share model for singleton clusters: per stratum, first ceil(0.7 n) by sha256 rank held."""
    rk = lambda g: hashlib.sha256(f"ccg-p0f-split-v1|{ev}|{g['anchorId']}".encode()).hexdigest()
    dev = []
    for st in STRATA:
        grp = sorted((g for g in recs if g["derivedStratum"] == st), key=rk)
        dev += grp[-(-len(grp) * 7 // 10):]
    return sorted(dev, key=rk), rk


def test_m3a_counting_top_up_moves_the_lowest_rank_dev_clusters_in_order():
    recs = bulk({s: 20 for s in STRATA})                       # share 14 held per stratum: H = 70 < 75, dev supply 30
    ev = commit(31)
    dev, rk = dev_after_share(recs, ev)
    assert len(dev) == 30
    res = run_split(recs, commit_=ev)
    moved = [l["cluster"] for l in res["log"] if l["rule"] == "minimum-counting"]
    assert moved == [g["anchorId"] for g in dev[:5]]           # exactly the 5 lowest-rank, in rank order
    assert res["report"]["H"] == 75
    still_dev = {g["anchorId"] for g in dev[5:]}
    assert {a for a in still_dev if res["side"][res["cluster_of"][a]] == "dev"} == still_dev


def test_m3a_stratum_top_up_also_takes_the_lowest_rank_cluster():
    recs = bulk({"http": 20, "mcp": 20, "scripts": 20, "wiring": 20, "flow": 10})     # flow: 7 held, 3 dev
    ev = commit(32)
    dev, rk = dev_after_share(recs, ev)
    flow_dev = [g for g in dev if g["derivedStratum"] == "flow"]
    res = run_split(recs, commit_=ev)
    first = [l for l in res["log"] if l["rule"] == "minimum-stratum"]
    assert len(first) == 1 and first[0]["cluster"] == flow_dev[0]["anchorId"]


def test_m3b_cluster_rank_is_the_minimum_over_its_anchors():
    u = unit("srv")
    p = con("p")
    X = [single("mcp") for _ in range(5)]
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[p]),
            rec(p, sym("srv/p.py", "srv/p.py::h"), "mcp", provider={"path": "srv/p.py"})] + X
    uid, pid = mk.locator_id(u), mk.locator_id(p)
    found = None
    for i in range(1, 3000):
        ev = commit(i)
        r = lambda a: sm.rank_of("ccg-p0f-split-v1|" + ev, a)
        xs = [r(g["anchorId"]) for g in X]
        if min(r(uid), r(pid)) < min(xs) and max(r(uid), r(pid)) > max(xs):
            found = ev
            break
    assert found, "no commit puts the pair's two ranks on both sides of every singleton"
    res = run_split(recs, UNITS, commit_=found)
    k = res["cluster_of"][uid]
    assert res["cluster_of"][pid] == k and res["clusters"][k]["primary"] == "mcp"
    # items: 2 + 5 = 7 -> T = 5.  By the MIN rank the pair is walked first (held, 2) then x1..x3 (5): pair held at share.
    # By the max rank it would be walked last, after x1..x5 reach 5, and land in dev.
    pulled_in = {l["cluster"] for l in res["log"] if l["rule"].startswith("minimum")}
    assert res["clusters"][k]["key"] not in pulled_in
    assert res["clusters"][k]["rank"] == min(sm.rank_of("ccg-p0f-split-v1|" + found, a) for a in (uid, pid))


def test_m3c_reserve_skips_clusters_without_a_counting_item():
    recs = bulk({s: 40 for s in ("http", "mcp", "scripts", "wiring")}) + bulk({"flow": 40}, counting=False)
    ok = None
    for i in range(1, 300):
        res = run_split(recs, commit_=commit(i))
        held_desc = sorted((c for c in res["clusters"] if res["side"][c["id"]] != "dev"), key=lambda c: c["rank"], reverse=True)
        if res["report"]["trigger"] is False and any(not c["items"][0]["counting"] for c in held_desc[:45]):
            ok = res
            break
    assert ok is not None
    reserve = [c for c in ok["clusters"] if ok["side"][c["id"]] == "reserve"]
    assert reserve and all(any(g["counting"] for g in c["items"]) for c in reserve)
    assert ok["report"]["reserveCounting"] == 35 and ok["report"]["reserveClusters"] == 35


def test_contact_verb_normalizes_dedups_and_sorts(tmp_path):
    raw = [
        {"locator": {"type": "contract", "kind": "mcp_tool", "keyNorm": "mcp__srv__tool"}, "source": "S30"},
        {"locator": {"type": "contract", "kind": "mcp_tool", "keyNorm": "tool"}, "source": "S30"},
        {"locator": {"type": "contract", "kind": "http", "keyNorm": "get /api/items/{item_id}"}, "source": "S14"},
        {"locator": {"type": "symbol", "path": "./a.py", "qname": "a.py::C.x#2"}, "source": "S28"},
        {"locator": {"type": "unit", "rootPath": "srv", "lang": "python"}, "source": "S30"},
    ]
    src = tmp_path / "raw.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    out = tmp_path / "work" / "contact.jsonl"
    r = cli("contact", "--in", src, "--out", out)
    assert r.returncode == 0 and "contact-anchors: 4" in r.stdout and "lines: 4" in r.stdout and "tool" not in r.stdout
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [x["anchorId"] for x in rows] == sorted(x["anchorId"] for x in rows) and len(rows) == 4
    by_src = {x["locator"].get("kind") or x["locator"]["type"]: x for x in rows}
    assert by_src["mcp_tool"]["locator"]["keyNorm"] == "tool"
    assert by_src["mcp_tool"]["anchorId"] == mk.locator_id(con("tool"))
    assert by_src["http"]["locator"]["keyNorm"] == "GET /api/items/{}"
    sy = [x for x in rows if x["locator"]["type"] == "symbol"][0]
    assert sy["locator"]["qname"] == "a.py::C.x" and sy["locator"]["path"] == "a.py"
    assert sy["anchorId"] == mk.locator_id(sym("a.py", "a.py::C.x"))
    assert out.read_text().endswith("\n") and ", " not in out.read_text().splitlines()[0]          # canonical JSON
    out2 = tmp_path / "work" / "contact2.jsonl"
    cli("contact", "--in", src, "--out", out2)
    assert out2.read_bytes() == out.read_bytes()
    # the output is a fixed point of the verb's input shape
    out3 = tmp_path / "work" / "contact3.jsonl"
    cli("contact", "--in", out, "--out", out3)
    assert out3.read_bytes() == out.read_bytes()


def test_contact_http_form_changes_the_masked_key(tmp_path):
    raw = [{"locator": {"type": "contract", "kind": "http", "keyNorm": "POST /hooks/x7k2q9w4m1p8z3v6", "form": "consumer"}, "source": "S14"},
           {"locator": {"type": "contract", "kind": "http", "keyNorm": "POST /hooks/x7k2q9w4m1p8z3v6"}, "source": "S14"}]
    rows = sm.build_contact(raw)
    keys = {r["locator"]["keyNorm"] for r in rows}
    assert keys == {mk.normalize_http_key("POST /hooks/x7k2q9w4m1p8z3v6", "consumer"), mk.normalize_http_key("POST /hooks/x7k2q9w4m1p8z3v6", "provider")}
    assert len(keys) == 2 and not any("form" in r["locator"] for r in rows)


def test_split_manifest_and_check_report_refuse_a_raw_contact_file(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"locator": sym("a.py", "a.py::f"), "source": "S14"}) + "\n", encoding="utf-8")
    write_work(tmp_path, bulk({"http": 3}))
    d = heldout_dir_with(tmp_path, {})
    rp = tmp_path / "r.md"
    rp.write_text("x\n", encoding="utf-8")
    assert cli("split", "--match", tmp_path / "match.json", "--context", tmp_path / "context.json", "--contact", raw,
               "--eval-commit", commit(1), "--heldout-dir", d, "--dev-dir", tmp_path / "dev").returncode == 4
    assert cli("manifest", "--heldout-dir", d, "--eval-commit", commit(1), "--protocol", "P", "--contact", raw,
               "--out", tmp_path / "m.json").returncode == 4
    assert cli("check-report", "--report", rp, "--heldout-dir", d, "--contact", raw).returncode == 4


def test_verify_fails_when_files_omits_a_required_name(tmp_path):
    make_split(tmp_path)
    ho = tmp_path / "E" / "heldout"
    man = sm.build_manifest(ho, commit(21), "P", 0)
    assert sm.verify_manifest(man, ho) == []
    assert sm.verify_manifest(dict(man, files=[]), ho) and "ac5-http.jsonl: not listed" in sm.verify_manifest(dict(man, files=[]), ho)
    for i in range(len(man["files"])):
        cut = dict(man, files=man["files"][:i] + man["files"][i + 1:])
        assert sm.verify_manifest(cut, ho) == [f"{man['files'][i]['name']}: not listed"]
    (ho / "ac24.jsonl").write_text("", encoding="utf-8")                      # exists in heldout/ but is not listed
    assert sm.verify_manifest(man, ho) == ["ac24.jsonl: not listed"]
    full = sm.build_manifest(ho, commit(21), "P", 0)
    assert [f["name"] for f in full["files"]][-1] == "ac24.jsonl" and sm.verify_manifest(full, ho) == []


def test_check_report_heldout_dir_must_be_complete(tmp_path):
    rp = tmp_path / "r.md"
    rp.write_text("held-out item a.py::f.\n", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    r = cli("check-report", "--report", rp, "--heldout-dir", empty)
    assert r.returncode == 2 and "holds no ac5-*.jsonl" in r.stderr
    r = cli("check-report", "--report", rp, "--heldout-dir", tmp_path / "nowhere")
    assert r.returncode == 2 and "holds no ac5-*.jsonl" in r.stderr
    d = heldout_dir_with(tmp_path, {"ac5-flow.jsonl": [held("a.py::f")]})
    (d / "ac5-reserve.jsonl").unlink()
    r = cli("check-report", "--report", rp, "--heldout-dir", d)
    assert r.returncode == 2 and "ac5-reserve.jsonl" in r.stderr
    (d / "ac5-reserve.jsonl").write_text("", encoding="utf-8")
    assert cli("check-report", "--report", rp, "--heldout-dir", d).returncode == 1       # complete dir, planted qname
    (d / "ac5-mcp.jsonl").unlink()
    assert cli("check-report", "--report", rp, "--heldout-dir", d).returncode == 2


def test_check_report_refuses_heldout_and_heldout_dir_together(tmp_path):
    d = heldout_dir_with(tmp_path, {})
    rp = tmp_path / "r.md"
    rp.write_text("clean\n", encoding="utf-8")
    full = [d / f"ac5-{s}.jsonl" for s in STRATA] + [d / "ac5-reserve.jsonl"]
    assert cli("check-report", "--report", rp, "--heldout-dir", d).returncode == 0
    r = cli("check-report", "--report", rp, "--heldout-dir", d, "--heldout", *full)
    assert r.returncode == 2 and "exactly one" in r.stderr


def test_contact_normalizer_applies_nfc_before_the_id(tmp_path):
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    assert composed != decomposed
    a = sm.normalize_contact_locator(sym(f"{composed}.py", f"{composed}.py::f"))
    b = sm.normalize_contact_locator(sym(f"{decomposed}.py", f"{decomposed}.py::f"))
    assert a == b and a["path"] == f"{composed}.py"
    rows = sm.build_contact([{"locator": sym(f"{composed}.py", f"{composed}.py::f"), "source": "S"},
                             {"locator": sym(f"{decomposed}.py", f"{decomposed}.py::f"), "source": "S"}])
    assert len(rows) == 1
    h = sm.normalize_contact_locator({"type": "contract", "kind": "mcp_tool", "keyNorm": f"{decomposed}"})
    assert h["keyNorm"] == composed


def test_contact_refuses_config_locators(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"locator": cfg(".mcp.json", "mcpServers.cypher"), "source": "S30"}) + "\n", encoding="utf-8")
    r = cli("contact", "--in", raw, "--out", tmp_path / "c.jsonl")
    assert r.returncode == 4 and "not an anchor type" in r.stderr and not (tmp_path / "c.jsonl").exists()
    with pytest.raises(ValueError):
        sm.normalize_contact_locator({"type": "weird"})
