"""P1-32a: the evaluation verbs (docs/plans/connected-code-graph-p1-gates.md P1-32a).

Synthetic, planted inputs only. The S47 stored work inputs live under the custodian root and are never read here:
the "byte for byte" tests run the stored replay record (``s47_scripted_steps.py``) and the new verbs on the same
synthetic inputs and compare output bytes.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parent.parent
TOOLS = Path(os.environ.get("CCG_TOOLS_DIR", CPG / "eval" / "tools"))
for _p in (str(TOOLS), str(CPG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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


def hist_match(hists_by_stratum, t):
    recs = []
    for s, h in hists_by_stratum.items():
        for bits, n in h.items():
            recs += [{"derivedStratum": s, "history": bits, "inScope": True, "members": []} for _ in range(n)]
    return {"schema": "ccg-match-v1", "records": recs, "t": dict(t)}


# =============================================================================================
# capture: the trigger flag is cleared after a KA2 run (P0 nit)
# =============================================================================================

def test_capture_trigger_is_cleared_once_ka2_has_run_but_below_threshold_is_kept():
    h = {"101": 28, "100": 12, "001": 7}                       # completeness 0.941: below 0.95
    t2 = cap.estimate(h, 2)
    t3 = cap.estimate(h, 3)
    assert t2["trigger"] is True                                # KA2 still owed
    assert t3["trigger"] is False                               # EP 6.5: at most one extra pass; it ran
    assert t3["belowThreshold"] is True and t2["belowThreshold"] is True   # the report caveat survives


def test_capture_trigger_stays_false_for_t3_even_when_empty_and_true_for_empty_t2():
    assert cap.estimate({}, 3)["trigger"] is False
    assert cap.estimate({}, 2)["trigger"] is True


def test_capture_build_uses_each_strata_own_t_for_the_trigger():
    t = dict.fromkeys(STRATA, 2)
    t["flow"] = 3
    m = hist_match({"http": {"101": 28, "100": 12, "001": 7}, "flow": {"111": 5, "110": 20, "100": 12, "001": 7}}, t)
    out = cap.build(m)
    assert out["strata"]["http"]["trigger"] is True
    assert out["strata"]["flow"]["trigger"] is False
    assert out["strata"]["flow"]["belowThreshold"] is True


# =============================================================================================
# scripted S47 steps: apply-corrections, ambiguity, split-ac24 vs the stored replay record
# =============================================================================================
REPLAY = TOOLS / "s47_scripted_steps.py"
EV = "cf7b2042cce7bad5d0ab7876940bcea00343d914"


def jl_write(path, rows, canonical=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((mk.canon(r) if canonical else json.dumps(r)) + "\n" for r in rows), encoding="utf-8")


def sm_cli(*args, env=None):
    return subprocess.run([sys.executable, str(TOOLS / "split_manifest.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=CPG, env=env)


def contract(kind, key):
    return {"type": "contract", "kind": kind, "keyNorm": key}


def make_work(tmp):
    """A synthetic S47 work dir: 4 (group, lang) cells with 9 unambiguous questions each, 4+4 ambiguous ones, and the
    drop cases; plus a scoped match file with union items some of which the verifier marked incorrect."""
    work = tmp / "work"
    qw, an1, an2, amap = [], [], [], []
    seq = 0
    for group in ("contract-point", "wiring"):
        for lang in ("en", "pt"):
            for i in range(9):
                seq += 1
                key = (f"GET /items{seq}/{{item_id}}" if seq % 2 else f"GET /items/abc{seq}def456ghi789jkl") if group == "contract-point" else f"tool_{seq}"
                kind = "http" if group == "contract-point" else "mcp_tool"
                g = {"anchors": [contract(kind, key)], "direction": "impact", "scope": None, "useCase": "impact"}
                qw.append({"qid": f"QW-{seq}", "lang": lang, "group": group, "text": f"q {seq} ça va?", "intent": "unambiguous",
                           "gold": g, "goldOptions": None, "ambiguityKind": None})
    for lang, n in (("en", 5), ("pt", 3)):
        for i in range(n):
            seq += 1
            opts = [{"anchors": [contract("http", f"POST /a{seq}/{{x}}")], "direction": "impact", "scope": None, "useCase": "impact"},
                    {"anchors": [contract("mcp_tool", f"tool_{seq}")], "direction": "impact", "scope": None, "useCase": "impact"}]
            qw.append({"qid": f"QW-{seq}", "lang": lang, "group": "contract-point", "text": f"ambiguous {seq}", "intent": "ambiguous",
                       "gold": None, "goldOptions": opts, "ambiguityKind": "two-readings"})
    # drop cases: annotators disagree; both say ambiguous on an unambiguous intent; both unambiguous on ambiguous intent
    for kind in ("disagree", "disagree2", "wunamb-aamb", "wamb-aunamb"):
        seq += 1
        intent = "ambiguous" if kind == "wamb-aunamb" else "unambiguous"
        g = {"anchors": [contract("mcp_tool", f"drop_{seq}")], "direction": "impact", "scope": None, "useCase": "impact"}
        qw.append({"qid": f"QW-{seq}", "lang": "en", "group": "wiring", "text": f"drop {kind}", "intent": intent,
                   "gold": None if intent == "ambiguous" else g, "goldOptions": [g, g] if intent == "ambiguous" else None,
                   "ambiguityKind": "x" if intent == "ambiguous" else None})
    labels = {"disagree": ("ambiguous", "unambiguous"), "disagree2": ("unambiguous", "ambiguous"), "wunamb-aamb": ("ambiguous", "ambiguous"),
              "wamb-aunamb": ("unambiguous", "unambiguous")}
    drops = {q["text"].split()[1]: q["qid"] for q in qw if q["text"].startswith("drop ")}
    for i, q in enumerate(qw):
        aid = f"A{i:03d}"
        amap.append({"aid": aid, "qid": q["qid"]})
        if q["text"].startswith("drop "):
            l1, l2 = labels[q["text"].split()[1]]
        else:
            l1 = l2 = q["intent"]
        an1.append({"aid": aid, "label": l1})
        an2.append({"aid": aid, "label": l2})
    jl_write(work / "questions" / "QW.jsonl", qw)
    jl_write(work / "questions" / "AN1.jsonl", an1)
    jl_write(work / "questions" / "AN2.jsonl", an2)
    jl_write(work / "aid-map.jsonl", amap)
    tmpl, ver, recs = [], [], []
    for n in range(1, 9):
        iid = f"item{n}"
        recs.append({"itemId": iid, "x": n})
        tmpl.append({"n": n, "itemId": iid, "kind": "union" if n % 2 else "single"})
        ver.append({"n": n, "verdict": "incorrect" if n in (1, 2, 3) else ("unclear" if n == 5 else "correct")})
    jl_write(work / "s46-marks-template.jsonl", tmpl)
    jl_write(work / "s46-verify.jsonl", ver)
    (work / "s45").mkdir(parents=True, exist_ok=True)
    (work / "s45" / "match-scoped.json").write_text(
        json.dumps({"schema": "ccg-match-v1", "records": recs, "t": {"http": 2}}, ensure_ascii=False) + "\n", encoding="utf-8")
    return work


def run_replay(work, ho, dev):
    r = subprocess.run([sys.executable, str(REPLAY), "--work", str(work), "--eval-commit", EV, "--heldout-dir", str(ho),
                        "--dev-dir", str(dev)], capture_output=True, text=True, cwd=CPG)
    assert r.returncode == 0, r.stderr
    return r.stdout


def run_verbs(work, ho, dev, ev=EV):
    out = ""
    r = sm_cli("apply-corrections", "--match-scoped", work / "s45" / "match-scoped.json", "--marks-template",
               work / "s46-marks-template.jsonl", "--verify", work / "s46-verify.jsonl",
               "--out-match", work / "s45" / "match-corrected.json", "--corrections", work / "corrections.jsonl")
    assert r.returncode == 0, r.stderr
    out += r.stdout
    r = sm_cli("ambiguity", "--questions", work / "questions", "--aid-map", work / "aid-map.jsonl",
               "--out", work / "ambiguity.jsonl", "--log", work / "s45" / "ac24-log.json")
    assert r.returncode == 0, r.stderr
    out += r.stdout
    r = sm_cli("split-ac24", "--in", work / "ambiguity.jsonl", "--eval-commit", ev, "--heldout-dir", ho, "--dev-dir", dev)
    assert r.returncode == 0, r.stderr
    out += r.stdout
    return out


@pytest.fixture()
def replayed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    wa, wb = make_work(a), make_work(b)
    so = run_replay(wa, a / "ho", a / "dev")
    sv = run_verbs(wb, b / "ho", b / "dev")
    return a, b, so, sv


def test_scripted_steps_ac24_files_are_byte_identical_to_the_replay_record(replayed):
    a, b, _, _ = replayed
    for side in ("ho", "dev"):
        x, y = (a / side / "ac24.jsonl").read_bytes(), (b / side / "ac24.jsonl").read_bytes()
        assert x and x == y, side


def test_scripted_steps_counts_are_30_held_out_and_14_dev_as_s47_produced(replayed):
    a, b, so, sv = replayed
    assert len((b / "ho" / "ac24.jsonl").read_text().splitlines()) == 30
    assert len((b / "dev" / "ac24.jsonl").read_text().splitlines()) == 14
    assert "held-out: 30" in sv and "dev: 14" in sv


def test_scripted_steps_corrections_match_and_log_are_byte_identical(replayed):
    a, b, _, _ = replayed
    assert (a / "work" / "s45" / "match-corrected.json").read_bytes() == (b / "work" / "s45" / "match-corrected.json").read_bytes()
    assert (a / "work" / "corrections.jsonl").read_bytes() == (b / "work" / "corrections.jsonl").read_bytes()
    assert (a / "work" / "s45" / "ac24-log.json").read_bytes() == (b / "work" / "s45" / "ac24-log.json").read_bytes()
    removed = [json.loads(l)["itemId"] for l in (b / "work" / "corrections.jsonl").read_text().splitlines()]
    assert removed == ["item1", "item3"]                    # incorrect AND union only; item2 is incorrect but 'single'


def test_scripted_steps_stdout_is_the_replay_stdout(replayed):
    _, _, so, sv = replayed
    assert sv == so


def test_scripted_steps_ambiguity_drop_rules_and_gold_rewrites(replayed):
    _, b, _, _ = replayed
    log = json.loads((b / "work" / "s45" / "ac24-log.json").read_text())
    assert log["outcomes"] == {"unambiguous": 36, "ambiguity": 8, "drop-disagree": 2,
                               "drop-writer-unamb-ann-amb": 1, "drop-writer-amb-ann-unamb": 1}
    assert sum(log["rewrites"].values()) > 0                 # {item_id} -> {} rewrites were logged, not silent
    held = [json.loads(l) for l in (b / "ho" / "ac24.jsonl").read_text().splitlines()]
    keys = {a["keyNorm"] for r in held if r["group"] == "contract-point" and r["outcome"] == "unambiguous" for a in r["gold"]["anchors"]}
    assert any(k.endswith("/{#}") for k in keys) and not any("{?}" in k for k in keys)    # provider form, never the consumer one


def test_scripted_steps_split_rule_by_rank(replayed):
    _, b, _, _ = replayed
    held = [json.loads(l) for l in (b / "ho" / "ac24.jsonl").read_text().splitlines()]
    dev = [json.loads(l) for l in (b / "dev" / "ac24.jsonl").read_text().splitlines()]
    assert [r["qid"] for r in held] == sorted(r["qid"] for r in held)
    for side in (held, dev):
        assert all(r["rank"] == mk.sha("ccg-p0f-ac24-v1|" + EV + "|" + r["qid"]) for r in side)
    cells = {}
    for r in held + dev:
        if r["outcome"] == "unambiguous":
            cells.setdefault((r["group"], r["lang"]), []).append((r["rank"], r["qid"] in {h["qid"] for h in held}))
    for v in cells.values():
        v.sort()
        assert [h for _, h in v] == [True] * 6 + [False] * 3       # the 6 lowest ranks held out, the rest dev
    amb = {}
    for r in held + dev:
        if r["outcome"] == "ambiguity":
            amb.setdefault(r["lang"], []).append((r["rank"], r["qid"] in {h["qid"] for h in held}))
    for v in amb.values():
        v.sort()
        assert [h for _, h in v] == [False] + [True] * (len(v) - 1)   # lowest rank to dev, the others held out


def test_split_ac24_seed_includes_the_eval_commit(tmp_path):
    w = make_work(tmp_path / "a")
    run_verbs(w, tmp_path / "a" / "ho", tmp_path / "a" / "dev")
    sm_cli("split-ac24", "--in", w / "ambiguity.jsonl", "--eval-commit", "0" * 40, "--heldout-dir", tmp_path / "o" / "ho",
           "--dev-dir", tmp_path / "o" / "dev")
    assert (tmp_path / "o" / "ho" / "ac24.jsonl").read_bytes() != (tmp_path / "a" / "ho" / "ac24.jsonl").read_bytes()


def test_scripted_verbs_refuse_missing_input_with_exit_4_and_print_no_item_content(tmp_path):
    r = sm_cli("apply-corrections", "--match-scoped", tmp_path / "none", "--marks-template", tmp_path / "n2",
               "--verify", tmp_path / "n3", "--out-match", tmp_path / "o", "--corrections", tmp_path / "c")
    assert r.returncode == 4
    w = make_work(tmp_path / "w")
    (w / "questions" / "AN2.jsonl").write_text("", encoding="utf-8")            # every question loses its AN2 label
    r = sm_cli("ambiguity", "--questions", w / "questions", "--aid-map", w / "aid-map.jsonl", "--out", tmp_path / "o2", "--log", tmp_path / "l")
    assert r.returncode == 4


# =============================================================================================
# exposure (EP 9.2 exposure.jsonl, 12.1 step 6)
# =============================================================================================

def exposure(tmp, gate, results, name="exposure.jsonl"):
    rp = tmp / f"results-{gate}.jsonl"
    jl_write(rp, results)
    return sm_cli("exposure", "--exposure", tmp / name, "--gate", gate, "--results", rp)


def read_exposure(tmp, name="exposure.jsonl"):
    return [json.loads(l) for l in (tmp / name).read_text().splitlines()]


def test_exposure_first_gate_appends_one_row_per_item_canonical_and_sorted(tmp_path):
    (tmp_path / "exposure.jsonl").write_text("", encoding="utf-8")                 # empty at S47
    r = exposure(tmp_path, 1, [{"itemId": "b", "hit": True}, {"itemId": "a", "hit": False}])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "exposure.jsonl").read_text() == (
        '{"firstGate":1,"hit":false,"itemId":"a"}\n{"firstGate":1,"hit":true,"itemId":"b"}\n')
    assert "appended: 2" in r.stdout and "hits: 1" in r.stdout and "itemId" not in r.stdout


def test_exposure_an_item_is_counted_once_at_its_first_gate(tmp_path):
    exposure(tmp_path, 1, [{"itemId": "a", "hit": True}, {"itemId": "b", "hit": False}])
    r = exposure(tmp_path, 2, [{"itemId": "a", "hit": False}, {"itemId": "b", "hit": True}, {"itemId": "c", "hit": True}])
    assert r.returncode == 0, r.stderr
    rows = {x["itemId"]: x for x in read_exposure(tmp_path)}
    assert rows["a"] == {"itemId": "a", "firstGate": 1, "hit": True}          # not overwritten by gate 2's miss
    assert rows["b"] == {"itemId": "b", "firstGate": 1, "hit": False}         # nor a miss turned into a hit
    assert rows["c"] == {"itemId": "c", "firstGate": 2, "hit": True}
    assert "appended: 1" in r.stdout and "already: 2" in r.stdout and "hits: 1" in r.stdout     # only c is a newly appended hit


def test_exposure_creates_the_file_and_is_idempotent_for_the_same_gate(tmp_path):
    exposure(tmp_path, 1, [{"itemId": "a", "hit": True}])
    first = (tmp_path / "exposure.jsonl").read_bytes()
    r = exposure(tmp_path, 1, [{"itemId": "a", "hit": True}])
    assert r.returncode == 0 and (tmp_path / "exposure.jsonl").read_bytes() == first and "appended: 0" in r.stdout


@pytest.mark.parametrize("results", [
    [{"itemId": "a", "hit": "yes"}],                                   # hit must be a boolean
    [{"itemId": "a", "hit": 1}],
    [{"itemId": "a"}],
    [{"hit": True}],
    [{"itemId": "", "hit": True}],
    [{"itemId": "a", "hit": True}, {"itemId": "a", "hit": False}],     # one result per item per gate
])
def test_exposure_refuses_malformed_results_and_leaves_the_file_unchanged(tmp_path, results):
    exposure(tmp_path, 1, [{"itemId": "z", "hit": True}])
    before = (tmp_path / "exposure.jsonl").read_bytes()
    r = exposure(tmp_path, 2, results)
    assert r.returncode == 4, (r.stdout, r.stderr)
    assert (tmp_path / "exposure.jsonl").read_bytes() == before


def test_exposure_refuses_a_gate_before_one_already_recorded_and_a_non_positive_gate(tmp_path):
    exposure(tmp_path, 3, [{"itemId": "a", "hit": True}])
    assert exposure(tmp_path, 2, [{"itemId": "n", "hit": True}]).returncode == 4
    assert exposure(tmp_path, 0, [{"itemId": "n", "hit": True}], name="fresh.jsonl").returncode == 4     # no record yet: only the >= 1 rule bites
    assert not (tmp_path / "fresh.jsonl").exists()
    assert "n" not in {x["itemId"] for x in read_exposure(tmp_path)}


def test_exposure_refuses_a_corrupt_existing_record(tmp_path):
    (tmp_path / "exposure.jsonl").write_text('{"itemId":"a","firstGate":1}\n', encoding="utf-8")   # no hit
    assert exposure(tmp_path, 2, [{"itemId": "n", "hit": True}]).returncode == 4
    (tmp_path / "exposure.jsonl").write_text('{"itemId":"a","firstGate":1,"hit":true}\n{"itemId":"a","firstGate":2,"hit":false}\n', encoding="utf-8")
    assert exposure(tmp_path, 3, [{"itemId": "n", "hit": True}]).returncode == 4                    # duplicate item


# =============================================================================================
# regate-list (EP 12.1 steps 3-5)
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


_n = [1000]


def single(stratum, counting=True):
    _n[0] += 1
    return rec(sym(f"f{_n[0]}.py", f"f{_n[0]}.py::f"), sym(f"g{_n[0]}.py", f"g{_n[0]}.py::g"), stratum, counting)


def bulk(n_by_stratum):
    return [single(s) for s, n in n_by_stratum.items() for _ in range(n)]


def commit(i):
    return f"{i:040x}"


@pytest.fixture(scope="module")
def gate1():
    """200 single-item clusters (40 per stratum); gate 1's split holds out, keeps in reserve and sends to dev."""
    recs = bulk({s: 40 for s in STRATA})
    res = sm.split(recs, [], set(), commit(21))
    by = {"held": [], "dev": [], "reserve": []}
    for c in res["clusters"]:
        by[res["side"][c["id"]]] += c["items"]
    return recs, res, by


def go(gate1_, missed=(), fresh=(), contact=(), units=(), promote=False, recs=None):
    r0, _, _ = gate1_
    return sm.regate_list(recs or r0, list(units), set(contact), commit(21), list(missed), list(fresh), promote)


def test_regate_gate1_fixture_has_held_dev_and_reserve_items(gate1):
    _, _, by = gate1
    assert len(by["held"]) > 0 and len(by["dev"]) > 0 and len(by["reserve"]) == 35


def test_regate_miss_cluster_moves_to_dev_and_the_other_held_items_are_kept(gate1):
    _, _, by = gate1
    miss = by["held"][0]
    out = go(gate1, missed=[miss["anchorId"]])
    kept_ids = {g["itemId"] for g in out["held"]}
    assert miss["itemId"] not in kept_ids and miss["itemId"] in {g["itemId"] for g in out["dev"]}
    assert kept_ids == {g["itemId"] for g in by["held"]} - {miss["itemId"]}
    assert set(out["flags"].values()) == {"kept"} and out["report"]["kept"] == len(by["held"]) - 1
    assert out["report"]["missedClusters"] == 1 and out["report"]["movedOutByStratum"][miss["derivedStratum"]] == 1
    assert {g["itemId"] for g in out["reserve"]} == {g["itemId"] for g in by["reserve"]}      # untouched


def test_regate_discard_rule_item_already_in_dev_held_or_reserve_is_dropped_not_routed(gate1):
    _, _, by = gate1
    fresh = [dict(single("http"), itemId=g["itemId"]) for g in (by["dev"][0], by["held"][0], by["reserve"][0])]   # same itemId, new anchor
    for f in fresh:
        f["anchorId"] = "e" * 32                                               # would be a fresh-cluster item if it were not discarded
    out = go(gate1, fresh=fresh)
    rep = out["report"]
    assert rep["discarded"] == 3 and rep["freshCluster"] == 0 and rep["freshToDev"] == 0 and rep["keptClusterFresh"] == 0
    assert len(out["held"]) == len(by["held"])


def test_regate_discard_rule_applies_to_a_repeat_inside_the_fresh_list(gate1):
    f = single("http")
    out = go(gate1, fresh=[f, dict(f)])
    assert out["report"]["discarded"] == 1 and out["report"]["freshCluster"] == 1


def test_regate_routing_by_anchor_dev_contact_kept_and_fresh(gate1):
    recs, res, by = gate1
    dev_a, held_a = by["dev"][0]["anchorId"], by["held"][1]["anchorId"]
    ghost = "d" * 32                                                            # in C, no item in the union
    f_dev, f_c, f_kept, f_new = (single("http") for _ in range(4))
    f_dev["anchorId"], f_c["anchorId"], f_kept["anchorId"] = dev_a, ghost, held_a
    for i, f in enumerate((f_dev, f_c, f_kept)):
        f["itemId"] = f"fresh-{i}"
    out = sm.regate_list(recs, [], {ghost}, commit(21), [], [f_dev, f_c, f_kept, f_new])
    flags = out["flags"]
    assert f_dev["itemId"] not in flags and f_c["itemId"] not in flags                  # both went to dev
    assert f_dev["itemId"] in {g["itemId"] for g in out["dev"]} and f_c["itemId"] in {g["itemId"] for g in out["dev"]}
    assert flags[f_kept["itemId"]] == "kept-cluster-fresh" and flags[f_new["itemId"]] == "fresh-cluster"
    assert out["report"]["freshToDev"] == 2 and out["report"]["keptClusterFresh"] == 1 and out["report"]["freshCluster"] == 1


def test_regate_a_kept_cluster_is_joined_scored_and_not_counted_as_fresh(gate1):
    _, _, by = gate1
    f = single("http")
    f["anchorId"] = by["held"][2]["anchorId"]
    base = go(gate1)["report"]
    out = go(gate1, fresh=[f])
    assert f["itemId"] in {g["itemId"] for g in out["held"]}                               # joined: it is on the list
    assert out["report"]["H"] == base["H"] + 1                                             # and scored
    assert out["report"]["freshCounting"] == base["freshCounting"] == 0                    # but not fresh evidence
    assert out["flags"][f["itemId"]] == "kept-cluster-fresh"


def test_regate_fresh_item_on_a_missed_cluster_goes_to_dev(gate1):
    _, _, by = gate1
    miss = by["held"][3]
    f = single("http")
    f["anchorId"] = miss["anchorId"]
    out = go(gate1, missed=[miss["anchorId"]], fresh=[f])
    assert f["itemId"] in {g["itemId"] for g in out["dev"]} and f["itemId"] not in out["flags"]


SRV = [{"rootPath": "srv", "lang": "python"}]


def srv_system():
    t1, t2, u = con("t1"), con("t2"), unit("srv")
    recs = [rec(t1, sym("srv/a.py", "srv/a.py::h"), "mcp", provider={"path": "srv/a.py"}),
            rec(t2, sym("srv/b.py", "srv/b.py::h"), "mcp", provider={"path": "srv/b.py"})] + bulk({s: 40 for s in STRATA})
    return recs, t1, t2, u


def pick_commit(recs, ok):
    """The first deterministic commit (1..60) whose gate-1 split satisfies ``ok(res)``."""
    for i in range(1, 60):
        res = sm.split(recs, SRV, set(), commit(i))
        if ok(res):
            return i
    raise AssertionError("no commit gives the layout")


def test_regate_unit_item_goes_to_dev_when_its_served_set_has_an_anchor_in_c():
    recs, t1, t2, u = srv_system()
    fresh_u = rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t1, t2])          # a unit item with no gate-1 cluster
    ev = pick_commit(recs, lambda r: all(r["side"][r["cluster_of"][mk.locator_id(t)]] == "held" for t in (t1, t2)))
    out = sm.regate_list(recs, SRV, {mk.locator_id(t1)}, commit(ev), [], [fresh_u])         # t1 in C (its cluster is dev by contact)
    assert fresh_u["itemId"] in {g["itemId"] for g in out["dev"]} and out["report"]["freshToDev"] == 1
    out = sm.regate_list(recs, SRV, set(), commit(ev), [], [fresh_u])                       # negative control: both served tools kept
    assert out["flags"][fresh_u["itemId"]] == "fresh-cluster" and out["report"]["freshToDev"] == 0


def test_regate_unit_item_goes_to_dev_when_its_served_set_has_an_anchor_moved_to_dev_by_a_miss():
    recs, t1, t2, u = srv_system()
    fresh_u = rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t1, t2])
    ev = pick_commit(recs, lambda r: all(r["side"][r["cluster_of"][mk.locator_id(t)]] == "held" for t in (t1, t2)))
    out = sm.regate_list(recs, SRV, set(), commit(ev), [mk.locator_id(t1)], [fresh_u])
    assert fresh_u["itemId"] in {g["itemId"] for g in out["dev"]}


def test_regate_reserve_is_promoted_as_fresh_clusters_only_on_request(gate1):
    _, _, by = gate1
    off = go(gate1)
    on = go(gate1, promote=True)
    assert off["report"]["freshCluster"] == 0 and off["report"]["reserveLeft"] == 35
    assert on["report"]["freshCluster"] == 35 and on["report"]["reserveLeft"] == 0 and on["report"]["freshCounting"] == 35
    assert on["report"]["floorMet"] is True and off["report"]["floorMet"] is False
    assert {g["itemId"] for g in on["held"] if on["flags"][g["itemId"]] == "fresh-cluster"} == {g["itemId"] for g in by["reserve"]}


def test_regate_floor_is_35_counting_fresh_cluster_items(gate1):
    for n, met in ((34, False), (35, True), (36, True)):
        out = go(gate1, fresh=[single("mcp") for _ in range(n)])
        assert out["report"]["freshCounting"] == n and out["report"]["floorMet"] is met, n
    # non-counting fresh items are on the list but never reach the floor
    out = go(gate1, fresh=[single("mcp") for _ in range(30)] + [single("mcp", counting=False) for _ in range(9)])
    assert out["report"]["freshCluster"] == 39 and out["report"]["freshCounting"] == 30 and out["report"]["floorMet"] is False
    assert out["report"]["floor"] == sm.FRESH_FLOOR == 35


def test_regate_composition_shortfall_per_stratum_that_had_a_miss(gate1):
    _, _, by = gate1
    miss = next(g for g in by["held"] if g["derivedStratum"] == "http")
    out = go(gate1, missed=[miss["anchorId"]], fresh=[single("mcp") for _ in range(2)])
    assert out["report"]["shortfallByStratum"] == {"http": 1}                         # 1 moved out, 0 fresh http
    out = go(gate1, missed=[miss["anchorId"]], fresh=[single("http")])
    assert out["report"]["shortfallByStratum"] == {"http": 0}


def test_regate_refuses_a_missed_anchor_that_is_not_on_the_held_out_list(gate1):
    _, _, by = gate1
    for bad in (by["dev"][0]["anchorId"], by["reserve"][0]["anchorId"], "9" * 32):
        with pytest.raises(ValueError):
            go(gate1, missed=[bad])


def test_regate_refuses_fresh_records_without_scope_and_counting(gate1):
    f = single("http")
    del f["inScope"]
    with pytest.raises(ValueError):
        go(gate1, fresh=[f])
    f = single("http")
    del f["counting"]
    with pytest.raises(ValueError):
        go(gate1, fresh=[f])


def test_regate_out_of_scope_fresh_items_are_counted_and_never_listed(gate1):
    f = single("http")
    f["inScope"] = False
    out = go(gate1, fresh=[f])
    assert out["report"]["outOfScope"] == 1 and out["report"]["freshCluster"] == 0 and f["itemId"] not in out["flags"]
    assert f["itemId"] not in {g["itemId"] for g in out["dev"]}


def test_regate_served_anchor_rule_moves_a_kept_unit_and_is_not_a_miss():
    t1, t2, u = con("t1"), con("t2"), unit("srv")
    recs = [rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[t1, t2]),
            rec(t1, sym("srv/a.py", "srv/a.py::h"), "mcp", provider={"path": "srv/a.py"}),
            rec(t2, sym("srv/b.py", "srv/b.py::h"), "mcp", provider={"path": "srv/b.py"})] + bulk({s: 40 for s in STRATA})

    def layout(r):                       # unit (with its probe) held, the other served tool held, in separate clusters
        p = r["probe"][mk.locator_id(u)]
        other = mk.locator_id(t2) if p == mk.locator_id(t1) else mk.locator_id(t1)
        return r["side"][r["cluster_of"][mk.locator_id(u)]] == "held" and r["side"][r["cluster_of"][other]] == "held" \
            and r["cluster_of"][other] != r["cluster_of"][mk.locator_id(u)]
    ev = pick_commit(recs, layout)
    res = sm.split(recs, SRV, set(), commit(ev))
    p = res["probe"][mk.locator_id(u)]
    other = mk.locator_id(t2) if p == mk.locator_id(t1) else mk.locator_id(t1)
    out = sm.regate_list(recs, SRV, set(), commit(ev), [other], [])
    assert out["report"]["missedClusters"] == 1 and out["report"]["servedAnchorMoves"] == 1     # the unit followed, not a miss
    assert mk.locator_id(u) not in {g["anchorId"] for g in out["held"]}
    assert mk.locator_id(u) in {g["anchorId"] for g in out["dev"]}


def test_regate_new_held_list_never_contains_an_item_whose_anchor_is_in_dev_or_c(gate1):
    recs, res, by = gate1
    out = go(gate1, missed=[by["held"][0]["anchorId"]], fresh=[single("flow") for _ in range(5)], promote=True)
    dev_anchors = {g["anchorId"] for g in out["dev"]}
    assert not dev_anchors & {g["anchorId"] for g in out["held"]}


def write_gate1(tmp, recs, contact_ids=()):
    (tmp / "match.json").write_text(json.dumps({"schema": "ccg-match-v1", "records": recs}), encoding="utf-8")
    (tmp / "context.json").write_text(json.dumps({"units": []}), encoding="utf-8")
    jl_write(tmp / "contact.jsonl", [{"anchorId": a, "locator": {}, "source": "S14"} for a in contact_ids], canonical=False)


def make_manifest(tmp, ev=None):
    """Gate 1's held-out manifest, produced by the real pipeline (`split` then `manifest`) from the files in ``tmp``."""
    r = sm_cli("split", "--match", tmp / "match.json", "--context", tmp / "context.json", "--contact", tmp / "contact.jsonl",
               "--eval-commit", ev or commit(21), "--heldout-dir", tmp / "g1" / "ho", "--dev-dir", tmp / "g1" / "dev")
    assert r.returncode == 0, r.stderr
    r = sm_cli("manifest", "--heldout-dir", tmp / "g1" / "ho", "--eval-commit", ev or commit(21), "--protocol", "p",
               "--contact", tmp / "contact.jsonl", "--out", tmp / "manifest.json")
    assert r.returncode == 0, r.stderr
    return tmp / "manifest.json"


def regate_cli(tmp, ho, dev, extra=(), manifest=None, ev=None):
    manifest = manifest or make_manifest(tmp)
    return sm_cli("regate-list", "--match", tmp / "match.json", "--context", tmp / "context.json", "--contact", tmp / "contact.jsonl",
                  "--eval-commit", ev or commit(21), "--manifest", manifest, "--heldout-dir", ho, "--dev-dir", dev,
                  "--report", tmp / "rep.json", *extra)


def test_regate_cli_writes_the_new_list_files_counts_only_and_is_deterministic(tmp_path, gate1):
    recs, _, by = gate1
    write_gate1(tmp_path, recs)
    jl_write(tmp_path / "missed.jsonl", [{"anchorId": by["held"][0]["anchorId"]}])
    fresh = [single("mcp") for _ in range(36)]
    (tmp_path / "fresh.json").write_text(json.dumps({"records": fresh}), encoding="utf-8")
    outs = []
    for i in range(2):
        r = regate_cli(tmp_path, tmp_path / f"ho{i}", tmp_path / f"dev{i}", ["--missed", tmp_path / "missed.jsonl", "--fresh", tmp_path / "fresh.json"])
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1]
    assert "fresh-counting: 36" in outs[0] and "floor-met: true" in outs[0] and "missed-clusters: 1" in outs[0]
    assert "itemId" not in outs[0] and "f10" not in outs[0]
    for f in (tmp_path / "ho0").iterdir():
        assert f.read_bytes() == (tmp_path / "ho1" / f.name).read_bytes()
    assert sorted(f.name for f in (tmp_path / "ho0").iterdir()) == sorted(
        [f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl", "regate-flags.jsonl"])
    flags = {json.loads(l)["itemId"]: json.loads(l)["flag"] for l in (tmp_path / "ho0" / "regate-flags.jsonl").read_text().splitlines()}
    listed = {json.loads(l)["itemId"] for s in STRATA for l in (tmp_path / "ho0" / f"ac5-{s}.jsonl").read_text().splitlines()}
    assert set(flags) == listed and sum(v == "fresh-cluster" for v in flags.values()) == 36
    devd = {json.loads(l)["itemId"] for s in STRATA for l in (tmp_path / "dev0" / f"ac5-{s}.jsonl").read_text().splitlines()}
    assert by["held"][0]["itemId"] in devd and not devd & listed
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["floorMet"] is True and rep["freshCounting"] == 36


def test_regate_cli_never_overwrites_an_existing_list(tmp_path, gate1):
    recs, _, _ = gate1
    write_gate1(tmp_path, recs)
    assert regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev").returncode == 0
    before = {f.name: f.read_bytes() for f in (tmp_path / "ho").iterdir()}
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev2")
    assert r.returncode == 4 and "never overwrites" in r.stderr
    assert {f.name: f.read_bytes() for f in (tmp_path / "ho").iterdir()} == before


def test_regate_cli_exit_4_on_a_bad_missed_anchor_and_on_missing_input(tmp_path, gate1):
    recs, _, by = gate1
    write_gate1(tmp_path, recs)
    jl_write(tmp_path / "missed.jsonl", [{"anchorId": by["dev"][0]["anchorId"]}])
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev", ["--missed", tmp_path / "missed.jsonl"])
    assert r.returncode == 4
    assert regate_cli(tmp_path, tmp_path / "ho2", tmp_path / "dev", ["--fresh", tmp_path / "absent.json"]).returncode == 4


def test_regate_fresh_item_on_a_reserve_cluster_anchor_is_a_fresh_cluster_not_a_kept_one(gate1):
    _, _, by = gate1
    f = single("mcp")
    f["anchorId"] = by["reserve"][0]["anchorId"]
    out = go(gate1, fresh=[f])
    assert out["flags"][f["itemId"]] == "fresh-cluster"


def test_regate_fresh_unit_item_brings_its_own_served_set():
    recs, t1, t2, u = srv_system()
    tx = con("tx_only_in_the_fresh_item")
    fresh_u = rec(u, cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[tx])
    ev = pick_commit(recs, lambda r: all(r["side"][r["cluster_of"][mk.locator_id(t)]] == "held" for t in (t1, t2)))
    out = sm.regate_list(recs, SRV, {mk.locator_id(tx)}, commit(ev), [], [fresh_u])
    assert fresh_u["itemId"] in {g["itemId"] for g in out["dev"]}                       # S(u) = gate-1 set + the item's serves[]
    out = sm.regate_list(recs, SRV, set(), commit(ev), [], [fresh_u])
    assert out["flags"][fresh_u["itemId"]] == "fresh-cluster"


def test_regate_cli_passes_the_contact_file_and_eval_commit_through(tmp_path, gate1):
    recs, _, by = gate1
    ghost = "c" * 32
    write_gate1(tmp_path, recs, contact_ids=[ghost])
    f = single("http")
    f["anchorId"] = ghost
    (tmp_path / "fresh.json").write_text(json.dumps({"records": [f]}), encoding="utf-8")
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev", ["--fresh", tmp_path / "fresh.json"])
    assert r.returncode == 0, r.stderr
    assert "fresh-to-dev: 1" in r.stdout                                                # the contact file reached the router
    jl_write(tmp_path / "missed.jsonl", [{"anchorId": by["held"][0]["anchorId"]}])
    r = regate_cli(tmp_path, tmp_path / "ho2", tmp_path / "dev2", ["--missed", tmp_path / "missed.jsonl"])
    assert r.returncode == 0, r.stderr                                                  # commit 21's held cluster: the seed reached split


def test_regate_cli_equals_the_library_for_the_same_inputs_and_honours_promote_reserve(tmp_path, gate1):
    recs, _, by = gate1
    write_gate1(tmp_path, recs)
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev", ["--promote-reserve"])
    assert r.returncode == 0, r.stderr
    lib = sm.regate_list(recs, [], set(), commit(21), [], [], True)
    got = {json.loads(l)["itemId"] for s in STRATA for l in (tmp_path / "ho" / f"ac5-{s}.jsonl").read_text().splitlines()}
    assert got == {g["itemId"] for g in lib["held"]} and "fresh-cluster: 35" in r.stdout
    other = sm.regate_list(recs, [], set(), commit(22), [], [], True)                       # a different seed: a different split
    assert {g["itemId"] for g in other["held"]} != got


# =============================================================================================
# freeze-check (-p1-ml.md 6.2) and oracle-candidates (6.4)
# =============================================================================================

def git(repo, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    r = subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def write(repo, rel, text):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def commit_all(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


PLANT_A = sym("zz/srv.py", "zz/srv.py::planted_fn_zq7")
PLANT_I = sym("yy/cli.py", "yy/cli.py::caller_fn_k3")
PLANT = "see zz/srv.py::planted_fn_zq7 here"


def planted_record():
    aid, iid = mk.locator_id(PLANT_A), mk.locator_id(PLANT_I)
    return {"itemId": mk._ids.tuple_id(["evalItem", aid, iid]), "anchorId": aid, "impactedId": iid,
            "originals": [{"schema": "ccg-key-v1", "anchor": PLANT_A, "impacted": PLANT_I, "lines": [7]}]}


@pytest.fixture()
def freeze_repo(tmp_path):
    """A scratch repo: S47, then P0 / P1 / mixed / custodian-output / tools / deleted-file commits, a leak planted
    in the files of each class."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, "README.md", "readme\n")
    write(repo, "docs/gone.md", "to be deleted by a P1 commit\n")
    write(repo, "docs/mixed.md", "base\n")
    s47 = commit_all(repo, "S47 split")
    write(repo, "docs/p0-only.md", f"P0 text. {PLANT}\n")                           # leak in a non-P1 file: excluded
    write(repo, "docs/mixed.md", "p0 touch\n")
    write(repo, "docs/nonp1.md", f"misc text. {PLANT}\n")
    commit_all(repo, "C-1001 P0 close: wrap")
    write(repo, "docs/p1-clean.md", "nothing here\n")
    write(repo, "docs/p1-leak.md", f"a P1 doc. {PLANT}\n")                             # leak in a P1 file: the hit
    write(repo, "cpg/eval/tools/tool.py", f"# comment {PLANT}\n")                      # tools are NOT custodian output
    write(repo, "cpg/eval/dev/ac5-http.jsonl", f"{PLANT}\n")                           # custodian output: excluded
    write(repo, "cpg/eval/heldout-manifest.json", f"{PLANT}\n")
    write(repo, "cpg/eval/decisions/d.md", f"{PLANT}\n")
    write(repo, "cpg/eval/preregistration.jsonl", f"{PLANT}\n")
    (repo / "docs" / "gone.md").unlink()
    commit_all(repo, "C-1001 P1-13: build")
    write(repo, "docs/mixed.md", "p1 touch\n")
    write(repo, "docs/plan-set.md", "plan\n")
    commit_all(repo, "C-1001: P1 plan set")
    write(repo, "docs/nonp1.md", f"misc text again. {PLANT}\n")                         # a second non-P1 commit on the same path
    rec = commit_all(repo, "other work")
    return repo, s47, rec


def freeze(repo, s47, rec, *, heldout=None, dev=(), extra=()):
    ho = heldout or [repo.parent / f"ac5-{s}.jsonl" for s in STRATA] + [repo.parent / "ac5-reserve.jsonl"]
    for i, p in enumerate(ho):
        if not p.exists():
            jl_write(p, [planted_record()] if i == 0 else [])
    return sm_cli("freeze-check", "--repo", repo, "--s47", s47, "--record", rec, "--heldout", *ho, *extra)


def test_freeze_check_path_set_exclusions_and_the_one_planted_hit(freeze_repo):
    repo, s47, rec = freeze_repo
    out = repo.parent / "report.json"
    r = freeze(repo, s47, rec, extra=["--out", out])
    assert r.returncode == 1, (r.stdout, r.stderr)                                    # hits
    rep = json.loads(out.read_text())
    st = {p["path"]: p for p in rep["paths"]}
    assert st["docs/p0-only.md"] == {"path": "docs/p0-only.md", "status": "excluded", "reason": "non-p1-commits"}
    assert st["docs/nonp1.md"]["reason"] == "non-p1-commits"                         # two non-P1 commits, no P1 one
    for p in ("cpg/eval/dev/ac5-http.jsonl", "cpg/eval/heldout-manifest.json", "cpg/eval/decisions/d.md", "cpg/eval/preregistration.jsonl"):
        assert st[p]["reason"] == "custodian-output", p
    assert st["docs/mixed.md"]["status"] == "checked"                                # a mixed path counts as P1
    assert st["docs/plan-set.md"]["status"] == "checked"                             # 'C-1001: P1 plan set' matches
    assert st["docs/gone.md"]["status"] == "deleted"                                  # in the diff, absent at the record: listed
    assert "README.md" not in st                                                     # untouched in the range: not in the set
    assert st["docs/p1-clean.md"]["hits"] == []
    assert len(st["docs/p1-leak.md"]["hits"]) == 1 and len(st["cpg/eval/tools/tool.py"]["hits"]) == 1   # exactly one each
    assert rep["hits"] == 2 and rep["failed"] == 0
    assert "hit: docs/p1-leak.md " in r.stdout and "hit: cpg/eval/tools/tool.py " in r.stdout
    assert "planted_fn_zq7" not in r.stdout and "zz/srv.py" not in r.stdout           # hashes only


REAL_P1_SUBJECTS = [                                    # copied from `git log --format=%s 511b14cf..HEAD` shapes
    "C-1001 P1: ledger - crash root cause (runaway mutant, no memory cap), 5b re-dispatch",
    "cpg C-1001 P1-13: build", "cpg C-1001 P1-01: x", "cpg C-1001 P1-48 + P1-02: runner and toolchain", "cpg C-1001 P1 N-ML: note",
    "C-1001: P1 plan set", "C-1001 P1T-30",
]


@pytest.mark.parametrize("subject", REAL_P1_SUBJECTS)
def test_freeze_check_real_p1_subject_shapes_are_p1_commits(subject):
    assert sm.is_p1_commit(subject)
    assert sm.classify_path("cpg/ccg/atrest.py", [subject]) is None                          # the file stays in the scan set


@pytest.mark.parametrize("subject", ["C-1001 P0 close: wrap", "coordination: U5.10b delivered", "C-1002 P1-3: other initiative",
                                     "python-web-quirks-kb W7a: fixture", "decision-models U5.9 fix round"])
def test_freeze_check_non_p1_commits_are_still_excluded(subject):
    assert not sm.is_p1_commit(subject)
    assert sm.classify_path("cpg/ccg/atrest.py", [subject]) == "non-p1-commits"


def test_freeze_check_p1_tag_in_the_body_counts_and_the_rule_is_documented_direction(freeze_repo):
    assert sm.is_p1_commit("update notes\n\nPart of C-1001 P1-05 dispatch")                  # message, not only subject: over-inclusion is safe
    assert sm.classify_path("docs/x.md", ["C-1001 P0 close", "cpg C-1001 P1-3: x"]) is None   # mixed: counts as P1
    assert sm.classify_path("docs/x.md", []) is None                                          # no attributable commit: stays in
    assert sm.classify_path("cpg/eval/tools/x.py", ["unrelated"]) == "non-p1-commits"
    assert sm.classify_path("cpg/eval/dev/x", ["C-1001 P1-3"]) == "custodian-output"
    assert sm.classify_path("cpg/eval/device.py", ["C-1001 P1-3"]) is None                    # prefix needs the trailing slash


def test_freeze_check_a_real_git_commit_with_the_tag_only_in_the_body_keeps_its_file_in_the_scan(freeze_repo):
    repo, s47, _ = freeze_repo
    write(repo, "cpg/ccg/bodytag.py", f"# {PLANT}\n")
    rec = commit_all(repo, "update notes\n\nPart of C-1001 P1-05 dispatch")
    rep = sm.freeze_check(repo, s47, rec, [planted_record()], [], [])
    assert next(p for p in rep["paths"] if p["path"] == "cpg/ccg/bodytag.py")["status"] == "checked"


def test_freeze_check_a_real_git_commit_with_a_cpg_prefixed_subject_keeps_its_file_in_the_scan(freeze_repo):
    repo, s47, _ = freeze_repo
    write(repo, "cpg/ccg/prefixed.py", f"# {PLANT}\n")
    rec = commit_all(repo, "cpg C-1001 P1-48 + P1-02: runner and toolchain\n\nbody line")
    rep = sm.freeze_check(repo, s47, rec, [planted_record()], [], [])
    row = next(p for p in rep["paths"] if p["path"] == "cpg/ccg/prefixed.py")
    assert row["status"] == "checked" and len(row["hits"]) == 1


def test_freeze_check_planted_probe_maps_back_privately_to_its_item(freeze_repo):
    repo, s47, rec = freeze_repo
    pm = repo.parent / "map.json"
    r = freeze(repo, s47, rec, extra=["--probe-map", pm])
    hashes = {l.split()[-1] for l in r.stdout.splitlines() if l.startswith("hit: ")}
    m = json.loads(pm.read_text())
    assert len(hashes) == 1 and all(len(h) == 12 for h in hashes)
    assert m[next(iter(hashes))] == [planted_record()["itemId"]]


def test_freeze_check_clean_repo_exits_0_and_positive_control_passes(freeze_repo):
    repo, s47, rec = freeze_repo
    jl_write(repo.parent / "other.jsonl", [])
    other = sym("nowhere/q.py", "nowhere/q.py::unrelated_fn_x9")
    ho = [repo.parent / f"ac5-{s}.jsonl" for s in STRATA] + [repo.parent / "ac5-reserve.jsonl"]
    for p in ho:
        jl_write(p, [])
    aid, iid = mk.locator_id(other), mk.locator_id(PLANT_I)
    jl_write(ho[0], [{"itemId": mk._ids.tuple_id(["evalItem", aid, iid]), "anchorId": aid, "impactedId": iid,
                      "originals": [{"schema": "ccg-key-v1", "anchor": other, "impacted": sym("nowhere/z.py", "nowhere/z.py::zq"), "lines": []}]}])
    r = sm_cli("freeze-check", "--repo", repo, "--s47", s47, "--record", rec, "--heldout", *ho, "--positive-control")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "hits: 0" in r.stdout and "control: ok" in r.stdout


def test_freeze_check_a_blind_checker_fails_the_positive_control(freeze_repo, monkeypatch):
    rec_ = planted_record()
    assert sm.positive_control("some artifact text", [rec_], [], []) is True
    monkeypatch.setattr(sm, "check_report", lambda *a, **k: ([], 3))                         # a checker that sees nothing
    assert sm.positive_control("some artifact text", [rec_], [], []) is False


def test_freeze_check_unreadable_file_is_a_failure_exit_4_never_a_skip(freeze_repo, monkeypatch):
    repo, s47, rec = freeze_repo
    real = sm.read_at

    def flaky(r, rev, path):
        if path == "docs/p1-clean.md":
            raise ValueError("unreadable")
        return real(r, rev, path)
    monkeypatch.setattr(sm, "read_at", flaky)
    rep = sm.freeze_check(repo, s47, rec, [planted_record()], [], [])
    assert rep["exit"] == 4 and rep["failed"] == 1
    assert next(p for p in rep["paths"] if p["path"] == "docs/p1-clean.md")["status"] == "failed"
    assert rep["hits"] == 2                                                                  # the other files were still checked


@pytest.mark.parametrize("rc", [2, 4])
def test_freeze_check_per_file_exit_2_or_4_is_a_failure_even_with_no_hits(freeze_repo, monkeypatch, rc):
    repo, s47, rec = freeze_repo
    monkeypatch.setattr(sm, "check_text", lambda *a, **k: (rc, [], 0))
    rep = sm.freeze_check(repo, s47, rec, [planted_record()], [], [])
    assert rep["failed"] > 0 and rep["exit"] == 4
    assert rep["checked"] == 0 and rep["hits"] == 0


def test_freeze_check_a_submodule_path_is_a_real_unreadable_file_failure(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, "a.md", "a\n")
    s47 = commit_all(repo, "S47")
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{s47},sub")
    git(repo, "commit", "-m", "C-1001 P1-1: gitlink")
    rec = git(repo, "rev-parse", "HEAD")
    rep = sm.freeze_check(repo, s47, rec, [planted_record()], [], [])
    assert [p["status"] for p in rep["paths"]] == ["failed"] and rep["exit"] == 4


def test_freeze_check_refuses_an_incomplete_heldout_file_list_with_exit_2(freeze_repo):
    repo, s47, rec = freeze_repo
    only = repo.parent / "ac5-http.jsonl"
    jl_write(only, [planted_record()])
    r = sm_cli("freeze-check", "--repo", repo, "--s47", s47, "--record", rec, "--heldout", only)
    assert r.returncode == 2 and "omits" in r.stderr


def test_freeze_check_bad_revision_is_exit_4(freeze_repo):
    repo, s47, rec = freeze_repo
    r = freeze(repo, s47, "deadbeef" * 5)
    assert r.returncode == 4


def test_freeze_check_oracle_candidates_are_extra_probes_passed_by_heldout(freeze_repo):
    repo, s47, rec = freeze_repo
    jl_write(repo.parent / "edges.jsonl", [{"consumer": sym("ui/./client.ts", "ui/client.ts::fetchOrders"),
                                            "key": con("GET /orders/{order_id}", "http"), "lines": [4, 3]}])
    o = repo.parent / "oracle-candidates.jsonl"
    r = sm_cli("oracle-candidates", "--in", repo.parent / "edges.jsonl", "--out", o)
    assert r.returncode == 0, r.stderr
    row = json.loads(o.read_text())
    assert row["originals"][0]["schema"] == "ccg-key-v1" and row["originals"][0]["lines"] == [3, 4]
    assert row["originals"][0]["anchor"]["keyNorm"] == "GET /orders/{}" and row["originals"][0]["impacted"]["path"] == "ui/client.ts"
    write(repo, "docs/oracle-leak.md", "the consumer ui/client.ts::fetchOrders misses\n")
    rec2 = commit_all(repo, "C-1001 P1-47: oracle note")
    ho = [repo.parent / f"ac5-{s}.jsonl" for s in STRATA] + [repo.parent / "ac5-reserve.jsonl"]
    for p in ho:
        jl_write(p, [])
    clean = sm_cli("freeze-check", "--repo", repo, "--s47", s47, "--record", rec2, "--heldout", *ho)
    assert clean.returncode == 0 and "hits: 0" in clean.stdout
    withor = sm_cli("freeze-check", "--repo", repo, "--s47", s47, "--record", rec2, "--heldout", *ho, o)
    assert withor.returncode == 1 and "hit: docs/oracle-leak.md " in withor.stdout


def test_oracle_candidates_refuse_a_non_contract_key_and_dedupe(tmp_path):
    e = {"consumer": sym("a.py", "a.py::f"), "key": con("t1")}
    assert len(sm.oracle_candidates([e, dict(e)])) == 1
    with pytest.raises(ValueError):
        sm.oracle_candidates([{"consumer": sym("a.py", "a.py::f"), "key": sym("b.py", "b.py::g")}])


def test_freeze_check_public_dev_items_and_contact_remove_a_probe_and_both_arguments_reach_the_check(freeze_repo):
    repo, s47, rec = freeze_repo
    dev = repo.parent / "dev-ac5-http.jsonl"
    jl_write(dev, [planted_record()])                                              # the same value in a dev item: public
    r = freeze(repo, s47, rec, extra=["--dev", dev])
    assert r.returncode == 0 and "hits: 0" in r.stdout, (r.stdout, r.stderr)
    contact = repo.parent / "contact.jsonl"
    jl_write(contact, [{"anchorId": mk.locator_id(PLANT_A), "locator": PLANT_A, "source": "S14"}])      # in C: public
    r = freeze(repo, s47, rec, extra=["--contact", contact])
    assert r.returncode == 0 and "hits: 0" in r.stdout, (r.stdout, r.stderr)


def test_freeze_check_a_failed_positive_control_is_exit_4_even_when_nothing_hit(freeze_repo):
    repo, s47, rec = freeze_repo
    ho = [repo.parent / f"ac5-{s}.jsonl" for s in STRATA] + [repo.parent / "ac5-reserve.jsonl"]
    for p in ho:
        jl_write(p, [])                                                          # no held-out probe at all: the control cannot plant one
    r = sm_cli("freeze-check", "--repo", repo, "--s47", s47, "--record", rec, "--heldout", *ho, "--positive-control")
    assert r.returncode == 4 and "hits: 0" in r.stdout and "control: FAILED" in r.stdout


# --- M1: the re-derived gate-1 state must be the one the manifest recorded -----------------------

def planted_gate1(tmp, gate1_, contact_from_held=True):
    recs, _, by = gate1_
    held_a = by["held"][0]["anchorId"]
    write_gate1(tmp, recs, contact_ids=[held_a, "c" * 32] if contact_from_held else ["c" * 32])
    return recs, held_a


def assert_refused_nothing_written(r, ho, dev):
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "does not match the manifest" in r.stderr and "nothing written" in r.stderr
    assert not ho.exists() and not dev.exists()


def test_regate_manifest_premise_it_hashes_the_same_bytes_the_rederivation_writes(tmp_path, gate1):
    planted_gate1(tmp_path, gate1)
    man = json.loads(make_manifest(tmp_path).read_text())
    assert [f["name"] for f in man["files"]] == [f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl"]
    res = sm.split(json.loads((tmp_path / "match.json").read_text())["records"], [], {json.loads(l)["anchorId"] for l in (tmp_path / "contact.jsonl").read_text().splitlines()}, commit(21))
    assert sm.gate1_problems(res, man, commit(21), 2) == []
    held, _ = sm.split_files(res)
    assert {n: sm.entry_of(n, b)["sha256"] for n, b in held.items()} == {f["name"]: f["sha256"] for f in man["files"]}


def test_regate_manifest_unchanged_inputs_are_accepted(tmp_path, gate1):
    planted_gate1(tmp_path, gate1)
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev")
    assert r.returncode == 0, r.stderr


def test_regate_manifest_a_dropped_contact_anchor_is_refused_and_nothing_is_written(tmp_path, gate1):
    _, held_a = planted_gate1(tmp_path, gate1)
    man = make_manifest(tmp_path)
    lines = (tmp_path / "contact.jsonl").read_text().splitlines()
    kept = [l for l in lines if held_a not in l]                                   # drop the anchor that moved a cluster to dev
    assert len(kept) == len(lines) - 1
    (tmp_path / "contact.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert_refused_nothing_written(regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev", manifest=man), tmp_path / "ho", tmp_path / "dev")


def test_regate_manifest_a_dropped_item_less_contact_anchor_is_caught_by_the_anchor_count(tmp_path, gate1):
    planted_gate1(tmp_path, gate1)
    man = make_manifest(tmp_path)
    lines = (tmp_path / "contact.jsonl").read_text().splitlines()
    (tmp_path / "contact.jsonl").write_text("\n".join(l for l in lines if "c" * 32 not in l) + "\n", encoding="utf-8")
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev", manifest=man)
    assert_refused_nothing_written(r, tmp_path / "ho", tmp_path / "dev")
    assert "contactAnchors" in r.stderr and "ac5-" not in r.stderr                    # the lists are identical: only the count differs


def test_regate_manifest_a_different_eval_commit_or_match_is_refused(tmp_path, gate1):
    recs, _ = planted_gate1(tmp_path, gate1)
    held_id = gate1[2]["held"][1]["itemId"]
    man = make_manifest(tmp_path)
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev", manifest=man, ev=commit(22))
    assert_refused_nothing_written(r, tmp_path / "ho", tmp_path / "dev")
    assert "evalCommit" in r.stderr and "splitSeed" in r.stderr
    (tmp_path / "match.json").write_text(json.dumps({"schema": "ccg-match-v1", "records": [g for g in recs if g["itemId"] != held_id]}), encoding="utf-8")   # a held-out item missing
    r = regate_cli(tmp_path, tmp_path / "ho2", tmp_path / "dev2", manifest=man)
    assert r.returncode == 1 and "does not match the manifest" in r.stderr and not (tmp_path / "ho2").exists()


def test_regate_manifest_a_corrupted_manifest_hash_is_refused_by_field_name_only(tmp_path, gate1):
    planted_gate1(tmp_path, gate1)
    man = make_manifest(tmp_path)
    d = json.loads(man.read_text())
    d["files"][0]["sha256"] = "0" * 64
    man.write_text(json.dumps(d), encoding="utf-8")
    r = regate_cli(tmp_path, tmp_path / "ho", tmp_path / "dev", manifest=man)
    assert_refused_nothing_written(r, tmp_path / "ho", tmp_path / "dev")
    assert "ac5-http.jsonl: sha256" in r.stderr


def test_regate_manifest_is_required(tmp_path, gate1):
    planted_gate1(tmp_path, gate1)
    r = sm_cli("regate-list", "--match", tmp_path / "match.json", "--context", tmp_path / "context.json", "--contact", tmp_path / "contact.jsonl",
               "--eval-commit", commit(21), "--heldout-dir", tmp_path / "ho", "--dev-dir", tmp_path / "dev")
    assert r.returncode == 2


def test_regate_a_fresh_unit_item_routed_to_dev_does_not_expose_another_fresh_unit_item():
    """Routing reads cluster sides and C only. A fresh item sent to dev in the same call does not feed ``exposed()``: the only
    fresh items that route to dev through S(u) are unit items, and a unit anchor is never in another unit's served set."""
    srv2 = [{"rootPath": "srv", "lang": "python"}, {"rootPath": "srv2", "lang": "python"}]
    ta, tb = con("ta"), con("tb")
    recs = [rec(ta, sym("srv/a.py", "srv/a.py::h"), "mcp", provider={"path": "srv/a.py"}),
            rec(tb, sym("srv2/b.py", "srv2/b.py::h"), "mcp", provider={"path": "srv2/b.py"})] + bulk({s: 40 for s in STRATA})
    ev = next(i for i in range(1, 60) if all((lambda r: r["side"][r["cluster_of"][mk.locator_id(t)]] == "held")(sm.split(recs, srv2, set(), commit(i))) for t in (ta, tb)))
    u1, u2 = rec(unit("srv"), cfg(".mcp.json", "mcpServers.srv"), "wiring", serves=[ta]), \
        rec(unit("srv2", "python"), cfg(".mcp.json", "mcpServers.srv2"), "wiring", serves=[tb])
    out = sm.regate_list(recs, srv2, {mk.locator_id(ta)}, commit(ev), [], [u1, u2])
    assert u1["itemId"] in {g["itemId"] for g in out["dev"]}                      # S(srv) holds a contacted anchor
    assert out["flags"][u2["itemId"]] == "fresh-cluster"                          # srv2 is unaffected by u1's routing


def test_freeze_check_refuses_out_and_probe_map_inside_the_repo_tree(freeze_repo):
    repo, s47, rec = freeze_repo
    for flag in ("--out", "--probe-map"):
        r = freeze(repo, s47, rec, extra=[flag, repo / "docs" / "private.json"])
        assert r.returncode == 2 and "inside the repo tree" in r.stderr
        assert not (repo / "docs" / "private.json").exists()
