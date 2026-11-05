"""Unit tests for the S27 metric helpers (spikes/frontend-fitness/s27_metrics.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"))
import s27_metrics as m  # noqa: E402


def meth(file, l, le, name, full):
    return {"file": file, "l": l, "le": le, "name": name, "fullName": full}


def M(*ms):
    import collections
    d = {"methods": collections.defaultdict(dict), "mlist": collections.defaultdict(lambda: collections.defaultdict(list)), "byFull": {}}
    for x in ms:
        d["methods"][x["file"]][x["l"]] = x
        d["mlist"][x["file"]][x["l"]].append(x)
        d["byFull"][x["fullName"]] = x
    return d


def test_wilson_and_rank():
    assert m.wilson(152, 155) == (0.9446, 0.9934)
    assert m.nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 9, 10) == 9
    assert m.mcnemar_exact(6, 0) == 0.03125


def test_locate_prefers_named_method_on_shared_line():
    mm = M(meth("a.py", 5, 9, "<fakeNew>", "a:<fakeNew>"), meth("a.py", 5, 9, "__init__", "a:__init__"))
    t = {"path": "a.py", "defLine": 5, "qname": "a.py::C.__init__"}
    assert m.locate(mm, t, 0) == "a:__init__"


def test_locate_annotation_span_and_lambda_is_symbol():
    mm = M(meth("A.java", 10, 14, "run", "A.run"))
    assert m.locate(mm, {"path": "A.java", "defLine": 12, "qname": "A.java::A.run"}, 0) == "A.run"
    assert not m.NONSYMBOL_NAME.search("<lambda>3") and m.NONSYMBOL_NAME.search("<fakeNew>")


def test_strict_needs_a_matched_call():
    mm = M(meth("a.py", 3, 4, "f", "a:f"))
    ro = {"calleeName": "f"}
    tr = {"T": [{"path": "a.py", "defLine": 3, "qname": "a.py::f", "role": "impl"}]}
    base = {"resolvedSymbols": [], "matched": False}
    j = m.judge(ro, tr, base, mm, 0)
    assert j["foundLenient"] and not j["found"]
    j2 = m.judge(ro, tr, {**base, "matched": True}, mm, 0)
    assert j2["found"]


def test_recall_decision_uses_stratum():
    metrics = {c: {"internalSites": 155, "recall": {"k": 152, "wilson95": [0, 1]}} for c in ("off", "on")}
    strata = {"nonContact": {c: {"internalSites": 149, "recall": {"k": 146, "wilson95": [0, 1]}} for c in ("off", "on")}}
    d = m.recall_decision("python", metrics, strata, "recall")
    assert d["worse"] == "fail" and d["off"]["missBudgetAt98"] == 2 and d["off"]["misses"] == 3


def _site(file, l, full, ext=False):
    return {"matched": True, "nm": 0, "call": {"name": "x", "l": 1, "c": 1, "code": "x()"},
            "res": [{"fullName": full, "file": file, "l": l, "vendored": False, "ext": ext}]}


def test_a5_resolved_target_is_classified_by_fullname_not_line():
    # two METHODs on one line; a (file, line) map keeps the LAST one. The target is the first (a real symbol).
    mm = M(meth("a.py", 5, 9, "__init__", "a:__init__"), meth("a.py", 5, 9, "<fakeNew>", "a:<fakeNew>"))
    mm["sites"] = {"k1": _site("a.py", 5, "a:__init__"), "k2": _site("a.py", 5, "a:<fakeNew>")}
    out = m.analyse("python", "off", mm, [{"ordinal": 1, "siteKey": "k1"}, {"ordinal": 2, "siteKey": "k2"}], 0)
    assert [r["fullName"] for r in out[1]["resolvedSymbols"]] == ["a:__init__"]      # fails if keyed by (file, line), last wins
    assert [r["fullName"] for r in out[2]["resolvedNonSymbol"]] == ["a:<fakeNew>"]


def test_a6_lambda_target_at_a_truth_location_is_a_symbol_and_found():
    mm = M(meth("f.go", 7, 9, "<lambda>1", "f:<lambda>1"))
    mm["sites"] = {"k": _site("f.go", 7, "f:<lambda>1")}
    ro = {"ordinal": 1, "siteKey": "k", "calleeName": "zzz"}          # no name-match credit: only resolution can find it
    ans = m.analyse("go", "off", mm, [ro], 0)[1]
    tr = {"T": [{"path": "f.go", "defLine": 7, "qname": "f.go::X.<anon>#1", "role": "impl"}]}
    j = m.judge(ro, tr, ans, mm, 0)
    assert j["found"] and j["foundRes"] and j["resolved"] and j["correct"]    # fails if `<lambda>` is listed as a non-symbol
