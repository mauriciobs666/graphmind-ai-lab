"""S57 canonical-schema validator for truth/<lang>.final.jsonl and adjudication/<lang>.noise.json
(cpg/spikes/frontend-fitness/validate_final.py; closes analyst gate blocker B1).

Each rule has a planted-violation test. Run: cd cpg && .venv/bin/python -m pytest -q tests/test_validate_final.py
CCG_FF_DIR points the suite at a COPY of the spike directory (mutation testing).
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

FF = Path(os.environ.get("CCG_FF_DIR") or (Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"))
sys.path.insert(0, str(FF))
import validate_final as vf  # noqa: E402


def good(ordinal=1):
    return {
        "ordinal": ordinal, "siteId": f"{ordinal:016x}", "site": "src/a.py:10",
        "T": [{"path": "src/b.py", "qname": "src/b.py::f", "defLine": 3, "role": "impl"}],
        "N": [], "external": [], "notACall": False, "dynamic": False, "scored": True,
        "notScoredReason": None, "basis": "agreed", "strata": ["AI"], "evidence": "src/a.py:10 calls f",
    }


def rules(rec):
    return {r for r, _ in vf.validate_record(rec)}


def mut(**kw):
    r = good()
    r.update(kw)
    return r


def test_good_record_has_no_violation():
    assert vf.validate_record(good()) == []


def test_good_record_with_optional_T_extras_and_null_qname():
    r = good()
    r["T"][0].update(qname=None, qnameDerived=True, macroGenerated=False)
    assert vf.validate_record(r) == []


def ext_only(**kw):
    r = good()
    r.update(T=[], external=["react"], strata=["AE"])
    r.update(kw)
    return r


PLANTED = [
    ("missing-key", lambda: {k: v for k, v in good().items() if k != "site"}, "keys"),
    ("unknown-key", lambda: mut(citations=["a.py:1"]), "keys"),
    ("old-evidence-key-citation", lambda: {**{k: v for k, v in good().items() if k != "evidence"}, "citation": "a.py:1"}, "keys"),
    ("ordinal-not-int", lambda: mut(ordinal="1"), "ordinal"),
    ("ordinal-bool", lambda: mut(ordinal=True), "ordinal"),
    ("ordinal-zero", lambda: mut(ordinal=0), "ordinal"),
    ("dynamic-not-bool", lambda: mut(dynamic="no"), "bools"),
    ("T-empty-path", lambda: mut(T=[{"path": "", "qname": "a::f", "defLine": 1, "role": "impl"}]), "T"),
    ("T-qname-int", lambda: mut(T=[{"path": "a.py", "qname": 5, "defLine": 1, "role": "impl"}]), "T"),
    ("T-item-not-object", lambda: mut(T=["a.py"]), "T"),
    ("siteId-empty", lambda: mut(siteId=""), "siteId"),
    ("site-dict", lambda: mut(site={"path": "a.py", "line": 3, "column": 1, "callText": "f"}), "site"),
    ("site-no-line", lambda: mut(site="src/a.py"), "site"),
    ("external-bare-string", lambda: ext_only(external="react"), "external"),
    ("external-none", lambda: mut(external=None), "external"),
    ("external-dict-items", lambda: ext_only(external=[{"kind": "external", "package": "react"}]), "external"),
    ("external-empty-string", lambda: ext_only(external=[""]), "external"),
    ("N-not-list", lambda: mut(N="x"), "N"),
    ("N-non-string-item", lambda: mut(N=[1]), "N"),
    ("T-not-list", lambda: mut(T=None), "T"),
    ("T-role-outside-impl-decl", lambda: mut(T=[{"path": "a.py", "qname": "a::f", "defLine": 1, "role": "internal"}]), "T"),
    ("T-missing-defLine", lambda: mut(T=[{"path": "a.py", "qname": "a::f", "role": "impl"}]), "T"),
    ("T-defLine-zero", lambda: mut(T=[{"path": "a.py", "qname": "a::f", "defLine": 0, "role": "impl"}]), "T"),
    ("T-unknown-extra", lambda: mut(T=[{"path": "a.py", "qname": "a::f", "defLine": 1, "role": "impl", "x": 1}]), "T"),
    ("T-extra-not-bool", lambda: mut(T=[{"path": "a.py", "qname": "a::f", "defLine": 1, "role": "impl", "qnameDerived": "yes"}]), "T"),
    ("basis-outside-vocabulary", lambda: mut(basis="label-confirmed"), "basis"),
    ("scored-not-bool", lambda: mut(scored="true"), "bools"),
    ("notACall-not-bool", lambda: mut(notACall=0), "bools"),
    ("scored-with-reason", lambda: mut(notScoredReason="notACall"), "notScoredReason"),
    ("unscored-without-reason", lambda: mut(scored=False, T=[]), "notScoredReason"),
    ("reason-outside-vocabulary", lambda: mut(scored=False, T=[], notScoredReason="because"), "notScoredReason"),
    ("notACall-but-scored", lambda: mut(notACall=True, T=[], notScoredReason=None), "scored-consistency"),
    ("dynamic-empty-T-but-scored", lambda: ext_only(dynamic=True, external=["x"]) | {"T": []}, "scored-consistency"),
    ("scored-with-nothing-decided", lambda: mut(T=[], strata=["AE"]), "nothing-decided"),
    ("label-only-without-AS", lambda: mut(basis="label-only", strata=["AI"]), "label-only-AS"),
    ("strata-not-list", lambda: mut(strata="AI"), "strata"),
    ("strata-compound-tag", lambda: mut(strata=["audit-U:AI"]), "strata"),
    ("strata-duplicate", lambda: mut(strata=["AI", "AI"]), "strata"),
    ("evidence-list", lambda: mut(evidence=["src/a.py:10"]), "evidence"),
    ("evidence-empty", lambda: mut(evidence=""), "evidence"),
    ("evidence-no-path-line", lambda: mut(evidence="looks fine to me"), "evidence"),
]


@pytest.mark.parametrize("name,make,rule", PLANTED, ids=[p[0] for p in PLANTED])
def test_planted_violation(name, make, rule):
    assert rule in rules(make())


def test_unscored_sites_may_decide_nothing():
    r = mut(notACall=True, T=[], scored=False, notScoredReason="notACall")
    assert vf.validate_record(r) == []


def test_empty_strata_is_allowed():
    assert vf.validate_record(mut(strata=[])) == []


def test_scored_line_may_decide_through_external_alone_or_N_alone():
    assert vf.validate_record(ext_only()) == []
    assert vf.validate_record(mut(T=[], external=[], N=["synthesized copy(): A.kt:5"])) == []


def test_T_item_that_is_not_an_object_says_so():
    assert any("not an object" in m for _, m in vf.validate_record(mut(T=["a.py"])))


def test_dynamic_with_targets_stays_scored():
    assert vf.validate_record(mut(dynamic=True)) == []


# ---- file level ---------------------------------------------------------------------------------

def test_file_ordinals_must_run_one_to_n_in_order():
    rows = [good(1), good(3)]
    assert any(v.rule == "ordinal-sequence" for v in vf.validate_rows(rows))


def test_file_duplicate_siteId():
    a, b = good(1), good(2)
    b["siteId"] = a["siteId"]
    vs = vf.validate_rows([a, b])
    assert any(v.rule == "siteId-duplicate" and v.ordinal == 2 for v in vs)


def test_file_clean_rows_pass():
    assert vf.validate_rows([good(1), good(2)]) == []


def test_file_reports_ordinal_of_bad_line():
    bad = good(2)
    bad["basis"] = "nope"
    vs = vf.validate_rows([good(1), bad])
    assert [(v.ordinal, v.rule) for v in vs] == [(2, "basis")]


def test_file_must_match_input_ordinals_and_siteIds():
    rows = [good(1), good(2)]
    assert vf.validate_rows(rows, expected=[(1, rows[0]["siteId"]), (2, rows[1]["siteId"])]) == []
    assert any(v.rule == "input-mismatch" for v in vf.validate_rows(rows, expected=[(1, rows[0]["siteId"]), (2, "zzz")]))
    assert any(v.rule == "input-mismatch" for v in vf.validate_rows(rows, expected=[(1, rows[0]["siteId"])]))


# ---- noise.json ---------------------------------------------------------------------------------

def good_noise(rows):
    n = len(rows)
    return {
        "lang": "x", "N": n, "comparable": n, "referenceSilent": 0,
        "agreement": {"agreed": n, "comparable": n, "rate": 1.0, "wilson95": [0.5, 1.0]},
        "adjudicated": {"n": 2, "W": 1, "byWrong": {"label": 1, "neither": 1}, "byErrorKind": {"target": 1, "none": 1}},
        "referenceError": {"wrong": 0, "comparable": n, "rate": 0.0, "wilson95": [0.0, 0.5]},
        "audit": {"AI": {"N_s": 2, "n": 2, "k": 0, "wilson95": [0.0, 0.5]},
                  "AE": {"N_s": 0, "n": 0, "k": 0, "wilson95": None},
                  "AS": {"N_s": 0, "n": 0, "k": 0, "wilson95": None}},
        "handError": {"e_all": {"point": 0.1, "conservative": 0.2}, "e_res": {"point": 0.04, "conservative": 0.1},
                      "noiseLimited": False},
        "census": {"targetsRead": 5},
        "closedClasses": {"count": 0, "byKind": {"role": [], "coordinate": [], "shape": []}},
        "unscored": {"byReason": {}, "underivable": 0},
        "basisCounts": {"agreed": n},
        "flagByVersion": {"v1.10": {"figure": 0.1, "flagged": True}, "v1.11": {"figure": 0.06, "flagged": True},
                          "v1.12": {"figure": 0.04, "flagged": False}},
        "openPointsS27": {"classLineTargets": "none found: scan of 2 definition lines", "anonymousLambdaIndices": [], "externalsAbsentFromTree": []},
        "detail": {"anything": "verbatim original"},
    }


def nrules(noise, rows=None):
    return {r for r, _ in vf.validate_noise(noise, rows)}


def test_good_noise_passes():
    rows = [good(1), good(2)]
    assert vf.validate_noise(good_noise(rows), rows) == []


def mutate_noise(f):
    rows = [good(1), good(2)]
    n = good_noise(rows)
    f(n)
    return n, rows


NOISE_PLANTED = [
    ("missing-key", lambda n: n.pop("census"), "noise-keys"),
    ("legacy-key", lambda n: n.update(handErrorRate={}), "noise-keys"),
    ("N-not-int", lambda n: n.update(N="2"), "noise-N"),
    ("N-differs-from-final", lambda n: n.update(N=3, comparable=3), "noise-N"),
    ("comparable-plus-silent", lambda n: n.update(referenceSilent=1), "noise-comparable"),
    ("agreement-rate-not-ratio", lambda n: n["agreement"].update(rate=0.5), "noise-agreement"),
    ("wilson-not-pair", lambda n: n["agreement"].update(wilson95=[0.5]), "noise-agreement"),
    ("adjudicated-byWrong-sum", lambda n: n["adjudicated"]["byWrong"].update(label=5), "noise-adjudicated"),
    ("adjudicated-errorKind-vocab", lambda n: n["adjudicated"]["byErrorKind"].update(schema=1, none=0, target=0) , "noise-adjudicated"),
    ("adjudicated-W", lambda n: n["adjudicated"].update(W=2), "noise-adjudicated"),
    ("audit-k-over-n", lambda n: n["audit"]["AI"].update(k=3), "noise-audit"),
    ("audit-stratum-missing", lambda n: n["audit"].pop("AS"), "noise-audit"),
    ("handError-flag-contradicts-e_res", lambda n: n["handError"].update(noiseLimited=True), "noise-handError"),
    ("handError-e_res-missing", lambda n: n["handError"].pop("e_res"), "noise-handError"),
    ("census-targetsRead-null-with-closed-class", lambda n: (n["census"].update(targetsRead=None), n["closedClasses"].update(count=1, byKind={"role": [1], "coordinate": [], "shape": []})), "noise-census"),
    ("closedClasses-count", lambda n: n["closedClasses"].update(count=4), "noise-closedClasses"),
    ("closedClasses-kind-vocab", lambda n: n["closedClasses"]["byKind"].update(target=[1]), "noise-closedClasses"),
    ("closedClasses-share-over-cap", lambda n: n["closedClasses"].update(count=2, byKind={"role": [1, 2], "coordinate": [], "shape": []}), "noise-closedClasses"),
    ("unscored-differs-from-final", lambda n: n["unscored"].update(byReason={"notACall": 1}), "noise-unscored"),
    ("basisCounts-differs-from-final", lambda n: n.update(basisCounts={"agreed": 1, "adjudicated": 1}), "noise-basisCounts"),
    ("basisCounts-vocab", lambda n: n.update(basisCounts={"agreed": 1, "bogus": 1}), "noise-basisCounts"),
    ("flagByVersion-missing", lambda n: n.pop("flagByVersion"), "noise-keys"),
    ("flagByVersion-not-dict", lambda n: n.update(flagByVersion=[]), "noise-flagByVersion"),
    ("flagByVersion-missing-version", lambda n: n["flagByVersion"].pop("v1.11"), "noise-flagByVersion"),
    ("flagByVersion-extra-version", lambda n: n["flagByVersion"].update({"v1.13": {"figure": 0.0, "flagged": False}}), "noise-flagByVersion"),
    ("flagByVersion-entry-missing-flagged", lambda n: n["flagByVersion"]["v1.11"].pop("flagged"), "noise-flagByVersion"),
    ("flagByVersion-entry-extra-key", lambda n: n["flagByVersion"]["v1.11"].update(value=0.06), "noise-flagByVersion"),
    ("flagByVersion-figure-above-1", lambda n: n["flagByVersion"]["v1.10"].update(figure=1.5), "noise-flagByVersion"),
    ("flagByVersion-figure-negative", lambda n: n["flagByVersion"]["v1.12"].update(figure=-0.01), "noise-flagByVersion"),
    ("flagByVersion-figure-string", lambda n: n["flagByVersion"]["v1.11"].update(figure="0.06"), "noise-flagByVersion"),
    ("flagByVersion-flagged-not-bool", lambda n: n["flagByVersion"]["v1.11"].update(flagged=1), "noise-flagByVersion"),
    ("flagByVersion-flagged-contradicts-v1.10", lambda n: n["flagByVersion"]["v1.10"].update(flagged=False), "noise-flagByVersion"),
    ("flagByVersion-flagged-contradicts-v1.12", lambda n: n["flagByVersion"]["v1.12"].update(flagged=True), "noise-flagByVersion"),
    ("flagByVersion-v1.10-not-e_all", lambda n: n["flagByVersion"]["v1.10"].update(figure=0.11), "noise-flagByVersion"),
    ("flagByVersion-v1.12-not-e_res", lambda n: n["flagByVersion"]["v1.12"].update(figure=0.03), "noise-flagByVersion"),
    ("flagByVersion-v1.11-above-v1.10", lambda n: n["flagByVersion"]["v1.11"].update(figure=0.12), "noise-flagByVersion"),
    ("flagByVersion-v1.11-below-v1.12", lambda n: n["flagByVersion"]["v1.11"].update(figure=0.03, flagged=False), "noise-flagByVersion"),
    ("openPointsS27-missing", lambda n: n.pop("openPointsS27"), "noise-keys"),
    ("openPointsS27-null", lambda n: n.update(openPointsS27=None), "noise-openPointsS27"),
    ("openPointsS27-empty-dict", lambda n: n.update(openPointsS27={}), "noise-openPointsS27"),
    ("openPointsS27-empty-list", lambda n: n.update(openPointsS27=[]), "noise-openPointsS27"),
    ("openPointsS27-string", lambda n: n.update(openPointsS27="none"), "noise-openPointsS27"),
    ("openPointsS27-no-classLineTargets", lambda n: n["openPointsS27"].pop("classLineTargets"), "noise-openPointsS27"),
    ("openPointsS27-categories-omitted", lambda n: n.update(openPointsS27={"classLineTargets": "none found: x"}), "noise-openPointsS27"),
    ("detail-missing", lambda n: n.pop("detail"), "noise-keys"),
]


@pytest.mark.parametrize("name,f,rule", NOISE_PLANTED, ids=[p[0] for p in NOISE_PLANTED])
def test_noise_planted_violation(name, f, rule):
    n, rows = mutate_noise(f)
    assert rule in nrules(n, rows)


def test_referenceError_wilson_may_be_null_but_not_malformed():
    rows = [good(1), good(2)]
    n = good_noise(rows)
    n["referenceError"]["wilson95"] = None
    assert vf.validate_noise(n, rows) == []
    n["referenceError"]["wilson95"] = [0.5]
    assert "noise-referenceError" in nrules(n, rows)


def nmsgs(noise, rows=None):
    return [m for _, m in vf.validate_noise(noise, rows)]


def _n(f, rows=None):
    n = good_noise(rows or [good(1), good(2)])
    f(n)
    return n


def test_noise_messages_pin_each_subrule():
    rows = [good(1), good(2)]
    cases = [
        (lambda n: n.update(N=0, comparable=0), "N=0", None),
        (lambda n: n["agreement"].update(comparable=1), "agreement.comparable", rows),
        (lambda n: n["agreement"].update(agreed=3, rate=1.5), "agreed <= comparable", rows),
        (lambda n: n["agreement"].update(wilson95=[0.9, 0.1]), "wilson95", rows),
        (lambda n: n["agreement"].update(wilson95=[0.0, 2.0]), "wilson95", rows),
        (lambda n: n["adjudicated"]["byWrong"].update(bogus=0), "vocabulary", rows),
        (lambda n: n["adjudicated"]["byErrorKind"].update(bogus=0), "vocabulary", rows),
        (lambda n: n["adjudicated"].update(byWrong={"label": 1, "neither": 2}, W=1), "must each sum", rows),
        (lambda n: n["adjudicated"].update(byErrorKind={"target": 1, "none": 2}), "must each sum", rows),
        (lambda n: n["audit"]["AI"].update(n=3), "k <= n <= N_s", rows),
        (lambda n: n["audit"]["AI"].update(k=1, n=0, N_s=0), "k <= n <= N_s", rows),
        (lambda n: n["handError"].update(e_all={"point": 0.01, "conservative": 0.2}), "cannot exceed", rows),
        (lambda n: n["closedClasses"].update(count=3), "distinct ordinals", rows),
        (lambda n: n["closedClasses"].update(count=1, byKind={"role": [9], "coordinate": [], "shape": []}), "outside 1..N", rows),
        (lambda n: n["closedClasses"]["byKind"].update(target=[1]), "byKind needs exactly", rows),
        (lambda n: n["census"].update(targetsRead=-1), "targetsRead must be", rows),
        (lambda n: n["census"].update(targetsRead="5"), "targetsRead must be", rows),
        (lambda n: n.update(unscored={"byReason": {"notACall": 0}, "underivable": 0}), "unscored needs", None),
        (lambda n: n.update(basisCounts={"agreed": 1}), "!= N", None),
        (lambda n: n.update(basisCounts={"agreed": 1, "bogus": 1}), "basisCounts keys", None),
        (lambda n: n.update(detail="x"), "detail", rows),
        (lambda n: n.update(handError=dict(n["handError"], noiseLimited=True)), "contradicts", rows),
        (lambda n: n.update(handError={"e_all": {"point": 0.5, "conservative": 0.6}, "e_res": {"point": 0.06, "conservative": 0.7}, "noiseLimited": False}), "contradicts", rows),
    ]
    for f, needle, r in cases:
        n = _n(f)
        assert any(needle in m for m in nmsgs(n, r)), (needle, nmsgs(n, r))


def test_underivable_count_must_match_final():
    rows = [good(1), mut(ordinal=2, siteId="2", scored=False, T=[], notScoredReason="underivable", external=["x"])]
    rows[1]["siteId"] = "0000000000000002"
    n = good_noise(rows)
    n["basisCounts"] = {"agreed": 2}
    n["unscored"] = {"byReason": {"underivable": 1}, "underivable": 1}
    assert vf.validate_noise(n, rows) == []
    n["unscored"] = {"byReason": {"underivable": 1}, "underivable": 0}
    assert any("unscored" in m for m in nmsgs(n, rows))


def test_openPointsS27_dict_without_classLineTargets_is_refused_even_with_three_entries():
    rows = [good(1), good(2)]
    n = good_noise(rows)
    n["openPointsS27"] = {"a": "none found", "b": "none found", "c": "none found"}
    assert any("classLineTargets" in m for m in nmsgs(n, rows))


def test_flagByVersion_messages_pin_each_subrule():
    rows = [good(1), good(2)]
    cases = [
        (lambda n: n["flagByVersion"]["v1.11"].update(figure=0.12), "v1.11.figure"),
        (lambda n: n["flagByVersion"]["v1.11"].update(figure=0.03, flagged=False), "v1.11.figure"),
        (lambda n: n["flagByVersion"]["v1.10"].update(figure=0.11), "e_all"),
        (lambda n: n["flagByVersion"]["v1.12"].update(figure=0.03), "e_res"),
        (lambda n: n["flagByVersion"]["v1.12"].update(flagged=True), "figure > 0.05"),
        (lambda n: n["flagByVersion"]["v1.11"].update(flagged=False), "figure > 0.05"),
        (lambda n: n["flagByVersion"]["v1.11"].update(figure=True), "number"),
        (lambda n: (n["handError"]["e_all"].update(point=1.5), n["flagByVersion"]["v1.10"].update(figure=1.5)), "in [0,1]"),
        (lambda n: (n["handError"]["e_res"].update(point=-0.01), n["flagByVersion"]["v1.12"].update(figure=-0.01)), "in [0,1]"),
    ]
    for f, needle in cases:
        n = _n(f)
        assert any(needle in m for m in nmsgs(n, rows)), (needle, nmsgs(n, rows))


def test_flagByVersion_boundary_0_05_is_not_flagged_and_chain_may_tie():
    rows = [good(1), good(2)]
    n = good_noise(rows)
    n["handError"]["e_all"]["point"] = n["handError"]["e_res"]["point"] = 0.05
    n["handError"]["noiseLimited"] = False
    n["flagByVersion"] = {v: {"figure": 0.05, "flagged": False} for v in ("v1.10", "v1.11", "v1.12")}
    assert vf.validate_noise(n, rows) == []


def test_openPointsS27_may_be_a_nonempty_list():
    rows = [good(1), good(2)]
    n = good_noise(rows)
    n["openPointsS27"] = [{"kind": "classLineTargets", "statement": "none found"}]
    assert vf.validate_noise(n, rows) == []


def test_census_null_allowed_when_no_closed_class():
    rows = [good(1), good(2)]
    n = good_noise(rows)
    n["census"]["targetsRead"] = None
    assert vf.validate_noise(n, rows) == []


def test_vocabulary_constants_are_single_source():
    assert vf.BASIS == ("agreed", "audit-confirmed", "adjudicated", "audit-overturned", "sweep-overturned", "label-only")
    assert vf.ROLES == ("impl", "decl")


def test_basis_constant_drives_validation(monkeypatch):
    monkeypatch.setattr(vf, "BASIS", vf.BASIS + ("audit-sourced",))
    assert vf.validate_record(mut(basis="audit-sourced")) == []


# ---- CLI ----------------------------------------------------------------------------------------

def write_lang(tmp_path, rows, noise):
    t, a = tmp_path / "truth", tmp_path / "adj"
    t.mkdir(), a.mkdir()
    (t / "x.final.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (a / "x.noise.json").write_text(json.dumps(noise))
    (a / "x.input.jsonl").write_text("".join(json.dumps({"ordinal": r["ordinal"], "siteId": r["siteId"]}) + "\n" for r in rows))
    return t, a


def run_cli(t, a):
    return subprocess.run([sys.executable, str(FF / "validate_final.py"), "--truth", str(t), "--adjudication", str(a), "x"],
                          capture_output=True, text=True)


def test_cli_exit_zero_when_clean(tmp_path):
    rows = [good(1), good(2)]
    t, a = write_lang(tmp_path, rows, good_noise(rows))
    p = run_cli(t, a)
    assert p.returncode == 0, p.stdout + p.stderr


def test_cli_exit_nonzero_and_names_file_and_ordinal(tmp_path):
    rows = [good(1), good(2)]
    rows[1]["basis"] = "nope"
    t, a = write_lang(tmp_path, rows, good_noise(rows))
    p = run_cli(t, a)
    assert p.returncode != 0
    assert "x.final.jsonl" in p.stdout and "ordinal 2" in p.stdout and "basis" in p.stdout


def test_cli_missing_file_is_a_violation(tmp_path):
    t, a = tmp_path / "t", tmp_path / "a"
    t.mkdir(), a.mkdir()
    assert run_cli(t, a).returncode != 0


def test_cli_unparseable_line_is_a_violation(tmp_path):
    rows = [good(1)]
    t, a = write_lang(tmp_path, rows, good_noise(rows))
    (t / "x.final.jsonl").write_text("{not json\n")
    p = run_cli(t, a)
    assert p.returncode != 0 and "line 1" in p.stdout
