"""normalize_final.py: shape-only converter from the seven S57 final/noise shapes to the canonical one
(validate_final.py docstring). Run: cd cpg && .venv/bin/python -m pytest -q tests/test_normalize_final.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

FF = Path(os.environ.get("CCG_FF_DIR") or (Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"))
sys.path.insert(0, str(FF))
import normalize_final as nf  # noqa: E402
import validate_final as vf  # noqa: E402

REAL = Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"

RAW = {
    "ordinal": 1, "siteId": "ab", "site": "a.py:3", "T": [], "N": [], "external": ["x"], "notACall": False,
    "dynamic": False, "scored": True, "notScoredReason": None, "basis": "agreed", "strata": ["AE"],
    "evidence": "a.py:3 ok",
}


def norm(**kw):
    r = dict(RAW)
    r.update(kw)
    return nf.normalize_row(r, site_lookup={"ab": "a.py:3"})


def test_canonical_row_is_a_fixed_point():
    assert norm() == RAW and list(norm()) == list(vf.FINAL_KEYS)


@pytest.mark.parametrize("raw,want", [
    (None, []), ("react", ["react"]), (["a", "b"], ["a", "b"]),
    ([{"kind": "external", "package": "builtin"}], ["builtin"]),
])
def test_external_shapes(raw, want):
    assert norm(external=raw)["external"] == want


def test_external_dict_with_other_kind_is_refused():
    with pytest.raises(ValueError):
        norm(external=[{"kind": "local", "package": "x"}])


def test_site_dict_becomes_path_line():
    assert norm(site={"path": "a.py", "line": 3, "column": 1, "callText": "f"})["site"] == "a.py:3"


def test_missing_site_comes_from_lookup():
    r = {k: v for k, v in RAW.items() if k != "site"}
    assert nf.normalize_row(r, site_lookup={"ab": "q.java:9"})["site"] == "q.java:9"


def test_site_conflicting_with_lookup_is_refused():
    with pytest.raises(ValueError):
        nf.normalize_row(dict(RAW, site="other.py:1"), site_lookup={"ab": "a.py:3"})


@pytest.mark.parametrize("key,val", [("citation", "a.py:3"), ("citations", ["a.py:3", "a.py:3", "b.py:4"]), ("evidence", ["a.py:3", "b.py:4"])])
def test_evidence_spellings(key, val):
    r = {k: v for k, v in RAW.items() if k != "evidence"}
    r[key] = val
    out = nf.normalize_row(r, site_lookup={})
    assert out["evidence"] == (val if isinstance(val, str) else "; ".join(dict.fromkeys(val)))
    assert "citation" not in out and "citations" not in out


def test_two_evidence_spellings_at_once_is_refused():
    with pytest.raises(ValueError):
        norm(citation="a.py:3")


def test_compound_strata_are_split():
    assert norm(strata=["audit-U:AS", "x"])["strata"] == ["audit-U", "AS", "x"]


def test_T_extras_survive():
    t = {"path": "p", "qname": None, "defLine": 2, "role": "impl", "qnameDerived": True}
    assert norm(T=[t])["T"] == [t]


def test_label_set_covers_every_decision_field_and_ignores_shape():
    a = nf.label_set(dict(RAW, external="x"), site="a.py:3")
    b = nf.label_set(dict(RAW, external=[{"kind": "external", "package": "x"}]), site="a.py:3")
    assert a == b
    assert a != nf.label_set(dict(RAW, external=["y"]), site="a.py:3")
    assert a != nf.label_set(dict(RAW, notACall=True), site="a.py:3")
    assert a != nf.label_set(dict(RAW, basis="adjudicated"), site="a.py:3")
    t = {"path": "p", "qname": "q", "defLine": 2, "role": "impl"}
    assert nf.label_set(dict(RAW, T=[t]), site="a.py:3") != nf.label_set(dict(RAW, T=[dict(t, role="decl")]), site="a.py:3")


def test_label_set_mismatch_is_detected():
    with pytest.raises(AssertionError):
        nf.assert_labels_equal([dict(RAW, siteId="ab")], [dict(norm(), external=["z"])], {"ab": "a.py:3"})


# ---- noise ----------------------------------------------------------------------------------------

def test_adjudicated_combo_keys_are_split():
    a = nf.adjudicated_from_combos(73, 18, {"both|coordinate": 8, "label|target": 10, "neither|none": 55})
    assert a == {"n": 73, "W": 18, "byWrong": {"both": 8, "label": 10, "neither": 55},
                 "byErrorKind": {"coordinate": 8, "target": 10, "none": 55}}


def test_schema_errorKind_is_renamed_shape():
    assert nf.errorkinds({"target": 1, "schema": 3}) == {"target": 1, "shape": 3}


def test_noise_envelope_for_unknown_language_is_refused():
    with pytest.raises(KeyError):
        nf.normalize_noise("cobol", {}, [])


TOOLS = REAL / "adjudication" / "tools"
# pre-canonical (v1.11) snapshots the custodians saved; the live truth/ and adjudication/ files have since changed shape
SNAP = {
    "go": (TOOLS / "go/orig/go.final.v1.11.jsonl", TOOLS / "go/orig/go.noise.v1.11.json"),
    "java": (TOOLS / "java/v111/java.final.v111.jsonl", TOOLS / "java/v111/java.noise.v111.json"),
    "cpp": (TOOLS / "cpp/cpp.final.v111.jsonl", TOOLS / "cpp/cpp.noise.v111.json"),
}


def snap_dirs(lang, tmp_path):
    """tmp truth/ and adjudication/ dirs holding the v1.11 snapshot of `lang` (input.jsonl and sample are shape-stable)."""
    final, noise = SNAP[lang]
    if not (final.exists() and noise.exists()):
        pytest.skip("v1.11 snapshot absent")
    t, a = tmp_path / "truth", tmp_path / "adj"
    t.mkdir(parents=True), a.mkdir(parents=True)
    (t / f"{lang}.final.jsonl").write_bytes(final.read_bytes())
    (a / f"{lang}.noise.json").write_bytes(noise.read_bytes())
    (a / f"{lang}.input.jsonl").write_bytes((REAL / "adjudication" / f"{lang}.input.jsonl").read_bytes())
    return t, a


@pytest.mark.parametrize("lang", sorted(SNAP))
def test_v111_snapshots_normalize_to_valid_canonical_with_equal_labels(lang, tmp_path):
    t, a = snap_dirs(lang, tmp_path)
    nf.normalize_lang(lang, t, a, REAL / "sample", tmp_path / "out")
    # v1.11 sources carry no openPointsS27 (a v1.12 key): that missing key is the only thing left to report
    left = vf.validate_lang(lang, tmp_path / "out", tmp_path / "out", a)
    assert len(left) == 1 and "noise-keys" in left[0] and "openPointsS27" in left[0] and "flagByVersion" in left[0], left


def test_nested_openPointsS27_is_carried_to_the_envelope_top_level(tmp_path):
    t, a = snap_dirs("go", tmp_path)
    noise = json.loads((a / "go.noise.json").read_text())
    noise["detail_note"] = {"v1_12": {"openPointsS27": {"classLineTargets": "none found"}}}
    rows = nf.normalize_lang("go", t, a, REAL / "sample", tmp_path / "o1")
    out = nf.normalize_noise("go", noise, rows)
    assert out["openPointsS27"] == {"classLineTargets": "none found"} and list(out)[-2:] == ["openPointsS27", "detail"]
    assert out["lang"] == "go" and out["N"] == len(rows)
    assert vf.validate_noise(out, rows) == [] or all("openPointsS27" not in m for _, m in vf.validate_noise(out, rows))


@pytest.mark.parametrize("lang", vf.LANGS)
def test_live_files_validate(lang):
    """Validator-only on the live data: depends on the files being canonical, never on the normalizer."""
    if not (REAL / "truth" / f"{lang}.final.jsonl").exists():
        pytest.skip("S57 outputs absent")
    assert vf.validate_lang(lang, REAL / "truth", REAL / "adjudication") == []


def test_normalizer_never_writes_into_its_sources(tmp_path):
    t, a = snap_dirs("go", tmp_path)
    before = (t / "go.final.jsonl").read_bytes()
    with pytest.raises(ValueError):
        nf.normalize_lang("go", t, a, REAL / "sample", t)
    with pytest.raises(ValueError):
        nf.normalize_lang("go", t, a, REAL / "sample", a)
    assert (t / "go.final.jsonl").read_bytes() == before and not (a / "go.final.jsonl").exists()


def test_noise_whose_unscored_counts_disagree_with_final_is_refused(tmp_path):
    t, a = snap_dirs("cpp", tmp_path)
    rows = nf.normalize_lang("cpp", t, a, REAL / "sample", tmp_path / "out")
    noise = json.loads((a / "cpp.noise.json").read_text())
    nf.normalize_noise("cpp", noise, rows)  # consistent: fine
    tampered = [dict(r) for r in rows]
    next(r for r in tampered if r["notScoredReason"] == "notACall")["notScoredReason"] = "dynamic"
    with pytest.raises(ValueError):
        nf.normalize_noise("cpp", noise, tampered)


def test_evidence_list_with_non_string_is_refused():
    r = {k: v for k, v in RAW.items() if k != "evidence"}
    r["evidence"] = ["a.py:3", 7]
    with pytest.raises(ValueError):
        nf.normalize_row(r, {})


def test_strata_split_dedupes_in_first_seen_order():
    assert norm(strata=["audit-U:AI", "AI"])["strata"] == ["audit-U", "AI"]


def test_label_set_sees_N_and_assert_labels_equal_sees_length():
    assert nf.label_set(dict(RAW, N=["x"]), "a.py:3") != nf.label_set(RAW, "a.py:3")
    with pytest.raises(AssertionError):
        nf.assert_labels_equal([RAW, RAW], [norm()], {})


def test_noise_checks_against_final_and_closed_class_claims(tmp_path):
    gt, ga = snap_dirs("go", tmp_path / "go")
    rows = nf.normalize_lang("go", gt, ga, REAL / "sample", tmp_path / "go_out")
    noise = json.loads((ga / "go.noise.json").read_text())
    assert nf.normalize_noise("go", noise, rows)["detail"] == noise  # original kept verbatim
    claimed = json.loads(json.dumps(noise))
    claimed["role"]["labelErrors"] = 1
    with pytest.raises(ValueError):
        nf.normalize_noise("go", claimed, rows)
    # java's noise.json carries basisCounts, so the cross-check against the final file is exercised there
    jt, ja = snap_dirs("java", tmp_path / "java")
    jrows = nf.normalize_lang("java", jt, ja, REAL / "sample", tmp_path / "java_out")
    jnoise = json.loads((ja / "java.noise.json").read_text())
    bad = [dict(r) for r in jrows]
    next(r for r in bad if r["basis"] == "agreed")["basis"] = "adjudicated"
    with pytest.raises(ValueError):
        nf.normalize_noise("java", jnoise, bad)


def test_two_openPointsS27_in_the_source_are_refused(tmp_path):
    t, a = snap_dirs("go", tmp_path)
    rows = nf.normalize_lang("go", t, a, REAL / "sample", tmp_path / "o")
    noise = json.loads((a / "go.noise.json").read_text())
    noise["x"] = {"openPointsS27": {"classLineTargets": "a"}}
    noise["y"] = {"openPointsS27": {"classLineTargets": "b"}}
    with pytest.raises(ValueError):
        nf.normalize_noise("go", noise, rows)


def test_already_canonical_input_is_refused_with_a_message(tmp_path):
    t, a = snap_dirs("go", tmp_path)
    out = tmp_path / "out"
    nf.normalize_lang("go", t, a, REAL / "sample", out)
    (out / "go.input.jsonl").write_bytes((a / "go.input.jsonl").read_bytes())
    with pytest.raises(nf.AlreadyCanonical, match="already canonical"):
        nf.normalize_lang("go", out, out, REAL / "sample", tmp_path / "out2")


def test_cli_requires_out_and_refuses_canonical_cleanly(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        nf.main(["go"])
    assert e.value.code == 2 and "--out" in capsys.readouterr().err
    assert not hasattr(nf, "DEFAULT_OUT")
    t, a = snap_dirs("go", tmp_path)
    nf.normalize_lang("go", t, a, REAL / "sample", tmp_path / "out")
    (tmp_path / "out" / "go.input.jsonl").write_bytes((a / "go.input.jsonl").read_bytes())
    rc = nf.main(["go", "--out", str(tmp_path / "o2"), "--truth", str(tmp_path / "out"), "--adjudication", str(tmp_path / "out"),
                  "--sample", str(REAL / "sample")])
    assert rc == 2 and "already canonical" in capsys.readouterr().err


def test_no_hard_coded_tmp_path_in_the_tooling():
    for name in ("normalize_final.py", "validate_final.py", "hoist_open_points.py", "populate_flag_by_version.py"):
        assert "/tmp" not in (FF / name).read_text(), name
