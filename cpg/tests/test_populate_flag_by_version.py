"""populate_flag_by_version.py: fill the required top-level `flagByVersion` of an adjudication/<lang>.noise.json
from figures the custodians stored (never computed). Run: cd cpg && .venv/bin/python -m pytest -q tests/test_populate_flag_by_version.py
"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

FF = Path(os.environ.get("CCG_FF_DIR") or (Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"))
sys.path.insert(0, str(FF))
import populate_flag_by_version as pf  # noqa: E402


def doc(**detail):
    return {"lang": "x", "handError": {"e_all": {"point": 0.075, "conservative": 0.1}, "e_res": {"point": 0.0, "conservative": 0.01}},
            "basisCounts": {}, "openPointsS27": {"classLineTargets": "n"}, "detail": detail}


SRC = {"x": ("detail", "stored", "e_res")}


def test_figures_come_from_top_level_and_the_stored_v111_path():
    d = doc(stored={"e_res": 0.0417})
    out, changed = pf.populate(copy.deepcopy(d), SRC["x"])
    assert changed
    assert out["flagByVersion"] == {"v1.10": {"figure": 0.075, "flagged": True}, "v1.11": {"figure": 0.0417, "flagged": False},
                                    "v1.12": {"figure": 0.0, "flagged": False}}
    assert list(out).index("flagByVersion") == list(out).index("openPointsS27") - 1


def test_everything_else_is_unchanged():
    d = doc(stored={"e_res": 0.0417})
    out, _ = pf.populate(copy.deepcopy(d), SRC["x"])
    out.pop("flagByVersion")
    assert out == d


def test_missing_v111_figure_stops_and_names_the_path():
    with pytest.raises(pf.NoStoredFigure, match="detail.stored.e_res"):
        pf.populate(doc(other=1), SRC["x"])


def test_non_numeric_v111_figure_is_not_accepted():
    with pytest.raises(pf.NoStoredFigure):
        pf.populate(doc(stored={"e_res": "0.04"}), SRC["x"])
    with pytest.raises(pf.NoStoredFigure):
        pf.populate(doc(stored={"e_res": True}), SRC["x"])


def test_idempotent_and_existing_different_value_refused():
    once, _ = pf.populate(doc(stored={"e_res": 0.0417}), SRC["x"])
    twice, changed = pf.populate(copy.deepcopy(once), SRC["x"])
    assert not changed and twice == once
    once["flagByVersion"]["v1.11"]["figure"] = 0.3
    with pytest.raises(ValueError):
        pf.populate(once, SRC["x"])


def test_superseded_verdictByVersion_is_removed_only_when_asked():
    d = doc(stored={"e_res": 0.0417}, v112={"verdictByVersion": {"a": 1}, "keep": 2})
    kept, _ = pf.populate(copy.deepcopy(d), SRC["x"])
    assert "verdictByVersion" in kept["detail"]["v112"]
    out, changed = pf.populate(copy.deepcopy(d), SRC["x"], drop=("detail", "v112", "verdictByVersion"))
    assert changed and out["detail"]["v112"] == {"keep": 2}
    again, changed2 = pf.populate(copy.deepcopy(out), SRC["x"], drop=("detail", "v112", "verdictByVersion"))
    assert not changed2 and again == out


def test_source_table_covers_exactly_the_seven_languages_with_detail_paths():
    assert set(pf.V111_SOURCE) == {"python", "javascript", "go", "java", "csharp", "cpp", "kotlin"}
    assert all(p[0] == "detail" for p in pf.V111_SOURCE.values())


def test_file_roundtrip_keeps_format_and_second_run_is_a_noop(tmp_path):
    p = tmp_path / "x.noise.json"
    p.write_text(json.dumps(doc(stored={"e_res": 0.0417}), indent=1))
    assert pf.populate_file(p, SRC["x"]) is True
    text = p.read_text()
    assert not text.endswith("\n") and json.loads(text)["flagByVersion"]["v1.11"]["figure"] == 0.0417
    assert pf.populate_file(p, SRC["x"]) is False and p.read_text() == text


def test_unrecognised_formatting_is_refused_untouched(tmp_path):
    p = tmp_path / "x.noise.json"
    p.write_text(json.dumps(doc(stored={"e_res": 0.0417})))
    before = p.read_bytes()
    with pytest.raises(ValueError):
        pf.populate_file(p, SRC["x"])
    assert p.read_bytes() == before


def test_file_untouched_when_a_figure_is_missing(tmp_path):
    p = tmp_path / "x.noise.json"
    p.write_text(json.dumps(doc(), indent=1))
    before = p.read_bytes()
    with pytest.raises(pf.NoStoredFigure):
        pf.populate_file(p, SRC["x"])
    assert p.read_bytes() == before


def test_boundary_figure_0_05_is_not_flagged():
    d = doc(stored={"e_res": 0.05})
    d["handError"]["e_all"]["point"] = 0.05
    out, _ = pf.populate(d, SRC["x"])
    assert {v["flagged"] for v in out["flagByVersion"].values()} == {False}
    d = doc(stored={"e_res": 0.0501})
    assert pf.populate(d, SRC["x"])[0]["flagByVersion"]["v1.11"]["flagged"] is True


def test_the_per_language_v111_source_table_is_pinned():
    assert pf.V111_SOURCE == {
        "python": ("detail", "handError", "e_res"),
        "javascript": ("detail", "handErrorRate", "withoutRoleErrors"),
        "go": ("detail", "original_v1_11", "handErrorRate", "pointWithoutRoleErrors"),
        "java": ("detail", "original_v111", "handErrorRate", "withoutRoleErrors", "point"),
        "csharp": ("detail", "handErrorRate", "withoutRoleErrors", "point"),
        "cpp": ("detail", "v111Original", "handErrorRate", "e_res", "point"),
        "kotlin": ("detail", "handError", "v1.11_e_res_for_comparison"),
    }
    assert pf.DROP == {"cpp": ("detail", "v112", "verdictByVersion")}
