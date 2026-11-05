"""hoist_open_points.py: move the nested `openPointsS27` of an adjudication/<lang>.noise.json to top level
(v1.12 §5.4 step 7 makes it a required top-level key). Run: cd cpg && .venv/bin/python -m pytest -q tests/test_hoist_open_points.py
"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

FF = Path(os.environ.get("CCG_FF_DIR") or (Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"))
sys.path.insert(0, str(FF))
import hoist_open_points as hop  # noqa: E402

OP = {"classLineTargets": "none found", "anonymousLambdaIndices": [], "externalsAbsentFromTree": []}


def noise(detail):
    return {"lang": "x", "N": 1, "basisCounts": {"agreed": 1}, "detail": detail}


@pytest.mark.parametrize("detail,path", [
    ({"openPointsS27": OP, "a": 1}, ("detail",)),
    ({"v112": {"openPointsS27": OP, "b": 2}, "a": 1}, ("detail", "v112")),
    ({"v1_12": {"openPointsS27": OP}, "a": 1}, ("detail", "v1_12")),
])
def test_nested_key_is_moved_to_top_level_and_nothing_else_changes(detail, path):
    before = noise(copy.deepcopy(detail))
    after, changed = hop.hoist(copy.deepcopy(before))
    assert changed and after["openPointsS27"] == OP
    expect = copy.deepcopy(before)
    node = expect
    for k in path:
        node = node[k]
    del node["openPointsS27"]
    expect["openPointsS27"] = OP
    assert after == expect
    assert list(after).index("openPointsS27") == list(after).index("detail") - 1  # placed just before detail


def test_idempotent_and_noop_when_already_top_level():
    top = noise({"a": 1})
    top["openPointsS27"] = OP
    again, changed = hop.hoist(copy.deepcopy(top))
    assert not changed and again == top
    once, _ = hop.hoist(noise({"v112": {"openPointsS27": OP}}))
    twice, changed2 = hop.hoist(copy.deepcopy(once))
    assert not changed2 and twice == once


def test_missing_everywhere_is_refused():
    with pytest.raises(ValueError):
        hop.hoist(noise({"a": 1}))


def test_two_nested_copies_are_refused():
    with pytest.raises(ValueError):
        hop.hoist(noise({"openPointsS27": OP, "v112": {"openPointsS27": OP}}))


def test_top_level_and_nested_both_present_is_refused():
    n = noise({"openPointsS27": OP})
    n["openPointsS27"] = OP
    with pytest.raises(ValueError):
        hop.hoist(n)


def test_similar_but_different_key_is_not_touched():
    with pytest.raises(ValueError):
        hop.hoist(noise({"original_v111": {"openPoints": [1]}}))


def test_file_roundtrip_preserves_format_and_is_idempotent(tmp_path):
    p = tmp_path / "x.noise.json"
    p.write_text(json.dumps(noise({"v112": {"openPointsS27": OP}, "é": "ü"}), indent=1))  # the custodians' format
    assert hop.hoist_file(p) is True
    text = p.read_text()
    assert text.endswith("}") and not text.endswith("\n") and "\\u00e9" in text and json.loads(text)["openPointsS27"] == OP
    assert hop.hoist_file(p) is False and p.read_text() == text


def test_file_with_unrecognised_formatting_is_refused_untouched(tmp_path):
    p = tmp_path / "x.noise.json"
    p.write_text(json.dumps(noise({"openPointsS27": OP})))  # compact, not the indent=1 the custodians wrote
    before = p.read_bytes()
    with pytest.raises(ValueError):
        hop.hoist_file(p)
    assert p.read_bytes() == before


def test_key_nested_inside_a_list_is_found():
    after, changed = hop.hoist(noise({"history": [{"openPointsS27": OP}]}))
    assert changed and after["openPointsS27"] == OP and after["detail"] == {"history": [{}]}
