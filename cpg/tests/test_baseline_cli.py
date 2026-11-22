"""The generator CLI of ``baseline.py`` (analyst gate pass 2 B1): ``bound`` stamped from the gate record, ``in_scope`` from the exclusions,
the ``noise-baseline*.jsonl`` / ``noise-recall*.json`` files, and the round trip through ``noise.py score`` (exit 0)."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval" / "tools"))
import baseline as B  # noqa: E402
import baseline_helpers as H  # noqa: E402
import noise as N  # noqa: E402
import test_noise as TN  # noqa: E402  (the answers/labels/score helpers of the consumer's own tests)

ANCHORS = [{"anchorId": "SECRET0", "type": "symbol", "symbolId": "A"}, {"anchorId": "SECRET1", "type": "contract", "cpId": "CP1"},
           {"anchorId": "SECRET2", "type": "config", "entryId": "CE1"}, {"anchorId": "SECRET3", "type": "unresolved"}]
EXCLUSIONS = TN.EXCL + ["*.ts"]


def gold(anchor, path, qname, lines, hit, key_error=False):
    return {"anchorId": anchor, "impacted": {"type": "symbol", "repoId": "r1", "path": path, "qname": qname}, "lines": lines,
            "hit": hit, "keyError": key_error}


HITS = [gold("SECRET0", "e.py", "e.py::E", [2], True), gold("SECRET0", "b.py", "b.py::B", [2], False),
        gold("SECRET0", "zz.py", "zz.py::Z", [2], True, key_error=True), gold("SECRET0", "p2.py", "p2.py::P2", [2], True),
        gold("SECRET1", "r.py", "r.py::R", [2], True), gold("SECRET2", "p2.py", "p2.py::P2", [2], False),
        {"anchorId": "SECRET2", "impacted": {"type": "config", "repoId": "r1", "path": "cfg.yml", "keyPath": "a.b"}, "lines": [1],
         "hit": True, "keyError": False},
        gold("SECRET3", "d.py", "d.py::D", [2], False)]


def _w(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def setup(tmp_path, record=None, hits=HITS):
    cust = TN._draw(tmp_path, TN._bounded(TN._scoring_answers()))
    TN._label_all(cust)
    (tmp_path / "e.json").write_text(json.dumps(EXCLUSIONS))
    H.write_snapshot(cust / "snapshot-adjacency.jsonl")
    _w(cust / "system-hits.jsonl", hits)
    _w(tmp_path / "anchors.jsonl", ANCHORS)
    (tmp_path / "gate.json").write_text(json.dumps({"defaultDepthBound": 2} if record is None else record))
    return cust


def gen(tmp_path, cust, **over):
    args = {"--anchors": tmp_path / "anchors.jsonl", "--exclusions": tmp_path / "e.json", "--gate-record": tmp_path / "gate.json",
            "--gate-id": "g", "--custodian": cust, **over}
    return B.main([x for k, v in args.items() for x in (k, str(v))])


def rows(cust, name="noise-baseline.jsonl"):
    return {r["anchorId"]: r for r in (json.loads(l) for l in (cust / name).read_text().splitlines())}


def test_the_generated_files_round_trip_through_noise_score_with_exit_zero(tmp_path, capsys):
    cust = setup(tmp_path)
    assert gen(tmp_path, cust) == 0
    capsys.readouterr()
    assert TN._score(tmp_path, cust, ["--gate-record", str(tmp_path / "gate.json")]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["baseline"]["anchors"] == 4 and rep["baseline"]["unresolved"] == 1 and "keyScoped" in rep
    assert rep["recall"]["system"]["n"] == len(HITS) and rep["recall"]["paired"]["both"] >= 1
    assert rep["keyScoped"]["recall"]["system"] == rep["recall"]["system"]           # same items, same system figure


def test_bound_is_stamped_from_the_record_on_every_row_including_the_unresolved_one(tmp_path):
    cust = setup(tmp_path, record={"defaultDepthBound": 3})
    assert gen(tmp_path, cust) == 0
    for name, want in (("noise-baseline.jsonl", 10), ("noise-baseline-key.jsonl", 8)):
        r = rows(cust, name)
        assert {x["bound"] for x in r.values()} == {3}
        assert r["SECRET0"]["count"] == want                    # bound 3 lists D too (the record's bound, not a default of 2)
    assert rows(cust)["SECRET3"] == {"anchorId": "SECRET3", "type": "unresolved", "count": 0, "countInScope": 0, "countConfirmedOwn": 0,
                                     "items": [], "unresolved": True, "bound": 3}      # note §4's shape: no byName


def test_in_scope_is_the_exclusion_filter_not_a_silent_default(tmp_path):
    cust = setup(tmp_path)
    assert gen(tmp_path, cust) == 0
    a = rows(cust)["SECRET0"]
    assert (a["count"], a["countInScope"]) == (9, 6)            # H P3 Q are .ts: excluded by "*.ts"
    (tmp_path / "e.json").write_text(json.dumps(["nothing/"]))
    assert gen(tmp_path, cust) == 0
    assert rows(cust)["SECRET0"]["countInScope"] == 9


def test_both_variants_are_written_and_the_key_variant_is_smaller_for_a_symbol_anchor(tmp_path):
    cust = setup(tmp_path)
    assert gen(tmp_path, cust) == 0
    assert rows(cust)["SECRET0"]["count"] == 9 and rows(cust, "noise-baseline-key.jsonl")["SECRET0"]["count"] == 7
    for name in ("noise-recall.json", "noise-recall-key.json"):
        r = json.loads((cust / name).read_text())
        assert r["itemSetSha256"] == N.item_set_sha256(HITS) and r["configGold"] == 1


def test_each_variant_recall_is_computed_from_its_own_rows(tmp_path):
    cust = setup(tmp_path)
    assert gen(tmp_path, cust) == 0
    kind = json.loads((cust / "noise-recall.json").read_text())
    key = json.loads((cust / "noise-recall-key.json").read_text())
    assert kind["system"] == key["system"] and kind["baseline"][1] == key["baseline"][1]
    assert key["baseline"][0] == kind["baseline"][0] - 2        # P2 under SECRET0 and under SECRET2 is a kind-wide member only: kind hit, key miss


def test_stdout_is_counts_only(tmp_path, capsys):
    cust = setup(tmp_path)
    capsys.readouterr()
    assert gen(tmp_path, cust) == 0
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert rep["bound"] == 2 and rep["kind"]["anchors"] == 4 and rep["kind"]["unresolved"] == 1
    for needle in ("SECRET", "e.py", "CP1", "A\""):
        assert needle not in out


def test_the_custodian_inside_the_repo_is_refused_and_nothing_is_written(tmp_path):
    cust = setup(tmp_path)
    inside = B.REPO / "cpg" / "never-created-baseline-test"
    assert gen(tmp_path, cust, **{"--custodian": inside, "--snapshot": cust / "snapshot-adjacency.jsonl",
                                  "--hits": cust / "system-hits.jsonl"}) == 2
    assert not inside.exists()


@pytest.mark.parametrize("record", [{}, {"defaultDepthBound": "2"}, {"defaultDepthBound": 2, "gateId": "other"}])
def test_a_bad_gate_record_is_exit_4_with_no_files(tmp_path, record):
    cust = setup(tmp_path, record=record)
    assert gen(tmp_path, cust) == 4
    assert not (cust / "noise-baseline.jsonl").exists() and not (cust / "noise-recall.json").exists()


def test_a_gold_anchor_missing_from_the_anchor_list_is_exit_4_with_no_files(tmp_path):
    cust = setup(tmp_path, hits=HITS + [gold("SECRET9", "e.py", "e.py::E", [2], True)])
    assert gen(tmp_path, cust) == 4
    assert not (cust / "noise-baseline.jsonl").exists()


def test_a_resolved_anchor_absent_from_the_snapshot_is_exit_4(tmp_path):
    cust = setup(tmp_path)
    _w(tmp_path / "anchors.jsonl", ANCHORS[:1] + [{"anchorId": "SECRET1", "type": "contract", "cpId": "CPX"}] + ANCHORS[2:])
    assert gen(tmp_path, cust) == 4


def test_exclusions_and_the_gate_are_required_arguments(tmp_path):
    cust = setup(tmp_path)
    with pytest.raises(SystemExit):
        B.main(["--anchors", str(tmp_path / "anchors.jsonl"), "--gate-record", str(tmp_path / "gate.json"), "--gate-id", "g",
                "--custodian", str(cust)])


def test_a_planted_snapshot_for_a_different_bound_is_caught_by_noise_score_not_silently_scored(tmp_path, capsys):
    cust = setup(tmp_path)
    assert gen(tmp_path, cust) == 0
    (tmp_path / "gate.json").write_text(json.dumps({"defaultDepthBound": 3}))         # the record moved after the files were generated
    capsys.readouterr()
    assert TN._score(tmp_path, cust, ["--gate-record", str(tmp_path / "gate.json")]) == 4
