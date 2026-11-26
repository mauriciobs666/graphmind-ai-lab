"""The t3 carry-forward comparator (``eval/results/co9-t3/compare_listed.py``, ``noiseFix.probeRehearsal`` of the ``ac5-gate-1-r4`` record): it must read a planted
deletion, an added item, an edited item id, a changed chain, a changed field of an item (status, distance, reason, band, subject lines, any) and a changed group
structure as DIFFERENT, and identical input as identical; ``--run`` builds the answers from a ``run.json`` and fails on an entry without an answer."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parents[1]
HERE = CPG / "eval" / "results" / "co9-t3"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CPG / "eval" / "tools"))
import compare_listed as C  # noqa: E402


def item(unit, sym, chain, status="possible"):
    return {"unitId": unit, "subject": {"type": "symbol", "symbolId": sym, "path": "p.py", "qname": "p::f"}, "status": status, "reason": "r", "chain": chain}


def world():
    return [
        {"anchorId": "a1", "groups": [], "items": [item("u1", "s1", [{"id": "x", "kind": "Link"}]), item("u1", "s2", [{"id": "y", "kind": "Link"}])]},
        {"anchorId": "a2", "groups": [{"groupId": "g1", "members": [item("u2", "s3", [{"id": "z", "kind": "Link"}])]}], "items": [item("u2", "s4", [{"id": "w", "kind": "Link"}])]},
    ]


def write(path, answers):
    path.write_text("".join(json.dumps(a) + "\n" for a in answers))
    return path


@pytest.fixture
def base(tmp_path):
    answers = world()
    triples = C.triples(answers)
    baseline = tmp_path / "listed-triples.jsonl"
    C.write_baseline(baseline, triples)
    return answers, baseline, C.listed_set_sha256(triples), tmp_path


def run(base, answers):
    _, baseline, sha, tmp = base
    return C.compare(write(tmp / "answers.jsonl", answers), baseline, sha)


def test_group_members_are_listed_items(base):
    assert len([t for t in C.triples(base[0]) if t[0] != "group"]) == 4 and len([t for t in C.triples(base[0]) if t[0] == "group"]) == 1


def test_identical_input_passes(base):
    r = run(base, copy.deepcopy(base[0]))
    assert r["identical"] is True and r["added"] == r["removed"] == r["chainChanged"] == 0 and r["items"] == 4 and r["sha256Equal"] is True


def test_order_of_answers_and_items_is_irrelevant(base):
    a = copy.deepcopy(base[0]); a.reverse(); a[0]["items"].reverse()
    assert run(base, a)["identical"] is True


def test_planted_deletion_fails(base):
    a = copy.deepcopy(base[0]); a[0]["items"].pop()
    r = run(base, a)
    assert r["identical"] is False and r["removed"] == 1 and r["added"] == 0 and r["sha256Equal"] is False


def test_planted_deleted_group_member_fails(base):
    a = copy.deepcopy(base[0]); a[1]["groups"][0]["members"].clear()
    assert run(base, a)["identical"] is False


def test_planted_added_item_fails(base):
    a = copy.deepcopy(base[0]); a[0]["items"].append(item("u1", "s9", []))
    r = run(base, a)
    assert r["identical"] is False and r["added"] == 1 and r["removed"] == 0


def test_planted_edit_of_an_item_id_fails(base):
    a = copy.deepcopy(base[0]); a[0]["items"][0]["subject"]["symbolId"] = "s1x"
    r = run(base, a)
    assert r["identical"] is False and r["added"] == 1 and r["removed"] == 1


def test_planted_chain_change_fails(base):
    a = copy.deepcopy(base[0]); a[0]["items"][0]["chain"][0]["id"] = "x2"
    r = run(base, a)
    assert r["identical"] is False and r["chainChanged"] == 1 and r["added"] == r["removed"] == 0


@pytest.mark.parametrize("field,value", [("status", "confirmed"), ("distance", 9), ("reason", "other"), ("directness", "indirect"), ("viaData", True),
                                         ("band", "x"), ("split", "y"), ("unitId", "u1b")])
def test_a_planted_change_of_any_item_field_fails(base, field, value):
    a = copy.deepcopy(base[0]); a[0]["items"][0][field] = value
    r = run(base, a)
    assert r["identical"] is False and r["fieldsChanged"] + r["removed"] >= 1


def test_a_planted_change_of_a_subject_line_fails(base):
    a = copy.deepcopy(base[0]); a[0]["items"][1]["subject"]["path"] = "q.py"
    r = run(base, a)
    assert r["identical"] is False and r["fieldsChanged"] == 1 and r["added"] == r["removed"] == 0


def test_an_item_moved_into_a_group_fails(base):
    a = copy.deepcopy(base[0]); a[0]["groups"].append({"groupId": "g9", "members": [a[0]["items"].pop()]})
    r = run(base, a)
    assert r["identical"] is False and r["groupDiffs"] >= 1


def test_a_changed_group_header_fails(base):
    a = copy.deepcopy(base[0]); a[1]["groups"][0]["groupId"] = "g-other"
    assert run(base, a)["identical"] is False


def test_a_member_moved_between_groups_fails(base):
    a = copy.deepcopy(base[0]); a[1]["groups"].append({"groupId": "g2", "members": [a[1]["groups"][0]["members"].pop()]})
    assert run(base, a)["identical"] is False


def test_group_member_lists_are_part_of_the_set(tmp_path):
    m1, m2 = item("u3", "s5", []), item("u3", "s6", [])
    ans = [{"anchorId": "a3", "items": [], "groups": [{"groupId": "g", "members": [m1]}, {"groupId": "g", "members": [m2]}]}]
    reordered = [{"anchorId": "a3", "items": [], "groups": [{"groupId": "g", "members": [m2]}, {"groupId": "g", "members": [m1]}]}]
    assert C.listed_set_sha256(C.triples(ans)) == C.listed_set_sha256(C.triples(reordered))   # the same two groups in the other order are the same set
    three = [{"anchorId": "a3", "items": [], "groups": [{"groupId": "g", "members": [m1, m2]}, {"groupId": "g", "members": []}]}]
    assert C.listed_set_sha256(C.triples(ans)) != C.listed_set_sha256(C.triples(three))


def test_an_empty_group_added_fails(base):
    a = copy.deepcopy(base[0]); a[0]["groups"].append({"groupId": "g3", "members": []})
    assert run(base, a)["identical"] is False


def test_a_duplicate_pair_fails(base):
    a = copy.deepcopy(base[0]); a[0]["items"].append(copy.deepcopy(a[0]["items"][0]))
    assert run(base, a)["identical"] is False


def test_a_baseline_that_is_not_the_pinned_one_fails(base):
    answers, baseline, sha, tmp = base
    r = C.compare(write(tmp / "answers.jsonl", answers), baseline, "0" * 64)
    assert r["identical"] is False and r["baselineMatchesPinned"] is False


def test_a_tampered_baseline_fails_even_for_the_matching_answers(base):
    answers, baseline, sha, tmp = base
    lines = baseline.read_text().splitlines()
    baseline.write_text("\n".join(lines[:-1]) + "\n")
    r = C.compare(write(tmp / "answers.jsonl", answers), baseline, sha)
    assert r["identical"] is False and r["baselineMatchesPinned"] is False


def test_cli_exit_status_and_counts_only_output(base):
    answers, baseline, sha, tmp = base
    f = write(tmp / "answers.jsonl", answers)
    ok = subprocess.run([sys.executable, str(HERE / "compare_listed.py"), str(f), "--baseline", str(baseline), "--pinned-sha", sha], capture_output=True, text=True)
    assert ok.returncode == 0 and set(json.loads(ok.stdout)) >= {"identical", "items", "added", "removed", "chainChanged"}
    a = copy.deepcopy(answers); a[0]["items"].pop()
    bad = subprocess.run([sys.executable, str(HERE / "compare_listed.py"), str(write(tmp / "b.jsonl", a)), "--baseline", str(baseline), "--pinned-sha", sha], capture_output=True, text=True)
    assert bad.returncode == 1 and "s1" not in bad.stdout and "a1" not in bad.stdout


def run_json(answers, extra=None):
    doc = {"answers": {a["anchorId"]: {"answer": {"items": a["items"], "groups": a["groups"]}} for a in answers}}
    doc["answers"].update(extra or {})
    return doc


def test_run_mode_builds_the_answers_from_run_json(base):
    answers, baseline, sha, tmp = base
    f = tmp / "run.json"; f.write_text(json.dumps(run_json(answers)))
    assert C.compare(f, baseline, sha, run=True)["identical"] is True
    out = subprocess.run([sys.executable, str(HERE / "compare_listed.py"), "--run", str(f), "--baseline", str(baseline), "--pinned-sha", sha], capture_output=True, text=True)
    assert out.returncode == 0


def test_run_mode_defaults_missing_groups_to_empty(base):
    answers, baseline, sha, tmp = base
    doc = run_json([a for a in answers if a["anchorId"] == "a1"])
    doc["answers"]["a1"]["answer"].pop("groups")
    f = tmp / "run.json"; f.write_text(json.dumps(doc))
    assert len(C.answers_from_run(f)) == 1 and C.answers_from_run(f)[0]["groups"] == []


def test_run_mode_fails_on_an_entry_without_an_answer(base):
    answers, baseline, sha, tmp = base
    f = tmp / "run.json"; f.write_text(json.dumps(run_json(answers, {"a3": {"harnessError": "boom"}})))
    with pytest.raises(ValueError, match="1 entr"):
        C.compare(f, baseline, sha, run=True)
    out = subprocess.run([sys.executable, str(HERE / "compare_listed.py"), "--run", str(f), "--baseline", str(baseline), "--pinned-sha", sha], capture_output=True, text=True)
    assert out.returncode != 0 and "boom" not in out.stdout + out.stderr


def test_the_committed_baseline_is_the_one_the_record_pins():
    lines = (CPG / "eval" / "preregistration.jsonl").read_text().splitlines()
    rec = [json.loads(ln) for ln in lines if '"stratum":"ac5-gate-1-r4"' in ln]
    if not rec:
        pytest.skip("no ac5-gate-1-r4 record yet (commit 1 of the two-commit order); the post-R4 run must show 0 skipped")
    assert len(rec) == 1
    pin = rec[0]["noiseFix"]["t3ListedSet"]
    triples = C.load_baseline(HERE / "listed-triples.jsonl")
    n_groups = len([t for t in triples if t[0] == "group"])
    assert (len(triples) - n_groups, n_groups) == (pin["count"], pin["groups"])
    assert C.listed_set_sha256(triples) == pin["sha256"] and C.pinned_sha() == pin["sha256"]
