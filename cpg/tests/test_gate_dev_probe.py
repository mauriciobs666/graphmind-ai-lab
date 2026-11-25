"""CO-5 / P6-1: the dev-gold probe is an ATTESTATION bound to the run it vouches for, not a typed file (``docs/plans/connected-code-graph-p1-closeout.md`` §3.5,
``docs/reviews/connected-code-graph-p1-32b-impl.md`` Pass 6). The gate computes the dev items per stratum from the committed ``cpg/eval/dev/ac5-<s>.jsonl`` itself,
compares the freeze (record) commit, the harness sha map, the dev gold hashes, the snapshot id the file carries, checks every key-error id
against the committed stakeholder-confirmation record, and pins the file's sha256 in the marker. One plant per field, each written red first. Offline, synthetic."""
import json

import pytest

import gate_helpers as H
from gate_helpers import G, dev_probe_doc
from gate_run_helpers import Env, GATE, OBS, many_world

import gate_run as R

KE_RECORD = "cpg/eval/decisions/ccg-dev-key-errors-v1.json"


def repo_of(tmp_path):
    e = Env(tmp_path)
    return e


def probe(e, **kw):
    return dev_probe_doc(e.repo, **kw)


def check(e, doc, gated=G.STRATA_GATED):
    G.dev_probe_check(doc, e.repo, H.head(e.repo), gated)


def ke(i, **over):
    return {"itemId": i, "confirmedBy": "user", "date": "2026-11-24", **over}


def confirm(repo, ids):
    (repo / KE_RECORD).write_text(json.dumps({"schema": "ccg-dev-key-errors-v1", "confirmed": [ke(i) for i in ids]}) + "\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "confirm")


def refuses(e, doc, match, gated=G.STRATA_GATED):
    with pytest.raises(G.PrecheckRefusal, match=match):
        G.dev_probe_check(doc, e.repo, H.head(e.repo), gated)


# ------------------------------------------------------------------------------------------------ what passes
def test_a_probe_built_from_the_repos_own_files_passes(tmp_path):
    e = repo_of(tmp_path)
    check(e, probe(e))


def test_the_dev_items_come_from_the_committed_gold_not_from_the_probe(tmp_path):
    """A forged probe that names one hit per stratum is not enough: the gold has more items than it lists."""
    e = repo_of(tmp_path)
    doc = probe(e)
    for s in ("http", "mcp", "scripts"):
        doc["byStratum"][s]["hitIds"] = doc["byStratum"][s]["hitIds"][:1]
    refuses(e, doc, "http")


def test_a_withheld_stratum_is_not_read_and_flow_is_read_only_when_gated(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e, strata=("http", "mcp", "scripts"))
    check(e, doc)
    refuses(e, doc, "flow", G.gated_strata(True))
    check(e, probe(e, strata=("http", "mcp", "scripts", "flow")), G.gated_strata(True))


# ------------------------------------------------------------------------------------------------ one plant per binding field
def test_a_complete_document_of_another_schema_version_is_refused(tmp_path):
    e = repo_of(tmp_path)
    refuses(e, probe(e, schema="ccg-dev-probe-v0"), "dev-gold probe")


@pytest.mark.parametrize("bad", [None, [], "x", {}, {"schema": "ccg-dev-probe-v0"}])
def test_a_missing_or_foreign_document_is_a_refusal(tmp_path, bad):
    refuses(repo_of(tmp_path), bad, "dev-gold probe")


def test_the_old_shape_with_bare_counts_is_not_accepted(tmp_path):
    e = repo_of(tmp_path)
    old = {"schema": G.DEV_PROBE_SCHEMA, "byStratum": {s: {"items": 1, "hits": 1, "keyErrors": 0} for s in ("http", "mcp", "scripts")}}
    refuses(e, old, "dev-gold probe")


def test_a_probe_of_another_commit_is_refused(tmp_path):
    e = repo_of(tmp_path)
    refuses(e, probe(e, recordCommit="f" * 40), "record commit")


def test_a_probe_with_no_commit_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    del doc["recordCommit"]
    refuses(e, doc, "record commit")


def test_one_changed_harness_sha_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    k = sorted(doc["harnessSha256"])[0]
    doc["harnessSha256"][k] = "0" * 64
    refuses(e, doc, "harness")


def test_a_harness_file_missing_from_or_added_to_the_probe_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    del doc["harnessSha256"][sorted(doc["harnessSha256"])[0]]
    refuses(e, doc, "harness")
    doc = probe(e)
    doc["harnessSha256"]["cpg/eval/tools/extra.py"] = "1" * 64
    refuses(e, doc, "harness")


def test_a_harness_file_edited_after_the_probe_was_written_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    (e.repo / "cpg/eval/tools/gate.py").write_text("# edited after the rehearsal\n")
    refuses(e, doc, "harness")


def test_a_dev_gold_file_changed_after_the_probe_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    f = e.repo / "cpg/eval/dev/ac5-scripts.jsonl"
    f.write_text(f.read_text() + json.dumps({"itemId": "added".ljust(32, "0"), "derivedStratum": "scripts"}) + "\n")
    refuses(e, doc, "dev gold")


def test_a_dev_item_added_with_the_hash_updated_still_lacks_a_hit(tmp_path):
    """A forger who recomputes the gold hash still cannot hit an item he did not list."""
    e = repo_of(tmp_path)
    f = e.repo / "cpg/eval/dev/ac5-scripts.jsonl"
    doc = probe(e)
    f.write_text(f.read_text() + json.dumps({"itemId": "added".ljust(32, "0"), "derivedStratum": "scripts"}) + "\n")
    doc["devGoldSha256"]["scripts"] = H.sha_of(f)
    refuses(e, doc, "scripts")


def test_a_gold_hash_missing_for_a_gated_stratum_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    del doc["devGoldSha256"]["mcp"]
    refuses(e, doc, "dev gold")


def test_a_stray_rehearsal_id_of_any_value_is_ignored_not_required_or_validated(tmp_path):
    """P7-2: the id bound nothing and was dropped; a probe written before that, carrying a garbage id, still passes and one without it passes too."""
    e = repo_of(tmp_path)
    check(e, probe(e))
    check(e, probe(e, rehearsalId="not hex"))


@pytest.mark.parametrize("value", [None, "", 7])
def test_the_snapshot_id_must_be_present(tmp_path, value):
    e = repo_of(tmp_path)
    refuses(e, probe(e, snapshotId=value), "snapshot")


# ------------------------------------------------------------------------------------------------ hits and key-errors, by id
def test_a_dev_miss_blocks_in_any_gated_stratum(tmp_path):
    e = repo_of(tmp_path)
    for s in ("http", "mcp", "scripts"):
        doc = probe(e)
        doc["byStratum"][s]["hitIds"].pop()
        refuses(e, doc, s)


def test_a_hit_id_that_is_not_a_dev_item_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    doc["byStratum"]["http"]["hitIds"][0] = "not-an-item".ljust(32, "0")
    refuses(e, doc, "http")


def test_a_duplicated_id_cannot_stand_for_a_missing_one(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    ids = doc["byStratum"]["mcp"]["hitIds"]
    ids[-1] = ids[0]
    refuses(e, doc, "mcp")


def test_an_id_listed_twice_in_addition_to_a_complete_list_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    doc["byStratum"]["scripts"]["hitIds"].append(doc["byStratum"]["scripts"]["hitIds"][0])
    refuses(e, doc, "scripts")


def test_an_id_listed_as_both_hit_and_key_error_is_refused(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    doc["byStratum"]["http"]["keyErrorIds"] = [doc["byStratum"]["http"]["hitIds"][0]]
    refuses(e, doc, "http")


def test_all_items_as_key_errors_with_an_empty_record_is_refused(tmp_path):
    """The reviewer's plant: 0 hits and every item a 'key-error' passed as a bare count."""
    e = repo_of(tmp_path)
    ids = H.dev_ids("scripts")
    refuses(e, probe(e, key_errors=ids), "key-error")


def test_a_key_error_confirmed_in_the_committed_record_passes(tmp_path):
    e = repo_of(tmp_path)
    ids = H.dev_ids("http")[:1]
    confirm(e.repo, ids)
    check(e, probe(e, key_errors=ids))


def test_a_key_error_not_in_the_record_is_refused_even_when_another_is(tmp_path):
    e = repo_of(tmp_path)
    ids = H.dev_ids("http")
    confirm(e.repo, ids[:1])
    refuses(e, probe(e, key_errors=ids[:2]), "key-error")


def test_a_confirmation_record_edited_after_its_commit_is_not_read(tmp_path):
    e = repo_of(tmp_path)
    ids = H.dev_ids("http")[:1]
    (e.repo / KE_RECORD).write_text(json.dumps({"schema": "ccg-dev-key-errors-v1", "confirmed": [ke(ids[0])]}) + "\n")      # not committed
    refuses(e, probe(e, key_errors=ids), "key-error")


@pytest.mark.parametrize("text", ["not json", "[]", json.dumps({"schema": "other", "confirmed": []}), json.dumps({"schema": "ccg-dev-key-errors-v1"}),
                                   json.dumps({"schema": "ccg-dev-key-errors-v1", "confirmed": ["x"]})])
def test_an_unusable_confirmation_record_is_a_refusal_even_with_no_key_errors(tmp_path, text):
    e = repo_of(tmp_path)
    (e.repo / KE_RECORD).write_text(text)
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "bad record")
    refuses(e, probe(e), "confirmation")


_ID = "a" * 32


@pytest.mark.parametrize("entries", [
    [ke(_ID, confirmedBy=None)], [{k: v for k, v in ke(_ID).items() if k != "confirmedBy"}], [ke(_ID, confirmedBy="")], [ke(_ID, confirmedBy="tdd-engineer")],
    [ke(_ID, confirmedBy="User")],
    [{k: v for k, v in ke(_ID).items() if k != "date"}], [ke(_ID, date=None)], [ke(_ID, date="")], [ke(_ID, date="24/11/2026")], [ke(_ID, date="2026-13-40")],
    [ke(_ID, date="2026-11-24T10:00:00")], [ke(_ID, date=20261124)], [ke(_ID, date="20261124")], [ke(_ID, date="2026-W48-2")],
    [ke(_ID), ke(_ID, date="2026-11-25")], [{"itemId": _ID}]])
def test_a_key_error_entry_without_a_stakeholder_and_a_date_is_a_refusal(tmp_path, entries):
    """The entry says who confirmed it and when; an entry that does not, or that names anyone but the stakeholder, or repeats an id, is not a record."""
    e = repo_of(tmp_path)
    (e.repo / KE_RECORD).write_text(json.dumps({"schema": "ccg-dev-key-errors-v1", "confirmed": entries}) + "\n")
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "bad entries")
    refuses(e, probe(e), "confirmation")


def test_a_well_formed_entry_is_accepted_whether_or_not_a_probe_lists_it(tmp_path):
    e = repo_of(tmp_path)
    (e.repo / KE_RECORD).write_text(json.dumps({"schema": "ccg-dev-key-errors-v1", "confirmed": [ke(_ID)]}) + "\n")
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "entry")
    check(e, probe(e))
    assert G.confirmed_key_errors(e.repo) == {_ID}


def test_a_missing_confirmation_record_is_a_refusal(tmp_path):
    e = repo_of(tmp_path)
    H.git(e.repo, "rm", "-q", KE_RECORD)
    H.git(e.repo, "commit", "-qm", "gone")
    refuses(e, probe(e), "confirmation")


def test_the_refusal_names_strata_and_counts_only(tmp_path):
    e = repo_of(tmp_path)
    doc = probe(e)
    doc["byStratum"]["scripts"]["hitIds"].pop()
    with pytest.raises(G.PrecheckRefusal) as ei:
        check(e, doc)
    assert "scripts" in str(ei.value) and "2" in str(ei.value) and "3" in str(ei.value)


# ------------------------------------------------------------------------------------------------ the file and the dev-rehearsal exemption
def test_read_dev_probe_returns_the_document_and_the_sha256_of_the_exact_bytes(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"a": 1}\n')
    doc, sha = G.read_dev_probe(f)
    assert doc == {"a": 1} and sha == H.sha_of(f)
    with pytest.raises(G.PrecheckRefusal, match="dev"):
        G.read_dev_probe(tmp_path / "absent.json")
    f.write_text("not json")
    with pytest.raises(G.PrecheckRefusal, match="dev"):
        G.read_dev_probe(f)


def dev_world_records(e, ids):
    """Records of ``ids`` (dev http items) that EQUAL the committed dev gold: the gold file is rewritten to the world's record per id and committed."""
    recs = [{**e.w.rec, "itemId": i, "derivedStratum": "http"} for i in H.dev_ids("http")]
    (e.repo / "cpg/eval/dev/ac5-http.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "dev gold = world")
    return [r for r in recs if r["itemId"] in ids]


def _dev_records(e, stratum="http"):
    return [json.loads(l) for l in (e.repo / f"cpg/eval/dev/ac5-{stratum}.jsonl").read_text().splitlines()]


def test_a_list_made_only_of_dev_gold_records_is_a_dev_rehearsal_and_any_other_is_not(tmp_path):
    e = repo_of(tmp_path)
    dev = _dev_records(e)
    assert G.is_dev_rehearsal(e.repo, dev) is True
    assert G.is_dev_rehearsal(e.repo, dev + [{"itemId": "heldout".ljust(32, "0")}]) is False
    assert G.is_dev_rehearsal(e.repo, []) is False


def test_a_record_reusing_a_dev_item_id_with_other_content_is_not_exempt(tmp_path):
    """P7-1: the exemption is by content, not by id: a held-out-shaped record carrying a dev itemId needs the probe."""
    e = repo_of(tmp_path)
    dev = _dev_records(e)
    relabelled = {**dev[0], "impacted": {"type": "symbol", "repoId": "r", "qname": "held/out.py::Fn", "path": "held/out.py"}}
    assert G.is_dev_rehearsal(e.repo, [relabelled]) is False
    assert G.is_dev_rehearsal(e.repo, dev[1:] + [relabelled]) is False


def test_the_same_dev_record_with_another_key_order_or_duplicated_is_still_exempt(tmp_path):
    e = repo_of(tmp_path)
    dev = _dev_records(e)
    assert G.is_dev_rehearsal(e.repo, [dict(reversed(list(dev[0].items()))), dev[0]]) is True


def test_a_second_record_for_a_dev_id_that_differs_from_the_committed_one_is_not_exempt(tmp_path):
    """Every canonicalised record for the id must equal a committed record: one honest copy does not cover a relabelled twin."""
    e = repo_of(tmp_path)
    dev = _dev_records(e)
    assert G.is_dev_rehearsal(e.repo, [dev[0], {**dev[0], "lines": [999]}]) is False


# ------------------------------------------------------------------------------------------------ through score_inputs_check
def _check(e, doc, records=None):
    pf = G.preflight(e.repo, GATE, observed=OBS)
    return G.score_inputs_check(e.repo, GATE, pf, records or [e.w.rec], dev_probe=doc)


def test_score_inputs_check_needs_a_valid_probe_for_a_non_dev_list(tmp_path):
    e = repo_of(tmp_path)
    assert set(_check(e, probe(e))) == {"kind", "movedOut", "patterns"}
    with pytest.raises(G.PrecheckRefusal):
        _check(e, None)
    with pytest.raises(G.PrecheckRefusal, match="http"):
        bad = probe(e)
        bad["byStratum"]["http"]["hitIds"].pop()
        _check(e, bad)


def test_score_inputs_check_needs_no_probe_for_a_dev_rehearsal_list(tmp_path):
    e = repo_of(tmp_path)
    recs = dev_world_records(e, H.dev_ids("http"))
    _check(e, None, recs)


# ------------------------------------------------------------------------------------------------ the snapshot of the build
def test_the_probe_must_vouch_for_the_snapshot_of_the_build_the_gate_made():
    G.check_probe_snapshot({"snapshotId": "s" * 32}, "s" * 32)
    with pytest.raises(G.Refusal, match="snapshot"):
        G.check_probe_snapshot({"snapshotId": "s" * 32}, "t" * 32)


# ------------------------------------------------------------------------------------------------ the run verb
def run_main(e, monkeypatch, *extra, **over):
    monkeypatch.setenv("HOME", str(e.home))
    return R.main(e.argv(*extra), deps=e.deps(**over))


def test_run_with_a_dev_miss_exits_5_before_the_build_and_the_marker_and_spends_nothing(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 2)
    bad = dev_probe_doc(e.repo)
    bad["byStratum"]["scripts"]["hitIds"].pop()
    e.dev_probe.write_text(json.dumps(bad))
    assert run_main(e, monkeypatch) == R.EXIT_RESOLVE == 5
    assert "scripts" in capsys.readouterr().err
    assert e.events == [] and not e.markers() and not e.asked and not e.run_files()
    e.dev_probe.write_text(json.dumps(dev_probe_doc(e.repo, record_commit=e.record_commit)))                                  # fixed: the same list now runs (nothing was spent)
    assert run_main(e, monkeypatch) == R.EXIT_OK


def test_run_without_a_probe_file_or_with_a_foreign_commit_is_refused_before_the_marker(tmp_path, monkeypatch):
    e = Env(tmp_path)
    many_world(e, 1)
    e.dev_probe.write_text(json.dumps(dev_probe_doc(e.repo, recordCommit="f" * 40)))
    assert run_main(e, monkeypatch) == R.EXIT_RESOLVE and e.events == [] and not e.markers()
    e.dev_probe.unlink()
    assert run_main(e, monkeypatch) == R.EXIT_RESOLVE and e.events == [] and not e.markers()


def test_run_refuses_a_probe_that_vouches_for_another_build_after_the_build_with_the_remedy_and_no_marker(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    e.dev_probe.write_text(json.dumps(dev_probe_doc(e.repo, record_commit=e.record_commit, snapshot_id="9" * 32)))
    assert run_main(e, monkeypatch) == R.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "snapshot" in err and "new empty store" in err and not e.markers() and not e.asked


def test_the_marker_pins_the_probes_sha256(tmp_path, monkeypatch):
    e = Env(tmp_path)
    many_world(e, 1)
    assert run_main(e, monkeypatch) == R.EXIT_OK
    assert e.marker()["pins"]["devProbeSha256"] == H.sha_of(e.dev_probe)


def test_a_resume_with_a_different_probe_file_is_refused(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 2)
    e.ask_hook = lambda req, n: (_ for _ in ()).throw(ConnectionError("c")) if n == 2 else None
    assert run_main(e, monkeypatch) == R.EXIT_RESUMABLE
    e.dev_probe.write_text(json.dumps(dev_probe_doc(e.repo, record_commit=e.record_commit, note="c")))            # a different (still valid) attestation
    e.ask_hook = None
    assert run_main(e, monkeypatch, "--resume") == R.EXIT_REFUSED
    assert "devProbeSha256" in capsys.readouterr().err


def test_a_dev_rehearsal_list_runs_with_no_probe_and_pins_none(tmp_path, monkeypatch):
    e = Env(tmp_path)
    recs = dev_world_records(e, H.dev_ids("http")[:1])
    e.set_records(recs)
    e.dev_probe.unlink()
    monkeypatch.setenv("HOME", str(e.home))
    argv = ["run", "--gate-id", GATE, "--lists", str(e.lists), "--custodian", str(e.cust), "--repo", str(e.repo)]
    assert R.main(argv, deps=e.deps()) == R.EXIT_OK
    assert e.marker()["pins"]["devProbeSha256"] is None


def test_a_dev_only_list_does_not_read_a_probe_file_it_is_given_and_pins_none(tmp_path, monkeypatch):
    e = Env(tmp_path)
    e.set_records(dev_world_records(e, H.dev_ids("http")[:1]))
    e.dev_probe.write_text("garbage, not even json")                                           # named on the command line, never read for a dev-only list
    assert run_main(e, monkeypatch) == R.EXIT_OK
    assert e.marker()["pins"]["devProbeSha256"] is None


def test_the_run_reads_the_flow_stratum_only_when_the_flow_flag_is_true(tmp_path, monkeypatch, capsys):
    from test_gate_run_crash import commit_flow_decision
    e = Env(tmp_path)
    many_world(e, 1)
    commit_flow_decision(e)
    e.dev_probe.write_text(json.dumps(dev_probe_doc(e.repo, record_commit=e.record_commit, strata=("http", "mcp", "scripts"))))
    assert run_main(e, monkeypatch) == R.EXIT_RESOLVE and "flow" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ the score verb: the same FILE (the content was checked by run)
def test_score_refuses_a_probe_file_other_than_the_one_the_run_pinned(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.dev_probe.write_text(json.dumps(e.probe_doc(note="c")))                       # valid, but not the file the run was started under
    assert e.main() == 2
    cap = capsys.readouterr()
    assert "probe" in cap.err and cap.out == "" and not (e.cust / "gate-report.json").exists()


def test_score_refuses_a_missing_probe_argument_for_a_run_that_pinned_one(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    assert e.main(e.argv(**{"--dev-probe": None})) == 2
    assert "probe" in capsys.readouterr().err


def test_score_accepts_the_pinned_file_after_a_harness_only_cure_changed_the_hashes_it_carries(tmp_path, monkeypatch, capsys):
    """-p1-gate-marker-ml.md §3: a harness-only fix is re-scored on the recorded answers; the probe's content was checked at ``run`` and its identity is pinned."""
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    (e.repo / "cpg/eval/tools/gate.py").write_text("# harness-only fix\n")
    H.git(e.repo, "commit", "-qam", "fix")
    e.main()
    assert json.loads(capsys.readouterr().out)["scoreLabel"] == "after-adaptation"


def test_score_of_a_dev_only_list_ignores_a_probe_file_it_is_given(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.set_records(dev_world_records(e, H.dev_ids("http")[:1]))
    m = json.loads((e.cust / f"gate-run-{e.key}.marker").read_text())
    m["pins"]["devProbeSha256"] = None
    (e.cust / f"gate-run-{e.key}.marker").write_text(json.dumps(m))
    assert e.main() in (0, 1) and json.loads(capsys.readouterr().out)["verdict"]               # the file in --dev-probe is a valid one whose sha is NOT the pin


def test_score_of_a_dev_only_list_needs_no_probe_and_the_marker_pins_none(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.set_records(dev_world_records(e, H.dev_ids("http")[:1]))
    m = json.loads((e.cust / f"gate-run-{e.key}.marker").read_text())
    m["pins"]["devProbeSha256"] = None
    (e.cust / f"gate-run-{e.key}.marker").write_text(json.dumps(m))
    assert e.main(e.argv(**{"--dev-probe": None})) in (0, 1) and json.loads(capsys.readouterr().out)["verdict"]


# ------------------------------------------------------------------------------------------------ the real record
def test_the_real_key_error_record_is_valid_and_names_only_dev_gold_items():
    """The record in this checkout (working copy; its commit is the freeze round's) passes the harness's own validation, and each id is a real dev item."""
    from pathlib import Path
    root = Path(G.__file__).resolve().parents[3]
    path = root / G.DEV_KEY_ERRORS
    ids = G.confirmed_key_errors(root, lambda _p: path.read_text(encoding="utf-8"))
    dev = {json.loads(l)["itemId"] for s in G.STRATA for l in (root / G.DEV_GOLD.format(s)).read_text().splitlines() if l.strip()}
    assert ids and ids <= dev


# ------------------------------------------------------------------------------------------------ F7: the exemption over every stratum
@pytest.mark.parametrize("stratum", ["http", "mcp", "scripts", "flow", "wiring"])
def test_a_committed_record_of_each_stratum_is_exempt_and_its_content_twin_is_not(tmp_path, stratum):
    e = repo_of(tmp_path)
    mixed = [_dev_records(e, s)[0] for s in ("http", "mcp", "scripts", "flow", "wiring")]
    assert G.is_dev_rehearsal(e.repo, mixed) is True
    twin = {**_dev_records(e, stratum)[0], "lines": [999]}
    assert G.is_dev_rehearsal(e.repo, [twin if r["derivedStratum"] == stratum else r for r in mixed]) is False


# ------------------------------------------------------------------------------------------------ F4: the record is read at the freeze commit
def test_a_key_error_record_widened_after_the_record_commit_is_refused(tmp_path):
    """The analyst's F4 scenario: freeze with an empty record, then a later commit confirms a miss as a key-error."""
    e = repo_of(tmp_path)
    frozen = H.head(e.repo)
    ids = H.dev_ids("http")[:1]
    confirm(e.repo, ids)                                                               # a later, committed widening
    G.dev_probe_check(probe(e, key_errors=ids), e.repo, H.head(e.repo), G.STRATA_GATED)                  # at the later commit it IS the record: accepted
    with pytest.raises(G.PrecheckRefusal, match="confirmation"):
        G.dev_probe_check(probe(e, key_errors=ids, record_commit=frozen), e.repo, frozen, G.STRATA_GATED)   # the freeze commit's record is empty: refused


def test_a_record_that_does_not_exist_at_the_record_commit_is_refused(tmp_path):
    e = repo_of(tmp_path)
    H.git(e.repo, "commit", "--allow-empty", "-qm", "x")
    with pytest.raises(G.PrecheckRefusal, match="confirmation"):
        G.confirmed_key_errors(e.repo, record_commit="0" * 40)


# ------------------------------------------------------------------------------------------------ F1: the Q8 predicate excuses dev items, with counts
FENCED = "```\nplain\n```\n"                    # line 2 is inside a fence: shape B for a markdown file


def _scripts_gold(e, *, path="doc.md", http_too=False):
    """Rewrite the scripts dev gold so each record carries an impacted path and gold line, and commit; -> the records."""
    recs = [{"itemId": i, "derivedStratum": "scripts", "counting": True, "lines": [2], "impacted": {"path": path}} for i in H.dev_ids("scripts")]
    (e.repo / "cpg/eval/dev/ac5-scripts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "scripts gold")
    return recs


def _no_hit_for(doc, ids):
    for s in doc["byStratum"].values():
        s["hitIds"] = [i for i in s["hitIds"] if i not in ids]
    return doc


def test_a_dev_item_the_predicate_withholds_is_excused_from_the_hit_requirement(tmp_path):
    e = repo_of(tmp_path)
    _scripts_gold(e)
    gone = H.dev_ids("scripts")[:1]
    doc = _no_hit_for(probe(e), gone)
    with pytest.raises(G.PrecheckRefusal, match="scripts"):
        G.dev_probe_check(doc, e.repo, H.head(e.repo), G.STRATA_GATED, file_text=lambda rel: None)     # no text: nothing is withheld, the miss stands
    G.dev_probe_check(doc, e.repo, H.head(e.repo), G.STRATA_GATED, file_text=lambda rel: FENCED)      # fenced: excused


def test_the_excuse_is_the_predicates_alone_a_plain_line_that_is_not_listed_is_refused(tmp_path):
    e = repo_of(tmp_path)
    _scripts_gold(e)
    doc = _no_hit_for(probe(e), H.dev_ids("scripts")[:1])
    with pytest.raises(G.PrecheckRefusal, match="scripts"):
        G.dev_probe_check(doc, e.repo, H.head(e.repo), G.STRATA_GATED, file_text=lambda rel: "plain\nplain\n")


def test_only_the_scripts_stratum_is_excusable(tmp_path):
    """The shapes are scripts-stratum shapes: a fenced-looking http dev item that is not a hit is still a miss."""
    e = repo_of(tmp_path)
    recs = [{"itemId": i, "derivedStratum": "http", "counting": True, "lines": [2], "impacted": {"path": "doc.md"}} for i in H.dev_ids("http")]
    (e.repo / "cpg/eval/dev/ac5-http.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "http gold")
    doc = _no_hit_for(probe(e), H.dev_ids("http")[:1])
    with pytest.raises(G.PrecheckRefusal, match="http"):
        G.dev_probe_check(doc, e.repo, H.head(e.repo), G.STRATA_GATED, file_text=lambda rel: FENCED)


def test_withheld_counts_are_per_shape_and_total_over_counting_scripts_items_only():
    recs = [{"itemId": str(n), "derivedStratum": "scripts", "counting": True, "lines": [2], "impacted": {"path": "d.md"}} for n in range(3)]
    recs += [{"itemId": "x", "derivedStratum": "scripts", "counting": False, "lines": [2], "impacted": {"path": "d.md"}},
             {"itemId": "h", "derivedStratum": "http", "counting": True, "lines": [2], "impacted": {"path": "d.md"}}]
    assert G.withheld_counts(recs, lambda rel: FENCED) == {"scriptsCounting": 3, "withheld": 3, "byShape": {"B": 3}}
    assert G.withheld_counts(recs, lambda rel: None) == {"scriptsCounting": 3, "withheld": 0, "byShape": {}}


@pytest.mark.parametrize("withheld,total,ok", [(19, 79, True), (20, 79, False), (0, 0, True), (1, 4, True), (2, 4, False)])
def test_the_cap_is_a_quarter_of_the_counting_scripts_items(withheld, total, ok):
    counts = {"scriptsCounting": total, "withheld": withheld, "byShape": {}}
    if ok:
        G.check_withheld_cap(counts)
    else:
        with pytest.raises(G.PrecheckRefusal, match="cap"):
            G.check_withheld_cap(counts)


def _scripts_list(e, n):
    recs = [{**e.w.rec, "itemId": f"s{k:02d}".ljust(32, "0"), "derivedStratum": "scripts", "lines": [2], "impacted": {"path": "doc.md"}} for k in range(n)]
    e.set_records(recs)
    return recs


def test_run_over_the_cap_exits_5_before_the_marker_and_spends_nothing(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    _scripts_list(e, 4)
    monkeypatch.setattr(G, "eval_commit_reader", lambda repo: (lambda rel: FENCED if rel == "doc.md" else None))
    assert run_main(e, monkeypatch) == R.EXIT_RESOLVE == 5
    assert e.markers() == [] and e.refresh_kw is None and "cap" in capsys.readouterr().err


def test_run_under_the_cap_records_the_withheld_counts_in_the_marker(tmp_path, monkeypatch):
    e = Env(tmp_path)
    _scripts_list(e, 4)
    monkeypatch.setattr(G, "eval_commit_reader", lambda repo: (lambda rel: None))
    assert run_main(e, monkeypatch) == R.EXIT_OK
    assert e.marker()["withheldShapes"] == {"scriptsCounting": 4, "withheld": 0, "byShape": {}}


def test_score_reports_the_withheld_shape_counts_beside_the_verdict(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    monkeypatch.setattr(G, "eval_commit_reader", lambda repo: (lambda rel: None))
    e = ScoreEnv(tmp_path, monkeypatch)
    assert e.main() in (0, 1)
    rep = json.loads(capsys.readouterr().out)
    assert rep["withheldShapes"] == {"scriptsCounting": 0, "withheld": 0, "byShape": {}}


def test_the_default_file_reader_returns_text_of_the_tree_at_eval_commit(tmp_path):
    e = Env(tmp_path)
    (e.repo / "cpg/eval/tools/gate.py").write_text("# edited after EVAL_COMMIT\n")                  # the reader is the build tree's, not the working copy's
    read = G.eval_commit_reader(e.repo)
    assert read("cpg/eval/tools/gate.py") == "# cpg/eval/tools/gate.py\n"
    assert read("no/such/file.md") is None
    assert G.eval_commit_reader(H.make_repo(tmp_path / "bare"))("cpg/README.md") is None            # no EVAL_COMMIT file: nothing is withheld
