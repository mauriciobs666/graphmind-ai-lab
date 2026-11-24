"""CO-5: the dev-gold pre-gate probe (``docs/plans/connected-code-graph-p1-closeout.md`` §3.5). A gated stratum with FEWER dev hits than dev items (not only zero)
blocks the start unless the whole shortfall is a stakeholder-confirmed ``key-error``; the dev result is a file the rehearsal writes (``gate.DEV_PROBE_SCHEMA``),
and the check runs in ``score_inputs_check``: ``run`` refuses before the build and the marker (exit 5, nothing spent), ``score`` refuses first. Offline, synthetic."""
import json

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, GATE, OBS, dev_probe_doc, many_world

import gate_run as R


def probe(**by_stratum):
    return dev_probe_doc(**by_stratum)


# ------------------------------------------------------------------------------------------------ the pure check
def test_all_dev_items_hit_passes():
    G.dev_probe_check(probe())


def test_a_flow_stratum_that_is_not_gated_is_not_read():
    doc = probe()
    del doc["byStratum"]["flow"]
    G.dev_probe_check(doc)                                                                  # flow is withheld at the default gated set


@pytest.mark.parametrize("stratum", ["http", "mcp", "scripts"])
def test_one_dev_miss_in_any_gated_stratum_blocks(stratum):
    with pytest.raises(G.PrecheckRefusal, match=stratum):
        G.dev_probe_check(probe(**{stratum: {"items": 49, "hits": 48, "keyErrors": 0}}))


def test_a_shortfall_that_is_not_zero_is_still_blocking():
    """Fewer hits than items, not only zero hits (a stratum at 40 of 49 is not a start)."""
    with pytest.raises(G.PrecheckRefusal, match="scripts"):
        G.dev_probe_check(probe(scripts={"items": 49, "hits": 40, "keyErrors": 0}))
    with pytest.raises(G.PrecheckRefusal, match="scripts"):
        G.dev_probe_check(probe(scripts={"items": 49, "hits": 0, "keyErrors": 0}))


def test_a_shortfall_wholly_confirmed_as_key_error_passes():
    G.dev_probe_check(probe(http={"items": 18, "hits": 16, "keyErrors": 2}))


def test_a_shortfall_only_partly_confirmed_blocks():
    with pytest.raises(G.PrecheckRefusal, match="http"):
        G.dev_probe_check(probe(http={"items": 18, "hits": 15, "keyErrors": 2}))


def test_more_hits_and_key_errors_than_items_is_an_inconsistent_file_not_a_pass():
    with pytest.raises(G.PrecheckRefusal, match="http"):
        G.dev_probe_check(probe(http={"items": 18, "hits": 18, "keyErrors": 1}))


def test_a_withheld_stratum_miss_does_not_block():
    doc = probe()
    doc["byStratum"]["wiring"] = {"items": 11, "hits": 0, "keyErrors": 0}
    G.dev_probe_check(doc)


def test_a_true_flow_flag_makes_the_flow_stratum_part_of_the_probe():
    G.dev_probe_check(probe(), G.gated_strata(True))
    with pytest.raises(G.PrecheckRefusal, match="flow"):
        G.dev_probe_check(probe(flow={"items": 5, "hits": 4, "keyErrors": 0}), G.gated_strata(True))


@pytest.mark.parametrize("bad", [
    None, [], "x", {}, {"schema": "other", "byStratum": {}}, {**dev_probe_doc(), "schema": "ccg-dev-probe-v0"}, {k: v for k, v in dev_probe_doc().items() if k != "schema"}, {"schema": G.DEV_PROBE_SCHEMA}, {"schema": G.DEV_PROBE_SCHEMA, "byStratum": []},
])
def test_a_missing_or_malformed_probe_is_a_refusal(bad):
    with pytest.raises(G.PrecheckRefusal):
        G.dev_probe_check(bad)


@pytest.mark.parametrize("stratum_value", [
    None, {}, {"items": 0, "hits": 0, "keyErrors": 0}, {"items": 5, "hits": 5}, {"items": "5", "hits": 5, "keyErrors": 0},
    {"items": True, "hits": True, "keyErrors": 0}, {"items": 5, "hits": 6, "keyErrors": -1}, {"items": 5.0, "hits": 5, "keyErrors": 0},
])
def test_a_gated_stratum_that_is_absent_empty_or_not_integer_counts_is_a_refusal(stratum_value):
    doc = probe()
    doc["byStratum"]["mcp"] = stratum_value
    with pytest.raises(G.PrecheckRefusal, match="mcp"):
        G.dev_probe_check(doc)


def test_the_refusal_names_counts_and_strata_only():
    with pytest.raises(G.PrecheckRefusal) as ei:
        G.dev_probe_check(probe(scripts={"items": 49, "hits": 42, "keyErrors": 3}))
    assert "scripts" in str(ei.value) and "42" in str(ei.value) and "49" in str(ei.value)


def test_read_dev_probe_reads_the_file_and_refuses_a_missing_or_unparsable_one(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps(probe()))
    assert G.read_dev_probe(f) == probe()
    for bad in (tmp_path / "absent.json",):
        with pytest.raises(G.PrecheckRefusal, match="dev"):
            G.read_dev_probe(bad)
    f.write_text("not json")
    with pytest.raises(G.PrecheckRefusal, match="dev"):
        G.read_dev_probe(f)


# ------------------------------------------------------------------------------------------------ through score_inputs_check
def _check(e, doc):
    pf = G.preflight(e.repo, GATE, observed=OBS)
    return G.score_inputs_check(e.repo, GATE, pf, [e.w.rec], dev_probe=doc)


def test_score_inputs_check_refuses_a_missing_probe_and_a_failing_one(tmp_path):
    e = Env(tmp_path)
    assert set(_check(e, probe())) == {"kind", "movedOut", "patterns"}
    with pytest.raises(G.PrecheckRefusal):
        _check(e, None)
    with pytest.raises(G.PrecheckRefusal, match="http"):
        _check(e, probe(http={"items": 18, "hits": 17, "keyErrors": 0}))


# ------------------------------------------------------------------------------------------------ the run verb: pre-marker, exit 5, nothing spent
def test_run_with_a_dev_miss_exits_5_before_the_build_and_the_marker_and_spends_nothing(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 2)
    e.dev_probe.write_text(json.dumps(probe(scripts={"items": 49, "hits": 48, "keyErrors": 0})))
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(e.argv(), deps=e.deps())
    err = capsys.readouterr().err
    assert rc == R.EXIT_RESOLVE == 5 and "scripts" in err and "48" in err
    assert e.events == [] and not e.markers() and not e.asked and not e.run_files()
    e.dev_probe.write_text(json.dumps(probe()))                                              # fixed: the same list now runs (nothing was spent)
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_OK


def test_run_without_a_dev_probe_file_is_refused_before_the_marker(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    e.dev_probe.unlink()
    monkeypatch.setenv("HOME", str(e.home))
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_RESOLVE
    assert e.events == [] and not e.markers()


def test_run_accepts_a_shortfall_confirmed_as_key_error(tmp_path, monkeypatch):
    e = Env(tmp_path)
    many_world(e, 1)
    e.dev_probe.write_text(json.dumps(probe(http={"items": 18, "hits": 17, "keyErrors": 1})))
    monkeypatch.setenv("HOME", str(e.home))
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_OK


def test_the_run_reads_the_flow_stratum_only_when_the_flow_flag_is_true(tmp_path, monkeypatch, capsys):
    from test_gate_run_crash import commit_flow_decision
    e = Env(tmp_path)
    many_world(e, 1)
    commit_flow_decision(e)
    e.dev_probe.write_text(json.dumps(probe(flow={"items": 5, "hits": 4, "keyErrors": 0})))
    monkeypatch.setenv("HOME", str(e.home))
    flow_decision = e.repo / "cpg" / "eval" / "decisions" / "ccg-p1-flow-python-v2-decision.json"
    assert flow_decision.is_file()
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_RESOLVE and "flow" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ the score verb: the same check, first
def test_score_refuses_a_failing_dev_probe_with_nothing_written(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.dev_probe.write_text(json.dumps(probe(mcp={"items": 20, "hits": 19, "keyErrors": 0})))
    assert e.main() == 2
    cap = capsys.readouterr()
    assert "mcp" in cap.err and cap.out == "" and not (e.cust / "gate-report.json").exists()


def test_score_reads_the_flow_stratum_when_the_run_recorded_a_true_flow_flag(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    dec = e.repo / "cpg" / "eval" / "decisions" / "ccg-p1-flow-python-v2-decision.json"
    dec.parent.mkdir(parents=True, exist_ok=True)
    dec.write_text(json.dumps({"decision": {"verdict": "TRUE:measured"}}))
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "decision")
    e.flow_flag = True
    e.set_records([e.w.rec])
    e.dev_probe.write_text(json.dumps(probe(flow={"items": 5, "hits": 4, "keyErrors": 0})))
    assert e.main() == 2 and "flow" in capsys.readouterr().err
