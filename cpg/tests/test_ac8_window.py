"""P1-31: the AC-8 detector's window and the `conn` verdict (`cpg/sandbox/ac8/window.py`).

Spec: `docs/plans/connected-code-graph-egress-spec3.md` §8.1 (P-d, the detector control scenario) and §8.2 row 8.1. Pure functions over
text files: no Docker, no network. Every failure code is asserted by name, so a verdict that merely "fails" cannot pass for one that
fails for the right reason.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

import ac8_helpers  # noqa: F401  (puts cpg/sandbox/ac8 on sys.path)
import window as W

T0 = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(seconds=60)


def conn(ts: str = "2026-11-11T12:00:30.000Z", **kw) -> dict:
    return {"ts": ts, "srcAddr": "172.18.0.2", "srcPort": 40000, "dstAddr": "172.18.0.3", "dstPort": 443, "bytesOut": 0, "bytesIn": 5, **kw}


def probe_rec(label=W.CONTROL_PROBE_LABEL, **kw) -> dict:
    return {**conn("2026-11-11T12:00:20.000Z"), "probe": label, **kw}


def write_dir(d, conns=(), start=T0, flush=T1, raw_conn=None, residue=0, probes=None, final_residue=0):
    d.mkdir(parents=True, exist_ok=True)
    (d / "conn.jsonl").write_text(raw_conn if raw_conn is not None else "".join(json.dumps(c) + "\n" for c in conns))
    if start is not None:
        (d / "start.marker").write_text(json.dumps({"marker": "start", "ts": W.format_ts(start)}) + "\n")
    if flush is not None:
        (d / "flush.marker").write_text(json.dumps({"marker": "flush", "ts": W.format_ts(flush), "residue": residue}) + "\n")
    if final_residue is not None:
        (d / "final.marker").write_text(json.dumps({"marker": "final", "ts": W.format_ts(T1 + timedelta(seconds=1)), "residue": final_residue}) + "\n")
    if probes is not None:
        (d / "probe.jsonl").write_text("".join(json.dumps(p) + "\n" for p in probes))
    return d


def control_dir(d, **kw):
    """A good detector-control directory: CONTROL_CONNS conns inside the window, the one correct-nonce probe record, residue 0."""
    kw.setdefault("conns", [conn(srcPort=40000 + i) for i in range(W.CONTROL_CONNS)])
    kw.setdefault("probes", [probe_rec()])
    return write_dir(d, **kw)


# ------------------------------------------------------------------ timestamps


@pytest.mark.parametrize("text,ok", [("2026-11-11T12:00:30.000Z", True), ("2026-11-11T12:00:30Z", True), ("2026-11-11T12:00:30.123456Z", True),
                                     ("2026-11-11T12:00:30.000", False), ("2026-11-11 12:00:30.000Z", False),
                                     ("2026-13-11T12:00:30.000Z", False), ("2026-11-11T12:00:30.1234567Z", False), (None, False), (5, False)])
def test_parse_ts(text, ok):
    assert (W.parse_ts(text) is not None) is ok


def test_parse_ts_value_and_roundtrip():
    assert W.parse_ts("2026-11-11T12:00:30.500Z") == T0 + timedelta(seconds=30.5)
    assert W.format_ts(T0 + timedelta(milliseconds=7)) == "2026-11-11T12:00:00.007Z"


# ------------------------------------------------------------------ the conn log


def test_read_jsonl_valid_and_empty(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(conn()) + "\n")
    recs, errs, exists = W.read_jsonl(p)
    assert (len(recs), errs, exists) == (1, [], True)
    p.write_text("")
    assert W.read_jsonl(p) == ([], [], True)


def test_read_jsonl_missing_file_is_not_zero_connections(tmp_path):
    assert W.read_jsonl(tmp_path / "nope.jsonl") == ([], [], False)


@pytest.mark.parametrize("bad", [
    {"ts": "2026-11-11T12:00:30.000Z"},                      # fields missing
    {**conn(), "extra": 1},                                  # a field beyond the seven
    conn(srcPort=70000), conn(dstPort=-1), conn(dstPort=True), conn(bytesIn=-1), conn(bytesOut="0"), conn(srcAddr=""), conn(ts="yesterday"),
    [1, 2],
])
def test_read_jsonl_malformed_record_is_an_error_never_skipped(tmp_path, bad):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(bad) + "\n")
    recs, errs, _ = W.read_jsonl(p)
    assert recs == [] and len(errs) == 1


def test_read_jsonl_not_json_and_partial_line(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(conn()) + "\nnot json\n" + json.dumps(conn())[:20])
    recs, errs, _ = W.read_jsonl(p)
    assert len(recs) == 1
    assert any("not JSON" in e for e in errs) and any("trailing partial line" in e for e in errs)


# ------------------------------------------------------------------ markers


def test_read_markers_ok_and_missing(tmp_path):
    d = write_dir(tmp_path / "d")
    assert W.read_markers(d) == ({"start": T0, "flush": T1, "residue": 0, "final": T1 + timedelta(seconds=1), "final_residue": 0}, [])
    d2 = write_dir(tmp_path / "d2", start=None, residue=4)
    m, errs = W.read_markers(d2)
    assert m == {"start": None, "flush": T1, "residue": 4, "final": T1 + timedelta(seconds=1), "final_residue": 0} and errs == []


@pytest.mark.parametrize("content", ['{"marker":"flush","ts":"2026-11-11T12:00:00.000Z"}', '{"marker":"flush","ts":"2026-11-11T12:00:00.000Z","residue":0}',
                                     '{"marker":"start"}', "not json",
                                     '{"marker":"start","ts":"x"}', '{"marker":"start","ts":"2026-11-11T12:00:00.000Z","x":1}', "[]"])
def test_read_markers_malformed_is_an_error_and_none(tmp_path, content):
    d = write_dir(tmp_path / "d")
    (d / "start.marker").write_text(content)
    m, errs = W.read_markers(d)
    assert m["start"] is None and len(errs) == 1 and errs[0].startswith("start.marker")


@pytest.mark.parametrize("content", ['{"marker":"flush","ts":"2026-11-11T12:01:00.000Z"}',
                                     '{"marker":"flush","ts":"2026-11-11T12:01:00.000Z","residue":-1}',
                                     '{"marker":"flush","ts":"2026-11-11T12:01:00.000Z","residue":"0"}',
                                     '{"marker":"flush","ts":"2026-11-11T12:01:00.000Z","residue":true}',
                                     '{"marker":"flush","ts":"2026-11-11T12:01:00.000Z","residue":0,"x":1}'])
def test_a_flush_marker_must_carry_an_integer_residue(tmp_path, content):
    d = write_dir(tmp_path / "d")
    (d / "flush.marker").write_text(content)
    m, errs = W.read_markers(d)
    assert m["flush"] is None and m["residue"] is None and len(errs) == 1 and errs[0].startswith("flush.marker")
    v = W.verdict_for_dir(d)
    assert not v.ok and any(f.startswith("marker-malformed") for f in v.failures)


def test_a_start_marker_with_a_residue_is_malformed(tmp_path):
    d = write_dir(tmp_path / "d")
    (d / "start.marker").write_text('{"marker":"start","ts":"2026-11-11T12:00:00.000Z","residue":0}')
    assert W.read_markers(d)[1][0].startswith("start.marker")


@pytest.mark.parametrize("residue", [1, 17])
def test_a_nonzero_flush_residue_fails_the_verdict(tmp_path, residue):
    v = W.verdict_for_dir(write_dir(tmp_path / "d", residue=residue))
    assert not v.ok and [f for f in v.failures if f.startswith("flush-residue")] == [
        f"flush-residue: {residue} connection(s) were still unlogged when the detector's drain stopped"]


# -- the termination drain's final marker (security N1): written after the SIGTERM drain, it must be present and report residue 0

def test_a_missing_final_marker_fails_the_verdict(tmp_path):
    v = W.verdict_for_dir(write_dir(tmp_path / "d", final_residue=None))
    assert not v.ok and v.failures == ["marker-missing:final: the detector did not finish its termination drain (it was killed, or never stopped)"]


@pytest.mark.parametrize("residue", [1, 9])
def test_a_nonzero_final_residue_fails_the_verdict(tmp_path, residue):
    v = W.verdict_for_dir(write_dir(tmp_path / "d", final_residue=residue))
    assert not v.ok and v.failures == [f"final-residue: {residue} connection(s) were still unlogged when the termination drain stopped"]


@pytest.mark.parametrize("content", ['{"marker":"final","ts":"2026-11-11T12:01:01.000Z"}', '{"marker":"final","ts":"2026-11-11T12:01:01.000Z","residue":-1}',
                                     '{"marker":"flush","ts":"2026-11-11T12:01:01.000Z","residue":0}', "not json"])
def test_a_malformed_final_marker_is_an_error_not_a_pass(tmp_path, content):
    d = write_dir(tmp_path / "d")
    (d / "final.marker").write_text(content)
    m, errs = W.read_markers(d)
    assert m["final"] is None and m["final_residue"] is None and len(errs) == 1 and errs[0].startswith("final.marker")
    v = W.verdict_for_dir(d)
    assert not v.ok and any(f.startswith("marker-malformed") for f in v.failures)


def test_judge_final_residue_none_fails_and_zero_passes():
    assert W.judge([], T0, T1, final_residue=None).failures[0].startswith("marker-missing:final")
    assert W.judge([], T0, T1, final_residue=0).ok


def test_control_check_a_missing_or_nonzero_final_marker_fails_the_control(tmp_path):
    problems = W.control_check(control_dir(tmp_path / "a", final_residue=4))
    assert len(problems) == 1 and "final" in problems[0] and "4" in problems[0]
    assert "unusable" in W.control_check(control_dir(tmp_path / "b", final_residue=None))[0]


def test_control_check_catches_a_verdict_that_ignores_the_final_residue(tmp_path, monkeypatch):
    d = control_dir(tmp_path / "c")
    real = W.judge
    monkeypatch.setattr(W, "judge", lambda *a, **k: real(*a, **{**k, "final_residue": 0}))
    problems = W.control_check(d)
    assert len(problems) == 2 and all("final" in p for p in problems) and "final-residue" in problems[0]


def test_judge_with_a_flush_marker_and_no_residue_fails_and_zero_passes():
    v = W.judge([], T0, T1, residue=None)
    assert not v.ok and v.failures == ["flush-residue: the flush marker reports no residue"]
    assert W.judge([], T0, T1, residue=0).ok
    assert W.judge([], T0, None, residue=None).failures == ["marker-missing:flush"]   # no flush marker: that one failure, not two


def test_read_jsonl_probe_records_need_a_label(tmp_path):
    f = tmp_path / "probe.jsonl"
    f.write_text(json.dumps(probe_rec("answer/start")) + "\n")
    recs, errs, ex = W.read_jsonl(f, W.PROBE_FIELDS)
    assert (len(recs), errs, ex) == (1, [], True)
    f.write_text(json.dumps(conn()) + "\n" + json.dumps(probe_rec("")) + "\n" + json.dumps(probe_rec(5)) + "\n")
    recs, errs, ex = W.read_jsonl(f, W.PROBE_FIELDS)
    assert recs == [] and len(errs) == 3
    assert W.read_jsonl(f)[0] == [conn()] and len(W.read_jsonl(f)[1]) == 2   # as conn records: the label is an unexpected field


# ------------------------------------------------------------------ the verdict


def test_zero_conn_in_a_closed_window_passes():
    v = W.judge([], T0, T1)
    assert v.ok and v.failures == []


def test_a_conn_inside_the_window_fails():
    v = W.judge([conn()], T0, T1)
    assert not v.ok and len(v.inside) == 1 and v.failures[0].startswith("conn-in-window")


def test_a_conn_before_the_start_marker_fails():
    v = W.judge([conn("2026-11-11T11:59:59.999Z")], T0, T1)
    assert not v.ok and v.failures[0].startswith("conn-outside-window:before-start") and len(v.outside) == 1


def test_a_conn_after_the_flush_marker_fails():
    v = W.judge([conn("2026-11-11T12:01:00.001Z")], T0, T1)
    assert not v.ok and v.failures[0].startswith("conn-outside-window:after-flush")


@pytest.mark.parametrize("ts", ["2026-11-11T12:00:00.000Z", "2026-11-11T12:01:00.000Z"])
def test_a_conn_on_a_marker_instant_is_inside_and_still_fails(ts):
    assert W.judge([conn(ts)], T0, T1).failures[0].startswith("conn-in-window")


def test_missing_start_marker_fails_even_with_no_conn():
    v = W.judge([], None, T1)
    assert not v.ok and v.failures == ["marker-missing:start"]


def test_missing_flush_marker_fails_even_with_no_conn():
    assert W.judge([], T0, None).failures == ["marker-missing:flush"]


def test_conn_with_no_window_is_flagged_outside():
    v = W.judge([conn()], None, None)
    assert any(f.startswith("conn-outside-window:no-window") for f in v.failures) and "marker-missing:start" in v.failures


def test_inverted_window_fails():
    v = W.judge([], T1, T0)
    assert not v.ok and v.failures[0].startswith("window-inverted")


def test_missing_conn_log_fails_a_detector_that_never_ran_proves_nothing():
    v = W.judge([], T0, T1, conn_file_exists=False)
    assert not v.ok and v.failures[0].startswith("detector-log-missing")


def test_malformed_log_lines_fail():
    v = W.judge([], T0, T1, errors=["conn.jsonl:2: not JSON", "marker start.marker: x"])
    assert [f.split(":")[0] for f in v.failures] == ["detector-record-malformed", "marker-malformed"]


def test_window_must_open_before_the_first_policy_record():
    v = W.judge([], T0, T1, policy_times=[T0 - timedelta(milliseconds=1), T0 + timedelta(seconds=1)])
    assert not v.ok and v.failures[0].startswith("window-starts-late")
    assert W.judge([], T0, T1, policy_times=[T0]).ok  # the same instant is not before it
    assert W.judge([], T0, T1, record_times=[T0 - timedelta(milliseconds=1)]).failures[0].startswith("window-starts-late")


def test_window_must_close_after_the_last_record():
    v = W.judge([], T0, T1, record_times=[T1 + timedelta(milliseconds=1)])
    assert not v.ok and v.failures[0].startswith("window-ends-early")
    assert W.judge([], T0, T1, record_times=[T1]).ok


def test_verdict_for_dir_reads_the_detector_directory(tmp_path):
    assert W.verdict_for_dir(write_dir(tmp_path / "ok")).ok
    assert not W.verdict_for_dir(write_dir(tmp_path / "c", [conn()])).ok
    assert not W.verdict_for_dir(write_dir(tmp_path / "m", start=None)).ok
    d = write_dir(tmp_path / "gone")
    (d / "conn.jsonl").unlink()
    assert W.verdict_for_dir(d).failures[0].startswith("detector-log-missing")


# ------------------------------------------------------------------ the detector control scenario


def test_control_check_passes_when_the_verdict_fails_where_it_must(tmp_path):
    assert W.control_check(control_dir(tmp_path / "c")) == []


def test_control_check_needs_a_real_conn(tmp_path):
    problems = W.control_check(control_dir(tmp_path / "c", conns=[]))
    assert any("no conn record" in p for p in problems) and any(f"expected {W.CONTROL_CONNS}" in p for p in problems)


def test_control_check_needs_both_markers_and_both_logs(tmp_path):
    assert "unusable" in W.control_check(control_dir(tmp_path / "a", start=None))[0]
    d = control_dir(tmp_path / "b")
    (d / "conn.jsonl").unlink()
    assert "unusable" in W.control_check(d)[0]
    d = control_dir(tmp_path / "p")
    (d / "probe.jsonl").unlink()
    assert "unusable" in W.control_check(d)[0]


def test_control_check_flags_a_real_conn_outside_the_window(tmp_path):
    conns = [conn(srcPort=40000 + i) for i in range(W.CONTROL_CONNS - 1)] + [conn("2026-11-11T12:05:00.000Z")]
    assert any("not inside the window" in p for p in W.control_check(control_dir(tmp_path / "c", conns=conns)))


@pytest.mark.parametrize("n", [W.CONTROL_CONNS - 1, W.CONTROL_CONNS + 1])
def test_control_check_needs_exactly_the_expected_number_of_conns(tmp_path, n):
    problems = W.control_check(control_dir(tmp_path / "c", conns=[conn(srcPort=40000 + i) for i in range(n)]))
    assert len(problems) == 1 and f"holds {n} conn record(s), expected {W.CONTROL_CONNS}" in problems[0]


@pytest.mark.parametrize("probes", [[], [probe_rec(), probe_rec()], [probe_rec("answer/start")]])
def test_control_check_needs_exactly_the_one_correct_nonce_probe_record(tmp_path, probes):
    problems = W.control_check(control_dir(tmp_path / "c", probes=probes))
    assert len(problems) == 1 and "probe records" in problems[0]


def test_control_check_a_nonzero_residue_fails_the_control(tmp_path):
    problems = W.control_check(control_dir(tmp_path / "c", residue=2))
    assert len(problems) == 1 and "residue 2" in problems[0]


def test_control_check_catches_a_verdict_that_ignores_the_residue(tmp_path, monkeypatch):
    d = control_dir(tmp_path / "c")
    real = W.judge
    monkeypatch.setattr(W, "judge", lambda *a, **k: real(*a, **{**k, "residue": 0}))
    problems = W.control_check(d)
    assert len(problems) == 1 and "flush-residue" in problems[0]


def test_control_check_catches_a_verdict_that_never_fails(tmp_path, monkeypatch):
    d = control_dir(tmp_path / "c")
    monkeypatch.setattr(W, "judge", lambda *a, **k: W.Verdict(True))
    problems = W.control_check(d)
    assert len(problems) == 6 and all("did not fail" in p for p in problems)


def test_control_check_catches_a_verdict_that_always_fails(tmp_path, monkeypatch):
    d = control_dir(tmp_path / "c")
    real = W.judge
    monkeypatch.setattr(W, "judge", lambda conns, *a, **k: real(conns, *a, **k) if conns else W.Verdict(False, ["boom"]))
    assert any("did not pass" in p for p in W.control_check(d))


def test_control_check_catches_a_verdict_blind_to_one_side_of_the_window(tmp_path, monkeypatch):
    d = control_dir(tmp_path / "c")
    real = W.judge

    def blind_after(conns, start, flush, **k):  # forgets conns after the flush marker
        return real([c for c in conns if W.parse_ts(c["ts"]) <= flush], start, flush, **k)
    monkeypatch.setattr(W, "judge", blind_after)
    problems = W.control_check(d)
    assert len(problems) == 1 and "after flush" in problems[0]


# ------------------------------------------------------------------ the CLI


def run_cli(*args):
    return subprocess.run([sys.executable, "-I", str(W.__file__), *args], capture_output=True, text=True)


def test_cli_verdict_exit_codes(tmp_path):
    ok = run_cli("verdict", str(write_dir(tmp_path / "ok")))
    assert ok.returncode == 0 and "PASS" in ok.stdout
    bad = run_cli("verdict", str(write_dir(tmp_path / "bad", [conn()])))
    assert bad.returncode == 1 and "FAIL conn-in-window" in bad.stdout
    assert run_cli("nope").returncode == 64


def test_cli_control_exit_codes(tmp_path):
    assert run_cli("control", str(control_dir(tmp_path / "c"))).returncode == 0
    r = run_cli("control", str(write_dir(tmp_path / "e")))
    assert r.returncode == 1 and "FAIL" in r.stdout
