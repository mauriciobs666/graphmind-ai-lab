"""P1-32b ``gate_run``: the once-only marker as a state machine (``docs/plans/connected-code-graph-p1-gate-marker-ml.md`` §1) and its kill-at-each-step
tests. A planted ``Killed`` (a ``BaseException``) stands in for ``SIGKILL``: nothing in the run may write on it, so the marker stays ``started``.
Offline and synthetic; nothing here is held-out data."""
import json
import os
import shutil
from pathlib import Path

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, Killed, many_world, GATE, OBS

import gate_run as R
from clients.common import AnchorNotResolved, NoCredentials, StoreAuthError, StoreError


def killing_at(k):
    """An ask hook that kills the process at the k-th question (1-based), before it returns."""
    def hook(req, n):
        if n == k:
            raise Killed()
    return hook


def clean_run_answers(tmp_path_factory, n):
    e = Env(tmp_path_factory.mktemp("clean"))
    many_world(e, n)
    e.go()
    return json.loads(e.run_files()[0].read_text())


# ------------------------------------------------------------------------------------------------ the states
def test_the_marker_is_started_while_asking_and_asked_after_the_last_answer(tmp_path):
    e = Env(tmp_path)
    seen = []
    e.ask_hook = lambda req, n: seen.append(e.marker()["state"])
    e.go()
    assert seen == ["started"] and e.marker()["state"] == "asked"


def test_the_marker_payload_holds_every_pin_the_resume_compares(tmp_path):
    e = Env(tmp_path)
    e.go()
    m = e.marker()
    assert set(m["pins"]) == {"recordCommit", "harnessSha256", "snapshotId", "prereg", "coreTreeHash", "label", "flowFlag", "withheld", "probeMapSha256"}
    assert m["pins"]["snapshotId"] == H.SNAP and set(m["pins"]["prereg"]) == set(G.PREREG_FIELDS) and m["pins"]["label"] == "first-contact"
    assert m["head"] == H.head(e.repo) and m["startedAt"] and m["gateId"] == GATE and m["records"] == 1 and m["resumes"] == []


def test_every_answer_is_on_disk_before_the_next_ask(tmp_path):
    e = Env(tmp_path)
    many_world(e, 4)
    lines_at_ask = []
    e.ask_hook = lambda req, n: lines_at_ask.append(len(e.answers_file().read_text().splitlines()) if list(e.cust.glob("run-*.answers.jsonl")) else 0)
    e.go()
    assert lines_at_ask == [0, 1, 2, 3] and len(e.answers_file().read_text().splitlines()) == 4


def test_each_answer_line_is_fsynced_before_the_next_ask(tmp_path, monkeypatch):
    e = Env(tmp_path)
    many_world(e, 3)
    real, log = os.fsync, []

    def spy(fd):
        log.append(os.readlink(f"/proc/self/fd/{fd}"))
        return real(fd)
    monkeypatch.setattr(os, "fsync", spy)
    synced_at_ask = []
    e.ask_hook = lambda req, n: synced_at_ask.append(sum(1 for f in log if f.endswith(".answers.jsonl")))
    e.go()
    assert synced_at_ask == [0, 1, 2] and sum(1 for f in log if f.endswith(".answers.jsonl")) == 3


# ------------------------------------------------------------------------------------------------ kill at each step
def test_a_kill_right_after_the_marker_leaves_started_and_no_answers_and_the_resume_asks_everything(tmp_path, tmp_path_factory):
    e = Env(tmp_path)
    many_world(e, 3)
    e.ask_hook = killing_at(1)
    with pytest.raises(Killed):
        e.go()
    assert e.marker()["state"] == "started" and not e.run_files()
    e.ask_hook, e.asked = None, []
    e.go(resume=True)
    assert len(e.asked) == 3 and e.marker()["state"] == "asked"
    assert json.loads(e.run_files()[0].read_text())["answers"] == clean_run_answers(tmp_path_factory, 3)["answers"]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_a_kill_after_answer_k_resumes_asking_only_the_missing_anchors_and_never_re_asks_a_recorded_one(tmp_path, k):
    e = Env(tmp_path)
    recs = many_world(e, 5)
    e.ask_hook = killing_at(k + 1)
    with pytest.raises(Killed):
        e.go()
    assert len(e.answers_file().read_text().splitlines()) == k
    first_asked = list(e.asked[:k])
    e.ask_hook, e.asked = None, []
    e.go(resume=True)
    assert len(e.asked) == 5 - k and not set(e.asked) & set(first_asked)
    assert sorted(e.asked + first_asked) == sorted(r["anchorId"] for r in recs)
    run = json.loads(e.run_files()[0].read_text())
    assert len(run["answers"]) == 5 and e.marker()["state"] == "asked"


def test_a_kill_during_the_last_append_leaves_a_torn_line_that_the_resume_discards_and_re_asks(tmp_path, monkeypatch):
    e = Env(tmp_path)
    recs = many_world(e, 3)
    real_append = R._append_line

    def tearing(path, obj):
        if len(e.asked) == 2:                                           # the second answer: half a line reaches the disk, then the process dies
            line = json.dumps(obj, sort_keys=True, separators=(",", ":"))
            with open(path, "ab") as fh:
                fh.write(line[: len(line) // 2].encode())
            raise Killed()
        return real_append(path, obj)
    monkeypatch.setattr(R, "_append_line", tearing)
    with pytest.raises(Killed):
        e.go()
    assert not e.answers_file().read_bytes().endswith(b"\n")
    monkeypatch.setattr(R, "_append_line", real_append)
    e.asked = []
    e.go(resume=True)
    assert len(e.asked) == 2                                            # the torn anchor and the never-asked one
    lines = e.answers_file().read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(l) for l in lines)
    assert len(json.loads(e.run_files()[0].read_text())["answers"]) == 3


def test_a_torn_last_line_that_ends_in_a_newline_but_is_not_json_is_also_discarded(tmp_path):
    e = Env(tmp_path)
    many_world(e, 2)
    e.ask_hook = killing_at(2)
    with pytest.raises(Killed):
        e.go()
    with open(e.answers_file(), "ab") as fh:
        fh.write(b'{"anchorId": "x", "req\n')
    e.ask_hook, e.asked = None, []
    e.go(resume=True)
    assert len(e.asked) == 1 and len(e.answers_file().read_text().splitlines()) == 2


def test_a_corrupt_line_that_is_not_the_last_is_a_harness_error_not_a_silent_re_ask(tmp_path):
    e = Env(tmp_path)
    many_world(e, 3)
    e.ask_hook = killing_at(3)
    with pytest.raises(Killed):
        e.go()
    f = e.answers_file()
    ls = f.read_text().splitlines()
    f.write_text("{not json\n" + ls[1] + "\n")
    e.ask_hook, e.asked = None, []
    with pytest.raises(G.HarnessError):
        e.go(resume=True)
    assert e.asked == []


def test_a_kill_during_the_run_json_write_leaves_no_run_json_and_the_resume_writes_it_without_asking(tmp_path, monkeypatch):
    e = Env(tmp_path)
    many_world(e, 2)
    monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(Killed()) if str(b).endswith(".json") else shutil.move(a, b))
    with pytest.raises(Killed):
        e.go()
    monkeypatch.undo()
    assert not e.run_files() and e.marker()["state"] == "started" and len(e.asked) == 2
    e.asked = []
    e.go(resume=True)
    assert e.asked == [] and len(json.loads(e.run_files()[0].read_text())["answers"]) == 2 and e.marker()["state"] == "asked"


def test_a_resume_after_a_completed_run_is_refused_and_nothing_is_asked(tmp_path):
    e = Env(tmp_path)
    e.go()
    e.asked.clear()
    with pytest.raises(G.Refusal, match="asked|complete|resum"):
        e.go(resume=True)
    assert e.asked == []


def test_a_resume_with_no_marker_is_refused(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="nothing to resume"):
        e.go(resume=True)
    assert e.events == []


def test_a_run_without_resume_against_a_started_marker_is_refused(tmp_path):
    e = Env(tmp_path)
    many_world(e, 2)
    e.ask_hook = killing_at(2)
    with pytest.raises(Killed):
        e.go()
    e.ask_hook = None
    with pytest.raises(G.Refusal, match="already run"):
        e.go()


# ------------------------------------------------------------------------------------------------ resume pins
def killed_run(tmp_path, n=3, k=2):
    e = Env(tmp_path)
    many_world(e, n)
    e.ask_hook = killing_at(k)
    with pytest.raises(Killed):
        e.go()
    e.ask_hook, e.asked = None, []
    return e


@pytest.mark.parametrize("pin", ["recordCommit", "harnessSha256", "snapshotId", "prereg", "coreTreeHash", "label"])
def test_a_resume_is_refused_when_any_pinned_value_differs_from_the_marker_and_asks_nothing(tmp_path, pin):
    e = killed_run(tmp_path)
    m = next(iter(e.markers()))
    doc = json.loads(m.read_text())
    doc["pins"][pin] = "tampered" if not isinstance(doc["pins"][pin], dict) else {**doc["pins"][pin], "tampered": 1}
    m.write_text(json.dumps(doc))
    with pytest.raises(G.Refusal, match=pin):
        e.go(resume=True)
    assert e.asked == [] and json.loads(m.read_text())["state"] == "started"


def test_a_resume_after_the_harness_was_edited_and_committed_is_refused_on_the_harness_pin(tmp_path):
    e = killed_run(tmp_path)
    f = e.repo / G.HARNESS_FILES[0]
    f.write_text(f.read_text() + "# adapted\n")
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "adapt")
    with pytest.raises(G.Refusal):                                      # the label is after-adaptation now: refused unless allowed
        e.go(resume=True)
    with pytest.raises(G.Refusal, match="harnessSha256"):               # and when allowed, the pinned hash differs
        e.go(resume=True, allow_after_adaptation=True)
    assert e.asked == []


def test_a_resume_after_the_record_commit_changed_is_refused(tmp_path):
    e = killed_run(tmp_path)
    with pytest.raises(G.Refusal, match="recordCommit"):
        e.go(resume=True, record_commit_sha=e.eval_commit, allow_after_adaptation=True)
    assert e.asked == []


def test_a_resume_after_the_snapshot_changed_is_refused(tmp_path):
    e = killed_run(tmp_path)
    e.snap.sid = "c" * 32
    e.engine_sid = "c" * 32
    with pytest.raises(G.Refusal, match="snapshotId"):
        e.go(resume=True)
    assert e.asked == []


def test_a_resume_after_a_prereg_field_changed_is_refused_by_the_preflight(tmp_path):
    e = killed_run(tmp_path)
    with pytest.raises(G.Refusal, match="first-contact"):
        e.go(resume=True, observed={**OBS, "defaultDepthBound": 3})
    assert e.asked == []


def test_a_resume_with_a_dirty_cpg_tree_is_refused_by_the_preflight(tmp_path):
    e = killed_run(tmp_path)
    (e.repo / "cpg/ccg/a.py").write_text("A = 9\n")
    with pytest.raises(G.Refusal, match="first-contact"):
        e.go(resume=True)
    assert e.asked == []


def test_a_resume_on_another_list_has_no_marker_to_resume(tmp_path):
    e = killed_run(tmp_path)
    other = H.item(99, H.contract("http", "GET /zzz"), H.sym("svc/z.py::z"))
    e.snap.add_node("ContractPoint", other["anchorId"])
    e.set_records([other])
    with pytest.raises(G.Refusal, match="nothing to resume"):
        e.go(resume=True)


def test_a_resume_with_a_different_engine_snapshot_is_refused(tmp_path):
    e = killed_run(tmp_path)
    e.engine_sid = "d" * 32
    with pytest.raises(G.Refusal, match="snapshot"):
        e.go(resume=True)
    assert e.asked == []


def test_a_resume_repeats_the_resolve_dry_run_and_refuses_on_a_failure(tmp_path):
    e = killed_run(tmp_path)
    spec = list(e.table)[-1]
    e.overrides[spec] = {"status": "unresolved", "candidates": []}
    with pytest.raises(R.ResolveRefusal):
        e.go(resume=True)
    assert e.asked == []


def test_a_resume_is_refused_when_a_run_json_already_exists(tmp_path):
    e = killed_run(tmp_path)
    (e.cust / f"run-{G.list_key(json.loads(l) for l in e.lists.read_text().splitlines())}.json").write_text("{}")
    with pytest.raises(G.Refusal, match="run file|score"):
        e.go(resume=True)


def test_a_resume_tolerates_one_in_flight_cache_entry_and_no_more(tmp_path):
    e = killed_run(tmp_path, n=4, k=3)                                  # two recorded answers
    with pytest.raises(G.Refusal, match="cache"):
        e.go(resume=True, cache_len=lambda: 4)
    e.go(resume=True, cache_len=lambda: 3)
    assert e.marker()["state"] == "asked"


def test_a_resume_that_skips_the_build_reuses_the_store_and_never_rebuilds(tmp_path):
    e = killed_run(tmp_path)
    e.events.clear()
    e.go(resume=True)
    assert [ev for ev in e.events if ev[0] == "refresh"] == []


# ------------------------------------------------------------------------------------------------ failure classes
TRANSIENT = [ConnectionError("x"), TimeoutError("x"), OSError("x"), ConnectionResetError("x"), KeyboardInterrupt()]
DETERMINISTIC = [ValueError("x"), KeyError("x"), TypeError("x"), G.HarnessError("x"), RuntimeError("unknown"), AttributeError("x"),
                 NoCredentials("x"), AnchorNotResolved("x", status="none", detail={}), NotImplementedError("x")]


def _store_error_over(cause):
    try:
        try:
            raise cause
        except Exception:
            raise StoreError("the store is not reachable") from None
    except StoreError as exc:
        return exc


@pytest.mark.parametrize("exc", TRANSIENT + [_store_error_over(ConnectionError("c")), _store_error_over(TimeoutError("t"))], ids=repr)
def test_connection_timeout_oserror_and_interrupt_are_classified_transient(exc):
    assert R.classify(exc) == "transient"


@pytest.mark.parametrize("exc", DETERMINISTIC + [StoreError("the answer does not validate"), _store_error_over(ValueError("v"))], ids=repr)
def test_everything_else_including_an_unknown_class_is_deterministic(exc):
    assert R.classify(exc) == "deterministic"


def test_a_redis_connection_or_timeout_error_is_transient():
    import redis
    assert R.classify(redis.exceptions.ConnectionError("x")) == "transient" and R.classify(redis.exceptions.TimeoutError("x")) == "transient"
    assert R.classify(redis.exceptions.ResponseError("x")) == "deterministic" and R.classify(redis.exceptions.AuthenticationError("x")) == "deterministic"


def failing_at(e, k, exc):
    def hook(req, n):
        if n == k:
            raise exc
    e.ask_hook = hook


@pytest.mark.parametrize("exc", [ConnectionError("c"), TimeoutError("t"), OSError("o"), KeyboardInterrupt()], ids=repr)
def test_a_transient_failure_after_the_marker_writes_failed_transient_and_is_resumable(tmp_path, exc):
    e = Env(tmp_path)
    many_world(e, 3)
    failing_at(e, 2, exc)
    with pytest.raises((R.RunFailed, KeyboardInterrupt)) as caught:
        e.go()
    m = e.marker()
    assert m["state"] == "failed:transient" and m["errorClass"] == type(exc).__name__ and not e.run_files()
    assert len(e.answers_file().read_text().splitlines()) == 1
    e.ask_hook, e.asked = None, []
    e.go(resume=True)
    assert len(e.asked) == 2 and e.marker()["state"] == "asked"


@pytest.mark.parametrize("exc", [NoCredentials("n"), StoreAuthError("a"), G.Refusal("the snapshot changed")], ids=repr)
def test_a_run_wide_deterministic_failure_after_the_marker_writes_failed_deterministic_and_is_not_resumable(tmp_path, exc):
    e = Env(tmp_path)
    many_world(e, 3)
    failing_at(e, 2, exc)
    with pytest.raises(R.RunFailed):
        e.go()
    assert e.marker()["state"] == "failed:deterministic" and not e.run_files()
    e.ask_hook, e.asked = None, []
    with pytest.raises(G.Refusal, match="deterministic|resum"):
        e.go(resume=True)
    assert e.asked == []


def test_an_in_graph_anchor_that_does_not_resolve_after_the_marker_is_a_recorded_harness_error_not_a_miss(tmp_path):
    e = Env(tmp_path)
    exc = AnchorNotResolved("x", status="unresolved", detail={"status": "unresolved", "candidates": []})
    e.go(ask_factory=lambda built, snap: R.make_ask(__import__("gate_run_helpers").FakeClient(exc=exc), stamp=H.stamp(), in_graph=lambda req: True))
    run = json.loads(e.run_files()[0].read_text())
    assert list(run["answers"].values()) == [{"harnessError": "HarnessError"}] and e.marker()["state"] == "asked"


def test_no_failure_class_ever_removes_the_marker(tmp_path, monkeypatch):
    removed = []
    for name in ("remove", "unlink", "rmdir"):
        real = getattr(os, name)
        monkeypatch.setattr(os, name, lambda p, *a, _r=real, **k: (removed.append(str(p)), _r(p, *a, **k))[1])
    real_rmtree = shutil.rmtree
    monkeypatch.setattr(shutil, "rmtree", lambda p, *a, **k: (removed.append(str(p)), real_rmtree(p, *a, **k))[1])
    for i, exc in enumerate([NoCredentials("n"), ConnectionError("c"), Killed()]):
        sub = tmp_path / f"e{i}"
        sub.mkdir()
        e = Env(sub)
        failing_at(e, 1, exc)
        with pytest.raises(BaseException):
            e.go()
        assert len(e.markers()) == 1
    assert not [r for r in removed if r.endswith(".marker")]


# ------------------------------------------------------------------------------------------------ disclosure
def test_run_json_discloses_a_resume_with_counts_before_and_after_and_the_transient_class_names(tmp_path):
    e = Env(tmp_path)
    many_world(e, 4)
    failing_at(e, 3, ConnectionError("c"))
    with pytest.raises(R.RunFailed):
        e.go()
    e.ask_hook, e.asked = None, []
    e.go(resume=True)
    d = json.loads(e.run_files()[0].read_text())["resume"]
    assert d["resumed"] is True and d["resumeCount"] == 1 and d["answerCountsBefore"] == [2] and d["answerCountsAfter"] == [2]
    assert d["transientClasses"] == ["ConnectionError"] and "not re-asked" in d["statement"]
    assert json.loads(e.run_files()[0].read_text())["preflight"]["label"] == "first-contact"        # a resumed gate is not relabelled


def test_an_uninterrupted_run_discloses_resumed_false(tmp_path):
    e = Env(tmp_path)
    e.go()
    d = json.loads(e.run_files()[0].read_text())["resume"]
    assert d["resumed"] is False and d["resumeCount"] == 0 and d["answerCountsBefore"] == [] and d["transientClasses"] == []


def test_two_resumes_are_both_counted(tmp_path):
    e = Env(tmp_path)
    many_world(e, 5)
    failing_at(e, 2, TimeoutError("t"))
    with pytest.raises(R.RunFailed):
        e.go()
    failing_at(e, 2, TimeoutError("t"))
    e.asked = []
    with pytest.raises(R.RunFailed):
        e.go(resume=True)
    e.ask_hook, e.asked = None, []
    e.go(resume=True)
    d = json.loads(e.run_files()[0].read_text())["resume"]
    assert d["resumeCount"] == 2 and d["answerCountsBefore"] == [1, 2] and d["transientClasses"] == ["TimeoutError", "TimeoutError"]


# ------------------------------------------------------------------------------------------------ the CLI over the states
def test_cli_exit_codes_distinguish_spent_resumable_and_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    e = Env(tmp_path)
    many_world(e, 3)
    failing_at(e, 2, ConnectionError("c"))
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_RESUMABLE == 6
    cap = capsys.readouterr()
    assert cap.out == "" and "failed:transient" in cap.err and "ConnectionError" in cap.err and cap.err.count("Traceback") == 0
    e.ask_hook = None
    assert R.main(e.argv(), deps=e.deps()) == 2                                        # no --resume: refused
    capsys.readouterr()
    assert R.main(e.argv("--resume"), deps=e.deps()) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["asked"] == 3


def test_cli_a_run_wide_error_mid_run_is_caught_with_its_class_and_exit_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    e = Env(tmp_path)
    failing_at(e, 1, StoreAuthError("down"))
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_SPENT == 3
    cap = capsys.readouterr()
    assert "failed:deterministic" in cap.err and "StoreAuthError" in cap.err and "Traceback" not in cap.err and "SENTINEL" not in cap.err


def test_cli_a_resume_refusal_exits_2_without_content(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    e = Env(tmp_path)
    e.go()
    assert R.main(e.argv("--resume"), deps=e.deps()) == 2
    cap = capsys.readouterr()
    assert cap.err.count("REFUSED") == 1 and "SENTINEL" not in cap.err


def test_a_resume_of_a_failed_transient_run_is_started_again_while_it_asks(tmp_path):
    e = Env(tmp_path)
    many_world(e, 3)
    failing_at(e, 2, ConnectionError("c"))
    with pytest.raises(R.RunFailed):
        e.go()
    seen = []
    e.ask_hook, e.asked = lambda req, n: seen.append(e.marker()["state"]), []
    e.go(resume=True)
    assert seen == ["started", "started"]
