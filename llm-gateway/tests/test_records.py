"""Call records: one row per attempt (plan §3.7; FR-5, AC-17, AC-19, AC-24)."""

import multiprocessing
import os
import sqlite3
import stat
import threading

import pytest

from llmgateway.records import Attempt, RecordError, RecordStore, new_call_id
from llmgateway.types import FailureClass as F


def attempt(**kw):
    base = dict(
        call_id="c1",
        attempt=1,
        component="answer.interpret",
        scopes=("ccg:system:main",),
        call_site="interpret",
        provider="lm",
        model="qwen2.5-7b",
        hosted=False,
        outcome="ok",
        input_tokens=10,
        output_tokens=5,
        duration_ms=120,
        loaded_state="loaded",
    )
    base.update(kw)
    return Attempt(**base)


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path / "state")


def _writer(args):
    path, n, prefix = args
    s = RecordStore(path)
    for i in range(n):
        s.write(attempt(call_id=f"{prefix}{i}", run_id="eval-1", input_tokens=1, output_tokens=1))


def test_store_files_are_private(tmp_path):
    s = RecordStore(tmp_path / "state")
    s.write(attempt())
    assert stat.S_IMODE(os.stat(tmp_path / "state").st_mode) == 0o700
    for name in os.listdir(tmp_path / "state"):
        assert stat.S_IMODE(os.stat(tmp_path / "state" / name).st_mode) == 0o600, name


def test_symlinked_db_is_not_followed(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    target = tmp_path / "elsewhere"
    target.write_text("")
    os.symlink(target, d / "records.sqlite3")
    with pytest.raises(OSError):
        RecordStore(d)


def test_row_carries_the_fr5_fields_for_local_and_hosted(store):
    store.write(attempt(call_id="a"))
    store.write(
        attempt(
            call_id="b",
            provider="claude",
            model="claude-sonnet-4-5-20250929",
            hosted=True,
            run_id="eval-7",
            loaded_state=None,
        )
    )
    a, b = store.attempts("a")[0], store.attempts("b")[0]
    assert set(a) == set(b)  # same fields from a local and a hosted call (AC-19)
    for key in (
        "call_id",
        "attempt",
        "component",
        "scopes",
        "call_site",
        "run_id",
        "provider",
        "model",
        "hosted",
        "kind",
        "reason",
        "input_tokens",
        "output_tokens",
        "duration_ms",
        "outcome",
        "failure_class",
        "loaded_state",
    ):
        assert key in a, key
    assert a["hosted"] is False and b["hosted"] is True
    assert b["run_id"] == "eval-7" and a["scopes"] == ("ccg:system:main",)


def test_no_column_can_hold_headers_error_bodies_request_ids_or_org_ids(store):
    cols = set(store.columns())
    assert not [
        c
        for c in cols
        if any(
            w in c
            for w in ("header", "error", "body", "request_id", "org", "workspace", "key", "cred")
        )
    ]


def test_fallback_leaves_two_records_with_one_call_id(store):
    cid = new_call_id()
    store.write(
        attempt(
            call_id=cid,
            attempt=1,
            provider="claude",
            hosted=True,
            outcome="failed",
            failure_class=F.RATE_LIMIT,
            input_tokens=None,
            output_tokens=None,
        )
    )
    store.write(attempt(call_id=cid, attempt=2, kind="fallback", reason=F.RATE_LIMIT))
    rows = store.attempts(cid)
    assert [r["attempt"] for r in rows] == [1, 2]
    assert rows[0]["hosted"] and rows[0]["failure_class"] == "rate_limit"
    assert rows[1]["kind"] == "fallback" and rows[1]["reason"] == "rate_limit"
    assert not rows[1]["hosted"]


def test_duplicate_attempt_is_refused(store):
    store.write(attempt())
    with pytest.raises(RecordError, match="already recorded"):
        store.write(attempt())


def test_text_is_dropped_unless_allowed(store):
    store.write(attempt(call_id="t", request_text="hello", response_text="world"))
    row = store.attempts("t")[0]
    assert row["request_text"] is None and row["response_text"] is None
    store.write(
        attempt(call_id="u", request_text="hello", response_text="world"), text_allowed=True
    )
    row = store.attempts("u")[0]
    assert row["request_text"] == "hello" and row["response_text"] == "world"


@pytest.mark.parametrize(
    "kw",
    [
        dict(call_id="has space"),
        dict(model="m\nAuthorization: Bearer x"),
        dict(attempt=0),
        dict(attempt=True),
        dict(hosted=1),
        dict(outcome="boom"),
        dict(outcome="failed"),  # failed without a class
        dict(failure_class=F.AUTH),  # ok with a class
        dict(kind="fallback"),  # fallback without a reason
        dict(reason=F.OUTAGE),  # first with a reason
        dict(input_tokens=-1),
        dict(duration_ms="3"),
        dict(loaded_state="Loaded: {error body}"),
        dict(scopes=["list"]),
        dict(run_id="x" * 201),
        dict(request_text=b"bytes"),
    ],
)
def test_ill_typed_attempts_are_refused(kw):
    with pytest.raises(RecordError):
        attempt(**kw)


def test_failure_class_must_be_the_enum_not_a_string():
    with pytest.raises(RecordError):
        attempt(outcome="failed", failure_class="rate_limit")


def test_tokens_used_sums_input_and_output_for_the_run(store):
    store.write(attempt(call_id="r1", run_id="eval-1", input_tokens=7, output_tokens=3))
    store.write(attempt(call_id="r2", run_id="eval-1", input_tokens=None, output_tokens=5))
    store.write(attempt(call_id="r3", run_id="eval-2", input_tokens=100, output_tokens=100))
    store.write(attempt(call_id="r4"))
    assert store.tokens_used("eval-1") == 15
    assert store.tokens_used("eval-2") == 200
    assert store.tokens_used("nope") == 0


def test_concurrent_threads_lose_nothing(tmp_path):
    s = RecordStore(tmp_path / "state")
    errors = []

    def work(t):
        try:
            for i in range(25):
                s.write(
                    attempt(call_id=f"t{t}-{i}", run_id="eval-1", input_tokens=1, output_tokens=1)
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(s.attempts()) == 200 and s.tokens_used("eval-1") == 400


def test_concurrent_processes_lose_nothing(tmp_path):
    path = str(tmp_path / "state")
    RecordStore(path)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.map(_writer, [(path, 30, f"p{i}-") for i in range(4)])
    s = RecordStore(path)
    assert len(s.attempts()) == 120 and s.tokens_used("eval-1") == 240
    assert sqlite3.connect(s.path).execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_reopen_keeps_rows_and_rejects_a_newer_schema(tmp_path):
    s = RecordStore(tmp_path / "state")
    s.write(attempt())
    assert len(RecordStore(tmp_path / "state").attempts()) == 1
    con = sqlite3.connect(s.path)
    con.execute("PRAGMA user_version=99")
    con.close()
    with pytest.raises(RecordError, match="schema version"):
        RecordStore(tmp_path / "state")


def test_unwritable_store_raises_record_error_without_detail(store, monkeypatch):
    con = sqlite3.connect(store.path)
    con.execute("DROP TABLE attempts")
    con.close()
    with pytest.raises(RecordError, match="cannot write"):
        store.write(attempt())
