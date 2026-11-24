"""P1-32b Pass 4 (P4-1, P4-2): the failure-disposition table (``docs/plans/connected-code-graph-p1-gate-marker-ml.md`` §3), tested with the exceptions the
PRODUCTION client raises. The fault is injected at the lowest layer, the database driver (a ``CypherReader`` over a db whose ``ro_query`` raises the family),
and passes through the real ``clients.common.Client`` (which wraps every fault in ``StoreError`` ``from None``), ``gate_run.make_ask``, ``_Persisting``, and the
marker of ``run_gate``. No exception class here is hand-built to look like the wrapper's: a rule keyed on the outer type, or a test that pins what a hand-made
class does, cannot pass this grid. The grid enumerates the families from the plan's table: a row without a case fails (``test_every_row_of_the_table_has_a_case``).
Offline and synthetic; nothing here is held-out data."""
import errno
import json
import re
from pathlib import Path

import pytest
import redis.exceptions as RE

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world

import gate_run as R
from ccg.engine.reader import CypherReader
from clients.common import Client

PLAN = Path(__file__).resolve().parents[2] / "docs" / "plans" / "connected-code-graph-p1-gate-marker-ml.md"
STAMP = {"snapshotId": H.SNAP, "verdictLogOffset": 0, "verdictLogHash": G.EMPTY_VERDICT_LOG_HASH}
N_ANCHORS = 3


# ------------------------------------------------------------------------------------------------ fault injection at the db
def implicit(outer, inner):
    """``outer`` raised while handling ``inner`` (the interpreter's ``__context__``, what the driver does: ``raise TimeoutError(...)`` inside ``except socket.timeout``)."""
    try:
        try:
            raise inner
        except BaseException:                                                          # noqa: BLE001
            raise outer
    except BaseException as e:                                                         # noqa: BLE001
        return e


def hung_server():
    """What a paused server gives the driver: redis' ``TimeoutError`` raised while handling the socket's builtin ``TimeoutError``."""
    return implicit(RE.TimeoutError("Timeout reading from socket"), TimeoutError("timed out"))


class _Result:
    def __init__(self, rows):
        self.result_set = rows


class FaultDb:
    """A falkordb-py db whose graph answers every ``ro_query`` with ``fault()`` (a fresh exception per call) or, with ``rows``, with those rows."""

    def __init__(self, fault=None, rows=None):
        self.fault, self.rows, self.calls = fault, rows, 0

    def select_graph(self, key):
        return self

    def ro_query(self, text, params=None, timeout=None):
        self.calls += 1
        if self.fault is not None:
            raise self.fault()
        return _Result(self.rows(params))


def real_client(db, builder=None):
    """The production ``Client`` over the production ``CypherReader``; only the db is a double."""
    kw = {"builder": builder} if builder else {}
    return Client("gridsys", host="127.0.0.1", port=1, reader_factory=lambda system, host, port: CypherReader(db, system), **kw)


def ask_factory_of(client):
    return lambda built, snap: R.make_ask(client, stamp=STAMP, in_graph=lambda req: True)


def answers(e):
    f = list(e.cust.glob("run-*.answers.jsonl"))
    return [json.loads(l) for l in f[0].read_text().splitlines()] if f else []


def stop(e, ask_factory, *, canary_ok=True, **kw):
    """One process lifetime of the run: -> the state it ended in (``done`` when it completed)."""
    first = not e.markers()

    def canary():
        if not canary_ok:
            raise TimeoutError("canary deadline")
    try:
        e.go(ask_factory=ask_factory, canary=canary, resume=not first, **kw)
    except R.RunFailed as exc:
        return exc.state
    return "done"


def env(tmp_path):
    e = Env(tmp_path)
    many_world(e, N_ANCHORS)
    return e


# ------------------------------------------------------------------------------------------------ the table, one case per row
#: ``kind``: transient (``failed:transient``, nothing recorded, resumable) | stall (the repeat-stall row) | run-wide (``failed:deterministic``, nothing
#: recorded) | engine-error (a per-anchor miss; ``cls`` is the chain's innermost class name).
def _cp_row(params):
    return [[f"cp:{params['key']}", "http", params["key"]]]


FAMILIES = [
    pytest.param(1, "transient", lambda: RE.ConnectionError("Error 111 connecting"), None, None, id="1-redis-ConnectionError"),
    pytest.param(2, "stall", hung_server, None, None, id="2-redis-TimeoutError-hung-server"),
    pytest.param(3, "transient", lambda: RE.BusyLoadingError("Redis is loading the dataset in memory"), None, None, id="3-BusyLoadingError"),
    pytest.param(4, "transient", lambda: OSError(errno.ENOSPC, "No space left on device"), None, None, id="4-OSError-ENOSPC"),
    pytest.param(4, "transient", lambda: OSError(errno.EDQUOT, "Disk quota exceeded"), None, None, id="4-OSError-quota"),
    pytest.param(4, "transient", lambda: OSError(errno.EMFILE, "Too many open files"), None, None, id="4-OSError-fd-limit"),
    pytest.param(5, "engine-error", lambda: RE.ResponseError("Query timed out"), None, "ResponseError", id="5-Query-timed-out"),
    pytest.param(6, "run-wide", lambda: RE.AuthenticationError("invalid username-password pair or user is disabled."), None, None, id="6-AuthenticationError"),
    pytest.param(6, "run-wide", lambda: RE.AuthorizationError("not authorized"), None, None, id="6-AuthorizationError"),
    pytest.param(6, "run-wide", lambda: RE.AuthenticationWrongNumberOfArgsError("wrong number of arguments for AUTH"), None, None,
                 id="6-AuthenticationWrongNumberOfArgsError"),
    pytest.param(7, "run-wide", lambda: RE.NoPermissionError("this user has no permissions to run the 'graph.ro_query' command"), None, None, id="7-NoPermissionError"),
    pytest.param(8, "run-wide", lambda: RE.ReadOnlyError("You can't write against a read only replica."), None, None, id="8-ReadOnlyError"),
    pytest.param(9, "engine-error", lambda: RE.ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value"), None, "ResponseError", id="9-WRONGTYPE"),
    pytest.param(9, "engine-error", lambda: RE.ResponseError("Type mismatch: expected Integer but was String"), None, "ResponseError", id="9-Cypher-error"),
    pytest.param(10, "engine-error", None, _cp_row, "AnswerError", id="10-answer-does-not-validate"),
]


def fixture_of(fault, rows):
    """-> (db, builder): row 10's db answers (the anchor resolves) and the builder returns an answer the real ``answer.validate`` refuses."""
    db = FaultDb(fault=fault, rows=rows)
    return db, ((lambda reader, anchor, question=None: {}) if rows else None)


def test_every_row_of_the_table_has_a_case():
    text = PLAN.read_text(encoding="utf-8")
    table = text[text.index("Disposition table"):]
    rows = {int(m.group(1)) for m in re.finditer(r"^\s*\| (\d+) \|", table, re.M)}
    covered = {p.values[0] for p in FAMILIES}
    assert rows and rows == covered, f"the plan's table has rows {sorted(rows)}; the grid covers {sorted(covered)}"


@pytest.mark.parametrize("row,kind,fault,rows,cls", FAMILIES)
def test_the_real_client_wraps_every_family_in_StoreError_and_the_policy_reads_the_chain(row, kind, fault, rows, cls):
    """The premise of the grid: the outer type is ``StoreError`` for every row (so a rule on the outer type is dead), and the policy decides on the chain."""
    from clients.common import StoreError
    db, builder = fixture_of(fault, rows)
    ask = R.make_ask(real_client(db, builder), stamp=STAMP, in_graph=lambda req: True)
    req = {"useCase": "impact", "anchors": [{"type": "contract", "kind": "http", "keyNorm": "GET /a0"}]}
    try:
        ask(req)
    except R.EngineAskError as exc:
        assert kind == "engine-error" and isinstance(exc.__cause__, StoreError)
        assert R.ask_failure_entry(exc)["exceptionClass"] == cls
    except StoreError as exc:
        assert kind in ("transient", "stall", "run-wide")
        assert R.ask_failure_entry(exc) is None
        assert R.classify(exc) == ("deterministic" if kind == "run-wide" else "transient")
    else:
        pytest.fail("the fault did not reach the caller")


@pytest.mark.parametrize("row,kind,fault,rows,cls", FAMILIES)
def test_the_disposition_of_each_row_through_make_ask_persisting_and_the_marker(tmp_path, row, kind, fault, rows, cls):
    e = env(tmp_path)
    db, builder = fixture_of(fault, rows)
    factory = ask_factory_of(real_client(db, builder))
    if kind == "transient":
        assert stop(e, factory) == "failed:transient"
        assert e.marker()["state"] == "failed:transient" and e.marker()["stall"] is None and answers(e) == [] and not e.run_files()
        assert stop(e, factory) == "failed:transient"                                          # resumable: not refused, not spent
        assert answers(e) == []
    elif kind == "run-wide":
        assert stop(e, factory) == "failed:deterministic"
        assert e.marker()["state"] == "failed:deterministic" and answers(e) == [] and not e.run_files()    # no miss, no recall claim
        with pytest.raises(G.Refusal):
            stop(e, factory)                                                                   # a spent list: --resume is refused
    elif kind == "engine-error":
        assert stop(e, factory) == "done" and e.marker()["state"] == "asked"
        got = answers(e)
        assert len(got) == N_ANCHORS and all(x["outcome"] == "engine-error" and x["exceptionClass"] == cls and "via" not in x for x in got)
        assert all(set(x) == {"anchorId", "outcome", "exceptionClass"} for x in got)           # class name only: no message, no request
    else:                                                                                      # row 2: the repeat-stall rule, on the real hang's chain
        assert stop(e, factory) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 1} and answers(e) == []
        assert stop(e, factory) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 2} and answers(e) == []
        assert stop(e, factory) == "failed:transient"                                          # stop 3: the first anchor is converted, the second one stalls
        first = answers(e)
        assert len(first) == 1 and {k: v for k, v in first[0].items() if k != "anchorId"} == \
            {"outcome": "engine-error", "exceptionClass": "TimeoutError", "via": "repeat-stall", "stalls": 3}
        assert e.marker()["stall"] == {"index": 1, "count": 1}


def test_row_2_with_the_canary_failing_stays_failed_transient_forever(tmp_path):
    e = env(tmp_path)
    factory = ask_factory_of(real_client(FaultDb(fault=hung_server)))
    for _ in range(6):
        assert stop(e, factory, canary_ok=False) == "failed:transient"
    assert answers(e) == [] and e.marker()["stall"] is None and not e.run_files()


# ------------------------------------------------------------------------------------------------ the precedence: run-wide > connection/OS/host > timeout > rest
def _timeout_over(inner):
    """A hung-server timeout whose cause chain also holds ``inner`` (a second fault on the same path)."""
    def make():
        return implicit(RE.TimeoutError("Timeout reading from socket"), implicit(TimeoutError("timed out"), inner()))
    return make


PRECEDENCE = [
    pytest.param(_timeout_over(lambda: RE.ConnectionError("reset")), "failed:transient", id="timeout-over-ConnectionError-never-converts"),
    pytest.param(_timeout_over(lambda: ConnectionResetError("reset")), "failed:transient", id="timeout-over-ConnectionResetError-never-converts"),
    pytest.param(_timeout_over(lambda: OSError(errno.ENOSPC, "full")), "failed:transient", id="timeout-over-ENOSPC-never-converts"),
    pytest.param(_timeout_over(lambda: RE.BusyLoadingError("loading")), "failed:transient", id="timeout-over-BusyLoading-never-converts"),
    pytest.param(_timeout_over(lambda: RE.NoPermissionError("no perms")), "failed:deterministic", id="run-wide-NoPermission-beats-timeout"),
    pytest.param(_timeout_over(lambda: RE.ReadOnlyError("readonly")), "failed:deterministic", id="run-wide-ReadOnly-beats-timeout"),
    pytest.param(_timeout_over(lambda: RE.AuthenticationError("refused")), "failed:deterministic", id="run-wide-Authentication-beats-timeout"),
    pytest.param(lambda: implicit(RE.AuthenticationError("refused"), OSError(errno.ENOSPC, "full")), "failed:deterministic", id="run-wide-beats-OSError"),
    pytest.param(lambda: implicit(OSError(errno.ENOSPC, "full"), RE.ResponseError("Type mismatch")), "failed:transient", id="host-fault-beats-a-plain-response-error"),
]


@pytest.mark.parametrize("fault,state", PRECEDENCE)
def test_precedence_over_the_whole_chain_and_a_non_timeout_never_converts(tmp_path, fault, state):
    e = env(tmp_path)
    factory = ask_factory_of(real_client(FaultDb(fault=fault)))
    for _ in range(5 if state == "failed:transient" else 1):
        assert stop(e, factory) == state
    assert answers(e) == [] and not e.run_files() and e.marker()["state"] == state
    assert e.marker()["stall"] is None                                                                  # no precedence case is counted as a stall
    assert R.converts_on_repeat_stall(fault()) is False


@pytest.mark.parametrize("blocker", [MemoryError, RecursionError, KeyboardInterrupt, R._Terminated], ids=lambda c: c.__name__)
def test_a_timeout_raised_while_handling_a_process_fault_or_an_interrupt_is_not_a_stall(blocker):
    assert R.converts_on_repeat_stall(implicit(TimeoutError("t"), blocker())) is False
    assert R.converts_on_repeat_stall(implicit(RE.TimeoutError("t"), blocker())) is False


LOOKALIKES = [type(name, (Exception,), {}) for name in ("TimeoutError", "ConnectionError", "BusyLoadingError")]      # not module-level names: they would shadow the builtins here


@pytest.mark.parametrize("cls", LOOKALIKES, ids=lambda c: c.__name__)
def test_a_class_that_only_shares_the_name_of_a_driver_fault_is_not_one(cls):
    """Only the driver's (``redis``) classes and the builtins are placed by name: a product's own ``TimeoutError`` is the product's."""
    from clients.common import StoreError
    exc = explicit(StoreError("x"), cls("mine"))
    assert R.classify(exc) == "deterministic" and R.converts_on_repeat_stall(exc) is False
    wrapped = explicit(R.EngineAskError("StoreError"), exc)
    assert R.ask_failure_entry(wrapped) == {"outcome": "engine-error", "exceptionClass": cls.__name__}


# ------------------------------------------------------------------------------------------------ the chain walk itself
def _linked(n, *, innermost):
    """``n`` exceptions, each raised while handling the next (``__context__``), the last one ``innermost``; -> the outermost."""
    cur = innermost
    for i in range(n - 1):
        cur = implicit(RuntimeError(f"link {i}"), cur)
    return cur


def test_the_chain_walk_follows_cause_and_context_outermost_first():
    try:
        try:
            raise ValueError("inner")
        except ValueError as v:
            raise KeyError("middle") from v
    except KeyError as k:
        mid = k
    out = implicit(OSError("outer"), mid)
    assert [type(c).__name__ for c in R.exception_chain(out)] == ["OSError", "KeyError", "ValueError"]


def test_the_chain_walk_is_bounded_and_survives_a_cycle():
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__context__, b.__context__ = b, a
    assert R.exception_chain(a) == [a, b]
    deep = _linked(40, innermost=RE.NoPermissionError("deep"))
    assert len(R.exception_chain(deep)) == R.CHAIN_LIMIT == 16


def test_a_fault_inside_the_bound_is_seen_one_beyond_it_is_not():
    """The bound is part of the rule (~16 links): a store error whose host fault is the 16th link is transient, the 17th is not walked (fail closed)."""
    from clients.common import StoreError
    host = OSError(errno.ENOSPC, "full")
    assert R.classify(implicit(StoreError("x"), _linked(15, innermost=host))) == "transient"           # 1 + 15 links: the host fault is the 16th
    assert R.classify(implicit(StoreError("x"), _linked(16, innermost=host))) == "deterministic"      # the 17th
    assert R.classify(implicit(StoreError("x"), _linked(15, innermost=RE.NoPermissionError("p")))) == "deterministic"


def test_a_resume_that_fails_before_its_marker_write_says_nothing_about_the_store(tmp_path, monkeypatch, capsys):
    """No build ran in a resuming process, so ``STORE_REMEDY`` ("the built graph stays in this store") would be false there."""
    e = env(tmp_path)
    factory = ask_factory_of(real_client(FaultDb(fault=lambda: RE.ConnectionError("down"))))
    assert stop(e, factory) == "failed:transient"
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(e.argv("--resume"), deps=e.deps(engine_snapshot_id=lambda: "b" * 32))
    err = capsys.readouterr().err
    assert rc == R.EXIT_REFUSED and "differs" in err and "new empty store" not in err


def explicit(outer, inner):
    """``outer`` raised ``from inner`` (``__cause__``): the other way a chain is linked (a client that wrapped with ``from exc`` instead of ``from None``)."""
    try:
        raise outer from inner
    except Exception as e:                                                             # noqa: BLE001
        return e


def test_an_explicit_cause_links_the_chain_like_an_implicit_context():
    from clients.common import StoreError
    assert R.classify(explicit(StoreError("x"), OSError(errno.ENOSPC, "full"))) == "transient"
    assert R.classify(explicit(StoreError("x"), RE.NoPermissionError("p"))) == "deterministic"
    assert R.classify(explicit(StoreError("x"), RuntimeError("y"))) == "deterministic"
    assert R.converts_on_repeat_stall(explicit(StoreError("x"), TimeoutError("t"))) is True
    assert R.converts_on_repeat_stall(explicit(StoreError("x"), implicit(TimeoutError("t"), ConnectionError("c")))) is False
    assert [type(c).__name__ for c in R.exception_chain(explicit(StoreError("x"), OSError("o")))] == ["StoreError", "OSError"]


def test_only_a_store_error_is_transient_by_what_it_wrapped_any_other_outer_type_is_the_products():
    """A decision pinned for the method owner (reported, not ruled): the table's rows are all ``StoreError`` outers; an engine that raises its OWN type while
    handling a host fault has handled it, so the fault is the product's (``engine-error``), as ``classify`` always read it."""
    own = implicit(ValueError("the product's own"), OSError(errno.ENOSPC, "full"))
    assert R.classify(own) == "deterministic"
    assert R.ask_failure_entry(explicit(R.EngineAskError("ValueError"), own)) == {"outcome": "engine-error", "exceptionClass": "OSError"}
