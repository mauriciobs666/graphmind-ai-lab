"""P1-32b round D: three coverage grids, each enumerated from the CODE'S OWN registries (an AST scan, ``errno.errorcode``, the ``builtins`` and
``redis.exceptions`` and ``clients.common`` exception classes, a call trace), not from a hand list. A cell with no plant, or a plant that reddens, fails;
a NEW writer call site, exception class or ``score`` refusal that no table entry accounts for fails too (the entry forces a decision).

1. marker write sites x SIGKILL points: every static call site of a marker/run-file writer is exercised, and a process killed right before and right after
   each dynamic call of each writer, in each run scenario, leaves a state from which the next command finishes the run and ``score`` reads it.
2. exception families x ``ask_failure_entry``: every errno, builtin, redis and ``ClientError`` class, bare and as the cause of an engine error.
3. ``score`` refusals x ``run``: every refusal reachable from ``gate.main score`` is either raised by a function ``run`` also calls before it takes the
   marker (traced), or accounted for as depending on something that exists only after the run."""
import ast
import builtins
import errno
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world

import gate_run as R
from test_gate_run_crash import in_child, was_sigkilled, score, stall_env, stop, suicide, kill_after_rename_of

TOOLS = Path(G.__file__).parent
WRITERS = ("_write_marker", "first_run_marker", "_atomic_write")


# ================================================================================================ 1. marker writers x SIGKILL points
def static_sites():
    """``{(file, line, enclosing function, writer)}`` over every call of a writer in ``gate.py`` and ``gate_run.py`` (the definitions excluded)."""
    out = set()
    for name in ("gate.py", "gate_run.py"):
        tree = ast.parse((TOOLS / name).read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for c in ast.walk(fn):
                if isinstance(c, ast.Call):
                    f = c.func
                    callee = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                    if callee in WRITERS and fn.name not in WRITERS:
                        out.add((name, c.lineno, fn.name, callee))
                    elif callee in WRITERS and fn.name in WRITERS and fn.name != callee:
                        out.add((name, c.lineno, fn.name, callee))              # a writer calling another writer (``_write_marker`` -> ``_atomic_write``)
    return out


class Tracer:
    """Wraps the three writers; counts the calls, remembers the caller's (file, line) of each, and kills the process at (writer, index, before|after)."""

    def __init__(self, kill=None):
        self.kill, self.count, self.sites = kill, {w: 0 for w in WRITERS}, set()

    def install(self):
        for w in WRITERS:
            mod = G if w == "first_run_marker" else R
            real = getattr(mod, w)

            def wrapped(*a, _w=w, _real=real, **k):
                fr = sys._getframe(1)
                self.sites.add((Path(fr.f_code.co_filename).name, fr.f_lineno, fr.f_code.co_name, _w))
                i = self.count[_w]
                self.count[_w] += 1
                if self.kill == (_w, i, "before"):
                    suicide()
                out = _real(*a, **k)
                if self.kill == (_w, i, "after"):
                    suicide()
                return out
            setattr(mod, w, wrapped)


def recover(e, stalling=False):
    """The next command(s) an operator would run: ``run`` when there is no marker, ``--resume`` while it is not ``asked``. -> nothing; asserts it ends."""
    for _ in range(8):
        if e.markers() and e.marker()["state"] == "asked":
            return
        first = not e.markers()
        kw = {"ask_factory": e.stall_ask, "canary": e.canary} if stalling else {}
        try:
            e.go(resume=not first, **kw)
        except R.RunFailed as exc:
            assert exc.state == "failed:transient"
    raise AssertionError("the run could not be finished")


def scenario_env(tmp, name):
    """-> (env, phase): ``phase()`` is the command the kill is planted in; the state before it is prepared here, in this process."""
    e = Env(tmp)
    many_world(e, 3)
    stalling = name.startswith("stall")
    if stalling:
        e = stall_env(tmp / "s")
    if name == "fresh":
        return e, lambda: e.go(), False
    if name == "resume-after-kill":
        assert was_sigkilled(in_child(lambda: (setattr(e, "ask_hook", lambda req, n: suicide() if n == 2 else None), e.go())))
        return e, lambda: e.go(resume=True), False
    if name == "resume-after-transient":
        e.ask_hook = lambda req, n: (_ for _ in ()).throw(ConnectionError("c")) if n == 2 else None
        with pytest.raises(R.RunFailed):
            e.go()
        e.ask_hook = None
        return e, lambda: e.go(resume=True), False
    if name == "finalise":
        key = G.list_key([json.loads(l) for l in e.lists.read_text().splitlines()])
        assert was_sigkilled(in_child(lambda: (kill_after_rename_of(f"run-{key}.json"), e.go())))
        return e, lambda: e.go(resume=True), False
    if name == "stall-first-stop":
        return e, lambda: stop(e), True
    if name == "stall-conversion":
        assert stop(e) == "failed:transient" and stop(e) == "failed:transient"
        return e, lambda: stop(e), True
    raise AssertionError(name)


SCENARIOS = ["fresh", "resume-after-kill", "resume-after-transient", "finalise", "stall-first-stop", "stall-conversion"]


def cells(tmp_path_factory):
    """Dry-run each scenario to learn how many times each writer is called and from where -> the grid of kill cells and the dynamic call sites."""
    out, seen = [], set()
    for sc in SCENARIOS:
        e, phase, stalling = scenario_env(tmp_path_factory.mktemp(sc), sc)
        t = Tracer()

        def dry():
            t.install()
            phase()
        assert in_child_exit(dry, e, t) == 0
        seen |= t.reported
        for w, n in t.reported_counts.items():
            out += [(sc, w, i, pos) for i in range(n) for pos in ("before", "after")]
    return out, seen


def in_child_exit(fn, e, t):
    """Run ``fn`` in a child and bring ``t``'s counters back through a pipe (the child's memory is not the parent's)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            fn()
        except BaseException:                                                          # noqa: BLE001
            code = 70
        os.write(w, json.dumps({"counts": t.count, "sites": sorted(t.sites)}).encode())
        os._exit(code)
    os.close(w)
    data = b""
    while chunk := os.read(r, 65536):
        data += chunk
    _, st = os.waitpid(pid, 0)
    doc = json.loads(data)
    t.reported_counts = doc["counts"]
    t.reported = {tuple(s) for s in doc["sites"]}
    return os.WEXITSTATUS(st) if os.WIFEXITED(st) else -1


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    return cells(tmp_path_factory)


def test_grid1_every_static_writer_call_site_is_exercised_by_some_scenario(grid):
    _, dynamic = grid
    static = static_sites()
    assert static, "the AST scan found no writer call site: the registry is broken"
    missing = sorted(s for s in static if not any(d[0] == s[0] and d[3] == s[3] and d[1] == s[1] for d in dynamic))
    assert not missing, f"writer call sites no scenario reaches (add a scenario): {missing}"


def test_grid1_the_scan_sees_the_known_writers():
    assert {w for *_, w in static_sites()} == set(WRITERS)


def test_grid1_every_kill_cell_leaves_a_state_the_next_command_finishes_and_score_reads(grid, tmp_path_factory, monkeypatch, capsys):
    cell_list, _ = grid
    assert len(cell_list) >= 30
    for sc, w, i, pos in cell_list:
        e, phase, stalling = scenario_env(tmp_path_factory.mktemp("cell"), sc)
        t = Tracer(kill=(w, i, pos))

        def planted():
            t.install()
            phase()
        status = in_child(planted)
        assert was_sigkilled(status), f"{sc}/{w}#{i}/{pos}: the plant did not fire"
        for m in e.markers():                                                           # never empty, never torn
            doc = json.loads(m.read_text())
            assert doc["state"] and "pins" in doc, f"{sc}/{w}#{i}/{pos}: marker is not whole"
        recover(e, stalling)
        assert e.marker()["state"] == "asked", f"{sc}/{w}#{i}/{pos}"
        rc, rep, err = score(e, monkeypatch, capsys)
        assert rep is not None, f"{sc}/{w}#{i}/{pos}: score refused: {err}"


# ================================================================================================ 2. exception families x ask_failure_entry
def make(cls):
    for args in ((), ("x",), (1, "x"), ("x", b"", 0, 1, "y"), ("x", "y", "z", 1, 2), ("x", ("f", 1, 1, "t"))):
        try:
            return cls(*args)
        except Exception:                                                               # noqa: BLE001
            continue
    try:
        return cls.__new__(cls)                                                         # a class whose constructor wants keywords
    except Exception:                                                                   # noqa: BLE001
        return None


def wrapped(cause):
    from test_gate_run_crash import engine
    return engine(cause)


def test_grid2_every_errno_is_a_host_fault_never_recorded_never_an_engine_error():
    assert len(errno.errorcode) > 80
    for code in errno.errorcode:
        exc = OSError(code, "x")
        assert R.ask_failure_entry(exc) is None and R.ask_failure_entry(wrapped(exc)) is None, errno.errorcode[code]


def test_grid2_every_builtin_exception_class():
    seen = 0
    for name, cls in sorted(vars(builtins).items()):
        if not (isinstance(cls, type) and issubclass(cls, BaseException)):
            continue
        exc = make(cls)
        if exc is None:
            continue
        seen += 1
        host = isinstance(exc, (OSError, MemoryError, RecursionError)) or isinstance(exc, KeyboardInterrupt)
        if isinstance(exc, Exception):
            bare, wr = R.ask_failure_entry(exc), R.ask_failure_entry(wrapped(exc))
            if host:
                assert bare is None and wr is None, name
            else:
                assert bare == {"harnessError": name} and wr == {"outcome": "engine-error", "exceptionClass": name}, name
    assert seen > 40


@pytest.mark.parametrize("cls", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_grid2_a_base_exception_is_never_recorded_as_an_anchors_entry(tmp_path, cls):
    path = tmp_path / "answers.jsonl"

    def ask(req):
        raise cls()
    p = R._Persisting(ask, path, {})
    with pytest.raises(cls):
        p({"anchors": [{"type": "contract", "cpId": "c" * 32}]})
    assert not path.exists()


def test_grid2_every_redis_exception_class():
    import redis.exceptions as RE
    transient = {"ConnectionError", "TimeoutError", "BusyLoadingError"}
    auth = {"AuthenticationError", "AuthorizationError", "AuthenticationWrongNumberOfArgsError"}
    seen = 0
    for name, cls in sorted(vars(RE).items()):
        if not (isinstance(cls, type) and issubclass(cls, Exception)) or cls.__module__ != RE.__name__:
            continue
        exc = make(cls)
        if exc is None:
            continue
        seen += 1
        names = {c.__name__ for c in cls.__mro__}
        if isinstance(exc, OSError) or names & (transient | auth):
            assert R.ask_failure_entry(exc) is None and R.ask_failure_entry(wrapped(exc)) is None, f"{name}: a connection, timeout or credential fault is the run's"
        else:
            assert R.ask_failure_entry(exc) == {"harnessError": name} and R.ask_failure_entry(wrapped(exc))["outcome"] == "engine-error", name
    assert seen > 10


CLIENT_ERROR_RUN_WIDE = {"NoCredentials", "StoreAuthError", "ModelUnavailable", "NotBuilt"}
CLIENT_ERROR_PLACED = {"ClientError", "BadRequest", "AnchorNotResolved", "StoreError", "RawRefused", "ResultTooLarge", "RefreshFailed"}


def client_error_classes():
    from clients import common
    out, stack = set(), [common.ClientError]
    while stack:
        c = stack.pop()
        out.add(c)
        stack += c.__subclasses__()
    return out


def test_grid2_every_client_error_class_is_accounted_for_run_wide_or_placed():
    classes = client_error_classes()
    unknown = sorted(c.__name__ for c in classes if c.__name__ not in CLIENT_ERROR_RUN_WIDE | CLIENT_ERROR_PLACED)
    assert not unknown, f"a new ClientError subclass: decide whether it is run-wide or the engine's, and add it to the table: {unknown}"
    for c in classes:
        exc = make(c)
        if c.__name__ in CLIENT_ERROR_RUN_WIDE:
            assert R.ask_failure_entry(exc) is None and R.ask_failure_entry(wrapped(exc)) is None, c.__name__
        else:
            assert R.ask_failure_entry(wrapped(exc))["outcome"] == "engine-error", c.__name__
            assert R.ask_failure_entry(exc)["harnessError"] == c.__name__, c.__name__


# ================================================================================================ 3. score refusals x run
def score_refusal_sites():
    """``{(function, message prefix)}`` of every ``raise Refusal/PrecheckRefusal`` in ``gate.main`` and in every ``gate.py`` function reachable from it."""
    tree = ast.parse((TOOLS / "gate.py").read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    seen, stack = set(), ["main"]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        stack += [c.func.id for c in ast.walk(funcs[f]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in funcs]
    out = set()
    for f in seen:
        for n in ast.walk(funcs[f]):
            if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and getattr(n.exc.func, "id", "") in ("Refusal", "PrecheckRefusal"):
                a = n.exc.args[0] if n.exc.args else None
                text = a.value if isinstance(a, ast.Constant) else "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in a.values) if isinstance(a, ast.JoinedStr) else "?"
                out.add((f, text[:40]))
    return out


#: refusals that depend on something that exists only AFTER the run (its files, the live store), or are the score verb's own output guard
AFTER_THE_RUN = {
    ("main", "the run file is not the run of these lis"): "the run file's list key",
    ("main", "the run has no marker in state `asked` f"): "the marker's state after the run",
    ("main", "the live :Snapshot node differs from the"): "the live store compared with the run's pin, at score time",
    ("main", "the run file carries no recorded flow fl"): "the run file's content",
    ("main", "the run file's flow flag differs from th"): "the run file against the marker's pin",
    ("main", "the lists do not reproduce the withheld "): "the run file's withheld counts",
    ("require_run_context", "the run file carries no preflight block:"): "the run file's content",
    ("decide_gate", "{} scored items lie in a stratum that is"): "the scored items; withheld strata never reach them by construction",
    ("write_custodian", "held-out-derived files go under the cust"): "score's output guard; run guards the same root with within_custodian",
}
#: the same function is called by run before the marker, but the refusal needs an argument only score has
SCORE_ARGUMENT = {
    ("score_inputs_check", "the committed record is a re-gate: --flags is required"[:40]): "--flags",
    ("score_inputs_check", "the committed record is gate 1: --flags is refused"[:40]): "--flags",
}


def traced_pre_marker_calls(tmp_path):
    """The names of the ``gate.py`` functions called by ``run`` before it takes the marker (and by ``_default_deps`` for its guards)."""
    import argparse
    e = Env(tmp_path)
    called, live = set(), {"on": True}
    gate_file = G.__file__

    def prof(frame, event, arg):
        if event == "call" and live["on"] and frame.f_code.co_filename == gate_file:
            if frame.f_code.co_name == "first_run_marker":
                live["on"] = False
            else:
                called.add(frame.f_code.co_name)

    sys.setprofile(prof)
    try:
        e.go()
        live["on"] = True                                                               # the production wiring's own guards, before any connection
        R._default_deps(argparse.Namespace(system="ccg", host="127.0.0.1", port=16420, graph=None))
    finally:
        sys.setprofile(None)
    return called


def test_grid3_every_score_refusal_is_raised_by_run_before_its_marker_or_accounted_for(tmp_path):
    sites, called = score_refusal_sites(), traced_pre_marker_calls(tmp_path)
    assert len(sites) >= 20 and "score_inputs_check" in called
    unaccounted = sorted(s for s in sites if s[0] not in called and s not in AFTER_THE_RUN and s not in SCORE_ARGUMENT)
    assert not unaccounted, f"a refusal of score that run does not evaluate before its marker, and no table entry says why: {unaccounted}"
    stale = sorted(k for k in set(AFTER_THE_RUN) | set(SCORE_ARGUMENT) if k not in sites)
    assert not stale, f"table entries for refusals that no longer exist: {stale}"
    reached = sorted(s for s in AFTER_THE_RUN if s[0] in called)
    assert not reached, f"accounted as after-the-run but run already reaches them (drop the entry): {reached}"


@pytest.mark.parametrize("extra,match", [({"kind": __import__("gate_run_helpers").DROP}, "gate kind")])
def test_grid3_a_pre_marker_refusal_plant_stops_run_with_nothing_spent_and_score_refuses_the_same_input(tmp_path, extra, match):
    e = Env(tmp_path, record_extra=extra)
    with pytest.raises(G.PrecheckRefusal, match=match):
        e.go()
    assert e.markers() == [] and e.events == []
