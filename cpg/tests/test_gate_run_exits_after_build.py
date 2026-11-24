"""P1-32b Pass 4, P4-3: every exit between the build and the marker leaves the store holding the built live graph (the next fresh run is refused by the
live-graph probe), so every one of them tells the operator what to do with the store (``gate.STORE_REMEDY``), whatever its exit status. The plants go through
``gate_run.main`` with injected deps, as the analyst's did. A fault BEFORE the build, and a failure AFTER the marker (resumable), need no such text. Offline."""

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world

import gate_run as R

REMEDY_WORDS = ("new empty store", "ACL", "rebuild", "not spent")


def has_remedy(text: str) -> bool:
    return all(w in text for w in REMEDY_WORDS)


def boom(exc):
    def f(*a, **k):
        raise exc
    return f


def snapshot_factory_of(e, **kw):
    return lambda built: e.snap


def run(e, monkeypatch, capsys, **over):
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(e.argv(), deps=e.deps(**over))
    cap = capsys.readouterr()
    return rc, cap.err


# ------------------------------------------------------------------------------------------------ the six planted exits, one per site
def test_engine_snapshot_id_mismatch_exit_2_names_the_remedy(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    rc, err = run(e, monkeypatch, capsys, engine_snapshot_id=lambda: "b" * 32)
    assert rc == R.EXIT_REFUSED and "differs" in err and has_remedy(err) and not e.markers()


def test_a_non_empty_cache_exit_2_names_the_remedy(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    rc, err = run(e, monkeypatch, capsys, cache_len=lambda: 3)
    assert rc == R.EXIT_REFUSED and "cache is not empty" in err and has_remedy(err) and not e.markers()


@pytest.mark.parametrize("site", ["snapshot_factory", "engine_snapshot_id", "resolve_factory", "ask_factory"])
@pytest.mark.parametrize("exc", [ConnectionError("down"), RuntimeError("boom")])
def test_an_injected_fault_after_the_build_exit_4_names_the_remedy(tmp_path, monkeypatch, capsys, site, exc):
    e = Env(tmp_path)
    many_world(e, 1)
    rc, err = run(e, monkeypatch, capsys, **{site: boom(exc)})
    assert rc == R.EXIT_BAD and f"bad input: {type(exc).__name__}" in err and has_remedy(err) and not e.markers()
    assert "down" not in err and "boom" not in err                                                  # the class name only, as before


def test_an_interrupt_between_the_build_and_the_marker_names_the_remedy(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    rc, err = run(e, monkeypatch, capsys, resolve_factory=boom(KeyboardInterrupt()))
    assert rc == R.EXIT_RESUMABLE and has_remedy(err) and not e.markers()


def test_a_refusal_that_already_names_the_remedy_names_it_once(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    e.overrides = {k: {"status": "unresolved", "candidates": []} for k in e.table}
    rc, err = run(e, monkeypatch, capsys)
    assert rc == R.EXIT_RESOLVE and err.count("new empty store") == 1


# ------------------------------------------------------------------------------------------------ and no remedy where the store is not stranded
def test_a_fault_before_the_build_says_nothing_about_the_store(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    rc, err = run(e, monkeypatch, capsys, refresh_fn=boom(ConnectionError("down")))
    assert rc == R.EXIT_BAD and "new empty store" not in err


def test_a_failure_after_the_marker_is_resumable_and_says_nothing_about_the_store(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 2)
    e.ask_hook = lambda req, n: (_ for _ in ()).throw(ConnectionError("c")) if n == 2 else None
    rc, err = run(e, monkeypatch, capsys)
    assert rc == R.EXIT_RESUMABLE and "failed:transient" in err and "new empty store" not in err


def test_a_refusal_before_the_build_says_nothing_about_the_store(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    rc, err = run(e, monkeypatch, capsys, live_graph_present=lambda: False, engine_snapshot_id=lambda: e.engine_sid)
    assert rc == R.EXIT_OK
    rc, err = run(e, monkeypatch, capsys)                                                           # the list was run: refused before the build
    assert rc == R.EXIT_REFUSED and "already run" in err and "new empty store" not in err


def test_a_refusal_while_taking_the_marker_itself_names_the_remedy(tmp_path, monkeypatch, capsys):
    """The marker take is the last step of the window: a refusal there (a concurrent run won the ``O_EXCL``) still leaves the built store stranded."""
    e = Env(tmp_path)
    many_world(e, 1)

    def taken(*a, **k):
        raise G.Refusal("this list was already run (a concurrent run took the marker)")
    monkeypatch.setattr(G, "first_run_marker", taken)
    rc, err = run(e, monkeypatch, capsys)
    assert rc == R.EXIT_REFUSED and "concurrent run" in err and has_remedy(err)
