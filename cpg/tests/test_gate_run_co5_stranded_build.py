"""CO-5 / P5-1 (``docs/reviews/connected-code-graph-p1-32b-impl.md`` Pass 5): a failure INSIDE the build, after ``swap`` renamed the build key to the
live key, strands the store like any later exit does, so it carries ``gate.STORE_REMEDY``; a failure before the rename does not (the next fresh run is
clean), and the two are told apart by asking the live-graph probe after the failure. Offline; plants through ``gate_run.main`` with injected deps."""

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world

import gate_run as R

REMEDY_WORDS = ("new empty store", "ACL", "rebuild", "not spent")


def has_remedy(text: str) -> bool:
    return all(w in text for w in REMEDY_WORDS)


class Store:
    """The live-graph probe and a refresh double that sets it (the rename) and then raises ``exc`` (or raises before the rename when ``renamed`` is false)."""

    def __init__(self, exc, renamed):
        self.live, self.exc, self.renamed = False, exc, renamed

    def probe(self):
        return self.live

    def refresh(self, system, repo_root=None, **kw):
        if self.renamed:
            self.live = True
        raise self.exc


def run(e, monkeypatch, capsys, **over):
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(e.argv(), deps=e.deps(**over))
    return rc, capsys.readouterr().err


def planted(tmp_path, exc, renamed):
    e = Env(tmp_path)
    many_world(e, 1)
    s = Store(exc, renamed)
    return e, dict(refresh_fn=s.refresh, live_graph_present=s.probe)


def test_a_raise_after_the_rename_exit_4_names_the_remedy(tmp_path, monkeypatch, capsys):
    e, over = planted(tmp_path, RuntimeError("boom"), renamed=True)
    rc, err = run(e, monkeypatch, capsys, **over)
    assert rc == R.EXIT_BAD and "bad input: RuntimeError" in err and has_remedy(err) and not e.markers()
    assert "boom" not in err


def test_an_interrupt_after_the_rename_names_the_remedy(tmp_path, monkeypatch, capsys):
    e, over = planted(tmp_path, KeyboardInterrupt(), renamed=True)
    rc, err = run(e, monkeypatch, capsys, **over)
    assert rc == R.EXIT_RESUMABLE and has_remedy(err) and not e.markers()


def test_a_raise_before_the_rename_names_nothing_about_the_store(tmp_path, monkeypatch, capsys):
    e, over = planted(tmp_path, RuntimeError("boom"), renamed=False)
    rc, err = run(e, monkeypatch, capsys, **over)
    assert rc == R.EXIT_BAD and "bad input: RuntimeError" in err and "new empty store" not in err and not e.markers()


def test_an_interrupt_before_the_rename_names_nothing_about_the_store(tmp_path, monkeypatch, capsys):
    e, over = planted(tmp_path, KeyboardInterrupt(), renamed=False)
    rc, err = run(e, monkeypatch, capsys, **over)
    assert rc == R.EXIT_RESUMABLE and "new empty store" not in err


def test_a_probe_that_itself_fails_leaves_the_original_error_and_no_remedy(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    calls = []

    def probe():
        calls.append(1)
        if len(calls) > 1:
            raise ConnectionError("store gone")
        return False
    rc, err = run(e, monkeypatch, capsys, refresh_fn=Store(RuntimeError("x"), False).refresh, live_graph_present=probe)
    assert rc == R.EXIT_BAD and "bad input: RuntimeError" in err and "store gone" not in err
    assert "new empty store" not in err                                                        # a probe that cannot answer makes no claim about the store


def test_no_probe_injected_means_no_claim_about_the_store(tmp_path, monkeypatch, capsys):
    e, over = planted(tmp_path, RuntimeError("boom"), renamed=True)
    over["live_graph_present"] = None
    rc, err = run(e, monkeypatch, capsys, **over)
    assert rc == R.EXIT_BAD and "new empty store" not in err
