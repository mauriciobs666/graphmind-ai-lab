"""CO-5 / P5-3 (``docs/reviews/connected-code-graph-p1-32b-impl.md`` Pass 5): ``score --repo`` refuses a checkout whose record commit is not the one the
run recorded (a clean checkout at another commit would score the same answers under other scope patterns). Offline."""
import json

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world
from test_gate_run_crash import GATE, OBS


def score(e, monkeypatch, capsys, mutate=None):
    monkeypatch.setenv("HOME", str(e.home))

    def pf(repo, gid):
        out = G.preflight(repo, gid, observed=OBS)
        return mutate(out) if mutate else out
    argv = ["score", "--gate-id", GATE, "--lists", str(e.lists), "--run", str(e.run_files()[0]), "--custodian", str(e.cust), "--dev-probe", str(e.dev_probe), "--port", "16420"]
    rc = G.main(argv, snapshot=e.snap, repo=e.repo, preflight_fn=pf)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_score_at_the_run_s_record_commit_is_unchanged(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    e.go()
    rc, out, err = score(e, monkeypatch, capsys)
    assert rc in (0, 1) and "record commit" not in err and json.loads(out)["verdict"]


def test_score_refuses_a_checkout_whose_record_commit_differs_from_the_run_s(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    e.go()
    rc, out, err = score(e, monkeypatch, capsys, mutate=lambda pf: {**pf, "recordCommit": "f" * 40})
    assert rc == 2 and "record commit" in err and out == ""
    assert not (e.cust / "gate-report.json").exists()
