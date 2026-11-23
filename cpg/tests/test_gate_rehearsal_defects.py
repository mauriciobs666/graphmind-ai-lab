"""The dev dress rehearsal's defects D1-D4, D6, D7 (``docs/test-reports/connected-code-graph-p1-gate-rehearsal-report.md``), offline tier.

Every one of them was invisible to the offline suite because a fake stood where the real component stands. Each test here either runs the REAL parser or
constructor on the seam (``ccg.manifest.load_manifest``, ``clients.common.Client.raw_query``, ``ccg.refresh.RefreshReport``, ``extractors.indexers.run``'s audit
resolver, ``falkordb``'s constructor arguments) or states the contract of a seam the live tier (``test_gate_rehearsal_defects_live.py``) proves on a real
password-protected store. Synthetic data only."""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest
from redis.exceptions import ResponseError

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world, GATE, OBS

import gate_run as R

SYS = "gtd1"
CPG = Path(__file__).resolve().parents[1]
EMPTY_KEY = "Invalid graph operation on empty key"


# ================================================================================================ D1: no credential-less connection
class FakeFalkor:
    """``falkordb.FalkorDB``'s stand-in: records the credential every connection was opened with; one graph per key; ``ro_query`` answers from ``graphs``."""
    opened: list = []
    graphs: dict = {}

    def __init__(self, host=None, port=None, username=None, password=None, **kw):
        FakeFalkor.opened.append({"host": host, "port": port, "username": username, "password": password})

    def select_graph(self, key):
        outer = self

        class Graph:
            def ro_query(self, cypher, params=None, timeout=None):
                if key not in FakeFalkor.graphs:
                    raise ResponseError(EMPTY_KEY)
                return types.SimpleNamespace(result_set=FakeFalkor.graphs[key])

            def query(self, *a, **k):
                raise AssertionError("the read-only reader never issues GRAPH.QUERY")
        return Graph()

    def list_graphs(self):
        raise AssertionError("no reader role may run GRAPH.LIST (the ACL refuses it): the probe cannot use it")


@pytest.fixture
def falkor(monkeypatch):
    FakeFalkor.opened, FakeFalkor.graphs = [], {}
    monkeypatch.setattr("falkordb.FalkorDB", FakeFalkor)
    return FakeFalkor


def cred(system, role):
    return f"pw-{system}-{role}"


def test_d1_the_snapshot_reader_opens_the_store_as_the_read_only_raw_role_with_its_credential(falkor):
    G.snapshot_graph("127.0.0.1", 16431, None, SYS, credential=cred)
    assert falkor.opened == [{"host": "127.0.0.1", "port": 16431, "username": f"ccg_raw_{SYS}", "password": f"pw-{SYS}-raw"}]


def test_d1_the_reader_credential_is_never_a_write_role(falkor):
    G.snapshot_graph("127.0.0.1", 16431, None, SYS, credential=cred)
    assert falkor.opened[0]["username"].startswith("ccg_raw_") and not any(r in falkor.opened[0]["username"] for r in ("build", "verdict"))


def deps_for(falkor, **kw):
    a = argparse.Namespace(system=SYS, host="127.0.0.1", port=16431, graph=None)
    return R._default_deps(a, credential=cred, **kw)


def test_d1_every_default_connection_of_the_run_verb_carries_a_role_credential(falkor):
    falkor.graphs[f"ccg_{SYS}"] = [["s" * 32]]
    deps = deps_for(falkor)
    assert deps["live_graph_present"]() is True
    assert deps["snapshot_factory"](None).snapshot_id() == "s" * 32
    assert falkor.opened and all(o["username"] == f"ccg_raw_{SYS}" and o["password"] for o in falkor.opened), falkor.opened


def test_d1_the_live_graph_probe_is_a_read_only_query_because_no_reader_role_may_list_graphs(falkor):
    deps = deps_for(falkor)
    assert deps["live_graph_present"]() is False                                       # the server's "empty key" refusal is the absence
    falkor.graphs[f"ccg_{SYS}"] = [[0]]
    assert deps["live_graph_present"]() is True


def test_d1_a_refused_credential_is_not_read_as_an_absent_graph(falkor, monkeypatch):
    from redis.exceptions import AuthenticationError

    class Refusing(FakeFalkor):
        def select_graph(self, key):
            class Graph:
                def ro_query(self, *a, **k):
                    raise AuthenticationError("bad credential")                          # the driver's first command is where a refused AUTH surfaces
            return Graph()
    monkeypatch.setattr("falkordb.FalkorDB", Refusing)
    with pytest.raises(AuthenticationError):
        deps_for(falkor)["live_graph_present"]()


def test_d1_score_opens_its_snapshot_reader_with_the_same_role_credential(falkor, monkeypatch):
    monkeypatch.setattr("clients.common._credential", cred)
    G.snapshot_graph("127.0.0.1", 16431, None, SYS)                                    # the default factory of `gate.py score`
    assert falkor.opened[-1]["username"] == f"ccg_raw_{SYS}" and falkor.opened[-1]["password"] == f"pw-{SYS}-raw"


# ================================================================================================ D2: the manifest comes from the working tree, and is pinned
def test_d2_the_build_receives_the_head_manifest_though_the_clone_at_eval_commit_has_none(tmp_path):
    e = Env(tmp_path, late_manifest=True)
    seen = {}

    def refresh(system, repo_root=None, **kw):
        seen.update(root=Path(repo_root), **kw)
        seen["clone_has_manifest"] = (Path(repo_root) / "cpg/ccg/system.yaml").exists()
    e.go(refresh_fn=lambda *a, **k: (refresh(*a, **k), clean())[1])
    assert seen["clone_has_manifest"] is False                                         # the premise: EVAL_COMMIT predates the manifest
    assert Path(seen["manifest"]) == e.repo / "cpg" / "ccg" / "system.yaml" and Path(seen["manifest"]).is_file()
    assert seen["root"] != e.repo                                                       # the repository the build reads is still the clone


def test_d2_the_real_manifest_loader_accepts_what_the_build_is_given_and_names_the_derived_system():
    from ccg.manifest import load_manifest
    assert G.manifest_system(CPG.parent) == load_manifest(CPG / "ccg" / "system.yaml").system != "ccg"


def test_d2_the_manifest_sha256_is_a_marker_pin(tmp_path):
    import hashlib
    e = Env(tmp_path, late_manifest=True)
    e.go(refresh_fn=lambda *a, **k: clean())
    want = hashlib.sha256((e.repo / "cpg" / "ccg" / "system.yaml").read_bytes()).hexdigest()
    assert e.marker()["pins"]["manifestSha256"] == want


def test_d2_a_resume_under_another_manifest_is_refused(tmp_path):
    e = Env(tmp_path, late_manifest=True)
    many_world(e, 2)
    with pytest.raises(R.RunFailed):
        e.go(refresh_fn=lambda *a, **k: clean(), ask_factory=raising(ConnectionError("blip")))
    assert e.marker()["state"] == "failed:transient"
    m = e.marker()
    path = next(e.cust.glob("gate-run-*.marker"))
    path.write_text(json.dumps({**m, "pins": {**m["pins"], "manifestSha256": "0" * 64}}) + "\n")
    with pytest.raises(G.Refusal, match="manifestSha256"):
        e.go(resume=True)


def test_d2_the_default_system_is_the_manifests_not_ccg():
    a = argparse.Namespace(system=None, host="127.0.0.1", port=16431, graph=None, repo=str(CPG.parent))
    assert R._default_deps(a)["system"] == "graphmind-ai-lab"


def test_d2_an_explicit_system_that_disagrees_with_the_manifest_is_refused_before_the_build(tmp_path):
    from gate_run_helpers import YAML_SYS
    e = Env(tmp_path, system_yaml="system: realsys\n" + YAML_SYS)
    with pytest.raises(G.Refusal, match="manifest"):
        e.go(system="other", refresh_fn=lambda *a, **k: pytest.fail("the build must not start"))
    assert not e.markers() and not list(e.cust.glob("gate-build-*"))


# ================================================================================================ D3: the sandbox writer's audit goes to the private dir
def test_d3_the_sandbox_writer_resolves_the_private_audit_dir_during_the_build_and_the_environment_is_restored(tmp_path, monkeypatch):
    from ccg import sandbox_audit
    sys.path.insert(0, str(CPG))
    from extractors.indexers import run as indexers_run
    monkeypatch.delenv("CCG_AUDIT_DIR", raising=False)
    e = Env(tmp_path)
    seen = {}

    def refresh(system, repo_root=None, **kw):
        seen["kw"] = Path(kw["audit_dir"])
        seen["default"] = sandbox_audit.default_audit_dir()               # what ``refresh`` and ``assemble`` read
        seen["writer"] = indexers_run._audit_dir()                         # the real sandbox writer's resolver
        seen["env"] = os.environ.get("CCG_AUDIT_DIR")                      # what the offline-run.sh child inherits
        return clean()
    e.go(refresh_fn=refresh)
    assert seen["default"] == seen["writer"] == seen["kw"] and seen["env"] == str(seen["kw"])
    assert e.cust.resolve() in seen["kw"].resolve().parents
    assert "CCG_AUDIT_DIR" not in os.environ                                   # the harness process is not left re-pointed


def test_d3_a_developer_audit_variable_is_restored_after_the_build_even_when_it_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "dev-audit"))
    e = Env(tmp_path)

    def boom(*a, **k):
        assert os.environ["CCG_AUDIT_DIR"] != str(tmp_path / "dev-audit")
        raise RuntimeError("build failed")
    with pytest.raises(RuntimeError):
        e.go(refresh_fn=boom)
    assert os.environ["CCG_AUDIT_DIR"] == str(tmp_path / "dev-audit")


def report(*statuses):
    """The REAL ``ccg.refresh.RefreshReport`` holding one unit per status."""
    from ccg.refresh import RefreshReport, UnitReport
    r = RefreshReport("x")
    for i, st in enumerate(statuses):
        r.units.append(UnitReport(unit_id=f"u{i}", repo_id="r", language="python", root_path=f"p{i}", status=st, built_commit="c" * 40))
    return r


def clean():
    return H.clean_report()


def raising(exc):
    """An ask factory whose ask raises ``exc`` (inside the run, after the marker)."""
    def factory(built, snap):
        def ask(req):
            raise exc
        return ask
    return factory


@pytest.mark.parametrize("statuses,word", [(("fresh", "not-mapped"), "not-mapped"), (("fresh", "stale-kept"), "stale-kept"),
                                            (("not-mapped",) * 12 + ("fresh",) * 5, "12")])
def test_d3_a_build_with_a_unit_not_fresh_is_refused_with_counts_not_exit_zero(tmp_path, statuses, word):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match=word):
        e.go(refresh_fn=lambda *a, **k: report(*statuses))
    assert not e.markers() and not e.run_files()                                       # nothing is spent


def test_d3_an_all_fresh_build_goes_through(tmp_path):
    e = Env(tmp_path)
    many_world(e, 1)
    assert e.go(refresh_fn=lambda *a, **k: report("fresh", "fresh"))["asked"] == 1


def test_d3_a_build_that_returns_no_unit_report_is_refused_not_trusted(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="report"):
        e.go(refresh_fn=lambda *a, **k: None)


def test_d3_the_refusal_after_the_build_tells_the_operator_what_to_do_with_the_store(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal) as ei:
        e.go(refresh_fn=lambda *a, **k: report("not-mapped"))
    assert "new empty store" in str(ei.value) and "ACL" in str(ei.value)


# ================================================================================================ D4: the canary asks at the real store limit
class LimitedStore:
    """``falkordb``'s raw connection with the pinned instance's ``TIMEOUT_MAX``: a larger limit is refused with the server's own text."""

    def __init__(self, timeout_max=10_000):
        self.timeout_max, self.asked = timeout_max, []

    def __call__(self, host, port, *, username, password, socket_timeout=None):
        return self

    def select_graph(self, key):
        return self

    def ro_query(self, cypher, params, timeout):
        self.asked.append(timeout)
        if timeout > self.timeout_max:
            raise ResponseError("The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value")
        return types.SimpleNamespace(result_set=[["s" * 32]], header=[(1, "s.snapshotId")])


def test_d4_the_pinned_deadline_is_the_engines_query_limit_not_a_literal():
    from ccg.engine import config
    assert R.STALL_DEADLINE_S * 1000 == config.QUERY_TIMEOUT_MS


def test_d4_the_default_canary_answers_on_a_store_whose_timeout_max_is_the_pinned_one(monkeypatch):
    from clients.common import Client
    monkeypatch.setattr("clients.common._credential", cred)
    store = LimitedStore()
    client = Client(SYS, host="127.0.0.1", port=16431, raw_connect=store)
    a = argparse.Namespace(system=SYS, host="127.0.0.1", port=16431, graph=None)
    deps = R._default_deps(a, credential=cred, client_factory=lambda system, host, port: client, db_factory=lambda h, p: types.SimpleNamespace(select_graph=lambda k: k))
    out = deps["canary"]()                                                             # raises RawRefused on a deadline above TIMEOUT_MAX
    assert out["rows"] == [["s" * 32]] and store.asked == [10_000]


def test_d4_the_marker_pins_the_deadline_the_canary_actually_uses(tmp_path):
    e = Env(tmp_path)
    many_world(e, 1)
    e.go(refresh_fn=lambda *a, **k: clean(), canary=lambda: None)
    assert e.marker()["pins"]["stall"]["deadlineS"] == R.STALL_DEADLINE_S == 10


# ================================================================================================ D6: both refusals tell the operator what to do
def test_d6_the_resolve_refusal_says_the_list_is_unspent_and_what_the_store_needs(tmp_path):
    e = Env(tmp_path)
    many_world(e, 1)
    e.overrides = {k: {"status": "unresolved", "candidates": []} for k in e.table}
    with pytest.raises(R.ResolveRefusal) as ei:
        e.go(refresh_fn=lambda *a, **k: clean())
    msg = str(ei.value)
    assert "none=1" in msg and "total=1" in msg                                          # counts still there
    assert "not spent" in msg and "new empty store" in msg and "ACL" in msg and "rebuild" in msg
    assert H_no_gold(msg, e)


def H_no_gold(msg, e):
    return "SENTINEL" not in msg and e.w.qname not in msg


def test_d6_the_live_graph_refusal_names_the_remedy(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal) as ei:
        e.go(live_graph_present=lambda: True, refresh_fn=lambda *a, **k: pytest.fail("no build"))
    msg = str(ei.value)
    assert "new empty store" in msg and "ACL" in msg and "rebuild" in msg and "not spent" in msg
    assert not e.markers()


# ================================================================================================ D7: score on a run that never completed is the marker refusal
def score_cli(e, monkeypatch, run_path):
    monkeypatch.setenv("HOME", str(e.home))
    argv = ["score", "--gate-id", GATE, "--lists", str(e.lists), "--run", str(run_path), "--custodian", str(e.cust), "--port", "16420"]
    return G.main(argv, snapshot=e.snap, repo=e.repo, preflight_fn=lambda repo, gid: G.preflight(repo, gid, observed=OBS))


@pytest.mark.parametrize("fail,state", [(ConnectionError("blip"), "failed:transient"), (ValueError("outside the ask"), "failed:deterministic")])
def test_d7_score_of_a_failed_run_without_a_run_file_is_exit_2_with_the_marker_refusal(tmp_path, monkeypatch, capsys, fail, state):
    e = Env(tmp_path)
    many_world(e, 2)

    def broken(path, obj):                                                           # a fault OUTSIDE the engine's ask is a run-wide, deterministic one
        raise fail
    if state.endswith("deterministic"):
        monkeypatch.setattr(R, "_append_line", broken)
    with pytest.raises(R.RunFailed):
        e.go(refresh_fn=lambda *a, **k: clean(), **({"ask_factory": raising(fail)} if state.endswith("transient") else {}))
    assert e.marker()["state"] == state
    assert not e.run_files()
    key = e.marker()["listSha256"]
    rc = score_cli(e, monkeypatch, e.cust / f"run-{key}.json")
    err = capsys.readouterr().err
    assert rc == 2 and "marker" in err and "FileNotFoundError" not in err


def test_d7_score_with_no_marker_and_no_run_file_is_exit_2(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 1)
    assert score_cli(e, monkeypatch, e.cust / "run-none.json") == 2
    assert "FileNotFoundError" not in capsys.readouterr().err
