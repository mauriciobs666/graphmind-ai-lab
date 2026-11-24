"""P1-32b Pass 4, P4-5: six round D/E checks that no test could fail (the analyst's surviving mutants E7, E21, E23, D18, E14, E16), each now a plant that
reddens the mutant. Offline and synthetic; nothing here is held-out data."""
import argparse
import json
import types
from pathlib import Path

import pytest
from redis.exceptions import AuthenticationError, ConnectionError as RedisConnectionError, ResponseError

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world, GATE, OBS, YAML_SYS

import gate_run as R

CPG = Path(__file__).resolve().parents[1]
EMPTY_KEY = "Invalid graph operation on empty key"


# ------------------------------------------------------------------------------------------------ E7: an empty unit list is not a complete build
@pytest.mark.parametrize("units", [[], None, (), "x"], ids=["empty-list", "none", "tuple", "string"])
def test_e7_a_build_report_whose_unit_list_is_empty_or_not_a_list_is_refused_with_the_remedy(units):
    with pytest.raises(G.Refusal) as ei:
        G.require_complete_build(types.SimpleNamespace(units=units))
    assert "no unit report" in str(ei.value) and G.STORE_REMEDY in str(ei.value)


def test_e7_one_fresh_unit_is_complete():
    G.require_complete_build(types.SimpleNamespace(units=[types.SimpleNamespace(status="fresh")]))


# ------------------------------------------------------------------------------------------------ E21: `gate.py score` defaults to the manifest's system
def test_e21_score_without_system_reads_the_graph_of_the_manifests_system_not_ccg(tmp_path, monkeypatch):
    e = Env(tmp_path, system_yaml="system: manifestsys\n" + YAML_SYS)
    many_world(e, 1)
    e.go(system="manifestsys")
    seen = {}

    def snapshot_graph(host, port, graph, system, **kw):
        seen["system"] = system
        raise G.Refusal("stop here: the system was chosen")
    monkeypatch.setattr(G, "snapshot_graph", snapshot_graph)
    monkeypatch.setenv("HOME", str(e.home))
    argv = ["score", "--gate-id", GATE, "--lists", str(e.lists), "--run", str(e.run_files()[0]), "--custodian", str(e.cust), "--dev-probe", str(e.dev_probe), "--port", "16431"]
    assert G.main(argv, repo=e.repo, preflight_fn=lambda repo, gid: G.preflight(repo, gid, observed=OBS)) == 2
    assert seen == {"system": "manifestsys"}


def test_e21_an_explicit_system_wins_over_the_manifests(tmp_path, monkeypatch):
    e = Env(tmp_path, system_yaml="system: manifestsys\n" + YAML_SYS)
    many_world(e, 1)
    e.go(system="manifestsys")
    seen = {}

    def snapshot_graph(host, port, graph, system, **kw):
        seen["system"] = system
        raise G.Refusal("stop here: the system was chosen")
    monkeypatch.setattr(G, "snapshot_graph", snapshot_graph)
    monkeypatch.setenv("HOME", str(e.home))
    argv = ["score", "--gate-id", GATE, "--lists", str(e.lists), "--run", str(e.run_files()[0]), "--custodian", str(e.cust), "--dev-probe", str(e.dev_probe), "--port", "16431", "--system", "explicit"]
    G.main(argv, repo=e.repo, preflight_fn=lambda repo, gid: G.preflight(repo, gid, observed=OBS))
    assert seen == {"system": "explicit"}


# ------------------------------------------------------------------------------------------------ E23: `run --repo X` reads X's manifest
def test_e23_the_default_deps_read_the_manifest_of_the_repo_given_not_the_checkouts_own(tmp_path):
    other = tmp_path / "other"
    (other / "cpg" / "ccg").mkdir(parents=True)
    (other / "cpg" / "ccg" / "system.yaml").write_text("system: othersys\n" + YAML_SYS)
    a = argparse.Namespace(system=None, host="127.0.0.1", port=16431, graph=None, repo=str(other))
    assert R._default_deps(a)["system"] == "othersys"
    own = argparse.Namespace(system=None, host="127.0.0.1", port=16431, graph=None, repo=str(CPG.parent))
    assert R._default_deps(own)["system"] != "othersys"                                       # the two repos differ, so the first assertion is not a coincidence


def test_e23_a_repo_whose_manifest_names_no_system_falls_back_to_ccg(tmp_path):
    other = tmp_path / "bare"
    (other / "cpg" / "ccg").mkdir(parents=True)
    (other / "cpg" / "ccg" / "system.yaml").write_text(YAML_SYS)
    a = argparse.Namespace(system=None, host="127.0.0.1", port=16431, graph=None, repo=str(other))
    assert R._default_deps(a)["system"] == "ccg"


# ------------------------------------------------------------------------------------------------ D18: an unusable eval manifest is refused before the marker
@pytest.mark.parametrize("text", ["exclusions: [unterminated\n", "- just\n- a list\n", ""], ids=["unparseable", "not-a-mapping", "empty"])
def test_d18_an_unusable_eval_manifest_is_refused_by_score_inputs_check_and_so_by_both_verbs(tmp_path, text):
    e = Env(tmp_path)
    many_world(e, 1)
    pf = G.preflight(e.repo, GATE, observed=OBS)                                              # computed while the tree is clean and the manifest valid
    records = [json.loads(l) for l in e.lists.read_text().splitlines()]
    (e.repo / "cpg" / "eval" / "eval-manifest.yaml").write_text(text)
    with pytest.raises(G.PrecheckRefusal, match="eval manifest is unusable"):
        G.score_inputs_check(e.repo, GATE, pf, records)


def test_d18_the_valid_manifest_passes_the_same_check(tmp_path):
    e = Env(tmp_path)
    many_world(e, 1)
    pf = G.preflight(e.repo, GATE, observed=OBS)
    records = [json.loads(l) for l in e.lists.read_text().splitlines()]
    assert set(G.score_inputs_check(e.repo, GATE, pf, records, dev_probe=json.loads(e.dev_probe.read_text()))) == {"kind", "movedOut", "patterns"}


# ------------------------------------------------------------------------------------------------ E14 / E16: only the server's "empty key" reply is an absent graph
class Probe:
    """``falkordb.FalkorDB``'s stand-in whose ``ro_query`` raises ``fault()`` (or answers)."""

    def __init__(self, fault=None):
        self.fault = fault

    def __call__(self, host=None, port=None, username=None, password=None, **kw):
        return self

    def select_graph(self, key):
        return self

    def ro_query(self, cypher, params=None, timeout=None):
        if self.fault is not None:
            raise self.fault()
        return types.SimpleNamespace(result_set=[[1]])


def probe_of(fault):
    a = argparse.Namespace(system="p4sys", host="127.0.0.1", port=16431, graph=None)
    return R._default_deps(a, credential=lambda s, r: "pw", db_factory=Probe(fault))["live_graph_present"]


@pytest.mark.parametrize("fault", [lambda: ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value"),
                                   lambda: ResponseError("NOPERM this user has no permissions to run the 'graph.ro_query' command"),
                                   lambda: ResponseError("Query timed out"),
                                   lambda: ResponseError("Type mismatch: expected Integer"),
                                   lambda: RedisConnectionError("Error 111 connecting"), lambda: AuthenticationError("bad credential"),
                                   lambda: ConnectionRefusedError("refused"), lambda: TimeoutError("timed out"),
                                   lambda: Exception("Invalid graph operation on empty key"), lambda: RuntimeError("empty key")],
                         ids=["WRONGTYPE", "NOPERM", "Query-timed-out", "Cypher-error", "redis-ConnectionError", "AuthenticationError", "refused", "timeout",
                              "plain-Exception-with-the-empty-key-text", "RuntimeError-empty-key"])
def test_e14_e16_anything_but_the_servers_empty_key_reply_is_raised_never_read_as_absent(fault):
    with pytest.raises(Exception) as ei:
        probe_of(fault)()
    assert type(ei.value) is type(fault())                                                    # the driver's own exception, unchanged


def test_e14_e16_the_servers_empty_key_reply_is_absent_and_an_answer_is_present():
    assert probe_of(lambda: ResponseError(EMPTY_KEY))() is False
    assert probe_of(None)() is True


# ------------------------------------------------------------------------------------------------ P4-6: the restore survives a signal handler raising inside it
class Interrupt(BaseException):
    """What the SIGTERM handler raises."""


@pytest.mark.parametrize("preset", [None, "/dev/dev-audit"], ids=["unset", "set"])
def test_p46_the_audit_variable_is_restored_when_an_interrupt_fires_inside_the_restore_itself(tmp_path, monkeypatch, preset):
    import os
    if preset is None:
        monkeypatch.delenv("CCG_AUDIT_DIR", raising=False)
    else:
        monkeypatch.setenv("CCG_AUDIT_DIR", preset)
    e = Env(tmp_path)
    real_pop, fired = os.environ.pop, []

    def pop_that_is_interrupted_once(*a, **k):
        if not fired:
            fired.append(1)
            raise Interrupt()
        return real_pop(*a, **k)
    real_setitem = os.environ.__class__.__setitem__

    def setitem_that_is_interrupted_once(self, key, value):
        if key == "CCG_AUDIT_DIR" and value == preset and not fired:
            fired.append(1)
            raise Interrupt()
        return real_setitem(self, key, value)
    with monkeypatch.context() as patched:
        patched.setattr(os.environ, "pop", pop_that_is_interrupted_once)
        patched.setattr(os.environ.__class__, "__setitem__", setitem_that_is_interrupted_once)
        with pytest.raises(Interrupt):                                                          # the interrupt is not swallowed ...
            G.build_at_eval_commit(e.eval_commit, tmp_path / "dest", repo=e.repo, refresh_fn=lambda *a, **k: H.clean_report(), system="s", audit_dir=tmp_path / "aud")
    assert fired == [1]
    assert os.environ.get("CCG_AUDIT_DIR") == preset                                           # ... and the variable is restored                                           # ... and the variable is restored


# ------------------------------------------------------------------------------------------------ P4-7
def test_p47_gate_score_takes_repo_and_derives_the_system_from_that_repos_manifest(tmp_path, monkeypatch):
    e = Env(tmp_path, system_yaml="system: reposys\n" + YAML_SYS)
    many_world(e, 1)
    e.go(system="reposys")
    seen = {}

    def snapshot_graph(host, port, graph, system, **kw):
        seen["system"] = system
        raise G.Refusal("stop here")
    monkeypatch.setattr(G, "snapshot_graph", snapshot_graph)
    monkeypatch.setenv("HOME", str(e.home))
    argv = ["score", "--gate-id", GATE, "--lists", str(e.lists), "--run", str(e.run_files()[0]), "--custodian", str(e.cust), "--dev-probe", str(e.dev_probe), "--port", "16431", "--repo", str(e.repo)]
    G.main(argv, preflight_fn=lambda repo, gid: G.preflight(repo, gid, observed=OBS))
    assert seen == {"system": "reposys"}
