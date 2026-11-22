"""P1-32b run wiring (``gate_run.py``): the engine-ask adapter and the ``run`` verb. Offline and synthetic: a fake ``ask``, a fake refresh,
throwaway git repos and an in-memory snapshot. Nothing here is held-out data and the run verb is never pointed at the real lists."""
import json

import pytest

import gate_helpers as H
from gate_helpers import G

import gate_run as R
from clients.common import AnchorNotResolved, StoreError

YAML_EM = "exclusions:\n  paths: [\"cpg/\"]\nprotocol:\n  documents: []\n  protocolString: \"\"\n"
YAML_SYS = "repos:\n  - exclusions:\n      paths: [\"cpg/\"]\n"
OBS = {"algorithm": {"id": "bfs-v1", "version": "1.2"}, "defaultDepthBound": 2}
GATE = "ac5-gate-1"


class FakeClient:
    """Stands in for ``clients.common.Client``: records the CLI-shape requests, answers or raises."""

    def __init__(self, doc=None, exc=None):
        self.doc, self.exc, self.calls = doc, exc, []

    def ask(self, request=None, *, question=None, cache_bypass=False):
        self.calls.append(request)
        if self.exc:
            raise self.exc
        return self.doc


# ------------------------------------------------------------------------------------------------ the ask adapter
def test_contract_request_is_issued_in_the_cli_shape():
    w = H.World()
    c = FakeClient(w.answer())
    ans = R.make_ask(c)(G.build_request(w.rec, w.cp))
    assert c.calls == [{"useCase": "impact", "anchor": {"kind": "contract", "key": "GET /a", "contractKind": "http"}}]
    assert ans == w.answer()                                  # the full answer JSON, untouched


def test_symbol_and_config_requests_are_issued_in_the_cli_shape():
    c = FakeClient({})
    ask = R.make_ask(c)
    s = H.item(1, H.sym("svc/a.py::f"), H.sym("svc/b.py::g"))
    k = H.item(2, H.cfg("compose.yaml", "services.s"), H.sym("svc/b.py::g"))
    ask(G.build_request(s, s["anchorId"]))
    ask(G.build_request(k, k["anchorId"]))
    assert c.calls == [{"useCase": "impact", "anchor": {"kind": "symbol", "repo": "r", "qname": "svc/a.py::f"}},
                       {"useCase": "impact", "anchor": {"kind": "config", "repo": "r", "path": "compose.yaml", "keyPath": "services.s"}}]


def test_the_cli_request_passes_the_clients_own_parser():
    from clients import common
    for rec in (H.item(1, H.sym("a.py::f"), H.sym("b.py::g")), H.item(2, H.cfg("c.yaml", "k"), H.sym("b.py::g")), H.World().rec):
        cli = R.cli_request(G.build_request(rec, rec["anchorId"]))
        assert common.parse_request(cli)[0] == "impact"
        need, may = common._ANCHOR_KEYS[cli["anchor"]["kind"]]
        keys = set(cli["anchor"]) - {"kind"}
        assert need <= keys and not keys - need - may


def test_unit_anchor_is_asked_through_run_asks_from_its_probe_contract():
    c1, c2 = H.contract("mcp_tool", "alpha"), H.contract("mcp_tool", "beta")
    unit = {"type": "unit", "repoId": "r", "rootPath": "svc", "lang": "python"}
    rec = H.item(1, unit, H.cfg("compose.yaml", "services.s"), serves=[c1, c2])
    other = H.item(2, c1, H.sym("a.py::f"))                   # alpha has one item, beta none: the probe is beta
    snap = H.DictSnapshot()
    snap.add_node("ContractPoint", G._mk.contract_id(c2))
    c = FakeClient({})
    G.run_asks([rec, other], snap, R.make_ask(c), lambda: 0)
    assert c.calls == [{"useCase": "impact", "anchor": {"kind": "contract", "key": "beta", "contractKind": "mcp_tool"}}]


def test_an_unresolved_anchor_becomes_a_not_resolved_answer_that_scores_as_a_miss_not_a_void():
    w = H.World()
    exc = AnchorNotResolved("x", status="ambiguous", detail={"status": "ambiguous", "candidates": [{}, {}]})
    ask = R.make_ask(FakeClient(exc=exc), stamp=H.stamp())
    run = G.run_asks([w.rec], w.snap, ask, lambda: 0)
    ans = run["answers"][w.cp]["answer"]
    assert ans["interpretation"]["status"] == "ambiguous" and ans["synthesized"] == "anchor-ambiguous"
    res = G.score_gate([w.rec], run, w.snap, ())
    assert res["items"][w.rec["itemId"]]["outcome"] == "miss" and res["items"][w.rec["itemId"]]["reason"] == "not-resolved"
    assert res["stops"] == {}


def test_a_store_failure_is_not_swallowed():
    w = H.World()
    with pytest.raises(StoreError):
        R.make_ask(FakeClient(exc=StoreError("down")), stamp=H.stamp())(G.build_request(w.rec, w.cp))


# ------------------------------------------------------------------------------------------------ the run verb
class Env:
    """A throwaway repo with a committed prereg record and EVAL_COMMIT, a home with the custodian area, and the synthetic world."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.cust = self.home / ".local" / "share" / "ccg-eval" / "e12"
        self.cust.mkdir(parents=True)
        self.w = H.World(qname="svc/SENTINEL_QNAME.py::SentinelFn")
        self.lists = self.cust / "ac5-http.jsonl"
        self.lists.write_text(json.dumps(self.w.rec) + "\n")
        self.repo = repo = H.make_repo(tmp_path)
        for rel in G.HARNESS_FILES:
            f = repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"# {rel}\n")
        (repo / "cpg/eval/eval-manifest.yaml").write_text(YAML_EM)
        (repo / "cpg/ccg").mkdir(exist_ok=True)
        (repo / "cpg/ccg/system.yaml").write_text(YAML_SYS)
        H.git(repo, "add", "-A")
        H.git(repo, "commit", "-qm", "harness")
        self.eval_commit = H.head(repo)
        (repo / "cpg/eval/EVAL_COMMIT").write_text(self.eval_commit + "\n")
        H.git(repo, "add", "-A")
        H.git(repo, "commit", "-qm", "eval commit")
        t = G.observe_tree(repo)
        rec = {"schema": "ccg-prereg-v1", "stratum": GATE, "packSetHash": t["packSetHash"], "coreTreeHash": t["coreTreeHash"], **OBS}
        (repo / "cpg/eval/preregistration.jsonl").write_text(json.dumps(rec) + "\n")
        H.git(repo, "add", "-A")
        H.git(repo, "commit", "-qm", "record")
        self.events = []

    def refresh(self, system, repo_root=None, **kw):
        self.events.append(("refresh", str(repo_root)))

    def ask(self, req):
        self.events.append(("ask",))
        return self.w.answer()

    def go(self, **over):
        kw = dict(repo=self.repo, gate_id=GATE, lists=[self.lists], custodian=self.cust, home=self.home, observed=OBS, refresh_fn=self.refresh,
                  snapshot_factory=lambda built: self.w.snap, ask_factory=lambda built, snap: self.ask, cache_len=lambda: 0)
        kw.update(over)
        return R.run_gate(**kw)


def test_run_writes_the_run_json_that_score_reads_and_returns_counts_only(tmp_path):
    e = Env(tmp_path)
    out = e.go()
    assert out == {"records": 1, "asked": 1, "label": "first-contact"}
    run = json.loads(next(e.cust.glob("run-*.json")).read_text())
    res = G.score_gate([e.w.rec], run, e.w.snap, ())
    assert res["items"][e.w.rec["itemId"]]["outcome"] == "hit"
    assert [ev[0] for ev in e.events] == ["refresh", "ask"]


def test_run_builds_the_refresh_target_at_eval_commit_under_the_custodian_root(tmp_path):
    e = Env(tmp_path)
    e.go()
    (_, built), = [ev for ev in e.events if ev[0] == "refresh"]
    assert built.startswith(str(e.cust.resolve())) or built.startswith(str(e.cust))
    assert H.git(e.repo, "-C", built, "rev-parse", "HEAD").strip() == e.eval_commit


def test_run_takes_the_first_run_marker_and_a_second_run_is_refused_before_any_build(tmp_path):
    e = Env(tmp_path)
    e.go()
    assert list(e.cust.glob("gate-run-*.marker"))
    e.events.clear()
    with pytest.raises(G.Refusal, match="already run"):
        e.go()
    assert e.events == []


def test_run_refuses_a_dirty_cpg_tree_before_any_build_or_ask(tmp_path):
    e = Env(tmp_path)
    (e.repo / "cpg/ccg/a.py").write_text("A = 5\n")
    with pytest.raises(G.Refusal, match="first-contact"):
        e.go()
    assert e.events == [] and not list(e.cust.glob("*.marker")) and not list(e.cust.glob("run-*.json"))


def test_run_refuses_when_a_prereg_field_differs(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="first-contact"):
        e.go(observed={**OBS, "defaultDepthBound": 3})
    assert e.events == []


def test_run_refuses_without_the_committed_prereg_record(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="preregistration"):
        e.go(gate_id="ac5-gate-nope")
    assert e.events == []


def test_run_refuses_a_custodian_root_outside_the_allowed_area(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="custodian"):
        e.go(custodian=tmp_path / "elsewhere")
    assert e.events == [] and not (tmp_path / "elsewhere").exists()


def test_run_refuses_held_out_lists_read_from_outside_the_custodian_area(tmp_path):
    e = Env(tmp_path)
    stray = tmp_path / "stray.jsonl"
    stray.write_text(json.dumps(e.w.rec) + "\n")
    with pytest.raises(G.Refusal, match="custodian"):
        e.go(lists=[stray])
    assert e.events == []


def test_run_refuses_a_non_empty_cache_and_still_asks_nothing(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="cache is not empty"):
        e.go(cache_len=lambda: 2)
    assert ("ask",) not in e.events and not list(e.cust.glob("*.marker"))


def test_a_new_list_is_not_blocked_by_an_earlier_run_and_keeps_the_earlier_run_file(tmp_path):
    e = Env(tmp_path)
    e.go()
    first = next(e.cust.glob("run-*.json")).read_text()
    second = H.item(2, H.contract("http", "GET /a"), H.sym("svc/other.py::g"))
    e.lists.write_text(json.dumps(second) + "\n")
    e.go()
    assert len(list(e.cust.glob("*.marker"))) == 2 and len(list(e.cust.glob("run-*.json"))) == 2
    assert first in [f.read_text() for f in e.cust.glob("run-*.json")]


def test_run_refuses_a_run_when_stored_gold_does_not_renormalize(tmp_path):
    e = Env(tmp_path)
    bad = {**e.w.rec, "anchorId": "f" * 32}
    e.lists.write_text(json.dumps(bad) + "\n")
    with pytest.raises(G.Refusal, match="renormalize"):
        e.go()
    assert e.events == []


def test_cli_run_prints_counts_only_and_leaks_no_item_content(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(["run", "--gate-id", GATE, "--lists", str(e.lists), "--custodian", str(e.cust), "--repo", str(e.repo)],
                deps=dict(observed=OBS, refresh_fn=e.refresh, snapshot_factory=lambda b: e.w.snap, ask_factory=lambda b, s: e.ask,
                          cache_len=lambda: 0))
    cap = capsys.readouterr()
    assert rc == 0 and json.loads(cap.out) == {"records": 1, "asked": 1, "label": "first-contact"}
    for secret in ("SENTINEL", e.w.rec["itemId"], e.w.rec["anchorId"], e.w.cp, e.w.sid, "GET /a"):
        assert secret not in cap.out and secret not in cap.err


def test_cli_run_refusal_exits_2_with_no_content_on_either_stream(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(["run", "--gate-id", GATE, "--lists", str(e.lists), "--custodian", str(tmp_path / "elsewhere"), "--repo", str(e.repo)],
                deps=dict(observed=OBS, refresh_fn=e.refresh, snapshot_factory=lambda b: e.w.snap, ask_factory=lambda b, s: e.ask))
    cap = capsys.readouterr()
    assert rc == 2 and cap.out == "" and "REFUSED" in cap.err and "SENTINEL" not in cap.err + cap.out


def test_run_writes_nothing_outside_the_custodian_root_and_the_repo_stays_clean(tmp_path):
    e = Env(tmp_path)
    before = set(p for p in tmp_path.rglob("*") if ".git" not in p.parts and not str(p).startswith(str(e.home)))
    e.go()
    after = set(p for p in tmp_path.rglob("*") if ".git" not in p.parts and not str(p).startswith(str(e.home)))
    assert after == before
    assert G._porcelain(e.repo) == []
