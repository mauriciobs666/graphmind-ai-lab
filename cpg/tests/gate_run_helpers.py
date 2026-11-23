"""Shared plumbing for the ``gate_run`` tests (P1-32b round B): a throwaway repo with a committed prereg record and EVAL_COMMIT, a home with
the custodian area, the synthetic world, a fake client and a fake resolver. Nothing here is held-out data."""
from __future__ import annotations

import json

import gate_helpers as H
from gate_helpers import G

import gate_run as R

YAML_EM = "exclusions:\n  paths: [\"cpg/\"]\nprotocol:\n  documents: []\n  protocolString: \"\"\n"
YAML_SYS = "repos:\n  - exclusions:\n      paths: [\"cpg/\"]\n"
OBS = {"algorithm": {"id": "bfs-v1", "version": "1.2"}, "defaultDepthBound": 2}
GATE = "ac5-gate-1"


DROP = object()                      # ``Env(record_extra={'kind': DROP})``: the committed record lacks that field


class Killed(BaseException):
    """A planted SIGKILL: no handler of the run may write anything on it (it is not an ``Exception``)."""


class FakeClient:
    """Stands in for ``clients.common.Client``: records the CLI-shape requests, answers or raises."""

    def __init__(self, doc=None, exc=None):
        self.doc, self.exc, self.calls = doc, exc, []

    def ask(self, request=None, *, question=None, cache_bypass=False):
        self.calls.append(request)
        if self.exc:
            raise self.exc
        return self.doc


ID_KEY = {"symbol": "symbolId", "contract": "cpId", "config": "entryId"}


def resolution_of(a_type: str, nid: str, **extra) -> dict:
    return {"status": "resolved", "candidates": [{"type": a_type, ID_KEY[a_type]: nid}], **extra}


def spec_key(spec) -> str:
    return json.dumps(spec, sort_keys=True)


class Env:
    """A throwaway repo with a committed prereg record and EVAL_COMMIT, a home with the custodian area, and the synthetic world."""

    def __init__(self, tmp_path, record_extra=None, late_manifest=False, system_yaml=YAML_SYS):
        """``late_manifest``: the system manifest is committed AFTER ``EVAL_COMMIT`` (as in the real repository, where ``cpg/ccg/system.yaml`` was added
        after the pinned commit): the clone at ``EVAL_COMMIT`` does not hold it."""
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.cust = self.home / ".local" / "share" / "ccg-eval" / "e12"
        self.cust.mkdir(parents=True)
        self.w = H.World(qname="svc/SENTINEL_QNAME.py::SentinelFn")
        self.lists = self.cust / "ac5-http.jsonl"
        self.lists.write_text(json.dumps(self.w.rec) + "\n")
        self.probes = self.cust / "s47-probe-map.json"
        self.probes.write_text("{}\n")
        self.repo = repo = H.make_repo(tmp_path)
        for rel in G.HARNESS_FILES:
            f = repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"# {rel}\n")
        (repo / "cpg/eval/eval-manifest.yaml").write_text(YAML_EM)
        (repo / "cpg/ccg").mkdir(exist_ok=True)
        if not late_manifest:
            (repo / "cpg/ccg/system.yaml").write_text(system_yaml)
        H.git(repo, "add", "-A")
        H.git(repo, "commit", "-qm", "harness")
        self.eval_commit = H.head(repo)
        (repo / "cpg/eval/EVAL_COMMIT").write_text(self.eval_commit + "\n")
        if late_manifest:
            (repo / "cpg/ccg/system.yaml").write_text(system_yaml)
        H.git(repo, "add", "-A")
        H.git(repo, "commit", "-qm", "eval commit")
        t = G.observe_tree(repo)
        rec = {"schema": "ccg-prereg-v1", "kind": "gate", "stratum": GATE, "packSetHash": t["packSetHash"], "coreTreeHash": t["coreTreeHash"], **OBS,
               **(record_extra or {})}
        rec = {k: v for k, v in rec.items() if v is not DROP}
        (repo / "cpg/eval/preregistration.jsonl").write_text(json.dumps(rec) + "\n")
        H.git(repo, "add", "-A")
        H.git(repo, "commit", "-qm", "record")
        self.events = []
        self.refresh_kw = None
        self.asked: list[str] = []
        self.resolved: list[dict] = []
        self.table: dict[str, tuple[str, str]] = {}
        self.overrides: dict[str, dict] = {}
        self.engine_sid = H.SNAP
        self.snap = self.w.snap
        self.ask_hook = None
        self.set_records([self.w.rec])

    # -- the list on disk and what the fake resolver knows about it
    def set_records(self, records, path=None):
        (path or self.lists).write_text("".join(json.dumps(r) + "\n" for r in records))
        locators = G.locator_index(records)
        probes = json.loads(self.probes.read_text())
        for rec in records:
            aid = G.ask_id_of(rec, probes) if rec["anchor"].get("type") != "unit" or probes.get(rec["anchorId"]) else None
            if aid is None:
                continue
            req = G.build_request(rec, aid, locators)
            a = req["anchors"][0]
            self.table[spec_key(R.cli_request(req)["anchor"])] = (a["type"], aid)

    def set_probes(self, probes: dict):
        self.probes.write_text(json.dumps(probes) + "\n")

    # -- the injected seams
    def refresh(self, system, repo_root=None, **kw):
        self.refresh_kw = kw
        self.events.append(("refresh", str(repo_root)))
        return H.clean_report(system)                                                       # the REAL report type: a complete build

    def resolve(self, spec):
        self.events.append(("resolve",))
        self.resolved.append(spec)
        k = spec_key(spec)
        if k in self.overrides:
            return self.overrides[k]
        t, nid = self.table[k]
        return resolution_of(t, nid)

    def ask(self, req):
        self.events.append(("ask", bool(list(self.cust.glob("gate-run-*.marker")))))      # the second field: the marker already existed
        a = req["anchors"][0]
        self.asked.append(a.get("cpId") or a.get("symbolId") or a.get("entryId"))
        if self.ask_hook:
            self.ask_hook(req, len(self.asked))
        return self.w.answer()

    def go(self, **over):
        kw = dict(repo=self.repo, gate_id=GATE, lists=[self.lists], custodian=self.cust, home=self.home, observed=OBS, refresh_fn=self.refresh,
                  snapshot_factory=lambda built: self.snap, ask_factory=lambda built, snap: self.ask, cache_len=lambda: 0,
                  resolve_factory=lambda built, snap: self.resolve, probe_map=self.probes, engine_snapshot_id=lambda: self.engine_sid)
        kw.update(over)
        return R.run_gate(**kw)

    def argv(self, *extra):
        return ["run", "--gate-id", GATE, "--lists", str(self.lists), "--custodian", str(self.cust), "--repo", str(self.repo), *extra]

    def deps(self, **over):
        d = dict(observed=OBS, refresh_fn=self.refresh, snapshot_factory=lambda b: self.snap, ask_factory=lambda b, s: self.ask,
                 cache_len=lambda: 0, resolve_factory=lambda b, s: self.resolve, engine_snapshot_id=lambda: self.engine_sid)
        d.update(over)
        return d

    # -- what is on disk
    def markers(self):
        return sorted(self.cust.glob("gate-run-*.marker"))

    def marker(self) -> dict:
        (m,) = self.markers()
        return json.loads(m.read_text())

    def run_files(self):
        return sorted(self.cust.glob("run-*.json"))

    def answers_file(self):
        (f,) = self.cust.glob("run-*.answers.jsonl")
        return f


def many_world(e: Env, n: int):
    """``n`` distinct contract anchors, each in the gate graph and resolvable; the fake ask answers every one with the world's answer."""
    recs = []
    for i in range(n):
        anchor = H.contract("http", f"GET /a{i}")
        rec = H.item(10 + i, anchor, H.sym(f"svc/c{i}.py::f"))
        e.snap.add_node("ContractPoint", rec["anchorId"])
        recs.append(rec)
    e.set_records(recs)
    return recs
