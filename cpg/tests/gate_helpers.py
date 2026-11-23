"""Synthetic fixtures for the P1-32b gate harness tests (no held-out data, no store): a throwaway git repo, key items in the
dev-record shape, an in-memory gate snapshot and answer builders. Values are invented; nothing here is item content of any list."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import os
import sys

TOOLS = Path(os.environ.get("GATE_TOOLS_DIR") or Path(__file__).resolve().parents[1] / "eval" / "tools")   # a mutation run points this at a copy
sys.path.insert(0, str(TOOLS))
import gate as G  # noqa: E402
import match_keys as MK  # noqa: E402

SNAP = "a" * 32
ZERO_HASH = G.EMPTY_VERDICT_LOG_HASH


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


def make_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    for rel, text in {
        "cpg/ccg/a.py": "A = 1\n", "cpg/packs/p/pack.yaml": "rules: []\n", "cpg/eval/note.json": "{}\n",
        "cpg/tests/test_x.py": "X = 1\n", "cpg/README.md": "# r\n", "cpg/eval/tools/gate.py": "# gate v1\n",
        "cpg/eval/tools/match_keys.py": "# mk\n", "cpg/eval/preregistration.jsonl": "",
    }.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    return repo


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


# ----------------------------------------------------------------------------- key items (dev-record shape)

def sym(qname: str, path: str | None = None) -> dict:
    return {"type": "symbol", "repoId": "r", "qname": qname, "path": path or qname.split("::")[0]}


def cfg(path: str, key_path: str) -> dict:
    return {"type": "config", "repoId": "r", "path": path, "keyPath": key_path}


def contract(kind: str, key: str) -> dict:
    return {"type": "contract", "kind": kind, "keyNorm": key}


def item(n: int, anchor: dict, impacted: dict, lines=(10,), stratum: str = "http", counting: bool = True, **extra) -> dict:
    rec = {"itemId": f"it{n:02d}".ljust(32, "0"), "anchor": anchor, "impacted": impacted, "lines": list(lines),
           "derivedStratum": stratum, "counting": counting, "serves": [], "provider": None}
    if anchor["type"] == "contract":
        rec["anchorId"] = MK.contract_id(anchor)
    else:
        rec["anchorId"] = hashlib.md5(json.dumps(anchor, sort_keys=True).encode()).hexdigest()
    rec["impactedId"] = hashlib.md5(json.dumps(impacted, sort_keys=True).encode()).hexdigest()
    rec.update(extra)
    return rec


# ----------------------------------------------------------------------------- the in-memory gate snapshot

LABEL_ID = {"Symbol": "symbolId", "CallSite": "callSiteId", "Param": "pointId", "Ret": "pointId", "Cond": "pointId",
            "ContractPoint": "cpId", "Link": "linkId", "ConfigEntry": "entryId"}


class DictSnapshot:
    """Same methods as ``gate.StoreSnapshot``: nodes ``{(label, id)}``, stored edges ``{(etype, (sl, sid), (dl, did))}``."""

    def __init__(self, snapshot_id: str = SNAP):
        self.sid, self.nodes, self.edges, self.symbols, self.entries, self.loaded = snapshot_id, set(), set(), {}, {}, set()
        self.queries = 0

    def add_symbol(self, sid: str, qname: str, start: int, end: int, path: str | None = None, owned=None) -> str:
        self.symbols[sid] = {"symbolId": sid, "qname": qname, "path": path or qname.split("::")[0], "lineStart": start, "lineEnd": end,
                             **({"ownedRegions": owned} if owned else {})}
        self.nodes.add(("Symbol", sid))
        return sid

    def add_entry(self, eid: str, path: str, key_path: str) -> str:
        self.entries[eid] = {"entryId": eid, "path": path, "keyPath": key_path}
        self.nodes.add(("ConfigEntry", eid))
        return eid

    def add_node(self, label: str, nid: str) -> str:
        self.nodes.add((label, nid))
        return nid

    def add_edge(self, etype: str, src: tuple, dst: tuple) -> None:
        self.edges.add((etype, src, dst))
        self.loaded.add(etype)

    # the protocol
    def snapshot_id(self): return self.sid
    def node_exists(self, label, nid): self.queries += 1; return (label, nid) in self.nodes
    def edge_joins(self, etype, src, dst): self.queries += 1; return (etype, src, dst) in self.edges
    def edge_type_loaded(self, etype): return etype in self.loaded
    def symbol(self, sid): return self.symbols.get(sid)
    def config_entry(self, eid): return self.entries.get(eid)
    def symbols_of_path(self, path): return [s for s in self.symbols.values() if s["path"] == path]


def stamp(snapshot_id: str = SNAP, offset: int = 0, vhash: str = ZERO_HASH) -> dict:
    return {"snapshotId": snapshot_id, "verdictLogOffset": offset, "verdictLogHash": vhash}


def hop0(kind: str, nid: str) -> dict:
    return {"kind": kind, "id": nid, "state": "value"}


def hopn(kind: str, nid: str, etype: str, d: str, status: str = "confirmed", **kw) -> dict:
    return {"kind": kind, "id": nid, "state": "value", "edgeType": etype, "dir": d, "status": status, **kw}


def sym_subject(sid: str, qname: str, start: int, end: int, **kw) -> dict:
    return {"type": "symbol", "symbolId": sid, "qname": qname, "path": qname.split("::")[0], "lineStart": start, "lineEnd": end, **kw}


def ans_item(subject: dict, chain: list, status: str = "confirmed", unit: str = "u") -> dict:
    return {"subject": subject, "repoId": "r", "unitId": unit, "status": status, "directness": "direct", "distance": 1,
            "viaData": False, "chain": chain, **({"reason": "name-match"} if status == "possible" else {})}


def request(anchor_ids: list[str], direction: str = "impact") -> dict:
    return {"useCase": "impact", "direction": direction, "anchors": [{"type": "contract", "cpId": a} for a in anchor_ids]}


def answer(req: dict, items=(), groups=(), snapshot_id: str = SNAP, status: str = "resolved", **kw) -> dict:
    doc = {"answerSchemaVersion": "1.1", "interpretation": {"status": status, **({"request": req} if status == "resolved" else {})},
           "stamp": stamp(snapshot_id), "items": list(items), "groups": list(groups), "depthBound": {"unexplored": 0}}
    doc.update(kw)
    return doc


def group(members: list) -> dict:
    return {"kind": "other-possible-callees", "key": "k", "reason": "resolution-unverified", "count": len(members), "members": list(members)}


# ----------------------------------------------------------------------------- a small world: anchor CP -> Link -> CallSite -> Symbol

class World:
    """contract point (``GET /a``), its Link, a consumer CallSite and the consumer symbol ``svc/c.py::call`` (lines 5-20, chain of 4 hops)."""

    def __init__(self, qname: str = "svc/c.py::call", lines=(10,), span=(5, 20)):
        self.snap = DictSnapshot()
        self.anchor = contract("http", "GET /a")
        self.cp = MK.contract_id(self.anchor)
        self.link, self.cs, self.sid = "1" * 32, "2" * 32, "3" * 32
        self.snap.add_node("ContractPoint", self.cp)
        self.snap.add_node("Link", self.link)
        self.snap.add_node("CallSite", self.cs)
        self.snap.add_symbol(self.sid, qname, *span)
        self.snap.add_edge("ON", ("Link", self.link), ("ContractPoint", self.cp))
        self.snap.add_edge("VIA", ("CallSite", self.cs), ("Link", self.link))
        self.snap.add_edge("HAS_CALL", ("Symbol", self.sid), ("CallSite", self.cs))
        self.rec = item(1, self.anchor, sym(qname), lines=lines)
        self.qname, self.span = qname, span

    def chain(self):
        return [hop0("ContractPoint", self.cp), hopn("Link", self.link, "ON", "<-"), hopn("CallSite", self.cs, "VIA", "<-"),
                hopn("Symbol", self.sid, "HAS_CALL", "<-")]

    def item(self, **kw):
        return ans_item(sym_subject(self.sid, self.qname, *self.span), self.chain(), **kw)

    def request(self):
        return request([self.cp])

    def answer(self, items=None, **kw):
        return answer(self.request(), [self.item()] if items is None else items, **kw)

    def run(self, ans=None, snapshot_id=SNAP):
        return {"snapshotId": snapshot_id, "answers": {self.cp: {"request": self.request(), "answer": self.answer() if ans is None else ans}}}

    def score(self, ans=None, records=None, patterns=(), snapshot_id=SNAP):
        return G.score_gate(records or [self.rec], self.run(ans, snapshot_id), self.snap, patterns)

    def outcome(self, ans=None, **kw):
        res = self.score(ans, **kw)
        return res["items"][self.rec["itemId"]], res


# ----------------------------------------------------------------------------- a store fed by the LOADER's row builder

def loader_symbol_props(sid: str, qname: str, start: int, end: int, owned=None, path: str | None = None) -> dict:
    """The Symbol node properties exactly as ``ccg.assemble.rows`` writes them (``ownedRegions`` flat), built by the producer's own function."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ccg.assemble import rows as R
    b = R._Builder()
    rec = {"type": "symbol", "id": sid, "repoId": "r", "fileId": "f" * 32, "qname": qname, "path": path or qname.split("::")[0],
           "lineStart": start, "lineEnd": end, "kind": "function", "name": qname.split("::")[-1], **({"ownedRegions": owned} if owned else {})}
    R._record(b, rec, "u", "r", {}, {sid}, {}, {}, {sid: 1})
    return b.rows.nodes["Symbol"][0]


class LoaderRunner:
    """A read-only runner (``run(cypher, params) -> rows``) over Symbol rows built by the loader, answering ``StoreSnapshot``'s symbol queries."""

    def __init__(self, symbols: list[dict]):
        self.symbols = symbols

    @staticmethod
    def _row(p):
        return [p["symbolId"], p["qname"], p["path"], p["lineStart"], p["lineEnd"], p.get("ownedRegions")]

    def __call__(self, cypher: str, params=None):
        if "(s:Symbol {symbolId: $id})" in cypher:
            return [self._row(p) for p in self.symbols if p["symbolId"] == params["id"]]
        if "(s:Symbol {path: $p})" in cypher:
            return [self._row(p) for p in sorted(self.symbols, key=lambda p: p["symbolId"]) if p["path"] == params["p"]]
        raise AssertionError(f"unexpected query: {cypher}")


def clean_report(system: str = "ccg", n: int = 1):
    """The REAL ``ccg.refresh.RefreshReport`` of a complete build: ``n`` units, every one ``fresh`` (what the gate build requires of ``refresh``)."""
    from ccg.refresh import RefreshReport, UnitReport
    rep = RefreshReport(system)
    for i in range(n):
        rep.units.append(UnitReport(unit_id=f"u{i}", repo_id="r", root_path=f"p{i}", language="python", status="fresh", built_commit="c" * 40))
    return rep
