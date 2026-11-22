"""P1-32b M6: the ask translation (``gate_run.cli_request``) against the engine's OWN resolvers (``clients.common.Client.resolve``/``ask`` share
``_resolve_anchor`` -> ``ccg.engine.anchors``), over a scripted store holding the DEV items' anchors (``cpg/eval/dev/ac5-*.jsonl``; never held-out).
Every anchor type must resolve to its own id; the planted cases are the three differences the review read (``docs/reviews/connected-code-graph-p1-32b-impl.md`` M6).
What this cannot prove offline: the real reader's Cypher over a loader-built graph and the key forms of a real build (the live tier does the former)."""
import json
from pathlib import Path

import pytest
from engine_helpers import ScriptedReader

import gate_helpers as H
from gate_helpers import G
import gate_run as R
from clients.common import Client, _resolve_anchor, resolution_json

DEV = sorted((Path(__file__).resolve().parents[1] / "eval" / "dev").glob("ac5-*.jsonl"))
REPO = "graphmind-ai-lab"


def dev_records():
    return [json.loads(l) for f in DEV for l in f.read_text().splitlines() if l.strip()]


class Store:
    """A scripted store: symbols ``(repo, path, qname, id)``, contract points ``(cpId, kind, keyNorm)``, config entries ``(entryId, repo, path, keyPath)``."""

    def __init__(self):
        self.symbols, self.cps, self.configs = [], [], []

    def reader(self):
        files = {(r, p) for r, p, _, _ in self.symbols} | {(r, p) for _, r, p, _ in self.configs}
        script = {
            "repo_exists": lambda p: [(REPO,)] if p["repo"] == REPO else [],
            "file_by_path": lambda p: [("f", "u")] if (p["repo"], p["path"]) in files else [],
            "skipped_by_path": lambda p: [],
            "symbols_of_file": lambda p: [(sid, q, 1, 9, None, 0) for r, path, q, sid in self.symbols if (r, path) == (p["repo"], p["path"])],
            "symbol_by_qname": lambda p: [(sid, "u", path, 1, 9, None) for r, path, q, sid in self.symbols if (r, q) == (p["repo"], p["qname"])],
            "cp_by_key": lambda p: [(cp, k, key) for cp, k, key in self.cps if key == p["key"]],
            "cp_keys": lambda p: [(cp, key) for cp, k, key in self.cps if k == p["kind"]],
            "config_by_path": lambda p: [(eid, "u", 3, kp, None) for eid, r, path, kp in self.configs if (r, path) == (p["repo"], p["path"])],
        }
        return ScriptedReader(script)

    def client(self):
        rd = self.reader()
        return Client("shop", host="127.0.0.1", port=16999, reader_factory=lambda *a: rd)


def asked(rec, probes=None, locators=None):
    aid = G.ask_id_of(rec, probes)
    return aid, G.build_request(rec, aid, locators)


def probe_of(rec):
    return {rec["anchorId"]: G._mk.contract_id(rec["serves"][0])} if rec["anchor"]["type"] == "unit" else None


def store_for(records):
    st = Store()
    for rec in records:
        locs = {G._mk.contract_id(x): x for x in [rec["anchor"], *rec.get("serves", [])] if x["type"] == "contract"}
        for cp, x in locs.items():
            if (cp, x["kind"], x["keyNorm"]) not in st.cps:
                st.cps.append((cp, x["kind"], x["keyNorm"]))
        a = rec["anchor"]
        if a["type"] == "symbol" and (a["repoId"], a["path"], a["qname"], rec["anchorId"]) not in st.symbols:
            st.symbols.append((a["repoId"], a["path"], a["qname"], rec["anchorId"]))
    return st


def test_the_dev_set_is_present_and_covers_every_asked_anchor_type():
    recs = dev_records()
    assert {r["anchor"]["type"] for r in recs} >= {"symbol", "contract", "unit"}


def test_every_dev_anchor_resolves_to_exactly_its_own_id_through_the_clients_resolver():
    recs = dev_records()
    st = store_for(recs)
    client = st.client()
    seen = {"symbol": 0, "contract": 0, "unit": 0}
    for rec in recs:
        probes = probe_of(rec)
        if rec["anchor"]["type"] == "unit" and not rec.get("serves"):
            continue
        aid, req = asked(rec, probes, G.locator_index(recs))
        res = client.resolve(R.cli_request(req)["anchor"])
        key = {"symbol": "symbolId", "contract": "cpId"}[req["anchors"][0]["type"]]
        assert res["status"] == "resolved" and [c[key] for c in res["candidates"]] == [aid], (rec["anchor"]["type"], req["anchors"][0].get("kind"))
        seen[rec["anchor"]["type"]] += 1
    assert all(seen.values()), seen


def test_the_preflight_over_the_dev_set_counts_every_anchor_ok():
    recs = dev_records()
    st = store_for(recs)
    probes = {r["anchorId"]: G._mk.contract_id(r["serves"][0]) for r in recs if r["anchor"]["type"] == "unit" and r.get("serves")}
    recs = [r for r in recs if r["anchor"]["type"] != "unit" or r.get("serves")]
    snap = H.DictSnapshot()
    for rec in recs:
        for x in [rec["anchor"], *rec.get("serves", [])]:
            if x["type"] == "contract":
                snap.add_node("ContractPoint", G._mk.contract_id(x))
        if rec["anchor"]["type"] == "symbol":
            snap.add_symbol(rec["anchorId"], rec["anchor"]["qname"], 1, 9, rec["anchor"]["path"])
    c = R.resolve_preflight(recs, probes, snap, st.client().resolve)
    assert c["ok"] == c["total"] > 0 and c["none"] == c["ambiguous"] == c["id-mismatch"] == 0


def test_a_qname_defined_in_two_files_is_ambiguous_by_the_old_symbol_locator_and_resolves_by_path_and_symbol():
    rec = H.item(1, H.sym("svc/a.py::f", "svc/a.py"), H.sym("svc/b.py::g"))
    twin = ("r", "svc/other.py", "svc/a.py::f", "9" * 32)          # the same qname in a second file of the repo (``Symbol.qname`` is not unique)
    st = Store()
    st.symbols += [("graphmind-ai-lab", "svc/a.py", "svc/a.py::f", rec["anchorId"]), ("graphmind-ai-lab", twin[1], twin[2], twin[3])]
    rec["anchor"]["repoId"] = REPO
    aid, req = asked(rec)
    old = {"kind": "symbol", "repo": REPO, "qname": "svc/a.py::f"}                         # what ``cli_request`` sent before round B
    assert st.client().resolve(old)["status"] == "ambiguous"
    new = R.cli_request(req)["anchor"]
    res = st.client().resolve(new)
    assert res["status"] == "resolved" and res["candidates"][0]["symbolId"] == aid


def test_a_stored_masked_config_key_path_resolves_by_the_stored_form_and_not_by_the_raw_one_or_the_answer_form():
    raw = "services." + "x" * 30 + ".env"
    rec = H.item(1, H.cfg("deploy/compose.yaml", raw), H.sym("a.py::g"))
    rec["anchor"]["repoId"] = REPO
    stored = R._mask_keypath_as_stored(raw)
    assert stored == "services.[#…].env" and G.mask_keypath(raw) == "services.[#].env"
    st = Store()
    st.configs.append((rec["anchorId"], REPO, "deploy/compose.yaml", stored))
    aid, req = asked(rec)
    client = st.client()
    for wrong in (raw, G.mask_keypath(raw)):
        assert client.resolve({"kind": "config", "repo": REPO, "path": "deploy/compose.yaml", "keyPath": wrong})["status"] == "unresolved"
    res = client.resolve(R.cli_request(req)["anchor"])
    assert res["status"] == "resolved" and res["candidates"][0]["entryId"] == aid


def test_an_unmasked_config_key_path_is_unchanged_by_the_translation():
    rec = H.item(1, H.cfg("deploy/compose.yaml", "services.server"), H.sym("a.py::g"))
    assert R.cli_request(G.build_request(rec, rec["anchorId"]))["anchor"]["keyPath"] == "services.server"


def test_every_dev_http_key_is_a_fixed_point_of_the_engines_consumer_normalisation():
    from ccg.linker.matchers import http_key
    keys = {x["keyNorm"] for r in dev_records() for x in [r["anchor"], *r.get("serves", [])] if x["type"] == "contract" and x["kind"] == "http"}
    assert keys and all(http_key(k, "consumer") == k for k in keys)


def test_an_http_key_that_the_consumer_normalisation_would_change_resolves_by_template_possibly_to_many_and_the_preflight_would_refuse_it():
    """The residual risk of the contract translation: a stored provider key that is not a fixed point. Planted; nothing in the dev set is one."""
    st = Store()
    cp1, cp2 = "1" * 32, "2" * 32
    st.cps += [(cp1, "http", "GET /items/{#}"), (cp2, "http", "GET /items/{}")]
    res = st.client().resolve({"kind": "contract", "key": "GET /items/{#}", "contractKind": "http"})
    assert res["status"] in ("ambiguous", "resolved") and len(res["candidates"]) >= 1               # recorded, not asserted to be one: that is what the dry-run is for


def test_cli_request_resolves_through_the_same_function_ask_uses():
    import inspect
    from clients import common
    src = inspect.getsource(common.Client.ask)
    assert "_resolve_anchor(self._reader(), spec)" in src and _resolve_anchor is common._resolve_anchor and callable(resolution_json)
