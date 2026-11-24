"""CO-1 driver (adapted copy of p1t-20/asks.py: same request_of/ask_anchor/matches/judge; production code untouched)

Original docstring: P1T-20 driver (AC-1 HTTP, AC-2 MCP and wiring, AC-4 scripts on DEV anchors) and P1T-19 (AC-11 answer forms), against the P1T-38 build.

Dev anchors only (cpg/eval/dev/ac5-*.jsonl); no held-out file or custodian directory is read. No engine, client or tool code is modified: this only
imports them. Env: P1T_REPO (committed clone), P1T_OUT (private area), CCG_CREDENTIALS_DIR, P1T_PORT, plus CCG_SYSTEM.
Pass rule (docs/test-plans/connected-code-graph-p1.md section 7, "P1T-20 sample and pass rule"): per sampled anchor, one structured impact request;
every dev item of the anchor is listed as confirmed or possible, and its chain has file and line on every located hop (AC-18 reading of the helpers);
AC-2 wiring items are `config` items; AC-4's launch chain is the Link's bound[] (observed and reported separately). A miss is a defect (per anchor, no
recall floor) except key-error (gold wrong, with evidence; none is claimed here). A filter matching no anchor => blocked.
Usage: asks.py p1t20 | p1t19
"""
import copy, json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

REPO, OUT, PORT = Path(os.environ["P1T_REPO"]), Path(os.environ["P1T_OUT"]), int(os.environ["P1T_PORT"])
assert PORT >= 16500 and PORT not in (6379, 6390, 16390, 16400)
SYSTEM = "graphmind-ai-lab"
os.environ["CCG_SYSTEM"] = SYSTEM
sys.path[:0] = [str(REPO / "cpg"), str(REPO / "cpg/eval/tools"), str(REPO / "cpg/tests")]
import ccg                                                       # noqa: E402
assert Path(ccg.__file__).resolve().is_relative_to(REPO.resolve())
import match_keys as MK                                          # noqa: E402  (read-only: the EP's own E() and h1_symbol_credit)
import gate as G                                                 # noqa: E402  (stored_keypath only)
import fixture_gate_p1_ac_helpers as AC                          # noqa: E402  (the AC-18 chain checker)
from clients.common import Client, ClientError                   # noqa: E402

DEV = REPO / "cpg/eval/dev"
FILES = {k: [json.loads(l) for l in (DEV / f"ac5-{k}.jsonl").read_text().splitlines() if l.strip()] for k in ("http", "mcp", "scripts", "wiring")}
client = Client(SYSTEM, host="127.0.0.1", port=PORT)


def request_of(anchor):
    t = anchor["type"]
    if t == "contract":
        return {"useCase": "impact", "anchor": {"kind": "contract", "key": anchor["keyNorm"], "contractKind": anchor["kind"]}}
    if t == "symbol":
        return {"useCase": "impact", "anchor": {"kind": "symbol", "repo": anchor["repoId"], "qname": anchor["qname"]}}
    return None                                                   # a unit anchor is asked from a custodian-side probe (S47): not asked here


_cache = {}
def ask_anchor(anchor):
    key = json.dumps(anchor, sort_keys=True)
    if key in _cache:
        return _cache[key]
    req = request_of(anchor)
    if req is None:
        out = {"error": "unit-anchor-not-asked"}
    else:
        t = time.time()
        try:
            out = {"doc": client.ask(req)}
        except ClientError as e:
            out = {"error": e.code, "detail": {k: (v if k != "candidates" else len(v)) for k, v in (e.detail or {}).items()}}
        except Exception as e:                                    # noqa: BLE001 - recorded as a harness fault
            out = {"error": "harness:" + type(e).__name__}
        out["s"] = round(time.time() - t, 2)
    _cache[key] = out
    return out


def regions(s):
    r = s.get("ownedRegions")
    if r and not all(isinstance(x, (list, tuple)) for x in r):
        r = [[r[i], r[i + 1]] for i in range(0, len(r), 2)]
    return r


def matches(x, rec):
    """The EP H1 identity of a dev item against an answer item's own subject (the answer's copies; no store read)."""
    s, e = x["subject"], rec["impacted"]
    if e["type"] == "symbol":
        if s.get("type") != "symbol":
            return False
        sub = {"qname": s["qname"], "lineStart": s["lineStart"], "lineEnd": s["lineEnd"], "ownedRegions": regions(s)}
        try:
            return bool(MK.h1_symbol_credit(sub, {"qname": e["qname"]}, rec["lines"]))
        except Exception:                                         # noqa: BLE001
            return False
    if e["type"] == "config":
        if s.get("type") != "config" or s.get("path") != e["path"]:
            return False
        got, want = G.stored_keypath(s["keyPath"]), G.stored_keypath(e["keyPath"])
        return got == want or got.startswith(want + ".") or got.startswith(want + "[")
    return False


def members(doc):
    yield from doc["items"]
    for g in doc["groups"]:
        yield from g["members"]


def judge(rec, res):
    """-> dict(listed, ok, cls, ...) for one dev item against one ask result."""
    if "doc" not in res:
        return {"hit": False, "cls": "ask:" + res["error"]}
    doc = res["doc"]
    cands = [x for x in members(doc) if matches(x, rec)]
    if not cands:
        return {"hit": False, "cls": "not-listed"}
    best = None
    for x in cands:
        if x["status"] not in ("confirmed", "possible"):
            continue
        sub = {"interpretation": doc["interpretation"], "items": [x], "groups": []}
        viol = AC.ac18_violations(sub, "item")
        if viol:
            best = best or {"hit": False, "cls": "chain-no-file-line", "viol": len(viol)}
            continue
        bound = any(h["kind"] == "Link" and h.get("bound") for h in x["chain"])
        return {"hit": True, "cls": "hit", "status": x["status"], "distance": x["distance"], "chainLen": len(x["chain"]),
                "boundOnLink": bound, "subjectType": x["subject"]["type"]}
    return best or {"hit": False, "cls": "listed-bad-status"}


def symbol_in_store(rec):
    """Diagnostic for a miss: does the store hold a Symbol (or ConfigEntry) for the dev item's identity? raw read as ccg_raw."""
    e = rec["impacted"]
    try:
        if e["type"] == "symbol":
            q = f"MATCH (s:Symbol {{repoId: {json.dumps(e['repoId'])}, path: {json.dumps(e['path'])}}}) RETURN count(s)"
        else:
            q = f"MATCH (c:ConfigEntry {{repoId: {json.dumps(e['repoId'])}, path: {json.dumps(e['path'])}}}) RETURN count(c)"
        return client.raw_query(q)["rows"][0][0]
    except Exception as ex:                                       # noqa: BLE001
        return "raw-error:" + type(ex).__name__



def tally(rows):
    return {"hits": sum(r["hit"] for r in rows), "total": len(rows), "misses": dict(Counter(r["cls"] for r in rows if not r["hit"]))}


def anchor_view(doc):
    """Compact view of one answer: every item's subject (path/qname or config keyPath), status, distance, first-hop kinds."""
    out = []
    for x in members(doc):
        s = x["subject"]
        out.append({"type": s.get("type"), "id": s.get("qname") or (s.get("path"), s.get("keyPath")), "status": x["status"], "distance": x["distance"],
                    "chain": [h["kind"] for h in x["chain"]]})
    return out


def graph_diag(anchor):
    """Why-diagnostic for a symbol anchor: does the Symbol exist, and which CALLS edges / call sites reach it (raw read as ccg_raw)."""
    if anchor["type"] != "symbol":
        return {}
    q = ("MATCH (s:Symbol {repoId: %s, qname: %s}) OPTIONAL MATCH (c)-[r:CALLS]->(s) "
         "RETURN count(DISTINCT s), count(DISTINCT c)" % (json.dumps(anchor["repoId"]), json.dumps(anchor["qname"])))
    try:
        rows = client.raw_query(q)["rows"]
        return {"anchorSymbols": rows[0][0], "callsEdgesIn": rows[0][1]}
    except Exception as ex:                                       # noqa: BLE001
        return {"diag-error": type(ex).__name__}


def main():
    out = {"head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(), "port": PORT, "strata": {}}
    for k in ("scripts", "http", "wiring"):
        recs = FILES[k]
        rows = []
        for r in recs:
            res = ask_anchor(r["anchor"])
            j = judge(r, res)
            j.update(itemId=r["itemId"], anchorId=r["anchorId"], anchorType=r["anchor"]["type"], impactedType=r["impacted"]["type"],
                     impacted=r["impacted"].get("qname") or (r["impacted"]["path"], r["impacted"].get("keyPath")), lines=r["lines"])
            if r["anchor"]["type"] == "unit":
                j["cls"] = "unit-anchor-not-asked"
            if not j["hit"]:
                j["inStore"] = symbol_in_store(r)
            rows.append(j)
        anchors = {}
        for r in recs:
            aid = r["anchorId"]
            if aid in anchors:
                continue
            res = ask_anchor(r["anchor"])
            anchors[aid] = {"anchor": r["anchor"], "ask": ("error:" + res["error"]) if "doc" not in res else
                            {"items": anchor_view(res["doc"]), "interpretation": res["doc"]["interpretation"].get("status")},
                            **graph_diag(r["anchor"])}
        out["strata"][k] = {"items": len(recs), "anchors": len(anchors), "tally": tally(rows), "rows": rows, "anchorAnswers": anchors}
        print(k, "items", len(recs), "anchors", len(anchors), tally(rows))
    (OUT / "co1-asks.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
