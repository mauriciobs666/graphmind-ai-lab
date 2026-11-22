"""Planted anchors for the dry-run over every anchor kind (item 3). Gold fields are cloned from dev records; only the anchor, anchorId and itemId are replaced.
Values come from the stored graph of a built throwaway (cpIds and keys of contract points, one masked config entry); none is held-out data."""
import copy, json, os, sys
sys.path.insert(0, os.path.join(os.environ["REPO"], "cpg/eval/tools"))
import match_keys as mk
from pathlib import Path
cust = Path(os.environ["CUST"]); src = [json.loads(l) for l in (cust / "lists" / "dev-all.jsonl").read_text().splitlines()]
base = {s: next(r for r in src if r["derivedStratum"] == s and r["anchor"]["type"] == "contract") for s in ("http", "mcp")}
sym = next(r for r in src if r["derivedStratum"] == "scripts" and r["anchor"]["type"] == "symbol")
def plant(tpl, anchor, aid, tag):
    r = copy.deepcopy(tpl); r["anchor"] = anchor; r["anchorId"] = aid; r["itemId"] = ("%032x" % abs(hash(tag)) )[-32:].rjust(32, "0"); r["serves"] = []; r["writers"] = []
    return r
P = []
P.append(("http {?} a", plant(base["http"], {"type": "contract", "kind": "http", "keyNorm": "* /channels/{?}/threads"}, None, "p1")))
P.append(("http {?} b", plant(base["http"], {"type": "contract", "kind": "http", "keyNorm": "* /threads/{?}/messages"}, None, "p2")))
P.append(("mcp_tool", plant(base["mcp"], {"type": "contract", "kind": "mcp_tool", "keyNorm": "add"}, None, "p3")))
P.append(("http ?query (not in graph)", plant(base["http"], {"type": "contract", "kind": "http", "keyNorm": "GET /health?x=1"}, None, "p4")))
P.append(("config masked key", plant(base["http"], {"type": "config", "repoId": "graphmind-ai-lab", "path": "model-bench/results/runs/nlq-structured-query-mistralai_ministral-3-3b-instruct-2512-2026-09-20T01:28:55Z.json",
                                                    "keyPath": "items[38].scoreable.[#…]"}, "02330ec5a2e59f9d2f8ad661a9ea99a9", "p5")))
for name, r in P:
    if r["anchor"]["type"] == "contract":
        r["anchorId"] = mk.contract_id(r["anchor"])
bad = copy.deepcopy(sym); bad["anchor"]["qname"] = bad["anchor"]["qname"] + ".__no_such_symbol__"; bad["itemId"] = "f" * 32
for tag in ("ok", "bad"):
    recs = [r for r in src if r is not src[0]] + [r for _, r in P] + ([bad] if tag == "bad" else [])
    (cust / "lists" / f"l-x{tag}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    print(tag, len(recs))
for name, r in P: print(name, r["anchorId"][:8])
