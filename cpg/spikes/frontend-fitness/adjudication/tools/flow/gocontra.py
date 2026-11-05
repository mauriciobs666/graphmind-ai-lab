import os, json, sys
from pathlib import Path
from collections import defaultdict
FF = Path(__file__).resolve().parents[3]                      # cpg/spikes/frontend-fitness
CORP = FF.parents[1] / ".ccg-work" / "corpora"                 # cpg/.ccg-work/corpora
S = Path(os.environ.get("FLOW_WORK", "/tmp/ccg-flow-adj"))     # work dir: run/, runfix/, var/ (outside the repo)
TOOLS = Path(__file__).resolve().parent; sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(FF))
import s28_metrics as SM, s28_core as C
lang = "go"
samp = {json.loads(l)["ordinal"]: json.loads(l) for l in open(FF/"sample"/"go-flow.jsonl")}
pairs = [json.loads(x) for x in open(FF/"adjudication"/"go-flow-audit.jsonl")]
pairs = [d for d in pairs if d.get("decision") == "invalid" and d.get("convention") in ("callee-value", "field-write")]
src = C.Src(C.CORPORA/lang/"src")
for v in ("shipped", "R1"):
    F = C.load_facts(S/"var"/f"go.{v}.jsonl", lang)
    byfile = defaultdict(list)
    for mid, m in F["m"].items(): byfile[C.file_of(F, m)].append(m)
    for d in pairs:
        s = samp[d["ordinal"]]; m, how = SM.match_method(F, byfile, s)
        if not m: print(v, d["ordinal"], d["convention"], "method-absent"); continue
        M = C.build_method(lang, F, src, m["id"], C.file_of(F, m))
        sk, sd = C.resolve_endpoint(M, d["pair"]["source"]); tk, td = C.resolve_endpoint(M, d["pair"]["sink"])
        st = "endpoint-unresolved" if sk is None or tk is None else ("connected" if C.reach(M, sk, tk) else "not-connected")
        print(v, "o%d" % d["ordinal"], d["convention"], d["pair"]["sink"].get("calleeName"), st, td.get("reason", ""))
