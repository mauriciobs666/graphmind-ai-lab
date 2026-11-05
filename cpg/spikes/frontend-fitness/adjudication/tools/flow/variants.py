"""Score shipped / R1 / R2 / R3 projections per language and dump per-pair status + diagnostics. Scratch, read-only on repo."""
import os, json, sys, glob
from pathlib import Path
from collections import defaultdict, deque
FF = Path(__file__).resolve().parents[3]                      # cpg/spikes/frontend-fitness
CORP = FF.parents[1] / ".ccg-work" / "corpora"                 # cpg/.ccg-work/corpora
S = Path(os.environ.get("FLOW_WORK", "/tmp/ccg-flow-adj"))     # work dir: run/, runfix/, var/ (outside the repo)
TOOLS = Path(__file__).resolve().parent; sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(FF))
import s28_metrics as SM, s28_core as C
lang = sys.argv[1]
f1 = glob.glob(str(S/"run"/lang/"*.facts.jsonl")); f2 = glob.glob(str(S/"runfix"/lang/"*.facts.jsonl"))
assert len(f1) == 1 and len(f2) == 1, (f1, f2)
out = {}
def split(src, keep, dst):
    with open(src) as fh, open(dst, "w") as o:
        for line in fh:
            k = json.loads(line)["k"] if line.startswith('{"k"') else json.loads(line)["k"]
            if k in ("d", "d2", "d4"):
                if k == keep:
                    o.write(line if k == "d" else line.replace(f'"k": "{k}"', '"k": "d"').replace(f'"k":"{k}"', '"k":"d"'))
            else:
                o.write(line)
V = {"shipped": (f1[0], "d"), "R1": (f1[0], "d2"), "R2": (f2[0], "d4")}
samp = [json.loads(l) for l in open(FF/"sample"/f"{lang}-flow.jsonl")]
gold = [json.loads(l) for l in open(FF/"truth"/f"{lang}.flow-final.jsonl")]
tags, removed = SM.load_tags(lang)
src = C.Src(C.CORPORA/lang/"src")
res = {}
for name, (fp, keep) in V.items():
    dst = S/"var"/f"{lang}.{name}.jsonl"; dst.parent.mkdir(exist_ok=True)
    split(fp, keep, dst)
    F, r, answers, spur, mm, mo, per_m = SM.analyse_cfg(lang, "off", {"factsFile": str(dst)}, samp, gold, tags, removed, src, "")
    res[name] = (F, r, answers, per_m)
    print(name, "pooled", r["pooled"]["k"], r["pooled"]["n"], "data", r["data"]["k"], "ctl", r["control"]["k"], "spurious", r["spuriousPairs"]["k"], r["spuriousPairs"]["n"])
# R3: R2 facts, synthetic calls (non-points) and drop-filtered nodes kept as transparent intermediates
F, r2, answers2, per_m2 = res["R2"]
r3 = {}
R3stats = defaultdict(int)
for ordn, M, resolved in per_m2:
    if M is None: continue
    adj = defaultdict(set)
    for d in F["d"].get(M.id, []):
        if d["dk"] == "cs": df = ("cs", d["did"])
        elif d["dk"] == "p":
            si = M.pidx.get(d["di"]); df = ("p", si) if si is not None else None
        else: df = None
        if df is None: continue
        use = (d["uk"], d["uid"])
        if df != use: adj[df].add(use)
    valid = {p[0] for p in M.params.values()} | {x[0] for x in M.rets} | {x[0] for x in M.conds} | {x[0] for x in M.calls}
    # R3 projection edges: valid -> valid through synthetic-only intermediates
    g = next(x for x in gold if x["ordinal"] == ordn)
    gd = set((rk[0], rk[1]) for pr, rk in zip(g["pairs"], resolved) if rk and rk[0] is not None and rk[1] is not None and pr["kind"] == "data")
    e3 = set()
    for a in valid:
        seen = {a}; q = deque([a])
        while q:
            x = q.popleft()
            for y in adj.get(x, ()):
                if y in seen: continue
                seen.add(y)
                if y in valid:
                    if y != a: e3.add((a, y))
                else: q.append(y)
    R3stats["dataEdges"] += len(e3); R3stats["spurData"] += len(e3 - gd); R3stats["ctl"] += len(M.ctl)
    gc = set((rk[1], rk[0]) for pr, rk in zip(g["pairs"], resolved) if rk and rk[0] is not None and rk[1] is not None and pr["kind"] == "control")
    R3stats["spurCtl"] += len(M.ctl - gc)
    for idx, rk in enumerate(resolved):
        if rk is None: continue
        sk, tk = rk
        if sk is None or tk is None: continue
        seen = {sk}; q = deque([sk]); hit = False
        while q and not hit:
            x = q.popleft()
            for y in adj.get(x, ()):
                if y == tk: hit = True; break
                if y not in seen: seen.add(y); q.append(y)
        r3[(ordn, idx)] = hit
# control diagnostics on shipped: transitive CONTROLLED_BY reach (sink -> ... -> cond)
Fs, rs, answers_s, per_m_s = res["shipped"]
ctl_trans = {}; ctl_callcond = {}; sink_is_cond = {}
for ordn, M, resolved in per_m_s:
    if M is None: continue
    up = defaultdict(set)
    for p, c in M.ctl: up[p].add(c)
    condcode = {c[0]: (c[1], c[2]) for c in M.conds}
    for idx, rk in enumerate(resolved):
        if rk is None or rk[0] is None or rk[1] is None: continue
        sk, tk = rk
        seen = {tk}; q = deque([tk]); hops = None; dist = {tk: 0}
        while q:
            x = q.popleft()
            for c in up.get(x, ()):
                if c not in dist:
                    dist[c] = dist[x] + 1; q.append(c)
        ctl_trans[(ordn, idx)] = dist.get(sk)
        sink_is_cond[(ordn, idx)] = bool(tk[0] == 'cs' and ('c', tk[1]) in condcode)
status = {}
for key in answers_s:
    a = answers_s[key]
    status[f"{key[0]}|{key[1]}"] = {"kind": a["kind"], "shipped": a["status"], "R1": res["R1"][2][key]["status"], "R2": res["R2"][2][key]["status"],
        "R3": (True if res["R2"][2][key]["status"] == "connected" else r3.get(key)) , "ctlHops": ctl_trans.get(key), "sinkIsCond": sink_is_cond.get(key)}
json.dump({"R3stats": dict(R3stats), "status": status, "summary": {n: {"pooled": res[n][1]["pooled"], "data": res[n][1]["data"], "control": res[n][1]["control"], "spurious": res[n][1]["spuriousPairs"]} for n in res}}, open(S/"var"/f"{lang}.status.json", "w"))
k3 = sum(1 for v in status.values() if v["R3"]); print("R3 pooled", k3, len(status), "spurious", R3stats["spurData"]+R3stats["spurCtl"], R3stats["dataEdges"]+R3stats["ctl"])
