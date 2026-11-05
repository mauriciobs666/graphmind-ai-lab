import json, hashlib, re, collections
from pathlib import Path
FF = Path("/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness")
SRC = Path("/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/csharp/src")
ADJ = FF / "adjudication"
def rj(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
inp = rj(ADJ / "csharp.input.jsonl")
ref = {r["ordinal"]: r for r in rj(FF / "truth/csharp.ref.jsonl")}
gen = rj(ADJ / "csharp.gen-audit.jsonl")
H = lambda s, sid: hashlib.sha256(f"ccg-p0a-v1-aud|csharp|{s}|{sid}".encode()).hexdigest()

_cache = {}
def lines(path):
    if path not in _cache:
        try: _cache[path] = (SRC / path).read_text(encoding="utf-8-sig").split("\n")
        except Exception as e: _cache[path] = None
    return _cache[path]
def ctx(path, line, before=4, after=5):
    L = lines(path)
    if L is None: return None
    a, b = max(1, line - before), min(len(L), line + after)
    return [f"{i}: {L[i-1]}" for i in range(a, b + 1)]

def callee_name(ct):
    ct = re.sub(r"<[^<>]*(<[^<>]*>[^<>]*)*>", "", ct)  # strip generic args
    ct = ct.replace("new ", "").strip()
    m = re.findall(r"[A-Za-z_]\w*", ct)
    return m[-1] if m else ct

def cell_of(r):
    lab = r["label"]
    if lab["notACall"]: return "notACall"
    parts = sorted({(c["kind"], c.get("package") or c["path"]) for c in lab["callees"]})
    return "|".join(f"{k}:{v}" for k, v in parts) if parts else "empty"

def stratum_of(r):
    if r["status"] != "agreed": return None
    lab = r["label"]
    inc = r["inCorpusLabel"] or r["declaredLabel"]
    if inc: return "AI"
    return "AS" if r["reference"]["static"]["kind"] == "silent" else "AE"

strata = collections.defaultdict(list)
for r in inp:
    s = stratum_of(r)
    if s: strata[s].append(r)
SIZE = {"AI": 60, "AE": 60, "AS": 30}
byord = {r["ordinal"]: r for r in inp}

def site_view(r, extra=None):
    st = r["site"]
    d = {"ordinal": r["ordinal"], "siteId": r["siteId"], "file": f"{st['path']}:{st['line']}", "callText": st["callText"],
         "label": r["label"], "reference": r["reference"], "context": ctx(st["path"], st["line"])}
    if extra: d.update(extra)
    return d

def defctx(r):
    out = []
    for p, dl in r["inCorpusLabel"] + r["declaredLabel"]:
        out.append({"side": "label", "at": f"{p}:{dl}", "context": ctx(p, dl, 2, 4)})
    for p, dl in r["inCorpusReference"]:
        if [p, dl] not in (r["inCorpusLabel"] + r["declaredLabel"]):
            out.append({"side": "reference", "at": f"{p}:{dl}", "context": ctx(p, dl, 2, 4)})
    return out

queue = open(ADJ / "csharp.queue.jsonl", "w", encoding="utf-8")
audit = open(ADJ / "csharp.audit.jsonl", "w", encoding="utf-8")
counts = collections.Counter()
def emit(o):
    queue.write(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n"); queue.flush(); counts[o["category"]] += 1

# 1 needs-adjudication
for r in inp:
    if r["status"] == "needs-adjudication":
        emit({"category": "needs-adjudication", **site_view(r, {"reasons": r["reasons"], "targetContext": defctx(r)})})

# 2 audits
pop = {}
usel = set()
for s in ("AI", "AE", "AS"):
    items = sorted(strata[s], key=lambda r: H(s, r["siteId"]))
    n = min(SIZE[s], len(items)); pop[s] = len(items)
    U = items[:n]; uc = {cell_of(r) for r in U}
    seenC = set(); C = []
    for r in items[n:]:
        c = cell_of(r)
        if c not in uc and c not in seenC:
            seenC.add(c); C.append(r)
    # sibling candidates: same cell or same callee name, other sites of the language, hash order
    for draw, lst in (("U", U), ("C", C)):
        for rank, r in enumerate(lst, 1):
            cell = cell_of(r); nm = callee_name(r["site"]["callText"])
            audit.write(json.dumps({"stratum": s, "draw": draw, "rank": rank, "ordinal": r["ordinal"], "siteId": r["siteId"],
                "hash": H(s, r["siteId"]), "cell": cell, "stratumN": len(items), "n": n if draw == "U" else None, "outcome": None}) + "\n")
            sib = [o["ordinal"] for o in sorted(inp, key=lambda o: H(s, o["siteId"]))
                   if o["ordinal"] != r["ordinal"] and (cell_of(o) == cell or callee_name(o["site"]["callText"]) == nm)]
            emit({"category": f"audit-{draw}", "stratum": s, "rank": rank, "cell": cell, "stratumN": len(items),
                  **site_view(r, {"targetContext": defctx(r)}), "siblingCandidatesInSweepOrder": sib,
                  "siblingKey": {"cell": cell, "calleeName": nm}})
audit.close()

# 3 genaudit
for g in gen:
    r = byord[g["ordinal"]]
    emit({"category": "genaudit", "stratum": g["stratum"], "rank": g["rank"], "classes": g["classes"], **site_view(r, {"targetContext": defctx(r)})})

# 4 census
cen = collections.defaultdict(lambda: {"labelKinds": set(), "refKinds": set(), "ordinals": set()})
for r in inp:
    lab = r["label"]
    for c in lab["callees"]:
        if c["kind"] in ("internal", "declared"):
            e = cen[(c["path"], c["defLine"])]; e["labelKinds"].add(c["kind"]); e["ordinals"].add(r["ordinal"]); e.setdefault("qname", c["qname"])
    rf = ref[r["ordinal"]]
    for t in [rf["static"]] + list(rf.get("impls", [])):
        if t.get("kind", "internal") in ("internal", "declared") and "path" in t:
            e = cen[(t["path"], t["defLine"])]; e["refKinds"].add(t.get("kind", "impl")); e["ordinals"].add(r["ordinal"]); e.setdefault("qname", t.get("qname"))
def fcform(path, dl):
    L = lines(path)
    if not L: return "unreadable"
    txt = ""
    for i in range(dl - 1, min(len(L), dl + 14)):
        txt += L[i] + "\n"
        d = 0
        for ch_i, ch in enumerate(L[i]):
            pass
    # first of '{', '=>', ';' at paren depth 0 after the declaration start
    d = 0; i = 0
    while i < len(txt):
        ch = txt[i]
        if ch in "([": d += 1
        elif ch in ")]": d -= 1
        elif d <= 0:
            if txt.startswith("=>", i): return "body (expression-bodied =>)"
            if ch == "{": return "body ({)"
            if ch == ";": return "bodyless (;)"
        i += 1
    return "unclear"
cn = 0
for (p, dl), e in sorted(cen.items()):
    cn += 1
    emit({"category": "census", "file": f"{p}:{dl}", "qname": e.get("qname"), "labelKinds": sorted(e["labelKinds"]),
          "refKinds": sorted(e["refKinds"]), "sites": sorted(e["ordinals"]), "fcPreliminaryForm": fcform(p, dl),
          "context": ctx(p, dl, 1, 6)})
queue.close()
print(dict(counts)); print("strata", pop)
print("census distinct", cn, "declared-label", sum(1 for e in cen.values() if "declared" in e["labelKinds"]))
