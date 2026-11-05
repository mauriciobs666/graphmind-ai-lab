"""S28 (P0-21) core: read an extract.sc facts file, derive the bundle-style points and the projection's DATA /
CONTROLLED_BY edges per method, and join the S56 flow labels to them (ML §7.3, flow-brief template "What S28 does
with the labels").  No Joern, no network; reads only the facts file, the clean corpus tree, `sample/<lang>-flow.jsonl`
and the label set it is handed.  The point derivation mirrors `vertical-slice/normalize.py` (the S11 normalizer for
Python and TS/JS), generalised to the other frontends; every deviation is a named definition in DEFINITIONS.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
CORPORA = CPG / ".ccg-work" / "corpora"

DEFINITIONS = {
    "points": "Points per method are those of bundle spec §5.4-§5.6 as normalize.py derives them: param (index), ret "
              "(ordinal over the method's own RETURNs sorted by line, column, id), cond (every CDG source), callsite "
              "(callee name + ordinal over same-name call sites sorted by the position of the callee's last identifier, "
              "found in the SOURCE from the call's start; a call whose name never occurs in the source is synthetic and is "
              "not a point, and an edge touching it is dropped, as normalize.py drops it).",
    "paramIndex": "spec index = Joern index - 1 after dropping the implicit `this` (index 0, named `this`) for javascript, "
                  "java, csharp and cpp; python: Joern index 0 (self/cls) is -1, then index - 1; go: Joern index 0 (the receiver) is -1, then index - 1; a C# extension-method `this` parameter "
                  "(Joern code starts `this `) is -1 and the rest count from 0.",
    "edges": "DATA edge = (def point, use point) from the `d` facts, def != use; CONTROLLED_BY edge = (point, cond) from the "
             "`cb` facts with CDG self-edges dropped and (point, cond) pairs de-duplicated (-spec2 §1.4 G-4).",
    "join": "param by spec index (the name is only a sanity diagnostic); ret by ordinal (line must equal the label's, else the "
            "unique ret on the label's line); call by (calleeName, ordinal) when that call site is on the label's line, else "
            "the unique call site of that calleeName on the label's line (tag `ordinalShift`), else unresolved; cond by line "
            "and collapsed code (equal, else equal ignoring all whitespace [Joern re-spaces `(int, float)` as `(int,float)`], else one "
            "contains the other, on that line; several candidates at the same tier = unresolved).  A pair with an unresolved endpoint is a miss, listed as `endpoint-unresolved`.",
}


def collapse(s: str, limit: int = 200) -> str:
    return " ".join(s.split())[:limit]


def int_(v, d=-1):
    try:
        return int(v)
    except Exception:
        return d


# ---------------------------------------------------------------------------------------------- facts
LINE_OFFSET = {"csharp": 1}  # csharpsrc2cpg numbers lines from 0 (observed: the callee name sits on l+1 for 3516 of 4000 calls)


def load_facts(path: Path, lang: str = "") -> dict:
    off = LINE_OFFSET.get(lang, 0)
    F = {"m": {}, "p": defaultdict(list), "r": defaultdict(list), "c": defaultdict(list), "cs": defaultdict(list),
         "d": defaultdict(list), "cb": defaultdict(list), "t": {}, "f": [], "stats": None}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            k = o["k"]
            if off and k in ("m", "p", "r", "c", "cs") and isinstance(o.get("l"), int) and o["l"] >= 0:
                o["l"] += off
                if k == "m" and isinstance(o.get("le"), int) and o["le"] >= 0:
                    o["le"] += off
            if k == "m":
                F["m"][o["id"]] = o
            elif k in ("p", "r", "c", "cs", "d", "cb"):
                F[k][o["m"]].append(o)
            elif k == "t":
                F["t"][o["id"]] = o
            elif k == "f":
                F["f"].append(o["name"])
            elif k == "stats":
                F["stats"] = o
    return F


class Src:
    def __init__(self, root: Path):
        self.root = root
        self._c = {}

    def lines(self, rel: str):
        if rel not in self._c:
            p = self.root / rel
            try:
                self._c[rel] = p.read_text(encoding="utf-8", errors="replace").split("\n")
            except Exception:
                self._c[rel] = None
        return self._c[rel]


def file_of(F, m) -> str:
    f = m["file"]
    if f and f != "<empty>" and not f.startswith("<"):
        return f
    t = F["t"].get(m["pid"])
    if t and t["file"] and not t["file"].startswith("<"):
        return t["file"]
    return f


# ---------------------------------------------------------------------------------------------- call sites
def callee_name(c: dict) -> str:
    n = c["name"]
    if n in ("<operator>.new", "<init>"):
        m = re.match(r"\s*new\s+([A-Za-z_$][\w$.:]*)", c["code"])
        if m:
            return re.split(r"[.:]+", m.group(1))[-1]
        m = re.match(r"\s*(super|this)\s*\(", c["code"])
        if m:
            return m.group(1)
        return "<dynamic>"
    if "." in n and not n.startswith("<"):  # c2cpg names a constructor call by its qualified name (`ns.Type`)
        n = n.rsplit(".", 1)[1]
    return n or "<dynamic>"


def locate_callees(lines, calls, col_slack=1):
    """(line, col) of each call's callee last identifier from source; calls whose name is absent are not returned."""
    starts = defaultdict(list)
    for c in calls:
        starts[(c["l"], c["c"], callee_name(c))].append(c)
    pos = {}
    for (ln, cl, name), grp in starts.items():
        if name == "<dynamic>" or not (1 <= ln <= len(lines)):
            continue
        grp.sort(key=lambda c: (len(c["code"]), int(c["id"])))
        off = max(cl - 1 - col_slack, 0)
        if off > len(lines[ln - 1]):
            off = 0
        text = "\n".join(lines[ln - 1: ln + 400])[off:]
        isnew = grp[0]["name"] in ("<operator>.new", "<init>") and callee_name(grp[0]) not in ("super", "this")
        pat = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])" +
                         ("" if isnew else r"\s*(?:\?\.|!)?\s*(?:<[^()]*?>|\[[^()\[\]]*?\])?\s*\("))
        hits = list(pat.finditer(text))
        for k, c in enumerate(grp):
            if k < len(hits):
                before = text[:hits[k].start()]
                dl = before.count("\n")
                col = (len(before) - (before.rfind("\n") + 1)) + (off if dl == 0 else 0)
                pos[c["id"]] = (ln + dl, col)
    return pos


# ---------------------------------------------------------------------------------------------- params
def param_map(lang: str, m: dict, ps: list) -> dict:
    """Joern parameter index -> spec index (None = not a bundle param, e.g. the implicit this)."""
    ps = sorted(ps, key=lambda p: p["i"])
    out = {}
    if lang == "python":
        # pysrc2cpg numbers a method's `self`/`cls` as index 0 and every other parameter from 1 (a plain function starts at 1)
        return {p["i"]: (-1 if p["i"] == 0 else p["i"] - 1) for p in ps}
    if lang == "go":
        return go_param_map(m, ps)
    # javascript, java, csharp, cpp: Joern index 0 is the implicit `this`
    ext = lang == "csharp" and any(p["i"] >= 1 and p.get("code", "").startswith("this ") for p in ps)
    if ext:
        idx = 0
        for p in ps:
            if p["i"] == 0 and p["name"] == "this":
                out[p["i"]] = None
            elif p.get("code", "").startswith("this "):
                out[p["i"]] = -1
            else:
                out[p["i"]] = idx
                idx += 1
        return out
    for p in ps:
        if p["i"] == 0 and p["name"] == "this":
            out[p["i"]] = None
        else:
            out[p["i"]] = p["i"] - 1
    return out


def go_param_map(m, ps):
    # gosrc2cpg: the receiver, when there is one, is index 0 (code `(b *T)`); ordinary parameters start at 1.
    return {p["i"]: (-1 if p["i"] == 0 else p["i"] - 1) for p in ps}


# ---------------------------------------------------------------------------------------------- per-method projection
class Method:
    pass


def build_method(lang, F, src: Src, mid: str, file: str):
    m = F["m"][mid]
    M = Method()
    M.id = mid
    M.file = file
    M.name = m["name"]
    M.full = m["full"]
    M.l = m["l"]
    M.le = m["le"]
    ps = F["p"].get(mid, [])
    pm = param_map(lang, m, ps)
    M.params = {}  # spec index -> (point key, name)
    for p in ps:
        si = pm.get(p["i"])
        if si is not None:
            M.params[si] = (("p", si), p["name"])
    M.pidx = {p["i"]: pm.get(p["i"]) for p in ps}
    rets = sorted(F["r"].get(mid, []), key=lambda r: (r["l"], r["c"], int(r["id"])))
    M.rets = [(("r", r["id"]), k, r["l"]) for k, r in enumerate(rets)]
    M.conds = [(("c", c["id"]), c["l"], collapse(c["code"])) for c in F["c"].get(mid, [])]
    lines = src.lines(file) or []
    calls = [c for c in F["cs"].get(mid, []) if c["name"] or c["name"] == "<operator>.new"]
    pos = locate_callees(lines, calls)
    keyed = defaultdict(list)
    M.synthetic = 0
    for c in calls:
        if c["id"] in pos:
            keyed[callee_name(c)].append(c)
        else:
            M.synthetic += 1
    M.calls = []  # (point key, calleeName, ordinal, line)
    for name, grp in keyed.items():
        grp.sort(key=lambda c: (pos[c["id"]], len(c["code"]), int(c["id"])))
        for k, c in enumerate(grp):
            M.calls.append((("cs", c["id"]), name, k, pos[c["id"]][0]))
    valid = {p[0] for p in M.params.values()} | {r[0] for r in M.rets} | {c[0] for c in M.conds} | {c[0] for c in M.calls}
    # edges
    M.data = set()
    M.droppedData = 0
    for d in F["d"].get(mid, []):
        use = (d["uk"], d["uid"])
        if d["dk"] == "cs":
            df = ("cs", d["did"])
        elif d["dk"] == "p":
            si = M.pidx.get(d["di"])
            df = ("p", si) if si is not None else None
        else:
            df = None
        if df is None or df not in valid or use not in valid:
            M.droppedData += 1
            continue
        if df != use:
            M.data.add((df, use))
    M.ctl = set()
    M.droppedCtl = 0
    for e in F["cb"].get(mid, []):
        pt = (e["pk"], e["pid"])
        cd = ("c", e["cond"])
        if pt not in valid or cd not in valid:
            M.droppedCtl += 1
            continue
        if pt != cd:
            M.ctl.add((pt, cd))
    adj = defaultdict(set)
    for a, b in M.data:
        adj[a].add(b)
    M.adj = adj
    M.points = len(valid)
    return M


def reach(M, a, b):
    """directed chain of >= 1 DATA edges from a to b; returns hop count of the shortest chain, or None."""
    seen = {a}
    frontier = [a]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for x in frontier:
            for y in M.adj.get(x, ()):
                if y == b:
                    return d
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        frontier = nxt
    return None


# ---------------------------------------------------------------------------------------------- endpoint join
def resolve_endpoint(M, ep: dict):
    """-> (point key | None, detail dict)."""
    pt = ep["point"]
    if pt == "param":
        hit = M.params.get(ep["index"])
        det = {"want": "param", "index": ep["index"], "name": ep.get("name")}
        if hit is None:
            byname = [si for si, (k, n) in M.params.items() if n.lstrip("*") == ep.get("name", "").lstrip("*")]
            det.update(reason="no-param-index", projectionParams={si: n for si, (k, n) in sorted(M.params.items())},
                       nameHit=byname)
            return None, det
        if hit[1].lstrip("*") != (ep.get("name") or "").lstrip("*"):
            det["nameMismatch"] = hit[1]
        return hit[0], det
    if pt == "ret":
        line = ep["line"]
        cand = [r for r in M.rets if r[1] == ep["ordinal"]]
        det = {"want": "ret", "ordinal": ep["ordinal"], "line": line}
        if cand and cand[0][2] == line:
            return cand[0][0], det
        online = [r for r in M.rets if r[2] == line]
        if len(online) == 1:
            det["via"] = "unique-ret-on-line"
            return online[0][0], det
        det.update(reason="ret-none" if not online else "ret-several", projectionRets=[[r[1], r[2]] for r in M.rets])
        return None, det
    if pt == "call":
        name, k, line = ep["calleeName"], ep["ordinal"], ep["line"]
        det = {"want": "call", "calleeName": name, "ordinal": k, "line": line}
        prim = [c for c in M.calls if c[1] == name and c[2] == k]
        if prim and prim[0][3] == line:
            return prim[0][0], det
        online = [c for c in M.calls if c[1] == name and c[3] == line]
        if len(online) == 1:
            det["via"] = "ordinalShift"
            return online[0][0], det
        det["reason"] = "call-several-on-line" if online else "call-none"
        det["projectionSameName"] = [[c[2], c[3]] for c in M.calls if c[1] == name][:12]
        return None, det
    if pt == "cond":
        line, code = ep["line"], collapse(ep["code"])
        det = {"want": "cond", "line": line, "code": code}
        online = [c for c in M.conds if c[1] == line]
        eq = [c for c in online if c[2] == code]
        if len(eq) == 1:
            return eq[0][0], det
        if len(eq) > 1:
            det.update(reason="cond-several-equal", n=len(eq))
            return None, det
        nows = lambda x: "".join(x.split())  # noqa: E731  (Joern re-spaces commas/operators; the text is the same)
        ew = [c for c in online if nows(c[2]) == nows(code)]
        if len(ew) == 1:
            det["via"] = "equal-ignoring-whitespace"
            return ew[0][0], det
        if len(ew) > 1:
            det.update(reason="cond-several-equal", n=len(ew))
            return None, det
        ct = [c for c in online if c[2] and code and (nows(c[2]) in nows(code) or nows(code) in nows(c[2]))]
        if len(ct) == 1:
            det["via"] = "contains"
            return ct[0][0], det
        det.update(reason="cond-several-contains" if ct else "cond-none",
                   projectionConds=[[c[1], c[2][:80]] for c in M.conds if abs(c[1] - line) <= 1][:8])
        return None, det
    raise ValueError(pt)


def point_label(M, key) -> str:
    """stable text for a point (used in the spurious pair key)"""
    if key[0] == "p":
        return f"param:{key[1]}"
    for c in M.calls:
        if c[0] == key:
            return f"call:{c[1]}#{c[2]}@{c[3]}"
    for r in M.rets:
        if r[0] == key:
            return f"ret:{r[1]}@{r[2]}"
    for c in M.conds:
        if c[0] == key:
            return f"cond:{c[1]}:{c[2]}"
    return str(key)


def point_obj(M, key) -> dict:
    if key[0] == "p":
        return {"point": "param", "index": key[1], "name": M.params[key[1]][1]}
    for c in M.calls:
        if c[0] == key:
            return {"point": "call", "calleeName": c[1], "ordinal": c[2], "line": c[3]}
    for r in M.rets:
        if r[0] == key:
            return {"point": "ret", "ordinal": r[1], "line": r[2]}
    for c in M.conds:
        if c[0] == key:
            return {"point": "cond", "line": c[1], "code": c[2]}
    return {"point": "?", "key": list(key)}
