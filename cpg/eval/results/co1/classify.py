"""CO-1 classification (dev gold only): joins the gold items of scripts (49), http (18) and wiring (11) with (a) the engine answers (co1-asks.json),
(b) the REAL extractor records of the build (bundles in $CO1_BUNDLES: call sites + valueSets of each impacted symbol), (c) the snapshot's file list
(git ls-files of the clone) for the tier-2 uniqueness count. The mechanism label per item is a reading of the source line (MECH below, by dev item order);
everything else is computed. Output: classification.json (+ prints the aggregates used by the report)."""
import json, os, glob, re, subprocess
from collections import Counter
from pathlib import Path
REPO = Path(os.environ["P1T_REPO"]); B = Path(os.environ["CO1_BUNDLES"]); OUT = Path(os.environ["P1T_OUT"])
files = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True).stdout.split("\n")
scripts = [f for f in files if f.endswith((".sh", ".py"))]
sym, cs_by, vs = {}, {}, {}
for f in glob.glob(str(B / "*.bundle.jsonl")):
    for l in open(f):
        r = json.loads(l); t = r.get("type")
        if t == "symbol": sym[r["id"]] = r
        elif t == "callsite": cs_by.setdefault(r["symbolId"], []).append(r)
        elif t == "valueSet": vs.setdefault(r["callSiteId"], {})[r["arg"]] = r["values"]
by_q = {s["qname"]: s for s in sym.values()}
asks = json.load(open(OUT / "co1-asks.json"))
MECH = {  # scripts, by dev item order
 0: "parse-defect (CO-2a) + `. $HERE/x` inside a `bash -c` string", 1: "front-matter hook launch ($HOME/.claude path)",
 2: "prompt mention (inline span)", 3: "computed command word `$REPO_DIR/scripts/x.sh` (exec path)", 4: "exec with computed path (`$(dirname ..)/../../scripts/x.sh`)",
 5: "exec with computed path (`$(dirname ..)/../../scripts/x.sh`)", 6: "curl URL in shell (http consumer)", 7: "`.`/source with `$(dirname \"$0\")/shlib/x.sh`",
 8: "computed command word `$REPO_DIR/scripts/x.sh` (exec path)", 9: "`.`/source with `$(dirname \"$0\")/shlib/x.sh`", 10: "exec with computed path (`$(dirname ..)/../../scripts/x.sh`)",
 11: "`.`/source with `$(dirname \"$0\")/shlib/x.sh` (gold line is a `# shellcheck source=` comment)", 12: "curl URL in shell (http consumer)",
 13: "computed command word `$HERE/x.sh` (exec path, one-segment tail)", 14: "`.`/source with `$(dirname \"$0\")/shlib/x.sh` (gold line is a comment)",
 15: "path literal `./scripts/x.sh` in shell (gold line holds no call)", 16: "prompt mention (inline span, within a table row)", 17: "prompt mention (inline span)",
 18: "`-m` package (exec $PY -m modelbench)", 19: "computed command word `$REPO_DIR/scripts/x.sh` (exec path)", 20: "computed command word `$REPO_DIR/scripts/x.sh` (exec path)",
 21: "prompt mention (inline span)", 22: "exec with computed path (`$(dirname ..)/../../scripts/x.sh`)", 23: "Python subprocess argument (`[\"bash\", str(_BOOTSTRAP)]`, path from a Path expression)",
 24: "contract-anchored: Python embedded in a shell script calls the MCP tool", 25: "prompt mention (inline span, table row)", 26: "exec with computed path (`$(dirname ..)/../../scripts/x.sh`)",
 27: "prompt mention (inline span, table row)", 28: "front-matter hook launch ($HOME/.claude path)", 29: "prompt mention (inline span and fenced block)",
 30: "computed command word `$REPO_DIR/scripts/x.sh` (exec path)", 31: "prompt mention (inline span, table row)", 32: "`.`/source with `$HERE/x.sh` (one-segment tail)",
 33: "prompt mention in a fenced block only (a fence is not a call)", 34: "parse-defect (CO-2a) + `. $HERE/x` inside `bash -c` strings", 35: "`-m` package (exec $PY -m mcp_monitor)",
 36: "computed command word `${core_dir}/x.sh` (sibling script, one-segment tail)", 37: "prompt mention in a fenced block only (a fence is not a call)",
 38: "curl URL in shell (http consumer)", 39: "prompt mention (inline span, table row)", 40: "prompt mention (inline span, table row)",
 41: "prompt mention (prose, no span: gold line names the instruction)", 42: "prompt mention (prose; gold line holds no script token)", 43: "`$PY $HERE/server.py` (exec; one-segment tail)",
 44: "prompt mention (inline span, table row)", 45: "computed command word `$REPO_DIR/scripts/x.sh` (exec path)", 46: "computed command word `$ROOT/run.sh` (one-segment tail)",
 47: "`$HERE/pipeline.sh` as an argument of awk (file read, not a launch) + parse-defect", 48: "computed command word `$REPO_DIR/scripts/x.sh` (exec path)"}
HTTP_MECH = {0: "curl URL in inline Markdown span (carrier)"}
WIRING_MECH = {0: "server URL (compose service; unit anchor)", 1: "server URL (kiro mcpServers; unit anchor)", 2: "server URL (mcp-monitor server block; unit anchor)",
               3: "tool name in an allow-list (kiro allowedTools)", 4: "tool name in a tools list (kiro tools)", 5: "server URL (.mcp.json; unit anchor)",
               6: "launches a script (tank bash allow-list path)", 7: "launches a script (command in TOML server block)", 8: "launches a script (tank prompt + allow-list)",
               9: "launches a script (tank prompt + allow-list)", 10: "tool name in a prompt (kiro `prompt` field)"}

def tail_of(vals):
    """literal tail of a valueSet alternative list: the const text after the last unknown piece of a template; (tail, headKinds)"""
    out = []
    for v in vals:
        if "template" in v:
            t, heads = "", []
            for p in v["template"]:
                if "const" in p: t += str(p["const"])
                else: t = ""; heads.append(next(iter(p.values())) if isinstance(p, dict) else "?")
            out.append((t, heads))
        elif "const" in v: out.append((str(v["const"]), []))
    return out

def norm_tail(t):
    segs = [s for s in t.split("/") if s not in ("", ".")]
    while segs and segs[0] == "..": segs.pop(0)
    return segs

def suffix_matches(segs):
    if not segs: return 0
    suf = "/".join(segs)
    return sum(1 for f in scripts if f == suf or f.endswith("/" + suf))

def evidence(r):
    a, im, L = r["anchor"], r["impacted"], r["lines"]
    ev = {"symbolInBundles": False, "namingSites": []}
    s = by_q.get(im.get("qname"))
    ev["goldLineText"] = ""
    try:
        txt = (REPO / im["path"]).read_text().splitlines()
        ev["goldLineText"] = " // ".join(txt[l - 1].strip()[:90] for l in L[:2])
        win = " ".join(txt[max(l - 2, 0):l + 1] for l in L) if False else " ".join(x for l in L for x in txt[max(l - 2, 0):l + 1])
    except Exception: win = ""
    key = (a.get("qname") or "").split("::")[0].rsplit("/", 1)[-1] if a.get("type") == "symbol" else ("/health" if "health" in a.get("keyNorm", "") else a.get("keyNorm", ""))
    ev["goldWindowMentionsAnchor"] = bool(key and key.replace(".py", "").replace(".sh", "") in win) if a.get("type") == "symbol" else bool(key and key in win)
    if not s: return ev
    ev["symbolInBundles"] = True
    base = key
    for c in cs_by.get(s["id"], []):
        near = any(abs(c["line"] - l) <= 1 for l in L)
        callee = c["calleeName"]; args = vs.get(c["id"], {})
        hit = None
        ctoks = [callee] if callee != "<dynamic>" else []
        for tok in ctoks:
            segs = norm_tail(tok)
            if segs and segs[-1] == base and a.get("path", "").endswith("/".join(segs)):
                import posixpath
                caller_dir = posixpath.dirname(im["path"]); anc, d = [], caller_dir
                while True:
                    anc.append(d)
                    if not d: break
                    d = posixpath.dirname(d)
                t1 = any(posixpath.normpath(posixpath.join(b, tok)) == a["path"] for b in anc) if not tok.startswith("/") else False
                hit = {"where": "calleeName", "text": tok, "tailSegments": len(segs), "snapshotSuffixMatches": suffix_matches(segs), "tier1ByCallerOrAncestorDir": t1}
        if not hit:
            for k, v in args.items():
                for t, heads in tail_of(v):
                    if base and t.endswith(base) and not re.search(r"\s", t):
                        segs = norm_tail(t); hit = {"where": "valueSet " + k, "tail": t, "headKinds": heads, "tailSegments": len(segs), "snapshotSuffixMatches": suffix_matches(segs)}
                        break
                if hit: break
        if hit:
            hit.update(line=c["line"], near=near, callee=callee, status=c["status"], reason=c["reason"])
            ev["namingSites"].append(hit)
    ev["nSites"] = len(cs_by.get(s["id"], []))
    ev["tier1Resolvable"] = any(x.get("tier1ByCallerOrAncestorDir") for x in ev["namingSites"])
    ev["tier2Resolvable"] = any(x.get("tailSegments", 0) >= 2 and x.get("snapshotSuffixMatches") == 1 for x in ev["namingSites"])
    return ev

rows = []
for stratum, mech in (("scripts", MECH), ("http", HTTP_MECH), ("wiring", WIRING_MECH)):
    recs = [json.loads(l) for l in (REPO / f"cpg/eval/dev/ac5-{stratum}.jsonl").read_text().splitlines() if l.strip()]
    arows = asks["strata"][stratum]["rows"]
    for i, (r, ar) in enumerate(zip(recs, arows)):
        assert ar["itemId"] == r["itemId"]
        row = {"stratum": stratum, "idx": i, "itemId": r["itemId"], "anchorType": r["anchor"]["type"],
               "anchor": r["anchor"].get("qname") or r["anchor"].get("keyNorm") or r["anchor"].get("rootPath"),
               "impacted": r["impacted"].get("qname") or (r["impacted"]["path"] + "#" + r["impacted"].get("keyPath", "")), "impactedLang": None,
               "hit": ar["hit"], "cls": ar["cls"], "status": ar.get("status"), "mechanism": mech.get(i, "already built (hit)")}
        if stratum == "scripts":
            row.update(evidence(r))
            s = by_q.get(r["impacted"]["qname"]); row["impactedLang"] = s["lang"] if s else "shell(symbol absent: parse-error)"
        rows.append(row)
(OUT / "classification.json").write_text(json.dumps(rows, indent=1, default=str))
sc = [r for r in rows if r["stratum"] == "scripts"]
print("scripts by impacted lang:", Counter(r["impactedLang"] for r in sc))
def has(r, near=None):
    return any(s["where"] == "calleeName" and (near is None or s["near"] == near) for s in r.get("namingSites", []))
for lang in ("markdown", "shell"):
    g = [r for r in sc if r["impactedLang"] == lang]
    print(lang, len(g), "items;",
          "callee names anchor (anywhere in impacted symbol):", sum(has(r) for r in g), "; near gold line:", sum(has(r, True) for r in g),
          "; an arg valueSet carries anchor basename:", sum(any(s["where"].startswith("valueSet") for s in r["namingSites"]) for r in g),
          "; near:", sum(any(s["where"].startswith("valueSet") and s["near"] for s in r["namingSites"]) for r in g))
print("gold window mentions anchor:", Counter((r["impactedLang"], r["goldWindowMentionsAnchor"]) for r in sc))

print("tier1 (exact path via caller/ancestor dir) resolvable:", Counter((r["impactedLang"], r.get("tier1Resolvable")) for r in sc))
print("tier2 (>=2-seg tail, unique snapshot suffix) resolvable:", Counter((r["impactedLang"], r.get("tier2Resolvable")) for r in sc))
print("a naming site exists:", Counter((r["impactedLang"], bool(r.get("namingSites"))) for r in sc))
