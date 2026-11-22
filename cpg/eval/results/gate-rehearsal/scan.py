"""Leak scan: counts occurrences of dev-gold strings (32-hex ids, paths, qnames, keyNorms) in the given files. Prints counts only, never the matched strings.
usage: scan.py <file>...   (gold terms come from cpg/eval/dev/ac5-*.jsonl)"""
import glob, json, re, sys, os
terms = set()
def walk(v):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in ("why", "provider", "originals", "writers"): walk(x) if k != "why" else None
            else: walk(x)
    elif isinstance(v, list):
        for x in v: walk(x)
    elif isinstance(v, str):
        if re.fullmatch(r"[0-9a-f]{32}", v) or ("/" in v and len(v) >= 8) or ("::" in v) or (len(v) >= 10 and re.search(r"[A-Z]+ /", v)):
            terms.add(v)
        if re.search(r":\d+$", v) and "/" in v: terms.add(v.rsplit(":", 1)[0])
for f in glob.glob(os.path.join(os.environ.get("REAL", "/home/mauricio/prg/graphmind-ai-lab"), "cpg/eval/dev/ac5-*.jsonl")):
    for l in open(f):
        walk(json.loads(l))
terms = {t for t in terms if len(t) >= 8}
tot = 0
for p in sys.argv[1:]:
    try: txt = open(p, errors="replace").read()
    except OSError: print(f"{os.path.basename(p)}: unreadable"); continue
    hits = sum(1 for t in terms if t in txt)
    tot += hits
    print(f"{os.path.basename(p)}: bytes={len(txt)} gold-terms-found={hits}")
print(f"terms-checked={len(terms)} total-hits={tot}")
