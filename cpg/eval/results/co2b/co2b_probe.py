import sys, pathlib, tempfile, subprocess, re, posixpath
sys.path.insert(0,'.')
import script_resolver_helpers as H
from ccg.assemble import rows as R
root=pathlib.Path("/home/mauricio/prg/graphmind-ai-lab")
files={}
for f in subprocess.run(["git","-C",str(root),"ls-files","*.sh"],capture_output=True,text=True).stdout.split():
    try: files[f]=(root/f).read_text()
    except Exception: pass
w=H.assemble(pathlib.Path(tempfile.mkdtemp()),files,langs=("shell",))
b=w.bundles[0]
vs={(r["callSiteId"],r["arg"]):r["values"] for r in b if r["type"]=="valueSet"}
sym={r["id"]:r for r in b if r["type"]=="symbol"}
scripts={r["path"] for r in b if r["type"]=="symbol" and r["kind"]=="script"}
n=0
for r in b:
    if r["type"]!="callsite" or r["targets"]: continue
    named=R._site_path(r,"shell",vs)
    if not named or not named[0]: continue
    tail=R._segments(named[1])
    if len(tail)!=1: continue
    caller=sym[r["symbolId"]]["path"]; d=caller.rpartition("/")[0]
    sib=(d+"/" if d else "")+tail[0]
    # head variable text
    src=files[caller].splitlines()[r["line"]-1].strip()
    m=re.search(r'\$\{?(\w+)',src); var=m.group(1) if m else None
    asg=[l.strip() for l in files[caller].splitlines() if var and re.match(r'\s*(local\s+|readonly\s+)?'+var+'=',l)]
    print(("SIB " if sib in scripts else "no  "), caller.split("/")[-1], r["line"], "|", src[:70], "|", (asg[:1] or ["?"])[0][:80])
