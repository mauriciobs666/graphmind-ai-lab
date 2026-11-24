import sys, pathlib, tempfile, subprocess, json
sys.path.insert(0,'.')
import script_resolver_helpers as H
root=pathlib.Path("/home/mauricio/prg/graphmind-ai-lab")
files={}
for f in subprocess.run(["git","-C",str(root),"ls-files","*.sh","*.md"],capture_output=True,text=True).stdout.split():
    try: files[f]=(root/f).read_text()
    except Exception: pass
w=H.assemble(pathlib.Path(tempfile.mkdtemp()),files)
print("files",len(files),"scriptCALLS",sum(1 for e in H.calls_edges(w) if e["reason"]!="resolved-call"),"dropped",w.rows.dropped)
want=[("G","skills/joern-cpg/scripts/export-cpg.sh","skills/joern-cpg/scripts/pipeline.sh"),
("G","claude/scripts/guard-destructive-ops.sh","claude/scripts/guard-broad-bash.sh"),
("G","mcp-monitor/run.sh","mcp-monitor/scripts/demo_falkor_chat.sh"),
("H","skills/joern-cpg/scripts/git-provenance.sh","skills/joern-cpg/scripts/pipeline.sh"),
("H","cypher-mcp/server.py","cypher-mcp/run.sh")]
for fam,a,imp in want:
    try: d=H.ask_impact(w,a)
    except Exception as e: print(fam,a,"anchor:",type(e).__name__); continue
    got=[i for i in d["items"] if i["subject"]["path"]==imp]
    print(fam,a.split("/")[-1],"<-",imp.split("/")[-1], "NOT LISTED" if not got else [(i["distance"],i["status"],[ (h["kind"],h.get("edgeType"),h.get("path","").split("/")[-1],h.get("line")) for h in i["chain"]][:2]) for i in got])
