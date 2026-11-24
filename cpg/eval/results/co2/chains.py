import json, os, sys
sys.argv=["x"]
import importlib.util
spec=importlib.util.spec_from_file_location("asks","/home/mauricio/prg/graphmind-ai-lab/cpg/eval/results/co2/asks.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
want=[("guard-destructive-ops.sh::<script>","guard-broad-bash.sh::<script>"),("git-provenance.sh::<script>","pipeline.sh::<script>"),("git-provenance.sh::<script>","test-stamp-wiring.sh::<script>"),("pipeline.sh::<script>","test-stamp-wiring.sh::<script>")]
for r in m.FILES["scripts"]:
    a=r["anchor"].get("qname","")
    imp=r["impacted"].get("qname","")
    for wa,wi in want:
        if a.endswith(wa) and imp.endswith(wi):
            res=m.ask_anchor(r["anchor"])
            for x in m.members(res["doc"]):
                if m.matches(x,r):
                    print(a.split("/")[-1],"<-",imp.split("/")[-1],x["status"],x["distance"])
                    for h in x["chain"]:
                        print("   ",h["kind"],h.get("edgeType"),h.get("dir"),h.get("status"),h.get("reason"),h.get("path"),h.get("line"),h.get("calleeName") or h.get("qname"))
                    break
