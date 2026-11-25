"""CO-3 dev measurement, offline (the pattern of `eval/results/co2b/co2b_dev.py`): the real shell and Markdown extractors, the real
`shell-http-client` pack runtime, linker, assembler and answer builder over every tracked `*.sh` and `*.md` file of the working tree.
No store, no build. The Python provider of `GET /health` (`falkor-chat/server/falkorchat/api.py:58`) needs Joern, so the provider is a
hand-made fact anchored at one script; the dev gold's other 17 http items (Python and JS consumers) are not re-measured here.
Run from `cpg/`: `.venv/bin/python eval/results/co3/co3_dev.py`. Aggregate output only (S1 record `eval/decisions/ccg-noise-s1-v1.json`)."""
import collections, json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, "tests")
import co3_helpers as H
from script_resolver_helpers import script_symbol

root = pathlib.Path(__file__).resolve().parents[4]
files = {}
for f in subprocess.run(["git", "-C", str(root), "ls-files", "*.sh", "*.md"], capture_output=True, text=True).stdout.split():
    try:
        files[f] = (root / f).read_text()
    except Exception:
        pass
PROVIDER = "falkor-chat/scripts/test_queries.sh"
tmp = pathlib.Path(tempfile.mkdtemp())
w = H.assemble_http(tmp, files, PROVIDER)
doc = H.ask_contract(w)
got = {(i["subject"]["path"], i["status"]) for i in doc["items"]}
want = [("claude/teco/coordination-techniques.md", "http-md-inline-span"), ("falkor-chat/scripts/load_test.sh", "L"),
        ("falkor-chat/scripts/seed_nlq_eval_corpus.sh", "L"), ("claude/teco/hooks/ensure-environment.sh", "L")]
out = {"files": len(files), "items": len(doc["items"]), "truncated": doc.get("truncated"),
       "devTargets": [{"path": p, "family": fam, "hit": any(g[0] == p for g in got), "status": next((s for q, s in got if q == p), None)} for p, fam in want]}
cons = collections.Counter((c["ruleId"], c["status"], c["reason"]) for c in w.facts.consumers if c["packId"] == "shell-http-client")
out["consumersByRuleStatusReason"] = {"/".join(k): v for k, v in sorted(cons.items())}
out["consumersByKeyShape"] = dict(collections.Counter(("GET /health" if c["key"] == "GET /health" else "other") for c in w.facts.consumers if c["packId"] == "shell-http-client"))
print(json.dumps(out, indent=1))
(pathlib.Path(__file__).with_name("summary.json")).write_text(json.dumps(out, indent=1) + "\n")
