"""CO-1 wiring probe (plan 3.4): for each of the 11 dev wiring items, what the store holds for the config element (ConfigEntry exists?
a Link BOUND_BY it? launchFact/bound?) and what the answer for the anchor lists. Dev gold only; read-only raw queries as ccg_raw. Output: wiring-probe.json"""
import json, os, sys
from pathlib import Path
REPO, OUT, PORT = Path(os.environ["P1T_REPO"]), Path(os.environ["P1T_OUT"]), int(os.environ["P1T_PORT"])
assert PORT >= 16500 and PORT not in (6379, 6390, 16390, 16400)
os.environ["CCG_SYSTEM"] = "graphmind-ai-lab"
sys.path[:0] = [str(REPO / "cpg")]
from clients.common import Client
c = Client("graphmind-ai-lab", host="127.0.0.1", port=PORT)
q = lambda s: c.raw_query(s)["rows"]
out = {"global": {}, "items": []}
for name, s in {"ConfigEntry": "MATCH (n:ConfigEntry) RETURN count(n)", "Link": "MATCH (n:Link) RETURN count(n)",
                "BOUND_BY": "MATCH ()-[r:BOUND_BY]->() RETURN count(r)", "TO": "MATCH ()-[r:TO]->() RETURN count(r)",
                "Link_BOUND_BY_ConfigEntry": "MATCH (l:Link)-[:BOUND_BY]->(:ConfigEntry) RETURN count(l)",
                "ConfigEntry_in_repo_config_paths": "MATCH (n:ConfigEntry) WHERE n.path IN ['falkor-chat/compose.yaml','kiro/.kiro/agents/falkor-chat-demo.json','mcp-monitor/config.example.toml','.mcp.json','opencode/agents/tank/opencode.json'] RETURN count(n)"}.items():
    try: out["global"][name] = q(s)[0][0]
    except Exception as e: out["global"][name] = "error:" + type(e).__name__ + ":" + str(e)[:100]
recs = [json.loads(l) for l in (REPO / "cpg/eval/dev/ac5-wiring.jsonl").read_text().splitlines() if l.strip()]
for r in recs:
    im = r["impacted"]
    row = {"itemId": r["itemId"], "anchorType": r["anchor"]["type"], "path": im["path"], "keyPath": im["keyPath"]}
    try:
        row["configEntryInStore"] = q("MATCH (n:ConfigEntry {repoId:%s, path:%s, keyPath:%s}) RETURN count(n)" % tuple(json.dumps(x) for x in (im["repoId"], im["path"], im["keyPath"])))[0][0]
        row["linkBoundByIt"] = q("MATCH (l:Link)-[:BOUND_BY]->(n:ConfigEntry {repoId:%s, path:%s, keyPath:%s}) RETURN count(l)" % tuple(json.dumps(x) for x in (im["repoId"], im["path"], im["keyPath"])))[0][0]
        pre = "n.repoId=%s AND n.path=%s AND (n.keyPath=%s OR n.keyPath STARTS WITH %s OR n.keyPath STARTS WITH %s)" % tuple(json.dumps(x) for x in (im["repoId"], im["path"], im["keyPath"], im["keyPath"] + ".", im["keyPath"] + "["))
        row["configEntriesUnderKeyPath"] = q("MATCH (n:ConfigEntry) WHERE %s RETURN count(n)" % pre)[0][0]
        row["linkBoundByUnder"] = q("MATCH (l:Link)-[:BOUND_BY]->(n:ConfigEntry) WHERE %s RETURN count(DISTINCT l)" % pre)[0][0]
    except Exception as e:
        row["error"] = type(e).__name__ + ":" + str(e)[:120]
    out["items"].append(row)
(OUT / "wiring-probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["global"])); print([(x["keyPath"], x.get("configEntryInStore"), x.get("linkBoundByIt"), x.get("configEntriesUnderKeyPath"), x.get("linkBoundByUnder"), x.get("error")) for x in out["items"]])
