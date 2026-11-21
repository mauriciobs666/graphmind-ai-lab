"""M4 equivalence probe (ws:test only, wiped before and after): with `e` unreferenced
after the pattern, FalkorDB folds (seed)->(e)<-(related) into one traversal whose
duplicate paths collapse, so `WITH related LIMIT 5` already yields distinct rows.

WIPES the target graph before and after. Refuses without an explicit throwaway target:
    python m4_distinct_probe.py <port>        (graph ws:u12-m4-probe on 127.0.0.1:<port>)
Never point it at a suite's or a live instance's graph."""
import sys
from falkordb import FalkorDB
if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    sys.exit("usage: m4_distinct_probe.py <throwaway-port>  (refuses to default to any host/graph)")
g = FalkorDB(host="127.0.0.1", port=int(sys.argv[1])).select_graph("ws:u12-m4-probe")
g.query("MATCH (n) DETACH DELETE n")
g.query("CREATE (s:Message {msgId:'s'}),(r:Message {msgId:'r'}),(e1:Entity {entityId:'e1'}),(e2:Entity {entityId:'e2'}),"
        "(s)-[:MENTIONS]->(e1),(s)-[:MENTIONS]->(e2),(r)-[:MENTIONS]->(e1),(r)-[:MENTIONS]->(e2)")
for w in ["WITH DISTINCT related LIMIT 5", "WITH related LIMIT 5"]:
    q = ("MATCH (seed:Message {msgId:'s'}) CALL { WITH seed OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
         + w + " RETURN collect(related.msgId) AS ex } RETURN ex")
    print(f"{w!r:35} -> {g.ro_query(q).result_set}")
    print("   plan:", " / ".join(l.strip() for l in g.explain(q).plan))
print("raw rows with e projected:", g.ro_query("MATCH (seed:Message {msgId:'s'}) OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) RETURN related.msgId, e.entityId").result_set)
g.query("MATCH (n) DETACH DELETE n")
