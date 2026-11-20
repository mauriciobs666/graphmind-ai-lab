"""analyst follow-up probe for K-097, throwaway graph ws:probe-k097-review2 only.
Checks the hand-appended claim in graph-dba's output (not produced by the harness):
the versioned CAS on a NON-waiting run with a MATCHING version yields 0 at the index scan,
and writes nothing. Deletes the graph at the end."""
from falkordb import FalkorDB
KEY = "ws:probe-k097-review2"
conn = FalkorDB(host="127.0.0.1", port=6379)
assert KEY not in conn.list_graphs()
g = conn.select_graph(KEY)
try:
    g.query("CREATE INDEX FOR (n:WorkflowRun) ON (n.runId)")
    g.query("CREATE INDEX FOR (n:WorkflowRun) ON (n.status)")
    g.query("UNWIND range(1,500) AS i CREATE (:WorkflowRun {runId:'pad'+toString(i), status: CASE WHEN i % 2 = 0 THEN 'waiting' ELSE 'done' END, ctx:'{}', ctxVersion: i % 3})")
    g.query("CREATE (:WorkflowRun {runId:'p1', status:'running', ctx:'{\"keep\":1}', ctxVersion:4})")
    CAS = ("MATCH (r:WorkflowRun {runId: $runId}) "
           "WHERE r.status = 'waiting' AND coalesce(r.ctxVersion, 0) = $ctxVersion "
           "SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx, r.ctxVersion = $ctxVersion + 1 "
           "RETURN r.runId AS runId, r.status AS status, r.ctxVersion AS ctxVersion")
    plan = conn.connection.execute_command("GRAPH.PROFILE", KEY, "CYPHER runId='p1' ctxVersion=4 ctx='{\"LOSER\":1}' " + CAS)
    print("PROFILE CAS, run 'running', version MATCHES:")
    for line in plan:
        print("   ", line.decode() if isinstance(line, bytes) else line)
    print("after:", g.ro_query("MATCH (r:WorkflowRun {runId:'p1'}) RETURN r.status, r.ctx, r.ctxVersion").result_set)
finally:
    g.delete()
    print("deleted", KEY, "still listed:", KEY in conn.list_graphs())
