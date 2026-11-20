"""K-097 ABA probe — throwaway graph `ws:probe-k097-review` only (analyst re-run copy of the graph-dba harness).

Part A drives the REAL shipped code (Services.submit_workflow_input ->
WorkflowExecutor.resume -> Repository.resume_run_with_ctx) and injects a second
submitter between submitter B's service-side read and B's CAS, using the same
proxy-repo technique `tests/test_workflow_timers.py::_RacingRepo` uses.

Part B runs the PROPOSED versioned CAS Cypher raw against the same scratch graph
with the same interleaving, plus the legacy (no ctxVersion) default and PROFILEs.

Run:  cd falkor-chat/server && .venv/bin/python <this file>
The graph is created here and must be GRAPH.DELETEd afterwards (done by --cleanup).
"""
from __future__ import annotations

import itertools
import json
import sys

from falkordb import FalkorDB

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.repository import Repository
from falkorchat.services import Services, WorkflowRunNotWaitingError

WS = "probe-k097-review"
KEY = f"ws:{WS}"
CTX = CallContext(ws=WS, actor="u1")

conn = FalkorDB(host="127.0.0.1", port=6379)
g = conn.select_graph(KEY)
existing = conn.list_graphs()
if "--cleanup" in sys.argv:
    assert KEY in existing
    g.delete()
    print("deleted", KEY)
    sys.exit(0)
assert KEY not in existing, f"{KEY} already exists — refusing to reuse"

# --- schema: the bootstrap_schema.sh subset these queries touch (indexes, then the
#     runId constraint — index-before-constraint) ------------------------------
for ddl in [
    "CREATE INDEX FOR (n:Step) ON (n.key)",
    "CREATE INDEX FOR (n:Step) ON (n.stepUid)",
    "CREATE INDEX FOR (n:WorkflowDefSnapshot) ON (n.key)",
    "CREATE INDEX FOR (n:WorkflowDefSnapshot) ON (n.version)",
    "CREATE INDEX FOR (n:WorkflowRun) ON (n.runId)",
    "CREATE INDEX FOR (n:WorkflowRun) ON (n.status)",
    "CREATE INDEX FOR (n:WorkflowRun) ON (n.startedAt)",
    "CREATE INDEX FOR (n:StepRun) ON (n.stepRunId)",
    "CREATE INDEX FOR (n:StepRun) ON (n.status)",
]:
    g.query(ddl)
conn.connection.execute_command(
    "GRAPH.CONSTRAINT", "CREATE", KEY, "UNIQUE", "NODE", "WorkflowRun",
    "PROPERTIES", "1", "runId",
)

repo = Repository(conn)
ids = (f"id{n}" for n in itertools.count(1))
services = Services(repo, clock=lambda: 1000, id_gen=lambda: next(ids))
sr_ids = (f"sr{n}" for n in itertools.count(1))
sr_clock = itertools.count(2000)
executor = WorkflowExecutor(
    services, repo, llm=None, guard_judge=None,
    id_gen=lambda: next(sr_ids), clock=lambda: next(sr_clock),
)
services.set_executor(executor)


def j(o):
    return json.dumps(o, separators=(",", ":"))


GUARD_APPROVED = j({"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve"})
GUARD_TICKET = j({"kind": "cmp", "path": "ctx.ticket", "op": "exists"})
GUARD_DONE = j({"kind": "cmp", "path": "ctx.finish", "op": "truthy"})

# Def 1: approval (human) --approve--> provision (human, parks) --ticket--> granted
repo.materialize_snapshot(
    WS, key="aba-two-step", version="1", name="ABA two-step", kind="process",
    start_key="approval",
    steps=[
        {"key": "approval", "type": "human", "config": j({
            "waitsForHuman": True, "fields": ["decision"],
            "expects": {"decision": ["approve", "reject"]}})},
        {"key": "provision", "type": "human", "config": j({
            "waitsForHuman": True, "fields": ["ticket"]})},
        {"key": "granted", "type": "decision", "config": "{}"},
    ],
    transitions=[
        {"from": "approval", "to": "provision", "on": "approved",
         "guard": GUARD_APPROVED, "order": 0},
        {"from": "provision", "to": "granted", "on": "ticketed",
         "guard": GUARD_TICKET, "order": 0},
    ],
)
# Def 2: one bare human step that re-parks on itself until ctx.finish
repo.materialize_snapshot(
    WS, key="aba-same-step", version="1", name="ABA same step", kind="process",
    start_key="note",
    steps=[
        {"key": "note", "type": "human", "config": j({"waitsForHuman": True})},
        {"key": "end", "type": "decision", "config": "{}"},
    ],
    transitions=[{"from": "note", "to": "end", "on": "finish",
                  "guard": GUARD_DONE, "order": 0}],
)
# Def 3: R-1 baseline — approval --approve--> granted (terminal, no re-park)
repo.materialize_snapshot(
    WS, key="r1-baseline", version="1", name="R1", kind="process",
    start_key="approval",
    steps=[
        {"key": "approval", "type": "human", "config": j({
            "waitsForHuman": True, "fields": ["decision"],
            "expects": {"decision": ["approve", "reject"]}})},
        {"key": "granted", "type": "decision", "config": "{}"},
    ],
    transitions=[{"from": "approval", "to": "granted", "on": "approved",
                  "guard": GUARD_APPROVED, "order": 0}],
)


class RacingRepo:
    """Proxy for the executor's repo: the FIRST resume_run_with_ctx call (submitter
    B's CAS) first runs `on_cas` (submitter A, end to end), then proceeds."""

    def __init__(self, inner, on_cas):
        self._inner, self._on_cas, self._fired = inner, on_cas, False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def resume_run_with_ctx(self, ws, *, run_id, ctx):
        if not self._fired:
            self._fired = True
            print("   [inject] B has read+merged; B's CAS is about to land.")
            print("   [inject] B's merged ctx =", ctx)
            self._on_cas()
            r = self._inner.get_run(ws, run_id=run_id)
            print(f"   [inject] A done: status={r['status']} atStep={r['atStepKey']} "
                  f"stepCount={r['stepCount']} ctx={r['ctx']}")
        return self._inner.resume_run_with_ctx(ws, run_id=run_id, ctx=ctx)


def trail(run_id):
    return [(s["stepKey"]) for s in repo.read_step_runs(WS, run_id=run_id)]


def scenario(title, def_key, a_input, b_input):
    print(f"\n=== {title} ===")
    started = services.start_workflow_run(CTX, def_key=def_key, version="1", run_ctx={})
    run_id = started["runId"]
    r0 = repo.get_run(WS, run_id=run_id)
    print(f" start: status={r0['status']} atStep={r0['atStepKey']} ctx={r0['ctx']}")
    a_result = {}

    def submitter_a():
        a_result["out"] = services.submit_workflow_input(CTX, run_id=run_id, input=a_input)
        print("   [A] response:", a_result["out"])

    executor._repo = RacingRepo(repo, submitter_a)  # noqa: SLF001 — injection seam
    try:
        out = services.submit_workflow_input(CTX, run_id=run_id, input=b_input)
        print(" [B] response: 200", out)
    except WorkflowRunNotWaitingError as exc:
        print(" [B] response: 409", exc)
    finally:
        executor._repo = repo  # noqa: SLF001
    r = repo.get_run(WS, run_id=run_id)
    print(f" final: status={r['status']} atStep={r['atStepKey']} stepCount={r['stepCount']}"
          f" ctx={r['ctx']}")
    print(" StepRun trail:", trail(run_id))
    return run_id, r


print("PART A — shipped status-only CAS (real Services/Executor/Repository)")
scenario("A1 R-1 baseline: A drives to terminal before B's CAS",
         "r1-baseline", {"decision": "approve"}, {"decision": "reject"})
rid, r = scenario("A2 ABA, re-park at a DIFFERENT step",
                  "aba-two-step", {"decision": "approve"}, {"decision": "reject"})
ctx = json.loads(r["ctx"])
print(" ABA-A2 verdict:", "REPRODUCED" if r["status"] == "waiting"
      and ctx.get("decision") == "reject" and "approval" in trail(rid) else "refuted",
      "- ctx.decision =", ctx.get("decision"),
      "but the trail advanced approval->provision on decision == 'approve'")
rid, r = scenario("A3 ABA, re-park at the SAME step",
                  "aba-same-step", {"comment": "from-A"}, {"other": "from-B"})
ctx = json.loads(r["ctx"])
print(" ABA-A3 verdict:", "REPRODUCED" if "comment" not in ctx else "refuted",
      "- A's accepted (200) key 'comment' present in final ctx:", "comment" in ctx)

# ---------------------------------------------------------------------------
print("\nPART B — proposed versioned CAS (raw Cypher on the same scratch graph)")
READ = ("MATCH (r:WorkflowRun {runId: $runId}) "
        "RETURN r.status AS status, r.ctx AS ctx, "
        "coalesce(r.ctxVersion, 0) AS ctxVersion")
CAS = ("MATCH (r:WorkflowRun {runId: $runId}) "
       "WHERE r.status = 'waiting' AND coalesce(r.ctxVersion, 0) = $ctxVersion "
       "SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx, "
       "    r.ctxVersion = $ctxVersion + 1 "
       "RETURN r.runId AS runId, r.status AS status, r.ctxVersion AS ctxVersion")
PLAIN_RESUME = ("MATCH (r:WorkflowRun {runId: $runId}) WHERE r.status = 'waiting' "
                "SET r.status = 'running', r.waitingThreadId = '', "
                "    r.ctxVersion = coalesce(r.ctxVersion, 0) + 1 "
                "RETURN r.runId AS runId, r.status AS status, r.ctxVersion AS ctxVersion")
SUSPEND = ("MATCH (r:WorkflowRun {runId: $runId}) WHERE r.status = 'running' "
           "SET r.status = 'waiting', r.waitingThreadId = $threadId "
           "RETURN r.runId AS runId, r.status AS status")


def q(cy, **p):
    return g.query(cy, p)


def ro(cy, **p):
    return g.ro_query(cy, p).result_set


# Legacy run: created WITHOUT ctxVersion (what every pre-K-097 run looks like).
q("CREATE (:WorkflowRun {runId:'legacy1', status:'waiting', ctx:'{\"seed\":1}', "
  "waitingThreadId:'', stepCount:0, maxSteps:12, startedAt:1})")
print(" legacy keys:", ro("MATCH (r:WorkflowRun {runId:'legacy1'}) RETURN keys(r)")[0][0])
st, c0, v0 = ro(READ, runId="legacy1")[0]
print(f" B1: B reads status={st} ctx={c0} ctxVersion={v0} (coalesce default)")
# A: reads the same base, wins with its own merge, its drive re-parks the run.
res = q(CAS, runId="legacy1", ctx='{"seed":1,"decision":"approve"}', ctxVersion=v0)
print("     A CAS rows:", res.result_set)
print("     A drive re-parks:", q(SUSPEND, runId="legacy1", threadId="").result_set)
print("     state before B's CAS:", ro(READ, runId="legacy1"))
res = q(CAS, runId="legacy1", ctx='{"seed":1,"decision":"reject"}', ctxVersion=v0)
print(f"     B CAS (stale v={v0}) rows: {res.result_set} "
      f"properties_set={res.properties_set}")
after = ro(READ, runId="legacy1")[0]
print("     state after B's CAS:", after)
print(" B1 verdict:", "REFUSED, nothing written" if not res.result_set
      and after[0] == "waiting" and "approve" in after[1] else "NOT refused")

# A chat-path plain resume between B's read and B's CAS must also refuse B.
q("CREATE (:WorkflowRun {runId:'v2', status:'waiting', ctx:'{}', ctxVersion:0, "
  "waitingThreadId:'th', stepCount:0, maxSteps:12, startedAt:2})")
st, c0, v0 = ro(READ, runId="v2")[0]
print(" B2: plain (chat) resume:", q(PLAIN_RESUME, runId="v2").result_set,
      "then re-park:", q(SUSPEND, runId="v2", threadId="th").result_set)
res = q(CAS, runId="v2", ctx='{"x":1}', ctxVersion=v0)
print(f"     B CAS (stale v={v0}) rows: {res.result_set}; state:", ro(READ, runId="v2"))
# Fresh-version CAS succeeds (no false refusal).
st, c0, v1 = ro(READ, runId="v2")[0]
res = q(CAS, runId="v2", ctx='{"x":1}', ctxVersion=v1)
print(f" B3: fresh CAS (v={v1}) rows: {res.result_set}")

# Plans: pad the label so the index choice is meaningful.
q("UNWIND range(1, 500) AS i CREATE (:WorkflowRun {runId:'pad'+toString(i), "
  "status: CASE WHEN i % 2 = 0 THEN 'waiting' ELSE 'done' END, ctx:'{}', "
  "ctxVersion: i % 3, startedAt: i})")
q("CREATE (:WorkflowRun {runId:'prof1', status:'waiting', ctx:'{}', ctxVersion:4, startedAt:9})")
for label, v in [("winner (v matches)", 4), ("loser (stale v)", 3)]:
    plan = conn.connection.execute_command(
        "GRAPH.PROFILE", KEY,
        f"CYPHER runId='prof1' ctxVersion={v} ctx='{{}}' " + CAS)
    print(f"\n PROFILE CAS — {label}:")
    for line in plan:
        print("   ", line.decode() if isinstance(line, bytes) else line)
    q("MATCH (r:WorkflowRun {runId:'prof1'}) SET r.status='waiting', r.ctxVersion=4")
plan = conn.connection.execute_command(
    "GRAPH.PROFILE", KEY, "CYPHER runId='prof1' " + READ)
print("\n PROFILE read (get_run's added column):")
for line in plan:
    print("   ", line.decode() if isinstance(line, bytes) else line)
