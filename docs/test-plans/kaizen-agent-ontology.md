# Kaizen agent/learning-note ontology — Test Plan (M8, S6)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — (M8)

## 1. Scope & objective

Step **S6** of `docs/plans/kaizen-agent-ontology.md` (Version 3, approved after 3 `analyst` review
passes, `docs/reviews/kaizen-agent-ontology.md`). A live, black-box confirmation — against the
real, deployed `cypher-mcp` container (image `cypher-mcp:cb712173ab57`) and the real, shared
`kaizen_team` FalkorDB graph — of two things a static review cannot itself confirm:

- **(a)** That `authorize_write()`'s §3.1a closure (the widened
  `_FOREIGN_TRIGGER_RE = MERGE|DELETE|SET|REMOVE` check) actually rejects the three adversarial
  cross-clause "decoy `CREATE`" attacks the review traced (Attacks A, B, C — including C's `SET`
  and `REMOVE` sub-variants), end to end, on the running system.
- **(b)** That one real distillation dry-run (producer-write → MENTIONS-write → count-and-decide
  read → partial resolve ×2 → full node removal) behaves as `graph-dba`'s design
  (`docs/plans/kaizen-agent-ontology-graph.md` §2-§4) specifies, using only disposable identities,
  leaving the graph clean afterward.

This is **not** a full acceptance pass over the whole M8 feature (S1's own scripted live
acceptance sub-step already covers the 4 new shapes' happy paths against a scratch graph and a
one-time real-graph check; `tdd-engineer`'s offline suite covers exhaustive unit-level regex/parse
boundaries). This plan is deliberately narrow, matching the plan's own scoping of S6.

**Out of scope**, explicitly: re-deriving CPG freshness (none loaded for `cypher-mcp`/`claude`);
re-testing shapes 1-2's pre-existing non-adversarial behavior (S1 already regression-tests this);
load/performance; the 13 retargeted agent prompts' prose (S3, `cobb`-owned, doc-only); `cobb`'s
`SKILL.md` §5 prose itself (S4, doc-only) — only the underlying Cypher shapes it prescribes are
exercised here.

## 2. References

- `docs/plans/kaizen-agent-ontology.md` (Version 3) — §4 S6 row, §6 Risks.
- `docs/reviews/kaizen-agent-ontology.md` — all 3 passes, Finding 1 (Attacks A/B), Pass 2's new
  finding (Attack C, `SET`/`REMOVE`), Pass 3's verification.
- `docs/plans/kaizen-agent-ontology-graph.md` — §2 producer-write, §3 MENTIONS-write, §4
  deletion/resolve shapes, §4.1 count-and-decide read.
- `cypher-mcp/server.py` (shipped, commit `e01045b`) — not re-read line-by-line here; this plan
  trusts the review's grounding and tests the deployed behavior instead.

**CPG:** considered, not relevant — this is a live black-box QA pass against a running MCP
container and FalkorDB graph, not a static code-reading task; `GRAPH.LIST` shows no CPG loaded for
`cypher-mcp`/`claude` (confirmed by the plan and both review passes, re-confirmed not needed here
since no file is being read for structural analysis).

## 3. Risk assessment

Highest risk, per the plan's own §6: a regression between the twice-reviewed design and the
actually-deployed, rebuilt container — i.e., the fix described in the plan text does not match
what `cypher-mcp:cb712173ab57` actually executes. Both halves of this plan target exactly that
seam. Secondary risk: the distillation read/write sequence (count-before-tag ordering, partial vs.
full delete) behaving differently on live data than on `graph-dba`'s paper design.

## 4. Test items

All writes below use `mcp__cypher__query(graph='kaizen_team', cypher=..., agent=...)` directly —
the real call path a real agent uses. Disposable identity suffix for this run: `4e24af1e`.

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-001 | Seed one disposable victim entry via a legitimate producer-write | High (setup) | functional |
| TP-002 | Attack A — decoy `CREATE` chained with `MATCH ... DETACH DELETE victim` → rejected | Blocker-risk | contract/security |
| TP-003 | Attack B — decoy `CREATE` chained with forged `MERGE (:Agent)-[:PRODUCED]->(...)` → rejected | Blocker-risk | contract/security |
| TP-004 | Attack C (SET) — decoy `CREATE` chained with `MATCH ... SET victim.author=..., victim.fact=...` → rejected | Blocker-risk | contract/security |
| TP-005 | Attack C (REMOVE variant) — decoy `CREATE` chained with `MATCH ... REMOVE victim.author` → rejected | Blocker-risk | contract/security |
| TP-006 | Confirm victim entry unmodified after TP-002…005 | High | functional |
| TP-007 | Real producer-write (disposable agent) → entry created, `PRODUCED` edge present | High | functional/e2e |
| TP-008 | Curator MENTIONS-write (`agent='cobb'`) tags the entry with a disposable mentioned agent | High | functional/e2e |
| TP-009 | Count-and-decide read reports 2 remaining edges (1 `PRODUCED`, 1 `MENTIONS`) | High | functional |
| TP-010 | Resolve `PRODUCED` edge only → node still exists, `MENTIONS` edge remains | High | functional/e2e |
| TP-011 | Resolve `MENTIONS` edge → node now fully gone | High | functional/e2e |
| TP-012 | Both disposable `:Agent` nodes remain (by design, not a leak) | Medium | functional |
| TP-013 | Cleanup — no leftover disposable `:KaizenEntry` nodes after the run | High | housekeeping |

### Preconditions
- `kaizen_team` graph reachable via `mcp__cypher__query`; `Agent.agentId` and `KaizenEntry.entryId`
  uniqueness constraints already `OPERATIONAL` (confirmed live before this plan: `CALL
  db.constraints()` shows both as `UNIQUE`/`OPERATIONAL`).
- Current state confirmed before starting: `MATCH (n) RETURN labels(n), count(n)` → only
  `['KaizenEntry']`, 26 nodes, zero `:Agent` nodes.

### Steps & expected results

**TP-001**: `MERGE (a:Agent {agentId:'_qa_selftest_producer_4e24af1e'}) CREATE
(a)-[:PRODUCED {sessionId:'qa-dryrun'}]->(k:KaizenEntry {entryId:'qa-selftest-victim-4e24af1e',
date:'2026-08-22', fact:'disposable QA victim entry', evidence:'qa-engineer S6 dry-run',
context:'kaizen-agent-ontology S6', suggestedHome:'unsure', createdAt:'2026-08-22T00:00:00Z'})`,
`agent='_qa_selftest_producer_4e24af1e'`. Expected: authorized, one node + one edge created.

**TP-002/003/004/005**: each attack statement declares `agent='_qa_selftest_attacker_4e24af1e'`
(not curator, not the victim's producer), targets `entryId:'qa-selftest-victim-4e24af1e'` as the
victim, and uses a distinct decoy `entryId` (`qa-selftest-decoy-4e24af1e-a/b/c/d`) so any
unexpected success is individually attributable. Expected for all four: the tool's response is a
rejection (no write executed) — verified both by the response text and by re-reading the victim
entry/graph state afterward (TP-006).

**TP-006**: re-read `qa-selftest-victim-4e24af1e` after TP-002-005: entry still exists, `fact`/
`author`-equivalent state unchanged, no forged `:Agent {agentId:'_qa_selftest_forged_4e24af1e'}`
node or edge exists, none of the four decoy entries exist.

**TP-007**: `MERGE (a:Agent {agentId:'_qa_selftest_producer2_4e24af1e'}) CREATE (a)-[:PRODUCED
{sessionId:'qa-dryrun-2'}]->(k:KaizenEntry {entryId:'qa-selftest-e2e-4e24af1e', ...})`,
`agent='_qa_selftest_producer2_4e24af1e'`. Expected: authorized.

**TP-008**: `MATCH (k:KaizenEntry {entryId:'qa-selftest-e2e-4e24af1e'}) MERGE (a:Agent
{agentId:'_qa_selftest_mentioned_4e24af1e'}) MERGE (k)-[:MENTIONS]->(a)`, `agent='cobb'`. Expected:
authorized (curator MENTIONS-write shape).

**TP-009**: graph-dba §4.1's count query against `qa-selftest-e2e-4e24af1e`. Expected:
`producedEdges=1, mentionEdges=1`.

**TP-010**: `MATCH (:Agent)-[p:PRODUCED]->(k:KaizenEntry {entryId:'qa-selftest-e2e-4e24af1e'})
DELETE p`, `agent='cobb'`. Expected: authorized; re-read confirms node still exists, `PRODUCED`
edge gone, `MENTIONS` edge still present.

**TP-011**: `MATCH (k:KaizenEntry {entryId:'qa-selftest-e2e-4e24af1e'})-[m:MENTIONS]->(:Agent
{agentId:'_qa_selftest_mentioned_4e24af1e'}) DELETE m`, `agent='cobb'`. Expected: authorized;
re-read confirms the `KaizenEntry` node no longer exists.

**TP-012**: `MATCH (a:Agent) WHERE a.agentId IN
['_qa_selftest_producer_4e24af1e','_qa_selftest_producer2_4e24af1e',
'_qa_selftest_mentioned_4e24af1e'] RETURN a.agentId`. Expected: all three still present (by
design — `:Agent` nodes are never deleted).

**TP-013**: after TP-002-011 complete (including cleaning up the TP-001 victim entry via the
curator full-clear shape), confirm zero `:KaizenEntry` nodes remain with any
`qa-selftest-*4e24af1e*` `entryId`, and total `:KaizenEntry` count is back to the pre-run baseline
(26).

## 5. Entry / exit criteria

**Entry**: container rebuilt and confirmed (given, per task brief); constraints `OPERATIONAL`
(confirmed above). **Exit**: all 13 items executed with evidence recorded; any attack that
unexpectedly succeeds is treated as a Blocker and reported immediately, run halted at that point
rather than proceeding to attempt cleanup silently; graph left with zero residual test data
(TP-013) assuming no Blocker.

## 6. Out of scope (restated)

Full-feature acceptance (already covered elsewhere per §1); the 13 prompts' prose; `SKILL.md`'s
prose; performance/load; anything requiring code changes (this is execution-only, per the
`qa-engineer` mandate not to fix code under test).
