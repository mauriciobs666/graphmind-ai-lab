# Code Property Graph (Joern + FalkorDB) — Getting Started — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-201…C-208, C-301…C-307 (M1–M3)

## Scope & verdict

Static factual/architectural review of `docs/manuals/cpg-getting-started.md` (the `tico`-authored
end-user manual for the Joern-CPG-in-FalkorDB capability), per the split `teco` routes on a manual
review: this half checks factual/architectural claims and clarity for a non-technical audience;
the behavioral half (do the walkthroughs work against the running app) is `qa-engineer`'s and is
**not** covered here.

Checked against the primary sources, not against the manual's own framing: `claude/graph-dba/graph-dba.md`,
`claude/analyst/analyst.md`, `claude/architect/architect.md`, `claude/qa-engineer/qa-engineer.md`,
`claude/README.md`, `cypher-mcp/README.md`, `cypher-mcp/server.py`, `skills/joern-cpg/SKILL.md`,
`docs/requirements/joern-cpg-pipeline.md`, `skills/cpg-analysis/SKILL.md`, `docs/BACKLOG.md`, and
`claude/tico/tico.md` (Mode 3, for the manual-authoring convention itself). Also live-verified
several of the manual's numeric/behavioral claims directly against the running `cpg_falkorchat`
graph through `mcp__cypher__query` (see evidence below) rather than trusting the docs alone.

**Verdict: approve with suggestions.** No blockers. Every specific factual claim I checked —
agent responsibilities, MCP tool guarantees, pipeline/build behavior, scope boundaries — is
accurate and current against the post-cobb-pass (commit `9c116b1`) state of the agent definitions,
not stale. Two things are worth the owner's attention before/at the next revision: a scope/clarity
question about how much pipeline-internals the diagrams should show, and an **adjacent defect**
(not in the manual itself) that undercuts the manual's promise if a reader hits it.

## Findings

### Major

**M1 — The live "graph not found" error the manual's own FAQ points a reader toward cites a retired agent name.**
The manual's FAQ says: *"Ask `graph-dba` to build one, or to confirm the exact graph name currently
loaded"* (`docs/manuals/cpg-getting-started.md:163`), consistent with `graph-dba` being the current,
correct owner of CPG builds (confirmed: `claude/graph-dba/graph-dba.md` frontmatter and commit
`cbf26c4` "retire joern agent, fold CPG generation into graph-dba"). But the actual tool response a
reader gets when a graph doesn't exist still says otherwise. Live-verified:

```
mcp__cypher__query(graph="cpg_this_graph_should_not_exist_probe", cypher="MATCH (n) RETURN count(n)")
→ "Graph '...' does not exist. Loaded graphs: cpg_falkorchat, cpg_salesperson, ws:acme, reference,
   ws:test. If no CPG is loaded, building and loading one is the joern agent's job (joern-cpg
   pipeline) — this tool only queries."
```

`cypher-mcp/server.py:327` hard-codes "the joern agent's job" — a name that no longer exists in the
roster (`claude/README.md` has no `joern` agent; it was folded into `graph-dba`). The same stale
string is baked into `docs/plans/cpg-query-access.md:627`. This is **not a defect in the manual** —
the manual itself never names "joern" as an agent and correctly routes to `graph-dba` throughout —
but a reader who follows the manual's own advice ("ask an agent to do it... if any of this is
missing, the agents report it in plain language") and then triggers this exact error will see the
tool contradict the manual's agent-routing story, which undercuts trust in both. Recommend routing
a fix for `cypher-mcp/server.py`'s error string (and the design doc) to whoever owns that component
(`graph-dba`/`coder`, since it's a one-line string literal, not a design change) — out of `tico`'s
own remit to fix, but worth flagging so it doesn't ride along silently. This does not block
approval of the manual itself.

### Minor

**m1 — Pipeline-internals diagrams edge past the "never internal architecture/file layout" rule the manual's own convention sets.**
`claude/tico/tico.md` (Mode 3) states manuals are audience = end user and must "never [document]
internal architecture, file layout, or implementation choices (that's what `docs/plans/` is for)."
The Overview `flowchart` (`docs/manuals/cpg-getting-started.md:37-48`) and the build-walkthrough
`sequenceDiagram` (lines 81-97) name the specific transform script (`cpg-to-falkordb.py`) and the
specific Joern export format token (`neo4jcsv`) — both are implementation choices from
`skills/joern-cpg/SKILL.md`, not user-facing concepts. This is a defensible judgment call, not a
clear violation: the manual explicitly scopes itself for a stakeholder who might "run the pieces by
hand" (line 12), which justifies naming *some* pipeline stages. But `cpg-to-falkordb.py` and
`neo4jcsv` specifically add no value to a reader deciding whether/when to ask for a CPG build —
they're the kind of detail a `docs/plans/`-level doc carries. Suggested fix: keep the four
conceptual stages (parse → export → transform → load) but drop the literal script/format names from
the diagram labels, e.g. "export (Joern's CSV format)" instead of "export (neo4jcsv format)", and
"transform export → Cypher" instead of naming `cpg-to-falkordb.py`.

**m2 — The manual never mentions the `redis-cli` fallback, despite explicitly targeting a reader who might run pieces by hand.**
Line 12 says this manual is written so "a human stakeholder can also follow along, ask an agent to
do it, or run the pieces by hand." For the *query* side specifically, `cpg-analysis/SKILL.md` and
`cypher-mcp/README.md` both document that `redis-cli GRAPH.QUERY`/`GRAPH.RO_QUERY` is the documented
fallback and the *only* path outside Claude Code (OpenCode/Kiro aren't wired — backlog C-310). A
stakeholder trying to "run the pieces by hand" outside a Claude Code session has no path today
without that fallback, and the manual doesn't mention it exists. This is a completeness gap, not a
factual error — the manual's claims about the MCP tool are all accurate for the Claude-Code path it
does describe. Suggested fix: one sentence in the "asking the CPG a question" walkthrough or FAQ,
e.g. "Outside Claude Code, the same graph can be queried directly with `redis-cli GRAPH.QUERY` — ask
an agent, or see `cypher-mcp/README.md` for the exact command."

## What's solid

- **Every checked factual claim is accurate and current**, not stale relative to the `9c116b1`
  cobb pass: agent responsibility routing (`graph-dba` build-side; `analyst`/`architect` impact,
  `analyst` RCA/code-review, `qa-engineer` test-gap) matches `claude/README.md`'s catalog and each
  agent's own frontmatter/body verbatim in substance.
- **The MCP guarantees are correct and live-verified**, not just doc-sourced: read-only enforcement
  (`GRAPH.RO_QUERY`), a typo'd graph name not materializing an empty key, `EXPLAIN` working and
  `PROFILE` being actively refused with the documented fallback message, and truncation being
  display-only with the true row count always reported — all confirmed both in `cypher-mcp/README.md`
  and by directly exercising `mcp__cypher__query` against the live `cpg_falkorchat` graph during this
  review (110,048 nodes / 734,929 edges — consistent with the "41 files → 110k nodes / 735k edges"
  figure `skills/joern-cpg/SKILL.md` cites, and with the manual's "~2,700 nodes / ~18,000 edges per
  Python file" rule of thumb).
- **Scope claims (not RAG, not a coverage tool) are accurate**, matching `docs/requirements/joern-cpg-pipeline.md`'s
  Out-of-scope section and `skills/cpg-analysis/SKILL.md`'s test-gap description word-for-word in
  substance.
- **Destructive-reload framing is accurate and appropriately non-technical**: "refuses to run...
  guarded... explicit, visible step... hard-gated to require human approval" matches
  `skills/joern-cpg/SKILL.md`'s "Reloading is deliberate (destructive)" section and `graph-dba.md`'s
  `guard-destructive-ops.sh` hook, without leaking the hook's implementation into the manual.
  Similarly, the "Walkthrough — checking you're ready" section shows good restraint versus the
  containerization details in `cypher-mcp/README.md` (no mention of `docker-run.sh`,
  `host.docker.internal`, etc.) — exactly the altitude a manual should hold.
- **The `@graph-dba build a CPG for <repo/path>` invocation shown is a real Claude Code convention**
  (the `@`-mention subagent syntax), not an invented one.
- **Backlog tracking IDs in the header (`C-201…C-208, C-301…C-307`) check out** against
  `docs/BACKLOG.md`'s M2/M3 sections — both delivered, both cover exactly this capability.

## Open questions

- Should `m1`'s diagram trims (script/format names) happen now, or is the current level of detail
  an intentional choice for the stated "run it by hand" audience? This is `tico`'s call — I've
  flagged it as a judgment call, not a factual defect.
- The stale "joern agent" string in `cypher-mcp/server.py`/`docs/plans/cpg-query-access.md` (M1) is
  outside this manual's remit to fix — worth confirming with the caller whether it should be routed
  to `graph-dba`/`coder` as a follow-up, since it's a live, user-facing inconsistency even though it
  isn't a manual defect per se.
