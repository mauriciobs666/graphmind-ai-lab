# CPG MCP server/tool rename — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-19

## Intent
The MCP server and its single tool are still named after CPG (`cpg/mcp/`, `.mcp.json` server key
`"cpg"`, tool `mcp__cpg__query`), even though M5 (`docs/requirements/generic-cypher-mcp.md`)
already widened it into a general, graph-agnostic Cypher tool — it now reaches any FalkorDB
graph, not just Code Property Graphs, and M6 (`docs/requirements/generic-cypher-mcp2.md`) is
rolling that generic use out to the whole agent team's working memory. The stakeholder noticed
the name no longer describes what the tool does, and wants it renamed to reflect that.

## Problem & current state
`cpg` was the right name when the tool was purpose-built and scoped to CPG analysis
(`docs/requirements/cpg-query-access.md`). Since M5, the tool is mechanically and by-decision
generic — any named FalkorDB graph, read and (narrowly, attributed) write — but every visible
surface still says "cpg": the directory (`cpg/mcp/`), the `.mcp.json` server key, the tool name
an agent actually invokes (`mcp__cpg__query`), and the README/docs describing it. A newcomer (or
an agent reading its own routing description) sees "cpg" and reasonably assumes CPG-only scope,
which is no longer true.

This name is referenced across **60+ files** repo-wide — not just the `cpg/` component itself,
but `claude/AGENTS.md`, multiple agents' operative prompts and kaizen history, `skills/`
(`cpg-analysis`, `joern-cpg`, `agent-maintenance`), and a long list of `docs/plans`, `docs/reviews`,
`docs/test-plans`, `docs/test-reports` — even `mcp-monitor/`'s own docs, which cite it as an
example. This is a wide, cross-component rename, not a cosmetic single-file tweak.

## User stories
*(to be filled in as the interview proceeds)*

## Functional requirements
*(to be filled in as the interview proceeds)*

## Out of scope
*(to be filled in as the interview proceeds)*

## Acceptance criteria
*(to be filled in as the interview proceeds)*

## Open questions
- What should the new name actually be?
- Does "rename" mean the tool-facing name only (`mcp__cpg__query` → `mcp__<x>__query`), or also
  the directory (`cpg/` → `<x>/`), the `.mcp.json` server key, and the component's own identity
  throughout `AGENTS.md`/`README.md`?
- Does the CPG-*specific* capability (Joern-built Code Property Graphs, the `cpg-analysis` skill,
  `graph-dba`'s CPG pipeline) keep the word "cpg" anywhere, or does that also get relabeled?
- Sequencing/risk: is this a single atomic rename, or does it need a transition
  period/compatibility shim so in-flight sessions or cached references don't break?

## Decision log
- 2026-08-19 — Session opened. Raised mid-interview during the M6 (`generic-cypher-mcp2`)
  session: stakeholder noted "cpg doesn't reflect that it is now generic" and asked `tico` for a
  naming suggestion. `tico` offered an informal opinion (`cypher` over `graph` or a
  `falkordb`-flavored name, the latter risking confusion with the *official* `@falkordb/mcpserver`
  that M5's requirements doc already discusses and rejects), explicitly flagged as a suggestion,
  not a decision. Stakeholder chose to track the rename as its **own** delivery rather than fold
  it into M6, given the 60+-file blast radius — new document opened here, own topic slug (not a
  family member of `generic-cypher-mcp`/`generic-cypher-mcp2`, since this is about the tool's own
  identity, not the kaizen-inbox rollout).
