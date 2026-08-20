# Generic Cypher MCP — team-wide kaizen inbox rollout — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (`docs/requirements/generic-cypher-mcp2.md`, M7)

## Goal & definition of done

Deliver `docs/requirements/generic-cypher-mcp2.md` (Status: Ready for design, FR-1…FR-14,
AC-1…AC-13). Roll out the graph-backed kaizen working-memory pattern M5 proved on `graph-dba`
alone to the other eleven agents (`analyst`, `architect`, `cobb`, `coder`, `data-scientist`,
`devops`, `frontend-engineer`, `qa-engineer`, `tdd-engineer`, `teco`, `tico`): each gets its own
graph-backed raw-capture layer (same author/curator mechanism as M5, no redesign), a one-time
import of its current `kaizen/inbox.md`, then deletion of that file once the import is confirmed;
`kaizen/history.md` stays markdown/unchanged; a new team-wide query surface reaches every migrated
agent's raw learnings in one query (FR-7, new relative to M5); entry structure is locked as a
canonical 5-field contract plus a new session-ID field on new entries (FR-8/FR-8a); every doc
describing the kaizen-inbox convention is updated to match reality; the new-agent creation
convention is updated so a newly created agent is born on the graph-backed pattern; rollout is
explicitly **incremental** (FR-13) — partial migration is valid progress, not a failure state;
`graph-dba`'s own already-frozen `kaizen/inbox.md` (kept in-repo since M5) is also deleted as part
of this delivery (FR-14, a recorded supersession of M5's FR-4).

**Sequencing note (from the M6 coordination doc):** this design work was deliberately queued
behind `cpg-mcp-rename` (M6) so it would be designed and built directly against the final
`cypher`/`mcp__cypher__query` tool identity rather than being touched by that rename mid-flight.
M6 is now fully closed (`docs/plans/cpg-mcp-rename-coordination.md`, archived) — this delivery
starts clean.

## Unit ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U2 | `analyst` | `a7ebedd70af393e90` | in-flight | `docs/reviews/generic-cypher-mcp2.md` — plan-gate review of `docs/plans/generic-cypher-mcp2.md` | — |
| U1 | `architect` | `aed0cfdde5136e045` | delivered | `docs/plans/generic-cypher-mcp2.md` — decisions: (1) one shared graph `kaizen_team`, `author`-partitioned, not per-agent `kaizen_<agent>` graphs — zero new server code for FR-7 vs. new fan-out logic; (2) `CLAUDE_CODE_SESSION_ID` env var (live-verified in-session) answers FR-8a; (3) 15 independently-dispatchable units — 4 docs/substrate (D1–D3, G0) + 11 per-agent migrations (A1–A11, each bundling data-migration + cobb's paired prompt edit) + 2 acceptance passes (Q1 interim, Q2 closing); (4) no companion `-graph.md` note (FR-8 schema locked, no new modeling beyond §3.1/3.2). `kaizen_graph_dba` confirmed live at 0 entries (cobb already distilled M5's pilot data) — consolidation into `kaizen_team` is schema-only, no data migration. `cypher-mcp/server.py`'s write-auth confirmed graph-name-agnostic by full read — zero logic changes needed anywhere. AC-1…AC-13 all mapped to concrete live/static checks (§5). 8 open items flagged for plan-gate (§6), most notably: shared-graph-vs-per-agent call, retiring the empty `kaizen_graph_dba` key (not literally required by any FR/AC), and whether cobb's self-maintenance carve-out extends to editing its own `cobb.md`. | plan gate (`analyst`) → — |

## Resume note (paused 2026-08-19, session interrupted for an urgent unrelated task)

**Current state:** U1 (architect's plan) is delivered and committed. U2 (`analyst` plan-gate review)
was dispatched and is **in-flight**, agent id `a7ebedd70af393e90`, no result received yet in this
session.

**To resume:**
1. Check whether agent `a7ebedd70af393e90` is still reachable (`ListAgents`, or `SendMessage` to
   that id asking for status). If it resolves, wait for/retrieve its result normally.
2. If it no longer resolves (cold session, process gone), re-dispatch the same plan-gate review
   from scratch — the brief is fully reconstructable from this ledger's U2 row and
   `docs/plans/generic-cypher-mcp2.md` §6's own 8 flagged open items; nothing was lost, since no
   partial review output was ever committed.
3. Once U2's review lands: verify its claims independently (per this repo's standing practice —
   see `docs/plans/cpg-mcp-rename-coordination.md` for the pattern this delivery is following),
   resolve any `needs changes` verdict back to `architect` (same agent id `aed0cfdde5136e045` if
   still reachable, else a fresh dispatch with the review's path), and only start implementation
   (§4's 15 units) once the plan gate reaches `approve`/`approve with suggestions`.
4. **Do not start implementation units before the plan gate closes** — this is a repo-standing
   practice (two-gate rule: plan gate catches design-level blockers before any code/doc-migration
   work begins), not just this delivery's choice.
5. Implementation dispatch, once unblocked, should follow §4's own step-table sizing (15 units,
   already right-sized — no further splitting needed) and this repo's standard same-file
   serialization / parallel-independent-unit dispatch pattern.

## Notes

- FR-13/AC-10 make incremental delivery a first-class requirement, not a compromise — the plan
  should give the implementer(s) a real per-agent (or per-batch) unit of work, not one atomic
  12-agent step.
- FR-7 (team-wide query surface) is new relative to M5 and has no prior art in this repo yet —
  worth explicit design attention (one shared graph with an agent-partition property vs.
  per-agent graphs plus a federated query helper vs. something else), left open by the
  requirements doc on purpose ("Context for the architect").
- FR-8a's session-ID field needs a concrete mechanism for how an agent obtains its own Claude
  Code session ID at write time — also explicitly left to the architect.
- This is cross-cutting (touches `claude/<every-agent>/kaizen/`, `claude/AGENTS.md`,
  `claude/README.md`, `docs/BACKLOG.md`, and the mechanism `generic-cypher-mcp.md`/M5 already
  built) — no single component owns it; docs land at repo-root `docs/plans/`, matching
  `generic-cypher-mcp2.md`'s own location.
- Per this delivery's own step-table sizing rule: once the plan lands, dispatch implementation
  units sized to the plan's own step boundaries (likely per-agent or per-batch given FR-13),
  not as one mega-dispatch.
