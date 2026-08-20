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
| U2 | `analyst` | `a7ebedd70af393e90` | delivered | `docs/reviews/generic-cypher-mcp2.md` — 2 blockers, 4 majors, 3 minors. **B1**: `teco`/`tico`/`data-scientist` lack `mcp__cypher__query` in their frontmatter `tools:` allowlist — units A6/A10/A11 as scoped cannot execute (plan's "no agent-wiring step needed" claim is false for these 3). **B2**: AC-11 (delete `graph-dba`'s own frozen `kaizen/inbox.md`) has no owning unit anywhere in §4 — grepped, confirmed absent. **M1**: `claude/scripts/audit-team.sh` check 1 hard-requires `kaizen/inbox.md` per agent (verified by running it live) — this plan breaks team certification tooling team-wide, no unit updates it. **M2**: 5 doc-scoped write-guard hook scripts hardcode `<agent>/kaizen/inbox.md` as an allowed path + name it in escalation text — stale + a latent guard weakening (would silently permit resurrecting a deleted inbox), untouched by any unit. **M3**: `claude/teco/teco.md`'s own coordination-duty prose (2 passages, outside its Learning-capture section) references every delegate's `kaizen/inbox.md` — unit A6 doesn't reach it; same failure shape as M5's own B1 (fixed list under-covering real scope) — analyst notes the plan's own AC-8 grep would literally surface M1/M2/M3 together as unplanned acceptance-time defects. **M4**: second opinion against open item 7 — `cobb` self-editing its own `cobb.md` Learning-capture section is not clearly covered by its existing self-maintenance carve-out; recommends reassigning to a different owner. 3 minors (stale line-count drift in §2's survey; D2's SKILL.md §5 rewrite under-scoped; `cobb` named owner on 13/15 units — a throughput-concentration risk, not a defect). "What's solid" section confirms the core shared-graph mechanism-reuse claim, `kaizen_graph_dba` empty state, and `CLAUDE_CODE_SESSION_ID` all independently re-verified (session ID re-confirmed from a second, cold session). **teco has not yet independently re-verified these findings** — queued for next resume, see Resume note. | plan gate → **needs changes** |
| U1 | `architect` | `aed0cfdde5136e045` | delivered | `docs/plans/generic-cypher-mcp2.md` — decisions: (1) one shared graph `kaizen_team`, `author`-partitioned, not per-agent `kaizen_<agent>` graphs — zero new server code for FR-7 vs. new fan-out logic; (2) `CLAUDE_CODE_SESSION_ID` env var (live-verified in-session) answers FR-8a; (3) 15 independently-dispatchable units — 4 docs/substrate (D1–D3, G0) + 11 per-agent migrations (A1–A11, each bundling data-migration + cobb's paired prompt edit) + 2 acceptance passes (Q1 interim, Q2 closing); (4) no companion `-graph.md` note (FR-8 schema locked, no new modeling beyond §3.1/3.2). `kaizen_graph_dba` confirmed live at 0 entries (cobb already distilled M5's pilot data) — consolidation into `kaizen_team` is schema-only, no data migration. `cypher-mcp/server.py`'s write-auth confirmed graph-name-agnostic by full read — zero logic changes needed anywhere. AC-1…AC-13 all mapped to concrete live/static checks (§5). 8 open items flagged for plan-gate (§6), most notably: shared-graph-vs-per-agent call, retiring the empty `kaizen_graph_dba` key (not literally required by any FR/AC), and whether cobb's self-maintenance carve-out extends to editing its own `cobb.md`. | plan gate (`analyst`) → — |

## Resume note (updated 2026-08-19, session paused again for an unrelated urgent task)

**Current state:** U1 (architect's plan) delivered and committed. U2 (`analyst` plan-gate review)
**delivered** — `docs/reviews/generic-cypher-mcp2.md`, verdict **needs changes** (B1, B2 blockers;
M1–M4 majors; m1–m3 minors — full summary in U2's ledger row above). **The review file itself is
committed** (see commit log), but **teco has not yet independently re-verified its findings**, and
**no fix has been dispatched to `architect` yet** — session paused immediately after the review
landed, before any of that follow-up work started.

**To resume, in order:**
1. Read `docs/reviews/generic-cypher-mcp2.md` in full (don't rely on the ledger summary alone).
2. Independently spot-check the two blockers at minimum before trusting them (per this repo's
   standing practice — the pattern `docs/plans/cpg-mcp-rename-coordination.md`'s U2/U2-regate rows
   follow): B1 — read `teco`/`tico`/`data-scientist`'s frontmatter `tools:` lines yourself, confirm
   `mcp__cypher__query` is genuinely absent; B2 — grep the plan text for
   `claude/graph-dba/kaizen/inbox.md` yourself, confirm no unit executes its deletion. Spot-check at
   least one major (M1 is cheap: run `claude/scripts/audit-team.sh` live and read its check-1 logic).
3. Dispatch the fix back to `architect`, agent id `aed0cfdde5136e045` if still reachable via
   `SendMessage` (resumes from its own transcript — cheaper and more context-aware than a fresh
   spawn), else a fresh `Agent` call with the review's path and a summary of what changed. Brief it
   on all 2+4+3 findings, not just the blockers — the majors (especially M1's tooling-breakage and
   M3's teco.md cross-reference) are load-bearing, not cosmetic.
4. Once the fix lands, re-dispatch the **same** `analyst` (agent id `a7ebedd70af393e90` if still
   reachable) for a re-gate pass — don't accept the fix on the architect's word alone.
5. **Do not start implementation units (§4's 15) before the plan gate reaches
   approve/approve-with-suggestions.** Two-gate rule, standing repo practice, not just this
   delivery's choice.
6. Implementation dispatch, once unblocked, follows §4's own step-table sizing — but note the
   review's own findings will likely reshape that table (B1 needs a frontmatter-wiring step folded
   somewhere; B2 needs G0's scope widened; M2/M3 need scope added to A3/A4/A6/A7/A10/A11 as
   applicable) — re-read the fixed plan's step table fresh rather than assuming §4 as originally
   written still matches 1:1.

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
