# Graph-dba kaizen distillation — first live production run coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (no backlog id; stakeholder-triggered end-to-end validation of the M5 `generic-cypher-mcp` pilot)

## Context

M5 (`generic-cypher-mcp`, closed, `docs/plans/generic-cypher-mcp-coordination.md`) moved
`graph-dba`'s raw kaizen capture from `claude/graph-dba/kaizen/inbox.md` to a working-memory
FalkorDB graph, `kaizen_graph_dba` (`:KaizenEntry` nodes). QA's own acceptance pass (U7, AC-5 of
that coordination) already ran `cobb`'s distillation procedure once, live, as part of *proving*
AC-5 — one entry (the `META_DATA`-absence finding) was genuinely promoted and cleared, graph count
6→5. That was a test exercise of the mechanism, not a maintenance pass.

The stakeholder now wants the mechanism exercised as **real maintenance work** — the first
genuine end-to-end production run, not a test fixture — covering both raw-capture sources:

- `claude/graph-dba/kaizen/inbox.md` — carries a "FROZEN — 2026-08-18" note: content was
  one-time-imported into the graph, `graph-dba` no longer appends here. Expected: no action
  needed on the file itself, but confirm rather than silently skip.
- The live `kaizen_graph_dba` graph — confirmed via direct `redis-cli GRAPH.QUERY` at dispatch
  time to hold exactly **5** `:KaizenEntry` nodes (`entryId`/`date`): `58ad5ace-…`/2026-08-16,
  `f8c28d75-…`/2026-08-16, `6e5d6451-…`/2026-08-16, `7f0e3cf1-…`/2026-08-17,
  `80ef4889-…`/2026-08-17 — the 5 of the original 6 not already promoted by QA's AC-5 run.

Procedure: `skills/agent-maintenance/SKILL.md` §5 (read → verify → route → log-then-clear;
append-to-`history.md`-before-graph-clear is non-negotiable for `graph-dba` entries specifically).

## Units

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `cobb` | `a0aa4e8c34c52cca4` | delivered | inbox.md confirmed inert; 4/5 entries promoted (`claude/graph-dba/falkordb-quirks.md`, `falkor-chat/docs/DESIGN.md` §6.2, `skills/joern-cpg/SKILL.md` ×2), 1 entry (`6e5d6451…`, unreconciled `DETACH DELETE` count) kept open as K-007 + left live in graph; `claude/graph-dba/kaizen/history.md` updated; own-process learning promoted into `skills/agent-maintenance/SKILL.md` §5 step 2 | `analyst` → — |
| U2 | `analyst` | `aa0ab705481dc015d` | delivered | diff-scoped re-gate at `docs/reviews/graph-dba-kaizen-distillation.md` — verdict **approve**, 1 minor (no forward pointer from kept-open `6e5d6451…` node to K-007) + 1 open question (kept-open semantics: clear node once logged, or leave live) — teco independently re-verified graph count (1 row) matches | self → **approve** |
| U3 | `cobb` | `a0aa4e8c34c52cca4` (resumed) | accepted | `skills/agent-maintenance/SKILL.md` §5 — decided "clear once logged" for every disposition (kept-open included, file-based agents too) + a dedup-check rule before opening a new backlog item for an already-tracked entry; consequence: cleared `6e5d6451…` (already fully recorded in `history.md`/K-007), `kaizen_graph_dba` now 0 `:KaizenEntry` nodes. teco independently re-verified: live `count(e)` → 0, `SKILL.md` diff reads coherently with dated origin notes, `history.md` new entry matches, `plan.md` correctly left untouched (K-007 already the durable record). | teco spot-check → **accepted** |

## Sequencing

Single unit, one review gate: `analyst` re-gates the resulting diff (docs/prompt/KB edits +
`history.md` + final graph state), diff-scoped, same as every other `cobb` distillation pass.
`teco` independently spot-checks (graph count, file diffs) before closing, mirroring
`kaizen-inbox-distillation2-coordination.md`'s precedent.

## Close (teco, 2026-08-18)

`analyst`'s re-gate (U2) came back **approve** with one minor + one open question, both about the
graph-based "kept open" disposition's semantics rather than any defect in the 4 promotions. Routed
both back to `cobb` (same delegate, resumed by agent id) rather than the user — this is `cobb`'s
own procedural domain (the distillation convention it designed and maintains), not a product
decision. `cobb` (U3) decided "clear once logged" applies uniformly across every disposition
(including kept-open, and retroactively to file-based agents too), wrote it as an explicit rule
into `skills/agent-maintenance/SKILL.md` §5 alongside the dedup-check the minor finding asked for,
and — as the direct consequence of that decision — cleared the one remaining `6e5d6451…` node
(already fully recorded in `history.md` + K-007). `teco` independently re-verified: live
`kaizen_graph_dba` `MATCH (e:KaizenEntry) RETURN count(e)` → **0**; `SKILL.md`/`history.md` diffs
read coherently and match the self-report; `plan.md` correctly untouched this round.

**Net result of this pilot's first real production run:** of the original 6 raw entries (5 live +
1 already promoted by QA's AC-5 test), all 6 are now fully distilled — 5 promoted into durable
homes (`falkordb-quirks.md`, `falkor-chat/docs/DESIGN.md`, `skills/joern-cpg/SKILL.md` ×2, plus
QA's earlier `cpg-model.md` promotion), 1 converted into a tracked backlog item (K-007) with no
raw capture left behind. `kaizen_graph_dba` is empty — confirming the "bounded by clear-on-promote
design" claim `docs/plans/generic-cypher-mcp-graph.md` made at design time, and demonstrated once
already by QA's acceptance pass, now demonstrated again under real, non-test conditions. The
distillation procedure itself came out of this run measurably improved: two explicit rules
(re-derive-don't-just-cite; kept-open clears too, with a dedup check) that didn't exist before this
pass, now written into `agent-maintenance` SKILL.md for every future distillation, file-based
agents included, not just `graph-dba`.

Committed by explicit path (excludes `claude/tico/kaizen/inbox.md`, unrelated pre-existing
uncommitted work from a different session): `claude/cobb/kaizen/history.md`,
`claude/cobb/kaizen/inbox.md`, `claude/graph-dba/falkordb-quirks.md`,
`claude/graph-dba/kaizen/history.md`, `claude/graph-dba/kaizen/plan.md`,
`falkor-chat/docs/DESIGN.md`, `skills/agent-maintenance/SKILL.md`, `skills/joern-cpg/SKILL.md`,
`docs/plans/graph-dba-kaizen-distillation-coordination.md`,
`docs/reviews/graph-dba-kaizen-distillation.md`. No backlog/HISTORY.md entry — this run carries no
backlog id (`Tracks: —`), matching the `kaizen-inbox-distillation2-coordination.md` precedent for
an ungated stakeholder-triggered sweep. `docs/reviews/graph-dba-kaizen-distillation.md` stays
`Status: active` (not archived) — this is a distillation-pass review, not a milestone-closing plan
review, again mirroring the `kaizen-inbox-distillation`/`…2` precedent of leaving that review kind
open as a living reference.
