# Graph-dba kaizen distillation — post-implementation re-gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Scope & verdict

Diff-scoped review of `cobb`'s first genuine (non-acceptance-test) maintenance pass over
`graph-dba`'s `kaizen_graph_dba` FalkorDB graph, per `docs/plans/graph-dba-kaizen-distillation-coordination.md`
(unit U1). Reviewed the full uncommitted working-tree diff except `claude/tico/kaizen/inbox.md`
(out of scope per brief, unrelated pre-existing work):

- `claude/graph-dba/falkordb-quirks.md` (1 new bullet, entry `58ad5ace…`)
- `falkor-chat/docs/DESIGN.md` §6.2 (1 new bullet, entry `f8c28d75…`, corrected)
- `skills/joern-cpg/SKILL.md` (2 edits, entries `7f0e3cf1…` and `80ef4889…`)
- `claude/graph-dba/kaizen/plan.md` (new K-007, from kept-open entry `6e5d6451…`)
- `claude/graph-dba/kaizen/history.md` (dated entry documenting all 5 dispositions)
- `skills/agent-maintenance/SKILL.md` §5 step 2 (cobb's own-run process learning)
- `claude/cobb/kaizen/history.md` / `claude/cobb/kaizen/inbox.md` (cobb's own bookkeeping)

Baseline re-derived directly, not taken on cobb's report: `mcp__cypher__query(kaizen_graph_dba,
"MATCH (e:KaizenEntry) RETURN e.entryId, e.date, e.author, e.context, e.suggestedHome")` now
returns exactly **1** row, `entryId 6e5d6451-72fa-400c-b002-52757727f805`, `date 2026-08-16` —
matches the expected post-run state exactly.

**CPG: not applicable — this review's artifacts are prose/knowledge-base docs and the
`kaizen_graph_dba` working-memory graph, not source code any `cpg_*` Code Property Graph
indexes; the live-graph verification needed here is the `kaizen_graph_dba` reads/checks above,
not a CPG.**

**Verdict: approve.** All four promotions and the one kept-open disposition are independently
verifiable and hold up; the two flagged judgment calls were both sound. No blockers, no majors.
One minor process-consistency observation on the "kept open + also opened K-007" dual-tracking,
below.

## Findings

### Minor — kept-open entry has no forward pointer to K-007, and the graph write-authorization model can't add one

`claude/graph-dba/kaizen/plan.md` K-007 correctly cites the live entry's `entryId` in its Notes
field, but the reverse link doesn't exist: the live `:KaizenEntry` node itself (confirmed via
direct query — `keys(e)` is exactly `['entryId','date','fact','evidence','context',
'suggestedHome','author','createdAt']`) carries nothing that says "already tracked as K-007." A
future distillation pass that reads this entry fresh (per §5 step 1, ordered by date) has no
signal from the graph alone that a backlog item already exists for it — only a manual
cross-check against `plan.md`'s Notes fields would catch it, and nothing in `agent-maintenance`
SKILL.md §5 currently instructs a future distiller to do that cross-check before opening a
second backlog item for the same fact.

This isn't cobb's fault in the sense of a mistake this run — the MCP tool's write
authorization for `kaizen_graph_dba` only allows `graph-dba` to `CREATE` its own entries and a
curator to `DETACH DELETE` by id; there is no sanctioned `SET` to annotate a node in place. Given
that constraint, cobb's choice (K-007 in `plan.md` for durability/visibility, node left live in
the graph for the structural-arithmetic detail a terse bullet shouldn't have to duplicate) is a
reasonable read of "unverifiable ≠ discard" — but it's worth closing the gap explicitly rather
than relying on the next distiller to remember. Suggested improvement: add a line to
`agent-maintenance` SKILL.md §5 step 3/4 — when a `graph-dba` entry is kept open *and* a backlog
item is opened for it, the future distillation pass must check the relevant `kaizen/plan.md` for
an existing K-item citing the entry's `entryId` before opening a new one for the same fact (a
one-line dedup rule, cheap to add, closes the drift class before it happens once).

### Open question — for `graph-dba`, does "kept open" mean "leave the node live" or "clear it, K-007 is the durable record"?

The brief flags this and it's worth stating a view: the general §5 step 4 text says, for a
file-based agent, "the processed entry is then removed directly from `inbox.md`" after every
disposition, because "the history entry is the durable record either way" — that reads as a
uniform rule (promoted, discarded, *and* kept-open entries all get removed from the raw capture
once logged). `graph-dba`'s graph-based equivalent, as actually exercised here, deviates from
that: the node stays live for the kept-open case specifically. Both readings are defensible
(the general rule optimizes for "the raw capture is transient scratch space, not a second
backlog"; cobb's choice here optimizes for "don't lose the structural-arithmetic detail behind a
terse `plan.md` line"), and the skill text doesn't actually address the graph case's
kept-open disposition at all — this is a genuine ambiguity in the procedure, not something cobb
got wrong. My view: I'd lean toward clearing the node once `history.md` + `plan.md` (K-007) carry
the full record — matching the file-based convention and removing the future-duplicate-item risk
above — but keeping it live is not unreasonable given the graph is explicitly cobb-uneditable
(no in-place annotation possible) and the detail genuinely doesn't fit a one-line backlog
bullet. This is a call for `cobb`/the stakeholder to make explicitly in the skill, not something
this review can resolve unilaterally.

## What's solid

- **Every re-verifiable claim in this diff reproduced independently, not just as cobb reports
  it.** `GRAPH.EXPLAIN` on a never-created key: reproduced live (`ERR Invalid graph operation on
  empty key`), and I additionally confirmed via `GRAPH.LIST` before/after that the key is *not*
  materialized by the failed `EXPLAIN` call — a stronger check than cobb's own history entry
  claims to have run, and it held. `MODULE LIST` confirms the cited module version (`41811`)
  matches the live instance. The `.py` file count (65, `find falkor-chat/server/{falkorchat,tests}
  -name '*.py' | wc -l`) matches exactly. `git log -S'CREATE (sr)-[:RAN]->(cur)'` confirms commit
  `3921f87`, dated 2026-07-12 — five weeks before the raw entry's 2026-08-16 date, exactly as
  claimed.
- **The `f8c28d75…` correction (the most-scrutinized call) is right, and right to make.** Read
  `record_step_and_advance` at its current location, `falkor-chat/server/falkorchat/repository.py:1370-1391`:
  line 1381 is `CREATE (sr)-[:RAN]->(cur)` — a real edge the original raw entry's evidence block
  flatly denied ("no edge is ever created from `StepRun` to `Step`"). Grepped the entire shipped
  surface (`repository.py`, `docs/QUERIES.md`, `docs/DESIGN.md`) for any `MATCH`/traversal of
  `RAN` — none exists; every hit is either the `CREATE`, the schema-diagram line, or prose. So
  the corrected `DESIGN.md` §6.2 bullet's every clause holds: the edge exists, it's currently
  write-only, and the *practical* blast-radius conclusion the original entry reached is
  preserved (correct conclusion, narrower true reason). Promoting a corrected version rather
  than either the literal (wrong) original or a bare discard was the right call: verbatim
  promotion would have shipped a false absolute claim into `DESIGN.md`, a project document other
  agents will trust without re-deriving it themselves.
- **`skills/agent-maintenance/SKILL.md` §5 step 2's new sentence is a real, non-redundant
  addition.** The pre-existing text ("is it still true? Re-check cheaply... environment facts
  rot on upgrades") is about staleness over time; the new sentence ("re-derive the fact
  yourself... a citation can be real and still misdescribe what's there") is about a citation
  being wrong *the day it was written*, a different failure mode entirely. Cobb's authority to
  self-promote this mid-run is explicit in its own prompt (`claude/cobb/cobb.md:86`: "you are
  the maintainer, so same-run promotion with full §1/§2 bookkeeping is in-bounds for you
  alone") — this isn't an overreach of the general "agents never promote their own entries"
  rule, which targets producing agents, not the maintainer.
- **`claude/cobb/kaizen/inbox.md` was actually cleared, not left duplicated.** The entry that
  triggered the same-run promotion never persists as visible inbox content (write-then-clear
  nets to a placeholder-comment-only diff) — consistent with the template's contract, and
  `history.md` carries the durable record of what was learned and why, per §5's own logic.
- **K-007 is well-formed** against the established `plan.md` convention (table row +
  `### K-007` section with Status/Priority/Rationale/Proposed change, matching K-005/K-006's
  shape, plus an appropriate extra Notes field disclosing the dual-tracking decision).
- **`falkordb-quirks.md` and `skills/joern-cpg/SKILL.md` placements are correct as described** —
  the `GRAPH.EXPLAIN` bullet sits under "Ops, config & tooling," directly above the existing
  `GRAPH.PROFILE`-isn't-read-only entry it cross-references (confirmed by reading both); the
  `tee`/exit-code bullet sits right after the "Loading at scale" bullet in Gotchas, as claimed;
  the reframed Scale bullet correctly drops the stale 41-file number in favor of a
  measure-your-own-repo instruction, while preserving the per-file rate that both worked runs
  (41-file and 60-file) actually corroborate.
- **`claude/graph-dba/kaizen/history.md`'s single dated entry is complete and accurate** against
  every other file in the diff — the disposition, destination, and reasoning it states for each
  of the 5 entries matches what actually landed in `falkordb-quirks.md`, `DESIGN.md`,
  `SKILL.md` ×2, and `plan.md`, and it correctly identifies which entry stayed live and why.

## Open questions

- See "kept open vs. cleared" above — a stakeholder/`cobb` decision on the graph-based agent's
  kept-open convention, not a defect to fix unilaterally.
- Should the SKILL.md §5 dedup-check I suggest (minor finding above) be added now, or deferred
  until a second kept-open-with-backlog-item case actually recurs? Either is defensible; I'd
  lean toward adding it now since the cost is one sentence and the failure mode (a silently
  duplicated K-item) is cheap to prevent and annoying to notice after the fact.
