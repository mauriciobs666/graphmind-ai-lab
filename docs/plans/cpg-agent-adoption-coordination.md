# CPG agent adoption — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** cpg-agent-adoption (M4, proposed — C-4xx TBD in `docs/BACKLOG.md`)

## Goal & definition of done

Deliver `docs/requirements/cpg-agent-adoption.md` (Status: Ready for design, FR-1…FR-9 /
AC-1…AC-6). Widen which agents discover and use a loaded CPG, make that discovery a default
orientation step, and let a consulting agent judge/flag graph staleness — without touching the
MCP read path (FR-8), without automatic rebuild (FR-6/FR-7), and without a proactive build-out
(FR-7). AC-6 requires the downstream plan to state explicitly that it **extends** — not silently
overrides — the consumer-scope boundary set at M2 (`docs/plans/m2-cpg-analysis-skill.md`) and M3
(`docs/requirements/cpg-query-access.md`).

Design ownership is split per the requirements doc's own "Out of scope" section: `cobb` for
agent/skill/prompt/hook wiring (roster, discovery step, staleness-surfacing UX), `graph-dba` for
freshness mechanics (FR-5/FR-6 technical piece). Sequenced (not parallel): `graph-dba`'s
freshness-marker design is a narrow, self-contained technical question with no dependency on the
roster/discovery decisions; `cobb`'s primary plan needs to *cite* graph-dba's concrete recipe to
be coherent, so it runs second, reading graph-dba's delivered note by path.

## Unit ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `graph-dba` | `a33157978bdfa69ef` | delivered | `docs/plans/cpg-agent-adoption-graph.md` — verified: `:CpgBuildInfo` singleton node (BUILT_AT/SOURCE_PATH/SOURCE_COMMIT/SOURCE_DIRTY), stamped at end of `pipeline.sh`'s `--load` branch, read via new `references/freshness.md` recipe through unchanged `mcp__cpg__query`; FR-7/FR-8 explicitly confirmed; no backfill for the two live graphs (reasoned) | — |
| U2 | `cobb` | `a191d727c48f69561` | delivered | `docs/plans/cpg-agent-adoption.md` — verified: roster widened to 6 (added `coder`/`tdd-engineer`/`frontend-engineer`, 5 excluded with stated reasons), discovery via per-agent description/body edits (not root AGENTS.md), `cpg_<component>` naming-guess mechanic, evidence-trail convention (§3), explicit AC-6 reconciliation (§4), file-by-file task list (§6), BACKLOG.md M4 proposal C-401…C-407 (§7) | — |
| U3 | `analyst` | `ac8ee9e1b713aa43e` | delivered | `docs/reviews/cpg-agent-adoption.md` — **verdict: approve with suggestions, zero blockers**. 1 Major (§7 backlog-proposal FR/AC-tagging gap: FR-4/AC-3/AC-4/AC-5 never tagged despite closing-sentence claim of full coverage), 2 minor (SKILL.md nav-table Consumer column missing from task list; §4 doesn't mention graph-dba's sibling plan adds a 5th recipe file). Everything independently re-verified checked out. | plan gate: **approve w/ suggestions** |
| U2-fix | `cobb` | `a191d727c48f69561` | delivered | patched `docs/plans/cpg-agent-adoption.md` in place — teco spot-checked §6/§7: all 3 findings correctly addressed (FR/AC tags accurate, no false blanket claim; SKILL.md nav-table sync added to task list; §4 credits graph-dba's 5th recipe file) | — |
| U4a | `graph-dba` | `ac1c46eb205da2134` | in-flight | freshness-marker implementation (pipeline + skill recipe) | — |
| U4b | `cobb` | — | queued (after U4a) | agent/skill/AGENTS.md wiring implementation + BACKLOG.md M4 section + HISTORY.md entry | — |
| U5 | `analyst` | — | queued (after U4b) | `docs/reviews/cpg-agent-adoption.md` (diff-scoped re-gate section) | code gate |
| U6 | `qa-engineer` | — | queued (after U5) | `docs/test-plans/cpg-agent-adoption.md` + `docs/test-reports/cpg-agent-adoption-report.md` | acceptance |

## Notes

- Uncommitted `falkor-chat/docs/plans/graphrag-eval-coordination.md` (K-026) found in the working
  tree at session start belongs to a **different, prior-session coordination** with its own live
  agent ids (`a4b2370c17130742d` analyst, `af6a040439b6b2515` data-scientist) in flight. Not
  touched by this coordination — flagged to the user, not acted on here.
- Two review gates per team convention: U3 is the plan-level gate (design-level blast radius,
  reconciliation with M2/M3), U5 is the diff-scoped re-gate after implementation.
- `docs/BACKLOG.md` numbering convention: hundreds digit = milestone (C-2xx=M2, C-3xx=M3), so
  this milestone's items are **C-4xx** and belong in a new `## M4 — …` section plus a new
  milestone-map row; U4b is briefed to add both.
