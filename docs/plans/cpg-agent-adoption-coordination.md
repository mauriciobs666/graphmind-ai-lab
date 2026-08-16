# CPG agent adoption — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** cpg-agent-adoption (M4, proposed — C-4xx TBD in `docs/BACKLOG.md`)

> **PAUSED 2026-08-16 (user: "pause when possible, out of credits").** Stopped dispatching new
> work immediately. U6 (`qa-engineer`, agentId `a4e1114544491d778`) was already in-flight at pause
> time — not cancelled (no cancel mechanism used), so it may complete and notify after this
> session ends. **Whoever resumes: check U6's result first** (its agentId still resolves via
> `SendMessage` if it hasn't already notified) before re-dispatching anything for U6.
>
> **State at pause: everything through U5 is delivered, independently verified, and committed**
> (`35b108f`: U1-U4a; `50f9aaa`: U4b-1..5; `c84815c`: U5 review). Nothing uncommitted except this
> ledger edit itself. U5's diff-scoped code gate came back **approve**, zero blockers/majors/
> minors. Only U6 (acceptance pass against AC-1…AC-6) remains before this coordination closes —
> its own deliverables (`docs/test-plans/cpg-agent-adoption.md` +
> `docs/test-reports/cpg-agent-adoption-report.md`) are not yet written.
>
> **To resume:** read this ledger top to bottom, reconcile against `git status`/`git log` (should
> match exactly — nothing changed since `c84815c`, unless U6 notified and its deliverable files
> exist on disk uncommitted), check whether U6's agentId still resolves, then: independently spot-
> check U6's test plan/report against the actual observed behavior it claims (per standing
> practice — never accept a summarized verdict on its word alone), update the ledger, commit U6's
> verified deliverables, and close the coordination (flip this document's own `Status:` to
> `archived` per root `AGENTS.md`'s doc-lifecycle table — `teco` is the owner of its own
> coordination doc's archival flip).

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
| U4a | `graph-dba` | `ac1c46eb205da2134` | delivered | `skills/joern-cpg/scripts/pipeline.sh` (+18 lines, stamping step), `skills/cpg-analysis/references/freshness.md` (new), `skills/cpg-analysis/SKILL.md` (+1 nav row) — teco spot-checked `git diff`: exactly the 3 items, no scope creep; `bash -n` clean; Cypher parse-verified live via `GRAPH.RO_QUERY` rejection message; freshness query against live `cpg_falkorchat` returns 0 rows as predicted. Committed `35b108f`. | — |
| U4b-1 | `cobb` | ~~`a2bfa37fedd4fa6fb`~~ → `a368106cb10410e95` | delivered | `skills/cpg-analysis/SKILL.md` — teco spot-checked `git diff`: §1 discovery paragraph (cpg_<component> guess + freshness-as-probe + fallback order) and §4 nav-table impact-analysis row (gained coder/tdd-engineer/frontend-engineer, other rows untouched) both correct; description 983 chars (under 1024 budget); YAML frontmatter parses clean | — |
| U4b-2 | `cobb` | ~~`a1f1a9e022791aec2`~~ → `a011bdf9f9bdbd53b` | delivered | `claude/{analyst,architect,qa-engineer}/*.md` reworded to default-orientation framing + freshness-check bundling + `CPG:` evidence-trail line in each deliverable skeleton, `kaizen/history.md` dated entries in all three — teco spot-checked `git diff`: content correct, consistent phrasing across all three, no scope creep; `agent-maintenance` lint + `audit-team.sh` reported clean | — |
| U4b-3 | `cobb` | ~~`a3f85c0b85c0f834e`~~ → `ae209e01bed194ddc` | delivered | `claude/{coder,tdd-engineer}/*.md` new-consumer wiring (Orient/Understand-first step + evidence-trail line) + kaizen entries — teco spot-checked `git diff`: after one same-delegate fix-and-close-out round (description was left in old conditional framing on first pass; corrected to the default-orientation pattern all other five agents use), both files' descriptions now verbatim-match `analyst`/`architect`/`qa-engineer`'s opening clause, body/evidence-trail content correct, kaizen entries updated with a same-day addendum noting the correction | — |
| U4b-4 | `cobb` | ~~`a2e79f6e358cd6e14`~~ → `a1c8fd7429130a580` | delivered | `claude/frontend-engineer/frontend-engineer.md` (Orient-first item 4, grounded in `cpg_salesperson`/`chatbot.py`; evidence-trail line in Step 4) + `kaizen/history.md` dated entry — teco spot-checked `git diff`: content correct, no scope creep, description/body consistent | — |
| U4b-5 | `cobb` | `a165aa58317311e30` | delivered | `claude/README.md` (6 rows reworded to default-orientation framing, verified verbatim against each agent's landed frontmatter/body), `skills/README.md` (`cpg-analysis` row widened to 6 consumers), `docs/BACKLOG.md` (new M4 section, C-401…C-407, milestone-map row marked 🟡 pending U5/U6 — matches M2/M3 precedent), `docs/HISTORY.md` (dated M4 entry, explicitly notes U5/U6 still queued) — teco spot-checked all four diffs, correct and honest about gate state. Root `AGENTS.md` deliberately left unchanged (`git diff` confirms zero touch) — its `skills/` bullet was already consumer-agnostic, no stale claim to fix. | — |
| U5 | `analyst` | `a89295ef21d32b51d` | accepted | `docs/reviews/cpg-agent-adoption.md` §"Pass 2 — Diff-scoped code gate (U5)" — teco spot-checked: SKILL.md description independently recounted at 983 chars (matches); scope/FR-8/consistency claims read as thoroughly grounded (live `mcp__cpg__query`, direct full-file reads, `audit-team.sh`, `bash -n`). One nit noted (pre-existing YAML strict-parse quirk on `tdd-engineer.md`/`frontend-engineer.md`, confirmed pre-diff, not actionable). `audit-team.sh`'s overall FAIL traced to an unrelated K-026 file untouched by either commit — correctly out of scope here. | code gate: **approve** |
| U6 | `qa-engineer` | `a4e1114544491d778` | in-flight | `docs/test-plans/cpg-agent-adoption.md` + `docs/test-reports/cpg-agent-adoption-report.md` | acceptance |

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
- **U4b-1..4 original dispatch (agent ids struck through above) all failed with an account/session
  usage-limit error** ("You've hit your session limit... progress saved"), not a content problem —
  a transient platform failure per standing practice, not a deficient result. Before re-dispatching,
  `git diff` confirmed each had landed exactly one small partial edit and nothing else: SKILL.md's
  frontmatter `description` widened to six consumers; `analyst.md`/`coder.md`/`frontend-engineer.md`
  each got their `description` clause only (no body-prompt section, no evidence-trail line, no
  kaizen entry on any of the six target agent files); `architect.md`/`qa-engineer.md`/
  `tdd-engineer.md` were untouched. No damage, nothing to roll back. Re-dispatched fresh (not
  resumed via SendMessage — session-limit kills call for a fresh agent with a state-recovery brief
  per standing practice) with each brief stating precisely what's already on disk vs. what
  `docs/plans/cpg-agent-adoption.md` §2.4/§3/§6 still specifies.
- U4b-3's first pass left `coder`/`tdd-engineer`'s `description` in the old conditional framing
  ("With a loaded Joern CPG, uses...") instead of the default-orientation framing plan §2.1
  mandates for all six agents — closed via a same-delegate follow-up (resumed via SendMessage,
  not a fresh dispatch); both now verbatim-match `analyst`/`architect`/`qa-engineer`'s opening
  clause. Verified via `git diff`.
- **Minor, non-blocking phrasing variance flagged for U5's awareness, not treated as a defect:**
  five of the six wired agents' `description` clauses open with the identical sentence "Checks
  whether a relevant CPG exists as part of its normal orientation and, when one does, uses…";
  `frontend-engineer`'s reads "Checks for a relevant Joern CPG (`cpg_salesperson` today) as part
  of that orientation and uses…" — same default-orientation semantics (not the old conditional
  framing), just not verbatim-identical wording, and arguably clearer for naming the concrete
  graph. Plan §2.1/§2.4 requires the default-orientation *reframing*, not identical prose across
  agents.
