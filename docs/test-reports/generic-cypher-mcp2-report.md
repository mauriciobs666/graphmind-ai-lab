# Generic Cypher MCP — team-wide kaizen inbox rollout — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** C-701…C-721 (M7)

## Summary

Closing acceptance pass (unit `Q2`) for `docs/plans/generic-cypher-mcp2.md` (M7), executed
2026-08-20 against the live working tree at `HEAD` `6a48937` (the M7 delivery itself is still
uncommitted working-tree state, per `git status` — expected, `teco`/`tico` have not yet
committed Wave 2). All twelve test items in
[`docs/test-plans/generic-cypher-mcp2.md`](../test-plans/generic-cypher-mcp2.md) (TP-001…TP-012,
covering AC-1, AC-2, AC-4…AC-10, AC-12, AC-13) were executed live. AC-3 and AC-11 are confirmed
**superseded** (not tested as acceptance criteria — TP-003 confirms the supersession itself,
citing `docs/requirements/generic-cypher-mcp2.md`'s in-place "Superseded 2026-08-20" markers and
the stakeholder decision in `docs/plans/generic-cypher-mcp2-coordination.md`).

**Verdict: PASS with noted open items.** The core mechanism (shared `kaizen_team` graph,
author-partitioning, the two enforced write shapes, `sessionId`, FR-12/AC-9's no-seed convention)
is live, correct, and exercised end to end, including a real `cobb` distillation dispatch against
the *finished, consolidated* graph — a workflow no prior gate in this delivery had proven. This
pass also independently found one **new**, previously-untracked documentation defect (D-1,
`claude/cobb/cobb.md`) not caught by the plan-gate or by any implementation unit's own
self-check, and confirms the two already-known, already-approval-pending data-fidelity defects
and the two deliberately-held-back graph keys are exactly where the brief said they'd be —
consistent with FR-13's incremental-delivery design, not new problems.

`CPG: not applicable — no loaded Joern CPG covers claude/, skills/, or cypher-mcp/ (confirmed
live via a fresh mcp__cypher__query probe this session against a nonexistent graph name, not
just cited from the plan's own §CPG line — the live loaded-graphs list is ws:test,
cpg_falkorchat, reference, ws:qa-tico-workflows-manual, ws:acme, cpg_salesperson, ws:eval,
kaizen_team, kaizen_analyst, kaizen_teco); this delivery is agent-prompt/graph-schema/
documentation work, verified here by driving the running FalkorDB graph and the running MCP tool
directly, not a static code-impact or test-gap question a CPG would usefully answer.`

## Results table

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-001 | AC-1 | **PASS** | Live `mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry {author:'graph-dba'}) RETURN ...")`, no `agent` param, from this (non-`graph-dba`) session — returned `graph-dba`'s entries with full field set, no gating. |
| TP-002 | AC-2 | **PASS** (1 known defect, tracked separately) | `analyst`'s frozen `kaizen/inbox.md` read in full; all 5 original entries (2026-08-11 ×2, -15, -16, -19) matched against `kaizen_team {author:'analyst'}` by date+fact. The 2026-08-16 entry (`fe2007f5-…`) carries the already-known, already-tracked 1-byte evidence-field defect — confirmed still present, not re-diagnosed. |
| TP-003 | AC-3/AC-11 | **PASS** | `docs/requirements/generic-cypher-mcp2.md`: FR-4, AC-3, FR-14, AC-11 all marked `Superseded 2026-08-20`, original text struck through, citing the stakeholder decision. `git status --short`: none of the 12 `claude/*/kaizen/inbox.md` files appear modified or deleted. `ls claude/*/kaizen/inbox.md \| wc -l` → 12. |
| TP-004 | AC-4 | **PASS** | Live write (`agent='qa-engineer'`, `author:'qa-engineer'`, `sessionId` set) accepted; independent second read by `entryId` returned the entry with `sessionId` populated; curator-cleared (`agent='cobb'`) immediately after — zero permanent pollution. First write attempt was blocked by the Claude Code Auto-Mode classifier (harness-level, not the delivery); an identical retry succeeded — noted in Feedback, not a delivery defect. |
| TP-005 | AC-5 | **PASS** | Real `cobb` subagent dispatch, scoped to one raw `graph-dba` entry (`a3f4e1b2-…`) in the **consolidated** `kaizen_team` graph — a workflow proof M5's own AC-5 never covered (that ran against the interim per-agent-graph shape). Independently re-verified: `claude/graph-dba/kaizen/history.md` has the new 2026-08-20 dated entry (read in full, correct What/Why/Verified/Order-of-operations shape); `kaizen_team {author:'graph-dba'}` count confirmed 2→1 by a fresh read; the cleared `entryId` returns 0 rows; the untouched second entry (`d4f8b1c3-…`) confirmed still present. Append-before-delete ordering honored (`history.md` write confirmed before the `DETACH DELETE` call). **Bonus finding, fixed in the same pass, not a residual defect:** the dispatch independently discovered `cypher-mcp/README.md`'s own curator-clear example was broken (`{entryId:'...'}`, no space, fails `_CURATOR_CLEAR_RE`'s literal `entryId: ` requirement at `server.py:265-269`) — confirmed by direct regex read and by two live calls (no-space rejected, space-present succeeded); fixed in the same dispatch (`cypher-mcp/README.md`, docs only, no code change), logged in `claude/cobb/kaizen/history.md`. |
| TP-006 | AC-6 | **PASS** | Both directions rejected by `authorize_write()` with an explicit author/agent-mismatch message, before any write; follow-up counts confirmed zero phantom nodes for both attempted `entryId`s. |
| TP-007 | AC-7 | **PASS** | One unfiltered `MATCH (e:KaizenEntry) RETURN e.author, ...` against `kaizen_team` returned rows spanning all 7 currently-populated authors (`analyst`, `architect`, `data-scientist`, `graph-dba`, `qa-engineer`, `teco`, `tico`) in one query. |
| TP-008 | AC-8 | **FAIL** (1 new defect, D-1 below) | Widened repo grep + full reads of `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, `docs/BACKLOG.md`'s M7 section, `cypher-mcp/server.py`'s live-served doc-string, `cypher-mcp/README.md` — all accurate, `kaizen_team` throughout. All **12** agents' own operative prompts read in full: 11 of 12 are clean (`kaizen_team`, `author`-partitioned, `sessionId`, no stale `inbox.md`-seeding or per-agent-graph reference). `claude/cobb/cobb.md` is the one exception — see D-1. |
| TP-009 | AC-9 | **PASS** (D-1 also touches this AC's spirit — see below) | `skills/agent-maintenance/SKILL.md` §1 read in full: no `inbox.md`-seeding step remains. Independent fresh isolated-scratch re-run (new scratch copy, not `cobb`'s prior one): a synthetic agent with `<name>.md`+`kaizen/{plan,history}.md` and **no** `inbox.md` produced exactly `PASS  synthagent: kaizen plan + history present` on check 1, with the run correctly `FAIL`ing overall on checks 2/4/5/5b (deployment, roster, catalogs, boundary pairs) — reproduces `cobb`'s S4 result independently, not taken on its word. Live, unmodified `claude/`: `bash claude/scripts/audit-team.sh` — all 12 agents `PASS` check 1 specifically; overall run `FAIL`s only on the pre-existing, unrelated, already-committed `falkor-chat` PII-leak check (check 7), confirmed out of scope. Full-prompt scan (TP-008(c)) found `claude/cobb/cobb.md:65` still literally instructs "seed it on creation" — D-1 — a contradiction of AC-9's intent inside `cobb`'s own operative prompt, though `SKILL.md`'s actual Creating procedure (the document `cobb` is told to load for this duty) is correct and does not seed. |
| TP-010 | AC-10 | **PASS** | `Q1`'s ledger row confirmed: real live checks (AC-1/AC-4/AC-6/AC-7) mid-rollout, scoped to then-migrated agents, cross-checked per-author counts against the ledger — genuinely live, not a paper pass. TP-001…TP-007 above confirm the same criteria hold now that all 12 have migrated — the incremental property held throughout. |
| TP-011 | AC-12 | **PASS** | `MATCH (e:KaizenEntry) RETURN DISTINCT keys(e), e.author` — every author's entries carry either the 8-field base set or that set plus `sessionId`; no other variation across 7 authors / 32 entries. |
| TP-012 | AC-13 | **PASS** | `sum(CASE WHEN sessionId IS NOT NULL ...)` → 1 of 32 entries carries `sessionId` (a post-consolidation write), 31 do not (imported/pre-consolidation) — distinguishable exactly as FR-8a specifies (absent, not null-valued). |

**11 of 12 PASS; 1 FAIL (TP-008, on account of D-1 — a single, narrowly-scoped, previously-untracked
defect).**

## Defects

### D-1 (new, found by this pass) — `claude/cobb/cobb.md` carries two stale/contradictory statements about the very convention this delivery ships

**Severity: High.** Not a break in current live behavior (the authoritative sources — `skills/
agent-maintenance/SKILL.md` §1 for agent creation, and `cobb.md`'s own "## Learning capture"
section for `cobb`'s actual write target — are both correct) — but it is a real, reproducible
contradiction inside `cobb`'s own always-loaded operative prompt, on the exact two points
(no-inbox-seeding, `kaizen_team` retarget) this whole delivery exists to land. `cobb.md` is the
one file the plan's own `C-cobb` unit designates as the sole legitimate self-edit (§3.7 of the
plan) — no independent reviewer checked this diff (the resume note flags an independent-review
gate as this repo's standing practice for implementation diffs; no `analyst` review of the "cobb
batch" prompt-edit unit appears in the coordination ledger), which is plausibly how this survived.

**Steps to reproduce:**
1. `Read claude/cobb/cobb.md`, lines 61–73 ("Maintenance duties — kaizen & documentation").
2. Observe line 65: *"Every artifact you touch carries a living `kaizen/{plan,history}.md`
   (agents additionally a learnings `inbox.md` — **seed it on creation**)."*
3. Compare against `skills/agent-maintenance/SKILL.md:62`: *"**No `inbox.md` is seeded for a new
   agent** (FR-12/AC-9, ...)"* — a direct, literal contradiction between the two documents
   `cobb` operates from for the same duty.
4. Observe line 71: *"...captures run-time environment discoveries as raw, dated, evidence-backed
   entries directly into its own working-memory FalkorDB graph, `kaizen_<agent>`, ... a pattern
   piloted on `graph-dba` and migrated team-wide 2026-08-20 ... clear it — a curator-scoped
   `DETACH DELETE` through `mcp__cypher__query` (`agent='cobb'`) against the agent's own
   `kaizen_<agent>` graph."* — present tense, describing the **interim** per-agent-graph shape
   (`ccf9c8b`) that this very delivery superseded with the consolidated `kaizen_team` graph. Ten
   of the twelve `kaizen_<agent>` keys this sentence points at no longer exist (`G1`, retired).
5. Compare against the same file's own lines 84–98 ("## Learning capture"), which correctly
   targets `kaizen_team`, `author`-partitioned, with `sessionId` — i.e. `cobb.md` disagrees with
   itself between its "Maintenance duties" section and its "Learning capture" section.
6. Confirmed by targeted grep: `grep -n "seed" claude/*/*.md skills/agent-maintenance/SKILL.md`
   returns exactly one live-instruction hit for "seed it on creation" — `claude/cobb/cobb.md:65`
   — none of the other 11 agent prompts carry an equivalent bullet.

**Expected:** `cobb.md`'s "Maintenance duties" section describes the actual, current convention
(no inbox seeding; `kaizen_team`, author-partitioned, as the capture/curator-clear target for
every agent including `cobb`) — consistent with `SKILL.md` and with `cobb.md`'s own "Learning
capture" section.

**Actual:** Two sentences in `cobb.md`'s "Maintenance duties" section describe the pre-M7
convention (inbox-seeding, per-agent `kaizen_<agent>` graphs), unchanged by the `C-cobb` unit's
self-edit, which only touched the "Learning capture" section (lines 84–98).

**Evidence:** direct file reads, `grep -n "seed" claude/*/*.md skills/agent-maintenance/SKILL.md`,
`grep -rnE 'kaizen_[A-Za-z{<][A-Za-z_{}<>-]*' claude/ skills/ cypher-mcp/ docs/BACKLOG.md
docs/requirements/generic-cypher-mcp2.md AGENTS.md` triaged by hand (TP-008).

**Recommended fix** (not applied — QA does not edit the code/prompt under test): a targeted,
`cobb`-self-edited 2-sentence fix to lines 65 and 71, matching the plan's own §3.3 recipe and
FR-12/AC-9's "no `inbox.md` for a new agent" wording, sized similarly to the already-landed
"Learning capture" section rewrite. Given the self-edit carve-out, `cobb` is the only agent
authorized to make this edit directly; an independent `analyst` read of the diff afterward would
close the coverage gap that let this survive the original unit.

### Already-known, already-tracked (not new findings — re-confirmed present, per the brief)

- **`e40a95fe-…` (`author='teco'`, in `kaizen_team`)** — 2 fields lost backtick-wrapping.
  Confirmed still present, unresolved, blocked on a `cobb` curator-clear pending stakeholder
  approval for that MCP write. Not re-diagnosed here.
- **`fe2007f5-…` (`author='analyst'`, in `kaizen_team`)** — 1-byte newline-escaping slip in
  `evidence`. Confirmed still present (independently re-surfaced by TP-002's own spot-check, same
  entry), unresolved, same blocked-fix path as above.
- **`kaizen_teco` and `kaizen_analyst`** — confirmed still live (the live loaded-graphs probe,
  §CPG line above, lists exactly `kaizen_team`, `kaizen_analyst`, `kaizen_teco` among `kaizen_*`
  keys) — deliberately not yet retired by `G1`, since their source text is the input to the two
  fixes above. All other 10 `kaizen_<agent>` keys confirmed retired (not merely claimed retired).

## Coverage & gaps

**Covered:** the shared-graph mechanism end to end (read, author-write, curator-clear, rejection,
team-wide query, `sessionId`), the migration's data fidelity (spot-checked, consistent with every
`C-<agent>` unit's own byte-exact self-check in the ledger), the certification tooling's actual
live behavior (not just its source) in both the live tree and a fresh isolated copy, every one of
the 12 agents' own operative prompts read in full (not just the 4 catalog docs the plan's AC-8
mapping names), and — the one workflow no prior gate proved — a real `cobb` distillation pass
against the finished, consolidated graph.

**Gaps, deliberate (see test plan §1.1/§9):** no re-diff of all 4 populated agents' migrations
(one spot-check, `analyst`, done instead — the other 3 already have their own per-unit
content-diff in the ledger); no re-run of `cypher-mcp`'s offline/in-container suites (`S1`'s
scope, already green); no fuzzing of `authorize_write()`'s regex beyond AC-6's requirement-level
directions; no load/concurrency/security testing (no real risk at this scale/trust model).

**Residual risk:** D-1 is the only genuinely new, unaddressed item this pass leaves behind, and
it's narrow (one file, two sentences, no code) and non-blocking in practice (the authoritative
sources are correct) — but it is the exact kind of drift that, left long enough, could mislead a
future `cobb` dispatch or a future agent-creation pass. The two known data-fidelity defects and
the two held-back graph keys are exactly the state the brief described — real, tracked, awaiting
a stakeholder approval this pass has no authority over, and (per FR-13) not evidence the
delivery is unfinished.

## Minor / housekeeping notes (not filed as defects)

- **`kaizen_team {author:'qa-engineer'}` carries one uncleaned residual test entry**
  (`cda51378-…`, from `Q1`'s own AC-4 exercise) — self-labeled "safe to skip in distillation," 8
  entries live where the ledger's snapshot said 7. This pass's own TP-004 write was
  self-cleaned (curator-cleared in the same test item) to avoid adding a second one; recommend
  `cobb` sweep `cda51378-…` on its next `qa-engineer` distillation pass.
- **`docs/BACKLOG.md`'s M7 section (C-701…C-721) and the M7 milestone-map row both still carry
  🔵 (proposed), not ✅ (done)** — unlike every closed milestone M1–M6. This is expected
  sequencing (the flip normally follows a closing acceptance verdict, which this report now
  supplies) rather than a defect; recommend `teco`/`cobb` flip both to ✅ once this report's
  **PASS with noted open items** verdict is accepted.
- **The Claude Code Auto-Mode classifier transiently blocked one legitimate, in-shape MCP write**
  (TP-004's first attempt) before any FalkorDB call — an identical retry succeeded immediately.
  Harness-level friction, not a `cypher-mcp` or `kaizen_team` defect; noted for awareness, not
  actionable by this delivery.

## Feedback & recommendations

1. **Fix D-1** — a `cobb` self-edit of `cobb.md:65,71`, then an independent `analyst` (or
   `qa-engineer`) read of the diff, closing the review-coverage gap that let a self-edited file
   ship with an internal contradiction unnoticed.
2. **Extend the independent-review-gate practice to self-edited files specifically** — this
   repo's standing practice (resume note item 5) already calls for a diff review gate on
   implementation deliverables; `cobb.md`'s self-edit carve-out (§3.7 of the plan, a deliberate
   and correct design decision on its own terms) means `cobb` is both author and sole editor of
   that one file, which is exactly the shape most likely to let a partial edit (only one of two
   affected sections touched) go unnoticed. A cheap, generalizable mitigation: whenever a
   self-edit unit closes, route a one-line "did every section I was supposed to touch actually
   change?" grep-diff check to a second agent — cheaper than a full review, would have caught D-1.
3. **`cobb.md:65`'s general "Kaizen" bullet is a good candidate for the same simplification the
   two `-graph`/`-ml` document kinds already model** — pointing at `SKILL.md`'s Creating
   procedure by reference rather than restating (and now, for the second time, dating) the
   inbox-seeding detail inline would have made this specific defect structurally impossible.
4. **The AC-8 grep pattern (`kaizen_[A-Za-z{<][A-Za-z_{}<>-]*`) also matches `kaizen_team` itself**
   — every run of it needs a manual `kaizen_team` exclusion pass before triage, which this report
   did but which cost real reviewer attention across ~70 raw hits. A small, durable improvement:
   anchor the pattern to exclude the literal `team` alternative
   (`kaizen_(?!team\b)[A-Za-z{<][A-Za-z_{}<>-]*`) so future re-runs of this exact check (a
   plausible recurring audit, given this pattern already survived 3 plan-gate revisions) start
   from a cleaner signal.
5. **`cypher-mcp/README.md`'s curator-clear example bug (found and fixed during TP-005) is a
   real, if small, catch worth calling out positively** — a copy-pasteable example that itself
   fails against the tool's own regex is a genuine footgun for the next agent who trusts it
   verbatim; good that this pass's real distillation exercise surfaced it rather than a synthetic
   one. Separately worth a low-priority follow-up: whether `_CURATOR_CLEAR_RE` itself should
   tolerate the no-space form (`server.py:265-269`), rather than requiring documentation to warn
   around it — `cobb` flagged this as an open question in its own `history.md` entry; not this
   pass's call to resolve.

---

**Artifacts:** test plan
[`docs/test-plans/generic-cypher-mcp2.md`](../test-plans/generic-cypher-mcp2.md) · this report
[`docs/test-reports/generic-cypher-mcp2-report.md`](./generic-cypher-mcp2-report.md).
