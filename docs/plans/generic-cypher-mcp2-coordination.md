# Generic Cypher MCP — team-wide kaizen inbox rollout — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (`docs/requirements/generic-cypher-mcp2.md`, M7)

## Goal & definition of done

**Superseded 2026-08-20 — read the "Stakeholder decision" section below before trusting anything
in this section; it is kept for history, not as the current target.** Originally: deliver
`docs/requirements/generic-cypher-mcp2.md` (Status: Ready for design, FR-1…FR-14, AC-1…AC-13) —
roll out the graph-backed kaizen working-memory pattern M5 proved on `graph-dba` alone to the
other eleven agents (`analyst`, `architect`, `cobb`, `coder`, `data-scientist`, `devops`,
`frontend-engineer`, `qa-engineer`, `tdd-engineer`, `teco`, `tico`): each gets its own graph-backed
raw-capture layer (same author/curator mechanism as M5, no redesign), a one-time import of its
current `kaizen/inbox.md`, then **deletion of that file** once the import is confirmed;
`kaizen/history.md` stays markdown/unchanged; a new team-wide query surface reaches every migrated
agent's raw learnings in one query (FR-7, new relative to M5); entry structure is locked as a
canonical 5-field contract plus a new session-ID field on new entries (FR-8/FR-8a); every doc
describing the kaizen-inbox convention is updated to match reality; the new-agent creation
convention is updated so a newly created agent is born on the graph-backed pattern; rollout is
explicitly **incremental** (FR-13) — partial migration is valid progress, not a failure state;
`graph-dba`'s own already-frozen `kaizen/inbox.md` (kept in-repo since M5) is also **deleted** as
part of this delivery (FR-14, a recorded supersession of M5's FR-4).

**Current definition of done (post-2026-08-20 stakeholder decision, see below):** the same rollout,
to the same 12 agents, but **no `kaizen/inbox.md` file is ever deleted, for any agent, permanently**
— FR-4/AC-3, FR-12/AC-9 (as originally worded), FR-14/AC-11 are all superseded by that standing
rule (unit `T1`, `docs/plans/generic-cypher-mcp2.md` v3, corrects the requirements doc itself to
match). Consolidation target is **one shared graph, `kaizen_team`, author-partitioned** — not
per-agent `kaizen_<agent>` graphs, which is what a separate, `teco`-uncoordinated `cobb` execution
(commit `ccf9c8b`, 2026-08-20) built in the interim; this delivery now also includes reconciling
that already-shipped state onto the `kaizen_team` design, not just building the pattern fresh.
FR-7 and FR-8a stand unchanged. Whether "never delete" also reaches the now-redundant
`kaizen_<agent>` **graph keys** themselves (not the `inbox.md` files) is still open — see the
Stakeholder decision section.

**Sequencing note (from the M6 coordination doc):** this design work was deliberately queued
behind `cpg-mcp-rename` (M6) so it would be designed and built directly against the final
`cypher`/`mcp__cypher__query` tool identity rather than being touched by that rename mid-flight.
M6 is now fully closed (`docs/plans/cpg-mcp-rename-coordination.md`, archived) — this delivery
starts clean.

## Unit ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| S0 | `graph-dba` | `a55a6de4a73df8b4c` | delivered | `kaizen_team` substrate live: `CREATE INDEX`/`GRAPH.CONSTRAINT CREATE` on `KaizenEntry.entryId`, confirmed `OPERATIONAL` via `db.indexes()`/`db.constraints()`, graph empty (0 entries) at completion — issued via **direct `redis-cli` against `falkordb-dev`**, not `mcp__cypher__query`, because the tool's `authorize_write()` only recognizes two write shapes (a `KaizenEntry` CREATE, or a curator clear) and rejects any DDL outright, by design — the plan's brief wrongly assumed the MCP tool could do this. `graph-dba` logged the finding as a `kaizen_graph_dba` entry. **teco assessment: not a security concern** — same target FalkorDB instance `mcp__cypher__query` itself reaches (host-gateway routing, not a different engine); index/constraint creation is additive, not a `guard-destructive-ops.sh`-matched pattern; direct FalkorDB DDL is `graph-dba`'s normal, authorized operational mode (same as `bootstrap_schema.sh` elsewhere in the repo). No other unit in this plan needs DDL — the 12 `C-<agent>` units are plain `KaizenEntry` `CREATE`s (in-scope for the MCP tool), and `G1`'s `GRAPH.DELETE` was already specified as direct `redis-cli` by the plan itself. Unblocks all 12 `C-<agent>` units. | — |
| S1 | `coder` | `ab116e7dcae39055d` | delivered | retargeted 9 occurrences (`server.py`:4, `README.md`:5) `kaizen_graph_dba`→`kaizen_team`, doc-strings/comments only, zero logic diff. Verified: `grep` zero old-name matches; `test_server_instructions_are_present_and_bounded` green + mutation-tested (tightened bound, confirmed real fail, reverted); host-venv suite 84 passed/7 deselected; `build.sh` rebuilt (new hash `aa088de045e2`, proving content actually changed); in-container gate 75 passed/7 deselected. | — |
| T1 | `tico` | `aa2588c7c9da70047` | delivered | `docs/requirements/generic-cypher-mcp2.md` revised in place, committed by `tico` itself (`1ba3b31`, within its own doc-kind commit authority) — FR-4/FR-14/AC-3/AC-11 marked `Superseded 2026-08-20` (struck-through original text kept), Out-of-scope bullet's now-false rationale corrected, new 2026-08-20 decision-log entry matching the doc's existing pattern, `Last updated` bumped, `Status` unchanged (correction, not a re-interview). FR-12/AC-9 untouched as instructed. Chose in-place over successor doc, reasoning logged in the doc itself. | — |
| S4+S3+S2 | `cobb` | `a08f804f88a241a78` | delivered | **S4**: `audit-team.sh` check 1 narrowed to plan+history; isolated-scratch verification per P3-M1's method (synthetic agent PASSes check 1, FAILs 2/4/5/5b as expected); live 12-agent re-run all PASS check 1. **S3**: `SKILL.md` retargeted to `kaizen_team`, no more inbox-seed-on-creation, §5 template rewritten historical; grep 15 hits, 13 `kaizen_team` + 2 legitimately past-tense, no live prescriptive per-agent pointer survives. **S2**: `claude/README.md`/`claude/AGENTS.md`/root `AGENTS.md`/`docs/BACKLOG.md` catalogs updated, FR-7 query copied verbatim, `docs/BACKLOG.md` gets M7 body section + milestone-map row. Full `audit-team.sh` run: all 12 PASS check 1, overall FAIL only on the pre-existing unrelated `falkor-chat` PII leak (confirmed already-committed, out of scope). No `claude/<agent>/<agent>.md` or `inbox.md` touched (boundary respected — reserved for `C-<agent>` units). | — |
| U2-regate2 | `analyst` | `a9b11279e4ef62c5b` | delivered | `docs/reviews/generic-cypher-mcp2.md`, "Pass 3 — Re-gate of Version 3" — **0 blockers, 3 new majors, 4 minors.** P2-B1/P2-B2 both confirmed fixed on the merits (not just the label). New majors: (1) `S4`'s done-condition (b) proved unachievable by actually running `audit-team.sh` on a synthetic agent in an isolated scratch copy — a stub-only agent directory is never enumerated at all, so the assertion as worded can't pass either way; fix is to scope the check-1 assertion narrowly and name the isolated-tree location. (2) v3's §3 compression dropped two Cypher artifacts other units still cite by reference (the FR-7 query `S2` tells `cobb` to copy, and the `UNWIND`/`CREATE` migration query 12 agents each execute independently) — need restoring. (3) The blanket inbox-header retarget (`kaizen_<agent>` → `kaizen_team`) would falsify **past-tense provenance text** in `analyst`/`teco`/`qa-engineer`/`data-scientist`'s frozen headers ("its 5 entries **were imported into** `kaizen_analyst`" is true history, not a stale pointer) — needs scoping to exclude those 4. Frozen-header carve-out independently confirmed sound (all 12 headers already self-scope their own immutability to exclude the header line). **Verdict: approve with suggestions — reviewer's own recommendation: apply in place, no 4th gate needed**, these are narrow wording/content fixes, not design changes. | plan gate → **approve with suggestions** |
| U2-regate | `analyst` | `a9b11279e4ef62c5b` | delivered | `docs/reviews/generic-cypher-mcp2.md`, "Pass 2 — Re-gate of Version 2" section — 2 blockers, 4 majors, 6 minors, 1 nit. **P2-B1**: FR-12/AC-9 ("no `kaizen/inbox.md` ever created for a *new* agent") still contradicted by `skills/agent-maintenance/SKILL.md`'s creation procedure + §5 Inbox template, both of which still say to seed one — **teco-verified live, confirmed**. **P2-B2**: V2's precedent claim ("cobb self-edited `cobb.md` without incident, so the same applies to `architect`/`graph-dba` self-editing their own prompts") is false — `architect.md`/`graph-dba.md` both carry an explicit "never edit your own agent definition" clause `cobb.md` lacks, and `ccf9c8b` never touched `graph-dba.md` at all — **teco-verified live, confirmed**; fix: `cobb` must own both self-edits, not `architect`/`graph-dba` themselves. 4 majors (AC-8's grep pattern misses 10/12 prompts; the 12 frozen `inbox.md` headers permanently point at the now-wrong `kaizen_<agent>` graph with no fix unit; the 4 dropped FR/ACs aren't routed back to `tico`'s requirements doc or this coordination doc's own "definition of done"; `S3`'s grep-confirmed file list misses 2 FR-12-critical `SKILL.md` lines). §3.6 author-binding claim independently verified by live execution (real `authorize_write()` call). §6 open items answered: no self-edit for architect *or* graph-dba (both route through cobb); make `S0` a hard predecessor of the `C-` units (moots the constraint-idempotency question, closes a duplicate-on-retry hole); G1's gate (a hand-built Cypher count-check) is the real risk, not its batching. | plan gate → **needs changes** |
| U2 | `analyst` | `a7ebedd70af393e90` | delivered | `docs/reviews/generic-cypher-mcp2.md` — 2 blockers, 4 majors, 3 minors. **B1**: `teco`/`tico`/`data-scientist` lack `mcp__cypher__query` in their frontmatter `tools:` allowlist — units A6/A10/A11 as scoped cannot execute (plan's "no agent-wiring step needed" claim is false for these 3). **B2**: AC-11 (delete `graph-dba`'s own frozen `kaizen/inbox.md`) has no owning unit anywhere in §4 — grepped, confirmed absent. **M1**: `claude/scripts/audit-team.sh` check 1 hard-requires `kaizen/inbox.md` per agent (verified by running it live) — this plan breaks team certification tooling team-wide, no unit updates it. **M2**: 5 doc-scoped write-guard hook scripts hardcode `<agent>/kaizen/inbox.md` as an allowed path + name it in escalation text — stale + a latent guard weakening (would silently permit resurrecting a deleted inbox), untouched by any unit. **M3**: `claude/teco/teco.md`'s own coordination-duty prose (2 passages, outside its Learning-capture section) references every delegate's `kaizen/inbox.md` — unit A6 doesn't reach it; same failure shape as M5's own B1 (fixed list under-covering real scope) — analyst notes the plan's own AC-8 grep would literally surface M1/M2/M3 together as unplanned acceptance-time defects. **M4**: second opinion against open item 7 — `cobb` self-editing its own `cobb.md` Learning-capture section is not clearly covered by its existing self-maintenance carve-out; recommends reassigning to a different owner. 3 minors (stale line-count drift in §2's survey; D2's SKILL.md §5 rewrite under-scoped; `cobb` named owner on 13/15 units — a throughput-concentration risk, not a defect). "What's solid" section confirms the core shared-graph mechanism-reuse claim, `kaizen_graph_dba` empty state, and `CLAUDE_CODE_SESSION_ID` all independently re-verified (session ID re-confirmed from a second, cold session). **teco has not yet independently re-verified these findings** — queued for next resume, see Resume note. | plan gate → **needs changes** |
| U1 | `architect` | `aed0cfdde5136e045` | delivered | `docs/plans/generic-cypher-mcp2.md` — decisions: (1) one shared graph `kaizen_team`, `author`-partitioned, not per-agent `kaizen_<agent>` graphs — zero new server code for FR-7 vs. new fan-out logic; (2) `CLAUDE_CODE_SESSION_ID` env var (live-verified in-session) answers FR-8a; (3) 15 independently-dispatchable units — 4 docs/substrate (D1–D3, G0) + 11 per-agent migrations (A1–A11, each bundling data-migration + cobb's paired prompt edit) + 2 acceptance passes (Q1 interim, Q2 closing); (4) no companion `-graph.md` note (FR-8 schema locked, no new modeling beyond §3.1/3.2). `kaizen_graph_dba` confirmed live at 0 entries (cobb already distilled M5's pilot data) — consolidation into `kaizen_team` is schema-only, no data migration. `cypher-mcp/server.py`'s write-auth confirmed graph-name-agnostic by full read — zero logic changes needed anywhere. AC-1…AC-13 all mapped to concrete live/static checks (§5). 8 open items flagged for plan-gate (§6), most notably: shared-graph-vs-per-agent call, retiring the empty `kaizen_graph_dba` key (not literally required by any FR/AC), and whether cobb's self-maintenance carve-out extends to editing its own `cobb.md`. | plan gate (`analyst`) → **needs changes** |
| U1-revision-fix2 | `architect` | `a41b5ee9f9a49ada2` | accepted | Version 4 polish pass folding in Pass 3's 3 majors (S4/Q2 done-condition scoped to check-1 only; restore the FR-7 + `UNWIND`/`CREATE` migration Cypher artifacts dropped by v3's §3 compression; scope the inbox-header retarget to exclude the true past-tense provenance occurrence in 4 files) + 4 minors. Reviewer's own call: no further gate needed after this. | plan gate → **approve with suggestions** (Pass 3, carried forward) |
| U1-revision-fix | `architect` | `a41b5ee9f9a49ada2` | delivered | `docs/plans/generic-cypher-mcp2.md` (Version 3) — both blockers fixed: new `S3` drops the `inbox.md`-seed-on-creation step from `skills/agent-maintenance/SKILL.md` entirely + new `S4` narrows `audit-team.sh` check 1 to `plan.md`+`history.md` only (P2-B1); `C-architect`/`C-graph-dba` prompt edits reassigned to `cobb` per the self-edit prohibition, new §3.7 (P2-B2). All 4 majors + 6 minors + 1 nit adopted (AC-8 grep fixed, frozen-header corrections folded into each `C-<agent>` unit, new `T1` unit — `tico` — supersedes FR-4/AC-3/FR-14/AC-11 in the requirements doc, `S3`'s file list rebuilt from a fresh 14-line grep, `G1`'s gate strengthened count→content-diff, `S0` made a hard predecessor of all 12 `C-<agent>` units). Table now 21 units (added `S4`, `T1`). **Two items deliberately left open, not decided unilaterally:** (1) whether "never delete" reaches the `kaizen_<agent>` graph *keys* themselves, not just the `inbox.md` files — `G1` now requires an explicit stakeholder confirmation as its literal first step before any `GRAPH.DELETE`; (2) this coordination doc's own stale "Goal & definition of done" section, flagged as teco's to fix, not architect's. | plan gate → — |
| U1-revision | `architect` | `a41b5ee9f9a49ada2` | delivered | `docs/plans/generic-cypher-mcp2.md` (Version 2) — confirms V1's `kaizen_team` design as correct, cobb's `ccf9c8b` per-agent execution to be reconciled *to* the plan; independently found FR-4/AC-3 also need dropping (not just FR-14/AC-11 as briefed — the "never delete inbox.md" decision is blanket, not graph-dba-specific); live-verified 5 of 12 `kaizen_<agent>` graphs exist (20 entries total, matches brief); B1 confirmed closed, M1 confirmed moot, M2 cosmetic-only, M3 confirmed still valid (fixed in new unit `C-teco`), M4 resolved by precedent; new §3.6 on `authorize_write()` author-binding (each agent must migrate its own data — `cobb` can't do it for them); fresh 19-unit step table (§4: S0-S3 substrate, 12 per-agent `C-<agent>` consolidation units, G1 graph-key retirement, 2 acceptance passes); §5 maps every FR/AC. Open items flagged for gate: `C-architect`'s self-edit ownership, a carried-over FalkorDB constraint-idempotency gap, G1 batching/risk notes. | plan gate (`analyst`) → — |

## Resume note (updated 2026-08-20 — PLAN GATE CLOSED, ready for implementation dispatch)

**Current state: the plan-gate phase is done.** `docs/plans/generic-cypher-mcp2.md` is at
**Version 4**, gated through 3 rounds (`docs/reviews/generic-cypher-mcp2.md`, Pass 1 needs changes
→ Pass 2 needs changes → Pass 3 **approve with suggestions**, 0 blockers), with Pass 3's 3 majors
+ 4 minors folded into V4 by `architect` and spot-verified directly by `teco` against the live
document (§3.1/§3.6 carry the two previously-dropped Cypher artifacts verbatim; §4.1's `S4` row and
§4.2's header-retarget scoping match the reviewer's fix exactly; no garbled table cells). Per the
reviewer's own explicit recommendation, **no 4th gate round was run** — these were narrow,
mechanical fixes, not design changes. **Nothing has been implemented yet** — §4's 21-unit step
table (`S0`–`S4`, `T1`, 12 `C-<agent>`, `G1`, `Q1`/`Q2`) is fully specified and ready to dispatch,
but zero units have been executed.

**To resume:**
1. Read `docs/plans/generic-cypher-mcp2.md` §4 in full (the step table) — don't dispatch from this
   ledger's summary alone, the plan is the source of truth for exact done-conditions and file
   paths.
2. **Dispatch order, per the plan's own `Depends on` column:** `S0` (graph-dba, `kaizen_team`
   substrate) first — it is a hard predecessor of all 12 `C-<agent>` units. `S1`/`S2`/`T1` have no
   dependencies and can go in parallel with `S0` (different owners, different files: `coder`,
   `cobb`, `tico` respectively). `S4` before `S3` (`S3` depends on `S4`). Once `S0` lands, the 12
   `C-<agent>` units are independently dispatchable in parallel (different files, different graph
   keys, different `agent=` write-authorization identities per §3.6 — no shared-state collision).
   `G1` last, incrementally as each `C-<agent>` unit confirms its migration, per its own row.
   `Q1`/`Q2` acceptance passes close the delivery (§4.4).
3. **Per this repo's own step-table sizing rule**, dispatch per-unit or small independent clusters
   — not one mega-dispatch of all 21. Given the volume (21 units ≈ 15+ agent dispatches, most of
   them `cobb` for the prompt/header halves), consider batching `cobb`'s many prompt-edit halves
   into fewer, larger `cobb` dispatches (it's the same owner touching disjoint files across many
   units — an efficiency case for batching *within one owner*, distinct from the mega-dispatch
   anti-pattern the sizing rule warns against, which is about the coordinator handing one agent an
   entire multi-owner table) while keeping each `C-<agent>`'s *data*-migration half
   (owner = that agent itself, author-bound per §3.6) separately dispatched per agent.
4. Every `C-<agent>` unit's own first step is a **live re-derivation** of what it's migrating — the
   plan's counts (20 entries across 4 graphs, etc.) are 2026-08-20 snapshots, not truth to dispatch
   against blindly; each unit re-checks live per its own done-condition.
5. Independent review gate for the implementation diffs (not just the plan) still applies per this
   repo's standing practice — `Q1`/`Q2` are acceptance passes per §4.4, but a code/prompt-diff
   review gate (`analyst`) before acceptance is this repo's default for a delivery this size; decide
   scope (per-batch vs. one closing gate) at dispatch time.

## Stakeholder decision (2026-08-20) — Option B, corrected to a single shared graph, no inbox deletion ever

**Context.** While this coordination sat paused at U2's `needs changes` gate, the stakeholder
separately redirected `cobb` (outside this coordination, directly) to migrate every agent's
learnings capture to the graph immediately. `cobb` executed that same session (commit `ccf9c8b`):
all 13 target agents now write `:KaizenEntry` nodes to their **own** `kaizen_<agent>` graph
(mirroring `graph-dba`'s pre-existing `kaizen_graph_dba` pattern), not the single shared
`kaizen_team` graph U1's plan above had already recommended. 20 pre-existing entries were migrated
verbatim into 4 of those graphs (`analyst` 5, `data-scientist` 4, `qa-engineer` 6, `teco` 5).
Every agent's `kaizen/inbox.md` is now a frozen historical snapshot — no longer appended to, but
**not deleted**. `data-scientist` and `tico` gained `mcp__cypher__query` in their frontmatter as
part of this. Full detail: `claude/cobb/kaizen/plan.md`'s "Learnings capture redesigned
team-wide (2026-08-20)" entry, and `claude/AGENTS.md`'s updated convention paragraph. (Do not
confuse this with `docs/plans/cpg-agent-adoption2.md` — that is a separate, narrower delivery,
CPG-freshness-check ownership only, unrelated to the graph-migration mechanism.)

**Decision, asked of the stakeholder 2026-08-20 given the above (this coordination's own report):
Option B — keep this coordination alive for what cobb's direct execution doesn't cover, rather
than treat it as fully superseded — with one correction to Option B's own framing:**

1. **Consolidate onto ONE shared graph, not per-agent graphs.** The stakeholder wants all agents'
   learnings in a single graph — i.e. **U1's original `kaizen_team` design is correct and cobb's
   per-agent-graph execution should be brought in line with it**, not the other way around. This
   is real, non-trivial follow-up work: migrate the data already sitting in the 4 populated
   `kaizen_<agent>` graphs (20 entries) into `kaizen_team` (author-partitioned, per U1 §3.1/3.2),
   retire/repoint the 13 `kaizen_<agent>` graph keys, and re-edit every agent's Learning-capture
   prompt section a **second** time (cobb already edited all 13 once, to point at
   `kaizen_<agent>`; they now need to point at `kaizen_team` with an `author` field instead).
2. **No `kaizen/inbox.md` file is ever deleted, for any agent, including `graph-dba`'s own
   already-frozen one.** This is a locked stakeholder decision, explicitly overriding **AC-11**
   and **FR-14** in `docs/requirements/generic-cypher-mcp2.md` (which called for deletion after
   confirmed import) and the plan's original "import then delete" step. Every `kaizen/inbox.md`
   stays in the repo permanently as a frozen historical snapshot — this is now the standing
   convention, not a transitional state. U2's blocker **B2** ("AC-11 has no owning unit") is
   therefore **moot**, not a gap to fix — the AC itself is dropped.
3. **FR-7 (team-wide query surface) and FR-8a (session-ID field) are still wanted and still
   undelivered** — cobb's execution didn't touch either. These remain in scope for the revised
   plan, and FR-7 gets *easier*, not harder, once (1) lands (a single `kaizen_team` graph needs no
   fan-out query at all).

**Resolved 2026-08-20 — the one item V3 left open (`architect`, correctly, declined to decide
unilaterally): "never delete" covers `kaizen/inbox.md` files only, not the `kaizen_<agent>` graph
keys.** Once the 4 populated per-agent graphs' entries are consolidated into `kaizen_team` and
verified, the 12 `kaizen_<agent>` graph keys **are to be deleted** (`GRAPH.DELETE`, `graph-dba`-
owned destructive op, per `G1`'s design in `docs/plans/generic-cypher-mcp2.md` v3). `G1`'s
first-step stakeholder-confirmation gate is now satisfied by this record — `graph-dba` does not
need to re-ask via `tico` at execution time; cite this line instead.

**Dispatched 2026-08-20:** revision brief sent to `architect` (fresh spawn — the U1 agent id
`aed0cfdde5136e045` is from a prior session and did not resolve via `SendMessage`), carrying this
decision plus U2's review (`docs/reviews/generic-cypher-mcp2.md`) re-triaged against it: **B1**
(teco/tico/data-scientist missing `mcp__cypher__query`) is **now closed** by cobb's migration —
confirm and cite, don't re-solve. **B2** is moot per point 2 above. **M1** (`audit-team.sh` check 1
hard-requires `kaizen/inbox.md` per agent) is **now moot** — nothing is ever deleted, so the check
keeps passing by construction; confirm live. **M2** (write-guard hooks hardcode
`kaizen/inbox.md` paths) and **M3** (`teco.md`'s own prose referencing every delegate's
`kaizen/inbox.md`) — re-verify accuracy now that the file persists permanently rather than
disappearing; likely non-issues but not asserted without a fresh check. **M4** (cobb self-editing
`cobb.md`'s own Learning-capture section) — already happened without incident in the live
migration; document as resolved precedent rather than reassigning.

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
