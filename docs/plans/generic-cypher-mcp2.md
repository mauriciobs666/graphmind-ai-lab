# Generic Cypher MCP — team-wide kaizen inbox rollout — implementation plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** — (M7) · **Version:** 5

Design for [`../requirements/generic-cypher-mcp2.md`](../requirements/generic-cypher-mcp2.md)
(FR-1…FR-14, AC-1…AC-13), per
[`generic-cypher-mcp2-coordination.md`](./generic-cypher-mcp2-coordination.md) unit U1/U1-revision.
Builds on, and does not redesign, the M5 mechanism —
[`generic-cypher-mcp.md`](./generic-cypher-mcp.md) (tool mechanism: the optional `agent`
parameter, the two enforced write shapes) and [`generic-cypher-mcp-graph.md`](./generic-cypher-mcp-graph.md)
(the `:KaizenEntry` schema, `author` as a plain property, curator-clear semantics) — both taken as
given and cited by path, not re-derived. Written directly against the post-M6 tool identity
(`cypher-mcp/`, `mcp__cypher__query`) — no dual-naming transition needed.

**CPG:** not applicable — this delivery is agent-prompt/graph-schema/documentation design work;
independently re-confirmed again in this revision (`mcp__cypher__query(graph='nonexistent_probe2',
...)` → the live loaded-graphs list contains no CPG relevant to `claude/`, `skills/`, or
`cypher-mcp/`; only `cpg_falkorchat`/`cpg_salesperson`, unrelated application codebases).

---

## Revision note — 2026-08-20 (Version 5)

**Not a plan-gate revision — a correction during Wave-2 implementation dispatch**, routed to
`architect` by `teco` because this plan is its document and outside `teco`'s own write guard. Per
the reviewer's own Pass-3 recommendation (Version 4's note below), this document dispatches without
a further gate unless something new surfaces at design level; this correction is a scope reduction
forced by real-world permission enforcement, not a design change, so no fourth gate is opened. Full
record: `docs/plans/generic-cypher-mcp2-coordination.md`'s ledger row `` `cobb` batch
(prompt+header, all 12 + self-migration) `` — cited here, not re-derived.

**The header-retarget half of §4.2/P3-M3 is dropped, permanently, by stakeholder decision — real
permission-system enforcement, not a design choice this plan gets to keep making.** During Wave-2
dispatch, `cobb` attempted the per-agent `kaizen/inbox.md` header retarget that this plan's §4.2
(step 3 of the per-`C-<X>` unit description) and P3-M3 (Version 4's note below) authorize — on the
theory, stated explicitly in both places, that the header note sits outside each file's own
*"Content below is preserved for historical reference and will not change"* immutability promise.
The actual Claude Code permission system **denied 3 of the 12 edits** (`analyst`, `data-scientist`,
`qa-engineer`) with the reason **"this is frozen"** — a real, human-level enforcement signal, not a
misconfiguration `cobb` could route around. Asked to choose between (a) proceeding with all 12
header retargets anyway, or (b) dropping the header-retarget half entirely and reverting the one
file already changed (`teco`'s), **the stakeholder chose (b).** `cobb` reverted `teco`'s header edit
via `git checkout` and confirmed via `git diff`/`git status` that all 12 `claude/*/kaizen/inbox.md`
files are byte-identical to `HEAD` — the header-retarget half of every `C-<agent>` unit is fully
dropped, not deferred.

Effects on this document, recorded here as a narrative correction rather than edited into the step
table (the table cells stay exactly as gated, as the historical record of what Version 4 designed
and `analyst` approved — this note is the pointer a later reader follows to know what actually
happened):

1. **§4.2's per-agent header-retarget instruction** (the second half of step 3: *"...and retargets
   `claude/<X>/kaizen/inbox.md`'s header note..."*) **and P3-M3's "Content below" scoping
   rationale** (Version 4's revision note below, plus the "header retarget is scoped narrowly"
   paragraph immediately under the §4.2 table) **are superseded by this 2026-08-20 stakeholder
   decision.** Real-world permission enforcement holds every `kaizen/inbox.md` fully frozen, header
   note included — contradicting this plan's own reading that the header sat outside the file's
   immutability promise. The textual analysis was a reasonable reading of the header's own wording;
   the enforcement layer disagreed, and the stakeholder sided with enforcement over the plan's
   reading.
2. **Every `C-<agent>` row's done-condition in §4.2's table that references a header retarget — all
   12 rows' `Files` column and `Done-condition highlights` column — is now read as N/A — dropped,
   not achieved.** This also covers the paragraph directly below the §4.2 table describing the
   4-file (`analyst`/`teco`/`qa-engineer`/`data-scientist`) provenance-clause scoping exercise —
   that scoping is moot now that no header is touched at all, for any of the 12. Nothing in §4.2's
   table itself is edited by this note; only its header-retarget content is to be read as
   not-achieved going forward.
3. **Secondary, independent correction — factual, not a scope change.** The `C-graph-dba` row's
   parenthetical, *"no entries, so no provenance clause to protect,"* is wrong on a direct read of
   the file: `graph-dba`'s `claude/graph-dba/kaizen/inbox.md` header does carry a genuine past-tense
   provenance clause, dated **2026-08-18** — predating the general 2026-08-20 cross-team migration
   this plan's `C-<agent>` units cover. The row conflated two different facts about two different
   artifacts: "0 entries in `kaizen_graph_dba`" (true — and the correct reason cited for that row's
   *data*-migration side being a no-op) with "no provenance clause in the header" (false — the
   header's provenance clause predates and is unrelated to the `kaizen_graph_dba` entry count). Moot
   for actual editing now that no header is ever touched, but the row's stated premise was incorrect
   and is corrected here for anyone reading the plan later.

**Version bumped 4→5** because this note materially changes how 12 of §4.2's table rows (plus the
scoping paragraph beneath it) are to be read — not a typo fix, and consistent with this document's
own established pattern of pairing a dated revision note with a `Version:` bump (V2, V3, V4 below).
**No other content in this document changes** — §3.7, §4.1, §4.3, §4.4, §5, and §6 still describe
the plan exactly as Version 4 gated it; only the header-retarget half of §4.2 (table cells left
intact as historical record) and its P3-M3 rationale are affected.

---

## Revision note — 2026-08-20 (Version 4)

**Version 3 was plan-gated a third time** (`docs/reviews/generic-cypher-mcp2.md`, "Pass 3 —
Re-gate of Version 3"). Verdict: **approve with suggestions** — 0 blockers, both Pass-2 blockers
(P2-B1, P2-B2) confirmed genuinely closed on the merits, 3 new majors, 4 minors. The reviewer's
own explicit recommendation was to apply the majors in place and dispatch without a fourth gate
("these are one-or-two-sentence done-condition edits — none touches the design, the unit set, the
ownership model, or any FR/AC mapping"). This revision does exactly that — all 3 majors and all 4
minors are adopted; nothing is deferred.

- **P3-M1 (`S4`'s done-condition (b) unachievable)** — the reviewer executed
  `claude/scripts/audit-team.sh` against a synthetic agent in an **isolated scratch copy** of the
  tree (never the live `claude/`) and found: without a `<name>.md`, the directory is never
  enumerated at all (`FAIL no agents found`); with a stub `<name>.md` added so it *is* enumerated,
  check 1 now passes but checks 2/4/5/5b (deployment, roster, catalogs) still fail by construction
  for any synthetic agent, so the overall run is `FAIL` either way — a done-condition asking for a
  clean full-script pass cannot be satisfied for reasons entirely outside this delivery's scope.
  Fixed by scoping `S4`'s (and `Q2`'s) assertion narrowly to **check 1's own logic**, run in an
  isolated tree, not a full-script pass (§4.1, §4.4 below).
- **P3-M2 (§3's compression dropped two Cypher artifacts other units cite by pointer, plus one false
  claim about the document's own layout)** — restored the FR-7 team-wide query verbatim in §3.1 and
  the `UNWIND`/`CREATE` migration-query shape verbatim in a restored §3.6 subsection (both were
  present in Version 2, dropped when §3 was compressed to avoid a third full reproduction — the
  *rationale* was safe to compress, the *artifacts* other units point to by reference were not).
  The false "V2 note preserved above the V3 note" claim (there is no V2 note in this document — it
  only exists in git history) is corrected to a commit-pinned pointer.
- **P3-M3 (blanket inbox-header retarget would falsify true history in 4 files)** — `analyst`,
  `teco`, `qa-engineer`, `data-scientist`'s frozen headers each carry **two** `kaizen_<agent>`
  occurrences in opposite tenses: a true past-tense provenance clause ("were imported into the
  `kaizen_<agent>` graph") and a prescriptive pointer ("new raw learnings are written directly
  into… `mcp__cypher__query(graph='kaizen_<agent>', …)`"). A blanket substitution would rewrite the
  first into a false historical claim. §4.2's instruction is now scoped to retarget **only the
  prescriptive clause** in those 4 files, leaving the provenance clause untouched; the other 8
  agents' headers carry only the prescriptive form and retarget in full, as before. Also adopted
  the reviewer's stronger justification for the frozen-header carve-out itself: every one of the 12
  headers already scopes its own immutability promise to *"**Content below** is preserved for
  historical reference and will not change"* — the header note was never inside that promise to
  begin with, which settles the "frozen ≠ read-only for this edit" question more cleanly than V3's
  own "accuracy edit, not new content" framing.
- **P3-m1** — `G1`'s first step and §6 open item 1 were asking `graph-dba` to re-confirm a question
  that is already answered: `docs/plans/generic-cypher-mcp2-coordination.md`'s "Resolved
  2026-08-20" paragraph (lines 126-132, read directly for this revision) states plainly that the
  never-delete decision covers `kaizen/inbox.md` files only, the 12 (now up to 13, per the live
  count — see below) `kaizen_<agent>` graph keys **are** to be deleted, and that this record itself
  discharges `G1`'s confirmation gate — *"`graph-dba` does not need to re-ask via `tico` at
  execution time; cite this line instead."* `G1` now cites it instead of instructing a re-ask; §6
  item 1 is closed, not carried forward as open.
- **P3-m2** — AC-8's widened-root grep (64 files) has no bucket for the 17 arbitrary fixture-name
  occurrences in `cypher-mcp/tests/test_server.py` under its two-way historical/gap classification.
  Added a third bucket and pre-classified that file as out of scope; narrowed the `docs/` root to
  the two files any unit actually touches (`docs/BACKLOG.md`, `docs/requirements/generic-cypher-mcp2.md`)
  to keep the check signal-dense.
- **P3-m3** — `S3` and `S4` both need to land for the FR-12 asymmetry to hold at every point in
  time (`S3` alone, mid-window, would reintroduce Pass 1's M1 finding for any agent created in that
  gap). `S3` now depends on `S4`.
- **P3-m4** — §4.2's table lost the two-actor annotation on 5 of 12 rows and had no `Depends on`
  column once `S0` became a hard predecessor. All 12 rows now annotate the actor split uniformly,
  and a `Depends on` column states `S0` for every row.

---

## Revision note — 2026-08-20 (Version 3)

**Version 2 went through `analyst`'s plan-gate a second time** (`docs/reviews/generic-cypher-mcp2.md`,
"Pass 2 — Re-gate of Version 2"). Verdict: **needs changes** — 2 blockers, 4 majors, 6 minors, 1
nit. `teco` independently spot-checked both blockers before dispatching this fix; they hold. This
revision addresses every finding below, re-verifying each live rather than taking the review (or
`teco`'s dispatch message) on its word.

**P2-B1 (FR-12/AC-9 still unmet) — resolved by explicit direction from `teco`'s dispatch, not by
architect's own pick between the review's two branches.** The review posed this as a genuine
stakeholder-level fork: (a) amend FR-12/AC-9 to accept the permanent frozen `inbox.md` stub as a
now-structural invariant, or (b) deliver FR-12 as written — no `inbox.md` for a new agent, ever —
and update `claude/scripts/audit-team.sh` to hold that alongside the separate, already-settled
"never delete an *existing* one" rule. `teco`'s dispatch message states the asymmetry to hold
directly: *"existing agents' `inbox.md` files are never deleted (permanent), but new agents from
here on never get one created in the first place. Both are true at once."* That is branch (b),
stated as a directive, not offered as a choice — this revision implements it as such, attributed
here rather than silently invented. Concretely: `S3` now removes the `inbox.md`-seeding step from
`skills/agent-maintenance/SKILL.md` entirely (§1's Creating procedure + §5's "Inbox template"
block, both re-read in full for this revision); a new unit `S4` updates
`claude/scripts/audit-team.sh` check 1 so it stops hard-requiring `inbox.md` for *any* agent
(existing or new) while still requiring `plan.md`/`history.md` — existing agents keep passing
because their frozen file is still physically present (nothing deletes it), a new agent passes
because the check no longer looks for one at all. This is a strict simplification of Pass 1's own
suggested fix ("accept either the file-triple or a live `kaizen_team` presence check") — dropping
the `inbox.md` requirement outright avoids giving a fast, deterministic, offline script a live
FalkorDB dependency.

**P2-B2 (`C-graph-dba`/`C-architect` self-edit) — the review's fix is adopted outright, no
counter-argument.** Independently re-read `claude/architect/architect.md:67` and
`claude/graph-dba/graph-dba.md:87` in this revision: both end their Learning-capture section with
the literal clause *"never edit your own agent definition"* — `grep -rln "never edit your own
agent definition" claude/` returns all 11 non-`cobb` agent prompts, `cobb.md` is the sole
exception, confirming the carve-out is `cobb`-specific by construction, not a coincidence of who
self-edited last. And `git show --stat ccf9c8b` (re-run in this revision) still does not list
`claude/graph-dba/graph-dba.md` among its 43 files — no precedent exists for `graph-dba` self-editing
either. Both `C-architect`'s and `C-graph-dba`'s prompt-edit halves now route through `cobb`,
identical to the other nine non-`cobb` agents — `cobb` self-edits only `cobb.md`, the one file
whose prompt genuinely omits the prohibition. §6 open item 1 (self-edit ownership) is answered:
**no**, neither self-edits.

**The four majors, re-verified and fixed:**
- **P2-M1 (AC-8's grep pattern)** — `kaizen_<agent>` is a template placeholder that appears in no
  actual prompt; run verbatim it missed 10 of 12 agent files. Replaced with the review's
  independently-tested pattern, §5.2 below, widened to search `docs/` and root `AGENTS.md` too
  (roots `S2` already touches, outside the old command's scope).
- **P2-M2 (the 12 frozen `kaizen/inbox.md` header notes)** — each names its now-superseded
  `kaizen_<agent>` graph prescriptively, as a copy-pasteable live query, not as past-tense history —
  squarely inside FR-11/AC-8's "no contradiction" bar. Folded into every `C-<agent>` unit's Files
  column: the header note gets the same `kaizen_team` retarget as the prompt file, landed by the
  same actor (`cobb`) in the same unit. **This is editing an already-`FROZEN`-labeled file's header
  note for factual accuracy, not writing new content to it** — a distinction this revision states
  explicitly per `teco`'s ask, because V2 treated "frozen" as strictly read-only everywhere else and
  this is the one deliberate exception, scoped narrowly to the graph-name pointer.
- **P2-M3 (two other stale documents, no unit reaches either)** — `docs/requirements/generic-cypher-mcp2.md`
  still states FR-4/AC-3/FR-14/AC-11 in their original, now-superseded form, and its Out-of-scope
  section still argues the git-history rationale the stakeholder has since reversed; a new unit
  **`T1`, owner `tico`**, fixes it (§4.1 below) — FR-12/AC-9 need **no** requirements-doc edit,
  since branch (b) above delivers them as written, not supersedes them.
  `docs/plans/generic-cypher-mcp2-coordination.md`'s own "Goal & definition of done" section is
  the *second* stale document the review found and is explicitly **not** this plan's unit to own
  (`architect`'s plan doesn't govern `teco`'s coordination doc) — flagged directly to `teco` in
  this revision's final report, not silently dropped, not silently added as a unit here.
- **P2-M4 (`S3`'s occurrence list was itself incomplete)** — missed lines 444–445 (the `kaizen_{name}`
  curly-brace spelling inside the seeded template, exactly the FR-12-critical lines) because the
  prior enumeration was built from angle-bracket forms only. Re-grepped fresh for this revision:
  `grep -n "kaizen_" skills/agent-maintenance/SKILL.md` → **14** lines (3, 61, 201, 324, 327, 328,
  347, 391, 417, 426, 436, 444, 445, 460). `S3`'s done-condition (§4.1) now states the grep as the
  verification, not a line-number list that goes stale the moment anyone else touches the file —
  same fix applied to `S1` (P2-m3, below), which had the identical failure shape.

**The six minors and one nit — all adopted:**
- **P2-m1** — `G1`'s gate was a count check on data it then irreversibly destroys, and the
  migration query is a hand-built Cypher string literal over entries containing backticks/
  apostrophes; a quote-escaping slip could truncate a field while the count still reconciled. Every
  `C-<agent>` unit's verification step is now a **content comparison** (`entryId`+all five fields,
  diffed), not a count.
- **P2-m2** — this plan's own live snapshot was already stale the day V2 was written:
  `kaizen_architect` now exists (1 entry, written by `architect` itself mid-revision — re-confirmed
  again in this revision, see Findings below). Every enumerated count throughout this document is
  now explicitly marked "as of 2026-08-20, re-derive at dispatch," and `G1`'s done-condition leads
  with the live re-list rather than treating it as an afterthought.
- **P2-m3** — `S1`'s file/line enumeration under-counted both files (missed 2 of 4 `server.py` hits,
  stated "3 mentions" for `README.md`'s actual 5). Fixed the same way as P2-M4: grep-as-done-condition.
- **P2-m4** — `S0` (kaizen_team substrate provisioning) is now a **hard predecessor** of every
  `C-<agent>` unit, closing the duplicate-on-retry hole (`C-<agent>` uses `CREATE`, not `MERGE`) and
  mooting the constraint-idempotency open question (§6 open item 2, V2) outright — `S0` now always
  runs against an empty, never-before-written `kaizen_team`, so the unverified-behavior question
  never arises for this delivery.
- **P2-m5** — `S2` now also adds an `## M7` row to `docs/BACKLOG.md`'s "Milestone map" table
  (§41–53 of that file), not just the body section — every one of M1–M6 has both.
- **P2-m6** — corrects V2's own claim that the 5 hook scripts' escalation text "stays accurate": it
  does not — e.g. `guard-review-doc-writes.sh:12` calls the frozen file "its learnings inbox" as a
  live allowed write target, and the glob still *permits* (without escalating) an append to a file
  the convention now declares permanently frozen. Low-priority, still no dispatched unit (per the
  review's own concurrence) — §6's open item wording widened to cover escalation text and the
  permissive glob, not just header comments.
- **P2-n1** — `G1`'s done-condition now names the execution surface (`redis-cli` against the
  FalkorDB container) explicitly, not left implicit for a cold dispatch.

**Two things this revision does *not* resolve, both left open on purpose:**
1. **Which branch for the never-delete rule's reach onto the `kaizen_<agent>` graph *keys*
   themselves** (as opposed to the `inbox.md` *files*, already settled) — the review raised this as
   a genuine, unanswered tension the plan's own `G1` runs against: the stakeholder's stated reason
   for never deleting `inbox.md` is preserving a historical record, and `G1` destroys the graph keys
   holding the same records (relocated, not discarded, into `kaizen_team` — and the `inbox.md`
   originals separately preserve the same content too, which is this plan's own argument for why
   `G1` is still fine). Flagged in §6, not silently assumed, and `G1`'s own done-condition requires
   this confirmation before it dispatches.
2. **`docs/plans/generic-cypher-mcp2-coordination.md`'s own stale "Goal & definition of done"
   section** (P2-M3's second half) — explicitly `teco`'s document, not architect's plan to amend;
   surfaced in this revision's final report to `teco` directly.

---

## 1. Goal & scope

Consolidate the kaizen working-memory pattern — already mechanically live for all 12 agents as of
`cobb`'s ccf9c8b commit, each writing to its **own** `kaizen_<agent>` graph — onto **one shared
graph, `kaizen_team`, `author`-partitioned**, and deliver the three capabilities that execution did
not build: FR-7 (a single query reaching every agent's raw learnings), FR-8a (a `sessionId` field
on new entries), and FR-12/AC-9 as literally written (no `kaizen/inbox.md` is ever created for a
newly created agent — Revision note, branch (b)). In scope:

- Migrating the entries currently sitting in the populated `kaizen_<agent>` graphs into
  `kaizen_team`, attributed via `author`, verified by content comparison (not count).
- Retiring the graph keys this leaves redundant — re-listed live at dispatch time, not assumed from
  this document's own snapshot (P2-m2). The never-delete decision's reach is settled (P3-m1,
  `docs/plans/generic-cypher-mcp2-coordination.md`'s "Resolved 2026-08-20" paragraph): it covers
  `kaizen/inbox.md` files only, not these graph keys, which are retired once migrated.
- A **second** edit to every one of the 12 agents' own Learning-capture prompt sections, retargeting
  `kaizen_<agent>`/`agent='<agent>'` to `kaizen_team` with an `author: '<agent>'` field, adding the
  `sessionId` field (FR-8a) — owned by `cobb` for **all twelve**, including `architect` and
  `graph-dba` (P2-B2; `cobb` self-edits only its own file).
- The matching retarget of each of the 12 agents' own `kaizen/inbox.md` header note (P2-M2),
  landed by `cobb` in the same unit as the prompt edit.
- The team-wide query surface (FR-7) — zero new server code once `kaizen_team` is in place,
  documented as a recipe in `claude/README.md`'s Kaizen section.
- **FR-12/AC-9 delivered as written**: removing the `inbox.md`-seeding step from the new-agent
  creation convention, and updating `claude/scripts/audit-team.sh` so certification holds both
  "an existing agent's `inbox.md` is never deleted" and "a new agent never gets one" simultaneously
  (Revision note).
- `claude/scripts/audit-team.sh`, the 5 doc-scoped write-guard hooks, and `claude/teco/teco.md`'s
  coordination prose — fixed where a real live defect survives re-verification.
- Every doc describing the convention brought back to describing `kaizen_team` reality:
  `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, `docs/BACKLOG.md` (new M7 body section
  **and** Milestone-map row), `skills/agent-maintenance/SKILL.md`, `cypher-mcp/server.py` +
  `cypher-mcp/README.md`, and — new in this revision — `docs/requirements/generic-cypher-mcp2.md`
  (owned by `tico`, `T1`).

**Explicitly out of scope, per the stakeholder's binding decision:** deleting any `kaizen/inbox.md`
file, for any agent, ever. Unchanged from V1/V2's own Out of scope: falkor-chat integration,
documents-as-graph-data, `BACKLOG.md`-as-graph, the stakeholder's own direct read/write access,
guaranteed semantic search, hardened/cryptographic access control, git-history rewriting,
redesigning the write mechanism itself, the MCP server/tool rename (M6, closed).

---

## 2. Context & findings

*(V1/V2's findings, still valid and not re-derived unless noted: the write-authorization mechanism
in `cypher-mcp/server.py` is graph-name-agnostic; `CLAUDE_CODE_SESSION_ID` answers FR-8a, now
independently confirmed three times across three cold sessions; FalkorDB has no cross-graph query.)*

### Findings re-verified or added in this revision (2026-08-20, Pass-2 fix)

- **The live `kaizen_*` graph inventory has grown by one since V2's snapshot, exactly as P2-m2
  found — re-confirmed again, right now, for this revision:** `mcp__cypher__query(graph=
  'nonexistent_probe2', ...)`'s "unknown graph" error still lists `kaizen_graph_dba,
  kaizen_analyst, kaizen_data-scientist, kaizen_qa-engineer, kaizen_teco, kaizen_architect` — six
  keys, not the five V2's snapshot stated. `kaizen_architect` holds exactly one entry
  (`entryId=a1e3c9d4…`, `author='architect'`, `date='2026-08-20'`) — the one this very plan's own
  revision work wrote during V2's investigation. **Every count in this document is a snapshot,
  explicitly re-derived at dispatch by each unit's own live-recheck step** — this is not a defect
  to fix once, it is FR-13's incremental window operating exactly as designed, and it will keep
  happening for as long as this plan stays undispatched.
- **`architect.md:67` and `graph-dba.md:87` both carry "never edit your own agent definition,"
  confirmed by direct read in this revision** (not just cited from the review): the clause is the
  final sentence of each agent's Learning-capture section, immediately after "the team maintainer
  (`cobb`) reads it, verifies, and promotes entries." `cobb.md` has no equivalent sentence anywhere
  — confirmed by `grep -rln "never edit your own agent definition" claude/`, which returns all 11
  non-`cobb` agent prompts (plus each one's own `kaizen/history.md`, an unrelated hit — the
  histories quote the clause when logging that this convention was adopted) and never `cobb.md`.
- **`skills/agent-maintenance/SKILL.md`'s §1 "Creating" procedure (read in full, lines 55–62) and
  §5's "Inbox template" block (read in full, lines 434–461) both currently instruct seeding a
  frozen-stub `kaizen/inbox.md` for a newly created agent** — confirmed directly, not inferred:
  §1 step 1 says *"also seed an `inbox.md` from the §5 template's frozen-stub variant"*; §5's
  template block is headed *"seed on creation"* and its body text states the file *"exists only to
  satisfy the standard kaizen triad (`audit-team.sh` check 1)."* This is the literal contradiction
  P2-B1 named — FR-12/AC-9 cannot be true while this text stands.
- **`claude/scripts/audit-team.sh:74-79`'s check 1, read again in this revision, is unconditional**:
  `[ -f "$CL/$a/kaizen/plan.md" ] && [ -f "$CL/$a/kaizen/history.md" ] && [ -f "$CL/$a/kaizen/inbox.md" ]`
  — the script's own header comment (lines 12-13) also documents the check as "the
  kaizen/{plan,history,inbox}.md triple," which `S4` below must update alongside the executable
  logic, not just the logic alone.
- **The 5 doc-scoped write-guard wrapper scripts, read in full again for this revision**, confirm
  P2-m6's correction precisely: each one's escalation string (not just its header comment) still
  describes the agent's `kaizen/inbox.md` as a live write target — e.g.
  `guard-review-doc-writes.sh:6` ("appending to its own learnings inbox (`kaizen/inbox.md` — the
  learning-capture loop)") and `:12` ("its Write/Edit are for review documents and its learnings
  inbox only"); `guard-tico-doc-writes.sh:12` ("its learnings inbox is the one other allowed
  target"). None of the five is touched by any unit in this table — confirmed low-priority per the
  review's own concurrence, tracked as §6 open item.
- **`docs/BACKLOG.md`'s "Milestone map" table (lines 41-53) carries one row per milestone, M1
  through M6** — confirmed by direct read; `S2`'s M7 addition needs a matching row, not just the
  body section it already specified.
- **`docs/requirements/generic-cypher-mcp2.md`'s own Decision log section already has precedent for
  exactly this kind of in-place correction** — its two 2026-08-19 "Reconsidered" entries (dropping
  the frozen-`inbox.md` step, extending the reversal to `graph-dba`) are dated notes appended to a
  still-active document, not a forked successor. `T1` (§4.1) follows the same pattern for the
  FR-4/AC-3/FR-14/AC-11 supersession — the mechanical choice between in-place revision and a
  successor document is `tico`'s own call per root `AGENTS.md`'s collision rule 5, not preempted
  here.

---

## 3. Design & rationale — the four points left to the architect (V1) + consolidation mechanics (§3.6) + self-edit correction (§3.7)

*(§3.1–§3.5 unchanged from V2, itself unchanged from V1 except for two 2026-08-20 callouts — the
core shared-graph design was never in question at either plan-gate pass; every finding across both
review passes is about executability and coverage, never the design itself. Not reproduced a third
time in full here — see Version 2's own text, preserved in git history, or the live document below
§3.6 for the parts still load-bearing.)*

### 3.1 / 3.2 — FR-7's team-wide query surface, and one shared `kaizen_team` vs. per-agent graphs

Unchanged: one shared graph, `author`-partitioned, costs zero new server code for FR-7 and makes
`cobb`'s curator-clear simpler, not harder, versus the per-agent-graph shape `cobb`'s ad hoc
`ccf9c8b` execution actually built. This is the stakeholder's own binding decision (2026-08-20),
not merely architect's recommendation — full reasoning is in Version 2's own text, retrievable via
`git show <sha-of-the-V2-commit>:docs/plans/generic-cypher-mcp2.md` (V2 was superseded in place by
V3/V4, so its text lives only in git history, not elsewhere in this document — corrected in this
revision, P3-M2, from a false claim that it was preserved above).

**The FR-7 query itself (restored verbatim, P3-M2 — dropped by V3's §3 compression, still cited by
pointer from `S2`'s done-condition below without being reproduced anywhere in that version):**

```cypher
MATCH (e:KaizenEntry)
RETURN e.author, e.date, e.fact, e.evidence, e.context, e.suggestedHome
ORDER BY e.date
```

One ordinary `mcp__cypher__query` call against `kaizen_team`, no `author` filter, today's tool,
unchanged server-side — this is the whole of FR-7's delivery mechanism and the exact text `S2`
documents in `claude/README.md`'s Kaizen section as the copy-pasteable recipe. A caller wanting one
agent's slice adds `{author: '<agent>'}` to the `MATCH` pattern — one extra clause, not a different
tool.

### 3.3 — FR-8a's session-ID mechanism

Unchanged: `CLAUDE_CODE_SESSION_ID`, read at write time, included as an optional `sessionId`
property. Independently re-confirmed in three separate cold sessions now (V1's own investigation,
Pass 1's re-gate, Pass 2's re-gate) — as solid a live-verified fact as this delivery has.

```cypher
CREATE (k:KaizenEntry {
  entryId: '<uuid4>', date: '<YYYY-MM-DD>', fact: '<the fact, one line>',
  evidence: '<what was run/read/observed>', context: '<the task where it surfaced, one line>',
  suggestedHome: 'prompt | knowledge base | project docs | unsure',
  author: '<agent-slug>', createdAt: '<ISO-8601 write time>',
  sessionId: '<value of $CLAUDE_CODE_SESSION_ID, or omit this key entirely if unavailable>'
})
```
called as `mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='<agent-slug>')` — the
exact text every `C-<agent>` unit installs, replacing that agent's current `kaizen_<agent>`-targeted
block.

### 3.4 / 3.5 — Sequencing philosophy; no companion `-graph.md` note

Unchanged from V1/V2.

### 3.6 — Consolidation mechanics: who is authorized to write what

Unchanged from V2, and independently **confirmed by execution, not just reading**, in Pass 2's
re-gate: the reviewer loaded `cypher-mcp/server.py` in `cypher-mcp/.venv` and ran this plan's own
migration-query shape through `authorize_write()` directly — `agent='analyst'` against an
`author:'analyst'` `CREATE` authorizes; `agent='cobb'` against the same `CREATE` is rejected with
an explicit author-mismatch message; a decoy `CREATE (k:KaizenEntry {author:'cobb'})` embedded in
an entry's free-text `fact` field does not desync `_author_claims()`. **`cobb` genuinely cannot
perform the data-migration write for anyone but itself** — the two-actor `C-<agent>` unit shape
(agent migrates its own data; `cobb` edits the prompt text) is forced by the mechanism, confirmed
by execution, not a stylistic choice.

**The migration query shape itself (restored verbatim, P3-M2 — dropped by V3's §3 compression, the
delivery's single most escaping-sensitive statement since it is executed 12 times, in 12 isolated
agent contexts, over free-text fields containing backticks/apostrophes — §4.2 below describes it by
reference only, this is what "that reference" actually is):**

```cypher
UNWIND [
  {entryId: '<uuid4-1>', date: '<YYYY-MM-DD>', fact: '<escaped text>', evidence: '<escaped text>',
   context: '<escaped text>', suggestedHome: '<...>', createdAt: '<ISO-8601>'},
  {entryId: '<uuid4-2>', ...}
] AS e
CREATE (k:KaizenEntry {
  entryId: e.entryId, date: e.date, fact: e.fact, evidence: e.evidence,
  context: e.context, suggestedHome: e.suggestedHome,
  author: '<X>', createdAt: e.createdAt
})
```
called as `mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='<X>')`. The `author:
'<X>'` literal lives **once**, in the outer `CREATE` clause — never per-row inside the `UNWIND`
list — because that is exactly the span `_author_claims()` inspects (§3.6 above); a per-row
`author` literal would either fail to authorize or (worse) silently authorize the wrong claim if a
future entry's `fact` text happened to contain a decoy `author:` substring. No `sessionId` key is
included for these migrated entries — they are historical imports, not new writes (FR-8a, §3.3).

### 3.7 — Self-edit ownership, corrected in this revision (new)

V2 extended `cobb`'s established self-maintenance precedent to `architect` (flagged as an open
question) and to `graph-dba` (asserted as settled, citing `ccf9c8b`). Both were wrong, for two
independent reasons, both now directly verified rather than inferred:

1. **The prohibition is textual, not customary.** `cobb.md`'s Learning-capture section is the
   *only* one of the 12 agent prompts that omits "never edit your own agent definition" — every
   other agent's own prompt forbids exactly the edit V2 assigned to two of them. The carve-out was
   never "the maintainer can self-edit and nobody else has tried" — it is "eleven agents' own
   prompts explicitly say no, and the twelfth's doesn't."
2. **The `graph-dba` precedent V2 cited doesn't exist.** `git show --stat ccf9c8b`'s 43-file list
   (re-run again for this revision) does not include `claude/graph-dba/graph-dba.md` — it needed no
   edit at that time because it was already graph-backed, target-correct at that moment. There is
   no prior instance of `graph-dba` editing its own prompt to point to.

Both `architect.md`'s and `graph-dba.md`'s Learning-capture retargets now route through `cobb`,
identical in shape to the other nine non-`cobb` agents. `cobb` self-edits exactly one file:
`cobb.md`.

---

## 4. Implementation step table

Twenty-one units: 5 substrate (`S0`–`S4`) + 1 requirements-doc correction (`T1`) + 12 per-agent
consolidation (`C-<agent>`) + 1 graph-key retirement (`G1`) + 2 acceptance (`Q1`, `Q2`). "Depends
on" lists **hard** blockers only; anything else is safely parallel, sized for FR-13's
per-agent-or-per-batch requirement. **Every count named below is a 2026-08-20 snapshot — each
unit's own first step is a live re-derivation, never a read of this table.**

### 4.1 Substrate and cross-cutting fixes

| # | Owner | Files | Depends on | Done-condition |
|---|---|---|---|---|
| **S0** | `graph-dba` | (no tracked files — live graph DDL only) | — (hard predecessor of every `C-<agent>` unit, per P2-m4) | `CREATE INDEX FOR (e:KaizenEntry) ON (e.entryId)` + a uniqueness constraint on `kaizen_team`'s `:KaizenEntry.entryId`, issued against what this unit's own live check confirms is still an empty/nonexistent `kaizen_team` key (true by construction now that it's a hard predecessor — no other unit may write to `kaizen_team` before this lands). Making `S0` a hard predecessor retires the FalkorDB-constraint-idempotency open question outright (§6): the constraint is never re-issued against an already-populated graph in this delivery. |
| **S1** | `coder` | `cypher-mcp/server.py`, `cypher-mcp/README.md` | — | `grep -n 'kaizen_graph_dba' cypher-mcp/server.py cypher-mcp/README.md` returns zero matches after the edit, `kaizen_team` present in their place (server-doc-string generalization: "graph-dba's kaizen working memory" → "the team's kaizen working memory, author-partitioned"); `test_server_instructions_are_present_and_bounded` re-verified green (≤2000 chars); `cypher-mcp/build.sh` run once by hand, in-container test gate green. **As of 2026-08-20** (re-derive at dispatch, per P2-m3): `server.py` has 4 occurrences (lines 118, 134, 251, 763 — the latter two are code comments, easy to miss on a line-range-only read); `README.md` has 5 (not 3). |
| **S2** | `cobb` | `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, `docs/BACKLOG.md` | — | All three catalog paragraphs describe `kaizen_team`, author-partitioned, as the standing convention; `claude/README.md`'s Kaizen section documents the FR-7 one-query recipe **— §3.1's `MATCH`/`RETURN`/`ORDER BY` block, copied verbatim, full field list, not elided (P3-M2)** — as a copy-pasteable example, and states every `kaizen/inbox.md` as a **permanent** frozen snapshot (not "required to exist," which undersells the invariant); `docs/BACKLOG.md` gets both a new `## M7` body section (§4.4 below has the text) **and** a new row in the "Milestone map" table (lines 41-53) mirroring M1–M6's format (P2-m5). |
| **S3** | `cobb` | `skills/agent-maintenance/SKILL.md` | `S4` (P3-m3 — the FR-12 asymmetry only holds once both land; `S3` alone, mid-window, would reintroduce Pass 1's M1 finding for any agent created in that gap) | (1) Every `kaizen_<agent>`/`kaizen_<name>`/`kaizen_{name}` occurrence becomes `kaizen_team` with an `author`-filtered pattern; (2) §1's "Creating" procedure step 1 **no longer seeds an `inbox.md` for a new agent at all** — replaced with: point the new agent's Learning-capture section directly at the §3.3 `kaizen_team` recipe, no file created; (3) §5's "Inbox template" block is rewritten from "seed on creation" framing to a short historical note describing the 12 existing frozen files' shared header shape (useful only as a reference for `C-<agent>`'s header-note retarget below, never seeded again); (4) §5's distillation procedure (verify → route → log → clear) retargeted to `kaizen_team`, curator-clear no longer needing to resolve which per-agent graph an `entryId` lives in. **Verification is the grep itself, not a line list** (P2-M4's own fix, applied here too): `grep -n 'kaizen_' skills/agent-maintenance/SKILL.md` — no occurrence survives except a past-tense Origin note. As of 2026-08-20: 14 matching lines (3, 61, 201, 324, 327, 328, 347, 391, 417, 426, 436, 444, 445, 460) — both the angle-bracket (`kaizen_<name>`) and curly-brace (`kaizen_{name}`) spellings are present and must both be caught. |
| **S4** | `cobb` | `claude/scripts/audit-team.sh` | — | Check 1 (`audit-team.sh:76`, currently a three-way `-f` conjunction) narrowed to require only `plan.md` + `history.md` — `inbox.md`'s presence is no longer part of the pass/fail condition, in either direction. The script's header comment (lines 12-13, "the kaizen/{plan,history,inbox}.md triple") updated to match. **Verification is scoped to check 1's own logic, not a full-script pass** (P3-M1 — proved by the reviewer's own execution that a full-script pass is unachievable for unrelated reasons: `audit-team.sh:63-67` never enumerates a directory lacking `<name>.md`, so a synthetic agent with no `<name>.md` can't reach check 1 at all; adding a stub `<name>.md` to make it enumerable then trips checks 2/4/5/5b — deployment, roster, catalogs — which fail for any synthetic agent by construction, regardless of check 1's own state). Verify instead, **in an isolated copy of the repo tree** (never live `claude/` — copy `audit-team.sh` to `<scratch>/sub/scripts/audit-team.sh` so its `ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"` resolves to `<scratch>` and audits `<scratch>/claude/`, not the real one): a scratch agent with `<name>.md` + `kaizen/{plan,history}.md` and **no** `inbox.md` produces `PASS  <name>: kaizen plan + history present` on check 1 specifically — the overall run still `FAIL`s on checks 2/4/5/5b, expected and irrelevant, check 1's own line is the assertion. Separately, re-run the unmodified 12-agent `claude/` collection live and confirm all 12 still `PASS` check 1 (their `inbox.md` is still physically present, just no longer gates). |
| **T1** | `tico` | `docs/requirements/generic-cypher-mcp2.md` | — | FR-4, AC-3, FR-14, AC-11 marked superseded, with the 2026-08-20 stakeholder decision cited by date, following the document's own existing "Reconsidered" decision-log pattern (2026-08-19 entries, same document); the Out-of-scope bullet's now-reversed git-history rationale sentence ("which is exactly why an in-repo frozen copy is no longer needed") voided/corrected. **FR-12/AC-9 need no edit** — this delivery's branch (b) choice (Revision note) delivers them as originally written, it does not supersede them. Whether this lands as an in-place revision (with a dated note, `Version:` bump) or a successor document is `tico`'s own call per root `AGENTS.md`'s collision rule 5 — this unit states the required content, not the document mechanics. |

### 4.2 Per-agent consolidation (each independent beyond the `S0` predecessor; §3.6/§3.7 explain the actor shape)

Each unit `C-<X>`: (1) `<X>` re-checks live, first, whether `kaizen_<X>` currently holds any
`:KaizenEntry` nodes (an "unknown graph" response means nothing to migrate, skip to step 3);
(2) if nonzero, `<X>` builds the `UNWIND`/`CREATE` migration query — **the exact shape restored in
§3.6 above**, one `author` literal in the outer `CREATE` clause, never per-row — and calls
`mcp__cypher__query(graph='kaizen_team', cypher=<that text>, agent='<X>')`; **verifies by content
comparison, not count** (P2-m1): `MATCH (e:KaizenEntry {author:'<X>'}) RETURN e.entryId, e.date,
e.fact, e.evidence, e.context, e.suggestedHome, e.createdAt ORDER BY e.entryId` against `kaizen_team`,
diffed field-by-field against the same query run against `kaizen_<X>` before migration; (3) `cobb`
(for all 12, including `architect` and `graph-dba` — P2-B2/§3.7; `cobb` self-edits only its own
file) rewrites `<X>`'s own Learning-capture section to the §3.3 recipe (`kaizen_team`,
`author: '<X>'`, `sessionId` field added), **and** retargets `claude/<X>/kaizen/inbox.md`'s header
note (P2-M2 — an accuracy edit the header itself authorizes: every one of the 12 headers scopes its
own immutability promise to *"**Content below** is preserved for historical reference and will not
change"* — not "this file," so the header note was never inside that promise to begin with, P3-M2's
stronger grounding of the carve-out, adopted here in place of V3's own weaker "accuracy edit, not
new content" framing).

**The header retarget is scoped narrowly, per P3-M3 — not a blanket `kaizen_<X>`→`kaizen_team`
substitution.** Every one of the 12 headers carries a *prescriptive* clause ("New raw learnings are
written directly into the graph… `mcp__cypher__query(graph='kaizen_<X>', …)`") — that clause always
retargets to `kaizen_team` with an `author:'<X>'` filter, in all 12 files. **Four of them**
(`analyst`, `teco`, `qa-engineer`, `data-scientist` — the agents that had entries at `ccf9c8b`-time)
**additionally carry a past-tense provenance clause** ("Its N entries… **were imported into** the
`kaizen_<X>` FalkorDB graph") that states a true historical fact and **must be left untouched** — a
blanket substitution would rewrite true history ("were imported into `kaizen_team`," a graph that
did not exist on that date) into a false claim, inside a file the stakeholder has made permanent.
The other 8 agents' headers carry only the prescriptive clause and retarget in full. `teco`'s unit
additionally fixes the two stale cross-reference passages at `teco.md:72,89`.

**All 12 rows below depend on `S0`** (kaizen_team substrate, hard predecessor — P2-m4/P3-m4); the
`Depends on` column restates it per row so the dispatch table doesn't rely on a reader remembering
the heading note.

| # | Agent(s) — two-actor split (P2-B2/§3.7) | Data to migrate (**as of 2026-08-20 — re-derive live at dispatch**) | Files | Depends on | Done-condition highlights |
|---|---|---|---|---|---|
| **C-analyst** | `analyst` (data) / `cobb` (prompt + header) | 5 entries (`kaizen_analyst`) | `S0` | `claude/analyst/analyst.md:91,102`, `claude/analyst/kaizen/inbox.md` (header, prescriptive clause only) | Content-diff verified; prompt + header (scoped) retargeted |
| **C-data-scientist** | `data-scientist` (data) / `cobb` (prompt + header) | 4 entries (`kaizen_data-scientist`) | `S0` | `claude/data-scientist/data-scientist.md:91,102`, `claude/data-scientist/kaizen/inbox.md` (header, prescriptive clause only) | Content-diff verified; prompt + header (scoped) retargeted |
| **C-qa-engineer** | `qa-engineer` (data) / `cobb` (prompt + header) | 6 entries (`kaizen_qa-engineer`) | `S0` | `claude/qa-engineer/qa-engineer.md:80,91`, `claude/qa-engineer/kaizen/inbox.md` (header, prescriptive clause only) | Content-diff verified; prompt + header (scoped) retargeted |
| **C-teco** | `teco` (data) / `cobb` (prompt + header) | 5 entries (`kaizen_teco`) | `S0` | `claude/teco/teco.md:122,133` (data+prompt) + `:72,89` (M3 fix) + `claude/teco/kaizen/inbox.md` (header, prescriptive clause only) | Content-diff verified; prompt + header (scoped) retargeted; both stale cross-reference passages fixed |
| **C-graph-dba** | `graph-dba` (data) / `cobb` (prompt + header) | 0 entries (`kaizen_graph_dba` — re-confirm live immediately before `G1` acts) | `S0` | `claude/graph-dba/graph-dba.md:74-87`, `claude/graph-dba/kaizen/inbox.md` (header, full retarget — no entries, so no provenance clause to protect) | Live re-check confirms still 0; **`cobb`**, not `graph-dba`, retargets the prompt (§3.7) |
| **C-architect** | `architect` (data) / `cobb` (prompt + header) | **1 entry** (`kaizen_architect` — this delivery's own V2 investigation wrote it; not 0 as V2 stated — P2-m2) | `S0` | `claude/architect/architect.md:56,67`, `claude/architect/kaizen/inbox.md` (header — provenance-clause question moot, `architect`'s frozen header predates this entry and has no provenance clause of its own to protect; full retarget) | Content-diff verified (1 entry, `author:'architect'`); **`cobb`**, not `architect`, retargets the prompt (§3.7) |
| **C-cobb** | `cobb` (data + prompt + header — the one legitimate self-edit, §3.7) | 0 (graph key doesn't exist as of 2026-08-20) | `S0` | `claude/cobb/cobb.md:71,86,97`, `claude/cobb/kaizen/inbox.md` (header, full retarget — no entries) | Live re-check; self-edited (3 prompt mentions) |
| **C-coder** | `coder` (data) / `cobb` (prompt + header) | 0 (graph key doesn't exist) | `S0` | `claude/coder/coder.md:40,51`, `claude/coder/kaizen/inbox.md` (header, full retarget) | Live re-check; `cobb` edits |
| **C-devops** | `devops` (data) / `cobb` (prompt + header) | 0 (graph key doesn't exist) | `S0` | `claude/devops/devops.md:100,111`, `claude/devops/kaizen/inbox.md` (header, full retarget) | Live re-check; `cobb` edits |
| **C-frontend-engineer** | `frontend-engineer` (data) / `cobb` (prompt + header) | 0 (graph key doesn't exist) | `S0` | `claude/frontend-engineer/frontend-engineer.md:86,97`, `claude/frontend-engineer/kaizen/inbox.md` (header, full retarget) | Live re-check; `cobb` edits |
| **C-tdd-engineer** | `tdd-engineer` (data) / `cobb` (prompt + header) | 0 (graph key doesn't exist) | `S0` | `claude/tdd-engineer/tdd-engineer.md:58,69`, `claude/tdd-engineer/kaizen/inbox.md` (header, full retarget) | Live re-check; `cobb` edits |
| **C-tico** | `tico` (data) / `cobb` (prompt + header) | 0 (graph key doesn't exist) | `S0` | `claude/tico/tico.md:151,162`, `claude/tico/kaizen/inbox.md` (header, full retarget) | Live re-check; `cobb` edits |

### 4.3 Graph-key retirement

| # | Owner | Depends on | Done-condition |
|---|---|---|---|
| **G1** | `graph-dba` | `S0` (must have landed); each `C-<agent>` unit whose key it retires | **The never-delete-reach question is settled — no re-ask needed.** `docs/plans/generic-cypher-mcp2-coordination.md`'s "Resolved 2026-08-20" paragraph (lines 126-132) states the never-delete decision covers `kaizen/inbox.md` *files* only; the `kaizen_<agent>` graph *keys* **are** to be deleted once their entries are consolidated and verified, and that record itself discharges this unit's confirmation gate — cite it directly, `graph-dba` does not open this unit by asking `tico` (P3-m1, corrects V3's own instruction to re-ask). First **executed** step: **re-list loaded graphs live** (`mcp__cypher__query` against any nonexistent probe name) — do not trust this document's 6-key snapshot (`kaizen_graph_dba`, `kaizen_analyst`, `kaizen_data-scientist`, `kaizen_qa-engineer`, `kaizen_teco`, `kaizen_architect`, as of 2026-08-20) without re-deriving it, since FR-13's incremental window means more may have appeared. For each key whose corresponding `C-<agent>` unit has confirmed a content-diff-verified migration (or confirmed 0 entries for `kaizen_graph_dba`): `GRAPH.DELETE <key>`, issued via `redis-cli` against the FalkorDB container (P2-n1 — the execution surface is a Redis command, not an `mcp__cypher__query` write shape, per §3.6), gated by `graph-dba`'s own destructive-ops hook approval. Can run incrementally as `C-<agent>` units land — does not need to wait for all twelve. |

### 4.4 Acceptance

| # | Owner | Files | Depends on | Done-condition |
|---|---|---|---|---|
| **Q1** | `qa-engineer` | — (interim check) | ≥3 of the 12 `C-<agent>` units | AC-1, AC-4, AC-6, AC-7 exercised live against `kaizen_team` for whichever agents have migrated so far |
| **Q2** | `qa-engineer` | `docs/test-plans/generic-cypher-mcp2.md`, `docs/test-reports/generic-cypher-mcp2-report.md` | `S0`–`S4`, `T1`, all 12 `C-<agent>` units, `G1` | AC-1, AC-2, AC-4…AC-10, AC-12, AC-13 each exercised live (§5.2); AC-3/AC-11 confirmed dropped (T1 landed, not silently absent); AC-9 confirmed delivered as written — `qa-engineer` independently re-runs `S4`'s **isolated-tree, check-1-scoped** assertion (never a full-script pass on live `claude/` — P3-M1), not re-taken on `cobb`'s word; the unmodified 12-agent `claude/` collection also re-confirmed still `PASS` on check 1 at closing. |

**`docs/BACKLOG.md` additions (for `S2`):**

Milestone-map row (after the M6 row, same table, lines 41-53):

```markdown
| **M7 — Generic Cypher MCP, team-wide rollout** | All 12 agents' raw kaizen capture consolidated onto one shared `kaizen_team` graph (`author`-partitioned), FR-7's one-query team-wide surface and FR-8a's `sessionId` field delivered, FR-12/AC-9 delivered as written (no `inbox.md` for a new agent) — an interim ad hoc per-agent-graph rollout (`ccf9c8b`, 2026-08-20) is reconciled onto this design | `C-701 → C-721` |
```

Body section (mirroring the M5/M6 format):

```markdown
## M7 — Generic Cypher MCP, team-wide rollout

`mcp__cypher__query`'s write path (M5) is rolled out from `graph-dba` alone to all twelve agents,
consolidated onto one shared graph (`kaizen_team`) with `author` as the per-agent partition — no
new write mechanism (FR-1), zero `cypher-mcp/server.py` logic changes. An interim ad hoc rollout
(commit `ccf9c8b`, 2026-08-20) built one graph per agent instead; this milestone reconciles that
onto the stakeholder-confirmed shared-graph design, and separately delivers FR-12/AC-9 as written
(no `kaizen/inbox.md` for a newly created agent, alongside the standing "never delete an existing
one" rule). Requirements: [`requirements/generic-cypher-mcp2.md`](./requirements/generic-cypher-mcp2.md)
(FR-1…FR-14 / AC-1…AC-13, two superseded — see plan §5.2) · plan:
[`plans/generic-cypher-mcp2.md`](./plans/generic-cypher-mcp2.md) · coordination:
[`plans/generic-cypher-mcp2-coordination.md`](./plans/generic-cypher-mcp2-coordination.md).

### Items
- **C-701 — Repo-wide catalog docs.** 🔵 `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`,
  `docs/BACKLOG.md` retargeted to `kaizen_team`. Owner: `cobb` (S2).
- **C-702 — Agent-creation + distillation convention, FR-12/AC-9 delivered.** 🔵
  `skills/agent-maintenance/SKILL.md` retargeted to `kaizen_team`; new-agent `inbox.md` seeding
  removed entirely. Owner: `cobb` (S3).
- **C-703 — Server doc-strings.** 🔵 `cypher-mcp/server.py`/`README.md` `kaizen_graph_dba` →
  `kaizen_team`. Owner: `coder` (S1).
- **C-704 — `kaizen_team` provisioning.** 🔵 Index/constraint, hard predecessor of all migrations.
  Owner: `graph-dba` (S0).
- **C-705 — Certification tooling, FR-12 asymmetry.** 🔵 `claude/scripts/audit-team.sh` check 1
  narrowed to plan+history; inbox.md never required. Owner: `cobb` (S4).
- **C-706 — Requirements doc correction.** 🔵 FR-4/AC-3/FR-14/AC-11 marked superseded. Owner:
  `tico` (T1).
- **C-707…C-718 — Per-agent consolidation** (one item per agent, all 12; data migration by the
  agent itself, prompt + inbox-header retarget by `cobb`, except `cobb` self-edits its own). 🔵
  each (C-<agent>).
- **C-719 — Graph-key retirement.** 🔵 Live-relisted `kaizen_<agent>` keys `GRAPH.DELETE`d once
  migrated — never-delete-reach question already resolved (coordination doc, "Resolved
  2026-08-20"), no re-confirmation gate. Owner: `graph-dba` (G1).
- **C-720 / C-721 — Acceptance passes.** 🔵 Interim (Q1) and closing (Q2), AC-1…AC-13 (two
  superseded) exercised live.
```

---

## 5. FR/AC verification mapping

### 5.1 Functional requirements — status and covering unit

| FR | Status | Covering unit(s) / reason |
|---|---|---|
| FR-1 (reuse M5's mechanism, no new write shape) | Already delivered by `ccf9c8b` | Confirmed by execution in Pass 2 (§3.6) |
| FR-2 (write new learnings into own graph) | Already delivered by `ccf9c8b`, wrong target | All 12 `C-<agent>` retarget to `kaizen_team` |
| FR-3 (one-time import of `inbox.md` content) | Already delivered for the 4 populated agents | `C-analyst`/`C-data-scientist`/`C-qa-engineer`/`C-teco` relocate that data into `kaizen_team` |
| **FR-4** (delete `inbox.md` after import) | **Superseded** | `T1` records the supersession in the requirements doc; overridden by the never-delete decision |
| FR-5 (`history.md` unchanged) | Already true | No unit needed |
| FR-6 (any agent reads any other's graph) | Already true mechanically | No unit needed — trivial once data is in `kaizen_team` |
| FR-7 (single query reaches every agent) | **Outstanding**, this delivery's core new capability | `S0` + `S2` (documented recipe) + all `C-<agent>` |
| FR-8 (locked 5-field schema) | Already true | No unit needed |
| FR-8a (`sessionId` field) | **Outstanding** | §3.3 recipe installed by every `C-<agent>` unit |
| FR-9 (author/curator write shapes) | Already true | No unit needed |
| FR-10 (`cobb`'s distillation cadence unchanged) | Needs target-graph updated | `S3` |
| FR-11 (docs describe actual behavior) | Partially delivered by `ccf9c8b` | `S2`, `S3`, `C-teco`'s M3 fix, `C-<agent>`'s inbox-header fix (P2-M2) |
| **FR-12** (new-agent convention, no `inbox.md` seeded) | **Outstanding — delivered as written this revision (branch (b), §Revision note)** | `S3` (removes the seed step) + `S4` (audit-team.sh holds the asymmetry) |
| FR-13 (incremental, not atomic) | Design property of §4's table | Every unit is independently dispatchable |
| **FR-14** (delete `graph-dba`'s frozen `inbox.md`) | **Superseded** | `T1` |

### 5.2 Acceptance criteria — verification mapping

| AC | Verification approach | Altitude |
|---|---|---|
| AC-1 | Live query against `kaizen_team` filtered to a migrated agent's `author`, from a different agent's context | Live |
| AC-2 | Per-agent content-diff (not count) between the original `kaizen_<agent>` data and `kaizen_team {author:'<X>'}` post-migration | Live, per-unit self-check |
| **AC-3** | **Superseded** (`T1`). No unit executes a deletion; `Q2` confirms the *absence* of any deletion. | — |
| AC-4 | Agent writes one new entry via the §3.3 recipe against `kaizen_team`; independent second read confirms it | Live |
| AC-5 | `cobb` runs a real distillation pass (append to `history.md`, confirm, curator-clear against `kaizen_team`) | Live, full workflow |
| AC-6 | Mismatched `author`/`agent` write attempt rejected — already confirmed by execution in Pass 2 (§3.6); `Q2` re-runs it against `kaizen_team` specifically | Live |
| AC-7 | One `MATCH (e:KaizenEntry) RETURN e.author, e.date, ... ORDER BY e.date` (no author filter) against `kaizen_team`, spanning ≥2 distinct authors | Live, the direct FR-7 proof |
| AC-8 | `grep -rlE 'kaizen_[A-Za-z{<][A-Za-z_{}<>-]*' claude/ skills/ cypher-mcp/ docs/BACKLOG.md docs/requirements/generic-cypher-mcp2.md AGENTS.md` (P2-M1's corrected pattern, widened roots narrowed again in this revision per P3-m2 — the `docs/` root alone returned 64 files, 19 of them historical M5/M6/earlier-pass docs under `docs/plans/`, `docs/reviews/`, `docs/test-plans/`, `docs/test-reports/` that no unit here touches; narrowed to the two `docs/` files this delivery's units actually touch) before/after `S0`–`S4`/`C-<agent>`; every post-migration hit sorts into **three** buckets, not two (P3-m2's fix): genuinely historical (past-tense, e.g. a frozen `inbox.md`'s provenance clause), a real remaining gap, or **an arbitrary fixture/example graph name in test code — semantically irrelevant, out of scope** — `cypher-mcp/tests/test_server.py` (17 occurrences, `kaizen_graph_dba` used only as a placeholder graph key passed to `run_query()`) is pre-classified into this third bucket, not swept for renaming and not filed as a defect | Static, repeatable |
| AC-9 | Read `skills/agent-maintenance/SKILL.md` §1 post-`S3`: confirm **no `inbox.md`-seeding step remains at all** for a new agent (not just a graph-name check, per P2-B1); independently re-run `S4`'s **check-1-scoped, isolated-tree** assertion (P3-M1 — never a full `audit-team.sh` pass, which cannot pass for a synthetic agent for reasons outside this delivery's scope) and confirm `PASS` on check 1 specifically | Static + live |
| AC-10 | `Q1` (interim pass): re-run AC-1/AC-4/AC-6 scoped to whichever agents have consolidated at that point | Live, exercised mid-rollout |
| **AC-11** | **Superseded** (`T1`) — `graph-dba`'s `kaizen/inbox.md` is never deleted | — |
| AC-12 | `MATCH (e:KaizenEntry) RETURN DISTINCT keys(e)` across ≥2 different agents' entries in `kaizen_team`, confirm identical key sets modulo `sessionId` | Live |
| AC-13 | One entry with `sessionId IS NOT NULL` (new, post-consolidation) and one with `sessionId IS NULL` (imported) both present and distinguishable | Live |

---

## 6. Open items for the plan-gate reviewer

**Resolved since V3, not carried forward as open:** whether the never-delete decision's reach
extends to the `kaizen_<agent>` graph *keys* (not just the `inbox.md` files) — V3's own open item
1 — is now settled by `docs/plans/generic-cypher-mcp2-coordination.md`'s "Resolved 2026-08-20"
paragraph (lines 126-132, cited directly in `G1`'s done-condition, P3-m1): yes, delete the keys,
no re-confirmation needed at dispatch time.

1. **`docs/plans/generic-cypher-mcp2-coordination.md`'s own stale "Goal & definition of done"
   section** (Pass 2's P2-M3, second half) is explicitly outside this plan's authority to fix —
   flagged directly to `teco` in this revision's dispatch report, not silently dropped, not
   silently absorbed as an extra unit here.
2. **The 5 doc-scoped write-guard hook scripts' stale escalation text** (P2-m6's correction to V2's
   own claim) remains unfixed by design — low-priority, concurred by all three review passes, worth
   a one-line fix next time `cobb` touches those files for an unrelated reason.
3. **Unit ownership still concentrates on `cobb`** — now 11 of 12 `C-<agent>` prompt-edit-plus-header
   halves plus `S2`/`S3`/`S4` (P2-B2's fix removed `architect`'s and `graph-dba`'s self-edits, which
   *increases* `cobb`'s load versus V2, not decreases it). Still not a defect, still worth
   surfacing to whoever coordinates dispatch.
4. **The "13 agents" vs. 12 discrepancy** in the original dispatch brief remains immaterial — the
   roster and unit count in §4 are correct at 12, confirmed again via `ls claude/` in this
   revision.

**Pass 3's own recommendation, recorded here rather than re-litigated:** the reviewer explicitly
judged a fourth plan-gate pass would not pay for itself, since all three of that pass's majors were
narrow done-condition wording fixes with no design content, on a plan not yet executed against.
This revision applies them in place per that recommendation; `teco` dispatches without a further
gate unless this fix introduces something new.
