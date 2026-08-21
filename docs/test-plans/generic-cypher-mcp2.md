# Generic Cypher MCP — team-wide kaizen inbox rollout — Test Plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** C-701…C-721 (M7)

## 1. Scope & objective

Closing acceptance pass (unit `Q2` of `docs/plans/generic-cypher-mcp2-coordination.md`) for the
`generic-cypher-mcp2` feature (M7) — verifying **at behavior/acceptance altitude** that the
delivered system actually produces AC-1, AC-2, AC-4…AC-10, AC-12, AC-13 (AC-3 and AC-11 are
confirmed **superseded**, not tested — §1.1), by driving the live `mcp__cypher__query` tool, the
live `kaizen_team` graph, a real `cobb` distillation dispatch, `claude/scripts/audit-team.sh`
against both the live tree and a fresh isolated scratch copy, and the delivered documentation —
observing what actually happens, not re-reading the plan or the unit ledger's own narration of
what happened.

Everything upstream of this pass is already gated and not re-litigated here:
- Plan gate (`docs/reviews/generic-cypher-mcp2.md`, 3 passes, final verdict **approve with
  suggestions**, 0 blockers).
- 21 implementation units (`S0`–`S4`, `T1`, 12 `C-<agent>`, `G1`, `Q1`) delivered per
  `docs/plans/generic-cypher-mcp2-coordination.md`'s unit ledger, each with its own live
  self-verification (content-diff, not count, per unit).
- `Q1`'s interim pass — AC-1, AC-4, AC-6, AC-7 already exercised live mid-rollout, **PASS**.

What none of those gates individually prove is whether the **finished, consolidated** system
holds together as a whole — every agent's prompt actually pointing at `kaizen_team`, the
certification tooling actually reflecting the new invariant, the two intentionally-incremental
loose ends actually being the *only* loose ends, and the one workflow (`cobb`'s real distillation)
that only a live end-to-end dispatch against the finished `kaizen_team` graph can prove. That is
this document's one job.

### 1.1 What is explicitly not under test

- **AC-3, AC-11 — superseded**, confirmed by reading `docs/requirements/generic-cypher-mcp2.md`
  directly: both are marked `Superseded 2026-08-20` in place, struck-through original text kept,
  citing the stakeholder decision recorded in
  `docs/plans/generic-cypher-mcp2-coordination.md`'s "Stakeholder decision (2026-08-20)" section.
  Not re-litigated here — this pass instead confirms the *absence* of any deletion (folded into
  TP-003).
- The 16 offline unit tests / in-container gate for `cypher-mcp/server.py` (`S1`'s own scope,
  already green — `test_server_instructions_are_present_and_bounded` re-verified, `build.sh`
  rebuilt with a new content hash).
- Fuzzing `authorize_write()`'s regex internals beyond the requirement-level AC-6 directions —
  already covered by the unit suite plus live execution proof in the plan-gate re-verification
  (§3.6 of the plan).
- Load/concurrency/security testing — a dozen agents, trusted self-identification explicitly
  scoped to "well-behaved callers can't do this by accident" (FR-9), no concurrent writers in
  this rollout. No real risk at this scale.

## 2. References

- `docs/requirements/generic-cypher-mcp2.md` — FR-1…FR-14 / AC-1…AC-13 (source of test items
  below; FR-4/AC-3/FR-14/AC-11 superseded 2026-08-20).
- `docs/plans/generic-cypher-mcp2.md` (Version 5) — §3 design/mechanics, §4 the 21-unit step
  table, §5.2 the AC→verification mapping this plan expands into concrete test items.
- `docs/plans/generic-cypher-mcp2-coordination.md` — unit ledger (every unit's delivered scope
  and self-verification), the Stakeholder decision section, the Resume note.
- `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, `docs/BACKLOG.md` — checked for AC-8.
- `skills/agent-maintenance/SKILL.md` — checked for AC-9 (creation procedure) and read for AC-5's
  distillation procedure (§5), which `cobb` is dispatched to run for real.
- `claude/scripts/audit-team.sh` — driven live (unmodified `claude/`) and against a fresh isolated
  scratch copy (AC-9, per P3-M1's method — never a full-script pass on the live tree).
- Every one of the 12 agents' own operative prompts (`claude/<agent>/<agent>.md`) — checked for
  AC-8, spot-read for AC-4's write recipe.
- `claude/<agent>/kaizen/inbox.md` (all 12) — checked unchanged for AC-3's dropped-deletion
  confirmation.
- Delivered artifact under live test: `cypher-mcp/server.py` (via the running `cypher` MCP
  server, not read directly for its logic — this pass drives it).

## 3. CPG relevance check (per this agent's own standing orientation)

Live-checked before writing test items, not assumed from the plan's own "not applicable" note:
querying `mcp__cypher__query` against a deliberately-nonexistent probe graph name returns the
live loaded-graphs list — `ws:test, cpg_falkorchat, reference, ws:qa-tico-workflows-manual,
ws:acme, cpg_salesperson, ws:eval, kaizen_team, kaizen_analyst, kaizen_teco` — no `cpg_claude`,
`cpg_skills`, or `cpg_cypher-mcp`-shaped graph exists for any of the trees this delivery touches
(`claude/`, `skills/`, `cypher-mcp/`).

`CPG: not applicable — no loaded Joern CPG covers claude/, skills/, or cypher-mcp/ (confirmed live
via a fresh mcp__cypher__query probe this session, not just cited from the plan's own §CPG line);
this delivery is agent-prompt/graph-schema/documentation work, and the checks below are live
behavioral/acceptance checks against a running FalkorDB graph and a running MCP tool, not a
static code-impact or test-gap question a CPG would usefully answer.`

## 4. Live-environment grounding (confirmed before writing test items)

- `mcp__cypher__query` connection is fresh: a live probe against `kaizen_team` returns real rows,
  not `graph_not_found_message()`.
- `kaizen_team` currently holds **32** `:KaizenEntry` nodes across 7 distinct authors (`analyst`
  8, `architect` 4, `data-scientist` 4, `graph-dba` 2, `qa-engineer` 8, `teco` 5, `tico` 1) — the
  coordination ledger's own snapshot (`analyst` 8, `architect` 4, `data-scientist` 4,
  `graph-dba` 2, `qa-engineer` 7, `teco` 5, `tico` 1, 8 agents at 0) is a few dispatches stale, as
  its own text warns: `qa-engineer` is 8 not 7 live — the 8th is `Q1`'s own uncleaned AC-4 test
  entry (`cda51378-…`, self-labeled "safe to skip in distillation"), a minor housekeeping residue
  flagged in §9, not a defect.
- Loaded `kaizen_*` graph keys: exactly `kaizen_team`, `kaizen_analyst`, `kaizen_teco` — the other
  10 `kaizen_<agent>` keys are confirmed retired (not merely claimed retired), matching the
  ledger's `G1` row and the "deliberately still live" note for `analyst`/`teco`.
- The two known, already-documented data-fidelity defects (`e40a95fe-…` author `teco`,
  `fe2007f5-…` author `analyst`) are confirmed still present, unresolved, in `kaizen_team` —
  treated per the brief as known open defects pending a `cobb` curator-clear that needs
  stakeholder approval, not new findings and not a blocker.
- `git status` confirms all 12 `claude/*/kaizen/inbox.md` files carry no working-tree
  modification (not in the modified-file list) — consistent with the "byte-identical to `HEAD`,
  header-retarget dropped" state the plan's Version 5 revision note and the ledger both record.
- `claude/scripts/audit-team.sh` check 1's source (`:74-84`) confirmed by direct read: the
  three-way `-f` conjunction is gone, replaced by a `plan.md`+`history.md`-only check; the header
  comment (`:9-12`) matches.

## 5. Risk assessment & coverage strategy

The highest residual risk at this altitude is exactly what no per-unit self-check could see:
whether the **finished, cross-cutting** state is actually coherent — 12 independently-edited
prompts all pointing at the same graph with no straggler, the certification tooling's actual
live behavior (not just its source), and the one workflow this whole design serves (`cobb`
distilling raw capture into durable knowledge) actually working end to end against the
consolidated graph, which no per-unit self-check exercised (`M5`'s own equivalent, `AC-5`, was
proven once against the *interim* per-agent-graph shape, before this delivery's own second
retarget to `kaizen_team` — not automatically still true without a fresh live run).

**Coverage decisions, explicit:**
- AC-1, AC-6, AC-7, AC-12, AC-13 are cheap, deterministic live checks — one test item each.
- AC-2 is a **sampled** content-diff (not all 4 populated agents re-verified from scratch) —
  each `C-<agent>` unit already ran its own byte-exact diff at migration time (per-unit
  self-checks in the ledger); this pass re-derives one spot-check independently (`analyst`,
  chosen because it also carries a known open defect worth cross-referencing) rather than
  repeating all 4, which would just re-run what the unit ledger already shows passing.
- AC-4 and AC-6 are exercised **fresh** by this pass (not merely re-cited from `Q1`), because
  they are cheap, self-cleaning, and this is the closing gate — no reason to take even `Q1`'s
  own live run on its word when a second independent one costs one write+read+cleanup.
- AC-5 requires a **real `cobb` dispatch** — the same reasoning M5's own QA pass used (this is
  the one criterion needing a real acceptance exercise, not a unit test) — but this delivery's
  version of AC-5 is not redundant with M5's: M5 proved the workflow once, against the interim
  per-agent-graph shape; the workflow has never been proven against the **consolidated**
  `kaizen_team` shape this delivery actually ships. Dispatched once, against one real raw
  `graph-dba` entry, scoped narrowly to avoid touching either known-defect entry or the two
  deliberately-held-back graph keys.
- AC-8 is the widest test item — a repo-wide grep plus a full read of every doc FR-11 names,
  **plus each of the 12 agents' own operative prompts**, not just the 4 catalog docs the plan's
  own AC-8 mapping enumerates — because a stale per-agent prompt is exactly the kind of drift
  no catalog-level grep alone would catch (and, per §6 below, this pass finds exactly that).
- AC-9 independently re-runs `S4`'s isolated-tree, check-1-scoped method from scratch — a fresh
  scratch copy, not `cobb`'s prior scratch copy — per the brief's explicit instruction not to take
  the prior result on its word.
- AC-10 is satisfied by citing `Q1`'s own already-delivered, already-gated interim pass (PASS,
  per the ledger) plus this pass's own observation that the rollout is now, in fact, complete for
  all 12 agents (the incremental property held throughout, verifiably, per the unit ledger's
  own per-unit dated entries) — no new live check needed beyond what TP-001…TP-007 already prove
  for the finished state.

**Deliberately not tested, and why:** see §1.1.

## 6. Test items

### TP-001 — AC-1: cross-agent live read

**Preconditions:** `kaizen_team` live and populated (confirmed §4).

**Steps:** `mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry {author:
'graph-dba'}) RETURN e.date, e.fact, e.evidence, e.context, e.suggestedHome ORDER BY e.date")` —
no `agent` param, from this QA session (not `graph-dba`), i.e. "another agent queries a migrated
agent's working memory directly."

**Expected result:** Returns `graph-dba`'s entries with the standard 5 markdown-equivalent
fields populated, no error, no identity gating on reads.

**Priority:** High. **Type:** Acceptance (live tool call).

### TP-002 — AC-2: import completeness + field-fidelity spot-check

**Preconditions:** `analyst`'s frozen `claude/analyst/kaizen/inbox.md` (5 original entries) and
its `kaizen_team {author:'analyst'}` entries both readable.

**Steps:** (a) Read `claude/analyst/kaizen/inbox.md` in full. (b)
`mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry {author:'analyst'}) RETURN
e.entryId, e.date, e.fact, e.suggestedHome ORDER BY e.date")`. (c) Match each of the 5 original
`## YYYY-MM-DD — <fact>` headings against a returned row by date+fact text.

**Expected result:** All 5 original entries (2026-08-11 ×2, 2026-08-15, 2026-08-16, 2026-08-19)
present in `kaizen_team`, fact text matching (allowing only the markdown→property structural
mapping). Additional post-migration entries (dated 2026-08-20) are expected and fine — FR-13
incremental capture continuing after migration, not a fidelity problem.

**Priority:** High. **Type:** Acceptance (live tool call + static diff).

### TP-003 — AC-3/AC-11: confirmed superseded, no deletion occurred

**Preconditions:** None beyond repo access.

**Steps:** (a) `grep -m1 -H 'Superseded' docs/requirements/generic-cypher-mcp2.md` for FR-4,
AC-3, FR-14, AC-11. (b) `git status --short` — confirm no `claude/*/kaizen/inbox.md` appears as
deleted or modified. (c) `ls claude/*/kaizen/inbox.md | wc -l` — confirm all 12 still present.

**Expected result:** (a) All four marked `Superseded 2026-08-20`, original text struck through
and kept, citing the stakeholder decision. (b) Zero `inbox.md` files touched. (c) 12 files present.

**Priority:** High. **Type:** Static (requirements-doc read + git + filesystem).

### TP-004 — AC-4: a fresh author-write, then an independent read, self-cleaned

**Preconditions:** `kaizen_team` live.

**Steps:** (a) `mcp__cypher__query(graph='kaizen_team', cypher=<CREATE ... author:'qa-engineer',
sessionId:'<this session's id>'>, agent='qa-engineer')`. (b) A second, independent read by
`entryId` (no `agent`). (c) Curator-clear via `agent='cobb'` immediately after, leaving zero
permanent pollution.

**Expected result:** (a) Write accepted. (b) Second read returns the entry immediately, with
`sessionId` populated. (c) Cleanup succeeds, node gone.

**Priority:** High. **Type:** Acceptance (live tool call, write + read + cleanup).

### TP-005 — AC-5: a real `cobb` dispatch, distillation against the *consolidated* graph

**Preconditions:** At least one raw, genuinely-promotable `:KaizenEntry` in `kaizen_team` not
already claimed by another test item — `graph-dba`'s `a3f4e1b2-…` (2 entries live, confirmed §4).

**Steps:** Dispatch `cobb` as a real subagent with a brief scoping it to exactly one entry
(`a3f4e1b2-…`), instructing the documented 4-step sequence (`agent-maintenance` skill §5: verify
→ route → append to `claude/graph-dba/kaizen/history.md`, confirm → curator-clear from
`kaizen_team`), explicitly forbidding it from touching the two known-defect entries, the two
deliberately-held-back graph keys, or any settings/permissions. Independently re-verify
afterward: `history.md` has the new entry; `kaizen_team {author:'graph-dba'}` count went 2→1;
the specific `entryId` is gone.

**Expected result:** All four steps executed in order (append confirmed before delete), verified
independently, not taken on `cobb`'s report alone.

**Priority:** Critical — the one criterion this pass cannot satisfy any other way, and the one
workflow M5's own AC-5 proof never covered (it ran against the interim per-agent-graph shape,
not this delivery's consolidated one). **Type:** Acceptance/e2e (real subagent dispatch, real
durable side effect).

### TP-006 — AC-6: cross-attribution rejected, both directions

**Preconditions:** None beyond a live connection.

**Steps:** (a) `CREATE (... author:'cobb' ...)` declared with `agent='qa-engineer'`. (b) The
reverse: `author:'qa-engineer'`, `agent='cobb'`.

**Expected result:** Both rejected by `authorize_write()` before any write occurs, explicit
author/agent-mismatch message; a follow-up count check confirms no phantom node either direction.

**Priority:** High. **Type:** Acceptance (live tool call, negative test, both directions).

### TP-007 — AC-7: the one-query team-wide surface

**Preconditions:** `kaizen_team` populated by ≥2 distinct authors (true, §4).

**Steps:** `mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry) RETURN
e.author, e.date, e.fact, e.evidence, e.context, e.suggestedHome ORDER BY e.date")` — the exact
recipe `claude/README.md`'s Kaizen section documents — no `author` filter.

**Expected result:** Rows spanning every author currently present (7, §4) in one query — the
direct FR-7 proof, not eleven-plus separate lookups.

**Priority:** High. **Type:** Acceptance, the direct FR-7 proof.

### TP-008 — AC-8: no doc, and no agent's own prompt, contradicts actual behavior

**Preconditions:** None beyond repo access.

**Steps:** (a) `grep -rlE 'kaizen_[A-Za-z{<][A-Za-z_{}<>-]*' claude/ skills/ cypher-mcp/
docs/BACKLOG.md docs/requirements/generic-cypher-mcp2.md AGENTS.md`, then triage every hit into
the plan's own 3 buckets (historical/past-tense, a real remaining gap, or an out-of-scope test
fixture). (b) Read `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`'s `claude/` bullet,
`docs/BACKLOG.md`'s M7 section, `cypher-mcp/server.py`'s doc-strings (via the live tool's own
served instructions) and `cypher-mcp/README.md`. (c) Read **each of the 12 agents' own operative
prompt** (`claude/<agent>/<agent>.md`) in full, not just its Learning-capture section, checking
for any surviving reference to the interim per-agent `kaizen_<agent>` shape or to the dropped
inbox-seeding step.

**Expected result:** (a) Every hit outside the pre-classified fixture bucket
(`cypher-mcp/tests/test_server.py`, 17 arbitrary occurrences) is genuinely historical
(past-tense, inside a frozen `inbox.md`'s provenance clause or a dated `history.md` entry) — no
live, present-tense claim survives. (b) All five catalog/server docs describe `kaizen_team`,
author-partitioned, accurately. (c) No agent's own prompt still instructs writing to, or reading
from, a per-agent `kaizen_<agent>` graph, and none still instructs seeding an `inbox.md` on
agent creation.

**Priority:** High. **Type:** Static (repo-wide grep + full document/prompt reads).

### TP-009 — AC-9: no `inbox.md`-seeding step survives, anywhere it could fire

**Preconditions:** None beyond repo access and a disposable scratch directory.

**Steps:** (a) Read `skills/agent-maintenance/SKILL.md` §1's Creating procedure in full — confirm
no `inbox.md`-seeding instruction remains. (b) **Independently** (fresh scratch copy, not
`cobb`'s prior one) build an isolated tree per `S4`'s method: `<scratch>/claude/scripts/
audit-team.sh` (copied) + one synthetic agent directory with `<name>.md` + `kaizen/{plan,
history}.md` and **no** `inbox.md`; run the copied script against `<scratch>/claude/`. (c) Read
every one of the 12 agents' own prompt for a lingering "seed it on creation"-style instruction
that would contradict (a) if followed literally. (d) Run the real, unmodified
`claude/scripts/audit-team.sh` against the live 12-agent `claude/` tree, isolate check 1's output
lines.

**Expected result:** (a) confirmed, no seeding step. (b) The synthetic agent produces exactly
`PASS  <name>: kaizen plan + history present` on check 1, independent of the run's overall
FAIL on unrelated checks (2/4/5/5b — deployment, roster, catalogs — which fail for any synthetic
agent by construction, per P3-M1, and are not this criterion's concern). (c) No contradicting
instruction survives in any of the 12 prompts. (d) All 12 live agents `PASS` check 1 specifically.

**Priority:** High. **Type:** Static + live (isolated-tree execution, independently re-derived).

### TP-010 — AC-10: incremental delivery was real progress throughout, not merely claimed

**Preconditions:** `Q1`'s interim pass already delivered (ledger: **PASS**).

**Steps:** (a) Read the `Q1` ledger row in full — confirm it independently exercised AC-1/AC-4/
AC-6/AC-7 mid-rollout, scoped to whichever agents had migrated at that point, with a real per-
author count cross-check against the ledger. (b) Confirm (via TP-001…TP-007 above) that the same
criteria now hold for the **complete**, 12-agent-migrated end state — i.e. the property AC-10
asks for (partial progress being verifiably real, not merely claimed) held at `Q1`'s checkpoint
and continues to hold now that the rollout is finished.

**Expected result:** `Q1` genuinely exercised live checks mid-rollout (not a paper pass); the
same criteria hold at closing.

**Priority:** Medium (largely already proven by `Q1`; this item confirms continuity, not fresh
mechanism). **Type:** Static (ledger read) + citation of TP-001…TP-007.

### TP-011 — AC-12: identical field-set across authors, modulo `sessionId`

**Preconditions:** `kaizen_team` populated by ≥2 authors.

**Steps:** `mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry) RETURN DISTINCT
keys(e) AS ks, e.author AS a")`.

**Expected result:** Every distinct key-set is either the 8-field base set (`entryId`, `date`,
`fact`, `evidence`, `context`, `suggestedHome`, `author`, `createdAt`) or that same set plus
`sessionId` — no author's entries carry any other field, missing or extra.

**Priority:** Medium. **Type:** Live, one query.

### TP-012 — AC-13: `sessionId` distinguishes new entries from imported ones

**Preconditions:** Same as TP-011.

**Steps:** `mcp__cypher__query(graph='kaizen_team', cypher="MATCH (e:KaizenEntry) RETURN count(e)
AS total, sum(CASE WHEN e.sessionId IS NOT NULL THEN 1 ELSE 0 END) AS with_session")`.

**Expected result:** At least one entry with `sessionId IS NOT NULL` (a new, post-consolidation
write) and at least one with `sessionId IS NULL` (an older/imported entry) both present —
distinguishable on this basis, per FR-8a's design (imported entries never carry the field at
all, not the field present-but-empty).

**Priority:** Medium. **Type:** Live, one query.

## 7. Environment & data setup

- No environment bring-up needed — the shared FalkorDB instance is already running with
  `kaizen_team` live and populated (confirmed §4), and the `cypher` MCP server connection is
  already fresh.
- TP-004's write is self-cleaning (curator-cleared in the same test item). TP-005's write/clear
  is a real, intentional, non-reverted side effect (a genuine distillation), scoped to avoid the
  two known-defect entries and the two held-back graph keys. TP-006's two writes are expected to
  be **rejected** and therefore create nothing to clean up (verified by the test item itself).
- No destructive operation in the `guard-destructive-ops.sh` sense (`GRAPH.DELETE`, `FLUSHALL`/
  `FLUSHDB`, volume/container wipes) is used anywhere in this plan — every delete here is the one
  recognized, narrow, per-entry curator-clear shape, run through the MCP tool's own
  authorization, not a raw destructive Redis command.
- TP-009(b)'s isolated scratch tree is created under this session's scratchpad directory and
  discarded afterward — it never touches the live `claude/` tree.

## 8. Entry/exit criteria

**Entry:** Plan gate closed (`docs/reviews/generic-cypher-mcp2.md`, verdict **approve with
suggestions**, 0 blockers); all 21 implementation units delivered per the coordination ledger;
live `kaizen_team` connection confirmed fresh (§4, done).

**Exit:** All twelve test items (TP-001…TP-012) executed and recorded pass/fail/blocked with
evidence in the test report. Any AC where observed live behavior diverges from the requirement,
or any doc/prompt found contradicting actual behavior, is filed as a defect, severity by
user/stakeholder impact — distinguished explicitly from the two known-tracked, already-approved-
pending-fix defects named in the brief.

## 9. Explicitly out of scope

- Re-diffing all 4 populated agents' migrations from scratch (already content-diff-verified per
  unit in the ledger; TP-002 spot-checks one independently rather than repeating all 4).
- Re-running the 16 offline `cypher-mcp` unit tests or the in-container gate (already green,
  `S1`).
- A fresh prompt-quality lint of all 12 agents' full prompts beyond the AC-8/AC-9-relevant scan
  in TP-008/TP-009 — a full lint is `analyst`'s altitude, not this pass's.
- Fuzzing `authorize_write()`'s regex internals beyond AC-6's requirement-level directions.
- Load/concurrency/security testing (§1.1).
- Deciding or executing the fix for the two known-tracked data-fidelity defects, or the
  curator-clear of the two held-back `kaizen_<agent>` keys — both are explicitly gated on a
  stakeholder approval outside this pass's authority; this pass only confirms they remain in the
  documented, non-blocking state (§4).
