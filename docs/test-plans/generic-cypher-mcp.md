# Generic Cypher MCP — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-506 (M5)

## 1. Scope & objective

Acceptance pass (unit U7 of `docs/plans/generic-cypher-mcp-coordination.md`) for the
`generic-cypher-mcp` feature (M5) — verifying **at behavior/acceptance altitude** that the
delivered system actually produces AC-1…AC-8, by driving the live `mcp__cpg__query` tool, the
live `kaizen_graph_dba` graph, a real `cobb` dispatch running its actual distillation procedure,
and the delivered documentation, and observing what happens — not by re-reading the code or the
already-closed unit tests.

Everything upstream of this pass is already gated and not re-litigated here:
- Plan gate (`docs/reviews/generic-cypher-mcp.md`, 3 passes, final verdict **approve**) — the tool
  mechanism and enforcement design.
- Code re-gate on U4/U4-fix (`server.py`, offline+live test suites, 84 passed/7 deselected) —
  the enforcement logic's unit-level correctness.
- `teco`'s independent verification of U5 (migration: 6 nodes, index+constraint operational,
  `inbox.md` frozen note present) and U6/U6-fix (docs: `claude/AGENTS.md`, `claude/README.md`,
  `claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`, `skills/agent-maintenance/SKILL.md` §5,
  `docs/BACKLOG.md`, both agents' `history.md`).

What none of those gates could confirm — because none of them actually drove the live tool, the
live graph, or a real `cobb` dispatch — is whether the delivered behavior survives contact with
the running system. That is this document's one job, per `docs/plans/generic-cypher-mcp.md` §8.2
("Per-acceptance-criterion strategy (step 5, qa-engineer)"), which is adopted here as the test
strategy rather than re-derived from scratch — this plan follows §8.2's table directly, expanding
each row into a concrete, evidence-producing test item.

## 2. References

- `docs/requirements/generic-cypher-mcp.md` — AC-1…AC-8 (source of test items below), FR-1…FR-11
  for rationale.
- `docs/plans/generic-cypher-mcp.md` §8.2 (per-AC strategy), §3.1–§3.6 (mechanism detail),
  Version 1.2.
- `docs/plans/generic-cypher-mcp-graph.md` — `:KaizenEntry` schema, curator-clear semantics, the
  append-before-delete ordering (context for AC-5).
- `docs/plans/generic-cypher-mcp-coordination.md` — unit ledger, U4/U5/U6 delivered scope and
  `teco`'s independent verification of each.
- `docs/reviews/generic-cypher-mcp.md` — plan-gate and code-re-gate history (not re-litigated).
- `claude/graph-dba/kaizen/inbox.md` — the frozen file under test (AC-3).
- `claude/graph-dba/kaizen/history.md` — the append target under test (AC-5).
- `skills/agent-maintenance/SKILL.md` §5 — the distillation procedure `cobb` is dispatched to run
  for real (AC-5), and the doc text checked for AC-7.
- `claude/AGENTS.md` — checked for AC-7.
- `docs/requirements/cpg-query-access.md` — checked for AC-8.
- Delivered artifact under live test: `cpg/mcp/server.py` (via the running `cpg` MCP server, not
  read directly — this pass drives it, doesn't re-review it).

## 3. CPG relevance check (per this agent's own standing orientation)

Live-checked before writing test items, not assumed: `GRAPH.LIST` (via `redis-cli`, direct host
access) returns exactly `ws:test`, `cpg_falkorchat`, `reference`, `ws:qa-tico-workflows-manual`,
`ws:acme`, `cpg_salesperson`, `ws:eval`, `kaizen_graph_dba` — no `cpg_cpg`/`cpg_mcp`-shaped graph
exists for `cpg/mcp/server.py` itself, confirming live (not just citing the architect's plan note)
that no Joern CPG covers the code under test.

`CPG: considered, not relevant — cpg/mcp/server.py (the code implementing the write path under
test) has no loaded Joern CPG (confirmed live via GRAPH.LIST, this session); this is a live
behavioral/acceptance pass that drives the running MCP tool and the running FalkorDB graph
directly, not a static code-impact or test-gap question a CPG would usefully answer.` No
freshness-marker check applies since no CPG is in use for this pass's own reasoning.

## 4. Live-environment grounding (confirmed before writing test items)

- `mcp__cpg__query` connection is fresh: `MATCH (e:KaizenEntry) RETURN count(e)` against
  `kaizen_graph_dba` returns a real row (`n = 6`), not `graph_not_found_message()` — same check
  `graph-dba` ran in U5 to confirm its own reconnection was genuine.
- `kaizen_graph_dba` currently holds exactly 6 `:KaizenEntry` nodes, one label (`CALL db.labels()`
  → `KaizenEntry` only).
- `claude/graph-dba/kaizen/inbox.md` carries the FROZEN note dated 2026-08-18 at its top, all 6
  original entries preserved below it (confirmed by direct read, §5.2 below has the git-diff
  proof).
- Git HEAD (`540bb3a`) predates this entire delivery — every file this feature touched
  (`cpg/mcp/server.py`, `docs/BACKLOG.md`, `claude/AGENTS.md`, etc.) shows as modified-but-uncommitted
  in `git status`, which is expected: `teco` has not yet committed the M5 delivery, and this
  acceptance pass is the last gate before that happens. `git diff` against HEAD is therefore the
  correct comparison for AC-3 (it shows exactly what this feature changed on `inbox.md`).

## 5. Risk assessment & coverage strategy

The highest risk at this altitude is exactly what no prior gate could see: the live FalkorDB
round-trip (the "empty key" vs. "read-only" classification is a *live* engine behavior, not
something a fake-graph unit test fully stands in for), and — the one criterion the brief and the
plan both flag explicitly — whether `cobb`'s distillation procedure, as actually documented in
`skills/agent-maintenance/SKILL.md` §5, produces the right real-world sequence (`history.md`
append confirmed *before* the graph delete) when a real agent runs it under its own judgment,
not a scripted stand-in.

**Coverage decisions, explicit:**
- AC-1, AC-2, AC-3, AC-6, AC-8 are single, cheap, deterministic live checks — one test item each,
  no sampling question.
- AC-4 exercises the "graph exists" write branch specifically (distinct code path from AC-2's
  "empty key" migration branch already proven by U5) — one write + one independent read.
- AC-5 is the one criterion requiring a **real dispatch**, per the brief and per
  `docs/plans/generic-cypher-mcp.md` §8.2's own words ("this is the criterion the brief flags as
  needing a real acceptance pass, not a unit test") — dispatched once, against one real raw entry,
  because the whole point is a genuine, non-repeatable side effect (an entry actually gets
  promoted and cleared); running it twice would either promote two entries (reducing the graph
  below what AC-2/AC-4 need for a clean read) or be redundant theater on a second, throwaway entry.
- AC-7 is a targeted grep+read, not a full prompt-quality lint — the delivered docs were already
  diff-gated by `analyst` (U6 code re-gate) for wording correctness; this pass's job is only to
  confirm the *live, current* state of those files matches what AC-7 requires, which a targeted
  read settles.

**Deliberately not tested (and why):**
- No load/perf/concurrency/security testing — six kaizen entries, a trusted-identity model
  explicitly scoped to "well-behaved callers can't do this by accident" (FR-8), no concurrent
  writers in this pilot. None of these angles carry real risk here.
- No re-test of the 16 offline unit tests (§8.1 of the plan) or the in-container gate — already
  green per U4/U4-fix, and re-running them would duplicate the code re-gate's job, not this
  pass's.
- No exhaustive fuzzing of `authorize_write()`'s regex edge cases (aliasing evasion, decoy
  substrings, etc.) — already covered by the 16-case unit suite and the two independent-review
  mutation-tests `teco` ran during U4/U4-fix's spot-checks. AC-6 here re-confirms the two
  requirement-level directions live, not the regex's internal edge cases.

## 6. Test items

### TP-001 — AC-1: unprompted, unauthenticated live read

**Preconditions:** `kaizen_graph_dba` live and populated (confirmed §4).

**Steps:** `mcp__cpg__query(graph='kaizen_graph_dba', cypher='MATCH (e:KaizenEntry) RETURN e.date,
e.fact, e.evidence, e.context, e.suggestedHome, e.author ORDER BY e.date')` — no `agent` param,
called from this QA session (not `graph-dba`), i.e. exactly "an agent other than `graph-dba`
queries it."

**Expected result:** Returns all 6 entries with `date`/`fact`/`evidence`/`context`/`suggestedHome`
populated (the same fields today's markdown carries) via a plain graph traversal, including
entries `cobb` has not yet distilled — no error, no gating on identity.

**Priority:** High. **Type:** Acceptance (live tool call).

### TP-002 — AC-2: import completeness + field-fidelity spot-check

**Preconditions:** Same as TP-001.

**Steps:** (a) `mcp__cpg__query(graph='kaizen_graph_dba', cypher='MATCH (e:KaizenEntry) RETURN
count(e) AS n')`. (b) Pick 2 entries from TP-001's result and diff their `fact`/`evidence`/
`context`/`suggestedHome` text against `git show 540bb3a:claude/graph-dba/kaizen/inbox.md`'s
corresponding markdown sections.

**Expected result:** (a) `n = 6`. (b) Both spot-checked entries' graph fields match the
pre-migration markdown verbatim (allowing only the markdown→property structural mapping the
schema defines, e.g. splitting the `## date — fact` heading into `date`+`fact`).

**Priority:** High. **Type:** Acceptance (live tool call + static diff).

### TP-003 — AC-3: `inbox.md` unchanged except the frozen note, and unambiguous

**Preconditions:** None beyond repo access.

**Steps:** (a) `git diff 540bb3a -- claude/graph-dba/kaizen/inbox.md`. (b) `Read` the rendered
file in full.

**Expected result:** (a) The diff shows only an addition (the frozen-note block) — no existing
line altered or removed. (b) The rendered file opens with an unambiguous "this file is frozen,
historical-only, don't append here, go to the graph instead" statement before any of the
preserved historical content.

**Priority:** High. **Type:** Static (git diff) + live-file read.

### TP-004 — AC-4: a fresh author-write against the *existing* graph, then an independent read

**Preconditions:** `kaizen_graph_dba` already exists (exercises the "`ro_query` fails with
'read-only'" branch of `run_query()`, not the "empty key" migration branch AC-2 already proved —
a genuinely different code path per `docs/plans/generic-cypher-mcp.md` §3.1).

**Steps:** (a) `mcp__cpg__query(graph='kaizen_graph_dba', cypher="CREATE (k:KaizenEntry
{entryId: '<fresh uuid4>', date: '2026-08-18', fact: 'QA acceptance-test entry for AC-4 — safe to
ignore/delete, not a real learning', evidence: 'Written live by qa-engineer during the U7
acceptance pass (docs/test-plans/generic-cypher-mcp.md TP-004) to prove the author-write path
against an already-existing graph.', context: 'generic-cypher-mcp U7 acceptance pass', suggestedHome:
'unsure', author: 'graph-dba', createdAt: '<import-run ISO-8601 timestamp>'})", agent='graph-dba')`.
(b) A second, independent read: `MATCH (e:KaizenEntry {entryId: '<same id>'}) RETURN e` (no
`agent`). (c) Confirm no second/duplicate copy exists anywhere else (this graph is the only place
raw entries live post-migration, so "no second copy" reduces to "exactly one node with this
`entryId`" — checked via `count(e)`).

**Expected result:** (a) Write accepted — `format_write_result()` reports a real write (not a
rejection). (b) The second read returns the entry, immediately, with no rebuild/sync step. (c)
Exactly one node with this `entryId` exists. **Then, per the brief's explicit allowance**, this
synthetic entry is curator-cleared by this pass itself (`agent='cobb'`, the one recognized
curator-clear shape) immediately after (b)/(c) are confirmed, so the test leaves zero permanent
graph pollution — logged as a deliberate, self-cleaning side effect, not left implicit.

**Priority:** High. **Type:** Acceptance (live tool call, write + read + cleanup).

### TP-005 — AC-5: a real `cobb` dispatch running the actual four-step distillation sequence

**Preconditions:** At least one raw `:KaizenEntry` remains in `kaizen_graph_dba` genuinely worth
promoting (6 candidates after TP-002, unaffected by TP-004's self-cleaning synthetic entry).
`skills/agent-maintenance/SKILL.md` §5 carries the documented 4-step sequence (confirmed present,
§4).

**Steps:** Dispatch `cobb` as a real subagent (`Agent` tool) with a brief stating: this is a live
acceptance-test exercise of `cobb`'s own graph-backed distillation procedure
(`agent-maintenance` skill §5) for the `generic-cypher-mcp` feature's AC-5; pick **one** of the 6
real migrated `graph-dba` entries, ideally one genuinely promotable (not throwaway test noise —
this is real distillation of real backlog knowledge, a legitimate side effect); run its own
documented ordering exactly (read → verify → append to `history.md`, confirm the edit succeeded →
only then curator-clear the graph node, `agent='cobb'`); report back what it read, verified,
appended, and cleared, including confirmation the `history.md` edit succeeded *before* the delete
ran.

**Expected result:** `cobb`'s report shows all four steps executed, in order, with the ordering
constraint honored (append confirmed before delete). Independently verified afterward by this
pass (not taken on `cobb`'s word alone): (a) `claude/graph-dba/kaizen/history.md` has a new,
dated entry matching the promoted fact; (b) `kaizen_graph_dba` now has 5 `:KaizenEntry` nodes, not
6 (`MATCH (e:KaizenEntry) RETURN count(e)`); (c) the specific promoted `entryId` is gone from the
graph (`MATCH (e:KaizenEntry {entryId: '<id>'}) RETURN e` → 0 rows).

**Priority:** Critical — the one criterion this whole pass exists to prove cannot be satisfied any
other way. **Type:** Acceptance/e2e (real subagent dispatch, real durable side effect).

### TP-006 — AC-6: cross-attribution rejected, both directions

**Preconditions:** None beyond a live connection.

**Steps:** (a) `mcp__cpg__query(graph='kaizen_graph_dba', cypher="CREATE (k:KaizenEntry {entryId:
'<uuid>', date: '2026-08-18', fact: 'AC-6 negative-test — should be rejected', evidence: 'n/a',
context: 'n/a', suggestedHome: 'unsure', author: 'cobb', createdAt: '<ts>'})",
agent='graph-dba')` — declares `agent='graph-dba'` while the `author:` literal claims `'cobb'`.
(b) The reverse: same shape, `author: 'graph-dba'`, `agent='cobb'`.

**Expected result:** Both calls are rejected by `authorize_write()` before `target.query()` is
ever invoked (no partial write) — the tool's rejection message states the author/agent mismatch
explicitly (FR-8). A follow-up count check confirms neither attempted node was created.

**Priority:** High. **Type:** Acceptance (live tool call, negative test, both directions).

### TP-007 — AC-7: no doc still describes `graph-dba`'s raw capture as appending to `inbox.md`

**Preconditions:** None beyond repo access.

**Steps:** `Read` `claude/AGENTS.md` in full and `skills/agent-maintenance/SKILL.md` §5 in full
(already read in this session's context-gathering pass — re-confirmed here as the test item, not
re-derived). Scan both for any sentence describing `graph-dba`'s *current* raw-capture behavior as
appending to `kaizen/inbox.md`.

**Expected result:** `claude/AGENTS.md` explicitly carves `graph-dba` out of the generic
"`inbox.md` is the agent's own append-only learnings capture" statement, naming the graph
instead. `skills/agent-maintenance/SKILL.md` §5 states the graph-backed capture/distillation
sequence for `graph-dba` specifically (schema, the 4-step ordering) with `graph-dba`'s
`kaizen/inbox.md` described as a frozen historical snapshot, not a live write target. No
unconditional "every agent appends to its inbox.md" claim remains uncarved.

**Priority:** Medium (already diff-gated once at U6; this is a live-state confirmation, not a
fresh review). **Type:** Static document read.

### TP-008 — AC-8: the requirements-pointer note, no contradiction elsewhere

**Preconditions:** None beyond repo access.

**Steps:** `grep -m1 -H 'Status:\|Note:' docs/requirements/cpg-query-access.md`.

**Expected result:** The header block shows both the original `Status:` line and the new
`**Note:**` line pointing at `generic-cypher-mcp.md` FR-1's supersession of the "Non-CPG graphs /
general agent access to FalkorDB" out-of-scope line. No other document found (in the reading done
for this pass, §2's references) contradicts this — i.e. nothing still asserts non-CPG graph
access is out of scope for the whole repo.

**Priority:** Medium. **Type:** Static, one command.

## 7. Environment & data setup

- No environment bring-up needed — the shared `falkordb-dev` instance is already running with
  `kaizen_graph_dba` live and populated (confirmed §4), and the `cpg` MCP server connection is
  already fresh (confirmed §4).
- TP-004 and TP-006 write to the live graph; TP-004's write is a self-cleaning synthetic test
  entry (curator-cleared by this pass in the same test item); TP-006's two writes are expected to
  be **rejected** and therefore create nothing to clean up (verified by the test item itself).
- TP-005 is the one test item with a real, intentional, non-reverted side effect: one real
  `graph-dba` kaizen entry gets genuinely promoted to `history.md` and cleared from the graph.
  This is by design (per the brief: "a legitimate side effect, not just theater") and is the
  expected end-state, not cleanup debt.
- No destructive operation in the `guard-destructive-ops.sh` sense (`GRAPH.DELETE`, `FLUSHALL`/
  `FLUSHDB`, volume/container wipes) is used anywhere in this plan — TP-004/TP-005's
  `DETACH DELETE` calls are the one recognized, narrow, per-entry curator-clear shape, run through
  the MCP tool's own authorization, not a raw destructive Redis command.

## 8. Entry/exit criteria

**Entry:** Plan gate closed (`docs/reviews/generic-cypher-mcp.md`, verdict **approve**); U4/U4-fix
and U6/U6-fix both closed and independently verified by `teco` per the coordination ledger; live
`kaizen_graph_dba` connection confirmed fresh (§4, done).

**Exit:** All eight test items (TP-001…TP-008) executed and recorded pass/fail/blocked with
evidence in the test report. Any AC where observed live behavior diverges from the requirement is
filed as a defect, severity by user/stakeholder impact.

## 9. Explicitly out of scope

- Re-running the 16 offline unit tests or the in-container gate (already green, U4/U4-fix,
  independently re-verified by `teco`).
- A fresh prompt-quality lint of the six U6/U6-fix-edited docs — already diff-gated; TP-007 checks
  their *current live state* against AC-7's specific claim, not their overall prose quality.
- Fuzzing `authorize_write()`'s regex internals beyond the requirement-level AC-6 directions —
  already covered by the 16-case unit suite plus `teco`'s two independent mutation-tests during
  U4/U4-fix.
- Testing the stretch-goal vector-search capability (FR-7) — explicitly not required for this
  delivery, not built.
- Load/concurrency/security testing — no real risk at this pilot's scale and trust model (§5).
