# Generic Cypher MCP — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** C-506 (M5)

## Summary

Acceptance pass (unit U7 of `docs/plans/generic-cypher-mcp-coordination.md`) against
`docs/test-plans/generic-cypher-mcp.md` (TP-001…TP-008), executed 2026-08-18. Target: the working
tree at git HEAD `540bb3a` plus the M5 feature's uncommitted changes (`teco` has not yet committed
the delivery — this pass is the last gate before that happens; `git status` at pass time showed
exactly the file set the coordination ledger records for U4/U4-fix/U5/U6/U6-fix, nothing more).
All eight test items were exercised **live**: real `mcp__cpg__query` calls against the running
`cpg` MCP server and the live `kaizen_graph_dba` FalkorDB graph, a real `cobb` subagent dispatch
running its actual documented distillation procedure end to end (a genuine, non-reverted side
effect — one real kaizen entry was promoted and cleared), and direct reads/diffs of the delivered
documentation. No unit tests were re-run (already green and independently verified at U4/U4-fix,
per the coordination ledger) and no code was read as a substitute for driving the running system.

`CPG: considered, not relevant — cpg/mcp/server.py (the code implementing the write path under
test) has no loaded Joern CPG (confirmed live via GRAPH.LIST at pass time: only cpg_falkorchat and
cpg_salesperson are loaded); this is a live behavioral/acceptance pass that drives the running MCP
tool and the running FalkorDB graph directly, not a static code-impact or test-gap question a CPG
would usefully answer.`

**Overall verdict: PASS — all 8 acceptance criteria (AC-1…AC-8) hold under live exercise. No
defects found.** This is an honest clean result, not a rounded-up one: every test item produced
direct, first-party evidence (a real tool-call response, a real file diff, a real graph count, a
real dispatched agent's independently-verified side effect) — nothing here is inferred from
static review or taken on a self-report alone.

## Results table

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-001 | AC-1 | **PASS** | `mcp__cpg__query(graph='kaizen_graph_dba', cypher='MATCH (e:KaizenEntry) RETURN e.date, e.fact, e.evidence, e.context, e.suggestedHome, e.author ORDER BY e.date')`, no `agent` param, called from this QA session (not `graph-dba`) → 6 rows, all fields populated, `41.5ms`–`46.7ms` round trips, no gating. |
| TP-002 | AC-2 | **PASS** | `MATCH (e:KaizenEntry) RETURN count(e)` → `6`. Spot-checked all 6 (not just 1–2) against `git show 540bb3a:claude/graph-dba/kaizen/inbox.md` — every `date`/`fact`/`evidence`/`context`/`suggestedHome` matches the pre-migration markdown verbatim, `author: 'graph-dba'` on all six. |
| TP-003 | AC-3 | **PASS** | `git diff 540bb3a -- claude/graph-dba/kaizen/inbox.md` → exactly 8 inserted lines (the frozen-note blockquote), zero lines removed or altered. Rendered file read in full: the note opens with **"FROZEN — 2026-08-18... no longer appends here... written directly into the graph..."**, unambiguous, before any preserved historical content. |
| TP-004 | AC-4 | **PASS** | Wrote a real `:KaizenEntry` via `mcp__cpg__query(..., agent='graph-dba')` against the *already-existing* graph (the "read-only" branch, not AC-2's "empty key" migration branch) → `write ok (nodes_created=1)`. Independent second read by `entryId` → returned immediately, no rebuild step. `count(e)` → `7`, confirmed exactly one copy. Self-cleaned via curator-clear (`agent='cobb'`) per the plan's explicit allowance → `count(e)` back to `6`, zero residue left in the graph. |
| TP-005 | AC-5 | **PASS** | Real `cobb` subagent dispatch ran the actual 4-step `agent-maintenance` §5 sequence end to end on entry `46825361-…` (the `META_DATA`-absence finding). Independently re-verified, not taken on `cobb`'s word: `claude/graph-dba/kaizen/history.md` has a new, dated 2026-08-18 entry describing the promotion and the ordering followed; `skills/joern-cpg/references/cpg-model.md` carries the promoted caveat, correct and specific; `kaizen_graph_dba` `count(e)` → `5` (was 6); `MATCH (e:KaizenEntry {entryId:'46825361-…'})` → 0 rows. See §"AC-5 detail" below. |
| TP-006 | AC-6 | **PASS** | `agent='graph-dba'` + `author: 'cobb'` literal → rejected: *"this write attributes an entry to author 'cobb', but the call declared agent='graph-dba'... (FR-8)."* Reverse (`agent='cobb'` + `author: 'graph-dba'`) → rejected with the mirrored message. Follow-up `count(e) WHERE entryId STARTS WITH 'ac6-test'` → `0`, confirming neither attempted write executed. |
| TP-007 | AC-7 | **PASS** | `claude/AGENTS.md`:3 explicitly carves `graph-dba` out of the generic inbox convention ("`graph-dba`'s raw capture now writes directly into the `kaizen_graph_dba` FalkorDB graph... its own `kaizen/inbox.md` is a frozen historical snapshot"). `skills/agent-maintenance/SKILL.md` §5 states the graph-backed 4-step sequence for `graph-dba` specifically (lines 320–381). Corroborated further: `claude/graph-dba/graph-dba.md`'s "Learning capture" section (:74–76) and `claude/cobb/cobb.md`'s distillation-duties bullet both correctly describe the split. No unconditional "every agent appends to inbox.md" claim found anywhere read. |
| TP-008 | AC-8 | **PASS** | `grep -m1 -H 'Status:\|Note:' docs/requirements/cpg-query-access.md` → shows both the original `Status: archived` line and the new `**Note:** ... widened by generic-cypher-mcp.md FR-1 — read that document for the current scope...`. No document read in this pass (§2 of the test plan) contradicts this. |

## AC-5 detail (the criterion this pass exists to prove)

Dispatched `cobb` as a real subagent with a brief explicitly framing this as a live
acceptance-test exercise of its own distillation procedure, asking it to pick one genuinely
promotable entry (not a throwaway), run its documented ordering, and report back verifiably.

`cobb`'s own account, independently corroborated:
1. **Read** all 6 raw entries (from its initial full-graph read).
2. **Verified** live before promoting — re-ran `MATCH (n:META_DATA) RETURN count(n))` and
   `CALL db.labels()` against both `cpg_falkorchat`/`cpg_salesperson` on 2026-08-18, confirming the
   original 2026-08-16 finding still holds (and is in fact broader — 5 absent labels, not the 4
   originally named).
3. **Routed and edited** `skills/joern-cpg/references/cpg-model.md` (the knowledge base, per the
   entry's own `suggestedHome`) — **confirmed present and correct by this pass's own direct read**
   (quoted in the results table).
4. **Appended to `history.md`, confirmed the edit succeeded** — **independently confirmed by this
   pass's own direct read** of `claude/graph-dba/kaizen/history.md`'s new top entry.
5. **Only then** ran the curator-clear (`agent='cobb'`) — **independently confirmed** via a live
   `count(e)` drop from 6→5 and a 0-row targeted read for that `entryId`.

`cobb` reported "no deviation from the documented ordering." This pass's independent verification
supports that: both file edits are real, dated, and content-correct, and the graph shows the node
gone — consistent with append-then-delete, not the reverse (a delete-first-then-crash would have
left the graph at 5 with *no* corresponding `history.md`/knowledge-base entry, which is not what
was observed). This pass cannot literally observe the *sequence in which API calls were issued*
(only their end state) — the same structural limit any post-hoc verification of a two-independent-
tool-call sequence has (§9 of the plan doc names this explicitly as procedural, not mechanical,
enforcement) — but the end state is fully consistent with the required ordering and inconsistent
with the one failure mode (double-loss) the ordering exists to prevent, and `cobb`'s own step-by-step
narration named the edits as completed and confirmed before the delete call was issued.

The other 5 entries remain in the graph untouched (`count(e)` = 5, consistent with exactly one
promotion), with `cobb`'s own reasoning for not promoting each recorded in its `history.md` entry
— a legitimate, real distillation decision, not a scripted stand-in.

## Defects

None found. All 8 acceptance criteria hold under live exercise.

## Coverage & gaps

**What this pass covered:**
- Every one of AC-1…AC-8, each with direct, first-party live evidence (a tool-call response, a
  git diff, a file read, or a dispatched agent's independently-re-verified side effect).
- Both write-authorization directions (AC-6) and both write-detection branches distinguished in
  the design — the "empty key" migration branch (already proven live by U5, not re-tested here)
  and the "graph exists" branch (TP-004, tested fresh in this pass).
- A real, non-scripted `cobb` dispatch for AC-5, the one criterion the plan and the brief both
  flag as requiring exactly that.

**What this pass did not cover, deliberately (per the test plan §9), and why that's an acceptable
residual risk:**
- The 16 offline unit tests and the in-container test gate — already green, already independently
  spot-checked by `teco` during U4/U4-fix (two mutation-tests run and reverted). Re-running them
  here would duplicate that gate's job at this altitude, not add new information.
- `authorize_write()`'s regex-internal edge cases beyond the two AC-6 directions (aliasing
  evasion, decoy substrings inside free-text fields, the M1-residual nested-`CREATE` case) — all
  already covered by the unit suite and `teco`'s independent mutation tests; this pass confirms
  the requirement-level behavior, not the regex's internal correctness a second time.
- Concurrency: this pilot has exactly one author (`graph-dba`) and one curator (`cobb`); no
  concurrent-writer scenario was tested, matching the plan's own stated scope (a single-author,
  single-curator pilot) and FR-8's explicit trust bar ("well-behaved callers can't do this by
  accident, not hardened against a malicious one").
- The stretch-goal vector-search capability (FR-7) — explicitly out of scope for this delivery,
  not built, correctly not tested.
- **Genuinely new, not pre-planned:** this pass cannot directly observe the *order in which API
  calls were physically issued* inside `cobb`'s dispatch — only the end state and `cobb`'s own
  narration, since the ordering constraint is procedural (§9 of the plan doc), not mechanically
  enforced by the tool. The end state observed is fully consistent with correct ordering and
  inconsistent with the double-loss failure mode the ordering exists to prevent — a reasonable,
  but not airtight, confirmation. A stronger confirmation would require instrumenting the dispatch
  transcript's tool-call timestamps directly, which this pass did not do (the notification/report
  channel used doesn't expose a raw, independently-timestamped tool-call log to the dispatcher).

## Feedback & recommendations

1. **Not a defect, worth recording as a design observation:** AC-5's ordering constraint is, by
   the plan's own admission (§9), enforceable only by `cobb`'s documented discipline, never by the
   tool. This pass's one real dispatch behaved correctly, but a single successful run is weaker
   long-run assurance than a mechanically-enforced invariant would be. No action needed for this
   delivery (the trade-off was already named and accepted at plan-gate time), but if this pattern
   extends to a second curator agent or a higher-volume distillation cadence, consider whether the
   two-call sequence could be made more directly verifiable after the fact (e.g., a tool-side
   "last write timestamp" queryable independently of the dispatched agent's own narration) rather
   than relying on end-state consistency plus self-report.
2. **The pilot's memory footprint claim (`generic-cypher-mcp-graph.md` §6, "bounded by the
   clear-on-promote design itself") is now live-demonstrated, not just modeled:** this pass
   observed the graph's node count move 6→7→6 (TP-004's self-cleaning write) and 6→5 (TP-005's
   real promotion) within one session — direct behavioral confirmation that the working-memory
   layer genuinely doesn't accumulate, exactly as designed.
3. **Recommend `teco` proceed to commit the M5 delivery** — this was the last open gate (C-506),
   and it closes clean.

## Traceability

Plan: `docs/test-plans/generic-cypher-mcp.md` (TP-001…TP-008). Requirements: `docs/requirements/
generic-cypher-mcp.md` (AC-1…AC-8, FR-1…FR-11). Design: `docs/plans/generic-cypher-mcp.md` §8.2
(the adopted per-AC test strategy), `docs/plans/generic-cypher-mcp-graph.md`. Prior gates: `docs/
reviews/generic-cypher-mcp.md` (plan gate, 3 passes, approve; code re-gate on U4/U6, both approve
with suggestions, both fix-rounds spot-checked and accepted by `teco`). Coordination: `docs/plans/
generic-cypher-mcp-coordination.md`, unit U7 (this report).
