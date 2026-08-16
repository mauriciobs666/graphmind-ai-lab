# CPG agent adoption — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** cpg-agent-adoption (M4)

## Summary

Acceptance pass (U6) against `docs/test-plans/cpg-agent-adoption.md`, executed 2026-08-16 against
HEAD `c84815c` (the U5 code-gate commit; feature commits `35b108f` + `50f9aaa` unchanged since).
Three live subagent dispatches (D1 `coder`, D2 `architect`, D3 `tdd-engineer`) were run against
real, small, read-only-safe code-level tasks — two against components with a loaded CPG
(`cpg_falkorchat`, `cpg_salesperson`), one against a component with none (`mcp-monitor/`) — plus a
document check for AC-6. No files were mutated by this pass; every direct FalkorDB query used
(`GRAPH.LIST`, `MATCH … RETURN`) was read-only, and all three dispatch prompts explicitly required
investigation-only, no-edit behavior.

`CPG: not applicable — this acceptance pass verifies agent-prompt wiring behavior; the two live
CPGs are the *targets* the dispatched subagents were tested against, not something this test pass
itself queried for its own authoring (see test plan §3).`

**Overall verdict: FAIL — the discovery mechanism (AC-1) and the reconciliation statement (AC-6)
hold, but the evidence-trail and freshness-surfacing behavior (AC-2, AC-3, AC-4) show reproducible
gaps in all three live dispatches, each a different failure mode.** This is precisely the class of
gap the U5 diff gate explicitly could not check ("prompt-level commitments, not runtime-observable
from a static diff read... genuine behavioral confirmation is U6's job") — the wiring is present
and correctly worded (both static gates verified that twice), but does not yet reliably survive
contact with a real dispatched agent making its own judgment calls under an actual task. AC-5 gets
a weak, unconfirmable pass (see TP-005).

## Results table

| ID | AC(s) | Dispatch | Result | Evidence |
|---|---|---|---|---|
| TP-001 | AC-1 | D1 (`coder`, falkor-chat) | **PASS** | D1 discovered `cpg_falkorchat` unprompted (never named in the prompt) and used it for the impact-analysis question — see D1 transcript, "I checked for a loaded CPG (`cpg_falkorchat`) and used it for the call-graph query." |
| TP-002 | AC-2/AC-3/AC-4 | D1 (`coder`) | **PARTIAL — AC-2 FAIL, AC-3 PASS, AC-4 PARTIAL** | See DEF-1, DEF-2-adjacent note below. AC-3: freshness signal communicated (0-row marker + git-log corroboration). AC-2: no literal `CPG:` line emitted. AC-4: caution surfaced and no silent trust/rebuild, but no explicit refresh *suggestion* as the convention requires. |
| TP-003 | AC-1 | D2 (`architect`, salesperson) | **PASS** | D2 discovered `cpg_salesperson` unprompted and used it — "`CPG: used cpg_salesperson — MATCH (m:METHOD {NAME:"write_message"})...`". |
| TP-004 | AC-2/AC-3/AC-4 | D2 (`architect`) | **PARTIAL — AC-2 PASS, AC-3 FAIL, AC-4 FAIL** | See DEF-2. AC-2's line is present and correctly shaped. AC-3/AC-4 fail: D2 explicitly declined to run the freshness check ("I did not run a freshness check... a staleness gap here would have shown up as a grep/CPG mismatch, and there was none"). |
| TP-005 | AC-5 | D3 (`tdd-engineer`, mcp-monitor) | **WEAK/UNCONFIRMED PASS** | No observable delay or narrated friction (53s, 7 tool calls, task-proportionate) — but see TP-006: zero evidence the discovery check ran at all, so a "no noise because it's cheap" reading and a "no noise because it never ran" reading are equally consistent with the transcript. Not falsifiable as tested; flagged, not counted as a clean pass. |
| TP-006 | AC-2 (not-relevant shape) | D3 (`tdd-engineer`) | **FAIL** | See DEF-3. D3's full final response contains zero occurrences of "CPG" in any form — no discovery narration, no `CPG: considered, not relevant` line, nothing. |
| TP-007 | AC-6 | Document check | **PASS** | `docs/plans/cpg-agent-adoption.md` §4 states, in its own words, that the plan "extends — it does not override, narrow, or silently diverge from" the M2/M3 consumer-scope boundary, with four supporting bullets. Already independently re-verified twice by the two prior gates (Pass 1 open-questions discussion, Pass 2 point 3); this pass adds a third confirmation, not a fresh review. |

## Defects

### DEF-1 — `coder` (D1) omits the required `CPG:` evidence-trail line despite genuinely using the CPG

**Severity:** Moderate (content is present and correct; the specific spot-check mechanism AC-2 is built around fails).

**AC affected:** AC-2.

**Steps to reproduce:** Dispatch `coder` on a small, unprompted impact-analysis task against a
component with a loaded CPG (exact prompt used: "in `falkor-chat/server`... if `Services.
post_message`'s signature changed... what would break?", no mention of CPG anywhere).

**Expected result:** Per `claude/coder/coder.md` step 5 ("Verify and report"): *"Include a `CPG:`
line — exactly one of `CPG: used <graph> — <clause>` / `CPG: considered, not relevant — <clause>`
/ `CPG: not applicable — <clause>`."*

**Actual result:** D1's response demonstrably used `cpg_falkorchat` (narrated in prose: "I checked
for a loaded CPG (`cpg_falkorchat`) and used it for the call-graph query, then cross-verified every
result with a direct `grep`/`Read` pass"; also "**CPG freshness note:** `cpg_falkorchat` has no
`CpgBuildInfo` marker...") but never emits a line matching the literal convention shape `CPG: used
<graph> — <clause>`. A stakeholder running `grep -i "^CPG:"` or even `grep "CPG:"` against this
transcript gets **zero matches**, despite the agent having done exactly the right thing internally.

**Evidence:** Full D1 transcript (subagent `a8f33c578b855e444`), reproduced in this pass's session.

### DEF-2 — `architect` (D2) treats the mandated freshness check as optional and explicitly skips it

**Severity:** Major (the plan's own design explicitly rules this out: *"run the freshness check...
as part of that same step — not a separate, optional pass"* — `cpg-agent-adoption.md` §2.3 — and
D2 did exactly the thing that sentence forbids, with a stated rationale).

**AC affected:** AC-3 (no currency signal communicated at all), cascading into AC-4 (no staleness
judgment possible without a signal, so no refresh suggestion either).

**Steps to reproduce:** Dispatch `architect` on a small, unprompted impact-analysis task against a
component with a loaded CPG (exact prompt used: "in `salesperson/chatbot.py`... suppose we
changed `write_message`'s signature... what would break?", no mention of CPG anywhere).

**Expected result:** Per `claude/architect/architect.md`'s "Investigate the codebase first" step
(reworded at C-403 per the plan): when a relevant CPG is used, also run the freshness check
(`skills/cpg-analysis/references/freshness.md`) "as part of that same step," note what it says, and
surface a refresh suggestion if it looks stale.

**Actual result:** D2 used the CPG correctly (AC-1/AC-2 both hold for this dispatch — see TP-003/
TP-004) but its response states explicitly: *"I did not run a freshness check against current file
mtimes since this is a small, single-function query fully corroborated by grep — a staleness gap
here would have shown up as a grep/CPG mismatch, and there was none."* This is a reasoned judgment
call, not an oversight — but it is exactly the "separate, optional pass" framing the plan's wording
was written to prevent, and the result is that no currency signal (marker present/absent, age,
`sourceCommit`) reaches the deliverable at all, which is what AC-3 literally requires. Note also
that D2's own cross-check reasoning ("would have shown up as a grep/CPG mismatch") wouldn't
actually catch a **stale-but-consistent** case — a CPG that's stale but happens to still agree with
current source on this one function wouldn't produce a mismatch either, so the substitute check
doesn't fully cover what the freshness check would have.

**Evidence:** Full D2 transcript (subagent `a7901e7f0ce7e3856`).

### DEF-3 — `tdd-engineer` (D3) produces zero evidence of the CPG discovery step on a no-CPG component

**Severity:** Major (this is the exact silent-gap AC-2 exists to eliminate: *"silence is what this
convention rules out, not brevity"* — `cpg-agent-adoption.md` §3).

**AC affected:** AC-2 (not-relevant shape), and indirectly AC-5 (cannot confirm the check happened
cheaply if there's no evidence it happened at all).

**Steps to reproduce:** Dispatch `tdd-engineer` on a small, unprompted test-gap task against a
component with **no** loaded CPG (exact prompt used: "in `mcp-monitor/`... what test coverage
exists for the regex-matching logic in `mcp_monitor/config.py`... what gaps?", no mention of CPG
anywhere).

**Expected result:** Per `claude/tdd-engineer/tdd-engineer.md`'s "Understand first" step (wired at
C-404) and the evidence-trail convention (C-406): the agent's normal orientation includes a
discovery check, and its deliverable includes a `CPG: considered, not relevant — <clause>` line
(e.g., "mcp-monitor has no loaded CPG").

**Actual result:** D3's full final response — a thorough, well-organized test-gap analysis — never
mentions "CPG," "cpg-analysis," `mcp__cpg__query`, or FalkorDB in any form. There is no way to
distinguish, from the transcript alone, whether the discovery check ran and correctly found no
match (the intended AC-5 behavior) or never ran (a wiring failure) — which is precisely the
ambiguity the evidence-trail convention was designed to close for a stakeholder spot-check.

**Evidence:** Full D3 transcript (subagent `a4df0e658a260e258`).

## Coverage & gaps

**What this pass covered:**
- AC-1 (unprompted discovery when a CPG exists) — exercised twice, on two different agents and two
  different components/graphs. Both held cleanly.
- AC-2/AC-3/AC-4 (evidence trail + freshness signal + stale-surfacing) — exercised twice, on the
  same two dispatches. Neither dispatch cleanly satisfied all three; each failed a different piece
  (DEF-1: format; DEF-2: the check itself was skipped).
- AC-5 (no-op cost on a CPG-absent component) — exercised once. The "no delay" half is plausible
  but unconfirmable given DEF-3.
- AC-6 (extends-not-overrides statement) — document-existence check, third independent
  confirmation across the feature's three gates.

**What this pass did not cover (deliberately, per the test plan's §5/§9, or discovered as a gap
during execution):**
- No scenario with an actual populated `:CpgBuildInfo` marker (non-trivial age, `sourceCommit`
  set) — both live graphs still return zero rows (confirmed live before and unchanged during this
  pass). This means AC-4's *positive* branch — an agent that finds a marker, judges it stale by
  the recipe's two-check method, and surfaces a concrete refresh suggestion naming the graph — has
  **never been observed in a real dispatch**, on this feature or any prior gate. Both live graphs'
  current zero-row state exercises only the "no marker at all" edge of AC-3/AC-4, not the "marker
  present, genuinely old" case the freshness mechanism's more interesting logic (age vs. actual
  git-log movement) is built for.
- `qa-engineer` (self) and `frontend-engineer` were not separately dispatched, per the test plan's
  stated sampling rationale (§5) — this is a residual risk, not a verified pass, for those two
  agents' actual live behavior specifically. Given that 3/3 tested dispatches each showed a
  distinct evidence-trail/freshness gap, this residual risk should not be assumed low by extension
  — the untested two could just as plausibly show a fourth and fifth distinct failure mode.
- No test of what happens when a *staged/dirty* freshness marker is judged (`SOURCE_DIRTY: true`)
  — moot until a graph carries one at all.

## Feedback & recommendations

1. **The "not a separate, optional pass" instruction, as currently worded, did not survive an
   agent's own judgment call in a live dispatch (DEF-2).** `architect` reasoned its way past the
   freshness check using a plausible-sounding but incomplete substitute (grep/CPG agreement doesn't
   rule out "stale but coincidentally still correct on this one function"). Recommend `cobb`
   consider strengthening the wording from an instruction an agent can reason around to something
   closer to a hard sequencing rule ("query the freshness marker in the same tool call/step you
   query for the answer — before deciding whether the answer needs cross-verification, not after"),
   or accept the current design's latitude but explicitly document the tradeoff.
2. **The evidence-trail line's exact literal format (`CPG: <shape> — <clause>`) is not being
   reliably emitted even when the underlying behavior is correct (DEF-1) or produces zero signal
   even when the behavior may be correct (DEF-3).** Since AC-2's whole mechanism is a stakeholder
   `grep`-ing a transcript, an agent that does the right thing but phrases it as "CPG freshness
   note:" instead of "CPG:" defeats the mechanism just as effectively as one that skips the step
   entirely. Consider whether a stronger anchor (e.g., requiring the line be the literal last line
   of the deliverable, or requiring it verbatim rather than "include a line matching this shape")
   would survive paraphrase-under-task-pressure better.
3. **Recommend a follow-up live dispatch once either live CPG picks up a real `:CpgBuildInfo`
   marker** (i.e., after either graph's next on-demand rebuild) to close the coverage gap noted
   above — AC-4's positive/actionable branch has genuinely never been observed end-to-end.
4. **Not a defect, but worth recording:** this pass is itself the concrete demonstration of why the
   U5 gate correctly deferred behavioral confirmation to U6 — all three findings here were
   invisible to a diff read (the prompt wording that produced them is exactly the wording U5
   verified as "correct" and "matching the plan almost word-for-word"). Static prompt-wiring review
   and live-dispatch behavior are genuinely different failure surfaces for this class of feature.

## Traceability

Plan: `docs/test-plans/cpg-agent-adoption.md` (TP-001…TP-007). Requirements: `docs/requirements/
cpg-agent-adoption.md` (AC-1…AC-6). Prior gates: `docs/reviews/cpg-agent-adoption.md` (Pass 1 plan
gate — approve with suggestions; Pass 2 diff gate — approve). Coordination: `docs/plans/
cpg-agent-adoption-coordination.md` (unit U6, this report).
