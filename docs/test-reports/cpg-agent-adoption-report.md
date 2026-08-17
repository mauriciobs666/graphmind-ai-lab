# CPG agent adoption — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** cpg-agent-adoption (M4)

## Pass 1 — Acceptance pass (U6)

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

---

## Pass 2 — Live-dispatch re-pass (U9)

### Summary

Follow-up live-dispatch acceptance re-pass (U9 of `docs/plans/cpg-agent-adoption-coordination.md`),
executed 2026-08-16 against HEAD `4780a3a` — one commit ahead of `59af4df` (the U7-fix commit under
test; the extra commit only adds the U9 dispatch row to the coordination ledger). Two prompt-wording
fix rounds landed on the wiring Pass 1 tested: U7 (`bafc3a7`) and U7-fix (`59af4df`), both
diff-gated by `analyst` (`docs/reviews/cpg-agent-adoption.md` Pass 3, "approve with suggestions").
Re-confirmed via direct `grep` that `claude/{coder,architect,tdd-engineer}/*.md` carry the exact
wording Pass 3 reviewed — no drift between the diff gate and this dispatch.

Re-ran the §4-equivalent live grounding before dispatching (not assumed unchanged): `GRAPH.LIST`
still shows `cpg_salesperson` and `cpg_falkorchat` loaded; `MATCH (b:CpgBuildInfo) RETURN
b.BUILT_AT, b.SOURCE_PATH, b.SOURCE_COMMIT, b.SOURCE_DIRTY` against both still returns **zero
rows**. Same "no marker at all" condition as Pass 1 — this re-pass again cannot exercise AC-4's
positive/actionable branch (a genuinely stale, populated marker), only the harder "unknown, stale
by convention" edge, which is in fact the more demanding case for DEF-2's specific failure mode.

Three re-dispatches, same shape/difficulty as the originals, different specific target
functions/questions per the task brief (an identical replay would be weaker evidence that a
wording fix generalizes than a same-shape-different-specifics rerun): D1′ `coder` on
`Repository.materialize_snapshot` (`falkor-chat/server`, `cpg_falkorchat`); D2′ `architect` on
`handle_submit` (`salesperson/chatbot.py`, `cpg_salesperson`); D3′ `tdd-engineer` on the
match-and-launch logic in `mcp_monitor/launcher.py` (`mcp-monitor/`, no loaded CPG). All three
dispatch prompts were investigation-only, no-CPG-mention, no-file-edit, matching Pass 1's
discipline; all three subagents confirmed no files were edited.

`CPG: not applicable — this re-pass verifies agent-prompt wiring behavior, same reasoning as
Pass 1 (§3 of the test plan); the two live CPGs are the targets the dispatched subagents were
tested against, not something this pass itself queried for its own authoring.`

**Overall verdict: PASS, with one new minor finding (DEF-4).** DEF-1 and DEF-2 close cleanly —
both show a clear behavioral change in the *opposite* direction from the original failure, not
just a differently-worded restatement of the same gap. DEF-3's core failure (total silence on
CPG in any form) also closes — a `CPG:` line is now present — but the re-pass surfaces a new,
narrower defect: the line's shape doesn't match the plan's own worked example for this exact
scenario. This is the honest reading of a partial result, not a summary rounded up to "clean
pass": two of three defects are unambiguously closed, the third trades a major silent-gap defect
for a minor wording-precision one.

### Results table

| ID | Defect | Dispatch | Result | Evidence |
|---|---|---|---|---|
| DEF-1 re-check | `coder` omits literal `CPG:` line | D1′ (`coder`, falkor-chat) | **CLOSED** | See below — literal, correctly-shaped `CPG:` line present as the deliverable's closing line. |
| DEF-2 re-check | `architect` skips freshness check | D2′ (`architect`, salesperson) | **CLOSED** | See below — freshness check demonstrably run *before* the cross-verification decision, reversing the original failure's exact sequencing. |
| DEF-3 re-check | `tdd-engineer` zero CPG mention | D3′ (`tdd-engineer`, mcp-monitor) | **CLOSED (silence), new minor DEF-4 (shape)** | See below — a `CPG:` line is present (silence resolved) but uses the wrong one of the three convention shapes. |

### DEF-1 re-check — `coder` (D1′) — CLOSED

**Dispatch:** "In `falkor-chat/server`, `Repository.materialize_snapshot`... suppose we changed
its signature — specifically, added a new required parameter. What would break?" (no CPG mention,
investigation-only).

**Evidence — D1′'s closing line, quoted verbatim:**

> `CPG: considered, not relevant — cpg_falkorchat exists but has no freshness marker (zero
> CpgBuildInfo rows), so direct source inspection was used instead of the graph for this trace.`

This is a literal, correctly-shaped `CPG:` line, present as the deliverable's last line — exactly
what DEF-1 found missing in Pass 1 (D1 had used the CPG internally but never emitted a matching
literal line, e.g. writing "**CPG freshness note:**" instead of "`CPG:`"). D1′ took a different,
and arguably more disciplined, path than D1: rather than using the CPG and grep-corroborating it
while forgetting the evidence line, D1′ ran the discovery check, found the CPG present but with no
freshness marker, judged that "a reason for caution" (per the freshness recipe's own documented
guidance for the zero-row case), used direct source inspection instead, and explicitly surfaced a
refresh suggestion: *"If this component gets a CPG-driven review again, it's a candidate for a
`graph-dba` rebuild first."* This satisfies AC-2 (line present, correctly shaped, `grep`-able),
AC-3 (freshness signal communicated), and AC-4 (refresh suggestion surfaced, no silent trust, no
self-triggered rebuild) in one pass. DEF-1's specific failure mode — correct internal behavior,
missing/mismatched literal line — does not reproduce.

### DEF-2 re-check — `architect` (D2′) — CLOSED

**Dispatch:** "In `salesperson/chatbot.py`... suppose we changed [`handle_submit`'s] signature —
specifically, added a new required parameter... What would need to change elsewhere?" (no CPG
mention, investigation/plan-sketch only).

**Evidence — D2′'s closing "CPG" paragraph, quoted verbatim:**

> `CPG: used cpg_salesperson — confirmed handle_submit has exactly one caller (chatbot.py:<module>,
> i.e. the top-level script body at line 95) via MATCH (m:METHOD {NAME:"handle_submit"}) and a
> CALL edge lookup, corroborating the rg grep. Ran the freshness check (MATCH (b:CpgBuildInfo)
> RETURN ...) as required before trusting the graph: zero rows — this CPG predates the
> freshness-stamping feature (or its build never reached that step), so there is no
> builtAt/sourceCommit signal at all. Given that, I cross-verified the CPG's answer against rg
> directly rather than relying on it alone, and the two agree. For anything beyond this
> single-symbol lookup..., I'd suggest asking graph-dba to rebuild cpg_salesperson first so future
> queries carry a real freshness signal.`

This is the exact inversion of the original DEF-2 failure. D2 (Pass 1) explicitly *skipped* the
freshness check and used grep/CPG agreement as a post-hoc rationale for not needing it ("I did not
run a freshness check... a staleness gap here would have shown up as a grep/CPG mismatch, and
there was none"). D2′ ran the freshness query first, as the tightened instruction requires ("query
the freshness marker... in that same tool call/step, before you decide whether the CPG's answer
needs further cross-verification"), found zero rows, and *then* used the cross-verification as the
correct response to that signal — not a substitute for running it. It also surfaces an explicit
refresh suggestion (AC-4). AC-2/AC-3/AC-4 all hold cleanly. DEF-2's specific failure mode — the
freshness check treated as an optional, reasoned-around step — does not reproduce; the sequencing
the U8 diff gate's pronoun-ambiguity fix was written to force ("skipping it in favor of a
substitute check... doesn't satisfy this") held up against a real dispatch's own judgment call.

### DEF-3 re-check — `tdd-engineer` (D3′) — CLOSED (silence), new minor DEF-4 (shape)

**Dispatch:** "In `mcp-monitor/`, what test coverage exists for the match-and-launch logic in
`mcp_monitor/launcher.py`... What gaps are there?" (no CPG mention, investigation-only).

**Evidence — D3′'s closing line, quoted verbatim:**

> `CPG: not applicable — this task is Python test-coverage investigation in a repo with no
> cpg_mcp-monitor graph indicated by any project doc, and the question is answerable directly from
> source/tests without call-graph or data-flow queries.`

The original failure — D3's full response containing zero occurrences of "CPG" in any form — does
not reproduce: a stakeholder running `grep -i "^CPG:"` against this transcript now gets a match.
That is DEF-3's core mechanism (AC-2's anti-silence guarantee) working.

However, this is the wrong one of the plan's three defined shapes. Per `docs/plans/
cpg-agent-adoption.md` §3, `CPG: not applicable` is defined for "tasks with no code-level component
at all" — and the plan's own worked example for the *opposite* case (a component that does have
code, but no loaded CPG) is the `considered, not relevant` shape, verbatim: *"`CPG: considered, not
relevant — this change is in opencode/, which has no loaded CPG`"*. D3′'s task — test-gap analysis
of real production code in `mcp_monitor/launcher.py` — is unambiguously a code-level task; per the
plan's own definition and its own nearly-identical worked example, this should have been `CPG:
considered, not relevant — ...`, not `CPG: not applicable — ...`. Filed as new minor defect DEF-4
below.

### DEF-4 (new) — `tdd-engineer` selects the wrong `CPG:` shape for a no-CPG-loaded-component task

**Severity:** Minor (the anti-silence mechanism AC-2 exists to guarantee is intact — a broad
`grep "^CPG:"` finds a line; only a shape-specific spot-check, e.g. `grep "considered, not
relevant"`, would miss it).

**AC affected:** AC-2 (shape precision, not the presence/silence guarantee).

**Steps to reproduce:** Dispatch `tdd-engineer` on a small, unprompted, read-only-safe,
code-level task against a component with no loaded CPG (exact prompt used: "in `mcp-monitor/`...
what test coverage exists for the match-and-launch logic in `mcp_monitor/launcher.py`... what
gaps?", no CPG mention). Observe the final `CPG:` line's shape.

**Expected result:** Per `docs/plans/cpg-agent-adoption.md` §3's own worked example for this exact
scenario: `CPG: considered, not relevant — <clause>` (e.g. "mcp-monitor has no loaded CPG").

**Actual result:** `CPG: not applicable — this task is Python test-coverage investigation in a
repo with no cpg_mcp-monitor graph indicated by any project doc...` — the shape defined for tasks
with *no code-level component at all*, which does not describe this task.

**Evidence:** D3′'s full closing line, quoted above.

### Coverage & gaps (Pass 2)

**What this re-pass covered:** the same three AC-2/AC-3/AC-4 gaps Pass 1 found, on the same three
agents, against comparable but not identical tasks — DEF-1 and DEF-2 confirmed closed with direct
behavioral evidence (not just re-worded prompts); DEF-3 confirmed closed on its core failure mode
(silence) but with a new, narrower shape-selection defect (DEF-4) discovered in the process.

**What this re-pass did not cover (carried forward from Pass 1, still open):**
- No scenario with a populated `:CpgBuildInfo` marker — both live graphs still return zero rows
  (re-confirmed live, unchanged since Pass 1). AC-4's positive/actionable branch (an agent that
  finds a genuinely stale marker and surfaces a concrete refresh suggestion naming the graph, as
  distinct from the "no marker at all" case both this pass and Pass 1 exercised) has still never
  been observed in a real dispatch, on this feature or any prior gate. This is the same coverage
  gap Pass 1 flagged (Feedback #3) and it remains open — recommend the same follow-up once either
  live graph picks up a real rebuild.
- `qa-engineer` (self) and `frontend-engineer` were still not separately dispatched — same sampling
  rationale as Pass 1 (§5 of the test plan); this residual risk is unchanged by this re-pass.
- DEF-4 itself is new evidence that a residual "the untested two could show a distinct failure
  mode" (Pass 1's own words) risk is not hypothetical — three re-dispatches surfaced a fourth
  distinct defect shape not predicted by Pass 1's findings.

### Feedback & recommendations (Pass 2)

1. **DEF-1 and DEF-2's fixes generalize, not just replay.** Both re-dispatches used different
   target functions than the originals and produced behavior that is the clear opposite of the
   original failure mode, not a coincidental pass. This is reasonably strong evidence the U7/U7-fix
   wording changes are doing real work, not just happening to match one specific prompt.
2. **DEF-4 suggests the three-shape `CPG:` convention could use one more worked example.** The
   plan's §3 gives one example each for `used` and `considered, not relevant`, and a
   parenthetical scope note for `not applicable` ("rare... keeps the convention total") but no
   worked example for it. An agent choosing between "considered, not relevant" and "not
   applicable" for a component-with-no-CPG case has exactly one relevant example to pattern-match
   against (the `opencode/` one, which does read as "considered, not relevant") — D3′ still
   picked the other shape, suggesting the boundary between "no code-level component" and "code
   component, but no CPG for it" isn't as self-evident from the current wording as the single
   example assumes. Recommend `cobb` consider either tightening the `not applicable` definition
   with a counter-example of what it is *not* for, or accepting this as a low-severity, rare
   edge case not worth further prompt surface.
3. **The "no populated freshness marker has ever been observed in a live dispatch" gap is now two
   passes old.** Recommend this graduate from a report footnote to an actual follow-up trigger —
   e.g. a note on the next `graph-dba` rebuild of either graph to ping `qa-engineer` for a
   targeted AC-4-positive-branch dispatch while the marker is fresh, before it goes stale again.
4. **Not a defect, but worth recording:** this re-pass is itself a second concrete demonstration of
   the same principle Pass 1's Feedback #4 recorded — static diff review (Pass 3's gate) correctly
   identified the fix wording as well-targeted and traceable to the report's own recommendations,
   but could not and did not claim the fix would survive a live agent's own judgment call. It
   didn't, not fully (DEF-4). Two data points now support treating "diff-gated" and
   "behaviorally-confirmed" as genuinely separate claims for this class of feature, not a formality
   layered on top of the same information.

### Traceability (Pass 2)

Plan: `docs/test-plans/cpg-agent-adoption.md` §10 (re-pass addendum). Requirements: `docs/
requirements/cpg-agent-adoption.md` (AC-1…AC-6). Prior gates: `docs/reviews/
cpg-agent-adoption.md` Pass 3 (U7 fix-round diff gate — approve with suggestions). This pass:
`docs/plans/cpg-agent-adoption-coordination.md` unit U9.
