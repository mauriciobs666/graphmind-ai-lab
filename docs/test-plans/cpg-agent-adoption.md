# CPG agent adoption — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** cpg-agent-adoption (M4)

## 1. Scope & objective

Acceptance pass (unit U6 of `docs/plans/cpg-agent-adoption-coordination.md`) for the
`cpg-agent-adoption` feature (M4) — verifying **at behavior/acceptance altitude** that the
delivered prompt wiring actually produces the behavior AC-1…AC-6 require, by dispatching real
subagents on small, realistic, read-only-safe tasks and observing what they do — not another
static read of the diff.

Two prior gates already covered the other altitudes and are **not re-litigated here**:
- `docs/reviews/cpg-agent-adoption.md` Pass 1 (plan gate, `analyst`) — approve with suggestions,
  all three findings fixed in place (U2-fix).
- `docs/reviews/cpg-agent-adoption.md` Pass 2 (diff-scoped code gate, `analyst`, U5) — approve,
  zero blockers/majors/minors, one non-actionable pre-existing nit. That pass explicitly scoped
  AC-1…AC-5 as "prompt-level commitments, not runtime-observable from a static diff read... genuine
  behavioral confirmation... is U6's job, not this gate's." This document is that job.

This feature is agent-prompt wiring (six Markdown files + one skill doc), not a running service —
so "driving the app" here means invoking a wired agent as a subagent on a task that plausibly
touches a live CPG's component, and reading its actual transcript/report for the specific signals
AC-1…AC-5 require: unprompted CPG discovery, a `CPG:` evidence line, a freshness signal, and
(when the signal reads stale) a refresh suggestion rather than silence or a self-triggered rebuild.

## 2. References

- `docs/requirements/cpg-agent-adoption.md` — FR-1…FR-9, AC-1…AC-6 (source of test items below).
- `docs/plans/cpg-agent-adoption.md` (`cobb`) §1 (roster), §2 (discovery mechanic), §3 (evidence
  trail), §4 (AC-6 reconciliation).
- `docs/plans/cpg-agent-adoption-graph.md` (`graph-dba`) §1–§2 (freshness marker + read recipe).
- `docs/reviews/cpg-agent-adoption.md` — both prior gates, for what's already verified.
- `docs/plans/cpg-agent-adoption-coordination.md` — unit ledger, exact deliverables per unit.
- Delivered artifacts under test: `claude/{analyst,architect,qa-engineer,coder,tdd-engineer,
  frontend-engineer}/*.md`, `skills/cpg-analysis/SKILL.md`, `skills/cpg-analysis/references/
  freshness.md`, `claude/README.md`, `skills/README.md`, `docs/BACKLOG.md` M4 section.
- Commits under test: `35b108f` (U1–U4a) and `50f9aaa` (U4b-1..5), both already gated at `c84815c`
  (U5 review commit) at HEAD.

## 3. CPG relevance check (per this agent's own standing orientation)

This QA pass is itself over agent-prompt Markdown and one skill doc — not a source tree a CPG
covers. `CPG: not applicable — this acceptance pass verifies agent-prompt wiring behavior; the
two live CPGs (`cpg_falkorchat`, `cpg_salesperson`) are the *targets* the dispatched subagents are
tested against, not something this test plan itself queries for its own authoring.` No relevant
CPG exists for "agent prompt Markdown" as a domain (same reasoning `cobb`'s own design gives for
excluding itself from the roster, §1 of `cpg-agent-adoption.md`).

No CPG-derived test-gap analysis is used to shape this plan's coverage — the feature under test is
prompt behavior, not a codebase with production/test asymmetry a CPG would usefully map.

## 4. Live-environment grounding (confirmed before writing test items)

- `redis-cli GRAPH.LIST` → `cpg_salesperson`, `cpg_falkorchat` both present, alongside four
  unrelated workspace graphs (`ws:eval`, `ws:acme`, `ws:qa-tico-workflows-manual`, `ws:test`) and
  `reference`. Both target CPGs are live as the requirements/plan docs assume.
- `MATCH (b:CpgBuildInfo) RETURN b.BUILT_AT, b.SOURCE_PATH, b.SOURCE_COMMIT, b.SOURCE_DIRTY`
  against **both** graphs → **zero rows** on both. Neither graph has been rebuilt since U4a's
  pipeline change landed — confirms the design's own "no backfill, wait for next rebuild" rollout
  decision (`cpg-agent-adoption-graph.md` §5) still holds. This is a **deliberately interesting
  live condition** for this test pass: the freshness recipe's own documented behavior for this
  exact case is "treat zero rows the same as stale... a reason for caution, not an error to
  debug" — so a wired agent that actually reads and applies the recipe (not just the mechanical
  discovery step) should surface a refresh-consideration even though nothing is provably "old,"
  because nothing is provably fresh either. TP-002/TP-004 test whether that nuance actually
  survives from recipe text into a dispatched agent's real behavior.

## 5. Risk assessment & coverage strategy

The requirement (AC-1…AC-6) makes six claims about *behavior that must actually manifest*, not
just be documented. The highest risk is exactly what the U5 gate flagged as out of its own scope:
a prompt edit that reads correctly in isolation but doesn't survive contact with a real dispatched
agent — an instruction buried too deep in a body prompt to actually change behavior, a discovery
step an agent silently skips under task pressure, an evidence-trail line an agent forgets to
include, or a freshness nuance (zero rows ≈ stale) that gets flattened to "found nothing, moving
on" in practice.

**Sampling strategy, and why it's sufficient without exhaustive per-agent coverage:**

Both prior gates already verified, file-by-file, that the wiring is *present* and *worded
correctly* on all six agents (U5's point-by-point #1–#2, near-verbatim wording across five of six,
one reasoned variance on `frontend-engineer`). Re-deriving that here would duplicate U5's work at
QA altitude for no new information. What U5 explicitly could **not** verify is whether the wording
actually *drives* behavior in a live dispatch — that's this document's one job. Three live
dispatches are the minimum that credibly exercises every AC at least once across a genuinely
different agent/component/CPG-presence combination each time, rather than three near-identical
runs of the same scenario:

| Dispatch | Agent | Roster status | Component | CPG state | ACs exercised |
|---|---|---|---|---|---|
| D1 | `coder` | newly wired (C-404) | `falkor-chat/server` | `cpg_falkorchat` loaded, no freshness marker | AC-1, AC-2, AC-3, AC-4 |
| D2 | `architect` | already-wired, reworded to default framing (C-403) | `salesperson/chatbot.py` | `cpg_salesperson` loaded, no freshness marker | AC-1, AC-2, AC-3, AC-4 (second agent/component pair — confirms D1 isn't a fluke of one agent's style) |
| D3 | `tdd-engineer` | newly wired (C-404) | `mcp-monitor/` | **no CPG loaded for this component** | AC-5 (and, as a bonus, AC-2's "not relevant" shape) |

This satisfies the task brief's minimum bar exactly: at least one already-wired agent reworded to
default framing (`architect`), at least one newly-wired agent (`coder`, plus `tdd-engineer` as a
second newly-wired sample for free on the AC-5 dispatch), both live CPGs touched, and the AC-5
no-op case tested against a component genuinely absent from both graphs.

**Deliberately not tested (and why):**
- `qa-engineer` (this agent) and `frontend-engineer` are not separately dispatched — U5 already
  confirmed their wiring is worded identically/near-identically to the three dispatched agents
  (same discovery clause, same evidence-trail convention), and a fourth/fifth dispatch would add
  compute cost without a materially different behavioral question to answer. `graph-dba`'s own
  wiring (producer, not a new AC-1 consumer) is out of this pass's scope per the roster design.
- No test item exercises a **CPG with an actual freshness marker present** (`sourceCommit`
  populated, non-trivial age) — because neither live graph has one right now (§4). This is a
  genuine coverage gap, not a deliberate exclusion; flagged in the report's Gaps section rather
  than faked with a synthetic marker (inserting one would mutate a shared graph other work
  depends on, which this agent's guardrails don't permit without asking).
- AC-6 gets a document-existence + one-paragraph confirmation, not a fresh static review — both
  prior gates already re-verified `cpg-agent-adoption.md` §4's literal wording twice (Pass 1 and
  Pass 2 §3 point 3). Re-reading the same four bullets a third time at QA altitude adds nothing.
- No load/perf/security testing — this is prompt wiring; none of those angles carry real risk here.

## 6. Test items

### TP-001 — AC-1: unprompted CPG discovery, newly-wired agent (`coder`)

**Preconditions:** `cpg_falkorchat` loaded (confirmed §4). `coder.md` carries the C-404 wiring
(confirmed present via file read, U5-gated).

**Steps:** Dispatch `coder` as a subagent with a small, read-only-safe, impact-analysis-shaped
task grounded in `falkor-chat/server` (`Services.post_message`, `falkorchat/services.py:726`) —
explicitly framed as investigation-only, no file edits, so the dispatch stays cheap and
side-effect-free. Do not mention the CPG, `cpg-analysis`, or FalkorDB anywhere in the prompt.
Observe the returned transcript/report.

**Expected result:** Without being told, the agent's orientation step discovers `cpg_falkorchat`
exists (e.g., queries it via `mcp__cpg__query`/first-guesses the `cpg_<component>` name) and
consults it for the impact-analysis question rather than defaulting straight to grepping/reading
files by hand.

**Priority:** High. **Type:** Acceptance/e2e (subagent dispatch).

### TP-002 — AC-2/AC-3/AC-4: evidence trail + freshness signal + stale-surfacing, `coder` (same dispatch as TP-001)

**Preconditions:** Same dispatch as TP-001 (one dispatch answers multiple test items — this
mirrors how a stakeholder would actually spot-check a single transcript).

**Steps:** Read D1's full response for (a) a `CPG:` line matching one of the three convention
shapes (§3 of `cpg-agent-adoption.md`), (b) any mention of the graph's currency/freshness, (c),
given both live graphs currently return zero `CpgBuildInfo` rows (§4), whether the agent surfaces
a refresh suggestion rather than silently treating the graph as current or attempting a rebuild
itself.

**Expected result:** A `CPG:` line is present and correctly shaped (AC-2). Some currency signal is
communicated — at minimum "no freshness marker found" (AC-3). Given that signal reads as
"unknown/stale-equivalent" per the recipe's own documented guidance, the agent surfaces a
refresh-suggestion framing (e.g., "consider asking `graph-dba` to rebuild `cpg_falkorchat`") rather
than silence or a self-initiated rebuild (AC-4).

**Priority:** High. **Type:** Acceptance/e2e.

### TP-003 — AC-1: unprompted CPG discovery, already-wired agent reworded to default framing (`architect`)

**Preconditions:** `cpg_salesperson` loaded (confirmed §4). `architect.md` carries the C-403
reword (confirmed present via file read, U5-gated).

**Steps:** Dispatch `architect` as a subagent with a small, read-only-safe, impact-analysis-shaped
design question grounded in `salesperson/chatbot.py` (e.g., what would break if `write_message`'s
signature changed) — investigation/plan-sketch only, no file edits. Do not mention the CPG
anywhere in the prompt. Observe the returned transcript/report.

**Expected result:** Same as TP-001, for a second agent/component pair — confirms the behavior
isn't an artifact of one agent's particular phrasing or one component's CPG.

**Priority:** High. **Type:** Acceptance/e2e.

### TP-004 — AC-2/AC-3/AC-4, `architect` (same dispatch as TP-003)

**Preconditions/Steps/Expected:** Same shape as TP-002, applied to D2's transcript against
`cpg_salesperson`.

**Priority:** High. **Type:** Acceptance/e2e.

### TP-005 — AC-5: no-material-delay-or-noise on a CPG-absent component (`tdd-engineer`)

**Preconditions:** No `cpg_mcp-monitor` (or similarly named) graph exists (confirmed via §4's
`GRAPH.LIST` output — `mcp-monitor` is not among the loaded graphs). `tdd-engineer.md` carries the
C-404 wiring (confirmed present via file read, U5-gated).

**Steps:** Dispatch `tdd-engineer` as a subagent with a small, read-only-safe, test-gap-shaped task
grounded in `mcp-monitor/` (e.g., what test coverage exists for the regex-matching logic in
`mcp_monitor/config.py`) — investigation-only, no file edits. Do not mention the CPG anywhere in
the prompt. Observe wall-clock time to a substantive response and whether the transcript surfaces
any friction, retries, or narration disproportionate to a single miss query.

**Expected result:** The discovery check (guess `cpg_mcp-monitor` or similar, get the tool's
not-found error listing the actually-loaded graphs, conclude none match, stop) costs one cheap
query and produces at most a one-line `CPG: considered, not relevant — ...` note — no visible
delay, no retry storm, no multi-paragraph detour before the agent gets to the actual task.

**Priority:** Medium (this is a "does the no-op stay cheap" check, not a correctness-critical
path). **Type:** Acceptance/e2e.

### TP-006 — AC-2 "not relevant" shape, `tdd-engineer` (same dispatch as TP-005)

**Preconditions/Steps:** Same dispatch as TP-005.

**Expected result:** The response includes a `CPG: considered, not relevant — <clause>` line (or
equivalent phrasing matching that convention shape) rather than silence about the CPG check
altogether — confirming AC-2's "or an explicit, reasoned not-relevant-here" branch, not just the
"used" branch already covered by TP-002/TP-004.

**Priority:** Medium. **Type:** Acceptance/e2e.

### TP-007 — AC-6: downstream plan states extension, not override

**Preconditions:** None beyond repo access.

**Steps:** Read `docs/plans/cpg-agent-adoption.md` §4 ("Reconciliation with M2/M3"). Confirm the
section exists, is not a placeholder, and states in its own words that the plan extends rather
than overrides the M2/M3 consumer-scope boundary, with concrete supporting bullets (M2 recipe
count unchanged, M3 read path unchanged, only consumer list/default-ness widens, both prior
documents remain historically accurate as written).

**Expected result:** Document-existence + literal-statement check passes. Not re-litigated as a
fresh static review — both prior gates (`docs/reviews/cpg-agent-adoption.md` Pass 1 open questions
§, Pass 2 point 3) already independently re-verified this claim's substance twice.

**Priority:** Low (already double-gated; this is a closing formality per the task brief).
**Type:** Document check.

## 7. Environment & data setup

- No environment setup needed beyond repo access and the already-running shared `falkordb-dev`
  instance (both live CPGs already loaded, confirmed §4).
- All three dispatches (D1/D2/D3) are explicitly framed as **read-only investigation** in their
  prompts — no file edits requested, keeping them cheap and leaving no cleanup burden. Subagents
  retain their normal tool access (including Write/Edit), so their prompts explicitly instruct
  "investigation only, report your findings as your final response, do not edit any files" to keep
  this pass side-effect-free without relying on a tool-level restriction.
- No FalkorDB mutation is performed by this test pass itself — every direct query in §4 and used
  by the dispatched agents is a read (`MATCH ... RETURN`, `GRAPH.LIST`).

## 8. Entry/exit criteria

**Entry:** Both prior gates (plan, U3; diff, U5) closed at **approve**/**approve with suggestions**
with all findings resolved — confirmed via `docs/reviews/cpg-agent-adoption.md`. Both live CPGs
confirmed loaded (§4).

**Exit:** All seven test items (TP-001…TP-007) executed and recorded pass/fail/blocked with
evidence in the test report. Any AC where the observed dispatch behavior diverges from the
requirement is filed as a defect with severity by user/stakeholder impact, not by how hard it was
to find.

## 9. Explicitly out of scope

- Re-verifying prompt-file wording/frontmatter/line anchors — already exhaustively checked by the
  U5 diff gate; this pass tests behavior, not text.
- Testing all six wired agents individually — three dispatches is the sampling strategy (§5); the
  other three (`qa-engineer`, `analyst`, `frontend-engineer`) are covered by the same verbatim/
  near-verbatim wording U5 already confirmed, not by a fresh dispatch here.
- Any scenario requiring a populated `CpgBuildInfo` marker (non-trivial age, `sourceCommit` set) —
  neither live graph has one; synthesizing one would mutate a shared graph other work depends on.
  Flagged as a coverage gap in the report, not faked.
- MCP tool contract, `redis-cli` fallback mechanics, `pipeline.sh` stamping logic — all mechanical
  concerns already covered by U4a's own live verification and the U5 diff gate; this pass takes
  the read path and the stamping mechanism as given (FR-8, confirmed untouched).
