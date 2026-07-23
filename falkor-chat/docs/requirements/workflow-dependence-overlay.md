# Workflow Def Data-Dependence Overlay (CPG-style) — Feature Requirements
> Status: Interviewing · Last updated: 2026-07-23
> Source: design conversation with the stakeholder (Mauricio), 2026-07-22. Backlog companion: **K-032** (`docs/BACKLOG.md`). This doc is the WHAT/WHY; K-032 carries the framing sketch and pointers to owners.

## Intent
The stakeholder wants the workflow-**definition** graph to carry richer, program-analysis-style
relationships between steps — the way a **Code Property Graph (CPG / Joern)** overlays control-flow
and data-dependence edges on code — while preserving today's flexibility that "a step is just an
agent with a system prompt."

Through a CPG lens, the def graph today materializes **only its control-flow layer**:
`(:Step)-[:TRANSITION {guard, order}]->(:Step)` is a guarded CFG, `(:WorkflowDef)-[:HAS_STEP]->(:Step)`
is one-level AST containment, `(:WorkflowDef)-[:START]->(:Step)` is the entry, and the run-time
`(:StepRun)-[:NEXT]->(:StepRun)` is an executed CFG path. **The data-dependence layer is missing** —
which `ctx`/`output` keys each step *reads* and *writes*.

The goal is to **materialize that dependence layer as real graph structure at publish time**, so a
workflow def can be **statically analyzed when it is published (seed time)** — turning three classes
of authoring error that today surface only at live-run time (or never) into checks that run before a
def is ever executed. The underlying value is **cheaper, earlier, honest feedback to workflow
authors**, made especially worthwhile because published defs are effectively immutable.

> Note (framing vs. need): "connect the nodes with CPG-style relationships" is the stakeholder's
> proposed *shape*. The underlying need is *publish-time static analysis of a workflow def —
> dangling-read detection, reachability, and change-impact — grounded in dependence information that
> already exists but is trapped in opaque strings.* Recorded as the leading design direction for
> graph-dba/architect, not a locked model.

## Problem & current state
- The dependence information is **not absent — it is trapped inside opaque serialized `guard`/`config`
  strings** on `Step` and `TRANSITION` (rule 8: those strings are parsed app-side only, never filtered
  in Cypher). Because it is invisible as graph structure, it cannot be queried or checked.
- The publish-time validator (`services._validate_def_spec` + `guards.validate_cmp`) **already walks
  the `cmp` guard tree** — most of the read-set extraction already happens at publish and is then
  thrown away.
- Three classes of authoring error are therefore uncatchable at authoring time:
  1. **Dangling read** — a guard reads a `ctx`/`output` key that no upstream step ever writes. This is
     exactly the documented but **un-enforced hazard "n-3"** (`AGENTS.md`): a `decision` step with
     all-conditional outgoing transitions and no `waitsForHuman` self-loops until it exhausts the step
     budget. Today it is discovered only when a run parks or burns out.
  2. **Unreachable step / dead branch** — a step with no path from `START`. No check exists.
  3. **Change-impact / blast radius** — "if I change step X's output shape, which downstream guards
     break?" There is no way to answer this. It matters *specifically because* published defs are
     create-only + immutable (K-031): a def edit costs a version bump + snapshot republish and risks a
     `reference`↔`ws:{id}` split-brain, so knowing the blast radius **before** the edit has real value.
- This is a **follow-up, not an M3 gate** — M3 (workflow engine) is delivered and QA-accepted (K-025).

## User stories
- As a **workflow author**, I want a def that reads a `ctx`/`output` key nothing upstream writes to be
  **rejected when I publish it**, so that I find the error at seed time instead of watching a run park
  or exhaust its step budget.
- As a **workflow author**, I want a def with an **unreachable step** rejected at publish, so that dead
  branches never ship.
- As a **workflow author / operator**, before I edit an (immutable) published def, I want to see the
  **blast radius** — which downstream steps and guards depend on a given step's outputs — so that I can
  weigh the cost of a version bump + republish before committing to it.
- As a **workflow author using an agent step**, I want steps whose reads *cannot* be derived statically
  (e.g. an `llm`-judged guard, an agent with a system prompt) to be **honestly marked "reads unknown"**
  rather than silently reported as reading nothing, so that the analysis tells me precisely which parts
  of my flow are statically checkable versus trust-the-model.

## Functional requirements
- **FR-1** — When a workflow def is published, the system **derives a data-dependence overlay** and
  materializes it as graph structure on the **definition** graph: relationships recording which
  `ctx`/`output` keys each `Step` **reads** and which it **writes**. The overlay is a compile output of
  publish, not information parsed at read time.
- **FR-2** — Read/write sets are extracted from the **existing closed `cmp` guard family** and the
  step's declared config: reads from the `ctx.`/`output.` roots referenced in a step's outgoing
  `cmp`/branch guards; writes from a `human`/`wait` step's declared `config.expects` and a step's
  declared outputs. No new expression language is introduced (see Out of scope).
- **FR-3** — A step whose reads **cannot be derived statically** (e.g. an `llm`-kind guard, an `agent`
  step judged in natural language) is marked with an explicit **"reads unknown"** signal — it must
  never be recorded as reading zero keys. The analysis must be **sound**: it never emits a dependence
  edge it cannot justify, and it never claims completeness it does not have.
- **FR-4** — Publishing a def whose guard **reads a key no upstream step writes** (a dangling read) is
  **rejected at publish** with a def-spec authoring error, *unless* that step is marked "reads unknown"
  for the key in question (an unknown read cannot be proven dangling). This closes hazard n-3 at seed
  time. _(Interaction with run-initial `ctx` keys — see OQ1.)_
- **FR-5** — Publishing a def containing a step **unreachable from `START`** is **rejected at publish**
  with a def-spec authoring error.
- **FR-6** — An author/operator can **query the change-impact / blast radius** of a step — the
  downstream steps and guards that read that step's outputs — by traversing the materialized overlay,
  with no string parsing at query time. _(First-slice vs. follow-on sequencing — see OQ2.)_
- **FR-7** — The overlay is derived and materialized **atomically as part of the existing publish**
  (the same act that writes the def + its `Step`s + the `START` edge), so it cannot drift from the def
  it describes. It must not become a separate axis of `reference`↔`ws:{id}` split-brain.
- **FR-8** — The overlay lives **on the definition graph only** (tens of nodes; overlay edges a
  single-digit multiple → RAM non-issue). It is **never** materialized onto run-time `StepRun` data
  (thousands of nodes, RAM-bound). "Why did *this run* branch here" remains a join through `RAN` into
  the def overlay, not a second copy of the layer.

## Out of scope
- **A general expression language or data-flow engine.** The analysis rides the existing closed `cmp`
  guard family only; the `expr` guard kind stays an unimplemented `NotImplementedError` seam. Guarding
  against scope creep toward a general DDG engine is an explicit non-goal.
- **Probabilistic / best-effort dependence edges.** An unsound edge yields confident-wrong impact
  answers — worse than none. Undecidable reads are marked unknown (FR-3), never guessed.
- **Any run-side (`StepRun`) overlay or run-time RAM cost** (FR-8).
- **Parsing `guard`/`config`/`ctx` strings inside Cypher at read time** — rule 8 holds; extraction is
  an app-side publish-time compile step, and only the resulting edges hit the graph.
- **Auto-fixing or auto-editing defs.** The feature *detects* and *rejects*; it never rewrites a def.
- **Enforcing K-029's symmetric-`decision` invariant as a language rule.** This feature provides the
  graph mechanism that *could* back that rule; the rule itself is K-029's decision.

## Acceptance criteria
- **AC-1** — Given a def with a `cmp` guard reading `ctx.X` and an upstream step declaring it writes
  `X`, when the def is published, then the def graph reads back with the corresponding **reads/writes
  overlay edges**, and the dependence between the two steps is answerable in a single graph traversal
  (no string parsing).
- **AC-2** — Given a def whose guard reads a `ctx`/`output` key **no upstream step writes** (and the
  reading step is not "reads unknown" for that key), when it is published, then publish is **rejected**
  with a def-spec authoring error and no def is written.
- **AC-3** — Given a def containing a step with **no path from `START`**, when it is published, then
  publish is **rejected** with a def-spec authoring error.
- **AC-4** — Given a step whose reads cannot be derived statically (e.g. an `llm`-kind guard), when the
  def is published, then that step is recorded with the **"reads unknown"** marker and is **not** shown
  as reading zero keys.
- **AC-5** — Given a published def, when an author asks which downstream steps/guards read a chosen
  step's outputs, then the answer is produced by traversing the overlay — listing those steps/guards
  correctly, including none when there are no dependents. _(Deliverable in this feature vs. a follow-on
  slice — OQ2.)_
- **AC-6** — Given any published def, when the run graph is inspected, then **no** overlay edges exist
  on `StepRun` nodes, and the per-workspace RAM delta attributable to the overlay is negligible.

## Open questions
- **OQ1 (leverage: high — soundness)** — Dangling-read detection must account for keys present in a
  run's **initial `ctx`** (seeded at run start, written by no step). Does the def declare its expected
  start-context keys (so those are legitimate "sources" and not dangling reads), or does the check
  treat any key not written by a step as dangling? Getting this wrong makes FR-4 either unsound (misses
  real dangling reads) or noisy (rejects valid defs). Needs a stakeholder/architect decision.
- **OQ2 (leverage: high — scope)** — Which capabilities are in the **first slice**? K-032's sketch
  scopes slice 1 to the overlay + **dangling-read (FR-4)** + **unreachable-step (FR-5)** checks, with
  `llm`-guard read handling and the **change-impact query (FR-6)** as follow-on slices. Confirm whether
  FR-6/AC-5 are in-scope for this feature or deferred.
- **OQ3 (leverage: medium)** — Should the two publish checks be **hard rejections** (`WorkflowDefSpecError`,
  as written) or **warnings** an author can override? Assumed hard rejection to match K-032's
  "rejected at publish" language and the existing publish-validator posture.
- **OQ4 (leverage: low)** — Is "reads unknown" a per-step marker or per-(step, key) precision? Assumed
  granular enough that a step with one statically-known read and one unknown read is not wholly opaque
  (FR-3/FR-4 lean on this).

## Assumptions (recorded — subject to confirmation at readback)
- **A-1** — The overlay's checks are **hard publish-time rejections**, not warnings (OQ3). Chosen to
  match K-032 and the existing `_validate_def_spec` behavior.
- **A-2** — FR-6/AC-5 (blast-radius query) are captured as **requirements of the overall feature** but
  their delivery may be a **follow-on slice** after the two rejection checks (OQ2). The architect owns
  slicing.
- **A-3** — "Upstream" for the dangling-read check means *reachable-predecessor along `TRANSITION`
  paths from `START` up to the reading step* — i.e. reachability-aware, not merely "any step in the
  def declares the write." Flagged because a naive any-step-writes-it check would be unsound for
  branchy defs; final semantics are an architect/graph-dba decision.

## Decision log
2026-07-22 — Source & intent → design conversation with the stakeholder; "workflow nodes connected
using CPG/Joern-style relationships, keeping the agent-with-a-system-prompt flexibility." Filed as
backlog K-032. Framing recorded: def graph already has the CFG layer; the missing piece is the
data-dependence (READS/WRITES) layer, trapped in opaque `guard`/`config` strings.
2026-07-22 — Payoff (the WHY) → publish-time (not live-run) detection of (1) dangling read = the
un-enforced n-3 hazard, (2) unreachable step, (3) change-impact/blast-radius — the last made valuable
by immutable, create-only published defs (K-031).
2026-07-22 — Hard constraints accepted (not to be reopened in design): derive at publish, never parse
in Cypher (rule 8); overlay built inside the existing atomic publish/materialize, not a follow-up
write; static-only, on the def graph, never on `StepRun`; honest "reads unknown" for
non-statically-derivable reads — sound, never a probabilistic guess.
2026-07-22 — Non-goals accepted: no general expression language / data-flow engine (rides the closed
`cmp` family; `expr` stays a `NotImplementedError` seam); not an M3 gate (M3 is ✅).
2026-07-23 — Requirements drafted from the design-conversation brief. Status kept **Interviewing**: no
closing readback with the stakeholder yet; OQ1–OQ4 open, assumptions A-1…A-3 recorded pending
confirmation before flipping to Ready for design.
