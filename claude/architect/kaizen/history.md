# Kaizen — Change History: architect

> Dated log of actual changes to the `architect` agent. Most recent first.

## 2026-07-09 — Consume tico's requirements doc by path (handoff symmetry)
- **What:** "Understand the request" now states that a feature requirements document from `tico` may arrive as a path (`<component>/docs/requirements/<slug>.md`) — read it first as the stakeholder-confirmed WHAT/WHY the plan turns into a HOW; its acceptance criteria feed the test strategy.
- **Why:** `tico` was created 2026-07-09 as the requirements half of a tico→architect handoff; the consumer side must state the convention too (agent-maintenance §4 handoff symmetry).
- **Plan items:** none.

## 2026-07-09 — K-002 ✅: live handoff validation (teco K-001 run, falkor-chat M3 slice 1)
- **What:** The architect ran as the planning half of a real orchestrated delivery — teco
  delegated it the M3 decomposition + slice-1 plan for falkor-chat. It produced
  `falkor-chat/docs/plans/m3-workflow-engine.md` (Part A: six kaizen items K-020…K-025 in the
  component's exact item format; Part B: full slice-1 plan with data model, DDL reconciliation,
  query shapes, service surface, build order, enumerated suite-count expectations) and returned
  the path. Two isolated-context implementers executed it cold: graph-dba (gate) and
  tdd-engineer (impl) — no re-investigation loops, suites landed green (query 193/193,
  pytest 196), structural parity + idempotency proven.
- **Friction observed (the K-002 payload):** one plan gap — `publish_workflow_def` was specced
  without a `start_key` parameter; the implementer resolved it (exactly one step declares
  `start: True`) and it was surfaced as a contract to lock at K-022. One gate-level design
  amendment — the plan's `STARTS WITH stepUid` scoping PROFILEd as a label scan, so graph-dba
  added a `HAS_STEP` containment edge; a reasonable division of labor (live PROFILE data is the
  gate's job), not a plan defect. Verdict: the six-section template held; no template change
  needed from a single datapoint — recheck if the parameter-contract gap recurs.
- **Prompt changes:** none.
- **Plan items:** K-002 ✅ done (moved here — the live validation this item waited on; evidence
  shared with teco K-001, see `claude/teco/kaizen/history.md` 2026-07-09). Plan is now empty of
  active items.

## 2026-07-08 — Plan-doc handoff by default, subagent-context awareness, hook-enforced Write/Edit (K-001 ✅, K-003 ✅)
- **What:** Four changes from a team-level design review (architect as a member of teco's roster):
  1. **Plan document is now the default deliverable** (K-001 ✅): step 5 rewritten — write the plan to `<component>/docs/plans/<slug>.md` (kebab-case; repo-root `docs/plans/` for cross-component work) and return the *path* + the "ready to implement" summary; inline delivery only for quick assessments. Handoff section updated to match ("implement the plan at `<path>`"). Convention matches what falkor-chat already used de facto (`falkor-chat/docs/plans/m2-graphrag.md`).
  2. **Subagent-context awareness:** new opener paragraph — the brief is the architect's *entire* context (no user conversation, no other agents' work) and its final message is terminal (`AskUserQuestion` unavailable to subagents). Step 1 reframed: design-changing ambiguity → return findings + the one or two sharp open questions *as the deliverable* and stop, instead of "ask questions" (impossible mid-run).
  3. **TDD-destined plans:** test-strategy section now says to sequence as an ordered list of behaviors/test cases (red→green) when the repo mandates TDD or the implementer is `tdd-engineer` (this user's default preference).
  4. **Harness-enforced read-only-on-code** (K-003 ✅): new subagent-scoped `PreToolUse` hook `architect/hooks/guard-plan-doc-writes.sh`, wired in frontmatter (`matcher: Write|Edit`, absolute path — devops precedent). Escalates any `Write`/`Edit` targeting a path outside `docs/plans/` (or `/tmp` scratchpad) to the human (`permissionDecision: "ask"`); fail-open on unparseable input with the prompt guardrail as backstop. Smoke-tested: code path → ask; absolute + relative `docs/plans/` → pass; `/tmp/` → pass; garbage stdin → pass.
  - Sibling + catalog sync (same change): `teco.md` now hands off the architect's plan **by path** (never paraphrased into the brief) and gained a coordination-doc convention (`docs/plans/<slug>-coordination.md`, teco K-003 ✅); `coder.md` orient step expects the plan-document path and reads the file as source of truth (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md` updated).
- **Why:** Review found (a) inline-by-default delivery was optimized for a human caller and pessimal for the orchestrated path — teco copying a plan "verbatim" into a brief is a lossy telephone game, while a file handed off by path is lossless and durable; (b) the prompt assumed an interactive caller it doesn't have as a subagent; (c) the read-only contract was prompt-only while a working hook precedent existed in-repo (`devops/hooks/guard-destructive-ops.sh`).
- **Decision — Bash deliberately NOT hooked:** the hook closes the realistic *accidental* failure mode (the editing tools drifting into source). Mutating the tree via Bash would be a deliberate guardrail violation, which prompt-guarding handles reliably for Opus-class models, and pattern-matching "bash writes" is brittle/noisy. Accepted residual risk; escalation path recorded in the plan's parking lot.
- **Plan items:** K-001 ✅ done, K-003 ✅ done (both moved here); K-002 remains open — the convention is baked but the live architect→coder validation run is still pending.

## 2026-06-21 — Added `Edit`, scoped to plan/design docs
- **What:** Added `Edit` to the frontmatter `tools` list (`Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent`). Reworded the Guardrails so `Write`/`Edit` are explicitly for **one purpose** — authoring (`Write`) and revising in place (`Edit`) the plan/design document — and never source/tests/config. Updated catalog entries that previously asserted "no `Edit`" (`claude/AGENTS.md`, root `AGENTS.md`). The `description` and `claude/README.md` ("does NOT edit source code" / "without editing code") were left unchanged — still accurate, since the agent edits plan docs, not code.
- **Why:** User asked to enable `Edit` for the architect. Flagged the design tension (its whole contract is read-only-on-code; the description is also the auto-delegation routing signal) and confirmed intent: **plan docs only**, not code. Previously the agent could only `Write` (overwrite) plan files; `Edit` lets it amend a plan in place during an architect→coder iteration without rewriting the whole document. Tool gating still can't enforce "plan docs only" (both Write and Edit can target any path) — the prompt guardrail carries that, same as before.
- **Plan items:** advances the spirit of K-003 (tool gating) but in the *loosening* direction; K-003 (stricter gating) left open — see plan note.

## 2026-06-20 — Dropped "senior" framing
- **What:** Removed "senior" from the `description` ("Senior software architect" → "Software architect") and the body opener ("You are a senior software architect" → "You are a software architect"). Mirrored in the catalog entries (`claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md`).
- **Why:** User flagged the overconfidence concern with seniority framing. Evidence (persona-prompting studies, e.g. Zheng et al. 2024) shows role labels are weak-to-neutral for correctness and authority framing can dent calibration; behavior is driven by the concrete process + guardrails, not the title. Chose the most conservative option (drop the word entirely) over keeping it. Note: this goes one step *further* than the 2026-06-05 collection precedent, which dropped boasts but **kept** "Senior" as an altitude signal — so architect/coder are now inconsistent with cobb/tdd-engineer/graph-dba/dra-claudia until those are harmonized (flagged to user).
- **Plan items:** —

## 2026-06-20 — Created
- **What:** Created the `architect` subagent (`architect/architect.md`, `model: opus`). Read-only design/planning agent: investigates the codebase, weighs trade-offs, and produces a step-by-step implementation plan/spec (goal & scope, context & findings, design & rationale, ordered steps, test strategy, risks & open questions). Tools restricted to `Read, Grep, Glob, Bash, Write, WebFetch, WebSearch, Agent`; **no `Edit`/`NotebookEdit`** and a hard guardrail that `Write` is for the plan document only — it never edits source/tests/config. Designed as the planning half of an **architect→coder handoff**: the plan stands alone so an isolated-context implementer can execute it.
- **Why:** User asked to create two complementary Claude Code subagents, "the architect" and "the coder," distinct from the existing `tdd-engineer` (strict TDD implementer) and the OpenCode `coding-senior`, with a sequential architect→coder handoff.
- **Plan items:** seeded K-001..K-003.

## Decisions recorded at creation
- **Why `Write` but no `Edit`:** the agent must be able to emit a *durable* plan document for the handoff (an isolated coder context won't see the architect's investigation otherwise), but must not surgically modify existing code. Omitting `Edit`/`NotebookEdit` + a prompt guardrail signals "planning only" while still allowing the deliverable. `Bash` is investigation-only by guardrail. Tool gating can't fully enforce "plan docs only" (Write can overwrite any path) — the guardrail carries that.
