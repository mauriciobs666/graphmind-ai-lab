# Kaizen — Improvement Plan: tico

> Forward-looking backlog for the `tico` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-31

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | high | 🔵 | Live e2e spin: a real `claude --agent tico` interview on a genuine feature request |
| K-002 | 2026-07-09 | — | ⚪ | ~~SendMessage continuation for interview rounds~~ — moot in first-order mode |
| K-003 | 2026-07-09 | low | 🔵 | Requirements→plan traceability (architect plan cites FR-ids) |
| K-004 | 2026-07-29 | high | 🔵 | Live e2e spin of Modes 2 & 3 (explanation + first real user manual) |
| K-005 | 2026-07-29 | low | 🔵 | Formal update to `docs/plans/doc-reference-convention.md` for the new `manuals/` kind (architect-owned doc) |
| K-006 | 2026-07-31 | high | 🔵 | Live e2e spin of the demo-environment offer → `devops` delegation → confirmed-teardown loop |

### K-001 — Live e2e spin (first-order)
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** the live-interview design is unexercised; same validate-by-running discipline as teco's K-001.
- **Proposed change:** launch `claude --agent tico` on a genuinely vague feature request (e.g. in `falkor-chat/`) and run the interview to "Ready for design". Verify: `initialPrompt` kicks off correctly, the doc is updated *during* the conversation (not batched), the guard hook passes conforming writes silently, one-thread-at-a-time pacing holds, and the readback/explicit-confirmation gate fires before the status flip.
- **Notes:** also worth one delegated invocation to see the subagent fallback degrade gracefully.

### K-002 — SendMessage continuation for rounds
- **Status:** ⚪ rejected 2026-07-09
- **Rationale (original):** make re-invoked interview rounds cheaper by continuing the spawned agent.
- **Why rejected:** the first-order redesign removed the rounds protocol from the primary path — as the main-session agent tico converses natively. The subagent fallback keeps doc-as-state and needs no continuation machinery.

### K-003 — Requirements→plan traceability
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** FR-ids exist in tico's template but nothing downstream references them; an architect plan that cites FR-ids makes coverage checkable.
- **Proposed change:** once a real tico→architect handoff has run, consider asking the architect (prompt or convention) to map plan steps / test strategy to FR-ids.
- **Notes:** don't build until a real handoff shows the need.

### K-004 — Live e2e spin of Modes 2 & 3
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Modes 2 (didactic explanation) and 3 (user manuals) are unexercised — same validate-by-running discipline as K-001, which only covers Mode 1.
- **Proposed change:** run `claude --agent tico` once asking it to explain a real project aspect (verify: grounded in actual docs/code, plain-language, light-suggestion framing, inline Mermaid when it fits) and once asking it to write a first real manual end-to-end (verify: `docs/manuals/<slug>.md` created with the header block, Mermaid used only where it earns its keep, guard hook passes the write silently, commit-as-you-go holds, **the offered verification-pass bullet actually fires and correctly spawns `qa-engineer`/`analyst` via `Agent` on acceptance**). Also worth one delegated (subagent) invocation of the Mode 3 fallback to see it complete in one pass from a self-contained brief.
- **Notes:** added when Modes 2/3 were introduced (2026-07-29); do the interview e2e (K-001) and this one independently — different mode, different failure surface.

### K-005 — Formal doc-reference-convention update for `manuals/`
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** root `AGENTS.md` was extended directly with the new `manuals/` kind (needed immediately for tico's new capability), but the formal, heavily-reviewed spec at `docs/plans/doc-reference-convention.md` (Owner: `architect`, Status: active, Tracks: C-322) still doesn't mention it. That document is architect's, not cobb's, to extend with a proper versioned changelog entry.
- **Proposed change:** an `architect` pass adding a `manuals/` mention to the convention plan (changelog entry, any affected checks/tables) — purely additive, doesn't reopen any existing ruling (D1/D4/D6 etc.).
- **Notes:** not blocking — root `AGENTS.md` is what's actually enforced/read every session; this is closing the paper trail, not fixing a behavior gap.

### K-006 — Live e2e spin of the demo-environment delegation
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** the demo-environment offer (2026-07-31) is unexercised — same validate-by-running discipline as K-001/K-004, which only cover the interview and Modes 2/3's own writing. This one crosses an agent boundary (tico → devops), so it has its own failure surface: does the offer actually surface at the right moment, does the `Agent` brief carry enough for devops to orient without back-and-forth, and — the discipline most worth watching — does tico actually stop and ask before requesting teardown rather than assuming a good moment.
- **Proposed change:** in a real `claude --agent tico` session (Mode 2 or 3, over a component devops can plausibly boot, e.g. `falkor-chat/`), let the stakeholder accept a demo offer; verify devops comes back with the environment actually up (real command output, not a claimed green) and tico correctly attributes what's running to devops's work. Then verify the teardown path: tico asks before tearing down, waits for a yes, and only then delegates cleanup — never auto-triggers it on topic change or session end.
- **Notes:** run independently of K-001/K-004; a different mode and a different agent boundary than either.

## Parking lot / ideas
- **Prompt-quality lint (2026-07-29, authoring pass over the Mode 2/3 addition):** clean on contradiction, ambiguity, persona, and composition (root `AGENTS.md`'s new `manuals/` convention and tico.md agree, no restatement). Two minors, not acted on: (a) **cognitive load** — the prompt grew 98→152 lines adding two modes; still followable in one pass today, but if it grows further, split Mode 2/3's craft guidance into an on-demand skill rather than keep inlining. (b) **coverage** — no explicit guidance for "a new manual would overlap an existing one" or "researching a manual surfaces what looks like an actual bug" (vs. a docs gap); low-value to prescribe pre-emptively, revisit if either happens in practice.
- **Existing requirements docs still carry the pre-2026-07-27 unbolded status line** — do **not** hand-normalise them mid-interview. The one-time backfill across all active feature documents is step 3 of `docs/plans/doc-reference-convention.md` (owner: `coder`, whose writes aren't doc-guarded); after it lands, `tico` only ever writes the new form (noted 2026-07-27).
- A `docs/requirements/` template file vs the inline template (only if the inline one drifts across features).
- Non-functional requirements section (performance, security) — add when a feature actually needs one rather than padding every doc.
- Project-scoped `agent` setting to make tico the default session agent in a requirements-heavy phase — only if launching via `--agent` proves to be friction.
- **Proposal declined 2026-07-30 (cobb review):** widening tico's commit authority to cover
  artifacts produced by agents it summons under Mode 3 (analyst/qa-engineer verification
  passes), so tico becomes "an orchestrator like teco." Declined — breaks the
  write-scope==commit-scope invariant every commit-capable agent holds, conflicts with
  `audit-team.sh`'s single-orchestrator assumption and the team's "tico is not a delegation
  target" framing, and isn't backstopped by any hook. tico was right to hold the line and not
  self-waive its guardrail even under stakeholder pressure. Full reasoning in
  `claude/cobb/kaizen/history.md` (2026-07-30 entry).
  **RESOLVED 2026-07-30 — closed by explicit stakeholder decision, do not re-open.** The
  stakeholder ruled directly (not via cobb's recommendation, and not the softer
  session-scoped-summoned-only variant either): "I dont want the subagents to proliferate
  commits, tico (you) and teco are special and have coordination rights." So: `analyst`/
  `qa-engineer`/every other specialist stays **without** commit authority, permanently, not
  pending a recurrence — and `teco`'s own commit authority (previously undocumented — it had
  quietly used `Bash` to commit four uncommitted deliverables this same day with no guardrail
  text backing it) is now formally documented in `teco.md`, scoped to its integrator role, and
  backstopped by a new `audit-team.sh` check 8 that fails if any agent other than `tico`/`teco`
  ever claims the same authority. Full implementation in `claude/teco/kaizen/history.md`
  (2026-07-30 entry) and `claude/cobb/kaizen/history.md` (2026-07-30 entry). This question is
  closed — if uncommitted-deliverable friction recurs, the fix is routing through `tico`/`teco`
  (both now empowered for exactly this), never a third agent gaining commit rights.
