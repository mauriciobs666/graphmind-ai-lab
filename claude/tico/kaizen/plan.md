# Kaizen — Improvement Plan: tico

> Forward-looking backlog for the `tico` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-29

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | high | 🔵 | Live e2e spin: a real `claude --agent tico` interview on a genuine feature request |
| K-003 | 2026-07-09 | low | 🔵 | Requirements→plan traceability (architect plan cites FR-ids) |
| K-004 | 2026-07-29 | high | 🔵 | Live e2e spin of Modes 2 & 3 (explanation + first real user manual) |
| K-005 | 2026-07-29 | low | 🔵 | Formal update to `docs/plans/doc-reference-convention.md` for the new `manuals/` kind, now also carrying a collision-rule-5 gloss (architect-owned doc) |
| K-006 | 2026-07-31 | high | 🔵 | Live e2e spin of the demo-environment offer → `devops` delegation → confirmed-teardown loop |
| K-010 | 2026-08-29 | high | 🔵 | Live e2e spin of a proactive review-shaped consult (announce-then-proceed, no acceptance wait) |
| K-011 | 2026-08-29 | high | 🔵 | Live e2e spin of a fast-tracked direct Q&A + multi-turn follow-up via `SendMessage` |
| K-012 | 2026-08-29 | medium | 🔵 | Live check that a consult declines `coder`/`tdd-engineer`/`frontend-engineer` and multi-unit sequencing, pointing at `teco` |

### K-001 — Live e2e spin (interactive)
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** the live-interview design is unexercised; same validate-by-running discipline as teco's K-001.
- **Proposed change:** launch `claude --agent tico` on a genuinely vague feature request (e.g. in `falkor-chat/`) and run the interview to "Ready for design". Verify: `initialPrompt` kicks off correctly, the doc is updated *during* the conversation (not batched), the guard hook passes conforming writes silently, one-thread-at-a-time pacing holds, and the readback/explicit-confirmation gate fires before the status flip.
- **Notes:** also worth one delegated invocation to see the subagent fallback degrade gracefully.

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

### K-005 — Formal doc-reference-convention update for `manuals/`, and a collision-rule-5 gloss
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** root `AGENTS.md` was extended directly with the new `manuals/` kind (needed immediately for tico's new capability), but the formal, heavily-reviewed spec at `docs/plans/doc-reference-convention.md` (Owner: `architect`, Status: active, Tracks: C-322) still doesn't mention it. That document is architect's, not cobb's, to extend with a proper versioned changelog entry.
- **Proposed change:** an `architect` pass adding (a) a `manuals/` mention to the convention plan (changelog entry, any affected checks/tables) — purely additive, doesn't reopen any existing ruling (D1/D4/D6 etc.); and (b) a gloss on collision rule 5's "approved, gated, or executed against" test, from a 2026-08-20 kaizen distillation finding (`tico/kaizen/history.md`, 2026-08-21 entry): a document reaching its approval **gate** alone (e.g. `Status: Ready for design`) does not by itself force a successor document if nothing downstream has actually *executed* against the specific content being revised — `docs/requirements/generic-cypher-mcp2.md`'s own 2026-08-20 Decision-log entry and `docs/plans/generic-cypher-mcp2.md`'s `T1` precedent paragraph both establish this reading in practice, but the rule's literal disjunctive text doesn't yet say so, so a future agent reading it cold could misapply it.
- **Notes:** not blocking — root `AGENTS.md` is what's actually enforced/read every session; this is closing the paper trail, not fixing a behavior gap. Both sub-items are the same underlying job (an architect pass adding clarifying content to the same owned document) — batch them rather than splitting into two K-items.

### K-006 — Live e2e spin of the demo-environment delegation
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** the demo-environment offer (2026-07-31) is unexercised — same validate-by-running discipline as K-001/K-004, which only cover the interview and Modes 2/3's own writing. This one crosses an agent boundary (tico → devops), so it has its own failure surface: does the offer actually surface at the right moment, does the `Agent` brief carry enough for devops to orient without back-and-forth, and — the discipline most worth watching — does tico actually stop and ask before requesting teardown rather than assuming a good moment.
- **Proposed change:** in a real `claude --agent tico` session (Mode 2 or 3, over a component devops can plausibly boot, e.g. `falkor-chat/`), let the stakeholder accept a demo offer; verify devops comes back with the environment actually up (real command output, not a claimed green) and tico correctly attributes what's running to devops's work. Then verify the teardown path: tico asks before tearing down, waits for a yes, and only then delegates cleanup — never auto-triggers it on topic change or session end.
- **Notes:** run independently of K-001/K-004; a different mode and a different agent boundary than either.

### K-010 — Live e2e spin of a proactive review-shaped consult
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** the announce-then-proceed mechanism (2026-08-29) is unexercised — does the
  one-line announcement actually fire before the `Agent` dispatch (not after), does tico correctly
  pick a review-shaped case (a requirements assumption or manual claim) vs. a fast-track question,
  and does the stakeholder's in-the-moment stop/redirect actually work if exercised.
- **Proposed change:** in a real `claude --agent tico` session (any of Modes 1-3), let a genuine
  review-shaped need arise and observe: the announcement precedes the call, the specialist is
  briefed with the artifact by path, the returned finding is folded into the doc (not just quoted
  back), and the consult is logged in the decision log (Mode 1) or stated directly (Modes 2/3).
- **Notes:** run independently of K-011 — different interaction shape, different failure surface
  (folding a finding back in vs. relaying an answer inline).

### K-011 — Live e2e spin of fast-track Q&A + multi-turn follow-up
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** the `SendMessage`-based same-delegate continuation is entirely new — this is the
  first time tico has ever resumed a delegate rather than spawning cold each time. The stopping
  rule (resolved / can't-progress / ~3-4 rounds) is prompt-level self-discipline, not hook-enforced,
  so it needs a real run to see whether it actually holds under a genuinely unresolved question.
- **Proposed change:** ask tico a technical question it can't verify alone (e.g. an ontology
  design rationale, routed to `graph-dba`); verify the fast-track answer comes back inline with no
  artifact created, then force a follow-up round (ask a clarifying question) and confirm tico
  continues the *same* delegate via `SendMessage` rather than a fresh `Agent` spawn — check the
  transcript/agentId, don't just trust the narration. Separately, contrive a case that doesn't
  resolve in ~4 rounds and verify tico actually stops and reports "not resolved" rather than
  continuing indefinitely or silently giving up.
- **Notes:** this is the highest-risk item of the three — a `SendMessage` misfire (wrong agentId,
  or a fresh spawn mislabeled as a continuation) would be a silent correctness failure, not a
  crash. Also worth confirming the missing-`subagent_type` guard (`guard-tico-agent-dispatch.sh`)
  actually escalates on a real malformed dispatch, not just the direct hook test `cobb` ran.

### K-012 — Live check of the roster/boundary refusal
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** AC-7's decline behavior (asked to consult `coder`/`tdd-engineer`/
  `frontend-engineer`, or to sequence/gate multiple delegated units) is easy to get right in
  isolation and easy to erode later as the consult mechanism gets used more — worth one concrete
  check now that it's shipped, rather than assuming the prompt text alone is sufficient.
- **Proposed change:** in a live session, ask tico to "have `coder` fix X" and, separately, to
  "get `analyst` and `qa-engineer` to work through this in sequence" — confirm tico declines both
  and points at `teco`, rather than reinterpreting either as a single-topic consult it's allowed
  to run.
- **Notes:** low cost, quick to run — bundle with K-010 or K-011's session rather than spinning up
  a fourth session just for this.

## Parking lot / ideas
- **`tico.md`'s archived-flip ownership over-claims against root `AGENTS.md` (pre-existing; found by `cobb`'s C2 lint, not introduced by it).** Mode 3 states "You are this kind's owner — you perform that flip yourself, on the same evidence basis as any other doc kind." Root `AGENTS.md` assigns `archived` to **`teco`, at milestone close** (its write guard auto-allows the mechanical one-token edit) and routes only the *non-mechanical* archived flips to the by-kind owner. Root `AGENTS.md` wins — it is the convention's home and explicitly carves the mechanical case out. `tico.md` cites the right table and states the conclusion absolutely. Proposed rewrite: *"You are this kind's owner: the `superseded` flip is yours, as is any `archived` flip that needs judgment — the mechanical archived flip at milestone close is `teco`'s (root `AGENTS.md`)."* Left out of C2 deliberately: it is an authority correction, not a compression, and belongs under its own gate.
- **Prompt-quality lint (2026-07-29, authoring pass over the Mode 2/3 addition):** clean on contradiction, ambiguity, persona, and composition (root `AGENTS.md`'s new `manuals/` convention and tico.md agree, no restatement). Two minors, not acted on: (a) **cognitive load** — the prompt grew 98→152 lines adding two modes; still followable in one pass today, but if it grows further, split Mode 2/3's craft guidance into an on-demand skill rather than keep inlining. (b) **coverage** — no explicit guidance for "a new manual would overlap an existing one" or "researching a manual surfaces what looks like an actual bug" (vs. a docs gap); low-value to prescribe pre-emptively, revisit if either happens in practice.
- **Existing requirements docs still carry the pre-2026-07-27 unbolded status line** — do **not** hand-normalise them mid-interview. The one-time backfill across all active feature documents is step 3 of `docs/plans/doc-reference-convention.md` (owner: `coder`, whose writes aren't doc-guarded); after it lands, `tico` only ever writes the new form (noted 2026-07-27).
- A `docs/requirements/` template file vs the inline template (only if the inline one drifts across features).
- Non-functional requirements section (performance, security) — add when a feature actually needs one rather than padding every doc.
- Project-scoped `agent` setting to make tico the default session agent in a requirements-heavy phase — only if launching via `--agent` proves to be friction.
