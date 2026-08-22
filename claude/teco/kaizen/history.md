# Kaizen — Change History: teco

> Dated log of actual changes to the `teco` agent. Most recent first.

## 2026-08-21 — Mid-run escalation: delegates can stop on a high-stakes fork and be resumed, instead of guessing (mid-run-escalation FR-1..FR-5)
- **What:** Three edits to `teco.md`, applied per `claude/docs/plans/mid-run-escalation.md` §2
  (analyst-reviewed, verdict approve with suggestions — `claude/docs/reviews/mid-run-escalation.md`
  — Findings 2-4 folded in during this implementation):
  1. **Step 2** — the ledger `Status` enum gains a new value, **`paused`** (a unit whose delegate
     stopped mid-run with an open question, now relayed and awaiting an answer); a `paused` row
     repurposes the `Deliverable` column to carry the question + relay date instead of a path
     (noted explicitly as the one `Status` value that breaks that column's normal path-typed
     convention — Finding 4), with a full seven-column example row added next to the existing
     sample. The "open a coordination doc" trigger gains a third, **reactive** condition: any unit
     that escalates via stop-and-ask forces a coordination doc into existence (backfilling a ledger
     row per already-dispatched unit) even below the 3-unit/gate threshold, since a paused unit's
     `agentId` and question must survive a compaction and only the ledger — not context — persists
     that.
  2. **Step 3** — the Subagent-awareness bullet now carves out one narrow exception to "cannot ask
     mid-run": a **high-stakes fork** (would change scope, touch something irreversible, or waste
     substantial downstream work if guessed wrong) may be stopped on and returned as the unit's
     result instead of guessed or held for the final report. The brief clause to fold into every
     dispatch (with a qualifying/non-qualifying worked example) is now scoped — skipped for a unit
     already classified **mechanical** per Model routing, since a mechanical dispatch structurally
     cannot hit a fork worth stopping for (Finding 3).
  3. **Step 4** — four new bullets: recognizing a paused result by shape and relaying it (first-order
     via `AskUserQuestion`, or in-report as a subagent — including the two-hop `SendMessage` chain
     this implies when teco itself is a delegated subagent: its own dispatcher must resume
     teco-as-subagent first, before teco-as-subagent can perform the inner resume — Finding 2);
     resuming the same delegate via `SendMessage` by its ledger `agentId` once answered (with the
     existing step-5 addressing-failure fallback cross-referenced); the non-blocking guarantee (a
     `paused` unit stalls only itself and its structural dependents, no cap/deadline/auto-escalation
     — deliberate, per the requirements doc); no fixed cap on stop-and-ask round trips per unit.
  Also updated: `claude/README.md`'s `teco` catalog entry (one clause describing the capability,
  inserted after the existing `SendMessage`/`agentId` sentence).
- **Why:** `claude/docs/requirements/mid-run-escalation.md` (Ready for design, confirmed
  2026-08-21) — the stakeholder wanted to relax the standing "no mid-run questions" rule for
  genuinely high-stakes forks now that `SendMessage`-based resume is proven (K-007, K-013), so an
  undecided fork doesn't get guessed into a deliverable or only surface after the fact. Designed by
  `cobb` per `claude/AGENTS.md`'s routing convention (agent/prompt engineering, not a codebase
  change); gated by `analyst` (approve with suggestions, no blocker) before this implementation.
- **Verified:** Read-through of the three landed `teco.md` edits against each of AC-1..AC-5 and
  against the plan's own §7 mapping; no hook, frontmatter, or tool-grant file touched anywhere in
  this change (confirmed by inspection — `claude/AGENTS.md` needed no edit, per the plan's own
  scope discipline). No automated suite covers prompt text; a live dry-run exercise of the actual
  relay/resume path is still a follow-up, not performed in this pass.
- **Plan items:** none opened.

## 2026-08-21 — Commit-authority note updated: universal interactive-mode grant supersedes "not extended" claim in part
- **What:** The Guardrails "Why the boundary differs from `tico`'s" bullet now (a) states
  explicitly that both teco's integrator grant and tico's own-doc grant are **unconditioned on
  interactive-vs-subagent mode** (they apply either way, tied to role not invocation), and (b)
  corrects the 2026-07-30 "not extended to any other specialist" claim, which the 2026-08-21
  universal grant (below) partially supersedes: every agent now separately carries a narrower
  interactive-only commit grant for its own verified work, void as a delegated subagent — teco's
  and tico's broader, mode-unconditioned grants are unaffected and remain the only ones of that
  shape.
- **Why:** `tico` reported (via a `kaizen_team` entry) that it lacked commit authority over
  subagent deliverables from a Mode-3 verification pass it orchestrated; the stakeholder, put on
  the spot for a decision, ruled beyond that narrow case — every agent gets an interactive-mode
  commit exception, not just tico/teco. Full rationale, the `claude/AGENTS.md` rewrite, and the
  `audit-team.sh` check-8 redesign: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — K-015 ✅ closed: dispatch-sizing rule validated on K-028's real oversized implementation

- **What:** K-028 (falkor-chat workflow timers) supplied the first live instance crossing the
  ~3-step/5-file boundary the 2026-08-11 sizing rule targets — its implementation touched
  `services.py`, `repository.py`, `schemas.py`, `api.py`, `config.py`, `app.py`, `executor.py`,
  plus `QUERIES.md`/`DESIGN.md`/`start_server.sh` and 5 test files (15+ files total), and hit a
  plan-level defect mid-implementation forcing a full mechanism redesign. teco did not hand this
  to one mega-dispatch: it split the implementation into named ledger units by concern —
  **U3a** ("core logic": `services.py`'s sweep/invariant, `executor.py` docstring) vs. **U3b**
  ("wiring": `schemas.py`/`api.py`/`config.py`/`app.py` + docs + tests) — and tracked the
  defect-driven rework as its own distinct rows (**U3a-fix**, **U3c**) rather than silently
  absorbing it back into a growing single dispatch. Per-unit costs stayed well inside the K-042
  baseline (458k tok/222 tools) despite the mid-run redesign: U1 (plan) 212k/42, U2 (plan gate)
  164k/49, **U3a (core logic, the largest single unit) 307k/134**, U5 (QA) 175k/59 — no unit came
  close to K-042's mega-dispatch cost, and no scope was silently dropped (QA: PASS, zero defects,
  12/12 planned items).
- **Why:** K-015 tracked the rule as unproven since 2026-08-11 — "a claim, same epistemic shape as
  K-013's unexercised `SendMessage` loop" — because every prior coordination's dispatches (K-026
  included) had stayed single-unit and never actually crossed the boundary that would exercise it.
- **Disposition:** ✅ confirmed — the rule holds under real, defect-heavy pressure, not just in
  the prompt. One refinement worth carrying into K-016's consolidation pass: the split that
  actually happened here was **by logical concern** ("core logic" vs. "wiring"), not a literal
  file-count tally against the ~3-step/5-file threshold at decomposition time — the concern-based
  split happened to keep every unit's footprint far below the threshold anyway. The rule's intent
  (bounded per-unit cost, no dropped scope) was met; its mechanism, as actually practiced, is
  closer to "cluster by concern" than "count files." Consider restating it that way if K-016
  touches this bullet.
- **Left open:** the parking-lot idea "architect plans annotate dispatch-unit boundaries" was
  *not* exercised here — architect's plan didn't pre-mark U3a/U3b clusters; teco derived the split
  itself at dispatch time. Still open, still worth raising when K-016 or that item is worked.
- **Plan items:** K-015 (✅ done, moved out of Active).

## 2026-08-21 — Optimization pass from a stakeholder-requested in-depth analysis: dispatch guard hook, status-flip carve-out, tool-grant reconciliation (K-012 ✅), Cost ledger column, gate-as-you-go, AskUserQuestion dual-mode

- **What:** seven changes from a stakeholder-requested "analyze teco in depth / optimize its way
  of work" session, three of them stakeholder-decided explicitly (status flips, AskUserQuestion,
  consolidation timing).
  1. **New `PreToolUse` hook `hooks/guard-agent-dispatch.sh` (matcher `Agent|Task`)** — escalates
     any `Agent` dispatch missing `subagent_type` to the human. The 2026-08-21 silent-
     `general-purpose` trap already had a prompt bullet, but prompt discipline alone had let two
     such dispatches through; this makes the omission mechanically impossible to land silently.
     Standalone agent-owned script (the `security-expert` exploitation-guard precedent), fail-open,
     `ask`-only, jq→python3. Tested through the deployment symlink: present/missing/empty
     `subagent_type` and garbage input all behave per contract.
  2. **Status-flip carve-out in `guard-coordination-doc-writes.sh` (stakeholder-approved):**
     before deferring to the shared core, the wrapper auto-allows an `Edit` on a `docs/**.md`
     whose old/new strings differ only in the canonical `**Status:**` field flipping to
     `archived` (python3 masks the field on both strings and requires byte-equality of the rest).
     Rationale: the K-026 close spent **five separate agent spawns** on one-token archival flips.
     Root `AGENTS.md`'s lifecycle section now names `teco` as the performer of the mechanical
     flip (by-kind owner table retained for anything beyond it); teco.md's milestone-close bullet
     and Guardrails updated to match. Tested: pure flip (relative + absolute path) → allow;
     flip+other change, flip to a non-`archived` token, non-docs path, `Write` → escalate.
  3. **Tool-grant reconciliation (K-012 ✅ closed).** Fresh-session probe (spawned teco run):
     runtime tools are exactly `Read, Bash, Agent, SendMessage, Write, Edit, mcp__cypher__query`.
     `ListAgents` absent (like the known `Grep`/`Glob` gap) — all three dropped from frontmatter
     per K-012's own pre-agreed disposition. `mcp__cypher__query` **live-verified** (a
     `kaizen_team` read returned; parking-lot item resolved) — the "not yet live-verified"
     caveats on the CPG-freshness duty removed. Guardrails' runtime-tool-set bullet rewritten:
     frontmatter now matches probed reality; the "a grant is not proof" lesson retained.
  4. **Ledger gains a `Cost` column** (step 2): record the completion notification's reported
     tokens/tool-uses per unit — the data K-015 (dispatch-sizing validation) needs to ever be
     judged against numbers. Feasibility confirmed this session: the probe's own completion
     notification carried `30982 tokens / 1 tool use`.
  5. **Gate-as-you-go** (step 2): sequence each unit's review gate immediately after its
     delivery, never batched at coordination close — K-026's pauses left four delivered units
     ungated and uncommitted across sessions, exactly the crash-exposure this rule shrinks.
  6. **AskUserQuestion dual-mode (stakeholder-approved):** frontmatter gains `AskUserQuestion`;
     Pause-vs-proceed now distinguishes first-order runs (`claude --agent teco` — ask the
     decision as a structured question and continue) from subagent runs (tool withheld by the
     harness — return the decision summary as before). K-026 showed teco frequently runs
     first-order, where every decision point previously cost a full stop-and-report.
  7. **Stale-text fixes:** Guardrails' "and your own kaizen inbox" clause dropped (inbox deleted
     2026-08-21); step 5's "don't assume an enumeration tool" hedge replaced with the probed
     fact. (The hook wrapper's own stale inbox glob/comment turned out to be already fixed by a
     concurrent edit outside this session — found mid-change when the file differed from its
     first read.)
- **Deferred by stakeholder decision:** the full consolidation/KB-split pass filed as **K-016**
  (high) for a dedicated `cobb` pass rather than run in the same session that touched the prompt
  in seven places. Two new parking-lot ideas: architect-side dispatch-boundary annotations
  (feeds K-015), cross-session slug-echo convention.
- **Why:** stakeholder asked for an in-depth analysis of teco and then "let's get to work" on its
  findings. The through-line: promote prompt-level discipline to mechanical enforcement where it
  has already failed (1), stop paying agent spawns for mechanically-verifiable edits (2), align
  declared capability with probed reality (3), and attack the documented credit-burn pattern with
  data capture + smaller exposure windows (4, 5).
- **Verified:** `bash claude/scripts/audit-team.sh` — full PASS before and after (diff, not bare
  gate). Both hook scripts `bash -n` clean and scenario-tested through `~/.claude/agents/teco/`.
- **Docs touched:** `claude/teco/teco.md` · `claude/teco/hooks/guard-agent-dispatch.sh` (new) ·
  `claude/teco/hooks/guard-coordination-doc-writes.sh` · root `AGENTS.md` (lifecycle/flip
  authority) · `claude/AGENTS.md` (hook machinery) · `claude/README.md` (teco row + deployment
  hooks note) · `claude/teco/kaizen/{plan,history}.md`.
- **Plan items:** K-012 ✅ (moved here) · parking-lot `mcp__cypher__query` probe ✅ · K-015
  updated (Cost column feeds it) · K-016 opened · two parking-lot ideas added.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes. In the same session, `kaizen_teco` (the per-agent graph this file's own 2026-08-20 entry below describes) was also retired — `graph-dba` `GRAPH.DELETE`d it after cross-checking all 5 of its entries against this file's 2026-08-21 distillation entry (below), all already promoted or kept-open-as-`K-018`-and-cleared.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there (including this agent's own 9-entry distillation, immediately below) has already been distilled and cleared. Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — Coverage fix: dropped stale "kaizen-inbox entry" from the commit-authority grant (team certification, §7 fold-in)

- **What:** The commit-authority grant ("The grant" bullet under Guardrails' `Bash` section)
  listed "a plan, review, test plan/report, or kaizen-inbox entry you've verified fits" as the
  deliverable kinds teco may commit for a coordinated specialist. Dropped "or kaizen-inbox entry."
- **Why:** Caught during a user-requested full team-coherence certification's §7 lint fold-in.
  Since the 2026-08-20 team-wide graph migration, no agent produces a fresh `kaizen/inbox.md`
  entry any more — every agent's raw learnings capture writes straight into the shared
  `kaizen_team` graph via `mcp__cypher__query`, not a file. A specialist teco coordinates can
  therefore never hand back a kaizen-inbox-entry deliverable for teco to commit; the clause
  described a delivery shape that no longer exists. Same stale-phrase pattern found and fixed the
  same pass in `tico.md`'s commit-authority grant (`claude/tico/kaizen/history.md`, same date).
- **Verified:** `bash claude/scripts/audit-team.sh` — same 113 PASS / 2 pre-existing FAILs before
  and after (diff, not bare gate).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-21 — Distilled all 9 pending raw-capture entries from `kaizen_team` (`cobb`, §5 pass)

- **What:** `cobb` ran the full agent-maintenance §5 distillation against every `kaizen_team`
  node with `author:'teco'` — the raw capture written since the 2026-08-20 team-wide graph
  migration (teco's `kaizen/inbox.md` itself stays a frozen, already-imported snapshot; this is
  the *new* capture that accrued in the graph afterward). All 9 verified, dispositioned, and
  cleared from the graph in this pass; none discarded outright.
  1. **`7994edd7…` (2026-08-15, nested notification bubbles to the live ancestor; `SendMessage`
     force-resumes a dormant target) → promoted** to
     `skills/agent-standards/claude-code.md`, new "Nested-delegation notification routing"
     subsection, explicitly dated/caveated as a live observation, not a confirmed stable
     contract — matches the entry's own `suggestedHome: unsure`.
  2. **`e40a95fe…` (2026-08-15, a delegate that can't address teco by name is relayed through
     "main" as a `<system-reminder>` block — legitimate, not an injection) → promoted**, same
     new subsection, paired with entry 1 (same K-026 incident).
  3. **`a77a32a3…` (2026-08-16, parallel dispatches sharing a live shared-DB fixture
     cross-contaminated despite disjoint files) → promoted** into `teco.md` §3 "Dispatch,"
     sharpening the existing same-file/shared-key serialization rule to cover concurrent
     live-suite exercise, not only destructive overlap.
  4. **`f7070b80…` (2026-08-17, a `completed` notification's `<result>` can be a stale
     mid-task placeholder from a delegate's own unfinished background step) → promoted**
     into `teco.md` §4 "Track what's in flight," new bullet after "Never state or predict a
     pending delegate's result."
  5. **`9ec17ba5…` (2026-08-18, a clean QA PASS on a brand-new mechanism isn't full
     state-space coverage) → promoted** into `teco.md` Guardrails, new bullet next to the
     review-gate guardrail.
  6. **`f1a2b3c4…` (2026-08-20, a coordinator's "proceed" ≠ real user approval on a
     harness-gated write; never relay a delegate's self-modify-permissions proposal) →
     promoted**, split across two homes: `teco.md` Guardrails (the operative rule for teco's
     own coordination authority) and `skills/agent-standards/claude-code.md` Hooks section (the
     underlying harness-classifier fact, a sibling to the existing auto-mode Bash-classifier
     bullet).
  7. **`a2b3c4d5…` (2026-08-20, `guard-destructive-ops.sh` didn't fire for a live
     `GRAPH.DELETE` run inside a nested subagent's Bash context) → kept open, not promoted
     to teco.md.** This is a hook-wiring question the entry itself flagged for `cobb` to
     triage, not a teco-side behavior gap. `cobb` re-fetched `code.claude.com/docs/en/hooks`
     (2026-08-21): hooks are documented to fire identically for a subagent whether run as the
     main session agent or a nested delegate, no exception noted — so the observed gap isn't
     explained by any documented behavior. Filed as `K-018` in `claude/cobb/kaizen/plan.md`
     (high priority), with the leading hypothesis that it's actually explained by entry 9
     below (a `subagent_type`-omitted dispatch silently running as `general-purpose`, with no
     `graph-dba` hooks at all) — not confirmed from the coordination doc alone, needs a live
     re-check on a future `graph-dba` dispatch. Node cleared from the graph regardless per §5's
     "kept open" disposition — the durable record is this note plus `K-018`, not a lingering
     graph node.
  8. **`b3c4d5e6…` (2026-08-20, `mcp__cypher__query` table rendering truncates a cell at
     ~300 chars; FalkorDB Cypher distinguishes `\n` from `\\n` in string literals) → partially
     promoted.** The truncation half was **already fully documented** in
     `cypher-mcp/README.md` ("Result format and truncation" — `CYPHER_MCP_MAX_CELL`/
     `CYPHER_MCP_MAX_CHARS`), confirmed by `cobb` re-deriving the exact same 300-char cap
     firsthand while reading these very entries out of the graph, independent of the entry's
     own claim — discarded as a duplicate. The `\n`/`\\n` escaping half was genuinely
     undocumented — promoted to the same README, "Writing through this tool" section.
  9. **`b1e3a1f0…` (2026-08-21, `Agent` silently defaults to `subagent_type: general-purpose`
     when omitted — no error, no hooks/persona/tools for the named agent) → promoted**, high
     priority, into `teco.md` Guardrails as a new bullet next to the existing
     narrower-than-frontmatter runtime-tool-set warning — same "don't trust the frontmatter/the
     brief alone" family. Cross-referenced by entry 7's `K-018` as the likely (unconfirmed)
     root cause of that separate incident.
- **Why:** user asked to "work on teco's inbox" — the file `kaizen/inbox.md` is itself a frozen,
  already-distilled 2026-08-20 snapshot, so the live equivalent is teco's pending raw capture in
  the shared `kaizen_team` graph; first distillation pass against it since the migration.
- **Verified:** every entry's full field text was read via `size()` + multi-column `substring()`
  paging (the harness's own per-cell display truncates at ~300 chars) before any disposition
  decision, not acted on from a truncated partial read; the one live-checkable external claim
  (hook parity for nested subagents, entry 7) was re-checked against current official docs rather
  than taken on the entry's own framing.
- **Plan items:** none opened in `teco/kaizen/plan.md` itself (K-018 opened in `cobb`'s own
  plan.md instead, per its "flagged for cobb" suggested home).
- **Docs touched:** `claude/teco/teco.md`, `claude/teco/kaizen/history.md` (this entry),
  `skills/agent-standards/claude-code.md`, `cypher-mcp/README.md`, `claude/cobb/kaizen/
  {plan,history}.md` (cross-artifact bookkeeping, logged from `cobb`'s side).

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_teco`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_teco` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — its 5 pre-existing
  entries were parsed out programmatically and imported into the graph verbatim (entryId
  assigned, `author: 'teco'`), preserving every field; its own header explains the freeze and
  gives the live-read query. The trailing "Your write guard allows exactly this inbox path"
  clause was dropped — the write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP
  tool, so it no longer applies to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer (`cobb`) verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Step-table sizing rule: incident narrative moved out of operative prompt text
- **What:** The dispatch-sizing bullet (§3, "Delegate with complete briefs") kept its operative rule verbatim (~3-step/5-file decomposition boundary, one unit per step/small cluster) but dropped the inline K-042 incident narrative (the 458k-token/222-tool-call whole-landing dispatch, the dropped-test-files detail, the stakeholder quote) in favor of a dated pointer to this file's own 2026-08-11 entry, which already carries the full story. −~85 words in the prompt body.
- **Why:** User-directed prompt-verbosity reduction, item 2 of the parked diagnosis (`cobb/kaizen/plan.md`) — an origin story belongs in the change log it's already recorded in, not repeated inline in the instruction every session pays to load. The rule itself is unchanged; only the narrative moved.
- **Plan items:** —

## 2026-08-19 — CPG freshness centralized here; `mcp__cypher__query` added
- **What:** Took over the CPG freshness check that `analyst`/`architect`/`coder`/`tdd-engineer`/`frontend-engineer`/`qa-engineer` used to run themselves (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`). Added `mcp__cypher__query` to `tools:`; new §3 bullet — guess the graph key, run the freshness recipe (`skills/cpg-analysis/references/freshness.md`) before dispatching a unit likely to touch a CPG, state the result in the brief. Guardrails note flags the grant as **not yet live-verified** — teco's frontmatter already has a known live-tool-set-narrower-than-declared gap (`Grep`/`Glob`, verified 2026-08-10), so `mcp__cypher__query` needs the same live probe before this duty can be trusted.
- **Why:** User-directed prompt-verbosity reduction surfaced the freshness check as ~130 words duplicated verbatim across six agents; user chose full centralization on teco (accepting the standalone-run capability loss) over a per-agent dedup via a shared skill pointer.
- **Plan items:** new parking-lot item — live-verify the `mcp__cypher__query` grant on a real coordination before relying on the freshness duty.

## 2026-08-16 — K-013 ✅ and K-014 ✅ closed by real evidence from K-026's own coordination (review-only, no prompt change)

- **What:** reviewing `teco/kaizen/plan.md`'s active table against the just-closed K-026
  GraphRAG-eval coordination ledger (`falkor-chat/docs/plans/graphrag-eval-coordination.md`)
  found two of the four open items had actually been exercised, live, during that coordination —
  the datapoints just hadn't been written back yet.
  - **K-013 (exercise `SendMessage` continuation for real) — closed.** The ledger's `U2b-gate`/
    `U2b-fix` rows show `analyst` gated Unit 2b "needs changes" (Blocker B-1, Major M-1) → a fix
    dispatched to `tdd-engineer` → the re-gate row reads `analyst (resume a4b2370c17130742d,
    re-gate)` — `teco` resumed the *same* analyst by its own ledger `agentId` instead of a fresh
    spawn, and that analyst "independently re-verified both fixes itself... re-ran suites...
    checked `GRAPH.LIST` directly for B-1" without being re-briefed on its own earlier findings.
    Real evidence context is preserved across a `SendMessage` resume, not just a claim.
  - **K-014 (agentId ledger cell has no enforcement) — closed, no checker added.** Every one of
    ~20 unit rows in K-026's multi-session coordination has its `agentId` cell filled, with
    exactly one explicitly-justified exception (`U-bug`, inherited from a prior session with
    genuinely no id to carry forward — noted as unresolvable, not silently blank). Self-discipline
    held under sustained real load; per K-014's own stated criterion ("if it doesn't get skipped,
    leave it"), no checker is warranted from this evidence.
  - **K-012 (`ListAgents` materializes) and K-015 (dispatch-sizing rule) — still open.** Neither
    got exercised: nothing in the K-026 coordination invoked `ListAgents`, and every unit
    (including the closing qa-engineer/doc-closeout ones) stayed single-unit, never crossing the
    ~3-step/5-file boundary that would test the sizing rule.
- **Why:** the user asked directly, mid-session, what became of "the teco improvement item we were
  testing" — prompted this reconciliation rather than letting the evidence sit unlogged in a
  component's coordination doc where it would never surface again.
- **Verified:** read the coordination doc's ledger rows directly (not a relayed summary) before
  crediting either closure.
- **Plan items:** K-013 ✅, K-014 ✅ (moved to plan.md's done-notes block); K-012, K-015 remain
  active, both updated with a 2026-08-16 "still no evidence" note.

## 2026-08-16 — Step 4 gains a misrouting-vs-staleness distinction for incoming messages (cobb, cross-session peer-addressing near-miss)

- **What:** one new bullet in step 4 ("Track what's in flight"), right after the existing
  "incoming resume/pause message: intent authoritative, facts aren't" rule: a message describing
  a task/coordination **absent from this session's own ledger or active context** is a
  *misrouting* signal, not merely a staleness one — `SendMessage` addresses peers by bare agent
  name, which resolves ambiguously when more than one independently-launched session shares that
  name, so the sender may simply have the wrong `teco`. Pause and confirm identity with your own
  user before doing *anything* in response, even read-only verification.
- **Why:** a live incident, not a hypothetical — a `teco` session (a different one, mid-coordination
  on `cpg-agent-adoption`) received a full K-026 resume brief from `cobb`, who had picked it off
  `ListAgents` assuming (wrongly, from stale prior-session context) it was the K-026 coordinator.
  That session's own human caught the mismatch; by then it had already done a small amount of safe
  read-only work (a test re-run, a state-restoring reseed) before declining — harmless here, but
  exactly the kind of spend this bullet now heads off explicitly instead of leaving to luck/a human
  catch. Full mechanism writeup lives in `skills/agent-standards/claude-code.md`'s new "Cross-session
  peer addressing" section (`cobb`'s companion promotion, same run).
- **Verified:** `bash claude/scripts/audit-team.sh` clean before and after.
- **Plan items:** none opened — direct fix from a live incident, not a backlog item.

## 2026-08-15 — Review-only pass: filed K-015 (validate the dispatch-sizing rule live)

- **What:** no prompt change. Stakeholder asked what's next for `teco`, specifically flagging
  the "big work packages" episode from the last end-to-end run. Reviewed `plan.md`/`history.md`/
  `inbox.md` (one undistilled entry remains, 2026-08-12 — see below) and confirmed the dispatch-sizing
  rule that answers that exact episode (K-042 Landing 1, 2026-08-11 entry) has shipped in
  `teco.md` §3 but has **zero live-run evidence since** — no coordination has exercised it under
  real conditions. Filed as **K-015** (high priority) in `plan.md`, cross-linked to close
  alongside the still-open **K-012** (`ListAgents` fresh-session probe) and **K-013**
  (`SendMessage` continuation) on the same next live run, since one end-to-end coordination can
  produce evidence for all three at once.
- **Why:** per the kaizen convention, a review-only pass still records ideas surfaced during
  review rather than letting them evaporate at the end of the conversation.
- **Plan items:** K-015 filed (🔵 proposed).

## 2026-08-15 — Distilled the 2026-08-12 inbox entry: discarded from teco (suggested home didn't fit), redirected to `cobb`'s own kaizen

- **What:** ran the agent-maintenance skill §5 procedure on the sole entry in `teco/kaizen/inbox.md`
  (2026-08-12, "a review-gated unit with no coordination doc is nearly unrecoverable after a
  mid-session credit crash").
- **Verified — incident is real, not embellished:** `docs/plans/kaizen-inbox-distillation2-coordination.md`
  (Owner: `teco`, `Tracks: — (no backlog id; stakeholder-triggered cobb sweep)`) and
  `docs/reviews/kaizen-inbox-distillation2.md` confirm it exactly: `cobb` ran a 39-file team-wide
  kaizen distillation directly (not via `teco`), `analyst` gated it "needs changes," the fix pass
  had to be resumed cold (`U1` row: `agentId (prior session, not resumable)`), and the recovering
  session (operating as `teco`) reconstructed state from the review alone, confirmed **K-041**
  already covered one of the review's "open questions" (matches the inbox entry's claim exactly),
  and closed cleanly (commit `db39ade`). The entry's cited filename
  (`kaizen-distillation-2026-08.md`) was the doc's pre-rename name — `analyst`'s own U3 renamed it
  to `kaizen-inbox-distillation2.md`, noted in the coordination doc; not a fabrication, a stale
  filename from before the rename.
- **Why the suggested home ("teco.md step 2/3: open the ledger at first dispatch for any
  review-gated sequence") doesn't fit teco:** that rule **already existed** — step 2 has opened a
  coordination doc whenever *any* unit carries a review gate (not only at the 3-unit threshold)
  since the 2026-08-10 ledger pass, which **predates** this incident. The incident's `U1` (the
  original `cobb` sweep + `analyst` gate) never had a ledger because it was **never coordinated by
  teco at all** — direct stakeholder → `cobb` → `analyst`, by design (`cobb` is meant to be
  directly invokable for agent-maintenance work, per its own `description`). Teco's existing rule
  had no chance to apply; there is no teco-side gap to close.
- **Disposition: discard from teco's inbox, redirect the underlying observation to `cobb`'s own
  kaizen** (`claude/cobb/kaizen/plan.md`, parking lot) as a self-directed note — the actual
  mitigation that saved the recovery (the review's self-sufficient baseline-commit + explicit
  scope list) is already `analyst`'s standing review-header practice, so this is confirmed-good-
  practice-under-fire, not a new rule; logged as a soft parking-lot idea, not a prompt change, per
  §5's "highest bar: every session pays for it" for anything landing in an always-loaded prompt —
  one data point, no repeat, no runtime-behavior gap identified.
- **Plan items:** none advanced in teco's own plan (K-012/013/014/015 unaffected); see
  `claude/cobb/kaizen/{plan,history}.md` for the redirected entry.

## 2026-08-11 — Dispatch-sizing standing rule + 4 smaller promotions from the inbox distillation (stakeholder's "never a landing this big again" directive)

- **What:** `cobb` distilled all six entries in `teco/kaizen/inbox.md` (§5), triggered by a
  stakeholder report of several sessions blowing past 400k tokens. Five promoted into `teco.md`,
  one discarded as already self-resolved.
- **Diagnosis (the actual ask):** verified verbosity vs. orchestration as the cause, not assumed.
  Agent prompt bodies are 42–274 lines (already through two team-wide slimming passes,
  2026-07-11/2026-07-24); `coder.md` is ~1.4k tokens, `teco.md` ~5.7k. The cited incident — K-042
  Landing 1, one `coder` dispatch covering 6 plan steps / ~10 files, **458k tokens / 222 tool
  calls / ~45 min**, per `/context` — is context-length growth from the sheer volume of file
  reads/diffs/test-run output accumulated across one unbroken 6-step, 10-file session, not from a
  large system prompt (which is a small, roughly-constant fraction of that context, not something
  resent 222 times). The same unit's `analyst` gate then found 3 of the 11 test files the plan
  names — 3 of the 5 rewired consumer bindings — silently dropped from its own stated scope — a
  correctness cost, not just a token cost, from the same oversizing. **Verdict: orchestration
  (dispatch sizing), not verbosity, is what's driving these specific blowouts.** No verbosity
  contributor found worth a further slimming pass.
- **Promoted into `teco.md`'s "Delegate with complete briefs" (§3):**
  1. **Dispatch-sizing rule** (the core promotion) — a plan step table spanning more than ~3 steps
     or 5 files is the decomposition boundary: one unit per step/small cluster, sequenced as
     dependent same-file briefs, never one landing-wide mega-brief. Tied explicitly to the
     stakeholder's own words, quoted in the prompt, so the rule can't silently erode.
  2. QA-found-defect fix briefs: read the defect's own root-cause docstring/AC, not just the
     suggested-fix line (a live repro proves one path broken, not the only one).
  3. Documentation-impact scan: for a rename/removal blast radius, sweep unfiltered
     (`grep -rn <token> .`), not `--include='*.ext'` — extension globs silently miss dotfile
     config (`.env.example`).
  4. Track-what's-in-flight: `SendMessage` a premise-invalidating finding to a still-running
     sibling immediately, don't hold it.
  5. Track-what's-in-flight: an incoming resume/pause message's *intent* is authoritative, its
     *factual state claims* are not — re-verify against `git log`/the ledger before acting on them.
- **Discarded:** the "specialist's own knowledge base can be stale in a build-version-specific
  way" entry — already fully self-corrected: `claude/graph-dba/falkordb-quirks.md`'s
  `db.indexes()` entry already carries the "corrects the earlier claim... verified 2026-08-10"
  note the inbox entry was asking for, written by `graph-dba` itself during the same run.
- **Why:** stakeholder-reported context blowouts across recent sessions; the stakeholder's own
  quote ("please never again create a landing so big") had been sitting as an unpromoted inbox
  entry since 2026-08-10 despite being an explicit standing directive.
- **Verified:** `bash claude/scripts/audit-team.sh` clean before and after (diff, not a bare
  gate). No personal identifiers introduced.
- **Bookkeeping note for the next distillation pass:** the inbox held **6 headed (`## `) entries**
  plus one **headless continuation block** — a stray `- **Evidence:**` bullet with no `## ` heading
  of its own, sitting directly under the "K-042 Landing 1... ran past 370k tokens" entry and
  narrating the same unit's *completed* numbers (458k tokens / 222 calls / the dropped-test-files
  finding). Treating it as part of that preceding entry (not a 7th, separately-dispositioned one)
  is correct — don't re-count it as a separate entry in a future pass.
- **Docs touched:** `claude/teco/teco.md` · `claude/teco/kaizen/{history,inbox}.md`.

## 2026-08-10 — Coordination state moves out of the context window: canonical ledger, in-flight tracking, `agentId`-addressed continuation

- **What:** the largest structural pass on `teco.md` since its creation, from a `cobb` review of
  how teco coordinates, tracks units, and routes between *running* and *fresh* agents.
  1. **Frontmatter:** `tools:` gained `ListAgents` — step 5's "fall back to a cold spawn only if
     the identifier no longer resolves" was previously untestable from inside teco.
  2. **Steps 3 and 4 split into sub-bullets** before anything was added to them (two prior
     parking-lot deferrals, §7 dimension 4): each now carries one rule per bullet.
  3. **New step 2 ledger, mandatory at 3+ units or any gated unit** (replacing the unmeasurable
     "for large or long-running work"): `| Unit | Owner | Agent id | Status | Deliverable | Gate →
     verdict |`, with a closed status vocabulary (`queued` · `in-flight` · `delivered` · `gated` ·
     `accepted` · `abandoned`). Stated explicitly: the ledger, not teco's context window, is where
     a unit's state lives.
  4. **New step 1 resume path:** an existing coordination doc for the slug is the state of record —
     read it and reconcile `in-flight` rows against `git log` and the tree before dispatching.
  5. **New step 4, "Track what's in flight":** `Agent` runs in the background by default; never
     state or predict a pending delegate's result; a transient platform failure (500/timeout/kill)
     is not a deficient result but a re-dispatch with a **state-recovery brief** (inspect
     `git status`/`git diff`, continue from actual state); a unit superseded mid-flight is
     `abandoned`, its result discarded.
  6. **Identity recorded at dispatch, always** — every `Agent` call returns an `agentId`; it goes
     in the unit's ledger row immediately, not "when a follow-up seems likely" (which asked teco to
     predict the future). That id is what `SendMessage` addresses; `ListAgents` is how resolution
     is checked.
- **Why (evidence, not opinion):** `SendMessage` appears **42×** across this box's transcripts and
  **0×** in any confirmed teco run — K-007 shipped 2026-07-29 and had never fired. All five real
  coordination docs on disk invent a different Status table and **none** records who is running or
  how to reach them. The delegate id lived only in teco's context, the one thing lost to
  compaction, on exactly the long coordinations where continuation matters.
- **Two proven failure modes turned into rules.** `model:"haiku"` doc-closeouts fabricated numbers
  twice (2026-07-31: a 70% threshold existing nowhere in the codebase; 2026-08-09: a breakdown that
  doesn't arithmetically add up), both caught only by teco's own re-verification. Step 3's model
  routing now confines haiku to **mechanical** units and requires summarizing briefs to carry
  *"state only figures you directly observed in this run's command output; never decompose a total
  into a breakdown you did not observe"*; step 5 pairs it with mandatory re-verification of any
  summarized number — the cheaper model tier never buys out the verification.
- **Five standing practices promoted from the user's AutoMem file into the committed prompt**
  (double analyst gate — plan gate **and** diff-scoped re-gate, replacing the weaker "and/or";
  mutation-testing green-on-arrival tests; shared-file serialization + single ownership of shared
  DB state; independent verification of a self-reported recovery; the tree-mutating-git
  prohibition — `stash`/`checkout <path>`/`restore` — now explicitly binding **inside implementer
  briefs**, with `git show <ref>:<path>` named as the safe baseline read). A live probe confirmed
  **the memory *index* reaches a subagent but not the entry bodies**: teco saw the
  `teco-process-lessons` one-line gloss and nothing behind it, so it had a teaser it could not act
  on. The committed prompt is the right home regardless.
- **Also:** Guardrails' dense commit-authority paragraph split into grant / never-mutate-the-tree /
  boundary-vs-`tico` / no-hook-backstop (the parking-lot note said to split it the next time
  Guardrails gained an addition, and it just did); the milestone-close bullet now states that root
  `AGENTS.md`'s by-kind flip table **controls over a document's own `Owner:`** where they disagree.
- **Verified:** `claude/scripts/audit-team.sh` — 98 PASS / 0 FAIL before, re-run clean after (diff,
  not a bare gate, per `agent-maintenance` §4).
- **Plan items:** closed three parking-lot items (step-3 density, Guardrails bullet split,
  model-routing evidence clause); opened K-012, K-013, K-014.

## 2026-08-10 — Learnings inbox distilled: 5 entries → 2 to teco.md, 2 to project docs, 1 closed by probe

- **What:** processed every pending entry per `agent-maintenance` §5.
  1. **2026-07-31 + 2026-08-09 (haiku doc-closeout fabrications, two independent instances)** →
     **promoted to `teco.md`** (step 3 model routing + step 5 numeric re-verification). Two
     datapoints of the same shape made this the highest-value entry in the inbox.
  2. **2026-08-09 (archival-flip authority: by-kind table vs. a document's own `Owner:`)** →
     **promoted to root `AGENTS.md`** lifecycle section (the table now says it controls) **and** one
     clause in teco's own milestone-close bullet. A fact about the project's doc convention belongs
     in project docs, not hoarded in one agent's prompt.
  3. **2026-07-29 (`node` not on `PATH` on this WSL2 box; two sessions rediscovered the
     workaround)** → **promoted to `falkor-chat/AGENTS.md`**, next to the bootstrap env note.
  4. **2026-07-29 ("continue via SendMessage" not backed by an available tool)** → **closed by
     live probe**, not promoted: a 2026-08-10 read-only probe of a real teco run shows
     `SendMessage` present as a full tool definition with **no** ToolSearch step and **no**
     deferred-tool reminder anywhere in its context. The entry described the pre-K-007 state
     (`SendMessage` was absent from `tools:` until 2026-07-29); it is no longer true.
- **Inbox cleared.**
- **Plan items:** none (distillation).

## 2026-07-30 — Commit authority formalized: `Bash` may now `git add`/`git commit` a coordinated deliverable, by explicit path
- **What:** Documented, for the first time, teco's authority to `git add`/`git commit` a
  specialist's deliverable it is actively coordinating — a plan, review, test plan/report, or
  kaizen-inbox entry it has already verified fits (step 4) — by explicit path, one coherent unit
  per commit, never `git add -A`/`.`/`-a`, never `push`/`reset`/`rebase`/amend, never anything
  outside the coordination it's actively running. Three touch points, no frontmatter/tool change
  (`Bash` was already granted): (1) Guardrails gained a dedicated bullet, placed right after the
  existing "you coordinate, you don't do the specialists' jobs" bullet, which now drops its stale
  "never mutating the tree" clause since that's no longer literally true; (2) step 4 gained one
  sentence — commit a verified deliverable rather than leave it sitting uncommitted; "verified but
  uncommitted" is now explicitly unfinished integration, not a stopping point; (3) cross-referenced
  in `claude/README.md` (teco + tico rows) and a new paragraph in `claude/AGENTS.md`'s Hook
  machinery section explaining this is prompt-level, not hook-backed. **Scoping deliberately
  differs from `tico`'s existing grant** (2026-07-23): tico's commit scope mirrors its own
  Write/Edit guard exactly (it only ever commits what it itself authored); teco's commit scope is
  wider than its own Write/Edit guard (which reaches only the coordination doc + its inbox)
  because its role — integrator of a whole coordinated unit's output — is structurally different
  from tico's. Both grants are pinned to the same stakeholder line so neither reads as
  self-expanding: "tico and teco are special and have coordination rights."
- **Deterministic backstop added:** `claude/scripts/audit-team.sh` gained **check 8** —
  `COMMIT_AUTHORS=("tico" "teco")`; every other agent's `<name>.md` fails the audit if it ever
  comes to claim `git add`/`git commit` authority, and `tico`/`teco` fail if their documented
  grant ever goes missing. This exists because no `PreToolUse` hook can gate a *prose* capability
  the way the doc-scoped/destructive-ops hooks gate Write/Edit paths and Bash command patterns —
  the grep-based check is the only mechanical trip-wire available for "did commit authority
  quietly spread to a third agent." Full audit re-run clean (95+ PASS, 0 FAIL) after the change.
- **Verified the four commits that prompted this change were safe, not just retroactively
  justified.** Read all four (`15d3ad5`, `4fe43a0`, `10f13ae`, `38e020d`) via `git show --stat`:
  each touches exactly the files its subject line names — `docs/reviews/cpg-getting-started.md`
  (analyst's review); `docs/test-plans/cpg-getting-started.md` +
  `docs/test-reports/cpg-getting-started-report.md` (qa-engineer's plan+report, one coherent
  unit); `claude/analyst/kaizen/inbox.md` (analyst's own learning); `claude/qa-engineer/kaizen/inbox.md`
  (qa-engineer's own learning) — no unrelated file in any diff, consistent with explicit-path
  staging rather than `git add -A`/`-a`. No `push`/`reset`/`rebase`/amend in the sequence (four
  distinct hashes, ~30s apart, matching four sequential `git commit` calls, not one commit
  rewritten). This matches the newly-formalized scope precisely — every committed file was a
  verified deliverable from a specialist teco was actively coordinating (the
  `docs/manuals/cpg-getting-started.md` review-gate rollout), none of it teco's own authorship.
  Nothing found that needed flagging as unsafe.
- **Why:** stakeholder decision, relayed via `cobb`: extend commit/coordination authority to
  `tico` and `teco` specifically (declining `cobb`'s earlier 2026-07-30 recommendation to instead
  extend narrow per-doc-kind commit rights to `analyst`/`qa-engineer`) — closes the gap between
  what teco had already done once in this session (four commits, undocumented authority) and what
  its prompt claimed ("never mutating the tree", no carve-out at all).
- **Plan items:** none opened (the fix was direct, not a backlog item); one §7 minor logged to
  `plan.md`'s parking lot (bullet density).

## 2026-07-29 — Manuals join the routing table, handoff contracts, doc scan, and review-gate defaults
- **What:** Four small additions reflecting tico's new Mode 2/3 (didactic explanation + user-manual maintenance, same day): (1) a new routing-table row — live explanations stay pause→user (tico isn't a delegation target), but a self-contained manual write/update is delegable to tico like any other subagent deliverable; (2) the tico handoff-contract line now names `docs/manuals/<slug>.md` alongside the requirements doc; (3) the documentation-impact scan bullet now lists user manuals (flag, don't write — `tico` owns them); (4) the "Work ships independently reviewed" guardrail gained a manuals entry: split by claim — `qa-engineer` verifies walkthroughs against the running app, `analyst` checks architectural/factual claims and clarity. The manuals-delegable routing row also notes the review gate still applies when teco routes a manual update this way.
- **Why:** user ruling following the 2026-07-29 team certification, which flagged manuals as the one doc kind with no independent-review gate; user chose the qa-engineer/analyst split (behavioral vs. everything else) and "mandatory in teco + offered in tico's first-order sessions" for how forced the gate should be.
- **Plan items:** none.

## 2026-07-29 — PII leak fixed in this file (found by the team certification pass)
- **What:** The K-009 entry below (added earlier the same day) had embedded the literal
  flattened `~/.claude/projects/...` transcript-directory path, which leaks the OS username —
  genericized to `<flattened-repo-path>`. Working-tree fix only; the leak reached one shared
  commit (`e7ec4a3`) before being caught — not rewritten, per the repo's don't-rewrite-shared-history
  norm.
- **Why:** Surfaced by `claude/scripts/audit-team.sh` check 7 during the 2026-07-29 team
  certification (see cobb's kaizen history for the full pass).
- **Plan items:** none.

## 2026-07-29 — Learnings inbox distilled: 3 entries → 1 promoted to teco.md, 1 to agent-standards, 1 discarded as duplicate
- **What:** Processed all three pending entries in `kaizen/inbox.md` (agent-maintenance skill §5):
  1. **2026-07-25 — "`.mcp.json` server materializes only at session start; subagents inherit MCP tools from the parent session"** — genuinely new, not previously captured anywhere in the repo (checked `AGENTS.md`, `cypher-mcp/README.md`, `skills/agent-standards/claude-code.md`). **Promoted** to `skills/agent-standards/claude-code.md` § MCP → Lifecycle (a harness-level fact, not teco-specific — belongs in the on-demand reference cobb maintains, not an always-loaded prompt), with a `Verified: 2026-07-25` stamp and the cpg-query-access delivery as evidence.
  2. **2026-07-25 — "verifying 'no new audit failures' needs a diff against the last commit, not a re-read of the gate's verdict"** — checked `skills/agent-maintenance/SKILL.md` §4 and found it **already promoted**, word-for-word disposition, same origin date (2026-07-25) and same task (`docs/plans/cpg-query-access.md` rework). **Discarded** as a duplicate of an already-landed promotion — nothing to do.
  3. **2026-07-27 — "a brief that fences off `claude/` silently disables the delegate's own learnings inbox"** — teco's own coordination mistake, still live risk (any future brief that excludes `claude/` for collision-avoidance repeats it). **Promoted** into teco's own prompt (step 3, appended to the model-routing sentence): carve out the delegate's `kaizen/inbox.md` explicitly, or have the learning come back in the report, whenever a brief fences off a subtree containing it.
- **Why:** user asked to process the inbox after the K-006/008/009/010/011 backlog pass. Each entry got the full §5 treatment (verify still true / not already documented, route to exactly one destination, log, clear) rather than a blanket append.
- **Plan items:** none (inbox distillation, not a plan item).

## 2026-07-29 — K-006, K-008, K-009, K-010, K-011 ✅: all five open plan items closed
- **What:** Worked the full active backlog in one pass.
  - **K-008 (verified, then adopted):** Live-tested whether the `Agent` tool's per-call `model` override reaches a call made *from inside* a subagent — spawned a `general-purpose` agent that itself called `Agent(model:"haiku", run_in_background:true, ...)`; grepped the resulting nested transcript (`agent-<id>.jsonl`) and found `"model":"claude-haiku-4-5-20251001"`, confirming the override is honored one level down. Added a sentence to step 3: pass `model: "haiku"` on cost-insensitive units (routine doc touch-ups, small-diff re-reviews, suite runs); anything with design/code-quality stakes stays on the inherited model.
  - **K-009 (audited, then dropped):** Grepped all 5 of teco's own session transcripts (`~/.claude/projects/<flattened-repo-path>-claude-teco/*.jsonl`) for direct `WebFetch`/`WebSearch` tool_use. Found exactly one hit: 2026-07-24, during the K-002 agent-teams evaluation, teco fetched `code.claude.com/docs/en/agent-teams` and `/agent-view` directly instead of delegating — research that its own routing table already assigns to `cobb` ("Agent/subagent/skill/prompt/hook engineering"). One mis-routed use in the whole history doesn't justify the grant; dropped `WebFetch`, `WebSearch` from `tools:`.
  - **K-006 (decided):** The independent-review guardrail's defaults line named `analyst` for "plans and code" without saying whether that covered `graph-dba` design notes or `cobb` agent/skill deliverables. Made both explicit in the same clause rather than adding a new row: `plans and code (including graph-dba design notes and cobb's agent/skill artifacts) → analyst`.
  - **K-010 (trimmed):** Description's trailing clause ("Does not design solutions and routes non-trivial implementation to a specialist; may fix a genuinely trivial single-file no-brainer directly instead of delegating it" — 170 chars) shortened to "Delegates non-trivial implementation; may fix a trivial single-file no-brainer itself." (88 chars) — the routing table's row 1 already carries the full tie-breaker prose, so the description only needs the routing signal.
  - **K-011 (pruned):** Removed the three single-use Bash allow-rules (exact escaped Cypher literals from the K-001 probe run) from `.claude/settings.local.json`, leaving only the `test_queries.sh` entry.
- **Why:** user asked to work the teco kaizen backlog. K-008 and K-009 were verification-gated/evidence-gated rather than pure opinion calls, so both were resolved empirically (live nested-call test; transcript grep) instead of by inference.
- **Plan items:** K-006 ✅, K-008 ✅, K-009 ✅, K-010 ✅, K-011 ✅ — all moved here; active table now empty.

## 2026-07-29 — Credit/interface analysis backlogged as K-008..K-011 (review only, no source change)
- **What:** A user-requested analysis of teco's interfaces and credit consumption produced four new plan items, filed (not implemented): **K-008** (high) — route cost-sensitive delegations to a cheaper model via the `Agent` tool's own per-call `model` param, distinct from the per-agent frontmatter pin the team already rejected; needs live verification it reaches nested calls before adopting. **K-009** (medium) — audit whether teco itself ever uses its `WebFetch`/`WebSearch` grants (vs. delegating research), drop if unused. **K-010** (low) — scheduled token-cost recompression: description regrew 568→694 chars (+22%) since 2026-07-25, body regrew 9,866→12,656 (+28%) since 2026-07-11, both from legitimate feature additions rather than waste. **K-011** (low, hygiene) — prune three single-use Bash allow-rules left in `.claude/settings.local.json` from the one-off K-001 probe run.
- **Why:** the same analysis session that produced K-007 (above) surfaced these as lower-priority or verification-gated items not to act on immediately; recorded per the agent-maintenance skill's "record new ideas even on a review-only pass" rule rather than left informal in chat.
- **Plan items:** K-008, K-009, K-010, K-011 opened (all 🔵 proposed).

## 2026-07-29 — K-007 ✅: `SendMessage` continuation replaces cold respawn in the defect→fix→re-run loop
- **What:** Three touch points, no catalog change needed (internal execution mechanism, not a routing/deliverable-contract change). (1) `tools:` gained **`SendMessage`** — it was absent, so K-007 as previously worded would have been unshippable even after a step-4 rewrite. (2) Step 3 gained one clause: note the name/id each `Agent` call returns for any unit carrying a review gate or likely to need a follow-up round, since that identifier is what a later `SendMessage` addresses. (3) Step 4's two re-brief paths — the review "needs changes"/qa-defects loop, and the K-004 deficient-result path (errored/out-of-turns/off-brief/empty) — now both `SendMessage` the original delegate by that identifier (resumes from its own transcript, no re-explaining context) instead of a fresh `Agent` call; cold respawn is reserved for when the identifier **no longer resolves** (no name/id was ever returned, or a newer agent has since taken the same name) — the actual boundary condition per `SendMessage`'s own tool description, not the "errored/out-of-turns" split first drafted (see below).
- **Why:** the session's own analysis (prompted by a user request to find credit/interface optimizations) identified this as the highest-value unshipped lever: the defect→fix→re-run loop was re-explaining full context to a cold `Agent` spawn every retry cycle. `SendMessage`'s live tool description (fetched via `ToolSearch` this session, not a cached doc page) resolved K-007's open verification question — it explicitly states a send "resumes it from its transcript" for a named agent, "even after an agent completes," matching the `Agent` tool's own description ("use SendMessage with the agent's ID or name ... resumes it with full context"). This is the current harness's own self-description, stronger evidence than the two doc pages (`agent-teams`, `agent-view`) the original K-007 note flagged as describing two different mechanisms — still worth confirming empirically on the first real re-brief cycle, but no longer blocking.
- **Self-caught fix during drafting:** the first draft of the step-4 rewrite said "fall back to a fresh `Agent` call... when the original agent errored out entirely or exhausted its turn budget" — directly contradicting the same sentence's "deficient" category, which lists "errored, ran out of turns" as `SendMessage`-retry triggers. Caught by a §7-style self-check before this entry was written; corrected to the identifier-resolution boundary instead (see above).
- **Verified no regression:** `claude/scripts/audit-team.sh` re-run clean on teco (no teco-related FAIL; the 4 pre-existing FAILs are root `AGENTS.md` missing `coder`/`devops`/`frontend-engineer`/`tdd-engineer` — unrelated drift, out of scope here, reported not chased).
- **Plan items:** K-007 ✅ done (moved to plan.md's done-notes block).

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Milestone-close freeze becomes a coordination duty; coordination docs open with the header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** Two body edits, no frontmatter and no hook change. (1) *How you work* step 2 gains one line — *"Open the document with the header block from root `AGENTS.md`."* — the canonical sentence, byte-identical across the six producing prompts. (2) *Documentation curation* gains a third bullet: at milestone close, list every document the close freezes and make flipping each one's header to `Status: archived` a **done-condition of the closing unit, routed to that document's owner** (root `AGENTS.md` carries the per-kind routing table); nothing moves; `teco` coordinates and performs only the flip the table assigns it — its own `docs/plans/<slug>-coordination.md`.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4, blocker **B5** and M2. Under D4 a frozen document no longer moves to `archive/` — it gets `Status: archived` in place — which turns "archiving" from a file operation nobody had to schedule into a **flip somebody must be told to perform**, at a moment (`milestone close`) only the coordinator sees. Without this bullet the lifecycle signal the whole convention rests on would simply never be set. Routing rather than performing is forced by the guard topology, not by ceremony: `teco`'s `PreToolUse` allowlist reaches `docs/plans/*` only, so a flip it performed on a review, requirements doc, test plan or test report would raise an interactive human approval prompt **per file** — `falkor-chat/docs/reviews/` alone holds four active documents. The routing table is pointed at, not copied, for the same reason the header block is (v1.4 M20): root `AGENTS.md` is already in every agent's context via the root `CLAUDE.md` `@AGENTS.md` import, so the hop costs nothing while a second copy would drift. `claude/README.md` row 7 re-checked — it already describes `teco` as documentation curator who makes doc updates part of every unit's done-condition, which is exactly what this bullet instantiates; no catalog edit needed.
- **Plan items:** none. (K-006/K-007 untouched.)

## 2026-07-25 — Trivial single-file no-brainer fixes: teco may make them directly instead of delegating
- **What:** Relaxed the "coordinates, never implements" invariant one notch: teco may now make a genuinely trivial, single-file, no-design-needed fix (a typo, an obvious one-liner, a config value, a rename) directly instead of spinning up a specialist for it. Four touch points, no hook-allowlist change: (1) frontmatter `description`'s closing line now reads "does not design solutions and routes non-trivial implementation to a specialist; may fix a genuinely trivial single-file no-brainer directly instead of delegating it" (was an unqualified "Does NOT design or write code itself"); (2) opening persona paragraph states the exception inline; (3) Routing table gained a leading row (trivial single-file no-brainer → teco directly, tie-breaker: multiple files/design judgment/security-data-model-test-critical → delegate instead); (4) Guardrails' coordination bullet and ceremony bullet updated to match. The `PreToolUse` hook (`guard-coordination-doc-writes.sh`) is **unchanged in behavior** — its allowed globs still only cover `docs/plans/` and the kaizen inbox, so a trivial fix still hits the "ask" escalation and needs a one-time human approval; only the escalation *message* was reworded (no longer "deny by default", now "approve if this is genuinely that kind of trivial fix"). This keeps a human check on every non-coordination-doc write teco makes, trivial or not — it just stops teco from having to pretend the option doesn't exist.
- **Why:** User request: too much delegation overhead going to `coder` for small no-brainer changes. Discussed the trade-off first (this reopens ground settled by the 2026-07-08 architect/teco K-003 hook-enforcement work) and the user chose the narrowest of three options offered — prompt-level permission for trivial edits only, hook left as the safety net — over widening the hook's allowlist or just trimming routing ceremony elsewhere.
- **Plan items:** none (out-of-band user request). Worth revisiting if the "ask" escalation for trivial fixes turns out to fire often enough to reintroduce the friction this was meant to remove — that would be the signal to reconsider widening the hook allowlist (the second, rejected option).

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 661 → 568 chars (-14%): tightened phrasing, dropped restated detail. `teco` has no boundary pairs in `claude/scripts/audit-team.sh`; full audit re-verified green regardless. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — K-002 ✅: agent-teams evaluation closed — reject team-lead reframe; SendMessage sub-case spun to K-007
- **What:** Read `code.claude.com/docs/en/agent-teams` and `/en/agent-view` (the concrete step K-002 asked for) and closed with disposition: **reject** reframing teco as an agent-teams lead. Agent teams are experimental (opt-in `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`), built for teammates that talk directly to each other on independent, discussion-benefiting work (parallel review lenses, competing-hypothesis debugging, cross-layer ownership) — the docs are explicit that "for sequential tasks... or work with many dependencies, a single session or subagents are more effective." Teco's actual loop (decompose → sequence on dependencies → delegate → independently-reviewed gate) is exactly that latter shape; teams would add token overhead for no matching benefit. The 2026-07-12 sub-case (defect→fix→re-run re-spawning cold agents) turned out not to be an agent-teams question at all — it's answered by `SendMessage` continuation of the original delegate (confirmed available for `Agent`-tool subagents per the harness's own tool description), independent of the experimental teams flag. Spun off as **K-007**.
- **Why:** User asked to follow through on K-002's own proposed next step (read the docs, assess fit) rather than leave the plan item open indefinitely.
- **Plan items:** K-002 ✅ done (moved here); opened K-007 (adopt SendMessage continuation in step 4's defect→fix→re-run loop).

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `teco`'s `guard-coordination-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed coordination-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-16 — Applied K-004 + K-005 to teco.md (from the §7 lint)
- **What:** Two surgical prompt additions, user-approved from the same-day §7 lint. **K-004** — Step 4 ("Integrate & verify") gained a *deficient-result* path: when a delegate errors, runs out of turns, or returns something off-brief/empty (explicitly distinct from a *blocker* that changes direction and a review *verdict*), re-brief the same owner once with the gap made explicit, and pause to the user if it recurs or the unit is mis-scoped — "rather than re-spawning blindly". **K-005** — the Documentation-curation "Scan at decomposition" list now names `docs/HISTORY.md` (which takes an entry for every delivered change) and `docs/BACKLOG.md` "where the module uses the convention", closing the gap between teco's curator role and the module-documentation convention in root `AGENTS.md`. No frontmatter/description/catalog change — role unchanged, so the catalog entries still describe teco correctly. K-006 left proposed (not approved).
- **Why:** User approved acting on the two higher-value lint findings; both were surgical additions at teco's existing altitude, not a rewrite.
- **Plan items:** K-004 ✅, K-005 ✅ (moved to the done-notes block in plan.md); K-006 stays open.

## 2026-07-16 — §7 prompt-quality lint (review-only, no prompt change)
- **What:** cobb ran the new `agent-maintenance` §7 single-artifact prompt-lint against `teco.md` across all six dimensions, resolving teco's full load-set (root + `claude/` `CLAUDE.md`→`@AGENTS.md` chain, the injected specialist `description`s, the coordination-doc write guard) for the composition check. **Persona:** clean. **Contradiction / ambiguity / cognitive load:** clean bar minor nits (parked). **Coverage + composition:** three findings filed — K-004 (no deficient/failed-delegate-result path), K-005 (doc-curation scope omits the module `docs/HISTORY.md`/`BACKLOG.md` conventions from `AGENTS.md` — highest-value, surfaced only by the composition load-set resolution), K-006 (no independent reviewer assigned for agent-engineering deliverables). No blocker; no source change.
- **Why:** Smoke test of the §7 procedure cobb authored the same day; teco is a mature, certified prompt so a clean-ish result was expected and validated that §7 surfaces real gaps without manufacturing findings.
- **Plan items:** opened K-004, K-005, K-006; minors parked.

## 2026-07-12 — K-003 ✅: review-gate invariant proven on the first fully-gated run — kept, no prompt change
- **What:** Closed K-003 with disposition **(a) keep the invariant** — "work ships
  independently reviewed; when you trim ceremony, the review gate is the last thing to go."
  falkor-chat **K-022 Landing 1** (U1–U10, committed `3921f87`) ran as the team's first fully-gated
  coordinated delegation with the analyst post-implementation review as a non-negotiable
  done-condition, and the cost datapoint the plan asked for is now recorded in
  `falkor-chat/docs/plans/m3-executor-coordination.md` ("Cost datapoint" table). **No prompt
  change** — the datapoint vindicates the existing guardrail rather than forcing the (b) rewrite
  to risk-signal-gated review.
- **Evidence / reasoning from the datapoint:**
  1. **The gate is cheap.** Analyst review = ~149k tokens / 25 tool uses / ~7 min — ~12% of the
     ~1.20M-token, ~4h gated run, a thin slice on top of the six implementation delegations.
  2. **The gate paid.** On a diff the implementers considered done it returned
     approve-with-suggestions with **1 major (M-1, the drive try/except) + 3 minor + 3 nit** —
     exactly the class of defect the K-020/21 "review left to the user" skip would have shipped
     unseen.
  3. **The headline ~12× vs. the K-001 baseline is a units artifact, not the gate's cost** — 10
     units + independent gate vs. an ungated 2-unit slice (~100k / 23 / ~45 min). Per-unit the run
     is comparable; the review is the cheap part.
  4. Therefore the concern that opened K-003 — "an invariant that never fires is hopeful prose" —
     is resolved: it fired, cheaply, and caught real signal. Keeping review-by-default is the
     right risk posture; the low marginal cost means the default stands even at n=1.
- **Honest caveat (recorded, not blocking):** this is **one** gated run. It proves the gate can pay
  its way and is affordable, not that every gate will catch a major. The cost is low enough that
  "keep the default, skip only with stated justification for genuinely trivial units" remains
  correct pending more datapoints — re-examine if a run of gates comes back all-nits at real cost.
- **Why:** User asked to close the K-003 thread now that K-022 is committed. The experiment ran end
  to end (gate enforced + datapoint captured); the disposition is the last step the plan item
  named ((a) keep / (b) rewrite).
- **Plan items:** **K-003 ✅ done** (moved here). No change to `teco.md`, `README.md`, or the
  context catalogs — behavior/routing unchanged; this is a decision to *keep* the current prompt.
  Counterparts still open on their own agents: `analyst` K-001 (its code-review shakedown — the
  same run validated it; closeable on analyst's side) and `qa-engineer` K-003 (defect→fix→re-run
  loop — **unexercised**, the review returned 0 blockers so no needs-changes loop fired).

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist + integration check
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the coordination-doc write guard's allowlist gained exactly teco's own inbox path. Step 4 (Integrate & verify) additionally gained the learnings-ride-the-handoff check: when a specialist's result reports a durable environment discovery, confirm it was filed in that agent's inbox (a one-line check, not a gate).
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture during runs, curated promotion by cobb. Teco is the collection point on orchestrated work — the integration check catches learnings a delegate forgot to file. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — graph-dba added to the handoff contracts (certification fix)
- **What:** The "Handoff contracts" list gained the `graph-dba` entry: implementer-bound design work (data model, schema/DDL, ingestion/migration) arrives as a design note at `<component>/docs/plans/<slug>-graph.md`; quick consults and tuning diagnoses stay inline. Matches the same-day addition of the convention to graph-dba's own prompt (its kaizen K-004).
- **Why:** Team-coherence certification (2026-07-11): graph-dba was the only design-producing specialist whose deliverable teco had to paraphrase into the next brief — the exact lossy handoff the "by path, never paraphrased" rule exists to prevent.
- **Plan items:** none (graph-dba K-004 on the producer side).

## 2026-07-11 — Prompt body compressed (token-cost pass, part 2)
- **What:** Body compressed in place, 15,023 → 9,866 chars (−34%): the routing table's per-agent capability prose was cut down to pure routing judgment (tie-breakers, boundaries, pipeline defaults), explicitly leaning on the injected frontmatter descriptions teco already receives at spawn through its `Agent` tool; "How you work", documentation curation, pause rules, and guardrails were tightened without dropping any rule or contract. All 11 specialist names remain in the file (audit check 4 green, full audit pass); frontmatter (description, tools, hook) unchanged. No on-demand reference file — teco uses its whole body every run, so offloading would just add a mandatory Read.
- **Why:** teco.md was the team's second-heaviest prompt and loads on every teco spawn; the injected description catalog already carries each specialist's capabilities, so restating them in the body was pure duplication (~1,450 tokens saved per spawn).
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1286 to 659 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-coordination-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/plans/*|*/docs/plans/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Independent review made the default mindset
- **What:** independent review is now a standing principle, not an optional gate. Four touch points: (1) a new guardrail — "**Work ships independently reviewed**": no deliverable is accepted on its producer's word alone, teco's own integration check is fit/completeness (not a substitute for review), and every significant deliverable defaults to a reviewer who didn't produce it (plans/code → `analyst`, ML methodology → `data-scientist`, behavior/acceptance → `qa-engineer`); skipping a gate is the justified exception for genuinely trivial, low-risk units, stated explicitly in the report. (2) Step 2 now assigns each unit its **review gate** alongside owner/inputs/done-condition. (3) The typical-feature paragraph flips `analyst` from "slotted in where the stakes warrant it" to the **default review gate**, and the match-ceremony-to-task rule gains "when you trim ceremony, the review gate is the last thing to go, not the first." (4) The frontmatter `description` advertises the default.
- **Why:** User request: teco should "always have in his mindset the need for the work to be independently reviewed." The previous phrasing made review an exception teco had to argue itself into; the risk posture the user wants is the inverse — review by default, skip only with justification.
- **Plan items:** none.

## 2026-07-10 — Standing documentation-curator duty
- **What:** teco is now the team's **documentation curator**, keeping project docs always in sync with delivered work. Four touch points: (1) a new "Documentation curation" section with the standing rules — documentation-impact scan at decomposition (READMEs, `AGENTS.md`/`CLAUDE.md`, design/reference docs, catalogs, recorded in the coordination doc), affected docs named in the unit's brief with same-change updates part of the deliverable (the unit's owner writes them; agent/skill docs → `cobb`), verification by actually reading the flagged docs at integration (stale docs = incomplete unit → re-brief), and pre-existing drift reported as a follow-up rather than silently chased; (2) step 2 runs the scan as part of the breakdown; (3) step 4 makes documentation part of done; (4) the frontmatter `description` advertises the curator duty. teco still never writes these docs itself — `Write`/`Edit` stays hook-scoped to the coordination doc.
- **Why:** User request: teco should "keep track of the docs updates, being the curator for an always updated documentation." Curation (track → brief → verify) fits teco's coordinator identity and existing hook scope; the writing routes to the owner of each change.
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/teco/hooks/guard-coordination-doc-writes.sh`) to `$HOME/.claude/agents/teco/hooks/guard-coordination-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/teco` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — Roster: added data-scientist (AI/ML/DS advisory specialist)
- **What:** the routing table gained a `data-scientist` row (AI/ML/data-science **method** questions — model/embedding selection, retrieval strategy, RAG/GraphRAG evaluation design, quality metrics, experiment/A-B design, statistical validity — plus methodology reviews and model/retrieval-underperformance diagnosis; boundary notes: advisory-only — implementation of its recommendations routes to the implementers with its note as the brief, general correctness review stays with `analyst`, in-graph vector mechanics/Cypher with `graph-dba`); the handoff-contracts list gained its two deliverables (method note `docs/plans/<slug>-ml.md`, methodology review `docs/reviews/<slug>-ml.md`, hook-enforced advisory-only writes); the frontmatter parenthetical now includes it.
- **Why:** an AI/ML/data-science specialist joined the team; the orchestrator's roster must enumerate every delegate with its current contract (the drift class the 2026-07-09 interface review exists to catch).
- **Plan items:** none.

## 2026-07-09 — Roster: added frontend-engineer (UI-depth implementer)
- **What:** the routing table gained a `frontend-engineer` row (UI-heavy front-end work — components, styling, accessibility, client-side state, front-end performance, Streamlit screens — with the boundary note that back-end/non-UI code stays with `coder`/`tdd-engineer` and incidental template touches don't need the specialist); the frontmatter parenthetical and the typical-feature pipeline now include it among the implementers.
- **Why:** a front-end specialist joined the team; the orchestrator's roster must enumerate every delegate (the drift class the 2026-07-09 interface review existed to catch).
- **Plan items:** none.

## 2026-07-09 — Roster restructured into an explicit routing table + handoff contracts
- **What:** "The team you coordinate" reformatted from prose bullets into two artifacts: a **routing table** (task shape → owner → tie-breaker/boundary, one row per routable signal, including the "requirements vague → pause, recommend tico" row and the two built-ins) and a **handoff contracts** list (per-agent document paths and by-path handoff rules for tico/architect/analyst/qa-engineer). Content is unchanged — same roster, same routing rules, same contracts — only made scannable and self-checkable; the typical-feature pipeline paragraph kept as-is. Catalogs (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`) describe routing behavior, not prompt format — verified accurate, no edits needed.
- **Why:** User asked how teco decides routing and for a "clear configuration". Routing is LLM judgment over prompt text; the clearest configuration of that judgment is an explicit decision table teco self-checks before each delegation (the parking-lot "routing cheat-sheet" idea, now fully addressed — including the coder-vs-tdd tie-breakers on both implementer rows).
- **Plan items:** parking-lot "routing cheat-sheet / decision tree" ✅ resolved.

## 2026-07-09 — Roster: analyst gained RCA routing
- **What:** analyst's roster entry (and the frontmatter parenthetical) now also routes **cause-unknown defects/failures** to it for a root cause analysis at `<component>/docs/reviews/<slug>-rca.md`, whose suggested fix then briefs the implementer (typically `tdd-engineer`, reproduction test first) by path.
- **Why:** analyst extended with an RCA mode the same day (user request); the orchestrator's roster must describe each specialist's current contract.
- **Plan items:** none.

## 2026-07-09 — Roster: added analyst (plan & code review gate)
- **What:** Added `analyst` to the frontmatter specialist list and the roster, slotted it into the typical-feature pipeline as an optional review gate (after architect on high-blast-radius plans and/or after the implementer before QA), and extended step 4's defect loop to cover a "needs changes" review verdict (re-brief the owner with the review path, then re-review). The roster entry encodes the handoff contract: review doc at `<component>/docs/reviews/<slug>.md`, handed off by path, review-only on code (hook-enforced).
- **Why:** New team member created 2026-07-09 — the orchestrator's roster must be updated in the same change as the agent (agent-maintenance §2 step 3; the qa-engineer/devops roster-drift lesson).
- **Plan items:** none.

## 2026-07-09 — tico reframed: first-order agent, not a delegation target
- **What:** Removed tico from the frontmatter routing list; its roster entry now marks it **not a delegation target** — tico runs as the user's own main-session agent (`claude --agent tico`) and teco **consumes** its requirements doc (`<component>/docs/requirements/<slug>.md`) by path, treating vague/uncaptured requirements as a pause point that recommends a tico interview. Pipeline reads **tico (user-run) → architect → implementers → qa**.
- **Why:** User ruling, same day as the roster addition below: tico is a first-order conversational agent, not a subagent — the interview must be a live conversation, which delegation can't provide.
- **Plan items:** none.

## 2026-07-09 — Roster: added tico (product-owner interviewer, upstream of architect)
- **What:** Added `tico` to the frontmatter specialist list and the roster, and prefixed the typical-feature pipeline with it (**tico → architect → implementers → qa**, skipped when requirements are already clear). The roster entry encodes the round-trip contract: tico's question batches are a pause point — relay to the user verbatim, re-delegate with the answers + the doc path (`<component>/docs/requirements/<slug>.md`); the finished doc hands to the architect by path.
- **Why:** New team member created 2026-07-09 — the orchestrator's roster must be updated in the same change as the agent (agent-maintenance §2 step 3; the qa-engineer/devops roster-drift lesson).
- **Plan items:** none.

## 2026-07-09 — Roster: implementer routing de-personalized (efficiency rule)
- **What:** Replaced the coder/tdd-engineer routing guidance in the roster. Dropped the *"(This user prefers TDD — lean toward `tdd-engineer` for implementation unless told otherwise)"* note; both bullets now carry a task-shape rule — route by **efficiency, not ceremony**: detailed architect plan ready to execute → `coder`; bug fix (repro test first), safety-net refactor, test-focused work, or clear-contract feature → `tdd-engineer`.
- **Why:** User ruling: personal-preference notes don't belong in agent prompts — their standing preferences are quality and efficiency, expressed as objective routing rules. Part of the same-day coder/tdd-engineer boundary fix (coder K-001 ✅).
- **Plan items:** none (out-of-band).

## 2026-07-09 — K-001 ✅: live nested-delegation validation run (falkor-chat M3 slice 1)
- **What:** Ran teco end-to-end on a real assignment — kick off falkor-chat **M3 — Workflow
  engine**, decompose the milestone, deliver slice 1 (K-020 def model + K-021 snapshot
  materialization). Launch brief + observation checklist: `k001-run-brief.md` (executed verbatim).
  Scored against the checklist from the run transcript + independent re-verification:
  1. **Depth — PASS.** teco (opus) spawned architect → graph-dba → tdd-engineer (one `Agent` call
     each, sequenced on their upstream artifacts); all three nested runs completed with no
     depth-related degradation observed.
  2. **Path-based handoff — PASS.** All three delegate briefs carried the plan-doc path
     (`docs/plans/m3-workflow-engine.md`); the plan was never paraphrased wholesale into a brief
     (briefs ~6.7–7.7 KB, self-contained context + path).
  3. **Brief fidelity — PASS.** Every brief included the "this brief is your entire context"
     framing and the blockers-back-as-deliverable reminder. No observed information loss; the
     one plan gap (no `start_key` param on `publish_workflow_def`) was an *architect plan*
     omission, resolved sensibly by the implementer and surfaced by teco as a follow-up —
     exactly the intended behavior.
  4. **Hook enforcement — PASS (unexercised).** teco's own Write/Edit calls (1 Write + 5 Edits)
     all targeted its coordination doc (`m3-workflow-engine-coordination.md`); the
     guard-coordination-doc-writes hook never needed to fire.
  5. **Decision points — PASS.** The §13 guard-expression-language question was correctly
     assessed as *not forced* by slice 1 (opaque strings, evaluated at run time) and deferred to
     K-022's architect pass with an explicit return-to-user; `ws:acme`/`reference` kept
     additive-only; zero scope creep (executor/linkage/proof flows untouched).
  6. **Integration & honesty — PASS.** teco re-ran both suites itself and reported truthfully;
     independently re-verified afterwards: `test_queries.sh` **193/193**, pytest **196** — both
     matching teco's claims. Nothing committed (correct; review left to the user).
- **Why:** K-001 was the open proof that an orchestrator subagent works in practice — depth,
  context-passing fidelity, and result quality were validated on a real deliverable, not a toy.
- **Prompt changes:** **none needed** — the run surfaced no prompt weakness. Deliverables landed
  in falkor-chat (see `falkor-chat/docs/HISTORY.md` 2026-07-09). Run cost datapoint: ~100k
  subagent tokens / 23 tool uses / ~45 min for a 2-item slice with 3 nested specialists.
- **Plan items:** K-001 ✅ done (moved here). Same-run evidence closed **architect K-002**
  (plan executed cold by an isolated implementer) and updated **coder K-002** (contract proven
  via tdd-engineer; coder-specific run still open). K-002 (agent teams) remains the sole active item.

## 2026-07-09 — Interface review: roster completed (qa-engineer, devops) + guard hook + brief/verify upgrades
- **What:** Thorough review of teco and its interfaces produced five prompt changes and one new artifact:
  1. **Roster completed** — `qa-engineer` (with its `docs/test-plans/` / `docs/test-reports/` artifact conventions) and `devops` (environment blockers routed there instead of bounced to the user) added to the roster, the frontmatter `description`, and the typical-feature pipeline (now `architect → implementer → qa-engineer`, `devops` unblocking env issues). Both agents postdate teco's creation (qa-engineer 2026-07-01, devops ~2026-07) and had never been folded in.
  2. **Brief template generalized** (step 3) — path-based handoff is now the rule for *every* document deliverable (architect plan named as the canonical case, qa plan/report as the other standing instance); briefs must remind delegates they can't ask mid-run (blockers/questions come back as the deliverable).
  3. **Parallel-delegation mechanics** (step 3) — independent delegations go out as parallel `Agent` calls in one turn; dependent ones sequence on their upstream artifact.
  4. **Verify step clarified** (step 4) — running the project's suites/scripts is in-bounds read-only verification; acceptance-level verification routes to `qa-engineer`, with the defect→fix→re-run loop (re-brief implementer with the report path, re-run failed items — qa-engineer kaizen K-003's teco side).
  5. **Guard hook (harness enforcement parity with architect)** — new `teco/hooks/guard-coordination-doc-writes.sh` wired in frontmatter (matcher `Write|Edit`): any target outside `docs/plans/` (or `/tmp`) escalates to the human (`permissionDecision: "ask"`); same fail-open jq→python3 contract as the architect/devops hooks. Unit-driven: allowed path passes silently, violating path emits the ask JSON.
  - **Counterpart fixes in the same change:** `tdd-engineer` gained the plan-doc-path handoff line (mirroring coder) + subagent-awareness ("return the question/blocker as your result"); `qa-engineer` gained the same subagent-awareness in its scope step and environment guardrail. Catalogs synced: `claude/AGENTS.md`, `claude/README.md` (teco row + hook-gotcha list), root `AGENTS.md` teco cell.
- **Why:** Review found teco's core design sound but stale at the edges: two specialists were invisible to it (it literally could not route QA or infra work), its doc-scoping guardrail was prompt-only while the identical architect contract is hook-enforced, and the delegation protocol's key rules (path handoff, no-mid-run-questions) existed only as special cases instead of general brief requirements.
- **Plan items:** parking-lot "routing cheat-sheet" idea partially addressed (complete roster + routing signals per entry); K-001 (live nested-delegation run) and K-002 (agent teams) remain open.

## 2026-07-08 — Path-based architect handoff + coordination-doc convention (K-003 ✅)
- **What:** Two prompt changes, synced with the architect's same-day overhaul: (1) step 3 no longer says to pass the architect's plan **verbatim** — the architect now writes its plan to `<component>/docs/plans/<slug>.md` and teco hands the implementer the **path** with an instruction to read the file itself, never a paraphrase; the roster's architect line states the convention. (2) K-003 resolved: teco's coordination/work-breakdown doc gets a fixed convention too — `<component>/docs/plans/<slug>-coordination.md`, co-located with the architect's plan (baked into step 2). Catalog entries updated (`claude/AGENTS.md`).
- **Why:** Design review of the architect found the verbatim copy-through was the weakest link in the teco pipeline: a long plan returned as a subagent message and re-pasted into a brief risks truncation/paraphrase, and leaves no durable artifact. A file handed off by path is lossless, cheap to brief, and reviewable after the fact. The coordination-doc convention rode along since it was the same decision (architect K-001 fixed the location).
- **Plan items:** K-003 ✅ done (moved here); K-001 note updated — the live nested-delegation validation is still pending but no longer needs to stress brief fidelity for the plan itself.

## 2026-07-05 — Added `Edit` (scoped to the coordination doc)
- **What:** Added `Edit` to teco's frontmatter tools (`Read, Grep, Glob, Bash, Agent, Write, Edit, WebFetch, WebSearch`). Updated the guardrail to `Write`/`Edit` = **coordination/work-breakdown document only** (Write to create, Edit to revise in place as steps complete) — still **never** source/tests/config. Also tightened "How you work" step 2 to mention editing the doc in place. Mirrored the wording in `claude/AGENTS.md`.
- **Why:** User asked to give teco the `Edit` tool. With `Write` only, teco could create a coordination doc but had to overwrite it wholesale to update it; `Edit` lets it surgically revise the doc across a long-running orchestration (mark steps done, append findings). Scoped deliberately to the coordination doc — parallels `architect`, which carries `Write`+`Edit` guardrailed to its plan doc — so teco's "coordinate, don't implement" identity is preserved.
- **Plan items:** none (out-of-band user request); relevant to K-003 (coordination-doc convention).

## 2026-06-20 — Created
- **What:** Created the `teco` subagent (`teco/teco.md`, `model: opus`). Technical coordinator / tech lead: decomposes a multi-step goal into a sequenced work breakdown and **delegates each unit to the right specialist** (architect, coder, tdd-engineer, graph-dba, cobb; Explore/Plan built-ins) via the `Agent` tool, then integrates and verifies. **Hybrid mode:** delegates execution itself by default but pauses and returns to the user at genuine decision points / blockers / ambiguity. Tools: `Read, Grep, Glob, Bash, Agent, Write, WebFetch, WebSearch` — **no `Edit`/`NotebookEdit`** (it coordinates, doesn't implement); `Write` is for the coordination doc only; `Bash` read-only by guardrail.
- **Why:** User asked for a third agent on top of the architect→coder pair — "teco the technical coordinator" — to orchestrate the specialist roster.
- **Plan items:** seeded K-001..K-003.

## Decisions & verification recorded at creation
- **Subagents CAN delegate to subagents — verified 2026-06-20** against `code.claude.com/docs/en/sub-agents`. The doc enumerates the tools withheld from subagents (`AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode`, `ScheduleWakeup`, `WaitForMcpServers`); the `Agent`/Task tool is **not** withheld, so an orchestrator subagent is viable. (Older lore said subagents couldn't spawn subagents — that constraint no longer holds per the live doc. Claude Code now also has first-class *agent teams* and *background agents*.)
- **Key limitation baked into the prompt:** `AskUserQuestion` is unavailable to subagents, so teco **cannot ask interactively** — the hybrid design has it *return* to the user with the decision instead of guessing. teco also doesn't see the parent conversation, and delegated agents don't see teco's or each other's context → the prompt mandates **self-contained briefs** (pass the architect's plan verbatim to the implementer, etc.).
- **No `name`-conflict / collection consistency:** dropped any "senior" framing to match the 2026-06-20 harmonized collection. Defaults implementation routing toward `tdd-engineer` given the user's documented TDD preference.
