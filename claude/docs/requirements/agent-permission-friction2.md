# Agent permission-escalation friction, phase 2 (`coder`) — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-23
> **Extends:** `claude/docs/requirements/agent-permission-friction.md` (archived; phase-2 forecast)

## Intent
Reduce `coder`'s permission-escalation prompts on legitimate, in-remit work, same goal as the
phase-1 round for `cobb`/`qa-engineer`/`tdd-engineer`/the five doc-scoped agents — without
weakening escalation for a genuine accidental drift. Phase 1 explicitly scoped `coder` out because
"zero live instances were ever collected this round despite the evidence-first approach applying
throughout." This document opens now that a live instance exists.

## Problem & current state
`coder` carries `permissionMode: acceptEdits` and **no custom `PreToolUse` write guard at all** —
confirmed today (`grep -n "hooks:\|permissionMode" claude/coder/coder.md` → only the
`permissionMode` line, no `hooks:` block). Per phase 1's root-cause finding
(`claude/docs/plans/agent-permission-friction.md` §1), `acceptEdits` alone does **not** reliably
suppress the plain default confirm-before-Edit/Write prompt — that's exactly why `tdd-engineer`
(same shape: broad implementer, no guard) still needed a hook-based fix in phase 1 despite already
having `acceptEdits` since July. `coder` is in that identical shape today, unfixed.

The cost compounds **per edit, not per task** — same finding as phase 1's `tdd-engineer` instances
7/8 ("a single legitimate TDD cycle can trigger several separate confirmations against the same
file"). Instances 1-3 here are all one continuous `coder` pass over the single active K-050 plan,
and each individual `Write`/`Edit` — source file, its paired test file, another source file —
triggered its own separate prompt. Stakeholder, 2026-08-23: "yeah it is annoying, all the time
recurring requests" — confirms the friction isn't an occasional nuisance, it's continuous
interruption across one otherwise-uninterrupted implementation task.

## User stories
- As the stakeholder, I don't want to be interrupted approving `coder`'s `Write`/`Edit` when it's
  squarely `coder` doing its own in-remit implementation work on an active, gated plan — same story
  as phase 1's, just for `coder`.
- As the stakeholder, I still want to be asked when `coder` does something genuinely outside its
  remit (e.g. touching another specialist's documented deliverable path) — the interruption should
  mean something when it happens, exactly as phase 1 preserved for the five agents it fixed.

## Functional requirements
- **FR-1 (governing):** A `coder` `Write`/`Edit` that stays within its own broad implementer remit —
  any source or test file needed for its current, approved task — must not require a manual
  per-action confirmation. A `coder` `Write`/`Edit` genuinely outside that (e.g. a path that's
  another specialist's documented deliverable kind — a plan, review, requirements, manual,
  test-plan/test-report doc, an agent definition or kaizen file, a team catalog or skill package, an
  MCP-standards doc, or the project backlog) must still escalate for approval. Evidenced by
  instances 1-4 below — all one continuous `coder` pass on the same active, gated plan, all
  correctly recognized as in-remit, none challenged.

## Instances observed (live)
1. **2026-08-23 — `coder`, `Edit` on `falkor-chat/server/falkorchat/services.py`.** Coincides with
   `coder` implementing the gated `document-ingestion-coordination` plan (K-050,
   `falkor-chat/docs/plans/document-ingestion-coordination.md`, Status: active, owner `teco`) — repo
   state shows `repository.py`/`services.py` modified and a new `chunking.py`, consistent with
   in-progress implementation of that plan. **Confirmed legitimate** by the stakeholder — squarely
   `coder`'s own in-remit implementation work. → first evidence instance for this round. (Edited
   again shortly after — same `services.py`, shown as `server/falkorchat/services.py` relative to a
   different cwd this time, confirmed the only such file in the repo — same compounding-cost pattern
   as phase 1's `tdd-engineer` instances 7/8: one file, several separate confirmations across one
   continuous implementation pass.)

2. **2026-08-23 — `coder`, `Create` (Write) on `falkor-chat/server/tests/test_chunking.py`.** Pairs
   with the new `falkor-chat/server/falkorchat/chunking.py` from the same K-050 implementation work
   — `coder`'s remit explicitly includes tests for the code it writes. No challenge: clearly in-remit,
   same task as instance 1. → second evidence instance; first Create (not Edit) and first test-file
   instance for this round.

3. **2026-08-23 — `coder`, `Edit` on `falkor-chat/server/tests/test_repository.py`.** Pairs with the
   already-modified `falkor-chat/server/falkorchat/repository.py`, same K-050 task as instances 1-2.
   No challenge: clearly in-remit. → third evidence instance, second `Edit` (not Create) on a test
   file. (`repository.py` itself edited again shortly after — same compounding-cost pattern as
   instance 1's `services.py` repeat.)

4. **2026-08-23 — `coder`, `Edit` on `falkor-chat/server/falkorchat/embedding.py`.** Cross-checked
   against `document-ingestion-coordination.md`'s ledger: row U12 shows `coder` currently `in-flight`
   on "Stage 2: chunk embeddings + standalone search (FR-3)" — exactly where this edit lands. No
   challenge: clearly in-remit. → fourth evidence instance; first genuinely new source file this
   round (not a repeat, not paired with an already-seen file).

## Out of scope
- **The exact deny-list/guard shape** (which paths to enumerate, whether it reuses
  `guard-broad-write.sh`) — a HOW decision for the architect, not this document. Phase 1's
  `tdd-engineer` fix (§6) is the obvious precedent given the matching evidence shape, but this
  document doesn't dictate reuse.
- **The auto-mode-classifier-vs-`PreToolUse`-hook interaction for subagent-delegated writes**
  (former open question 3) — root-caused and documented (`skills/agent-standards/claude-code.md`,
  2026-08-24), but not fixable within a repo-local guard script. Per `cobb`'s explicit
  recommendation, this document's eventual fix should still ship — it puts `coder` on equal footing
  with the five already-shipped agents, not behind them. The suggested live isolation test is a
  candidate follow-up, not a prerequisite.
- **Bash-triggered confirmations** — no live evidence this round. If a future instance surfaces one,
  it's a new document, not a revision of this one (per phase 1's precedent scoping `coder`'s Bash
  question out entirely).
- **The destructive-ops guards, git-commit authority scoping** — unaffected, same as phase 1.

## Acceptance criteria
- **AC-1:** Given `coder` performs a `Write`/`Edit` within its own broad implementer remit (any
  source/test file for its current approved task) from a top-level interactive `coder` session, when
  the tool call runs, then no manual confirmation prompt appears.
- **AC-2 (safety net preserved):** Given `coder` performs a `Write`/`Edit` genuinely outside its
  remit (another specialist's documented deliverable path), when the tool call runs, then a manual
  "ask" confirmation still appears — unchanged from today.
- **AC-3 (known limitation, not a failing condition):** For a `coder` `Write`/`Edit` that is
  Task/`Agent`-delegated (e.g. via a `teco` session) rather than run from a top-level interactive
  session, the guard's hook decision will be correct (explicit `"allow"` on an in-remit match), but
  live suppression of the prompt is not guaranteed today, per the documented auto-mode-classifier gap
  (`skills/agent-standards/claude-code.md`, 2026-08-24). This is a pre-existing condition shared with
  all five phase-1-fixed agents, not a regression introduced by this feature — verifying or closing
  it depends on the live isolation test `cobb` recommended, outside this document's scope.
- **AC-4 (regression check):** `coder`'s existing, unfixed friction on paths outside its remit is
  unaffected — this feature narrows *when* confirmation fires, it does not remove the safety net.

## Open questions
None currently open — see Decision log for how each was resolved or reclassified.

- *Fix shape (deny-list like `tdd-engineer`'s vs. something narrower)* — reclassified: this is a
  HOW question, not a WHAT/WHY one. It doesn't belong as an open question in a requirements
  document at all; it's the architect's call once this document is ready, informed by the evidence
  below (all 4 instances match `tdd-engineer`'s "broad implementer, no single folder" shape).
- *Bash-confirmation instances* — none observed this round. A fact, not a blocker; noted in Out of
  scope in case future evidence surfaces it.
- *The cross-agent regression risk (`analyst`, `tdd-engineer` both still prompting despite a
  correct hook `"allow"`)* — **investigated and resolved as "root-caused, not a blocker."** Dispatched
  to `cobb` (2026-08-24, commit `6193083`, `claude/cobb/kaizen/history.md`): a real, live-reproduced,
  undocumented interaction between Claude Code's auto-mode permission classifier and `PreToolUse`
  hooks, specific to Task/`Agent`-tool-delegated writes (exactly the shape both regression instances
  had — relayed via a concurrent `teco` session). Not a guard-script bug, stale deployment, or
  version regression — all ruled out with direct evidence (symlinks live, git history clean since
  2026-08-21, reproduced on two CLI versions, no nested-repo trust gap). Phase 1's own root-cause
  finding (§1.3, "hook allow suppresses the prompt... every time") over-claimed its source: the docs
  only guarantee that relative to settings.json ask/deny rules, and are silent on the auto-mode
  classifier's interaction with a hook's `"allow"` on a **subagent-delegated** write specifically.
  Promoted into `skills/agent-standards/claude-code.md` (2026-08-24) as a durable fact for future
  write-guard design. **Not fixable from `cobb`'s remit** (no guard-script/settings.json lever; the
  only lever — changing the account/project default away from `auto` mode — is a broad,
  costly-to-reverse call flagged rather than made unilaterally) and, critically, **`cobb`'s explicit
  recommendation is that phase-2 design should proceed anyway**: this gap already affects the five
  already-shipped guards equally, so waiting for it doesn't put `coder` in any worse a position than
  today's shipped agents are already in. A live isolation test (fresh, non-concurrent, top-level
  `--agent analyst` session, mode-bar watched live) was suggested as a follow-up, not a gate — only
  runnable interactively by the stakeholder.

## Decision log
- 2026-08-23 — Stakeholder: "we recently implemented permission friction and i did not have
  evidence on coder yet" → opens phase 2 of `agent-permission-friction`, per the explicit forecast
  in the phase-1 doc's closing decision (2026-08-21). New document on the ordinal slug
  (`agent-permission-friction2.md`) since phase 1 is `archived` (approved/executed).
- 2026-08-23 — First live instance relayed mid-turn (`coder` → `services.py`) while this document
  was being opened; logged as instance 1, not yet confirmed legitimate.
- 2026-08-23 — Stakeholder confirmed instance 1 legitimate ("Yes, legitimate") → counted as first
  evidence instance. Stakeholder also chose to re-adopt phase 1's "I'm only sharing approved cases"
  shortcut for the rest of this round → every further relayed instance counts as evidence without a
  per-instance confirmation, until/unless the stakeholder flags one as unsure (phase 1 precedent:
  U1, still relayed as an open question when genuinely unsure).
- 2026-08-23 — Stakeholder: "please be critic about the cases in the first phase you correctly
  recognized some cases I proposed by mistake so please feel free to challenge me if you think it
  is not applicable" → **"pre-confirmed" mode is not a rubber stamp.** Same standard as phase 1's
  C1/C2 counter-examples (`data-scientist` → `tests/eval/probe_ministral_judge.py`,
  `cobb` → `docs/BACKLOG.md`, both initially proposed as evidence and reclassified on reflection)
  and U1 (flagged as ambiguous unprompted, even mid-"only sharing approved cases"): `tico` will
  actively flag a relayed `coder` instance that looks out-of-remit (e.g. a path with no visible
  connection to an approved plan/task) rather than silently counting it, even though individual
  reconfirmation is otherwise skipped.
- 2026-08-23 — Instance 2 (`coder` → `falkor-chat/server/tests/test_chunking.py`, Create) relayed;
  assessed in-remit (pairs with instance 1's `chunking.py`), no challenge raised, counted as evidence.
- 2026-08-23 — Instance 3 (`coder` → `falkor-chat/server/tests/test_repository.py`, Edit) relayed;
  assessed in-remit (pairs with instance 1's `repository.py`), no challenge raised, counted as
  evidence.
- 2026-08-23 — Stakeholder: "yeah it is annoying, all the time recurring requests" → confirms the
  per-edit compounding cost (same shape as phase 1's `tdd-engineer` finding) rather than an
  occasional one-off; folded into Problem & current state.
- 2026-08-23 — `analyst` → `docs/reviews/document-ingestion-impl.md` (Create) relayed. Path matches
  `analyst`'s already-shipped phase-1 allowlist; static check confirms `guard-doc-writes.sh` emits
  the explicit `"allow"` on match today. Asked whether it actually prompted (not assumed either
  way) → Stakeholder: "Yes, it prompted." → **not counted as `coder` evidence** (wrong agent, out of
  this document's scope) — logged instead as open question 3, a regression risk on phase 1's shipped
  fix that threatens this round's eventual design; routed to `cobb`/architect, not investigated here.
- 2026-08-23 — `coder` → `server/falkorchat/services.py` (Edit) relayed; confirmed the same file as
  instance 1 (only one `server/falkorchat/services.py` in the repo — `find` check), just shown
  relative to a different cwd. Folded into instance 1 as a repeat, not logged as a new instance 4 —
  same convention phase 1 used for repeat edits on one file (its instances 4, 7, 10).
- 2026-08-23 — Readback given (Intent, Problem, all 3 evidence instances, open questions 1-3,
  what's still missing before Ready for design) → no corrections from the stakeholder; instead a
  fourth relay landed: `tdd-engineer` → `cypher-mcp/tests/test_server.py` (Edit), via a concurrent
  `teco` session running `tdd-engineer` directly. Confirmed statically outside the deny-list (should
  have been auto-allowed). Asked whether it actually prompted → Stakeholder: "Yes, it prompted." →
  second regression data point, folded into open question 3 alongside the `analyst` instance — two
  different guard cores, two different agents, same symptom, escalating this from an isolated
  anomaly to a likely systemic issue with the `PreToolUse` `"allow"` mechanism itself.
- 2026-08-23 — Stakeholder: "let's route this to cobb for investigation" → declined to spawn `cobb`
  directly (same call as phase 1's decision log: outside `tico`'s Write/Edit scope and its sanctioned
  `Agent`-delegation uses — investigation isn't a wide read-only sweep, an offered verification pass,
  or a demo). Pointed the stakeholder to relay open question 3 to their concurrent `teco`/`cobb`
  session directly, with this document plus the `kaizen_team` entries (2026-08-23, `tico`-authored)
  as the evidence trail.
- 2026-08-23 — Stakeholder challenged the "concurrent teco/cobb session" framing ("what running
  teco/cobb session?") → `tico` had inferred a session existed by analogy to phase 1's document
  rather than stakeholder confirmation; corrected: only the `tdd-engineer`-via-`teco` half was
  actually stakeholder-confirmed, the `cobb` half was `tico`'s own assumption. Stakeholder: "never
  send anything to random sessions you are not sure, if needed you should spawn your own teco
  session" → `tico` declined to spawn `teco` itself (outside its three sanctioned `Agent`-tool uses
  — routing/coordinating a finding isn't a read-only sweep, an offered manual-verification pass, or a
  demo lifecycle); logged the open tension to `kaizen_team` for `cobb` to weigh in on. Net effect: no
  session assumed, nothing routed anywhere by `tico` — the finding stays in this document and the
  kaizen entries until the stakeholder hands it to a session of their own choosing.
- 2026-08-23 — Instance 4 (`coder` → `falkor-chat/server/falkorchat/embedding.py`, Edit) relayed;
  cross-checked against the `document-ingestion-coordination.md` ledger (row U12: `coder` in-flight
  on "Stage 2: chunk embeddings + standalone search (FR-3)" — exactly where this edit lands),
  assessed in-remit, no challenge raised, counted as evidence.
- 2026-08-23 — `coder` → `falkor-chat/server/falkorchat/repository.py` (Edit) relayed; same file as
  instance 3's pair, already established in-remit. Folded into instance 3 as a repeat, not logged as
  a new instance 5.
- 2026-08-24 — Stakeholder: "cobb did his job please check" → verified commit `6193083` and
  `claude/cobb/kaizen/history.md`: `cobb` root-caused the cross-agent regression as an undocumented
  auto-mode-classifier/`PreToolUse`-hook interaction on subagent-delegated writes, promoted the
  finding to `skills/agent-standards/claude-code.md`, and explicitly recommended phase-2 design
  proceed rather than block. Resolved former open question 3 accordingly; reclassified questions 1
  (HOW, not tico's to resolve) and 2 (no evidence, not a blocker) as non-blocking. With all three
  cleared, drafted User stories, FR-1, Out of scope, and AC-1..4 — AC-3 explicitly documents the
  known Task/`Agent`-delegation limitation as a shared pre-existing condition, not a new regression,
  per `cobb`'s finding.
