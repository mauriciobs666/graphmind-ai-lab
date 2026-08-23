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

## Instances observed (live)
1. **2026-08-23 — `coder`, `Edit` on `falkor-chat/server/falkorchat/services.py`.** Coincides with
   `coder` implementing the gated `document-ingestion-coordination` plan (K-050,
   `falkor-chat/docs/plans/document-ingestion-coordination.md`, Status: active, owner `teco`) — repo
   state shows `repository.py`/`services.py` modified and a new `chunking.py`, consistent with
   in-progress implementation of that plan. **Confirmed legitimate** by the stakeholder — squarely
   `coder`'s own in-remit implementation work. → first evidence instance for this round.

2. **2026-08-23 — `coder`, `Create` (Write) on `falkor-chat/server/tests/test_chunking.py`.** Pairs
   with the new `falkor-chat/server/falkorchat/chunking.py` from the same K-050 implementation work
   — `coder`'s remit explicitly includes tests for the code it writes. No challenge: clearly in-remit,
   same task as instance 1. → second evidence instance; first Create (not Edit) and first test-file
   instance for this round.

3. **2026-08-23 — `coder`, `Edit` on `falkor-chat/server/tests/test_repository.py`.** Pairs with the
   already-modified `falkor-chat/server/falkorchat/repository.py`, same K-050 task as instances 1-2.
   No challenge: clearly in-remit. → third evidence instance, second `Edit` (not Create) on a test
   file.

## Open questions
1. Is `coder`'s fix shaped like `tdd-engineer`'s (phase 1 §6 — broad implementer, inverse
   deny-list guard over `guard-broad-write.sh`), or does live evidence point somewhere else (e.g.
   a narrower path pattern, or a different tool class like Bash)? Needs more instances before
   this is answerable — this is a WHAT/WHY question about what's actually firing, not a
   solutioning question (the HOW is the architect's, once this document is ready).
2. Are there Bash-confirmation instances too? Phase 1 flagged that `coder`'s friction might not be
   pure `Write`/`Edit` at all ("whatever it's hitting is suspected to be a *different* mechanism
   (e.g. Bash/terminal-command confirmations) than the `Write`/`Edit` confirmation this document
   resolves"). Worth keeping an eye out for as more instances land.
3. **Risk, not `coder` evidence — flagged, not resolved here:** 2026-08-23, `analyst` `Create` on
   `docs/reviews/document-ingestion-impl.md` **still triggered a manual confirmation prompt**,
   confirmed by the stakeholder, despite the path matching `analyst`'s already-shipped phase-1
   allowlist (`docs/reviews/*|*/docs/reviews/*` in `guard-review-doc-writes.sh`) and
   `guard-doc-writes.sh` statically confirmed today to emit the explicit `permissionDecision:
   "allow"` JSON on a match (`grep -n permissionDecision claude/scripts/guard-doc-writes.sh`). This
   looks like a **regression on phase 1's shipped fix**, not a gap in this document's scope — but it
   directly threatens whatever fix this phase-2 round eventually designs for `coder`, since it would
   rely on the identical `PreToolUse` `"allow"` mechanism. Not investigated further here (outside
   `tico`'s remit — no hook/execution debugging); needs routing to `cobb`/an architect pass before
   this document's eventual FR is trusted to actually suppress `coder`'s prompts once implemented.

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
