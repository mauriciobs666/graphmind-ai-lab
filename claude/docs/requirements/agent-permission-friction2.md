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

## Instances observed (live)
1. **2026-08-23 — `coder`, `Edit` on `falkor-chat/server/falkorchat/services.py`.** Coincides with
   `coder` implementing the gated `document-ingestion-coordination` plan (K-050,
   `falkor-chat/docs/plans/document-ingestion-coordination.md`, Status: active, owner `teco`) — repo
   state shows `repository.py`/`services.py` modified and a new `chunking.py`, consistent with
   in-progress implementation of that plan. Pending stakeholder confirmation this was in-remit
   (not yet confirmed as legitimate — this document does not assume it).

## Open questions
1. Is instance 1 confirmed as `coder` doing its own approved-plan implementation work (in-remit),
   or something else? (Needed before it counts as FR evidence, same discipline as phase 1's
   instance-by-instance confirmation.)
2. Same "im only sharing approved cases" shortcut from phase 1 — does the stakeholder want to
   re-adopt it for this round (treat every relayed instance as pre-confirmed), or confirm each one?
3. Is `coder`'s fix shaped like `tdd-engineer`'s (phase 1 §6 — broad implementer, inverse
   deny-list guard over `guard-broad-write.sh`), or does live evidence point somewhere else (e.g.
   a narrower path pattern, or a different tool class like Bash)? Needs more instances before
   this is answerable — this is a WHAT/WHY question about what's actually firing, not a
   solutioning question (the HOW is the architect's, once this document is ready).

## Decision log
- 2026-08-23 — Stakeholder: "we recently implemented permission friction and i did not have
  evidence on coder yet" → opens phase 2 of `agent-permission-friction`, per the explicit forecast
  in the phase-1 doc's closing decision (2026-08-21). New document on the ordinal slug
  (`agent-permission-friction2.md`) since phase 1 is `archived` (approved/executed).
- 2026-08-23 — First live instance relayed mid-turn (`coder` → `services.py`) while this document
  was being opened; logged as instance 1, not yet confirmed legitimate.
