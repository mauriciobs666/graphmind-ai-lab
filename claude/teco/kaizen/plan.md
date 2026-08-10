# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-10

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-012 | 2026-08-10 | high | 🔵 | Verify `ListAgents` actually materializes in a teco run |
| K-013 | 2026-08-10 | med | 🔵 | Exercise the `SendMessage` continuation loop once, for real |
| K-014 | 2026-08-10 | low | 🔵 | Nothing enforces that the ledger's `agentId` cell gets filled |

### K-012 — verify `ListAgents` actually materializes
- **Status:** 🔵 proposed · **Priority:** high
- **Rationale:** the 2026-08-10 probe found teco's *runtime* tool list to be exactly
  `Read, Bash, Agent, SendMessage, Write, Edit` — **`Grep` and `Glob` are declared in
  frontmatter and simply absent at runtime**, silently, with no error. A frontmatter grant is
  therefore not proof of availability on this harness build, and `ListAgents` was granted the
  same day on the assumption that it is.
- **Confound found the same day:** a verification probe run *from the session that made the
  grant* reported `ListAgents` absent — but custom agent definitions load at **parent-session
  start**, so that session was still holding the pre-edit `teco.md`. The negative result is
  therefore inconclusive for `ListAgents` (it is conclusive for `Grep`/`Glob`, which predate the
  session). Re-probe from a **fresh session**.
- **Already de-risked:** step 5 no longer depends on the answer — it says to attempt the
  `SendMessage` and treat an addressing error as non-resolution, and Guardrails records the
  runtime tool set explicitly. If a fresh-session probe shows `ListAgents` still absent, drop the
  frontmatter grant as decoration; if present, add the enumeration step back to step 5.

### K-013 — exercise the `SendMessage` continuation loop for real
- **Status:** 🔵 proposed · **Priority:** medium
- **Rationale:** the mechanism has been prompt-level since 2026-07-29 and has **never fired**
  (0 occurrences in any confirmed teco run vs. 42 repo-wide). The 2026-08-10 pass made it
  addressable (`agentId` in the ledger) but still unexercised; an unexercised path is a claim.
- **Proposed change:** on the next coordination with a review gate, deliberately route the
  re-check through `SendMessage` and record the first datapoint here — did resuming actually
  preserve the delegate's context, and what did it save vs. a cold respawn?

### K-014 — the `agentId` ledger cell has no enforcement
- **Status:** 🔵 proposed · **Priority:** low
- **Rationale:** the whole continuation path now depends on a cell teco fills by self-discipline;
  no hook or script checks it. Same enforcement-parity shape as the milestone-close freeze below.
- **Proposed change:** decide from evidence after a few coordinations — if the cell gets skipped,
  a cheap `audit`-style checker over `*-coordination.md` files is the fix; if it doesn't, leave it.

> **K-006 — review-default list has no reviewer for agent-engineering deliverables — ✅ done
> 2026-07-29** (moved to history.md). Disposition: made explicit in the existing "Work ships
> independently reviewed" defaults clause rather than a new row — `plans and code (including
> graph-dba design notes and cobb's agent/skill artifacts) → analyst`.
>
> **K-008 — per-call `model` override for cost routing — ✅ done 2026-07-29** (moved to
> history.md). Live-verified the override reaches an `Agent` call made from inside a subagent
> (nested transcript showed `"model":"claude-haiku-4-5-20251001"`); adopted into step 3.
>
> **K-009 — tool-grant audit: WebFetch/WebSearch — ✅ done 2026-07-29** (moved to history.md).
> Audited all 5 teco session transcripts; found one use (2026-07-24, arguably mis-routed —
> should have gone to cobb). Dropped both from `tools:`.
>
> **K-010 — scheduled token-cost recompression — ✅ done 2026-07-29** (moved to history.md).
> Trimmed the description's trailing clause from 170 to 88 chars; body left as-is (still below
> its pre-compression peak, not urgent).
>
> **K-011 — prune stale Bash allow-rules — ✅ done 2026-07-29** (moved to history.md). Removed
> the three single-use K-001-probe entries from `.claude/settings.local.json`.
>
> **K-007 — SendMessage continuation instead of cold respawn — ✅ done 2026-07-29** (moved to
> history.md). Step 3 now has teco note each delegate's returned name/id when a follow-up round
> is likely; step 4's two re-brief paths (review "needs changes"/defects, and a deficient result)
> both `SendMessage` the original delegate by that identifier instead of a fresh `Agent` call,
> falling back to cold respawn only when the identifier no longer resolves. `tools:` gained
> `SendMessage` (was missing — K-007 was otherwise unshippable).
>
> **K-004 — deficient/failed-delegate-result path — ✅ done 2026-07-16** (moved to history.md).
> Step 4 now handles a deficient result (errored / out of turns / off-brief / empty): re-brief the
> same owner once with the gap explicit, pause to the user if it recurs or the unit is mis-scoped —
> distinguished from a *blocker* and a review *verdict*.
>
> **K-005 — doc-curation scan includes HISTORY.md / BACKLOG.md — ✅ done 2026-07-16** (moved to
> history.md). The documentation-impact scan now lists `docs/HISTORY.md` (entry per delivered change)
> and `docs/BACKLOG.md` where the module uses the convention.
>
> **K-002 — agent teams / background agents evaluation — ✅ done 2026-07-24** (moved to
> history.md). Disposition: **reject** reframing teco as an agent-teams lead — its loop is
> sequential/dependency-gated, exactly the shape the agent-teams docs say a single
> session/subagents handles better than teams. The 2026-07-12 sub-case (defect→fix→re-run
> respawning cold) isn't an agent-teams question — it's answered by `SendMessage` continuation
> of the original delegate. Spun off as **K-007**.
>
> **K-001 — validate nested delegation end-to-end — ✅ done 2026-07-09** (moved to history.md).
> Live run: falkor-chat M3 slice 1 through teco → architect → graph-dba → tdd-engineer, all
> checklist items passed. Launch brief + observation checklist preserved at
> [`k001-run-brief.md`](./k001-run-brief.md).
>
> **K-003 — review-gate invariant: prove it or renegotiate it — ✅ done 2026-07-12** (moved to
> history.md). Disposition **(a) keep the invariant** — the first fully-gated run (falkor-chat
> K-022 Landing 1) enforced the analyst gate and captured the cost datapoint: the gate is cheap
> (~7% of wall time, ~12% of tokens) and caught a major + minors on a "done" diff. No prompt
> change. Counterparts still open: `analyst` K-001, `qa-engineer` K-003 (unexercised — 0 blockers).

## Parking lot / ideas
- ~~Guardrails commit bullet is dense (2026-07-30); step-3 density (2026-07-29); model-routing
  evidence clause (2026-07-29).~~ *(✅ All three resolved 2026-08-10 — steps 3/4 split into
  sub-bullets, Guardrails' commit bullet split four ways, the dated evidence clause dropped from
  the operative instruction. See history.md.)*
- **Watch the milestone-close freeze in a real close (noted 2026-07-27).** The new curation bullet is prompt-level only — nothing enforces that the `Status: archived` flips actually land, and the owners performing them (`architect`, `analyst`, `tico`, `qa-engineer`, `data-scientist`, `graph-dba`) have no matching instruction in their own prompts yet; they learn it from the brief. If a close ships with documents left `active`, the fix is either a line in each owner's prompt or the optional checker (step 7 of `docs/plans/doc-reference-convention.md`, which today gates nothing) — decide from evidence, not now.
- ~~A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.~~ *(✅ Resolved 2026-07-09: the roster is now an explicit routing table — task shape → owner → tie-breaker — with the coder-vs-tdd efficiency rule on both implementer rows, plus a separate handoff-contracts list. See history.md.)*
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
- Minor §7-lint nits (2026-07-16, low value — noted not filed): (a) the Guardrail "`Write`/`Edit` is for the **coordination doc only**" is stricter than the enforcement it describes (the hook escalates only writes *outside* `docs/plans/`, permitting any file there) — prose and backstop are intentionally different scopes but read as if aligned. (b) The implementer-routing efficiency rule is stated three times (description, routing table, How-you-work) — deliberate reinforcement, some redundancy. (c) The Handoff-contracts list restates specialist doc paths that also live in each specialist's injected `description`, mild tension with teco's own "don't re-derive [descriptions]" line — but this is the §4 handoff-symmetry pattern (state on both sides), so it's a feature with a drift cost, not a defect.
