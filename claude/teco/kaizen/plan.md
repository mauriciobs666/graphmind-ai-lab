# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-21

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-016 | 2026-08-21 | high | 🔵 | Consolidation pass on `teco.md` (dedicated `cobb` pass): merge same-family incident bullets, split rare-path rules into an on-demand `coordination-techniques.md` knowledge base |

### K-016 — consolidate teco.md + split rare-path rules into an on-demand knowledge base
- **Status:** 🔵 proposed · **Priority:** high
- **Rationale:** an in-depth analysis (2026-08-21, stakeholder-requested) found `teco.md` at
  ~4.7k words — the team's heaviest prompt, regrowing after two slimming passes because each
  kaizen distillation appends another incident-derived bullet. Steps 3–4 now carry several
  same-family rules ("don't trust a claim — verify it": stale placeholder results, fabricated
  haiku numbers, self-reported recoveries, stale message state, unverified docs) that could merge
  into one principle with sub-cases. The 2026-07-11 rationale for giving teco no on-demand
  knowledge base ("teco uses its whole body every run") no longer holds: incoming-message
  misrouting/staleness handling, the milestone-close freeze, and CPG freshness are rare-path
  rules paid for on every spawn.
- **Proposed change:** a dedicated `cobb` pass — (1) merge the verify-family bullets; (2) create
  `claude/teco/coordination-techniques.md` (the `analyst`/`qa-engineer`/`graph-dba` on-demand-KB
  pattern) and move the rare-path rules there with prompt-side one-line pointers; (3) re-run the
  §7 lint on the slimmed prompt. Stakeholder chose "dedicated pass" over doing it inside the
  2026-08-21 session (which already touched the prompt in several places).
- **Notes:** the same analysis's distillation-side counterweight — apply §5's "every session pays
  for it" bar more aggressively when the suggested home is teco's always-loaded prompt — belongs
  to `cobb`'s procedure, worth raising in the same pass.

> **K-015 — validate the dispatch-sizing rule on a real oversized plan — ✅ done 2026-08-21**
> (moved to history.md). K-028's implementation (15+ files across `services.py`, `repository.py`,
> `schemas.py`, `api.py`, `config.py`, `app.py`, `executor.py`, docs, and 5 test files, plus a
> mid-run plan-level defect forcing a mechanism redesign) finally crossed the ~3-step/5-file
> boundary this rule targets. teco split the work into named units by concern (**U3a** core logic
> vs. **U3b** wiring) and tracked defect-driven rework as its own distinct units (**U3a-fix**,
> **U3c**) rather than one mega-dispatch. Per-unit costs stayed well inside the K-042 baseline
> (458k tok/222 tools): largest single unit (U3a) 307k tok/134 tools; QA closed PASS, zero
> defects, no scope silently dropped. Rule confirmed under real pressure. One refinement flagged
> for K-016: the split-that-worked was by logical concern, not a literal file-count tally —
> consider restating the rule that way.

> **K-012 — verify `ListAgents` actually materializes — ✅ done 2026-08-21** (moved to
> history.md). Fresh-session probe (a spawned teco run, current `teco.md`): defined tools were
> exactly `Read, Bash, Agent, SendMessage, Write, Edit, mcp__cypher__query` — `ListAgents`
> absent alongside the known-absent `Grep`/`Glob`. Disposition per this item's own de-risking
> note: all three dropped from the frontmatter as decoration; step 5's attempt-`SendMessage`-
> first fallback already didn't depend on enumeration. The same probe live-verified
> `mcp__cypher__query` (a `kaizen_team` read succeeded), closing the parking-lot item below.
>
> **K-013 — exercise the `SendMessage` continuation loop for real — ✅ done 2026-08-16** (moved
> to history.md). K-026's own coordination ledger (Unit 2b needs-changes → fix → re-gate cycle)
> resumed the same `analyst` agent by its `agentId` for the re-gate; it re-verified its own
> earlier findings without being re-briefed on them — context genuinely preserved, not a cold
> respawn.
>
> **K-014 — the `agentId` ledger cell has no enforcement — ✅ done 2026-08-16** (moved to
> history.md). De-risked by evidence, not by adding a checker: every unit row in K-026's
> multi-session, ~20-unit coordination has its `agentId` filled, with exactly one explicitly-
> justified exception (a unit inherited from a prior session with genuinely no id to carry
> forward). Self-discipline held under real, sustained load — no checker needed per K-014's own
> stated criterion.
>
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
- **Grant `SendMessage` to dispatched specialists for self-reporting (noted 2026-08-21).** Live
  during the K-028 coordination (K-015's validation vehicle): `architect` and `analyst` each
  finished a unit and could not report the result back to teco directly — `architect` stated
  outright it had no `SendMessage` tool available; `analyst` said it would send one but no such
  message ever reached teco. Both times the launching/parent session had to observe the
  specialist's own completion and relay the result into teco by hand to keep the coordination
  moving. Raw fact logged to `kaizen_team` (`teco-20260821-specialist-sendmessage-gap`,
  `suggestedHome: prompt`) for `cobb` to verify/route. If adopted, the change is at least two
  parts: (1) a `cobb` agent-standards pass adding `SendMessage` to the relevant specialists'
  `tools:` allowlists, and (2) a protocol line in each one's own prompt — relay your result to the
  coordinator's `agentId` when the brief names one, and only that address, not open-ended
  messaging to arbitrary agents/sessions. Worth weighing against the fact that the current
  "parent relays" workaround already works; this would mainly save a manual step, not unblock
  anything. Decide when `cobb` next touches specialist tool grants, not as a standalone push.
- ~~Live-verify the `mcp__cypher__query` grant (added 2026-08-19 for the centralized CPG-freshness duty, `docs/plans/cpg-agent-adoption2.md`).~~ *(✅ Resolved 2026-08-21 — the K-012 fresh-session probe called it live: a `kaizen_team` read succeeded. Caveat removed from Guardrails; the freshness duty stands on a verified tool. See history.md.)*
- **Architect plans annotate dispatch-unit boundaries (noted 2026-08-21).** The dispatch-sizing rule (K-015) asks teco to derive unit boundaries from a step table under pressure; if `architect`'s plan template instead marked suggested dispatch clusters in the step table itself, teco's job becomes verification, and K-015 gets exercised far more easily. Cross-agent change — belongs to `architect`'s prompt/handoff contract; raise when K-015 or K-016 is worked.
- **Cross-session addressing hygiene (noted 2026-08-21).** The 2026-08-16 misrouting incident is mitigated by a pause-and-confirm rule; a cheaper preventive convention would be: the coordination doc records its session's identity, and any inter-session `SendMessage` must echo the coordination slug — a message without a matching slug is declined without analysis. Decide from the next incident, not now.
- ~~Guardrails commit bullet is dense (2026-07-30); step-3 density (2026-07-29); model-routing
  evidence clause (2026-07-29).~~ *(✅ All three resolved 2026-08-10 — steps 3/4 split into
  sub-bullets, Guardrails' commit bullet split four ways, the dated evidence clause dropped from
  the operative instruction. See history.md.)*
- **Watch the milestone-close freeze in a real close (noted 2026-07-27).** The new curation bullet is prompt-level only — nothing enforces that the `Status: archived` flips actually land, and the owners performing them (`architect`, `analyst`, `tico`, `qa-engineer`, `data-scientist`, `graph-dba`) have no matching instruction in their own prompts yet; they learn it from the brief. If a close ships with documents left `active`, the fix is either a line in each owner's prompt or the optional checker (step 7 of `docs/plans/doc-reference-convention.md`, which today gates nothing) — decide from evidence, not now.
- ~~A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.~~ *(✅ Resolved 2026-07-09: the roster is now an explicit routing table — task shape → owner → tie-breaker — with the coder-vs-tdd efficiency rule on both implementer rows, plus a separate handoff-contracts list. See history.md.)*
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
- Minor §7-lint nits (2026-07-16, low value — noted not filed): (a) the Guardrail "`Write`/`Edit` is for the **coordination doc only**" is stricter than the enforcement it describes (the hook escalates only writes *outside* `docs/plans/`, permitting any file there) — prose and backstop are intentionally different scopes but read as if aligned. (b) The implementer-routing efficiency rule is stated three times (description, routing table, How-you-work) — deliberate reinforcement, some redundancy. (c) The Handoff-contracts list restates specialist doc paths that also live in each specialist's injected `description`, mild tension with teco's own "don't re-derive [descriptions]" line — but this is the §4 handoff-symmetry pattern (state on both sides), so it's a feature with a drift cost, not a defect.
