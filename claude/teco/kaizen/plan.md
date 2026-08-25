# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-08-24

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
- **2026-08-24 (C1 pass-2 lint, `cobb`) — K-016 is now the *only* remaining lever.** Prompt-waste
  C1 passes 1+2 took `teco.md` 5,948 → 5,343 w by editorial means (class 5/6/7). A mechanical
  repeated-phrase scan over the result finds well under 200 w of cross-line restatement left, and
  cobb can name only ~115 w of further defensible class-7 cuts. **The editorial floor with every
  rule intact is ~5,200–5,250 w** — the file is ~60 distinct rules at ~85 w each, not a narrative
  file. Anything below that requires moving rules out of always-loaded context, i.e. this item.
  Caveat for whoever executes it: the strongest KB candidate by word count is the paused-unit /
  stop-and-ask protocol (~380 w, step 4), but it is a **reactive** protocol — teco must recognize
  the trigger to know to load the file, so only the mechanics (two-hop chain, resume addressing)
  can move; the trigger and ledger shape must stay inline. Same test applies to the
  misrouting/staleness rules already named above.

## Parking lot / ideas
- **A delegation-summary table cites, it does not restate (routed here 2026-08-25, prompt-waste
  Stage D).** The plan (`claude/docs/plans/prompt-waste-reduction.md`) specified this rule for
  `architect.md`, but `architect` authors no delegation-summary table — the unit/step ledger in
  `plans/<slug>-coordination.md` is **this** agent's artifact, so the rule was generalized to "a
  recap table" there and its specific instance lands here. Not added to `teco.md` in that unit: it
  is a rule change, and §4.0's rollback contract keeps those out of a compression commit. Worth
  weighing on the next `teco.md` pass — a coordination ledger that restates each unit's decisions
  rather than citing the plan section is the highest-volume instance of the duplication the whole
  plan targets, and `teco.md` is the largest file on the team (5,377 w).
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
- **Architect plans annotate dispatch-unit boundaries (noted 2026-08-21).** The dispatch-sizing rule (K-015) asks teco to derive unit boundaries from a step table under pressure; if `architect`'s plan template instead marked suggested dispatch clusters in the step table itself, teco's job becomes verification, and K-015 gets exercised far more easily. Cross-agent change — belongs to `architect`'s prompt/handoff contract; raise when K-015 or K-016 is worked.
- **Cross-session addressing hygiene (noted 2026-08-21).** The 2026-08-16 misrouting incident is mitigated by a pause-and-confirm rule; a cheaper preventive convention would be: the coordination doc records its session's identity, and any inter-session `SendMessage` must echo the coordination slug — a message without a matching slug is declined without analysis. Decide from the next incident, not now.
- **Watch the milestone-close freeze in a real close (noted 2026-07-27).** The new curation bullet is prompt-level only — nothing enforces that the `Status: archived` flips actually land, and the owners performing them (`architect`, `analyst`, `tico`, `qa-engineer`, `data-scientist`, `graph-dba`) have no matching instruction in their own prompts yet; they learn it from the brief. If a close ships with documents left `active`, the fix is either a line in each owner's prompt or the optional checker (step 7 of `docs/plans/doc-reference-convention.md`, which today gates nothing) — decide from evidence, not now.
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
- Minor §7-lint nits (2026-07-16, low value — noted not filed): (a) the Guardrail "`Write`/`Edit` is for the **coordination doc only**" is stricter than the enforcement it describes (the hook escalates only writes *outside* `docs/plans/`, permitting any file there) — prose and backstop are intentionally different scopes but read as if aligned. (b) The implementer-routing efficiency rule is stated three times (description, routing table, How-you-work) — deliberate reinforcement, some redundancy. (c) The Handoff-contracts list restates specialist doc paths that also live in each specialist's injected `description`, mild tension with teco's own "don't re-derive [descriptions]" line — but this is the §4 handoff-symmetry pattern (state on both sides), so it's a feature with a drift cost, not a defect.
