# Kaizen — Change History: coder

> Dated log of actual changes to the `coder` agent. Most recent first.

## 2026-08-28 — Broad-write guard added (prompt-friction fix, closes coder's FR-2 gap)
- **What:** Added `hooks/guard-coder-broad-write.sh` (thin wrapper over
  `claude/scripts/guard-broad-write.sh`, deny-list kept in lockstep with tdd-engineer's
  `guard-tdd-broad-write.sh`) and wired it in frontmatter (`hooks:` PreToolUse `Write|Edit`,
  plus `permissionMode: acceptEdits` matching the rest of the team).
- **Why:** Transcript evidence from teco session `2e0c2f42` (2026-08-27/28): every hook-free
  coder `Write`/`Edit` in an auto-mode teco orchestration hit a permission prompt (waits of 11s
  to 5.3h), while guard-carrying agents wrote silently in the same hours. An explicit PreToolUse
  `"allow"` is the one mode-independent prompt suppressor
  (`claude/docs/plans/agent-permission-friction.md` §1.3); coder was the implementer that never
  received one.

## 2026-08-25 — `kaizen_team` distillation: 10 raw entries processed (2 promoted, 2 discarded as superseded, 2 discarded as low-value, 1 kept open as K-005, 3 tagged `MENTIONS`→`graph-dba`)
- **What:** Ran `agent-maintenance` §5 over every `coder`-produced entry in `kaizen_team` — 4 legacy
  (`author: 'coder'`) + 6 current-shape (`PRODUCED` edges), all dated 2026-08-21/24/25, from K-028
  workflow-timers and K-050 M5 document-ingestion work. Each verified by re-deriving the fact
  myself (source reads, live `curl`, a Python repro script, or a fresh `mcp__cypher__query` count),
  not by re-confirming the entry's own citation.
  1. **`c4a9d1e6…` (circular-import direction) — promoted.** Empirically re-verified with a
     throwaway `pkg/a.py↔b.py` (fails both orders), a class-body variant (fails identically — the
     entry's own "not a class body" framing checked out), and a function-body variant (succeeds) —
     CPython 3.12. Added to `skills/python-web-quirks/SKILL.md` as a new section, tightened and
     corrected from the raw entry's slightly garbled wording; frontmatter `description` updated.
  2. **`d8f3b6a2…` (~8KB `Step.key` crashes the shared FalkorDB container) — discarded, superseded.**
     `falkor-chat/docs/BACKLOG.md` K-049 already tracks this exact incident (opened the same day,
     same author/context); `graph-dba/falkordb-quirks.md` already carries a **far more precise**,
     confirmed root cause (verified 2026-08-22: SIGSEGV, exact 4096/4097-byte boundary on any
     `UNIQUE`-constrained property, isolated-container repro, RCA doc) that supersedes this entry's
     own "root cause NOT confirmed, ~8KB" framing entirely. Nothing left to add.
  3. **`b7e2c9a1…` (TestClient teardown cancels tasks regardless of app lifespan) — promoted.**
     Re-reproduced fresh with a minimal FastAPI app whose lifespan never cancels its own task —
     `task.done()==True, task.cancelled()==True` right after the `with TestClient(app):` block
     exits anyway (starlette 1.3.1 / fastapi 0.139.0 / anyio 4.14.1, same versions the skill file's
     other entries already cite). Added to `skills/python-web-quirks/SKILL.md`.
  4. **`a3f1e8c2…` (unconditional wait/human transition can never suspend) — discarded, superseded.**
     This is the *same* defect `falkor-chat/docs/HISTORY.md`'s 2026-08-21 K-028 entry already
     documents exhaustively under "How it got here" — the v1/v2 unconditional-fallback design was
     caught by exactly this `coder`-authored finding during implementation, `teco` independently
     re-verified it against live source, and v3 (the shipped, QA-accepted design) replaced it with
     the `ctx.timerFired` marker-guard mechanism. The raw entry is the pre-fix stepping stone; the
     project record already supersedes it in full.
  5. **`c1e8a0a2…` (`test_queries.sh` reference-wipe false-negatives `verify_workflows.sh`'s
     snapshot check) — kept open, opened `plan.md` K-005.** Verified TRUE by tracing actual source
     (`services.py:1748`, `repository.py:1717`) — a real, currently-live bug: `_read_structure`'s
     unguarded `ro_query` against a fully-`GRAPH.DELETE`d `reference` raises an "empty key"
     `ResponseError` that `verify_workflows.sh`'s `read()` wrapper turns into a whole-diff `ABSENT`,
     falsely reporting an intact `ws:<id>` snapshot as missing too — contradicting
     `diff_def_snapshot`'s own docstring claim to handle this gracefully (true only when
     `reference` exists-but-empty, not when the graph key itself is gone). Both fixes (an
     AGENTS.md doc correction, a code fix or a BACKLOG defect) land in `falkor-chat/`, outside
     `cobb`'s write remit — see K-005 for the full trace and the recommended `teco` routing.
  6–8. **`7e3d1a2b…` (`count(*)` undercounts parallel edges), `7f3c2e1a…` (undirected
     relationship-property-filter pattern silently degrades to directed), `b2d8f4a1…` (REFINES
     `7f3c2e1a…`: the trigger is any relationship-property predicate, inline or `WHERE`, not just
     an inline filter) — tagged `MENTIONS`→`graph-dba`, not cleared.** All three are FalkorDB
     engine-behavior facts (not "how `coder` should behave"), none found already in
     `graph-dba/falkordb-quirks.md`'s "Cypher dialect & query behavior" section, and none
     independently reproducible by me — confirming a parallel-edge/write-based Cypher behavior
     needs a live write-capable probe, which is `graph-dba`'s tool access, not a curator's. Tagged
     `MENTIONS` per `agent-maintenance` §5 step 3's "substantively about a different agent" rule;
     only each entry's `PRODUCED` edge was resolved this pass (`otherRemaining=1` after — the fresh
     `MENTIONS` edge — so no `DETACH DELETE`), leaving the node live for `graph-dba`'s own future
     distillation pass to verify/promote/clear.
  9. **`a3f8c2e1…` (a pytest `FakeRepo` with one shared `since_rows` attribute can't test two
     merged repo methods independently) — discarded.** Already fixed in the actual test code
     (`self.hybrid_rows`/`self.chunk_rows`, falling back to `since_rows`) — the fix is
     self-documenting in the fixture itself; the underlying lesson ("give each mocked behavior its
     own controllable state") is standard test-double practice, not a durable/non-obvious
     environment fact worth a standing knowledge-base entry.
  10. **`b7e1d4a2…` (LM Studio reachable at `localhost:1234` in this sandboxed dev env) —
      discarded.** Re-verified live (`curl` succeeded, same as the entry). `falkor-chat/AGENTS.md`
      already flags the `pytest -m live` → LM Studio dependency; whether the local server happens
      to be running right now is transient session state, not a stable environment fact, and the
      "check reachability before assuming untestable" tip is generic testing advice not specific
      enough to earn a standing entry.
- **`MENTIONS` tags added:** `7e3d1a2b…`, `7f3c2e1a…`, `b2d8f4a1…` → `graph-dba` (all three
  `mcp__cypher__query(agent='cobb')` `MERGE`s, committed before any clearing ran this pass, per the
  ordering invariant).
- **Cleared from `kaizen_team` this pass:** `c4a9d1e6…`, `d8f3b6a2…`, `b7e2c9a1…`, `a3f1e8c2…`
  (legacy, unconditional `DETACH DELETE`); `a3f8c2e1…`, `b7e1d4a2…`, `c1e8a0a2…` (current-shape,
  `otherRemaining==0` after resolving the sole `PRODUCED` edge → full `DETACH DELETE`). **Left
  live** (current-shape, only the `PRODUCED` edge resolved): `7e3d1a2b…`, `7f3c2e1a…`, `b2d8f4a1…`.
- **Docs touched:** `claude/coder/kaizen/{plan,history}.md` (this file + K-005) ·
  `skills/python-web-quirks/SKILL.md` (two new sections + frontmatter `description`).
- **Plan items:** K-005 opened (kept-open disposition, item 5 above).

## 2026-08-25 — K-003 closed — the scope guardrail now defers to plan fidelity on the one overlapping case
- **What:** `:11` said a mid-build plan defect means *"stop and say so with a concrete proposal"*; `:35` said surface plan defects *"as notes for the user — don't just do them."* Items 2 and 3 of that list plainly mean note-and-continue, so the collision was item 1 only — but for a mid-build plan defect the two rules said halt and don't-halt. `:35` now defers: **a plan defect that blocks the current step stops the work; one that doesn't, becomes a note.** +26 w.
- **Cited by bolded name, not "(step 1)" as K-003 proposed.** `coder.md` carries both a "What you optimize for" bullet list and a numbered "How you work" list whose step 1 is **Orient** — so "(step 1)" would have pointed at the wrong rule. An ambiguous pointer is worse than a fragile one.
- **The gate lint found K-003 half-closed and the fix was extended.** The contradiction was resolved but the failure mode K-003's rationale singles out — *"it bites hardest when `coder` runs delegated: 'stop' there means returning the unit undone"* — was not. `coder.md` has **no general subagent protocol** (nothing like `security-expert.md:20`); its only two subagent branches are case-scoped to baseline blockers (`:19`) and destructive asks (`:36`), and neither reaches a mid-build plan defect. So a delegated `coder` that finished steps 1–2 of five and hit a blocking defect at step 3 had no instruction to report the completed work — §6's "skips a required artifact element", and a unit `teco` cannot resume from. Added: *"Delegated, 'stops' means returning the defect and your proposed fix as your result, alongside the steps you did complete"* — modelled on `:19`, four lines up, which already handles the structurally identical case correctly.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint 1 major (fixed in-commit), 1 nit accepted.

## 2026-08-25 — K-004 closed — `:12` now names the local-vs-project tiebreak (conventions-precedence family)
- **What:** `:12`'s conventions rule mixed a project-scope authority (*"already in the codebase"*) with a local-scope discovery heuristic (*"reading neighboring code"*). They agree in a consistent codebase and diverge in exactly the case worth a rule. One sentence appended; +25 w.
- **`:29` is not part of the collision and was left alone** — *"Its conventions… win over your defaults"* runs on the project-vs-**agent** axis, pointing away from the agent's habits just as `:12` does. Confirmed by the lint.
- **The three "judged and kept, do not re-litigate" restatements** in `kaizen/plan.md`'s parking lot are undisturbed — none touches conventions or style, and the addition is a tiebreak *between* two clauses in one bullet, not a restatement of either. **K-003** (`:11` vs `:35`, stop vs. note-and-continue) is untouched and stays open: a separate rule change.
- **The rule, byte-identical in all three implementer prompts:** *"Where a file or folder deviates locally from the project norm, match it, not the norm — mixing both in one place is worse than either applied consistently."* Identity is deliberate — three paraphrases would recreate exactly the divergence that made this a `agent-maintenance` §4 **check-5 boundary-reciprocity** problem rather than three separate nits.
- **Not relocated to a shared file, and the reason is stronger than "no plausible home."** Finding 15 says a rule binding N>1 agents belongs where all N read it, and this binds three. But no shared file owns *how to author code*, and root `AGENTS.md` would be an **actively bad** home: its document conventions are stated as absolutes (`never begins with m<digit>`, the closed role set, the closed `Status:` set), and a general "a local deviation beats the project norm" principle sitting beside them hands every agent a lever to justify deviating from them. The rule would not sit inertly there; it would undercut them. So: byte-identical duplication, **plus a mechanical identity guard** — `audit-team.sh` **check 10**, which fails when the sentence is in some but not all three, and passes when it is in all three or none (removing it everywhere is a legitimate family decision; removing it from one is the defect). Recorded as plan finding 22.
- **Why the guard was necessary and not belt-and-braces.** All three kaizen plans said "fix together or not at all" and nothing read those plans at edit time. The mitigation was prose in files nothing loads. Stage F had shipped the enforcement machinery the same day, and this is exactly the class of guarantee it exists for: deterministic, must-always-hold, therefore a mechanism rather than hopeful prose.
- **The family boundary is a test, not a judgment call:** *two co-equal scope claims in one prompt with no tiebreak between them.* `graph-dba:53` and `devops` were checked and excluded — not because they are a "different axis" (that was my first, wrong reasoning; `graph-dba:53` is structurally identical and has a real local-vs-project instance in this repo, since `cpg_*` graphs carry Joern's imported schema against the `PascalCase`/`UPPER_SNAKE` norm) but because each states **one** scope, so there is nothing to tiebreak. Recorded so the next reader does not re-open it, notice the shape match, and conclude the exclusion was arbitrary.
- **Verified:** `audit-team.sh` **PASS**, check 10 green at 3/3; negative-tested at 2/3 (FAIL, exit 1) and 0/3 (PASS). `cobb` §7 lint: 0 blockers, 1 major, 3 minors, 2 nits — all applied.

## 2026-08-24 — Prompt-waste compression, Stage C6: inventoried, **zero edits** — judged at its editorial floor
- **What:** `coder.md` was one of C6's four files (`claude/docs/plans/prompt-waste-reduction.md`, Stage C). Full class-5/6/7 inventory run; **no edit made**, and that is the finding, not a shortfall. 1,240 w before and after. Wave 1 + the wave-2 micro-fix had already taken everything class-5/6 this file ever carried.
- **Class-6 residual: 0 w.** Mechanically confirmed — zero hits for dates, FR/AC tags, authority markers, supersession trails, or `kaizen/history.md` pointers anywhere in the file. Nothing left to cut in the cheap category, permanently (plan finding 11).
- **Class-7 residual: ~34 w, all of it certified keeps** under finding 5 ("needed twice", not "said twice"):
  - **"Don't claim what you didn't run" (guardrail) vs. step 5's "Report what you actually ran and saw."** Step 5 is the report-writing *procedure* (show the output; report `passed`/`skipped`/`deselected`, since a suite can exit 0 with a chunk silently unrun); the guardrail is the absolute prohibition. Same pair `qa-engineer` certified as a keep at C5.
  - **"Ask before destructive or environment-changing actions" (guardrail) vs. step 2's "ask before installing or mutating the environment."** Two decision points — baseline setup vs. anywhere in the build — each carrying its own subagent carve-out. Also the pair `qa-engineer` certified at C5; the carve-out duplication is a **certification requirement** (`agent-maintenance` §4 check 3), not a style choice.
  - **The three scope statements** — "Minimal blast radius" (scope of edits) / "Don't silently exceed scope" (reporting obligation) / "don't silently diverge" (what to do when the plan is *wrong*). Three rules, not one restated.
- **Gate (a) inventory — trivially satisfied:** no text removed, so every rule survives byte-identical.
- **Two pre-existing defects found and routed, not bundled** (§4.0 rollback contract — both fixes are rule changes):
  - **K-003** — `:11` and `:35` prescribe *different actions for the same trigger* (a mid-build plan defect: "stop and say so" vs. "surface as notes for the user"). `cobb`'s lint dissented from this session's initial reading that the three scope statements were cleanly distinct, and it is right: this pair is a genuine contradiction, worst when `coder` runs delegated (where "stop" means returning the unit undone with no way to ask).
  - **K-004** — `:12`'s conventions ambiguity: "already in the codebase" (project scope) vs. "reading neighboring code" (local scope), which return different answers when a file deviates locally. Same family as `tdd-engineer` **K-006**, weaker instance — here the second clause reads as the *method* for the first rather than a competing authority, so the collision is inside `:12` alone. `:29` ("its conventions win over your defaults") runs on the project-vs-*agent* axis and does not collide.
- **Verified:** `audit-team.sh` PASS (check 8 row intact, file untouched); `cobb` §7 lint — clean on all seven dimensions for this file, the two findings above pre-existing.

## 2026-08-23 — Freshness-clause grammar fix (Stage B wave 2 micro-shape)
- **What:** "a `teco`-issued brief that states the graph's freshness, take it as given" → "when a `teco`-issued brief states the graph's freshness, take it as given" — closing the hanging-topic construction cobb's wave-1 lint flagged as minor; applied uniformly across all files carrying the clause. No rule change; both branches intact.

## 2026-08-23 — Prompt-waste compression, Stage B wave 1 (boilerplate sweep)
- **What:** Applied the three pilot-validated boilerplate compressions from
  `claude/docs/plans/prompt-waste-reduction.md` (§3 doctrine, Stage B), same shapes as the
  `architect.md` pilot. (1) CPG-freshness clause (§ "Orient"): dropped the "(2026-08-19)" date and
  the redundant "without re-deriving staleness yourself" tail; rule now reads "CPG freshness is
  `teco`'s responsibility, not yours: a `teco`-issued brief that states the graph's freshness, take
  it as given; running standalone, use the CPG's answers as current." (2) Interactive-commit-grant
  bullet (§ Guardrails): dropped the provenance sentence "Stakeholder decision, 2026-08-21 — see
  `kaizen/history.md`." and ", same as before"; the "(spawned via `Agent`/`Task`)" clarifier moved
  from the interactive-definition parenthetical to the carve-out sentence (was stated in both).
  (3) Learning capture: intro dropped "directly" and "identified by a real `:Agent` node it's
  `PRODUCED`-linked to," (the Cypher template below shows the MERGE + PRODUCED edge); tail dropped
  the inbox-replacement history sentence and "exactly like the old inbox was".
- **Rule inventory (gate a), edited regions — all preserved:** freshness is `teco`'s / brief taken
  as given / standalone = current (block 1); interactive-mode definition, explicit-path grant,
  full never-list, delegated-subagent carve-out, deliverable left for `teco` post-verification
  (block 2); capture trigger + graph + Cypher template, skip-known-facts, raw-capture/`cobb`
  promotes, never edit own definition (block 3). Verbatim `CPG:` three-form sentence and
  audit-check-8 tokens untouched.
- **Removed class-5/6 material, recorded where:** inbox-replacement history → this file's
  2026-08-21 "kaizen/inbox.md deleted" entry; commit-grant provenance → this file's 2026-08-21
  grant entry + `claude/AGENTS.md` § Hook machinery; freshness centralization date → this file's
  2026-08-19 "Freshness-check clause removed" entry.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint pass on the result.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Guardrails bullet: when running interactively (`claude --agent coder`, a human
  present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own verified
  code changes from the session, by explicit path, never bulk-staged/pushed/reset/rebased/
  amended; the grant does not apply when spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Verify and report"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there has already been distilled and cleared — and this file's own pre-migration content (if any) was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry below). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_coder`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_coder` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," and "never edit your own agent definition" (no write-guard clause — coder has no doc-scoped write guard). Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Freshness-check clause removed (centralized on teco)
- **What:** Dropped the CPG freshness-check paragraph from the CPG-orientation step — still checks whether a relevant CPG exists and uses it via `cpg-analysis`, but no longer queries the `:CpgBuildInfo` freshness marker itself. That responsibility is now `teco`'s alone (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`); running standalone (no `teco`-issued brief), staleness is simply not checked.
- **Why:** User-directed prompt-verbosity reduction: the freshness paragraph was ~130 words, byte-identical across six agent files. Stakeholder chose full centralization over a per-agent dedup, accepting the standalone-run capability loss.
- **Plan items:** —

## 2026-08-16 — U7 fix round: freshness-check sequencing hardened, `CPG:` line anchored (DEF-1/DEF-2/DEF-3)
- **What:** Two wording tightenings per `docs/plans/cpg-agent-adoption-coordination.md` unit U7,
  following U6's `qa-engineer` live-dispatch acceptance pass
  (`docs/test-reports/cpg-agent-adoption-report.md`). (1) The freshness-check sentence now reads
  "query the freshness check … in that same tool call/step, before deciding whether the result
  needs further cross-verification — this is not a separate, optional judgment call" (previously
  "also run the freshness check … as part of that same step") — closes DEF-2 (`architect`
  reasoned its way past the check with a grep/CPG-agreement substitute that doesn't rule out
  "stale but coincidentally consistent"). (2) The `CPG:` line instruction now reads "written
  verbatim and required in all three cases including when the CPG isn't relevant — not
  paraphrased, not dropped" — closes DEF-1 (`coder`, this agent, used the CPG correctly but wrote
  loose prose ("**CPG freshness note:**…") instead of the literal line) and DEF-3 (`tdd-engineer`
  dropped the line entirely on the not-relevant branch). Applied identically (phrasing pattern,
  not restructuring) across all six wired agents; only `coder`/`architect`/`tdd-engineer` were
  live-tested, but the near-verbatim wiring pattern means the same gap plausibly existed in
  `analyst`/`qa-engineer`/`frontend-engineer` too.
- **Why:** U6's acceptance pass found the M4 wiring (U4b/U4b-2) was correctly worded but didn't
  survive contact with a real dispatched agent's own judgment calls — all three live-tested
  dispatches failed a different way (format, skip, silence). Design intent
  (`docs/plans/cpg-agent-adoption.md` §2.3, §3) unchanged: still agent judgment on staleness
  threshold, still no self-triggered rebuild, still a suggestion not a hard rule about *when*
  something counts as stale — only the sequencing and the anchoring got tightened.
- **Plan items:** none new; closes U7.
- **Same-day addendum (U8 diff-gate follow-up):** `analyst`'s U8 diff gate
  (`docs/reviews/cpg-agent-adoption.md`, Pass 3 — approve with suggestions, zero blockers)
  flagged two minors and a nit against this same freshness sentence: (a) `frontend-engineer.md`
  was missing the "tool call/" qualifier the other five files carried, undercutting the U7
  ledger row's and commit message's "identically" claim; (b) the trailing "this is not a
  separate, optional judgment call" had an ambiguous pronoun referent — a literal reading could
  bind "this" to the cross-verification *decision* rather than the freshness *query itself*,
  exactly the room DEF-2's `architect` dispatch used to reason past a softer version of this
  sentence; (c) nit — "query the freshness check" mismatched a reference-doc noun with a query
  verb, when the actual queried object is the `:CpgBuildInfo` marker (the report's own
  recommendation said "marker"). Fixed all three: the sentence now reads "…query the freshness
  marker (per `skills/cpg-analysis/references/freshness.md`) in that same tool call/step, before
  you decide whether the CPG's answer needs further cross-verification — running the freshness
  check itself is not optional, and skipping it in favor of a substitute check (e.g. grep
  agreement) doesn't satisfy this." — byte-identical across all six files now. The `CPG:`-line
  wording from the original U7 pass was untouched (U8 raised no finding against it).

## 2026-08-16 — M4 cpg-agent-adoption: wired as a new `cpg-analysis` consumer

- **What:** Three edits per `docs/plans/cpg-agent-adoption.md` §2.4/§6 step 3 (`cobb`-owned
  design, U4b implementation unit). **Description:** added "With a loaded Joern CPG, uses the
  `cpg-analysis` skill for impact analysis before changing a function — what calls it, what else
  would break — instead of grepping by hand." **Step 1 "Orient":** added a check for a relevant
  CPG (first-guess `cpg_<component>` naming per `skills/cpg-analysis/SKILL.md` §1) bundled with
  the freshness check (`skills/cpg-analysis/references/freshness.md`) in the same step, noting
  the result in the report and surfacing a refresh suggestion — never a silent rebuild — if
  stale. **Step 5 "Verify and report":** added the `CPG:` evidence-trail line (plan §3 — `used
  <graph> — <clause>` / `considered, not relevant — <clause>` / `not applicable — <clause>`).
- **Why:** M4 widens the `cpg-analysis` roster from three consumers (`analyst`, `architect`,
  `qa-engineer`) to six — `coder` is a new consumer because its "what calls this / what would
  break" question before changing a function is exactly the impact-analysis recipe's target, and
  both live CPGs (`cpg_falkorchat`, `cpg_salesperson`) are Python codebases `coder` already works
  in. Plan §1's `coder` row has the full roster reasoning.
- **Plan items:** none (design-driven, not backlog-driven).
- **Addendum (same day):** the description clause initially shipped in the conditional "With a
  loaded Joern CPG, uses…" framing (carried over from a state-recovery instruction that locked it
  as already-landed). The coordinator caught that this left `coder`/`tdd-engineer` as the only two
  of the six wired agents not on plan §2.1's mandated default-orientation framing — the sibling
  unit's `analyst`/`architect`/`qa-engineer`/`frontend-engineer` edits all use "Checks whether a
  relevant CPG exists as part of its normal orientation and, when one does, uses…". Reworded to
  match: "Checks whether a relevant CPG exists as part of its normal orientation and, when one
  does, uses the `cpg-analysis` skill for impact analysis before changing a function — what calls
  it, what else would break — instead of grepping by hand." Body-prompt and evidence-trail
  additions were already correct and untouched.

## 2026-08-11 — Inbox distillation, corrected: 8 entries routed (5 promoted, 1 discarded as redundant, 2 promoted late after an `analyst` review caught them missing)

- **What:** `cobb` processed all 8 entries in `coder/kaizen/inbox.md` (§5), triggered by a
  stakeholder report of context blowouts and teco's explicit request to sweep this inbox. **This
  entry replaces a first version that mis-credited two entries to `coder` that actually came from
  `analyst`'s and `architect`'s inboxes** (the `urllib` timeout taxonomy and the LM Studio `/v1`
  200-envelope quirk — both correctly logged in *those* agents' own history entries) and, as a
  result, left two real `coder` entries with no logged disposition at all. Caught by `analyst`'s
  independent review (`docs/reviews/kaizen-distillation-2026-08.md`, B-1).
- **The 8 real entries and their dispositions:**
  1. FastAPI `response_model_exclude_unset=True` omit-vs-null — **discarded**, already fully
     covered by `python-web-quirks.md`'s pre-existing "nested models" entry.
  2. A `pytest --collect-only` baseline moving under you mid-run (concurrent agent on the same
     tree) — **promoted**: `coder.md` step 5 ("Verify and report") now says to report the
     *attributed* delta (own diff's contribution) alongside entry→exit counts when the tree may be
     shared.
  3. A green pytest exit code isn't evidence an integration suite ran; read the skip count —
     **promoted**: `coder.md` step 5 now carries the same skip/deselected-count clause
     `tdd-engineer.md`'s "Verify honestly" step already had — this was a live asymmetry (the agent
     that filed the learning didn't have it yet).
  4. FastMCP/`mcp` stdio EOF-response-loss — **promoted**, merged with `devops`'s 2026-07-26
     refinement (which corrects this entry's "always the last reply" framing to "a race, can drop
     more than one") → `claude/cobb/TESTING.md` (new gotcha subsection).
  5. FalkorDB no string-repetition operator — **promoted** → `claude/graph-dba/falkordb-quirks.md`
     (this is the only agent/history pair that owns this promotion; an earlier version of
     `graph-dba`'s own history entry also claimed it and has been corrected).
  6. `audit-team.sh`'s `grep -c FAIL` overcount-by-one — **promoted** → `skills/agent-maintenance/
     SKILL.md` §4.
  7. `monkeypatch.setenv` no-op against an import-time-frozen module constant — **promoted** →
     `skills/python-web-quirks/SKILL.md` (general Python fact; the falkor-chat-specific instance
     was already independently captured in `falkor-chat/docs/DESIGN.md` §14.7).
  8. Function-local deferred-import monkeypatch timing — **promoted** → same skill file.
- **On coder/tdd-engineer convergence (flagged as a judgment call in the review's open questions):**
  made the call to converge on suite-reporting discipline specifically — both are implementers
  whose "done" claim rests on the same suite, and a skip-count blind spot is exactly the kind of
  gap that should not differ by which implementer happened to touch the code. Did **not** merge
  their broader disciplines (TDD-cycle narration, the attributed-delta clause's fuller framing,
  etc.) — those reflect genuinely different working styles (red-green-refactor vs. plan-execution)
  that shouldn't converge just because one prompt happens to be shorter.
- **Verified:** `bash claude/scripts/audit-team.sh` clean. No personal identifiers introduced.
- **Docs touched:** `claude/coder/{coder.md,kaizen/{history,inbox}.md}` ·
  `skills/python-web-quirks/SKILL.md` · `claude/cobb/TESTING.md` ·
  `claude/graph-dba/{falkordb-quirks.md,kaizen/history.md}` · `skills/agent-maintenance/SKILL.md`.

## 2026-08-09 — Description gained a `python-web-quirks` skill routing clause
- **What:** Frontmatter `description` gained one clause: in a Python web/async codebase, the
  agent consults the new `skills/python-web-quirks/SKILL.md` for asyncio/FastAPI/Starlette/
  pydantic gotchas — mirroring how `cpg-analysis` is wired into `analyst`/`architect`. No body
  change; skills are progressively disclosed and self-describe.
- **Why:** `python-web-quirks` was created distilling three general Python/web-framework facts
  from `analyst`'s learnings inbox (asyncio `create_task` GC-safety, `BackgroundTasks` vs.
  `threading.Thread` concurrency bounds, FastAPI/pydantic `exclude_unset` on nested models).
  Stakeholder wired it to `coder`/`tdd-engineer`/`architect`/`analyst` at minimum since all four
  plausibly implement or review Python web/async code. See `claude/analyst/kaizen/history.md`
  (2026-08-09) for the full distillation record and `skills/README.md` for the catalog entry.
- **Plan items:** none.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 547 → 384 chars (-29%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (coder↔tdd-engineer, coder↔frontend-engineer) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Learnings inbox distilled (first pass; 6 entries, inbox cleared)
- **What:** Ran the `agent-maintenance` §5 distillation over `kaizen/inbox.md` — 6 entries accumulated
  2026-07-16 → 2026-07-21, all falkor-chat environment facts from K-022/K-024 work, never before
  distilled. All 6 verified against the live repo; **none routed to the coder's prompt** (every one is a
  fact about *falkor-chat*, not about how the coder should behave), so the prompt is unchanged.
  Routing:
  - **Already documented → discarded (3).** (1) *Published defs are immutable per version; re-seeding
    an edited prompt silently keeps the old config, and `reference`/`ws:<id>` go stale independently* —
    now covered exhaustively by the `seed_workflows.sh` row in `falkor-chat/AGENTS.md`, consistent with
    `docs/DESIGN.md` §144/§147/§544. Re-verified the mechanism still holds (`repository.py`
    `_PUBLISH_CYPHER` is still `MERGE … ON CREATE SET`). (5) *pytest wipes `reference` at setup, so the
    last test's defs survive and masquerade as a seeded def* — same AGENTS.md row already spells out the
    false `already present — no-op` signal. (3) *the `_drive_loop` byte-identity lock's quoted byte count
    is wrong (SHA `71055f756280` right, 2844 ≠ actual 2860)* — already an open **Doc-drift** item at
    `falkor-chat/docs/BACKLOG.md:399` with the same diagnosis, and the surviving plan docs
    (`docs/archive/plans/m3-process-flow.md:396`, `…-coordination.md:38`) now say "verify by SHA only;
    every byte count quoted is wrong". Nothing to add.
  - **Promoted to project docs (3).** (4) *zero-transition publish → bare `IndexError` on a half-written
    def* — the **service-layer guard has since landed** (`services.py` `_validate_def_spec`, K-024 U4b
    O-6, rejects with `WorkflowDefSpecError` → 400), but `docs/QUERIES.md` §11.1 documented neither the
    trap nor why this query is deliberately unguarded while §4's mention block is; added a ⚠️ note there
    stating the collapse mechanism, the unrepairable-poisoning consequence, where the guard actually
    lives, and that any new caller bypassing the service layer must re-validate. (2) *pytest is
    destructive to `reference`, and a green pytest line hides ~half the suite skipping when FalkorDB is
    down* — the `seed_workflows.sh` row carried this from the seeding side, but the **M1 server pytest
    bullet**, where someone running pytest actually looks, did not; added two bullets to
    `falkor-chat/AGENTS.md` (destructive-at-setup + re-seed, and read the skip count — verified
    `conftest.py:54` `pytest.skip` guard). (6) *`ruff check .` baseline is already red* — verified still
    red today, exactly one pre-existing `I001` at `falkorchat/llm.py:13`; documented as a known baseline
    in the same AGENTS.md section so an implementer doesn't misread it as their own regression.
- **Why:** §5 — capture is cheap and unreviewed, promotion is curated; facts about *a project* belong in
  project docs where every agent sees them, never hoarded in one agent's private files. Half the inbox
  had already been absorbed into project docs by the K-024/M3-close doc sweeps, which is the loop
  working as intended.
- **Deliberately NOT done (flagged as follow-up, not landed silently):** the one-line `llm.py` import
  reorder that would make ruff green, and any repository-level guard on `_PUBLISH_CYPHER`. Both are code
  changes to falkor-chat, outside a distillation pass's remit.
- **Plan items:** none closed; K-002 evidence noted in `plan.md`.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so the user kept having to re-grant write permission to `coder` across sessions even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`. `acceptEdits` auto-accepts file edits and common filesystem commands for paths in the working directory/`additionalDirectories`, independent of session-level grants.
- **Why:** User asked why they always had to give write permission to `coder`; root cause confirmed against current Claude Code docs (`sub-agents.md`, `permissions.md`) rather than assumed.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 822 to 535 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Description: route-away clause to the new `frontend-engineer`
- **What:** the `description`'s routing tail gained a second route-away rule: UI-heavy front-end work (components, styling, accessibility, client-side state, a Streamlit screen) → `frontend-engineer`. Catalog rows (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`) updated to match; the pair was added to `scripts/audit-team.sh` `BOUNDARY_PAIRS`.
- **Why:** a UI-depth specialist implementer (`frontend-engineer`) joined the team; boundary reciprocity requires the adjacent generalist implementer to name it so routers see the contract from both sides.
- **Plan items:** none (driven by frontend-engineer's creation).

## 2026-07-09 — K-001 ✅: efficiency-based routing boundary with `tdd-engineer` (de-personalized)
- **What:** Rewrote the `description`'s routing tail. Was "for strict test-first discipline, prefer tdd-engineer" — a subjective tiebreaker; now routes by **task shape / efficiency**: a detailed plan/spec ready to execute → `coder` (tests alongside); a bug fix, safety-net refactor, test-focused work, or clear-contract feature → `tdd-engineer`. Made symmetric: `tdd-engineer`'s description (which previously shadowed coder's trigger — "whenever the user asks to implement a feature" — and never pointed back) now carries the mirror rule. Synced everywhere the rule is repeated: teco's roster (the "(this user prefers TDD — lean toward tdd-engineer)" note removed), `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`, and `cobb/TESTING.md`'s rationale column.
- **Why:** User ruling on the overlap review: use `tdd-engineer` only where test-first is genuinely the efficient path, and when a detailed plan already exists the most efficient implementer wins — plus, personal-preference notes ("this user prefers TDD") don't belong in agent prompts; the user's standing preferences are **quality and efficiency**, encoded as objective routing rules.
- **Plan items:** K-001 ✅ (closed — the descriptions no longer collide; the tiebreaker is objective at routing time).

## 2026-07-09 — Subagent-awareness on the two "ask" spots (teco interface review follow-up)
- **What:** Step 2 (baseline) "ask before installing or mutating the environment" and the "Ask before destructive or environment-changing actions" guardrail now both say what to do when running as a subagent (e.g. delegated by `teco`): return the blocker / request as the result instead of trying to ask mid-run — subagents can't ask. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** Sweep after the 2026-07-09 teco interface review found the "ask" phrasing assumed an interactive session across several delegates (same fix applied to tdd-engineer, qa-engineer, graph-dba the same day; architect already handled it via questions-as-deliverable).
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-07-08 — Architect handoff arrives as a plan-document path
- **What:** Step 1 (Orient) updated: an `architect` handoff is now a plan **document** at `<component>/docs/plans/<slug>.md` — the coder gets the path in its brief, reads the file itself, and treats it as source of truth (gaps filled by reading code, not guessing). Synced with the architect's same-day change making the plan doc its default deliverable and teco's switch to path-based handoff.
- **Why:** The previous flow had the orchestrator paste the plan into the brief — lossy and unreviewable. Reading the artifact directly makes the handoff lossless regardless of who invokes the coder.
- **Plan items:** advances K-002 (transport fixed; live validation run still pending).

## 2026-06-20 — Dropped "senior" framing
- **What:** Removed "senior" from the `description` ("Senior software engineer" → "Software engineer") and the body opener ("You are a senior software engineer who builds" → "You are a software engineer who builds"). Mirrored in the catalog entries (`claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md`).
- **Why:** User raised the overconfidence concern with seniority framing; persona-prompting evidence (e.g. Zheng et al. 2024) shows role labels are weak-to-neutral for correctness while authority framing can hurt calibration. Quality is carried by the concrete process + guardrails ("don't fake green," "report only what you ran"), not the title. Chose the most conservative option (drop the word). Goes further than the 2026-06-05 precedent that kept "Senior" as an altitude signal — architect/coder now differ from the rest of the collection until harmonized (flagged to user).
- **Plan items:** —

## 2026-06-20 — Created
- **What:** Created the `coder` subagent (`coder/coder.md`, `model: opus`). Senior implementer that executes an approved plan/spec end-to-end: orients on the plan + code, establishes a green baseline (with explicit handling for already-red and can't-run-here cases), implements in small reversible increments, tests alongside, refactors under green, and reports only results it actually ran. Inherits all tools (no `tools` key), following the `tdd-engineer` K-003 precedent of keeping the implementer flexible.
- **Why:** User asked for two complementary Claude Code subagents, "the architect" and "the coder," with an architect→coder handoff. The `coder` is the implementation half.
- **Plan items:** seeded K-001..K-002.

## Decisions recorded at creation
- **Distinction from `tdd-engineer`:** both implement, but `tdd-engineer` is *strictly* test-first (red→green→refactor as the defining discipline). `coder` is plan-driven and pragmatic: it tests behavior thoroughly and never ships untested code, but doesn't mandate writing the failing test first unless the project requires it. The `description` explicitly routes strict-TDD requests to `tdd-engineer` so auto-delegation doesn't collide. Revisit if the two over-trigger on the same prompts.
- **Why inherit all tools:** an implementer needs Read/Write/Edit/Bash plus the ability to fetch docs and delegate; mirrors the deliberate `tdd-engineer` choice (K-003 there). Revisit only if broad access causes surprise.
