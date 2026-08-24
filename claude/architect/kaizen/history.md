# Kaizen — Change History: architect

> Dated log of actual changes to the `architect` agent. Most recent first.

## 2026-08-23 — Freshness-clause grammar fix (Stage B wave 2 micro-shape)
- **What:** "a `teco`-issued brief that states the graph's freshness, take it as given" → "when a `teco`-issued brief states the graph's freshness, take it as given" — closing the hanging-topic construction cobb's wave-1 lint flagged as minor; applied uniformly across all files carrying the clause. No rule change; both branches intact.

## 2026-08-23 — Prompt compression: narratives → history pointers (waste-reduction pilot, Stage A)

- **What:** Compressed `architect.md` 1,738 → 1,547 words per `claude/docs/plans/prompt-waste-reduction.md` §3 (Stage A pilot). Every behavioral rule and mechanism preserved; only class-5/6/7 material (narratives, provenance retellings, duplicate restatements) moved or tightened. Frontmatter, hooks, and all externally-cited anchors untouched: the three verbatim `CPG:` forms, `git add`/`git commit` + "delegated subagent" (audit-team.sh check 8), "never edit your own agent definition" (cited by `docs/plans/generic-cypher-mcp2-coordination.md` P2-B2), and the step name "Investigate the codebase first" (cited by `docs/test-reports/cpg-agent-adoption-report.md:81`).
- **Moved/removed clauses (gate b — each verified already recorded before removal):**
  1. Learning-capture inbox-replacement history ("This replaces the earlier `kaizen/inbox.md`-append convention… fully distilled and removed 2026-08-21") → this file, 2026-08-21 inbox-deletion entry.
  2. Commit-grant provenance retelling ("same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`") → compressed to the dated pointer; full story in this file's 2026-08-21 commit-grant entry.
  3. CPG-freshness decision retelling ("that responsibility is centralized… when a `teco`-issued brief states…") → rule + `(2026-08-19)` pointer kept; full story in this file's 2026-08-19 freshness-centralization entry.
  4. Class-7 dedups: isolated-context restatement in "Handoff" (canonical statement stays in the intro); `PRODUCED`/`:Agent`-edge description in the learning-capture intro (the Cypher template itself shows it); "(use `Write` to create it, `Edit` to amend it in place)"; "not paraphrased, not dropped" (covered by "written verbatim"); minor phrasing tightenings throughout.
- **Rule inventory (gate a):** 32 class-1/2 clauses inventoried pre-edit; all 32 mapped to surviving locations post-edit — zero rule loss. Notables checked one-by-one: the six-section plan skeleton, the CPG-line three-form contract + `not applicable` disambiguation, ambiguity→open-questions-as-deliverable, tico-requirements-first, Explore delegation, data-scientist ML delegation, plan-doc default + path convention + header block, hook-gate-by-pattern guardrail, compress-by-pointer guardrail (both halves), interactive-commit grant + never-list + delegated-subagent carve-out, learning-capture template + skip-rules.
- **Why:** Stakeholder-directed waste reduction — prompts carry rules and one-clause whys; stories live here. Gates: audit-team.sh PASS; cobb §7 lint (gate c) returned **pass with notes, zero rule loss** — its one minor (a telegraphically-clipped CPG-freshness clause that also widened "teco-issued brief" to "a brief") was applied same-session via cobb's suggested rewrite, plus its pre-existing-nit fix (the bare `(2026-08-19)` pointer gained the `kaizen/history.md` anchor). **Same-day calibration ruling (stakeholder, reviewing this pilot):** provenance never earns prompt space at all — no dates, no "stakeholder decision" markers, no `kaizen/history.md` pointers; a rule's non-negotiability is expressed by stating it absolutely, and this file is the standing greppable home for where each rule came from. Both provenance parentheticals (commit-grant, CPG-freshness) accordingly deleted outright; doctrine updated (`prompt-waste-reduction.md` v3, §3 classes 5/6). Normative citations that a rule *uses* (the `CPG:` forms' spec path, the root-`AGENTS.md` header block) stay. Final: 1,545 words. Precedent: this file's own 2026-08-19 de-dup entries — same method, now doctrine.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** The Bash guardrail's "investigation only" bullet now also grants: when running
  interactively (`claude --agent architect`, a human present turn-by-turn — not a delegated
  subagent), may `git add`/`git commit` its own plan/design document from the session, by
  explicit path, never bulk-staged/pushed/reset/rebased/amended; the grant does not apply when
  spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Context & findings"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered). It had been frozen — never written to — since the 2026-08-20 graph migration (see that date's entry below, which already confirms this file's own pre-migration content was imported into the graph verbatim at the time).
- **Why:** user-directed team-wide cleanup, "no point keeping a file already in git history." Before deleting any of the 12 agents' frozen inboxes, `cobb` live-confirmed `kaizen_team` — the single shared graph every agent's raw capture has routed through since the 2026-08-20 consolidation — holds **zero** entries for any agent: every raw capture any agent ever wrote there (including this agent's own distillation, immediately above) has since been fully distilled and cleared. Combined with the migration-time import guarantee above, nothing in this file was ever a live, undistilled input to anything — it was a pure redundant backup copy. Same session also completed `G1`'s last 2 of 12 `kaizen_<agent>` graph-key retirements (`kaizen_analyst`/`kaizen_teco`, executed by `graph-dba`), closing `docs/plans/generic-cypher-mcp2-coordination.md`'s one remaining open item.
- **Verified:** live `mcp__cypher__query` count against `kaizen_team` (0 entries) before any deletion.
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` distillation: 4 entries — 1 promoted to Guardrails, 3 discarded (2 already-documented elsewhere, 1 already-actioned)

- **What:** `cobb` processed all 4 `author:'architect'` entries in the shared `kaizen_team` graph
  (agent-maintenance skill §5), all from the same 2026-08-20 episode (authoring/revising
  `docs/plans/generic-cypher-mcp2.md` through several plan-gate rounds).
  - **Promoted (1) → new Guardrails bullet:** `c4a8d2f1` — two plan-revision-rigor lessons from
    that episode's Pass-3 plan-gate majors: compressing a plan section to "see prior version,
    unchanged" breaks any other section that cites the compressed content literally (an
    isolated-context implementer needs the actual Cypher block/command, not a git-history
    pointer); and a blanket find-and-replace across N near-identical files/sections needs the N
    verified textually identical first (a tense mismatch among them means one substitution is
    wrong for some).
  - **Discarded (3):**
    - `a1e3c9d4` — already fully documented in `cypher-mcp/README.md` (the "Read-only" and "Graph
      discovery" sections: a graph materializes only on write, and querying an unknown graph name
      returns the full list of loaded graphs in the error text).
    - `b7f2e1a8` — same episode, same underlying fact (a self-edit "precedent" the plan claimed
      for `architect`/`graph-dba` didn't actually exist, verified against `git show --stat`) as
      `analyst`'s own kaizen capture of this same 2026-08-20 review round — already promoted, with
      the identical origin note, into `claude/analyst/review-techniques.md` ("Ground truth for
      'may an agent edit its own definition?'"). Promoting it again here would duplicate, not add.
    - `f3a7c2e1` — the action it describes (add a dated Version-5 revision note to
      `docs/plans/generic-cypher-mcp2.md` recording the header-retarget instruction's real-world
      enforcement override) is **already done**: the document's header already reads `Version: 5`
      and carries a "Revision note — 2026-08-20 (Version 5)" section. Nothing left to act on.
  - **Docs touched:** `claude/architect/{architect.md,kaizen/history.md}`.
- **Why:** User-requested distillation pass, continuing the oldest-first queue (data-scientist,
  then architect).
- **Plan items:** none opened — the one promotable entry landed directly; the other three needed
  no forward-looking action.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_architect`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_architect` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query. The trailing "Your write guard allows exactly this inbox path" clause was dropped — the
  write guard gates `Write`/`Edit`, not the `mcp__cypher__query` MCP tool, so it no longer applies
  to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Freshness-check clause removed (centralized on teco)
- **What:** Dropped the CPG freshness-check paragraph from the CPG-orientation step — still checks whether a relevant CPG exists and uses it via `cpg-analysis`, but no longer queries the `:CpgBuildInfo` freshness marker itself. That responsibility is now `teco`'s alone (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`); running standalone (no `teco`-issued brief), staleness is simply not checked.
- **Why:** User-directed prompt-verbosity reduction: the freshness paragraph was ~130 words, byte-identical across six agent files. Stakeholder chose full centralization over a per-agent dedup, accepting the standalone-run capability loss.
- **Plan items:** —

## 2026-08-16 — Inbox distillation (on-demand, cobb): 2 of 3 entries promoted to `cpg-analysis` skill, 1 already fixed
- **What:** `cobb` processed the three same-day inbox entries at the stakeholder's explicit
  request (not a periodic sweep). **Verified each against live state first:**
  1. `mcp__cypher__query(graph="cpg_falkorchat", "MATCH (n) RETURN count(n)")` → 166,789 nodes (graph
     exists); `graph="cpg_falkor-chat"` → not found, confirming the entry's evidence. **But**
     `skills/cpg-analysis/SKILL.md` §1 already reads "the component-directory name with hyphens
     stripped (`falkor-chat` → `cpg_falkorchat`, …)" — that exact fix landed *earlier the same day*
     in commit `50f9aaa` (U4b-1..5, a different, already-merged unit of the `cpg-agent-adoption`
     milestone), before this distillation pass ran. **Disposition: discard, no skill edit** — the
     architect's dispatch hit the pre-fix skill text mid-flight; the fact it flagged is now fully
     covered by wording that shipped independently the same day, verified by re-reading the file.
  2. Live-reran the batch-`IN` query shape (`MATCH (m:METHOD) WHERE m.NAME IN [...]`) against
     `cpg_falkorchat` — reproduces as described. Not previously documented as a general technique
     (`IN [...]` appeared only inside `code-review.md`'s fixed sink list, never as a stated
     lookup-efficiency habit). **Promoted** to `skills/cpg-analysis/SKILL.md` §3, as a new idiom
     bullet after "Anchor a target method."
  3. Live-reran both the `FILENAME`-scoped and unscoped variants of the caller query against
     `cpg_falkorchat` — both execute as described (shapes match; exact row counts not
     re-verified against the moving `executor.py`/`test_executor_agent.py` source, consistent
     with this skill's own "Verified figures are dated evidence, not targets" policy). Not
     previously named as a technique (`impact-analysis.md` Q1 only documented filtering tests
     *out* via `STARTS WITH 'tests/'`, not the two-pass scoped-then-unscoped pattern for telling
     the design answer apart from the test-surface answer). **Promoted** to
     `skills/cpg-analysis/references/impact-analysis.md`, as a new callout + verified example
     after Q1's existing paragraph.
- **Why:** Stakeholder-requested, scoped to exactly these three entries (not a full inbox sweep).
  Standing distillation duty (agent-maintenance skill §5).
- **Docs touched:** `skills/cpg-analysis/SKILL.md`, `skills/cpg-analysis/references/impact-analysis.md`,
  `claude/architect/kaizen/inbox.md` (cleared to the standard empty placeholder — no other entries
  were pending).
- **Plan items:** —

## 2026-08-16 — U7 fix round: freshness-check sequencing hardened, `CPG:` line anchored (DEF-1/DEF-2/DEF-3)
- **What:** Two wording tightenings per `docs/plans/cpg-agent-adoption-coordination.md` unit U7,
  following U6's `qa-engineer` live-dispatch acceptance pass
  (`docs/test-reports/cpg-agent-adoption-report.md`). (1) The freshness-check sentence now reads
  "query the freshness check … in that same tool call/step, before deciding whether the result
  needs further cross-verification — this is not a separate, optional judgment call" (previously
  "also run the freshness check … as part of that same step") — closes DEF-2 (`architect`, this
  agent, used the CPG and correctly emitted the `CPG:` line, but explicitly declined to run the
  freshness check, reasoning that a grep/CPG agreement substitute was sufficient — it isn't,
  since a stale-but-coincidentally-still-correct CPG wouldn't produce a mismatch either). (2) The
  `CPG:` line instruction now reads "written verbatim and required in all three cases including
  when the CPG isn't relevant — not paraphrased, not dropped" — closes DEF-1 (`coder`) and DEF-3
  (`tdd-engineer`), both format/omission failures on this same wiring pattern. Applied identically
  (phrasing pattern, not restructuring) across all six wired agents; only
  `coder`/`architect`/`tdd-engineer` were live-tested, but the near-verbatim wiring pattern means
  the same gap plausibly existed in `analyst`/`qa-engineer`/`frontend-engineer` too.
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
  exactly the room this agent's own DEF-2 dispatch used to reason past a softer version of this
  sentence; (c) nit — "query the freshness check" mismatched a reference-doc noun with a query
  verb, when the actual queried object is the `:CpgBuildInfo` marker (the report's own
  recommendation said "marker"). Fixed all three: the sentence now reads "…query the freshness
  marker (per `skills/cpg-analysis/references/freshness.md`) in that same tool call/step, before
  you decide whether the CPG's answer needs further cross-verification — running the freshness
  check itself is not optional, and skipping it in favor of a substitute check (e.g. grep
  agreement) doesn't satisfy this." — byte-identical across all six files now. The `CPG:`-line
  wording from the original U7 pass was untouched (U8 raised no finding against it).

## 2026-08-16 — M4 cpg-agent-adoption: discovery wording defaulted, freshness-check bundled, evidence-trail line added
- **What:** Three edits per `docs/plans/cpg-agent-adoption.md` §2.4/§3 (U4b). (1) Frontmatter
  `description` reworded from "With a loaded Joern CPG, uses the `cpg-analysis` skill for
  call-graph impact analysis" to "Checks whether a relevant CPG exists as part of its normal
  orientation and, when one does, uses the `cpg-analysis` skill for call-graph impact analysis" —
  conditional → default-orientation framing. (2) "How you work" step 2 ("Investigate the codebase
  first") gained a sentence: check whether a relevant CPG exists (first guess `cpg_<component>`,
  per `skills/cpg-analysis/SKILL.md` §1), and when one is found and used, also run the freshness
  check (`skills/cpg-analysis/references/freshness.md`) as part of the same step, noting what it
  says in Context & findings and surfacing a refresh suggestion — not a silent rebuild — if it
  looks stale. (3) The plan skeleton's item 2 ("Context & findings") gained the one-line `CPG:`
  evidence-trail convention (`CPG: used <graph> — <clause>` / `CPG: considered, not relevant —
  <clause>` / `CPG: not applicable — <clause>`).
- **Why:** M4 (`cpg-agent-adoption`) widens CPG discovery from a conditional check to a default
  orientation step across the three already-wired consumers (`analyst`, `architect`,
  `qa-engineer`), bundles the freshness recipe into that same step (FR-6's surfacing half), and
  adds a spot-checkable `CPG:` evidence trail (AC-2). Per `docs/plans/cpg-agent-adoption.md`
  §2.1-2.3, §3, §6 step 2.
- **Plan items:** none.

## 2026-08-11 — Inbox distillation: 1 entry, merged with analyst's duplicate finding — no prompt change

- **What:** `cobb` processed the single entry in `architect/kaizen/inbox.md` (§5) — the LM Studio
  `/v1` missing-prefix / 200-envelope quirk, independently captured by both `architect` and
  `analyst` during the same K-042 work.
- **Disposition:** the falkor-chat-specific half was already fully documented in
  `falkor-chat/docs/DESIGN.md` §14.8 before this pass; the general, reusable half (the 200+envelope
  shape, and the `urlparse` schemeless-base-URL trap) was promoted once, from `analyst`'s copy of
  the same finding, into `skills/python-web-quirks/SKILL.md` — no separate action needed here.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/architect/kaizen/{history,inbox}.md`.

## 2026-08-09 — Held entries 15/16 promoted: consolidated Kiro-facts edit landed
`cobb` closed out the two held-for-consolidated-follow-up entries (exact-CWD-only local-agent
discovery, no upward walk; and the `mcpServers` remote-entry schema using `url` with no `type`
field): the race window against `analyst`'s held entry 28 (same target file) is over, so both
facts were re-verified (now against `kiro-cli 2.16.2`, up from `2.14.1` at original
live-verification — both held) and written into `skills/agent-standards/kiro.md` — discovery
fact into the CLI custom-agents `Location` bullet, `mcpServers` schema fact into the
`mcpServers` config-key bullet (which also corrected the doc's prior "each needs `command`"
phrasing to cover the `url`-keyed remote case). `inbox.md` entries 15 and 16 cleared; `inbox.md`
is back to the standard empty placeholder.

## 2026-08-09 — Description gained a `python-web-quirks` skill routing clause
- **What:** Frontmatter `description` gained one clause, appended to the existing `cpg-analysis`
  routing sentence: in a Python web/async codebase, the agent also uses the new
  `skills/python-web-quirks/SKILL.md` for asyncio/FastAPI/Starlette/pydantic gotchas. No body
  change.
- **Why:** `python-web-quirks` was created distilling three general Python/web-framework facts
  from `analyst`'s learnings inbox. Stakeholder wired it to `coder`/`tdd-engineer`/`architect`/
  `analyst` at minimum, mirroring the existing `cpg-analysis` wiring pattern. See
  `claude/analyst/kaizen/history.md` (2026-08-09) for the full distillation record and
  `skills/README.md` for the catalog entry.
- **Plan items:** none.

## 2026-08-09 — Inbox entry #1 routed to a separate architect-persona task, not edited here
- **What:** Entry #1 (2026-07-19, `_drive_loop` byte-identity-lock SHA-reproduction command) is
  disposed as: **promoted to `falkor-chat/docs/DESIGN.md`** (the doc-drift note near §6.2, where
  the lock's SHA is already quoted with corrected byte-count guidance but no robust reproduction
  command) — via a dedicated architect-persona task, teco-coordinated, running in parallel, **not
  edited by cobb**. Content and destination were fully decided in the 2026-08-09 read-only
  proposal pass; only execution was delegated. Cleared from `kaizen/inbox.md`.
- **Why:** Role-responsibility principle — this is `falkor-chat/` project-doc content within a
  component `architect` owns the plans/design for; a team-maintainer session (cobb) editing it
  directly would cross a boundary that belongs to the producing discipline, not the team
  maintainer. Distillation duty (agent-maintenance skill §5), teco-coordinated.
- **Plan items:** —

## 2026-08-09 — Guardrails: verify what a hook pattern-matches before treating its prompt as a gate
- **What:** Added one Guardrails bullet: *"Verify hook gates by pattern, not intent. When a plan
  step's verification depends on a `PreToolUse` hook firing, check what the hook actually
  pattern-matches (the command text, not the intent) before treating the prompt as a gate — a
  destructive operation wrapped inside a script can bypass a hook that only greps literal command
  strings."* No other prompt/frontmatter/catalog change.
- **Why:** Distilled from inbox entry #9 (2026-07-25, `guard-destructive-ops.sh` matches the Bash
  *command string*, so `skills/joern-cpg/scripts/pipeline.sh --reset` bypassed the destructive-ops
  approval prompt entirely — the token the guard greps for never appears in the command text the
  hook sees). Stakeholder judged this **high-recurrence** (hook-gated approval prompts are this
  team's central safety mechanism — five doc-scoped write guards plus the destructive-ops guards
  across `devops`/`graph-dba`/`qa-engineer`) and **already-proven-costly** (the bypass this entry
  describes actually happened and shipped; the fix landed as C-311 on 2026-08-08, roughly two
  weeks after the gap was knowable). Cleared from inbox alongside the rest of the distillation pass
  (see the batched entry below) — its *specific* finding (the `pipeline.sh --reset` bypass) is
  independently discarded there as already fixed; this entry is the promotion of its *general*
  habit.
- **Plan items:** —

## 2026-08-09 — Inbox distillation: 13 entries discarded (already fully covered elsewhere)
- **What:** Processed 13 inbox entries against current repo state (agent-maintenance skill §5):
  2026-07-24 `MERGE … ON CREATE SET` create-only-for-properties/additive-for-structure;
  2026-07-24 `OPTIONAL MATCH` + `collect(DISTINCT …)` non-aggregated-field-is-a-grouping-key;
  2026-07-24 falkor-chat def-publish-has-no-graph-seam; 2026-07-24 FalkorDB silently ignores
  `EXPLAIN`/`PROFILE` inside `GRAPH.QUERY`; 2026-07-24 `tools:` allowlist makes new MCP tools
  invisible to `analyst`/`architect`; 2026-07-25 check-7-forbids-absolute-home-paths +
  `.mcp.json`'s portable `$CLAUDE_PROJECT_DIR` form; 2026-07-25 `teco`'s write guard can't own
  edits outside `docs/plans/`; 2026-07-25 `guard-destructive-ops.sh` command-string bypass
  (specific finding only — general habit promoted separately above); 2026-07-25 MCP output over
  threshold persists to disk, truncation notices belong at the head; 2026-07-25 an MCP server's
  `instructions=` is injected every session and is probe-verifiable; 2026-07-26 leading-slash
  markdown links aren't agent-followable; 2026-07-26 this repo cites paths as backticked strings,
  not markdown links, so a link-checker is nearly blind here; 2026-07-27 root `AGENTS.md` reaches
  subagents too via `CLAUDE.md`'s `@AGENTS.md` import.
  **Verified each still-true-but-superseded or already-promoted:** the MERGE-additive defect was
  closed outright by **K-034** (`falkor-chat/docs/HISTORY.md:60`, 2026-08-01) — re-publish now
  fails loudly (409) instead of silently minting parallel structure, so the trap the entry warned
  about no longer exists. The grouping-key hazard and the all-rows-read decision it drove are
  documented verbatim in `falkor-chat/docs/QUERIES.md:925-941` (the K-031 §11.2 callout). The
  def-publish-no-seam finding and its throwaway-`ws:`-probe workaround are recorded as executed in
  `falkor-chat/docs/HISTORY.md:903-919` (K-031 V-1). The `EXPLAIN`/`PROFILE` behavior is documented
  verbatim in `skills/cpg-analysis/SKILL.md:65-75`. The `tools:` allowlist gotcha is documented in
  `skills/agent-standards/claude-code.md:373-375`, and the concrete instance was closed the day
  after capture (this file, 2026-07-25 entry: `mcp__cypher__query` added to `architect`'s and
  `analyst`'s `tools:`). The check-7/`.mcp.json` portable-form fact is documented in
  `skills/agent-standards/claude-code.md:259-276` and is what the repo's actual `.mcp.json` uses
  today. The `pipeline.sh --reset` bypass is fixed (C-311, `claude/scripts/guard-destructive-ops.sh`,
  2026-08-08). The MCP output-limit/disk-persistence and server-`instructions=`-injection facts are
  both documented near-verbatim in `skills/agent-standards/claude-code.md:307-347`. The
  `teco`-write-guard-scope fact is stated directly in `claude/teco/teco.md:67`. The leading-slash
  and backticked-path-citation facts are exactly what root `AGENTS.md`'s citation convention now
  states — those two entries **became** the shipped convention. The subagent-reachability fact is
  documented in `skills/agent-standards/claude-code.md:141-142,175` and is what motivated this
  agent's own 2026-07-27 "header block from root `AGENTS.md`" prompt line (this file). In every
  case the promotion had already happened, dated the same day or within days of capture, as a
  byproduct of shipping the plan that surfaced the fact — only the inbox-clear step was
  outstanding. Entry #4's secondary "verify a live probe has a graph/tenancy seam before
  scheduling it" habit is **parked, not promoted** — recorded in `kaizen/plan.md`'s parking lot
  (judged narrow/single-occurrence; revisit on recurrence). All 13 cleared from
  `kaizen/inbox.md`; entries #1, #15, #16 handled separately (see the adjacent 2026-08-09 entries
  and `kaizen/inbox.md`, which still carries #15/#16 pending a consolidated `skills/agent-standards/kiro.md`
  follow-up).
- **Why:** Standing distillation duty (agent-maintenance skill §5), teco-coordinated pass over the
  full inbox (267 lines, 16 entries — the largest in the team).
- **Plan items:** —

## 2026-07-28 — Inbox entry distilled: `audit-team.sh` is not a usable bare pass/fail done-condition
- **What:** Processed the 2026-07-25 inbox entry *"`audit-team.sh` passes is an unusable plan
  done-condition"* (agent-maintenance skill §5). **Verified still true:** re-ran
  `claude/scripts/audit-team.sh` — check 7 still greps every tracked file in the repo for
  personal identifiers, so a plan step worded as a bare "assert it passes" is unsatisfiable
  the moment any unrelated leak exists anywhere in the repo. **Routed to project docs**, not
  this agent's always-loaded prompt (the fact only bites when a plan step references this one
  script — too narrow to pay for on every session): added a callout to
  `skills/agent-maintenance/SKILL.md` §4, right after the deterministic-half paragraph the
  entry was actually about, stating the fix (assert "no new FAIL line" against a captured
  before-state, not a bare pass). The entry's own personal-path fragment (`claude/joern/…`,
  `docs/plans/m2-cpg-analysis-skill.md:327`) was stale bookkeeping only — the `joern` agent it
  cited was retired 2026-07-28 (see `claude/graph-dba/kaizen/history.md`) and the four leak
  sources it listed were fixed the same day (this task) — so nothing else needed carrying
  forward. Entry removed from `claude/architect/kaizen/inbox.md`.
- **Why:** Standing distillation duty (agent-maintenance skill §5) — cobb processed this entry
  on the coordinator's explicit request, tied to the same-day joern-agent cleanup that had
  left `claude/joern/kaizen/inbox.md:19` (cited by this entry) a dangling path.
- **Plan items:** —

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Plans open with the canonical header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line added to *How you work* step 5, immediately after the plan-document convention: *"Open the document with the header block from root `AGENTS.md`."* No frontmatter, hook, `description` or catalog change.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6 makes a three-field header (`Status:` · `Owner:` · `Tracks:`) the lifecycle signal that replaces the milestone filename prefix and the move-to-`archive/` rule; a plan is the most-cited document kind in the repo, and `architect` is the agent that creates it, so the field is only ever present if this prompt asks for it. The line is a **pointer, not an inlined template** (v1.4 M20): root `AGENTS.md` reaches every agent through the root `CLAUDE.md` `@AGENTS.md` import, so the second hop costs nothing, while eight copies of a still-settling block would drift — §9.6 stays the one place the block is stated. The sentence is byte-identical across all six producing prompts on purpose; the convention's coverage check greps for it literally. `claude/README.md` row 9 re-checked — it cites the plan write path and the hook, not the document's internal structure; no edit needed.
- **Plan items:** none. Note for whoever picks up the parking-lot "self-review checklist before delivering a plan" idea: the header block is now the first thing that checklist should assert.

## 2026-07-25 — `tools:` allowlist gains `mcp__cypher__query` (M3 / C-304)
- **What:** Frontmatter `tools:` now ends `…, Agent, mcp__cypher__query`. `claude/README.md` row 9 updated to say the `cpg-analysis` skill reaches the graph through that MCP tool and why the allowlist entry is required. No body or `description` change — the impact-analysis CPG clause added on 2026-07-19 stays accurate.
- **Why:** M3 replaces the CPG read path with a single MCP tool, `mcp__cypher__query(graph, cypher)` (`docs/plans/cpg-query-access.md` S5). **`tools:` is an allowlist, not a hint** — an agent that declares one does not see MCP tools absent from it, so without this line the tool would have been invisible to `architect` (and `analyst`) while `qa-engineer`, which declares no allowlist, inherited it. `redis-cli GRAPH.QUERY` remains the documented fallback and is the only path under OpenCode/Kiro.
- **Verification note:** this is the *edit*; the live proof (a cold `architect` actually calling the tool) needs the server wired in S3 and is verified in S9, per the plan's m-4 split.
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 627 → 453 chars (-27%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (architect↔data-scientist) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change to `coder`/`tdd-engineer`/`frontend-engineer`. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe here specifically: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions` — modes can't loosen what a hook tightens. `architect`'s `guard-plan-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed plan-doc paths) keeps working exactly as before; only writes it would already let through silently — writes inside the allowed doc paths — stop re-prompting every session.
- **Plan items:** none.

## 2026-07-19 — CPG capability wired into the routing description (M2 / C-207)
- **What:** Frontmatter `description` gained one clause: for call-graph impact analysis over code with a loaded Joern CPG in FalkorDB, the architect uses the `cpg-analysis` skill (graph-dba-owned). `claude/README.md` catalog entry updated to match. No body change (skill is progressively disclosed).
- **Why:** M2 delivered the `cpg-analysis` skill; `architect` is a named consumer of the impact-analysis recipe (FR-10). C-207 makes the routing contract advertise it. Wired by cobb as part of Gate-2b (skill passed the standards vet).
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 761 to 498 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-plan-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/plans/*|*/docs/plans/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/architect/hooks/guard-plan-doc-writes.sh`) to `$HOME/.claude/agents/architect/hooks/guard-plan-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/architect` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — data-scientist boundary clause (description + delegate-the-method step)
- **What:** Frontmatter `description` now names the `data-scientist` as the supplier of a design's AI/ML/DS method (model/embedding selection, retrieval strategy, evaluation methodology, experiment design), and "How you work" step 4 (Decide) instructs delegating such method calls to it via the `Agent` tool — it returns a method note at `<component>/docs/plans/<slug>-ml.md` (or inline) that the plan folds in, rather than the architect guessing the method. Pair `architect:data-scientist` added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` (check 6, description symmetry).
- **Why:** The `data-scientist` agent was created 2026-07-09 explicitly to work alongside the architect; the consumer side must state the convention too (agent-maintenance §4 handoff symmetry).
- **Plan items:** none.

## 2026-07-09 — Consume tico's requirements doc by path (handoff symmetry)
- **What:** "Understand the request" now states that a feature requirements document from `tico` may arrive as a path (`<component>/docs/requirements/<slug>.md`) — read it first as the stakeholder-confirmed WHAT/WHY the plan turns into a HOW; its acceptance criteria feed the test strategy.
- **Why:** `tico` was created 2026-07-09 as the requirements half of a tico→architect handoff; the consumer side must state the convention too (agent-maintenance §4 handoff symmetry).
- **Plan items:** none.

## 2026-07-09 — K-002 ✅: live handoff validation (teco K-001 run, falkor-chat M3 slice 1)
- **What:** The architect ran as the planning half of a real orchestrated delivery — teco
  delegated it the M3 decomposition + slice-1 plan for falkor-chat. It produced
  `falkor-chat/docs/plans/m3-workflow-engine.md` (Part A: six kaizen items K-020…K-025 in the
  component's exact item format; Part B: full slice-1 plan with data model, DDL reconciliation,
  query shapes, service surface, build order, enumerated suite-count expectations) and returned
  the path. Two isolated-context implementers executed it cold: graph-dba (gate) and
  tdd-engineer (impl) — no re-investigation loops, suites landed green (query 193/193,
  pytest 196), structural parity + idempotency proven.
- **Friction observed (the K-002 payload):** one plan gap — `publish_workflow_def` was specced
  without a `start_key` parameter; the implementer resolved it (exactly one step declares
  `start: True`) and it was surfaced as a contract to lock at K-022. One gate-level design
  amendment — the plan's `STARTS WITH stepUid` scoping PROFILEd as a label scan, so graph-dba
  added a `HAS_STEP` containment edge; a reasonable division of labor (live PROFILE data is the
  gate's job), not a plan defect. Verdict: the six-section template held; no template change
  needed from a single datapoint — recheck if the parameter-contract gap recurs.
- **Prompt changes:** none.
- **Plan items:** K-002 ✅ done (moved here — the live validation this item waited on; evidence
  shared with teco K-001, see `claude/teco/docs/HISTORY.md` 2026-07-09). Plan is now empty of
  active items.

## 2026-07-08 — Plan-doc handoff by default, subagent-context awareness, hook-enforced Write/Edit (K-001 ✅, K-003 ✅)
- **What:** Four changes from a team-level design review (architect as a member of teco's roster):
  1. **Plan document is now the default deliverable** (K-001 ✅): step 5 rewritten — write the plan to `<component>/docs/plans/<slug>.md` (kebab-case; repo-root `docs/plans/` for cross-component work) and return the *path* + the "ready to implement" summary; inline delivery only for quick assessments. Handoff section updated to match ("implement the plan at `<path>`"). Convention matches what falkor-chat already used de facto (`falkor-chat/docs/archive/plans/m2-graphrag.md`).
  2. **Subagent-context awareness:** new opener paragraph — the brief is the architect's *entire* context (no user conversation, no other agents' work) and its final message is terminal (`AskUserQuestion` unavailable to subagents). Step 1 reframed: design-changing ambiguity → return findings + the one or two sharp open questions *as the deliverable* and stop, instead of "ask questions" (impossible mid-run).
  3. **TDD-destined plans:** test-strategy section now says to sequence as an ordered list of behaviors/test cases (red→green) when the repo mandates TDD or the implementer is `tdd-engineer` (this user's default preference).
  4. **Harness-enforced read-only-on-code** (K-003 ✅): new subagent-scoped `PreToolUse` hook `architect/hooks/guard-plan-doc-writes.sh`, wired in frontmatter (`matcher: Write|Edit`, absolute path — devops precedent). Escalates any `Write`/`Edit` targeting a path outside `docs/plans/` (or `/tmp` scratchpad) to the human (`permissionDecision: "ask"`); fail-open on unparseable input with the prompt guardrail as backstop. Smoke-tested: code path → ask; absolute + relative `docs/plans/` → pass; `/tmp/` → pass; garbage stdin → pass.
  - Sibling + catalog sync (same change): `teco.md` now hands off the architect's plan **by path** (never paraphrased into the brief) and gained a coordination-doc convention (`docs/plans/<slug>-coordination.md`, teco K-003 ✅); `coder.md` orient step expects the plan-document path and reads the file as source of truth (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md` updated).
- **Why:** Review found (a) inline-by-default delivery was optimized for a human caller and pessimal for the orchestrated path — teco copying a plan "verbatim" into a brief is a lossy telephone game, while a file handed off by path is lossless and durable; (b) the prompt assumed an interactive caller it doesn't have as a subagent; (c) the read-only contract was prompt-only while a working hook precedent existed in-repo (`devops/hooks/guard-destructive-ops.sh`).
- **Decision — Bash deliberately NOT hooked:** the hook closes the realistic *accidental* failure mode (the editing tools drifting into source). Mutating the tree via Bash would be a deliberate guardrail violation, which prompt-guarding handles reliably for Opus-class models, and pattern-matching "bash writes" is brittle/noisy. Accepted residual risk; escalation path recorded in the plan's parking lot.
- **Plan items:** K-001 ✅ done, K-003 ✅ done (both moved here); K-002 remains open — the convention is baked but the live architect→coder validation run is still pending.

## 2026-06-21 — Added `Edit`, scoped to plan/design docs
- **What:** Added `Edit` to the frontmatter `tools` list (`Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent`). Reworded the Guardrails so `Write`/`Edit` are explicitly for **one purpose** — authoring (`Write`) and revising in place (`Edit`) the plan/design document — and never source/tests/config. Updated catalog entries that previously asserted "no `Edit`" (`claude/AGENTS.md`, root `AGENTS.md`). The `description` and `claude/README.md` ("does NOT edit source code" / "without editing code") were left unchanged — still accurate, since the agent edits plan docs, not code.
- **Why:** User asked to enable `Edit` for the architect. Flagged the design tension (its whole contract is read-only-on-code; the description is also the auto-delegation routing signal) and confirmed intent: **plan docs only**, not code. Previously the agent could only `Write` (overwrite) plan files; `Edit` lets it amend a plan in place during an architect→coder iteration without rewriting the whole document. Tool gating still can't enforce "plan docs only" (both Write and Edit can target any path) — the prompt guardrail carries that, same as before.
- **Plan items:** advances the spirit of K-003 (tool gating) but in the *loosening* direction; K-003 (stricter gating) left open — see plan note.

## 2026-06-20 — Dropped "senior" framing
- **What:** Removed "senior" from the `description` ("Senior software architect" → "Software architect") and the body opener ("You are a senior software architect" → "You are a software architect"). Mirrored in the catalog entries (`claude/README.md`, `claude/CLAUDE.md`, root `AGENTS.md`).
- **Why:** User flagged the overconfidence concern with seniority framing. Evidence (persona-prompting studies, e.g. Zheng et al. 2024) shows role labels are weak-to-neutral for correctness and authority framing can dent calibration; behavior is driven by the concrete process + guardrails, not the title. Chose the most conservative option (drop the word entirely) over keeping it. Note: this goes one step *further* than the 2026-06-05 collection precedent, which dropped boasts but **kept** "Senior" as an altitude signal — so architect/coder are now inconsistent with cobb/tdd-engineer/graph-dba/dra-claudia until those are harmonized (flagged to user).
- **Plan items:** —

## 2026-06-20 — Created
- **What:** Created the `architect` subagent (`architect/architect.md`, `model: opus`). Read-only design/planning agent: investigates the codebase, weighs trade-offs, and produces a step-by-step implementation plan/spec (goal & scope, context & findings, design & rationale, ordered steps, test strategy, risks & open questions). Tools restricted to `Read, Grep, Glob, Bash, Write, WebFetch, WebSearch, Agent`; **no `Edit`/`NotebookEdit`** and a hard guardrail that `Write` is for the plan document only — it never edits source/tests/config. Designed as the planning half of an **architect→coder handoff**: the plan stands alone so an isolated-context implementer can execute it.
- **Why:** User asked to create two complementary Claude Code subagents, "the architect" and "the coder," distinct from the existing `tdd-engineer` (strict TDD implementer) and the OpenCode `coding-senior`, with a sequential architect→coder handoff.
- **Plan items:** seeded K-001..K-003.

## Decisions recorded at creation
- **Why `Write` but no `Edit`:** the agent must be able to emit a *durable* plan document for the handoff (an isolated coder context won't see the architect's investigation otherwise), but must not surgically modify existing code. Omitting `Edit`/`NotebookEdit` + a prompt guardrail signals "planning only" while still allowing the deliverable. `Bash` is investigation-only by guardrail. Tool gating can't fully enforce "plan docs only" (Write can overwrite any path) — the guardrail carries that.
