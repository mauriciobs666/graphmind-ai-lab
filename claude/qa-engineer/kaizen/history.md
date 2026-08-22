# Kaizen — Change History: qa-engineer

> Dated log of actual changes to the `qa-engineer` agent. Most recent first.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Guardrails bullet: when running interactively (`claude --agent qa-engineer`, a
  human present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own
  verified deliverable(s) from the session (the test plan/report, or an authored test file), by
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

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Summary"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a code-level task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there (including this agent's own distillation, immediately below) has already been distilled and cleared — and this file's own pre-migration content was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` distillation: 10 entries — 7 promoted, 2 discarded as already-covered/self-flagged, 1 discovered a stale doc claim and fixed it
- **What:** `cobb` processed all 10 `author:'qa-engineer'` entries in the shared `kaizen_team`
  graph (agent-maintenance skill §5). Verified each against the live system before disposing;
  fields were read via single-column `substring()` paging per entry `a1b2c3d4`'s own finding
  (below) rather than a multi-column table read, to avoid the corruption it describes.
  - **Promoted (5) → new sections in `claude/qa-engineer/qa-testing-techniques.md`:**
    - `1cb831d4` — a twice-verified prompt-compliance gate (plan + diff, both `analyst`, zero
      findings) still failed 3 different ways across 3 live dispatches (`cpg-agent-adoption` M4
      U6, DEF-1/2/3); static gates on prompt wording are necessary, never sufficient, when
      acceptance criteria are prompt-compliance claims — budget live-dispatch sampling as its own
      test layer. Folded `fe00857c` into the same section (a "confirm N defects closed" re-pass
      surfaced a 4th, previously-masked defect — read full output, not just the targeted clause).
    - `3f55c37b` — a `pytest -m live` suite with ~50 sequential local-LLM calls ran 175.92s, past
      Bash's 120s foreground default; expect it and use `run_in_background`/`Monitor` proactively.
    - `1fd61032` — `claude mcp list`/`claude mcp get <name>` via `Bash` is a fresh CLI process,
      independent of the session's own (possibly stale) MCP-client binding — verifies a rename/
      reconfig without a session restart.
    - `c92f6a18` — a first-attempt `mcp__cypher__query` write blocked by Auto-Mode's classifier,
      with an identical retry succeeding immediately, is a harness false start, not a defect
      signal by itself — retry once before concluding the write path is broken.
    - `a1b2c3d4` — cross-referenced (full writeup landed in `cypher-mcp/README.md`, next bullet).
  - **Promoted (2) → project docs (facts about the shared MCP tool, not qa-engineer-private):**
    - `a1b2c3d4` → `cypher-mcp/README.md` "Result format and truncation": a multi-column chunking
      query with 2+ long-string columns can render corrupted/duplicated text in the chat
      transcript even though `format_result()` is a plain deterministic join and the underlying
      data is fine (verified via `size()`/`substring()` on the raw field) — page long fields one
      `substring()` per query, single `RETURN` column, never several long chunks in one row.
    - `75167f4d` → `skills/cpg-analysis/references/freshness.md` Limits section: a `.git`-less
      scratch-copy build isn't necessarily stuck on raw-age-only — a dispatch that independently
      confirms the real repo-relative source path from task context can still validly run the
      stronger `git log` check; flagged the adjacent trap (the marker's literal `sourcePath` run
      through `git log` silently returns a spurious "zero commits" instead of erroring).
  - **Verified live and found *more* than it originally claimed — `13d8e8eb`:** re-tested with a
    real `CREATE`/`DETACH DELETE` round trip against `kaizen_team` (2026-08-21). The entry's claim
    (write-result counters render as floats, e.g. `nodes_created=1.0`) still holds — **and** it
    directly contradicted `cypher-mcp/README.md`'s own text, which claimed (with a specific,
    plausible-sounding citation to the pinned client's source) these render as plain `int`s. Fixed
    the README to state the live-verified behavior and note the contradiction, rather than
    promoting the entry into a knowledge base that would have sat beside an already-wrong doc.
  - **Discarded (2):**
    - `cda51378` — self-flagged in its own `fact` field as a Q1 acceptance-check test entry, "not
      a durable team learning, safe to skip in distillation."
    - `b7a1e4d2` — the curator-clear space-requirement fact it reports is already documented in
      `cypher-mcp/README.md`'s "Writing through this tool" section (per the entry's own evidence,
      the README was fixed in the same pass that produced this entry) — promoting again would
      duplicate, not add.
  - **Docs touched:** `claude/qa-engineer/qa-testing-techniques.md` (5 new sections) ·
    `cypher-mcp/README.md` (write-result float correction + multi-column rendering caution) ·
    `skills/cpg-analysis/references/freshness.md` (Limits section addition).
- **Why:** User-requested distillation pass ("let's work on qa-engineer's inbox").
- **Plan items:** none opened — every actionable entry had a direct, concrete promotion target;
  nothing here needed to wait as a `plan.md` backlog item.

## 2026-08-21 — Enforcement-parity fix: Guardrails now describes the new doc-write guard (team certification, §4 judgment half)

- **What:** The `guard-qa-doc-writes.sh` hook below (same 2026-08-21 rollout) shipped without a
  matching Guardrails line — the destructive-ops hook was already documented, but the *second*
  hook (added the same day) was silent machinery from this file's own perspective. Added one new
  Guardrails bullet describing it accurately: `docs/test-plans/*`/`docs/test-reports/*` writes are
  auto-approved (no prompt), everything else falls through unmediated to the ambient permission
  flow — this hook only ever skips a redundant prompt, it never escalates.
- **Why:** Caught during a user-requested full team-coherence certification
  (`claude/cobb/kaizen/history.md`, 2026-08-21 certificate entry) — enforcement parity is one of
  §4's five judgment-checklist items. Same root cause, same day, as the matching fixes on
  `tdd-engineer.md` and `cobb.md` (both hooks landed without their prompts being updated).
- **Verified:** `bash claude/scripts/audit-team.sh` clean (113 PASS / 2 pre-existing, unrelated
  FAILs, unchanged by this fix).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-21 — Added a `Write|Edit` doc-write guard for test-plans/test-reports (agent-permission-friction FR-3)

- **What:** `qa-engineer` previously had no `Write`/`Edit` guard at all — only its pre-existing
  `Bash`-scoped destructive-ops guard. Added `claude/qa-engineer/hooks/guard-qa-doc-writes.sh`, a
  second `PreToolUse` hook (frontmatter now carries two matcher entries under one `hooks:` block,
  same pattern `security-expert` already uses) over the shared `claude/scripts/guard-doc-writes.sh`
  core, allowlisting `docs/test-plans/*`/`docs/test-reports/*` (component-relative variants too,
  each doubled for an absolute `file_path`). Deliberately uses the core's new `on_mismatch="pass"`
  mode, **not** the default `ask`: qa-engineer's remit is its two doc kinds *plus* whatever
  source/test files phase-3 execution needs, so a non-matching write (e.g. an authored test file)
  falls through to the ambient permission flow exactly as before instead of newly escalating —
  verified this is load-bearing via mutation test (temporarily dropped the `pass` arg, confirmed a
  test-file write started wrongly escalating, then restored and reconfirmed silent pass-through).
- **Why:** Requirements doc `claude/docs/requirements/agent-permission-friction.md` (FR-3,
  instance 4): a manual confirmation was firing on `qa-engineer` writing its own versioned test
  plan/report, squarely in-remit. Same root cause as cobb's FR-1 entry above/below (see design doc
  `claude/docs/plans/agent-permission-friction.md` §1, `analyst`-reviewed, verdict approve) — an
  explicit hook `"allow"` is the mechanism that actually suppresses the prompt.
- **Plan items:** —

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_qa-engineer`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_qa-engineer` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — its 6 pre-existing
  entries were parsed out programmatically and imported into the graph verbatim (entryId
  assigned, `author: 'qa-engineer'`), preserving every field; its own header explains the freeze
  and gives the live-read query. The "defects belong in the test report, not here" distinction
  was kept.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details (defects belong in the test report, not here)," and "never edit your own agent definition" (no write-guard clause — qa-engineer has no doc-scoped write guard). Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Freshness-check clause removed (centralized on teco)
- **What:** Dropped the CPG freshness-check paragraph from the CPG-orientation step — still checks whether a relevant CPG exists and uses it via `cpg-analysis`, but no longer queries the `:CpgBuildInfo` freshness marker itself. That responsibility is now `teco`'s alone (`docs/plans/cpg-agent-adoption2.md`, extending the archived `cpg-agent-adoption.md`); running standalone (no `teco`-issued brief), staleness is simply not checked.
- **Why:** User-directed prompt-verbosity reduction: the freshness paragraph was ~130 words, byte-identical across six agent files. Stakeholder chose full centralization over a per-agent dedup, accepting the standalone-run capability loss.
- **Plan items:** —

## 2026-08-16 — U7 fix round: freshness-check sequencing hardened, `CPG:` line anchored (DEF-1/DEF-2/DEF-3)
- **What:** Two wording tightenings per `docs/plans/cpg-agent-adoption-coordination.md` unit U7,
  following this agent's own U6 live-dispatch acceptance pass
  (`docs/test-reports/cpg-agent-adoption-report.md`), applied here to `qa-engineer`'s own wiring
  by `cobb` as U7's fix. (1) The freshness-check sentence now reads "query the freshness check …
  in that same tool call/step, before deciding whether the result needs further
  cross-verification — this is not a separate, optional judgment call" (previously "also run the
  freshness check … as part of that same step") — closes DEF-2 (`architect` reasoned its way past
  the check with a grep/CPG-agreement substitute that doesn't rule out "stale but coincidentally
  consistent"). (2) The `CPG:` line instruction now reads "written verbatim and required in all
  three cases including when the CPG isn't relevant — not paraphrased, not dropped" — closes
  DEF-1 (`coder`, loose prose instead of the literal line) and DEF-3 (`tdd-engineer`, dropped
  entirely on the not-relevant branch). `qa-engineer` itself was not one of the three live-tested
  dispatches in U6, but carries the same near-verbatim wiring pattern, so the fix is applied
  identically here rather than assumed unnecessary — the report explicitly flags that untested
  agents' compliance shouldn't be assumed by extension.
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

## 2026-08-16 — M4 cpg-agent-adoption: discovery wording defaulted, freshness-check bundled, evidence-trail line added
- **What:** Three edits per `docs/plans/cpg-agent-adoption.md` §2.4/§3 (U4b). (1) Frontmatter
  `description` reworded from "With a loaded Joern CPG, uses the `cpg-analysis` skill for
  test-gap analysis (production code no test reaches)" to "Checks whether a relevant CPG exists
  as part of its normal orientation and, when one does, uses the `cpg-analysis` skill for
  test-gap analysis (production code no test reaches)" — conditional → default-orientation
  framing. (2) Phase 1 REASON's "Read the sources of truth" bullet gained a sentence: check
  whether a relevant CPG exists for the code under test (first guess `cpg_<component>`, per
  `skills/cpg-analysis/SKILL.md` §1) — useful here for test-gap analysis specifically — and when
  one is found and used, also run the freshness check
  (`skills/cpg-analysis/references/freshness.md`) as part of the same step, noting what it says
  in the report and surfacing a refresh suggestion — not a silent rebuild — if it looks stale.
  (3) The test report's "Summary" section gained the one-line `CPG:` evidence-trail convention
  (`CPG: used <graph> — <clause>` / `CPG: considered, not relevant — <clause>` / `CPG: not
  applicable — <clause>`).
- **Why:** M4 (`cpg-agent-adoption`) widens CPG discovery from a conditional check to a default
  orientation step across the three already-wired consumers (`analyst`, `architect`,
  `qa-engineer`), bundles the freshness recipe into that same step (FR-6's surfacing half), and
  adds a spot-checkable `CPG:` evidence trail (AC-2). Per `docs/plans/cpg-agent-adoption.md`
  §2.1-2.3, §3, §6 step 2.
- **Plan items:** none.

## 2026-08-11 — Inbox distillation: 15 entries — new `qa-testing-techniques.md` knowledge base (3 entries), 2 to falkor-chat/AGENTS.md, 5 to falkor-chat/DESIGN.md §14.7, 4 folded into graph-dba/cpg-analysis work already in flight, 1 discarded as already covered elsewhere

- **What:** `cobb` processed all 15 entries in `qa-engineer/kaizen/inbox.md` (§5).
- **Promoted:**
  - New on-demand knowledge base `claude/qa-engineer/qa-testing-techniques.md` (pointed to from
    `qa-engineer.md`'s opening section): the WSL2-no-native-browser-automation / Windows Chrome +
    raw CDP fallback, `tmux` for driving a genuinely interactive TUI, and "doctor"/health-check
    subcommands not being guaranteed read-only.
  - `falkor-chat/AGENTS.md`: the `ENABLE_AGENT`/`WORKFLOW_ENABLED` flag-dependency trap, and the
    `pytest -m live` vs. default-run `reference`-wipe marker distinction.
  - `falkor-chat/docs/DESIGN.md` §14.7: **five** new "QA/acceptance-testing gotchas" bullets — a
    `verify_workflows.sh` `reference`-MISSING FAIL not blocking live execution, a `waitsForHuman`
    run resuming on *any* next thread message (not just an `@mention`), `/input`'s response not
    carrying its own failure's `error` reason, **MCP `send_message` never scheduling the
    responder/workflow trigger (only the REST route does)**, and `ModelGateway` requiring an
    explicit `baseURL` with no implicit per-package default. (A prior version of this entry said
    "four" and left the MCP one unnamed — caught by `analyst`'s review, M-1/m-3 in
    `docs/reviews/kaizen-distillation-2026-08.md`.) The MCP `send_message` asymmetry is a product
    gap, not just a testing gotcha — it is already tracked as **K-041** in
    `falkor-chat/docs/BACKLOG.md:1242`, delivered 2026-08-01, so no new backlog item was filed.
  - The `RESULTSET_SIZE` silent-cap finding (this entry corroborated `graph-dba`'s independent
    capture of the same mechanism) and the `METHOD.CODE`-is-narrow / `redis-cli --no-raw` flat-
    stream-parsing / boolean-quoting-not-casing entries (3 more) → folded into `skills/cpg-analysis/
    SKILL.md` and `cypher-mcp/README.md` as part of the same edit that processed `graph-dba`'s
    matching entries (see that agent's 2026-08-11 history entry for the full list).
- **Discarded — already covered elsewhere (1):** the Bash-tool backgrounding entry
  ("`cmd &` backgrounded manually inside a Bash call can stall the tool for its full timeout, and
  its `cd` doesn't persist to the next call") — already landed in
  `skills/agent-standards/claude-code.md`. Missed in the first pass of this entry; added here on
  `analyst`'s review (M-1).
- **Skipped (folded into the KB instead, not a project-doc target):** the `kiro-cli doctor`
  mutation note's suggested project-doc target (`kiro/docs/plans/kiro-demo-agent.md` §2.3) is now
  `Status: archived` — per the repo's doc convention, an archived document is immutable except for
  header-pointer metadata, so this fact was captured only in the new `qa-testing-techniques.md`
  knowledge base instead.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/qa-engineer/{qa-engineer.md,qa-testing-techniques.md,
  kaizen/{history,inbox}.md}` · `falkor-chat/AGENTS.md` · `falkor-chat/docs/DESIGN.md`.

## 2026-07-29 — New verification target: a tico-authored user manual's walkthroughs
- **What:** `tico` gained a new doc kind, user manuals (`<component>/docs/manuals/<slug>.md`), and the team certification pass flagged manuals as the one doc kind with no independent-review gate. User decision: split the review — `qa-engineer` drives the running app through the manual's walkthroughs (the behavioral half), `analyst` checks the rest. Added a short section after the intro paragraph: the manual's own walkthroughs *are* the spec, each step is a test item, scale the test plan/report to the manual's size (not full feature-QA ceremony), same topic slug as the manual. Explicitly excludes the manual's factual/architectural claims — `analyst`'s half, don't duplicate. Frontmatter `description` updated to name the new target and the analyst/qa-engineer split.
- **Why:** user ruling following the 2026-07-29 team certification's open observation (logged in `cobb/kaizen/plan.md`, now resolved). Routed through `teco`'s existing "independent review" default (its own kaizen carries the matching entry).
- **Plan items:** none — no prior plan item covered this; not adding one since it's already implemented.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Doc convention: `archive/` move rule dropped, filename grammar made non-negotiable, header block required (step 1 of `docs/plans/doc-reference-convention.md`)
- **What:** Four prompt edits. (1) The PLAN phase's "detect the convention first" bullet lost its `docs/archive/<same-subdir>/` sentence — under D4 a frozen document no longer moves, it gets `Status: archived` in its own header — and lost the `/milestone` half of *"named for the feature/milestone under test"*, which licensed exactly the `m<n>-` filename prefix the new grammar prohibits. (2) That bullet's *"if a component uses a different convention, follow that"* escape, and the same escape in the **Match the project** principle 26 lines below, are now subordinated: the filename grammar is repo-wide (root `AGENTS.md`) and **not component-negotiable**. (3) The test-plan and test-report structures each gained the canonical line *"Open the document with the header block from root `AGENTS.md`."* — a pointer, not an inlined template, because root `AGENTS.md` is already in every agent's context via the root `CLAUDE.md` `@AGENTS.md` import.
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §12 step 1 (decisions D1/D4/D6, two analyst review rounds plus a spot-check). Both `AGENTS.md` files flip in the same change: leaving the prompt's *"never into `archive/`"* against a rule with no `archive/` destination is the contradiction the step exists to prevent. `claude/README.md` row 16 re-checked — it cites write paths, not the archive rule, so no catalog edit was needed.
- **Plan items:** none opened or closed; K-002's 2026-07-11 closure note is annotated in `plan.md` because the convention it recorded has been superseded.

## 2026-07-25 — CPG read path moves to `mcp__cypher__query`; catalog row updated (M3 / C-304)
- **What:** `claude/README.md` row 16 now records that the `cpg-analysis` test-gap work queries the graph through the **`mcp__cypher__query`** MCP tool, and that this agent inherits the tool automatically because it declares no `tools:` allowlist. **No frontmatter change** — deliberately: adding an allowlist here to "declare" the MCP tool would newly restrict every other tool the agent inherits.
- **Why:** M3 replaces the CPG read path with a single MCP tool (`docs/plans/cpg-query-access.md` S5). Recording the *reason* this agent needed no edit, while `analyst` and `architect` did, is the point of the entry — the asymmetry is a property of their allowlists, not of their capabilities. `redis-cli GRAPH.QUERY` remains the documented fallback and is the only path under OpenCode/Kiro.
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 798 → 663 chars (-16%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (qa-engineer↔analyst, qa-engineer↔tdd-engineer) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change to `coder`/`tdd-engineer`/`frontend-engineer`/`architect`. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this doesn't weaken `qa-engineer`'s own guard: its `guard-destructive-ops.sh` hook matches Bash command patterns (`GRAPH.DELETE`, `FLUSHALL`/`FLUSHDB`, volume wipes, `docker rm -f`), unrelated to `acceptEdits` (which only covers Edit/Write and common filesystem commands) — and `PreToolUse` hooks fire before any permission-mode check regardless, so a hook `"ask"` decision would survive even if the two overlapped.
- **Plan items:** none.

## 2026-07-19 — CPG test-gap capability wired into the routing description (M2 / C-207)
- **What:** Frontmatter `description` gained one clause: for test-gap analysis over a loaded Joern CPG in FalkorDB, the qa-engineer uses the `cpg-analysis` skill (graph-dba-owned) to find production code no test structurally reaches. `claude/README.md` catalog entry updated to match. No body change (skill is progressively disclosed).
- **Why:** M2 delivered the `cpg-analysis` skill; `qa-engineer` is the named consumer of the test-gap recipe (FR-13, structural reachability — not runtime coverage). C-207 makes the routing contract advertise it. Wired by cobb as part of Gate-2b (skill passed the standards vet).
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Destructive-ops guard + tdd-engineer boundary in description (certification fixes)
- **What:** (1) Frontmatter now wires a `PreToolUse` Bash guard — `qa-engineer/hooks/guard-destructive-ops.sh`, a thin wrapper over the new shared core `scripts/guard-destructive-ops.sh` — escalating `GRAPH.DELETE`/`FLUSHALL`/`FLUSHDB`/volume wipes/container force-removal to human approval; the "never mutate the environment" guardrail now names it as the harness backstop (enforcement parity). (2) The `description` now routes unit-level test-first implementation to `tdd-engineer` (the boundary was previously stated only in the body and on qa's side of the pair); `tdd-engineer:qa-engineer` added to `audit-team.sh` `BOUNDARY_PAIRS`. Catalog row updated.
- **Why:** Team-coherence certification (2026-07-11): the agent drives running apps against the shared live FalkorDB with unrestricted Bash, but its no-mutation rule was prompt-only hope while devops had the harness gate; and the qa↔tdd altitude boundary was asymmetric at the description (routing-contract) level.
- **Plan items:** implements cobb K-011 on this agent's side.

## 2026-07-11 — Module docs convention updated (kaizen→BACKLOG, archive/ rule)
- **What:** The PLAN phase's "detect the convention first" bullet now cites backlog IDs from `docs/BACKLOG.md` (modules no longer have `kaizen/plan.md`) and adds the `docs/archive/<same-subdir>/` rule: completed-milestone docs are frozen there — new test plans/reports go to the active `docs/test-plans/`/`docs/test-reports/` dirs, never into `archive/`. This closes K-002's intent from the other side: the convention is now defined once in the root `AGENTS.md` (module documentation convention) rather than only inferred.
- **Why:** Repo-wide docs unification (2026-07-11, see `falkor-chat/docs/HISTORY.md`): module-level `kaizen/{plan,history}.md` retired into `docs/{BACKLOG,HISTORY}.md` + `docs/archive/`. Agent-folder kaizen pairs (this file) are unchanged.
- **Plan items:** K-002 effectively resolved by the root-`AGENTS.md` convention + `falkor-chat/AGENTS.md` key-docs rows.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 844 to 575 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — analyst boundary clause (description + intro)
- **What:** Frontmatter `description` and the intro's deferral paragraph now route *static* judgment — reviewing a plan, diff, or module by reading and reasoning, without executing the system — to `analyst`, mirroring analyst's new clause routing new black-box/acceptance execution here. The pair is mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry). Catalogs synced (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): qa-engineer named tdd-engineer but not analyst, leaving the static-review vs. executed-verification boundary invisible to routers.
- **Plan items:** none.

## 2026-07-09 — Subagent-awareness lines (teco interface review)
- **What:** Three clauses added during the teco interface review: workflow step 1's "ask one sharp question", the EXECUTE-phase "ask before installing or mutating the environment" bullet, and the never-mutate-the-environment guardrail now all say what to do when running as a subagent (e.g. delegated by teco) — return the sharp question / approval request as the result (marking affected items blocked) instead of trying to ask mid-run, which subagents can't do. Catalog entry (`claude/AGENTS.md`) updated. In the same change, **teco itself gained the K-003 loop**: its roster now includes qa-engineer (with the `docs/test-plans/` / `docs/test-reports/` path-handoff conventions), its pipeline ends in a QA pass when warranted, and its integrate-&-verify step encodes defect → re-brief implementer with the report path → re-run failed items.
- **Why:** The agent's "ask" phrasing assumed an interactive session; under teco delegation that would stall or misfire. The teco-side change closes the orchestration half K-003 anticipated.
- **Plan items:** K-003's teco side is now in teco's prompt; K-003 stays open pending a live orchestrated defect→fix→re-run cycle.

## 2026-07-01 — true delegated run confirmed (auto-routing works)
- **What:** after the session reloaded its subagent registry, invoked `qa-engineer` for real via the `Agent`/Task tool (`subagent_type: qa-engineer`) on a focused follow-up pass against falkor-chat M1 (residual gaps: room-wide `read_messages`, DEF-1 regression). The subagent ran its own playbook end-to-end and **appended** to the existing plan + report (didn't overwrite): TP-026 + TP-027 both PASS, baseline 57/57, DEF-1 still reproduces.
- **Why:** close the loop on the K-004 registry-reload gotcha — prove the agent is routable and behaves correctly under genuine delegation, not just as a cobb proxy.
- **Result:** ✅ auto-routing works; the agent honored the self-contained brief (subagents don't share context), respected the append-don't-overwrite instruction, obeyed the environment pre-authorization, started/stopped the server itself, and left the environment clean. Confirms the K-004 gotcha is purely a session-start registry-load timing issue.
- **Docs touched:** falkor-chat test-plan + report (appended by the subagent).

## 2026-07-01 — first spin (proxy-run) against falkor-chat M1
- **What:** exercised the agent's four-phase playbook end-to-end on the falkor-chat M1 server (REST + MCP). Produced `falkor-chat/docs/archive/test-plans/m1-chat-mcp.md` and `.../test-reports/m1-chat-mcp-report.md`. Result: 22/22 functional+contract items passed on a 57/57 baseline; found DEF-1 (MCP endpoint 405s at `/mcp`, only `/mcp/` works — README/DESIGN mismatch).
- **Why:** validate the new agent's methodology yields a usable strategy → plan → execute → report cycle.
- **Run mode:** **proxy** — run by cobb following the qa-engineer prompt, NOT via Task delegation. Reason: Claude Code loads the subagent registry at **session start**, so the freshly-symlinked `qa-engineer` was not yet routable in the session that created it (`Agent(subagent_type='qa-engineer')` → "agent type not found"). Expected behavior; a new session picks it up.
- **Playbook validation (what worked):** the "verify before asserting" rule caught a wrong hypothesis (assumed `ServiceError`→500 because `api.py` lacks handlers; actually `app.py` maps them 404/400). Evidence-over-assertion produced a clean, reproducible defect. Doc-convention detection (`docs/test-plans/` + `docs/test-reports/`, kebab per feature) worked. Environment-approval guardrail behaved (needed cobb's explicit pre-authorization to touch the shared DB).
- **Docs touched:** falkor-chat test-plan + report (new); `falkor-chat/docs/HISTORY.md` note.
- **Plan items:** validated K-001 need (templates would have sped the plan/report authoring); added K-004 (first-run smoke-eval + document the registry-reload gotcha in the agent README/testing notes).

## 2026-07-01 — created
- **What:** authored the `qa-engineer` subagent — a QA / functional-testing specialist that (1) reasons about risk to build a test strategy, (2) writes it to a versioned test plan following the component's doc conventions (`docs/test-plans/<kebab>.md`), (3) executes it by authoring automated functional/acceptance tests, running existing suites, AND driving the running app black-box, and (4) writes a test report (`docs/test-reports/<kebab>-report.md`) with results, defects, and feedback. `model: opus`, inherits all tools (needs Write/Edit/Bash to author tests, run suites, and drive apps).
- **Why:** user asked for a functional-testing agent that reasons → plans → executes → reports. Fills the behavior/acceptance-altitude gap next to `tdd-engineer` (unit, test-first) and `coder` (implementation).
- **Design decisions (user-confirmed):** execution mode = "both — author, run, and drive"; artifact location = per-component `docs/` dirs (detect each component's convention). Name `qa-engineer` chosen by cobb (user went idle on the name question) to match the role-named technical specialists (`tdd-engineer`, `graph-dba`, `coder`, `architect`).
- **Boundaries drawn:** does NOT fix code under test unless asked (defers to coder/tdd-engineer); never mutates the shared FalkorDB environment without approval; evidence-over-assertion; extends past the unit layer rather than duplicating it.
- **Docs updated:** `claude/README.md` (catalog + kaizen list), `claude/AGENTS.md` (agent context), root `AGENTS.md` (repo catalog). Deployed via `~/.claude/agents/qa-engineer` → `claude/qa-engineer` symlink.
- **Plan items:** seeded K-001 (reusable plan/report templates), K-002 (pin artifact-location convention in component AGENTS.md), K-003 (defect→fix→re-run handoff).
