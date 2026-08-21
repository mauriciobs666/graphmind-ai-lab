# Kaizen — Change History: tdd-engineer

> Dated log of actual changes to the `tdd-engineer` agent. Most recent first.

## 2026-08-21 — Enforcement-parity fix: Guardrails now describes the deny-list write guard (team certification, §4 judgment half)

- **What:** The 2026-08-21 rollout below (`guard-tdd-broad-write.sh`, wired via frontmatter
  `hooks:`) shipped the hook itself but never touched this file's body — `tdd-engineer.md` had no
  Guardrails line describing it at all, "silent machinery" by the §4 enforcement-parity
  definition (a wired hook not described in the prompt it guards). Added one new Guardrails
  bullet, first in the list, stating what's unrestricted (source/test files), what escalates (the
  named deny-list doc kinds + `docs/BACKLOG.md`), and the same "don't attempt it expecting a
  rubber-stamp" framing every other guarded agent's prompt already uses.
- **Why:** Caught during a user-requested full team-coherence certification (`claude/cobb/kaizen/history.md`,
  2026-08-21 certificate entry) — enforcement parity is one of §4's five judgment-checklist items,
  and this was a same-day gap from the hook's own landing, not old drift.
- **Verified:** `bash claude/scripts/audit-team.sh` clean (check 1's hook-existence/executable
  check doesn't inspect prompt *prose*, so it couldn't have caught this itself — the §4 judgment
  half exists precisely for gaps the deterministic script can't see).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-21 — Added a broad-implementer `Write|Edit` deny-list guard (agent-permission-friction FR-2)

- **What:** `tdd-engineer` previously had no `Write`/`Edit` guard at all. Added
  `claude/tdd-engineer/hooks/guard-tdd-broad-write.sh`, wired via a new frontmatter `hooks:`
  block, over a **new** shared core, `claude/scripts/guard-broad-write.sh` — the inverse shape
  from `guard-doc-writes.sh`: a deny-list (allow by default, escalate only on a match) rather than
  an allow-list, because tdd-engineer's remit is genuinely "the whole codebase, this task," not
  one doc kind. The deny-list covers every other specialist's documented deliverable-path
  convention (`docs/plans/*`, `docs/reviews/*`, `docs/requirements/*`, `docs/manuals/*`,
  `docs/test-plans/*`, `docs/test-reports/*`, agent definitions/kaizen under `claude/*`, the team
  catalog files, cobb's skill packages, `cypher-mcp/README.md`) plus `docs/BACKLOG.md` — kept in
  the deny-list deliberately so a tdd-engineer → `BACKLOG.md` write keeps escalating exactly as
  today, per the requirements doc's unresolved instance U1 (not decided either way by this
  change). Every entry doubled (bare + `*/`-prefixed) for an absolute `file_path`.
  Mutation-tested (temporarily dropped the `docs/BACKLOG.md` entries, confirmed the guard
  wrongly fell back to `allow` on that path, then restored and reconfirmed `ask`).
- **Why:** Requirements doc `claude/docs/requirements/agent-permission-friction.md` (FR-2 general,
  instances 7-8, AC-3's named tdd-engineer example): a manual confirmation was firing on ordinary
  in-remit test/source-file edits despite `acceptEdits` since 2026-07-24. Root cause (design doc
  `claude/docs/plans/agent-permission-friction.md` §1, `analyst`-reviewed across two passes,
  verdict approve): frontmatter `permissionMode` is silently ignored/overridden by the parent
  session's mode in documented cases; an explicit hook `"allow"` is the mechanism that actually
  suppresses the prompt regardless of ambient mode. `frontend-engineer`/`devops`/`graph-dba` are
  designed-for the same treatment but deliberately not touched this round (zero live evidence);
  `coder`'s friction and instance U1 stay explicitly out of scope, per the requirements doc.
- **Plan items:** —

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_tdd-engineer`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_tdd-engineer` (FalkorDB, via `mcp__cypher__query`) instead of appending
  to `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," and "never edit your own agent definition" (no write-guard clause — tdd-engineer has no doc-scoped write guard). Behavior unchanged.
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
  paraphrased, not dropped" — closes DEF-1 (`coder`) and DEF-3 (`tdd-engineer`, this agent,
  produced a thorough, correct test-gap analysis on a no-CPG component with zero mention of "CPG"
  in any form — indistinguishable from the discovery step never running at all). Applied
  identically (phrasing pattern, not restructuring) across all six wired agents; only
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
  `cpg-analysis` skill for RCA and impact analysis before writing a reproduction test — the
  actual call path to the symptom, what else exercises the function — and for test-gap analysis
  when scoping what to test next." **Step 1 "Understand first":** added a check for a relevant
  CPG (first-guess `cpg_<component>` naming per `skills/cpg-analysis/SKILL.md` §1) bundled with
  the freshness check (`skills/cpg-analysis/references/freshness.md`) in the same step, noting
  the result in the report and surfacing a refresh suggestion — never a silent rebuild — if
  stale. **Step 5 "Verify honestly":** added the `CPG:` evidence-trail line (plan §3 — `used
  <graph> — <clause>` / `considered, not relevant — <clause>` / `not applicable — <clause>`).
- **Why:** M4 widens the `cpg-analysis` roster from three consumers (`analyst`, `architect`,
  `qa-engineer`) to six — `tdd-engineer` is a new consumer because its reproduction-test-first
  work benefits directly from RCA/impact recipes ("what's the actual call path to the symptom,"
  "what else exercises this function"), and test-gap analysis is a natural companion to "what
  should I be testing." Plan §1's `tdd-engineer` row has the full roster reasoning.
- **Plan items:** none (design-driven, not backlog-driven).
- **Addendum (same day):** the description clause initially shipped in the conditional "With a
  loaded Joern CPG, uses…" framing (matched to `coder`'s clause per the dispatch instruction).
  The coordinator caught that this left `coder`/`tdd-engineer` as the only two of the six wired
  agents not on plan §2.1's mandated default-orientation framing — the sibling unit's
  `analyst`/`architect`/`qa-engineer`/`frontend-engineer` edits all use "Checks whether a relevant
  CPG exists as part of its normal orientation and, when one does, uses…". Reworded to match:
  "Checks whether a relevant CPG exists as part of its normal orientation and, when one does,
  uses the `cpg-analysis` skill for RCA and impact analysis before writing a reproduction test —
  the actual call path to the symptom, what else exercises the function — and for test-gap
  analysis when scoping what to test next." Body-prompt and evidence-trail additions were already
  correct and untouched.

## 2026-08-09 — Description gained a `python-web-quirks` skill routing clause
- **What:** Frontmatter `description` gained one clause: in a Python web/async codebase, the
  agent consults the new `skills/python-web-quirks/SKILL.md` for asyncio/FastAPI/Starlette/
  pydantic gotchas — mirroring how `cpg-analysis` is wired into `analyst`/`architect`. No body
  change. **Note:** this edit landed via `Edit` while the file already had unrelated in-flight
  changes to step 5 ("Verify honestly") from a concurrent session — the edit applied cleanly and
  did not touch that content; flagging the concurrency here for visibility, not because anything
  needed reconciling.
- **Why:** `python-web-quirks` was created distilling three general Python/web-framework facts
  from `analyst`'s learnings inbox. Stakeholder wired it to `coder`/`tdd-engineer`/`architect`/
  `analyst` at minimum. See `claude/analyst/kaizen/history.md` (2026-08-09) for the full
  distillation record and `skills/README.md` for the catalog entry.
- **Plan items:** none.

## 2026-08-09 — Learnings-inbox distillation, first pass (5 prompt additions; 3 discards)

- **What:** Processed all 9 entries in `kaizen/inbox.md` (dated 2026-07-15 through 2026-07-31) —
  the agent's first-ever distillation pass. Applied to `tdd-engineer.md`:
  1. New **Principles** bullet — symmetric fixture teardown for shared/global state (from the
     2026-07-24 "fixture wipes at setup, not teardown" entry).
  2. New **Principles** bullet — gate optional/slow tests with the runner's marker/tag mechanism,
     not a bare reachability skip (from the 2026-07-15 "reachability-skip doesn't gate a live test"
     entry).
  3. Expanded the **Cover the edges** bullet with positional/anchored-rule coverage guidance — vary
     the anchored position (first/middle/last) and check what the consumer does with the whole
     result (merged the 2026-07-24 "single-line corpus can't test a line-anchored rule" entry and
     the 2026-07-24 "positional accept-rule anchored on one list element" entry — the same
     underlying lesson, rediscovered twice in the same K-027 review pass).
  4. New **Principles** bullet — when merging two callers onto one shared helper, compare what each
     caller *does* with the result (validating vs. acting consumer), not just how similar the
     parsing looks (from the 2026-07-24 "two extractors, opposite safety postures" entry).
  5. Rewrote workflow step 5 ("Verify honestly") to require reading `passed`/`skipped`/`deselected`
     counts, not just exit code, and to run the *whole* suite (not just new/reproduction tests)
     after a fix, keeping the concrete "adjacent, unrelated-looking pre-existing test catches
     idempotency bugs" illustration (merged the 2026-07-24 "green exit code isn't evidence a suite
     ran" entry and the *second half* of the 2026-07-31 "idempotency guard can reuse an
     accumulator" entry).
  - **Discarded, already covered:** the 2026-07-15 empty-`UNWIND`-in-`materialize_snapshot` entry
    — fully superseded by `falkor-chat/docs/BACKLOG.md` K-030 (🔵 proposed), which documents the
    same defect in more depth (both `publish_def` and `materialize_snapshot`, the partial-write
    hazard, the fix). Nothing left to add.
  - **Discarded, too narrow/inconclusive for this pass:** the 2026-07-24 "first `docker ps` hangs"
    entry (single-occurrence, root cause explicitly not isolated by the reporting agent, and reads
    more like a devops/team-wide environment-probing fact than a TDD-specific one — no shared
    cross-agent home exists today, so parking it in this prompt would misfile it) and the *first
    half* of the 2026-07-31 "idempotency guard" entry (reuse-an-existing-accumulator-instead-of-a-
    new-flag is a real but narrow craftsmanship tip, not a durable prompt-level rule). Both are
    easy to re-raise if the pattern recurs.
- **Why:** Team-maintainer distillation pass (agent-maintenance skill §5), dispositions
  stakeholder-approved after cobb's read-only proposal pass the same day. Two independent
  near-duplicate lesson-pairs (the two positional-rule entries; the skip-count and
  whole-suite-after-fix entries) were merged into single prompt edits rather than kept as separate
  redundant bullets — stakeholder's explicit call.
- **Plan items:** none. `inbox.md` cleared to empty.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** closes the standing "is opus warranted vs. sonnet?" revisit item — model tier is no longer this agent's decision.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 495 → 434 chars (-12%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (tdd-engineer↔coder, tdd-engineer↔qa-engineer) re-verified green. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change to `coder`. File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission on every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`. `acceptEdits` auto-accepts file edits and common filesystem commands for paths in the working directory/`additionalDirectories`, independent of session-level grants.
- **Why:** Same root cause as `coder` (see its 2026-07-24 kaizen entry) — applied to the other implementer agents for consistency, at user request.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt: durable, non-obvious environment facts discovered during runs are appended as dated, evidence-backed inbox entries; the agent never promotes its own entries.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Inbound RCA handoff + qa-engineer boundary (certification fixes)
- **What:** (1) Workflow step 1 now names the second doc-path input the team routes here: an `analyst` RCA at `<component>/docs/reviews/<slug>-rca.md` — its reproduction evidence is the first RED, its suggested fix the target. (2) The `description` now routes acceptance/black-box QA passes to `qa-engineer`, making the altitude boundary symmetric at the routing-contract level (qa's side already named tdd-engineer); `tdd-engineer:qa-engineer` added to `audit-team.sh` `BOUNDARY_PAIRS` so the symmetry is scripted.
- **Why:** Team-coherence certification (2026-07-11, handoff symmetry): analyst and teco both route RCA docs to this agent, but its own prompt only named architect plans as doc-path input; and the qa↔tdd boundary was one-sided.
- **Plan items:** closes the 2026-07-11 parking-lot handoff-symmetry item (same-day).

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 606 to 446 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Efficiency-based routing boundary with `coder` (description narrowed + made symmetric)
- **What:** Rewrote the `description`'s trigger. Was "use proactively whenever the user asks to implement a feature, fix a bug, refactor…" — which shadowed `coder`'s trigger on any feature task; now scoped to where test-first is the **efficient path** (bug fix with reproduction test first, safety-net refactor, adding/improving tests, feature with a clear up-front behavior contract) and points back to `coder` for executing an already-detailed plan/spec (the pointer was previously one-directional, coder→tdd only). Catalogs synced: teco roster (personal-preference note removed), `claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`.
- **Why:** User ruling on the coder/tdd-engineer overlap review: route between the implementers by efficiency, not by an assumed blanket TDD preference; personal-preference framing removed from agent prompts (the user's standing preferences are quality and efficiency). Closes coder K-001 from this side.
- **Plan items:** none active (out-of-band; counterpart of coder K-001 ✅).

## 2026-07-09 — Plan-doc-path handoff + subagent-awareness (teco interface review)
- **What:** Two workflow additions, made during the teco interface review: (1) step 1 now states that an `architect` plan arrives as a **document path** (`<component>/docs/plans/<slug>.md`) — read the file itself as the source of truth; its test-strategy section is the red→green sequence. Mirrors the line `coder` has carried since the 2026-07-08 path-based-handoff change. (2) Step 2 and the red-suite + environment-blocker branches of step 3 gained subagent-awareness: when running delegated (e.g. by teco), "ask one sharp question" / "ask whether to fix first" / "ask before installing" becomes "return the question/blocker as your result" — subagents can't ask mid-run. Catalog entry (`claude/AGENTS.md`) updated.
- **Why:** teco routes implementation to this agent *preferentially* (user's TDD preference), yet the handoff contract was documented only on `coder`; and the "ask" phrasing silently assumed an interactive session the agent doesn't get under delegation.
- **Plan items:** none (out-of-band, driven by teco's 2026-07-09 review).

## 2026-06-20 — Dropped "Senior" from description (collection harmonization)
- **What:** Frontmatter `description` "Senior software engineer who implements…" → "Software engineer who implements…". Catalog row in `claude/README.md` "Senior engineer who implements…" → "Software engineer who implements…".
- **Why:** Collection-wide harmonization. The new `architect`/`coder` agents dropped "senior" entirely over the overconfidence concern; this brings tdd-engineer in line. **Supersedes the 2026-06-05 decision** that deliberately *kept* "Senior" as a role/altitude signal — the collection now omits it everywhere, relying on concrete process + guardrails for altitude/calibration instead.
- **Plan items:** —

## 2026-06-05 — Dropped tenure-boast framing
- **What:** Removed "with decades of experience" from the `description` and "with decades of hands-on experience" from the opening body line (now "You are a software engineer who works across many languages, paradigms, and frameworks."). Kept "Senior" in the description as a role/altitude signal.
- **Why:** User feedback — the "decades of experience" framing reads as cocky and doesn't change behavior. Applied collection-wide (also graph-dba, dra-claudia).
- **Plan items:** —

## 2026-06-05 — Implemented K-005 (third cold-start branch: tests can't run in this env)
- **What:** Edited `tdd-engineer.md` workflow step 3. Added a third sub-branch alongside "greenfield" and "suite already red on arrival": **"Framework exists but the suite can't run here."** It instructs the agent to recognize an environmental block (deps not installed, missing runtime/toolchain, required build/service step) as *not* a code RED, avoid misattributing it or thrashing on setup, report the blocker plainly, propose the bootstrap step (`npm install`, `uv sync`, build, etc.), and ask before installing/changing the environment — establishing a runnable baseline before the first RED.
- **Why:** Closes K-005. The original two branches silently assumed an existing framework was runnable; in practice a present-but-unexecutable suite is common and was previously unhandled, risking false REDs.
- **Plan items:** K-005 ✅ (closed, removed from active backlog). Active backlog now empty; K-003 deferred remains the only standing decision.

## 2026-06-05 — K-003 deferred (keep tools unconstrained)
- **What:** Decision only, no file change to the agent. Marked K-003 ⚪ deferred — `tdd-engineer` keeps no `tools` key and continues to inherit all tools.
- **Why:** User chose to keep the agent flexible for now (able to spawn subagents and fetch docs mid-task) rather than restrict to a focused TDD set. Recorded so it isn't re-proposed; revisit only if broad tool access causes surprise in practice.
- **Plan items:** K-003 ⚪ deferred.

## 2026-06-05 — Implemented K-004 (discoverability: catalog + context files)
- **What:** Collection-level docs for the three Claude agents (no agent-prompt change). Created `claude/README.md` (human catalog: one row per agent with what/when/model + links to each source file and `kaizen/` folder; kaizen index; conventions). Created `claude/CLAUDE.md` (agent-context: concise per-agent pointers to source + kaizen, maintenance rules, "don't paste full prompts"). Extended repo-root `AGENTS.md` to register the `claude/` collection — added it to Structure, Component docs (pointing at `claude/README.md` + `claude/CLAUDE.md`), and "Working in this repo".
- **Why:** Closes K-004 — `tdd-engineer` (and cobb, dra-claudia) were invisible to both humans browsing and other agents; no catalog or context entry existed and root `AGENTS.md` covered only OpenCode/salesperson. Satisfies cobb's dual-audience documentation rule.
- **Plan items:** K-004 ✅ (closed, removed from active backlog). Shared deliverable also benefits cobb and dra-claudia.

## 2026-06-05 — Review #2 (no behavior change)
- **What:** Re-reviewed `tdd-engineer.md` at the user's request ("just review/advise"); no prompt edit. Re-verified K-004's discoverability claim against the repo: confirmed no `claude/README.md`, and root `AGENTS.md` documents only the OpenCode/salesperson components — the only `CLAUDE.md` lives under `opencode/agents/severino`, so the three Claude agents (cobb, dra-claudia, tdd-engineer) have no catalog and no context entry. Added **K-005** (workflow step 3 misses the "framework exists but tests can't run in this env — deps/toolchain not installed" case; risk of misreading an environmental failure as a code RED). Added a parking-lot idea on advanced test techniques (table-driven/property-based/mutation testing) as optional, low-priority enrichment.
- **Why:** User chose the review-only path; kaizen rules say record new findings in plan.md even without implementing. Verified rather than assumed the still-open items remain accurate.
- **Plan items:** added K-005; re-verified/annotated K-004; K-003 unchanged.

## 2026-06-05 — Implemented K-001 (test altitude) and K-002 (cold-start/red-baseline)
- **What:** Edited `tdd-engineer.md`. **K-001:** reconciled the "unit test" framing with the agent's broader stated scope — dropped "unit" from the `description` ("failing test", "add/improve tests"); reworded the RED step to "smallest possible test — a unit test by default; reach for an integration or contract test only when that's the genuine seam"; added a new **Right altitude of test** principle (smallest honest test, write real integration/contract tests at seams a unit can't reach, prefer many fast units + a thin higher-level layer); softened the isolation principle to "Fast, isolated, deterministic **by default**" with a note that integration/contract tests are deliberately slower/broader but still deterministic. **K-002:** rewrote workflow step 3 from "Find the test command" to "Establish a green baseline" with two explicit branches — greenfield (set up the minimal runner first as its own announced step, confirm a real baseline) and suite-already-red-on-arrival (stop, report failures, ask fix-first vs. proceed; never build on red or misattribute a pre-existing failure).
- **Why:** User asked to implement backlog items K-001 and K-002. K-001 removed a wording tension that could push the agent to force-fit unit tests where a higher-level test is the honest seam; K-002 closed the cold-start and red-baseline gaps the original step 3 silently assumed away.
- **Plan items:** K-001 ✅, K-002 ✅ (both closed, removed from the active backlog).

## 2026-06-05 — Bootstrapped kaizen + first review (no behavior change)
- **What:** Created `tdd-engineer/kaizen/plan.md` and `history.md` (the agent predated the kaizen convention and had neither). Conducted a review of `tdd-engineer.md` without editing the prompt. Seeded the backlog with K-001 (unit-vs-broader-scope tension), K-002 (cold-start / red-baseline handling), K-003 (tool-permissions decision), K-004 (catalog/discoverability gap), plus parking-lot ideas (no-auto-commit note, coverage-as-guide, flaky-test handling, opus-vs-sonnet cost).
- **Why:** User asked to work on the agent and chose "just review, advise" + "bootstrap kaizen files." Review found the prompt fundamentally solid (clean frontmatter, tight red/green/refactor loop, strong guardrails and anti-hallucination stance); the actionable findings are scope-framing, cold-start coverage, and housekeeping/discoverability — captured as backlog rather than applied.
- **Plan items:** seeded K-001, K-002, K-003, K-004.

## 2026-05-29 — Agent created (retroactively logged)
- **What:** Initial authoring of the `tdd-engineer` agent — a senior engineer that implements features and fixes strictly via Test-Driven Development (red → green → refactor). Frontmatter `name: tdd-engineer`, `model: opus`, routing-oriented `description` with proactive-use triggers (implement a feature, fix a bug, refactor with a safety net, add/improve tests). Body covers the TDD loop, principles, invocation workflow, communication style, and guardrails.
- **Why:** User wanted a dedicated test-first engineer agent (memory: "Prefers TDD"). Logged here retroactively since the agent predates the kaizen convention; date approximated from the source file's mtime.
- **Plan items:** —
