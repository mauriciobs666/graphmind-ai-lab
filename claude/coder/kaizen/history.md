# Kaizen — Change History: coder

> Dated log of actual changes to the `coder` agent. Most recent first.

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
