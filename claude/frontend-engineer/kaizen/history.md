# Kaizen — Change History: frontend-engineer

> Dated log of actual changes to the `frontend-engineer` agent. Most recent first.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_frontend-engineer`), mirroring `graph-dba`
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_frontend-engineer` (FalkorDB, via `mcp__cypher__query`) instead of
  appending to `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had
  no pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the inbox path, "skip task-specific details," and "never edit your own agent definition" (no write-guard clause — frontend-engineer has no doc-scoped write guard). Behavior unchanged.
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
  "query the freshness check … in that same step, before deciding whether the result needs
  further cross-verification — this is not a separate, optional judgment call" (previously "also
  run the freshness check … as part of the same step") — closes DEF-2 (`architect` reasoned its
  way past the check with a grep/CPG-agreement substitute that doesn't rule out "stale but
  coincidentally consistent"). (2) The `CPG:` line instruction now reads "written verbatim and
  required in all three cases including when the CPG isn't relevant — not paraphrased, not
  dropped" — closes DEF-1 (`coder`, loose prose instead of the literal line) and DEF-3
  (`tdd-engineer`, dropped entirely on the not-relevant branch). `frontend-engineer` itself was
  not one of the three live-tested dispatches in U6, but carries the same near-verbatim wiring
  pattern, so the fix is applied identically here rather than assumed unnecessary — the report
  explicitly flags that untested agents' compliance shouldn't be assumed by extension.
- **Why:** U6's acceptance pass found the M4 wiring (U4b/U4b-2) was correctly worded but didn't
  survive contact with a real dispatched agent's own judgment calls — all three live-tested
  dispatches failed a different way (format, skip, silence). Design intent
  (`docs/plans/cpg-agent-adoption.md` §2.3, §3) unchanged: still agent judgment on staleness
  threshold, still no self-triggered rebuild, still a suggestion not a hard rule about *when*
  something counts as stale — only the sequencing and the anchoring got tightened.
- **Plan items:** none new; closes U7.
- **Same-day addendum (U8 diff-gate follow-up):** `analyst`'s U8 diff gate
  (`docs/reviews/cpg-agent-adoption.md`, Pass 3 — approve with suggestions, zero blockers)
  found this file's own freshness sentence was missing the "tool call/" qualifier the other five
  files carried ("…in that same step…" vs. their "…in that same tool call/step…"), undercutting
  the U7 ledger row's and commit message's "identically" claim — the divergence wasn't
  deliberate, just a copy-seam miss. The same review also flagged, across all six files: (b) the
  trailing "this is not a separate, optional judgment call" had an ambiguous pronoun referent —
  a literal reading could bind "this" to the cross-verification *decision* rather than the
  freshness *query itself*, exactly the room DEF-2's `architect` dispatch used to reason past a
  softer version of this sentence; (c) nit — "query the freshness check" mismatched a
  reference-doc noun with a query verb, when the actual queried object is the `:CpgBuildInfo`
  marker (the report's own recommendation said "marker"). Fixed all three: the sentence now
  reads "…query the freshness marker (per `skills/cpg-analysis/references/freshness.md`) in
  that same tool call/step, before you decide whether the CPG's answer needs further
  cross-verification — running the freshness check itself is not optional, and skipping it in
  favor of a substitute check (e.g. grep agreement) doesn't satisfy this." — byte-identical
  across all six files now, closing the (a) gap this file specifically had. The `CPG:`-line
  wording from the original U7 pass was untouched (U8 raised no finding against it).

## 2026-08-16 — Wired as a new `cpg-analysis` consumer (M4 cpg-agent-adoption)
- **What:** Frontmatter `description` gained a clause naming the `cpg_salesperson` CPG check and the `cpg-analysis` skill as part of orientation before changing shared UI code. Body: "Orient first" numbered list gained item 4 — check for a relevant CPG (first-guess `cpg_<component>` naming, `skills/cpg-analysis/SKILL.md` §1; concretely `cpg_salesperson` for `salesperson/chatbot.py` today), and when found/used, run the freshness check (`skills/cpg-analysis/references/freshness.md`) as part of the same step, noting it and surfacing a refresh suggestion (never a silent rebuild) if stale. Step 4 "Verify in the running UI" gained a one-line `CPG:` evidence-trail convention (`used <graph> — <clause>` / `considered, not relevant — <clause>` / `not applicable — <clause>`) in the final report.
- **Why:** `docs/plans/cpg-agent-adoption.md` (M4, `cobb`-owned design) widened the `cpg-analysis` roster from three consumers (`analyst`/`architect`/`qa-engineer`) to six, adding `coder`/`tdd-engineer`/`frontend-engineer`. `frontend-engineer` was named in because this lab's frontend work today *is* `salesperson/chatbot.py`, already covered by `cpg_salesperson` — the same impact-analysis question (`who else calls this`) `coder` already asks. No `tools:` frontmatter change needed; the agent already omits `tools:` and inherits `mcp__cypher__query`.
- **Plan items:** none.

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 682 → 574 chars (-15%): tightened phrasing, dropped restated detail, kept every routing/boundary clause. `claude/scripts/audit-team.sh` boundary-pair symmetry (frontend-engineer↔coder) re-verified green. No body/catalog change.
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

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1156 to 678 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-09 — Created
- **What:** initial version of the agent — front-end specialist implementer: web platform (semantic HTML, modern CSS, JS/TS, React & peers), accessibility, responsive layout, state/data-flow design, front-end performance, front-end testing, plus Streamlit/Python-UI fluency. Orient-first discipline (never assumes a stack), plan-by-path handoff from architect, subagent-aware, `model: opus`, inherits all tools (implementer — no write-scope hook).
- **Why:** the team had generalist implementers (coder, tdd-engineer) but no UI-depth specialist; front-end work (components, styling, a11y, performance, future falkor-chat UI) deserved the same specialist treatment graph-dba gives the data layer.
- **Wiring:** added to teco's routing table + description roster, all three catalogs (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`), symlinked into `~/.claude/agents/`, and paired with `coder` in `scripts/audit-team.sh` `BOUNDARY_PAIRS` (coder's description gained the reciprocal route-away clause).
- **Plan items:** seeded K-001 (shakedown run), K-002 (visual verification tooling).
