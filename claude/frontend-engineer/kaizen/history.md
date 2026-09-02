# Kaizen — Change History: frontend-engineer

> Dated log of actual changes to the `frontend-engineer` agent. Most recent first.


## 2026-09-02 — U6 of `salesperson-ui`: the three-site `cpg_salesperson` / Streamlit repo fact corrected

- **What:** three edits, all correcting statements the `salesperson-ui` work makes false.
  (1) Frontmatter `description` — dropped `(cpg_salesperson today)` from the CPG-check clause; the
  clause itself (check for a relevant CPG, use `cpg-analysis` before changing shared UI code)
  stands. (2) "Orient first" item 4 — the concrete pointer *"today that's concretely
  `cpg_salesperson` for `salesperson/chatbot.py`"* replaced with the durable fact that answers the
  same question: this lab's CPGs are `pysrc2cpg`-built and therefore Python-only, so **no
  front-end source is in any of them** — what a CPG answers for a front-end engineer is the
  behavior of the server behind the UI (`cpg_falkorchat`). (3) The repo-terrain sentence — *"the
  running UIs are Streamlit apps (`salesperson/chatbot.py`), and `falkor-chat/` may grow a web
  front-end"* replaced with the two surfaces that actually exist: falkor-chat's deliberately
  minimal developer-facing vanilla-JS UI (`falkor-chat/web/`) and the business-facing React + TS
  SPA the FastAPI server mounts at `/shop`, plus the retired-Streamlit anti-precedent.
- **Why:** `analyst`'s plan gate on `docs/plans/salesperson-ui.md` (finding M9, `docs/reviews/salesperson-ui.md`)
  found this prompt states two things the coordination invalidates, and that the fix routes to
  `cobb`. The Streamlit app is retired; the `salesperson/` name is taken over by the new React
  storefront; `cpg_salesperson` holds the *retired* app's code (359 `METHOD` nodes) under a name
  that will suggest the new one.
- **Evidence for the Python-only claim (live, this session):** `cpg_falkorchat` filename extensions
  are `py` × 8524 and extension-less × 521 — zero `.js`/`.ts`/`.html`/`.css`. So `falkor-chat/web/app.js`,
  the one front-end already in the repo, is not in a CPG and never was.
- **Deliberately not pinned:** neither the new component's directory (`salesperson/` vs.
  `salesperson-ui/`, still in flux at the time of writing) nor `deprecated/salesperson/` is named —
  the prompt now describes the surfaces, not the paths, so the next rename is not a fourth site.
- **Plan items:** closes the 2026-08-24 C6-lint watch note (the `cpg_salesperson` three-site rot
  it predicted, arrived) and retires the two now-obsolete "judged and kept" entries it guarded.

## 2026-08-25 — K-003 closed — conventions precedence stated once (conventions-precedence family)
- **What:** `:56`'s local-scope claim (*"indistinguishable from a good existing file in the same folder"*) had no stated precedence against the project-scope material. One sentence appended at `:56`, where the local claim is made; +25 w.
- **This instance is prophylactic, not corrective — recorded so nobody deletes it later.** `:17` (*"Discover conventions; don't import your favorites"*) and `:77` (*"The project's stack and idiom win over your favorite library"*) both run on the project-vs-**agent** axis, not project-vs-local. So unlike `coder:12` and `tdd-engineer:36`, this file does **not** carry two co-equal *scope* claims — it has one local-scope claim and two anti-preference claims. The copy is justified by uniform treatment across declared routing partners, not by a live ambiguity here. A future reader who derives this analysis will be tempted to call the copy dead weight; `audit-team.sh` check 10 now stops that edit, and this note explains why it exists.
- **The rule, byte-identical in all three implementer prompts:** *"Where a file or folder deviates locally from the project norm, match it, not the norm — mixing both in one place is worse than either applied consistently."* Identity is deliberate — three paraphrases would recreate exactly the divergence that made this a `agent-maintenance` §4 **check-5 boundary-reciprocity** problem rather than three separate nits.
- **Not relocated to a shared file, and the reason is stronger than "no plausible home."** Finding 15 says a rule binding N>1 agents belongs where all N read it, and this binds three. But no shared file owns *how to author code*, and root `AGENTS.md` would be an **actively bad** home: its document conventions are stated as absolutes (`never begins with m<digit>`, the closed role set, the closed `Status:` set), and a general "a local deviation beats the project norm" principle sitting beside them hands every agent a lever to justify deviating from them. The rule would not sit inertly there; it would undercut them. So: byte-identical duplication, **plus a mechanical identity guard** — `audit-team.sh` **check 10**, which fails when the sentence is in some but not all three, and passes when it is in all three or none (removing it everywhere is a legitimate family decision; removing it from one is the defect). Recorded as plan finding 22.
- **Why the guard was necessary and not belt-and-braces.** All three kaizen plans said "fix together or not at all" and nothing read those plans at edit time. The mitigation was prose in files nothing loads. Stage F had shipped the enforcement machinery the same day, and this is exactly the class of guarantee it exists for: deterministic, must-always-hold, therefore a mechanism rather than hopeful prose.
- **The family boundary is a test, not a judgment call:** *two co-equal scope claims in one prompt with no tiebreak between them.* `graph-dba:53` and `devops` were checked and excluded — not because they are a "different axis" (that was my first, wrong reasoning; `graph-dba:53` is structurally identical and has a real local-vs-project instance in this repo, since `cpg_*` graphs carry Joern's imported schema against the `PascalCase`/`UPPER_SNAKE` norm) but because each states **one** scope, so there is nothing to tiebreak. Recorded so the next reader does not re-open it, notice the shape match, and conclude the exclusion was arbitrary.
- **Verified:** `audit-team.sh` **PASS**, check 10 green at 3/3; negative-tested at 2/3 (FAIL, exit 1) and 0/3 (PASS). `cobb` §7 lint: 0 blockers, 1 major, 3 minors, 2 nits — all applied.

## 2026-08-24 — Prompt-waste compression, Stage C6: one edit — a section heading's redundant scope parenthetical
- **What:** `frontend-engineer.md` was one of C6's four files (`claude/docs/plans/prompt-waste-reduction.md`, Stage C). 1,604 → 1,600 w. **One edit:** `### Python-native UIs (this lab uses them)` → `### Python-native UIs`.
- **Why (class 7):** the lab's Streamlit usage is stated 30 lines earlier and far more specifically at `:20` — *"In this repo the running UIs are Streamlit apps (`salesperson/chatbot.py`)"* — inside the always-read orientation section, which the agent reaches first. The heading's parenthetical restated it vaguely, and the section body is unconditional Streamlit expertise either way. Not needed twice.
- **Gate (a) inventory — all preserved:** the Streamlit rerun-model / `st.session_state` / caching / widget-key expertise, the "respect its grain" rule, and the repo-usage fact itself (at `:20`, its authoritative home).
- **Class-6 residual: 0 w.** The one regex hit in the file ("stakeholder-facing", `:69`) is a domain noun, not an authority marker. Seventh consecutive file at zero (plan finding 11).
- **Total residual ~39 w — near the top of the eleven-file band (20–40)**, and structurally so, not as leftover residue. This prompt has four layers that each **re-aim the same cross-cutting rules** at a different altitude: conventions, UI states, and test scope each appear three times, legitimately. See the C6 unit record's candidate finding 13.
- **Judged and kept** (recorded in `plan.md`): `today that's concretely `cpg_salesperson`…` (kept under finding 10's scope-above-rot test — this agent is project-scoped, unlike `devops`, and no standing refresh obligation is attached); the `falkor-chat/` "may grow a web front-end — check its docs" clause (a class-1 anti-trigger against assuming falkor-chat is Streamlit like salesperson); the "Every UI state is a requirement" principle.
- **One pre-existing defect found and routed, not bundled** (§4.0 rollback contract — the fix is a rule change): the conventions-precedence ambiguity at `:17`/`:56`/`:77`. See `plan.md`, and **fix it together with `tdd-engineer` K-006 and `coder` K-004** rather than alone.
- **Verified:** `audit-team.sh` PASS; `cobb` §7 lint — clean on all seven dimensions for this file, 0 blockers, 0 majors, no finding attributable to the edit.

## 2026-08-23 — Prompt-waste Stage B wave 2: three boilerplate blocks compressed to pilot shapes
- **What:** CPG-freshness clause, interactive-commit-grant bullet, and learning-capture intro/tail compressed to the pilot-validated wordings in `architect.md`/`coder.md` (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B), including wave 2's uniform freshness form ("when a `teco`-issued brief states the graph's freshness, take it as given").
- **Removed (class 5/6, already on record):** the grant's "same as before. Stakeholder decision, 2026-08-21 — see `kaizen/history.md`" — this file's 2026-08-21 grant entry; the tail's inbox-replacement sentence + ", exactly like the old inbox was" — this file's 2026-08-21 inbox-deletion entry; the freshness clause's "(2026-08-19)" date and "without re-deriving staleness yourself" restatement — this file's 2026-08-19 freshness-centralization entry; the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement (mechanics live in the Cypher template below); the grant parenthetical's "— not spawned via `Agent`/`Task` as an isolated delegate" (moved into the carve-out sentence).
- **Gate (a) inventory — all preserved:** grant scope (own verified changes, this session, explicit path), full never-list, delegated-subagent carve-out + audit check-8 tokens, freshness rule both branches, `cpg_salesperson` concrete pointer, Cypher template + call line verbatim, "skip task-specific/already-documented", "raw capture: `cobb` promotes; never edit your own definition".
- **Verified:** `audit-team.sh` PASS; cobb §7 lint pass.

## 2026-08-21 — Interactive-mode commit grant added (team-wide stakeholder decision)
- **What:** New Principles bullet: when running interactively (`claude --agent frontend-engineer`,
  a human present turn-by-turn — not a delegated subagent), may `git add`/`git commit` its own
  verified code changes from the session, by explicit path, never bulk-staged/pushed/reset/
  rebased/amended; the grant does not apply when spawned as a delegated subagent.
- **Why:** Direct stakeholder ruling, 2026-08-21, after `tico` hit exactly this gap closing out a
  Mode-3 verification pass (its own commissioned artifacts left uncommitted, since only
  `tico`/`teco` had any commit authority). Rather than pin the fix to those two, the stakeholder
  ruled the exception should reach every agent, gated by invocation mode, not identity — full
  rationale, the `claude/AGENTS.md` rewrite, and the `audit-team.sh` check-8 redesign in
  `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — `CPG:` line gained a `not applicable` vs. `considered, not relevant` disambiguation (C-408)

- **What:** `cobb` added one clause to this agent's `CPG:` evidence-trail sentence (§ "Verify in the running UI"): `not applicable` is now explicitly scoped to a task with no code-level component at all, distinct from `considered, not relevant` (a UI task in a component that simply has no loaded CPG). See `claude/cobb/kaizen/history.md`'s matching 2026-08-21 entry for the full reasoning and the defect this closes (`docs/BACKLOG.md` C-408, DEF-4).
- **Why / Verified / Plan items:** see the master entry above.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there has already been distilled and cleared — and this file's own pre-migration content (if any) was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry below). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — Persona fix: dropped stale "senior" framing (team certification, §7 fold-in)

- **What:** Opening line "You are a **senior front-end engineer** — a specialist implementer..."
  → "You are a **front-end engineer** — a specialist implementer...". Dropped the one word.
- **Why:** Caught during a user-requested full team-coherence certification's §7 lint fold-in.
  The team dropped "senior" framing collection-wide on 2026-06-20 (overconfidence concern;
  persona-prompting evidence shows role labels are weak-to-neutral for correctness —
  `claude/cobb/kaizen/history.md`, 2026-06-20 entry, "Collection harmonization"). This file
  (created after that sweep) had never been checked against it — genuine drift, along with the
  same finding on `data-scientist.md` (`claude/data-scientist/kaizen/history.md`, same date).
- **Verified:** `bash claude/scripts/audit-team.sh` — same 113 PASS / 2 pre-existing FAILs before
  and after (diff, not bare gate).
- **Plan items:** none opened — direct fix from a live certification finding.

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
