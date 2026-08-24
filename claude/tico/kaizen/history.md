# Kaizen — Change History: tico

> Dated log of actual changes to the `tico` agent. Most recent first.

## 2026-08-24 — K-008 incident 1 closed: a fourth sanctioned `Agent` use — offering to route a finished artifact onward. **K-008 ✅ fully closed.**
- **What:** the last open half of K-008, put to the stakeholder as two questions (may tico dispatch at all, and to which targets). Both answered toward the wider end of what was offered.
  - **May tico dispatch? Yes — and it may *offer* proactively**, not only execute an explicit request. This makes routing behave like the demo and Mode-3 verification offers: tico raises it, the stakeholder accepts, tico dispatches. The narrower "explicit request only" option was declined.
  - **Which targets? `architect`, plus `analyst`/`qa-engineer` for routing** — not "any named specialist". `analyst`/`qa-engineer` were already reachable for *verifying a manual tico wrote*; they are now also reachable for routing work to, which is a different purpose. `graph-dba`/`devops`/`coder` etc. remain out of reach (devops stays reachable for the demo lifecycle only).
- **The incident this closes** (2026-07-31): the stakeholder asked tico to "call the architect" to route a QA finding as a design question; tico declined, since that was not one of its three sanctioned `Agent` uses.
- **Drift mitigation, against `cobb`'s recorded risk that this makes tico "a delegation hub instead of a stakeholder-facing writer" — encoded, not assumed:**
  - Only an artifact **already written and complete** may be routed. tico hands over an artifact; it never composes the delegate's work for it, and never designs the answer itself while the delegate runs. This is what keeps the fourth use distinct from tico *solutioneering*, which was the original reason for the boundary.
  - **"Routing is not coordinating"** is stated explicitly in Guardrails: one artifact handed over, the result reported back. **Sequencing several units, gating them, or chaining one delegate's output into another's brief is `teco`'s job** — and if that's what the situation needs, tico must say so and point at `teco` rather than grow into it. This is the line that keeps the team's single-coordinator model intact.
  - Still **never unilateral**: offering is new, dispatching without acceptance is not.
- **Where it landed:** a new cross-mode bullet in "Running the conversation (all modes)" next to the demo offer (routing can arise in any mode, not just Mode 1's handoff); the Guardrails `Agent`-boundary enumeration; and a pointer from Mode 1's Handoff paragraph, which already named "an architect pass over the doc" as the natural next step and can now offer to perform it. Mode 3's verification offer is cross-referenced as the same mechanism with its own protocol, rather than restated.
- **Propagated:** `claude/README.md`'s tico row. `claude/AGENTS.md` carries no statement of tico's `Agent` scope, so it needed no change — verified by grep, not assumed.
- **K-008 is now fully closed** — incident 2 (commit scope) on 2026-08-24, incident 1 (Agent routing) here.
- **Verified:** `audit-team.sh` PASS on every tico check.

## 2026-08-24 — Two stakeholder decisions on commit authority: an explicit request is a trigger; K-008 incident 2 closed
- **What:** both questions were surfaced by `cobb`'s C2 lint and put to the stakeholder directly (`AskUserQuestion`, framed as two separate decisions because they are — one is about *when* tico commits, the other about *what* it may commit).
- **Decision 1 — timing (new).** The departure-trigger list was closed (different document · out of Mode 1 · session closing), so a literal reading let tico decline a direct "commit this now" because no departure point had been reached. Stakeholder chose the **narrow** option over generalizing it to all self-pacing rules: one clause added — "**or whenever the stakeholder asks you to commit**, which is always a valid trigger wherever you are in the work." K-009's batching is otherwise untouched: tico still never self-triggers on a settled thread, a decision cluster, or a topic switch. Mode 3 inherits automatically, since its bullet points at Mode 1's trigger list rather than restating it.
- **Decision 2 — scope (K-008 incident 2, closed).** Stakeholder chose **shape (a)**: the commit allowance now tracks what the `Write`/`Edit` guard actually let through, rather than being pinned to `docs/requirements/` + `docs/manuals/`. New clause (c) in the Guardrails grant: a file **tico itself wrote in the current session** that the guard let through, including a one-off the human approved at the escalation prompt (the `docs/BACKLOG.md` case from 2026-07-31).
  - **Blast-radius mitigation, against `cobb`'s recorded risk that shape (a) "starts to look like `git add -A` by another route":** the grant is bounded three ways — the file must be one tico **itself wrote**, in **this session**, and staged **by explicit path**. It reaches no file tico didn't write and no earlier session's writes. The write-approval is what confers committability.
  - **This deliberately breaks the write-scope==commit-scope identity** tico previously held, which `cobb` had defended on 2026-07-30 when declining a related widening. The stakeholder's reasoning: a human who has just approved the write at an escalation prompt has already supplied the review that identity was standing in for. Recorded because it overturns a prior cobb recommendation on the record, not because it is in doubt.
  - **K-008 stays open** — incident 1 (routing an already-written finding to `architect` as a fourth sanctioned `Agent` use) was not decided and is untouched.
- **Changed in the same commit, per `claude/AGENTS.md`'s maintenance rule:** `tico.md` (both clauses), `claude/AGENTS.md` "Git-commit authority" (the standing-grants bullet said tico's grant "mirrors its Write/Edit guard exactly" — no longer true), `claude/README.md`'s tico row (same claim, plus the commit-cadence summary it had never carried), and `kaizen/plan.md` (K-008 partial close; the timing item retired from the parking lot).
- **Not a compression change** — this is authority, and it ships separately from unit C2's commit for exactly that reason.
- **Verified:** `audit-team.sh` PASS on every tico check.

## 2026-08-24 — Prompt-waste C2: provenance removed, Mode 3 commit rule un-staled (3,627 → 3,503 w)
- **What:** Unit C2 of `claude/docs/plans/prompt-waste-reduction.md`. Ran as **one pass**: the inventory found essentially no class-7 duplication, so there is no C2 pass 2. 10 edits. Includes the two commit paragraphs deliberately deferred from Stage B wave 2 as non-standard.
- **A behavior change, not a compression — flagged separately because it is the highest-risk item in this unit.** Mode 3's commit bullet read "same discipline as Mode 1: **after a manual section lands**, stage and commit exactly that file" — a within-document trigger at exactly the grain K-009 twice ruled out. It labelled itself "same discipline as Mode 1" while contradicting it. Now: commit the manual's file **when you leave that manual, same departure triggers as Mode 1, not as each section lands**. `cobb` traced this independently and confirmed **stale, not deliberate**: the bullet was authored 2026-07-29 (`8582c49`) as the analogue of Mode 1's *then* per-edit rule, and the 2026-08-19 pass explicitly left it on the reasoning that section granularity "satisfies the new rule as written" — a justification K-009 consumed on 2026-08-23 without anyone revisiting Mode 3. No manuals-specific rationale for section granularity exists anywhere in this file or `plan.md`. **This clause heads C2's observation-window watch list.**
- **Removed (class 5/6, verified on record first):** Mode 1's "17 commits in under an hour" incident and its "direct stakeholder correction the same day (2026-08-23) … (K-009)" provenance (both K-009 entries below); "The stakeholder treats requirements docs as code and wants…" authority framing (2026-07-23 entry); the "now commits far less often than per-topic" comparison to the superseded rule; the grant's "since 2026-08-21, mirroring `teco`'s integrator authority but narrower in scope" and the dated universal-grant parenthetical (2026-08-21 entries); Mode 3's "This is a **new** doc kind" (a claim that would age).
- **Promoted out of deleted narrative — the substance survived its story:** "A topic switch *within* a document is not a departure" was previously carried **only** by the class-5 comparison prose. Deleting the narrative without promoting it would have lost the K-009 refinement's actual content. `cobb`: the best edit in the diff.
- **Class 7 — only two candidates in the whole file, both shipped:** the "No solutioneering" guardrail's re-explaining cross-reference (Mode 2 states it in full where Mode 2 operates; a 4-word pointer kept on `cobb`'s advice, since inside a single always-loaded file the "where does the agent stand" test is weaker than across a load-set), and the `Agent`-boundary guardrail's teardown restatement (the demo bullet carries the per-time semantics in full).
- **Gate (a) — verified present after the edits:** check-8 tokens (`git add`/`git commit` ×1, "delegated subagent" ×2); the requirements-doc and manual templates; `Interviewing | Ready for design`; the three Mermaid diagram kinds and the skip-the-diagram rule; the `AskUserQuestion` 2-option constraint; the `replace_all` terminology trap; the prior-decision-reversal grep; the demo bring-up/teardown protocol; the Mode-3 verification-pass offer; the subagent degradation paths; the Cypher template. Zero residual dates or backlog ids.
- **Verified:** `audit-team.sh` — tico's checks green (one unrelated pre-existing repo FAIL at `claude/docs/reviews/write-guard-classifier-gap.md:58` from commit `374a350`, another session's document, untouched). `cobb` §7 lint: **0 blockers, 1 major, 6 minor**; the major and four minors fixed before commit, two routed to `plan.md`.
  - **Major — created by this compression, and a new instance of C1's finding 1.** Both commit bullets' **bold leads** still read "**Commit as you go**" while their bodies say the opposite. Pre-compression the surrounding narrative cushioned it; removing the narrative left a bolded instruction to commit often above a plain-text instruction not to — the highest-attention token carrying the wrong rule. Not a lost rule: a rule whose **salience** moved when the prose around it was repaired. Both leads are now "**Commit at document boundaries**", and the Handoff cross-reference that named the old label was repaired with them (3 sites consistent, 0 stale).
  - **Minors fixed:** "when you leave **it**" → "when you leave that manual — same departure triggers as Mode 1"; "This holds whether the session touches one document or several" → "…even when the session touches **only one** document" (the compressed original carried a *direction* the short form had lost); the Mode-3 header-block label, which my own rewrite had made collide with the line above it; and Mode 2's third `Explore` restatement (the all-modes bullet covers it — the **inverse** of C1's major, where two occurrences governed two different actors; here both govern tico's own reads).
- **`cobb`'s floor estimate for this file: ~3,450–3,500 w with every rule intact.** At 3,503 the file is done; set no further word target for it.
- **Keep-list for any later pass:** the Handoff sentence "make sure the doc's final state is committed" is now **load-bearing** in a way it wasn't before — document-level batching means a whole session can sit uncommitted until one departure point, and that sentence is the only backstop against a missed departure. It will look like a class-7 cross-reference to a future dedup sweep. It is not.

## 2026-08-23 — Prompt-waste Stage B wave 2: learning-capture block compressed to pilot shape
- **What:** Learning-capture intro and tail compressed to the pilot-validated wording (`claude/docs/plans/prompt-waste-reduction.md` v4, §3 doctrine + Stage B). Only this block — the broad mode-unconditioned commit-grant paragraph is *not* the shared boilerplate shape and stays for Stage C2.
- **Removed (class 5/6, already on record):** the tail's inbox-replacement sentence ("This replaces the earlier `kaizen/inbox.md`-append convention…") and ", exactly like the old inbox was" — this file's 2026-08-21 inbox-deletion entry; the intro's ":Agent node it's `PRODUCED`-linked to" mechanics restatement — the mechanics live in the Cypher template directly below.
- **Gate (a) inventory — all preserved:** capture trigger (durable, non-obvious fact in discipline), full Cypher template + `mcp__cypher__query` call line verbatim, "skip task-specific details and anything already documented", "raw capture: `cobb` reads/verifies/promotes; never edit your own agent definition".
- **Verified:** `audit-team.sh` PASS; cobb §7 lint pass.

## 2026-08-23 — K-009 refinement: dropped "decision cluster closes" / topic-switch as independent commit triggers
- **What:** immediately after the resolution below shipped, the stakeholder reviewed the actual
  wording and flagged that the shipped rule still let a **topic switch within a single document**
  count as "switching away" — functionally the same over-triggering as the old rule's "a cluster
  of related decisions closes," since both fire many times inside one document. Rewrote the same
  `tico.md` bullet: the only commit triggers now are leaving the document entirely (switching to a
  *different* document), stepping out of Mode 1, or the interview/session closing — settled
  threads and decision clusters accumulate uncommitted no matter how many close along the way.
  Corrected the bullet's own claim that single-document behavior "collapses to the same as
  before" — it doesn't; single-document sessions now commit less often too, not just
  multi-document ones, since decision/topic-level pauses no longer independently trigger a commit.
- **Why:** direct stakeholder correction, same session — "this is not a good point, occurs too
  often -> a cluster of related decisions closes." Acted on immediately rather than treated as a
  future backlog item, since it's the stakeholder narrowing their own just-made K-009 decision,
  not a fresh open question needing another round-trip.
- **Plan items:** none — refinement of the same-day K-009 resolution above, K-009 stays closed.

## 2026-08-23 — K-009 resolved: stakeholder chose document-level commit batching
- **What:** put K-009's open question directly to the stakeholder (`AskUserQuestion`, four
  options: document-level batching, a time-boxed cap, session-end/checkpoint-only, or "keep the
  current rule, the 17-commit session was a one-off"). Stakeholder chose **document-level
  batching**: one commit per document per natural pause, not per thread/decision-log line, even
  while several documents are being interleaved in the same session. Rewrote Mode 1's "Commit as
  you go" bullet in `tico.md` accordingly — the unit of commit is now the document, not the
  thread: settled threads within a document accumulate uncommitted, and a commit fires only when
  tico is about to switch away from that document (to a different document/topic/mode) or the
  session/interview is closing. Explicitly noted this collapses to the old per-pause behavior in
  a single-document session and only changes the cadence when several documents are interleaved.
  Marked **K-009 ✅ done** in `plan.md`, moved out of the active table.
- **Why:** K-009 (opened same day, prior entry below) was deliberately left open rather than
  guessed a second time — the 2026-08-19 tightening had already tried "batch at natural pause
  points" for this exact complaint and it didn't hold under a higher-multiplicity (six-document)
  session, so a second unilateral rewrite risked repeating the same failure or overcorrecting.
  This entry is that stakeholder input landing.
- **Plan items:** K-009 closed.

## 2026-08-23 — `kaizen_team` distillation: 2 entries (tico's per-agent `kaizen/inbox.md` no longer exists — both were legacy graph entries)
- **What:** `cobb` read tico's entries in the shared `kaizen_team` graph (legacy `author`-shape
  query — both entries predate M8's `:Agent`/`PRODUCED` edges; the current-shape query returned
  zero rows, confirming neither is post-M8). Two entries, both dated 2026-08-22:
  1. **`a3f5e6d2-9c1b-4e77-8a2f-6b1d0c9e4f31` — promoted to the prompt.** Fact: a blind
     `replace_all` during an in-session terminology reversal (`falkor-chat/docs/requirements/
     document-ingestion.md`, "file" → "document") also corrupts meta-sentences that *describe*
     the swap ("preferred term is X, not Y" → "preferred term is X, not X"), not just the target
     product nouns. **Re-derived, not just cited:** `git show cd0b300` (the commit that actually
     landed the reversal) shows the terminology note and its decision-log line already correctly,
     narratively rewritten by hand ("briefly used 'file' ..., then reverted to 'document' ...") —
     no committed version of the file ever carried the broken form, confirming the corruption was
     caught and hand-fixed *before* commit, exactly as the entry describes. Added one bullet to
     Mode 1 (after "Write as you go"): rewrite a terminology-swap's own meta-sentences by hand,
     `replace_all` only the product-noun occurrences elsewhere.
  2. **`e1f2a3b4-7c6d-4e8f-9a1b-2c3d4e5f6a7b` — kept open, not re-promoted.** Fact: tico
     committed too frequently again (near one commit per small edit) in a six-document capability-
     family interview, despite the 2026-08-19 fix that already retuned "Commit as you go" for
     this exact stakeholder complaint. **Re-derived:** `git log` on
     `falkor-chat/docs/requirements/` for 2026-08-22 shows 17 commits in ≈51 minutes (19:03–19:54),
     several under 2 minutes apart, across interleaved document-ingestion and business-entities-
     family threads — confirms the complaint. Not re-promoted as a second silent prompt tweak:
     the first tightening was tried for this exact complaint and didn't hold under a
     higher-multiplicity (many-document) session, so a second unilateral guess risks the same
     failure or overcorrecting the other way. Opened **K-009** in `plan.md`, modeled on K-008's
     "flag, don't guess" pattern — needs the stakeholder's actual tolerance, not another inferred
     rewrite.
  Both graph nodes cleared after this entry and the K-009 backlog item were written (legacy
  entries — `DETACH DELETE` by `entryId`, curator-gated, `agent='cobb'`).
- **Why:** requested distillation pass ("distill tico inbox") — tico's own `kaizen/inbox.md` file
  no longer exists (removed 2026-08-21, agent-maintenance skill §5); its raw capture lives in the
  shared `kaizen_team` graph instead, so this pass read that graph directly.
- **Plan items:** K-009 opened (kept open, entry 2); entry 1 fully promoted, no K-item needed.

## 2026-08-21 — Team-wide follow-on: universal interactive-mode commit grant (does not change tico's own grant)
- **What:** Immediately after the entry below landed, the stakeholder — asked how to route the
  commit request for *this* change — ruled further: every agent, not just tico/teco, should get
  an interactive-mode commit exception. tico's own grant (below) is unchanged by this — it was
  already unconditioned on interactive-vs-subagent mode and already covers more than the new
  universal grant does — but the Bash guardrail bullet gained one trailing sentence clarifying the
  relationship (yours applies either way; the new universal one is interactive-only and narrower).
  Full team-wide implementation (all 13 agents, `claude/AGENTS.md` rewrite, `audit-team.sh`
  check-8 redesign): `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Why:** direct stakeholder ruling, same session, same day — see also the user's own correction
  that agents "should not refuse when asked by the stakeholder," which shaped how this and the
  preceding request were handled (act on a direct stakeholder ask rather than routing it through
  another pause).
- **Verified:** `bash claude/scripts/audit-team.sh` — clean, all 13 agents pass check 8.
- **Plan items:** none opened — direct implementation of an explicit stakeholder decision.

## 2026-08-21 — Commit authority extended to a Mode-3 verification pass's returned artifact (`kaizen_team` distillation + live stakeholder decision)
- **What:** tico's `git add`/`git commit` grant now covers a second case beyond its own doc
  kinds: the returned artifact of a `qa-engineer`/`analyst` verification pass tico itself offered
  under Mode 3 and the stakeholder accepted, once tico has read it and confirmed it fits, by
  explicit path — mirroring `teco`'s integrator authority (step 5: verify a coordinated
  specialist's deliverable, then commit it) but narrower in scope: only the one
  ad-hoc-orchestration case tico's own guardrails actually sanction (the offered Mode-3
  verification pass), not any specialist's deliverable in general. Edited: `tico.md`'s Bash
  guardrail bullet (now names both grants explicitly) and Mode 3's verification-pass bullet (now
  says to commit each returned artifact once confirmed to fit); `claude/AGENTS.md`'s "Git-commit
  authority" section and `claude/README.md`'s tico catalog row updated to state the extension and
  cross-reference this entry. `claude/scripts/audit-team.sh` unaffected (check 8 only verifies
  *which* agents claim commit authority in their own prompt, not its scope) — re-ran clean, same
  113 PASS as before.
- **Why:** a `kaizen_team` entry from tico (`entryId` `e7a1c9d4-3f2b-4a6e-9c1d-8b5f0a2e6d71`, dated
  2026-08-21) reported that closing out `docs/manuals/graph-ontology.md`'s verification pass —
  tico directly delegated to `devops` (env verify), `analyst` (→ `docs/reviews/graph-ontology.md`),
  `qa-engineer` (→ `docs/test-plans/graph-ontology.md` + `docs/test-reports/graph-ontology-report.md`),
  and `graph-dba` (dropping two stale kaizen graphs) — left the analyst/qa-engineer artifacts
  untracked, since tico's Bash guardrail at the time only let it commit files its own Write/Edit
  guard covered. Stakeholder feedback captured in the entry: tico should get the same commit
  permit teco has and, as orchestrator of that pass, should own committing what it commissioned,
  not just its own document. Verified live: re-read `tico.md`'s prior Bash bullet and `teco.md`'s
  integrator grant (Guardrails, "The grant"/"Why the boundary differs from `tico`'s") side by
  side — confirmed the asymmetry was real (teco's grant is scoped to its integrator *role*; tico's
  was scoped to its Write/Edit *guard*, which never covered a delegate's own deliverable). Put to
  the stakeholder directly (this is a scope-of-authority judgment call, same class as K-008,
  `cobb`'s not to guess) via `AskUserQuestion`: ruled **extend tico's commit authority**, scoped to
  the ad-hoc-orchestration case, mirroring teco.
- **Does not reopen the 2026-07-30 closed ruling** (parking lot, below): that ruling was about
  whether a *third* agent (analyst/qa-engineer/etc.) gets its own commit rights — still closed, no
  change here. This widens only `tico`'s existing, stakeholder-granted scope to catch up with a
  case its guardrails already sanction (the Mode-3 offered verification pass) but its commit grant
  hadn't yet been written to cover.
- **Relationship to K-008 (`plan.md`):** related but distinct — K-008's two open incidents (routing
  a QA finding to `architect` as a design question; committing an arbitrary one-off `BACKLOG.md`
  entry outside tico's doc kinds) are neither shape considered here and remain unresolved. K-008
  stays open; a cross-reference note was added to it rather than closing it.
- **Kaizen-graph distillation:** entry `e7a1c9d4-3f2b-4a6e-9c1d-8b5f0a2e6d71` — **promoted to the
  prompt** (see What, above). Cleared from `kaizen_team` after this history append landed
  (curator `DETACH DELETE` via `cobb`, append-before-clear order per the agent-maintenance skill
  §5).
- **Plan items:** K-008 extended with a cross-reference note, not closed.

## 2026-08-21 — `kaizen/inbox.md` deleted (content already fully captured elsewhere)

- **What:** `cobb` deleted this agent's frozen `kaizen/inbox.md` (git history retains it in full, unaltered) as part of a team-wide cleanup of all 12 agents' frozen inboxes.
- **Why:** user-directed — "no point keeping [it] since it's already git history." Verified lossless first: `kaizen_team` (the shared graph every agent's raw capture routes through since 2026-08-20) was confirmed completely empty before any deletion — every entry any agent ever wrote there (including this agent's own distillation, immediately below) has already been distilled and cleared — and this file's own pre-migration content was already imported into the graph system verbatim back on 2026-08-20 (see that date's entry). Full rationale and verification method: `claude/cobb/kaizen/history.md`, 2026-08-21 entry.
- **Verified:** see `cobb`'s entry (cross-agent verification, not repeated per file).
- **Plan items:** none opened — pure cleanup, no behavior change.

## 2026-08-21 — `kaizen_team` graph distillation: 1 entry — collision-rule-5 gloss flagged for `architect`, folded into K-005
- **What:** Processed tico's one pending `kaizen_team` entry (`entryId`
  `b2f1c8b0-6d1a-4e2a-9c3d-7a1f0e5d9c21`, dated 2026-08-20): "root `AGENTS.md` collision rule 5
  does not turn on a requirements doc's own Status flip alone — the real test is whether anything
  downstream has actually executed against the specific content being changed," captured during
  the `generic-cypher-mcp2` plan's `T1` unit (superseding FR-4/AC-3/FR-14/AC-11 in place rather
  than forking a successor document).
- **Verified (re-derived, not just cited):** read `docs/requirements/generic-cypher-mcp2.md`'s own
  2026-08-20 Decision-log entry in full and `docs/plans/generic-cypher-mcp2.md`'s `T1` row plus the
  precedent paragraph directly above it. Both independently confirm the instance and the reasoning
  verbatim — the document had reached its "Ready for design" **gate** on 2026-08-19, but nothing
  had since **executed** against the reversed FR-4/FR-14 content (no unit ever ran the deletion the
  gate would have authorized), so revision in place — not a successor document — was the correct
  call under root `AGENTS.md`'s collision rule 5. The fact is already fully recorded as an
  instance, with complete reasoning, in both cited project docs — no new edit needed to preserve
  *this instance*.
- **Not promoted as a standalone project-docs edit — folded into K-005 instead:** the entry's real
  ask is that root `AGENTS.md`'s collision rule 5 (and its source spec,
  `docs/plans/doc-reference-convention.md`) states only the literal disjunctive test ("approved,
  **gated**, or executed against" → any true means fork a successor) without this narrower gloss —
  a reached gate alone doesn't force a successor if nothing has executed against the specific
  content changing. A future agent reading the rule cold, without knowing to dig up this precedent,
  could misapply it. That's a convention-text edit to a document `architect` owns
  (`docs/plans/doc-reference-convention.md`, Tracks: C-322) — per this file's own K-005 precedent
  (2026-07-29: formal updates to that document are `architect`'s pass to make, not a side effect of
  a kaizen distillation or an agent-prompt edit), it isn't `cobb`'s to make unilaterally. K-005
  already tracks "an architect pass adding formative content to `doc-reference-convention.md`"
  (there: the `manuals/` doc-kind mention) — extended (see `plan.md`) to also carry this
  collision-rule-5 gloss, so both land in the same architect pass rather than opening a
  near-duplicate second K-item for the same underlying job.
- **Plan items:** K-005 extended, not opened fresh.
- **Graph:** entry `b2f1c8b0-6d1a-4e2a-9c3d-7a1f0e5d9c21` cleared from `kaizen_team` after this
  append landed (curator `DETACH DELETE` via `cobb`, append-before-clear order per the
  agent-maintenance skill §5).

## 2026-08-21 — Coverage fix: dropped stale "your kaizen inbox" from the commit-authority bullet (team certification, §7 fold-in)

- **What:** The Bash bullet's commit-authority grant listed "your kaizen inbox" among the files
  tico may `git add`/`git commit` (alongside its Write/Edit guard's actual two doc kinds). Removed
  the phrase.
- **Why:** Caught during a user-requested full team-coherence certification's §7 lint fold-in —
  two independent problems, not one: (1) `kaizen/inbox.md` has been a frozen historical snapshot
  since the 2026-08-20 graph migration (this file's own Learning-capture section, further down,
  already correctly says so) — nothing will ever make it dirty again, so the grant is dead prose;
  (2) it was never even true as written — this file's own Write/Edit-guard bullet only names
  `docs/requirements/` and `docs/manuals/`, never `kaizen/inbox.md`, so "files your Write/Edit
  guard already allows you to touch" didn't actually cover the thing being granted. The same stale
  phrase pattern was found and fixed the same pass in `teco.md`'s commit-authority grant
  (`claude/teco/kaizen/history.md`, same date).
- **Verified:** `bash claude/scripts/audit-team.sh` — same 113 PASS / 2 pre-existing FAILs before
  and after (diff, not bare gate).
- **Plan items:** none opened — direct fix from a live certification finding.

## 2026-08-20 — Learnings capture migrated to a working-memory graph (`kaizen_tico`), mirroring `graph-dba`; `mcp__cypher__query` granted
- **What:** The "Learning capture" closing-protocol section now writes a `:KaizenEntry` node
  directly into `kaizen_tico` (FalkorDB, via `mcp__cypher__query`) instead of appending to
  `kaizen/inbox.md`. `kaizen/inbox.md` is now a frozen historical snapshot — it had no
  pre-existing entries to migrate; its own header explains the freeze and gives the live-read
  query. Frontmatter `tools:` gained `mcp__cypher__query` — this agent previously had no MCP
  tool access at all, needed now for this capture path. The "session" (not "run") wording and the
  trailing "Your write guard allows exactly this inbox path" clause's removal both preserved
  tico's existing conventions/constraints correctly — the write guard gates `Write`/`Edit`, not
  the `mcp__cypher__query` MCP tool, so it no longer applies to this capture path.
- **Why:** User-directed team-wide redesign ("I will migrate all agents to write their learnings
  to the graph like graph-dba"), reversing yesterday's file-based Learning-capture dedup (entry
  below) — the user determined the whole team should follow `graph-dba`'s existing graph-based
  capture pattern instead of the file-based inbox convention.
- **Plan items:** —

## 2026-08-19 — Learning-capture paragraph de-duplicated against the inbox's own header
- **What:** Trimmed the "Learning capture" paragraph: dropped "(fact, evidence, suggested home; format in the file header)" and "The inbox is raw capture — the team maintainer verifies and promotes entries into prompts, knowledge bases, or project docs" — both already stated verbatim in `kaizen/inbox.md`'s own header template (agent-maintenance skill §5), which the agent necessarily opens to append. Kept: the discipline-specific fact-kind clause, the "session" (not "run") wording matching tico's interactive nature, the inbox path, "skip task-specific details," "never edit your own agent definition," and the write-guard clause. Behavior unchanged.
- **Why:** User-directed prompt-verbosity reduction, item 1 of the parked diagnosis (`cobb/kaizen/plan.md`) — the mechanics were literally duplicated (prompt + inbox header say the same thing), not just similar boilerplate; pointing at the file's own header removes the duplication without losing information, since the agent reads that file to act anyway.
- **Plan items:** —

## 2026-08-19 — Inbox distillation: remaining 3 entries — 1 prompt addition, 2 project-docs additions, 1 follow-up K-item opened on `cobb`
Closes out the inbox (now empty) alongside the same-day commit-cadence distillation below.

- **`AskUserQuestion` hard-rejects <2 options` (2026-08-17):** Verified directly against the
  tool's own JSON schema (`"minItems": 2` on `options`) — still holds, no live test needed.
  **Promoted to the prompt:** the "Offer options when they unblock" bullet (Mode 1 craft
  guidance) now states the ≥2-option constraint explicitly and says to fall back to plain
  conversation for a single-option free-text follow-up rather than inventing a filler option.
- **FalkorDB web console live on `:3000`** (2026-08-17): Verified against
  `falkor-chat/scripts/start_falkordb.sh` (still publishes `-p "${FALKORDB_WEB_PORT}:3000"`,
  default 3000, prints the URL on start) — holds. Already documented in `falkor-chat/README.md`
  (the "Web console" row), so the gap wasn't the fact itself but that agents debugging graph
  state via `cypher-mcp` had no pointer to it from there. **Promoted to project docs:** added a
  "Visual inspection" note to `cypher-mcp/README.md` (ahead of its "Checking and restarting"
  section) pointing at the console and cross-referencing `falkor-chat/README.md` as the source.
  Not a prompt change — this is infra any agent might need, not tico-specific.
- **New-agent proposal is a tico interview, filed under `claude/docs/requirements/`**
  (2026-08-17): Verified — `claude/docs/requirements/security-expert.md` exists, `Status: Ready
  for design`, `Owner: tico`, confirming the precedent. **Promoted to project docs:** added a
  bullet to `claude/AGENTS.md`'s Maintenance rules documenting the convention (tico interview →
  `claude/docs/requirements/<slug>.md` → cobb design), citing the security-expert doc as
  precedent. **Also surfaced a follow-up finding, not itself part of the distillation:** that
  doc has sat at Ready for design since 2026-08-17 with no `cobb` design pass against it yet —
  opened as K-016 in `cobb`'s own `kaizen/plan.md` rather than acted on silently here, since
  designing a new agent is a scope call, not a bookkeeping step.

## 2026-08-19 — Inbox distillation: 1 entry — "Commit as you go" retuned to batch at pause points, not per-edit
- **What:** Verified the 2026-08-19 inbox entry against `git log` for the same-session commit
  run it described (`generic-cypher-mcp2` + `cpg-mcp-rename` interviews): confirmed 12
  `docs(requirements): ...` commits from tico across the two documents, several a single
  decision-log line or one FR tweak apart (e.g. `generic-cypher-mcp2 — scope ... settled`
  immediately followed by `— team-wide query surface, cobb cadence/self-migration settled`).
  Matches the entry's account and the stakeholder's own "you seen to be committing too often."
  Promoted to the prompt: Mode 1's "Commit as you go" bullet now reads "batch at natural pause
  points, not after every edit" — commit when a *thread* settles (a readback, a settled cluster
  of decisions, a topic/mode switch, or session close), staging whatever changed since the last
  commit, rather than after each individual `Edit` call. The "never bundle unrelated files" /
  "never batch two different documents into one commit" invariants are unchanged. Mode 3's
  manual-commit bullet ("after a manual section lands") already batches at section granularity,
  which satisfies the new rule as written — left unchanged.
- **Why:** direct, dated stakeholder feedback captured in the inbox; re-derived (not just
  cited) from `git log` before promoting, per the distillation SOP's verification step.
- **Plan items:** none (fully promoted, not kept open — no K-item needed).

- **What:** `cobb` processed all 4 entries in `tico/kaizen/inbox.md` (§5). (A prior version of this
  entry's header said "3 entries" and omitted the `version`/`defVersion` entry below entirely —
  caught by `analyst`'s review, M-1 in `docs/reviews/kaizen-distillation-2026-08.md`.)
- **Promoted:** the prior-decision-provenance check — grep `docs/requirements/` and `docs/plans/`
  status logs for a decision the new request might reverse, before the first interview question —
  into "Running the conversation → Do your homework silently." This entry had been sitting **held**
  since 2026-07-25 (a note on the entry itself deferred it explicitly to "the next distillation
  pass" — this is that pass).
  - Note: this held-note's own mechanics ("check whether X already landed before trusting a
    holding claim") became a *second*, independently-surfaced finding, this time in `analyst`'s
    inbox — promoted separately into `analyst.md`'s Guardrails, same distillation pass.
- **Discarded as superseded:** the "tico can't honour a commit request on a dirty unrelated tree"
  entry (2026-07-19) — tico's write/commit guardrails were substantially reworked since (the
  2026-07-30 stakeholder decision on git-commit authority, `claude/AGENTS.md`), and the current
  prompt's Guardrails already scope commits tightly enough that this specific scenario can't recur
  as described.
- **Discarded — already tracked:** the 2026-07-31 entry recording that falkor-chat's
  `POST /workflow-runs` request field is `version` while the `WorkflowRun` node property (and most
  conversational docs) say `defVersion` — its suggested project-doc callout
  (`DESIGN.md` §14.4 or `QUERIES.md` §12.12) is already tracked as **K-040** in
  `falkor-chat/docs/BACKLOG.md:1210`, so no new doc edit or backlog item was needed.
- **Not promoted — moved to `plan.md` K-008, flagged for the stakeholder:** the entry recording
  the stakeholder pushing back twice on tico's write/`Agent` boundaries and asking to relax them.
  Two shapes were on the table in the entry itself, neither self-evidently right; this is a
  genuine scope-of-authority judgment call the maintainer shouldn't make unilaterally, so it's
  recorded as an open plan item instead of guessed at.
- **Verified:** `bash claude/scripts/audit-team.sh` clean.
- **Docs touched:** `claude/tico/{tico.md,kaizen/{history,inbox,plan}.md}`.

## 2026-08-09 — K-007 closed: live check confirms the no-`initialPrompt` opening works (language mirroring + mode inference)
- **What:** Ran three genuinely fresh `claude --agent tico -p "<opener>"` one-shot sessions (no `--continue`/`--resume`, no prior context) to validate the same-day removal of `initialPrompt` and the language-mirror rule (previous entry, below):
  1. **English, Mode-2-shaped opener** ("How does the write-guard hook system work for the doc-scoped agents in this repo?") → tico answered fully in English, entered Mode 2 directly (no canned greeting, no forced "which job is this" menu), grounded the answer in the real files (quoted its own frontmatter hook wiring, `guard-tico-doc-writes.sh`'s actual allowed-glob string, the shared `guard-doc-writes.sh` core, even cross-referencing `architect`'s K-003 rationale for why `Bash` isn't covered), included a Mermaid sequence diagram, and closed with a Mode-2-style check-in ("Does that land, or do you want me to go one level deeper on...").
  2. **Portuguese opener**, same question translated → tico answered fully in Portuguese, same Mode 2 shape (own Mermaid diagram, own grounded facts, closing check-in in Portuguese) — natural mirroring held with **zero explicit instruction** telling it to, confirming the removed rule was never doing the useful part of the job; the bug was specifically the ungrounded `initialPrompt` line, not tico's ordinary in-conversation language behavior.
  3. **Genuinely ambiguous opener** ("I'm not happy with how the login screen works.") — could plausibly be Mode 1 (a change request) or Mode 2 (an explanation of current behavior), and no login screen actually exists as shipped code anywhere in the repo. tico did **not** guess a mode and did **not** ask a generic "which mode do you want?" — it first did real homework (grepped the repo, found `salesperson` has no auth UI, `falkor-chat` has a single hardcoded tenant with real auth deferred to backlog item K-016, and one hit was just `kiro/DESIGN.md`'s unbuilt vision), then asked a **concrete, grounded** clarifying question offering three specific interpretations (shape a not-yet-built falkor-chat auth requirement, add a new login screen to `salesperson`, or something outside the repo). This is a stronger result than the plan item asked for — grounding the disambiguating question in investigation rather than asking an abstract meta-question.
  - Verified no side effects: `git status --short` after all three runs showed nothing new under `docs/requirements/` or `docs/manuals/` — Mode 2 (all three landed there, correctly, since none was actually a fresh feature request or manual task) never touches `Write`/`Edit`, so the write-guard hook was never exercised by this check, as expected.
- **Why:** closes K-007, seeded the same day the `initialPrompt`/language-rule removal shipped — validates the fix actually fixed the reported bug (no more premature-language guessing) without breaking mode inference or introducing a worse failure mode (e.g. over-eager guessing, or an unhelpful generic menu question).
- **Plan items:** K-007 done, removed from the active table (per the done→history convention; no rejected/deferred marker needed).

## 2026-08-09 — Dropped the canned `initialPrompt` greeting and the explicit language-mirror rule; mode is inferred, not asked; renamed "first-order" → "interactive"
- **What:** Three related changes. (1) **Removed the `initialPrompt` frontmatter field entirely** — tico no longer opens a fresh main-session with a forced "introduce yourself, then ask which of the three jobs this is" greeting; it now waits for the stakeholder's real opening message. (2) **Removed the explicit "Respond in the user's language (English by default; mirror Portuguese if they write in it)" line** at the end of the prompt — with no more ungrounded first line, there's no turn left where the model has to guess a language before the stakeholder has written anything, so the rule was both the wrong fix and no longer needed. (3) **Rewrote the "Running the conversation" opening** so mode selection is explicitly inference-first: "infer which of the three jobs this session is from the stakeholder's opening message... don't open with a forced menu, and don't default to Mode 1 out of habit," with a direct clarifying question reserved for genuinely ambiguous openers. (4) **Renamed "first-order agent" → "interactive agent"** throughout (description, body, `claude/README.md`, `claude/AGENTS.md`, `claude/teco/teco.md`'s reference to tico, and `skills/agent-standards/claude-code.md`'s generic mechanism section) — "first-order" was cobb's own coinage and read as "primary/foremost" rather than the intended "converses live with a human, not a delegation target for a background task."
- **Why:** the user reported tico's opening line was consistently in Portuguese despite the explicit English-default instruction, and separately questioned whether spending a turn on a canned self-introduction was worth it at all. Root cause, diagnosed live: `initialPrompt` auto-submits as the first *user* turn (per the main-session mechanism), but it carries no real linguistic evidence — nobody has actually "written" anything yet — so the "mirror if they write in it" condition was vacuously false and the literal English default should have won; instead the model most likely leaned on other context in the session (the operator's git identity, a Portuguese-signaling name) to guess a language, overriding the stated default. Rather than trying to patch the instruction to be more forceful, the fix removes the ungrounded first line altogether: no canned greeting means no turn where the model has to guess anything about the stakeholder before they've said a word, and mode selection already had a "don't assume" rule (Mode 2 example) that just needed to become the actual entry point instead of a footnote under a forced-question flow. The self-introduction ("I'm tico...") was also identified as near-pure ceremony — the operator already knows which agent they launched — while the routing question underneath it (which of three modes) is real, necessary work that survives the change, just inferred instead of asked. Separately, the user corrected an inline claim of mine that tico was "the only first-order agent" — teco is also designed to converse with and pause for the human (its own "Hybrid: drives execution but pauses to the user at decision points" framing, unchanged by this pass); the word "first-order" itself, not the roster, was the flawed part, hence the rename (user's pick, via `AskUserQuestion`, over "foreground" and "human-facing").
- **Also promoted:** the `initialPrompt`-plus-language-default gotcha into `skills/agent-standards/claude-code.md`'s main-session section, marked observed/not-doc-sourced, so the next agent author who wires an `initialPrompt` greeting doesn't rediscover the same failure mode.
- **Plan items:** seeded K-007 (live check of the new opening: language, and mode inferred correctly from the first real message).

## 2026-07-31 — Can offer a live demo environment via `devops` (all modes)
- **What:** New bullet in "Running the conversation (all modes)": tico may proactively suggest spinning up a live demo ("want to see this live?") whenever actually seeing the running system would settle something faster than more narration — a feature under discussion (Mode 1/2) or a manual's walkthrough (Mode 3). On acceptance, it delegates to `devops` via `Agent` with a self-contained brief (component + what to bring up); devops orients and boots it — tico never touches Docker/Compose/infra itself. Cleanup is stricter than bring-up: tico always asks the stakeholder before requesting teardown ("should I have devops clean this up now?"), every time, not just once at the start. The Guardrails bullet scoping `Agent` usage was extended to name this third permitted use (alongside Explore sweeps and the Mode-3 verification offer) — still not a delegation of tico's own core jobs. `devops.md`'s Boundaries & handoffs gained the reciprocal note (tico may hand it a demo-environment bring-up/cleanup request, treated like any other caller's lifecycle op, its own destructive-ops gate unchanged). Catalog rows in `claude/README.md` updated for both agents.
- **Why:** user request — tico is stakeholder-facing across all three modes and increasingly the place where "can I just see it?" comes up; rather than tico narrating or a human manually booting something, it can ask devops to bring the environment up and, on confirmation, tear it down again. Two design calls were the user's, not inferred (asked via `AskUserQuestion`): (1) trigger model — tico may **proactively offer**, not just react to an explicit stakeholder request; (2) cleanup discipline — **always confirm before tearing down**, the most conservative of the three options offered (vs. auto-teardown at topic end, or leaving it running until asked).
- **Plan items:** seeded K-006 (live e2e spin of the demo-environment offer/delegate/confirm-teardown loop).

## 2026-07-29 — Mode 3 gains an offered (not forced) verification step
- **What:** Following the team certification's observation that `docs/manuals/` had no independent-review gate (unlike every other doc kind), Mode 3 gained one bullet: before calling a *new or substantially rewritten* manual done, offer the stakeholder a verification pass — `qa-engineer` to walk the walkthroughs against the running app, `analyst` for architectural/factual claims and clarity, or both — spawned via the existing `Agent` tool with a self-contained brief. Explicitly an *offer*, not a self-enforced gate: declined is fine, and a small addendum/typo fix skips the offer entirely (right-sized ceremony). Added a matching Guardrails bullet scoping `Agent` to wide read-only sweeps (Explore) and this one offered pass — not a general delegation license.
- **Why:** user ruling (2026-07-29): reviewer split is qa-engineer/analyst by claim type (behavioral vs. everything else); gate strength is "mandatory via teco coordination, offered in tico's own first-order sessions" — a hard mandatory gate in a live conversational session was rejected as too much ceremony for a first-order agent.
- **Plan items:** resolves the open question logged in `cobb/kaizen/plan.md` (2026-07-29 certification parking-lot entry).

## 2026-07-29 — Two new duties: didactic project explanations + user-manual maintenance
- **What:** tico gained two modes alongside requirements interviews, made explicit throughout the prompt as **Mode 1/2/3**: **Mode 2** answers "how does X work"/"why was Y built that way" about any project aspect, live, in plain language — grounded in the real docs/code (never invented), with clearly-flagged light suggestions allowed on design-shaped questions (stakeholder ruling: speculation is welcome as long as it's visibly opinion, not decided into a requirements/plan doc). **Mode 3** authors/maintains end-user documentation at a new doc kind, `<component>/docs/manuals/<slug>.md`, illustrated with Mermaid diagrams (`flowchart`/`sequenceDiagram`/`stateDiagram-v2`) wherever a picture replaces a paragraph of narration — explicit "skip it when a sentence covers it just as well" guardrail against decorative diagrams. Frontmatter: `description` rewritten around the three modes; `initialPrompt` now asks which job the session is instead of assuming an interview. Write-guard hook **renamed** `guard-requirements-doc-writes.sh` → `guard-tico-doc-writes.sh` (its scope is no longer requirements-only) and its allowed-path globs extended to `docs/manuals/*`; escalation message updated to name both kinds. Subagent-fallback section split: requirements still degrade to one interview round, but an explanation/manual request from a self-contained brief can complete in one pass (no live back-and-forth needed for those). `docs/requirements/`-only language in Guardrails/Handoff updated to cover both owned kinds.
  Root `AGENTS.md`'s documentation convention got a matching extension: `manuals/` added to the per-module `docs/` kind list, called out as the one **end-user-facing** kind (vs. the rest being engineering-process docs), folded into the collision-rule family chain (optional terminal member, only when a manual shadows one feature 1:1) and the owner-by-kind archived-flip table (`requirements/*` and `manuals/*` → `tico`). `claude/teco/teco.md`'s routing table and documentation-impact scan gained a manuals-aware row/bullet: live explanations stay pause-to-user (tico isn't a delegation target), but a manual update from a brief that already states the facts is delegable to tico like any other subagent deliverable. `claude/README.md`'s tico catalog row and `claude/AGENTS.md`'s one-line roster rewritten for the three-mode shape.
- **Why:** user request — tico is already the team's stakeholder-facing agent; the user wanted it to also explain any project aspect didactically and own illustrated user manuals in `docs/`, rather than adding a separate agent for an already-stakeholder-facing concern. Three product decisions were the user's call (asked via `AskUserQuestion`, not inferred): manuals live at a new per-component `docs/manuals/` kind (not repo-root, not an informal loose convention); manuals carry the standard header-block lifecycle (`Status: active`, rarely changing); Mode 2 may offer light, clearly-flagged design speculation rather than staying strictly descriptive.
- **Deliberately not done this pass:** `docs/plans/doc-reference-convention.md` — the large, formally-reviewed spec (Owner: `architect`, Tracks: C-322) that root `AGENTS.md`'s convention was built from — was **not** edited; extending someone else's owned, heavily-proof-structured artifact for an additive change belongs to a dedicated `architect` pass, not a side effect of an agent-prompt edit. Tracked as K-005.
- **Plan items:** seeded K-004 (live e2e spin of Modes 2 & 3) and K-005 (formal convention-doc update, routed to `architect`).

## 2026-07-27 — Unpinned from `model: opus` (team-wide)
- **What:** Removed the `model: opus` frontmatter line. The field is now absent, so the agent runs on Claude Code's default — `model` **defaults to `inherit`** (re-verified 2026-07-27 against `code.claude.com/docs/en/sub-agents`), i.e. the model the session/system default selects. No other frontmatter or body change.
- **Why:** User no longer wants the team locked to Opus. Model choice belongs at the session level (one decision, changeable with `/model`), not duplicated across 13 frontmatter files where it silently overrides whatever the user picked.
- **Plan items:** —

## 2026-07-27 — Requirements-doc template adopts the canonical bolded header block (step 2 of `docs/plans/doc-reference-convention.md`)
- **What:** One line changed — the status line inside the `Structure:` template of the deliverable section. `> Status: Interviewing | Ready for design · Last updated: YYYY-MM-DD` becomes `> **Status:** Interviewing | Ready for design · **Owner:** \`tico\` · **Tracks:** <id(s)> (<M<n>>) · **Last updated:** YYYY-MM-DD`. Nothing else in the prompt moved: no frontmatter, no hook, and — the step's explicit proof obligation — the readback bullet that flips `Status` to **Ready for design** only on the stakeholder's explicit confirmation is untouched, as is `claude/README.md`'s user-facing statement of that promise (catalog row re-checked, no edit needed: it cites the write guard and the gated flip, not the template's form).
- **Why:** `docs/plans/doc-reference-convention.md` v1.4 §9.6, blocker **B4**. The repo-wide header block is now normative in root `AGENTS.md` and its canonical form is **bolded labels**; `tico`'s template was the unbolded dialect, and every requirements document written from it inherited that form, so the convention's own conformance regex could not have passed the three files it was told not to change. The convention absorbs `tico`'s two `Status` **values** verbatim rather than renaming them to lowercase tokens (§9.6, blocker B3) — the gated transition is product behaviour, not the architect's to change — so only the label's asterisks change, plus the two new fields (`Owner:`, `Tracks:`) the block requires. **Bolding a label is a form change, not a value change.**
- **Plan items:** none.

## 2026-07-24 — Description slimmed further (second team-wide token-cost pass)
- **What:** Frontmatter `description` compressed 593 → 538 chars (-9%): tightened phrasing, dropped restated detail. `tico` has no boundary pairs in `claude/scripts/audit-team.sh`; full audit re-verified green regardless. No body/catalog change.
- **Why:** All 13 agents' descriptions are auto-injected into every session and subagent spawn; the roster grew to 13 (graph-dba, joern added) since the first pass on 2026-07-11, and per-agent `/context` output showed room to cut further. User-requested via a `/context` token audit.
- **Plan items:** none.

## 2026-07-24 — Frontmatter: `permissionMode: acceptEdits`
- **What:** Added `permissionMode: acceptEdits` to the frontmatter, matching the same-day change across the team (`coder`, `tdd-engineer`, `frontend-engineer`, `architect`, `qa-engineer`, `analyst`, `devops`, `graph-dba`, `joern`, `teco`). File-edit/write approvals are session-scoped in Claude Code (unlike Bash approvals, which persist permanently per repo+command), so users otherwise have to re-grant write permission every session even with a global `Edit`/`Write` allow rule in `~/.claude/settings.json`.
- **Why:** Verified against current Claude Code docs (`hooks-guide.md` "Hooks and permission modes") that this is safe: `PreToolUse` hooks fire *before* any permission-mode check, and a hook's `"ask"` decision still forces the prompt even under `acceptEdits`/`bypassPermissions`. `tico`'s `guard-requirements-doc-writes.sh` hook (escalates to ask on any Write/Edit outside the allowed requirements-doc paths) keeps working exactly as before; only writes it would already let through silently stop re-prompting every session.
- **Plan items:** none.

## 2026-07-23 — Tico may commit its own deliverable
- **What:** The Bash guardrail no longer bans all tree mutation — it now carves out `git add`/`git commit`, scoped to exactly the paths the Write/Edit guard already allows (the requirements doc(s), the kaizen inbox), staged by explicit path only (`git add -A`/`.`/`commit -a` still forbidden). Added a "Commit as you go" bullet alongside "Write as you go" and a note in the Handoff section to commit the doc's final state before closing.
- **Why:** User ruling: they treat requirements docs as code and want tico to version its own files as part of authoring them, not leave commits to a human or downstream agent. This is a deliberate narrowing of the prior full Bash-mutation ban (see the 2026-07-09 creation entry and `guard-doc-writes.sh`'s design note on architect K-003) — not a hook change: Bash mutation stays prompt-guarded by convention, only the *scope* of what's permitted moved. No push/reset/rebase/amend; no bulk-staging flags.
- **Plan items:** none.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 802 to 585 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-requirements-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/requirements/*|*/docs/requirements/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/tico/hooks/guard-requirements-doc-writes.sh`) to `$HOME/.claude/agents/tico/hooks/guard-requirements-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/tico` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

## 2026-07-09 — Redesigned as a first-order conversational agent
- **What:** tico now runs as the **main-session agent** (`claude --agent tico`) — verified 2026-07-09 against `code.claude.com/docs/en/sub-agents`: the main thread takes on the definition's prompt/tools/model, frontmatter hooks still fire in main-session mode, and `initialPrompt` auto-submits as the first user turn. Prompt rewritten around a **live interview**: one thread at a time, reflect-back confirmations, `AskUserQuestion` (added to `tools`) for option picks, the doc updated as the conversation progresses, and a readback + explicit stakeholder confirmation gating the "Ready for design" flip. The round-based protocol shrank to a degraded subagent fallback ("If you are invoked as a subagent anyway"). teco no longer delegates to tico — it consumes the doc by path and pauses to the user when requirements need capturing.
- **Why:** User ruling on the initial design: tico is not a subagent but a first-order agent meant to be conversational — the rounds protocol optimized for the wrong constraint.
- **Plan items:** K-002 ⚪ rejected (continuation machinery moot in first-order mode); K-001 retargeted to a live `claude --agent tico` session.

## 2026-07-09 — created
- **What:** initial version of `tico` — conversational product-owner subagent that interviews the user/stakeholder about a feature request and owns the feature requirements document (`<component>/docs/requirements/<slug>.md`). Round-based interview protocol (subagents can't `AskUserQuestion`, so each invocation folds answers into the doc and returns the next question batch as its deliverable; the doc is the durable state between rounds). Write/Edit scoped to the requirements doc, harness-enforced by `hooks/guard-requirements-doc-writes.sh` (same PreToolUse "ask"-escalation pattern as architect/teco/devops). Model `opus`; tools mirror the architect's investigation set.
- **Why:** the team had design (architect) through delivery (coder/tdd/qa) covered, but nothing upstream capturing WHAT/WHY before the architect decides HOW — vague feature requests went straight to design. Requested by the user 2026-07-09.
- **Plan items:** seeded K-001 (e2e spin), K-002 (SendMessage rounds), K-003 (FR-id traceability).
