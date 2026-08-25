# Prompt & output waste reduction — agent team

> **Status:** active · **Owner:** `claude` · **Tracks:** — · **Version:** 5

*Rev 2 (2026-08-23): added live-deployment ground rules, per-unit rollback machinery, breakage
detection/abort criteria, staggered Stage B, two-pass rule for the heaviest cut.*
*Rev 3 (2026-08-23, pilot calibration — stakeholder ruling): class-5/6 material is deleted
outright, no inline dated pointers; only normative citations (specs/templates the rule uses)
stay. §3 table and Stage B block dispositions updated accordingly.*
*Rev 4 (2026-08-23, stakeholder ruling): commit granularity is one **complete unit** per commit
(all files the unit touched — prompts, kaizen histories, catalogs — landing and rolling back
together), not one file per commit. §4.0 and Stage B updated.*

*Rev 5 (2026-08-25): added Stage G — living-document compaction. Stages A–F bound the register
agents write in; nothing bounds the documents they write into, and the convention has 0 subtractive
verbs. Filed, not executed: §4.0's clean-tree precondition is unmet and the amendment is
stakeholder-gated.*

Reduce verbosity at its two roots — narrative-laden agent prompts (the register the agents
imitate) and accretion-friendly output conventions — without losing a single behavioral rule.
Guiding invariant, stated by the stakeholder: **history has its own dedicated file
(`kaizen/history.md`); the prompt captures only the essence.**

## 1. Goal & scope

**Goal.** Every always-loaded agent prompt carries rules and mechanisms, not stories; deliverable
conventions stop documents growing without bound. Measured outcome: prompt corpus shrinks
substantially (est. 25–45% per heavy file) with **zero rule loss**, verified per file.

**In scope:** the 13 agent prompts (`claude/<name>/<name>.md`), the three shared context files
(root `AGENTS.md`, `claude/AGENTS.md`, `falkor-chat/AGENTS.md`), small output-discipline additions
to `architect.md`/`analyst.md`, **two** convention amendments (review accretion, Stage E pass 3;
living-document compaction, Stage G), and a durable ratchet guard in the `agent-maintenance` skill.

**Out of scope:** frontmatter `description` fields (routing contracts — touching them cascades
into `claude/README.md`/roster updates for no waste win); on-demand knowledge bases
(`falkordb-quirks.md`, `review-techniques.md`, etc. — fact-dense, loaded only when needed);
`Status: archived` documents (immutable by convention); rewriting existing falkor-chat docs
(history — only *future* output changes). *Scope note (2026-08-25): `falkor-chat/docs/DESIGN.md`
and `BACKLOG.md` were rewritten anyway, outside this plan (`3e2b378`, `0f48b8a`, `88bb71b`) —
which is what surfaced Stage G. Those two instances are done; Stage G is the rule that stops them
recurring, and its own G2/G3 units are the remaining instances.*

## 2. Findings this plan rests on (measured 2026-08-23, this session)

- Verbosity is agent-specific, not model-wide: `tico`/`qa-engineer` outputs are lean;
  `architect`/`analyst`/`graph-dba`/`data-scientist` outputs are not.
- Style transfer is the primary cause: plans average 36.9 words/sentence vs. 26–33 in the
  prompts/AGENTS.md that form their context; em-dash and bold density match. `architect.md`
  already says "don't pad" and `analyst.md` "prune ruthlessly" — both are outweighed by the
  surrounding register.
- The kaizen ratchet: `teco.md` grew 915→5,987 words (Jun→Aug), `architect.md` 851→1,738,
  `analyst.md` 1,685→2,569 — lessons land with inline incident narratives and never compress out.
- The narratives are already duplicated: spot-checked cuts (458k-token dispatch, silent
  `general-purpose` trap) exist in full in `claude/teco/kaizen/history.md` (:110, :136, :427).
- Precedent: the method below was already applied once, successfully — `teco` history :310
  (2026-08-11): rule kept verbatim, narrative → dated pointer, −85 words.
- Output-side amplifiers: `docs/reviews/llm-provider-config.md` grew 572→2,455 lines by appending
  six implementation reviews that the `-impl` convention (`analyst.md:57`) says belong in a
  separate file; one graph-dba decision is restated 4× (~600 words) inside
  `docs/plans/document-ingestion.md`.

## 3. The editing doctrine (the safety mechanism)

Every sentence in a prompt is classified before it is touched. **Default when uncertain: keep** —
a kept narrative costs words; a lost rule costs a repeated incident.

| # | Class | Test | Disposition |
|---|---|---|---|
| 1 | **Rule** | Imperative the agent must follow ("always pass explicit `subagent_type`") | Keep, verbatim or tighter |
| 2 | **Mechanism** | Needed at run time without extra reads: Cypher templates, ledger columns, verdict scales, exact tokens/paths | Keep inline |
| 3 | **Why-essence** | The one clause that makes a counterintuitive rule stick ("omission spawns `general-purpose` silently — no error, no hooks") | Keep, ≤1 clause |
| 4 | **Consequence detail** | Secondary failure mode ("a later `SendMessage` resume inherits the wrong identity") | Keep only if it changes what the agent does; else class 5 |
| 5 | **Incident narrative** | The story: what happened, which units, what it cost, who caught it | **Move**: verify present in that agent's `kaizen/history.md` (append first if absent), then **delete — no inline pointer** (stakeholder calibration, pilot: dates/file pointers contaminate context; the history file is the standing, greppable home) |
| 6 | **Provenance/governance** | "Stakeholder decision, 2026-08-21", supersession trails, "this replaces the earlier convention…" | **Delete** (after the same verify-in-history step). Non-negotiability is expressed by stating the rule absolutely, never by citing authority or dates. Distinguish from a **normative citation** — a path the rule requires the agent to *use* (a spec like the `CPG:` forms' source, a template location) — which stays |
| 7 | **Duplicate restatement** | Same rule stated twice in one file; cross-references that re-explain instead of pointing | One canonical statement + short cross-ref. **Test: not "is it said twice" but "is it *needed* twice"** — name the decision point that loses it and ask whether the agent stands there (C1 finding 5) |

**Class 6 is discharged across this team (closed at C6, 2026-08-24 — finding 14).** All thirteen
agent prompts measure **0 w of removable class-6 residual**; eleven were measured directly at
C3–C6, the rest carry none by construction after Stage B. The class stays in this table because it
governs *new* writing and any prompt this lab adds later — but for the existing team the sweep is
**complete, not merely current**. Three certified exceptions remain, each an exception by rule:
`cobb.md`'s `verified 2026-07-10 against …` stamp (mandated by that file's own Drift-resistance
principle, and needed a second time in the `agent-maintenance` skill for a load-set reason the C6
lint verified), `cobb.md`'s `pre-M8` query-shape discriminator, and `graph-dba.md`'s `(successor to
RedisGraph)` anti-trigger. **Operationally: name a file's citation habit, cut it, and class 6 is
done permanently. A sweep that opens with a provenance grep over an existing prompt is re-running a
search whose answer is known to be empty — go straight to the class-7 semantic read.**

**Anti-trigger vs. supersession trail — the class-1/class-6 boundary (added at C2).** A sentence
that says *X is not a trigger* looks like a reference to a superseded rule and is often deleted as
class 6. The test: **would the agent plausibly infer that trigger with the sentence absent?**
*Yes* ⇒ the sentence is a live **anti-trigger** (class 1, keep) — e.g. "not as each section lands"
in a manual that is section-structured, or "a topic switch within a document is not a departure."
*No* ⇒ it exists only as an artifact of the old rule's wording (class 6, delete) — e.g. "commits
far less often than per-topic," where the per-topic trigger has no independent source. Both calls
occurred in the same C2 bullet, in opposite directions.

Hard preservation constraints (mechanically checked, not judgment):

- `audit-team.sh` check 8 greps every agent file for `` `git add`/`git commit` `` **and** the
  case-insensitive phrase **"delegated subagent"** — both tokens must survive in every prompt.
- `audit-team.sh` :118 requires `teco.md` to name **every** agent — the roster words stay.
- The verbatim `CPG:` line contract (`docs/plans/cpg-agent-adoption.md` §3) is a cross-document
  interface — the required forms stay exact; only the prose *around* them may compress.
- Hook wiring (frontmatter `hooks:` blocks) is untouched.

## 4. Step-by-step

### 4.0 Execution ground rules (live deployment)

**The working tree is production.** `~/.claude/agents/<name>` symlinks into
`claude/<name>/` in this repo (verified 2026-08-23) — a saved prompt edit is live for the next
agent session *before* any commit. Therefore:

- **Clean tree before each unit** — no unrelated uncommitted changes, so a single-file revert is
  always trivial and unambiguous.
- **One complete unit per commit**, committed promptly after gates (a)–(e) pass — a commit is a
  complete set: every file the unit touched (prompt edits, their `kaizen/history.md` entries,
  any catalog updates) lands together, so it also rolls back together. A unit may span several
  agents (a Stage B wave is one unit); what's forbidden is splitting one unit across commits or
  bundling unrelated work in. The commit is the rollback unit.
- **Complete each file's edit in one sitting** — never leave a prompt half-edited across a pause;
  a half-edited prompt is a live, untested hybrid.
- **Revert procedure** (the "broken agent" answer): `git checkout <last-good-sha> --
  claude/<name>/<name>.md` restores the exact prior prompt instantly — content-safe because the
  history-first gate guarantees nothing was ever destroyed, only moved. Re-run `audit-team.sh`
  after any revert.

Each unit's gates (all, every unit): **(a)** rule inventory — before editing, list every
class-1/2 clause in the old file; after, map each to its surviving location; unmapped ⇒ restore.
**On a unit touching more than one file, derive the "after" half by re-grepping every moved token
across every file in the unit — map from the grep output, never from the edit list** (finding 17).
**(b)** history-first — every class-5/6 clause removed is confirmed present in
`kaizen/history.md` *before* the prompt edit is committed (append if missing).
**(c)** `cobb`'s single-artifact prompt-quality lint (`agent-maintenance` skill §7) on the result.
**(d)** `./claude/scripts/audit-team.sh` green.
**(e)** the dated `kaizen/history.md` compression entry (standing maintenance rule) **carries the
gate-(a) inventory mapping and the list of moved clauses** — persisted where a later incident
investigation will look, not left in session context.

### Stage A — Pilot: `architect.md` (1,738 w → ~1,250)

One representative mid-size file. Apply §3; expected cuts: Guardrails commit-grant paragraph
(governance history → deleted), Learning-capture tail ("this replaces the earlier inbox
convention…" — pure class 6), CPG-clause surrounding prose. **Stakeholder reviews the diff before
Stage B** — this is the calibration point for how aggressive the doctrine reads in practice.

### Stage B — Fleet boilerplate sweep (staggered, never one batch)

The pilot already exercises all three boilerplate blocks on `architect.md` — Stage B reuses the
pilot's validated compression shape, it does not invent one. Rollout in **two waves**: wave 1 =
three agents (one per discipline shape: an implementer, a reviewer, an advisory specialist),
then an observation window (§6); wave 2 = the remaining nine only after wave 1 shows no
regression. Each wave is one unit = one commit (§4.0).

The three blocks near-identical across agents, compressed identically:
1. Learning-capture tail — keep the Cypher template (class 2) + "raw capture; `cobb` promotes;
   never edit your own definition"; drop the inbox-removal history (~50 w × 13).
2. Interactive-commit-grant paragraph — keep grant scope + never-list + carve-out phrase
   (check-8 tokens); provenance → deleted.
3. CPG-freshness clause — keep the rule ("teco-issued brief states freshness: take it as given");
   drop the decision retelling.

*Wave-1 executed 2026-08-23 (`coder`, `analyst`, `data-scientist`); cobb lint pass ×3. Lint
carry-forwards:* **wave 2 adds a fourth micro-shape** — the shared freshness clause's
hanging-topic construction ("a `teco`-issued brief that states…, take it as given") becomes
"when a `teco`-issued brief states the graph's freshness, take it as given", applied uniformly
to all four already-compressed files (`architect` included) plus any wave-2 file carrying the
clause; *and C3 picks up two pre-existing analyst waste candidates* (the traps-list "have bitten
a review before" framing; the placeholder-token bullet's incident narration).

*Wave-2 executed 2026-08-23, stakeholder-directed ahead of wave 1's observation-window close
("next wave please"). All nine remaining agents, blocks as present per file: `tdd-engineer`/
`frontend-engineer`/`qa-engineer` all three blocks + micro-shape; `security-expert` freshness +
grant + capture-intro (tail already compressed — postdates the inbox era); `graph-dba`/`devops`/
`cobb` grant + capture; `teco`/`tico` capture only — their non-standard broad commit-grant
paragraphs and teco's centralized freshness paragraph are deliberately deferred to C1/C2.
Retro micro-fix applied to `architect`/`coder`/`analyst`. Gates (a)–(e) green; audit PASS.
Both Stage B observation windows now run concurrently.*

### Stage C — Heavy singles, descending payoff

| Unit | File | Now | Target | Notes |
|---|---|---|---|---|
| C1 | `teco.md` | 5,987 w | **done: 5,377 w** (both passes) | Largest and most narrative-dense. **Two passes, not one** (rule: any file with a >30% projected cut): pass 1 = unambiguous class-5/6 cuts only (narratives, provenance); pass 2 = class-7 dedup (judgment-heavier), only after pass 1's observation window closes clean. Each pass is its own commit/rollback unit |
| C2 | `tico.md` | 3,627 w | **done: 3,503 w** (one pass) | Ran as a single pass — the inventory found ~no class-7 duplication. Also un-staled Mode 3's commit rule against K-009 |
| C3 | `analyst.md` | 2,569 w | **done: 2,473 w** (one pass) | Evidence-traps list stayed (class 3/4 payloads); trims were provenance and restatement. **Measured at its editorial floor** — residual class-6/7 inventory across the whole file is under 25 w. Structural lever routed to `analyst` K-003 |
| C4 | `security-expert.md` + `devops.md` | 2,437 + 2,212 w | **done: 2,357 + 2,110 w** (one pass) | Both cuts were incident/origin provenance, not rules. `devops`'s repo-specific orientation example **deleted on its third rot** rather than refreshed a third time. **Both measured at their editorial floor** — residual 22 w each. One pre-existing MAJOR routed to `security-expert` K-005 rather than bundled |
| C5 | `tdd-engineer.md` + `qa-engineer.md` + `data-scientist.md` | 2,163 + 2,094 + 2,098 w | **done: 2,154 + 2,071 + 2,083 w** (one pass) | Five edits; class-6 residual now **zero in all three files**. One was a deliberate rule change — `qa-engineer`'s test-report path no longer offers ", or the component's convention", which the same file twice calls non-negotiable. **All three at their editorial floor**; `qa-engineer` is the first file to land *above* the floor band, for a structural reason. Three pre-existing minors routed to `tdd-engineer` K-006 and `qa-engineer` K-005/K-006 rather than bundled |
| C6 | `cobb.md` + `graph-dba.md` + `frontend-engineer.md` + `coder.md` | 2,006 + 1,807 + 1,604 + 1,240 w | **done: 1,976 + 1,806 + 1,600 + 1,240 w** (one pass) | Five edits, **four** of them recommended cuts — the split rule fired a second time and was discharged C4-style, by deferring half of one candidate. `cobb.md` deliberately last (§7 risk register) and the only unit linted by a **reverted-to-HEAD** copy of its own agent, which closes F1. Two edits were declared corrections, not compression: `cobb.md`'s lint bullet said **six** dimensions where its own skill says seven, and one always-on skill description asserted the existence of files deleted in `6fdc107`. `coder.md` took **zero** edits — inventoried and judged at floor. **All four at 0 w removable class-6**; four pre-existing minors routed to `coder` K-003/K-004, `frontend-engineer` K-003, `cobb` K-020 |

Serialized or in review-gated pairs — never two units editing one file.

**Stage C runs one pass by default (set at C2, 2026-08-24).** The two-pass rule was keyed to
projected cut size (>30%), which finding 4 already retired as a predictor. The split's real value
was never volume but **risk isolation** — pass 2's characteristic defect (cutting a rule that is
needed twice) is a different failure mode from pass 1's (scope drift while repairing prose), and
separating them gives each its own rollback unit and observation window. That argument survives,
because it depends on class-7 edits being riskier *per edit*, not more numerous. So: fold the
dedup sweep into the single pass as a **semantic read**, still emitting a keep-list (finding 3).
**Split into a second gated pass only if** the sweep yields **more than 5 candidate cuts**, **or**
any candidate touches an **audit-enforced token** (check 8's `git add`/"delegated subagent",
`teco`'s roster, the `CPG:` line contract) **or a grant/authority clause**. Under this rule C1
correctly ran as two passes (18 dedup edits, one touching the commit grant) and C2 correctly ran
as one (2 candidates, neither near a token); C3–C6 decide from their own inventories.

*Clarified at C5: "**more than 5 candidate cuts**" means **recommended** cuts, not raw candidates
surfaced.* A candidate that survives the finding-5 test is a **judged-and-kept**, not a cut — it
consumed judgment but produced no edit, and the split exists to isolate *edits*. C5's sweep raised 7
raw candidates and recommended 0, so the rule did not fire; had the count been read raw, C5 would
have split for zero edits. State both numbers in the unit record so the margin is visibly tested
rather than assumed.

**Method warning — an n-gram scan is not a class-7 detector.** Used at C2 and it *inverts*:
prompt duplication is paraphrase-level by construction (one author restating one rule in each
section's local register, choosing different words each time), while verbatim repeats in a prompt
are overwhelmingly class-2 mechanism. A 6-gram scan on `tico.md` returned 5 hits, all of them
things that must be **kept** (two brief templates, a `git add <path>` form, a path convention),
and missed 4 real repeats — two of which shared no contiguous wording at all. Use it as a
**keep-list generator**, never as a cut-list. Class 7 requires the semantic read.

*C1 pass 1 executed 2026-08-24 (5,948 → 5,728 w; 15 edits, class-5/6 only). Gates (a)–(e) green;
cobb §7 lint pass-with-findings, all five findings fixed before commit. Three findings this unit
adds to the doctrine, all carried in `claude/teco/kaizen/history.md`:*

1. ***The class-5/6 cut is not where the risk lives — the prose repair around the hole is.***
   *The one real defect this pass produced came from **rewording a rule while deleting the story
   attached to it**: a tightened commit-grant sentence silently wrote `teco` out of the universal
   interactive-mode grant. Nothing was "lost" by deletion; scope moved during the rewrite. §6's
   attribution step 2 gains a third branch — rule kept, narrative correctly cut, but the
   surviving sentence's **scope** changed. On every remaining unit, re-read each reworded
   sentence as a rule diff, not a length diff.*
2. ***On a rule-dense file, class-5/6 removal alone buys ~4%, not ~40%.*** *Read the C1–C6 word
   targets as the sum of both passes; a single narrative sweep will not approach them. Cobb's
   independent estimate, given C1's load-bearing keep-list: **teco lands nearer ~4,300–4,600 w
   with every rule intact**, which §7 ("targets are estimates, not quotas") counts as a pass.
   Revise C1's ~3,400 target to that band.*
3. ***A pass-2 keep-list is a required output of pass 1.*** *Pass 1 is the cheapest moment to
   identify what looks like class-7 duplication but is mechanism (a verbatim brief template, an
   example row that **is** the spec, audit-enforced tokens restated three times). C1's list is in
   teco's kaizen entry; every later two-pass unit should produce one.*

*Two items rode along, both logged: a stale clause corrected inside an edited sentence (teco's
write guard described as reaching "the coordination doc and your own inbox" — the inbox was
deleted 2026-08-21), and a pre-existing repo-level `audit-team.sh` FAIL from commit `6193083`
(username leaked into a transcript path in `claude/cobb/kaizen/history.md`) fixed as its own
separate commit `c3f621d`, deliberately not bundled into the unit.*

***C1 pass 1's observation window CLOSED clean, 2026-08-24**, on a two-round synthetic probe
(§6's substitute where no organic dispatch occurs) — full result in `claude/teco/kaizen/history.md`.
Every watch-list rule fired, including all seven whose narratives pass 1 removed; the repaired
commit-grant clause was confirmed **behaviorally**, not just by re-reading it. **C1 pass 2 is
unblocked.** Two things the probe taught about probe design itself, for the remaining units:*

- ***Probe the rewritten clauses, not just the deleted ones.*** *The highest-value evidence came
  from the two rules whose surviving wording pass 1 changed (the commit grant, the milestone-close
  flip) — consistent with finding 1 above. A probe that only exercises rules whose narratives were
  cut tests the safe half of the change.*
- ***A single scenario under-covers; use a resume, not a second probe.*** *Round 1 left two rules
  untriggered (model routing had no mechanical unit; the commit grant was suppressed by the
  probe's own no-commit constraint). A `SendMessage` resume closed both cheaply, with context
  intact — and produced the probe's strongest signal, an unprompted self-diagnosis of a rule the
  agent had bent in round 1. Design round 2 to force what round 1 could not reach.*

***C1 pass 2 executed 2026-08-24 — C1 complete at 5,377 w** (5,948 → 5,377 over both passes,
−9.6%). 18 dedup edits, then 6 restorations from the lint. Gates (a)–(e) green; `cobb` §7 lint
pass-with-findings (1 major, 6 minor, 0 blockers), all fixed before commit. Detail in
`claude/teco/kaizen/history.md`. **Three findings that change the rest of this plan:***

4. ***The word targets in the Stage C table are wrong, and C1 is the evidence.*** *`cobb`
   re-measured mechanically (repeated-5-gram scan, not impression): **under 200 w of cross-line
   restatement remained** in a 5,728-w file, most of it class-2 mechanism that must repeat. Its
   verdict — `teco.md`'s editorial floor is **~5,200–5,250 with every rule intact**; the file is
   ~60 distinct rules at ~85 w each and stopped being narrative after pass 1. **Set no word target
   for C2–C6 before that unit's own inventory.** §1's original 25–45% estimate was calibrated on
   narrative density these files no longer carry post-Stage-B. Per §7, a file above target with
   every rule intact passes — the band moves, not the file.*
5. ***Pass 2's characteristic defect differs from pass 1's: a rule stated twice **on purpose**.***
   *C1's one major was cutting "delegate wide searches to `Explore`" from step 1 as a duplicate of
   the `Explore` routing row — but the row routes *someone else's unit*, while the step-1 sentence
   governed *teco's own orientation reads*. Same words, two decision points, two different actors.
   **The pass-2 test is not "is this said twice" but "is it needed twice."** Ask per occurrence:
   which decision point loses it, and does the agent stand there? Two lesser instances confirmed
   the shape — a trailing clause that was a scope **extender** rather than a second why, and a
   routing row whose reminder existed to block an inference the row itself invited.*
6. ***Below the editorial floor the only lever is structural, and it is outside this plan.***
   *`teco/kaizen/plan.md` **K-016** (split rare-path rules into an on-demand
   `coordination-techniques.md`) now carries C1's measurement. Caveat recorded there: a
   **reactive** protocol can move only its mechanics, never its trigger — the agent must recognize
   the trigger to know to load the file. A later Stage C unit that hits the same floor routes to
   K-016-style progressive disclosure, not to cutting rules to reach a number.*

*Declined deliberately: ~115 w of further trims `cobb` named post-lint. Shipping them would mean
unreviewed edits for a ~2% gain against a floor just certified, and one touches the commit-grant
paragraph — pass 1's regression site, which must not be re-edited inside the unit whose lint has
already run. Available to a future unit under its own gate.*

***C3 executed 2026-08-24 (2,510 → 2,473 w, −37, −1.5%).** Five edits, one pass; both wave-1 lint
carry-forwards discharged. Gates (a)–(e) green; `cobb` §7 lint **0 blockers, 0 majors** (3 minors,
2 nits) — the first unit in this plan with no MAJOR, and the first where the pre-declared
"considered and rejected" list did the work the lint used to. Detail in
`claude/analyst/kaizen/history.md`. **Two findings:***

7. ***The editorial floor is not a `teco` anomaly — it is where a post-Stage-B file sits, and it
   is reached in one pass.*** *`analyst.md`'s residual class-6/7 inventory across the whole file
   after C3 is **under 25 words** — three candidates totalling ~22 w, none near an enforced token.
   Finding 4 already emptied C4–C6's target cells to "after inventory"; C3 says go further and
   **calibrate what those inventories should be expected to return**. Two files now land at ~1.5–4% for a class-5/6+7
   sweep, and the second was a mid-size file at less than half `teco`'s length, so the floor tracks
   **rule count, not word count**. §1's headline ("25–45% per heavy file") is retired for anything
   post-Stage-B; the honest expectation for C4–C6 is single digits, and a unit that reports 2% with
   a certified inventory has **passed**, not underperformed.*
8. ***Progressive disclosure has one non-negotiable design constraint, and the payoff argument is
   consistency rather than tokens.*** *Generalizing finding 6's caveat into a rule: an offloaded
   reactive rule must leave a **trigger stub** in the prompt — the mechanism moves, the trigger
   never does. The failure mode is the tempting one: a single vague pointer ("consult X for
   evidence traps") requires recognizing a trap as a trap in order to know to load the file that
   names the traps. Circular, and it fails **silently**, which is why it survives review. Test each
   candidate rule for a trigger recognizable from the task surface **without already knowing the
   rule's content**; all six of analyst's traps pass, so the offload is safe at ~9% net. But the
   reason to do it is that `review-techniques.md` **already holds entries of the same genre**, with
   no stated criterion for which home a new lesson takes — that split, not the word count, is what
   keeps generating drift. Opened as `analyst` **K-003**, the second K-016-style item this plan has
   produced rather than absorbed.*

***C4 executed 2026-08-24 (4,649 → 4,467 w over two files: `security-expert.md` 2,437 → 2,357,
−3.3%; `devops.md` 2,212 → 2,110, −4.6%).** Six edits, one pass. Gates (a)–(e) green; `cobb` §7 lint
**0 blockers, 0 majors on the C4 edits** — the second consecutive unit with no MAJOR of its own
making, again with the "considered and rejected" list pre-declared. Detail in
`claude/security-expert/kaizen/history.md` and `claude/devops/kaizen/history.md`. **The split rule
fired for the first time and was obeyed:** the class-7 sweep produced exactly one candidate, it sat
inside a section headed "advisory, not authority," and splitting the unit for an 11-w gain was
disproportionate — so it was deferred, then judged a genuine finding-5 keep and recorded in
`security-expert/kaizen/plan.md` as **judged-and-kept** rather than left as a phantom pass-2. **Two
findings:***

9. ***A post-Stage-B file's residual is dominated by its own idiosyncratic citation habit, not by
   shared boilerplate — so an inventory's first move is to name that habit.*** *Stage B removed
   what the files had in common; what remains is per-file and clusters tightly. Before its last cut,
   `security-expert`'s entire residual was **FR-tags** — because it is the only agent on this team
   built from a formal requirements doc, and therefore the only prompt with FR-numbers to leak.
   `devops`'s was **repo-specific lore**, because it is the only user-scoped agent and the only one
   that carried a worked example of this repo. Generic boilerplate scanning finds nothing on either.
   Corollary for C5/C6: **identify the file's one citation habit first**, and expect it to account
   for most of the inventory. Finding 7's data set is now four files at 22–34 w residual within a
   ~360-w size band — the floor is not just a magnitude, it is remarkably **tight** across files of
   similar rule count.*
10. ***A recurring "refresh it when it goes stale" maintenance item is a deletion candidate, not a
    maintenance item — and the tell is that it rots in a different slot each time.*** *`devops`'s
    orientation example carried its own upkeep rule in the backlog ("if this repo's infra changes
    materially, trim/refresh it"). That remedy was applied **twice** (2026-07-09 image tag,
    2026-07-11 start-script consolidation) and the block rotted a **third** time anyway — with the
    fact installed by the first repair still accurate. **Repairing the fact that broke last time
    never protects the facts that break next.** The rot had also seeded a **stale backlog**: three
    parking-lot items in the same file were wrong or dangling, two of them describing work already
    delivered. Test for the shape: an always-loaded block of environment facts, plus a standing
    instruction to keep refreshing it, is a maintenance obligation the prompt cannot discharge —
    delete it and let the agent's own orientation step read the authoritative source. Weigh
    **scope** above rot when deciding: the decisive argument here was not staleness but that
    `devops` is **user-scoped**, making a one-repo snapshot a false anchor in every other project,
    on the one agent whose whole remit is "don't generalize from another repo."*

***C5 executed 2026-08-24 (6,355 → 6,308 w over three files: `tdd-engineer.md` 2,163 → 2,154,
−0.4%; `qa-engineer.md` 2,094 → 2,071, −1.1%; `data-scientist.md` 2,098 → 2,083, −0.7%).** Five
edits, one pass. Gates (a)–(e) green; `cobb` §7 lint **0 blockers, 0 majors, and zero findings
introduced by the edits** — the third consecutive unit with no MAJOR of its own making, and the
first where the lint found nothing at all attributable to the unit. Three pre-existing minors routed
to `tdd-engineer` K-006 and `qa-engineer` K-005/K-006. Detail in the three agents'
`kaizen/history.md`. **One edit was a deliberate rule change, declared to the lint as such rather
than slipped through as compression:** `qa-engineer`'s test-report path dropped ", or the component's
convention" — a report filename *is* filename grammar, which root `AGENTS.md` fixes repo-wide and
which that prompt already states twice is not component-negotiable. Read as a rule diff it is
strictly narrower and the removed half was never validly exercisable; `cobb`'s re-read found it also
resolved a second intra-file contradiction nobody had cited. **Two findings:***

11. ***Class 6 empties out. Past a habit-targeted pass, the editorial floor is a pure class-7
    floor.*** *All three C5 files measured **0 w of class-6 residual** — no dates, no authority
    markers, no supersession trails, no provenance attributives left anywhere. That is a stronger
    result than C3 or C4, where residual was still mixed, and it sharpens finding 9's corollary into
    a stopping rule: **name the file's citation habit, cut it, and class 6 is done.** What remains
    after that is only class-7 paraphrase — the judgment-bound category, where the finding-5 test
    ("needed twice", not "said twice") decides every case and the honest answer is usually keep. A
    unit that reports zero class-6 residual has finished the cheap half of the work permanently, not
    just for that pass. If C6 replicates it, the doctrine's §3 table can say so outright.*
12. ***A file's residual tracks its **organizing structure**, not just its rule count — and when the
    named habit turns out to be a certified keep, the structure is where to look next.*** *`qa-engineer`
    landed at **40 w**, the first file above the 22–34 w band that findings 7/9 established across
    four files (now seven: 22, 22, 25, 29, 31, 34, 40). The cause is not leftover residue. Its
    four-phase loop deliberately restates cross-cutting rules — environment mutation, evidence
    discipline, convention matching — at each phase where they fire; that is finding 5's "needed
    twice" pattern operating three times in one file. **A prompt organized as a sequence of phases
    will always carry more legitimate restatement than one organized as principles + workflow**, so
    calibrate an inventory against the file's shape rather than one flat band. The same unit supplies
    the companion half: `qa-engineer`'s named habit (its doc-convention override clause) was already
    **certified as a required keep** by `docs/reviews/doc-reference-convention.md` m17 — *"or the
    rewritten `:28` is contradicted from 26 lines below"* — so the habit-first method returned almost
    nothing there. Finding 9 predicted the habit would account for most of the inventory; C5 shows
    the habit can instead account for **none** of it, and that this is diagnostic rather than a
    failed sweep. When the habit is a keep, stop scanning for provenance and read the structure.*

***C6 executed 2026-08-24 (6,657 → 6,622 w over four files: `cobb.md` 2,006 → 1,976, −1.5%;
`graph-dba.md` 1,807 → 1,806; `frontend-engineer.md` 1,604 → 1,600; `coder.md` 1,240 → 1,240,
**zero edits**).** Five edits, one pass, plus a 6-w correction to `skills/agent-maintenance/SKILL.md`'s
always-on frontmatter description. Gates (a)–(e) green; `cobb` §7 lint **0 blockers, 0 majors
attributable to the C6 edits** — the fourth consecutive unit with no MAJOR of its own making. Four
pre-existing minors routed to `coder` K-003/K-004, `frontend-engineer` K-003, `cobb` K-020. Detail in
the four agents' `kaizen/history.md`. **This unit closes Stage C.**

**Three things C6 did differently, each worth carrying forward:**

- **The pre-edit-self safeguard was met literally, for the first time.** §7's risk register says
  `cobb`'s own compression is "linted by its pre-edit self before the edit lands," which Stage B
  could not honour — the live deployment symlink makes an edit production before any lint runs.
  C6's sequence: **propose → `git checkout HEAD -- cobb.md` → lint (edits supplied as text in the
  brief) → re-apply.** No `git show HEAD:` restatement needed. `cobb` **plan.md** F1 closed on this.
- **Two edits were declared corrections rather than compression**, continuing C5's practice of
  telling the lint which is which. `cobb.md`'s lint bullet claimed **six** §7 dimensions and
  enumerated six; the skill says "work the **seven** dimensions below." This agent has been running
  a seven-dimension lint from a six-dimension instruction since 2026-07-16 — including at every gate
  (c) of this plan. On the lint's own advice the numeral was **dropped**, not incremented: a numeral
  duplicating another artifact's list length is a drift surface that had already failed once, and
  without it a future eighth dimension leaves the prompt incomplete rather than wrong.
- **A MAJOR that the unit's own edits *activated* was fixed in-commit, not filed.**
  `skills/agent-maintenance/SKILL.md:3` still said "`kaizen/inbox.md` is a frozen historical relic"
  — and a skill's **description is always-on** (`cobb.md:42` states that rule). The moment `cobb.md`
  stopped claiming those twelve files exist, its always-loaded skill description started
  contradicting it. **Rule for the remaining stages: a pre-existing defect that this plan's edits
  make *live* is in scope for this plan's commit; one that merely stays latent is not.** That line
  is what separates it from the four minors routed out.

**The split rule fired a second time, on the "grant/authority clause" branch, and was obeyed.**
Recommended cuts came to **exactly 5** against a "more than 5" ceiling — a zero margin — and one
candidate sat in the **same bullet** as `cobb`'s curator-`DETACH DELETE` authorization. The lint held
that "touches" must be read from the rule's purpose: finding 1 established the risk is the **prose
repair around the hole**, not the deletion, so proximity governs, and same-bullet is tighter than the
same-*section* proximity that fired the rule at C4. Discharged as C4 actually discharged it — **defer,
not split**: the stale-fact half shipped, the provenance half is held as a pre-analyzed candidate,
and the recommended count dropped to 4. *Two independent near-misses on one unit (count at the
ceiling, proximity at the boundary) is itself signal — read a zero margin as a reason to obey, not
to argue.* **Two findings:**

13. ***Residual tracks the number of layers that re-aim the same cross-cutting rules — not section
    count, not word count, not rule count.*** *Finding 12 named organizing structure as the variable;
    C6 isolates it, because it holds the confounders still. `frontend-engineer` (~39 w) and
    `graph-dba` (~20 w) are comparable in length (1,600 / 1,806 w) and both have four top-level
    sections, yet they sit at opposite ends of the band. The difference is what those sections **do**:
    frontend-engineer's four layers (domain expertise → workflow → principles → boundaries) each
    re-aim the same cross-cutting rules at a different altitude — conventions, UI states, and test
    scope each appear three times, legitimately — while graph-dba's top layer is *reference
    mechanism* (GraphBLAS, the pinned deployment, the Cypher subset) with no workflow counterpart, so
    nothing restates. `coder` at 1,240 w has frontend's four-layer shape and the set's **highest
    density** (2.7%), which rules out length. So: a "summary + workflow + principles + guardrails"
    prompt will always run high and a "reference + workflow" prompt will always run low. **Calibrate
    against the shape, and stop reading a high residual as a failed sweep.** Eleven-file data set:
    20, 22, 22, 25, 29, 29, 31, 34, 34, 39, 40.*
14. ***Class 6 is finished. §3's table can say so outright.*** *Finding 11 asked C6 to replicate C5's
    zero-class-6 result and set the consequence in advance. It replicated: **all four C6 files measure
    0 w of removable class-6 residual**, making it **eleven of eleven** across the whole team. Every
    date, authority marker, supersession trail, FR-tag and `kaizen/history.md` pointer that could
    leave a prompt has left. Three certified exceptions remain and are exceptions by rule, not by
    oversight: `cobb.md`'s `verified 2026-07-10 against …` stamp (mandated by that file's own
    Drift-resistance principle, and duplicated in the skill for a load-set reason the lint verified),
    its `pre-M8` query discriminator, and `graph-dba`'s `(successor to RedisGraph)` anti-trigger.
    **The doctrine's operational form is now: name the file's citation habit, cut it, and class 6 is
    done permanently — everything after that is class-7 judgment, where the finding-5 test decides
    and the honest answer is usually keep.** A future sweep that opens with a provenance grep is
    re-running a search whose answer is known to be empty.*
15. ***Waste is created at specification time, and the check is "how many agents does this rule
    bind?"*** *Stage D found the plan's own spec placing two cross-cutting rules into one agent's
    prompt — the exact defect Stages A–C spent six units removing. Neither was sloppy drafting;
    both read as obviously correct until the agents-bound count is taken (six for revision notes,
    three for later-pass dispositions), at which point both are obviously misplaced. **The
    doctrine's classes describe waste already in a file; this finding is about not creating it.**
    Two corollaries, both cheap: a rule whose topic the target file never otherwise mentions is
    almost always misplaced (the clause would introduce the topic solely to qualify it); and a rule
    specified in two places at once is not a "split" — check the other place's literal wording
    before defending the division, which is exactly the check I skipped and gate (c) caught.*

### Stage D — Output discipline (small, surgical *additions*)

- `architect.md`: resolve the §"stand alone" vs. §"compress by pointer" tension explicitly:
  *stand-alone means the implementer never re-derives a decision — state each decision **once**,
  in one canonical section; elsewhere cite the section; quote a sibling note's conclusion once and
  cite it for rationale; a delegation-summary table cites, it does not restate.*
  ~~Plus: revision history is one dated line, not a "Revision note" narrative.~~ **Relocated to
  Stage E at execution (2026-08-25)** — `architect.md` mentions revision notes nowhere, so the
  clause would have introduced the topic solely to qualify it, and rule 5 (≥5 revising agents) is
  its only mandated home.
- `analyst.md`: a finding is evidence + why + concrete fix in **≤~15 lines**, overflow to an
  appendix; ~~a later pass records a closed finding as **one disposition line**, full prose only
  for *new* findings;~~ **relocated to Stage E at execution (2026-08-25)** — Stage E's own bullet
  below already specified this rule verbatim, for three reviewing agents rather than one;
  implementation reviews **always** open the `-impl` file — never append to the
  plan review (the 2,455-line file is the incident this prevents).
- Guard on the budgets themselves: **never drop evidence to fit a budget — appendix it**; the cap
  bounds the finding's inline body, not the review's rigor.
- Net additions ≤~120 words across both files — measured against the same budget they impose.

**Executed 2026-08-25 — done: `architect.md` 1,429 → 1,471 w (+42), `analyst.md` 2,325 → 2,375 w
(+50). Net +92 against the ≤~120 budget**, with the tilde doing no work. Three additions shipped
(architect's once-canonical rule; analyst's ≤~15-line finding cap with its appendix escape and
never-drop guard; the `-impl` never-append prohibition, folded into the existing sentence that
already named the suffix rather than added as a new one) and two relocated to Stage E, above.

- **The relocation test is finding 13 applied to an addition, and it is the stage's main result.**
  Both relocated items are cross-cutting rules that the spec had aimed at a single agent's prompt.
  The test that caught them: *count the agents the rule actually binds.* Revision notes bind six;
  later-pass disposition lines bind three. A rule with N>1 users belongs in the file all N read.
  I applied this test myself to the revision-note clause and then **failed to apply it to the
  disposition-line clause** — `cobb`'s gate-(c) lint fired it back, with the decisive evidence
  being that Stage E's own bullet already carried the identical rule verbatim. **Stage D was one
  gate-(c) pass away from shipping a rule into two always-loaded files at once.**
- **A prompt that never mentions a topic is the strongest relocation signal.** `architect.md` says
  nothing about revision notes, so the mandated clause would have introduced the topic *solely to
  qualify it*. That is worse than finding 13's usual shape (a layer re-aiming a rule the file
  already carries), and it is detectable before writing a word.
- **Additions get the same class-6 discipline as cuts, and passed.** Zero dates, authority
  markers, or history pointers entered either file. Finding 14's "class 6 is finished" survives
  the one stage that could have reopened it.
- **Two placement corrections from the lint, both free.** "a summary table cites, it does not
  restate" → "a **recap** table" (the general word reached architect's *step* table, the one
  artifact the same file twice insists must stay concrete); and `-ml.md` → `-ml.md`/`-graph.md`,
  because §2's motivating incident — one decision restated 4× in `document-ingestion.md` — was a
  **`-graph.md`** note, so the rule as first written did not bind the case that generated it.
- **One alignment edit outside the addition** (`architect.md` step 4): "fold its *recommendation*
  into the plan" → "its *conclusion*", so step 4 and the new Handoff rule use one word for one
  thing instead of inviting the restatement the new rule forbids.

### Stage E — Shared context files (stakeholder-gated)

Same doctrine on root `AGENTS.md` (1,950 w), `claude/AGENTS.md` (1,848 w — densest narrative,
1 em-dash/36 w), `falkor-chat/AGENTS.md` (1,762 w). Higher blast radius: these bind humans and
every tool. Existing frozen documents untouched.

**The one *convention* amendment (root `AGENTS.md`, collision rule 5) — now four parts, three of
them relocated here at Stage D's execution** (2026-08-25; each was specified as a single agent's
prompt edit, and each turned out to bind 3–6 agents, making a one-agent prompt the wrong home —
finding 13). Rule 5 is the only place all of them are mandated, so they land as one amendment
rather than scattering across `architect.md`, `analyst.md`, and rule 5:

1. A review's later `## Pass N` section is **compact by rule** — verdict + new findings in full.
2. A prior finding gets **one disposition line**: the disposition (fixed / not fixed / superseded)
   plus the evidence rechecked. *(From Stage D `analyst.md`. Binds `analyst`, `data-scientist`
   (`reviews/<slug>-ml.md`), `security-expert` — every agent whose deliverable is a review.)*
3. A **revision note is one dated line**, not a narrative. *(From Stage D `architect.md`. Binds
   every agent that revises a pre-approval document — `architect`, `teco`, `tico`, `qa-engineer`,
   `data-scientist`, `graph-dba`.)*
4. **Resolve which branch a re-review lands in** — rule 5's `## Pass N` clause sits in the **No**
   branch, but the canonical later pass (analyst reviews → owner fixes → analyst re-reviews) is a
   document that has been *executed against*, which the **Yes** branch routes to `<slug>2.md`.
   ~~The convention does not currently answer; pick one.~~ **Premise corrected at execution
   (2026-08-25):** the convention *was* answered once, in practice — `docs/reviews/kaizen-inbox-distillation.md`
   and `…2.md` are a genuine ordinal-successor review pair, `active`, correctly cross-linked with
   `Extends:`/`Extended by:`, and not a family member inheriting a "2" from its topic. So this is
   not a gap being filled but a **precedent being overridden** for same-role later passes, and it
   is recorded as such. *(Raised by `cobb`'s gate-(c) lint at Stage D as F6 — not fixable inside
   any agent prompt, and cheapest to settle while rule 5 is open.)*

**Stage E runs as three units, not one** (set at execution, 2026-08-25): **pass 1** = the class-5/6
sweep across all three files; **pass 2** = `claude/AGENTS.md`'s "Git-commit authority" section
alone; **pass 3** = the four-part rule-5 amendment above. Pass 2 is split out because the section is
a **grant/authority clause**, the split rule's named branch, and it is the largest single class-6
concentration in the stage. Pass 3 is split out because it is a *rule change*: §4.0 makes the commit
the rollback unit, and a rule addition bundled into a compression commit cannot be reverted
independently of the compression.

***Stage E pass 1 executed 2026-08-25 (5,779 → 5,382 w over three files: root `AGENTS.md`
1,983 → 1,808, −8.8%; `claude/AGENTS.md` 1,973 → 1,779, −9.8%; `falkor-chat/AGENTS.md`
1,823 → 1,795, −1.5%).** Gates (a)–(e) green; `audit-team.sh` PASS before and after (checks 5 and 5b
both read these files). `cobb` §7 lint **0 blockers, 4 majors, 3 minors, 3 nits — all majors and
minors fixed in-commit**, ending the four-unit streak of units with no MAJOR of their own making.
Detail in `claude/cobb/kaizen/history.md`. Three declared rule-shaped removals (milestone status out
of root per finding 10; the `:KaizenEntry` schema out of root, a class-7 edit riding in a class-5/6
pass because it was tangled in the same bullet as the migration narrative; the `no inbox.md`
anti-trigger out of root per finding 15) and three declared corrections. Class-6 residual: **0 w in
root and `falkor-chat/AGENTS.md`**, one judged-and-kept normative `FR-10` citation in
`claude/AGENTS.md`. **Two findings:**

16. ***Finding 7's "single digits" floor is scoped to post-Stage-B files, and nothing in this plan
    said so.*** *Pass 1 returned −8.8% and −9.7% on the two files it swept properly — two to twenty
    times C3–C6's rate, from the identical doctrine. The variable is not size, structure, or rule
    count (findings 7/9/12/13's whole progression); it is simply **whether the file has ever been
    swept**. Stage B emptied the shared boilerplate from thirteen prompts before Stage C measured
    any of them, so every Stage C number is a **second**-pass number, and finding 7 generalized from
    a sample where that was invisible because it was universal. §1's retired 25–45% estimate was
    never wrong for a first sweep — it was wrong about which files were still getting one.
    **Operationally: ask "has this file been swept before?" before calibrating an inventory, and
    expect the original band on anything that answers no.** The corollary matters more than the
    number: `falkor-chat/AGENTS.md` returned −1.5% in the same unit under the same doctrine, because
    it is a **fact-dense reference file** whose class-6 load was genuinely small — so first-sweep
    status raises the ceiling, it does not set the floor.*
17. ***On a multi-file unit, gate (a)'s mapping must be re-derived from the files, never from
    intent.*** *Pass 1's one gate-level failure: I reported a rule as removed from root `AGENTS.md`
    when the edit had landed in `claude/AGENTS.md` instead — leaving the rule, and its `FR-` tag, in
    the always-loaded file and deleting it from the maintenance bullet where a maintainer actually
    stands. The state I brought to the gate was the **exact inverse** of the state I described. It
    survived my own review because a two-file rule reads as one decision ("move it to the narrower
    file") and I checked that the decision was right rather than that it had been executed. Finding
    1 makes gate (a) the safety net for this entire method, so a mapping that misreports which file
    a rule left is not a safety net. **Add to gate (a) for any unit touching more than one file:
    after editing, re-grep every moved token across every file in the unit, and map from the grep
    output — not from the edit list.** Cheap, mechanical, and it is the only step that would have
    caught this.*

***Stage E pass 2 executed 2026-08-25 — the deferred grant/authority block. `claude/AGENTS.md`
1,779 → 1,668 w (−111, −6.2%); 1,973 → 1,668 across both passes (−15.5%), the largest single-file
reduction in this plan.*** *Plus one `claude/teco/teco.md` line as a declared correction (net 0 w).
Gates (a)–(e) green; `audit-team.sh` PASS. `cobb` §7 lint **0 blockers, 2 majors, 2 minors, 1 nit —
all fixed in-commit**. Class-6 residual in the section: **0 w**, so `claude/AGENTS.md` as a whole now
matches finding 14's eleven-of-eleven prompt result. Detail in `claude/cobb/kaizen/history.md` and
`claude/teco/kaizen/history.md`. **Every finding this unit produced was a scope or meaning change in
new prose; not one was a lost rule** — the strongest confirmation finding 1 has had. **Two
findings:**

18. ***A grant defined by contrast lives in two files, and this plan has no gate that reads the
    second one.*** *Pass 2 deleted a class-6 supersession clause from `claude/AGENTS.md` ("extension
    B deliberately breaks the write-scope==commit-scope identity `tico` previously held"). The C2
    anti-trigger test, run on `tico`, correctly returned **No** — `tico.md` enumerates all three
    cases explicitly, so `tico` cannot infer the identity. The deletion was right for `tico` and
    wrong for the corpus: **`teco.md:129` asserted that identity as live fact**, dating to
    `eb318d4` (2026-07-30) and falsified twice since, and the deleted sentence was the only marker
    anywhere that it was stale. Fixed at the source rather than by restoring provenance to a
    just-cleaned file. **Generalization: the C2 anti-trigger test must run over the whole load-set,
    not over the one agent the sentence is about — ask "who else states this rule, and does their
    wording still agree?"** The reciprocal statement in another agent's prompt is a required
    update when a grant moves, not an optional one; gates (a)–(e) check the edited file and its
    own history, and nothing checks the file that describes it from outside.*
19. ***Folding stacked amendments into one list trades syntactic isolation for a lexical scope
    marker — and the isolation was load-bearing.*** *The block held three dated amendments to one
    grant, each restating the whole grant to add a case; folding them into one enumerated list is
    the correct class-7 read and the largest readability win in Stage E. But while each extension
    was its own **sentence**, a limit stated inside one could not reach the others **by
    construction**. After the fold, one limit sat 30 words downstream of its "for that third case"
    marker with a causal clause wedged between, and its subject had degraded to a bare "it" whose
    nearest antecedents were the wrong nouns. I verified it still excluded the adjacent case and
    **missed that it also reached a third grant** — one carrying no such limit since 2026-07-30 and
    part of no extension. **When collapsing N stacked amendments into one list, every limit that was
    previously scoped by sentence boundary needs an explicit non-reach clause** ("that case alone …
    and it does not narrow the others"). This is C1's regression site hit the same way twice: scope
    moving during a rewrite that removed no rule.*

***Stage E pass 3 executed 2026-08-25 — the four-part collision-rule-5 amendment, and the stage's
only addition. Root `AGENTS.md` 1,808 → 1,885 w (+77).*** *Gates (a)–(e) green; `audit-team.sh`
PASS. `cobb` §7 lint **0 blockers, 2 majors, 3 minors, 2 nits — all fixed in-commit**. Zero dates,
authority markers, or history pointers entered the file, so finding 14 survives Stage E's addition
as it survived Stage D's. **This unit closes Stage E.***

*Part 4's ruling: **a `reviews/` document revises in place regardless of the selector's answer.** The
ordinal successor exists to freeze a document that *directs* work; a re-review's value is pass 1 and
pass 2 read together. **This overrides a live precedent rather than filling a gap** (see the
corrected premise above), and `docs/reviews/kaizen-inbox-distillation{,2}.md` is **not retrofitted**
— it followed the convention as it then stood, per the same no-retrofit logic that governs pre-M8
kaizen entries and `archived` documents. No grandfather clause was added to `AGENTS.md`: that would
be class-6 supersession, undoing Stage E pass 1 to describe one file.*

*The load-bearing fix was not in the amendment at all. **Rule 5's trigger said "the same kind and
topic" while rule 1's primary key is `(component, kind, topic-slug, role)`** — role was missing, so
`reviews/x-ml.md` and `reviews/x-impl.md` literally matched "a second document of the same kind and
topic." Latent before, that produced only a wrong-but-separate file; part 4 would have converted it
into an instruction to **append**, which is §2's 2,455-line incident exactly. Corrected to "the same
kind, topic, **and role**" — two words, and it makes part 4 self-scoping. The Yes branch's own
examples (`executor2.md` / `executor2-coordination.md`) already assumed role was held fixed.*

20. ***For a convention edit, inventory the instances before writing the rule.*** *Both of this
    unit's majors were found by reading the repo's document tree — `git ls-files | grep reviews/`
    was the entire method — and neither was visible in the diff, the prompts, or the plan. **A
    convention amendment's blast radius is the set of files already obeying the old convention**,
    and gates (a)–(e) do not look there: they read the edited artifact, its history, and the audit
    script. Note this is the exact inverse of finding 14's advice for a *prompt* sweep ("a sweep
    that opens with a provenance grep is re-running a search whose answer is known to be empty") —
    **for a convention edit the corpus grep is the first move, not the wasted one**, because the
    question is what already complies rather than what residue remains.*
21. ***The case for relocating a cross-cutting rule is drift-resistance, not token economy, and the
    naive arithmetic says the opposite.*** *Per-instance the relocation looks obviously cheap: part
    2 (~45 w × 3 reviewing agents) plus part 3 (~10 w × 6 revising agents) ≈ 195 w across nine
    prompt-instances versus ~55 w here. But **per session** it inverts — root `AGENTS.md` loads in
    every session in every tool, while `analyst.md` loads only when `analyst` runs, so nine prompt
    copies are often the cheaper choice on tokens. The decisive argument is finding 15's and it is
    about correctness: **a rule binding six agents that lives in one prompt is simply wrong for the
    other five, and a rule copied nine times drifts nine ways.** State this wherever a relocation is
    justified — a future reader applying "lean context" naively will reach the opposite conclusion
    and try to push these rules back into prompts.*

### Stage F — Ratchet guard (make it stick)

- `agent-maintenance` skill: add the promotion rule — *a kaizen entry promoted into a prompt
  lands as rule + ≤1-clause why, nothing else; the evidence, story, and provenance stay in
  `kaizen/history.md`* — and prompt-waste as a §7 lint dimension, so prompts don't regrow the
  same weight. **Pulled forward, 2026-08-23** (stakeholder-directed, right after the pilot
  calibration): executed by `cobb` against its own skill, ahead of Stages B–E.
- Optional (decide at execution): a soft `audit-team.sh` word-count advisory (warn >2,500 w/agent,
  never fail) as the drift tripwire.

### Stage G — Living-document compaction (the closeout ratchet)

**The gap.** The module-documentation convention has a rule for how a document **freezes**
(`Status: archived`, in place) and none for one that **never can**. Stages A–F fixed the register
agents write in and the review genre; §1 scoped component documents out as history. But a living
document is not history — it is re-read whole every session, and nothing in the convention ever
takes weight out of one.

**Measured 2026-08-25 (finding 20's corpus-first method — `git ls-files`, then per-file counts).**
Root `AGENTS.md`'s convention section contains **15 additive/retentive verbs and 0 subtractive
ones**; `grep -E 'prune|compress|remove|delete|shrink|concise|brief'` over it returns **0 hits**.
The one mechanism that ever removed content from the live tree — "move it to `archive/` when the
milestone closes" — was itself deliberately replaced by a status token. Correct for link
integrity; nothing replaced its compaction effect.

**The axis the convention is missing** — it treats these identically:

| Kind | Examples | Growth |
|---|---|---|
| **Read-whole (living)** | `BACKLOG.md`, `DESIGN.md`, `AGENTS.md`, `README.md`, `QUERIES.md` | Must be bounded — it can never freeze |
| **Read-by-lookup** | `HISTORY.md`, `reviews/`, closed `plans/`, `test-reports/` | Unbounded is **correct**; `falkor-chat/docs/HISTORY.md` at 3,481 lines is healthy |

**Two independent instances, same cause, different location** — which is what makes this a
convention gap rather than an authoring lapse. Closeout is additive, so weight lands wherever the
closeout ritual happens to write:

| | `falkor-chat/docs/BACKLOG.md` | `docs/BACKLOG.md` (root) |
|---|---|---|
| Accretes in | item **bodies** — 1,317 of 2,025 lines (65%), ~57 lines per delivered item | milestone-map **cells** — the M7 cell is ~500 w including a "superseding the … framing above" trail |
| Stale current-state header | `## Active` claimed M3 in progress; M3 closed 2026-07-21 | `## Handoff — teco drives M2 (2026-07-18)`; the document is at M8 |
| Fixed | `0f48b8a` + `88bb71b` (instances only) | not touched |

Note the root backlog's **item** convention is the positive precedent — a delivered item is a
compact bullet, ~10 lines. The rule below generalizes what it already does right, and closes where
it leaks.

**The amendment (root `AGENTS.md`, module-documentation convention).** Two bullets:

1. **A living document is compacted at milestone close, not only appended to.** The documents read
   whole to be used — `BACKLOG.md`, `DESIGN.md`, `AGENTS.md`, `README.md`, `QUERIES.md` — can
   never freeze, so they never shed weight on their own; `HISTORY.md`, `reviews/` and closed
   `plans/` are read by lookup and may grow without bound. At milestone close, in the same pass
   that flips that milestone's documents to `archived`, **`teco`** reduces every delivered item in
   the module's `BACKLOG.md` to **one index row** (id, title, date, milestone) and drops from the
   living documents each section that now describes finished work — a "currently in progress"
   header, a plan-doc row for a document that now exists, a delivered-ticket annotation.
   **Verify present in `HISTORY.md` before deleting**: the closeout is a move, not a discard —
   the same history-first gate Stage B–E used on prompts.
2. **A milestone-map row says what the milestone is and when it landed.** Gate sequences, defect
   trails and superseded framings are `HISTORY.md`'s.

**Why it lands in the convention, not in `teco.md`** (finding 21): the rule binds every agent that
closes a milestone or edits a living document, and `teco.md` is wrong for the other five.

**Blast radius — the files already obeying the old convention** (finding 20: this is the first
move, not the last). Living documents by size: `falkor-chat/docs/QUERIES.md` 2,412 ·
`docs/BACKLOG.md` 816 · `falkor-chat/docs/DESIGN.md` 756 · `cypher-mcp/README.md` 750 ·
`falkor-chat/docs/BACKLOG.md` 717 · `falkor-chat/docs/SERVER.md` 500 · 13 `claude/*/kaizen/plan.md`
(21–224). `QUERIES.md` is **not** a target: it is a section-cited reference read by lookup, and it
carries 2 delivered-narrative markers in 2,412 lines — already clean.

**Units.**

| G1 | The two-bullet amendment to root `AGENTS.md`. Rule change ⇒ its own commit (§4.0). |
| G2 | `docs/BACKLOG.md` (root) — apply it: milestone-map cells to one or two sentences, `## Handoff` header retired or re-pointed at M8. |
| G3 | `claude/*/kaizen/plan.md` sweep — same question, 13 small files; likely a no-op, confirm rather than assume. |

**Not in scope.** Rewriting `HISTORY.md`, `reviews/` or any `archived` document; `QUERIES.md`;
`falkor-chat/docs/BACKLOG.md` and `DESIGN.md` (already done — `0f48b8a`/`88bb71b`/`3e2b378`).

**Gates.** (a) and (b) as written — for a convention edit, gate (a)'s inventory is the corpus grep
above, and gate (b) is the per-item `HISTORY.md` check. (c)/(d)/(e) apply only to units touching a
prompt or skill; G1–G3 touch none, so `cobb`'s §7 lint and `audit-team.sh` are informational.
**Precondition unmet at filing:** §4.0 requires a clean tree and the working tree carries 12
unrelated modified files. **Stakeholder-gated like Stage E** — this changes the convention every
agent and tool loads.

## 5. Verification strategy

- Per-unit: the four gates in §4 (rule inventory, history-first, cobb §7 lint, audit script).
- Whole-plan acceptance: (1) `audit-team.sh` PASS; (2) corpus word-count table before/after in
  this document; (3) spot behavioral probe — one representative dispatch per compressed heavy
  agent (e.g. a small architect plan, an analyst review of a small diff) checked for the *rules*
  still firing (CPG line present, verdict scale, finding format), not for output style.
- Style change in *deliverables* is expected to lag prompt change (the register also lives in
  AGENTS.md and the docs the agents read) — re-measure output density (words/sentence, restatement
  count) on the first post-Stage-E feature family, not immediately.

## 6. Breakage detection & rollback

**Definition of broken.** A compressed agent is *broken* when, in a real run, it violates a rule
its pre-compression prompt enforced: skips a required artifact element, oversteps a boundary its
prompt used to hold (hooks catch the write-path cases, not the judgment cases), or visibly lacks
context it used to have ("why would I…?" confusion at a spot the cut narrative used to cover).
Style regression is *not* breakage.

**Per-agent watch list** — the rules to spot-check in the next real dispatches of each heavy
agent (compiled from gate (a)'s inventory at unit time; starting set):

| Agent | Watch for |
|---|---|
| `architect` | `CPG:` line present & verbatim; plan lands in `docs/plans/`; no source edits attempted |
| `analyst` | Verdict scale used; findings severity-ranked; `-impl` split honored; evidence-vs-inference distinction kept |
| `teco` | `subagent_type` on every dispatch; agentId recorded at dispatch; review gate sequenced per unit; paused-unit handling; step-table dispatch splitting |
| `tico` | Status tokens only it may flip; WHAT/WHY (no HOW) in requirements |

**Observation window.** A unit stays *delivered*, not *closed*, until the agent's next **3 real
dispatches** (or ~1 week, whichever comes first) pass the watch list. Stage-gating (B wave 2,
C1 pass 2, Stage E) keys off *closed*, not *delivered*. Where no organic dispatch occurs, a
synthetic probe (small representative task) substitutes.

**On breakage — the ladder:**
1. **Revert first, diagnose second** — `git checkout <last-good-sha> -- claude/<name>/<name>.md`
   (§4.0); the agent is whole again immediately.
2. **Attribute** — check the violated rule against gate (a)'s inventory mapping (persisted in the
   kaizen history entry, gate (e)): *rule dropped* ⇒ gate failure, restore it; *rule kept but no
   longer firing* ⇒ the cut narrative was load-bearing (class 3/4 misclassified as 5/6) —
   reinstate a one-clause why, not the full story, and re-run gates.
3. **Record** — the misclassification lands in that agent's `kaizen/history.md` and adjusts the
   doctrine's calibration for remaining units.
4. **Abort criterion** — a second confirmed breakage across *different* units halts the rollout
   entirely: the doctrine itself is miscalibrated; reassess §3 with the stakeholder before any
   further unit. One breakage is a unit-level event; two is a plan-level event.

## 7. Risks & open decisions

- **Load-bearing narrative cut by mistake** — the central risk; mitigated by the class-3/4 "keep"
  rules, keep-when-uncertain default, the pilot calibration gate, history-first (nothing is
  destroyed, only moved), and the §6 ladder (revert-first, per-unit observation windows, abort
  criterion) when a cut proves load-bearing in practice.
- **Compression of `cobb.md` degrading the lint gate itself** — `cobb` is the gate for every
  other unit, so it goes **last** (already C6): every other compression is linted by the intact
  `cobb`; `cobb`'s own compression is linted by its pre-edit self before the edit lands.
- **Inbound references into prompt wording** — before each unit, grep the repo for citations of
  that file's phrases/sections (`grep -rn '<name>.md' — minus .git`) and preserve cited anchors.
- **Execution routing** — repo convention says agent-prompt edits are `cobb`'s remit with an
  `analyst` gate; alternatively the main session executes directly with the same §4 gates.
  *Recommendation: main session executes, `cobb` lint as the gate* — the doctrine table is the
  spec either way. Stakeholder call.
- **Targets vs. safety** — word targets are estimates, not quotas: a file landing above target
  with every rule intact **passes**; a file hitting target by dropping a rule fails gate (a).
