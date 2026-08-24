# Prompt & output waste reduction — agent team

> **Status:** active · **Owner:** `claude` · **Tracks:** — · **Version:** 4

*Rev 2 (2026-08-23): added live-deployment ground rules, per-unit rollback machinery, breakage
detection/abort criteria, staggered Stage B, two-pass rule for the heaviest cut.*
*Rev 3 (2026-08-23, pilot calibration — stakeholder ruling): class-5/6 material is deleted
outright, no inline dated pointers; only normative citations (specs/templates the rule uses)
stay. §3 table and Stage B block dispositions updated accordingly.*
*Rev 4 (2026-08-23, stakeholder ruling): commit granularity is one **complete unit** per commit
(all files the unit touched — prompts, kaizen histories, catalogs — landing and rolling back
together), not one file per commit. §4.0 and Stage B updated.*

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
to `architect.md`/`analyst.md`, one amendment to the review-accretion convention, and a durable
ratchet guard in the `agent-maintenance` skill.

**Out of scope:** frontmatter `description` fields (routing contracts — touching them cascades
into `claude/README.md`/roster updates for no waste win); on-demand knowledge bases
(`falkordb-quirks.md`, `review-techniques.md`, etc. — fact-dense, loaded only when needed);
`Status: archived` documents (immutable by convention); rewriting existing falkor-chat docs
(history — only *future* output changes).

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
| 7 | **Duplicate restatement** | Same rule stated twice in one file; cross-references that re-explain instead of pointing | One canonical statement + short cross-ref |

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
| C1 | `teco.md` | 5,987 w | ~4,300–4,600 w (revised at pass 1) | Largest and most narrative-dense. **Two passes, not one** (rule: any file with a >30% projected cut): pass 1 = unambiguous class-5/6 cuts only (narratives, provenance); pass 2 = class-7 dedup (judgment-heavier), only after pass 1's observation window closes clean. Each pass is its own commit/rollback unit |
| C2 | `tico.md` | 3,531 w | set after inventory | Not yet read in this analysis — inventory first, then target |
| C3 | `analyst.md` | 2,569 w | ~1,900 w | Evidence-traps list stays (class 3 payloads); trims are provenance and restatement |
| C4 | `security-expert.md` + `devops.md` | 2,471 + 2,266 w | after inventory | |
| C5 | `tdd-engineer.md` + `qa-engineer.md` + `data-scientist.md` | ~2,200 w each | after inventory | |
| C6 | `cobb.md` + `graph-dba.md` + `frontend-engineer.md` + `coder.md` | ≤2,054 w each | after inventory | Lightest; may need only Stage B |

Serialized or in review-gated pairs — never two units editing one file.

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
separate commit `c3f621d`, deliberately not bundled into the unit. Pass 2 is gated on this pass's
observation window closing clean.*

### Stage D — Output discipline (small, surgical *additions*)

- `architect.md`: resolve the §"stand alone" vs. §"compress by pointer" tension explicitly:
  *stand-alone means the implementer never re-derives a decision — state each decision **once**,
  in one canonical section; elsewhere cite the section; quote a sibling note's conclusion once and
  cite it for rationale; a delegation-summary table cites, it does not restate.* Plus: revision
  history is one dated line, not a "Revision note" narrative.
- `analyst.md`: a finding is evidence + why + concrete fix in **≤~15 lines**, overflow to an
  appendix; a later pass records a closed finding as **one disposition line**, full prose only for
  *new* findings; implementation reviews **always** open the `-impl` file — never append to the
  plan review (the 2,455-line file is the incident this prevents).
- Guard on the budgets themselves: **never drop evidence to fit a budget — appendix it**; the cap
  bounds the finding's inline body, not the review's rigor.
- Net additions ≤~120 words across both files — measured against the same budget they impose.

### Stage E — Shared context files (stakeholder-gated)

Same doctrine on root `AGENTS.md` (1,950 w), `claude/AGENTS.md` (1,848 w — densest narrative,
1 em-dash/36 w), `falkor-chat/AGENTS.md` (1,762 w). Higher blast radius: these bind humans and
every tool. Includes the one *convention* amendment (root `AGENTS.md`, collision rule 5): a
review's later `## Pass N` section is **compact by rule** — verdict + one-line disposition per
prior finding + new findings only. Existing frozen documents untouched.

### Stage F — Ratchet guard (make it stick)

- `agent-maintenance` skill: add the promotion rule — *a kaizen entry promoted into a prompt
  lands as rule + ≤1-clause why, nothing else; the evidence, story, and provenance stay in
  `kaizen/history.md`* — and prompt-waste as a §7 lint dimension, so prompts don't regrow the
  same weight. **Pulled forward, 2026-08-23** (stakeholder-directed, right after the pilot
  calibration): executed by `cobb` against its own skill, ahead of Stages B–E.
- Optional (decide at execution): a soft `audit-team.sh` word-count advisory (warn >2,500 w/agent,
  never fail) as the drift tripwire.

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
