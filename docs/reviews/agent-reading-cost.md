# Agent reading cost — requirements review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1

**Scope.** `docs/requirements/agent-reading-cost.md` (untracked, `Status: Interviewing`, 1,957
words), reviewed as the gate before `tico` flips it to Ready for design. Baselines: root
`AGENTS.md` (context-file convention, growth-stops bullet, collision rule 5, live-spec close rule),
root `REVIEW.md`, `docs/requirements/document-size-discipline.md`, the superseded
`falkor-chat/docs/requirements/summary-nodes.md`,
`claude/docs/requirements/agent-knowledge-base-strategy.md`,
`docs/requirements/connected-code-graph.md`, and the typed-decision-models and connected-code-graph
document families and ledgers. First version, so the growth-stops check covers the decision-log
shape and restated facts only; there is no earlier revision to compare growth against.

**Verdict: needs changes** — one blocker (B-1), seven majors.

CPG: not applicable — a requirements document with no code-level component under review.

### Blocker

**B-1 — The two core FRs have no acceptance criterion.** *Verified* by reading every AC's trace
tag. No AC names FR-1 (retrieve a part without reading the whole document) or FR-2 (plan step ↔
code, both directions). AC-5 and AC-6 assume retrieval works, but they test citation and staleness,
not retrieval. FR-5 appears only inside AC-1 ("counted per FR-5"), which never checks that counting
is consistent. FR-11 (four ways to create a summary) has no AC either (minor on its own: the
superseded doc had none). As written, retrieval and the document↔code link can be delivered or
missed with every AC green.
*Fix:* Add an AC for FR-1: given a named section of each in-scope document kind, the agent gets
that section and its citation without the rest of the document. Add an AC for FR-2: given a plan
step from a finished unit, the code it changed is returned, and given that code, the plan step and
the requirement behind it are returned. Add an AC for FR-5: two runs of the same unit report tokens
with the same components (resumes, nested agents, gate). Either add an AC for FR-11 or mark it
explicitly unaccepted in this feature.

### Major

**M-1 — AC-1's baseline is the ledger record, and the ledger cannot supply it.** *Verified*: the
U20–U32 Cost cells in `docs/plans/connected-code-graph-coordination.md` read "—" for U26 and
U28–U31. U21 is "resume leg" only and U20 is two legs ("83k + 137k"). The document's own Problem
section says ledgers count tokens inconsistently (FR-5's reason to exist). AC-1 compares FR-5-counted
replays with "recorded originals" that were not counted per FR-5, so 5 of the 13 units have no
number and the rest are mixed.
*Fix:* "The baseline for each replayed unit is a replay of the same unit without retrieval, counted
per FR-5. Recorded ledger costs are context only."

**M-2 — AC-2–AC-4 (quality held) trace to no FR and have no control or threshold.** *Verified* by
reading. The Intent's second success measure ("review quality does not [go down]") has no FR, so
AC-2–AC-4 trace only to Intent. AC-2 compares one LLM replay against history: a gate run can miss a
finding by chance, and with no replay-without-retrieval control, a miss cannot be attributed to
retrieval. AC-3 has no observation window, no definition of a "defect found after a gate", and no
measured "before". AC-4 has no "before" rate on record. Nothing asserts that tokens actually go
down: AC-1 only *sets* a target.
*Fix:* Add an FR for "review and test quality is no worse with retrieval than without". Make AC-2
compare against a no-retrieval replay of the same unit, with a stated number of repetitions. Give
AC-3 a window and a recorded pre-retrieval rate. Add an AC: "after the target is set, delivered
units meet it". Route the method to `data-scientist`.

**M-3 — FR-1 makes HISTORY.md retrievable, and Out of scope excludes it.** *Verified* against root
`AGENTS.md`'s context-file convention, which names `<module>/docs/HISTORY.md` as the home for
history moved out of a context file. FR-1 makes "any document that receives information moved out
of an `AGENTS.md` file (FR-4)" a retrieval target. Out of scope excludes `HISTORY.md` as
knowledge-base content. An FR-4 move into HISTORY.md therefore makes it both in scope and out of
scope.
*Fix:* "History moved out of an `AGENTS.md` goes to `HISTORY.md`, which stays outside retrieval.
Only non-history information moves to retrievable homes." Alternatively, make HISTORY.md an
explicit exception in Out of scope.

**M-4 — FR-8 removes what collision rule 5 protects without stating a replacement, and is silent
on existing families.** *Verified.* Rule 5 keeps gated text intact because gates and reviews cite
it. The unfiltered sweep (`git grep` for `Extended by|collision rule|ordinal on the slug`) shows
that code and READMEs also cite the successor chain: `decision-models/README.md:10`,
`decision-models/decisionmodels/__init__.py:3` and `model-bench/modelbench/decisions.py:2`
("as extended by plan2 v1.1 §3.2…"). FR-8 does not say whether the text a gate approved must stay
recoverable. AC-7's "any active plan family gets one current design" implies consolidating existing
families (typed-decision-models has six plans). That contradicts
`docs/requirements/document-size-discipline.md` Out of scope, "A dedicated retrofit pass over
existing documents (FR-7)", and leaves those code citations dangling. FR-8 also binds the solution:
FR-7 alone could be met by a derived current view without changing the convention.
*Fix:* Add three clauses. (a) "The text a gate approved stays recoverable, and existing citations
to it still resolve." (b) "Existing active families are / are not consolidated." If they are,
state that this overrides document-size-discipline's no-retrofit line. (c) Decide whether FR-8 is
a stakeholder-mandated mechanism or just FR-7's outcome, and log the answer.

**M-5 — FR-4 changes the context-file convention without saying so, and its AC cannot fail on
size.** *Verified* against root `AGENTS.md`. The convention's question is "what is true now, and
what will bite me". FR-4's core is "what will bite, and where to look", which drops "what is true
now". That may be intended, but unlike FR-8 it is not declared as a supersession. "Short" has no
measure (the convention gives ~2,500 words as a smell), and AC-11 checks only that moved content is
retrievable and not lost: an `AGENTS.md` that does not shrink passes it. "No information is lost"
has no check method. FR-4 also names only `AGENTS.md`, while the convention covers `CLAUDE.md` and
`.kiro/steering/*.md` too.
*Fix:* State whether FR-4 supersedes the convention's question. Give "short" a measure, or make it
the existing smell. Add to AC-11 "every moved item is listed with its new home". Name which
context-file kinds FR-4 covers.

**M-6 — One document joins two features with different components, users and success measures.**
*Inferred* from scope, plus the close rule in root `AGENTS.md` ("a feature is one backlog item,
else one requirements document"). FR-1–FR-8 serve agents reading repo files and are proven by
replaying C-1001 units. FR-9–FR-12 are a falkor-chat graph capability for humans and workflows.
None of AC-1–AC-4 measures their Intent ("save time and compute"), and no FR links them. One
document means one Ready-for-design flip and one close, so chat summaries cannot close until the
replay study does. `Tracks: —` names no component backlog for either half.
*Fix:* Either split the summaries back out as their own requirements document (in `falkor-chat`)
and cite it, or declare two independently closable tracks, each with its own success AC and its
owning component's backlog ID, and state whether FR-9 serves FR-1/FR-7.

**M-7 — FR-6's scope against FR-12 is ambiguous, and AC-6 ignores code.** *Verified* by reading.
FR-6 ("never silently out of date… says it is stale") and FR-12 (snapshot; staleness only
*detectable*) give different freshness rules. Only the sub-heading separates them. Team documents
are expected to live in `ws:agent-team`, a falkor-chat workspace (Intent), and FR-11 lets an agent
create a summary, so a summary over team documents could fall under either rule. AC-6 covers only
an edited document. A code location cited under FR-2/FR-3 that moved after indexing is never tested.
*Fix:* "FR-6 applies to retrieved parts of repo files and code locations. Summaries, including any
over team documents, follow FR-12." Extend AC-6 to a code location changed after indexing.

### Minor

**m-1 — Problem evidence is accurate as of a commit it does not name, and is already stale.**
*Verified.* At `1784168b`: impl review 28,765 words with 22 passes, family 129,004 words, plan plus
plan2–5 = 25,339, `-ml` 9,581, root + `model-bench` `AGENTS.md` = 5,858. All match. Ledger was
10,125 (claimed 10.7k). At HEAD plus the working tree: impl review 32,534 / 25 passes, ledger
12,626, family ~153k tracked, and plan6 (7,025 words) is a fifth extending successor. Cost-column
claims (150–330k typical, inconsistent counting) match.
*Fix:* Pin the measurement commit in the Problem section.

**m-2 — The fold-in dropped summary-nodes OQ3 and the comparison basis.** *Verified* against
`falkor-chat/docs/requirements/summary-nodes.md`. All four FRs, the four out-of-scope lines, the
three ACs and the five decisions carry over. OQ1 became an Out-of-scope line and OQ2 a decision.
Two things did not: OQ3 ("does anything summarize threads/channels today?", the current-state
baseline), and FR-4's "compared to the covered nodes' own timestamps". FR-6 vs FR-12 consistency
is M-7.
*Fix:* Carry OQ3 as an open question or a stated current-state fact. Say against what a summary's
time is compared.

**m-3 — Relation to the knowledge-base strategy is unstated, and BACKLOG.md is unplaced.**
*Verified.* `claude/docs/requirements/agent-knowledge-base-strategy.md` Out of scope reads "this
document stays scoped to agent on-demand knowledge bases", with BACKLOG-to-graph explicitly
deferred. This document adds project documents to the same retrieval substrate without citing that
boundary, and never says whether `BACKLOG.md` (read whole) is a retrieval target.
*Fix:* Add "Extends the knowledge-base strategy's corpus to team documents. `BACKLOG.md` stays out
of scope (cite)."

**m-4 — AC-7's sweep scope and FR-8's kind scope are loose.** *Verified* by an unfiltered sweep.
Rule 5 is stated only in root `AGENTS.md` (lines 233–258, plus references at 195, 216 and 273). No
agent prompt and not `document-size-discipline.md` restates it. AC-7 names only "convention or
agent prompt", so the README/code citations in M-4 fall outside it. FR-8 says "plans"; FR-1 says
plans include `-ml`/`-graph` notes, which have their own successors (e.g.
`connected-code-graph-eval-protocol2-ml.md`).
*Fix:* Make AC-7 an unfiltered repo sweep with every hit owned, and state whether FR-8 covers
`-ml`/`-graph` notes.

**m-5 — AC-5 has no sampling rule.** "Given any agent output" does not say which outputs are
inspected or how a claim "drawn from" a lookup is identified.
*Fix:* "On every gate output of the replayed units, each claim citing a retrieved part names its
section or location."

### Nit

**n-1 — Decision-log pointers are stale.** Five entries still read "TBD" ("FRs TBD", "ACs TBD",
"AC TBD", "baseline AC TBD", "extent TBD") although the FRs/ACs now exist. The log is append-only
(document-size-discipline FR-5), so leave them. Otherwise the log is compliant: one question → one
answer, and the one reversal names the entry it overturns.

### Growth-stops check

New document, so there is no revision growth to classify. The decision-log shape complies (n-1
aside). No restated facts were found: document-size-discipline, K-102 and the KB strategy are each
cited in a clause, and the summary-nodes content is carried because its source is superseded.

### What's solid

- The Problem section is measured, sourced and reproducible (m-1 only pins the commit).
- The supersession bookkeeping is right in both headers (`Supersedes:` / `Superseded by:`).
- FR-1–FR-3 and FR-9–FR-12 are stated as WHAT, not HOW (embedding and vector are removed from the
  carried summary FRs).
- The quality-held bar is chosen deliberately, and the target is set from measurement rather than
  guessed.

### Open questions

- M-4(b) and M-5's supersession of the convention's question are stakeholder rulings, not edits.
- M-6 is a stakeholder call: split or two tracks.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: B-1 and M-1.

- (addition) In a requirements gate, every FR is named by at least one AC, or is explicitly marked
  unaccepted. An AC that compares against recorded history (ledger costs, past findings) is checked
  against that record's actual completeness.

## Pass 2

**Scope.** Revised `docs/requirements/agent-reading-cost.md` (2,380 words, was 1,957) and the
restored `falkor-chat/docs/requirements/summary-nodes.md`, which matches HEAD: `Status:
Interviewing`, clean in `git status`.
**Verdict: approve with suggestions.** No blocker or major remains. The new findings below are
minor, and none blocks the Ready-for-design flip.

CPG: not applicable — a requirements document with no code-level component under review.

### Pass 1 dispositions

- B-1 — fixed: AC-12 → FR-1, AC-13 → FR-2, AC-14 → FR-5. FR-11 left with the split. Every live FR
  is now named by an AC (checked FR-1–FR-8, FR-13, FR-14).
- M-1 — fixed: AC-1's before figure is the next 10–15 units counted per FR-5. Recorded costs no
  longer feed it.
- M-2 — fixed by stakeholder ruling: FR-13 and FR-14 added. AC-3 has a window and a before
  bound, AC-16 asserts the target. The missing no-retrieval control is accepted and stated in
  AC-2 ("catches a large drop only").
- M-3 — fixed: FR-1 keeps `HISTORY.md` outside retrieval, consistent with Out of scope.
- M-4 — fixed: FR-7/FR-8 apply to families started after delivery, gated text stays recoverable,
  existing citations still resolve, and existing families are out of scope. That agrees with
  document-size-discipline's no-retrofit line.
- M-5 — fixed: FR-4 keeps the convention's question, "smaller" is measured against the start of
  delivery, and AC-11 lists moved items. One residue is N2-3.
- M-6 — superseded: the summaries were split back out (Out of scope line; summary-nodes restored).
- M-7 — fixed: FR-6 and AC-6 cover code locations. The FR-12 overlap left with the split.
- m-1 — fixed: pinned to `1784168b`, ledger 10.1k (matches my Pass 1 count). One new claim is
  wrong; see N2-1.
- m-2 — superseded by the split; OQ3 is back in summary-nodes.
- m-3 — fixed: the corpus extension is stated, and `BACKLOG.md` is out of scope with a citation.
- m-4 — fixed: AC-7 is a whole-repo sweep with owners, and `-ml`/`-graph` notes are part of a
  plan family.
- m-5 — fixed: AC-5 is scoped to the AC-2 re-runs.
- n-1 — not fixed, as recommended (the log is append-only).

### Stakeholder answers folded in

- Stay one feature? → No, summaries split out (Out of scope).
- Baseline, given that no recorded cost covers a gate? → Count the next 10–15 units per FR-5
  (AC-1, AC-16).
- No-retrieval gate rerun for AC-2? → No; recorded blocker/major findings only (AC-2).
- Escape check only as a gross-regression guardrail? → Yes (AC-3).
- Existing plan families? → New work only; gated text recoverable (FR-7, FR-8, AC-7).
- How is "short" measured? → Smaller than at the start of delivery, moved items listed (FR-4,
  AC-11).
- Do `-ml`/`-graph` notes belong to a plan family? → Yes. Is Kiro steering in FR-4? → No (FR-7,
  Out of scope).

### New findings

**N2-1 (minor) — The Problem section says a gate's cost is "never recorded", which is false.**
*Verified*: `docs/plans/typed-decision-models-coordination.md` records a separate gate cost in 6
Cost cells (e.g. U0 "gate 201k / 47", P4 "gate 239k / 37"). It is the connected-code-graph
U20–U32 rows that record none, and that is what the data-scientist consult found.
*Fix:* "A gate's own cost is recorded only sometimes, and never for connected-code-graph U20–U32."

**N2-2 (minor) — AC-11 makes every `AGENTS.md` shrink, including retired and already-minimal
ones.** *Verified*: `git ls-files '*AGENTS.md'` includes `deprecated/salesperson/AGENTS.md` and
`deprecated/opencode/agents/severino/AGENTS.md`, which root `AGENTS.md` says are never maintained
or refactored. It also includes `opencode/AGENTS.md` at 148 words. AC-11 as written can only pass
by editing retired files, or by trimming a file that is already all core.
*Fix:* "Each `AGENTS.md` outside `deprecated/` is smaller than at the start of delivery, unless it
already holds only the core."

**N2-3 (minor) — FR-4's change is not in the decision log.** *Verified* by reading lines 159–160
and 175–178. FR-4 no longer answers the logged "What happens to the automatic `AGENTS.md` load? →
short core (what bites, where to look)". It now keeps the convention's "what is true now", and no
entry overturns or refines that decision. A design reader comparing the log with FR-4 cannot tell
which one is authoritative.
*Fix:* Append one entry: "Does FR-4 keep the context-file convention's question? → Yes. Refines
'What happens to the automatic `AGENTS.md` load?'."

**N2-4 (minor) — AC-2 and AC-13 re-run past units but do not pin the documents they retrieve.**
*Inferred* from FR-6 and AC-2. AC-2 pins the deliverable version, but retrieval serves the
*current* plan and requirements: the connected-code-graph family has since gained successors such
as `connected-code-graph-p0-2.md` and `-bundle-spec2.md`. A re-run gate could then report findings
against newer design text, or get stale notices. Either way, a miss or a hit would not be
comparable with the original gate.
*Fix:* "AC-2's and AC-13's retrieval serves the documents as of the original gate's commit."

**N2-5 (minor, growth-stops) — Rationale and facts stated twice.** *Verified* by reading:
- AC-2, AC-3 and AC-4 each end with a sentence of rationale ("this catches a large drop only", and
  similar) that restates their decision-log answers.
- FR-7's "Existing plan families stay as they are" restates the Out-of-scope line.
*Fix:* Drop the three rationale sentences and FR-7's last sentence, leaving the decision log and
Out of scope as their homes.

**N2-6 (nit) — Shape.**
- One decision-log entry asks two questions ("`-ml`/`-graph`… Kiro…"); split it next time, since
  the log is append-only.
- FR-7's first line is 138 characters, not wrapped like the rest of the document.
- The ACs are listed out of numeric order (AC-12–AC-16 sit before AC-7 and AC-11).

### Growth-stops check

+423 words net. About 390 words left with the summaries. The roughly 800 words added break down
as follows:
- **New scope:** FR-13, FR-14, AC-12, AC-13, AC-14 and AC-16 fill gaps that B-1 and M-2 found.
- **Rework, mostly justified:** AC-2–AC-4 now carry the window, threshold and control that M-2
  asked for. The growth beyond that is the rationale sentences in N2-5.
- **Decision log:** nine appended entries, as the append-only rule allows. The data-scientist
  consult entry lists findings, but it has no other home.

No restated history otherwise.

### Proposed REVIEW.md lines

Unchanged from Pass 1.

## Pass 3

**Scope.** This pass reviews the implementation plan `docs/plans/agent-reading-cost.md` **v1.1**
(`architect`, untracked, 7,269 words) against `docs/requirements/agent-reading-cost.md` (Ready for
design, `53a21487`), root `AGENTS.md`, root `REVIEW.md` and `falkor-chat/REVIEW.md`. Feasibility
claims were checked against falkor-chat source, live `ws:agent-team`, `~/.claude/projects`
transcripts, the C-1001 ledger and review, and `git grep`. Nothing was run against
`ws:team-docs`, because it does not exist yet.

**Verdict: needs changes.** There are no blockers and four majors. M3-1 and M3-2 break the
replay and token-count designs on the actual U20–U32 record. M3-3 leaves FR-8's consistency
sweep unable to see the commit rule it changes. M3-4 is a sequencing gap. Twelve minors, three
nits.

CPG: used cpg_falkorchat — confirmed that `IngestionPipeline` has one production construction
site (`app.py:619`); the graph is stale, so the result was cross-checked against source.

### Coverage and scope

- **Every FR and AC maps to a unit and a test.** *Verified* against §3 and §4:
  - FR-1 → B1/B4/B5/C1 (AC-12) · FR-2 → B6 (AC-13) · FR-3 → C1 (AC-5) · FR-4 → C2/C3 (AC-11)
  - FR-5 → A2 (AC-14) · FR-6 → B1/B4/B6 (AC-6) · FR-7 → B1 `versions`/C4 (AC-7)
  - FR-8 → C4 (AC-7) · FR-13 → D2/D3 (AC-2–AC-4) · FR-14 → A5/D3 (AC-1, AC-16)
- **Units without a requirement:**
  - B2 and B3 are enabling infrastructure for FR-1 and FR-6, which is acceptable.
  - E1 is not covered by any requirement (m3-9).
- **Out of scope is respected.** C-1001 is read-only. Reviews, ledgers, `HISTORY.md` and
  `BACKLOG.md` are kept out of `search`. `HISTORY.md` stays reachable by address only, which is
  consistent with FR-1. Existing families keep rule 5 (§2.8 point 4). No Kiro steering file is
  touched.
- **Stakeholder answers are reflected faithfully:**
  - Q1: §2.3 has one workspace, path tags and `--component`/`--all`.
  - Q2: §2.8 commits each version at gate dispatch, for new families only.
  - Q3: §4 replays every pass with a blocker or major.
  - Q4: A4 makes no reader-visible change before A5.
  - Q5: the Phase A precondition plus the §5 contingency.
  - Q2's placement is M3-3 below. Q3's replay count is unverified (M3-1).

### Major

**M3-1 — `ar replay prepare` assumes one gate pass is one spawned agent whose first user entry
is its brief. The U20–U32 record contradicts that.** *Verified* in the transcripts and the
ledger:
- `agent-a72f2b8992f574958` gated S01 Pass 1. S05 Pass 1 then arrived as a SendMessage leg *while
  S01 was running* (19:56), followed by S05 Passes 2 and 3 (21:56, 22:05). §2.9 step 1 would
  hand all four replays the S01 brief, with base commits taken from the file's start.
- U22's `graph-dba` gate is marked "resumed".
- U26, U28 and U30 have no gate id in `docs/plans/connected-code-graph-coordination.md`.
- Original briefs name the main tree's absolute root ("repo root `$HOME/prg/…`"). A
  replay run in a worktree would read today's files, which defeats AC-2's as-of fidelity.
- Docs-only deliverables, and other documents the gate read, were uncommitted at gate time
  (docs-only batching), so "newest commit before the start" is not what the gate saw. Builders
  also wrote some files through Bash, which a `Write`/`Edit` replay cannot reconstruct.

*Fix:*
- Make the replay unit an (agent id, leg). The brief is the leg's user text, and the base commit
  comes from the leg's timestamp.
- Recover missing gate ids from the parent session's `Agent` `tool_use` ids.
- Rewrite the repo root path in the brief to the worktree.
- B7's probes must include a docs-only pass (S04 or S05) and a resumed leg, not only U27.
- State how a halted reconstruction is counted in AC-2's denominator.
- A3(f) owns whether a cold spawn is comparable with a resumed leg.

**M3-2 — FR-5's "a run is one `agent-<id>`" cannot attribute an agent that served several
units.** *Verified*: `a72f…` is the U20 gate and also the U24 gate (ledger rows U20 and U24).
§2.6 would give both units the same three-leg total. "A gate is its own run" does not prevent
this, because cross-unit resumes are current `teco` practice. Q4 also forbids changing that
practice during A4.

*Fix:* `ar tokens` attributes **legs** (split at each string user entry). Each record carries
`--unit` per leg range. A2 adds a fixture with one agent serving two units, and its expected
figures are written by hand.

**M3-3 — The docs-only commit-once rule lives in five places. C4 edits one, and its sweep cannot
see the rest.** *Verified*: the rule is stated in:
- `claude/AGENTS.md:241-249`;
- `claude/teco/teco.md:146` and `:175`;
- `claude/tico/tico.md` (consult rule, l.44, and "Coordinating a docs-only chain");
- `claude/docs/manuals/agent-team.md:390`, `:466`.

`docs/requirements/commit-granularity.md` is `archived`. C4's pattern matches none of the
commit-rule wording, so FR-8 ("every … prompt … that relies on the rule agrees") fails silently.
`tico`, which runs the requirements → plan → review chains, is the agent that would break Q2.

*Fix:*
- C4 names all five homes.
- Add a second unfiltered pattern: `docs-only (chain|batch)|terminal state|commit(s|ted)? once`.
- Ask `tico` to log the Q2 exception in this requirements document's decision log, since it
  amends a standing convention. Today it is recorded only in the plan.

**M3-4 — B4's live acceptance depends on K-102 (B3), but B3 is not in B4's Depends column.**
*Verified* live: `ws:agent-team` has 14 `failed` documents.
- 11 have an unembedded chunk, so the embed itself failed.
- 3 are fully embedded, so something after the embed failed them, most likely extraction.

Turning extraction off on 8201 removes only the second class. `:8200` keeps extracting on the
same one-model GPU (`falkor-chat/AGENTS.md`; K-070). A ~3,000-unit rebuild can then end with
units left uncovered for K-102's reason rather than B4's.

*Fix:* B4 depends on B3's RCA. B3's RCA must cover the embed-failure class. Alternatively, B4's
acceptance becomes "zero uncovered after `ar sync --reconcile`", with the first-pass uncovered
count recorded.

### Minor

**m3-1 — B2 names `app.py:292` as a construction site. It is not code.** *Verified*: lines
285–296 are the `::` example in `create_app`'s docstring. Source and `cpg_falkorchat` both show
one production site, `app.py:619`. *Fix:* gate only `_build_default_app` and leave
`create_app`'s injected parameter alone.

**m3-2 — `start_team_docs.sh` duplicates `start_agent_team.sh`.** *Verified*: the existing
script already takes `FALKORCHAT_WS_ID` and `AGENT_TEAM_PORT`, and its own example uses
`AGENT_TEAM_PORT=8201`. *Fix:* pass the flag through `start_agent_team.sh`, or write a wrapper
of a few lines. Skip the Agent-roster seed only if it causes harm.

**m3-3 — The serialization order is not encoded.** *Verified* in the §3 table: after C2, units
C1, C3 and C4 have no mutual dependency, yet C3 and C4 both edit `claude/AGENTS.md`. C4's text
hedges ("or its §2.7 home if C3 has already moved it"). *Fix:* fix the order in the Depends
column, e.g. C2 → C4 → C3 → C1.

**m3-4 — The item-id regex rejects the plan's own examples.** *Verified* by running it:
`P0-08` → `P0`, `N2-1` → `N2`, `P2-M1` → `P2`. A trailing period is also absorbed (`S09.`).
AC-13's aliases come from `P0-08`. *Fix:* the B1 tests must include a coverage probe over id
shapes: multi-segment ids, letter-after-dash ids and sentence-final punctuation. The regex
itself is the implementer's choice.

**m3-5 — AC-11's no-loss check only verifies verbatim moves, and has no control.** `ar find`
matches exact text. C3 is not specified as a pure relocation the way C2 is. *Fix:* C3 also
relocates first and rewords in a later commit. The control is a planted removed paragraph with
no manifest row, which must turn the check red.

**m3-6 — The root history home contradicts root `AGENTS.md`.** That file says
`docs/HISTORY.md` is `cpg/`'s own log and that other entries go elsewhere. The plan sends root
`AGENTS.md` history and C2's manifest there. Also, `salesperson/` has no `docs/HISTORY.md` (its
home is falkor-chat's). *Fix:* pick the home explicitly. If it stays `docs/HISTORY.md`, C2
rewrites the `cpg/` ownership statement.

**m3-7 — Lazy sync makes `ar search` block.** Each search parses ~155 files and may wait up to
60 s per changed unit under the lock. The hash-compare label already makes freshness explicit.
Two related gaps:
- `limit` is capped at 200 (`schemas.MAX_LIST_LIMIT`), which bounds the over-fetch a scoped
  search can use. B5 should state that ceiling.
- Every REST delete leaves an audit node (`repository.delete_document`).

*Fix:* search posts changed units and does not wait for them. `--rebuild` drops and
re-bootstraps the graph rather than deleting one document at a time.

**m3-8 — The retention precondition is not met today, and the date estimate is inferred.**
*Verified*:
- No `cleanupPeriodDays` is set in `~/.claude/settings*.json`.
- The oldest transcript is 18 days old (2026-10-14), not ~30, so the "late November" estimate is
  unsupported.
- Headless runs from another directory land under another project slug (seen for the scratchpad
  probes), so D2's worktree replays escape the default `--projects-dir`.

*Fix:* the A2-dispatch check copies at once if the setting is absent. The replay worktree's
slug is passed to `ar tokens`.

**m3-9 — E1 has no requirement.** §2.5 says "No acceptance criterion depends on symbols". *Fix:*
file E1 as a trigger-gated backlog item rather than a unit of this plan.

**m3-10 — AC-4's test invents a rule.** "≥1 mutant per changed function" is not "the same rule
applied before retrieval". *Fix:* take the after-rule from A5's recorded before practice.

**m3-11 — The C4 sweep is 2,467 hits across 382 files.** *Verified* by running the pattern
verbatim at the repo root. One table row per hit is impractical. *Fix:* rule per file (each
archived file is no-change as a block, with the list of hits), keeping the sweep unfiltered.

**m3-12 (growth-stops) — Restated facts and rationale.**
- §1 restates falkor-chat live-spec behaviour (chunk sizes, search fields, REST routes) instead
  of citing `falkor-chat/docs/SERVER.md` §1.4 and §2.2 (root `REVIEW.md`, live-spec rule).
- §1's `AGENTS.md` word table duplicates A5's start counts and is already off
  (`decision-models/AGENTS.md` is 665 at HEAD, not 631).
- §2.3's per-component-workspace argument and "This is still the corpus extension…" argue a
  settled Q1.
- §7 restates §3.

*Fix:* cite, or drop. Growth from v1.0 to v1.1 cannot be classified, because v1.0 was not
retained.

### Nit

- Nested runs: `.meta.json` carries `parentAgentId` (seen on a `spawnDepth: 2` file), which is
  simpler than scanning for `toolUseId`.
- `runs.jsonl` records should carry the counter's commit sha, so the baseline is pinned
  (`falkor-chat/REVIEW.md`).
- §2.3/§5 should cite the open K-070, K-091 and K-064 (the extract/embed swap and the
  update-detection signals that sync will trigger).

### Methodology risks for A3's `data-scientist` gate (named, not adjudicated)

- **Agents serving several units, and resumed legs.** These affect both the FR-5 attribution
  (M3-2) and AC-2's comparability (M3-1).
- **Unit mix shifts between the windows.** The baseline is the next C-1001 units; the after
  window covers later phases. Docs-only and code units differ, and roles have small n per window.
- **The headline is dominated by cache reads.** U24 is 13.2M of 14.15M, a figure I reproduced
  exactly from the transcript. A reading cut may show up mainly in priced-down cache reads.
- **Uncontrolled changes during the A4 window.** Other features may change what agents read.
  A5 should log them.
- **AC-2 selection bias.** Only passes that can be reconstructed get replayed.
- **AC-3's before window mixes two families** with different review regimes.

### What's solid

- Text always comes from git or the file, and never from the index. FR-6 holds by construction,
  and as-of-commit becomes one `--at`.
- One workspace with path tags is the right call, given one workspace per process. *Verified*:
  `config.WS_ID` is a process constant, and `get_context()` returns it.
- The extraction flag is feasible as described. *Verified*: with `ingestion_pipeline=None`, no
  extract job is scheduled and `pendingJobs` counts embeds only (`background.py:208-227`). Update
  detection falls back to the embed worker's repository.
- FR-5's per-message rule is grounded. *Verified*: U24 recomputes to 14,153,036, and resumes
  append to the same file.
- Deriving the plan↔code link from git rather than storing it is proportionate. The commit
  subjects and step-table shape exist as described (`6f4e34f9`, `p0-2.md` §3).
- Making C2 a pure relocation turns no-loss into a diff.

### Open questions

- M3-2: may `teco` keep resuming gate agents across units during the measurement windows, or
  must the counter split legs? The fix above assumes the counter splits them.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: M3-1, M3-2.

- (addition) A plan that counts, replays or attributes past agent runs checks its run unit
  against real transcripts and ledgers: resumed legs, one agent serving several units, missing
  ids, and briefs that carry absolute paths.

## Pass 4

**Scope.** This pass reviews the plan `docs/plans/agent-reading-cost.md` **v1.2** (untracked,
6,402 words, down from 7,269), with the requirements decision log's three new entries
(uncommitted).

**Verdict: needs changes.** There is one major, N4-1: the replay verification of M3-1's fix
does not fit how gates actually read files. There are four new minors and one nit. Nothing found
here touches Phase A, so A2 and A3 can start.

CPG: considered, not relevant — the one code change is the single `app.py:619` site, which was
checked in Pass 3. This pass checks transcripts, scripts and sweeps.

### Pass 3 dispositions

- **M3-1** — fixed in part.
  - Fixed, rechecked against §2.9 and B7:
    - the replay unit is now one leg;
    - missing gate ids are recovered through `tool_use`;
    - the root path is rewritten;
    - a docs-only probe and a resumed-leg probe are added;
    - the AC-2 denominator is stated.
  - Not fixed: the layered-tree verification (N4-1).
- **M3-2** — fixed. §2.6 legs, attribution and A2's two-unit fixture match the stakeholder's
  per-leg decision. Residue: N4-2.
- **M3-3** — fixed in part.
  - Fixed: C4 writes into the rule's homes, adds a second sweep, and has the decision-log
    precondition (entry present).
  - Not fixed: the sweep pattern I proposed misses one home (N4-3).
- **M3-4** — fixed. B4 now depends on "B3 (RCA)", and its acceptance is zero uncovered after
  `--reconcile`. B3's RCA covers both failure classes and the `:8200` decision.
- **m3-1** — fixed. §1, §2.3 and B2 cite `:619` only.
- **m3-2** — fixed. The wrapper's exports survive `start_agent_team.sh`'s `exec uvicorn`, which
  runs with no `env -i`. Residue: N4-5.
- **m3-3** — fixed. C2 → C4 → C3 → C1 is encoded in the Depends column.
- **m3-4** — fixed. §4 item 2 has an id-shape coverage probe.
- **m3-5** — fixed. Every file moves in two commits, and AC-11 has a planted-paragraph control.
- **m3-6** — fixed. The manifests go to `agent-reading/docs/HISTORY.md`, and `salesperson/`
  history goes to falkor-chat. Residue: N4-4.
- **m3-7** — fixed. Search no longer waits, the 200 ceiling is stated, and `--rebuild` uses
  `GRAPH.DELETE` (guarded in C1).
- **m3-8** — fixed. The copy happens at once when the setting is absent, and replay cost uses the
  worktree's slug.
- **m3-9** — fixed. The connected-code-graph `SymbolSource` moves to the backlog (§6).
- **m3-10** — fixed. AC-4 follows A5's recorded before practice.
- **m3-11** — fixed. The rule-5 sweep is ruled per file.
- **m3-12** — fixed. §1 cites `falkor-chat/docs/SERVER.md` §1.4, §1.12 and §2.2. The word table
  and the §7 restatement are gone, and the §2.3 argument is trimmed.
- **Nits** — fixed: `parentAgentId`, the counter sha in records, and the K-070/K-064/K-091 cites.
- **Open question** (cross-unit resumes during the windows?) → yes, attributed per leg. Folded
  into §2.6 and A4, and logged.

### Growth-stops check

The plan shrank by 867 words. New text is rework answering Pass 3: §2.6 legs, §2.9 replay, C4's
two sweeps. It replaces the earlier text and is not stacked on it. The revision note is one dated
line. There is no finding→resolution table. No restated history was found.

### New findings

**N4-1 (major) — Gates mostly read files through Bash, so the `Read`-results layer of the
replay verifies almost nothing.** *Verified* in six U20–U32 gate transcripts:

| Gate | `Read` calls | Bash file reads (`cat`, `sed -n`, `head`, …) |
|---|---|---|
| U27 `a582…` | 1 | 58 |
| U21 `a52a…` | 2 | 49 |
| U23 `a82b…` | 5 | 66 |
| U22 `a4d1…` | 1 | 51 |
| U32 `a596…` | 4 | 96 |
| `a72f…` | 7 | 40 |

This leaves two possible readings of §2.9 step 3, and both fail:
- **Only `Read` counts as "read".** Then nearly every file falls through to the builder layer
  plus the base commit, unchecked. That is the uncommitted-docs gap of M3-1 again.
- **Bash reads count as "read".** Then nearly every pass halts as not reconstructable, and under
  §2.9's denominator rule AC-2 cannot be met.

A further problem: the gate's own later reads of the review document show its own edits, not
the state at the start of the leg.

*Fix:*
- Verify against every capture made before the leg's first own write:
  - a `Read` whose `toolUseResult.file` has `numLines == totalLines` gives the whole raw content
    (that field is present, so line-number prefixes need no stripping);
  - partial reads and Bash `cat`/`sed`/`grep` outputs are line-level consistency checks.
- Per pass, list the files that are verified, consistent and never captured.
- B7's four probes report those rates. The stakeholder rules on D2's feasibility from them
  before D2 is dispatched.

**N4-2 (minor) — The leg rule splits every spawn in two.** *Verified*: all 273 subagent
transcripts on disk carry a harness `<system-reminder>` string user entry about 16 ms after the
spawn prompt. By §2.6's definition, `a72f…` therefore has five legs, and A2's done-condition
("splits into its four legs") turns red. *Fix:* entries starting with `<system-reminder>` do not
open a leg. Add a fixture for it.

**N4-3 (minor) — The commit-rule sweep misses a home.** *Verified*:
- `claude/teco/coordination-techniques.md:254-257` and `claude/teco/teco.md:125` state
  "docs-only-chain batching" exceptions. The hyphen defeats C4's pattern, which I proposed in
  Pass 3.
- `docs-only[- ](chain|batch)|terminal (state|commit)|commit(s|ted)? once` gives 122 hits in 46
  files. It adds `coordination-techniques.md`.
- §1 also says "five live homes" but lists four files.

*Fix:* use the widened pattern, list the homes by file and section (including `tico.md`'s
"Commit at document boundaries", l.68), and update the count.

**N4-4 (minor) — "History of a repo-wide convention is cited, not moved" is assumed, not
checked.** *Inferred* from §2.7. Not every history clause in root `AGENTS.md` has a requirements
document behind it. The context-file bullet's precedent cites `claude/cobb/kaizen/history.md`. An
item with no such home would simply be dropped, which breaks AC-11's "no information lost".
*Fix:* C2's manifest lists each cited-not-moved item with the address it resolves to (`ar get`).
An item that has no such address is moved like any other.

**N4-5 (minor) — A fresh `ws:team-docs` has no actor to ingest as.** *Traced, not run.*
- `services.py:1326-1329` raises `UnknownActorError` when `ctx.actor` (default `u1`) is neither a
  `User` nor an `Agent`.
- `start_agent_team.sh` seeds only `Agent` nodes, and no script creates `User u1` for that
  workspace. `ws:agent-team`'s `u1` exists, but no script created it.
- v1.1's actor check was dropped.
- `--rebuild` (`GRAPH.DELETE` plus `create_workspace.sh`) also drops the Agent roster.

*Fix:* the wrapper sets `FALKORCHAT_USER_ID` to a seeded agent id, or seeds a `User`.
`--rebuild` re-runs the same seed. B2 gets a test: an ingest on a fresh workspace returns 201.

**N4-6 (nit) — Three guard copies.** `guard-destructive-ops.sh` exists in `devops`, `graph-dba`
and `qa-engineer`. C1 should name which copies gain `ar sync --rebuild`.

### Proposed REVIEW.md lines

Pass 3's proposal stands. Add to it, from N4-1: "…and a reconstruction from transcripts is
checked against every tool the agent actually read with (Bash included), not only `Read`."

## Pass 5

**Scope.** This pass reviews the plan `docs/plans/agent-reading-cost.md` **v1.3** (untracked,
6,808 words, up from 6,402). Its claims were checked against the six gate transcripts sampled in
Pass 4, `falkor-chat/docs/QUERIES.md` §2, `seed_demo.sh`, `start_agent_team.sh`,
`bootstrap_schema.sh` and `claude/scripts/`.

**Verdict: approve with suggestions.** There are no blockers and no majors. Four new minors
follow (N5-1 to N5-4). N5-1 is the one to settle before B7 starts.

CPG: considered, not relevant — no code-level claim changed since Pass 4; this pass checks
transcripts, scripts and documents.

### Pass 4 dispositions

- **N4-1** — fixed. §2.9 step 3 now takes captures from Bash as well as `Read`, ignores captures
  made after the gate's own writes, and sorts every file into four classes. B7's probes report
  the rates, and the stakeholder rules before D2 runs (§3 table "sh rules on D2"). §4 item 6
  tests each rule. Residue: N5-1.
- **N4-2** — fixed. §2.6 says `<system-reminder>` never opens a leg, and §4 item 1 has a fixture
  for it.
- **N4-3** — fixed. §1 lists five files, including `coordination-techniques.md` and `tico.md`
  l.68. C4 uses the widened pattern. I re-ran it and got 122 hits in 46 files, which matches
  "about 120 in 46".
- **N4-4** — fixed in part. The manifest now gives an `ar get` address for every item that is
  cited rather than moved. The fallback home is N5-3.
- **N4-5** — fixed in part. B2 has `seed_team_docs.sh`, which ensures a `User ar-sync` the same
  way as `QUERIES.md` §2 and `seed_demo.sh` step 1. It has a positive and a negative test, and
  `--rebuild` re-seeds. Residue: N5-2.
- **N4-6** — fixed. *Verified*: `claude/scripts/guard-destructive-ops.sh` is the shared core.
  The `devops`, `graph-dba` and `qa-engineer` hooks `exec` it, and `guard-broad-bash.sh:67`
  calls it for `coder` and `tdd-engineer`.

### Growth-stops check

The plan grew by 406 words, and all of it is rework. Most of it is mechanism the fixes need:
§2.9's capture rules, B2's seed script, B7's rate reporting and stakeholder ruling, C1's guard
path, and §4's tests. One part restates facts (N5-4).

### New findings

**N5-1 (minor) — The closed Bash read set parses few gate commands, and a pass whose deliverable
was never captured still counts as replayed.** *Verified* by classifying every Bash call in the
six gates:

| Gate | Single closed-set command (a leading `cd` allowed) | Several chained reads, output concatenated | Piped, multi-line or scripted |
|---|---|---|---|
| `a72f…` | 2 | 1 | 40 |
| `a582…` (U27) | 9 | 3 | 55 |
| `a52a…` | 0 | 1 | 52 |
| `a82b…` | 7 | 3 | 56 |
| `a4d1…` | 6 | 4 | 35 |
| `a596…` | 13 | 9 | 93 |

The common shape is `cd …; f=…; sed -n a,bp $f; /usr/bin/grep -n X $f | cut -c1-200`. It has a
shell variable, several reads whose outputs run together, an absolute `grep` path and a `cut`
that truncates lines. Code gates also execute files (`bash $S status`, pytest, scripts copied to
a scratch directory) without printing them. Those files never reach any class, so the
never-captured list undercounts.

AC-2 still stays honest on the input side: the rates are shown to the stakeholder before D2. On
the output side, a pass counts as replayed even when its deliverable is never captured, and only
"contradicted" blocks it.

*Fix:*
- Parse the dominant shape: `;`/`&&` sequences, `var=path` assignments, `/usr/bin/grep`, and
  `| head`/`| cut -c` as truncating filters. Split concatenated output only where the command
  shapes allow it.
- List files the gate executed, or that unparsed commands mention, as never captured.
- D2 reports AC-2 per class. A pass counts only when its deliverable files are verified or
  consistent, unless the stakeholder's B7 ruling says otherwise.

**N5-2 (minor) — `start_team_docs.sh` seeds before FalkorDB is guaranteed to be up.** *Traced,
not run.*
- The wrapper runs `seed_team_docs.sh` (`create_workspace.sh` plus the `User` ensure) before it
  `exec`s `start_agent_team.sh`.
- `start_agent_team.sh`'s step 1 is the step that starts FalkorDB.
- `bootstrap_schema.sh:406` exits when `PING` gets no answer.

So a cold bring-up fails, including the one `ensure-environment.sh` runs.

*Fix:* the wrapper first waits for `PING` the way `start_agent_team.sh` step 1 does, or runs
`start_falkordb.sh`. Add a B2 check that bring-up succeeds with FalkorDB down.

**N5-3 (minor) — `claude/cobb/kaizen/history.md` is the wrong home for convention history
(§2.7).** Root `AGENTS.md` says modules use `<module>/docs/` and that `kaizen/` is for agent
folders only. `cobb`'s history is a dated log of `cobb`'s own maintenance actions. Moving older
convention prose into it backdates content into a log of someone else's actions. The same
applies to the `claude/` row.

*Fix:*
- `claude/` history → `claude/docs/HISTORY.md`. The `claude/docs/` tree already exists, and the
  module convention says "adopt when first needed".
- Repo-wide convention history that has no existing record → carried verbatim in C2's manifest
  entry in `agent-reading/docs/HISTORY.md`, which is already the manifest home and is read by
  lookup.

`cobb`'s history keeps only `cobb`'s own entries.

**N5-4 (minor, growth-stops) — §1 restates rules that §2.6 and §2.9 now own.** The new
"Transcripts" bullets (Bash-vs-`Read` counts, `toolUseResult.file`, the 16 ms reminder) repeat
facts that §2.6 (the leg rule) and §2.9 step 3 state as design. Their evidence is in this
review (Pass 4, N4-1 and N4-2). *Fix:* cut the three bullets to one clause citing Pass 4.

### Proposed REVIEW.md lines

Pass 4's proposal stands. Nothing is added.
