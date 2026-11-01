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
