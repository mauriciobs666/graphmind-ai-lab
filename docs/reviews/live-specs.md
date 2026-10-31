# Live specs — requirements review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1

**Scope.** `docs/requirements/live-specs.md` (untracked first version, Status: Interviewing,
read in full), checked against root `AGENTS.md` (Status set, archived-flip ownership, milestone
close, BACKLOG/HISTORY, growth-stops), `docs/requirements/document-size-discipline.md`,
`claude/teco/teco.md`, `claude/analyst/analyst.md`; "Problem & current state" figures re-measured
(`wc -w`, `git log`, `git rev-list`). Gate for the Ready-for-design flip (doc-size FR-9/AC-9).

**Verdict: needs changes** — one blocker (FR-2/FR-7 silently overturn `teco`'s recorded
documentation rules), four majors. The feature's intent is clear and the FR set is mostly
product-altitude; the fixes are requirement-text and decision-log additions, not a redesign.

CPG: not applicable — requirements/process document, no code-level component under review.

### Blocker

**B1 — FR-2 and FR-7 contradict `teco`'s recorded documentation rules with no supersession
requirement.** `claude/teco/teco.md:152` — "you curate; you don't write the docs yourself";
`:155` — "Docs ride in the brief … implementers update docs alongside code" in the same change;
`:24` — teco does no deep work (trivial single-file fixes only, `:48`). FR-7/FR-10 make `teco`
the *author* of the spec update and of a whole new model-bench spec; FR-2 makes the update happen
"once, at feature close" rather than per unit in the same change. Neither the FRs, the decision
log, nor AC-9 names these rules as overturned, so the architect inherits two contradictory
authorities and AC-9 ("no prompt still allows closing without the spec update") would pass while
`teco.md` still forbids teco from doing FR-7. **Fix:** either (a) add a requirement that
explicitly supersedes `teco.md` §Documentation curation for the live spec (who writes, when), with
a decision-log entry naming what it overturns and AC-9 extended to "no prompt still assigns the
spec update otherwise"; or (b) restate FR-7/FR-10 as "`teco` ensures the spec is updated (author
routed per its rules)" — a stakeholder call, so put it as a question before the flip.

### Major

**M1 — "Feature close" is undefined and collides with "milestone close".** Root `AGENTS.md`
(closed Status set; `archived` flipped by `teco` *at milestone close*) and `teco.md:156` freeze
documents at milestone close; FR-2, FR-7, FR-8, FR-11 and AC-4 key everything to "feature close"
and archive "the feature's documents" there. Nothing says whether a feature = a milestone, a
backlog item, or a requirements slug, nor whether archiving moves earlier. Consequence: AC-4 is
uncheckable (no event to observe) and the archived-flip timing rule is silently changed.
**Fix:** define feature close in one FR (e.g. "the point a requirements slug's delivery is
accepted"), and state its relation to milestone close — either "feature close precedes, and does
not replace, the milestone-close freeze" or a supersession of root `AGENTS.md`'s flip timing.

**M2 — "Once, at feature close" leaves the spec stale while dependent work proceeds.** FR-2
updates the spec only at close; FR-12 requires every new plan to delta against the spec. A second
feature planned in the same component while the first is open (today: `typed-decision-models`
touches `model-bench/` with an untracked `docs/plans/typed-decision-models2.md`) deltas against a
spec missing the first feature's behavior, and later units of a long feature read a spec that
contradicts the code. **Fix:** a requirement for the overlap case (e.g. an open feature's plan is
the authority for its own not-yet-specced behavior and a concurrent plan cites it), and an
explicit statement of how features in flight at delivery are treated (does FR-2 bind
`typed-decision-models`' close? does the FR-10 catch-up include it?).

**M3 — AC-1 is not checkable as written.** "Reports no behavior the code has that the spec omits"
has no granularity — the falkor-chat spec set (≈48k w) cannot state every behavior of the code,
so a literal check never passes and a lenient one never fails. The same gap leaves FR-7's
per-feature check without a pass/fail criterion (AC-4 only requires it be "recorded"). **Fix:**
define the altitude in an FR (e.g. externally observable contracts: interfaces, data shapes,
guarantees, invariants) and have AC-1 and AC-4 both judge against it; AC-4 should require the
check's verdict, not just its existence.

**M4 — FR-11 adoption is ambiguous for components with no live spec.** FR-2 says a feature
cannot close until "the component's live spec" states its behavior; FR-11 says other components
(`cpg/`, `mcp-monitor/`, `cypher-mcp/`, `salesperson/`, …) adopt at their first close, while Out
of scope excludes their catch-up. Does the first close in `mcp-monitor/` require writing a whole
spec (a de facto catch-up), a spec covering only the shipped feature, or nothing? Also unstated:
whether `salesperson/` (hosted by falkor-chat, no `docs/` of its own) falls under falkor-chat's
spec set. **Fix:** one sentence in FR-11 for the no-spec case, plus the `salesperson/` answer;
no AC covers FR-11 — add one.

### Minor

- **m1 — Factual claim overstated.** Line 24: K-079's Layer 1 contract is "recorded only in"
  the plan — it is also in `falkor-chat/server/tests/eval/nlq_scoring.py` docstring (its
  authoritative home per plan §3.4 D5) and `falkor-chat/docs/HISTORY.md:12`. The point (absent
  from the live spec — confirmed: no match in `DESIGN.md`/`SERVER.md`/`QUERIES.md`) stands;
  reword to "not in the live spec".
- **m2 — Reviews figure.** "reviews 118k w" = `small-model-benchmarking.md` + `-impl` (61,340 +
  57,020); omits `-ml` (11,093; total 129,453) and `model-bench/docs/reviews/` (19 files). Say
  which files, or use the total.
- **m3 — AC-3 vs today's spec, and what counts as history.** `DESIGN.md:14` carries a revision
  note and the spec set has ~120 `K-0NN`/date hits; FR-10 doesn't say the catch-up strips them.
  Some `K-` refs are open-work pointers (`DESIGN.md:325`, K-029 residual) — current state, not
  history. State in FR-3/FR-10 that the catch-up removes history and whether open-item pointers
  are allowed.
- **m4 — Untested FR clauses.** FR-9's "removes rules now enforced elsewhere" and FR-8's owner
  of applying proposed lines have no AC/owner; FR-6's "team reviewer" is defined only by AC-6's
  list (omits `cobb`, which reviews agents/skills — intended?). Name the applier and the reviewer
  set in the FRs.

### Nits

- **n1 — Decision log renders as one paragraph** (lines 108–129: no list markers or blank
  lines); use `- ` bullets as `document-size-discipline.md` does.
- **n2 — Rationale in log entries.** Entries 1–3 and 5 carry reasons ("nobody needs to read
  them…", "for later RCA…", "the emerging convention (Claude Code Review, Qodo, Devin)"); FR-5 of
  doc-size wants question → answer + FR/AC only.
- **n3 — Restated facts.** "Related" restates doc-size's out-of-scope list and FR-1 restates the
  existing "frozen documents stay, `archived`" rule; cite instead.

### Growth-stops check (doc-size FR-8)

First version (untracked, no prior revision) — no rework growth to judge. Decision log is
append-only and each entry names its FR/AC; shape issues are n1/n2. Restated facts: n3. No
finding→resolution content. Scope overlap with doc-size: FR-12 constrains a new plan's content at
first version and FR-10 is a one-time retrofit — both areas doc-size explicitly left out of scope,
so complementary, not contradictory; FR-12 extends doc-size FR-6 ("one home per fact") to the
live spec, consistent.

### Verified

Word counts (`DESIGN.md` 8,519, `SERVER.md` 9,806, `QUERIES.md` 29,382; falkor-chat process
docs 790,994 over 237 files; model-bench README+AGENTS 4,335; plan 122,252, coordination 88,997,
`-ml` 60,852); `DESIGN.md` 41 commits vs 318 to `plans/`+`reviews/`, last touched 2026-09-20
(`7a270329`), 58 falkor-chat commits since. `REVIEW.md` semantics in the decision log match the
Claude Code Review docs (root + subdirectory, review-only instructions).

### What's solid

Clear intent and user stories; FR-1/FR-3/FR-12 give a crisp division between live spec and
process record; the out-of-scope list and REVIEW.md size smells are well bounded; AC-5–AC-8 are
concretely checkable.

### Open questions (for the stakeholder)

1. Who authors the spec update at feature close — `teco` itself (overturning `teco.md`
   §Documentation curation) or the implementer/architect routed by `teco`? (B1)
2. Is a "feature" a requirements slug, a backlog item, or a milestone? (M1)
3. At a no-spec component's first close: whole spec, feature-only spec, or none? (M4)

## Pass 2

**Scope.** Revised `docs/requirements/live-specs.md` (157 lines), re-checked against Pass 1 and
the same baselines; backlog numbering re-measured across components.

**Verdict: needs changes** — one new major (P2-M1), which leaves the drift problem in place for
every component except falkor-chat. Everything else in Pass 1 is resolved.

CPG: not applicable — requirements/process document, no code-level component under review.

### Pass 1 dispositions

- B1 — fixed: FR-7 makes `teco` accountable, delegating the writing, with an explicit partial
  supersession of `teco.md:155` (live-spec content only); AC-9 extended; consistent with
  `teco.md:152`.
- M1 — fixed in substance (feature = backlog item closing on the BACKLOG→HISTORY move, archiving
  stays at milestone close, FR-2/AC-4), but see P2-M1 on "`K-`numbered".
- M2 — fixed: FR-13.
- M3 — fixed: AC-1 scoped to externally observable contracts plus verdict rule; FR-7 blocks close
  on "needs changes"; AC-4 requires the verdict. AC-1 wording still awaits stakeholder OK (P2-m1).
- M4 — fixed: FR-11 whole-component spec before first close, AC-10.
- m1 — fixed (line 24). m2 — fixed (129k w), new figure wrong (P2-m2). m3 — fixed (FR-10).
  m4 — fixed (FR-8 applier, AC-7 pruning, AC-6 incl. `cobb`).
- n1 — fixed. n3 — fixed (FR-1 and Related now cite).
- n2 — not fixed, correctly: the log is append-only (doc-size FR-5); later entries are clean.

Stakeholder answers folded in (question → answer): who writes the spec → `teco` ensures,
delegates (FR-7) · what is a feature → one backlog item, archive at milestone close (FR-2) ·
parallel plans → cite spec plus open features' plans (FR-13) · no-spec component → whole-component
spec before first close (FR-11).

### New findings

**P2-M1 (major) — "A feature is one `K-`numbered backlog item" (FR-2) excludes most of the
repo, including model-bench.** Only `falkor-chat/docs/BACKLOG.md` uses `K-` numbers (98 hits);
`model-bench/docs/BACKLOG.md` has 14 unnumbered bullet items and no `K-` item of its own,
`mcp-monitor/docs/BACKLOG.md` is unnumbered, root `docs/BACKLOG.md` (cpg) uses `C-`. And
requirements-driven work is often not a backlog item at all: `typed-decision-models` (the
in-flight model-bench feature) and this document both carry `Tracks: —`. Under FR-2 such work
never "closes", so FR-2/FR-7/FR-8 never fire and model-bench's new spec drifts exactly as
`DESIGN.md` did — the problem FR-10 pays to fix once. AC-4/AC-10 also become unreachable for
components without `K-` items. **Fix:** define a feature as "one backlog item, or one
requirements slug where no backlog item tracks it", closing when its delivery is recorded in
`HISTORY.md`; drop "`K-`numbered" (or say each component's own numbering). Stakeholder call,
since it amends the M1 answer — log it as a new entry naming the one it refines.

**P2-m1 (minor) — AC-1's pending stakeholder OK is not in Open questions.** The brief says AC-1's
contract-scope wording is `tico`'s proposal awaiting approval, yet Open questions reads "None"
and no log entry records it. Under root `AGENTS.md`'s growth-stops bullet an open stakeholder
question stays listed until answered; list it there (and log the answer) before the flip.

**P2-m2 (minor) — New figure understated.** Line 21: "`model-bench/docs/` (≈ 100k w)" —
`find model-bench/docs -name '*.md' | xargs cat | wc -w` gives 153,547.

**P2-m3 (minor) — Last decision-log entry is a disposition list.** Line 155–157 ("B1, M1–M4
resolved…; m1–m4, n1, n3 fixed; n2 not applied") is the finding→outcome record doc-size FR-4/
AC-5 put in the review (this pass carries it). Not removable now (append-only); do not add more
such entries.

**P2-n1 (nit)** — FR-13 has no AC; folding a clause into AC-8 ("and, where an open feature
changes that behavior, cites its plan") covers it. Out of scope line 87 ("catch-up of any other
component") reads against FR-11's whole-component spec at first close; "a one-time catch-up"
would remove the tension.

### Growth-stops check

Growth from Pass 1 is new scope (FR-13, AC-10 — new requirements from stakeholder answers) plus
appended log entries (allowed); the rework (FR-1, FR-2, FR-6–FR-11, AC-1/4/6/7/9, Problem) was
replaced in place. No restated history apart from P2-m3; Related and FR-1 now cite rather than
restate. Log entries 13–16 are one question → one answer with FR/AC; entry 13 names what it
overturns.

## Pass 3

**Scope.** Revised `docs/requirements/live-specs.md` (164 lines) against Pass 2.

**Verdict: approve with suggestions** — no blocker or major remains; the growth-stops check
passes, so the Ready-for-design flip is open to the stakeholder's confirmation.

CPG: not applicable — requirements/process document, no code-level component under review.

### Pass 2 dispositions

- P2-M1 — fixed: FR-2 now covers the component's own backlog numbering or a requirements document;
  close happens when delivery is recorded in `HISTORY.md`; AC-4/AC-9/AC-10 now say "feature"; log
  entry 18 refines entry 14.
- P2-m1 — fixed: AC-1 depth is confirmed and now stated in FR-3 and logged (entry 19); Open
  questions is empty, which is correct.
- P2-m2 — fixed (line 21, ≈154k w; measured 153,547).
- P2-m3 — accepted as is; no further outcome-list entries were added.
- P2-n1 — fixed (AC-8 covers FR-13; Out of scope line 89 now says "one-time catch-up").

Stakeholder answers folded in: what a feature is outside `K-` numbering → own numbering or a
requirements document (FR-2) · how deep the spec must cover the code → externally observable
contracts (FR-3, AC-1).

### New findings

**P3-m1 (minor) — Changes with neither a backlog item nor a requirements document never trigger
FR-2.** A direct fix that changes a contract and gets a `HISTORY.md` entry (common for follow-up
fixes) is not a "feature", so the spec is never updated for it. Suggest FR-2 also cover "any
other change recorded in `HISTORY.md` that alters a contract the spec states". This can be
settled in design, so it does not block the flip.

**P3-n1 (nit) — FR-3's contract list reads as falkor-chat's.** "tool/API surface, graph schema,
workflow semantics, configuration" does not obviously cover model-bench (CLI, pack format,
result/report schema, scoring rules). Mark it "e.g." in FR-3, and have AC-1 cite FR-3 rather than
repeat the list, so the fact has one home.

### Growth-stops check

The growth is new scope (FR-3's depth clause, entries 18–19, which are appended and each one
question → one answer with its FR/AC). FR-2, AC-4/8/9/10 and the Problem figure were reworked in
place. No restated history or facts beyond the P3-n1 duplicate.
