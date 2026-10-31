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

## Pass 4

**Scope.** Implementation plan `docs/plans/live-specs.md` (untracked, first version, read in
full) against `docs/requirements/live-specs.md` (Ready for design), the ledger
`docs/plans/live-specs-coordination.md`, and the repo: the write guards (`claude/scripts/
guard-doc-writes.sh`, `guard-broad-write.sh`, the `architect`/`analyst`/`cobb`/`coder` wrappers),
the eight reviewer prompts, an unfiltered AC-9 sweep, the falkor-chat spec set, and
`docs/plans/typed-decision-models-coordination.md`. Re-measured: drift baselines (`git log -1`:
`7a270329`/`a5d8e807`/`9ba0d788`) and code-commit counts since them (23/3/0), as §2 states.

**Verdict: needs changes** — three majors. The design (rule homes, `REVIEW.md` as a separate
file, per-document catch-up, the switch-over) is sound, and the fixes are done-condition and gate
edits, not a redesign.

CPG: not applicable — plan review of conventions, prompts and docs; no code-level change under
review (`cpg_falkorchat` is mid-rebuild and the plan assigns it to the AC-1 checks only).

### Major

**P4-M1 — AC-9 has uncovered hits, and the §5 sweep cannot prove AC-9.** An unfiltered
`grep -rn` (minus `.git`/`.venv`/`node_modules`) finds hits that no unit's done-condition owns:
- `claude/docs/manuals/agent-team.md:118` ("code + tests + doc updates") and `:190` ("Doc updates
  are written into the done condition") — the `tico`-owned team manual still says implementers
  update docs in each unit. Not in any unit.
- `falkor-chat/README.md:23` ("Full rationale … ops blueprint: `docs/DESIGN.md`") and `:237`
  ("full blueprint (… §14 M1 app architecture)") — `DESIGN.md` described as something other
  than the live spec, and stale (§14 moved to `SERVER.md`). The root "Component docs" table
  lists this README as the entry doc. Not in any unit.
- `falkor-chat/docs/SERVER.md:480` — "update `AGENTS.md` … in the same change, per the repo's
  documentation rule", inside the spec itself. L5b's history list does not name it, and the AC-3
  regex does not match it.
- `falkor-chat/docs/DESIGN.md:1` — H1 "Design Document & Blueprint". Not in L5a's list.

The §5 command also fails both ways. Its pathspec `claude/*/*.md` recurses (git pathspec `*`
crosses `/`), so it returns hundreds of kaizen and process-doc hits for the L3d gate to "rule".
Its regex misses `teco.md:138` "Documentation is part of done", `:3` "doc updates are part of
every unit's done-condition" and `:92` "fold doc updates into the done-conditions". It never
reaches READMEs or manuals. **Fix:** (1) add the manual to a `tico` unit (it can run any time
after L3d), the README lines to L2b, and `SERVER.md:480` and the H1 to L5b/L5a. (2) Replace §5's
AC-9 check with a sweep over prompts, skills, context files, READMEs and manuals, using the
`same change|alongside|part of done|doc updates|blueprint|DESIGN\.md` phrase set and excluding
`kaizen/` and the process kinds, plus a table that maps each hit to its unit. The L3d gate re-runs
it.

**P4-M2 — `analyst` gates its own permissions and prompt, and recurring `REVIEW.md` changes
have no independent check** (premise (b)). Premise (b) holds: `guard-review-doc-writes.sh`
allows only `docs/reviews/*|*/docs/reviews/*`, so a root `REVIEW.md` write would escalate, and
the widening is needed if `analyst` is the author. But three gaps follow:
- L3a widens `analyst`'s own write guard and L3b edits `analyst.md`, and `analyst` gates both.
  The beneficiary of a permission and the subject of a prompt is not an independent gate.
  (`cobb`'s allowlist does not cover `claude/*/hooks/*.sh`, so L3a also brings a human prompt. That
  is a checkpoint, but it is not the gate, and §6 "Guard prompts" does not list it.)
- At each close (§3.4 step 4), `analyst` applies lines that are mostly proposed in its own
  reviews. `cobb` gates only the L2a seed. `teco`'s line check compares additions with proposals,
  but **pruned lines have no proposal to compare against**. So a *Do not report* addition or a
  deletion narrows `analyst`'s own review bar with no one else's judgment.
- The seed gate (`cobb` on L2a) is fine.

**Fix:** gate L3a and the `analyst.md` part of L3b by someone other than `analyst`
(`security-expert` advisory plus `cobb`'s own lint, or the human at the L3a prompt; say which).
In step 4, require prunes to be proposed in a review too, each citing the test, CI job or hook
that now enforces the rule. `teco` line-checks the `REVIEW.md` diff in both directions against
the proposals and confirms that each cited enforcer exists. A *Do not report* addition or a prune
goes to `cobb`, as the seed does.

**P4-M3 — The AC-1 checks are per document, but AC-1 is judged over the spec set, and the
parallel writers have no contract-to-document map.** AC-1 asks whether "falkor-chat's spec set"
omits a contract. L5a–c run in parallel, each with its own check "against shipped code". The
§3.5 inventory lists surfaces but does not say which document owns each one. Several surfaces
plausibly belong to two documents: model resolution (`DESIGN.md` §1.3 and `SERVER.md` §1.8),
K-079's Layer 1 contract (`QUERIES.md` or `SERVER.md`), workflow semantics (`DESIGN.md` §6 and
`executor.py` in `SERVER.md`). Two failures follow. Both `graph-dba` writers absorb the same
drift into two homes, which breaks "a fact has one home". Or a per-document checker reports as
omitted a contract that its sibling holds or is about to hold. **Fix:** add a "home document"
column to the falkor-chat inventory in §3.5. Scope each L5 check to its document's surfaces plus
contradictions. After L5a–c are gated, add one set-level AC-1 pass over the three documents
together, which can be L5d's precondition.

### Minor

- **P4-m1 — The AC-6 command can never pass.** I ran it verbatim. `git grep -L 'REVIEW.md' --
  claude/{…}/*.md` also lists eight on-demand knowledge files
  (`analyst/review-techniques.md`, `cobb/TESTING.md`,
  `data-scientist/{lm-studio-model-notes,statistical-method-techniques}.md`,
  `graph-dba/falkordb-{quirks,reference}.md`, `qa-engineer/qa-testing-techniques.md`,
  `teco/coordination-techniques.md`), and those should never carry the pointer. Use
  `claude/<agent>/<agent>.md` for each of the eight.
- **P4-m2 — History left in the spec that AC-3 does not catch, and inbound citations that break.**
  `DESIGN.md` §14 (`:792–814`) is a "moved to `SERVER.md`" redirect table, which is history with
  no date or version, so the AC-3 regex misses it. Removing it, or §12/§13 or `SERVER.md` §1.6,
  breaks live citations at `falkor-chat/server/falkorchat/services.py:3258`, at
  `skills/agent-standards/kiro.md:71` (both cite `DESIGN.md` §14.x/§15.x) and at
  `falkor-chat/docs/BACKLOG.md:623,636`. Have L5a delete §14 and move its mapping into the
  falkor-chat `HISTORY.md` catch-up entry (L5d). Add an inbound-citation fix list to L5a/L5b,
  routed to `coder`, `cobb` and `teco` by file. Archived documents keep their stale citations,
  as frozen documents do.
- **P4-m3 — The switch-over is transitional text in an always-loaded file, and Q1 affects more
  than L6.** §3.2 has root `AGENTS.md` "state the switch-over". Once L3d lands, that sentence is
  history, which the context-file convention excludes. Keep it in the ledger and in the
  `HISTORY.md`/kaizen delivery records. Also, "Blocks L6 only" is not quite true. Under the
  per-component reading below, L2b's close bullet and L3d's `teco.md` close step must not bind a
  feature that is in flight while its component has no spec. So answer Q1 before L2b is
  dispatched.
- **P4-m4 — §3.4 step 3 has no review home for a slug-less close.** FR-2 counts "any other change
  recorded in `HISTORY.md` that alters a contract". Such a change has no `<slug>`, so there is no
  `reviews/<slug>-impl.md` for the check to go in. Name a fallback (for example
  `reviews/<history-entry-topic>-impl.md`, or the fix's RCA document).

### Nits

- **P4-n1 — Premise (a) is true, but the stated reason is weaker than it looks.**
  `guard-plan-doc-writes.sh` "escalates" by returning `ask`, not a deny, for any Write/Edit
  outside `docs/plans/` or `/tmp`. The plan accepts that same per-file human prompt for `cobb` in
  L2b. The writer choices stand on the "Why" column (domain and code reading). Cite that column as
  the reason, not the guard.
- **P4-n2 — The `REVIEW.md` → `analyst` row is in the wrong table.** The root by-kind owner table
  routes `archived` flips, and `REVIEW.md` has no `Status:`. State `REVIEW.md` ownership in the
  feature-close bullet or in each file's opening line.
- **P4-n3 — §1 carries finding dispositions** ("P3-m1 is already folded in … P3-n1's wording …").
  Under the growth-stops bullet a plan states the current design and cites findings by ID at
  most. Drop the clause; the requirements already show the result.
- **P4-n4 — Inventory gaps.** model-bench: add `runner.py` (the run lifecycle `DESIGN.md`
  covers) and `hostinfo.py`. falkor-chat: add the auth/tenancy seam and the `/shop` static mount
  (`app.py`), which are contract surfaces `SERVER.md` states today.

### Premises, sizes, sequencing

- **(a)** True as stated; see P4-n1. **(b)** The guard exists and is needed; there is a self-gating
  problem, which P4-M2 covers.
- **Root `AGENTS.md` +150 words.** Root `AGENTS.md` is 3,198 words against the ~2,500 smell. The
  two bullets are new scope, which the context-file convention allows, and moving the
  review-only sentences out offsets part of it. The cap is proportionate. Keep delegation details
  in `teco.md`, as §3.1 already does.
- **`REVIEW.md` seeds** (≤ 600 root, ≤ 400 per component) leave FR-9 headroom (1,000/800).
  Seed rule 1 moves text rather than copying it, which is correct.
- **Ordering.** L3a → L2a → L2b → {L3b ∥ L3c} → L3d is sound. L3b and L3c touch disjoint prompt
  files and kaizen folders. Within the chain, each shared file has one writer: root `AGENTS.md`
  is L2b's and `claude/README.md`/`teco.md` are L3d's. Only concurrent outside sessions need the
  ledger check, as §4 says. L5a–c are file-disjoint from the convention chain and from each
  other. The only coupling between them is P4-M3's shared contract surfaces, which the home-document
  map removes. L5 against typed-decision-models U9–U12 (`SERVER.md`, `DESIGN.md`) is
  serialized, as §6 says. The pinned sha is correct.
- **Growth-stops check.** This is a first version, so there is no rework growth to judge. One
  disposition clause (P4-n3). No restated history otherwise.

### FR/AC coverage

Every FR and AC has an owning unit: FR-1/3/4 via L2b and L6, FR-2/7/8/11–13 via L2b and L3d,
FR-5/9 via L2a, FR-6 via L3b–L3d, FR-10 via L5a–c and L6. AC-1 to AC-10 map as §5 says. The
exceptions are AC-9 (P4-M1) and AC-6's check (P4-m1). The AC-6 list is complete: the eight
prompts cover everyone who writes a review or gate verdict. `architect`'s "rulings" in
`docs/plans/typed-decision-models-coordination.md` are design decisions, not verdicts, and
`coder`, `tdd-engineer`, `frontend-engineer` and `devops` write neither.

### Q1 — model-bench spec timing (independent answer)

**Is there a conflict?** Yes, but the requirements do not create it. The plan's team-wide L3d
switch-over does. typed-decision-models has U2–U7 and U9–U12 still open, so it will close after
L3d. Under §3.4 it then owes model-bench's spec before its closing entry (FR-2's "not before the
spec states the behavior it shipped", and FR-11's no-spec clause). The stakeholder's decision is
that the spec comes after the chain closes, and that collides with §3.4. The requirements
themselves do not force the collision. AC-4 and AC-10 bind features that close "after delivery",
and live-specs cannot be delivered before FR-10's model-bench spec (L6) exists. FR-11 already
makes adoption per component.

**My resolution: honor the decision literally and make the switch-over per component.**
- falkor-chat adopts the close sequence at L3d. Its catch-up is serialized ahead of any close that
  touches a document still in L5.
- model-bench adopts at L6.
- Any other component adopts at its first close after L3d, through FR-11.
- typed-decision-models closes under its own briefs: units dispatched before the switch-over
  finish under their briefs, which §3.4 already says for units, extended here to the feature.
  Then L6 writes the whole-component spec from shipped code, which includes typed-decision-models'
  behavior, and its AC-1 check covers that behavior.

The cost matches the plan's recommendation: the spec is written once, after the code settles.
This option also needs no amendment and does not reinterpret the stakeholder's words, and
typed-decision-models' close does not wait on another chain's deliverable. The carve-out lives in
the ledger and in the typed-decision-models close brief, not in `AGENTS.md` (P4-m3). One condition
applies: if another model-bench feature reaches its close before L6, FR-11 applies to it, and L6
must land first.

The plan's recommendation (L6 inside typed-decision-models' close) is also consistent with the
requirements. It reads "after the chain closes" as "after its build units". If the stakeholder
meant to avoid writing the spec against churning code, both options satisfy that. Choose by
whether typed-decision-models' closing `HISTORY.md` entry may wait on L6.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*.
- "When a change widens an agent's permissions or edits its prompt, that agent is not its gate."
- "A 'no X remains' acceptance check is an unfiltered sweep, and every hit is mapped to an owning
  unit. A filtered or path-limited sweep is a bound, not proof. Run the plan's check command
  verbatim: a check that cannot fail, or cannot pass, is a finding."

### What's solid

- The rule-homes table (§3.1) keeps always-loaded growth bounded.
- Each catch-up has a git-based writer method and an inventory-based checker method that are
  independent of each other.
- The writer choices are grounded in the actual guards; `coder`'s deny-list does allow the spec
  paths.
- The drift baselines and counts are accurate.
- The switch-over rule for in-flight units is right.
- The AC-3 rule on open `K-` references follows FR-10 exactly.

### Open questions

1. Q1, above: either reading is consistent with the requirements; the stakeholder chooses.
2. Who gates L3a and `analyst.md`'s part of L3b (P4-M2): `security-expert` advisory with `cobb`'s
   own lint, or the human?

## Pass 5

**Scope.** `docs/plans/live-specs.md` v1.1 (2,992 words, read in full) against Pass 4 and the
stakeholder answers of 2026-10-31. Every overrule was re-checked against the tree, and so were
the fix interactions. The §5 AC-9 sweep and the AC-6 command were run verbatim.

**Verdict: approve with suggestions.** No blocker or major remains. Two minors and one nit are
below; none blocks dispatch. Fix P5-m1 and P5-m2 before L2b and L3c are dispatched,
respectively.

CPG: not applicable — plan review of conventions, prompts and docs; no code-level change under
review.

### Pass 4 dispositions

- P4-M1 — fixed. The §5 sweep, run verbatim, gives 107 hits. Every hit that needs an edit maps to
  L2c, L3c, L5a–c or L7. I read the hits outside the table: catalog "same change" rules, and
  "alongside" or `DESIGN.md` citations in `coder.md`, `teco.md:53/163/176`, `claude/README.md:8/14/17`,
  `claude/AGENTS.md`, `tico.md:44`, manuals and skills. All are correctly no-change. A sweep of the
  `opencode/` and `kiro/` prompts outside its paths finds no prompt hits, and no `.kiro/steering`
  exists.
- P4-M2 — superseded by the stakeholder answer. The guard is not widened. `analyst` drafts,
  `teco` applies and `cobb` gates (§3.3, L2a/L2b), and `security-expert` gates `analyst.md`
  (L3a). Prunes now cite their enforcer, and the line check runs in both directions (§3.4
  step 4). The residue is in P5-m1 and P5-m2.
- P4-M3 — fixed. There is a home-document table, scoped L5 checks and a set-level L5s before L5d.
  Every module the table names exists (`modelconfig.py`, `querygen.py`, `storefront_api.py`,
  `fingerprint.py`, `hostinfo.py`, …). The table's coverage gap is P5-n1.
- P4-m1 — fixed. The new loop prints the 8 prompt files today, as expected before L3, and no
  knowledge files.
- P4-m2 — **overrule confirmed; my fix was wrong.** The coordinator's grep returns 19 hits in
  `falkor-chat` outside `docs/`. 18 are source or test docstrings citing `DESIGN §13/§14.x/§15`
  (`config.py`, `api.py`, `db.py`, `services.py`, `repository.py` ×3, `tools.py` ×2, `app.py`,
  `web/app.js`, `guards.py` ×2, `bootstrap_schema.sh`, `test_guards.py`, `test_embedding.py`, …).
  Pass 4 cited only `services.py:3258`, and deleting §14 would have cost about 18 code edits.
  A present-tense §14 map plus frozen numbering resolves all of them at no code cost and is not
  history.
- P4-m3 — fixed. The switch-over and the typed-decision-models arrangement live in the ledger
  (§3.1, last row, and §3.4), and Q1 is answered.
- P4-m4 — fixed (§3.4 step 3 fallback).
- P4-n1 — fixed. Writers are "chosen by domain and code reading".
- P4-n2 — fixed differently. There is no owner row; `REVIEW.md` upkeep sits in the feature-close
  bullet (§3.2), which is a better home than the row I suggested.
- P4-n3 — fixed. Only the one-line revision note remains, which is allowed.
- P4-n4 — fixed (`runner.py`, `hostinfo.py`, auth seam, `/shop`).

Stakeholder answers folded in: Q1 → L6 runs inside typed-decision-models' close, before its
closing entry (§3.6 matches, ledger-held) · who gates `REVIEW.md` and `analyst.md` → `analyst`
drafts, `teco` applies, `cobb` gates, `security-expert` gates `analyst.md` (§3.3, L3a match).

### Fix interactions checked

- **Frozen numbering against AC-3's widened regex.** The kept §14 map does not collide with the
  regex. It is restated in present tense, so the only `moved (out )?to` hits today are the
  `DESIGN.md:16–17` revision note (removed anyway) and `QUERIES.md:1771` ("moved to a DIFFERENT
  waiting step", a query comment). The latter is a guaranteed hit that must be ruled
  not-history. The "or each ruled" clause covers it, so this is no finding. The `DESIGN.md` §1.3
  citations (6 live, all about the model stack and dimension) still resolve after model
  resolution moves to `SERVER.md`, because §1.3 keeps the stack.
- **`teco` applies, `cobb` gates, and §1's rule.** These collide; see P5-m1.
- **`teco` writing `REVIEW.md` against `teco.md`.** These collide; see P5-m2.
- **`security-expert` as the L3a gate.** `security-expert.md:50` says "No standing gate, ever",
  but every invocation is explicit, and a one-unit dispatch by `teco` is one. No conflict.

### New findings

**P5-m1 (minor) — §1's rule, as worded, is broken by §3.4 step 4 for additions.** §1 says "no
agent gates a change to … text it wrote or applied." At close, a non-narrowing `REVIEW.md`
addition is applied by `teco` and then checked only by `teco`'s own two-way line check. Under §1's
wording that is self-gating. Narrowing changes are correctly routed to `cobb`, or to `analyst`
when `cobb` proposed them. The practical risk is low, since an addition only adds checks. But a
rule that the plan's own close step breaks will be cited against it. **Fix (choose one, in §1 and
§3.4 step 4):** (a) say that the line check is an integrity check, not a gate, and that additions
are ungated by design (FR-8 asks only that `teco` ensure they are applied); or (b) have `cobb` gate
every close-time `REVIEW.md` diff. (a) is the proportionate one.

**P5-m2 (minor) — `teco.md:152` still forbids what L2b and §3.4 step 4 assign to `teco`.**
`teco.md:152` says "you curate; you don't write the docs yourself". L3c's done-condition carves
out l.3, l.92, l.138 and l.155, but not l.152. After L3c, `teco`'s prompt would tell it both to
apply `REVIEW.md` lines verbatim and never to write docs. **Fix:** add l.152 to L3c's list: verbatim
application of proposed `REVIEW.md` lines is curation, not authorship. L2b runs before L3c, so
its brief needs to state the exception.

**P5-n1 (nit) — The home-document table has no catch-all.** Contract-bearing modules the table
does not name include `guards.py` (workflow guard semantics, `DESIGN.md` §6.1), the retrieval
read path (§8; the search tools), `extraction.py`/`fusion.py`/`chunking.py`, `llm.py` and
`pricing.py`. The checker "enumerate[s] each surface from the code", so L5s will find them, but no
rule says which document they go in. Add one sentence: L5s assigns a home to any surface that
enumeration finds outside the table and reports it.

### Growth-stops check

v1.1 shrank (3,179 → 2,992 words) while adding L2c, L5s and L7, so the rework replaced text in
place. The revision note is one dated line. §5's "including every one Pass 4 lists" cites the
review rather than restating it. No finding→resolution table, and no restated history or facts.
Pass.

## Pass 6

**Scope.** Unit L2a: the seed content for the three `REVIEW.md` files, drafted per
`docs/plans/live-specs.md` v1.2 §3.3, for `teco` to apply verbatim (L2b) and `cobb` to gate. Each
fenced block below is the whole file as it should land. Read: root `AGENTS.md`, the eight reviewer
prompts, a keyword search of every review last committed since 2026-09-01 (117 documents), and the
finding titles of the falkor-chat and model-bench ones. Each cited finding was read through its
stated claim. No verdict: this pass drafts and gates nothing.

CPG: not applicable — drafting review-instruction text from documents; no code-level change.

### Proposed REVIEW.md lines — `REVIEW.md`

```markdown
# Review standing instructions

## What counts as serious

- In a spec-update check, a contract the shipped diff changed that the live spec omits or
  contradicts is a blocker: the feature cannot close while the verdict is "needs changes".

## Always check

- A revision of a not-yet-`archived` requirements, plan or review document, against root
  `AGENTS.md`'s growth-stops bullet. For any growth, say whether it is new scope or rework, and
  report rework growth, restated history and restated facts as findings. There is no numeric
  threshold.
- A plan for a component with a live spec states only the change against it, citing the affected
  spec sections. Report restated spec content as a finding. Where the plan depends on behavior
  that an open (not yet closed) feature changes, it cites that feature's plan; a missing citation
  is a finding.
- A spec update is checked against the shipped code (the diff from the feature's baseline sha),
  not against the plan. It also states no change history and keeps its section numbering (root
  `AGENTS.md`, live-spec bullet).
- Every check the work relies on (a test, a pin, a guard, a gate's acceptance command, a live
  probe) can both fail and pass. Name the mutant, planted input or positive control that turns it
  red. Its expected values are never read from the constant or function it checks. A check that
  cannot fail, or cannot pass, is a finding.
- A "no X remains" check is an unfiltered sweep, and every hit is mapped to an owner: the unit
  that edits it, or an explicit no-change ruling. A filtered or path-limited sweep, or a pattern
  that matches only the already-known sites, is a bound, not proof. Run a prescribed check
  command verbatim, from the working directory it states.
- Whether a finding should become a standing rule. If it should, propose it in your review as a
  `### Proposed REVIEW.md lines` block naming the target file and heading. Each line is an
  addition or a prune, and the review's prose states its source: the finding ID or, for a
  recurring class, the review paths and finding IDs you read. Propose a line only if it changes
  what a reviewer does. A rule that a test, CI job or hook already enforces is not added, and a
  prune names the test, CI job or hook that now enforces it. With an addition, also propose
  pruning any line in that file that is now enforced elsewhere.

## Do not report

- In a spec-update check, a spec that departs from the plan where it matches the shipped code.
  The plan is not the baseline.

## Evidence bar

- State whether each finding was verified (run, traced end to end, checked against docs) or
  inferred. A fix you suggest is a claim: run it, and check its reach (callers, inbound
  citations) before you propose it.
```

**Sources, by heading and position** (category numbers are §3.3's: 1 moved from root
`AGENTS.md`, 2 duty this feature creates, 3 class recurring in two or more reviews since
2026-09-01; every review cited below was read at the finding named):

- *Serious* 1 — category 2: FR-7 and AC-1 (an omission or contradiction blocks the close).
- *Always check* 1 — category 1: the growth-stops bullet's review-only sentences, root
  `AGENTS.md` l.182–184 ("Every review gate checks a revision against this bullet …"), moved by
  L2c, not copied. Recurrence also observed: `docs/reviews/typed-decision-models2.md` m9,
  `typed-decision-models3.md` m8, `falkor-chat/docs/reviews/salesperson-cart-product-id.md` F7, P2-1.
- *Always check* 2 — category 2: FR-12 and FR-13.
- *Always check* 3 — category 2: FR-7 (shipped code, not plan; baseline sha per §3.4 step 3);
  the no-history and frozen-numbering clause cites the §3.2 live-spec bullet rather than restating it.
- *Always check* 4 — category 3. Checks that cannot fail or cannot pass:
  `falkor-chat/docs/reviews/salesperson-query-graph-data.md` m1, N1;
  `falkor-chat/docs/reviews/extract-transient-retry.md` M-3, N-4, P3-2;
  `falkor-chat/docs/reviews/salesperson-single-model.md` M3, B1;
  `docs/reviews/typed-decision-models3.md` m1, m7; `docs/reviews/typed-decision-models.md` m3, m4;
  `docs/reviews/typed-decision-models-impl.md` P2-m1, P6-m1. Self-confirming expected values:
  `falkor-chat/docs/reviews/salesperson-cart-product-id-impl.md` F1,
  `model-bench/docs/reviews/s4-gate-followups.md` F3, `typed-decision-models.md` N3.
- *Always check* 5 — category 3, and Pass 4's second proposed line. `docs/reviews/live-specs.md`
  P4-M1; `docs/reviews/typed-decision-models.md` N4 (misses sites by construction) and m7;
  `docs/reviews/salesperson-ui.md` M9 (command cannot pass verbatim; root-anchored glob);
  `falkor-chat/docs/reviews/salesperson-query-graph-data.md` N1 (wrong working directory, pathspec
  matches nothing).
- *Always check* 6 — category 2: the proposal format (FR-8, FR-9, AC-7, §3.3, §3.4 step 4,
  including that a prune cites its enforcer).
- *Do not report* 1 — category 2: FR-7's "against the shipped code (not the plan)", stated as its
  narrowing half. It is a *Do not report* line, so it is narrowing; `cobb`'s L2b gate covers it.
- *Evidence bar* 1 — category 3. A reviewer's own suggestion that proved wrong because it was
  not checked: `docs/reviews/live-specs.md` P4-m2 (Pass 5 disposition: the fix missed about 18
  inbound code citations), `falkor-chat/docs/reviews/salesperson-single-model.md` P3-3 ("My
  Pass 1 nit … was wrong": `/mcp` is not mounted in storefront mode).

**Not seeded: Pass 4's first line** ("that agent is not its gate"). It fails the category-3 bar.
In reviews since 2026-09-01 the class appears only in this review (P4-M2, P5-m1). The older
precedents, `docs/reviews/analyst-inbox-distillation.md` l.9 and
`docs/reviews/graph-dba-kaizen-distillation.md` l.108, both date from 2026-08-19, outside the
window. It is also not a duty FR-1–FR-13 create; it is this plan's own §1 rule. A later review
that hits it again can propose it.

**Not moved: collision rule 5's "A later `## Pass N` is compact by rule …"** (root `AGENTS.md`
l.236–239). It is review-authoring practice, but §3.2 moves only the growth-stops sentences, and
widening L2c's scope is the architect's call, not a seed's.

### Proposed REVIEW.md lines — `falkor-chat/REVIEW.md`

```markdown
# falkor-chat — review standing instructions

## What counts as serious

- Text the model reads (a system prompt, a tool description, a tool result, a hold or error
  message) is behavior. A change to it that invites a fabricated value, a bypass of a hold, or a
  wrong reply language is at least major unless a measurement covers it.

## Always check

- A change to a published workflow def's create-only fields (`config.tools`, `systemPrompt`,
  `config.model`) cannot be published under an old or uncommitted version name by any step in
  the order the plan runs them: a seed, `start_demo.sh`, or a re-seed after a default `pytest`
  wipes `reference`. Per `falkor-chat/AGENTS.md`, "Three traps when bumping `SALESPERSON_DEF`".
- A measured comparison (an eval, a live run, a re-run against a baseline) pins its baseline to
  an explicit commit sha, and no step of the run overwrites the file it compares against.

## Evidence bar

- A reported suite count says whether the tests that wipe `reference` ran (`wf_repo` tests,
  `./scripts/test_queries.sh`; `falkor-chat/AGENTS.md`, "Key scripts"). When shared state must
  survive, run the review-safe subset (`falkor-chat/AGENTS.md`, "Probing shared graph state
  without mutating it") and name what it left unrun. A run that wipes `reference` is reported
  with its re-seed sequence.
```

**Sources** (all category 3, from falkor-chat's own corpus since 2026-09-01; the two `AGENTS.md`
citations point at the hard rules rather than restating them, per §3.3):

- *Serious* 1 — `falkor-chat/docs/reviews/salesperson-cart-product-id.md` F1 (major, the hold
  message teaches a re-fire) and F6 (a miss result invites a fabricated price);
  `salesperson-cart-product-id2.md` M1 (major, the in-line language cue removed);
  `salesperson-fabricated-prices-impl.md` n2 (an unmeasured tool-description change).
- *Always check* 1 — `salesperson-cart-product-id.md` F2 (major, a seed onto a wiped `reference`
  stores `v9` content under `v8`) and `salesperson-cart-product-id-impl.md` F2 (the publish reads
  the working tree, not the commit).
- *Always check* 2 — `salesperson-single-model.md` M1 (Q2 overwrites the tracked file that is its
  own baseline); `salesperson-query-graph-data.md` m2 (baseline not pinned to a sha) and m1 (the
  "tracked baseline not overwritten" check is vacuous).
- *Evidence bar* 1 — `salesperson-cart-product-id-impl.md` F5 (the reported counts cannot show
  whether the five `wf_repo` tests ran). The same handling recurs as reviewer notes, not finding
  IDs: `falkor-chat/docs/reviews/agent-team-ingestion-produced-by.md` l.116,
  `falkor-chat/docs/reviews/document-ingestion2-impl.md` l.843 and l.1238,
  `docs/reviews/salesperson-ui-impl.md` l.5680, l.5968, l.6298, l.6580.

Not seeded: `falkor-chat/AGENTS.md`'s "Rules for future work" 1–8. I did not observe a recurring
review finding against any of them in the window, and rule 5 (suite green before a schema or
query commit) is already a gate condition rather than a reviewer instruction.

### Proposed REVIEW.md lines — `model-bench/REVIEW.md`

```markdown
# model-bench — review standing instructions

## What counts as serious

- A bar, threshold or verdict rule that a degenerate candidate can meet (one that always answers
  the majority class, or never hands off), or that no candidate can meet under the planned
  procedure, is at least major.

## Always check

- Bar and verdict tests probe both sides of every boundary, including the equality case, so a
  `<`/`<=` mutation turns one red.
- Every selection among runs, candidates or settings (last-wins, a re-run, a re-tune, a re-fit
  on the same pack version) is recorded and counted against the evaluation split. An
  unrecorded selection on evaluation data is a finding.
- A claim about LM Studio load state (one load per run, which model is resident, a reload between
  replicates) is checked against the code path and procedure that load the model, including JIT
  loads triggered by a call.
- A change to what a stored result, hash or fingerprint means has canonical vectors whose
  expected values were written independently of the function under test, and covers every rule
  its docstring claims (key order, non-ASCII, which fields enter the hash).
```

**Sources** (all category 3, from model-bench's corpus since 2026-09-01; typed-decision-models
spans root-level `docs/reviews/typed-decision-models*.md`, its family's home under FR-1's
out-of-scope re-homing):

- *Serious* 1 — `docs/reviews/typed-decision-models.md` M5 (major, "≥90% agreement" met by a
  degenerate model) and P5-M2 (major, fitting margins let a point close without its bar measured);
  `docs/reviews/typed-decision-models3.md` B1 (blocker, no bar can be met).
- *Always check* 1 — `docs/reviews/typed-decision-models.md` m4 (bar tests probe only the failing
  side) and `docs/reviews/typed-decision-models-impl.md` P2-m1 (a pin that uses the default value
  cannot catch it being hardcoded). This is the model-bench form of root *Always check* 4; keep it
  only if `cobb` judges the boundary case worth naming separately.
- *Always check* 2 — `docs/reviews/typed-decision-models.md` M3 (twice: requirements and plan
  passes) and N1 (major, the reuse count misses a re-tune that keeps its `candidateId`);
  `docs/reviews/typed-decision-models2.md` M1 (major, last-wins is an unrecorded selection).
- *Always check* 3 — `docs/reviews/typed-decision-models2.md` m2 ("one load per run" does not hold
  under JIT), `docs/reviews/typed-decision-models3.md` B1 (an explicit reload makes every
  replicate 2 `same-load`) and `docs/reviews/small-model-benchmarking-impl.md` P12-5 (`warm_up`'s
  residency reasoning inverted).
- *Always check* 4 — `docs/reviews/typed-decision-models.md` N3 (the canonical vectors confirm
  themselves) and m3 (an inert widen mutation on `input_hash`'s field table);
  `docs/reviews/typed-decision-models-impl.md` P6-m1 (two of `question_spec_hash`'s stated rules
  unpinned).

The model-bench corpus is concentrated in one feature family. Four of the five lines draw only on
typed-decision-models reviews; `typed-decision-models.md`, `-2`, `-3` and `-impl` are separate
review documents, and that is how I counted "two or more". The `small-model-benchmarking*`
reviews (2026-09-17, in the window) were searched by finding title for these classes, not read
whole; a later proposal may add lines from them. Not seeded: a line citing `model-bench/AGENTS.md`'s
"Load-bearing invariants". L6 shrinks that section to spec pointers, so a citation of it would
go stale within the feature.

### Duplicate list (for L3a–L3c)

| Seed line | Prompt line it duplicates | Overlap |
|---|---|---|
| Root *Always check* 1 (growth-stops) | `claude/analyst/analyst.md` l.66, "Gating a revision of a document under root `AGENTS.md`'s growth-stops bullet: run its check, say whether any growth is new scope or rework …" | Full for the check; the prompt's `## Pass N` and Ready-for-design clauses stay. |
| Root *Always check* 4 (a check can fail) | `claude/analyst/analyst.md` l.66, "never into the step it is meant to catch … it can never redden" | Partial: the prompt line is about placing a suggested guard; keep it, point the general clause at `REVIEW.md`. |
| Root *Always check* 5 (sweeps) | `claude/analyst/analyst.md` l.88, "A plan's prescribed acceptance-check command is a claim too — run it verbatim before approving"; l.85, "A `git grep`/`git ls-files` count is a bound, not a fact …" | Full for l.88's first sentence (its placeholder example can stay); partial for l.85, which is about untracked artifacts. |
| Root *Evidence bar* 1 | `claude/analyst/analyst.md` l.84 ("Distinguish what you verified … from what you infer") and l.86 ("A fix you suggest is a claim, not a nit — run it"); `claude/qa-engineer/qa-engineer.md` l.72 and `claude/security-expert/security-expert.md` l.89 ("Evidence over …") | Full for analyst l.84's first clause and l.86's first sentence; partial for the qa-engineer and security-expert lines, which are lens-specific and should stay. |

No duplicates found in `data-scientist.md`, `graph-dba.md`, `cobb.md`, `tico.md` or `teco.md`
(searched for each seed's key terms). `teco.md` l.110 ("mutation-test the green-on-arrival
tests") is an implementer-brief instruction, not a reviewer one, so it is not on the list.

### Sizes (`wc -w` on each fenced block's content)

| Block | Words | Target (§3.3) | FR-9 smell | Lines |
|---|---|---|---|---|
| `REVIEW.md` | 479 | ≤ 600 | ~1,000 | 9 (1 moved, 5 duty, 3 recurring) |
| `falkor-chat/REVIEW.md` | 205 | ≤ 400 | ~800 | 4 (all recurring) |
| `model-bench/REVIEW.md` | 191 | ≤ 400 | ~800 | 5 (all recurring) |

Counts include each block's H1 and headings. No seed line restates a rule a test, CI job or hook
enforces: each is a reviewer judgment about the artifact (whether a check can fail, what a plan
restates), which a hook or suite cannot make. I did not audit every hook to confirm this.

## Pass 7

**Scope.** Unit L2b gate (`cobb`): the three uncommitted files `REVIEW.md`,
`falkor-chat/REVIEW.md` and `model-bench/REVIEW.md` that `teco` applied from Pass 6. Read: plan
§1, §3.3, §3.4, §4; requirements FR-5–FR-9, FR-12, FR-13, AC-5; Pass 6; root and
`falkor-chat/AGENTS.md`; the eight reviewer prompts; every tracked hook, `claude/scripts/*.sh`,
`.claude/settings.json`; 14 cited findings across 9 reviews.

CPG: not applicable — the artifact is review-instruction text, no code change.

**Verdict: approve with suggestions.** All three files are byte-identical to Pass 6's fenced
blocks (`diff` against each extracted block: no output). `wc -w` gives 479 / 205 / 191, within
FR-9's smells and §3.3's targets (AC-5 met). Every line sits under one of FR-5's four headings.
Each line is something a reviewer does or reports, and none is exhortation. P7-2 should land
before L2c starts. The other findings can land at any point before L3a.

### Checks run

- **Verbatim apply** — verified. Each `markdown` fence was extracted from Pass 6 and diffed
  against its file: identical.
- **Already enforced elsewhere (FR-9)** — verified, with one partial exception (P7-3). No
  `settings.json` declares a hook (`.claude/settings.json` has permissions only). Every tracked
  hook checks a path or an operation: `guard-*-writes.sh`, `guard-*-bash.sh`,
  `guard-destructive-ops.sh`, `guard-agent-dispatch.sh`, `guard-exploitation-approval.sh`,
  `ensure-environment.sh`. None reads document content. No CI exists: no `.github/`, no
  pre-commit config, no Makefile. `claude/scripts/audit-team.sh` checks prompts, not reviews.
  No seed line is enforced. The only partial enforcer is in falkor-chat (P7-3).
- **Recurrence spot-check** — verified at the cited line, 14 findings across 9 reviews:
  `salesperson-query-graph-data.md` m1 (l.152), m2 (l.160), N1 (l.426);
  `salesperson-single-model.md` M1 (l.58), P3-3 (l.377);
  `salesperson-cart-product-id-impl.md` F1 (l.45), F5 (l.110); `typed-decision-models.md` m4
  (l.441), N3 (l.615), m3 (l.434), M5 (l.239); `typed-decision-models2.md` m2 (l.200);
  `typed-decision-models3.md` B1 (l.110); `model-bench/docs/reviews/s4-gate-followups.md` F3
  (l.122). Each says what Pass 6 says it says. All nine reviews were last committed between
  2026-10-15 and 2026-10-31, inside the 60-day window (from 2026-09-01). One citation-precision
  issue is P7-6.
- **Citations into `AGENTS.md`** — verified today. `falkor-chat/AGENTS.md` "Key scripts" (l.67),
  "Three traps when bumping `SALESPERSON_DEF`" (l.94) and "Probing shared graph state without
  mutating it" (l.152) all resolve. The `reference` wipe fact is at l.127, under "Key scripts",
  and the review-safe `test_services.py` subset is in the "Probing" section. Root "growth-stops
  bullet" resolves to l.173–184. Root "live-spec bullet" does not exist yet, because L2c adds
  it (P7-5). The breakage that L2c's move causes elsewhere is P7-2.

### Findings

**P7-1 (minor) — The duplicate list misses two reviewer-prompt duplicates of root *Always
check* 1.** Verified (`grep -n growth-stops claude/*/*.md`). Pass 6 says "No duplicates found in
`data-scientist.md`". Two prompt lines do duplicate root *Always check* 1:

- `claude/data-scientist/data-scientist.md` l.82: "the review also runs its growth-stops check".
- `claude/security-expert/security-expert.md` l.72: "the review also runs that bullet's check".

Both prompts are L3a files. The rows below are the full list, for L3a–L3c to use in place of Pass
6's table:

| Seed line | Prompt line | Overlap / L3 action |
|---|---|---|
| Root *Always check* 1 | `claude/analyst/analyst.md` l.66, "**Gating a revision of a document under root `AGENTS.md`'s growth-stops bullet:** run its check …" | Full for the check. Keep the `## Pass N` and Ready-for-design clauses (L3a). |
| Root *Always check* 1 | `claude/data-scientist/data-scientist.md` l.82, "the review also runs its growth-stops check" | Full. Becomes the `REVIEW.md` pointer (L3a). |
| Root *Always check* 1 | `claude/security-expert/security-expert.md` l.72, "the review also runs that bullet's check" | Full. Becomes the `REVIEW.md` pointer (L3a). |
| Root *Always check* 4 | `claude/analyst/analyst.md` l.66, "never into the step it is meant to catch … it can never redden" | Partial, as Pass 6 says. |
| Root *Always check* 5 | `claude/analyst/analyst.md` l.88 and l.85 | As Pass 6 says. |
| Root *Evidence bar* 1 | `claude/analyst/analyst.md` l.84, l.86; `qa-engineer.md` l.72; `security-expert.md` l.89 | As Pass 6 says. |

Three more lines say "the growth-stops check" or "that bullet's check", but they are coordinator
or authoring lines, not reviewer duplicates: `claude/teco/teco.md` l.144 (re-gate brief, L3c),
`claude/tico/tico.md` l.110 (Ready-for-design condition, L3b) and root `AGENTS.md` l.264 (`Status:`
set, L2c). They are not duplicates, but each is a citation that L2c breaks (P7-2). Pass 6 also
read "mutation" and "unfiltered" in `teco.md`. l.110 is an implementer-brief line and l.154 is
teco's own blast-radius sweep. Both are no-change, which confirms Pass 6.

**P7-2 (minor, apply before L2c) — After L2c, the inbound "growth-stops check" citations name a
check that `AGENTS.md` no longer states, and the seed line restates the bullet's scope.**
Verified by reading each citing line. L2c moves "Every review gate checks a revision against this
bullet …" out of root `AGENTS.md`. Six lines then point at text that is no longer there:
`analyst.md` l.66, `data-scientist.md` l.82, `security-expert.md` l.72, `teco.md` l.144,
`tico.md` l.110 and root `AGENTS.md` l.264.

The seed line also restates the bullet's scope as "requirements, plan or review document". That
drops the bullet's "(not `-coordination`)" exclusion, so the two copies already differ. Naming
the check, and citing the scope rather than restating it, gives all six a stable target.

Replace root `REVIEW.md`'s first *Always check* line with:

```markdown
- **Growth-stops check.** On a revision of a document that root `AGENTS.md`'s growth-stops bullet
  covers, say for any growth whether it is new scope or rework, and report rework growth,
  restated history and restated facts as findings. There is no numeric threshold.
```

L2c's citation left in the bullet then reads "Reviews run the growth-stops check (root
`REVIEW.md`)." L2c also repoints root `AGENTS.md` l.264 to the same check. L3a–L3c repoint the
five prompt lines.

**P7-3 (minor) — falkor-chat *Always check* 1 has a partial enforcer that the line doesn't
name.** Verified (`falkor-chat/scripts/verify_salesperson.sh` header, check 6;
`start_demo.sh` l.313–316). Check 6 compares every stored step `config` with the shipped
`proof_defs` constant, so it catches a version stored from an earlier or uncommitted tree.
`start_demo.sh` runs it as a preflight after seeding. That is detection after the version is
burned (create-only), not prevention. It covers only `salesperson` and `order-fulfillment`:
`verify_workflows.sh` has no equivalent. Under FR-9 the line stays, but a reviewer should know
which part a script already shows. Replace the line with:

```markdown
- A change to a published workflow def's create-only fields (`config.tools`, `systemPrompt`,
  `config.model`) cannot be published under an old or uncommitted version name by any step in
  the order the plan runs them: a seed, `start_demo.sh`, or a re-seed after a default `pytest`
  wipes `reference`. Per `falkor-chat/AGENTS.md`, "Three traps when bumping `SALESPERSON_DEF`".
  For `salesperson` and `order-fulfillment`, `./scripts/verify_salesperson.sh` (check 6, also
  `start_demo.sh`'s preflight) shows such a publish only after the version is burned: in a plan
  review check the order; in an implementation review, a green run after the last seed is the
  evidence.
```

**P7-4 (nit) — Root *What counts as serious* 1 restates a process fact.** Inferred from the
plan, not run. "The feature cannot close while the verdict is 'needs changes'" is FR-7's close
rule. Its home is root `AGENTS.md`'s feature-close bullet (§3.1, written by L2c). The clause
doesn't change what a reviewer does, because severity already decides the verdict. Replace the
line with:

```markdown
- In a spec-update check, a contract the shipped diff changed that the live spec omits or
  contradicts is a blocker.
```

**P7-5 (info, L2c constraint) — "root `AGENTS.md`, live-spec bullet" dangles until L2c.**
Verified: no such bullet exists today. This is harmless, because no reviewer prompt points at
`REVIEW.md` before L3a. L2c must give the new bullet a bold lead that this citation finds (e.g.
`**A live spec …**`), and must keep the growth-stops bullet's bold lead intact.

**P7-6 (info) — `typed-decision-models.md` reuses finding IDs across passes, so its bare
citations are ambiguous.** Verified: m3 (l.119, l.434), m4 (l.126, l.441), M5 (l.239, l.380) and
m7 (l.268, l.457). I checked each of Pass 6's uses. The intended finding is m3 l.434, m4 l.441,
M5 l.239 and m7 l.457, and each supports its line. A future proposal citing that document should
add the line number.

### Judgment calls

**1. `model-bench/REVIEW.md` "bar tests probe both sides": keep it, widened.** It isn't redundant
with root *Always check* 4. That line asks for *a* failing mutant. `typed-decision-models.md` m4
is a suite that did have one (351 ms → not met reddens an always-met bar) and still missed the
direction flip. So this line names *which* mutant, and that changes what a reviewer does.

The class recurs more strongly than Pass 6 shows.
`docs/reviews/small-model-benchmarking-ml.md` m-ML-7 (l.626, 2026-09, in window) is a
`> 1.0`→`>= 1.0` mutation that "survived all 314 tests". `typed-decision-models-impl.md` P2-m1 is
a different class: a default value used as the test value. The line's sources are therefore m4 and
m-ML-7. m-ML-7 is a report-clause comparison, not a bar, so replace the line with:

```markdown
- A test of a bar, threshold, verdict or report-clause comparison probes both sides of its
  boundary, including the equality case, so a `<`/`<=` or `>`/`>=` mutation turns one red.
```

**2. Self-gating rule not seeded: `analyst` was right.** Verified on two counts:

- **Window.** It fails §3.3's category-3 bar. An unfiltered `git grep` over every review for
  self-gating phrasing finds four documents. Two are outside the window
  (`graph-dba-kaizen-distillation.md`, last commit 2026-08-19; `analyst-inbox-distillation.md`,
  2026-08-19). Two are false positives, about an agent's *own prompt* rather than its gate
  (`doc-reference-convention.md` l.842, `small-model-benchmarking-impl.md` l.4082). That leaves
  only this review.
- **Home.** It isn't a reviewer's behavior. Assigning gates is the coordinator's duty, and
  `claude/teco/teco.md` l.176 ("No deliverable is accepted on its producer's word alone") and
  plan §1 already own it. In `REVIEW.md` it would be a second copy of a coordination rule.

If it recurs, the right proposal is a plan-review check ("a unit's gate is never its writer or
applier"), sourced from that recurrence.

### Not changed

- *Do not report* 1 is narrowing, so this gate covers it (§3.4 step 4). Approved: it is FR-7's
  "not the plan" half, and it is correctly scoped to spec-update checks.
- model-bench *Always check* 4 partly overlaps root *Always check* 4 (independently written
  expected values). Kept: its docstring-coverage clause is model-bench-specific and is sourced
  (`typed-decision-models-impl.md` P6-m1).
- Sizes after P7-2–P7-4 and judgment call 1 stay well inside §3.3's targets. Each replacement
  changes a line by about ±25 words.

## Pass 9

**Scope.** Gate for unit L5b (`analyst`). It checks the uncommitted rewrite of
`falkor-chat/docs/SERVER.md` (12,080 words; the baseline `git show c0eeb348:falkor-chat/docs/SERVER.md`
has 9,806) against the shipped code at **pinned sha `c0eeb348`**. `git diff --quiet c0eeb348 -- server/ web/ config/ scripts/`
is clean. Read: plan §3.5 and §5; requirements FR-3, AC-1 and AC-3; root and `falkor-chat/REVIEW.md`;
and the writer's drift inventory. I built the inventory independently from `api.py`, `mcp.py`,
`config.py`, `app.py`, `storefront_api.py`, `tools.py`, `querygen.py` and `modelconfig.py`, plus
`services.py`, `background.py`, `trigger.py`, `responder.py`, `schemas.py` and `web/app.js`
wherever a claim reaches into them. Mid-pass, the coordinator added three cross-reference items
from L5c. They were checked against the current uncommitted `falkor-chat/docs/QUERIES.md`.

CPG: used cpg_falkorchat — to confirm `_schedule_background`'s only caller (`mcp.send_message`, l.212) and that `AGENT_NAME` has no use besides its definition (`AGENT_ID` has 6 field accesses).

**Verdict: needs changes.** The coverage is complete, and the four writer claims hold. Four
statements contradict or omit an externally observable contract. Root `REVIEW.md` ranks each of
those as a blocker, and AC-1 at L5s would report each one. Every fix is a sentence or less.

### Inventory vs. document (verified)

| Surface | Code | `SERVER.md` | Result |
|---|---|---|---|
| Dev REST routes (`api.py`) | 44 decorators | 44 paths in §1.4's three tables (11 chat + 18 documents + 15 workflow) | complete. The writer's "~35" counts table rows, where confirm/reject/recheck share a row |
| `/health` re-registration (`app.py:497`) | 1 | §1.3, §1.4 | match |
| `/shop/api` (`storefront_api.py`) | 11 routes, cred deps at l.1083–1474 | 11 rows with matching paths, creds and body bounds | match |
| MCP tools (`mcp.py`) | 25 `@mcp.tool()` (27 grep hits, 2 in docstrings) | 25 in §2.2 | match |
| Config keys (`config.py`) | 26 env vars + 4 legacy | all in §1.3 | complete. Only `modelconfig.py:291` reads other env vars, for `{env:NAME}` substitution, as §1.3 says |
| Agent tools (`tools.build_builtin_registry`, l.1340–1354) | 13 | 13 in §1.11 | match. Checked: retrieve τ/cap/k = 0.5/5/10, filter cap 20, `querygen` filter ≤4 / returns 1–6 / limit 1–50 (default 20), both dataset allowlists, finding strings |
| Error envelope (`app._register_error_handlers`) | 404/400 split, def and run maps, `SearchNotAvailableError` 503 | §1.4 | match |
| Mounts and lifespan (`app.py:403–551`) | `/mcp` + alias, `/shop` `_SPAStaticFiles`, `/` last | §1.3, §1.4, §1.5, §2.1 | match |
| Trigger rule and responder (`trigger.py:52–95`, `responder.py`) | 4-step rule, `channel_id=None` | §1.12 | match |

**AC-3:** the plan's §5 grep, run verbatim on `SERVER.md`, returns 0 hits. The `K-` references
(K-016, K-018, K-074, K-080) are all open headings in `BACKLOG.md`. A wider sweep for `M<n>`, `S<n>`,
`U<n>`, "as of", "now", "delivered", "previously", plan/review paths, `FR-`/`AC-`/`D-` tags finds no
history. The two "legacy" hits name a live guard. **Numbering:** §1–§1.10 and §2–§2.3 keep their
numbers. Two titles lost their history suffixes ("locked for M1", "(as built, M1)"). §1.11 and §1.12
are appended. The inbound `§1.6` citations (`DESIGN.md` l.949, `falkor-chat/docs/plans/workflow-catalog-lookup.md`
l.359) now land on the one-line pointer, which is acceptable. **Growth** (+2,274 words) is new
scope: the catch-up's missing surfaces (document REST, 18 MCP tools, 16 config keys, §1.11, §1.12).
`SERVER.md` is a live spec, not a growth-stops kind. Restated facts are M2.

### Writer claims

1. **MCP `send_message` schedules the trigger: confirmed, by execution.** I wrote a probe in a
   scratch directory. It configures `mcp` with a fake `Services` and an inline `_schedule`, then calls
   `send_message`. It scheduled `['_safe_embed', '_safe_run_workflow']` with a trigger wired,
   `['_safe_embed', '_safe_respond']` with only a responder, and `['_safe_embed']` with neither.
   `app.py:403–410` passes the same `trigger`/`responder` to `mcp.configure`. The old §1.7 bullet
   (baseline l.548) has been stale since `17c2fa03` (K-041).
2. **Step-budget exhaustion on `/input`: confirmed.** A probe of `Services.submit_workflow_input`
   used a stub executor whose `resume` returns `"failed"`, as `executor._fail_budget`
   (`executor.py:1495–1501`) does without raising. It returned
   `{'runId','status':'failed','ctx':<submitted merge>}` with no `error`. The graph `ctx` held the
   note, and the envelope did not carry it. `services.py:2439` (`merged if fault_ctx is None`) is
   the cause. `tests/test_process_input.py:686` pins the no-`error` half for `POST /workflow-runs`.
   §1.4 l.422–425 states this correctly.
3. **`FALKORCHAT_AGENT_NAME` is parsed but unread: confirmed.** The CPG has one identifier, at
   `config.py:87`, and no field access. `grep` finds no reader in `falkorchat/`. See n2 for how the
   row reads.
4. **`RAG_QUERY_TIMEOUT_MS` is not configuration: confirmed.** It is the module constant
   `services.py:156` (`= 5000`), read from no environment variable. Keeping it out of §1.3 is
   correct.

### Coordinator's L5c items

1. **Confirmed** (Q-L5c-1). `services.ingest_document` calls
   `repository.create_document_with_auto_supersede` (`services.py:1222`). The current `QUERIES.md`
   §14.1 says `create_document` "has no server caller", and §14.9 is the ingest write. `SERVER.md`
   l.279, l.280, l.1025 and l.1026 cite §14.1. See m1.
2. **Confirmed.** `QUERIES.md` §14.7 (delete/deletion/list) and §14.8 (`SUPERSEDES` review and
   history) exist. Their "—" cells in `SERVER.md` are m1.
3. **Stated in `SERVER.md`.** §1.12 l.967–971 covers one daemon thread per job, unbounded, inside
   the tool call, a thread-start failure being logged and skipped, and threads untracked by
   `_lifespan`. It is restated in §1.7 l.654–659 (M2). The posture survives L5c's removal. Its size
   is understated (n1).

### Findings

**B1 (blocker): §1.7 l.627–628 says `FALKORCHAT_WORKFLOW_ENABLED=1` alone builds a
`ModelGateway`.** Verified: `app._build_default_app` returns at `app.py:588–592` when
`ENABLE_AGENT` is off. `WORKFLOW_ENABLED` is read only at l.630, inside the agent branch. This
contradicts the code and §1.3's own row (l.172, "With `ENABLE_AGENT` off it changes nothing"). The
sentence was carried over unchanged from the baseline (l.520) and drifted against the new table.
**Fix:** "`FALKORCHAT_ENABLE_AGENT=1` builds a `ModelGateway.from_env()`…", or cut the bullet's
first sentence and cite §1.3/§1.8 (M2).

**B2 (blocker): §1.3 l.172 says that with the engine unwired "every workflow-run route answers
`503 WorkflowEngineDisabledError`".** Verified: `_require_executor` is called only by
`start_workflow_run`, `submit_workflow_input`, `resume_workflow_run` and `sweep_due_workflow_runs`
(`services.py:2288/2385/2620/2697`). With the engine unwired, `GET /workflow-runs/{id}`,
`/step-runs`, `/trace` and `GET /threads/{tid}/workflow-runs` serve normally (`services.py:2817–2860`).
**Fix:** "…`POST /workflow-runs`, `POST /workflow-runs/{runId}/input` and `POST /workflow-runs/due`
answer `503 …`; the run reads still serve." §1.4's error-map row (l.406) is already scoped
correctly.

**B3 (blocker): §1.4 l.341 says "path ids ≤ 200".** Verified in `api.py`. `Path(max_length=…)` is
declared only on `document_id` (delete/deletion/history), `match_id`, `run_id`, and the
structure/materialize/snapshot `key`/`version`. The following are plain `str`, so a 300-character
id reaches the service instead of answering `422`: `channel_id` (l.80, l.87), `thread_id`
(l.95, l.130, l.395, l.405), `msg_id` (l.149), `GET /documents/{document_id}` (l.251),
`GET /workflow-defs/{key}` (l.436) and every `{ws}`. This sentence is new, not from the baseline.
**Fix:** name the bounded parameters ("document, match and run ids and def/snapshot key/version
path segments ≤ 200; other path ids are unbounded"), or drop the clause.

**B4 (blocker): §1.12 omits ingestion with nothing wired, which is the default configuration.**
Verified. With `FALKORCHAT_ENABLE_AGENT` off, `embed_worker`, `ingestion_pipeline` and
`repo_for_detection` are all `None` (`app.py:399–401`). Then `_schedule_chunk_processing` never
calls `start_document_progress` (`background.py:213`), so a document created with
`status:'processing'` (`repository.py:1080`) stays `processing` permanently. `_schedule_update_detection`
returns early (`background.py:317`), so no `SUPERSEDES` suggestion is ever made. The l.964–965
detection sentence reads as unconditional. **Fix:** one sentence. "With neither wired, nothing is
scheduled: the document stays `processing` and no update detection runs." Make the detection
sentence conditional on the same wiring.

**M1 (major): §1.4's run error map understates two producers.** Verified. (a) The "fault during the
drive" row (l.407) gives two examples. `_drive_or_fault` (`services.py:2594–2597`) catches four:
`ModelResolutionError` and `ProviderCallError` too. Those reach `POST /workflow-runs` for any def
with an agent node or an LLM guard. (b) `404 WorkflowRunNotFoundError` is listed as "unknown run
id". `POST /workflow-runs` also returns it when the snapshot has no START or is not materialized
(`services.py:2339–2343`). (c) Neither body is stated: `POST /workflow-runs` answers
`{runId, status, defKey, defVersion, trace[, error]}` with **no** `ctx`, and `/input` answers
`{runId, status, ctx[, error]}`. Without that, the "status **and** ctx" paragraph (l.413) reads as
if it applies to both. **Fix:** list the four fault types, add "or a start whose snapshot is
missing" to the 404 row, and give the two bodies in one line.

**M2 (major): facts restated across sections, one of which has already drifted (B1).** The step
budget is stated at §1.4 l.422–425 and §1.7 l.651–653. MCP vs REST schedulers are stated at
§1.12 l.967–972 and §1.7 l.654–659. The legacy-env refusal is stated at §1.3 l.185–187 and §1.8
l.744–745. The two config files are stated at §1.7 l.627–636 and §1.8 l.737–743. Each pair
currently agrees except the last, which is B1. **Fix:** each §1.7 hazard keeps its testing
consequence and cites the home section for the fact. Delete §1.3 l.185–187, because §1.8 is its
home.

**M3 (major): §1.11 omits `query_graph_data`'s default projection.** Verified:
`querygen.default_returns` (`querygen.py:323–360`, applied at `tools.py:1218`) fills an
**absent** `returns` with `["<var>.name"]` when the label has a `name`. The exception is an
identity `=` lookup, which stays invalid and abstains with "no matching data found". So a
completion that omits `returns` produces a name list, where §1.11's "1–6 `returns`" implies
rejection. Per `falkor-chat/REVIEW.md`, the tool result is behavior. **Fix:** one sentence after
the request-shape paragraph.

**m1 (minor): `QUERIES.md` cross-references are stale after L5c.** Ingest cites §14.1 at
l.279/280/1025/1026; it should be §14.9 `create_document_with_auto_supersede`. The "—" cells for
list/delete/deletion (l.282, l.284–285, l.1029–1031) should be §14.7. History and the
document-update review (l.286–289, l.1032–1035) should be §14.8. Also update l.257's note, which
says "—" means the query lives only in `repository.py`. This holds only if L5c's `QUERIES.md` lands
as drafted; serialize with it.

**m2 (minor): §1.5 l.579–582's list of dev-UI calls omits the workflow-def reads.**
`web/app.js:288` calls `GET /workflow-defs`, and l.307 calls
`GET /workflow-defs/{key}/versions/{version}` (def list and structure view).

**n1 (nit):** §1.12 l.969 says "a max-size batch ingest can spawn thousands". The bound is about
20 × ⌈500,000 / 850⌉ × 2 ≈ 23,500 threads per call (`chunking.split_into_chunks` size
1000/overlap 150, `MAX_BATCH_SIZE` 20, two jobs per chunk). State the arithmetic, or say "tens of
thousands".

**n2 (nit):** the §1.3 l.170 row "Parsed but read by nothing in the server" is accurate. But
`scripts/seed_demo.sh:47` and `start_server.sh:116/202` read the same variable directly as the
seeded `Agent`'s display name. Add that clause so nobody drops it from the scripts.

### §1.9 placement ruling (for L5s)

§1.9 describes `executor._assemble_messages`/`_append_turn` (`executor.py:233`, l.1362). That is
**executor semantics**, which the §3.5 home table assigns to `DESIGN.md`. `DESIGN.md` §6.1's "The
`agent` node loop → Prompt" bullet (current l.413–418) already states the order and the same-role
merge. **Ruling:** home it in `DESIGN.md` §6.1. Fold §1.9's two non-duplicate facts into that
bullet: the system message is resent on every tool-loop `chat` call, and the `CONTEXT` block gets
no fresh turn per iteration, so standing constraints belong in `systemPrompt`. Then reduce
`SERVER.md` §1.9 to a one-line pointer, keeping its number for the inbound citation
(`claude/analyst/kaizen/history.md` l.75). This is not a defect of L5b. The behavior §1.9 states
matches the code.

### What's solid

The tenancy, storefront-credential and cross-cutting-error prose matches the code line for line.
§1.11's 13-tool table and dataset allowlists match the code exactly. The §1.3 config table is
complete and its defaults are right. The four disputed claims were all right.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: this Pass, B1 and M2. A restated fact drifted
inside one live-spec document while the writer edited the other copy.

```markdown
- In a live-spec check, a fact stated in two sections of the spec set is a finding, and every
  copy is checked against the code, not only the one the change edited.
```

## Pass 8

**Scope.** Unit L5a gate (`analyst`): the uncommitted rewrite of `falkor-chat/docs/DESIGN.md`,
checked against the shipped code at **`c0eeb348`**. `git diff --quiet c0eeb348 -- falkor-chat/server
falkor-chat/scripts falkor-chat/config falkor-chat/web` exits 0, so the working tree is that code.
Read: plan §3.5 and §5; requirements FR-3, AC-1 and AC-3; root and `falkor-chat/REVIEW.md`; the
pre-edit original; the rewritten `SERVER.md` section list; `BACKLOG.md`. The contract inventory
was built from the code first (`cpg_falkorchat`, plus source reads of `repository.py`,
`executor.py`, `services.py`, `guards.py`, `background.py`, `retry.py` and
`scripts/bootstrap_schema.sh`). I read the writer's `drift.md` only afterwards. I did not run the
pytest suite or `test_queries.sh`: no code changed, and both wipe `reference`.

CPG: used cpg_falkorchat — independent writer/reader inventory for `lshBand*`, `MENTIONS`,
`REPLY_TO` and the banding call sites; the rest came from source reads (`scripts/` is not in the
graph).

**Verdict: needs changes.** The schema, DDL, write-path table, executor agent loop, run model, AC-3
history removal, numbering and §14 map all hold against the code. Two statements of workflow
semantics remain that the code contradicts or that omit something. Both were carried over from the
original text, and AC-1 counts each one against the catch-up. Both fixes are one or two sentences.

### Writer claims (rulings)

1. **Lock value**: confirmed by running the recipe. The §6.2 recipe, run verbatim, gives
   `6cd35d650b5d` at the worktree, at `c0eeb348` and at `15a9749b`. It gives `71055f756280` at
   `15a9749b~1`, so the old value has been stale since K-056 (`15a9749b`). Positive control: adding
   a trailing space to line 5 of the extracted body gives `c62ed522d75b`. The awk bound captures
   exactly one `def` (`_drive_loop`, l.695–758, ending at `# ── seams`, l.759).
2. **`MENTIONS` and `REPLY_TO` have no writer**: confirmed. The CPG, over non-test methods, finds
   one literal for each, and both are reads: `Repository.hybrid_search` l.891 and
   `Repository.get_message` l.2414. A grep of `scripts/` and `web/` (not in the CPG) finds the only
   `CREATE (m)-[:REPLY_TO]` in `test_queries.sh` l.212, which is test seeding. That hit is the
   positive control for the pattern.
3. **`Document.lshBand0..7` are never written**: confirmed by the CPG. In production code the only
   nodes whose `CODE` contains `lshBand` sit in `find_update_shortlist` (the read-side f-string
   `d.lshBand{i} = $band{i}`, l.1985) and in the `lsh_bands` docstring. `lsh_bands` and
   `minhash_signature` have one production caller, `background._safe_detect_update` (l.272–273).
   It passes the bands to the shortlist read and stores nothing. Positive control: the same query
   finds the test-only writer `tests/test_repository.py::_document_with_lsh` (l.1781–1802).
   `create_document_with_auto_supersede` (l.1889) writes no band property.
4. **§7.1/§7.2 match `bootstrap_schema.sh`**: confirmed, line by line. All 20 constraint-backed
   `ws:` labels match, as do the 4 relationship indexes, all 17 hot-filter indexes (including
   `lshBand0..7` from the l.207 loop), the 3 full-text indexes, 2 vector indexes with default dim
   `1536`, and the `reference` table (`WorkflowDef` composite, `Entity`, `Step`, `Product` plus 3
   filter indexes). `grep -n OrderLine scripts/bootstrap_schema.sh` returns nothing, while `Order`
   matches (l.287/l.354). So "`OrderLine` has no index" is true.

### Mechanical checks

- **AC-3**: the verbatim §5 grep on `DESIGN.md` gives 0 hits. `K-` references: K-016, K-018,
  K-029 and K-033. Each has an open `### K-0NN … (🔵 proposed …)` heading in `BACKLOG.md`.
- **AC-9 sweep** (DESIGN.md only): 0 hits.
- **Numbering**: the original and the rewrite have identical heading-number sequences. Only the H1
  text differs ("Design Document & Blueprint" became "Design"). §12 is a pointer. §14 is a
  present-tense map.
- **§14 map**: all 13 rows resolve to existing `SERVER.md` headings (§1, §1.1–§1.8, §2,
  §2.1–§2.3). The inbound source citations use only §14, §14.1–§14.8, §15 and §15.1–§15.3, all
  covered.
- **QUERIES citations**: every `QUERIES §x.y` cited in DESIGN exists in the rewritten
  `QUERIES.md`.
- **Growth-stops**: not applicable. `DESIGN.md` is a living spec, not a `requirements/`, `plans/`
  or `reviews/` document. The growth from 818 to 955 lines is new scope: previously omitted
  contracts (commerce, documents, `WorkspaceConfig`, the agent loop).

### Findings

**P8-B1 (blocker, verified): §4 and §9 say a def is materialized automatically. The code never
does that.** §4 l.165–167: "When a workspace first uses `defKey@v`, we **materialize** …". §9
publish row l.819: "write to `reference`; materialize snapshot into consuming `ws:{id}` graphs".
In the code, publish (`services.publish_workflow_def`, l.1906) writes only `reference`.
Materialization is a separate explicit operation (`services.materialize_def` l.1977;
`POST /workflow-defs/{key}/versions/{version}/materialize`, `api.py` l.471), and the seed scripts
call it. No run-start path materializes: `materialize_snapshot` has no caller in `app.py`,
`trigger.py`, `executor.py` or `start_workflow_run`. A start against a version that was never
materialized raises `WorkflowRunNotFoundError` (`services.py` l.2340, "snapshot … has no START")
and returns 404 (`app.py` l.175). A def author who follows §4 gets a 404.
*Fix:* in §4, "A version is materialized into a workspace by an explicit materialize call
(`SERVER.md` §1.4) before any run can start there; a run against an unmaterialized version is
refused (404) and nothing is written." In §9 split the row: "Publish workflow def: write to
`reference`" and "Materialize def: copy one published version into `ws:{id}`". Keep the topology
409 on both.

**P8-B2 (blocker, verified): the publish-invariant list is presented as complete but omits four
timer rejections.** §6.1 l.447–454 lists the invariants as complete ("any violation rejects the
publish"). It is missing these rejections from `services._validate_def_spec` (l.1799–1860):
(a) only a `wait`/`human` step may declare `waitForSeconds`/`waitUntil`; (b) a step may declare at
most one of the two; (c) `waitForSeconds` must be a positive number ≤ `MAX_WAIT_FOR_SECONDS`
(180 days, l.193); (d) `waitUntil` must be a positive epoch-ms. The marker guard is also exact:
`{kind:'cmp', path:'ctx.timerFired', op:'eq', value:<own key>}`. A def author reading §6.1 cannot
predict these 400s. The current pointer to `docs/plans/workflow-timers.md` is not the spec (FR-3).
*Fix:* replace the timer clause at l.451–453 with (a)–(d) plus the exact guard shape. That adds
about two lines.

**P8-m1 (minor, verified): §5.4 names an edge and a cap that do not exist.** l.338–339 says "Entity
fan-out (`MENTIONS`) … mitigate by capping entity extraction per message". §5.1 and §8 of the same
file say `MENTIONS` has no writer. Entity fan-out actually comes from `ABOUT` (Chunk→Entity),
`RELATES_TO` and `SAME_AS`. Extraction is already capped at 20 entities and 20 relationships per
**chunk** (`extraction.py` l.61–62). *Fix:* restate §5.4 with those edges and the per-chunk cap.

**P8-m2 (minor, verified): the guard semantics omit three facts.** (a) An `llm` guard may carry
its own `model`. It is resolved as the `guard` kind and checked at publish
(`services._declared_model_refs` l.633–664). §6.1 shows `{kind:'llm', text}` only, and l.453–454
says only "a step naming a model". (b) A guard whose kind is unknown or missing, or that is a
non-JSON string, passes publish (`_validate_def_spec` l.1740–1747 validates only the cmp family).
It then raises `NotImplementedError` at drive time and fails the run (`guards.evaluate_guard`
l.271). "The guard string is one of …" together with "strict at publish" implies such a guard would
be rejected at publish. (c) The sort key is `(guard == "", order, to)` (`executor.py` l.1448), not
`(guard == "", order)`. *Fix:* add `model` to the llm form and "or a guard naming one" to the
publish line. State that only `cmp` guards are validated at publish, and that any other unknown
kind fails the run when reached. Add `to` to the sort key.

**P8-m3 (minor, verified): the §6.2 engine-loop sentence lists three outcomes. The code has four.**
At l.577–580, when no guard fired, the step has outgoing transitions and does not declare
`waitsForHuman`, `_drive_loop` re-executes the same step ("OUTCOME C — re-loop", l.749–753), within
the `maxSteps` budget. This is stated only for `decision` steps (the K-029 note), but it applies
to every step type, `agent` included. *Fix:* add "or re-run the same step (no guard fired,
transitions exist, no park declared; bounded by `maxSteps`)".

**P8-m4 (minor, verified): the §9 reset row understates what is deleted.**
`_RESET_PARTICIPANT_CYPHER` (`repository.py` l.4006ff.) also deletes the participant's `Customer`
node, which holds the profile `name` and `deliveryAddress`. It also deletes `OrderLine`s and
re-points `User.threadId` to the new thread. A reset therefore erases the saved profile, and the row
does not say so. *Fix:* name `Customer` (profile) and `OrderLine` in the row, and add "and
re-points `User.threadId`".

**P8-m5 (minor, verified, depends on `teco`): §13 l.924–929 says the open items are "tracked in
`BACKLOG.md`".** Retention and cross-workspace analytics are tracked (`BACKLOG.md` l.882–883), and
so is K-018. The pre-production `GRAPH.CONFIG` review has no item. The writer listed it as an
orphan for `teco` to file (plan §3.5). The text becomes true when that item is filed. If it is not
filed, drop that clause.

**Nits.** (n1) The following references point at QUERIES sections or a plan's section but carry no
document prefix, so they read as references to DESIGN sections that do not exist: l.55 and
l.264–265 (`§9.2/§5`, `§9.1`), l.328 and l.703 (`§9`), and l.625 (`§4.3`, which is
`m3-process-flow.md`'s). (n2) Some lines still carry history residue: l.173 "…stays a silent
no-op, unchanged", l.523 "New node type → new DDL", and l.632–633 "The executor's drive loop is
**not modified** to support any of it". Each reads relative to a past change. (n3) The §14 row for
`§14.6` has lost its label. That is harmless if L5b reduces `SERVER.md` §1.6 to a pointer. (n4)
`CartItem.productId` appears in both §7.1 tables.

**Outside this file, for `teco`.** `BACKLOG.md` K-033 (l.540–548) still quotes `71055f756280`,
which claim 1 shows is stale.

### What's solid

The §5.1 arrow schema matches every label and relationship type the server code writes or reads:
23 labels and 32 relationship types from an inventory of `CREATE`/`MERGE`/`MATCH` patterns. It
also matches the property sets of every `CREATE`/`MERGE` map I checked, the storefront participant
included. The agent-loop section matches `executor.py` exactly: `THREAD_CONTEXT_WINDOW = 20`,
`DEFAULT_MAX_ITERATIONS = 4`, `_GROUNDING_SOURCES`, `MODEL_CORRECTABLE_TOOL_ERRORS`, the four signal
names, and the implicit `post_message` fallback. So do the `maxSteps` semantics
(`DEFAULT_STEP_BUDGET = 12`, `>` checked only on outcomes A and C, `"step budget exceeded"`), the
§9 retry policy (`retry.py` l.122) and the 0.1 noise floor (`background.py` l.33). The dormant-edge
and `lshBand` notes are accurate, and they are welcome.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: P8-B1 and P8-m1. Both are contradictions carried
over unchanged from the pre-edit text. A check of the rewritten lines alone would have passed them.

- Addition: "In a spec-update check, read the lines the update left unchanged too. A contradiction
  carried over from the previous text is a finding, the same as a new one."

No prune: no test, CI job or hook checks spec prose.

## Pass 11

**Scope.** Unit L2c gate (`analyst`): `cobb`'s uncommitted edits to root `AGENTS.md`,
`falkor-chat/AGENTS.md` and `falkor-chat/README.md` (`git diff -- AGENTS.md falkor-chat/AGENTS.md
falkor-chat/README.md`). Baseline: plan v1.2 §3.1, §3.2, §3.4, §4 (L2c row), §5 (AC-7, AC-9);
requirements FR-2, FR-3, FR-4, FR-7, FR-8, FR-11, FR-12, FR-13, AC-2, AC-7, AC-9; Pass 7 P7-2,
P7-4, P7-5; root `REVIEW.md`; ledger `docs/plans/live-specs-coordination.md` l.14, l.35, l.39.
Prompt files (L3a–L3c) and the spec documents themselves (L5a–c) are out of scope.

CPG: not applicable — the artifact is convention prose in context files, no code change.

**Verdict: approve with suggestions.** The five §3.4 steps are all present and in order, and so
are "no close on needs changes" and the `REVIEW.md` upkeep rule. The live-spec bullet matches
FR-3, FR-4, FR-12 and FR-13. Both `REVIEW.md` citations resolve, and P7-2, P7-4 and P7-5 are
satisfied. Size is within the cap. M1 has to land before L3c's gate, because AC-7's prescribed
check fails as written. Applied together, M1 and m1 change no word count (verified below).

### Checks run

- **Size** — verified. `wc -w AGENTS.md` gives 3,347, and `git show HEAD:AGENTS.md | wc -w` gives
  3,198: +149, within the +150 cap. `falkor-chat/AGENTS.md` grows by 20 words (2,768 → 2,788).
  The `awk 'length($0)>700'` bar shows no hit in any line L2c touched. Its only hits are
  `falkor-chat/AGENTS.md` l.87 and l.88, which were also over the bar at HEAD (l.85 and l.86
  there). See i1 for the bar command's line numbers.
- **Context-file rule (rewrite, not append)** — verified. Each pre-existing entry L2c touched was
  replaced in place: the table row, the growth-stops tail, the `BACKLOG.md` surface and the
  `Status:` citation. Nothing was appended after an earlier author's text. The two new bullets
  are new scope (FR-2–FR-4, FR-7, FR-8, FR-11–FR-13) and carry no history, dates or transitional
  text. No rework growth: the moved growth-stops sentences (38 words) became a 7-word citation,
  the `BACKLOG.md` surface shrank by 1 word, and the table label and `Status:` citation add 4. The file was over its ~2,500-word smell before L2c,
  and the plan accepted that.
- **Citations** — verified. Root `REVIEW.md`'s "root `AGENTS.md`, live-spec bullet" resolves to
  the bold lead **A live spec is a component's current behavior** (l.160), as P7-5 requires.
  "Reviews run the growth-stops check (root `REVIEW.md`)" (l.196) and the `Status:`-set citation
  (l.276) both resolve to `REVIEW.md`'s bold **Growth-stops check.**, as P7-2 requires. The
  growth-stops bullet's own bold lead is unchanged, so `REVIEW.md`'s "root `AGENTS.md`'s
  growth-stops bullet" still resolves. "The next bullet bounds it" (l.179) still points at the
  growth-stops bullet, because the new bullets were inserted above l.174.
- **Review-only rules left in root `AGENTS.md`** — read in full and ruled. The moved sentence is
  gone, and `REVIEW.md` *Always check* 1 holds it. The remaining review-related lines are
  document conventions, not review judgment: "The review's next `## Pass N` records outcomes"
  (l.193), the compact-`## Pass N` rule in collision rule 5 (l.248–251), the `reviews/*` → `analyst`
  flip owner and the Ready-for-design condition. None belongs in `REVIEW.md`. P7-4's clause now
  has its home in step (3).
- **AC-9 sweep** — run verbatim from the repo root: 90 hits. That is fewer than §5's 107 because
  concurrent L5 edits have landed. Each hit in the three L2c files has an owner:
  - Edited by L2c: root l.90; `falkor-chat/AGENTS.md` l.12 and l.201;
    `falkor-chat/README.md` l.23 and l.237. Root l.190 (now the `BACKLOG.md` line, l.202) no
    longer matches.
  - No change: root l.52 and l.96 (`kiro/DESIGN.md`), and root l.175, the living-document
    bullet. That bullet lists `DESIGN.md` as a document read whole and never frozen, which agrees
    with a live spec. `falkor-chat/AGENTS.md` l.14, l.27, l.61 and l.192, and
    `falkor-chat/README.md` l.193, cite sections only.
  - `grep -i 'clos(e|es|ed|ing)|done|deliver'` over the three files finds no line that lets a
    feature close without the spec update.
- **AC-2 (falkor-chat half)** — `grep -n 'live spec' falkor-chat/AGENTS.md` hits l.12, which names
  all three documents. The "Key documents" rows were also checked against `DESIGN.md`:
  dropping "roadmap" is correct, because §12 is already a pointer.

### Findings

**M1 (major, verified) — The close bullet never names pruning, so AC-7's check fails.** §5 AC-7
reads "root `AGENTS.md`'s close bullet names applying and pruning". Step (4) at l.169–171 says
"applies the reviews' proposed `REVIEW.md` lines". `grep -n -i prune AGENTS.md` finds nothing.
FR-9 makes removal a duty of whoever adds a rule. The always-loaded text gives `teco` no cue that
a proposal can delete a line, and only root `REVIEW.md` *Always check* 6 says so, which `teco`
reads only for line checks. Replace "proposed `REVIEW.md` lines verbatim" with "proposed
`REVIEW.md` additions and prunes verbatim" (+2 words, offset by m1).

**m1 (minor, verified) — The bullet loses FR-2's "the spec update happens once, at that close",
and its lead reads as a trigger.** In "A feature closes once the live spec states what it
shipped", "once" can mean "as soon as". That reading makes a spec edit close the feature, which
inverts step (5). The bullet also never says that the close is the *only* time the spec is
updated. That is the half of FR-2 that AC-9's "same change" clause depends on, and for
implementers it is otherwise stated only in L3c's `teco.md`. Rewrite the lead and offset the
words in the live-spec bullet (this also fixes the dangling "Never `archived`, they carry"):

```markdown
  - **A live spec is a component's current behavior** — the documents its `AGENTS.md` names:
    externally observable contracts, no change history. They are never `archived`, carry no
    `Version:` or revision note and keep their numbering (an emptied section keeps its heading
    and a pointer). A plan against it states only the change, citing the affected sections and
    any open feature's plan it relies on.
  - **The live spec is updated once, at a feature's close.** A feature is one backlog item,
    else one requirements document, or a `HISTORY.md`-recorded spec-contract change. `teco`'s
    close: (1) every build unit is gated; (2) a delegated writer updates the cited sections and
    any contract the diff changed; (3) `analyst` checks that against the shipped diff
    (`reviews/<slug>-impl.md`); no close on "needs changes"; (4) `teco` applies the reviews'
    proposed `REVIEW.md` additions and prunes verbatim, their only source; `cobb` gates a
    narrowing one (`analyst` if `cobb` proposed it); (5) the `HISTORY.md` entry, last, cites the
    updated sections and the check. A spec-less component gets a whole-component spec, checked
    as in (3), before its first close.
```

Verified on a scratch copy with M1 applied: `wc -w` = 3,347, unchanged from L2c's figure (+149). No
inbound citation quotes the old lead (`grep -rn 'closes once'` matches only l.165). If "plan it
relies on" reads too loose for FR-13, keep "whose changes it relies on" and accept +151. That
call belongs to `teco`.

**m2 (minor, inferred) — "Narrowing" is undefined in every always-loaded or review file.** Step
(4) gates "a narrowing one", but only plan §3.4 defines the term: "a prune or a *Do not report*
addition". Neither root `REVIEW.md` nor root `AGENTS.md` defines it. Defining it inline costs +4
words, which the cap does not allow. Add it to ledger carry-forward (b) (l.39) for L3c's
`teco.md` close step, with this text: "A narrowing change is a prune or a *Do not report*
addition."

**n1 (nit) — The root table row's "live spec:" prefix reads as labeling `DESIGN.md` alone.**
L2c kept the ` · ` separators, so `SERVER.md` and `QUERIES.md` look like separate entries. This
is word-neutral. Replacement for l.90:

```markdown
| `falkor-chat/` | `falkor-chat/README.md` · `falkor-chat/AGENTS.md` · live spec: `falkor-chat/docs/DESIGN.md` (graph), `falkor-chat/docs/SERVER.md` (server process), `falkor-chat/docs/QUERIES.md` |
```

**n2 (nit) — l.277 is 114 characters.** The `Status:`-set repoint pushed it past the file's ~100
wrap. Rewrap l.274–281 with no change to content.

**i1 (info, verified; outside L2c's brief) — The context-file bar command reports the wrong line
numbers.** `awk 'length($0)>700{print FILENAME": "NR}' $(git ls-files '*AGENTS.md')` prints `NR`,
which counts lines across all the files read so far. Today it reports `falkor-chat/AGENTS.md: 770,
771` for the lines that are actually l.87 and l.88. Replacing `NR` with `FNR` fixes it. This is a
separate one-token edit to root `AGENTS.md` ("Context-file convention", "The bar"), for `teco`
to route. It is not an L2c defect.

### Rulings on `cobb`'s three judgment calls

1. **model-bench left out of the root Component-docs table until L6: upheld.** A row naming
   `model-bench/docs/DESIGN.md` and `FORMATS.md` before they exist would be a dangling citation in
   an always-loaded file. Ledger l.39(a) already moves that row to L6. Plan §4's L6 Files column
   still omits root `AGENTS.md`, so the L6 brief must carry the row, and L6 joins the
   shared-file serialization for root `AGENTS.md` (§4 "Shared files").
2. **Close-step details deferred to L3c's `teco.md`: upheld.** §3.1 makes `teco.md` the home of
   "how `teco` runs the close (delegation, line check)". The root bullet keeps the sequence and
   the gate assignments, which FR-2/FR-7/FR-8 need, and the ledger records the hand-off
   (l.39(b)). Add m2's definition of "narrowing" to that carry-forward.
3. **`BACKLOG.md` bullet names only "the component's live spec": accepted, no change.** The
   bullet's exception applies when an item leaves the backlog. A delivered item leaves through a
   feature close, and FR-11 (step "A spec-less component gets a whole-component spec … before its
   first close") guarantees that the spec exists by then. The remaining case is a rejected option
   in a spec-less component that is never delivered. The bullet's "or the requirements doc for a
   scope decision" mostly covers that case, and what is left is rare. A fallback ("`README.md`
   while it has none", +6 words) would break the cap for little gain. If the case recurs, revisit
   it then.

### What's solid

The growth-stops move is clean. The review-only half went to `REVIEW.md` as P7-2 asked, and the
authoring half stays. The new bullets state rules, not history. `falkor-chat/AGENTS.md` l.12 names
the set and cites its home instead of restating it. The README and "Key documents" rewordings drop
"blueprint" and "Cypher patterns" without any claim that the current `DESIGN.md` contradicts.

## Pass 12

**Scope.** Re-check of the L5b fix round on the uncommitted `falkor-chat/docs/SERVER.md` (12,201
words, up from 12,080). It is checked against code at **pinned sha `c0eeb348`**; `git diff --quiet c0eeb348 -- server/ web/ config/ scripts/`
is clean. Each Pass 9 finding was re-executed with network-free probes in the session scratchpad,
each paired with a positive control. Also checked: the fixes against each other, the two changes
beyond the findings, and every `QUERIES.md` citation against the current headings. §1.9 is out of
scope, per the ruling being applied in `DESIGN.md` §6.1.

CPG: considered, not relevant — this pass re-executed probes against unchanged code; the Pass 9 CPG facts (callers of `_schedule_background`, uses of `AGENT_NAME`) are untouched.

**Verdict: approve.** All 11 findings are fixed. No pointer dangles, and nothing new was found.
The AC-3 grep returns 0 (re-run). Numbering is unchanged (§1–§1.12, §2–§2.3).

### Dispositions

- **B1 — fixed.** §1.7 l.627 now reads "the two config files of §1.8", with no
  `WORKFLOW_ENABLED` claim. Re-executed by importing `falkorchat.app` with
  `FALKORCHAT_OPENCODE_CONFIG` unset. With `ENABLE_AGENT=0 WORKFLOW_ENABLED=1` the app builds, so
  `from_env` is never reached. The control, `ENABLE_AGENT=1`, raises `ModelConfigError`.
- **B2 — fixed.** §1.3 l.172 points to §1.4's error map. Its 503 row names exactly start, `/input`
  and `/due`, and says the reads still serve. Re-executed with `create_app` over a stub repo and
  the engine unwired: `POST /workflow-runs` returns 503, while `GET /workflow-runs/r1` returns 404
  and is served.
- **B3 — fixed.** §1.4 Bounds names the bounded path segments. Re-executed with 300-character ids:
  `GET /workflow-runs/{id}` and `DELETE /documents/{id}` return 422, while `GET /messages/{id}` and
  `GET /documents/{id}` are served (404). That matches the new text in both directions.
- **B4 — fixed.** §1.12 l.966–969 covers the unwired case and makes update detection conditional.
  Re-executed `background._schedule_chunk_processing` and `_schedule_update_detection` with
  nothing wired: `[]`. The control, with a worker and repo wired, schedules
  `['progress', '_safe_embed_chunk', '_safe_detect_update']`.
- **M1 — fixed.** The error map lists all four drive-fault types and the
  `"<type>: <message>"` error format, which matches `services.py:2607`. It also has the
  missing-snapshot 404. "Response bodies" gives both shapes, and the ctx paragraph is now scoped to
  `/input`. Re-ran the Pass 9 `/input` probe: `{'status': 'failed', 'ctx': <merge>}`, no `error`.
- **M2 — fixed. Every new pointer resolves to a section that still states the fact:**

  | Pointer | Target |
  |---|---|
  | §1.3 legacy-env line | §1.8 l.741–743 |
  | §1.7 two config files | §1.8 l.734 |
  | §1.7 step budget | §1.4 Response bodies / step-budget paragraph |
  | §1.7 MCP restart | §1.12, "die silently with the process" |
  | §1.3 B2 pointer | §1.4's 503 row |

  None of these targets was rewritten away.
- **M3 — fixed.** The paragraph after the request shape matches `querygen.default_returns`
  (`querygen.py:323–360`), including the identity-lookup abstention and the explicit-`[]` rejection.
- **m1 — fixed.** Ingest cites §14.9; list/delete/deletion cite §14.7; history and
  document-updates cite §14.8, in both §1.4 and §2.2. The "—" note at l.257 is gone, and no "—"
  query cell remains. All 43 `§` citations in the §1.4 and §2.2 query columns were looped against
  `QUERIES.md`'s `##`/`###` headings. Every one resolves, except `§1.3`/`§1.8`, which are
  `SERVER.md`'s own sections. A planted `§14.10` is reported missing, as a positive control.
- **m2 — fixed.** §1.5 now lists "the def list and def structure" (`web/app.js:288`, l.307).
- **n1 — fixed.** §1.12 states about 23,500, with the arithmetic (20 × ⌈500,000/850⌉ × 2).
- **n2 — fixed.** The `AGENT_NAME` row names `seed_demo.sh`, reached via `start_server.sh`.

### Changes beyond the findings

- **§1.7 `options.baseURL` reduced to a pointer — correct.** §1.8 l.744 still states "every
  provider needs an explicit `options.baseURL` — there is no per-npm-package default". §1.7 keeps
  only the testing consequence: re-resolve, don't just parse.
- **Readiness query cell §11.2/§11.5/§12.15 — correct.** `services.check_demo_readiness`
  (`services.py:2140`) reaches `read_def_structure` (§11.2), `read_snapshot_structure` (§11.5) and
  `diff_def_snapshot`, which uses both. It reads `postSuccess` through `read_recent_post_success`
  (§12.15). All three headings exist.

**Growth** is +121 words. B4, M1(c) and M3 are new scope: contracts that were missing. M2 is a
net cut. `SERVER.md` is not a growth-stops kind.

## Pass 10

**Scope.** Gate for unit L5c (`analyst`). It checks the uncommitted rewrite of
`falkor-chat/docs/QUERIES.md` (26,762 words; the baseline `git show c0eeb348:falkor-chat/docs/QUERIES.md`
has 29,382) against the shipped code at **pinned sha `c0eeb348`**. `git status` shows no change under
`falkor-chat/server/` or `falkor-chat/scripts/` since that commit. Read: plan §3.5 and §5;
requirements FR-3, AC-1 and AC-3; root and `falkor-chat/REVIEW.md`; and the writer's inventory
(`l5c.md`), but only after building my own. The inventory checks the text, not just that a query
exists:

- every string literal in `Repository`, compared to the doc's `cypher` blocks with whitespace and
  `//` comments ignored;
- the reverse direction, every doc block against the code literals;
- a per-section word diff against the baseline, with every removed line read in the sections that
  shrank by 60 words or more.

I did not run `test_queries.sh` or pytest, because no code changed and both wipe `reference`. The
only shared-instance commands were `GRAPH.EXPLAIN` and `GRAPH.RO_QUERY` on `falkordb-dev`.

CPG: used cpg_falkorchat — it gives an independent list of the 111 non-test methods that call `query`/`ro_query` (all in `repository.py`), and call sites by `CALL.NAME` for the "no server caller" and library markings.

**Verdict: needs changes.** Coverage is complete and the query text is exact. All four writer claims
hold. History removal is clean, and the numbering is unchanged. No current contract was lost: every
large cut is history, plan provenance or DDL that now lives in `DESIGN.md` (checked), or
transport detail that now lives in `SERVER.md` (checked). One sentence in §14.9 contradicts
observable behavior. Root `REVIEW.md` ranks that as a blocker, and AC-1 at L5s would report it. The
fix is one clause.

### Inventory vs. document (verified)

| Check | Result |
|---|---|
| Query-sending methods (CPG, non-test) | 111. Each is named in the doc except `resolve_member_kinds`, whose query appears verbatim under §2 "Member-kind lookup" (nit N1) |
| Code literals found in a doc block | 117 of 123 (the baseline doc matched 84). For the other 6: the unfiltered `list_documents`/`list_document_updates` branches are described as the same text with the filter dropped, which is exact. `post_agent_answer_first` is given as "the §4 first-message write plus the seed block", which is exact (nit N1). `_PUBLISH_CYPHER`/`_READ_META_CYPHER`/`_READ_TRANSITIONS_CYPHER` with `label=WorkflowDef` match exactly; the `WorkflowDefSnapshot` versions differ only by `d`→`snap` (m1) |
| Doc blocks with no server literal | each one is marked as a library or suite query, a diagnostic, a §4.x backfill script, an elided `…` reply/band block, or a §15.3 template |
| "Library query" / "no server caller" markings | each confirmed to have no non-test caller: the `identity` user MERGE (no server `identity` write), list channel members, the reply write (`REPLY_TO`'s only server literal is the `get_message` read, l.2414), the thread window (`NEXT*1..` appears nowhere in `server/`), the snapshot point lookup, the channel-scoped search, and `create_document`/`create_entity` (no `.create_document(`/`.create_entity(` outside `repository.py`). The seed attributions are wrong (m2) |
| Numbering | the heading diff shows the only numbered additions are §14.7–§14.9 and §15.3. Three unnumbered subsections were added (existence checks, `thread_has_head`, `ping`), and these move no numbers |
| AC-3 sweep, run verbatim | 9 hits, every one `defVersion:`/`currentVersion:` matching `Version:`, ruled not history. `K-030`, `K-033`, `K-064`, `K-076` and `K-090` are each open in `BACKLOG.md`. A wider sweep for "previously", "no longer", "Stage", "fixed" and similar finds only behavioral prose |
| Growth-stops | not applicable: a live spec is outside the bullet. The new text is new scope, catch-up for shipped surfaces (§14.7–§14.9, §15.3), not rework |

### Writer claims (rulings)

1. **§18's five constants are byte-identical**: confirmed. Each `_ENSURE_PARTICIPANT_CYPHER`,
   `_RESET_PARTICIPANT_CYPHER`, `_RESET_ALL_PARTICIPANTS_CYPHER`, `_CURRENT_ORDER_CYPHER` and
   `_ORDER_OWNERSHIP_CYPHER` value (read with `ast.literal_eval`) equals one whole doc block
   exactly. Positive control: changing `orderId`→`orderID` in a scratch copy of the doc makes
   `_CURRENT_ORDER_CYPHER` report False.
2. **`read_thread` returns `coalesce(author.userId, author.agentId) AS authorId`**: confirmed at
   `repository.py:741`, and §4 now states it.
3. **`list_threads` anchors on `Channel.channelId`**: confirmed at `repository.py:283`
   (`MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t:Thread)`, then `ORDER BY t.updatedAt`).
4. **The suite assertion is wrong, not the query**: confirmed with `GRAPH.EXPLAIN` on `ws:acme` and
   `ws:test`, which both carry the `SUPERSEDES` `[matchId, status]` relationship index.
   `MATCH (a)-[r:SUPERSEDES {matchId:'sm1'}]->(b) RETURN r.status` plans as
   `Edge By Index Scan | [r:SUPERSEDES]` alone. `assert_index_scan` (`test_queries.sh:52-62`)
   requires the substring "Node By Index Scan", so l.1695 can never pass. Positive control: the
   labelled form `(a:Document)` adds `Node By Label Scan | (a:Document)`, which the helper's
   `assert_not_contains` half does catch. The doc's §14.8 sentence ("profiles as `Edge By Index
   Scan`") is correct. See M1.

### Findings

**B1 (blocker) — §14.9 says update detection runs after every ingest; the code skips it in two
cases.** Verified. §14.9: "Run by `background._safe_detect_update`, scheduled once per ingested
document after the create." The code schedules it only when `not receipt.get("autoSuperseded")`
(`api.py:191`, `api.py:232`, `mcp.py:323`, `mcp.py:363`). Even then it is a no-op when neither
embed worker nor extraction pipeline is wired: `app.py:399` `repo_for_detection` is `None`, and
`background._schedule_update_detection` returns at l.317-318. You can observe this: an auto-superseded
ingest never yields a `pending` `SUPERSEDES` suggestion (`list_document_updates`). `SERVER.md`
§1.12 states the rule correctly, so §14.9's version is a restated fact that has drifted.
*Fix:* "scheduled after an ingest that the exact tier did not auto-supersede (conditions:
`SERVER.md` §1.12)".

**M1 (major; route to `teco` to file, then `coder`/`tdd-engineer`) — `test_queries.sh` is red at
`c0eeb348`, and no backlog item tracks it.** Verified (claim 4). The writer's isolated run gave
460/461, and the failure is l.1695. A suite that is known red stops gating anything, and
`falkor-chat/AGENTS.md` says to keep it green. `BACKLOG.md` has no "Edge By Index" entry.
*Fix:* add a relationship-index helper (`assert_contains "Edge By Index Scan"` plus the existing
`assert_not_contains "Node By Label Scan"`) and call it at l.1695. The mutant that must turn it red
is the labelled-endpoint query above.

**M2 (major; route to `teco` to file) — the LSH band signal is dead, and `K-064` rests on the
opposite premise.** Verified. Nothing in `server/` or `scripts/` writes `Document.lshBand<i>`; the
only writers are `test_queries.sh:1875-1876` fixtures. A live read finds 0 of 455 `Document`
nodes in `ws:acme`/`ws:agent-team` with `lshBand0`. §14.9 states this correctly. However, the
`K-064` body (`BACKLOG.md` ~l.774) calls the band signal "the primary, fingerprint-based net" and
says "AC-2 held in Stage E's live fixture via the band signal". In production the title lookup is
the only signal. This is outside `QUERIES.md`, so it does not count toward this verdict. It needs
its own backlog item, and `K-064` must be rewritten to stop relying on the band signal.

**m1 (minor) — §11.4 and §11.5 show text the server never sends.** Verified by substitution. The
server formats `_PUBLISH_CYPHER`/`_READ_*_CYPHER` with the root variable `d`; the doc's snapshot
blocks use `snap`. They are semantically identical, but the doc header says "The Cypher here is the
text `repository.py` sends." *Fix:* render the snapshot blocks with `d`, or label them "§11.1/§11.2
text with `WorkflowDefSnapshot` substituted".

**m2 (minor) — "Add user to channel" and "Add agent to a channel" say "Run by
`scripts/seed_demo.sh`".** Verified. `seed_demo.sh:110-117` runs a different query: both
`MERGE (…)-[:MEMBER_OF]->(c)` writes in one statement, with no `role`/`joinedAt`. The documented
text runs only in `test_queries.sh` (l.163, l.493). The server writes `role`/`joinedAt` only
through §18.2. *Fix:* "Suite-only (`seed_demo.sh` writes a role-less variant); the server uses
§18.2."

**m3 (minor) — ruling on §14.4: it is an acceptable index, but it keeps a second copy of a
mapping.** It restates none of `SERVER.md`'s surface (no signatures, caps, status codes or
schedulers), so it does not count as a restatement under the home table. But the route→query
mapping is also `SERVER.md` §1.4's `QUERIES.md` column, and the two drifted during this pass. When
I first read `SERVER.md`, it mapped `POST /documents` to §14.1 and listed
delete/list/history/updates as "—". L5b's concurrent revision has since aligned it with §14.4. A
fact has one home. *Fix:* cut §14.4 to a pointer to the `SERVER.md` §1.4 table (the heading stays
for inbound citations), or drop the column from `SERVER.md`. L5s should check whichever remains.

**N1 (nit).** `resolve_member_kinds` is the only query method not named in its heading. "(§4 m4)"
(a suite message id) and "(OQ3 contract …)" (§9.1, a plan question id) are opaque to a reader of
the spec. Say what they mean, or drop them.

### What's solid

Every query text matches the code. The rewrite's own drift fixes hold: §4's `authorId`, the
`get_message` columns, `search_chunks`' `documentCurrent` filter and aliases, the §14.1/§14.5
"no server caller" notes, and the §14.9 band-signal statement, each checked against code. The cuts
in §12.14–§12.16, §13, §14.1a, §14.3, §14.5, §14.6, §15.1, §15.2, §16 and §17 removed history and
provenance and kept every contract I traced: the zero-row and CAS contracts, the `postedCount`
`int` cast, the profile's "NULL never clears" rule, the timer sweep's due-agnostic read, the
category fallback result shapes, and the cart's no-price lines. The DDL these sections dropped
exists in `DESIGN.md` (§7.1 index table, `SAME_AS` constraint). The MCP thread fan-out note exists
in `SERVER.md`.

### Proposed REVIEW.md lines

None. B1 is already covered by the "a fact has one home" rule and by root `REVIEW.md`'s
contradiction rule. M1's lesson, that a plan-shape assertion must be checked against
`GRAPH.EXPLAIN` of the exact query, is too narrow to change what a reviewer does in general.

## Pass 13

**Scope.** I re-checked the L5a fix round on the uncommitted `falkor-chat/docs/DESIGN.md`, which
grew from 9,731 to 10,097 words, against the code at `c0eeb348`. The code tree is unchanged since
Pass 8. I re-executed every check against the tree: source reads, the verbatim AC-3 grep, and one
live probe on a throwaway graph. Growth-stops does not apply because this is a living spec. The
growth comes from closing Pass 8's omissions and from content moved in from `SERVER.md` §1.9; none
of it is rework padding.

CPG: considered, not relevant — every item here was settled by reading the source and running a
live probe; nothing needed call-graph reach.

**Verdict: approve with suggestions.** Every Pass 8 blocker is fixed and matches the code. The
fixes agree with each other and with `SERVER.md`. The one open item (P8-m5) belongs to `teco`.

### Pass 8 dispositions

- **P8-B1: fixed.** The rewritten §4 Decision paragraph and the new §9 rows ("Publish workflow def:
  write to `reference` only" and "Materialize workflow def … explicit call, §4") agree with each
  other. They also match the code:
  - `materialize_def` raises `WorkflowDefNotFoundError` for an unpublished version, which returns
    404 (`app.py` l.151–155).
  - A run against an unmaterialized version raises `WorkflowRunNotFoundError` (404).
  - `SERVER.md` agrees: §1.4 l.303 for the materialize route, l.331 for the not-found → 404
    mapping, and l.402 for the run start.
- **P8-B2: fixed.** §6.1 l.466–471 matches `_validate_def_spec` l.1799–1860 clause by clause:
  - only a `human`/`wait` step may declare a timer key, and at most one of the two;
  - `0 < waitForSeconds ≤ 15 552 000` (`180*24*3600`, l.193; bool rejected);
  - `waitUntil` is a positive epoch-ms;
  - the exact `ctx.timerFired` guard is required. "Extra fields allowed" is correct, since the
    check reads four keys with `.get()`.
- **P8-m1: fixed, and the overrule is confirmed by execution.**
  - Code: `create_entity_with_auto_match` (l.2146–2148) links each new entity to
    `ORDER BY candidate.createdAt ASC LIMIT 1`. `ingestion.py` l.147–163 creates one `Entity` and
    one `ABOUT` edge per extracted mention, and its own comment says "every mention still gets its
    own Entity node".
  - Probe: calling `Repository.create_entity_with_auto_match` five times on `ws:l5a-probe` with
    the same `("acme corp","Org")` and `createdAt` 1000–1004 gave 5 nodes. The first returned
    `exactMatched=False` and the next four returned `candidateEntityId='e0'`. The only `SAME_AS`
    target was `e0`, with in-degree 4, all `confirmed`. Afterwards the graph was deleted and
    `EXISTS` returned 0.
  - My Pass 8 reading named the right edges but no hub, which leaves the incorrect implication
    that `ABOUT`/`RELATES_TO` degree grows. The new text is the correct one. The stub-repair bound
    (`extraction.py` l.51–62, up to `20 + 2×20`) is correct.
- **P8-m2: fixed.** `{kind:'llm', text, model?}`, the publish line naming the `llm` guard, and
  "only `cmp` is validated at publish" all match the code: `_declared_model_refs`;
  `_validate_def_spec` l.1740–1747; `guards._load_obj` l.591–599, which turns a non-JSON or
  non-object string into `{}` and so reaches `NotImplementedError` at l.271. The sort key
  `(guard == "", order, to)` matches `executor.py` l.1448.
- **P8-m3: fixed.** The l.598–604 sentence matches `_drive_loop` l.724–758: re-run, `StepRun`
  recorded before the branch, the `maxSteps` check, park before re-run, and `done` only with no
  outgoing transition.
- **P8-m4: fixed.** The reset row matches `_RESET_PARTICIPANT_CYPHER`. It names `Customer` (the
  profile), carts, items, orders and `OrderLine`s, and says `User.threadId` is re-pointed.
- **P8-m5: not fixed, and it depends on `teco`.**
  `grep -n -i 'GRAPH.CONFIG' falkor-chat/docs/BACKLOG.md` returns nothing, so §13 l.948–953 still
  calls an untracked item "tracked". Stop condition: that grep returns a K-item, or the clause is
  dropped from §13.
- **Nits: all four fixed.** The QUERIES/plan prefixes are added at l.55, 268–269, 332, 649 and
  727. "unchanged", "New node type" and "not modified" are gone. The §14.6 row carries "TDD build
  order". `CartItem` appears only in the constraint table.
- **Outside this file: not fixed.** `BACKLOG.md` K-033 l.540/543 still quotes `71055f756280`.
  This belongs to `teco`.
- **AC-3:** the verbatim grep gives 0 hits. The K-refs are K-016, K-018, K-029 and K-033, all open.

### New: folded `SERVER.md` §1.9 facts (§6.1 "Prompt", l.421–431)

I verified this against `_assemble_messages` l.1362–1399, `_append_turn` l.233–250 and the tool
loop l.956–1049:
- the speaker is `displayName`, else `authorId`, else `"member"`;
- `human_speaker_label` applies to non-assistant turns only;
- `CONTEXT:` carries compact, sort-keyed JSON and is merged into a trailing user turn;
- the list is built once per node execution and passed whole to each `llm.chat`;
- only the `assistant` and `tool` turns are appended after it;
- `llm.py` adds no messages of its own.

### New nits

- **P13-n1.** At l.428, "the single `system` message leads every call" holds only when
  `systemPrompt` is non-empty: l.1393–1394 adds no system message otherwise. Suggested wording: "the
  `systemPrompt`, when set, is the single leading `system` message".
- **P13-n2.** At §4, "refused (`404 WorkflowRunNotFoundError`)" describes the REST start. An
  `@mention` start (`trigger.py` → `start_workflow_run`) raises the same error, but it is logged
  rather than returned as an HTTP 404. Suggested wording: "refused (`WorkflowRunNotFoundError`;
  404 over REST)".

## Pass 15

**Scope.** Re-gate of L5c's fix round on `falkor-chat/docs/QUERIES.md` (26,767 words, +5), at
pinned **`c0eeb348`** (`git diff --quiet c0eeb348 -- server scripts` exits 0). Also checked: teco's
four `SERVER.md` ingest cells (§1.4 REST l.277–278, §2.2 MCP rows) against `api.py:179/225`,
`mcp.py:316/356` and `background.py:214`. Every check below was re-executed, not read from the
fixer's report.

CPG: considered, not relevant — no code changed since Pass 10's CPG inventory at the same sha. The re-checks are text matches against `repository.py` and reads of the four call sites.

**Verdict: approve with suggestions.** All Pass 10 findings on `QUERIES.md` are fixed. One minor
precision gap remains in teco's `SERVER.md` cells. It belongs to L5b's document and does not
contradict an externally observable contract.

### Dispositions

- **B1 — fixed.** §14.9 (l.2565–2566) now says detection is scheduled after an ingest the exact
  tier did not auto-supersede, and only when a worker is wired, citing `SERVER.md` §1.12. This
  matches `api.py:191/232`, `mcp.py:323/363` and `app.py:399`.
- **m1 — fixed.** I re-ran the Pass 10 substitution check. `_PUBLISH_CYPHER`, `_READ_META_CYPHER`
  and `_READ_TRANSITIONS_CYPHER`, formatted with `WorkflowDefSnapshot`, are now each found
  verbatim (whitespace-insensitive) in the doc: True ×3. In Pass 10 the same check gave False ×3,
  so it has failed and passed on this text. §11.6 keeps `snap`, as `list_snapshots` itself does
  (it matches in the code-to-doc run).
- **m2 — fixed.** Both membership notes (l.86, l.576) now say "Suite-only (`test_queries.sh`);
  `scripts/seed_demo.sh` writes a role-less variant". That matches `seed_demo.sh:110-117`.
- **m3 — fixed.** §14.4 (l.1996) is a one-line pointer to the `SERVER.md` §1.4 `QUERIES.md`
  column. The heading is kept.
- **N1 — fixed.** `resolve_member_kinds` is in the §2 heading (l.145). "(§4 m4)" and "OQ3" no
  longer occur.
- **M1, M2 — not fixed.** They are out of `QUERIES.md`'s scope and routed to `teco` to file. A
  backlog item for each is still owed.
- **Regressions — none.** The code-to-doc match is still 117/123, with the same 6 explained
  residues. §18's five constants are still exact-block equal. The AC-3 grep gives the same 9
  property-name hits.

### New finding

**m4 (minor; L5b's document, `SERVER.md`) — the ingest cells mark only §14.1a as conditional, but
§14.3 `list_document_chunks` is conditional too.** Verified. At all four call sites
(`api.py:178-185`, `api.py:224-231`, `mcp.py:316-321`, `mcp.py:356-362`), `list_document_chunks`
runs only inside `if embed_worker is not None or ingestion_pipeline is not None`. That is the same
guard that leads to `start_document_progress` (`background.py:214-218`, which also needs a `repo`).
The cells read "§14.9, §14.3 (+ §14.1a)", which implies §14.3 runs on every ingest. The pairings
themselves are correct; the batch rows also correctly skip errored items (`receipt["status"] !=
"processing"`). *Fix:* "§14.9; then, when a worker is wired (§1.12), §14.3 `list_document_chunks`
and §14.1a `start_document_progress`" in each of the four cells.

## Pass 14

**Scope.** Unit L3b gate (`analyst`; `cobb` wrote it, and `cobb.md` is `cobb`'s own prompt): the
uncommitted edits to `claude/graph-dba/graph-dba.md`, `claude/cobb/cobb.md`,
`claude/tico/tico.md` and their three `kaizen/history.md` files. Read: plan §3.1, §3.3, §4 L3b
row, §5 AC-6; requirements FR-6, FR-12, FR-13, AC-6; Pass 7 P7-1, P7-2; root `REVIEW.md`; root
`AGENTS.md` live-spec bullet (l.160–164); the full `tico.md`, and the review-shaped parts of
`cobb.md` and `graph-dba.md`.

CPG: not applicable — the artifact is agent-prompt text, no code change.

**Verdict: approve with suggestions.** One minor finding and one nit. No blocker or major.

### Checks run

- **No permission, tool, frontmatter or hook change** — verified. Each file's frontmatter block
  (`graph-dba` and `cobb` to l.11, `tico` to l.16) is byte-identical to `HEAD`, and every diff
  hunk starts below it. `git diff --stat` touches no `.claude/`, hook or `claude/scripts/` file.
  `cobb.md` gains exactly one line: How-you-work step 6.
- **AC-6 loop, run verbatim from the repo root** — it prints only `claude/teco/teco.md`, which
  is L3c's. L3a's four files already carry the pointer in the working tree, so they no longer
  show up. Positive control: a scratch copy of `tico.md` with `REVIEW.md` renamed is listed. The
  check can go red.
- **Pointer fit.** `cobb`: step 6 covers its gates of `REVIEW.md` files and agent work, its
  prompt-lint reviews and its team certification. `graph-dba`: step 9 sits in the How-you-work
  list, which is read whole as the system prompt, so it is read before any review. The list order
  carries no sequencing, so the placement is fine.
- **`tico` scoping is complete.** I read every review, gate and verdict line (`grep -n -i`).
  `tico` routes gates (l.57) and reads `analyst`'s verdict before the Ready-for-design flip
  (l.110, l.169). For a manual it calls `qa-engineer`/`analyst` and does not self-certify (l.163).
  It checks that a consult deliverable fits (l.44), which is integration, not a verdict. Its only
  verdicts of its own are the Integrating check, and an `accepted` on a trivial unit with no gate
  (l.57–58). The pointer covers both. See P14-n1 for the wording.
- **P7-2 repoint** — fixed. `tico.md` l.110 reads "growth-stops check (root `REVIEW.md`)", which
  resolves to `REVIEW.md`'s bold **Growth-stops check.** `tico.md` l.66 and `graph-dba.md` l.64
  still cite the growth-stops *bullet* for authoring rules, and that bullet still exists.
- **P7-1 duplicates** — none in these three files. A grep for growth-stops, mutant, unfiltered,
  positive-control and evidence-bar phrasing finds no reviewer duplicate in `cobb.md`,
  `graph-dba.md` or `tico.md`. P7-1's table has no L3b row.
- **Kaizen** — each entry is the `agent-maintenance` template shape (What / Why / Plan items) and
  sits at the top, most recent first. The three L3b entries are the only ones citing L3b; L3a's
  four entries cite L3a.
- **Growth.** About 180 words across the three prompts, all new scope (FR-6, FR-12) plus one
  repoint. Nothing is restated.

### Findings

**P14-m1 (minor) — `graph-dba`'s delta-plan pointer paraphrases half of the rule it cites.**
Verified by reading the cited bullet and the L3a sibling. `graph-dba.md` l.64 says only "states
only the change against it". The cited root `AGENTS.md` bullet (l.163–164) also requires "citing
the affected sections and any open feature's plan it relies on". Those are FR-12's second half
and all of FR-13. A paraphrase that reads as complete invites a `-graph.md` note with no section
citations, and that is a finding under root `REVIEW.md` *Always check* 2. L3a's
`data-scientist.md` l.81 states the whole rule, so the two pointers to one rule differ. Replace the
added sentence with the `data-scientist` wording:

```markdown
For a component with a live spec, the document states only the change against it, citing the affected spec sections and any open feature's plan it relies on (root `AGENTS.md`, live-spec bullet).
```

Its reach is one sentence. The kaizen entry's "What" should then say "… states only the change
against it, citing affected spec sections and any open feature's plan".

**P14-n1 (nit) — "line check" is `teco` vocabulary, and nothing in `tico.md` defines it.**
Verified: `grep -n 'line check' claude/tico/tico.md` hits only the new sentence. `tico`'s own
check is the done-condition check in the sentence just before it, so the reader has to infer the
link. Suggested replacement for the added sentence:

```markdown
Before that check, and before any gate verdict of your own, read and apply the root `REVIEW.md` and the `REVIEW.md` of each component the work touches.
```

If you take it, edit the matching words in the `tico` kaizen "What" line too.

### Proposed REVIEW.md lines

None. Neither finding is a recurring class.

## Pass 16

**Scope.** Unit L3a gate (`analyst`), except `analyst.md`, which `security-expert` gates. The
artifact is `cobb`'s uncommitted edits to `claude/qa-engineer/qa-engineer.md`,
`claude/security-expert/security-expert.md`, `claude/data-scientist/data-scientist.md` and their
three `kaizen/history.md` files. Read: plan §3.1, §3.3, §4 L3a row, §5 AC-6; requirements FR-6,
FR-12, FR-13, AC-6; Pass 6's duplicate list; Pass 7 P7-1 and P7-2; root `REVIEW.md`; root
`AGENTS.md` live-spec bullet (l.160–164); all of `qa-engineer.md`; the review-shaped parts of the
other two prompts; the three agents' `kaizen/plan.md`.

CPG: not applicable — the artifact is agent-prompt text, with no code change.

**Verdict: approve with suggestions.** There are two minor findings and one nit, and no blocker
or major. The growth-stops check does not apply: agent prompts and kaizen files are outside the
bullet's `requirements/`/`plans/`/`reviews/` scope.

### Checks run

- **No frontmatter, hook or tool-grant change.** Verified. Each prompt's frontmatter block is
  byte-identical to `HEAD` (`diff` of the extracted blocks gives no output). The only changed paths
  in `claude/` and `.claude/` are prompt and `kaizen/history.md` files, none of them hooks.
- **AC-6 loop, run verbatim from the repo root.** It prints only `claude/teco/teco.md`, which
  belongs to L3c. Each of the three L3a prompts had 0 `REVIEW.md` matches at `HEAD` and has 1 now,
  so the new pointer is what clears it, not some earlier mention. Positive control: `teco.md` is
  listed, so the check can go red.
- **Nothing lost in the merges.** The old `security-expert.md` l.72 and `data-scientist.md` l.82
  lines ran the check "when the reviewed artifact is a document [the growth-stops bullet]
  covers". Root `REVIEW.md` *Always check* 1 keeps that condition ("a revision of a document that
  root `AGENTS.md`'s growth-stops bullet covers") and adds the action the old lines left to the
  bullet: say whether growth is new scope or rework, and report rework growth, restated history
  and restated facts. Narrowing to "a revision" is not a loss, because a first version has no
  growth to check. Both merges match P7-1's "becomes the `REVIEW.md` pointer" ruling.
- **Duplicates.** Verified against Pass 6's list and P7-1's table. `qa-engineer.md` l.73 ("Evidence
  over assertion") and `security-expert.md` l.89 ("Evidence over vibes") are partial overlaps that
  are lens-specific and stay. `data-scientist.md` has no other duplicate. The kaizen entries
  say the same.
- **Delta-plan pointer against FR-12/FR-13.** `data-scientist.md` l.81 matches its home, the root
  `AGENTS.md` live-spec bullet (l.163–164): "states only the change … citing the affected
  sections and any open feature's plan it relies on". The citation "root `AGENTS.md`, live-spec
  bullet" resolves to the bold lead **A live spec is a component's current behavior**. FR-12's
  review half reaches `data-scientist` through the `REVIEW.md` pointer on l.82. Whether "relies
  on" fully renders FR-13's "behavior that an open feature changes" is a question about the home
  sentence, which L2c gated, not about this pointer.
- **Kaizen entries.** Verified. Each is newest-first under the H1/blurb, in the
  What / Why / Plan items shape of the 2026-10-29 entries. The `wc -w` deltas are accurate
  (2128→2150, 2536→2541, 2101→2141, re-counted against `HEAD`). The cited Pass 7 IDs and plan
  sections exist. "Gated by `analyst`" becomes true when this pass approves.

### Findings

**P16-1 (minor) — `qa-engineer` reads `REVIEW.md` only after it has executed, so it cannot
apply the execution-shaped checks.** Verified by reading the full prompt. The bullet is at
`qa-engineer.md` l.61, in Phase 4 REPORT, which comes after Phase 3 EXECUTE. Unlike
`graph-dba`'s How-you-work list (Pass 14), the four phases are a sequence. The bullet is read
before the verdict, so FR-6's "reads" is met. Its "applies" is not. Several lines are things a
tester does while running, not while writing:

- root *Always check* 4: name the mutant or positive control that turns a check red;
- *Always check* 5: run a prescribed check command verbatim, sweep unfiltered;
- `falkor-chat/REVIEW.md`'s "green run after the last seed is the evidence".

At Phase 4 these mean re-execution, or they get skipped. The bullet also sits in a list headed
"covering:", which lists report contents, but it is an action. `analyst` places its own pointer
in its read step (`analyst.md` l.45).

Fix: delete l.61 and add this bullet after Phase 1's "Read the sources of truth" bullet
(l.36):

```markdown
- **Read the review instructions** — root `REVIEW.md` and the `REVIEW.md` of every component the tested work touches; apply them while you execute and before you write any review or gate verdict.
```

The `qa-engineer` kaizen "What" line then reads "Phase 1 (REASON) gained one bullet: …" instead
of "Phase 4 (REPORT) gained one bullet: …".

**P16-2 (minor) — Two kaizen parking-lot notes still track a pointer that L3a removed, but the
history entries say "Plan items: none".** Verified (`grep -n growth-stops
claude/*/kaizen/plan.md`). `claude/security-expert/kaizen/plan.md` l.17 and
`claude/data-scientist/kaizen/plan.md` l.84 each say "The U2 pointer … if not, sharpen the
clause rather than restating the root bullet". L3a replaced that U2 sentence with the
`REVIEW.md` pointer, and the check is no longer in the root bullet. The note still has a job,
which is to confirm the check is run on first live use. Its target and remedy are now wrong, and
`claude/AGENTS.md` l.308 puts `kaizen/{plan,history}.md` in the same change. Replace both
notes, in place, with:

```markdown
- **Growth-stops check: confirm on first live use (2026-10-29).** This agent reaches the check through its root `REVIEW.md` pointer, and it is judgment-enforced (`docs/plans/document-size-discipline.md` D6). On the first in-scope revision or gate, check this agent actually ran it; if not, sharpen the pointer rather than restating `REVIEW.md`.
```

Then change each of the two history entries' "Plan items: none." to "Plan items: the
growth-stops parking-lot note repointed at the `REVIEW.md` pointer." The `qa-engineer` plan has
no such note.

**P16-3 (nit) — The `security-expert` and `data-scientist` pointers sit in deliverable text, not
in the read/investigate step.** Inferred: placement only. Both agents are static reviewers whose
lists are not strict sequences, so reading at verdict time still lets them apply the checks. That
is why this is a nit and P16-1 is minor. For the same consistency with `analyst.md` l.45, the
pointer sentence could move. In `security-expert.md`, move it to the end of How-you-work step 2
(l.65). In `data-scientist.md`, move it to the end of step 2 (l.72), changing "Before any review
or gate verdict" to "Before any methodology review or gate verdict". Take or leave.

### Proposed REVIEW.md lines

None. P16-1 is a placement in one prompt, not a recurring review-finding class.

## Pass 17

**Scope.** Unit L5s (`analyst`): the set-level AC-1 gate on falkor-chat's live spec. It covers
`falkor-chat/docs/DESIGN.md`, `SERVER.md` and `QUERIES.md` together, as committed at **`9d240859`**
(the working tree matches), against shipped code at **`c0eeb348`**. Since that commit nothing has
changed under `falkor-chat/server`, `scripts`, `config` or `web`. I read plan §3.5 and §5,
requirements FR-3, AC-1 and AC-3, root and `falkor-chat/REVIEW.md`, and Passes 8–15. I built the
inventory from the code. I did not take it from the L5 inventories. I did not run pytest or
`test_queries.sh`. Every probe below is offline: the server venv imports the modules directly with
stub repositories, and nothing touches FalkorDB. Growth-stops does not apply, because these are
live specs.

CPG: used cpg_falkorchat — to inventory the non-test methods of every module outside the §3.5 home table, and the production call sites of `call_with_retry`, `split_into_chunks`, `compute_line_total`, `evaluate_guard`/`validate_cmp`, `find_fuzzy_candidates` and `minhash_signature`, which home each surface; `scripts/` was read directly.

**Verdict: needs changes. AC-1 is not met.** The three documents agree on almost everything, and
every cross-document citation resolves. Two contradictions remain, though, and both are visible
only across documents or from outside the L5 inventories:

- `DESIGN.md` states that a workspace's vector dimension can never change. A shipped script
  changes it.
- `DESIGN.md` §9 holds a third copy of the ingest-scheduling rule that Pass 9 B4 and Pass 10 B1
  fixed in the other two documents. That copy still reads as unconditional.

Four catch-all surfaces are under-stated. Each one goes to the writer of its home document, per
the plan's L5s rule.

### Set checks (verified)

| Check | Result |
|---|---|
| Cross-document `§` citations | I ran a script over all three files that extracts each `DESIGN`/`SERVER`/`QUERIES` `§` citation and looks for a matching `##`–`####` heading. 85 citations checked. The only misses are the 12 `DESIGN §14.x`/`§15.x` keys in DESIGN §14's own map, which is the positive control (those numbers are, correctly, not DESIGN headings). |
| SERVER §1.4/§2.2 query cells → QUERIES | 105 `§` references checked. Every one resolves; the only misses are SERVER's own §1.3/§1.8/§1.12. Pass 15 m4 is fixed: all four ingest cells now say "when a worker is wired". I spot-read the cited sections for the storefront state, readiness and ingest rows, and each states the fact the cell claims. |
| DESIGN §14 map | All 13 rows resolve to `SERVER.md` headings. §1.6 is a pointer, which is acceptable for the one docstring-free `§14.6` row. |
| AC-3 (§5 grep, verbatim, all three files) | DESIGN 0 hits, SERVER 0 hits. QUERIES has 9 hits, all of them the property names `defVersion:`/`currentVersion:`, ruled not history. The K-refs are K-016, -018, -029, -030, -033, -064, -074, -076, -080 and -090. Each has an open `###` heading in `BACKLOG.md`. |
| Inbound code citations | 42 `DESIGN §` citations in `server/`, `web/`, `scripts/` and `tests/`. Each resolves to a section that states what the citing comment expects, except two (m3). The code has no `SERVER §`/`QUERIES §` citations to dangle: they all name QUERIES sections, and the cell loop covers those sections. |

### Findings

**B1 (blocker, verified; `DESIGN.md`): the spec says a workspace's vector dimension can never
change, but a shipped tool changes it.**
- DESIGN l.69 says "fixed at workspace creation". l.765 says "fixed per workspace at bootstrap".
  l.919 says "Fixed at workspace creation, never after". l.79 says an embedding upgrade is
  "Re-embed only; no schema change".
- `scripts/migrate_embeddings.sh` → `embedding_migration.migrate` (`09e30282`, HISTORY
  2026-09-20) re-embeds every `Message`/`Chunk` and stamps `n.embeddingModel` (l.183–184).
  It then drops and recreates both vector indexes at the new dimension (l.207–211) and moves
  `embeddingModelOverride`. Its preconditions: stop that workspace's server first, and restart
  it afterwards to clear the dimension cache.
- Neither the operation nor the `Message.embeddingModel`/`Chunk.embeddingModel` property
  appears anywhere in the spec set.
- The same gap covers workspace creation. `create_workspace.sh` runs `bootstrap_schema.sh` and
  then pins `WorkspaceConfig.embeddingModelOverride` to the current default.
  `start_server.sh:197` and `start_demo.sh:270` call it. DESIGN §7.1 names only `bootstrap_schema.sh`.
- `backfill_document_current.sh` is also unmentioned. It stamps `currentVersion`/`documentCurrent`
  where they are absent, and default chunk search returns nothing until it has run.
- Why the scoped checks missed it: the migration landed **before** DESIGN's drift baseline
  (`git merge-base --is-ancestor 09e30282 7a270329` → true). The writer's `baseline..pin` log
  window cannot see it, and the L5a inventory read only `bootstrap_schema.sh` from `scripts/`.

*Fix (DESIGN §1.3, §5.1, §7.1, §11):* a dimension is chosen at creation and changed only by
`migrate_embeddings.sh`. The script re-embeds, records `embeddingModel` per row, drops and
recreates both vector indexes, and moves the override. The workspace's server is stopped
before it runs and restarted after. Add `embeddingModel` to the `Message`/`Chunk` schema. State
that `create_workspace.sh` is how a served workspace is made (bootstrap + pin), and state the
`backfill_document_current.sh` precondition.

**B2 (blocker, verified; `DESIGN.md` §9): the third copy of the ingest-scheduling rule still
reads as unconditional.** Pass 9 B4 and Pass 10 B1 fixed this rule in `SERVER.md` §1.12 and
`QUERIES.md` §14.9, and both now say "only when a worker is wired". DESIGN §9 still reads as if
it always applies:

| Row | Text | What the code does |
|---|---|---|
| "Document background jobs" (l.836) | "per chunk, one embed job and one extract job … `ready` when the count reaches 0" | With neither worker wired (the default), `api.py:178` and `mcp.py:316` skip `_schedule_chunk_processing`. `start_document_progress` never runs, so the document stays `processing` permanently. Each job also exists only when its own worker is wired (`background.py` `total_jobs`). |
| "Update detection" (l.837) | "background job after a non-auto-superseded ingest" | `_schedule_update_detection` returns when `repo_for_detection` is `None` (`app.py:399`, `background.py:317`). |

*Fix:* keep only the write pattern and the status and retry semantics in these two rows. Replace
the "when" with "scheduled per `SERVER.md` §1.12", which is the home of the scheduling rule. The
proposed rule below covers this class: one fact, three copies, and the copy left unchecked drifts.

**M1 (major, verified by execution; `DESIGN.md` §6.1): guard semantics are named but not
stated.** A def author cannot predict a publish rejection or a run's path from §6.1. It says
"whitelisted ops … two path roots … depth/width/node caps … structured boolean verdict" but
never gives the ops, caps, path rules or verdict policy. The baseline text did not state them
either. *Contract text to add* (probe output in the Appendix):
- The ops are `eq ne lt le gt ge in contains exists truthy`.
- `exists`/`truthy` take no `value`; every other op requires one.
- `eq` uses plain JSON equality with no coercion (`"1"≠1`).
- An ordering op on incomparable types is `false`.
- Caps: depth 5, nodes 32, width 8.
- `not` takes exactly one child. `all([])` is `true` and `any([])` is `false`. A nested child
  that carries `op` may omit `kind`.
- Path `output` is the raw step output. `output.<k>…` reads the step output parsed as a JSON
  object. `ctx.<k>…` reads the run ctx. A bare `ctx` and empty segments are rejected, and there
  is no list indexing.
- On a missing path, `ne` is `false` but `not(eq)` is `true`.
- `llm` guard:
  - The judge sees the node's `understanding` object (from the step output, else
    `ctx.understanding`). Only when that is empty does it see the last 6 thread turns, ≤ 400
    chars each. The prompt is capped at 6000 chars, and the oldest turns are dropped first.
  - The reply must be one JSON object `{decision, rationale}` that owns its lines.
  - Only a literal `true` advances. An unparseable reply, a non-bool `decision`, or a rationale
    asserting an un-negated deficiency is `false` (the bias is to suspend).
  - With no judge wired, the run fails.
- A null guard behaves like `""`.

**M2 (major, verified by execution; `SERVER.md` §1.11, cited from the `/shop/api/state` row):
the cart-pricing contract is missing, and one value contradicts the code.** `pricing.py` plus
`services._priced_cart_lines` (l.2998–3035):
- A line's `lineTotal` is `price × quantity`, unrounded (3 × 0.1 → `0.30000000000000004`).
- `total` is the sum of the line totals.
- A cart line whose `productId` no longer resolves in `reference` is **omitted** from `items`
  and `total`, and is **not** snapshotted by `place_order`. The `place_order` Cypher
  (`repository.py:3721ff.`) nevertheless `DETACH DELETE`s **every** `CartItem`, so that line
  disappears silently.
- A cart holding only such lines reads as empty, and `place_order` returns the empty-cart string.
- l.877's empty-cart value `{"items": [], "total": 0.0}` is wrong. The tool returns
  `{"items": [], "total": 0}`, because `sum([])` is an `int`. The `0.0` comes from the tool's
  docstring and the test stub (`test_tools.py:96`), not from the code.

*Fix:* one paragraph under the tool table. Cite it from the `/shop/api/state` row, and correct
`0.0` → `0`.

**M3 (major, verified by reading; `SERVER.md` §1.8): the provider wire contract stops at
resolution.** §1.8 covers which model is called but not what is sent or which replies count as a
tool call. That decides whether a candidate model can drive an agent node at all
(`llm.py:134–178`, `_parse_content_tool_calls` l.296ff.). *Contract text to add:*
- `POST {base}/chat/completions` with `model`, `messages`, `tools` (on a tool turn) and the
  overlay's per-model params.
- A tool call is read from `choices[0].message.tool_calls`. When that field is absent, the
  server checks `content` in this order:
  1. a JSON object `{name, arguments}`, `{action, args}`, `{function:{…}}` or `{tool_calls:[…]}`;
  2. bare `name({...})` call syntax.
  Anything else is plain text.
- A missing `choices` is `ProviderCallError(kind="response_shape")`.

Cite this paragraph from the M1 judge sentence instead of repeating it.

**M4 (major, verified by execution; `DESIGN.md` §5.1/§9): the ingestion pipeline's observable
parameters are missing.** *Contract text to add:*
- `Entity.type` is the closed set `Person, Organization, Location, Product, Event, Concept,
  Other`. An out-of-set or missing type is stored as `Other`, and so is every stub entity
  synthesized for a relationship endpoint. This is the value set of `query_graph_data`'s
  `Entity.type` facet, `SERVER.md` §1.11.
- Chunks are split on paragraph, then sentence (`. ! ?` or newline), then a hard cut at 1000
  chars. Every chunk after the first carries the previous chunk's last 150 chars, so a chunk can
  reach 1150 chars (probe: 2,500 chars → 1000/1150/650).
- A fuzzy `SAME_AS` is type-filtered: it links to the top-ranked of ≤ 5 full-text candidates
  with `confidence` = full-text score and technique `fuzzy_fulltext`. An exact match uses
  `exact_normalized_name_type`.
- A `SUPERSEDES` suggestion's `confidence` is the Jaccard ratio over 5-word shingles of
  case-folded, whitespace-collapsed text. `textNormalizedHash` is a SHA-256 of that same
  normalized text.
- Neither `fuzzy_fulltext` nor the shingle size is stated anywhere in the set.

**m1 (minor, verified; `QUERIES.md` §1): "The app calls it (or its equivalent) once when a new
workspace is created" is false.** No server module calls `bootstrap_schema.sh`; the start scripts
call `create_workspace.sh`. *Fix:* reduce the line to a pointer to DESIGN §7.1 (B1's text).

**m2 (minor, verified; `QUERIES.md`): it restates DESIGN facts, and one copy has already gone
incomplete.**
- §12.5 l.1393–1399 restates DESIGN §6.2's `maxSteps` tripwire note almost verbatim. Both
  copies match the code today.
- §14 "Versioning flags" (l.1830–1833) restates DESIGN §5.1's version-lifecycle property.
- §14.6 l.2278, §15 l.2610–2617 and §16 l.2760–2788 restate DESIGN §5.1/§7.1 schema and index
  lists. §16's block gives `Customer {customerId, createdAt}` and drops the profile fields that
  DESIGN §5.1 lists.

*Fix:* replace each with a `DESIGN.md` §5.1/§6.2/§7.1 citation. Query-local notes stay.

**m3 (minor, verified; owner `coder` via `teco`, not a spec writer): two inbound code citations
dangle.**
- `scripts/bootstrap_schema.sh:378` cites "still the DESIGN §13 open question" for the
  1536 default. DESIGN §13 has no such question.
- `scripts/test_queries.sh:346` cites DESIGN §1.3 for "Score is cosine distance". That fact now
  lives in DESIGN §7.1/§8.

*Fix:* repoint both comments (§7.1 and §8). The spec should not regain the text.

**Nits.** (n1) DESIGN §9's retry clause should add "a timeout is not retried"
(`retry.is_transient`, `retry_timeouts=False`). (n2) The `SERVER.md` l.878 `add_to_cart` cell
restates DESIGN §6.1's grounding hold in a clause. It is consistent, so cite §6.1 instead.

**Carried, still open (no new finding):** P8-m5. `grep -n -i 'GRAPH.CONFIG'
falkor-chat/docs/BACKLOG.md` returns nothing, so DESIGN §13 still calls an untracked item
"tracked" (`teco`). Pass 10 M1/M2 are still owed backlog items (`teco`).

### Catch-all homing table

| Surface (code) | Home | Stated adequately? |
|---|---|---|
| `guards.py` + `app._LlmGuardJudge` (guard semantics, verdict policy) | `DESIGN.md` §6.1 | No: M1 |
| `retry.py` (provider retry classifier and policy) | `DESIGN.md` §9 "Document background jobs" | Yes; n1 |
| `chunking.py` | `DESIGN.md` §9 "Ingest document" | No: M4 |
| `extraction.py` (taxonomy, caps, stub repair) | `DESIGN.md` §5.1 / §5.4 | Caps yes, taxonomy no: M4 |
| `fusion.py`, `ingestion.py` | `DESIGN.md` §9 "Extract entities" | Partly: M4 |
| `update_detection.py`, `background._safe_detect_update` | `DESIGN.md` §9 "Update detection" | Partly: M4, B2 |
| `background.py` scheduling | `SERVER.md` §1.12 | Yes; DESIGN copy: B2 |
| Retrieval (`services.hybrid_search`/`search_documents`, retrieve tool) | shape `DESIGN.md` §8; surfaces `SERVER.md` §1.11/§1.4/§2.2; text `QUERIES.md` §6/§14.3 | Yes |
| `llm.py`, `transport.py` (wire request, reply parsing, fallback) | `SERVER.md` §1.8 | Fallback yes, wire/reply no: M3 |
| `pricing.py` + `services._priced_cart_lines` | `SERVER.md` §1.11 | No: M2 |
| `embedding.py` (dimension guard) | `SERVER.md` §1.8 | Yes |
| `responder.py`, `trigger.py` | `SERVER.md` §1.12 | Yes (Pass 9) |
| `logformat.py`, `config/logging.json` | `SERVER.md` §1.10 | Yes |
| `scripts/create_workspace.sh`, `migrate_embeddings.sh` + `embedding_migration.py`, `backfill_document_current.sh` | `DESIGN.md` §7.1 / §5.1 | No: B1 |
| Other `scripts/` (seed, verify, start, load, eval corpus) | operator tooling: `falkor-chat/README.md`/`AGENTS.md`, outside the FR-4 set | Ruled out of spec scope (open question 1) |
| `proof_defs.py` def content (`salesperson@v9`, `order-fulfillment@v1`) | — | Open question 1 |

### What's solid

Every query text, route, tool and config key that the L5 checks inventoried still matches. All
190 cross-document and query-cell citations resolve. The set has no contradiction between
SERVER and QUERIES. `DESIGN.md` §8 (retrieval shape), §9's retry policy and the dormant-schema
notes match the code.

### Open questions

1. **Is workflow def content and operator tooling spec surface?** My ruling is no. The FR-3 spec
   covers engine semantics, the schema and the graph-writing scripts (B1). Seeded def content
   (`proof_defs.py`: prompts, tool grants, `maxSteps` 40/16/24) and the seed, verify and start
   scripts stay in code and `README.md`. If the stakeholder rules def content in, it needs a home
   (`SERVER.md` §1.11 already tracks grants) and a new L5 pass.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: Pass 9's proposed line, now recurring across
three passes (Pass 9 B4, Pass 10 B1, this Pass's B2: the same fact drifted in each document of
the set). I endorse Pass 9's addition as written, with no new line. Its "two sections of the spec
set" already spans documents.

Target: root `REVIEW.md`, *Always check*. Source: B1. The drift predated the writer's git-log
window, and the scoped inventory skipped `scripts/`.

```markdown
- A whole-component spec check (a catch-up or a first spec) builds its inventory from every
  shipped file, scripts and operator tooling included. It never builds it from a git-log window:
  a document edited after a feature landed can still omit that feature.
```

No prune. No test, CI job or hook checks spec coverage.

### Appendix: offline probes (server venv, `-I`, stub repositories, no FalkorDB)

```
ops ['contains','eq','exists','ge','gt','in','le','lt','ne','truthy']
validate_cmp: regex→reject  truthy w/o value→ok  eq w/o value→reject  bare ctx→reject
              bare output→ok  depth 6→reject  width 9→reject
evaluate: ne on missing→False  not(eq) on missing→True  all[]→True any[]→False
          eq "1" vs 1→False  output.x on '{"x":2}'→True
_coerce_verdict: "true"(str)→False  "not enough information"→False  "no more information is needed"→True
extract(type "Company", rel to unlisted "Bob") → [{Acme, Other}, {Bob, Other}]
split_into_chunks('a'*2500) → [1000, 1150, 650]
get_cart (a@0.1×3 + delisted 'gone') → items [a], total 0.30000000000000004
place_order → lines [a]   (Cypher deletes every CartItem: repository.py place_order)
cart of only 'gone' → {'items': [], 'total': 0}; place_order → None
ViewCartTool on empty cart → {"items": [], "total": 0}
```

Positive controls: each probe yields both outcomes on the same function. For example,
`_coerce_verdict` returns True for the negated cue and False for the bare cue, and `validate_cmp`
returns ok and reject. The citation scripts report the 12 map keys as missing.

## Pass 18

**Scope.** Unit L3c gate (`analyst`). The artifact is `cobb`'s uncommitted edits to
`claude/teco/teco.md`, `claude/architect/architect.md`, `claude/README.md`,
`claude/teco/kaizen/{history,plan}.md` and `claude/architect/kaizen/history.md`. Read: plan §3.1,
§3.4, §4 L3c row, §5; requirements FR-6–FR-8, FR-12, AC-6, AC-9; root `AGENTS.md` l.160–173;
root, `falkor-chat/` and `model-bench/` `REVIEW.md`; Passes 11 (m2, rulings) and 16;
`docs/reviews/live-specs-security.md` S5; ledger l.17, l.35–45. I did not write this text.

CPG: not applicable — the artifact is agent-prompt and catalog text, with no code change.

**Verdict: approve with suggestions.** There is no blocker or major, one minor for `teco` and one
nit. The growth-stops check does not apply, because prompts, catalogs and kaizen files are outside
the bullet's scope.

### Checks run

- **AC-9 sweep, verbatim from the repo root:** 91 hits, the same as `cobb`'s count. Rulings on the
  six files (the kaizen files fall outside the pathspecs):
  - `teco.md` l.3, l.92, l.138 and l.155 each carry a live-spec carve-out. These four are the
    plan's named sites, and they are fixed.
  - `teco.md` l.53 ("tests alongside"), l.168 and l.181 ("alongside") are no-change.
  - `claude/README.md` l.7: the "doc updates part of every unit's done-condition" clause is now
    followed by the live-spec exception, so it is fixed. Its "alongside `tico`" phrase is
    no-change.
  - `claude/README.md` l.8, l.14 and l.17 ("alongside") and l.185 (the catalog "same change"
    rule) are no-change.
  - `architect.md` has no hits.

  `cobb`'s rulings agree. I also grepped the implementer prompts (`coder`, `tdd-engineer`,
  `frontend-engineer`, `graph-dba`) for a doc-update instruction the pattern would miss and
  found none.
- **AC-6 loop:** prints nothing. Positive control: `teco.md` had 0 `REVIEW.md` matches at `HEAD`
  and has 4 now, so at `HEAD` the loop would have listed it.
- **Bar `awk` (`FNR`):** it prints `falkor-chat/AGENTS.md: 87` and `88`. Both lines predate this
  work (Pass 11 i1), and L3c does not touch that file. This is a smell, not a gate.
- **Word counts:** re-counted against `HEAD`. They are 8,296→8,585, 1,582→1,614 and
  5,587→5,641, as reported.
- **Frontmatter:** `yaml.safe_load` of the edited block gives the same five keys. Only
  `description` differs from `HEAD`. `tools`, `permissionMode` and `hooks` are byte-identical.
- **architect pointer:** the sentence matches `graph-dba.md`/`data-scientist.md` word for word,
  apart from the document noun. It resolves to root l.160 ("A live spec is …").

### Rulings on the five questions

1. **Commissioned, nothing beyond it.** Each L3c-row item is present:
   - the carve-outs at l.3, l.92, l.138 and l.155;
   - the curation sentence (l.152);
   - the scan flagging spec sections (l.154);
   - the Feature close bullet, carrying ledger carry-forward (b);
   - the pointer (l.137), which covers line checks as "any verdict of your own";
   - the README row and the `architect.md` pointer.

   Two edits are outside the row: the l.144 growth-stops repoint, and the S5 sentences. Each is
   brief-commissioned, by ledger l.35 (P7-2) and l.45 (S5) respectively.
2. **The bullet cites the root and does not restate it.**
   - Step (2) adds only the delegation default, which matches plan §3.4 step 2.
   - Step (3) adds only the slug-less home, which matches §3.4 step 3.
   - Step (4) adds the integrity check. Its "each prune's cited enforcer" equals §3.4's "each
     cited enforcer", because only prunes cite one (root `REVIEW.md`, last *Always check*).
   - The narrowing bullet refers to "step (4)'s gate" and does not restate who gates.
   - Steps (1) and (5), the "needs changes" block and the spec-less clause are left to the root.

   There is no contradiction in any step.
3. **S5 rulings.**
   - **(a) is sound.** I ran a coverage probe over the axes a `REVIEW.md` change varies on: the
     four headings plus "a new heading", against add/prune and widen/narrow.
     - Every prune qualifies.
     - So does every *Do not report* addition.
     - So do a severity cap or downgrade, a scope limiter or exemption on an existing check, and
       an evidence relaxation.
     - A reword is a prune plus an addition, so it qualifies.
     - Additions that widen do not qualify.
     - A line whose effect is unclear (for example, one that redefines a term) is caught by "when
       unsure".

     It does not contradict the root, which uses "narrowing" without defining it. It supersedes
     Pass 11 m2's closed set and does not contradict it: both of m2's members are kept as "always
     qualify", and the additions are exactly S5's three bypass cases. The cost is more `cobb`
     gates, and FR-8 does not require those lines to go ungated. The home is right. `teco` is the
     only agent that classifies and routes at step (4), so the definition belongs where the
     router reads it. Root `AGENTS.md` cannot take the words, because L2c is at its cap (Pass 11
     m2). Plan §3.4's parenthetical, "(a prune or a *Do not report* addition)", now states less
     than `teco.md` does. The plan has been executed against and will be archived, so do not
     revise it. `teco.md` is authoritative.
   - **(b) is sound in `teco.md`, and its consequence should be explicit.** "Never relaxes the
     prompt" means a gated narrowing line can relax only `REVIEW.md`-level or unprompted checks.
     Relaxing a check a prompt requires takes a prompt edit, which is `cobb`'s. That design holds
     together, and the conflict report sends the case to `cobb`. **L3d propagation: go**, to the
     seven pointer prompts, with one done-condition. The unit first rules on every existing
     narrowing line against each prompt. Today that is root `REVIEW.md` *Do not report* 1 (a spec
     departing from the plan where it matches the code), set against `analyst.md` l.36
     (conformance). I read these as not in conflict: conformance compares code to plan, and the
     line governs spec to plan. The unit should still rule on it, because after propagation a
     conflict leaves the gated line inert. A single root `REVIEW.md` line would cost fewer words,
     but it is weaker against S5's threat (a poisoned proposal), since it sits inside the file it
     governs. Reviewers keep "as a finding". `teco.md`'s copy rightly omits it.
4. **Frontmatter `description`. The premise is supported, by inference.** I cannot see the brief.
   However, the plan row commissions "l.3", and l.3 at `HEAD` is the `description` line. A brief
   that barred every frontmatter edit would therefore contradict the plan it executes. `teco`'s
   own verification (ledger l.17) checks only `tools`/`permissionMode`/`hooks`. **The edit is
   acceptable, and in fact required by the plan.** The YAML still parses, and no permission key
   moved.
5. **Sweeps:** see *Checks run*.

### Findings

**P18-1 (minor, verified; owner `teco`, not `cobb`) — The switch-over has no ledger text yet.**
Plan §3.4 puts the switch-over in "ledger text, not `AGENTS.md`": units dispatched earlier finish
under their brief, and later ones make no live-spec edit. `grep -n -i switch-over` on both open
ledgers finds only L3c's row label. Without that line, `teco.md` l.138's categorical "a build unit
that edited the spec is off-brief" would misfire on a unit that was legitimately briefed before the
switch-over. With it, the plan's in-change spec instructions also become close-step items, for
example `docs/plans/typed-decision-models3.md` §3.7, U9's `SERVER.md` §1.8 line. U9 is not
dispatched yet, so the exposure is small. Fix: in the L3c commit step, add this dated decision
line to `docs/plans/live-specs-coordination.md`, plus a one-line pointer in
`docs/plans/typed-decision-models-coordination.md`:

```markdown
- 2026-10-31: Switch-over at the L3c commit (<sha>): units dispatched earlier finish under their brief; later ones make no live-spec edit, and a plan's in-change spec instruction (e.g. `docs/plans/typed-decision-models3.md` §3.7, U9's `SERVER.md` §1.8 line) becomes a close-step item.
```

**P18-n1 (nit) — The README row drops the gate's exception.** "`cobb` gating any narrowing one"
leaves out the root's "(`analyst` if `cobb` proposed it)". Replacement:
"`cobb` (`analyst` for a line `cobb` proposed) gating any narrowing one", +6 words. Take or leave.

### What's solid

The Feature close bullet carries only the how and resolves every step to the root. The narrowing
definition is the first one that closes S5's bypasses without un-gating anything m2 gated. The
kaizen entries are accurate, and the `teco` plan note mirrors P16-2's repoint.

### Proposed REVIEW.md lines

None. P18-1 is a one-off ledger omission, and P18-n1 is catalog wording.
