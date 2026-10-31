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
