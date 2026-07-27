# Review — cross-document reference & naming convention (plan v1.1)

> **Reviewer:** `analyst` · **Date:** 2026-07-27 · **Artifact:**
> `docs/plans/doc-reference-convention.md` v1.1 (architect, untracked, 1145 lines).
> **Baseline:** working tree at `583e132`; the plan's own evidence commits `9bbfbb5` and `649b02a`.
> **Method:** static review + independent re-measurement. Every number below that I state as
> verified was produced by a command I ran in this repo; commands are quoted so they can be
> re-run or disputed. I did **not** re-derive §1.1's full reference census (the plan's script was
> not committed — see M4).
> **Verdict: NEEDS CHANGES** — 3 blockers · 5 majors · 9 minors.

---

## 0. Summary

This is an unusually well-grounded plan. I tried to break the measurement layer and mostly
failed: the two sweep decompositions reproduce **to the exact changed line**, and the §9.1
document census, the §9.4 role-suffix table, the 55/54 in-body counts, the 8-of-25 `Status:`
figure, the 39/15 rename pricing and the write-guard non-interaction claim are all **exactly
right** (§3 below). The two headline conclusions — **stop moving documents (D4)** and **rename
nothing (§10.1)** — are sound, and I endorse both.

The problems are not in the arithmetic. They are in three places:

1. **One decision-bearing factual claim is false**, and correcting it *inverts* the evidence for
   D1 (B1). The repo's entire broken-link baseline is drift in the very citation form the plan
   recommends.
2. **The naming grammar has a hole the repo has already fallen into once**, and the plan's own
   worked example of it is misdiagnosed (B2).
3. **The header block (§9.6) is under-scoped and unowned.** It collides with an existing,
   stakeholder-gated `Status:` vocabulary in `tico.md` that §9.7 says needs no change (B3), it
   defines values nothing normalizes (M2), and no step or prompt says who maintains it.

Proportionality: the archiving half (D4, S1–S3) is cheap and I'd ship it. The naming half is
mostly cheap too. The **4-field header is where the discipline cost concentrates**, and one of
its four fields (`Updated:`) does not earn its keep (M5).

**Split as the brief asks:**

- **Must change before implementing:** B1, B2, B3, M1, M2.
- **Worth doing:** M3, M4, M5, m1, m2, m3.
- **Acceptable as-is:** the D4 recommendation, the rename-nothing conclusion and its
  "how a reader tells old from new" answer, the §9.7 write-guard analysis, the S5 deferral,
  and the §10.5 honest do-nothing comparison.

---

## 1. Blockers

### B1 — The broken-link diagnosis is false, and correcting it reverses D1's strongest evidence

**Evidence.** The plan states three times that the repo's 3 broken relative links point at
never-created documents:

- §1.1 R3, lines 119–121: *"three never-created `workflow-def-structure-read` /
  `k027-parse-robustness` docs"*
- §1.3, line 203 and §4.4 line 543: carried forward as the baseline
- §6 **D5**, line 638: *"three never-created documents"*, with option **(b) Leave; they document
  planned-and-never-built deliverables.**

All three targets exist:

```
$ ls falkor-chat/docs/reviews/workflow-def-structure-read.md \
     falkor-chat/docs/plans/workflow-def-structure-read.md \
     falkor-chat/docs/reviews/k027-parse-robustness.md
(all three present)
```

The actual defect is an off-by-one `../`. `falkor-chat/docs/BACKLOG.md:785`:

```markdown
> [`docs/reviews/workflow-def-structure-read.md`](../reviews/workflow-def-structure-read.md)
```

From `falkor-chat/docs/BACKLOG.md` the correct target is `reviews/…`, not `../reviews/…`. Same
at :787 and :895.

**Why it matters — two consequences, the second is the important one.**

1. **D5 is unanswerable as posed.** Option (b)'s justification is factually wrong, and option
   (a) is not a "content fix" requiring judgement — it is deleting three `../` tokens. The
   correct disposition is: fix them, trivially, and *not* as a separate deliberation.
2. **This is direct evidence against the composed form, which D1 recommends.** All three broken
   links are the **composed** spelling — backticked repo-anchored label + relative link target,
   written twice on one line — and the two halves drifted. I counted the population:

   ```
   $ git grep -ohE '\[`[^`]+\.md`\]\([^)]+\)' -- '*.md' | wc -l
   143
   ```

   143 composed references, 3 of them broken: **the composed form produced 100% of this repo's
   broken-link baseline, at a measured ~2% drift rate.** §6.1 cites those same 143 as evidence
   *for* the form (*"so it is not an invention"*) without noticing that its own R3 number is that
   form failing. The plan does acknowledge drift as a theoretical cost of (b)/(c) and offers S4
   as the mitigation — but S4 is optional and deferred behind D2, so under the recommended
   sequence the mitigation never lands.

**Suggested improvement.**

- Correct R3, §1.3, §4.4 and D5 to say *"3 relative link targets in `falkor-chat/docs/BACKLOG.md`
  carry an extra `../`; all three targets exist"*, and collapse D5 to a one-line fix folded into
  S2 (no stakeholder decision needed).
- Restate D1 with this datapoint on the table: the composed form's *measured* defect rate in this
  repo is 3/143. That does not automatically kill (c) — (c) only mandates the link "where it adds
  navigational value" — but it changes the recommendation's honesty. Either strengthen (c) with a
  precondition (*"adopt (c) only together with S4, which is the only thing that detects
  label/target drift"*), or recommend (a) and note that GitHub navigation costs a copy-paste.

---

### B2 — The naming grammar has no legal name for a second primary document of a kind on a topic, and its own worked example is misdiagnosed

**Evidence.** §9.5 rule 5 (lines 893–903) handles "two documents of the same kind + role + topic
at different times" with:

- **Default:** revise in place, bump `Version:`.
- **Escape hatch:** *"append an **ordinal to the role token** — `x-impl2.md`, `x-report2.md`.
  Never a newly invented word."*

Both escape-hatch examples carry a role token. A **primary** document has none (§9.4: role
`(none)` = "the primary document of that kind"). So for a second plan on topic `x`, the grammar
offers no name: `x2.md` is not "an ordinal on the role token", and rule 3 forbids a new slug for
the same topic while rule 2 calls a downstream slug change a defect.

The default collides with D4. The accepted D4 marker text (§2.2, line 269) reads *"Frozen record;
**do not execute or amend**."* Revising a frozen plan in place is exactly amending it. §9.6
further defines `Status: superseded` and an optional `Superseded by:` pointer — so the model
*does* contemplate a successor document, but never says what it is called. `Superseded by:`
points at a filename the grammar cannot generate.

**The plan's own example gets this wrong.** §9.5 asserts *"`m3-executor-landing2.md` would have
been a `Version: 2` section of `executor.md`."* The document's own header
(`falkor-chat/docs/archive/plans/m3-executor-landing2.md:3-7`) says otherwise:

```
> **Status:** proposed (architect design-patch, 2026-07-12). Planning-only — no code/DDL changed.
> **Extends:** `docs/archive/plans/m3-executor.md` (approved plan, §6 trigger / §7 safety / Phases 4–5)
```

It is a design-patch **extending an already-approved, partly-executed plan**, deliberately kept
separate so the approved artifact stayed intact. Folding it into `executor.md` as a `Version: 2`
section would have rewritten a document that a review gate had already signed off on and an
implementer had already built from. The convention's prescription would have destroyed
information in the one case the repo actually produced.

**Why it matters.** N1 writes rule 5 into root `AGENTS.md`. The next architect asked to extend a
delivered plan has three options, all bad: amend a document marked "do not amend", invent a slug
(rule-2 defect), or improvise a filename outside the grammar. This is precisely the "second
document of the same kind on the same topic at different times" case the brief asked me to attack,
and the grammar does not survive it.

**Suggested improvement.** Add a fourth branch to rule 5, stated explicitly:

> **A superseding or extending primary document** — when the earlier one is `delivered`,
> `archived` or otherwise must stay intact — takes an **ordinal on the slug**:
> `<topic-slug>2.md` (e.g. `executor2.md`), with `Supersedes:`/`Extends:` in the header and
> `Superseded by:` added to the earlier document. The family stays greppable because the slug is a
> prefix. Revise-in-place applies only while `Status:` is `draft` or `active`.

Then re-classify `m3-executor-landing2.md` in §9.5 as an *instance of that branch*, not as a
rule-5 default failure — it is currently the plan's headline example and it argues the opposite of
what the evidence shows.

---

### B3 — §9.6's header collides with `tico`'s existing gated `Status:` vocabulary; §9.7's "exactly two prompt sentences change" is wrong

**Evidence.** §9.7 (line 943) says `tico` → *"✅ identical — **none**"*. That is true of tico's
**write path** and false of its **header contract**:

```
claude/tico/tico.md:37: > Status: Interviewing | Ready for design · Last updated: YYYY-MM-DD
claude/tico/tico.md:71: …flip `Status` to **Ready for design** only on their explicit confirmation…
claude/README.md:8:     …"Ready for design" only on your explicit confirmation.
```

`tico` already mandates a two-field header with a **closed, two-value** `Status:` set and a
`Last updated:` field. §9.6 mandates a four-field header with a *different* closed set
(`draft·active·delivered·superseded·archived`) and renames `Last updated:` → `Updated:`. Neither
of tico's values is in §9.6's set, and `Ready for design` is a *gated* state flipped only on
explicit stakeholder confirmation (tico.md:71) — it is load-bearing product behaviour, not
cosmetics.

This is confirmed on disk: all three `falkor-chat/docs/requirements/*.md` and both
`docs/requirements/*.md` carry tico-vocabulary `Status:` lines
(`Interviewing`, `Ready for design`, `Delivered ✅`).

The plan itself establishes the standard this violates. §4.2 (line 489): *"The one thing that must
NOT be incremental is D itself: root `AGENTS.md` and `qa-engineer.md` must flip in the same
commit, or `qa-engineer` will keep writing … against a rule that no longer has an `archive/`
destination."* The identical argument applies to `tico`: the day N1 lands, root `AGENTS.md` and
`tico.md:37` state contradictory `Status:` vocabularies, and `tico` will keep emitting
`Interviewing`.

**Why it matters.** D7 is the plan's highest-value step and its payoff is
`grep -m1 -H 'Status:' docs/plans/*.md` *naming* the state. If five of the twenty-five active
documents (all of `requirements/`) speak a different, prompt-mandated vocabulary, the grep is
complete in presence and incoherent in content. That is the same "looks complete while
incomplete" hazard D7 exists to prevent, moved one level down.

**Suggested improvement.** Pick one and write it into §9.6 and the N2 scope:

- **(preferred)** Make §9.6's set a superset that absorbs tico's: `draft` (≡ Interviewing) ·
  `ready` (≡ Ready for design) · `active` · `delivered` · `superseded` · `archived`, and change
  N2 to a **three**-prompt edit (`analyst`, `qa-engineer`, `tico` — plus tico's kaizen and
  `claude/README.md:8`); or
- Scope §9.6 explicitly to `plans/`, `reviews/`, `test-plans/`, `test-reports/` and state that
  `requirements/` keeps tico's vocabulary, with the mapping written down so a reader of the grep
  knows both dialects.

Either way, §9.7's summary line must change from *"Six of the seven fit exactly; two prompt
sentences change"* — it is at least three prompts once the header is part of the convention, and
the §5 documentation-impact table and §8's one-commit recipe must gain the `tico` rows.

---

## 2. Majors

### M1 — Forward-only naming will not hold: `<slug>` is not a prohibition, and `qa-engineer.md:28`'s "detect the convention, follow *that*" survives the ≈4-word edit

**Evidence.** §9.7 and §10.5 rest on "five of seven contracts fit unchanged" plus two sentence
edits. The write-path templates verified line-accurate
(`architect.md:40`, `tico.md:33`, `data-scientist.md:71–72`, `graph-dba.md:51`, `teco.md:56`,
`analyst.md:51,60`, `qa-engineer.md:28,41` — all confirmed) all say `<slug>` / `<kebab>`. **None
of them forbids anything.** `m4-executor` is a perfectly good kebab-case slug.

The corpus an agent orients from is 59% milestone-prefixed and, under the recommended
"rename nothing", stays that way:

```
$ find . -path '*/docs/*' -name '*.md' | grep -cE '/m[0-9]-'   # 36 of 61 feature docs
```

And `qa-engineer.md:28` — the line N2 edits — reads in full:

> *"**Detect the convention first.** Look at how the component already stores docs/plans … Write
> test plans to a parallel `docs/test-plans/<kebab-feature>.md` …, kebab-case, named for the
> feature/milestone under test. … **If a component uses a different convention, follow *that*.**"*

Dropping `/milestone` (≈4 words, as §9.7 prices it) leaves *"Detect the convention first"* and
*"If a component uses a different convention, follow that"* intact. A qa-engineer that obeys
those two instructions against `falkor-chat/docs/archive/test-plans/` — 4 of its 5 files
`m<n>-`-prefixed — will re-derive the milestone prefix from the corpus, correctly, per its own
prompt. The plan's Finding N1 (*"Left alone this goes to four schemes"*) therefore under-states
the risk: it survives N2 as scoped.

Mitigating fact I verified: root `AGENTS.md` **is** injected into subagent context (it is present
in this review session's own context), so N1's rule does reach every agent. But an explicit
component-level "follow *that*" override in a prompt beats a general rule in `AGENTS.md`.

**Suggested improvement.**

- Rewrite `qa-engineer.md:28` properly rather than surgically: keep "detect the convention" for
  *component-specific* deviations, but subordinate it — *"…follow that, **except for the filename
  grammar, which is repo-wide (root `AGENTS.md`) and not component-negotiable**."*
- Add the **prohibition** — not just the grammar — to the one-sentence rule in root `AGENTS.md`:
  *"a new document's basename never begins with `m<digit>-`, `k<digit>`, or a date."* A grammar
  that permits is not a rule that forbids.
- Revisit §3.2 item 6 for this specific clause. Item 6's argument ("put the rule once, don't
  duplicate into 5 prompts") is sound for the *citation* rule, which no prompt currently states.
  It is weaker for the *filename* rule, because every one of those 5 prompts **already carries a
  filename template** — adding four words (`, never prefixed with a milestone or ID`) to a
  template that already exists is not the same cost as inventing a new duplicated paragraph.
  Price the two separately; the plan currently treats them as one decision.

### M2 — `Status:` is defined but unowned and un-normalized; five values where D4 needs two

**Evidence — nobody owns the field.** Nothing in the plan or in any prompt says who sets or flips
`Status:`, or when. The only enforcement point named is §3.2 item 5 (a clause in `teco.md`'s
documentation-impact scan) and it is marked *"Optional, recommended."* N5 explicitly declines
script enforcement. So the plan makes `Status:` the sole lifecycle signal (D4 + the withdrawn §7
mitigation) while leaving its maintenance to an optional prompt clause.

**Evidence — the 8 existing lines are not normalized.** N3 backfills only the 17 documents that
*lack* a `Status:` line. The 8 that have one use 8 different free-text vocabularies, and **zero**
of them are in §9.6's set:

```
docs/plans/cpg-query-access.md                   Status: **approved (re-gate 2026-07-25: …)**
docs/requirements/cpg-query-access.md            Status: **Delivered ✅** — AC-1…AC-4 met …
docs/requirements/joern-cpg-pipeline.md          Status: M1 … **delivered ✅** · M2 …
falkor-chat/docs/plans/graphrag-eval-ml.md       Status:** proposed method note …
falkor-chat/docs/plans/workflow-def-structure-read.md  Status:** revised, **awaiting re-gate** …
falkor-chat/docs/requirements/{agent-import,summary-nodes,workflow-dependence-overlay}.md
                                                 Status: Ready for design | Interviewing …
```

Post-N3, `grep -m1 -H 'Status:'` returns 25 of 25 — and 8 of those 25 answers are
`approved (re-gate …)`, `revised, awaiting re-gate`, `proposed method note`. That is presence
without a lifecycle. The plan's D7 payoff (*"names the state instead of implying it"*, line 640)
is not delivered by N3 as scoped.

**Evidence — the value set is over-modelled for D4's job.** `joern-cpg-pipeline.md`'s real state
is *"M1 delivered, M2 delivered, M3 in progress"* — a single-valued field cannot hold it. And
§9.6 note 2 (*"a plan can be `delivered` while its milestone is still open"*) plus §2.2's marker
(`Status: archived 2026-07-26 — M3 closed`) mean each document is touched **twice** across its
life (active→delivered, delivered→archived). D4's actual requirement is binary: live vs frozen.

**Suggested improvement.**

- **Extend N3 to normalize all 25**, not backfill 17. It is still content-only and zero
  path-strings; it changes the step from 17 insertions to ~25 touched lines, and it is what makes
  the D7 payoff real. Update N3's file list, its `git diff --stat` expectation, and D7's option
  (a) text accordingly.
- **Name an owner and a trigger** for the field, in root `AGENTS.md` and in `teco.md` — and
  promote §3.2 item 5 from *optional* to required. Under D4 the status line *is* the archival
  sweep; leaving its trigger optional reintroduces exactly the "status markers rot" risk §7 rates
  Low on the strength of that clause existing.
- **Cut the set to three** (`active` · `superseded` · `archived`), or state the transitions and
  who performs each. `draft` and `delivered` add two more touches per document for information
  already in `BACKLOG.md`/`HISTORY.md` — the same argument §9.3 uses to remove the milestone from
  the filename applies to them.

### M3 — Finding N4's grep claim is false, which weakens D6's case

**Evidence.** §9.1 Finding N4, line 772: *"`git grep workflow-def-structure-read` does not
surface the impl review."*

```
$ git grep -n workflow-def-structure-read -- 'falkor-chat/docs/reviews/k031-structure-read-impl.md'
:4:  > **Baseline:** `docs/plans/workflow-def-structure-read.md` **v2** + its two gates
:5:  > (`docs/reviews/workflow-def-structure-read.md`, round 1 + re-gate …)
:105: re-gate's RG-m3 (`docs/reviews/workflow-def-structure-read.md:603-611`) offered (a) and (b)…
```

`git grep -l workflow-def-structure-read -- '*.md'` returns 7 files **including** the impl review.
The family is fully discoverable by content grep; only the *basename* diverges.

**Why it matters.** N4 is the plan's single concrete "already produced a defect" evidence for the
`-impl` role and for rule 2, and it is the stated reason D6 option (b) is *"a false economy"*
(line 639). The real cost is narrower: the family is not discoverable by directory listing or
filename glob. That is still worth fixing, but it does not carry the weight the plan puts on it.

Related, and it cuts the other way: `analyst.md:51` **already** says *"kebab-case slug matching
the artifact under review"* — i.e. rule 2 in embryo, already in a prompt, **already breached** by
`k031-structure-read-impl.md`. That is better evidence than N4's grep claim, and it is evidence
about **enforcement** (see M-note in §4 on D8), not about the rule's absence.

**Suggested improvement.** Restate N4 as *"the basename diverges, so the family is invisible to
`ls`/glob though visible to content grep"*, and move the `analyst.md:51`-already-breached
observation into N4 as the stronger evidence. Then re-check D6's (b)-is-a-false-economy claim
against the corrected premise — I still think (a) is right, but for the `-impl` naming, not for
family discoverability.

### M4 — §1's census is not reproducible, yet D3/S2 write its numbers into HISTORY as the new baseline

**Evidence.** §1 line 63: *"produced by a read-only census I wrote for this assessment (scratch
script; **not committed**)"*. D3 then asks the stakeholder to **retire** the committed 442
baseline — with the argument *"a number nobody can decompose cannot be driven down"* (line 636) —
and S2/N4 write **3 / 87 / 15** into `docs/HISTORY.md` and `falkor-chat/docs/HISTORY.md` in its
place.

Those three numbers are, today, exactly as unreproducible as the 442. And the one I could check
cheaply does not match. My independent pass over `git ls-files '*.md'` (fenced blocks masked,
`<>*{}` placeholders skipped, targets resolved relative to the citing file) finds **4** broken
relative links, not 3 — the fourth being `claude/architect/kaizen/inbox.md:201 → ../relative.md`,
plainly an illustrative placeholder. So the plan's 3 is *defensible*, but only under an exclusion
rule that is nowhere stated. That is the 442's failure mode reproduced at smaller scale.

There is also a sequencing problem: S2 (record the baseline) is scheduled **before** S4 (build the
checker), and S4 is optional and gated on D2. If D2 says no, HISTORY permanently carries three
numbers no committed artifact can regenerate.

**Suggested improvement.** Make S2 depend on the census being reproducible. Either:

- Promote the census script to a deliverable of **S2**, not S4 — commit it as
  `claude/scripts/check-doc-links.sh` in report-only form (it already exists; committing it costs
  nothing) and have S2's HISTORY entry cite *the script and its exact invocation* alongside the
  numbers; or
- If D2 declines the checker, drop the numeric baseline from S2 entirely and record only the
  qualitative finding (two anchoring conventions; archival rot is confined to dated records).

Also state the exclusion rules (placeholders, illustrative paths, `<…>` forms) in the HISTORY
entry, since they are what makes 3 vs 4 a choice rather than a fact.

### M5 — `Updated:` will rot, duplicates `git log`, and nothing checks it

**Evidence.** §9.6 defines four required fields and justifies `Updated:` as *"Carries the
chronology the filename no longer does"* (line 919). But:

- Nothing in the plan tells any agent to bump it. N5 declines script enforcement; S4's optional
  census (line 1106–1110) checks only for `^[mk][0-9]` basenames and the presence of `Status:`.
- `git log -1 --format=%ad -- <file>` answers the same question authoritatively, for free, and
  cannot rot.
- The one field that genuinely cannot come from git is `Tracks:`, which the plan correctly calls
  *"the only genuinely new field"* (line 920).

§9.6's own standard — *"every addition erodes the reason to adopt it"* (line 929) — argues against
its inclusion. A required field that (a) duplicates a free authoritative source, (b) has no
maintainer, and (c) is unchecked is the definition of a discipline cost without a benefit; it will
be stale within two revisions and then actively misleading.

**Suggested improvement.** Demote `Updated:` to the optional list beside `Version:`, or redefine
it narrowly as *"the date `Status:` last changed"* — which makes it a lifecycle fact with a
defined trigger rather than a general freshness claim, and gives it a maintainer (whoever flips
`Status:`, per M2). Keep the required block at three fields: `Status:` · `Owner:` · `Tracks:`.
That is a materially easier rule to follow and loses nothing.

---

## 3. What's solid — verified

Stated so these are not churned along with the rest. Each reproduced independently.

| Plan claim | Result |
|---|---|
| §9.1 — 67 `*/docs/*.md`, 6 fixtures, 61 feature docs, 25 active / 36 archived | **exact** |
| §9.1 — schemes: 36 milestone (6 active/30 archived), 4 ID (2/2), 21 bare (17/4) | **exact** |
| §9.1 — active breakdown 12 plans / 8 reviews / 5 requirements / 0 test-plans / 0 test-reports | **exact** |
| §9.3 — 55 of 61 name a `[CK]-\d{3}` in-body; 54 of 61 name an `M<n>` | **exact** (scripted re-count) |
| §9.1 N3 — role-suffix counts: `-coordination` 6, `-report` 6, `-ml` 4, `-impl` 4, `-landing2` 1, `-queries` 1, `-skill` 2, `-sweep` 2, `-graph` 0, `-rca` 0 | **exact, all ten** |
| §1.2 Sweep A — 2 docs, 8 files; outbound 9 changed lines = 6 depth churn (3+3) + 2 self-citation + 1 normalisation | **exact, to the changed line** |
| §1.2 Sweep B — 26 files, 20 renames, **2** `../`-bearing changed lines inside the moved docs | **exact** |
| §10.1 — renaming the 6 active `m<n>-` docs = 39 occurrences / 15 files; 4 of the 15 are dated records | **exact** (both `git grep` commands reproduce verbatim) |
| §10.2 — k031 re-slug = 4 occurrences / 3 files | **exact** |
| §9.5 rule 4 — all 16 citations of `m2-cpg-analysis-skill.md` carry a directory; zero bare | **exact** |
| §9.7 — write-path contract lines (`architect.md:40`, `tico.md:33`, `data-scientist.md:71–72`, `graph-dba.md:51`, `teco.md:56`, `analyst.md:51,60`, `qa-engineer.md:28,41`) | **all line-accurate** |
| §9.1 N1 / §5 — `qa-engineer.md:28` is the only line in all 13 prompts mentioning "milestone" | **exact** |
| §9.7 — five `PreToolUse` doc guards, directory-only globs, shared `claude/scripts/guard-doc-writes.sh`, `case`-glob matcher, escalation = interactive `ask` | **exact** — and the script is additionally **fail-open** when the path can't be extracted, which strengthens the conclusion |
| §3.1 — `claude/AGENTS.md` has zero docs-tree/archive mentions; `skills/` has exactly one incidental hit (`skills/joern-cpg/references/cpg-model.md:66`) | **exact** |
| §3.3/§10.5 — `audit-team.sh` already FAILs with exactly 2 (username + home-path leaks) | **exact** (ran it) |
| §4.3 S2 — `docs/HISTORY.md:54`'s "remain as empty [active directories]" is false | **exact** — neither `docs/test-plans/` nor `docs/test-reports/` exists |
| §3.3/§7 — CI is path-filtered to `falkor-chat/**`; no `.kiro/steering/`; a GitHub `origin` exists | **all confirmed** |
| §6 D7 — 8 of 25 active docs carry a `Status:` line; the N3 list of 17 is exactly the complement | **exact** (the 9th hit is a prose false positive in this plan — see m1) |

Beyond the numbers, three judgement calls are right and well-argued and I would not reopen them:

- **D4 (stop moving documents)** — the argument that a path segment signals lifecycle only to a
  directory browser, while agents always arrive by link, is correct and the `9bbfbb5`/m-26
  evidence for it is real.
- **Rename nothing (§10.1)**, including the "how a reader tells old from new" answer. *"From the
  filename, they can't — and they don't need to"* is the honest answer, and it holds **because**
  rule 4 is true (verified: 16/16 citations carry a directory). The `m3-followups-coordination.md`
  counter-example — a topic whose slug genuinely *is* the milestone — is the right stress case and
  is answered correctly.
- **The §9.7 write-guard non-interaction analysis** and the resulting choice of `coder` as N3's
  owner.

---

## 4. Minors and nits

**m1 — N3's self-verification is a substring grep that already false-positives.** The done-condition
(lines 1076–1082) is `head -12 "$f" | grep -q 'Status:'`. Run against
`docs/plans/doc-reference-convention.md` **today** it passes — because line 5's changelog prose
contains the literal `` `Status:` `` — yet that file is on N3's own to-do list. The check also
verifies 1 of the 4 fields it gates, and validates no value. Suggest: anchor it
(`grep -qE '^> \*\*Status:\*\*'`) and check all four field names. Also, `git diff --stat` will show
16, not 17, files: this plan is untracked (`git status --porcelain` → `?? docs/plans/doc-reference-convention.md`).

**m2 — §2.1 double-counts option A's saving.** The table prices **A** at *"eliminates outbound
depth churn — 8 (4.5%)"* and **D** at *"all of it — 179 (100%)"*. Under D nothing moves, so A's 8
are already inside D's 179; the recommendation "D + A" cannot claim both. The plan is honest about
this in prose (line 303: A rides along *"not because of archiving … but because Finding R2 is a
live defect"*) — the table should say A's incremental archival saving over D is **0**, and its
justification is R2 alone. As written, a skim of §2.1 over-sells A.

**m3 — §8's decision state is inconsistent with §6.** Line 695–697: *"D1 still open … D6/D7/D8 new
and open. **Nothing else blocks.**"* But §6 shows **D2, D3 and D5 unmarked**, i.e. also open — and
D2 gates S4, D3 gates S2's HISTORY content, D5 is a standalone fix. An implementer reading §8 (the
"Ready to implement" section, explicitly the consolidated summary) would start S2 without D3's
ruling. Fix §8 to list all five open decisions.

**m4 — §2.3 still says "33 files in `falkor-chat/docs/archive/`"** (line 335). The v1.1 changelog
flags the correction to 34 but the body was never edited. Verified: 34. Cosmetic, but it is the
one number a reader is told is wrong and then reads unchanged.

**m5 — §9.7's `case`-glob claim is overstated.** Line 956: *"the matcher is a shell `case` glob,
which **cannot** express 'must not start with `m<digit>`'."* With `shopt -s extglob`, bash `case`
handles `docs/plans/!([m][0-9])*`. The conclusion is unaffected — the *sound* reasons are the other
two the plan gives (an interactive human prompt is the wrong altitude for a naming nit; and
`qa-engineer`/`coder`, who write test-plans/reports, have no such hook at all). Drop the
"cannot express" clause so the argument isn't undermined if someone revisits it.

**m6 — §10.1's `grep -L 'Status:'` stops discriminating the moment N3 lands.** It is offered as the
mechanical answer to "old vs new", but N3 empties it for all 17 active documents in the same
change. Post-N3 it distinguishes nothing; only the second one-liner
(`grep -m1 -H 'Status:'`) earns a place in `AGENTS.md`. Also see m1 — it is substring-matched.

**m7 — the §10.1 rename pricing is `git grep`-based and therefore untracked-blind.** Both quoted
commands use `git grep`. This plan is untracked and cites `m2-cpg-analysis-skill.md`,
`m3-archive-sweep.md`, `m3-followups-coordination.md` and others several times, so 39/15 is a
**lower bound** once the plan is committed. This strengthens the plan's conclusion (renaming is
even more expensive), so no action is needed beyond a parenthetical — but the same blindness
applies to the S4 checker (§3.3 caveat 2), where the plan does call it out.

**m8 — N3's owner is priced as mechanical but carries ~17 small judgement calls.** `Owner:` and
`Tracks:` must be *inferred* for documents lacking `Author:`/`Date:` (e.g. what does
`falkor-chat/docs/reviews/m3-archive-sweep.md` track? who owns `wsl2-memory-diagnostic.md`?).
That is fine for `coder`, but the step should say "reuse existing values where present; where
absent, derive `Tracks:` from the document's body `[CK]-\d{3}` mention and leave `—` if none" so
two implementers produce the same thing.

**m9 — D7 and D8 are not really stakeholder decisions.** **D7** is the direct, already-priced
consequence of a D4 the stakeholder has accepted, and the plan itself says do-nothing *"is not
defensible"* (line 1139) — offering option (c) *Skip* is ceremonial. **D8** is a technical call
(should we add a script check?) with four verified reasons already decided in N5; escalating it
reads as deferring a judgement the architect has in fact made. Both are low-stakes, but the
stakeholder is cost-sensitive and a decision list that mixes real forks (D1, D6) with
already-settled ones dilutes the real ones. Suggest: fold D7 into D4's ruling as a consequence,
and record D8 as an architect decision with the four reasons, flagged for objection rather than
posed as a question. **D1 and D6 are genuine, fairly optioned, and honestly recommended** — D1
especially, where the recommendation explicitly identifies the reversible direction of the bet.

---

## 5. On the brief's proportionality question

The stakeholder is cost-sensitive and this plan established its own ceiling on savings. My read of
where the ongoing discipline cost actually lands:

- **Worth it, clearly.** D4 + the S1–S3 prose edits. One paragraph, ~4 sentences, no file moves,
  self-verifying, and it converts ~8 path edits per archived document into 1 line. Ship it.
- **Worth it, cheaply.** The filename grammar minus the prefix (§9.2–§9.4). It is what the agents
  already emit, and the marginal cost per new document is genuinely zero — *provided* M1 is fixed,
  because as scoped it will not actually change behaviour.
- **Worth it only if fixed.** N3 / the header. It is the plan's highest-value step and I agree
  with that ranking, but at 4 required fields with an unowned lifecycle field, a colliding
  vocabulary (B3), an unchecked freshness field (M5) and 8 un-normalized existing lines (M2), the
  version in the plan will not deliver the payoff it is sold on. **Three fields
  (`Status:`/`Owner:`/`Tracks:`), one absorbed value set, one named owner, all 25 normalized** —
  that version I would ship without hesitation.
- **Not worth it as recommended.** The **composed citation form** under D1(b)/(c) without S4.
  It mandates writing every path twice, and this repo's only measured evidence about that form is
  that it produced 100% of its broken links (B1). Either take S4 with it or take (a).
- **Correctly rejected.** S5 (bulk repath), all renames, un-archiving, option B (flatten), option
  C (IDs). The cost analyses behind each are sound.

---

## 6. Open questions for the caller / stakeholder

1. **D1 remains genuinely stakeholder-only** — do you read these documents on github.com? The
   plan is right that nobody else can answer it. B1 changes the *evidence* for the answer, not who
   answers it.
2. **B3's fork is a product question, not just a technical one:** should a requirements document's
   `Status:` keep tico's interview vocabulary (`Interviewing` → `Ready for design`, gated on your
   explicit confirmation), or be folded into the lifecycle set? I recommend absorbing it into a
   6-value set so one grep answers everything, but the gated `Ready for design` semantics are
   yours, not the architect's, to change.
3. **B2's escape hatch needs a naming call** — `executor2.md` (ordinal on the slug) is my
   recommendation, but any convention works as long as one is stated; the plan currently states
   none.

---

*Findings route to `architect` for plan revision. Nothing in this review was fixed in place; the
plan document is unmodified.*

---
---

# Part II — re-review of plan v1.2

> **Reviewer:** `analyst` · **Date:** 2026-07-27 · **Artifact:**
> `docs/plans/doc-reference-convention.md` **v1.2** (architect, untracked, 1637 lines), including
> the new **§11** finding-ID → disposition table.
> **Baseline:** working tree at `583e132`; Part I (above) is the prior round and is kept verbatim
> as the audit trail.
> **Method:** re-read both documents end to end; re-ran every measurement the disposition table
> claims, plus the three claims the architect says Part I got wrong; independently attacked the
> rewritten §9.5 rule 5, the rewritten §9.6 header, and the enlarged N2/N3 scope. Commands are
> quoted so they can be disputed. I did **not** re-derive §1.1's full 2,525-reference census (still
> uncommitted — see M4's disposition).
> **Verdict: NEEDS CHANGES** — 2 blockers · 4 majors · 9 minors.
> **But read the split in §13 first:** all three v1.1 blockers are closed, both new blockers are
> one-sentence fixes confined to **N3** and **N2d**, and **S1–S3 + N1 + N2a/b + N4 are approved to
> implement as written, today.**

---

## 7. Summary of the round

v1.2 is a materially better document than v1.1 and it answers the review honestly, including where
it disagrees. **All three v1.1 blockers are closed at the level I raised them**, the two
"fixed with a variation" dispositions are improvements on what I suggested (B2's ordinal-on-slug is
strictly simpler than v1.1's two-hatch grammar; M4's "drop the number" is more honest than my
"commit the script"), and the one partial rejection (M1's third suggestion) is **correct on the
evidence** — I re-grepped and confirm the claim. The three corrections the architect aims back at me
are all three right, including the extglob one, which I ran.

The new problems are almost entirely **second-order effects of v1.2's own fixes**, which is the
normal failure mode of a revision round:

1. **M2's fix (cut `draft`/`delivered`) collided with B2's fix (branch on the earlier document's
   `Status:`)** — the token that used to select the branch no longer exists, so rule 5's selector
   now routes its own headline example to the wrong branch (**M6**).
2. **B3's fix absorbed `tico`'s *values* but not `tico`'s *syntax*** — `tico` writes
   `> Status: Interviewing`, unbolded; the anchored done-condition m1 asked for demands
   `> **Status:**`. N3 therefore cannot pass its own gate on the three documents it is told not to
   change (**B4**).
3. **M2's fix named `teco` the owner of the `Status:` flip** — and `teco` is the one agent whose
   `PreToolUse` guard allowlists `docs/plans/*` only, so the flip is harness-blocked for
   `reviews/`, `requirements/`, `test-plans/` and `test-reports/` (**B5**).

Both blockers are cheap: one is a sentence about spelling, the other is a routing choice between two
options I name below. Neither touches the plan's design.

---

## 8. Disposition audit — all 17 v1.1 findings

Verified independently, not accepted from §11. "Closed" = the defect I named is gone and nothing
downstream still leans on it.

| ID | §11 claims | My verification | Verdict |
|---|---|---|---|
| **B1** | Fixed; R3/§1.3/§4.4/§6.1 corrected, D1 **reversed**, D5 collapsed into S2 | Re-ran the link census: **4** broken relative links in tracked files, **all 4 composed form** (`falkor-chat/docs/BACKLOG.md`:785/787/895 + `claude/architect/kaizen/inbox.md`:201) — the plan's count and classification are exact. `git grep -ohE '\[`[^`]+\.md`\]\([^)]+\)' -- '*.md' \| wc -l` → **143**, so 3/143 ≈ 2% reproduces. `grep -n 'never-created'` returns only the two places that *quote* the false claim to correct it (§1.1 R3, D5) — **no surviving argument leans on it**. D1's recommendation is genuinely (a) now, and §6.1 states the population-is-the-defect-population error in its own words. | ✅ **closed** |
| **B2** | Fixed with a variation — ordinal on the **slug**, role-ordinal hatch withdrawn, `landing2` re-derived | The grammar now names a successor for a primary document, `Superseded by:` finally has a legal target, and withdrawing `x-impl2.md` really is a *simplification* (one ordinal rule, not two). The `landing2` re-derivation is correct and its "the separate document was the right call; only the filename was wrong" reading matches the file's own `Extends:` header. **But three of the four cases I said I would attack are not yet single-answer** — see **M6** (branch selector), **M7** (amending a frozen document), **M8** (a topic that *is* a milestone, for a **new** document). | ⚠️ **closed for the case raised; 3 new seams** |
| **B3** | Fixed via the preferred option — `tico`'s values absorbed **verbatim**; scope corrected to 4 prompts + root `AGENTS.md` | The gate genuinely survives: absorbing `Interviewing`/`Ready for design` as literal members means `tico.md`:71 and `claude/README.md`:8 need no edit, and N2c's byte-identical done-condition is the right proof. I re-grepped **all 13 agent definitions** for a header/status contract — `grep -niE '^\s*>?\s*\*{0,2}(status\|owner\|tracks\|version\|last updated)\b' claude/*/[a-z]*.md` returns **exactly one line, `tico.md`:37** — so **the corrected scope is complete at the prompt level**. Two residues: the **syntax** collision (**B4**) and the fact that no producing agent's prompt tells anyone to *write* the header at creation (**M9**). | ⚠️ **value collision closed; syntax collision open** |
| **M1** | 2 of 3 adopted; duplicating into 5 more prompts **rejected with evidence** | **The rejection is correct.** I re-grepped with a wider net than the plan's (`follow that`, `existing convention`, `match the …`, `discover`, `mirror`, `house style`): the other override clauses are `frontend-engineer`:18 (UI code), `coder`:13 (code style), `tdd-engineer`:33 (test framework), `graph-dba`:46 (graph labels), `devops`:70 (infra) — **none is a document-filename contract**. One miss inside the same file: **`qa-engineer.md`:54** *"Discover and follow each component's framework, runner, file layout, naming, and **doc conventions** … Read the component's `AGENTS.md` first"* is a second override, weaker but on-topic (**m17**). | ✅ **rejection upheld** (one in-file addendum) |
| **M2** | Fixed all three — normalise all 26, `teco` named owner, set cut | Normalisation to 26 is right and I verified the population (`find … -not -path '*/archive/*'` → **26**, of which **8** carry a `Status:` line today). Set cut to 5 is the right call. **The owner half is the problem**: see **B5**. Dropping `delivered` also destabilised rule 5: see **M6**. | ⚠️ **2 of 3 closed** |
| **M3** | Fixed — N4 restated, `analyst.md`:51-already-breached adopted, D6 re-derived | N4 now says exactly what is true (`ls`/glob-invisible, content-grep-visible) and D6's justification is rebuilt on the enforcement evidence rather than the false grep claim. D6(b) is still rejected, now for a reason that survives. | ✅ **closed** |
| **M4** | Fixed with a variation — numbers dropped rather than script committed; **my 4 confirmed**; new 10-illustrations hazard | Choosing "record nothing unreproducible" over "commit the script" is the better answer for a cost-sensitive stakeholder and I withdraw my first option. The new hazard is real and **understated**: my own pass over the untracked plan flags **13**, not 10 (the extra three are `(target)`, `../reviews/…` and one table cell) — which *is* the plan's point: the count is a function of an unwritten exclusion rule. §3.3 caveat 0 is the right home for it. Residue: §7's checker row still says *"Record the baseline in HISTORY so a future sweep compares against a number"* (**m11**). | ✅ **closed** |
| **M5** | Fixed — `Updated:` demoted, 3 fields, §9.3's dependent claim withdrawn | Verified: §9.6 lists `Updated:` under *"Explicitly not fields"*, `Last updated:` survives only as `tico`'s, and §9.3's chronology sentence is explicitly retracted at line 1284. | ✅ **closed** |
| **m1** | Fixed — anchored regex + value whitelist, all 3 fields, false-positive document listed | The anchor is right and it is what exposes **B4**. Two leftovers: the `head -8` window misses `docs/plans/cpg-query-access.md`'s existing `Status:` at **line 11** (**m12**), and N5 still uses a *12*-line window for the same check. | ⚠️ **fixed; window inconsistent** |
| **m2** | Fixed — A's incremental saving priced at 0 | Verified at §2.1's new note. | ✅ |
| **m3** | Fixed — §8 tables all decisions | Verified: §8 now lists D1–D8 with state and what each gates. One sequencing gap remains (**m18**). | ✅ |
| **m4** | Fixed — 34 | Verified in body at §2.3. | ✅ |
| **m5** | Fixed **with a correction to me** — my `!([m][0-9])*` is wrong; `!([mk][0-9]*)` works | **Ran it.** `shopt -s extglob; case "docs/plans/m3-x.md" in docs/plans/!([m][0-9])*)` → **MATCH** (my pattern flags nothing; the trailing `*` lets the negation consume just `m`). `docs/plans/!([mk][0-9]*)` → correctly NOMATCH on `m3-x.md` and `k031-x.md`, MATCH on `executor.md` and `machine-learning.md`. **The architect is right and I was wrong.** My m5 *conclusion* stands unchanged and is now better supported: the "cannot express" clause had to go, and the two sound reasons carry the argument — verified again that `qa-engineer` has only a `guard-destructive-ops.sh` hook and `coder` has **no** hook at all, so the write guards could never cover the agents that write test-plans, test-reports and N3's headers. | ✅ **closed; correction accepted** |
| **m6** | Fixed — `grep -L` dropped | Verified at §10.1. | ✅ |
| **m7** | Fixed — 39/15 labelled a lower bound | Verified. | ✅ |
| **m8** | Fixed — derivation rules added | Rules exist and are the right three. They still leave one systematic ambiguity (**m16**) and they do not resolve *where* a pre-existing `Status:` line is normalised (**m12**). | ⚠️ **mostly** |
| **m9** | Fixed — D7 folded into D4, D8 recorded as an architect decision flagged for objection | Verified in §6 and §8; D8's four reasons are stated at N5 and the "flagged for objection, not posed as a question" framing is exactly right. | ✅ |

**§11's own arithmetic checks out:** 14 fixed + 2 with a variation + 1 partial rejection = 17 = 3 + 5 + 9.

---

## 9. New blockers

### B4 — N3 cannot pass its own done-condition on the three `tico` documents it is told not to change

**Evidence.** §9.6 absorbs `tico`'s *values*. It does not absorb `tico`'s *spelling*. On disk
(`claude/tico/tico.md`:37 and all three documents written from it):

```
claude/tico/tico.md:37                                  > Status: Interviewing | Ready for design · Last updated: YYYY-MM-DD
falkor-chat/docs/requirements/agent-import.md:2         > Status: Ready for design · Last updated: 2026-07-22
falkor-chat/docs/requirements/summary-nodes.md:2        > Status: Interviewing · Last updated: 2026-07-12
falkor-chat/docs/requirements/workflow-dependence-overlay.md:2   > Status: Interviewing · Last updated: 2026-07-23
```

`Status:` is **unbolded**. N3's done-condition (plan lines 1525–1532, the anchoring I asked for in
m1) is:

```bash
grep -qE '^> \*\*Status:\*\* (Interviewing|Ready for design|active|superseded|archived)' <<<"$h"
```

All three fail it. And N3's instruction for exactly those files (line 1489–1490) is: *"The last
three are `tico`'s and **already conform** — their … values are canonical under §9.6 and **must not
be changed**; they need only `Owner:`/`Tracks:`."*

**Why it matters.** N3 is the plan's highest-value step and the one it says to ship if only one
thing ships. An implementer in an isolated context — `coder`, by the plan's own assignment — hits a
direct contradiction between the instruction and the gate, and must invent the resolution: bold the
three headers (contradicting "must not be changed"), or relax the check (discarding m1's fix). Both
choices then propagate: if the canonical form is bold, `tico.md`:37's **template** must gain the
bold too, or `tico`'s very next requirements document is non-conformant on the day it is written —
and N2c's scope says only *"gains `Owner:`/`Tracks:`"*. §7's risk row and §8's verification line
both assert the `tico` interaction is *verified* safe; it is verified safe for the **gate**
(`:71`/`README.md`:8, which genuinely never change) and unverified for the **syntax**.

**Suggested improvement** — one sentence in §9.6 plus two words of scope, no design change:

- State the canonical **form**, not just the vocabulary: *"the header line is
  `> **Status:** <token> …`; the canonical token is the first thing after `Status:`, and the field
  labels are bolded."*
- Say in N3 that the three `tico` documents are normalised **in form only** — bolded, `Owner:`/
  `Tracks:` added — with the **value strings untouched**, and that this is not a value change.
- Extend N2c to: *"`:37`'s template gains `**Owner:**`/`**Tracks:**` **and bolds `Status:`/`Last
  updated:`**; the two value strings are byte-identical."* The `:71`/`README.md`:8 done-condition is
  unaffected and remains the right proof.

*(Alternative, equally acceptable: keep the unbolded form as canonical and relax the regex to
`^> \*{0,2}Status:\*{0,2} ?(…)`. Either resolution is fine — the plan must pick one.)*

---

### B5 — `teco` is named owner of the `Status:` flip, but its own write guard blocks 3 of the 5 document kinds

**Evidence.** M2's fix makes `teco` the standing owner of the archival flip (§3.2 item 5, §9.6's
"who flips it" column, N2d, and §7's status-rot mitigation). `teco`'s `PreToolUse` guard
(`claude/teco/hooks/guard-coordination-doc-writes.sh`, wired in its frontmatter for `Write|Edit`)
passes exactly:

```
'docs/plans/*|*/docs/plans/*|teco/kaizen/inbox.md|*/teco/kaizen/inbox.md'
```

and `claude/scripts/guard-doc-writes.sh` returns `permissionDecision: "ask"` for everything else —
an **interactive human approval prompt per file**. Bash is deliberately not covered, and using it to
route around the guard is called out in the script header as a guardrail violation, so there is no
legitimate escape.

At a milestone close, the documents to flip span `plans/`, `reviews/`, `requirements/`,
`test-plans/` and `test-reports/`. **`teco` can flip only `plans/` silently**; every review,
requirements doc, test plan and test report escalates. Concretely, `falkor-chat/docs/reviews/`
alone holds **4** active documents today — 4 approval prompts at the next close, for a step the plan
prices at "one line per document".

**Why it matters.** This is D4's entire replacement for `git mv` + ~8 inbound repaths. §7 rates
"status markers rot" **Medium** *on the strength of N2d existing*; if the named owner is
harness-blocked for 3 of 5 kinds, the mitigation is weaker than rated and the realistic outcome is
that flips on non-plan documents quietly don't happen. The plan already applied exactly this
reasoning one step earlier — N3's owner is `coder` *"because the write guards make this awkward for
the doc-scoped agents"* (line 1506) — and then didn't apply it to the recurring case. §9.7's
"verified non-interaction" analysis tested one direction only (can a *filename* rule break the
guards? no) and never the other (can the *named owner* perform the *newly required write*?).

**Suggested improvement** — pick one, and state it in §9.6's "who flips it" column, N2d and §7:

- **(preferred, and cheaper)** `teco` **coordinates** the flip; **each kind's owner performs it** —
  `architect` → `plans/` (guard allows), `analyst` → `reviews/` (allows), `tico` → `requirements/`
  (allows), `qa-engineer`/`coder` → `test-plans/`, `test-reports/` (no doc guard at all). This is
  the existing guard topology, needs no hook edit, and matches `teco`'s own charter ("coordinates,
  doesn't do"). N2d's clause becomes *"…make the `Status:` flip a done-condition of the closing
  unit, routed to each document's owner"*.
- **(alternative)** `cobb` extends `teco`'s allowlist to
  `*/docs/{plans,reviews,requirements,test-plans,test-reports}/*` in the same commit as N2d — a
  one-line hook change, but it widens a deliberately narrow guard and should be argued on its own
  merits, not smuggled in as a side effect of a docs convention.

Either way, add a §7 row: *"the archival flip is a guarded write for 3 of 5 kinds"* with the chosen
mitigation.

---

## 10. Majors

### M6 — rule 5's branch selector no longer selects: dropping `delivered` (M2) removed the token B2's fix keys on

**Evidence.** §9.5 rule 5 says the two branches are *"chosen by the earlier document's `Status:`"*:

- branch 1 — *"**While the earlier document is `active` (or `Interviewing`)** — revise it in place"*;
- branch 2 — *"**Once the earlier document is `archived`, `superseded`, or otherwise must stay
  intact** — write a successor"*.

But §9.6 dropped `delivered` and `draft`, and note 2 states the resulting lifecycle in full: *"a
plan/review is written `active` and flipped once to `archived`."* So **an approved, gated,
partly-executed plan whose milestone is still open is `active`** — there is no other token for it.
Read literally, the selector routes exactly the `m3-executor.md` situation to **branch 1: revise in
place** — the disposition the same section spends 18 lines proving was wrong (*"would have rewritten
a signed-off document, i.e. destroyed information"*). The whole weight falls on *"or otherwise must
stay intact"*, which is an undefined, subjective test doing the work a token used to do.

The same collapse affects the token's stated meaning: `active` = *"live; **may be executed and
amended in place**"* is now the label carried by delivered-but-not-yet-archived documents, i.e. the
grep D4 depends on answers "amendable" for documents that are finished.

**Why it matters.** N1 writes rule 5 into root `AGENTS.md`, where it is the only guidance the next
architect gets. B2 asked for one legal answer per case; for the case that motivated the rule, the
selector as written gives the wrong one.

**Suggested improvement** — do not re-add `delivered`; sharpen the selector instead:

> *"Revise in place **only while the document has not yet been approved, gated, or executed
> against**. Once it has — even if its milestone is still open and its `Status:` is still `active` —
> it must stay intact: write a successor (`<slug><n>.md`)."*

and adjust `active`'s meaning column to *"live; amendable until it has been approved/gated or
executed against."* One sentence in each of §9.5 and §9.6.

### M7 — rule 5 requires amending a document that `Status: archived` forbids amending, and applies `Superseded by:` to the `Extends:` case where it is wrong

**Evidence.** Rule 5, branch 2 (lines 1182–1184): *"The successor's header carries `Supersedes:` …
**or** `Extends:` …, and the **earlier document gains `Superseded by:`**."* Two problems:

1. The trigger for branch 2 is that the earlier document is `archived` — whose §9.6 meaning is
   *"frozen record; **do not execute or amend**"* (the same wording as D4's accepted marker, §2.2).
   Writing `Superseded by:` into it **is** an amendment. The rule instructs a violation of the
   status it just tested for.
2. In the `Extends:` case the plan explicitly says the earlier document *"remains authoritative"* —
   yet it gains `Superseded by:`, and §9.6 defines `superseded` as *"a successor exists;
   `Superseded by:` required"*. `m3-executor.md` was **not** superseded by `-landing2`; marking it
   so would misstate the very history §9.5 says the separate document existed to protect.

**Why it matters.** These are the two pointers that make the successor branch navigable. If they
are contradictory, an implementer either skips the back-pointer (the family becomes one-directional
and `Superseded by:` stays unused, exactly the B2 defect) or mislabels a still-authoritative
document.

**Suggested improvement.** Two clauses in §9.5 rule 5:

- *"Adding or updating a header pointer is **metadata, not an amendment** — it is the one edit
  permitted on an `archived` document."* (Also worth one clause in §9.6's `archived` row.)
- Split the back-pointers: `Supersedes:` ⇄ `Superseded by:` (earlier document also flips to
  `Status: superseded`); `Extends:` ⇄ `Extended by:` (earlier document's `Status:` **unchanged** —
  it stays authoritative).

### M8 — no legal name for a genuinely milestone-scoped topic, for a **new** document; the repo produces this class every milestone

**Evidence.** Three rules meet and give three different answers:

- §9.7/§5's prohibition: *"a new document's basename **never begins with** `m<digit>`, `k<digit>`,
  or a date."*
- §9.4: the milestone goes in `Tracks:`, *"**Never** the filename."*
- §9.5 rule 3: *"A topic slug is never reused for a different topic."*

Now name the M4 follow-ups coordination document. `m4-followups-coordination.md` is prohibited.
`followups-coordination.md` is legal once — and then rule 3 blocks it for M5, because "M5
follow-ups" is a different topic. `followups-m4-coordination.md` satisfies the *prohibition*
(nothing forbids a milestone token mid-slug) but contradicts §9.4's *"never the filename"*. §10.2
identifies this class correctly — *"its slug **is** the milestone; renaming it requires **inventing**
a topic name"* — but answers it only for the **rename** question ("don't"). The forward question is
unanswered, and the repo has already produced the class twice
(`m3-followups-coordination.md`, `m3-archive-sweep.md`).

**Why it matters.** The first document the convention is applied to after N1 may well be M4's
follow-ups ledger. An agent hitting three mutually exclusive rules will improvise — which is Finding
N1's "three schemes become four", reintroduced by the rule meant to stop it.

**Suggested improvement.** Add a fourth sentence to §9.4 or §9.5 stating the exception explicitly.
My recommendation, because it keeps the prohibition intact and stays greppable:

> *"When a document's topic genuinely **is** a milestone or a recurring per-milestone activity, the
> milestone token belongs **inside the slug, never as a prefix**: `followups-m4-coordination.md`,
> `archive-sweep-m4.md`. The prohibition is on the prefix — it is what makes the filename sort and
> read by topic — not on the token."*

### M9 — nothing tells any agent to *write* the header at creation time; only `teco` is told to flip it

**Evidence.** §9.6's header applies to all five kinds. The prompts that produce those documents
carry **complete document skeletons** and none mentions a header field:

- `claude/analyst/analyst.md`:53–61 — a four-part review skeleton ("Scope & verdict / Findings /
  What's solid / Open questions") plus a separate RCA skeleton. No header.
- `claude/architect/architect.md`:40 — the write path; the plan's structure is prose. No header.
- `claude/qa-engineer/qa-engineer.md`:29 — an itemised test-plan structure ("scope & objective ·
  references · risk assessment · test items …"). No header.
- Verified repo-wide: `grep -niE '^\s*>?\s*\*{0,2}(status|owner|tracks|version|last updated)\b'
  claude/*/[a-z]*.md` → **one hit, `tico.md`:37**.

So after N1–N3, the only prompt-level header instruction in the collection belongs to `tico`, and
the only maintenance instruction is `teco`'s flip. Every plan, review, test plan and test report
written after the convention lands depends on the producing agent noticing one clause in root
`AGENTS.md` while following a detailed skeleton in its own prompt that omits it.

**Why it matters.** This is v1.2's own M1 argument turned on the header: the plan concluded that a
component-level prompt clause beats a general `AGENTS.md` rule, rewrote `qa-engineer.md`:28 on that
basis — and then relies on the general rule alone for the field N3 exists to establish. N3
normalises 26 documents once; without a creation-time instruction the 27th document starts the drift
back, and §7 has no row for it.

**Suggested improvement** — one of, and say which:

- **(cheapest, recommended)** Add the single header line to the three document skeletons that
  already exist (`analyst.md`'s review skeleton, `qa-engineer.md`'s test-plan/report structure,
  `architect.md`'s deliverable paragraph) — *"open with `> **Status:** active · **Owner:** … ·
  **Tracks:** …`"*. Three sentences; `cobb` is already editing three of these files in N2, so the
  marginal cost is near zero and it is a *template* edit, not a new paragraph — the same pricing
  argument §3.2 makes for the filename rule.
- **or** accept the drift explicitly and add a §7 row (*"new documents omit the header — mitigation:
  root `AGENTS.md` only; re-census at the next milestone close"*), so the stakeholder is buying a
  known risk rather than an unstated one.

---

## 11. Minors and nits

**m10 — §5's table still carries the stale "optional" `teco` row.** Line 800 adds
*"`claude/teco/teco.md`:65 — **PROMOTED from optional to required**"* (N2d); line 814 still reads
*"`claude/teco/teco.md` (+ kaizen) | ***optional*** — one clause in the documentation-impact scan |
S1"*. Two rows for the same edit with opposite dispositions, in the table an implementer works from.
Delete the second.

**m11 — §7's checker row contradicts D3(a).** *"Record the baseline in HISTORY so a future sweep
compares against a number"* is the v1.1 position that M4/D3(a) retired (*"do not replace it with
anything numeric"*). Restate as *"record the baseline **in the script's own header/report**"*.

**m12 — the `Status:`-window rules disagree (three windows), and one target sits outside all of
them.** N3's done-condition reads `head -8`; N5's census says *"lacks `Status:` in its first 12
lines"*; §9.6 says *"immediately under the H1"*. `docs/plans/cpg-query-access.md` — one of the 8 to
normalise — carries its `Status:` at **line 11**, inside a 10-line header block. The derivation rule
("keep the existing text verbatim as a trailing clause") doesn't say whether the normalised line is
written in place at :11 or moved under the H1. Say: *"the canonical line is written immediately under
the H1; an existing `Status:` elsewhere in the header block is folded into it"*, and use one window
everywhere.

**m13 — "The 17 to add" lists 18 paths, and one of them no longer needs adding.** Counted from the
block at lines 1494–1503: 5 + 5 + 4 + 4 = **18** (17 + the m1 false-positive document, which the
table counts on a separate row). Also, v1.2 gave *this plan* a conformant header at line 3, so the
false-positive row is already discharged: the real N3 workload is **25** documents (24 tracked, since
the review is untracked too). The `git diff --stat` expectation of 24 tracked files is correct by
coincidence of the arithmetic; the prose isn't.

**m14 — §9.1 (25 active, 8 active reviews) and §10.3/§10.5 (26, 9) disagree inside one document.**
The delta is this review's own arrival in `docs/reviews/`. Verified today: **26** active feature docs
= 12 plans + 9 reviews + 5 requirements. Add a one-line note under §9.1's table that the inventory is
as of `583e132` and §10.3 recounts with the untracked pair, or re-run the counts once.

**m15 — N2c's done-condition is line-number-addressed and breaks if the `:37` edit adds a line.**
`sed -n '71p' claude/tico/tico.md` compares by position; adding `Owner:`/`Tracks:` on their own
line(s) shifts `:71` and the check fails while the gate is intact (or, worse, passes on the wrong
line). Use content: `git diff HEAD -- claude/tico/tico.md | grep -E '^[-+].*Ready for design'`
returns nothing, and `git diff HEAD -- claude/README.md` is empty.

**m16 — the `Status:` derivation rule will mark live work "do not execute or amend."** *"`archived`
if the document's milestone is closed per `HISTORY.md`"* applied by `coder` to
`docs/plans/cpg-query-access.md` — whose own line 11 says *"approved … **→ in implementation**"*, and
whose milestone (M3) closed at `583e132` — yields `archived`, i.e. "frozen record; do not execute."
Same hazard for `m3-followups-coordination.md`, whose follow-ups may outlive M3. Suggest: derive
`archived` from *"the document's own work is complete (its backlog item is closed in `BACKLOG.md`)"*,
default to `active` when in doubt — and, since §10.3 already lists all 26 paths, **write the expected
token beside each one**. That converts 26 judgement calls into a reviewable list and costs the
architect ten minutes.

**m17 — `qa-engineer.md`:54 is a second override clause the §3.2 grep missed.** *"**Match the
project.** Discover and follow each component's framework, runner, file layout, naming, and **doc
conventions** … Read the component's `AGENTS.md` first."* It doesn't match the plan's pattern
(`follow that|detect the convention|discover them`) but does the same work as `:28`. It does **not**
disturb M1's rejection — it is in the file already being rewritten — but N2a should cover it, or the
rewritten `:28` is contradicted from 26 lines below.

**m18 — the D6 fork and N3's independence aren't reconciled.** §8 says D6 gates N1+N2 and that N3 is
a consequence of the accepted D4. But N3's vocabulary and its done-condition regex are defined in
**§9.6**, which lands in root `AGENTS.md` only via **N1**. If D6 = no, N3 ships 25 headers speaking a
vocabulary that is written down nowhere normative. Say explicitly that **the §9.6 header block half
of N1 is D4-consequent and ships regardless**, and only the filename grammar / role set / collision
rules are D6-gated.

**m19 — the citation rule inherits the M1 problem the plan fixed for filenames.** Forward-only
root-anchoring is asserted in one place (root `AGENTS.md`) against a corpus where the *opposite*
spelling dominates the files agents read first — I count **15** module-anchored backticked `docs/…`
refs in `falkor-chat/AGENTS.md`, **59** in `falkor-chat/docs/HISTORY.md`, **64** in
`falkor-chat/docs/BACKLOG.md`. §7 does rate "nobody follows the citation rule" Medium, so this is a
strengthening note rather than a new risk. Cheap, proportionate mitigation the plan doesn't take:
**normalise the ~15 in `falkor-chat/AGENTS.md` during S1, which already edits that file's line 112**
— it is live guidance an agent reads before writing, not a dated record, so the O-2 "correct as
written" argument doesn't protect it.

---

## 12. What's solid in v1.2 — verified

- **B1's correction is complete and honest.** The reversal of D1 is the hardest kind of revision to
  make — the plan's headline recommendation, undone by its own evidence — and §6.1 states the
  reasoning error in its own voice rather than burying it. 4/4 composed, 143, 3/143, no surviving
  downstream dependency: all re-measured.
- **B2's variation beats what I proposed.** Withdrawing the role-ordinal hatch so one rule replaces
  two makes the grammar smaller, and `x2-impl.md` does read correctly. The `landing2` re-derivation
  is right.
- **B3's absorption is the right resolution of a product/technical boundary.** Absorbing verbatim
  means nobody has to rule on `tico`'s gated semantics — the fork I escalated is *dissolved*, not
  answered, which is strictly better. N2c's byte-identical done-condition is the correct proof
  obligation and I would keep it (with m15's fix).
- **The three corrections aimed at me are all correct**: the extglob pattern (ran it), the
  "only document-filename override" grep (re-grepped wider), and the 10-illustrative-links hazard
  (my own pass finds 13 — the point holds a fortiori).
- **The decision set is now fair and honest.** D1 and D6 are genuine stakeholder forks with
  recommendations and stated reversibility; D2/D3 are cheap technical calls with a stated default if
  no ruling arrives; D5 is closed on evidence; D7 is folded into the decision it was always a
  consequence of; **D8 is recorded as the architect's own call with its four reasons and flagged for
  objection** — which is exactly the right handling and I would not reopen any of them. Nothing here
  is a technical question escalated to look consultative.
- **Unchanged and still endorsed:** D4, rename-nothing (§10.1–§10.2), the S5 deferral, the rejections
  of A′/B/C and un-archiving, the §9.7 guard analysis (in the direction it tested), and the §10.5
  do-nothing comparison, which remains admirably willing to argue against its own plan.
- **N3 is still correctly the highest-value step.** With D4 accepted, `Status:` is the only lifecycle
  signal; today it covers 8 of 26 documents in 8 vocabularies (re-verified file by file). The
  extension from backfill-17 to normalise-26 is what makes the payoff real, it remains content-only
  and zero path strings, and its gate is a genuine self-check — once B4 and m12/m16 are fixed.

**On proportionality.** v1.2's cost is up (4 prompts + 4 kaizen pairs + 4 README re-checks + 26
headers, vs. "two sentences"), and the plan states this up front instead of hiding it — the increase
is honest and each increment traces to a defect I raised. In absolute terms it is still one
paragraph, ~5 sentences, 26 one-line headers and 3 deleted tokens, against a measured ~8 path edits
per archived document. **For a cost-sensitive stakeholder this is still clearly proportionate**, and
M9's fix (three template sentences) is the last increment I would spend. If the stakeholder wants to
spend less, the honest place to cut is **D6** (the naming grammar) — not N3, and not the citation
rule.

---

## 13. What may be implemented, and in what order

Verdict is **needs changes**, but the blockers are narrowly scoped. Split explicitly:

**Approved to implement as written, now** *(no dependency on B4/B5)*:

1. **S1 + N1 + N2a + N2b — one commit.** Root `AGENTS.md` (status-marker rule, citation rule,
   filename grammar **as a prohibition**, role set, collision rules, 3-field header),
   `falkor-chat/AGENTS.md`:112, `qa-engineer.md`:28 (rewrite — fold in **m17**'s `:54`),
   `analyst.md`:51 (`-impl`). *Carry M6/M7/M8's rule-5 clarifications into the `AGENTS.md` text if
   they land first; otherwise N1 can state rule 5 as §9.5 has it and be patched, since it is prose in
   one file.*
2. **S2 + N4** — HISTORY entries (reproducible metrics only), the three `../` deletions, the false
   "empty active directories" correction, the k031 nit filed.
3. **S3** — the 3 forward-looking rot repaths.
4. **S4** — optional, per D2; if built, it must carry the stated placeholder-exclusion rule.

**Blocked until the plan is amended** *(both fixes are minutes of editing, not redesign)*:

- **N3** — blocked on **B4** (canonical form vs. `tico`'s unbolded `Status:`), and should carry
  **m12** (one window, where the line goes), **m13** (18-vs-17, and this plan already conforms) and
  **m16** (expected token per file) in the same amendment.
- **N2c** — blocked on **B4**'s scope decision (does `:37`'s template bold its labels?) and should
  take **m15**'s content-based done-condition.
- **N2d** — blocked on **B5** (who actually performs the flip). If the routing answer is
  "each kind's owner", N2d's wording changes and no hook is touched.

**Verify during implementation** *(no plan change needed)*:

- The S1 diff contains **zero** `*.md` path-string edits, and S2's contains exactly three deletions.
- `bash claude/scripts/audit-team.sh` stays at the C-309a baseline — re-confirmed today: **FAIL, 2**
  (username leak + home-path leak). No new FAILs.
- `claude/tico/tico.md`:71 and `claude/README.md`:8 unchanged (content-matched, per m15).
- After N3, the anchored loop prints nothing **and** a spot check of three documents shows the
  pre-existing free text preserved after the canonical token (§9.6 note 3's non-destructive claim).

**Recommended before the next round, not blocking:** M6, M7, M8 (three sentences in §9.5/§9.4 — they
harden rules that ship into `AGENTS.md` in step 1, so landing them first is cheaper than patching),
M9 (three template sentences or a §7 row), and the minors.

---

## 14. Open questions for the caller / stakeholder

1. **B5's routing choice is partly a policy question**, not purely technical: widening `teco`'s
   write guard is a permanent loosening of a deliberately narrow guardrail. My recommendation
   (each kind's owner flips its own documents, `teco` coordinates) avoids the question entirely, but
   if you prefer `teco` doing it directly, the guard edit is yours to approve.
2. **D1 is still yours and only yours** — do you read these documents on github.com? v1.2's (a) is
   the right default and is the reversible direction; nothing else in the plan waits on it.
3. **D6 is the one place to cut cost if you want to.** N3 and the citation rule stand on their own;
   declining the filename grammar loses the `-impl` fix and lets a fourth naming scheme appear, and
   the plan is honest that nobody is blocked by `m3-executor.md` today.
4. **M9's fix costs three sentences in three prompts.** If you would rather not grow the prompt
   surface further, say so and the answer is the §7 risk row instead — but the header will then
   decay from the day N3 lands, and that should be a decision rather than a surprise.

---

*Part II routes to `architect` for a v1.3 amendment. Nothing was fixed in place; the plan document
is unmodified, and Part I is preserved verbatim as the audit trail.*
