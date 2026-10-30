# Review — Document size discipline (requirements)

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Scope & verdict

Reviewed `docs/requirements/document-size-discipline.md` (Last updated 2026-10-29, `Status: Interviewing`)
at requirements altitude, against root `AGENTS.md` "Module documentation convention" (lines 160–225)
and `claude/docs/plans/prompt-waste-reduction.md` Stage G. I did not read the connected-code-graph
family. Spot-checked: requirements 8,995 w and `review-techniques.md` 20,008 w both match.

**Verdict: needs changes.** Two blockers. FR-1 as written also binds `HISTORY.md`. FR-3/AC-2 collide
with the `## Pass N` rule for reviews, and the doc does not say how that is resolved. Both need a
stakeholder answer, not an editorial fix. The rest is sound and close to ready.

CPG: not applicable — requirements/process document with no code-level component.

## Findings

### B1 (blocker) — FR-1 is unscoped and captures every never-archived lookup document

FR-1 says "a document that is not yet `archived` is treated as read whole". The decision log (line 128)
widens this to "a non-archived document **of any kind**". `HISTORY.md` and `QUERIES.md` are never
archived, and `AGENTS.md` line 210 requires "append an entry per delivered change". Under FR-1 they
would become read-whole and bound. The same goes for active `test-reports/`, which "Out of scope"
excludes. FR-2 limits FR-3–FR-6 to three kinds, but FR-1's reclassification carries no such limit.
AC-1 then requires that every convention "reflects FR-1", so a designer would have to rewrite the
lookup row for all kinds.
**Fix:** scope FR-1 to the FR-2 kinds: "an in-scope document (FR-2) that is not yet `archived`…".
Then say explicitly that `HISTORY.md`, `QUERIES.md` and out-of-scope kinds keep the Stage G
classification. Correct the decision-log entry too, or add a one-line follow-up entry.

### B2 (blocker) — FR-3/AC-2 vs the reviews `## Pass N` rule is a real conflict, left unresolved

`AGENTS.md` lines 214–222 have three relevant rules. A review revises in place "regardless", gets a
dated `## Pass N` section, and "a re-review's value is pass 1 and pass 2 read together". A later pass
gives one disposition line per earlier finding. The requirements pull against this in three places:
- AC-2 says "no new section that exists only to record the revision". A Pass N section records a
  revision of the gated document, so it fails a literal reading.
- FR-3 says rework "replaces the affected text in place". A Pass 2 that re-dispositions Pass 1 findings
  is rework of existing content. Taken literally, it replaces Pass 1, which destroys the side-by-side
  value the convention protects.
- FR-4 moves dispositions and stakeholder questions *into* the review, so reviews grow by design.

The compact-Pass-N rule may already be the intended "growth stops" for reviews: a new pass is new
scope, and earlier findings shrink to one line each. But the doc never says this, and the two readings
lead to opposite designs. This is a stakeholder decision (Q1).
**Fix:** add an FR, or a clause in FR-3, that states the review rule outright. Either "a new review pass
is new scope; the existing compact-Pass-N rule is its growth limit", or the alternative the
stakeholder picks. Also cite `AGENTS.md`'s reviews exception in the "Prior decision" paragraph.

### M1 (major) — FR-4 names a home for dispositions but no writer, and the writer may be guard-blocked

Today the plan author writes dispositions in the plan (§2 tables). FR-4 moves them to the review. The
architect's write guard (`claude/architect/hooks/guard-plan-doc-writes.sh:13`) allows only
`docs/plans/*`, so the author cannot write there without an escalation. The other option is that the
reviewer records dispositions in Pass N, which matches the existing rule. But then the author's
reasons ("handled F3 by X, deliberately not Y") have no home, because FR-4 also bans them from the
plan. Note that `docs/reviews/connected-code-graph-p0-impl.md` exists alongside the family, so this
pattern is live. This needs an answer (Q2).

### M2 (major) — FR-4 bans stakeholder questions from plans, but they need somewhere to go before any review exists

A first-version plan raises open questions before any review document exists. FR-4/AC-4 leave those
questions with nowhere to go. Q3.

### m1 (minor) — Wrong section cited for Stage G

FR-1 and the Prior-decision paragraph cite `prompt-waste-reduction.md` §3. §3 is "The editing
doctrine". Stage G is under §4, at line 801, and its read-whole/read-by-lookup table is at lines 819–823.
**Fix:** cite "§4 Stage G".

### m2 (minor) — FR-3 and FR-8/AC-2 use different growth thresholds

FR-3 says rework "does not add **material** size". FR-8 and AC-2 tell the reviewer to report "**any**
growth" from rework. Pick one. "Material" plus reviewer judgment fits the no-number decision.

### m3 (minor) — FR-5/AC-5 "one line" is ambiguous, and the log handles reversals poorly

Most entries here take 2–3 source lines because of 100-column wrapping. So AC-5's "single line" needs
a definition: one list item, or one physical line? The log is append-only, so reversed decisions
stay with stale cross-references. Line 131 says ledgers are in scope "(FR-2)". Line 136 says "Review
and ledger only (FR-4)". Both are false now. Consider requiring a superseded entry to point at its
replacement, for example "→ reversed below".

### m4 (minor) — The size-guidance claim in "Problem & current state" is slightly overstated

"No agent prompt … carries any size guidance" is inaccurate. `claude/analyst/analyst.md:66` caps each
finding at about 15 lines, and `AGENTS.md` carries the compact-Pass-N rule. The point still holds for
*document growth*. Narrow the claim to "no guidance on revision growth".

### n1 (nit) — AC-9 comes before AC-8

Renumber the ACs, or reorder them.

## FR-8 self-check (light)

- **Restated facts:** the "Prior decision" paragraph paraphrases the Stage G rule. It quotes and cites
  it, so this is acceptable.
- **History beyond one-line log entries:** none outside the log.
- **The log itself:** it partly fails its own FR-5. Entry 1 ("option 1 of three offered") and entry 2
  (a narrative prior-decision check with rationale) are not question→answer. The final entry is
  status ("→ pending"), not a decision.

AC-1 and the AC-2–AC-9 mapping: every FR has an AC (FR-1→AC-1, FR-3→AC-2/3, FR-4→AC-4, FR-5→AC-5,
FR-6→AC-6, FR-7→AC-8, FR-8→AC-7, FR-9→AC-9; FR-2 is a scope FR that the ACs apply). No solution
language beyond naming the gate owner (`analyst`), which is a process fact, not a design.

## What's solid

- The "growth stops, no number" framing is consistent across the Intent, FR-8, AC-2, Out of scope and
  the log.
- Ledgers are dropped consistently from FR-2, FR-4 (questions go to the review only) and Out of scope.
  The only stale traces are in the append-only log (m3).
- FR-7 and AC-8 (trim at the next ordinary revision, archived documents untouched) are clear and cheap.
- The problem evidence that I spot-checked reproduces.

## Stakeholder questions (leverage order)

1. **Reviews:** is a new `## Pass N`, compact under the existing rule, "new scope" that is allowed to
   grow? Or should a review be rewritten in place to its current state, giving up "pass 1 and pass 2
   read together"? (B2)
2. **Who writes a disposition, and where does the author's reasoning go?** Does the reviewer record
   it in the next pass, or does the plan author write into the review? The second option needs the
   architect/specialist write guards widened. (M1)
3. **Where do a plan's open stakeholder questions live before its first review exists?** (M2)
4. **Confirm FR-1's scope:** only the FR-2 kinds, with `HISTORY.md`/`QUERIES.md`/test reports keeping
   the Stage G classification? (B1)
5. **Decision-log reversals:** when a later answer reverses an earlier one, should the earlier entry
   be marked? (m3)

## Pass 2

2026-10-29 — re-read the revised `docs/requirements/document-size-discipline.md` in full.

**Verdict: approve with suggestions.** Both blockers and both majors are resolved, and the fixes add
no new blocker or major finding. The two minors below can be fixed at flip time.

CPG: not applicable — requirements/process document with no code-level component.

### Dispositions

- **B1 — fixed.** FR-1 (lines 50–55) applies only to the FR-2 kinds and names `HISTORY.md`,
  `QUERIES.md` and test reports as unchanged. AC-1 (lines 96–99) now also rejects bounding anything
  outside the FR-2 kinds.
- **B2 — fixed.** FR-3 (lines 62–63) makes a compact `## Pass N` new content, and earlier passes stay.
  AC-2 (lines 102–103) exempts it. This matches the `AGENTS.md` reviews rule.
- **M1 — fixed.** FR-4 (lines 65–68) has the reviewer record each outcome in its next pass, and design
  reasons stay in the plan as design. Nothing needs to be written across the guard any more.
- **M2 — fixed.** FR-4 (lines 68–70) and AC-4 (lines 107–108) keep open questions in the plan only
  until they are answered.
- **m1 — fixed.** The citation now reads "§4, Stage G" (line 51).
- **m2 — fixed.** FR-3 now treats any growth from rework as a violation (line 62), which matches FR-8
  and AC-2.
- **m3 — fixed.** FR-5 and AC-5 (lines 71–73, 110–112) define one question and one answer plus the
  FR/AC, and require a reversing entry to name what it overturns. Line 164 applies this to the
  ledger entries.
- **m4 — fixed.** Lines 28–31 narrow the claim and cite both existing caps.
- **n1 — fixed.** The ACs are now in order AC-1 to AC-9.

### New findings

**m5 (minor) — The B1 entry does not apply the new reversal rule to itself.** Line 163 narrows FR-1 to
the FR-2 kinds. That partly overturns line 139 ("a non-archived document **of any kind**"), but line
163 does not name line 139. Line 164 names only the ledger entries. Under FR-5 and AC-5 as revised,
that is a reversal with no pointer, and a reader of line 139 still gets the old reach. **Fix:** add
"Overturns: 'Bound while active … of any kind' (narrowed)" to line 163.

**m6 (minor) — A plan's settled stakeholder answers lose their provenance.** Under FR-4 an answered
question is removed from the plan and folded into the design. The plan then says *what* was decided,
but nothing in the in-scope kinds records *that the stakeholder decided it*, or when. Only
requirements documents have a decision log, and ledgers are out of scope. A later reviewer cannot
tell a stakeholder ruling from an architect's choice. **Fix (stakeholder question):** should the
reviewer's next pass record "Q → stakeholder answer" in one line, in the same way M1 handles
dispositions? Or is the ledger, which is out of scope here, enough?

**FR-8 self-check:** line 159 (consult result) and line 166 (process choice) are status lines, not
decisions. AC-5 applies only after delivery, so this is acceptable. Line 165 sets the "regardless of
wrap" reading. Nothing else is restated.

## Pass 3 — implementation plan

2026-10-29. This pass reviews `docs/plans/document-size-discipline.md` (`active`, uncommitted)
against the requirements as approved in Pass 2. I spot-checked every edit target it cites and ran
the AC-1 probes.

**Verdict: approve with suggestions.** There are no blockers. Fix M1–M3 in place before U1 is applied.
CPG: not applicable — convention and prompt text only.

Plan Q1 (collision rule 5's one-line revision note replaces the previous one each round rather than
stacking, for requirements/plans/reviews) → stakeholder: yes, per the architect's recommendation
(2026-10-29).

**Coverage.** Every FR and AC maps to an edit:
- FR-1: E1 and E5. FR-2, FR-6, FR-7: E2.
- FR-3: E2 and E3. FR-4: E2, E3 and the architect/graph-dba/data-scientist rows.
- FR-5: E2 and `tico.md:47/66/100`. FR-8: E2 and the reviewer rows.
- FR-9: E4 and `tico.md:110`, but see M2.

E3's second clause records folded-in answers in Pass N, which also closes Pass 2 m6.

**Targets.** I checked these against the files, and all match:
- root `AGENTS.md` l.162–163 (lookup list), l.215 (rule 5), l.245 (Status set), and l.187, which is
  the open-item bullet that depends on rule 5;
- `tico.md` l.47, 66, 100 and 110, and `plan-authoring-techniques.md:51`;
- `graph-dba.md:63`, `data-scientist.md:81–82` and `security-expert.md:72`;
- the prompt word counts.

The `qa-engineer` exclusion is correct: its guard (`guard-qa-doc-writes.sh:21`) allows only
`test-plans/` and `test-reports/`.

### Findings

**M1 (major) — The AC-1 probe is too broad to run as written.** A git pathspec `*` also matches `/`,
so `'claude/*/*.md'` sweeps every `kaizen/history.md`, `claude/docs/plans/*`, reviews and test
reports. The first probe returns about 100 hits, mostly kaizen prose ("read whole before ruling").
That makes "every hit must agree with E1" impossible to check. Today it does catch the real
contradictions, `AGENTS.md:162–163` and `prompt-waste-reduction.md:821,845`, but buried among the
rest. I ran a version with the `:(glob)` magic: `':(glob)**/AGENTS.md' ':(glob)claude/*/*.md'
':(glob)skills/*/SKILL.md'` plus the plan path. It cuts the output to 18 hits. After E1 and E5 the
remaining hits are prose about something else (`analyst.md:34`, `falkordb-quirks.md:892`, …).
**Fix:** use the `:(glob)` pathspecs and list the expected non-conflicting hits, so U3 compares
against a known residue.

**M2 (major) — A second ready-for-design path in `tico` is missed (FR-9).** `tico.md:169`, the
delegated Mode 1 path, returns "the ready-for-design confirmation" after one round. It never
reaches the l.110 readback the plan edits. **Fix:** add a `tico.md:169` row to U2: report "ready
pending analyst gate" rather than ready.

**M3 (major) — E2 is over its own word cap, and the root growth figure is understated.** E2 as
drafted is 176 words, against "at most ~130". E1 adds about 15 words (45 words in, about 30 out),
and E3 plus E4 add about 40. Root `AGENTS.md` therefore grows by about 230 words, to about 3,180,
not "~160 to ~3,110". The offset mitigation ("next candidate is the rationale prose") is not tied to
any unit. **Fix:** cut E2 to 130 words or fewer. The reviewer-check sentence is repeated in the
`analyst` row and can collapse to "every review gate checks a revision against this bullet". Then
either make an offset edit part of U1 with a target, or have the human accept the growth
explicitly.

**m1 (minor) — U4 cites the wrong manual line.** `agent-team.md:335` is about the decision log, not
the flip. The only "only on your OK" is at l.115.

**m2 (minor) — D4 contradicts U2.** D4 says "no prompt edit is needed" for return messages, but the
U2 `architect` row adds "the return message lists the findings handled". Drop one of the two.

**m3 (minor) — The `teco.md:56` anchor is a weak fit.** l.56 is the analyst routing row. The
re-gate brief guidance lives at l.144, which is the better host for "re-gate runs the growth-stops
check".

**m4 (minor) — The `cobb` bookkeeping misreads the agent edit rule.** `claude/AGENTS.md:308`
requires `kaizen/{plan,history}.md`, but U2 lists only `history.md`.

**n1 (nit) — The plan does not yet comply with FR-4 for Q1.** Q1 is now answered. FR-4 and AC-4
mean the next revision should remove it from §6 and from "Ready to implement" ("Q1 is open"), and
fold the answer into D3.

**FR-8 self-check.** The plan is current-state only and cites FR and finding IDs without restating
the requirements. §6's "Pass N rule" bullet paraphrases FR-3, which is acceptable as a risk note.

### Ownership and sequencing

U1→U2 is sound: U2's pointers cite E2's bold lead, so U2 cannot land before U1. Giving U1 to the
human matches the convention's stakeholder ownership, U2 to `cobb` matches `claude/AGENTS.md:308`,
and U3 runs after both. No stakeholder questions are open.

## Pass 4

2026-10-29. Re-read the plan (Version 2) and re-ran both AC-1 probes.

**Verdict: approve.** Every Pass 3 finding is fixed or resolved by decision. The fixes add nothing
above a nit.

Pass 3 M3 (root `AGENTS.md` growth) → stakeholder: accept as is; E2 stays ~176 words, ~+230 words to
~3,180 accepted, no offset unit (2026-10-29).

### Dispositions

- **M1 — fixed.** The §5 probe now uses `:(glob)` pathspecs. Re-run, it returns 19 hits, and the
  plan's expected-residue list accounts for every one: `AGENTS.md:162–163`,
  `prompt-waste-reduction.md` l.44, l.117 and l.806–914 (eight hits), `analyst.md:34`,
  `falkordb-quirks.md:892`, `guard-testing-techniques.md:279/291` and `python-web-quirks/SKILL.md:8/758`.
  No unexpected hit.
- **M2 — fixed.** §2 (l.48–50) and a new U2 row (l.156) cover `tico.md:169`, and AC-9 now cites
  `tico.md:110/169`.
- **M3 — resolved by decision.** E2 is labelled "~176 words, accepted as is" (l.107). §2 (l.53) and
  §6 (l.192–193) both state ~+230 w to ~3,180, with no offset unit. All three agree with the
  stakeholder line above.
- **m1 — fixed.** U4 now cites only line 115.
- **m2 — fixed.** D4 (l.83–84) now assigns the return-message clause to the author rows in U2.
- **m3 — fixed.** The U2 anchor is now `teco.md:144`.
- **m4 — fixed.** U2 lists both `kaizen/plan.md` and `kaizen/history.md` (l.160).
- **n1 — fixed.** D3 records the answer, and "No questions are open" (l.206).

### New findings

**n2 (nit) — The residue list has two stale numbers.** The plan says "18 hits today", but the probe
returns 19. That was my Pass 3 miscount, which the plan inherited. Also, E5 inserts a line under
`### Stage G` (l.801), which shifts l.806–914 down by one or two lines at U3 time. **Fix:** say "19",
and cite the range as "Stage G section" rather than by line numbers, or note the shift.

**n3 (nit) — D4 says the same thing twice.** l.82 "The orchestrator already relays that message" and
l.83 "Orchestrators already relay" duplicate each other. Drop one.
