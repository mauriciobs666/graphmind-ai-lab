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
