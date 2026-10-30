# Document size discipline — Implementation review (I1 + I2)

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## 1. Scope & verdict

**Reviewed:** the uncommitted I1 and I2 diffs against baseline `66cb571c` (`git diff 66cb571c -- <path>`).
I1 is root `AGENTS.md` E1–E4 plus `claude/docs/plans/prompt-waste-reduction.md` E5. I2 is the
pointer edits in 8 prompt/technique files plus `kaizen/{plan,history}.md` for 7 agents. The
baselines are plan `docs/plans/document-size-discipline.md` v2 (§4 U1/U2, §5),
`docs/requirements/document-size-discipline.md` (FR-1..FR-9, AC-1..AC-9),
`docs/reviews/document-size-discipline.md` Passes 1–4, and the I3 done-condition in the ledger.
`docs/plans/connected-code-graph-*` is excluded, because it belongs to another session.

**Verdict: needs changes.** It has one major finding, and the fix is one sentence in `tico.md`, owned by `cobb`.
I1 is verbatim and correctly placed. Every U2 row landed within budget, and both of `cobb`'s
deviations still carry their FR. The major is a pre-existing line in `tico.md`, next to the
rewritten l.66, which still tells `tico` to hand-edit decision-log lines and narrate the "why".
That contradicts FR-5 and the new l.66. The plan's second probe was meant to catch this kind of
line, but its regex misses the hyphenated form "decision-log". A short Pass 2 on the fix is
enough; nothing else needs re-gating.

CPG: not applicable — convention and prompt text only; no code-level component.

## 2. Findings

### M1 (major): `tico.md` l.67 contradicts FR-5 and the new l.66 → `cobb`

**Evidence.** `claude/tico/tico.md:67` ("A terminology reversal needs a hand edit…"): when a
`replace_all` would mangle "a terminology note or decision-log line", tico should "Rewrite those
sentences by hand to narrate the change (what it was, what it is now, why)". This line was
untouched by I2 (origin `b23a4fc2`). It sits directly under the rewritten l.66 ("in the
template's shape with no narrative"). It conflicts with root `AGENTS.md`'s new bullet "A
not-yet-`archived` … stops growing across revisions" (an entry is one question → one answer, and
a reversal is a new entry naming the one it overturns). It also conflicts with FR-5 ("earlier
entries are never edited", "no narrative or rationale").

**Why it matters.** At the next terminology reversal in an interview, tico follows l.67, edits
earlier log entries, and adds rationale. AC-5's behavioural check then fails on the first log
that exercises this path. The plan §5 second probe (`decision log`) does not match
`decision-log`, which is why the miss went unflagged. Plan §2's "Live conflicts in prompts" also
missed it (plan grounding gap; noted for `architect`, but no plan revision is needed).

**Fix (≈ +10 w net).** Replace l.67's last sentence with: "Never `replace_all` into decision-log
entries: they are append-only, and the reversal is a new entry naming the one it overturns (root
`AGENTS.md`, growth-stops bullet). Rewrite any other meta-sentence (a terminology note) by hand to
state the current term, and only `replace_all` the product-noun occurrences elsewhere."

**Acceptance assertion.** `git grep -n -i -E 'decision[- ]log' -- claude/tico/tico.md` returns
no line that tells tico to edit an existing entry or record a "why".

**Same sweep (nit).** `tico.md:114` says "log the closing decision-log line" when archiving. That
is an entry with no question → answer shape. Phrasing it as a Q→A entry costs nothing.

### m1 (minor): architect's bold lead "never appends" overstates the rule → `cobb`

**Evidence.** `claude/architect/architect.md:56`: "**A revision rewrites, never appends**". Root
`AGENTS.md`'s growth-stops bullet says "Only new scope adds words", and FR-3 and AC-3 allow
growth for new scope. **Why it matters:** if a stakeholder answer opens a new design area, the
architect is told never to add a section. It would then either suppress the content or cram it
into an unrelated section, and both damage the plan. The U2 row had no bold lead; the lead is
`cobb`'s addition. **Fix:** "**Rework rewrites in place; only new scope adds**" (same citation).
The body sentence can stay.

### m2 (minor): `tico.md:47`'s consult-entry shape has no FR/AC slot → `cobb`

**Evidence.** `tico.md:47` prescribes `Consult: <agent> on <question> → <result>`, following plan
D5. The template at l.100 and the growth-stops bullet require "plus the FR/AC it affects", and
AC-5 says "every decision-log entry … (with the FR/AC affected)". **Why it matters:** two rules in
the same prompt prescribe different shapes for entries in the same log. Every consult entry
literally fails AC-5. **Fix:** `Consult: <agent> on <question> → <result> (FR/AC affected)`.
**Also on this line (nit):** "state the same three things" lost its referent, because the old
"who was asked, why, and what came back" was replaced. It now has to map onto
agent/question/result. Say "the same entry's three parts".

### m3 (minor): two gate-running paths have no pointer → `cobb` (plan §2 omission → `architect`, FYI)

**Evidence.** `graph-dba` writes gate sections into `reviews/` documents (e.g. the
spec-amendment gate sections in `docs/reviews/connected-code-graph-p0-impl.md`, commit
`363b5b1b`). `tico.md`'s "Coordinating a docs-only chain → Review gates" bullet dispatches
revisions and re-gates; this plan's own P1–P4 ran there. Neither has a pointer. The plan §2
writer table lists `graph-dba` only for `-graph.md`, and `tico` only for `requirements/`.
**Why it is only minor:** the growth-stops bullet is always loaded and already says "Every review
gate checks a revision against this bullet". Author and reviewer prompts (`architect`, `analyst`)
also carry their own clauses, so FR-8 is not lost. The pointers are reinforcement, and teco's
l.144 has one. **Fix (optional):** one clause each, about 17 w, mirroring
`security-expert.md:72`.

### n1 (nit): teco's sentence splits its host bullet's argument → `cobb`

`claude/teco/teco.md:144` places the growth-stops sentence between "Brief the re-gate to check
same-revision fixes against each other." and "A delegate reporting belt-and-braces is the signal
to check the interaction…". The last sentence now reads as if it follows from the growth-stops
clause. Move the new sentence to the end of the bullet.

### n2 (nit): the label "growth-stops bullet" is not in the bullet → `teco` (plan wording, `architect` if changed)

Every pointer cites "growth-stops bullet", and E3/E4 use the same label. The bullet's bold lead,
though, is "…stops growing across revisions". `grep growth-stops AGENTS.md` lands on E3 (l.231)
and E4 (l.264), not E2 (l.173). Plan D1 says to cite "by its bold lead". In practice it resolves,
so take it or leave it.

## 3. Gate checks

**G1 — AC-1 inventory (plan §5 `:(glob)` grep).**
- **Recomputed: 20 hits, not 19.** The baseline at `66cb571c` returns 19, by file: AGENTS 2 ·
  analyst 1 · prompt-waste 11 · falkordb-quirks 1 · guard-testing 2 · python-web-quirks 2.
  E1's new sentence adds `AGENTS.md:165` ("read whole"), so the ledger's "19" is the pre-I1 count.
  This is not a defect; `teco` should correct the ledger's prior.
- **Every hit is accounted for:**
  - **E1 text:** `AGENTS.md:162/164/165`.
  - **Stage G section of `prompt-waste-reduction.md`, under E5's pointer:** l.809, 823, 824, 826,
    848, 866, 872, 915, 917. The section runs from l.801 to EOF, with no later heading.
  - **Other topics in `prompt-waste-reduction.md`:** l.44 (the plan's goal statement) and l.117
    (the "widen the list" technique).
  - **Off-topic prose:** `analyst.md:34`, `falkordb-quirks.md:892`,
    `guard-testing-techniques.md:279/291`, `skills/python-web-quirks/SKILL.md:8/758`.
  - **No unexplained hit.**
- **Second probe** (`revision note|decision log|Ready for design|Pass N`, same paths): every hit
  is consistent with E2–E4 and the U2 rows. `prompt-waste-reduction.md` l.578–585 still says "one
  dated line", which E3 keeps. The `claude/AGENTS.md:306` flow is unaffected.
- **Widened probe** (hyphen/space variants, e.g. `decision[- ]log`): finds `tico.md:67` (M1).

**G2 — FR → carrying text.**

| FR | Carried by |
|---|---|
| FR-1 | Root `AGENTS.md`, bullet "A living document is compacted at milestone close" (E1: by-lookup list + "also counts as read whole" sentence); `prompt-waste-reduction.md` `### Stage G` pointer line (E5) |
| FR-2 | Root `AGENTS.md`, growth-stops bullet's bold lead ("`requirements/`, `plans/` (not `-coordination`) or `reviews/`") |
| FR-3 | Growth-stops bullet ("Only new scope adds words. Rework replaces…"); collision rule 5 (E3a: revision note replaces previous line); compact `## Pass N` rule; pointers: `architect.md` Guardrails, `graph-dba.md` How-you-work 7, `data-scientist.md` method-note deliverable, `plan-authoring-techniques.md` "Reconciling a diverging delegated note", `teco.md` step-5 re-gate bullet |
| FR-4 | Growth-stops bullet ("A plan or specialist note states the current design only…", open-question sentence, "next `## Pass N` records outcomes"); rule 5 compact-pass clause (E3b: question → answer); return-message clause in `architect.md`/`graph-dba.md`/`data-scientist.md`; `analyst.md` deliverable item 2 |
| FR-5 | Growth-stops bullet (decision-log sentences); `tico.md` Traceability (l.47), "Write as you go" (l.66), Decision-log template (l.100). **Contradicted by `tico.md:67` (M1)**; shape mismatch m2 |
| FR-6 | Growth-stops bullet ("A fact has one home, and other documents cite it"); D1 pointer pattern (prompts cite, not restate) |
| FR-7 | Growth-stops bullet, last sentence ("An existing document complies at its next ordinary revision") |
| FR-8 | Growth-stops bullet ("Every review gate checks… no numeric threshold"); `analyst.md` item 2; `data-scientist.md` methodology-review deliverable; `security-expert.md` "Your deliverable: the review"; `teco.md:144` re-gate brief (coverage gap m3, mitigated) |
| FR-9 | Root `AGENTS.md` closed `Status:` set (E4); `tico.md` "Close with a readback" (l.110) and isolated-context Mode 1 (l.169); `analyst.md` item 2 |

**Deviations from the U2 table:**
- **(a) analyst clause appended to item 2:** carries FR-4/FR-8/FR-9. All four required actions
  are present. Dropping "always" loses nothing, because the clause is unconditional. Placement
  inside "Findings" is apt, since the check's output is findings. Accepted.
- **(b) tico decision-log rule split between l.66 and the l.100 template:** carries FR-5. Each
  piece lives in one place:
  - the template gives the dated shape, the FR/AC slot, append-only, and the reversal rule;
  - l.66 gives "no narrative" and "rework replaces", and points at the template.
  This follows FR-6 better than stating the rule twice. Accepted, subject to M1 and m2.

**G3 — growth-stops check applied to the edits.**
- **New scope vs. rework:** all growth is new scope (a new rule plus its pointers). No restated
  history in any prompt or in `AGENTS.md`. Kaizen `history.md` entries are by-lookup logs, so
  their narrative is expected.
- **Restatement:** the pointers cite the bullet and state only their own action. The partial
  echoes (`analyst.md` "new scope or rework", `tico.md:66` "rework … replaces") are the
  per-role actions the U2 rows prescribed.
- **Net words per U2 row** (porcelain word-diff, all ≤ ~50):

  | Row | Net words |
  |---|---|
  | architect | +42 |
  | plan-authoring-techniques | +1 |
  | graph-dba | +39 |
  | data-scientist (2 rows) | +38 / +17 |
  | security-expert | +20 |
  | analyst | +47 |
  | tico (5 lines) | +7 / +16 / +9 / +12 / +15 |
  | teco | +24 |

- **Prompt bodies (wc):**

  | Prompt | Before → after |
  |---|---|
  | architect | 1,537 → 1,579 |
  | graph-dba | 1,814 → 1,853 |
  | data-scientist | 2,046 → 2,101 |
  | security-expert | 2,516 → 2,536 |
  | analyst | 2,713 → 2,760 |
  | tico | 5,927 → 5,986 |
  | teco | 8,272 → 8,296 |

- **Root `AGENTS.md`:** 2,950 → 3,190 w (+240 vs. ~230 planned; accepted, plan §6). Per edit:
  E1 +25 (exactly the ≤ +25 cap), E2 +177, E3 +10/+16, E4 +12. No line > 700 chars.

**I1 verbatim-ness and placement.**
- E1, E2 and E5 were compared programmatically against the plan quotes (whitespace-normalized)
  and are identical.
- E3a, E3b and E4 were checked by eye against the word-diff, and match.
- Placement: E2 sits right after E1's bullet and before the `BACKLOG.md` bullet. E5 sits directly
  under `### Stage G`. The diff touches only those five spots. The bolded "one dated line, not a
  narrative" survives, appearing twice: rule 5 and the "open item" bullet.

**G4 — AC-8.** No touched file has `Status: archived` in its header. The only header blocks are
`prompt-waste-reduction.md` (`active`) and the ledger (`active`). All other `grep -m1 Status:`
matches are body prose. Pass.

**G5 — cross-edit interactions.**
- **E3a vs. the "open item" bullet:** consistent. E3a is scoped "under the growth-stops bullet",
  so it applies only to FR-2 documents. The open-item bullet (about `BACKLOG.md` items) still
  shares the exact phrase, which keeps "deliberately the same words" true.
- **E1 vs. the rest of its bullet:** consistent. The by-lookup list no longer names `reviews/` or
  closed `plans/`, and the added sentence hands FR-2 documents to the next bullet before the "So
  at milestone close…" closeout. (Readability: "So" now follows the exemption sentence. Harmless.)
- **E4 vs. `tico.md:110/169`:** consistent. E4 says the flip happens "after" the analyst review,
  and l.110 requires readback **and** review. Neither orders readback against review, and both
  put the review before the flip, which satisfies AC-9. l.169 defers to l.110 by section name.
  `plan-authoring-techniques.md:51` ("updating the header's one-line revision note") agrees with
  E3a.
- **Interactions found:** l.66 vs. l.67 (M1) and l.47 vs. l.100 (m2).

**G6 — `claude/scripts/audit-team.sh`.** Re-run here, it ends `RESULT: FAIL` with exactly five
FAIL lines. All five are the personal-identifier class: user.name, user.email, hostname,
username, home path. They are in `model-bench/packs/…`, `opencode/docs/test-reports/…`,
`claude/docs/plans/bypass-permissions-subagent-gap-coordination.md`, and similar files. No I1/I2
file appears in any FAIL listing, and a direct identifier grep over all touched files finds
nothing. The NOTE lines (`analyst`/`teco`/`tico` above 2,500 w) are advisory. This confirms the
ledger's 2026-10-30 note.

## 4. What's solid

- I1 is verbatim to the approved plan, in the right places, and within its word constraints.
  E1 lands exactly on its +25 cap.
- The U2 pointers follow D1: each one is a citation plus an action, and none restates the rule.
  Where possible the edits rewrite existing lines (all five `tico` edits do, at +59 w total
  against a 5,900 w prompt).
- Kaizen bookkeeping is complete for all 7 agents (a history entry plus a parking-lot
  first-live-use check, which fits D6's judgment-based enforcement). No frontmatter changed, so
  `claude/README.md` correctly stays untouched.

## 5. Open questions

- **FR-9 and the gate's verdict (for the stakeholder, via `tico`; not a defect in I1/I2).** E4,
  `tico.md:110` and AC-9 require that an analyst review *exists* before the flip. None requires
  the review to *pass*, so a "needs changes" review literally satisfies the gate. The
  requirements' own flip waited for an approving Pass 2, so the intent seems clear. Should the
  rule say "an `analyst` review whose verdict is not needs changes"?

## Pass 2

*2026-10-30. Re-checked against the tree, baseline `66cb571c`. Scope: `cobb`'s fix round, plus the
stakeholder's verdict-condition answer in root `AGENTS.md`, `tico.md` and
`docs/requirements/document-size-discipline.md`.*

**Verdict: approve with suggestions.** No blocker or major remains. There is one new minor (p2-m1)
and two new nits.

CPG: not applicable — convention, prompt and requirements text only.

**Pass 1 dispositions**
- **M1 — fixed.** I ran the Pass 1 assertion
  `git grep -n -i -E 'decision[- ]log' -- claude/tico/tico.md`. It hits l.47/66/67/68/114/169/171,
  and none of them tells `tico` to edit an entry or record a "why". l.67 now reads "Never edit an
  earlier decision-log entry: a reversal is a new entry in the template's shape". `cobb`'s
  shorter wording holds: the template at l.100 itself carries "Append-only; a reversal names the
  entry it overturns", so the pointer loses nothing.
- **M1 same-sweep nit (l.114) — fixed.** It now reads "closing decision-log entry in the
  template's shape".
- **m1 — fixed.** The lead at `architect.md:56` is now "Rework rewrites in place; only new scope
  adds".
- **m2 — fixed.** `tico.md:47`'s shape now ends `(FR/AC affected)`, matching l.100. Its nit is
  fixed too: the line now says "the same entry's three parts".
- **m3 — not fixed, by decision (`teco`).** The root bullet is always loaded, and `tico`/`teco`
  are over budget. This is acceptable, because Pass 1 had already rated it reinforcement only.
- **n1 — fixed.** The `teco.md:144` sentence now closes the bullet.
- **n2 — not fixed.** Take-or-leave; it stands as is.
- **Open question → stakeholder answer (2026-10-30):** Does the flip need only an existing
  analyst review? → No: the review's latest verdict must not be "needs changes" (FR-9, AC-9).

**Verdict-condition texts against each other**
- **The three texts:**
  - root `AGENTS.md` `Status:` set: "whose latest verdict is not "needs changes"";
  - `tico.md:110`: "whose latest verdict is not "needs changes"";
  - FR-9/AC-9: "approve or approve with suggestions, never needs changes".
- **They are equivalent.** The analyst verdict scale is closed at three values (`analyst.md`
  item 1), so "not needs changes" and "one of the other two" are the same set.
- **No ordering conflict.** E4 says "after" the review, and AC-9 says "otherwise the flip does not
  happen".
- **l.169 is consistent.** It defers to l.110 by section name, so leaving it unchanged is correct.
- **On restating instead of citing:** `tico.md:110` repeats the condition rather than citing the
  root bullet. That is the D1 pattern (the agent's own action), and the root `AGENTS.md` and
  prompt texts are outside FR-6's in-scope kinds, so it is not a finding.
- **Plan drift.** Plan §4 E4 now differs from the live root text. The requirements doc is the
  authority, and the plan has been executed, so this is a note for its archival, not a finding.

**Growth-stops check on the requirements change** (first live use on a requirements doc)
- **Size:** 2,024 → 2,097 w (+73), split as FR-9 +18, AC-9 +10 and the new log entry +45 (word-diff; the `wc` total is +73).
- **New scope vs. rework:** all of it is **new scope**. The stakeholder added a condition that
  did not exist before.
- **Placement:** FR-9 and AC-9 were rewritten in place, not given an addendum. No section exists
  only to record the revision, there is no restated history, and nothing else changed. Compliant.
- **Decision-log entry against FR-5:** it has one question, one answer and the FR/AC affected,
  and it names the entry it refers to. Compliant.
- **The "Overturns, by narrowing" label** is accepted, with nit p2-n2 below. It reuses this log's
  own precedent (the 2026-10-29 "Correction (Pass 2 m5)" entry). Naming the related entry is
  harmless.

**New findings**
- **p2-m1 (minor) — the requirements header's `Last updated:` is stale → `tico`.**
  - **Evidence:** `docs/requirements/document-size-discipline.md` l.3 still reads
    `Last updated: 2026-10-29`, but FR-9, AC-9 and the log changed on 2026-10-30. Root
    `AGENTS.md` assigns this field to `tico` ("`tico` keeps it").
  - **Why it matters:** the header is the only place that dates the document's current text. A
    reader comparing it against the review would see a requirements doc older than the gate that
    approved its change.
  - **Fix:** bump it to 2026-10-30.
- **p2-n1 (nit) — the new root `AGENTS.md` line is not re-wrapped → `teco`.** `AGENTS.md:264`
  is 136 chars, while the surrounding text wraps at about 100. This is cosmetic only; it is well
  under the 700-char bar.
- **p2-n2 (nit) — "Overturns" is slightly strong → `tico`.** The 2026-10-30 entry adds a
  condition to the "Analyst gate added" decision, and that decision still stands. FR-5's naming
  duty is for reversals, so "Refines" would be more accurate than "Overturns, by narrowing".
  Take or leave.

**AC-1 re-run:** 20 hits, unchanged from Pass 1's accounting, which matches teco's re-run. Root
`AGENTS.md` is now 3,198 w (+8 from E4's condition), and no line exceeds 700 chars.
