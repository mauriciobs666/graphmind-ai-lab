# Document size discipline — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 2

*Revised 2026-10-29 in place for `docs/reviews/document-size-discipline.md` Pass 3.*

## 1. Goal & scope

Implements `docs/requirements/document-size-discipline.md` (FR-1..FR-9, AC-1..AC-9). The work is
text only: it edits the root convention, one pointer line in an earlier plan, and seven agent
prompts. It ships no tooling. Out of scope: anything the requirements' "Out of scope" lists, any
growth-measuring script (FR-8 has no threshold), and the on-demand technique files other than the
one line in §4 U2.

## 2. Context & findings

- **Stage G locus.** Only one live convention text classifies documents as read whole or read by
  lookup: root `AGENTS.md` lines 160–163, the bullet "A living document is compacted at milestone
  close". A repo-wide `git grep` for `without bound|by lookup|read-whole` finds nothing else in any
  `AGENTS.md`, agent prompt or skill. The other hit is the provenance plan,
  `claude/docs/plans/prompt-waste-reduction.md` §4 Stage G. It is `active`, owner `claude`, and has
  a table at l.819–823 plus draft amendment wording at l.842–846. The remaining hits are in
  reviews, ledgers and HISTORY prose, which are records rather than conventions.
- **Root `AGENTS.md` rules this feature touches.** Three rules: collision rule 5's "dated
  revision note — one dated line" (l.215), the compact-`## Pass N` rule (l.220–223), and the
  `Ready for design` row of the Status set (l.245). The bullet "An open item is rewritten" ties
  itself to rule 5's wording ("deliberately the same words"), so that phrase must survive.
- **Who writes in-scope documents.** Guard globs and prompt deliverable sections show these
  writers:

  | Agent | In-scope writes |
  |---|---|
  | `architect` | `plans/` (guard `claude/architect/hooks/guard-plan-doc-writes.sh`) |
  | `graph-dba` | `plans/*-graph.md` (`graph-dba.md:63`) |
  | `data-scientist` | `plans/*-ml.md` and `reviews/*-ml.md` (`data-scientist.md:81–82`) |
  | `analyst`, `security-expert` | `reviews/` only |
  | `tico` | `requirements/`, plus its own ledger (out of scope) |
  | `teco` | dispatches revisions and re-gates (`teco.md:56`) |

  Agents with no edit:
  - `qa-engineer` writes only `test-plans/` and `test-reports/`, which are out of scope.
  - `coder` and `tdd-engineer` have those directories on their escalation lists.
  - `devops`, `frontend-engineer` and `cobb` have no in-scope deliverable.
- **Live conflicts in prompts.**
  - `tico.md:47` logs a consult and then "updated once it concludes". That edits an earlier
    entry, which FR-5 forbids.
  - `tico.md:66` and the template at l.99–100 give the entry shape but no reversal rule.
  - `tico.md:110` flips to Ready for design on the readback alone, and the delegated Mode 1 path
    at `tico.md:169` returns "the ready-for-design confirmation" after one round. Neither has an
    analyst gate (FR-9).
  - `claude/architect/plan-authoring-techniques.md:51` says to rewrite in place "with a short
    revision note".
- **Word weight.** Root `AGENTS.md` is 2,950 w, against a ~2,500 w smell; U1 adds ~230 w (§6). Prompt bodies:
  `analyst` 2,713, `tico` 5,927, `teco` 8,272, `security-expert` 2,516, `data-scientist` 2,046,
  `graph-dba` 1,814, `architect` 1,537. `claude/scripts/audit-team.sh` check 9 prints an
  advisory NOTE above 2,500 w and never fails.
- **Write guards (FR-4).** Every reviewer can already write `docs/reviews/`: `analyst` and
  `security-expert` through their guards, `data-scientist` through its guard, and `graph-dba` has
  no doc guard. Authors never need to write a review.

CPG: not applicable — the change is convention and prompt text only, with no code.

## 3. Design & rationale

- **D1. One canonical home.** The rule text goes once, into root `AGENTS.md`, which every
  in-repo session already loads. Each prompt gets a pointer clause: the action that agent's role
  takes, plus a citation of the bullet by its bold lead. Rejected: putting the full rule in every
  prompt. That breaks FR-6, invites drift, and costs about 7× the words.
- **D2. Stage G is narrowed, not rewritten (FR-1).**
  - The read-whole list keeps its four living documents. One added sentence says that a
    non-archived FR-2 document counts as read whole but is bounded by the new bullet, not by the
    milestone-close list.
  - On the lookup list, `reviews/` and "closed `plans/`" become "`-coordination` ledgers and any
    `archived` document". Ledgers stay unbounded, which matches the requirements' exemption.
  - Nothing outside the FR-2 kinds gains a bound (AC-1).
- **D3. A revision note is replaced, not stacked.** This applies to FR-2 documents only. Rule 5
  keeps "one dated line, not a narrative". For in-scope documents, that line replaces the
  previous one, because the trail belongs in the review (FR-4) and a stacked line is growth from
  rework (FR-3). The stakeholder confirmed this (recorded in the review's Pass 3).
- **D4. The reviewer gets the change list from the hand-off.** The author's return message, not
  the document, lists which findings were handled and which stakeholder answers were folded in.
  The orchestrator already relays that message, so the reviewer can write its one-line outcomes
  without diffing blind. Orchestrators already relay; the author prompts (`architect`,
  `graph-dba`, `data-scientist`) get the one clause in their U2 rows.
- **D5. A consult gets one log entry, written when it concludes.** It reads "Consult: `<agent>`
  on `<question>` → `<result>`" and is appended once, never as a pending entry edited later
  (FR-5).
- **D6. No growth-check tooling.** A word-delta script would bring back the numeric threshold
  that FR-8 rules out. The reviewer's judgment is the mechanism.

## 4. Units & sequence

**U1 — root convention edits (applied by the human).** U1 goes first, because U2's pointers cite
the bullet E2 creates. Each edit below gives its wording. The human may reword, but E1–E3 must
keep the constraints noted on them.

- **E1** — Stage G bullet, l.160–163. Replace the sentence that runs from "one read **by
  lookup**" to "and that is correct." with:

  > one read **by lookup** — `HISTORY.md`, `QUERIES.md`, `test-reports/`, `-coordination`
  > ledgers, and any `archived` document — may grow without bound, and that is correct. A
  > not-yet-`archived` `requirements/`, `plans/` or `reviews/` document also counts as read whole,
  > but the next bullet bounds it, not the closeout below.

  Constraint: the rest of the bullet stays unchanged, and the edit is net ≤ +25 words.

- **E2** — a new bullet, inserted right after E1's bullet (~176 words, accepted as is, §6):

  > **A not-yet-`archived` `requirements/`, `plans/` (not `-coordination`) or `reviews/` document
  > stops growing across revisions** (`docs/requirements/document-size-discipline.md`). Only new
  > scope adds words. Rework replaces the affected text in place: answering findings, folding in
  > answers, fixing errors. A plan or specialist note states the current design only. Its reasons
  > are stated as design, findings are cited by ID at most, and it has no finding→resolution
  > table or disposition list. An open stakeholder question stays until it is answered, then it is
  > folded in and removed. The review's next `## Pass N` records outcomes (collision rule 5). A
  > requirements decision log is append-only: each entry is one question → one answer, plus the
  > FR/AC it affects. A reversing entry names the entry it overturns. A fact has one home, and
  > other documents cite it. Every review gate checks a revision against this bullet. For any
  > growth it says whether it is new scope or rework, and it reports rework growth, restated
  > history and restated facts as findings; there is no numeric threshold. An existing document
  > complies at its next ordinary revision.

- **E3** — collision rule 5, two clauses:
  - After "**one dated line, not a narrative**", insert: "— under the growth-stops bullet it
    replaces the previous line". The bolded phrase itself stays, because the "open item" bullet
    depends on it.
  - After "never re-argued prose", insert: "; so does each stakeholder answer folded into the
    artifact since the last pass (question → answer)".
- **E4** — the Status set, l.245. Change "the second only on explicit stakeholder confirmation" to
  "the second only on explicit stakeholder confirmation, after an `analyst` review of the document
  that includes the growth-stops check".
- **E5** — `claude/docs/plans/prompt-waste-reduction.md`. Add one line directly under the
  `### Stage G` heading:

  > *Partly superseded for not-yet-archived `requirements/`/`plans/`/`reviews/`
  > (`docs/requirements/document-size-discipline.md` FR-1); the live rule is root `AGENTS.md`.*

  The table and the draft wording stay untouched; the pointer satisfies AC-1's "wherever it is
  written".

**U2 — prompt pointers (`cobb`).** U2 starts after U1 lands. Each edit is at most ~50 words added
net. It cites root `AGENTS.md`'s growth-stops bullet and restates only the action that agent
takes. Where an existing line can absorb the clause, rewrite that line rather than adding a new
one.

| File : anchor | Edit |
|---|---|
| `claude/architect/architect.md`, Guardrails | When revising after a review or a stakeholder answer, rewrite the affected text in place to the current design. The return message lists the findings handled and the answers folded in; the plan does not. |
| `claude/architect/plan-authoring-techniques.md:51` | Change "with a short revision note" to "updating the header's one-line revision note". |
| `claude/graph-dba/graph-dba.md:63`; `claude/data-scientist/data-scientist.md:81` | The same clause as `architect`, for `-graph`/`-ml` notes. |
| `claude/data-scientist/data-scientist.md:82`; `claude/security-expert/security-expert.md:72` | When the artifact is an in-scope document, the review runs the growth-stops check. |
| `claude/analyst/analyst.md`, after deliverable item 2 | Gating an in-scope revision means four things: run the growth-stops check, state new-scope vs. rework for any growth, give the Pass N one-line outcomes for findings and folded-in answers, and always run the check on a requirements document before it is flipped. |
| `claude/tico/tico.md:47` | Rewrite in place per D5. |
| `claude/tico/tico.md:66`, template l.100 | Each entry is one question → one answer plus the FR/AC it affects, with no narrative. A reversal names the entry it overturns. Rework of an FR/AC replaces its text. |
| `claude/tico/tico.md:110` | Flip only after the readback **and** an `analyst` review covering the growth-stops check; route the review through the existing consult. |
| `claude/tico/tico.md:169` | The delegated round returns "ready pending the `analyst` gate", never a bare ready-for-design confirmation; the flip follows l.110. |
| `claude/teco/teco.md:144` (re-gate brief guidance) | The revision brief says to rewrite in place, and the re-gate brief asks for the growth-stops check. |

For each edited agent, `cobb` also:
- updates `kaizen/plan.md` and appends an entry to `kaizen/history.md`, per
  `claude/AGENTS.md:308`;
- edits `claude/README.md` only if a frontmatter description changes (none is planned);
- runs `claude/scripts/audit-team.sh`, which must show no FAIL, and reports each prompt's word
  count before and after.

**U3 — review gate (`analyst`).** One static review of the U1 and U2 diffs. It runs the AC-1
inventory (§5), maps every FR to the text that now carries it, and applies the FR-8 check to the
edits themselves. This plan's own gate is the first live use of the rule.

**U4 — optional, `tico`.** In `claude/docs/manuals/agent-team.md`, line 115 describes the flip as
"only on your OK". Add the analyst gate so the manual matches FR-9. Manuals are outside
FR-2's growth rule, but they still need to be accurate.

## 5. Verification

| AC | How it is verified | When |
|---|---|---|
| AC-1 | `git grep -n -i -E 'without bound\|by lookup\|read[- ]whole\|unbounded' -- ':(glob)**/AGENTS.md' ':(glob)claude/*/*.md' ':(glob)skills/*/SKILL.md' claude/docs/plans/prompt-waste-reduction.md` (glob magic, so `*` does not cross `/`). Expected residue after U1 (18 hits today): the E1 text in root `AGENTS.md`; `prompt-waste-reduction.md` l.806–914, under E5's pointer, and l.44/117, which are about other things; off-topic prose at `analyst.md:34`, `falkordb-quirks.md:892`, `guard-testing-techniques.md:279/291` and `skills/python-web-quirks/SKILL.md:8/758`. Any other hit is a finding. A second probe, `-E 'revision note\|decision log\|Ready for design\|Pass N'` over the same paths, must find nothing that contradicts E2–E4 or the U2 rows | U3 |
| AC-2, AC-3, AC-4, AC-6, AC-7 | Static: U3 confirms that E2/E3 and the reviewer pointers carry each rule. Behavioural: the first gate on an in-scope revision after delivery states new scope vs. rework, and its Pass N carries the one-line outcomes. `teco`/`tico` check that when accepting the gate | U3; first live cycle |
| AC-5 | Static: `tico.md` l.47/66/100 and E2. Behavioural: the first decision log written after delivery | U3; next interview |
| AC-8 | `git diff --name-only` for U1/U2, then `grep -m1 'Status:'` on each file: none is `archived` | U3 |
| AC-9 | Static: E4 and `tico.md:110/169`. Behavioural: the next flip to Ready for design has an analyst review dated earlier | U3; next flip |

The behavioural ACs cannot be proven at delivery. Their evidence is the first ordinary cycle
afterwards (FR-7).

## 6. Risks & open questions

- **The Pass N rule.** FR-3 settles the conflict between reviews and growth. E3 only adds a
  clause, and earlier passes stay intact.
- **Write guards.** No change is needed (§2).
- **Word creep.** Root `AGENTS.md` grows by ~230 w (E1 ~15, E2 ~176, E3+E4 ~40), to ~3,180.
  Decision: the stakeholder accepts this growth with no offset unit. `tico` and `teco` are
  already far above 2,500 w, so U2 rewrites existing lines rather than adding new ones, and
  `cobb` reports the deltas.
- **The first revision of an existing document may shrink sharply** (FR-7). Reviewers must not
  treat removed history as lost content: the outcomes now sit in their Pass N.
- **Judgment-based enforcement (D6).** Compliance depends on reviewers actually running the
  check. The AC-7 evidence is what shows whether they do.

## Ready to implement

Implement `docs/plans/document-size-discipline.md` in this order: U1, the human edits to root
`AGENTS.md` (E1–E4) and the pointer line in `prompt-waste-reduction.md` (E5); then U2, `cobb`'s
pointer edits to seven prompts plus kaizen plan/history and the audit run; then U3, the `analyst` gate
with the AC-1 grep inventory; U4 is an optional `tico` manual touch-up. No questions are open.
