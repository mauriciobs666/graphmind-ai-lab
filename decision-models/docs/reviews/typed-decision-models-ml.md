# Typed decision models: fit question texts, methodology review

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Reviews:** `decision-models/fit/*/questions.json` (U6f.4c)

## Pass 1 — U6f.4c question texts (2026-11-19)

Scope: plan4 §3.11 "Question texts" and note §Q2.3. Read: each point's `rubric.md` and `questions.json`
only; no item of any split. Checks per file: non-redundant diversity, leakage beyond the rubric, a proper
control, systematic answer bias, ambiguity. Threshold choice absorbs a text's overall lean (the ladder fits
`l`/`h`/`t`); what matters is that a text does not reorder items against the rubric.

| Point | Verdict |
|---|---|
| `guard-judge` | approve with suggestions |
| `join-in-judge` | approve with suggestions |
| `message-type-classifier` | needs changes (one rubric contradiction, one omission) |

### guard-judge — approve with suggestions

- **Control** `g-control`: one line, no rubric content ("given the evidence shown" is the format). Proper.
- **Content.** Every clause traces to the rubric; nothing added. `g-rubric` is the whole rubric condensed;
  `g-source` (who supplied it, promises, pre-shown turns), `g-completeness` (`missing` list, enough
  information) and `g-wording` (hedges, questions back, latest wins, literal condition) split the boundary
  cases with little overlap. Usable diversity.
- **m1 (minor).** Only `g-rubric` carries the rubric's default ("unsure: false"). The other three are
  written as lists of ways to be false and omit the true-side cases the rubric states (a fact in other
  words; facts across turns, or in `known`). Their `p` then orders items by "any defeater present", a
  coarser ranking. Requested edit, append to `g-source` and `g-completeness`: "A fact put in other words
  than the condition's, or spread over several turns, still counts." (one sentence each; `g-wording`
  already covers true-side cases).
- **n1 (nit).** `g-source`, "Who supplied the fact the condition asks for?", opens with a non-yes/no
  question; the closing "Does the condition hold?" mostly repairs it. Acceptable.

### join-in-judge — approve with suggestions

- **Control** `j-control`: one line, no rubric content. Proper.
- **Content.** Faithful to the rubric; no agent name or example item leaked (the rubric's "Nova" is not
  used). `j-addressee`, `j-continuity` and `j-room` map onto the rubric's three true-branches and its
  boundary cases (name in passing, after the agent's turn, room questions, several people talking); little
  redundancy with `j-rubric` beyond the shared principle, which is the point.
- **m1 (minor).** `j-continuity` ("expects a reply only if…") and `j-room` ("only if… nobody… no named
  person") are conjunctive, so they lean to *false* and omit the rubric's true cases a text should still
  rank (a message addressed to the agent by name inside a room discussion; small talk put to the agent).
  The threshold absorbs the lean, not the missing cases. Requested edit, append to `j-room`: "A message
  addressed to the agent by name is for the agent whatever else the room is doing." and to `j-continuity`:
  "A follow-up that asks about or corrects what the agent said counts even when others are present."
  (`j-room` already says small talk by name.)
- **n1 (nit).** `j-addressee` and `j-room` open with a non-yes/no question ("Who…", "Is the new message
  a question…"); a 4B model could answer that one. Both end with the true question; keep, or drop the
  opening question mark framing in `j-room` to "The new message may be a question or request to the room."

### message-type-classifier — needs changes

- **Control** `m-control`: bare label names, empty choice instructions, one question line. Proper.
- **Per-label instruction texts that differ, not only framing: acceptable.** A choice question's spec
  carries per-choice `instructions` by design (plan4 §3.11 rules; the loader requires the same value set,
  checked as a set against `expected`), so a text may redefine labels. Four rubric-derived variants are
  the diversity the ladder can use; the cost is a selection effect over four non-control texts on one
  calibration split, which the control tie-break and the untouched eval split already bound (fit note §3).
  `m-definitions` and `m-precedence` are close (definitions plus a precedence sentence); keep both, since
  precedence is a different failure surface.
- **M1 (major, by reading) — `m-severity` and `m-precedence` contradict the rubric on off-topic
  questions.** The rubric: "a question about something unrelated to the business is `other`". In
  `m-severity`, "nothing went wrong" → "an order, a question or other" with `question` = "Nothing went wrong
  and the sender wants information"; in `m-precedence`, precedence puts `question` before `other` and
  `question` = "A plain request for information". Both route off-topic questions to `question`, a
  systematic bias on a rubric-stated boundary case (`m-definitions` and `m-context` handle it). Requested
  edits: `m-severity` `question`: "Nothing went wrong and the sender wants information about the business."
  `m-precedence` `question`: "A plain request for information about the business, or one on which a
  purchase depends." and `other`: "Nothing above applies, including a question unrelated to the business."
- **m1 (minor) — the previous message is never mentioned in `m-precedence` or `m-severity`.** The item
  shows `PREVIOUS MESSAGE`; the rubric says classify `MESSAGE` only. Requested edit, append to both
  `instructions`: "Classify the MESSAGE; the previous message is context only." (`m-definitions` and
  `m-context` already say it.) Their escalation lines also omit "already raised, shown by the previous
  message"; `m-severity` has "already raised", `m-precedence` should add "including as shown by the
  previous message".
- **n1 (nit).** `m-context` complaint "a first problem" and escalation "repeats a problem already raised"
  are consistent; its `order` choice repeats the rubric's returns rule only for no-fault returns, correct.

### Questions to the stakeholder

None. All edits are wording inside the files' own scope; they are `coder`'s, and the files are unfitted so
an edit costs nothing (sha256 is taken before the first call).

## Pass 2 — edits applied (2026-11-19)

Read `message-type-classifier/questions.json` and the two reworded `join-in-judge` texts only.

- **message-type M1 — fixed.** `m-severity` `question` and `m-precedence` `question` now say "about the
  business"; `m-precedence` `other` says "including a question unrelated to the business".
- **message-type m1 (previous message) — fixed** in both instructions. **m1 (repeat shown by previous
  message) — fixed** in `m-precedence` escalation.
- **join-in `j-room` rewording — accepted.** "Addressed to the agent itself, and not merely naming it"
  keeps the rubric's name-in-passing boundary, which my literal text would have contradicted.
- **join-in `j-continuity` rewording — accepted.** "While the exchange is still between that person and
  the agent" matches the rubric's first true-branch condition and its several-people rule.
- guard-judge m1/n1 and join-in n1: unchanged, optional, not blocking.

**Final verdicts:** `guard-judge` approve with suggestions (unchanged); `join-in-judge` approve (the
suggestions applied; the nit is optional); `message-type-classifier` approve.
