# Connected code graph — methodology review of the P1-35 interpreter design-fault diagnosis

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-35)

Subject: `docs/plans/connected-code-graph-p1-interpreter-design-fault.md` (architect), read against
`docs/plans/connected-code-graph-p1-interpreter-model-ml.md`, `docs/plans/connected-code-graph-p1-ml.md` §4.2-§4.3 and
`docs/plans/connected-code-graph-eval-protocol-ml.md` §10. No held-out item, custodian directory or dev gold beyond
`cpg/eval/dev/ac24.jsonl` was read. Pass 1.

## Verdict: approve with suggestions

The diagnosis holds and the plan (option A) is methodologically sound. Four things must change before the prompt is edited
(F1-F4); the fresh-set controls (F5) and tries list are fixed in `docs/plans/connected-code-graph-p1-interpreter-model2-ml.md`.

## Answers

**(1) Diagnosis: sound, with one gap and one limit.**
I replayed the saved ministral and qwen2.5 drafts through `resolve_draft` on the live store (`ccg_graphmind-ai-lab`, read-only). Every
missed qid reproduced the logged verdict kind and option count (ministral QW-7 `unresolved`/lookup `salesperson endpoints.ts`, QW-13
`unknown-name`/0, QW-47 `unknown-name`/0; qwen2.5 QW-7 `several-symbols`/2, QW-11 `several-symbols`/5, QW-13 `unknown-name`/1,
QW-39 and QW-42 2 options, QW-3 `no-anchor`). Ideal drafts, which I built independently: `postMessage` + file `endpoints.ts` resolves to
`salesperson/src/api/endpoints.ts::postMessage`; `send_message` as `tool` (with or without `falkor-chat` as `server`) resolves; `POST
/threads/{thread_id}/messages` + `impact` resolves; `ingest_documents` with `direction: null` gives `two-use-cases` with both directions,
and with `depends-on` resolves silently. So no resolver defect is visible against §4.2, and the unfound-name ask is the
pre-registered gold-blind policy. Gap and limit: F3 and F7.

**(2) The prompt edit: general in form, fitted by construction, and two of the five rules reach back into dev vocabulary.** F1, F2, F4.
The edit is derived from dev misses, which EP §10.2 allows and the architect correctly treats as spending the 14 items. It does not
leak held-out (answer 4). Leakage that remains is about dev2's meaning, not its validity: the blind writer comes from the same
writer population as dev1 and held-out, so dev2 measures transfer to **fresh items from the same writer**, not to a new writer. Say so in the report.

**(3) Fresh dev set: adequate in design; independence needs five added controls (F5) and power is lower than the earlier note stated (F8).**

**(4) Held-out contamination: none through the prompt edit.** The edit's author read dev gold only (the plan says so; I found nothing
contradicting it). `heldout-manifest.json` is not touched by the edit. The residual paths are procedural: the prompt author or the
cue gate reading dev2 (barred in the note), a prompt edit after dev2 results (barred), and the writer's sandbox seeing the prompt (to be confirmed, F5).
A prompt that now carries the writer's idioms raises the expected held-out rate if those idioms recur; that is the purpose of dev-tuning, and it is
disclosed in the record, not a leak.

**(5) Tries list:** `qwen/qwen3-4b-2507`, then `nvidia/nemotron-3-nano-4b`, then `mistralai_ministral-3-3b-instruct-2512`; `qwen2.5-3b-instruct`
dropped. Stop rules are in the successor note §5.

## Findings

### F1 — major — rule 4's `relations of / relações de` null cue is a dev-item token, and the pin that held it out must be rewritten, not deleted

`cpg/tests/test_front_prompt.py` asserts `"relations" not in words(PROMPT)` ("the cue of dev item QW-47 is taught nowhere") and no 3-gram overlap with
any dev question. Rule 4 retires the first pin and adds the QW-47 token itself. Retiring it is legitimate because dev1 is spent, but a one-token
cue teaches a lookup, and the held-out set tests reading. **Fix:** state the null rule by function (the question asks how the anchor connects to the
rest, or names both directions, or neither) with at least three paraphrase cues, `relations of / relações de` being one; keep the 3-gram,
dev-question and anchor-name pins as written. Rule 4's first half ("who or what calls, uses or consumes the anchor" to the `impact` cue) is
not fitting: `docs/plans/connected-code-graph-p1-ml.md` §4.3 already states "a passive 'what calls X' is `impact`" as the intended frozen cue. Replace
the test that forbids a bare "calls" in the direction field by one that pins the passive/active split.

### F2 — major — rule 5 names `MCP` and gives no positive instruction

"MCP is neither" is QW-47's word, and a 3-4B model told only what a token is not has no action to take. **Fix:** fx-only wording
("a protocol or product label in brackets after a tool name, such as `(FXQ)`, is not the tool and not the server: give it `other` or leave it out"), and extend the anchor-name
pin from `/ _ .` tokens to bare identifiers and acronyms in dev questions and gold anchors (`MCP`, `postMessage`, `send_message` and the like).
The resolver ignores a group made only of `other`/`server` spans, so `other` is safe (`docs/plans/connected-code-graph-p1-ml.md` §4.2).

### F3 — minor — the diagnosis omits two `not-built` misses

qwen2.5 QW-19 (ambiguous gold) came back with use case `root-cause` (`not-built`), and ministral QW-30 came back `not-built` (the architect attributes it to a
scope read). Neither is addressed by the five rules, and neither is a resolver defect. Both are model misreadings of a field the prompt does state. They are a
residual cost: with QW-30-type misses unaddressed, a model that repaired every other miss lands at the threshold with no slack. No sixth rule is requested; list them as expected residuals.

### F4 — major — the plan adds two examples against a ceiling of four

The test `1 <= len(EXAMPLES) <= 4` and `docs/plans/connected-code-graph-p1-ml.md` §4.2 ("at most 4 few-shot examples") bind. The prompt has four now. Raising to six
is reasonable (rules 1, 3 and the null class need a demonstration) but it is a change to a stated bound, so it is made in the successor note (ceiling six; the
two new examples fx-only, different sentence frames from any dev question, one of them Portuguese) and the pin is updated in the same commit.
Report prompt length (tokens, both prompts) in the commit; it is not a gate.

### F5 — major — controls missing from the fresh-set method

(a) **Anchor reuse.** The blind writer will, by chance, pick dev1's popular anchors (`send_message`, `post_message`, `POST /threads/{}/messages`); an item with
dev1's gold anchor set tests the edit on a fitted case. Discard such items before rank selection. (b) **Over-generation and a fixed rule.** The shape needs 3 survivors per cell after annotator
drops, dedupe and the dev1 filter; ask for 6 per cell and 6 ambiguity candidates, select by hash rank under a seed fixed before the file is built, one top-up round per short cell, no choosing by difficulty. (c) **Blindness.** The writer's sandbox must exclude `cpg/ccg/front/`, `docs/plans/`, `cpg/eval/dev/`
and `cpg/eval/results/`. I did not verify this; `teco` confirms from `cpg/eval/headless/README.md`. (d) **Order.** Prompt commit and its cue gate come first; dev2 is authored afterwards; nobody who edited or gated the prompt reads dev2. (e) **Qids** unique
across dev, held-out and reserve; the custodian's probe-hash check also runs on normalized question text. The shape (12 + at least 2, three per cell, two or more `depends-on`) and the thresholds stay.

### F6 — minor — harness string and record fields

The dev summary's `recommendation` ("never tune the prompt, grammar or resolver on the dev items") contradicts EP §10.2 and will be printed again on dev2; the
text is wrong for dev items under this protocol, not for held-out. The optional `promptHash` in `devAttempts` is agreed; the record must also state the dev1 prompt hash.

### F7 — minor — "prompt silence" versus capability is a hypothesis until dev2

By the architect's own trigger reading only QW-13 and QW-7 miss the same way on all three runs; QW-11 and QW-47 miss differently per model, and qwen3 stored no drafts.
Replay confirms what each draft did; it cannot say whether a better prompt would have fixed a 3-4B model. dev2 is the only test of that, and the report should call the edit
an intervention under test.

### F8 — minor — power statement corrected (mine, in the earlier note, inherited by the plan's §8 risks)

`docs/plans/connected-code-graph-p1-interpreter-model-ml.md` §2 gives a pass chance of "0.44 x 0.70 = 31%" at true rates 0.85/0.70. The 2/2 ambiguous requirement is both items passing,
so 0.70² = 0.49, not 0.70. Computed (binomial, independent items): unambiguous P(>=11 of 12) = 0.443 at 0.85; joint with 2/2 at 0.70 it is 0.217, and 0.520 over three tries if independent. The
earlier note is executed-against and stays; the corrected figures are in the successor note §6.

## Checks that must be able to fail (root `REVIEW.md`)

The plan's pins (one per rule) and the dev2 controls each need a named planted input that turns them red; the successor note §3 and §4 name them.
No `REVIEW.md` line is proposed.

## Growth-stops check (root `REVIEW.md`)

The architect's plan states the current design and cites findings by none; no finding-resolution table. Its §2 cites the saved-draft replay and its own scripts. No growth issue.

## Pass 2 — wording re-read of the committed prompt (`2bf875bd`, promptHash `f91e01b0…`)

**Verdict: approve. dev2 authoring may start.** Static read of `cpg/ccg/front/prompts/interpret.txt` and `cpg/tests/test_front_prompt.py`; no held-out or custodian path read.

- **F1 fixed.** Rule 4b is a class rule (`how the anchor connects to the rest of the system, names both directions, or names neither`) with five example cues, `relations of / relações de` among them; example 6 is a PT "conexões" question not in the cue list, so the class is demonstrated beyond the tokens. The 4a split is pinned (the only `calls` in the `depends-on` part is the subject-worded one).
- **F2 fixed.** Rule 5 is `fx`-only (`(FXQ)`), says what to do (`other` or leave it out); pin 9 covers bare identifiers and acronyms from dev questions and gold anchors. No dev name, anchor or `MCP` in the prompt; the 3-gram pin holds on the final text.
- **F4 fixed.** Six examples, all `fx-*`, one PT new, ceiling pin updated.
- **Dev idiom:** no dev phrase or anchor appears. Shared register remains by design: "callers", "renaming ... field" (example 5 echoes the shape of a rename-a-parameter question). Acceptable: it is a rule demonstration over made-up names, and dev1 is spent.
- **Ambiguities a 3B model may misread (none blocking):** (a) R2 says never add `:`, while a question written "path line N" (the form of dev QW-3, on which qwen2.5 normalised to `path:564`) has no instruction for the line; a model may now emit only the file span and lose the line. Prompt is still editable before dev2 exists; if the owner wants it covered, add one sentence and re-run the pins and the hash, otherwise record it as a known residual. (b) `names neither` widens `null`; the retained "Never answer null when the question implies a direction" is the counterweight, and dev2 measures it. (c) R5 has no worked example, text only.
- **Pins bite.** On a copy (note: run without `-I`, which ignores the copy's path), baseline 19 passed; mutants: delete R3 sentence, 1 red; `(FXQ)` to `(MCP)`, 2 red (R5 pin and pin 9); remove the null class text keeping the cues, 1 red; remove the `calls` from the impact cue, 1 red. A mutant that deletes only the worked sentence `in "PUT ..." the span is ...` after R1's rule sentence stays green by design (the rule sentence and examples 5/6 carry the pin); not a finding.
- **Conditions unchanged:** nobody who edited or gated the prompt reads dev2; no prompt edit once dev2 authoring starts (note §2).

## Pass 3 — dev2 shortfall (`docs/test-reports/connected-code-graph-p1-ac24-dev2-report.md`, counts only)

**Verdict: approve with a procedural reading; no stakeholder decision needed** (12 + 2, 11/12 and 2/2 and the cells are unchanged). The successor note §4 items 4-5 and §7 are revised in place.

**(a) The held-out/reserve filter.** My §4 item 4 said "equal or near-equal in text, or by probe hash"; the intent (EP §12.1, "an item already in dev/held-out/reserve is discarded") is item duplication. The custodian read it as "any shared anchor probe", which drops items that merely share one anchor. That reading is stricter than the intent and biases dev2 away from the anchors held-out contains, which makes dev2 a worse predictor of held-out. Leakage: none either way. The prompt is frozen, dev2 tunes nothing, no model has run on dev2, and the decision uses counts only. Keep: text shingle against dev1/held-out/reserve (15 drops), dev1 anchor-set equality (12 drops; dev1 is the fitted set, this one protects the check). Replace the anchor-probe drop by **full-gold equality** with a held-out/reserve item (use case, direction, anchor set, scope). A waiver for contract-point/en alone is rejected: a cell-specific fix after seeing counts invites selection; the rule is applied to every cell. A top-up is not needed first: it was already used and yields about 40% survival.
**Exact instruction for the custodian:** *Redo selection on the existing 74 candidates, no new session. Held-out and reserve: drop an item only if (i) its normalized text shingle-matches one (as before) or (ii) its full gold (useCase, direction, anchor set, scope) equals the gold of one, compared in-process; do not drop on a shared anchor probe alone. Dev1 filters unchanged. Then rank selection with the committed seed: 3 per unambiguous cell (>= 2 `depends-on` per group), 2 ambiguous in the order `two-use-cases`, `several-symbols`, then the rest, first by rank within each. Report counts only: per-filter drops (the new full-gold drops separately), survivors per cell, ambiguous survivors by kind, and `heldout-manifest.json` sha256 before and after. If any unambiguous cell still has fewer than 3 survivors, or fewer than 2 ambiguous survive, write no file and return the counts.* I cannot compute the new survivor counts (it needs held-out gold); the report's figure of 6 survivors in contract-point/en without the probe filter is the upper bound.

**(b) The ambiguous slots.** Zero `two-use-cases` survived the §10.3 join: 6 of 6 candidates (3 en, 3 pt, two sessions) failed it, 3 by annotator disagreement and 3 by unanimous "unambiguous" against the writer's label, as the report's drop counts add up. Yes, that is itself evidence and must be disclosed: humans do not read these writer-intended direction-less questions as ambiguous reliably (0 of 6 passed; Wilson 95% upper bound 0.39 on the pass rate for this writer and kind, computed), consistent with dev1 holding one such item. It is no evidence about held-out, which I have not read. Effects on dev2: (i) an `unknown-name`/`no-anchor` slot is passed by any non-resolving answer (EP §10.4), so the 2/2 is easier; with one `several-symbols` slot (all gold options required) and one trivial slot the joint pass probability, binomial, lies between q = p² and q = p for the two ambiguous items: 0.636-0.691 at true 0.92, 0.320-0.377 at 0.85, 0.176-0.220 at 0.80 (computed). (ii) R4b, the null-direction rule, has no out-of-sample item in dev2, only the unambiguous items' direction cues test the direction work (R4a). Over-asking stays controlled by the 12 unambiguous items. The slots are filled as in the instruction; a third round for `two-use-cases` is rejected (0 of 6, with unanimity, says more candidates from the same writer will not pass the join; cost is two more sessions).

**(c) Path.** The procedural reading above, then the dev runs as in note §5. Disclosures for the record and the final report: the filter reading changed after the first selection attempt (counts only, before any model run); dev2's ambiguous slots lack `two-use-cases`; R4b untested out of sample. A shortfall after the rerun returns here; a change of counts, thresholds or cells is the stakeholder's.
