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
