# Connected code graph — P1 interpreter: revised prompt, fresh dev set, tries and stop rules

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-35) · **Extends:** `docs/plans/connected-code-graph-p1-interpreter-model-ml.md` · **Reviews:** `docs/reviews/connected-code-graph-p1-interpreter-design-fault-ml.md`

States only the change against `docs/plans/connected-code-graph-p1-ml.md` §4.2-§4.3 and `docs/plans/connected-code-graph-p1-interpreter-model-ml.md`,
for the revised prompt in `docs/plans/connected-code-graph-p1-interpreter-design-fault.md` §4. The earlier note's tries list bound the old prompt hash
(`494fb949…`); the four dev runs on it stay in the record as spent. Everything not named here stands.

## 1. Question and decision

Decision: after the prompt edit, which fresh dev set, which models in which order, and when to stop. Constraint: held-out stays unread and unspent; nothing below is
run until this note is committed.

## 2. What changes against §4.2-§4.3

- **Few-shot ceiling:** six (was four). The two new examples are `fx-*` only, use sentence frames unlike any dev question, and one is Portuguese. Report prompt tokens, old and new, in the commit (not a gate).
- **Dev set for the entry check:** `cpg/eval/dev/ac24-r2.jsonl` (§4), not `ac24.jsonl`. Thresholds unchanged: >= 11/12 unambiguous and 2/2 ambiguous; unusable on more than 3 of 14 counts as failed.
- **The prompt is frozen at the commit that precedes dev2 authoring.** No edit to `interpret.txt`, grammar, sampler or resolver after dev2 exists or after any dev2 run starts. A resolver or harness defect found by a fixture test
  returns to `architect`; affected tries are re-scored offline from the stored drafts, labelled as such, and are not new tries.
- **Record:** `devAttempts` lists all four old tries with `promptHash` `494fb949…` and every new try with its own; the record states that the prompt was revised after the first dev check failed and the first dev set was spent.

## 3. Prompt wording gate (`data-scientist` re-reads the committed text before dev2 authoring starts)

Acceptance for each rule; `tdd-engineer` writes each pin red first and names its mutant.

| Rule | Acceptance | Mutant that must redden the pin |
|---|---|---|
| 1 route verb | states a route span keeps its HTTP verb exactly as written; `fx-*` names | delete the sentence |
| 2 span | one contiguous stretch; no joined words; no added `/` or `:`; a component name written before a file name stays out of the file span | delete the sentence |
| 3 parameter | a parameter, field or option named only as the thing being changed is not an anchor; the owning tool, route or function is | delete the sentence |
| 4a direction | `impact` cue covers who or what calls, uses or consumes the anchor; `depends-on` keeps "the anchor itself calls or uses"; pin the active/passive split | swap the two cues |
| 4b null | a class rule (how the anchor connects to the rest, both directions, neither) with at least three paraphrase cues, `relations of / relações de` one of them | remove the class sentence, keeping the cue words |
| 5 hint | `fx-*` wording; a protocol or product label after a tool name is `other` or omitted; says what to do | replace by the word `MCP` |

Standing pins kept as written: no 3-word phrase shared with any dev question, no dev question in the prompt, no dev anchor name; the example-pair and example-validity pins with the ceiling now six.
The pin "no `relations` word anywhere" is retired (dev1 is spent) and replaced by the class-rule pin above. Extend the anchor-name pin to bare identifiers and acronyms from dev questions and gold anchors (planted: `MCP` in the prompt must fail).
The no-held-out rule is unchanged: nothing about held-out text enters any prompt, pin or example.

## 4. Fresh dev set

Writer `QW` (blind), annotators `AN1`/`AN2`, custodian a fresh `qa-engineer`, under EP §6, §10.1 and §10.3, with these controls:

1. **Order:** prompt commit, analyst gate on the diff, my wording re-read (§3); then the writer starts. Nobody who edited or gated the prompt reads dev2 (a person, including `architect` and `data-scientist`, reading it before the runs finish voids it as an entry check).
2. **Blindness:** the writer's sandbox excludes `cpg/ccg/front/`, `docs/plans/`, `cpg/eval/dev/`, `cpg/eval/results/`, the saved drafts and any miss list. `teco` confirms this from `cpg/eval/headless/README.md` before the session (unverified here).
3. **Candidates:** 6 per (group, language) cell of `code-location`/`contract-point` x `en`/`pt` (24), and 6 ambiguity candidates, at least two `two-use-cases` and two `several-symbols`, both languages. At least 2 of the 6 per cell are `depends-on`.
4. **Filters before ranking** (custodian, counts only reported): drop any item whose gold anchor set equals a dev1 gold anchor set; drop any item whose normalized question text equals or shingle-matches a dev1, held-out or reserve item, and any item whose full gold (use case, direction, anchor set, scope) equals the gold of a held-out or reserve item (the custodian compares in-process and reports counts only; sharing a single anchor with a held-out item is not a duplicate and is not dropped, since the filter would otherwise bias dev2 away from the anchors held-out contains); drop annotator disagreements and writer/annotator intent mismatches (EP §10.3). Qids are unique across dev, held-out and reserve.
5. **Selection:** hash rank with seed `ccg-p1-ac24-r2|<EVAL_COMMIT>`, fixed here: first 3 survivors per cell (>= 2 `depends-on` per group over both languages); ambiguous: two items, taken by kind in the order `two-use-cases`, `several-symbols`, then the rest, first by rank within each (a kind with no survivor is skipped; the count stays two). One top-up round (same blind instructions) per short unambiguous cell, already used for dev2; a cell still short after it returns the shortfall (a count or threshold change is scope).
6. **Sealed before runs:** commit the file and its sha256; no item is added, removed or relabelled afterwards, including after a miss that looks like a gold error (reported, not struck).
7. `heldout-manifest.json` is byte-identical before and after; `s47_scripted_steps.py` and `split_manifest.py` are not run. The harness runs with `--dev-items cpg/eval/dev/ac24-r2.jsonl`.

What dev2 measures: transfer of the revised prompt to fresh items from the same writer population. It does not measure novelty of idiom, and it shares that property with held-out, which is why it is a fair predictor of it. A pass is a smoke test.

## 5. Tries and stop rules (pre-stated)

Order, one dev run each, repeat 1, one model resident, `lms load <id> -c 16384 -y`, fingerprint and loaded context confirmed, drafts stored:

1. `qwen/qwen3-4b-2507` (Q4_K_M). Every dev1 miss of this model (QW-7, QW-13, QW-11, QW-47) is a construct the five rules target; a repaired run leaves one miss of slack. Needs the dev2 canary re-recorded.
2. `nvidia/nemotron-3-nano-4b`. Not run before; a different family; first an `fx-*` probe (non-dev) confirming a grammar-constrained reply has no thinking text and no truncation at `max_tokens` 384, otherwise it counts as failed without a dev run.
3. `mistralai_ministral-3-3b-instruct-2512`. Last because its dev1 misses include a `not-built` over-ask (QW-30) that no new rule addresses, so a fully repaired run still sits at 11/12 with no slack.

`qwen2.5-3b-instruct` is dropped (dev1: 6/12 and 0/2, six over-asks; the dev1 counts are spent for claims but suffice to order tries). Larger models (option C of the design-fault note) are not on this list; adding them is a stakeholder decision.

Rules:
- **S1** Stop at the first model that meets >= 11/12 and 2/2 with unusable <= 3/14. It is then fixed (record `interpreter`, canary, `RECOMMENDED_MODEL` / §4.1 updated in one change); `prereg-record` is built from that run; the held-out run proceeds once as §4.3.
- **S2** At most three dev2 runs; no model re-run; no change of temperature, `max_tokens`, context, prompt, grammar or dev2 between or after tries.
- **S3** A try with unusable > 3/14 (template 400, thinking text, truncation) is failed, not retried; a 400 is recorded with its status.
- **S4** No early stop on partial results and no skipping an untried model after a failure; a pass ends the search, a failure does not end it until the third run.
- **S5** If all three fail: report "AC-24 not met at P1, carried to P2" with all seven dev counts, the prompt hashes, the missed qids with drafts and their classes (span, hint, direction, unfound-name ask, not-built), and no held-out run. No dev3, no further prompt edit; any continuation is a new stakeholder decision.
- **S6** After the dev2 count of a try is recorded, the same loaded model may be run once over `ac24.jsonl` as a non-gating regression diagnostic (does the new prompt lose items the old one passed: QW-3, QW-30, QW-32). It is reported separately and edits nothing.
- **S7** Every try, passed or failed, is listed in `devAttempts`.

## 6. Evaluation design and power

- Metric: unambiguous understood k/12 and ambiguous k/2 on dev2; report Wilson 95% intervals (11/12 is [0.646, 0.985], computed in EP §10.5; 12/12 is [0.758, 1.000], computed here), over-asking, unusable, missed qids, and the miss class per qid. No ranking of tries is claimed.
- Binomial pass probabilities of the entry check, items independent, true rate p on unambiguous items and q on the two ambiguous (computed):

| true p | P(>=11 of 12) | P(12 of 12) | joint with 2/2, q = p | joint with 2/2, q = 0.70 |
|---|---|---|---|---|
| 0.92 | 0.751 | 0.368 | 0.636 | 0.368 |
| 0.85 | 0.443 | 0.142 | 0.320 | 0.217 |
| 0.80 | 0.275 | 0.069 | 0.176 | 0.135 |
| 0.70 | 0.085 | 0.014 | 0.042 | 0.042 |

- Reading: a model with a true 0.92 fails the check about one run in three (q = p, 0.364 computed as 1 - 0.636); a 0.80 model passes it 17.6% of the time (q = p). The check separates 0.92 from 0.70 or below (0.636 against 0.042) and says little between 0.80 and 0.92.
  Over three independent tries a 0.80 model passes at least once with probability 0.44 and a 0.85 model 0.69 (q = p); tries share the dev2 items, so these are upper bounds. A pass is compatible with a true rate near 0.65 (the lower Wilson bound of 11/12).
  This corrects `docs/plans/connected-code-graph-p1-interpreter-model-ml.md` §2, which used 0.70 instead of 0.70² = 0.49 for the two-item ambiguous requirement (0.217 at 0.85/0.70, not 31%).
- Thresholds are not changed after dev1: the entry check is a screen, the claim is the held-out run.

## 7. Risks and open questions

- dev2's ambiguous slots hold no `two-use-cases` item (0 of 6 candidates passed the annotator join), so the null-direction rule (R4b) has no out-of-sample test in the entry check, and an `unknown-name` or `no-anchor` slot is passed by any non-resolving answer. The 2/2 is easier than the earlier table assumed; §6 gives the bounds. Disclose both in the report and the record.

- The revised prompt is longer for a 3-4B model and can lose items the old one passed; dev2 measures that, dev1 cannot. S6 reports it.
- Fresh-set tooling (pinned writer binary, custodian root, `EVAL_COMMIT` checkout) is unverified; `teco` runs `status`/`prepare` first.
- Residual `not-built` over-asks (a use case or scope misread) are unaddressed by design; they cost slack, and appear in the S5 report if the search fails.
- The dev2 pass probability for a model near 0.80 is low (0.176); if all three tries fail with counts of 9-10/12, the report says the check could not distinguish them from a 0.92 model with bad luck (a 0.92 model fails with 0.364 probability), not that AC-24 is disproved.
