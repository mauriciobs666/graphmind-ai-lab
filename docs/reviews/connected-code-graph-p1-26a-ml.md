# Connected code graph P1-26 dispatch 1: methodology review of the deterministic front

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 P1-26 (dispatch 1)

Verdict: **approve with suggestions.** Findings M1 and M2 land before dispatch 2 and before `ccg-prereg-ac24-v1` is committed. M3 is a decision for the stakeholder.

Scope: the uncommitted tree against `e200a8a2`. Read `docs/plans/connected-code-graph-p1-ml.md` §4.2–§4.3, EP §10.4, `docs/plans/connected-code-graph-p1.md` P1-26, `cpg/eval/dev/ac24.jsonl`, `cpg/ccg/front/{__init__,parsers,resolve}.py`, the four `test_front_*.py` files and `front_helpers.py`.

Run: 119 passed, 1 xfailed (the strict QW-27 as-asked test). Probes below were executed read-only against the code.

## Pass 1

### 1. Method conformance

| ML note element | Implementation | Verdict |
|---|---|---|
| Verbatim-substring guard | `guarded`: per-span `text in question`, byte and case exact. Tests assert zero store reads for a fabricated span. | Conforms. |
| Per-group join (symbol+file suffix, file+line, tool/symbol+server, route) | `_group`, `path_suffix_match` (segment boundary), `server_matches` (path-segment, noise words dropped). | Conforms. `file+line` has no suffix rule (see M3). |
| `null` direction expands to both | `directions()`; also takes `readings`. | Conforms. |
| Ranking | Candidate-major (candidate, then impact before depends-on). | **Not in the note** (see M2). |
| CAP=5 | Applied to every `options` list. | **Deviates** (see M2). The note caps the did-you-mean list only. |
| Did-you-mean order | Edit distance, then qname, cut at 5. | Conforms, with a nit (m2). |
| Empty group means `unresolved` | The delegate returns `ambiguous` with `no-anchor`. | Sound deviation (see section 5). |

The never-picks-between-readings rule holds. Two groups that each resolve raise `NotBuiltAsk`. A single-group draft with more than one candidate gives `ambiguous`.

### 2. Leakage and contamination

- **Held-out:** no held-out text and no `QW-` id outside the 14 dev items appears in `cpg/ccg/front` or the tests (grep, verified). `cpg/eval/dev/ac24.jsonl` is unchanged. The dev set is for scoring and tests under §4.2, so reading dev text in tests is allowed.
- **Parsers:** no dev-specific hardcoding. The word lists are generic: `LINE_WORDS`, `SCOPE_KEYS`, extensions, HTTP verbs, and the closed whole-remainder words. `test_every_dev_question_is_prose_the_parsers_do_not_claim` pins that the parser claims 0 of 14 dev items. It is a safe constraint and shows the parser is not tuned on dev.
- **Test doubles:** `dev_store()` is built from the dev gold anchors plus distractors. The drafts are written by hand from the question text, and every span is asserted to be a substring through `run()`.
  - This is a gold-encoding world. What it tests is the join logic. It cannot show that the real graph resolves the same way: line ranges, nesting depth and the `falkor-chat` path segment are invented to match the gold.
  - It is not leakage into the evaluation, because the resolver has no gold access and no per-question branch. The measurement of real-graph resolution is the dev entry check (P1-35), and unit-test greens must not be reported as pre-registered-metric evidence.
  - Suggestion (minor, m1): one read-only live probe of the five named drafts against the real `ccg` graph before dispatch 2, reported as a probe.
- **Tautological negatives:** `test_qw3_negative…` and `test_qw11_negative…` hand the resolver `direction="impact"` and assert it answers impact. They cannot fail on the property §4.2 names (the interpreter returning `null` for a stated direction). That property is dispatch 2's, and the labels should say so (m1). `test_the_negatives_never_come_back…` (`parse` returns `None` for them) is a real check, though of the parser only.

### 3. The "fully read" rule (parsers)

The design is conservative. A question is claimed only when every token is an anchor, a scope, or a whole-remainder closed word. A direction is stated only by `impact` or `depends on`. All prose goes to the interpreter. For QW-3 and QW-11 the parser returns `None` (tested), so neither is claimed and neither can come back `null` from here.

**M1 (major, verified by execution): `depends on` is a direction-flip hazard.** `parse("depends on a/b.py:3")` returns direction `depends-on`, and so does `parse("a/b.py:3 depends on")`. In English, "depends on X" with X as the anchor most naturally means "what depends on X", which is the `impact` direction. "X depends on" is the reverse and means `depends-on`. The parser silently resolves one reading of an ambiguous phrase. Resolved single direction on the wrong reading is the silent failure EP §10.4 scores worst, and it bypasses the interpreter, which is where the null/expand safety lives. The test `("depends on: a/b.py:3", "depends-on")` pins the hazard.
- Fix: accept as direction words only the kebab tokens `impact` and `depends-on` exactly as written (reject the spaced form), or drop `depends on` and let the phrase go to the interpreter (`None`). Add a planted-input test that `depends on a/b.py:3` is `None` or `null`-direction, never `depends-on`.
- Also run the original dev claim check again afterwards (still 0 of 14).

Minor: `parse("Node.js")` and `parse("e.g.")` return a draft, because a dotted token is a file or qname. That only matters for a question consisting solely of that token, but a bare qualified-looking token is cheap to over-claim. Consider requiring that a symbol-only draft has a recognized anchor form in the structured sense (m3).

### 4. CAP trimming and ranking: does it bias a pre-registered metric?

**M2 (major, verified by execution): the cap is applied to complete readings, with an unregistered ordering.** EP §10.4 scores a `several-symbols`/`two-use-cases` item as understood only if `options` contain every gold entry and `|options| ≤ CAP_i = max(5, |goldOptions_i|)`. Probes with `run` defined in 3 and 4 files:

- `direction=null`, 3 candidates: 6 readings are cut to 5; `m2 depends-on` is lost.
- `direction=null`, 4 candidates: `m3` is absent in both directions.

Two consequences:
1. **Lenient bias.** Take an item whose gold is 3 candidates × `impact` (QW-19's shape). If the interpreter wrongly returns `null`, uncapped output is 6 options, above `CAP_i`, which is correctly scored "not understood". Truncated to 5 it still contains all 3 impact readings and passes with 2 reported extras. The resolver launders an interpreter direction error into a pass exactly at c=3. The effect is asymmetric toward `impact` gold: a `depends-on` gold with `null` loses the c=3 reading.
2. **Strict bias.** A correct complete list longer than 5 (gold of 2 directions × 3+ symbols, `CAP_i` ≥ 6) is trimmed below the scorer's own cap and fails on a correct answer. The resolver's fixed 5 cannot know `CAP_i`.

Neither is gold-aware, so this is a validity defect rather than cheating. On dev, QW-19 states its direction, so neither consequence shows there. A held-out run would first show it, and it is cheap to remove now.

- Fix: apply `CAP` to **did-you-mean lists only** (as §4.3 says, "the resolver's did-you-mean list is capped at 5 by construction"). Return complete readings uncapped (the engine already bounds candidates at `config.OPTIONS_MAX` = 20). Then an interpreter that over-asks direction is scored as the metric intends. Order stays candidate-major (a deterministic presentation choice); record the ordering rule and the "cap = did-you-mean only" rule in `ccg-prereg-ac24-v1` so neither moves after the run.
- Test: 4 candidates × `null` returns 8 options, and a 6-suggestion unknown-name list returns 5.

### 5. No-anchor: `ambiguous`, not `unresolved`

Sound. The answer schema requires non-empty `lookups` for `unresolved` (`docs/plans/connected-code-graph-answer-spec.md` §3), and a draft whose groups were all dropped or that names no file has no failed path. EP §10.4 scores `unknown-name`/`no-anchor` gold as understood on `status != resolved`, so `ambiguous` with `options: []` scores identically. The answer spec says `options` "may be `[]`" for these kinds. Use `unresolved` only on a failed named path (as implemented).
- The note's "a group left empty makes the answer `unresolved`" is an erratum against the schema. I will amend §4.2 to say `ambiguous`/`no-anchor`, with `unresolved` reserved for a failed named path.
- One consequence for the guard: a fabricated `file` span dropped from a `symbol`+`file` group lets the symbol resolve alone. That is acceptable, because the symbol text is in the question and the span cannot add a constraint the user did not write.

### 6. QW-27 (partial path with a line)

The delegate handled the open case correctly: the as-asked test is `xfail(strict=True)`, so it flips to a failure the moment it is fixed (a working mutant guard), and a separate test pins the current "not found" behaviour.

My view:

- **(a) Add a bounded path-suffix read. Recommended.** The note already rules path-suffix matching for `symbol`+`file` (and the implementation gets it by filtering `symbol_by_name`). Users write partial paths routinely, as QW-27 and QW-30 both show, so file+line must behave the same way. Today `falkorchat/api.py::f` resolves but `falkorchat/api.py:603` says "not found", which is incoherent. It is a capability, not a test accommodation. This is `graph-dba`'s call and an update to `-p1-graph.md` §5, for example a basename-equality read with the suffix filter applied in code (no scan). Several matching files give several candidates, so `ambiguous`, never a guess.
- **(b) Filesystem heuristic: reject.** It makes the resolver depend on the checkout rather than the snapshot, breaks AC-16 pin-and-replay (the read is not in `recipe.queries`), and is non-deterministic across machines.
- **(c) Amend the named test.** As a replacement this is a measurement-integrity problem. It changes a named test after seeing it fail, to make it green without any capability change, and leaves the dev item QW-27 itself unresolved. The dev entry check (≥ 11/12 unambiguous) would still count it, and one miss uses the whole slack. As a supplement the delegate's full-path variant (`test_qw27_partial_file_and_line_resolve…through_the_full_path`) is fine because it is labelled and does not replace the original. Keep both.
- If (a) is deferred: leave the strict xfail in place and record QW-27 as a known miss against the 11/12 budget in the dev entry report. Do not edit the item or the test.

**M3 (decision):** the stakeholder or `teco` picks (a) or the recorded known miss before the dev entry check. This is the only fork needing a human answer.

## Minor and nits

- **m1:** label the QW-3/QW-11 resolver tests as resolver-side only; add the real negative (interpreter null on a stated direction) to dispatch 2.
- **m2 (did-you-mean distance):** `_suggest` measures edit distance to the whole symbol part, so `Outer.helper` is penalised by its class prefix against asked `helper`. Compare to the last dotted segment (note says "edit distance"; this is a consistent reading). Verify against the ordering test.
- **m3:** `Node.js`/`e.g.` over-claim (see section 3).
- **nit:** the QW-47 test compares `options` to `goldOptions` as an ordered list; the scorer treats them as a set. Compare as a set plus the separate rank assertion.

## Growth-stops and REVIEW.md checks

- The new review documents no history; nothing restated from the live spec.
- Checks that can both fail and pass: the strict xfail flips on a fix; the guard tests assert zero reads; the `depends on` and cap defects above are planted-input tests to add (M1, M2). No `REVIEW.md` addition proposed.

## Required before dispatch 2 / prereg

1. M1: remove or tighten the `depends on` cue, with a planted test.
2. M2: cap only did-you-mean lists; record ordering and cap scope in `ccg-prereg-ac24-v1`.
3. M3: decide (a) or the known miss for QW-27.

## Pass 2

Verdict: **approve with suggestions.** M4 (new) should be fixed or explicitly accepted before the dispatch 2 prompt freeze. Rechecked: 152 front tests pass (run); `git diff HEAD -- cpg/eval` empty, so `cpg/eval/dev/ac24.jsonl` is untouched (verified).

### Dispositions

- **M1 (`depends on` cue): fixed.** `DIRECTION_WORDS` is now the literal tokens `impact` and `depends-on`. Probes: `depends on a/b.py:3` and `a/b.py:3 depends on` give `None`; `depends-on a/b.py:3` gives `depends-on`.
- **M2 (cap scope): fixed.** `options()` is candidate-major and never cut mid-candidate. Probes: 4 candidates × `null` gives 8 options; 12 candidates gives 20, whole candidates dropped past `OPTIONS_MAX`. CAP=5 remains on did-you-mean lists only (`_suggest`). The leniency and strictness biases are gone below 10 candidates. The dropped-whole-candidate edge at ≥11 candidates × 2 directions is an engine bound, which I accept.
- **M3 (QW-27): fixed, option (a).** The as-asked test `test_qw27_partial_file_line_as_asked_resolves_to_gold` has the same draft, question id and `as_gold` assertion as the earlier xfail body, with the strict xfail removed. It now also asserts the read order (exact reads, then `file_by_suffix`). The full-path test is kept. The dev item is untouched.
- **m1 (negatives labels):** not changed, still dispatch 2's real check. Carried.
- **m2 (distance on last segment): fixed** in `_suggest`.
- **m3 (`Node.js`, `e.g.` over-claim): fixed.** The `NOT_ANCHORS` list is generic, not dev-derived; probes give `None`.
- **nit (ordered gold compare):** not re-checked, cosmetic.

### Judged

**(2) Suffix read: consistent and measurement-safe.** It runs only when every exact read says "not found". A path held as a skipped file never falls back. It joins at a `/` boundary (`$sep + path ENDS WITH "/" + asked`), is bounded at `OPTIONS_MAX + 1`, and is deterministic (`ORDER BY`). That extends the §4.2 suffix rule from `symbol`+`file` to `file`+`line`, which is the same rule, not a new one. Several file hits give `ambiguous` and never a pick. Skipped-only gives `unresolved` with skipped-file lookups, which is the schema's case. The read names recipe queries, so AC-16 replay still holds. The caveat is that `file_by_suffix` is a label scan, which is a performance matter for `graph-dba`.

**(1) The new rule for a group that found nothing beside a resolving group: scorer-safe, but over-broad (M4).**
- Effect on the EP §10.4 scorer. It is gold-blind and only ever moves a question from `resolved` toward asking.
  - Unambiguous gold: now "not understood" through over-asking, which is reported with its own `k/24` rate. The change can only lower the pass rate and cannot flatter it.
  - Gold `unknown-name`/`no-anchor`: understood on `status != resolved`, unchanged.
  - Gold `several-symbols`/`two-use-cases`: options still contain the real readings (containment holds). Did-you-mean extras are added, which may push `|options|` above `CAP_i = max(5, |gold|)` and fail an otherwise correct list. That is stricter, not looser, and the extras are reported.
- It is safe for pre-registered AC-24/AC-5 scoring in direction, but note it is a rule the note did not have. Record it in the prereg record.
- **M4 (major, verified by execution).** `unfound` counts any group with no candidates, no lookups and no suggestion. A second group the interpreter adds with only an `other` span, or a redundant mapped code-file group, therefore turns a resolvable question into `ambiguous no-anchor`. Probe: `f`+`a.py` resolves alone, but with an extra `("thing","other")` group or an extra `("a.py","file")` group the result is `ambiguous`/`no-anchor`. A 4B interpreter producing a harmless extra group would burn the ≥11/12 dev entry budget through no fault of the anchor. Fix: let `unfound` count only a group that carried a name-bearing span the asker could have meant (`symbol`, `route`, `tool`, or `file`+`line`/config file that failed), and ignore groups made only of `other`/`server` spans or of a mapped file with no anchor form. Add the planted test above. If the team prefers strictness, accept it explicitly and record the expected cost.
- Minor: when several files match and there is at most one candidate, the kind is `no-anchor` yet `options` can hold that one reading. Harmless to the scorer (status `ambiguous`); state it in the spec section.

**(3) Fixes against each other.** I found no conflict: the suffix path feeds `_symbol`'s did-you-mean (`loc.mapped` hits) and the uncapped complete readings coherently; the `unfound` rule and M2 interact only through extras (above).

### For `ccg-prereg-ac24-v1`

1. Option order: candidate-major, impact before depends-on, rank 1.. in that order; complete readings uncapped except whole candidates past `OPTIONS_MAX` = 20.
2. CAP=5 scope: did-you-mean lists only, ordered by edit distance to the last dotted segment, then rank key.
3. `/`-bounded suffix join, after exact reads miss; skipped-only is `unresolved`; several files are `ambiguous`.
4. A group that names something found nowhere beside a resolving group asks (`ambiguous`, `unknown-name`/`no-anchor`), as amended by M4.
5. A deterministic-parser answer carries no `provenance` (no model call); only interpreter-read answers carry the fingerprint.
6. `no-anchor` with no failed path is `ambiguous`, not `unresolved` (schema); direction cues claimed by the parser are only the literal tokens `impact` and `depends-on`.
