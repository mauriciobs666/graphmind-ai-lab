# Review: C-1001 P1-24, the answer renderers (diff-scoped)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Uncommitted `cpg/ccg/render/` (`__init__.py`, `common.py`, `text.py`, `markdown.py`), `cpg/tests/test_render.py`, `cpg/tests/golden/render/answer-ac1.{txt,md}` and the `ccg/render/` README row. Baseline: the P1-24 row in `docs/plans/connected-code-graph-p1.md` and its section in `docs/plans/connected-code-graph-p1-answers.md`; `docs/plans/connected-code-graph-answer-spec.md` §7.1-§7.3 and `-answer-spec2.md` §3-§4; root `REVIEW.md` applied (no `cpg/REVIEW.md` exists). Not reviewed: the normalize README row, `ccg/answer.py`, sandbox/infra/broker/packrt.

**Verdict: approve with suggestions.** F-3 and F-4 hold under every hostile input I planted, in both formats, against a real CommonMark parser. One major (minted fields trust the validator, including the LLM-bound `summary_input`) and a few minors, all documentation or a one-line guard. No blocker.

CPG: considered, not relevant — new, uncommitted, self-contained module whose only inbound dependency is `ccg.answer`; the committed graph does not contain it.

## Checks run

- `tests/test_render.py`: 128 passed. Full `cpg` default suite (`.venv/bin/python -m pytest`, default markers): 6105 passed, 1 skipped (the slow leaf-mutation test), 15 deselected (live).
- **F-3 hostile fuzz (executed).** Planted by key name (my own walker, not `DATA_FIELDS` or the delegate's `plant`) into 9 documents (rich, ambiguous, group, launch, wire, stale, unit, flow, 1.1 provenance), 8 payloads (CommonMark/HTML `[x](javascript:..)`, `![i](..)`, `<script>`, `<!--`, `]]>`, `# H1` and list/quote starts; backtick runs 1 to 5; `\n`, `\r\n`, U+202E, U+200B, U+2028, ESC `[31m`, NUL; 700-char value ending in a backtick plus an image; the prompt-injection sentence; value that is one backtick-wrapped word; whitespace-only; combining mark before a backtick), both formats: 144 renders. Result: zero marker occurrences outside code spans under `markdown-it-py` 3.0.0 (CommonMark; tokens inspected, so none is a `link_open`, `image`, `html_inline`, `html_block`, `fence` or `code_block`), and zero under the delegate's regex and under a per-line variant of it. **Spread across the three definitions: none.** Separately, 3,000 random strings over a hostile alphabet plus 19 hand-picked edge values (empty, spaces, 499/500/501-char cuts at a backtick) through `fence_data`, embedded in a sentence, a list item and a heading: always exactly one `code_inline` and no stray inline token.
- **F-4 (executed).** A 25-letter segment placed in every F-4 site (hop, `Bound`, config subject and anchor `keyPath`; `config_key`, `env_var`, `http` `keyNorm` in hop and anchor; `Launch.mountPath/service/envVar` in a hop and in `Bound.launch`) across 7 documents, both formats, 88 renders: never reaches the output.
- **Mutation (item 8), on a copy in `scratchpad/mut24_x7`, never in place.** 43 mutants applied to a private copy of `ccg/` and `tests/` (unfence or unmask each data site in turn, drop `data()`'s fence or its mask, summary leaking a path / `qname` / `Unit.reason`, port derived from `launch`, `viaData` from the flag, depth `>` to `>=`, status rank flipped, lifted order unsorted, `line` dropped, commit slice 12 to 7, groups' members dropped in each renderer, grouping key ignored, hop details dropped from Markdown). Of the 41 that applied, 40 are killed; 2 of my first list were no-op edits and were discarded. One survived: see m3.

## Findings

### Major

**M1. The renderers and `summary_input` trust every non-`DATA_FIELDS` string; a document the validator rejects renders (and reaches the summary prompt) raw.** Verified by execution. The docstring at `ccg/render/common.py:11-12` says "F-3 and F-4 are applied regardless, so a builder bug cannot reach a client through the text"; that holds for the 35 data fields only. Minted fields are printed as they are: `Item.reason`, `Hop.reason`, `Group.reason` (`reason_text`, `reason_text` in `summary_input`'s `reasons` dict), `Provenance.endpoint` (`common.py:164`), `Unit.lang`, `Item.repoId`, `Launch.port` (`launch_text`: `f"{f} {v}"`), `Bound.port`. I set each to `PWNREASON <script>x</script>` or similar on `b_possible`, `b_group`, `b_flow`: `validate` rejects, and `render_text`, `render_markdown` and `summary_input` all emit the string verbatim (`summary_input` as `reason <string>: n`). `summary_input` is the one output the plan says must carry only engine-minted text (P1-24 row; spec §7.2's "aggregate-only summary's input"), and it is the one whose consumer is a model.

Why it matters: the validator is the stated safety net (spec §7.3 "so a builder bug cannot reach a client"), and nothing in P1-24, P1-25 or P1-26 says a caller must run it first. A cache hit, a replayed pinned answer, or a hand-built document would skip it.

Suggested improvement (design, not run beyond the plant above; the assertion that would catch a wrong fix is a test that plants the strings above and expects `summary_input` to contain neither): (a) in `summary_input`, count a `reason` only if it matches the schema's reason pattern `^[a-z][a-z0-9-]*(:[a-z0-9]+)?$` (else the bucket `other`), and likewise `buildStatus` and `direction` against their enums; (b) correct the `common.py` docstring and README row to "applied to data fields; minted fields are trusted to `validate`", and add to the P1-25/P1-26 rows "calls `answer.validate` before `render_*` / `summary_input`". Fixing it by fencing minted fields too is the stronger option but changes the plain-language templates; the architect's call.

### Minor

**m1. Plan row P1-24 lists `qname`, `Cond.code`, `calleeName` under F-4 masking; the delegate's reading (F-3 fence only) is right, and the plan wording is the defect.** `docs/plans/connected-code-graph-p1-answers.md` P1-24 "Behaviours" says those fields "pass `ccg.answer.mask_identifier` (F-4)". `answer.F4_KINDS` is the closed set `keyPath, config_key, env_var, http, mountPath, service, envVar` and `mask_identifier` raises on any other kind (`ccg/answer.py:377-381`). Answer spec §7.3 gives exactly that set; `-bundle-spec2.md` F-4 (row at line 972) names only `ConfigEntry.keyPath`, `config_key`/`http` keys and `launchFact.mountPath/service/envVar`; the list with `Cond.code`, `calleeName`, `qname` in `-answer-spec.md` §7.2 is F-3's. So not a spec gap and not a missed item: a plan wording slip that conflates the two lists. Fix: `architect` edits the plan row to put those three under "fenced (F-3)" only. The delegate's test `test_qname_and_callee_name_are_fenced_but_not_masked` pins the correct reading. (Verified: read both specs; ran the F-4 probe.)

**m2. P1-23's "folded in rendering only at or above the 0.95 precision threshold" has no owner; P1-24 correctly does not do it, but the sentence is ambiguous.** Nothing in P1-24's section requires it, and no input to a renderer carries a precision figure (the document has `groups`, no threshold; the 0.95 is a fold-or-list decision per language, assigned to `data-scientist` in the N-ML note, `docs/plans/connected-code-graph-p1.md:431`, item 1). The renderers print `groups` as the builder made them and list every member (`counts_line`: "all listed below"; `test_every_folded_member_is_listed_with_its_own_chain`), which satisfies spec §6 "folded groups enumerate every member". Consumers get what they need: P1-25 gets `render_text`/`render_markdown`/`summary_input`; P1-26's summary gets `summary_input`. Fix: P1-23 (or P1-39) states that the fold decision is made by the builder (it either emits a group or lists the members as items) and the renderer only lays out what it is given.

**m3. Gap: `Scope.repoIds` rendering is unpinned.** Mutant (`repoIds[:1]` in `request_text`) survived the 128 tests. The values are pattern-minted, so low stakes; one assertion in a test that renders a `scope` with two `repoIds` closes it.

### Nit

- The F-3 content tests define "inside a code span" with the delegate's own regex (`_SPAN`, `test_render.py:34`). It agrees with CommonMark on this renderer, shown above, but is not independent; `markdown-it-py` is importable by system python here but not in `cpg/.venv`, so adding it as a `dev` dependency for one cross-check test is optional.
- `render_*` of a resolved answer with zero items prints "0 items listed (0 confirmed, 0 possible)." and never the words "no impact" (verified on the golden with `items=[]`, with and without a `not-mapped` unit); it also adds no "may be incomplete" caveat when a unit is `not-mapped`. Within the plan's wording; mention only for P1-25's text.

## What's solid (verified)

- F-3 fence: every data site goes through `data()`; spread across CommonMark, regex and line-wise definitions is nil; the 500-point cut and `...[+N]` count appear inside the span (`test_a_long_value_is_cut_at_500_code_points_and_counted` plus my 700-char plant).
- F-4: all 13 field-site shapes masked before fencing; fail-closed on non-tokenizing input (`test_a_keypath_that_does_not_tokenize_fails_closed`); expected masked strings are written out literally in `f4_cases`, not read from the code.
- Item 3: no data field reaches `summary_input` (11 documents, every data key planted with a marker, 0 leaks; three leak mutants killed). The caveat is M1.
- Item 5: `has_data_hop` reads the chain (flag-true with no `DATA` hop says nothing of data flow; flag-false with a `DATA` hop says "via data flow"); `port_text` uses only `Bound.port`/`portOrigin` (a mutant that falls back to `launch.port` is killed); the depth note reads both fields and uses `> 0`; neither format says "no impact" for an empty, ambiguous or unresolved answer.
- Item 7: the text golden is independently checked by `test_the_golden_text_states_each_fact...` (seven phrases written out). I compared both golden files to `answer-ac1.json` by hand (5 hops, lines 16-23/954/17/107/73/118, `PROVIDES`/`TO`/`VIA`/`HAS_CALL`, the four `LIFTED_VIA` entries in `ord` order): correct. The Markdown golden is pinned by the same builder only; the format-parametrized content tests cover its sentences, so the exposure is formatting only.
- Item 6, scope of the answer: the renderers do not call `validate`; for the 35 data fields that is safe (F-3/F-4 applied unconditionally); for everything else see M1.

## Open questions

1. M1 remedy: validate in the callers (P1-25/P1-26 plan rows) plus a `summary_input` guard, or fence minted strings in the renderers too? Recommended: the first; it needs no template change.
