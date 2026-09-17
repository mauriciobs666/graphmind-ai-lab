# `model-bench` S7 — `chat-responder` pack, Steps 0-2: code gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** U168 (S7)

## Scope & verdict

Reviewed against `docs/plans/small-model-benchmarking-s7-spec.md` (read in full), the following
uncommitted files: `modelbench/scoring/grounding.py` (new), `modelbench/report.py`'s two new
functions (`_render_role_caveat`, `_render_speed`) and their two `compare_report` call sites,
`packs/chat-responder-grounded-answers/pack.json` and `prompts/system.md` (new),
`tests/test_scoring_grounding.py` (new), and the new test classes in `tests/test_report.py` /
`tests/test_packs.py`. Out of scope: `items.jsonl`/`PROVENANCE.md` (Steps 3-4, not yet authored —
confirmed absent, correctly so) and any other uncommitted file in the working tree (unrelated
concurrent work, untouched by this review).

**CPG: considered, not relevant** — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(confirmed live, `GRAPHS`); this is a code-level task in a component with no CPG.

**Verdict: approve with suggestions.** The scorer, manifest, and report additions all match the
spec precisely — verified by direct reading, byte-for-byte manifest comparison, my own leak-proof
fixture, and two independent mutation probes (one on the already-closed `aggregate` finding, one
of my own choosing). Full suite green (1682 passed, 3 deselected — matches the expected baseline
exactly) and `ruff check .` clean. One real, spec-named gap (the `tests/test_runner.py` wiring
integration case) is missing from this diff; one genuine test-coverage hole I found independently
(`_format_directive`'s own wording is entirely untested). Neither blocks Steps 0-2 as delivered,
but both should close before the stage is called done.

## Findings

### Major — the spec-mandated `tests/test_runner.py` integration case is missing from this diff

§4's file/module layout table commits `tests/test_runner.py` (changed) to "one small integration
case confirming `_drive_single_call_items` reaches `grounding.build_messages`/`grounding.score_item`
end to end against a `FakePack`-shaped `chat-responder` fixture and a stub LLM (mirrors the existing
`guard-judge`/`nlq-generator` integration fixtures already in this file)". `git status` confirms
`tests/test_runner.py` is untouched, and `grep -n "chat-responder\|grounding" tests/test_runner.py`
returns nothing — no such case exists anywhere in the file.

This matters because the wiring seam it would prove is exactly the one §2.3 identified as
load-bearing: `_drive_single_call_items` (`runner.py:370-375`) resolves `build_messages` via
`getattr(scorer, "build_messages", None)` and silently falls back to the answer-key-leaking generic
`_item_chat_messages` if that lookup misses (e.g., a `pack.json` `"scorer"` typo, or
`_load_item_scorer`'s module resolution failing in a way `_scorer_problems`'s import-only check
doesn't catch). Every existing test proves `grounding.build_messages` itself never leaks
(`tests/test_scoring_grounding.py::TestBuildMessages`, and I independently reproduced this with a
standalone fixture) — but nothing proves the runner actually *reaches* it for a real
`chat-responder` pack. The risk is tempered (the identical generic path is already proven for
`guard-judge`/`nlq-generator`, and Step 6's live run will exercise this before publication) but the
spec named this test explicitly and it should land before Steps 0-2 are called complete — route to
whichever agent picks up the remaining Step 0 work, or open it as an explicit follow-up unit.

### Minor — `_format_directive`'s own wording is completely untested; my mutation confirms zero coverage

`grounding.py:92-106`'s `_format_directive` is the function that actually tells the model what word
budget / paragraph / forbidden-pattern constraints apply to *this* reply — the mechanism by which
`format_checks`' math has any hope of being satisfied by a real model. No test in
`tests/test_scoring_grounding.py` asserts anything about its output: `TestBuildMessages`'s
`test_system_message_starts_with_the_prompt_files_content` only checks `.startswith(...)` against
the static `system.md` content, never the appended directive.

I confirmed this is a real gap by mutation: replacing the function body with `return
"MUTATED-NO-OP"` (a completely wrong, unrelated string, in place of the real per-item constraint
sentence) leaves `tests/test_scoring_grounding.py` and `tests/test_report.py` **fully green** (169
passed, 0 failed). Suggested fix: add one or two assertions to `TestBuildMessages` — e.g. build a
`fmt` with a known `maxWords`/`mustBeSingleParagraph`/`forbiddenPatterns` combination and assert the
resolved numbers/clauses appear in `messages[0]["content"]`, plus one case for the
all-constraints-absent branch (`"no additional format constraints apply"`). The spec itself flags
the wording as "cheap to revise" (§3.4/§7), which is exactly why the mechanism producing it — not
the prose — deserves a pin: a future edit that silently drops a clause (e.g., forgets to append
`forbiddenPatterns`'s clause) would go undetected indefinitely.

### Verified — the already-closed `aggregate` finding is genuinely fixed

`TestAggregate::test_an_unrunnable_item_is_excluded_from_every_metrics_denominator` is present
exactly as named. I reproduced the regression it guards against directly: replacing `aggregate`'s
`declaring(name)` helper with `return list(items)` (i.e., no longer filtering on
`scoreable.get(name)`) makes this exact test fail (`assert 3 == 2`), while the rest of the suite
stays green — confirming the guard is load-bearing, not merely present. File restored afterward;
full `test_scoring_grounding.py` suite re-confirmed green (38 passed).

## What's solid

- `grounding.py` matches §3.4 precisely, function-for-function: `looks_like_abstention`,
  `resolve_format`, `checklist_pass`, `format_checks`, `build_messages`, `score_item`, `aggregate`
  all present with the exact signatures, merge order, and branch logic the spec specifies.
- §2.3's leak-freedom claim holds under both the shipped test and my own independently-constructed
  fixture (secret sentinel values in `mustContain`/`mustNotContain`/`provenance`/`mustAbstain`
  confirmed absent from every rendered message).
- `pack.json` is byte-for-byte identical (structurally) to §3.2's own JSON block; `prompts/system.md`
  matches §3.3's described contract (general prose contract, no per-item numbers hand-written in).
- `_render_role_caveat`/`_render_speed` are correctly additive-only: `git diff` on
  `modelbench/results.py`/`roles.py`/`runner.py`/`packs.py` is empty, the two new `report.py`
  functions are called at exactly the spec-described positions, `_render_speed` is gated on
  `run.latency is not None` and operates on the post-exclusion `runs`/`arm_names` pair (consistent
  keys, no `KeyError` risk), and `tests/test_report.py`'s own regression test
  (`test_regression_existing_guard_judge_report_is_unchanged_and_speed_is_a_pure_addition`) proves
  the new section is additive against a pre-existing fixture rather than asserted in isolation.
- Full suite green at the exact expected baseline (1682 passed, 3 deselected), `ruff check .` clean.
- Test style throughout mirrors `classification.py`/`extraction.py`'s established fixture and
  mutation-pair conventions faithfully (each format-check axis moves independently, the format
  merge is tested at all three arities, the `result is None` branch is tested for both
  `timeout`/`no_response`).

## Open questions

- Was `tests/test_runner.py`'s integration case deliberately deferred to a separate, later unit (a
  scoping choice already made by `teco`), or is its absence from this diff simply not yet done?
  The spec's §4 table names it as part of "this stage" without pinning it to a numbered step, which
  is itself a minor ambiguity in the spec worth `teco` resolving explicitly rather than leaving
  implicit. — **Resolved by Pass 2 below**: closed via U170.

## Pass 2 — 2026-09-17

**Verdict: approve.** Both Pass 1 findings closed by U170.

- **Major (missing `tests/test_runner.py` integration case) — fixed.**
  `test_drive_single_call_items_chat_surface_reaches_the_real_grounding_scorer_end_to_end`
  monkeypatches `_load_item_scorer` to return the real `modelbench.scoring.grounding` module (not a
  fake stand-in) and drives a real `chat-responder`-shaped item through `_drive_single_call_items`,
  asserting the sent messages carry `build_messages`'s own `CONTEXT:`/`QUESTION:` shape and that
  `score_item`'s real verdict (`groundingRate` counted correctly) comes through. I independently
  reproduced the regression this guards against: forcing `build_messages = None` in
  `runner.py`'s chat branch (simulating a scorer-resolution miss falling back to the leaking
  generic path) reddens exactly this test and the pre-existing fake-scorer fallback test, nothing
  else. File restored byte-identical afterward (`git diff --stat` empty).
- **Minor (`_format_directive` untested) — fixed.** `TestBuildMessages` gained
  `test_format_directive_names_all_three_resolved_constraints` and
  `test_format_directive_states_no_constraints_apply_when_all_three_are_absent`. I independently
  reproduced the original gap: replacing `_format_directive`'s body with a constant unrelated
  string reddens exactly these two new tests, nothing else in the 40-test file. File restored
  byte-identical afterward.
- Full suite re-run clean at 1685 passed, 3 deselected (matches `teco`'s own report); `ruff check .`
  clean.

No new concerns raised by this pass. Both fixes are additive, narrowly targeted at the finding they
close, and independently verified to actually catch the regression they claim to guard against
(not merely present).
