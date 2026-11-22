# Connected code graph: P1-35 prompt-edit unit, implementation gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-35)

CPG: considered, not relevant — a prompt text, its pins and one harness string; no call-graph question.

## Scope and verdict

Commit `2bf875bd` (`cpg/ccg/front/prompts/interpret.txt`, `cpg/tests/test_front_prompt.py`, `cpg/eval/tools/ac24_smoke.py` one string, `cpg/tests/test_ac24_smoke.py`) against `docs/plans/connected-code-graph-p1-interpreter-design-fault.md` §4-§5, `docs/plans/connected-code-graph-p1-interpreter-model2-ml.md` §3 and root `REVIEW.md`. The wording's dev-set leakage is the data-scientist's gate and is not judged here.

**Verdict: approve with suggestions.** No blocker. All nine pins exist and every spec-named mutant reddens its pin. Two minor findings and three informational limits.

Run (from `cpg/`, `.venv/bin/python -m pytest`): `tests/test_front_prompt.py tests/test_ac24_smoke.py tests/test_front_interpret.py` exit 0, 268 passed. Full `tests` run timed out (exit 124, Appendix A): unverified beyond the three files. Mutants were run on a scratch copy of `cpg/` (the real tree never touched; `git diff --quiet HEAD -- ccg eval tests` clean afterwards).

## Findings

### m1 (minor) Module docstring of `ac24_smoke.py` still states the retired recommendation
`cpg/eval/tools/ac24_smoke.py:11` reads `a failure, or more than 1 of the 14 replies unusable, reports "evaluate a larger model" and never "tune on dev"`. The unit changed the string at line 552 but not this sentence, which now describes output the code no longer prints and repeats the EP §10.2 contradiction the unit set out to remove. Unfiltered `grep -rn "larger model\|tune on dev\|never tune"` over `cpg/` and `docs/` (excluding reviews and HISTORY): the only other live hit is the plan's own quotation of the old string. Fix: reword line 11 to the new recommendation. No pin needed (prose); a pin on the docstring would be noise.

### m2 (minor) Two spec deliverables not in the commit
(a) Plan §4 "report prompt length in tokens, old and new, in the commit": the commit message has no token figures (not a gate, but owed). (b) Plan §5 asks the optional `promptHash` key in `validate_dev_attempts` in the same commit as the string fix ("one `harnessHash` change"); `ATTEMPT_KEYS` (`ac24_smoke.py:641`) is still exactly four keys, so a `devAttempts` entry cannot carry the per-try prompt hash the record must disclose. The coordination ledger (line 1144) records this as a deliberate separate harness unit. Acceptable because no record is built yet and `harnessHash` is only taken from the final tree; it must land (red pin first) before `prereg-record`, and the dev2 tries must not be recorded without it. Track it as an open unit so the dev2 canary step does not start without it.

### i1 (informational) Pin 9 covers identifier-like dev vocabulary only, case-sensitively
Planted by me, all against the committed prompt copy:

| mutant | pin 9 / suite |
|---|---|
| prose gains `readMessages or HTTPS` (outside dev vocabulary, inside the "protocol or product label" class) | 19 passed: not caught |
| example question gains `OAuth` | 19 passed: not caught (the `fx-` pin checks spans, not question text) |
| prose gains lower-case `mcp` | 19 passed: not caught (dev token `MCP` is matched case-sensitively) |
| `MCP`, `postMessage`, `send_message`, `postMessage` inside the new example | each reddens pin 9 |

This is what plan §4 item 9 specifies (dev vocabulary, all-caps/underscore/inner-capital tokens), so it is not a defect of the unit, and pin 9 is not vacuous: its three planted mutants and the `{"MCP","postMessage","send_message"} <= ids` positive control (the extractor sees what it must) make it fail. It does leave "made-up `fx-*` names only" for example *questions* and prose unenforced; a future edit adding a real product name outside the dev set passes every pin. If wanted, add a pin that every identifier-like token in the example questions starts with `fx`; not required for this gate.

### i2 (informational) Pin 9 passes on the old prompt
As the brief says. It is therefore a regression guard, not evidence the edit is correct; the mutants above are its evidence.

### i3 (informational) Cue-count pin boundary
`>= 3` of the six null cues: dropping to four passes (as intended), dropping to two (`m6c`) reddens. Boundary is right.

## Mutant results (all on scratch copy; "red" names the pin)

| Pin / spec mutant | Result |
|---|---|
| 1 delete one new example (last, and the PUT one) | red: six-examples pin (the PUT deletion also reddens the R1 route pin) |
| 2 R1 delete the verb sentence; also drop the verb from the route example span | red: route-verb pin, both |
| 3 R2 delete the three-sentence span rule | red |
| 4 R3 delete the parameter sentence | red |
| 5 R4a swap the two cues; also add a bare "calls" to the `depends-on` part | red, both |
| 6 R4b remove the class sentence keeping every cue word | red |
| 7 R5 replace by `MCP`; replace `(FXQ)` by `(MCP)` | red (R5 pin and pin 9) |
| 9 insert `MCP`; `postMessage`; `send_message` | red each |
| 8 standing: non-`fx` identifier in example span | red (example-span pin) |

Each mutant was first checked to differ from the original text (a no-op raises); the file was regenerated from `HEAD` before every mutant.

## Consumers of `promptHash` and the old hash `494fb949…`

Old hash recomputed: `git show 2bf875bd^:…/interpret.txt | sha256` begins `494fb949`; new begins `f91e01b0` (matches the commit message).

Unfiltered `grep -rl "494fb949"` over the repo (excluding only `.venv`, `node_modules`, `.git`): three files, all prose, no code, no result file, no test.
- `docs/plans/connected-code-graph-p1-interpreter-design-fault.md`, `...-model2-ml.md`: the spent-record disclosure the old-try `devAttempts` must cite. Ruling: no change (they name the old hash on purpose).
- `docs/plans/connected-code-graph-coordination.md:1144`: a dated ledger line. No change.

Other consumers: `ccg/front/interpret.py:66` (`prompt_hash()` reads the file; `:227` provenance; `:220` `cache_mod.key_of(...)`) all derive, nothing hard-coded. Tests build the expected value from `sha("prompts/interpret.txt")` / `I.prompt_hash()` (`test_front_interpret.py:152-155`, `test_front_prompt.py:227`), and `test_front_interpret.py:283-294` pins that each cache-key input, `prompt_hash` included, changes the key (the "two prompt texts give two keys" pin the plan asked for already exists). `coreTreeHash`/`packSetHash` are computed from the tree (`ac24_smoke.py` `core_tree_hash`) and tests use `tmp_path`; no stored digest. The canary digest is a digest of model replies written only into a run/record file, none in the repo; `CANARY_QUESTIONS` unchanged; re-record is the dev2 canary step. No `prereg-record` exists yet. Hit-by-hit: all ruled, none edited.

Grammar: `git diff 2bf875bd^ 2bf875bd -- cpg/ccg/front/grammar` is empty, so `grammarHash` is unchanged; both new example replies validate against it (`test_there_are_six_examples...` runs `Draft202012Validator` and `to_draft` over all six). Example count is six, ceiling `<= 6` and `== 6`, as plan §4.

## Harness string

`ac24_smoke.py:552` now equals the plan's string character for character (grep -c of the full literal: one hit in the tool, one in the test). The test pin is `==` against an independently typed literal, so any edit to the string reddens it. The text no longer says "never tune on the dev items" and agrees with the model2 note §2 (no edit after dev2 exists; held-out never used for tuning). It also prints on a dev1 failure, where "after dev2 exists" is still true. The only residual contradiction is m1.

## What is solid
- Pins compute expected values from independent literals and the dev file, not from the prompt; mutants were all killed on first try.
- The standing pins were kept (n-gram, dev-question, example-validity, example-axes, `fx-` spans); one retired pin (`"relations" not in words`) is replaced as the plan directs.
- `resolver`, `grammar`, `SAMPLING` untouched.

## Open questions
None for the owner; m2(b) needs a scheduled harness unit before the record is built.

## Appendix A: full suite
Full `cd cpg && .venv/bin/python -m pytest tests -q` hit my 580 s timeout (exit 124, no result): not a green claim. Left unrun: the rest of the suite beyond the three files above; the unit touches only the prompt file, two test files and one string, and the ledger records `test_front_prompt` + `test_ac24_smoke` green (189) by `teco`.
