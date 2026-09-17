# `model-bench` S6 — code gate: Step 0/1 production diff

> **Status:** active · **Owner:** `analyst` · **Tracks:** U162 (S6)

## Scope & verdict

Independent code gate, unit U162, over `model-bench/modelbench/packs.py`,
`model-bench/modelbench/results.py`, `model-bench/modelbench/scoring/toolcalls.py`,
`model-bench/modelbench/report.py`, and their accompanying tests/fixtures — the whole diff is
commit `e8e5d9c` ("S6 spec Steps 0-1 - H-bound check, manifest fix, prose-detector wiring", U152).
The brief named `0794af9` as the commit of record; that hash is actually **Steps 2-4** (content
authoring: the 12 scripts, calibration corpus, `PROVENANCE.md`) — the production-code diff this
gate is scoped to is `e8e5d9c`, one step earlier in the same stage. `git diff e8e5d9c HEAD --
modelbench/packs.py modelbench/results.py modelbench/scoring/toolcalls.py modelbench/report.py` is
empty, confirming the brief's own claim that this diff is unaffected by later work — the hash
mismatch is cosmetic, not a scoping error. Reviewed against `docs/plans/small-model-benchmarking-s6-spec.md`
§2.3 (the `H` check's design), §2.5 (the prose-detector wiring), §4 Step 0/Step 1, and §5 Step 0/
Step 1's own "Done when" clauses. This is the first and only code gate for this diff.

**Verdict: needs changes.** One major finding (a real, reproduced crash risk in the new
`validate_pack` axis) is enough to warrant a fix-round before stage close, though it does not
threaten any already-shipped artifact today. Everything else — design fidelity, wiring, and test
coverage against the spec's own "Done when" clauses — checks out clean.

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(per the spec's own §1 confirmation, re-checked live in the S6 spec dated 2026-09-16); this is a
code-level task in a component with no CPG.

## Findings

### Major — `_clean_through_turn_h_problems` crashes instead of reporting a problem when `data.conversations` is declared but not yet authored

`packs.py:906-935`. The new axis calls `pack.iter_scripts()` unguarded. `iter_scripts()` opens
`self.data_path("conversations")` directly — if the manifest declares the key but the file does
not exist on disk, this raises `FileNotFoundError`, uncaught, out of `validate_pack` entirely
(`validate_pack`'s own docstring contract: "`[]` means valid... a fixture can fail one, several, or
none" — implying a list of problem strings is always returned, never an exception for a data-shape
issue). This is not hypothetical: it is the **exact, real, already-lived state of this stage**.
I reconstructed the real `tool-caller-shop-assistant` pack exactly as it stood at commit `e8e5d9c`
(manifest declares `data.conversations`/`data.prosePseudoCallCalibration`, `sampling.scripts: 12`,
`metrics.cleanThroughTurnH.H: 4`, but — confirmed via `git show e8e5d9c:.../conversations.jsonl`
failing with "exists on disk, but not in e8e5d9c" — no `conversations.jsonl` yet, per the spec's own
§2.1) and ran `load_pack`/`validate_pack` against it with the current code:

```
CRASHED: FileNotFoundError [Errno 2] No such file or directory: '.../conversations.jsonl'
```

The codebase already has a stronger precedent for exactly this scenario:
`_row_count_identity_problems` (`packs.py:670-678`) wraps its own read of the same file in
`try/except (OSError, json.JSONDecodeError)` specifically so a not-yet-authored `conversations.jsonl`
degrades to a reported problem, not a crash. `_clean_through_turn_h_problems` should do the same —
wrap `pack.iter_scripts()` (or the `data_path`/`open` call inside it) and turn an `OSError` into a
problem string, mirroring `_row_count_identity_problems`'s own message shape. Note in fairness:
`_answerability_stamp_problems` (`packs.py:881-898`, pre-existing, not part of this diff) has the
identical unguarded-iterator weakness against `items.jsonl` — this diff followed the weaker of two
existing conventions rather than the stronger one, so this is a missed opportunity to close a
latent gap, not a wholly novel pattern. Blast radius today is zero (the real pack now has
`conversations.jsonl`, landed in the later commit `0794af9`), but the next pack scaffolded with
`metrics`/`sampling` before its `conversations.jsonl` exists — this pack's own history, one commit
later — will hit this again.

I also mutation-tested the adjacent guard directly: removing `if not scripts: return []` (the
line guarding `min()` against an empty-but-present `conversations.jsonl`) and re-ran the full suite
— **1625 passed, 3 deselected, no failures** — confirming that branch, too, is unguarded by any
test (would raise `ValueError: min() arg is an empty sequence` on a pack with rows but zero
scripts). Reverted immediately after (`git diff --stat` on `packs.py` empty post-revert). Same root
cause, same fix: wrap the whole `pack.iter_scripts()`-through-`min()` sequence in the file's
existing `try/except OSError` idiom, and add a fixture/test pair for "declares the axis, file
absent or empty" the same way Step 0 already built one for "declares the axis, `H` too large."

### What's solid

- **Design fidelity is exact.** `_clean_through_turn_h_problems` mirrors `_answerability_stamp_
  problems`'s shape as the spec specifies (scoping guard, single new problem string naming pack/
  `H`/offending script/its turn count), wired into `validate_pack`'s call sequence at the documented
  spot, with its own docstring naming the plan citation. The new axis is genuinely independent of
  the other seven — it reads `metrics.cleanThroughTurnH.H` against turn counts, nothing any other
  axis touches, and does not shadow `_row_count_identity_problems`'s distinctness/count checks.
- **`Pack.iter_prose_calibration()`'s error handling matches `data_path`'s own precedent exactly**
  (confirmed by reading `data_path`, `packs.py:260-265`, and by the passing test
  `test_iter_prose_calibration_raises_on_a_missing_manifest_key`, which asserts the `PackConfigError`
  fires with the key name in the message).
- **The `prosePseudoCallDetector` `None`-vs-populated distinction is correct on both sides.** Scorer
  side: guarded by manifest-key membership (never a bare `try/except`, per spec), and correctly
  propagates `None` when `prose_detector_precision_recall` itself returns `None` (empty corpus) —
  a slightly wider `None` condition than the spec's literal text ("`None` when the manifest declares
  no calibration data at all") but a reasonable, correctly-reasoned extension: an empty corpus is
  as measurement-void as a missing key, per `prose_detector_precision_recall`'s own docstring rule.
  Renderer side: the two message shapes match the spec's literal wording exactly, confirmed by
  `test_render_funnel_prints_the_prose_detector_precision_recall_when_populated` (deliberately
  distinct precision/recall values, catching a swapped-field mutant) and
  `test_render_funnel_names_the_prose_detectors_absence_when_none`.
- **Test coverage matches every "Done when" clause I checked.** Both directions of the `H` check
  (within bounds / exceeding, `test_validate_pack_accepts_an_h_within_every_scripts_length` /
  `test_validate_pack_rejects_h_exceeding_the_shortest_scripts_turn_count`), the "absent either key"
  scoping case, both calibration present/absent cases on the scorer side and the renderer side, and
  a round-trip test for the new `results.py` field (populated and `None`).
- **The shared `valid` fixture's `conversations.jsonl` upgrade is correct and necessary.** Before
  this diff it held a bare two-field row shape (`{"scriptId", "turnIndex"}`) that `iter_scripts()`
  would have rejected with a `KeyError` had anything ever called it — nothing did, since only the
  row-count-identity check read this file, and only via raw JSON. `_clean_through_turn_h_problems`
  is the first caller to actually parse this fixture through `Conversation`/`Turn`, and the diff
  correctly upgraded all 12 rows to the real schema rather than leaving a fixture that only
  accidentally validated.
- **The `pack.json` three-field manifest correction (§2.4) is exactly and only what the spec
  specifies**, confirmed by diffing the shipped file: `historyReplay` → `structured-replies-only`,
  `representToolSchemasEachTurn` → `true`, `maxTokens` → `1024`; `packVersion` correctly left at
  `0.1.0` in this commit (confirmed via `git show e8e5d9c:.../pack.json`), bumped to `0.2.0` only in
  the later Step 2-3 commit as the spec requires.
- **Suite and lint are green.** `.venv/bin/python -m pytest -q` → `1625 passed, 3 deselected in
  7.8s` (the 3 deselected are the `live`-marked LM Studio tests, correctly out of scope for an
  offline gate). `.venv/bin/ruff check .` → `All checks passed!`.

## Open questions

None — the one finding above has a concrete, mechanically obvious fix (wrap the new axis's file
read the same way its sibling already does) and does not require a design decision from anyone.
