# `rank --reference` on a continuous verdict metric — implementation review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** —

## 1. Scope & verdict

Reviewed: the uncommitted working-tree diff of `modelbench/report.py`, `modelbench/stats.py`,
`tests/test_report.py`, `tests/test_stats.py` (`git diff` against `HEAD`, +1051/-37 lines) —
`tdd-engineer`'s implementation of D-1's fix, against `docs/plans/rank-continuous-reference.md`
(`architect`) and its prerequisite `docs/plans/rank-continuous-reference-ml.md`
(`data-scientist`), both already gated `approve with suggestions` in
`docs/reviews/rank-continuous-reference.md` / `-ml.md`. Not re-reviewed: the plan's own design —
that gate already happened; this pass is conformance (does the diff do what the plan says) plus an
independent code-quality/test-coverage pass on what actually landed. Baseline suite size at the
plan-review gate was `1778 passed, 3 deselected`; I re-ran it myself (§2) rather than trusting that
number forward.

**Verdict: approve with suggestions.** No blocker. The diff is a faithful, conformant
implementation of the plan; the full suite is green (`1798 passed, 3 deselected` — the expected
+20 new tests); `ruff check .` is clean; both live acceptance runs (D-1's exact repro, and the
`guard-judge-understanding` regression spot-check) exit 0 with correct output. One major finding —
a real, confirmed test-coverage gap of the same class the coordination round already found and
fixed once in this unit, in a sibling location it didn't reach — plus two minor notes.

**CPG:** considered, not relevant — per the plan's own §1 (`GRAPHS` lists no `cpg_model-bench`) and
the prior plan review's independent re-confirmation; still true, and this is a code-level component
with no loaded CPG, not a task with no code-level component at all.

## 2. What I verified, and how

- **Full suite, real run:** `.venv/bin/python -m pytest -q` from `model-bench/` →
  `1798 passed, 3 deselected in 8.55s`.
- **Lint, real run:** `.venv/bin/ruff check .` → `All checks passed!`
- **Live repro of D-1, real run:** `./run.sh rank --pack embedder-graphrag-retrieval --reference
  bm25` → exit 0, renders `` `alpha_family=0.05, k=4, alpha_used=0.0125 (98.75% CI)` `` and a
  four-candidate continuous table, all `not distinguishable` at this data (plausible: `bm25` isn't
  far off the embedding candidates at n=38) — no `MetricKindError`, no stray "pp"/"80% power"
  sentence for `mrr` anywhere in the report (§2.3's adjacent defect confirmed fixed live, not just
  in tests).
- **Live regression spot-check, real run:** `./run.sh rank --pack guard-judge-understanding
  --reference "google/gemma-3-4b"` (a real stored model key, `results/runs/`) → exit 0, renders the
  old binary shape exactly (95% CI column, Holm-adjusted threshold column, "not tested (Holm stops
  here)" states) — no visible regression against the pre-existing shape.
- **Plan-review suggestions folded in, checked against the actual diff, not the plan's own claim:**
  M1 (`stats.alpha_used` shared helper, `modelbench/stats.py` new function + both call sites
  updated) — present and used by both `continuous_verdict` and
  `_render_reference_family_continuous`'s caption. m1 (per-candidate `_pairing_tally` under the
  ordinary-mixed refusal) — present, and pinned by
  `test_rank_report_reference_family_mixed_pack_refuses_whole_no_partial_table`
  (`tests/test_report.py:4107`, asserts `"paired n: 5 of 5"` inside the refused section). m2
  (explicit `resolved_kinds == {"continuous"}` instead of a bare `elif`) — present at
  `report.py`'s `rank_report`, with a comment citing the finding by name. m3 (`_rank_resolving_
  power_lines`'s coarser kind check) — correctly left as-is per `data-scientist`'s own "not
  blocking" verdict on that finding; not a defect in this pass.
- **The named integration test, hand-traced, not just read for existence:**
  `test_rank_report_reference_family_continuous_row_uses_the_real_combined_k_not_k1`
  (`tests/test_report.py:3970`) builds `diffs` engineered (at the real `B=10000`, `seed=pack.seed`)
  so the flip candidate is `distinguishable` at the uncorrected `k=1` and `not distinguishable` at
  the real combined `k=2` (one flip candidate + one filler candidate, so `correction_k =
  len(family) * len(candidates) = 1*2 = 2`, not the trivial 1). It asserts the rendered row says
  `"not distinguishable"` — which only holds if `correction_k=correction_k` actually reached
  `stats.continuous_verdict`, not a reverted `None`/`k=1`. I confirmed the fixture's premise by
  reading `stats.continuous_verdict`'s bootstrap path and the `_family_ci_levels` widening
  directly: this is a real, discriminating assertion, not a vacuous one.
- **The `guard-judge-understanding` byte-for-byte golden** (`_GUARD_JUDGE_REFERENCE_FAMILY_GOLDEN`,
  `tests/test_report.py:4291`) — read in full: it is a genuine, fully-computed multi-table markdown
  render (real Wilson intervals, real Holm thresholds, real MDD/floor sentences), not a stub or a
  substring check. `assert md == _GUARD_JUDGE_REFERENCE_FAMILY_GOLDEN` is a real equality pin.

## 3. Findings

### Major

**M1 — `_render_reference_family_continuous`'s `design_effect`/`basis` selection is untested and
silently corruptible, the same class of gap the coordination round already found once in this
unit's `correction_k` plumbing (`tests/test_report.py:3970`), in a sibling argument it didn't
reach.** `modelbench/report.py`'s new `_render_reference_family_continuous` (around line 983)
calls:

```python
design_effect=max(reference_run.designEffect, candidate.designEffect),
basis=min((reference_run.basis, candidate.basis), key=_BASIS_STRENGTH.__getitem__),
```

mirroring the pre-existing binary path's identical pattern (`report.py:1474-1475`) — the direction
is correct by inspection (`max` design effect and the weaker/`min`-strength `basis` are the
conservative choices, consistent with `AGENTS.md`'s "nothing that shapes a decision carries a
default" posture). But every fixture that reaches this new function
(`_rank_mrr_arm`, used by all of tests 9-19 in the plan's own numbering) builds arms through
`tests/conftest.py`'s `run()` helper without overriding `design_effect`/`basis`, so every arm in
every continuous-reference-family test carries the same default `design_effect=1.0,
basis="by-construction"`. I confirmed this is a real, not hypothetical, gap by mutation-testing it
directly — in an isolated scratch copy, never touching the tracked working tree — flipping
`max→min` (design effect) and independently `min→max` (basis) in this one call site and re-running
the full `tests/test_report.py` suite (`174 passed` unchanged both times, evidence in the
appendix). Getting this backwards in a real edit would silently narrow the printed CI (an
anti-conservative interval) for a candidate whose own basis or design effect differs from its
reference's — exactly the "wrong-but-plausible-looking number" failure class this component's own
convention says must never ship unpinned, and exactly the class of defect this same unit's second
review round already found and closed once (for `correction_k`) without noticing the sibling gap
one line below it.

*Suggested fix:* one integration test, mirroring
`test_rank_report_reference_family_floor_demotion_fires_at_the_candidate_axis_correction_k`'s own
shape (`tests/test_report.py:3866`, which pins the binary path's identical `max(designEffect)`
selection by setting the *reference's* `designEffect=2.0` while candidates stay at the default):
build a continuous reference-family fixture where the reference and at least one candidate carry
different `designEffect`/`basis` values, and assert the rendered decision or CI width differs from
what the *lower* design effect / *stronger* basis would have produced — not merely that a number
appears, but that the choice of `max`/`min` (not `min`/`max`) is what the test would catch if
reverted. Land this as its own test rather than folding it into an existing one, the same way
`test_rank_report_reference_family_continuous_row_uses_the_real_combined_k_not_k1` was landed as
its own test for `correction_k` — a coverage probe over "every non-constant argument this new
function forwards into `stats.continuous_verdict`," not a single hand-picked case, since the next
one found this way (there is no guarantee `design_effect`/`basis` are the last such argument).

### Minor

**m1 — The plan's own §5 citation nit (m4 from the plan review) is still open in the plan
document.** `docs/plans/rank-continuous-reference.md:721` still attributes `run(...)`,
`embeddings_fields`/`deterministic_fields` to `tests/test_report.py` rather than their real home,
`tests/conftest.py` (unchanged since the plan review flagged it). Zero effect on the shipped code —
the names resolve either way — but it's a loose end on a document the plan-review gate already
flagged; worth a one-line fix whenever that document is next touched, not urgent enough to block
this implementation review.

**m2 — `docs/plans/rank-continuous-reference-coordination.md`'s U4 (BACKLOG/HISTORY/manual
cleanup) is still queued, correctly.** Not a defect — the plan's own step 9 correctly scopes the
manual-callout removal to a later step owned by `tico`/`teco`, and the coordination ledger shows
U4 queued behind this unit's gate. Flagging only so the reviewer of *that* step knows this
implementation review does not cover it — `docs/BACKLOG.md`'s D-1 entry and
`docs/manuals/small-model-benchmarking.md`'s known-issue callout are both still open as of this
review.

## 4. What's solid

- **Faithful to the plan on every load-bearing point I checked against the real diff, not the
  plan's narration of it:** `stats.continuous_verdict`'s `correction_k` parameter, the combined
  `k = len(family) * len(candidates)` divisor, the `_resolve_reference_kinds` N-ary resolver (and
  its deliberate non-reuse of `_metric_kind`'s pairwise-anchored shape), the three-way dispatch in
  `rank_report`, the four-state continuous decision vocabulary, and the `ContinuousVerdict.text`
  coverage-label fix are all present, in the shape and at the call sites the plan specifies.
- **The M1 plan-review finding was fixed correctly, not just acknowledged** — `stats.alpha_used` is
  a genuine one-line, one-home helper, called from both sides of the seam the finding identified.
- **The regression discipline is real, not asserted.** The binary combined-ladder construction
  block is textually untouched (only newly gated behind `resolved_kinds == {"binary"}`), and the
  byte-for-byte golden plus the two pre-existing mechanism tests all pass unchanged. I confirmed
  this live, not just via the suite.
- **The specific gap the brief flagged (`correction_k` silently reverting) has a real, correctly
  targeted regression test** — traced by hand against the actual bootstrap/`_family_ci_levels`
  mechanics, not merely present by name.
- **Lint and both live acceptance paths are clean.**

## 5. Open questions

None that block approval or need the requester's input — M1 above is actionable by the implementer
without further design input (it is a test-coverage gap in already-correct code, not a design
question).

## Appendix — M1 mutation evidence

Performed in an isolated copy under the session scratchpad (`/tmp/.../scratchpad/mb_mutant`), a
full copy of `modelbench/`, `tests/`, `packs/`, `results/`, `pyproject.toml` — the tracked working
tree in `/home/mauricio/prg/graphmind-ai-lab/model-bench` was never modified (`diff` against a
pre-mutation backup confirmed identical afterward).

```
$ PYTHONPATH="$MUT" .venv/bin/python -m pytest -q tests/test_report.py   # design_effect: max→min
174 passed in 1.57s

$ PYTHONPATH="$MUT" .venv/bin/python -m pytest -q tests/test_report.py   # basis: min→max (on top)
174 passed in 1.49s
```

Both mutations — either alone reverses the conservative direction of the interval this function
prints — leave every test in `tests/test_report.py` green. This is the confirmation behind Major
finding M1 above.
