# `rank --reference` on a continuous verdict metric — plan review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** —

## 1. Scope & verdict

Reviewed: `model-bench/docs/plans/rank-continuous-reference.md` (`architect`), against its
prerequisite method note `model-bench/docs/plans/rank-continuous-reference-ml.md`
(`data-scientist`) and the real current tree (`modelbench/report.py`, `modelbench/stats.py`,
`tests/test_report.py`, `tests/test_stats.py`, `tests/conftest.py`). Not reviewed: the
implementation (none exists yet — this is the design-gate pass, U2 in the coordination ledger).

Every line-number and test-name citation I spot-checked in both documents (a large majority of
them, including all the ones the brief called out by name) matched the real tree exactly — I read
`_metric_kind`, `_paired_rows`/`_paired_diffs`, `_render_reference_family`, `_rank_resolving_power_
lines`, `rank_report`'s reference-family construction block (`report.py:1445-1483`, byte-for-byte
as cited), `compare_report`'s mixed-kind/continuous branches (`report.py:1664-1755`), and
`stats.continuous_verdict`/`stats.verdict` (`stats.py:1216-1301`, `1651-1769`) directly, and hand-
traced the new `_resolve_reference_kinds` algorithm and step 7's message-selection chain against
every combination the brief asked about. I also reproduced D-1 live:

```
$ ./run.sh rank --pack embedder-graphrag-retrieval --reference bm25
model-bench: item 'gr-01' metric 'mrr' lives in `measures` (a continuous measurement) and has no
boolean outcome; call `scored_value` instead
$ echo $?
2
```

and confirmed the full existing suite is green today (`1778 passed, 3 deselected`), establishing a
clean baseline for the plan's regression claims.

**Verdict: approve with suggestions.** No blocker — the design is sound, its N-ary
disagreement-detection algorithm is correct against every fixture shape I traced by hand, its
message-selection chain is correct for every combination named in the brief, its polarity-reuse
claim holds (verified from first principles, not just trusted), and its byte-identity regression
claim for `guard-judge-understanding` holds under hand-tracing of the new dispatch. One major and
several minor findings below are worth fixing before or during implementation, none of which
changes the plan's overall approach.

**CPG:** considered, not relevant — the plan's own §1 records that `GRAPHS` lists no
`cpg_model-bench`; I independently re-confirmed this is still true. `model-bench` is a code-level
component with no loaded CPG, not a task with no code-level component — this is `considered, not
relevant`, not `not applicable`.

## 2. Findings

### Major

**M1 — The new continuous caption duplicates a formula `stats.py` already owns, against this
component's own stated invariant.** `_render_reference_family_continuous` (plan §4 step 5) computes
`alpha_used = pack.metrics.alpha_family / correction_k` itself, purely to print it in the table's
caption — a second, independent computation of the exact expression `continuous_verdict` already
computes internally (`stats.py:1721`, `alpha_used = alpha_family / k`) and carries on
`ContinuousVerdict.alpha_used`. `model-bench/AGENTS.md`'s load-bearing invariant is explicit:
"`stats.py` implements `docs/plans/small-model-benchmarking-ml.md` and **no other source** — every
formula, constant, threshold and verdict string is that note's" — and the plan's own §3.4 invokes
the identical "two copies of a formula is one copy and one bug" convention to justify extracting
`_aggregate_kind`. The plan is aware of this specific seam (§4 step 5's docstring, test 10's
description: "closing the 'two independent computations of the same trivial formula' seam") but
resolves it only with a drift-detection pin, not by removing the duplication. A future change to
either formula (e.g. widening `_family_ci_levels`'s definition) drifts the two apart silently until
test 10 happens to catch it — the class of risk this component's own convention exists to close by
construction, not by test.

*Suggested fix:* add a small, public one-line helper next to `_family_ci_levels` in `stats.py` —
e.g. `def alpha_used(alpha_family: float, k: int) -> float: return alpha_family / k` — and have
both `continuous_verdict` (replacing its local `alpha_used = alpha_family / k` at stats.py:1721)
and `_render_reference_family_continuous`'s caption call it. One formula, one home; the caption still
prints before any `cv` exists, so it must call this helper directly rather than reading `cv.
alpha_used` off a candidate's result (that ordering constraint stays, only the formula's *home*
moves).

### Minor

**m1 — §3.8's rejection of a per-metric tally under refusal reads more into the `-ml` note than the
note says.** §3.8 rejects mirroring `compare_report`'s existing mixed-kind refusal, which still
renders a per-candidate `_pairing_tally` line even while refusing the verdict (`report.py:1681-1706`,
confirmed by direct read) — citing `-ml` §3.3's "skip both the binary-ladder and continuous-family
code below entirely — no partial table for either kind" as justification for omitting *all*
diagnostic output. Read in `-ml`'s own context, that sentence is about skipping the verdict-table/
ladder-*construction* code, not the separate, purely-descriptive tally line `compare_report` renders
regardless of verdict outcome. This plan's refusal paths are unreached by any pack shipped today
(§2.3, honestly stated), so the practical cost is nil now — but it is a real, acknowledged
divergence from the precedent this plan otherwise claims to mirror throughout, resting on a
citation that does not obviously settle the question, for a case (an ordinary pre-registered mixed
family) that is far more plausible to ship than the disagreement case. *Suggested fix:* either add
the same `_pairing_tally` line under both refusal messages (matching `compare_report`'s output
shape exactly), or keep the current design but replace the citation with the plan's own reasoning
("no resolvable kind exists to build a tally by, so none is attempted") rather than resting on a
quote that reads differently in its own source.

**m2 — Step 7's continuous-branch `elif` doesn't literally test what its comment claims.**
```python
if not reference_family_refused and resolved_kinds == {"binary"}:
    ...
elif not reference_family_refused:  # resolved_kinds == {"continuous"}
    correction_k = len(family) * len(candidates)
```
For every real pack (non-empty `family`), `resolved_kinds` can only be `{"binary"}`,
`{"continuous"}`, or a 2+-element set (which sets `reference_family_refused = True` and skips this
`elif` entirely) — so the comment is accurate by elimination today. If `family` is ever empty
(a malformed pack, or a test fixture bug — `rank_report` does not itself enforce non-empty
`verdictMetrics`), `resolved_kinds` is `set()`, `reference_family_refused` stays `False`, and this
branch silently sets `correction_k = 0` while its own comment claims the continuous case. Harmless
today only because `members` is empty in the same scenario, so nothing downstream reads
`correction_k`. *Suggested fix:* write the condition explicitly — `elif resolved_kinds ==
{"continuous"}:` — so the branch is self-verifying rather than correct by an argument the reader has
to reconstruct.

**m3 — `_rank_resolving_power_lines`'s kind-gate (step 6) resolves kind more weakly than the
reference-family path does, for the more exposed of the two functions.** Step 6 gates on
`_aggregate_kind(first_agg)` — the kind of the *first* non-`None` aggregate found by iterating
`runs` in order — rather than the N-ary cross-arm-agreement check `_resolve_reference_kinds`
provides for the reference-family path. In the (today unreached, admittedly defensive-only) case of
a genuine cross-arm kind disagreement, this function could resolve "binary" from the first arm and
proceed to print a McNemar/Wilson-shaped resolving-power sentence for a pack that, per §3.5's own
framing, has "a data-integrity problem, not a foreseen pre-registration choice." This function runs
unconditionally for every pack with `len(runs) >= 2` — independent of `--reference` — so it is
arguably *more* exposed than the reference-family path this plan builds the stronger check for. Not
a blocker (the disagreement scenario is unreached by any pack today, same status the plan grants
the other defensive paths), but the asymmetry is worth a line in §6's risks rather than silence.

**m4 — Citation nit: the fixture-helper paragraph in §5 misattributes three helpers' home.**
"`run(..., role="embedder", call_surface="embeddings", ...)`, `embeddings_fields`/
`deterministic_fields` (all in `tests/test_report.py`...)" — these three are defined in
`tests/conftest.py` and imported into `test_report.py` via `from conftest import (...)`
(`tests/test_report.py:12`). Functionally inert for an implementer (the names resolve either way
inside `test_report.py`), but the brief specifically asked me to flag citations that don't hold up,
and this one doesn't.

## 3. What's solid

- **D-1's diagnosis is exactly right.** I reproduced the crash live at the exact site the plan
  names (`_paired_rows` inside `rank_report`'s reference-family construction block) and confirmed
  `_aggregate_item_mismatches` — which runs earlier in the same function — is already correctly
  kind-aware (`report.py:466-468`), so the plan correctly scoped the fix to the one unconditional
  loop that isn't.
- **`_resolve_reference_kinds` (§3.4) is correct.** I hand-traced it against all four fixture shapes
  §5 tests 5-8 describe (universal agreement, no arm declares, two candidates disagree with each
  other, a candidate disagrees with a declaring reference) and it resolves each correctly, including
  the specific gap it was built to close in the `-ml` note's own pairwise-anchored-at-reference
  sketch.
- **Step 7's three-way message-selection chain (§4 step 7, second block) is correct** for every
  combination the brief named: a metric that itself disagrees, a metric with a disagreeing sibling,
  an ordinary pre-registered mixed family, and the combined three-metric case (test 16) — traced by
  hand against the actual `if`/`elif`/`else` order.
- **The polarity-reuse claim holds**, verified independently from first principles rather than
  trusted: for a higher-is-better metric (the only kind any shipped continuous metric is today),
  negating the reference-minus-candidate raw diff produces "positive = candidate is better"
  identically whether the underlying accessor is boolean (`_paired_rows`) or continuous
  (`_paired_diffs`) — and the same mechanical argument extends correctly to a hypothetical future
  lower-is-better continuous metric too.
- **The byte-identity regression claim (test 19) holds** under hand-tracing of the new dispatch for
  `guard-judge-understanding`'s real, kind-consistent data: the new `_resolve_reference_kinds` call
  and disagreement bookkeeping are purely internal and produce no visible output difference when
  every arm agrees, and the binary-ladder construction block, once reached, is textually identical
  to today's code.
- **Grounding is unusually strong.** Nearly every line-number and test-name citation across both
  documents checked out exactly against the real tree — a genuine strength worth naming, since it's
  precisely what lets an implementer trust the plan without re-deriving it.

## 4. Open questions

None that block approval. All findings above are actionable by the plan's owner or its implementer
without further input from the requester.
