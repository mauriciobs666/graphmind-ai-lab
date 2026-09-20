# Small-Model Catalog Sweep — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — (M9)

## Summary

Acceptance-level verification of the five per-pack comparison reports (FR-6 through FR-12,
`docs/requirements/small-model-catalog-sweep.md`) rendered from the already-completed,
independently-verified 68-run live sweep, session `catalog-sweep-2026-09-19` (72 stored files
confirmed under that session tag: `grep -l catalog-sweep-2026-09-19 results/runs/*.json | wc -l`
→ `72` — 68 real model runs + 4 auto-generated `bm25` reference-arm records the embedder pack
always stores). No separate test-plan document was written for this pass — the requirements doc's
own FR/AC section serves as the plan, per the task brief; this report follows that section's
structure directly.

**Verdict: four of five reports pass FR-6–FR-12/AC-3/AC-4 cleanly. One report
(`nlq-structured-query`) has a real FR-6 gap**: two of the 16 in-scope chat/vlm models that have a
genuine stored run for this pack under this session are completely absent from the rendered ranked
table, with no banner or note anywhere in the report explaining why — traced to a root cause in
`modelbench/report.py`, not a data or CLI-usage problem. Filed as Defect 1 below. A handful of
lower-severity observations are filed alongside it.

**CPG:** considered, not relevant — this task verifies rendered report *output* against the
requirements doc's FR/AC criteria over already-executed live data (a black-box acceptance check),
not a code-level test-gap analysis of `model-bench`'s own source; no CPG query added value here.

**Commands run** (from `model-bench/` as working directory, per `README.md`'s Quick start):

```
./run.sh rank --pack embedder-graphrag-retrieval --session catalog-sweep-2026-09-19 --footprints footprints.json
./run.sh rank --pack guard-judge-understanding --session catalog-sweep-2026-09-19 --reference qwen/qwen3-4b-2507 --footprints footprints.json
./run.sh rank --pack nlq-structured-query --session catalog-sweep-2026-09-19 --reference qwen/qwen3-4b-2507 --footprints footprints.json
./run.sh rank --pack tool-caller-shop-assistant --session catalog-sweep-2026-09-19 --reference qwen/qwen3-4b-2507 --footprints footprints.json
./run.sh rank --pack chat-responder-grounded-answers --session catalog-sweep-2026-09-19 --reference qwen/qwen3-4b-2507 --footprints footprints.json
```

`--reference qwen/qwen3-4b-2507` was applied only to the four chat-role packs, per FR-8's own
wording ("each of the other in-scope **chat/vlm models**... compared against one designated
reference model... already tested against all four **chat-role packs**"). The embedder pack was
judged not applicable: it has an entirely disjoint model roster (4 embedding models), and
`qwen/qwen3-4b-2507` — a chat/vlm model — has no stored run under this pack at all, so it cannot
serve as a reference there. Confirmed correct: passing `--reference` on the embedder pack was not
attempted, and no reviewer disputed the omission.

## Five report paths (AC-3)

- `model-bench/reports/embedder-graphrag-retrieval-rank-20260920-01.md`
- `model-bench/reports/guard-judge-understanding-rank-20260920-01.md`
- `model-bench/reports/nlq-structured-query-rank-20260920-01.md`
- `model-bench/reports/tool-caller-shop-assistant-rank-20260920-01.md`
- `model-bench/reports/chat-responder-grounded-answers-rank-20260920-01.md`

All five rendered under `reports/` with the CLI's own `-rank-YYYYMMDD-NN` sequencing (AC-3: pass
for all five).

## Task 1 — `footprints.json`

Path: `model-bench/footprints.json`. **Method:** bits-per-weight × parameter-count-from-catalog-id,
calibrated against the requirements doc's three worked examples: `prism-ml/bonsai-27b` (Q1_0,
~3.5–5.5 GB, used verbatim), `text-embedding-granite-embedding-278m-multilingual` (Q8_0, trivially
small), `text-embedding-qwen3-embedding-4b` (Q4_K_M, ~2.2–2.5 GB). From the last example: Q4_K_M ≈
4.5 bits/weight (0.5625 bytes/weight) — 4e9 × 0.5625 = 2.25 GB, midpoint of the doc's own 2.2–2.5 GB
range. Applied the same rate to every other Q4_K_M model. Q8_0 ≈ 1.0 byte/weight (nominal 8-bit),
applied to every Q8_0 model. Q5_K_S ≈ 5.5 bits/weight (0.6875 bytes/weight, standard GGUF k-quant
figure) for the one Q5_K_S model. Quantization + catalog id for all 20 models came from a live
`GET http://localhost:1234/api/v0/models` call (verified reachable and current).

**All 20 in-scope models are covered — no open questions left**, though one genuinely needed
verification rather than guessing: `google/gemma-4-e2b`'s catalog id doesn't parse via the
established "digit-before-b" heuristic (`E2B` denotes Google's "effective parameters" naming, not
a raw parameter count — Gemma-family edge models report a much larger *total* on-disk parameter
count than their "effective" compute number). Per the task's explicit instruction not to guess,
I ran a `WebSearch` (AWS Bedrock model card, HuggingFace, LM Studio model page all agree): **5.1B
total parameters / 2.3B effective (PLE) parameters.** On-disk footprint follows *total* params
(every weight is stored regardless of activation), so I used 5.1e9 × 0.5625 bytes/weight ≈ 2.87 GB
→ `"~2.7-3.0 GB (Q4_K_M)"`. This is a verified figure from named sources, not a guess.

```json
{
  "text-embedding-qwen3-embedding-4b": "~2.2-2.5 GB (Q4_K_M)",
  "text-embedding-granite-embedding-278m-multilingual": "~0.28 GB (Q8_0)",
  "text-embedding-qwen3-embedding-0.6b": "~0.6 GB (Q8_0)",
  "text-embedding-nomic-embed-text-v1.5": "~0.08 GB (Q4_K_M)",
  "qwen/qwen3-4b-2507": "~2.2-2.5 GB (Q4_K_M)",
  "qwen/qwen3-4b-thinking-2507": "~2.2-2.5 GB (Q4_K_M)",
  "smollm3-3b": "~3.0 GB (Q8_0)",
  "stablelm-zephyr-3b": "~3.0 GB (Q8_0)",
  "qwen2.5-3b-instruct": "~3.0 GB (Q8_0)",
  "qwen2.5-coder-3b-instruct": "~3.0 GB (Q8_0)",
  "llama-3.2-3b-instruct": "~3.2 GB (Q8_0)",
  "stable-code-instruct-3b": "~3.0 GB (Q8_0)",
  "mistralai_ministral-3-3b-instruct-2512": "~3.0 GB (Q8_0)",
  "mistralai/ministral-3-3b": "~3.0 GB (Q8_0)",
  "gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf": "~2.2-2.5 GB (Q4_K_M)",
  "google/gemma-3-4b": "~2.2-2.5 GB (Q4_K_M)",
  "google/gemma-4-e2b": "~2.7-3.0 GB (Q4_K_M)",
  "qwen3.5-2b-claude-4.6-opus-reasoning-distilled": "~1.3-1.5 GB (Q5_K_S)",
  "nvidia/nemotron-3-nano-4b": "~2.2-2.5 GB (Q4_K_M)",
  "prism-ml/bonsai-27b": "~3.5-5.5 GB (Q1_0)"
}
```

Verified in the rendered reports: every in-scope model's footprint cell renders a value (no `—`
gaps for any of the 20). The only `—` footprint cells observed are on the embedder report's `bm25`
row (Defect/Observation 3 below) — expected, since `bm25` is not in `footprints.json` and is not
one of the 20 in-scope models.

## Task 3 — line-by-line verification

Per the brief, `embedder-graphrag-retrieval` and `guard-judge-understanding` were checked in full
line-by-line detail; `nlq-structured-query`, `tool-caller-shop-assistant`, and
`chat-responder-grounded-answers` were spot-checked — the `nlq` spot-check surfaced Defect 1, so it
received the same depth as the two "detailed" packs once the anomaly appeared.

### `embedder-graphrag-retrieval` (detailed check)

| Item | Result | Evidence |
|---|---|---|
| FR-6 | **Pass** | All 4 in-scope embedding models present (`text-embedding-qwen3-embedding-4b`, `-nomic-embed-text-v1.5`, `-granite-embedding-278m-multilingual`, `-qwen3-embedding-0.6b`). A 5th row, `bm25`, also appears — not an in-scope model (see Observation 3). |
| FR-7 | **Pass** | Ranked table (not a matrix), each row carries its own 95% CI, resolving-power sentence reads "n=38 effective querys" (typo noted, Observation 4) recomputed for the 38-query design. |
| FR-8 | **N/A, correctly** | No `--reference` passed; judged not applicable to this pack (see Summary). No pairwise verdict of any kind appears in the report — confirmed. |
| FR-11 | **Pass** | `> recall@10 = 37/38 at this pack's own item set: only 1 item is available to win, and McNemar needs 6 — this ranking can detect a materially worse embedder but cannot certify a better one (-ml §7.4).` — restates `-ml` §7.4's finding verbatim in substance, not a citation. |
| FR-12 | **Pass** | `latency p95` and `footprint` columns present on every row. |
| AC-3 | **Pass** | Rendered to `reports/embedder-graphrag-retrieval-rank-20260920-01.md`. |
| AC-4 | **N/A** | AC-4 is scoped to "a chat-role pack's report" — embedder is not a chat-role pack, so this AC does not bind it. Confirmed by re-reading AC-4's own wording rather than assuming. |

### `guard-judge-understanding` (detailed check)

| Item | Result | Evidence |
|---|---|---|
| FR-6 | **Pass** | Both `falseAdvanceRate` (n=40) and `falseSuspendRate` (n=30) tables list all 16 in-scope chat/vlm models. |
| FR-7 | **Pass** | Two ranked tables (guard-judge has no headline metric, per FR-7's "or each `verdictMetrics` member" clause) — not a matrix. Per-model Wilson CIs present. Resolving-power sentences: "n=40 effective items"/"n=30 effective items", matching `-ml` §7.3's own per-slice `n`. |
| FR-8 | **Pass** | One combined reference-anchored family of **30 tests** (15 candidates × 2 verdict metrics), Holm-Bonferroni corrected *jointly* — confirmed by cross-checking the printed per-row thresholds: the same candidate (e.g. `nvidia/nemotron-3-nano-4b`) carries different Holm ranks (0.0100 in the `falseAdvanceRate` table, 0.0500 in `falseSuspendRate`) consistent with one shared 30-slot Holm ladder, not two independent 15-slot ones. No pair outside the `qwen/qwen3-4b-2507` anchor receives a verdict. |
| FR-11 | **Pass** | `> Two co-equal class-conditional error rates, no single headline: floor 15.0/20.0 pp, MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate at the two-member alpha_mdd=0.025 (-ml §7.3).` — matches `-ml` §7.3's own table numbers exactly (15.0/21.9 for `clear_suspend`, 20.0/28.7 for `clear_advance`). |
| FR-12 | **Pass** | Both tables carry `latency p95` and `footprint`. |
| AC-3 | **Pass** | Rendered to `reports/guard-judge-understanding-rank-20260920-01.md`. |
| AC-4 | **Pass** | `_exploratory — no significance claim outside this family_` caption present before each reference-anchored-family table; FR-11's caveat is restated (see above). |

### `nlq-structured-query` (spot-check → escalated to full check)

| Item | Result | Evidence |
|---|---|---|
| FR-6 | **FAIL — Defect 1** | Ranked table shows only **14 of the 16** in-scope chat/vlm models. `qwen/qwen3-4b-thinking-2507` and `stable-code-instruct-3b` both have a genuine stored run for this pack under `catalog-sweep-2026-09-19` (`grep -l catalog-sweep-2026-09-19 results/runs/nlq-structured-query-*.json` lists all 16 modelKeys, including both) but are completely absent from the rendered table, with no banner, footnote, or any other mention anywhere in the report. See Defect 1 below for root cause. |
| FR-7 | **Pass** (for the 14 shown) | Ranked table, per-model Wilson CI, resolving-power sentence "n=34 effective items" matches `-ml`'s corrected true denominator. |
| FR-8 | **Pass, with one artifact of Defect 1** | Holm-Bonferroni family of 15 tests vs. `qwen/qwen3-4b-2507` correctly computed; the two affected models still appear in the *family* table as `no verdict — no paired data` rows (so a careful reader piecing the two tables together could infer something is wrong) — but this is not a substitute for FR-6 compliance in the main ranked table. |
| FR-11 | **Pass** | `> The true denominator is 34, not 40 — 6 items are structurally unanswerable and excluded (-ml §7.2, v1.25 note).` — restates `-ml`'s n=34 finding verbatim in substance. |
| FR-12 | **Pass** (for the 14 shown) | `latency p95` and `footprint` present on every visible row. |
| AC-3 | **Pass** | Rendered to `reports/nlq-structured-query-rank-20260920-01.md`. |
| AC-4 | **Pass, but incomplete given Defect 1** | Exploratory labeling and caveat restatement both present for the 14 models that are shown; the two missing models are outside the scope of "every number" only because they were never rendered as numbers at all. |

### `tool-caller-shop-assistant` (spot-check)

| Item | Result | Evidence |
|---|---|---|
| FR-6 | **Pass** | All 16 in-scope models appear in the `cleanThroughTurnH` ranked table. |
| FR-7 | **Pass** | Ranked table, per-model Wilson CI, resolving-power sentence "n=12 effective conversations" matches `-ml` §7.1/§7.2's tool-caller design exactly (floor 50.0 pp, MDD 57.8 pp, both cited verbatim in the report). |
| FR-8 | **Pass** | One reference-anchored family, 15 tests vs. `qwen/qwen3-4b-2507`, Holm-Bonferroni. |
| FR-11 | **Pass** | `> The analysis unit is scripts, n=12 (3 shapes x 4 scripts): floor 50.0 pp, MDD80 57.8 pp at this n (-ml §7.2/§4.5).` |
| FR-12 | **Pass** | Present on every row. |
| AC-3 | **Pass** | Rendered to `reports/tool-caller-shop-assistant-rank-20260920-01.md`. |
| AC-4 | **Pass** | Exploratory caption present; caveat restated. |
| — | **Observation 2** | `llama-3.2-3b-instruct`'s row shows `0/3` rather than `0/12` for `cleanThroughTurnH` — confirmed in the raw run record (`cleanThroughTurn.n = 3`) that only 3 of this model's 12 conversations produced a scoreable "clean through turn" outcome for this model specifically. Unlike Defect 1, this is **not silently dropped** — the true, smaller denominator is shown plainly in the table (`0/3`), which is the honest behavior FR-6/FR-7 call for. Flagging only as a residual-risk note: a reader skimming point estimates without reading the `k/n` column could miss that this row rests on 1/4 the effective data of its neighbors. |

### `chat-responder-grounded-answers` (spot-check)

| Item | Result | Evidence |
|---|---|---|
| FR-6 | **Pass** | All 16 in-scope models appear in the `groundingRate` table. |
| FR-7 | **Pass** | Ranked table, per-model Wilson CI, resolving-power sentence "n=30 effective items" matches `-ml`'s chat-responder row (floor 20.0 pp, MDD 25.1 pp, both cited verbatim). |
| FR-8 | **Pass** | One reference-anchored family, 15 tests vs. `qwen/qwen3-4b-2507`, Holm-Bonferroni. |
| FR-11 | **Pass** | `> **Reply quality is not measured by this pack.** groundingRate is a deterministic containment check against the retrieved context, never a judgement of how good, helpful, or well-written a reply is (FR-21a — the judged-quality layer is deferred, docs/BACKLOG.md).` This restates the pack's *actual, currently-true* scope limitation (the shipped deterministic-only layer per `README.md`'s Status section), rather than `-ml` §6.2's now-stale "golden data does not exist" note written before the 30-item golden set was built and human-verified — judged the correct choice, since FR-11 asks for the caveat that is true of the report being read, not a superseded planning note. |
| FR-12 | **Pass** | Present on every row. |
| AC-3 | **Pass** | Rendered to `reports/chat-responder-grounded-answers-rank-20260920-01.md`. |
| AC-4 | **Pass** | Exploratory caption present; caveat restated. |

## Defects

### Defect 1 — `nlq-structured-query` ranked table silently omits two in-scope models with 100%-parse-failure stored runs (FR-6 violation)

**Severity: High** (a stakeholder reading this report to pick an NLQ model would never learn that
`qwen/qwen3-4b-thinking-2507` and `stable-code-instruct-3b` were even attempted, let alone that
they failed completely — the opposite of FR-5's "visible, not silently dropped" principle, applied
here to the report layer rather than the sweep layer it was written for).

**Steps to reproduce:**
```
cd model-bench
grep -l catalog-sweep-2026-09-19 results/runs/nlq-structured-query-*.json | wc -l   # → 16 (all 16 chat/vlm models have a stored run)
./run.sh rank --pack nlq-structured-query --session catalog-sweep-2026-09-19 \
    --reference qwen/qwen3-4b-2507 --footprints footprints.json
```

**Expected:** the ranked table lists 16 rows (FR-6: "covering every in-scope model that has a
stored result for that pack under the sweep's session identifier"), or, if a model's data cannot
be ranked, the report names the exclusion the way it already does for a pack-version/hash/schema
mismatch or an aggregate/items mismatch (the `> **INVALID RESULTS EXCLUDED**` banner already
implemented in `modelbench/report.py`'s `rank_report`).

**Actual:** the table lists 14 rows. `qwen/qwen3-4b-thinking-2507` and `stable-code-instruct-3b`
are absent, with zero mention anywhere in the rendered markdown (no banner, no footnote). Both
models surface only as `no verdict — no paired data` rows deep in the reference-anchored-family
table — easy to miss, and not a substitute for the main ranked table's own coverage claim.

**Root cause (traced, not guessed):** both models' stored records
(`results/runs/nlq-structured-query-qwen_qwen3-4b-thinking-2507-2026-09-19T23:45:36Z.json`,
`results/runs/nlq-structured-query-stable-code-instruct-3b-2026-09-20T01:18:34Z.json`) show
`"parseFailures": 40` out of 40 items — every item's `scoreable` is `{}` and `outcome` is
`"parse_failure"`. The pack's `layer1ExactMatchRate` aggregate is therefore genuinely and
consistently `{n: 0, successes: 0}` — this is *not* an aggregate/items mismatch
(`_aggregate_item_mismatches` correctly finds no disagreement, since the aggregate honestly
reports `n=0`), so it never reaches the existing "INVALID RESULTS EXCLUDED" path.

Instead, `modelbench/report.py`'s `BinaryMetric.rate` returns `None` when `n == 0`
(`results.py:110-112`, by design — dividing by zero), and `_metric_value`/`_rank_rows`
(`report.py:125-141`, `1070-1081`) then exclude the run from the ranked table under the docstring's
own stated rule: *"a model with no data for this pack is absent, not 'worst'"*. That rule is
correct for a model that genuinely never ran against this pack — but here it is applied to a model
that **did** run, produced 40 real (if unparseable) live replies, and consumed real wall-clock time
(both records carry a full latency block), and whose 100% failure rate is itself a decision-relevant
finding, not an absence of one. The code conflates "no data" with "data, but zero of it scored" —
two different situations that `FR-6`/`FR-5`'s honesty principle treats differently.

By contrast, `guard-judge-understanding`'s classification scorer has a defensive fallback (default
to "suspend" when the judge's verdict can't be parsed), so `prism-ml/bonsai-27b`'s 85/85
parse-failure run on that pack *still* carries `n=40`/`n=30` aggregates and renders correctly (with
a legitimately poor/degenerate score, visibly) — confirming this is specific to packs whose scorer
has no such fallback (`nlq-structured-query`'s extraction scorer), not a universal one every pack
shares.

**This is a report/CLI code gap, not a data problem — I have not fixed it** (routes to
`coder`/`tdd-engineer` via `teco`, per this agent's guardrails). A minimal fix shape: when
`rank_report` excludes a run from a metric's table because its aggregate has `n=0` while the run
itself is otherwise valid and consistent, name it the same way an aggregate/items mismatch is
already named, rather than dropping it silently.

### Observation 2 — `tool-caller-shop-assistant`: `llama-3.2-3b-instruct`'s denominator shrinks to 3/12 (visible, not a defect)

See table above. Filed as a coverage/residual-risk note, not a defect — the reduced `n` is shown
plainly in the `k/n` column, which is the honest behavior. Recommend a future enhancement (not
raised as a defect): a footnote or asterisk on any row whose per-metric `n` differs from the pack's
modal/expected `n` for that metric, so this is not left to a reader noticing the `k/n` column
unprompted.

### Observation 3 — `embedder-graphrag-retrieval`: an extra `bm25` row is not an in-scope model

The embedder pack always auto-stores a `bm25` deterministic reference-arm record alongside each
embedding-model run (confirmed expected/documented sweep behavior, not an error). `rank`'s own
`_select_rank_arms` is documented as "every distinct model with a stored run for this pack... never
a fixed arm count" — it does not, and by design cannot, distinguish "in-scope" from "incidentally
stored," so `bm25` appears as a 5th ranked row beside the 4 in-scope embedding models. FR-6 does not
forbid this (it specifies a floor — every in-scope model must appear — not a ceiling), and the row
is clearly labeled and carries `—` for latency/footprint, so it is not readily mistaken for one of
the 20 in-scope models. Not filed as a defect; noted so a future reader of this report isn't
surprised by the discrepancy between "4 embedding models" and "5 rows."

### Observation 4 — minor wording/rendering nits (cosmetic, not filed as defects)

- Embedder report: "at n=38 effective **querys**" — should read "queries." Pure typo in
  `modelbench/report.py`'s resolving-power sentence template, reproducible on every embedder
  `rank` invocation.
- Guard-judge report: the hypothetical-family sentence ("If this pack's optional reference-anchored
  family (FR-8) were run...") is printed **unconditionally**, even when a real reference family
  *was* run and is rendered immediately below it (confirmed: this sentence appears identically
  across all four chat-role reports regardless of whether `--reference` was passed). Reads oddly
  as "if... were run" directly above a table showing it having just run. Also carries a stray comma
  before the em dash in the guard-judge rendering ("...jointly across both verdict metrics, — that
  family..."). Neither affects correctness of the numbers; both are template-text quality issues
  worth a follow-up polish pass.

## Coverage & gaps

**Covered:** FR-6, FR-7, FR-8, FR-11, FR-12, AC-3, AC-4 checked against all five rendered reports,
two (`embedder-graphrag-retrieval`, `guard-judge-understanding`) at full line-by-line depth, three
spot-checked (one of which, `nlq-structured-query`, was escalated to full depth once the anomaly
surfaced). Footprint estimation method verified against the requirements doc's three worked
examples and one figure independently confirmed via web search rather than guessed.

**Not covered, deliberately (out of this task's stated scope):** FR-9/FR-10 and the consolidated
document (the brief scoped this pass to the five per-pack reports only); the sweep's own
zero-operational-failures claim and the 72-file/68-run count (already independently verified twice
per the task background, not re-verified here); the correctness of the underlying statistics
implementation itself (`stats.py`, already gated per Track A's own review,
`docs/reviews/small-model-catalog-sweep-impl.md`) — I checked that the *numbers rendered* are
internally consistent with `-ml`'s published formulas and worked examples, not that the formulas
are re-derived from first principles.

**Residual risk:** Defect 1's root cause (an `n=0`-vs-"no run" conflation in `rank_report`) is
scorer-shape-dependent — it manifests wherever a role's scorer has no parse-failure fallback and a
model produces 100% unparseable output. `nlq-structured-query`'s extraction scorer is the one
confirmed instance in this sweep's data; I did not exhaustively check every model/pack/metric
combination in the sweep for the same signature (only the two anomalies this investigation actually
surfaced), so a third silently-dropped case elsewhere in the 68 runs cannot be ruled out from this
pass alone without a full metric-by-metric denominator scan.

## Feedback & recommendations

1. **Fix Defect 1** in `modelbench/report.py`: an `n=0` metric aggregate on an otherwise-valid,
   consistent run should be named as excluded (mirroring the existing
   `> **INVALID RESULTS EXCLUDED**` banner mechanism), not silently filtered out of
   `_rank_rows`/`_metric_value`. Route to `coder`/`tdd-engineer`.
2. Consider a full metric-by-metric denominator scan across all 68 sweep records (comparing each
   model's actual `n` per verdict metric against the pack's expected/modal `n`) before treating any
   of these five reports as final for stakeholder consumption — this pass found the two clearest
   cases but was not exhaustive (see Residual risk above).
3. Fix the "effective **querys**" typo in the embedder resolving-power sentence template.
4. Consider gating the "if this pack's optional reference-anchored family... were run" hypothetical
   sentence on `reference is None`, or rewording it, so it doesn't read as counterfactual directly
   above a table showing that exact family having just run.
5. `--footprints` worked exactly as documented (verbatim string pass-through, `RankUsageError` on a
   malformed file) — no CLI defect found in that flag.

## Kill-resilience note

A scratch-file write for incremental findings tracking (as the task brief requested) was blocked by
this agent's own tool guardrail ("subagents return findings as text, not report files"). Kill
resilience instead came from the five rendered reports themselves (already durably written to
`reports/` as each `rank` command completed) and from `footprints.json` (written early, Task 1,
before any report generation began) — a kill at any point after Task 1 would have lost only
in-progress verification notes held in this agent's own context, not the underlying artifacts.
