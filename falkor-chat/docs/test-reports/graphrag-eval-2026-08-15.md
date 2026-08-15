# GraphRAG retrieval + generation evaluation report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-026 (M2.5-quality)

Generated 2026-08-15T21:00:23.819194+00:00 by `server/tests/eval/generate_report.py` (K-026, `docs/plans/graphrag-eval.md` §5 Unit 3). Machine-generated — re-run the script to refresh rather than hand-editing this file.

## Retrieval baseline (Unit 2b)

- **recall@10:** 97.4%
- **recall@5:** 89.5%
- **MRR:** 0.6259
- **n (golden pairs measured):** 38

These numbers are the vector-only @1024 baseline over `Services.hybrid_search` against `ws:eval` (`docs/plans/graphrag-eval.md` §3 D6). Per D6, the first committed `retrieval_baseline.json` requires an explicit `data-scientist` methodology sign-off ("is this a reasonable floor") before it is treated as gating for future runs — the test having passed is not, by itself, sufficient.

## Judge layer (Unit 3)

- **judge model:** `qwen/qwen3-4b-2507`
- **agent-under-test model:** `qwen/qwen3-4b-2507`
- **run at:** 2026-08-15T20:59:21.421188+00:00

### Calibration sub-pass

- **sample size:** 10 fixed (`golden_judge_calibration.jsonl`) triples
- **faithfulness agreement:** 90.0%
- **relevance agreement:** 70.0%
- **parse failures:** 0 / 10 (a parse failure resolves conservatively — faithfulness=None, relevance=False — and is counted here, never silently dropped from the denominator)

**Small-N caveat (D4).** Judge-vs-human agreement is raw percent exact-agreement over a ~10-example calibration set, not Cohen's kappa — `m3-guard-calibration.md` (this codebase's prior same-model-judge precedent) found kappa badly behaved even at N~21-26, and a ~10-example set only makes that worse. At N~10, a single disagreement swings the reported number by 10 points: read it as a directional signal, not a statistically defensible claim against the method note's ~0.7 threshold.

### Generation sub-pass

- **sample size:** 20 live-generated items (sampled from `golden_retrieval.jsonl`)
- **faithful:** 20 true / 0 false / 0 abstained (no retrieved context)
- **relevant:** 20 true / 0 false
- **parse failures:** 0 / 20

**Same-model judge limitation.** The judge model (`qwen/qwen3-4b-2507`) is identical to the agent-under-test model for this run.
- *Calibration numbers* (judge-vs-human agreement on fixed, independently-authored triples) are largely unaffected by self-preference bias — the judge did not generate the content it's scoring. Read subject to the existing small-N caveat (D4), not this one.
- *Generation-sub-pass faithfulness/relevance numbers* (the judge scoring its own model's live output) are structurally exposed to self-preference bias. A high score here is a same-model directional signal, **not independent validation**, and must not be read as if a distinct judge produced it. **A passing calibration number does not license trusting these — they are two different validity claims.**
- Gross/obvious failures (flat contradiction of the retrieved context, answering the wrong question) likely remain catchable even here — the bias risk concentrates in borderline/subjective calls, where it becomes indistinguishable from genuine quality.
- `data-scientist` sign-off status on these numbers: **[pending / not yet reviewed]**.


## Corpus & golden set

- **corpus:** 121 messages across 12 threads (`ws:eval`)
- **embedding model:** `lmstudio/text-embedding-qwen3-embedding-0.6b` @ dim 1024
- **seeded at:** 2026-08-15T16:30:18Z
- **golden-retrieval set size:** 38 pairs (`golden_retrieval.jsonl`)
- **self-retrieval-inflation guard:** **PASS** — no golden query is a verbatim substring of (or superstring containing) its own target message text
