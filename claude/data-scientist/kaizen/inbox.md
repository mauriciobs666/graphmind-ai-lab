# Kaizen — Learnings Inbox: data-scientist

> **FROZEN — 2026-08-20.** This file is a historical snapshot only. Its 4 entries (as of this
> date) were imported into the `kaizen_data-scientist` FalkorDB graph
> (`claude/cobb/kaizen/history.md`, 2026-08-20 entry); `data-scientist` no longer appends here.
> New raw learnings are written directly into the graph and are immediately queryable by any
> agent: `mcp__cypher__query(graph='kaizen_data-scientist', cypher='MATCH (e:KaizenEntry) RETURN
> e.date, e.fact, e.evidence, e.context, e.suggestedHome, e.author ORDER BY e.date')`. Content
> below is preserved for historical reference and will not change.

> Append-only capture of durable, non-obvious environment facts the `data-scientist` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-08-15 — When judge and agent-under-test collapse onto the same model, the self-preference-bias caveat must be split per sub-pass, not applied blanket

- **Evidence:** `falkor-chat/docs/plans/graphrag-eval.md` v4 D1 collapses Unit 3's judge onto the
  same `qwen/qwen3-4b-2507` instance as the agent-under-test (stakeholder-directed hardware
  constraint) and writes one blanket self-preference-bias caveat covering "any faithfulness/
  relevance numbers Unit 3 reports." On inspection this conflates two sub-passes with different
  exposure: the *calibration* sub-pass judges fixed, independently-authored triples (not
  self-generated — self-preference bias barely applies, it's still a legitimate rubric-following
  signal), while the *generation* sub-pass judges the model's own live output (classic
  self-preference exposure — the judge may favor its own phrasing/reasoning). A reader who sees a
  passing calibration number and extends that trust to the generation numbers is making exactly the
  error the caveat exists to prevent, and a single undifferentiated caveat doesn't stop that read.
  This codebase already had one prior instance of the identical structural issue, independently
  named: `falkor-chat/docs/archive/plans/m3-guard-calibration.md` risk #4, "self-preference
  (inherited, DS risk #3)... unmeasurable with this set alone" for the intake/research guard judge.
- **Context:** methodology sign-off on K-026's Unit 3 judge-layer plan
  (`falkor-chat/docs/reviews/graphrag-eval-ml.md`), where the stakeholder deliberately traded away
  the "never the model-under-test judging itself" guidance for a real hardware constraint.
- **Suggested home:** prompt (data-scientist's LLM-as-judge validity section) — the general rule
  "when judge collapses onto agent-under-test, the caveat must distinguish content the judge
  generated itself from content it didn't; only the former carries self-preference risk" is reusable
  well beyond this one K-026 baseline and beyond falkor-chat.

## 2026-08-16 — A zero-tolerance "current >= baseline" regression gate on a golden-set metric at n≈38 is stricter than the sample supports, and the arithmetic to show it is cheap

- **Evidence:** `falkor-chat/server/tests/eval/test_retrieval_eval.py`
  (`test_retrieval_metrics_meet_or_beat_baseline`) hard-fails on any `recall_at_10` drop below the
  committed baseline (0.9737, n=38), no slack, while allowing MRR a 5%-relative floor. At n=38 a
  single golden pair flipping hit→miss moves recall@10 by exactly 1/38 ≈ 2.6 points — about a fifth
  of the metric's own Wilson 95% CI width (≈13 points) at that n, and the baseline's sum-of-scores
  (0.97368... × 38 = 37.0 exactly) shows it is literally one pair-flip from both the ceiling (1.0)
  and the gate's failure floor. A metric already near ceiling with zero slack below it will fail on
  ANN tie-breaking noise or index-rebuild nondeterminism as readily as on a genuine regression — and
  it also cannot register a genuinely better change (no room to move up). Cheap diagnostic: compute
  `1/n` as the "one-pair delta" and compare it against a naive Wilson/Wald CI at the reported n
  *before* signing off on any hard pass/fail gate built on a golden set this size — if the one-pair
  delta is a large fraction of the CI width, the gate needs either a tolerance band (mirroring
  whatever slack the paired metric already gets) or an explicit "route first failure to manual
  triage" policy instead of an unconditional hard fail. Also worth checking whether the golden
  pairs are drawn independently or clustered (e.g., k pairs per source thread/document) — clustering
  means the true variance is understated by treating n as the pair count, not the corrected count.
- **Context:** M-4 baseline sign-off on K-026's `retrieval_baseline.json`
  (`falkor-chat/docs/reviews/graphrag-eval-ml.md`, "Baseline sign-off" section).
- **Suggested home:** prompt (data-scientist's evaluation-engineering / golden-set section) — the
  "compute the one-unit delta and compare it to the CI width before blessing a hard gate" check is a
  reusable pre-sign-off habit for any small-golden-set regression gate, not specific to retrieval or
  to this repo.

## 2026-08-17 — A live-run report's "provenance" fields (model/quant/temp/baseURL) can silently diverge from the repo's config on a per-box basis, and the mismatch is checkable read-only before the run

- **Evidence:** `falkor-chat`'s model resolution is two hand-edited files, and the provider file
  (`FALKORCHAT_OPENCODE_CONFIG`, defaulting to `$HOME/.config/opencode/opencode.json`) is a
  **cross-project, machine-local** file outside the repo, not something `git blame`/`grep` can
  verify. On the box this session ran on, that default file declared `lmstudio` at a LAN IP
  (unreachable from the sandbox) listing only an unrelated model, while `config/models.json`'s
  `defaults.guard` named `lmstudio/qwen/qwen3-4b-2507` — a model the default provider file never
  mentions. `ProviderCatalog`/`_resolve_element` (`modelconfig.py`) validates only the **provider
  id**, not the model id, against the file, so this kind of mismatch resolves silently (wrong/
  unreachable `baseURL`) rather than failing loudly at construction. Separately, `localhost:1234`
  *was* live and did serve the expected model — findable with two read-only curls:
  `curl :1234/v1/models` (presence) and LM Studio's own `curl :1234/api/v0/models` (adds
  `quantization` and `state: loaded|not-loaded` per model — the exact fields a calibration/eval
  report's provenance header needs and the repo cannot supply from static config alone).
- **Context:** `falkor-chat/docs/plans/guard-judge-calibration-ml.md` (K-027 item 3) — grounding
  the archived protocol's §8 provenance requirements (model id/quant/temperature) against the
  actually-shipped `config/models.json`, which sets **no** `temperature` key anywhere in the repo
  for any kind — a genuine, not just unrecorded, uncontrolled variable for any determinism-sensitive
  eval design that assumes temperature is pinned near 0.
- **Suggested home:** prompt (data-scientist's evaluation-engineering section, provenance/
  reproducibility sub-point) — two reusable habits: (1) before trusting a report's provenance
  header on any project using a machine-local provider config, live-check the actually-reachable
  endpoint (LM Studio's `/api/v0/models` for quant+state, or the provider's equivalent) rather than
  reading only the repo's static config, since the two can diverge per-box with no loud failure;
  (2) grep the whole repo for `temperature` (or the sampling-param equivalent) before writing any
  non-determinism-handling section (k replicates, flip-rate, etc.) that assumes a pinned value —
  an unset sampling parameter is a silent gap in exactly the kind of report that most needs it
  pinned and recorded.

## 2026-08-17 — A conjunctive "fails as a bloc" probe-set gate can pass while most of its individual probes fail, and the summary line hides that unless it's checked case-by-case

- **Evidence:** `docs/archive/plans/m3-guard-calibration.md` §7 gates the guard-judge's
  materiality-probe set (`ca-04`/`ca-05`/`ca-08` vs. `cs-04`) with an explicit AND across all three
  advance-probe cases plus the adversarial suspend-probe. In the 2026-08-17 live run
  (`docs/test-reports/guard-judge-calibration-2026-08-17.md`), 2 of the 3 individual probes
  (`ca-04`, `ca-08`) failed — with rationales that echo the fixture's own `missing` field content
  almost verbatim, i.e. exactly the pattern-matching failure mode the probes exist to catch — while
  the third (`ca-05`) and the adversarial case (`cs-04`) both passed, so the bloc AND never
  triggered and the report correctly wrote "Passed." A reader trusting the one-line summary would
  see zero signal where there is in fact a 2-of-3 hit rate on a purpose-built diagnostic. The
  bloc-AND design (my own, from the archived protocol) is defensible as a *gate* — a single miss
  among plausible near-misses shouldn't block a wire decision — but it is the wrong granularity for
  a *report summary line*, which should default to reporting the per-probe hit rate and only fold
  it into a bloc/no-bloc verdict as a second sentence.
- **Context:** methodology sign-off on the K-027 item 3 calibration report
  (`docs/reviews/guard-judge-calibration-ml.md`).
- **Suggested home:** prompt (data-scientist's evaluation-engineering section) — the general rule:
  "when a probe *set* is gated with AND/OR logic for pass/fail purposes, still report the
  per-probe outcome individually in prose, not just the boolean bloc result — the aggregate can
  mask a real partial pattern at exactly the small-N scale where the qualitative read matters most."
  Applies to any multi-case diagnostic probe (materiality probes here; likely recurs in any
  LLM-as-judge calibration that uses small hand-built adversarial/materiality probe clusters).
