# Typed Decision Models — Requirements and Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1

**Scope.** `docs/requirements/typed-decision-models.md` (first version, `Interviewing`, readback
confirmed 2026-10-30), reviewed at requirements altitude as the gate before the
`Ready for design` flip. Grounding was checked against `falkor-chat/server/falkorchat/guards.py`,
`responder.py`, `fusion.py`, `falkor-chat/scripts/seed_workflows.sh` (the `triage@v1` def),
`falkor-chat/config/models.json`, `model-bench/modelbench/lmstudio.py`,
`model-bench/results/index.csv`, `model-bench/packs/guard-judge-understanding/` (items counted by
tier), `docs/requirements/small-model-benchmarking.md` (archived; called "SMB" below) and
`falkor-chat/docs/requirements/document-ingestion.md`. I did not check the Laya figures (~20 ms,
determinism, the informal error rates' raw data) or Laya's model-card claims; I had no run
artifacts to check them against.

**Verdict: needs changes.** One blocker: FR-1/FR-2/FR-3 are only defined for the guard judge, but
the doc puts four decision points in scope. Two majors: FR-7 says "under its existing honesty
rules", but SMB's rules contradict FR-5 and FR-3. The intent, the guard-judge half, and the
decision log are sound. The fixes are mostly a per-point table plus a few open questions, not a
redesign.

CPG: considered, not relevant — a requirements-altitude review. I checked the factual claims by
reading a handful of named files directly, so a call-graph query would have added nothing.

**Growth-stops check (root `AGENTS.md`).** This is the first version, so all of its text is new
scope. There is no rework growth and no restated history. Each decision-log entry is one question
→ one answer plus the FR/AC it affects. Inherited facts are cited, not restated (the ~15 pp
resolving power and the no-auto-apply rule both point at SMB). No prior findings or answers to
fold in. Clean.

### Findings

**B1 (blocker) — FR-1, FR-2 and FR-3 have no defined incumbent, bar or fallback at three of the
four in-scope decision points.** The FRs are written around "the incumbent LLM-only judge". Only
the guard judge has one:

- *Responder answer-or-not.* The scope table says "decided inside the responder". In fact it is a
  locked deterministic rule: reply iff the message @mentions the agent and is not agent-authored
  (`responder.py:7-10`, `_should_respond` at `:66-72`), and the second condition is the hard
  loop-guard. A rule that runs in microseconds can't be beaten 2× (FR-2), there is no LLM to hand
  off to (FR-3), and "parity" against a rule that is correct by construction is undefined. The doc
  never says what question the model would answer: "reply to a non-mentioned message?" or
  "decline a mentioned one?".
- *Entity-match borderline.* The "fuzzy match" in `fusion.py` only *generates* candidates. Any hit
  becomes `'suggested'` and lands in a pending queue that a human or a connected agent confirms
  (`fusion.py:8-11,112-121`; `document-ingestion.md` FR-9/FR-10: "not linked/merged
  automatically"). So the incumbent decider is that reviewer, not an index. Whether the model
  replaces the reviewer, which would amend FR-9, or pre-sorts the queue changes everything
  downstream.
- *Intent triage.* Per the doc, it "does not exist". So there is no incumbent for FR-1, nothing to
  be 2× faster than for FR-2, no LLM to hand off to for FR-3, and nothing consumes the label. For
  a pick-one question, FR-1's two error types (false positive and false negative) don't apply
  either; it needs a per-class or confusion-based statement.

AC-1 and AC-3 name only the guard judge. As written, the architect can't derive acceptance for
three quarters of the scope.

*Suggested improvement.* Add one row per decision point covering: the incumbent (or "none — new
capability"), the parity reference, the speed reference, the hand-off target (if any), and who
consumes the decision. Where you can't fill a row yet, list it as an open question. Re-scoping
isn't required, since the stakeholder chose "all four"; the bars just need to exist for each.

**M1 (major) — FR-5 contradicts SMB FR-19, which FR-7 inherits.** FR-7 requires measurement "under
its existing honesty rules". SMB FR-19 says "an LLM may draft labels but **a human verifies every
one**". FR-5 removes human review for evaluation sets, which is a stakeholder decision and
legitimate. But SMB is archived, so as written the doc puts model-bench under two rules that
contradict each other. *Suggested improvement:* state in FR-7 (or FR-5) that this effort amends
SMB FR-19 for LLM-labeled decision-point packs. Keep the rest of SMB's honesty rules: intervals,
fingerprint, pairing, provenance. Record which provenance fields stand in for "verified by", and
point to the amendment's single home.

**M2 (major) — FR-3's hand-off arm is a composite of two models; SMB FR-1 allows exactly one model
per run.** FR-3 judges parity "on the **combined** outcome" (decision model plus LLM fallback), and
FR-7 says parity is established in model-bench. SMB FR-1 says "a run evaluates **exactly one
model** against one named task pack". *Suggested improvement:* say explicitly that a hand-off
configuration (decision model + threshold + fallback model) counts as one measurable arm, and that
its fingerprint records all three. That is a WHAT-level amendment; how to build it is the
architect's call.

**M3 (major) — Nothing says the hand-off threshold and calibration can't be tuned on the
evaluation set.** FR-4/AC-3 forbid overlap between training and evaluation *examples*. But the
FR-3 confidence cut-off, and any recalibration (the model card warns of over-confidence), are
fitted parameters. With 85 guard items, choosing the cut-off on those same items lets the combined
FR-1 result overfit them. *Suggested improvement:* extend FR-4/AC-3 to "not trained, calibrated, or
threshold-tuned on its evaluation examples". `falkor-chat/server/tests/eval/golden_judge_calibration.jsonl`
(10 items, cited in SMB FR-21) is an existing candidate for a separate set; whether it is big
enough is a design question.

**M4 (major; a risk the stakeholder's own decision creates, flagged rather than contested) — FR-1
is an absence-of-evidence test, so a small or noisy evaluation set makes it easier to pass.** "Not
measurably worse" at the ~15 pp resolving power means a candidate can be up to ~15 pp worse and
still pass. Stored results: the incumbent's false-advance rate is 4/40 = 10% and its false-suspend
rate is 4/30 = 13% (`index.csv` rows 47-48). A candidate at roughly 9/40 or 8/30 could plausibly
come out "not distinguishable". Shrinking the evaluation set widens the interval and makes passing
easier. The bar itself is the stakeholder's call. *Suggested improvement:* add an open question:
"Is a minimum evaluation-set size, or an explicit tolerated margin, required per point?" Then the
choice is made knowingly. For the synthetic points, cheap generation lets this be answered with a
larger n.

**m1 (minor) — The informal Laya-vs-qwen comparison probably uses different denominators.** The
qwen3-4b figures match the stored runs exactly: `falseAdvanceRate=4/40; falseSuspendRate=4/30`
(`index.csv` rows 47-48, pack 1.0.0). But the pack has 55 expected-false items: 40
`clear_suspend` plus 15 `boundary`, and the boundary items are scored separately (qwen 5/15). Laya's
"≈ 31%" fits 17/55 (30.9%) and fits no integer over 40 (12/40 = 30%, 13/40 = 32.5%). Laya's 47%
fits 14/30. Over 55, qwen would be 9/55 ≈ 16%. This is inferred from the arithmetic, not checked
against Laya's raw run. *Suggested improvement:* state Laya's figures as k/n over the same strata
as qwen's, or note the difference. The conclusion ("not a drop-in") holds either way.

**m2 (minor) — The "average" in FR-2/AC-6 isn't the statistic the stored baseline uses.** The
stored qwen3-4b latencies are p50s of 812 and 653 ms (p95 1265/1079), which supports
"~650–800 ms" (verified). FR-2's "about 0.7 s, so about 350 ms" treats a median as a mean. The mean
is likely higher given the p95, so the target is conservative, not wrong. *Suggested
improvement:* name the statistic (mean per decision over the evaluation set) and require the
incumbent to be re-measured in the same session (SMB FR-17 reference arm), not taken from the
table above.

**m3 (minor) — "Cheaper" and "more reliable" are in the Intent but no FR or AC measures either.**
FR-9 defines cheaper as "less local compute per decision", but no AC measures compute, so latency
is implicitly the proxy. FR-1 asks for parity, not improvement, so the reliability gain is
implicitly the structural one (no malformed output; `_coerce_verdict`'s bias-to-suspend at
`guards.py:488-506` is the free-text failure the doc describes, verified). *Suggested
improvement:* say so in one clause each, or add the measurement.

**m4 (minor) — Two gaps worth an open question before design.**
(a) The guard verdict's `rationale` is written into the `node_rationale` trace
(`executor.py:1520-1550`, `guards.py:19-23`). A decision model returns a probability, not a
rationale. Is a probability-only trace acceptable at each point?
(b) Production's intake guard runs on the recent-turns path (`seed_workflows.sh`, the D14 comment
on `intake`), while the 85-item pack mixes 51 `understanding` items and 34 `turns` items. Is
parity required on the path production actually uses, given that the turns subset alone (12/16)
resolves even less?

**n1 (nit) — "Message intent triage" collides with the existing `triage@v1` workflow**
(`config.TRIGGER_DEF_KEY = "triage"`, `seed_workflows.sh:11`), which clarifies requests and does
not classify them. Rename (e.g. "message intent classification") or add a one-line
disambiguation. Also: FR-9 is listed between FR-6 and FR-7.

**n2 (nit) — FR-6 has no AC, and FR-9 has none either.** FR-6 reads as a constraint ("one-off
build cost"), and FR-9's "nothing may prevent" can't be tested at this altitude. That's fine if
intended. If so, AC-4's "names the labeling source" could also name the labeler model, which
makes FR-6 checkable. AC-4 or AC-7 could also say "synthetic inputs" in the result, alongside the
labeling source, since an FR-8 evaluation on synthetic inputs and LLM labels measures something
narrower than real-traffic correctness.

### What's solid

- The FR-1 two-sided error bar, with the hand-off folded into the combined outcome (FR-3), and
  FR-2's whole-mix averaging are mutually consistent. A hand-off share of up to about half still
  meets ≤350 ms at ~20 ms vs ~700 ms, so there's no hidden contradiction.
- FR-5's "agreement with the labeling LLM, not correctness, and every result must say so", and the
  labeler ≠ incumbent rule (AC-4), are an honest, well-framed answer to the cost of having no human
  review.
- The inherited no-auto-apply rule is cited and is consistent with FR-7: model-bench reports, a
  human acts. Factual claims verified: `CallSurface = Literal["chat", "embeddings"]`
  (`lmstudio.py:63`), the incumbent `qwen/qwen3-4b-2507` (`models.json` `guard` default), the 85
  items, the bias-to-suspend fallback, and the 650–800 ms band.
- The decision log is clean and traceable.

### Open questions (for tico / stakeholder)

1. Per decision point (B1): what is the incumbent, the parity reference, the speed reference, and
   the hand-off target? For entity-match, would the model act as FR-10's "connected AI agent"
   confirmer, or change FR-9's no-auto-link rule?
2. Is a minimum evaluation-set size or an explicit tolerated margin wanted (M4)?
3. Is a probability-only trace acceptable where a rationale is traced today (m4a)?
4. Should guard-judge parity be judged on production's recent-turns path specifically (m4b)?

## Pass 2

**Scope.** The revised `docs/requirements/typed-decision-models.md` (readback re-confirmed
2026-10-30), re-read in full. Grounding re-checked: `index.csv` rows 47-48 (qwen3-4b 4/40,
boundary 5/15, 4/30; p50 653/812 ms), the pack's tier counts, `responder.py:66-72`, and
`document-ingestion.md` FR-9/FR-10.

**Verdict: approve with suggestions.** There are no blockers. Every Pass-1 blocker and major is
fixed, or settled by an explicit stakeholder decision. One new major (M5) is a one-clause
clarification: fold it in at the flip, or route it to design as a named constraint. The minors
can be taken or left.

CPG: considered, not relevant — a requirements-altitude re-review; the factual claims were
checked by reading the cited files directly.

**Growth-stops check.**
- **New scope:** the scope table's new columns and two new-ability rows, FR-10–FR-14,
  AC-8–AC-14, two out-of-scope rows (entity matches, acting on a type), and ten decision-log
  entries.
- **Rework, replaced in place (no stacked text):** the Laya figures, FR-1's "(speed-up points)"
  qualifier, FR-2's same-session rule, FR-3's one-measured-candidate sentence, FR-4's no-tuning
  clause, FR-5's amendment and guard clause (the old "85 items remain usable" sentence is gone),
  AC-3/AC-4, and FR-9's renumbering. Nothing is restated as history.
- **Log hygiene (nits; the log is append-only, so they stand as written):**
  - The entry at lines 207-209 asks "Minimum evaluation-set size?" but answers a different
    question (the human-set carve-out).
  - The size answer then arrives in a second entry, at line 212.
  - The "Overturns FR-5's 2026-10-30 carve-out" clause points at FR-5's *text*, not at a log
    entry. No log entry ever recorded that carve-out.
- **Restated fact (nit):** FR-5's parenthetical "(a human verifies every golden label)" restates
  SMB FR-19 instead of only citing it.

**Pass-1 dispositions.**
- B1 — **fixed.** Per-point incumbent / hand-off / consumer table added, entity matching removed
  from scope, and FR-10 is the bar for the incumbent-less points.
- M1 — **fixed.** FR-5 explicitly amends SMB FR-19 for this feature's packs only.
- M2 — **fixed.** FR-3: the combined configuration is one measured candidate.
- M3 — **fixed** for specialized models (FR-4/AC-3). The residual scope gap is m6 below.
- M4 — **resolved by stakeholder decision.** There is a minimum of 200 items (FR-11) and no
  tolerated margin. The residual risk is noted under m5.
- m1 — **fixed.** Laya and qwen are now both counted per stratum, and qwen's figures match
  `index.csv`. Laya's new counts (14/40 = 35%, 12/30 = 40%) don't reconcile with Pass 1's 31%/47%.
  That's unverifiable (an informal run with no stored artifact) and harmless to the conclusion.
- m2 — **fixed.** Same-session measurement, and "median ~650–810 ms" is correct.
- m3 — **not fixed** (low stakes). The Intent still promises "more reliable … cheaper" and
  "rating decisions", but no FR/AC measures compute or reliability gain, and no rating point is
  in scope.
- m4 — **fixed** by stakeholder answers (FR-13/AC-10, FR-14/AC-11).
- n1 — **fixed.** Renamed, with a `triage@v1` disambiguation, and FR order restored.
- n2 — **fixed.** AC-4 now names the labeler and the agreement caveat; FR-6/FR-9 are checkable
  through AC-4.

**Stakeholder answers folded in.**
- Which points are in scope? → Guard judge as a speed-up, plus message-type classification and
  join-in as new abilities; entity matching is out. (FR-10, FR-12, scope table)
- Which parity safeguards? → No tuning on evaluation items, and a minimum set size with no margin.
  (FR-4, FR-11)
- Join-in: which mistake is worse? → Both equally. (FR-10)
- Human set for the guard? → No. All labels come from the frontier model. (FR-5)
- Probability-only trace? → Yes. (FR-14, AC-11)
- Guard parity per evidence kind? → Yes. (FR-13, AC-10)
- Minimum set size? → 200 per point. (FR-11, AC-13)
- New-ability bar? → ≥90% agreement. (FR-10, AC-12)
- New-ability hand-off? → Allowed. (FR-10)
- What consumes classification? → A label only. (AC-14, out of scope)
- New-ability latency? → ≤350 ms average. (FR-10, AC-12)

### New findings

**M5 (major) — FR-10's "≥90% agreement" can be met by a degenerate model when one class
dominates, and the generator sets the class mix.** Plain agreement on an imbalanced set rewards
always predicting the majority class. A join-in set that is 90% "don't reply" is passed by a model
that never joins in. A message-type set dominated by "question" or "chit-chat" passes the same way.
FR-8 makes the inputs synthetic, so the class mix is whatever the generator produces, and the
requirements don't constrain it. For yes/no, "both error types weighed equally" implies a balanced
metric but doesn't name one. For pick-one, nothing is said at all. *Suggested improvement:* say
the 90% is measured per class (each class, or the class average), not on the pooled set. Or state
the evaluation set's required class mix. Also say whether 90% is the point estimate or the
interval's lower bound. At n = 200, an observed 90% has a lower 95% bound around 85%.

**m5 (minor) — FR-1's "roughly 15 pp at this lab's sample sizes" is stale under FR-11, and FR-13
splits FR-11's 200 items again.** At 200 or more items the paired resolving power is
substantially finer than 15 pp (inferred, not computed against model-bench's own instrument).
FR-13 then requires parity per evidence kind *and* per error type. If 200 is the per-point total,
each cell (kind × true class) is about 50 items, which brings resolving power back toward the
coarse end. *Suggested improvement:* drop the 15 pp parenthetical or tie it to n, and say whether
FR-11's 200 applies per point or per FR-13 cell.

**m6 (minor) — The tuning and model-bench rules are scoped to parity, not to new abilities.**
- AC-3 covers only "a specialized model … parity result (AC-1)". An off-the-shelf model with a
  tuned hand-off cut-off, or any FR-10 (AC-12) result, falls outside it.
- FR-7 requires model-bench before use only for "FR-1 parity".
- AC-5 compares with "the incumbent LLM's result for that role". New-ability points have no
  incumbent, and their roles aren't in SMB FR-20a's closed role vocabulary.

*Suggested improvement:* extend AC-3 to "any parity or new-ability result", and FR-7/AC-5 to the
FR-10 bar. Then state that model-bench gains the new roles.

**m7 (minor) — FR-12 has no acceptance criterion for the @mention path.** The scope row ("reply
although not @mentioned") implies an @mention still always gets a reply. But FR-12 only says the
locked rule "changes", and AC-8/AC-9 cover only the loop guard and the docs. *Suggested
improvement:* add an AC that an @mentioned, non-agent message still gets a reply whatever the
model says. Or, if the model may also decline mentioned messages, state that.

**Risks created by stakeholder decisions (flagged, not contested).**
- **Guard correctness anchor.** FR-5 now measures guard parity as agreement with the frontier
  labeler on synthetic inputs. The 85 human-curated items, the only correctness-verified guard
  data, are explicitly not the evaluation set. A candidate can reach agreement parity and still
  regress against human judgment, and nothing would show it. Reporting the 85 as a diagnostic
  beside the parity result would surface that without changing the bar. Whether to do so is the
  stakeholder's call.
- **Join-in exposure.** The join-in decision runs on every non-agent message, is trained and
  evaluated only on synthetic traffic (FR-8), and at the 90% bar can be wrong on up to about 1 in
  10 decisions. Its false joins are user-visible as unprompted agent replies. Combined with the
  "both mistakes equal" answer, this is an accepted exposure. Design should know it is there.

## Pass 3

**Scope.** First review of the implementation plan `docs/plans/typed-decision-models.md` (v1.1,
`active`, Owner `architect`, uncommitted), checked against `docs/requirements/typed-decision-models.md`
(Ready for design, read in full incl. FR-11a and FR-10's five labels). Grounding was checked by
reading: `model-bench/AGENTS.md`, `README.md`, `modelbench/{fingerprint,lmstudio,packs,runner,results,roles,stats}.py`,
`scoring/classification.py`, `packs/guard-judge-understanding/pack.json`, `results/index.csv`;
`falkor-chat/server/falkorchat/{guards,app,responder,trigger,background,api,mcp,modelconfig,executor,repository}.py`,
`config/models.json`, `docs/{DESIGN,QUERIES}.md`, `falkor-chat/AGENTS.md`; and SMB requirements
out-of-scope. I also ran three read-only checks: an in-process probe that adds both new fingerprint
profiles to `REQUIRED_BY_SCHEMA[1]` and re-validates all 95 stored `results/runs/*.json` records
(0 invalid; `ARM_KINDS` stays `{deterministic, model}`); a `GET /api/v0/models` (the `laya` entry
matches §2.3 verbatim); and a fetch of the LM Studio decisions doc page. I did not POST to
`/v1/decisions` (that would load a model) and did not run any test suite (a plan review changes
no code).

**Verdict: needs changes.** No blockers. The architecture is sound and well grounded: separate
components, data-defined decision points, a composite arm that counts as one candidate, and bars
printed but never acted on. Seven majors need fixing before teco splits the work. Most are local
spec edits: a trace-format crash on the fallback path, honesty refusals that can be skipped by
leaving out a field, eval-set reuse across candidates, the Holm direction for a parity claim, the
`stats.py` single-source invariant, copies of the renderer and policy left unpinned before
measurement, and two oversized units.

CPG: used cpg_falkorchat — confirmed `_build_llm_judge`'s 11 call sites (1 production, 10 tests/eval) and that `maybe_respond` has exactly two production callers (`trigger.maybe_trigger`, `background._safe_respond`), so join-in is self-contained.

**Growth-stops check (root `AGENTS.md`).**
- **Revision handling:** v1.1 was revised in place, which is correct because nothing gated
  v1.0. It carries a one-line dated note and no stacked `Update:` text.
- **New scope in v1.1:** none. **Rework:** SQ-1's five labels went into §3.2/Q1 (no stale
  six-label text left), SQ-2 into §3.6, and the verified shape into §2.3.
- **Restated content (nits):** the `/api/v0` probe question appears three times (§2.3, U1
  step 1, R-1). R-1 could cite U1 instead. §8's "Stakeholder input" line repeats the v1.1 note.
- **Prior findings and folded answers:** no prior plan findings. Both stakeholder answers are
  folded in consistently with the requirements.
- **Size:** no line is over 700 characters.

**Traceability (§7) verified row by row.** Every FR (incl. FR-11a) and all 17 ACs have a row.
Four rows are only partly satisfied by the design they cite: AC-13/AC-4/AC-7/AC-17 (M2), FR-3's
observability (m5), FR-4/AC-3 under eval reuse (M3), and AC-7 for the side-check stratum (m1).

### Findings

**M1 (major) — The specified trace suffix raises `TypeError` on the `llm-fallback` path, and
that turns a decision-model outage into failed runs.** Per §3.7, the trace appends
` [{decided_by} p={probability:.3f}]` whenever `decided_by` is set. `llm-fallback`
(decision error, timeout, or composite-mismatch bypass) has no probability, so `None:.3f`
raises. `_trace_step` runs inside the drive loop, right after `_record` (`executor.py:725-726`),
with no isolation. With `trace` on, every fallback would raise out of `_drive`, which is the
opposite of the fail-safe the fallback exists for. (The payload kind is `guard_judgment`, not
`node_rationale` as §2.2 says: `executor.py:1543-1550`.) *Fix:* print ` p=…` only when the
probability is not `None`. Add a U10 test with "decision error, trace on → run advances or
suspends normally, payload `[llm-fallback]`".

**M2 (major) — The honesty refusals can be skipped by leaving out an optional field, and the
bar kind is a declaration, not a derivation.** The §3.5 `validate_pack` axes (AC-13, AC-4, AC-7,
AC-17) apply only to packs where `labels.source == "llm-unreviewed"`. A pack with a
`decisionPoint` block but no `labels` block gets the bar section with none of the refusals. AC-13
says *any* parity or new-ability result. Likewise, "bar kind is derived, never declared" from
`incumbent` null/non-null: `incumbent` is itself pack data, so a guard pack with `incumbent: null`
is scored on FR-10 instead of FR-1. *Fix:*
- Key the refusals on `decisionPoint` being present, and require `labels` there.
- Bind the bar kind to the role through a pinned table (`guard-judge` → speed-up; the two new
  roles → new-ability). Refuse a mismatching `incumbent`.
- Refuse `run --candidate` on a pack with no `decisionPoint`.
- For a speed-up pack, refuse (or print) a candidate whose `handoffModel` ≠ `incumbent` (FR-3:
  "hand the rest to the existing LLM judge").

Mutation-test each axis: delete the key, and each refusal must still fire.

**M3 (major) — Nothing accounts for reusing one eval split across candidates, so FR-4's "no
cut-off tuning on eval items" can be broken by selection.** The fit-manifest check (§3.5) only
proves that the *declared* fit inputs are disjoint from the eval set. But U7 → (fail) → U6
re-fit or U8 retrain → U7 again, on the same pack version, lets a human pick the cut-off or
instructions that pass the eval set. That is tuning on eval items, and R-2 makes it the likely
path for the guard. *Fix:*
- Have the report print `candidates previously run on this pack version: K (ids)`. This is a
  within-pack `load_history(packId)` read, so FR-23-clean.
- Add a plan rule: a candidate refitted after seeing an eval result makes its adoption claim on
  a fresh eval split (a new `datasetVersion` → new `packVersion`). Ask U0 to set the budget
  under Q2/Q3.
- Name the closure record when a point misses its bar: a `HISTORY.md` entry plus a test report
  stating "does not meet", so "stops there" leaves an artifact.

**M4 (major) — "Built on the existing paired verdict machinery" with Holm makes parity *easier*
to pass as the metric family grows.** FR-1 holds iff *no* per-kind metric is distinguishably
worse. Holm lowers each test's α to control false claims of *difference*. For a claim that
rests on the absence of a difference, that is anti-conservative: four metrics instead of two make
"worse" harder to detect and "holds" easier to print. Rule 7 (`verdict()` never returns
`distinguishable` below `observable_floor`) does the same at ~50 items per cell. Q3 asks for "verdict family and α" but not the direction. *Fix:* in
U0 Q3, state the constraint that a multiplicity or floor rule must never make FR-1 easier to
pass. Require every `holds` line to print the resolving power beside it, e.g. `holds (can only
detect ≥ N pp at this n)`. Route to `data-scientist`.

**M5 (major) — The new statistics have no home that respects `model-bench/AGENTS.md`'s "`stats.py`
implements `docs/plans/small-model-benchmarking-ml.md` and no other source".** The FR-2 ratio of
means and its interval (Q4), the per-answer Wilson family (Q5), and the FR-1 parity reading (Q3)
come from a new `-ml` note. §3.3 calls the mean "a new report-level statistic" without naming the
module. U3's AGENTS.md edit covers only "the new profiles and the bar constants". *Fix:* decide
in the plan whether these go in `stats.py` (then amend the invariant to name both notes, by
section) or in a new module that cites the new note exclusively. Either way, the estimator's
`levels`/`clamp`/seed and `bar_holds`' `direction` take **no defaults** (AGENTS "nothing that
shapes a decision carries a default").

**M6 (major) — The three-copy contract (§3.2) is pinned only after measurement, and two of its
copies are never pinned.**
- The model-bench ↔ falkor-chat renderer cross-check lands in U9, after U7 has measured. U9
  also builds no falkor-chat join-in or message-type renderer to check (§3.7/U9 lists the
  client, config and `apply_policy` only).
- The decision-models `render.py` ↔ model-bench check is never assigned. A divergence there
  fits thresholds on a different input than the one measured, silently.
- `apply_policy` also has three copies (model-bench, falkor-chat, `fit.py`) with no shared
  vector. One `>=`/`>` flip changes which items hand off.

*Fix:* treat §3.2.1 as the canonical text. Add shared golden vectors for both
renderers and for `apply_policy` (incl. `p == l`, `p == h`, `l == h`) in U4/U5, before U6
fits. Make U9 own the falkor-chat renderers and prompts so its cross-check has a subject.

**M7 (major) — U1 and U3 are too large for gated delegation, and U3 mixes a refactor with
behavior changes.**
- **U1** spans six modules, incl. the invariant-dense `fingerprint.py`. Split it into: the
  `lmstudio.decide` probe plus fixture; the fingerprint profiles (alone, with the
  stored-records regression and the `ARM_KINDS`/`CALL_SURFACES` pins); the pure
  `decisions.py`; then the packs/runner/cli wiring.
- **U3** bundles the `classification` refactor, whose byte-identical regression only means
  something if no behavior changes alongside it, with eight behavior changes. Land the
  refactor first as its own gated unit, then the pack contract, the report bar section, and the
  importer.
- **Missing dependency:** Q6 (determinism → `basis`) has no consuming unit. U1's done-condition
  stores a run, and `RunResult.basis`/`designEffect` are required with no default
  (`results.py:695-701`). Name U1/U2 as Q6 consumers, or have U1 use a stated provisional basis
  and block U7 on Q6.

**m1 (minor) — AC-7's pack-level `labels.inputs: "synthetic"` is false for the guard pack's own
side-check stratum.** The 85 copied items are human-curated, not synthetic. *Fix:* refuse any
item whose `provenance.origin != synthetic` in stratum `eval`, and exempt `side-check` by
stratum. Also have U0 Q1 forbid side-check items as generation seeds. A paraphrase of one would
pass the exact-hash leakage check and contaminate FR-15.

**m2 (minor) — Stratum and the per-kind verdict metrics need declared outcomes, not report
filtering.** Today `score_item` declares exactly one scoreable metric per item and derives the
by-path rates in `aggregate` from `detail.path` (`classification.py:232-276`). Paired verdicts
need per-item `counts`. So: each eval item must declare its per-kind metric, side-check items
must declare only `sideCheck*`, and `stratum` must be required, with no default, on
decision-point packs. That way `report.py` "infers nothing" (AGENTS) and a stratum-less item can
never count toward `MIN_EVAL_ITEMS`. Also print the *scored* n beside FR-11's 200 when items are
withheld.

**m3 (minor) — The `INPUT_FIELDS_BY_ROLE` pin's widen mutation is inert.** `input_hash` filters
on `if k in item`, so widening the table with a key no item carries changes nothing. Widening
with a label-like key (`tier`, `expected`) is the dangerous case: identical inputs with flipped
labels then hash apart and escape the leakage check. *Fix:* the shared golden vector's item
carries non-input fields (`itemId`, `tier`, label, `stratum`), so both shrink and widen redden.
Also key the table one way (role vs. point: §3.5 says `role`, §3.4 says `point`).

**m4 (minor) — The bar tests only probe the failing side.** §5 tests 351 ms → not met, but not
350 ms → met, 0.90 → met, or ratio 2.0 → met, so a `<`/`<=` direction mutation stays green.
FR-10 says "350 ms or less" and "at least 90%". Add the equality cases, plus a test that
`bar_holds` ignores the interval (§3.6 claims one).

**m5 (minor) — FR-3's "how often each point hands off must be observable" is designed only for
the guard.** The INFO line is specified under "Guard front". Join-in records nothing, and
message type has `messageTypeBy` only. R-6 relies on "the log line". *Fix:* emit the same line
from all three points (or from `DecisionClient`'s caller in `decisions.py`).

**m6 (minor) — The measured guard hand-off is not quite the runtime one.** The scorer parses
with `bool(parsed.get("decision"))` and never applies `_coerce_verdict`'s strict-`True` and
rationale-contradiction checks (`classification.py:253-261` vs `guards.py:488-506`). Both arms
share the gap, so parity is fair, but §3.7's "the only composite that ever runs is the one that
was measured" overstates it. Port the check into the hand-off parse, or state the divergence.

**m7 (minor) — The AC-9 sweep has no mechanical done-condition, and its file list misses
statements.** The prescribed grep returns 68 hits needing judgment, and "finds no '@mention
only' statement" isn't a predicate. It also skips `app.py:564` ("`AgentResponder` replies to an
`@mention`"), `config.py:85`, and `api.py`/`mcp.py`'s comments. U11 omits DESIGN §5's `Message
{msgId, text, role, createdAt, threadId}` schema line. *Fix:* widen the file list to
`server/falkorchat/*.py`, list the expected rewritten sites in the unit brief, and add DESIGN §5
to U11.

**Nits.**
- §2.3 says "documented". The LM Studio page documents only a predicate request and
  `answers`, not choice/score or the response fields. Cite tico's probe capture in
  `ws:agent-team` (doc `316050e8…`), which corroborates the plan's shape (choices as
  `{value, instructions}`, `output_tokens` always 0).
- Q6 quotes the requirements as saying Laya "varied slightly across reloads". No version of
  the requirements says that; they say "deterministic across repeats".
- §3.7 cites "QUERIES.md §4" for the set-query, but §2.2 correctly cites §6.
- `CallSurface` is also declared at `fingerprint.py:41`, not only `lmstudio.py:63`. `"handoff"`
  is a composite, not an LM Studio surface, so `lmstudio.CallSurface` needn't gain it.
- `timeoutSeconds: 5` and `_DEFAULT_KIND_TIMEOUTS["decision"] = 10.0` give the same call two
  sources. Choose one.
- The candidate's `handoffModel` (`qwen/…`) and falkor-chat's (`lmstudio/qwen/…`) differ in
  form, so "copied from the candidate" needs a stated mapping.
- §3.6 should cite SMB's out-of-scope "Hard pass/fail regression gating", the rule actually
  being reconciled, not only the README.
- FR-2's same-session check must read top-level `run.sessionId` (AGENTS invariant).
- Join-in should take turns strictly before `msg_id` rather than "drop the new message",
  because concurrent posts may follow it.
- The KINDS rejection rationale is weaker than stated: `extraction` is already a non-`KINDS`
  kind (`models.json`, `ingestion.py:137`). The measured-composite reason stands on its own.

### What's solid

- The component split: `decision-models/` holds the hosted labeler and fine-tuning, which keeps
  model-bench zero-dependency and FR-23-clean. FR-8 is enforced by an import allowlist test.
- Additive schema-1 profiles. Verified: all 95 stored records still validate, `ARM_KINDS` is
  unchanged, and `temperature` is forbidden on `model:decisions`. Flat `handoff*` fields reuse
  the forbidden-set machinery unchanged.
- Join-in ordering: the loop guard is first, an @mention is handled before the decider runs,
  `join_in=None` gives today's behavior, and both "decider never called" tests are specified.
  Hand-offs and errors fail closed. `_build_llm_judge` and its 10 test callers stay untouched.
- The measured-composite bypass. Recalibration as a monotone map for predicates, with the
  correct caveat for choice confidence. Bars are printed, never gate an exit code, and adoption
  stays a human config edit.
- Message-type schema work is routed through `graph-dba` (U11), and the RAM line is assigned.

### Open questions

1. M3: does the stakeholder accept that a candidate refitted after seeing an eval result needs
   a fresh eval split, i.e. more labeling cost, or is "report the reuse count" enough?
2. M4: should FR-1 parity print the resolving power on every `holds` line? That is a
   presentation choice for `data-scientist` and the stakeholder, since FR-1 has no margin.

## Pass 4

**Scope.** The plan `docs/plans/typed-decision-models.md` v1.2 (revised in place, still
uncommitted), re-read in full against the requirements at `bf9eae77`. I re-ran the plan's AC-9
grep verbatim and rechecked the cited code sites (`executor.py:725-726`, `:1543-1550`;
`classification.py:232-276`; `results.py:695-701`; the guard pack's item shape). No suites were
run.

**Verdict: approve with suggestions.** Every Pass 3 major and minor is fixed. Two new majors,
both one-paragraph edits, should be folded in before teco delegates U1. They are not blockers:
- N1: the FR-4 reuse count can be undercounted.
- N2: §8's "in parallel" contradicts the units' real dependencies.

teco can verify those two edits with a line check. N3–N5 can be taken during implementation.

CPG: considered, not relevant — a re-review of plan text. The code sites it cites were rechecked by direct reads, and the Pass 3 CPG results still apply.

**Growth-stops check (v1.1 → v1.2, 839 → 1003 lines).**
- **Rework, not new scope:** the growth is the Pass 3 answers (§3.2.2 vectors,
  `decision_stats.py`, `BAR_KIND_BY_ROLE`, the reuse and scored-n lines, `record_decision`, the
  U7 closure record, the unit split, R-5/R-6) plus the two stakeholder answers. None of it is
  feature scope beyond the requirements.
- **Revised in place:** a single dated line records the revision, the same in-place pattern this family used after requirements Pass 1. No
  `Update:` text is stacked.
- **Restated content:** the `/api/v0` question is now stated once (U1a), with R-1 citing it. Two
  small duplicates remain (nits):
  - §2.3's determinism bullet restates the requirements' Problem section instead of only citing
    it.
  - R-10 repeats §3.2.2's skip rule.
- **Size:** no line is over 700 characters.

**Stakeholder answers folded in (requirements `bf9eae77`).**
- M3: is a fresh evaluation set required after a re-tune informed by evaluation results? → No.
  Every result reports the reuse count instead. (FR-4)
- M4: must every parity line print its detectable difference? → Yes, always. (FR-1)

**Pass 3 dispositions.**
- **Majors:**
  - M1 — **fixed.** The ` p=` suffix prints only when the probability is non-`None` (§3.7). U10
    tests "decision error with trace on → normal advance/suspend, `[llm-fallback]`". The kind is
    corrected to `guard_judgment`, which matches `executor.py:1543-1550`.
  - M2 — **fixed.** The refusals key on `decisionPoint`, `labels` is required, `BAR_KIND_BY_ROLE`
    is pinned, `--candidate` is refused without `decisionPoint`, and `handoffModel ≠ incumbent`
    is refused on speed-up packs. U3b deletes the key each refusal reads.
  - M3 — **fixed per the stakeholder answer** (the reuse line in §3.5, R-5, the U7 closure
    record). Its counting rule has a new defect, N1.
  - M4 — **fixed.** Q3 carries the no-loosening constraint, and every `holds` prints its
    detectable difference (§3.6).
  - M5 — **fixed.** `decision_stats.py` implements the new note only. `stats.py`'s invariant is
    unchanged, and there are no defaults on decision-shaping arguments.
  - M6 — **fixed.** §3.2.2 sets up canonical vectors in model-bench, and the policy and hash
    vectors land in U1c, before any fit. The new falkor-chat renderers are owned by U9. One
    caveat is N3.
  - M7 — **fixed.** U1 and U3 are split, U3a is a refactor only, and U1d records the provisional
    basis with U7 blocked on Q6. The dependency order has a new problem, N2.
- **Minors:**
  - m1 — **fixed.** Per-item `origin` is checked in `eval`, `side-check` is exempt by stratum, and
    side-check items are never used as seeds.
  - m2 — **fixed.** Per-item declared per-kind and `sideCheck*` metrics, required `stratum`, and
    the scored n is printed.
  - m3 — **fixed.** The hash vector items carry non-input fields, and the table is keyed by role
    in both components.
  - m4 — **fixed.** U1c tests the equality boundaries, `direction` has no default, and
    `bar_holds` has no interval parameter.
  - m5 — **fixed.** `record_decision` is called at all three points.
  - m6 — **fixed.** The decision-point guard parse ports `_coerce_verdict`.
  - m7 — **partly fixed.** The site list is now explicit and DESIGN §5 is in U11, but the new
    done-condition has the problems in N4.
- **Nits:** all taken. Q6's "varied" wording is superseded, because the requirements now record
  the two-load difference themselves.

### New findings

**N1 (major) — The reuse count can be undercounted in two ways.** §3.5 counts "distinct
candidate `armKey`s … on the same `packId` and `packContentHash`":
- **A re-tuned candidate that keeps its `candidateId`** (the file edited in place) has the same
  `armKey`, so it counts once.
- **A re-import under a new `packVersion`** with the same eval items, after a manifest, side-check
  or provenance edit, changes `packContentHash` and resets K to 0.

FR-4 asks how many candidates have run "on that same evaluation set". Two smaller problems:
- "No later than this run's `startedAt`" includes the run itself, but U3c's test says "only
  earlier".
- In a `compare` with several arms, "this run" is undefined.

*Fix:*
- Key the evaluation set by a hash of the sorted `eval`-stratum `input_hash`es plus their labels,
  and record that hash in the fingerprint.
- Count distinct `candidateContentHash` values, not `armKey`s.
- Define K per candidate arm as strictly earlier runs.
- Test both undercounts: same id with new content counts twice, and a re-versioned pack with the
  same eval set keeps K.

**N2 (major) — §8 step 1 says "U1a–U1d, U2 and U3a in parallel", but the units depend on each
other.**
- U1d's decision-model-only arm scores through `score_answer`, and U2's step 5 parses through
  `answer_from_chat`. Both are hooks that U3a creates.
- No unit is assigned to add the three hooks to the `ItemScorer` Protocol in `runner.py`.
- U2 extends U1d's `_drive_decision_items`, and both edit `runner.py`.
- U1d's `_build_fingerprint` needs U1b's profiles.

Run in parallel as written, these units collide on `runner.py` or block on missing hooks.
*Fix:* state the order as a graph: U1a ∥ U1b ∥ U3a, then U1c, then U1d (which also adds the
Protocol hooks), then U2. Serialize U1d → U2 on `runner.py`, as §4 already does for U3c/U4 on
`report.py`.

**N3 (minor) — The canonical vectors confirm themselves unless their expected outputs are
written by hand.** U1c authors `apply-policy.jsonl` and `input-hash.jsonl` in the same unit as
`apply_policy`/`input_hash`. If the expected values are produced by running the new function,
a boundary bug becomes canonical and every copy agrees with it, and the same goes for the
render vectors in U4. *Fix:* §3.2.2 should say expected outputs are derived from the spec, never
from the code under test:
- policy rows from §3.3's inequalities;
- render strings from §3.2.1's canonical text;
- `render-guard.jsonl` from falkor-chat's real `_render_judge_user`, the source of truth.

Also note that U3b adds a falkor-chat test, so its gate must run that suite with
`falkor-chat/server` as the working directory.

**N4 (minor) — The AC-9 done-condition fails on a document the sweep may not edit, and it misses
sites by construction.** Run verbatim now, the grep also hits
`falkor-chat/docs/plans/demo-environment-bringup.md:100`, an `active` devops plan whose
troubleshooting line stays true. So "no hits outside `HISTORY.md`" would force a cross-owner
edit. The pattern only matches the four known strings, so it confirms the list rather than
finding new sites. For example, `docs/manuals/llm-provider-config.md:24,230` ("the assistant
that replies to @mentions") is not matched, and manuals are `tico`'s, not U12's implementer's.
*Fix:*
- Exclude `docs/plans`, `docs/reviews` and `docs/test-reports` (records), or list the expected
  residual hits.
- Route any `docs/manuals/` rewording to `tico`.
- Keep the analyst read of each site as the actual check.

**N5 (minor) — U1c names roles and input fields before they exist.** U1c lands
`BAR_KIND_BY_ROLE` (three roles) and `INPUT_FIELDS_BY_ROLE`, but `join-in-judge` and
`message-type-classifier` enter `roles.py` only in U4. A pin that binds these tables to `ROLES`,
which is the AGENTS rule, either fails in U1c or has to be written loosely. The message-type
hash fields and renderer also depend on U0 Q1 (`priorTurn`), which neither U1c nor U4 lists as a
dependency. *Fix:*
- U1c defines only the `guard-judge` rows.
- U4 adds the two new roles' rows together with the role tables and their pins.
- Mark U4 "needs U0 Q1".
