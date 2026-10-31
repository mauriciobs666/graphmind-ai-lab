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

## Pass 5 — U0 method note (typed-decision-models-ml.md)

**Scope.** `docs/plans/typed-decision-models-ml.md` (first version, uncommitted, ~4,860 words),
against plan v1.3 §4 U0 (Q1–Q7), §3.3–§3.6 and §6, the requirements, and `modelbench/stats.py`.
I re-ran the note's three scratchpad scripts with `python3 -I` and recomputed the parity figures a
second way (below). I read the code the note's failure handling depends on: `runner.py:376-410`,
`scoring/classification.py:264-285`, `falkor-chat/server/falkorchat/app.py` (`_LlmGuardJudge`),
`guards.py:262` and `executor.py:691-693, 1451-1462`. I did not re-check the §Q4.4 break-even figures,
which teco already re-ran.

**Verdict: needs changes.** The statistics are correct. Every figure I re-ran matches, and Q3
meets its binding constraint. None of the findings below is a blocker. Three majors still need
an edit or a stakeholder answer before U3c, U5 or U7 consume the note:
- **P5-M1:** the failure-handling table (§Q4.3) misstates what the runtime does and changes the
  chat arm without saying so.
- **P5-M2:** the fitting margins act as gates and can close a point before its bar is measured.
- **P5-M3:** two decisions are presented as method, but the stakeholder owns them: the
  two-replicate rule and a labelling volume of about 9–14×.

CPG: considered, not relevant — a statistics note. There is no `cpg_modelbench`. The code it
depends on was checked by direct reads.

**Figures re-run.**
- `parity.py` matches every §Q3 figure:
  - the MDD₈₀ table;
  - the reuse-adjusted limits, 13.1→15.2→16.4→17.2 and 8.2→9.1→9.4→9.6;
  - 1 − 0.8^{K+1} = 0.20/0.36/0.49/0.59;
  - the null false-fail rates, 1.05%/4.86%/9.04%;
  - the four §Q3.6 cells.
- `p_worse(8, 2) = 56/1024 = 0.0547`.
- `fr10.py` matches §Q5.3, §Q5.4, the half-widths 0.077/0.060 and the `T_fit` figures
  0.08/0.71/0.93.
- `latency.py` matches 0.476/0.471 and the worked ratios 3.02 [2.65, 3.51] and 2.10 [1.90, 2.35].
- **Independent check of the power formula.** I enumerated the trinomial (b, c) outcomes
  directly. The rejection region was evaluated in exact `Fraction`s rather than through
  `crit(m)`. This reproduced all twelve MDD cells I tried, at n = 50/75/100/150 and several ψ̂,
  including the 14.5 pp at ψ̂ = 0.16. Power is monotone in δ on the grid at n = 50, ψ̂ = 0.10 and
  at n = 100, ψ̂ = 0.20, so "first grid point = ceiling" holds. A 40,000-draw Monte Carlo gives
  power 0.7999 at δ = 13.1 pp, n = 50, against 0.8013 exact.
- **One stated figure is wrong:** §Q1.3's "60 is the smallest size…". See P5-m1.

**Q3's binding constraint: met.**
- **No correction or floor.** Cells are unadjusted, with no Holm and no Rule 7. The test is
  one-sided at 0.05, so the "worse" direction is tested at 0.05 rather than 0.025. That makes
  rejection easier, which is the safe side.
- **A bigger family never helps.** Combining cells and replicates as a conjunction means P(holds)
  cannot rise when a member is added.
- **Direction.** H₀ is "not worse". It is rejected on b_w in excess of c_w, so "worse" is
  the rejection and `holds` is a failure to reject. That is the shape FR-1 and the stakeholder's
  "no explicit margin" answer imply.
- **Resolving power.** The printed MDD₈₀ and `U` together make a "too few items to tell" pass
  visible, as FR-1 requires. §Q3.6 row 1 shows exactly that case.

**Growth-stops check (first version, so no prior revision exists).** The length is mostly method.
There is no debate-log prose and no finding→resolution table. Restated text is small:
- §0's R-list;
- the FR-2 quote in §Q4.1;
- the same-session rationale in §Q4.4.

Two sections are transient and should not stay in an `active` note. See P5-n2.

### Findings

**P5-M1 (major) — §Q4.3's chat-failure rows misstate the runtime and change existing behaviour
that is not on the handed list.**
- **Wrong runtime claim.** For a failed or timed-out hand-off or incumbent chat call, the table
  scores the item as "§3.2's parse fallback (guard: suspend)" because that is "what the runtime
  would have done". It is not.
  - `_LlmGuardJudge.__call__` calls `llm.complete` with no `try`.
  - `guards.py:262` and `executor.py:1453` do not catch the exception either.
  - So a failed judge call reaches `_drive`'s `except Exception` → `_fail_with_note`
    (`executor.py:691-693`). The run fails; it does not suspend. Only *unparseable* output becomes
    `decision: False`.
- **Unlisted change to the chat arm.** Today a failed chat call records `wallClockMs=None` and
  `scoreable={}` (`runner.py:393-399`, `classification.py:275-284`). The rows "x_i = elapsed" and
  "scored as the parse fallback" therefore change the existing chat arm, and they need
  elapsed-on-failure capture that does not exist. Handed item 2 covers only the decision call.

*Fix:*
- Keep today's semantics for chat-call failures in both arms: withheld, counted, dropped
  pairwise for FR-1, and printed beside FR-2/FR-10 as a count.
- Alternatively, if the note keeps imputation, state it as a measurement convention rather than
  runtime fidelity, and add elapsed capture plus chat-arm scoring to the handed list.

**P5-M2 (major) — The fitting margins act as gates, so a point can close without its bar being
measured.**
- **Where the margins bite.** §Q2.2 sets `T_fit = 0.94`. It then marks a class infeasible if the
  hand-off LLM's own calibration agreement is below `T_fit`. §Q2.1 marks a band infeasible when
  `s > s_max − 0.05`, and an infeasible candidate "is not run on eval".
- **Effect.** A hand-off LLM at 0.92 calibration agreement on one class would have to clear
  FR-10's bar of 0.90 on eval. Instead it never reaches eval. In effect the bar became 0.94 at the
  fitting stage. The note never says outright that 0.94 is a fitting target and that the eval bar
  stays 0.90 (§Q5.1).
- **Inconsistent margins.** FR-10's 350 ms fitting constraint carries no margin, unlike FR-2's.

*Fix:*
- **Fit to the margin, but fall back below it.** When the fit cannot reach 0.94, fit to the
  highest attainable target ≥ 0.90. Record the target in the fit manifest, and run eval.
- **Narrow "infeasible".** Reserve it for cases where even the requirement value is unattainable
  on calibration: 0.90, `s_max` itself, or 350 ms.
- **Say it once.** Add one sentence: "T_fit is a fitting target, never a bar (FR-10's bar is
  0.90)."
- **Pick one rule for the latency margin.** Either give the 350 ms constraint the same treatment
  as FR-2's, or drop FR-2's −0.05.

**P5-M3 (major) — Two of the note's rulings are stakeholder decisions, but it presents them as
method.**
- **(a) Q6's "met in both replicates".**
  - The plan did delegate the replicate question to U0. But a bar that must hold in two separate
    measurements is a more pessimistic reading than SQ-2's "judged on the measured value".
  - It adds a candidate run per point.
  - It roughly doubles the null false-fail rate the note prices in §Q3.2. With independent
    replicates, 1.05/4.86/9.04% becomes at most 2.1/9.5/17.3%. Correlated replicates give
    less.
  - The evidence for it is sound: 14/30 vs 12/30 means at least 2 of 30 items flipped. But the
    rule needs a yes.
- **(b) §Q1.3's set sizes and §Q1.2's three blind votes multiply the paid labelling.** Against the
  plan's prior of 200 eval items per point, labelled once (600 calls):
  - **Kept items: 3.0×.** Eval is 400 + 200 + 300 = 900, and calibration matches it, for 1,800.
  - **Labeller calls: 9×**, at three votes per item (5,400 calls).
  - **About 14×** with the 1.6× over-generation for filling cells (≈ 2,880 items labelled, ≈ 8,640
    calls).
  - **U8, if triggered.** Its train split (≥ 3,500 kept items, ×1.6, ×3) adds about 16,800
    calls, ≈ 42× in total.
  - **Unpriced.** The variants per seed that calibration and train "may use" are unspecified.
    If every variant is labelled, multiply again.

*Fix:* mark both rulings "stakeholder decision", as handed item 6 is. Put the multiplier in the
note's §Q1.3.

**P5-m1 (minor) — §Q1.3's "60 is the smallest size at which…" is false.** P(pass | a = 0.85) is a
sawtooth in n, because the bar is `10·k ≥ 9·n`.
- Exact values: n = 44 gives 0.190, n = 49 gives 0.123, and n = 59 gives 0.106. Multiples of 10
  are local worsts.
- Every n ≥ 52 satisfies ≤ 0.20. n = 51 is the last that fails.

The choice of 60 is defensible. The justification is not. *Fix:* "every n ≥ 52 keeps it ≤ 0.20,
and 60 is the round size above that". This matters because withholding can leave a class at
n < 60.

**P5-m2 (minor) — §Q4.3 drops a failed decision call from FR-1, which contradicts the note's own
governing principle.**
- The runtime answer on such an item is the incumbent's, so it is a concordant pair. Dropping it
  leaves `p_worse` unchanged, but it shrinks `n_c` and ψ̂, so the printed rates and MDD change.
- Combined with the new "not assessable (scored n < 200)" rule, a candidate whose decision calls
  often time out becomes unassessable, even though the runtime would serve it fine.
- That rule is itself a new verdict state. Plan §3.5 only prints the scored n, and the rule is
  not on the handed list.

*Fix:* keep these items as concordant pairs. List "not assessable" for the architect to absorb.

**P5-m3 (minor) — §Q4.3's "censoring" assumes the bench's decision-call timeout differs from the
runtime's.**
- If the bench used the point's `timeoutSeconds`, a timeout's elapsed time would be exactly what
  the runtime spends, and nothing would be censored.
- The candidate file carries no timeout today, and `RunConfig.requestTimeoutSeconds` is the chat
  timeout.

*Fix:* the candidate carries `decisionTimeoutSeconds`, which the bench uses and §3.7 copies
verbatim, as it already copies `question`/`policy`. The "≤ R" then becomes unnecessary.

**P5-m4 (minor) — §Q3.5's "upper bound" holds for non-adaptive tries only.** Re-tunes informed by
eval results are adaptive (FR-4 allows them). Adaptive selection can exceed 1 − 0.8^{K+1}.
*Fix:* add the qualifier. The printed figure stays as advice.

**P5-m5 (minor) — Places an implementer must guess.**
- `labelVotes` is a scalar (0, ⅓, ⅔, 1). That fits predicates, but for the 5-label choice the
  §Q7 soft label needs a vote vector.
- The "8-word span" for the shingle checks has no tokenization rule: case, punctuation and
  whitespace.
- §Q1.5 leaves the `{speaker}` value unstated: role, display name or agent name.
- §Q3.3 does not say what prints when MDD is unattainable (`n_c < 5`; the script's `mdd` returns
  `None`). `stats.UnattainablePower` is the precedent.

**P5-m6 (minor) — FR-8 says inputs are generated "(by the frontier labeler)". §Q1.2 deliberately
uses a different frontier model to generate them.** The intent survives, and R-4 asked for the
separation. It is still a literal mismatch with a stakeholder-worded FR. *Fix:* `tico` rewords
FR-8's parenthetical to "a hosted frontier model". That is a decision-log line, not a new
question.

**P5-n1 (nit)** — §Q3.3's "Not best case" is only partly true. Whenever ψ̂ < δ, ψ = δ and π_c = 0,
which *is* the best case. All of the n = 50 ψ̂ ≤ 0.10 cells are best case. Reword: "never below
best case, and equal to it when ψ̂ < MDD".

**P5-n2 (nit)** — Two sections are transient:
- **"Reproducing the figures"** describes scripts that live in a session scratchpad, and those
  scripts do not print every cited figure. The slow-tier `s_max` column and 0.34–0.40 are hand
  arithmetic. Commit the scripts or cut the section to one line.
- **"Handed to architect"** is a routing list. Delete it once v1.4 absorbs it (growth-stops: no
  disposition lists).

**P5-n3 (nit)** — §Q2.3's arXiv 2609.26758 exists. Its abstract (alphaXiv) reports that only
yes/no option names fell below chance, which is a reversed ranking that thresholds cannot fix.
That bears directly on the guard's `advance` predicate. Replace "unverified" with this.

### Handed-to-architect routing

| # | Real plan gap? | Method the plan absorbs? | Stakeholder? |
|---|---|---|---|
| 1 Replicates in §3.6/U7 | Yes. Plan §3.3/U7 anticipated it ("if Q6 requires") | Yes: grouping, per-replicate lines, reload step | **Yes**, for the "met in both" rule and the extra run (P5-M3a) |
| 2 Failed decision call not withheld | Yes. It reverses plan §3.3's explicit "withholds… not handed off" | Yes, it is runtime-faithful. **Extend** it with elapsed-on-failure capture and the chat-failure rows (P5-M1), and keep the pairs as concordant (P5-m2) | No |
| 3 FR-1 line gains `U`, ψ̂ and the reuse-adjusted limit | Yes | Yes | No. Covered by the FR-1 detection-limit answer |
| 4 `priorTurn` for message type | Anticipated (§3.5 "if U0 adds it"; U4 needs U0 Q1) | Yes. It touches U4 vectors and U11's preceding-row read | No |
| 5 `decision-models/` additions | Yes | Yes. Also add the `labelVotes` shape for choice and the tokenization (P5-m5) | **Yes**, for the labelling cost (P5-M3b). FR-8 wording goes to `tico` (P5-m6) |
| 6 Cell minimums: print, not refuse | Yes. §3.5 has only the 200 total | Yes, as **print** | Only if a refusal is wanted. I agree with printing |
| 7 Within-load re-check; `inputTokens` | Yes, small | Yes | No |
| 8 Per-label `minConfidence` policy | No, not yet. It is conditional on U6 | Defer. At most one contingent line in §3.3 | No |

**Missing from the list** (the architect should add them):
- the "not assessable (scored n < 200)" verdict state (P5-m2);
- the decision-call timeout source (P5-m3);
- what U7 records when a fit is infeasible: whether that is a closure record like a missed bar,
  once P5-M2 narrows "infeasible".

### What's solid

- **Q3 is the right instrument.** The exact one-sided test with no correction is monotone by
  construction. The note rejects Holm and Rule 7 for the right reason: they are absence-claim
  hazards. The plug-in-ψ̂ MDD is more honest than `stats`' strict-dominance figure.
- **Every figure reproduces.** Each one I re-ran matches, two ways for the power.
- **Q4.1 uses a ratio of means.** That is the correct reading of "on average across the whole
  mix". The bootstrap is deterministic and reuses `stats.percentile`.
- **Q1's design protects the eval set.** One variant per cluster keeps `designEffect = 1.0` true.
  Near-miss seeds, blind three-vote labelling and intent kept as a sampling variable (never a
  filter) are sound defences against an easy set.
- **Q2.5's argument is right.** A threshold needs a ranking, not calibrated probabilities, so no
  calibration map is needed for choice confidence.
- **Q7 has a go/no-go export test.** It also refits on the served GGUF.

### Open questions

1. **Stakeholder.** Does a bar claim require the bar to be met under two separate model loads?
   The cost is one more candidate run per point, and an equal candidate's false-fail rate rises to
   at most about 2×. (P5-M3a)
2. **Stakeholder.** Labelling volume of about 9× the plan's prior reading in labeller calls (≈ 14×
   with over-generation; ≈ 42× if U8 runs). Accept it, or trade it down? The levers are two votes
   plus a tie-break, a calibration set smaller than eval, or 50 per guard cell at the cost of the
   MDD rising from 6.7–14.1 to 13.1–19.9 pp. (P5-M3b)

## Pass 6

**Scope.** This pass re-gates the method note v1.1 (`docs/plans/typed-decision-models-ml.md`,
~5,990 words, uncommitted) and the requirements diff (`git diff --
docs/requirements/typed-decision-models.md`), against Pass 5 and plan v1.3. I re-ran the three
scripts in `docs/plans/typed-decision-models-ml/` with `python3 -I`. I re-read the code that
§Q4.3 now relies on:
- `executor.py:691-693, 719`;
- `runner.py:393-399`;
- `guards.py:567`;
- `background.py:358-369`.

I also checked the §Q2.3 abstract figures against alphaXiv. The stakeholder answers are binding
and are not findings here: both loads, the full labelling volume, and an infeasible fit still runs
and counts toward K.

**Verdict: the note needs changes; the requirements edits get approve with suggestions.**
- **Note.** Every figure re-runs, and every Pass 5 finding is addressed. But three of the
  revision's own fixes conflict with the note's governing principle or with each other (P6-M1 to
  P6-M3). Each is a few-line edit that teco can verify by line check. A seventh pass is not
  needed unless the fix changes the design.
- **Requirements.** They agree with the note, including `tico`'s reading that FR-2's 2× also
  needs both loads. The growth is new scope. One clarifying clause in FR-4 is needed (P6-M3), and
  there are two small restatements (P6-r1). Nothing here is "needs changes", so **Ready for
  design stands**.

CPG: considered, not relevant — a re-review of a statistics note and a requirements diff. The
code sites were checked by direct reads, and there is no model-bench CPG.

**Figures re-run.**
- `parity.py` matches every §Q3 figure. It also gives the two-replicate bounds 2.08/9.49/17.27%
  and the unattainable-MDD rows at n = 3–4.
- `fr10.py` gives the sawtooth (largest failing n = 51) and the labelling arithmetic: 1,800 kept
  items, 5,400 calls (9×), 8,640 calls (14.4×), and 25,440 calls with U8 (42×).
- `latency.py` now prints the slow-tier columns 0.405/0.393 and 0.338/0.404.
- §Q2.3's abstract figures, 70.4 pp and 93.8% → 23.2% AUC, are verbatim on alphaXiv.

**Pass 5 dispositions.**
- P5-M1 — **fixed for the guard.** §Q4.3 now matches the guard path: `_drive`'s
  `except Exception` → `_fail_with_note` (`executor.py:691-693`) and `wallClockMs=None`
  withholding (`runner.py:393-399`). It is **not fixed for the new abilities**; see P6-M1.
- P5-M2 — **fixed for FR-10.** §Q2.0 says margins are never bars, and the §Q2.2 ladder relaxes to
  0.90 and then to `s_350`. It is **not fixed for FR-1's quality target**; see P6-M2.
- P5-M3 — **answered by the stakeholder.** (a) Both loads → FR-11a, the decision log and §Q6.
  (b) Volume accepted → the decision log and §Q1.3, with the figures re-run above.
- P5-m1 — **fixed.** "Every n ≥ 52", floor 52, design size 60.
- P5-m2 — **fixed.** A failed decision call is now a concordant pair, and "not assessable" is
  handed item 4. A counting gap remains; see P6-m1.
- P5-m3 — **fixed.** `decisionTimeoutSeconds` is handed item 3. P6-m3 covers the fit side.
- P5-m4 — **fixed.** The non-adaptive qualifier is added. Its new phrase "a floor on the
  inflation" is wrong; see P6-n1.
- P5-m5 — **fixed.**
  - `labelVotes` is a vote map.
  - The tokenization is specified.
  - The speaker rule matches `guards.py:567`.
  - The unattainable form is printed by `parity.py`.
- P5-m6 — **fixed.** FR-8 now says "a hosted frontier model, which need not be the labeler".
- P5-n1 — **fixed** (§Q3.3 "never below best case").
- P5-n2 — **fixed.** The scripts print every cited figure, and the handed list carries its
  removal rule. A residual nit is in P6-n1.
- P5-n3 — **fixed.** The figures are verified. The note's "a search summary lists" Laya is not
  supported by the abstract, which names only Jev.

**Stakeholder answers folded in since Pass 5.**
- Must a bar hold across loads? → Yes, in each of two loads (FR-11a; FR-1, FR-10, AC-1/6/12).
- Accept the labelling volume? → Yes, at full design size (no FR change).
- Does an infeasible fit still run on eval? → Yes, once, and it counts toward K (FR-4).
- Must the generator be the labeler? → No (FR-8).

### New findings

**P6-M1 (major) — §Q4.3 drops a new-ability hand-off chat failure, but the runtime fails closed on
it. That makes FR-10 easier to meet.**
- **The runtime.** A chat failure fails the run only on the guard. Join-in runs inside
  `background._safe_respond`, whose `except Exception` logs and swallows
  (`background.py:358-369`). So a hand-off LLM failure there produces **no reply**, which is a
  decision of `no`. Plan §3.7 also has `_safe_classify` mirror `_safe_embed`, so the classifier
  case produces **no label**.
- **The note.** §Q4.3 row 2 drops these items from FR-10. That conflicts with its own first
  governing principle, and with its third ("no rule may make a bar easier"). An item the
  composite would get wrong (a `yes`-labelled join-in item, or any message type) leaves the
  denominator.

*Fix:*
- Restrict row 2 to the guard and the incumbent arm.
- For a new-ability hand-off chat failure, score the fail-closed answer, as row 3 does, with
  `y_i` = elapsed.
- That needs elapsed capture on the hand-off chat call as well. Amend handed item 2, which says
  chat failures "keep today's withholding … in both arms".

**P6-M2 (major) — The guard's quality target never relaxes, so §Q2.0's "one rule covers all three
bars" is not true for FR-1.**
- **What the note does.** §Q2.1 fits `D_c ≤ 0`, a point estimate stricter than FR-1's actual
  test (one-sided exact McNemar at 0.05). §Q2.0 calls it "the requirement value", so it has no
  ladder.
- **What that loses.** The guard is the point R-2 says is bound by hand-off share. When
  `D_c ≤ 0` needs `s > s_2`, a band with a small positive `D_c` that does not reject on
  calibration could meet both bars. The fit never looks for one. It returns
  `fit-infeasible: latency`, and the eval run (now mandatory) tests a band that is bound to fail
  FR-2.

*Fix:* add the missing rung. When `D_c ≤ 0` forces `s > s_2 − 0.05` (or `> s_2`), relax each
cell to "the calibration cell's one-sided exact McNemar does not reject". Take the smallest `s`
meeting `s_2`, and record it as `below-margin` with the target used. That is FR-1's requirement
value, applied the same way §Q2.2 relaxes to 0.90.

**P6-M3 (major) — "Runs once on eval" conflicts with "met in both loads".**
- **Where it appears.** FR-4's new sentence, §Q2.0 and handed item 8 each say every candidate is
  "run on the evaluation set once".
- **The conflict.** FR-11a and §Q6 make "met" require two load replicates. Read literally, an
  infeasible-fit candidate gets one run and can never be "met". If that run happens to meet the
  bar, nobody is told to run replicate 2.

*Fix:* "once" means one evaluation, and an evaluation is the two-replicate pair. Replicate 2 may
be skipped when replicate 1 already fails, because the conjunction is decided. K counts the pair
once, which §Q6's content-hash rule already gives.
- `tico` adds "(in both loads, FR-11a)" to FR-4's sentence.
- The note fixes §Q2.0 and item 8 to match.

**P6-m1 (minor) — A failed decision call is kept but not counted.**
- §Q4.3 counts "every dropped row". Concordant pairs from decision-call failures are kept rows,
  so nothing prints them.
- A candidate whose decision calls all time out prints FR-1 `holds` in every cell, at ψ̂ ≈ 0 and
  with a tight MDD. Only FR-2 reveals it.

*Fix:* print `decisionCallFailures k/n` per replicate beside `handoffShare`, and add it to handed
item 2.

**P6-m2 (minor) — Handed item 6 says the `priorTurn` rows land "before U4".** Plan v1.3 deliberately
lands them **in** U4, with the role tables and their pins (the Pass 4 N5 fix). "Before U4" brings
back N5's unbindable pins. *Fix:* "in U4". U4 already lists U0 Q1 as a dependency.

**P6-m3 (minor) — Fit-side gaps an implementer would have to guess.**
- `decisionTimeoutSeconds` is an **input** to `fit.py`, because the fit uses it before the
  candidate file exists. It is not only a field of the candidate. Say where it comes from (handed
  item 3).
- `fitOutcome` is single-valued, but `ranking`, `latency` and `agreement` can be infeasible
  together. Define the precedence, or make it a set.
- Handed item 8 puts the per-instruction AUC "in the fit manifest". Plan §3.4 defines the manifest
  as the sorted `input_hash` list, hashed for the leakage check, so the AUC there changes a pinned
  format. Put it in the candidate's provenance or in a separate fit report.

**P6-n1 (nits).**
- §Q3.5's "the printed figure is a floor on the inflation" is wrong. Under adaptive re-tunes it is
  no bound in either direction. Say so.
- Duplicated facts. The equal-candidate false-fail rate appears twice in different roundings
  (§Q3.2 1.1/4.9/9.0%, §Q6 1.05/4.86/9.04%). "Target, not a bar" is stated three times (§Q2.0
  heading, §Q2.0 body, §Q5.3). Keep one home for each and cite it.
- Handed item 7's "`tico` is rewording FR-8" is now stale.
- §Q2.3's "a search summary lists" Laya: the abstract names only Jev. Mark it unverified.
- `docs/plans/typed-decision-models-ml/` is a code directory inside a document kind. That falls
  outside the filename grammar, and it is untracked. Commit it as the note's companion, or give it
  a home in a component when U3c lands.

**P6-r1 (minor, requirements) — Small restatements in the diff.**
- FR-1's "in each of two separate loads (FR-11a)" and FR-10's "Like every bar, it must hold in
  each of two separate loads (FR-11a)" restate FR-11a, which is the rule's home. The citation
  alone suffices. The ACs restating it is normal.
- FR-11a's "A parity bar (FR-1, FR-2)" calls FR-2 a parity bar. Say "a speed-up point's bars
  (FR-1, FR-2)".

**Growth-stops check.**
- **Note: +~1,130 words, mostly legitimate.**
  - *New scope* (stakeholder answers, plus one new diagnostic): §Q2.0 fit outcomes, the
    infeasible-runs rule, §Q2.3's AUC diagnostic, the §Q1.3 labelling-volume block and §Q6's
    cost.
  - *Rework* that replaced text in place: §Q4.3, rebuilt rather than appended; §Q1.3's sawtooth;
    the specs that P5-m5 asked for.
  - The handed list grew from 8 to 11 items: the three I listed as missing in Pass 5.
  - Restated facts are P6-n1's duplicates. §Q4.3's first bullet narrates the falkor-chat call
    chain, but it is design evidence that is cited once, and it is acceptable.
  - No history or debate prose.
- **Requirements: new scope.** There are four decision-log entries, each one question → one
  answer with its FR, and "extends SQ-2" is named. The two restatements are P6-r1. No line is
  over 700 characters.

### Handed-to-architect list (11 items) as plan v1.4 input

The list is correct and almost complete. The `[settled]` markings on items 1, 7 and 8 are right,
and no other item needs one.

**Amend:**
- **Item 2:** add the new-ability hand-off rows (P6-M1) and the failure count (P6-m1).
- **Item 3:** `decisionTimeoutSeconds` is a fit input (P6-m3).
- **Item 6:** "in U4", not "before U4" (P6-m2).
- **Item 7:** drop the stale `tico` clause.
- **Item 8:** "once" means the two-replicate pair (P6-M3), and the AUC goes outside the manifest
  (P6-m3).

**Add:** the guard's FR-1 relaxation rung (P6-M2) as a `fit.py` objective change under item 8.

## Pass 7

**Scope.** This pass checks the method note v1.2 and `tico`'s requirements edits for P6-M3 and
P6-r1. Following the coordinator's brief, the design check covers only the four changes the
author declared: P6-M1, P6-M2, P6-m1 and P6-m3. I computed `p_worse` over (b, c) grids myself to
test §Q2.1's new selection rule.

**Verdict: needs changes. There is one major, P7-M1, and the stopping condition is below.** Every
Pass 6 finding is addressed. But the rule P6-M2 introduced for choosing a band rewards discordance,
so it tends to pick bands that FR-1's eval test is least able to catch.

**Stopping condition (falsifiable, line-checkable).** §Q2.1's selection at the McNemar rungs (2 and
3) must use a key that never prefers a band with a larger worst-cell excess, and, at equal excess,
never prefers a band with more discordance. One rule that passes: the lexicographic key
`(max_c D_c ↑, max_c m_c ↑, s ↑)`, with McNemar non-rejection kept as the constraint.

Check it on this case: two bands identical except in one cell. Band A has (b, c) = (2, 0). Band B
has (19, 16). The rule must pick A, and A must also beat a band with (3, 1). If the next revision
meets this and changes nothing else, no Pass 8 is needed.

CPG: considered, not relevant — a statistics note and a requirements diff. No model-bench CPG
exists.

**Pass 6 dispositions.**
- P6-M1 — **fixed.** §Q4.3 row 3 scores a new-ability hand-off failure fail-closed, with
  `y_i` = elapsed. It cites `background.py:358-369`, which I rechecked. Item 2 is extended.
- P6-M2 — **fixed in structure.** There is now a four-rung ladder with a McNemar relaxation rung.
  The new selection key is P7-M1.
- P6-M3 — **fixed.** §Q2.0 and item 8 define one evaluation as the replicate pair; replicate 2 is
  skippable, and the pair counts once toward K. FR-4 now reads "(in both loads, FR-11a)".
- P6-m1 — **fixed.** `decisionCallFailures` and `handoffCallFailures` print per replicate (§Q4.3,
  item 2).
- P6-m2 — **fixed.** Item 6 says "in U4".
- P6-m3 — **fixed.** The timeout is a `fit.py` input (§Q2). There is a precedence order
  (`ranking` > `agreement` > `latency`), and the fit report is separate from the manifest.
- P6-n1 — **fixed except one nit.** These are fixed:
  - §Q3.5 now says the bound holds "in neither direction";
  - the false-fail rate has one home (§Q3.2), which §Q6 cites;
  - "never bars" is stated once (§Q2.0);
  - the stale clause is gone;
  - Laya is marked unverified.

  `docs/plans/typed-decision-models-ml/` is still untracked under a document kind. That is
  teco's call at commit time.
- P6-r1 — **fixed.** FR-1 and FR-10 say "judged per FR-11a". FR-11a reads "a speed-up point's
  (FR-1 parity, FR-2 speed)".

**Design check of the four declared changes.**
- **P6-M1 (fail-closed scoring plus elapsed capture): sound.** It follows the runtime, which
  swallows the error and gives no reply or label, and it is the conservative reading.
  - *Nit:* a failed hand-off's elapsed time is bounded by the bench's chat timeout
    (`requestTimeoutSeconds`), not by the runtime's provider timeout or retry chain. FR-10
    latency for those rows is only approximate. Printing the counts makes that visible.
- **P6-m1 (the two failure counts): sound.** An all-failing candidate is now visible beside its
  FR-1 `holds` lines.
- **P6-m3 (precedence and fit report): sound.** The precedence only selects which outcome is
  named, and the fit report keeps every bar's value.
  - *Nit:* item 8 does not say where the fit report lives or whether the importer copies it.
    One clause in plan v1.4 settles it.
- **P6-M2 (max-min `p_worse`): one defect, P7-M1.** Two parts of it are fine:
  - **No leakage.** `fit.py` reads calibration only (plan §3.4). The rule is deterministic given
    calibration. Eval items are seed-disjoint, hash-disjoint and shingle-checked, and refused at
    run time if they overlap. So the selection cannot see eval.
  - **No bad interaction with K or replicates.** K counts content hashes, so a re-fit under a
    changed rule is K+1 and a rerun of the same fit is not. The replicate conjunction is
    applied after selection and only makes the bar harder. I found no interaction.

### New findings

**P7-M1 (major) — The max-min `p_worse` selection rewards discordance, so it picks bands that the
eval test is least able to catch. That conflicts with §0's "no rule may make a bar easier".**
- **Why.** At fixed net excess D, `p_worse` rises with the discordant count:
  - D = 2: (2,0) gives 0.25, (4,2) gives 0.344, (10,8) gives 0.407, (18,16) gives 0.432;
  - D = 3: (3,0) gives 0.125, (19,16) gives 0.368, which beats (2,0)'s 0.25 at D = 2.
- **Effect.** The rule prefers more disagreement with the LLM. It can even prefer a larger net
  excess, as long as the band is noisier. On eval, that band has higher ψ, so the test has less
  power and the printed MDD is larger. Its `holds` is therefore more likely for the same true
  worsening.
- **Why it slipped through.** The author's goal, keeping the band off the rejection edge, is
  right. But `p_worse` is a test statistic, so it mixes up the size of the effect with how much
  evidence there is for it.
- **Mitigation.** The printed ψ̂ and MDD disclose it. But the fit should not optimise toward
  weak tests.

*Fix:* the stopping condition above. Rank first by worst-cell excess (`max_c D_c`, rung 1's own
tie-break inverted), then by discordance, then by `s`. The edge is avoided because the least-excess
band is chosen, and noise is never rewarded.

## Pass 8

**Scope.** I reviewed the method note at v1.5, against the U1a live probe (`ws:agent-team`
document `480ef825…`). The checks:
- the over-limit premise, against the code;
- bias from the forced hand-off;
- its interaction with the v1.4 kept-failure rules and the guard ladder;
- the input caps, recomputed;
- growth-stops.

I read this code: `falkor-chat` `retry.py`, `transport.py:140-160`, `llm.py`
(`FallbackClient`) and `app.py:690-730` (`_render_judge_user`, `JUDGE_USER_MAX_CHARS`);
`config/models.json` `timeouts`; `model-bench` `lmstudio.py` (`_raw_post`, `decide`,
`LMStudioCallFailed`). I re-rendered the 85 side-check items. Whether long messages hand off or
fail closed is with the stakeholder, so the over-limit treatment is judged on its merits below.

**Verdict: needs changes, with two majors and a line-checkable stopping condition.** The
forced-hand-off method is sound and does not loosen any bar. But v1.5 does not finish wiring it:
- **P8-M1:** the over-limit case is missing from the failure table, so an implementer reading the
  table today would score an over-limit item under the "decision call fails" row.
- **P8-M2:** the new mechanics have no route into the plan.

**Stopping condition (falsifiable; a line check, no Pass 9).** The next revision must do all of
these and change nothing else:
1. Add an over-limit row to the §Q4.3 table, placed above the decision-call-failure row. Give an
   entry in all four columns. State that over-limit items are excluded from
   `decisionCallFailures`.
2. In §Q2.1–§Q2.3, state that an over-limit calibration item:
   - is a hand-off in every band (it has no `p`);
   - enters `s`, and enters `t̄_{L|H}` with the failed call's elapsed time included;
   - is excluded from the ROC-AUC diagnostic.
3. State what happens to an over-limit item on a candidate with no `handoffModel`.
4. Replace the "Absorbed by …2.md" line with a short **v1.5 handed list** naming the plan edits
   (P8-M2).

CPG: considered, not relevant — a method note plus code-site reads. There is no model-bench CPG,
and the falkor-chat sites were read directly.

**Pass 7 disposition.**
- P7-M1 — **fixed.** §Q2.1 sorts by `(max_c D_c ↑, max_c m_c ↑, s ↑)`, and the
  (2,0) / (3,1) / (19,16) check is stated.

**1. Over-limit premise, verified against the code.**
- **Distinguishable by body text: yes in falkor-chat, with a structured field. In model-bench,
  only through the exception message.**
  - falkor-chat: `transport.py:146-152` raises `ProviderCallError(kind="http", status=500,
    detail=body)`, with the body capped at 500 characters. The probed text, "input (N tokens) is
    too large to process…", starts the body, so it is reliably inside `detail`.
  - model-bench: `_raw_post` puts the body only into the message string (`_truncate`, 500
    characters). `LMStudioCallFailed` carries `status` and no body field. Detecting over-limit
    there means parsing `str(exc)`.
  - The string is llama.cpp's, as surfaced by one LM Studio version. A wording change would turn
    over-limit into an ordinary decision failure. That fails safe: guard → `llm-fallback`; new
    abilities → fail closed, which is stricter. It is still a silent change of method.
- **"retry.py would otherwise retry it": not true today.**
  - `is_transient` does class every 5xx as transient (`retry.py:89-90`).
  - But `call_with_retry` is used only by `ingestion.py:138` and `embedding.py:259`. The guard
    judge, the responder and the planned `DecisionClient` path (plan §3.7) have no retry
    wrapper.
  - `FallbackClient` (`llm.py`) applies only to chat chains, and the shipped defaults are single
    models.
  - So "never retry it" is a guard against a future wrapper, not a fix for current behaviour.
    The note's "timeouts are not retried (`retry.py`)" is true but cites the wrong reason: nothing
    on these paths retries at all.
  - `timeouts.guard` and `timeouts.agent` are 180 s, as the note says.

**2. Bias.**
- **No bar becomes easier to meet relative to the runtime the bench mirrors.**
  - Over-limit is a function of the input only, so it is identical in both replicates.
  - On the guard, a forced hand-off is the LLM's answer: concordant, as any hand-off is.
  - FR-2 is charged the failed call *plus* the hand-off, which makes it harder.
  - §Q2.3 lets the fit see that longer instructions cost hand-offs.
- **Two residual effects (P8-m2):**
  - The over-limit share on eval is whatever the generator produced. If it is about 0, the
    forced path is never exercised end to end, and eval understates production's `s` and
    latency. `overLimit k/n` and `record_decision` make that visible after the fact.
  - On new abilities, long items score the hand-off LLM's agreement rather than Laya's. That is
    faithful to the recommended runtime, but per-class agreement then mixes the two models in a
    proportion set by length.
- **Under the fail-closed alternative:**
  - The guard is unchanged in effect, because its only safe fallback is still the LLM
    (`llm-fallback`).
  - On new abilities, over-limit items score the fail-closed answer: join-in `no` (agreement on
    true-`no`, disagreement on true-`yes`); message type, a disagreement in every class.
  - That makes FR-10 harder, concentrated in the classes whose messages run long (likely
    `complaint`/`escalation`). It can push the fit to `fit-infeasible: agreement`.
  - It also makes FR-10 latency easier: only the failed call is charged.
  - Those items would count in `decisionCallFailures`, or in a fail-closed `overLimit`. The
    over-limit row would read "as the decision-call-failure row".

**3. Interactions.**
- **Kept-failure table: conflict (P8-M1).** The over-limit treatment exists only as prose.
  - An over-limit item *is* an HTTP 500 decision-call failure, so the table's "candidate
    decision call fails" row matches it.
  - Under that row the guard gets a concordant pair at `y_i` = elapsed + the incumbent's `x_i`,
    instead of a real hand-off call. The new abilities get fail-closed, the opposite of the
    recommended treatment.
  - The counts can double-count, because `decisionCallFailures` is not told to exclude
    over-limit items.
- **Guard ladder: unspecified, not wrong (P8-m1).** Over-limit calibration items have no `p`.
  - The band search over "distinct calibration `p` values" must treat them as hand-offs in
    every band. They raise `s`'s floor, add 0 to every `D_c`, and their `t̄_{L|H}` must include
    the failed call.
  - If that floor exceeds `s_2`, every rung but 4 is infeasible, which is correct.
  - The §Q2.3 AUC must exclude them.
- **Decision-only candidate (P8-m1).** With no `handoffModel`, a forced hand-off is impossible.
  The note does not say whether the item falls back to the guard's LLM or fails closed.
- **Load rule × over-limit: no conflict.** Nothing suggests a 500 reloads the model.

**4. Input caps, recomputed from the renderers.**

| Point | Text cap (chars) | Fixed labels + speakers | ≈ tokens at 4 chars/token |
|---|---|---|---|
| guard | 6,000 whole message (`JUDGE_USER_MAX_CHARS`, `app.py:692`) | included | ~1,500 |
| join-in | 6 × 400 prior + 400 new = 2,800 | ~50 + 7 speaker names | ~700+ |
| message type | 400 prior + 2,000 = 2,400 | ~30 + 1 speaker name | ~600+ |

- **The note's figures match** for turn text.
- **The side-check range reproduces.** Re-rendered, the items span 134–436 characters:
  `understanding` 171–436, `turns` 134–430.
- **Unstated: the question text counts too.** The probe's two-question call reported 113 input
  tokens, summed across questions, for a 62-character input (about 12 tokens). That leaves about
  44 tokens of question per question even with short instructions. So the usable evidence budget
  is 512 minus the question tokens, roughly 1,500–1,850 characters at 4 chars/token, not 2,048. The note says the question counts (§Q2.3)
  but does not carry that into the caps paragraph (nit).
- **The 4 chars/token estimate** is unmeasured, as the note says.

### New findings

**P8-M1 (major) — §Q4.3's table has no over-limit row, and its decision-call-failure row
captures the case.** See interaction 1. *Fix:* stopping condition, item 1. Under fail-closed, the
row reads "as the decision-call-failure row", and it still needs the count rule.

**P8-M2 (major) — v1.5 adds plan mechanics with no route to the plan.** The handed list says
"Absorbed by `docs/plans/typed-decision-models2.md`", but these edits are new:
- over-limit detection in both clients: a structured `body` (or `detail`) field on
  `LMStudioCallFailed`, a match on a stable substring (`too large to process`), and the
  verbatim probe body pinned as a fixture next to `lmStudioAppVersion`;
- `reason="over-limit"` in `ItemResult.detail` and in `record_decision`;
- chat timeouts set to the runtime's 180 s in both arms;
- the explicit-load precondition (§Q6's load rule);
- plan2's R-13, "silent context truncation", which is now false.

*Fix:* a v1.5 handed list, stopping condition item 4. It is `architect`'s to absorb, after the
stakeholder's long-message answer.

**P8-m1 (minor) — How over-limit items enter fitting and decision-only candidates is
unspecified.** See interaction 2 and 3. *Fix:* stopping condition, items 2–3.

**P8-m2 (minor) — The forced path may be untested on eval.** With no length quota, eval may
contain no over-limit items. *Fix:* nothing in the note. The handed list should ask for a U1e or
U2 offline test that feeds the pinned 500 body through `decide` and the runner. The bar line
prints `overLimit 0/n` as a reminder.

**P8-n1 (nit)** — §Q4.3 cites `retry.py` for "timeouts are not retried". The operative fact is
that no retry wrapper exists on the guard, responder or decision path. The caps paragraph should
also subtract the instruction tokens.

**Growth-stops (5,616 → 6,093 words, claimed as new scope).** Mostly new scope: new evidence and
the method it forces (over-limit, timeout sources, the load rule, Q2.5's confidence fact). The
rejected-alternatives list is design rationale and is acceptable. Restated facts: the probe's
505/513 boundary, the 500 text and the 3,286 ms cold call. Their home is the probe document and,
after U1e, `model-bench/docs/HISTORY.md`. Cite them rather than restate them (minor). No history
or debate prose.
