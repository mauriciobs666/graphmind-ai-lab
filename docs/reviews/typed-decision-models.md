# Typed Decision Models — Requirements Review

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
