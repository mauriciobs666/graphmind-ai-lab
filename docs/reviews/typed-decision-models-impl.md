# Typed Decision Models — Implementation Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

Code gates for the model-bench units of `docs/plans/typed-decision-models.md`. The plan review is
`docs/reviews/typed-decision-models.md`.

## Pass 1

*2026-10-30. U1b, U3a and U1a (offline half), gated per unit.*

### Scope & verdict

**Reviewed:** the uncommitted working-tree diff `git diff dc2f8884 -- model-bench/`, plus the untracked
`model-bench/tests/fixtures/classification/` and `model-bench/tests/fixtures/lmstudio/decisions_response.json`.
The baseline is plan v1.3: §2.3, §3.3, §4 U1a/U1b/U3a and §5. Conventions come from
`model-bench/AGENTS.md`. `cpg/` and the other `docs/` changes were not reviewed.

**Verdicts per unit:**

| Unit | Verdict | Blockers | Majors |
|---|---|---|---|
| U1b, fingerprint profiles | **approve with suggestions** | 0 | 0 |
| U3a, classification split | **approve** | 0 | 0 |
| U1a, `LMStudio.decide` (offline) | **approve with suggestions** | 0 | 0 |

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded, and the diff is small enough to read and run directly.

**What I ran:**
- **Full suite** (from `model-bench/`): `2074 passed, 4 deselected, 2 xfailed`. `ruff check .` is clean.
- **Baseline suite** at `dc2f8884`, extracted with `git archive` into the scratchpad: `1814 passed, 3 skipped`.
  The 3 skips are the `requires_falkor_chat` guard, which skips because the scratch tree has no `falkor-chat/`.
  That is 1817 tests, which matches HISTORY's stated baseline.
- **Per-file collection deltas:** fingerprint +198, results +1, classification +8, lmstudio +50 (plus 1 deselected).
  These reproduce HISTORY's running totals of 2016, 2024 and 2074.
- **Mutants**, each run on a copy that was then restored, with `cmp`-verified restores. The working tree is unchanged.

### U1b — fingerprint profiles

**Verdict: approve with suggestions.**

I checked the field sets by executing them against plan §3.3:
- `model:decisions` has 33 fields: the embeddings 26 plus exactly the seven §3.3 names.
- `model:handoff` has 45 fields: the decisions 33 plus exactly the twelve §3.3 `handoff*` names.
- There are no missing fields, no extra fields, and both sets are supersets of the ones below them.
- **Tiers:** every `handoff*` field has the same tier as its chat counterpart. `handoffTemperature` is `present`, like `temperature`, and the rest are `nonempty`.
- `fitManifestSize` is `present`. §3.3 does not state a tier for it, and a count of 0 is a real answer, so this is a sound reading.
- `ARM_KINDS` = `{model, deterministic}` and `CALL_SURFACES` = the four suffixes.

**Schema 1 with no bump.** This is consistent with the rules. `model-bench/AGENTS.md` states no explicit bump rule. What the code relies on (`fingerprint.py`, the `REQUIRED_BY_SCHEMA` comment) is that a record is validated against its own schema entry. No existing profile's required set changed. The forbidden sets grew only by 19 new names. The stored-records test is the evidence that no stored record carries any of those names. A newer record with `callSurface: "decisions"`, read by an older build, quarantines as `unknown`. A schema-2 bump would produce the same result (`benchSchemaVersion` `unknown`), so the bump buys nothing.

**The stored-records test does load every record.** There are 95 files in `results/runs/*.json`, and `git ls-files` agrees. The test:
- asserts `len >= 95`;
- loads each pack through `load_history`, with `invalid == []`;
- asserts that the loaded `runId` set equals the set of `runId`s in the files, so no record can be skipped silently;
- asserts byte equality of every file after the read.

I made the chat profile require `evalSetHash`, and this test reddened. So the test is not vacuous.

**Findings**

- **U1b-n1 (nit). The "three profiles" text is now stale in two places.**
  - `model-bench/AGENTS.md:44` still reads "its members would become the **three** profiles".
  - `modelbench/fingerprint.py:233` reads "would make its members the three *profiles*".
  - Both describe the counterfactual of deriving `ARM_KINDS` from the forbidden mapping, which would now give five. The edit rewrote the parenthetical on line 39–40 in place, which is correct, but missed these two counts in the same paragraph.
  - **Fix:** say "the profiles", with no count, in both places, so the count cannot drift again.
- **U1b-n2 (nit). Line 40 of `model-bench/AGENTS.md` is 123 characters.** The file wraps at about 100. Rewrap it. This is cosmetic and is not a context-file bar breach, since the bar is 700.

### U3a — classification split

**Verdict: approve.**

**The golden is derived from the pre-split code.** I checked this by executing it, independently of the test and of the implementer's scratch script:
1. I extracted the **whole** `model-bench/modelbench` package at `dc2f8884` with `git archive`. This was not only `classification.py`, so `results.py`, `runner.py` and `packs.py` are the baseline's too.
2. I imported it ahead of the working tree. `modelbench.__file__` resolved into the scratch tree, and the module had no `answer_from_chat`.
3. I ran the test module's own `split_rescore_lines`/`split_digests` over the 85-item real pack.

Results:
- The pre-split tree gives `1445` cases, `all = 273ecf13…0c313`. This is equal to `tests/fixtures/classification/score_item_before_split.json`, every per-variant digest included.
- The working tree gives the same.
- The two full-text dumps (old and new) are byte-identical (`cmp`). They are also byte-identical to the implementer's `u1/u3a_before_full.txt`.
- The implementer's `u1/classification_head.py` is byte-identical to `git show dc2f8884:…/classification.py`.

**Corpus coverage of the old `score_item`:** every branch is reached.
- **The `result is None` split:** `timeout` → `fail`, and the else branch through `no_response`. `withheldFor="load"` is not exercised, but it takes the same else branch.
- **Parse outcomes:** parse `None` → `parse_failure`, from `garbage`, `""`, `None`, mid-sentence and two-own-line. Parsed truthy and falsy values include the string `"false"`, `0`, `null` and the no-key reply.
- **The count's two arms:** `falseSuspendRate` vs the other two metrics, over all three tiers × both paths. The real pack has all six (tier, path) cells.
- **The one unreachable combination**, `(True, parse_failure)`, is impossible by construction.
- **The equivalence classes are thin.** The 15 replies fall into only three outcome classes. That is enough here, because `ItemResult` depends only on `(advanced, outcome)`. The corpus exists to drive the parse, and it does.

**Deviations**

- **Deviation 2 (`score_answer` has no `decided_by`/`probability` yet): acceptable, but it leaves a hand-off note for U1d.**
  - Plan §4 U1d says "U3a's `classification` functions satisfy" the §3.3 hooks, and §3.3's `score_answer` signature carries `decided_by, probability`.
  - So **U1d**, not U2, is the first unit whose Protocol will name them, even though `detail.decidedBy` is listed under U2.
  - HISTORY's "whichever unit first writes `detail.decidedBy` adds them" is ambiguous on that point. See Open question 1.
  - **The hole to avoid:** when the parameters arrive, the old pack's `score_item` path must not start writing `decidedBy` into `detail`. The U3a golden test will catch that if it happens, so the safety net is in place.
- **Deviation 5 (`AttributeError` instead of `KeyError`): acceptable.** I reproduced it on both trees. It needs **both** a malformed item (missing `tier`, missing `itemId` or an unknown tier) **and** a `ChatResult.message` that is not a Mapping. The old code raised `KeyError` and the new code raises `AttributeError`, because the parse now runs before the item lookup. Both inputs are programmer errors, and both versions still raise. Nothing catches either exception type, and no later unit depends on it.

**Findings:** none.

### U1a — `LMStudio.decide` (offline)

**Verdict: approve with suggestions.**

The §2.3 shape claims check out against the plan and the probe document (`ws:agent-team` `316050e8…`).
- **Request:** `{model, input, questions}`, with choice options as `{value, instructions}` objects. This is asserted in `test_decide_posts_model_input_and_questions_to_the_decisions_route`.
- **Response:**
  - top level is `{model, answers, usage}`, with no `stats`/`model_info`/`runtime`;
  - a predicate answer is `{type, name, probability}`;
  - a choice answer is `{type, name, choice, probabilities:[{value, probability}], confidence}`;
  - `usage` has the five keys, with `output_tokens: 0`.
- **The "no raw body" claim is true.** The probe document records the schema and timings but no response body.
- **The `_provenance` string correctly flags what was invented:** the values, the names, and the inner keys of `*_tokens_details`. The `*_tokens_details` keys are named in §2.3 but appear in neither the probe document nor a capture.
- **Taxonomy:** `decide` and the decisions warm-up are each asserted over all 7 `_EXPECTED_FOR_POST` cells, and I counted 7 collected tests each.

**Specific check 1 — the surviving `model=` mutant.**
I reproduced it. Replacing the response's `model` with the requested `model` leaves all 160 tests green.

- **Does anything downstream rely on the response-reported model? No, not today.**
  - No code in `modelbench/` reads `DecisionResult.model`, and none reads `EmbedResult.model`.
  - The fingerprint's `modelKey` comes from the catalog `ModelInfo` (plan §3.3: "model identity, verbatim from the LM Studio catalog").
  - No unit in plan §4 consumes the served model.
- **It is a pre-existing pattern.** The same mutant applied to `embed()` also survives the full suite (2074 green).
- **So it is not a correctness defect.** It is a public field whose provenance contract (response, not request) is unpinned. It is rated minor (U1a-m1) because U1d/U2 could reasonably start cross-checking served against requested, and a regression to the requested value would then silently attest whatever was asked for.

**Findings**

- **U1a-m1 (minor). The `DecisionResult.model` provenance is unpinned.**
  - The only assertion, `result.model == "laya"` at `tests/test_lmstudio.py:886`, uses a body whose `model` equals the requested one, so request and response cannot be told apart.
  - **Fix (`tests/test_lmstudio.py`):** one test that requests `model="laya"` against a body whose `"model"` is different (e.g. `"ggml-org/laya"`) and asserts `result.model == "ggml-org/laya"`. Add a body with no or non-string `model`, asserting `result.model is None`.
  - Optionally do the same for `embed`, which has the identical gap from before this unit.
- **U1a-m2 (minor). The closed bounds of `_probability` are unpinned.**
  - Mutating `0.0 <= value <= 1.0` to `0.0 < value` or to `value < 1.0` survives all 160 tests. The malformed table only has `-0.01` and `1.5`.
  - A real `1.0` or `0.0` from Laya is plausible after float rounding. Under either mutant, that item would raise `LMStudioCallFailed` and be **withheld**, which silently biases the scored set toward non-extreme items.
  - **Fix (`tests/test_lmstudio.py`):** accept-tests for a predicate `probability` of exactly `0.0` and `1.0`, and a choice `confidence` of `1.0`. One parametrized test is enough.
- **U1a-m3 (minor). Two pass-throughs on the decisions warm-up are unpinned.**
  - Hardcoding `timeout_s=7.0` in the `decide` call inside `warm_up` survives. So does hardcoding `wasResidentBefore=True`, because the only success test passes `True`.
  - The warm-up is the call that carries the cold-load budget (`firstCallTimeoutSeconds`, §3.6, about 2.5 s cold on Laya). A constant there would time out a cold JIT load, or misreport a cold load as warm.
  - The timeout gap is pre-existing on the embeddings warm-up: the same mutant there survives the full suite.
  - **Fix (`tests/test_lmstudio.py`):** have the opener in `test_warm_up_on_decisions_surface_makes_one_decide_call…` assert `timeout == 300.0`, as the `decide` test already does with 7.5, and parametrize `was_resident_before` over `{True, False}`, asserting it round-trips.
- **U1a-m4 (minor). The HISTORY claim "residency … broken … each reddens" is broader than the evidence.**
  - Per U1a-m3, the `wasResidentBefore=True` mutant survives, so this is at best true of the particular mutant tried.
  - **Fix:** after U1a-m3's test lands, the claim becomes true. Otherwise narrow the sentence in `model-bench/docs/HISTORY.md`'s U1a entry.
- **U1a-m5 (minor). The fixture-replacement obligation lives only in HISTORY and the fixture itself, not on the queued probe.**
  - Ledger row "U1a-probe live `/api/v0` probe" (`docs/plans/typed-decision-models-coordination.md:20`) frames the probe as a route question about `/api/v0`.
  - If that route does not exist, nothing on the queued row says a raw **`/v1`** body must still be captured into `decisions_response.json`. `test_live_decide_predicate_and_choice…` calls `decide` on `/v1` but saves nothing.
  - **Fix (`docs/plans/typed-decision-models-coordination.md`, `teco`'s ledger):** the probe row's done-condition includes "capture the raw body of the route `decide` will use into `model-bench/tests/fixtures/lmstudio/decisions_response.json` and drop `_provenance`". The `_provenance` text and HISTORY are clear enough for a reader. The gap is that no actor's queue carries the step.
- **U1a-n1 (nit).** `decide` refuses a `score` answer only **after** the POST. That is deviation 4, see below. A candidate whose question is `score` would therefore turn every item into `unrunnable`, not refuse up front. Nothing in the plan uses `score`, so this belongs to U1c's `Candidate.load` and U1d's answer-space check, which should refuse a `score` question at load. See Open question 2.

**Deviations**

- **1. Synthetic fixture values: acceptable.** No verbatim body exists to copy (I confirmed this in the probe document), and the shape matches §2.3. The only gap is U1a-m5.
- **3. `warm_up(question=None)` default: acceptable.** `(question is None) == (call_surface == "decisions")` refuses both mismatches, and both directions are tested. The default can never select anything, so the "nothing that shapes a decision carries a default" invariant is not breached. The alternative, a required keyword on all three surfaces, would have forced edits to every existing runner call. That is U1d's territory.
- **4. `decide` refuses `score` answers: acceptable.** It fits plan §6 R-1: no probe has recorded a `score` answer's shape, and no planned decision point uses `score`. The refusal message names the type, and the test asserts that. The one hole is U1a-n1.

### Docs (HISTORY.md, AGENTS.md)

- **HISTORY entries.** There is one per unit, newest first, appended (correct for a by-lookup log). The figures I could check all match:
  - 95 records;
  - 85 × 17 = 1445 cases and the 15-reply corpus;
  - 33/45 fields and the 19 new names;
  - 7 grid cells;
  - the baseline of 1817, which I re-ran;
  - the per-unit test-count deltas, which I reproduced by per-file collection;
  - the final count, 2074/4/2.
  The one overstatement is U1a-m4.
- **The `AGENTS.md` edits were rewritten in place, not appended.** The profile list on line 37–40 replaced the old three-profile parenthetical. The live-test count `3` → `4` was edited in the same line, and I counted 4 `mark.live` tests. The remaining stale text is U1b-n1.
- **"Current state" (`model-bench/AGENTS.md:12`) still describes only the closed SMB feature.** It is not wrong. But `decide` and two profiles now exist with no runner path that produces them, so `run --candidate` does not exist yet. U3c owns the AGENTS invariants for this feature (plan §4), so I raise no finding. One clause there would stop a reader assuming the arm works.

### What's solid

- **The U3a proof is the strongest kind available:** whole-tree baseline, `repr` rather than JSON (so `True` ≠ `1`), per-variant and overall digests, and a corpus that the implementer grew after a surviving mutant (`keyless-beside-key`).
- **The U1b profiles are derived, not transcribed** (`{**embeddings, **adds}`). The pinning tests transcribe them by hand, so a drift in either one reddens. The forbidden sets are pinned by value per profile.
- **`_parse_decision_answers` is strict in the right places:**
  - type agreement with the question asked;
  - declared choice values, on both `choice` and every `probabilities[].value`;
  - exactly-once answering;
  - `bool`/`NaN`/`inf` refused as probabilities;
  - one `LMStudioCallFailed(status=None)` raise site, added to the status-scenario completeness pin.
- **The live probe was not run without approval,** and that is stated in four places.

### Open questions

1. **Who adds `decided_by`/`probability` to `classification.score_answer`, U1d or U2?** Plan §4 U1d declares the Protocol hook and says U3a's functions satisfy it. HISTORY defers to "whichever unit first writes `detail.decidedBy`" (U2). My recommendation: put it in **U1d's brief**, together with a test that the old pack's path still writes no `decidedBy`. The U3a golden test already guards that.
2. **Should a `score` question be refused at candidate load (U1c `Candidate.load`) or at the U1d answer-space check?** Either works. It just needs to be named in one brief so U1a-n1 doesn't fall between units.
