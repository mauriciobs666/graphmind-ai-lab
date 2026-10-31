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

## Pass 2

*2026-10-31. The Pass 1 fix round and U1c, reviewed against the uncommitted `model-bench/` diff from
`825d5d93`.*

### Verdict

| Delivery | Verdict |
|---|---|
| Fix round (U1b-n1/n2, U1a-m1…m4) | **approve with suggestions**: one residual gap, P2-m1 |
| U1c, `decisions.py` + vectors | **approve with suggestions**: P2-m2, P2-n1…n3; no blocker or major |

**Suite (run from `model-bench/`):** `2294 passed, 4 deselected, 2 xfailed`, and `ruff` is clean.
- The total is 2091 after the fix round plus the 203 tests in `test_decisions.py`. That matches HISTORY.
- The fix round's 17 new `test_lmstudio` tests (3+3+8+2+1) account for the step from 2074 to 2091.

**Mutants:** every mutant ran on a copy that was then restored. Each restore was checked with `cmp`.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded, and the diff is small enough to read and execute directly.

### Pass 1 dispositions

- **U1b-n1: fixed.** "three profiles" became "the profiles" at both `model-bench/AGENTS.md:46` and `fingerprint.py:233`.
- **U1b-n2: fixed.** The paragraph was rewrapped, and no line in it is longer than 100 characters any more.
- **U1a-m1: fixed.**
  - The requested-model mutant now reddens 3 tests on `decide` and 3 on `embed`.
  - The embed twin, which I had not asked for, is also fixed. Its mutant reddens 3 tests.
- **U1a-m2: fixed.** Making either bound of `[0, 1]` open reddens 4 tests each.
  - The rows cover predicate `probability`, choice `confidence`, and `probabilities[]` at both bounds.
  - JSON integers `0`/`1` are also covered, and they read back as `float`.
- **U1a-m3: partly fixed (see P2-m1).**
  - `wasResidentBefore=True` is now caught.
  - A hardcoded `timeout_s=7.0` is caught on both the decisions warm-up and its embeddings twin.
  - A hardcoded `timeout_s=300.0` survives on both.
- **U1a-m4: not fixed as stated.** The rewritten HISTORY sentence now says that a warm-up's "`timeout_s` hardcoded … reddens". Per P2-m1, that is true for 7.0 and false for 300.0.
- **U1a-m5:** deferred to U1e (plan2 §4.2), as ledgered. Not reviewed here.
- **Open question 1 (who adds `decided_by`/`probability`):** answered by plan2 §3.3 / §4.1. U1d adds them, together with `failed_call`, with no defaults, and with the old-pack `decidedBy` test.
- **Open question 2 (where `score` is refused):** answered by plan2 §3.2. `Candidate.load` refuses it, and U1c delivers that; see below.

**P2-m1 (minor). The warm-up timeout pins use the default value, so they cannot catch the default being hardcoded.**
- **The mutant:** `timeout_s=300.0` hardcoded in the decisions warm-up's `decide` call, or in the embeddings warm-up's `embed` call.
- **It survives** all 177 `test_lmstudio.py` tests in both places.
- **Why:** 300.0 is `RunConfig.firstCallTimeoutSeconds`'s default (`runner.py:129`, `cli.py:528`). Both new tests pass `timeout_s=300.0` and assert `timeouts == [300.0]`.
- **Why it matters:** hardcoding the default is the realistic form of this regression. It would silently ignore an operator's `--first-call-timeout`.
- **Fix (`model-bench/tests/test_lmstudio.py`):** use a value no code path defaults to, such as `123.5`, in `test_warm_up_on_decisions_surface_passes_its_budget_and_residency_through` and `test_warm_up_on_embeddings_surface_passes_its_budget_through`.
- **Then fix the HISTORY claim:** the U1a entry's "`timeout_s` hardcoded … each reddens" in `model-bench/docs/HISTORY.md` becomes true once the test changes. Until then, narrow it.

### U1c — `decisions.py` and the first golden vectors

**Verdict: approve with suggestions.**

**Spec coverage.** I checked v1.3 §4 U1c/§5 and plan2 §3.2/§4.1/§5 row by row. Everything is present:
- `Candidate.load` with its content hash. It refuses a `score` question, an ill-typed `decisionTimeoutSeconds` (NaN, ±inf, `bool`, ≤ 0), an unknown `fit.outcome`, a `fit.qualityTarget` belonging to the other bar kind, and a missing `fit`.
- `apply_policy`.
- `input_hash` and `eval_set_hash`.
- `INPUT_FIELDS_BY_ROLE`, `BAR_KIND_BY_ROLE` and `HANDOFF_FAILURE_BY_ROLE`, each with only the `guard-judge` row. The keys of `HANDOFF_FAILURE_BY_ROLE` are pinned to those of `BAR_KIND_BY_ROLE`.
- `DECISION_POINT_CHAT_TIMEOUT_SECONDS = 180`.
- `bar_holds` and the §5 boundary pairs.

**The requirement constants match the requirements.** `docs/requirements/typed-decision-models.md` gives FR-2 "at least 2× faster", FR-10 "at least 90%" and "350 ms or less", and FR-11 "at least 200 items".

**The golden vectors were derived independently of the code.** I recomputed them by hand, without importing `decisions`:
- **`input-hash.jsonl`, 3 of the 4 rows.** I hand-wrote each row's canonical JSON (keys sorted, no whitespace, non-ASCII kept raw) and hashed it with `printf … | sha256sum`. The results equal the file's `expected` values for `876a4957…` (the understanding path, which the field-reorder row shares), `a479c6c6…` (turns, non-ASCII) and `cb0492ac…` (absent field omitted).
- **`eval_set_hash`.** The test's `_EVAL_SET_HASH` `3af02801…` is reproduced by `sha256sum` over the three sorted lines, each ending in `\n`.
- **`apply-policy.jsonl`, all 22 rows.** I checked each one against v1.3 §3.3's inequalities. The rows I read closely:
  - `l==0, p==0` gives hand-off, because 0 is not below 0;
  - an empty band at 0 gives *yes* for every p, because `p ≥ h = 0`;
  - an empty band at 1 with `p=.999` gives *no*;
  - `c == 0` with confidence 0 gives the choice.
  All of them are correct. The plan's five named boundaries (`p == l`, `p == h`, `l == h`, `confidence == c`, `c == 0`) are present and asserted.
- **Mutants of the vector-pinned code redden.** Removing `ensure_ascii=False` reddens 4 tests. Changing `>=` to `>` on confidence reddens 3. Dropping the trailing `\n` in `eval_set_hash` reddens 3. The value move in `HANDOFF_FAILURE_BY_ROLE` reddens 1. Making the empty-band rule one-way reddens 2.

**The implementer's open calls**

- **`eval_set_hash` separator, `\n` after every line: sound.** The plan's formula does not fix a separator. This one is pinned by a hand-derived value, and HISTORY records the choice.
  - The hash is a **stored-data** format: it goes into every candidate fingerprint, and K counts compare it across runs. Its single home should be the plan (v1.3 §3.5 or plan2), not HISTORY alone; see Open question 2.
  - It constrains no other component. Only model-bench computes `evalSetHash` (v1.3 §3.5), and decision-models copies only `input_hash`.
- **`PolicyDecision(handoff, answer)` as the vector `expected` shape: acceptable.** U9 and U5/U6 can read it directly. A choice label can never be confused with a hand-off, because `handoff` is a separate field. The one weakness is P2-n2.
- **The extra `Candidate.load` checks: sound.**
  - Requiring every field, with `handoffModel` as an explicit `null`, matches plan2 §3.2's "a missing … field" and turns a typo in a required key into a refusal.
  - Allowing unknown keys is safe: every key the code reads is required, a typo cannot be silently skipped, and `contentHash` covers any extra key anyway.
  - **None of this constrains U1d/U2/U5 wrongly.** U6's `fit.py` must write `provenance` and `fitManifest` in full. That is what v1.3 §3.3's sample already shows.
  - **One constraint may need revisiting because of the over-limit ruling.** See the 512-token paragraph below.
- **`bar_holds` raising on non-finite or `bool` input: right.** It makes U3c handle an empty or undefined measurement as `not assessable` before it reaches a bar, instead of letting NaN compare `False` into a silent "not met". It does not constrain U3c wrongly, because plan2 §3.6 already routes those states to `not assessable`.
- **The `HANDOFF_FAILURE_BY_ROLE` value pin ahead of U2: harmless.** It is a transcription now, and U2's withheld-versus-kept test adds the behavioural pin. U4 must update it when it adds the two `fail-closed` rows, which plan2 §4.1 U4 already lists.
- **The claimed equivalent mutant (yes/no branch order): true only under a precondition the function does not check.** I ran it: all 203 tests stay green. The two orders differ only when `noBelow > yesAtOrAbove`, and `apply_policy` accepts that silently (P2-m2).

**What the 512-token over-limit ruling means for U1c** (`ws:agent-team` `480ef825…`: HTTP 500 above 512 tokens per question; the stakeholder rules that over-limit inputs hand off):
- **`apply_policy`'s contract does not need to change.** It maps an *answer* to a decision. An over-limit call has no answer, so routing it to hand-off happens before `apply_policy`: U1d/U2's call handling, and U9's client.
  - If that routing must be byte-equal across the three copies, it needs its own vector family, for example an error-classification table.
  - That vector family belongs to U1d/U9, not to a widened `apply_policy`.
- **One U1c rule *is* affected: the "empty band iff `handoffModel` is null" rule in `_policy_problem`.**
  - Under the ruling, a candidate whose policy never hands off on confidence still needs a hand-off model for over-limit inputs.
  - Today that combination (empty band plus a non-null `handoffModel`) is refused (`test_candidate_load_refuses_an_empty_band_with_a_hand_off_model`).
  - A decision-model-only candidate (`handoffModel: null`) cannot honour the ruling at all, so its over-limit items would have to stay kept failures or be withheld.
- **This is a design decision for `architect` (Open question 1), not a U1c defect.** The code implements v1.3 §3.3 as written. If the rule relaxes to one direction (a non-empty band requires a hand-off model), the fix is a one-line change in `_policy_problem` plus a flip of that test.

**Findings**

- **P2-m2 (minor). `apply_policy` gives order-dependent results on a band it never validates.**
  - With `yesAtOrAbove=0.4, noBelow=0.6` and `p=0.5`, it returns *yes*. The swapped branch order returns *no*. I executed both.
  - `Candidate.load` refuses such a band, but three copies will exist: falkor-chat's U9 copy reads its own config, and decision-models' `fit.py` searches (l, h) pairs.
  - So the "equivalent mutant" is equivalence by convention, and two correct-looking copies can disagree on input that no vector covers.
  - **Fix (`model-bench/modelbench/decisions.py`, `model-bench/tests/test_decisions.py`):** have `apply_policy` raise `ValueError` itself unless `0 ≤ noBelow ≤ yesAtOrAbove ≤ 1` (and `0 ≤ minConfidence ≤ 1`), and add a refusal test. The branch-order mutant then becomes equivalent by construction.
  - Whether the refusal also becomes a vector row (an `expected: {"error": …}` shape) is the implementer's call. Without it, the copies are held only by each one's own suite.
- **P2-n1 (nit). Two validation regexes accept a trailing newline.**
  - `_SHA256_RE` and `_CANDIDATE_ID_RE` use `$`, which matches before a trailing `\n`. I executed both: `"0"*64 + "\n"` passes as a sha256, and `"guard\n"` passes as a candidate id.
  - **Fix (`model-bench/modelbench/decisions.py`, `model-bench/tests/test_decisions.py`):** use `.fullmatch` or `\Z`, and add one `"…\n"` row to each refusal test.
- **P2-n2 (nit). `PolicyDecision` can represent a state that should not exist.** `PolicyDecision(handoff=False, answer=None)` can be constructed, so the invariant "`answer is None` iff `handoff`" lives only in a docstring.
  - **Fix (`model-bench/modelbench/decisions.py`, `model-bench/tests/test_decisions.py`):** a `__post_init__` check. It would also give U9's and decision-models' copies a stated rule to transcribe.
- **P2-n3 (nit). The fix-round result was written into the delivered U1a HISTORY entry.** The entry's figures and mutant list were rewritten in place, rather than recorded as a separate dated line. A by-lookup log reads more cleanly with an entry per change.
  - **Fix (`model-bench/docs/HISTORY.md`):** this is optional, since the edit is accurate apart from P2-m1. Next time, record a fix round as its own dated line.

**Docs.**
- The HISTORY U1c entry is accurate: 22 and 4 rows, hand-derived vectors (which I confirmed), the mutant list (the equivalent-mutant claim is qualified by P2-m2), and `2294/4/2`.
- The `model-bench/AGENTS.md` "Current state" sentence that adds `decisions` is a rewrite in place. "No `run` path uses it yet" is true now.

**What's solid.**
- The vectors really are independent of the code.
- Each closed set is pinned by a transcription.
- The JSON-type traps are guarded: NaN tokens, `bool` as a number, and a list or object where a closed-set string belongs (unhashable).
- `CandidateError` is a `ValueError`, ready for exit 4.

### Open questions

1. **(architect) Over-limit hand-off versus "empty band iff no hand-off model".** Under the 512-token ruling, should `Candidate.load` accept an empty band with a hand-off model (an overflow-only hand-off)? And what is a decision-model-only candidate's over-limit item: kept failure, withheld, or refused? This is the one U1c contract the ruling touches. It should be settled before U1d builds on `_policy_problem`.
2. **(architect) Which document owns the `eval_set_hash` line format?** It is now a stored-data format, and HISTORY is a record, not a spec. I recommend one sentence in plan2's next revision, citing the separator.

## Pass 3

*2026-10-31. The Pass 2 fix round, reviewed against the uncommitted `model-bench/` diff from `da877b2a`.*

**Verdict: approve.** Every Pass 2 finding I raised is fixed, and each fix was confirmed by running it. There are no new findings.

**Checks run:**
- **Suite:** `2311 passed, 4 deselected, 2 xfailed`, run from `model-bench/`. That is 2294 plus 17 new `test_decisions.py` tests. `ruff` is clean.
- **Mutants:** each was run on a copy, then restored and checked with `cmp`.
- **Fixtures:** `tests/fixtures/` has no diff.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded, and the change is 4 files.

### Dispositions

- **P2-m1: fixed.** A hardcoded `timeout_s=300.0` now reddens on both warm-ups: 2 tests on decisions and 1 on embeddings. `7.0` still reddens 2.
  - The U1a HISTORY claim ("`timeout_s` hardcoded … reddens") is now true. That closes **U1a-m4** as well.
- **P2-m2: fixed.**
  - `apply_policy` now calls `_policy_bounds_problem` before either branch runs.
  - Removing that call reddens 11 tests.
  - Dropping `low <= high` from the shared helper reddens 2 tests: one through `apply_policy`, one through `Candidate.load`.
  - `Candidate.load` still refuses the same bands, because it calls the same helper.
- **P2-n1: fixed.** Both regexes now use `fullmatch`. Reverting either one to a `$`-anchored `match` reddens its new `"…\n"` row (1 test each).
- **P2-n2: fixed.** Removing `PolicyDecision.__post_init__` reddens 4 tests. A one-way version (rejecting only a hand-off that carries an answer) reddens the `decided-with-no-answer` row. All 22 `apply-policy.jsonl` rows still construct.
- **P2-n3: not taken.** It was optional. Both new HISTORY items are one dated line each, as the convention asks.

### Claim 1: the branch-swap mutant is now equivalent. Confirmed.

- **By execution:** I swapped the order of the yes and no branches. All 220 `test_decisions.py` tests stay green.
- **By exhaustive comparison:** I ran `apply_policy` over 169 (l, h) pairs that include invalid bounds, NaN and `bool`, crossed with 16 values of p, including p outside [0, 1], NaN, ±inf and `bool`. Of these 2,704 inputs, 1,952 were refused by the bounds check. The remaining 752 all agree with an independently written version that checks *no* first.
- **The argument:** the two orders can differ only when both `p >= h` and `p < l` hold, which requires `l > h`. The bounds check refuses `l > h` before either branch runs.
- **This holds even for a p that `decide` would never pass**, such as NaN. With NaN, both comparisons are false, so both orders give a hand-off.

### Claim 2: no error row in `apply-policy.jsonl`. The call is right for U1c.

- **The plan:** plan2 §4.1 keeps the rows unchanged.
- **The shape:** an `expected: {"error": …}` row would give the falkor-chat and decision-models copies a second shape to parse inside one file.
- **What is left uncovered is narrow.** A copy that leaves out the bounds refusal stays order-independent as long as it is never handed `l > h`. Any band that is measured passes through `Candidate.load`, which refuses `l > h`. So the risk is confined to inputs that are never measured: a hand-edited falkor-chat `models.json`, or a `fit.py` search that tries `l > h`.
- **If the three copies must agree on refusals too,** the right vehicle is a separate family file, for example `decision-contract/policy-bounds.jsonl` (policy → `refused: true|false`). That leaves `apply-policy.jsonl`'s shape and rows untouched. It belongs in U5's and U9's briefs, not in U1c. See the open question.

### Open question

1. **(architect / teco) Should U5 and U9 pin the bounds refusal in their own copies?** The options are each copy's own suite, or a shared `policy-bounds.jsonl` family. Either way, it needs to be named in those briefs so the refusal is transcribed rather than dropped.

## Pass 4

*2026-10-31. U1d.1, the contracts half of U1d, reviewed against the uncommitted `model-bench/` diff
from `3f01faaa`. Baseline: plan3 v1.2 §3.2, §3.3, §4.1 U1d, §5 U1d; plan2 v1.1 §3.2, §3.3, §4.1
U1d, §5 U1d. The runner, pack, CLI and Protocol-hook halves (U1d.2/U1d.3) are out of scope.*

**Verdict: approve with suggestions.** One minor finding (P4-m1), no blocker or major. All five
judgment calls hold; call 5 exposes a real plan3 seam, routed as an open question.

**Checks run** (from `model-bench/`):
- **Suite:** `2381 passed, 4 deselected, 2 xfailed`; `ruff` clean.
  `test_every_stored_run_record_still_validates_unchanged` and the U3a golden
  `test_score_item_rescores_the_real_pack_byte_identically_to_before_the_split` both pass.
- **Body row:** `over-limit.jsonl`'s `body-document-at-500` body is byte-equal to the text between
  the markers of `ws:agent-team` `a1d388f4…`: 232 UTF-8 bytes, sha256 `5a93988e…`. The `/api/v0`
  row equals probe document `480ef825…`'s HTTP-200 body verbatim.
- **Mutants**, each run on the file in place, then restored and checked with `cmp` (and the four
  source files' md5 afterwards). Implementer's claims, all red: case-folded body (1), status check
  dropped (1), one-way rule reverted to "iff" (4), `body=None` on a failed error-body read (1),
  body truncated (1), `fitQualityTarget` relaxed to `present` (4), `reason` not written (2). My own
  argument-level mutants, all red: the 2xx non-JSON site and the malformed-decisions site passing
  a body instead of `None` (1 each, via the raise-site table), `errors="surrogateescape"` (1),
  marker shortened to `"too large"` (1).
- **HISTORY:** the U1d.1 entry's figures (8 rows, 232 bytes, 35/47 fields) match the fixture and
  the pinned schemas; it states no suite count. Its mutant list matches what I ran.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded, and the diff is 13 files read
and executed directly.

### Judgment calls

1. **`c == 0` with a hand-off model now accepted: true to plan.** Plan3 §3.2 defines "can hand off"
   as "a non-empty band, **or `minConfidence > 0`**", so `c == 0` cannot hand off on policy and is
   the choice-side overflow-only composite. U9 applies "the same one-way rule", so no copy diverges.
2. **Extra rows `{500, null}` and `{null, null}`: sound.** Plan3's "`null`" names no status; both
   readings are covered, and `{null, null}` is every non-HTTP raise site's real shape. The plan
   gives falkor-chat "the same signature" (`int | None`), so U9's copy can take both rows.
3. **`score_answer`'s refusals: true, and they constrain nothing wrongly.** Only
   `guard-judge-understanding` uses `classification`, and `HANDOFF_FAILURE_BY_ROLE["guard-judge"]`
   is `withhold`, so `failed_call="handoff"` cannot legitimately reach it. U4's `agreement.py`
   needs the opposite (a kept hand-off failure is legal there), so it must not copy
   `_decision_detail` verbatim; that is U4's brief, not a defect here. The refusal set is
   incomplete, though: P4-m1.
4. **`probability: None` on failed items: consistent with plan3.** Plan3 §3.3's table says every
   over-limit row, failed ones included, has `detail.probability = None`; writing it on every
   failed item gives one decision-surface shape. U4 should write the same shape.
5. **The guard "hand-off fails, withhold" row cannot go through `score_answer`: a correct reading,
   and a real seam in plan3.** That column requires `detail.reason = "over-limit"` on a withheld
   item. `score_answer` needs an answer or a `failed_call` (both absent there), and the withheld
   path, `score_item(result=None)`, builds an `ItemResult` with an empty `detail` and has no
   `reason` parameter. Plan3 §3.3's "Scorer hook delta" names only `score_answer`, so no vehicle is
   specified. §3.6 counts this item in `overLimit` (k and n), so the counter will read a `reason`
   no specified path writes. See Open question 1.

### Findings

**P4-m1 (minor). `score_answer` refuses some impossible keyword states but not the over-limit
ones.**
- **Evidence (executed):** over the 72 combinations of `answer` × `decided_by` × `probability` ×
  `failed_call` × `reason`, against an independent legality predicate written from plan2 §3.3 and
  plan3 §3.3's table, 6 illegal states are accepted. None is over-refused. Three contradict the
  plan directly:
  - `decided_by="decision-model"` with `reason="over-limit"` (an over-limit call has no decision);
  - `decided_by="handoff"`, `reason="over-limit"`, `probability=0.5` (plan3: "Every row has
    `detail.probability = None`");
  - `reason="over-limit"` with neither `decided_by` nor `failed_call` (an answer with no source).
  Three are my inference: a `decided_by` item with `probability=None` but no over-limit reason
  (two states for `decision-model`, one for policy `handoff`). The guard's question is a
  predicate, so a decided item always has a `p`.
- **Why it matters:** these are exactly the states a U1d.2/U2 runner bug produces: a stale `p`
  carried into the over-limit branch, or the wrong `decided_by`. `handoffShare`, `overLimit` and
  `pair_items` would then disagree about the same item, silently. The implementer's own design
  is to refuse impossible states at this seam; the over-limit axis is the one it misses.
- **Fix (`model-bench/modelbench/scoring/classification.py` `_decision_detail`;
  `model-bench/tests/test_scoring_classification.py`):** refuse `reason` without a `decided_by` or
  `failed_call`, `reason` with `decided_by="decision-model"`, and `reason` with a non-`None`
  `probability`. Decide the other three explicitly; I recommend refusing too. Pin them with a
  **cross-product probe**, not a list of shapes: a test that enumerates all 72 combinations and
  compares accept/refuse against a hand-written legality predicate. Land the fix before U1d.2 wires
  the runner to it.

**Stopping condition for P4-m1:** that probe exists, passes, and goes red when any one of the new
refusals is removed.

### What's solid

- The one-way rule is a minimal condition change. Both former refusal tests became acceptance
  tests, the band one at three edges.
- `body` is required at all 13 raise sites, and the raise-site table pins every value, so a new
  site cannot inherit a default.
- The over-limit vectors were derived independently, and each named rule (status, case, marker)
  has its own reddening row.
- The old pack's `detail` stays byte-identical, proven by the golden re-score and a test over
  every real item.

### Open questions

1. **(architect, before U2) What writes the guard's withheld over-limit record?** Plan3 §3.3's
   withhold column needs `detail.reason` (and, for the per-cell `overLimit` prints of §3.6, likely
   `path`/`tier`) on an item that never reaches `score_answer`. Two options:
   - a `reason` parameter on the scorer's withheld branch;
   - the runner rebuilding the frozen `ItemResult` with `dataclasses.replace`.
   Name one in U2's brief; otherwise U2 has to invent it.
2. **(U1d.2 heads-up, no finding) `detail.inputTokens` has no vehicle either.** `ItemResult` is
   frozen, and `score_answer` builds `detail`. Adding a fifth no-default keyword, `input_tokens`,
   keeps the "every decision-surface key passes through one seam" shape that P4-m1's probe relies
   on.

## Pass 5

*2026-10-31. The P4-m1 fix, reviewed against the uncommitted `model-bench/` diff from `f3b853d2`
(`classification.py`, `test_scoring_classification.py`, `HISTORY.md`).*

**Verdict: approve with suggestions.** P4-m1 is fixed and its stopping condition is met. One new
minor finding, P5-m1, sits on the axis the fix added.

**Checks run** (from `model-bench/`): suite `2389 passed, 4 deselected, 2 xfailed`; `ruff` clean.
The U3a golden re-score and `test_every_stored_run_record_still_validates_unchanged` pass.
Mutants were run on the file, then restored and checked with `cmp`.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded, and the change is one
function and its tests.

### Dispositions

- **P4-m1: fixed.** I removed six of the 18 refusals one at a time, and each reddens
  `test_score_answer_accepts_exactly_the_legal_keyword_combinations`:
  - over-limit with `decision-model`;
  - over-limit with a `probability`;
  - over-limit with no source;
  - a decided item with no `probability`;
  - a failed item with `input_tokens`;
  - a withheld item with no `withheldFor` (2 tests).
  Writing `inputTokens` as `None` also reddens.
- **`_legal` is not a mirror of the code.** Its branch order follows `_decision_detail`, but each
  clause cites a plan rule, and it holds up two ways:
  - **My Pass 4 predicate**, run unchanged over the original 72 combinations (with
    `input_tokens`/`withheldFor` `None`), agrees with the code on all 72.
  - **Six combinations I derived by hand** all come out as the plans require. Accepted: a policy
    hand-off with `p` and tokens; a kept over-limit decision failure; a decided item withheld for
    `"load"`. Refused: an over-limit hand-off with tokens; a kept failure with tokens; a withheld
    over-limit item with `decided_by`.
- **Open question 1 → the architect's ruling, one legal case added. Sound.** The withheld
  over-limit record now passes through the one seam that writes decision keys, so the runner never
  patches a frozen `ItemResult`. The other route was `dataclasses.replace` in the runner. Both
  options I named in Pass 4 work; this one keeps the legality check in one place.
- **Open question 2 → `input_tokens` is a fifth no-default keyword. Sound.** It is written beside
  `probability`, and the old pack writes no `inputTokens` key, which the existing dedicated test
  now checks.
- **Pass 4's three inferred refusals were adopted, on good grounds.** v1.3 §3.3 defines
  `probability` as "a predicate's p, or a choice's confidence", which every returned decision
  carries. So only an over-limit hand-off is decided with no `p`.
- **HISTORY:** one dated line, and its figures (288 combinations, 2389 tests) match.

### Findings

**P5-m1 (minor). The new `withheldFor` axis is checked for one state only.**
- **Evidence (executed):** `score_answer` accepts all of these:
  - `failed_call="decision"` with `withheldFor="timeout"` or `"no_response"`;
  - an answered decided item, or the old pack's answer, with `"timeout"`/`"no_response"`;
  - the withheld over-limit item with `withheldFor="load"`.
- **What the plans allow:** plan2 §3.3 gives a kept item `withheldFor=None` unless residency
  withholds it for `"load"`. Plan3 §3.3's withhold column gives `"timeout"` or `"no_response"`. A
  failing call wins over the residency guard (`case_f`), so a no-answer item is never `"load"`.
- **Why the probe misses it:** its withheld dimension is `{None, "timeout"}`, and `_legal` ignores
  that dimension except in the withheld over-limit state.
  `test_a_withheld_over_limit_guard_item_carries_only_its_reason[load]` pins a state that plan3
  does not have.
- **Why it matters:** "decision timeout → withheld, as today" is exactly the behaviour plan2 §3.3
  changed. A U1d.2 runner that keeps the old timing would build a kept failure that
  `runner.py:603` also counts in `latencyWithheldForNoResponse`, and nothing would refuse it.
- **Fix (`model-bench/modelbench/scoring/classification.py` `_decision_detail`;
  `model-bench/tests/test_scoring_classification.py`):**
  - `"timeout"`/`"no_response"` are legal only in the withheld over-limit state;
  - `"load"` is legal only with an answer or a kept decision failure;
  - widen the probe's withheld dimension to `{None, "load", "timeout", "no_response"}` (576
    combinations), with `_legal` encoding both rules;
  - drop `"load"` from the dedicated test's parametrize.

**Stopping condition:** the widened probe passes, and removing either new `withheldFor` clause
reddens it.

**Timing:** land P5-m1 before U1d.2 builds the kept decision-call failure on top of this seam.

## Pass 6

*2026-10-31. U1d.2, the plain `decisions` arm, reviewed against the uncommitted `model-bench/` diff
from `c0eeb348`. Baseline: v1.3 §3.3 and §4 U1d, with plan2/plan3 where they amend them. U1d.3's
parts and U2's are out of scope. The repo-root `live-specs*`/`standard-local-model*` files were not
reviewed.*

**Verdict: approve with suggestions.** Two minor findings, one nit; no blocker or major.

**Checks run** (from `model-bench/`):
- **Suite and lint:** `2440 passed, 4 deselected, 2 xfailed`; `ruff` clean. The done-condition
  test, `test_a_fixture_candidate_run_stores_a_valid_model_decisions_record_offline`, passes.
- **Mutants:** each was run on a copy that was then restored and checked with `cmp`.
  - **Claimed, all red (1 test each):** the role check dropped; the `type != "decision"` check
    dropped; the hand-off-model refusal dropped; the `decision.handoff` guard dropped; the decide
    timeout hardcoded to `120.0`.
  - **Mine, red:**
    - `questionSpecHash` hashed over the name alone (1);
    - the residency probe frozen at `baseline_residency` (1);
    - the warm-up sent a substitute question (1).
  - **Mine, survived:** `ensure_ascii=False` removed from `question_spec_hash`. All 2440 tests
    stay green (P6-m1).
- **The refusal-message-only kill is adequate, and necessary.** With the role check dropped, a
  guard candidate on an `nlq-generator` pack still refuses with exit 4 before any call. It falls
  through to `eval_set_hash` ("no input fields are defined for role 'nlq-generator'"), and only
  the message match reddens the test.
  - That fallback holds only because `INPUT_FIELDS_BY_ROLE` has a single row. Once U4 adds
    `join-in-judge`, a guard candidate on a join-in pack would pass the hash, the
    predicate-versus-predicate answer-space check and the hooks check, and then run.
  - So the message pin is the only thing that tells which guard fired.
  - When U4 lands, its brief should add a behavioural row: a guard candidate on a join-in pack
    refuses.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Interim exit 3 on a failed decision call: safe

- **It fails closed.** `RunRefused` is raised before `store`, so nothing is written and no partial
  record can carry a verdict. It also cannot reach `score_answer` with a call-failure
  `withheldFor` (illegal since P5-m1).
- **The cost is availability only.** One transient failure, or one over-limit item, loses the
  whole run. No live candidate run is due before U7, and U7 needs U1d.3 and U2.
- **It is a clean seam for U1d.3.** The `except` block already has `started` and the over-limit
  classification, so U1d.3 replaces one raise with a kept item and touches nothing else.

### The implementer's calls

1. **The `questionSpecHash` formula: a seam, and it is half-pinned.** v1.3 §3.3 names the field
   but gives no formula, and the hash is stored data that U3c's replicate grouping will compare
   across runs. Plan3 §3.8 is the natural home: it is where `eval_set_hash`'s line format went.
   Only part of the formula is pinned (P6-m1).
2. **Ignoring the response `model`: acceptable.** No surface reads it: chat and embeddings parse
   `model` and never use it either. Identity comes from the catalog entry, the request names
   `model_info.id`, and the residency probe withholds an item whose model is not resident. Plan3
   §3.7's explicit load (U2) is where a served-model check would belong, if one is wanted.
3. **Declared `fitManifest*` values: acceptable for now. U2's scope needs one sentence (P6-m2).**
4. **`check_attestation_staleness` returns `"unavailable"` on `decisions`: true to plan.** v1.3
   §3.3 says "For `decisions` alone it is `"unavailable"`, as for embeddings".
5. **`compare --models` and `models --tested` on `armKey`: as the plan says.** `rank` on
   `modelKey` is a nit (P6-n1).
6. **`results/index.csv` rebuilt: right.** `INDEX_COLUMNS` changed, so a stale index would
   disagree with its own header. `rebuild_index` derives it, and teco verified the diff is only
   the appended cells.

### Findings

**P6-m1 (minor). Stored data: two of `question_spec_hash`'s stated rules are unpinned.**
- **Evidence (executed):** removing `ensure_ascii=False` survives the whole suite. The only
  external pin, `_QUESTION_CANONICAL` in `test_runner_decisions.py`, is a flat ASCII predicate.
  So neither "non-ASCII kept raw" nor nested key order (a choice's `choices` objects) is
  exercised, although the docstring claims both.
- **Why it matters:** a guard or message-type instruction containing an em dash or a
  Portuguese label would hash differently under a refactor nobody notices. Stored records would
  then split one candidate's replicates by hash.
- **Fix (`model-bench/tests/test_decisions.py`):** add hand-derived vectors, written as
  canonical JSON and hashed with `sha256sum` the way `input-hash.jsonl` was. Use a choice question
  whose `choices` objects have keys out of order, plus non-ASCII text in `instructions`.
- **Fix (`architect`):** state the formula in plan3's next revision, beside §3.8.
- **Stopping condition:** the `ensure_ascii` mutant and a nested `sort_keys` mutant each redden a
  test.

**P6-m2 (minor). The AC-3 leakage refusal is assigned to the hand-off arm's unit, but
decision-only candidates need it too.**
- **Evidence:** v1.3 §4 U2 ("hand-off arm") owns "the leakage refusal (AC-3) and the fingerprint
  `fitManifest*` fields". `_decision_fingerprint_fields` writes the declared values unchecked, and
  `_prepare_decision_arm` runs no leakage check.
- **Why it matters:** a U2 built to its brief's title would add the refusal on the `handoff` path
  only. Every `decisions`-surface record would then carry an unverified `fitManifestHash` and no
  AC-3 check, and U6 emits decision-only candidates whenever `handoffModel` is null.
- **Fix (U2 brief, `teco`):** have U2 put the manifest read, the hash/size check and the leakage
  refusal in `_prepare_decision_arm`, which both surfaces already share. Add a test row: a
  decision-only candidate with an overlapping manifest refuses with exit 4.

**P6-n1 (nit). `rank` still deduplicates on `modelKey`.**
- **The effect:** `_select_rank_arms` (`model-bench/modelbench/cli.py:248`) keeps one run per
  `modelKey`. Two candidates on `laya` would collapse silently to the newer one, and
  `rank_report`'s `DuplicateModelInReport` backstop cannot fire.
- **Why it is only a nit:** plan2 §5 U3c makes `rank` refuse on decision-point packs, and no
  real decision-point pack exists yet.
- **Fix:** key the dedup on `armKey`. It equals `modelKey` on every existing arm, so the change is
  free.

### What's solid

- **`RunConfig` refuses both and neither,** and argparse enforces the same exclusivity.
- **`_build_fingerprint` refuses `decision_fields` on any other surface,** so the
  `model:decisions` fields can neither go missing nor leak.
- **Every §3.3 refusal is checked before any model call,** and the tests assert that directly.
- **`decision_input` is a pure extraction:** byte-equal to the chat judge's user message on all
  85 real items.
- **The item clock and the load probe follow the existing single-call driver's pattern.**

## Pass 7

*2026-10-31. U1d.3, the run-time deltas, reviewed against the uncommitted `model-bench/` diff
from `5bc5bf4b`. Baseline: plan2 §3.2–§3.4, §3.6 and §4.1 U1d; plan3 §3.3, §3.6 and §4.1 U1d; the
method note's §Q6; Pass 6's P6-m1 and P6-n1.*

**Verdict: approve with suggestions.** One nit (P7-n1) and one note for U4's brief. No blocker,
major or minor.

**Checks run** (from `model-bench/`):
- **Suite:** run by teco; I re-ran `test_runner_decisions.py` and `test_decisions.py` under every
  mutant.
- **Mutants:** each was run on a copy that was then restored and checked with `cmp`.
  - **Claimed, all red:** the re-check run after a load-withheld item (1); choice equality
    ignoring `confidence` (1); an over-limit item counted in `decisionCallFailures` (2); the pin
    replaced by 120 (5); `question_spec_hash` without `ensure_ascii=False` (2, so P6-m1's
    surviving mutant now dies).
  - **Mine, argument-level, all red:** the re-check's input changed to `"warm-up."` (2); the
    re-check's timeout set to `300.0` instead of `decisionTimeoutSeconds` (2); the kept item's
    load probe read from `baseline_residency` (1).

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Dispositions

- **P6-m1: fixed.** The `ensure_ascii` mutant reddens 2 tests. teco reproduced both vectors with
  `sha256sum`.
- **P6-n1: fixed.** `_select_rank_arms` dedupes on `armKey`, and a test keeps two candidates on
  one decision model.

### Interaction check: the kept failure through `score_answer`

**Every state the runner can now produce is legal. Executed, not inferred.** I ran a four-item
run through `run_pack`: two decided items, then an over-limit 500 and a decide timeout, with
the last two items' residency snapshots empty.
- **Both kept failures came out legal.** Each had outcome `fail`, `withheldFor="load"` and
  `failedCall="decision"`; the first also carried `reason="over-limit"`. `score_answer`
  accepted both: P5-m1 allows `"load"` on a kept decision failure.
- **The load rule then behaved as specified.** It recorded `basis="assumed"` with no re-check
  call (exactly 4 `decide` calls).
- **The record round-trips.** `store` then `load_history` gives zero invalid records,
  `requestTimeoutSeconds=180.0`, and the `reason` survives.

The runner never builds a call-failure `withheldFor`: `kept` takes its value only from
`_load_withheld_for`, which returns `None` or `"load"`.

### The implementer's calls

1. **`requestTimeoutSeconds` defaults to `None`: true, and needed.** Without it an explicit 120
   cannot be told from no flag.
   - The only readers are the three drivers. They receive `cfg` after `run_pack` has replaced it
     with the resolved value.
   - Mapping `0` to `None` keeps the old `0 or 120` behaviour.
   - Every stored value is a resolved float, which is what U3c's equality backstop compares.
2. **Counts appended to `perClass`, with a `TypeError` when it is missing: acceptable.**
   - They are named metrics outside `verdictMetrics`, so the report renders them exploratory.
   - They are outside `_aggregate_item_mismatches`, which checks the verdict family only.
   - U2 extends the same function with `handoffShare`/`handoffCallFailures`.
   - **Constraint for U4:** `agreement.py`'s aggregates must carry a `perClass` tuple. That is a
     shape requirement nothing states, so name it in U4's brief.
3. **`n` is every item, load-withheld ones included: true to plan.** Plan2 §3.3 and plan3 §3.6
   say `n` is "attempted", and a load-withheld item did make its decision call.
4. **A kept failure's outcome is `"fail"`: sound.** Plan2 does not fix an outcome. `detail.failedCall`
   is what U3c's `pair_items` reads, and classification's `aggregate` counts only
   `parse_failure` outcomes.
5. **`basis` falls back to `assumed` on a raise: true for the re-check, half-true for the probe.**
   See P7-n1.
6. **`_same_decision` uses exact equality: correct.**
   - §Q6 says "a different `p`" gives `"assumed"`.
   - Plan2 §3.4 says "not exactly equal" (predicate `probability`; choice `choice` and
     `confidence`).
   - v1.3 §2.3 says Laya "repeats exactly" within a load.
   - A tolerance would hide the very reload the rule exists to detect. Leaving out
     `probabilities[]` matches plan2's field list.

### Findings

**P7-n1 (nit). The load-rule probe catches only `LMStudioUnreachable`/`TimeoutError`, not the
other errors `residency()` raises.**
- **Evidence (executed):** `residency()` calls `catalog()`, which raises `LMStudioCallFailed`
  on a non-200 GET or a malformed body. A stub raising it at the end probe escapes `run_pack`
  uncaught (`ESCAPED LMStudioCallFailed GET /api/v0/models: HTTP 500`). `cli` has no handler for
  it, so a run whose items all completed is lost to a traceback.
- **The overclaim:** the docstring ("a probe that could not run … is `"assumed"`") and the HISTORY
  entry ("a failed probe … give `assumed`") both claim more than the code does. The between-item
  probes and step 10 share the gap; that is pre-existing and out of scope.
- **Fix (`model-bench/modelbench/runner.py` `_within_load_basis`):** add `LMStudioCallFailed` to
  the probe's `except`. `"assumed"` is the fail-safe answer. Add one test row; a stub probe
  raising `LMStudioCallFailed` is enough. Otherwise narrow the docstring and HISTORY to the two
  caught types.

### What's solid

- **The exit-3 interim was replaced in place,** with the over-limit classification reused, as
  Pass 6 predicted.
- **The load rule follows §Q6's order exactly:** probe, then load-withheld, then re-check against
  the *mandatory* warm-up only.
- **The pin and its refusal apply up front,** before host read or any call, and to the chat arm
  on a decision-point pack too.
- **The counts are read from `detail`,** never re-derived from the call, so the counts and the
  per-item record cannot disagree.
