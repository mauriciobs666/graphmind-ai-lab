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

## Pass 8

*2026-10-31. U2.1, the shared load and preparation infrastructure, reviewed against the uncommitted
`model-bench/` diff from `763e95e4` plus the untracked `tests/fixtures/lmstudio/api_v1_load_probe.json`.
Baseline: plan3 v1.2 §3.7, §4.1 U2 (load bullets), §5; v1.3 §3.4–§3.5 and §4 U2 (fit manifest,
AC-3); plan2 v1.1 §3.5. The `handoff` surface (U2.2) and `model-bench/REVIEW.md` (`cfdaa1f3`) are
out of scope.*

**Verdict: needs changes.** One major (P8-M1), two minors, one nit. The major is small and is
cheapest now, before U2.2 inherits the field and before any decision record is stored.

**Checks run** (from `model-bench/`):
- **Suite and lint:** `2549 passed, 4 deselected, 2 xfailed`; `ruff` clean. Both match HISTORY, as
  does the new-test count (19 + 9 + 9 + 3 = 40, counted from the diff).
  `test_every_stored_run_record_still_validates_unchanged` passes. `git status` shows nothing
  under `results/`; the newest file in `results/runs/` dates from 2026-09-20.
- **Fixture against `ws:agent-team` a93905a6:** `load_laya_request`, `load_laya`, `load_qwen` and
  `laya_instance` are byte-identical to the probe document's bodies under compact
  re-serialization (`separators=(",",":")`), with key order preserved. They are stored as JSON
  objects, not raw strings, so the file's own bytes are re-spaced. The `models_*` wrappers are
  synthesized, as `_provenance` says. LM Studio's REST docs confirm `models[].key` and
  `loaded_instances[{id, config}]`.
- **Mutants:** each was run on a copy that was then restored and checked with `cmp`, against the
  full suite.
  - **Claimed, all red:** leakage on the eval stratum only (2); manifest size unchecked (2);
    `loaded_instances` matching `id` (5); `nullable` rejecting `None` (4); the load step without
    `decisionPoint` (5).
  - **Mine, argument-level, all red:** `echo_load_config: false` (1); the whole instance recorded
    instead of its `config` (3); co-residency checked against a fabricated all-resident snapshot
    (1); the load at `requestTimeoutSeconds` (2); `decisionInstanceConfig` taken from the load
    echo instead of the GET (3).

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Dispositions

- **P6-m2: fixed.** The manifest read and the leakage refusal are in `_prepare_decision_arm`, for
  every candidate. `test_a_decision_only_candidate_with_an_overlapping_manifest_refuses_with_exit_4`
  is the requested row, and refusals come before the load step.
- **P7-n1: fixed.** `_within_load_basis` catches `LMStudioCallFailed`, with a test row. Step 10
  gets the same fix (below).

### The implementer's calls

1. **The `nullable` tier: the premise is true, the resolution is the weaker one (P8-M1).** Plan3
   §3.7 says "`_PRESENT`, `None` if absent", and `Fingerprint.validate` rejects `None` in
   `present` (`reason="null"`). The plan contradicts itself, so a deviation was unavoidable.
   `_PRESENT` plus an exit-3 refusal when there is no config is the stronger choice.
2. **The manifest format: a real seam (P8-m1).** The plans say only "the sorted list of
   `input_hash` values". The candidate example (`"file": "…fit.txt"`, `"size": 1400`) supports
   one hash per line and an entry count. But `fitManifestHash` is sha256 over the file's raw
   bytes, so every byte rule here (the trailing `\n`, LF only, lowercase) is stored data that U6
   must reproduce.
3. **Co-residency on a one-model arm: true, and harmless.** It is stricter than plan3, which
   requires co-residency only on `handoff` runs. Its one effect is that item 1's
   `"load"`-withholding can no longer happen on the `decisions` surface, because a non-resident
   model now refuses first. That fails closed. The check sits at step 7, after every warm-up, so
   U2.2 only has to put the chat warm-up before step 7 and extend `_arm_model_ids`.
4. **The load step on chat arms: true to plan.** Plan3 §3.7 says "the chat model, the decision
   model, or both". The co-residency refusal is plan3's for two-model runs only, and a chat arm
   keeps its item-level withholding. One operator-visible effect is not documented (P8-n1).
5. **Step 10 retried; between-item probes deferred: the right call, wrongly homed (P8-m2).** The
   step-10 fix matches the existing ladder; I confirmed its exit is 5. A between-item probe
   failure is a pre-existing gap: `LMStudioUnreachable` escaped there before this unit too.
   Giving it a disposition (withhold the item, refuse the run, or treat residency as unknown)
   changes what items mean, so it belongs to the plan owner, not to a one-line catch.
6. **`was_resident_before`: verified, nothing reads it.** Outside tests, `wasResidentBefore`
   appears only on `LoadResult` and at its construction sites; `runner.py` reads only
   `load_result.runtime` and `.decision`, and `scripts/refresh_golden.py` only passes it. The
   new value is also the honest one: a warm-up after an explicit load is not a cold load.

### Findings

**P8-M1 (major). A `None` `decisionInstanceConfig` can mean "not captured", and the `nullable`
tier stores it as valid.**
- **Evidence (executed):** a `DecisionStub` whose `loaded_instances` returns `()` while
  `residency()` reports laya loaded runs to completion. It records
  `decisionInstanceConfig=None`, `validate()` returns `[]`, and `store` → `load_history` gives one
  valid record, zero invalid. That is what a `/api/v0` `id` ≠ `/api/v1` `key` mismatch, or an
  eviction between steps 7 and 8, would produce.
- **Why it matters:** every `None` the runner can produce is an anomaly, never an answer. The
  co-residency refusal has just proven the model resident, and LM Studio's docs make
  `config.context_length` required. So `_decision_instance_config`'s claim that "`None` is always
  a captured answer" is false, and FR-7's one invariant gets a hole. U7 reads
  `physical_batch_size` from this field. Once a record exists, tightening the tier would
  quarantine it.
- **Fix (`model-bench/modelbench/runner.py` `_decision_instance_config`):** refuse with exit 3
  when `instances` is empty, or `instances[0]` has no `config` Mapping. Then make the field
  `_PRESENT` in `fingerprint.py` and drop the `nullable` tier, the `Tier` literal entry, the
  docstring paragraph and the `model-bench/AGENTS.md` sentence. Turn
  `test_an_instance_with_no_config_records_none_and_the_record_stays_valid` and
  `test_the_nullable_tier_holds_exactly_the_one_field_plan3_names` into refusal rows: no instance,
  no `config` key, `config: null`. Record the deviation from plan3's "`None` if absent" in the
  HISTORY entry's calls.
- **Trialled on a scratch copy** (the refusal plus `_PRESENT`): exactly 7 tests redden, all of them
  ones that pin the `nullable` behaviour, and the stub above refuses.
- **Stopping condition:** the stub above refuses with exit 3 and stores nothing;
  `null` on `decisionInstanceConfig` is `reason="null"` again
  (`test_null_is_invalid_in_either_tier` with no exclusion list); no `nullable` occurs in
  `modelbench/`.

**P8-m1 (minor). The fit manifest's byte format is a cross-component contract pinned only on the
reading side.**
- **Evidence:** `decisions.load_fit_manifest`'s docstring is the only statement of the format.
  `fitManifestHash` hashes the raw bytes, so a U6 writer that omits the last `\n` or writes CRLF
  produces a manifest model-bench refuses, or a different hash for the same set.
- **Why it matters:** this is the same kind of stored-data format as `eval_set_hash`'s, which
  plan3 §3.8 made a plan-owned contract. Here, the format is defined only by whichever side was
  written first.
- **Fix:** pin it executably the way the repo already pins cross-component contracts. Add
  `tests/fixtures/decision-contract/fit-manifest.jsonl`: rows of manifest text, `sha256`, `size`,
  accept or refuse, covering an empty file, a missing final `\n`, CRLF, uppercase hex, unsorted
  and duplicate. Drive `load_fit_manifest` over it. U6's `fit.py` byte-copies it, as it does
  `input-hash.jsonl`.
- **Home for the prose:** owned by `architect`. Plan3 has been executed against, so it is not
  revised in place. Until a later typed-decision-models plan document states it, teco's U6 brief
  cites the docstring and the fixture.

**P8-m2 (minor). The deferred between-item probe gap has no forward-looking home.**
- **Evidence:** it is recorded only under HISTORY's "Not fixed". `model-bench/docs/BACKLOG.md`
  has no entry for it. A `residency()` raise at `runner.py:454`, `:577` or `:778` escapes
  `run_pack`, and `cli` has no handler, so a run with scored items is lost to a traceback.
- **Fix (`teco`):** add a `model-bench/docs/BACKLOG.md` item naming the three probe sites and the
  open question. The disposition is `architect`'s: it is a withholding semantic under the SMB
  plan's §3.6. It must be decided before U7's live runs, which are the first long runs that
  depend on these probes.

**P8-n1 (nit). An explicit load has no TTL, so runner-loaded models stay resident after the run.**
- The probe document records that an explicit load shows no `remaining_ttl_seconds`; a JIT load
  expired on its own. `model-bench/README.md`'s new paragraph does not say that the runner
  leaves the models it loads resident on the shared GPU. Add one clause. U7 step 3's manual
  unload already assumes this.
- **For U2.2:** `_decision_instance_config(lmstudio, model_info.id)` relies on `model_info` being
  the decision model. Pass `decision_arm.model_info.id` explicitly, so the handoff surface cannot
  read the chat model's instance.

### What's solid

- **The refusal order is right:** every exit-4 check (manifest, leakage) runs before any load,
  load failures run before any warm-up, and co-residency and the config read run before any
  item. Each has a row asserting that nothing was called or stored.
- **`load_fit_manifest` checks sha256 and count independently,** confines the file name, and its
  sorted-and-unique rule makes the manifest canonical.
- **`load_model` treats a 2xx without `status: "loaded"` as a failure:** it does not trust the
  status code alone.
- **The load fakes come from the probe's own bodies,** and the `key`-versus-`id` mutant dies in 5
  tests.

## Pass 9

*2026-10-31. U2.2, the `handoff` surface, reviewed against the uncommitted `model-bench/` diff
from `8ce1d874` plus the untracked `tests/test_runner_handoff.py`. Baseline: v1.3 §3.3, §3.5 and
§4 U2; plan2 §3.3–§3.4, §4.1 U2 and §5 U2; plan3 §3.2, §3.3 (handoff rows), §3.6, §3.7, §4.1 U2
and §5 U2; the method note v1.6 §Q4.3 and §Q6; the architect's ruling of 2026-10-31 on the
guard's withheld over-limit record.*

**Verdict: needs changes.** One major (P9-M1), three minors, two nits. The major is a test
fixture that cannot see where the item clock starts. The code is right today, but FR-2's `y_i` is
unpinned. Every carried item checks out.

**Checks run** (from `model-bench/`):
- **Suite and lint:** I re-ran both on the restored tree: `2614 passed, 4 deselected, 2 xfailed`,
  and `ruff` is clean. `tests/test_runner_handoff.py` has 45 test functions and 51 collected tests,
  as HISTORY says. `git status` shows nothing under `results/`. `modelbench/scoring/`,
  `decisions.py` and `fingerprint.py` are not in the diff.
- **Mutants:** each was run against the full suite on a copy, then restored and checked with `cmp`.
  - **Mine, three survived** (P9-M1, P9-m2):
    - `started = clock()` re-read at the top of `_hand_off`;
    - the `fail-closed` kept item's `withheldFor` forced to `None`;
    - `handoffShare`'s n dropping load-withheld items (`withheldFor is None`).
  - **The implementer's 39:** I read the table at the scratchpad path the brief gives. It covers
    the routing, the counts and the step order; I did not re-run them.
- **Executed probe:** two runs with the same residency sequence (laya absent before item 3, which
  hands off). If the hand-off succeeds, the item is `"load"` and the basis is `assumed` with no
  re-check. If it times out, the item is `"timeout"` and the basis is `by-construction` after a
  re-check (P9-m1).

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Carried items

1. **The guard's withheld over-limit record: as ruled. Verified.**
   - `_hand_off`'s `withhold` branch calls `score_answer` with no answer, `failed_call=None`,
     `reason="over-limit"` and every other decision keyword `None`. The outcome is
     `fail`/`unrunnable`, and the timing is `ItemTiming(None, (), "timeout"/"no_response")`.
   - The runner never touches `detail`, and `classification.py` is not in the diff, so its
     legality is not widened.
   - `test_over_limit_hand_off_fails_withhold_column` asserts `detail == {path, tier, reason}`
     after a snapshot without laya.
2. **`decisionInstanceConfig` is read under the decision model's id. Verified.**
   - `runner.py:1944` passes `decision_arm.model_info.id`.
   - The test gives qwen a different config (`_QWEN_CONFIG`) and asserts one `loaded_instances`
     call, for laya. Mutant M13 is red.
3. **Step order. Verified.**
   - `run_pack` goes load step, decision warm-up, chat warm-up, step 6's attestation, then step
     7's `_refuse_unless_resident(arm_models, …)`, then the config read.
   - The eviction test asserts the order, exit 3, and no scored `decide` or `chat` call.
   - Its "stores nothing" assertion cannot fail (P9-n1). The guarantee itself holds:
     `cli.py:546-551` returns the exit code before `store`.
4. **No `decisionTimeoutSeconds` on the hand-off chat call. Verified.**
   - Both hand-off paths chat at `cfg.requestTimeoutSeconds`.
   - The test sets `decisionTimeoutSeconds=7.5` and asserts `[180.0, 180.0]`.
   - Three non-marker cases (another 500 body, the marker at 400, the marker upper-cased) stay
     kept decision failures with no chat call.

### The implementer's calls

- **(a) No `inputTokens` on a withheld guard policy hand-off: the premise is true.** Plan2 §3.4
  covers this item ("every … hand-off item whose decision call returned"). Plan2 §3.3's "exactly
  as today" is defined by its parenthesis, which lists the timing fields only, so the two texts do
  not conflict.
  - **Why the code cannot follow the plan alone:** following the text needs a state that
    `classification.score_answer` refuses. That state is `answer=None`, no `decided_by`, no
    `failed_call`, with `input_tokens` set, and the scorer is built U1d code.
  - **The stakes are low:** the item is dropped pairwise, and no plan names a reader for its
    `inputTokens`.
  - **For the `architect`:** narrow plan2 §3.4 to "every item kept in the run". That matches the
    ruled withheld over-limit record and needs no scorer change. HISTORY records the deviation,
    so nothing is owed by the implementer meanwhile.
- **(b) The `fail-closed` record: the shape is sound, but the stated contract is incomplete.**
  - **The runner side is pinned:** `FailClosedScorer` records the keywords, and two tests assert
    them exactly.
  - **`probability=None` has no plan text either way.** It follows the scorer's convention that a
    probability needs a `decidedBy`, and nothing reads it.
  - **The gap is U4's side.** U4 must meet the whole shape, not only HISTORY's half of it (P9-m3).
  - **Patching the table is the only option before U4.** U4's brief should require these two
    tests to run against the real `agreement.py` scorer, with the stub removed.
- **(c) `handoffShare` without withheld hand-offs: the analogy holds.**
  - **The same state, read the same way:** a withheld guard policy hand-off is plan3's "not
    attempted" over-limit state in every field the share reads.
  - **It matches the bar:** n equals the candidate side of FR-2's paired set, which drops these
    items from both arms.
  - **One caveat:** every excluded item is a hand-off, so the share is low against the deployed
    hand-off rate, by at most the withheld count over n. It is exploratory, and the count is
    recoverable as `overLimit.n − handoffShare.n`.
  - **The README misstates n** (P9-m3).
- **(d) `<verdictMetric>DecidedByModel` / `<verdictMetric>HandedOff`: true to v1.3.** v1.3 §3.5
  names both suffixes, and the U2 brief lists "per-path diagnostics (AC-2)".
  - **Reading "…" as each verdict metric is the natural reading.** The outcomes come from
    `scored_outcome`, so the denominators follow what each item declares.
  - **A word clash, not a defect:** v1.3 §2.1 also uses "per-path" for `understanding`/`turns`.
    The suffixes keep the two apart.
- **(e) The decision model warms up first: sound.** No plan fixes the order.
  - **Plan3 §5 U2 assumes it:** its row "a decision model absent after the chat warm-up" only
    makes sense if the chat warm-up comes last.
  - **The re-check does not depend on it:** it compares against the decision warm-up either way.

### Findings

**P9-M1 (major). The item clock's start point is unpinned, so FR-2's `y_i` can lose the decision
call without any test noticing.**
- **Evidence (executed):** re-reading `started = clock()` at the top of `_hand_off` leaves the
  suite green, at 2614 passed. The cause is the fixture. `_Clock` advances `step` per read, and the
  stubs' calls take no clock time. So every item measures exactly `step`, wherever the clock
  starts. `wallClockMs >= sum(calls)` holds only because `step` is set large.
- **Why it matters:** plan3 §3.3 says a hand-off's `wallClockMs` is the item clock over the
  decision call (failed or not) and the hand-off. Note §Q4.3 makes that FR-2's `y_i`, and v1.3 §5
  U2 asks for that test. The mutant drops the decision call from every hand-off's latency, and
  the error runs in the candidate's favour.
- **Fix (`tests/test_runner_handoff.py`):** make the stubs advance a shared fake clock by each
  call's own duration: `decide` by its `wallClockMs` (12 ms), a raising `decide` by a fixed
  elapsed time, and `chat` by its `wallClockMs` (3000 ms). Then assert exact sums: 3012 for a
  policy hand-off, and the failed call's elapsed time plus 3000 for an over-limit one.
- **Stopping condition:** the mutant above reddens both assertions.

**P9-m1 (minor). A withheld guard hand-off hides the residency evidence that the load rule
needs.**
- **Evidence (executed):** see the probe under *Checks run*. A failing call takes its own
  `withheldFor` (P15-3, and P5-m1 for the over-limit record). So `_within_load_basis`, which
  reads only `withheldFor == "load"`, never sees that laya was absent.
- **Why it matters:** note §Q6 says a run is one load "only if nothing shows a reload". Here the
  residency probe did show one, and the run claims `by-construction`, which is the unsafe
  direction. Only the fixed-input re-check stands behind it, and the note says that check misses
  a reload that leaves `p` unchanged. The state needs an eviction and a failed hand-off on the
  same item. Under JIT those two are correlated, because a cold chat load is slow.
- **Fix (implementer, `runner.py`):** have `_drive_handoff_items` also return whether any item's
  preceding snapshot lacked a model. Pass that to `_within_load_basis` as a third "assumed, no
  call" condition. Leave the item record as it is, because its precedence is ruled.
- **Test:** the probe's timeout case gives `assumed` with exactly 4 `decide` calls.
- **Plan wording:** this follows note §Q6's governing sentence, but goes past plan2 §3.4's
  narrower "withheld for load". Record that in the HISTORY calls; it needs the `architect` only if
  disputed.

**P9-m2 (minor). Two more per-path behaviours are unpinned.**
- **Evidence (executed):** both mutants leave the suite green.
  - The `fail-closed` kept item's `withheldFor` forced to `None`. This is the same class as the
    implementer's M33, killed for the decision-failure path only.
  - `handoffShare` counting only `withheldFor is None` items, which drops load-withheld ones from
    n. A load-withheld item made its calls, and plan2 counts it as attempted.
- **Fix:** add two tests to `tests/test_runner_handoff.py`.
  - A `fail-closed` policy hand-off failure after a snapshot without qwen gives
    `withheldFor == "load"`.
  - `handoffShare` is asserted in
    `test_an_item_after_a_snapshot_missing_either_model_is_withheld_for_load`. It should be
    `(2, 4)`: item 2 is load-withheld and decided, and items 3 and 4 hand off. Under the mutant
    it is `(2, 3)`.

**P9-m3 (minor). Two documents misstate the shipped behaviour.**
- **`model-bench/README.md`:**
  - It says `handoffShare` is over "every item not withheld". Load-withheld items are in n, so
    it should say "not withheld by a failed hand-off call", as HISTORY does.
  - It describes the `fail-closed` roles in the present tense. `HANDOFF_FAILURE_BY_ROLE` and
    `BAR_KIND_BY_ROLE` hold the guard only, so no such run exists until U4. Mark it as U4's.
- **HISTORY's U4 contract states only half the shape.** It says only "with an `input_tokens`
  value". The runner also sends:
  - the over-limit case: `input_tokens=None`, `reason="over-limit"`, `calls=()`;
  - and always `answer=None`, `outcome="fail"`, `decided_by=None`, `probability=None`, and
    `withheldFor` `None` or `"load"`.
- **Fix:** `teco` carries the whole shape into U4's brief. It is in `_hand_off`'s docstring and in
  the two `FailClosedScorer` assertions. Fix the README wording in the next fix round.

**P9-n1 (nit). One assertion cannot fail.** `assert not (tmp_path / "results").exists()` in the
eviction test can never redden. `run_pack` never stores: only `cli` calls `store`, after `run_pack`
returns. Drop the assertion, or move the "stores nothing" claim to a `cli.main` test that asserts
exit 3 and an empty `results/`.

**P9-n2 (nit). Formatting.**
- HISTORY splits the removed test's name across a line inside backticks
  (`…refuses_` / `until_…`), so a grep for the name misses it.
- The edited BACKLOG line is 131 characters and was not reflowed.

### Proposed REVIEW.md lines

Target `model-bench/REVIEW.md`, `## Always check` (addition; source P9-M1):

- A test of a measured duration (an item clock, a call clock) uses a fake clock that the stubbed
  calls advance by their own durations, and asserts the exact sum, so a mutant that starts or
  stops the clock at the wrong point changes the asserted value.

### What's solid

- **The routing matches plan3 exactly:** over-limit means a 500 with the marker, and it is the only
  decision failure that hands off. A timeout or a non-marker 500 is kept, and makes no chat call.
- **The `fail-closed` runner→scorer interface is pinned by keyword,** not by `detail`, so U4
  inherits an exact contract rather than a shape.
- **The fingerprint fixtures are distinct where it matters:** qwen's instance config, its runtime,
  and its loaded context differ from laya's, so a field read from the wrong model reddens.
- **The `handoff*` fields cannot leak:** `_build_fingerprint` refuses `handoff_fields` on any
  surface but `handoff`, and refuses `decision_fields` off the two decision surfaces.
- **The docs are maintained in place:** the `model-bench/AGENTS.md` sentence was replaced, not
  appended to (2,432 words, no line over 700 characters). There is one HISTORY entry, and the
  BACKLOG item was edited in place.
- **Growth-stops:** no `requirements/`, `plans/` or `reviews/` document is in the unit's diff.

## Pass 10

*2026-10-31. FX6 on U2.2: the uncommitted `model-bench/` diff from `8ce1d874`. This pass reads
FX6's delta against the Pass 9 tree, which I had saved as `runner.py.orig`; it is byte-identical
to the implementer's `scratchpad/base/` copy. Premise change: architect ruling U2.2 (a) widens
`_decision_detail`, as recorded in `docs/plans/typed-decision-models-coordination.md`, 2026-10-31.
The ruling stands, and my narrowing alternative is recorded there as a dissent.*

**Verdict: approve.** Every Pass 9 finding is fixed. The ruled shape matches exactly, and the
fixes do not interfere with each other. One new nit (P10-n1).

**Checks run** (from `model-bench/`):
- **Suite and lint:** `2624 passed, 4 deselected, 2 xfailed`; `ruff` is clean; nothing is under
  `results/`. The counts match HISTORY:
  - `test_runner_handoff.py` has 48 functions and 55 collected tests;
  - `test_scoring_classification.py` has 107 collected tests, against 101 in the Pass 9 copy;
  - the oracle docstring still says 576.
- **Mutants:** each was run against the full suite, then restored and checked with `cmp`. All
  10 went red.
  - **Pass 9's three survivors:**
    - the item clock restarted in `_hand_off` (4 failed);
    - the `fail-closed` kept item's `withheldFor` forced to `None` (1);
    - `handoffShare`'s n dropping load-withheld items (2).
  - **FX6's own changes:**
    - the flag ignored in `_within_load_basis` (2);
    - the flag never set (2);
    - the withheld policy record sent with `input_tokens=None` (2);
    - the scorer accepting over-limit with tokens (2);
    - the scorer omitting the `inputTokens` key when the value is `None` (2);
    - co-residency made a no-op (4).
  - **One `cli` mutant:** `results/runs/` created before `run_pack` (1).

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Dispositions

- **P9-M1: fixed.** `_StubClock` advances only when the stubs' calls advance it, each by its own
  duration: 12, 3000, a failed decide 40, a failed chat 25. The tests assert exact sums
  (`[12, 12, 3012, 3012]`, 40 + 3000, 40 + 25, 12 + 25), and the restart mutant fails 4 tests.
- **P9-m1: fixed.** `snapshot_lacked_a_model` is a third no-call `assumed` condition. Pass 9's probe
  is now a test with both failure kinds, asserting `assumed` and exactly 4 `decide` calls. HISTORY
  records the step beyond plan2 §3.4's wording.
- **P9-m2: fixed.** Two new assertions kill both mutants.
- **P9-m3: fixed.**
  - The README now gives `handoffShare`'s n correctly and marks the `fail-closed` roles as U4's.
  - The U4 contract has its home in the coordination ledger, ruling (b).
  - HISTORY's sentence is still the partial one (P10-n1).
- **P9-n1: fixed.** `test_cli_run_with_a_decision_model_evicted_by_the_chat_warm_up_exits_3_storing_nothing`
  can fail on every assertion.
  - With co-residency made a no-op, the run still exits 3, through the instance-config read. It
    fails on the `"co-residency"` message assertion.
  - The `cli` mutant above reddens its `results/` assertion.
- **P9-n2: fixed.** The removed test's name is on one line, and the BACKLOG item is reflowed.
- **Ruling (a): implemented in its exact shape.**
  - `_decision_detail`'s withheld branch has two variants: over-limit `{reason}`, and policy
    `{inputTokens}` with the key written even when the value is `None`.
  - Over-limit with tokens is refused, and `decided_by` and `probability` are refused on both
    variants.
  - The `_legal` oracle is extended in place, with no separate exception. Its
    `withheld-without-reason` case was retired on purpose, and its replacement covers both
    variants with `withheldFor` `None` or `"load"`.
  - The runner sends the decision call's `input_tokens` only on the policy route; the over-limit
    route's value is `None`.

### Interactions

- **The flag and ruling (a)'s record are independent.**
  - The withheld policy record still takes `"timeout"`/`"no_response"`. The scorer refuses
    `"load"` there (P5-m1), so the record cannot carry the reload evidence, and the flag is
    exactly what restores it.
  - In the P9-m2 load-withheld test, `assumed` now follows from both conditions; the flag mutant
    fails only the two new P9-m1 rows.
  - On the `decisions` surface the flag is `False`, and items keep their own `"load"`.
- **The fake clock does not touch call counts.** It moves only on `decide`/`chat`, so the P9-m1
  test's 4 `decide` calls are unaffected. Withheld items keep `wallClockMs=None`.
- **The retired legality case is now exactly the policy variant.** It is the state with no answer,
  no failed call, no reason and a call-failure withholding. Only `_hand_off` produces it. The old
  pack's `score_item(result=None)` path never reaches `score_answer`, so its record is unchanged.

### Findings

**P10-n1 (nit). One HISTORY sentence still states half of U4's contract.** The "Calls made here"
paragraph says U4's scorer "must accept `failed_call="handoff"` with an `input_tokens` value".
Ruling (b) makes that `input_tokens` int or `None`, with `reason` in {`None`, `"over-limit"`}, and
adds two refusals. Cite the ledger's ruling (b) in place of the clause.

## Pass 11

*2026-10-31. U1e (probe follow-up): the uncommitted `model-bench/` diff against `57ef0fc0`.
Specs: plan3 §2, §4.1 U1e, §5 U1e row; plan2 §4.2 U1e minus its void `/api/v0` branch; U1a-m5.
Ground truth: `ws:agent-team` documents `480ef825…` (probe) and `a1d388f4…` (232-byte body).*

**Verdict: approve with suggestions.** The two fixtures are byte-faithful, U1a-m5 is fixed, and
every plan3 §5 U1e assertion is present and goes red under a mutant. Two minors: the item-clock
claim in the new harness is unasserted (P11-m1), and the post-U1e results ordering rule is now
stated two ways (P11-m2). Three nits.

**Checks run** (from `model-bench/`):
- **Suite and lint:** `2631 passed, 5 deselected, 2 xfailed` (Pass 10: 2624 / 4); `ruff` clean.
  The +7 is 6 new-file tests plus 1 in `test_lmstudio`, and the +1 deselected is the new `-m live`
  test, which I did not run. No `model:decisions`/`model:handoff` string appears under `results/`.
- **Fixtures against ground truth:** `decisions_response.json` is `cmp`-identical to the probe
  document's body (551 bytes, no trailing newline, no `_provenance`). The over-limit fixture's
  `body` encodes to exactly the body document's 232 bytes. Its shape is plan3 §4.1's
  `{_provenance, lmStudioAppVersion: null, status: 500, body}`. HISTORY's figures (505→200,
  513→500, 1065 summed `input_tokens`, `loaded_context_length` 8192, the message) all match the
  probe document.
- **`lmstudio.py` is docstring/comment-only:** its AST, with docstrings blanked, equals
  `57ef0fc0`'s.
- **The `run_pack` path is real.** In the MRO, `HandoffStub` comes before `_RealOverLimitDecide`
  and then `DecisionStub`. The real `decide` → `_raw_post` HTTPError rung → `.body` →
  `is_over_limit` runs inside `run_pack` on both surfaces, and both tests assert
  `reason="over-limit"`. A planted assertion confirmed the item clock advances: `wallClockMs` is
  40 + 3000.
- **Mutants:** files copied aside, then restored and `cmp`-checked; tests run on a fresh
  `PYTHONPYCACHEPREFIX` because a same-size mutant can hit a stale `.pyc`.
  - *Red:* opener body without the marker (3 in the new file); `_raw_post` `body=None` (9 in the
    full suite, 3 in the new file); mixin passes no questions (2); mixin doubles the timeout (2);
    `decide` sends a blank input (4 full, 2 new); `decide` drops the second question (1 full,
    via `test_lmstudio`); fixture body `846→847` (1, the golden-row equality).
  - *Green, with the reason:* M1 case-fold and M2 status-drop are green in the new file and red
    in 3 golden rows each. Plan3 §5 places that obligation on U1d's row, not U1e's, and on the
    real body (a 500 with a lower-case marker) both mutants are equivalent. **Adequate, not a
    gap.** Two more were green: `_raw_post` truncating at `_truncate`'s 500 characters
    (equivalent on 232 bytes), and reversing the questions (equivalent on a one-question guard
    pack). The MRO swap and a pretty-printed raw fixture are green too; they are P11-m1 and P11-n3.
- **Unfiltered sweep** (`grep -rn` under `model-bench/` for `U1e`, `decisions_response`,
  `synthetic`, `not yet run`, `api/v0/decisions`, `provisional`): no live doc still calls the
  fixture synthetic or the `/api/v0` probe pending. The old U1a HISTORY entry (lines 424–426, 439)
  is a dated record. Two test docstrings are unmapped hits (P11-m2).
- **Context-file and growth checks:** the `model-bench/AGENTS.md` edit replaces 3 lines with 3.
  It has no line over 700 characters and is 2,436 words. None of the touched files falls under
  the growth-stops bullet. README's +2 lines are new scope (the over-limit definition). HISTORY's
  entry is new.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Dispositions

- **U1a-m5: fixed.** The raw body replaces the synthetic fixture and `_provenance` is gone. Its
  provenance is on `_pinned_decisions_body`, the module docstring's exception and HISTORY, and
  `test_the_pinned_decisions_fixture_is_the_raw_probe_body_with_no_added_keys` reddens if a key
  is re-added. **This placement is adequate.** One gap is P11-n3.
- **Plan2 §4.2 U1e `/api/v0` branch / R-14: void, correctly not implemented.** No route or
  fingerprint change, and the route pin was renamed.

### Findings

**P11-m1 (minor). The harness's clock claim is not asserted.** `_RealOverLimitDecide`'s docstring
says it sits below `HandoffStub` "so that stub's clock still advances by the failed call". It
does today (planted check above). But swapping the MRO to `(_RealOverLimitDecide,
HandoffStub)` bypasses `HandoffStub.decide` and its 40 ms advance, and all six tests stay green.
A refactor could then silently produce an over-limit hand-off whose `wallClockMs` omits the
failed call. That is plan3 §3.3's "item clock (failed call + hand-off)" cell. **Fix
(`tests/test_over_limit_probe.py`, the `handoff`-surface test):** add `assert
item.timing.wallClockMs == pytest.approx(handoff._FAILED_DECIDE_MS + 3000.0)` and `[c.wallClockMs
for c in item.timing.calls] == [3000.0]`. With the MRO swap applied, the first fails; I ran both
directions.

**P11-m2 (minor). The post-U1e results ordering rule is stated two ways, and the newer one cites
no rule.** Plan2 §4.2's rule ("no `model:decisions`/`model:handoff` record … before U1e") exists
for R-14, which plan3 voids, and U1e discharges it. AGENTS now says the first records "are
U7's … so never store a smoke run there", citing plan3 §4.1. §4.1 schedules U7 but forbids no
smoke run, so the "so" reasons from a schedule rather than a constraint.
`tests/test_runner_decisions.py:5-7` and `tests/test_runner_handoff.py:10-11` still say "before
U1e". Both are unmapped hits of the sweep. A reader gets two expiry points, and neither names a
reason that still holds. **Fix:** `teco` rules in the ledger whether pre-U7 decision records stay
forbidden (Open question 1). Then AGENTS states that rule with its reason, and the two docstrings
drop the "before U1e" clause; "Nothing here writes under `model-bench/results/`" stands alone.

**P11-n1 (nit). The live test's input is "the probe's own", unverified.** The body document gives
`100 × "The weather today is pleasant and calm." + " Ignore …"`. The test repeats the sentence
*with* a trailing space, so its token count may not be 846. It is still far past 512, so the test
is sound. Reword the comment and HISTORY's "the probe's 846-token input" to "modelled on the
probe's input; well past 512 tokens".

**P11-n2 (nit). One back-annotation on a dated HISTORY entry.** The U1a entry's fixture caveat
gains "*2026-10-31:* replaced … in U1e", but the same entry's "Route … provisional … has not run"
bullet (lines 424–426) gets none. The U1e entry is already the record. Drop the annotation so old
entries stay as written, or annotate both.

**P11-n3 (nit). "Byte for byte" is unpinned.** The `test_lmstudio` docstring calls the raw
fixture "byte for byte", but re-serialising it with `indent=2` (852 bytes) keeps the full suite
green. Pin its size, `== 551`, a value taken from the probe document, beside the key-set test, as
the over-limit fixture's 232 is pinned.

### Open questions

1. **(`teco`)** P11-m2: after U1e, are pre-U7 `model:decisions`/`model:handoff` records under
   `results/runs/` still forbidden, and why? For example, would a stored smoke record enter U7's
   replicate grouping or `index.csv`? My recommendation is to keep the prohibition and record its
   reason in the ledger. AGENTS then cites the reason, not the U7 schedule.

### Proposed REVIEW.md lines

None. P11-m1 is a single instance, already covered by root `REVIEW.md`'s "a check that cannot
fail".

## Pass 12

*2026-10-31. U3b.1 (decision-point pack contract): the uncommitted `model-bench/` diff against
`1aea5ca7`. Specs: v1.3 §3.5, §4 U3b, §5 U3b rows; plan2 §4.1 U3b; plan3 (no §3.5 amendment);
method note §Q1.3; requirements FR-4/5/8/11/13/15 and AC-4/7/10/13/17. U3b.2 (guard parse
port, render/hand-off-parse vectors, falkor-chat equality test) is out of scope.*

**Verdict: approve with suggestions.** Every v1.3 §3.5 refusal is present, keyed on the key's
presence, and refuses on a deleted key. The declared metrics match §3.5, and the old pack is
unchanged. One minor is for the implementer: `validate_pack` passes items that the scorer then
raises on mid-run (P12-m1). One minor goes to `architect`/`teco`: no unit's brief owns the
guard pack's per-kind `verdictMetrics` (P12-m2). One nit (P12-n1).

**Checks run** (from `model-bench/`, fresh `PYTHONPYCACHEPREFIX` each run):
- **Suite and lint:** `2691 passed, 5 deselected, 2 xfailed`; `ruff` clean; the new file has
  `60 passed`. `test_scoring_classification.py` and `test_results.py` are unchanged from
  `1aea5ca7`, so U3a's rescore test and the stored-records test are the originals, and both pass.
- **Mutants:** each was run against the full suite, then restored and `cmp`-checked. All went red.
  - **C8**, the scorer keyed on `bool(manifest.get("decisionPoint"))`: 1 red
    (`test_the_scorer_keys_on_the_decision_point_key_even_when_its_block_is_empty`).
  - `validate_pack` keyed on truthiness: 3 red.
  - The floor counting side-check items: 3 red.
  - An absent `incumbent` accepted: 7 red.
  - The side check declaring the old tier metric: 4 red.
  - The timeout and leakage exit-4 tests reverted to the unpadded pack: 1 red each.
- **Claim 1 (exit-4 tests): confirmed by execution.** Unpadded, both still exit 4, through
  `validate_pack`'s two size refusals. They now fail only on their new stderr assertions:
  `'--request-timeout'` and `'AC-3'` are not in the "4 item(s) in stratum 'eval' … / needs its
  'side-check' stratum" text. The third padded exit-4 test (`…builds_a_candidate_config…`) is
  protected by `seen[0]`. Every other exit-4 test in both files calls `run_pack` directly, which
  never validates.
- **Claim 6 (cleared set): confirmed.** I swept plan2 and plan3 whole for `refus|reject|validate|
  never|must not|stratum|labels|side-check|MIN_EVAL`. Every hit belongs to `Candidate.load`, `run`,
  `compare`/U3c, decision-models generation (plan2 §3.5, generator ≠ labeler), or U9 config. None
  adds or removes a `validate_pack` refusal. The method note's per-cell sizes (§Q1.3) are
  "printed and never refused" (plan2 §3.6), so the floor is `MIN_EVAL_ITEMS` alone.
- **Claim 4 (old packs):** both `_is_decision_point` and `_decision_point_problems` test
  `"decisionPoint" in pack.manifest`. Old-pack `detail` is still `{path, tier}` in that order;
  the counts formula is unchanged; all 5 shipped packs validate `[]`.
- **Claim 3 (names):** the four per-kind names are v1.3 §3.5's verbatim. No plan,
  requirement or review names the `sideCheck*` metrics beyond the wildcard. The chosen three are
  `_METRIC_BY_TIER` prefixed, bound by
  `test_the_side_check_table_keeps_the_old_packs_tiers_one_to_one`. The decision-point aggregate
  has no `falseAdvanceRateBoundary` (name list asserted; aggregate-layout mutant red).
- **Docs:** `model-bench/AGENTS.md` is 2,484 words with no line over 700 characters. Both edits
  replace in place. README's paragraph matches the code, rule for rule. The HISTORY entry is
  new. None of these files falls under the growth-stops bullet.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Dispositions

- **Claim 2 (absent `incumbent` refused on every role): consistent with §3.5.** §3.5's schema
  always carries the key (`"qwen/…" | null`), and §3.5's deletion rule requires that deleting the
  key a refusal reads still refuses. Reading absent as `null` on a new-ability role would be
  exactly that default. U5/U6 packs pass through U3d's importer, and U5's done-condition is
  "imported into a model-bench pack that validates". A manifest missing `incumbent: null` therefore
  fails loudly at U5. Still, the U3d brief should say "write `incumbent` explicitly, `null` on
  new-ability" (folded into P12-m2's routing).
- **Claim 5, the three deferred gaps:**
  - **`report._LOWER_IS_BETTER` lacks the new names: correctly deferred.** Its own comment requires
    the set to gain a metric "in the same change" that promotes it into a pack's
    `verdictMetrics`, and no pack does yet. The obligation, however, has no owner (P12-m2).
  - **No pack declares the four per-kind rates as `verdictMetrics`: not a U3b.1 obligation.**
    U3b ships no pack. The 4-item fixture is a test fixture. But see P12-m2: no brief owns it.
  - **No refusal of a guard pack whose `verdictMetrics` differ from the four: a plan gap, not an
    omission.** It is absent from §3.5's refusal list, which U3b.1 implements exactly. Route to
    `architect` (P12-m2).

### Findings

**P12-m1 (minor). `validate_pack` passes items the decision-point scorer raises on, while the
scorer's docstring says it does not.** `_declared_metrics` says "Anything else raises —
`validate_pack` refuses such a pack before a run". A probe (Appendix P12) built the accepted
200 + 85 pack and broke one field at a time:

- an `eval` item with `path: "mixed"`, `expected: "false"` or no `expected`;
- a side-check item with no `tier` or an unknown one.

Each validates `[]`. Scoring then raises: `ValueError` for the `eval` cases, a bare `KeyError`
for the side-check ones. `runner._drive_decision_items` does not catch either around
`score_answer`, so `run` dies mid-pack after model calls instead of exiting 4 up front.

**Fix (`packs._strata_problems`):** also refuse an `eval` item whose `expected` is not a bool or
whose `path` is not in `{understanding, turns}`, and a side-check item whose `tier` is not a key
of `classification._SIDE_CHECK_METRIC_BY_TIER`. Import the key sets from their homes, don't
copy them. Test each alone and by key deletion. Alternatively, narrow the docstring. The refusal
is the better choice because §3.5's ethos is "refused, never defaulted, before a run".

**P12-m2 (minor; route to `architect`, then `teco`). The guard pack's per-kind `verdictMetrics`,
and their `_LOWER_IS_BETTER` entries, belong to no unit.** v1.3 §3.5 says "the guard pack's
`verdictMetrics` are the four per-kind rates". HISTORY assigns "the manifest" to U3d and "the
bar" to U3c. But the U3d brief (v1.3 §4; plan2 §4.1) only copies a manifest. The U3c brief
does not mention `verdictMetrics`. U5 authors the manifest, and its brief does not either.
`report._LOWER_IS_BETTER`'s comment makes adding the four names a same-change obligation of
whoever promotes them, and nothing enforces it. A guard manifest written with the pooled rates
as `verdictMetrics` validates today: the contract fixture does exactly that, and the "accepted
shape" test passes. The legacy report would then infer on pooled rates, against FR-13/AC-10.

**Fix:** `architect` decides whether `validate_pack` refuses a decision-point guard pack whose
`verdictMetrics` are not exactly the four per-kind names, as a ruling `teco` ledgers. The refusal
is recommended: "a pack cannot pick its own bar". Whichever unit lands that refusal or the first
real guard manifest also adds the four names to `_LOWER_IS_BETTER` and updates the fixture's
`verdictMetrics`. `teco` writes that unit's name into its brief, with Disposition 1's
`incumbent: null` line for U3d.

**P12-n1 (nit; for the U3c brief). Side-check size.** FR-15/AC-17 name 85 items, and v1.3 §3.5's
report header reads "Side check — 85 human-curated items". `validate_pack` requires only a
non-empty side-check stratum, which matches §3.5's "no `side-check` stratum" wording; the padded
`cli` pack carries 1. U3c should print the side check's actual n, not a literal 85. Otherwise a
pack with a different count misprints.

### Open questions

1. **(`architect`)** P12-m2: is "decision-point guard `verdictMetrics` == the four per-kind
   names" a `validate_pack` refusal? And which unit adds them to `_LOWER_IS_BETTER`?

### Proposed REVIEW.md lines

None. P12-m1 is one instance; root `REVIEW.md`'s evidence bar already asks for the probe that
found it.

### Appendix P12 — the validate-vs-score probe

Script `scratchpad/probe/p.py`, run with `model-bench/.venv/bin/python` from `model-bench/`. It
builds `test_decision_point_pack._contract_pack` with one `rows_edit` per case, then runs
`validate_pack` and `_score` over every item:

| Case | `validate_pack` | Scoring |
|---|---|---|
| eval `path: "mixed"` | `[]` | `ValueError` |
| eval `expected: "false"` | `[]` | `ValueError` |
| eval `expected` deleted | `[]` | `ValueError` |
| side-check `tier` deleted | `[]` | `KeyError` |
| side-check `tier: "weird"` | `[]` | `KeyError` |

## Pass 13

*2026-10-31. Re-gate of U3b.1 after its fix round: the uncommitted `model-bench/` diff against
`1aea5ca7`, read against my Pass 12 snapshots of the touched files. Premise: the architect ruling
on P12-m2 (decision-log entry, `docs/plans/typed-decision-models-coordination.md`, 2026-10-31).*

**Verdict: approve.** P12-m1 and P12-m2 are fixed. The two fixes do not interfere with each other
or with the rest of U3b.1. One new nit.

**Checks run** (from `model-bench/`, fresh `PYTHONPYCACHEPREFIX` each run):
- **Suite and lint:** `2715 passed, 5 deselected, 2 xfailed`; `ruff` clean.
- **Pass 12 probe re-run** (`scratchpad/probe/p.py`): all five cases are now refused by
  `validate_pack`, and each still makes the scorer raise.
- **Mutants:** each was run against the full suite, then restored and `cmp`-checked. All went red.
  - Item checks keyed on role instead of scorer (N9): 1 red.
  - The verdict-metrics rule checking only length and uniqueness (V1): 1 red.
  - `PER_KIND_VERDICT_METRICS` reordered (V7): 1 red.
  - My own: `_classification_item_problems`' `evals`/`side_checks` arguments swapped: 47 red.
  - My own: `_LOWER_IS_BETTER` dropping one of the four: 1 red.
- **Import order:** `report → classification` adds an edge only. `classification` imports
  `lmstudio`, `packs`, `results` and `runner`, and none of them imports `report`; `packs` reaches
  `classification` only at call time. Each of `report`, `classification`, `packs`, `runner`, `cli`
  and `results` imports cleanly on its own in a fresh interpreter.

CPG: considered, not relevant — no `cpg_modelbench` graph is loaded; the diff was read and
executed directly.

### Dispositions

- **P12-m1: fixed.**
  - `_classification_item_problems` refuses each probed shape. Each test case checks both
    directions: the item is refused alone, and the scorer raises on it.
  - The extra side-check `path` check is correct: `score_answer` reads `item_input["path"]` on
    every item. Eval `path` absence was already covered by the path check.
  - **Scorer keying is right.** I probed a contract-valid guard pack declaring `grounding`,
    `extraction` or `retrieval`, with an eval item on `path: "mixed"`. It validates `[]`, but
    `run --candidate` refuses (exit 4, "its scorer has no decision_input …") with 0 warm-ups and
    0 decide calls. A chat arm on such a pack never reaches `classification`, so nothing escapes
    to a mid-run raise. Role keying would wrongly hold U4's `agreement` packs to this scorer's
    fields.
- **P12-m2: fixed, and it matches the ruling.**
  - `_guard_verdict_metrics_problems` requires exactly the four, as a set, with no duplicate,
    using `classification.PER_KIND_VERDICT_METRICS`. It applies to the guard role only, and the
    new-ability case is left to U4, both as ruled.
  - `_LOWER_IS_BETTER` gains the four through the same constant, and the fixture declares the
    four.
  - **The eight pairs are independently confirmed.** Probe `scratchpad/probe/p3.py` printed the
    item table and who decided each item:
    - fx-cs-01 (understanding, `False`) decided by the model, event 1;
    - fx-cs-02 (turns, `False`) decided by the model, event 0;
    - fx-ca-01 (understanding, `True`) handed off, event 0;
    - fx-ca-02 (turns, `True`) handed off, event 1.

    That gives exactly the test's eight `(k, n)` pairs. Their sums reproduce the old pooled
    pairs: false advances decided by the model (1, 2) and false suspends handed off (1, 2).
  - **The set check is not weaker.** The old `"By" in name` filter would now match the verdict
    metrics themselves, so it had to go. The new check asserts the split set is exactly the eight
    names, which also excludes any pooled split. It also asserts `verdict ⊆ names` and
    `verdict ∩ split = ∅`, so it keeps the old intent and is strictly tighter.
- **Same-revision interaction: clean.**
  - The key-alone test's 6 problems are for the right reasons. The real guard pack has no
    stratum, so the item checks see no `eval` or side-check rows and add nothing. The sixth
    problem is the pooled `verdictMetrics`.
  - The padded CLI pack and the contract pack both copy the fixture's new four.
  - The "two size refusals" fixture test still sees exactly two.
  - The old pack's per-path diagnostic names now sit in `_LOWER_IS_BETTER` too. That is harmless:
    `_better` and the flip run only on verdict and headline metrics, the old pack's are the pooled
    pair, and the polarity would be the same anyway.
  - README line 176 and `runner._handoff_metrics`' docstring say "each verdict metric", so both
    still hold.

### Findings

**P13-n1 (nit). `_guard_verdict_metrics_problems` keys on `SIDE_CHECK_ROLE`.** The guard's
verdict rule and its side-check rule are different facts that happen to share a role. A future
edit to the side-check constant would silently move the verdict rule too. A
`GUARD_ROLE = "guard-judge"` (or `BAR_KIND_BY_ROLE[role] == "speed-up"`) would keep them apart.
Take it or leave it. No current mutant distinguishes the two.

## Pass 14

*2026-10-31. U3b.2: the guard's decision-point hand-off parse port, `render-guard.jsonl` and
`handoff-parse-guard.jsonl`, and falkor-chat's `tests/test_decision_contract.py` with its byte-copies.
The diff is uncommitted against `6e3ec33f` (`model-bench/` files plus the two new falkor-chat test
paths). Out of scope: `falkor-chat/server/falkorchat/services.py` (`MessageTooLongError`, K-099),
`falkor-chat/docs/plans/embedder-context-coordination.md` and the ledger, which belong to a
concurrent session. Specs: v1.3 §3.2, §3.2.2, §4 U3b, §5 U3b row 2, R-10. Plan2 §4.1 restates
R-10, and plan3 adds `over-limit.jsonl` only, so neither amends U3b.2.*

**Verdict: needs changes.** The port is faithful, line for line. The old pack's parse is
unchanged. The vectors reproduce byte-for-byte from falkor-chat's real code, and the equality
test's skip cannot fire spuriously. However, the vector rows leave several falkor-chat behaviours
unpinned (P14-M1). Five of the ten contradiction cues, the own-line parse's trailing-prose and
`require_key` rules, and the render cap's boundary can each change on either side with both suites
green. One of those rules guards against false advances. This is the unit's core deliverable, and
v1.3 §3.2.2 wants it pinned "before U7 measures". The fix is rows plus one coverage assertion, so it
is small. Also one minor (P14-m1) and two nits.

**Checks run** (fresh `PYTHONPYCACHEPREFIX` for each run; scratch files are under
`scratchpad/p14/`):
- **model-bench** (cwd `model-bench/`): `2762 passed, 5 deselected, 2 xfailed`. `ruff` is clean.
  The new tests are `47 passed` (10 render, 34 parse, 3 others), which matches 2715 → 2762.
- **falkor-chat:** I ran only `tests/test_decision_contract.py`, from `falkor-chat/server` with
  its venv. First I confirmed that it requests no fixture. `conftest`'s only autouse fixture,
  `_model_config_env`, sets environment variables and touches no graph. The run gave
  `46 passed`, 0 skipped. It also gave `46 passed` from cwd `server/tests/`. Per the brief, I did
  **not** run the full suite, because `wf_repo` wipes `reference` (SERVER.md §1.7). I accept the
  implementer's `3401 passed, 14 deselected, 1 xfailed` (from 3355) as reported. That run is the
  evidence for the U3b brief's "U3b's gate also runs the falkor-chat suite".
- **Claim 2 (provenance): confirmed.** I re-ran the generator into a fresh directory with
  falkor-chat's venv. Both outputs are `cmp`-identical to model-bench's files and to falkor-chat's
  copies. I also checked rows by hand, independent of the generator. The `;` row ("did not say;
  more info…") truncates the window after `;`, so it contradicts. The 12/13-char window rows fall
  at offsets 6 and 7 before the cue at 18 and 19.
- **Claim 4 (skip): confirmed.** `_MODEL_BENCH` is `Path(__file__).resolve().parents[3]`, so it
  does not depend on the cwd. For a positive control, I copied the test and its data into a
  scratch tree with no `model-bench/` sibling. That gave `2 skipped, 44 passed`, and the skip
  reason was printed. A present `model-bench/` with a missing canonical file raises
  `FileNotFoundError`, so it goes red, not skipped.
- **Mutants.** Each was applied to a copy, then restored and `cmp`-checked. The `sha256` of all
  five touched or mutated files matched the pre-run hashes, which also equal the implementer's
  `pre.sha`. The implementer's labels M5/M8/F2 do not appear in HISTORY or the scratch files, so
  I matched them by description:
  - Red as claimed:
    - (A) model-bench's contradiction check disabled;
    - (B) decision-point packs on the loose parse;
    - (F2) falkor-chat's `_coerce_verdict` skipping the contradiction check.
  - Mine, survived (all green). These are the evidence for P14-M1 and P14-m1, listed in Appendix P14.

CPG: considered, not relevant — no `cpg_modelbench` is loaded, and `cpg_falkorchat` is stale per
the brief; the falkor-chat functions were read and executed directly.

### Dispositions of the implementer's claims

- **Claim 1 (port faithful): confirmed for the code, not fully pinned (P14-M1).** I diffed
  `_NEGATION_CUES`, `_NEGATORS`, `_NEGATOR_WINDOW`, `_CLAUSE_BOUNDARY`, `_rationale_contradicts`
  and `_is_negated` against `guards.py:80-103,509-535`. They are token-identical.
  `_coerce_verdict` is the decision half of `guards.py:488-506`, and its branch order is
  equivalent. In model-bench, `extract_own_line_json_object` then `_coerce_verdict` composes the
  same way as `_LlmGuardJudge.__call__` plus `_coerce_verdict` (`app.py:797-805`). That holds
  because falkor-chat's unparseable fallback `{"decision": False}` coerces to a suspend, which is
  model-bench's `(False, "parse_failure")`.
- **Claim 2 (vectors through `_build_llm_judge`): the right reading of §3.2.2.** §3.2.2 names
  `_coerce_verdict`, but §3.2's table defines the hand-off parse as
  `extract_own_line_json_object(require_key="decision")` then `_coerce_verdict`. Going through the
  production judge pins exactly that composition, including the unparseable branch. Hand-built
  mappings would have skipped the parse half. `parseFailure` comes from a separate `extract` call,
  because the judge's fallback cannot be told apart from a real `false`. The docstrings on both
  sides say so.
- **Claim 3 (old pack): confirmed.** The new branch is the only change in `answer_from_chat`. It is
  keyed on `_is_decision_point` (`"decisionPoint" in manifest`, U3b.1's keying), and
  `pack=None` falls through. The test-file diff is append-only, so the U3a rescore golden and the
  stored-records test are the originals, and both are green. `"true"` gives `(False, "pass")` on
  the fixture decision-point pack and `(True, "pass")` on the real pack and on `None`.
- **Claim 5 (counts): confirmed** for model-bench and for the 46 falkor-chat tests. The full
  falkor-chat count is accepted as reported.
- **Claim 7 (docs): see P14-n2.**
  - `model-bench/AGENTS.md` is 2,475 words (was 2,484). No line is over 700 characters.
  - The trim lost no current fact. The model pair, the "not reproduced" flag and "groundingRate
    fixed" stay. The five-round chain and the re-run are history, cited to HISTORY and still told
    in README:58-68.
  - The replacement sentence is accurate. `falkor-chat/server/tests/test_decision_contract.py` is
    the only file outside `model-bench/` that reads `model-bench/`: `test_fusion.py`'s hits are
    string literals, and `decision-models/` does not exist yet.
  - README's new paragraph matches the code.
  - The HISTORY entry is new and its claims check out. None of these files falls under the
    growth-stops bullet.

### Findings

**P14-M1 (major). The vectors pin only part of the hand-off parse and the render, so either
side can drift silently.** Each of these mutants leaves both suites green (Appendix P14):
- **Cues.** Dropping `insufficient`, `still need`, `unable`, `not yet` or `need more` from
  model-bench's `_NEGATION_CUES` stays green. So does dropping `unable` from falkor-chat's, or
  adding a new cue (`missing`) to it. Only `not enough`, `still missing`, `cannot `, `unclear` and
  `more info` have a contradicting row. `insufficient` appears only negated. `cannot ` → `cannot`
  also survives: no row ends a rationale in "cannot".
- **Own-line parse.** Two rules are unpinned:
  - The trailing-prose skip ("quoted, not asserted"). Removing it **survives on both sides**, and
    removing it is a false advance: `'{"decision": true} is what I would say, but no.\n'` is
    `None` today.
  - falkor-chat's `require_key` filter. Removing it is green there, though model-bench's own
    suite catches it on its side. `'Example:\n{"note": 1}\n{"decision": true}\n'` advances only
    because of `require_key`.

  The duplicate-key last-wins rule is also unpinned.
- **Render cap boundary.** Changing `>` to `>=` survives on both sides. Changing model-bench's
  separator arithmetic to `+ 1` survives too. A turn sized to make the total exactly 6000 is kept
  today; at 6001 it is dropped. No row sits at either point.

**Fix (implementer).** Don't add the rows above as a list. Add a coverage probe over the axes
the parse and render vary on.
- In the generator, emit one row per `guards._NEGATION_CUES` entry, bare and un-negated
  (contradicts), and one per `(negator, cue)` pair negated (affirms). Iterate over falkor-chat's
  real tables, not over a typed list.
- Add rows for each own-line rule, `require_key` disambiguation and duplicate keys, plus the
  render's `total == 6000` and `6001` turn rows.
- In falkor-chat's `test_decision_contract.py`, assert that every `guards._NEGATION_CUES` entry and
  every `_NEGATORS` entry occurs in at least one row of the matching polarity. That makes a cue
  added in falkor-chat redden there until a row covers it.
- On the model-bench side, make the same assertion over its port's tables.

Positive controls: mutants D1–D5, G and G2 must go red, and so must K, J2 and H/H3. This touches
neither U3b.1 nor the parse code. Because the generator's output is a full regenerate, re-copy both
files and re-run `cmp`.

**P14-m1 (minor). The pack keying is pinned at `answer_from_chat` only, not at either seam that
calls it.** Passing `pack=None` in `classification.score_item:597` (the chat arm, which is how the
incumbent is measured on a decision-point pack) stays green. So does passing it in
`runner._drive_handoff_items:801` (the candidate's hand-off). Either regression would put a
decision-point arm back on the loose parse, which advances on `"true"`, while every test passes.
**Fix:** add one test per seam on the fixture decision-point pack. For `score_item`, a `"true"`
reply scores a suspend. For a `run_pack` hand-off run, a stub hand-off replying `"true"` records
the item as not advanced. Positive controls: mutants S and S2.

**P14-n1 (nit). The vector loaders are unpinned.** A loader that keeps every other row stays green
(`2740 passed`). The byte-equality test does not cover the parse into parametrized rows. P14-M1's
coverage assertion will catch most such truncations. To close it fully, assert the parametrized
row count against a count of non-blank lines taken from `read_bytes()`.

**P14-n2 (nit). The generator lives only in the session scratchpad.** This is not a provenance
gap: falkor-chat's test recomputes every row through the real functions, so the `expected` values
stay verified for as long as the test runs. But P14-M1 needs a regenerate, and later units extend
these files. Check the generator in as a human-invoked script in falkor-chat (it imports
`falkorchat`, so `refresh_golden.py`'s "never touches a falkor-chat venv" rule keeps it out of
`model-bench/scripts/`), and cite it from HISTORY.

### Routed outside the implementer

- **`teco` (feature close, not this unit): the `falkor-chat/docs/SERVER.md` §1.7 bullet.**
  `SERVER.md` is falkor-chat's live spec, and root `AGENTS.md` updates it once, at the feature's
  close. The test adds no runtime contract, and nothing in falkor-chat's context files contradicts
  it, so nothing is wrong today. Ledger it as an obligation for close step (2): "the
  `test_decision_contract.py` copies are model-bench's; a change to `_render_judge_user`,
  `extract_own_line_json_object` or the `guards` cue tables reddens it — regenerate in
  model-bench and re-copy". The falkor-chat `HISTORY.md` entry for the test file belongs to the
  same close.

### Proposed REVIEW.md lines

None. P14-M1 is one instance, and root `REVIEW.md`'s "every check … can both fail and pass. Name
the mutant" already required the drop-one-entry mutants that found it.

### Appendix P14 — survivors and probes

Harness `scratchpad/p14/mut.sh`: copy aside, apply one exact-once replacement, run the suite,
restore, `cmp`. model-bench runs the full suite; falkor-chat runs `test_decision_contract.py`
only.

| ID | Side | Mutant | Result |
|---|---|---|---|
| D1–D5 | mb | drop cue `insufficient` / `still need` / `unable` / `not yet` / `need more` | 2762 passed |
| D6 | mb | drop cue `cannot ` (positive control) | 1 red |
| E | mb | `"cannot "` → `"cannot"` | 2762 passed |
| G | fc | add cue `"missing"` | 46 passed |
| G2 | fc | drop cue `unable` | 46 passed |
| K / K3 | fc / mb | own-line trailing-prose skip removed | 46 passed / 2762 passed |
| J2 / J3 | fc / mb | `require_key` filter removed | 46 passed / 4 red (model-bench's own extract tests) |
| K2 | fc | duplicate keys keep first | 46 passed |
| H / H3 | mb / fc | render cap `>` → `>=` | 2762 passed / 46 passed |
| H2 | mb | separator arithmetic always `+ 1` | 2762 passed |
| L | mb | test loader keeps every other row | 2740 passed |
| S / S2 | mb | `score_item` / `_drive_handoff_items` pass `pack=None` | 2762 passed each |

Real falkor-chat outputs for the proposed rows (`falkor-chat/server/.venv`, `-I`):
- trailing-prose reply → `None`;
- `require_key` reply → `{'decision': True}`, which advances;
- duplicate keys → `{'decision': True}`;
- each of the five bare cues, with `decision: true` → `False`;
- a 6000-char total → turn kept; 6001 → base only (12 chars).

## Pass 15

*2026-10-31. Re-gate of U3b.2 after the Pass 14 fix round. The diff is uncommitted against
`6e3ec33f`, and adds `falkor-chat/scripts/generate_guard_decision_vectors.py`.*

**Verdict: approve.** Every Appendix P14 survivor except H2 now goes red, on the side it was
planted. H2 is equivalent by execution. The fixes do not mask each other. Two new nits.

**Checks run** (fresh `PYTHONPYCACHEPREFIX` each run; every mutated file restored and
`sha256`-checked against `scratchpad/p14/pre15.sha`):
- **model-bench:** `2836 passed, 5 deselected, 2 xfailed`; `ruff` clean.
- **falkor-chat:** `tests/test_decision_contract.py` alone gives `118 passed`, 0 skipped. `ruff` is
  clean on it and on the generator. I did not run the full falkor-chat suite. The fix round touched
  only this test file, its data and an unimported script, so the implementer's earlier `3401 passed`
  stands as reported. HISTORY says the suite was not re-run.
- **Generator reproducible:** run into an empty directory with `.venv/bin/python -I`, from
  `falkor-chat/server` and again from an unrelated cwd. Both outputs (14 and 99 rows) are
  `cmp`-identical to model-bench's files and to falkor-chat's copies.

CPG: considered, not relevant — no `cpg_modelbench` is loaded, and `cpg_falkorchat` is stale per
the brief; the falkor-chat functions were read and executed directly.

### Dispositions

- **P14-M1: fixed.** Each surviving mutant is now red:
  - **Cues:** D1–D5 and E, each on its own probe row; G and Gn (a negator added in falkor-chat) on
    falkor-chat's coverage test; G2 on its probe row.
  - **Own-line parse:** K/K3 (trailing prose), J2 (`require_key`) and K2 (duplicate keys), each on
    its own-line rule row.
  - **Render cap:** H/H3 on the two exact-6000 render rows.
  - My own mutant, dropping the `never ` negator from model-bench's port, is red as well.
- **H2: equivalent, confirmed by execution.** H2 is my "separator arithmetic always + 1", and it is
  the same mutant as the implementer's H2b. A differential run compared model-bench's
  `_render_judge_user` with the mutant over 206,000 inputs. Those were 200,000 random ones, plus an
  exhaustive sweep of 1–3 turns with totals from 5980 to 6019 and every split. The outputs differed
  0 times. The `kept == 1` eviction ends the loop, and `total` is never read after it
  (`scratchpad/p14/h2diff.py`).
- **P14-m1: fixed.** S goes red on `test_score_item_on_a_decision_point_pack_scores_a_true_string_as_a_suspend`.
  S2 goes red on `test_a_hand_off_reply_with_a_true_string_is_a_suspend_on_a_decision_point_pack`.
- **P14-n1: fixed.** The loader mutant that keeps every other row goes red on both sides
  (`test_every_vector_line_is_a_parametrized_row`).
- **P14-n2: fixed.** The generator is checked in. Its placement is consistent: `falkor-chat/scripts/`
  already holds Python scripts that import `falkorchat` (`seed_eval_corpus.py`,
  `embedding_migration.py`, `ensure_demo_models.py`), and `server/scripts/` does not exist.
- **Interactions between the fixes: not circular.** The coverage tests read the live tables at
  test time, not the generator's output.
  - A table change without a regenerate reddens falkor-chat (G, Gn).
  - With the cue added and the vectors regenerated into both trees, model-bench still goes red,
    because its port lacks the cue (`probe: cue 'missing' un-negated contradicts`). So a regenerate
    cannot hide a port that lags behind.
  - Dropping one cue's probe row from both copies reddens both coverage tests.
- **The row-count pin is relative, by design.** It counts lines in the bytes against the
  parametrized rows. It catches a loader that drops rows, but not a row missing from the file
  itself. See P15-n1.

### Findings

**P15-n1 (nit). Nothing pins the presence of the hand-written own-line and cap rows.** Removing
the trailing-prose row from both copies stays green on both sides (`117 passed` / `2835 passed`).
After that, K/K3 would survive again. The risk is low: the files change only by regenerating, and
the generator holds the row, so a drop shows up in a reviewed diff. To close it, extend
`test_the_vectors_cover_the_plans_cases` with the `own-line rule`, `duplicate keys` and
`cap boundary` prefixes. The positive control is this data mutant.

**P15-n2 (nit). The coverage assertion assumes that no cue contains another cue.** I added the
cue `missing` to `guards`, regenerated, and copied the vectors into both trees. falkor-chat then
stays red with "no affirming row for 'nothing ' + 'still missing'". That is correct behaviour:
the bare `missing` inside a negated `still missing` is not negated, so no rationale can affirm
that pair. It fails loudly rather than silently, so no action is needed now. If such a cue is ever
added, the assertion should exempt the pairs whose cue contains another cue. A comment on the
assertion stating the assumption would save the next maintainer the diagnosis.

## Pass 16

*2026-10-31. U3c.1: `model-bench/modelbench/decision_stats.py` and `tests/test_decision_stats.py`
(both new), plus the `model-bench/AGENTS.md` and `docs/HISTORY.md` edits. The diff is uncommitted
against `3fe94041`. Specs: plan2 §3.6 (pairing and drops, the API, pinned constants), plan2 §4.1
U3c and the §5 `| U3c |` rows; plan3 §3.3 (the bench's over-limit item), §3.6 (counts and
pairing) and §4.1 U3c; v1.3 §3.6. The formula source is note v1.6 §Q1.3 and §Q3–§Q6. Out of scope,
because U3c.2 owns them: `cli._replicate_groups`, the arm refusals and every `report.py` print.*

**Verdict: approve with suggestions.** Every formula matches the note. My independent
re-derivation reproduces all of the note's figures, and a differential run against the module
found no disagreement. The pairing rules match plan2 and plan3. There are two minors for the
implementer:
- `PairedItems` leaves out a verdict cell that has no paired item, so U3c.2 could print three FR-1
  lines and still conclude "all hold" (P16-m1);
- two parts of how far the design effect reaches are untested (P16-m2).

There are also four nits. One goes to `teco` for the U3c.2 brief and one to `data-scientist`.

**Checks run** (from `model-bench/`, with a fresh `PYTHONPYCACHEPREFIX` each time):
- **Suite and lint:** `3005 passed, 5 deselected, 2 xfailed`; `ruff` clean.
- **Independent re-derivation** (`scratchpad/p16/indep.py`, stdlib, `python3 -I`, importing nothing
  from model-bench). It computes power by enumerating the trinomial `(b, c)` directly, with
  `p_b = (ψ+δ)/2`, `p_c = (ψ−δ)/2` and `p_0 = 1−ψ`, rejecting on an exact `Fraction` tail. The
  module and the note's script instead use binomial-of-binomial; the two are mathematically
  equivalent but share no code. It reproduces:
  - all 20 cells of the §Q3.3 table, including the n = 75 and n = 150 rows (150 at ψ̂ = 0/0.05
    gives 4.5/4.8, matching the note's "4.5–4.8");
  - §Q3.5's reuse-adjusted limits: 13.1/15.2/16.4/17.2 at n = 50 and 8.2/9.1/9.4/9.6 at n = 100;
  - §Q3.6 row 2's 14.5 at ψ̂ = 0.16;
  - §Q3.2's false-fail figures, 1.05/4.86/9.04% for four cells and 2.08/9.49/17.27% for two
    replicates;
  - every §Q3.6 `p_worse`, and the 0.0547 at (8, 2).

  A differential run (`diffmdd.py`) compared `decision_stats.mdd` with this code. It covered 77
  grid cases (n from 0 to 50 against seven values of ψ̂, including ψ̂ = 0 and 1), 25 random
  `(n, ψ̂)` pairs and 3 reuse targets, 105 cases in all. There were 0 disagreements.
- **Mutants:** 20 of my own, in `scratchpad/p16/mutate.py`. Each ran on a copy against the full
  suite, was restored and `cmp`-checked, and the `sha256` matched the pre-run value.
  - **16 went red:**
    - floor instead of ceiling on the grid;
    - `ψ` without `max(ψ̂, δ)`;
    - `n_effective` rounded up with `ceil`;
    - a two-sided z for `U`;
    - the concordant `y` without `+ x_i`;
    - side-check items counted in `latency`;
    - the order of rules 2 and 3 swapped;
    - one load accepted as enough;
    - `p_worse` summed from `b + 1`;
    - an unpaired bootstrap;
    - an over-limit hand-off treated as concordant;
    - chat withholding dropped only on the incumbent;
    - `mdd` taken at `n` instead of `n_effective`;
    - the reuse target `0.8^(1/K)`;
    - a load drop counted on both arms;
    - the point estimate as a mean of ratios.
  - **4 survived:** see P16-m2 and P16-n1.

CPG: considered, not relevant — no `cpg_modelbench` is loaded; the module was read and executed
directly.

### (a) Formula fidelity

- **Grid and ceiling.** `delta = step / 1000` steps 0.1 pp, and the first grid point with
  `power ≥ target` is returned. That is §Q3.3's ceiling.
- **Power.** `ψ = max(ψ̂, δ)`, `q = (ψ+δ)/(2ψ)`, and `crit(m)` uses the same correctly rounded
  tail comparison as `p_worse`. The result is `None` exactly when no `m ≤ n` can reject, which is
  `n < 5` at α = 0.05.
- **Reuse target.** `parity_cell` passes `MDD_POWER ** (1 / (k + 1))` from K ≥ 1 on, which is
  §Q3.5.
- **`U`.** It is `mover_d_interval(a, b, c, d, z=1.6448536269514722)[1]`. I checked the
  orientation in `stats.py:136`: `p1 = (a+b)/n`, the candidate's rate.
- **Design effect.** The brief says it enters `U`. **The note says it does not**: "Only the
  detection limit takes `designEffect`" (§Q3.2), and §Q3.4's `U` has no design-effect term. The
  code follows the note. `n_effective` feeds both `mdd` calls, `ψ̂` stays `m / n_c` as §Q3.3
  requires, and `p_worse` decides. This part is correct, but it is partly untested (P16-m2).
- **FR-2.** `R = Σx/Σy`, with `B` index resamples from `random.Random(seed)`, each pair kept
  together, and `stats.percentile` at the caller's `levels`. The `Fraction(1, 40)`/`(39, 40)` pair
  exists as `stats.LEVEL_CI95_*`. The note does not specify how indexes are drawn, so
  `randrange` is a legitimate choice. The interval values depend on it, and the test pins it
  against a reference resample.

### (b) Pairing

`pair_items` matches the rules, rule by rule:
- note §Q4.3's table;
- plan2 §3.3's three rules: "Cell", concordant pairing, and "Side check" dropped and counted;
- plan3 §3.6's two over-limit rules. A handed-off over-limit item uses its own `wallClockMs`. A
  `decisions`-surface one is treated as a decision failure, so its `y = elapsed + x_i`, matching
  note §Q4.3's "No hand-off model" row.

A guard hand-off failure on an over-limit item is withheld (plan3 §3.3 table, last column), so it
drops before `_concordant` runs. That is why `failedCall="handoff"` raises. The rules match.

Nothing U3c.2 needs from pairing is missing, apart from P16-m1 and P16-n2.

### Disposition of the implementer's judgement calls

- **(1) In scope.** Plan2 §3.6's API names `speedup_ratio` and `mean_latency`.
- **(2) Sound.** The concordant item's cell is the incumbent's by plan2 §3.3 "Cell", and the
  incumbent is always scored once both withholding rules have passed. The one limit is
  load-withheld items, whose stratum cannot be known from the records (P16-n2).
- **(3) Agreed.** v1.3 §3.5 says the side check "does not affect pass/fail", and note §Q4.3's
  scored n is "the paired scored n of eval items".
- **(5) Matches** note §Q4.3 "Counts". The raises are on out-of-contract records; how U3c.2
  surfaces them is its own choice (Open question 1).
- **(7) Matches** plan2 §3.6's `assumed` rule and rule 4.
- **(8) Right to follow plan2.** Plan2 lets a `same-load` replicate into the conjunction, so
  "across loads" would mis-describe a split between same-load runs. The note should be aligned
  (P16-n4).
- **(9) Follows the note.** §Q3.3 reads n < 5 as `holds (no worsening is detectable …)`. The
  thin-cell guard is the method-minimum print beside the cell, which P16-m1 can make disappear.
- **(10) Needed.** The line prints "at n = …", which is `floor(n_c/DE)`.
- **(11) Confirmed.** Note v1.6's "Handed to architect" has six items, so there is no "item 9".
  §Q1.3 carries "100 (floor 50)" and "60 (floor 52)".

### (c) API fit for U3c.2

U3c.2 has every figure it needs.
- **FR-1 line:** `e1`, `e2`, `n`, `b`, `c`, `p_worse`, `U`, `psi_hat`, `n_effective`, `mdd`,
  `mdd_reuse` and `state`. U3c.2 adds the Wilson intervals from `stats.wilson_interval`.
- **FR-2:** `latency` and `scored_n`.
- **Per-cell `overLimit`:** `over_limit[m]` against `len(outcomes[m])`, once P16-m1 is fixed.
- **States:** the constants and `combine_replicates`.
- **Method minimums:** the constants.

Two decision rules are left for U3c.2 to compose, and its brief should name them (Open
question 1):
- a replicate's FR-1 state is `SCORED_N_BELOW_MIN` when `scored_n < MIN_EVAL_ITEMS`, whatever the
  cells say (plan2 §3.6), else `NOT_MET` if any cell is `worse`, else `MET`;
- a `pair_items` raise must become a printed state, not a crash.

### (d) Growth-stops check (`model-bench/AGENTS.md`)

`AGENTS.md` falls under the context-file convention ("rewritten, not appended"), not the
growth-stops bullet. Applying the same test anyway:
- It is 2,454 words (was 2,475), and no line is over 700 characters.
- **New scope:** the `decision_stats.py` invariant paragraph, which v1.3 §3.6 requires ("U3c
  adds one `model-bench/AGENTS.md` invariant line"), and the `decision_stats` entry in the module
  list.
- **Rework:** the `--negative-control` paragraph is compressed and no fact is lost. The
  duplicated record, the smoke-check purpose, newest-per-key under `--models` and oldest-first
  otherwise all survive.
- No history is restated.

### Findings

**P16-m1 (minor, implementer). A verdict cell with no paired item is missing from
`PairedItems.outcomes` and `over_limit`.** The docstring says "every metric the incumbent
declares is here". A probe (`scratchpad/p16/probe.py`) paired three understanding/`False` items.
`outcomes` held only `falseAdvanceRate` and `falseAdvanceRateByUnderstanding`, and the three
other verdict cells were absent. A U3c.2 loop over `outcomes`, or an `all(...)` over its lines,
would print three FR-1 lines and their method-minimum prints, and could return `met` without
the fourth. A loop over the four names would raise `KeyError` instead.
- **Fix:** seed `outcomes` with `()` and `over_limit` with `0` for every cell in
  `_VERDICT_CELLS_BY_ROLE[role]` before the loop. `parity_cell(())` already gives the note's
  n = 0 line.
- **Positive control:** this probe, asserting all four keys present.

**P16-m2 (minor, implementer). Part of how far the design effect reaches is untested.** Two
mutants survive the full suite:
- `mdd_reuse` computed at `n` instead of `n_effective`;
- `U` computed from a table shrunk to `n_effective`.

The reuse limit should use `n_effective` like the plain one (§Q3.3, §Q3.5). `U` should not move
with the design effect at all (§Q3.2: "only the detection limit takes `designEffect`"). Today
the effect is nil, because `designEffect = 1.0` is final for candidate runs (plan2 §3.4).
However, the API takes the argument and claims both behaviours.
- **Fix:** one test at `design_effect = 2.0`, `k = 1`. Assert that `mdd_reuse == mdd(floor(n/2),
  ψ̂, …, 0.8**(1/2))`, and that `U` equals the same cell's `U` at `design_effect = 1.0`.

**P16-n1 (nit, implementer). The α boundary is unpinned.** `p <= alpha` → `p < alpha` survives,
because `p_worse` is dyadic and never equals 0.05. That makes the mutant equivalent at
`FR1_ALPHA`, but not for the `alpha` parameter. Pin it with `parity_cell` on `(5, 0)` at
`alpha=1/32`, which should be `worse`. The other survivor, `power >= target` → `>`, is equivalent
in practice: a float power never lands exactly on the target. No action.

**P16-n2 (nit, `teco` for the U3c.2 brief). `drops` does not separate eval from side-check
items.** A load-withheld side-check item lands in the same `("incumbent", "load")` count as an
eval one (probe). U3c.2's "counts beside each bar" would then include side-check drops, which do
not affect any bar.

The stratum of a withheld item cannot be recovered from the records: the incumbent has no
scores and the candidate may have none. Separating the drops therefore means `pair_items` would
need the pack's per-item strata. Decide in the U3c.2 brief whether the bar's drop counts must be
eval-only. If they must, U3c.2 adds that argument. If not, print the caveat.

**P16-n3 (nit, implementer). HISTORY overstates one test.** It says the §Q3.3 table is
re-derived "at n = 50/75/100". The tests cover the n = 50 and n = 100 rows, but only one n = 75
cell (`(75, 0.20)`). Say so, or add the row. My independent run reproduces all four rows.

**P16-n4 (nit, `data-scientist`). Note §Q6 says "unstable across loads — not met"; plan2 §3.6
says "unstable across replicates — not met".** The code follows plan2, which is right because
same-load replicates enter the conjunction. Align the note at its next revision.

### Open questions

1. **(`teco`, U3c.2 brief)** Name the per-replicate FR-1 state rule (scored n below 200 takes
   precedence over the cells), how a `pair_items` raise is printed, and P16-n2's drop scope.

### Proposed REVIEW.md lines

None. Each finding is a single instance, already covered by the "every check can fail and
pass" line.

## Pass 17

*2026-10-31. Re-gate of the U3c.1 fix round: the uncommitted diff against `042d3838` in
`model-bench/modelbench/decision_stats.py`, `tests/test_decision_stats.py` and `docs/HISTORY.md`.
The formula source is note §Q3.*

**Verdict: approve.** All four findings sent to the implementer are fixed, and each Pass 16
mutant now goes red. The rewritten tests are stricter than before, not weaker. Seeding the empty
cells changes nothing for a cell that has items. No new findings.

**Checks run** (from `model-bench/`, with a fresh `PYTHONPYCACHEPREFIX` each time):
- **Suite and lint:** `3013 passed, 5 deselected, 2 xfailed`; `ruff check` clean.
  `tests/test_decision_stats.py` alone gives 177 passed.
- **Mutants** (`scratchpad/p17/mutate.py`): eight in all, each run against the full suite. After
  each run the file was restored and checked with `cmp`. Afterwards the `sha256` matched the
  pre-run value and `git status` showed the same three files modified.
- **Differential** (`scratchpad/p17/p17diffplug.py`, a pytest plugin): it wrapped every
  `pair_items` call in the test file and ran `042d3838`'s module on the same input. Over 21
  calls, all 25 cells that the old module populated matched exactly, comparing `outcomes`,
  `over_limit`, and `parity_cell` at DE 1.0/K 0 and DE 2.0/K 1. `latency`, `scored_n` and
  `drops` were identical. The only extra keys were verdict cells holding `()` and `0`.

CPG: considered, not relevant — no `cpg_modelbench` is loaded; the module was mutated and
executed directly.

### Dispositions

- **P16-m1: fixed.** Removing the seed from both dicts turns 15 tests red. Removing it from
  `outcomes` only gives 12 red, from `over_limit` only 4 red, and seeding every cell but the last
  14 red. The `dict.fromkeys(cells, [])` shared-list trap gives 3 red, including the new
  `test_every_verdict_cell_is_present_even_with_no_paired_item`.
- **P16-m2: fixed.** `mdd_reuse` at `n` gives 1 red, and `U` from the table shrunk to
  `n_effective` gives 1 red. Both are caught by `test_only_the_detection_limits_take_the_design_effect`.
- **P16-n1: fixed.** `p <= alpha` → `<` gives 1 red, `test_a_p_worse_exactly_at_alpha_is_worse`.
- **P16-n3: fixed.** The §Q3.3 test now has all five n = 75 cells. The expected values are the
  note's, from table line 325 (8.8, 8.8, 9.4, 13.3, 16.3), and `decision_stats.mdd` computes
  them. A fresh stdlib script (`scratchpad/p17/cell.py`, `python3 -I`) gives 9.4 at ψ̂ = 0.10
  and 16.3 at ψ̂ = 0.30. It enumerates the trinomial with an integer exact tail and shares no
  code with the module. My Pass 16 run already reproduced all four rows.
- **P16-n2, P16-n4:** not for the implementer, and unchanged. They stand with `teco` (U3c.2
  brief) and `data-scientist`.

### Fix interaction

- **The helpers do not weaken any test.** `_cells()`/`_limits()` take their keys from
  `classification.PER_KIND_VERDICT_METRICS`. That is the scorer's declaration, not
  `decision_stats`' private mapping, so a wrong key the module seeds still fails the
  exact-`dict` comparison. Every rewritten assertion keeps its earlier keys and values and now
  also requires the three empty cells, so each is stricter. No comparison became vacuous: in the
  `_cells()`-only cases, any pair that lands in a verdict cell or a pooled metric breaks
  equality. The A2/A3 mutants show those assertions still fail when the empties are missing.
- **Seeding is neutral for cells that have items.** `setdefault` appends to the seeded list and
  `.get(metric, 0)` adds to the seeded 0, so values are unchanged. The differential confirms it
  on every test input. The only behavioural difference is dict key order, with verdict cells
  first, and no consumer depends on it.

### Growth-stops check (`model-bench/docs/HISTORY.md`)

`HISTORY.md` is read by lookup, so one appended entry per delivered change is correct. The new
entry is a single 14-line block. It does not restate the U3c.1 entry, and it cites P16 IDs only.
Its claims check out: the three understanding/`False` items, the `design_effect = 2.0`, `k = 1`
test, the `(5, 0)` at `alpha = 1/32` case, every n = 75 cell, the four mutants killed, and 177
tests.

### Findings

None.

### Proposed REVIEW.md lines

None.

## Pass 18

*2026-10-31. U3c.2: the uncommitted `model-bench/` diff against `01f6f2f4`, covering
`modelbench/{cli,report,decision_stats}.py`, `tests/{test_decision_bar.py (new),
test_decision_stats.py, test_report.py}`, `README.md`, `AGENTS.md` and `docs/HISTORY.md`. Specs,
with the later plan winning: v1.3 §3.5–§3.6, §4 U3c and §5 U3c rows; plan2 §3.3, §3.6, §4.1 U3c
and §5 U3c rows; plan3 §3.3, §3.6, §4.1 U3c and its U3c test row; note v1.8 §Q3–§Q6. Out of
scope: `typed-decision-models4.md`, and `run.designEffect = 1.0` per cell (note §Q6 defers the
per-cell estimate past U5).*

**Verdict: needs changes** (one major, small fix). Every plan2 §3.6 refusal, state and print
line is present and pinned, as are plan3's `overLimit` prints and the P12-n1/P16-n2 rulings. The
`pair_items` split is sound, and the U3c.1 guarantees hold. The major: once candidates no longer
pass through `compare_report`'s `runs`, the existing pack-content guard no longer reaches them.
A bar is then stated over an incumbent and replicates measured on different pack content, with
no banner (P18-M1). There is also one minor test gap in the disagreement line (P18-m1) and two
nits.

**Checks run** (from `model-bench/`, with a fresh `PYTHONPYCACHEPREFIX` each time):
- **Suite and lint:** `3070 passed, 5 deselected, 2 xfailed`; `ruff check .` clean. The test
  counts in HISTORY are right: `test_decision_bar.py` has 42 functions, which collect as 45 tests,
  and `test_decision_stats.py` gains 12.
- **Mutants:** 26 in all, each run against the full suite. For each one the file was copied aside,
  then restored and `cmp`-checked. Afterwards, `sha256sum -c` over `modelbench/*.py` and the new
  test file matched the pre-run values. The scripts are `scratchpad/p18/mutate.py` and
  `mutate_s.py`.
  - **18 of my own, on plumbing:** 12 went red, 6 survived. The survivors are P18-m1 and
    Disposition (c). Appendix P18 lists them.
  - **8 re-run from the implementer's mutant classes, as HISTORY names them,** because the
    implementer's table is not trusted. All 8 went red: rank refusal, empty cell dropped,
    hard-coded 85, incumbent latency used as the candidate's, caveat moved after the bar, FR-2
    ignoring scored n, first run not forced to `load`, incumbent identity check removed.
- **Probes** (`scratchpad/p18/probe_mixed.py`, `probe_inputs.py`):
  - A relabel between the incumbent run and the replicates makes `pair_items` raise, because the
    declared metric follows the label. That prints `not assessable`.
  - An edit to the inputs alone does not. The incumbent ran on content `ebd24cb9` and both
    replicates on `ad5b18ed`, and the report still prints `FR-1 … not met` and `FR-2 … met` with
    no mismatch banner (P18-M1).
- **Load state** (model-bench `REVIEW.md`): `residentModelsAtStart` is captured at
  `runner.py:1843`, before `_load_absent` (1879). So `same-load` really does mean "resident
  before this run's own load", and the U7-sequence test drives it through real `run_pack`.

CPG: considered, not relevant — no `cpg_modelbench` is loaded (`GRAPHS`); the code was read,
mutated and executed directly.

### (a) Contract fidelity to plan2 §3.6

Each item below was checked line by line against the code and its test:
- **Refusals.** `--session`/`--models`, unknown key, more than one chat arm, a chat arm that is
  not the incumbent, a chat-arm run count other than one, two content hashes, different chat
  timeouts, `--negative-control` and `rank` each refuse with exit 2 and write nothing.
- **Equal off-pin timeouts.** A timeout of 120 s or `None` on every arm prints every bar
  `not assessable (chat timeout ≠ 180 s)`, which is plan2 §5's row.
- **Replicate kinds.** `assumed` / `load` / `same-load` follow the plan, and the first
  non-assumed run is a `load` even when resident.
- **K.** It is taken from the earliest run and printed once per arm. With interleaved D/H, D's K
  is 0 and H's is 1.
- **Reuse-adjusted limit.** It prints at K ≥ 1, and a printed `k=0` mutant goes red.
  `mdd_reuse is None` occurs exactly when `mdd is None`, since at n ≥ 5 the grid reaches power 1
  at δ = 1. So the `and line.mdd_reuse is not None` guard cannot hide a limit.
- **Labeler caveat.** It is the first result line.
- **Leakage, reuse and fit-outcome lines.** All three are present.
- **Per-replicate counts.** Each replicate prints `handoffShare`, `decisionCallFailures` and
  `overLimit`.
- **Scored n and drops.** The scored eval n prints beside FR-11's 200, with eval-only drops.
- **FR-1 lines.** All four print in note §Q3.4's form, each followed by its method minimum and
  per-cell `overLimit k/n`, `0/n` included.
- **FR-2.** It is `bar_holds(R, 2.0, "at-least")` with the pinned B and the pack seed.
- **Side check.** The header prints the pack's actual count (P12-n1, pinned against an 85 mutant),
  with side-check drops shown separately.
- **Across the replicates.** The disagreement line follows, then `combine_replicates` per bar.

### (b) The `pair_items` API change

The split is sound.
- `strata` is required and keyword-only, and a test pins that.
- An item with no known stratum raises. So does a scored incumbent item whose metrics contradict
  its stratum, which guards against a stale `strata`.
- Withheld items count in `drops` or `side_check_drops` by the pack's stratum alone.

The U3c.1 guarantees are intact.
- Cell seeding is unchanged, and P16-m1's four cells reach the report: the empty-cell test
  prints `0/0 [—]`, and the S2 mutant goes red.
- Concordant pairing and over-limit pairing are untouched.

Neither changed test is weaker.
- The side-check decision-failure test now asserts both maps, which is stricter.
- The renamed test keeps its assertion byte for byte. The claim its old comment made, that a
  side-check pair adds nothing to scored n, is now asserted by
  `test_a_withheld_side_check_item_is_counted_apart_from_the_bars_drops` (`scored_n == 1`).

### (c) Plumbing

These went red, each turning at least one test red:
- the session filter on the replicate group;
- K computed from the latest run, from this arm's runs only, without `evalSetHash`, or with `<=`;
- the incumbent's timeout left out of the mismatch set;
- residency keyed on `candidateId`;
- the arms swapped into `pair_items`;
- side-check drops merged into the bar's drops;
- a printed `k=0`;
- scored n `<` changed to `<=`;
- the override for an inconsistent replicate.

The survivors are P18-m1, plus one equivalent-in-effect mutant: the chat-arm count read across
sessions. Its failure is loud (a spurious refusal), never a silent wrong bar, so no action.

### (d) U4 fit

Nothing here blocks U4.
- `ReplicateGroup`, `_decision_point_arms`, `combine_replicates`, `DecisionPointPack.strata` and
  `_disagreement_line` are role-agnostic. `_final_answer` works on `agree[<label>]` outcomes.
- Two facts belong in the U4 brief:
  - `_render_decision_bar` computes `blocked` (no incumbent, timeout off the pin) only after the
    speed-up early return. The FR-10 side must hoist the timeout block, because note §Q4.3 applies
    the 180 s pin to new abilities too.
  - `pair_items` raises on a new-ability role. U4 adds its own drop accounting over the candidate
    run (plan2 §3.6).

### (e) Growth-stops (`model-bench/AGENTS.md`)

The context-file convention applies, so the rule is "rewritten, not appended".
- The file has 2,502 words (2,454 at the base) and no line over 700 characters.
- **New scope:** one invariant paragraph, which v1.3 §4 U3c requires (bar constants,
  `BAR_KIND_BY_ROLE`, the `runs`/`replicates` split).
- **Rework, done in place:**
  - The `groundingRate` "is fixed" sentence is removed. It was history, and it is in
    `docs/HISTORY.md` at line 1180.
  - The `(U1d–U2.2 in docs/HISTORY.md)` aside and `results.py:688` are removed.
  - The raw-reply paragraph is compressed. Only examples were lost.
- No history is restated. The one remaining dated clause, the P11-m2 ruling, is a live constraint
  until U7, with a citation. See P18-n2 for the budget.

### Dispositions of the implementer's judgement calls

1. **Correct, and the citation checks out.** Plan2 §3.6 "Bar states" and note §Q4.3 "Not
   assessable" put scored n first, and note §Q3.1 says "FR-1 holds iff all four cells hold".
   On a `not assessable` replicate, each cell line ends with that state.
2. **Agreed.** The exit-code policy is "0 whenever the tool ran and reported", and this answers
   P16 Open question 1.
3. **Agreed.** This is P16-m1 carried through to the print.
4. **Agreed, and plan2 §5's row says so.** Timeout precedence masks an inconsistent incumbent's
   reason, but both states are `not assessable`, so the effect is only on the wording.
5. **Literally plan2 §3.6** ("makes its group's bars not assessable").
6. **Agreed.** Plan2 §3.6 and note §Q6 set no stratum limit.
7. **Lenient against v1.3 §3.5's "stored candidate runs"** (P18-n1).
8. **Agreed.** A duplicated arm would build a duplicate group, and refusing it selects nothing.
9. **Sound, but only while the replicates ran on this pack's content.** That holds because
   `run` refuses an overlap with exit 4 and `fitManifestSize` is a required candidate field. The
   "this pack's M items" half is what P18-M1 breaks.
10. **Agreed.** `decision_point` is pack data, `replicates` is plan2's own name, and the pairing
    key is fixed by the pack.
11. **Agreed.** See (b).

### Findings

**P18-M1 (major; `coder`, and `teco` ledgers it as a U3c addition). A bar is stated over arms
measured on different pack content, with no banner.** Plan2 §3.6 passes candidates only as
`replicates`, so `compare_report`'s `PACK VERSION MISMATCH`/`PACK CONTENT HASH MISMATCH` banner
(`report.py` §"banners") now sees the chat arm alone and can never fire. Before U3c.2, the same
`compare` put the candidates into `runs`, and the comparison was flagged and labelled unpaired
(v1.3 AC-3, AGENTS "never silently mixing").
- The bar section checks no content hash. It also reads `strata`, the eval and side-check counts
  and "this pack's M items" from the pack on disk, which may be a third content.
- `pair_items` catches a label or `path` edit, because the declared metric changes. It does not
  catch an edit to the inputs. My probe edited six conditions between the incumbent run
  (`ebd24cb9`) and the replicates (`ad5b18ed`), and the report printed `FR-1 … not met` and
  `FR-2 … met` with no notice. A U3d re-import that fixes a condition typo, inside a reused
  session, reaches exactly this case.
- **Fix:** in `_render_decision_bar`, if `{incumbent, every replicate}`'s
  `fingerprint.packContentHash` plus the loaded pack's `contentHash` hold more than one value,
  set `blocked = "not assessable (arms ran on different pack content: <hash8 list>)"`. That uses
  the same mechanism as the timeout block. A cli usage refusal would also do. The implementer
  chooses which, and states it in README.
- **Assertion that catches it being wrong:** a `test_decision_bar` test built like
  `probe_inputs.py` (an input-only edit between the incumbent and replicate runs) asserts every
  bar carries that reason. Deleting the check must turn that test red.

**P18-m1 (minor; `coder`). The disagreement line is pinned only on its final-answer count.** Four
mutants survive the full suite:
- `decidedBy` never compared (the test expects `0/200` either way);
- the one-sided-withhold count dropped (expected `0`);
- `assumed` replicates included in the comparison;
- only replicate 2 compared, which matters from 3 replicates on.

A fifth survivor is in the per-replicate block: side-check metrics also printed as `(pooled)`
eval lines. That puts exploratory side-check rates in the eval section.

Plan2 §3.6 specifies all four disagreement behaviours (note §Q6's diagnostic).
- **Fix:** extend `test_the_disagreement_line_compares_each_replicate_with_the_first` with one
  handed-off-versus-decided item (`decidedBy` 1/n) and one item withheld on one side only. Add a
  3-replicate case with an `assumed` middle replicate, asserting `replicate 3 vs replicate 1` and
  no `vs` line for the assumed one. In `test_a_side_check_item_scores_exploratory_lines_only`,
  assert that no `sideCheck… (pooled)` line exists. Each of the five mutants in Appendix P18 must
  then turn red.

**P18-n1 (nit; `coder`). K is a lower bound when the pack has invalid records.**
`_replicate_groups` reads only `valid`, but v1.3 §3.5 counts "stored candidate runs". A record
excluded as invalid still looked at the evaluation set, so K, and with it the reuse-adjusted
limit, can be understated. The `INVALID RESULTS EXCLUDED` banner does print, but far below the
K line.
- **Fix:** when `invalid` is non-empty, append `(valid records only; N invalid excluded)` to the
  K line, and pin it with one test. Counting invalid records' envelopes would need
  `InvalidRecord` to carry its fingerprint, which is more than this nit is worth.

**P18-n2 (nit; `coder`). `model-bench/AGENTS.md` is 2 words over its ~2,500 budget, and its new
paragraph ends on rationale.** The paragraph's last clause ("a withheld record carries nothing
to tell its stratum by") restates `pair_items`' docstring. Drop it; that brings the file back
under budget.

### Proposed REVIEW.md lines

Target `model-bench/REVIEW.md`, `## Always check`. Source: P18-M1. U4 reuses this path, so the
line changes what U4's reviewer checks:

- A change that routes runs around `compare_report`'s `runs` (as `replicates`, or as any new
  argument) shows that every guard the `runs` path applies still reaches them: the pack-version
  and content-hash banners, aggregate consistency, and the schema banner. Each one is pinned by a
  test on the rerouted runs.

### Appendix P18 — mutants

| Mutant | Result |
|---|---|
| M1 replicate-group session filter dropped | red (≥5) |
| M2 K from the latest run | red (1) |
| M3 K ignores `evalSetHash` | red (1) |
| M4 K counts a same-`startedAt` run | red (1) |
| M5 K from this arm's runs, not history | red (≥4) |
| M6 chat-arm runs counted across sessions | survived (loud failure only; no action) |
| M7 chat timeout left out of the cli mismatch set | red (1) |
| M8 resident-at-start keyed on `candidateId` | red (2) |
| M9 `pair_items` arms swapped | red (≥5) |
| M10 bar drops include side-check drops | red (1) |
| M11 printed `k=0` in the FR-1 line | red (1) |
| M12 `assumed` replicates in the disagreement line | **survived** (P18-m1) |
| M13 disagreement for replicate 2 only | **survived** (P18-m1) |
| M14 scored-n `<` → `<=` | red (≥4) |
| M15 inconsistent replicate does not override the group | red (1) |
| M16 `decidedBy` never compared | **survived** (P18-m1) |
| M17 one-sided-withhold count dropped | **survived** (P18-m1) |
| M18 pooled lines take side-check metrics | **survived** (P18-m1) |
| S1–S8 (implementer classes, listed under Checks run) | red (1–2 each) |

## Pass 19

*2026-10-31. Re-gate of the U3c.2 fix round. It is still uncommitted on `01f6f2f4` and touches
`modelbench/report.py`, `tests/test_decision_bar.py`, `README.md`, `AGENTS.md` and
`docs/HISTORY.md`. Its delta was read against Pass 18's saved `report.py` snapshot; `cli.py` is
byte-identical to that snapshot.*

**Verdict: approve with suggestions.** Every Pass 18 finding is fixed, and all five Pass 18
survivors now go red. Two new findings, both about tests of the content guard:
- one minor: the coordinator's `g.runs[0]` mutant is **not** equivalent, and it survives
  (P19-m1);
- one nit: nothing pins the guard winning over earlier reasons (P19-n1).

**Checks run** (from `model-bench/`, with a fresh `PYTHONPYCACHEPREFIX` each time):
- **Suite:** `3076 passed, 5 deselected, 2 xfailed`.
- **Mutants:** 11 in all, from `scratchpad/p19/mutate19.py` and two one-off copies. Each was
  copied aside, restored and checked with `cmp`. Afterwards `sha256sum -c` over `modelbench/*.py`
  and `tests/*.py` matched the pre-run values.
  - The five Pass 18 survivors: all red.
  - N2, the loaded pack's hash left out: red (1).
  - N3, a content-blocked replicate still paired: red (3).
  - N4, the K line's invalid note dropped: red (1).
  - The `pair_items` catch narrowed to `KeyError`: red (1).
  - Survivors: N1, the content guard losing to earlier reasons (P19-n1), and C1, `g.runs[0]` per
    group (P19-m1).
- **Probe** `scratchpad/p19/probe_c1.py`. The incumbent and replicate 1 ran on the pack's
  content, and replicate 2 on input-edited content.
  - Unmutated: every bar prints `not assessable (arms ran on different pack content: a186a681,
    ebd24cb9)`.
  - Under C1: `FR-1 … met` and `FR-2 … met`, exit 0.

CPG: considered, not relevant — no `cpg_modelbench` is loaded; the code was mutated and executed
directly.

### Dispositions

- **P18-M1: fixed.** `_pack_content_reason` is applied last in `_render_decision_bar`, and a
  `different_content` block skips pairing. Three tests cover it: the input-edit probe, a
  replicate-only mismatch, and the pack edited after every arm ran. N2 and N3 go red. Coverage
  across replicate positions is incomplete (P19-m1).
- **P18-m1: fixed.** M12, M13, M16, M17 and M18 each go red. The 4-replicate test (load, assumed,
  load, load) is needed for M13.
- **P18-n1: fixed.** The K line ends `(valid records only; N invalid excluded)`; N4 goes red.
- **P18-n2: fixed.** The clause is dropped. `AGENTS.md` is 2,492 words, with no line over 700
  characters.

### Interactions

- **Content guard against an inconsistent replicate.** In `_render_replicate`, a replicate's own
  state (`LOAD_CHANGED`, `_REPLICATE_INCONSISTENT`) still takes precedence over `blocked`. So an
  inconsistent replicate's group prints `not assessable (replicate i: aggregates disagree …)` on
  its overall lines, while its sibling replicates print the content reason. Both are
  `not assessable`, and a replicate is never paired across contents. It affects only the wording.
  No action.
- **Content guard against the off-pin timeout, an inconsistent incumbent and no incumbent.** The
  guard overwrites all three. That is the right order: those states still pair and print figures,
  and the content state must not. N1 shows nothing pins it (P19-n1).
- **Block or refuse.** Plan2 §3.6 refuses selection-shaped problems: the arm set, mixed candidate
  hashes, mismatched timeouts. It states the property of a stored record as `not assessable`: an
  equal off-pin timeout, `assumed`, inconsistent aggregates.
  - A pack edited after every arm ran is a record property that no arm selection can fix, so
    stating it is the consistent choice.
  - An arm-against-arm mismatch is closer to the timeout-mismatch refusal the coder cites. Either
    choice was acceptable (P18-M1's fix offered both).
  - Putting one rule in `compare_report` means every caller gets it. Accepted.
- **The pairing-error test now forges `packContentHash`.** That is the only way to reach the path
  now. A runner validates the pack before running, so a stratum-less pack never produces records.
  An item-set difference, the other natural trigger, now implies a content difference, which the
  guard catches first. The `pair_items` catch is therefore defensive. It still deserves its test,
  and the `KeyError`-narrowing mutant shows the test is live. The test's comment states the
  forgery. Accepted.

### Growth-stops

- **`model-bench/AGENTS.md`** is under the context-file convention. It is 10 words shorter than
  in Pass 18, and the only change is the removed clause.
- **The content guard's home.** The guard is user-visible report behaviour, so README is the
  right home. Its constraint lives in the `_pack_content_reason` and `DecisionPointPack`
  docstrings. AGENTS already states the `runs`/`replicates` split, which is the fact that bites.
  Agreed.
- **`docs/HISTORY.md`** is read by lookup. Folding the fix round into the still-uncommitted U3c.2
  entry is right, since it is one delivered change. Its claims check out against the code and the
  runs above, including 51 tests in `test_decision_bar.py`. One caveat: "replicates … left out
  of it" names the coder's whole-set mutant, which goes red, not C1.

### Findings

**P19-m1 (minor; `coder`). The content guard is not pinned for a later replicate.** Under C1
(`runs = [g.runs[0] for g in replicates.values()]`) the full suite stays green. The equivalence
argument does not hold:
- `_replicate_groups` refuses two **`candidateContentHash`** values in a group;
- it does not refuse two **`packContentHash`** values.

`test_a_replicate_on_other_content…` puts the odd content in replicate 1, which C1 still reads.
The probe puts it in replicate 2, and C1 then prints `met` over mixed content.
- **Fix:** add the mirror case. The incumbent and replicate 1 run on the pack's content, and
  replicate 2 on edited content (`probe_c1.py`'s shape), then `_assert_every_bar`. C1 must then
  turn red.

**P19-n1 (nit; `coder`). "Applied last, so it wins" is untested.** N1 survives the full suite:
`blocked = blocked or _pack_content_reason(…)`. Under it, an off-pin timeout plus a content
mismatch prints the timeout reason. It also pairs items across the two contents, and their
figures print. The states stay `not assessable`.
- **Fix:** one test with equal off-pin timeouts on every arm and an input edit between the arms.
  Assert the content reason on every bar. N1 must then turn red.

### Proposed REVIEW.md lines

None new. Pass 18's proposed line (guards still reach runs rerouted around `runs`) covers P19-m1's
class.

## Pass 20

*2026-10-31. Gate of U2.3 (plan4 §3.10, §5 `| U2.3 |`, R-21): between-item probe failures.
Uncommitted on `827e953e`, in `model-bench/`: `modelbench/runner.py`, the new
`tests/test_runner_probe_failures.py`, `docs/BACKLOG.md`, `docs/HISTORY.md` and `AGENTS.md`.
Read against plan4 §3.10, its U2.3 row and R-21, and SMB plan §3.6 clause (iv) together with its
fourth-disposition paragraph.*

**Verdict: approve with suggestions.** The helper and its four call sites match §3.10 on every
surface, and every wiring point I mutated goes red except one. There are four findings and no
blocker or major:
- two minor findings about R-21's wording: the BACKLOG item (P20-m1, `tdd-engineer`) and plan4
  itself (P20-m2, `architect`);
- two nits (P20-n1, P20-n2, `tdd-engineer`).

**Checks run** (from `model-bench/`):
- **Suite:** `3111 passed, 5 deselected, 2 xfailed`. `ruff check .` is clean.
- **Mutants:** 21, all from `scratchpad/p20/p20_mutate.py` in a private directory. Each one ran
  the full suite with its own `PYTHONPYCACHEPREFIX`, and the file was restored and checked with
  `cmp` after each. At the end, `runner.py`'s sha256 was `29b3be139de0dca8…`, the same as at the
  start.
  - **Mine (13).** 12 went red:
    - v1-only failing open: 1 red;
    - exit 5 instead of 3: 6;
    - a fixed `"unreachable"` message: 1;
    - the stderr line sent to stdout: red;
    - the line printed twice: red;
    - `residency()` retried instead of `[]`: red;
    - the chat, handoff or tool-caller site reverted to a bare `residency()`: red at each;
    - the decisions site's label without the item id: red;
    - the tool-caller label keyed on `shape` instead of `scriptId`: red;
    - the exception class dropped from the line: red.
    The one survivor is A5, a catch widened to `except Exception` (P20-n1).
  - **Re-sample of the implementer's table (8).** All 8 went red:
    - treat-as-resident at the chat site: 4;
    - failing open: red;
    - refusing every failure: red;
    - catching `LMStudioUnreachable` alone: red;
    - catching `TimeoutError` alone: 24;
    - the load rule over a failed chat call: red;
    - the load rule over a failing turn: 4;
    - a run kept on an outage: 6.
- **Coverage (c).** All 33 tests call the real `run_pack`, or `cli.main` for the store check,
  against subclasses of the existing `StubLMStudio`, `DecisionStub` and `HandoffStub`. None of
  them stubs `_between_item_residency`, `probe` wiring or a driver. The stubs replace only
  `residency()`'s sequence and `probe()`'s answers, and they count `probe_calls`.

CPG: considered, not relevant — no `cpg_modelbench` is loaded; the code was read, mutated and
executed directly.

### Contract fidelity

- **Chat and embeddings** (`_drive_single_call_items`). A `[]` snapshot reaches the existing
  `_load_withheld_for` only on the completed path. The `LMStudioCallTimeout` and
  `LMStudioCallFailed` branches `continue` before any load producer runs, which is SMB §3.6's
  precedence. B5 pins this.
- **`_turn_timings`.** A failing disposition still returns before the load expression, so the
  same precedence holds there, and B6 pins it. **The tool-caller snapshot is per conversation:**
  the snapshot taken after conversation *k* guards every turn of *k+1*. A failed probe after `s1`
  therefore withholds `s2`'s turns for `"load"`, and the test asserts exactly that item. The
  label names the conversation the snapshot follows, which is the only one `_turn_timings`
  knows. The snapshot after the last conversation guards nothing (P20-n2).
- **Decisions.** Both the kept-failure branch and the success branch read
  `_load_withheld_for(model_info, resident)`, so item 2 is `"load"` either way. The load rule
  then gives `"assumed"` from the `"load"` item. I checked `827e953e`'s `runner.py` lines 598 and
  628: these lines are unchanged.
- **Handoff.** `_models_withheld_for(models, [])` is `"load"`, so `snapshot_lacked_a_model` flips.
  Consider a guard item whose hand-off chat call fails after a failed probe. It keeps its own
  `"timeout"` or `"no_response"`, as SMB precedence requires, and the flag still makes the run
  `"assumed"`. I traced this by reading; no test covers this combination, and that is acceptable
  because the flag's logic is pre-existing.
- **Outage.** Any value other than `"api-v0"` raises `RunRefused(_PROBE_MESSAGES[...], exitCode=3)`
  with step 2's exact text, and `from exc`. `cli.py:750-752` prints it once and returns the code.
  `run_pack` stores nothing, and the `cli.main` test asserts that no `results/` directory
  appears. **The right instance:** `run_pack` has one `lmstudio`, and the helper takes it from
  each driver's own parameter. The `probe_calls == 2` assertion (step 2, then the re-probe) would
  catch a probe of anything else.

### Judgement calls

1. **No stored field changes.** I agree. At `827e953e`, a kept decision failure already takes
   `withheldFor` from the snapshot (lines 598 and 787/877 for handoff), so `[]` reaches it with
   no new field.
2. **The extra `before: str`.** I accept it as a deviation from §3.10's one-argument signature.
   The helper cannot otherwise name the item, and the plan requires the item to be named. On
   `tool-caller`, "the conversation after 's1'" is accurate, because the next script is not in
   scope at `_turn_timings`.
3. **"Stores nothing" asserted through `cli.main`.** This is right, because only `cli` stores.
   The other three surfaces assert the `RunRefused` exit code and message on `run_pack`, which
   is enough for them.
4. **The two forms of skipping the re-probe.** Both are real, and they are distinct: failing
   open is §3.10's rejected option 2, and refusing on any failure is rejected option 1. Both go
   red in my re-run.
5. **BACKLOG drops "traceback".** The claim about `_raw_post` is correct. I checked
   `lmstudio.py:728-760`, where every rung maps to `LMStudioCallTimeout` or
   `LMStudioCallFailed`; a refused connection is a `URLError` (rung 3) and becomes
   `LMStudioCallFailed`. Every item driver catches both classes, so plan4 R-21 is wrong about
   item calls. However, a traceback path does exist outside item calls, and neither document
   names it (P20-m2).
6. **The short HISTORY entry.** I agree with keeping it short. The "33" count and the mutant
   classes match the file and my re-sample.

### Growth-stops

- **`model-bench/AGENTS.md`** is under the context-file convention. It has 2,495 words and no
  line over 700 characters. The new paragraph is new scope: a "what will bite me" invariant
  with a plan citation. Three passages were compressed to offset it. Each fact they dropped is
  in `docs/HISTORY.md`, so these are moves, not losses:
  - the qwen-vs-ministral sample-size result, at lines 1322-1327;
  - `EXIT_NOTESTSCOLLECTED`, at line 5399;
  - "8 collection errors, exit 2", at line 5422.
  No finding.
- **BACKLOG.** The P8-m2 item is replaced in place, not appended to, and the delivered P8-m2 is
  recorded in the new HISTORY entry. Its content is inaccurate in two places (P20-m1).
- **HISTORY.md** is read by lookup and has one new entry. No finding.

### Findings

**P20-m1 (minor; `tdd-engineer`). The R-21 BACKLOG item understates what SMB §3.6 already
rules.** The item says that clause (iv) "asks for a re-probe after a timeout" and that the gap
"needs an `architect` ruling on which item-call failures re-probe". But SMB §3.6's
fourth-disposition paragraph already rules both cases for scored chat calls:
- a *connection* failure "re-probes exactly as clause (iv) specifies";
- an error response "does not re-probe and does not exit 3".
`LMStudioCallFailed.status is None` separates the two. Only the decisions and handoff kept
failures, and the multi-turn `tool-caller` turn, are unruled.
- **Fix:** rewrite the item in place. It should state that, for chat and embeddings, SMB rules a
  re-probe after a timeout or a `status is None` failure, and that no driver implements it. The
  `architect` ask should narrow to the decision surfaces and the turn. Add P20-m2's `prime`
  path if `architect` brings it under R-21. (Verified against the SMB plan text and
  `lmstudio.py`.)

**P20-m2 (minor; `architect`). Plan4 R-21's "or a traceback" is wrong for item calls, and the
real traceback path is unnamed.** P20 judgement call 5 shows that every item call degrades to a
scored or kept failure. An LM Studio outage can still escape as a traceback, though, through the
embedder's `prime` hook:
- `runner.py:470` calls `prime` with no `try`;
- an uncached corpus reaches `lmstudio.embed` at `scoring/retrieval.py:341`;
- nothing in `run_pack` catches the resulting `LMStudioCallFailed` or `LMStudioCallTimeout`;
- `cli.py:750` catches only `RunRefused`.
I traced this by reading and did not reproduce it.
- **Fix:** correct R-21 in place. Either drop "or a traceback", or name `prime`'s corpus embed as
  the one call that has one, and rule whether it joins the gap U2.3 filed. A probable shape is a
  re-probe that exits 3 on an outage, like the warm-up. That is `architect`'s call.

**P20-n1 (nit; `tdd-engineer`). Nothing pins that the helper lets other exceptions through.**
A5 (`except Exception as exc:`) survives the full suite with 3111 passed. Under A5, a defect
that raises from `residency()` would be re-probed and silently become `[]`, a `"load"`
withholding, instead of a traceback.
- **Fix:** in `tests/test_runner_probe_failures.py`, add one chat case whose item-2 snapshot is a
  `RuntimeError`. Assert that it propagates out of `run_pack` and that `probe_calls == 1`. A5
  must then go red.

**P20-n2 (nit; `tdd-engineer`, take or leave). The `tool-caller` snapshot after the last
conversation guards nothing.** `_turn_timings` probes after the last determinism-probe script,
and in the tests after `s3`. If that probe fails while the server is up, the stderr line says
"before the conversation after 's3' … so that item counts as not shown resident", and there is
no such conversation or item. If the server is down, the run exits 3 here instead of at step
10's exit 5. Both outcomes store nothing, so the only effect is the wording.
- **Fix:** make the message neutral ("what follows counts as not shown resident"). Alternatively,
  leave it as is.

### Proposed REVIEW.md lines

None. P20-n1's class, a narrowed catch with nothing pinning its narrowness, is already covered
by root `REVIEW.md`'s rule that every check can both fail and pass.

## Pass 21 — U4.1 new roles, tables, renderers, parses, vectors (2026-11-01)

*2026-11-01. Gate of U4.1, uncommitted on `d7c7465a`: model-bench `modelbench/{roles,decisions,packs}.py`,
the new `scoring/agreement.py`, tests, 5 new vectors and 6 new `input-hash.jsonl` rows, `AGENTS.md`/`README.md`/`HISTORY.md`;
falkor-chat `scripts/generate_guard_decision_vectors.py`, `server/tests/test_decision_contract.py`
and the new `normalize-turns.jsonl` copy. Read against v1.3 §3.2, §3.2.1, §3.2.2, §3.5, §3.6, §4 U4 and
the `| U4 |` rows; plan2 §3.1, §3.3, §4.1 U4; plan3 §4.1 U4; plan4/plan5 where they name U4; note
§Q1.5; the ledger's U4.1–U4.3 rows and its "U4 brief" lines.*

**Verdict: approve with suggestions.** The tables, renderers, parses and hash rows match the plans,
and every vector matches an independent check. There is no blocker or major. There are two minor
findings: a plan deliverable that no U4.x row owns (P21-m1, `teco`), and a renderer rule that its
vectors don't pin (P21-m2, `coder`). There are three nits.

**Checks run:**
- **model-bench** (from `model-bench/`): `3275 passed, 5 deselected, 2 xfailed`, and `ruff check .`
  is clean.
- **falkor-chat** (from `server/`): `tests/test_decision_contract.py` alone gives `131 passed`. That
  is the review-safe subset; the tests that wipe `reference` were not run.
- **Generator re-run** into a scratch dir: `render-guard`, `handoff-parse-guard` and
  `normalize-turns.jsonl` are byte-identical to model-bench's canonical copies, and all three
  falkor-chat copies are `cmp`-equal to model-bench's.
- **Mutants:** 6 of mine, each on a live file copied aside, run against the full suite and
  restored with `cmp`. The final sha256 over `modelbench/` and `tests/` matches the starting one.
  Results, by number of failing tests:

  | Mutant | Result |
  |---|---|
  | M1 `input_hash` writes an absent field as `null` | 3 red |
  | M2 message-type `priorTurn` speaker re-derived by the row rule | 6 red |
  | M3 join-in new message's speaker skips `authorId` | **survives** (P21-m2) |
  | M4 `ensure_ascii=True` | 7 red |
  | M5 one `expected` edited in model-bench's `normalize-turns.jsonl` | 1 red in model-bench, 1 red in falkor-chat |
  | M6 non-string new-message text passed through `str()` | **survives** (P21-n3) |
- **Delegate's 58 mutants:** I read `scratchpad/u41/mutant_results.json`. Every mutant went red,
  and all are restored.

CPG: considered, not relevant — a `cpg_falkorchat` CALL lookup for `_recent_turns` returned no rows,
so I used grep, which finds one caller (`guards.py:244`); no model-bench CPG is loaded, and the code
was read and executed directly.

### Rulings (a)–(d)

- **(a) Canonical item form, guard exempt: sound.** v1.3 §3.5's `if k in item` stays the guard's
  rule: `input-hash.jsonl`'s absent-is-omitted row pins it, and M1 turns it red. For new-ability
  items, `validate_pack` makes the rule a no-op. decision-models agrees with that form:
  `points.input_field_errors` demands "exactly the input fields", and its message-type generator
  writes `priorTurn: null` for an opening message. So both sides hash the same canonical form. The
  guard's only path-dependent items are its human-curated side-check rows (`cs-07`).
- **(b) `ANSWER_SPACE_BY_ROLE`, `MESSAGE_TYPE_LABELS`: accept.** They put v1.3 §3.2's answer-space
  column into data. A table is needed so that the per-class `verdictMetrics` set derives from the
  role. If it derived from the pack's own `answer`, a pack could choose its own classes. The table
  also adds a refusal to the guard pack; the guard contract fixture still passes it.
- **(c) Import of the private `_normalize_turns`: accept.** It is in the same package, and one
  transcription of `_recent_turns` is better than two. `normalize-turns.jsonl` now pins it. The
  model-bench copy does not tolerate non-Mapping rows, which is the item domain the generator
  docstring excludes.
- **The four carried items are all delivered:**
  - Pass 6: `test_a_guard_candidate_on_a_join_in_pack_refuses_on_role_before_any_call` asserts that
    no load call and no chat call happen. The delegate's mutant 57 turns it red (2 tests).
  - P13-n1: see (d).
  - The exact per-class refusal: 11 refused shapes and 2 accepted orders.
  - P2-F3: `normalize-turns.jsonl` (see the vector checks below).
- **(d) Keyed on the bar kind: accept.** The P13-n1 test re-keys join-in to speed-up and goes red on
  the pair. The delegate's mutant 49 turns it red.

### Vector provenance

- **Hand-written vectors.** `write_hand_vectors.py` imports only `json`, `sys` and `pathlib`, and its
  expected strings are literals. I wrote a separate implementation from v1.3 §3.2.1 and note §Q1.5
  text alone. It agrees on all 18 render rows, including:
  - every 400 and 2000 cut, tested on both sides of its boundary;
  - every prior turn filtered out, which omits the block;
  - filtering before taking the last 6;
  - an agent speaker on `priorTurn`.
- **Hash rows.** I recomputed all 10 `input-hash.jsonl` rows with stdlib `hashlib` over v1.3 §3.5's
  formula, and all 10 match.
- **Gap.** One speaker branch of the new message is not pinned (P21-m2).

### Findings

**P21-m1 (minor; `teco`). Two U4 obligations are in no U4.x row.**
- **The two hand-off prompts.** v1.3 §4 U4 places "the two hand-off prompts as pack data" in U4
  (§3.2's `prompts/join-in-handoff.md` and `prompts/message-type-handoff.md`). No later plan
  reassigns them, and U4.1 did not write them. The ledger's U4.2 row (predicate/choice,
  fail-closed, `reason`, `perClass`) does not name them, and neither does the U4.3 row. The
  `handoff` surface sends a pack's `systemPrompt`, so without these prompts no new-ability pack can
  run that surface. v1.3 §3.2.2 lists hand-off prompts among the copies falkor-chat (U9) must keep
  equal.
- **The new-ability item checks.** `_classification_item_problems` (`packs.py:1242`, P12-m1) runs
  only when `scorer == "classification"`. So nothing refuses a join-in item whose `expected` is not
  a bool, or a message-type item whose label is outside the answer space, and a run would raise
  partway through. The U4.2 row does not carry this check.
- **Fix:** add both to the U4.2 row: the prompts as pack data, plus an `agreement` item check that
  mirrors P12-m1. Or name another owner for the prompts and state their copy rule for U9.
- **Evidence:** I grepped all five plans and the ledger rows. Verified.

**P21-m2 (minor; fresh `coder`). The new message's `authorId` fallback is not pinned.**
- The `render-join-in.jsonl` case "speaker rule: … on prior turns and the new message" gives its
  new message no `displayName` and no `authorId`, so only the `member` branch runs.
- M3 replaces the rule with `displayName or "member"`, and the suite still passes (3275).
- So a U9 or U5.2b-render port that skips `authorId`, or treats an empty `displayName` as a value,
  would pass these vectors. model-bench `REVIEW.md` requires the vectors to cover every rule the
  docstring claims, and `render_join_in_input`'s docstring claims "the same speaker rule".
- **Fix:** add one hand-written row whose `newMessage` is
  `{"displayName": "", "authorId": "u9", "text": "…"}`, expected to render as `u9: …`. M3 must then
  go red. Land the row before U5.2b-render copies the vectors, so it needs no re-copy.
- **Evidence:** executed.

**P21-n1 (nit; `teco`). The behavioural pins for mutants 5 and 22 belong in the U4.2 and U4.3
briefs.**
- Exact-table pins are acceptable until then. They are hand transcriptions, they can fail, and they
  don't read their expected values from the table.
- Plan2 §3.3 says the value pin is the kept-versus-withheld test.
  - **U4.2 brief:** run P9-m3's real-scorer test over *both* new-ability roles, with
    `HANDOFF_FAILURE_BY_ROLE` unpatched. Run on join-in alone, the test leaves mutant 5
    (message-type → `withhold`) to the exact-table pin.
  - **U4.3 brief:** the FR-10 section on a message-type pack prints the `item` unit, which turns
    mutant 22 red.

**P21-n2 (nit; `coder`). Stale text.**
- `README.md:227` still says a pack's `role` is "one of the five FR-21 roles" and lists five;
  `ROLES` now has seven.
- `tests/test_roles.py:90` and `:125` quote `UnknownRole`'s old docstring, "There is no sixth".
- **Fix:** edit both in place.

**P21-n3 (nit; `coder`, take or leave). A non-string new-message text renders as `""`.**
`render_join_in_input` has this rule, which no plan states, and M6 survives it.
`render_message_type_input` raises on the same input. decision-models' schema forces a non-empty
string. **Fix:** drop the coercion, so a malformed item raises in the same way in both renderers.
Alternatively, pin the rule with a vector row.

### Growth-stops

- **`model-bench/AGENTS.md`** has 2,488 words and no line over 700 characters.
  - **New scope:** the role-tables sentence and the `validate_pack` sentence are rewritten in place.
  - **Compressions:** three passages were compressed, and none loses a live constraint.
    - `test_package.py` drops "Keep it: pytest exits 5…". Its rationale is in `HISTORY.md:5444-5449`,
      and a 3,275-test suite cannot be empty.
    - The sentence-scoping paragraph and the `sessionId` paragraph each keep their rule and their
      failure mode.
- **`HISTORY.md`** is read by lookup and has one entry. Its figures reproduce: 3275/5/2, 131, six
  hash rows, and 58 mutants all red.

### Proposed REVIEW.md lines

None. P21-m2 is covered by model-bench `REVIEW.md`'s vector-coverage line.

## Pass 22 — U4.2 agreement scorer (2026-11-01)

*2026-11-01. Gate of U4.2, uncommitted in `model-bench/` on `1554d560`. Files:
`scoring/agreement.py`, `decision_stats.agreement_items`/`AgreementItems`,
`packs._new_ability_label_problems`, the two new `packs/*-synthetic/prompts/*-handoff.md` files,
tests, and `AGENTS.md`/`README.md`/`HISTORY.md`. Read against v1.3 §3.2, §3.4 (importer), §3.6;
plan2 §3.3, §3.6, §4.1 U4; plan3 §3.3 and §4 U4; plan4 §3.9; note §Q4.3; and the ledger's U4.2 row
and "U4 brief"/"U4.2 brief" lines, including P21-m1 and P21-n1. `data-scientist` reviews the
prompts' wording; this pass does not.*

**Verdict: approve with suggestions.** The scorer follows plan2 §3.3 and plan3 §3.3 on every record
state I traced. Both carried items are met:
- **P21-m1:** the hand-off prompts are pack data, and an `expected` outside the answer space is
  refused.
- **P21-n1:** the kept-versus-withheld tests run on both roles with the table unpatched. Moving
  message type's `HANDOFF_FAILURE_BY_ROLE` row to `withhold`, which was 1 red at U4.1, is now
  7 red.

There is no blocker and no major. Two findings are minor: an unowned scorer refusal (P22-m1) and a
`parseFailures` test that one mutant slips past (P22-m2). There are two nits.

**Checks run** (from `model-bench/`):
- **Suite:** `3371 passed, 5 deselected, 2 xfailed`; `ruff` is clean; `AGENTS.md` has 2,485 words
  and no line over 700 characters.
- **Mutants:** 6 of mine, on live files copied aside, each run against the full suite and restored
  with `cmp`. The final sha256 over `modelbench/`, `tests/` and `packs/` matches the starting one.
  Results, by number of failing tests:

  | Mutant | Result |
  |---|---|
  | D1 `decision_input` drops join-in's prior turns | 1 red |
  | D2 `detail.answer` set to the raw answer, not the fail-closed final | 9 red |
  | D3 `scored_n` counts load drops | 1 red |
  | D4 `parseFailures` counts outcome `"fail"` | **survives** (P22-m2) |
  | D5 `agreement_items` walks items in reverse `itemId` order | 2 red |
  | message type's `HANDOFF_FAILURE_BY_ROLE` row → `withhold` | 7 red |
- **Scorer probe:** a join-in contract pack declaring each other scorer, run through
  `validate_pack` (P22-m1).
- **`validate`** on the prompts-only directory gives a clean "pack.json is absent" refusal.

CPG: considered, not relevant — no model-bench CPG is loaded; the code was read and executed
directly.

### Delegate calls (a)–(g)

- **(a) An over-limit hand-off failure has `inputTokens=None`: right.** plan3 §3.3's table says
  "Every row has `detail.probability = None` and `detail.inputTokens = None`", and that table
  includes the fail-closed failure column. Ruling (b)'s `int|None` covers the case where a hand-off
  follows a decision call that returned. plan3, the later plan, narrows it for over-limit, and
  `_decision_detail` does exactly that.
- **(b) Reusing `ClassificationAggregates`: accept.** `results._aggregates_from_dict` dispatches on
  `kind` alone. No `report.py` path treats `kind == "classification"` as meaning the guard; I
  grepped every `isinstance(… Aggregates)` call.
- **(c) `detail.answer` on every record: accept.** D2 shows the disagreement line's input is pinned.
  One exception: a chat-arm record withheld before any reply carries no `answer`. That record has
  `scoreable={}`, so nothing reads it.
- **(d) Message type's `None` from a parsed reply is a disagreement: accept.** v1.3 §3.2 says
  "otherwise there is no label", and note §Q4.3 scores no label as a disagreement. It stays outcome
  `"pass"`, so it is not counted as a parse failure.
- **(e) The chat arm keeps unanswered calls withheld: accept.** Note §Q4.3 fails a call closed only
  for the new ability's *hand-off* (candidate) call. The reference arm is no bar's input.
  `agreement_items` raises on a `timeout` or `no_response` withhold, so the chat arm cannot be fed
  in by mistake.
- **(f) Drops in `decision_stats.agreement_items`: accept.** plan2 §3.6 calls for "the same drop
  accounting over the candidate run", and `pair_items` keeps the guard's in the same module. The
  function matches note §Q4.3's FR-10 columns:
  - only `load` drops;
  - a kept failure's `y_i` is its elapsed time;
  - an over-limit item is scored on the hand-off answer, or fail-closed if the hand-off fails.
- **(g) Prompts in `packs/<role>-synthetic/prompts/`: no collision today.**
  - Nothing enumerates `packs/`: `cli` resolves `packs/<id>/pack.json` by id, and the
    `test_packs.py` reads name their packs.
  - `results/` is untouched, so the frozen-record test is unaffected.
  - `validate` refuses the directory cleanly.
  - **The collision is in the future, in U3d's brief.** `HISTORY.md` now promises that U3d's
    importer writes `pack.json` beside `prompts/` and points `prompt.systemPrompt` at the prompt
    file, and that U9 byte-copies each prompt with a guarded equality test. No plan or ledger row
    carries either promise (P22-n1).

### Findings

**P22-m1 (minor; fresh `coder`). A decision-point pack can declare another role's scorer.**
- My probe re-roled the contract pack as join-in and set its `scorer` to `classification`,
  `grounding` and `extraction` in turn. `validate_pack` returns `[]` for all three.
- On message type, `classification` is refused only by accident: its bool-`expected` check fires.
- Under v1.3 §3.6 the role fixes the scorer ("Their scorer is a new `scoring/agreement.py`"), on
  the same principle as "a pack cannot pick its own bar".
- What a run does with a wrong scorer is inferred, not traced. It either raises partway through or
  stores guard-cell metrics, which `agreement_items` later rejects.
- **Fix:** in `_decision_point_problems`, refuse a decision-point pack whose `scorer` is not its
  role's: `guard-judge` → `classification`, the new abilities → `agreement`. Key the mapping on the
  role as a pinned table, with a test per role and a planted wrong scorer that must turn it red.
- **Why it is a finding:** it comes from the same contract this unit extended, and it is a
  one-function change.

**P22-m2 (minor; fresh `coder`). One mutant slips past the `parseFailures` test.**
- Under D4, `aggregate` counts outcome `"fail"` instead of `"parse_failure"`, and all 3371 tests
  still pass.
- The test that should catch it,
  `test_aggregate_counts_parse_failures_and_keeps_withheld_items_out_of_every_class`, pairs one
  parse failure with one `timeout`-withheld item whose outcome is `"fail"`. Both counts come to 1.
- A kept failed call is also outcome `"fail"`. So the printed "replies the scorer could not parse"
  count could fold in kept failures and no test would notice.
- **Fix:** add a kept failure (`failed_call="handoff"`, outcome `"fail"`) to that test, or make the
  withheld item `no_response`, and assert `parseFailures == 1`. D4 must then turn it red.

**P22-n1 (nit; `teco`). Two promises in `HISTORY.md` have no owner in the ledger.** Add them to the
U3d brief and the U9 brief:
- U3d writes `pack.json` into the existing `packs/<role>-synthetic/` (so it must not create the
  directory exclusively), names `prompts/<point>-handoff.md` as `prompt.systemPrompt`, and does not
  overwrite `prompts/`;
- U9 byte-copies both prompts, with a guarded equality test.

**P22-n2 (nit; fresh `coder`). `HISTORY.md` says "18 mutants", but its own list names 19**, as the
ledger's count does. **Fix:** correct the figure in place.

**The seams to U4.3 and U3e.2:** nothing is missing. U4.3's inputs all exist:
- per-class `k/n` from `outcomes`;
- the class minimum from `len(outcomes[m])`;
- the mean latency from `latency`;
- `not assessable` from `scored_n`;
- drops by reason;
- per-class `overLimit` from `over_limit`;
- the disagreement line from `detail.answer`, `decidedBy` and `failedCall`.

One question needs a decision before U4.3 runs (Open questions). The parallel `itemId`s for FR-10's
design effect are absent, but plan4 §3.9 gives them to U3e.2 explicitly.

### Open questions

- **For the U4.3 brief (`teco`):** plan2 §3.6's disagreement line counts "items with a final answer
  in both". Does message type's no-label `None`, from a reply that parsed, count as a final answer?
  Call (d) says yes for scoring. If U4.3 also treats `None` against `None` as replicate agreement,
  that is a choice nobody has stated.

### Growth-stops

- **`model-bench/AGENTS.md`** went from 2,488 to 2,485 words.
  - **Rewritten in place:** three entries, for the new-ability scorer, the `validate_pack`
    sentence, and the drop-accounting home.
  - **Compressed:** the CLI sentence (the pointer to the README stays) and the context-file
    preamble line. Neither drops a live constraint.
- **`HISTORY.md`** gets one entry, which is read by lookup. Its counts reproduce, except P22-n2.

### Proposed REVIEW.md lines

None.

## Pass 23 — U4.3 FR-10 bar section (2026-11-01)

*2026-11-01. Gate of U4.3, uncommitted in `model-bench/` on `1784168b`: `report.py` (the FR-10
section, `_comparison_block`, `_answer_agreement`), the new `tests/test_decision_bar_fr10.py`, and
`HISTORY.md`/`README.md`/`AGENTS.md`. Read against v1.3 §3.6 and its U4 test rows (AC-15); plan2
§3.6, §4.1 U4 and §5 U4; plan3 §4 U4; plan4 §3.9; requirements FR-10, FR-11a, AC-12, AC-15; note
§Q4–§Q6; and the ledger's U4.3 row and "U4 brief"/"U4.3 brief" lines, including the
data-scientist's disagreement ruling. Pass 18 (d) is the source of the blocking carry.*

**Verdict: approve with suggestions.** Every plan2 §3.6 FR-10 line, state and minimum is present.
The guard's printed output is byte-identical to `1784168b` across 31 rendered reports. The
ruling's fixture matches the ledger. The delegate's blocking reading is the right one.
There are no blockers and no majors. There are two minors, both untested plumbing:
- the FR-10 latency seed and order (P23-m1);
- the reference arm's part in the content block (P23-m2).

There are also two nits.

**Checks run.** All of these ran on private copies in `scratchpad/p23/` (`wt/` = working tree,
`base/` = the tree with `report.py` from `git show 1784168b:`). After each mutant the live file was
restored and `cmp`-checked.
- **Suite:** the `wt/` copy gives `3424 passed, 3 skipped, 5 deselected, 2 xfailed`. The 3 skips
  are the falkor-chat-sibling guards, which are absent from a copy. So this agrees with teco's
  3427. `ruff` is clean. `AGENTS.md` has 2,482 words (2,485 at the base) and no line over 700
  characters.
- **Guard output unchanged:** I ran both full suites with a capture plugin (`plug/capplug.py`)
  that copies every `compare --out` report. That gave 31 guard reports from `test_decision_bar.py`
  (2,553 lines). Every one is byte-identical, `cmp`, with no normalisation needed. They include
  both disagreement-line tests, so the guard keeps its old line.
- **Mutants (mine):**

  | Mutant | Result |
  |---|---|
  | M1 FR-10 latency `seed=pack.seed` → `0` | **survives the full suite** (P23-m1) |
  | M2 latency values reversed, so not in `itemId` order | **survives the full suite** (P23-m1) |
  | M3 per-class `overLimit` prints the run total | red |
  | M4 `no label in both` counts `or` instead of `and` | red |
  | M5 latency state ignores `blocked` | red |
  | M6 the class line's call site passes DE 2.0 | red |
  | M7 the content block omits the chat arm on new abilities | **survives the full suite** (P23-m2) |
  | P21-n1: `roles.py` message type → `"query"` | red (2, including §Q5.4) |
- **Note §Q5.4:** I recomputed Wilson 95% (z = 1.95996) for 58, 56, 54, 53 and 55 of 60. The
  results are [0.886, 0.991], [0.841, 0.974], [0.799, 0.953], [0.778, 0.942] and [0.819, 0.964],
  which is the note's column digit for digit. `escalation` is the only class not met, since
  10·53 < 9·60.

CPG: considered, not relevant — no model-bench CPG is loaded; the code was read, mutated and
executed directly.

### Delegate calls

1. **The blocking reading: the delegate is right.**
   - Pass 18 (d) asked for one thing only. It noted that `blocked` (no incumbent, timeout off the
     pin) is computed after the speed-up return. Then it said "the FR-10 side must hoist the
     *timeout* block, because note §Q4.3 applies the 180 s pin to new abilities too."
   - The ledger's "timeout/no-incumbent" (2026-10-31 line) paraphrases more than that finding
     says.
   - FR-10 is single-arm: v1.3 §3.6, note §Q4.2 and §Q5.1, and plan2 §3.6 "New-ability points
     pair nothing". Plan2 §4.1 U7 step 1 runs the hand-off LLM's chat arm "as a reference" on
     new abilities. So blocking FR-10 on a missing reference would refuse a verdict that no
     input supports refusing.
   - **What this means for U7:**
     - A new-ability report renders with or without the reference arm.
     - If the arm is in `--models`, its timeout and content hash still block every FR-10 bar.
       `cli` already refuses mixed timeouts, and `_comparison_block` folds the arm's hash into
       the content check.
     - So U7 must run the reference arm on the same pack content as the replicates. See P23-m2,
       which covers whether that coupling is intended.
2. **A class with n = 0: consistent.**
   - The precedence (scored n, then any `not met`, then an empty class, then `met`) is
     `combine_replicates`' own order applied within a replicate: a decided conjunction beats an
     undecidable member.
   - The new reason starts `not assessable (`, so `combine_replicates` accepts it and carries
     its text, as the overall line in `test_a_class_with_no_scored_item…` shows.
   - Plan2 §3.6 lists only `scored n < 200` as a reason. But an empty class has no measured value
     (FR-11a), and it can never print `met`. That satisfies model-bench `REVIEW.md`'s
     degenerate-candidate rule.
3. **Guard output unchanged: verified** (Checks run). The only guard-adjacent text that changed is
   the no-bar-kind fallback, which now reads "has no requirement bar". No current role reaches it.
4. **The disagreement ruling: implemented exactly.**
   - I recounted the fixture by hand. `n` covers a–e, so `n` = 5. `k` = c, d, so `k` = 2.
     `no label in both` = b, e, so 2. `failures` = f, so 1. `one-sided` = g, so 1.
   - `decidedBy` is 1/5, because only e's first side is a hand-off.
   - Exclusion keys on `withheldFor` and then `failedCall`, never on `answer is None`.
   - The join-in parse-failure `False` agrees.
   - M4 goes red, and the guard keeps its line (item 3).
5. **§Q5.4: reproduced** (Checks run). The test's expected column is the note's, not computed by
   the code.
6. **The U3e.2 seam is sufficient.**
   - `_fr10_class_interval` is called once per class per replicate. That is the granularity plan4
     §3.9 needs, and M6 pins the call site.
   - U3e.2 still owes three things: the parallel `itemId`s on `AgreementItems` (plan4 §3.9, "U4's
     per-class FR-10 input gains the same parallel ids"), the per-class DE at that call site
     instead of `run.designEffect`, and the `DE <x.xx> (genRunId ICC)` print. All three are
     U3e.2's by plan.
7. **My plumbing mutants:** see the table above. M1, M2 and M7 survive.

### Findings

**P23-m1 (minor; fresh `coder`). The FR-10 latency line's seed, order and interval are unpinned.**
- Note §Q4.2 fixes `ȳ` over values sorted by `itemId`, with the pack's seed.
- Every FR-10 test uses one constant `ms`, and the fixture says "no interval below is asserted".
  So M1 (seed → 0) and M2 (order reversed) pass all 3,424 tests.
- A wrong seed or order changes the printed interval, which is the uncertainty a human reads
  beside the bar.
- **Fix:** one test with non-constant per-item `ms` that is asymmetric in `itemId` order. Assert
  the printed `seed <n>` against the seed read from the test's `pack.json`. Assert the printed
  `[lo, hi]` against `stats.mean_bootstrap_interval` called in the test, with `itemId`-sorted
  values and that seed, at the patched B. M1 and M2 must then turn red.

**P23-m2 (minor; fresh `coder` after `teco` picks the intent). The reference arm blocks FR-10, but
the docs say no bar reads it, and nothing pins either.**
- `_comparison_block` passes the chat arm to `_pack_content_reason` on both bar kinds. So a
  reference arm on other content makes every FR-10 bar `not assessable`.
- `README.md` ("a chat arm is a reference no bar reads") and the `Reference chat arm:` line say
  otherwise.
- M7, which leaves the arm out on new abilities, survives.
- Blocking is the conservative choice: it can only withhold a verdict, and it keeps U7's session
  on one pack content. I recommend keeping it.
- **Fix:** state it in README and in the reference line ("its timeout and pack content still
  apply"). Add a test where the reference arm's `packContentHash` differs and assert every FR-10
  bar carries the different-content reason. M7 must then turn red.

**P23-n1 (nit; `teco`). Record call 1's ruling in the U7 brief.**
- A new-ability bar needs no reference arm.
- If one is run, it must share the replicates' pack content and the 180 s timeout.

**P23-n2 (nit; fresh `coder`, with P23-m1). `README.md` line 205 is 167 characters.** It is the
new paragraph's last line, left unwrapped. Rewrap it to the file's ~100-character width.

### Growth-stops

- **`model-bench/AGENTS.md`** went from 2,485 to 2,482 words. One entry was rewritten in place,
  naming `_comparison_block` and FR-10 as single-arm. That is new scope folded into an existing
  entry, with no restated history.
- **`README.md`** gains the new-ability side of the bar paragraph. That is new scope.
- **`HISTORY.md`** gets one entry, read by lookup. Its counts reproduce: 20 functions collect as 21
  tests, and the total is 3,427.

### Proposed REVIEW.md lines

None. P23-m1 is a single instance. The existing boundary rule did its job on the bars themselves.

## Pass 24 — U3d decision-point importer (2026-11-01)

*2026-11-01. Gate of U3d, uncommitted on `6ffeca23`. Files: `scripts/import_decision_pack.py`
and `tests/test_import_decision_pack.py` (both new), `packs/guard-judge-synthetic/prompts/judge.md`
(new), and `README.md`/`AGENTS.md`/`docs/HISTORY.md`. Read against v1.3 §3.4 "Importer", §3.5 and
U3d; plan2 §4.1 U3d; plan4 §3.7, §4.2 U3d and §5's U3d row. Carried items: P12-m2 and P22-n1.
decision-models was read only.*

**Verdict: approve with suggestions.** Both halves follow v1.3 §3.4 and plan2 U3d:
- the eval split only, `PROVENANCE.md`, the version gate, and the candidate with its manifest
  but never the fit report;
- every plan4 §4.2 provenance field, with the §5 U3d row's `input_hash` claim pinned for all
  three roles;
- both carried items: the four per-kind `verdictMetrics` from `classification`'s constant, an
  explicit `incumbent`, an existing directory, plain overwrite, and `prompts/` untouched.

There are no blockers and no majors. There are three minors and two nits.

**Checks run** (on a private copy, `scratchpad/p24/wt/`):
- **Suite:** `3537 passed, 3 skipped, 5 deselected, 2 xfailed`. The 3 skips are the
  falkor-chat-sibling guards, which a copy cannot run, so this agrees with teco's 3540. The new
  file collects 111 tests, as `HISTORY.md` says.
- **Mutants:** 7 of mine, run against the importer and pack tests. The live files' sha256 was
  re-checked afterwards: OK.

  | Mutant | Result |
  |---|---|
  | X1 `splits.json` git check skipped (literal SHA) | **survives** (P24-m1) |
  | X2 side-check rows lose `labelSource: human` | red |
  | X3 version gate ignores `PROVENANCE.md` | red |
  | X4 generator = labeler refusal removed | red |
  | X5 candidate overwrite check removed | red |
  | X6 `SAMPLING_SEED` changed | survives (call 4: no pin needed) |
  | X7 `maxTokens` 256 → 512 | **survives** (P24-n2) |
- **Prompt bytes:** the new `judge.md` is `cmp`-identical to `guard-judge-understanding`'s. Parsing
  `falkor-chat/server/falkorchat/app.py`'s `_JUDGE_SYSTEM_PROMPT` with `ast`, it equals the file
  only after `rstrip("\n")` (738 against 739 characters). That is the ledger's U9 premise, and it
  is true.
- **Sweeps:** `splits` in plans 2, 4, 5 and 6 and in `decision-models/` (code, `AGENTS.md`,
  `README.md`), and `.gitignore`/`git check-ignore` on `decision-models/data/`.

CPG: considered, not relevant — no model-bench CPG is loaded; the script was read, mutated and
executed directly.

### The five questions

1. **The guard prompt file: in scope, and the right source.**
   - P22-n1 makes the importer require an existing pack directory and prompt. The new abilities'
     prompts came with U4.2, but no unit provided the guard's. U3d is the first unit that needs
     it, and having the importer copy it would break "never writes `prompts/`".
   - The incumbent pack's file is the right source. FR-1 and FR-2 compare arms on the same pack
     prompt, so the synthetic pack measures what the stored guard history measures.
   - `test_the_guard_handoff_prompt_is_a_byte_copy…` binds the two files. Whatever U9 decides
     about the newline must therefore change both files together (P24-n1).
2. **`splits.json`: no defined shape exists.**
   - v1.3 §3.4's tree names the file (`items.jsonl, splits.json, generation.json`).
   - Plan2 §3.5 and plan4 §3.6 define split *assignment* (by `seedId`, within cells), not a
     format. Plan4 §3.7 defines `generation.json` and item keys only. Plans 5 and 6 are silent,
     and decision-models has no split code.
   - The importer's assumed shape is reasonable. The predicate `labelVotes` keys
     (`"true"`/`"false"` in the test) are just as unstated. Where they should be homed is P24-m2.
3. **The git checks fit U5i.**
   - Plan4 §4.1 makes "dataset version committed" the U5c done-condition. U5i (`teco`) runs
     after it, so a clean, committed source is exactly what U5i will have.
   - `decision-models/data/` is not ignored (`check-ignore` exit 1).
   - An untracked, modified or out-of-repo file refuses before any write (tested).
   - **Gap:** the guard's side-check source is hashed but not git-checked. Its row reads
     `— (model-bench)`. That is acceptable for a model-bench file the same commit can carry, but
     the gap is real.
   - X1's survival is P24-m1.
4. **Constants not in the plan.**
   - **`sampling.seed 20261101`: accept.** It is pack data, fixed before any data exists and the
     same across re-imports. It selects nothing, so it needs no pin.
   - **Temperature 0.0 and `maxTokens` 256: accept.** They copy `guard-judge-understanding`'s
     prompt block. Production's `config/models.json` also sets temperature 0, with no token cap.
     The run fingerprint records `handoffTemperature`/`handoffMaxTokens`, so any divergence is
     visible.
   - **Exit codes 0/1/2: accept.** They match `refresh_golden.py`.
   - **Candidate overwrite (identical → no-op, other bytes → refused): accept.** It is plan2
     §3.6's "a file edited in place is a different candidate".
5. **The pack-count test: a note is not enough** (P24-m3).

### Findings

**P24-m1 (minor; fresh `coder`). The per-file git check and recorded SHA are unpinned.**
- Every test stubs `source_sha` with one value (`_FAKE_SHA`). The provenance test asserts only
  that the value appears somewhere, so X1 passes all 263 importer and pack tests.
- An unchecked `splits.json` could change eval membership from an uncommitted edit, and
  `PROVENANCE.md` would not show it.
- **Fix:** make the stub return a distinct SHA per path and record its calls. Assert it was
  called for exactly `items.jsonl`, `splits.json` and `generation.json`, and that each
  provenance row carries its own SHA. X1 must then turn red.

**P24-m2 (minor; `architect`, then `teco` for the U5.10 brief). The dataset contract the importer
reads has no home.**
- The `splits.json` shape and the predicate `labelVotes` key spelling exist only in the
  importer's docstring and its test builder.
- If U5.10 diverges, the importer refuses loudly ("must be a list of itemId strings"), so nothing
  is silently mis-imported. But the mismatch is found at U5i, after a paid campaign.
- **Fix:**
  - State both items beside plan4 §3.7 in plan6, which is in flight and `Extends:` plan4.
  - Pin them with one minimal dataset vector (`generation.json`, `items.jsonl`, `splits.json`)
    under `tests/fixtures/decision-contract/`. The importer's tests import it, and U5.10 checks
    its `finalize` output's shape against a byte-copy under the existing guarded-equality
    pattern (v1.3 §3.2.2).

**P24-m3 (minor; fresh `coder`, in this fix round).
`test_every_shipped_pack_validates_unchanged_and_none_is_a_decision_point_pack` breaks on the
first import.**
- It asserts `len(roots) == 5` and that no pack has `decisionPoint`.
- U5i is a `teco` data unit whose done-condition is "the pack validates", so the first import
  would leave the suite red until a code edit is delegated.
- **Fix:** name the five legacy packs and keep the assertions for them. Separately, validate every
  `packs/*-synthetic/` that has a `pack.json`. The committed imports then gain a suite check
  instead of breaking one. A planted sixth legacy pack and a planted invalid synthetic pack must
  each turn it red.

**P24-n1 (nit; `teco`). Add to the U9 brief:**
- `guard-judge-synthetic/prompts/judge.md` is a second copy of the prompt behind the newline
  question, bound to the first by a byte test.
- Any prompt edit changes the pack's content hash, so it also needs a re-import under a new
  `--pack-version`. The importer is the only writer of `pack.json`.

**P24-n2 (nit; fresh `coder`). The guard's prompt parameters are not bound to the incumbent
pack's.**
- X7 survives.
- **Fix:** add a test asserting that the imported guard manifest's `temperature`/`maxTokens`
  equal `guard-judge-understanding/pack.json`'s.
- Also rewrap `README.md`'s new 113-character line (the `judge.md` sentence).

### Growth-stops

- **`model-bench/AGENTS.md`** went from 2,482 to 2,490 words, with no line over 700 characters.
  Two entries (the prompt home and the FR-23 bullet) were rewritten in place to name the
  importer. That is new scope, with no history restated.
- **`README.md`** gains a usage paragraph: new scope.
- **`HISTORY.md`** gets one entry, read by lookup. Its counts reproduce.

### Proposed REVIEW.md lines

None. P24-m1 is a single instance.

## Pass 25 — U3e.1 pack provenance (2026-11-01)

*2026-11-01. Gate of U3e.1, uncommitted in `model-bench/` on `8fcb341c`. Files: `modelbench/packs.py`,
`modelbench/report.py`, the `decision_point_guard` fixture pack, the eval-item builders in five test
files, `tests/test_import_decision_pack.py`, `README.md` and `docs/HISTORY.md`. Read against plan4
§3.9 ("Pack contract" and "Interface changes"), §4.2 U3e.1 and §5's U3e.1 row, and the committed
importer (`scripts/import_decision_pack.py`, `362dc56b`). U3e.2 is out of scope except for the
interface it consumes.*

**Verdict: approve with suggestions.** All three parts of the §5 U3e.1 row are present and
pinned:
- the two `validate_pack` refusals, each with a deleted-key and wrong-type case;
- `gen_run_ids` equal to the eval items' `itemId → genRunId` map, on the contract pack and on an
  imported pack for each of the three roles;
- the exact two-model caveat line.

The importer's output passes the new checks: it always writes `labels.generator` with `provider`,
and a non-empty string `genRunId` on every eval row (`import_decision_pack.py:184-194`, `:227-229`).

There are no blockers and no majors. There is one minor and one nit.

**Checks run:**
- **Suite** (from `model-bench/`): `3572 passed, 5 deselected, 2 xfailed`. Ruff is clean. The
  collected counts match `HISTORY.md`: 177 in `test_decision_point_pack.py` and 118 in
  `test_import_decision_pack.py`. 158 + 19 new matches the 17 new functions plus 2 rows in the
  deleted-key table.
- **Probe** (`scratchpad/p25/probe.py`, on `_contract_pack` copies):
  - deleting `labels.labeler.provider` gives `validate_pack == []`, and `from_pack` then raises;
  - deleting `labels.generator.provider` does the same;
  - a generator model equal to the labeler's gives `validate_pack == []`.
- **Mutants:** 3 of mine. Each file was copied aside to `scratchpad/p25/` first, and `cmp` matched
  after restoring.

  | Mutant | Result |
  |---|---|
  | X1 `validate_pack` checks only for the `genRunId` key, not a non-empty string | red (4) |
  | X2 `from_pack` takes any truthy `genRunId` and stores `str()` of it | **survives** (P25-n1) |
  | X3 the generator refusal runs only when the labeler's model is valid | **survives** (P25-n1) |

CPG: considered, not relevant — no model-bench CPG is loaded; the change was read, probed and
mutated directly.

### The four questions

1. **Generator provider: the premise holds, and the gap was already there for the labeler.**
   - At `8fcb341c`, `_labels_problems` checked only `labels.labeler.model`, while
     `DecisionPointPack.from_pack` already required `.provider` as well. The probe shows the same
     result now for both blocks.
   - `compare` never calls `validate_pack` (`cli.py:450-490`), so `from_pack` is its only check.
   - **Under §3.9:** the wording ("refuses a missing `labels.generator.model`") is met to the
     letter. But `from_pack`'s docstring claims every pack it refuses is one "`validate_pack`
     refuses", which is false for `provider`.
   - **Cost:** a pack can validate and run, then fail `compare` with exit 4. Fixing `pack.json`
     changes the content hash, so every stored arm then reports `not assessable (arms ran on
     different pack content)` and must be re-run.
   - **Reach:** only a hand-built pack can hit this, because the importer always writes
     `provider`.
   - **Acceptable as a copy? Not quite.** The fix is one predicate and closes the gap for both
     blocks: P25-m1.
2. **A generator equal to the labeler: no requirement, so no finding.**
   - Plan2 §3.5 and plan4 §3.7 place the refusal at generation. The importer repeats it
     (`:441`).
   - §3.9's pack contract lists exactly two refusals, and this is not one of them.
   - The method note (`typed-decision-models-ml.md` "Generator ≠ labeler") states it as a property
     of the method, and names "the caveat line naming both models" as its counterweight. A
     hand-built pack with equal models would print the same name twice in the caveat, which is
     visible.
   - Adding the refusal would be defence in depth, not conformance. That is the architect's call,
     not a defect.
3. **Side-check exemption: consistent.**
   - §3.9 says "an eval item without … `genRunId`", and builds `gen_run_ids` "for eval items".
   - The importer's side-check rows are `guard-judge-understanding`'s items, re-stamped
     (`:244-251`), and they have no `genRunId`.
   - The test pins both that the 85 side-check items have no `genRunId` and that the pack is
     accepted.
   - **Interface for U3e.2:**
     - FR-1 pairs are eval-only (`pair_items(strata=)`).
     - A new-ability pack can hold no side-check item (`packs.py:1303`), so FR-10's input is
       eval-only too.
     - `gen_run_ids` is `Mapping[str, str]`, keyed by `str(itemId)`, the same key `strata` uses.
     - So every outcome id U3e.2 looks up resolves. Keys, eval-only and the value type all match
       §3.9.
4. **Docs: faithful, and nothing in `AGENTS.md` is now wrong.**
   - A word-level diff of `README.md` against `8fcb341c` shows only the intended insertions, in
     both paragraphs: "labels caveat, naming the labeler and the input generator", "and a
     `generator.model`", "and a non-empty string `genRunId`", and "and `genRunId`". The rest is
     rewrap.
   - `AGENTS.md` does not list the `labels` checks or the caveat text. Its decision-point
     paragraph (`:89-99`) is still true.

### Findings

**P25-m1 (minor; fresh `coder`, in this fix round). `validate_pack` accepts a pack that
`from_pack` refuses.**
- Evidence: `packs.py` `_labels_problems` checks `.model` only. `report._model_name` requires
  `.provider` and `.model`. The probe gives `validate_pack == []` with either block's `provider`
  deleted.
- **Why it matters:** the failure comes after a run, at `compare` (exit 4). The repair changes
  `contentHash`, which makes every stored arm not assessable.
- **Fix:**
  - Have `_labels_problems` refuse an absent, empty or non-string `provider` for both blocks.
  - Put the predicate in one place. For example, `report._model_name` could call a helper in
    `packs`, so the two checks cannot drift apart again.
  - Correct `from_pack`'s docstring if any difference remains.
- **Test:** do not add one test per shape. Write one coverage probe over the axes:
  - block (`labeler`, `generator`) × key (`provider`, `model`) × corruption (deleted, `""`,
    non-string);
  - for each case, assert that `validate_pack` is non-empty if and only if `from_pack` raises.
  - Deleting the new provider check must then turn the probe red.

**P25-n1 (nit; same `coder`). Two branches are unpinned.**
- X2: `from_pack`'s non-string `genRunId` guard. No test gives it a non-string value, and
  `compare` does not validate first.
- X3: no test deletes both the labeler and the generator. That pair should list both problems,
  like the existing labeler-plus-incumbent test.
- **Fix:** extend `test_the_decision_point_pack_refuses_an_eval_item_without_gen_run_id` with `""`
  and `7`. Add a test that deletes both blocks and asserts two problems. X2 and X3 must then turn
  red.

### Growth-stops

- **`model-bench/AGENTS.md`** is unchanged at 2,490 words. Nothing in it is now wrong (question 4).
- **`README.md`** inserts new-scope words in place. The rest is rewrap, and no changed line is over
  100 characters.
- **`HISTORY.md`** gets one entry, read by lookup. Its counts reproduce.

### Proposed REVIEW.md lines

None. P25-m1 is a single instance of a validator and a loader drifting apart. If it recurs, propose
"a loader's refusal must be a subset of `validate_pack`'s, pinned by one probe".

## Pass 26 — U3e.2 per-cell design effect (2026-11-01)

*2026-11-01. Gate of U3e.2, uncommitted in `model-bench/` on `291216f7`. Files:
`modelbench/decision_stats.py`, `modelbench/report.py`, `tests/test_decision_stats.py`,
`tests/test_decision_bar.py`, `tests/test_decision_bar_fr10.py`, `README.md` and `docs/HISTORY.md`.
Read against plan4 §3.9 and §5's U3e.2 row, and against the note's §Q6 "Design effect"
(`docs/plans/typed-decision-models-ml.md`).*

**Verdict: approve with suggestions.** The change does what §3.9 and the U3e.2 row ask:
- **The formula** is §3.9's exactly, and agrees with §Q6 (Kish `m₀`, floor at 1.0, discordance for
  FR-1, agreement for FR-10).
- **`outcome_ids`** is filled in the same pass as `outcomes` and stays parallel after drops.
- **The two call sites** switched to the per-cell value, and every cell and class line prints
  `DE <x.xx> (genRunId ICC)`.
- **Unchanged:** `RunResult.designEffect`, `basis` and `compare_report`. The diff touches no
  `results.py` or `runner.py` line, and `report.py:1026`, `:1673` and `:2954` still take the
  run-level maximum.

There are no blockers and no majors. There are two minors and two nits. Two of the three questions
are ML-method questions and route to `data-scientist`, below.

**Checks run** (from `model-bench/`, with a private `PYTHONPYCACHEPREFIX`):
- **Suite:** `3601 passed, 5 deselected, 2 xfailed`. Ruff is clean. The test-function counts
  match `HISTORY.md`: 83 → 94, 50 → 52 and 22 → 23.
- **Independent estimator check** (`scratchpad/p26/de_probe.py`):
  - The probe recomputes the ICC as a variance component, `σ̂²_a = (MSB − MSW)/n₀` and
    `ICĈ = σ̂²_a/(σ̂²_a + MSW)`, written separately from `design_effect`.
  - On 2,000 random unbalanced groupings (2–8 runs of 1–30 items) it matches `design_effect` to
    within 1.8e-15.
  - `1.0 ≤ DE ≤ max(1, m₀)` held in every case.
- **Reproduction of the 24.86 cell** (question 3): rebuilt from the test's own fixture.
- **FR-10 `KeyError` probe:** one stored replicate's first `itemId` was renamed to `ghost-item`,
  keeping its `packContentHash`. `cli.main(["compare", …])` raised an uncaught
  `KeyError: 'ghost-item'` (P26-m1).
- **Mutants:** 5, run against the three touched test files. Each file was copied to
  `scratchpad/p26/` first, then restored and checked with `cmp`.

  | Mutant | Result |
  |---|---|
  | Y1 FR-1 outcome `cand and not inc` (b-direction discordance only) | **survives** (P26-m2) |
  | Y2 FR-1 outcome is the candidate's error (the implementer's M6) | red (1) |
  | Y3 FR-10 groups misses (`not a`) instead of agreement | survives — **equivalent**: `MSB`/`MSW` are invariant under `y → 1 − y` |
  | Y4 `n₀ = N/g` (mean run size) | red (1) |
  | Y6 FR-10 prints `DE 1.00` while its interval uses the class's value | red (1) |

CPG: considered, not relevant — no model-bench CPG is loaded. The change was read, probed and
mutated directly.

### Pass 25 dispositions

- **P25-m1:** fixed. Rerunning `scratchpad/p25/probe.py` at `291216f7`, `validate_pack` now refuses
  a deleted `provider` for both blocks. The coverage probe
  `test_validate_pack_refuses_exactly_the_model_blocks_from_pack_refuses` (block × key ×
  corruption) is present.
- **P25-n1:** fixed. `from_pack`'s non-string `genRunId` cases and
  `test_deleting_both_model_blocks_gives_two_problems` are present. X2 and X3 were not re-run.

### The three questions

1. **The FR-1 outcome: it conforms. This is a code question, and it is settled.**
   - §Q6 "Estimate" says, verbatim, "the per-item outcome: discordance for FR-1". The plan's
     §3.9 says "`cand ≠ inc` per pair". `report.py:2279-2283` groups `cand != inc`.
   - Executed evidence:
     - Y2 (the candidate's own error) turns red.
     - `test_fr1_groups_discordance_so_errors_both_arms_share_leave_de_at_one` shows shared,
       concordant errors in one run leave `DE 1.00`.
   - **Coverage gap:** no test places a *c*-direction (incumbent-worse) discordance in a run, so a
     b-only mutant (Y1) survives. That is P26-m2.
   - Whether discordance is the right clustered quantity for an exact McNemar cell is the note's
     decision, already taken. It is not open here.
2. **`n₀`: the code is right. Whether it is the right estimator is an ML question.**
   - The code-level answer is yes. `n₀ = (N − Σm²/N)/(g − 1)` is the standard average group size
     in the unbalanced one-way random-effects ANOVA. With it, §3.9's ICC equals the
     variance-component form, which the probe confirms numerically.
   - **Route to `data-scientist`:** §Q6 says only "one-way ANOVA ICC" and gives no `n₀`, so the
     estimator is fixed by the plan, not the note. `model-bench/AGENTS.md` says `decision_stats.py`
     implements the note §Q3–§Q6 "and no other source", so §Q6 should state the estimator in one
     line.
   - Whether the ANOVA ICC suits a rare binary outcome is also `data-scientist`'s to decide. Here
     the outcome is discordance at `ψ̂ ≈ 0.02` with few runs per cell, where that estimator is
     known to be unstable.
3. **DE 24.86: the arithmetic is correct. Whether that size is intended is an ML question.**
   - The fixture's `falseAdvanceRateByUnderstanding` cell holds 197 items. The runs are 47 / 47 /
     47 / 47 / 8 (`hot`) / 1, so `m₀ = Σm²/N = 45.18`.
   - The four discordant items all sit in `hot` (4 of 8), giving `ICĈ = 23.86/44.18 ≈ 0.54` and
     `DE = 24.86`. `n_eff = ⌊197/24.86⌋ = 7`, and the line prints "could only detect ≥ 77.2 pp at
     n = 7".
   - **There is a built-in cap: `DE ≤ m₀`**, because `ICĈ ≤ 1` whenever `n₀ ≥ 1`. The probe held
     this in all 2,000 cases. So `n_eff ≥ N/m₀ = N²/Σm² ≥ 1` always, and nothing downstream sees
     `n_eff < 1`.
   - The fixture's 24.86 comes from its runs of 47 items per cell. §Q6 "Size" expects `m₀ ≈ 3`
     per cell, which would bound DE near 3.
   - **Route to `data-scientist`:** whether a cap or shrinkage below `m₀` is intended, and whether
     `m₀ ≈ 3` holds for the guard's actual run layout. Plan4 §3.8 runs 25 seeds per run, both
     evidence kinds per run, with four cells. Nothing in the code is missing relative to the
     spec.

**Also checked:**
- **Where `DE` is printed:**
  - FR-1: after "(one-sided 95%)" and before " — <state>", on every cell line, including
    not-assessable replicates whose cell lines still render.
  - FR-10: after the interval and before the state, including `0/0 [—]; DE 1.00` on an empty
    class.
  - A replicate with no pairing (inconsistent, different content, no incumbent) prints no cell
    lines, so it prints no `DE`.
  - See P26-n1 on the `1.00` fallback.
- **`gen_run_ids[item_id]` against `from_pack`'s guarantee:**
  - **FR-1 cannot hit it.** `pair_items` refuses any id whose stratum is not eval or side-check.
    It also refuses a side-check item that declares a per-kind cell. `strata` and `gen_run_ids`
    are built from the same rows, so every id in a cell is an eval id with a `genRunId`.
  - **FR-10 can.** `agreement_items` never consults the pack (P26-m1).

### Findings

**P26-m1 (minor; fresh `coder`, in this fix round). FR-10's design effect raises an uncaught
`KeyError` on a record item that is not in the pack.**
- Evidence: `_render_fr10_replicate` passes `agreement.outcome_ids` straight into
  `gen_run_ids[item_id]` (`report.py`, `_cell_design_effect`). The probe above gets an uncaught
  `KeyError: 'ghost-item'` out of `cli.main`.
- The FR-1 path turns the same record into "not assessable (cannot pair …)", because `pair_items`
  checks ids against `strata`.
- **Why it matters:** before U3e.2, FR-10 ignored item ids. Now a malformed record, one that
  agrees with its own content hash but not with the pack, crashes `compare` with a traceback. The
  README's closed exit-code contract (`0` whenever the tool ran and reported) promises it never
  does.
- **Fix:** give `agreement_items` the pack's `strata`, as `pair_items` has, and make it raise
  `ValueError` on an id that is not an eval item of the pack. The existing `except ValueError` in
  `_render_fr10_replicate` then prints "not assessable (cannot read this replicate's items: …)".
- **Test:** the ghost-id record must give exit 0 and that line. Removing the new check must turn
  it red.

**P26-m2 (minor; same `coder`). FR-1's discordance is pinned in one direction only.**
- Evidence: Y1 (`cand and not inc`) passes all 283 tests in the three files. Both FR-1 DE tests
  use b-direction flips (candidate worse) or no discordance.
- **Why it matters:** §Q6 groups discordance, `b + c`. A cell whose clustered discordance is
  incumbent-worse (*c*) would silently get `DE 1.00` and a too-narrow MDD.
- **Fix:** add a `test_decision_bar.py` case with the regrouped `hot` run holding *c*-direction
  discordance (the incumbent errs, the candidate does not) that asserts `DE > 1`. Y1 must then
  turn red.

**P26-n1 (nit; `architect`, with `data-scientist`). `DE 1.00 (genRunId ICC)` is printed where no
ICC was estimated.**
- `design_effect` returns the 1.0 fallback for an empty cell, a single run, all-singleton runs and
  constant outcomes. The line still names "genRunId ICC" as its source: for example,
  `agree[escalation]: 0/0 items [—]; DE 1.00 (genRunId ICC)`, and every n = 1 FR-1 cell in the
  fixture.
- This conforms to §3.9's literal "Each such line prints `DE <x.xx> (genRunId ICC)`", and 1.0 is
  the value actually applied. But it reads as "measured: no clustering".
- **Decide whether** the fallback should print, for example, `DE 1.00 (not estimable: <reason>)`.
  If yes, §3.9 changes first.

**P26-n2 (nit; same `coder`). README rewrap.**
- `README.md:202` (163 characters) and `:206` (119 characters) are the new bar-section lines.
  Every other line in the file is at most 101 characters.
- The wording is faithful to §Q6 "Scope": it widens the limits and never moves a decision.

### Growth-stops

- **`model-bench/AGENTS.md`** is unchanged at 2,490 words. Its line 109 (`RunResult.designEffect`
  required) is still true.
- One `AGENTS.md` claim now leans on the plan: "implements the note §Q3–§Q6 and no other source",
  while `n₀` comes from plan4 §3.9. That is question 2's routing, not an `AGENTS.md` edit.
- **`README.md`** inserts new-scope words in place, with no history restated.
- **`HISTORY.md`** gets one entry, read by lookup. Its counts reproduce.

### Proposed REVIEW.md lines

None. P26-m1 is the second instance of a loader-versus-validator gap, after P25-m1, but in a
different form: a renderer trusting record ids. Its fix mirrors an existing guard, so it does not
yet justify a rule.

## Pass 27 — U3e.3 family-pooled design effect (2026-11-01)

*2026-11-01. Gate of U3e.3, uncommitted in `model-bench/`. The note and plan changes are committed
at `749154e8`. Files: `modelbench/decision_stats.py`, `modelbench/report.py`, the three
decision-stats and decision-bar test files, `README.md` and `docs/HISTORY.md`. Read against the
note v1.14 §Q6 "Design effect" (`docs/plans/typed-decision-models-ml.md`), plan4 v1.3 §3.9 and
§5's U3e.3 row, and the hand fixture in `docs/reviews/typed-decision-models.md` Pass 13.
`decision-models/` was not opened.*

**Verdict: approve.** The code is faithful to §Q6 and §3.9 on every point the brief lists:
- the contributing rule (`g_c ≥ 2`, `N_c > g_c`);
- per-cell sums of squares, the pooled `n₀`, the unfloored `icc`, and each cell's own Kish `m₀`;
- 1.00 for an empty cell, and the formula for a non-empty non-contributing cell;
- both reason texts, verbatim;
- the family line's place and format, FR-10 pooling every class including empty ones, and FR-1
  pooling the four verdict cells on `cand ≠ inc`.

There are no blockers, majors or minors. There is one nit. Question 2 is a method question and goes
to `data-scientist`.

**Checks run** (from `model-bench/`, with a private `PYTHONPYCACHEPREFIX`):
- **Suite:** `3619 passed, 5 deselected, 2 xfailed`. Ruff is clean. The test-function counts
  match `HISTORY.md`: 96 → 101, 53 → 57 and 24 → 29.
- **Independent recomputation** (`scratchpad/p27/indep.py`): exact `Fraction` arithmetic written
  apart from the code.
  - On Pass 13's fixture it gives `ICĈ = 721/2557` and `contributing = 2`, with DE A 1.403,
    B 1.677, C 1.282, D 1.000 and an empty E 1.000. `family_design_effect` returns the same.
  - On the FR-10 clustered fixture as its docstring describes it, it gives `ICĈ = 0.76230`,
    `yes` 9.64 and `no` 26.92.
- **Mutants:** 4 of mine, run against the three test files. Each file was copied to
  `scratchpad/p27/` first, then restored and checked with `cmp`.

  | Mutant | Result |
  |---|---|
  | Z1 FR-1 discordance `cand and not inc` (P26-m2's guard, re-run against the family form) | red (1) |
  | Z4 `n₀` divided by `Σ(N_c − g_c)` instead of `Σ(g_c − 1)` | red (10) |
  | Z11 FR-1 family over every paired metric (pooled and side check), not the four cells | red (27) |
  | Z12 an empty cell's `m₀` set to 2 (it would take the formula) | red (1) |
- **Removal of `design_effect(outcomes_by_run)`:** a grep finds no reference to `design_effect(`
  or `_cell_design_effect` left in `modelbench/`, `tests/` or `scripts/`. The remaining
  `design_effect` names are `parity_cell`'s keyword and `stats.design_effect` in `runner.py`, which
  is unrelated. `README.md` and `AGENTS.md` name neither. `HISTORY.md` records the removal.

CPG: considered, not relevant — no model-bench CPG is loaded. The change was read, recomputed and
mutated directly.

### Pass 26 dispositions

- **P26-m1:** fixed. The ghost-id probe, re-run, gives exit 0 and "not assessable (cannot read this
  replicate's items: agreement_items: no eval stratum in the pack for items ['ghost-item'])".
- **P26-m2:** fixed. A *c*-direction test is present (`b=0, c=4`, `test_decision_bar.py:922-929`),
  and Z1 is red.
- **P26-n1:** superseded. §Q6 now prints a not-estimable family line, and its "a fallback never as
  a measurement" is met.
- **P26-n2:** fixed. No `README.md` line is over 101 characters.
- **The routed questions (`n₀`; DE ≤ m₀ and a cap):** answered by note v1.14 §Q6. `n₀` is now stated
  there, and "Rejected: a cap below `m₀` …".

### The five questions

1. **Exact match to §Q6 and §3.9: yes.**
   - **Empty and non-contributing cells:** an empty cell gives `m₀ = 1` and so DE 1.0. A
     non-empty non-contributing cell takes `1 + (m₀ − 1)·ICĈ`, as in the fixture's C (one run,
     1.282) and D (singletons, 1.000).
   - **Reasons:** the module constants are character-identical to note line 788.
   - **Placement:** the family line is appended after the FR-1 heading (`report.py`, in
     `_render_replicate`) and after the FR-10 heading, before any cell or class line. Tests pin
     `head + 1` for both bars and one family line per replicate. Early returns print none.
   - **Format:** `{icc:+.3f}` gives the ASCII sign. The not-estimable form is
     `… not estimable (<reason>) — DE 1.00`.
   - **FR-10** passes `agreement.outcomes`, whose every class is present, `()` when empty.
   - **FR-1** passes exactly `PER_KIND_VERDICT_METRICS` on `cand != inc`. Z11 and Z1 are both red.
2. **The constant class's DE 26.92 is what §Q6 prescribes as written. Whether it should be is a
   method question for `data-scientist`.**
   - **Recomputed:** `no` holds 140 items in 4 runs of 35, all agreeing. It contributes (`g_c =
     4`, `N_c = 140 > 4`) with zero SSB and SSW, so the pooled `ICĈ = 0.76230` applies:
     `1 + (35 − 1)·0.7623 = 26.92`.
   - §Q6 "Per cell" gives every non-empty cell the formula. No exception exists for a cell with no
     within-cell variation, and "Why pooled" accepts that different ICCs are averaged. So the code
     is faithful.
   - **A side effect to put to `data-scientist`:** the constant class *raises* the ICC that then
     penalizes it. It adds 136 within-cell degrees of freedom and no sum of squares, which lowers
     MSW: `yes` alone gives ICĈ 0.745 (DE 9.44), and pooled gives 0.762 (DE 9.64).
   - **What it does in practice:**
     - `no`'s Wilson interval for 140/140 goes from [0.973, 1.000] at DE 1 to [0.575, 1.000] at
       26.92. That is Wilson at 5.2 of 5.2, so the class is treated as about five independent
       items.
     - It stays bounded: `DE ≤ m₀` gives `n/DE ≥ N/m₀ = N²/Σm² ≥ 1`, and Wilson stays within
       [0, 1].
     - It remains limit-only (§Q3.2), because FR-10's state reads the measured `k/n`. The test
       asserts `DE 26.92 — met`.
   - This is not a code finding.
3. **`pooled over 1 cells`: nit at most** (P27-n1).
4. **Untouched or removed cleanly:** yes.
   - The diff touches neither `results.py` nor `runner.py`.
   - `report.py`'s run-level `max(…designEffect…)` sites are outside the hunks, and `compare_report`
     is untouched.
   - A test pins `RunResult.designEffect == 1.0` on the stored records.
   - The removal of `design_effect(outcomes_by_run)` is covered under "Checks run".
5. **Growth-stops:** see below.

### Findings

**P27-n1 (nit; `data-scientist` for the note's template, then the same implementer). `pooled over 1
cells` / `1 classes`.**
- §Q6 "Print" gives the template with a fixed plural, and the code renders it literally. A
  one-contributing-cell family therefore reads `pooled over 1 cells`. The rendering test pins
  `pooled over 1 classes`.
- If wanted, the note gains a singular form (`cell`/`class` at `k = 1`). Then
  `_family_design_effect_line` and that test follow. Nothing else depends on the string.

### Growth-stops

- **`model-bench/AGENTS.md`** is unchanged at 2,490 words, and nothing in it is now wrong.
- **`README.md`**: the bar-section sentence is rewritten in place for the family line. The old
  per-cell `(genRunId ICC)` wording is gone, not kept beside the new text. No line is over 101
  characters.
- **`HISTORY.md`**: one new entry, read by lookup. Its counts reproduce. It sits above, and does not
  edit, the U3e.2 entry it supersedes.

### Proposed REVIEW.md lines

None.

## Pass 28 — U2.4 the `prime` guard (2026-11-01)

*2026-11-01. Gate of U2.4, uncommitted in `model-bench/`. Files: `modelbench/runner.py`
(`_drive_single_call_items`), the new `tests/test_runner_prime_guard.py`, `docs/BACKLOG.md`
(R-21), `docs/HISTORY.md` and `README.md`. Read against plan5 §3.1 and §4's U2.4 row
(`docs/plans/typed-decision-models5.md`). `decision-models/` was not opened.*

**Verdict: approve with suggestions.** The guard is what §3.1 rules: it catches the two classes,
raises `RunRefused(exitCode=3)` `from exc`, does not re-probe, names the corpus embed, ends with
"nothing was scored", and lets every other exception through. Every check in the U2.4 row is
pinned by a test. There is one minor, which the plan's own wording leaves open and which predates
this unit, and one nit.

**Checks run** (from `model-bench/`, with a private `PYTHONPYCACHEPREFIX`):
- **Suite:** `3625 passed, 5 deselected, 2 xfailed`, and ruff is clean. The new file has 6 tests,
  as `HISTORY.md` says.
- **Probes,** in `scratchpad/p28/`:
  - Three malformed 2xx embed bodies were passed through the real `LMStudio.embed` parse, then
    through `cli.main run` on a cache miss (P28-m1).
  - A `sha256` snapshot of the whole workspace root was taken before and after a refused
    (timeout) run.
- No mutants of my own. The implementer's 7 and teco's one cover the guard's branches.

CPG: considered, not relevant — no model-bench CPG is loaded. The call path was read directly.

### The checks asked for

- **The catch set and exit 3 match.** `_warm_up` (`runner.py:1808-1824`) maps exactly
  `LMStudioCallTimeout` and `LMStudioCallFailed` to exit 3, with no re-probe. U2.4 mirrors it.
- **Transport failures cannot escape.**
  - `embed` posts through `_raw_post`, whose ladder ends in only those two classes: HTTP error,
    timeout, `URLError`, `HTTPException` and `OSError`.
  - `_parse_json` maps an undecodable body to `LMStudioCallFailed`, and a missing `data` list
    also raises `LMStudioCallFailed`.
  - `LMStudioUnreachable` is raised only on the GET paths (`lmstudio.py:627`, `:666`).
- **A malformed 2xx body can still escape** (P28-m1).
- **"Stores nothing" holds on disk.**
  - After a refused run, the whole root's file tree and bytes are identical to before.
  - `runner.py` has no write call at all. Only `cli`'s `store` writes, and the refusal returns
    before it.
  - This holds for any `prime` failure, because `prime` runs before the first item.
- **The README is accurate.** The new exit-3 cause, "failing the embedder's corpus embed before any
  item", matches the code. It applies only on a cache miss, which the sentence does not need to
  say. One line is over width (P28-n1).

### Findings

**P28-m1 (minor; `architect` for the ruling, then `tdd-engineer`). A malformed 2xx embed body
still ends `run` in a traceback.**
- Evidence: `LMStudio.embed` (`lmstudio.py:827-831`) silently drops entries without an
  `embedding` list, and checks neither the vector count against `len(texts)` nor the element
  type. `retrieval.prime` then fails outside the catch set. Through the real parse and
  `cli.main`:

  | 2xx body | Result |
  |---|---|
  | `{"data": [{}, {}]}` | uncaught `ValueError` (zip shorter) |
  | one vector for a 2-document corpus | uncaught `ValueError` (zip shorter) |
  | a non-numeric embedding | uncaught `TypeError` |

  Nothing is stored in any of these cases.
- **Why it matters:** the closed exit-code contract is broken by a server answering 200 with the
  wrong shape. That is the same contract-breaking-body case `decide()` already maps to
  `LMStudioCallFailed(status=None)`. The defect predates U2.4, and §3.1's "other exceptions pass
  through" makes U2.4 faithful as written.
- **Fix:** have `embed` raise `LMStudioCallFailed(status=None)` when the vector count is not
  `len(texts)` or an entry is not a non-empty list of numbers, matching `decide()`'s taxonomy.
  The U2.4 guard then covers the corpus case with no runner change.
- **Test:** the per-item `embed` site (`runner.py:515`) consumes the same parse, so state its
  disposition in the ruling. Write one probe over the axes: body shape (missing key, short count,
  wrong type) × call site (corpus, item).

**P28-n1 (nit; the same implementer). The rewrapped exit-3 line is too long.**
- `README.md:225` is 148 characters. Rewrap it to the file's ≤ 101.

### Growth-stops

- **`docs/BACKLOG.md`:** R-21 was rewritten in place. It drops only the delivered `prime` pointer,
  and its open "Unruled, for `architect`" ask is kept. The pointer's content is recorded in the new
  `HISTORY.md` entry, so nothing was discarded.
- **`HISTORY.md`:** one new entry, read by lookup. Its counts reproduce (6 tests; suite
  3619 → 3625).
- **`README.md`:** one exit-3 cause inserted in place. `AGENTS.md` is unchanged and nothing in it
  is now wrong.

### Proposed REVIEW.md lines

None.

## Pass 29 — U6.6 the dataset contract vector (2026-11-01)

*2026-11-01. Gate of U6.6, uncommitted. Files: `model-bench/scripts/import_decision_pack.py`
(`label_vote_keys`, the `labelVotes`-key refusal, docstring), `model-bench/tests/test_import_decision_pack.py`,
the new `model-bench/tests/fixtures/decision-contract/dataset-shape.jsonl` and its copy under
`decision-models/tests/fixtures/decision-contract/`, `decision-models/tests/test_decision_contract.py`,
`model-bench/README.md` and both `HISTORY.md` files. Read against plan6 v1.5 §3.7 and its §4 U6.6
and §5 U5.10 rows (`docs/plans/typed-decision-models6.md`), plan4 §3.6–§3.7, plan5 §3.5, plan2
§3.5, and decision-models' `points/*/point.json`, `points.py` and `audit.run_record`. Out of scope:
`opencode/`, `docs/plans/agent-reading-cost*` and the coordination ledger.*

**Verdict: approve with suggestions.** The vector is §3.7 as written. It has one line per point in
`ROLES` order, each `{role, generation, items, splits}`. The `generation` key tree is complete, and
it includes `spawnTemplate.driver`, `safeguardSkipped`, `formMismatch`, `bindingMismatch` and one
run. There are three items, one per split, and `splits` is `{rngSeed, eval, train, calibration}`.
`labelVotes` keys are the answer space as strings. The refusal rejects exactly a key set other than
that space, and every mutant of it I tried is killed. The copy is guarded like the existing two.
falkor-chat is unaffected. One major is a plan gap in how U5.10 compares value types, which the
vector makes concrete. It goes to `architect` before U5.10's brief and does not block U6.6. There is
one minor and one nit.

**Checks run** (each suite from its own component directory, with a private `PYTHONPYCACHEPREFIX`):
- **Suites:** model-bench `3651 passed, 5 deselected, 2 xfailed`; decision-models
  `671 passed, 2 xfailed`; ruff clean in both. All match the brief and the `HISTORY.md` entries. The
  26 new model-bench tests and the 5 new decision-models tests add up to those counts.
- **Refusal mutants** (`import_decision_pack.py`, test file only; restored and `cmp`-equal):

  | Mutant | Result |
  |---|---|
  | refusal removed (`if False:`) | 11 failed (as reported) |
  | subset only (`not set(votes) <= keys`) | 2 failed (the dropped-key rows) |
  | superset only (`not set(votes) >= keys`) | 1 failed (the extra key) |
  | case-folded keys | 6 failed |
  | refusal applied to message type only | 8 failed |
  | refusal skipped on the guard | 2 failed |

- **Vector mutants** (both fixtures copied aside, restored, and `cmp`-equal to each other and the
  originals):
  - `rngSeed` deleted from the canonical guard line: decision-models' equality test fails (1); it
    passes after a re-copy (45 passed). model-bench: 1 failed. This is the U6.6 done-condition.
  - `"true"` renamed `"True"` on a join-in train item in both files: decision-models' points check
    fails (1), and model-bench's projection test fails (1).
- **Shipped packs:** `packs/*-synthetic/` hold only `prompts/`, with no `items.jsonl`. No pack has
  been imported, so nothing that previously validated can now be refused. Outside the importer and
  tests, no model-bench code reads `labelVotes` (sweep over `*.py`/`*.json`/`*.jsonl`).
- **falkor-chat:** not run (it wipes a shared graph). Read `server/tests/test_decision_contract.py:41`:
  `_VECTOR_FILES` names three other files. No test or script in falkor-chat or model-bench globs
  `decision-contract/`. `scripts/generate_guard_decision_vectors.py:286-288` writes only its own
  three files.

CPG: considered, not relevant — no model-bench or decision-models CPG is loaded (`GRAPHS`). The
paths were read directly.

### The checks asked for

- **1. Is the vector faithful and producible?**
  - **The item key set is the plans' key set, not an accident of the importer.** Plan2 §3.5 adds
    `seedId` and `labelVotes`, plus `threadPosition` on message type. Plan4 §3.7 adds `genRunId`
    and `genRound`, which replace v1.3's generator-call hash. v1.3 §3.4 gives `origin`, and the
    importer docstring and plan6 §3.7 give `itemId`. With the input fields and `expected`, that is
    exactly the importer's `required` tuple. The equal-key-sets rule therefore asks `finalize` for
    nothing the plans do not name. Diagnostics read seed axes from the store by `seedId`, not from
    `items.jsonl` (plan4 §3.6), so nothing downstream needs an extra item key.
  - **The items fit decision-models' schema.** All nine pass `input_field_errors` (run). The guard's
    `understanding: {}` on a `turns` item and `turns: []` on an `understanding` item are the forms
    `presentWhen` requires (`points.py:164-165`, `:218-224`). Message type's `priorTurn` is `null` on
    an opening item and an object on a follow-up.
  - **A `formMismatch` entry on the guard line only is right.** By plan5 §3.5, `formMismatch` is a
    binding error on `missingForm`, and only the guard declares one (`emptinessAxis: missingForm`,
    with `immaterial` among `nonEmptyFor`). join-in has no `seedBindings`. Message type's only binding
    is `presentWhen`, which is a `bindingMismatch`. So a faithful `finalize` cannot write a
    non-guard form entry, and the empty `{}` on those lines is producible.
  - **The vector does constrain the producer in one way the plans leave open:** the value-type
    comparison (P29-M1).
- **2. The refusal.** It sits after the counts check, so `labelVotes` must be counts summing to 3
  first and must then have exactly the role's key set (`import_decision_pack.py:251-255`). It only
  applies to projected (eval) items, as the pre-existing rule for non-eval items does. JSON object
  keys are always strings, so a boolean key cannot arrive from a file. The six mutants above show
  every direction of the set comparison is pinned. Nothing that validated before can now be refused
  (see "Shipped packs").
- **3. Guarded equality.** The new file is a third member of the same parametrized
  `_VECTOR_FILES` equality test (`test_decision_contract.py:41`, `:54-58`). It uses the same
  model-bench-absent skip, and it is also in the raw-line row-count test. The points check
  (`test_the_dataset_shape_items_fit_this_packages_points`) derives its expected vote keys from
  decision-models' own `point.json`, not from the vector or from model-bench's `label_vote_keys`.
  It is therefore an independent check, and it can fail (the planted `"True"` above).
- **4. Growth-stops.** See the section below.
- **5. `model-bench/AGENTS.md:186-189`.** It is wrong and should be fixed in this unit (P29-m1).

### Findings

**P29-M1 (major; `architect`, before U5.10's brief; does not block U6.6). No faithful `finalize`
can pass §3.7's per-entry value-type comparison against the vector as specified.**
- **Evidence (traced by reading; U5.10 does not exist yet):**
  - **Runs.** §3.7 compares each `runs[]` entry's value types "against the vector's entry", and
    the vector has one ingested run with `voidClass`/`voidReason` set to `null`. A void run's
    record carries strings there (`audit.py:699-723` `combine`, `:727-743` `run_record`).
    `run_record` also yields `None` for `claudeCodeVersion`, `model`, `sessionContextSha256`,
    `thinking` and `effort` whenever the transcript lacks them (`audit.py:677-692`).
  - **Items.** §3.7 compares items "per entry in the same way", but it does not say against which
    vector item. Message type's `priorTurn` is `null` on one vector item and an object on another,
    so an eval split holding both opening and follow-up items mismatches whichever single entry is
    chosen.
- **Why it matters:** U5.10's implementer will either redden on any campaign with a void run or a
  follow-up item, or quietly weaken the comparison. Either way the contract check is lost.
- **Fix (a design choice for `architect`):** state in §3.7 how a container entry is compared.
  - **Option (a):** against the union of types the vector shows for that key across its entries.
    The vector then gains a void run beside the ingested one, which revises "one `runs[]` entry".
    Its items already show both shapes for every nullable or empty-able field.
  - **Option (b):** a named list of nullable keys.
  - **Either way,** an entry with no vector counterpart is refused, never passed vacuously. For
    example, a non-empty `formMismatch` on a non-guard line has nothing to compare against.
  - If option (a) is chosen, the vector edit is a small follow-up to U6.6 with both copies re-made.

**P29-m1 (minor; `coder`, this unit). `model-bench/AGENTS.md:186-189` names falkor-chat as "the one
reverse edge" byte-copying `decision-contract/` files.**
- **Evidence:** decision-models' suite already byte-copied `render-guard.jsonl` and
  `input-hash.jsonl` before U6.6. This unit adds `dataset-shape.jsonl`, which only decision-models
  copies.
- **Why it matters:** this is the always-loaded context for anyone editing a canonical vector. As
  written, it tells that editor to expect only falkor-chat's equality test to redden. An edit to
  `dataset-shape.jsonl` then breaks decision-models' suite unnoticed.
- **Fix:** replace the sentence in place, with no history clause. For example: "The reverse edges
  are test-only: falkor-chat's and decision-models' suites each byte-copy the
  `tests/fixtures/decision-contract/` files they list and assert them equal to ours unless
  `model-bench/` is absent (typed-decision-models v1.3 §3.2.2)." `decision-models/AGENTS.md:27-32`
  is already correct.

**P29-n1 (nit; `coder`). The importer test builder still describes a producer shape that §3.7 now
forbids.**
- **Evidence:** `_source_item` (`test_import_decision_pack.py:110-112`) writes `intent` and `split`
  as "producer-side fields the pack never carries". The module docstring calls the builder's output
  "the shape the plans give the producer". The vector's equal-key-sets rule rules both fields out of
  `items.jsonl`.
- **Fix:** keep the fields, because the test that the importer ignores unknown keys is still worth
  having. Relabel them as "extra fields the importer must ignore", and drop "the shape the plans
  give the producer" from the docstring, which now points at the vector anyway.

### Growth-stops

- **`model-bench/README.md`:** one 4-line insertion in the importer paragraph (`@@ -298,0 +299,4`).
  It is new scope, the new refusal and the vector citation, with no rework growth, and every line
  is ≤ 101 characters.
- **Both `HISTORY.md` files:** one new entry each, read by lookup. Their counts reproduce: 26 and 5
  tests; 3625 → 3651 and 666 → 671; and refusal removed → 11 failed.
- **`model-bench/AGENTS.md`:** unchanged, but now wrong (P29-m1).

### Proposed REVIEW.md lines

None.
