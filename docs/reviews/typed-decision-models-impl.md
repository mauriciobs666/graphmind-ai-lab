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
