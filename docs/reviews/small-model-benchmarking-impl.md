# Small-LLM benchmarking tool (`model-bench/`) — S1 implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/small-model-benchmarking.md` §4 S1

## 1. Scope & verdict

**Reviewed:** commit `ab91419` (`feat(model-bench): S1 core — fingerprint, results, stats, report`),
the whole diff against `0522ffd` (S0) — 18 files, +3834 lines, all under `model-bench/`.

**Baseline:** `docs/plans/small-model-benchmarking.md` **v1.4** §4 "S1" and its nine numbered
done-conditions, plus §3.3, §3.4/§3.4.1–3, §3.5, §3.6a, §3.9, §5 and Appendix A;
`docs/plans/small-model-benchmarking-ml.md` **v1.4** §3.2/§3.3/§3.4/§7.1/§7.2/§9;
`docs/reviews/small-model-benchmarking.md` Passes 1–2 (B-1, B-3, N-1, N-2).

**Not reviewed / deferred:** the *statistical validity* of `stats.py`'s formulas — that is the
concurrent `data-scientist` methodology pass. I judged `stats.py` only for engineering correctness,
seam design and faithfulness to `-ml` §3.4's six stated rules, and I say below where I deferred.
No `falkor-chat/` file was read or touched.

**CPG:** considered, not relevant — no CPG exists for `model-bench/` (new component) and
`cpg_falkorchat` is stale and covers different code, per the dispatch brief; every finding here
comes from reading the files, running the suite, and 29 source mutations.

**Verdict: needs changes.** — 1 blocker, 6 majors, 7 minors, 4 nits.

The blocker is one rendered-output defect of the same species the author already caught and fixed
(the CI-orientation bug): the family-wise correction that `-ml` §3.3 and plan §3.3(ii) make
**mandatory** is *printed* but not *applied*, and the table that prints it contradicts the verdicts
it sits beside. The majors are, with one exception, gaps in the **test** side rather than the
source: six of my mutations survived the full suite, and every one of them lands on a stated
honesty guarantee. The core mechanisms themselves — the fingerprint refusal, quarantine-on-read,
schema-keyed validation, the `armKind` forbid half, the missing-headline path and DC-5(c)'s
unit-id resolution — all hold under direct attack.

**What I ran, verbatim, from `model-bench/`:**

| command | result |
|---|---|
| `.venv/bin/python -m pytest -q` | `233 passed in 0.51s`, exit 0 |
| `.venv/bin/ruff check .` | `All checks passed!`, exit 0 |
| 29 source mutations (copy of the tree in a scratch dir; the repo working tree was never modified) | 19 killed, **10 survived** — see §2 and Appendix A |

Zero network imports in `modelbench/` (Appendix A.1). The git working tree was not mutated.

## 2. Findings

### Blocker

**B-1 — Holm–Bonferroni is printed but never applied; the family-wise table contradicts the
verdicts it sits beside.**

`modelbench/report.py:280-296` computes `stats.holm_thresholds(p_values, alpha=0.05)` and prints one
threshold per metric, but `modelbench/report.py:257-265` calls `stats.verdict(...)` **without**
`alpha_step`, so every metric is decided at `resolving.alpha == 0.05/k` — plain Bonferroni, the most
conservative step for all k. `stats.verdict`'s `alpha_step` parameter
(`modelbench/stats.py:413, 453`) exists for exactly this and is passed by nothing; mutating it away
(`alpha = resolving.alpha`) leaves all 233 tests green. `stats.holm_thresholds` also omits Holm's
step-down stopping rule, and mutating it to return a constant `alpha` for every metric is likewise
green.

`-ml` §3.3 is explicit that Holm *is* the decision procedure ("order the p-values; test the smallest
at α/k, the next at α/(k−1), … stopping at the first non-rejection — **and** print the adjusted
threshold beside each p-value"), and plan §3.3(ii) restates it as mandatory. What ships prints one
rule and applies another. The failure is visible in rendered output, reproduced here at k=2 with
b=8/c=0 and b=6/c=0 (full render in Appendix A.2):

```
### falseSuspendRate
Not distinguishable at this sample size. … does not reach alpha=0.025 (b=6, c=0, p=0.031). …
### Family-wise error control
| falseAdvanceRate | 0.008 | 0.0250 |
| falseSuspendRate | 0.031 | 0.0500 |     <- p <= its own printed threshold, yet "not distinguishable"
```

A reader applying the printed rule concludes `falseSuspendRate` cleared its Holm step; the verdict
two paragraphs above says it did not. The error direction is conservative (no false "better" is
produced), which is why I stop short of calling it a validity defect — but a measuring instrument
emitting two contradictory statements about the same number is the exact defect class the author
already treated as must-fix. `guard-judge` (k=2) is the pack this fires on, and it lands at S4.

*Suggested fix (implementer):* in `report.py`, rank the family by p-value and pass
`alpha_step=holm_step[i]` into `verdict()`, applying the step-down stop (once one metric fails its
step, every later one is non-rejected regardless of its own p); keep `resolving.alpha` at `0.05/k`
so Rule 4's precondition 3 still holds. Add two tests: one pinning `holm_thresholds([0.008, 0.031])
== [0.025, 0.05]` by value, and one on the rendered k=2 report asserting the verdict and the printed
threshold agree in the case above. *Note for `data-scientist`:* whether Holm's step-down or plain
Bonferroni is the intended decision is a methodology call the note already made — I am reporting the
implementation's divergence from it, not re-opening it.

### Majors

**M-1 — `load_history` silently drops a record whose `packId` is blank or absent: neither `valid`
nor `invalid`.**

`modelbench/results.py:394` filters by pack *before* validating:
`if run.fingerprint.get("packId") != packId and schema in REQUIRED_BY_SCHEMA: continue`. `packId` is
a `REQUIRED_NONEMPTY` field, so a record whose `packId` was blanked or deleted on disk fails the
`!=` test, is skipped, and appears in **neither** returned list. Reproduced (Appendix A.3):

```
A  blanked packId  -> valid: []  invalid: []
A2 absent  packId  -> valid: []  invalid: []
```

AC-2's guarantee is "excluded on read **and named**", and this module's own docstring says an
unreadable record "is a finding, not an absence". Here it is an absence: the comparison quietly
loses an arm and the report says nothing. The suite does not catch it because DC-1's read-side test
blanks only `kvCacheSetting`; the exhaustive per-field loop in `tests/test_fingerprint.py` runs
against `Fingerprint.validate()`, never through `load_history`.

*Suggested fix:* validate the fingerprint first and quarantine anything with a `packId` problem
before applying the pack filter — i.e. skip only when `packId` is present, non-empty and different.
Add a read-side test parametrized over the required fields, not just one of them.

**M-2 — `RunResult.designEffect`/`basis` carry defaults, rebuilding gate B-1's "default by
omission" shape at the record seam.**

`modelbench/results.py:242-243`: `designEffect: float = 1.0` and `basis: Basis = "assumed"`. The
whole point of `-ml` §3.4 Rule 2, which `stats.resolving_power` honours exactly, is that "a default
of `1.0` would rebuild B-1 by omission: the caller who forgets clustering is exactly the caller the
gate found". `RunResult` is the object S2's runner constructs, and it lets that caller omit the
design effect and get the anti-conservative value. `basis` defaulting to `"assumed"` is fail-safe
and correct; `designEffect = 1.0` is not.

I verified the fix is free: deleting both defaults (making the two fields required) leaves the suite
at **233 passed** unchanged, because `from_dict` already supplies `d.get("designEffect", 1.0)` /
`d.get("basis", "assumed")` for legacy records and `conftest.run()` passes both explicitly.

*Suggested fix:* drop both defaults on the dataclass; keep them in `from_dict` only, where they mean
"a record written before these fields existed".

**M-3 — the fail-safe basis/design-effect propagation — decision 4's entire justification — is
untested; two mutations survive.**

`modelbench/report.py:242-243`:

```python
design_effect = max(a.designEffect, b.designEffect)
basis = "by-construction" if a.basis == b.basis == "by-construction" else "assumed"
```

Both mutate green: forcing `design_effect = 1.0`, and forcing `basis = "by-construction"`
unconditionally — 233 passed each time. Every report fixture uses `design_effect=1.0,
basis="by-construction"` (`conftest.run()`'s defaults), so the clustered/assumed branch of
`verdict()` is exercised **only** through direct `stats.verdict` calls in `tests/test_stats.py`,
never through `compare_report`. This is the mechanism plan review N-2 asked for ("wire the probe
outcome to `basis`; a non-identical probe degrades to `assumed`, which moves McNemar out of the
decision seat"), and at report level nothing holds it in place.

*Suggested fix:* one report test with `a.basis="by-construction"`, `b.basis="assumed"` asserting the
rendered line says `decided by: cluster-bootstrap` and carries the anti-conservative label; one with
`a.designEffect=1.0, b.designEffect=2.0` asserting the printed design effect is `2.00`, not `1.00`.

**M-4 — the required-field and forbidden-field contracts are pinned only by parametrizing over the
code under test, so a set that *shrinks* stays green.**

`tests/test_fingerprint.py:249,256,295` parametrize over `REQUIRED_BY_SCHEMA[1]["model"]` and
`FORBIDDEN_BY_ARM_KIND["deterministic"]` — the very objects under test. Removing an entry removes
its test case rather than failing one:

- deleting `"loadedContextLength": _NONEMPTY` from `_MODEL_SCHEMA_1` → **230 passed** (three cases
  silently uncollected, zero failures);
- subtracting `{modelType, modelCapabilities, modelCapabilitiesPresent}` from the `deterministic`
  forbidden set → **230 passed**.

The second case is exactly the author's **decision 3** — forbidding the three model-catalog fields
that plan §3.4.1's enumeration omits while its prose says "forbids every model field". That decision
is correct (see §4 below) and is currently held in place by nothing but the set-difference
expression itself. A future edit that follows the plan's literal enumeration reverts it in green.

*Suggested fix:* assert the sets themselves against literals in the test file — e.g.
`assert set(REQUIRED_BY_SCHEMA[1]["model"]) == {…the 30 names of `_MODEL_SCHEMA_1`, per plan §3.4.2…}` and
`assert FORBIDDEN_BY_ARM_KIND["deterministic"] == frozenset({…})` — then keep the parametrized loops
for the per-field behaviour. A shrinking set then fails loudly instead of shrinking the suite.

**M-5 — the paired-*n* precondition intersection (`-ml` §4.3, risk R2) is untested; the mutation
survives.**

`modelbench/report.py:77-78` drops an item from the pair when either arm's `scoreable[metric]` is
false, which is `-ml` §4.3's "a precondition failure must never be laundered into the numerator".
Replacing the condition with `if False:` leaves 233 tests passing. No fixture in the suite ever sets
`scoreable=False`, even though `conftest.item()` takes a `scoreable` parameter for it. R2 is rated
**high** in `-ml` §10 ("a model that collapses early scores *better* on the conditional counts").

*Suggested fix:* a report test where arm B's items 0–9 have `scoreable[metric] = False`, asserting
the rendered `n=` is 30 rather than 40 and that the dropped items appear in neither numerator nor
denominator.

**M-6 — a comparison with fewer than two arms prints a false reason, and `--models` is how a user
gets there.**

`modelbench/report.py:143-150`: `_comparison_pair` returns `None` both when `len(runs) < 2` and when
both arms are deterministic, and `report.py:223-230` prints one explanation for both:

```
_None: no verdict is computed between two deterministic arms — a deterministic arm is reproducible
from its pack version and arm parameters, so a difference between two of them is a pack change,
not a finding (§3.4.1)._
```

Reproduced with a single model arm, and with zero arms (Appendix A.4). The route in is
`modelbench/cli.py:97`: `candidates = [by_key[m] for m in wanted if m in by_key]` silently drops a
`--models` key that matches no stored run, so `--models cand,incumbnet` renders a one-arm report
asserting a deterministic-arm reason that is untrue. Mutating the filter away entirely
(`candidates = list(candidates)`) is also green — `test_compare_selects_the_named_models` asserts
only `"third" in out`, which holds whether or not the filter runs, so that test passes while testing
nothing.

*Suggested fix:* split the two cases in `_comparison_pair` (return a reason, or raise) and render a
distinct line for "fewer than two arms selected"; in `cli.py`, exit `2` naming any `--models` key
with no stored run; strengthen the CLI test to `assert "cand" not in out`.

### Minors

**m-1 — a record belonging to a *different* pack is quarantined into this pack's exclusion block
when its schema is unknown.** `results.py:394` short-circuits the pack filter on
`schema in REQUIRED_BY_SCHEMA`, so an `embedder-graphrag-retrieval` record at
`benchSchemaVersion: 99` is reported as an AC-2 exclusion in a `guard-judge` comparison (reproduced,
Appendix A.3 case B). Same for the `unparseable` branch, which runs before any pack check. Defensible
for `unparseable` (the record cannot declare its pack); not for `unknown_schema`, where `packId` is
right there and readable. *Fix:* apply the pack filter to `unknown_schema` records whose `packId` is
present and different; keep it off `unparseable`.

**m-2 — a content-hash-only divergence is labelled "unpaired (different pack version)" while the
banner above it says the versions match.** `report.py:126-134`. One report, two adjacent lines,
contradicting each other. *Fix:* `"unpaired (different pack version or content hash)"`, or two
distinct labels. The existing test covers only the version case.

**m-3 — `report._unit_ids` is dead code, and the module docstring names it as the mechanism.**
`report.py:54-57` is defined and called by nothing (`grep` over `modelbench/` and `tests/`); the real
resolution is the inline `index = pack.analysisUnitIndex` at `report.py:72,79`. The docstring at
`report.py:18` reads "What closes it is `_unit_ids` below", which is false as written — a reader
auditing gate N-1's closure is pointed at a function that never runs. ruff's `E,F,W,I` selection does
not flag an unused module-level private function. *Fix:* either call `_unit_ids` from `_paired_rows`
or delete it and correct the docstring to name `_paired_rows`.

**m-4 — untested CLI/derived surfaces.** Two more survivors: `models --tested --role <r>`'s filter
removed entirely (`results.py:509`) → green, no test passes `--role`; and `index.csv`'s
`latencyMsP95` computed at the 50th percentile → green, the index test asserts only the header and
the runId. Low stakes, but `--role` is a shipped flag with zero coverage.

**m-5 — `PackRef.contentHash` is vestigial at S1.** `pack_ref_from_manifest` sets it to `""` by
design (`packs.py:129`) and nothing reads it — the AC-3 banner correctly reads each run's own
`fingerprint.packContentHash`. Harmless now; a field that is always empty is a trap for the S2 author
who fills it in and expects the report to use it. *Fix:* a one-line comment at the field, or make it
`str | None = None` so "not yet computed" is expressible.

**m-6 — `compare` filters history by the pack *directory name*, not the manifest's `packId`.**
`cli.py:117` passes `args.pack` to `load_history`, while `pack.packId` is available two lines above.
They coincide by the §3.3 convention (`packs/<pack-id>/`), so this is latent, not live. *Fix:* pass
`pack.packId`.

**m-7 — `store()` fails with a raw `FileNotFoundError` when `runId` is not a bare filename.**
`results.py:355-356` does `target / f"{run.runId}.json"` with no check. Plan §3.5 specifies the
`modelSlug` sanitisation precisely because real model keys contain `/`
(`qwen/qwen3-4b-2507`), and the slugging is S2's runner. Today an unslugged id raises
`FileNotFoundError: …/runs/pack-qwen/qwen3-4b-2507-….json` from `pathlib` — loud, but not a named
reason, and a `runId` segment that happens to name an existing directory would write outside
`runs/`. *Fix:* reject a `runId` containing a path separator in `store()`, citing §3.5's slug rule.

### Nits

**n-1** — `tests/test_results.py:161` is a tautology:
`assert "packId" not in inspect.signature(load_history).parameters or True` is `True` for every
possible input. The line below it does the real work; delete this one.

**n-2** — `Fingerprint` is `frozen=True` but `fields` is a live `Mapping` the caller still holds, and
`__hash__` (`fingerprint.py:199-200`) hashes only the sorted *field names*, so two fingerprints
differing in every value collide. Correct (equal objects hash equal) but degenerate. *Fix:* wrap in
`MappingProxyType(dict(...))` in `__post_init__`, or hash the sorted items.

**n-3** — `RunResult.from_dict` (`results.py:267`) defaults a missing `aggregates` block to
`{"kind": "classification"}`, silently fabricating an empty `ClassificationAggregates` for a record
that has none. In a module whose thesis is "an unreadable record is a finding, not an absence", this
one absence is repaired instead of reported.

**n-4** — `resolving_power_line`'s fourth sentence hardcodes "generalization to **unwritten
scripts**" (`report.py:120-121`) for every unit kind, so an item-level pack renders "conditional on
the 40 items … generalization to unwritten scripts". `-ml` §7.2 publishes the string only for the
tool-caller pack, so this is not a contract breach — but the `_SAMPLE_NOUN` map already exists two
lines up and would carry it.

## 3. Done-condition audit

I checked each condition by mutating the thing it claims to protect, not by reading the test name.

| DC | holds? | how I checked |
|---|---|---|
| **1** AC-2 exclusion + the three §3.4.2 tier states | **yes, with a hole** | `[]` valid / `""` invalid / `null` invalid all pinned per field; read-side quarantine killed by mutation M24. Hole = **M-1** (`packId`). |
| **2** AC-3 banner on `packVersion` and on `packContentHash` alone, comparison still rendered | **yes** | Both banner mutations killed (M21); suppressing the verdict section on a version mismatch is killed (M23). Label wording: **m-2**. |
| **3** AC-4 wording, and 40/40 vs 34/40 is *not* it | **yes** | `test_the_forty_of_forty_case_is_distinguishable` asserts both directions; the three §3.2e strings are pinned verbatim in `test_stats.py:378-412`. |
| **4** `-ml` fixtures to the note's tolerance + Rule 1 raise + ρ=1 identity | **yes, with a stated deviation** | 5 rows × p at 1e-12 and bounds at full published precision; ρ=1 identity and its inverse both present. Tolerance deviation verified correct — §4 item 1. |
| **5** MDD not constant, not naive; B-1 detector | **yes** | `TypeError` on `int` for both `min_detectable_difference` and `…_exact`; ceiling-vs-round mutation kills 8 tests. See §4 item 4 on the report-level half. |
| **5(c)** which key is the unit id | **yes — all three assertions, and (1) does independent work** | Mutating `pairingKey[index]` → `pairingKey[-1]` still raises `DuplicateAnalysisUnit`, so assertion (2) stays green — and assertion (1) fails. Mutating to `item.itemId` fails the test too. The spy captures the argument actually passed to `from_units`, and the negative control (48 unique conversation ids accepted) is real. |
| **5b** the §7.2 verbatim line | **yes** | Asserted as one string; the `n_eff < 20` power-ceiling branch mutation is killed. |
| **6** `armKind` forbid half | **yes, with a hole** | The forbidden loop's removal kills 24 tests; `modelKey: "bm25"` fails on write; two deterministic arms are never ranked (mutation killed). Hole = **M-4** (the set can shrink in green). |
| **7** schema versioning, both directions | **yes** | `test_an_older_known_schema_record_stays_valid` moves the current schema to 2 *and* asserts the schema-2 record with a missing field is invalid — this is the test the author's own mutation pass rewrote, and it now discriminates. |
| **8** `headlineMetric: null` | **yes** | Forcing the headline branch to `if True:` fails the test; omission-vs-null is pinned in `metrics_from_manifest`. |
| **9** `--negative-control` smoke check | **yes** | Asserts `b=0, c=0` and is labelled a smoke check in its own docstring, as DC-9 requires. |

## 4. The seven items the brief asked me to verify

1. **(→ `data-scientist`) The MOVER-D tolerance is genuinely unassertable as written — confirmed.**
   `-ml` §3.2c/§9.1 mandate 1e-9 absolute *on the proportion*, but the §3.2c table publishes bounds
   at 4 dp of a percentage point = 1e-6 as a proportion. The `34,6,0,0` lower bound computes to
   `3.176286944306023` pp = `0.03176286944…`; the published `3.1763` pp = `0.031763` differs by
   **1.31 × 10⁻⁷**, two orders above the mandated tolerance. No implementation can satisfy 1e-9
   against that table. The author's substitute — equality at the full published precision,
   `round(bound*100, 4) == published` — is a tolerance of ±5 × 10⁻⁷ proportion, tighter than the
   published resolution and looser than double-precision noise. **Correct call.** And the
   load-bearing claim checks out: at `z = 1.96` the same bound is `3.176004750966589` pp, a
   divergence of `2.82 × 10⁻⁴` pp (matching the note's own "at most 3.0 × 10⁻⁴ pp, largest on this
   row"), which rounds to `3.1760` and breaks the assertion —
   `test_the_pinned_z_constant_is_load_bearing_at_this_tolerance` pins exactly that. *For
   `data-scientist`:* the fix is one clause in `-ml` §3.2c/§9.1 — either state the tolerance as
   "equality at the published 4-dp-of-pp precision", or republish the table at ≥10 significant
   digits and keep 1e-9.

2. **(→ `architect`) `PackRef` needed `pairingKey` and `analysisUnit` — confirmed, and Appendix A is
   the stale document.** §3.3 (v1.4) says "`report.py` resolves the unit id from this field … no
   call site chooses it, and there is no parameter through which a caller could", and DC-5(c)
   requires the fixture to build a `PackRef` declaring both. Appendix A's five-field `PackRef`
   predates that block and cannot express it. The two added fields are the minimum. *Fix:* amend
   Appendix A rather than the code.

3. **(→ `architect`) Forbidding `modelType`/`modelCapabilities`/`modelCapabilitiesPresent` on a
   `deterministic` arm — correct, and §3.4.1's enumeration is the defect.** §3.4.1's prose ("forbids
   every model field") and its stated rationale ("recording a KV-cache setting beside a BM25 score
   would imply the score depends on it") both cover the three; the enumeration omits them.
   Implementing the set as `frozenset(_MODEL_SCHEMA_1) - frozenset(_DETERMINISTIC_SCHEMA_1)` derives
   the rule instead of transcribing it, which is the better shape — it cannot go stale when a model
   field is added at schema 2. *Fix:* amend §3.4.1's enumeration to match the prose (and note the
   test gap M-4, which is the only thing making this decision fragile).

4. **`RunResult` gaining `designEffect`/`basis` — necessary, and I would have flagged their absence
   as a blocker.** `-ml` §3.4 Rule 4 decides *which instrument may decide* from exactly these two,
   `report.py` cannot recompute either (the basis comes from the determinism probe, which only the
   runner sees — plan §5 test 12b, and plan-review N-2 says the same), and `resolving_power` refuses
   to be called without them. Additive to a seam S3 doesn't exist against yet, and both survive
   `to_dict`/`from_dict` round-trip with legacy defaults. Two caveats, filed above: the defaults
   themselves (**M-2**) and the fact that the propagation logic is untested (**M-3**). On DC-5's
   clause "report.py refuses to render one when the required input is absent" — with a default of
   `1.0` the input can never *be* absent, so that clause is satisfied only vacuously; M-2's fix is
   also the fix for that.

5. **`FieldProblem.reason` gaining `"unknown"` — correct and minimal.** An unrecognized `armKind` and
   a future `benchSchemaVersion` are genuinely neither `absent`, `empty`, `null` nor `forbidden`;
   forcing either into one of the four would mislabel a record the build simply cannot interpret,
   and `InvalidRecord.reason == "unknown_schema"` (which Appendix A *does* define) would have no
   field-level counterpart. Appendix A should gain the fifth value.

6. **`modelbench/packs.py` contains no pack loader — verified, claim holds.** I read the whole file
   (136 lines). There is no `hashlib` import and no content hashing, no `ast`/`importlib` and no
   import allowlist walk, no data-file read and no row-count identity check — the three things S2's
   `load_pack`/`validate_pack` own. What is there is `PackRef`, `PackMetrics`,
   `metrics_from_manifest` (§3.3's metrics rules), `check_sampling_contract` (§3.3's *structural*
   route only, explicitly deferring route (ii) to S2) and `pack_ref_from_manifest`, a plain
   `json.loads` of `pack.json` that leaves `contentHash` empty. Creating the module was the right
   call: `PackRef` lives in `packs` per Appendix A, and `compare` genuinely cannot resolve its
   analysis unit or verdict family without the manifest. **One seam risk for S2 to notice, not a
   defect:** `check_sampling_contract` is now enforced in two places (at manifest read, and again
   fail-closed in `compare_report`), and S2's `validate_pack` will be the third — S2 should call the
   existing function rather than re-implement the rule.

7. **§5 test 4 (`packs.content_hash()`) is genuinely out of S1's reach.** S1's `Create` list in §4
   does not include `packs.py` at all (it is S2's), the hash is defined in §3.3 as SHA-256 over every
   file in a pack directory, and no pack directory exists until S3. §5's numbered list is not
   stage-scoped — tests 7–12 are equally undeliverable at S1 — so its absence is sequencing, not an
   omission. The AC-3 banner S1 *does* ship reads each run's recorded `packContentHash`, which is the
   right source at this stage and is tested.

## 5. What's solid

- **DC-5(c) is the best-built test in the diff.** The spy captures the argument actually handed to
  `from_units`, so assertion (1) fails independently of the raise — I proved it by mutating the
  index to `pairingKey[-1]`, where the guard still fires and assertion (1) is the only thing that
  catches it. The negative control is a real control, not a restatement.
- **`stats.py` is a faithful, readable transcription of `-ml` §3.4's six rules**, with every
  no-default keyword-only input, the `n_effective: float` refusal, the exact-bisection MDD with a
  genuine ceiling, and the duplicate guard moved into `__post_init__` so it holds on every
  construction route rather than only through `from_units` — a small improvement on the note.
- **The closed union of aggregate dataclasses does what §3.5 claims**: there is no field on
  `ClassificationAggregates` to hold a pooled accuracy and none on `ToolCallAggregates` to hold a
  blended percentage, so the refusals are type facts a reviewer can check without running anything.
- **The CI-orientation defect the author found by reading rendered output** was real and its fix is
  right, including the deliberate choice to keep the *non*-significant strings in A−B orientation so
  the printed difference always sits inside the printed interval.
- **The schema-2 test rewrite** genuinely discriminates now: it moves `BENCH_SCHEMA_VERSION` to 2 and
  asserts both directions in one load, so an implementation that validates against the current
  schema fails it.
- **Documentation is accurate and proportionate.** `model-bench/AGENTS.md` was rewritten, not
  appended to; `HISTORY.md`'s entry states the two decisions, the defect and the exact verification
  commands; both stated non-features and the S2 boundary are asserted by a test rather than promised.

## 6. Open questions

1. **Does the Holm fix (B-1) belong to S1's re-gate or to S4?** No shipped pack has `k > 1` until
   `guard-judge` at S4, so a coordinator could reasonably defer it. My recommendation is to fix it
   now — the code is warm, the seam (`alpha_step`) is already built, and a contradictory rendering
   that nothing raises on is precisely what gets rediscovered as a defect six months later.
2. **`_percentile` (both copies — `stats.py:152` and `results.py:441`) uses nearest-rank rounding
   rather than interpolation.** For B=10 000 bootstrap draws this is invisible; for
   `latencyMsP95` over a handful of items it is not. The plan says nothing about the definition. I
   have not filed it as a finding because no requirement pins it — but S2 should decide and write it
   down before latency figures start being compared across runs. `data-scientist`'s call, not mine.

---

## Appendix A — evidence

### A.1 — verification performed

Working directory `model-bench/` throughout; the repo working tree was never modified. Mutation
testing ran against a copy of `modelbench/` + `tests/` + `pyproject.toml` in the session scratchpad.

- `.venv/bin/python -m pytest -q` → `233 passed in 0.51s`, exit 0.
- `.venv/bin/ruff check .` → `All checks passed!`, exit 0.
- Import sweep over `modelbench/`: no `urllib`, `socket`, `http`, `requests`, `httpx` or
  `subprocess`; stdlib only (`math`, `random`, `json`, `csv`, `argparse`, `sys`, `dataclasses`,
  `pathlib`, `typing`, `functools`, `types`, `datetime`, `inspect`). Offline claim confirmed.
- `grep -rn "_unit_ids" modelbench/ tests/` → one definition, one docstring mention, zero call sites.

**29 mutations, 10 survivors.** Survivors, each named in a finding above:

| # | mutation | result | finding |
|---|---|---|---|
| M1 | `report.py` basis never degrades to `"assumed"` | 233 passed | M-3 |
| M2 | `report.py` `design_effect` forced to `1.0` | 233 passed | M-3 |
| M3 | `cli.py` `--models` filter removed | 233 passed | M-6 |
| M4 | `report.py` `scoreable` precondition filter removed | 233 passed | M-5 |
| M5 | `stats.holm_thresholds` returns a constant `alpha` | 233 passed | B-1 |
| M6 | `stats.verdict` ignores `alpha_step` | 233 passed | B-1 |
| M8 | `deterministic` forbidden set loses the three catalog fields | 230 passed (3 uncollected) | M-4 |
| M18 | `loadedContextLength` dropped from `_MODEL_SCHEMA_1` | 230 passed (3 uncollected) | M-4 |
| M27 | `index.csv` `latencyMsP95` computed at p50 | 233 passed | m-4 |
| M28 | `models_with_stored_results` `--role` filter removed | 233 passed | m-4 |

Killed (19): unit id → `itemId` (1 fail) and → `pairingKey[-1]` (2 fails); headline synthesised when
null; `cluster_bootstrap` resampling observations; MDD ceiling → round (8 fails); `store()` skipping
validation (2); `check_sampling_contract` bypassed; `observable_floor` hardcoded to `6/n` (4);
power-ceiling sentence always printed; two deterministic arms ranked; `validate()`'s forbidden loop
skipped (24); invalid block suppressed (2); hash banner removed; schema-span line removed; verdicts
suppressed on version mismatch; `load_history` never re-validating (3); exploratory label dropped;
marginal overlap forced `False`; same-day report sequence capped at `-01`.

### A.2 — B-1 reproduction (k = 2, `guard-judge` shape)

Two hand-built arms over 40 paired items: `falseAdvanceRate` at b=8/c=0 (p = 0.0078125),
`falseSuspendRate` at b=6/c=0 (p = 0.03125), pack `verdictMetrics = [falseAdvanceRate,
falseSuspendRate]`, `headlineMetric = null`. Rendered output, elided:

```
### falseAdvanceRate
cand is better than incumbent on falseAdvanceRate: +20.0 pp (95% CI [7.1, 34.8] pp), n=40 paired
items (unit: item, design effect 1.00), McNemar exact p=0.008 (b=8, c=0).

### falseSuspendRate
Not distinguishable at this sample size. The effect-size interval [3.2, 29.1] pp excludes zero but
the exact paired test does not reach alpha=0.025 (b=6, c=0, p=0.031). Reported as not
distinguishable: the exact test is the decision rule.

### Family-wise error control
Holm–Bonferroni across the 2 pre-registered verdict metrics; every figure above is computed at
alpha=0.025.

| metric            | McNemar p | Holm-adjusted threshold |
| falseAdvanceRate  | 0.008     | 0.0250                  |
| falseSuspendRate  | 0.031     | 0.0500                  |
```

Under Holm, `0.008 ≤ 0.025` rejects and `0.031 ≤ 0.050` then rejects; the report declares the second
metric not distinguishable while printing the threshold it cleared.

### A.3 — M-1 / m-1 reproduction (`load_history`)

One valid `guard-judge` record stored via `store()`, then hand-edited on disk:

```
A  fingerprint.packId = ""                 -> valid: []           invalid: []
A2 fingerprint.packId deleted              -> valid: []           invalid: []
B  packId=embedder-…, benchSchemaVersion=99 -> valid: []          invalid: [('other-pack', 99, 'unknown_schema')]
C  file truncated to half its bytes         -> invalid: [(None, 'unparseable')]
```

Case B is a record belonging to a different pack, surfaced in this pack's exclusion block (m-1).
Cases A/A2 are the silent drop (M-1).

### A.4 — M-6 reproduction

```
ONE arm only  -> '_None: no verdict is computed between two deterministic arms — …'
ZERO arms     -> '_None: no verdict is computed between two deterministic arms — …'
hash-only div -> 'Comparison kind: **unpaired (different pack version)** (§3.7).'   [m-2]
```

### A.5 — M-2 verification

Removing both defaults from `RunResult` (`designEffect: float`, `basis: Basis`, no `= …`) and
re-running the suite in the scratch copy: **233 passed in 0.49s**, unchanged. The defaults carry no
load at S1 and can be dropped without touching a test.
