# Small-LLM benchmarking tool (`model-bench/`) — S1 implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/small-model-benchmarking.md` §4 S1

**Pass 1** gated `ab91419` (needs changes). **Pass 2** re-gated `3ad27d3` (approve with
suggestions). **Pass 3** re-gates `95b4c88` — jump to [`## Pass 3`](#pass-3--2026-09-03) for the
current verdict; the earlier passes are kept intact because the three are meant to be read together.

## Pass 1 — 2026-09-03

### 1. Scope & verdict

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

### 2. Findings

#### Blocker

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

#### Majors

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

#### Minors

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

#### Nits

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

### 3. Done-condition audit

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

### 4. The seven items the brief asked me to verify

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

### 5. What's solid

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

### 6. Open questions

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

### Appendix A — Pass 1 evidence

#### A.1 — verification performed

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

#### A.2 — B-1 reproduction (k = 2, `guard-judge` shape)

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

#### A.3 — M-1 / m-1 reproduction (`load_history`)

One valid `guard-judge` record stored via `store()`, then hand-edited on disk:

```
A  fingerprint.packId = ""                 -> valid: []           invalid: []
A2 fingerprint.packId deleted              -> valid: []           invalid: []
B  packId=embedder-…, benchSchemaVersion=99 -> valid: []          invalid: [('other-pack', 99, 'unknown_schema')]
C  file truncated to half its bytes         -> invalid: [(None, 'unparseable')]
```

Case B is a record belonging to a different pack, surfaced in this pack's exclusion block (m-1).
Cases A/A2 are the silent drop (M-1).

#### A.4 — M-6 reproduction

```
ONE arm only  -> '_None: no verdict is computed between two deterministic arms — …'
ZERO arms     -> '_None: no verdict is computed between two deterministic arms — …'
hash-only div -> 'Comparison kind: **unpaired (different pack version)** (§3.7).'   [m-2]
```

#### A.5 — M-2 verification

Removing both defaults from `RunResult` (`designEffect: float`, `basis: Basis`, no `= …`) and
re-running the suite in the scratch copy: **233 passed in 0.49s**, unchanged. The defaults carry no
load at S1 and can be dropped without touching a test.

---

## Pass 2 — 2026-09-03

### Scope & verdict

**Re-gated:** commit `3ad27d3` (`fix(model-bench): close both S1 gates — 296 tests, zero surviving
mutations`), the whole diff against `ab91419` — 15 files, +1947/−169. Baseline: Pass 1's 18 findings
(1 blocker, 6 majors, 7 minors, 4 nits), plus the concurrent `data-scientist` gate's findings, which
this commit closes in the same pass and which I judged only where they changed a **seam or a test**.

**Deferred to `data-scientist`, as briefed** — I did not re-derive any of them: Rule 7's
construction, `paired_cluster_bootstrap`'s `sqrt(DEFF)` inflation, the floor truncation direction,
and the α/k-versus-Holm-step question. One observation for that pass is at the end of this section.

**CPG:** considered, not relevant — still no CPG for `model-bench/`; Pass 2 is a diff read plus
execution, and every claim below comes from a command I ran.

**Verdict: approve with suggestions.** — **0 blockers, 1 major, 2 minors, 2 nits**, all new; all 18
Pass 1 findings are fixed, verified individually rather than accepted. **S2 can be dispatched.**

**What I ran, from `model-bench/`:**

| command | result |
|---|---|
| `.venv/bin/python -m pytest -q` | `296 passed in 2.08s`, exit 0 |
| `.venv/bin/python -m pytest -q -m ""` | `296 passed` — nothing gated out, confirming the coordinator's check |
| `.venv/bin/ruff check .` | `All checks passed!`, exit 0 |
| **my 10 Pass 1 survivors, re-run** | **10/10 killed** |
| 13 fresh mutations on the new code | 12 killed, 1 survived (P2-1) |
| 11 further mutations on new guards/edges | 8 killed, 2 survived (P2-2, P2-4) |

### New findings

**P2-1 (major) — a `"measured"` basis at design effect 1.0 is untested at Rule 4's branch
condition, and widening the branch to admit it survives the suite.**

`modelbench/stats.py:610` is `mcnemar_may_decide = resolving.design_effect == 1.0 and
resolving.basis == "by-construction"`, which is Rule 4 exactly. Mutating it to
`resolving.basis in ("by-construction", "measured")` — letting a measured basis into the McNemar
seat — leaves **296 passed**. `test_an_assumed_basis_also_moves_the_decision_off_mcnemar` covers
`"assumed"`; the third enum value has no test at this boundary. The code is right and the gap is in
the suite, but `"measured"` became live in *this* commit (`report.py:371`'s `min()` over
`_BASIS_STRENGTH` now preserves it instead of collapsing to `"assumed"`), and S2's runner is what
will start producing it. *Fix:* one `verdict()` test at `deff=1.0, basis="measured"` asserting
`decided_by == "cluster-bootstrap"`, and one report test with both arms `"measured"` asserting the
same — the mirror of the two tests that already exist for `"assumed"`.

**P2-2 (minor) — `Fingerprint.validate()` and `load_history` disagree about
`benchSchemaVersion: true`, so `store()` writes a record the reader immediately quarantines.**
`load_history` gained `isinstance(schema, bool)` (`results.py:447`); `validate()` (`fingerprint.py`)
did not, and `True in REQUIRED_BY_SCHEMA` is `True` because `True == 1`. Verified: `validate()`
returns `[]`, `store()` writes the file, `load_history` returns it as `unknown_schema` — with
`InvalidRecord.benchSchemaVersion=True` in a field typed `int | None`. Removing the `load_history`
guard is also green, so neither side is tested. *Fix:* move the bool check into `validate()` beside
the existing `benchSchemaVersion` / `reason="unknown"` branch, so both enforcement points agree;
add one test.

**P2-3 (minor) — `holm_steps`' `None`-filter plus an un-`strict` `zip` can drop a metric silently.**
`stats.py:759` ends `return [s for s in steps if s is not None]` (a type narrowing — every index is
assigned today), and `report.py:401` consumes it as `zip(tables, steps, tallies)`. If the ladder ever
returns short, `zip` truncates and a pre-registered verdict metric vanishes from the report with no
error. This is the public API S2 wires against. *Fix:* `assert len(steps) == k` (or build the list
without a filter) in `holm_steps`, and `zip(..., strict=True)` in `report.py` — the component is
3.12, so `strict=` is available.

**P2-4 (nit) — two-thirds of `store()`'s runId guard is unreachable and none of it is tested.**
`results.py:390`'s `run.runId in {"", ".", ".."}`: `Path(".").name` and `Path("..").name` are both
`""`, so `"." != ""` and `".." != ""` are already caught by the first clause; only `""` needs the
set. Dropping the whole clause is green. *Fix:* reduce to `or not run.runId`, and add the empty-id
case to the existing runId test.

**P2-5 (nit) — `packs.py:120`'s docstring still says `contentHash` "is left empty here" while the
code now assigns `None`.** The sentence two lines below (`None` until S2's `load_pack`) is the
correct one.

### Disposition of Pass 1 findings — 18/18 fixed

| # | Disposition | Evidence I rechecked |
|---|---|---|
| **B-1** Holm printed not applied | **Fixed, and more completely than asked** | Two-pass `compare_report`; `holm_steps` implements the step-down stop; `verdict(alpha_step=, holm_tested=)` is wired. Re-rendered my Pass 1 contradiction case (p=0.008 / 0.031, k=2): the family table now reads `distinguishable` / `not distinguishable — below the observable floor`, so no reader can reach the opposite conclusion from a bare threshold. Mutations M5, M6 now killed; 4 further Holm mutations (stop removed, `holm_tested` ignored, `_decision` collapsed, threshold constant) all killed. The added `decision` column is the right call — fixing the decision alone would have left the complaint half-open. |
| **M-1** blank `packId` silently dropped | **Fixed** | Re-ran my reproduction: blanked → `invalid: [('cand','field')]`, deleted → same. The filter now drops only a record that *says* it belongs elsewhere (`results.py:445`). |
| **M-2** `designEffect`/`basis` defaults | **Fixed** | Both required on the dataclass; legacy fallback moved to `from_dict` only. |
| **M-3** basis/DEFF propagation untested | **Fixed** | M1 and M2 both killed, by three named tests including `test_the_design_effect_is_the_max_of_the_two_arms`. Residual: P2-1. |
| **M-4** contracts parametrized over themselves | **Fixed, generally** | `EXPECTED_MODEL_SCHEMA_1` / `EXPECTED_DETERMINISTIC_SCHEMA_1` are hand-written literals **by name and by tier**; I checked their independence two ways — they are plain dict literals not derived from `modelbench`, and their 30 model names match plan §3.4.2's own "26 + 4 = 30" enumeration. Dropping a field (M18) fails; *relaxing a tier* `nonempty`→`present` (M18b, a mutation neither of us had tried) fails too; shrinking the forbidden set (M8) fails. The arm-kind and schema-key sets are pinned as well, so the class is closed rather than the two instances. |
| **M-5** `scoreable` intersection untested | **Fixed** | M4 killed; and the fix went past the finding — `PairedRows` now tallies `asymmetry`/`only_in_*`/`unscoreable_both` and prints them beside every verdict, so a shrunken paired *n* is visible rather than merely honest. Swapping `asymmetry_a`/`asymmetry_b` and mis-attributing `only_in_a` are both killed. |
| **M-6** false no-verdict reason + silent `--models` drop | **Fixed, both halves** | `_NO_VERDICT_REASON` keyed by cause; re-ran one-arm and zero-arm renders and got the "fewer than two arms were selected" text. CLI: `--models cand,incumbnet` now prints a named reason and returns **2**; the correct pair still returns 0. M3 killed. |
| **m-1** cross-pack quarantine leak | **Fixed** | Another pack's schema-99 record no longer appears in this pack's exclusion block; still correctly *off* `unparseable`, as I recommended. |
| **m-2** hash divergence mislabelled | **Fixed** | Now `unpaired (same pack version, different content hash)`. |
| **m-3** `_unit_ids` dead, docstring false | **Fixed** | `_paired_rows` calls it (`report.py:106`); the module docstring names it correctly. |
| **m-4** untested `--role` / p95 | **Fixed** | M27 and M28 both killed, by two new named tests. |
| **m-5** vestigial `contentHash` | **Fixed** | Now `str \| None`, `None` at S1; see the seam note below. Residual doc drift: P2-5. |
| **m-6** pack directory name vs `packId` | **Fixed** | `cli.py:134` passes `pack.packId`. |
| **m-7** raw `FileNotFoundError` on an unslugged `runId` | **Fixed** | Now a named `ValueError` citing §3.5's slug rule; verified with `qwen/qwen3-4b-2507`. Residual: P2-4. |
| **n-1** tautological assertion | **Fixed** | No `or True` remains anywhere in `tests/`. |
| **n-2** mutable `fields`, name-only `__hash__` | **Fixed** | `MappingProxyType(dict(...))` in `__post_init__`; `__hash__` over a `json.dumps(sort_keys=True)` canonical form. Both mutations killed; the `default=repr` choice is right — `hash(tuple(sorted(items)))` would raise on the list-valued fields. |
| **n-3** fabricated `ClassificationAggregates` | **Fixed** | `d["aggregates"]` and `d["kind"]`; the `KeyError` surfaces as `unparseable`. |
| **n-4** "unwritten scripts" for every unit kind | **Fixed** | Renders "unwritten items" / "unwritten queries" via the existing `_SAMPLE_NOUN`. |
| **OQ-2** `_percentile` definition | **Correctly left alone** | Opened as plan v1.5 §6 R-13 for `data-scientist`; the new index test pins p50 ≠ p95 without pinning a definition, which is exactly the right shape for an open methodological question. |

### The new public API, judged as a seam (S2 wires against it)

- **`holm_thresholds` → `holm_steps` returning `HolmStep(p, rank, threshold, tested, rejected)`** —
  right shape, and a strict improvement: the old `list[float]` could not express the step-down at
  all, which is why B-1 was possible. `rejected` and `tested` are separate rather than one tri-state,
  which is what lets `report.py` print a threshold for a member past the stop (as §3.3 requires)
  while saying it means nothing. No caller can now get the threshold without the stop. One caveat:
  P2-3.
- **`verdict(..., holm_tested: bool = True)`** — defaulting to `True` is the right default for a
  k=1 family and keeps every existing call site valid. `Verdict` gained `floor_demoted` and
  `holm_tested`, both of which `report.py` reads; no new caller can produce a "distinguishable"
  without passing through Rule 7, because the check is inside `verdict()` on every path rather than
  in the report.
- **`BinaryMetric.unit`, required with no default** — the right call, and the docstring gives the
  right reason (the value a forgetful caller wants is the one that licenses the interval). It does
  make every S2 scorer state its denominator unit, which is the point. `_metric_from_dict` refusing
  a `.get` fallback means a pre-`unit` stored record now fails to load; there are none in the repo
  (`results/runs/` does not exist yet), so this is free **now** and would not have been one stage
  later — worth noting in the S2 brief rather than fixing.
- **`RunResult.designEffect` / `.basis`, now required** — as recommended; `from_dict` keeps the
  legacy path.
- **`PackRef.contentHash: str | None`** — **the code side is right and Appendix A should follow.**
  `""` is indistinguishable from "a hash was computed and came back empty" in the one field whose
  entire job is identity. Appendix A's `(packId, packVersion, contentHash)` triple describes a
  *loaded* pack, which is S2's concern; `PackRef` at S1 is the reference handed to a report, and it
  has no hash to carry because the AC-3 banner reads each run's own recorded
  `fingerprint.packContentHash`. Suggested plan wording for the queued sweep: `contentHash` is
  `str | None`, `None` meaning "not loaded", and a `PackRef` returned by S2's `load_pack` never has
  it `None` — which keeps the triple total exactly where the triple is claimed.

### One observation, routed to `data-scientist` rather than filed as a finding

Rule 7 compares `|diff|` against `resolving.observable_floor`, which is computed at the
family-adjusted `α/k`, while Holm tests a later-ranked member at its own looser step. The two
interact: in my re-render of the Pass 1 case, `falseSuspendRate` (p=0.031) *cleared* its Holm step of
0.05 and was then demoted by the α/k floor of 17.5 pp. The rendering is coherent — the decision
column and the verdict text both name the floor as the reason — so nothing in my lane is wrong. But
the consequence is that for k ≥ 2 at these n, Holm's step-down buys nothing on a binary metric that
the α/k floor does not take back, which may or may not be the intended reading of `-ml` §3.3 + Rule
7. That is the α/k-versus-Holm-step question the brief already routed to you; this is a concrete
reproducible instance of it.

### Appendix B — Pass 2 evidence

**24 mutations, 3 survivors** (all against a scratch copy; the repo working tree was never
modified). Survivors: `mcnemar_may_decide` widened to admit `"measured"` (P2-1, 296 passed);
`load_history`'s `isinstance(schema, bool)` guard removed (P2-2, 296 passed); `store()`'s
`{"", ".", ".."}` clause removed (P2-4, 296 passed).

Killed, by area — **Holm/Rule 7 (6):** step-down stop removed · `holm_tested` ignored ·
`threshold = alpha` for all · `alpha_step` ignored · floor check removed · floor check `<` → `<=`.
**Statistics presentation (5):** `sqrt(DEFF)` inflation dropped · floor truncation → round ·
truncation bin-edge guard dropped · `UnattainablePower` guard removed · unattainable clause
bypassed. **Report (6):** pooled count given a Wilson interval · pooled footnote suppressed ·
pairing tally not printed · `only_in_a`/`only_in_b` conflated · `asymmetry_a`/`asymmetry_b` swapped ·
`_decision` collapsed. **Basis/DEFF (4):** `basis = a.basis` · `design_effect = 1.0` ·
`_BASIS_STRENGTH` order flipped · `min` → `max`. **Fingerprint/results/CLI (5):** forbidden set
shrunk · required field dropped · **tier relaxed `nonempty` → `present`** · `__hash__` value-blind ·
mapping copy removed · `--models` filter removed · `--role` filter removed · index p95 → p50 ·
`paired_cluster_bootstrap` DEFF guard removed.

**P2-2 reproduction:**

```
validate() on benchSchemaVersion=True -> []          # accepted
store() accepted: boolschema.json                    # written
load_history -> valid: []  invalid: [('boolschema', 'unknown_schema', True)]
```

## Pass 3 — 2026-09-03

### Scope & verdict

**Re-gated:** commit `95b4c88` (`fix(model-bench): S1 second gate round — floor at the unadjusted
alpha, McNemar as a veto`), the whole diff against `3ad27d3` — 10 files, +827/−157. Baseline:
Pass 2's five findings (P2-1…P2-5), `docs/plans/small-model-benchmarking.md` **v1.6** §4 S1 and §5,
and the rendered output of the built harness. This pass was run by a **fresh reviewer**; nothing
from Pass 2's session carried over except the two written passes above.

**Deferred to `data-scientist`, as briefed** — every formula, constant, tolerance and verdict
string is `docs/plans/small-model-benchmarking-ml.md`'s. I judged them only where the *code* prints
a claim it does not compute. Two observations are routed to that pass at the end of this section.
Note that `-ml` is **being revised concurrently**: my baseline is the committed v1.6 that `95b4c88`
was written against, not the uncommitted working-tree edit.

**CPG: considered, not relevant** — no CPG graph exists for `model-bench` (no `cpg_model-bench`;
the dispatch brief confirms it). Every finding below comes from reading the files, **rendering
reports and reading them as a user**, running the suite, and **86 source mutations of my own**.

**Verdict: needs changes.** — **1 blocker, 6 majors, 5 minors, 3 nits**, all new. All five Pass 2
findings are fixed, verified individually; P2-4's correction is accepted and recorded below.

**What I ran, from `model-bench/`:**

| command | result |
|---|---|
| `.venv/bin/python -m pytest -q` | `314 passed in 3.67s`, exit 0 |
| `.venv/bin/python -m pytest -q -m "" -rsx` | `314 passed` — nothing deselected, nothing skipped or xfailed |
| `.venv/bin/python -m pytest --collect-only -q -m ""` | `314 tests collected` — collected count equals run count |
| `.venv/bin/ruff check .` | `All checks passed!`, exit 0 |
| `python -m modelbench compare / models / index rebuild` end-to-end on a temp root | exit 0, artifacts as documented |
| **86 mutations** in six batches | **71 killed, 15 survived** (2 equivalent, 1 near-equivalent, **12 genuine gaps**) |

### New findings

**P3-1 (blocker) — an arm carrying *no data at all* for a metric is scored as failing every item,
and the §4.3 tally whose job is to make that visible reports all zeros.**

`report.py:122-134`: `a_scoreable = item.scoreable.get(metric, True)` admits an item that never
mentions the metric, and `a_ok.append(item.counts.get(metric, 0) > 0)` then scores it **`False`** —
a loss. Absence becomes a finding. Rendered (Appendix C.1): an arm whose 10 items carry `counts={}`
and `scoreable={}` produces

> `cand is better than incumbent on falseAdvanceRate: +100.0 pp (95% CI [60.8, 100.0] pp) … p=0.002`

with `paired n: 10 of 10 items (asymmetry: 0 … 0 unscoreable in both …)`. The function's own
docstring says a precondition failure "must never be laundered into the numerator" (`-ml` §10 R2,
rated **high**); here missing data is laundered into the *denominator's complement* instead.
**Both defaults are untested** — mutating either survives all 314 tests (C11, C12). S2's scorers are
what will emit these mappings, so this is also the S2 seam most likely to produce a confident wrong
number. *Fix:* treat `metric not in item.counts` as not-scoreable for that arm, so it routes through
the existing `asymmetry_*` / `unscoreable_both` counters and the shrunken paired *n* is visible; add
one test per default.

**P3-2 (major) — the "not distinguishable" verdict asserts the observed difference is below the
MDD without checking, and the assertion is false on ordinary data.**

`stats.py:612-615` ends every non-`None` MDD clause with `…; the observed {X} pp is below that.` —
unconditional. `-ml` §3.2e verdict 2 publishes the sentence in an example where it *is* below; the
code applies it whenever the else-branch is reached. Reproduced on the real guard-judge shape
(n=85, k=2): the report prints *"resolves differences of >=10.5 pp … the observed 10.6 pp is below
that"*. Also at n=40, k=1: 20.0 pp against a 19.1 pp MDD (Appendix C.2). Swept exhaustively: **614
of 5 525** else-branch tables at n=40, and **8 378 of 35 644** at n=85, print the false clause. No
test asserts this clause at all (`grep "is below that" tests/` → nothing). It matters because the
sentence misattributes the cause: the difference was large enough, the *discordance split* was not
— which is precisely what the "best case" caveat exists to say. *Fix:* branch on
`abs(diff) < resolving.mdd80` and print the other statement when it is not; the replacement wording
is `data-scientist`'s to settle (§3.2e), the missing comparison is not.

**P3-3 (major) — on the default fail-safe path the decision sentence tells the reader clustering
was declared and the interval widened for it, when neither happened.**

`stats.py:857-863` appends, unconditionally on the cluster path: *"widened by
sqrt(DEFF)={x:.2f} **for the declared clustering**"*. At `design_effect == 1.0, basis == "assumed"`
— which `AGENTS.md` states is the path **every** comparison carries until S2's determinism probe
lands — this renders `sqrt(DEFF)=1.00`, i.e. nothing was widened, and no clustering was declared.
The real reason McNemar was displaced is the unverified `basis`, and the sentence never names it
(the provenance line prints `assumed` separately, three lines away). The only test of this string
(`test_stats.py:589`) pins the `DEFF=1.41` case, where the wording is true. *Fix:* make the clause
conditional — name the design effect when `> 1.0`, and name the **basis** when it is 1.0, e.g.
"…because this comparison's design effect is `assumed` rather than established by construction".

**P3-4 (major) — `--negative-control` writes a durable report indistinguishable from a real
comparison, and the code comment claims otherwise.**

`cli.py:114-116`'s comment reads "the mode's own docstring **and the report** say why this cannot
fail (`-ml` §9)". The report says nothing: I ran it end to end and `grep -ic negative` on the
produced `reports/guard-judge-understanding-20260903-02.md` returns **0**. What a reader gets is an
ordinary-looking verdict (`b=0, c=0`, "not distinguishable") with both arms bearing the same label
and a tally reading "0 present in cand only, 0 in cand only", filed next to the real comparison
under a filename that differs only in its sequence number. `-ml` §9 and plan §3.9(5) are explicit
that the real negative control is **two independent runs, not two copies**, and that two copies
"cannot fail" — so a stored artifact that reads as a validated null is the one output this tool's
value claim cannot afford. *Fix:* pass the mode into `compare_report` and render a banner naming it
a wiring smoke check with `b = c = 0` by construction; or, cheaper, prefix the markdown in
`_cmd_compare`. Either way, correct the comment.

**P3-5 (major) — the bootstrap seed is a magic literal in `report.py`, duplicating a manifest field
`PackRef` does not carry, and it is never printed.**

`report.py:424` passes `bootstrap_seed=20260902`. Plan §3.3's manifest declares
`sampling.seed: 20260902` — the same number, in the pack, where it belongs — and `PackRef` has no
field for it, so the pack's declaration cannot reach the decision. `-ml` §3.2d requires the seed to
go into the fingerprint "so a report is reproducible"; at S1 it is in neither the fingerprint nor
the rendered report, so a reader handed a bootstrap-decided verdict cannot reproduce the interval.
Today every comparison takes this path (P3-3). *Fix:* add `seed: int` to `PackRef`, read it in
`pack_ref_from_manifest`, pass it through, and print it beside the `decided by:` line; leave the
fingerprint half to S2 with a note in the S2 brief.

**P3-6 (major) — the α-attribution that M-ML-6 established is untested wherever the two αs
differ, and the test that looks like it pins it does not.**

Two mutations survive all 314 tests: `provenance()` printing `alpha_family` instead of `alpha_mdd`
(A4), and `unattainable_clause()` quoting `b_min(alpha_family)` instead of `b_min(alpha_mdd)` (A8).
Both are k>1-only divergences, and the suite's only k=2 assertion on an α is
`test_report.py:372`'s `assert "alpha=0.025" in md` — which is satisfied by the **family-wise
paragraph** (`"computed at the family-adjusted alpha=0.025"`), not by the MDD sentence it is
placed to guard. So the report could print `resolves differences of >=10.5 pp … alpha=0.05` while
computing at 0.025 and nothing would fail. A8 additionally produces a self-contradicting sentence
(`b_min=6 … at that alpha`, where the named alpha's `b_min` is 7). *Fix:* in the k=2 test, assert
the **full provenance parenthetical** verbatim (`design effect 1.00, by-construction, alpha=0.025`)
rather than a bare substring; add one `unattainable_clause` test at k=2 with `n_eff` in `[6, 7)`.

**P3-7 (major) — the exploratory-label test cannot fail; inverting the filter it guards is green.**

`report.py:483-488` selects `m.name not in family`. Inverting it to `in family` — which labels the
**pre-registered verdict metrics** "exploratory — no significance claim" and hides the genuinely
exploratory ones — leaves **314 passed** (E17). The test
(`test_a_metric_outside_the_verdict_family_is_labelled_exploratory`, `test_report.py:393`) asserts
only `"sideMetric" in md` (also true from the Arms table) and `"exploratory — no significance
claim" in md` (true of whichever metric got listed). It asserts the presence of two strings in a
document, not the pairing between them — the presentation layer against itself. Plan §5 test 11b and
§3.3 both make this a requirement-level claim. *Fix:* assert the rendered line whole —
``"- `sideMetric` — exploratory — no significance claim" in md`` — and add the negative:
``f"- `{METRIC}` — exploratory" not in md``.

**P3-8 (minor) — `compare --session` is entirely untested.** Deleting the filter
(`cli.py:98`) leaves 314 passed (E16); `grep -n session tests/test_cli.py` returns nothing. It is
one of `compare`'s four options and the one FR-16's same-session pairing rests on. Pass 1's m-4
closed the same gap for `--role`; `--session` was missed. *Fix:* one test storing two runs under
different `sessionId`s and asserting the filtered arm set.

**P3-9 (minor) — `index.csv`'s `valid` column is untested.** Hardcoding
`_index_row(run, valid=True)` (`results.py:550`) survives (F14), so a regression that marks every
stored record usable would not be caught. The index is the only place an operator sees which of a
history's runs are usable at a glance. *Fix:* extend the existing index test with one invalid
record and assert its `valid` cell is `no`.

**P3-10 (minor) — the absent-vs-null distinction the fingerprint module is built around is not
enforced at its own seam.** `Fingerprint.from_dict` defaults `armKind` to `""` (`fingerprint.py:201`),
which `validate()` reports as `absent`; changing the default to `None` — reported as `null` —
survives (F8). The module docstring makes "absent is not empty, and `null` is neither" its first
principle, and §3.4.2's three states are tested for the *fields* but not for the discriminator.
*Fix:* one test asserting a record with no `armKind` key yields `FieldProblem("armKind", "absent")`
and one with `"armKind": null` yields `"null"`.

**P3-11 (minor) — two design-effect guards are untested, and one of them fails ugly.**
Removing `verdict()`'s `design_effect < 1.0` precondition (`stats.py:694`) survives (F6) — that is
Rule 4's precondition 4, and `-ml` §9 check 2(c) names the other three, all of which *are* tested
(D10/D11/D13 killed). Removing `resolving_power`'s `design_effect <= 0` check (`stats.py:489`)
also survives (F5); with it gone a `design_effect` of 0 raises a bare `ZeroDivisionError` instead of
the named error, at the one seam S2's runner supplies. *Fix:* one `pytest.raises(ValueError)` each.

**P3-12 (minor) — `compare_report`'s headline-membership guard is untested, and its failure mode is
a bare `StopIteration`.** Deleting the guard (`report.py:287-290`) survives (C13); the only test of
the rule goes through `metrics_from_manifest`, which a `PackRef` built in code — as S2 and every
fixture here do — bypasses. Without the guard, `next(v for m, v, _ in computed if m == …)`
(`report.py:470`) raises `StopIteration` with no message. *Fix:* one test constructing a `PackRef`
directly with an out-of-family headline and asserting `PackConfigError`.

**P3-13 (nit) — a second literal `0.05`, in the module that declares there is only one.**
`holm_steps(p_values, *, alpha: float = 0.05)` (`stats.py:893`) sits 830 lines below
`ALPHA_FAMILY`'s docstring — *"One home, because … a second literal `0.05` is how they drift
apart"*. `report.py:408` always passes explicitly, so nothing is wrong today. Plan §4 S1 prints the
same `= 0.05` in its signature block, so the fix is a matched pair: `alpha: float = ALPHA_FAMILY`
here, and the same in the plan.

**P3-14 (nit) — two small untruths in names.** `report.py:116` initialises `only_in_b = 0` and then
recomputes it unconditionally at line 136 — a dead assignment that reads as a running counter.
`cli.py:151` calls `_report_path(root, args.pack)` while the parameter is named `pack_id` and the
docstring says `<pack-id>`; that is the other half of Pass 1's m-6, which fixed the `load_history`
call and left the filename on the directory name.

**P3-15 (nit) — two defensive guards nothing distinguishes.** Removing `wilson_interval`'s
`max(0.0, …)/min(1.0, …)` clamps survives (E9); the effect is real but tiny — the unclamped bound
is `1.0000000000000002` at `s=n=16` and `-6.9e-18` at `s=0`, which renders as `1.000` and `-0.000`
in the Arms table. Removing `pack_ref_from_manifest`'s `"analysisUnit" not in sampling` check also
survives (F10), degrading a named `PackConfigError` to a `KeyError` the CLI happens to catch.
One assertion each would close both.

### Disposition of Pass 2's findings — 5/5 fixed, one with a correction to *my* premise

| # | Disposition | Evidence I rechecked |
|---|---|---|
| **P2-1** `"measured"` at DEFF 1.0 untested | **Fixed** | Re-ran the Pass 2 survivor: widening `mcnemar_may_decide` to `basis in ("by-construction", "measured")` now **fails** (B6, `1 failed, 155 passed`). Both mirror tests exist. |
| **P2-2** `validate()`/`load_history` disagree on a bool schema | **Fixed at both points** | Live repro: `validate()` → `[FieldProblem('benchSchemaVersion','unknown')]`; `store()` **refuses**; a hand-written file lands as `('boolschema','unknown_schema', None)` — the bool no longer reaches the `int \| None` field. Removing either guard, or the `None`-narrowing, is killed (C1, C2, C3). |
| **P2-3** short Holm ladder could drop a metric | **Fixed, and the equivalence claim is sound** | `strict=True` is genuinely tested (B1 killed), and a ladder that returns short is killed by `holm_steps` itself (B2). I re-derived the implementer's *equivalent-by-construction* survivor independently: 20 000 random families (k=1..6, including ties, duplicates and `NaN` p-values) give **zero** differences between the dict form and the old placeholder+`None`-filter form. The claim holds; `by_index` is still the better shape because it turns a missing rank into a `KeyError` here. |
| **P2-4** runId guard partly unreachable | **Fixed — and my Pass 2 premise was wrong** | I verified the asymmetry myself: `Path(".").name == ""` so `"."` **is** already caught by `runId != Path(runId).name`, but **`Path("..").name == ".."`**, so `".."` is **not**, and dropping it would write `results/runs/..json`. Pass 2 said "two-thirds … unreachable"; the correct figure is **one-third**, and the guard's present `{"", ".."}` is right. All three ids are now tested (C4, C5, C6 each killed). |
| **P2-5** `contentHash` docstring said "empty" | **Fixed** | `packs.py:79-82` and the `pack_ref_from_manifest` docstring both say `None`. |

### What's solid

The α-routing this round exists for is right where it counts and pinned where it can be:
computing the floor at `alpha_mdd`, the MDD at `alpha_family`, swapping the pair at the report's
call site, collapsing `alpha_mdd` to `alpha_family`, running Holm at `alpha_mdd`, and naming the
wrong α in `floor_clause` are **all killed** (A1, A2, A5, A6, A7, A3). So is the whole B-ML-2/m-ML-6
surface — the veto as a conjunction, the veto dropped, the disjunction, the `Rule7Violation` raise,
the `alpha_step` range premise, and the floor compared against the printed value rather than the
exact one (B3–B8). The floor printer's direction, its bin-edge guard and its `x/precision`-vs-`x*1000`
form are all pinned (B9, B10). Rule 4's other three preconditions, the CI orientation flip, the
Holm step-down, the pooled-count interval refusal, the basis/DEFF propagation and the pairing tally
are all killed. **71 of 86 mutations died**, and the twelve gaps above are the residue.

### Two observations routed to `data-scientist`, not filed as findings

1. **The floor-demotion paragraph reads as a contradiction.** Rendered at `DEFF=2.0`, `(34,6,0,0)`:
   *"differences below 30.0 pp cannot reach significance at any observed outcome, at any Holm step
   (alpha <= 0.05) … in conjunction with McNemar's exact test (**p=0.031**) as a necessary
   condition"*. A reader sees a p below 0.05 in the same paragraph as a sentence saying no observed
   outcome can reach significance. The resolution — McNemar's p is computed over raw units and is
   invalid under clustering, so the floor over `n_eff` governs — is nowhere in the rendered text.
   Whether the floor sentence should carry a "per effective unit" qualifier on the clustered path is
   yours.
2. **P3-2's replacement wording.** The missing comparison is an engineering defect and I have filed
   it as one; the sentence that should print when the observed difference *exceeds* the MDD is
   §3.2e's to specify.

### Carried a third time, and it is `architect`'s

**Plan §5's numbered test list is still not stage-scoped.** Items 1–6, 11b and part of 12 are
S1-reachable; 7, 7b, 8, 9, 10, 11, 12b and 13–19 are S2+. Nothing in §5 says so, so all three gates
of this component have had to re-derive the split from §4's stage blocks — and a gate that derives
its own checklist is a gate that can derive it differently next time. *Fix (one pass, no content
change):* tag each numbered item with the stage that owes it — `1. **(S1)** \`fingerprint.validate()\`
— …` — or add a two-column item→stage table under §5's preamble. §4's stage blocks already carry
the information; §5 just never states it.

### Appendix C — Pass 3 evidence

**C.1 — P3-1 reproduction.** Arm A: 10 items with `counts={M: 1}`, `scoreable={M: True}`. Arm B:
the same 10 `pairingKey`s with `counts={}`, `scoreable={}` — no data for the metric whatsoever.

```
paired rows kept: 10 | unscoreable_both: 0 | asymmetry_a/b: 0 0
b_ok (arm with NO metric key at all): [False] * 10
### falseAdvanceRate
cand is better than incumbent on falseAdvanceRate: +100.0 pp (95% CI [60.8, 100.0] pp),
n=10 paired items (unit: item, design effect 1.00), McNemar exact p=0.002 (b=10, c=0).
- paired n: 10 of 10 items (`asymmetry`: 0 … 0 unscoreable in both; 0 present in … only) — §4.3
```

**C.2 — P3-2 sweep.** Exhaustive over every `(a,b,c,d)` with `a+b+c+d = n`, counting tables that
reach `verdict()`'s final `else` branch and whose `|diff|` exceeds `mdd80`:

| configuration | `mdd80` | else-branch tables | printing the false clause |
|---|---|---|---|
| n=40, k=1 (α_mdd 0.05) | 19.1 pp | 5 525 | **614** |
| n=85, k=2 (α_mdd 0.025) — the guard-judge pack | 10.5 pp | 35 644 | **8 378** |
| n=12, k=1 — the tool-caller pack | 57.8 pp | 311 | 0 |

Two rendered instances: `n=85, (60,21,12,22)` → *"…>=10.5 pp … the observed 10.6 pp is below
that"*; `n=40, (11,13,5,11)` → *"…>=19.1 pp … the observed 20.0 pp is below that"*.

**C.3 — 86 mutations, six batches, 15 survivors.**

| batch | area | run | killed | survivors |
|---|---|---|---|---|
| A | α routing and attribution | 8 | 6 | A4, A8 (→ P3-6) |
| B | Holm, Rule 7, the veto, the floor printer | 10 | 10 | — |
| C | fingerprint / results / report guards | 14 | 11 | C11, C12 (→ P3-1), C13 (→ P3-12) |
| D | stats core, preconditions, resampling | 16 | 16 | — |
| E | verdict prose, CLI, index, banners | 20 | 16 | E8, E9 (→ P3-15), E16 (→ P3-8), E17 (→ P3-7) |
| F | constructors, manifests, derived artifacts | 18 | 12 | F5, F6 (→ P3-11), F8 (→ P3-10), F10 (→ P3-15), F11, F14 (→ P3-9) |

**Two survivors are equivalent mutants, stated so they are not re-chased.** *E8* — `b_min`'s loop
condition `>` → `>=` — is indistinguishable at every α the code can produce: `mcnemar_exact(b,0)`
is `2^(1-b)`, and `0.05/k` for k=1…10 is never exactly one of those values (it differs only at a
dyadic α such as 0.03125). *F11* — `analysisUnitIndex` hardcoded to `0` — is equivalent because
`check_sampling_contract` forces `analysisUnit == pairingKey[0]`; the *behaviour* is still pinned
(D1, using `pairingKey[-1]`, is killed).

**C.4 — P2-4's asymmetry, re-derived.** `runId != Path(runId).name` catches `"."` (name `""`) and
`"a/b"`; it does **not** catch `""`, `".."` or `"..."`. So of Pass 2's three-id set only `"."` was
unreachable. (`"..."` is also uncaught but is a legal filename and harmless.)

**C.5 — a hazard for the coordinator, not for the code.** This session's scratchpad is **shared
with the parallel `data-scientist` gate**: my mutation driver was overwritten mid-pass by that
session's file of the same name, which silently ran *its* mutation list under my invocation. Every
survivor reported above was therefore **re-verified in a fresh sandbox built from
`git archive 95b4c88`**, isolated by `PYTHONPATH`, whose baseline is `314 passed`; all nine
re-checks reproduced. `git diff --stat -- model-bench` is empty at the end of this pass — the
working tree was never modified by me, and the other session's edits to
`docs/plans/small-model-benchmarking-ml.md` and `falkor-chat/server/**` are untouched.
