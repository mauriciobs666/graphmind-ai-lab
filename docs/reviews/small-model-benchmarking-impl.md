# Small-LLM benchmarking tool (`model-bench/`) — S1 implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/small-model-benchmarking.md` §4 S1

**Pass 1** gated `ab91419` (needs changes). **Pass 2** re-gated `3ad27d3` (approve with
suggestions). **Pass 3** re-gated `95b4c88` (needs changes). **Pass 4** re-gated `d55f4d8` (needs
changes). **Pass 5** gated `8fc2341`, the first of S1e's three implementation units (needs changes).
**Pass 6** re-gated its fix round `c523a35` (needs changes). **Pass 7** re-gates `f409905` — jump to
[`## Pass 7`](#pass-7--2026-09-07) for the current verdict; the earlier passes are kept intact
because they are meant to be read together. Passes 1–4 gate the S1 build; Passes 5–7 are S1e's first
unit (§4 S1e Tables A and B, the `fingerprint.py` re-key), whose remaining two units are not yet
delivered. **Pass 7 §3 says which half of this document to trust** — the findings held, three
suggested fixes did not.

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

---

## Pass 4 — 2026-09-03

### Scope & verdict

**Re-gated:** commit `d55f4d8` (`fix(model-bench): S1 Pass 3 gate round — absence is not an
outcome`), reviewed as the **current state of `model-bench/`** at that commit rather than as a
diff. Baseline: Pass 3's fifteen findings, plan **v1.7** §4 S1 / §5, and the rendered output of the
built harness. Scope is S1 — `modelbench/{stats,report,results,packs,cli}.py` and their tests; S2's
absent modules are out of scope and are not reported as findings. Run by a **fresh reviewer**;
nothing carried over from Passes 1–3 except the three written passes.

**Deferred to `data-scientist`, as briefed** — every formula, constant, tolerance and verdict string
is `docs/plans/small-model-benchmarking-ml.md` v1.7's. One item is routed there below; I judged the
rest only where the *code* prints a claim it does not compute.

**CPG: considered, not relevant** — no CPG graph exists for `model-bench` (the dispatch brief
confirms it, and `cpg_model-bench` is not loaded). Every finding below comes from reading the files,
**rendering reports and reading the English**, running the suite, and **128 source mutations of my
own**, all in a sandbox built from `git archive d55f4d8` outside the repository.

**Verdict: approve with suggestions.** — **0 blockers, 5 majors, 5 minors, 3 nits**, all new. All
fifteen Pass 3 findings are fixed, each verified individually. **The stakeholder judgement is in
[§ Is this converging?](#is-this-converging) — the residue does not block S2.**

**What I ran, from `model-bench/`:**

| command | result |
|---|---|
| `.venv/bin/python -m pytest -q` | `353 passed in 4.52s`, exit 0 |
| `.venv/bin/python -m pytest -q -m "" -rsx` | `353 passed` — nothing deselected, skipped or xfailed |
| `.venv/bin/python -m pytest --collect-only -q -m ""` | `353 tests collected` — collected equals run |
| `.venv/bin/ruff check .` | `All checks passed!`, exit 0 |
| `compare` / `compare --negative-control` end-to-end on temp roots | see P4-1, P4-5 |
| exhaustive `verdict()` branch sweep, 8 `(n, k)` configurations | see the `data-scientist` routing |
| **128 mutations**, four batches | **109 killed, 19 survived** (4 equivalent, **15 genuine gaps**) |

### New findings

**P4-1 (major) — `--negative-control` writes a durable report whose banner is false, and
self-contradicting, when no arms were selected.**

`report.py:331-332` emits `_NEGATIVE_CONTROL_BANNER` before `_comparison_pair` is consulted. With no
stored runs for the pack, `cli.py:113`'s `if negative_control and candidates` is false and
`_select_arms` returns `[]` — so the report opens with

> **NEGATIVE CONTROL (WIRING SMOKE CHECK)** — both arms are the *same stored record*, so
> `b = c = 0 by construction` and this comparison **cannot fail**.

and then, ten lines below in the same document, *"None: fewer than two arms were selected, so there
is nothing to compare."* Verified end-to-end through the CLI: **exit 0**, report written to
`reports/<packId>-<date>-01.md`. This is P3-4's own failure mode — a durable artifact asserting
something untrue of itself — re-entered through the case P3-4's fix did not cover. Mutation Z03
(`if negative_control and len(runs) >= 2`) survives all 353 tests, so nothing pins either behaviour.
*Fix:* gate the banner on `len(runs) >= 2`, or emit a distinct one-line refusal naming the mode and
the empty history; add the test. **On the brief's question — the "cannot fail" claim is otherwise
true of the code:** with two copies of one record `b = c = 0` ⇒ `diff = 0`, so neither `p <= alpha`
nor a zero-excluding CI is reachable on either path and `distinguishable` cannot be returned. The
defect is the zero-arm case, not the claim.

**P4-2 (major) — the report's Holm wiring has no regression test, and breaking it reproduces the
Pass 1 blocker with a green suite.**

`report.py:483` passes `holm_tested=step.tested`. Hardcoding it to `True` leaves **353 passed**
(mutation W16). Rendered under that mutation, on a k=2 family with both p-values at 0.039 (b=8,
c=1, n=40), the report prints

> `cand is better than incumbent on mB: +17.5 pp (95% CI [2.9, 32.6] pp) … McNemar exact p=0.039`

three paragraphs above a family table whose own row for `mB` reads `not tested (Holm stops here)` —
a significance claim for a metric Holm never tested, contradicted inside the same document. That is
verbatim Pass 1's blocker (the ladder printed, the correction not applied). `verdict()`'s *internal*
use of `holm_tested` is pinned (Z28 killed) and the sibling wiring `alpha_step=step.threshold` is
pinned (W15 killed); only this one keyword is unguarded. *Fix:* one test on the two-metric family
above asserting `"is better than" not in md` for the stopped member and the `not tested` row beside
it. Full rendered pair in Appendix D.1.

**P4-3 (major) — the family-wise paragraph's α attribution is unpinned; swapping the two αs inside
it is green.**

`report.py:521-527` prints *"Every **MDD** above is computed at the family-adjusted
alpha={alpha_mdd}; every **observable floor** is computed at the unadjusted alpha={alpha_family}"*.
Exchanging the two expressions leaves **353 passed** (W08). The rendered result labels 0.05 as
family-adjusted and 0.025 as unadjusted, and contradicts the provenance parenthetical three
paragraphs above, which still prints `alpha=0.025` for the MDD. This is **P3-6 one section over**:
that finding pinned the provenance parenthetical and the floor's own α, and left the paragraph whose
whole job is to explain the pair. Third pass running in which an α-attribution string is found
unpinned. *Fix:* assert the paragraph's two clauses verbatim in the existing k=2 report test.

**P4-4 (major) — P3-1's deliberate deferral: judged, and it does not hold. The Arms table and the
paired-rows refusal print mutually exclusive statements about the same metric in the same report.**

Rendered (Appendix D.2), an arm whose aggregates declare `BinaryMetric(successes=0, n=10)` for a
metric no item declares scoreable produces, in one document:

> `| cand | falseAdvanceRate | 0/10 | 0.000 | [0.000, 0.278] |`   (`report.py:387-390`)
>
> **No verdict: no paired data.** … An arm carrying no data for a metric is not an arm that failed
> every item of it. (`report.py:229-233`)

The round's three reasons, taken in turn. (i) *"the arm misreports, the reporter does not infer"* —
true, and not exculpatory: the report is the instrument, and an instrument that renders a
self-inconsistent input without noticing is one you cannot stand behind. (ii) *"the table is
labelled descriptive"* — `_DESCRIPTIVE_NOTE` (`report.py:32-35`) caveats the **interval**
(*"Per-arm intervals are Wilson score intervals…"*); it says nothing about the rate, and `0.000`
over a denominator of 10 is a claim that ten items were scored. (iii) *"the cross-check is S2's
because S2's scorer produces both"* — the *contract* is S2's, but the *check* is S1-local and needs
nothing S2 provides: at `compare_report` time both `run.items` and `run.aggregates` are in hand, and
for each verdict-family `BinaryMetric` whose `unit` is the analysis unit, `metric.n` must equal the
count of items with `scored_outcome(metric) is not None`. `IncompleteItemRecord` is the precedent —
refuse an internally inconsistent record rather than render it. *Fix:* that one comparison, raising
(or banner-ing) on mismatch, plus a test. **It must land before S3**, the first stage that produces
real scored data: a net added after the thing it protects has shipped is how all four of these
passes began.

**P4-5 (major) — `IncompleteItemRecord` takes down the whole comparison with an uncaught traceback,
outside the documented exit-code set, naming neither the record nor its path.**

`results.py:147` raises; `report.py:127` is the only caller; `cli.py:146-152` catches only
`PackConfigError`. Verified end-to-end: one item with `scoreable={"m": True}, counts={}` in **one**
of two otherwise-valid stored records aborts `compare` with an uncaught `IncompleteItemRecord`, exit
1 — not one of `cli.py`'s closed set (`0/2/3/4/5`) — **no report written at all**, and the valid arm
lost with it. `store()` accepts the record (its validation is fingerprint-only) and `load_history`
accepts it, so the refusal lands at the furthest possible point from its producer, and AC-2's actual
mechanism — *excluded on read **and named*** in the `INVALID RESULTS EXCLUDED` block — is bypassed.
On the brief's two questions: **the refusal cannot be bypassed** on the paired intersection (X01,
X02, X03 all killed; a truthy non-bool `scoreable` value still demands a count), though it never
fires for an item present in only one arm, which is dropped before `scored_outcome` is called; and
the message names the **item and metric** but not the run, so a scorer author holding forty stored
records gets `item 'i9' …` and a grep. *Fix:* validate items in `load_history` and route a failure to
`InvalidRecord(reason="field")` so the record is excluded and named; or minimally catch it in
`_cmd_compare`, return `EXIT_FINGERPRINT`, and name `run.runId` and `path`.

**P4-6 (minor) — an aggregate `BinaryMetric` with `n == 0` vanishes from the Arms table, and the
guard that drops it is also the only thing preventing a `ZeroDivisionError`.** `report.py:376`'s
`and metric.n` is doing two jobs and is tested in neither: relaxing it to `metric.n >= 0` leaves
**353 passed** (Y08), which means no test constructs a zero-denominator aggregate anywhere. Rendered,
a two-arm comparison in which one arm declares the metric with `n=0` prints a **one-row** "Arms"
table, with nothing saying the second arm is missing. *Fix:* one test asserting the row is rendered
as `0/0 — no observations` (or explicitly asserting the drop, if the drop is the decision), and one
asserting no exception.

**P4-7 (minor) — the §4.3 tally's arm-B-only half is entirely untested.** `only_in_b = 0`
(`report.py:141`, mutation X06) and `considered = len(a_keys)` (`report.py:146`, Z04) both survive
all 353 tests; the printed labels and the other four counters are pinned (W01, W02, W03, Z06 all
killed). Asymmetric coverage caused by the *second* arm is exactly what §4.3 rule 2's tally exists to
surface. *Fix:* one fixture where arm B carries two pairing keys arm A does not, asserting both the
denominator and the `in {b_label} only` cell.

**P4-8 (minor) — the marginal-overlap diagnostic is unpinned in both directions.** Inverting the
printed `yes`/`no` (`report.py:495`, Z20) and computing both margins from the same arm
(`stats.py:770-771`, Z21) each leave 353 passed. `tests/test_report.py:180` asserts only that the
label substring is present, and `tests/test_stats.py:657` is the suite's only assertion on the value
— `is True`. No test anywhere asserts it is ever `False`. It is a declared diagnostic and `-ml` §3.1
says it is inert in this regime, which caps the severity; it is still a printed FR-15 output that can
say the opposite of the truth with a green suite. *Fix:* one `is False` case in `stats`, and assert
the rendered line whole in `report`.

**P4-9 (minor) — the manifest-`packId`-not-directory-name fix is half-pinned.** `cli.py:135`'s
`load_history(root, packId=pack.packId)` survives being re-pointed at `args.pack` (Y47), while the
`_report_path` half P3-14 closed is pinned (Y45 killed). The two lines are the two halves of Pass 1's
m-6, and they carry consecutive comments saying so. *Fix:* extend the P3-14 test's fixture so the
pack **directory** name differs from the manifest's `packId` and assert the arms still load.

**P4-10 (minor) — the report cannot distinguish two arms of the same model, which is the exact shape
of plan §5 test 19a.** Rendered on two independent runs of one `modelKey` in different sessions
(Appendix D.3), the Arms table has two identical `arm` cells, the §4.3 tally reads *"0 scoreable for
qwen/qwen3-4b-2507 only, 0 scoreable for qwen/qwen3-4b-2507 only"*, and a significant verdict would
read *"X is better than X"*. `runId` and `sessionId` are never printed. §5 test 19a — *"the highest-
value single test in the harness"* — is precisely two independent runs of the same model, so the
report is unreadable for the one comparison the value claim rests on. *Fix:* in `_arm_label`, when
two arms share a `modelKey`, append the distinguishing `sessionId` (or `runId`).

**P4-11 (nit) — `mover_d_interval`'s three clamps are untested, the unclosed twin of P3-15.**
Replacing `max(0.0, lo_rad)` with `abs(lo_rad)` (`stats.py:133`, W11) and dropping the `[-1, 1]`
clamps (`stats.py:136`, W12) both survive. Both are reachable — swept over every `(a,b,c,d)` with
`n <= 60`: 6 tables have a negative radicand (magnitude ~3e-17) and 14 produce a bound outside
`[-1, 1]` (magnitude ~2e-16) — and both are sub-display, exactly P3-15's finding. The docstring
calls them *"required, not cosmetic"*, which is now the only claim of its kind with no test.

**P4-12 (nit) — `holm_tested`'s effect on `floor_demoted` is untested.** Dropping it from
`stats.py:840`'s `fired` conjunction survives (Z29). It is unreachable on the `mcnemar-exact` path
(Rule 7 is a theorem there), but on the cluster path it would relabel a Holm-untested member's
decision cell as *"below the observable floor"*. One assertion.

**P4-13 (nit) — three untested renderer details.** The `_NO_PAIRED_DATA_TALLY` pointer sentence
(`report.py:468`, W06), the exploratory-metric dedup that stops both arms listing the same metric
twice (`report.py:572-574`, Z11), and which stored record `--negative-control` duplicates
(`cli.py:117` — `candidates[-1]` survives, X13; nothing documents that it is the first).

### Routed to `data-scientist`, not filed as a finding

**The `|diff| == mdd80` boundary prints "is above that" at exact equality.** `stats.py:664`'s
comparison is strict, so equality takes the alternate branch — **the right branch**, since the MDD
claim is `>=` — but `stats.py:666` then renders *"the observed 10.0 pp is above that"* when it is
exactly equal. I confirmed equality is reachable **through `verdict()`'s else branch**, not merely
through `_mdd_clause` in isolation: sweeping every `(a,b,c,d)` at eight `(n, k)` configurations,
**1 300** tables print the alternate clause at exact equality, the first being n=90, k=2,
`(a=0, b=6, c=15, d=69)` — `diff = -10.0` pp against `mdd80 = 0.1` exactly. The sentence is §3.2e's,
published verbatim, so the wording is yours. **One second-order check came back clean:** the clause's
closing *"this comparison is not strictly dominant (b=…, c=…)"* is never printed with `b == 0` or
`c == 0` in any of those sweeps, so it never contradicts the counts beside it.

### Disposition of Pass 3's findings — 15/15 fixed

| # | Disposition | Evidence I rechecked |
|---|---|---|
| **P3-1** absence scored as a loss | **Fixed** | `scored_outcome` is the sole decider. Restoring either old default is killed (X01, X02); `counts[m] > 0` → `>= 0` killed (X03); the four tally counters killed (X04, X05, W02, W03); the empty-intersection refusal and its Holm `—` cell killed (X07, X08, X09). **Residue judged separately: P4-4, P4-5.** |
| **P3-2** unconditional "is below that" | **Fixed** | Conditional at `stats.py:664`. Unconditional, inverted and `<=` all killed (X19, X18, X17). |
| **P3-3** false widening clause | **Fixed** | `>= 1.0`, branch swap, and a constant `assumed` all killed (X20, X21, X22). |
| **P3-4** `--negative-control` unlabelled | **Fixed for the two-arm case**; the zero-arm case is new (**P4-1**). Banner suppressed and "may fail" both killed (X10, X11). |
| **P3-5** bootstrap seed a literal | **Fixed** | `PackRef.seed`, no default. The literal restored in either place is killed (X14, X15), as are the manifest check (Y36) and printing the seed off the bootstrap path (X16). |
| **P3-6** α attribution untested | **Fixed at the two named seams**; the family-wise paragraph is not (**P4-3**). `provenance` and `unattainable_clause` naming the wrong α are killed (X23, X24), as is `floor_clause`'s (X25). |
| **P3-7** exploratory label test could not fail | **Fixed** | Inverting the filter now fails (Y01). |
| **P3-8** `--session` untested | **Fixed** | Deleting the filter fails (Y03). |
| **P3-9** `index.csv` `valid` untested | **Fixed** | Hardcoding it True fails, both ways (Y04, Z34). |
| **P3-10** `armKind` absent-vs-null | **Fixed** | `d.get("armKind", None)` fails (Z02). |
| **P3-11** two design-effect guards | **Fixed** | Both fail: removing `verdict()`'s precondition (Y20) and restoring `resolving_power`'s old `<= 0` bound (X30). |
| **P3-12** headline-membership guard | **Fixed** | Removing it fails (Y02). |
| **P3-13** second literal `0.05` | **Fixed by removal, and the reading of the plan is correct.** Plan v1.7 §4 S1 line 1414 shows `def holm_steps(p_values, *, alpha: float)` with no default, and the comment below it reads *"none of these parameters gets a literal default here"*. Grepped: every other `0.05` in `modelbench/` is inside a comment or docstring; `ALPHA_FAMILY` is the only literal. |
| **P3-14** two untruths in names | **Fixed** | Dead assignment gone; `_report_path(root, pack.packId)` pinned (Y45). The sibling `load_history` call is not — **P4-9**. |
| **P3-15** two undistinguished guards | **Fixed for the two named** | `wilson_interval`'s clamps and `pack_ref_from_manifest`'s `analysisUnit` check both fail now (Z01, Y37). The same class in `mover_d_interval` is untouched — **P4-11**. |

**m-ML-8's surviving mutant is genuinely equivalent, and the claim that matters is pinned.** The
survivor is the restoration of an *identical* duplicate of the MDD stem in `report.py`; identical
text renders identically, so no test can distinguish it — equivalent by definition rather than by
argument. What the finding was actually about is the **drift**, and that is killed: editing the stem
in `stats.py` while a stale duplicate remains fails the suite (X26).

### What's solid

**The computational core is not where the residue is.** Every mutation I aimed at a number died:
both αs and their routing (X23–X25, Y18, Y19, Y39), the floor's truncation and its bin-edge guard
(X27, X28), the MDD's ceiling and its unfloored/floored denominator asymmetry (Y29, Y30), Rule 7 on
both paths including the compare-against-the-printed-value trap (Y26, Y27), the McNemar veto and
`mcnemar_may_decide`'s two conjuncts (Y24, Y25), Holm's ordering, threshold formula and step-down
stop (X29, Z32, Y34), the design-effect/basis fail-safe propagation and its two legacy `from_dict`
defaults (Y05, Y06 in batch Y, Z36, Z37), the paired table's b/c classification and unit-diff sign (Z22, Z23),
the winner-first CI flip (Y28), every AC-2/AC-3 banner (Y13–Y16, Z38), the
`_require_effective` type detector (Y31) and the duplicate-analysis-unit backstop (Y33). **109 of
128 mutations died**, and of the nineteen survivors four are equivalent and the fifteen above are
the residue — none of them a wrong number.

**Four survivors are equivalent mutants, stated so they are not re-chased.** `_unit_ids`'
`analysisUnitIndex` → `0` (Pass 3's F11 again: `check_sampling_contract` forces it, and the
behaviour is still pinned — `index = -1` is killed, Y12). `b_min`'s loop starting at `b = 2` — it
can only matter if `mcnemar_exact(1, 0) = 1.0 <= alpha`, i.e. `alpha >= 1.0`. `holm_steps`'
`p <= threshold` → `<`: `mcnemar_exact` returns a dyadic rational and `0.05/k` is not dyadic —
swept every `(b, c)` with `b + c <= 60` against `0.05`, `0.025` and `0.05/3`, **zero** exact
equalities. `mcnemar_exact(c, b)`: the function is symmetric in its two arguments by construction
(`m = b + c`, `k = min(b, c)`).

<a id="is-this-converging"></a>
### Is this converging? — the stakeholder judgement, explicitly

**Yes, and the residue does not block building S2 on top of S1.**

The count is flat (Pass 3: 1 blocker + 6 majors; Pass 4: 0 blockers + 5 majors) but the *character*
has changed, and that is the signal. Pass 3's blocker rendered `+100.0 pp, p=0.002` against an arm
holding no data, and four of its majors were sentences that were simply false. **Nothing in Pass 4
is a wrong number.** Two of my five majors are report-surface defects reachable only through an
operator mistake (P4-1) or a scorer that does not exist yet (P4-4); the other three are **missing
nets over code that is correct** (P4-2, P4-3, P4-5).

The reason Pass 3 found more than Pass 2 is worth stating plainly for the stakeholder: Pass 3 was
the first pass to render reports and read the English, and to run a mutation campaign at scale. The
*method* changed, not the code's quality. Pass 4 applied that same method plus 128 mutations to a
larger surface and found no blocker and no false printed clause on any path I could reach. That is
what convergence looks like from the outside.

**What I would gate on, and where.** None of P4-1…P4-13 changes a signature S2 wires against, and
none produces a wrong number today, so **S2 (adapter, host info, runner) can be dispatched now**.
Two of them should gate **S3** — the first stage that produces real scored data — rather than S2:

- **P4-4**, the aggregates-versus-items cross-check, because it is the net that catches the first
  real scorer's first mistake, and it is S1-local and cheap.
- **P4-2**, the Holm-wiring regression test, because it is the only thing standing between this
  build and a verbatim recurrence of the Pass 1 blocker.

P4-1, P4-3 and P4-5 are a half-day of work and can ride alongside S2. Everything from P4-6 down is a
follow-up.

**One process observation, and it is `architect`'s, not the implementer's.** Plan §5's
stage-attribution table landed at v1.7 and closes Pass 3's thrice-carried item — I read this pass's
scope off it directly and did not have to re-derive the S1/S2 split. It is the first gate of this
component that did not.

### Open questions

1. **P4-4's severity depends on a scope call I cannot make.** If S2's scorers will be built to
   compute `aggregates` *from* `items` in one pass, the inconsistency is unrepresentable and the
   cross-check is belt-and-braces. If they are two independent code paths — which is what
   `ClassificationAggregates` being a stored field rather than a derived property implies — the
   check is load-bearing. `architect` owns which shape S2 takes.
2. **P4-10 may be `architect`'s rather than a bug.** Disambiguating two same-model arms could equally
   be solved by the runner refusing to compare two runs of one model outside the negative-control
   mode. §5 test 19a requires exactly that comparison, so it cannot simply be refused — but where the
   label comes from is a design call.

### Appendix D — Pass 4 evidence

**D.1 — P4-2 reproduction.** k=2 family, both metrics at `(b=8, c=1)` in n=40, so both p-values are
0.039: rank 0 is tested at 0.025 and fails, Holm stops, rank 1 is `tested=False` with a printed
threshold of 0.05. Unmutated, `mB` renders *"Not distinguishable at this sample size. Not tested:
Holm–Bonferroni stops at the first non-rejection…"*. With `holm_tested=True` hardcoded at
`report.py:483` — **353 passed** — the same fixture renders:

```
### mB
cand is better than incumbent on mB: +17.5 pp (95% CI [2.9, 32.6] pp), n=40 paired items
(unit: item, design effect 1.00), McNemar exact p=0.039 (b=8, c=1).
…
| mB | 0.039 | 0.0500 | not tested (Holm stops here) |
```

**D.2 — P4-4 reproduction.** Two arms, ten items each, every item `scoreable={m: False}`; each arm's
stored aggregate declares `BinaryMetric(m, successes=0, n=10, unit="item")`:

```
| qwen/qwen3-4b-2507 | falseAdvanceRate | 0/10 | 0.000 | [0.000, 0.278] |
…
**No verdict: no paired data.** No item is scoreable for `falseAdvanceRate` in both arms … An arm
carrying no data for a metric is not an arm that failed every item of it (`-ml` §4.3).
- paired n: 0 of 10 items (… 10 unscoreable in both …) — §4.3
```

**D.3 — P4-10 reproduction.** Two runs of `qwen/qwen3-4b-2507`, `sessionId` `s1` and `s2`, 40 items
each. `Comparison kind: **paired, cross-session**` is correct; every arm-identifying string in the
report below it is the same token twice, and neither `runId` nor `sessionId` appears anywhere.

**D.4 — 128 mutations, four batches, 19 survivors.**

| batch | area | run | killed | survivors |
|---|---|---|---|---|
| X | the P3-1/P3-2/P3-3/P3-5 fix surface, the negative-control banner, the MDD boundary | 30 | 28 | X06 (→ P4-7), X13 (→ P4-13) |
| Y | the other Pass 3 fixes, `verdict()`'s preconditions, banners, packs, `load_history`, CLI | 46 | 43 | Y08 (→ P4-6), Y11 *(equivalent)*, Y47 (→ P4-9) |
| Z | untouched surface: tally, `_decision`, resolving-power line, `stats` core, index, exit codes | 38 | 30 | Z03 (→ P4-1), Z04 (→ P4-7), Z11 (→ P4-13), Z20/Z21 (→ P4-8), Z25 *(equivalent)*, Z29 (→ P4-12), Z33 *(equivalent)* |
| W | tally strings, family-wise prose, MOVER-D, the Holm wiring seam | 14 | 8 | W06 (→ P4-13), W08 (→ P4-3), W11/W12 (→ P4-11), W13 *(equivalent)*, W16 (→ P4-2) |

Three further mutations in batches Z and W turned out to be syntactic no-ops on inspection and are
excluded from the 128 rather than counted as survivors.

**D.5 — isolation.** Every mutation ran in `/tmp/gate-p4-eng-mbx/`, built from
`git archive d55f4d8 model-bench | tar -x`, against a pristine copy restored between mutations, with
`PYTHONPATH` pointed at the sandbox (verified: `modelbench.__file__` resolves inside it). Every
scratch file carries the `_mbx` suffix. `git status --porcelain -- model-bench` is empty at the end
of this pass; nothing outside `docs/reviews/small-model-benchmarking-impl.md` was written in the
repository.

## Pass 5 — 2026-09-07

### 1. Scope & verdict

**Reviewed:** commit `8fc2341` (`feat(model-bench): S1e Tables A and B — the fingerprint re-key`),
the whole diff against `bc63765` — 6 files, +640/−76, all under `model-bench/`. This is the **first
of three** S1e implementation units; two follow (Tables C+D+E+G on `stats.py`, Table F on
`results.py`/`report.py`).

**Baseline:** `docs/plans/small-model-benchmarking.md` **v1.15** §4 S1e preamble, **Table A**,
**Table B**, §3.4.1, §3.4.2, §3.4.4a, §4 S1 done-conditions **DC-1**, **DC-6**, **DC-12**, §7 rule 5
and Appendix A; `docs/plans/small-model-benchmarking-ml.md` v1.18 where a table cites it.

**Not reviewed:** the plan itself — this is a code review, not a tenth plan gate. Passes 1–9 of
`docs/reviews/small-model-benchmarking.md` were read for *why* the design is as it is, not re-opened.
`stats.py`, `results.py`, `report.py` and `test_report.py` are untouched by this commit and were read
only where this diff's behaviour reaches them.

**CPG:** considered, not relevant — no CPG exists for `model-bench/` and none was loaded during this
pass; every finding below comes from reading the tree, running the suite, and four constructed
counter-implementations in an isolated sandbox.

**Verdict: needs changes.** — 0 blockers, 4 majors, 2 minors, 1 nit.

The re-key itself is correct and well pinned; every mechanism Tables A and B exist to establish
holds under direct attack (eight constructed counter-implementations, six killed). The four majors
are all on the **proof** side rather than the behaviour side, and two of them are the same shape:
a decision the implementer got *right* and defended in a comment, with nothing in the suite holding
it there — `to_dict`'s deterministic omission (**P5-2**) and the refusal of the retired residency
size key (**P5-3**) each survive a counter-implementation that reverses them. **P5-1** is a genuine
behavioural defect, small in blast radius: the new discriminator collapses three states into one
where the old one keeps them apart, and the test comment asserts the opposite of what the code does.
**P5-4** is the test written to catch a half-applied edit, which does not catch it.

**None of the four is blocked on unbuilt work**, so none may ride as a follow-up. All four are
closable now, and **P5-3's other half is a plan edit** (`architect`), not a code change.

**This does not have to block the `stats.py` unit.** The fix round for P5-1/P5-2/P5-4 touches
`modelbench/fingerprint.py` and `tests/test_fingerprint.py` only — files neither remaining S1e unit
opens — so it can run before, after, or beside Tables C+D+E+G without a file collision. What must
not happen is S1e's round closing on DC-12 with these open. See §6 for the one correction the
`stats.py` unit's brief needs (**P5-6**).

### 2. What I re-ran myself

| Check | Command / method | Observed |
|---|---|---|
| Suite | `.venv/bin/python -m pytest -q`, cwd `model-bench/` | **472 passed in 5.33s** |
| Table A residual 1 | `grep -rFc lmsCliCommit modelbench tests --include='*.py'` | **0** on all 16 files |
| Table A residual 2 | `grep -rFc sizeBytes modelbench tests --include='*.py'` | **0** on all 16 files |
| Table B residual 1 | `grep -rFc FORBIDDEN_BY_ARM_KIND modelbench tests --include='*.py'` | **0** on all 16 files |
| Table B residual 2 | `grep -rFn 'frozenset(FORBIDDEN' modelbench --include='*.py'` | **no match** |
| Table B residual 3 | `grep -rFn 'REQUIRED_BY_SCHEMA[1]["model"]' modelbench tests --include='*.py'` | **no match** |
| Table B residual 4 | `grep -rFn 'set(REQUIRED_BY_SCHEMA[1]) == {"model",' modelbench tests --include='*.py'` | **no match** |
| Tree state | `git diff --stat 8fc2341 -- model-bench/` | empty, before and after every mutation |

Mutation isolation: `/tmp/.../scratchpad/mbx/`, built from `git archive 8fc2341 model-bench | tar -x`,
run with `PYTHONPATH` pointed at the sandbox (verified: `modelbench.__file__` resolves inside it),
sandbox baseline **472 passed**. Each mutated file was copied aside and **restored by copy**
immediately after its run. No `git restore`, no tree-mutating git command, nothing written in the
repository outside this review document.

### 3. Adjudication of the implementer's six reported items

**(1) The DC-1 / Table-A-residual tension — real, and *not* fully dischargeable in code.**
*Confirmed real.* DC-1 (plan `docs/plans/small-model-benchmarking.md` §4 S1, line 2649-2656) states
the behaviour by naming both retired keys: *"an element carrying `modelKey` or `sizeBytes` … is
**invalid**"*. Table A's second residual is `grep -rFc sizeBytes modelbench tests --include='*.py'`
→ **0**, unscoped across production **and** tests, and `-F` matches comments. A test written the way
DC-1's sentence invites puts that residual at 1 and fails DC-12 on a faithful implementation. That
is §7 rule 5(b)'s own trap shape, inside the plan that wrote the rule. The plan is half-aware of it
— DC-12's own note (line 2907-2916) reasons that `modelKey` can have no residual and that DC-1's
assertion is rule 5(b)'s named alternative — but it draws that conclusion for `modelKey` only and
never notices that the *other* key's residual then forbids the assertion from naming it.

*The resolution is the best available in code, and it is not sufficient.* Making the rule key-set
exact (`fingerprint.py`, `_residency_problems`, the `sorted(set(element) - RESIDENCY_ELEMENT_KEYS)`
loop) does refuse both retired keys by construction, and the suite asserts the rule's generality
through three stand-in keys (`test_fingerprint.py:199-220`) and a `bytesOnDisk` half-swap
(`:223-235`). But **no test constrains the `sizeBytes` half**, and none can while the residual
stands: see finding **P5-3**, where a counter-implementation carrying a one-name tolerance list
passes all 472 tests. So the tension is discharged *in design* and left open *in proof*, and closing
it needs a plan-side edit. See **P5-3**.

**(2) The element-shape check applied to both residency snapshots — correct, not scope creep.**
Plan §3.4.4a (line 1112-1114) states the `{id, state}` shape over **`residentModelsAtStart` /
`residentModelsAtEnd`** jointly — "each surviving entry recorded as `{id, state}` with the literal
`state` string kept" — and §3.4.2 gives both fields the same `present` tier. Table A's row names only
`residentModelsAtEnd` because that is the *fixture* site carrying the retired element, not because
the rule is one-sided; the row's own text says the fix is "DC-1's assertion, not this row". Narrowing
the check to the named field would leave the identical silent mismatch open on the other snapshot —
the very failure mode §3.4.2 line 819-825 describes. The choice is also **pinned**: mutation M3
(`_RESIDENCY_FIELDS` narrowed to `{"residentModelsAtEnd"}`) → **3 failed, 469 passed**.

**(3) The `ProblemReason` choice and the `field` path grammar — the grammar is not invented; the
reason widening is right but leaves the plan stale.**

*The path grammar is precedent, not invention.* `modelbench/results.py:466` already ships
`FieldProblem(field=f"items[{item.itemId}].counts.{metric}", reason="absent")` — the same
`<field>[<index>].<key>` grammar, predating this diff. The only consumer is `report.py:534`, which
renders `p.field` as opaque text (`f"\`{p.field}\` ({p.reason})"`) and parses nothing. Appendix A
types the member `field: str` and imposes no grammar. So the implementer's choice is *consistent with
the codebase*, not a new convention, and it under-claimed by calling it invented.

*The reasons are each used in their established sense, with one legitimate widening that the plan
has not caught up with.* `absent`/`null`/`empty`/`forbidden` on element keys are exact analogues of
their field-level meanings one level down. `unknown` is the widening: Appendix A (line 5225) defines
it as *"a **discriminator** this build cannot interpret — an unrecognised `armKind`, or a
`benchSchemaVersion` from the future"*, and the code now also returns it for a non-list snapshot, a
non-mapping element, and a non-string key value. Those are type errors, not discriminators — but
none of the other four fits, so the closed five force `unknown`, and the module docstring was
correctly updated to say so. What is left open is Appendix A, which now describes a narrower
`unknown` than the code implements. That is a plan sweep, not a code change — see **P5-5**.

**(4) `callSurface` on a deterministic record ⇒ `FieldProblem("callSurface", "forbidden")` — the
right reading, and the plan states it twice.** §4 S1's signature block (line 2245) reads
`callSurface: Literal["chat", "embeddings"] | None   # v1.9 (§3.4.1, §3.4.4a); None iff
deterministic`, and Appendix A's `Fingerprint` row (line 5226) repeats it: *"`callSurface` is `None`
iff `armKind == "deterministic"`"*. "iff" is binding in both directions, so a deterministic record
carrying a surface is claiming a call that arm never makes — the same species of unmeasurable claim
`{"modelKey": "bm25"}` is, which is the module's whole thesis. `forbidden` is also the right token:
§3.4.1 says a discriminator "is incapable of appearing in a **derived** forbidden set", and the code
honours that — the reason is returned directly from the discriminator branch, never via
`FORBIDDEN_BY_ARM_PROFILE`. Pinned at `test_fingerprint.py:181-185`.

**(5) `to_dict` omitting `callSurface` on a deterministic arm — reasoning correct, round-trip claim
verified, decision unpinned.** The reasoning is the plan's: §3.4.2 reserves `null` for "we did not
capture this", and a deterministic arm's absence of a surface is a different fact. **The round-trip
claim is true — I executed it rather than reading it:** `to_dict()` on a deterministic fingerprint
carries no `callSurface` key; `from_dict(to_dict()) == fp` holds and `.validate() == []` for all
three profiles (`model:chat`, `model:embeddings`, `deterministic`). But **nothing constrains the
decision**: mutation M2, which writes `"callSurface": self.callSurface` unconditionally — i.e.
`null` on a deterministic arm, the exact shape the comment rejects — passes all **472** tests. See
**P5-2**.

**(6) The `validate()` `KeyError` left unfixed — correct to leave, and genuinely unreachable, but
for a reason worth writing down.** I confirmed unreachability by execution rather than by reading:
with `ARM_KINDS == {deterministic, model}` and `CALL_SURFACES == {chat, embeddings}`, the set of
profiles the two guards admit is exactly `{deterministic, model:chat, model:embeddings}` — the key
set of `REQUIRED_BY_SCHEMA[1]` — so `reachable − profiles == ∅`. The residual exposure is the
pre-existing one: a *schema 2* whose profile set differs from schema 1's, since `ARM_KINDS`,
`CALL_SURFACES` and `FORBIDDEN_BY_ARM_PROFILE` are all derived from `REQUIRED_BY_SCHEMA[**1**]`
while the required-set lookup is `REQUIRED_BY_SCHEMA[**schema**]`. That shape shipped before this
change (`FORBIDDEN_BY_ARM_KIND` was schema-1-only too) and no schema 2 exists. Leaving it is right.

One thing the implementer did not claim and that carries the argument: after the re-key the guard is
no longer an exact key-set membership test but **two independent projections** of the key set, and
what keeps the projections honest is that all four sets are pinned **by value** in the suite —
`REQUIRED_BY_SCHEMA[1]`'s keys (`test_fingerprint.py:443`), `FORBIDDEN_BY_ARM_PROFILE`'s (`:482`),
`ARM_KINDS` (`:455`) and `CALL_SURFACES` (`:456`). Adding a schema-1 profile therefore fails loudly
at `:443` rather than reaching a `KeyError`. That is sufficient, and it is why I raise no finding
here — but it is an invariant held by four separate literals a reader must combine, so if the
`stats.py` or Table F unit is ever tempted to derive one of those literals from another, that is the
line to refuse.

### 4. Findings

#### P5-1 — `callSurface` collapses *absent*, *empty* and *null* into one reason, and the suite's own comment claims it does not — major

`modelbench/fingerprint.py`, `from_dict` (`callSurface=d.get("callSurface")`) and `validate()`'s
`elif not self.callSurface: return [FieldProblem(field="callSurface", reason="absent")]`. Executed,
not read:

| stored `callSurface` | `armKind` (for comparison) | `callSurface` |
|---|---|---|
| key missing | `absent` | `absent` |
| explicit `null` | **`null`** | **`absent`** |
| `""` | `absent` | `absent` |

The module's first principle is *"absent is not empty, and `null` is neither"*, and review **P3-10**
made exactly this distinction load-bearing for the *first* discriminator — `test_fingerprint.py:522`
exists because defaulting a missing `armKind` to `None` instead of `""` "survived all 314 tests".
The second discriminator, added by this diff, does not have it. Worse, the suite asserts the
collapse while describing it as the opposite: `test_fingerprint.py:168-171` reads *"it is **absent**,
not empty or null: the same three-state discipline §3.4.2 applies to the fields applies to the
discriminator"* immediately above an assertion that `callSurface=""` reports `absent` — under that
discipline a blank `nonempty` value reports `empty`. `:542-547` compounds it, calling a missing key
"one written before the profile existed **or by something that lost it**", which is precisely the
two states `armKind`'s own test keeps apart.

The harm is a misdiagnosis on the AC-2 surface: `report.py:534` renders the quarantine line from
`p.reason`, so a stored record carrying `"callSurface": null` — written by something that had the
value and lost it — is reported to the operator as never written.

**Suggested improvement**, either of two, but not the present state:
(a) implement the distinction — give `from_dict` a missing-key sentinel distinct from a literal
`null` (`d["callSurface"] if "callSurface" in d else _ABSENT`), branch `is None → "null"`,
`not value → "absent"`, and relax the deterministic branch to a truthiness test so the
omit/round-trip identity in **P5-2** still holds; or
(b) if the distinction is judged not worth carrying for a discriminator whose only writer is
`to_dict`, **say so** — replace the two comments above with the honest statement that `callSurface`
reports `absent` for all three, and why that differs from `armKind`. What cannot stand is a comment
asserting a discipline the code beneath it does not implement.

#### P5-2 — `to_dict`'s deterministic-arm omission is a decision no test constrains — major

`fingerprint.py`, `to_dict`: `surface = {} if self.callSurface is None else {...}`. The reasoning is
right (adjudication 5 above) and the round-trip holds. But **mutation M2** — replacing the whole
expression with an unconditional `"callSurface": self.callSurface`, i.e. writing `null` on a
deterministic arm, the exact shape the comment rejects — is **472 passed**. The round-trip test
(`test_fingerprint.py:315-329`) cannot see it, because `from_dict` maps a missing key and a literal
`null` to the same `None`: **P5-1's collapse is what makes P5-2 invisible.** The decision is
therefore held in place by a comment alone, in a stored-record shape that S2's runner and S3's
`load_history` both read, and this is the codebase's recurring defect class — correct for the current
caller, silently revertible by the next.

**Suggested improvement:** one assertion beside the round-trip test —
`assert "callSurface" not in Fingerprint(armKind="deterministic", callSurface=None,
fields=deterministic_fields()).to_dict()`, plus its positive twin on a model arm. Fixing P5-1(a)
would kill M2 as a side effect; this assertion kills it either way and is worth having regardless.

#### P5-3 — DC-1's `sizeBytes` half is asserted nowhere, and no test can assert it while Table A's second residual stands — major

**Mutation M1** — `RESIDENCY_ELEMENT_KEYS` left alone, but the extra-key loop changed to
`sorted(set(element) - RESIDENCY_ELEMENT_KEYS - frozenset({"sizeBytes"}))`, i.e. a residency element
carrying the retired size key is accepted — is **472 passed**. DC-1 names that key explicitly as a
case that must be **invalid**; the shipped behaviour is correct, and nothing in the suite holds it
there.

This is not primarily an implementer error — it is the trap adjudicated in item (1). Every
code-side discharge I can construct either spells `sizeBytes` in a `.py` file (residual → 1, DC-12
fails on a correct implementation) or asserts something weaker. One discharge does exist and I flag
it as an option rather than a recommendation: the residual is scoped `--include='*.py'`, so a
`tests/data/*.json` fixture carrying the retired element would pin DC-1 exactly and leave the
residual at 0 — arguably more faithful, since the retired shape is a *stored record* shape, but it
also reads as routing around a check, and that call is not mine to make.

**Suggested improvement — plan-side, owner `architect`, and it is the one item here that cannot be
closed in `model-bench/` code.** Reword DC-1 so the behaviour is stated without the token: *"an
element whose key set is anything other than `{id, state}` — including either key of the retired
`lms ps --json` element — is invalid"*. That preserves DC-1's meaning, keeps Table A's residual
sound (it is the right residual: the sole pre-edit occurrence is the `conftest.py` fixture, so
scoping it to `modelbench/` would make it zero *before* the edit and prove nothing), and lets the
suite pin the rule by value. DC-12's note at plan line 2907-2916 should gain the same correction: it
reasons the `modelKey` half correctly and never notices that the surviving residual forbids the
assertion from naming the other half.

#### P5-4 — the half-swap test is named for a failure it does not detect — major

`test_fingerprint.py:223-235`, `test_a_half_swapped_residency_element_is_invalid`, asserts
`_problems(...) != []`. Both its cases swap one key of the retired pair and keep the other, so the
*missing* key of `{id, state}` alone already produces a problem — the assertion is satisfied without
the extra-key rule existing at all. **Mutation M5** — the entire extra-key `forbidden` loop deleted,
so an element may carry any key whatsoever — fails **exactly the 6 parameters of
`test_a_residency_element_carrying_any_other_key_is_refused`** and **both** ids of the half-swap test
**pass**. The test whose docstring says *"a fixture edit that stopped halfway fails here rather than
shipping green"* is the one test in the group that would not have caught it.

**Suggested improvement:** assert the problem, not its non-emptiness —
`assert FieldProblem(field="residentModelsAtEnd[0].bytesOnDisk", reason="forbidden") in problems`
for the size case and the matching `…[0].modelKey` for the identity case, keeping the two ids.
That also makes this the test that carries P5-3's design half honestly.

#### P5-5 — Appendix A's `unknown` is now narrower than the code — minor

Plan Appendix A line 5225 defines `unknown` as *"a **discriminator** this build cannot interpret — an
unrecognised `armKind`, or a `benchSchemaVersion` from the future"*. `fingerprint.py` now also
returns it for a non-list residency snapshot, a non-mapping element, and a non-string element value
— type errors, correctly assigned (the other four reasons all mean something else) but outside the
row's stated scope. The module docstring was swept; the plan was not, and it cannot be by this unit.

**Suggested improvement — owner `architect`, one line:** widen Appendix A's `FieldProblem` row from
"a discriminator this build cannot interpret" to "a **value** this build cannot interpret", naming
the residency-element type cases beside the two it already names. Precedent: v1.5 swept Appendix A
for exactly this reason when `unknown` was introduced.

#### P5-6 — this diff shifted `tests/test_results.py` by 36 lines, so the next unit's Table C row now points at the wrong line — minor

Plan §4 S1e Table C's last site row is `tests/test_results.py:507` — the comment recording R-13 as
open and `_percentile` as having two copies. Verified: that comment is at **`:507` in `bc63765`** and
at **`:543` in `8fc2341`**, moved by this unit's two added test blocks. Table B's and Table F's
`test_results.py` pins (`:62`, `:71`, `:239`) shifted likewise. Nothing in `modelbench/results.py`
or `modelbench/stats.py` moved — those files are untouched, so every Table C/D/E/F/G pin into them
(`results.py:573`, `:599-600`, `:354-359`, `:385`, `:584`; `stats.py:159`, `:263`, `:292`, `:296`)
is still good.

This is unavoidable line drift, not a defect in the diff — but Table C's row is now false, and the
plan's own §7 rule 5 treats a *counts-at-a-named-commit* claim that does not reproduce as a finding
(it corrected two such line numbers at v1.11). **Suggested improvement:** the `stats.py` unit's brief
re-derives that row by grep (`grep -rFn _percentile tests/test_results.py`) rather than trusting
`:507`, and `architect` re-pins Table C's row at the next plan revision.

#### P5-7 — `model-bench/AGENTS.md`'s new paragraph is a live constraint and is accurate; about half of it is a third copy — nit

Verified against the bar it sets for itself: no line exceeds 700 characters and the whole file is
**1,726 words** (`awk 'length($0)>700'` → no output; `wc -w` → 1726). Verified accurate: the claim
that "every `armKind == "model"` filter in `results.py` and `report.py` is unchanged" holds — the
four surviving sites are `results.py:335`, `:637` and `report.py:387`, `:427`, all two-valued, none
touched by this diff. The `ARM_KINDS` trap is a genuine live constraint: it changes what the next
editor does, and the failure it names is silent-green-then-dead-harness.

The nit: the paragraph's first half (the three profile keys, the derivation of `armProfile`, the
union-minus-mine set operation) restates what `fingerprint.py`'s module docstring says at the code
and what the `HISTORY.md` entry says as record — the "never a third copy" smell. **Take or leave:**
trim to the two facts that bite an editor who has not opened the module — the `ARM_KINDS`
decoupling and the fact that `validate()` checks residency element shape because the `present` tier
does not — and cite `modelbench/fingerprint.py` for the rest.

### 5. The mutation audit

Eight constructed counter-implementations, all against `modelbench/fingerprint.py` in the sandbox,
each restored by copy immediately after its run. The implementer reported eleven mutations killed;
I did not re-run its eleven — I constructed my own, weighted toward the assertions I doubted rather
than toward the ones a re-run would confirm.

| # | The wrong implementation | Result |
|---|---|---|
| M1 | extra-key loop excepts one retired name (`- frozenset({"sizeBytes"})`) — a residency element carrying the retired size key is accepted | **472 passed — SURVIVED** (→ P5-3) |
| M2 | `to_dict` writes `"callSurface": self.callSurface` unconditionally, so a deterministic arm stores `null` | **472 passed — SURVIVED** (→ P5-2) |
| M3 | `_RESIDENCY_FIELDS` narrowed to `{"residentModelsAtEnd"}` — the row's literal reading | 3 failed — killed |
| M4 | the `not isinstance(element[key], str) → unknown` branch deleted | 1 failed — killed |
| M5 | the extra-key `forbidden` loop deleted entirely | 6 failed — killed, **but only by one test** (→ P5-4) |
| M6 | `_DISCRIMINATORS = ("armKind",)` — `callSurface` left in `fields` | 4 failed — killed |
| M7 | `_EMBEDDINGS_HAVE_NO` drops `maxTokens` — the derived 26-field set silently becomes 27 | 9 failed — killed |
| M8 | the `callSurface` decision moved behind the required-set mapping, so a surfaceless model record answers with 30 `absent` problems and the true one last | 2 failed — killed |

M5 is the informative kill. Its six failures are *exactly* the six parameters of
`test_a_residency_element_carrying_any_other_key_is_refused`; both ids of
`test_a_half_swapped_residency_element_is_invalid` passed, which is P5-4. So the extra-key rule rests
on a single test — adequate, but with no redundancy, and the test written to be that redundancy is
the one that does not work.

**Not reached.** I did not mutate `tests/conftest.py` or `tests/test_results.py`, and I did not
attempt a mutation of the `benchSchemaVersion` ordering (untouched by this diff and covered by
Pass 1–4). Anyone re-running this should start there.

### 6. Effect on the two remaining S1e units

**Tables C+D+E+G (`stats.py`) — nothing in this diff obstructs them, with one correction.**
`modelbench/stats.py` and `modelbench/results.py` are byte-identical to `bc63765`, so every line pin
those tables carry into production code still resolves (`stats.py:159`, `:263`, `:292`, `:296`;
`results.py:573`, `:599-600`). The single interaction is **P5-6**: Table C's `tests/test_results.py`
row must be re-derived rather than trusted. Table C's own residuals
(`grep -rFc _percentile modelbench/results.py` → 3, `modelbench/stats.py` → 3) are untouched by this
diff and still read their stated *before* values.

**Table F (`results.py`/`report.py`) — nothing contradicted, and one precedent set in its favour.**
Table F's rule is that a measure of `0.0` survives a round trip *as `0.0` and not as absent*
(DC-13(b)). This diff establishes the complementary half of the same discipline one type over: a key
that is *not applicable* is **omitted**, `null` stays reserved for *not captured*, and a present
falsy value is a real value (`to_dict`'s comment; `test_empty_list_is_valid_for_a_required_present_field`,
`test_temperature_zero_is_valid_for_a_required_present_field`). That is consistent with Table F, not
in tension with it — provided **P5-2** lands, since without it the omission half is a comment rather
than a behaviour, and Table F's implementer would be reading an unenforced precedent.

**Shared files.** All three units touch `tests/conftest.py`, and Tables C and F touch
`tests/test_results.py`, which this unit also edited — so the plan's serialisation of the three units
is doing real work and should hold. The fix round for **P5-1 / P5-2 / P5-4** touches only
`modelbench/fingerprint.py` and `tests/test_fingerprint.py`, which **neither** remaining unit opens.

### 7. What's solid

- **The re-key itself is right and is pinned by value where it matters.** `REQUIRED_BY_SCHEMA[1]`'s
  three profile keys, `FORBIDDEN_BY_ARM_PROFILE`'s three, `ARM_KINDS`'s two and `CALL_SURFACES`'s two
  are each asserted against an independently written literal, and the three required-field contracts
  are pinned by name *and tier* against hand-transcribed dictionaries — with
  `EXPECTED_EMBEDDINGS_SCHEMA_1` deliberately transcribed rather than derived from
  `EXPECTED_MODEL_SCHEMA_1`, which is exactly the M-4 discipline and is the reason M7 died.
- **The union-minus-mine derivation is a set operation, never a list**, and it earns its keep: the
  four fields it forbids on `model:embeddings` appear nowhere as a written-down list in production
  code, and `test_the_forbidden_sets_are_pinned_against_literals` catches a shrinking set loudly.
- **The `ARM_KINDS` decoupling** — the defect Table B exists to prevent — is implemented, commented
  at the constant, recorded in `HISTORY.md`, raised to `AGENTS.md` as a live constraint, and pinned
  by value at `test_fingerprint.py:455`.
- **`callSurface` is required with no default**, so the type system enumerated the construction
  sites, which is §7 rule 5's *adds rather than retires* half working as designed.
- **The per-required-field loops now run over both model profiles** (30 + 26, absent and null), where
  only one profile had them; and the discriminator ordering is real, not asserted — M8 died.
- **Scope discipline held.** `stats.py`, `results.py`, `report.py` and `test_report.py` are untouched;
  `README.md` was correctly left alone; the commit message states what it did and what it judged.

### 8. Open questions

1. **P5-1's fork is the caller's, not mine.** (a) implement the three states for `callSurface`, or
   (b) state honestly that it has one. (a) is the consistent choice and costs a sentinel plus a
   truthiness relaxation; (b) is free and is defensible for a discriminator whose only writer is
   `to_dict`. I recommend **(b) plus P5-2's assertion** — the distinction earns its keep for
   `armKind` because `from_dict` must default it, and `callSurface`'s `None` is already spoken for by
   the plan's "`None` iff deterministic". What is not available is leaving the comments as they are.
2. **P5-3's plan edit is `architect`'s and small, but it is the second time §7 rule 5(b)'s own trap
   has been found inside the document that wrote the rule** (plan-gate P6-2 was the first, on
   Table B's fourth residual). Whether that warrants anything beyond the one-line DC-1 reword — a
   sweep of the other six tables' done-conditions for the same shape — is a call for `architect` and
   the stakeholder, not for this review.

## Pass 6 — 2026-09-07

### 1. Scope & verdict

**Reviewed:** commit `c523a35` (`fix(model-bench): Pass 5 fix round — three states for callSurface,
two decisions pinned`), the diff `8fc2341..c523a35 -- model-bench/` — 4 files, +207/−26:
`modelbench/fingerprint.py`, `tests/test_fingerprint.py`, `docs/HISTORY.md`, `AGENTS.md`.

**Baseline:** Pass 5's seven findings, plus the plan sections each rests on — §3.4.1, §3.4.2, §4 S1's
signature block (line 2245), DC-1, DC-6, Appendix A's `FieldProblem` and `Fingerprint` rows.

**Not reviewed:** the plan (P5-3's plan half and P5-5 are `architect`'s and are dispositioned, not
re-argued, below). `stats.py`, `results.py`, `report.py`, `test_report.py` and `conftest.py` are
untouched by this diff.

**CPG:** considered, not relevant — no CPG exists for `model-bench/`.

**Verdict: needs changes.** — 0 blockers, 0 majors, **1 minor** (**P6-1**, new), 0 nits. Pass 5's
four majors and both plan-side items are **fixed**; P5-7 is **withdrawn**.

This is a strong fix round. Every Pass 5 finding is closed at the mechanism rather than at the
symptom, and the round corrected the gate twice — both corrections check out (§3.2), and the design
I recommended for P5-1 is one the suite now actively refuses (mutation N3, §4). The single new
finding is **P6-1**: `to_dict` keys its omission on the *value* rather than on the arm kind, so a
model record whose surface is `null` serialises to one whose surface is *absent* — the exact
information loss P5-1 was raised to prevent, one method over. Unreachable through today's writers
(`store()` validates first), reachable through `model-bench migrate`, which §3.4.3 commits to. The
fix is one line and I verified it costs nothing: **475 passed**, round trip symmetric, P5-2's pin
still green.

It is a minor, and under the standing ruling a minor is still fixed rather than carried — but **it
should not hold the `stats.py` unit.** P6-1 touches `modelbench/fingerprint.py` alone, which neither
remaining S1e unit opens.

Per the docs convention this pass is compact: one disposition line for each Pass 5 finding, and full
treatment only for what is genuinely new.

### 2. Disposition of Pass 5's findings

| # | Disposition | Evidence I rechecked |
|---|---|---|
| **P5-1** | **Fixed** — with a better mechanism than I proposed, and my framing of it was partly wrong (§3.1, §3.2) | `validate()` gains an `elif self.callSurface is None → "null"` branch ahead of the `absent` one; `from_dict` gains a per-arm missing-key sentinel. Executed: on a model record, missing key → `absent`, `""` → `absent`, stored `null` → **`null`** |
| **P5-2** | **Fixed** — pinned, not commented | `test_a_deterministic_record_omits_the_call_surface_rather_than_storing_null` asserts `"callSurface" not in reference_arm.to_dict()` plus the positive twin on both model surfaces. See §3.2 for the mutation that confirms it was needed |
| **P5-3** | **Fixed, both halves** — and the plan's fix is better than the reword I suggested | Plan **v1.16** restates DC-1 over the element's *key set* and re-scopes Table A's second residual to `modelbench` + `tests/conftest.py`, so the literal is legal in `tests/test_fingerprint.py`. Re-ran: `grep -rFn sizeBytes modelbench tests/conftest.py --include='*.py'` → **no match**; the token now appears only at `tests/test_fingerprint.py:240,251,254,269,270`, the prescribed home. `test_the_retired_residency_element_is_refused_by_value` names **both** retired keys and, on both snapshots, asserts the two `forbidden` problems *and* the two `absent` ones |
| **P5-4** | **Fixed** | `test_a_half_swapped_residency_element_is_invalid` now carries an `expected` parameter and asserts the surviving retired key's own `FieldProblem` — `residentModelsAtEnd[0].modelKey` / `…sizeBytes`, both `forbidden` — instead of `!= []`. The second case's stand-in `bytesOnDisk` is replaced by the real `sizeBytes`, which v1.16's re-scope makes legal |
| **P5-5** | **Fixed on the plan side; code comment swept too** | Appendix A's widening was `architect`'s; the module's `ProblemReason` comment now reads "a **value** this build cannot interpret, in three families (plan Appendix A)" and enumerates them |
| **P5-6** | **Not fixed, and correctly not fixed here** — it is a brief-and-plan item for the `stats.py` unit, not a code change. Carried forward unchanged | `tests/test_results.py` is untouched by `c523a35`, so the R-13 comment is still at `:543` against Table C's `:507` |
| **P5-7** | **Withdrawn** — see §3.4 | — |

### 3. Adjudications

#### 3.1 The substituted fix for P5-1 — the coder's design is right and mine was worse

I recommended (b), or (a) via a distinct `_ABSENT` sentinel. The coder implemented (a) with a
**per-arm** missing-key sentinel — `""` on a model record, `None` on a deterministic one
(`fingerprint.py`, `from_dict`: `missing = None if arm_kind == "deterministic" else ""`). **Its
objection to my `_ABSENT` is correct and I should not have proposed it.** `callSurface` is typed
`Literal["chat", "embeddings"] | None` in the plan's §4 S1 signature block (line 2245) and again in
Appendix A's `Fingerprint` row; a third sentinel inhabitant widens a type the plan closes. And it
breaks the deterministic round trip: `to_dict` omits the key on that arm, so `from_dict` must
reconstruct `None` there — a blanket sentinel reconstructs `""`, which the deterministic branch
(`if self.callSurface is not None`) then reports as `forbidden`. I verified that branch directly:
`Fingerprint(armKind="deterministic", callSurface="", …).validate()` →
`[FieldProblem(field='callSurface', reason='forbidden')]`. So a blanket `""` would report a correct
reference arm as carrying a forbidden surface — exactly what `AGENTS.md`'s new clause says.

The per-arm sentinel keys off `armKind`, which `from_dict` reads one line earlier and which is
already the record's primary discriminator, so it introduces no new state and no new inhabitant. It
is the right substitution. Executed and confirmed: both valid profiles still round-trip equal and
valid.

#### 3.2 The coder's two challenges to Pass 5 — one is right outright, one is right and I was wrong

**(b) — the `armKind` precedent. The coder is right; my prose overstated it.** Executed, both
discriminators, all three stored shapes:

| stored shape | `armKind` | `callSurface` (after `c523a35`) |
|---|---|---|
| key missing | `absent` | `absent` |
| `""` | `absent` | `absent` |
| explicit `null` | `null` | `null` |

`armKind` gives **two reasons over three shapes**, not three — it collapses missing-key with `""`
for the same reason `callSurface` now does, because `""` *is* its missing-key sentinel. My Pass 5
finding quoted the module's "absent is not empty, and `null` is neither" and objected that
`callSurface=""` reported `absent` where "under that discipline a blank `nonempty` value reports
`empty`". That objection had no precedent in the module and I should not have made it: the
discipline that is load-bearing for a *discriminator* is `null`-versus-the-rest, which is what P3-10
established and what was genuinely missing. **The substance of P5-1 stands and is fixed; the
`empty` half of my framing was wrong.** The replacement comment at `test_fingerprint.py:175-181` —
"Two reasons for three stored shapes, and the suite says which two rather than claiming three" — is
the accurate statement, and it also repairs the comment I criticised.

**(a) — "fixing P5-1(a) would kill M2 as a side effect". False, and the coder was right to pin P5-2
separately.** *(Mutation evidence in §4.)* The claim was wrong under the coder's design and, on
re-examination, under my own too: with a missing-key sentinel in place, an unconditional
`"callSurface": self.callSurface` writes `null` on a deterministic arm, `from_dict` reads the key as
*present* with value `None`, and `None` is that arm's legitimate value — so the record round-trips
equal and validates clean, and the mutation is invisible to every test that does not assert the
stored shape directly. P5-2's separate pin is not belt-and-braces; it is the only thing that catches
it. Recorded here rather than defended: the gate was wrong on the mechanism.

#### 3.3 P6-1 (new, minor) — `to_dict`'s omission is keyed on the value, not the arm, so a model record's lost surface is laundered back into an absence

`fingerprint.py`, `to_dict`: `surface = {} if self.callSurface is None else {…}`. The condition is
the *value*, so it fires on a **model** record whose surface is `None` too. Executed:

```
Fingerprint(armKind="model", callSurface=None, fields=model_fields())
  .validate()            -> [FieldProblem(field='callSurface', reason='null')]
  .to_dict()             -> no 'callSurface' key
  from_dict(to_dict())   -> callSurface == ''   (not equal to the original)
  .validate()            -> [FieldProblem(field='callSurface', reason='absent')]
```

So a record diagnosed **`null`** — "something had the value and lost it" — serialises to a record
diagnosed **`absent`** — "never written". That is precisely the information loss P5-1 was raised to
prevent, reintroduced one method over, and the round trip is asymmetric for that shape where it is
total for every other.

**Is it reachable today? No — and that is the whole of the defence.** `store()` calls
`validate()` and raises `InvalidFingerprint` before `RunResult.to_dict()` (`results.py:416`,
`:308`), so this package never serialises a fingerprint it has refused; `load_history` re-validates
and quarantines without rewriting. The coordinator's reading is right that far, and the `null`
reason's job — diagnosing a **foreign** record on AC-2's quarantine line — is served correctly.

**It is still a defect, for the reason this coordination has now hit five times: correct for its
current caller, wrong for a caller the plan commits to adding.** §3.4.3 states that
`REQUIRED_BY_SCHEMA` is "mutable by design: **`model-bench migrate`** and the schema-2 regression
test both key off it". A `migrate` step reads records with `from_dict` and writes them back with
`to_dict` — that is what a migration is — and on this path a `null` surface silently becomes an
omitted key. The record that gets migrated is by definition one that did not validate under the
old contract, so the invalid-record path is the *only* path `migrate` walks.

**Suggested improvement — one line, and it makes the round trip total without touching P5-2's pin:**

```python
surface = {} if self.armKind == "deterministic" else {"callSurface": self.callSurface}
```

Omission then means "this arm calls no surface" (the fact §3.4.2 reserves it for), and a model
record with no captured surface stores `null` (the fact §3.4.2 reserves *that* for) and reads back
as `null`. Verified in §4 that this keeps the suite green, including both halves of P5-2's pin.

#### 3.4 P5-7 — withdrawn, on the coder's rebuttal

I am withdrawing it rather than restating it, and the reason is that my finding was built on a
miscount. I called the `AGENTS.md` paragraph's first half "a third copy" of the module docstring and
the `HISTORY.md` entry. That is not what the convention's "never a third copy" targets: an
always-loaded context file exists to tell an editor what is true **without opening the module**, so
the docstring is not a competing copy of it — it is the same fact at a different reading surface,
for a reader who has already opened the file. The record is `HISTORY.md`, and only `HISTORY.md`.
Two surfaces plus one record is the split the convention prescribes, not a violation of it.

The measurable bars, re-run: `awk 'length($0)>700' AGENTS.md` → no output; `wc -w` → **1,790**
against the ~2,500 smell. The added clause is a live constraint by the file's own test — it changes
what the next editor does, and the thing it prevents is a plausible simplification
(`d.get("callSurface")`) that silently reverts P5-1 — and it is accurate: I confirmed the failure it
names by executing the deterministic-with-`""` case in §3.1. Adding it rather than trimming was the
right call.

This is a withdrawal, not a deferral: there is no residue and nothing is carried.

### 4. Mutation audit

Four counter-implementations against `model-bench/modelbench/fingerprint.py`, each run alone —
`cp` aside, mutate, run, `cp` back, `diff -q` (all four reported `restored-ok`). The suite baseline
is **475 passed**.

| # | The wrong implementation | Result | What it settles |
|---|---|---|---|
| N1 | `to_dict` writes `"callSurface"` unconditionally (the Pass 5 M2 mutation, re-run against the fixed code) | **1 failed**, 474 passed — and the single failure is `test_a_deterministic_record_omits_the_call_surface_rather_than_storing_null` | My Pass 5 claim that P5-1(a) would kill M2 as a side effect is **false**. Nothing but P5-2's own pin catches it |
| N2 | *not a defect mutation* — my proposed P6-1 fix, `surface = {} if self.armKind == "deterministic" else {…}` | **475 passed** | The fix is free: no test moves, and the model/`None` round trip becomes equal with the reason preserved as `null`, while deterministic still omits and still round-trips |
| N3 | the per-arm sentinel replaced by a blanket `missing = ""` — **the design I recommended in Pass 5** | **2 failed**: `test_round_trips_through_a_dict[deterministic]` and `test_the_arm_kind_discriminator_keeps_absent_distinct_from_null` | The coder's refusal of my `_ABSENT` proposal is not a preference — the suite refuses it |
| N4 | the `elif self.callSurface is None → "null"` branch deleted (P5-1 reverted) | **2 failed** | P5-1's fix is pinned in both of its surfaces, matching the coordinator's independent reproduction |

N3 is the one worth recording. The Pass 5 gate proposed a fix that the Pass 6 suite fails on two
distinct assertions, and the implementer's substitution is the one the plan's own type
(`Literal["chat","embeddings"] | None`) admits. A gate's suggested fix is a claim like any other, and
this one did not survive being run.

### 5. What's solid

- **All four Pass 5 majors are closed at the mechanism, not at the symptom.** P5-1 gained a branch
  *and* a sentinel; P5-2 and P5-4 gained assertions that fail on the exact counter-implementations
  that motivated them; P5-3 was closed on both sides, the plan side by a re-scope that is better than
  the reword I asked for — it keeps the residual's *before* value at 1, which my reword would have
  left unaddressed.
- **The fix round corrected the gate twice**, on `armKind`'s precedent and on M2's side-effect claim,
  and both corrections check out against the tree. The replacement comments say something true and
  narrower than what they replace ("two reasons for three stored shapes").
- **`test_the_retired_residency_element_is_refused_by_value` is stronger than DC-1 requires**: it
  asserts the two `forbidden` problems *and* the two `absent` ones, on **both** snapshots, so the
  retired element is diagnosed rather than merely rejected.
- **Scope held again.** `results.py`, `stats.py`, `report.py`, `test_report.py` and `conftest.py` are
  untouched; the `AGENTS.md` edit is four lines and stays inside both of the file's own bars.

### 6. Open items

- **P6-1** — one line in `to_dict`, verified green (N2). Owner: the same coder. Not blocked on
  anything.
- **P5-6** — carried forward, and it is **not residue from this diff**: `tests/test_results.py`'s
  R-13 comment sits at `:543` against plan Table C's `:507`, so the `stats.py` unit's brief must
  re-derive that row by grep. It is an input to the next unit, not an open defect in this one.

Nothing else is open, and nothing is deferred by choice.

## Pass 7 — 2026-09-07

### 1. Scope & verdict

**Reviewed:** `f409905` (`fix(model-bench): P6-1 — the key is omitted for one record, not a class of
them`), the diff `c523a35..f409905 -- model-bench/` — `modelbench/fingerprint.py`,
`tests/test_fingerprint.py`, `docs/HISTORY.md`. **CPG:** considered, not relevant — none exists for
`model-bench/`.

**Verdict: needs changes** — 0 blockers, 0 majors, **1 minor** (**P7-1**), 0 nits. **P6-1 is fixed**,
and the coder was right to refuse my one-liner. The new finding is not in the code — the conjunction
is correct — but in the test that holds it: two further narrowings pass the suite, and one of them
launders the same information P6-1 was raised about. One `parametrize` closes it. **It does not
block the `stats.py` unit** (`fingerprint.py` / `test_fingerprint.py` only).

### 2. The three answers

**(1) The conjunction is one rule, not two special cases — the coder's reading holds.** Omission is
a *claim about the record*: "this arm calls no surface." Neither half can make that claim alone.
`armKind == "deterministic"` says the arm *should* carry no surface and nothing about whether it
does; `callSurface is None` says none is carried and nothing about whether that is legitimate. Only
the conjunction is a fact, which is why each half alone deletes different evidence on the same
`migrate` path. It is also the direct serialisation image of §3.4.1's "`None` **iff**
deterministic" — an `iff` is a conjunction of two implications, and dropping either half of the
condition drops one of them.

The residual observation, and it is not a finding: that `iff` is now **transcribed in two places** —
`validate()`'s deterministic branch and `to_dict`'s `omit` — with nothing tying them together, and
which value a profile pins is a *schema* fact (§3.4.1) that `to_dict` now hard-codes. At schema 1
there is exactly one such pin and the duplication is cheap. If a third transcription ever appears,
or if a schema-2 profile pins another discriminator, that is the moment to lift the rule into the
schema rather than the moment to write it a third time.

**(2) No — it pins the two narrowings the author tried, and two others survive.** Both run alone,
each restored by copy, tree byte-identical after each:

| Candidate `omit` | Suite | What it does |
|---|---|---|
| `self.callSurface is None and self.armKind != "model"` | **477 passed — SURVIVES** | Omits the key on a record whose `armKind` is `""` or unrecognised and whose surface is `null`; `from_dict` restores `""`, so the round trip is no longer equal and a stored `null` is laundered into the missing-key sentinel. **P6-1's laundering, one arm-kind over** — and an unrecognised `armKind` is exactly the record `migrate` exists to walk |
| `self.armKind == "deterministic" and not self.callSurface` | **477 passed — SURVIVES** | Omits on `("deterministic", "")`, where the shipped rule writes `""`; `from_dict` restores `None`, so a record that validates `forbidden` migrates into one that validates **clean** — the evidence-deletion my one-liner was refused for, narrowed to the blank-string case |

I confirmed the shipped code holds on both shapes it is not tested against: for `armKind` `""` and
`"robot"` with `callSurface=None`, the key is present and the round trip is equal.

**P7-1 (minor).** `test_an_invalid_call_surface_survives_a_round_trip_with_its_reason` asserts the
right five things over two shapes chosen as the two rejected narrowings, so it pins *those two
refutations* rather than the property its own docstring claims — that the round trip is total. The
totality was established by executing all the shapes, twice, but that verification lives in two
transcripts and not in the suite; nothing stops the next edit from reintroducing it.
**Fix:** replace the two-case `parametrize` with the product — `armKind` over
`{"model", "deterministic", "", <unrecognised>}` × `callSurface` over `{"chat", "", None}` — keeping
the same five assertions minus the reason-equality where `validate()` short-circuits on `armKind`.
That kills both candidates above and makes the docstring true.

**(3) Two overstatements, one of them mine.** Both in the same sentence, in the code comment
(`fingerprint.py`, `to_dict`) and in the `HISTORY.md` entry: *"an invalid record is the only kind it
[`migrate`] walks"* / *"by definition walks records that did not validate"*. That is false, and it is
false in my Pass 6 phrasing first — the coder inherited it. §3.4.3 states a record is validated
against **its own** schema entry, "so an added field at a later version never invalidates an older
record": most records a migration walks are valid under their own version. The true and sufficient
claim is narrower — `migrate` is the one writer that serialises records `store()` never validated,
so it is the only path on which an invalid record *can* be written. The finding survives the
correction; the sentence does not, and it should be corrected in both places.

Second, a count that does not reproduce: `HISTORY.md` says the round trip was verified over **eight**
`(armKind, callSurface)` shapes, twice; the coordinator's independent enumeration reports **nine**.
One of the two is wrong and neither is in the suite — which is P7-1 from the other side. Closing
P7-1 by parametrizing over the product makes the number a fact the file states rather than a claim
it asserts.

### 3. On the pattern — I agree it is real, and it is narrower than it looks

Three recommendations of mine were overruled this round and all three were rightly overruled: the
`_ABSENT` sentinel, the `armKind` three-state framing, and the `to_dict` one-liner. Every *finding*
held, including P6-1. The split is not luck, and the mechanism is worth writing down for whoever
reads this document next:

**The findings come from executing the code; the fixes came from reasoning about it.** Each finding
was a mutation that survived or a call whose output I printed. Each bad fix was a design I derived
from reading the plan and the module, and none of the three was subjected to the standard I hold
counts and greps to. The one I *did* run — Pass 6's N2, the `to_dict` one-liner, **475 passed** —
was wrong anyway, and P7-1 is why: running a suggested fix against a green suite proves only as much
as the suite constrains, and the property that fix broke was exactly the one nothing pinned. So the
sharpened rule is not "run your suggested fix" but **"a suggested fix is not evidence unless you can
name the assertion that would catch it being wrong"** — and where you cannot, say so and hand the
design decision to the implementer, which is where it belongs.

**How to read this document:** trust the findings, which are evidence-backed and were each
reproduced by at least one other party; treat the *suggested improvements* as one option costed by
someone who did not have to make it work. That is the right division of labour between a gate and an
implementer, and this round demonstrated it three times.
