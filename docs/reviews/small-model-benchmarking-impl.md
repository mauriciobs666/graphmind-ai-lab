# Small-LLM benchmarking tool (`model-bench/`) — S1 implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/small-model-benchmarking.md` §4 S1

**Pass 1** gated `ab91419` (needs changes). **Pass 2** re-gated `3ad27d3` (approve with
suggestions). **Pass 3** re-gated `95b4c88` (needs changes). **Pass 4** re-gated `d55f4d8` (needs
changes). **Pass 5** gated `8fc2341`, the first of S1e's three implementation units (needs changes).
**Pass 6** re-gated its fix round `c523a35` (needs changes). **Pass 7** re-gated `f409905` (needs
changes). **Pass 8** gated `cc28d48`, S1e's second and largest unit (needs changes). **Pass 9**
re-gated its fix round `7f865e2` (needs changes). **Pass 10** re-gated `93b0e42` (needs changes;
N4/N5 closed at `e162ba9`, outside this document). **Pass 11** re-runs DC-12's end-of-round check
over all eight landed S1e tables at `bb24a44` — jump to [`## Pass 11`](#pass-11--2026-09-09) for the
current verdict; the earlier passes are kept intact because they are meant to be read together.
Passes 1–4 gate the S1 build; Passes 5–7 are S1e's first unit (§4 S1e Tables A and B, the
`fingerprint.py` re-key); Passes 8, 9 and 10 are its second (Tables C, D, E and G); Pass 11 is the
round-level re-run, once all eight tables (through Table H) have landed. **Pass 7 §3 says which half
of this document to trust** — the findings held, three suggested fixes did not; Passes 8, 9 and 10
are written to that standard and name, for each suggested fix, the assertion that catches it being
wrong. Five of my fixes have now been rightly overruled and none of my findings has.

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

## Pass 8 — 2026-09-08

### 1. Scope & verdict

**Reviewed:** commit `cc28d48` (`feat(model-bench): S1e Tables C, D, E and G — one percentile,
parameterised clamp and levels`), the diff `c19f875..cc28d48 -- model-bench/` — 9 files, +885/−238:
`modelbench/{stats,results,report,packs}.py`, `tests/{test_stats,test_results,test_report}.py`,
`model-bench/AGENTS.md`, `model-bench/docs/HISTORY.md`. This is §4 S1e Tables **C, D, E and G**,
S1e's largest unit. Plan citations are pinned to
`git show 1ed8599:docs/plans/small-model-benchmarking.md` and
`git show 1ed8599:docs/plans/small-model-benchmarking-ml.md`, an `architect` being mid-revision to
v1.17 in the working copy. **CPG:** considered, not relevant — none exists for `model-bench/`.

**Verdict: needs changes** — 0 blockers, **3 majors** (**P8-1**, **P8-2**, **P8-3**), 3 minors
(**P8-4**, **P8-5**, **P8-6**), 1 nit (**P8-7**).

Both adjudications go the implementer's way: `Verdict.bound_by` is not a closed type being widened
(§2.1) and `results.py:599-600` is **blocked on unbuilt work**, not deferred by choice (§2.2). The
statistics are right — I re-derived the closed form against an independent implementation (§5) — and
the seven-test churn cost exactly one assertion (§3), which is P8-2. The three majors are each one
or two lines to close and all three sit inside this unit's own blast radius. Re-ran to establish
them, not inherited: **519 passed**, `ruff` clean, and **9 mutations** run one at a time with the
file restored byte-identical between each (§4, Appendix E).

I found **no fourth residual defect**: all five of this round's residual counts reproduce exactly as
the coordinator reported (Appendix E, table 1), so **F1/F2/F3 stand as the whole of the plan's
residual debt** and none of my findings is a re-statement of them.

### 2. The two adjudications

**2.1 `Verdict.bound_by` and `envelope_arms()` — right not to stop, and one thing the fork's
premise did not cover.**

`Verdict` is closed by neither document, so widening it is not the fork it was told to stop at.
Appendix A (plan `1ed8599`, line 5423) makes `Verdict` "**`-ml` §3.4's, verbatim** — not restated
here"; `-ml` §3.4's own preamble then says "**Types are illustrative; the seven rules are binding**"
(`-ml` line 728). §3.4 prints no `class Verdict` body at all. The only field list anywhere is the
non-exhaustive "`Verdict` carries `mcnemar_p`, `b`, `c`, `marginal_overlap`, `floor_demoted` and
`holm_tested`" (`-ml` line 1346), which omits six fields the shipped `Verdict` demonstrably has.
Meanwhile Rule 4's "**name which arm bound each bound**" (`-ml` line 1210) is binding and needs a
carrier. The implementer's own distinction holds too: it left `conservative_envelope`'s return type
alone, which the plan's Table D row *does* close.

The one counter-argument, and it does not carry: the note refuses "a field that is `None` until it
is not" (`-ml` line 1347). `bound_by` is not that shape — it is `None` **iff**
`decided_by == "mcnemar-exact"`, a discriminated union over a field `Verdict` already carries, the
same `None`-iff form as §3.4.1's `callSurface`/`armKind` that Pass 7 §2(1) endorsed. What is missing
is what Pass 7 asked for *there*: **nothing pins the iff**, and `_decided_by_line` transcribes the
discriminator a second time by branching on `bound_by` rather than on `decided_by`. Folded into
P8-2's fix.

`envelope_arms()` is the right seam in shape — the alternatives are worse, and the code comment at
`stats.py:1112-1116` refuses the renderer-recomputes option for the correct reason. What the seam
did cost is **P8-5**, and that is one helper away.

**2.2 `results.py:599-600` — blocked on unbuilt work.**

Table C's cell (plan `1ed8599`:3269) says the two values "come from the run's own `LatencyBlock`
**(§4 S2)**" — the row assigns its own replacement to S2 in its own text, and Appendix A (plan line
5419) records `LatencyBlock` as "produced by S2". `grep -rn LatencyBlock modelbench tests
--include='*.py'` returns **one** line, and it is the disclosure comment this diff added. There is
no S1 spelling of that cell: `-ml` §11's two floors are applied *inside* the block that does not
exist, so a percentile taken here bypasses them however it is written. The half that *is* actionable
at S1 — no private copy, `results.py` importing `stats.percentile` — was done, and drives Table C's
first residual to 0 (verified). Named in the code (`results.py:590-597`) and in `HISTORY.md`.

Not "deferred by choice", and I checked the discriminating question rather than taking the label:
there is no cheaper faithful S1 alternative, because the floors have no S1 home.

### 3. The seven deleted/renamed tests — one assertion lost, and it is P8-2

| Test | Claim | Verified |
|---|---|---|
| `…seed_comes_from_the_pack_and_is_printed_beside_the_instrument` | parenthetical gone | **No loss.** The replacement asserts `"seed" not in md` over the whole page — strictly stronger than the deleted `"20260902" not in md` |
| `…the_printed_seed_is_the_seed_the_interval_was_resampled_at` | nothing resamples | **No loss.** Its successor pins the two-arm rendering on `(4, 5, 3, 0)`, the separating table |
| `…the_seed_is_not_printed_where_no_bootstrap_decided_anything` | carried by the above | **Half lost** — its `assert "- decided by: mcnemar-exact" in md` was the only positive assertion of that branch. **P8-2** |
| `test_the_bootstrap_path_refuses_without_a_seed` | unrepresentable | **Confirmed** — `bootstrap_seed` is gone from `verdict()`, and `…takes_no_diffs_no_b_and_no_seed` asserts the parameter set |
| `…envelope_refuses_a_table_that_does_not_describe_its_rows` | unrepresentable | **Confirmed** — no `diffs` argument survives for `n` to disagree with. See the nit in §7 on the `n <= 0` refusal that *is* still representable |
| rename → `test_verdict_refuses_a_design_effect_below_one` | no second raise left to order against | **Confirmed for the code as delivered** — and only because of **P8-3**. `verdict`'s envelope branch reaches `envelope_arms`, which checks `design_effect` nowhere. Restoring that refusal restores the second raise; see P8-3 for how the ordering comes back with it |

### 4. The two self-reported mutants

**D1 — refuted as *equivalent*; correct as *unreachable on the levels the current caller supplies*,
which is a different claim.** With `>=` → `>` at `stats.py:337` the suite passes (519), but
`exact_paired_quantiles((0, 1, 1, 0), levels=(Fraction(1, 4), Fraction(3, 4)))` returns
`(0.0, 1.0)` mutated against `(-1.0, 0.0)` shipped, and at `levels=(Fraction(1, 4), Fraction(1))`
the mutant raises `IndexError`. Both are legal levels for the signature, and `Fraction(1)` is one
`percentile`'s own test asserts. Filed as **P8-4**. The implementer's narrower claim is *true* and I
confirmed it independently: an exhaustive sweep of every `(b, c)` at n ∈ {10, 20, 30, 40} finds **no**
tie at `LEVEL_CI95_LO`/`LEVEL_CI95_HI` (Appendix E, table 3).

**D4 — confirmed killed.** Replacing `envelope_arms`'s widened exact arm with the bare
`exact_paired_quantiles(...)` fails `test_both_arms_are_widened_about_the_same_point_by_the_same_factor`
on **all three** parametrized tables — including `(34, 6, 0, 0)`, where the implementer reported it
invisible. The reason is the fix's own design: the new test asserts at the **arm** level through
`envelope_arms`, not through the envelope, so MOVER-D binding both bounds no longer hides it.

### 5. The statistics at review altitude

- **`percentile`'s rank is `-ml` §11.2.1's expression, character for character** — `max(1, min(x,
  -(-level.numerator * x // level.denominator)))` against the note's line 2581. Substituting the
  level-first float spelling `math.ceil(float(level) * x)` kills two tests (§11.10 2a and 2b). The
  four `LEVEL_*` values are correct. `test_the_percentile_is_the_inverse_empirical_cdf_over_its_whole_domain`
  is a genuine oracle: the expectation is written as `inf{ v : F(v) >= level }` and never as the
  rank, over 60 sample sizes × 20 levels, on a shuffled sample whose values are not their indices.
- **The closed form is right, checked against a second implementation, not read.** I re-derived the
  exact distribution by an independent n-step convolution over the three per-unit values and it
  agrees with `exact_paired_quantiles` on all 7 tables tried, degenerate `b = c = 0` included
  (Appendix E, table 2). The weight `n!/(n₊!n₀!n₋!)·b^n₊·(a+d)^n₀·c^n₋` over `total = n**n` is
  exactly `n**n · P(multinomial)`, so the arithmetic is integer-exact and §11.2.1's bin-edge hazard
  is *removed* rather than guarded, as Table D claims. The selector is `F(s) >= p` in integers — the
  same operator `percentile` applies to a sample — so the two agree by construction.
- **`_widen`'s `clamp`, required with no default:** the argument holds, and the tests carry it in the
  one way a grep cannot. The `assert paired_cluster_bootstrap(diffs, clamp=(0.9, 1.5), **kw) ==
  (0.9, 1.5)` line pins **both** components against arbitrary values, which is the right answer to
  F1 — Table E's residuals match text the edit destroys, and this assertion does not.
- **Routed to `data-scientist`, not adjudicated here.** Whether a `(-1, 1)` clamp belongs on the
  envelope's *arms* at all is a methodology question P8-1 touches but does not settle: the clamp is
  what creates the tie, and an alternative reading of Rule 4's "widened the same way" applies the
  support bound to the **composed** interval instead, which would dissolve P8-1 rather than patch
  its symptom. That is a statistical call, not mine.

### 6. Findings

**P8-1 (major) — the `- decided by:` audit names the arm that did not bind, once the clamp binds
both.** `modelbench/stats.py:1119-1122`. The attribution is computed on the **clamped** arms, so
when `_widen`'s `(-1.0, 1.0)` binds both, the comparison is between two equal numbers and the
`<=`/`>=` tie-break silently prints `MOVER-D`. Measured at `(a=5, b=0, c=7, d=0)`, n=12, DEFF 4.0:
the *unclamped* lower bounds are MOVER-D −1.0301 and exact −1.0833, so the **exact paired bootstrap**
is the binding arm — and `verdict(...).bound_by` is `('MOVER-D', 'MOVER-D')`, which the renderer
prints as `lower bound: MOVER-D`. Swept over every table at n ∈ {12, 38, 40}: **0** misattributions
at DEFF 1.0 and 1.2, **2 per bound** at 1.5, **29–32 per bound** at 4.0, 680 in all (Appendix E,
table 4). Unreachable today (every §3.8 pack declares DEFF 1.00) and reachable the moment S2's
determinism probe measures a design effect ≥ 1.5 — the "invisible rather than urgent, free rather
than deferred" shape Table G uses on itself. Why it matters: this bullet exists *because* "a
sentence naming an instrument that did not produce the number beside it is the defect class this
document exists to remove" (`-ml` line 1223), and the mutation `<=,>=` → `<,>` survives the suite,
so nothing pins the tie-break either way. **Suggested fix — a design call, so here is the assertion
that judges it:** attribute from the arms *before* `_widen` clamps them (`envelope_arms` already has
them), or give a clamp-bound bound its own token. Either way,
`verdict(_outcomes(5, 0, 7, 0), resolving=_rp(12, deff=4.0, basis="measured"), metric_name="m",
family=["m"]).bound_by[0]` must stop being `"MOVER-D"`. It is today; I ran it.

**P8-2 (major) — the `mcnemar-exact` rendering of that same bullet is now asserted nowhere.**
`modelbench/report.py:334-335`. Mutating the `bound_by is None` branch to
`return "- decided by: MUTANT"` gives **519 passed**. `grep -rn 'decided by: mcnemar-exact' tests/`
returns exactly one line — `test_report.py:922` — and it is a `not in` on the envelope path. The
positive assertion lived in the deleted `test_the_seed_is_not_printed_where_no_bootstrap_decided_anything`
and did not move to either replacement. This is the one genuine loss in the seven-test churn, and it
is on the `by-construction` path every tool-caller comparison takes at DEFF 1.00. **Fix:** one line
— `assert "- decided by: mcnemar-exact" in md` — added to an existing `_nested_arms()` report test
(`test_report.py:106` or `:857`); no new test needed. **Fold in §2.1's missing invariant** while
there: `assert (v.bound_by is None) == (v.decided_by == "mcnemar-exact")` inside
`test_the_verdict_records_which_arm_bound_each_printed_bound`, which already constructs both paths.

**P8-3 (major) — `conservative_envelope`/`envelope_arms` silently accept `design_effect < 1.0`, a
refusal the shipped code had and Table D did not retire.** `modelbench/stats.py:343-367`. At
`c19f875`, `conservative_envelope(diffs, (34,6,0,0), design_effect=0.25, B=200, seed=1)` raises
`ValueError: design_effect must be >= 1.0 (-ml §3.4 Rule 4, precondition 4)` — run against the
`git show c19f875:` source, not inferred. Today `conservative_envelope((34,6,0,0),
design_effect=0.25)` returns `(0.0909, 0.2204)`, **narrower** than the `(0.0318, 0.2907)` at 1.00.
Table D retires only the `n != len(diffs)` guard; this one retired by accident, because it lived in
`paired_cluster_bootstrap`, the call the collapse removed. `verdict()` still guards it, so nothing
that runs today is wrong — but both functions are public, `envelope_arms` is new, and this diff's
own `AGENTS.md` edit sits four lines from the sentence "below 1 it *inflates* effective *n* and
shrinks both printed bounds". **Fix:** restore the raise in `envelope_arms`, reusing
`paired_cluster_bootstrap`'s exact message. **And the honest cost:** that reintroduces a second raise
— what the retired ordering half of `test_verdict_refuses_a_design_effect_below_one` existed to
disambiguate. It is recoverable, because the two messages are *already* distinguishable
(`stats.py:1071` prefixes `verdict() precondition 4:`), so the ordering re-pins as
`pytest.raises(ValueError, match=r"verdict\(\) precondition 4")` in place of the retired seed device.
The assertion that catches the fix being absent: `pytest.raises(ValueError, match="precondition 4")`
on `conservative_envelope((34, 6, 0, 0), design_effect=0.5)`.

**P8-4 (minor) — D1's classification, and the two-line fixture that settles it.** See §4. The mutant
is distinguished by a legal input, so "equivalent" overstates it; "unreachable at
`LEVEL_CI95_LO`/`LEVEL_CI95_HI`" is exactly right and is what the reader needs. **Fix:**
`assert exact_paired_quantiles((0, 1, 1, 0), levels=(Fraction(1, 4), Fraction(3, 4))) == (-1.0, 0.0)`.
This is not merely a mutation-killer: `(0, 1, 1, 0)` has CDF `1/4, 3/4, 1` over `s ∈ {-2, 0, 2}`, so
the fixture lands the level **exactly on an atom boundary** — the one place `inf{ s/n : F(s) >= p }`
and `inf{ s/n : F(s) > p }` differ, and the property the docstring's "the two agree by construction"
claim rests on.

**P8-5 (minor) — the envelope's composition rule is written twice, and `envelope_arms`'s docstring
says it is not.** `stats.py:414` (`conservative_envelope`) and `stats.py:1118` (`verdict`) both spell
`min(exact[0], mover[0]), max(exact[1], mover[1])`. Both are independently pinned — swapping each
`min`/`max` costs 5 and 4 failures respectively — so this is not a live bug. What is wrong is the
pair of claims around it: `conservative_envelope` is now **dead in production** (only tests call it,
`grep` above), while `envelope_arms`'s docstring asserts "`conservative_envelope` composes it, and
`verdict()` reads the attribution off it, so **neither recomputes the other's arithmetic**" — which
`verdict` does. Plan §3.9's own rule is that two copies of a formula is one copy and one bug.
**Fix:** one `_compose(mover, exact)` helper called from both, or have `envelope_arms` return the
composed pair beside the arms. Either makes the docstring sentence true and restores
`conservative_envelope` to a live caller's path.

**P8-6 (minor) — `exact_paired_quantiles`'s refusals are a weaker second copy of `percentile`'s and
`paired_bootstrap`'s, and none of them is pinned.** `stats.py:326-330`. Deleting the six-line
transposed-pair guard outright gives **519 passed**: its twin in `paired_bootstrap` is tested
(`test_paired_bootstrap_refuses_a_transposed_level_pair`), this one is not. The same function is
also *weaker* than `percentile` on the two refusals it does not carry: a `float` level reaches
`.numerator` and dies with `AttributeError` rather than §11.2.2's `TypeError`, and a level above 1
falls off the end of the atom loop and raises `IndexError` on `bounds[1]` rather than the
`(0, 1]` `ValueError`. Not reachable from `envelope_arms`, which hard-codes the pair — reachable from
the signature, which advertises `levels` as the caller's. **Fix:** parametrize the existing
transposed-pair test over both callables (its body already has the right shape), and route the level
through the same two refusals `percentile` applies — cheapest as a shared `_check_level(level)`, one
call in each.

**P8-7 (nit) — `AGENTS.md`: both edits earn their place; two small things.** Edit 2 is
unambiguously right — it *corrects a now-false clause* (`PackRef.seed` as what "the report both uses
and prints"; the report no longer prints it) and folds `levels` and `clamp` into the existing
no-defaults list, which is a live constraint an editor adding a default would otherwise revert. Edit
1 earns its place too, and for a sharper reason than it states: it is the guard against the *wrong*
reaction to F2's residual count of 2. Two things. **(i)** That clause states a disposition of an
open finding `architect` has not ruled on — if v1.17 resolves F2 by renaming or by a shared
`_quantile_from_cdf`, the always-loaded file is the highest-cost place for the stale version to sit;
half a clause citing F2 as open would keep it honest. **(ii)** The insertion left line 81 at 131
characters where the file otherwise wraps near 100. Both smells are clear at the bar that matters:
1,952 words, `awk 'length($0)>700'` returns nothing. Neither is worth a change before v1.17 lands.

**One nit with no number, because it is not this diff's regression.** The `n <= 0` refusal is now
written in both `envelope_arms` and `exact_paired_quantiles` and `grep -rn 'describes no rows'
tests/` returns nothing — no test reaches either. It is the natural home for the intent of the
deleted `…refuses_a_table_that_does_not_describe_its_rows`, whose *own* guard was correctly retired
as unrepresentable. And `HISTORY.md`, thorough as it is, does not say that the estimator swap
**moves the printed figures** — over `1..100`, `latencyMsP50` goes 51 → 50 — which is a visible
`index.csv` change across this commit for anyone comparing runs over it.

### 7. What's solid

- **The statistics.** The closed form verified against an independent convolution on 7 tables; the
  rank expression verbatim from `-ml` §11.2.1 and killed by its float substitution; the four level
  constants correct. This is the part it would have been easiest to get subtly wrong.
- **The residual honesty.** All three plan findings are real, correctly characterised, and reported
  in the commit message and `HISTORY.md` rather than quietly worked around — including the refusal
  to rename `exact_paired_quantiles` to make F2's grep read 1, which would have destroyed the exact
  property that residual exists to have. All five residual counts reproduce (Appendix E, table 1).
- **The test work is a step up.** The inverse-ECDF oracle written from the definition; the
  `(0.9, 1.5)` clamp assertion that answers F1 where no grep can; `test_both_arms_are_widened…`
  parametrized over the tables that separate the arms — which is what killed D4;
  `results_module.percentile is stats.percentile` as **identity**, not behaviour.
- **R-13/M27 is genuinely closed.** Recomputing the `latencyMsP95` cell at `LEVEL_P50` — M27's own
  defect, verbatim — now fails a test. It used to stay green; that is the whole point of Table C.
- **The rendered bullet matches the note's published example byte for byte**
  (`-ml` line 1214), which is not something the plan restated and had to be taken from the note.

### 8. Open questions

1. **P8-1's fix is a `data-scientist` question before it is a `coder` one** (§5, last bullet).
   Attributing from the unclamped arms is the local patch; applying the support bound to the
   composed interval instead removes the tie entirely. Rule 4's "widened the same way" is what
   decides, and it is a methodology call.
2. **P8-7(i) waits on v1.17.** If `architect` resolves F2 other than by leaving
   `exact_paired_quantiles` named as it is, `AGENTS.md`'s new clause needs the same edit in the same
   pass — worth putting on the v1.17 sweep list now rather than discovering later.

### Appendix E — Pass 8 evidence

Everything below was run this session under `model-bench/` with `.venv/bin/python`. Baseline
**519 passed** in ~3.0 s; every mutation was applied to a `cp`-backed copy, run alone, restored, and
`diff -q`'d byte-identical before the next.

**Table 1 — the five residual counts, reproduced.**

| Residual | Stated | Measured | Note |
|---|---|---|---|
| C-1 `grep -rFc _percentile modelbench/results.py` | 0 | **0** | — |
| C-2 `grep -rFc _percentile modelbench/stats.py` | 0 | **0** | — |
| C-3 `grep -rEn 'def [A-Za-z_]*(percentile\|quantile)' modelbench` | 1 | **2** | `stats.py:264` + `:464` — **F2**, already raised |
| D-1 `grep -rFn bootstrap_seed modelbench tests` | 0 | **1** | `test_stats.py:878`, substring of `test_cluster_bootstrap_seed_…` — **F3**, already raised |
| D-2 `grep -rFn 'cluster-bootstrap' modelbench tests` | 0 | **0** | — |
| E-1/E-2, G-1/G-2 | 0 each | **0 each** | E's pair is blind by construction — **F1**, already raised |

**Table 2 — the closed form against an independent n-step convolution** (a second implementation of
the exact distribution, not a re-read of the first), at `(LEVEL_CI95_LO, LEVEL_CI95_HI)`:

`(0,6,0,34)`, `(4,5,3,0)`, `(1,25,12,2)`, `(34,6,0,0)`, `(2,19,7,12)`, `(0,0,0,7)`, `(3,0,4,1)` —
**7 of 7 agree exactly**, degenerate `b = c = 0` (both quantiles 0) included.

**Table 3 — the D1 tie search.** Every `(b, c)` at n ∈ {10, 20, 30, 40}, testing
`level.denominator · cum == level.numerator · n**n` at both shipped levels: **0 ties**. So the
selector's `>=` is never exercised as an equality on this path — the implementer's claim — while
`(0, 1, 1, 0)` at `Fraction(1, 4)` exercises it immediately at a level the signature accepts.

**Table 4 — P8-1's misattribution sweep.** Every `(a, b, c, 0)` at n ∈ {12, 38, 40}, comparing the
arm the code names (clamped) against the arm that actually binds (unclamped):

| DEFF | 1.0 | 1.2 | 1.5 | 2.0 | 4.0 | 7.0 |
|---|---|---|---|---|---|---|
| n=12 (per bound) | 0 | 0 | 2 | 4 | 20 | 39 |
| n=38 (per bound) | 0 | 0 | 2 | 4 | 29 | 101 |
| n=40 (per bound) | 0 | 0 | 2 | 4 | 32 | 101 |

**Table 5 — the nine mutations.**

| # | Mutation | Result |
|---|---|---|
| 1 | `_decided_by_line`'s `bound_by is None` branch → a constant string | **519 passed — SURVIVES** → P8-2 |
| 2 | `conservative_envelope`'s `min`/`max` swapped | 5 failed — killed |
| 3 | `verdict`'s inline `min`/`max` swapped | 4 failed — killed (and 2+3 together are P8-5) |
| 4 | atom selector `>=` → `>` | **519 passed — SURVIVES** → P8-4 (not equivalent; see table 3) |
| 5 | exact arm not `sqrt(DEFF)`-widened (D4) | 3 failed — killed on all three tables |
| 6 | `bound_by` tie-break `<=,>=` → `<,>` | **519 passed — SURVIVES** → P8-1 |
| 7 | `exact_paired_quantiles`'s transposed-pair guard deleted | **519 passed — SURVIVES** → P8-6 |
| 8 | `latencyMsP95` cell computed at `LEVEL_P50` (M27 verbatim) | 1 failed — killed |
| 9 | rank → `math.ceil(float(level) * x)` | 2 failed — killed (§11.10 2a and 2b) |

**Working-tree discipline.** No source, test, config or plan file was modified. `git status
--porcelain` at close shows `docs/reviews/small-model-benchmarking-impl.md` as this review's only
entry under my hand; the `claude/**` and `docs/plans/small-model-benchmarking.md` entries belong to
the concurrent sessions the brief named. Nothing staged, nothing committed.

## Pass 9 — 2026-09-08

### 1. Scope & verdict

**Reviewed:** commit `7f865e2` (`fix(model-bench): Pass 8 fix round — six findings closed, P8-1 held
for Rule 4a`), the diff `cc28d48..7f865e2 -- model-bench/` — 5 files, +431/−43:
`modelbench/stats.py`, `tests/{test_stats,test_report}.py`, `model-bench/AGENTS.md`,
`model-bench/docs/HISTORY.md`. Six of Pass 8's seven findings were in scope; **P8-1 was excluded by
the coordination** and is now blocked on unbuilt work, its root ruled in note v1.19
`docs/plans/small-model-benchmarking-ml.md` §3.4 **Rule 4a**.

**CPG:** considered, not relevant — none exists for `model-bench/`.

**Verdict: needs changes** — 0 blockers, **2 majors** (**N1**, **N2**), 0 minors, 1 nit (**N3**).

**All six in-scope Pass 8 findings are closed** (§2), and both deliberate deviations are correct —
the first one reverses the question that was put to me, because the NaN-safe spelling was already
the module's own before this round (§3.1), and the second overrules me on a plan revision written
after my pass (§3.2). The six retired test ids cost nothing: resolved at **id** level by collecting
both trees, every retired assertion survives in a strictly stronger form (§4). `_compose` is a good
seam for Rule 4a and two of its three deltas were forced by the scope boundary rather than chosen
(§5).

The two majors are **one respelling at two sites**. P8-3 diagnosed the predicate correctly and fixed
it at one of the three places that carried it: `verdict()` (`stats.py:1126`) and
`paired_cluster_bootstrap` (`:228`) are still on `< 1.0`, and the second of them returns
`(-1.0, 1.0)` for a NaN — verbatim the symptom that justified the deviation, on `-ml` §3.2d's
continuous entry point. Both fixes are the edit this round already made once, and each needs one
`parametrize` the neighbouring test already carries.

Re-ran only what the coordinator could not: **2 mutations**, each applied to a `cp`-backed copy, run
alone, restored and `diff -q`'d byte-identical — swapping `_compose`'s `min`/`max` (**9 failures**,
the union of Pass 8's separately-pinned 5 and 4) and reordering `exact_paired_quantiles`' two level
checks (**2 failures**, exactly the two rows §4 predicts). Plus the direct NaN probes behind N1/N2.
I did not re-run the suite, the 14-mutation set, `ruff`, the word counts or F2's closure; those are
the coordinator's and are taken as given.

### 2. Disposition of Pass 8's findings — 6 of 6 in scope closed, 1 held

| # | Sev. | Disposition | Evidence I rechecked |
|---|---|---|---|
| **P8-2** | major | **Fixed** | `test_the_decided_by_bullet_names_mcnemar_exact_where_one_instrument_decided` (`test_report.py:1722`) asserts `"- decided by: mcnemar-exact\n" in md` on `_nested_arms()`, plus `"conservative envelope" not in md`. The `bound_by`-iff assertion I asked to fold in **also landed**, at `test_stats.py:1393` |
| **P8-3** | major | **Fixed, with a deviation I accept** — §3.1 | `envelope_arms` raises at `stats.py:376`; parametrized over `{0.5, 0.25, 0.999999, 0.0, -1.0, nan}` × `{conservative_envelope, envelope_arms}`, and the retired ordering property is back as `test_the_two_envelope_refusals_name_which_layer_raised`. **The fix is right and incomplete — §6 N1/N2** |
| **P8-4** | minor | **Fixed, and better than I proposed** | `test_the_exact_quantile_takes_the_atom_the_level_lands_on` parametrizes three rows over `(0, 1, 1, 0)` — on the boundary, below, above — where I suggested one. The docstring states the reachability scope correctly rather than repeating my "not equivalent" phrasing as a slogan |
| **P8-5** | minor | **Fixed** — `_compose` at `stats.py:392`, both callers through it. Judged as a seam in §5 | `conservative_envelope` is `return _compose(*envelope_arms(...))`; `verdict` is `ci = _compose(mover_arm, exact_arm)`. `test_the_two_routes_to_the_envelope_compose_the_arms_identically` asserts the two routes agree over four DEFFs |
| **P8-6** | minor | **Fixed, and wider than I proposed** | `_check_level` is one home for both refusals; `exact_paired_quantiles` now raises the note's `TypeError`/`ValueError` where it gave `AttributeError`/`IndexError`. I asked for a parametrized transposed test; it parametrized **all three** level refusals over both estimators |
| **P8-7** | nit | **Superseded — my suggestion was wrong** — §3.2 | Plan v1.17's F2 row rules against a rename and restates the target as two survivors named; a clause calling F2 open would be false. Separately, the 131-char line I nitted was rewrapped: `AGENTS.md`'s longest line is now 122, and that one predates this round |
| unnumbered nit | — | **Fixed** | `test_an_empty_paired_table_is_refused_by_every_function_that_takes_one` with `match="describes no rows"`; `HISTORY.md` now records the `latencyMsP50` 51 → 50 figure move |
| **P8-1** | major | **Held — blocked on unbuilt work, not deferred** | Its root is ruled in `-ml` v1.19 §3.4 Rule 4a and the implementation is a separate unit. The coordination's M14 control surviving at 550 passed is the right evidence that nothing was half-moved in the meantime; I did not re-run it |

### 3. The two deliberate deviations — both correct, and the first one reverses the question

**3.1 P8-3's spelling: `not design_effect >= 1.0` rather than my `< 1.0`. Accept — and the "two
spellings now coexist" concern does not survive contact with the module.** The NaN-safe spelling was
**already there before this round**: `resolving_power` has carried
`if not design_effect >= 1.0:  # NaN-safe: '< 1.0' would admit a NaN design effect` since before
`cc28d48` (`git show cc28d48:model-bench/modelbench/stats.py`, line 792). So the deviation did not
introduce a second spelling — it moved one more site onto the spelling the module already used for
this exact precondition, comment and all. My `< 1.0` would have been the divergent one.

That reframes the coordinator's question. Coexistence is not the defect; **which sites are still on
the unsafe spelling** is, and two are: `paired_cluster_bootstrap` (`stats.py:228`) and `verdict`
(`stats.py:1126`). Both are live NaN holes, one of them reproducing the exact symptom that justified
this deviation. Filed as **N1** and **N2**.

**3.2 P8-7: it corrects me, and it is right. One line, because that is what a correction is worth.**
I asked for "half a clause citing F2 as open." Plan v1.17 (`b6f578c`) closes F2 — its row at
`docs/plans/small-model-benchmarking.md:5528` restates the residual's target as **"2 → 2 with both
survivors named"**, says plainly that "the count is no longer the check — the named line set is",
and rules against a rename *because surviving a rename is the property that residual exists to
have*. A clause calling F2 open would be false, and false in the always-loaded file. What shipped
instead — **"Do not rename it to make §11.10(3)'s grep read 1"**, with the ruling cited — is the
live constraint the file is for. Verified against the plan, not taken on report.

That is the **second** suggested fix of mine overruled this round, and the fourth across Passes 6–9.
The split is the one Pass 7 §3 named: every *finding* has held; the *fixes* are one option costed by
someone who did not have to make them work. Here the implementer had two things I did not — the
module's existing spelling, and a plan revision written after my pass.

### 4. The six retired test ids — nothing is asserted less than it was

I resolved this at **id** level rather than by reading diffs, by collecting both trees: `git archive
cc28d48 model-bench` into a scratch directory and `pytest --collect-only -q` against each. **519 →
550, exactly 6 ids retired and 37 added**, matching the reported `+37/−6`. The six are three `def`s,
one of them parametrized four ways:

| Retired id | What it asserted at `cc28d48` | Successor | Delta |
|---|---|---|---|
| `test_paired_bootstrap_refuses_a_transposed_level_pair` | `raises(ValueError)` on `(HI, LO)` and on `(LO, LO)`, over `[1.0, 0.0, -1.0]`, `B=100`, `seed=1` | `test_every_level_pair_estimator_refuses_a_transposed_pair[paired_bootstrap]` — **identical call and fixture** | gains `match="ordered lower then upper"`; gains an `exact_paired_quantiles` instance |
| `test_percentile_rejects_a_float_level` | `raises(TypeError)` on `percentile(range(20), level=0.05)` | `test_every_level_estimator_rejects_a_float_level[percentile]` — **identical call** | gains `match="exact rational level"`; gains the second estimator |
| `test_percentile_rejects_a_level_outside_the_unit_interval[level0…3]` | `raises(ValueError)` on `{0, −1/20, 21/20, 2}`, **plus** a trailing `percentile(range(20), level=Fraction(1)) == 19` run inside all four | `…rejects_a_level_outside_the_unit_interval[level0…3-percentile]`, same four levels and call; the `Fraction(1)` assertion moves **verbatim** into `test_a_level_of_exactly_one_is_admitted_and_returns_the_largest_value` | gains `match=r"level must lie in \(0, 1\]"`; gains the second estimator; the `(0, 1]` closed end stops being an incidental tail executed four times and becomes a named test |

**Nothing lost, and one thing genuinely gained.** Every retired assertion is present in a strictly
stronger form, and the `match=` additions are not cosmetic here — I checked the case where they
carry real weight. `_LEVEL_CALLERS["exact_paired_quantiles"]` calls
`exact_paired_quantiles((0, 1, 1, 0), levels=(level, Fraction(1)))`, so at `level ∈ {21/20, 2}` the
**transposed-pair** guard would also raise `ValueError` — a bare `pytest.raises(ValueError)` would
pass on two of the four rows without `_check_level` existing at all. The `match=` is what makes
those rows test the refusal they name, and it also pins the check *order* inside
`exact_paired_quantiles` (levels validated before the transposed comparison), which nothing else
states. That is the one place this refactor could have silently weakened a test, and it did not.

### 5. `_compose` as a seam for Rule 4a — a good seam, and two of its three deltas were forced

**Rule 4a's target** (`-ml` §3.4 Rule 4a, point 2 and point 4): `envelope_arms` widens both arms with
`clamp=None`; one private composer takes the two **unclamped** arms, clamps its own result to
`(-1.0, 1.0)`, computes `bound_by` from the composed unclamped value against the support on a
**three**-token set with strict comparisons, and returns `(interval, bound_by)`.
**What shipped:** `_compose(mover, exact) -> tuple[float, float]` — no clamp, interval only,
`bound_by` still inline in `verdict()` on the two-token set. Three deltas, all self-recorded.

**They are one decision with three faces, not three half-steps.** Clamping inside `_compose` is only
correct if `envelope_arms` stops clamping in the same edit — otherwise the arms arrive already
clamped and the composed clamp is a no-op that *reads* as applied, which is worse than absent. And
moving the clamp out of `envelope_arms` changes the values `bound_by` is computed from, which **is**
P8-1. So the alternative to "does not clamp" was never "clamps"; it was "does P8-1", which the
coordination excluded. Delta 2 follows from delta 1 — `bound_by` cannot move in until `_compose`
holds the unclamped values and the support, because Rule 4a's third token is computed from
`u_lo < L` / `u_hi > U` — and delta 3 is delta 2 restated. Moving `bound_by` in *now*, on the
two-token set, would have put the wrong computation in the right place: that is the half-step this
avoided, not the one it took.

**Nothing has to be undone.** The next unit's edit is: `clamp=(-1.0, 1.0)` → `clamp=None` in
`envelope_arms`; `_compose` gains the support constant, the clamp and the `bound_by` computation and
returns a pair; its two call sites adjust by one subscript and one unpack; `verdict()`'s four inline
`bound_by` lines are deleted. The function's **name, location, privacy, both callers and the Rule 4a
paragraph in its own docstring all survive** — the return type widens, which is an extension. And
the seam is strictly better than not having it: Rule 4a itself says "compose-and-clamp gets exactly
one home, **which closes `P8-5` as collateral**", so the note is already written on the assumption
that one composer exists. Landing it early means Rule 4a's edit touches one function instead of
hunting two spellings — which is exactly the work P8-5 asked to remove.

**The docstring's forward claim is sourced, not asserted.** "The two placements commute exactly, so
the printed numbers do not move when it lands" is Rule 4a's measurement — 173 472 combinations, zero
differences — and Rule 4a is cited two sentences earlier, so a reader can reach it.

**Recording the deltas at the seam rather than in a handoff note is right, with one obligation
attached.** A handoff note is a document that has to be found; a docstring is read by whoever opens
the function to change it, and the only readers of a module-private composer are its two callers and
its next editor. The risk runs the other way: once Rule 4a lands, "it is not applied here yet"
becomes false in the one place the next reader trusts most. **That paragraph must be rewritten in
the same commit that applies Rule 4a** — worth naming now, because it is the kind of line that
survives an edit by looking like context.

**Verified, not read:** swapping `_compose`'s `min`/`max` costs **9 failures**, the union of Pass 8's
separately-pinned 5 and 4. One home, one mutation, both routes.
`test_the_two_routes_to_the_envelope_compose_the_arms_identically` is not a tautology — it asserts
`verdict(...).ci == conservative_envelope(table, design_effect=deff)` exactly, over five tables × four
design effects, including the tables where the arms disagree about which is conservative.

### 6. New findings — P8-3's fix is right and lands at one of the three sites that had the defect

**N1 (major) — `verdict()`'s own precondition 4 is still NaN-blind, so the layer-ordering property
this round restored is false at exactly one value.** `stats.py:1126` spells
`if resolving.design_effect < 1.0:`, which is `False` for a NaN. Ran it: with
`dataclasses.replace(rp, design_effect=nan, n_effective=nan)` — the same constructor bypass the
suite's own sub-1 fixtures use — `verdict()` raises `design_effect must be >= 1.0 (-ml §3.4 Rule 4,
precondition 4)`, which is **`envelope_arms`' message, not its own**; at `deff=0.5` it correctly
raises `verdict() precondition 4: …`. The value is caught, but by the inner layer, and Rule 4's
"checked **here and before any instrument is selected**" is violated for NaN.
`test_the_two_envelope_refusals_name_which_layer_raised` cannot see it: it takes `deff: float = 0.5`
as a **default argument** and is not parametrized — twelve lines below a sibling that *is*
parametrized over `nan` for precisely this reason. **Fix:** respell `:1126` as
`if not resolving.design_effect >= 1.0:`, and give the ordering test the same `deff` list its
sibling already carries. **The assertion that catches it:** that test at `deff=float("nan")`. It
fails today.

**N2 (major) — `paired_cluster_bootstrap` still returns a maximally wide interval for a NaN design
effect, which is the exact symptom that justified the deviation.** `stats.py:228` is still
`if design_effect < 1.0:`. Ran it:
`paired_cluster_bootstrap([1.0, 0.0, -1.0, 1.0], design_effect=float("nan"), B=50, seed=1,
clamp=(-1.0, 1.0), levels=(LEVEL_CI95_LO, LEVEL_CI95_HI))` returns **`(-1.0, 1.0)`** — the full
support conjured out of a missing number, verbatim the failure the commit message reproduces at
`envelope_arms` and cites as its reason for deviating. This one needs **no bypass at all**:
`design_effect` is a bare float parameter here, with no `resolving_power` in the path, and this is
`-ml` §3.2d's **continuous** entry point — the surface Rule 8's `continuous_verdict()` is specified
to call, where the value arrives from a pack manifest. **Fix:** the same one-line respelling, and
extend `test_paired_cluster_bootstrap_refuses_a_design_effect_below_one` with the `parametrize` its
envelope sibling twenty lines away already has.

*N1 and N2 together are one sentence: the round diagnosed the predicate correctly and fixed it at
one of the three sites that carry it.* `resolving_power` was already safe; `envelope_arms` is now;
`verdict()` and `paired_cluster_bootstrap` are not. Both fixes are the edit this round already made
once.

**N3 (nit) — `AGENTS.md` now describes the NaN-safe guard using the NaN-unsafe spelling.** The
rewritten bullet reads "`resolving_power` refuses `design_effect < 1.0` at construction", while
`resolving_power` (`stats.py:855`) deliberately spells it `not design_effect >= 1.0` under a
`# NaN-safe` comment. In an always-loaded file, in the round whose whole finding is that `< 1.0` is
the wrong predicate, that sentence invites exactly the simplification the code comment exists to
prevent. **Fix:** three words — "refuses any `design_effect` not `>= 1.0`". The rest of both edits is
right, and the ordering clause added beside it — *"only `verdict()`'s message names itself, which is
what keeps the two orderable"* — is a genuine live constraint that N1 makes more load-bearing, not
less.

### 7. What's solid

- **The retired-test refactor is a net strengthening, and I checked it at id level rather than by
  reading diffs** (§4). Every one of the six retired assertions survives in a stronger form, and the
  `match=` additions do real work: without them two of the four `exact_paired_quantiles` rows would
  pass off the transposed-pair guard. Mutating the check order proves it — 2 failures, exactly those
  two rows.
- **Two of the three fixes went wider than I asked.** P8-4 got three parametrized rows covering the
  operator's neighbourhood where I proposed one fixture; P8-6 parametrized all three level refusals
  over both estimators where I asked only for the transposed one, and wrote down the totality
  argument (`level <= 1` is what makes the atom loop always append) that the refusal now carries.
- **Both deviations are argued from evidence the implementer gathered**, not from preference: a
  reproduction against the baseline for the NaN spelling, and a plan revision written after my pass
  for F2. Pass 7 §3's split holds for a fourth round — the findings held, the fixes did not.
- **P8-1 was held cleanly rather than half-moved.** The M14 control surviving at 550 passed is the
  right evidence for that, and `_compose`'s docstring makes the next unit's edit legible without a
  handoff document.

## Pass 10 — 2026-09-08

### 1. Scope & verdict

**Reviewed:** commit `93b0e42`, the diff `7f865e2..93b0e42 -- model-bench/` — 4 files, +113/−12:
`modelbench/stats.py` (2/2), `tests/test_stats.py` (28/5), `model-bench/AGENTS.md` (8/5),
`model-bench/docs/HISTORY.md` (75/0). A narrow closure pass on Pass 9's **N1**, **N2** and **N3**;
**P8-1 remains held**, blocked on `-ml` §3.4 Rule 4a's own unit (plan v1.20, in flight).

**CPG:** considered, not relevant — none exists for `model-bench/`.

**Verdict: needs changes** — 0 blockers, **1 major** (**N4**), **1 minor** (**N5**), 0 nits.

**All three Pass 9 findings are closed** (§2), both by exactly the respelling named, and N2's test
gained a `match=` it never had. The two sweeps are six well-chosen examples rather than the property
— but I checked the rejection domain for a gap and there is none, and the axis that mattered turns
out to be *which side of the predicate is under test* (§3). The end-of-line comment shape is the
module's own, established before the pins existed, so nothing bent (§4). N3's widening holds, and my
three words would have left the hazard live (§5).

**N4 is the answer to the fourth-path question, and it is a real one.** The four guards close the
*below* side of `>= 1.0` by construction and nothing closes the *above* side or the data: `+inf`
passes every guard, and `verdict()` then returns `ci = (-1.0, 1.0)` with
`bound_by = ('MOVER-D', 'MOVER-D')` — a full-support interval attributed to a named instrument. A
`nan` or `inf` anywhere in `diffs` does the same on the continuous entry point, which has no guard on
its data at all. The mechanism is `_widen`'s clamp rather than the predicate, which is one step past
where the round's own `AGENTS.md` sentence stops. **Not blocked:** I ran the candidate one-line guard
in `_widen` — **560 passed**, all three shapes refused, and it does not collide with Rule 4a, because
under Rule 4a `_widen` still runs before `_compose` clamps.

**P8-1 remains legitimately held**, blocked on Rule 4a's own unit (plan v1.20). Nothing in this pass
is deferred by choice: N4 and N5 are both closeable now.

Spent only on what the coordinator could not check: the two probes behind N4, one candidate-fix run,
and an id-level read of the two sweeps. I did not re-run the suite, the guard grep, the five
mutations, the line-count pins or the word counts.

### 2. Disposition of Pass 9's findings — 3 of 3 closed

| # | Sev. | Disposition | Evidence I rechecked |
|---|---|---|---|
| **N1** | major | **Fixed** | `stats.py:1126` reads `not resolving.design_effect >= 1.0`, and `test_the_two_envelope_refusals_name_which_layer_raised` is parametrized over six values including `nan`. Ran it: at `deff=nan`, `verdict()` now raises its **own** `verdict() precondition 4: …` message where it previously fell through to `envelope_arms`' |
| **N2** | major | **Fixed, and slightly wider than I asked** | `stats.py:228` respelled; the test swept over the same six **and gained `match="precondition 4"`**, which it never had — it was a bare `pytest.raises(ValueError)`. Ran it: the NaN call raises where it returned `(-1.0, 1.0)` |
| **N3** | nit | **Fixed, wider than I asked, and the widening holds** — §5 | The bullet names all four sites and the mechanism; `AGENTS.md`'s longest line is unchanged at 122, which predates this round |
| **P8-1** | major | **Still held — blocked on unbuilt work** | `-ml` §3.4 Rule 4a's own unit, plan v1.20, in flight. The coordinator's N-M5 control surviving at 560 is the right evidence that clamp placement is untouched; I did not re-run it |

### 3. Do the sweeps assert the property, or six examples? — six, correctly chosen, and the question is on the wrong axis

**They are six examples, not the property** — the property's domain is *every float for which
`not d >= 1.0`*, which is uncountable and unsweepable. But the six are a **partition of the rejection
domain by predicate-failure mode**, not a list: two ordinary sub-1 values, one boundary-adjacent
(`0.999999`, which a `<`/`<=` slip admits), zero (the division hazard), negative (the `sqrt` domain
hazard), and `nan` (the predicate hazard). Each row can fail for a different implementation reason,
which is what distinguishes a partition from an anthology.

**I checked for a gap inside that domain and there is none.** The one class the six omit is `-inf`;
`not -inf >= 1.0` is `True`, so both guards fire and both messages match `-1.0`'s exactly —
behaviourally the same row. Nothing in the rejection domain is unrepresented.

**But both tests only ever look at one side of the predicate, and that is where the risk went.**
N1's residual was never inside the rejection domain: `+inf` **satisfies** `>= 1.0`, so it passes all
four guards, and no sweep of a *refusal* test could ever have found it. That is **N4** (§6). The
useful answer to the question as posed is therefore: the axis that mattered here was not
property-versus-examples but *which side of the predicate is under test*, and the answer is that
neither test looks at the accepting side.

**The strictly stronger form, if the coordination wants it**, is one loop rather than six rows:
assert that `verdict()`'s guard and `envelope_arms`' guard **agree** — same accept/reject decision,
`verdict()`'s message first when both reject — over a generated domain that includes the specials.
That is closed under the predicate rather than enumerated, and it extends to the accepting side for
free, which is exactly what would have caught N4.

### 4. The end-of-line comments — readability was paid for, and the plan constraint did not bend the code

**The shape is the module's own, and it predates the pin pressure.** `resolving_power` has carried
`if not design_effect >= 1.0:  # NaN-safe: '< 1.0' would admit a NaN design effect` since before
`cc28d48` — established in Pass 9 §3.1 against `git show cc28d48:…`. So the two new comments match
the site that already had this exact guard with this exact comment shape. Being line-count-neutral
is a *consequence* of matching the existing convention, not the reason for it; had the convention
been a block, the pins would have argued for a block. The plan constraint and the house style
agreed here, which is why nothing had to bend.

**Measured rather than taken:** the four `# NaN-safe` lines are **97, 96, 85 and 97** characters
against `line-length = 100` (`stats.py:228, 376, 855, 1126`) — I measure 97 for both new ones where
the brief said 95 and 93; either figure is under the limit and `ruff` is clean, so the discrepancy
changes nothing.

**And they are not boilerplate, which is the real test of whether a 97-character line earns its
width.** Each says something different and site-specific: `:228` records *what the defect produced*
("a NaN here returned the full support as an interval"), `:1126` records *how it escaped* ("a NaN
fell through to the arms' own raise"). A reader at either site learns the thing they could not
recover from the predicate. Four near-identical comments would have been the failure; these are not
that.

### 5. N3's widening holds — and my three words would have left the hazard live

I asked for three words that removed a false sentence. The coder's argument is that removing a false
sentence is not the same as installing a true constraint, and it is right: nothing in the three-word
version stops the next editor simplifying `verdict()`'s `not x >= 1.0` back to `x < 1.0` — which is
**precisely what this round just undid at two sites**, and which the *previous* round's version of
that same bullet had invited. The test for an always-loaded line is whether it changes what the
reader does next. *"The predicate is that way round at all four sites … do not simplify any of them"*,
plus the one-clause mechanism, does. *"Refuses any `design_effect` not `>= 1.0`"* does not.

It is a live constraint rather than history: the four guards exist now, the temptation is
demonstrably live, and the mechanism is one clause rather than a narrative of how it was found.
**Second time this coder has widened one of my nits with a reason, and both times rightly** — the
same split Pass 7 §3 named, now at five overruled fixes and no overruled finding.

**One rider, and it belongs to N4 rather than to the bullet.** The mechanism clause is true and is
attached only to the `design_effect` predicate, which invites a reader to conclude that the four
guards close the NaN class. They do not (§6). When N4 is closed, that bullet earns one more clause;
writing it now would be documenting a defect rather than stating a constraint, so it should wait.

### 6. Yes — there is a fourth path, and it is the accepting side of the same predicate

**N4 (major) — a `design_effect` of `+inf`, or a non-finite value anywhere in `diffs`, reaches the
same clamp and prints as a real interval.** The four guards close the *below* side by construction;
nothing closes the *above* side or the data. `not inf >= 1.0` is `False`, so `+inf` passes every one
of them, `math.sqrt(inf)` is `inf`, `_widen` returns `(-inf, +inf)`, and the clamp's `max(-1.0, …)` /
`min(1.0, …)` return the support bounds. All measured this session:

| Call | Returns |
|---|---|
| `verdict(_outcomes(34,6,0,0), resolving=<deff=inf>, …)` | **`ci = (-1.0, 1.0)`, `bound_by = ('MOVER-D','MOVER-D')`** — a full-support interval attributed to a named instrument |
| `conservative_envelope((34, 6, 0, 0), design_effect=inf)` | `(-1.0, 1.0)` |
| `paired_cluster_bootstrap(diffs, design_effect=inf, clamp=(-1.0, 1.0), …)` | `(-1.0, 1.0)` |
| `paired_cluster_bootstrap([nan] + diffs, design_effect=1.0, clamp=(-1.0, 1.0), …)` | `(-1.0, 1.0)` — **no guard on `diffs` at all** |
| the same with `[inf] + diffs` | `(-1.0, 1.0)` |
| `paired_bootstrap([nan] + diffs, …)` — no clamp applied | `(nan, nan)` — *visibly* wrong, which is the contrast that identifies the clamp as the launderer |

**The mechanism is `_widen`'s clamp, not the predicate** — which is the generalisation the round's own
`AGENTS.md` sentence invites and stops one step short of. `max(-1.0, nan)` returns `-1.0` and
`min(1.0, nan)` returns `1.0`, because every comparison with a NaN is `False`, so the clamp converts
"no number" into "the widest honest number" silently and in the direction that prints. Reachability
today is the same standing N1 and N2 had when they were filed as majors: `resolving_power` refuses
`inf` incidentally (`n_effective must be positive`), so the `verdict()` route needs the same
`dataclasses.replace` bypass N1 did — but `conservative_envelope`, `envelope_arms` and
`paired_cluster_bootstrap` are public, take bare floats, and the last is `-ml` §3.2d's continuous
entry point that Rule 8's `continuous_verdict()` is specified to call with a manifest value.

**Not blocked on unbuilt work — one line, at the point all three paths converge.** I ran the
candidate rather than proposing it: adding, in `_widen` after `widened` is computed,

```python
if not all(math.isfinite(b) for b in widened):
    raise ValueError(...)
```

gives **560 passed** — no existing test relies on a non-finite bound flowing through — and refuses
all three shapes above. **It does not collide with the in-flight Rule 4a unit:** under Rule 4a
`envelope_arms` widens with `clamp=None`, so `_widen` still runs and the guard still fires *before*
`_compose` clamps, which is where it needs to be. **The assertions that catch it being absent**, and
they fail today: `pytest.raises(ValueError)` on `conservative_envelope((34, 6, 0, 0),
design_effect=float("inf"))` and on `paired_cluster_bootstrap([float("nan")] + diffs,
design_effect=1.0, clamp=(-1.0, 1.0), …)`.

*Honest limit on that evidence, by Pass 7 §3's own rule:* a green suite proves only as much as the
suite constrains, and no test here exercises a deliberately infinite bound. The design question — a
result guard in `_widen` versus an input guard at each of the three entry points — is the
implementer's; the result guard is the one that closes all three with one assertion, and it is where
I would start.

*One clause of the same family, well below finding weight:* `percentile([1.0, nan, 3.0],
level=LEVEL_P50)` returns `nan`, because `sorted()` does not order a NaN. It prints as `nan` rather
than as a plausible number, so it launders nothing, and `_index_row`'s only float source is a timing.
Named here so it is not rediscovered as new.

### 7. The other new finding

**N5 (minor) — the six-value design-effect domain is now written out three times.**
`tests/test_stats.py:1430`, `:1451` and `:1724` each carry the literal
`[0.5, 0.25, 0.999999, 0.0, -1.0, float("nan")]`. Adding a seventh class — `-inf`, or `inf` once N4
lands — means three edits, and missing one is silent: the sweep still passes, one surface just stops
being swept. This is plan §3.9's own rule ("two copies of a formula is one copy and one bug") on the
test side, and it is the rule this round has now applied twice to the source (`_percentile`,
`_compose`). **Fix:** one module-level constant, e.g. `_SUB_ONE_DESIGN_EFFECTS`, referenced by all
three `parametrize` marks. **The assertion that catches it being wrong:** none is needed — the
change is mechanical and the three id-lists must stay identical, which `pytest --collect-only` shows
directly.

## Pass 11 — 2026-09-09

### 1. Scope & verdict

**Reviewed:** not a diff — **DC-12's end-of-round re-run**, over the tree at `bb24a44` (a docs-only
ledger commit atop `f17efa2`/`359c463`; `model-bench/` is byte-identical to `359c463`, confirmed by
`git show --stat`). All eight §4 S1e tables (A–H) have landed; this is the residual property's
**round-level** check — a later table's edit can move an earlier landed table's number — which has
never been run before now. Also re-ran: the disowning-mention sweep, the third-form/fragility sweep,
all eight tables' *enumerating* commands against their stated baselines, and the Table H row-text
equivalence claim from `bb24a44`'s commit message. `model-bench/`: **648 passed**, `ruff` clean,
tree clean (`git status`).

**CPG:** not applicable — none exists for `model-bench/`, and this pass is pure re-run/verification
of grep-based residuals, not a code-level task a CPG would help with.

**Verdict: needs changes** — 0 blockers, **1 major** (**M11-1**), 0 minors, **1 nit** (**M11-2**).
Both are new; nothing is carried from Pass 10 (N4/N5 closed at `e162ba9`/U62, confirmed in the
coordination ledger and not re-checked here — out of DC-12's scope).

**DC-12 itself is clean.** I extracted **24** residuals from §4 S1e (§2) — the plan's own stated
count, no miscount. All 24 re-run against the current tree and **all 24 hit their stated target**
(§2); no mismatch, so §3's mismatch-investigation step found nothing to investigate. Both standing
sweeps re-run clean over the current 24 (§4): no new disowning mention, and the fragility partition
is **20 robust / 4 third-form (Table E's pair, Table H's pair) / 0 blind** — independently
re-derived, not just quoted from the plan's own v1.23 self-report, which states the same numbers.
All eight tables' enumerating commands still return their stated counts and per-file splits against
their stated baselines — **zero drift**, extending U68's own Table H check to the other seven (§5).
The Table H row-text claim is **confirmed on both halves** (§6): the shipped form is equivalent to
Rule 4a's ruling, and the plan's own row text is inconsistent with its own residuals — the **ninth**
instance of the plan being the defective party, as flagged. **M11-1** is that this ninth instance is
not yet corrected in the plan text, and — per the standing rule — it is **not blocked on unbuilt
work**: it is a self-contained editorial fix to one table row, fixable now, so it may not be carried
as "deferred by choice."

### 2. All 24 residuals, extracted and re-run

Extraction method: read §4 S1e in full (plan lines ~3243–4237, Tables A–H), pulling every line
introduced by "Residual after the edit" (or, for Table H, the six-row residual table). Seven of the
eight tables state theirs in prose immediately after that heading; only Table H's are a markdown
table. Count: A 2, B 4, C 3, D 2, E 2, F 3, G 2, H 6 = **24**, matching DC-12's stated count exactly
— no reconciliation needed.

All commands run with `model-bench/` as the working directory, against `bb24a44`. "Target" is each
residual's stated **after** value (every table has landed, so "after" is what the current tree must
show).

| # | Tbl | Command (verbatim) | Target | Observed | Result |
|---|---|---|---|---|---|
| 1 | A | `grep -rFc lmsCliCommit modelbench tests --include='*.py'` | 0 | 0 | PASS |
| 2 | A | `grep -rFn sizeBytes modelbench tests/conftest.py --include='*.py'` | 0 | 0 | PASS |
| 3 | B | `grep -rFc FORBIDDEN_BY_ARM_KIND modelbench tests --include='*.py'` | 0 | 0 | PASS |
| 4 | B | `grep -rFn 'frozenset(FORBIDDEN' modelbench --include='*.py'` | 0 | 0 | PASS |
| 5 | B | `grep -rFn 'REQUIRED_BY_SCHEMA[1]["model"]' modelbench tests --include='*.py'` | 0 | 0 | PASS |
| 6 | B | `grep -rFn 'set(REQUIRED_BY_SCHEMA[1]) == {"model",' modelbench tests --include='*.py'` | 0 | 0 | PASS |
| 7 | C | `grep -rFc _percentile modelbench/results.py` | 0 | 0 | PASS |
| 8 | C | `grep -rFc _percentile modelbench/stats.py` | 0 | 0 | PASS |
| 9 | C | `grep -rEn 'def [A-Za-z_]*(percentile\|quantile)' modelbench --include='*.py'` | 2, named (`stats.py:601 percentile`, `stats.py:304 exact_paired_quantiles`) | 2, same two names | PASS |
| 10 | D | `grep -rEn '\bbootstrap_seed' modelbench tests --include='*.py'` | 0 | 0 | PASS |
| 11 | D | `grep -rFn 'cluster-bootstrap' modelbench tests --include='*.py'` | 0 | 0 | PASS |
| 12 | E | `grep -rFn 'max(clamp[0], widened[0])' modelbench --include='*.py'` | 1 (`stats.py:301`) | 1, same line | PASS |
| 13 | E | `grep -rFn 'min(clamp[1], widened[1])' modelbench --include='*.py'` | 1 (`stats.py:301`) | 1, same line | PASS |
| 14 | F | `grep -nF 'separationRaw: float \| None' modelbench/results.py` | 0 | 0 | PASS |
| 15 | F | `grep -nF 'separationZ: float \| None' modelbench/results.py` | 0 | 0 | PASS |
| 16 | F | `grep -rFn '{"binary", "continuous"}' modelbench --include='*.py'` | 0 | 0 | PASS |
| 17 | G | `grep -rFn 'percentile(means, level=LEVEL_CI95_LO)' modelbench --include='*.py'` | 0 | 0 | PASS |
| 18 | G | `grep -rFn 'percentile(means, level=LEVEL_CI95_HI)' modelbench --include='*.py'` | 0 | 0 | PASS |
| 19 | H | `grep -nF 'clamp=(-1.0, 1.0)' modelbench/stats.py` | 0 | 0 | PASS |
| 20 | H | `grep -nF 'SUPPORT_DIFF_PROPORTIONS[0]' modelbench/stats.py` | 1 (`:466`) | 1, same line | PASS |
| 21 | H | `grep -nF 'SUPPORT_DIFF_PROPORTIONS[1]' modelbench/stats.py` | 1 (`:467`) | 1, same line | PASS |
| 22 | H | `grep -nF '"MOVER-D" if mover_arm[0] <= exact_arm[0]' modelbench/stats.py` | 0 | 0 | PASS |
| 23 | H | `grep -nF 'arm if arm == "MOVER-D" else' modelbench/report.py` | 0 | 0 | PASS |
| 24 | H | `grep -nF 'tuple[str, str] \| None' modelbench/stats.py` | 0 | 0 | PASS |

**24/24 PASS.** Residual 9 (Table C's third) is the one whose stated target is deliberately
non-zero — the plan requires the count **and** the two survivor names (`percentile`,
`exact_paired_quantiles`) to match, per §7 rule 5(b)'s named-line-set clause; both hold.

### 3. Mismatch investigation

**Nothing to investigate — all 24 residuals hit their target on the first run.** No later table's
edit moved an earlier table's number: Table G's `stats.py:159` collision with Table C was already
resolved by ordering (C then G) at landing and both commands still read their stated values; Table
H's `envelope_arms`/`_compose` rewrite, which Table E's row explicitly warns is the collision point
("Table H meets this table, and this table does not move"), left Table E's two third-form residuals
at their stated 1/1, confirmed live in §2 rows 12–13. This is the round-level property working as
designed rather than by luck: every pair of tables the plan itself names as meeting on a line (C/G on
`stats.py:159`; E/H on `_widen`'s body and `envelope_arms`'s call sites; H/F on `report.py`) was
checked at its own seam and none moved the other's residual.

### 4. The two standing sweeps, re-run over the current 24

**4a. Disowning-mention sweep — no new instance, the three known ones still hold.** The three
previously-found instances (Table A residual 2, scoped to `modelbench` + `tests/conftest.py`; Table
C residual 9/"3", scoped to `modelbench` because the unscoped form matches `def
test_percentile_rejects_a_float_level`-shaped test names; Table H residual 19/"1", matched on the
keyword form `clamp=(-1.0, 1.0)` rather than the bare tuple, because the bare tuple also matches a
sentence in `_widen`'s own docstring) are all still in place verbatim in the landed commands (§2 rows
2, 9, 19) and all three still read their stated value. I additionally spot-checked every
broadly-scoped (`modelbench tests`) residual with a stated-zero target — rows 1, 3–6, 10–11 — for a
comment or test name that would inflate the count above zero: none exists at `bb24a44` (all six
observed at 0, §2). No fourth instance found.

**4b. Third-form/fragility sweep — 20 robust / 4 third-form / 0 blind, re-derived.** The
discriminator: does the matched span include text the table's own edit rewrites? I classified all 24
by reading each command against its site's edit, not by citing the plan's own v1.19/v1.23 self-report
(plan lines 3064–3090), which states the same numbers and is corroborating rather than the source
here:

- **Robust (20)** — span is either the retired token whole (rows 1–8, 10–11, 14–19, 22–24 — 19
  rows), or a construct whose *form* the edit leaves alone while changing a name/key/value inside it
  (row 6, `REQUIRED_BY_SCHEMA[1]`'s value-display subscript — the subscript syntax survives, only the
  set literal inside changes). That is 20.
- **Third-form / fragile (4)** — rows 12–13 (Table E) and 20–21 (Table H). Both pairs are stated over
  text a *parameterising* edit creates, because the edit rewrites the expression the literal was
  fused into (`_widen`'s clamp becoming an argument; `_compose`'s composed return becoming a clamp
  over a composition) — a first-form residual over the pre-edit text would go blind on exactly the
  half-application it exists to catch (documented at length on both tables' rows, and reproduced for
  Table E's pair in Pass 9/impl-gate F1). Both pairs still read their non-zero target live (§2), so
  neither has yet been asked to survive a further rewrite of the expression it pins — the risk is
  named, not realised.
- **0 blind.** No residual currently reads a value consistent with a faithful edit *and* with the
  defect it exists to catch — the failure mode both sweeps exist to find.

This matches the plan's own §7 rule 5(b) count ("twenty of twenty-four robust, four third-form (E 2,
H 2), none fragile") — independently reproduced rather than merely trusted.

### 5. Enumerating-command re-check — do all seven other tables' commands still return their stated counts?

**Confirmed: yes, all seven, exactly, including per-file splits.** U68 (the Table H unit) already
re-ran Table H's own two enumerating commands against its `e162ba9` baseline (`bound_by` → 15,
`envelope_arms` → 22) and reported them undisturbed by the three units landed since. I ran the other
seven tables' enumerating commands against **their own stated baselines** — `5878014` for the six
tables the plan says "keep their own commits" (A, B, C, D, E, G) and `e162ba9` for Table F (re-
pointed at v1.23) — using `git grep -c <pattern> <rev> -- <pathspec>`, which reads the tree at that
ref without touching the working copy:

| Table | Command | Baseline | Stated | Observed | Per-file match |
|---|---|---|---|---|---|
| A | `lmsCliCommit` | `5878014` | 3 | 3 | exact |
| B (1–6) | `armKind`, `FORBIDDEN_BY_ARM_KIND`, `ARM_KINDS`, `arm_kind`, `REQUIRED_BY_SCHEMA`, `EXPECTED_MODEL_SCHEMA_1` | `5878014` | 50, 10, 2, 18, 22, 3 | 50, 10, 2, 18, 22, 3 | exact, all six |
| C | `_percentile` | `5878014` | 7 | 7 | exact |
| D (1–4) | `bootstrap_seed`, `conservative_envelope`, `cluster-bootstrap`, `DecidedBy` | `5878014` | 29, 8, 27, 3 | 29, 8, 27, 3 | exact |
| D (subrow) | `paired_cluster_bootstrap`, `paired_bootstrap` | `5878014` | 13, 8 | 13, 8 | exact |
| E (1–2) | `_widen`, `paired_cluster_bootstrap(` | `5878014` | 7, 5 | 7, 5 | exact |
| F (1–8) | `scored_outcome`, `ItemResult(`, `ContinuousMetric`, `separation`, `named_metrics`, `isinstance(.*BinaryMetric`, `\.mean`, `"continuous"` | `e162ba9` | 17, 14, 5, 2, 12, 6, 3, 2 | 17, 14, 5, 2, 12, 6, 3, 2 | exact, all eight |
| G (1–3) | `97.5`, `paired_bootstrap(`, `paired_cluster_bootstrap(` | `5878014` | 2, 5, 5 | 2, 5, 5 | exact |

**Zero drift on every command, at every table's own stated baseline.** This is the answer to the
round-level question DC-12 exists for: no unit landed since a table's own baseline has silently moved
that table's site-finding surface. (I hit one tooling snag worth recording: `git grep -E
'isinstance(.*BinaryMetric'` fails — `-E` requires the literal `(` escaped, where the plan's plain
`grep -rn` treats it as basic-regex-literal; re-run without `-E`, matching git grep's default mode,
resolved it. Not a finding against the plan — the plan's own command is correct as written for plain
`grep`, and the discrepancy was mine.)

### 6. The known issue — Table H's row text versus the shipped form

**Confirmed on both halves asked.**

**(a) Equivalence.** The shipped `_compose` (`modelbench/stats.py:465-472`) computes
`u_lo, u_hi = min(mover[0], exact[0]), max(mover[1], exact[1])`, then `lo = max(SUPPORT_DIFF_PROPORTIONS[0],
u_lo)` / `hi = min(SUPPORT_DIFF_PROPORTIONS[1], u_hi)`, and attributes `bound_by` from `lo != u_lo` /
`hi != u_hi` rather than re-testing `u_lo`/`u_hi` against the support directly. Algebraically,
`max(S_LO, u_lo) != u_lo` iff `u_lo < S_LO` — exactly Rule 4a's strict support comparison. Verified
by execution too (not just traced):

```
compose_check(-1.5, 0.5)  -> lo=-1.0, hi=0.5, bound_lo='support', bound_hi='arm'   # escapes below
compose_check(-1.0, 0.5)  -> lo=-1.0, hi=0.5, bound_lo='arm',     bound_hi='arm'   # exactly at support
compose_check(-0.9, 1.5)  -> lo=-0.9, hi=1.0, bound_lo='arm',     bound_hi='support'
```

A bound sitting exactly *at* the support attributes to an arm, never to `"support bound"` — Rule 4a's
assertion 5, holding live, not just in the docstring's prose.

**(b) The plan's own row text is inconsistent with its own residuals.** The row at
`docs/plans/small-model-benchmarking.md:4131` (`stats.py:444`) prescribes `bound_by = (…)` "computed
from `u_lo`/`u_hi` against the support" **in addition to** the return line's own
`max(SUPPORT_DIFF_PROPORTIONS[0], u_lo)` / `min(SUPPORT_DIFF_PROPORTIONS[1], u_hi)`. Read literally,
that is a **second**, independent comparison of `u_lo`/`u_hi` against `SUPPORT_DIFF_PROPORTIONS[0]`/
`[1]` inside the `bound_by` branch — so an implementer following the row text as its own two lines
describe would spell each subscript **twice**, and residuals 20/21 (§2) — whose target is exactly
**1** — would read **2** on that literal, faithful reading. The shipped form avoids the second
mention by re-testing `lo != u_lo` (already-computed) instead of `u_lo < SUPPORT_DIFF_PROPORTIONS[0]`
(a fresh comparison), which is why it reads 1, not 2. This is the **ninth** instance of the plan
being the defective party on this coordination.

### 7. Findings

**M11-1 (major, owed to `architect`).** §4 S1e Table H's `stats.py:444` row (plan line 4131) is
internally inconsistent: its own prose ("computed from `u_lo`/`u_hi` against the support") describes
a `bound_by` implementation that, read literally, spells `SUPPORT_DIFF_PROPORTIONS[0]`/`[1]` a second
time, in contradiction with the same row's residuals 20/21 (target exactly 1, §2). The shipped code
sidesteps this correctly (§6a) and the implementer flagged it in the landing commit — but the plan
text itself is still wrong and has not been corrected. **Not blocked on unbuilt work**: this is a
self-contained rewrite of one row's `bound_by` sketch (e.g. reusing `lo`/`hi` via `lo != u_lo`/`hi !=
u_hi` rather than re-deriving a comparison against the support), fixable in the next plan revision
with no dependency on anything unbuilt — per the standing rule, it may not be carried as "deferred by
choice."

**M11-2 (nit).** `modelbench/stats.py:466-467`'s comment reads: `` `lo != u_lo` iff the support was
strictly below the composed lower bound ``. That is backwards: `lo != u_lo` iff `u_lo < SUPPORT_DIFF_PROPORTIONS[0]`
(the composed bound below the support, not the support below the bound) — confirmed by the second
line of §6a's execution trace (`u_lo=-1.5` against `S_LO=-1.0`: the support is *above*, not below,
and the clamp fires). The code is correct; only the comment's direction is stated backwards. Low
stakes — it doesn't affect any residual, test, or the round-level property — but worth a one-clause
fix (e.g. "iff the composed bound fell below the support") since it is exactly the kind of
misdirection the exact-text residual discipline exists to prevent elsewhere.

### 8. What's solid

DC-12's substance is fully clean: 24/24 residuals extracted and matched the plan's own stated count,
all 24 re-run and all 24 hit target, both standing sweeps re-run clean with an independent
re-derivation (not a re-citation) of the fragility partition, and all eight tables' enumerating
commands hold at their stated baselines with zero drift — the round-level property DC-12 exists to
prove is proven, not merely asserted. The Table H unit's own equivalence claim and its "ninth
defect" flag both check out under independent verification (algebra and execution, not just a
re-read of its reasoning).

### 9. Open questions

None that block S1's closure on DC-12's own terms. M11-1 is a plan-text fix `architect` can make
without further input; I did not attempt it myself (out of scope for `analyst`, per the guardrails).

## Pass 12 — 2026-09-09

### 1. Scope & verdict

**Reviewed:** the three S2 wave-1 commits as immutable git objects, never the working tree — `721e8c9`
(U71, the pack loader), `186d30b` (U72, the LM Studio adapter), `3924f3a` (U73, the row-count
exemption fix + all three `HISTORY.md` entries). **Baseline:** `docs/plans/small-model-benchmarking.md`
§3.3, §3.4.4a, §3.6 and §4 S2; `docs/plans/small-model-benchmarking-ml.md` §3.4 Rule 6 and §11.5.1.
**Out of scope and untouched:** `runner`/`hostinfo`/`convo`/`tooling`, the `attest`/`validate`/`run`
CLI, and every S1 surface except where these three commits changed it.

**Method.** Because a concurrent unit may be mutation-testing the same component, every execution in
this pass ran against `git archive 3924f3a`, extracted to a scratch tree, with the editable-install
meta-path finder stripped so `import modelbench` resolves to the snapshot (Appendix L.0). All three
mutations and all seven probes below were run there; the working tree was never read, written or
moved. `tests/test_packs.py` + `tests/test_lmstudio.py` at the snapshot: **61 passed, 1 deselected**.

**Verdict: needs changes.** One blocker, six majors, five minors, two nits. The blocker is not a
green-suite miss of the two known classes — it is a third: **an exception taxonomy that is total over
the connect phase and empty over the body-read phase**, which no fake response in the suite can reach.

**CPG: considered, not relevant — no Code Property Graph is loaded for `model-bench` (only
`cpg_falkorchat` and `cpg_deprecated_salesperson` exist on this instance), so call-graph and
data-flow questions in this pass were answered by reading and by execution against the snapshot.**

**On the brief's premises.** Two are wrong and both are findings, not quibbles. (a) The brief frames
U73 as having closed the exemption-versus-docstring gap; it closed one of three branches and its
own fix predicate opened a fourth (P12-6). (b) The brief asks whether `ChatResult`'s boundary is
total and whether any payload shape yields a wrong number rather than a `None`; the answer is yes,
but the wrong number is `wallClockMs`, which is not a `stats`-derived field at all (P12-4).

### 2. Findings

#### Blocker

**P12-1 — every failure raised while *reading* the response body escapes the adapter's exception
taxonomy, on all six operations.** `_raw_post`'s try/except ladder wraps only `self._opener(...)`
(`modelbench/lmstudio.py:371-388`); `raw = resp.read()` sits at `:391`, outside it. `_raw_get` has
the same shape (`:296-303` guarded, `:305` not). Executed against the snapshot with a response whose
`read()` raises (Appendix L.1): `http.client.IncompleteRead`, `ConnectionResetError` and a read-phase
`TimeoutError` all escape `chat()`, `catalog()` **and `probe()`** untyped — 9 of 9 cells. Why it
matters: §3.6's fourth disposition names "a dropped connection" as the `no_response` case the runner
must score `fail` and **continue** on, and `-ml` §11.5.1 needs a read-phase timeout to arrive as
`timeout` rather than as `no_response`; here both abort the run instead. Worse, `probe()` is
contractually three-valued — `run` exits `3` on `v1-only`/`unreachable` — and it now has a fourth
outcome that is an exception. `IncompleteRead` is not even an `OSError`, so a blanket socket catch
would not cover it. **Suggested fix (judge it, don't take it):** move the body read inside the same
ladder in both `_raw_get` and `_raw_post`, adding `http.client.HTTPException` as a rung. The
assertion that catches me being wrong is the coverage probe in §4B — it fails today at 9 cells and
must reach 0. Not pinned by any test because `_FakeResponse.read()` (`tests/test_lmstudio.py:60-61`)
cannot fail.

#### Majors

**P12-2 — `tools.module` may point outside the pack root, so a pack executes code its own
`contentHash` does not cover.** `Pack.load_tool_module` builds `self.root / module_rel`
(`packs.py:259`) with no containment check, and `content_hash` walks `root.rglob("*")` only
(`:281`). Executed (Appendix L.2): a manifest declaring `"module": "../outside.py"` validates
**CLEAN**, `load_tool_module()` executes the outside file, and editing that file leaves
`content_hash(root)` **unchanged**. That falsifies §3.3's own sentence — "Pack code is part of the
content hash, so a behavior change to a simulated tool is a version change like any other" — and with
it AC-3, the component's core identity claim. **Suggested fix:** resolve `module_rel` and refuse
anything not under `root` (`Path.resolve().is_relative_to(root.resolve())`), in `validate_pack` so it
is reported rather than raised. Cheap and self-contained; not blocked on anything.

**P12-3 — the AST allowlist's declared reach exceeds its mechanism, and one hole is in the
mechanism's own terms.** Three, all executed (Appendix L.3). (i) `tools.module` may be a `.pyc`: the
walk globs `*.py` (`packs.py:510`), so a pack shipping `tools/sim.pyc` validates CLEAN and
`load_tool_module` **executes the unscanned bytecode** — proven by the executed bytecode's own
forbidden import raising from inside it. This is a file-selection gap, not a syntactic-versus-semantic
one, and it defeats the check even under its narrow hygiene reading. (ii) `__import__('modelbench.results')`
and `importlib.import_module('modelbench.report')` both validate CLEAN and both actually reach the
module — expected, since stdlib is allowed wholesale, but it means the check is a *coupling* rule and
never containment. (iii) `packs.py:253` therefore overclaims: "what `validate_pack`'s AST check exists
to make safe before this ever runs against **an untrusted pack**". It makes nothing safe against an
untrusted pack. **Suggested fix:** constrain `tools.module` to a `.py` suffix in `validate_pack`
(closing (i)); rewrite `:253` to say the check is a coupling constraint over syntactic imports, not a
sandbox. Note the direct route is correctly refused — `import modelbench.stats` is caught — so the
mechanism works where it looks.

**P12-4 — `ChatResult.wallClockMs` stops at the response headers, not at the last byte of the body.**
`_raw_post` computes the clock at `:389`, before `resp.read()` at `:391`. §3.6's FR-11 table defines
it as "measured around the HTTP call from just before the request **to the last byte of the body**".
Executed (Appendix L.4): a response that spends 250 ms in `read()` yields
`ChatResult.wallClockMs = 0.001 ms`. This is the brief's "wrong number rather than a `None`", and it
is systematic and always short. Two consequences: `-ml` §11.5.1's gap is `latencyMs − (ttftMs +
generationMs)`, so the in-call-reload detector under-fires; and §3.6's stated *reason* for making the
client wall clock the headline rather than `stats.generation_time` — "it includes request assembly,
transport and the server's own queueing" — is exactly the term the code drops on the response side.
**Suggested fix:** move the `time.monotonic()` stop below `resp.read()`. No test pins the window
(every stub's `read()` is instantaneous), so a test asserting a slow-body stub's `wallClockMs >= 200`
is the assertion that would catch this being wrong.

**P12-5 — `warm_up`'s justification for not using `residentModelsAtStart` is inverted, and it
contradicts the plan.** `lmstudio.py:485-488` says `residentModelsAtStart` "is `[]` by construction on
a cold run and would misreport every cold warm-up as 'already resident' if used here instead."
`model in set()` is `False`, i.e. **not resident** — the correct answer. §3.6 names that source
explicitly: `coldLoadSeconds` is "recorded only when the model was not resident at start
(`residentModelsAtStart`, §3.4.4a)". The reasoning appears to be §3.6 clause (a) transplanted from a
different consumer — the *contamination guard's baseline for item 1*, where `[]` genuinely does
misfire. The substituted behaviour may well be an improvement (it sees a model loaded between
capture-order steps 3 and 4), but it is undeclared, costs an unlisted extra catalog GET inside step 4,
and its recorded reason is false — which is what will propagate into the runner unit. **Suggested
fix:** decide the substitution on its merits and rewrite the docstring to the true reason; or accept
`wasResidentBefore` as a parameter so the runner can pass step 3's snapshot. This is a decision I am
handing over, not one I would make for you.

**P12-6 — the row-count identity's exemption is *again* wider than its docstring, and two sibling
branches still no-op silently. This is round three of the same shape.** `packs.py:392` states the
exemption as "**Skips (returns `[]`) only when `scripts` itself is absent**". The mechanism is
`isinstance(scripts, int) and not isinstance(scripts, bool)` (`:409-411`), so it also skips on
`"scripts": "12"` and `"scripts": true`. Separately, `:420-425` returns `[]` whenever
`replicatesPerScript` is absent or non-int. Executed against packs built from the plan's **own §3.3
manifest literal** (Appendix L.5): the literal validates clean (good); a copy with 48 rows against
`12 × 1` is correctly rejected; but the same 48-row pack with `replicatesPerScript` **deleted**, or
with `replicatesPerScript: "1"`, or with `scripts: "12"`, returns **0 problems** in all three cases.
One deleted key turns off the check §3.3 built to catch the N-1 shortcut. The `HISTORY.md` entry
states the exemption correctly ("absent **or not a plain int**"); the code docstring does not.
Because this is the third round of the same defect and my last finding was generated by the previous
fix's own predicate, **§4A gives a stopping condition instead of a fourth round** — read it before
acting on this finding.

**P12-7 — the "fully valid pack" positive control is a pack §3.3 rules out for its declared role.**
`tests/fixtures/packs/valid/pack.json` declares `"role": "tool-caller"` with
`"pairingKey": ["conversationId", "turnIndex"], "analysisUnit": "conversationId"`. §3.3: "the analysis
unit is the *outermost* component of `pairingKey` … **For the tool-caller that is `scriptId`, never a
conversation id**", and the plan's own manifest literal declares
`["scriptId", "replicate", "turnIndex"]`. The fixture satisfies the mechanised half of the rule
(`analysisUnit == pairingKey[0]`) and violates the stated half, and it is the anchor for
`test_validate_pack_accepts_a_fully_valid_pack` (`test_packs.py:182-184`) plus three further "accepts"
assertions, `data_path`, `load_tool_module`, `content_hash` and the totality boundary. Every sibling
tool-caller fixture uses `scriptId`; this one does not, and its 12-conversation rows file is what
makes the identity arithmetic come out clean. That is the brief's "fixtures written against the
implementation rather than against the plan's manifest shape", in the positive control.
**Suggested fix:** re-key the fixture to the plan's shape. Whether §3.3's *role*-specific half
("never a conversation id") should become a checkable rule in `validate_pack` is `architect`'s
call, not something I would have the implementer invent — it is listed as an open question in §6.

#### Minors

**P12-8 — the test that hid P12-1 names a class its assertion does not pin.**
`test_chat_call_with_dropped_connection_raises_lmstudio_call_failed_not_timeout`
(`tests/test_lmstudio.py:383-394`) stubs `urlopen` itself to raise, i.e. a drop during connect. A
connection dropped mid-body — the likelier event over a twenty-minute run on a 16 GB box, and the one
§3.6 was written for — is not covered and escapes. Fifth instance on this coordination of the
name-outruns-assertions class; **§4B's probe is the fix that closes it as a class rather than as an
instance.**

**P12-9 — `test_load_tool_module_imports_the_pack_module_via_importlib` does not pin "via importlib".**
Mutation run (Appendix L.6): replacing `spec_from_file_location` with a `sys.path.insert` +
`import_module` route leaves **29/29 `test_packs.py` green**. `packs.py:249` claims the distinction
("via `importlib`, not `sys.path`") and the plan §3.3 requires it; the `sys.path` route is the one
that leaks the pack directory into the global import path and can shadow a stdlib name. Pin it by
asserting the loaded module's `__spec__.origin` is the pack file and that `sys.path` is unchanged
across the call.

**P12-10 — `catalog()` parses the body before checking the status, so an HTTP error reports the wrong
cause.** `lmstudio.py:335-337`: `_parse_json` runs first, `if status != 200` second. Executed: an
HTTP 500 with an HTML body yields `GET /api/v0/models: response body is not valid JSON: Expecting
value…`, and the `HTTP {status}` branch is unreachable for exactly the bodies error responses carry.
§3.4.4a turns on the operator getting a distinguishing message. Swap the two.

**P12-11 — the unit boundary is not uniform, and `ChatResult`'s "never raises" is narrower than
stated.** `ttftMs`/`generationMs` coerce through `float()`; `tokensPerSecond` is
`stats.get("tokens_per_second")` verbatim (`:149`), so a string source survives as a `str` in a field
typed `float | None` and fails later, at aggregation, far from the boundary whose job is to settle the
unit. Separately, the class docstring says construction "**never raises** on a missing or partial
`stats`" unconditionally, but `ChatResult(stats=[1])` raises `AttributeError` — `chat()`'s
`isinstance(..., Mapping)` guard (`:439`) is what actually holds the rule, not the type. Both
executed (Appendix L.7).

**P12-12 — the AST walk permits a relative import the loader cannot execute.** `_tool_import_problems`
skips `node.level > 0` by design (`packs.py:526-527`, documented as "names the pack's own code"), but
`spec_from_file_location` gives the module no package, so `from . import helper` validates CLEAN and
then raises `ImportError: attempted relative import with no known parent package` at
`load_tool_module`. Either give the spec `submodule_search_locations` or report relative imports;
today validate says yes and the loader says no.

#### Nits

**P12-13** — this document's H1 and `Reviews:` field still say "S1 implementation review" / "§4 S1"
while Passes 12 onward gate S2. One-clause header fix, owner's call.

**P12-14** — `content_hash` on a non-existent or empty directory returns the empty-input SHA-256
rather than refusing, and both exclusion tests use `path.parts` on the **absolute** path
(`packs.py:284`, `:510`), so a pack root that itself lives under a `__pycache__` component would hash
to nothing. Neither is reachable today.

### 3. What's solid

- **Both defect classes the brief named are genuinely closed where they were found.** Mutation-run
  against the snapshot: deleting `check_tool_calling_eligibility`'s `if role != "tool-caller"` guard
  reddens exactly `test_gate_does_not_run_on_an_embedder_pack_so_the_run_proceeds` — **the §3.6 scope
  is pinned by something that would redden if removed** (the brief's question 5: yes). Restoring
  U73's pre-fix exemption reddens exactly its new test. Dropping the `__pycache__` exclusion reddens
  exactly `test_content_hash_excludes_pycache`.
- **The `__pycache__` exclusion is correct and its stated reason is true, not assumed.** Executed:
  `load_tool_module()` does write `tools/__pycache__/sim.cpython-312.pyc` into the pack directory, and
  the hash is unchanged across it. I looked for other loader-produced artifacts and found none — the
  import machinery's only in-pack side effect for a source module is the bytecode cache, a sourceless
  `.pyc` produces none, and `__pycache__/` is ignored repo-wide so no working-tree noise results.
  The exclusion is complete for loader artifacts; P12-2/P12-3 are about code the hash never covered in
  the first place, which is a different failure.
- **`validate_pack` runs all three advertised axes on the plan's own manifest literal, not just on the
  fixtures'** — built from the §3.3 literal verbatim it validates clean, and the two cases §3.3 names
  as the identity's reason to exist are both caught with the right message. The `check_sampling_contract`
  reuse, the §3.3 totality boundary on both halves, and the `derive_call_surface` factoring are all as
  the done-condition asks.
- **The 7-entry catalog fixture supports every assertion built on it.** I checked each: no test
  depends on a property only a 19-entry payload has — the count assertion is a fixture-shape assertion,
  `residency() == []` holds identically at 7, and all three eligibility cases have a real entry. The
  per-entry `_provenance` citations check out against `docs/reviews/small-model-benchmarking.md:889`,
  `:995`, `:1002-1008` and `:1436`, and the one inferred field (`qwen/qwen3-4b-2507`'s `tool_use`) is
  labelled as inferred in the fixture, in the test module docstring **and** in `HISTORY.md`.
- **`HISTORY.md`'s three entries are unusually honest** — U72's self-caught name-outruns-assertions
  test is recorded with the mechanism that hid it, and U73's entry states the exemption *more*
  accurately than the code docstring it describes.

### 4. Two stopping conditions, offered in place of a fourth round

The brief asks me to say when findings are being generated **by** the fixes rather than found **in**
the artifact. The answer splits, and the split matters:

- **The `validate_pack` exemption thread is converging by generation.** P12-6 exists because U73's
  own `declares_scripts` predicate created the branch it reports. That is round three of one shape and
  a fourth round is predictable. **Promote §4A above P12-6.**
- **The adapter thread is not.** P12-1, P12-3, P12-4, P12-5 are first-pass findings on code that has
  never been through a fix round. Treat them as ordinary findings.

**A — the pack-validation stopping condition (falsifiable).** The wave is done on this thread when a
**coverage probe** exists over the axes `_row_count_identity_problems` actually varies on, not over a
list of shapes: the four manifest keys the route reads — `sampling.scripts`,
`sampling.replicatesPerScript`, `sampling.analysisUnit`, `data.conversations` — crossed with three
value-kinds — *plan-valid*, *absent*, *present but not the declared type* — over a rows file that
**violates** the identity (48 rows against a `12 × 1` declaration), plus the all-valid control. Each
of the 13 cells must either report at least one problem, or appear in a module-level exemption
constant carrying a one-line reason, and **the probe asserts that the set of silent cells equals that
constant exactly** — so a stale exemption fails as loudly as a missing one. The axis list is
generated from a constant the route itself consults, so adding a fifth key without extending the
probe fails rather than passing.
**If it fails:** it names the cell, and the response is a *decision* — narrow the predicate, or list
the cell with a reason — never a discovery, so it terminates in one round instead of producing round
N+1. **If it cannot be made to fail on today's code** (i.e. it passes as first written), that is
itself the signal that it was written against the implementation rather than the plan, and it should
be rejected: it must go red on at least the three cells P12-6 names before the fix.
*This is a design I am recommending, not one I ran; the implementer should overrule it if the
exemption-constant mechanism reads worse than an explicit narrowing.*

**B — the adapter stopping condition (falsifiable, and it fails today).** A probe over
(operation) × (failure phase) × (failure kind): the six public operations
{`catalog`, `residency`, `probe`, `chat`, `embed`, `warm_up`} × {connect/headers, body-read} ×
{timeout, non-2xx, connection drop, unparseable body}. Every cell must land in exactly one of
`LMStudioCallTimeout` / `LMStudioCallFailed` / `LMStudioUnreachable`, or — for `probe()` — one of its
three literal values; **no cell may raise anything outside `LMStudioError`.** I ran the body-read row
today: **9 of 9 cells escape** (Appendix L.1). Done when it reads 0.
**If it fails:** the failing cell names both the operation and the phase, so the fix is local and the
probe is the regression net. This one I did run, so it is evidence rather than a proposal.

### 5. Residuals, classified per the standing rule (nothing rides as a follow-up)

- **Blocked on work that does not exist yet — acceptable:** (a) the real `GET /api/v0/models` capture.
  §4 S2's done-condition cites "§2.5's captured 19-model response" and no such payload exists in this
  repo; the third eligibility case's `capabilities: ["tool_use"]` is likewise inferred from
  `tests/conftest.py` rather than captured. Blocked on a **human-run live LM Studio session**, which
  is also what unblocks re-pointing `test_catalog_parses_every_fixture_entry_into_model_info`'s
  7-entry assertions. (b) §4 S2's **R-1 probe** — the loaded-model catalog re-read and the
  `loadedContextLength`-on-embeddings question — blocked on the same session, since an agent may not
  load a model. Neither is this wave's.
- **Deferred by choice — not acceptable, hence findings above:** the `replicatesPerScript`/`scripts`
  silent skips (`packs.py:400-401` explicitly scopes them out) are P12-6; the read-phase taxonomy gap
  is P12-1; the wall-clock window is P12-4. None of these is blocked on anything unbuilt.

### 6. Open questions

1. **Should §3.3's role-specific half be mechanised?** The rule "for the tool-caller the analysis unit
   is `scriptId`, never a conversation id" is checkable only against a role→pairingKey table the plan
   does not currently define. P12-7 fixes the fixture; whether `validate_pack` should also refuse a
   `tool-caller` whose `pairingKey[0]` is not `scriptId` is `architect`'s, not the implementer's.
2. **Is `warm_up`'s residency substitution (P12-5) intended?** If yes, §3.6's `coldLoadSeconds`
   sentence names a source the adapter no longer offers, and the plan should say so; if no, the
   adapter should take the snapshot as a parameter.

### Appendix L — Pass 12 evidence

**L.0 — isolation.** `git archive 3924f3a model-bench | tar -x -C <scratch>`; a `sitecustomize.py` on
`PYTHONPATH` strips the setuptools editable meta-path finder so `modelbench.__file__` resolves inside
the snapshot (verified by printing it). Every result below was produced there. The working tree was
never read, written, added, committed or checked out.

**L.1 — read-phase escapes (P12-1).** A fake opener returning a response whose `read()` raises, for
three exception kinds × three operations:

```
IncompleteRead (body cut short)    chat  -> ESCAPED IncompleteRead: IncompleteRead(7 bytes read)
IncompleteRead (body cut short)    catalog -> ESCAPED IncompleteRead
IncompleteRead (body cut short)    probe -> ESCAPED IncompleteRead
ConnectionResetError mid-body      chat/catalog/probe -> ESCAPED ConnectionResetError [Errno 104]
socket timeout during read         chat/catalog/probe -> ESCAPED TimeoutError: timed out
```

**L.2 — `tools.module` traversal (P12-2).** Manifest `"tools": {"module": "../outside.py"}`:

```
validate_pack -> CLEAN
contentHash unchanged after editing the executed module: True
load_tool_module -> 'outside CHANGED'
```

**L.3 — allowlist probes (P12-3).**

```
1 __import__('modelbench.results')      validate CLEAN;  import reached modelbench.results
2 importlib.import_module('...report')  validate CLEAN;  import reached modelbench.report
3 import modelbench.stats               validate REFUSED  (the mechanism works where it looks)
4 tools/sim.pyc as tools.module         validate CLEAN;  bytecode EXECUTED unscanned
5 from . import helper                  validate CLEAN;  load_tool_module -> ImportError
```

**L.4 — wall-clock window (P12-4).** Stub whose `read()` sleeps 250 ms:
`ChatResult.wallClockMs = 0.001 ms`.

**L.5 — `validate_pack` against the plan's own §3.3 manifest literal (P12-6).**

| Pack built from the plan's literal | problems |
|---|---|
| as written, 12 rows keyed on `scriptId` | **0** (correct) |
| 48 rows against `12 × 1` | 2 — row count and per-unit count |
| `analysisUnit: conversationId`, `replicatesPerScript: 4`, 48 rows | 2 — Rule 6 and 48-distinct |
| 48 rows, `replicatesPerScript` **deleted** | **0** ← silent |
| 48 rows, `replicatesPerScript: "1"` | **0** ← silent |
| 48 rows, `scripts: "12"` | **0** ← silent |
| rows carrying no `analysisUnit` key at all | 1 |

**L.6 — mutations (all against a copy of the snapshot, never the working tree).**

| Mutation | Result |
|---|---|
| delete `if role != "tool-caller": return` | 1 failed — the embedder negative test. Scope **is** pinned |
| restore U73's pre-fix exemption | 1 failed — `..._missing_data_conversations`. Fix **is** pinned |
| drop the `__pycache__` exclusion | 1 failed — `test_content_hash_excludes_pycache` |
| `spec_from_file_location` → `sys.path` + `import_module` | **29/29 still green** (P12-9) |

**L.7 — `ChatResult` boundary shapes (P12-11).** `tokens_per_second: "51.4"` → `tps='51.4'` (`str`);
`time_to_first_token: "0.111"` → `111.0`; `: True` → `1000.0`; `stats` as a list via `chat()` → all
`None` (guarded); `ChatResult(stats=[1])` directly → `AttributeError`.

**L.8 — disposition of Pass 11's two findings.** **M11-1: fixed** — plan v1.24 rewrites §4 S1e Table
H's `stats.py:444` row to name `lo`/`hi` from the clamp and derive `bound_by` from `lo != u_lo` /
`hi != u_hi`; residuals 20/21 unchanged at 1 (plan changelog line 5, row at `:4133`). **M11-2: fixed**
— `modelbench/stats.py:468-472` now reads "`lo != u_lo` iff the composed lower bound ran strictly
below the support", with the `max`/`min` identity spelled out. No Pass 1–10 finding is re-raised here;
all were dispositioned in their own passes and none of them touches S2 surface.

---

## Pass 13 — 2026-09-09

### 1. Scope & verdict

**Reviewed:** four commits as immutable git objects — `dd40ede` (U74, `hostinfo.py` + the `attest`
CLI command; a **first** gate, never reviewed), `fed4e21` (U76, the `packs.py` fix round),
`17c6eb0` (U75, the `lmstudio.py` fix round) and `7f0006b` (U77, `ChatResult`). Everything was run
at the tip, `7f0006b`. **Baseline:** Pass 12's fourteen findings; plan §3.3, §3.4.4, §3.4.4a, §3.5,
§3.6, §3.6a and §4 S2; `-ml` §3.4 Rule 6 and §11.5.1. **Out of scope and untouched:**
`runner`/`convo`/`tooling`, the `validate`/`run` CLI, every S1 surface.

**Method.** `git archive 7f0006b model-bench | tar -x` into a scratch tree, with a
`sitecustomize.py` on `PYTHONPATH` stripping the setuptools editable meta-path finder so
`import modelbench` resolves inside the snapshot (verified by printing `modelbench.__file__`). All
17 mutations and all 6 probes below ran there; the working tree was never read, written or moved.
At the tip: **815 passed, 3 deselected**, `ruff check .` clean.

**Verdict: needs changes.** — **1 blocker, 3 majors, 5 minors, 4 nits.** The three fix rounds are
real: **17 of 17 mutations against their new mechanisms are killed** (§3), including every one of
U74's, which is a strong first gate. Every finding below sits at the **edge of a guard the fix round
itself installed**, which is the brief's question answered: the mechanisms are right and three of the
four *declared reaches* are not.

**CPG: considered, not relevant — no Code Property Graph is loaded for `model-bench` (only
`cpg_falkorchat` and `cpg_deprecated_salesperson` exist on this instance), so every claim below
comes from reading the four commits and executing against the snapshot.**

**On the brief's five pre-verified premises — three hold, two are incomplete, and the two are
findings.** (a) The §4A trap does fire on the mutation you ran, but the probe's silence criterion is
`validate_pack(pack) == []`, not the route's own output, so it does **not** hold the exemption in
place in general (P13-2, proven both ways). (b) The §4B probe does read 0 **on the cells its grid
spans**; the error-body read is a cell the grid cannot express and 21 of 21 escape there (P13-1).
(c) `tools.module` containment/suffix, (d) U74's four trip-wire outcomes and (e) U77's totality all
reproduce exactly as you found them — but U74's 13 structural mutations are structural only, and the
gap is a **tier** gap they cannot reach (P13-4).

### 2. Findings

#### Blocker

**P13-1 — the P12-1 fix moved the *success* body read into the ladder and left the *error* body read
outside it; 21 of 21 cells escape, and `_EXEMPT_CELLS` declares the cell unreachable.**
`_raw_get:336-340` and `_raw_post:391-397` read the error body **inside an `except` clause**
(`exc.read()`), where no later rung of the same `try` can catch it. Executed against the snapshot
(Appendix M.1): an `HTTPError` whose `read()` raises `IncompleteRead`, `ConnectionResetError` or a
read-phase `TimeoutError` escapes `catalog`, `residency`, `probe`, `chat`, `embed` and both
`warm_up` surfaces untyped — **21 of 21**, and it reaches the **shipped `attest` command** as a
traceback (M.4/D). This is not exotic traffic: `probe()`'s own `v1-only` diagnosis — §3.4.4a's second
distinguishing message — runs `exc.read()` on a 404 from `/api/v0/models` on **every** non-LM-Studio
server, so the error arm is normal-path code. `_raw_get`'s new docstring claims "`None` on **any**
connect- or read-phase failure"; `_EXEMPT_CELLS`'s comment claims `("read", "non_2xx")` is
structurally unreachable because "`urlopen()` raises `HTTPError` … *before* ever handing back a
response object". `HTTPError` **is** a response object, with a status and a `.read()`.
*Suggested fix (judge it):* wrap both `exc.read()` bodies in their own `try`, degrading to `None`
(GET) / `LMStudioCallFailed` (POST) — the status is already in hand, so the message survives the
lost body. *The assertion that catches me being wrong* goes in the §4B grid, **not** in the fixed
code: add a third phase value `error-body` to the `(phase, kind)` axis with `non_2xx` as its only
kind, delete `("read", "non_2xx")` from `_EXEMPT_CELLS`, and let the existing per-operation
parametrization cover it. Pass 12 said "done when it reads 0"; it reads 0 because the grid stops one
axis short.

#### Majors

**P13-2 — the §4A coverage probe measures `validate_pack`'s silence, not the row-count route's, so a
fourth round of P12-6's exact shape passes green.** `tests/test_packs.py:340` is
`if validate_pack(pack) == []`, while `ROW_COUNT_IDENTITY_EXEMPT_CELLS` is documented as "the one
case **this route** is sanctioned to skip silently" (`packs.py:412`). The two quantities differ on
two of the twelve cells today — `sampling.analysisUnit` absent and wrong-type report 1 problem from
the route and 2 from the validator, the extra one coming from `check_sampling_contract` (Appendix
M.2). Executed: re-inserting a P12-6-shaped silent branch (`if "analysisUnit" not in sampling:
return []` at the head of `_row_count_identity_problems`) leaves **`test_packs.py` at 35 passed** —
the probe stays green because a sibling axis catches the same manifest. The constant itself is
currently honest: computed against the route directly, the silent set **is** exactly
`{("sampling.scripts", "absent")}`. It is the guard holding it that is too wide. Part of this is my
own §4A wording, which said "must either report at least one problem" without naming *which
function* reports.
*Suggested fix, run both ways:* change line 340 to
`_row_count_identity_problems(pack, pack.manifest.get("sampling") or {}) == []`. Applied to the tip
it is **35 passed**; applied to the mutated tree it is **1 failed, 34 passed** — so it is green where
it should be and red where it must be.

**P13-3 — U77's coercion is wider than its claim: a non-finite or boolean `stats` value lands a
*number*, and the docstring promises `None`.** `ChatResult`'s docstring says a "string- or otherwise
wrong-typed source lands `None` rather than surviving untyped"; `_as_float`/`_seconds_to_ms`
(`lmstudio.py:203-224`) accept anything `float()` accepts. Executed end to end through `chat()`
(Appendix M.3), against a body carrying bare `NaN`/`Infinity` — which **Python's `json.loads`
parses by default**, so this needs no malformed transport, only a server that serialises a 0/0 rate:
`ttftMs=nan`, `generationMs=inf`, `tokensPerSecond=nan`. Consequences are all silent: those items
are counted in `statsCoveredCount`; `statistics.median` over a list containing `nan` returns an
arbitrary element with no error; `unexplainedMsMax` becomes `inf`; and §11.5.1's gap
`latencyMs − (ttftMs + generationMs)` becomes `-inf`, so **the in-call reload detector can never
fire on that item**. Separately `True` → `ttftMs=1000.0`, `tokensPerSecond=1.0`, while this same
component excludes `bool` from `int` deliberately in two other places
(`packs._row_count_identity_field_valid:437`, and `results`' bool guard added at Pass 2 P2-2).
*Suggested fix:* in both helpers, `return None` unless `math.isfinite(v)`, and reject `bool` before
coercing. *The assertion:* `ChatResult(stats={"tokens_per_second": float("nan")}).tokensPerSecond is
None`, plus one through `chat()` over a body containing a literal `NaN`.

**P13-4 — `validate_host_info` accepts three attested values the fingerprint's own tier will refuse,
so `attest` can write a `host.json` that kills a run twenty minutes later.** `lmStudioAppVersion`,
`kvCacheSetting` and `hostRamGb` are `_NONEMPTY` in `fingerprint.py:110-112`; `validate_host_info`
checks only presence and non-`null` (`hostinfo.py:113-121`). Executed: an `attested` block of
`{"lmStudioAppVersion": "", "kvCacheSetting": "", "hostRamGb": 0, "otherResidentWorkloads": []}`
returns **`[]`** — clean (M.4/A). `model-bench attest --set kvCacheSetting=` therefore writes a file
that passes capture-order step 1 and is refused by `store()` at step 10 (§3.4.5 point 1), after the
whole run. §3.4.4a's capture order exists precisely to make refusals cheap; this one is as
expensive as refusals get, and the cheap end is one tier lookup away. `otherResidentWorkloads` is
correctly exempt — it is `_PRESENT`, and `test_attest_other_resident_workloads_empty_string_is_an_
empty_list` pins `""` → `[]` deliberately.
*Suggested fix:* require the three to be non-empty (and `hostRamGb > 0`) in `validate_host_info`,
i.e. at attest time, **not** in `store()`, which is the step it is meant to catch.
*The assertion:* a test asserting the two tables agree —
`{n for n in ATTESTED_FIELD_NAMES if _NONEMPTY-tier in REQUIRED_BY_SCHEMA[1]["model"]}` is exactly
the set `validate_host_info` refuses empty — so a tier change on either side reddens rather than
drifting.

#### Minors

**P13-5 — `_cmd_attest` catches one of the two exceptions `hostinfo.attest` documents.**
`cli.py:_cmd_attest` handles `AttestProbeFailed` only; `attest()` also raises `HostInfoError` on its
own defensive re-validation. Executed: `model-bench attest --api-base-url ""` (with everything else
supplied) → uncaught `HostInfoError` traceback, exit `1` — outside §3.6a's closed set (M.4/B).
*Fix:* catch `HostInfoError` and return `EXIT_USAGE`.

**P13-6 — `attest` raises `EOFError` when a field is unset and stdin is not a terminal.**
`_gather_attested_fields` calls `input()` unguarded. Executed: `attest --set lmStudioAppVersion=…`
with empty stdin → `EOFError` traceback (M.4/C). §3.6a offers `--set` *as* the non-interactive
route, so a partly-specified non-interactive invocation is normal usage, not abuse. *Fix:* catch
`EOFError` and exit `2` naming the fields still unset.

**P13-7 — `residency_source` is a required parameter `check_attestation_staleness` reads on one of
its three paths, and the plan names it on two.** On `call_surface == "embeddings"` the function
returns `"unavailable"` unconditionally without comparing anything, while §3.4.5 says "On a
`model:embeddings` arm the check **degenerates to `residencySource` alone**"; on
`"first-observation"` the attested `residencySource` **is** present (`attest` always writes it) and
is comparable, but is not compared. Both are inert today because `_residency_source_for_probe`
returns a constant — and §3.4.4a names `residencySource` as the field that earns its keep "once
there is more than one answer", which is exactly when the hole opens. *Fix:* compare
`residencySource` on both paths (stored outcome unchanged); or drop the parameter from the
embeddings path. **Which the plan intends is a genuine ambiguity — see §5 OQ-1.**

**P13-8 — the one write `run` makes outside `results/` has no validation on either side.**
`write_host_info` "never validates" by contract, and `check_attestation_staleness` does not validate
the `updated_host` it hands back. Executed: a `host` with no `observedAtAttestation` yields a
back-fill whose `validate_host_info` returns
`['observedAtAttestation.residencySource: absent or empty']` (M.4/E) — a file that, once written,
makes every later run exit `5` until the operator re-attests. Unreachable through `read_host_info`,
reachable by any caller that skips it. *Fix:* have `check_attestation_staleness` raise
`HostInfoError` when `host` carries no `observedAtAttestation` — a precondition on its **caller's**
step 1, not a guard inside the step it protects.

**P13-9 — the plan's §4 S2 `warm_up` signature was not swept, and U75's "No plan correction needed"
is falsified by it.** On the substance the change is right: `runner.py` does not exist, nothing
outside `tests/test_lmstudio.py` calls `warm_up` (grepped repo-wide, all hits are docs or that file),
Appendix A already declares `LoadResult.wasResidentBefore`, and §3.6 already names
`residentModelsAtStart` as `coldLoadSeconds`' source — so the required no-default parameter serves
§3.4.4a step 3's snapshot at the call site the runner will become. But plan §4 S2's own code block
(`docs/plans/small-model-benchmarking.md:4264-4265`) still spells
`warm_up(self, model, *, call_surface, system_prompt, timeout_s)`, with no
`was_resident_before`. *Fix (→ `architect`, one line):* add the parameter to that block. Note the
adapter cannot check the value — `warm_up` returns the caller's own input verbatim — so the
assertion that it comes from **step 3** and not step 7 is owed by the runner unit's tests, and
belongs in that unit's brief.

#### Nits

**P13-10** — `_residency_source_for_probe(probe_result)` (`hostinfo.py:170`) ignores its only
parameter and returns a constant. Either drop the parameter or make the mapping real; as written the
signature promises a decision the body does not make.

**P13-11** — `"tools": {"module": ""}` validates CLEAN (`_tool_module_problems` returns `[]` on
falsy) and then `load_tool_module` raises `"tools.module is absent from the manifest"` — present-and-
empty reported as absent, in the one component whose thesis is that the two are different.

**P13-12** — `validate_host_info` accepts unknown keys inside `attested`, while §3.4.4 says the block
"is exactly the four FR-7 fields". `_parse_set_flags` already closes the CLI route, so this is only
reachable by hand-editing.

**P13-13** — P12-13 is unfixed and has grown: the H1 still says "S1 implementation review", the
`Reviews:` field still says "§4 S1", and the preamble's "jump to `## Pass 11` for the current
verdict" pointer is now two passes stale. Left alone deliberately — this pass's brief scopes my write
to the Pass 13 section. Owner's call, one edit for all three.

### 3. Disposition of Pass 12's fourteen findings

Seventeen mutations, **17 killed, 0 survived** — nine against the fix round's new mechanisms and
eight against U74's. Full table in Appendix M.5.

| # | Disposition | Evidence I rechecked |
|---|---|---|
| **P12-1** read-phase escapes | **Fixed on the success path; open on the error path** → **P13-1** | Moving `resp.read()` back outside the POST ladder reddens **25 tests**. §4B reads 0 over its grid. The `exc.read()` arm escapes 21/21. |
| **P12-2** `tools.module` traversal | **Fixed** | Containment check removed → 1 failed. Refused for `../`, absolute and dot-segment paths; in-root control silent. |
| **P12-3(i)** `.pyc` loads unscanned | **Fixed** | `.py`-suffix check removed → 1 failed. |
| **P12-3(ii)** coupling, not containment | **Accepted as by-design** | `load_tool_module`'s docstring now states it outright. |
| **P12-3(iii)** docstring overclaims safety | **Fixed** | "a coupling rule … never one whose code is safe to run untrusted" (`packs.py:249-256`). |
| **P12-4** wall clock stops at headers | **Fixed** | Clock now below `resp.read()`; reverting it → 1 failed (`_SlowReadResponse` pins ≥ the delay). |
| **P12-5** inverted `warm_up` justification | **Fixed, against the plan** | Docstring now states the true reason; `test_warm_up_never_probes_residency_itself` stubs no catalog route, so a re-probe reddens. Residue: **P13-9**. |
| **P12-6** exemption wider than its docstring | **Fixed in the route; the guard holding it is not** → **P13-2** | Route-level silent set computed by execution **is** exactly the constant. The probe measuring it is not the route. |
| **P12-7** positive control violates §3.3 | **Fixed** | `fixtures/packs/valid/pack.json` is now `pairingKey: ["scriptId","turnIndex"]`, `analysisUnit: "scriptId"`, `scripts: 12 × 1`, 12 rows. |
| **P12-8** test names a class it does not pin | **Fixed** | Subsumed by §4B's grid — 48 parametrized cells across six operations. |
| **P12-9** `importlib`-not-`sys.path` unpinned | **Not fixed** | No `__spec__`/`sys.path` assertion anywhere in `tests/test_packs.py` (grepped). Out of this round's declared scope; still stands. |
| **P12-10** parse before status | **Fixed** | Order swapped; reverting → 1 failed. |
| **P12-11** `ChatResult` boundary | **Fixed as to totality and type; the coercion's *reach* is not** → **P13-3** | `_as_float` → raw passthrough reddens 2; dropping the `Mapping` guard reddens 1. `stats` as list/str/int/`None` all construct. |
| **P12-12** relative import validates then fails | **Not fixed** | Still skipped at `packs.py:645`, now with a rationale at `:624`. The validate-yes/load-no divergence remains. |
| **P12-13** header says S1 | **Not fixed** → **P13-13** | Line 1 unchanged. |
| **P12-14** `content_hash` on an empty dir / abs `parts` | **Not fixed** | `packs.py:289,292` unchanged. Still unreachable. |

**Both Pass 12 residuals classified as blocked remain blocked and are not re-litigated here:** the
real `GET /api/v0/models` capture and §4 S2's R-1 probe both wait on a human-run live LM Studio
session, and no work in these four commits could have moved either.

### 4. What's solid

- **U74 is the best-gated unit in this wave, and it is a first gate.** All eight mutations I aimed at
  its mechanisms are killed, including the two that matter most: `attest` writing `runtimeName` into
  `observedAtAttestation` (plan-gate P4-6's exact breach) reddens 2, and the back-fill touching
  `attestedAt` reddens 1. `check_attestation_staleness` is **correct and complete against §3.4.5's
  four behavioural outcomes** — first-observation back-fills exactly the three runtime keys and
  leaves `attested`/`attestedAt` byte-identical, compared-equal proceeds, compared-mismatch carries
  `STALE_MESSAGE`, embeddings reports `"unavailable"` — and it does not mutate its input. Shipping it
  unwired is the right call and carries **no risk to the runner unit**: it is a pure function with a
  frozen return type, its `updated_host` back-fill contract is documented at the type, and the one
  hazard is P13-8's unvalidated write, which is a one-line precondition.
- **`_raw_post`'s ladder is genuinely well-built where it reaches.** Naming `TimeoutError` as its own
  rung (not a `URLError`), unwrapping `URLError.reason` for a wrapped timeout, and giving
  `http.client.HTTPException` a rung above `OSError` because `IncompleteRead` is not an `OSError` are
  each the right call for the right stated reason. §3.6's two dispositions — censored vs missing —
  are decided at the one layer that can tell them apart.
- **The three exemption mechanisms are each honest about their *contents*.** Widening
  `_EXEMPT_CELLS` by one cell reddens; widening `ROW_COUNT_IDENTITY_EXEMPT_CELLS` by one reddens;
  `test_row_count_identity_exempt_cells_each_carry_a_reason` refuses an entry with an empty reason.
  Every problem I found is about a guard's *reach*, never about a stale constant nobody would notice.
- **`ROW_COUNT_IDENTITY_KEYS` is the right shape and the round-three response is proportionate.**
  One constant consulted by both the route and its probe, a table-driven route replacing four
  hand-written clauses, and exactly one sanctioned silent cell with its reason at the constant. Fix
  P13-2's measurement and this thread is closed by construction rather than by vigilance.
- **`fed4e21`'s and `7f0006b`'s `HISTORY.md` entries record their own mutation runs**, including
  U76's honest note that its red-before-fix was an `ImportError` and would not have satisfied the
  reviewer's criterion, and U75's note that a first version of
  `test_warm_up_passes_was_resident_before_through_verbatim` covered only the chat branch and was
  caught by mutation before review. That is the class being caught upstream of me, which is the point.

### 5. Open questions

1. **(→ `architect`) Does "the check degenerates to `residencySource` alone" (§3.4.5) mean *compare
   it* or *give up*?** The code reads it as give-up. Both readings fit the prose, they differ only
   once a second provider exists, and the stored outcome is `"unavailable"` either way. One clause
   settles P13-7.
2. **Is `LoadResult.wasResidentBefore` worth its parameter?** It is a verbatim echo of the caller's
   own input, so it adds no fact to the record and the adapter cannot check it. Keeping it is
   defensible (it makes the `coldLoadSeconds` precondition explicit at the type); dropping it and
   letting the runner derive `coldLoadSeconds` from its own step-3 snapshot is simpler. Not a finding
   either way — a shape call for whoever briefs the runner unit.

### 6. The wave: which findings are generated *by* the fix round, and a stopping condition

**The honest split.** P13-4 is found *in* U74, a first gate — an ordinary finding. P13-1's escaping
code **predates** U75 (the `exc.read()` arm is unchanged in `17c6eb0`'s diff), but its *false
all-clear* is U75's. P13-2 and P13-3 are generated **by** the fix round outright, and P13-2 is partly
generated by **my own §4A wording**, which never said whose silence to measure. Two and a half of
four. So yes — this pass is now finding defects in the instruments rather than in the system, and
that is the signal to stop discovering and start auditing.

**What recurs is one thing, and it is not the guards.** Six instances now (P12-6, P12-11, P13-1's
`_EXEMPT_CELLS`, P13-2, P13-3, P13-4): **a guard's reach is stated in prose and its mechanism in
code, and nothing executable compares the two.** Narrowing predicates one at a time has cost four
rounds. A seventh probe would be the same move again.

**C — the reach audit (falsifiable, one round, closed inventory).** Enumerate every module-level
*set-shaped* guard constant in `modelbench/` plus the two in `tests/` — the inventory is closed and I
sized it: `grep -nE '^_?[A-Z][A-Z0-9_]* *(:[^=]*)?= *[({[]' modelbench/*.py` returns **35**, of which
**~22 are set-shaped** (the rest are report message strings) and the S1 half is already pinned by
literals from Pass 2's M-4 fix, leaving **~7 in S2**. For each, either **(i)** name the one test that
computes that set **by executing the specific function that consults it** — not a superset
validator — and asserts equality, or **(ii)** file it. Done when every entry has an (i) or an (ii).

**If it fails** — if **five or more** entries come back needing (ii) — the response is **not** five
fixes. At that count the recurrence belongs to the convention, not to any author, and the ruling is a
convention change stated once in `model-bench/AGENTS.md`: *a reach claim about a guard lives in an
asserted constant or it does not get written.* Prose reach claims already in the tree are then
deleted rather than defended, and the seventh instance cannot be authored.

**If it passes** — four or fewer (ii)s — they are ordinary findings, fixed in one round, and this
thread closes without a Pass 14 dedicated to it. Either way the audit terminates in one round,
because it is an enumeration over a list that already exists rather than a search.

---

### Appendix M — Pass 13 evidence

**M.0 — isolation.** `git archive 7f0006b model-bench | tar -x -C <scratch>/p13/tip`; a fresh
`sitecustomize.py` on `PYTHONPATH` filters any meta-path finder whose type name or module contains
`editable`. Confirmed by
`modelbench.__file__ == <scratch>/p13/tip/model-bench/modelbench/__init__.py`. A pre-existing scratch
snapshot from Pass 12 was discarded rather than reused (it carried an untracked `sitecustomize.py`
and `__pycache__`, neither of which is in `git ls-tree 7f0006b`). Suite at the tip: 815 passed, 3
deselected; ruff clean.

**M.1 — P13-1, the error-body read.** A fake opener raising an `HTTPError` subclass whose `read()`
raises, over 3 exception kinds × 7 operation/surface cells:

```
IncompleteRead   catalog residency probe chat embed warm_up(chat) warm_up(emb)  -> 7/7 ESCAPED
ConnectionReset  (same seven)                                                   -> 7/7 ESCAPED
ReadTimeout      (same seven)                                                   -> 7/7 ESCAPED
escaped 21 of 21
```

**M.2 — P13-2, route silence vs validator silence.** `_row_count_identity_problems` driven directly
against the same 12 cells the probe builds, beside `validate_pack`:

| cell | route problems | validate_pack problems |
|---|---|---|
| `sampling.scripts` valid / absent / wrong-type | 2 / **0** / 1 | 2 / **0** / 1 |
| `sampling.replicatesPerScript` valid / absent / wrong-type | 2 / 1 / 1 | 2 / 1 / 1 |
| `sampling.analysisUnit` valid / absent / wrong-type | 2 / **1** / **1** | 2 / **2** / **2** |
| `data.conversations` valid / absent / wrong-type | 2 / 1 / 1 | 2 / 1 / 1 |

Route-level silent set = validator-level silent set = `ROW_COUNT_IDENTITY_EXEMPT_CELLS` =
`{("sampling.scripts","absent")}` **today**; the `analysisUnit` rows are where the slack lives.
Mutation `if "analysisUnit" not in sampling: return []` at the head of the route →
`test_packs.py` **35 passed**. Same mutation with line 340 pointed at the route → **1 failed, 34
passed**; the unmutated tip with the same change → **35 passed**.

**M.3 — P13-3, non-finite through `chat()`.** Body
`{"stats":{"time_to_first_token": NaN, "generation_time": Infinity, "tokens_per_second": NaN}, …}`
— parsed by `json.loads` with no error:

```
through chat():  ttftMs=nan  generationMs=inf  tokensPerSecond=nan
statistics.median([nan,10,20,30,40] sorted) -> 20.0   (silent, arbitrary)
direct construction, stats={"...": True}    -> ttftMs=1000.0  generationMs=1000.0  tps=1.0
```

**M.4 — U74 probes.**

```
A  attested {"lmStudioAppVersion":"", "kvCacheSetting":"", "hostRamGb":0, ...}
                                       -> validate_host_info() == []   (P13-4)
B  attest --api-base-url ""            -> UNCAUGHT HostInfoError, exit 1      (P13-5)
C  attest with a field unset, no stdin -> UNCAUGHT EOFError                   (P13-6)
D  probe() raising IncompleteRead      -> UNCAUGHT IncompleteRead at the CLI  (P13-1)
E  host without observedAtAttestation  -> updated_host fails validate_host_info (P13-8)
F  four outcomes: first-observation / compared+equal / compared+stale / unavailable — all correct;
   back-fill writes exactly runtimeName, runtimeVersion, runtimeObservedAt; `attested` is the same
   object and `attestedAt` is unchanged; the input `host` is not mutated.
```

**M.5 — 17 mutations, 17 killed** (all against copies of the snapshot; the working tree was never
touched). Fix-round mechanisms (9): `_EXEMPT_CELLS` widened → 1 failed · `ROW_COUNT_IDENTITY_
EXEMPT_CELLS` widened → 1 · `tools.module` containment removed → 1 · `.py` suffix removed → 1 ·
wall clock back before `read()` → 1 · `catalog` parse-before-status → 1 · `_as_float` raw
passthrough → 2 · `__post_init__` `Mapping` guard removed → 1 · POST body read moved outside the
ladder → **25**. U74 (8): `residencySource` non-empty check dropped → 2 · `attest` writes
`runtimeName` → 2 · embeddings returns `"compared"` → 1 · `residencySource` dropped from the stale
comparison → 1 · back-fill rewrites `attestedAt` → 1 · `attest` exits `0` when unreachable → 2 ·
`hostRamGb` bool accepted → 1 · `read_host_info` skips validation → 1.

---

## Pass 14 — 2026-09-09

### 1. Scope & verdict

**Reviewed:** the §6 reach audit promised in Pass 13, run at `b5f719b` — plus, subsumed into it,
the three fix commits `2f64ea2` (U79: P13-1, P13-3), `b46708e` (U80: P13-2) and `39a2748` (U81:
P13-4/5/6/8/10/12), which had never been gated on their own. Suite at that snapshot: **834 passed,
3 deselected**, ruff clean.

**Scope ruling — the audit subsumes the diff re-gate; no separate Pass 14b is needed.** Three
reasons, and I would not have ruled this way on any of them alone. (a) The audit's inventory
**covers all three constants the diffs changed** — `_EXEMPT_CELLS`, `ROW_COUNT_IDENTITY_EXEMPT_
CELLS`, `ATTESTED_NONEMPTY_FIELD_NAMES` — which is why it waited for them. (b) Every Pass 13
finding was re-driven by **its own original probe** against the new snapshot, not read off the
diff (§3). (c) The diffs' non-constant surface is small and I executed all of it: the
`_coerce_finite_float` split, both `exc.read()` inner guards, `_gather_attested_fields`' EOF path,
`_cmd_attest`'s catch surface and the P13-8 precondition. What a constant-shaped audit would
*not* have covered, I covered by hand; there is nothing left for a diff pass to find.

**Verdict: needs changes** — but **exactly one required change, and it is not a code fix.** The
three commits are **approve**: all six findings they targeted are closed, verified by execution,
and nothing in them regressed. The audit, however, **fires its pre-stated failure branch** — five
guard constants, not four — so per Pass 13 §6 the answer is the convention line in §4, routed to
the human, *not* five find-and-fix rounds. I am holding myself to that branch.

**CPG: considered, not relevant — no Code Property Graph is loaded for `model-bench` (only
`cpg_falkorchat` and `cpg_deprecated_salesperson`), so the inventory was built by `grep` over
`modelbench/*.py` and every judgement below is a mutation actually run.**

**A correction I owe on Pass 13's own method.** Pass 13 Appendix M.0 claimed every result was
produced inside the snapshot. That is true of all 17 mutations (they ran under `pytest`, whose
rootdir insertion isolates correctly) and of the `-c` checks, but **false of the five standalone
probe scripts**: `python <script.py>` puts the *script's* directory on `sys.path[0]`, not the cwd,
so `modelbench` resolved through the editable install's `.pth` to the **working tree**. Stripping
the meta-path finder in `sitecustomize.py` is not sufficient on its own — the `.pth` injects the
path too. **Pass 13's findings are unaffected and I verified why rather than assuming it:**
`git diff 7f0006b a1e2234 -- model-bench/modelbench/ model-bench/tests/` is empty, so the tree
those probes hit was byte-identical to the reviewed commit. The house method needs one addition:
**put the snapshot first on `PYTHONPATH` and assert `modelbench.__file__` inside the probe**, which
is what every Pass 14 probe does (Appendix N.0).

**On the coordinator's three self-checks: two right, one right for a reason worth stating.**
(1) `("connect", "unparseable_body")`'s reason **is** true, not merely asserted — "unparseable
body" is a JSON-level fact and JSON needs bytes, which need a read; there is no connect-phase
referent. See the caveat in P14-6. (2) The `_PHASE_KINDS` / `_CONNECT_KINDS` reading is **correct**
and is the audit's fifth gap — confirmed by mutation, with the reverse direction as a control
(P14-5). (3) The P13-4 coupling holds: flipping `otherResidentWorkloads`' tier in `fingerprint.py`
alone, with zero `hostinfo.py` edits, reddens **71 tests**.

### 2. The audit

**Method.** For each constant, the question "does anything executable hold this constant's reach?"
is answered the only way it can be falsified: **shrink it by one member and run the suite.** A
constant whose reach is genuinely held reddens. One that is merely *parametrized over itself*
loses a test case and stays green — the M-4 shape from Pass 1. Three dispositions, decided before
running: **(i) held** — a shrink reddens; **(d) derived** — the constant is computed from another
source (`sys.stdlib_module_names`, a set difference, a `REQUIRED_BY_SCHEMA` key split) so
transcription drift cannot arise, and I verified each derivation actually tracks by mutating its
*source*; **(ii) gap** — a shrink is green.

**Inventory, closed and enumerated.** Pass 13 sized it at "35 constants, ~22 set-shaped, ~7
unpinned in S2". Re-run at `b5f719b` the grep returns 29, plus 7 more whose
`frozenset(...)`-comprehension form that pattern missed — 36. Of those, **24 are set-shaped**
under the criterion I stated (*a constant whose membership a guard consults to decide behaviour*,
as against a message string or a scalar bound), which excludes report.py's ten prose constants and
`stats.SUPPORT_DIFF_PROPORTIONS`. The §4B grid's four interlocking test-side registries count as
**one** entry, since they fail or hold together. **Total judged: 25.** Full table in Appendix N.1.

| Disposition | Count |
|---|---|
| **(i) held** — a one-member shrink reddens | **14** |
| **(d) derived** — drift structurally impossible, derivation verified by mutating its source | **6** |
| **(ii) gap** — a one-member shrink is green | **5** |

**The count is 5. The pre-stated failure branch fires.** I want to be plain that I did not go
looking for a fifth: `ROLES` was my expected fifth and it turned out **not** to be a guard
constant at all — nothing consults it, so a shrink only changes an error message's wording, and I
reclassified it out of the inventory rather than bank it. The fifth is `UNIT_KIND_BY_ROLE`, the
constant `ROLES` merely describes.

### 3. Disposition of Pass 13's thirteen findings

| # | Disposition | Evidence I re-ran |
|---|---|---|
| **P13-1** error-body read escapes | **Fixed** | My own 21-cell probe, re-run isolated: **escaped 0 of 21**. Both `exc.read()` calls now carry their own inner guard; GET folds to `None`, POST degrades the message and keeps `LMStudioCallFailed`. The grid gained a third phase, `error-body`, rather than a corrected cell — the right shape. |
| **P13-3** non-finite / bool coercion | **Fixed** | `NaN`/`Infinity` through `chat()` now land `None`, not `nan`/`inf`; `True` → `None`; `"51.4"` → `51.4` and `0.25` → `250.0` still work. `bool` excluded *before* coercion, as recommended. |
| **P13-2** probe measured the validator | **Fixed** | `tests/test_packs.py` now calls `_row_count_identity_problems(pack, sampling)`. **The exact mutation that passed green in Pass 13 — re-inserting `if "analysisUnit" not in sampling: return []` — now fails.** |
| **P13-4** attested values below the fingerprint tier | **Fixed, and by derivation rather than transcription** | `ATTESTED_NONEMPTY_FIELD_NAMES` is computed from `fingerprint.REQUIRED_BY_SCHEMA[1]["model:chat"]` tiers. Flipping a tier in `fingerprint.py` alone reddens 71. `otherResidentWorkloads` correctly still accepts `[]`. |
| **P13-5** uncaught `HostInfoError` | **Fixed** | `attest --api-base-url ""` → exit **2**. |
| **P13-6** `EOFError` with no stdin | **Fixed** | Now exit **2**, naming the three fields never given via `--set`. |
| **P13-8** unvalidated back-fill | **Fixed** | A `host` with no `observedAtAttestation` now raises `HostInfoError` naming the precondition, before either branch. |
| **P13-10** unused parameter | **Fixed** | Renamed `_residency_source_after_a_successful_probe()`, parameter gone. |
| **P13-12** unknown `attested` keys | **Fixed** | `attested` now refuses any key outside the four. |
| **P13-7** `residency_source` uncompared | **Open, correctly — routed, not dropped** | Still `"unavailable"` unconditionally on the embeddings path. It is §5 OQ-1, an `architect` plan-reading call, not an implementer's. |
| **P13-9** plan §4 S2 `warm_up` signature | **Open, `architect`'s** | `docs/plans/small-model-benchmarking.md:4264-4265` unchanged. |
| **P13-11** `tools.module: ""` validates then reports "absent" | **Not fixed** | `_tool_module_problems` still early-returns `[]` on falsy. Nit; still stands. |
| **P13-13** header says S1 | **Not fixed** | Line 1 unchanged. Out of my write scope again this pass. |

### 4. The ruling — the convention line

Pass 13 pre-committed: *at five or more, the recurrence belongs to the convention, not to any
author.* It came in at five. The response is therefore **one line in `model-bench/AGENTS.md`**, not
five fixes. **Exact text, to be placed under that file's existing conventions list — I have not
edited the file:**

```markdown
- **A guard's reach lives in an asserted constant, not in prose.** A module-level set or table a
  guard consults — a required-key set, an allowlist, an exemption list, a role→unit map — needs one
  test that drives *the function consulting it* and asserts the computed set equals the constant,
  so both a shrink and a widen redden. Without that test the docstring may not claim a reach
  (*only*, *every*, *never a sixth*). The five constants that failed this in the S2 audit are
  listed in `docs/reviews/small-model-benchmarking-impl.md` Pass 14.
```

523 characters on its longest line, well under the ~700 bar; it states a live constraint rather
than history, and cites its evidence in a clause rather than reproducing it.

**Scope of the ruling, decided rather than assumed: `model-bench/` only.** The class is real
elsewhere in this repo, but every instance I have evidence for is in this component, the audit
that produced the number was scoped to it, and a root-`AGENTS.md` rule would bind components I
have not audited on a count I did not measure there. Pass 13 pre-stated `model-bench/AGENTS.md`
and I am not widening it on the strength of a hunch. If the convention proves out here, promoting
it is `cobb`'s call with its own evidence.

**What the ruling does and does not require.** Going forward: a new guard constant ships with its
pin or it does not ship. Retrospectively it requires only that a constant either **gain the pin or
lose the prose claim** — one sweep over five named constants, not five investigations, because the
audit has already done the finding. Three of the five carry a prose reach claim today
(`_REQUIRED_MODEL_INFO_KEYS`, `UNIT_KIND_BY_ROLE`'s "there is no sixth, and no default", the §4B
kind registries); two do not and need only the pin.

### 5. The five gaps

Each is a **one-member shrink that leaves the full suite at 834 passed**. Consequences executed,
not inferred.

**P14-1 (major) — `_REQUIRED_MODEL_INFO_KEYS` (`lmstudio.py:261`) guards the taxonomy U79 just
closed, and nothing holds it.** Dropping `"quantization"` → 834 passed; the missing-key check then
passes and `ModelInfo(quantization=raw["quantization"])` raises a bare **`KeyError` out of
`catalog()`** — executed. That is P13-1's own defect class, one layer up: an untyped escape from
the adapter's exception taxonomy, and `quantization` is a `_NONEMPTY` fingerprint field. *Pin:* one
test asserting the set of keys `_model_info_from_raw` actually rejects, computed by driving it with
each key deleted in turn, equals `_REQUIRED_MODEL_INFO_KEYS`.

**P14-2 (major) — `UNIT_KIND_BY_ROLE` (`roles.py:25`) can lose a role in green, and `ROLES` is
unbound to it in both directions.** Dropping `"chat-responder"` → 834 passed (`"embedder"` reddens
only because fixtures use it, so three of five roles are pinned by accident). Adding a spurious
sixth entry to `ROLES` → 834 passed. The docstring says "There is no sixth, and no default"; two
declarations of that set sit eight lines apart with nothing binding them — the exact shape U81 just
removed from `hostinfo.py`. *Pin:* `assert set(UNIT_KIND_BY_ROLE) == set(ROLES)`, plus
`unit_kind(r)` driven over all five.

**P14-3 (minor) — `_SAMPLE_NOUN` (`report.py:63`) is read through `.get(rp.unit_kind,
unit_plural)`, so a missing key silently reverts the rendered noun.** Dropping `"query"` → 834
passed, and the render falls back to the generic plural — **Pass 1's n-4 defect exactly**
("unwritten scripts" for every unit kind), recorded as fixed at Pass 2 and held by nothing since.
*Pin:* assert the map's domain equals `set(UNIT_KIND_BY_ROLE.values())` and one render per noun.

**P14-4 (minor) — `_AGGREGATE_BY_KIND` (`results.py:364`) can lose a kind in green.** Dropping
`"grounding"` → 834 passed; `_AGGREGATE_BY_KIND[d["kind"]]` then raises `KeyError`, which
`load_history` surfaces as `unparseable` — a **valid record silently quarantined**, in the module
whose thesis is that an unreadable record is a finding, not an absence. Inert until S7 ships a
grounding pack, which is exactly when nobody will be looking. *Pin:* one round-trip per kind.

**P14-5 (minor) — the §4B grid's kind registries are unbound to `_PHASE_KINDS`, inside the fix that
established the class.** `_PHASE_KINDS["connect"]` declares four kinds, `_CONNECT_KINDS` defines
three. Adding `"dns_failure"` to `_CONNECT_KINDS` alone → **834 passed**: never exercised, never
noticed. The reverse is covered — adding a kind to `_PHASE_KINDS` alone reddens (control run), and
shrinking `_READ_KINDS` reddens via `_route_outcome`'s `KeyError`. So exactly one direction is
open. *Pin:* assert each registry's key set equals its `_PHASE_KINDS` domain minus the exempt
kinds.

**P14-6 (nit) — the surviving exemption is inconsistent with the reasoning U79 applied to its
sibling.** U79 removed `("read", "non_2xx")` from `_EXEMPT_CELLS` on the grounds that it "was never
a real cell to begin with — `non_2xx` is outside the `read` phase's domain entirely", and dropped
it from `_PHASE_KINDS["read"]`. `("connect", "unparseable_body")` is out of domain for the same
kind of reason, yet stays *in* `_PHASE_KINDS["connect"]` and is then exempted. Its stated reason is
true; its placement is the older treatment. Dropping it from the domain would empty `_EXEMPT_CELLS`
— and the test asserting the exemption set would then need to permit an empty one, which is worth
deciding rather than discovering.

### 6. What's solid

- **U81 fixed P13-4 by derivation rather than by transcription, which is more than I asked for.**
  I suggested a test asserting the two tables agree; the implementer made the second table *be* the
  first one's tiers, so there is no second table to disagree. Its own docstring names the shape it
  was avoiding. That is the right answer to this whole class and it is the model the convention
  line generalises.
- **U79 added a phase to the grid rather than a cell.** Pass 13 suggested a third phase value; the
  implementer took it and also re-derived why `("read", "non_2xx")` was never a cell, which is a
  better closure than the one recommended. The GET/POST asymmetry — GET folds to `None`, POST keeps
  the status it already holds and degrades only the message — is reasoned out at both call sites.
- **U80 is a four-line change that closes the round-three thread by construction.** The probe now
  measures its own route; the mutation that defeated it in Pass 13 reddens.
- **Six of the twenty-five constants are already derived rather than transcribed**, and every one
  of those derivations tracks under mutation of its *source*. The component was already halfway to
  the convention it now needs stated.

### 7. Open questions

Unchanged from Pass 13 and still `architect`'s: OQ-1 (does §3.4.5's "degenerates to
`residencySource` alone" mean compare-it or give-up — P13-7) and the one-line §4 S2 `warm_up`
signature sweep (P13-9). **Both Pass 12 residuals classified as blocked remain blocked** — the real
`GET /api/v0/models` capture and §4 S2's R-1 probe, both waiting on a human-run live LM Studio
session. Nothing in these three commits could have moved either.

---

### Appendix N — Pass 14 evidence

**N.0 — isolation, corrected.** `git archive b5f719b model-bench | tar -x -C <scratch>/p14/tip`.
Probes run as `PYTHONPATH=<snap>:<scratch>/p14 python <probe>` with a prepended guard asserting
`os.path.dirname(os.path.dirname(modelbench.__file__)) == <snap>`, which is the addition Pass 13's
method needed. Mutations run under `pytest` from the snapshot root, which isolates on its own.

**N.1 — the audit table.** 25 entries; 32 mutations run.

| # | Constant | Shrink applied | Result | Disp |
|---|---|---|---|---|
| 1 | `hostinfo.ATTESTED_FIELD_NAMES` | −`hostRamGb` | 15 failed | **i** |
| 2 | `hostinfo.ATTESTED_NONEMPTY_FIELD_NAMES` | source tier flipped in `fingerprint.py` | 71 failed | **d** |
| 3 | `packs.ROW_COUNT_IDENTITY_KEYS` | −`data.conversations` | 10 failed | **i** |
| 4 | `packs._ROW_COUNT_IDENTITY_KEY_HINTS` | −`sampling.analysisUnit` | 1 failed | **i** |
| 5 | `packs.ROW_COUNT_IDENTITY_EXEMPT_CELLS` | emptied | 2 failed | **i** |
| 6 | `packs._STDLIB_MODULE_NAMES` | source −`json` | 2 failed | **d** |
| 7 | `lmstudio._REQUIRED_MODEL_INFO_KEYS` | −`quantization` | **834 passed** | **ii** |
| 8 | `fingerprint._MODEL_CHAT_SCHEMA_1` | −`loadedContextLength` | 3 failed | **i** |
| 9 | `fingerprint._MODEL_EMBEDDINGS_SCHEMA_1` | derivation defeated | 13 failed | **d** |
| 10 | `fingerprint._DETERMINISTIC_SCHEMA_1` | −`pythonVersion` | 69 failed | **i** |
| 11 | `fingerprint.REQUIRED_BY_SCHEMA` | −`model:embeddings` | collection error | **d** |
| 12 | `fingerprint._EMBEDDINGS_HAVE_NO` | −`maxTokens` | 9 failed | **i** |
| 13 | `fingerprint.ARM_KINDS` | derivation replaced by a literal | 1 failed | **d** |
| 14 | `fingerprint.CALL_SURFACES` | filter defeated | collection error | **d** |
| 15 | `fingerprint.RESIDENCY_ELEMENT_KEYS` | −`state` | 70 failed | **i** |
| 16 | `fingerprint._RESIDENCY_FIELDS` | −`residentModelsAtEnd` | 11 failed | **i** |
| 17 | `fingerprint._DISCRIMINATORS` | −`callSurface` | 15 failed | **i** |
| 18 | `results._AGGREGATE_BY_KIND` | −`grounding` | **834 passed** | **ii** |
| 19 | `results._METRIC_DECODERS` | −`distribution` | 4 failed | **i** |
| 20 | `results.INDEX_COLUMNS` | −`quantization` | 5 failed | **i** |
| 21 | `report._BASIS_STRENGTH` | −`measured` | 6 failed | **i** |
| 22 | `report._SAMPLE_NOUN` | −`query` | **834 passed** | **ii** |
| 23 | `report._NO_VERDICT_REASON` | −`too-few-arms` | 9 failed | **i** |
| 24 | `roles.UNIT_KIND_BY_ROLE` | −`chat-responder` | **834 passed** | **ii** |
| 25 | `test_lmstudio` kind registries vs `_PHASE_KINDS` | +`dns_failure` in `_CONNECT_KINDS` only | **834 passed** | **ii** |

Out of inventory, with the criterion applied: `report.py`'s ten prose constants,
`stats.SUPPORT_DIFF_PROPORTIONS` (a scalar bound), and **`roles.ROLES`** — nothing consults it, so
a shrink changes only an error message; its unbound relationship to `UNIT_KIND_BY_ROLE` is folded
into entry 24 rather than counted twice. Controls run for entry 25: +kind in `_PHASE_KINDS` alone →
1 failed; −kind from `_READ_KINDS` → 7 failed.

**N.2 — Pass 13 findings re-driven.** `escaped 0 of 21` (P13-1, was 21 of 21) ·
`ttftMs=None generationMs=None tokensPerSecond=None` through `chat()` on a `NaN`/`Infinity` body
(P13-3, was `nan`/`inf`/`nan`) · P13-2's own mutation now `1 failed, 34 passed` in
`test_packs.py` (was 35 passed) · `attest --api-base-url ""` → exit 2 (P13-5) · `attest` with no
stdin → exit 2 naming the unset fields (P13-6) · `check_attestation_staleness` on a host with no
`observedAtAttestation` → `HostInfoError` (P13-8) · `validate_host_info` on
`{"lmStudioAppVersion": "", "kvCacheSetting": "", "hostRamGb": 0, ...}` → three `: empty` problems
(P13-4, was `[]`).

## Pass 15 — 2026-09-09

### 1. Scope & verdict

**Reviewed:** unit **U78** at `40a9bc8` — `modelbench/tooling.py` (99 lines), `modelbench/convo.py`
(337), `tests/test_tooling.py` (172), `tests/test_convo.py` (591). Snapshot pinned to `6d6d4fe`;
the four files are **byte-identical** at `6d6d4fe` and at `5cbdf9e`
(`git diff --stat 6d6d4fe 5cbdf9e -- model-bench/modelbench/ model-bench/tests/` → empty), so the
plan amendment did not move the code under me. Baseline at that snapshot: **873 passed, 3
deselected**, ruff clean. Judged against plan **v1.25** (`5cbdf9e`) §3.3 / §3.8.4 / §4 S2 /
Appendix A and `-ml` §4.1–§4.2.

**Verdict: needs changes** — one blocker and four majors, all of which survive the mid-pass
ruling and none of which the rework closes for free.

**CPG: considered, not relevant — no Code Property Graph is loaded for `model-bench` (only
`cpg_falkorchat` and `cpg_deprecated_salesperson`), so every inventory below was built by `grep`
and every judgement is a mutation or probe actually run.**

**Isolation, per Pass 14 §N.0's correction, which this pass needed:** `git archive 6d6d4fe
model-bench | tar -x`, snapshot **first on `PYTHONPATH`**, and every probe opens with
`assert os.path.dirname(os.path.dirname(modelbench.__file__)) == <snap>`. Three other units were
committing to this tree while I worked (`5cbdf9e`, `127fca3`, plus five uncommitted paths), and the
assert is what makes the results below statements about `40a9bc8` rather than about whatever the
tree held at the moment each probe ran.

### 2. The rescope, stated plainly

**Mid-pass, `architect` ruled item 1 against the built behaviour** (plan v1.25, `5cbdf9e`): a prior
turn is replayed from **what the model actually produced this run**, `expect` is a scoring oracle
with `scoring/toolcalls.py` as its only reader, `assemble` becomes
`assemble(turn_index, script, observed, cfg)` with the precondition `len(observed) == turn_index`,
`historyReplay` gains `structured-replies-only`, and a turn becomes a bounded iteration loop with
`prompt.maxIterationsPerTurn` as required pack data.

**So the scripted-`expect` replay path and `assemble`'s current signature are excluded from this
pass.** I had findings there — the `role: "tool"` message carrying the oracle's
`finalReplyMustContain` fragments as if they were the tool's return value, the prior turn's own
final assistant reply never being replayed at all, and the replayed context contradicting the live
`ToolEnvironment` state from the first failed write-mutating turn onward. **They are withdrawn, not
overlooked.** The third is now §3.8.4's own third reason for the ruling; the first two are what
v1.25's `finalReplyText` and its verbatim-`tool_calls` rule exist to fix. A later reader should not
read their absence below as an oversight, and §5–§6 are where the knowledge from them went.

Everything else stood, and the pass had substance without them: **9 findings, 15 mutations and 5
probe scripts run.**

### 3. Findings

**P15-1 (blocker) — `drive` propagates any error from `llm(...)`, so one failed turn destroys the
whole conversation's record and the remaining turns are never driven.** `-ml` §4.1 is explicit that
a turn that cannot be driven at all (*"LM Studio 400 / crash, as `gpt-oss-20b` produced"*) is
**recorded as `unrunnable` and reported in its own count — never as a failure, never silently
dropped**, and v1.25's §4 S2 restates it for `drive` by name (*"It never stops early on a bad turn
(`-ml` §4.1)"*). Executed (Appendix O.2, probe B): a 3-turn script whose turn 2 raises issues **2 of
3** LLM calls and returns **no `ConversationTrace` at all** — turn 1's completed record is lost with
the exception. No caller can repair this: `drive`'s loop is `range(len(script.turns))` from 0 with
no resume entry point and no partial return, so a runner-side `try` recovers nothing. `drive`'s
docstring — *"Every scripted turn runs unconditionally, in order — `drive` never stops early on a
bad turn"* — is the **eleventh** instance of the coordination's class-1 defect: the non-exception
mechanism *is* held (M16, a `break` on a tool-call-free turn → 2 failed), the exception path is not.
And `test_drive_never_catches_an_error_the_llm_callable_raises` pins the behaviour the note forbids
— class 2's named root cause, written against the implementation rather than against `-ml` §4.1.
*Fix:* wrap each turn's model call in the rework's iteration loop, record the turn with an
`unrunnable` cause on `TurnTrace`, and continue. *Where the test goes:* the rework unit's
`test_convo.py`, driving `drive` over a 3-turn script whose turn 2 raises — asserting **3**
`TurnTrace`s, `llm` called **3** times, turn 2 `unrunnable`, turns 1 and 3 intact. **Not a list of
exception shapes: a coverage probe over the axis** — parametrize over a *generated* set including a
synthesized `Exception` subclass the module has never seen, so a catch narrowed to any named type
reddens. One consequence to carry: `turnIndex` (§3.3's third `pairingKey` component) is **positional
only** in `ConversationTrace.turns`, so a dropped turn silently shifts every later pairing key.

**P15-2 (major) — `validate_pack` never inspects the `prompt` block, so every FR-9a axis reaches
the harness unvalidated and fails, if at all, mid-conversation.** Executed (Appendix O.2, probe 2):
the `valid` fixture pack amended to declare `historyReplay: "verbose"`,
`representToolSchemasEachTurn: "yes"`, `historyTurns: -3` returns **`validate_pack(pack) == []`**;
`assemble` refuses `"verbose"` only at run time, after model calls have been spent. Nothing in
`modelbench/` outside `convo.py` mentions `prompt`, `historyReplay` or `PromptConfig` (grep). §3.3's
whole design is that `run` calls `validate_pack` first and **fails closed** — a pack-configuration
error is not a run-time discovery. v1.25 sharpens this twice over: `maxIterationsPerTurn` is now
**required pack data with no default**, and a manifest omitting it would validate clean today.
*Fix:* a `prompt`-block route in `validate_pack`, driven by a field→validator table, importing
`convo`'s own mode constant rather than transcribing it. **Coverage probe over the axes, not a list
of bad manifests:** generate one manifest per table entry with that single field invalidated and
assert the set of fields producing a refusal **equals the table's domain**, so a `PromptConfig`
field added without a validator reddens. *Absorbs the `historyTurns` nit:* `-3` currently behaves
exactly as `0` (unbounded) — executed, probe E. *Disposition:* **not blocked** — `validate_pack`
ships today and this half is buildable now; only the manifest→`PromptConfig` constructor is the
rework unit's, and that is dispatched work, not unbuilt work.

**P15-3 (major) — `_HISTORY_REPLAY_MODES` and the `HistoryReplay` `Literal` are two declarations of
one set, eight lines apart, with nothing binding them — P14-2's shape exactly, and v1.25 is about to
widen both.** Mutations, all against the full suite: adding `"quiet"` to the **frozenset** alone →
**873 passed**; adding it to the **`Literal`** alone → **873 passed**; a one-member shrink → 2
failed. Consequence executed (probe 2): a member of the constant with no branch in `assemble` is
**accepted** and silently produces no history — it degrades to `none` without saying so. The rework
must add `structured-replies-only` to both declarations, and today nothing would catch adding it to
one. *Fix, and I ran it (Appendix O.3):* one test with three clauses — **(b)**
`_HISTORY_REPLAY_MODES == set(get_args(HistoryReplay))`; **(c)** every member renders a **distinct**
message list from a fixture with one prior tool-calling turn; and (a) the accepted set equals the
constant. As shipped the probe is **green on all three**; under the widen it is **red on (b) and
(c)** and — the point of §4 below — **green on (a)**. *Disposition:* not blocked.

**P15-4 (major) — the `rawArguments`/`parsedArguments` seam: the premise handed to me is right about
the malformed case, wrong about the general one, and the plan is not the thing that is wrong.**
Three separable answers, each executed (Appendix O.4). **(i) `rawArguments` *can* differ from
`parsedArguments`** — the dependency runs env→env, not drive→env: `tooling.py`'s docstring assigns
`parsedArguments` to *"what the environment's own dispatch logic made of them after its own
unit/boundary handling"*, which is exactly the input FR-8(d)'s `boundary_unit` subset needs, and
§3.8.4's *"the environment records every call"* reads the same way. **The distinction is not
vestigial and the plan is not wrong to declare it.** **(ii) But `tooling.py`'s claim that
`rawArguments` is *"what the model's tool call actually carried"* exceeds its mechanism** — class 1
again. On a malformed call `_parse_tool_arguments` degrades to `{}`, and executed: unparseable JSON,
a JSON array, a JSON scalar and a genuine `{}` all arrive at `dispatch` as the same four bytes.
**(iii) The evidence is not lost from the *record*** — `TurnTrace.chatResult.tool_calls[i]
["function"]["arguments"]` still holds the raw string — **but it is lost from the field whose name
promises it**, and the recovery join is named nowhere. *Fourth, and the one I would fix first:*
`drive` **dispatches** the malformed call, mutating FR-10 ground-truth state on a call the model
never validly made; under v1.25's loop that `{}` dispatch's return value is now fed back to the
model as a `tool` message, so a harness-side parse failure shapes the model's next iteration.
*Fix:* v1.25 §4 S2 has already created the category — *"for one that could not be dispatched (no
name, or the dispatch raised) the content is a JSON object naming the failure"*. An
unparseable-arguments call belongs in it: `_parse_tool_arguments` returns `dict | None`, `None` is
not dispatched, and `{}` recovers its single meaning. *Route:* the rework unit; it touches neither
`ToolEnvironment.dispatch`'s signature nor the pack data format. Correct `tooling.py`'s docstring in
the same change — and note that
`test_dispatch_record_round_trips_distinct_raw_and_parsed_arguments` hand-builds its two distinct
mappings, so it demonstrates the dataclass and **not** that any producer in the codebase can make
them differ. *Disposition:* not blocked.

**P15-5 (major) — a turn's dispatch slice is `env.trace()[trace_before:]`, and an environment whose
trace does not grow monotonically silently yields an *empty* slice with no error.** Executed
(probe F): a `ToolEnvironment` that clears its own trace inside `dispatch` — a plausible pack bug —
produces a real `add_to_cart` on turn 2 that `drive` records as **0 dispatches**. FR-10 ground truth
is silently lost in the module whose entire thesis is that ground truth is the trace and the state.
The Protocol's docstring *states* the requirement (*"must return calls in a stable order and never
drop or reorder an earlier entry"*) and the `isinstance` guard checks **method presence only** —
declared reach against implemented reach, at the one seam a third party implements. v1.25 sharpens
it: the slice now spans several iterations. *Fix:* after the turn's dispatches, read
`after = env.trace()` once and refuse when `len(after) < trace_before`. *Trade-off, named rather
than hidden:* this raises mid-run and so can abort a conversation — which is right, because it is a
**pack** defect and pack defects fail closed (§3.3), unlike the *model* failure P15-1 requires to be
recorded and driven past. *Where the test goes:* a `ResettingEnv` fixture in the rework unit's test
file — **never** an assertion bolted onto `StubEnvironment`/`_FullEnvironment`, which conform by
construction and could not redden. *Disposition:* not blocked.

**P15-6 (minor) — a turn's *multiple* tool calls are unpinned.** Truncating the dispatch loop to
`chat_result.tool_calls[:1]` leaves the suite at **873 passed** — no test in either file gives one
response more than one call. `-ml` §4.2(e) requires `duplicate_turn_rate`'s **within-turn** variant
and `spurious_turn_rate`, both over `|E(t)| ≥ 1`, and names K-061's same-turn `add_to_cart` as the
observed defect; neither is scoreable if only the first call reaches the trace. *Fix:* a coverage
probe over the fan-out axis — parametrize a turn's call count over `(0, 1, 2, 5)` and assert
`len(turn.dispatches)` equals the number of dispatchable calls in emission order at every value,
rather than adding one two-call test. *Disposition:* not blocked.

**P15-7 (minor) — `TurnTrace.messagesSent` is not pinned against what was actually sent.** Replacing
it with `()` leaves the suite at **873 passed**. It is the stored record of the context the model
saw, and v1.25's entire ruling is about what is in that list — a run whose `messagesSent` did not
describe its own prompt could not be audited for the thing the ruling turns on. *Fix:* one
assertion in the existing per-turn tests, `trace.turns[i].messagesSent == tuple(llm.calls[i]
["messages"])` for every `i`. Under the loop this becomes iteration 1's list plus the in-turn
working list; say which the field holds. *Disposition:* not blocked.

**P15-8 (minor) — `ConversationTrace` carries `scriptId` but neither `replicate` nor `shape`, so
§3.3's `pairingKey` is not derivable from the trace alone.** `pairingKey` is
`["scriptId", "replicate", "turnIndex"]` and §3.8.4 says `replicate` *"exists so that raising
`replicatesPerScript` later does not change the record shape"* — a trace that omits it **is** that
record-shape change, arriving early and while it is still free. `shape` ∈ `{A, B, C}` is §3.8.4's
reporting stratum and is dropped the same way. *Fix:* carry `replicate: int` and `shape: str` on
`ConversationTrace`, sourced from `script`, pinned by a `Conversation(replicate=2, shape="B")`
fixture — `test_drive_returns_a_conversation_trace_named_for_the_script` currently pins `scriptId`
alone (control: blanking it → 1 failed). *Disposition:* not blocked.

**P15-9 (minor) — `PromptConfig`'s Appendix A field names are unpinned, while `DispatchRecord`'s are
pinned.** Renaming `historyTurns` → `historyWindow` with its own tests swept leaves the suite at
**873 passed**; the same rename on `DispatchRecord.rawArguments` **reddens** (control). The names
are manifest keys, so a drift is a silently-unreadable `pack.json`, and v1.25 adds
`maxIterationsPerTurn` to the tuple. *Fix:* `test_prompt_config_fields_match_the_plans_appendix_a_
literal_in_order`, in the exact shape of the `tooling.py` one that already works, transcribed from
§3.3's manifest example. *Disposition:* not blocked.

**No residual on this pass is deferred by choice, and none is blocked on unbuilt work.** Every
finding above is actionable either in `packs.py` today (P15-2's validator half) or in the rework
unit that v1.25 already requires. The two genuinely blocked items are Pass 12's and are unchanged:
the verbatim live `GET /api/v0/models` capture and §4 S2's R-1 probe, both waiting on a human-run LM
Studio session — and v1.25's F-S2-1 ruling has now given the first of those a written interim shape
rather than leaving it as a citation to an artifact that does not exist.

### 4. A refinement Pass 14's own convention line needs, found by running it

The line landed in `model-bench/AGENTS.md` (uncommitted, working tree) as written: *"needs one test
that drives the function consulting it and asserts **the computed set equals the constant**, so both
a shrink and a widen redden."* **Executed against `_HISTORY_REPLAY_MODES`, that clause is a
tautology and reddens on nothing.** Probe 5, clause (a) — drive `assemble` over every member plus a
bogus value and collect what it accepts — is **green both as shipped and under the widen**, because
the only mechanism consulting the constant is a membership test, so the accepted set equals the
constant *for any constant*. Clauses (b) and (c) are what go red.

So the convention has three forms and only two of them bind:

- **(i) bind two independent declarations** — `set(UNIT_KIND_BY_ROLE) == set(ROLES)`,
  `_HISTORY_REPLAY_MODES == set(get_args(HistoryReplay))`. Non-tautological because the two
  declarations are written separately.
- **(ii) assert a *behavioural* consequence per member** — each member renders a distinct output,
  each key round-trips, each missing key produces the *typed* refusal rather than any refusal.
  Non-tautological because the behaviour is not the membership test.
- **(iii) assert the consulting function's accepted set equals the constant** — **tautological
  whenever the guard is a pure membership test.** P14-1's pin escapes it only because
  `_model_info_from_raw` has a *second* mechanism (`raw["quantization"]`) whose bare `KeyError` the
  pin can distinguish from the typed refusal; P14-3/4/5's pins are form (i) or (ii).

**Suggested amendment, for the human to apply since the line is theirs to place:** replace *"asserts
the computed set equals the constant"* with *"binds it to the other declaration of the same set, or
asserts a distinct behavioural consequence for every member — never merely that the guard accepts
what the guard's own constant contains, which is true of any constant."* I have not edited
`AGENTS.md`.

### 5. What the rework must carry forward — the highest-value output of this pass

Ten things U78 got right that a rewrite from the amended plan would plausibly lose. Each is verified
here, not remembered.

1. **`drive`'s `isinstance(env, ToolEnvironment)` guard, and its test's second assertion.** The unit
   caught its own docstring overclaiming and closed it by **building the guard** rather than
   softening the prose; the test asserts `llm.calls == []`, so *"before any LLM call"* is checked and
   not just the raise. It must be the first statement of the rebuilt `drive`.
2. **`tooling.py` is untouched by the ruling** — §4 S2's `tooling.py` sketch is identical across the
   amendment. Do not reopen it except for P15-4(ii)'s docstring correction.
3. **`_TOOL_ENVIRONMENT_METHODS` is a correct instance of the Pass 14 convention, written before the
   convention existed** — and it is form (i)+(ii), not the tautological (iii): a fifth Protocol
   method → **13 failed**, a one-member shrink of the constant → **1 failed**. Leave it alone.
4. **The Appendix-A field-order pin on `DispatchRecord`** (control: reddens on rename). It is the
   working template P15-9 asks for on `PromptConfig`.
5. **Per-turn dispatch isolation by `len(env.trace())` diff** — v1.25 keeps the mechanism verbatim
   (*"`dispatches` (that turn's own slice of `env.trace()`)"*), and its test reddens when widened to
   the whole trace. Carry both, plus P15-5's guard.
6. **`envState` read *after* the turn's own dispatches** — pinned (reading it before → 1 failed).
   Under the loop it must be read after the **last** iteration's dispatches; the existing
   `callCount` fixture extends directly.
7. **`tools=` on every turn regardless of `representToolSchemasEachTurn`** — v1.25 does not touch
   this and the reasoning holds (withholding it makes native calls structurally impossible from turn
   2 on any pack setting the flag `false`). The test survives verbatim.
8. **`_parse_tool_arguments`' *tolerance* is right in shape** — a malformed tool call is the model's
   failure to score, not the harness's to crash on. Only its lossy `{}` collapse is wrong (P15-4).
9. **Order assertions and the whole-list contract** — v1.25 restates both (system · schema block ·
   history · current user **last**; the whole list, never an increment), and the current tests catch
   an `insert(0, …)` mutation on **10** tests. Port them; do not re-derive them.
10. **The `representToolSchemasEachTurn` and `historyTurns` tests' deliberate generality** — N=1 and
    N=2 for the window, all three turns for "every turn". Both axes are unchanged in v1.25. A
    rewrite costs a signature edit; a re-derivation loses the generality the unit added on purpose.

### 6. Collisions the new `TurnTrace` / iteration-loop shape will hit

- **`_expected_exchange` and `_flatten_turn` are dead** under the ruling — they are `convo.py`'s
  only readers of `expect`, and §3.8.4 now names `scoring/toolcalls.py` as its **only** reader.
  *Gate for the rework, runnable:* `grep -c 'turn\.expect\|expect\.get\|expect = '
  modelbench/convo.py` returns **9** today and must return **0**; `Turn.expect`'s own field
  declaration and docstring are the only permitted survivors.
- **`chatResult` singular → `chatResults` plural.** §3.8.4 gives **iteration 1** a privileged role
  (*"the turn's emission form is read from iteration 1"*), so decide explicitly whether a
  `chatResult` property over `chatResults[0]` survives rather than letting every scorer reach for
  the index.
- **`finalReplyText: str | None`, `None` iff `capHit`** is a two-declaration invariant of exactly
  the shape §4 above is about. It needs one test driving `drive` over **both** branches — the
  docstring will not hold it, and this coordination has eleven instances proving that.
- **`wallClockMs` keeps its name and widens its meaning** — one call today, *"covering the whole
  turn"* under v1.25. The existing non-negative test passes under either scope, so nothing would
  catch the wrong one. Assert the turn's `wallClockMs` is `>=` the sum of its `chatResults`'
  `wallClockMs`.
- **`historyTurns` windowing now spans two sequences.** Today it windows `history[:turn_index]`;
  under `assemble(turn_index, script, observed, cfg)` each replayed prior turn `i` draws its user
  text from `script[i]` and its content from `observed[i]`, so the window must be applied to the
  **pair** — an index drift between the two is silent and produces a plausible transcript.
- **Both preconditions raise, and the old one must survive.** `0 <= turn_index < len(script)` is
  pinned today (dropping the negative half → 1 failed); v1.25 adds `len(observed) == turn_index`
  beside it, not in place of it.

### 7. What's solid

Beyond §5's ten: the **plan-literal check-input discipline** is real in this unit and it is the
answer to the coordination's named root cause. `test_dispatch_record_fields_match_the_plans_
appendix_a_literal_in_order` transcribes Appendix A's tuple and `test_assemble_transcribed_from_the_
plans_own_conversation_row_literal` drives §3.8.4's own `A-02` row JSON verbatim — including
`argChecks` and `terminal`, which nothing reads and everything must tolerate. Where the unit applied
that discipline it held; the four majors above are all in places where it did not reach (a
`prompt`-block validator that does not exist, a constant with no plan literal to transcribe, a
docstring sentence with no test). The HISTORY.md entry is honest about its own 14 mutations and
routes both open questions rather than assuming them — including the `rawArguments` one, which was
right to route and is answered in P15-4.

### 8. Open questions

- **`architect`** — Pass 14's OQ-1 (P13-7) and the `warm_up` signature (P13-9) are **both closed by
  v1.25** (items 4 and 5); nothing carries forward from Pass 14's §7 except the two blocked live
  items named at the end of §3.
- **The human** — §4's amendment to the convention line now in `model-bench/AGENTS.md`. It is a
  wording change to a line that has not yet been committed, and it is the difference between a pin
  that reddens and one that cannot.

---

### Appendix O — Pass 15 evidence

**O.1 — isolation.** `git archive 6d6d4fe model-bench | tar -x -C <scratch>/p15/tip`. Probes run as
`PYTHONPATH=<snap>:<scratch>/p15 .venv/bin/python <probe>` with a prepended
`assert os.path.dirname(os.path.dirname(modelbench.__file__)) == <snap>`. Mutations run by copying
the snapshot aside, editing the copy, and running `pytest -q` from the copy's root with the same
`PYTHONPATH`; the snapshot itself is never edited. Baseline **873 passed, 3 deselected**.

**O.2 — the 15 mutations.** Green = a gap. Every one was applied to its own fresh copy of the
snapshot and the snapshot itself was never edited, so no restore step can have leaked.

| # | Mutation | Result | Finding |
|---|---|---|---|
| 1 | `_HISTORY_REPLAY_MODES` + `"quiet"` | **873 passed** | P15-3 |
| 2 | `_HISTORY_REPLAY_MODES` − `"none"` | 2 failed | — |
| 3 | `HistoryReplay` Literal + `"quiet"` only | **873 passed** | P15-3 |
| 4 | fifth method added to `ToolEnvironment` | 13 failed | §5.3 |
| 5 | `_TOOL_ENVIRONMENT_METHODS` − `"state"` | 1 failed | §5.3 |
| 6 | `PromptConfig.historyTurns` → `historyWindow`, tests swept | **873 passed** | P15-9 |
| 7 | `envState` read before the turn's dispatches | 1 failed | §5.6 |
| 8 | current-turn user message `insert(0, …)` | 10 failed | §5.9 |
| 9 | `historyTurns` windowing block deleted | 2 failed | §5.10 |
| 10 | `break` out of the turn loop on a tool-call-free turn | 2 failed | P15-1 (control: the non-exception mechanism *is* held) |
| 11 | dispatch only `tool_calls[:1]` | **873 passed** | P15-6 |
| 12 | `0 <= turn_index` half of the range guard dropped | 1 failed | §6 |
| 13 | `DispatchRecord.rawArguments` → `raw_arguments`, tests swept | 1 failed | P15-9 (control) |
| 14 | `messagesSent=()` | **873 passed** | P15-7 |
| 15 | `ConversationTrace(scriptId="")` | 1 failed | P15-8 (control) |

**O.3 — probe 5, the P15-3 fix, run.** Clauses: **(a)** accepted set == constant; **(b)** constant
== `set(get_args(HistoryReplay))`; **(c)** every member renders a distinct message list from a
fixture with one prior tool-calling turn.

```
--- as shipped at 40a9bc8 ---            --- with "quiet" added to the constant ---
  pass (b)                                 FAIL (b): Literal != constant
  pass (a)                                 pass (a)      <- the tautology, §4
  pass (c): all 3 modes distinct           FAIL (c): ('none', 'quiet') render identically
  PROBE GREEN                              PROBE RED
```

**O.4 — probe A, the `rawArguments` seam.** A native call carrying `arguments: "{name: 'Pad', qty:
2"` (unparseable):

```
dispatched         : add_to_cart -> {}
rawArguments       : {}   parsedArguments: {}
env state mutated  : {'calls': 1}          <- FR-10 state moved on an invalid call
raw string only in : chatResult.tool_calls[0]["function"]["arguments"] = {name: 'Pad', qty: 2
_parse('[1,2]')    : {}     _parse('5') : {}     _parse('{}') : {}   <- all four collapse
```

**O.5 — probe B, the blocker.** 3-turn script, `llm` raises on turn 2: `llm calls issued: 2 of 3` ·
`raised: LM Studio 400` · `ConversationTrace returned: NONE`.

**O.6 — probe F, the trace-monotonicity gap.** `ResettingEnv` (clears its trace inside `dispatch`),
2 turns, one real call each: `turn 1 dispatches: 1` · `turn 2 dispatches: 0`.

**O.7 — probe 2, the unvalidated `prompt` block.** `valid` fixture amended to
`{"historyReplay": "verbose", "representToolSchemasEachTurn": "yes", "historyTurns": -3, …}` →
`validate_pack(load_pack(root))` returns **`[]`**; `assemble` raises
`unknown historyReplay 'verbose'` only when called. Probe E: `historyTurns=-1` replays the full
unbounded prefix, identical to `historyTurns=0`.
