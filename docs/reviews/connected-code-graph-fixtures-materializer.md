# Connected code graph — S39 materializer, schema, cross.yaml and test_fixtures gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P0) · **Reviews:** commit `3865a55a` against `docs/plans/connected-code-graph-p0-2.md` row S39, `docs/test-plans/connected-code-graph-fixtures.md` v1.3 §8-§9 and `docs/reviews/connected-code-graph-fixtures-trees.md` (Passes 1-2; carried F-6, F-8, F-4 skew key)

## 1. Scope and verdict

Diff of `3865a55a`: `cpg/fixtures/materialize.py`, `cpg/ccg/schemas/fixture-manifest-v1.schema.json`, `expected/cross.yaml`, `cpg/tests/test_fixtures.py`, and the forced edits to `expected/fx-{job,lib,web}.yaml`. This is a new topic (the trees review is S36-S38), so a new document (collision rule 5). No source, test or config edited; no git mutation. Only `root REVIEW.md` exists (no `cpg/REVIEW.md`).

**Verdict: needs changes.** The materializer, the history convention (F-6), the forced manifest edits and the AB-1/LNK-42/43 realization are sound. The blocker is test-side: one plan-counted AC-25 assertion cannot fail. Four majors are the same defect class: the suite has green values that no check ties to the plan, the source or the scenario text, so a corrupted row, branch or key still passes.

CPG: considered, not relevant — the subject is fixture trees, a materializer and tests; no loaded `cpg_*` graph covers them. Verification is by execution.

**Executed (all by me):**
- `cd cpg && .venv/bin/python -m pytest -q tests/test_fixtures.py`: **62 passed, 2 deselected** (6.8 s). `-m live`, run with `CCG_FIXTURES_REPORT`: **2 passed** (60 s; Docker, image `ccg-toolchain:a82faf5219b3` present).
- **Determinism:** two materializations (second under `umask 077`, `TZ=Asia/Tokyo`, `HOME=/tmp`) wrote byte-identical `--json` SHA maps (`cmp` equal). fx-svc: 5 commits, 5 branches; fx-job 5/2; fx-lib 3/1; fx-web 2/1.
- **Forced manifest edits** (`git diff 3865a55a~1 3865a55a` on the three yamls): only the `history`/`branches` block and the three fx-job skew keys (LNK-34, 35, 41) changed. Each key now equals the provider signature in `fx-svc/common/fx_svc_common/{fmt,split}.py` (`round_money(amount,*,mode)`, `split_money(amount,parts,*,remainder)`, `fmt_date(ts)`) and the fx-svc manifest. No row's id/kind/status/reason/gate/locator and no `*_line` field changed. The H4 anchor in fx-job (docstring + `# Reviewed for the quarterly audit.`) reproduces the old "line 2" comment; fx-lib H0/H3 anchors are the old `pre_h3_edits`. Heads are tested equal to the checked-in trees.
- **AB-1, LNK-42/43:** `legacy/` and `tools/` each carry their own `pyproject.toml`, `app.py` providers `/legacy/fee`, `/tools/run`, and `api/orders.py` consumers; `legacy/broken.py` (`def broken(:`) is added at H2 and stays through H4 (`buildFailure` H2→HEAD); H1 is an allow-empty commit with no diff from H0; `svc-tools` is "never built"; LNK-42/43 are `possible`/`target-not-mapped`/P1 in both manifest and cross.yaml. Seven mutants (M10 H1 gains a file, M11 tools `pyproject.toml` removed, M13 legacy route changed, M14 valid `broken.py`, M15 `buildFailure.from` H3, M16 AB-1 `rootPath`, plus the builder's own plants) are all killed. M12 (rename the legacy project) survives, harmlessly: the name is not a contract.

**Mutants** (each on a fresh copy of `cpg/`+`docs/` in the scratchpad, full non-live suite; killed = at least one red test):

| id | mutation | result |
|---|---|---|
| M4 | fx-svc H3 `to:` text edited | killed (`h0_and_h1_trees`) |
| M6 | materializer anchor check `n != 1` → `n < 1` | killed (`ops_are_strict`) |
| M8 | materializer cuts branches from HEAD, not their base | survives, equivalent: every branch base is H4 = HEAD |
| M9 | author email from `getpass` | killed (`history_shape`) |
| M22b | fx-lib `P_line` 14 → 99 | killed (TP-002) |
| M24 | Java signature type corrupted | killed (cross drift + TP-002) |
| **M1** | skew key arg `mode`→`rounding` in job+svc+cross, consistently | **survives** |
| **M2** | ov-job op `to`: `generated_at`→`generated_on` | **survives** |
| **M3b** | tf-mid op `to` made a syntax error | **survives** |
| **M3c** | tf-sink `to`: `check=True`→`check=False` | **survives** |
| **M3d** | tf-mid edit replaced by a one-line edit (no added line, TF-1's requirement) | **survives** |
| **M25** | ov-svc `base: H4`→`H3` | **survives** |
| **M20** | fx-web MR-TS reason `resolution-unverified`→`name-match` | **survives** |
| **M21** | fx-web AC6-DYN-3 reason `dynamic-import`→`runtime-built` | **survives** |
| **T1/T4** | A25-1 drops its `direct` conjunct / its `distance` conjunct (separately) | **survive** |
| **T2** | A25-4 forced `True` | **survives** |
| M5, M7 | drop `GIT_CONFIG_GLOBAL`; chmod every copied file 0755 | survive (M5 equivalent under `HOME=/nonexistent`; M7 see F-7) |

## 2. Findings

### F-1 [blocker, verified: M2 runs] A25-4 cannot fail, and A25-1's two conjuncts mask each other
`tests/test_fixtures.py:1176` computes `listed` from an answer that `build_answer` built from `rv["deps"]`, then compares it to `rv["deps"]`: no builder in `MUTANTS` can drop a dependency, and `test_ac25_planted_failing_answer_builders_turn_the_column_red` has no A25-4 branch. Forcing the A25-4 result `True` (T2) leaves 62 green, so 3 of the plan's 60 assertions (§3) are decoration. In A25-1 the `distance ==` and the `direct ==` conjuncts (`:1157`) each hide the other: the only planted mutant (`distance_shift`) moves both, so deleting either (T1, T4) stays green; a builder with the right distance and `direct` always true has no plant.
**Fix:** add two planted builders to `MUTANTS`, with the failing-kind assertions in the red-test: (e) drops the last dependency of every reverse answer (must turn exactly A25-4 red, 3 ids); (f) keeps distances and sets `direct=True` on every item (must turn exactly A25-1 red for the indirect items). Expected sets are enumerated from the plan's RV/DF/CD rows, not computed by `check_answer`.

### F-2 [major, verified: M2, M3b-d, M25] branch contents and bases are self-referential
`test_tp001_branches_are_cut_from_...` compares `git diff --name-only` with the manifest's own op paths, and the parent with the manifest's own `base` (so a materializer cutting from HEAD, M8, is undetectable while every base is H4). Nothing reads an op's `from`/`to`, nor a branch's `base`, against the plan. So the five branches that carry OV-1..5 and TF-1..3 (TP-011) can hold a syntax error (M3b), the wrong change (M2, M3c), lose TF-1's "adds a line above the sink" requirement (M3d), or branch off H3 (M25), all green. A faithful implementer reaching the "after" state would not notice; the first sign would be a P3/P5 engine red, with the fixture as suspect.
**Fix (coverage probe, not a list):** per branch × per op × {`base`, `path`, `from`, `to`}, mutate it in a copy and require the suite red; count mutants run = 5 branches' op fields, asserted. The detectors to add: (a) every branch tree parses (`ast.parse` for `.py`, via the existing resolvers elsewhere); (b) the plan's §5.8/§5.10 change-set text is encoded as an expected change per branch (for TF-1: `added_lines >= 1` between `H4` and `tf-mid` for the file; for `tf-sink`/`tf-src` a change to the named function's body only, via `resolve_python` spans); (c) `base` equals the plan's `off H4` for all five.

### F-3 [major, verified: M20, M21] status/reason/gate/key of the rows have no per-row tie to the plan
`census_violations` compares counts (per gate, kind, status) with §6. A reason swapped for another registry-valid one (M20, M21), on a row held by one manifest, passes: `cross.yaml` is the emitter's output over those same manifests, and TP-003 checks only registry membership and status/reason agreement. These manifests are the oracle for TP-010 and every later gate; Pass 1's own plan comparison (133 rows, 0 differences) was the only thing standing in for this check, and the F-8 gate-blindness note carried here was not closed for status/reason. (Parsing the plan is already done: `PLAN_TEXT` feeds the AC-25 order test.)
**Fix:** parse every §5 table row (id, status, reason, gate, key where the table has it) from `PLAN_TEXT` and require equality with `row_view` of each holder, with the `a / b` and `c`/`p` dialects handled. Coverage probe: for every row × {status, reason, gate, key}, a mutated copy is red (count asserted = rows × fields present), run as a meta-test, not 132 hand-written cases.

### F-4 [major, verified: M1] the skew key is tied to the manifests, not to the provider source
F-4 required one key form; `test_f4_skew_keys_...` checks a regex and equality across manifests. Corrupting an argument consistently in all three files (`mode`→`rounding`, M1) passes, though the provider is `round_money(amount, *, mode=...)`. This is the silent-argument class: the P2 version-skew rows (LNK-33/34/35/41) would then assert a key no extractor can derive.
**Fix:** derive each of the four keys from `fx-svc/common/fx_svc_common/{fmt,split}.py` with `ast` (module path + name + parameter list with `*` marker, defaults dropped) and compare. Probe axes: each key × {module, name, each parameter, `*` position}.

### F-5 [major, judgment on D2] the named-symbol rule is test-local, loose, and does not fix the shared floor
Live report (this run): `svc` 60/61 expected symbols found, `job` 15/16, `web` 33/42 (missing `API_PREFIX`, `DEFAULT_ITEM`, `ORDER_PATH`, `README`, `RELATED_ID`, `SAMPLE_ORDER`, `TOOL_LIST_ORDERS`, `TOPIC_ORDERS`, `TOPIC_PREFIX`), `lib-cli` 3/3, `lib-pricing` 4/4, `svc-common` 4/4, `svc-legacy` 1/1, `svc-tools` 1/1. The planted live test shows `<module>` passes D15's unit floor for a comments-only unit (`methods == {"<module>"}`, verdict `ok`) and fails the named rule; that part holds. Weaknesses: (i) the rule lives in `test_fixtures.py`; `cpg-content-check.sc` (the single shared check S11, S25, S28 use, P3-7) still passes a comments-only unit, so S11's "CPG passes D15" will not detect an empty parse in any unit whose files emit `<module>` — the planted test proves the hazard exists but nothing upstream closes it. (ii) the denominator includes constants and `README` (non-callable), and the threshold is `>= 0.5`; `web` could lose a third of its methods and pass. (iii) the name is the last qname segment, so `Rules.fee`/`PremiumRules.fee` count once. (iv) `files_with_zero_methods` is reported (`lib-*`: `<unknown>`) and never asserted. Judgment: sufficient to say "each unit parsed something real" for S39's gate; not a D15 replacement.
**Fix:** a one-sentence ruling for the plan owner that S11/S25/S28 adopt the named-symbol rule (or the per-file rule S11 is measuring) in `cpg-content-check.sc`; restrict the denominator to symbols whose manifest locator names a function/method/class, and raise the floor to a recorded value per unit (the numbers above), so a drop is a red, not a 50% allowance.

### F-6 [minor, verified] TP-018 is green by pinning one open finding
`PLANTER_FINDINGS` (`:1410`) pins `("fx-svc","api/app.py","/api","API_PREFIX")` and the test asserts the sweep returns exactly it. TP-018's expected is "every literal that can form a `Link` is in a row"; the suite is green while one is not. The pin is stale-proof (a vanished or new finding fails), which is right, but the decision (add `aux: fx-svc:api/app.py` to LNK-04, or rule the prefix outside the plan) is open. Owner `qa-engineer` (plan v1.3 revision); then delete the pin.

### F-7 [minor, verified: M7] determinism is relative, and one cross.yaml pin is a count
SHAs are compared run-against-run on one host; nothing pins a SHA or the file modes (M7 sets every file 0755 and stays green), so a host-state dependency that is stable (git version behavior, the source tree's exec bits via `os.access`) is invisible. The `40`/`43` counts in `test_cross_yaml_holds_...` are literals; `cross.yaml` equals the emitter's own output (drift test), so an emitter defect (it drops AB-1's `expected` and `item` text, and AB-2..4 are not in it) is self-consistent. **Suggest:** assert modes `{0644, 0755}` against `git ls-files -s`, and pin the head SHAs of the four repos in the test with a one-line "update on a deliberate fixture change" note.

### F-8 [nit] `fx-job` ov-job lost its `qname` annotations (`write_report`, `status`) when `edits` became `ops`; the OV rows carry the locators, so harmless.

## 3. What's solid
Materializer (`materialize.py`): fixed identity, dates and environment; strict anchors (M6 killed); marker-guarded `--force` with a refusal test; clean-tree check after build. History is one forward-only convention, rejected-construct tests in the schema, H0/H1/H3/H4 content tests. AB-1/LNK-42/43 realization and its gate (seven mutants killed). TP-016 window, TP-002 (lines, signatures, overloads), TP-004/013/014 census, TP-005 and TP-018 sweep with its (literal, enclosing element) key (the Pass 1 F-8 carry: closed). A25-3's four pairs verified to sort `possible` first by name, path and line. Live test passes all 8 units.

## 4. Re-gate stopping condition
Re-gate closes when, run on copies, **every** mutant in the §1 table marked survives (M1, M2, M3b, M3c, M3d, M25, M20, M21, T1, T2, T4) turns the non-live suite red, **and** the F-2 and F-3 probes report their enumerated mutant counts (branch ops × fields; rows × fields) with zero survivors, **and** the F-1 builders (e), (f) turn exactly A25-4 and exactly A25-1 red. F-5 closes with the plan owner's ruling and the per-unit floors recorded. F-6, F-7, F-8 do not gate. The probe's assertions belong in a round before the fixes that satisfy them.

## 5. Open questions
1. F-5: does the plan owner move the named-symbol rule into `cpg-content-check.sc` (changing S11/S25/S28's acceptance), or keep it S39-only?
2. F-6: add `aux` to LNK-04, or rule `/api` out of TP-018's scope?
3. Is AC-25 bound to S33's answer fields (OI-1) only at P1? The neutral builder here is the reference answer; this gate judged it only as a falsifiable column, not as an engine check.

Growth-stops: no `requirements/`, `plans/` or `reviews/` document was revised by the delivered commit (its only docs change is one ledger line); nothing to report.
