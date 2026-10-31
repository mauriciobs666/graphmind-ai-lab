# Typed Decision Models — Review of the Method-Amendments Plan

> **Status:** active · **Owner:** `analyst` · **Tracks:** — · **Reviews:** `docs/plans/typed-decision-models2.md`

## Pass 1

### Scope & verdict

**Reviewed:** `docs/plans/typed-decision-models2.md` v1.0 (uncommitted, ~3,800 words), which
extends plan v1.3 (`docs/plans/typed-decision-models.md`). I checked it against:
- the method note `docs/plans/typed-decision-models-ml.md` v1.3, reading every one of its 11
  "Handed to architect" items in the note's own words;
- the requirements (FR-1, FR-4, FR-8, FR-10, FR-11a and AC-1/6/12 as revised 2026-10-30);
- `docs/reviews/typed-decision-models.md` Passes 5–7, and the impl review's carried items
  (`docs/reviews/typed-decision-models-impl.md` Pass 1, open questions 1–2);
- the built and uncommitted code: `model-bench/modelbench/{fingerprint,lmstudio,runner,cli,results}.py`
  and `scoring/classification.py`, plus `report.py` (`_arm_names`, `rank_report`), `hostinfo.py`
  (attestation trip-wire) and `tests/test_{fingerprint,results}.py` (pins and the stored-records
  test);
- the falkor-chat sites the plan cites: `background.py:358-369`, `executor.py:686-693`,
  `repository.py:726`, `guards.py:538-567`.

I did not run the suite, because this is a plan review and no code changed. I re-checked the
§Q3.5 power target against `parity.py:70`.

**Verdict: needs changes.** There are no blockers. All 11 handed items and both carried items are
placed, and the replicate design is right in the direction that matters: it never counts a false
second load (see "Answer to brief check 2"). Six majors remain. Two of them let a bar get easier
through operator choices the plan leaves open (M1, M2), which breaks the note's binding principle.
One is a dependency hole that the queued probe could trigger at any time (M3). Three are
under-specified replicate mechanics that an implementer would have to guess (M4–M6).

**Stopping condition (falsifiable, for a line check of v1.1 with no Pass 2).** v1.1 is accepted
when each of these is true, and no section other than the ones they name has changed design:
1. **M1.** §3.6 refuses a decision-point `compare` whose session holds more than one run of the
   chat arm's `armKey`, and §5's U3c row tests that refusal.
2. **M2.** On a decision-point pack, the chat-call timeout of both the incumbent arm and the
   hand-off call is not operator-tunable. Either it is pinned (a constant or a candidate field),
   and `--request-timeout` is refused on such packs, or it is recorded per run and `compare`
   refuses arms that differ. §5 has a test row for the choice made.
3. **M3.** §4.3 has the edge `U2 → U1e`, or splits U1e so that its route/fingerprint half follows
   U2. U1e's condition reads "before any `model:decisions` or `model:handoff` record exists in
   `model-bench/results/runs/`" and cites `test_every_stored_run_record_still_validates_unchanged`
   as the check that enforces it.
4. **M4.** §3.6 states three things. First, which arms a decision-point `compare` takes with
   `--models` and without it. Second, what `compare_report` receives as `runs` beside
   `replicates`, and that no `stats.verdict`/Holm table renders over candidate replicates. Third,
   that `rank` and `--negative-control` refuse on a decision-point pack. It also names the
   function that computes the load count and the fingerprint field it reads.
5. **M5.** §3.6 defines one K per replicate group, as the K of its earliest run. §5 tests a
   session where two candidates' replicates interleave.
6. **M6.** §3.3 defines how a guard decision-call failure enters the FR-1 line: which cell it
   joins, where `candidate e₁/n` comes from, side-check items, `handoffShare`, and the
   disagreement line. Exactly one unit owns `classification.score_answer`'s `failed_call` branch.

Each minor is either fixed or declined with a one-line reason in the revision note.

CPG: considered, not relevant — there is no `cpg_modelbench` graph (`GRAPHS` lists only `cpg_falkorchat` among CPGs), and every model-bench and falkor-chat site the plan cites was read directly.

**Answer to brief check 2 (replicate load detection).**
- **No false "two loads".** `residentModelsAtStart` is `catalog()` filtered on
  `state != "not-loaded"` (`lmstudio.py:605-613`), captured at capture-order step 3, before the
  warm-up (`runner.py:801`, `:822`). If the decision model is absent there, the warm-up must load
  it, so that run is on a load distinct from every earlier one. The rule is sound in the direction
  that could produce a false `met`.
- **False "one load" is possible, and conservative.** Two cases: U7's own order with two
  candidates per point (m1), and any other task reloading `laya` between replicates. Both print
  `not assessable`, never `met`. R-11 says so correctly.
- **Within one run the claim is weaker** (m2). Under JIT, a mid-run eviction withholds one item
  for `"load"`, and the items after it run on a new load.
- **Last-wins.** It is fully avoided for candidate arms. It survives on the chat arm (M1), in
  `rank` (`cli._select_rank_arms`, keyed on `modelKey`) and in `--negative-control` (M4).
  `index.csv` is unaffected: `rebuild_index` writes one row per stored record with no dedup
  (`results.py:1176-1193`).

**Growth-stops / one-home check.** This is a first version (a successor, not a revision), so
revision growth does not apply. The successor mostly cites rather than restates. Its restated facts
and one factual slip are m9. The Extends rule (§1, "where the two disagree on a named point, this
document wins") is clear. It fails only where an amendment's replace scope is narrower than the
v1.3 text it contradicts (m6).

### Findings

**M1 (major) — The incumbent arm keeps last-wins, which is an unrecorded selection on FR-1/FR-2.**
- §3.6 says "Chat arms keep `_select_arms`' last-wins". In code that is
  `by_key = {r.modelKey: r for r in candidates}` (`cli.py:209`).
- An operator who runs the incumbent twice in the session gets the newest run, silently. FR-1 is
  paired against that run's errors and FR-2 against its latency. Re-running the incumbent until
  it comes out slow or error-prone makes both bars easier, and nothing in the report shows it.
- The plan refuses this exact shape for candidates ("silently picking one would be selection",
  §3.6). The same reasoning applies to the arm every replicate is paired with.

*Fix (§3.6, U3c, §5):* a decision-point `compare` refuses, as a usage error, when the session
holds more than one run of the chat arm's `armKey`. This matches the two-content-hash refusal. Add
the U3c test row "two incumbent runs in one session refuse".

**M2 (major) — The chat timeout is operator-tunable and unrecorded, and the successor puts it on
a bar path.**
- `RunConfig.requestTimeoutSeconds` defaults to 120 s, is set by `--request-timeout`
  (`cli.py:529`) and is in no fingerprint field (grep: `runner.py:130` and its call sites only).
- **FR-10.** A new-ability hand-off chat failure now enters the mean as `y_i` = elapsed, which
  is at most the timeout (§3.3; note §Q4.3 row 3). A shorter `--request-timeout` turns slow
  hand-offs into short fail-closed items, which lowers the mean.
- **FR-2.** A guard chat timeout drops the item from both arms. If R > 1, dropping a pair with
  `x ≈ y` raises R (Appendix A). So a short timeout raises R.
- `decisionTimeoutSeconds` was introduced precisely so that the decision call's timeout is pinned
  (note §Q4.3). The chat side of the same composite is left free.

*Fix (§3.2/§3.3, U1d/U2, U3c):* on decision-point packs, the incumbent arm and the hand-off call
use one chat timeout that the operator cannot set. Either pin it (a cited constant in
`decisions.py`, or a candidate field) and refuse `--request-timeout` on these packs, or record it
per run and have `compare` refuse arms that differ. The value is `data-scientist`'s call (open
question 1).

**M3 (major) — U1e can land before the code it edits exists.**
- U1e "populate[s] them in `runner._build_fingerprint`" (§4.2). The decisions branch of that
  builder is U1d's, and the hand-off branch, which inherits the fields, is U2's. Today
  `_build_fingerprint` sets runtime fields only on `chat` (`runner.py:770-772`).
- §4.3's graph has `PR → U1e → U7` but no `U1d → U1e` or `U2 → U1e`. The serialization line and
  §4.2 say only "if U1d or U2 is in flight". The probe is queued now
  (`typed-decision-models-coordination.md`, U1a-probe row), so U1e landing first is the likely
  case, not an edge case. It would leave U1d and U2 with briefs that never mention
  `runtimeName`.

*Fix (§4.3, §4.2):* add `U2 → U1e`. Alternatively, split U1e: its fixture/HISTORY half has no
code dependency, and its route/fingerprint half follows U2. U1e's brief gaps are m3.

**M4 (major) — Replicate-mode `compare` leaves the arm set and the generic tables undefined.**
- §3.6 defines groups only "for a candidate `armKey` in `--models`". With `--session` and no
  `--models`, `_select_arms` returns every run in the session as its own arm (`cli.py:200-221`),
  so each replicate becomes a separate arm.
- `compare_report` "receives `replicates`", but the plan never says what its existing `runs` is.
  If replicates enter `runs`, the generic paired tables run `stats.verdict`/Holm across them.
  Note §Q3.1 and §3.6's own constants line forbid that for decision bars.
- `rank` dedups by `modelKey` (`cli._select_rank_arms`), and `rank_report` raises
  `DuplicateModelInReport` (`report.py:1741`). Two candidates share `modelKey` `laya`.
  `--negative-control` duplicates `candidates[0]`. Neither is addressed in v1.3 or here.
- No unit or function owns the load-count computation, and "the decision model's key" names no
  fingerprint field. The plan says "the report infers nothing", so this has to live somewhere
  named.

*Fix (§3.6, U3c):* (a) a decision-point `compare` requires `--models`, or groups every candidate
`armKey` in the session itself; (b) state `runs` = the chat arm plus the first replicate of each
group (or none), with no `stats.verdict` table over candidate arms; (c) `rank` and
`--negative-control` refuse on a decision-point pack; (d) name the function, e.g.
`cli._replicate_groups(...)`, reading `fingerprint.modelKey`, with its U3c test.

**M5 (major) — K is not constant across a replicate group under U7's own order.**
- §3.6 says "K is unchanged: replicates share `candidateContentHash`". But v1.3 §3.5 computes K
  per run, over hashes with `startedAt` strictly earlier, excluding the run's own hash.
- U7 runs replicate 1 of each candidate, then replicate 2. So candidate D's replicate 2 starts
  after candidate H's replicate 1, and D gets K = k₀ on replicate 1 and k₀ + 1 on replicate 2
  (Appendix B).
- The reuse line is printed "beside that arm's label", once, while the reuse-adjusted limit
  (§Q3.5) is printed per replicate. So the plan's claim is false, and which K to print is a
  guess.

*Fix (§3.6, U3c, §5):* "A group's K is the K of its earliest run, printed once per arm and used
by every replicate's reuse-adjusted limit." Add a U3c row: two candidates interleaved
D1, H1, D2, H2 give D's K equal to D1's on both replicates.

**M6 (major) — A guard decision-call failure is kept, but how it reaches the FR-1 line is
unspecified.**
- §3.3 gives the item `scoreable={}`, and `pair_items` makes it concordant. That leaves four
  things open:
  - **Which cell?** The candidate declares none, so the cell must come from the incumbent's
    declared metric. That is inference at pairing time. AGENTS "declared, never inferred" binds
    `report.py`, and this rule needs stating as the sanctioned exception.
  - **`candidate e₁/n`** in the note §Q3.4 line must come from the paired set. The candidate's
    stored aggregates exclude these items (`classification.aggregate` groups by declared metric),
    so the two disagree by `decisionCallFailures`.
  - **Side-check items** with a failed decision call are not covered, and neither is whether a
    failure counts toward `handoffShare`.
  - **The disagreement line** (replicate i vs 1, "final answer and `decidedBy`") has no rule for
    an item with no `decidedBy`, or one withheld in only one replicate.
- **Ownership.** U1d adds `failed_call` to `classification.score_answer` and tests a kept
  `failedCall="decision"` item. U3b "maps `failed_call="decision"` to `scoreable={}`". U1d and U3b
  have no order between them (§4.3), so both units write the same branch.

*Fix (§3.3, §3.6, U1d/U3b):* one paragraph covering:
- the cell comes from the incumbent's declared metric;
- `e₁/n` is computed over the paired set;
- side-check failures are dropped and counted;
- a failure counts as not handed off;
- the disagreement denominator is items with a final answer in both replicates, with failures
  counted beside it.

Give the `failed_call` branch to U1d. U3b adds only the decision-point declared metrics.

**m1 (minor) — U7 step 3 under-counts loads for the second candidate.**
- U7 runs each candidate's replicate 1, then "unload…, then run replicate 2".
- With two candidates sharing `laya`, the first replicate-2 run reloads it. The second candidate's
  replicate 2 then starts with `laya` resident, does not count, and prints
  `not assessable (only 1 load…)`.
- This fails safe, but every point (two candidates each) would lose a session.

*Fix (§4.1 U7):* "Before **each** candidate's replicate 2, unload the decision model and confirm
with `lms ps`."

**m2 (minor; method → `data-scientist`) — The "one load per run" premise does not hold under JIT.**
- Note §Q6 rests on "residency withholding keeps every item inside one load". The between-item
  probe (`runner.py:367`, `:405`) withholds only the one item whose probe found the model absent.
  Under JIT that item's call reloads the model, and the later items run on a new load.
- The re-check (§3.4) catches this only if `p` differs on the one fixed input. It is issued with no
  residency probe, so after an eviction it JIT-loads and compares across two loads.

*Suggested (§3.4):* probe residency before the re-check. If the model is absent, or any candidate
item was withheld for `"load"`, set `basis="assumed"` without calling. `data-scientist` confirms
whether such a run still counts as a load.

**m3 (minor) — U1e's brief omits four steps and under-states its schema condition.**
- **Runtime capture.** The decisions `warm_up` returns `runtime=None` (`lmstudio.py:841-846`), and
  `_runtime_identity` is called only on `chat` (`runner.py:846-848`). Both must change.
- **The trip-wire must stay chat-fed.** `check_attestation_staleness` compares the last-observed
  `runtimeName`. Feeding it the decision model's runtime would alternate with chat runs and exit 5
  as stale.
- **Pins flip.** `test_fingerprint.py:832-838` pins `runtimeName`/`runtimeVersion` as forbidden
  on both decision profiles.
- **Schema condition.** It must cover `model:handoff` (it inherits `_DECISIONS_ADD`) and any
  stored record, such as a smoke `run --candidate` or a live U2 check, not only U7's. The rule is
  consistent with `fingerprint.py`'s "validated against the contract it was written under", and
  the stored-records test (`test_results.py:1532`) is the check that enforces it.

*Fix (§4.2 U1e, R-14):* name all four.

**m4 (minor) — Closure records: per point or per candidate?** v1.3 U7 writes one record per
*closed point*, at `typed-decision-models-<role>-report.md`. The successor gives "a missed bar"
that record. Each point measures two candidates, so the cases "both miss" and "one misses while
the other meets" collide on one path. *Fix (§4.1 U7):* one report per point when it closes or goes
to U8, carrying every measured candidate's replicate block and fit-outcome line. A miss on a point
that another candidate meets gets a `HISTORY.md` line only.

**m5 (minor) — Traceability is not updated.** v1.3 §7's rows for FR-4, FR-11a and AC-1/6/12 still
cite only `bar_holds`, U1c, U3c and U4. FR-11a's two loads now live in §3.6's groups and states and
in U7. FR-4's "every fitted candidate, both loads, counted once" lives in U6, U7 and M5's K rule.
*Fix:* add a five-row traceability delta in §4.

**m6 (minor) — Two places where v1.3 text survives and contradicts an amendment.**
- **The withholding passage.** §3.3 replaces only v1.3's "withholds the item". The next two
  sentences, "**It is not handed off.** The bench measures the measured path; production's
  fallback on error is a robustness path", still stand. The bench now models that fallback for
  FR-1/FR-2. Widen the replace to the whole passage.
- **`decision_stats` API.** §3.6 lists `speedup_ratio(x, y, *, seed, B)` with no `levels`, but
  v1.3 §3.6 makes `levels` required. Under "the successor wins" this silently drops it. Add
  `levels` (required), or say it is pinned with a §Q4.1 citation. `parity_cell(...)` has no
  signature at all. Give it one.

**m7 (minor) — Candidate validation is looser than its closed sets.**
- `fit.qualityTarget` is validated only for type. Validate it against a role-keyed closed set
  (guard: `D_c<=0`, `mcnemar-nonreject`; new abilities: `T=0.94`…`T=0.90`, `max-min-agreement`).
- §5's U1c row omits NaN, ±inf and `bool` for `decisionTimeoutSeconds` (the `finite` clause; AGENTS
  NaN precedent).
- `HANDOFF_FAILURE_BY_ROLE` has no consumer in U1c. Its value-move mutation can only redden on
  U2's behaviour test. Say that U1c pins the keys (bound to `BAR_KIND_BY_ROLE`) and that U2's
  withheld-vs-kept test is the value pin.

**m8 (minor) — Handed-item details with no owner.**
- The `ItemTiming` docstring delta (§3.3) is named in no brief. Give it to U1d.
- These note §Q1 items are not in U5's brief:
  - generating `priorTurn`, with display names under the speaker rule, at the 40/60
    opening/follow-up mix (§Q1.5);
  - the "> 3% vote-tie drop → fix the rubric before drawing eval" stop rule (§Q1.2);
  - the labeler diagnostics, unanimity and intent agreement (§Q1.2);
  - a thread-position item field for the exploratory per-position agreement. It must sit outside
    `INPUT_FIELDS_BY_ROLE`.

**m9 (minor) — Restated facts and one slip.**
- §3.5 restates note §Q1.3's sizes ("guard 400 eval (+ 85 side-check), join-in 200, message type
  300"). Cite the table.
- §2's falkor-chat bullets repeat note §Q4.3's evidence.
- §1 lists the ledgered fix round as "U1a-m1…m5". The ledger holds m1..m4, and m5 is U1e's
  (`typed-decision-models-coordination.md`).

**Nits.**
- §3.4 says `LoadResult`'s default keeps "its six construction sites" unchanged. There are five:
  `lmstudio.py:841/859/867`, `test_runner.py:165` and `test_cli.py:1032`. The sixth grep hit is a
  docstring.
- §3.2 says "three required fields" and shows two top-level keys.
- Note item 3 says "every decide call". §3.2 exempts the warm-up, which is right because of the
  cold load, but undeclared. Say so, and say the re-check compares against the mandatory warm-up,
  not a `--warmup N` extra.
- `combine_replicates` labels a same-load split "unstable across loads".
- v1.3's "not assessable: arms from different sessions" is unreachable once `--session` is
  required.
- State `answer is None` iff `failed_call is not None`.
- Note item 1 groups by `candidateContentHash`. §3.6 groups by `armKey` and refuses mixed hashes,
  which is stricter. Say that it is a deliberate deviation.

### What's solid

- **Absorption is complete.** I checked each of the 11 handed items against the note's text:
  - replicates (§3.6/U7);
  - failures (§3.3);
  - the timeout source (§3.2);
  - `not assessable` (§3.6);
  - the FR-1 line (§3.6);
  - `priorTurn` (§3.1/U4/U11);
  - `decision-models/` (§3.5/U5);
  - fit outcomes (§3.2/§3.5/U6/U7);
  - minimums printed (§3.6);
  - the re-check and `inputTokens` (§3.4);
  - per-label `minConfidence` (§3.8/U6a).

  Each has a section and an owning unit. Both carried items are in place: U1c refuses `score`, and
  U1d adds `decided_by`/`probability`/`failed_call` with a test that the old pack writes no
  `decidedBy`.
- **The built-code deltas are accurate.**
  - 33→35 and 45→47 follow from `_DECISIONS_ADD`/`_HANDOFF_ADD` (`fingerprint.py:142-172`).
  - `LoadResult` and `_WARMUP_DECISION_INPUT` are as described (`lmstudio.py:65`, `:363-372`,
    `:839`).
  - The `ItemTiming` docstring is quoted correctly (`results.py:227-230`).
  - The `runner.py:393-398` mapping and `Pack.seed` (`packs.py:118-129`) are as cited.
  - `fitOutcome`/`fitQualityTarget` are additive in schema 1, because `test_results.py:1532`
    validates every stored record and none carries either profile.
- **Every cited falkor-chat fact checks out:** `_safe_respond`'s swallow, `_drive`'s re-raise,
  `read_thread`'s location and `_recent_turns`' filter-then-slice.
- **The `--session` requirement and the two-hash refusal** close candidate-side selection cleanly.
- **The bar-state algebra is monotone.** `combine_replicates`' four rules never turn a `not met`
  or a one-load result into `met`, and §5's U3c row covers the decisive cases.
- **U1e and U6a are collision-free** in this plan's scope. v1.3 has neither id, and the other
  `U6a` is in the archived K-030 ledger, as §4.2 says.
- **`HANDOFF_FAILURE_BY_ROLE` as a pinned table** is the right shape. It puts the runtime
  asymmetry (guard fails the run; new abilities fail closed) in data, not prose.

### Open questions

1. **`data-scientist`, for M2:** what chat timeout should a decision-point pack pin? Either the
   runtime's guard/agent-kind provider timeout, so that hand-off failures are runtime-faithful, or
   a fixed bench value. Also, does the incumbent arm share it?
2. **`data-scientist`, for m2:** should a candidate run with a mid-run `withheldFor="load"` item,
   or with a residency miss before the re-check, still count as one load?

### Appendix

**A — Why dropping a pair raises R (M2).** Let R = X/Y > 1 over the paired set, and drop a pair
with x_k = y_k = a > 0. Then R' = (X − a)/(Y − a), and R' > R ⇔ Y(X − a) > X(Y − a) ⇔ X > Y. A guard
item whose hand-off chat call times out has y_k ≈ t_d + x_k, so y_k ≈ x_k. With the incumbent
also slow on it, a shorter timeout drops it from both arms and moves R up.

**B — K under U7's order (M5).** Two candidates per point, D (decision only) and H (hand-off),
with prior count k₀. Each K counts distinct earlier hashes, excluding the run's own (v1.3 §3.5).

| Run (in order) | Earlier distinct hashes, own excluded | K |
|---|---|---|
| D1 | prior | k₀ |
| H1 | prior + D | k₀ + 1 |
| D2 | prior + H | k₀ + 1 |
| H2 | prior + D | k₀ + 1 |

D's two replicates disagree on K. H's agree only by accident of order.
