# model-bench live spec (L6) — implementation review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (live-specs L6)

## Scope & verdict

**Reviewed:** `model-bench/docs/DESIGN.md` (11,494 words), `model-bench/docs/FORMATS.md` (3,505), the rewritten
`model-bench/README.md` (614) and `model-bench/AGENTS.md` (1,114), the Component-docs row in root `AGENTS.md`,
and the test change in `546bb914` (`tests/test_runner_decisions.py`). **Baseline:** the shipped code at
`546bb914` (pinned sha; the L6 files are the uncommitted working tree), per root `REVIEW.md`: a spec update is
checked against the shipped code, not the plan. This is model-bench's AC-1 check and typed-decision-models' FR-7
check (`docs/plans/live-specs.md` §3.6 Timing). Read the unchanged lines too: every section of both spec
documents was read in full, not only the typed-decision-models parts.

CPG: considered, not relevant — a documentation-versus-code check of a stdlib Python component with no loaded
`cpg_modelbench` graph; claims were verified by running the CLI/scripts and reading source.

**Verdict: approve with suggestions.** No blocker: every command-derived claim I re-ran held, every exact
constant and field set I diffed matched the code, the §3.5 code/contract surfaces are all indexed, and AC-3
sweeps are clean. Two **majors** are real spec-vs-code contradictions (F1, F2) and one is a false "every shipped
file" invariant (F3); all are small text fixes. **Counts: 0 blocker · 3 major · 7 minor · 3 nit.**

**Checks run (verified, not inferred):** `./run.sh --help` (7 commands); `validate --strict` (traceback, exit 1);
`validate` on a good and a missing pack (0 / 4); bogus command (2); `--help` (0); full suite
`.venv/bin/python -m pytest -q` from `model-bench/` → **3655 passed, 5 deselected, 2 xfailed** in 21.6 s;
`awk 'length($0)>700'` over `model-bench/AGENTS.md` and root `AGENTS.md` → no hits; AC-3 grep (the §5 pattern)
on all four files → zero hits, zero `K-` refs; a scripted `§n` resolver over all four documents → every pointer
resolves to a real heading; `git ls-files model-bench` diffed against DESIGN §11 (F3); the fingerprint field
sets of all five profiles diffed against `fingerprint.REQUIRED_BY_SCHEMA` (names, tiers and counts 30/26/36/48/11
all match); the 35 stored records and the 130-record `results/runs/` partition analysed with a script (below).

## Findings

### F1 — major — DESIGN §3.6 says a report "never refuses" on a timeout disagreement; §2.3 refuses it (exit 2)

*Verified (code read + both copies).* DESIGN §3.6 closes with "A report states, never refuses, when stored records
disagree on the timeout (§9.4)." DESIGN §2.3 lists, among the exit-`2` refusals of a decision-point `compare`
(nothing written), "arms whose `requestTimeoutSeconds` differ", and `cli._decision_point_arms` does exactly that
(`cli.py:392-403`, `DecisionPointUsageError` → `EXIT_USAGE`). The report-side `_TIMEOUT_REASON` only fires when
every arm shares one non-180 value (`report.py:2717-2727`). §9.4's "never change the exit code" is right only for
that same-value case. A reader trusting §3.6 would expect a report where the tool exits 2 and writes nothing.
**Fix:** reword §3.6's last sentence to: differing timeouts across the arms of a decision-point `compare` are an
exit-`2` refusal (§2.3); a shared non-180 value is stated in the report (§9.4); and make §9.4's "never change the
exit code" sentence say the same (a fact in two sections is one home plus a pointer).

### F2 — major — DESIGN §9.4 overstates the comparison-wide `not assessable` state

*Verified by reading `_comparison_block`, `_render_replicate` (`report.py:2255-2300`) and the group combiner
(`:2660-2700`); not run against a constructed record.* §9.4 says the comparison-wide state "applies to every bar
of every replicate" and lists "an incumbent **or replicate** whose aggregates disagree with its items". The code
differs in two ways: (a) only the **incumbent's** inconsistency is comparison-wide (`:2724-2725`); a replicate's
inconsistency is that replicate's own state (`:2276-2278`) and its group's overall line
(`:2689-2693`), and does not block another candidate's replicates; (b) an `assumed` replicate takes
`not assessable (load changed mid-run)` and ignores the comparison-wide state (`:2276`: `state = LOAD_CHANGED if
kind == "assumed" else blocked`, then `reason = state or blocked`; same at `:2497` for the new-ability kind). **Fix:** state the incumbent case as
comparison-wide, the replicate case as per-replicate/per-group, and say an `assumed` replicate keeps its own
load-changed state (already half-stated in §7.6) rather than the comparison-wide reason.

### F3 — major — DESIGN §11 claims "every shipped file" and omits tracked files

*Verified: `git ls-files model-bench` built independently, collapsed by directory, compared with §11's rows.*
§11's lead says "Every shipped file and where its contract lives", and the plan (§3.5) and root `REVIEW.md` require
the inventory be built from every shipped file. **No row** for: `README.md`, `AGENTS.md`, `docs/DESIGN.md`,
`docs/FORMATS.md`, `docs/BACKLOG.md`, `docs/HISTORY.md`, `docs/manuals/small-model-benchmarking.md`, the
`docs/{plans,requirements,reviews,test-plans,test-reports}/` trees and their five tracked `.gitkeep` files, and
`modelbench/scoring/__init__.py` (the `modelbench/scoring/` row names six scorers, not the package marker).
Everything contract-bearing (the other 22 `modelbench/` files, 8 packs, candidates, results, reports, scripts, shell scripts,
fixtures, `pyproject.toml`, `.gitignore`, `footprints.json`) **is** indexed, so no contract is lost; but the stated
invariant, and the "add a row in the change that adds a file" rule, are unusable while it is false.
**Fix:** add rows for the context files/docs tree ("not a contract; see root `AGENTS.md` documentation
convention") and fold `scoring/__init__.py` into the scorer row; or soften the lead to "every shipped file that
bears a contract" and list the exempt classes. Prefer the former, which the `git ls-files` check can verify.

### F4 — minor — latency figures: the spec omits two behaviors of the shipped code

*Verified (`runner.latency_block`, `results._index_row`).* (a) DESIGN §3.5 says the TTFT / prefill / tokens-per-s
medians "come only from calls with usable `stats` and a positive prompt-token count"; the code also gates all
three on the 45%-of-attempted-calls coverage floor (`runner.py:1344-1356`), so they are `None` below it. (b)
FORMATS §4.3 says the `latencyMsP50`/`latencyMsP95` index columns are "computed over the items' admitted wall
clocks" and `headlineMetrics` lists "`name=k/n` …"; the code computes them with plain `percentile` **bypassing
both coverage floors** and the max-rename (`results.py:1194-1195`, the in-code comment says so), and the column
holds **every** named metric, not only headline ones (`:1173`). A reader would expect the index to agree with the
report's `LatencyBlock`. **Fix:** add the floor to the §3.5 medians sentence; state in §4.3 that the index latency
cells are ungated medians/p95 and that `headlineMetrics` carries every aggregate metric.

### F5 — minor — `nlq-generator`: an uncaught `TypeError` is a shipped behavior the spec is silent on

*Verified: `tests/test_tools_exec.py` carries 2 XFAILs for it (suite output above), `docs/BACKLOG.md` lines
21-31 describe it, `scoring/extraction.py:241-262` catches only `MalformedSpecError`/`SchemaViolationError`.*
`tools/exec.py`'s `order_by` raises `TypeError` on a `None` sort key; DESIGN §6.4 ("Three gates, each with its own
recorded failure class") and FORMATS §1.4 present the executor as total. A model-generated spec ordering by a
sparse property would escape the three gates (I did not run a live reproduction; no shipped golden item reaches
it). **Fix:** one sentence in §6.4: "a spec whose `order_by` key is null on some row raises `TypeError` past all
three gates; open, `docs/BACKLOG.md`" (no `K-` id inside the spec; a pointer to the document is allowed).

### F6 — minor — "the only `0.05`" (DESIGN §8.3, §10) is false as stated

*Verified by `grep -n '0\.05\b' modelbench/*.py`.* §8.3: "`ALPHA_FAMILY` (`0.05`) is the only `0.05` in the
comparison code"; §10: "`ALPHA_FAMILY` stays the one `0.05`". `decision_stats.FR1_ALPHA: float = 0.05`
(`decision_stats.py:33`) is a second literal, and §8.5 itself lists "α `0.05`" among `decision_stats`' own
constants. The sentence is carried over from the old `AGENTS.md` and was already inexact. **Fix:** scope it ("the
only `0.05` in `stats.py`'s verdict path; the decision-point bars have their own `FR1_ALPHA`") or make
`FR1_ALPHA = ALPHA_FAMILY` an actual requirement; either way a reviewer grepping the rule must get a true answer.

### F7 — minor — §2.10 operator-tooling contracts omit flags the scripts accept

*Verified: `--help` of all four scripts.* `refresh_golden.py` also takes `--repo-root`, `--api-base-url` and
`--timeout` (default 300 s, `--embed-corpus` only); `import_decision_pack.py` takes `--root`. The rows list modes
and exit codes (verified: unknown pack → exit 1 naming the three known packs; `consolidate_sweep_reports.py`
missing file → 2, no marker → 1; `import_decision_pack.py` no args → 2; `run-sweep.sh`'s exit 42 / curl probe /
log format match FORMATS §8.3) but not these flags. **Fix:** add them to the two rows.

### F8 — minor — FORMATS never states the per-item `provenance` object

*Verified (`chat-responder` row: `{draftedBy, verifiedBy, corpusVersion}`).* The old README described the per-item
`provenance` object (`origin`, `originPath`, `originGitSha`, `copiedAt`, `draftedBy`, `verifiedBy`,
`corpusVersion`); the cut moved the pack-level contract to FORMATS §1 but the item-level shape appears nowhere
(FORMATS §1.3 only lists `provenance` as a field name; §1.2 says the manifest's is "read by no module"; DESIGN
§5.2 says validate ignores it). **Fix:** one row in FORMATS §1.3 or §1.1 stating the object's keys as the shipped
packs carry them (note they differ per pack; the old list was already broader than the shipped data) and that no
code reads it.

### F9 — minor — root `AGENTS.md` Structure bullet contradicts the new spec

*Verified (`AGENTS.md:61-64`).* "Delivered and closed (S0–S8): five packs across all five FR-21 roles" and
"Requirements/plan live at the repo root, not `model-bench/docs/`" sit beside the new DESIGN/AGENTS ("eight
shipped packs covering seven roles", three decision-point packs, `model-bench/docs/requirements/` exists). Not an
L6 edit, but L6's close is where it surfaces. **Fix (for `teco` at close):** replace the sentence with the current
pack/role count and a pointer to `model-bench/docs/DESIGN.md`.

### F10 — minor — stale `AGENTS.md`/README prose to confirm at close, and `HISTORY.md` pending

*Verified.* The brief's L6 file list includes `docs/HISTORY.md`; `git status` shows it unmodified. Per root
`AGENTS.md` (close step 5) the entry is written last, citing the updated sections and this check, so this is a
reminder, not a defect. The new `AGENTS.md` correctly drops "never store a `model:decisions` record before U7"
(35 stored records now exist; see Observations).

### N1 — nit — `match="requestTimeoutSeconds|assert"` in the planted-field test is near-vacuous

*Verified (`tests/test_runner_decisions.py`, `546bb914`).* `pytest.raises(AssertionError, match=…)` passes on any
assertion text containing "assert" (pytest rewriting always emits it). The test is still discriminating because
both assertions in the helper would fire on the planted `120.0`, but it does not say which. Match on
`"requestTimeoutSeconds"` alone (the first assertion's message is `path.name` plus the rewritten expression).

### N2 — nit — "older pack" names a pack by age (DESIGN §6.3, FORMATS §1.3, §1.5)

Use `guard-judge-understanding` (the pack's id). Not history language under AC-3, but a time-relative name drifts.

### N3 — nit — undefined requirement ids

`AGENTS.md` l.22 "the five FR-21 roles" and the spec's `FR-1`/`FR-2`/`FR-10` are not defined in the spec set (DESIGN
§7's intro does cite `docs/requirements/typed-decision-models.md`). One pointer clause for `FR-21` (small-model-
benchmarking requirements) would close it.

## Required checks, by number

**(1) AC-1, spec vs shipped code.** Command-derived claims re-run, all confirmed: seven commands; `validate --strict`
raises `NotImplementedError` (traceback, exit 1); `run --reference` is accepted and never read (`referenceKey`
appears only in `RunConfig` and its construction, `grep -rn referenceKey modelbench`); the `residentModelsAtEnd` exit-5
message does name `results/transcripts/` (`runner.py:2082`) although nothing writes there (DESIGN §3.8 is right, the
message is a code defect, see Observations); plain `compare` takes `runs[0], runs[1]` (`report._comparison_pair`) over
file-name order (`load_history` sorted glob; run ids sort by arm key then start); three decision-point packs (script
over `pack.json`: guard, join-in, message-type); **"35 stored decision records" is not what the data shows**: 130 stored
records, of which 35 carry `requestTimeoutSeconds` (17 chat arms of decision packs at 180 s + 8 `decisions` + 10
`handoff`); the candidate records are 18. No spec document states "35", so no text is wrong.
Typed-decision-models surfaces traced line by line against `runner.py`/`cli.py`/`decisions.py`/`decision_stats.py`/
`lmstudio.py`/`packs.py`/`hostinfo.py`: capture order steps 1-11 (`run_pack`), the explicit-load rule and
co-residency refusal, instance-config refusal paths, warm-up order and trip-wire outcomes, `_between_item_residency`
(re-probe, empty snapshot, stderr line), `_within_load_basis` (no-call and by-construction/assumed branches,
fail-safe), the four per-item flows incl. over-limit (`500` + case-sensitive marker), kept decision failure,
`withhold`/`fail-closed` hand-off failure and the runner-added metrics, `apply_policy` boundaries, `Candidate.load`
refusals, fit-manifest format and independent hash/count checks, leakage refusal on both strata, `input_hash`/
`eval_set_hash`/`question_spec_hash`, `combine_replicates`, reuse count K, replicate kinds, comparison arm
selection and every §2.3 exit-2 case, rank refusal on decision packs. **All matched** except F1/F2. Constants matched:
`MIN_EVAL_ITEMS 200`, `FR2 2.0`, `FR10 0.90`, `350`, `180`, default `120`, `300`, method minimums 50/52, 10 000
resamples, 0.1 pp grid, 0.80 power, corpus 121 docs, `nlq` 34 of 40 answerable, guard pack 85 side-check, per-turn
caps 400/2000, coverage floors 45/90. FORMATS §3 required-field sets per profile, §5.1 candidate keys and enums,
§1.2 manifest keys, §4.3 columns all match the code.
**Load basis** is stated by construction where it is (single-call, conversation with determinism probe) and as
"assumed" where the code fails safe; no overclaim found. `requestTimeoutSeconds` null-on-legacy is exact: of 130
stored records, 95 end before 2026-10-31 and none carries the field; 35 end after and all do.

**Test change (`546bb914`).** The cutoff cannot make the scan vacuous: 95 legacy records qualify and the real-data test
asserts `> 0`. It still guards the load rule: I recomputed the partition for seven cutoffs. Any cutoff in
[2026-09-21, 2026-11-19] is green (95 legacy, 0 carrying the field); `2026-01-01` gives 0 legacy (the `> 0` guard
reddens); `2026-11-20` and `2026-12-31` pull in 13 / 35 field-carrying records (the `not in body` assertion
reddens). A `from_dict` that defaulted the field to 120 fails the second assertion on every legacy record. The
planted-field test exercises the helper's refusing side (see N1). Residual: the narrowing now leaves records ended
on/after the cutoff un-scanned for this rule, which is correct (they may carry it) and unobservable today.

**(2) §3.5 inventory.** Built independently (F3). All 23 tracked `modelbench/` files are indexed except `scoring/__init__.py`; docs/context files have no rows.

**(3) AC-3 and live-spec rules.** Zero hits for the §5 pattern in all four files; zero `K-` references; header
blocks conform (`Status: active`, `Owner`, `Tracks: —`, no `Version:`/revision note); both documents are indexed and
numbered with a stability note; HISTORY/BACKLOG are cited, not restated. Duplicated facts (180 s pin in §2.9/§3.6/§7/§9.4,
exit codes in §2.2 and §3.2, 200-item floor in §7/§7.1/§7.5, newest-run-per-`armKey` in §2.3/§2.4) were each checked
in every copy: all agree with the code except the §3.6/§2.3 pair (F1). The spec delegates the statistical
formulas and verdict strings to `docs/plans/small-model-benchmarking-ml.md` and `docs/plans/typed-decision-models-ml.md`
(both `archived`, both resolve); that is the plan's §3.6 design and the docs say so up front; I flag it only as a
dependency to keep in mind: if either note is ever edited or removed, the spec loses its formulas.

**(4) Pointers and the cut.** Every README/AGENTS pointer resolves to a real heading (scripted resolver plus manual
read), including AGENTS's `DESIGN.md §n`/`FORMATS.md §n` citations; AC-2 `grep -n 'live spec' model-bench/AGENTS.md`
names the set (l.11). Diff against `546bb914:model-bench/{README,AGENTS}.md`: contract text moved to the spec with
nothing contractual lost except F8 (item `provenance` shape) and the dropped `docs/requirements/small-model-
benchmarking.md` pointer in the README (still in AGENTS's documentation map and root `AGENTS.md`'s row: fine). The
new README also **fixes** the old step 4 (`validate --pack … --strict`, which raised). AGENTS: 1,114 words (old 2,495),
no line over 700 characters, history language removed ("S0–S8 closed", "never store a record before U7", per-stage
rationale). Root `AGENTS.md` row change is accurate and matches falkor-chat's row shape.

**(5) Contradictions across the set and the manual (not edited).** Within DESIGN/FORMATS/README/AGENTS the only
contradiction found is F1 (plus F6's pair inside DESIGN). The unchanged manual
`model-bench/docs/manuals/small-model-benchmarking.md` (Last updated 2026-09-20, never touched by L6) is stale in
four places, for `tico`:
1. l.107 describes `run --reference <key>` as marking "a second model … you're comparing against": DESIGN §2.9 says it is
   accepted and ignored.
2. Nothing about decision points: no `run --candidate`, no `handoff`, no `compare --session --models` requirement, no
   `rank` refusal on decision-point packs (`grep` for `candidate|decision|handoff|synthetic` finds only unrelated hits).
3. l.172 "Leave off `--models` and it picks from whatever's stored" and §5a's three-model `compare --session` omit that
   the verdict covers only the **first two** arms in file-name order (the arms table lists all); §7a's "`compare` is
   always a two-arm comparison" is false for a decision-point `compare`.
4. The exit-code FAQ omits the decision-point causes (failed explicit load, config unreadable, off-pin timeout).

## What's solid

- Accuracy is high where it matters: the typed-decision-models behavior (surfaces, refusals, pairing, replicates, load
  rule, residency probe, fit manifest, leakage) matches the code point for point, including the sharp edges
  (fail-safe `assumed`, kept-not-handed-off failures, the 180 s pin applying to hand-off and incumbent calls).
- The document split (DESIGN read whole, FORMATS by lookup), the index tables, stable numbering, and the README/AGENTS
  reduction to pointers are well executed; AGENTS keeps only invariants and rules a change can silently break.
- AC-3 hygiene is clean; the spec cites its homes (HISTORY/BACKLOG) instead of narrating.
- The `546bb914` test change is sound and does not weaken the load rule's guard.

## Open questions

1. F3: add the docs/context-file rows (preferred) or narrow the "every shipped file" claim? Either is fine; pick one
   before the check re-runs.
2. F9: who owns the root Structure-bullet edit at close (`teco`, since root `AGENTS.md` is shared)?

## Observations for the backlog (not spec findings)

- The residency-at-end exit-5 message says "transcript retained at `results/transcripts/<runId>.jsonl`" (`runner.py:2082`)
  but no code writes transcripts (DESIGN §3.8). The message misleads an operator; fix the string or file an item.
- `refresh_golden.py --check-origins` currently reports `DRIFTED` for the `nlq-structured-query` pack's origins
  (`falkor-chat/scripts/seed_catalog.sh`, `…/querygen.py`) and exits 1 for the embedder pack: data state, not a
  spec claim; no action for L6.

### Proposed REVIEW.md lines

**Target: root `REVIEW.md`, heading "Always check"** (addition; source: F1, F6, and the recurrence of absolute
claims carried over from the old `AGENTS.md`; no prune is warranted, the nearest existing line, "a fact stated in
two sections", does not cover single-copy absolutes):

- In a live-spec check, each absolute word in the spec (*only*, *never*, *always*, *every*, *exactly one*, *the one*)
  is verified by an unfiltered grep or enumeration over the whole component, including modules added after the
  sentence was first written. A claim checked only against the module it was written about is a bound, not proof.

## Pass 2

**Verdict: approve.** No new finding at major or above. The close may proceed.

**Dispositions (verified against the tree, not re-argued):**
- F1 fixed: DESIGN §3.6, §2.3 and §9.4 now agree (differing timeouts across a decision-point `compare` = exit-2 refusal; a shared non-180 value is stated in the report); matches `cli.py:392-403` and `report.py:2717-2727`.
- F2 fixed differently, and the rewrite is correct against `_comparison_block`/`_render_replicate`: comparison-wide = no incumbent (speed-up), incumbent inconsistency (speed-up only), any timeout ≠ 180 (incumbent included on new-ability), content-hash disagreement, last wins; a replicate's own inconsistency is per-replicate; an `assumed` replicate keeps load-changed unless its own aggregates disagree. Consistent with §2.3 and §3.6.
- F3 fixed: DESIGN §11 rebuilt against `git ls-files model-bench`; README/AGENTS, `docs/` documents, `docs/manuals/`, the five process trees (and their `.gitkeep`) and `scoring/__init__.py` now have rows; no tracked file is without one.
- F4 fixed: §3.5 gates the three stats medians on the 45% floor; FORMATS §4.3 states the index latency cells are ungated and `headlineMetrics` holds every metric (matches `runner.py:1344`, `results.py:1173,1194`).
- F5 fixed: DESIGN §6.4 states the uncaught `TypeError`.
- F6 fixed differently, correct: the unfiltered grep (`= 0.05` assignments in `modelbench/*.py`) finds exactly `stats.py:65` and `decision_stats.py:33`; the others are docstring/comment prose.
- F7 fixed: `--repo-root/--api-base-url/--timeout` and `--root` present.
- F8 fixed, verified against data: chat-responder `{corpusVersion, draftedBy, verifiedBy}` (30/30), tool-caller `{basedOn, draftedBy, verifiedBy}` (12/12), no other pack's items carry one.
- F9, F10: applied/pending as stated by the coordinator; `docs/HISTORY.md` entry still to be written last. Not re-checked beyond that.
- N1 fixed and judged adequate: `match=re.escape(f"{src.name}\nassert 'requestTimeoutSeconds' not in")` pins the first assertion specifically (a mutant dropping it changes the message and reddens); it depends on pytest assertion rewriting, acceptable in this suite. `ruff` clean.
- N2 fixed (no "older pack" in DESIGN §6.3/FORMATS §1.3; FORMATS §4.1 "older-schema record" is a legitimate schema term). N3 fixed (FR-21 and FR-1/2/10 now carry a pointer).

**Re-runs:** suite `3655 passed, 5 deselected, 2 xfailed` (22 s); AC-3 pattern plus `K-[0-9]+` over all four files: zero hits; `awk 'length($0)>700'` on `model-bench/AGENTS.md`: none (AGENTS 1,117 words); same-revision cross-check F1 vs F2 vs §9.4 vs §2.3: no contradiction.

**Residual (nit, no action required for close):** none new.
