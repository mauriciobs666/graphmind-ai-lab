# Connected code graph: P1 close-out pre-registration text (`ccg-prereg-v1` / `ac5-gate-1`)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1 close-out, CO-6b) · **Reviews:** `docs/reviews/connected-code-graph-p1-closeout-prereg.md`, `docs/reviews/connected-code-graph-p1-closeout-prereg-ml.md`

Human-readable reading of the record in `cpg/eval/preregistration.jsonl` (stratum `ac5-gate-1`, `kind: gate`, the last line). The jsonl line is the record and the single home of every verbatim string and hash (S1 wording, the withheld-shape statement, the six entry-criterion items, the pins); this note says what each field commits to. Relies on: `docs/plans/connected-code-graph-p1-closeout.md` (§1, §3.5, §10, CO-6b/CO-9), `docs/plans/connected-code-graph-p1-gate-marker-ml.md` (last section), `docs/plans/connected-code-graph-p1-closeout-ml.md` (recall ruling), `docs/plans/connected-code-graph-p1-ml.md` §5.3, §6.1, `docs/plans/connected-code-graph-eval-protocol-ml.md` (EP) §9.4, §11, §12. No held-out file and no custodian directory was read; held-out figures are counts already in those documents. No writer exists for an ac5 record (`ac24_smoke.py` writes only the AC-24 one), so the line was built with `ac24_smoke.observe_tree` and `canonical_json`.

## 1. What is pinned

| Field | Value (read at writing) |
|---|---|
| `algorithm` | `{id: bfs-v1, version: "1.2"}` as `bfs.algorithm_version()` emits; `defaultDepthBound` 2 |
| F (`coreCommit`) | `db7f7f45…`, an ancestor of HEAD. Every file changed under `cpg/` since F is under `cpg/eval/` or `cpg/tests/` (the withheld-scoring harness commits `4aa24d09` and `299e2760`), so `coreTreeHash` and `packSetHash` equal the values at F |
| Harness pin | 12 entries (`G.HARNESS_FILES`), every sha256 recomputed from the committed tree: `gate.py` `306bad50…`, `gate_run.py` `15982281…`, `withheld_shapes.py` `152f1285…` (harness Pass 3, `docs/reviews/connected-code-graph-p1-closeout-harness-pin.md`) |
| Key-error record | `ccg-dev-key-errors-v1.json` (not in `HARNESS_FILES`), sha256 pinned: 3 user-confirmed dev key-errors |
| Scan script | `cpg/eval/results/co6b/no_workaround_scan.py`, sha256 pinned |
| Decisions by hash | `ccg-noise-s1-v1`, `ccg-noise-recall-rule-v1`, `ccg-p1-flow-python-v2-decision` (flag false). `ccg-noise-ratify-v1` is the requirements-log entry of 2026-11-10 (`docs/requirements/connected-code-graph.md`, append-only, so no hash) |

**What moves F:** a tracked file under `cpg/` outside `eval/**`, `packs/**`, `tests/**` and `*.md` (`coreTreeHash`); a pack file (`packSetHash`); a harness file (its sha). A `cpg/tests/**` edit moves neither hash but `run_context` refuses it after R. Each needs a new F or record and a new rehearsal. Docs and ledger commits do not.

## 2. Scope, floors, the scored set

- **Gated:** `http` 26, `mcp` 46, `scripts` 79 held-out counting items (151); dev 18, 20, 49.
- **Withheld:** `flow` (99 held-out; dev 28, 1 counting; flag false) and `wiring` (8 held-out; 11 dev, 0 expressible; stakeholder decision) as strata; and the scripts items of the Q8 shapes, which the harness now neither asks nor scores. No recall or interval is reported for any of them.
- **Scored set.** Verdict, floors and bounds are over the counting items outside the withheld shapes: pooled 151 − w, scripts 79 − w. The 25% cap is checked over all 79 counting scripts items before exclusion (the harness's definition). The ledger line before the ask carries c, the total w and the run count; the per-shape-letter counts stay in the custodian's record and are not a builder input. The §3.5 statement of the close-out plan says this verbatim and is byte-equal to the record. Harness details (Pass 3): the predicate excludes a withheld-shape scripts item whether or not it is `counting`; only counting items enter the cap's numerator and denominator; a record with no `counting` key is non-counting everywhere; an impacted file containing a bare CR is not withheld (fail closed, the item is asked); an `EVAL_COMMIT` edit after the run is out of the threat model, since `EVAL_COMMIT` is in the harness pin.
- **Floors:** 75 pooled, 8 per gated stratum, over the scored set. Pre-marker floor refusal: when the exclusion removed at least one counting item and the kept pooled counting items are under 75 or a kept gated stratum holds under 8 items, `run` exits 5 before the lock, the build and the marker; the list is not spent; counts only on stderr. A list with nothing withheld is not judged there; `score` judges its floors as before (`minimums-not-met`, not a pass, not a miss). At the cap neither can occur (pooled at least 132, scripts at least 60, arithmetic from the plan's counts), so the refusal guards a miscount.
- **Cap:** 20 or more of the 79 counting scripts items withheld refuses the spend. Closed outcome set: (a) leave the list unspent and carry AC-5 open, or (b) one fresh draw of scripts items in this unit, at most one (new blind draw, new record, shape count and cap recomputed, the redraw and its count reported; the over-cap list's scripts items may not return as candidates; builders get no shape-letter count of either draw before its ask). Raising the cap, narrowing or editing the predicate, or building a shape and spending the same list after seeing the count are excluded.
- **Objection carried, not hidden:** the withheld class is defined by what the engine cannot do. The mitigations are the 25% cap, the per-shape counts, the predicate's build-invariance (pinned, written by someone other than the extractor builder), and a claim that says exactly what it covers.

## 3. Entry criterion

The six items are verbatim in the record. Readings:

1. Every dev item of a gated stratum is a hit, a withheld shape (counts beside) or a stakeholder-confirmed key-error. Arithmetic: scripts 49 − 10 − 3 = 36 hits; http 18; mcp 20. Observed by the rehearsal's dev probe.
2. **No-workaround scan** (committed script, sha pinned; definition in its docstring and the record; R must contain it): 473 dev literals (ids 261, paths 84, qnames 110, contract keys 18) against 137 files (`cpg/ccg`, `cpg/extractors`, `cpg/packs`, any extension, plus `cpg/eval/tools/withheld_shapes.py`): **3 literal hits, none in the predicate**. Item 2 is never called "met": the criterion's letter ("finds no literal") is unmet by the README hit; the three hits are disclosed and ruled not a workaround by the analyst, not excused: `.mcp.json` and `AGENTS.md` are product file-type names in code (since 2026-11-11 and a declared census glob), `test-stamp-wiring.sh` is prose in `cpg/extractors/shell/README.md`. The scan is literal-only; the synthetic-tree tests of close-out §7 cover the rest (the analyst ran them). The script also prints the predicate's flagged dev set (10: B 2, C 2, I 2, J 2, K 1, M 1) and, given the rehearsal probe, each item's engine status; a flagged item the engine lists is over-withholding on dev and is reported.
3. **Dev noise** at the rehearsal: labelled Q ≤ 4, pooled and scripts, on the point estimate, largest across reruns at the tree (§5). Possible precision, p90 burden and the S1 reductions are reported there and do not gate it; they are deferred, not waived, to P2's first gate, which enforces them.
4. Rehearsal rule: a miss means fix, new F, new record, new rehearsal, no ask; the dev result is never AC-5 evidence. The ledger shows no rehearsal at F yet; the report states the run count and any tuning against dev.
5. **Q4 screen: not adopted** (stakeholder: triage, fix, fresh draw).
6. **Freeze-time check:** the stakeholder allowed the custodian to run `split_manifest.py freeze-check` once (2026-11-25). It opens held-out probes, so not run at writing, and its path set ends at R. Output is counts always and a 12-hex probe hash with its path only on a hit (read in `main`); `--out` and `--probe-map` must sit outside the repo. No builder reads held-out material after R. A contact hit after the rehearsal moves a cluster to dev and changes c, the floors, the withheld counts and the dev gold hash: the rehearsal is redone and the stakeholder's once-permission renewed.

**Key-error precedent.** Two of the three dev key-errors (the G parent-directory `$ROOT` item, the H Python-module item) are resolver limits. A held-out analogue is scored as a miss; the dev rulings are not precedent for held-out triage.

## 4. Ordering

| Step | Kind |
|---|---|
| F (`db7f7f45`) | tree equality harness-enforced (four fields equal, `cpg/` clean); coreCommit ancestry procedural (not checked) |
| Record commit R | harness-enforced (committed, one record per id; only `cpg/eval/` and `*.md` change after). R contains the jsonl, this note, the §3.5 edit and `cpg/eval/results/co6b/no_workaround_scan.py`, and leaves `cpg/` clean (an untracked file under `cpg/` fails the preflight) |
| CO-9 dev rehearsal at R, with the labelled dev Q | hits, key-errors, gold and harness hashes, snapshot id: harness-enforced by the probe; noise bars, labelled Q, run count: procedural |
| Custodian `freeze-check` at R, once | procedural (no reference in the gate code) |
| Ledger line committing c and the withheld counts | procedural |
| One held-out run | harness-enforced (once-only marker) |

The custodian and `teco` require the rehearsal's labelled Q, the run count, the freeze-check result and the ledger line before `gate_run run`; the harness checks none of them. I took the reviewer's order (rehearsal before freeze-check): a rehearsal miss is a live outcome, so checking afterwards spends one held-out read on a record that survived. The cost is stated: a contact hit after the rehearsal changes the dev gold hash, the probe then refuses, and the rehearsal is redone.

**Successors** (record ordinal `k` in `ac5-gate-1-r<k>`; the rehearsal uses a different ordinal, the tree ordinal `j`, §5). A rehearsal miss or a refusal before the marker spends nothing: a new first-gate record (`kind: gate`). A spent run: `kind: re-gate` with `movedOut` counts, a new id and `--flags`, floor 35 on fresh clusters over the scored set, the 25% cap over the fresh counting scripts items before exclusion; this covers a miss and also a spent run with no verdict (`minimums-not-met`, `blocked` by an engine defect, a harness error after the marker) not cured by re-scoring the recorded answers with the engine unchanged (`movedOut` `{}` if nothing is to move). One fresh draw at most in this unit. Each successor record needs its own freeze-check permission request: the current permission is once.

## 5. Dev labelling plan (rehearsal)

- **Labeller** `qa-engineer` (blind dispatch, not a pack author, not the CO-2/CO-3 builder), rubric EP §11.4. **Spot-check** by `analyst`: 20 labels by hash sort; floor 18 of 20 agree and no disagreement on a confirmed item; below it the second labeller labels the whole sample and bar 3 uses the larger Q.
- **gateId and seed:** `ac5-gate-1-dev-t<j>` (j the tree ordinal: a new F or record starts a new one), seed `ccg-noise-v1|<gateId>|<snapshotId of the first run at the tree>`. A rerun at an unchanged tree reuses the same seed and labels, so it cannot re-roll the sample; bar 3 is judged on the largest Q across reruns at the tree.
- **Sample:** a pooled draw over all dev answers (at least 100 possible items, EP §11.2) and a scripts-only draw (at least 60, min 5 per stratum, census if fewer; 60 is a labelling-load choice, not a power figure), so the 22 scripts answers have their own estimate.
- Both labellers are LLM agent roles: 18 of 20 agreement is not independent validation, and the labelled Q is a dev fit statistic that carries nothing about held-out noise.
- **Bar 3 is the observed point estimate** Q ≤ 4 (EP §11.3), with the bootstrap interval reported: 22 answers with one 124-item answer make the percentile bootstrap poorly calibrated, and an upper-bound rule would be unpassable.
- A missing labelled Q reads NOT MET. A labelled Q_scripts (or pooled Q) above 4, the largest across reruns, is a rehearsal miss; the iteration count and any resolver tuning against dev are disclosed. A first labelled Q above 4 is a live outcome, not a tail.

## 6. Claim, noise, bound, reporting

- **Noise at P1 (stakeholder, 2026-11-25): held-out noise is not measured.** The claim carries "noise not measured"; the AC-5 noise clause (Q, precision, p90, reduction) stays open and carries to P2. The held-out AC-5 verdict is the listing verdict only (every scored item a hit, floors met, no harness error, no unreviewed masked comparison: EP HA-5, a comparison where masking changed a segment, which makes the verdict `blocked` until the method owner reviews it); no noise bar decides it, and the dev labelled Q is not noise evidence for the claimed draw. The possible-precision, p90 and S1 reduction bars are deferred, not waived, to P2's first gate. Label-free burden block A and the ceiling B (an upper bound, never noise) are reported beside it. S1's wording governs wherever a reduction is evaluated (the rehearsal; held-out only if the custodian's baseline exists, then reported, not gating).
- **Claim sentence** (the approved wording, kept verbatim in the record): "the frozen engine listed every counting item of an untouched draw outside the named withheld shapes", withheld counts by stratum and shape beside it, plus "noise not measured". It is over the scored set. It says nothing about wiring, flow, withheld shapes, shapes absent from the writers' draw, noise, or the engine after CO-10/11; not "scripts recall".
- **Cluster bound:** `c / (c + 3.8416)`, `c` = the number of original S47 cluster ids that still hold at least one scored item (no re-clustering), the custodian's count, counts only, unknown at writing, committed in a ledger line before the ask and the claim stated after the run. 58 is a ceiling for the first list only; a fresh draw states its own `c`. Bounds 0.9286 (c = 50), 0.9379 (c = 58). Item bounds are recomputed over the scored set at the run.

  | true cluster recall r | 0.99 | 0.98 | 0.97 | 0.95 |
  |---|---|---|---|---|
  | c = 50 | 0.61 | 0.36 | 0.22 | 0.08 |
  | c = 58 | 0.56 | 0.31 | 0.17 | 0.05 |

  Arithmetic `r^c`, not measured. With the withheld shapes out of the scored set the table no longer understates the miss chance by a shape term, and no "near-certain miss" reading applies. `r` is recall over the scored set, which still includes held-out analogues of unbuilt shapes the predicate does not flag (misses); 0.95 to 0.99 are priors.
- **Hit rule (disclosed reading).** The AC-5 verdict follows EP §5.2: a distance-2 hit is a hit, so a coincident-only item still lists. The sentence "the scripts resolver recovers cross-script links" is written only over concordant hits, and not at all if any scripts item is coincident-only. The stakeholder has not objected; the requirements log (2026-11-24) records it as "awaiting objection", and the record keeps that state.
- **Dev-fit caveat:** the withheld set and its predicate came after the CO-1 read of all 49 dev items; a dev pass is a fit statistic, carries nothing about held-out recall.

## 7. Open stakeholder questions

1. Is the reading that the held-out AC-5 verdict is the listing verdict only, with the S1 reduction bar evaluated at the rehearsal (reported) and carried to P2 for held-out, what you intend by "noise not measured"? Recommendation: yes.
2. The recall ruling's answer (credit only concordant hits?) is still unanswered; the record carries the default as a disclosed reading.
3. The labellers (`qa-engineer`, `analyst`) and the 18-of-20 floor are proposals; confirm or name others.
