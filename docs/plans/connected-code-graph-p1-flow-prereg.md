# P1-34 flow re-measurement — pre-registration package (python)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-34 steps 3–4) · **Extends:** `docs/plans/connected-code-graph-p1-ml.md` (§3.1, §3.3, §8) · **Reviews:** `docs/reviews/connected-code-graph-p1-flow-prereg.md`

The entered draw (stakeholder decision 2026-11-11). Method rules are cited, not restated: `docs/plans/connected-code-graph-p0-ml.md` §4.4, §6.4, §7.3, §8.4 and Part F;
`docs/plans/connected-code-graph-p1-ml.md` §3.1, §3.3, Appendix `oc.py`; `docs/reviews/connected-code-graph-p1-09-ml.md` C1, C5, C7, C8, C11 and the core-freeze rule.
No held-out data is opened or named and no label exists. A rule fixed here does not change after the first label.

## 0. Findings from the real system

1. **corpusCommit `0caf85e3ccb757801d15d6babe55a63477a486dd`** (`cpg/spikes/frontend-fitness/corpora.yaml`, python: repo `proving` = this repo, `rootPath falkor-chat/server`);
   `git cat-file -t` = commit; the extracted tree re-hashes to `srcDigest 365761fb…4800d6`. `falkor-chat/server` is byte-identical at `0caf85e3` and `EVAL_COMMIT cf7b2042`.
2. **Frame size** (`run_fitness.flow_population`, no hash sort run): P0-ML §6.4 population **368**, all 30 P0 methods inside, **338** remain before the held-out exclusion; tier-2 population (26–60 lines) **163**, disjoint from P0.
3. **Core freeze holds:** `s28_core.py` and `extract.sc` are unchanged since `3089951f` (`git diff` empty); `JOIN_P0_ONLY = False` (`s28_core.py:43`), read as a module global at call time, so a runtime `C.JOIN_P0_ONLY = True` works. `s28_run.py` at `cebc4245` records `extractSha256`.
4. **The P0 scripts cannot run this draw as they are.** `run_fitness.py` (`flowdraw`, `flowbrief`, `flowcheck`) writes `sample/python-flow.jsonl`; `s28_metrics.py` reads P0's sample, truth and audit files, seeds the spurious draw `ccg-p0a-v1-spur|`, writes fixed `results/projection-<lang>.*` and `adjudication/<lang>-flow-*` names, and never reads a `*.adj.jsonl` ruling file; `s28_run.py` overwrites the P0 shipped-facts reference in `s28-run.json`.
   Python uses the plain `extract` key, so **runs are made without `--guard`** (`--guard` wraps the OLD projection and `--extract` has no effect there; only csharp uses it).

## 0.1 P1-34a — tooling spec (owner `tdd-engineer`, gate `analyst`; row in `docs/plans/connected-code-graph-p1-gates.md` under P1-34)

P1-34a is **behaviour-preserving for everything the seed depends on** (frame, `siteKey`, hash input, sort, P0-sample subtraction): this document publishes the seeds, and P1-34a lands after. It must not be edited after U2 (§4).

- **Flow-set `p1`:** every path takes `…-p1` names (`sample/python-flow-p1.jsonl`, `-p1-tier2.jsonl`, `-p1.brief.md`, `labels/python-flow-p1.jsonl`, `truth/python.flow-p1-final.jsonl`, `adjudication/python-flow-p1-*`); the Part F and spurious seeds take `ccg-p1a-v1-`. `flowcheck --set p1`.
- **`s28_metrics.py`:** `--flow-set p1`; `--record <cpg/eval/preregistration.jsonl>` (required for p1: it must hold both `ccg-p1-flow-python-v1` and `-v1-draw`, and the run refuses with `NOT-MEASURED:void` on any script, sample (pinned from the `-draw` record, since `-v1` is committed before the draw), corpus-commit or `srcDigest` mismatch); `--run-record <s28-run.json|s28-run.shipped-p0.json>` (a record other than the default makes the cell diagnostic); `--join new|p0` (`p0` sets `C.JOIN_P0_ONLY = True` at runtime, output marked `diagnostic`; `s28_core` is not edited); **one output directory per cell** `results/p1/<projection>-<join>/`. Only the decision cell (fixed/new) writes the miss and spurious files; they are inputs to `teco`'s list builder (§2.5) and are never given to an adjudicator. Regression test: the scorer at default (p0) flags is byte-identical to the pre-change scorer on the stored P0 facts. Disclosed, pre-existing, not a gate: at the default join neither the old nor the new scorer reproduces the committed `results/projection-python.json`; the committed answers, misses and spurious files match only with `JOIN_P0_ONLY = True` (a diagnostic about the P0 results, which this measurement does not use).
- **`flowdraw-p1`:** fails closed without a **non-empty** `--exclude` (at least one valid line; an empty file on the real seed would write the unfiltered first 30) and without `--counts` and `--tier2-out <path outside the repo>`; refuses the `ccg-p1a-v1-flow` seed unless the corpus digest equals `srcDigest`; **before applying the exclusion it asserts the counts 368 (population), 338 (after the P0 subtraction) and 163 (tier-2 population) and aborts otherwise; the expected counts are a parameter, defaulting to these when the corpus digest equals `srcDigest`** (the frame depends on the tree-sitter grammar versions; a synthetic corpus passes its own expected counts, and a planted discrepancy must abort); **validates every exclusion entry**: each must name an existing corpus file in strict path form (relative to `falkor-chat/server`, `/` separators; no `./`, no leading `/`, no `..`, no empty segment, no backslash; inside the corpus root, a symlink escape refused) with `1 ≤ defLine ≤` that file's line count, otherwise it aborts reporting a line number only, writing nothing; an entry that is valid but falls in no frame method is legitimate, so both counts may be 0 and are printed (a non-empty `--exclude` is still required, so the unfiltered draw is never the empty-file default); the three outputs (tier-1, tier-2, counts) are staged and renamed atomically; never writes or prints an unfiltered order, rank, hash or excluded identity; error paths echo neither the private file's path nor its lines (planted malformed-line test). It writes tier-1 (30) to `sample/python-flow-p1.jsonl` and tier-2 (15, new code) only to `--tier2-out`; `--counts` holds exactly `excludedTier1Frame` and `excludedTier2Frame`; it also prints the tree-sitter and python grammar versions. Tests and the `analyst` gate run (a) on a synthetic mini-corpus, and (b) on the real corpus with a **test seed** and a dummy non-empty exclusion that asserts 368/338/163, plus a planted-discrepancy abort test; never the real seed. The hash rule is checked against an independent re-implementation; planted P0-sample methods are excluded; a planted exclusion removes exactly that method and its enclosing method.
- **Decision script `cpg/spikes/frontend-fitness/flow_decide.py`** (B1): inputs = the decision-cell scoring output, the frozen truth file, the adjudication list (§2.5) and the **two** rulings files (rows `{ordinal, pairIndex, adjudication}`). **Ruling vocabulary** (P0's scorer vocabulary plus `valid`): `overturned` = the pair is not real; `miss-stands` = a real pair the projection misses; `endpoint-naming-error` = a real pair whose labelled endpoint names the wrong point (stays, still a miss); `valid` = a listed connected pair confirmed real. **Only `overturned` removes a pair.** The set `D` = list pairs ruled `overturned` by **both** adjudicators. For each configuration it prints `y`, `G` with exactly `D` removed (a connected pair in `D` leaves both `y` and `G`; a miss in `D` leaves `G` only), `20y`, `19G`, margin `M = 20y − 19G`, projection success `20k` vs `19n`, and the verdict; it also prints, as a diagnostic, the verdict with `D = ∅`. `M` falls by 1 per removed connected pair, so it is the number of invalid connected pairs the verdict tolerates. Mutant tests: a pair ruled by one adjudicator only leaves nothing; a doubly-ruled miss moves `G` by 1 and `M` by 19; a doubly-ruled connected pair moves `y` and `G` by 1 and `M` by −1 (`20(y−1) − 19(G−1) = M − 1`; the `flow_decide.py` docstring, which says `M+1`, is corrected); dropping a rulings file, `>` for `≥`, and swapping `20y`/`19G` each fail a test. It never reads a held-out path.

## 0.2 P1-34b — audit tooling spec (owner `tdd-engineer`, gate `analyst`; before U2; row in `-p1-gates.md` beside P1-34a)

`adjudication/tools/python/flowaud_draw.py` is hard-wired to P0 (`ccg-p0a-v1-flowaud`, `labels/python-flow.jsonl`, unsure-first F1). One new script `cpg/spikes/frontend-fitness/flow_p1_audit.py` (its sha in record `-v1`), synthetic fixtures only, mutant-tested, emitting **only ordinals, pair indexes and the unsure flag, never label content**:
- **`flowaud-p1`** on `labels/python-flow-p1.jsonl`: F1 = 60 pairs by ascending `SHA-256("ccg-p1a-v1-flowaud|python|pair|<ordinal>|<pairIndex>")` over **all** pairs (uniform), plus a separate set of up to 20 `unsure` pairs not already drawn, same hash; F2 = the first 6 methods by `…|method|<ordinal>`, and with `--extend` the next 3.
- **`adjlist-p1`** (run by `teco`, never by an adjudicator): reads the decision cell's answers and the frozen truth, writes the §2.5 adjudication list (`ordinal`, `pairIndex`, kind, source, sink, evidence; **no status, no totals**) and its sha256. Let `n` = pairs missed in either configuration. List = those `n` misses plus connected pairs (connected in both configurations; or all of them if fewer) drawn by ascending `SHA-256("ccg-p1a-v1-adj|python|0caf85e3…|conn|<ordinal>|<pairIndex>")`: **`120 − n` of them when `n ≤ 60`, `60` when `n > 60`** (then `n` is recoverable from the length, but `n > 60` is a FALSE under any plausible ruling). Order = ascending hash of `…|shuf|<ordinal>|<pairIndex>`. The builder enforces length, composition, seed and shuffle; `flow_decide.py` enforces only that every miss is listed, no duplicates, and each rulings file rules exactly the listed pairs.
- **Encoding, exactly:** every hash is the lowercase-hex SHA-256 of the UTF-8 string with `|` separators, decimal integers without padding, a **0-based** `pairIndex` (file order of `pairs`) and no trailing newline; the `…` in the seeds above stands for the literal seed string of §1.1 (`ccg-p1a-v1-adj|python|0caf85e3ccb757801d15d6babe55a63477a486dd|conn|<ordinal>|<pairIndex>` and likewise `|shuf|`; `ccg-p1a-v1-flowaud|python|pair|<ordinal>|<pairIndex>`).
- Tests: hashes against an independent re-implementation; a planted miss missing from the list aborts; row keys are whitelisted (a status field cannot leak); the P0 `flowaud_draw.py` output is not reused.

## 1. Pre-registration content

### 1.1 Draw

- **Seed** `ccg-p1a-v1-flow|python|0caf85e3ccb757801d15d6babe55a63477a486dd`; hash order and `siteKey` as P0-ML §4.4 (`run_fitness.hash_sort`); **sample = first 30**, ordinals 1–30. No salt: the seed is public so the draw is reproducible by anyone (ruling §3.4).
- **Frame** = the 368-method population **minus the 30 methods of `sample/python-flow.jsonl`** **minus every method whose span contains the definition line of a held-out or reserve AC-5 flow anchor** (file and span, so no qname grammar; an enclosing method of a nested anchor goes too). The custodian applies the last term (§3). Dev anchors stay; their overlap with the sample is counted by `teco` after the draw and recorded.
- **Spurious sample:** seed `ccg-p1a-v1-spur|python|0caf85e3…`, up to 50 projection pairs not in the golden set, drawn from the fixed/new cell, configuration `off`, as P0-ML §8.4. **Part F seeds:** `ccg-p1a-v1-flowaud|python|pair|<ordinal>|<pairIndex>` and `…|method|<ordinal>` (`flowaud-p1`, §0.2).
- **Tier-2 (contingency, P0-ML §8.4 two-stage).** The tier-2 file is held by the custodian, committed by digest only, and released to `teco` only if it fires. It fires iff the labelled pair count that `flowcheck` reports at the end of the labelling dispatch, before any audit, is **< 120**. The same labeller then labels once, in the same dispatch, the first **15** methods of the 26–60-line population (same seed and sort, same exclusions). The 120 floor is applied to the pooled count; still below is row 2.
  Pooling 26–60-line methods unweighted changes the estimand (longer methods weigh more); the decision is on the pooled figure (P0-ML §8.4) and the tier-1-only figure is reported beside it. Expected unused (`-p1-ml.md` Appendix: 0 of 200 resamples below 120).
- **JavaScript:** not measured (`-p1-ml.md` §3.2).

### 1.2 Records (two lines in `cpg/eval/preregistration.jsonl`, appended by `teco`)

`ccg-p1-flow-python-v1` is committed at **U2, before the draw**; it holds every pin, all known today or at U1.

```json
{"id": "ccg-p1-flow-python-v1", "governs": "P1-34 steps 3-4, registry row python.flowAvailable", "lang": "python",
 "corpus": {"repo": "proving", "commit": "0caf85e3ccb757801d15d6babe55a63477a486dd", "rootPath": "falkor-chat/server", "srcDigest": "365761fbdd26604c538f12beceb05811a8d77a32c8dd4ad01abc624a484800d6"},
 "coreFreeze": {"commit": "3089951f9ead86141c8a2de1d11e0d139518bbf7",
                "s28_core.py": "aaba9b1371d6dc6939eb3363576ac5053e8121dd253fd362d21e6a608c5ad615", "JOIN_P0_ONLY": false,
                "extract.sc": "6cdb498f3e9482a44322e5ef81b2dedc699165ab90d49b12b441d2a9d226ca7d"},
 "tools": {"s28_run.py": "54fc1bea7830e1ba45981ea39485b9cd2893647ebdbe73d3fdb42332d31a09b2 (cebc4245; no --guard; --extract cpg/extractors/joern/extract.sc)",
           "s28_metrics.py": "f7cc52af… (full sha256 recorded at U2 from `sha256sum`)", "run_fitness.py": "a8989bce… (same)", "flow_decide.py": "ef2f93c2… (same)", "flow_p1_audit.py": "ac2d158a… (same; not to change again)", "flow-brief.template.md": "<recorded at U2 from `sha256sum`>"},
 "cpgs": {"off": "a30c83636120146fd13f69c7d0772a2fad137a8231f1b2701c848c4182bf01ac", "on": "fc06c0c32a5c38026103986b99f6668bc16d2e0148ef628699a1cc7d3598bb60"},
 "shippedFacts": {"off": "a970a79cc4ceb67ae0fa76ce21f69b51cda0d1bda5accb0fc52935153bc4f5b0", "on": "2292a32e76f757cf2fce856b77e78db1dccb88ed2d9b486597e0846b7ed98890", "note": "the P0 shipped projection's facts; the P0 run record has no script sha, so the files are pinned by digest only"},
 "seeds": {"flow": "ccg-p1a-v1-flow|python|<commit>", "spur": "ccg-p1a-v1-spur|python|<commit>", "flowaud": "ccg-p1a-v1-flowaud|python|…", "adj": "ccg-p1a-v1-adj|python|<commit>"},
 "frame": {"populationAt0caf85e3": 368, "minusP0": 30, "heldoutExclusion": "by custodian, file+span, held-out and reserve anchors, counts only"},
 "rules": "this document §0.1, §1, §2, §5; -p1-ml.md §3.3; P0-ML §6.4, §7.3, §8.4, Part F",
 "decisionCell": "fixed projection (extract.sc @3089951f) x new join (JOIN_P0_ONLY=False), off and on, both must pass; the other three cells diagnostic only (P1-09 review C7, C11)",
 "entryValue": "465/481 (464/480 + omission), met with zero margin, designed on the measured methods"}
```

`ccg-p1-flow-python-v1-draw` is committed at **U4, after the draw and before the first label**: `{"id": …, "excludedTier1Frame": k1, "excludedTier2Frame": k3, "sampleSha256": …, "tier2Sha256": …, "briefSha256": …, "devAnchorOverlap": n, "treeSitter": "<pkg version>", "grammarPython": "<version>"}` (the tier-2 file is **not** committed, digest only). The draw is label- and Joern-independent, and the tool and rules are in git before it runs.

`cpg/eval/decisions/ccg-p1-flow-python-v1-decision.json` (after the run, before the registry edit) carries, per configuration: `y`, `G`, `M`, recall, cluster interval (P0-ML §7.3), single-edge share, projection success `k/n`, the fixed facts files' sha256, the toolchain image id (same for both runs); the omission rate with its flag; spurious fixed/new beside shipped/new; methods fully recovered; off/on discordant pairs; the four cells (`fixed/new` decision, the rest `diagnostic`); residual misses by cause (C8); the verdict with `D = ∅` (diagnostic); the adjudication list's and both rulings files' shas; the outcome code of §5; `flowAvailablePython`.

### 1.3 Four-cell procedure

Projection {shipped = P0's facts (digests above), fixed = this run's facts} x join {`p0`, `new`}, each in `off` and `on`, scored by the frozen `s28_core` on the same sample and truth; one output directory per cell (§0.1). Only fixed/new decides; the old-to-new difference at the fixed projection is the join (C11). The dev cells (307, 336, 425, 464 of 480) are cited, not merged.

## 2. Labeller brief and adjudication (hand to `qa-engineer` verbatim)

**2.1 Brief.** The P0 generator output, not a rewrite: `run_fitness.py flowbrief python` in flow-set `p1` from `flow-brief.template.md` (the `flow-brief.patch.md` generator patch is already applied). Differences from P0's `sample/python-flow.brief.md` are `{SAMPLE_FILE}`, `{LABEL_FILE}`, `{N_METHODS}` (30) only; `teco` diffs and any other difference stops the dispatch. The dispatch prompt is the brief plus 2.2, nothing else.

**2.2 Dispatch instruction (fresh `qa-engineer`, one dispatch, no other context):**
> Follow the attached brief exactly. Label the methods of `cpg/spikes/frontend-fitness/sample/python-flow-p1.jsonl` into `cpg/spikes/frontend-fitness/labels/python-flow-p1.jsonl`, one JSON line per method in ordinal order, never skipping. You may read only the corpus root `cpg/.ccg-work/corpora/python/src`, that sample file and your output file. Do not use Joern, a parser or AST library, a language server, a compiler, the web, MCP tools or any knowledge base, and open nothing else on this machine. After the last method run `python3 cpg/spikes/frontend-fitness/run_fitness.py flowcheck python --set p1`, repair every error, and report counts only. If `teco`'s follow-up message names a second file, label it the same way.

**2.3 Check.** `teco` reads the subagent transcript against an **exact-path allowlist** (the corpus root, the named sample file, the named output file, `flowcheck`); every other path or tool is a hit, and any of these substrings is a hit regardless: `joern`, `2cpg`, `cpg.bin`, `mcp__`, `WebFetch`, `WebSearch`, `cpg/eval`, `heldout`, `flow-anchors`, `ccg-eval`, `.local/share`, `EVAL_COMMIT`, `kit-src`, `results/`, `truth/`, `adjudication/`, `s28_`, `extract.sc`. A hit voids the labels (row 1 of §5). `flowcheck` must report `nErrors` 0 and `complete`.

**2.4 Part F-style audit, before any projection output exists** (P0-ML Part F; the adjudicators read no `results/`, facts or Joern output):
- **F1** (`data-scientist`): **60 pairs drawn uniformly** by the `flowaud-p1` pair hash; the `unsure` pairs are judged as a separate additional set (up to 20) and reported apart, so the flag rate (invalid/60 > 0.05, ≥ 4) is a population rate. Each is judged from source with file:line. F1-invalid pairs from either set leave `G` in the truth freeze (the unsure set is targeted, one-sided, and disclosed as such).
- **F2** (a **second** fresh `qa-engineer`, blind, same brief; the labeller cannot audit itself): relabel 6 methods by the `flowaud-p1` method hash into a separate file before opening `labels/python-flow-p1.jsonl` (`teco` checks the order); `data-scientist` judges every one-sided pair from source. Flag iff omission rate > 0.05 with ≥ 20 valid pairs (fewer: extend by 3 methods once, then `insufficient`). Same model and brief as the labeller, so omissions are correlated and the rate is biased down; it is a qualifier only (P0-ML §8.4).
- **Truth freeze:** `labels/python-flow-p1.jsonl` is never edited; `truth/python.flow-p1-final.jsonl` = golden set minus F1-invalid pairs plus F2 omissions of the 6 audited methods, written **before scoring**. A flagged F1 or F2 escalates per P0-ML Part F (a second fresh labeller relabels all 30 blind, the union is the truth), still within this draw.
- **Limit, disclosed:** adjudication is one-sided (only misses can leave `G`; connected pairs are audited only through F1's 60). A 2% invalid rate is about 10 pairs, the whole dev margin, and F1 (flag at ≥ 4 of 60) would not see it. §2.5's list also samples connected pairs, and the rate of doubly-ruled not-real pairs among them is reported beside `M`; neither detects a rate that small. The control is reporting `M` (§0.1), not detection: a TRUE with small `M` is recorded with `M` and the F1 interval, and the verdict is unchanged.

**2.5 Rulings, two-sided and blind to totals.** After scoring, `teco` builds, with `adjlist-p1` (§0.2), one **adjudication list**: every decision-cell (fixed/new) miss in either configuration (`n`), plus seeded connected pairs, `120 − n` of them when `n ≤ 60` and 60 when `n > 60` (seed `ccg-p1a-v1-adj|python|0caf85e3…`, hash order), **shuffled by the same seed**, rows carrying no status, no cell totals, no `y`, no `G`. The 50 spurious pairs are listed separately. Each adjudicator is handed **only `teco`'s shuffled list** and the corpus root. Reading `results/`, `adjudication/*misses*`, any facts file or Joern output, and the `flow_decide.py` output is forbidden (after U7 those files are in the repo and reveal which pairs are misses and how many); `teco` checks **both** rulings dispatches by transcript against the §2.3 allowlist and substrings (plus `results/p1`, `misses`, `facts`). **Both** `data-scientist` and a fresh `analyst` dispatch rule on **every row** independently, from source, into two files (`adjudication/python-flow-p1-rulings-{ds,analyst}.jsonl`) committed before `flow_decide.py` runs; neither sees the other's file or any `flow_decide.py` output. The ruling `analyst` is not the dispatch that gates U8. **Status-neutral instruction:** every row, miss or connected, is given the same options and the same words: `overturned` = not a real pair; `valid` = a real pair the projection (if it has it) rightly connects; `miss-stands` = a real pair; `endpoint-naming-error` = a real pair whose labelled endpoint names the wrong point. No option is described as belonging to a status. Only `overturned` removes a pair; `endpoint-naming-error` stays in `G` here, whereas P0's scorer removed it (conservative: it keeps a miss a miss). Diagnostic cells are not adjudicated.

## 3. Custodian step list (repo owner)

Everything runs on the owner's machine. The only things that come back are those in 3.3.

**3.1 Committed first** (public): this document, P1-34a, record `-v1`, `sample/python-flow.jsonl` (the 30 P0 methods), `corpora.yaml`.

**3.2 Steps, in order.**
0. Pull the commit holding the record. Verify `sha256sum cpg/spikes/frontend-fitness/run_fitness.py` equals the record's `tools.run_fitness.py`; stop on any difference, **before** the private file is created or touched by the tool.
1. `cd cpg && .venv/bin/python spikes/frontend-fitness/run_fitness.py fetch python` must report `reused` or `digest 365761fbdd26604c538f12beceb05811a8d77a32c8dd4ad01abc624a484800d6`; stop otherwise.
2. From the held-out **and reserve** AC-5 `flow` items whose anchor is python, write a private file **outside the repo**, never committed, one JSON line per anchor: `{"path": "<relative to falkor-chat/server, '/' separators, no './', no leading '/'>", "defLine": <1-based name line, within the file>}`; the tool aborts on any entry that is not of this form or names no corpus file, so a typo cannot silently exclude nothing; the form is exactly the tool's strict path form (no `./`, no leading `/`, no `..`, no empty segment, no backslash) (the subtree is identical at `0caf85e3` and `EVAL_COMMIT`, so no translation).
3. `cd cpg && .venv/bin/python spikes/frontend-fitness/run_fitness.py flowdraw-p1 python --exclude <private-file> --counts <counts.json> --tier2-out <path outside the repo>`. It aborts unless the counts 368/338/163 hold; it writes `sample/python-flow-p1.jsonl` (30) and, to `--tier2-out` (kept by you), the 15-method tier-2 file, and prints two integers and the tree-sitter and grammar versions. (`--exclude` must be non-empty.)
4. Check: 30 and 15 lines, none in `sample/python-flow.jsonl`; run `sha256sum` on the tier-2 file. Keep the private file and the tier-2 file where they are.

A `0`/`0` result with valid entries is normal (anchors usually lie outside the 338-method frame) and is reported as such.

**3.3 Returned:** `excludedTier1Frame` (k1), `excludedTier2Frame` (k3), `sample/python-flow-p1.jsonl`, the tier-2 file's sha256 (not the file), the tree-sitter and grammar versions. **Never returned:** the private file, any anchor path, line or qname, any skipped position, any held-out item or key, and any prefix count. If both counts are zero say so, not why.

**3.4 Ruling on the residual and on a salted seed.** Someone who recomputes the unfiltered hash order and diffs it against the sample could infer which methods in the first 30 were skipped (expected `k1 · 30/338`, usually 0; public code identities, not a held-out item or key). The prefix counts are no longer returned, so nothing motivates the diff, and the tool never writes the order. **A custodian-private salt (commit `sha256(salt)`, reveal after the freeze) is not adopted:** it removes the residual but departs from the stakeholder-specified public seed, makes the draw unverifiable until the reveal, and gives the custodian an unauditable re-roll. Reconsider only if the stakeholder wants zero residual at that price. Excluding all eight public `ccg-p0f-flow-v1` draw members is also rejected: the team would compute every anchor identity. The anchor list is derivable from public pins and `export_kit.py`, so custody is procedural under every design.

**3.5 Later.** The sample, labels and records fall inside the custodian's `freeze-check` window (`docs/plans/connected-code-graph-p1-gates.md`, P1-32a) before the `ac5-gate-1` record. A hit there signals a failed exclusion and does not reopen this decision.

## 4. Sequencing, resources, gates

| # | Unit | Owner | Output | Gate |
|---|---|---|---|---|
| U1 | P1-34a (§0.1) and P1-34b (§0.2), synthetic corpus only | `tdd-engineer` | tools, `flow_decide.py`, `flow_p1_audit.py`, tests green | `analyst` |
| U2 | commit this plan + record `-v1` (all pins) | `teco` | pre-registration in git before the draw | `analyst` |
| U3 | custodian draw (§3) | repo owner | sample files, k1, k3 | — |
| U4 | commit samples, brief, record `-draw` | `teco` | first-label barrier | `data-scientist` reads the filled brief |
| U5 | labelling, `flowcheck`, transcript check, tier-2 if triggered | `qa-engineer`, `teco` | `labels/python-flow-p1.jsonl` | `analyst` |
| U6 | Part F (§2.4), truth freeze | `data-scientist`, `qa-engineer` | `truth/python.flow-p1-final.jsonl` | `analyst` |
| U7 | offline runs and scoring | `graph-dba` | facts (`.ccg-work`), `results/p1/*` | — |
| U8 | `teco` first inspects the frozen truth for any field that correlates with outcome (e.g. the `evidence` strings of pairs added by F2 or its extension; the list whitelist passes `kind`, `source`, `sink`, `evidence` verbatim) and strips or neutralises it in the list builder's input, never in the truth file; then `adjlist-p1`, two-sided rulings (§2.5), `flow_decide.py`, decision record | `data-scientist`, ruling `analyst` | `-decision` | a different `analyst` dispatch |
| U9 | registry flip only if TRUE | `graph-dba` | `flow-availability.yaml` python row; P1-42 AC-25 fixtures and P1-37 `flow` stratum re-run | `analyst` |

U7 runs after U6. (1) Verify the two `shippedFacts` digests, then copy `cpg/.ccg-work/frontend-fitness/joern/python/s28-run.json` to `s28-run.shipped-p0.json`. (2) From `cpg/`, foreground, one container at a time: `.venv/bin/python spikes/frontend-fitness/s28_run.py python --only off --extract cpg/extractors/joern/extract.sc`, then `--only on`; each record must show `rc 0` and `extractSha256 = 6cdb498f…`; record `docker image inspect` id. (3) From the repo root or `cpg/`, the four cells (`--record` and `--run-record` resolve cwd-relative, then repo-relative; `--results`, `--answers`, `--truth` stay cwd-relative, so these are written from `cpg/`), no `--guard`, the decision cell last:
```
R=cpg/.ccg-work/frontend-fitness/joern/python/s28-run.shipped-p0.json; P=cpg/eval/preregistration.jsonl; M=spikes/frontend-fitness/s28_metrics.py
.venv/bin/python $M python --flow-set p1 --record $P --run-record $R --join p0     # shipped/old, diagnostic
.venv/bin/python $M python --flow-set p1 --record $P --run-record $R --join new    # shipped/new, diagnostic
.venv/bin/python $M python --flow-set p1 --record $P --join p0                     # fixed/old, diagnostic
.venv/bin/python $M python --flow-set p1 --record $P --join new                    # fixed/new, decision
.venv/bin/python spikes/frontend-fitness/flow_decide.py --record eval/preregistration.jsonl \
  --results spikes/frontend-fitness/results/p1/fixed-new/projection-python-p1.json \
  --answers spikes/frontend-fitness/results/p1/fixed-new/projection-python-p1.answers.jsonl \
  --truth spikes/frontend-fitness/truth/python.flow-p1-final.jsonl --list <list> \
  --rulings-ds spikes/frontend-fitness/adjudication/python-flow-p1-rulings-ds.jsonl \
  --rulings-analyst spikes/frontend-fitness/adjudication/python-flow-p1-rulings-analyst.jsonl
```
The last command runs after the rulings are committed (§2.5).

**Digest checks.** The tool (`s28_metrics.check_record`) checks sha256(`cpgFile`) against the record's `cpgs.off`/`cpgs.on` in every cell, and sha256(`factsFile`) against `shippedFacts.off`/`on` for the shipped projection. `teco` checks by hand, because no pin exists before the run, the **fixed**-projection facts sha256 (the `factsFile` digests, copied from the run record into the decision record) and the toolchain image id.

**Checks by hand (`teco`), not made by any tool:** before each of the two `s28_run.py` runs, `docker image inspect --format '{{.Id}}' <the image tag printed in the run log>`; the two ids must be equal and go into the decision record (row 1). Also by `teco`: transcripts (labeller, both rulings dispatches), `flowcheck`, F-escalation completion, the tier-2 sha256 at release. The tools check: script and sample shas, corpus commit and `srcDigest` (`s28_metrics.check_record`), list and rulings completeness (`flow_decide.py`), list shape (`adjlist-p1`).

**Resources.** P0 python records (`s27-run.json`, `s28-run.json`): content check 9–12 s and 1.0–1.2 GiB peak, extract 14 s and 1.2–1.3 GiB peak per configuration. The fixed projection was run on this CPG in P1-09 under a bounded harness (`cpg/.ccg-work/p1-09/tools/run.sh`: `-Xmx1500m`, 150 s kill) and completed, but no wall time or peak was recorded, so its figure is unmeasured. Host: 5.9 GiB total, about 4.2 GiB available. Caps: `timeout 600` per invocation, abort and report if `free -m` shows < 2.5 GiB available at a start or a run exceeds 3 GiB peak or 300 s; scoring loads the 37 MB facts file in one Python process. Never touch `falkordb-dev` or `~/.config/ccg`.

## 5. Decision table (first matching row applies; rows 2, 4 and 5 evaluate with `D` removed)

**Outcome rule.** The verdict is computed once, with exactly the doubly-ruled pairs `D` (§0.1) removed from `G` (and from `y` when connected); a pair ruled not-real by one adjudicator only stays. The verdict with `D = ∅` is reported, never decisive.

| # | Observation (committed files and `flow_decide.py` output) | Outcome | Registry |
|---|---|---|---|
| 1 | a rulings file is missing or incomplete (not every list row ruled), a rulings-dispatch transcript hit, a released tier-2 file whose sha256 differs from the recorded one (`teco` checks it at release), an edit to any pinned file after U2 (`s28_core.py`, `extract.sc`, `s28_run.py`, `s28_metrics.py`, `run_fitness.py`, `flow_decide.py`, `flow_p1_audit.py`, the brief template) voids the pre-registration; a recorded sha or digest differs at run time (`s28_core.py`, `extract.sc`, `s28_run.py`, `s28_metrics.py`, `run_fitness.py`, `flow_decide.py`, CPG, corpus `srcDigest`, toolchain image differing between `off` and `on`), a transcript-check hit, a `flowcheck` error left at the end, or an F1/F2 escalation not completed before scoring | **NOT-MEASURED:void** | stays false |
| 2 | `G < 120` after tier 2 | **FALSE:insufficient-sample** (`flow-unavailable:python:insufficient-sample`) | stays false |
| 3 | projection success `20k < 19n` in either configuration | **FALSE:precondition** | stays false |
| 4 | `20y < 19G` in either configuration (including one configuration passing and the other failing: the discordant pairs are listed and a defect filed in addition) | **FALSE:measured** | stays false, "not closed" with the figure |
| 5 | otherwise: both configurations `G ≥ 120`, `20y ≥ 19G`, `20k ≥ 19n` | **TRUE** | python flips true (`graph-dba`, gated) |

Every FALSE or NOT-MEASURED outcome means: flow stays false, the draw is spent, and no second draw exists without a `v2` seed and a new stakeholder decision. The omission flag, cluster interval, spurious rate, methods fully recovered and `M` are reported with the outcome and never change a row (P0-ML §8.4).
Pass probability: `-p1-ml.md` §3.1 and Appendix `oc.py` (about 0.55 at a true pair recall of 0.95, about 0.2 at 0.93; the bar at 480 pairs is 456). A FALSE is an expected-sized outcome, not a defect.

## 6. Risks

- Defaults taken: F2 uses a second labeller; the exclusion matches file and span and covers reserve anchors; no salt (§3.4).
- A false golden pair that the projection also connects stays in `G` (§2.4 limit); `M` is reported.
- Builders may have seen a drawn method (dev anchors, debugging): counted, not removed after the draw.
- Tracked `cpg/` files enter the later `ac5-gate-1` `coreTreeHash`: P1-34a, the samples and the labels are in git before that record (P1-09 review, freeze ordering); `teco` checks the order.

## 7. Done-condition for the pre-registration (U2 proceeds iff every item is true; a further review checks only these)

1. `git cat-file -t 0caf85e3ccb757801d15d6babe55a63477a486dd` is `commit`; the corpus tree digest equals `srcDigest` in the record.
2. `git diff 3089951f HEAD` over `s28_core.py` and `extract.sc` is empty; their sha256 and `JOIN_P0_ONLY = False` match record `-v1`.
3. P1-34a and P1-34b are committed, their tests are green, and the shas of `s28_metrics.py`, `run_fitness.py`, `flow_decide.py` and `flow_p1_audit.py` in record `-v1` equal the committed files; `s28_run.py` is `54fc1bea…09b2`.
4. P1-34a tests show: `flowdraw-p1` aborts on an empty or missing `--exclude`, on missing `--counts` or `--tier2-out`, and when the counts differ from 368/338/163; the real-corpus test uses a test seed and asserts 368/338/163; the §0.1 regression test passes; no test or gate runs the real seed (`git grep -n 'ccg-p1a-v1-flow|python|0caf85e3'` over `cpg/tests` and the tool tests finds none).
5. `flow_decide.py` mutant tests of §0.1 and the `flow_p1_audit.py` tests of §0.2 pass; its verdict rule is the §5 outcome rule and uses `D` only.
6. Record `-v1` contains every `<sha>` filled, and its text matches §1.2; the shipped-facts and CPG digests recompute from `.ccg-work`.
7. §2.5's list length, seed and the two-adjudicator rule, and §3's custodian steps (including step 0), are as written here and unedited since this item was last checked.
8. The `-p1-ml.md` header carries `Extended by:` for this note and the gates doc carries the P1-34a and P1-34b items.
