# Connected code graph, P1-36: noise and runtime-oracle tooling gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-36

## Pass 1

**Scope.** Commits `ef8bf540`, `a51bbdeb` (`cpg/eval/tools/oracle.py`, `cpg/tests/test_oracle.py`) and `49ed31af`
(`cpg/eval/tools/noise.py`, `cpg/tests/test_noise.py`) against `docs/plans/connected-code-graph-p1-gates.md` (P1-36),
`docs/plans/connected-code-graph-p1-ml.md` §5, §6.3-6.5, EP §11 (`docs/plans/connected-code-graph-eval-protocol-ml.md`
§11; the `-protocol2-ml.md` file named in the brief has no §11), the P1-47 gate and the coordination ledger tail. Synthetic
fixtures only; no held-out data, no custodian directory, no container. Scratch scripts live in the session scratchpad; nothing
under `cpg/` was edited (`git status` clean after every CLI probe). `cpg/.venv/bin/python -m pytest tests/test_noise.py
tests/test_oracle.py`: 97 passed.
`CPG: considered, not relevant — the questions are arithmetic and tool behaviour, answered by execution; the cpg_* graphs do not model them`

**Verdict: approve with suggestions.** The arithmetic is correct on every axis I re-derived independently. One major (M1) is a
silent-pass path that must be closed before any live `score` run; M2 is a scope gap the plan has to rule on.

### Questions 1-6, answered

1. **Allocation (verified by execution).** I re-implemented §11.2 from the EP text (Fractions, floor quotas, remainder by
   descending fraction, capped, leftover to the largest unsampled stratum) and compared it with `allocate` on 8 named shapes and
   20,000 random strata sets (1-12 strata, sizes 1-1000): zero differences. Named shapes: a stratum of 3 (min 3, rest 34/63
   split), total 60 below target (whole frame, 60), all strata capped at 6 (18), four-way ties (25 each), 25 strata at 500 (minima
   win: 125), one stratum, exactly 100, cap-heavy (5/20/65). Property check on every random case: `min(N_h,5) <= a_h <= N_h` and
   `sum = min(total, max(100, sum of minima))`. The tie-break (`(reason, band)` order) is the code's, not EP's (m3).
2. **Seeds and draw (verified).** `draw` equals an independent sha256 sort of `f"{seed}|{anchorId}|{itemId}"` with `seed =
   "ccg-noise-v1|G|S"` on a 9-stratum, 1,200-item frame (sample 100); swapping gate and snapshot changes the sample. The
   bootstrap `random.Random("ccg-noise-boot-v1|G1")`, one `randrange(len(anchors))` per pick in sorted-anchor order, B = 10000,
   matched my own loop exactly (Q interval `[2/5, 11/5]`, `used` 10000). `percentile_indices(10000) = (250, 9749)`
   (`0.975 * 10000` is exactly `9750.0`). Fallback: with `used < B` (resamples whose `RC + R̂` is 0 are dropped) the indices
   scale to `int(n*0.025)`, `int(n*0.975)-1` and `used` is reported; EP is silent on that case (n1).
3. **Estimates (verified against a hand fixture).** Four answers, 14 items: strata r1/1 (N=5, f=3, t=2), r2/2 (N=2, 1, 1), r3/3
   (N=3, n=2, f=2): hand `Ŵ = 7`, `R̂ = 3`, `RC = 3`, `Q = 7/6`, possible precision `3/10`, band 1 `Q = 3/5`, band 3 `R̂ = 0, RC = 0`
   gives `None`. The code returns exactly these as `Fraction`s. p90 is the ceiling rank (n = 62 gives the 56th smallest, n = 10
   gives 9, n = 100 gives 90). Reduction flag is cross-multiplied (`8/10` vs `7/10` flagged, equal or higher not; `0/0` not).
   Thresholds: Q = 4 exactly is met, `4R >= W` is the exact form of possible precision >= 0.20, p90 = 100 met, reduction = 5 met,
   and `41/10` vs `499/100` / 101 are not. Kappa on a 10-item hand table (`po = 0.8`, `pe = 0.52`) is 0.583 in both. **Raw
   agreement exactly 0.8 is stable** (`agree*5 < 4n`; 16/20, 8/10, 800/1000 stable, 15/20 and 799/1000 unstable), which matches
   "below 0.8 raw agreement" in `-p1-ml.md` §5.1 item 5. **Q-NC5** counts Link rows, not units: three identical unbound rows on
   a two-unit key give 3 (and 3 confirmed); a key with one unit, or a bound link, is not counted; duplicate providers in one unit
   do not make a second unit. That is what P1-36 words ("count of unbound links"); the report says "links".
4. **Oracle (verified).** Wilson matches the closed-form interval for every `(k, n)` with `n < 300` (zero differences) and gives
   the plan's `20/100 -> [0.133, 0.289]`. Per-segment matcher, one planted input per axis: empty segment (`/a//b`), missing and
   extra segment, literal mismatch, verb mismatch, `*` verb, lower-case verb, `{}`, `{#}`, runtime query and full URL: all
   behave as specified; a partial placeholder (`x{?}`), case, percent-encoding and trailing-slash mismatch stay unmatched.
   Scope: `/shop/api`, `/shop/api/`, `/shop/api/x`, `/shop/api?x=1`, `/shop/api/x?x=1`, an absolute URL are in scope;
   `/shop/apix`, `/shop/apix/y`, `/shop/api2/x`, `/shop`, `//shop/api/x`, `/shop/api%2Fx`, upper-case `/SHOP/api` are out. Test
   code: `host == "testserver"`, `cls == "testclient"`, a frame under `tests/` (also `falkor-chat/tests/`, `salesperson/tests/e2e/`)
   are `testclient`; `x/mytests/`, `tests_helpers/`, `test_a.py` outside a `tests/` dir are not (n2). End to end on a planted
   capture the file-grain and function-grain classes are separate, a `frame: null` row is `unattributed`, the misses go to the
   out-of-repo file and stdout carries counts only. **SPA prefix caveat:** see m2; the SPA edges do not miss by construction.
5. **Custodian guard and stdout (verified).** `noise.py draw` and `oracle.py --misses-out` refuse (exit 2, nothing created) for
   `<repo>/x`, `<repo>/cpg/../x`, `<repo>`, `<repo>/../<repo>/x`, a symlink to `cpg/` plus a new child, and `symlnk/..`. On a
   six-anchor synthetic frame with `SECRETANCHOR*`/`SECRETSYM*` ids, stdout and stderr of `draw` and `score` carry no planted id
   (grep 0); the ids appear only under `--custodian`. Error text on bad input names fields and counts, not item ids.
6. **Not implemented.** See M2.

### Findings

**M1 (major, verified live) `score` accepts a labels file that omits the census, and reports a better figure.** I ran
`noise.py score` with `noise-labels.jsonl` holding only the possible-item labels: exit 0, `RC = 0`, `Q = 0.0`,
`confirmedPrecision = {k: 0, n: 0, s: None}`, `census = 28`. `_read_labels` (`noise.py`) only rejects labels *outside* the draw;
nothing requires every census item (EP §11.1: "all of them are labelled") or every drawn item to carry a label. A missing
census label removes a real item from `RC` and a false one from the confirmed count, so Q (gated at <= 4) and the 0.95
confirmed-precision gate move in the favourable direction with no error. Partial sample labels are the same: a stratum is only
refused when it has none (`_sums`). Fix: in `_score_cmd`, after `_read_labels`, require `census ⊆ labels` and `drawn ⊆ labels`
(or per stratum `n_h` equal to the allocation), exit 4 with the counts missing, never ids. Tests: one census label removed, one
drawn label removed, one whole stratum removed; each must exit 4. Put the tests in the round that adds the check, and the
obligation to satisfy them in the `score` step's row, not inside the validator they exercise.

**M2 (major, verified by sweep) The tooling cannot produce the plan's reduction, the recall flag, Q-NC5 or the oracle's Link set.**
`git grep` over `cpg/` and `docs/` for `links.jsonl`, `noise-baseline`, `noise-recall` finds only `noise.py`, `oracle.py` and
their tests: there is no producer for (a) the naive over-lister burden per anchor, (b) the item-level recall of the baseline
and of the system (`noise-recall.json`), (c) the snapshot dump for `unbound_multi_unit`, (d) the oracle's Link dump. The
consumers also do not validate what they are fed: `baseline.get(a, 0)` turns a missing anchor into burden 0, and the `recall`
tuples are unchecked (`k <= n`, `n > 0`). **Does the plan let P1-37 proceed?** Partly. P1-37 does not list P1-36 as a
prerequisite, and `-p1-ml.md` §5.3 gates only Q <= 4 at gate 1; reduction, possible precision and p90 are *reported*. So P1-37 may
run with reduction and Q-NC5 reported "not computed: generator not built", provided P1-36 stays open in the ledger (its done
list includes the baseline and Q-NC5) and the report says so. Two conditions: the item-level recall rule (what counts as a baseline
hit) is fixed in a committed record **before** the gate answers are scored, because the flag it feeds is judged against the
system's recall and a rule written after the numbers is a tuned rule; and the generator (graph-dba, an adjacency export) validates
its output (every held-out anchor present, `k <= n`). Suggested ledger line: "P1-36 open: over-lister generator, recall rule, Link
and snapshot dumps".

**m1 (minor, verified by plant, latent at P1) Function-grain oracle edges miss by construction for methods and nested functions.**
The shim records `f_code.co_name` (`oracle_shims/sitecustomize.py:30`, e.g. `meth`), while static elements are dotted qnames
(`api.py::build_router.health`, dev items). `score` requires `el == element`, so a planted Link `pkg/c.py::Client.meth` against
a runtime `pkg/c.py::meth` row is a miss and is listed as a `function`-grain miss. All in-scope edges of the P1-47 capture are
file grain (11 SPA routes; per the P1-47 review the other app rows are out of the storefront scope), so the P1 number is
unaffected, but a later capture with a Python storefront client would inflate misses and push them to "extractor" triage. Fix:
match function grain on the qname's last dotted component, or record `co_qualname` in the next shim version; pin it with the
`Client.meth` plant.

**m2 (minor, inferred for the Link dump, verified for the matcher) The SPA prefix is fine; the Link dump's key form is unspecified.**
Runtime SPA keys carry `/shop/api`. So do the static ones: `salesperson/src/api/endpoints.ts` builds every path from
`const API_PREFIX = '/shop/api'`; bundle-spec3 §1 (`http` row, step 3 and its example) says the consumer `${API_PREFIX}/health` and
the mounted provider normalise to one key; the dev items link provider `GET /shop/api/messages` to
`endpoints.ts::getMessages`. So SPA edges do **not** miss by construction, and file grain supplies the element. The residual:
`key_matches` compares *every* static segment, and a Link keyed by a consumer-form key with a leading wrapper placeholder
(pack comment `{?}/shop/api/messages`; the normaliser prepends `/`, giving `GET /{?}/shop/api/messages`) has one extra segment
and never matches `/shop/api/messages` (verified: False). `{**}`, the fourth placeholder (`keynorm.py`), is treated as a literal
and a static trailing `/` does not match a runtime path without it. Owner: whoever writes the Link dump (graph-dba, M2): it must
emit the Link's provider/join `ContractPoint.keyNorm`, and a test in `test_oracle.py` should feed one row of the real dump
format. If the dump cannot, `oracle.py` must handle the leading `{?}` and `{**}`. I could not read a real Link row (no store
access here), so that part is unverified.

**m3 (minor, verified) Draw details are the implementation's, not EP's.** EP §11.2 fixes the seed string and "hash sort over
`anchorId|itemId`", not how the two combine. The code hashes `seed|anchorId|unitId:subjectId` (sha256 hex); the item id carries a
`unitId:` prefix, the spot check appends `|spot` to the seed (EP §11.4 gives no seed), remainder ties break by `(reason, band)`
order, and a capped stratum's share of the extra units falls to the leftover rule rather than the next remainder. My independent
reproduction followed the code on all of these, so the draw is reproducible only by reading `noise.py`. Fix: one sentence each in
`-p1-ml.md` §5.1 (or the prereg `ac5-gate-1`), so a verifier does not have to read the source.

**m4 (minor, verified) Agreement edge cases.** With no overlap (`n = 0`) `labelsUnstable` is `False` and `rawAgreement` `None`;
an empty second-labeller file reads as "stable". `labels2` may be any subset of the draw, not the 20 keys `spot` wrote, so the
EP §11.4 spot-check can be silently replaced by an easier subset. Fix: `n = 0` gives `labelsUnstable: None` and the possible
precision unmarked-but-flagged "agreement not measured"; require `labels2` keys to equal `noise-spot.jsonl`.

**n1 (nit)** The bootstrap's dropped-resample behaviour (`used < B`) is undocumented in EP §11.3; the docstring should say the
indices scale. **n2** `classify` treats only a `tests/` directory as test code; `test_*.py` files elsewhere are in scope (no
in-scope instance in the capture). **n3** Edge identity uses the unmasked path (a numeric id makes distinct edges); no storefront
route has one, so P1's n is unaffected. **n4** `pooledAcrossGrains` is in the report although the plan says never to pool; the
docstring labels it reference-only, a reader quoting the report may not. **n5** The guard refuses only the repo; it does not
require the custodian root, so `--custodian /tmp/x` is accepted (stated scope in the brief, kept).

### What's solid

- Allocation, estimates, band split, p90, thresholds, kappa, the 0.8 boundary and the bootstrap are exact (`Fraction`) and
  reproduce under independent re-derivation; I found no arithmetic defect.
- Wilson is shared by both tools and matches the plan's own worked interval.
- Out-of-scope items leave the frame before allocation and are counted per answer and reason; burden is reported twice.
- Guards and stdout hold: refusal precedes any write, symlink and `..` forms are refused, and no planted id reaches stdout or
  stderr.
- The oracle's rulings from the P1-47 gate (file grain, per-segment placeholders, scope fixed before Link data) are implemented
  and pinned by tests that fail on the plants above.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`, after the "A fail-closed path" bullet. Source: M1 (this review).

- A scorer that reads an externally produced labels or results file is run with each required record removed in turn (a census
  member, one sampled item, a whole stratum); each run must exit non-zero, never report a better figure. A check that only
  rejects extra records is a bound.

### Open questions

1. M2: does the owner accept P1-37 with reduction and Q-NC5 "not computed" and P1-36 held open, or is the over-lister generator
   to land first? I recommend proceeding, with the recall rule committed beforehand.
2. m3: pin the draw's hash input and tie rules in the plan text (data-scientist) or in the prereg record?

## Pass 2 (2026-11-21)

**Scope.** Commits `c53b453c` (noise.py, oracle.py edits), `16714712` (`cpg/eval/tools/baseline.py`, `snapshot_export.py`) and
the ledger commits, `396e87d8..HEAD`, against `docs/plans/connected-code-graph-p1-noise-export-graph.md` (§4 rules, §4.1 graph,
§6, §7) and `docs/reviews/connected-code-graph-p1-noise-export-ml.md`. Root `REVIEW.md` applied; `cpg/REVIEW.md` does not
exist. From `cpg/`: `.venv/bin/python -m pytest tests/test_baseline.py tests/test_snapshot_export.py tests/test_noise.py
tests/test_oracle.py` gave 291 passed, 6 deselected (live tier), exit 0. No store was touched (no throwaway needed; the live
tier was not run by me); no held-out or custodian data read; probes and mutants ran from scratchpad copies, `git status` clean.
`CPG: considered, not relevant — the questions are rule arithmetic and file-format composition, answered by execution on a synthetic graph`

**Verdict: needs changes.** The six rules, the §6 recall rule and the §7 burden pairing are implemented and match the §4.1
numbers exactly. But the baseline output does not compose with `noise.py score` (B1), and `score` accepts recall figures it
could check against the rows (M1).

### Dispositions of Pass 1

- **M1** fixed: `_require_complete`; one census label removed gives exit 4 (`1 of 3 census ... unlabelled`), re-run on a CLI plant.
- **M2** partly fixed: baseline, recall, Link and snapshot producers now exist; the glue that writes `noise-baseline.jsonl`
  and `noise-recall.json` does not (B1).
- **m1** not fixed, as the docstring rules (function grain `co_name` vs dotted qname); no in-scope function-grain edge at P1.
- **m2** fixed: Link dump emits the provider `keyNorm`, consumer form kept apart, unprovided rows classify only; `{**}` is still
  a literal in `oracle.py` but `link_stats` counts it (`keyNormCatchAll`).
- **m3** pinned in the note §7 (draw choices); the `ac5-gate-1` record that carries them does not exist yet.
- **m4** fixed: `n = 0` gives `labelsUnstable None` / "not measured"; an empty or wrong-keyed `labels2` exits 4 (planted, both).

### Findings

**B1 (blocker, verified by execution) The baseline's real output cannot be scored.** I built the §4.1 snapshot with
`baseline_helpers`, wrote it, ran `read_snapshot`, `naive_over_lister` (symbol, contract, config and one `unresolved` anchor)
and `baseline_recall`, wrote `noise-baseline.jsonl` / `noise-recall.json`, drew with `noise.py draw`, labelled, and ran
`noise.py score --gate-record`. Results: (a) exit 4 `bad input: 'bound'`: `naive_over_lister` rows carry no `bound`, which
`_check_baseline` (`noise.py:489`) and note §7.1 require; (b) after adding `bound`, exit 4 `bad input: 'byName'`: the
`unresolved` row (note §4: "not an error") has no `byName`, and `_variant` indexes `r["byName"]` (`noise.py:556`); §7.1 and §4 disagree on that
row's shape and the noise fixtures (`test_noise.py:789`) supply fields the producer never emits. (c) `baseline.py` has no
`main`; nothing writes `noise-baseline*.jsonl` / `noise-recall*.json`, so `bound` and `in_scope` (default `None`: `countInScope`
silently equals `count`) are glue's to supply. The `itemSetSha256`, `exKeyError` hash and `ensure_ascii` forms **agree byte for
byte** across `baseline._gold_line`, `noise.item_set_sha256` and my own spec-derived implementation (non-ASCII qname, unsorted
`lines`, a `config` item with `keyPath`, 5 items and the 4 without key errors). Fix: a `baseline.py` CLI (snapshot, anchors, gate
record, exclusions, custodian) that stamps `bound` from the record, passes `in_scope`, emits `byName: {0, 0}` on unresolved
rows, and as its done-condition pipes its own files through `noise.score` (the round trip root `REVIEW.md` already requires);
or make `_variant` tolerate the §4 unresolved shape.

**M1 (major, verified) `score` trusts recall figures the rows contradict.** On the composed fixture (true `system` 2/5), setting
`noise-recall.json` `system` to `[5,5]` or `[40,50]` (hash unchanged) exits 0 and moves `flagged` False to True; `paired` set to
`both: 50` with sums not matching is accepted. `_check_recall` (`noise.py:459`) checks only `0 <= k <= n`, `n > 0` and the hash.
`system-hits.jsonl` holds every `hit`, so `system` can be recomputed. Fix: require `system == [sum(hit), len(rows)]`,
`systemSym` over symbol-type rows, `configGold`, the four `paired` cells summing to `n`, `both + systemOnly == system.k` and
`both + baselineOnly == baseline.k`, for the headline, key-scoped and `exKeyError` blocks. Plants: each of those cells off by one
must exit 4. (`baseline.k` cannot be recomputed without the snapshot; the 2x2 identities bind it.)

**M2 (major, verified by plant + reading) Contract-anchor consumer seeds omit `via_in:contract`: ruling.** `_seeds`
(`baseline.py:166`) seeds `consumes` owners only. Rule 2 says "`consumes_in` **and** `via_in:contract`", and the engine reads both
(`bfs.py:151`, `consumes_in` + `EXTRA_READS` on_in/via_in at `bfs.py:87,400-420`). The §4.1 table agrees with the code (CP1/key has no `Q`,
though `L3` is ON `CP1` and `Q` is its via owner), so the note is inconsistent with itself; the code follows the table, not the
rule text. Plant: `X` calls `Q`; contract `CP1`, key scope: the engine would reach `Q` (1) and `X` (2); the baseline lists
neither (verified: items `A B C E F G H P3 R`). Direction is **not one-way**: it lowers burden (reduction understated, conservative)
but also lowers baseline recall, which makes the "reduction at lower baseline recall" flag fire more easily (favours the
product), and it can make a class-A dominance check report violations on the baseline's own bug. It matters only where a
via site does not also `CONSUME` the link's point; the §4.1 graph contains that shape, the real store is unmeasured. Fix: seed
owners of `via` rows whose link's `cpId` is the anchor at cost 1 (and count them as `Kind` consumers), update the CP1 rows (add `Q`; key 10),
or amend rule 2 to say consumes-only and why. Coverage probe over the axes the rule varies: via status {confirmed, possible} x
link point {= anchor, other, none} x site consumes the point {yes, no} x kind scope {kind, key}, with a caller of the via owner.

**m1 (minor, verified by mutation) Four surviving mutants, all data plumbing**, `-m "not live"` on a copy: (e2) key-scoped
`reductionAll` bootstrap given the in-scope burden instead of all-listed; (h) `baselineSym` computed from the system's hit vector;
(p) `symbolOnly` flag built from `system` instead of `systemSym`; (o) `r_all` recall arguments swapped (equivalent today: its
`flagged` is never read, so also dead). 22 of 26 mutants (recall pooling across anchors, every pairing in `score` and `_agg`, `<`
vs `<=`, consumer cost 0, config cost 1, `kind`/`key`, `ownedRegions` pairing, `ro_query`, last snapshot read, the `exKeyError` hash)
were killed. Pin e2/h/p with fixtures whose two hit vectors and two burdens differ per pairing.

**m2 (minor, verified) `run_export` leaves files behind on a late failure.** With the Q-NC5 count planted wrong,
`ValueError` is raised (exit 4) after `snapshot-adjacency.jsonl`, `links*.jsonl`, `nc5-unbound.jsonl` were written. Write to
temp names and rename after `read_snapshot` and the count agree. A closed port gives a `redis.ConnectionError` traceback, exit 1,
not 4 (no secret printed).

**m3 (minor) `--gate-record` is a stand-in with no identity.** It reads `defaultDepthBound` from any JSON; the `ac5-gate-1` record
does not exist (only `ccg-p1-flow-python-v2-decision.json` is in `cpg/eval/decisions/`), and `defaultDepthBound` occurs in no code
outside this tool. It cannot fail on the wrong record. Once the record exists: require its `gateId` equals `--gate-id` and pin its hash.

**n1 (nit)** `_check_recall` rejects `n = 0` in `systemSym`/`baselineSym`/`exKeyError` (a legitimate empty symbol set). `snapshot_export`
defaults to port 6390 (read-only role, but the shared store: make `--port` required). `item_set_sha256` raises `KeyError` on an
impacted item with neither `qname` nor `keyPath`, `_gold_line` hashes `None`; they differ only there.

### Checks that held

Custodian guard (`snapshot_export`): `<repo>/x`, `cpg/../cpg/y`, `<repo>/..` and a symlink to the repo plus a child exit 2, nothing
created. `RoRunner` only calls `ro_query` (mutation to `query` killed). Snapshot id read first and last. Kind scope, bound 0/1/3,
`<=`, anchor removal, name rows, in-scope split and unresolved rows reproduce the §4.1 table on my own run (symbol A 9/5,
CP1 11/7, CE1 5/4, K 1, D/M 0; key 7/9/3; bound 0/1/3 of A 4/6/10, CP1 5/7/13, CE1 5/5/6).

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: Pass 2 M1.

- A scorer that takes pre-aggregated figures (`k`, `n`, a 2x2, a hash) beside the rows they summarise recomputes each from the rows;
  plant each figure off by one and require a non-zero exit. A hash that pins identity does not pin the values.

(Pass 2 B1 is already covered by the existing "validates the shape of a record field that a later unit produces" line; no addition.)

### Open questions

1. M2: does the owner want the engine-faithful union (rule text) or consumes-only (table)? I recommend the union, rerun on the dev anchors
   with the dominance check before any scored answer, since the recall rule is pinned by hash.
2. B1: who owns the `baseline.py` CLI (`tdd-engineer`, as the note's §7 order implies)? The `ccg-noise-recall-rule-v1` hash
   should be taken after it lands, not before.

## Pass 3 (2026-11-21)

**Scope.** Fix round `4a3924d1` (`cpg/eval/tools/{baseline,noise,snapshot_export}.py`, `cpg/tests/test_baseline_cli.py` new) and
note commit `c26eb82f`. From `cpg/`: `pytest tests/test_baseline.py tests/test_baseline_cli.py tests/test_snapshot_export.py
tests/test_noise.py tests/test_oracle.py` gave 398 passed, 6 live deselected, exit 0. Re-ran each Pass 2 plant against the tree
(scratchpad copies; no store, no held-out data, no custodian dir).
`CPG: considered, not relevant — rule arithmetic and file-format composition, answered by execution on a synthetic graph`

**Verdict: approve with suggestions.** Every Pass 2 defect is fixed and re-verified by execution; the new findings are one
residual forgery path and note/pin hygiene, none silent on honest input.

### Dispositions

- **B1** fixed: `baseline.py` CLI (`--anchors --exclusions --gate-record --gate-id --custodian`) writes both variants; my
  four-anchor run (symbol, contract, config, `unresolved`) went CLI -> `noise.py draw` -> `score --gate-record`, exit 0, rows carry `bound`,
  hashes agree (the CLI's `baseline_recall` figures pass `_check_recall`). CLI guard refuses `<repo>/tests` (exit 2).
- **M1** fixed: `system` 2/5 -> 5/5 or 40/50 with the hash unchanged now exits 4; also refused: `paired` not summing to `n`,
  `exKeyError.system`, `exKeyError.paired`, `configGold`, `baseline` `[0,5]` against the 2x2. Residual: see N1.
- **M2** fixed (union seeding): planted `X->Q` on CP1/key now lists `Q` and `X` (items 11, confirmed-own 6; `L3` via confirmed gives 8);
  `L3` with `cpId` null adds nothing; a via on an ownerless site is dropped; `bySeed` still sums to `count`
  (key CP1: providerOnly 4, consumerOrConfigOnly 3, nameOnly 3); CP1 kind 11/7, key 10/6, symbol A and CE1 unchanged; dominance A has 0 violations for the
  system's `Q`/`X` items. Note table recomputed in `c26eb82f` and agrees.
- **m1** fixed: all four survivors now killed (e2, h, p by assertion on value; `o` removed with its dead code, `allListed` computed directly).
- **m2** fixed: atomic export verified: with the Q-NC5 count planted wrong, nothing is left and a pre-existing `links.jsonl` is unchanged
  (staged `.tmp`, replaced after `read_snapshot` and the count agree); a refused port now exits 4 (`RedisError`).
- **m3** partly fixed: `read_gate_record` checks `defaultDepthBound` (int >= 0, not bool) and refuses a differing `gateId`, but `gateId`
  is optional and the record is not hash-pinned (the record still does not exist).
- **Nits** fixed: `--port` required, `in_scope` required keyword, `n = 0` allowed for the Sym/`exKeyError` blocks, a gold item with neither
  `qname` nor `keyPath` raises in both hashers.

### Same-revision interactions (executed)

Union seeding vs `bySeed`/dominance: consistent (above). Recompute vs `exKeyError`: an unresolved anchor's item marked `keyError`, all symbol
items `keyError`, and all items `keyError` each regenerate and score at exit 0 (CLI and scorer agree on `n`, including `n = 0`). Atomic
export vs the custodian guard: the guard runs first; `.tmp` files are removed in `finally`. Required `in_scope`/`--port`: omission fails at
call/argparse. Mutation on a fresh copy: the 26 Pass 2 mutants plus 19 new ones (each seed cost and status, each recompute and 2x2 check,
publish-before-check, `gateId`, unresolved `byName`, CLI `bound`, `in_scope`, key variant, both guards, port default) all killed; N3/N9
died on a crash rather than the intended assertion, so I ran the atomic-export and guard cases directly (above).

**Unresolved-row choice.** Following note §4 (no `byName`/`bySeed`) and changing the consumer (`r.get("unresolved")`) is right: §4 is the
contract the generator already honours, the row has no sites to count, and the alternative would put fabricated zeros in a pinned record.

### New findings

**N1 (minor, verified) `baseline.k` is still forgeable consistently.** Setting `baseline`/`baselineSym`/`exKeyError` k and the 2x2 together
(`baseline` 1/5, `baselineOnly` 0, `neither` 3) exits 0 and moves `reduction.flagged` False -> True; the hash pins identity, not
`baseline.k`. It is custodian-written and `system.*` is now bound, so the risk is tamper-only. Fix: the generator also writes per-item
baseline hits (hashed with the item set) and `score` checks their sum and the 2x2 per item; or accept and say so in §6.

**N2 (minor) Stale note text before hashing.** §4's signature still reads `bound: int = 2` and `in_scope=None` (code: neither has a default);
§7.1's baseline-row list names `byName` without saying the `unresolved` row has none. Fix in §4/§7.1 before the hash.

**N3 (nit) `configGold` has two definitions:** the generator counts `type == "config"`, the scorer `len(rows) - len(symbol rows)`. A `unit`-typed
gold item (none in `cpg/eval/dev`: 115 symbol, 11 config) is refused loudly (exit 4). Align on one.

### Can §4-§6 + `baseline.py` be pinned by hash for `ccg-noise-recall-rule-v1`?

Yes, after N2, with one change to the pin set: `baseline.py` now imports `noise.path_excluded`, `noise.read_gate_record` and
`match_keys.h1_symbol_credit`, so its hash alone no longer pins the rule's behaviour. Pin `noise.py` and `match_keys.py` beside it (or
confirm `ac5-gate-1` already does, `-p1-ml.md` §6.1). N1 may be fixed first or ruled accepted; it changes `baseline.py`'s output, so it must precede the hash if taken.

### Proposed REVIEW.md lines

None new. Pass 2 M1's proposed line (recompute pre-aggregated figures; plant each off by one) stands; N1 is the case it names for a
figure the rows cannot determine: add "or state which figure is unbound".
