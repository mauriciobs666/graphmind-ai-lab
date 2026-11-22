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
