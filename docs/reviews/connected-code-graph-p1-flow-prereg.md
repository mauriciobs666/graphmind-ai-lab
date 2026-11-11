# Review — P1-34 python flow re-measurement pre-registration package

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-34 steps 3–4)

**Scope.** `docs/plans/connected-code-graph-p1-flow-prereg.md` (the note), against `docs/plans/connected-code-graph-p1-gates.md` P1-34,
`docs/plans/connected-code-graph-p1-ml.md` §3, §3.3, §8, `docs/plans/connected-code-graph-p0-ml.md` §6.4, §8.4, `docs/reviews/connected-code-graph-p1-09-ml.md`
C1/C5/C7/C8, and `cpg/spikes/frontend-fitness/{run_fitness,s28_run,s28_core,s28_metrics}.py`. Static review plus read-only re-derivation. Nothing under
`~/.local/share/ccg-eval/` or any held-out data was opened; no unfiltered hash order was computed or written.
`run_fitness.py` and `s28_metrics.py` were being modified in the tree by P1-34a while I reviewed (uncommitted); I judge the plan, not that tooling.

**Verdict: needs changes** (one blocker, five majors, all fixable by text edits before U2, the commit of this note; the draw and method are sound).

CPG: considered, not relevant — a statistical/procedural review of a measurement protocol over spike scripts; no code-level impact question.

## Verified (checked, not assumed)

- Pins recomputed: `s28_core.py` aaba9b13…5ad615 and `extract.sc` 6cdb498f…226ca7d are byte-equal at `3089951f` and in the tree (`git diff 3089951f HEAD` empty on both, unmodified);
  `s28_run.py` 54fc1bea…09b2 equals `git show cebc4245:` and the tree; `JOIN_P0_ONLY = False` (`s28_core.py:43`) and it is read as a module global at call time (`:345`, `:358`), so a runtime `C.JOIN_P0_ONLY = True` does work.
- `shippedFacts` off/on a970a79c…/2292a32e… and CPG shas a30c8363…/fc06c0c3… recomputed from `.ccg-work` (all four match); corpus `srcDigest` 365761fb… matches `run_fitness.tree_digest`; `falkor-chat/server` byte-identical at `0caf85e3` and `cf7b2042`.
- Population 368, P0 methods ⊂ population (30/30), 338 remain, tier-2 population 163, disjoint from P0 (re-derived with `flow_population`, no sort); `siteKey`s unique, so the tie rule never fires.
- Operating curve: I ran the `-p1-ml.md` appendix `oc.py` on the P0 pair counts (29 methods with pairs, 481 pairs). 30 methods: P(pass) 0.54–0.56 at true recall 0.95 (rho 0.1/0.2), 0.17–0.24 at 0.93, 0.80 at 0.965. The note's "about 0.55 / about 0.2" reproduces. `oc.py`'s criterion (`misses <= N − ceil(0.95N)`) equals `20y ≥ 19G`; 456/480 is right. P(G<120) was 0/200 resamples, so the tier-2 contingency is very unlikely to fire (4 pairs per method needed vs mean 16.6).
- Integer rule vs `s28_metrics` float rule (`pooled.p` rounded to 4 dp, `>= 0.95`): identical for G up to several hundred (a failing pair count is ≥ 1/(20G) below 0.95); no divergence at G≈480.
- Mechanism for the P0 record: `s28_run.py:45` re-reads and overwrites `s28-run.json` `<cfg>.extract`; copying it aside first is sufficient, and the P0 facts files persist under different `@AUDIT_ID@` names (sha-pinned).

## Findings

### B1 (blocker) — the figure that decides is computed outside any pinned tool, by an adjudicator who sees the margin; row 7's second reading is undefined

Evidence: §1.4 lets a miss leave `G`; `s28_metrics.py` never reads a `*.adj.jsonl` (grep: no hit; only the `-p1-ml.md` appendix `flow.py` does, ad hoc, with a hard-coded `m[22] += 1`).
So `y` and `G` after rulings (and after F1/F2) are computed by an unnamed, unpinned procedure, and the record pins neither it nor `s28_metrics` output semantics for it.
The adjudicator (`data-scientist`) rules every miss after the counts exist, knowing `456/480` is the bar; the dev result sat at zero margin, so one ruling can flip TRUE/FALSE.
Row 7 / §2.4 say a "late" ruling is disclosed and the decision computed "under both readings", but `readings` is only `["as-registered"]`: the other reading is never defined, so row 7 cannot fire and is not falsifiable.
Also every miss ruling is by construction post-results, so "late" as defined covers all of them.
Fix (text, before U2): (a) name and pin (sha in the record) a decision script that takes `y`, `G` from scoring output plus the rulings file and prints `20y`, `19G`; add a mutant test (one planted overturned miss moves `G` by one). (b) Define the two readings now (e.g. reading A: rulings as made; reading B: no miss leaves `G`; and for F2, omissions not added) and make row 7 compare exactly those. (c) Adjudicate from a miss list without cell totals, in a fixed order, and have `analyst` (not the method author) re-rule any ruling set whose removal moves `20y − 19G` across 0. (d) Say which cell's misses are adjudicated (decision cell only; the shipped/old cell has ~170 misses on dev, adjudicating those would be both heavy and a leak of diagnostic cells into `G`).

### M1 (major) — procedural control of the §3.4 residual is not made structural, and U1's own gate can trip it

Evidence: §3.4 relies on "forbids" and "`teco` does not diff". But `flowdraw-p1` run without `--exclude` (or by the U1 test/`analyst` gate against the real corpus with the real seed) writes the unfiltered order, which is exactly the thing to protect; the test list in §0.4 ("reproduces P0's `flowdraw` byte for byte with the P0 prefix and no exclusions") also contradicts the frame (the p1 frame subtracts the 30 P0 methods, so P0 byte-reproduction needs a mode flag the note does not name).
Fix: `flowdraw-p1` must fail closed without `--exclude` and `--counts`, refuse the `ccg-p1a-v1-flow` seed on any non-custodian path, never run in tests/gates except on a synthetic mini-corpus with a test seed (stated in §0.4 and in U1's gate text), and error paths must not echo the private file's path or lines (planted malformed-line test). Add a pin (sha) of `run_fitness.py` at U2 and a custodian step 0: verify that sha before feeding the private file to it; the record currently fills the tool shas only at U4, after the draw.

### M2 (major) — a `k2` return and a "forbidden to compute" rule are weaker than a commit-then-reveal salt

`excludedTier1Prefix30` (and `…Tier2Prefix15`) is returned and recorded in the public record, but nothing decides on it; it only tells the team that a held-out anchor sits in public hash-top-30, which is what makes the §3.4 diff worth doing. With E[k2] ≈ 0.09·k1 (30/338) it is usually 0.
Fix: stop returning `Prefix30`/`Prefix15` to the team (the custodian keeps them; the record needs only `k1`, `k3`). Optionally and better: the custodian adds a private salt to the seed, commits `sha256(salt)` in the record, reveals the salt after the `ac5-gate-1` freeze; this removes the residual completely at the cost of the team being unable to verify the draw until then. The note's rejection of "exclude all eight public members" is right (it needs the team to compute all eight identities, vs an expected ≈0.1–0.3 leaked) but note the anchor list is derivable from a public seed (`eval-manifest.yaml` `flowAnchors.seeds`) and `export_kit.py`: custody here is procedural for both designs. The §3.4 "8%" is 30/368; on the 338-method frame it is 8.9%. The sample files themselves do not reveal skipped positions (no hashes or ranks in them) apart from the diff above; the note's "committed by digest only" for tier-2 conflicts with §3.3 "the two sample files" returned: pick one.

### M3 (major) — P1-34a: outputs of the four scoring invocations collide; no regression test that P0 scoring is unchanged

§4 step 3 runs four `s28_metrics` invocations (decision, `--join p0`, and the two shipped-record ones) that all write `results/projection-python-p1*.json` and `adjudication/python-flow-p1-{misses,spurious}.jsonl` (`s28_metrics.py` writes fixed filenames per language; the miss/spurious files are the ones the adjudicator edits and the spurious sample comes from `off` of the decision cell). Later runs clobber earlier ones and the rulings. Fix: per-cell output names (or a `--out-suffix`) and run the decision cell last or into a separate directory; the adjudication files are written by the decision cell only. Add to §0.4: a default-flags run (`--flow-set p0 --join p0` with the P0 record) reproduces the committed `results/projection-python.json` byte for byte (modulo timestamps), and a diff of `s28_metrics.py` shows no change in the scoring path; `s28_metrics.py` decides the figure, so C5's "no scorer change after a fresh-draw seed" must be read to cover it (the note pins it only via a placeholder, `row 8` omits it, and its edit lands after the seeds are published in this very document: state that P1-34a is behavior-preserving and tested as such). P1-34a has no row in `-p1-gates.md` (P1-34 step 1 mentions only `s28_run`); add one.

### M4 (major) — tier-2 trigger text is contradictory and can make the sample size depend on results

§1.1: trigger is "adjudicated golden pairs of the 30 methods < 120" but "same labeller, same dispatch": adjudication (misses) occurs after scoring, so a trigger on adjudicated `G` cannot be pre-dispatch. Fix: trigger on the labeled pair count that `flowcheck` reports at the end of the labelling dispatch (before any audit or Joern output); G<120 after tier 2 is row 3. Also say tier-2 methods (26–60 lines) are pooled unweighted, which changes the estimand; say that explicitly or report tier-1 and pooled both (decision on pooled per P0-ML §8.4).

### M5 (major) — row-precedence and completeness gaps in the nine-row table; record pins incomplete

Rows 2/5 overlap (one config fails recall, the other passes: row 2 says FALSE:measured, row 5 says a defect is filed), rows 3/4 can both hold; give a precedence. Row 6 ("a Part F escalation that cannot complete") and row 7 are not observable as written. Row 8 lists scorer/`extract.sc`/`s28_run.py` only; add `s28_metrics.py`, `run_fitness.py`, the CPG and corpus digests, the toolchain image digest (`toolchainImage` is `None` in `s28_metrics.py`'s output), and the decision script (B1). The decision record should also carry the fixed facts files' sha256. The shipped facts' "produced by `spikes/vertical-slice/extract.sc`" claim is not checkable from the P0 run record (`extractSha256` is `None` there); say "per the P0 report", or drop the claim, since the shas pin the files anyway.

### m1 (minor) — transcript grep list misses the held-out/eval paths
Add `cpg/eval`, `heldout`, `flow-anchors`, `.local/share`, `EVAL_COMMIT`, `kit-src`; the labeller may read the whole corpus root, which includes this repo's tests. "docs/" and "cpg/" patterns clash with the permitted output path `cpg/spikes/…`; state the grep as exact-path allowlist rather than substrings.

### m2 (minor) — F1/F2 design
F1 prioritising `unsure` pairs (up to 20 of 60) inflates the invalid rate against the 0.05 flag; fine as conservative but the reported rate must not be quoted as the population rate. F2's second `qa-engineer` shares model and brief with the labeller, so correlated omissions bias the omission rate down; keep the flag as a qualifier only (it is). One-sided adjudication (misses can leave `G`, connected pairs are audited at ~12%) is an upward ratchet; the note's §6 states it, but say how large an undetected invalid rate would matter: a 2% invalid rate (~10 pairs) is the whole 464/480 margin and F1 (flag at ≥4/60) would not see it.

### m3 (minor) — counts and wording
§1.2 says "three custodian counts"; §3.2/§3.3 return four integers and the record has three fields (`excludedTier2Prefix15` unrecorded). Tier-2 python frame membership: `cmd_flowdraw` writes tier-2 only for `CENSUS_LANGS = {"javascript"}`, so python tier-2 is new code in `flowdraw-p1`; list it in P1-34a's tests (15 methods, disjoint from tier-1).

### m4 (minor) — resources
"2–3× estimate" for the fixed extract is unmeasured and 3× of the 1.2–1.3 GiB peak would hit the note's own 3 GiB abort rule; but the abort rule makes it non-load-bearing for the decision. The fixed projection was already run on this CPG in P1-09 (`cpg/.ccg-work/gate-p1-09`/`p1-09`); cite its measured wall/peak instead of an estimate. Host now: 5.9 GiB total, 4.2 GiB available (verified with `free -m`).

### m5 (minor) — document hygiene (growth-stops, `Extended by:`)
New document, so no revision growth. But it restates settled content against its own "cited, not restated" line: §1.4 rulings are P0-ML §8.4 text "restated"; §1.5 and the "Reading aid" repeat `-p1-ml.md` §3.1 numbers; the opening "Verdict on the brief's question… I am not stopping" is narration (a plan states the design). Cite instead. `docs/plans/connected-code-graph-p1-ml.md` header lacks `**Extended by:**` for this note (header pointer is the permitted edit); `teco` adds it. Coordination ledger already records delivery (`-coordination.md:809`).

## What's solid

The draw is an unbiased SRS of the stated frame: hash order is independent of the exclusion (anchor identity comes from a different seed), `siteKey` is unique, and the exclusion by file+span avoids a qname grammar. The extrapolation gap (anchors excluded from the measured population though the flag will be used on anchor units) is small (k1 methods out of 338) and is disclosed. The integer rule, the 0.55/0.2 probabilities, the four-cell discipline (only fixed/new decides), the core-freeze pins and the P0-record-aside mechanism check out. The custodian's step list leaks no identity beyond what §3.4 admits (private file stays outside the repo, counts only, no hashes/ranks in the sample files). `JOIN_P0_ONLY` runtime switch is sound.

## Open questions

1. Who computes `y`/`G` after rulings, with what script (B1)? 2. Is the salted-seed alternative (M2) acceptable to the stakeholder given the verification delay? 3. Does U1's gate (`analyst`) run on a synthetic corpus only (M1)?

## Pass 2 (revision in place; static; P1-34a tooling mid-edit, not judged)

**Verdict: needs changes** (no new blocker in the sense of an unsafe design; one decidability defect, B2, is a blocker for a pre-registration because it makes the pass rule ambiguous). Rechecked by re-reading the revised note, `-p1-ml.md` header and `-p1-gates.md` P1-34a item 5.

**Dispositions.**
- B1 (pinned script, readings, decision cell only, M reported): partly fixed. `flow_decide.py` is named and pinned in `-v1`, mutant tests listed, `M = 20y − 19G` correct (an invalid connected pair: `M−1`; an overturned miss: `G−1`, `M+19`; both verified by hand), two readings defined. Not fixed: see B2, B3.
- M1 (fail-closed tool, record split, custodian step 0): fixed in text; sequencing U1→U2 (`-v1` all pins)→U3→U4 (`-draw`)→U5 is sound. See M6, m6.
- M2 (no salt, only k1/k3 returned): ruling on the salt accepted (the unauditable re-roll argument holds). Prefix counts dropped: fixed. Committing the tier-2 file: regression, see M7.
- M3 (per-cell dirs, decision cell writes miss/spurious, regression test): fixed; one nit in m6. `-p1-gates.md` item 5 and `Extended by:` on `-p1-ml.md`: present, verified.
- M4 (tier-2 trigger on the `flowcheck` count): fixed.
- M5 (six ordered rows): total and mostly exclusive; see B2, m7.
- m1 (grep allowlist), m2 (F1 = 60 uniform, unsure separate, `M` as the control, correlation noted), m3 (two counts), m4 (resources honestly "unmeasured"), m5 (growth/restatement, narration removed): fixed. Remaining: the unsure-set invalid pairs' effect on `G` is not stated (m7).

### B2 (blocker) — the reading rule can never yield TRUE when A and B differ; "WITHHELD" semantics undefined
§5 *Readings*: A (rulings as made) removes pairs from `G`, so A's recall ≥ B's always; a disagreement therefore means A passes, B fails. The note then recomputes A on "pairs overturned by both" (A′) and says "if that verdict still differs from B's the outcome is row 4". B fails, so A′ ≠ B means A′ passes, which is sent to WITHHELD; A′ = B (fail) is FALSE. TRUE is unreachable unless B itself passes, which makes every ruling moot for the verdict, contradicting "the outcome rests on the pairs both adjudicators overturned". Fix: state the outcome rule as one sentence, e.g. "if A and B disagree, the outcome is the verdict of A′ (TRUE if A′ passes, else FALSE:measured); row 4 only if the second adjudicator cannot complete", or, if the intent is conservative ("TRUE needs B"), say so and drop the ruling step. Also give WITHHELD a consequence (stays false, spent draw, no second draw, same as row 5) so it is not an open door. Row 2 says "both readings" for `G < 120`: say B (the larger `G`) or "either", else a state with A `G` = 119, B `G` = 121 matches row 4 or 2 by reading.

### B3 (major) — "blind to totals" is nominal: the miss list length gives `M`
`data-scientist` holds the frozen truth (`G` known, U6) and receives a list of misses of known length `n`: `y = G − n`, so `M = G − 20n` is computable by counting. With `M` near 0 each ruling's effect is known (+19). Same for the analyst's re-ruling "restricted to those pairs": it sees only the candidates the first adjudicator overturned (priming toward confirming) and their number. Fix: adjudicate a seeded-shuffled list of all decision-cell misses **plus a seeded sample of connected pairs of at least equal size** (unlabelled), from the same hash rule; rule on the union; both adjudicators rule the whole list (the intersection of overturned misses is then the analyst-confirmed set). This hides `n` and the miss/hit status, and gives the missing two-sided check on connected pairs (an invalid rate among hits is estimated and reported beside `M`, closing the one-sided ratchet of §2.4's disclosed limit). Also make the analyst who re-rules different from the `analyst` gating U8.

### M6 (major) — the draw tool is environment-sensitive and untested on the real frame
The p1 frame (338 / tier-2 163) depends on the tree-sitter grammar versions in the custodian's venv; step 1 checks the corpus digest only, and U1 is synthetic-only, so no check ever asserts the frame is 368/338/163 before the private file is applied. A different grammar silently changes membership, hence the whole order. Fix: `flowdraw-p1` asserts population = 368, minus P0 = 338, tier-2 population = 163 (public numbers, `§0` re-derived) and aborts otherwise; add a test of that assertion on the real corpus with a **test seed** and a dummy exclusion (counts only), which the digest guard permits. Also require a non-empty `--exclude` file (the 8 python anchors always exist): an empty or dummy file on the real seed would write the unfiltered first 30, which fail-closed does not prevent.

### M7 (major) — committing the tier-2 file widens the residual for no use
The previous design kept tier-2 as digest only (expected unused, 0 of 200). Committing the 15 methods adds a second diff target (anchors in the 163-method tier-2 frame, 9.2% each per anchor in the first 15) and nothing in the decision needs the file before it is triggered. Revert: commit `tier2Sha256` in `-draw`, release the file to `teco` only if the count is < 120 (the digest still prevents a post-hoc choice). `k3` is harmless and may stay. The two returned integers leak nothing the earlier four did not; dropping the prefix counts is a strict improvement.

### Minor
- m6: regression test must write to a temp dir (not overwrite the committed `results/projection-python.json`) and read `s28-run.shipped-p0.json`, since `s28-run.json` is overwritten at U7; skip when `.ccg-work` is absent. `run_fitness.py` pins one file only (stdlib, `yaml`, tree-sitter): add the tree-sitter and grammar versions the custodian's venv reports to the `-draw` record.
- m7: state whether F1's separate unsure-set invalid pairs leave `G` at truth freeze; row 5 and rows 2–3 should say they evaluate on the post-analyst reading; the second adjudicator (`analyst`) also gates U8 (B3).
- Pass probability claims (0.55 / 0.2) remain verified; the bar 456 of 480 and `M` tolerance statements are correct.

## Pass 3 (confined to §7, Pass 2 dispositions and same-revision interactions; static)

**Verdict: needs changes (small).** U2 may proceed once N1–N3 below are folded in; none changes the design. Nothing under the custodian root was opened; no unfiltered order was computed.

**§7 items.**
1. True and verified now (`git cat-file -t` = `commit`; digest `365761fb…` re-derived in Pass 1).
2. True and verified now (`git diff 3089951f HEAD` empty on both files; shas and `JOIN_P0_ONLY = False` re-checked in Pass 1).
3. Coherent with §0.1/§1.2/§4 U1; not yet checkable (P1-34a uncommitted); `s28_run.py` sha 54fc1bea…09b2 verified.
4. Matches §0.1 text, with one gap: see N3 (the 368/338/163 assertion must not run on the synthetic corpus).
5. True: §0.1 and §5 both define the verdict from `D` only; `D = ∅` is diagnostic.
6. Coherent; digests recompute (Pass 1). The adjudication seed `ccg-p1a-v1-adj|` is not in the record's `seeds` (m-a below).
7. Self-referential but workable; §3.2 step 3 is wrong as written: see N3.
8. True: `Extended by:` present in `-p1-ml.md` header; gates item 5 names P1-34a.

**Pass 2 dispositions.** B2 fixed (one outcome rule; WITHHELD and the readings are gone; TRUE is reachable; `D` is an intersection, so one-adjudicator rulings never move the figure). B3 fixed in design (fixed-length shuffled two-sided list, two independent adjudicators, the ruling `analyst` is not the U8 gate) but see N1. M6 fixed (368/338/163 assertion, non-empty `--exclude`, real-corpus test with a test seed; versions in `-draw`) but see N3. M7 fixed (`--tier2-out` outside the repo, digest only, released only if labelled count < 120; §1.1, §1.2, §3.3, U5 agree). Minors fixed: regression test to a temp dir with the copied shipped record, F1 unsure-set rule, rows evaluated with `D` removed.

**Interaction checks.**
- Length-120 list and `M = G − 20n`: for n ≤ 60 the list is 120 rows whatever n is, so n is hidden. For n > 60 the list length is n (status revealed), but then `M0 = G − 20n` ≤ about −740 for G ≤ 480, and rescue would need ≥ ~39 double-overturned misses; with 30 methods G cannot reach the ~1220 that would make n = 61 matter. So no decision-relevant leak. Fine.
- Five rows, first match: total and exclusive for the states reachable with complete inputs (G < 120 with a precondition failure goes to row 2; row 4 includes off/on discordance; row 5 is the complement). `D` removes the same pair from `G` in both configurations and from `y` only where connected, so per-configuration accounting in §0.1 is consistent.
- Residual one-sidedness: misses are audited at 100%, connected pairs at about (120 − n)/(G − n), so an invalid connected pair is removed with lower probability than an invalid miss; the note's §2.4 "Limit, disclosed" still says connected pairs are audited "only through F1's 60", which is now stale (m-b).

**New findings.**
- **N1 (major) — the blindness of §2.5 is not enforceable as written.** After U7 the repo holds `results/p1/*/…answers.jsonl` and, per §0.1, the decision cell writes `adjudication/python-flow-p1-misses.jsonl`, "the files the adjudicator uses": both state which pairs are misses and how many, which defeats the hidden-`n` and hidden-status design; §0.1 and §2.5 also disagree on what the adjudicator is handed. Fix: the adjudicators receive only the `teco`-built list (no miss file); §2.5 forbids reading `results/`, `adjudication/*misses*` and facts, with the §2.3 style transcript check (exact-path allowlist) for both rulings dispatches; `flow_decide.py` output stays withheld until both rulings files are committed. Verified by reading the text; the files' contents inferred from `s28_metrics.py` as read in Pass 1.
- **N2 (major) — table incompleteness.** No row covers a rulings file missing or not covering every list row (the second adjudicator cannot complete), or a released tier-2 file whose sha256 differs from the recorded one. Add both to row 1 (`NOT-MEASURED:void`); and say `teco` checks the tier-2 sha256 at release.
- **N3 (major, custodian step and tool spec).** §3.2 step 3's command line lacks `--tier2-out <path outside the repo>` although the tool fails closed without it, so the literal step aborts. And §0.1 makes the 368/338/163 assertion unconditional, which a synthetic mini-corpus cannot satisfy: state it applies when the corpus digest equals `srcDigest` (or via an injected expected-counts parameter, with the abort mutant tested on a planted discrepancy), otherwise the synthetic tests cannot pass or the implementer weakens the assertion unnoticed.
- m-a: add `ccg-p1a-v1-adj|python|<commit>` to the record's `seeds`. m-b: delete or rewrite §2.4 "Limit, disclosed" to reflect §2.5's two-sided sample and report the invalid rate among sampled connected pairs beside `M`.
