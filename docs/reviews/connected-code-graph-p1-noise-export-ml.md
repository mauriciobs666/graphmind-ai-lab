# Connected code graph, P1-36: methodology review of the noise-export note (rulings F1-F5)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** P1-36 (P1)

## Pass 1

**Scope.** `docs/plans/connected-code-graph-p1-noise-export-graph.md` (graph-dba), against
`docs/reviews/connected-code-graph-p1-36-impl.md`, `cpg/eval/tools/noise.py`, `docs/plans/connected-code-graph-p1-ml.md` §5.1, §5.3, §6.5,
EP §5.2, §11 (`docs/plans/connected-code-graph-eval-protocol-ml.md`), `docs/plans/connected-code-graph-p1-gates.md` P1-36/P1-37,
`docs/plans/connected-code-graph-answer-spec.md` (golden AC-1 answer), `docs/plans/connected-code-graph-answer-spec2.md` §4.1,
`cpg/eval/tools/match_keys.py` (`_regions`, `h1_symbol_credit`), `cpg/ccg/engine/config.py` and the dev gold files
(`cpg/eval/dev/ac5-*.jsonl`, item types and `counting` only). No held-out file, custodian directory, store or container was read; I ran no
suite (static review plus a count over dev gold). `CPG: considered, not relevant: the questions are method choices over documents and one
constant; the cpg_* graphs do not model them.`

**Verdict: needs changes** (on the note's §4 rule 3/4 and §6, before `ccg-noise-recall-rule-v1` is written; the record pins the
sha256 of §6 and `baseline.py`, so the amendments must land first). The predicate of F1 is sound once amended (A1-A4 below). F4 and F5 are
accepted. F2 and F3 are ruled, with one stakeholder decision (S1) that does not block the gate-1 report. One defect in rule 3 biases the
reduction in the product's favour (M1).

## Rulings

### F1 — the item-level recall predicate: **sound, with four amendments**

**Predicate.** Gold item `g` is a baseline hit iff `g.impacted.type == "symbol"` and some `symbolId` in the baseline's item list **for
`g.anchorId`** names a snapshot `symbol` record `x` with `h1_symbol_credit(x, g.impacted, g.lines)` (E-normalised qname equal **and** a line of
`g.lines` in the span or in one owned-region pair). It is EP §5.2 H1 plus "listed", the same lines/qname test the system must pass, imported,
not reimplemented. Checked: `_regions` accepts `(a, b)` pairs and indexes `r[0]`, `r[1]`, so a flat store `ownedRegions` would raise or
mis-credit; the note's re-pairing is required and correct. Per-anchor matching (never pooled across anchors) is the right scope; I checked it
is the note's text.

**Fairness.** H2-H4 dropped is a relaxation that can only raise the baseline's hit count, i.e. it favours the comparator. Gaming
channel: none beyond the baseline's own burden, because the item list is fixed by deterministic rules from the snapshot and the anchor before any
gold is joined; the predicate adds no tunable. The one structural asymmetry, config gold items never hit, is by design (below).

**Denominator `n`.** Same `n` for both figures is right, but "every `counting: true` gold item" must be stated as **the set the harness scores,
not an independently recomputed filter**, otherwise equal `n` can hide different sets (the note's check compares only the integers). EP §5.1/P1-37:
the flow stratum is not run when its flag is false, the system's figure is as-run and ex-key-error side by side, and an anchor that does not
resolve gives an answer with no items.

Amendments to §6 (all required before the record):

- **A1 (population identity).** `baseline_recall` takes the harness's scored list of counting gold items, not the gold files. Both pairs carry
  `itemSetSha256` = sha256 of the sorted `(anchorId, impacted-key)` list; `score` exits 4 (counts only) when they differ, not only when `n` differs.
  For the ex-key-error figure the same items are removed from both, and the baseline pair is recomputed on that set.
- **A2 (unresolved anchors).** The note's "an anchor id absent from the snapshot raises" is wrong for this use: the system's answer for an
  unresolved or §5.4 key-error anchor exists with burden 0. Rule: such an anchor's items stay in `n` and are misses for both; the generator writes
  an explicit row `{anchorId, count: 0, countInScope: 0, unresolved: true}`; `score` prints the number of such rows. Burden 0 on both sides adds
  nothing to either Σ, so the ratio is unaffected.
- **A3 (config items).** Keep the headline pair over all counting items (the pre-registered "same hit rule", `n` = 159-type population), and
  report a second pair over **symbol-type items only** (`n_sym`). Without it the flag can fire purely from the baseline's structural zero on
  `config` items, which says nothing about its listing strategy. The writer prints the config gold count (as the note says) and `n_sym`.
- **A4 (paired and two-sided reporting).** Report the 2x2 of the same items: both hit, system only, baseline only, neither, beside the two
  Wilson intervals (never judge two marginal intervals by overlap). Reason: the pre-registered flag fires only when baseline recall is **lower**.
  Reduction is a burden ratio, and the baseline at the system's recall would need more burden if its recall is lower (so the reduction is understated,
  conservative) and less burden if its recall is higher (so the reduction is overstated). The dangerous direction, baseline recall **higher**, is
  not flagged. Keep the flag as ratified (literal rule, cross-multiplied, exact) and add, reported only: "baseline recall higher than system's: the
  reduction is not at equal recall and is an upper bound". This extends reporting, not the ratified bar.

**Non-gameability.** Passes: the baseline list is a function of (snapshot, anchor, bound, rule version); the gold is joined afterwards; the rule is
committed before any answer is scored. Residual: the same person who can see held-out answers must not run the generator before the record; the
generator needs only the snapshot and the anchors, and A2's rows reveal only unresolved counts.

**Record text** (`cpg/eval/decisions/ccg-noise-recall-rule-v1.json`; `teco` fills the two hashes after the amendments land):

```json
{
  "id": "ccg-noise-recall-rule-v1",
  "decides": "AC-5 gate 1 noise: the naive over-lister's item-level recall and reduction (P1-36, -p1-ml.md 5.1 item 3)",
  "owner": "data-scientist",
  "rule": "A gold item g is a baseline hit iff g.impacted.type == 'symbol' and some symbolId in the over-lister's item list for g.anchorId names a snapshot symbol x with match_keys.h1_symbol_credit(x, g.impacted, g.lines) (E(x.qname) == g.impacted.qname and a line of g.lines in [x.lineStart, x.lineEnd] or in one pair of x.ownedRegions, the flat store list re-paired). EP 5.2 H2-H4 are not applied to the baseline. A config gold item is never a baseline hit. Matching is per anchor.",
  "population": "n is the set of counting:true gold items the gate harness scores, identical for the system and the baseline, identified by itemSetSha256 (sorted (anchorId, impacted key)); the as-run figure and the ex-key-error figure are each computed on the same item set for both. Items under an anchor that does not resolve stay in n and are misses for both. Also reported: the pair over symbol-type items only.",
  "figures": {"system": "[k, n] from the harness", "baseline": "[k, n] from this rule", "paired": "2x2 over the same items, plus Wilson intervals"},
  "flag": "reduction is flagged iff baseline_k * system_n < system_k * baseline_n (exact). Reported, not a bar: if baseline recall is higher than the system's the reduction is not at equal recall and is an upper bound.",
  "baselineVariants": ["kind-wide (pre-registered reading, headline)", "key-scoped (reported beside it)"],
  "bound": "the baseline's bound is the ac5-gate-1 record's defaultDepthBound; the generator and score refuse unless every answer's depthBound.bound equals it",
  "burdenPairing": "in-scope reduction = sum(system in-scope burden) over sum(baseline countInScope); all-listed reduction = sum(system all-listed) over sum(baseline count); never crossed",
  "sha256": {"noise-export-note-section-6": "<teco>", "baseline.py": "<teco>"},
  "rulesChangedAfterScoring": "counts as after adaptation and is reported as such"
}
```

### F2 — kind-wide versus key-scoped: **literal kind-wide is the pre-registered reading; report key-scoped beside it; S1 decides what gates**

**Which is pre-registered.** Every source says "kind", never "contract" or "key": ML §4.2 (`connected-code-graph-ml.md`) "lists every name-match and every
provider of the kind"; `-p1-ml.md` §5.1 (3) and P1-36 "every provider and consumer of the same contract kind", syntactically a clause
separate from "within bound 2 reachable by...". The literal, unbounded, anchor-derived reading in the note's rule 4 is therefore the ratified
baseline. I verified no document narrows it to the key.

**What each does to the reduction.** I did not measure either (no snapshot, and held-out is closed to me); I state direction only. Kind-wide adds
every provider and consumer of the kind in the repository to every anchor's list that has a kind. For a contract anchor the gold items are
consumers of the key, which are by construction members of the kind-wide set: baseline recall on those anchors then tracks the system's at the cost
of a burden that scales with the number of kind members, not with the anchor. So the ≥ 5x reduction becomes a statement about how few of the kind's
members belong to the key, which is a property of the repository more than of the system's filtering; it can pass with little learned. Key-scoped (the
anchor's own contract points, unbounded, members only) is the baseline an engineer would run (grep the key), is a smaller list, and gives a smaller
reduction. The `-p1-ml.md` §5.1 sentence "a system that equals its baseline has learned nothing" is tested far better by key-scoped. I cannot say
whether kind-wide is vacuous without numbers; the cheap measurement is the dev dry run below.

**Ruling.** (1) Kind-wide, as written in rule 4, is the headline and the figure the record names. (2) The generator also emits the key-scoped
variant (a `kind_scope` parameter, default kind-wide: for a contract anchor its own `cpId`'s providers and consumers; for a symbol anchor the `cpId`s
it provides or consumes; for a config anchor the `cpId`s of its bound links) with its own `count`, `countInScope`, items and recall pair, reported
beside the headline with no threshold. Same record, same hash. (3) Reduction gates only from the next gate (`-p1-ml.md` §5.3), so nothing gated
is decided by gate 1's choice.

**S1 (stakeholder decision, not blocking gate 1; needed before the next gate's pre-registration).** Which reduction carries the ≥ 5x bar: **(a)** kind-wide, the
ratified wording, accepted as a permissive bar; **(b)** key-scoped, the strict engineer's baseline, which changes what was ratified and so needs
the stakeholder's explicit say; **(c)** both must hold, my recommendation, because it keeps the ratified figure and adds the one that can fail. I
recommend taking the decision **before** anyone sees dev or held-out reductions for the variants, so it is not chosen by the number.

**Coupled sub-choice (same record).** Rule 4 gives a **symbol** anchor a kind through the contract points it provides or whose consumes site it owns.
A symbol anchor has no contract kind of its own (script and flow anchors in particular: a script that calls one MCP tool receives the whole
MCP kind). That is an extension of the pre-registered text, upward on burden. Ruling: keep it (it is the same "participating in the kind" reading
under which the symbol is an impact source), but the report prints burden and reduction **by anchor type**, so the effect is visible and not buried in
the pooled sum.

### F3 — which burden pairs which reduction: **like with like; `count` with all-listed, `countInScope` with in-scope**

The note's edit 1 is right, and the committed `noise.py` has the defect it describes, in two places: `score` feeds `base = baseline[a]` (the
`count`) to both `reduction(base, bin_, ...)` and `reduction(base, ball, ...)`, and `bootstrap` divides `g["base"]` (also `count`) by `g["burden"]`,
which is the in-scope burden. So the committed in-scope reduction and its interval cross an all-listed numerator with an in-scope denominator.

**Ruling.** In-scope reduction = `Σ countInScope / Σ system in-scope burden`; all-listed =
`Σ count / Σ system all-listed burden`. Headline is **in-scope**, the frame the thresholds were conceived on (`-p1-ml.md` §5.1 item 2) and the one
the ratified p90 uses; the all-listed value is reported with the same flag. The bootstrap interval must be computed per pairing (baseline column
`countInScope` with in-scope burden, `count` with all-listed), so edit 1 also changes `_agg`/`bootstrap`, not only `score`. The in-scope filter on
the baseline (`path_excluded` over symbol paths) is the same predicate the frame uses on item paths, so the scope is symmetric.

**Symbols only versus the system's wider item types.** The baseline lists symbols only; the system's burden also counts `config` and `unit`
items. That asymmetry lowers the baseline's burden, so it is conservative against the reduction. Keep, and print the number of non-symbol system
items so the reader can see it. No change to the figure.

### F4 — `bound`: **2 is the pre-registered default; bind it by construction, not by convention**

`DEFAULT_DEPTH_BOUND = 2` (`cpg/ccg/engine/config.py`), spec2 §4.1 (the bound is not request-settable, V17 requires `depthBound.bound >= 2`), and EP
§12.3's record shape carries `defaultDepthBound: 2` (the example value is illustrative, but it is the shipped constant). The `ac5-gate-1` record is not
yet in `cpg/eval/decisions/`, so I read the shipped constant rather than a committed value. Ruling: accept 2, and make the baseline's bound **an input
read from the `ac5-gate-1` record's `defaultDepthBound`**, with `score` refusing (exit 4) unless every answer's `depthBound.bound` equals it; this
removes the "baseline not recomputed" case instead of ruling on it. A sensitivity at bound 1 and 3 is optional and reported only. The bound's *unit*
is not left open: see M1.

### F5 — the oracle's Link set: **omission accepted as conservative, with one additional report; `possible` counts as present, with a status split**

- **Provider-less links omitted.** Omission can only remove matches, so the oracle recall that results is a lower bound; accepted. Two cautions. (i) The
  note cites `-p1-ml.md` §6.5 for "every status counts"; §6.5 says only "present in the gate snapshot as a `Link`" and I found no "every status"
  text in the file, so the status rule is this note's (accepted below), not an inherited one. (ii) Oracle misses are triaged "pack, extractor or
  dynamic" and feed the next held-out set (§6.5); a miss that is really a provider-less link would be triaged as extractor. Required: the writer
  also emits `links-unprovided.jsonl` (same fields, custodian side) and `oracle.py` classifies a miss whose `(element, key)` appears there as
  `linked-no-provider` rather than `extractor`; counts only, as now. Print the number of omitted rows whose key lies under `/shop/api` (the oracle's
  scope) so a zero is visible.
- **`possible` links counted as present.** Accepted: oracle recall measures whether the static side captured the observed edge as a Link, and a `possible`
  link is that. It must not be read as confirmed coverage. Required: the oracle reports hits by the best status of the matching Link (`confirmed`
  versus `possible` only), counts only, no threshold, so a figure supported mostly by `possible` links is visible. The dump already carries
  `status`; `oracle.py` currently ignores it.
- Distinct `(element, keyNorm)` with best status then lowest `linkId` is deterministic; accepted.

## Findings

**M1 (major, by reading the system's golden answer and the note's rule 3) The baseline reaches one call step deeper than the system for contract
and config anchors, which inflates the reduction.** Rule 3 says seeds cost 0 and "provider or consumer membership never costs a hop", justified by "a
provider is at distance 0, as in the system's answer". The golden contract-mode answer (`docs/plans/connected-code-graph-answer-spec.md`, the AC-1
golden, and spec2 `B-chain`/`B-wire`) lists the contract's consumer symbol at `distance 1`, `direct`: the contract-to-consumer step costs one hop in the
system. Then at bound 2 the system lists the consumer (1) and the consumer's callers (2); the baseline lists the consumers (0), their callers (1) and
their callers' callers (2), one level more. More baseline items raise the baseline's burden, hence the reduction, and also its recall. Fix: in the
baseline, a consumer reached from a contract or config anchor costs 1, so reach from consumer seeds is `bound - 1` steps; providers, if the system lists
them at 0, stay at 0. I could not confirm the provider and config cases from the documents alone. **Validity check for the tests (cheap, decisive):**
on the dev anchors, with the dev snapshot's export, assert that the baseline's symbol set is a **superset** of the system's symbol items within the
bound for the same anchor (modulo items the system lists through non-call edges the baseline lacks); every violation is either a baseline bug or a
distance mismatch, and the count of violations is reported. A baseline that does not dominate the system's own listing is not a naive over-lister.

**M2 (major) Rule 4 on symbol anchors, rule 1 on already-resolved sites, and the flag's direction are upward or one-sided choices the record must name.**
They are not defects against the pre-registered text, but each inflates the baseline's burden and is fixed by the same record: name candidates of
sites that are also resolved count (literal "every name-match candidate of every call hop"); symbol anchors acquire a kind from their own contract
points (F2 coupled sub-choice); the flag fires only for lower baseline recall (A4). The record text above names all three; the report prints
burden by anchor type and the number of baseline items contributed by name rows on resolved sites versus unresolved ones.

**m1 (minor) The frame can hold duplicate `(anchorId, itemId)` rows.** `build_frame` appends every listed item and group member; labels are keyed by
`(anchorId, itemId)`, so a symbol listed both as an item and as a group member would count twice in `N_h`, draw twice and share one label. Fix: assert
uniqueness in the frame and exit 4 with the duplicate count (never ids); the answer spec says the system lists once, so the assertion should never fire.
Test: plant one duplicate.

**m2 (minor) The spot check is drawn from census plus sample, but the instability flag is applied to possible precision.** The 20 keys come from all
labelled items; confirmed items are likely most of them, so the figure that is flagged may rest on few or no possible-item pairs. Fix without changing
EP: report agreement split by confirmed and possible; when fewer than 10 spot keys are possible items, the possible-precision note is "agreement not
measured on possible items", the same wording class as the `n = 0` case. A Wilson interval on agreement at n = 20 is wide; report it.

**m3 (minor) A key id that does not resolve in `symbol` rows (A2) and an anchor's type projection.** `anchors` come from "the resolved anchor" in each
answer; a `unit` anchor arrives as its probe contract anchor (note §4). The generator must emit exactly one row per **answer** (not per gold anchor),
else a unit anchor and its probe contract anchor could be counted twice or never. Test: a `unit` and its probe in one run give one row.

**n1 (nit)** The note's worked graph pins `bound` 3 giving `D` as a mutant check; add the same row for a **contract** anchor, since M1 moves exactly that
count (with the M1 fix, `CP1` at bound 2 loses the callers two steps from `H`).
**n2 (nit)** "Reduction at equal recall" in EP §11.3 and ML §4.2 is not computed by Σ burden ratio unless recalls are close; the report should say the
figure is a burden ratio with both recalls beside it, not "at equal recall" (A4 supplies the data).

## The draw choices in `noise.py` ("Choices not fixed by EP §11") to pin

Pin each in `-p1-ml.md` §5.1 or in the `ac5-gate-1` record (the gate review's open question 2): I pin them in the **prereg record**, because they are
code-level facts a verifier needs to reproduce and the record is the pre-run commitment. Each choice below is accepted as implemented, with the one
exception and the additions stated.

1. **Hash input.** `sha256(f"{seed}|{anchorId}|{itemId}").hexdigest()`, ascending; `seed = "ccg-noise-v1|" + gateId + "|" + snapshotId`. Accepted. Ids are
   hex/identifier strings with no `|`; the record states that so a verifier need not argue separator collisions.
2. **`itemId = "<unitId>:<subjectId>"`** (symbolId, entryId or unitId by subject type). Accepted: it makes two subjects with the same id under different
   units distinct. State it in the record, because it changes the hash order against the bare id.
3. **Spot seed** `seed + "|spot"` over `anchorId|itemId` of the labelled keys, 20 keys. Accepted; subject to m2's reporting (the keys are drawn before any
   label value is read, so the draw cannot depend on the labels).
4. **Allocation ties and capped share.** Remainder ties by `(reason, band)` sorted order; capped stratum's unspent share to the leftover rule (largest
   unsampled stratum), not the next remainder. Accepted as the gate review verified it against EP text on 20,000 random strata sets; pin it. Note the
   quota is proportional to `N_h` over **all** strata on top of the minima, as EP says; state "not net of the minima" so a verifier does not re-derive it.
5. **Whole-stratum rule.** `N_h <= 5` whole; frame at or under the target (100) sampled whole. Accepted.
6. **Added by this review (not in the block, but the sample depends on them):** the frame is unique on `(anchorId, itemId)` (m1); strata keys are
   `(reason, band)` with `reason` always a string for a possible item (a `None` reason breaks the sorted stratum order: assert, counts only); items'
   `distance` is the system's reported distance, as EP §11.2 records.

## Over-lister rules, one by one (direction of bias if wrong)

| Rule | Assessment | Bias if left as written |
|---|---|---|
| 1 callers incl. name rows of resolved sites | literal to "every name-match candidate of every call hop"; accept, count the share | upward on baseline burden, naming required (M2) |
| 2 seeds | contract: providers and consumer owners; config: via-site owners | feeds M1 |
| 3 bound applies to call hops only, seeds cost 0 | **defect for consumers (system: distance 1)** | upward on baseline burden, favours the product (M1) |
| 4 kind-wide unbounded | the pre-registered reading; key-scoped reported beside it | upward on baseline burden (F2) |
| 5 symbols only, anchor removed for a symbol anchor | no config or unit items | downward on burden, conservative; config gold never hit (A3) |
| 6 own-status split | reporting only, never changes `count`; the chain definition (call rows only, bound steps) must use the same corrected distance as rule 3 | none on the headline |

## Evaluation design for the tooling (what proves it works)

- The note's §4.1 table plus: contract anchor at bound 2 under the M1 rule (a fixed expected set, derived by hand from the stated rule, not from
  the implementation); name rows on resolved sites counted once; `kind_scope="key"` variant (the note's own mutant: kind-wide restricted to key removes
  `P2`, `Q`).
- **Dominance check on dev** (M1): baseline symbols ⊇ system symbols within the bound for each dev anchor; report the violation count. This is design-time,
  on dev data only; the choice between variants is S1's and is made before the variants' reductions are seen.
- Dev dry run, sizing only: kind-wide and key-scoped `Σ count / Σ system burden` on the dev anchors, as a **measurement the record's author does not tune
  on** (no rule change after it except a defect, which is reported as "after adaptation" if it follows a held-out run).
- Acceptance for the arithmetic gate (analyst): the exact figures of A1-A4 reproduce from synthetic fixtures; `score` exits 4 on each of: differing
  `itemSetSha256`, a baseline anchor set unequal to the answers', `countInScope > count`, an unresolved row without the flag, a bound mismatch.

## Proposed REVIEW.md lines

None. The lesson (a baseline's distance unit and recall direction are part of the claim) is specific to this note, and the existing "arithmetic of a
published statistic" bullet in the specialist prompt covers it.

## Open questions

1. **S1** (stakeholder, before the next gate's pre-registration; not blocking gate 1): which reduction variant carries the ≥ 5x bar: (a) kind-wide, (b) key-scoped,
   (c) both. Recommended: (c), decided before the variants' numbers are seen.
2. graph-dba or architect: confirm the system's distance for **providers** and for **config anchors' consumers** at distance 0/1 against
   `docs/plans/connected-code-graph-graph.md` §3.3; the dominance check on dev will expose a mismatch either way.

## Pass 2

**Scope.** Revision `04b6e9c7` of the note (§4-§7). Verified by reading `cpg/ccg/engine/bfs.py` (`ROWS`) and by executing:
`tests/test_engine_bfs_pins.py` (8 passed), `tests/test_engine_bfs_rows.py::test_a_config_change_reaches_only_the_sites_it_binds...` (passed),
and two scratch walks (session scratchpad, nothing under `cpg/` edited) on synthetic graphs through `tests/test_engine_bfs.run`.

**Verdict: approve with suggestions.** The pass-1 findings are folded in. §6 can be pinned after the one-line edits of N2.

### Dispositions of pass 1

- F1 / A1-A4: fixed (§6: harness list, `itemSetSha256`, symbol-only pair, paired 2x2, two-sided statement; record text matches mine). One tightening, N2.
- F2: fixed (`kind_scope`, key variant, anchor-type reporting; S1 stays open, correctly). F3: fixed (§7 item 1 changes `score` and `bootstrap`).
- F4: fixed (bound from the record; `score` refuses a mismatch). F5: fixed (§7, `oracle.py`: status split, `linked-no-provider`).
- M1: **fixed, and my config claim withdrawn** (below). M2, m1, m2, m3: fixed as named (§4 `byName`, §6 named choices, §7 items 4-5, one row per answer, `unit` type refused).
- Draw pins: fixed (§7 list matches my six items).

### Ask 1: config-anchor consumers cost 2 — **confirmed; graph-dba is right**

`bound_in` (`ConfigEntry` -> `Link` in state `site`, increment 1), `via_in:site` (`Link` -> `CallSite`, increment 1), `owner:CallSite` (-> `Symbol`,
increment 0) sum to 2. Executed: the engine's own test asserts the listed consumers `C1`, `C3` at `{i.distance} == {2}` and passed. My pass-1
"I could not confirm" was an unverified guess about config; it is withdrawn. Contract-anchor consumers cost 1, confirmed by executing a contract
walk on a graph with provider `A`, `P3`, consumer `H` and a caller `R` of `H`: the system listed exactly `{H: 1}`.

Two facts the note should state (not defects): (i) in contract mode the engine lists **no provider** and no caller of the consumer (the consumer
arrives in state `end`), so "a provider is at distance 0, as in the system's answer" is wrong in wording: provider seeds are the baseline's own
choice from the pre-registered "every provider". (ii) In a flag-true unit the symbol-anchor walk lists direct callers only (executed: anchor
`A`, `B` calls `A`, `C` calls `B`: `{B: 1}`); the baseline's second call step is therefore deeper than anything the system does in such a unit,
by design of a naive baseline. Neither favours the product; both mean the reduction measures flow-sensitivity plus kind filtering, and the report
should say so. Rule 3's arithmetic (config seed at cost 2 with `bound` 2 gives no further step; at bound 1 config consumers are absent, as in the
engine, which does not follow an increment-1 row from the bound) is consistent with the engine. I re-derived the §4.1 rows `A`/kind (9), `A`/key (7),
`CP1`/kind (11), `CP1`/key (9), `CE1`/kind (5) and `byName` for `A` and `K` from the rules: all agree.

### Ask 2: interactions between the fixes

No interaction defect in the cost change, key variant, F3 pairing or the dominance check against each other. Findings:

**N1 (minor) "or whose id is absent from the snapshot" returns an unresolved row silently.** A2 was about answers the **system** left unresolved.
Absence of a resolved answer's anchor from the snapshot means a wrong or stale export; it should raise (as the first revision did), because
otherwise a wrong snapshot gives burden 0 everywhere and the run proceeds. Burden 0 is not favourable to the product, but it is a silent pass.
Fix: unresolved rows come only from the answer's `type: "unresolved"`; an absent id on a resolved answer raises with a count; `score` prints the
unresolved-row count (already stated) and the harness checks it equals the answers' unresolved count. Also the note's "(EP §5.4 key-error
anchors included)" conflates an anchor with an item: §5.4 concerns gold **items**, which are carried by `keyError`; an unresolved **anchor** is
the system's answer status. Reword.

**N2 (minor) The record must pin more than §6, and `itemSetSha256` needs a canonical form.** §6 states the hit rule and figures, but the over-lister's
definition lives in §4 rules 1-6; a hash of §6 alone lets §4 drift. Pin §4-§6 together (and `baseline.py`). The hash input
`anchorId|type:repoId:path:qname-or-keyPath` omits `lines`: two gold items with the same impacted key and different lines would be
indistinguishable, so add the sorted `lines`, and state the sort (Python string order, UTF-8) and a final `\n`-free join. Both are one-line edits and the
only changes needed before the hash is taken.

### Ask 3: dominance exclusion (`->` hop, `DATA`, `CONTROLLED_BY`, `HAS_PARAM`, `HAS_RETURN`, `HAS_COND`)

The exclusion cannot hide a violation **in the product's favour**, because a violation is a system item the baseline lacks, and a smaller baseline
lowers baseline burden and recall, i.e. the reduction is understated (the conservative direction of A4). Two real weaknesses remain:

**N3 (minor) Exclusion is by the system's chosen best chain.** The walk picks the chain by cost (possible hops, distance, length, own-possible),
so an item reachable by a pure reverse-call path within the bound can still carry a best chain with a `DATA` edge; it would be excluded and a baseline
bug on it invisible. Fix: check every item; report two counts, class A (best chain has none of the excluded edges; expected 0) and class B
(excluded), with class B split by the edge type that excluded it. A large class B share makes "checked" small and must be visible.

**N4 (minor) The check is one-sided: it detects a baseline too small, not too big.** The direction that favours the product is baseline
**inflation** (more listed items), and a bloated baseline passes dominance trivially. Because contract-mode never lists providers or callers, and
symbol-mode in a flag-true unit lists direct callers only (Ask 1), a large part of the baseline's burden is items the system never lists by
construction. Add to the report, counts only, the baseline items by source: `bySeed` = items reachable only from provider seeds, only from consumer
seeds or config seeds, only by name rows (`byName` exists), and only as kind-wide members, per anchor type. No threshold; it makes the
composition of the reduction readable and tells S1 what each variant buys.

### Is §6 stable enough to pin?

Yes after N2: §6 has no open choice, the figures and the flag are exact, and the dependencies it cites (§4 rule 4, S1 in §7) are named, not
undecided. S1 does not block the pin: both variants are emitted and neither carries a bar at gate 1.
