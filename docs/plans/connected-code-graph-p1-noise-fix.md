# Connected code graph P1: dev noise fix after the CO-9 rehearsal at R2 (plan)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (P1 close-out, CO-9)

CPG: considered, not relevant — a Python walk read directly plus an offline replay of the saved rehearsal answers and labels; no `cpg_*` graph was needed.

Change against: the answer engine's walk (`cpg/ccg/engine/bfs.py`, `cpg/ccg/engine/config.py`), contract `docs/plans/connected-code-graph-answer-spec2.md` §1.2 (version = fingerprint of the walk definition) and §4 (depth bound, `unexplored`). Relies on: `docs/test-reports/connected-code-graph-p1-gate-rehearsal-co9-t2-report.md` (the t2 figures), `docs/plans/connected-code-graph-p1-closeout-prereg.md` §3 item 3, §4 successors, §5 (labelling plan: unchanged pairs reuse labels at t3), §6 (claim), `docs/plans/connected-code-graph-p1-closeout-ml.md` (hit rules; no cap on the answer), `docs/plans/connected-code-graph-p1-ml.md` §5.3 (bars).

## 1. Goal and scope

Bring the labelled dev Q (pooled and scripts, point estimate) under the preregistered bar of 4 at the next rehearsal (t3, record R3) with the smallest rule that a reviewer can defend without any dev item. **Dev gate only:** held-out noise is not measured at P1, so the fix is a dev fit and says nothing about held-out noise. Out of scope: any recall change, any resolver/extractor/pack change, any rubric or label change, the held-out spend.

## 2. Findings (labelled sample of t2, counts; first labeller)

Method: `cpg/eval/tools/noise.py` (`estimate`, `bootstrap`) called from scratch scripts outside the repo on the saved `labelling/{pooled,scripts}` files. It reproduces the report: pooled Q 4.779, scripts 4.101, bands 3.244 / 7.698 (pooled) and 1.758 / 7.698 (scripts).

**False items by reason and band** (pooled draw; N frame, n labelled, f labelled false, estimated false W̃ and real R̃ by the stratified estimator). Totals W̃ 596.2, R̃ 94.8, RC 30.

| Reason | Band | N | n | f | W̃ | R̃ |
|---|---|---|---|---|---|---|
| prompt-mention | 2 | 198 | 23 | 23 | 198.0 | 0 |
| shell-dynamic | 2 | 172 | 20 | 15 | 129.0 | 43.0 |
| name-match | 2 | 4 | 4 | 4 | 4.0 | 0 |
| prompt-mention | 1 | 208 | 24 | 22 | 190.7 | 17.3 |
| unknown-key | 1 | 72 | 11 | 11 | 72.0 | 0 |
| partial-key | 1 | 18 | 7 | 1 | 2.6 | 15.4 |
| shell-dynamic, name-match | 1 | 19 | 11 | 0 | 0 | 19.0 |

Reading by distance, stratum and chain shape:
- **Band 2 is 55% of the false mass pooled (331 of 596) and 74% of scripts (331 of 447).** Every band-2 item is a scripts item whose first hop is a `possible` link into a flag-false unit (`prompt-mention`, `shell-dynamic` or `name-match`) followed by a **labelled fallback hop** (`flow-unavailable:<lang>`): a **callee run** (`HAS_CALL ->` then `CALLS ->`/`CALLS_NAME ->`, the V-both rows of `bfs.V_BOTH_ROWS`; 39 of 47 sampled band-2 items) or a **caller of the caller** (`calls_in:fn`, 8 of 47). The 23 prompt-mention callee runs are Markdown siblings: a document that mentions the anchor also mentions another script, which then lists as impacted. 0 of 31 sampled band-2 items whose chain holds a `prompt-mention` hop are real (Wilson upper 0.11). The shell callee run is the only band-2 source with real items (5 of 20 sampled, none through a `resolution-unverified` hop).
- **Band 1** (all hits at distance 1 in http and mcp; scripts D1 as well): `prompt-mention` documents 190.7 (mcp 78, scripts 113), `unknown-key` call sites 72 (http 52, mcp 20), `partial-key` 2.6. These are the remaining floor: pooled band-1 Q is 3.244 whatever happens at band 2.
- **Labels versus gold (observation, not a fix input).** 108 of 108 hit items are listed; 65 of the 208 band-1 `prompt-mention` frame items are gold-matched (real by the AC-5 gold). Of the 10 sampled gold-matched `prompt-mention` items the labeller marked 8 false ("prose, no runtime dependency"), against 14 of 14 false among the non-gold ones. Treating every gold-matched item as known-real (a sensitivity estimator, not preregistered, never the gate figure) gives baseline pooled Q 3.24 and scripts Q 3.75, both under 4. The spot-check (19 of 20, 20 of 20) shows the labeller applies EP §11.4 consistently; the tension is between EP §11.4 and the gold, see §8 question 2.

## 3. Options (offline replay over the saved answers and labels)

Replay = drop every frame item whose best chain meets the rule, re-estimate with `noise.estimate` over the kept frame (post-stratified inside each reason x band stratum), bootstrap over answers (B = 10000, seed label `ac5-gate-1-dev-t2`; the baseline interval is within 0.12 of the report's; the A + D interval is B = 2000, option D has none). Limits: (1) the labelled sample is the t2 draw, t3 redraws it, so the t3 point estimate will move (answer-resample share above 4 is the margin figure below); (2) a rule inside the walk could promote a non-best arrival of an item whose best chain it cuts (arrivals kept are non-dominated, best has the fewest possible hops, so rare); the t3 rehearsal is the measurement; (3) labels are two LLM roles, one labelled the sample.

| Option (general rule) | Items dropped pooled / scripts | False removed (est) | Real labelled lost | Q pooled [95%] | Q scripts [95%] | Possible precision pooled / scripts | Resamples with Q > 4 pooled / scripts | Dev hits lost |
|---|---|---|---|---|---|---|---|---|
| none (t2) | 0 | 0 | 0 | 4.779 [2.73, 8.89] | 4.101 [2.28, 9.62] | 0.137 / 0.172 | 0.74 / 0.57 | 0 |
| **A. Mention bound 1:** a `prompt-mention` hop is never followed to an arrival past distance 1 | 230 / 230 | 232 | 0 of 31 sampled (est -2, an estimator artefact) | **2.873** [1.49, 5.50] | **1.985** [0.90, 4.35] | 0.210 / 0.301 | 0.16 / 0.04 | 4 (all scripts, §4) |
| B. No flow fallback in a Markdown unit (prose has no flow) | 198 / 198 | 198 | 0 | 3.192 [1.75, 5.85] | 2.284 [1.22, 4.91] | 0.192 / 0.272 | 0.23 / 0.07 | 3 (not `c649a7ac`) |
| C. Fallback hops are not stacked on a chain that already holds a `possible` hop | 374 / 374 | 331 | est 43 (5 of 20 sampled shell items) | 3.244 [1.73, 5.58] | 1.758 [0.60, 3.79] | 0.163 / 0.301 | 0.23 / 0.02 | 4 |
| D. No callee run through a name-only edge (`CALLS_NAME`) | 85 / 85 | 69 | est 16 | 4.86 | 4.37 | 0.130 / 0.158 | n/a | 0 |
| A + D | 315 / 315 | 298 | est 17 | 2.760 [1.44, 5.02] | 1.813 [0.67, 3.95] | 0.207 / 0.310 | n/a | 4 |

Rejected, with the reason:
- **D alone** does not reach 4; **A + D** adds 0.11 of pooled margin for an estimated 17 lost real labelled items and a second rule: not the smallest.
- **Dropping `unknown-key` (72 false, 0 of 11 real)** or **band-1 `prompt-mention`**: the first is FR-8's own case (a runtime-built URL is listed as `possible`, AC-6), the second carries recall: all 36 mcp and 28 of 54 scripts hits are reached only through a distance-1 `prompt-mention` chain, so a cut loses hits. Both are labelled-class cuts a reviewer cannot defend as general.
- **B** in the walk is not smaller than A (it needs a language gate on the labelled rows) and covers less; in the registry (`Unit.flowAvailable: true` for Markdown, `cpg/ccg/registry/flow-availability.yaml`) it contradicts the file's rule that a row flips only by a measured flow recall.
- **C** is the widest: it removes the V-both fallback wherever the first hop is `possible`, loses an estimated 43 real labelled items, and touches the flow-fallback behaviour the 2026-11-19 decision accepted (9 of 28 dev flow items); the flow dev anchors were not asked in this run, so its flow cost cannot be replayed.
- **Raising the bar, changing the rubric or the estimator** is not an engine option: §8 question 2.

## 4. Design: option A, a mention bound (recommended)

**Rule.** A hop whose edge reason is in the mention class (`config.MENTION_REASONS = {"prompt-mention"}`) is followed only while its arrival distance is at most `config.MENTION_BOUND = 1`; the effective limit is `min(bound, MENTION_BOUND)`. The rule is by edge reason alone, in either column.

**Why it is defensible without dev items.** A `prompt-mention` edge says a name occurs in prose: it asserts no call, data or control dependence, and the extractor never marks it `confirmed` (`cpg/extractors/markdown/extract.py`). It is a sound first hop (a document that names the anchor may need updating) but a poor premise for a second one: from a mentioning document the walk would then list everything else the document mentions (callee run) or every other document mentioning it (caller of caller), each a doubt stacked on a doubt. The rule applies to any repository's prose, and the offline sample shows the effect it was meant to have (0 of 31 real).

**Where it lands.** `bfs.walk` already cuts an increment-1 row from an arrival that would pass the bound and records the stored edge on that arrival (`Entry.cuts`, `Walk.cut`, `Walk.unexplored`). The new cut is the same branch with a per-hop limit, so a cut edge is counted in `depthBound.unexplored` and nothing is dropped silently (FR-8, requirements decision 2026-10-28); no answer schema field changes. `render/common.py` words it as "reach stopped at distance 2"; for a mention cut the stop is at 1, a wording gap left for the live-spec writer.

**Contract effects (all stated once here).**
- **`algorithm.version` moves 1.2 to 1.3**: the definition gains `mentionBound` (and the fingerprint with it), pinned as a literal in `ALGORITHM_VERSIONS`. `definition()` emits `mentionBound` only when it is not `None`, and `walk()` takes `mention_bound` defaulting to `config.MENTION_BOUND` for the shipped table (`table is None`) and to `None` for an explicit table, so `definition(bfs.ROWS)` still maps to 1.1 and `definition(bfs.SHIPPED_ROWS)` to 1.2. `answerSchemaVersion` stays "1.2".
- **Walk changes; rows do not.** No row, edge, gate or state is added. Items that survive keep their chains, so AC-18 (full chain, file and line at each hop) and the H3 chain-replay rule hold unchanged; the closed list of item rules is untouched.
- **Spec2 §4** (depth bound and `unexplored`) gains one sentence at the feature's close; `-answer-spec2.md` §1.2's list of what the fingerprint cannot see is unchanged.

**Dev hit cost (replay over the 108 hits).** Only the four coincident scripts hits, all distance 2 through a `prompt-mention` hop, are lost; the other 50 scripts hits and all 18 http and 36 mcp hits are distance-1 chains and stay. Each, with its class in the committed dev gold and predicate:

| Item | Anchor / impacted | Listed today only by | Class |
|---|---|---|---|
| `7fc6a23c` | `bootstrap_schema.sh` / `test_queries.sh` | AGENTS.md sibling mentions (D2) | **none**: no shape letter, not a key-error; the gold names line 103, the launch is `EMBEDDING_DIM=4 ./scripts/bootstrap_schema.sh` at line 114, a working-directory-relative path the resolver cannot place (CO-2c removed such edges on purpose) |
| `c649a7ac` | `export-cpg.sh` / `joern-cpg/SKILL.md` | `shell-dynamic` then a mention (D2) | withheld shape B (flagged) |
| `d105f096` | `git-provenance.sh` / `test-stamp-wiring.sh` | SKILL.md sibling mentions (D2) | withheld shape I (flagged) |
| `f7a50995` | `pipeline.sh` / `test-stamp-wiring.sh` | SKILL.md sibling mentions (D2) | confirmed dev key-error (item 47) |

After A the scripts dev count reads 64 = 50 hits + 10 withheld-not-listed (8 + `c649a7ac`, `d105f096`) + 3 key-errors not listed + **1 item in no class (`7fc6a23c`)**. **Entry-criterion item 1 therefore does not hold at R3 until the stakeholder rules on `7fc6a23c`** (§8 question 1). Concordant hits become 50 of 50, so the record's claim rule no longer blocks "the scripts resolver recovers cross-script links" on dev (the held-out claim rule is evaluated at the run).

**Held-out cost, to be weighed.** The rule turns a held-out item that today lists only through a prose sibling (a coincident hit) into a miss. Dev saw 4 coincident hits in 54; 3 sat in flagged shapes or a key-error, 1 in none (1 in 47 unflagged non-key-error scripts items). A one-observation Poisson reading over the roughly 50 unflagged held-out scripts items gives about one such item and a chance of at least one near 0.6; the interval is wide and clusters make it wider. The held-out verdict is listing-only, so this is a real cost against a dev-only gain.

## 5. Freeze effects

Core change (`cpg/ccg/engine/bfs.py`, `config.py` move `coreTreeHash`): new F, new `coreTreeHash`, new `algorithm.version` 1.3 in R3; `packSetHash` is unchanged (no pack file is touched) and `defaultDepthBound` stays 2; the harness pin (`G.HARNESS_FILES`) is unchanged, so every harness sha stands (prereg §1). All `cpg/tests/**` edits land before F, since `run_context` refuses a test edit after R. Per prereg §4 successors: R3 is a new first-gate record (`ac5-gate-1-r3`, `kind: gate`), rehearsal count at R3 starts at 1 with tree ordinal t3, R2 and the t2 rehearsal become superseded, the no-workaround scan is rerun over the new tree, and the freeze-check at R3 needs its own permission request (the current permission names R2 only). A changed key-error record or predicate would change their pinned hashes (§8 question 1).

Review conclusions that stay valid: **CO-2** and **CO-3** (`docs/reviews/connected-code-graph-p1-closeout-co2.md`, `-co3.md`) judged the script resolver, the Markdown extractor and the shell packs, none of which this change touches; **CO-6c** (the EP/ML scope reconciliation, documents only) is untouched, since no stratum, floor or bound moves. Not carried over: the R2 record's analyst and `data-scientist` passes (a new record), and a fresh `analyst` review of the walk diff with mutation checks is needed.

## 6. Steps (3 steps, 5 files)

| # | Step | Files | Done when |
|---|---|---|---|
| 1 | **Red tests first** for the rule (§7 list 1-6), then the rule: `config.MENTION_BOUND`, `config.MENTION_REASONS`; in `bfs.walk` compute the per-move limit `min(bound, mention_bound)` for a move whose `mv.edge.reason` is in `MENTION_REASONS` and cut it through the existing `cuts` branch (any increment, so an increment-0 mention hop arriving past 1 is cut too); `walk(..., mention_bound=...)` with the default of §4; `Walk.mention_bound`; `definition(table, bound=, mention_bound=)` emitting `mentionBound` when not `None`; `Walk.version()` passes it; docstring of `bfs.py` and `UNEXPLORED_RULE` unchanged (the identifier string stays; `mentionBound` carries the fingerprint) | `cpg/ccg/engine/config.py`, `cpg/ccg/engine/bfs.py`, `cpg/tests/test_engine_bfs_guards.py` | rule tests green; `bfs.definition()` has `mentionBound` 1 |
| 2 | **Pin 1.3**: compute the new fingerprint, add `"1.3"` to `ALGORITHM_VERSIONS`, keep 1.1 and 1.2; update `PINNED_*` and the literal-"1.2" assertions that read the shipped default (`test_engine_bfs_guards.py` lines near 327 and 365, `test_engine_flow.py` near 429 and 620-623, `test_engine_answer.py` near 102, 1129, 1255); explicit-table assertions (`table=bfs.ROWS` is 1.1, `bfs.definition(bfs.SHIPPED_ROWS)` is 1.2) stay | `cpg/ccg/engine/bfs.py`, `cpg/tests/test_engine_bfs_guards.py`, `cpg/tests/test_engine_flow.py`, `cpg/tests/test_engine_answer.py` | `cd cpg && python -m pytest tests -k "engine"` green (rerun the full `cpg` suite; the live and fixture suites that need the store, and `tests/slice/test_engine_equivalence.py` at bound 60, must be unchanged or each moved row owned) |
| 3 | **Measure by the t3 rehearsal** (an offline replay of the new walk is not possible: the store is down and the rule changes the walk). Commit F, then R3 and the rehearsal per prereg §4-§5 (same labeller roles, fresh t3 draw, unchanged `(item, answer)` pairs reuse their t2 labels). Expected, from §3 and not a gate: pooled Q near 2.9, scripts near 2.0, scripts hits 50 | none (process) | R3 committed; rehearsal report states run count 1 at R3, the tuning disclosure of §9 and the 4 lost items |

## 7. Test strategy (red to green, in `cpg/tests/test_engine_bfs_guards.py`, synthetic graphs from `engine_walk_helpers.G`)

1. A `prompt-mention` hop from the anchor arriving at distance 1 is followed: the owner is an item at distance 1, chain unchanged.
2. A `prompt-mention` hop that would arrive at distance 2 is not followed: the item is absent, `Walk.unexplored` counts the stored edge (a mutant that drops the cut record fails).
3. The same shape with a `name-match` hop is followed (the rule is by reason, not by position).
4. A confirmed first hop then a `prompt-mention` second hop: cut (earlier hops do not exempt it).
5. An increment-0 `prompt-mention` hop (the contract `ON` link) arriving at base distance 2 is cut; at base 0 and 1 followed.
6. `bound=1` and `bound=60` requests: the limit is `min(bound, MENTION_BOUND)`, and a graph with no mention hop is bit-identical to the 1.2 walk.
7. Version pins: shipped default is 1.3 with `definition()["mentionBound"] == 1`; `table=bfs.ROWS` stays 1.1; `definition(bfs.SHIPPED_ROWS)` stays 1.2; changing `MENTION_BOUND` changes the fingerprint.
8. A depends-on walk with a mention hop at distance 2 obeys the same cut (the rule is column-free).
Mutation-check the limit comparison and the reason membership (the CO-2 and CO-3 builders' standard).

## 8. Stakeholder questions (not decided here)

1. **The dev item in no class, `7fc6a23c`.** After A, criterion item 1 needs a ruling. Options: (a) confirm it a dev key-error (a fourth entry in `ccg-dev-key-errors-v1.json`, whose sha the record pins); (b) add a withheld-shape letter (an edit of the pinned predicate); (c) extend the resolver to retry working-directory-relative paths against ancestor directories (reverses CO-2c, adds `possible` edges, cannot be measured offline); (d) leave the criterion unmet. Recommendation: (a), the same family as the confirmed items G and 47; held-out analogues stay scored as misses (the key-error precedent in the prereg §3).
2. **Does a prose document that names an anchor count as real for the noise estimate?** The gold counts such items as impacts and the rubric (EP §11.4) labels them false; §2 shows the effect (baseline pooled Q 3.24 on the gold-known-real reading). Recommendation: keep the labels and the rubric for the gate (changing the measurement after seeing the result is the tuning this gate forbids), report the sensitivity beside, and decide the rubric question for P2's measured gate.
3. **Is the held-out listing cost of A acceptable?** (§4, a coincident-only held-out item becomes a miss.) Without A the dev gate cannot pass and no held-out spend happens; with it the spend carries that risk. Recommendation: accept, since the dev gate is the stakeholder-set entry condition; the alternative is question 2 read the other way.
4. **A mention cut counted in `unexplored`** (no schema change, the render wording says "distance 2"): acceptable, or does it want its own field (a spec2 and schema change)? Recommendation: acceptable.

## 9. What the claim and the rehearsal report must disclose

- A rule chosen **after reading the t2 labelled result** (0 of 31 real in the cut class): the t3 Q is a refit statistic, rehearsal count at R3 is 1 but the tuning iteration count is t1, t2 and this design; the 4 lost dev items and their classes; the held-out coincident-hit cost of §4.
- **Dev fit, not held-out noise.** "Noise not measured" stands; no held-out Q, precision, p90 or reduction exists. The possible-precision, p90 and S1 bars stay reported-not-gated (prereg §3 item 3).
- Labels come from two LLM roles (first labeller `qa-engineer`, spot-check `analyst`, 19 of 20 and 20 of 20): not independent validation. The label-versus-gold tension (§2) and the sensitivity figure are reported beside the labelled Q, never as the gate figure.
- Risks left: pooled band-1 is a floor near 3.2 (documents and `unknown-key`), 16% of answer resamples still exceed 4 pooled, so a t3 miss is a live outcome; a second miss is a stop-and-ask, not a second rule found on the same labels. The flow-fallback dev result (9 of 28) belongs to the earlier tree; chains of code flow items carry no Markdown mention at distance 2, so no change is expected, but this run did not ask the flow anchors.

## Ready to implement

Document: `docs/plans/connected-code-graph-p1-noise-fix.md`. Implement only after the stakeholder answers §8 question 1 (and ideally 3). Recommended: option A, a mention bound of 1 in `bfs.walk` (new `algorithm.version` 1.3, 5 files, reported through `unexplored`); replay estimate pooled Q 2.87 and scripts 1.99, 4 dev hits lost of which 3 are already withheld or key-error and one (`7fc6a23c`) needs a ruling.
