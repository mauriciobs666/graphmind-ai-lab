# P1-39c flow-fallback result — reading and decision (`ccg-p1-flow-fallback-v1`)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-39c, answers plan P1-39 (4)) · **Extends:** `docs/plans/connected-code-graph-p1-ml.md` (§3.5, §6.6)

Sources: record `ccg-p1-flow-fallback-v1` (+ `-tool`, `-snapshot`) in `cpg/eval/preregistration.jsonl`; result `cpg/eval/results/ccg-p1-flow-fallback-v1.json` (every figure below was read from it, except where marked "computed here"); dev items `cpg/eval/dev/ac5-flow.jsonl` (read only to see which items are unlisted). Nothing was re-run.

## 0. Validity: not void

The record's voiding conditions are the refusal chain (pin shas, `-tool`/`-snapshot` lines in the committed blob, throwaway port, graph digest between phases), the twice-run byte-equality, and the V-union ⊇ set-union assertion. Phase B exited 0 and emitted the result, so all of them passed (the tool refuses or voids otherwise). The result carries `pinnedFilesVerified` 91, `extraPinsVerified` 23, `phase` "B".
Two disclosures, neither voiding. (1) "Quiet host" has no numeric bound in the record and the walks are deterministic (no sampling, no timing); the result file records the **phase B** load average (before 2.08 / 4.35 / 4.09, after 2.26 / 4.17 / 4.04); the phase A figure (1.02) is in the brief, not in the result, and I did not verify it. (2) `pinsVerifiedOnTree` names HEAD `a8b496de`, the result's `environment.head` is `59bc1022`; the record calls the former descriptive only.

## 1. What the rule says and what the data show

Rule (record `rule`; `-p1-ml.md` §3.5): ship the lowest dev burden among the maximum-`s` walks; tie → lower per-anchor maximum → fixed order as-built, V-both, V-flow, V-union.

| walk | listed / 28 | burden sum | burden max |
|---|---|---|---|
| as-built | 8 | 281 | 276 |
| V-flow | 8 | 333 | 328 |
| V-both | 9 | 417 | 405 |
| V-union | 9 | 417 | 405 |

- **Max `s` = 9/28 (0.321), reached by V-both and V-union; the rule picks V-both** (`rule.pick`, `maxListed` 9). Applied correctly: V-both and V-union tie on sum (417) and on max (405), so the pre-fixed order decides.
- **The tie is total, not marginal.** V-both and V-union have identical `listedIds`, `unlistedIds`, per-anchor burden and cluster cuts. V-flow lists exactly the as-built set (same 8 ids) at +52 burden: **V-flow's rows add nothing on dev**, so V-union's extra rows over V-both buy nothing either. The order picked V-both; the evidence would pick it too (53 vs 66 table rows per the record, fewer unmeasured Ret/arg(i) rows, `retLimit`).
- **`safeClaim: false`: no walk reaches 28/28.** By the record, "safe" is licensed only at n/n; H1 is false; 19 of 28 dev items are not listed by the picked walk; FR-8 is **not** claimed met for the residual (`noWalkAtNN`).
- **The pick is a screen, not an improvement.** V-both beats as-built by one item (`c7de2461…`, a JS callee-arg chain) at +136 burden (417 vs 281, +48%; 129 of it on the `resolve_llm` anchor). The paired difference is 1 discordant item against 0: exact McNemar p = 1.0 (computed here). The data do not show V-both better than the as-built walk; the rule, fixed before the data, selects it because 9 > 8. `secondLevelCalleeOnly` is empty: depth 2 did not earn its burden.
- **Denominator/cluster.** The 28 items are 4 anchor clusters (listed/n: `resolve_llm` 6/8, `loadChannels` 2/7, `advanceOrder` 1/10, `renderReadinessBadge` 0/3); 0 of 4 clusters fully listed (cluster Wilson upper bound 0.490). Two thirds of the 9 listed items come from one cluster. The pooled Wilson interval 0.179–0.507 treats items as independent and is therefore too narrow; read it as an optimistic bound. Burden is 97% one anchor (405 of 417), so the burden tie-break is effectively a one-anchor comparison. Language/signature cuts have no interval and cannot support a per-language claim; "language" is by impacted file, not anchor (by file: python 6/9, javascript 3/19).
- **In-sample.** V-flow and V-both were designed from the dev hop kinds (record `safeClaim`); the result is not validation of either.
- **Signature does not predict listing.** Inside `advanceOrder`, items with the identical single return-use chain (`endpoints.ts:157` to `hooks.ts:27`) are mostly unlisted while one is listed, so the unlisted items are an upstream resolution/edge question as much as a flow-hop one. This is an observation from the dev file and the result ids, not a tested explanation.
- **The F4 prediction failed** (`asBuiltComparison`): pooled 8 measured vs 10 predicted, javascript 2 vs 4, python 6/9 equal. Not voiding (the record uses the comparison as a report); it means the hop-kind model of the as-built walk overstated TS/JS recall by 2 items.

## 2. P1-39 decision

1. **Ship V-both as the flow-fallback default table** (routine, rule-determined). V-flow and V-union do not ship; the as-built table is not shipped (it would list 8 for 136 less burden, but the pre-registered rule, not a post-hoc preference, decides; a departure would be an unregistered choice and is not recommended on one-item evidence either way).
2. **Allowed wording** (docs, spec, report): "On the 4-anchor, 28-item development flow set, the V-both flag-false walk lists 9 of 28 items (pooled Wilson 0.18–0.51, optimistic because items cluster in 4 anchors; 0 of 4 anchor clusters fully listed); in-sample screen." and "FR-8 is not met for flow-dependent impact in flag-false languages; the fallback is a partial-recall walk."
   **Forbidden:** "safe", "complete", "meets/closes FR-8", "V-both lists more than as-built" (1 item, not significant), any claim for V-flow or V-union value or for depth 2, any generalisation to held-out or other repos, any per-language or per-signature rate.
3. **AC-25's control half is not closed** by this pick (`-p1-ml.md` §3.5 Consequence: V-both ships and does not list the guarded point in the general case; only a stakeholder waiver would close it). `-p1-gates.md` P1-42 already has the "reported not closed" branch.
4. **Follow-up measurement.** (a) No further dev measurement: re-walking the same 28 items is in-sample and cannot discriminate candidates. (b) The validation is the held-out `flow` stratum at the AC-5 gate (P1-37) for the shipped V-both table, outside this record; the question it answers is listed share on held-out, with no n/n expectation. (c) Recommended, exploratory only and labelled so: a read-only look by `graph-dba` at why the unlisted pure return-use and callee-arg chains in `advanceOrder`/`loadChannels` are not reached (missing or unresolved call edges versus a walk gap). If it is the edge, no fallback walk fixes it and flag-true would not either.
5. **Nothing in this table decision waits for flag-true units.** V-both contains no `Ret`/`arg(i)` row, so the context-insensitive Ret limit (`retLimit`) is not shipped. What does wait for flag-true units: any claim about flag-true precision, and the P1-34 python flag decision, which is independent (python flips only on a TRUE; JavaScript flow is not measured in P1, so JS runs the fallback throughout P1).

## 3. Stakeholder question (scope, not method)

The result means 19 of 28 dev flow items (16 impacted in JS/TS files) are silently absent from a flag-false answer, which is what FR-8 forbids. The rule made this "reported not closed", not a ship blocker. **Question:** accept V-both as the P1 fallback with FR-8 and AC-25's control half recorded as not met for flag-false flow, or require a mitigation? **Recommendation:** accept, and add an answer-level notice that flow-dependent impacts in flag-false languages are not enumerated (so the omission is stated rather than silent); do not wait on a better walk, since none measured here improves on dev.

## 4. Edits owed (path + symbol)

- `docs/plans/connected-code-graph-answer-spec2.md` §5 (owner: the spec owner, **before V-both ships**, per its own rule and `-p1-answers.md` P1-23): the closed labelled-row table ("The closed list of labelled rows") gains V-both's added callee rows; V18 (e) and the "Consequences in P1" bullets (callers/callees possible; "AC-25 is as index §5 states") are revised to match; §1.2 "Granularity": `ALGORITHM_VERSIONS` takes a new fingerprint literal, either folded into `1.1` if `1.1` is not yet published on `main`, otherwise `1.2` (the bump rule; check publication state at the time).
- `docs/plans/connected-code-graph-p1-answers.md`, P1-23 row, behaviour (1): replace "P1-39's rule picks the shipped table" by V-both, with the algorithm-version consequence; keep V-flow/V-union as non-shipped measured tables or remove them, in one stated choice. P1-39 row (4): note the measurement is done, the pick and `safeClaim` false, point to this note; its done-condition no longer waits on a pick.
- `cpg/ccg/engine/flow.py`, `candidate_table` and the shipped-table selector (owner `tdd-engineer`): default to `V-both`; one test per table stays. Caution: the live tool `cpg/eval/superset_dev.py` / `cpg/tests/test_engine_superset_dev.py` refuse when any pinned engine file differs, so any edit to `flow.py`/`bfs.py` makes the phase B live test refuse at HEAD; mark it as run-at-the-pinned-engine history (the committed result is the record) or give a re-run its own `v2` record.
- `docs/plans/connected-code-graph-p1-ml.md` §3.5 "Record" line and the header `Extended by:`: cite this note and the result path; no change to the rule text (it stays as registered).
- `docs/plans/connected-code-graph-p1-gates.md`, P1-42 (AC-25 clause): choose the "not closed" branch explicitly; the guarded-point item is not asserted `possible`.
- `docs/requirements/connected-code-graph.md`, FR-8 / AC-25 status (owner `tico`): record the stakeholder answer to §3.
- Report text for the AC-5 gate and P1-37's `flow` stratum: state the V-both pick and use §2.2's wording only.
- `docs/HISTORY.md` (root, this component): the P1-39c entry cites this note and the result, written last.

## 5. Risks

One dev draw is spent; a second walk comparison on these 28 items is exploratory by the record. The tie order was fixed in advance, so no post-hoc choice entered; the weakness is that 28 items and 4 clusters cannot rank walks, and the held-out stratum is the first real evidence.
