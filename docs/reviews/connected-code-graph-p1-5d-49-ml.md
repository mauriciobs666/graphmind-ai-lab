# Methodology review: P1-05 5d `self`-rule acceptance draw (30 of 511)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** P1-05 (5d)

**Scope.** The 30-site hand audit of the `self` rule (`docs/plans/connected-code-graph-p1-extraction.md`, 5d "`self` rule acceptance"), drawn into `cpg/tests/data/p1_05_5d_self_audit_draw.json` at pinned commit `8377c3268e7857e7a71a55d5cfdd030b8df5fa8d` (rule delivered in `69789943`). Read: the plan's 5d text, `docs/reviews/connected-code-graph-p1-5d-49.md` Pass 4 (done-condition 3), my demotion-cost note `docs/reviews/connected-code-graph-p1-5d-ml.md`, `_self_rule` in `cpg/extractors/joern/normalize/assemble_bundle.py`, the draw file. Ground truth: every file read with `git show 8377c32…:<path>` (no checkout). Scripts and outputs are in the session scratchpad only. `REVIEW.md`: root file only (no component one for `cpg/`); growth-stops check applied: this is a new document.

## Verdict: needs changes (1 miss in 30; the rule as shipped violates its own design; narrow fix, then re-draw)

The pre-stated bar is 30/30, "any miss fails the rule, which is then narrowed or removed, never accepted at a lower bar". I judged **29 correct, 1 miss** (site 6). The miss is not noise: it is the exact case the design says must stay `possible` ("a subclass override inside the unit"; `docs/reviews/connected-code-graph-p1-5d-ml.md` listed it as "1 is own-but-overridden (stays possible)"), and the shipped rule confirms it.

## 1. Per-site judgment (30 drawn sites, source at the pinned commit)

For each site I read the call line and the enclosing method and class (`ast` over the pinned source), and checked: the call text is `self.<callee>(`; `self` is the method's first parameter; the enclosing class defines the callee exactly once (decorators noted); no `self.<callee> =` or class-attribute assignment shadows it; no class in the repo names the enclosing class as a base and redefines the callee. The confirmed target equals the enclosing class's own method in 30/30 draws.

| # | Site (`path:line`, call) | Verdict |
|---|---|---|
| 0 | `model-bench/modelbench/packs.py:417` `self._resolve_tool_schemas` | correct |
| 1 | `falkor-chat/server/falkorchat/storefront.py:849` `context_for` | correct |
| 2, 3, 21, 25, 26, 28 | `repository.py:810, 3339, 213, 2092, 2950, 1071` `self._graph` (`Repository._graph`, def line 204) | correct x6 |
| 4 | `decision-models/decisionmodels/campaign.py:1401` `self.events` (`Campaign._append`) | correct |
| 5 | `decision-models/tests/finalize_bench.py:131` `self.install` (`Bench.__init__`) | correct |
| **6** | `falkor-chat/server/falkorchat/executor.py:718` `self._execute_step` in `WorkflowExecutor._drive_loop` | **MISS** |
| 7 | `campaign.py:1943` `_outcome_event` | correct |
| 8, 24 | `executor.py:726, 727` `_trace_step`, `_link_emissions` (test subclasses do not override these) | correct x2 |
| 9, 10, 11, 15, 17, 22 | `campaign.py:1713, 1953, 1848, 1824, 1750, 1862` `self.state` (`Campaign.state`, def line 1390, not a property) | correct x6 |
| 12 | `responder.py:95` `_should_respond` | correct |
| 13 | `model-bench/modelbench/lmstudio.py:820` `_raw_post` | correct |
| 14, 18, 23, 27 | `campaign.py:1833, 2053, 1697, 1570` `_resume_topup`, `_move_to_store`, `begin_round`, `_append` | correct x4 |
| 16 | `services.py:2869` `_drive_or_fault` | correct |
| 19 | `executor.py:956` `_read_thread_context` | correct |
| 20 | `storefront.py:1413` `_mark_turn_failed` | correct |
| 29 | `repository.py:3850` `self._order_status_row` (target is a `@staticmethod`; `self.static()` dispatches to it) | correct |

**Site 6, evidence.** `falkor-chat/server/tests/test_executor.py:317-320` (`_BoomExecutor(WorkflowExecutor)` overriding `_execute_step`) and `:555-556` (`_HandoffExecutor`, same). Both are in the same unit as `executor.py` (unit root `falkor-chat/server`) and are driven through `_drive_loop`, so `self._execute_step(...)` at `executor.py:718` dispatches to the override for those instances. The confirmed target (`WorkflowExecutor._execute_step`) is the base only.

**Why the rule let it through (root cause, read from `_self_rule`).** Condition (ii) counts class-method candidates *among Joern's `in` targets*. Joern types `self` to the enclosing class and had already narrowed the call to the one method, so the override never entered the candidate set and (ii) held vacuously. The docstring and the plan assume an in-unit override surfaces as a second candidate; that holds only for the sites Joern left name-linked. The synthetic golden (`tests/test_normalize_name_linked.py`) writes the candidate list by hand, so it cannot expose this. The facts carry no base-class data (`grep inherit|bases|extends` finds nothing in `extract.sc` or `normalize/`), so `assemble` cannot check it.

## 2. Draw reproducibility and bias

Observed this run:
- `populationIds` has 511 distinct ids, equals its own sorted order, equals `population` (511), and the per-unit `selfResolved` counts in the file sum to 511.
- `random.Random(20261119).sample(populationIds, 30)` equals `drawn` exactly, in order; 30 distinct ids.
- All 511 ids are callsite records in the stored bundles at the pinned commit (suffix `8377c32…`). For the 30 drawn: unit, line, callee and enclosing path match the file's fields. Caveat: those stored bundles are **normalizer 0.4** (header `tools.normalizerVersion`), so their `targets` are Joern's pre-5d links; one population id (`2d5121e7…`, `WorkflowExecutor._uid`) carries two targets there, a test module function and the class method, which is the "module function dropped" case the rule exists for.
- The stored `.ccg-work/joern/*/cpg/*.facts.jsonl` for `falkor-chat/server` contain no `rt` key (count 0), so the 0.6 population cannot be re-derived from them. I did not re-run the Joern/Docker extraction; the population list itself is therefore taken from the file, not regenerated. Recommendation: that is acceptable because the id list is stored in the file (the draw is a pure function of it); but "reproducible from the pinned commit + bundle set" holds only for the ids, not for the 0.6 selection, and the report should say so.
- Independent census over all 511 (not a sample): the enclosing symbol's class defines the callee exactly once in 511 of 511; the line text is `self.<callee>(` in 511 of 511; `self.<callee> =` assignment in the class: 0; repo-wide subclass redefining the callee: **1 of 511** (the site 6 id `a118e830…`, line 718, which is in the draw). That makes site 6 a systematic, not random, finding: it is the only such case in the whole population, and the 30-draw happened to include it (expected 30/511 = 5.9% inclusion probability for a single bad site).

Bias check of the frame. The frame is the full population the rule confirmed (all 511), sorted by content-hash id, so there is no selection step beyond the seed. Draw composition: falkor-chat/server 15, decision-models 13, model-bench 2, none from `salesperson`, `skills`, `claude`, `mcp-monitor` (37 sites between them in the file's per-unit counts; chance of none in 30 draws is about 0.10, unremarkable). The 30 contain 20 distinct (class, method) targets over 9 classes; the rule's failure mode is per class/method, so the effective n for that mode is below 30 and the Wilson floor below is mildly optimistic, not conservative.

## 3. What the result supports

- If it had been 30/30: Wilson 95% lower bound **0.886** on the rule's precision over the 511-site population (computed this run: 0.8865). With the observed 29/30: lower bound **0.833** (upper 0.994). The sample cannot support a precision claim near 0.95.
- Required precision: the plan names none for the rule; the only precision threshold that touches a `confirmed` status is the gate's pooled "confirmed precision >= 0.95" (`docs/plans/connected-code-graph-p1-gates.md`, report item 2), over all confirmed edges. The `self` sites are a small slice of that pool (the plan cites 27,804 confirmed Python call edges before demotion, not recomputed here); even at a 0.886 floor the loss is at most 511 x 0.114 = 58 edges, about 0.2% of that cited pool, so the rule cannot by itself breach 0.95 pooled. State that as the reason no per-rule threshold is needed, with the 30/30 bar as a pre-stated tripwire, not a precision claim.
- The census above is stronger than the draw for the structural failure modes (shadowing, redefinition, overrides in the repo): it observed 510/511 clean (Wilson lower 0.989), but it is an `ast` reading, not dispatch semantics (no cross-unit or dynamic subclasses; `setattr` and monkeypatching were not searched).

**The 117 Joern-narrowed sites are not a frame gap.** The acceptance frame is all 511 confirmed sites; the Joern-narrowed ones are inside it. What they lack is the strict reference in `test_normalize_equivalence`/overlay (no other class in the unit defines the name), which is a test-coverage gap for the bulk check, not a sampling gap for this audit. By my definition (another class anywhere in the unit's path prefix defines the name) the figure is 132 of 511, and 9 of the 30 drawn sites fall in it; I did not reproduce 117 (different definition, not run against the implementer's list), so do not cite my 132 as a correction. They matter for method: site 6 is among them. The override defect hides precisely in that group, so the overlay's strict reference, which skips it, would never have caught it.

## 4. The 67% vs 94% trade-off

Acceptable as a design choice, but the stated pin is stale. The plan quotes 375 (67%, unique-name) vs 524 of 557 (94%, class-aware) from my earlier note; the shipped rule confirmed 511 sites (observed), because Joern's own typing pre-narrows many same-name cases, so the shipped behavior is near the class-aware figure, not the 67% one (the 375/524/557 figures are cited, not recomputed). Kept-`possible` recall loss is cheap and honest. Edit the 5d text to state the shipped count and drop "unique-name" as the rule's description. The deferral to P2/SCIP is fine for the cross-unit and attribute-shadow limits; it is not acceptable for the in-unit override (below), which is a P1-detectable case.

## Findings

- **B1 (blocker, acceptance bar).** Site 6: confirmed target is the base method although two in-unit subclasses override it (`test_executor.py:317, 555`). 29/30 fails the pre-stated 30/30. Rule (ii) is vacuous whenever Joern pre-narrowed the call. Suggested fix: carry class base names into the scope table (a Python `class X(C)` list per type decl, e.g. `inheritsFromTypeFullName` in `extract.sc`, or `ast` over the stage tree as the overlay already does) and add condition (iv): no class in the unit that names the enclosing class as a base, directly or transitively, defines the callee; else the site stays `possible/name-match`. This costs one extract field (re-pin the script hash, equivalence overlay) and leaves roughly 510 of 511 confirmed. Alternative if the extract change is judged too heavy: drop the rule to the sites Joern left multi-candidate (rejected: it discards the pre-narrowed majority for one defect). Whichever is chosen, a probe golden with a real Joern-narrowed `self.m()` plus an in-unit override must be added; the existing synthetic golden hand-writes the candidate set and cannot see this.
- **M1 (major, method).** After the fix the acceptance must be **re-drawn** with the pre-stated seed over the *new* population (ids change), 30/30 again; do not reuse the old draw minus site 6 (that selects the sample on the outcome). The one-site population change means most ids are shared, so the new draw is likely close to the old one; report both.
- **M2 (major, plan text).** The 5d text still describes the rule as "unique-name" 67% and says an in-unit override "stays possible"; the second is false for pre-narrowed sites until B1 is fixed. Rewrite in place with the shipped count and the (iv) condition.
- **m1 (minor).** Record in the test report that the population list was taken from the draw file and not regenerated, and that the stored bundles are normalizer 0.4; the 0.6 facts are regenerable only by re-running extraction.
- **m2 (minor).** Report the census (510/511, one repo-wide override) beside the hand audit; it costs nothing and bounds the structural failure modes.
- **m3 (minor, wording).** The Wilson bound is a floor on precision over this population only; it does not cover subclasses outside the unit, which is the stated limit.

Not run: the Docker `slice and live` tier and any Joern/extract regeneration. Not read: held-out data.

## Open question for the caller

Is a test-double subclass override counted as a "miss" under the 30/30 bar? I counted it (the plan's wording: "including subclass ... cases"; the design says it must stay `possible`; the same shape in production code would be a wrong `confirmed`). If the stakeholder rules it out of scope, the draw reads 30/30 and the verdict becomes approve with suggestions (floor 0.886, with the in-unit-override limit then stated as a limit); I recommend fixing it instead: it is one extract field.
