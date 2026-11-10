# Review: P1-45 answer validator and schema edits (C-1001 P1)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed the uncommitted diff of `cpg/ccg/answer.py`, `cpg/ccg/schemas/answer-v1.schema.json` and `cpg/tests/test_answer.py` against `docs/plans/connected-code-graph-answer-spec2.md` (§3.5, §5, §7, §8) and the P1-45 section of `docs/plans/connected-code-graph-p1-answers.md`. Root `REVIEW.md` applied (no `cpg/REVIEW.md` exists). Not reviewed: engine code (P1-21/P1-23), which does not exist yet; reachability is judged from spec2's rows.

**Verdict: approve with suggestions.** No blocker. The validator clauses match spec2; the three clauses the delegate called implied are implied (each verified by execution). Two minor test gaps found by mutation.

CPG: considered, not relevant — a self-contained validator module and its tests, read directly; no cross-module call-graph question.

Checks run: `cpg/` as cwd, `.venv/bin/python -m pytest tests/test_answer.py` gives 632 passed. Mutants ran on a copy under the session scratchpad, never in place. Full `cpg` default suite (`-m "not live"`, cwd `cpg/`): 4306 passed, 1 skipped (slow leaf-mutation test, opt-in), 10 deselected (live).

## Findings

### F1 (minor, verified) — "Unit entered from a `Link`" has no test; mutant survives

`_check_unit_hop` (`answer.py:~625`) rejects a `Unit` hop whose predecessor is not a `Link`. Mutant: drop `or chain[i - 1]["kind"] != "Link"`. All 632 tests stay green. Executed on the mutant: `[SA, site, Unit]` (Link removed from `b_unit`) is ACCEPTED; on the real code it is rejected (`V5 ... entered from a Link hop by TO_UNIT ->`). Spec2 §3.5 and §8 case 8 name the clause only partly (edge `TO`, `unit != id`, direction flipped); the Link-predecessor half is unpinned.
Fix: add to `test_to_unit_items` a case that deletes the `Link` hop of `b_unit()` and expects `V5`. Run the mutant above to confirm it goes red.

### F2 (minor, verified) — `Bound.launch.{mountPath, envVar}` screening and the field-kind argument are unpinned

`_check_f4` screens `b["launch"][f]` for `mountPath`, `service`, `envVar` (`answer.py:~784`). The only test is `bound(d, 0)["launch"].update(service=UNMASKED)` (`test_answer.py:2499`). Two mutants survive: passing `"keyPath"` instead of `f`, and passing `"service"` instead of `f`. Both keep 632 green. The field kind selects different screen units (`mountPath` splits on `/`, `keyPath` tokenizes, `service` screens the whole text; `_screen_units`, `answer.py:355`), so a wrong kind is a real defect that the suite does not see. The hop-level `launch` has per-field cases (`test_answer.py:970-978`); the `Bound` copy has none.
Fix: parametrize the Bound case over `mountPath` (`/data/<UNMASKED>`), `service`, `envVar` as at lines 970-978, with a value that is accepted under its own kind but rejected under another (e.g. a `mountPath` of several clean segments).

### F3 (nit, verified) — equivalent mutants noted so the owner does not chase them

Dropping `packId` or `port` from `_BOUND_1_1_FIELDS` survives, but the schema already ties `packId` to `portOrigin: pack-default` and `port` to `portOrigin`, so one of the remaining names always co-occurs. Returning a junk site id for a wiring Link also survives, because a real `lifted.siteId` never equals it. No action.

## Answers to the brief's questions

**V5, V9, V12, V17, V18, V19 against spec2 wording.** Read each against §8's rule table (rows V5, V9, V12, V17-V19) and §3.5/§5.
- V5: wiring (`_check_link`, impact), depends-on entry (`VIA ->` from a `CallSite`) and exit (`TO ->`/`TO_UNIT ->`), `lifted` site, `Unit` last / unit==id / entered by `TO_UNIT` / `TO_UNIT` only at a `Unit`. Direction is read from `request.direction` (`answer.py:920`, `862`), as spec2 requires; the mutant fixing it to "impact" is killed.
- V9: `not-mapped` check present; mutant killed (`test_to_unit_items`).
- V12: `keyPath` only when present, plus the three launch fields (but see F2).
- V17: both directions and the `>= 2` bound for resolved 1.1 documents; the bound mutants (`< 1`, status guard dropped) are killed.
- V18 (a)-(e) match §5's list; mutants for (a), (a kind), (b), (c both halves), (d `prev` vs own unit), (e) and the two constants all killed. Expected values in `test_v18_constants_are_keyed_by_direction` are literals, not read from the constants.
- V19: four-name set; the mutant dropping one is killed.

**Reachability.** I traced §5's two table rows and the `B-flow`/`B-dflow` bases; I found no document a faithful engine would build that the validator rejects. The checks key on `(edgeType, dir)` rows exactly as §5 lists them, apply to 1.1 only, and `(c)` excludes `CALLS_NAME` as §5 says. Residual risk is only that P1-21/P1-23 have not run against it (spec2 §9 R-2); the two shipped halves meet first in P1-23's tests.

**The three removed clauses — all implied, verified by execution.**
- V5 wiring "no `lifted`": `_check_link` returns `None` for a wiring Link, so any `lifted` entry fails the per-site test. Executed: `B-wire` with a real site id planted in `lifted` is rejected (`V5 ... siteId is not the link's own call site's id`).
- Unit-hop `dir`: in depends-on the Link exit check allows only `TO ->`/`TO_UNIT ->`; in impact a Link may only precede a `CallSite` or a wiring `ConfigEntry`. Executed: `b_unit()` with the Unit hop's `dir` set to `<-` is rejected at the Link (`V5 ... left by TO -> or TO_UNIT ->`). The message names the Link hop rather than the Unit hop; acceptable.
- V17 "Bound without `keyPath`": the schema `oneOf` makes such a `Bound` carry `launch`, and `launch` is in `_BOUND_1_1_FIELDS`. Executed: `b_chain()` relabelled 1.0 (first `bound` entry launch-only) is rejected `V17`; mutant removing `"launch"` from the tuple fails `test_v17_a_1_0_bound_carries_no_1_1_field[launch-value0]`. The implication holds, and leans on `launch` staying in that tuple, which a test pins.

**`bound_of_link` lowest-`(siteId, ord)` mutant — out of scope here, agreed.** The answer document carries no `siteId` on a `Bound` (`bound[]` is per site by the read, spec2 §3.2), so the validator cannot see which edge supplied the entry. The rule is the engine's (P1-21's reads, `p1-answers.md` P1-21 behaviour list names it, and spec2 §8 gives P1-45 no such mutant). It must reappear in P1-21's gate.

**Test quality / own mutation.** About 30 mutants run, including data-plumbing arguments: `prev.get("unit")` to `h.get("unit")` (killed by `test_v18_labelled_rows`), the depends-on site returned as `nxt["id"]` instead of `prev["id"]` (killed), `_item_key` slot `0` to `1` (killed), `failedAt` not dropped from `facts_bytes` (killed), and the `Bound.launch` field-kind argument (survived: F2). Note that with `-x` several V5 mutants die first at the unrelated `test_every_required_field_is_required[UnitSubject.type]` because fixture instances fail to build; I re-ran the depends-on site mutant without `-x` and it is also killed by `test_to_unit_items`.

## What's solid

- Clause structure (`_check_link`, `_check_unit_hop`, `_check_flow_rows`) maps one-to-one onto spec2's rule rows and is small; the 1.1-only gating of V18/V19 and the 1.0 hygiene of V17 are right.
- The schema keeps unknown-key rejection (`cacheBypass` const true with `cacheHit: false`; `Bound` exactly one of `keyPath`/`launch`, port evidence coupling; `TO_UNIT` and `UnitSubject` added without removing enum entries).
- The existing golden is untouched (no golden in the diff), consistent with P1-45 item (5).
- Tests pin §5's literal tables and kill the plan's named mutants (V5 entry/exit/lifted, V9, V12 `KeyError`, V17, V18 b/d/e/direction, V19, `facts_bytes`).

## Open questions

- Reachability against a real engine is untestable until P1-21/P1-23; the caller may want P1-23's gate to re-run V18 over its walk output as a standing acceptance.
