# Review — Typed Decision Models Plan Successor 5

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Pass 1 (2026-11-01)

**Scope.** `docs/plans/typed-decision-models5.md` v1.3 (1,805 words), checked against these sources:
- plan4 v1.2 at `2d6af936`, swept whole;
- the note v1.12 §Q1.1–§Q1.3, diffed against v1.9 at `2d6af936`;
- `docs/reviews/typed-decision-models-impl.md` P20-m2;
- `docs/reviews/typed-decision-models4-impl.md` P3-F1, P3-F6, P3-F7;
- `docs/reviews/typed-decision-models.md` P11-M2 and Pass 12;
- the working-tree U5.3 code (`decisionmodels/{points,seeds}.py`, `points/*/point.json`);
- `model-bench/modelbench/runner.py`, `scoring/retrieval.py`, `docs/BACKLOG.md`.

I ran the decision-models suite read-only (`PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`): 330 passed, 2 xfailed, 1 failed (F1). My probes are in `scratchpad/p5/`. Nothing outside this file was written.

CPG: considered, not relevant — no `cpg_modelbench`/`cpg_decisionmodels` graph is loaded; the cited call paths were read directly.

### Verdict

**Needs changes.** Two majors, both small text fixes to plan5:
- **F1:** U5.2b's position (P4-F5 too) makes every later decision-models gate's green suite unreachable, and this is already red in the tree.
- **F2:** the rewritten `init --phase full` gate does not require a `proceed` outcome.

A line check by `teco` on F1–F2 can close the pass; the minors can ride with them.

### Findings

**F1 — major — U5.2b sits at the end of the decision-models chain, but U4 reddens decision-models' suite as soon as it lands.** *Owner: `architect` (plan5 §4 "Order", U5.2b row). Verified by running it.*
- `tests/test_decision_contract.py:45-48` asserts the byte-copy of `input-hash.jsonl` equal to model-bench's.
- U4.1 (in flight) appends 6 join-in/message-type rows to the canonical file (`git diff --stat`: +6).
- The suite now fails `test_the_vector_copy_is_byte_identical_to_model_benchs[input-hash.jsonl]`.
- Plan4 §4.3 orders U5.2b after U5.10, and plan5 keeps it there. So the green suite in U5.4–U5.10's done-conditions cannot be reached while U4 has landed and U5.2b has not.
- The ledger (2026-11-01, "Cross-unit red") plans to "pull U5.2b's vector-copy part forward", but no plan row commissions that.

A bare re-copy does not fix it: `hashing.INPUT_FIELDS_BY_ROLE` has no join-in or message-type row, so the copied rows raise `ValueError` (P4-F5, same finding from the code side).

**Fix:** split U5.2b as `docs/reviews/typed-decision-models4-impl.md` P4-F5 proposes. I checked that the split fits the chains and plan5's U5.2b row:
- **U5.2b-hash** goes into the decision-models chain as the next unit after U4.1 integrates. It covers the `hashing.py` rows, the `input-hash.jsonl` re-copy, the as-sets `inputFields` conformance, and retiring the P3-F6 xfails together with `PLANNED_INPUT_FIELDS`. It needs only U5.3's `points/`. The chain is serial on `HISTORY.md`, so inserting it there conflicts with nothing.
- **U5.2b-render** covers the renderer copies, the rubric "What an item looks like" fixture, and the join-in 6-turn obligation. It keeps plan4 §4.3's edges: after the U4 unit that lands the renderers, and before `U5c.message-type`.
- Plan5 §4 must say this itself, naming U4.1, which plan4 does not define. A ledger note is not a build instruction.
- Until U5.2b-hash lands, gates deselect the one test by node id, and the ledger records that (P4-F5).

**F2 — major — the guard `full` gate admits a `stop` decision.** *Owner: `architect` (§3.6 "Gates"). Verified by reading plan4 §3.5/§3.8 against plan5 §3.6.*
- Plan4 §3.5: `full` "refuses unless the guard pilot's report carries a recorded stakeholder decision". `record-decision` accepts `--outcome proceed|stop`, and no text checks the outcome.
- Plan5 rewrites the gate as "refuses unless the state is `passed` or `recheck-passed`, or `stakeholderDecision` carries that ref". Taken literally, that is a disjunction with no decision clause at all.
- Either reading lets `init --phase full` start guard's full spend after a recorded `stop`. This is the irreversible, costly case the gate exists for.

**Fix:** state the whole predicate in one place, for example: `stakeholderDecision.outcome == "proceed"` and (state ∈ {`passed`, `recheck-passed`} or (state = `recheck-failed` and the stakeholder ref is present)). Then add a U5.6 fixture: a `stop` decision with `passed` is refused.

**F3 — minor — the recheck's result has no defined writer, and the report keeps one result.** *Owner: `architect` (§3.6 "States", "On `failed`"). Inferred from reading.*
- `pilot.py recheck` "draws … issues … and reruns the same test". But the block's votes arrive over driver sessions, so `recheck` cannot evaluate them synchronously.
- `missingCueCheck` holds "`cue_check`'s output only", while the note §Q1.3 says "Both results join the decision package."

**Fix:**
- Name the command that evaluates the block once its votes are ingested and writes the state (presumably `pilot.py report`).
- Give the field both outputs, for example `{state, pilot: CueCheck, recheck: CueCheck|null}`.
- Add a U5.8 assertion that a `recheck-failed` report still carries the pilot's result.

**F4 — minor — four new checks have no red-able test.** *Owner: `architect` (§4 rows). Verified by reading the rows; the boundary case was computed exactly.*
- (a) **U5.8 cue check:** no row pins `≤`. One stratum with `m1 = 1`, `m0 = 19`, `x + y = 1` and `x = 1` gives exactly p = 1/20 (`scratchpad/p5/cue.py`, `Fraction`). It must fail, and a `<` mutant reddens it.
- (b) **U2.4:** "Other exceptions pass through" has no control. P20-n1 is the precedent: add a `RuntimeError` from `prime` that propagates, so a mutant widened to `except Exception` reddens.
- (c) **U5.6:** `fresh_rng_seed`'s redraw is never exercised. Monkeypatch `secrets.randbits` to return a held value, then a fresh one.
- (d) **U5.8:** nothing tests `recheckProjection` or the report's RNG-seed list. Add a hand-computed projection, and require the seed list to include the recheck block's seeds.

**F5 — minor — the 5% halt waits for the whole round.** *Owner: `architect` (§3.5 "Halt timing"). Inferred from the arithmetic.*

This changes plan4 §3.6 openly, not silently: plan4 and note §Q1.2 set no timing, and the ledger records "round end acceptable". But hits only grow, and the round's seed count is fixed at the draw. So a check after each first-pass run, against that fixed count, reaches the same decision as soon as it is certain. The end-of-round rule spends every first-pass opus run of a defective template first:
- message type: ⌈1.1 × 600⌉ = 660 seeds, 27 runs;
- join-in: 18 runs.

The guard pilot shields only guard.

**Fix:** evaluate after each first-pass ingest, with the denominator stated as drawn seeds. Also say whether the pilot and the recheck block are "rounds" for this halt.

**F6 — minor — §3.5 has no per-item reason precedence.** *Owner: `architect`; the binding change is `coder`'s. Verified by reading P4-F1's run and `points.py:229-245`.*

A schema-valid `turns` item checked against an `understanding` seed gives three `evidenceKind` errors and a `missingForm` error. Under §3.5 it counts as both `bindingMismatch` and `formMismatch`, which inflates `dropped.formMismatch` and that form's 5% halt with an item that never attempted the form. The coder's fix makes the binding silent when the field is "not a non-empty mapping", which covers `understanding: {}`. That removes this case, but it is not a rule. An item that both fails a genuine `equalsAxis` and breaks its form, or that hits the screen and fails a binding, still double-counts.

**Fix:** give each item exactly one reason per pass, in a stated total order. I agree with P4-F1 that a non-form binding error beats `formMismatch`. Put the screen hit in the order too: I suggest screen hit, then `bindingMismatch`, then `formMismatch`. The item takes one hit toward its 3-hit cap and counts toward one halt. Add a U5.8 fixture of the four-error item that asserts `bindingMismatch` only. A both-counted mutant reddens it.

**F7 — nit — screen semantics and citations.** *Owner: `architect`. Verified by running it.*
- "Any `seeds.ID_PATTERN` token is a hit anywhere" leaves the matching open. `re.search` finds `dido_123` inside "pedido_123", because the pattern is unanchored (`seeds.py:69`). Say token-bounded or substring.
- Plan5 §3.4's replacement string "note v1.9 §Q1.2 'Hits'" does not occur in plan4 §5 U5.8, which reads "(note v1.9 §Q1.2)". This is harmless: plan5's U5.8 row carries the new fixtures, and the note's other v1.9-cited passages ("Reruns", "Spend halt", pilot "Screen", "On a hit") are unchanged in v1.12 (diffed).

**F8 — nit — growth-stops.** *Owner: `architect`.* Plan5 is a new document, so all of its growth is new scope. It has one dated revision line and no finding→resolution table, and it does not restate plan4. Two small restatements remain:
- §3.6's "Output" and "Composition" bullets repeat the note's rule and the block's makeup. Keep the interface (input, `CueCheck` fields, states) and cite §Q1.3 for the rest.
- §2's "no longer in the note" is history.

### Check answers

1. **Amendments against their sources.**
   - **P20-m2:** closed. `runner.py:468-476` calls `prime` with no `try`; `retrieval.py:341` embeds on a miss; `_warm_up` (`:1806-1813`) maps both classes to exit 3. R-21's corrected sentence and the BACKLOG pointer (`BACKLOG.md:193`) are both in U2.4.
   - **P3-F1:** closed. Both refusals are there, and "do not fold `count`" is honoured.
   - **P3-F6:** closed. Both strict xfails exist in the tree.
   - **P3-F7:** closed. `rubric_body`, `label_answer_section` and `generator_instructions` exist (`points.py:50-82`).
   - **P11-M2 (a)–(d):** routed as Pass 12 records. (d) gives way to §3.6's gates (F2).
   - **§3.6 `cue_check`:** exactly the note's rule — strata (difficulty × answer), classes, convolution, one-sided, p ≤ 0.05. All four strata hold both classes (`point.json` conditional).
2. **Collisions with plan4.** My sweep of plan4 (Hits/v1.9, pilot outputs, `record-decision`, `init --phase full`, `validate_request`, §3.7, R-21, store deletion) found:
   - §3.2's seed line, §3.8's store deletion, §3.7's hash parts and R-21: each explicitly amended;
   - line 249/360: F2;
   - line 498: F7;
   - U5.2b's position: F1;
   - every other hit: no conflict.
3. **Unit rows.**
   - Each added obligation has a row; the test gaps are in F4.
   - U2.4 after U3e.2 is coherent with model-bench's serial `HISTORY.md` chain, and it gates nothing (U7's drivers never call `prime`).
4. **U5.3 as built.** It matches plan5. `seed_binding_errors` returns `BindingError(binding, message)`, and `.axis` reads `emptinessAxis` (`points.py:169-199`). The `emptinessAxis: missingForm` binding has `emptyFor` {empty, indirect, unlisted, ambiguous} and `nonEmptyFor` {immaterial, listed}, which is the note's table. A `turns` seed (`None`) is unbound.
5. **Halt timing:** F5. Per-item reason precedence (P4-F1): F6.
6. **Growth-stops:** F8.

### What's solid

- Ledger-owned seed identity: `seeds.py` stays pure, and both clashes are refused.
- The schema-before-binding split matches plan4 §3.3's out-of-space row.
- The xfails were placed so that they can fail.
- Hashing the semantic parts keeps format fixes out of U5.9's tranche trigger.

### Open questions

- F2 touches a stakeholder spend gate. Should a `recheck-failed` that the stakeholder accepts also need `outcome = proceed`? I assume yes.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: F1.
- When a plan's serial unit refreshes a byte-copied contract file, check that no unit in another chain changes the canonical file before that refresh lands. Otherwise the green-suite done-condition of every unit in between is unreachable.

## Pass 2 (2026-11-01)

**Scope.** Plan5 v1.4 (2,042 words) against Pass 1, P4-F1 and P4-F5. I also checked:
- the ledger's U4.1–U4.3 rows;
- v1.3 §4 U4 (`docs/plans/typed-decision-models.md:814-822`) and §8;
- model-bench's working tree, for where the renderers live;
- the `ID_PATTERN` uses in `decision-models/`.

My probes were read-only and nothing outside this file was written.

**Verdict: approve with suggestions.** Every Pass 1 finding is fixed. Checking the fixes against each other found one interaction gap (P2-F1), plus two minors and two nits. None of them blocks U5.2b-hash or U5.4.

CPG: considered, not relevant — no `cpg_modelbench`/`cpg_decisionmodels` graph is loaded; the cited sites were read directly.

### Pass 1 dispositions

- **F1: fixed.** §3.4 splits U5.2b, names the node-id deselection, and puts both halves right after U4.1 (§4 "Order"); `U5c.message-type` still waits on U5.2b-render. Whether the renderers are all in U4.1 is P2-F2.
- **F2: fixed.** §3.6 states one predicate that requires `outcome == "proceed"`, and `stop` always refuses. Every state now has an outcome:
  - no report, `failed` or `recheck-pending`: refused by both `init --phase full` and `record-decision --outcome proceed`;
  - `passed` or `recheck-passed`: admitted;
  - `recheck-failed`: admitted only with the ref.

  The U5.6 row has one fixture per branch, including `stop`+`passed`. The open question is answered: yes, `proceed` is needed.
- **F3: fixed.** `report` is the only evaluator, and `recheck-pending` is a new state. The report holds both `CueCheck`s, and `report` refuses a partly voted block. U5.8 asserts that a `recheck-failed` report keeps the pilot's result.
- **F4: fixed, all four.**
  - The p = 1/20 boundary case: `x = 1`, `y = 0`, `x + y = 1`, the same as my exact computation in Pass 1.
  - The `RuntimeError` passthrough, with an `except Exception` mutant.
  - `secrets.randbits` patched to a held value, then a fresh one.
  - `recheckProjection` and the recheck block's seeds in the report's list.
- **F5: fixed.** The halt runs per run, when `100 · first-pass hits > 5 · drawn` for one reason, and exactly 5% does not halt. Pilot blocks and the recheck block each count as a round with their own draw.
  - A regeneration's hit counts toward the cap only, so it can never reach a halt numerator. That matches the fixed denominator.
  - A recheck block's first pass is its own round of 100, separate from the pilot's 100. Neither can borrow the other's hits.
- **F6: fixed in rule; see P2-F1 for how it interacts with F5.** Precedence is screen hit, then `bindingMismatch`, then `formMismatch`, with one hit toward the cap. The four-error item and "screen beats binding" both have fixtures.
- **F7: fixed, with a residue (P2-n1).** "pedido_123" is no hit, and that case is in the U5.8 row.
- **F8: fixed.** §3.6's restated rule and composition are gone, as is §2's "no longer" aside.

### New findings

**P2-F1 — minor — precedence also decides which halt counts the item, so a defective template can hide behind another reason.** *Owner: `architect` (§3.5). Inferred from the rule text, by arithmetic.*

§3.5 says "One hit toward the seed's 3-hit cap **and toward that reason's halt**". A form failure on an item that also hits the screen, or that also fails a non-form binding, never reaches `formMismatch`'s numerator.

Example: 4% of items both hit the screen and break their form, and another 4% only break their form. That is 8% form failures, but neither halt crosses 5%. The halts exist to detect a defective template per reason, and precedence was introduced for cap and drop accounting (P4-F1's inflation of `dropped.formMismatch`). The one case P4-F1 meant to exclude, the wrong-kind item, already gets no form error once the coder's emptiness fix lands (`points.py` `_emptiness_errors` on an empty mapping).

**Fix:** keep precedence for the cap and for `dropped`. For the halts, count each item once under every reason it shows, excluding the form error of a wrong-kind item. Add a U5.8 fixture: items with a screen hit plus a form failure, with forms over 5% and screen hits under, must halt on `formMismatch`.

**P2-F2 — minor — "U4.2 and U4.3 touch no renderer" is true of the functions, not of their module.** *Owner: `architect` (§3.4 U5.2b-render "Its dependency"). Verified by reading.*
- In the tree, U4.1 created `modelbench/scoring/agreement.py`, which holds `render_join_in_input`, `render_message_type_input` and both hand-off parses.
- The ledger's U4.2 row is "`scoring/agreement.py` (predicate/choice, fail-closed, …)", and v1.3 §4 U4 lists the renderers under that same file.
- So U4.2 edits the renderers' module.

The split is still coherent. What decision-models pins is the vector files (byte-equality), and those are all U4.1's (`render-*.jsonl`, `normalize-turns.jsonl`, untracked in the tree). If U4.2 or U4.3 changed any of them, the guarded equality would redden decision-models again: the Pass 1 F1 seam would come back, this time detected.

**Fix:** add one clause to §3.4: a later U4.x change to any copied vector file re-copies it within that unit's integration, with the decision-models re-run in its gate. U4.1's own open `priorTurns: []`-versus-absent ruling (ledger, 2026-11-01) can still change `input-hash.jsonl` before integration, which is why "once U4.1 integrates" is the right trigger.

**P2-F3 — nit — the decision record's lifecycle is unstated.** *Owner: `architect` (§3.6).* "`stop` is always recordable", but nothing says whether a later `proceed` may overwrite it, for example a `stop` at `failed` followed by a recheck that passes. Nothing says either whether a `stop` at `failed` is "a final state" for deleting the store. State single-write or last-wins, and name the final states.

**P2-n1 — nit — the screen's boundary class is ASCII.** *Owner: `architect`. Verified by running it.*
- With `(?<![A-Za-z0-9_])`, "opções_12" is a hit on `es_12`, because `õ` counts as a boundary. With `(?<!\w)…(?!\w)`, which is Unicode in Python 3, it is not a hit.
- Pt-BR prose rarely has underscores, so the cost is a stray regeneration. Prefer `\w`.
- Separately, §3.4's "its own uses are `fullmatch`" is inexact: `tests/test_points.py:394` uses `ID_PATTERN.search` on templates. That use is stricter and fine there, so only the wording needs to change.

### Growth-stops

1,805 → 2,042 words (+237).
- **New scope** (P4-F1 and P4-F5 entered after Pass 1): the U4 context bullet, the U5.2b split with its two rows and Order, and the precedence rule.
- **Rework** (F2–F5), in place:
  - the predicate replaces the three gate bullets;
  - the per-run halt replaces the round-end rule;
  - `recheck-pending` with the numbered `On failed` steps replaces the loose recheck sentence;
  - the F4 obligations are test text, the minimum a red-able check needs.

The cuts offset it: §3.6's Output and Composition restatements, §2's history aside, and U5.2b's old bullet. The two "Why" lines state design reasons, and findings are cited by ID only. There is one dated revision line, no resolution table, and no plan4 restatement. **No finding.**

### Proposed REVIEW.md lines

None new. P2-F1's class, "a fix for one finding changes another fix's input", is already covered by checking interacting fixes against each other.
