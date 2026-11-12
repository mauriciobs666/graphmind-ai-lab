# Review: P1-23a the `site` hop state (schema enum, V5/V17, tests) (uncommitted) — `cpg/ccg/{answer.py,schemas/answer-v1.schema.json}`, `cpg/tests/test_answer.py`

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-23a)

## Scope & verdict

Reviewed `git diff` of `cpg/ccg/answer.py` (`_entered_from_config_entry`, `_check_link`, `_check_hops`, V17 in `_check_labels`, the `validate` call site), `cpg/ccg/schemas/answer-v1.schema.json` (`Hop.state`) and `cpg/tests/test_answer.py` (ENUMS, `b_site()`, the `site` section, the changed `b_config_launch_only()`), against `docs/plans/connected-code-graph-answer-spec2.md` §3.6, §6, §7.1, §7.2 V5/V17, §8, `docs/plans/connected-code-graph-p1-answers.md` P1-23 behaviour (0), and the origin `docs/reviews/connected-code-graph-p1-22.md` Pass 2 N1. Root `REVIEW.md` applied (no `cpg/REVIEW.md`). Not reviewed: the engine (another agent is editing `cpg/ccg/engine/`), P1-39's builder (not yet written).

Run (cwd `cpg/`, `ulimit -v 3000000`, `timeout -s KILL`): `tests/test_answer.py` 646 passed (includes the 1.0 golden `tests/golden/answer-ac1.json` validate, the schema-contract and ENUMS tests); `tests/test_render.py` 140 passed. Own probe script (scratchpad, 9 planted chains) and 9 mutants in unique scratch copies (ccg, tests, packs, infra, pyproject only): 8 killed, 1 survives (F1).

**Verdict: approve with suggestions** — no blocker; one minor test gap (F1), two spec-text corrections for the architect (F2, F3), one nit.

CPG: considered, not relevant — a validator/schema/test change read in full; no loaded `cpg_*` graph indexes this uncommitted code.

## Answers to the four questions

**(1) Version gating.** Correct, and it is what spec2 means. §3.6 labels the shape "(V5, 1.1)" and says "A 1.0 document has none (V17)". The 1.0 fixture `b_config()` (`test_answer.py:219-231`) has its `Link` in `contract` entered by `BOUND_BY <-` from a `ConfigEntry` — the shape the 1.1 clause forbids — so an ungated clause would reject a 1.0 document the v1 schema allows. Verified: M1 (drop `v11`) and M9 (`v11` always true) fail `test_dumps_bytes_are_identical_when_the_array_is_shuffled[Hop.bound]` (b_config on 1.0); `validate(b_config())` in `test_v17_site_is_rejected_on_a_1_0_document` pins it. The half "a `site` Link is entered by `BOUND_BY <-` from a `ConfigEntry`" is version-free but unreachable on 1.0 anyway (V17 fires first). 1.0 compat is preserved: the schema change is one appended enum value, the 1.0 golden validates, and no 1.0 doc can carry `site` (V17).

Proposed §7.2 V5 row, `site` sentence, to replace "**`site` state:** a `Link` hop has it exactly when entered by `BOUND_BY <-` from a `ConfigEntry` hop, and is then followed by its `CallSite` (§3.6)":
> **`site` state (§3.6).** On every document, a hop in state `site` is a `Link` hop entered by `BOUND_BY <-` from a `ConfigEntry` hop, and is followed by its `CallSite` hop (`VIA <-`), never by a wiring `ConfigEntry` hop and never last. On a `"1.1"` document the converse also holds: a `Link` hop entered by `BOUND_BY <-` from a `ConfigEntry` hop is in state `site`. A `"1.0"` document keeps the v1 state of that `Link` (it has none in `site`: V17).

Same two-version split belongs in §3.6's "Shape (V5, 1.1)" bullet (which says "exactly when" without the 1.0 carve-out) and the V17 row already lists `site`.

**(2) The changed fixture.** Not masking a defect: `b_config_launch_only()` is `to_11(b_config())`, i.e. a 1.1 config-anchor chain, and spec2 §3.6 now requires `site` for exactly that shape; leaving it `contract` would assert an invalid 1.1 document. Sweep: `BOUND_BY` and `ConfigEntry` → `Link` chains in repo outside the engine: `b_config`/`to_11(b_config)` (1.0 and the one changed 1.1 copy), `b_site` (new), `b_launch` (ConfigEntry→ConfigEntry, no `Link`), `b_chain`/`b_wire` (the `Link` is entered by `TO <-` from the golden, not a config entry; `b_wire`'s `Link` is followed by a wiring entry in `value`/`contract`, which the clause does not touch), `b_dep_11`/`b_unit` (depends-on, `VIA ->`). `tests/test_render.py` builds on `b_chain`/`b_wire` (140 pass); `tests/golden/answer-ac1.json` has no `ConfigEntry`; no `cpg/fixtures`, docs example or other test imports `ccg.answer` (grep). No silently unvalidated chain found. One staleness: spec2 §8 line 481 still describes the `Bound.launch` instance as "a 1.1 copy of `b_config()` whose first `Bound` holds `launch` only" — it now also needs the `Link` set to `site` (F3). Engine walk output is not yet validated anywhere (P1-39), so a `ConfigEntry`→`Link` real-walk check is P1-39's.

**(3) Rejections.** Probed (scratchpad `probe.py`, all against `validate`): `site` non-`Link` (rejected V5, test + M5 killed); `site` Link not entered by `BOUND_BY <-` (wrong edge, wrong dir: rejected, M2/M6 killed); 1.1 Link entered so in `contract|value|key|end|function` (rejected, parametrised test); `site` Link followed by wiring entry, or last (rejected; M3 killed); `site` on 1.0 (V17 `chain[1]`, M4 killed); `site` Link followed by a `CallSite` whose edge is `TO` or `VIA ->` (rejected by the pre-existing generic clause); `lifted[].siteId` naming another site (rejected, the pre-existing equality); `site` in depends-on (rejected: the depends-on `VIA ->` entry clause contradicts `BOUND_BY <-`, so it can never be accepted); `accessPath` on a `site` hop (V3). Accepted correctly: a mid-chain `ConfigEntry` (reached by `READS`) entering a `Link` in `site` (matches engine row `bound_in`, `bfs.py:139`, which fires from any `ConfigEntry` in `value`, not only the anchor) — and in `contract` it is rejected. V13: `site` is in the visited key, so the same `Link` once as `site`, once as `value` is not a repeat, twice as `value` is (test pins both). V18 reads only `function` and is untouched. Gaps: F1; `site` Link with no `lifted` is accepted — as for every `Link` before this change, and §3.6 says the siteId lives in the `CallSite` hop's id, so not a defect.

**(4) Schema.** Additive (one value appended, enum order irrelevant to validation); 1.0 golden validates (part of the 646). M8 (remove `site` from the enum) is killed. The schema `description`s do not enumerate the states.

## Findings

### F1 (minor, verified by surviving mutant) — the "entered from a `ConfigEntry`" predicate's `prev["kind"]` is unpinned
`answer.py:597` `prev["kind"] == "ConfigEntry"`. Mutant M7 (`prev["kind"] in ("ConfigEntry", "Symbol")`) survives all 646 tests: every negative test mutates the *edge* or *dir* or the hop's own state, none varies the previous hop's kind. A `site` Link entered by `BOUND_BY <-` from a `Symbol`/`CallSite` hop would then be accepted. Fix, one test: take `b_site()`, replace `chain[0]` with a `Symbol` anchor hop (or insert one before the `Link`) keeping `chain[1]` `site` + `BOUND_BY <-`, expect V5 (and V4/V3 noise aside, assert the `V5` message). Add the positive companion (a `ConfigEntry` reached by `READS` then a `site` `Link`, as probed above) so the "any ConfigEntry hop, not only the anchor" reading, which matches the engine row, is pinned too.

### F2 (minor, spec text) — spec2 §3.6 "Shape (V5, 1.1)" and the §7.2 V5 row say "exactly when" without the 1.0 carve-out
As written, the converse half contradicts the 1.0 fixture and §3.6's own "1.0 has none". The delegate's gating is right; the text should say it. Replacement wording above (Answer 1).

### F3 (minor, spec text) — spec2 §8 line 481 describes `b_config_launch_only` without the `site` state
Add "with its `Link` set to `site` (§3.6)". Without it a reader re-deriving the fixture from §8 builds a document the validator now rejects.

### N1 (nit) — messages and a redundant branch
`answer.py:609` the `site`-Link-not-followed-by-`CallSite` message says "never by a wiring ConfigEntry hop" but also fires when the `Link` is the last hop or is followed by any non-`CallSite`; say "is followed by its CallSite hop". The clause is also partly shadowed by the generic impact check below it (which, for a `CallSite` next hop, still checks `VIA <-`); harmless, kept because it makes the wiring rejection a named `site` one.

## What's solid
Gating split is minimal (one boolean threaded through `_check_hops`; the entry-shape half stays version-free); V17 fires before V5 on a 1.0 `site` doc, giving the right rule name; the V13 two-state test and the "same chain, Link not `site`" control inside `test_v5_a_site_link_is_not_followed_by_a_wiring_config_entry` are well built (the control proves the rejection is the `site` clause, not shape); mutants M1-M6, M8, M9 are killed.

## Open questions
None for the caller. The architect owns F2/F3 wording; the coverage-probe note for F1 is the `ConfigEntry`-position × entering-edge × entering-dir × previous-kind axes of the predicate, each varied once.
