# Typed decision models — methodology review

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** —

## Pass 1 — U5.3 rubrics, templates and seed design (2026-11-01)

*2026-11-01 — the method note moved to v1.10 (`docs/plans/typed-decision-models-ml.md`).*

*2026-11-01 — the method note moved to v1.11 (`docs/plans/typed-decision-models-ml.md`).*

*2026-11-01 — the method note moved to v1.12 (`docs/plans/typed-decision-models-ml.md`).*

**Scope.** `decision-models/points/**`, `decisionmodels/{seeds,points}.py` and their tests (working tree), judged against method note v1.9 §Q1.1–§Q1.5, plan4 §3.2, §3.4, §3.6 and §4.1, and FR-10/FR-12. Suite: 161 passed. Probe scripts are in the session scratchpad (`u53-ds/alloc.py`, `u53-ds/shingle.py`).

**Verdict: needs changes.** The seed arithmetic is correct and so is the top-up restriction. The rubrics follow FR-10 and the note. Two changes must land before generation: M1 before the guard pilot (U5p), and M2 before the join-in campaign. m1 should land before U5.9, and m2 before the message-type campaign.

### Major

**M1 — The guard's `missing` list can carry the intended answer.** `guard-judge/generator.md` tells the generator to keep `missing` "consistent with the intended answer". It defines a near-miss `false` item as one that "lacks exactly one fact". It never says whether that lacking fact appears in `missing`. The rubric then says "an entry in `missing` that the condition needs makes the answer `false`". The likely result is that opus lists the lacking fact for every `false` seed and leaves `missing` empty for every `true` one. The understanding cells, which are half of the guard's FR-13 parity, would then reduce to "is `missing` empty?", and a near-miss would no longer sit near the line. The human side check shows the cue is strong but not perfect. Among its `understanding` items, 31 of 33 `false` items have a non-empty `missing`, and 5 of 18 `true` items list an unneeded fact. So the synthetic set should not become more deterministic than that.
*Fix:* give understanding near-miss seeds a specific recipe:
- A near-miss `false` item leaves the lacking fact out of `known` and **does not list it** in `missing`, or states it ambiguously in `known`.
- A near-miss `true` item lists one entry in `missing` that the condition does not need.
- Clear items keep the natural form.

The pilot should report, per understanding cell, how often the label equals "`missing` is empty". I will fold the recipe into note §Q1.1.

**M2 — Join-in's addressee cue is crossed with intended answers the rubric rules out.** `cue` is a balanced axis drawn independently of `intendedAnswer` (`join-in-judge/point.json`). The rubric's `false` list puts both "an exchange between people" (cue_03) and "small talk" (cue_05) on the `false` side, with no exception. In a probe at round-1 size (RNG seed 12345), cue_03 or cue_05 seeds made up 93 of the 220 `true`-intent seeds, or 42%. Each of those seeds yields either a contrived item or a `false` label. That costs top-up spend for the `true` cell, and it concentrates vote disagreement in that cell. FR-10's within-answer bar then gets harder through label noise rather than through real difficulty. The rubric also has a gap. "Morning Nova, how's your week going?" is addressed to the agent by name, which points to `true`, and it is also small talk, which points to `false`. The defect is in note §Q1.1's taxonomy, and the implementation follows it faithfully.
*Fix:* make `cue` conditional on intent:
- `true` seeds draw from cue_01, cue_02 and cue_04;
- `false` seeds draw from all five cues.

Add a boundary rule. I recommend: small talk addressed to the agent by name that invites an answer is `true`, and a greeting to the room is `false`. I will amend note §Q1.1.

### Minor

**m1 — Largest-remainder rounding biases small top-up draws.** Apportionment is deterministic except on exact ties, so the leftover unit always goes to the heavier value. The probe (`draw_seeds('guard-judge', 1, r, design_stratum='turns')`) shows this:

| Axis | Result |
|---|---|
| `difficulty` | `clear` in 2000 of 2000 RNG seeds |
| `condition` | `cond_00` in 3000 of 3000 RNG seeds |
| `intendedAnswer` | unbiased: 1003 `true` / 997 `false` |

At round-1 sizes this is harmless, since each count is within one of its quota. Top-ups, though, are small and repeated, and they tilt toward `clear`. That undercuts §Q1.3's premise that a top-up "cannot make a cell easier".
*Fix:* place the leftover units by systematic sampling with a random start over the fractional remainders. Every count stays within one of its quota, and the expected count is exact. Apply this at every hierarchical level and to the balanced and conditional axes.

**m2 — Gaps in the message-type rubric at the confusable pairs.** Three cases are not covered:
- **(a) Off-topic questions.** An off-topic question fits both `question` ("asks for information") and `other` ("unrelated to the business"). The precedence rule then picks `question`, which contradicts `other`'s definition.
- **(b) Cancelling because of a problem.** "Cancel my order, it's two weeks late" could be an `order` (it cancels), a `complaint`, or an `escalation`, since "threatens… cancelling" is an escalation signal. The rubric does not separate carrying out a cancellation from threatening one.
- **(c) Returns for a changed mind.** A return with nothing wrong is not covered.

These are exactly the pairs the 250-item tranche rule tests (§Q1.2). Adding boundary lines now is cheaper than a rubric halt later.

**m3 — The guard's assistant turns can't be told from user turns.** `render_guard_input` never reads `role`, so turns render as `{displayName}: text`. The generator lets the assistant's `displayName` be any name. Yet the rubric's `false` rule, "only the assistant has stated the fact", requires the labeller to know which speaker is the assistant. The incumbent has the same handicap, so this adds label noise but does not bias the comparison.
*Fix:* the generator gives the assistant a name in the form production shows for an agent, such as a recognisable business-assistant name. The same applies to message type's `priorTurn.speaker`.

**m4 — Where the tie-breaks belong (question 2).** Each tie-break is acceptable:
- **Guard: "when unsure, answer `false`".** This repeats the incumbent's own prompt: "Answer true ONLY when the condition is clearly satisfied; when in doubt answer false" (`falkor-chat/server/falkorchat/app.py`, `_JUDGE_SYSTEM_PROMPT`). It therefore aligns the reference labels with what the incumbent was asked to do. The real FR-1 risk runs the other way: a rubric convention that *contradicts* the incumbent's prompt would count convention mismatches as incumbent errors, and it would favour the candidate, which is fitted to these labels.
- **Message-type precedence and join-in "unclear → `false`".** These are new abilities with no incumbent, so they cannot bias FR-1. For FR-10 they define the label itself, and they reduce three-way ties.

The tie-breaks belong in the rubrics, which `promptSha256` pins. The note should add one line of principle: a guard rubric convention must not contradict the incumbent's prompt. The join-in silence default is product behaviour. It is consistent with FR-12 (an @mention always gets a reply), and I will record it in the note.

**m5 — The provisional schemas (question 6).** Plan4 §4.3 already orders U4 → U5.2b → U5c.message-type → U5c.join-in, so no new-ability item is generated or labelled against a provisional contract. The remaining risk is that `points/` *describes* the rendering, and labels are bound to the rendered text. The relevant points:
- the rubrics' "What an item looks like" blocks;
- join-in's `maxItems: 8`, where the renderer keeps the last 6 turns and cuts the new message at 400 characters (plan v1.3 §3.2.1).

*Fix:* add `points/` conformance to U5.2b's done-condition. `inputFields` must equal U4's `INPUT_FIELDS_BY_ROLE`. A fixture rendered through the copied renderer must match the rubric's layout block. If join-in keeps up to 8 turns on purpose, to exercise the 6-turn cut, the generator must keep the agent's job visible within the last 6.

### Nits

- **n1.** Generation requests carry `conditionId`, `domainId`, `cueId` and `purposeId`. The generator never needs them. Keep them in the store, keyed by request key, to remove the leak channel the screen polices.
- **n2.** Guard items with 7–10 turns and join-in items with 7–8 turns lose their earliest turns to the cut, so some intent–label disagreement is expected by design. Diagnostics should split intent–label agreement by whether turns were cut.
- **n3.** Condition and domain are drawn independently, which pairs some conditions with domains they don't fit, such as `cond_03` (address) with `dom_08` (library help). This costs realism only. Let the generator adapt the business detail to fit the condition.

### Rulings on the implementer's questions

1. **Fixed, data-defined conditions, domains, purposes and cues: correct.** "Generator-authored" means written for generation, not taken from production. A fixed list keeps seeds reproducible from the RNG seed. It also keeps §Q1.3's exemption for condition sentences well-defined, and it stops conditions from correlating within a generation run. Nine non-production conditions meet the "≥ 8" requirement, and `cond_00` at weight 9/27 is exactly one third.
2. See m4 and M2's boundary rule.
3. **Allocation recomputed, and it matches.**
   - **Guard.** ⌈1.1 × 800⌉ = 880 seeds, so 220 per (kind, answer), split 146.7 / 73.3 → 147 clear / 73 near-miss.
   - **Guard conditions.** `cond_00` gets 880 × 9/27 = 293.3 → 294. The others get 65.2 → 65 or 66.
   - **Guard turn counts.** 44 seeds for each turn count.
   - **Message type.** 132 per label, split 88 clear / 44 near-miss. Opening is 35 + 18 = 53 (40.2%) and follow-up is 79, which matches §Q1.5's "about 40%".
   - **Balance.** Balanced axes are balanced across all seeds, not within each stratum. `cond_00` ranged from 65 to 79 per (kind, answer) in the probe, which is acceptable for a sampling variable.
   - **Rounding.** See m1.
4. **Top-up restriction: matches.** `_restricted` narrows only `designStratumAxis`, and intent stays unrestricted (1003 `true` / 997 `false` above). The rounding defect in m1 is separate.
5. **Leakage.**
   - Rubrics are per point and carry nothing specific to an item. No `seedId` reaches any request.
   - The generator sees intent by design, plus the ids in n1.
   - The real leak channel is structural: M1.
   - I checked the guard rubric and template against the 85 side-check items and their rationales. They share no 5-token span; the one shared 4-token span is "the user has not".
   - The 8-token shingle drop (§Q1.4) is the right tripwire, but it catches little. The generator never sees side-check text, so overlap can come only from chance or memorisation. The channel that actually exists is authoring. A test over the `points/` texts against the imported side-check shingle file (U5.10, `scripts/import_side_check_shingles.py`) would close it.
6. See m5.

## Pass 2 — U5.3 fix round (2026-11-01)

**Verdict: approve with suggestions.** Every Pass 1 finding that belongs to U5.3 is fixed, and I checked each by running it. Three small text fixes remain (P2-m1, P2-m2, P2-n1). None changes the design or blocks U5.4.

**Evidence.**
- **Suite.** 330 passed and 2 xfailed. `test_decision_contract.py::…[input-hash.jsonl]` was deselected, per the known U4.1 drift.
- **Probes** (scratchpad `u53-ds/`):
  - `pass2.py`: 200 round-1 guard draws of 880 seeds; 3,000 draws each at sizes 1, 2, 3 and 7; 100 round-1 join-in draws; 3,000 single-seed join-in draws.
  - `bind.py`: the obedience binding, run on hand-made items.

**Pass 1 dispositions.**
- **M1 — fixed.** The `missingForm` cases in `points/guard-judge/point.json` match note v1.11's weights. Realized round-1 draws reproduce the design:
  - non-empty `missing`: 27.3–28.2% of `true` items (mean 27.79%, design 27.8%) and 86.8–87.7% of `false` items (mean 87.49%, design 87.5%);
  - cue agreement: 79.3–80.2% (mean 79.85%, design 79.9%);
  - `missingForm` present exactly on `understanding` seeds, and `turnCount` exactly on `turns` seeds, in every draw.

  The binding (`emptinessAxis`) accepts each form's emptiness and rejects the opposite for `empty`, `listed`, `ambiguous` and `immaterial`, and it flags a miscopied condition. The `immaterial` text for clear seeds asks for a plainly unneeded presentation detail, as the side check's materiality probes do.
- **M2 — fixed.** Over 100 round-1 draws, `true` seeds draw only cue_01, cue_02 and cue_04 (7,328, 7,334 and 7,338), and `false` seeds draw all five (4,400 each). cue_05's text is narrowed. The rubric's "Small talk" boundary is present (`join-in-judge/rubric.md:65–67`).
- **m1 — fixed.** At draw size 1, `clear` is 0.655 against a design of 0.667, and every `missingForm` share is within Monte Carlo error of design. For example, `unlisted` is 0.041 against 0.042, and `ambiguous` 0.022 against 0.021. At sizes 2, 3 and 7 the shares match too. The systematic rounding composes with the multi-axis cases and with the conditional cue, as the join-in split at size 1 shows.
- **m2 — fixed.** The message-type rubric covers off-topic questions, cancelling because of a problem, and returns or exchanges, and its precedence is scoped to the business.
- **m3 — fixed.** Both generators fix the assistant as `Assistant` or `<business> Assistant`, and the guard rubric's speaker sentence matches.
- **m4 — fixed** in the note (v1.11 "Rubric conventions"). The rubrics carry the tie-breaks.
- **m5 — not U5.3.** It is routed to plan5, where U5.2b conformance covers it.
- **n1 — fixed.** No generator template names an `*Id` field. The ids stay in `Seed.axes`. Projecting them out of the request is U5.4's job.
- **n2 — open, not U5.3.** The split of intent–label agreement by turn cut belongs to U5.9's diagnostics. I will carry it at the note's next revision.
- **n3 — fixed.** `guard-judge/generator.md` has the "fits the business awkwardly" rule.

**New findings.**

**P2-m1 (minor) — The `ambiguous` form's text doesn't say `missing` is empty, but its binding requires it.** `guard-judge/generator.md` says the fact "is not listed in `missing`", which still lets `missing` hold other details. `emptyFor` includes `ambiguous`, so any such item is a `formMismatch`, which wastes a regeneration. A `false × near-miss` seed is exactly where a generator is tempted to pad `missing`. *Fix:* end the `ambiguous` rule with "`missing` is `[]`", as `unlisted` already does.

**P2-m2 (minor) — The message-type generator makes every follow-up depend on context.** `message-type-classifier/generator.md` says that for `follow-up`, "the type must depend on the message read together with `priorTurn`". In production many follow-ups stand alone ("Do you have it in blue?" after any turn). Forcing context dependence on all 60% of follow-up items skews the eval towards context-reading and makes the opening/follow-up comparison (§Q1.5) a comparison of two constructed difficulties. *Fix:* "`priorTurn` is the real preceding message; the message may stand on its own or need it, as in a real thread."

**P2-n1 (nit) — Two rules apply to the same `understanding` seed.** In `guard-judge/generator.md`, the `difficulty` text for a near-miss `true` seed ("one [fact] is given indirectly, late, or informally") overlaps the `missingForm` rules (`immaterial`/`indirect`). *Fix:* add "for `understanding` seeds, `missingForm` says how", so the generator follows one rule.

**Delegate question: `priorTurns: []` or omitted for an opening join-in item.** Method view only; the schema is U4's and U5.2b's to decide.
- **The rendered text is the same either way.** The renderer drops the block when there are no prior turns, so labels, the shingle check and the model's input are unaffected.
- **Only `input_hash` differs, and it must not.** The hash is item identity: duplicate detection, the FR-4 reuse count, and eval identity across pack versions. Two serializations of one input would split one item into two.
- **Requirement.** There must be one canonical form per role. Ingest normalizes to it before hashing, so the hash never sees the other form, and a vector row pins it.
- **Preference: `[]`, and it is weak.** `point.json`'s schema treats every field as required, and production's `prior_turns` is a list that is empty for an opening message. If U4 settles on omission, normalize to omission at U5 ingest; that costs nothing. Either way, U5.2b's copied vectors should include an opening join-in row.
