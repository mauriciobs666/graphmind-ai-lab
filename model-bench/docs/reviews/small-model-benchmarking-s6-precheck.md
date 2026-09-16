# Small-model-benchmarking S6 — `tool-caller` conversation scripts, agent pre-check (FR-19 process step 1)

> **Status:** active · **Owner:** `analyst` · **Tracks:** S6 (M6)

## Scope & verdict

Reviewed against `docs/plans/small-model-benchmarking-s6-spec.md` §2.2 (`Turn.expect` schema),
§2.6 (the FR-19 process this step belongs to), §3.2-§3.3 (the 12 scripts' turn outline and
coverage matrix), and §5 Step 5's own done-condition. Baseline artifact: `model-bench/packs/
tool-caller-shop-assistant/{conversations.jsonl,catalog.json,tools/schemas.json,tools/sim.py}` as
they actually stand on disk (commit `0794af9`), driven live rather than read and reasoned about.

**This is the agent pre-check named in §2.6 item 2 — not the stakeholder verification of §2.6 item
3.** I did not author any of the 12 scripts (a different `tdd-engineer` unit did), which is what
makes this check independent per the spec's own routing note (§4 Step 5). Nothing here substitutes
for the stakeholder's own Step 6 pass.

**Method — real environment, not inspection.** Built the real pack via `modelbench.packs.load_pack(
Path("packs/tool-caller-shop-assistant"))`, loaded `tools/sim.py` via `Pack.load_tool_module()`
(the same `importlib.util.spec_from_file_location` path `runner.py` uses), and drove every one of
the 12 scripts' turns against one fresh `ShopEnvironment` per script (state carried across a
script's own turns, matching `sim.py`'s own per-conversation-instance contract). For every
`toolRequired: true` turn: dispatched the scripted `tool`/`args`, printed the real return value, and
cross-checked every `finalReplyMustContain`/`finalReplyMustNotContain` price-looking literal against
`catalog.json` directly (lookup turns) or against the real running cart total computed by the same
add/remove sequence already dispatched earlier in that same script (`view_cart` turns). Also called
`validate_pack` on the real, on-disk pack (`[]`, clean) and, for the one finding below that needed
scorer-level evidence rather than surface data, called `modelbench.scoring.toolcalls.
argument_correctness`/`right_tool_chosen`/`spurious_and_duplicate` directly against the real
dispatch results. Driver script and full run log are in the appendix's own paths (not repo
artifacts — session scratchpad, redirected to a file as instructed rather than held only in
context).

**Verdict: needs changes.** 9 of 12 scripts are clean. 3 carry named issues from two distinct
findings — one a **blocker** (A-03 turn 4: a scorer-verified, guaranteed false failure on a model
that answers correctly) and one non-blocking pattern repeated across two scripts (B-02 turn 7, B-03
turn 6: a promised assertion the outline names but the authored turn never writes) — plus one
documentation-only defect in the
spec's own §3.3 coverage matrix (a citation naming the wrong turns, not a pack defect). Every named
issue is precise enough to hand to a fix unit without re-deriving the analysis; **Step 6
(stakeholder verification) should not begin until at least the A-03 blocker is fixed and
re-checked**, per §5 Step 5's own done-condition ("every named issue is fixed and re-checked before
Step 6 begins").

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(confirmed via `mcp__cypher__query` `GRAPHS`, 2026-09-16); this is a code-level task in a component
with no CPG, so "considered, not relevant" applies rather than "not applicable."

## Per-script disposition

| Script | Disposition | Issues |
|---|---|---|
| A-01 | Clean | — |
| A-02 | Clean | — |
| A-03 | **1 issue (blocker)** | Turn 4: `expect.args` cannot represent the compound compare-two-products call the turn's own customer question requires — see Finding 1 |
| A-04 | Clean | (one wording nit, non-blocking — see "What's solid" note) |
| B-01 | Clean | — |
| B-02 | **1 issue (minor)** | Turn 7: outline promises "confirming the post-order empty cart," authored turn asserts nothing — see Finding 2 |
| B-03 | **1 issue (minor)** | Turn 6: outline promises "confirming empty," authored turn asserts nothing — see Finding 2 |
| B-04 | Clean | — |
| C-01 | Clean | — |
| C-02 | Clean | — |
| C-03 | Clean | — |
| C-04 | Clean | — |

9 scripts fully clean, 3 carry a named issue (1 blocker + 2 instances of the same minor pattern).
A fourth, separate finding (Finding 3) is against the spec document's own §3.3 table, not against
any of the 12 scripts.

## Findings

### Finding 1 (blocker) — A-03 turn 4's `expect` block guarantees a false failure on a correct answer

`conversations.jsonl` A-03 turn 4: `{"user": "Which is cheaper -- the Bluetooth Speaker or the
Noise-Cancelling Headphones?", "expect": {"toolRequired": true, "tool": "lookup_product_fact",
"args": {"name": "Bluetooth Speaker"}, "finalReplyMustContain": ["79.99", "149.99"]}}`. The spec's
own outline (§3.2, A-03 row) names this turn's intent explicitly: "one tool, two distinguishable
calls, both required-tool-correct." Answering the question **correctly** requires looking up both
products — the turn's own `finalReplyMustContain` demands both `79.99` (Bluetooth Speaker) and
`149.99` (Noise-Cancelling Headphones) appear in the reply, and a model cannot state the second
price without calling `lookup_product_fact({"name": "Noise-Cancelling Headphones"})` too.

I drove exactly that correct, two-call behavior against the real environment and then ran the real
`modelbench/scoring/toolcalls.py` functions over the result (`_score_one_conversation`'s own logic,
`toolcalls.py:794-807`):

```
dispatch('lookup_product_fact', {'name': 'Bluetooth Speaker'})            -> {'found': True, 'price': 79.99}
dispatch('lookup_product_fact', {'name': 'Noise-Cancelling Headphones'})  -> {'found': True, 'price': 149.99}
right_tool_chosen({'lookup_product_fact'}, {'lookup_product_fact'})       -> True
argument_correctness({'name': 'Bluetooth Speaker'}, {'name': 'Bluetooth Speaker'})           -> allCorrect=True
argument_correctness({'name': 'Bluetooth Speaker'}, {'name': 'Noise-Cancelling Headphones'}) -> allCorrect=False, wrongValue=('name',)
spurious_and_duplicate(...)                                               -> spurious=False, duplicateWithinTurn=False, duplicateCrossTurn=False
```

`_score_one_conversation` (`toolcalls.py:795-807`) runs `argument_correctness` against **every**
dispatched call whose name matches the turn's single required tool name — both calls match here,
since both are `lookup_product_fact` — and checks each one against the **same** single
`expected_args = {"name": "Bluetooth Speaker"}`. The second, entirely necessary call is therefore
always scored `wrongValue`, which sets `args_all_correct = False` for the whole turn
(`toolcalls.py:796-807`), which makes `clean = False` (`toolcalls.py:843-853`) regardless of how
correct the model's actual answer was. Neither `right_tool_chosen` (same tool name both times) nor
`spurious_and_duplicate` (`(name, args)` signatures differ, so nothing collides) catches this —
confirmed by running both directly against the two real calls, shown above. So the outline's own
description of the risk here ("a real opportunity for a spurious/duplicate miscount," §3.3's
coverage-matrix row for FR-8(e)) is itself imprecise: this turn cannot ever register as spurious or
duplicate under the real scorer; it is guaranteed to register as an **argument-correctness**
failure on the second, correct call, every time a model answers the compound question thoroughly.

**Root cause:** `Turn.expect`'s schema (§2.2) supports exactly one required tool name and one
`args` dict, checked against every matching dispatched call — it has no way to express "this turn
legitimately needs two distinguishable calls to the same tool, each with its own expected args."
A-03 turn 4 is the only turn in all 12 scripts that asks a genuinely compound question, so this is
the one place the schema's single-call assumption breaks.

**Suggested fix (for the owning fix unit, not applied here):** the schema is not the cheapest thing
to change (it is read by `score_conversations` exactly as documented and every other turn in the
pack fits it); the turn is. Either (a) split A-03 turn 4 into two consecutive turns, each asking
about one product and each with its own single-call `expect` (loses the "compound turn" coverage
this turn was meant to add, but is schema-conformant), or (b) rephrase the turn so only **one** new
lookup is required — e.g. referencing a price the script already established in an earlier turn
rather than asking about two products neither yet looked up. Whichever direction is chosen, verify
the fix the same way this finding was found: dispatch the turn's own intended correct behavior
against the real environment and run it through `argument_correctness`/`right_tool_chosen`/
`spurious_and_duplicate` directly, not by re-reading the JSON.

### Finding 2 (minor) — B-02 turn 7 and B-03 turn 6 don't check what the outline says they check

Both turns are schema-legal (`toolRequired: true` turns may omit both `finalReplyMustContain` and
`finalReplyMustNotContain` per §2.2 — they simply fall into `unscoreableReturns` rather than
failing), so neither is a scoring bug. But the spec's own outline (§3.2) explicitly names each
turn's role as **verifying an empty cart**, and the authored `expect` blocks assert nothing that
could ever catch a wrong result there:

- **B-02 turn 7**: outline — `"(7) view_cart confirming the post-order empty cart"`. Authored:
  `{"toolRequired": true, "tool": "view_cart", "args": {}}` (no `finalReplyMustContain`/`Not`). Real
  dispatch after `place_order` clears the cart: `{"items": [], "total": 0.0}` — confirmed live.
- **B-03 turn 6**: outline — `"(6) view_cart confirming empty"`. Authored:
  `{"toolRequired": true, "tool": "view_cart", "args": {}}` (same omission). Real dispatch after
  `clear_cart`: `{"items": [], "total": 0.0}` — confirmed live.

In both cases the real environment does behave as the outline describes (cart genuinely empty), so
there is no live-environment mismatch — only a missed opportunity: as authored, neither turn can
ever fail even if a model hallucinated leftover cart contents in its reply, because nothing checks
the reply text. **Suggested fix:** add `"finalReplyMustContain": ["0.00"]` (or a phrase-based
assertion, e.g. `finalReplyMustNotContain` naming the removed/ordered items) to both turns' `expect`
blocks, matching the pattern every other `view_cart` turn in the pack already uses (B-01 turns 3/6,
B-02 turn 3/5, B-04 turn 4 all assert a literal total).

### Finding 3 (minor, documentation-only — not a pack defect) — §3.3's coverage-matrix citation for B-02's duplicate-instruction re-probe names the wrong turns

`docs/plans/small-model-benchmarking-s6-spec.md` §3.3's FR-8(e) row reads: *"the ministral-pattern
re-probe: B-02 (t3-4)."* The real re-probe pattern is between **turn 1** (`add_to_cart(Mechanical
Keyboard, quantity=2)`) and **turn 4** (`add_to_cart(Mechanical Keyboard, quantity=1)`, the "just
one more of the first thing I asked for" turn) — turn 4 is where a model exhibiting §8.4's
documented defect would incorrectly re-issue turn 1's exact call
(`spurious_and_duplicate`'s `duplicateCrossTurn` keys on the exact `(name, args)` signature,
confirmed by reading `toolcalls.py:330-364`). **Turn 3 is `view_cart`** — unrelated to any dispatch
signature and incapable of triggering the duplicate check the row is citing. A reader using this
row to locate B-02's re-probe coverage during Step 6 would look at the wrong turn. This is a defect
in the spec document's own table, not in `conversations.jsonl` — the pack's B-02 turn 4 itself is
correctly authored and does exercise the re-probe (confirmed live above, §3.7's own cross-reference
to B-02 turn 4 by name is accurate). **Suggested fix:** correct the citation to `B-02 (t1, t4)` in
`docs/plans/small-model-benchmarking-s6-spec.md` §3.3 (this document is `active`, so the correction
is an in-place edit per the repo's document-lifecycle convention, not a new document).

## What's solid

- **9 of 12 scripts are fully clean** — every tool name real, every `args` shape accepted the way
  the turn assumes, every price literal verified against the live catalog, every `view_cart` total
  verified against the real cart state built by that script's own prior turns (not just plausible —
  computed).
- **All 20 abstention turns' target product names are genuinely absent from `catalog.json`**
  (`Smart Watch`, `Gaming Monitor`, `Laptop Stand`, `Office Chair`, `Desk Organizer`, `Yoga Mat`,
  `External Hard Drive` — none appear in the 14-product catalog, confirmed by direct dispatch:
  every one returns `{"found": false}`), including the one deliberate near-miss fabrication guard
  (A-03 turn 9, `"Office Chair"` vs. the real `"Ergonomic Office Chair"`).
  `Turn 5` of A-03 (`"Laptop Stand"`) and `A-01` turns 5-6 abstain correctly too.
- **Both `boundaryRule` arguments (`minPrice: confusedWith:[10.01]`, `maxPrice:
  confusedWith:[4999,5000]`) are exercised with the correct, non-confused values** in every script
  the outline names (A-01, A-02, A-03, C-03) — the scripts never need to *use* a confused value
  themselves (the mechanism only fires if a live model's own dispatched value matches one), so this
  is working as designed, not a gap.
- **All six restraint turns are genuine** — none secretly needs a tool call given the real
  conversation state at that point (verified by reading each customer utterance against its
  script's actual prior turns, not just the `toolRequired: false` flag): A-02 t6/t9, A-04 t3, B-03
  t7, B-04 t6, C-02 t1.
- **`validate_pack(load_pack(...))` returns `[]`** on the real, on-disk pack — re-run live during
  this check, independently confirming Step 4's own result still holds.
- **Every script carries at least one `finalReplyMustContain`/`finalReplyMustNotContain` turn**,
  matching §3.3's own claim that FR-8(g) has a non-empty denominator in all 12.
- One non-blocking content nit, noted for completeness rather than as a finding: A-04 turn 3's
  restraint utterance ("you already told me that **a second ago**") slightly overstates how many
  times the Standing Desk Mat price was given (once, in the immediately preceding turn) — it still
  functions correctly as a restraint turn (no new information is requested), so this is phrasing
  polish, not a defect worth a fix unit's time on its own.

## Open questions

- Finding 1's two suggested fix directions (split vs. rephrase) are both viable; which one is
  chosen affects whether A-03 keeps 9 turns or grows to 10 — worth a quick call from whoever owns
  the fix-and-re-check pass (§5 Step 5's own "fixed and re-checked before Step 6 begins" clause)
  rather than this review picking one unilaterally.
- Finding 2's fix (adding a literal assertion to B-02 t7 / B-03 t6) is mechanical and low-risk; no
  open question there.

## Appendix — verification artifacts

- Driver script: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/
  4d079957-75eb-4526-b48b-7ce25e57c4fe/scratchpad/precheck.py` (session scratchpad, not a repo
  artifact — drives all 12 scripts against the real `ShopEnvironment`, one fresh instance per
  script, and cross-checks prices/totals/abstentions).
- Full run log (all 12 scripts, every turn's real dispatch result): same directory,
  `precheck_output.log`.
- Finding 1's scorer-level deep-dive (the `argument_correctness`/`right_tool_chosen`/
  `spurious_and_duplicate` run shown in Finding 1): same directory,
  `precheck_a03_t4_deepdive.log`.
