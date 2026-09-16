# `model-bench` S6 — `tool-caller` pack, part 2: the conversation scripts — implementation spec

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `docs/plans/small-model-benchmarking.md` (S6)

## 1. Goal & scope

Build the two data artifacts `docs/plans/small-model-benchmarking.md`'s "### S6 — `tool-caller`
pack, part 2" section requires (plan `:5879-5923`) — `packs/tool-caller-shop-assistant/
conversations.jsonl` (12 human-verified, fixed conversation scripts) and its `PROVENANCE.md` —
plus `prose_calibration.jsonl` (~20 labelled replies calibrating the prose-pseudo-call detector
S5 already shipped as a pure function, §2.5 below) and the small amount of production code needed
to make the pack S5 built actually **use** what S6 authors: a `validate_pack` invariant the plan
requires but S5 never built (§2.3), and the wiring that turns the prose-detector calibration data
into a printed precision/recall figure (§2.5) — neither is content authoring, both are real,
confirmed code gaps found by reading `modelbench/packs.py` and `modelbench/scoring/toolcalls.py`
directly, not inferred from the plan's prose. Also in scope: a manifest-fidelity correction to the
already-shipped `pack.json` (§2.4) — three fields where S5's shipped pack diverges, unflagged,
from the top-level plan's own canonical manifest literal for this exact pack, one of them directly
load-bearing for what S6's own known-answer validation measures. Then the stage's own
process-and-execution work: the FR-19 human-verification procedure the coordination has already
settled with the stakeholder (§3.5), the determinism probe (plan step 4), and the live
known-answer validation against `qwen/qwen3-4b-2507` and `mistralai/ministral-3-3b` (plan step 5).

**Out of scope:** anything S5 already shipped and closed (the storefront, the scorer's per-turn
pure functions, the three `report.py` renderers, `_load_conversation_scorer` — all confirmed live
by direct reading, §2.1); the deterministic `H`-bounded headline's own statistical machinery
(`stats.py`, `-ml` §3.4 — unchanged, this stage only supplies the data the machinery already
consumes correctly); a judged reply-quality layer (not this pack's role); re-opening the
2026-09-02 sizing decision (12×1, unchanged, `-ml` §4.5) or R-3's bisect ladder beyond running its
first rung (§3.4 below) — a non-reproduction, if it happens, is recorded and the bisect itself is
future work, not this stage's to execute past its first data point.

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(confirmed live by `teco`'s brief, re-checked 2026-09-16 via `mcp__cypher__query` `GRAPHS`); this
is a code-level task in a component with no CPG, so "considered, not relevant" applies rather than
"not applicable."

## 2. Context & findings

### 2.1 What S5 already shipped and needs no S6 work — confirmed by reading the shipped code directly

S5 is closed (`docs/plans/small-model-benchmarking-coordination.md`, "S5 is fully closed as of
this row," U139-U150). Reading the actual tree rather than trusting that record alone:

- **The storefront is real and on disk**: `packs/tool-caller-shop-assistant/{catalog.json,
  tools/schemas.json, tools/sim.py, prompts/system.md}` all exist and match S5 spec §3.3/§3.5.
  `catalog.json` holds 14 products across three categories (Electronics ×8, Office ×4, Kitchen
  ×2), price range $9.99-$199.99, one item at exactly $10.00 (`notebook-set`) and one at zero
  stock (`4k-webcam`, $59.50). `tools/schemas.json` declares the seven tools and already carries
  `boundaryRule` on two arguments: `filter_products.minPrice` (`confusedWith: [10.01]`) and
  `filter_products.maxPrice` (`confusedWith: [4999, 5000]`) — the dollars-vs-cents and
  off-by-a-cent boundary traps FR-8(d) exists to catch, ready for scripts to exercise (§3.2).
- **`modelbench/scoring/toolcalls.py` (1004 lines) is complete for every function S6's own
  scripts will drive**: `score_conversations` (the `ConversationScorer` assembly), all nine
  FR-8-letter functions, `clean_through_turn`/`hazard_points` (rule 5's censoring), and
  `outcome_vectors_differ` (the determinism probe's comparator) are all shipped, unit-tested
  against synthetic fixtures, and confirmed reading the module directly rather than trusting the
  ledger's own account of it.
- **`runner.py`'s `_load_conversation_scorer` is live** (`runner.py:244-250`, confirmed by
  reading it — resolves `modelbench.scoring.<pack.manifest["scorer"]>` exactly as
  `_load_item_scorer` does for the four item-level roles) — the brief's own ask to verify this
  rather than take it on faith. **`_drive_conversations` (`runner.py:461-541`) already wires the
  whole live-run path end to end**: it drives every scored script, then every
  `sampling.determinismProbeScripts` script a second time (`runner.py:519-531`), computes
  `basis` from `replicatesPerScript == 1 and ran and determinism_probe.get("identical")`
  (`runner.py:534-542`) and returns `design_effect = 1.0` unconditionally, per `-ml` §4.5.1's
  by-construction argument for the 12×1 design. **This stage owes zero runner/scorer code for the
  determinism probe or the `basis` computation — both are already correct and already tested.**
  What S6 owes here is *data* (the two named probe scripts) and a *live run* (§3.6, §4 Steps 7-8).
- **`report.py` already renders the funnel table, the per-turn-position table and the hazard
  curve** (S5 §4.4, confirmed present at the module's tool-caller-gated renderer functions) and
  already prints the standard verdict-string machinery (`-ml` §3.2/§3.4) for `cleanThroughTurnH`,
  the pack's one verdict metric. Nothing here needs S6 work either.
- **`pack.json` declares `data.conversations`/`data.prosePseudoCallCalibration` pointing at files
  that do not exist yet** — confirmed by `ls`: `conversations.jsonl`, `prose_calibration.jsonl`
  and `PROVENANCE.md` are all absent from `packs/tool-caller-shop-assistant/`. This matches S5
  spec §3.2's own statement that "this pack cannot pass `load_pack`/`validate_pack` until S6
  lands" — S6 is the stage that completes the pack for the first time, not a stage revisiting a
  finished one.
- **Baseline, run this session**: FalkorDB up (`redis-cli ping` → `PONG`); LM Studio reachable at
  `localhost:1234` with both `qwen/qwen3-4b-2507` and `mistralai/ministral-3-3b` present in
  `/v1/models` (re-checked live 2026-09-16, superseding the 2026-09-14 prior the brief flagged as
  two days stale) — availability only, not residency; §4 Step 7 below states what a live run
  still needs to confirm immediately before it drives either model.

### 2.2 `Turn.expect`'s schema — already fixed by S5's own scorer code, not this stage's to invent

`conversations.jsonl` is read by `Pack.iter_scripts()` (`packs.py:305-325`) into `convo.Conversation`/
`convo.Turn` objects; `Turn.expect` is a plain `Mapping[str, Any]` (`convo.py:174-187`) whose
**fields are read by `modelbench/scoring/toolcalls.py`, not by `convo.py`** (`convo.py`'s own
docstring: `Turn.expect` is "a scoring oracle... `assemble` never reads it"). Reading
`_required_tool_names`/`_score_one_conversation` (`toolcalls.py:724-857`) directly, the schema
every authored turn must conform to is:

```json
{"seq": 1, "user": "<the scripted customer utterance>",
 "expect": {
   "toolRequired": true,
   "tool": "lookup_product_fact",
   "args": {"name": "Wireless Earbuds"},
   "finalReplyMustContain": ["49.99"],
   "finalReplyMustNotContain": []
 }}
```

- **`toolRequired: false`** (or the key omitted) is a **restraint** turn — `_required_tool_names`
  returns `{}`, and `"tool"`/`"args"` are not read at all. This is the schema S6's own
  restraint-coverage turns use (§3.3).
- **`toolRequired: true`** requires exactly **`tool`** (one required tool name — this pack's
  `required_names` is always a singleton in practice, since every scripted customer request maps
  to one tool) and **`args`** (a mapping of expected argument name → expected value, checked via
  `argument_correctness`'s `_scalar_equal`, `-ml` §4.2(d)'s numeric-epsilon/canonical-string rule
  — never coerced across type).
- **`finalReplyMustContain`/`finalReplyMustNotContain`** are both optional, read by
  `reply_matches_tool`; a turn declaring neither falls into `unscoreableReturns` (FR-8(g) simply
  has nothing to check that turn) rather than failing — so a turn only exercises (g) when the
  script author writes at least one of these two lists. **Every scored turn intended to exercise
  (g) must declare at least one of them; a turn that should not (e.g. a cart-mutation confirmation
  with no fixed literal worth asserting) may legitimately omit both.**
- **`argChecks`** (the plan's own literal example, plan `:2190`) is **not read anywhere in
  `toolcalls.py`** — S5 spec §3.3 relocated the boundary/unit rule onto the pack's own
  `tools/schemas.json` (`boundaryRule`, already shipped, §2.1 above), so a scripted `argChecks`
  block is inert data the scorer tolerates and ignores if present. **S6's own scripts do not
  author `argChecks`** — there is nothing downstream that would ever read it, and authoring dead
  data invites a future reader to assume it does something.
- **`Conversation`'s own row fields** (`packs.py:315-325`): `scriptId`, `shape` (`"A"`/`"B"`/`"C"`
  — the reporting stratum), `replicate` (always `1` under the 12×1 design), `turns` (the array
  above), `description` (free text), `provenance` (a per-row object — §3.5 below fills
  `draftedBy`/`verifiedBy`/`basedOn` here, the plan's own example shape, `:2193`).

### 2.3 A real code gap: `validate_pack` never checks `H <= min(script length)`

The plan's S6 done-condition requires "the pack validates, including `H ≤ min(script length)`...
(§3.3)" (plan `:5908-5909`), and `-ml` §4.5.1's own footnote calls this "a validated pack
invariant, not an assumption" — violate it and the paired table silently loses every conversation
too short to reach turn `H`, shrinking `n_eff` in the *optimistic* direction with nothing printed
to say so. **Reading `modelbench/packs.py`'s `validate_pack` (`:886-926`) and its seven checked
axes directly: none of them is this one.** `_sampling_problems` (`:692-715`) checks the
`replicatesPerScript > 1` refusal and the row-count identity (script count × `analysisUnit`
distinctness); nothing anywhere reads `metrics.cleanThroughTurnH.H` against the authored scripts'
own turn counts. `report.py:126`'s own comment confirms the gap from the other side: *"This is
also the only place a violated `H <= min(script length)` would surface"* — i.e. today, silently,
at report time, not at `validate` time, which is exactly the "loses conversations too short to
reach turn `H`... silently" failure `-ml` §4.5.1 warns against. **This is a real, confirmed,
previously-unbuilt check — S6's own Step 0 (§4 below) closes it**, mirroring `_answerability_stamp_
problems`'s own pattern (`packs.py:867-885`): a small, role/metric-scoped function, added to
`validate_pack`'s call sequence, with its own fixture pack and test.

### 2.4 A real, confirmed manifest gap: three `pack.json` fields diverge from the plan's own canonical example, unflagged — one of them load-bearing for S6's own validation target

The top-level plan's §3.3 manifest literal (`:432-465`) is not an illustration — its prose reads
*"which is what `tool-caller-shop-assistant` declares"* (`:503-504`), naming the exact pack this
stage's data lands in. Diffing that literal against the shipped `packs/tool-caller-shop-assistant/
pack.json` (read directly, both sides quoted in full below) surfaces three divergences, **none
flagged anywhere in the S5 spec or its own two gate passes** (`analyst` U144's S5 code gate,
`qa-engineer` U148's acceptance — U148's own TP-004 confirms `_prompt_problems` validated
`historyReplay` as a *legal* enum member, never checked it against the plan's *canonical* value
for this pack, so the gap passed both gates unnoticed rather than being a disputed, resolved call):

| field | plan `:432-465` | shipped `pack.json` | consequence |
|---|---|---|---|
| `prompt.historyReplay` | `"structured-replies-only"` | `"structured"` | **load-bearing, §3.4 below** |
| `prompt.representToolSchemasEachTurn` | `true` | `false` | low — `convo.py`'s own docstring: independent of whether native tool-calling stays available, "a knob about restated prose" |
| `prompt.maxTokens` | `1024` | `512` | low-moderate — truncation risk on a verbose reply, no stated mechanism tie |

**`historyReplay` is the one that matters, and the plan is explicit about why (`:492-507`).**
`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §4.1 attributes the documented
turn-4 collapse to the model's own prior turns being replayed with **no visible evidence a tool
was ever used** — `structured-replies-only` is the one `historyReplay` value that reproduces
exactly that shape (native roles, final reply text only, no `tool_calls`, no `tool` messages).
`structured` — the value actually shipped — replays the **full** tool scaffolding on every prior
turn, which is precisely the *first rung of R-3's own bisect ladder* (plan `:6389-6392`: *"`structured`
moves the tool-evidence axis"*), not the baseline the known-answer validation is supposed to start
from. Running S6's step 5 against the shipped value would silently execute R-3's bisect rung 1
as if it were the baseline replication — exactly the ambiguity R-3 exists to keep attributable
("localises the cause to the axis that moved," `:6392`), collapsed by omission before the baseline
was ever tried.

**Resolved here, not deferred as an open question, on the same grounds this coordination's own
U146 precedent used for a comparable S5-spec-vs-shipped gap**: the plan's text is unambiguous (not
an example — a named declaration for this exact pack), the fix is mechanically safe (verified
below), and — critically — **the blast radius is small because nothing has run against this pack
yet**. S5's own spec states the pack cannot `validate`/`run` until S6 supplies
`conversations.jsonl`; no live run, no published report, no gated acceptance evidence has ever
been produced under the wrong value. Correcting it now, before the pack's first live run, is the
cheapest possible time — the alternative (leaving it, running S6's known-answer validation, and
discovering only then that the result cannot be attributed to "scripts differ" vs. "wrong replay
axis") is the expensive direction, and is exactly the failure this finding prevents while it is
still free to fix.

**Mechanically safe, checked directly**: `historyReplay` only changes what a **prior** turn
contributes to the *prompt sent to the model* (`convo.assemble`, per its own module docstring);
it has no effect on what `drive` **records** for the turn being driven right now —
`TurnTrace.dispatches`/`.finalReplyText`/`.turnDisposition` all come from `env.trace()`/the live
`ChatResult` regardless of replay mode. So `scoring/toolcalls.py` — S5's already-shipped, already-
tested code — needs no change; only the pack's own declared config differs from the plan for a
reason nobody stated. `tests/test_runner.py:279`'s own `"structured"` literal is a **synthetic**
test-only pack fixture for `_drive_conversations`'s generic mechanics, unrelated to this real
pack's manifest — not touched by this correction.

The other two fields are corrected for the same fidelity reason (an unflagged, unexplained
deviation from the plan's own named literal is itself the defect this coordination's honesty
discipline exists to catch) but are lower-stakes and not separately gated: `representToolSchemasEachTurn:
true` costs a few more tokens per turn (already budgeted — `-ml` §4.5.2's cost estimate is bounded
by `maxIterationsPerTurn`, not by this knob) and `maxTokens: 1024` only ever *raises* the ceiling a
reply can occupy, never truncates something the shipped `512` would have let through.
`sampling.seed` (plan `20260902`, shipped `20260913`) is **not** corrected — it is an arbitrary
bootstrap-resample seed with no stated mechanism tie, unlike the three fields above.

### 2.5 A real code gap: the prose-detector's precision/recall is never computed or printed

The plan requires (`-ml` §4.2(a)+(b), quoted at plan `:2296-2298`): *"the pack ships ~20 labelled
replies and the report prints the detector's own precision and recall."* S5 shipped
`detect_prose_pseudo_call`/`prose_detector_precision_recall` as pure functions
(`toolcalls.py:158-187`), unit-tested against small hand-built lists — but **grep confirms
`prose_detector_precision_recall` is never called from `score_conversations` or anywhere else in
production code** (`grep -rn prose_detector_precision_recall modelbench/` returns only the
function's own definition and its docstring cross-reference). Three consequences, all real:

1. **No `Pack` method reads `data.prosePseudoCallCalibration`.** `Pack.data_path(key)`
   (`packs.py:260-265`) is fully generic and already resolves the key correctly, but there is no
   `iter_items`/`iter_scripts`-shaped reader that turns the JSONL rows into the
   `Sequence[tuple[str, bool]]` `prose_detector_precision_recall` takes.
2. **`ToolCallAggregates` has no field to carry the result.** `determinismProbe`/`iterationSummary`
   are the two precedents (`Mapping[str, Any] | None`, generic `_encode`/`_decode` treatment,
   `results.py`) — nothing analogous exists for the detector's own precision/recall.
3. **`report.py` has no renderer for it.**

**This is S6's to close**, small and additive, following the `determinismProbe`/`iterationSummary`
pattern exactly (§4 Step 1 below): a `Pack.iter_prose_calibration()` reader, a
`ToolCallAggregates.prosePseudoCallDetector: Mapping[str, Any] | None` field
(`{"n": int, "precision": float, "recall": float}`, or `None` when the manifest declares no
calibration data — mirroring `prose_detector_precision_recall`'s own `None`-on-empty-corpus rule),
`score_conversations` calling it once per run when `"prosePseudoCallCalibration" in
pack.manifest.get("data", {})`, and one `report.py` line printing it (or naming its absence)
beside the funnel table's existing (a)+(b) partition line.

### 2.6 The FR-19 process — settled by the stakeholder, restated here in full per the brief's own instruction

From `docs/plans/small-model-benchmarking-coordination.md`, "## S6 kickoff — 2026-09-14" (read in
full): S6 authors 12 new conversation scripts whose `expect` blocks are, per the plan's own FR-19,
required to be "checked by a person against the simulated environment's actual behavior" before
`provenance.verifiedBy` is filled — and the plan itself names S6 as the one stage whose scoring
correctness "cannot be checked against anything except itself." `teco` put the choice to the
stakeholder with its own recommendation (full stakeholder review) and, in parallel, obtained an
independent `data-scientist` opinion formed *before* seeing the stakeholder's first answer. The
stakeholder's first answer — "independent agent review only" — diverged from FR-19's literal
wording; `data-scientist`'s opinion (received after) argued specifically against that option: an
independent *agent* reviewing an authoring *agent's* work is **correlated error, not independent
verification**, since both are simulating "what should this look like" rather than checking the
live environment's actual output — precisely the failure mode FR-19 exists to break. `teco`
surfaced that divergence back to the stakeholder in a second `AskUserQuestion` round rather than
silently proceeding on the first answer.

**Final, binding decision: "agent pre-check + your review."** Concretely, and binding for this
stage's `provenance.verifiedBy` step:

1. **Agents draft** the 12 scripts and their `expect` blocks (§4 Steps 2-3 below — content
   authoring against the real, on-disk `catalog.json`/`tools/schemas.json`/`tools/sim.py`).
2. **An independent, non-authoring agent pre-checks all 12** — reads every `expect` block against
   `tools/sim.py`'s actual dispatch logic and `catalog.json`'s actual data (not against what a
   model *might* produce) and reports likely mismatches before the stakeholder's own pass, to
   surface obvious errors cheaply and cut the stakeholder's review time. This agent must not be
   the one that drafted the script it is checking (§4 Step 5's own routing note).
3. **The stakeholder personally executes/verifies all 12** (not a sample) against the real
   simulated environment's actual behavior, before `provenance.verifiedBy` is filled on any row.
   This is the step FR-19 itself names and the one no agent step may substitute for, per
   `data-scientist`'s own argument above.

No conversation script's `provenance.verifiedBy` may be filled by anyone but the stakeholder
performing step 3, and no run of this pack that feeds a published report may claim
`known-answer validation` results (§3.7) drawn from scripts whose `verifiedBy` is still empty.

### 2.7 Environment, re-checked live this session

FalkorDB: `redis-cli ping` → `PONG`. LM Studio `/v1/models` at `localhost:1234` lists both
`qwen/qwen3-4b-2507` and `mistralai/ministral-3-3b` (2026-09-16, superseding the 2026-09-14 check
the brief flagged as stale) — confirms **availability**, not residency; §4 Step 7 states what
the live-run step must still confirm immediately before driving each arm (a model can be available
in LM Studio's catalog without being the one currently loaded into memory).

## 3. Design & rationale

### 3.1 Reconstruction method: shapes A/B/C, turn-by-turn, adapted to this pack's own catalog

`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.1 (read in full) describes three
conversation *patterns* run against **falkor-chat's own live storefront** (its own catalog,
products, and `salesperson@v7` tool set) — not a file this pack can copy. The plan's own step 1
says "reconstruct... turn by turn" (`:5883-5884`), which this document reads as: preserve each
shape's **turn count, turn-type sequence, and the specific defect-eliciting pattern** §8.1 used,
re-expressed against `tools/schemas.json`'s seven tools and `catalog.json`'s 14 products — never a
verbatim transcription of §8.1's own falkor-chat-specific product names or prices, which do not
exist in this pack.

- **Shape A (9 turns, read-only catalog)** — §8.1's own sequence: exact-name lookup, category
  filter, price-range filter + a repeat of it, an abstention pair (two products not in the
  catalog), then a second exact-name lookup repeated/rephrased three times.
- **Shape B (7 turns, write-mutating)** — §8.1's own sequence: add ×2 of item X, add ×1 of item Y,
  view cart, remove ×1 of item X, remove item Y entirely, view cart, add ×1 of item Z.
- **Shape C (4 turns, read-only, short)** — a turn-count-sensitivity probe: does the same onset
  point (§8.2's finding: turn 4, independent of content) recur in a conversation too short to
  reach turn 5 at all, or does a short conversation behave differently.

**Extension, per plan step 2 (`:5885-5891`)**: 4 distinct scripts per shape (12 total, each run
once, `replicatesPerScript: 1` — the settled 2026-09-02 sizing, unchanged), and coverage of the
FR-8 failure kinds §8.1's own three scripts were never designed to exercise — in particular
**restraint** (§8.1 has zero `R(t) = ∅` turns: every one of its turns needed a tool), and turns
whose `expect` block gives `reply_matches_tool` (g) and `argument_correctness`'s boundary/unit
subset (d) something concrete to check. **This reconstruction can produce 12 distinct,
non-contrived scripts covering every required FR-8 kind without inventing a scenario that does not
trace to §8.1's own method** — verified below by construction, not asserted: §3.2's per-script
table derives each script's role in the coverage matrix from the turn outline that generates it,
and §3.3's matrix is the union of those roles, checked column-by-column against the FR-8 letters
(a)-(g) plus restraint plus the two `boundaryRule` arguments. No script exists solely to fill a
matrix cell with no plausible customer intent behind it.

**What "coverage" means here, precisely — and why it is a denominator claim, not a forced-failure
claim.** A script cannot force a live model to fail in a particular way (that is what the
known-answer validation *measures*, not what a script *scripts*); what a script controls is
whether a given FR-8 count has a **non-empty denominator** at all across the 12-script pack. §8.1's
own three scripts left restraint's denominator at zero and left (g)'s denominator thin (few
`finalReplyMustContain` assertions); this reconstruction's job is to make sure every one of the
eight quantities S5's scorer computes has real turns to be computed over, so that a report the
known-answer validation produces never reads `n/a` on a count the pack could have given it data
for.

### 3.2 The twelve scripts, by shape — turn outline, not literal prose

Full customer-facing wording is content authoring left to the implementer (§4 Step 2) — English
prose that reads as a real customer, not a test fixture, is not something this document should
pre-write and have the implementer merely retype. What follows is the **structural** outline each
script must conform to: turn count, per-turn tool requirement, and the FR-8/boundary focus that
turn exists to exercise. `A-01` and `B-01` are the two **determinism-probe scripts**
(`sampling.determinismProbeScripts`, §3.4) and are therefore the two scripts held closest to
§8.1's own sequence — maximum fidelity to a previously-characterized pattern is exactly what you
want in the two scripts whose *second run* has to reproduce the *first* almost by definition.

**Shape A — 9 turns, read-only.**

| Script | Turns 1-9 outline | Primary coverage this script adds |
|---|---|---|
| **A-01** (determinism probe) | (1) exact-name lookup of a mid-priced item; (2) category filter; (3) price-range filter near the `maxPrice` boundary (`confusedWith: [4999, 5000]`); (4) a rephrased repeat of turn 3's same filter; (5)-(6) an abstention pair — two products not in `catalog.json`; (7)-(9) the turn-1 lookup repeated/rephrased three times | §8.1 shape-A fidelity (determinism baseline); (c)/(d)/(g) via turns 1,7-9; maxPrice boundary via turns 3-4; cross-turn duplicate potential via turns 7-9 and the turn-3/4 repeat |
| A-02 | (1) category filter (Electronics); (2) exact-name lookup; (3) `minPrice` boundary filter (`confusedWith: [10.01]`, `notebook-set` at exactly $10.00 sits on the line); (4) compound category+`maxPrice` filter; (5) exact-name lookup of the zero-stock item (`4k-webcam`, still a valid fact-bearing price lookup); (6) **restraint** — a courtesy reply needing no tool ("that's exactly what I needed, thanks!"); (7) exact-name lookup; (8) category filter (Kitchen); (9) **restraint** — a second courtesy turn | minPrice boundary; two independent restraint turns; a zero-stock item's price fact |
| A-03 | (1)-(2) two boundary-adjacent price filters (one just under `maxPrice`'s confusable value, one just over `minPrice`'s); (3) exact-name lookup with an explicit `finalReplyMustContain` price assertion; (4) a "which is cheaper, X or Y" compound turn — one tool, two distinguishable calls, both required-tool-correct but a real opportunity for a spurious/duplicate miscount; (5) abstention (a product not in the catalog); (6)-(7) two ordinary category filters across different categories; (8) exact-name lookup; (9) exact-name lookup with `finalReplyMustNotContain` guarding against a fabricated price on a near-miss product name | both `boundaryRule` arguments stressed together; the compare-two-products spurious/duplicate opportunity; explicit fabrication guard via `finalReplyMustNotContain` |
| A-04 | (1) category filter; (2) exact-name lookup; (3) **restraint** — an in-conversation callback ("you already told me that, thanks") after turn 2; (4) exact-name lookup of a different item; (5) price-range filter; (6) exact-name lookup with `finalReplyMustContain`; (7) abstention; (8) exact-name lookup; (9) exact-name lookup, rephrased repeat of turn 8 | a third and fourth restraint instance (diversifying restraint's own phrasing beyond A-02's); more (g) coverage; a second cross-turn-duplicate opportunity distinct from A-01's |

**Shape B — 7 turns, write-mutating.**

| Script | Turns 1-7 outline | Primary coverage this script adds |
|---|---|---|
| **B-01** (determinism probe) | (1) add ×2 of item X; (2) add ×1 of item Y; (3) `view_cart` (`finalReplyMustContain` the running total); (4) remove ×1 of item X; (5) remove item Y entirely (omit `quantity`); (6) `view_cart` (must reflect Y's removal); (7) add ×1 of item Z | §8.1 shape-B fidelity (determinism baseline); full add/remove/view cycle; `remove_from_cart`'s two calling conventions (partial-quantity vs. whole-line) |
| B-02 | (1)-(2) add two different items; (3) `view_cart`; (4) a near-repeat instruction on the SAME item already added ("also add 1 more of the first thing") — a direct, deliberate re-probe of §8.4's documented ministral duplicate-instruction defect (§3.7); (5) `view_cart`; (6) `place_order` (`finalReplyMustContain` an order-confirmation phrase); (7) `view_cart` confirming the post-order empty cart | the duplicate-instruction defect's own most literal re-creation; `place_order`'s cart-clearing contract exercised and asserted |
| B-03 | (1)-(2) add two items; (3) attempt to remove an item never added (a customer mistake, not a harness one — a real "remove the thing I didn't add" utterance); (4) `view_cart`; (5) `clear_cart`; (6) `view_cart` confirming empty; (7) **restraint** — "that's everything, thanks!" | a genuine customer-error turn distinct from a harness-induced one; `clear_cart` exercised; a fifth restraint instance, this time inside a write-mutating shape |
| B-04 | (1) category filter (read-only, opening the write-mutating shape with a lookup — realistic pre-purchase browsing); (2)-(3) add two filtered items; (4) `view_cart`; (5) `place_order`; (6) **restraint** — a closing courtesy turn after a completed purchase (the natural place to check the model does not keep calling `view_cart`/`place_order` once done); (7) a genuinely new request, a fresh add, confirming the conversation is not actually over | mixes a read-only turn into a write-mutating shape (an FR-8(c) cross-check: does `right_tool_chosen` correctly separate the two tool families within one conversation); a sixth restraint instance placed at the natural "done" point |

**Shape C — 4 turns, read-only, short.**

| Script | Turns 1-4 outline | Primary coverage this script adds |
|---|---|---|
| C-01 | (1) exact-name lookup; (2) category filter; (3) price-range filter; (4) exact-name lookup with `finalReplyMustContain` | the direct §8.1 shape-C replication — does turn 4's onset point (§8.2) recur at this pack's own turn 4 regardless of shape |
| C-02 | (1) **restraint** — an opening greeting/small-talk turn needing no tool; (2)-(4) three ordinary tool-required turns | tests whether the turn-4 onset (§8.2: "not correlated with what the turn asks, only with its position") survives when position 1 was never tool-bearing at all — a genuine probe of the position-not-content finding, not a contrived variant |
| C-03 | (1)-(2) two boundary-adjacent price filters (one per `boundaryRule` argument); (3) exact-name lookup; (4) exact-name lookup, rephrased repeat of turn 3 | the pack's third and final pairing of both boundary arguments inside one short script; a third cross-turn-duplicate opportunity |
| C-04 | (1)-(2) two abstentions (products not in the catalog); (3)-(4) two ordinary exact-name lookups, both `finalReplyMustContain`-bearing | abstention-heavy — does a model correctly abstain twice in a row before any tool succeeds, and does turn 4 still show the documented onset |

### 3.3 The coverage matrix, derived from §3.2's own table — not a second, hand-maintained list

Every cell below is read off §3.2's per-script outlines directly (a script column and a coverage
column that already state it); this table only regroups the same facts by FR-8 letter so the
"every count has a non-empty denominator" claim in §3.1 is checkable in one place, not
re-asserted:

| FR-8 letter / count | Scripts with a real turn for it |
|---|---|
| (a)+(b) native/prose/no-attempt partition | every required-call turn in all 12 scripts (the partition's denominator is `R(t) >= 1`, present throughout) |
| (c) right tool chosen | every required-call turn where the model actually dispatches something — A-04/B-04 additionally cross-check tool-family separation within one script |
| (d) argument correctness + boundary/unit | every turn with an `args` block; boundary/unit specifically: A-01 (t3-4), A-02 (t3-4), A-03 (t1-2), C-03 (t1-2) |
| (e) spurious/duplicate | cross-turn duplicate opportunities: A-01 (t7-9, t3-4), A-04 (t8-9), C-03 (t3-4); the ministral-pattern re-probe: B-02 (t1, t4) — corrected 2026-09-16 from an earlier `(t3-4)` citation naming turn 3 (`view_cart`, uninvolved in the duplicate-signature check) instead of turn 1 (the re-probe's own baseline call), per `model-bench/docs/reviews/small-model-benchmarking-s6-precheck.md` Finding 3; the compare-two-products spurious risk: A-03 (t4) |
| (f) stopping when done | every turn with `dispatched_count >= 1` (the denominator is any dispatched call, present throughout) |
| (g) final reply matches tool | every turn declaring `finalReplyMustContain`/`finalReplyMustNotContain` — at least one per script by design (§3.2's tables name at least one per row) |
| restraint | A-02 (t6, t9), A-04 (t3), B-03 (t7), B-04 (t6), C-02 (t1) — six restraint turns total, spanning all three shapes |
| `minPrice` boundary (`confusedWith: [10.01]`) | A-02 (t3), A-03 (t2 or t1), C-03 |
| `maxPrice` boundary (`confusedWith: [4999, 5000]`) | A-01 (t3-4), A-03, C-03 |

Restraint's six turns (zero in §8.1's own method) is the largest single addition this
reconstruction makes over the source material, which is exactly the gap plan step 2 names by name
("turns where **no** tool is required (the restraint count)," `:5888-5889`).

### 3.4 Determinism-probe scripts: keeping the placeholder ids, not inventing new ones

`pack.json`'s `sampling.determinismProbeScripts` already declares `["A-01", "B-01"]` — a
placeholder S5 spec §7 flagged explicitly as "this document's own guess... confirm with whoever
plans/executes S6 that `pack.json`'s declared ids are updated to match the real authored scripts."
**Resolved here: keep them.** §3.2 designs A-01 and B-01 as the two scripts held closest to §8.1's
own sequence precisely so they double as the most reliable determinism-probe candidates — the
long shape and the write-mutating shape, "where non-determinism is most likely to bite" (`-ml`
§4.5.1(iii)'s own reasoning for choosing one of each) — and naming them `A-01`/`B-01` from the
start means the placeholder never needs a second edit. No other pairing was considered: the plan's
own criterion (one shape-A, one shape-B) and this document's own fidelity criterion (probe the two
scripts least likely to have been *authored* with any variance built in) point at the same two
scripts for the same reason.

### 3.5 `PROVENANCE.md` — authored content, not copied content, so the sibling packs' table shape does not fit as-is

The three existing packs' `PROVENANCE.md` files (`nlq-structured-query`, `guard-judge-
understanding`, `embedder-graphrag-retrieval`) all record a **copy**: an origin file elsewhere in
the monorepo, a destination path, a source git SHA, a source SHA-256, a copy timestamp. This
pack's `conversations.jsonl`/`prose_calibration.jsonl` are **authored**, not copied — there is no
single origin file whose hash this pack's own data is a function of; §8.1's method is a *pattern*
this reconstruction follows, not a file it transcribes (§3.1). `PROVENANCE.md` therefore uses the
shape the plan's own per-row example already establishes (plan `:2193`: `{"draftedBy",
"verifiedBy", "basedOn"}`), promoted to a pack-level summary table:

```markdown
# Provenance — tool-caller-shop-assistant

> **Pack version:** 0.2.0 · **Generated:** <ISO-8601 timestamp, filled at authoring time>

`conversations.jsonl` and `prose_calibration.jsonl` are **authored content, not copied**: no
single origin file in this monorepo is their source. The 12 conversation scripts reconstruct the
turn-count/turn-type patterns of `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md`
§8.1's three conditions (A/B/C), re-expressed against this pack's own `catalog.json`/
`tools/schemas.json` — never a verbatim transcription of §8.1's own falkor-chat-specific product
names or prices, which do not exist here. See `docs/plans/small-model-benchmarking-s6-spec.md`
for the reconstruction method and the coverage matrix.

| scriptId | shape | draftedBy | basedOn | verifiedBy | verifiedAt |
|---|---|---|---|---|---|
| A-01 | A | <drafting agent's own identity> | §8.1 condition A (turn-by-turn pattern) | <stakeholder name, filled after live stakeholder verification> | <date> |
| … | … | … | … | … | … |

`prose_calibration.jsonl`'s ~20 labelled replies are hand-authored calibration data for
`scoring/toolcalls.py`'s `detect_prose_pseudo_call` heuristic — not derived from any live model
run, and not subject to FR-19's `verifiedBy` step (FR-19 governs the 12 conversation scripts'
`expect` blocks specifically; the calibration corpus's correctness is that its own labels are
true, checked by the same agent pre-check step that reviews the conversation scripts).
```

`packVersion` bumps `0.1.0 -> 0.2.0` when `conversations.jsonl`/`prose_calibration.jsonl` first
land (§4 Step 2-3 — a content-hash-changing edit to a pack that has shipped no run under its prior
version yet, so this is the pack's first real version rather than a revision of a published one)
and does **not** bump again when `provenance.verifiedBy` is filled per row at §4 Step 6 — filling
a previously-empty field is not a second content event worth a second version number, and
`content_hash` already excludes `PROVENANCE.md`, so the pack-level summary table's own edits never
move the hash either. `conversations.jsonl`'s **per-row** `provenance.verifiedBy` transition from
empty to filled **does** change `content_hash` (it is inside the hashed file), which is exactly
the record `AC-3`/FR-6 need — a run driven before stakeholder sign-off and a run driven after are,
correctly, two different pack identities.

### 3.6 Two distinct live-validation obligations, not one — the determinism probe is not item 19a

Three separate mechanisms, none interchangeable with either of the other two, and this document
keeps them separate throughout rather than treating any two as the same obligation:

1. **The §3.8.4 determinism probe** (plan step 4, `:5895-5897`; `-ml` §4.5.1(iii)) — already fully
   built and unit-tested by S5 (§2.1 above): **inside every single live run** of this pack,
   `_drive_conversations` automatically drives `A-01`/`B-01` a second time after the 12 scored
   scripts and records `determinismProbe`/`basis` on that run. It is not a separate action S6
   schedules — it happens as a side effect of any run, and needs no new code or new step of its
   own; §4 Step 7 below only needs to *confirm* it fired and recorded honestly on every run made.
2. **Item 19a — the negative control** (plan `:6300-6304`, owed by S6 per the stage-ownership
   table, plan `:5981`): **two independent, fresh runs of the *same* model**, compared against
   each other. This is a *different* pair of runs from the determinism probe's two scripts —
   19a's "two independent runs" means two full, separately-invoked runs of the *whole 12-script
   pack*, not two scripts inside one run. The plan is explicit that this is not the S1 smoke check
   ("not two copies of one record") and names it "the highest-value single test in the harness"
   because it catches harness bugs that would otherwise masquerade as a real model difference
   (`-ml` §9). **It must actually pass** (report `not distinguishable`, discordant counts roughly
   equal, the difference interval centred on zero) — the plan's own text for 19a carries no
   leniency on this point (`:6309`: "(a) must pass before (b) is interpreted at all"), unlike the
   determinism probe's own done-condition (§4 S6, plan `:5901-5906`), which accepts either outcome
   as long as it ran. The two obligations look similar (both are "run it twice and compare") and
   are not: this section states the difference explicitly so neither is mistaken for the other at
   implementation time.
3. **Item 19b — the known-answer validation** (plan step 5, `:5898-5899`; plan `:6305-6309`):
   `qwen/qwen3-4b-2507` vs. `mistralai/ministral-3-3b`, compared against §8.2's recorded finding.
   **Interpreted only once 19a has passed** — per the plan's own explicit ordering, not this
   document's invention.

§4 Steps 7-8 below run three live conversations-pack invocations in total — two of
`qwen/qwen3-4b-2507` (19a's negative control) and one of `mistralai/ministral-3-3b` — rather than
the two a literal reading of "run both models" might suggest, because 19a's own obligation is
additional to, not satisfied by, running each model once.

### 3.7 Known-answer validation: what "the same contrast" means for this pack, concretely

§8.2/§8.4 (read in full, `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md`) give two
concrete, checkable targets for item 19b, both restated here rather than left as "does it look
similar":

1. **`qwen/qwen3-4b-2507`: a near-total, position-locked collapse at turn 4.** §8.2's finding —
   39/40 conversations skip every domain tool call from the 4th user turn onward, independent of
   turn content, tool type, or shape — predicts this pack's own per-turn-position table shows
   `native`/`rightToolChosen` rates falling sharply at turn-position 4 (this pack's turns are
   0-indexed internally, so "the 4th user turn" is `TurnPositionRate.turnIndex == 3`) across A, B
   and C alike, and `cleanThroughTurn4` at or near 0% for this model. **Comfortably resolvable at
   this pack's own sizing** (`-ml` §4.5.3: `b=12, c=0` gives McNemar exact `p = 0.00049` against a
   model with no such collapse) — the pack "still does the job it was commissioned for" at this
   magnitude, per the note's own honest-consequence table.
2. **`mistralai/ministral-3-3b`: no turn-4 collapse, but its own duplicate-instruction defect at
   a rate this pack's floor cannot resolve.** §8.4's finding — 0/176 skip-and-fabricate instances,
   but 3/10 condition-B conversations show a duplicate re-issue of an already-completed
   `add_to_cart` call when a follow-up instruction references a related item. B-02's turn 4 (§3.2)
   is authored specifically to re-create that exact instruction shape. **This magnitude (~30 pp,
   `-ml` §4.5.3) sits below this pack's own 50.0 pp observable floor at n=12** — the honest,
   pre-stated expectation is that the report will say *"not distinguishable at this sample size"*
   on the duplicate-call rate even if the defect is genuinely present, and the report must say so
   in those words rather than let a `not distinguishable` verdict read as `absent`. This is stated
   here so nobody is surprised by it at report time and, worse, tempted to treat it as a harness
   defect.

**What counts as "the contrast appearing," precisely, and R-3's fallback if it does not.** The
contrast is `cleanThroughTurn4` (the verdict metric) returning `distinguishable`, ministral ahead,
decided by McNemar exact (this design's `basis` is `by-construction` whenever the determinism
probe both ran and matched, §2.1) — a near-certain outcome given target 1's own computed `p`. **If
it does not appear**, §2.4's `historyReplay` correction having already moved the pack to the
plan's own canonical baseline rung, the plan's own fallback applies as written: R-3's bisect is
executed (moving `historyReplay` to `structured`, then to `plaintext`, one axis at a time, per
plan `:6389-6392`) and its result recorded, and the pack ships flagged `known-answer validation:
not reproduced` in every report it generates until it does. **This stage does not edit any
script's `expect` block in response to a non-reproduction** — plan `:5913-5918`'s own warning
against "fitting the instrument to the expected answer" is the one thing this document treats as
non-negotiable regardless of how the live run comes out.

### 3.8 `prose_calibration.jsonl` — ~20 labelled replies, split by what they calibrate

`detect_prose_pseudo_call` (`toolcalls.py:148-165`) only ever runs on a turn's reply text when
`dispatched_count == 0` (`emission_form`'s own precondition) — so every calibration entry is a
**reply-text-only** string standing in for "what a model said on a turn where it made no native
tool call," labelled with whether that text was actually describing a tool call in prose
(`isPseudoCall: true`) or was an ordinary reply with no such intent (`false`). The detector's own
three regex families (`_PROSE_CALL_PATTERNS`, `toolcalls.py:148-155`) fix what the corpus needs to
cover for a real precision/recall figure rather than a vacuous one:

| Category | Count | Label | What it tests |
|---|---|---|---|
| Function-call-shaped fragment (`add_to_cart("Pad"`-style) | 4 | `true` | the first regex family, straightforwardly |
| `"tool_call":` JSON-key-shaped text | 3 | `true` | the second regex family |
| First-person "I'm calling/going to call/invoking" announcement | 3 | `true` | the third regex family |
| Ordinary conversational reply, no call-shaped syntax anywhere | 6 | `false` | the detector's baseline specificity |
| Near-miss text — parentheses/quotes present but not call-shaped (e.g. a price stated
  parenthetically, a quoted product name) | 4 | `false` | precision under text that could plausibly trip a looser pattern than the one actually shipped |

20 entries total, 10 positive / 10 negative — enough for `prose_detector_precision_recall`'s own
precision/recall to be a real fraction (not `0/0`) on every one of the five categories' sub-counts
while staying small enough to hand-author and hand-verify in one sitting, matching the plan's own
"~20" sizing. Row shape: `{"text": "...", "isPseudoCall": true|false}` — deliberately not
`Turn.expect`'s shape, since this file is never read as a `Conversation`; `Pack.
iter_prose_calibration()` (§4 Step 1) reads it directly into the `Sequence[tuple[str, bool]]`
`prose_detector_precision_recall` takes.

## 4. File/module layout

| File | New/changed | Owner in this stage |
|---|---|---|
| `modelbench/packs.py` | changed | `_clean_through_turn_h_problems` (§2.3), wired into `validate_pack`; `Pack.iter_prose_calibration()` (§2.5, §3.8) |
| `modelbench/results.py` | changed | `ToolCallAggregates.prosePseudoCallDetector: Mapping[str, Any] \| None` (§2.5) |
| `modelbench/scoring/toolcalls.py` | changed | `score_conversations` calls `prose_detector_precision_recall` against the pack's calibration data when declared (§2.5) |
| `modelbench/report.py` | changed | one new line printing the detector's precision/recall (or its absence) beside the funnel table (§2.5) |
| `packs/tool-caller-shop-assistant/pack.json` | changed | the three-field correction (§2.4); `packVersion` `0.1.0 -> 0.2.0` (§3.5) |
| `packs/tool-caller-shop-assistant/conversations.jsonl` | new | the 12 scripts (§3.2) |
| `packs/tool-caller-shop-assistant/prose_calibration.jsonl` | new | the ~20 labelled replies (§3.8) |
| `packs/tool-caller-shop-assistant/PROVENANCE.md` | new | §3.5 |
| `tests/test_packs.py` | changed | fixture pack + tests for the new `H <= min(script length)` check; a fixture + test for `iter_prose_calibration()` |
| `tests/test_scoring_toolcalls.py` | changed | a synthetic-fixture test for the new `prosePseudoCallDetector` wiring (calibration data present / absent) |
| `tests/test_results.py` | changed | round-trip test for `ToolCallAggregates.prosePseudoCallDetector` |
| `tests/test_report.py` | changed | renderer test for the new precision/recall line |
| `model-bench/docs/test-reports/small-model-benchmarking-s6-report.md` | new | the live-run record: determinism probe outcome, known-answer validation outcome, `basis` (§4 Step 8) |

## 5. Step sequence

S6 is content-authoring- and process-heavy rather than code-heavy (per the brief's own framing),
so the steps below mix small offline code steps, content-authoring steps, the FR-19 process steps
(§2.6), and the two live steps — sequenced so every later step builds on an already-verified
earlier one, and so the two live steps (which cost real LM Studio time and cannot be casually
re-run) come last, after everything that can be checked offline has been.

### Step 0 — The two code gaps + the manifest correction (offline)

`_clean_through_turn_h_problems` (§2.3): a new `validate_pack`-scoped function mirroring
`_answerability_stamp_problems`'s own shape (`packs.py:867-885`) — scoped to a pack whose manifest
declares both `sampling.scripts` and `metrics.cleanThroughTurnH.H` (absent either key is not this
function's problem, same convention as every other optional-block check in `validate_pack`), reads
every script via `pack.iter_scripts()`, and reports a problem naming the pack, the declared `H`,
and the offending script's own turn count when `H > min(len(s.turns) for s in pack.iter_scripts())`.
Wired into `validate_pack`'s call sequence (`packs.py:918-925`). New fixture pack under
`tests/fixtures/packs/` (a `tool-caller`-shaped fixture with one script shorter than the declared
`H`) plus its positive/negative test pair, mirroring `tests/fixtures/packs/row_count_violation/`'s
own precedent. `Pack.iter_prose_calibration()` (§2.5, §3.8): a small reader mirroring
`iter_items`/`iter_scripts` (`packs.py:293-325`) — yields `(text, isPseudoCall)` tuples from
`data.prosePseudoCallCalibration`'s JSONL rows, raising `PackConfigError` on a missing manifest
key exactly as `data_path` already does. Then the `pack.json` manifest correction (§2.4):
`historyReplay` `"structured"` -> `"structured-replies-only"`, `representToolSchemasEachTurn`
`false` -> `true`, `maxTokens` `512` -> `1024`; `packVersion` unchanged at this point (the version
bump belongs to Step 2-3, when `conversations.jsonl` first lands, per §3.5). **Done when:**
`pytest -q` is green with the new fixture/tests added, `ruff check .` is clean, and
`_clean_through_turn_h_problems`'s own two directions are each covered by a test (an `H` within
every script's length passes; an `H` exceeding the shortest script's length is refused, named by
scriptId and length in the message).

### Step 1 — Wire the prose-detector's precision/recall into the scorer and the report (offline)

`ToolCallAggregates.prosePseudoCallDetector: Mapping[str, Any] | None` (`results.py`, mirroring
`determinismProbe`'s field shape and generic `_encode`/`_decode` treatment — no special case
needed, same reasoning S5 spec §3.1 already gave for `determinismProbe`/`iterationSummary`).
`score_conversations` (`toolcalls.py`) calls `pack.iter_prose_calibration()` guarded by
`"prosePseudoCallCalibration" in pack.manifest.get("data", {})` (never a bare
`try`/`except PackConfigError`, matching `_determinism_probe`'s own `.get(..., ())` idiom rather
than exception flow for an expected-absent case), passes the result to
`prose_detector_precision_recall`, and stores `{"n": len(labelled), "precision": p, "recall": r}`
or `None` when the manifest declares no calibration key at all. One new `report.py` line, placed
beside the funnel table's existing (a)+(b) partition rendering, printing `"prose-pseudo-call
detector: precision X.XXX, recall X.XXX (n=N calibration replies)"` or `"prose-pseudo-call
detector: no calibration corpus declared — precision/recall unmeasured"` when `None`. **Done
when:** a synthetic-fixture test in `test_scoring_toolcalls.py` shows the field populated when a
duck-typed pack fixture declares calibration data and `None` when it does not; a round-trip test
in `test_results.py`; a renderer test in `test_report.py` covering both the populated and the
`None` case.

### Step 2 — Author `conversations.jsonl`'s 12 scripts (content authoring, offline)

Per §3.2's outline tables, against the real, on-disk `catalog.json`/`tools/schemas.json`. Every
turn's `expect` block conforms to §2.2's schema exactly (`toolRequired`, `tool`, `args`,
`finalReplyMustContain`/`finalReplyMustNotContain` — never `argChecks`). Every row's `provenance`
object carries `draftedBy` (the authoring agent's own identity) and `basedOn` (the §8.1 condition
and turn range this script's pattern derives from, per §3.5's `PROVENANCE.md` table);
`verifiedBy` is left **empty** at this step — it is filled only at Step 6, never before. Natural,
plausible customer English for every `user` field — a script that reads as a test fixture rather
than a customer defeats the point of a live known-answer validation. **Done when:** all 12 rows
parse via `Pack.iter_scripts()` with no error, `sampling.scripts: 12` and the row-count identity
hold (checked structurally at Step 4), and a hand cross-check against §3.3's coverage matrix
confirms every named cell has a real turn behind it.

### Step 3 — Author `prose_calibration.jsonl` + update `pack.json` + write `PROVENANCE.md` (content authoring, offline)

The ~20 labelled replies per §3.8's category table. `pack.json`: confirm
`sampling.determinismProbeScripts` still reads `["A-01", "B-01"]` (§3.4 — no change expected,
verify rather than assume) and bump `packVersion` `0.1.0 -> 0.2.0` (§3.5). `PROVENANCE.md` per
§3.5's shape, `verifiedAt`/`verifiedBy` columns left blank. **Done when:** `prose_calibration.jsonl`
has exactly the category counts §3.8 names (a hand count, not a claim), `pack.json` parses and its
`packVersion` reads `0.2.0`, and `PROVENANCE.md` names all 12 scripts by id with a `basedOn` entry
for each.

### Step 4 — `validate_pack` on the real, on-disk pack (offline)

The whole pack, assembled by Steps 0-3, run through `validate_pack` for real (not a fixture) —
mirroring S5 spec's own TP-004 precedent (U148's QA acceptance, §2.4 above) of checking every axis
against the real on-disk artifact rather than only in-memory fixtures. **Done when:**
`validate_pack(load_pack(Path("packs/tool-caller-shop-assistant")))` returns `[]` — every one of
`validate_pack`'s eight axes (the original seven plus Step 0's new one) passes clean, including
the row-count identity (12 distinct `scriptId`s, `sampling.scripts == 12`,
`replicatesPerScript == 1`) and the new `H <= min(script length)` check (`H = 4`, every script has
at least 4 turns by construction — shape C's own 4-turn scripts are the binding case).

### Step 5 — Agent pre-check (FR-19 process step 1, §2.6)

Dispatched to an agent that did **not** author the scripts in Step 2 (this document's own routing
note, §2.6 item 2) — reads every one of the 12 scripts' `expect` blocks against `tools/sim.py`'s
actual dispatch logic and `catalog.json`'s actual data directly (running the simulated environment
against each scripted turn, not reasoning about what a model might do), and reports any mismatch
between a scripted `expect` and what the real environment actually returns for that turn: a wrong
expected price, a category filter whose expected result set is wrong, an `args` block that does
not match what `tools/sim.py` actually requires, a `finalReplyMustContain` assertion that could
never be literally true given the tool's real return shape. **Not a substitute for stakeholder
review** (§2.6's own `data-scientist`-argued reason) — its job is to cut the stakeholder's review
time by catching mechanical authoring errors first, never to fill `verifiedBy`. **Done when:**
every one of the 12 scripts has a pre-check disposition (clean, or a named issue) and every named
issue is fixed and re-checked before Step 6 begins.

### Step 6 — Stakeholder verification (FR-19 process step 2, binding, §2.6)

The stakeholder personally executes/verifies all 12 scripts (not a sample) against the real
simulated environment's actual behavior. This step is outside any agent's authority to perform or
simulate — it is recorded here as a step this coordination dispatches to the stakeholder, not one
an agent executes on the stakeholder's behalf. `provenance.verifiedBy`/`verifiedAt` are filled per
row in `conversations.jsonl` (moving `content_hash`, per §3.5) and mirrored into `PROVENANCE.md`'s
own table only after every row is filled. **Done when:** all 12 rows carry a non-empty
`verifiedBy`, and no row is filled by anyone other than the stakeholder.

### Step 7 — Item 19a's negative control, live: two independent runs of the same model

Three live invocations of `model-bench run` against the now-complete pack, not two — §3.6's own
correction is what fixes this step's scope: **two independent runs of `qwen/qwen3-4b-2507`**
(`qwen-run-1`, `qwen-run-2` — freshly invoked processes, never one record copied) for item 19a's
negative control, plus **one run of `mistralai/ministral-3-3b`** (`ministral-run-1`), staged here
because it costs nothing to run alongside the second `qwen` invocation and Step 8 needs it anyway.
Immediately before each of the three invocations, re-confirm LM Studio residency for the model
about to run (§2.7's caveat: `/v1/models` shows availability, not residency) — the live-run
precondition `_drive_conversations`'s own between-item residency probe already checks per call,
but the **model actually loaded** should be confirmed once per run before it starts, not
discovered mid-run. Every run drives its own 12 scored scripts then `A-01`/`B-01` a second time
(already fully wired, `_drive_conversations`, §2.1 — no code path to build here), which is what
satisfies plan step 4's own "for both models" wording (`qwen` appears via `qwen-run-1`/`-2`,
`ministral` via `ministral-run-1`) without a fourth invocation. Then `model-bench compare
qwen-run-1 qwen-run-2`. **Done when:** `determinismProbe.ran == true` and `basis` is recorded
honestly on all three runs (no manual override of either value — whatever the live probe actually
returns is what is recorded, not what would be convenient for either comparison's statistical
power), **and** the `qwen-run-1`-vs-`qwen-run-2` comparison reports `cleanThroughTurnH: not
distinguishable`, with discordant counts roughly equal and the difference interval centred on
zero. **If it does not pass**, this is recorded as its own finding (a likely harness or
non-determinism defect — the plan names this exact test as the one most likely to catch a bug
that would otherwise present as a real model difference, `-ml` §9) and Step 8's own comparison is
**not** interpreted until the finding is understood, per the plan's explicit ordering (`:6309`).

### Step 8 — Item 19b, the known-answer validation, live + the test report

`model-bench compare qwen-run-1 ministral-run-1` — reusing `qwen-run-1` from Step 7 rather than
invoking a third `qwen` run (a stored run is a reusable artifact across multiple `compare`
invocations, and nothing about `qwen-run-1`'s own identity changes between the two comparisons it
participates in; this avoids one redundant live run at zero cost to either result's validity).
Compare the per-turn-position table and `cleanThroughTurn4` against §3.7's two named targets — but
**only once Step 7 has actually passed** (§3.6/§3.7's own gating rule, restated because it is the
one rule this step must not silently skip under time pressure). Write `model-bench/docs/
test-reports/small-model-benchmarking-s6-report.md`: both comparisons' funnel tables, per-turn-
position tables and hazard curves (per-failure-kind **and** per-turn-position, no blended headline
anywhere, per the plan's own AC-1 requirement — structurally guaranteed by `report.py` already,
not a discipline this step has to enforce by hand), the determinism probe's own recorded outcome
and `basis` on all three runs, item 19a's own verdict (pass/fail, stated plainly), and item 19b's
verdict — **either outcome is a pass** per the plan's own S6 done-condition (§6 below): the
contrast appearing, or R-3's bisect executed and its result recorded with the pack flagged
accordingly. **Done when:** the report exists, is human-readable, states both 19a's and 19b's
results plainly rather than requiring a reader to re-derive them from the raw JSON, and (per
§3.7's B-02 cross-reference) explicitly notes whether the ministral duplicate-instruction rate
landed inside or outside this pack's own stated resolving power rather than leaving a reader to
work that out unaided.

## 6. Test strategy

Per the coordination's stage-ownership table (plan `:5980-5983`), S6 owes numbered items **12**
(the `H <= min(script length)` clause), **16** (the pack's own end-to-end run — S5's synthetic
integration test was explicitly *not* a discharge of this item, S5 spec §6), **17**, and **19**
(**19a**, the negative control, **and** **19b**, the known-answer validation — two genuinely
distinct live obligations, §3.6), with **two separate, non-interchangeable gates layered on top
of each other, both real**: item 19's own text ("(a) must pass before (b) is interpreted at all,"
plan `:6309`) requires 19a to actually **pass** before 19b's own result may be read as meaning
anything; separately, the stage-ownership table's own S6 row states "§4 S6 gates on 19b being
*run and recorded*, never on the contrast appearing" (plan `:5981`) — the **stage's own
done-condition** is satisfied once 19b has been run and its outcome recorded, whatever that
outcome is, per R-3's own fallback (§3.7). These do not conflict: a 19a failure blocks
*interpreting* 19b's result, never blocks *running and recording* it, and the stage can still
close on a run-and-recorded 19b even while flagging 19a's own failure as an open, actionable
finding.

- **Item 12** (the `H <= min(script length)` clause) → Step 0's new `validate_pack` check, plus
  Step 4's real-pack validation run.
- **Item 16** (one full run per pack, end to end) → Step 7/8's three live runs, the first real,
  non-synthetic runs of this pack against a live model — discharged here for the first time in
  this coordination, per the stage-ownership table's own note that S5's offline integration test
  does not count toward it.
- **Item 17** → the pack's own `validate --strict` pass (Step 4) plus the sampling row-count
  identity and `analysisUnit` membership, both already-shipped `validate_pack` machinery (§2.1)
  now exercised against real, 12-script data for the first time.
- **Item 19a** (the negative control) → Step 7: `qwen-run-1` vs. `qwen-run-2`, must report `not
  distinguishable`.
- **Item 19b** (the known-answer validation) → Step 8: `qwen-run-1` vs. `ministral-run-1`,
  interpreted only once 19a has passed, recorded either way.

**Unit-tier additions this stage owns outright** (new code surface, §4 Step 0-1): the
`H <= min(script length)` check's own fixture-pack test pair; `Pack.iter_prose_calibration()`'s
own test; the `prosePseudoCallDetector` field's round-trip test and its scorer-wiring test
(present/absent calibration data, both directions); the new `report.py` renderer line's test
(populated and `None`).

**Acceptance-tier**: this stage's own `qa-engineer` pass (proposed as unit 6 in the dispatch
sequence, "Ready to implement" below) drives `validate --strict` against the real, fully-authored
pack, spot-checks a
sample of the 12 scripts' `expect` blocks against `tools/sim.py` directly (independent of both the
agent pre-check and the stakeholder review — a third, differently-motivated check, following this
coordination's own default-independent-review discipline), and confirms the test-report Step 8
produces states its own verdict in words a reader does not have to re-derive.

## 7. Risks & open questions

- **The `historyReplay`/`representToolSchemasEachTurn`/`maxTokens` manifest correction (§2.4)
  touches a file `analyst`'s S5 code gate (U144) and `qa-engineer`'s S5 acceptance (U148) both
  already reviewed and accepted.** This document resolves the divergence as a design ruling
  (§2.4's own reasoning: unambiguous plan text, mechanically safe, zero live-run history to
  protect) rather than treating it as a stop-and-ask fork, following this coordination's own U146
  precedent for a comparable plan-vs-shipped gap. **Flagged for `teco`'s attention specifically
  because it revises an already-gated file** — a lightweight confirmation that the reasoning holds
  (not a full re-gate: `analyst`/`qa-engineer`'s S5 review scope was "does the shipped code match
  the S5 spec," never "does the S5 spec's own manifest literal match the plan's," so nothing in
  their prior verdicts is actually contradicted by this correction) is recommended before Step 7's
  live run, so the correction is not discovered only after a live run has already used the wrong
  value.
- **The twelve scripts' exact English wording is not written here (§3.2's own scope note)** — the
  structural outline (turn count, tool requirement, FR-8/boundary focus) is fully specified and
  the coverage matrix (§3.3) is derived from it, but the implementer still owes genuine authoring
  judgment in making each turn read as a plausible customer utterance. This is deliberate (§3.2's
  own reasoning) and not a gap: a spec that pre-wrote all 108 turns' literal text would not be
  reviewable as a spec and would not leave the implementer doing authoring work at all.
- **B-02's turn 4 is a deliberate re-probe of §8.4's ministral duplicate-instruction defect, and
  §3.7 already predicts this pack's own floor (50.0 pp at n=12) cannot resolve it at its
  documented ~30 pp magnitude.** This is stated as an expectation, not a target to author around —
  if the live run's own result differs materially from this prediction (either direction), that is
  itself a finding worth a line in Step 8's test report, not a reason to revise B-02 after the
  fact.
- **The known-answer validation's own outcome is genuinely unknown until Step 8 runs.** §3.7
  names both targets and both outcomes (contrast appears / R-3's bisect) as passing conditions per
  the plan's own done-condition — this document does not predict which one occurs, only what each
  one means and what the report must say in either case.
- **`prose_calibration.jsonl`'s category counts (§3.8) are this document's own synthesis** — the
  plan states "~20 labelled replies" and "the detector's own precision and recall" with no
  specified split; the 10/10, five-category breakdown proposed here is a reasonable, checkable
  starting point tied directly to the detector's own three regex families plus two specificity
  checks, and is cheap to widen later (one small JSONL file, no scorer-side consequence) if a
  future reviewer finds it under-sized.
- **The `Pack.iter_prose_calibration()`/`prosePseudoCallDetector` wiring (§2.5, §4 Step 1) is a
  real, previously-unbuilt code gap this document found by reading `toolcalls.py` directly, not a
  gap the plan names explicitly by that description** — the plan's own text ("the report prints
  the detector's own precision and recall") is unambiguous about the *requirement*; this document
  supplies the *mechanism*, following the same discipline S5's own architect used for the five
  gaps it found and resolved rather than left open (S5 spec §2.3-§2.6).

## Ready to implement

Document: `docs/plans/small-model-benchmarking-s6-spec.md` (this file). Nine steps (§5): **Step 0**
(the `H <= min(script length)` `validate_pack` check + `Pack.iter_prose_calibration()` +
`pack.json`'s three-field manifest correction — all offline code, §2.3/§2.4/§2.5); **Step 1**
(wiring the prose-detector's precision/recall into `score_conversations`/`ToolCallAggregates`/
`report.py`, offline, §2.5); **Step 2** (author the 12 conversation scripts per §3.2's outline and
§3.3's coverage matrix, `verifiedBy` left empty); **Step 3** (author the ~20-reply calibration
corpus, confirm the determinism-probe script ids, bump `packVersion`, write `PROVENANCE.md`);
**Step 4** (`validate_pack` on the real, on-disk pack, must return `[]`); **Step 5** (agent
pre-check, non-authoring agent, FR-19 process step 1); **Step 6** (stakeholder verification, FR-19
process step 2, binding — fills `verifiedBy`); **Step 7** (three live runs — two of
`qwen/qwen3-4b-2507`, one of `mistralai/ministral-3-3b` — and item 19a's negative control,
`qwen`-vs-`qwen`, which must actually pass); **Step 8** (item 19b's known-answer validation,
`qwen`-vs-`ministral`, interpreted only once Step 7 passes, plus the test report). Five real gaps
found and resolved as design rulings, not left open: `validate_pack` never checked
`H <= min(script length)` (§2.3); the shipped `pack.json` diverges from the plan's own canonical
manifest on three fields, one of them (`historyReplay`) directly load-bearing for what the
known-answer validation measures (§2.4, flagged for a lightweight `teco` confirmation since it
revises an already-gated file, not treated as a stop-and-ask fork); the prose-pseudo-call
detector's precision/recall is computed nowhere in production code despite the plan requiring it
printed (§2.5); the determinism-probe placeholder script ids (`A-01`/`B-01`) are confirmed correct
by design rather than left for a second guess (§3.4); and item 19a's own negative control is a
distinct live obligation from the §3.8.4 determinism probe, owed in addition to it rather than
satisfied by it (§3.6). No genuinely unresolvable fork was hit — the `historyReplay` divergence
came closest, and is resolved above rather than escalated, for the stated reasons (unambiguous
plan text, zero live-run history, mechanically safe fix).

**Proposed next units, in sequence**, consistent with this coordination's own S4-S8 discipline
(stage spec -> `teco`-verified directly against source, no separate `analyst` plan gate ->
implementation -> `analyst` code gate -> `qa-engineer` acceptance -> stage close):

1. **Steps 0-1 (code)** — `tdd-engineer`, offline, test-first (both are small, well-specified
   additions to already-tested modules with an obvious red/green shape: the `H` check and the
   prose-detector wiring each have a clear failing-then-passing test pair named in §4 Step 0-1's
   own "Done when" clauses).
2. **Steps 2-3 (content authoring)** — `tdd-engineer` or `coder` (either fits; this is content
   authoring against a fixed schema and a fixed coverage matrix, not new production logic, so the
   choice is not load-bearing the way it would be for Steps 0-1). Sequenced after Steps 0-1 land,
   since Step 4's `validate_pack` run needs Step 0's new check to exist to be meaningful.
3. **Step 4 (validation)** — the same agent as Steps 2-3, immediately after, since it is a direct
   continuation of the same authoring unit (confirm the pack it just wrote actually validates).
4. **Step 5 (agent pre-check)** — a **fresh**, non-authoring agent (this document's own routing
   requirement, §2.6/§4 Step 5) — `analyst` is a reasonable choice (its own standing role is
   exactly "checks a deliverable it did not produce"), though any agent not carrying Steps 2-3's
   context would satisfy the independence requirement.
5. **Step 6 (stakeholder verification)** — dispatched by `teco` to the stakeholder directly, not
   an agent unit. Blocks Steps 7-8 until every one of the 12 rows carries a non-empty
   `verifiedBy`.
6. **Steps 7-8 (the three live runs, item 19a's negative control, item 19b's known-answer
   validation, and the report)** — `qa-engineer`, once Step 6 clears (`qa-engineer` already owns
   the acceptance-tier check named in §6 above, and running the three live arms plus writing the
   human-readable test report is a natural extension of that same pass rather than a separate
   unit).
7. **Stage close** — `analyst`'s code gate over Steps 0-1's production-code diff (small: two
   modules, roughly the size of a single S5 fix-round unit), then the doc-sync unit mirroring
   S4/S5's own `AGENTS.md`/`README.md`/`HISTORY.md` closing precedent.

