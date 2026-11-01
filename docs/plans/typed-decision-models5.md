# Typed Decision Models — Implementation Plan, Seed Identity and Method-Note Amendments

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.5 · **Extends:** `docs/plans/typed-decision-models4.md` · **Reviews:** `docs/reviews/typed-decision-models5.md`

*2026-11-01 — v1.5: review Pass 2: halts count every reason, vector re-copy rule, single-write decision, Unicode boundaries.*

## 1. Goal & scope

Amends plan4 v1.2 for P20-m2 (`docs/reviews/typed-decision-models-impl.md` Pass 20), P3-F1,
P3-F6, P3-F7, P4-F1, P4-F5 (`docs/reviews/typed-decision-models4-impl.md`), ML review Pass 1 m5,
P11-M2 and the method note's §Q1.1–§Q1.3, cited unversioned. The rest of plan4 stands.

## 2. Context & findings

- **The note.** §Q1.1 adds form obedience; §Q1.2 withholds `*Id` axes and widens "Hits"; §Q1.3
  "Pilot" holds the cue check, recheck block and reprint. U5.2b's conformance comes from
  `docs/reviews/typed-decision-models-ml.md` Pass 1 m5.
- **`points.seed_binding_errors`** returns `BindingError(binding, message)` per failure.
- **U4 is U4.1 → U4.2 → U4.3** (coordination ledger). Only U4.1 touches renderers and contract
  vectors; its new `input-hash.jsonl` rows redden decision-models' byte-copy test.
- **The `prime` call.** `runner._drive_single_call_items` calls `prime` unguarded; on a cache miss
  its `lmstudio.embed` can raise `LMStudioCallTimeout`/`LMStudioCallFailed`, which `cli` does not
  catch. `_warm_up` maps both to exit 3. U7's drivers never reach `prime`.

CPG: considered, not relevant — no `cpg_modelbench` or `cpg_decisionmodels` graph is loaded; the call paths were read directly.

## 3. Design amendments

### 3.1 The embedder `prime` corpus embed (P20-m2; plan4 §3.10, R-21)

**Ruling: `prime` gets its own guard and unit, U2.4, outside R-21.**
- **Why:** R-21 covers item calls, which end as scored or kept failures; `prime` runs before any
  item and the run cannot continue without it, so it takes the warm-up rule.
- **The guard.** `runner._drive_single_call_items` wraps its `prime(...)` call.
  - `LMStudioCallTimeout` and `LMStudioCallFailed` raise `RunRefused(exitCode=3)` with
    `from exc`. The message names the corpus embed and ends with "nothing was scored".
  - Other exceptions pass through. The catch is the runner's: scorers do not know `RunRefused`.
- **No re-probe** (both cases exit 3). **Not a U7 prerequisite.**
- **Plan4 R-21, corrected.** Its second sentence now reads: "A dead server during an item call
  gives `timeout`/`no_response` scored as model failures, or kept failures on the decision
  surfaces; no item call ends in a traceback."

### 3.2 Seed identity across draws (P3-F1; plan4 §3.5, §3.6)

**Ruling: the proposed owner and mechanism are right, with one addition.**
- **The owner is the ledger,** the only state that sees every draw; `seeds.py` stays pure. Folding
  `count` into `seeds._seed_id` would still let an identical repeat collide.
- **`campaign.record_draw(draw: SeedDraw)`** appends a `seed-draw` event holding
  `SeedDraw.to_record()`. It refuses, writing nothing and naming the clash, when the draw's
  `rng_seed` or any of its `seedId`s is already held.

  Both: two restricted draws on one RNG seed get distinct ids but correlated streams.
- **`campaign.fresh_rng_seed()`** mints `secrets.randbits(32)`, redrawing while the value is held.
  - Every `draw_seeds` call takes its RNG seed from it and passes `record_draw` before use:
    round 1 (U5.8), each top-up stratum (U5.9), each pilot block and the recheck block (§3.6).
  - A regeneration (§3.5) reuses its seed, so it is not a draw.
- **The addition.** The pilot's store is deleted (plan4 §3.8), so `pilot-report.json` lists every
  pilot RNG seed and `init --phase full` loads them as held: note §Q1.3's "seeds are not reused",
  made mechanical.

### 3.3 What `promptSha256.rubric` hashes (P3-F7; plan4 §3.7)

**Ruling: hash the semantic parts separately.** Each value is sha256 over the UTF-8 encoding of
one `Point` attribute:
- `rubric` ← `Point.rubric_body`;
- `requestTemplate.labeler` ← `Point.label_answer_section`;
- `requestTemplate.generator` ← `Point.generator_instructions`, composed. The generator has no
  separate body key.

**Why:** `runs[].requestSha256` already pins composed bytes, so U5.9's "new rubric hash" fires on
a rubric edit, not a format fix. **Rejected:** `rubric` as composed line 2 (double-counts
`label-answer.md`).

### 3.4 Generation requests, screen and U5.2b (plan4 §3.2, §3.3, §3.6, §4.1, §4.3)

- **Id-blind seed lines** (note §Q1.2 "Blind"; plan4 §3.2).
  - A generation seed line holds `key` plus only the seed axes whose key does not end in `Id`.
  - The writer projects the `*Id` keys out of `Seed.axes`; the store keeps them by request key.
  - `validate_request` refuses any `*Id` key on a generation line, whatever `points/` declares.
- **Screen** (note §Q1.2 "Hits", which plan4 §3.3, §3.6 and §5 U5.8 now cite unversioned).
  - An id-form hit is **token-bounded**: `screen.py` wraps `ID_PATTERN` in `(?<!\w)…(?!\w)`
    (Unicode `\w`), so "pedido_123" and "opções_12" are no hit and "(dom_07)" is.
    `seeds.ID_PATTERN` is unchanged.
  - The six `missingForm` names are in guard's `screen.metaPosition` (`point.json`).
- **U5.2b is split** (P4-F5), amending plan4 §4.3's U5.10 → U5.2b edge.
  - **U5.2b-hash:** the join-in and message-type `hashing.INPUT_FIELDS_BY_ROLE` rows, the
    `input-hash.jsonl` re-copy, `inputFields` equal to those rows as sets, and retiring the P3-F6
    xfails, `PLANNED_INPUT_FIELDS` and its `.get` fallback.
  - **U5.2b-render.** It copies the new renderers, their render vectors and `normalize-turns.jsonl`.
    A fixture rendered through the copied renderer matches the rubric's "What an item looks like"
    block. While join-in's `maxItems` (8) exceeds the 6-turn cut, `points/join-in-judge/generator.md`
    keeps the agent's job inside the last 6 turns, and `data-scientist` confirms it.
    - **Its dependency is U4.1,** which writes every copied vector file (U4.2 edits the
      renderers' module, not the vectors). It runs right after U5.2b-hash.
  - **A later U4.x change to a copied vector** re-copies it in that unit's own integration, with
    decision-models' suite in its gate.
  - **Until U5.2b-hash lands,** every decision-models gate deselects
    `tests/test_decision_contract.py::test_the_vector_copy_is_byte_identical_to_model_benchs[input-hash.jsonl]`
    by node id, and the coordination ledger records it.

### 3.5 Form obedience at generation ingest (note §Q1.1 "Guard `missing` forms"; plan4 §3.6)

- **Schema first.** `points.input_field_errors` stays answer validation (plan4 §3.3's
  out-of-space row): a failing item voids the run as run-attributable, never a `bindingMismatch`.
- **Then** the screen and `points.seed_binding_errors` (against the seed's stored axes) run.
- **One reason per item per pass,** by precedence: screen hit, then `bindingMismatch` (any
  binding error whose binding is not on `missingForm`), then `formMismatch`.
  - **Why:** a screen hit is redrawn regardless; a non-form binding error means the form was
    never attempted.
  - The precedence reason alone takes the seed's one hit toward its 3-hit cap and its `dropped`
    count. The halts instead count the item under every reason it shows.
- **Path, as a screen hit:** a `form-mismatch`/`binding-mismatch` ledger event queues the seed for
  regeneration. A drop counts under the third hit's reason. `dropped` gains
  `formMismatch: {<form>: n}` and `bindingMismatch` (plan4 §3.7's key set).
- **Halt timing** (each of the three reasons has its own `template` halt).
  - **When.** After each first-pass generation run is ingested, it halts once
    `100 · first-pass hits > 5 · drawn seeds` for one reason. Only first-pass hits count;
    regeneration hits count toward the cap only.
  - **Why:** hits only grow and the denominator is fixed, so it decides as end-of-round would, sooner.
  - **A round** here is one first-pass issue: round 1, each top-up round, each pilot draw or
    block, and the recheck block.

### 3.6 The pilot `missing` cue check (note §Q1.3 "Pilot"; plan4 §3.5, §3.8)

- **One function, pure stdlib (FR-8).** `pilot.cue_check(items) -> CueCheck` implements note
  §Q1.3's rule, which its docstring cites.
  - **Input**, per labelled `understanding` item: its stratum (difficulty × intended answer), its
    form class (counter-cue = `immaterial`, `unlisted`, `ambiguous`; otherwise concordant), and
    whether its majority label departs from intent.
  - **`CueCheck`:** per stratum `x`, `m1`, `y`, `m0` and both departure rates, then `Σx`, `p` and
    `failed`.
- **Report.** `pilot.py report` is the one evaluator. It writes
  `missingCueCheck: {state, pilot: CueCheck, recheck: CueCheck | null}`, so both results reach the
  decision package. The states are `passed`, `failed` (it also writes a `cue-check` halt to the
  pilot ledger), `recheck-pending`, `recheck-passed` and `recheck-failed`.
- **On `failed`.**
  1. `driver resolve --decision <text>` records the fix.
  2. `pilot.py recheck` refuses unless the state is `failed` and that resolve exists. It draws the
     note's block through `campaign.record_draw` (§3.2), queues it as pilot generation runs and
     votes, and sets `recheck-pending`. Votes batch at 50, as in the pilot.
  3. Once the driver has ingested every vote of the block, `pilot.py report` evaluates it and sets
     `recheck-passed` or `recheck-failed`. It refuses to evaluate a partly voted block.
  - **`stakeholderDecision` is single-write:** a second `record-decision`, or `recheck` after
    one, refuses. A `stop` at any state is final; a new decision needs a fresh pilot. The store is
    deleted (plan4 §3.8) only after a decision.
- **Recheck cost:** 4 opus generation runs, 6 sonnet label runs (2 × 3 votes), 300 item-votes,
  plus §3.5 regenerations; `report` writes `recheckProjection` from F and c.
- **Spend gate.** This is the one predicate for `init --phase full`, amending plan4 §3.5's "recorded
  stakeholder decision":

  `outcome == "proceed"` and (`state` ∈ {`passed`, `recheck-passed`} or (`state ==
  "recheck-failed"` and `stakeholderRef` is present)).

  `outcome` and `stakeholderRef` come from `stakeholderDecision`, so a recorded `stop` always
  refuses. `record-decision --outcome proceed` refuses outside the predicate (`--stakeholder-ref`
  writes `stakeholderRef`); a first `stop` is always recordable.
- **Round-1 reprint** (U5.9). `diagnostics.json` carries a `CueCheck` over all of guard's round-1
  `understanding` items, exploratory, with `failed` omitted.

## 4. Unit impact

Rows add to plan4 §4.1 and §5.

| Unit | Added done-condition |
|---|---|
| **U2.4 `prime` guard** (new; `tdd-engineer`) | §3.1. A stub `embed` raising each class on a cache miss gives exit 3 and stores nothing. A cache hit is unaffected. A `RuntimeError` from `prime` propagates (an `except Exception` mutant reddens), as does a catch-removed mutant. R-21's BACKLOG item drops its `prime` pointer; `HISTORY.md` gets an entry. |
| U5.3 (fix round) | P3-F6's strict xfails. **Conditional axes:** a multi-axis `when` (guard's `missingForm` given evidence kind, answer and difficulty) and a single-axis one (join-in's cue given the answer) never yield a value outside their row. **Apportionment** (note §Q1.1): expected counts over many RNG seeds at sizes 1–7 equal each quota; a largest-remainder `allocate` mutant reddens. |
| **U5.2b-hash** (new; `coder`) | §3.4. Suite green with no deselection; the P3-F6 xfails and `PLANNED_INPUT_FIELDS` are gone; copies `cmp`-equal. |
| **U5.2b-render** (new; `coder`) | §3.4: the rubric-layout fixture per new-ability point, the copied render and normalization vectors pass, copies `cmp`-equal, and the join-in turn rule is confirmed. |
| U5.4 | §3.4 id-blind seed lines. Positive controls, added to plan4 §5's three: a planted `domainId` on a generation line is refused, and a seed line from the writer holds no `*Id` key. |
| U5.6 | §3.2. Planted cases, ledger byte-equal after each: a held RNG seed refused; a fresh one with a held `seedId` refused. `secrets.randbits` patched to a held then a fresh value makes `fresh_rng_seed` return the fresh one. `init --phase full` loads the pilot's RNG seeds. **Spend gate** (§3.6): one fixture per predicate branch; `stop` with `passed` refuses; `recheck-failed` without the ref refuses; `failed` and `recheck-pending` refuse. |
| U5.8 | Round 1, pilot blocks and the recheck block draw through §3.2; the report's RNG-seed list includes the recheck block's. **Obedience** (§3.5): a mismatch fixture per form (all six) regenerates, and an obedient one passes. The four-error wrong-kind item is one `bindingMismatch`, with no `formMismatch`; a both-counted mutant reddens. Screen beats binding for the cap. 4% screen-plus-form and 4% form-only items halt on `formMismatch`, not screen. A schema failure voids the run. The halt fires after the first run past 5% of drawn seeds; exactly 5% does not. **Cue check** (§3.6): a hand-computed 2-stratum p-value. All counter-cue items following the cue fail; uniform departure passes; a pooled-strata mutant changes p; a constant stratum leaves p unchanged; `m1 = 1`, `m0 = 19`, `x = 1`, `y = 0` gives p = 1/20 and fails (a `<` mutant reddens). `recheck` refuses before `resolve`, issues 4 + 6 runs, and `report` refuses a partly voted block. A `recheck-failed` report keeps the pilot's `CueCheck`. `recheckProjection` matches a hand computation. A second `record-decision`, and `recheck` after `stop`, refuse. **Screen:** `cue_01` is a hit anywhere; "pedido_123" and "opções_12" are not; each `missingForm` name is a hit in a meta position, and "listed" in prose passes. |
| U5.9 | Top-ups draw through §3.2. Two top-up rounds give disjoint `seedId`s, and reusing round 1's RNG seed is refused. The tranche keys on `promptSha256.rubric`. `diagnostics.json` carries the reprint (§3.6). |
| U5.10 | `dropped` carries §3.5's keys; `promptSha256` per §3.3: a `label-answer.md` edit moves only `requestTemplate.labeler`, a `rubric.md` edit only `rubric`. |

**Order** (amends plan4 §4.3): U5.2b-hash, then U5.2b-render, are the next decision-models
units once U4.1 integrates; the chain does not otherwise wait for U4.1. U5.2b-render precedes
`U5c.message-type`. U2.4 follows U3e.2 and gates nothing. U5.4 starts once this plan is gated.

## 5. Ready to implement

Plan: `docs/plans/typed-decision-models5.md`, extending plan4. **Now:** U5.2b-hash after U4.1
integrates; U5.4 builds against §3.4; U5.8 against §3.5–§3.6; U2.4 queues after U3e.2.
