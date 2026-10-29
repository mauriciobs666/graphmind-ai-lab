# Review — `query_graph_data` row identity and category-miss recovery (plan gate)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-079 (F2), K-086

## Pass 1

*2026-10-28. Plan gate of `docs/plans/salesperson-query-graph-data.md` v1.*

### Scope & verdict

**Reviewed:** the plan in full, against `docs/reviews/salesperson-fabricated-prices-rca.md` (§3–§5,
F2 and its Prevention test), `docs/test-reports/salesperson-cart-product-id2-report.md` (DEF-C and
the browse table), `docs/plans/salesperson-cart-product-id2.md` (D8), `server/falkorchat/querygen.py`,
`tools.py` (`QueryGraphDataTool`, `FilterProductsTool`, `_QUERY_REQUEST_INSTRUCTIONS`),
`services.py` (`has_category`, `lookup_product`, `catalog_index`), `executor.py`
(`_GROUNDING_SOURCES`, `_collect_target_values`), `proof_defs.py` (`v9` prompt),
`tests/eval/nlq_scoring.py`, `run_nlq_golden_set_eval.py`, `nlq_golden_set.jsonl`, the test stubs,
and `salesperson/scripts/refire_probe.py`. Baseline: HEAD `03328dc1`; the plan is untracked.

**What I ran** (all read-only; scripts and outputs in the session scratchpad, `qgd/`):
- an independent re-derivation of the §6.1 probe from the registry and `compile()`, under the
  plan's axes and under wider ones (`probe.py`);
- a prototype of D1 plus three plausible implementation mutants, scored against both axis sets
  (`mutants.py`);
- a pair-by-pair check of all 40 golden pairs: 76 completions (each pair's canonical shape plus
  plausible variants) compiled with and without the D1 prototype, run with `GRAPH.RO_QUERY` on
  `reference` / `ws:nlq-eval`, and scored with `score_pair` (`golden_replay.py`);
- live read-only checks on `reference`, `ws:nlq-eval`, `has_category` and `lookup_product`;
- `pytest tests/test_querygen.py tests/test_query_graph_data_routing.py tests/test_tools.py -k
  "querygen or query_graph_data or routing or compile"`: 85 passed. These are stub-only tests with
  no `reference` fixture, and that is the baseline. I did not run the default suite or
  `test_queries.sh`.

**Verdict: needs changes.** There is one blocker. D1's facet exemption still lets the RC-1 shape
through. There are two majors: D7's known/unknown test disagrees with the query it is diagnosing,
and the probe does not cover three axes D1 depends on. Everything else is minor. The design itself
is sound and well grounded. Each fix is a clause or a test-axis change, not a redesign.

CPG: considered, not relevant — `cpg_falkorchat` is stale (built 2026-10-15, 18 server commits behind); every claim here was checked by grep/read and by running the compiler offline.

**Answers to the brief's questions, in one place:**
1. **Identity premise.** For `Product`, `name` is unique, and it holds. `Entity` names are not
   unique, but that does not reopen RC-1. Details in "What's solid" §S1.
2. **Red count.** Reproduced exactly: 2,024 cases; Product 30, Chunk 105, Entity 0, Document 0.
   The spread across readings is 135 → 159 → 1,070 → 1,239. The probe misses three axes (B1, M2).
3. **Golden pairs.** Confirmed by execution. 0 of 40 pairs change compiled Cypher under their
   canonical completion, and 0 correct→incorrect flips across all 76 variants. 9 variants do change
   shape: the RC-1 `p.price`-only lists and one superlative variant. Every one of them was already
   incorrect before the change (§S2).
4. **K-086.** The recovery grounds no `productId` (§S3). The "no def bump" claim holds against all
   three traps (§S4). R1 is under-scoped (m3).
5. **Interactions.** Two cross-part defects: M1, and m6 (D7 is bound to `schema.identity`).
   Grounding exposure: m7.
6. **§8 reachability.** The targets are achievable, and the Wilson arithmetic checks out. Run 0 is
   not pinned to a sha (m2). The `git status` guard is vacuous (m1). B5 is not a control (m4).
7. **Sizing.** Each of A and B fits one implementer run (§S5).

### Findings

#### B1 — blocker — D1's facet exemption admits the exact RC-1 shape (the tuple-distinct per-row list)

*Verified by running it.* D1 condition 3 exempts any request whose projected properties are all
facets. The plan's assumption is that such a request "is about the value domain, not about rows"
(§3.1). That holds only for the plain `RETURN DISTINCT` branch. When `order_by` names a property
outside `returns`, `compile()` takes the tuple-distinct branch (`querygen.py:492-532`): it
de-duplicates on `(facet, order key)` and returns **one row per entity**, with no identity column.
- `returns ["p.category"], order_by "p.name"` compiles to `WITH DISTINCT p.category AS c0, p.name AS
  c1 ORDER BY c1 ASC LIMIT $limit RETURN c0 AS \`p.category\``.
- On `reference` that returns **15** unkeyed category rows ordered by name: Displays, Cameras,
  Audio, …
- This is RCA §3 step 4 (`returns ["p.price"], order_by "p.name"`) with `category` in place of
  `price`. A model holding the `p.name` list from step 2 zips the two lists by position. The
  customer is then told a wrong category, the same fabrication class as RC-1.
- `Entity` (`e.type`, `order_by e.name`) and `Document` (`d.sourceFormat`, `order_by d.title`)
  have the same hole.
- §6.1's oracle copies the exemption, so the probe would ship green with the hole in place. The
  RCA's invariant is "every compiled multi-row shape projects the label's identity". This shape is
  multi-row.

**Fix.**
1. Narrow condition 3 to: all projected properties are facets **and** the compiled shape is the
   plain branch (`order_by is None or order_by in returns`).
2. State the same narrowing in the oracle, in D5's text and in §6.2 (T-A5's facet test plus a new
   `order_by`-outside case).
3. The A1 red count becomes **Product 39, Entity 9, Document 6, Chunk 105 (159)** under the
   plan's axes. I re-derived this with `probe.py`. The new Entity and Document reds are the
   facet × outside-order × non-lookup cases.

The assertion that catches it is the probe's own `expect_prepend` branch, once the oracle is
corrected. It goes red in A1 (pass-through) on exactly those 24 extra cases.

#### M1 — major — D7's "is it known?" test normalises, but the empty query it diagnoses matched raw and case-sensitively

*Verified live.* `compile()` normalises only `*Normalized` properties (`querygen.py:412-423`). So
`category = "peripherals"` and `name = "wireless mouse pro"` match **0 rows**. D7's two checks
normalise, though:
- `has_category` matches against `categoryNormalized`;
- `lookup_product` calls `normalize_name`.

So both answer "known":

| Filter value | rows | check says |
|---|---|---|
| `category = "peripherals"` / `"PERIPHERALS"` / `"Smart home"` | 0 | `has_category` True |
| `name = "wireless mouse pro"` | 0 | `lookup_product` found |

D7 then returns the old `{"items": [], "finding": "no matching data found"}`, which is the DEF-C
dead end ("we have none") on a case variant of a real category. §4.2's "a known category emptied
by a price bound" is not what that branch actually catches. D8 has no such gap, because there the
repository normalises the *query* too. In D7 the query and the check use different matchers.

**Fix.**
- **(a)** Test exact membership in D6's `categories` list, which is already read once per call and
  matches `compile()`'s exact `=`. So `value not in categories` → recovery. This needs no
  `has_category` call and no second normaliser.
- **(b)** On a `name` filter, fire when `lookup_product` is `None` **or** the row's `name` is not
  equal to the raw value. A `nameNormalized` filter is already normalised by `compile()`, so a hit
  there really is "emptied by another filter".
- Add T-B tests for `"peripherals"` and for a lower-cased product name, and a mutant that
  reintroduces `has_category`.
- MB3's kill test changes with the trigger.

This is a design call, because it changes D7's reuse of D8.1. It goes to `architect`.

#### M2 — major — the probe misses three axes that D1's oracle varies on; three plausible mutants survive it

*Verified with `mutants.py`.* §6.1's axes include only `count(v)` as an aggregate, single-filter
requests, and `=` on `identity[0]` only. D1, though, depends on:
- every aggregate function (`count/avg/min/max`, with or without a property);
- *some* filter in a list of up to 4;
- `=` on *any* identity property (`nameNormalized`, `entityId`, `documentId`).

Each mutant below is green on the plan's 2,024 cases:

| Mutant | Plan axes | Wider axes | Also killed by §6.2? |
|---|---|---|---|
| aggregate detected as `startswith("count(")` | 0 fail | 4,280 fail | no. T-A3 uses `count(p)`. This is the regroup harm condition 2 exists to prevent (`["p.category", "max(p.price)"]`) |
| lookup checks only `filters[0]` | 0 | 145 | no |
| lookup checks only `identity[0]` | 0 | 30 | T-A4 (`nameNormalized`) catches it for Product only |

**Fix:** widen the axes rather than adding shapes. Use:
- every property × all six ops as a single filter;
- every (non-identity `=` first, identity `=` second) pair;
- every subset × {none, `count(v)`, and each `min(v.<p>)`};
- aggregate-only lists (`count(v)`, `count(v.<p>)` per property).

That is 43,651 cases in **0.8 s**, measured. The red count to state in A1, under the current D1,
is Product 230 / Chunk 840 / Entity 0 / Document 0 (1,070). With B1 fixed it is Product 299 /
Entity 66 / Document 34 / Chunk 840 (1,239). Keep the rule "if A1's count differs, stop". The
architect should restate whichever axis set is adopted, so the count stays derivable.

#### m1 — minor — §8.2's "tracked baseline not overwritten" check is vacuous

*Verified.* The command runs from `falkor-chat/server` and then checks `git status --short
falkor-chat/server/tests/eval/`. Git resolves that pathspec relative to the cwd
(`…/server/falkor-chat/server/tests/eval/`), so it prints nothing whatever happened, with exit
code 0 and only a stderr warning. **Fix:** `git status --short tests/eval/` from the same cwd, or
`git -C "$(git rev-parse --show-toplevel)" status --short falkor-chat/server/tests/eval/`.

#### m2 — minor — Run 0's baseline is not pinned to an explicit sha

§8.2 says "the commit before Stage A". If docs commits interleave, or Run 0 is taken after Stage A
lands, that resolves by inference. **Fix:**
- `teco` records `git rev-parse HEAD` in the coordination ledger when it dispatches Stage A.
- Run 0 uses that sha, after checking `git diff --quiet <sha> <StageA-first-commit>^ --
  falkor-chat/server`.

Two related notes. The export contains only `falkor-chat/server`, so `config.py`'s default
`../config/models.json` is absent from it. That is harmless only because both config env vars are
set explicitly; say so, or export `falkor-chat/config` too. The plan's `sys.path` claim is right:
the editable finder is `sys.meta_path.append`-ed (`__editable___falkorchat_0_1_0_finder.py:76`),
so the exported tree wins.

#### m3 — minor — R1 is scoped too narrowly; D6's "or in other words" widens the mapping beyond translations

- **D6 now competes with an existing instruction.** `_QUERY_REQUEST_INSTRUCTIONS` tells the model
  to filter a named item on `name`. D6 tells it to map anything that names a category "in other
  words" onto the closed list.
- **The sharpest golden risk is `nlq-14`, not `nlq-04`.** For "Which category does the Wireless
  Earbuds Pro belong to?", a 3B model can map it to `category = "Audio"` with `returns
  ["p.category"]`. That returns a row, so the not-found pair fails (L3), and D7 (b) never sees it.
- **Product-type words change behaviour.** "Do you have a mouse?" now maps to Peripherals and lists
  a keyboard and a webcam as well. §8.3 does not measure that, and B2 is measured on one word only
  ("periféricos").

**Fix:**
- Drop "or in other words". "In any language" covers DEF-C.
- Name `nlq-13/14/15` in R1 and in L6's catalog watch list.
- Optionally, add one non-category product-type browse to §8.3 as a reported, ungated measure.

#### m4 — minor — B5 is not an English control, and adding `en` changes the load B7 compares against

`refire_probe.py`'s `browse` script sends the same text, "Do you have any periféricos?", in every
language. So `en` measures an English *session* that receives a Portuguese/Spanish word. It does
not measure an English category word. Trials also run all languages concurrently
(`ThreadPoolExecutor(max_workers=len(languages))`). That makes 3 concurrent conversations against
the demo's 2 turn workers, where id2 had 2. `turn_ms` then includes queueing, so B7 is not
comparable with id2's browse window. **Fix:**
- Relabel B5 ("en session, same word").
- Run `en` as a separate pass, so the pt-BR + es pooled run keeps id2's 2-way load.

#### m5 — minor — D3 rewrites a line that crashes today; pin it

*Verified.* `default_returns` evaluates `f.get("property") in _NAME_LOOKUP_PROPERTIES`
(`querygen.py:310-313`). For a completion with `"op": "="` and an unhashable `"property": ["name"]`
(`returns` absent), that raises `TypeError: unhashable type: 'list'`. `QueryGraphDataTool.run`
catches only `(ValidationError, ValueError)` (`tools.py:1126`), so the tool crashes instead of
abstaining.

D3's predicate fixes this incidentally if `identity` stays a tuple, because tuple membership uses
`==`. It does not if an implementer uses a frozenset. **Fix:** in A1, guard
`isinstance(property, str)` inside `_is_identity_lookup`, and add the one-line test (red today).

#### m6 — minor — D7 trigger (b) is bound to `schema.identity`, which is the wrong set for "can `lookup_product` resolve this?"

§5.1 reuses D3's predicate for trigger (b), on the grounds that drift is the danger. But the two
questions differ. F2 asks "which column labels the row". Trigger (b) asks "is this value a product
*name*". If §3.2's reversal trigger ever adds `productId` to `identity`, then `productId = "x"` →
`lookup_product(name="x")` → `None` → a false "no product has that exact name". **Fix:** have
trigger (b) use the name properties explicitly. Either filter the predicate to `{name,
nameNormalized} ∩ identity`, or add a registry assertion that every catalog identity property is
name-resolvable, and point to it from §3.2's reversal trigger.

#### m7 — minor — §5.5's "K-085's residual does not widen" holds for the recovery result, but not for the turn it steers to

The recovery itself surfaces no ID (§S3). Its hint, though, sends the turn to
`filter_products(category=…)`, which grounds every `productId` in that category for the rest of
the node execution (`executor._GROUNDING_SOURCES`). Before the change, these browse turns ended
having grounded nothing. This is endorsed `v9` behaviour and probably acceptable. Still, the
sentence should say "the recovery grounds nothing; following it grounds the category's IDs, as any
`filter_products` browse does". The browse probe starts from an empty cart, so it cannot see the
effect. K-085 owns measuring it.

#### n1 — nit — `Chunk`'s identity is an opaque UUID

The querygen prompt itself calls `entityId` "an internal identifier the reader cannot use". D1
would prepend `c.chunkId` (~20 tokens a row) to every chunk-attribute list. `knowledge_base` is not
on the storefront, so this is harmless today. Consider `("documentId", "seq")`-style identity, or
state the choice.

#### n2 — nit — L2 could be sharper

The `knowledge_base` path is byte-identical except for F2, at temperature 0
(`config/models.json`). So "KB Run 1 equals Run 0 per pair, except replay-explained differences"
is a stronger target than "≥ 12/19 against a stale prior". L6 half-says this already.

#### n3 — nit — empty-catalog edge

D6 passes `None` on an empty list, but D7's shape then carries `categories: []` with a hint saying
"these are all the catalog's categories". Specify which of `[]` or no recovery applies. This is
moot at 15 products.

### What's solid

- **S1 — the identity premise (Q1).**
  - *Product.* `name` is unique in the data: 15 nodes, 15 names, 15 `nameNormalized`, 15
    `productId`. It is also unique by construction. The only writer, `seed_catalog.sh`, `MERGE`s on
    `productId = slugify(name)` under a `Product.productId` UNIQUE constraint. Equal names give
    equal slugs and so one node. `normalize_name` collisions also collide in the slug. There is no
    constraint on `name` itself, but it cannot be duplicated through any write path that exists.
  - *Entity.* Names are **not** unique: 62 nodes, 39 names in `ws:nlq-eval`; `marlowe robotics` ×12;
    "chief technology officer" typed both Concept and Other. This is by design, since entities are
    unfused. It does not reopen RC-1. Entity's only non-identity property is a facet, so D1 never
    prepends on Entity. And a `[e.name, e.type]` row is keyed: duplicates show as several labelled
    rows, never as an unkeyed list.
  - *Chunk.* Uses `chunkId` (UNIQUE constraint, `bootstrap_schema.sh:308`).
  - *Document.* `title` is unconstrained but 12/12 distinct today, and facet-only like Entity.
  - *§3.2 on its merits.* I agree with `name`. RC-1 is a reader-side misjoin, and `name` is the
    reader's key. `productId` would make every `query_graph_data` browse a grounding source while
    K-085 is open. It would also add allowlisted, model-filterable surface and break every Layer 1
    list pair. One argument is weaker than written: the `v9` prompt "not endorsing" `query_graph_data`
    as an ID source would not stop a model from copying a `p.productId` it was shown. The grounding
    argument is the one that carries the decision. The reversal trigger is well chosen.
- **S2 — the golden claim, checked by execution (Q3).**
  - All 40 canonical completions compile **byte-identically** with and without D1, and score
    identically. They are: `=`-name lookups (01–05, 13–15, 21–24, 28–30), `p.name`/`e.name` lists
    and their `returns`-absent variants (06–12, 40, 25–27), superlatives (16, 17, 20), aggregates
    including the recorded `count(p.name)`/`count(e.entityId)` (18, 19, 31–33), and the six
    structurally unanswerable pairs (34–39).
  - The tracked qwen run's result columns match these canonical projections pair for pair.
  - The only shapes D1 changes are the RC-1 `p.price`-only lists and a price-returning superlative.
    All were incorrect before the change, so L4's 0/40 is reachable by a faithful implementation.
- **S3 — grounding (Q4).** `_collect_target_values` collects only keys equal to `productId` or
  ending in `.productId` (`executor.py:393-420`). D7's shape has `items`/`finding`/`hint`/
  `categories` (strings), so it contributes ∅. `productId` is not allowlisted, so `compile()`
  rejects any projection of it. K-078's same-turn grounding is untouched.
- **S4 — no def bump (Q4).**
  - Trap 1: none of `config.tools`/`systemPrompt`/`config.model` changes. The querygen prompt and
    tool shapes are server code, and `proof_defs.py` is untouched.
  - Trap 2 (`v6`) and trap 3 (chat-triggered) do not apply.
  - The hint routes only to `v9`-granted tools. `v9` already says "a later corrected one wins",
    which suits D7's retry.
  - `verify_salesperson.sh demo` is the right gate.
- **S5 — sizing (Q7).**
  - Stage A is two code files, two test files, a docstring and a comment.
  - Stage B is one services method, one prompt seam and one branch with 15 named tests. Its stub
    plan (default `[]` → `None` → byte-identical prompt) keeps every existing `query_graph_data`
    test unmodified. I checked both `StubServices` and `StubQueryServices`.
  - Each stage is one implementer run, even with B1/M1/M2 folded in.
- **Other confirmations.**
  - Post-validation placement (§3.3) is the right call, and `compile()` stays pure.
  - The aggregate exemption (condition 2) is the correct reading of grouped aggregates.
  - The 8-category figure matches `reference`.
  - The Wilson arithmetic checks out: 20/24 lower bound 0.641, 4/24 upper bound 0.359, 11/12 lower
    bound 0.646.
  - The replay design (§8.2) isolates F2 cleanly.

### Open questions

1. **M1's trigger semantics** need `architect`'s call. Exact membership in D6's list is my
   suggestion, but it departs from "reuse D8.1's `has_category`". The stakeholder may prefer to
   keep D8 parity and accept the case-variant gap.
2. **m3:** should the browse re-measure gain a second word, or a product-type word? That needs a
   `refire_probe.py` script addition, which the plan currently rules out (§7).

## Pass 2

*2026-10-28. Re-gate of plan v2 (`Version: 2`).*

**Verdict: approve with suggestions.** Every Pass 1 blocker and major is fixed. I checked each fix
by execution against the tree and the live graph, not by reading. Pass 2 finds three new issues:
two minor (N1, N2) and one nit (N3). Each was introduced by a v2 edit and has a one-line fix. None is a defect in the design.
The changes are small enough that a Pass 3 is not needed (stopping condition below).

CPG: considered, not relevant — `cpg_falkorchat` is still stale; every check here ran the compiler and read-only queries against the tree and graph.

**What I ran** (scratchpad `qgd/`, all read-only):
- `probe_v2.py`: an independent recount on the v2 axes;
- `exec_sweep.py`: all 44,323 probe cases executed with `GRAPH.RO_QUERY` before and after a v2 D1
  prototype, on `reference` and `ws:nlq-eval`;
- `golden_replay_v2.py`: the 76-variant golden replay against the v2 D1;
- `d7_matrix.py`: v2's D7 triggers against an execution oracle on `reference`.

### Pass 1 dispositions

- **B1 — fixed.** Condition 3 now exempts a facet-only projection only on the plain branch. T-A6
  and MA8 were added, and the oracle is defined from the compiled shape.
  - Recount: on v1's axes, 159 (39/9/6/105). On v2's axes, `probe_v2.py` gives **1,239**
    (Product 299, Entity 66, Document 34, Chunk 840), matching the plan.
  - The architect's `probe2.out` shows the facet-ignores-order mutant killed (169).
- **M1 — fixed, and the departure is confirmed.** `d7_matrix.py` crosses
  - `category` / `name` / `nameNormalized`, each as exact, case variant, translated and padded;
  - with none, `price < 1` or a second category filter;
  - plus category×name pairs.

  That gives 49 cases, 44 of them with an empty result. The oracle wants recovery iff the `=`
  value alone returns 0 rows under `compile()`. Results:

  | Trigger design | false fires | misses |
  |---|---|---|
  | **plan v2** | **0** | **0** |
  | raw `!=` on `nameNormalized` too (MB8) | 2 | 0 |
  | v1 `has_category` | 0 | 20 |

  - So MB8/T-B18 are right.
  - That is also what Pass 1's M1(b) already recommended: raw `!=` for `name` only, a hit under
    `nameNormalized` meaning "emptied by another filter". The split is not a departure.
  - Dropping `has_category` for exact membership is likewise confirmed.
- **M2 — fixed.**
  - The totals differ only through Document's aggregate axis. My Pass 1 axis gave Document 39
    returns per order/filter cell; v2's `AGGS` gives 47, and 47 × 4 orders × 21 filters = 3,948.
    I reproduced **44,323** independently.
  - The red count is axis-independent because aggregate cases are never `expect_prepend`.
  - All three Pass 1 mutants are killed on v2's axes (5,328 / 55 / 170, `probe2.out`).
- **m1 — fixed.** From `server/`, `git status --short tests/eval/` resolves:
  `git ls-files tests/eval/nlq_eval_results.json` lists the tracked file from that cwd.
- **m2 — fixed, with a new vacuous guard.** The sha is pinned via the coordination doc, and
  `falkor-chat/config` is exported. See N1 below.
- **m3 — fixed.**
  - "or in other words" is dropped.
  - The new last sentence ("a named product keeps filtering on name") is what keeps `nlq-14` on
    trigger (b) rather than on a category row.
  - L6 gains a watch list, and `browse-type` was added (see N3).
- **m4 — fixed (dropped, not relabelled).**
  - Rechecked: `refire_probe.py` passes `SCRIPTS[script]` for every language (`:103-105`).
  - So an `en` browse would only measure an English session receiving the Portuguese/Spanish
    word, and dropping it keeps id2's 2-way load.
  - The English-category path is covered by L1 (`nlq-06/07/09/10/11/12/40`). I accept this.
- **m5 — fixed.** The `isinstance` guard is in D3. The red-first test and MA9 are in A1/§6.2.
- **m6 — fixed.** `_CATALOG_NAME_PROPERTIES` is in `tools.py`, and §5.1 now has two predicates.
- **m7 — fixed** (§4.2, §5.5 wording).
- **n1 — not fixed, accepted.** With D1's rule that "any projected identity prop suffices", a
  composite identity would let `seq` alone count as identity. `chunkId` is right.
- **n2 — fixed** (L2 is now per-pair equality with Run 0).
- **n3 — fixed** (both triggers require a non-empty `categories`; T-B19, MB9).

### Same-revision fixes, checked against each other

- **B1's condition 3 × M2's widened axes × the aggregate ordering.**
  - The v2 D1 prototype over all 44,323 cases changes `RETURN` columns on exactly the 1,239
    oracle cases. Every other case compiles byte-identically.
  - The "aggregate before facet" ordering in §6.1 is **not load-bearing**. D1 is a conjunction of
    exemptions, so any aggregate exempts whatever order the checks run in. That is harmless, but
    an implementer should not read it as a constraint.
- **The M1 triggers × m6 × n3.**
  - Trigger (b) is bound to `_CATALOG_NAME_PROPERTIES` and trigger (a) to D6's list. Both are gated
    on the list being non-empty.
  - `d7_matrix.py` ran with the live 8-category list: 0/0 above.
  - With an empty list, D6 passes `None` and D7 does not fire, so the two stay consistent.
- **D6 (m3) × the golden claim.** D6 cannot move compiled shapes, only completions; that is L1/L6's
  job live. Offline, `golden_replay_v2.py` with the v2 D1 gives the same result as Pass 1:
  - **0/40** canonical completions change Cypher;
  - **0** correct→incorrect flips across 76 variants;
  - the same 9 non-canonical, already-incorrect variants change shape.
- **R7 — I agree it is pre-existing and separable.** `exec_sweep.py` results:

  | Before → after D1 | cases |
  |---|---|
  | ok → ok | 27,391 |
  | fail → fail | 16,932 |
  | ok → fail | **0** |
  | fail → ok | **0** |

  - The failing set is identical before and after, and D1 introduces and removes nothing.
  - 14,860 failures are R7's "ORDER BY cannot reference variables not projected or grouped".
  - **2,072 are a second pre-existing class R7 does not name:** `avg()` over a string property
    ("Type mismatch: expected Integer, Float, or Null but was String", e.g. `avg(v.category)`),
    which `compile()` accepts.
  - Both escape `run` as a FalkorDB error, which is not a `ServiceError`. So both reach the
    executor's fault net (`executor.py:1176-1178`) and kill the turn.
  - One backlog item ("`run` abstains on an execution error", or `compile()` rejecting both
    shapes) covers both. Widen R7's text to name the `avg`-over-string class.

### New findings

#### N1 — minor — the new Run 0 guard is vacuous from the cwd §8.2 works in

*Verified.* §8.2 adds `git diff --quiet <sha> <Stage-A-first-commit>^ -- falkor-chat/server`
without stating a cwd. The surrounding steps run in `falkor-chat/server`, and from there the
pathspec resolves to `server/falkor-chat/server`. It matches nothing, and `--quiet` exits 0, which
means "no diff". On a range that does touch the server (`a5d8e807^..a5d8e807`):

| Where run | exit code |
|---|---|
| from `server/` | 0 (vacuous) |
| from the repo root | 1 |
| from `server/` with `':/falkor-chat/server'` | 1 |

This is m1's defect again in the fix for m2. The `git archive` line on the same bullet fails loudly
from `server/` (`fatal: pathspec … did not match`, rc 128), so it is safe, but it should still be
consistent.

**Fix:** write both commands with root-anchored pathspecs (`':/falkor-chat/server'`,
`':/falkor-chat/config'`), or prefix `git -C "$(git rev-parse --show-toplevel)"`. Also: §8.2's
"That export changes which neither…" is garbled.

#### N2 — minor — Stage B's stub-keying note contradicts T-B10

The stub's `lookup_product` is an exact `dict.get` (`tests/test_tools.py`,
`StubServices.lookup_product`). §6.2 says tests key the dict "on the normalised name".
- Under that setup, T-B10's `lookup_product("Wireless Mouse Pro")` returns `None`. Trigger (b)
  then fires, and T-B10 goes red for the wrong reason.
- Worse, an implementer may "fix" the trigger to make it pass.
- T-B17/T-B18 need the normalised key, but T-B10 needs the raw one.

**Fix:** state per test which keys the dict holds. For example, key both `"Wireless Mouse Pro"` and
`"wireless mouse pro"` to the same row in T-B10/T-B17/T-B18. Do not make the stub normalise:
existing lookup tests key on raw names (`test_tools.py:503`), and a normalising stub would change
them, which B2's "pre-existing tests unmodified" rule forbids.

#### N3 — nit — `browse-type` sends Portuguese to the `es` session

`refire_probe.py` sends one text per script to every language, so the es half of B5 measures an es
session receiving "Vocês têm algum mouse?". That is m4's mismatch, reintroduced in its replacement.
B5 is ungated, so this is a nit. Either run `browse-type` with `--languages pt-BR --repeat 12`
(same n = 12), or say in B5 that the es half receives Portuguese.

### Stopping condition

This gate is converging. All three Pass 2 findings came from v2's own edits, not from the v1
design. No blocker or major remains. None of the three new findings touches §3.1, §4.2's trigger
logic or §6.1.

**Falsifiable stop:** v3 needs **no Pass 3** if three things all hold:
1. The plan's diff from v2 to v3 touches only:
   - §8.2's Run 0 bullet (N1);
   - §6.2's Stage B stub note and T-B10/T-B17/T-B18 setups (N2);
   - the `browse-type` lines in §7/§8.3 (N3);
   - R7's text.
2. From `falkor-chat/server`, v3's diff guard exits **1** on `a5d8e807^ a5d8e807`.
3. `probe_v2.py` still prints 44,323 / 1,239.

`teco` can check all three mechanically. The plan is untracked, so snapshot v2 before the edit and
compare the two copies with `diff`. Any other change to the plan reopens the gate.
