# Cart writes keyed by product ID, not the model's rendering of the name

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-078 · **Version:** 3 · **Reviews:** `falkor-chat/docs/reviews/salesperson-cart-product-id.md` · **Extended by:** `falkor-chat/docs/plans/salesperson-cart-product-id2.md`

*2026-10-22 v2: revised per analyst Pass 1 (F1–F8, n1–n3). Changes: conditional hold message, re-fire live condition, S4+S5 coupling, per-tool ID sources, mutant fixes, gate definitions, F1-landed wording.*

*2026-10-22 v3: text-only revision per analyst Pass 2 (P2-1, n4–n6): stakeholder answers folded in as settled, D7 worst-path count, `_HELD_REASON` comment, S2 line break, tree reference. No design change.*

## 1. Goal & scope

**Goal.** Close K-078 (`falkor-chat/docs/BACKLOG.md`): `add_to_cart`/`remove_from_cart` identify
the product by the catalog's stable `productId`, copied by the model from a tool result, never by
the model's rendering of the name. The K-058 off-turn guard keeps its purpose (stop the model
re-firing a previous turn's write) but grounds an ID-keyed write on **where the ID came from this
turn**, not on the user's text. Ship it as `salesperson@v9`.

**In scope:** the two cart write tools (`tools.py`), their service methods (`services.py`), the
K-058/K-061 guards (`executor.py`), the not-found shape of `lookup_product_fact`, the `v9` def
(`proof_defs.py`) with the K-079 F3 currency sentence folded in (§4), version pins, tests, docs.

**Out of scope:** mapping a translated name to a canonical one, or loosening text matching (both
rejected by the stakeholder); localized product names in the catalog; changing `productId` values
or format (they are keys on `CartItem`/`OrderLine` in every `ws:*` graph); K-079 F1 (landed
separately at `ac39345d`), F2 (a follow-up, §4), F4/F5; `lookup_product_fact` name matching itself.

## 2. Context & findings

CPG: considered, not relevant — `cpg_falkorchat` is stale (12 server commits behind), and every
symbol this plan touches was read directly from the current tree (`ac39345d`) instead.

**Code seams (verified by reading the source):**

- `tools.py` `AddToCartTool` / `RemoveFromCartTool`: schema requires `productName`; `run()` passes
  it to `services.add_cart_item(ctx, product_name=…)` / `remove_cart_item(ctx, product_name=…)`;
  unknown product → `{"found": false}`. `AddToCartTool.run` defaults `quantity` to 1;
  `RemoveFromCartTool.run` passes `quantity` through (omitted = whole line).
- `services.py` `Services.add_cart_item` / `remove_cart_item` resolve the name with
  `Services.lookup_product` (exact `Product.nameNormalized` match), then write by `productId`.
  The only non-test callers are the two tools (grep over `server/`).
- `repository.py` `Repository.lookup_products_by_id` already resolves a list of IDs to
  `{productId, name, price}` (QUERIES §15/§16 two-graph read) — usable as-is for ID resolution;
  no new Cypher is needed anywhere in this plan.
- **Does the model see `productId` today? Yes, in three tool results.**
  `Repository.lookup_product` and `Repository.filter_products` both project `p.productId`, and
  `LookupProductFactTool` / `FilterProductsTool` return those rows verbatim; `Services.get_cart`
  (`view_cart`) returns `productId` per line. `add_to_cart`/`remove_from_cart` results also echo
  it. `query_graph_data` rows carry only what its NLQ projection asked for (keys like `p.name`),
  so normally no ID. **But no tool result from an earlier turn is in the model's context:**
  `Executor._assemble_messages` rebuilds each turn from the system prompt, the thread's posted
  messages and the `CONTEXT` block only. So an ID is available to the model only if a tool
  returned it **in the current turn**.
- `executor.py`: `_WRITE_TARGET_ARG` maps both cart tools to `productName`;
  `Executor._handle_tool_call` holds a call whose target is not a substring (via
  `extraction.normalize_name`) of `turn_text` = the triggering message + the model's latest text
  this turn (`_target_mentioned_in_turn_text`, which fails open on empty text), then applies the
  K-061 same-turn dedup on `(name, _dumps(_resolve_dedup_arguments(...)))` over
  `dispatched_writes`. `Executor._note_off_turn_write_held` logs/traces the hold.
- `proof_defs.py` `SALESPERSON_DEF` is `v8`; `systemPrompt`, `config.tools` and `config.model`
  are create-only (`falkor-chat/AGENTS.md`, "Three traps when bumping `SALESPERSON_DEF`"); `v6`
  is burned. Tool **schemas** live in code, not in the def, so a schema change reaches every def
  version that grants the tool.

**Why es/pt-BR fail (from the U3 raw logs, `…/scratchpad/u3/lms-server-q3.log`):** the probe's
user text is always English ("Add the Wireless Mouse Pro to my cart.",
`salesperson/scripts/language_salience_probe.py` `SCRIPTED_TURNS`), and the model translates the
name into its tool arguments. It does the same in **`lookup_product_fact`** (12×
`name="Mouse sem fio Pro"`, plus `Mouse Pro inalámbrico`, `Mouse Pro Wireless`, … vs. 10× the
English name). So "look the name up first, then copy the ID" fails the same way unless the
not-found answer gives the model something to pick from. This shapes design decision D2.

**Baseline, recomputed from the n=75 Q3 run** (`single-model.jsonl` for participant→language,
read-only `GRAPH.RO_QUERY` on `ws:demo` for each participant's `CartItem`s and `OrderLine`s, and
`demo-q3-window.log` for warnings). Measure M1 = participants whose cart or a placed order holds a
`wireless-mouse-pro` line:

| Measure | en | pt-BR | es |
|---|---|---|---|
| M1, mouse reached cart/order (n=75 each) | **70** (93.3%, Wilson [85.3, 97.1]) | **1** (1.3%, [0.2, 7.2]) | **0** (0%, [0.0, 4.9]) |
| M2, lines for any other product | 0 | 0 | 0 |
| M3, participants whose total mouse quantity across `CartItem` + `OrderLine` is > 1 | 0 | 0 | 0 |
| K-058 holds (`off-turn write held`) | 0 | 74 | 70 (+2 `Mouse Wireless Pro`, unattributed) |

Total holds 146, all `add_to_cart`, every one a translated name, in 146 distinct runs (one hold
per run, never retried). 0 K-061 holds. 0 dead turns of 1,125. Turn p95 is 14.1 s overall and
14.1 s at turn index 2, the add turn. Per language at turn index 2 it is **en 20.1 s, pt-BR
10.1 s, es 10.1 s**: a held pt/es turn ends early, so en already sits about 9 s from the 29 s line
(per-language figures from the analyst re-derivation, review Appendix A). Wrong-language replies:
0/375 per language. Name→price pairs stated in replies: 31 wrong of 79 (K-079 RCA §2). en carts are mostly not converted
to orders (67 open carts, 3 orders), which is why M1 counts both.

## 3. Design & rationale

Seven decisions. Each is stated once here; §5 cites them by number.

**D1 — The cart write tools take `productId`; `productName` is dropped.** Both schemas declare
`productId` (string, required) plus the unchanged `quantity`. The description says the value is
an identifier copied exactly from a tool result in this turn, never typed from a name and never
translated. **The description names each tool's own sources, matching D3's table:** for
`add_to_cart`, `lookup_product_fact` or `filter_products`, and never `view_cart`; for
`remove_from_cart`, those two plus `view_cart`. `productName` is **not** kept, not even unadvertised. A kept name path
would keep the translation failure and need the text guard alongside the new one: two mechanisms
for one job. A call that sends only `productName` fails the executor's existing required-argument
check ("missing required argument(s) ['productId']"), which already re-prompts the model. An
extra `productName` next to a `productId` is ignored by the tool (and by the dedup key, D4).
*Back-compat:* tool schemas are code, not def data, so every stored def version that grants these
tools (`v2`…`v8`) gets the ID schema too. No def stores argument names (`config.tools` is a list
of tool names), so nothing in the graph breaks. Older prompts still speak of product names, which
is acceptable: only `v9` is pinned for the demo (D7).

**D2 — Where the ID comes from: current-turn tool results, with a catalog list on a failed
lookup.** IDs already reach the model in `lookup_product_fact` (found), `filter_products` and
`view_cart` results (§2). One change is needed: **`lookup_product_fact`'s not-found answer becomes
`{"found": false, "hint": "no exact match; to get a product's price or category, call
lookup_product_fact again with its exact catalog name from this list", "catalog": [{"productId":
…, "name": …}, …]}`**, every catalog product, sorted by name, **ID and name only, no price or
category**. The `hint` covers the K-079 class: on a pt/es *price* question, the miss returns real
names with no prices. Trial 14's $45.99 shows this model invents a price when none is in context,
so the hint points it at the call that does return one. A pt-BR/es model that looks up
"Mouse sem fio Pro" then gets a list it can pick from, and the cross-language match happens in the
model, not in our code. This is not the rejected name mapping: the server never guesses which
product was meant. It returns real rows and the model picks the ID. Cost: about 250 tokens, only on
a failed lookup. (Sorting by name is deliberate. `filter_products` sorts by price, and the K-079
misjoin came from two lists of the same products in different orders. With ID and name in the same
row, a misjoin is impossible either way.)

*Rejected:* (a) **a catalog index in the prompt or `CONTEXT` block every turn**: it adds about
250 tokens to every call on an 8,192-token model already near the U0 concurrency overflow, and
because every ID is always visible, provenance grounding (D3) loses its meaning. (b) **Localized
names in the catalog**: a data and seed change, still string matching, and still blind to the
model's own paraphrase. (c) **Prompt only** ("on a miss, call `filter_products`"): one more
iteration and one more 3B prompt-adherence bet. (K-079 F1, `ac39345d`, has since made
`filter_products(category="")` a working whole-catalog call, so `filter_products` remains a valid
source. It is just not the primary path.)

**D3 — K-058 for an ID-keyed write grounds on provenance, not text.** Answer to the brief's Q2:
**the purpose still needs grounding.** §9.2's repro is a later turn's tool loop re-issuing an
earlier turn's add, and an ID does not change that. Text cannot ground an ID. The user never types
it, and the canonical English name is not in a Portuguese customer's text. So the guard asks a
language-independent question instead: **did a tool this node execution dispatched return this
`productId`?** Concretely:

- The executor keeps, per node execution, `surfaced: dict[str, set[str]]` as a **local of
  `_run_agent_node`** beside `dispatched_writes`, never on `self`. `WorkflowExecutor` is a
  process-wide singleton (`_drive`'s comment), so state on `self` would carry grounding across
  turns *and across customers*. It maps each write tool to the normalized IDs it may target. After every
  **successful** dispatch, `_collect_target_values(content, arg)` parses the result as JSON (a
  non-JSON result contributes nothing) and walks it, collecting every string value whose key is
  `productId` or ends in `.productId` (the second form keeps a future K-079 F2 `query_graph_data`
  row like `{"p.productId": …}` counting with no further change, §4). Each value is added to the
  set of every write tool whose **source set** contains the tool just dispatched.
- **Source sets** (new table `_GROUNDING_SOURCES` beside `_WRITE_TARGET_ARG`):
  `add_to_cart` ← `lookup_product_fact`, `filter_products`, `query_graph_data`, `add_to_cart`;
  `remove_from_cart` ← the same plus `view_cart` and `remove_from_cart`.
  **`view_cart` deliberately does not ground an add.** The cart's contents are exactly the
  previous turn's writes, so "it was in the cart I just viewed" is the re-fire shape itself. A
  remove, by contrast, naturally targets what is in the cart.
- Before dispatch, a call to a tool in `_WRITE_TARGET_ARG` is **held** unless
  `normalize_product_id(value)` is in `surfaced[call.name]`. That covers a missing, empty,
  non-string, hallucinated, translated-slug or previous-turn ID. **The held result is
  conditional** (review F1). An unconditional "look it up, then retry" would hand a re-fire its
  own recovery path. The wording is a per-tool constant `_HELD_REASON[call.name]`:
  *add:* "`productId` <v> was not returned by `lookup_product_fact` or `filter_products` in this
  turn, so the call was not run. Only add a product the customer asked for in their current
  message. If they did ask for this one, look it up first with `lookup_product_fact` (by its
  catalog name) or `filter_products`, and pass the `productId` that result returns. If they did
  not, do not add it."
  *remove:* the same, with `view_cart` added to both source lists and "remove" for "add".
  This restores the old guard's main lever, model compliance with "only what the customer asked
  for now": in Q3, 146 of 146 held runs never retried. Observability is unchanged:
  `_note_off_turn_write_held`, trace kind `off_turn_write_held`, and the `off-turn write held`
  WARNING prefix. No script greps that prefix (review n1). It is kept for continuity with the
  U3/K-058 ad-hoc log analyses and their counts. Only the suffix wording changes.
- **No fail-open.** The old text check failed open on an empty `turn_text` (offline stub path).
  Provenance has no missing-data case, so offline tests that dispatch a cart write must script a
  surfacing tool call first (§6).
- `_target_mentioned_in_turn_text`, the `turn_text` parameter of `_handle_tool_call`, and
  `trigger_text`/`turn_text` in `_run_agent_node` are only used by this guard and are removed.

*Residual risk (§8 R3, accepted by the stakeholder with these mitigations; measured in §6.3 C2):* the guard's depth
is one decision, not a wall.
- A re-fire that is held can still go through if the model ignores the conditional reason, looks
  the product up, and re-adds it. This is pinned as accepted behaviour by test 26.
- On the pt/es path, a failed lookup's `catalog` list, or a whole-catalog `filter_products` or
  `query_graph_data` listing, grounds every product for the rest of that turn, so the guard
  holds nothing after it.

What remains is compliance with the conditional wording. The old guard's equivalent hole was a
re-fire whose product name the model had already written in its own text that turn. *Rejected:* grounding on "the canonical name
appears in the turn text" (fails for exactly the es/pt-BR users this item is about), and holding
only IDs already in the cart (needs a cart read inside the generic executor, misses re-fires after
`place_order` empties the cart, and false-holds a legitimate "one more").

**D4 — K-061 dedup keys on the arguments each tool honours, with the ID normalized.**
`_DEDUP_ARG_RESOLVERS` gets an entry for **both** tools. Each projects to exactly the arguments
its `run()` honours:
`add_to_cart` → `{"productId": normalize_product_id(id), "quantity": q or 1}`;
`remove_from_cart` → `{"productId": normalize_product_id(id), "quantity": arguments.get("quantity")}`,
the value exactly as `RemoveFromCartTool.run` passes it. Never use `or None`, which would merge an
explicit `0` with "omitted = whole line" (review F8).
The second keeps K-061's documented "omit = whole line" distinction. The projection also closes a
new instance of the §15.2 loophole: an ignored extra argument (`productName`, D1) or an ID's
letter case would otherwise give one request two different keys. Because K-058 now runs first,
K-061 only ever sees a grounded call (same ordering as today).

**D5 — Services resolve by ID; one shared normalizer.** New `extraction.normalize_product_id(value:
Any) -> str`: `value.strip().lower()` for a `str`, else `""`. It lives in `extraction.py`, home of
the shared normalizer `normalize_name`, so the executor guard, the dedup key and the services share
one function (K-058's own "no second normalizer" rule). This is ID **format** normalization, not name mapping.
Its docstring must say that the case-folding is only safe because every `productId` is lowercase,
and that this is guaranteed by `seed_catalog.sh` `_slugify` alone (all 15 live IDs on
`reference` verified lowercase). A future catalog source with mixed-case IDs must keep that
invariant or drop the `.lower()` (review n3).
Services:
- `Services.lookup_product_by_id(ctx, *, product_id) -> dict | None`: normalize, `""` → `None`,
  else `self._repo.lookup_products_by_id(product_ids=[pid])` and return the first row or `None`.
- `Services.add_cart_item(ctx, *, product_id: str, quantity: int)` and
  `Services.remove_cart_item(ctx, *, product_id: str, quantity: int | None = None)` replace the
  `product_name` keyword. The bodies stay the same except that resolution goes through
  `lookup_product_by_id`. The return shape is unchanged: `lookup_products_by_id` returns
  `{productId, name, price}`, which is everything the two methods return today.
- `Services.catalog_index(ctx) -> list[dict]`: `{productId, name}` for every product, sorted by
  `name`, from the existing `self._repo.filter_products(category=None, min_price=None,
  max_price=None, limit=CATALOG_INDEX_LIMIT)` (projected and re-sorted in Python, so no new Cypher).
  `CATALOG_INDEX_LIMIT = 50` sits in `services.py`, above the ~15-product demo scale. Silent
  truncation at that limit is the same posture `filter_products` already has.

**D6 — A hallucinated or malformed ID is held by the guard (D3) before it reaches a tool**, with
recovery instructions. The tool's own abstention is defence in depth for an ID that was surfaced
but no longer resolves (a product deleted mid-conversation): `{"found": false, "reason": "no
catalog product has productId '<id>'"}`. It never writes and never fabricates a line, mirroring
today's `{"found": false}`. `remove_from_cart` keeps its distinct "known product, nothing in
cart" outcome (`{"found": true, "removed": false, …}`).

**D7 — `salesperson@v9`.** `systemPrompt` is create-only and has to change, so this needs a new
version. **`v9`** (never `v6`). `config.tools`, `config.model`
(`lmstudio/mistralai/ministral-3-3b`, repeated per the bump trap), `requiredTools`,
`maxIterations` (8) and topology stay byte-identical to `v8`. The `systemPrompt` changes in two
places:
1. In the cart paragraph, replace "if a product name does not match anything in the catalog, say
   so plainly rather than adding it anyway" with:
   "To add an item, pass its `productId`, copied exactly from a `lookup_product_fact` or
   `filter_products` result in this same turn. A `productId` you only saw in `view_cart` is not
   enough for an add. To remove an item, you may also copy the `productId` from `view_cart`. A
   `productId` is an identifier: never translate it, reword it, or build it from a product name.
   Only add or remove what the customer asked for in their current message.
   Look products up by their catalog name. If `lookup_product_fact` finds no exact match, it
   lists the catalog; pick the product the customer means from that list and use its
   `productId`. If nothing in the list matches, say so plainly rather than adding anything."
2. Add K-079 F3 (§4) as one sentence at the end of the catalog paragraph: "All prices are in US
   dollars; always write them with `$`, never another currency symbol."

Add a `v9` note to the `SALESPERSON_DEF` comment block (per-version history lives there) and
re-point every pin (§5 S5). `maxIterations` stays 8: the worst pt/es add path is `get_profile`,
`add_to_cart` (held), `lookup_product_fact` (miss plus list), a second `lookup_product_fact` when
the model follows the miss hint, `add_to_cart`, `post_message`, which is 6. A call that sends only
`productName` adds one re-prompt, which makes 7. Both stay within 8.

**Interface changes, called out:** the `add_to_cart`/`remove_from_cart` tool schemas (model-facing
contract), the `lookup_product_fact` not-found shape (additive key), and the `Services.add_cart_item`
/ `remove_cart_item` keyword (`product_name` → `product_id`). The only callers are the two tools and
tests. No graph schema, query or data change.

## 4. Effect on what catalog tool results show the model (K-079 sequencing)

Interaction with `falkor-chat/docs/reviews/salesperson-fabricated-prices-rca.md` (K-079).

**What this plan changes in model-visible tool results:**

| Tool | Change | Price-bearing? |
|---|---|---|
| `lookup_product_fact` | not found: `{"found": false}` → `{"found": false, "hint": …, "catalog": [{productId, name}…]}`, sorted by name (D2). The `hint` sends a price question back to an exact-name lookup | **No.** ID and name only, deliberately, so a not-found answer can never become a second price list to misjoin |
| `add_to_cart` / `remove_from_cart` | schema `productName` → `productId`. The rare not-found result gains a `reason` (D6). Success rows are unchanged | Unchanged |
| executor hold message | reason wording only (D3) | No |
| `filter_products`, `view_cart`, `query_graph_data`, `place_order` | **unchanged** | — |

**The identity decision K-079 F2 needs: a product row's identity is `productId`.** It is the
catalog's `{label}Id` (unique-constrained, QUERIES §15). It is what `lookup_product_fact`,
`filter_products` and `view_cart` already return, and what the cart tools now take. F2's identity
column for `Product` should be `p.productId`, normally beside `p.name`. With it, every
`query_graph_data` row also becomes a D3 grounding source for free: the collector already accepts
`.productId`-suffixed keys.

**(a) Does F2 belong in this plan? No. It is a follow-up sequenced after this plan lands.**
Reasons: F2 changes the `querygen` contract (`QueryRequest`/`compile`, `DatasetSchema` per-label
identity) and the NLQ eval's Layer 1 "return exactly what was asked" rule. Its done-condition is a
Layer 1 re-run on ministral that needs LM Studio and a stakeholder go. Its blast radius (every
`query_graph_data` dataset, including `knowledge_base`) is unrelated to cart writes. Nothing in
K-078 depends on it: D2/D3 ground on `lookup_product_fact`/`filter_products`, not on
`query_graph_data`. Sequencing: **K-078 first, then F2's design note** (`architect`,
`falkor-chat/docs/plans/query-graph-data-row-identity.md` or the slug `teco` picks) **taking
`productId` as given.** F2 is code-only (no prompt), so it needs no def bump of its own.

**(b) Does F3 go into the same bump? Yes. It is in `v9` (D7 item 2).** `systemPrompt` is
create-only, so a separate F3 would cost a `v10` and a second set of pin moves days later. The
RCA asks for it to land "in the same change as K-078's rendering decision". The prompt-sentence
placement is chosen over a currency field on price rows: a field would touch every price-bearing
result (`lookup_product_fact`, `filter_products`, `view_cart`, `add_to_cart`, `place_order`,
`query_graph_data`) for a fact that never varies in a USD-only catalog. The cost: the K-078 live
check (§6) carries one extra prompt change, so it also reports non-`$` currency symbols
(baseline: 7 replies of 1,125), as a secondary measure, not a K-078 gate.

**K-079 F1 has landed (`ac39345d`).** It put the blank-`category` rule in
`Repository.filter_products`, for every caller, and added `tools.py` `_coerce_price_bound` for
blank or numeric-string price bounds. S2 branches from `ac39345d` or later. Its edit to
`LookupProductFactTool.run` sits directly above `_coerce_price_bound`, so a pre-`ac39345d` base
would conflict textually; there is no semantic overlap (`catalog_index` calls the repository with
`category=None`). F1 also makes a model-sent `filter_products(category="")` (5 in Q3) a working
whole-catalog ID source for the §6.3 run.

## 5. Step-by-step implementation

Paths are relative to `falkor-chat/server/` unless rooted. Run tests from `falkor-chat/server/`
with the venv. **Never run the default full suite on the shared instance** (it wipes `reference`,
`falkor-chat/AGENTS.md`); run the named files, and run `test_services.py` (FakeRepo-only) freely.
`test_executor_agent.py`, `test_tools.py` and `test_salesperson_scaffold.py` are stub-based. `test_storefront.py`
and `test_storefront_api.py` use a real `conn` and seed a catalog, so run them only when
re-seeding afterwards is acceptable, or leave them to `teco`'s gate.

| Step | What | Files | Done when |
|---|---|---|---|
| S1 | `normalize_product_id` (D5) | `falkorchat/extraction.py`; `tests/test_extraction.py` | Unit tests pass: strip+lower; non-`str`/`None` → `""`; idempotent |
| S2 | Services by ID + tool schemas/results (D1, D2, D5, D6) | `falkorchat/services.py`, `falkorchat/tools.py` (branch from `ac39345d`+, §4); `tests/test_services.py`, `tests/test_tools.py`, `tests/test_executor_agent.py` (`FakeCartServices` wraps the real `AddToCartTool`), `tests/test_storefront.py`, `tests/test_storefront_api.py` (`product_name=` call sites → `product_id=` with the fixture's slug, e.g. `"widget-001"`) | `grep -rn "product_name\|productName" falkorchat/` returns nothing cart-related; targeted tests green; mutants M7–M8 each turn ≥1 test red |
| S3 | Guard + dedup (D3, D4) | `falkorchat/executor.py`; `tests/test_executor_agent.py`, `tests/test_salesperson_scaffold.py` (stub schemas `productName` → `productId`) | §6 executor behaviours green; `_target_mentioned_in_turn_text` and the `turn_text` plumbing gone; named mutants M1–M6, M9 and M10 each turn ≥1 test red |
| S4 | `salesperson@v9` (D7) | `falkorchat/proof_defs.py`; `tests/test_salesperson_scaffold.py` (version assert `v9`, prompt-content asserts) | Scaffold tests green; `v9` config equals `v8`'s except `systemPrompt`; **committed together with S5** |
| S5 | Version pins `v8` → `v9` | `falkor-chat/scripts/start_demo.sh` (`FALKORCHAT_TRIGGER_DEF_VERSION` default + header), `scripts/seed_salesperson.sh` (default + header list), `scripts/verify_salesperson.sh` (default + header) | `grep -rn "v8" falkor-chat/scripts/{start_demo,seed_salesperson,verify_salesperson}.sh` shows history mentions only, **in the same commit as S4's `version` bump**; after that commit, `seed_salesperson.sh demo` then `verify_salesperson.sh demo` exit 0 (create-only publish, safe on `ws:demo`) |
| S6 | Docs | §7 list | Every §7 row done |
| S7 | Live check | `qa-engineer`, §6.3 | §6.3 numbers met, recorded in `falkor-chat/docs/test-reports/salesperson-cart-product-id-report.md` |

**S2 and S3 land in one commit** (two review units, one commit). After S2 alone,
`_WRITE_TARGET_ARG` still names `productName`, which an ID-keyed call never carries, so the
guard would silently skip every cart write. After S3 alone, every cart write would be held.

**S4 and S5 land in one commit** (review F2). `seed_salesperson.sh` publishes `{**SALESPERSON_DEF,
"version": $FALKORCHAT_SALESPERSON_DEF_VERSION}` with a `v8` default, and `verify_salesperson.sh`
compares the stored config with the *current* constant. Between S4 and S5, a seed onto a freshly
wiped `reference` (after any default pytest run) would store `v9` content under the name `v8`. That
is the burned-`v6` shape (`falkor-chat/AGENTS.md` trap 2), and `config` is create-only, so it
would burn `v8` permanently.

**S2 details.**
- `services.py`: add `lookup_product_by_id`, `catalog_index` and `CATALOG_INDEX_LIMIT`. Re-key
  `add_cart_item`/`remove_cart_item` on `product_id` (D5). Update both docstrings: "resolves
  `product_id` via `lookup_product_by_id`; `None` when it doesn't resolve (AC-3 abstention)".
- `tools.py` `AddToCartTool`/`RemoveFromCartTool`: schema per D1, with **per-tool** `productId`
  descriptions (review F3).
  *Add:* "The product's `productId`, copied exactly from a `lookup_product_fact` or
  `filter_products` result in this turn. An identifier: never translate it or build it from a
  name."
  *Remove:* the same, with "or `view_cart`" added to the source list.
  Tool descriptions: "a product" instead of "a named product". `run()` passes
  `product_id=arguments.get("productId", "")`. Not found → `{"found": false, "reason": …}` (D6).
  Update both class docstrings.
- `tools.py` `LookupProductFactTool.run`: on `None`, return `{"found": false, "hint": <D2 text>,
  "catalog": self._services.catalog_index(ctx)}`. Update the docstring's abstention sentence. Change the
  `name` parameter description to "The product's name as listed in the catalog." (the catalog
  names are English; this is a nudge, not a mapping).
- Tool-test stubs (`tests/test_tools.py` `StubServices`, `tests/test_executor_agent.py`
  `FakeCartServices`) gain the new method names/keywords.

**S3 details (`executor.py`).**
- `_WRITE_TARGET_ARG = {"add_to_cart": "productId", "remove_from_cart": "productId"}`. Rewrite
  its comment block: provenance grounding, the reason it replaced text grounding (K-078), and the
  "adding a write tool here?" checklist extended with "add its `_GROUNDING_SOURCES` row".
- New `_GROUNDING_SOURCES: dict[str, frozenset[str]]` per D3.
- New `_collect_target_values(content: str, arg: str) -> set[str]`: `json.loads` inside
  `try/except ValueError` (→ empty set). Walk dicts and lists recursively and collect
  `normalize_product_id(v)` for every `str` value `v` under a key `k` with `k == arg or
  k.endswith("." + arg)`. Drop `""`.
- `_resolve_add_to_cart_dedup_args` projects per D4. New `_resolve_remove_from_cart_dedup_args`.
  Rewrite the `_DEDUP_ARG_RESOLVERS` comment's `remove_from_cart` bullet, which now *has* an entry
  that normalizes the ID and keeps `quantity` raw.
- `_run_agent_node`: create `surfaced: dict[str, set[str]] = {t: set() for t in
  _WRITE_TARGET_ARG}` next to `dispatched_writes`, and thread it through both `_handle_tool_call`
  call sites in place of `turn_text`. Delete `trigger_text`/`turn_text`.
- `_handle_tool_call`: replace the K-058 block with the D3 check (hold unless
  `normalize_product_id(call.arguments.get(target_arg)) in surfaced[call.name]`). Held JSON
  `reason`: the conditional per-tool `_HELD_REASON[call.name]` text from D3 (module-level
  constants beside `_GROUNDING_SOURCES`, so the tests can assert on them; add one comment line
  beside `_HELD_REASON` saying the add message's two named sources are deliberately narrower than
  the guard's accepted set (which also includes `query_graph_data` and `add_to_cart`), so a later
  edit does not "fix" one to match the other). On the success path, after
  `satisfied.add`, run, for each write tool `w` with `call.name in _GROUNDING_SOURCES[w]`:
  `surfaced[w] |= _collect_target_values(content, _WRITE_TARGET_ARG[w])`. Rewrite the
  docstring's K-058 paragraph (provenance, sources, no fail-open) and its K-061 sentence "only
  ever sees a call whose target IS mentioned in this turn's own text" → "…whose target was
  surfaced this turn".
- `_note_off_turn_write_held`: keep the name, trace kind and WARNING prefix. Change the
  suffix "target not mentioned in this turn's own text" to "target not returned by a grounding
  tool this turn".
- Delete `_target_mentioned_in_turn_text`. Its `normalize_name` import has no other use in
  `executor.py` (verified by grep), so replace it with `normalize_product_id`.

## 6. Test strategy

### 6.1 Offline behaviours, in red→green order

Each behaviour is one test or a small cluster. Existing tests that encode the old contract are
**rewritten**, not deleted silently. The §9.2 repro test keeps its scenario and swaps to
IDs.

*`extraction` (S1)*
1. `normalize_product_id(" Wireless-Mouse-Pro ") == "wireless-mouse-pro"`; `None`, `3`, `{}` →
   `""`; applying it twice gives the same result.

*Services (S2, `test_services.py`, FakeRepo)*
2. `add_cart_item(product_id="prod2", quantity=2)` writes the line and returns `{productId, name,
   price, quantity}`. A brand-new customer still gets `ensure_customer`/`ensure_cart` first.
3. `add_cart_item` with an unknown ID, `""`, or a non-string returns `None` and **writes
   nothing** (no `ensure_*` either).
4. ID case/whitespace tolerance: `" PROD2 "` resolves to `prod2`. The FakeRepo IDs are lowercase,
   like the real slugs.
5. `remove_cart_item` by ID: partial, whole-line (omitted quantity), not-in-cart
   (`removed: False`), unknown ID → `None`.
6. `catalog_index` returns `{productId, name}` only (no `price`/`category` keys), sorted by name.
7. `lookup_product_by_id` returns the matching row for a known ID and `None` for an unknown one.

*Tools (S2, `test_tools.py`)*
8. The `add_to_cart`/`remove_from_cart` schemas require `productId` and have **no** `productName`
   property. The `add_to_cart` schema text (description + `productId` description) does **not**
   contain `view_cart`; the `remove_from_cart` one does (review F3).
9. `AddToCartTool.run({"productId": …})` passes `product_id=` through. Omitted `quantity` → 1.
   Unknown → `{"found": false, "reason": …}`.
10. `RemoveFromCartTool.run` passes `quantity=None` through when omitted. Unknown → `found:false`
    plus `reason`.
11. `LookupProductFactTool.run` on a miss returns `found:false`, a non-empty `hint`, and `catalog`.
    Every entry has exactly the keys `productId`, `name`. A hit is unchanged (still carries
    `productId`).

*Executor guard (S3, `test_executor_agent.py`, stub LLM + stub registry)*
12. **§9.2 repro, ID form.** Turn 2 is "Also add 1 Mechanical Keyboard K200". The model calls
    `lookup_product_fact` (result carries `mechanical-keyboard-k200`), then
    `add_to_cart(mechanical-keyboard-k200)`, then `add_to_cart(wireless-mouse-pro)`. The keyboard
    dispatches; the mouse is **held** (`off_turn_write_held` trace, WARNING). The held tool message
    is the conditional `_HELD_REASON["add_to_cart"]`: it contains "current message" and "do not
    add it".
13. **A cart view does not ground an add.** `view_cart` returns a line with `wireless-mouse-pro`,
    then `add_to_cart(wireless-mouse-pro)` → held.
14. **A cart view grounds a remove.** Same `view_cart`, then `remove_from_cart(wireless-mouse-pro)`
    → dispatched.
15. **No surfacing at all** (first call is `add_to_cart`, with or without thread context) → held.
    This replaces the old fail-open tests (no `turn_text` exists any more).
16. **Hallucinated/translated slug.** `filter_products` surfaced the catalog, then
    `add_to_cart("mouse-sem-fio-pro")` → held (not in the set); `add_to_cart("wireless-mouse-pro")`
    → dispatched.
17. **Language independence.** Thread trigger "Adicione o mouse sem fio, por favor". The flow
    `lookup_product_fact` miss with `catalog` list, then `add_to_cart(wireless-mouse-pro)` →
    dispatched. The pt-BR text never names the English product.
18. **Normalization on both sides.** Surfaced `wireless-mouse-pro`, call `"Wireless-Mouse-Pro "` →
    dispatched.
19. **Only successful dispatches surface.** A tool that raises a model-correctable `ServiceError`,
    or a held call, contributes no IDs. A non-JSON result (e.g. `place_order`'s empty-cart
    string) contributes none and does not raise.
20. **Dotted-key collection.** A `query_graph_data`-shaped result `{"items":[{"p.productId":
    "x"}]}` grounds `x` for `add_to_cart`. Nested dicts/lists are walked.
21. **Missing / empty / non-string `productId`.** Missing → existing required-arg re-prompt
    (unchanged). `""` or `5` → held (no fail-open on empty, unlike the old guard).
22. **K-061, ID form.** Two grounded identical adds in one turn → second held as a same-turn
    duplicate. `{"productId": "a"}` vs `{"productId": "A", "quantity": 1}` → one key (held). The
    same ID with `quantity` 1 then 2 → both dispatched. `{"productId": "a", "productName": "x"}` vs
    `{"productId": "a"}` → one key.
23. **K-061 remove semantics preserved.** `remove_from_cart(a)` (whole line) and
    `remove_from_cart(a, quantity=1)` → distinct keys, both dispatched.
24. **AC-9 test** (`test_cart_tool_dispatch_emits_no_extra_llm_call…`) is rewritten to surface the
    ID first. Its LLM-call accounting assertion is unchanged, adjusted for the extra iteration.

*Def (S4, `test_salesperson_scaffold.py`)*
25. `SALESPERSON_DEF["version"] == "v9"`; `config.model`, `tools`, `requiredTools`,
    `maxIterations`, topology equal `v8`'s values (assert literals). The prompt contains
    `productId`, "never translate", "current message", and "US dollars". It no longer contains
    "if a product name does not match anything in the catalog". The add sentence names no
    `view_cart` as an add source.

*Guard depth and scope (S3, review F1/F4)*
26. **Accepted depth, pinned.** Same setup as 12. After the mouse is held, the script continues:
    `lookup_product_fact("Wireless Mouse Pro")` (result carries `wireless-mouse-pro`), then
    `add_to_cart(wireless-mouse-pro)` → **dispatched**. This test documents that K-058 under ID
    grounding is one decision deep (§8 R3, accepted by the stakeholder with the mitigations). It is
    not a bug test.
27. **Per-node-execution scope.** One executor, two `_run_agent_node` calls (two turns). In the
    first, `lookup_product_fact` surfaces `x`. In the second, the first call is `add_to_cart(x)`
    with no lookup → **held**.

### 6.2 Named mutants (each must turn ≥1 test above red; run with `mutmut`-style hand edits, revert after)

| ID | Mutation | Killed by |
|---|---|---|
| M1 | `_GROUNDING_SOURCES["add_to_cart"]` gains `view_cart` | 13 |
| M2 | guard fails open on an empty `surfaced` set (old K-058 posture) | 15 |
| M3 | collector matches only the exact key `productId` (drop `.endswith`) | 20 |
| M4 | guard compares the raw value, not `normalize_product_id(value)` | 18 |
| M5 | the guard counts the call's own `productId` argument as surfaced (self-grounding) | 15 |
| M6 | dedup resolver keeps raw `productId` / extra arguments | 22 |
| M7 | `catalog_index` / lookup miss includes `price` | 6, 11 |
| M8 | `add_cart_item` calls `ensure_*` before resolving the ID | 3 |
| M9 | `remove_from_cart` dedup collapses omitted `quantity` to 1 | 23 |
| M10 | `surfaced` is hoisted to `self` (outlives the node execution; shared across customers on the singleton executor) | 27 |

*Ownership:* M1–M6, M9 and M10 belong to S3, and M7–M8 to S2 (§5 table). Pass 1's M5 ("collect
from failed or held results too") was an **equivalent mutant**, and is replaced. I verified this
against `_handle_tool_call`: none of those paths returns a `productId` key. The hold JSON is
`{held, reason}`, the AC-6 and malformed-argument results are plain strings, and so is the
absorbed-`ServiceError` result. Test 19 stays as a no-crash check on non-JSON content.

### 6.3 Live check (`qa-engineer`, after S1–S5; K-079 F1 is already in at `ac39345d`)

Two conditions on the same deployment. C1 is the gated TP-007 rerun. C2 measures the guard's
re-fire protection, which TP-007 cannot observe: its five turns contain a single add.

**Setup:** `seed_salesperson.sh demo` (publishes `v9`), `verify_salesperson.sh demo` exits 0,
then `start_demo.sh` (now pinned `v9`; the same model set and settings as U3: ministral `-c 8192`
plus the embedder, `qwen` absent, `:8200` idle). **C1:** the TP-007 probe, same shape as the baseline:
`salesperson/scripts/language_salience_probe.py run --trials 75 --base-url http://127.0.0.1:8000
--lmstudio-url http://localhost:1234 --out <scratch>/k078.jsonl` (default `PARALLEL 4`,
`TURN_WORKERS` 4).

**Measurement:** a read-only outcome script (`salesperson/scripts/cart_outcome_analysis.py`, new,
`qa-engineer`). It maps each probe participant (`participantId`, `language` from the JSONL turn
records) to `ws:demo` `CartItem`s (`(:Cart {customerId})-[:HAS_ITEM]->`) and `OrderLine`s
(`(:Customer {customerId})-[:PLACED]->(:Order)-[:HAS_LINE]->`) via `GRAPH.RO_QUERY` only. It
counts M1–M3 per language (definitions in §2; M3 sums a participant's mouse quantity across
`CartItem` **and** `OrderLine`, so an order ×1 plus a re-added cart line ×1 counts) and greps the demo log for `off-turn write held` /
`same-turn` holds. This is exactly how §2's baseline was computed.

**C1 done-condition, against the §2 baseline (n=75 per language).** The bar is the
stakeholder's (§8): es and pt-BR each ≥ 60/75, en ≥ 65/75, no gap gate.

| Measure | Baseline | Gate |
|---|---|---|
| M1 pt-BR | 1/75 [0.2, 7.2] | **≥ 60/75** (Wilson lower ≈ 69.6%) |
| M1 es | 0/75 [0.0, 4.9] | **≥ 60/75** |
| M1 en | 70/75 [85.3, 97.1] | **≥ 65/75** (no material regression) |
| M2, other-product lines, all languages | 0/225 | **≤ 2/225**, each hand-inspected and explained |
| M3, total mouse quantity > 1 (cart + order) | 0/225 | **≤ 2/225**, each hand-inspected |
| Dead turns | 0/1,125 | ≤ 1.2% (the U3 pass line) |
| Turn p95 (all) / turn index 2 p95, pooled | 14.1 / 14.1 s | ≤ 29 s each (the U3 line). **Report turn-2 p95 per language too** (baseline en 20.1, pt-BR 10.1, es 10.1 s): en has the least headroom |
| Wrong-language replies | 0/375 per language | **≤ 3/375 per language** (about the upper bound of 0/375's Wilson interval, 1.0%) |

Reported, not gated:
- K-058 holds per language, and their recovery rate (a held add followed by a dispatched add of
  the same product in the same turn);
- K-061 holds;
- the tool-call mix on turn 2;
- for K-079 F3, non-`$` currency symbols (baseline 7 replies);
- for the review's F6, the K-079 RCA's name→price corpus sweep (baseline 31 wrong of 79 stated
  pairs). This watches whether the miss list's price-free names invite invented prices.

**C2: add-then-add re-fire condition** (review F1; ml §9.1's design, minus the conditions K-058
never targeted). Drive conversations through the same storefront API as the probe
(`/shop/api/session` with `language`, `/shop/api/messages`), reusing `language_salience_probe.py`'s
plumbing in a small new driver, `salesperson/scripts/refire_probe.py` (`qa-engineer`). Each
conversation is a fresh participant with two scripted English turns:
- `same-category-immediate`: "Add 1 Wireless Mouse Pro to my cart." → "Also add 1 Mechanical
  Keyboard K200."
- `distinct-immediate`: "Add 1 Wireless Mouse Pro to my cart." → "Also add 1 Portable SSD 1TB."

That is 12 conversations per script per language, so **n = 24 per language, for en and pt-BR**.
pt-BR is the path where a miss list grounds the whole catalog. es is optional, at the same n.
Ground truth comes from the graph only, never from reply text: a **re-fire** is a participant
whose final mouse quantity, summed over `CartItem` and `OrderLine`, is > 1. Also count turn-2
`off-turn write held` lines per conversation.

Report each language against ml §9.2's pre-guard 1/24 (Wilson [0.7, 20.2]). **Reported, not gated**
(R3 is accepted, §8). Escalation trigger: **≥ 3/24 in either language** (Wilson
lower bound 4.3%, above the pre-guard point estimate) goes back to the stakeholder before K-078
closes.

*Why 60/75:* it puts es/pt-BR within about 13 points of en's baseline, and its Wilson lower bound
(69.6%) is far above both baselines' upper bounds (7.2%, 4.9%). A pass is unambiguous at this n.
Parity with en (≥ 65/75) is the stretch goal, not the gate. If pt-BR/es lands between 30 and 59,
that is a partial fix: report the held/recovery breakdown, and the fallback lever is §8 R2.

## 7. Documentation impact

| Document | Change | Who |
|---|---|---|
| `falkor-chat/docs/QUERIES.md` §15.1 | the abstention sentence (`{"found": false}`) gains the tool-level `catalog` list (the query itself is unchanged). The "`productId` in the projection (K-053)" note: `add_cart_item` now resolves by `productId` via `lookup_products_by_id`, no longer from a customer-named product | implementer (S6) |
| `falkor-chat/docs/QUERIES.md` §16 (≈ the `ensure_customer` note and the `Services.add_cart_item`/… list) | wording only where it implies name resolution | implementer |
| `falkor-chat/AGENTS.md` | `seed_salesperson.sh`, `verify_salesperson.sh` and `start_demo.sh` rows: `v8` → `v9`. **Replace, don't append** (root context-file rule) | implementer |
| `falkor-chat/README.md` (≈ "driven by `salesperson@v8`") | `v9` | implementer |
| `salesperson/README.md` (bring-up section: `v8` pin, `seed_salesperson.sh` comment, `FALKORCHAT_TRIGGER_DEF_VERSION=v8`) | `v9` | implementer |
| `falkor-chat/scripts/{start_demo,seed_salesperson,verify_salesperson}.sh` header comments | `v9` plus a one-line `v9` entry where the header lists versions | implementer (S5) |
| `server/falkorchat/proof_defs.py` `SALESPERSON_DEF` comment block | a `v9` note: why (K-078 + K-079 F3), what changed (two prompt edits), carried-forward fields | implementer (S4) |
| `executor.py` / `tools.py` / `services.py` docstrings and comment blocks | as listed in §5 S2/S3. The K-058 rationale moves from "text" to "provenance" in place, not as an appended paragraph | implementer |
| `falkor-chat/docs/HISTORY.md` | delivery entry for K-078 (+ K-079 F3) | `teco` at close |
| `falkor-chat/docs/BACKLOG.md` | remove K-078. K-079: F3 marked done by this change, F2 sequenced after it (§4) | `teco` |
| `falkor-chat/docs/test-reports/salesperson-cart-product-id-report.md` | new, the §6.3 C1 + C2 runs | `qa-engineer` |
| `salesperson/scripts/cart_outcome_analysis.py`, `salesperson/scripts/refire_probe.py` | new instruments, committed with the report, never left in a scratchpad (U3's P3-1 lesson) | `qa-engineer` |
| `falkor-chat/docs/SERVER.md`, `DESIGN.md`, `docs/manuals/*` | **no change.** Checked by grep: none describes the cart tool arguments or the K-058 guard | — |

## 8. Risks & open questions

**Risks.**
- **R1: the model guesses slugs instead of copying.** Catalog IDs are English-name slugs
  (`wireless-mouse-pro`), so a model can type a plausible one, or a translated one
  (`mouse-sem-fio-pro`), without any lookup. Effect: a guessed correct slug is held until surfaced
  (one extra iteration). A translated slug is held too, never written. Correctness is safe by D3;
  the cost is latency, measured in §6.3. Opaque IDs would remove the temptation but mean
  re-keying every `CartItem`/`OrderLine` in every workspace. That is out of scope, and only worth
  revisiting if §6.3 shows a hold-recovery problem.
- **R2: the model fails to pick from the `catalog` list,** or reports "not found" after the miss
  instead of picking. Fallback levers, in order: a stronger `v10` prompt sentence; then return the
  list on the cart tools' held message too, so recovery needs no separate lookup (this widens the
  grounding hole of D3, so measure M2 again).
- **R3: K-058 under ID grounding is one decision deep, not a wall** (D3 residual, review F1).
  A held re-fire passes if the model ignores the conditional reason and looks the product up.
  After a pt/es miss list, or a whole-catalog listing, nothing is held for the rest of the turn.
  The remaining defence is the model's compliance with "only what the customer asked for in their
  current message", which the hold reason and the `v9` prompt both state. Measured by §6.3 C2,
  against ml §9.2's pre-guard 1/24, and pinned offline by test 26. **The stakeholder
  accepted R3 with these mitigations** (conditional hold message, test 26, C2). A different
  mechanism is not designed here; candidates if C2 escalates (≥ 3/24 in either language, back to
  the stakeholder before close): a hold that cannot be lifted within the same node execution, or
  grounding that excludes whole-catalog results. Both cost legitimate recoveries.
- **R4: more iterations per add turn,** so latency rises. Turn-2 p95 is gated at ≤ 29 s. The
  concurrency overflow risk (U0) grows only with prompt size: the catalog list adds about
  250 tokens and only on a miss.
- **R5: in-flight conversations on `v8`** get the new tool schema with an older prompt. This is
  harmless on the dev box (the demo is restarted on `v9`), but `ws:acme`'s older salesperson
  versions behave the same way. Rollback: revert the commit and re-pin `v8`. `v9`'s stored snapshot
  stays in `reference`/`ws:demo` (create-only), unused. No data migration in either direction.
- **R6: textual adjacency in `tools.py`.** K-079 F1 landed at `ac39345d`
  (`_coerce_price_bound` sits directly after `LookupProductFactTool.run`). S2 must branch from
  `ac39345d` or later. There is no semantic overlap (§4).

**Stakeholder decisions (2026-10-22, relayed by `teco`).** Both are settled; neither blocks any step.
1. **R3 is accepted with the mitigations:** the conditional hold message, test 26, and C2. A C2 result
   of ≥ 3/24 in either language goes back to the stakeholder before K-078 closes.
2. **Gate bar:** es and pt-BR each ≥ 60/75, en ≥ 65/75 (§6.3). There is no gap gate.

## 9. Ready to implement

Implement the plan at `falkor-chat/docs/plans/salesperson-cart-product-id.md`, S1→S5 (S2+S3
in one commit, S4+S5 in one commit, branching from `ac39345d` or later, where K-079 F1 already
landed), then S6 docs, then S7 live check by `qa-engineer` (C1 gated, C2 reported, §8 decisions).

- Cart tools take `productId` (required). `productName` is gone; services resolve by ID through
  the existing `lookup_products_by_id`. No Cypher, schema or data change.
- The model gets IDs from current-turn tool results. `lookup_product_fact`'s miss now lists the
  catalog (ID and name only), plus a hint to re-look-up for a price, so a translated name still
  leads to a real ID.
- The K-058 guard grounds on provenance: a write's `productId` must have come from a grounding
  tool's result in this node execution (per-turn local state, never on the singleton executor).
  `view_cart` grounds removes, not adds, and the prompt and schemas say so per tool. The hold
  message is conditional ("only what the customer asked for now"). K-061 dedup keys on the
  honoured, normalized arguments, with `quantity` passed through exactly for removes.
- `salesperson@v9`: two prompt edits (ID guidance; K-079 F3 "US dollars"), everything else
  carried. Pins move in three scripts and three docs.
- Done when es and pt-BR each reach ≥ 60/75 filled carts (baseline 1 and 0), en ≥ 65/75
  (baseline 70), and wrong-product or doubled purchases stay ≤ 2/225 (the stakeholder's bar, §8). The C2 re-fire rate is reported against the pre-guard 1/24.
