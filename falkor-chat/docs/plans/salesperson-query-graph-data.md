# `query_graph_data` row identity and category-miss recovery

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-079 (F2), K-086 · **Version:** 3 · **Extends:** `docs/plans/workflow-nl-query-generation.md`, `docs/plans/workflow-nl-query-generation-ml.md` · **Reviews:** `docs/reviews/salesperson-query-graph-data.md`

## 1. Goal & scope

Two fixes on the one seam between `QueryGraphDataTool.run` (`server/falkorchat/tools.py`) and
`querygen` (`server/falkorchat/querygen.py`):

- **F2 (K-079, RC-1):** a `query_graph_data` result that lists per-row attributes always carries
  the column saying which entity each row is about. That closes the "two unkeyed lists zipped by
  position" misjoin in `docs/reviews/salesperson-fabricated-prices-rca.md` §3–§4.
- **K-086 (DEF-C):** a catalog `query_graph_data` question that names a category in another
  language, or names a category as if it were a product, no longer ends in
  `{"items": [], "finding": "no matching data found"}`. That finding is what the model turns into
  "we have no peripherals" (`docs/test-reports/salesperson-cart-product-id2-report.md` DEF-C).

**Out of scope:** K-079 F4/F5, K-085, K-087. No `Product` property is added to the query-facing
allowlist (`productId` in particular, §3.2). No `compile()` change, no graph/DDL/Cypher-library
change, no `salesperson` def bump (§4.5). The `knowledge_base` dataset gets F2 but not K-086's
recovery or category listing.

*2026-10-28: v1.*
*2026-10-28 v2: revised per `analyst` Pass 1 (B1, M1, M2, m1–m7, n1–n3); stakeholder decisions recorded (§3.2, §9).*
*2026-10-28 v3: Pass 2 N1 (root-anchored Run 0 guard/export), N2 (stub lookup keys), N3 (`browse-type` pt-BR only), R7 widened; nothing else changed.*

## 2. Context & findings

CPG: considered, not relevant — `cpg_falkorchat` is stale (built 2026-10-15, 18 server commits behind, including K-078's `tools.py` changes); every claim below was read from the current tree by grep/read.

**The hand-off today** (`QueryGraphDataTool.run`). Dataset lookup → an internal `step` LLM call
with `_build_query_request_system_prompt(schema)` → `extract_own_line_json_object(…,
object_pairs_hook=querygen.merge_duplicate_keys)` → `querygen.default_returns` →
`QueryRequest.model_validate` → `querygen.compile` → `services.run_structured_query` → `{"items":
rows}`. A parse, validation or compile failure, and an empty result, all return the same
`{"items": [], "finding": "no matching data found"}`.

**What lets RC-1 through.** `QueryRequest._returns_shape` and `compile()` accept any projection
list. Nothing ties a projection to the row's entity. `default_returns` fills only an **absent**
`returns`, and the RCA's failing completions sent `"returns": ["p.price"]` explicitly. I rechecked
both mechanisms in source: `default_returns` returns early on `"returns" in payload`, and
`compile` has no identity notion at all. `compile()` emits `RETURN DISTINCT` for every
non-aggregate projection, and the tuple-distinct `WITH` shape when `order_by` is outside
`returns`. That is why `RETURN DISTINCT p.price` collapsed the two $89.99 rows (RCA t36/t45).
`compile()` binds every `=` value **raw**, so `category`/`name` match case-sensitively. Only
`*Normalized` properties are normalised (the `endswith("Normalized")` branch). This matters for
§4.2.

**The registry** (`querygen.DatasetSchema`, `CATALOG_SCHEMA`, `KNOWLEDGE_BASE_SCHEMA`). It is
`labels: {label: {prop: type}}` and nothing else. `Product` exposes `name, nameNormalized,
category, price`. `productId` exists on the node but is **not** allowlisted. The `knowledge_base`
labels are `Entity {entityId, name, nameNormalized, type}`, `Document {documentId, title,
sourceFormat}` and `Chunk {chunkId, text, seq, documentId}`. `Document` and `Chunk` have no
`name`. `default_returns` recognises a single-entity lookup through the module constant
`_NAME_LOOKUP_PROPERTIES = {name, nameNormalized}` with `op "="`.

**What K-078 decided** (`docs/plans/salesperson-cart-product-id2.md` D2/D3/D8; `docs/HISTORY.md`
K-078 entries).
- A catalog row shown to the model carries `productId` **together with** `name`:
  `Services.catalog_index` is `{productId, name}`, and `filter_products` rows are `{productId,
  name, category, price}` (`docs/QUERIES.md` §15.2).
- `productId` is the identity for **cart writes**. The executor grounds a write only on an ID
  that a `_GROUNDING_SOURCES` tool returned in the same node execution
  (`executor._collect_target_values`, which also matches a dotted key such as `p.productId`).
  `query_graph_data` is already in both `_GROUNDING_SOURCES` rows. The comment there anticipates
  "a future row carrying `p.productId` (K-079 F2)".
- The frozen `salesperson@v9` system prompt (`proof_defs.SALESPERSON_DEF`) tells the model to copy
  a `productId` "from a `lookup_product_fact` or `filter_products` result". It never mentions
  `query_graph_data` as an ID source.
- D8: on an empty `filter_products` result whose `category` is unknown
  (`Services.has_category`, the repository's own match, no price bounds), `FilterProductsTool.run`
  returns `FILTER_CATEGORY_MISS_FINDING` + `FILTER_CATEGORY_MISS_HINT` + `catalog`
  (`catalog_index`). D8 rejected adding a `categories` list because the §4 token margin was
  *projected* thin. The id2 run then **measured** a C1 2-way peak of 5,871 tokens of 8,192, a
  2,321-token margin (report, "Reported measures" 1).

**DEF-C mechanics** (report §"Browse variant"). All 24 turns called only `query_graph_data`. The
internal completion filtered `category = "periféricos"` (es) or `name = "periféricos"` (pt-BR).
The 4 correct pt-BR turns were the ones whose *outer* question was English. The querygen call has
never been told what the category values are: `_describe_dataset_schema` lists property
**names** only. So copying the customer's word is the only move it has.

**The Layer 1 contract** (`server/tests/eval/nlq_scoring.py` module docstring, set-extraction
rule). A set pair flattens **every value of every column**, and "a query that returned an extra,
unrequested column … fail[s] … FR-2's contract is to return exactly what was asked". A scalar pair
needs exactly one row with exactly one column. The archived ml note
(`docs/plans/workflow-nl-query-generation-ml.md` §3 "Layer 1") defines a set answer as "an
unordered set of identifiers (item names / entity names)". The scorer's "exactly what was asked"
is stricter than the method note it implements.

**Golden set** (`server/tests/eval/nlq_golden_set.jsonl`, 40 pairs: 21 `catalog`, 19
`knowledge_base`). The last ministral run on record (`docs/HISTORY.md`, K-066 `nlq-12` fix) scored
**catalog 21/21, knowledge_base 12/19**. Its per-pair JSON was in a session scratchpad and is not
in the tree. The tracked `tests/eval/nlq_eval_results.json` is a 39-pair **qwen** run from
2026-08-30 with no `toolCompletion` field. It is no baseline for this change.

**Test seams.** `tests/test_tools.py` `StubServices` already stubs `has_category`, `catalog_index`
and `lookup_product`. Its `run_structured_query` returns the same canned rows for every call, and
many tests assert exactly one call (`[call] = svc.structured_query_calls`).
`tests/test_query_graph_data_routing.py` has its own `StubQueryServices`, which has
`run_structured_query` only. `tests/test_querygen.py` and `tests/test_querygen_live.py` call
`compile()` directly, so they are untouched by a change outside `compile()`.

## 3. F2 design — row identity

### 3.1 The rule (canonical statement; later sections cite it)

**D1. A per-row attribute projection always carries the row's identity.** After validation and
before `compile()`, `querygen.ensure_row_identity(request, schema)` returns `request` unchanged
**unless all four** of these hold, in which case it returns a request whose `returns` is
`[f"{var}.{identity[label][0]}", *request.returns]`, re-validated through
`QueryRequest.model_validate`:

1. the match's `label` has a declared identity (`schema.identity[label]`, non-empty);
2. **no** `returns` entry is an aggregate. A mixed list (`["p.category", "count(p)"]`) is a grouped
   aggregate: its non-aggregate columns *are* the grouping key, and prepending `name` would regroup
   it per product and silently change every count;
3. **no** projected property is in `schema.identity[label]` (`name` or `nameNormalized` already
   identifies the row), and the request is **not a facet enumeration**. A facet enumeration means
   every projected property is in `schema.facets[label]` **and** the compiled shape is the plain
   `RETURN DISTINCT` branch, i.e. `order_by is None or order_by in returns`. Only that branch
   enumerates the value domain ("which categories are there?" → 8 rows). With `order_by` outside
   `returns`, `compile()` takes the tuple-distinct branch and de-duplicates on `(facet, order
   key)`. That returns **one unkeyed row per entity**: `returns ["p.category"], order_by
   "p.name"` gives 15 category rows in name order (review B1, verified on `reference`). This is
   RC-1's shape with `category` in place of `price`, so it gets the identity;
4. the match is **not a single-entity lookup**: no filter with `op "="` on a property in
   `schema.identity[label]`. "How much is X?" answers one fact about an entity the question
   already names, and the scalar golden pairs depend on that.

Prepended, never appended: the identity is each row's first key, which is how a reader parses a
list. The re-validation can only fail on `max_length=6`. No registered label has more than 4
properties and duplicates are rejected, so that cannot happen today. If a future schema makes it
happen, the tool abstains through its existing `ValidationError` branch, never crashes.

**D2. The registry declares identity and facets as data** (`docs/plans/workflow-nl-query-generation.md` §3.3: "add data, not compiler
code"). `DatasetSchema` gains two fields with empty defaults, so hand-built schemas in tests keep
working:

```python
identity: dict[str, tuple[str, ...]] = field(default_factory=dict)  # label -> identity props; [0] is prepended
facets: dict[str, frozenset[str]] = field(default_factory=dict)     # label -> small-domain classifier props
```

| Dataset / label | `identity` (first = prepended) | `facets` |
|---|---|---|
| `catalog` / `Product` | `("name", "nameNormalized")` | `{"category"}` |
| `knowledge_base` / `Entity` | `("name", "nameNormalized", "entityId")` | `{"type"}` |
| `knowledge_base` / `Document` | `("title", "documentId")` | `{"sourceFormat"}` |
| `knowledge_base` / `Chunk` | `("chunkId",)` | `{}` |

Invariant, pinned by a registry test: every label of every `DATASET_REGISTRY` schema declares a
non-empty `identity`, and every `identity`/`facets` property is in that label's `labels` entry.

**D3. One definition of "single-entity lookup".** `default_returns` today has its own
`_NAME_LOOKUP_PROPERTIES`. It and `ensure_row_identity` will share one private predicate,
`_is_identity_lookup(label, filters, schema)`, and `_NAME_LOOKUP_PROPERTIES` is deleted. The
predicate is true iff some filter has `op == "="`, `isinstance(property, str)` and `property in
schema.identity.get(label, ())`.

The `isinstance` guard is load-bearing (review m5). Today a raw completion with `"op": "="`,
`"property": ["name"]` and `returns` absent makes `default_returns` raise
`TypeError: unhashable type: 'list'` on the frozenset membership. `QueryGraphDataTool.run`
catches only `(ValidationError, ValueError)`, so the tool crashes instead of abstaining. A1 pins
this with a red-first test, and the guard makes it abstain.
`default_returns` operates on the raw dict, so it passes `(f.get("property"), f.get("op"))` pairs
the way it does now. Behaviour change: for `Product` none, since identity is exactly
`{name, nameNormalized}`. For `Entity`, an `=` filter on `entityId` with `returns` absent now
abstains instead of defaulting to `e.name`. That is benign and consistent: echoing back the name of
an entity looked up by id is the fabrication bait `default_returns`' docstring already excludes.

**D4. Wiring.** In `QueryGraphDataTool.run`, inside the existing `try`:
`request = querygen.ensure_row_identity(querygen.QueryRequest.model_validate({...}), schema)`,
then `compile(request, schema)`. The `except (ValidationError, ValueError)` is unchanged.
`compile()` is **not** modified. It stays the pure "compile exactly this request" function that
`test_querygen.py`/`test_querygen_live.py` pin, and the identity rule is a separate, separately
probed normalisation. `_QUERY_REQUEST_INSTRUCTIONS` is **not** changed for F2. The repair is
deterministic, so asking the 3B model to remember it would add prompt risk for no gain.

What the model sees, on the RCA's two failing completions:
- `returns ["p.price"], order_by "p.name"` → `RETURN DISTINCT p.name, p.price ORDER BY p.name ASC
  LIMIT $limit` → `{"items": [{"p.name": "27-inch 4K Monitor", "p.price": 349.99}, …]}`. Once
  identity is in `returns`, the order key is too, so the plain branch is taken instead of the
  tuple-distinct one.
- `returns ["p.price"]`, no order (t36/t45) → `RETURN DISTINCT p.name, p.price` → 15 rows. The
  `DISTINCT` no longer collapses the two $89.99 products.
- `returns ["p.name"]` (the list half) is unchanged, since `name` is identity. Pairing by position
  is no longer needed, because every price row names its product.

### 3.2 Why `name`, not `productId`: departing from the RCA's conditional

**Decided by the stakeholder, 2026-10-28:** `name` is F2's identity column. The review verified
that `Product.name` is unique, both in the data (15 nodes, 15 names) and by construction:
`seed_catalog.sh` `MERGE`s on `productId = slugify(name)` under a UNIQUE constraint (review S1).
`Entity` names are not unique (unfused entities). That does not reopen RC-1: `Entity`'s only
non-identity property is a facet, and a `[name, type]` row is keyed even when names repeat.

The RCA says "if K-078 puts `productId` into rows shown to the model, F2's identity column should
be the same one". K-078 did, but always **next to `name`** (`catalog_index`, `filter_products`).
In K-078 the two do different jobs: `name` tells the reader which product a row is about, and
`productId` is the key for cart writes. RC-1 is a reader-side misjoin, and `name` is what fixes
it. Adding `productId` as well was weighed and rejected:

- **It needs an allowlist change.** `productId` is not in `CATALOG_SCHEMA`. Adding it lands in the
  outer tool description (`_describe_all_datasets`, in every agent call's context), and the
  querygen model could then filter or project on it. That is new model-facing surface beyond F2.
- **The frozen `v9` prompt does not endorse it.** The model is told to copy `productId` from
  `lookup_product_fact`/`filter_products` only. Making `query_graph_data` an endorsed ID source
  means a `v10` (create-only `systemPrompt`, `falkor-chat/AGENTS.md` trap 1). This argument is
  weaker than it reads: the prompt would not stop a model copying a `p.productId` it was shown.
  The grounding point below is the one that decides it.
- **It widens the grounding residual (decisive).** With `query_graph_data` in `_GROUNDING_SOURCES`, every
  list question would ground every listed ID for the rest of the turn. That is D8.3's accepted
  residual extended to the most common browse path, while K-085 (es re-add via a grounded ID) is
  still open.
- **It breaks Layer 1.** Every `p.name` list pair would gain slugs and fail exact-set match unless
  the scorer learned to drop them.
- **Cost:** about +8 tokens per row on every list result.

*Reversal trigger:* a measured need to add from a `query_graph_data` list in the same turn
(e.g. K-085's redesign, or a storefront probe where adds after a `query_graph_data` browse are
held at a visible rate). That is a `v10` + allowlist change. `_collect_target_values` already
handles `p.productId`. Adding `productId` to `identity` would **not** reach D7's trigger (b),
which is bound to the name properties explicitly (§4.2, review m6). A reversal therefore has no
second place to change. The `_GROUNDING_SOURCES` comment that promises it to F2 is corrected in
§7.

### 3.3 Alternatives rejected

- **Reject identity-free shapes in `compile()` (abstain).** That turns the RCA's price question
  into "no matching data found", which is the DEF-C dead end. A repair beats a refusal when the
  repair is unambiguous.
- **Pre-validation dict rewrite next to `default_returns`** (the RCA's placement). It has to
  re-derive shapes from unvalidated JSON (non-string entries, malformed matches) that Pydantic
  already rejects. Post-validation on a typed `QueryRequest` is smaller and cannot see garbage.
- **Prompt-only ("always include `<var>.name`").** Probabilistic, and it is exactly the instruction
  the 3B model already half-follows. The RCA asked for a contract, not a nudge.
- **A composite `Chunk` identity `(documentId, seq)`** (review n1). D1 prepends one column and
  treats any projected identity property as sufficient. With a composite, `seq` alone would
  wrongly count as identity. `chunkId` is the only unique key (UNIQUE constraint in
  `bootstrap_schema.sh`). An opaque but correct key beats an unkeyed list. It costs ≈ 20 tokens a
  row, and only on the `knowledge_base` dataset, which the storefront does not reach.
- **Exempt `limit == 1` superlatives** ("price of the cheapest product" → `p.price` LIMIT 1).
  Rejected: a lone price with no product is the RC-2 invitation, and no golden pair has that shape
  (the superlatives `nlq-16/17/20` already return `p.name`).

### 3.4 D5 — the Layer 1 contract amendment (exact text)

The `nlq_scoring.py` module docstring's set-extraction rule gains this paragraph (D5), and the
"return exactly what was asked" sentence is qualified by it:

> **Row identity (K-079 F2, `docs/plans/salesperson-query-graph-data.md` §3.1).** The mechanism
> returns exactly the requested columns, **plus the label's identity column, prepended, whenever
> the request projects per-row attributes without one** (D1's four conditions). A projection of
> facet properties only, such as `category` or `type`, is a value enumeration and stays exempt,
> but only without an `order_by` outside `returns`: that ordering makes it per-row. A golden pair whose
> question asks for a per-row attribute list ("prices of the Audio products") must therefore list
> the identity values (names) in `expected.values` as well. A scalar pair must be a
> single-entity lookup (an `=` filter on an identity property), an aggregate, or a superlative
> that projects the identity itself, because any other single-attribute shape now returns two
> columns. This restores the ml note's own definition of a set answer as a set of identifiers
> (`docs/plans/workflow-nl-query-generation-ml.md` §3).

**No scorer code change** and no golden-row change. By the case analysis in §8.1, every one of
the 40 pairs keeps its shape when the model emits the shape it emitted before. The review
confirmed this by execution: 40/40 canonical completions compile byte-identically with and
without D1, and none of 76 completion variants flips from correct to incorrect (review S2).
`test_nlq_golden_set_integrity.py` gains nothing: the rule is about the mechanism, not the
fixture.

## 4. K-086 design — category-miss recovery

DEF-C has two halves. The querygen call **does not know the category values**, so it copies the
customer's word. And an empty result is **indistinguishable from "we have none"**. One decision
addresses each: prevention (D6) and a deterministic backstop (D7). Both are catalog-only.

### 4.1 D6 — the catalog's categories are listed in the internal querygen prompt

- **New service method** `Services.catalog_categories(ctx) -> list[str]`: the sorted, distinct
  `category` values of the existing whole-catalog `repository.filter_products(category=None,
  min_price=None, max_price=None, limit=CATALOG_INDEX_LIMIT)` read. That is the same read and the
  same truncation posture as `catalog_index`, so there is **no new Cypher**. Today that is 8
  values: Accessories, Audio, Cameras, Displays, Peripherals, Smart Home, Storage, Wearables.
- **Prompt seam:** `_build_query_request_system_prompt(schema, *, categories: list[str] | None =
  None)`. With `categories` given, it appends one block after the schema line:

  > `Product.category is exactly one of: Accessories, Audio, Cameras, Displays, Peripherals, Smart
  > Home, Storage, Wearables. When the question names one of these categories, in any language,
  > filter on category with the listed value exactly as written here, never on name and never with
  > the question's own word. When the question names a specific product, keep filtering on its
  > name.`

  *v2 (review m3):* "or in other words" is dropped. It invited mapping a product-type word
  ("mouse") or a named but absent product (`nlq-14`'s "Wireless Earbuds Pro" → `Audio`) onto
  the closed list, and "in any language" is all DEF-C needs. The last sentence keeps the existing
  `_QUERY_REQUEST_INSTRUCTIONS` rule (a named item filters on `name`) in force rather than
  competing with it. "Exactly as written" matters because `compile()` matches `category` raw and
  case-sensitively (§4.2).

  With `categories=None`, the prompt is **byte-identical to today**, which pins the
  `knowledge_base` path.
- **Call site:** `QueryGraphDataTool.run` calls `self._services.catalog_categories(ctx)` only when
  `arguments["dataset"] == "catalog"`, before the LLM call, and passes the result in. An empty
  list (empty catalog) passes `None`.
- **Why the dataset-name check is explicit, not driven by `schema.facets`:** the values come from
  a catalog-specific service, and listing `Entity.type` for the `knowledge_base` dataset would move
  KB Layer 1 for a dataset no DEF-C customer touches. Generalising through `facets` is the obvious
  next step if K-081 (`nlq-27`, the `Product` label used for an `Entity.type` question) wants it.
  This change leaves it for later on purpose.
- **Where the cross-language match happens:** in the model, against a closed list. That is K-078's
  posture ("the cross-language match happens in the model, never in our code").
- **Cost:** about +45 tokens on the internal querygen request only (≈690 tokens today, RCA
  Appendix A), not on the agent's conversation. Plus one `ro_query` of ≤ 50 rows on `reference`
  per catalog call.

### 4.2 D7 — an empty catalog result that names an unknown category or product gets a recovery

`QueryGraphDataTool.run`, catalog dataset, `rows == []`, and D6's `categories` list non-empty.
Let `filters = request.matches[0].filters`; D1 never changes filters.

**What each trigger must test.** It must use the matcher the empty query used, and nothing
looser. `compile()` binds `category = $p` and `name = $p` **raw and case-sensitive**. It
normalises only `*Normalized` properties (the `filt.property.endswith("Normalized")` branch).
Live, `category = "peripherals"` and `name = "wireless mouse pro"` return 0 rows (review M1). A
normalising check (`has_category`, or `lookup_product` alone) would call those values "known",
fall through to "no matching data found", and recreate DEF-C on a case variant. So:

- **(a) unknown category:** some filter has `property == "category"`, `op == "="`, and
  `value not in categories`. `categories` is D6's list of raw stored values, already read once for
  this call. This is the same raw equality `compile()` used.
  - **This drops v1's reuse of K-078 D8.1's `has_category`.** D8 is correct for `filter_products`,
    because there the repository normalises the query *and* the check alike. Here the two
    matchers differ, and D7 must follow the query's. D8 itself is unchanged.
  - *Accepted limit:* the list inherits `CATALOG_INDEX_LIMIT` truncation (§9 R5). A category
    missing from a truncated list yields a needless but harmless recovery, never a dead end.
    This is moot at 15 products.
- **(b) unknown name**, bound to the **name** properties explicitly
  (`_CATALOG_NAME_PROPERTIES = {"name", "nameNormalized"}`, a `tools.py` constant), **not** to
  `schema.identity`. F2 asks which column labels a row; (b) asks whether `lookup_product` can
  resolve the value as a product name (review m6). For each filter with `op == "="` on one of
  them, take `row = self._services.lookup_product(ctx, name=str(value))`. The trigger fires when:
  - `name` filter: `row is None or row["name"] != value`. The raw `=` failed. `Product.name` is
    unique and `normalize_name` collisions collide in the slug (review S1), so `row["name"] ==
    value` holds exactly when some product's name equals `value` raw. This is the reviewer's fix,
    and it is complete for `name`.
  - `nameNormalized` filter: `row is None`. `compile()` already normalised this value, and
    `lookup_product` applies the same `normalize_name` (idempotent), so the two matchers agree. A
    hit means another filter emptied the result.

  This covers the pt-BR `name = "periféricos"` half. It also catches a translated or case-variant
  product name, which server code cannot tell apart from a category word, so the hint routes both.
- **Empty catalog** (review n3): when `categories` is empty, no recovery. That gives the old
  abstention, because "these are all the catalog's categories" over `[]` would be false. This is
  moot at 15 products.

Otherwise the result is the existing `{"items": [], "finding": "no matching data found"}`,
byte-identical. That covers a known category (exact, as stored) emptied by a price bound, a
product whose exact name exists but is filtered out by another condition, a filter-free or range-only empty result, and every parse, validation
or compile abstention. Each check runs at most once per trigger and only on an empty result.

The recovery shape (two new module constants beside `FILTER_CATEGORY_MISS_*`):

```python
QUERY_CATALOG_MISS_FINDING = "no catalog category or product has that exact name"
QUERY_CATALOG_MISS_HINT = (
    "these are all the catalog's categories; if the customer meant one of them, in any "
    "language, call filter_products with that exact category; to find one "
    "product, call lookup_product_fact with its name"
)
# result:
{"items": [], "finding": QUERY_CATALOG_MISS_FINDING, "hint": QUERY_CATALOG_MISS_HINT,
 "categories": <the same list D6 read for this call>}
```

- **`items: []` stays**, so Layer 1's `not_found` pairs (`nlq-13/14/15`) still score as empty
  (`score_pair` reads `items` only).
- **The finding avoids "no matching"**, the phrase models turn into "we have none" (K-078 R2).
- **The retry goes to `filter_products`, not `query_graph_data`.** `filter_products(category=
  "Peripherals")` is deterministic (no second LLM translation that could miss again). Its rows
  carry `name` and `price` together, so they are misjoin-free. Its `productId`s are the ones the
  `v9` prompt endorses for an add. A product-name miss is routed to `lookup_product_fact`, which
  owns K-078 D2's catalog-list recovery. Both tools are already granted in `v9`.
- **`categories` reuses D6's list**, read once per call. No extra read.
- **No `catalog` list, deliberately.** D8 attached `catalog_index` because a `filter_products`
  category miss is most often a *product* sent as a category. Here the ambiguity runs the other
  way. Without `catalog` the recovery is ≈ 80 tokens instead of D8's ≈ 445, and **the recovery
  result itself** grounds no `productId` (contrast D8.3).
  - **Following the hint does ground IDs** (review m7). `filter_products(category=…)` grounds
    every `productId` in that category for the rest of the node execution
    (`_GROUNDING_SOURCES`), as any `filter_products` browse does. Before this change those
    turns ended having grounded nothing. This is endorsed `v9` behaviour, and the exposure is
    one category's IDs, not the catalog's.
  - The browse probe starts from an empty cart, so it cannot see the effect. Measuring it belongs
    to K-085's re-measure, not here.

### 4.3 Reusing K-078's recovery vs alternatives

| Option | Verdict | Why |
|---|---|---|
| **Reuse D8 verbatim** (`FILTER_CATEGORY_MISS_*` + `catalog_index`, `has_category` as the test) | rejected | The hint is `filter_products`-specific ("so this is every product in the catalog instead"). `catalog_index` has no categories, so a browse can only present the whole list or guess which items are peripherals. That is D8's own reversal-trigger class. Also, `has_category` normalises while `compile()`'s `=` does not, so it would miss case variants (§4.2, v2). **Kept from D8:** `items: []` on the miss, and a hint that carries its own instruction because `v9` names only the lookup's list. |
| **Recovery only, no prompt listing** (D7 without D6) | rejected as sole fix | Every translated browse would cost an extra model iteration plus a hint-following step from a 3B model. D6 removes the miss at its source for the es shape (`category = "periféricos"`, right property, wrong value). D7 stays as the deterministic backstop. |
| **Steer category browsing to `filter_products`** (tool descriptions or prompt) | rejected | A prompt change is a `v10`. A description-only nudge still lands on D8's catalog-without-categories recovery for a translated word, and DEF-C shows the model choosing `query_graph_data` 24/24. |
| **Server-side synonym table or embedding match of categories** | rejected | A second, hand-maintained translation layer. It contradicts K-078's "match in the model", and it is unbounded per language. |
| **Add `category` to `catalog_index`** | rejected | It changes K-078 D2's deliberately minimal list for every lookup miss, and it is ~15 × 4 tokens on a path that doesn't need them. |

### 4.4 Interface changes, called out

- `query_graph_data`'s model-facing result gains a recovery branch (D7: new `finding` text,
  `hint`, `categories`).
- `_build_query_request_system_prompt` gains a keyword argument.
- New `Services.catalog_categories`.
- `DatasetSchema` gains `identity`/`facets` (D2).
- New `querygen.ensure_row_identity` (D1).
- F2 changes the **column set** of some `query_graph_data` results (a prepended identity column).

No REST/MCP route, graph, query-library, DDL or def change.

### 4.5 No `salesperson` def bump

Trap 1 in `falkor-chat/AGENTS.md` is about step `config` (`config.tools`, `systemPrompt`,
`config.model`), which is create-only. This design changes none of them:
- `config.tools` is a list of tool **names** (`proof_defs.SALESPERSON_DEF`), unchanged;
- tool result shapes, tool-module constants, `_QUERY_REQUEST_INSTRUCTIONS` and the querygen prompt
  builder are server code read at call time;
- the hint routes only to tools `v9` already grants (`filter_products`, `lookup_product_fact`);
- `v9`'s "If `lookup_product_fact` finds no exact match, it lists the catalog" sentence is
  untouched, and D7 carries its own instruction, as D8 did.

`./scripts/verify_salesperson.sh demo` must still exit 0 after deployment (§8.3). A def bump
becomes necessary only if §3.2's reversal trigger fires.

## 5. Interactions between F2 and K-086

1. **Two predicates, deliberately not one** (v2, review m6).
   - D3's `_is_identity_lookup` (over `schema.identity`) decides F2's exemption and
     `default_returns`' exemption. Those two must agree.
   - D7's trigger (b) asks a different question: can `lookup_product` resolve this value as a
     product name? So it is bound to `_CATALOG_NAME_PROPERTIES = {name, nameNormalized}`.
   - Today the two sets coincide for `Product`. If §3.2's reversal ever adds `productId` to
     `identity`, F2 changes and (b) does not, which is correct: `productId = "x"` must never reach
     `lookup_product(name="x")`.
   - The overlap that matters: every F2-exempt `=` name lookup that comes back empty is a
     (b) candidate. That is the intended `nlq-13`-style path (§9 R3).
2. **F2 cannot move D7's trigger.** Prepending a column to a `DISTINCT` projection can only split
   rows, never remove them. An empty result stays empty and a non-empty one stays non-empty, so
   D7 fires on exactly the requests it would fire on without F2. F2 never touches `filters`.
3. **`category` plays two roles.** It is an F2 *facet* (`RETURN DISTINCT p.category` with no
   outside `order_by` keeps enumerating 8 values, not 15 name/category pairs; with `order_by
   p.name` it is per-row and gets the identity, D1 condition 3) and D6/D7's *value domain*. The two are
   declared separately on purpose (`schema.facets` vs `Services.catalog_categories`): one is a
   static registry fact, the other live data. The registry test (D2) keeps `facets["Product"]`
   a subset of the allowlist. Nothing ties it to D6's call-site check, so the source comment at
   that check names the facet.
4. **The recovery's retry path is misjoin-free twice over.** D7 routes to `filter_products`, whose
   rows carry `name`+`price`. If the model retries `query_graph_data` instead, F2 guarantees the
   identity column.
5. **Grounding.** Neither part surfaces a `productId` from `query_graph_data`: F2 picks `name`
   (§3.2), and D7 carries no `catalog`. `_GROUNDING_SOURCES` behaviour is byte-identical. The
   turn D7 steers to `filter_products` does ground that category's IDs (§4.2, m7), as any
   `filter_products` browse does. K-085 owns measuring that.
6. **Layer 1 attribution.** F2 is deterministic given a completion. D6 changes the catalog
   completions themselves, since the prompt changes. The live re-run therefore cannot by itself
   say *which* part moved a pair. §8.1 separates them: a replay of the recorded completions with
   and without `ensure_row_identity` isolates F2 exactly, and whatever the replay does not
   explain is D6 or model variance.
7. **Token arithmetic.** F2 adds a name column only on shapes that lacked one: ≈ +8 tokens/row,
   ≤ 20 rows by default, i.e. on the RC-1 price list ≈ +120 tokens. D7 plus one `filter_products`
   follow-on ≈ +80 + 4×35 ≈ +220, on a path that today ends the turn wrongly. Both are well inside
   the measured 2,321-token C1 margin (§2). D6 adds ≈ 45 tokens to the internal call only.

## 6. Step table

Two stages, each one implementer run, dispatchable separately. **Stage A must land first.** Both
stages edit `QueryGraphDataTool.run` in `tools.py`, so they are serialised. Since v2 Stage B no
longer depends on D3's predicate (§5 item 1). Stage C is verification
(`qa-engineer`) after both. Paths are relative to `falkor-chat/server/` unless rooted. Run suites
with `server/` as the working directory (root `AGENTS.md`).

### Stage A — F2 (K-079)

| Step | What | Files | Done when |
|---|---|---|---|
| A1 | **Red-first probe.** (The `default_returns` `entityId` test from §6.2 lands here too.) D2 registry data (`identity`, `facets` fields plus the §3.1 table values). D3: `_is_identity_lookup`, with `default_returns` switched to it and `_NAME_LOOKUP_PROPERTIES` deleted. `ensure_row_identity` added as a **pass-through** (`return request`). The registry-invariant test. The coverage probe (§6.1). | `falkorchat/querygen.py`; `tests/test_querygen.py` | Probe **red on its assertion** (not on an import), with exactly **Product 299, Entity 66, Document 34, Chunk 840 (1,239)** failing cases out of
44,323 (§6.1). The `default_returns` unhashable-`property` test (D3, m5) is red too. Every other test green, including all `default_returns` tests in `test_tools.py`. Not committed alone: A1+A2 are one commit |
| A2 | Implement D1 in `ensure_row_identity`. Wire D4 into `QueryGraphDataTool.run`. Tool-level tests T-A1…T-A5 (§6.2). | `falkorchat/querygen.py`; `falkorchat/tools.py`; `tests/test_tools.py` | Probe green (44,323 cases). The m5 test and T-A1…T-A6 green. Mutants MA1–MA9 each turn ≥ 1 test red. Full offline suite green |
| A3 | Contract and comments: D5's paragraph in the `nlq_scoring.py` docstring. Correct the `_GROUNDING_SOURCES` comment in `executor.py`, which promises `p.productId` "(K-079 F2)": say F2 chose `name` and point to §3.2's reversal trigger. Update the `QueryGraphDataTool` class docstring for D1. | `tests/eval/nlq_scoring.py`; `falkorchat/executor.py` (comment only); `falkorchat/tools.py` (docstring) | `test_nlq_scoring.py` green (no code change). `git diff` of `executor.py` touches comments only |

### Stage B — K-086

| Step | What | Files | Done when |
|---|---|---|---|
| B1 | `Services.catalog_categories` (D6) | `falkorchat/services.py`; `tests/test_services.py` | T-B1, T-B2 green; MB1 killed |
| B2 | D6 prompt seam plus the catalog-only call site. `StubServices` gains `catalog_categories` (constructor kwarg `categories=None` → `[]`, with a call counter). `StubQueryServices` in the routing test gains the same method returning `[]`. | `falkorchat/tools.py`; `tests/test_tools.py`; `tests/test_query_graph_data_routing.py` | T-B3…T-B5 green. Every pre-existing `query_graph_data` test green **unmodified** apart from the stub additions |
| B3 | D7 recovery: two constants, the empty-result branch, the class docstring | `falkorchat/tools.py`; `tests/test_tools.py` | T-B6…T-B19 green; MB2–MB9 killed; full offline suite green |

Stage A touches 5 files (+1 comment-only), Stage B 5 files. Docs (§7) are each stage's last
commit, by the same implementer.

### Stage C — verification (`qa-engineer`)

§8 in full: offline gates, the Layer 1 live re-run plus the F2 replay on ministral, and the
`browse` re-measure. Report: `docs/test-reports/salesperson-query-graph-data-report.md`.

### 6.1 The coverage probe (RCA §5 "Prevention test")

`tests/test_querygen.py::test_every_attribute_projection_carries_row_identity`, parametrised over
`(dataset_name, label)` for **every** label of **every** `DATASET_REGISTRY` schema. It reads the
registry live, so a label added later is covered automatically. Each parametrised case loops
internally, collects failures, and asserts the list is empty, printing the first 10.

- **Axes** (var `v`; `props` = the label's properties, sorted; v2 widened per review M2):
  - **`returns`:**
    - every non-empty subset `S` of `props`, as `v.<p>`;
    - each such `S` with one aggregate appended, from
      `AGGS = [count(v), count(v.<p0>), avg(v.<p0>), min(v.<p0>), max(v.<p0>)]`, where `p0 =
      props[0]`. Every function appears, so a `startswith("count(")` detector dies;
    - each aggregate of `AGGS` alone.
  - **`order_by`:** `None`, or `v.<p>` for each `p` in `props`. That covers in-`returns` and
    outside-`returns`, i.e. both compile branches.
  - **`filters`:**
    - none;
    - every `p` in `props` × all six ops, as a single filter, with a value of `p`'s declared type
      (`"x"`, `1.5`, `2`);
    - every two-filter list `[= <non-identity p>, = <identity q>]`, which puts the lookup in
      **second** position, for every non-identity `p` and **every** identity `q`.
- **Size:** 44,323 requests: Product 13,775, Entity 13,300, Document 3,948, Chunk 13,300. They
  are pure `compile()` calls, 1.3 s measured. *Derivation:* (`2^n − 1` subsets × 6, plus 5)
  returns × `(n + 1)` orders × (`1 + 6n + |non-id| × |id|`) filters. For example, Product
  (`n = 4`, 2 non-identity, 2 identity) is 95 × 5 × 29.
- **Oracle**, written in the test from §3.1's text, *not* by calling the implementation and *not*
  by restating the v1 exemption list. v1's oracle copied condition 3 and so certified B1's hole.
  This one defines the facet exemption from what the **compiled shape** is:

  `expect_prepend = ¬any aggregate ∧ ¬any identity prop projected ∧ ¬(plain ∧ all projected
  props are facets) ∧ ¬any "=" filter on any identity prop`, where `plain = order_by is None or
  order_by in returns`.
- **Assertion** on `cols = _returned_columns(compile(ensure_row_identity(req, schema),
  schema).cypher)`:
  - if `expect_prepend`, then `cols == [f"v.{identity[0]}", *req.returns]`;
  - otherwise `cols == req.returns` **and** the cypher equals `compile(req, schema).cypher`
    (byte-identical).
  - Universal clause, the RCA's own invariant: every non-aggregate, non-lookup case that is not a
    *plain* facet enumeration projects some identity property.
- **`_returned_columns(cypher)`**, a test helper: take the text after the **last** `"RETURN "`,
  drop a leading `"DISTINCT "`, cut at `" ORDER BY "`/`" LIMIT "`, split on `", "`, and map each
  item to its backticked alias when it contains `` AS ` `` (the tuple-distinct branch), else to
  itself. This reads the compiled text, which is the RCA's "every *compiled* multi-row shape".
- **Red-first count** (A1, pass-through): exactly the `expect_prepend` cases fail. That is
  **Product 299, Entity 66, Document 34, Chunk 840 = 1,239**. If A1's count differs, stop: the
  oracle, the axes or the registry table disagrees with §3.1.
  - *Recomputed* by my own enumeration with a prototype of the v2 D1. The scratch script is
    `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6bf0232-c1ae-4f33-8aa9-de25f4a9b94a/scratchpad/probe/probe2.py`,
    output in `probe2.out` next to it.
  - It agrees with the review's 1,239. The review's Document **total** (3,276) differs from mine
    (3,948) because its aggregate axis differs. The red count does not, since aggregate cases are
    never `expect_prepend`.
  - Under v1's narrow axes the same oracle gives 159 (Product 39, Entity 9, Document 6,
    Chunk 105).
- **B1 × M2, checked together.** On the v2 D1 prototype, the same 44,323 cases kill:
  - the facet-ignores-order mutant: 169 failures (Product 69, Entity 66, Document 34);
  - aggregate-detected-as-`count(`-only: 5,328;
  - lookup-checks-`identity[0]`-only: 55;
  - lookup-checks-`filters[0]`-only: 170;
  - append-instead-of-prepend: 1,239.

  Every one of these was green on v1's 2,024 cases except the last. The two fixes compose
  correctly for two reasons:
  - D1 tests the aggregate exemption **before** the facet one. A facet + aggregate + outside
    `order_by` request stays exempt (a grouped aggregate) and is never prepended. That shape fails
    at execution regardless: FalkorDB rejects `ORDER BY` on a non-projected variable after
    aggregation, verified on `reference`. This is pre-existing and noted in §9 R7.
  - The widened `=`-on-facet filters × outside order are exactly where B1's new reds sit, so the
    narrowed exemption is exercised under every filter shape.

### 6.2 Behaviour tests, red → green order

**Stage A** (`tests/test_tools.py`, through `QueryGraphDataTool.run` with `StubQueryLLM` and
`StubServices`):
- **T-A1** — the RCA's verbatim second completion (`returns ["p.price"], order_by "p.name"`)
  compiles to `MATCH (p:Product) RETURN DISTINCT p.name, p.price ORDER BY p.name ASC LIMIT
  $limit`.
- **T-A2** — `returns ["p.price"]`, no order (t36/t45) → `RETURN DISTINCT p.name, p.price LIMIT
  $limit`.
- **T-A3** — `returns ["p.category", "count(p)"]` → compiled unchanged (grouped aggregate).
- **T-A4** — `= name` lookup with `returns ["p.price"]` → unchanged (`nlq-01` shape). The same
  holds for `nameNormalized`.
- **T-A5** — the KB `Entity` `= nameNormalized` + `returns ["e.type"]` case
  (`test_query_graph_data_workspace_scoped_dataset_resolves_graph_key_from_ctx`'s reply) →
  unchanged. Also `returns ["e.type"]` with no filter → unchanged (facet enumeration).
- **T-A6** (B1) — `returns ["p.category"], order_by "p.name"` → `RETURN DISTINCT p.name,
  p.category ORDER BY p.name ASC LIMIT $limit`. Prepended: that `order_by` makes it per-row.
  `returns ["p.category"], order_by "p.category"` → unchanged.
- **`default_returns`** (in `test_querygen.py` or `test_tools.py`, wherever its tests live):
  - an `Entity` `= entityId` filter with `returns` absent now abstains (D3's one behaviour change);
  - **m5, red today:** a raw completion `{"matches": [{"var": "p", "label": "Product",
    "filters": [{"property": ["name"], "op": "=", "value": "x"}]}]}` (no `returns`) through
    `QueryGraphDataTool.run` returns the abstention instead of raising `TypeError`.

**Stage A mutants** (each must turn ≥ 1 test red):
- MA1 — drop condition 2 (aggregates) → probe aggregate variant, T-A3.
- MA2 — drop the facet clause → probe, T-A5.
- MA3 — drop the lookup clause → probe, T-A4.
- MA4 — append instead of prepend → probe, T-A1.
- MA5 — use `identity[label][-1]` → probe.
- MA6 — remove the `ensure_row_identity` call from `run` → T-A1/T-A2.
- MA7 — `default_returns` keeps a literal `{name, nameNormalized}` → the `entityId` test.
- MA8 — facet exemption ignores `order_by` (v1's rule) → probe (169), T-A6.
- MA9 — drop the `isinstance(property, str)` guard and hold `identity` as a frozenset → the m5
  test.

**Stage B:**
- **T-B1** (`test_services.py`, `FakeRepo`) — `catalog_categories` returns sorted, distinct
  values (duplicate categories in the fake rows collapse).
- **T-B2** — it makes exactly **one** `filter_products` read with `category=None`, no bounds and
  `limit=CATALOG_INDEX_LIMIT`.
- **T-B3** — `_build_query_request_system_prompt(CATALOG_SCHEMA)` with no `categories` equals
  `_QUERY_REQUEST_INSTRUCTIONS.format(dataset_schema=_describe_dataset_schema(CATALOG_SCHEMA))`
  (byte-identity pin). With `categories=["Audio", "Peripherals"]` it ends with D6's block,
  naming both.
- **T-B4** — a catalog `run` sends a system prompt containing the stub's categories, and
  `catalog_categories` is called once.
- **T-B5** — a `knowledge_base` `run` calls `catalog_categories` **zero** times, and its system
  prompt equals the T-B3 no-categories form for `KNOWLEDGE_BASE_SCHEMA`.
Stage B's stub setup: `StubServices(categories=[…8 real values…])`. `lookup_product` is the
stub's exact-key `dict.get` (`products_by_name`), and it stays that way. Do **not** make the stub
normalise: existing lookup tests key on raw names, and B2 requires them unmodified.

A test that needs the repository's normalising lookup keys **both** the raw and the normalised
name to the same row, whose `name` is the real one:

`ROW = {"productId": "wireless-mouse-pro", "name": "Wireless Mouse Pro", …}`,
`products_by_name={"Wireless Mouse Pro": ROW, "wireless mouse pro": ROW}`

T-B10, T-B17 and T-B18 all use this dict. Each then exercises the trigger, not the stub's keying
(review Pass 2 N2).
- **T-B6** — es DEF-C completion, verbatim shape: `category = "periféricos"`, rows `[]` →
  exactly `{"items": [], "finding": QUERY_CATALOG_MISS_FINDING, "hint": QUERY_CATALOG_MISS_HINT,
  "categories": <stub list>}`. `catalog_categories` is called **once** for the whole run, and
  `category_checks == []`: `query_graph_data` never calls `has_category`.
- **T-B7** — pt-BR DEF-C shape: `name = "periféricos"`, rows `[]`, `lookup_product` → `None` →
  the same recovery.
- **T-B8** — `nameNormalized = "periféricos"` → recovery.
- **T-B9** — a known category, exact as stored (`category = "Peripherals"`), emptied by `price
  < 1` → `{"items": [], "finding": "no matching data found"}`, byte-identical.
- **T-B10** — a known exact name (`name = "Wireless Mouse Pro"`) + `price < 1`, rows `[]`,
  with the both-keys dict above. `lookup_product("Wireless Mouse Pro")` hits the raw key and
  `row["name"] == value` → old abstention.
- **T-B11** — no filters, or a range only, empty → old abstention, and `lookup_product` is not
  called.
- **T-B12** — an unparseable reply → old abstention, no checks called.
- **T-B13** — `knowledge_base`, `= name` filter, empty → old abstention, no checks called.
- **T-B14** — a non-empty result → no checks called.
- **T-B15** — the finding text contains no `"no matching"` (the R2 phrase), and the hint names
  `filter_products` and `lookup_product_fact`.
- **T-B16** (M1) — a case variant of a real category, `category = "peripherals"`, rows `[]` →
  recovery. It must not fall through to the old abstention.
- **T-B17** (M1) — a case variant of a real name, `name = "wireless mouse pro"`, rows `[]`, with
  the both-keys dict. `lookup_product` hits the normalised key, and the row's `name` is
  `"Wireless Mouse Pro"` → recovery (raw `!=`).
- **T-B18** (m6) — `nameNormalized = "wireless mouse pro"` + `price < 1`, rows `[]`, with the
  both-keys dict. `lookup_product` hits the normalised key → old abstention. The normalised matchers agree, so the product
  exists and another filter emptied the result.
- **T-B19** (n3) — `StubServices(categories=[])`, `category = "periféricos"`, rows `[]` → old
  abstention, no recovery.

**Stage B mutants:**
- MB1 — `catalog_categories` not deduplicated → T-B1.
- MB2 — drop the dataset check, so the KB gets the recovery → T-B13.
- MB3 — trigger (a) uses `has_category` (v1's design) instead of list membership → T-B16 (and
  T-B6's `category_checks == []`).
- MB4 — trigger (b) tests only `lookup_product is None` on a `name` filter → T-B17.
- MB5 — read `catalog_categories` a second time for the recovery → T-B6's call count.
- MB6 — run the checks before the empty-rows test → T-B14.
- MB7 — revert the finding to the generic text → T-B6.
- MB8 — trigger (b) uses the raw `!=` comparison on a `nameNormalized` filter too → T-B18.
- MB9 — recover even when `categories` is empty → T-B19.

## 7. Doc impact

| Surface | Change | Who / when |
|---|---|---|
| `server/tests/eval/nlq_scoring.py` module docstring | D5's paragraph. The "return exactly what was asked" sentence is qualified to cite it | implementer, A3 |
| `server/falkorchat/executor.py` `_GROUNDING_SOURCES` comment | Replace "so a future row carrying `p.productId` (K-079 F2) grounds with no further change" with: F2 chose `name`, `query_graph_data` surfaces no `productId` today, and the dotted-key match is kept for §3.2's reversal trigger | implementer, A3 |
| `server/falkorchat/querygen.py` | Module docstring: one line saying the identity rule lives in `ensure_row_identity`, not `compile()`. `DatasetSchema` docstring: document `identity`/`facets` | implementer, A1/A2 |
| `server/falkorchat/tools.py` | `QueryGraphDataTool` class docstring: D1 (identity prepended), D6 (category listing) and D7 (recovery branch, triggers, shape), in the style of `FilterProductsTool`'s | implementer, A3/B3 |
| `docs/QUERIES.md` §15.2 | After the D8 sentence: `Services.catalog_categories` is built from this same unfiltered query (distinct `category`, no new Cypher), and `query_graph_data`'s catalog recovery tests a category against that list exactly, as `compile()`'s raw `=` does, and a name through §15.1's lookup (plus a raw `name` comparison). The queries themselves are unchanged | implementer, B3 |
| `docs/QUERIES.md` (NLQ shapes) | **None.** `querygen`-compiled shapes are not in the library, and that is unchanged | — |
| `docs/DESIGN.md` | **None.** No graph, index or capacity change | — |
| `docs/SERVER.md` §1.8 | **None.** The model-resolution seam is unchanged (the facet read is a `services` call, not a model call) | — |
| `docs/plans/workflow-nl-query-generation.md` (archived) | Header pointer only: `Extended by: docs/plans/salesperson-query-graph-data.md` (a header pointer is the one edit permitted on an archived document, root `AGENTS.md`). Same for `-ml.md`, whose Layer 1 set rule D5 restores | `teco`, at close |
| `docs/BACKLOG.md` | K-079 narrowed to F4/F5. K-086 removed, once its `HISTORY.md` entry exists. K-081 gains one line saying D6's listing is the candidate mechanism if generalised through `schema.facets` (§4.1) | `teco`, at close |
| `docs/HISTORY.md` | Delivery entry: F2 + K-086, the Stage C numbers, citing this note and the report | `teco`, at close |
| `falkor-chat/AGENTS.md` | **None.** No new trap: no def bump, and the bump traps are unchanged | — |
| `salesperson/scripts/refire_probe.py` | One `SCRIPTS` entry, `browse-type`: "Vocês têm algum mouse?" (one turn), for §8.3's reported product-type measure (review m3). It is run in **pt-BR only**, because the probe sends one text to every language (review Pass 2 N3). `browse` itself is unchanged | `qa-engineer`, Stage C |

## 8. Verification plan

Order: §8.1 offline (implementer, every step) → §8.2 Layer 1 + replay (`qa-engineer`,
LM Studio) → §8.3 storefront browse (`qa-engineer`, demo up). §8.2 and §8.3 need the
stakeholder's go for LM Studio time (RCA F2 done-condition). Neither may run while `:8200` is
busy (`falkor-chat/AGENTS.md`, demo residency).

### 8.1 Offline (gate for both stages)

- `cd falkor-chat/server && .venv/bin/python -m pytest -q`: green. **A default run wipes
  `reference`** — re-seed (`bootstrap_schema.sh` → `seed_demo.sh` → `seed_catalog.sh` →
  `seed_salesperson.sh` / `seed_workflows.sh`) before §8.2/§8.3, per `falkor-chat/AGENTS.md`.
- Probe: 44,323 cases green. A1's red count was exactly Product 299 / Entity 66 / Document 34 / Chunk 840 (1,239).
- Every MA*/MB* mutant killed (§6.2), reported per mutant.
- **Static golden-set case analysis** (recomputed here from `nlq_golden_set.jsonl`, not taken from
  the RCA). Assuming each pair's completion keeps the shape its question implies:
  - **Catalog:**
    - `nlq-01..05` single-fact: an `=` name filter, exempt;
    - `nlq-06..12, 40` lists: `returns ["p.name"]` (or `default_returns`), identity present;
    - `nlq-13..15` not-found: exempt and empty;
    - `nlq-16, 17, 20` superlatives: return `p.name`;
    - `nlq-18, 19` counts: aggregates.
  - **knowledge_base:**
    - `nlq-21..24`: `=` name/`nameNormalized`, exempt;
    - `nlq-25..27`: return `e.name`;
    - `nlq-28..30`: exempt and empty;
    - `nlq-31..33`: counts;
    - `nlq-34..39`: structurally unanswerable, 0/6 both before and after.

  So statically, **0 of 40 pairs change shape**. That is the claim §8.2's replay checks.

### 8.2 NLQ eval Layer 1 on ministral — the F2 done-condition

**Setup:**
- Same as the K-066 run (`docs/test-reports/salesperson-single-model-report.md` E2/Q2).
- Load exactly `mistralai/ministral-3-3b` (8192 ctx) plus the embedder.
- `reference` seeded with the catalog, and `ws:nlq-eval` seeded (`./scripts/seed_nlq_eval_corpus.sh`).
- A scratch copy of `config/models.json` with only `defaults.step` =
  `lmstudio/mistralai/ministral-3-3b`.

**Run 1 — post-change tree (both stages in):**

```bash
cd falkor-chat/server
FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json \
FALKORCHAT_MODEL_CONFIG=<scratch>/models-step-ministral.json \
NLQ_EVAL_RESULTS_PATH=<scratch>/nlq-ministral-after.json \
  .venv/bin/python tests/eval/run_nlq_golden_set_eval.py
```

It must print `step model: lmstudio/mistralai/ministral-3-3b`. Afterwards, from the **same cwd**
(`falkor-chat/server`), `git status --short tests/eval/` must print nothing, because the tracked
baseline must not be overwritten. (v1 gave a repo-root pathspec from `server/`, which git resolves
relative to the cwd. It always printed nothing, so the check was vacuous; review m1.)

**Run 0 — same-day pre-change baseline.** The same command, writing `nlq-ministral-before.json`,
on **the sha recorded in `docs/plans/salesperson-query-graph-data-coordination.md`** (`teco` pins
it when dispatching Stage A).
- Before running, check that no server change slipped in between. From any cwd:

  `git -C "$(git rev-parse --show-toplevel)" diff --quiet <sha> <Stage-A-first-commit>^ -- falkor-chat/server`

  Exit 0 means no change; 1 means stop. A bare repo-relative pathspec run from
  `falkor-chat/server` silently matches nothing and exits 0 (review Pass 2 N1). Anchoring with
  `-C <toplevel>` instead of the reviewer's `':/falkor-chat/server'` is deliberate. `:/` works for
  `git diff`, but `git archive` rejects it from a subdirectory: "pathspec … matches files outside
  the current directory", rc 128, checked. So both commands use one form.
- Export it with

  `git -C "$(git rev-parse --show-toplevel)" archive <sha> falkor-chat/server falkor-chat/config | tar -x -C <scratch>/before`

  `falkor-chat/config` is included because `config.py`'s default `models.json` path resolves
  relative to the package. Both config env vars are set explicitly anyway (review m2).
- That export changes neither the user's working tree nor `.git`. Run it with `<scratch>/before/falkor-chat/server` as cwd. Use
the same venv's interpreter. The runner puts its own `server/` first on `sys.path` (the
`sys.path.insert(0, str(_SERVER_DIR))` at the top of `run_nlq_golden_set_eval.py`), so the
exported `falkorchat` should win over the editable install. Verify it once: print
`falkorchat.__file__` with the same interpreter and `sys.path` order, and check it is under
`<scratch>`. This gives a per-pair comparator under the same runtime and model, since K-066's
per-pair JSON is gone. Diff with `salesperson/scripts/nlq_compare.py before.json after.json`.

**Replay — isolates F2 deterministically, no LM Studio needed.** A scratch script, not committed.
For each record of Run 1:
1. parse `toolCompletion` exactly as `QueryGraphDataTool.run` does (`extract_own_line_json_object`
   + `merge_duplicate_keys` → `default_returns` → `model_validate`);
2. compile it **with and without** `ensure_row_identity`;
3. execute both read-only through `Services.run_structured_query`;
4. score both with `score_pair`.

The difference between the two columns is F2's effect alone.

**Targets — all must hold:**

| # | Measure | Target | Prior |
|---|---|---|---|
| L1 | catalog Layer 1, Run 1 | **21/21** | 21/21 (K-066 rerun, `HISTORY.md`) |
| L2 | knowledge_base Layer 1, Run 1 | **Equal to Run 0 pair by pair, except differences the replay attributes to F2** (the KB path is otherwise byte-identical, and `config/models.json` pins ministral at `temperature: 0`). Floor: **≥ 12/19** | 12/19 ministral, 13/19 qwen |
| L3 | not-found false-answer rate, Run 1 | **0/6** | 0/6 |
| L4 | Replay: pairs correct without F2 and incorrect with it | **0/40** | static analysis says 0 (§8.1) |
| L5 | Replay: pairs whose compiled Cypher differs with vs without F2 | **reported** (expected 0 for golden questions); every non-zero one listed with its completion | — |
| L6 | Run 0 → Run 1 per-pair flips correct→incorrect | **0 on catalog.** Watch list, since D6's listing makes these the likeliest to move (review m3): `nlq-13/14/15` (not-found, e.g. "Wireless Earbuds Pro" mapped to `category = "Audio"`), `nlq-02/04` (the category of a named product) and the category lists `nlq-06/07/09/10/11/12/40/19/20`. Every catalog pair whose *completion* changed between runs is listed, flip or not. On KB, any difference is listed with both completions (see L2) | — |

**If L1 fails:** the replay says whether F2 caused it. If F2 did not, the failing catalog pair's
two completions show what D6's listing changed. That goes back to `architect`, as a D6 wording
question, before close.

### 8.3 Storefront browse re-measure — the DEF-C done-condition

**Setup:** `./scripts/start_demo.sh` (storefront, `salesperson@v9`, 2 workers, `ws:demo`). Then,
read-only, `./scripts/verify_salesperson.sh demo` must exit 0: no def drift, confirming §4.5.

```bash
python3 salesperson/scripts/refire_probe.py run --out <scratch>/browse.jsonl \
    --languages pt-BR,es --schedule browse --repeat 12
python3 salesperson/scripts/refire_probe.py run --out <scratch>/browse-type.jsonl \
    --languages pt-BR --schedule browse-type --repeat 12        # reported only, run separately; pt-BR only (the text is Portuguese)
```

**No English control, on purpose** (v2, review m4).
- The `browse` script sends the same text, "Do you have any periféricos?", in every session. An
  `en` session would measure an English session receiving a Portuguese/Spanish word, not an
  English category word.
- The probe runs one conversation per language concurrently. Adding `en` would put 3
  conversations on the demo's 2 turn workers, against id2's 2. That inflates `turn_ms` with
  queueing and breaks B7's comparison.
- The English-category path is covered where it lives, at the querygen layer: Layer 1 `nlq-06`,
  `07`, `09`, `10`, `11`, `12` and `40` all name an English category (L1, L6).

The pt-BR + es run therefore keeps id2's exact load and script. Keep the LM Studio server log for
the window, because mechanism attribution needs the querygen completions and the tool calls.

**Classification:** by hand, the same four classes as the id2 report's browse table:
- lists the 4 Peripherals (Wireless Mouse Pro, Mechanical Keyboard K200, Gaming Mouse Pad XL,
  Webcam HD 1080p), possibly with translated names, and nothing else presented as a peripheral;
- presents the whole catalog as peripherals;
- ends without an answer;
- says the store has none.

**Targets (pt-BR + es pooled, n = 24):**

| # | Measure | Target | Prior (id2, 2026-10-26) |
|---|---|---|---|
| B1 | Says the store has no peripherals | **0/24** (gate) | 19/24 |
| B2 | Lists exactly the 4 Peripherals | **≥ 20/24** (gate) | 4/24 |
| B3 | Whole catalog presented as peripherals, or no answer | **≤ 2/24** (gate) | 1/24 |
| B4 | Dead turns / wrong-language replies | **0 / 0** (gate) | 0 / 0 |
| B5 | `browse-type` ("Vocês têm algum mouse?", pt-BR sessions only, n = 12): replies that present non-mouse Peripherals (keyboard, webcam) as mice, or say there are none | **reported, ungated**. This checks that D6 maps only category words (m3) | not run |
| B6 | Mechanism: first querygen completion filters `category = "Peripherals"` (D6 worked) / D7 recovery fired / recovery followed by `filter_products(category="Peripherals")` | **reported** per language | 0 / n.a. / n.a. |
| B7 | Turn p95 for the browse turn, split by path (first-call success vs recovery → `filter_products`) | **reported** next to id2's | id2 browse window |

**B7 comparability, stated honestly.** The load matches id2: same two languages, same 2-way
concurrency, 2 turn workers. The **paths** do not. id2's turns were one `query_graph_data` call
and a reply. Here a recovered turn adds a model iteration and a `filter_products` call. So B7 is
reported per path. Only the first-call-success path is like-for-like with id2, and the recovery
path's extra time is the cost of D7, not a regression.

**Why these thresholds:** B2 ≥ 20/24 is a lower Wilson-95% bound of ≈ 64%, and DEF-C's 4/24 has
an upper bound of ≈ 36%, so a pass cannot be noise. B1's 0 is the defect itself: a customer told
a whole category doesn't exist. **If B1 > 0 or B2 < 20:** B6 says which layer failed (D6 not
mapping, or D7's hint not followed), and that goes back to `architect` before close. No re-tuning
in the QA pass.

**Not re-measured here, on purpose:** C1/C2 (K-085) and the RC-1 en price-list incidence (2/75 in
S7, too rare for a 24-conversation run). F2's guard is the offline probe plus L4.

## 9. Risks & open questions

- **R1 — D6's wording perturbs a catalog golden pair (L1/L3).** The listing is new text in the
  internal prompt, and a 3B model can over-apply it.
  - **Sharpest case: `nlq-14`** ("Which category does the Wireless Earbuds Pro belong to?"). It
    could be mapped to `category = "Audio"`, `returns ["p.category"]`. That returns a row, which
    fails the not-found pair (L3), and D7 (b) never sees it. `nlq-13/15` are the same shape.
  - Lesser cases: `nlq-02/04` (the category of a named product) and every category list.
  - A product-type word ("mouse") mapped onto Peripherals is the storefront analogue, measured by
    B5.
  - *Mitigation:* v2 drops "or in other words", keeps "a named product filters on `name`" in
    force, and L6 lists these pairs. The replay separates F2 from D6, and the Run 0/Run 1 diff
    names the pair. *Fallback:* drop D6 and keep D7. Recovery alone still fixes the "we have none" reply (B1),
  at the cost of one extra iteration per translated browse.
- **R2 — D7's hint is not followed.** The model may say "that category doesn't exist" instead of
  calling `filter_products`. B6 measures it. The `categories` list itself states that Peripherals
  exists, so B1 should hold even then. B2 is what would suffer.
- **R3 — trigger (b) fires on a genuinely absent product** (`nlq-13`-style). The reply then sees a
  category list and a pointer to `lookup_product_fact`, instead of a bare "not found". That is
  more helpful, and Layer 1 is unaffected (`items: []`). Layer 2 (non-gating) phrasing may move.
- **R4 — F2 changes a non-golden question's answer shape.** E.g. "price of the cheapest product"
  now returns name + price (§3.3, accepted). A `Chunk` text list gains opaque `chunkId`s. That is
  harmless, and the knowledge base is not in the storefront.
- **R5 — `catalog_categories` truncation.** It inherits `CATALOG_INDEX_LIMIT = 50` rows, so a
  catalog over 50 products could under-list categories. That is the same posture as
  `catalog_index`, and moot at 15 products.
- **R6 — Latency.** +1 `reference` read per catalog `query_graph_data` call (sub-millisecond at 15
  nodes). On the recovery path, +1–2 reads and +1 model iteration. B7 reports it.
- **R7 — pre-existing, out of scope.** Two classes of shape compile but fail at execution.
  - **An aggregate with an `order_by` outside `returns`** (e.g. `["p.category",
    "min(p.price)"]`, `order_by "p.name"`): "ORDER BY cannot reference variables not projected or
    grouped", verified on `reference`.
  - **`avg()` over a string property** (e.g. `avg(v.category)`): "Type mismatch: expected
    Integer, Float, or Null but was String". `compile()` accepts it.

  The review's execution sweep of the 44,323 probe cases found 16,932 failing both before and after
  D1: 14,860 of the first class and 2,072 of the second, with 0 ok→fail and 0 fail→ok. So D1 neither
  causes nor fixes either class.

  `QueryGraphDataTool.run` does not catch execution errors, so both reach the executor's fault net
  and kill the turn. They are one candidate backlog item for `teco`: either `run` abstains on an
  execution error, or `compile()` rejects both shapes.
- **Rollback:** each stage is one revert. No data, graph or def state is created, so nothing
  needs migrating back.

**Stakeholder decisions (2026-10-28) — both v1 open questions are closed:**
1. *§3.2:* F2's identity column is **`name`**, decided. The review confirmed `Product.name` is
   unique (S1). Reversal stays as §3.2's trigger: a `v10` plus an allowlist change, only on a
   measured need.
2. *§8.2/§8.3:* **LM Studio time is approved** for the live runs. That is Run 0 + Run 1
   (≈ 2 × 40 pairs × ~3 s), 24 `browse` conversations and 12 `browse-type` conversations.

No open questions remain. M1's trigger semantics, which the review routed to `architect`, are
decided in §4.2.

## 10. Implementer recommendation & ready-to-implement summary

**Implementer: `tdd-engineer` for both stages.** Each stage has an up-front behaviour contract, and
Stage A's centrepiece is a red-first probe whose red count is specified (§6.1). The RCA asks for
exactly that sequencing. Stage B is a small, test-pinned branch with mutants listed. Dispatch A,
then B (both edit `QueryGraphDataTool.run`, so they are serialised), then Stage C to
`qa-engineer`.

**Ready to implement** — `falkor-chat/docs/plans/salesperson-query-graph-data.md`:
- **Stage A (F2):**
  - `querygen.DatasetSchema` gains `identity`/`facets` (§3.1 D2 table).
  - A post-validation `querygen.ensure_row_identity` prepends the label's identity (`p.name`) to
    any non-aggregate, non-lookup projection lacking one, except a *plain* facet enumeration (no
    `order_by` outside `returns`) (D1).
  - It is wired into `QueryGraphDataTool.run` before `compile()`, which is unchanged.
  - The 44,323-case registry-driven coverage probe goes red first (1,239: Product 299 / Entity
    66 / Document 34 / Chunk 840), then green.
  - The m5 `TypeError` is pinned and fixed.
  - The Layer 1 contract is amended in the `nlq_scoring.py` docstring (D5). No scorer or golden
    change.
- **Stage B (K-086):**
  - New `Services.catalog_categories` (no new Cypher).
  - The catalog's categories are listed in the internal querygen prompt (D6).
  - An empty catalog result triggers the recovery when it has either:
    - an `=` category **not exactly in** that list (raw, as `compile()` matched);
    - an `=` name that does not resolve **exactly** (`lookup_product` miss, or a raw-name
      mismatch).

    The recovery returns `{"items": [], finding, hint, categories}`, routing to
    `filter_products`/`lookup_product_fact` (D7).
  - No def bump.
- **Stage C:**
  - ministral Layer 1: catalog 21/21, KB equal to Run 0 per pair except replay-explained
    differences (floor 12/19), false-answer 0/6, replay F2-caused flips 0/40.
  - Run 0 uses the sha pinned in the coordination doc.
  - `refire_probe.py --schedule browse` (pt-BR, es × 12, id2's load): "none" 0/24, lists-the-4
    ≥ 20/24.
  - `browse-type` reported only.

## 11. DEF-2 assessment (2026-10-28)

Post-Stage-C addendum; §1–§10 are unchanged. Input: test report
`falkor-chat/docs/test-reports/salesperson-query-graph-data-report.md` (DEF-2, §8.3 B5/B6) and the
raw joined turns `browse-type-joined.txt` / `browse-joined.txt` (QA scratchpad, `stageC/`). Code
read at `HEAD`: `server/falkorchat/tools.py` (`_QUERY_REQUEST_INSTRUCTIONS`,
`_CATALOG_CATEGORIES_INSTRUCTION`, `_build_query_request_system_prompt`, `QueryGraphDataTool.run`,
`_names_unknown_value`), `server/falkorchat/querygen.py` (`QueryFilter.op`, `ensure_row_identity`),
`config/models.json`, `server/tests/test_tools.py` (T-B3/T-B4). No live turns were run for this
section.

CPG: considered, not relevant — `cpg_falkorchat` is stale per the brief; the question is about
prompt text and recorded completions, read directly by grep.

### 11.1 Facts the assessment rests on (verified)

- **F1 — the querygen call is deterministic per question string.** `config/models.json` sets
  `temperature: 0` for `lmstudio/mistralai/ministral-3-3b`. The 4 DEF-2 turns (t1/t4/t7/t10) all
  sent the *same* internal question, "quantos produtos de mouse estão disponíveis?", and got
  byte-identical completions. The 2 other querygen turns (t5/t9) sent "tem algum mouse no
  catálogo?" and also matched each other. So B5's "4 of 12" is **one** querygen decision, repeated
  because the agent step chose that phrasing 4 times. It is not a 4/6 querygen error rate. The
  12-turn run measures the agent's routing mix (6 `lookup_product_fact`, 4 + 2 `query_graph_data`),
  and only **2 distinct querygen inputs**.
- **F2 — the value `"Peripherals"` comes from D6.** The pre-K-086 prompt has no category values:
  `_describe_dataset_schema` lists labels and property names only, and the examples in
  `_QUERY_REQUEST_INSTRUCTIONS` name no category. The question says "mouse". Without the D6 block,
  "Peripherals" can appear only if the model guesses it.
- **F3 — D6 does not touch the agent's routing.** D6 lives only in the internal querygen system
  prompt (`QueryGraphDataTool.run`). The outer tool descriptions are unchanged, and D7 fired 0
  times. The `filter_products(category="mouse")` miss result seen in 10 of the 12 turns is D8's
  hint (id2), which predates K-086. So the pre-change tree would have sent the same internal
  questions at a comparable rate.
- **F4 — the DSL has no correct query for "products that are a mouse".** `QueryFilter.op` is
  `= <> < <= > >=`, with no `CONTAINS`/prefix operator. Querygen can answer "which products are
  mice" only by knowing the product's exact name, and it doesn't. The best result available to it
  is a superset that carries names, such as `category = "Peripherals"` with `returns ["p.name"]`.
  Given those names, the reply step can pick out "Wireless Mouse Pro", which is how the `browse`
  turns succeed.
- **F5 — D7, and every abstention path, is blind to aggregate shapes.** `count(...)` over an empty
  match returns one row (`[{"count(p)": 0}]`), not `[]`, so `if not rows:` never fires. A count
  that misses on a category or name therefore reaches the reply as a confident 0, with no finding
  and no hint. That was true before K-086 as well.
- **F6 — F2 (Stage A) does not affect these completions.** `ensure_row_identity` skips any
  `returns` list that contains an aggregate. With `categories=None`, `HEAD`'s querygen prompt is
  byte-identical to the pre-change one, and T-B3's first half pins that.
- **F7 — the golden set contains no product-type-word pair.** Every catalog pair that reaches a
  `category` filter names the category literally: `nlq-06/07/09/10/11/12/19/20/40`, plus
  `nlq-02/04/14`, which ask for a named product's category. Run 0 and Run 1 completions for those
  pairs are identical, apart from `nlq-12`'s duplicate-`filters` repair. Layer 1 therefore guards
  any D6 rewording against regressions on *category* words. It says nothing about *type* words, in
  either direction.

### 11.2 Q1 — Is DEF-2 caused by D6?

**What D6 causes:** the filter value (F2). The completion `category = "Peripherals"`,
`count(p)` → 4 is D6's doing.

**What is not established:** that D6 made the *customer outcome* worse. There is no pre-change
`browse-type` run. On the pre-change prompt, the most likely completions for "quantos produtos de
mouse estão disponíveis?" are inferred, not observed:

| Pre-change shape (inferred) | Result | Likely reply | Compared with DEF-2 |
|---|---|---|---|
| `name = "mouse"` or `category = "mouse"`, `count(p)` | `[{"count(p)": 0}]`, no D7 (F5) | "não temos mouse": DEF-C's own failure class, now for a type word | different wrong, arguably as bad (a lost sale against a false stock claim) |
| `filters: []`, `count(p)` | 15 | "temos 15 modelos de mouse", or the t5/t9 hedge | worse in the first reading, equal in the second |
| `filters: []` with `p.name`, or anything that carries names | names | can pick the mouse | better: the only case in which D6 is a regression |

The id2 baseline makes the first row the most likely. id2 showed the pre-change model copying the
customer's word into an `=` filter (`name = "periféricos"`). t5/t9 in this run show that
`filters: []` with `count` is also in its repertoire. The questions use "quantos" and "tem algum",
and both phrasings pull towards `count`, which is a property of the question rather than of D6.

**Verdict:** D6 causes DEF-2's *specific* wrong answer ("4 mice"). Whether D6 caused a
*regression* is unknown, and on the evidence it more likely replaced one wrong answer
(none, or 15) with another (4). The wrong-answer *class*, a type word reaching querygen with a
`count` phrasing, predates K-086. Its structural causes are F4 and F5, not D6's wording.

**What would settle it cheaply (minutes, no conversations):** a one-off script modelled on the QA
replay script (`stageC/scripts/replay.py`), run from `falkor-chat/server`. For each of the 2
observed internal questions × {`categories=None`, `categories=<the 8>`}, × 3 repeats to confirm
determinism:
1. call the `step` model once with
   `_build_query_request_system_prompt(querygen.CATALOG_SCHEMA, categories=…)` as the system
   message and the question as the user message;
2. parse, validate, `ensure_row_identity`, `compile`, and execute on `reference`;
3. print the completion and the rows.

That is 12 querygen calls. No `e8798351` checkout is needed (F6): `categories=None` *is* the
pre-change prompt. The rows show directly whether pre-change was "0", "15", or names. If the names
are useful, the reply step could still be shown it, but the tool rows alone decide whether a
regression exists.

### 11.3 Q2 — Is there a D6 wording revision, and where is the defect really?

**Where the harm is.** Mapping "mouse" onto `Peripherals` is not wrong in itself. Under F4 it is
the best superset the DSL can express, and with `returns ["p.name"]` the reply would see the 4
names and could offer the mouse, as the `browse` turns do. The false claim needs **both** the
mapping **and** a `count` projection, which strips the names. The reply step then attributes the
bare number to the customer's noun. So the defect sits at the junction of the mapping and the
`count` shape. A fix aimed only at the category mapping targets the wrong half.

**Candidate A — narrow the mapping (R1's original framing). Rejected.** It appends: "A kind of
product (e.g. "mouse") is not a category; never map it onto one of these values." This removes
the "4 mice", but the likely fallback is `name = "mouse"` or `category = "mouse"` with `count` →
`[{"count(p)": 0}]`. F5 means D7 cannot see that, so the reply says "não temos mouse". That is
DEF-C's failure for a type word, and the plan exists to remove it. Candidate A also gives a 3B
model a fuzzy category-word/type-word boundary to apply to the same words that must keep mapping
(the browse 24/24), so the DEF-C gate itself is at risk. Confidence that it nets out better
than today: low.

**Candidate B — keep the mapping, fix the shape (preferred, if anything changes now).** Replace
`_CATALOG_CATEGORIES_INSTRUCTION` with:

> `Product.category is exactly one of: {categories}. When the question names one of these
> categories, in any language, filter on category with the listed value exactly as written here,
> never on name and never with the question's own word. When the question names a specific
> product, keep filtering on its name. When the question asks about a kind of product that is not
> itself one of these categories (for example "mouse"), filter on the category that contains it
> and return p.name, never count(...): that category also holds other kinds of product.`

The first three sentences are D6 v2 verbatim, so the category-word behaviour behind B6's 24/24 is
not reworded. Only one sentence is added. It sets a *shape* rule for the type-word case and
endorses the mapping the model already makes (F4). It does not forbid it. `nlq-19` ("How many
products are in the Peripherals category?") names a category, so the new sentence's condition
excludes it.

**Confidence that ministral-3B honours B:** low to medium (my estimate, roughly even odds per
question string).
- Mapping the type word to its containing category is already observed in 1 of the 2 strings.
- Overriding a "quantos"/"how many" pull towards `count` with a conditional instruction placed
  last is the uncertain part. The m1 history and t1/t4 of `browse` (`count(p.name)` on a category
  browse) show this model reaching for `count` readily.
- Over-application is a second risk. A named but absent product (`nlq-13/14/15`) could be read as
  "a kind of product" and mapped to `Audio`. That is R1's sharpest case, and Layer 1 L3 would
  catch it.

Because querygen is deterministic (F1), each question string is effectively pass/fail. An
offline probe over a small string set says more than 12 conversations.

**Candidate C — code, no prompt change (noted, not proposed now).** Two variants:
- When the result is aggregate-only, echo the applied filters into the result, e.g.
  `{"items": [{"count(p)": 4}], "matched": {"category": "Peripherals"}}`. The reply could then say
  "4 periféricos" instead of "4 mice". That is still not "1 mouse", and it changes the tool's
  result contract, which the reply prompts and the Layer 1 scorer read.
- Close F5: an aggregate-only miss would get the same finding/hint treatment as `rows == []`.
  This is the durable fix for the "confident 0" class, wherever it comes from, and it is what would
  make Candidate A safe.

Both are design items, not wording tweaks.

### 11.4 Q3 — What would verifying a revision cost?

For Candidate B, the cost in the existing harness. LM Studio time is small. The process loop is
the cost.

| Item | What | Size |
|---|---|---|
| Offline querygen probe (new, throwaway) | §11.2's script, run on the revised prompt over ~12–16 strings: the 2 observed mouse questions; 3–4 other type words, pt-BR/es/en, each phrased as "quantos/hay/how many" and "tem algum/do you have"; the 6 observed `browse` internal questions, as the category-word guard. × 2 repeats | ~30 querygen calls, ~2 min. **This is the real discriminator** (F1), and it runs before any conversation |
| Code | one constant in `tools.py` | 1 edit |
| Tests | `_categories_block` in `server/tests/test_tools.py` types D6's wording from the plan on purpose, and T-B3's second half (`…ends_with_the_category_block`) and T-B4 (`…lists_the_catalogs_categories_once`) assert against it. Update the helper; both go red, then green. T-B3's first half (no categories, byte-identical) is untouched. Optional: a mutant pinning the new sentence's presence | 1 helper edit, 0 new tests needed |
| Layer 1 (R1 guard) | One ministral run against Run 1 (`nlq_compare.py`). The block is in every catalog querygen prompt, so `nlq-13/14/15/19` and the `category` pairs (F7) are the regression surface | 40 pairs, ~2–3 min |
| `refire_probe.py --schedule browse` | pt-BR, es × 12; B1–B4 again. B2's margin is 1 turn and B3 sits at its limit, so this is not optional | 24 turns, ~1.5 min |
| `refire_probe.py --schedule browse-type` | pt-BR × 12; B5 | 12 turns, ~1 min |
| Mechanism capture | `refire_probe.py` records neither the tool chain nor the querygen completion. QA's ad-hoc proxy (`stageC/scripts/proxy.py`) has to be stood up again, or B6/B5 go unexplained | setup time |
| Gates | implementer unit, `analyst` code gate, `qa-engineer` re-run plus report pass, plan v3 note | one more Stage B→C cycle |

**Statistical caveat.** With F1, `browse-type` n = 12 yields about 2 distinct querygen inputs, and
which ones appear depends on the agent's phrasing, not on the fix. A green `browse-type` after
Candidate B is weak evidence, and an unchanged 4/12 could simply mean the agent never sent the
fixed phrasing. The offline probe carries the verification. The live runs confirm only that
nothing else moved.

### 11.5 Q4 — Recommendation: close K-086 on its gates and backlog DEF-2, after the §11.2 counterfactual probe

**Decision rule.**
- **First, run the §11.2 counterfactual probe.** It is 12 querygen calls, minutes, and changes no
  code.
- **Pre-change completion gives 0 or 15** (the expected outcome): D6 did not regress this case.
  It swapped one wrong answer for another. Close K-086 and backlog DEF-2.
- **Pre-change completion returns names the reply could use:** D6 introduced a regression.
  Fix it before close with Candidate B (§11.3), verified per §11.4.

**Reasoning for close and backlog in the expected case:**
1. **K-086's done-conditions passed.** DEF-C: B1 0/24 and B2 21/24. F2: no Layer 1 regression.
   B5 was planned as reported-only because this risk was foreseen (R1).
2. **The cause is structural, not D6's wording.**
   - The DSL cannot express "products of a kind" (F4).
   - Aggregate results bypass every abstention and D7 (F5).
   - The golden set has no type-word pair (F7).

   A wording patch addresses none of these three, and Candidate A would make F5 bite harder.
3. **A fix now puts the DEF-C result at risk on thin margins.** B2 is 1 turn above its floor and B3
   sits at its limit. Any change to the shared block re-opens both, for a confidence of roughly
   even odds (§11.3).
4. **The available live check is weak.** The 12-turn `browse-type` check is ~2 querygen inputs
   (F1). A "fixed" verdict from it would not be trustworthy without the offline probe.

**Proposed backlog item, for `teco` to word:** "A product-type question reaching
`query_graph_data` gets a misleading count." It has three parts:
- (a) Candidate B, or a better shape rule, validated by the offline string-set probe;
- (b) close F5, so that an aggregate-only result over a filter that names an unknown category or
  product gets D7's finding and hint;
- (c) add 2–3 type-word pairs to the NLQ golden set, so that Layer 1 guards the class in future,
  and record the step trace in `refire_probe.py`, as the report's feedback item 3 asks.

**What would change this recommendation:** the stakeholder judging the "4 mice" stock claim
materially worse than a "none" reply, for example as a customer-trust or legal exposure in the
demo. In that case, fix now with Candidate B and budget the full §11.4 cycle. Candidate A is not
the fix in either case.
