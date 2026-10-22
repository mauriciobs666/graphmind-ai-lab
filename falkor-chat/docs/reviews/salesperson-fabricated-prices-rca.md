# RCA — storefront replies state fabricated or scrambled prices

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-079

CPG: considered, not relevant — `cpg_falkorchat` is stale (12 server commits behind) and the causal
chain runs through two pure functions (`querygen.compile`, `QueryGraphDataTool.run`) plus logged model
traffic, all verified directly against source, a pure-function compile and read-only Cypher.

**Scope.** Symptom: K-079 (`docs/BACKLOG.md`), evidence from
`docs/test-reports/salesperson-single-model-report.md` Q3. Baseline: the Q3 run (2026-10-22
09:00–09:49, code at `ad01aee7`, `salesperson@v8`, `mistralai/ministral-3-3b`, 8192 ctx) and the
current tree (`HEAD` `41ce1c2b`; the only `querygen.py`/`tools.py` change since the run is `7f3ad0e5`,
checked below — it does not touch this path). No live reproduction against LM Studio was run; none
was needed (§2).

## 1. Symptom & impact

Two `en` replies to "What products do you have?" listed every product with a price that belongs to a
**different** product (report #156 = trial 31, #286 = trial 57). Separately, five `en` replies (seven
turns) quote the right amount with `€` instead of `$`, and one `en` conversation (trial 14) quotes the
Wireless Mouse Pro at $45.99, a price no product has. Impact: a customer-facing demo states wrong
prices with confidence; in #156 the model even flagged two outliers as "unusually low" and kept them.

**The premise "the model invented them" is false for the headline case.** Every price in #156 is a
real catalog price returned by a tool, and the model zipped two separately-returned lists by position.
Only the trial-14 $45.99 is a genuine invention, and the `€` replies are the right number with the
wrong symbol. Three mechanisms, ranked by impact in §4.

## 2. Reproduction & evidence

**The run's artifacts are still on disk** (the report's "not durable" scratchpad survived):
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/u3/`
— `single-model.jsonl` (probe, 1,125 turns), `lms-server-q3.log` (LM Studio server log, which logs
every **generated** completion including tool calls and `query_graph_data`'s internal JSON — but not
request bodies, so tool *results* are reconstructed, see below), `demo-q3-window.log`. A parser that
pulls each logged completion with its timestamp, prompt tokens, tool calls and content is
`scratchpad/rca/preds.py` (same scratchpad root).

What was run (all read-only):

1. **Model traffic for the two turns** (parser over `lms-server-q3.log`). Both trials make the same
   tool sequence in the `en` session; full trace in Appendix A.
2. **Compiled the two logged `QueryRequest`s** with `querygen.compile` (a pure function, run from
   `server/.venv` with no I/O) — exact Cypher in Appendix A.
3. **Ran that Cypher read-only against `reference`** (`mcp__cypher__query`) — results in Appendix B.
4. **Positional join check.** Pairing result 1 (names, row *i*) with result 2 (prices, row *i*)
   reproduces reply #156 **byte-for-byte on all 15 price/name pairs**, including the two
   "unusually low" ones (27-inch 4K Monitor ← Gaming Mouse Pad XL's $19.99; Action Camera 4K ←
   Wireless Charging Pad's $24.99). #286 is the same list after the model re-checked the two flagged
   outliers with `lookup_product_fact`, corrected *only those*, and dropped one row.
5. **Corpus sweep** over all 1,125 replies for name→price pairs against the catalog: 79 pairs stated,
   31 wrong — 27 in #156/#286, 2 in trials 36/45 (Smart Home Hub $179.99, same mechanism, Appendix
   B), 2 in trial 14 (the $45.99). `€` appears in 7 replies (trials 0, 13×3, 28, 42, 54), always with
   the correct amount, `€29.99`.

**Ruled out by evidence, not argument:** context truncation (the failing completions ran at
2,445 and 3,031 prompt tokens of 8,192); "no tool call at all" (both turns called
`query_graph_data` twice and `filter_products` once); a tool returning wrong numbers (every number
returned is the real catalog value, `reference` checked); `7f3ad0e5` having already fixed it
(`querygen.default_returns` only fills an **absent** `returns`; these completions supplied
`"returns": ["p.price"]` explicitly, so the current tree compiles them identically — re-run in step 2
on `HEAD`).

## 3. Causal chain

Headline case (#156 / #286), each link backed:

1. User asks "What products do you have?". The model calls
   `query_graph_data(question=…, dataset="catalog")` — the system prompt
   (`server/falkorchat/proof_defs.py:401-457`) steers generic catalog questions to it.
2. The internal querygen call returns `returns: ["p.name"]` — what its own instructions tell it to
   do for any list question (`tools.py` `_QUERY_REQUEST_INSTRUCTIONS`: "When the question asks to
   list … return `<var>.name`"). Compiled: `RETURN DISTINCT p.name LIMIT 50`, **no `ORDER BY`** →
   15 names in node-scan order, tool result `{"items":[{"p.name":"Wireless Mouse Pro"},…]}`
   (`QueryGraphDataTool.run`, `tools.py:1000-1033`; row shape `repository.py:4613-4614`).
3. The model wants prices and tries the right tool, `filter_products`, but sends
   `{"category": ""}` instead of omitting it. `repository.filter_products` normalizes `""` to `""`
   and matches it with `coalesce($categoryNormalized, …)`, which only treats `null` as "no filter"
   (`repository.py:3547-3560`) → 0 rows → `{"items":[],"finding":"no matching products found"}`
   (verified: the same predicate returns `count 0` on `reference`). The one tool that returns
   name+price **in the same row** was reached and lost on an empty-string argument.
4. Fallback: `query_graph_data("What are the prices of all products in the catalog?")`. querygen
   returns `returns: ["p.price"], order_by: "p.name"` — prices with **no identifying column**.
   Nothing rejects that: `QueryRequest`/`compile` accept any single projection
   (`querygen.py:117`, `:440-537`). Compiled to the tuple-distinct shape → 15 bare prices **ordered
   by name**, `{"items":[{"p.price":349.99},…]}`.
5. The model now holds two unkeyed parallel lists in different orders (node-scan vs. alphabetical)
   and joins them by row index. Every price is real; 13–15 of them are on the wrong product.
6. Nothing downstream catches it: the K-056 `possible fabrication` signal
   (`executor.py:923-926`) fires only when **no** granted domain tool was dispatched — here three
   were. The demo log has 0 `possible fabrication` lines for the whole Q3 window.

Trials 36/45 (Smart Home Hub $179.99) are the same chain with the step-4 completion omitting
`order_by`: `RETURN DISTINCT p.price` returns 14 rows in node-scan order — `DISTINCT` collapses the
two $89.99 products — so 14 positional pairs happen to line up and the 15th name gets the last
price repeated (Appendix B).

Trial 14's $45.99 (turns 3–4): the `en` session's turn-3 completion dispatched only `get_profile`
(Appendix A); no tool result in that turn carries a price and the turn-2 reply never stated one. The
number is a genuine invention. K-056 stays silent because `get_profile` counts as "a granted domain
tool dispatched" (`executor.py:924` — any granted tool, not a price-grounding one).

`€` (7 replies, all `€29.99` after `add_to_cart`): the amount comes from the tool row; the symbol
comes from nowhere in the system. `Product` has no currency field (`scripts/seed_catalog.sh:80`,
"name/category/price only"), no tool result names a currency, and the system prompt never states one
— USD exists only as the storefront UI's default (`salesperson/src/i18n/format.ts:11-15`, "nothing in
this catalog carries a currency of its own"). The model picks the symbol from its prior.

**Hypotheses ruled out:** see the end of §2; additionally "the system prompt invites invention" —
it explicitly forbids guessing a price (`proof_defs.py`, "Never guess a price or category you have
not retrieved from a tool"), and in #156/#286 the model obeyed it literally: every number *was*
retrieved.

## 4. Root cause

**RC-1 (headline, #156/#286/t36/t45 — 29 of 31 wrong price pairs). Confidence: confirmed** (the
logged completions, compiled with the shipped compiler and executed read-only, reproduce #156 exactly;
the t36 variant reproduces from the third query in Appendix B).
`query_graph_data` can return a per-row attribute **without the column that says which entity the
row is about** (`returns: ["p.price"]`), and its other natural output for a list question is the
identity column without the attributes (`returns: ["p.name"]`, no `ORDER BY`). A "list everything
with prices" question therefore gets answered by two unkeyed result sets in different orders, and a
3B model joins them by position. The flaw is in the tool contract — `querygen.QueryRequest`/`compile`
accept a row-identity-free projection, and `QueryGraphDataTool.run` hands it to the agent as-is
(`tools.py:1012-1033`) — not in the system prompt and not in the model "inventing" anything.

- **Trigger:** the scripted probe turn "What products do you have?" followed by the model wanting to
  add prices, on `en` (the `es`/`pt-BR` sessions listed names only and never asked for prices).
  2 of 75 `en` trials hit the bad pairing; 2 more hit the `DISTINCT` variant and were 14/15 right by
  luck of node-scan order.
- **Contributing factor CF-1 — `filter_products` treats `""` as a real category.** The model reached
  for the right tool (rows carry `name`+`price` together) and passed `{"category": ""}`; the
  repository's `coalesce($categoryNormalized, …)` only treats `null` as "no filter"
  (`repository.py:3550-3552`) → "no matching products found" → the fallback to `query_graph_data`.
  In both failing trials this was the step that tipped the turn onto the bad path.
- **CF-2 — the K-056 fabrication signal cannot see either failure class.** It requires that *no*
  granted domain tool was dispatched (`executor.py:924`); a misjoin is grounded in tools, and an
  invention next to a `get_profile` call counts as grounded too. 0 warnings across Q3.
- **CF-3 — no automated content check.** The probe classifies language only; both reports caught the
  prices by hand-read, as an aside.

**RC-2 (trial 14, $45.99 — 2 pairs). Confidence: confirmed that no tool in that turn returned a
price** (logged completions); that the number came from the model's prior is inferred. Small-model
invention despite the prompt's "never guess a price". Not fixable at the source with this model;
the actionable part is CF-2.

**RC-3 (`€`, 7 replies). Confidence: confirmed absent from every input** (prompt, tool schemas,
catalog rows) — the symbol choice itself is inferred to be the model's prior. Nothing tells the
model the currency; the system's only USD declaration is the UI's formatter default.

## 5. Suggested fix & prevention

Ordered by when each can land relative to K-078 (`docs/plans/salesperson-cart-product-id.md`, still a
skeleton at the time of writing — its §4 is reserved for exactly this sequencing).

**F1 — `filter_products`: blank `category` means "not given" (CF-1). Owner: `tdd-engineer`. Can land
now; no K-078 overlap** (it changes argument handling, not the rows shown to the model).
In `FilterProductsTool.run` (`tools.py:497-508`), pass `category=None` when the argument is `None`
or normalizes to `""` (`extraction.normalize_name`, the same normalizer the repository applies —
verified: `""`, `"   "`, `"\t"` all map to `""`). Red-first test in `server/tests/test_tools.py`: `{"category": ""}` and
`{"category": "  "}` return the same 15 rows as `{}`. Probe the axis, don't list values: every
string the normalizer maps to `""` must behave as omitted — so assert on `normalize_name(x) == ""`
rather than on two literals. Verified: the same predicate with `null` returns all 15 products with
name and price in the same row; with `""` it returns 0.

**F2 — `query_graph_data` never returns attributes without row identity (RC-1). Owner: `architect`
(short design note), then `tdd-engineer`. Sequence after K-078's §4 decides what identity a catalog
row carries** — if K-078 puts `productId` into rows shown to the model, F2's identity column should
be the same one, and both touch the `QueryGraphDataTool.run` → `querygen` hand-off.
Candidate shape (a design for the architect to own, not a prescription): a pre-validation
normalization next to `querygen.default_returns` that, when `returns` contains a non-aggregate
projection and no identity projection for the matched label, prepends the label's identity
property — **except** for the single-entity lookup `default_returns` already recognizes (an `=`
filter on `name`/`nameNormalized`). `DatasetSchema` needs a per-label identity (`Document` and
`Chunk` have no `name`). Why it needs design, not just code: it amends the NLQ eval's Layer 1 contract
("return exactly what was asked", `tests/eval/nlq_scoring.py` header). By static reading, all 21
catalog golden pairs keep their shape (single-fact pairs carry the `=` name filter; list pairs
already return `name`; aggregates are exempt). This was **not run**, and the 19 `knowledge_base`
pairs were not checked. Done-condition: a Layer 1 re-run on ministral with no regression, which
needs LM Studio and the stakeholder's go. Side effect worth keeping: identity in the `DISTINCT` tuple
also stops `RETURN DISTINCT p.price` collapsing equal prices (the t36/t45 variant).
Prevention test (a red-first round **before** F2, so the assertion can go red): a coverage probe over
the axes the compiler varies on — every non-aggregate `returns` subset of each registered label's
properties × `order_by` present/absent/outside-returns × filter none/`=`name/other — asserting that
every compiled multi-row shape projects the label's identity. A hand-picked list of shapes would
miss the next one.

**F3 — state the currency (RC-3). Owner: `architect`, folded into K-078 or the next
`salesperson` version.** Two placements: a prompt sentence ("all prices are US dollars, `$`"), which
needs a def bump (`systemPrompt` is create-only, the AGENTS.md bump traps apply); or a currency field
on the price-bearing tool rows, which is the same row rendering K-078 is changing. Either way, do it
in the same change as K-078's rendering decision, not as a separate pass.

**F4 — make the K-056 signal see grounded-but-unrelated tool calls (CF-2, RC-2). Owner: `coder`.
Independent of K-078.** Count only price-grounding tools (`lookup_product_fact`, `filter_products`,
`query_graph_data`, `view_cart`, `add_to_cart`, `place_order`) when deciding whether a fact-bearing
reply was grounded (`executor.py:923-926`), not any granted tool. This is still advisory and still
blind to RC-1 misjoins, so it does not replace F2.

**F5 — catch it next time without a hand-read (CF-3). Owner: `qa-engineer`.** Add a catalog
price-consistency check to the probe analysis (`salesperson/scripts/single_model_analysis.py` or a
sibling): extract name→price pairs from each reply and compare them to `reference` (the regex sweep
in §2 step 5 is a working starting point), and report wrong pairs and non-`$` symbols as counted
metrics next to dead turns and language.

## Appendix A — logged model traffic (trial 31, `en`, turn 1)

From `lms-server-q3.log`, the `en`-session completions only (the `es`/`pt-BR` sessions run
interleaved). Trial 57 is identical up to the reply, then adds `lookup_product_fact` on the two
flagged outliers.

```
09:20:14 2012 tok  tool_calls query_graph_data {"question":"What products do you have?","dataset":"catalog"}
09:20:18/20 690 tok querygen -> {"matches":[{"var":"p","label":"Product","filters":[]}],"returns":["p.name"],"limit":50}
09:20:19 2215 tok  tool_calls filter_products {"category":""}
09:20:21 2241 tok  tool_calls query_graph_data {"question":"What are the prices of all products in the catalog?","dataset":"catalog"}
09:20:23 695 tok   querygen -> {"matches":[{"var":"p","label":"Product","filters":[]}],"returns":["p.price"],"order_by":"p.name"}
09:20:38 2445 tok  stop  "Here is a list of all the products we carry along with their prices: 1. Wireless Mouse Pro - $349.99 …"
```

`querygen.compile` output (current `HEAD`, identical at the run's `ad01aee7` for these inputs):

```
MATCH (p:Product) RETURN DISTINCT p.name LIMIT $limit                                          {limit: 50}
MATCH (p:Product) WITH DISTINCT p.price AS c0, p.name AS c1 ORDER BY c1 ASC LIMIT $limit RETURN c0 AS `p.price`  {limit: 20}
MATCH (p:Product) RETURN DISTINCT p.price LIMIT $limit   (t36/t45 variant, no order_by)          {limit: 20}
```

Trial 14 `en` turn 3 (09:09:48–54): the only `en` tool call is `get_profile`, then the $45.99 reply.

## Appendix B — read-only results on `reference` and the positional join

| row | query 1 `p.name` (node order) | query 2 `p.price` (by name) | true price of the row-1 name |
|---|---|---|---|
| 1 | Wireless Mouse Pro | 349.99 | 29.99 |
| 2 | Mechanical Keyboard K200 | 179.99 | 89.99 |
| 3 | Gaming Mouse Pad XL | 49.99 | 19.99 |
| 4 | Webcam HD 1080p | 79.99 | 59.99 |
| 5 | 27-inch 4K Monitor | 19.99 | 349.99 |
| 6 | USB-C Hub 7-in-1 | 34.99 | 39.99 |
| 7 | Laptop Stand Aluminum | 89.99 | 34.99 |
| 8 | Wireless Charging Pad | 199.99 | 24.99 |
| 9 | Noise Cancelling Headphones X3 | 109.99 | 199.99 |
| 10 | Bluetooth Speaker Mini | 89.99 | 49.99 |
| 11 | Portable SSD 1TB | 249.99 | 109.99 |
| 12 | Smartwatch Series 5 | 39.99 | 249.99 |
| 13 | Fitness Tracker Band | 59.99 | 79.99 |
| 14 | Action Camera 4K | 24.99 | 179.99 |
| 15 | Smart Home Hub | 29.99 | 89.99 |

Columns 2+3 are reply #156 verbatim. Query 3 (`RETURN DISTINCT p.price`, no order) returns 14 rows
in node order — 29.99, 89.99, 19.99, 59.99, 349.99, 39.99, 34.99, 24.99, 199.99, 49.99, 109.99,
249.99, 79.99, 179.99 — correct for rows 1–14, and Smart Home Hub's 89.99 is gone (a duplicate of
row 2's), so the model repeated 179.99 for row 15. That is the t36/t45 reply.

`reference` was re-seeded before the run and `seed_catalog.sh` is deterministic. Node order today
matching the run's order is inferred from the exact reproduction, not separately proven.
