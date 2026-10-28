# `query_graph_data` row identity and category-miss recovery — implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-079 (F2), K-086

## Pass 1 — code gate on Stage A (`6286de44`)

**Scope.** The diff of commit `6286de44` only (Stage A, steps A1–A3), against the approved plan
`docs/plans/salesperson-query-graph-data.md` v3 (§3, §6, §6.1, §6.2, §7), this reviewer's plan gate
`docs/reviews/salesperson-query-graph-data.md` (Passes 1–2), the coordination ledger row U3a, and
the RCA `docs/reviews/salesperson-fabricated-prices-rca.md` §F2. Stage B (K-086, D6/D7) is not
implemented and is not a finding. `falkor-chat/server` at HEAD is byte-identical to `6286de44`
(`git diff --stat 6286de44 HEAD -- falkor-chat/server` empty), so tests were run on the working tree.

**Verdict: approve with suggestions.** No blocker. One major test gap (M1): the rewrite's
preservation of every field except `returns` is untested, and Stage B's D7 relies on it; land the
one-line probe assertion before B1. Code conforms to §3.1 D1–D4 and §6 A1–A3, verified by
execution.

CPG: considered, not relevant — `cpg_falkorchat` is stale (built 2026-10-15, 19 commits behind) per the brief; callers of `querygen` were found by grep, and the change is local to `querygen.py`/`tools.py`.

### Findings

#### M1 (major) — nothing pins that `ensure_row_identity` changes `returns` and *only* `returns`; two data-plumbing mutants survive every test

*Evidence (executed).* Two mutants of the rewrite at `falkorchat/querygen.py` (the
`QueryRequest.model_validate({**request.model_dump(), "returns": [...]})` return):
- **MX1**: the rebuilt request carries `"matches": [{"var": match.var, "label": match.label}]`,
  so the filters are dropped. Result: `tests/test_querygen.py tests/test_tools.py
  tests/eval/test_nlq_scoring.py` → **193 passed**. Behaviour under MX1: "prices of the Audio
  products" compiles to `MATCH (p:Product) RETURN DISTINCT p.name, p.price LIMIT $limit`, i.e. all
  15 products, silently.
- **MX2**: the rebuilt request is built from parts and drops `order_dir`/`limit`. Result: **193
  passed**. Under MX2 "the most expensive product's price" (`order_by p.price DESC, limit 1`)
  becomes the cheapest one (by construction: `order_dir` falls back to `ASC`; not run).

*Why it matters.* The probe's `expect_prepend` branch asserts only the column list
(`cols == [v.<id>, *returns]`), and every T-A test with a prepend has no filters, the default
`order_dir` and the default `limit`, and compares only cypher text (`$limit` is a parameter). So the
1,239 rewritten cases are never checked for anything but their `RETURN` columns. Stage B depends on
the untested invariant. Plan §4.2 D7: "Let `filters = request.matches[0].filters`; D1 never changes
filters", and §5 item 2 "F2 never touches `filters`". B3 will read those filters from the
*rewritten* request. The code is correct today: `model_dump()` carries every field. But a refactor
of the rewrite (Stage B is the next editor of this path) can break it with a green suite.

*Suggested improvement.* In `test_every_attribute_projection_carries_row_identity`, for **every**
case assert
`ensure_row_identity(req, schema).model_dump(exclude={"returns"}) == req.model_dump(exclude={"returns"})`.
Also give the probe's requests a non-default `order_dir`/`limit` on at least part of the axis.
Without that, MX2 would still pass: a dropped field falls back to the default, which the request
also has. Checked in a scratch harness (`fixcheck.py`, Appendix A) over the probe's own 44,323
requests with `order_dir="DESC", limit=7`: **0** violations on `6286de44`, **1,186** under MX1,
**1,239** under MX2. Placement: a small round **before B1**, not inside B3. If it is authored
alongside D7, it is born satisfied by the code it is meant to guard.

#### n1 (nit) — `Document`'s prepended identity, `title`, is neither unique nor guaranteed non-empty

*Evidence (read).* `services.py:1223` stores `title=title or source_label or ""`, so untitled
documents share `""`. Versioned re-ingests (`currentVersion`) plausibly share a title too (inferred, not run). `Document`'s only non-identity
property is the facet `sourceFormat`, so the one prepend shape is `returns ["d.sourceFormat"]` with
`order_by` on `d.title`/`d.documentId`. Its `[title, sourceFormat]` rows can still collapse under
`DISTINCT` or name nothing. This is the approved §3.1 table, not a conformance gap, and it is
KB-only (not reachable from the storefront). *Suggestion:* none required. If `knowledge_base`
`query_graph_data` ever gets a live consumer, consider `("documentId", "title")`, by the same
argument §3.3 used for `Chunk`'s `chunkId`.

### Conformance checks (all executed; no finding)

| Check | Result |
|---|---|
| D1 exemption set and order | `ensure_row_identity` tests, in order: identity declared → no aggregate → no identity projected → not (`plain` ∧ all-facet), with `plain = order_by is None or order_by in returns` (B1 fix) → not `_is_identity_lookup`. Matches §3.1 exactly. Executed shapes, Appendix B |
| Red-first counts | Pass-through mutant (`return request` first): Product **299**/13,775, Entity **66**/13,300, Document **34**/3,948, Chunk **840**/13,300, i.e. 1,239/44,323, exactly the §6.1 numbers |
| D3 | `_NAME_LOOKUP_PROPERTIES` deleted (`grep` finds it only in plan/review prose). `default_returns` and `ensure_row_identity` both call `_is_identity_lookup`. Malformed raw shapes (list `label`/`var`, list `op`, string `filters`) return without raising |
| D2 registry | `identity`/`facets` values match the §3.1 table. The invariant test checks non-empty identity and identity/facets ⊆ allowlist |
| D4 wiring | `run()` is `ensure_row_identity(model_validate(...), schema)` inside the unchanged `try`; `compile()` untouched. The NLQ eval runner reaches F2, since it drives `QueryGraphDataTool` (`tests/eval/run_nlq_golden_set_eval.py:95,313`) |
| Grounding | `_WRITE_TARGET_ARG` targets `productId` only (`executor.py:342-345`), so the newly surfaced `p.name` grounds nothing |
| Own mutants | MX3 (`default_returns` passes `var` for `label`) killed by 3 tool tests. MX4 (`facets.get(match.var)`) killed by 3 probe cases + 2 tool tests. MX5 (`plain` compares `order_by` to bare property names) killed by 3 probe cases + T-A6. MX6 (rewrite drops `order_by`) killed by T-A1 + T-A6. MX1/MX2 survive (M1) |
| Suites | `pytest tests/test_querygen.py tests/test_tools.py tests/eval/test_nlq_scoring.py` → 193 passed (1.68 s). `querygen.py` restored after each mutant; `sha256sum` `2705b86e…17fe` before and after |

### Docs (A3, plan §7)

- **`nlq_scoring.py` D5 paragraph.** It is the plan's §3.4 text verbatim, plus a pointer to
  `querygen.ensure_row_identity`, and the "exactly what was asked" sentence is qualified. Accurate
  against the code, including the facet/`order_by` caveat.
- **`QueryGraphDataTool` docstring.** All four exemptions and the prepend are stated correctly.
  "`compile()` itself is unchanged" and "never `productId`" are both true.
- **`executor.py`.** The diff is comment-only. `_collect_target_values` does match dotted keys
  (`executor.py:396-423`), as the new comment claims.
- **`querygen.py`.** The module docstring line and the `DatasetSchema` docstring are present and
  accurate.
- **Other surfaces.** No other §7 surface belongs to Stage A: `QUERIES.md` §15.2 is B3, and
  HISTORY/BACKLOG/header pointers are teco's at close. `grep` finds no other doc claiming
  `query_graph_data` returns exactly the requested columns. The SPA does not consume
  `query_graph_data` output (`grep` in `salesperson/src`, `falkor-chat/web`: none).

### Stage B readiness

M1 is the one item that bears on Stage B. D7 reads filters from the post-F2 request, so the
"F2 never changes filters" invariant should be pinned before B3 builds on it. The new F2 tool tests
use non-empty `structured_query_rows`, so D7's empty-result branch never fires in them, and B2's
"pre-existing tests unmodified" condition is reachable once `StubServices.catalog_categories`
defaults to `[]`.

### What's solid

- **The oracle is independent.** The probe's `_expect_prepend` is derived from §3.1's text and the
  compiled shape, not from the implementation. Its universal clause catches anything the
  per-branch assertion misses on columns.
- **Byte-identity on the exempt side.** Every non-prepend case asserts byte-identical cypher, so
  exemptions cannot drift silently.
- **The T-A3 departure is a real improvement.** `[p.price, count(p)]` is the only tool-level case
  that pins the aggregate exemption independently of the facet one.
- **The live shapes behave as designed** (scratch graph, Appendix B):
  - the RC-1 price list returns 15 keyed rows, and the two $89.99 products no longer collapse;
  - the facet enumeration returns 8 categories;
  - the ordered-facet list is keyed;
  - lookups and grouped aggregates are unchanged.

### Open questions

None.

## Appendix

**A. Scratch harnesses.** All are under
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6bf0232-c1ae-4f33-8aa9-de25f4a9b94a/scratchpad/impl/`:
- `shapes.py` / `shapes.out` run 18 shapes through `default_returns` → `model_validate` →
  `ensure_row_identity` → `compile`, the `run()` sequence, and execute them on graph
  `scratch_analyst_f2`. That graph was copied from `reference`'s 15 `Product` nodes by a read-only
  read, and deleted afterwards.
- `fixcheck.py` holds M1's suggested assertion.
- `mutate.py` is the single-occurrence string mutator used for MX1–MX6.

**B. Executed shapes (excerpt of `shapes.out`).**

```
returns [p.price], order_by p.name     -> RETURN DISTINCT p.name, p.price ORDER BY p.name ASC        15 rows
returns [p.price]                       -> RETURN DISTINCT p.name, p.price                             15 rows
returns [p.price] order p.price DESC l1 -> RETURN DISTINCT p.name, p.price ORDER BY p.price DESC      [['27-inch 4K Monitor', 349.99]]
returns [p.category] order p.price      -> WITH DISTINCT p.name AS c0, p.category AS c1, p.price AS c2 ... RETURN c0 AS `p.name`, c1 AS `p.category`
returns [p.category]                    -> RETURN DISTINCT p.category                                  8 rows (unchanged)
category = Audio, returns [p.price]     -> WHERE p.category = $p0 RETURN DISTINCT p.name, p.price      2 rows
price < 50, returns [p.category]        -> RETURN DISTINCT p.category (facet enumeration, unchanged)   3 rows
name = X, returns [p.price]             -> RETURN DISTINCT p.price (lookup, unchanged)                 1 row
name = X, returns [p.price], order cat. -> tuple-distinct, unchanged                                   1 row
returns [p.category, count(p)]          -> RETURN p.category, count(p) (unchanged)                     8 rows
returns [x.price]                       -> abstains (ValueError: unknown var)
```

