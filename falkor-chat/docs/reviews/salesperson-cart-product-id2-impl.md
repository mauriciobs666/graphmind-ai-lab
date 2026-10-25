# Review — cart writes by product ID, S7 fixes: S1+S2 implementation (filter category-miss recovery)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-078

## Pass 1

**Reviewed:** the uncommitted working-tree diff against `96cebe2e`:
- `server/falkorchat/services.py` (`Services.has_category`);
- `server/falkorchat/tools.py` (`FILTER_CATEGORY_MISS_FINDING`/`_HINT`, `FilterProductsTool` docstring and `run`);
- `docs/QUERIES.md` §15.2;
- `tests/test_services.py` (28–29), `tests/test_tools.py` (30–34, `StubServices.has_category`), `tests/test_executor_agent.py` (35).

`docs/plans/salesperson-cart-product-id-coordination.md` was out of scope.

**Spec:** `falkor-chat/docs/plans/salesperson-cart-product-id2.md` v2 (D8, §5 S1/S2, §6.1 tests 28–35,
mutants N1–N7, §8 R-E/R-F) and its plan gate `falkor-chat/docs/reviews/salesperson-cart-product-id2.md`.

**Verdict: approve with suggestions.** There are no blockers and no majors. The code matches the
plan's S2 sketch line for line. Two minor test gaps, each shown by a surviving mutant, and one nit.

CPG: considered, not relevant — the diff touches two functions plus tests, and I read every
function they span (`Repository.filter_products`, `Services.catalog_index`,
`_handle_tool_call`, `_collect_target_values`, `_GROUNDING_SOURCES`, `_HELD_REASON`) directly.
`cpg_falkorchat` is stale per the plan's §2.

### What I ran

- **The three test files, stub-only:** `pytest tests/test_services.py tests/test_tools.py
  tests/test_executor_agent.py -k 'not _live'` gave **474 passed, 4 deselected**.
  - An earlier run with `-k 'not test_post_message_writes_agent_message_live'` gave 477 passed.
    That run also executed the two `graphrag_retrieve_*_live` tests. They use `repo` →
    `conn`, which wipes only `ws:test`, not `reference` (`tests/conftest.py:86-91`).
  - No `wf_repo`, storefront, or full-suite run was made.
- **The real tool against the stub, diffed with test 35's JSON** (script in the session scratchpad,
  `stubdiff.py`). Results are in the Appendix.
- **Six mutants,** each run on a copy of the file and then restored. All three files were confirmed
  byte-identical with `cmp`:

| Mutant | Result |
|---|---|
| N3: no `has_category` check | killed (4 red) |
| N6: check even when `category is None` | killed (1 red) |
| N7: `filter_products` removed from add grounding | killed (2 red) |
| **X1:** `has_category` also called on a **non-empty** result | **survives** (m1) |
| **X2:** `filter_products` removed from **remove** grounding | **survives** (m2) |
| X3: `StubServices` default flipped to `has_category=False` | no test changes (see check point 1) |

### Answers to the four check points

**1. Stub fixture traps: none found.**
- **The `True` default.** Flipping it to `False` (X3) turns nothing red. Every id2 test that
  reaches the empty-with-category branch passes `has_category=` explicitly (tests 30–33). The
  pre-existing filter tests that use the default either return rows or send no `category`, so
  they never reach the check. The default is the pre-id2 behaviour, which is the right default.
- **Test 35's hand-written JSON, key for key.** The real `FilterProductsTool.run` output, built
  from a `StubServices(has_category=False)` carrying the same two-entry catalog, has the same
  keys in the same order (`items`, `finding`, `hint`, `catalog`). `finding` and `catalog` are
  equal. Only `hint` differs: the test truncates it to its first clause.
  `_collect_target_values` produces the same grounded set from both, `{mechanical-keyboard-k200,
  wireless-mouse-pro}`. So the test passes for the right reason today (nit n1).

**2. `has_category` against the real `Repository.filter_products`: safe.**
- **Cost.** A category miss now makes three `reference` reads: the original filter,
  `has_category`, and `catalog_index`. A known category emptied by price makes two reads. A
  non-empty result or a price-only miss still makes one. All three reads use the same query,
  anchored on the `Product.price` index over ~15 products (`repository.py:3561-3568`).
- **Failure posture.** An engine fault in either extra read is not a `ServiceError`. It
  propagates out of `_handle_tool_call` (`executor.py:1166-1200`) to `_drive`'s fail net, the same
  as a fault in the original read. There is no new error class and no swallowed fault.
- **The S7 timeout.** DEF-6 was a cold-start timeout on `find_waiting_run_for_thread` in
  `ws:demo` (RCA §3). It was a different graph and a different query, and it did not recur. The
  extra reads add exposure only on the miss path, and only in proportion to its read count. No
  finding.
- **Blank and whitespace-only categories (verified).** `normalize_name` maps `""`, `"   "`,
  `"\t\n"`, `" "` and `"　"` to `""`. The repository turns `""` into `None`
  (`repository.py:3558-3560`), so `has_category` answers `True` on a non-empty catalog and the
  old shape is returned. One edge goes the other way: a zero-width space (`"​"`) is not
  stripped, so it takes the miss branch. That outcome is correct, since no category has that
  name, and harmless.
- **Non-string categories.** A non-string `category` still raises inside the *first*
  `filter_products` call, so `has_category` is never reached with one.

**3. Grounding: the miss `catalog` grounds both adds and removes, and it should.**
- **Both.** `filter_products` is in both rows of `_GROUNDING_SOURCES` (`executor.py:357-364`), and
  `_collect_target_values` walks the `catalog` list.
- **Removes are acceptable.**
  - A grounded remove of a product that isn't in the cart writes nothing wrong.
  - A remove of a product that *is* in the cart was already groundable through `view_cart`.
  - A `lookup_product_fact` miss list already grounds every product the same way.
  - So this is D3's accepted residual, not a new class of hole.
- **The gap.** Only the add half is pinned (m2).

**4. Interaction with the hint: consistent.**
- **Against the frozen `salesperson@v9` prompt** (`proof_defs.py:448-458`):
  - The add rule is "copied exactly from a `lookup_product_fact` or `filter_products` result in
    this same turn", and the `catalog` sits inside a `filter_products` result.
  - The prompt's pick instruction covers only a lookup miss. That is why
    `FILTER_CATEGORY_MISS_HINT` carries "pick it from this list and use its productId" itself.
- **Against `LOOKUP_MISS_HINT`.** The filter hint's price clause is the lookup hint without
  "again". `grep "again with its exact" tools.py` matches only `LOOKUP_MISS_HINT` (`:399`), which
  is S5's check, already met.
- **The finding text.** The new finding no longer contains "no matching products found", which
  the prompt's "if nothing matches, say so plainly" could otherwise attach to.
- **No new issue.** The filter hint has no counterpart to the prompt's "If nothing in the list
  matches, say so plainly". The general "if nothing matches" rule covers that case, and the
  wording is D8.2's approved text. The category-browse risk is the plan gate's m4, already
  recorded, and is not reopened here.

### Findings

**m1 — Minor — the extra read on a successful category filter is unpinned** (`tests/test_tools.py`
`test_filter_products_returns_items`, about line 545).
- **Evidence.** Mutant X1 adds a `has_category` call before `return json.dumps({"items": rows})`,
  and all 474 tests stay green.
- **Why it matters.** Plan §8 R-E bounds the cost as "absent from every non-empty result". The
  table makes N6 a contract pin for the price-only miss, but the success path, the hot path, has
  no such pin. A refactor that checks the category up front would double every
  category-filtered call silently.
- **Fix.** Add `assert svc.category_checks == []` and `assert svc.catalog_reads == 0` to
  `test_filter_products_returns_items`.
- **Done when** X1 turns at least one test red.

**m2 — Minor — D8.3's remove half is unpinned** (`tests/test_executor_agent.py` test 35;
`executor.py:361-363`).
- **Evidence.** Mutant X2 drops `filter_products` from `_GROUNDING_SOURCES["remove_from_cart"]`,
  and nothing goes red. No existing test grounds a remove through `filter_products` in any shape,
  so the gap predates id2.
- **Why it matters now.** D8.2/D8.3 and the new `FilterProductsTool` docstring say the list
  grounds adds **and** removes. That claim currently has no test behind it.
- **Fix.** Add a remove sibling of test 35: the same miss result, then
  `remove_from_cart(MOUSE)` is dispatched and `remove_from_cart("mouse-sem-fio-pro")` is held with
  `_HELD_REASON["remove_from_cart"]`.
- **Done when** X2 turns that test red.

**n1 — Nit — build test 35's miss from the shipped constants**
(`tests/test_executor_agent.py`, the new test).
- **Evidence.** The test hard-codes `"no catalog category has that name"` and a truncated `hint`.
  It matches the real output key for key (Appendix), and grounding is key-based, so the test is
  correct today.
- **Fix.** Use `tools.FILTER_CATEGORY_MISS_FINDING`/`_HINT` for those two values, so the pin
  stays tied to what the model actually receives.

### What's solid

- `run` reads `category` once, and the `is not None` test, the call order and the result shapes
  match the plan's §5 sketch exactly.
- `has_category` adds no second normalizer, and the blank-category behaviour comes from the
  repository, as D8.1 intends.
- The miss result reuses `catalog_index`, which stays price-free (K-079), and keeps `items: []`.
- Test 32 pins the price-emptied branch byte-identical to the pre-id2 shape.
- Test 33 pins both halves of N6 and the raw blank pass-through. Its `category_checks == [""]`
  assertion also kills a "truthy `category`" or "strip in the tool" variant.
- `QUERIES.md` §15.2 now attributes the abstention shape to the tool, which corrects the old
  sentence. It also states that the query is unchanged, and both statements are accurate.

### Open questions

None.

### Learnings (not ingested, per brief)

- In `falkor-chat/server`, a `-k` expression that excludes only one `_live` test from
  `test_tools.py` still runs `test_graphrag_retrieve_*_live`. Those tests request `repo` →
  `conn`, which wipes `ws:test` (not `reference`). Use `-k 'not _live'` for a stub-only run of
  that file.
- `extraction.normalize_name` does not strip U+200B (zero-width space), because neither
  `str.strip()` nor `\s` treats it as whitespace. Every other Unicode space tried (NBSP,
  ideographic space) is removed.

## Appendix

**Test 35 vs. real tool output** (`FilterProductsTool(StubServices(filter_result=[],
has_category=False, catalog=[KEYBOARD, MOUSE rows])).run({"category": "Mouse sem fio Pro",
"minPrice": 0, "maxPrice": 100})`):

```
keys equal: True ['items', 'finding', 'hint', 'catalog']
DIFF hint "that is not a catalog category, so this is every product in the catalog instead; if the c…" vs 'that is not a catalog category'
grounded real: {'mechanical-keyboard-k200', 'wireless-mouse-pro'}
grounded hand: {'mechanical-keyboard-k200', 'wireless-mouse-pro'}
default stub: {"items": [], "finding": "no matching products found"}
```
