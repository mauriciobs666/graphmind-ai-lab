# Review: cart writes keyed by product ID, implementation of S1–S3

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-078

## Pass 1

### Scope & verdict

**Reviewed:** the uncommitted working-tree diff against `847d4b0a`, limited to `falkor-chat/server/`
(`git diff 847d4b0a -- falkor-chat/server`, 12 files). The source files are `executor.py`,
`extraction.py`, `repository.py` (one docstring), `services.py` and `tools.py`. The test files are
`test_executor_agent.py`, `test_extraction.py`, `test_salesperson_scaffold.py`, `test_services.py`,
`test_storefront.py`, `test_storefront_api.py` and `test_tools.py`. The spec is
`falkor-chat/docs/plans/salesperson-cart-product-id.md` v3 (§3 D1–D6, §5 S1–S3, §6), plus its
plan-gate review `falkor-chat/docs/reviews/salesperson-cart-product-id.md` (Pass 1 and Pass 2).
**Not reviewed:** S4–S7 (`proof_defs.py`, version pins, docs, live check) and the coordination
ledger.

**Verdict: approve with suggestions.** I found no blockers and no majors. The guard, the dedup
key, the services and the tool results do what D1–D6 say. I checked the guard's whole state space
by reading the code and by running mutants, and it holds. The five findings below are test gaps
and small loose ends. None of them changes behaviour.

CPG: considered, not relevant — the code under review is an uncommitted working-tree diff, which no loaded `cpg_falkorchat` graph can contain (the plan also records it as 12 server commits stale), so every symbol was read from the tree directly.

**Evidence base.**
- **Targeted suites.** I ran `test_extraction.py`, `test_services.py`, `test_tools.py`,
  `test_executor_agent.py`, and the three `test_salesperson_scaffold.py` tests that use no real
  database fixture: **497 passed, 8 deselected**. The deselected tests are the `wf_repo`/`repo`
  tests.
- **Mutants.** I ran 9 mutants on a scratch copy of `falkorchat/`, imported through `PYTHONPATH`
  (cwd outside `server/`, import path confirmed). Baseline on the copy: 493 green across the four
  stub-only files. Results are in Appendix A.
- **Storefront files.** I judged `test_storefront.py` and `test_storefront_api.py` statically, as
  asked. Neither was run.

**Disclosure.** My first scratch-copy run passed `--deselect` with absolute node IDs. pytest
matched none of them, so the three `*_live` tests in `test_tools.py` ran once. Those tests use the
`repo`/`conn` fixtures. They wiped and re-bootstrapped the suite's own throwaway `ws:test` graph,
and `bootstrap_schema.sh test` ran DDL only on `reference`. No `wf_repo` test ran, `reference`'s
data was untouched, and no other `ws:*` graph was touched. Every later run used `-k "not _live"`.

### Findings

#### F1 — minor — The held-reason tests check the constant against itself, so a wrong add hold message passes

`test_off_turn_write_is_held_not_dispatched_on_confirmed_repro_shape` asserts
`held["reason"] == executor_mod._HELD_REASON["add_to_cart"].format(...)`, which compares the code
with itself. It also checks two substrings ("current message", "do not add it").
`test_off_turn_remove_from_cart_is_also_held` does the same for remove. **Verified by mutant D:**
I appended "or view_cart" to the add hold reason, telling the model that a cart view can ground
an add, and all 493 tests stayed green. Plan-gate F3 required the add schema and prompt never to
name `view_cart` as an add source. The schema has a test for this
(`test_add_to_cart_schema_never_names_view_cart_as_a_source`), but the hold message, which a
held model reads at exactly the moment it decides what to do next, has none. A model told
"or view_cart" would view the cart, retry, and be held again, burning iterations.

**Fix:** in `test_executor_agent.py`, add a test that asserts on the message's content, not its
identity. The add reason must not contain `view_cart` but must contain `lookup_product_fact` and
`filter_products`. The remove reason must contain all three. This mirrors the schema test.

#### F2 — minor — Nothing pins grounding inside a single model response, in either direction

The calls in one response run in order through `_handle_tool_call` with the live `surfaced` dict.
So in `[lookup_product_fact(A), add_to_cart(A)]` the add dispatches, and in
`[add_to_cart(A), lookup_product_fact(A)]` the add is held. That matches D3 ("dispatched earlier
in this node execution"). It is also the path a parallel-calling model takes on pt/es: a miss's
`catalog` grounds a same-response add of a guessed slug. **Verified by mutant B:** I made the
guard check a snapshot taken at the start of the iteration while updates still went to the live
set, so same-response grounding became invisible. All 493 tests stayed green. Every existing test
puts the lookup in its own iteration (`_lookup_first`, test 26), and the only multi-call response
holds two adds (`test_dotted_and_nested_product_id_keys_ground_an_add`).

**Fix:** add one parametrized test in `test_executor_agent.py` covering both orders within one
`ChatResult`. `[lookup, add]` must dispatch the add; `[add, lookup]` must hold it.

#### F3 — minor — A new write tool missing a table row fails on every dispatch at runtime, and no test catches it

The comment block above `_WRITE_TARGET_ARG` says a missing `_GROUNDING_SOURCES`/`_HELD_REASON`
row "fails loudly (`KeyError` …)". True, but the success-path loop in `_handle_tool_call`
(`for write_tool, target in _WRITE_TARGET_ARG.items(): … _GROUNDING_SOURCES[write_tool]`) runs on
**every** successful dispatch of **any** tool. So one forgotten row turns every agent node's first
successful tool call into an engine fault that propagates to `fail_run`, and only at runtime. No
test checks that the tables agree (grep: no test references `_GROUNDING_SOURCES` or
`_WRITE_TARGET_ARG`).

**Fix:** add a test asserting
`set(_WRITE_TARGET_ARG) == set(_GROUNDING_SOURCES) == set(_HELD_REASON) <= set(_DEDUP_ARG_RESOLVERS)`.
It goes in now as its own small test and stays green today. It is meant to go red for the future
edit that adds a write tool, not to be written alongside that edit.

#### F4 — minor — The miss `catalog` is the name-only half of the K-079 RC-1 pair; plan D2 claims more than it guarantees

D2 says that with ID and name in the same row "a misjoin is impossible either way". That holds
against `filter_products`, whose rows also carry names. But RC-1
(`falkor-chat/docs/reviews/salesperson-fabricated-prices-rca.md` §4) is a name-only list joined by
position to an identity-free **price** list, and until K-079 F2 lands `query_graph_data` can still
return such a list (`returns: ["p.price"]`). The miss `catalog` is a new name-only list of every
product (`LookupProductFactTool.run` → `Services.catalog_index`). So "miss, then a
`query_graph_data` price listing" puts both halves of RC-1 in one turn. The code follows the plan,
the `hint` is in place, and the name sort differs from `query_graph_data`'s unordered scan, so I
see no code change to ask for. Inferred from reading, not reproduced.

**Fix (measurement, routed to `qa-engineer` via S7):** in the reported-not-gated name→price sweep
(§6.3), tag each wrong pair by whether its turn had a `lookup_product_fact` miss followed by a
`query_graph_data` call. That separates "the miss list invites misjoins" from the existing RC-1
baseline. Plan wording, for `architect` on the next revision: D2 should say "impossible against
`filter_products`", not "either way".

#### F5 — minor — `test_salesperson_scaffold.py` is not stub-based, so its S3 edit has not had a run that leaves `reference` intact

Plan §5 calls `test_salesperson_scaffold.py` stub-based. Five of its eight tests request `wf_repo`,
which wipes `reference` (AST scan of fixture parameters, Appendix A). Its `_SCHEMAS` table, the S3
`productName` → `productId` edit, is served by a stub registry used only in those `wf_repo` tests
(`_SCHEMAS[name]` at line 212; the three safe tests are def-constant asserts). So the scaffold edit
gets exercised only by a run that wipes `reference`. The reported counts (implementer 496,
coordinator 493, mine 497 with the three safe scaffold tests) do not tell me whether anyone ran
the `wf_repo` five. The edit is a mechanical rename, so the code risk is low.

**Fix:** leave the five to `teco`'s gate, in the same re-seed window as the storefront files. If
someone already ran them, follow `falkor-chat/AGENTS.md`'s re-seed obligation. The read-only
`verify_catalog.sh` / `verify_salesperson.sh demo` show whether it is owed. I did not run them,
because the brief keeps me off `falkordb-dev`.

#### Nits

- **n1.** `_GROUNDING_SOURCES` lists `add_to_cart` under `add_to_cart` and `remove_from_cart`
  under `remove_from_cart`, and both entries do nothing. A write dispatches only when its
  normalized ID is already in its own set, and its result echoes `product["productId"]` from an
  exact `lookup_products_by_id` match, which is that same ID. Mutants A1/A2 (drop each) and Y
  (self-ground from the call's own arguments) all survive, as equivalents. The plan prescribed
  these entries, so keep them if you like, but the comment "each write tool also re-grounds the ID
  its own result echoes" implies they do something. Either say they are redundant or drop them.
- **n2.** `_collect_target_values` catches `TypeError`, which cannot happen:
  `_handle_tool_call` always passes a `str`
  (`content = out if isinstance(out, str) else str(out)`). It is harmless. The iterative walk is
  fine, though `json.loads` is still recursion-bound itself.
- **n3.** A test comment still describes the old flow. `tests/test_repository.py` (the comment
  above the `lookup_product` `productId` projection test) says it exists so
  "`services.add_cart_item` can key the `CartItem` on it". This is the same stale sentence the
  `repository.py` docstring edit fixed.

### Deviations reported by `coder`

1. **S2 grep hits three `productName` comments in `executor.py`: accepted.** All three explain the
   retirement (the K-058 rationale and the D4 "stray `productName`" note). None is a live name
   path. The done-condition's intent, no cart-related code path on names, is met. Every other hit
   is in tests that assert `productName` is gone or ignored.
2. **The remove hold text names `view_cart`: not a deviation.** D3 says verbatim "*remove:* the
   same, with `view_cart` added to both source lists". It is correct, and it is consistent with
   `_GROUNDING_SOURCES["remove_from_cart"]`.
3. **`(ValueError, TypeError)` and the iterative walk: accepted.** See n2.
4. **The added `quantity: 0` remove test: accepted, and needed.** It is the only test that kills
   the plan-gate F8 mutant (`or None` in `_resolve_remove_from_cart_dedup_args`). With it removed,
   the F8 mutant survives (Appendix A).
5. **The out-of-plan `repository.py` docstring edit: accepted.** The old text said `productId` was
   projected so `add_cart_item` could key on it, which stopped being true here. The new text is
   accurate. n3 is its missed twin in the tests.

### Answers to the brief's focus questions

- **Parallel calls in one response.** The calls run in order against live state. A lookup earlier
  in the response grounds a later write; a later lookup does not. A K-061 duplicate inside one
  response is held, because `dispatched_writes` is updated per call. This is correct per D3 but
  unpinned (F2).
- **The implicit `post_message` fallback.** It passes the same `surfaced`. `post_message` is in no
  `_WRITE_TARGET_ARG` or `_GROUNDING_SOURCES` row, so it neither checks nor contributes. No
  behaviour change.
- **Nested or dotted `productId`.** Both are collected: an exact key or a key ending in
  `".productId"`, at any depth, through dicts and lists. A string value under any other key is
  ignored, and so is a non-string value under the key (tests 20 and
  `test_collect_target_values_contract`). JSON-in-a-string would be missed, but no grounding tool
  emits it: `QueryGraphDataTool.run` returns `{"items": rows}` directly.
- **A hold followed by a successful lookup.** The add then dispatches (test 26, accepted R3). The
  held call seeds neither `dispatched_writes` nor `surfaced`, so the later add is neither a K-061
  hold nor pre-grounded.
- **Threading.** `surfaced` is a local of `_run_agent_node`, threaded explicitly through both call
  sites. The diff adds no `self.` state, and the module tables are only ever read. M10 (hoist to
  `self`) is killed by test 27, and the coordinator's per-call-copy mutant by 18 tests.
- **K-061.** The new key is a strict superset of the old holds: identical raw arguments always
  project identically. The extra merges are exactly the cases the tool already treats as the same
  request: ID case and whitespace (services normalize with the same `normalize_product_id`), an
  ignored extra argument, and for remove an explicit `quantity: null` against an omitted one (both
  pass `None` to `remove_cart_item`). No two requests the tool treats differently now collide.
  Explicit `0` stays distinct from omitted (deviation 4). A K-058 hold still runs first, and on
  missing, empty or non-string IDs it now holds calls the old guard used to let through to K-061.
- **New model-visible surfaces.** None of them is a fabrication or injection surface:
  - `LOOKUP_MISS_HINT` is a static string.
  - The `catalog` rows are `reference` `Product` data, the same trust level as the
    `filter_products` rows the model already sees.
  - The D6 `reason` and the hold `reason` echo only the model's own argument back to it.
  - Neither echo sits under a `productId` key. If one did, the call would ground itself. I checked
    that `_product_not_found` puts the ID only inside the `reason` string.
  - The RC-1 adjacency is F4.
- **Fixture fidelity against the real tools.**
  - `_found` matches `LookupProductFactTool`'s hit shape: `found` plus the
    `Repository.lookup_product` row with `productId`.
  - The test-17 miss fixture matches the real miss keys, with its own hint text.
  - `_catalog_listing` is a subset of `FilterProductsTool`'s `{"items": rows}`.
  - The `view_cart` fixture is a subset of `Services.get_cart`'s priced lines.
  - Stub `add_to_cart` results (`'{"found": true}'`) omit the real echoed `productId`. That only
    matters for the do-nothing self-source entries (n1).
  - `FakeCartServices.lookup_product` returns `productId: "widget"`, and the AC-9 test adds
    `"widget"`, consistent with the real wrapper.
  - The FakeRepo `lookup_products_by_id` matches the real exact-match/absent-row contract.
  - The `catalog_index` test seeds an order that differs from both input and price order, so the
    name sort is really pinned.
- **Storefront call sites (static).** Both files' `_catalog_rows` produce
  `productId = f"widget-{i:03d}"` next to `name = f"Widget {i:03d}"`, so every
  `product_id="widget-00N"` resolves to exactly the product the old `"Widget 00N"` name did. Each
  calling test seeds at least as many rows as the IDs it uses (e.g. 3 rows for the `widget-003`
  test), with one exception. `test_a_self_reset_clears_the_dead_turn_latch` seeds 1 row while
  `_busy_participant` adds `widget-002`, which abstains silently. That was already true under the
  name form (`"Widget 002"` also failed to resolve), so behaviour is the same, not a regression.
  `test_storefront_api.py`'s `_place_order` callers all use the default, the `seeded` fixture
  seeds 3, and `_place_order` asserts the order is non-`None`, so a wrong slug would fail loudly.

### What's solid

- **The guard follows D3 closely.** No fail-open, success-only surfacing, a per-node local, a
  per-tool conditional reason, and the observability prefix and trace kind kept. M1–M6, M9, M10
  and plan-gate F8 are each killed by a named test.
- **One normalizer, used everywhere.** `normalize_product_id` is used for the guard, the dedup
  key and the services, so "compare equal everywhere or nowhere" holds by construction. The
  docstring carries the lowercase-invariant caveat (plan-gate n3).
- **The services reuse `lookup_products_by_id` with no new Cypher.** An unknown, blank or
  non-string ID writes nothing, not even the `ensure_*` anchors, and the parametrized test
  asserts exactly that.
- **The tests replace the old contract rather than deleting it.** The §9.2 repro, the fail-open
  replacement (test 15), accepted depth (test 26) and scope (test 27) are all present and
  meaningful.

### Open questions

- **F5:** did any run of `test_salesperson_scaffold.py` include its five `wf_repo` tests? If so,
  the `reference` re-seed obligation applies before the next demo or live check.

### Stopping condition

There is no blocking stopping condition, because the verdict is approve. If F1–F3 are taken up,
they are done when mutant D (an extra "view_cart" source in `_HELD_REASON["add_to_cart"]`),
mutant B (the guard checks an iteration-start snapshot) and a deleted `_GROUNDING_SOURCES` row
each turn at least one test red, and the four stub-only files stay green.

## Pass 2

**Reviewed:** the working tree after the fix pass on F1–F3, n1 and n3. The fix touched
`executor.py` (`_GROUNDING_SOURCES` and its comment), `test_executor_agent.py` and a
`test_repository.py` comment.

**What I ran:**
- **The four stub-only files:** 496 green on a refreshed scratch copy (`-k "not _live"`), which
  matches your count.
- **The Pass 1 command:** 500 passed, 8 deselected (adds the three safe scaffold tests).
- **Mutants:** run against the current tree (table below).

**Verdict: approve.** Every stopping-condition mutant is killed. I confirmed all three deviations
by running them. The one new issue is a stale test comment.

### Dispositions

- **F1: fixed.** The content checks were folded into
  `test_off_turn_write_is_held_not_dispatched_on_confirmed_repro_shape` and
  `test_off_turn_remove_from_cart_is_also_held` rather than written as a new test. **Deviation
  confirmed:** mutant D (the add reason also names `view_cart`) now fails the repro test. The
  assertions check content (`lookup_product_fact`/`filter_products` present, `view_cart` absent
  for add, all three present for remove) and no longer compare the constant with itself. Placing
  them in the existing tests serves the purpose just as well.
- **F2: fixed.** `test_grounding_within_one_response_follows_call_order[True/False]` was added.
  Mutant B (the guard checks a snapshot taken at the start of the iteration) now fails
  `[True]`.
- **F3: fixed.** `test_guard_tables_cover_the_same_write_tools` was added. It fails when a
  `_GROUNDING_SOURCES` row is deleted, and it fails when a stray `_HELD_REASON` row is added.
  Deleting the row also fails 52 other tests through the runtime `KeyError`, which is the failure
  F3 described. **Deviation confirmed: dropping `<= set(_DEDUP_ARG_RESOLVERS)` is right, and my
  suggestion was wrong.**
  - `_resolve_dedup_arguments` reads `_DEDUP_ARG_RESOLVERS.get(name)` and falls back to the raw
    arguments, so a write tool with no resolver does not fault.
  - The `_WRITE_TARGET_ARG` checklist makes a resolver conditional on a post-guard default.
  - Dropping the `remove_from_cart` resolver (mutant R) fails only the behavioural
    case-variant test, never a crash. So a missing resolver weakens dedup; it is not a broken
    invariant.
- **n1: fixed, by dropping the rows.** **Deviation confirmed as behaviour-equivalent:** putting
  `add_to_cart` back into its own row, or `remove_from_cart` into its own, leaves all 496 green,
  the inverse of Pass 1's A1/A2. The new comment says why the rows are gone. It departs from the
  plan's D3 table only on paper (see Plan carry-over).
- **n3: fixed.** The `test_repository.py` comment now describes the K-078 role.
- **F4, F5, n2:** not in scope of this pass; unchanged.

| Mutant (current tree) | Result |
|---|---|
| D: the add hold reason also names `view_cart` | killed: `test_off_turn_write_is_held_not_dispatched_on_confirmed_repro_shape` |
| B: guard reads an iteration-start snapshot of `surfaced` | killed: `test_grounding_within_one_response_follows_call_order[True]` |
| `del _GROUNDING_SOURCES["remove_from_cart"]` | killed: `test_guard_tables_cover_the_same_write_tools` plus 52 more |
| extra `_HELD_REASON["x_tool"]` row | killed: `test_guard_tables_cover_the_same_write_tools` |
| restore the `add_to_cart` / `remove_from_cart` self-source rows | survive, 496/496 (equivalent, so dropping them is safe) |
| R: drop the `remove_from_cart` dedup resolver | killed by a behavioural test only (no fault), so not a table invariant |

### New finding

- **p2-n1 (nit).** One test comment is now false. In
  `test_a_held_write_and_a_non_json_result_surface_nothing_and_do_not_raise` it still says
  "`add_to_cart` is itself a grounding source for adds". After n1 it is not. The test's
  non-JSON-result check still holds, but the held-add half now passes simply because `add_to_cart`
  is not a source at all. Fix: reword the comment as "a held add contributes nothing, and
  `add_to_cart` is not a grounding source".

### Plan carry-over (`architect`) and timing

- **D3's `_GROUNDING_SOURCES` table without the self-source rows. Can wait** for the next
  plan revision or S6. The change is behaviour-equivalent (shown above), and S7 measures
  behaviour, not the table.
- **D2's "impossible either way" → "impossible against `filter_products`". The wording can
  wait.** But F4's **measurement** must reach `qa-engineer` before S7 runs. That means tagging
  each wrong name→price pair by whether its turn had a `lookup_product_fact` miss followed by
  `query_graph_data`. Without that tag, the reported sweep cannot separate the miss list from the
  RC-1 baseline. Carry it in the S7 brief or the §6.3 "Reported, not gated" list. Either way it
  has to be in front of the person who writes `cart_outcome_analysis.py`.

## Appendix A — Mutants run (scratch copy, `PYTHONPATH`, 493-test baseline green)

| ID | Mutation | Result |
|---|---|---|
| A1 | drop `add_to_cart` from `_GROUNDING_SOURCES["add_to_cart"]` | survives (equivalent; n1) |
| A2 | drop `remove_from_cart` from `_GROUNDING_SOURCES["remove_from_cart"]` | survives (equivalent; n1) |
| Y | a successful write adds its own normalized argument to `surfaced[call.name]` | survives (equivalent; n1) |
| D | `_HELD_REASON["add_to_cart"]` also names `view_cart` | **survives** (F1) |
| B | the guard reads an iteration-start snapshot of `surfaced`; updates still go to the live set | **survives** (F2) |
| F8 | `_resolve_remove_from_cart_dedup_args` uses `quantity or None` | killed: `test_same_turn_remove_omitted_and_explicit_quantity_are_different_keys[first1-second1]` (the `quantity: 0` case) |
| R | drop the `remove_from_cart` entry from `_DEDUP_ARG_RESOLVERS` (raw-args fallback) | killed: `test_same_turn_remove_identical_repeat_with_differently_cased_id_is_held` |
| X | the guard checks `surfaced["remove_from_cart"]` for every tool | killed: `test_view_cart_does_not_ground_an_add` |
| coordinator's | per-call copy of `surfaced` at the main-loop call site | killed (18 tests, the coordinator's run, not re-run) |

**Fixture scan.** I ran an AST pass over each targeted file, flagging any test function whose
parameters reach `wf_repo`/`conn`/`repo`/`catalog_repo`, directly or through a local fixture. It
found 5 tests in `test_salesperson_scaffold.py` (all `wf_repo`) and 3 `*_live` tests in
`test_tools.py` (`repo`). `test_extraction.py`, `test_services.py` and `test_executor_agent.py`
have none.

## Pass 3 — S4–S6

### Scope & verdict

**Reviewed:** the uncommitted working-tree diff against `f02faa69`, excluding the coordination
ledger: `server/falkorchat/proof_defs.py` (`salesperson@v9`), `server/tests/test_salesperson_scaffold.py`,
`scripts/{start_demo,seed_salesperson,verify_salesperson}.sh`, `AGENTS.md`, `README.md`,
`docs/QUERIES.md` and `salesperson/README.md`. The spec is plan v3 D7, §5 S4–S6 and §7, plus the
plan gate (F2, F3). The guard is the committed `4dac501e` one (`_GROUNDING_SOURCES` without the
self-source rows, Pass 2).

**Verdict: approve with suggestions.** There are no blockers and no majors. The `v9` prompt
agrees with the tool schemas, the hold reasons and the guard. `v9`'s def equals `v8`'s in every
field except `systemPrompt`, which I checked by evaluating both modules. S4+S5 are safe to commit
as one unit and then publish. The findings are a test gap (F1), a publish-ordering precaution
(F2) and one doc nit.

CPG: considered, not relevant — the diff is a prompt string, shell defaults and prose; no call-graph question arises, and no loaded `cpg_falkorchat` graph contains uncommitted code.

**What I ran:**
- **Scaffold tests without a database:** `-k "pins_ministral or carries or keys_cart"` gave 4
  passed, 5 deselected (the `wf_repo` tests).
- **`v8` and `v9` compared by evaluation.** I loaded `git show f02faa69:…/proof_defs.py` beside
  the tree module. With `systemPrompt` and `version` removed, `SALESPERSON_DEF` is equal
  (`True`), and every other upper-case module constant is unchanged. A sentence diff of the two
  prompts shows exactly D7's two edits: the `v8` name-matching clause is gone, the nine D7
  sentences match D7's text verbatim, and the US-dollar sentence closes the catalog paragraph.
- **Live state, read-only** (`mcp__cypher__query`): `reference` holds `salesperson` `v8` only,
  and `ws:demo` holds `v7` and `v8`. No `v9` exists on either side, so the publish creates it
  fresh.
- **S5's done-condition grep, verbatim:** it shows three `v8` hits, all history lines in
  `seed_salesperson.sh` (73, 89, 117). `bash -n` passes on all three scripts. A repo sweep for
  `salesperson@v<n>` and `TRIGGER_DEF_VERSION` pins finds no live `v8` left. `config.py`'s
  default is `v1`, and it belongs to the dev `triage` deployment.
- **Prompt mutants:** 9 hand edits to a scratch copy via `PYTHONPATH`. Two were killed: adding `view_cart` to the add sentence, and removing the USD sentence. The 7 survivors are tabled in F1.

### Findings

**F1 (minor): the prompt test leaves three behaviour-relevant properties unpinned, and its
`view_cart` check matches a keyword.** `test_salesperson_v9_keys_cart_writes_on_product_id_and_states_usd`
pins the add and remove sources, "never translate it", "current message", the miss-list sentence
and the USD sentence. These mutants all **survive** (4/4 green):

| Mutant | Why it matters |
|---|---|
| drop "in this same turn" from the add sentence | the turn scope is what the guard enforces |
| drop "copied exactly" | copying is the fix for translated IDs (R1) |
| drop "If nothing in the list matches, say so plainly rather than adding anything." | the only abstention against a wrong-product add (C1's M2 gate) |
| drop "use its `productId`" from the pick sentence | the miss-list → ID link |
| drop "or build it from a product name" | slug-guessing (R1) |
| drop "Look products up by their catalog name." | the lookup-name nudge |
| add "To increase the quantity of an item already in the cart, copy its `productId` from `view_cart`." | a `view_cart` add source without the word "add": exactly the re-fire shape D3 excludes |

The last row is why a keyword list is the wrong tool here. The loop only checks sentences that
contain both `view_cart` and `\badd\b`, so any other wording of a `view_cart` add source gets
through. **Fix: a coverage assertion over the axis, not a list of phrasings.** Assert that the
sentences mentioning `view_cart` are **exactly** two: the negative one, and the one starting
"To remove an item". Then pin "in this same turn" and "copied exactly" inside the add sentence,
and pin the abstention sentence verbatim. I ran this check against the tree (all true) and against
the quantity mutant (the `view_cart` assertion fails).
*Timing:* this changes tests only, never the published `v9` content, so it can go into the S4+S5
commit or follow it. It matters for the next bump, the `v10` that R2 names as the first fallback.
*Stopping condition:* each mutant in the table turns the v9 test red, and the tree stays 4/4
green.

**F2 (minor): the publish reads the working tree, not the commit.** `seed_salesperson.sh` and
`verify_salesperson.sh` both import `falkorchat.proof_defs` through the editable install, which
resolves to `falkor-chat/server/falkorchat/proof_defs.py` in the tree (I checked). They check
against the same constant, so a publish from the committed tree is consistent with
`verify_salesperson.sh`. Two orderings can still burn `v9`:
- **`start_demo.sh` also seeds `salesperson`.** A demo start before the commit publishes `v9`
  from whatever is in the tree at that moment, without anyone running the seed script.
- **Any prompt edit after the first publish** changes the constant but not the stored `v9`
  (create-only). After that, `verify_salesperson.sh` fails for good. This is the burned-`v6`
  shape. A follow-up prompt fix (for example a review suggestion that rewords D7) must therefore
  be a `v10`, never an amend.

Today the tree's `proof_defs.py` is exactly what will be committed, so nothing is wrong yet.
**Fix:** in the U5→U6 hand-off, check `git diff --quiet HEAD -- falkor-chat/server/falkorchat`
before the first `seed_salesperson.sh demo` or `start_demo.sh`. Record in the ledger that `v9`'s
prompt is frozen from that publish on.

**n1 (nit): QUERIES §15.1 reflow.** The edit splits a backticked citation across a line break
(`` `docs/plans/ `` / `` workflow-cart-and-totals-graph.md` ``, line 2628). It was one line at
`f02faa69`, and the split breaks grep and copy-paste of the path. Line 2621 is 122 characters,
and line 2622 is a short fragment. The file has one earlier split citation (3007), so this has
precedent, but it is still worth undoing here.

### Answers to the brief's checks

- **Prompt vs. as-built tools, `_HELD_REASON` and guard sources: consistent.**
  - *Add sources.* The prompt, the `AddToCartTool.productId` description and
    `_HELD_REASON["add_to_cart"]` all name `lookup_product_fact`/`filter_products` and never
    `view_cart`. The guard accepts those two plus `query_graph_data`. The prompt is narrower than
    the guard and never wider, which is the safe direction, and the `_HELD_REASON` comment says
    so on purpose.
  - *Remove sources.* The prompt adds `view_cart`, and so do the schema, the hold reason and
    `_GROUNDING_SOURCES["remove_from_cart"]`.
  - *Self-source rows.* The prompt never names a write tool as an ID source, so dropping those
    rows (Pass 2) has no prompt-side counterpart.
  - *Miss list.* "Pick … from that list and use its `productId`" is grounded: the miss is a
    `lookup_product_fact` result whose `catalog` rows carry `productId`, and
    `_collect_target_values` walks lists.
  - *Shared wording.* The prompt, the schemas and the hold reasons all say "in this (same) turn".
- **"Never translate" vs. the reply-language rule: no clash.** "Never translate" is scoped
  explicitly to "a `productId` is an identifier", and the language rule governs "your entire
  reply". The USD sentence fixes a symbol, not a language. The remaining tension is not a
  contradiction (see Open questions).
- **S4+S5 as one commit, then publish:** safe. The default version is `v9` in the constant, in
  both seed/verify scripts and in `start_demo.sh`. No `v9` exists yet in `reference` or
  `ws:demo`. Ordering caveat in F2.
- **Pinned vs. incidental text:** the exact-phrase pins (USD sentence at the paragraph end, the
  negative `view_cart` sentence) cost little, because any prompt edit already costs a version
  bump and a test edit. The gap is what is left unpinned (F1), not brittleness.
- **§7 rows for S4–S6: all done or rightly skipped.**
  - Done: QUERIES §15.1 (abstention plus the K-053 projection note), the three `AGENTS.md` rows
    (replaced, not appended), both READMEs, the script headers and the `proof_defs.py` `v9`
    note.
  - §16 not edited, correctly: I grepped §16 for name-resolution wording and found none.
  - SERVER.md / DESIGN.md / `docs/manuals/`: grepping for
    `productName|add_to_cart|K-058|off-turn` gives zero hits, so the no-change row holds. The
    one `v8` mention in SERVER.md (726) is history.
  - Owned by others: HISTORY/BACKLOG (`teco` at close), and D3's table and D2's wording
    (`architect`, Pass 2 carry-over, still open).
- **§15.1 claims vs. code: accurate.** `lookup_product` → `None` → `{"found": false, "hint",
  "catalog"}` (`tools.py` `LookupProductFactTool.run`). `catalog_index` is the whole-catalog
  `repository.filter_products(None, None, None, limit=CATALOG_INDEX_LIMIT=50)`, projected to
  `{productId, name}` and sorted by name. The cart services resolve through
  `lookup_product_by_id` → `lookup_products_by_id` (§16.9).

### What's solid

- The prompt edit is D7 verbatim, and every carried-forward field is byte-equal to `v8`'s.
- The prompt names ID sources per tool exactly the way the schemas and hold reasons do, which
  closes the plan gate's F3 on the model-facing side.
- The pins moved in all six places in one change, and the documentation was replaced in place,
  never appended.

### Open questions (for the S7 brief, not fixes)

- **Tool arguments are not carved out of the language rule.** "Governs your entire reply … no
  exceptions" still pushes a pt/es model to translate the lookup name. "Look products up by their
  catalog name" does not say that the catalog is English. The design accepts this and relies on
  the miss-list recovery, so the cost shows up as iterations, which the turn-2 p95 gate covers.
  Worth reporting in C1: the first-lookup miss rate per language. A carve-out is a `v10`
  decision for `architect`, and only if C1 shows the latency cost.
- **A confirmation-turn add** ("yes, add it" after the model offers) is in neither TP-007 nor C2.
  "Only … what the customer asked for in their current message" could make a 3B model refuse it.
  `v8`'s text guard would also have held that turn, so this is not a regression. It is unmeasured,
  though, and worth one scripted C2 variant if `qa-engineer` has budget.
