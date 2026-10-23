# Cart writes by product ID: filter-miss recovery and customer speaker label (S7 fixes)

> **Status:** active · **Owner:** `architect` · **Tracks:** K-078 · **Extends:** `falkor-chat/docs/plans/salesperson-cart-product-id.md` · **Version:** 2 · **Reviews:** `falkor-chat/docs/reviews/salesperson-cart-product-id2.md`

*2026-10-23 v2: revised per analyst Pass 1 (M1, m1–m4, n1–n3): turn 0–1 language pre-check with a stop rule, D8 measured in C1 not C2, pt-BR latency risk, contract-pin mutants, a category-browse variant, D8 size ≈ +445–457, test-40 capture, the `_recent_turns` note. Wrong-language gate ≤ 3/375 confirmed by `teco`.*

## 1. Goal & scope

**Goal.** Make the K-078 live check pass on the already-published `salesperson@v9` by fixing two
defects in code, then re-measuring:
- **DEF-2:** a `filter_products` call whose `category` names no catalog category returns the
  same recovery the `lookup_product_fact` miss does (a hint and the catalog list).
- **DEF-4:** in the storefront deployment, customer lines reach the model under a fixed
  `Customer` speaker label, not the participant's display name.

The RCA's diagnosis (`falkor-chat/docs/reviews/salesperson-cart-product-id-rca.md`) is the input.
This note judges its fix set and turns it into steps.

**Given (stakeholder, 2026-10-23):**
- Concurrent storefront turns are capped at 2. `devops` is landing
  `FALKORCHAT_STOREFRONT_TURN_WORKERS` default 2 in `start_demo.sh` (RCA O1).
- C2 es: fix the trigger (DEF-4) and re-measure. The R3 guard is not redesigned now.
- No `v10`. `salesperson@v9`'s prompt is frozen, so every change here is tool, executor or
  wiring code.

**Out of scope:**
- the R3 guard redesign (original plan §8 R3 candidates), unless C2 escalates again;
- DEF-5 (`R$` in pt-BR) and any prompt change;
- RCA O2 (LM Studio `--parallel 2`), O3 (larger context), O6 (retry on context overflow);
- catalog compaction (RCA §5.3 (iv)), deferred, see D11;
- the storefront's profile seeding of the display name (`Storefront.join`), see D10;
- `start_demo.sh` and the docs that mention `TURN_WORKERS` (`devops` owns them now).

## 2. Context & findings

CPG: considered, not relevant — the change touches four functions read directly from the current
tree (`2c741569`), and `cpg_falkorchat` is stale per the original plan §2.

**Verified by reading the source:**

- **`tools.py` `FilterProductsTool.run`.** On no rows it returns
  `{"items": [], "finding": "no matching products found"}`, whatever emptied the result. The
  `category` schema description is "Only products in this category.", and it names no category.
- **`repository.py` `Repository.filter_products`** owns category matching. It normalizes the raw
  `category` with `extraction.normalize_name` and compares it with `Product.categoryNormalized`.
  A blank `category` normalizes to `""`, which it treats as omitted (`None`). Neither
  `Services.filter_products` nor the tool normalizes anything.
- **`tools.py` `LookupProductFactTool.run`** builds the D2 miss as
  `{"found": false, "hint": LOOKUP_MISS_HINT, "catalog": Services.catalog_index(ctx)}`.
  `LOOKUP_MISS_HINT` says "call lookup_product_fact **again** with its exact catalog name from
  this list". That wording is wrong for a `filter_products` miss (D8).
- **`executor.py` grounding.** `_GROUNDING_SOURCES["add_to_cart"]` is `{lookup_product_fact,
  filter_products, query_graph_data}`, and `remove_from_cart` adds `view_cart`. After every
  successful dispatch, `_handle_tool_call` runs `_collect_target_values(content, "productId")`,
  which walks every dict and list in the result. So a `catalog` list inside a `filter_products`
  result grounds adds and removes **with no executor change**.
- **`executor.py` `WorkflowExecutor._assemble_messages`** is a `@staticmethod(config, run_ctx,
  thread_msgs)`. For every thread row it uses `speaker = displayName or authorId or "member"` and
  appends `f"{speaker}: {text}"` through `_append_turn`. Assistant rows get the agent's display
  name the same way (`Assistant: …` in the S7 traffic). There is one call site, in
  `_run_agent_node`, plus static calls in `tests/test_executor_agent.py`.
- **`WorkflowExecutor.__init__`** is all keyword-only after `services, repo`. It is constructed
  in `app.py` `_build_default_app` inside the `config.WORKFLOW_ENABLED` branch, where
  `config.STOREFRONT_ENABLED` (`FALKORCHAT_STOREFRONT_ENABLED`) is already read. It drives
  `dev_surface = not config.STOREFRONT_ENABLED`, and `create_app` refuses storefront and dev
  surface together.
- **Storefront threads have exactly one human author.** `Storefront.join` gives each participant
  their own `Channel`/`Thread` (`Repository.ensure_participant`). The only route that posts is
  `POST /shop/api/messages`, which posts into `who.thread_id`, the caller's own thread. The
  presenter routes (`/presenter/session`, `/presenter/participants`, `/presenter/reset-all`) post
  nothing.
- **In the storefront process, every in-flight ministral sequence is a storefront turn.**
  `order-fulfillment@v1` has only `decision`/`human` steps. `salesperson@v9`'s one transition
  guard is `{"kind": "cmp"}`, so the LLM judge never runs. The storefront mounts no `/mcp`, so
  there is no ingestion. The internal `query_graph_data` call runs inside a turn's tool call,
  while that turn's agent call is not in flight. The embedder has its own KV cache. So with
  `:8200` idle, ministral concurrency is at most `STOREFRONT_TURN_WORKERS`.

**DEF-4: who reads the prefix, and where else the name reaches the model.** Prefix readers come
from a grep for `displayName`/`speaker` over `server/falkorchat/`. The traffic counts come from my
read-only re-scan of S7's `proxy.jsonl`, 3,362 agent calls (session scratchpad, not durable,
beside the RCA's `k078-s7/` and `rca/`: `k078-design/leak.py`, `vectors.py`):

| Path | Reads the prefix? | Carries the display name to the model? |
|---|---|---|
| K-065 language logic | **No code reads it.** Language comes from `CONTEXT.language` plus the two `systemPrompt` sentences. But the probe's names end in the language code (`LSP-…-t001-es`), so every S7 customer line carried an incidental language cue (R-B) | — |
| `get_profile` | No. It reads `Customer` by `ctx.actor` | **Yes.** `Storefront.join` seeds `save_profile(name=display_name)`, so a `get_profile` result is `{"name": "LSP-…", …}` in the turn that calls it. 360 calls in S7, and 692 requests carried such a result in their tail |
| `save_profile` | No. Its `name` is whatever the model passes | Rarely: 1 S7 call passed the display name |
| The model's own replies | — | **Yes.** 135 replies greet by name ("Olá, LSP-392b38-t000-pt-BR!"). They are replayed as assistant history: 174 C1 requests carried one, 0 in C2 or the confirm run |
| `CONTEXT` block | — | **No.** The storefront's run ctx is `{language, threadId}`, and 0 requests had the name there |
| `guards.py` `_recent_turns` | Builds the judge's transcript from `displayName` | Not in the storefront, because the salesperson guard is `cmp` |
| `tools.py` `PostMessageTool` | Its comment explains the `mentions` name-vs-id error by the prefix | Not storefront-relevant |

**Which vector the 52 display-name `productId`s used** (my recount; the RCA's §3.2 split holds):
- 51 had the name **only** in the speaker prefix: no echoed reply in the history, no name-bearing
  tool result in the turn.
- 1 (C2 es) also had a name-bearing profile-tool result in its tail. The RCA says "none". That is a
  one-call discrepancy and changes nothing.

So the prefix is the active vector. The other two (the profile result, and the model's own
greeting) remain after D9. D10 settles them.

**Token figures re-derived** from the RCA's own per-call dataset (`rca/rows.pkl`, successful
calls only; `k078-design/budget.py`, `pairs.py`):
- largest single sequence (prompt + completion): **3,595** (C1 pt-BR, turn 4);
- largest cross-thread **pair** in flight at once: **7,074** (C1 pt-BR + es, both turn 4);
- pair p99 5,973 and p95 5,475.

The RCA's "~6.4k 2-way worst" is low: 6.4k is the worst pair *inside the failure groups*, not
across the run. Failed calls have no `usage`, so 7,074 is still a lower bound on S7's true 2-way
peak. §4 uses it.

## 3. Design & rationale

Decisions continue the original plan's numbering (D1–D7), so the two documents never share a
number. Each is stated once here, and later sections cite it.

**D8: DEF-2. A `filter_products` call naming an unknown category returns D2's recovery.**
I agree with the RCA's option (a). It is the only option that needs no `v10` and doesn't charge
every call. Three refinements answer the brief's three questions.

1. **How "no catalog category matches" is decided: by the repository, never by a second
   normalizer.** Add `Services.has_category(ctx, *, category: str) -> bool`:
   `bool(self._repo.filter_products(category=category, min_price=None, max_price=None,
   limit=1))`.
   - It passes the raw `category` through. `Repository.filter_products` normalizes it with
     `normalize_name` and compares it with `Product.categoryNormalized`, exactly as the original
     call did. A blank category is omitted there, so `has_category("")` is `True` on a non-empty
     catalog, and a blank never triggers the miss.
   - It is **without the price bounds** on purpose. The pt-BR calls carried bounds
     (`minPrice` 0–1, `maxPrice` 100 or 999). A check that reused the call's own bounds could not
     tell "unknown category" from "known category, nothing in range".
   - No new Cypher. It is one extra `ro_query` on `reference`, and only on an empty result.
   - *Rejected:* comparing `normalize_name(category)` in the tool or the service with the rows'
     raw `category`. That is a second normalization site, and it is only correct while the seed
     keeps `categoryNormalized == normalize_name(category)`, which is K-058's "no second
     normalizer" rule.
2. **What the result carries.** `FilterProductsTool.run` with empty rows, `category is not None`
   and `not has_category(ctx, category=category)` returns:

   ```json
   {"items": [], "finding": FILTER_CATEGORY_MISS_FINDING, "hint": FILTER_CATEGORY_MISS_HINT,
    "catalog": <Services.catalog_index(ctx)>}
   ```

   - The two new module constants sit in `tools.py` beside `LOOKUP_MISS_HINT`:
     - `FILTER_CATEGORY_MISS_FINDING = "no catalog category has that name"`;
     - `FILTER_CATEGORY_MISS_HINT = "that is not a catalog category, so this is every product in
       the catalog instead; if the customer named a product, pick it from this list and use its
       productId; to get a product's price or category, call lookup_product_fact with its exact
       catalog name from this list"`.
   - **The hint is not `LOOKUP_MISS_HINT`.** That one says "call lookup_product_fact **again**",
     which is false after a `filter_products` call. The frozen `v9` prompt tells the model to pick
     from the list only "if `lookup_product_fact` finds no exact match". So this result has to
     carry the pick instruction itself. The price half of the hint keeps D2's K-079 purpose: the
     list carries no prices, so a price question goes to an exact-name lookup.
   - **The finding no longer says "no matching products found"** on this branch. That phrase is
     the text the pt-BR model turned into "não encontrei" 35 times (R2's shape).
   - **`items: []` stays**, so every `filter_products` result keeps an `items` key, as its
     docstring promises.
   - **`catalog` is `catalog_index`, unchanged:** `{productId, name}` sorted by name, with no
     price and no category (D2, K-079).
   - **Every other empty result is byte-identical to today:** a known category emptied by a price
     range, a blank category, no category, or a price-only filter.
3. **How it reaches the guard's grounding: exactly as a lookup miss does, with no executor
   change.** `filter_products` is already in both write tools' `_GROUNDING_SOURCES` rows, and
   `_collect_target_values` walks the `catalog` list. So after a category miss every catalog ID
   is grounded for adds and removes for the rest of that node execution.
   - This widens original D3's accepted residual (*"a failed lookup's `catalog` list … grounds
     every product for the rest of that turn"*) to one more trigger. It is not a new class of hole.
   - `filter_products(category="")` already listed the whole catalog with IDs.
   - It is measured in **C1**, the only run that takes the filter path (S7: all 37
     `filter_products` calls were C1, 36 pt-BR + 1 en, turn indices 2–3; C2 and the confirm run
     had 0): M2/M3 plus §6.2 measure 7.

*Rejected alternatives* are RCA §3.3 (b)–(d), for its reasons. I also rejected **adding a
`categories` list** to the result (~12 tokens), which would let a translated category browse
retry `filter_products(category="Peripherals")`. That retry returns priced rows (~35 tokens each)
on a path the frozen prompt never mentions, and the §4 margin is too thin to fund it.
*Reversal trigger:* in §6.2's browse variant, ≥ 3/24 replies either present the whole list as
the category's contents or end without an answer. The hint's opening ("that is not a catalog
category") is the mitigation for the first; neither probe asked a browse question before, so the
variant is what makes the trigger observable.

**D9: DEF-4. In the storefront deployment, customer lines carry a fixed `Customer` label.**

*Premise check: the RCA is right on all three counts.*
- Step `config` is create-only (`falkor-chat/AGENTS.md`, trap 1), and `verify_salesperson.sh`
  compares every stored `config`, so a config switch means `v10`.
- Storefront threads have one human author (§2), so a label loses no information there.
- Threads with several humans exist only on the dev surface, which the switch never reaches.

*The seam is a constructor argument on `WorkflowExecutor`, set by `_build_default_app` from
`config.STOREFRONT_ENABLED`.*
- `WorkflowExecutor.__init__(…, human_speaker_label: str | None = None)` stores
  `self._human_speaker_label`.
- `_assemble_messages(config, run_ctx, thread_msgs, *, human_speaker_label: str | None = None)`
  stays a `@staticmethod`. When the label is set, a row whose `role != "assistant"` renders as
  `f"{human_speaker_label}: {text}"`. Assistant rows keep `displayName or authorId or "member"`,
  so the agent's own lines stay byte-identical. When the label is `None`, the output is
  byte-identical to today.
- `_run_agent_node` passes `human_speaker_label=self._human_speaker_label` at its one call site.
- `storefront.py` gains `CUSTOMER_SPEAKER_LABEL = "Customer"`. It matches the system prompt's
  "chatting with a customer", and the storefront owns that vocabulary.
- `app.py` `_build_default_app` passes
  `human_speaker_label=CUSTOMER_SPEAKER_LABEL if config.STOREFRONT_ENABLED else None` to its one
  `WorkflowExecutor(…)`. It is testable offline with the existing
  `tests/test_app.py` `test_workflow_wiring_is_gated_on_workflow_enabled` pattern, which
  monkeypatches `create_app` and reads the executor back as `services._executor`.
- **Out of D9's reach, deliberately:** `guards.py` `_recent_turns` still names speakers by
  `displayName` for an LLM judge. That is unreachable in the storefront (the salesperson guard is
  `cmp`). S4 adds one docstring line there saying so, so a future storefront def with an LLM
  guard doesn't rediscover it. An operator re-pointing `TRIGGER_DEF_KEY` in storefront mode also
  gets `Customer:`, which is harmless because those threads are single-human too.
- **No new env var, no `start_demo.sh` edit** (`devops` owns that file now), and no def or
  config change. `start_demo.sh` already sets `FALKORCHAT_STOREFRONT_ENABLED`. Every other
  construction of `WorkflowExecutor`, in tests and in the dev app, gets `None`.

*Rejected seams:*
- **Step `config`:** forces `v10`.
- **A run-ctx key from `Storefront`:** it renders into the `CONTEXT` block, costing tokens on
  every call. It is stored per run, so runs that are already parked would not carry it. It also
  pushes presentation into run data.
- **"One human in the thread" detection inside the executor:** it changes every single-human dev
  thread (`triage@v1`, `@bot1`) as well. It needs a member read per turn, and it guesses at
  deployment intent from data.
- **Renaming the probe's participants** (`language_salience_probe.py`): it changes the instrument
  against S7 and hides a product risk. A real customer may type a handle such as `joao-silva`.

**D10: the other two name vectors stay, and are measured.**
- The `get_profile` result and the model's greeting-by-name both come from
  `Storefront.join`'s profile seeding (`salesperson-ui` §4.10: the profile panel shows the typed
  name at once). Removing them means changing a requirement, or the frozen prompt's
  "call `get_profile` once, early".
- The evidence favours the prefix as the bait: 51/52 bait calls had no other vector, and no C2 or
  confirm-run request carried an echoed name (§2).
- *Reversal trigger:* the re-run attributes every display-name `productId` by vector (§6.2). Three
  or more via the profile or echo vectors across C2 and the confirm run go back to the
  stakeholder as a §4.10 question. I am not designing that change now.

**D11: catalog compaction stays deferred** (RCA §5.3 (iv)), for one more reason the RCA didn't
give. `[id, name]` pairs carry no `productId` key, so `_collect_target_values` would stop
grounding them. Compaction is therefore an executor change too, not a tool-result trim.
*Reversal trigger:* the re-run's measured 2-way peak exceeds 7,373 (90% of the pool, §4).

**D12: carry-overs from the code reviews.** They correct the original plan's text to match what
was built. They change no code.
- **D3 as built has no self-source rows** (`falkor-chat/docs/reviews/salesperson-cart-product-id-impl.md`
  Pass 2, n1). The live table is:
  - `_GROUNDING_SOURCES["add_to_cart"] = {lookup_product_fact, filter_products, query_graph_data}`;
  - `_GROUNDING_SOURCES["remove_from_cart"] =` the same plus `view_cart`.

  A write is never a source for itself. Restoring the rows is behaviour-equivalent (496/496 green
  either way), so original D3's table and §5 S3 bullet read with this correction.
- **D2's sort note reads "a misjoin is impossible against `filter_products`"**, not "either
  way" (impl review F4). The miss `catalog` is a name-only list. Until K-079 F2 lands,
  `query_graph_data` can still return an identity-free price list, which together with it makes
  RC-1's pair. S7 saw 0 such turns. D8's `catalog` inherits the same statement.

**Interface changes, called out:**
- `filter_products`' model-facing result on the unknown-category branch gains `hint` and
  `catalog`, and its `finding` text changes;
- new `Services.has_category`;
- new keyword `WorkflowExecutor(human_speaker_label=…)` and `_assemble_messages(…,
  human_speaker_label=…)`, both defaulting to today's behaviour;
- the storefront's model-visible transcript (customer lines read `Customer: …`).

There is no graph, query, def or env-var change.

## 4. Token budget under the 2-worker cap

**Binding quantity.** The pool is one 8,192-cell unified KV cache (ministral `-c 8192`,
PARALLEL 4, `kv_unified`; RCA §3.1). With `STOREFRONT_TURN_WORKERS=2` and `:8200` idle, at most
two ministral sequences are in flight at once (§2). Demand is the sum of their lengths, each
prompt + completion.

**Per-component deltas:**

| Component | Δ per sequence | Source |
|---|---|---|
| `v9` constant (prompt + schemas) | +227, already inside every S7 figure | RCA §2 item 2, confirmed |
| `lookup_product_fact` miss + `catalog` | +397 | RCA §2 item 3, n = 121, confirmed |
| **D8** `filter_products` category miss | **≈ 445–457** for the result, a net **≈ +430–441** over the 16-token empty it replaces | analyst Pass 1: 1,394 chars built from the real catalog and D8's strings, against the lookup miss's 1,211 chars = 397 tokens; no tokenizer run |
| **D8** follow-on calls on that turn | +42 (add ok) to +93 (a found lookup, then the add) | RCA §2 item 3 |
| **D9** `Customer` label | **−8 to −13 per customer line**: `LSP-358e50-t001-es` against one word, so up to −65 on turn 4 (5 customer lines) | RCA O4 gives −8; the upper end is my inference from the slug's length; no tokenizer run |
| Held add (D3 reason) | +103 to +111 | RCA §2 item 3 |

**Peak 2-way demand:**

| Scenario: two sequences in flight | Pair demand | Margin to 8,192 |
|---|---|---|
| S7 measured, worst cross-thread pair of successful calls (C1 pt-BR + es, turn 4) | **7,074** (lower bound, since failed calls have no `usage`) | 1,118 (13.6%) |
| Hard bound from S7: 2 × the largest single sequence (3,595) | 7,190 | 1,002 (12.2%) |
| + D8 on one sequence of that worst pair | ≈ 7,515 | ≈ 680 (8.3%) |
| + D8 on one sequence and D9 on both (turn 4) | ≈ 7,385–7,435 | ≈ 760–810 (9.2–9.9%) |
| Stacked worst single path, twice: turn-4 base max 2,684 + filter miss 457 + lookup miss 397 + held 111 + found 51 + ok 49 + completion 80, minus D9 65 ≈ 3,764 each | ≈ 7,530 | ≈ 660 (8.1%) |
| **Expected once the dead-turn cascade is gone:** the worst turn-2 add pair, es lookup path (max 3,086) + pt-BR on D8's path (≈ es's), minus D9 | **≈ 6,100–6,200** | **≈ 2,000 (24%)** |
| For reference, 3-way with no cap (S7 measured) | 8,863 | overflow |

**Reading the table.**
- Every 2-way scenario fits, so DEF-1 cannot recur from these two changes alone. `analyst`
  re-derived the worst S7 pair from the runtime log too (7,072, against the proxy's 7,074), and
  confirmed that a turn cannot outlive its worker, so the cap really bounds in-flight sequences
  to 2 (review Pass 1, point 1).
- The rows under the RCA's 10% review line (RCA §5.2 "Prevention") are all made of **turn-4
  sequences that re-add after a dead turn**. S7's six largest sequences are all that shape: a
  `lookup_product_fact` miss plus an add on the order-status turn, in pt-BR and es. They exist
  because of DEF-1's cascade, which the cap removes. The realistic peak is the "expected" row.
- The RCA's 2-way prior (~6.4k, ~1.8k headroom) understates S7's own pair maximum by ~670
  tokens. The cap is still sufficient, but the margin case for any future `v10` is ≈ 0.8–1.1k,
  not 1.8k.
- **Reversal trigger for D11:** a measured 2-way peak above 7,373 (90% of the pool) in the
  re-run.

## 5. Step-by-step implementation

**Conventions.**
- Paths are relative to `falkor-chat/server/` unless rooted. Branch from `2c741569` or later.
- Run **only the named test files** with the venv, from `falkor-chat/server/`:
  `test_services.py` (FakeRepo), `test_tools.py` and `test_executor_agent.py` (stubs), and
  `test_app.py -k test_<name>` for the one monkeypatch-only wiring test. The rest of
  `test_app.py` uses `conn`.
- **Never** run the default full suite, `wf_repo`-requesting tests, `test_storefront.py`,
  `test_storefront_api.py` or `test_salesperson_scaffold.py` (original plan §5; impl review F5).
- Mutants are hand edits, reverted after each one (original §6.2 practice).

| Step | What | Files | Done when |
|---|---|---|---|
| S1 | `Services.has_category` (D8.1) | `falkorchat/services.py`; `tests/test_services.py` | tests 28–29 green; mutants N1–N2 each turn ≥ 1 test red |
| S2 | The `filter_products` category-miss branch (D8.2) plus a grounding pin (D8.3) | `falkorchat/tools.py` (two constants, `FilterProductsTool.run`, docstrings for the class and its `category` behaviour); `tests/test_tools.py` (`StubServices` gains `has_category`; **rewrite** `test_filter_products_abstains_when_nothing_matches`, whose `"nonexistent"` category is now the miss branch, into tests 30 and 32); `tests/test_executor_agent.py` (test 35) | tests 30–35 green; N3–N7 killed. **S1+S2 are one commit** |
| S3 | The executor speaker label (D9) | `falkorchat/executor.py` (`WorkflowExecutor.__init__`, `_assemble_messages` signature and docstring, the `_run_agent_node` call site); `tests/test_executor_agent.py` | tests 36–39 green; every existing `_assemble_messages`/thread-fold test passes **unmodified**; N8–N11 killed |
| S4 | Storefront wiring (D9) | `falkorchat/storefront.py` (`CUSTOMER_SPEAKER_LABEL`); `falkorchat/app.py` (`_build_default_app`, one keyword); `tests/test_app.py` (test 40); `falkorchat/guards.py` `_recent_turns` docstring (one line, D9); `falkorchat/tools.py` `PostMessageTool` comment ("`{displayName}: {text}`, or `Customer: {text}` in the storefront") | test 40 green; N12 killed. **S3+S4 are one commit**: S3 alone is dead code, and S4 alone does not import |
| S5 | Documentation | §7 rows marked "implementer" | every such row is done. `grep -n "again with its exact" falkorchat/tools.py` matches only `LOOKUP_MISS_HINT` |
| S6 | The re-run | `qa-engineer`, §6.2 | numbers recorded in `falkor-chat/docs/test-reports/salesperson-cart-product-id2-report.md`, gates per §6.2 |

**S2 sketch** (the logic the tests pin; the wording is in D8.2):

```python
rows = self._services.filter_products(ctx, category=category, ...)
if rows:
    return json.dumps({"items": rows})
if category is not None and not self._services.has_category(ctx, category=category):
    return json.dumps({"items": [], "finding": FILTER_CATEGORY_MISS_FINDING,
                       "hint": FILTER_CATEGORY_MISS_HINT,
                       "catalog": self._services.catalog_index(ctx)})
return json.dumps({"items": [], "finding": "no matching products found"})
```

`category` is `arguments.get("category")`, read once and passed to both calls. There is no
`isinstance`/strip in the tool: a blank category reaches `has_category`, which the repository
answers as omitted, so it returns `True` and the call keeps today's result.

**No code step for D10, D11 or D12.** D12's two corrections are this note's text; the original
plan stays untouched except for its header pointer.

## 6. Test strategy

### 6.1 Offline reproduction tests, in red→green order

The numbering continues the original plan's §6.1 (1–27). Each test is one test or a small
cluster.

*`Services.has_category` (S1, `test_services.py`, FakeRepo)*

28. `has_category(category="Peripherals")` calls `repo.filter_products` exactly once, with
    `category="Peripherals"` passed through **raw**, `min_price=None`, `max_price=None` and
    `limit=1`. It returns `True` when the fake returns a row and `False` when it returns `[]`.
29. A raw `"  peripherals "` reaches the repository unchanged: the service never normalizes. The
    repository-side normalization is already pinned by the `test_repository.py` `filter_products`
    category tests, which use `wf_repo`; don't add or run a live test for it here.

*`filter_products` result shapes (S2, `test_tools.py`, `StubServices`)*

30. **The DEF-2 reproduction.** The pt-BR S7 call shape is `{"category": "Mouse sem fio Pro",
    "minPrice": 0, "maxPrice": 100}`, with `filter_result=[]` and `has_category → False`. It
    returns:
    - `items == []`;
    - `finding == FILTER_CATEGORY_MISS_FINDING`;
    - `hint == FILTER_CATEGORY_MISS_HINT`;
    - `catalog ==` the stub's `catalog_index`, every entry with exactly the keys `productId` and
      `name`.
31. The same with no price bounds (`{"category": "Mouse sem fio"}`) gives the miss branch too.
32. **A known category emptied by a price range is unchanged.** With `has_category → True` and
    `filter_result=[]`, the output is **exactly** `{"items": [], "finding": "no matching
    products found"}`. This is the rewritten half of the old abstention test.
33. **No category is unchanged.** A price-only filter that empties (`{"maxPrice": 1}`) gives the
    old empty shape, and `has_category` is **not called**. A blank `{"category": ""}` with
    `has_category → True` gives the old shape.
34. **The hint is right for this tool.** `FILTER_CATEGORY_MISS_HINT`:
    - contains `lookup_product_fact` and `productId`;
    - does **not** contain `again`;
    - contains no `$` and no digit;
    - differs from `LOOKUP_MISS_HINT`.

    A non-empty result still returns `{"items": rows}` (the existing
    `test_filter_products_returns_items` stays).

*Grounding (S2, `test_executor_agent.py`, stub LLM and registry)*

35. **A category-miss list grounds an add.** The scripted turn is `filter_products` (its stub
    returns test 30's JSON, with `wireless-mouse-pro` in `catalog`), then
    `add_to_cart("wireless-mouse-pro")` → **dispatched**, then `add_to_cart("mouse-sem-fio-pro")`
    → **held** with `_HELD_REASON["add_to_cart"]`. This pins D8.3: the D3 residual extends to
    this path, deliberately.

*Speaker label (S3, `test_executor_agent.py`)*

36. **No display name in the storefront prompt.** Call `_assemble_messages(config, {"language":
    "es", "threadId": "t1"}, rows, human_speaker_label="Customer")` with the user rows' display
    name `LSP-358e50-t003-es` and an assistant row named `Assistant`. Then:
    - every user line reads `Customer: <text>`;
    - the assistant line reads `Assistant: <text>`;
    - `LSP-358e50-t003-es` appears **nowhere** in the joined message contents, `CONTEXT`
      included.
37. **The default is byte-identical.** The same rows without the keyword produce the exact list
    the current code produces: a literal expected list, with `LSP-…: ` prefixes. The existing
    tests that already pin it (`test_agent_node_folds_thread_messages_into_prompt`, the K-048
    alternation tests, the `toolsUsed` tests) pass unmodified.
38. **K-048 still coalesces under the label.** Two consecutive customer rows, then `CONTEXT`,
    give one `user` message that contains both `Customer:` lines.
39. **The executor threads the label through.** `WorkflowExecutor(…, human_speaker_label=
    "Customer")` with `StubThreadServices` (display name `Alice`), then `_run_agent_node`. The
    LLM stub's first call has `Customer: reset my password` and no `Alice`.

*Wiring (S4, `test_app.py`, monkeypatch only)*

40. With the `test_workflow_wiring_is_gated_on_workflow_enabled` pattern (`ENABLE_AGENT`,
    `WORKFLOW_ENABLED` on, `create_app` faked):
    - `STOREFRONT_ENABLED=True` gives
      `services._executor._human_speaker_label == storefront.CUSTOMER_SPEAKER_LABEL`;
    - `STOREFRONT_ENABLED=False` gives `None`.

    The existing `fake_create_app(services=None, **kwargs)` keeps only `kwargs`, so this test's
    fake must also capture `services`.

**Named mutants.** Each must turn at least one test above red.

A **contract pin** asserts the call shape (one read, raw pass-through, no read on a price-only
miss), not output. A later refactor that legitimately changes the call shape updates the pin;
it is not a regression.

| ID | Mutation | Killed by |
|---|---|---|
| N1 | `has_category` passes a non-`None` price bound (e.g. `min_price=0`) or a `limit` other than 1 to the repository | 28, **contract pin (behaviour-equivalent):** every price is > 0 and any `limit` gives the same `bool` |
| N2 | `has_category` normalizes in Python (`normalize_name(category)`) before calling the repository; or compares against `catalog_index`/row `category` instead of calling it | 28, 29. The normalize-first variant is a **contract pin (behaviour-equivalent):** `normalize_name` is idempotent. The compare-in-Python variant is a second normalizer, the rule D8.1 forbids |
| N3 | the miss branch fires on **any** empty result (no `has_category` check) | 32 |
| N4 | the miss branch reuses `LOOKUP_MISS_HINT` | 34, 30 |
| N5 | the miss `catalog` is the whole-catalog `filter_products` rows (with `price`/`category`) | 30 |
| N6 | `has_category` is called when `category is None` (an extra read on every price-only miss) | 33, **contract pin (behaviour-equivalent):** `None` → whole catalog → `True` → the old shape |
| N7 | `filter_products` removed from `_GROUNDING_SOURCES["add_to_cart"]` | 35 |
| N8 | the label also replaces assistant rows' speaker | 36 |
| N9 | `displayName` keeps precedence (`m.get("displayName") or label`) | 36 |
| N10 | the `human_speaker_label` default is `"Customer"`, so dev threads change | 37 |
| N11 | the `_run_agent_node` call site doesn't pass `self._human_speaker_label` | 39 |
| N12 | the wiring is inverted, or always `None` | 40 |

### 6.2 Re-run spec (`qa-engineer`, after S1–S4 and `devops`' cap are in)

The new test-plan and report are **successors**, `falkor-chat/docs/test-plans/salesperson-cart-product-id2.md`
and `falkor-chat/docs/test-reports/salesperson-cart-product-id2-report.md`. The S7 documents were
executed against, so they stay intact (root `AGENTS.md`, collision rule 5).

**Setup** (as S7, original §6.3):
- `verify_salesperson.sh demo` and `verify_catalog.sh` exit 0 before and after;
- `start_demo.sh` with the 2-worker default (log the effective value), ministral `-c 8192`
  PARALLEL 4 plus the embedder, `qwen` absent, `:8200` idle;
- the observe-only proxy on `:1235` exactly as in S7;
- LM Studio's runtime log captured from **before** the first turn (S7's started late, at
  14:25:18).

**Warm-up (DEF-6):** one throwaway participant, one turn ("Hi there!"), before trial 0. Report
it on its own line, never in a gate.

**Language pre-check (before C1; review M1).** D9 removes the only in-line language cue. The
probe's customer text is always English (`language_salience_probe.py` `SCRIPTED_TURNS`), so after
D9 a turn-0 line reads `Customer: Hi there!` and `CONTEXT.language` is the only cue left. Every
earlier 0-wrong-language result (U3, K-065, S7) was taken with the suffix present.
- Run `language_salience_probe.py run --trials 12 --turns 2` (all three languages: 36
  conversations, 72 turns) on the same deployment, after the warm-up.
- **Stop rule:** **any** wrong-language reply, in **any** language, stops the run before C1 and
  goes back to `architect`. The review's rule names pt-BR and es; I added en, because K-065's
  original defect was an en participant answered in Spanish, and the `-en` suffix is gone too.
- Report it on its own, never in a C1 gate.
- *Fallback if it stops* (a design question for `architect`, not pre-decided): a label that
  carries the language with no `v10`, e.g. one rendered from `CONTEXT.language`.

**Runs:**
- **C1:** `language_salience_probe.py run --trials 75`, the same command as S7. The probe still
  drives three participants per trial, so the third queues.
- **C2:** `refire_probe.py`, both schedules, **en, pt-BR and es**, n = 24 each.
- **Confirmation variant:** the S7 TP-011 script, 12 trials × 3 languages.
- **Browse variant** (review m4; D8's `categories` trigger): one new `refire_probe.py` script
  (`qa-engineer` adds it to `SCRIPTS`): "Do you have any periféricos?", 12 trials × pt-BR and es.
  It is reported, not gated. Classify each reply: lists the peripherals (correct); presents the
  whole catalog as peripherals; or ends without an answer.

**Gates (C1; original §6.3 as set by the stakeholder):**

| Measure | Gate |
|---|---|
| M1 es, M1 pt-BR | each ≥ 60/75 |
| M1 en | ≥ 65/75 |
| M2 (other-product lines), M3 (mouse quantity > 1) | each ≤ 2/225, each hand-inspected |
| Dead turns | ≤ 1.2% (≤ 13 of 1,125), warm-up excluded |
| Turn p95, all turns and turn index 2, pooled | ≤ 29 s each, measured from post to reply, so queue wait counts |
| Wrong-language replies | ≤ 3/375 per language (original §6.3, confirmed by `teco` 2026-10-23). **Reported by turn index × language.** Attribution rule: a rise confined to turns 0–1 is D9's, because D8 cannot reach those turns (S7 had 0 `filter_products` calls there) |

**C2 is reported, not gated.** ≥ 3/24 in **any** language goes back to the stakeholder before
K-078 closes, and that is where R3's redesign returns.

**Newly reported measures** (RCA §5.3 item 4, plus this note's):
1. **KV headroom:** the peak 2-way and 3-way in-flight KV from `lms-runtime.log`, by `slots.py`'s
   method; the margin to 8,192; the max concurrently active slots (expect ≤ 2); and the launch
   instants above 7,373 (D11's trigger).
2. **Latency next to the dead-turn rate,** never alone. Report p95 over replied turns and over
   all turns (a dead turn counted at its failure time), plus the queue wait (post accepted → turn
   start) p50/p95 **per language**. pt-BR's D8 path turns a one-call reply into about three
   calls, so the trial's queued third participant waits longer.
3. **C2 es turn 2:** classify the first mouse-directed call as RCA §3.2 (a) guessed or translated
   slug, held; (b) lookup first, an un-held re-add; or (c) no attempt.
4. **Display-name `productId`s by vector:** speaker prefix (expected 0); `get_profile` result in
   the same turn; an echoed name in the assistant history. D10's trigger is ≥ 3 via the last two
   across C2 and the confirm run.
5. **DEF-2 path,** per language: `filter_products` category misses, and of those turns, how many
   ended in a dispatched add of the right product. pt-BR's R2 "not found after the list" count is
   the number to watch.
6. **Carried from S7:** K-058/K-061 holds and recovery, the turn-2 tool mix, non-`$` currency,
   the name→price sweep with the F4 tag, and the first-lookup miss rate.
7. **D8 grounding residual** (review m1): C1 turns where a category-miss `catalog` was followed
   by an add of a product the turn's customer line didn't name.

**Instrument note.** Customer lines in the proxy now read `Customer: …`, not `LSP-…: …`. The
RCA's scratch `build.py` parses customer lines with `LSP-[^:]+: `. Any re-analysis must key
customer lines on the fixed label and attribute participants by `CONTEXT.threadId`, as `build.py`
already does. The committed `cart_outcome_analysis.py` and `refire_probe.py` don't parse the
prefix (grep).

**Priors.** These are my estimates, re-derived from RCA §5.4 with §4's figures. They are not
promises.

| Gate | S7 | Prior | Main risk |
|---|---|---|---|
| Dead turns | 258/1,125 | 0–6 (P ≈ 0.85) | §4's thin turn-4 cascade rows if a dead turn still slips through; a third caller on LM Studio |
| es M1 | 59 | 66–75 (≈ 0.9) | — (44/44 alive in S7) |
| en M1 | 70 | 67–74 (≈ 0.9) | queueing pushes en's ~3.5-call add turn toward the latency line, not M1 |
| pt-BR M1 | 25 | 48–72 (≈ 0.55–0.65) | R2 after a *filter* miss list is untested; R-A |
| M2 | 0 | 0–2 (≈ 0.85) | the category-miss list grounds the whole catalog (D8.3) |
| M3 | 13 | 0–3 (≈ 0.8) | a C2-like re-add on turns 3/4 |
| p95 ≤ 29 s | 12.0 (flattered) | pooled 12–22 s, en turn-2 15–26 s (≈ 0.8) | 2-worker queueing behind en's ~3.5-call add turns **and** pt-BR's D8 path (one call → about three) |
| Wrong language | 0 | 0–3 (≈ 0.85) | R-B: the probe's names carried a language suffix on every customer line; the pre-check catches it first |
| C2 es (≥ 3/24 escalates) | 6/24 | 1–6/24, about an even chance it still escalates | RCA §3.2 outcome (b) |

## 7. Documentation impact

| Document | Change | Who |
|---|---|---|
| `falkor-chat/docs/plans/salesperson-cart-product-id.md` | header gains `Extended by:` this note. That is the only edit; D12's corrections live here | `architect` (done with this note) |
| `falkor-chat/docs/QUERIES.md` §15.2 (the abstention sentence, ≈ "the service maps that to `{"items": [], "finding": "no matching products found"}`") | add that the **tool** answers an unknown `category` with D8's `hint` + `catalog`, decided by re-running this same query with `category` only and `LIMIT 1` (`Services.has_category`). The query is unchanged. This also fixes the sentence's attribution: the tool builds that shape, not the service | implementer (S5) |
| `tools.py` `FilterProductsTool` docstring; `services.py` `has_category` docstring; `executor.py` `_assemble_messages` docstring (the "speaker-named" sentence) and `WorkflowExecutor.__init__`; `tools.py` `PostMessageTool` comment; `guards.py` `_recent_turns` docstring | as in §5, rewritten in place, never appended | implementer (S2–S4) |
| `falkor-chat/docs/SERVER.md` §1.3 (storefront surface) | one sentence: the storefront deployment labels customer lines `Customer` in the model's transcript (`WorkflowExecutor(human_speaker_label=…)`, D9). **After `devops`' concurrent `SERVER.md` edit is committed**, never on top of an uncommitted one | implementer (S5), sequenced by `teco` |
| `falkor-chat/AGENTS.md` | **no change.** Nothing here is an always-load trap, and `devops` is editing the file | — |
| `falkor-chat/docs/test-plans/salesperson-cart-product-id2.md`, `falkor-chat/docs/test-reports/salesperson-cart-product-id2-report.md` | new successors for the §6.2 re-run | `qa-engineer` |
| `falkor-chat/docs/HISTORY.md` | a delivery entry: D8, D9 and the re-run result, citing the RCA and this note | `teco` at close |
| `falkor-chat/docs/BACKLOG.md` | K-078 rewritten in place to its post-re-run state. Open items for RCA O2, O6 and catalog compaction (D11, with its trigger), and DEF-5 under K-079 F3, unless `teco` already filed them from the RCA | `teco` |
| `server/falkorchat/proof_defs.py`, `scripts/*`, `salesperson/*` | **no change.** No def bump, no pin move, no env var | — |

## 8. Risks & open questions

**Risks.**
- **R-A: pt-BR does not pick from a list it got from `filter_products`** (original R2 on a new
  path). es picks 44/44 after a *lookup* miss, but pt-BR after a *filter* miss is untested, and
  the frozen prompt names only the lookup's list. D8's hint carries the pick instruction to
  compensate.
  - *Signal:* §6.2 measure 5.
  - *Fallback:* none without `v10` (a prompt sentence). If pt-BR lands at 30–59, it is a partial
    fix, and the `v10` question goes back to the stakeholder together with DEF-5.
- **R-B: removing the prefix removes a language cue.** The probe's names end in `-en`, `-es` or
  `-pt-BR`, so S7's every customer line repeated the language the `CONTEXT` block names. No code
  read it (§2). Real customers' names carry no such cue, so the re-run is the more representative
  measurement, but K-065's wrong-language rate could rise. The original §6.3 gate stays at
  ≤ 3/375 per language, and §6.2's pre-check and its stop rule catch a rise before C1.
- **R-C: the name still reaches the model** through `get_profile` and the model's own greetings
  (D10). This is measured by vector. A guessed display-name `productId` is still held by D3
  whatever its vector, so the cost is an iteration, not a wrong write.
- **R-D: thin margin on the cascade path** (§4: ≈ 8–10% on turn-4 re-add pairs). Under the cap
  it only arises after a dead turn, which the cap makes rare. It is measured by §6.2 measure 1,
  with D11's trigger.
- **R-E: the extra read on empty results.** `has_category` adds one `reference` read per
  category-filtered empty result. It has the same plan as the original call (anchored on the
  `Product.price` range index), over ~15 products. Negligible, and absent from every non-empty result.
- **R-F: D8 widens D3's grounding residual** to a category-miss turn (D8.3). C1's M2/M3 and
  §6.2 measure 7 measure it (C2 never takes the filter path), and test 35 pins it as intended.
- **Rollback:** revert either commit (S1+S2, or S3+S4). Both are code-only: no data, def or
  pin moves.

**Open questions.** None block implementation.
1. *Stakeholder, only if triggered:* if D10's trigger fires (≥ 3 display-name `productId`s via
   the profile or echo vectors), should the storefront stop seeding the profile name at join
   (`salesperson-ui` §4.10)? That changes a requirement, so it goes to `tico`. My recommendation
   today is **no**: the evidence says the prefix was the bait.
2. *Stakeholder, carried from the RCA:* C2 ≥ 3/24 in any language reopens R3's guard design
   (original §8).

## 9. Ready to implement

Implement the plan at `falkor-chat/docs/plans/salesperson-cart-product-id2.md`: S1+S2 in one
commit (DEF-2), S3+S4 in one commit (DEF-4), then the S5 docs, then the S6 re-run by
`qa-engineer` once `devops`' 2-worker cap is in.

- **DEF-2:** an empty `filter_products` result whose `category` the repository doesn't know
  (`Services.has_category`, the same `categoryNormalized` match, without the price bounds)
  returns `items: []` plus a filter-specific `hint` and D2's price-free `catalog`. Everything
  else stays byte-identical. It grounds through the existing `_GROUNDING_SOURCES` with no
  executor change.
- **DEF-4:** `WorkflowExecutor(human_speaker_label=…)`, set to `"Customer"` only when
  `config.STOREFRONT_ENABLED`, renders customer lines as `Customer: …`. Assistant lines and every
  non-storefront deployment stay byte-identical. There is no def, config or env-var change.
- **Budget:** worst 2-way demand is ≈ 7.4–7.5k of 8,192 on a dead-turn cascade, and ≈ 6.1k
  expected.
- **Re-run order:** warm-up, then the 12-trial turn 0–1 language pre-check (any wrong-language
  reply stops the run and returns to `architect`), then C1, C2, the confirm and browse variants.
- **Done** at the original §6.3 gates (es and pt-BR ≥ 60/75, en ≥ 65/75, M2/M3 ≤ 2/225, dead
  turns ≤ 1.2%, p95 ≤ 29 s, wrong language ≤ 3/375). C2 is reported, and ≥ 3/24 escalates.
