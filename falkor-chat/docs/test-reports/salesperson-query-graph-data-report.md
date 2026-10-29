# `query_graph_data` row identity and category-miss recovery — Stage C test report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-079 (F2), K-086

Plan: `docs/test-plans/salesperson-query-graph-data.md`. Spec: `docs/plans/salesperson-query-graph-data.md`
§8.2/§8.3, plus review `docs/reviews/salesperson-query-graph-data-impl.md` Pass 2 m1.

## Summary

- **When / what:** 2026-10-28, 21:06–21:16 (-03:00), live on LM Studio. Run 0 on `e8798351` (the
  ledger's baseline, a `git archive` export). Run 1, the replay and both storefront runs on `HEAD`
  `5af62b86`, which contains Stage A `6286de44`, the tests `d5ba5728` and Stage B `9ba0d788`.
- **F2 done-condition (no Layer 1 regression on ministral): PASS.** Run 0 and Run 1 are both
  catalog 21/21, knowledge_base 12/19 and not-found 0/6. `nlq_compare.py` finds 0 per-pair
  differences. The replay gives L4 0/40 and L5 0.
- **DEF-C done-condition: PASS on the plan's classification.** B1 0/24, B2 21/24, B3 2/24 (at the
  ≤ 2 limit) and B4 0 dead / 0 wrong-language. The **first querygen completion filtered
  `category = "Peripherals"` in 24 of 24 turns**, so D6 did the work. D7 never fired.
- **Two new findings, neither a gate:**
  - DEF-1: browse replies mistranslate product names, e.g. Gaming Mouse Pad XL becomes "Pantalla
    para juegos XL".
  - DEF-2 (B5): D6 maps the product-type word "mouse" onto `Peripherals` in 4 of 12 `browse-type`
    turns, and the reply then claims "4 modelos de mouse".
- **m1:** there was no `name = <category word>` completion in this run, so m1's path never arose.
  All 3 pt-BR browse misses carried `returns`. Details are in the m1 section below.
- CPG: considered, not relevant — `cpg_falkorchat` is stale (brief); the work is a live behavioural
  measurement, and callers were found by grep.

## Environment

- **LM Studio:** nothing was loaded at 21:06. `scripts/ensure_demo_models.sh` (with `LMS_CLI` set to
  the Windows `lms.exe`) loaded exactly `mistralai/ministral-3-3b` (ctx 8192, PARALLEL 4) and
  `text-embedding-qwen3-embedding-0.6b`. `lms ps` listed only those two. They were still the only two
  loaded after the runs. Nothing was unloaded.
- **`:8200` (pid 8293):** left untouched. Its log has **0 lines** between 21:05:30 and the end of the
  live window. The last activity before that was at 21:02:45.
- **Demo:** it was down, so I started it with `start_demo.sh` (storefront, `salesperson@v9`, 2 turn
  workers, `ws:demo`), pointed through an observe-only proxy (`:1235 → :1234`) via a scratch copy of
  `opencode.local.json`. The proxy logged 115 requests, all HTTP 200. After the runs I stopped my
  uvicorn (pid 50408), the proxy and an `lms log stream`. `falkordb-dev` was never touched.
- **Model overlay:** Run 0 and Run 1 used a scratch `models.json` whose only change is
  `defaults.step = lmstudio/mistralai/ministral-3-3b`. The demo used the shipped overlay, as
  `start_demo.sh` does.
- **`reference`:** never wiped. I ran neither the offline pytest nor `test_queries.sh`.
- **Harness:** `refire_probe.py` gained the `browse-type` `SCRIPTS` entry, the one plan §7 assigns to
  `qa-engineer`. The run used sha256 `e3855959…9b03`. A docstring line for the new entry was added
  after the run. `language_salience_probe.py` is `5949774a…0787`, identical to id2's.
- **Raw data** (session scratchpad, not durable):
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6bf0232-c1ae-4f33-8aa9-de25f4a9b94a/scratchpad/stageC/`.
  It holds `run0.log`, `run1.log`, `nlq-ministral-{before,after}.json`, `compare.txt`,
  `completion-diff.txt`, `replay.{log,json}`, `browse.jsonl`, `browse-type.jsonl`, `proxy.jsonl`,
  `*-joined.{txt,json}`, `lms-server.log` and `scripts/`.

## Results

| ID | Outcome | Evidence |
|---|---|---|
| TP-000 | pass | models as above; `:8200` idle; pre-run `verify_salesperson.sh demo` rc 0, `verify_catalog.sh` rc 0; `ws:nlq-eval` has 62 `Entity` nodes |
| TP-001 | pass, but the check is vacuous | `git -C <top> diff --quiet e8798351 6286de44^ -- falkor-chat/server` → rc 0. `6286de44^` **is** `e8798351`, so the range is empty. That is consistent with the ledger's note that Stage A started on top of the baseline |
| TP-002 | pass | the export exited rc 0. `falkorchat.__file__` under the runner's `sys.path` order = `…/scratchpad/stageC/before/falkor-chat/server/falkorchat/__init__.py`. The export's `config/models.json` is identical to the tree's |
| TP-003 | pass | both runs printed `step model: lmstudio/mistralai/ministral-3-3b`, rc 0. Run 0 ran 21:07:21–21:08:48; Run 1 ran 21:08:56–21:10:22 |
| TP-004 (L1) | **pass** | Run 1 catalog **21/21** (target 21/21); Run 0 also 21/21 |
| TP-005 (L2) | **pass** | KB Run 1 **12/19** (floor ≥ 12/19). It is equal to Run 0 pair by pair: the same 7 misses (`nlq-27, 34, 35, 36, 37, 38, 39`), and **0** KB completions or tool results changed |
| TP-006 (L3) | **pass** | not-found false answers, Run 1: **0/6** (`nlq-13/14/15/28/29/30` all have empty `items`) |
| TP-007 (L4/L5) | **pass** | replay over Run 1's 40 completions: 38 executed both ways, and 2 abstained before F2 runs (`nlq-35` parse, `nlq-36` validation, both misses in both runs). **L4: 0/40** (target 0/40). The reverse case, incorrect without F2 and correct with it, is also 0. **L5: 0** compiled-Cypher differences (expected 0). The F2 column matches Run 1's `layer1Correct` on all 38 executed pairs. No execution error, so no R7 shape appeared |
| TP-008 (L6) | **pass** | `nlq_compare.py` → `diffs (0)`, so **0** catalog correct→incorrect flips (target 0). Every catalog pair whose completion changed is listed below |
| TP-009 | pass | `start_demo.sh` up at 21:11:20. Startup log: `storefront preflight ok: … def=salesperson@v9 products=15`. `verify_salesperson.sh demo` rc 0 with the demo up, so there is no def drift (§4.5) |
| TP-010 (B1–B4) | **pass** | see the browse section |
| TP-011 (B6) | done | see the browse section |
| TP-012 (m1) | done | see the m1 section |
| TP-013 (B5) | done, ungated; finding DEF-2 | see the browse-type section |
| TP-014 (B7) | done | see the browse section. id2 has no browse-window turn p95 to compare against |
| TP-015 | pass | `git status --short tests/eval/` from `falkor-chat/server` printed nothing, both after Run 1 and at the end |
| TP-016 | pass | after the runs: `verify_salesperson.sh demo` **rc 0** ("2 defs in sync"); `verify_catalog.sh` **rc 0** ("15 products") |

### L6: catalog pairs whose completion changed between Run 0 and Run 1 (none flipped)

| Pair | Change | L1 Run 0 → Run 1 |
|---|---|---|
| `nlq-05` | whitespace only (the `filters` list was reflowed) | ✓ → ✓ |
| `nlq-12` | Run 0 had a duplicate `filters` key (category, then price; `merge_duplicate_keys` merges them). Run 1 has a single two-filter list and adds `returns ["p.name"]` and `order_by p.price ASC` | ✓ → ✓ |
| `nlq-16` | whitespace only (the match was written on one line) | ✓ → ✓ |

- **Tool result changed, completion did not:** `nlq-13/14/15` (not-found). Run 1 returns D7's
  recovery (`"finding": "no catalog category or product has that exact name"`, hint and
  `categories`), with `items: []`. Layer 1 is unaffected, as R3 predicted.
- **R1 did not occur.** `nlq-14`'s completion is unchanged, with no `category = "Audio"` mapping.
- **The rest of the watch list did not move:** `nlq-02/04`, `06/07/09/10/11`, `40`, `19` and `20`
  had byte-identical completions.

## §8.3 — `browse` re-measure (DEF-C)

- **Run:** `refire_probe.py run --languages pt-BR,es --schedule browse --repeat 12`, 21:11:33–21:13:06.
  It produced 24 turns, all with `post_status` 200, 0 timeouts and 0 errors.
- **Mechanism:** every turn's tool chain came from the proxy, joined to the probe by thread language
  and trial order. For es, the reply text matches the posted `post_message` text.
- **Classification:** every reply was read by hand.

| Class (plan §8.3) | pt-BR | es | Pooled |
|---|---|---|---|
| Lists the 4 Peripherals (translated names allowed), nothing else as a peripheral | 9 | 12 | **21** |
| Presents the whole catalog as peripherals | 0 | 0 | 0 |
| Ends without an answer ("Sim, temos alguns periféricos disponíveis!", names none) | 2 (t1, t4) | 0 | 2 |
| Says the store has none | 0 | 0 | **0** |
| *Outside the four classes:* lists 1 of the 4 ("Mousepad para Gaming XL") | 1 (t2) | 0 | 1 |

| # | Measure | Target | Observed | Verdict |
|---|---|---|---|---|
| B1 | says the store has no peripherals | 0/24 | **0/24** | pass |
| B2 | lists exactly the 4 Peripherals | ≥ 20/24 | **21/24** | pass (see the DEF-1 caveat) |
| B3 | whole catalog, or no answer | ≤ 2/24 | **2/24** (0 + 2) | pass, at the limit |
| B4 | dead turns / wrong-language replies | 0 / 0 | **0 / 0** (pt-BR replies in Portuguese, es in Spanish) | pass |
| B6 | first querygen `category = "Peripherals"` / D7 fired / recovery → `filter_products(category="Peripherals")` | reported | pt-BR **12/12 / 0 / 0**; es **12/12 / 0 / 0** | — |
| B7 | turn p95 by path | reported | first-call-success path, n = 24: p50 8,036 ms, **p95 8,061 ms**, max 10,074 ms (pt-BR p95 8,054, es p95 8,959). Recovery path: **n = 0** | — |

**Mechanism detail:**
- All 24 turns called `query_graph_data` once, and none called `filter_products`. The internal
  question varied: es always sent "¿Hay productos relacionados con periféricos en el catálogo?";
  pt-BR sent "Do you have any peripherals (periféricos) in your catalog?", "Do you have any
  peripherals?", "há produtos relacionados a periféricos no catálogo?" or "disponíveis periféricos".
- Every querygen prompt carried D6's `Product.category is exactly one of` block.
- Every completion filtered `category = "Peripherals"`, so the result was never empty and the D7
  recovery was never needed. In id2, the same script produced 0 such filters.

**Why the 3 pt-BR replies missed (querygen chose the shape):**
- t1 and t4 used `returns ["count(p.name)"]`, which returned 4, so the reply affirmed but named no
  product.
- t2 used `returns ["p.name"]` with `limit 1`, so the reply named one product.

None of the three is an exact-match dead end, and none is the D7 path.

**B7 comparability:**
- All turns took the first-call path, so B7 has no recovery-path figure.
- id2's report gives no browse-window turn p95, only KV-slot figures, and its raw data directory no
  longer exists. A like-for-like latency comparison is therefore **not possible**, and I don't claim
  one.
- `turn_ms` is quantized by the probe's ~2 s state poll; the observed values cluster at 6,0xx /
  8,0xx / 10,0xx ms. Percentiles are linear-interpolated.

## m1 — `returns` on pt-BR browse misses

**Tally:**
- 3 pt-BR browse misses (t1, t2, t4). **3/3 carried `returns`**: `["count(p.name)"]`, `["p.name"]`
  with `limit 1`, and `["count(p.name)"]`.
- **0/12** pt-BR querygen completions used a `name = <category word>` filter. All 12 used
  `category = "Peripherals"`.

**What this does and doesn't show:**
- The validation-abstention path m1 describes (`name = "periféricos"` with no `returns`, which stops
  before D7) **did not occur in this run**.
- That is because D6 removed the `name = "periféricos"` shape seen in id2. It is not evidence that
  the m1 path is rare when that shape does appear.
- On this sample, m1 needs no `architect` follow-up. It remains a latent gap only if D6's mapping
  regresses.

## §8.3 — `browse-type` (B5, reported only)

- **Run:** `refire_probe.py run --languages pt-BR --schedule browse-type --repeat 12`
  ("Vocês têm algum mouse?"), 21:13–21:14.
- **Result:** 12 turns, 0 dead.

| Path | Turns | Reply |
|---|---|---|
| `filter_products(category="mouse")` → D8 miss → `lookup_product_fact("Wireless Mouse Pro")` | 6 (t0, 2, 3, 6, 8, 11) | offers the Wireless Mouse Pro ("Mouse Sem Fio Pro", $29,99), the catalog's one mouse. Correct |
| `query_graph_data` → querygen `category = "Peripherals"`, `returns ["count(p)"]` → 4 | 4 (t1, 4, 7, 10) | **"Sim, temos 4 modelos de mouse"**: presents the 4 Peripherals, keyboard and webcam included, as mice |
| `filter_products(category="mouse")` miss → `query_graph_data` → querygen with no filter, `count(p.name)` → 15 | 2 (t5, 9) | "não temos nenhum item específico chamado de 'mouse'… No entanto, há um produto chamado Wireless Mouse Pro…" It opens with "none" but offers the right product |

**B5:**
- Replies presenting non-mouse Peripherals as mice: **4/12**.
- Replies saying there are none outright: **0/12**. Two replies (2/12) open with "no item called
  'mouse'" and then offer the mouse.
- So D6 does **not** map only category words: in 4 of the 6 turns that reached querygen it mapped
  "mouse" onto `Peripherals`. That is review m3's risk (plan R1) realised. See DEF-2.

## Defects

**DEF-1 — Minor — Browse replies render catalog product names with wrong translations.**
- **Repro:** run `start_demo.sh`, then
  `refire_probe.py run --languages pt-BR,es --schedule browse --repeat 12`.
- **Expected:** the 4 names identify the 4 products, translated or not.
- **Actual:** the tool returns the exact English names (`Gaming Mouse Pad XL`, `Wireless Mouse Pro`,
  …), and the assistant step renders them as follows:
  - es: **"Pantalla para juegos XL"** (a gaming *screen*) 10/12, "Mouse de juego XL" 2/12, and
    "Mouse de juego Wireless Pro" 10/12;
  - pt-BR: **"Mouse de Gaming XL"** (a gaming *mouse*) 7/12, "Mouse de Gaming Pad XL" 1,
    "Mouse de Gaming Padro XL" 1, and "Mousepad para Gaming XL" 1.

  So in most lists the mouse pad reads as a different product.
- **Evidence:** `browse-joined.txt` (the tool result next to the reply, per turn).
- **Attribution:** this is the reply layer (the `assistant` step on ministral), not
  `query_graph_data`. The result rows are exact. It is not caused by F2 or K-086, but this change is
  what made the lists appear at all.
- **Effect on B2:** B2 counts these replies as "lists the 4", per the plan's "possibly with translated
  names". Under a stricter reading (every name identifies its product), B2 would fail. Flagged for
  `teco` to backlog.

**DEF-2 — Major (by user impact), ungated per plan B5 — D6 maps a product-type word onto a category.**
- **Repro:** run `start_demo.sh`, then
  `refire_probe.py run --languages pt-BR --schedule browse-type --repeat 12`.
- **Expected:** the one mouse is offered, and the keyboard and webcam are not called mice.
- **Actual:** in 4/12 conversations the internal question "quantos produtos de mouse estão
  disponíveis?" got querygen `category = "Peripherals"`, `count(p)` → 4, and the customer was told
  "temos 4 modelos de mouse". The catalog has one mouse.
- **Evidence:** `browse-type-joined.txt` t1/t4/t7/t10.
- **Routing:** per plan §8.3/R1, this is a D6 wording question for `architect`. The plan's fallback
  lists "drop D6, keep D7". Note that this run also shows D6 carrying all of DEF-C's fix, with D7
  firing 0 times, so dropping D6 would move B2 onto the untested D7 path.

No regression was found against F2 or the DEF-C gates. The 7 KB misses are pre-existing and
identical in both runs, including the relationship-traversal and conflicting-facts shapes that
plan §8.1 calls structurally unanswerable.

## Coverage & gaps

- **Covered:** §8.2 in full (Run 0, Run 1, compare, replay, tracked-file check) and §8.3 in full
  (`browse` at n = 24, `browse-type` at n = 12, mechanism capture through the proxy), plus m1.
- **Not exercised live:**
  - **D7's recovery path:** 0 firings in 36 storefront turns. It is exercised offline and in Run 1's
    `nlq-13/14/15` results.
  - **B7's recovery-path latency.**
  - **m1's validation-abstention path.**
- **Small samples:** B3 is exactly at its limit (2/24), and B2's margin is 1 turn. Both are point
  estimates on n = 24.
- **Out of scope, as planned:** the §8.1 offline suite, C1/C2, the RC-1 en price-list incidence,
  an en control, and Layer 2.

## Feedback & recommendations

1. **DEF-2:** send it to `architect` as the D6 wording question the plan foresees (R1). The aim is
   to keep "a product-type word is not a category" in force without losing the 24/24 category
   mapping seen here.
2. **DEF-1:** backlog it as a reply-layer product-name fidelity issue. One possible mitigation,
   untested here: tell the assistant step to keep catalog product names verbatim.
3. **Probe testability:**
   - `refire_probe.py` records neither the tool chain nor the querygen completion, so B6 and m1
     needed an ad-hoc proxy (`scratchpad/stageC/scripts/proxy.py`).
   - A durable observe-only proxy, or recording the run's step trace in the probe, would make these
     measures repeatable.
   - Separately, the B7 baseline should be recorded per window in future reports. id2's browse p95
     was not recorded, and that made B7 incomparable.
4. **Run 0 guard:** when `<Stage-A-first-commit>^` equals the baseline sha, the guard is an
   empty-range no-op, as it was here. It is still correct, but a future plan could say so.

## DEF-2 counterfactual probe (2026-10-28)

- **Spec:** `docs/plans/salesperson-query-graph-data.md` §11.2, judged against §11.5's decision rule.
- **Run:** 21:23:58–21:24:31 on `HEAD` `da469e47`, from `falkor-chat/server`.
- **Script:** a one-off scratch script modelled on the Stage C replay. It is not committed.
- **Raw output:**
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6bf0232-c1ae-4f33-8aa9-de25f4a9b94a/scratchpad/stageC/def2-probe/`
  (`probe.py`, `probe.log`, `probe.json`, `prompt_dump.py`, `prompt-*.txt`, `f6-check.txt`,
  `verify.log`).

**Method.**
- For each question × prompt variant × 3 repeats, one `step` call on `lmstudio/mistralai/ministral-3-3b`
  (`temperature: 0`, via the Stage C scratch overlay).
- The system message is `_build_query_request_system_prompt(querygen.CATALOG_SCHEMA, categories=…)`;
  the user message is the question.
- Each completion was then:
  1. parsed exactly as `QueryGraphDataTool.run` does (`extract_own_line_json_object` +
     `merge_duplicate_keys` → `default_returns` → `model_validate`);
  2. passed through `ensure_row_identity` and `compile`;
  3. executed read-only on `reference` through `Services.run_structured_query`.
- `categories` = the live `Services.catalog_categories` list: `Accessories, Audio, Cameras,
  Displays, Peripherals, Smart Home, Storage, Wearables`.

**F6 check (§11.1): verified.**
- I built the prompt twice: from the `e8798351` export, whose builder has no `categories`
  parameter, and from `HEAD` with `categories=None`.
- The two are **byte-identical**: 2,503 characters each, the same sha256
  `b49d612c…6ade6`, and `cmp` returned 0.
- So `categories=None` *is* the pre-change prompt.
- Separately, the `categories=<the 8>` prompt is byte-equal to the one querygen system prompt
  recorded in Stage C's proxy log. The probe's D6 variant is therefore the prompt the storefront
  actually sent.

**What querygen actually received.**
- In all 12 Stage C `browse-type` turns, the customer's text "Vocês têm algum mouse?" never reached
  querygen.
- The agent step either called `filter_products`/`lookup_product_fact`, or rephrased the text
  before calling `query_graph_data`:
  - "quantos produtos de mouse estão disponíveis?" (A, 4 turns);
  - "tem algum mouse no catálogo?" (B, 2 turns).
- A and B are taken verbatim from `browse-type-joined.txt`. The customer text was run as a
  supplementary input (6 extra calls, 18 in all), labelled as such.

**Results (18 calls).**

| Question | Variant | Repeats | Completion shape | Tool outcome / rows |
|---|---|---|---|---|
| A "quantos produtos de mouse estão disponíveis?" | `None` (pre-change) | r1–r3 identical | `name LIKE "%mouse%"`, `returns ["count(p)"]` | **validation abstention** (`op` must be one of `= <> < <= > >=`) → `{"items": [], "finding": "no matching data found"}` |
| A | `cats` (D6) | r1–r3 identical | `category = "Peripherals"`, `returns ["count(p)"]` | `[{"count(p)": 4}]`; byte-equal to the live Stage C completion |
| B "tem algum mouse no catálogo?" | `None` | r1–r3 identical | `name = "mouse"`, `returns ["count(p.name)"]` | `[{"count(p.name)": 0}]` |
| B | `cats` | r1–r3 identical | `filters: []`, `returns ["count(p.name)"]`, `limit 1` | `[{"count(p.name)": 15}]`; byte-equal to the live Stage C completion |
| *Supplementary:* "Vocês têm algum mouse?" | `None` | r1–r3 identical | `name = "mouse"`, `returns ["count(p.name)"]` | `[{"count(p.name)": 0}]` |
| *Supplementary* | `cats` | r1–r3 identical | `filters: []`, `returns ["count(p.name)"]`, `limit 1` | `[{"count(p.name)": 15}]` |

**Determinism:** in all 6 question × variant cells, the 3 repeats gave exactly 1 distinct
completion. Both D6-variant completions also match the live Stage C ones byte for byte.

**Decision-rule outcome (§11.5): the pre-change prompt gives "none" or 0, never product names.**
- **Question A, the DEF-2 turns:** the pre-change completion abstains, because the model reaches for
  a `LIKE` the DSL doesn't have (§11.1 F4, observed). The tool returns "no matching data found".
- **Question B:** the pre-change completion returns `count = 0`. That one row bypasses D7 and every
  abstention path (§11.1 F5, observed).
- **The supplementary customer text:** also 0.
- No pre-change cell returned names a reply could use. So **D6 did not regress this case**:
  - for A it replaced "none" with "4";
  - for B it replaced "0" with "15".

  Under §11.5 this is the expected branch: close K-086 on its gates and backlog DEF-2.

**Limits.**
- This probes the querygen step only. It does not show what the reply step would say given these
  rows. "No matching data found" and `count = 0` are *likely* to read as "não temos mouse", but no
  reply was generated.
- The agent step's choice of internal phrasing, and its routing mix, are not re-measured.
- During the probe, `qwen/qwen3-4b-2507` was also resident, idle with a 1 h TTL. It was not loaded
  by this session, and it was loaded after Stage C ended, so it was left alone. The probe calls only
  ministral, at `temperature: 0`, and both D6-variant completions reproduced Stage C's byte for
  byte.
- `:8200` logged 0 lines during the probe window. Both verify scripts exited 0 afterwards:
  `verify_salesperson.sh demo` rc 0, `verify_catalog.sh` rc 0.
