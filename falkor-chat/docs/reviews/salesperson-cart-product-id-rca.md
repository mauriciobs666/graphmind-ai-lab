# RCA — K-078 S7 live-check failures (KV overflow, pt-BR filter miss, es re-fire)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-078

CPG: considered, not relevant — the defects are runtime-capacity and model-behaviour failures, and
the causal links run through recorded model traffic and LM Studio's slot log, not through call
graphs; the few code seams involved (`Executor._assemble_messages`, `FilterProductsTool.run`,
`LookupProductFactTool.run`, `storefront` join) were read directly. `cpg_falkorchat` is also stale
per the plan.

## Scope

**Diagnosed:** DEF-1 to DEF-6 from `falkor-chat/docs/test-reports/salesperson-cart-product-id-report.md`
(S7, 2026-10-23), for `salesperson@v9` (`4dac501e` + `8a72f08d`). The baseline is U3 Q3 (`v8`,
`falkor-chat/docs/test-reports/salesperson-single-model-report.md`).

**Read:**
- the plan (`falkor-chat/docs/plans/salesperson-cart-product-id.md` v3: D2, D3, D7, R2–R4);
- the K-066 fit report (`salesperson-single-model-fit-report.md`);
- the shipped code: `executor.py` `_assemble_messages`, `tools.py`, `retry.py`, `storefront.py`
  join, `repository.py` `find_waiting_run_for_thread`.

**Raw data, all read-only** (session scratchpad, not durable), under
`…/scratchpad/k078-s7/` and `…/scratchpad/u3/`:
- `proxy.jsonl`: 3,584 chat calls, with request and response bodies and `usage`.
- `lms-runtime.log`: the llama.cpp slot log. Present for both runs; it starts at 14:25:18 for S7.
- `lms-server.log` / `u3/lms-server-q3.log`: LM Studio logs each response's `usage`, so the baseline
  has per-call token counts even though it had no proxy.
- The probe JSONLs.

**Not done:** no live calls, no load or unload, and no test runs (none were needed: every claim
below rests on recorded traffic). My analysis scripts are in `…/scratchpad/rca/` (`build.py`,
`common.py`, `groups.py`, `slots.py`). I did not recount the dead turns, filled carts or qty > 1,
per the brief.

**Confidence labels:**
- **confirmed:** computed from recorded traffic or logs, or traced end to end;
- **inferred:** reasoned from evidence, not directly observed.

## 1. Symptom & impact

| Defect | What breaks | Gate effect |
|---|---|---|
| DEF-1 | 22.9% of C1 turns die. Every in-flight LLM call fails at once with HTTP 400 `Context size has been exceeded`: 89 failure groups, each **exactly 3 calls, one per language** | Dead turns 258/1,125. Drags es M1 to 59/75. Causes all 13 M3 doubles |
| DEF-2 | pt-BR add turns call `filter_products(category="Mouse sem fio…")`, get `{"items": []}` and reply "not found" (35/75 add turns) | pt-BR M1 25/75 |
| DEF-4 → DEF-3 | The model passes the customer's display name as `productId`. In es C2 that held call opens a path to a second mouse add | C2 es re-fire 6/24 (escalation) |
| DEF-5 | pt-BR quotes `R$` (7 replies) | reported, not gated |
| DEF-6 | 3 turn-0 dead turns from a FalkorDB `Query timed out` at bring-up | 3 of the 258 |

## 2. Reproduction & evidence

There was no live reproduction (constraint). Everything below comes from recorded traffic.

1. **Per-call dataset.** `proxy.jsonl` → one row per chat call, with:
   - start time and duration (the proxy's `at` is the request start, `tools/lms_proxy.py:19`);
   - status, and `prompt_tokens` / `completion_tokens`;
   - thread, language and turn (from the `CONTEXT` block and the customer lines);
   - the current turn's tool calls and results.
2. **Constant prompt delta.** The same first call (`Hi there!`, 2 messages) costs:
   - **1,960 tokens in v8** (`u3/lms-server-q3.log`, mode 1,960–1,962);
   - **2,187 tokens in v9** (mode 2,185–2,189, n = 222);
   - so **+227 tokens on every agent call**.

   Source split, from the trees at `4dac501e^` and `HEAD`:
   - `systemPrompt`: 2,797 → 3,481 chars (+684);
   - `add_to_cart`/`remove_from_cart`/`lookup_product_fact` schemas: +110/+123/−7 chars.

   That is roughly 170 tokens of prompt and 57 of schema. The internal `query_graph_data` call is
   ~691 tokens in both runs: unchanged.
3. **Per-tool-result token cost.** For each consecutive pair of calls in one turn that differ by
   exactly one tool result, I took the difference in `prompt_tokens` minus the previous call's
   completion tokens:

   | Result kind | n | tokens p50 / p95 |
   |---|---|---|
   | `lookup_product_fact` miss + `catalog` (D2) | 121 | **397 / 397** |
   | `add_to_cart` held (D3 reason) | 141 | 103 / 111 |
   | `query_graph_data` | 222 | 181 / 181 |
   | `lookup_product_fact` found | 289 | 44 / 51 |
   | `add_to_cart` ok | 315 | 42 / 49 |
   | `filter_products` empty | 37 | 16 / 16 |

4. **In-flight KV occupancy, both runs** (`slots.py` over `lms-runtime*.log`).
   - **Method:** at each `launch_slot_`, sum the final `n_tokens` (from `release`) of every slot
     then processing. This is an upper bound on what is resident at that instant.

   | | Baseline Q3 (`v8`) | S7 (`v9`) |
   |---|---|---|
   | Max concurrently active slots | **3** | **3** |
   | 3-way in-flight sum p95 / max | 7,409 / **7,947** | 8,051 / 8,863 |
   | Launch instants over 8,192 | 0 | 85 |
   | Failure groups (in the log window) | 0 | 77, all of size 3; KV at failure 8,194–8,863, median 8,269 |

   - **Where the overflows trigger:** 72 of 77 are triggered while a slot ingests a prompt chunk
     (`failed to find a memory slot for batch of size 112`, …). Only 5 happen during pure decode.
   - **Both runs used the same runtime** (`llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.62.1`), so a
     runtime change is ruled out.
5. **Display-name `productId`s.** I scanned every `add_to_cart`/`remove_from_cart` argument in the
   responses.
6. **The `v8` vs `v9` tool mix** comes from LM Studio's `Model generated tool calls:` lines.

## 3. Causal chain

### 3.1 DEF-1 — KV-cache overflow

**The pool rule (confirmed).** ministral is loaded at `-c 8192`, PARALLEL 4, `kv_unified = 'true'`.
That makes one 8,192-cell pool shared by all slots. When the active sequences outgrow it, llama.cpp
fails **every** task in the batch:
- in each of the 77 logged groups, `send_error` fires for all three slots back to back;
- the fit report saw the same with 3 × ~3.1k prompts.

So a dead turn needs the *summed* length of the in-flight sequences to exceed 8,192. Any one prompt
is far below that (max 3,562).

**Chain:**
1. **The probe makes 3 sequences coincide by design.** One trial is three conversations (en, pt-BR,
   es) posted in lockstep. `TURN_WORKERS` 4 > 3 participants, so the storefront never queues, and all
   three run each turn's tool loop at the same time.
   - In both runs, at most 3 slots were ever active (§2 item 4). **The in-flight count did not grow.**
2. **The baseline already sat at 97% of the pool.** v8's worst 3-way instant was 7,947 of 8,192, a
   margin of **245 tokens**.
   - Nobody had measured that margin. U3 reported "0 overflows", and the fit report's "two is the
     safe ceiling at this prompt size" was never turned into a headroom number.
3. **v9 made every in-flight sequence longer:**
   - **+227 tokens on every agent call** (system prompt +~170, schemas +~57), every turn, every
     language;
   - **+397 tokens per `lookup_product_fact` miss** that stays in the turn's tail. es misses on
     72/75 add turns and pt-BR on 40/40 lookups. 149 of the 267 failed calls carried at least one
     catalog list. In 62 of 86 failure groups, two of the three sequences carried one;
   - **+~130 tokens per held add** (the held call plus the 103-token reason), and one more iteration.
     69 failed calls carried a held result.
4. **The constant delta alone was enough.**
   - Replaying the baseline's own 1,372 three-way instants with each sequence +227 tokens longer puts
     **75** of them over 8,192. S7 observed 85.
   - The sensitivity is steep: +100 per sequence → 4 instants over; +150 → 14; +300 → 245.
5. **Ruled out as contributors:**
   - **Thread history** shrank. Median reply length fell at every turn, e.g. turn 2: 248 → 143
     chars, partly because dead turns post nothing.
   - **More in-flight sequences:** ruled out, see step 1.
   - **A runtime change:** same `2.62.1` in both runs.
   - **`max_tokens`:** completions are 6–80 tokens, and overflows are triggered at prompt
     ingestion (72/77).
   - **Idle-slot cache:** 7 `purging slot` events across the whole run. Idle slots are evicted, not
     counted.

**Answer to "is the premise true?"** Yes in direction, wrong in shape.
- What failed is **the 3-way synchronized sum against a pool with 3% headroom**, not "longer prompts"
  in general. Per-call p50 rose only ~9% (2,243 → 2,433).
- The plan's R4 risk statement was wrong in three ways:
  - it counted only the miss list, not the +227 on every call;
  - it costed the list at ~250 tokens; it is 397;
  - it called the list "only on a miss", but a miss is the norm for es and pt-BR.

**Why turns 2–4.** The catalog and held tails first appear on turn 2. Then turns 3 and 4 re-add
after a dead turn 2, which re-enters the same tails. Calls carrying a catalog list on turns 3 and 4: es 43 and 47, pt-BR 28
and 38. en has none on any turn. That cascade is also the M3 path (report: every re-add follows a dead turn).

**Confidence: confirmed** (pool rule, margins, components, counterfactual replay). The
counterfactual is first-order: it assumes a removed component would not change the dynamics.

### 3.2 DEF-4 → DEF-3 — display name as `productId`, es re-fire

**Where the string enters (confirmed).** There are two seams:
- **The speaker prefix.** `Executor._assemble_messages` (`executor.py:1381`) renders every thread
  message as `f"{speaker}: {text}"` with `speaker = displayName`. Every customer line in every call
  therefore carries e.g. `LSP-358e50-t001-es: …`. The `CONTEXT` block carries only `language` and
  `threadId`.
- **The profile.** The storefront join seeds `save_profile(name=display_name)` (`storefront.py`
  ~769), so `get_profile` also returns `{"name": "LSP-…"}`.

**Which seam is active.** All **52** display-name `productId`s in the run came from the prefix:
- the split is C2 en 5 and es 14, then confirm-run en 10, es 12 and pt-BR 11;
- C1 had none;
- **none** had a `get_profile` result in the same turn.

**Why that string.** The probe's names (`language_salience_probe.py:383`,
`LSP-{run_id[:6]}-t{trial:03d}-{language}`) are slug-shaped, exactly like a `productId`
(`wireless-mouse-pro`).
- The v9 schema and prompt tell the model to pass an "identifier copied exactly".
- When the model wants to add something but has no ID surfaced this turn, the only ID-shaped token
  in its context is the speaker label.

The confirm run shows it most plainly: on "Yes, add it to my cart." (no product named), 33 of 35
held values are the display name.

**The six es re-fires (confirmed, all six).** Conversations `d19970`, `5f3ac6`, `45632a`, `92681f`,
`ceafd4` and `b556c3` (`distinct-immediate`) all follow the same sequence:
1. lookup the SSD, then add the SSD;
2. `add_to_cart(productId="LSP-…-es", quantity=1|2)`, which is **held**;
3. `lookup_product_fact("Wireless Mouse Pro")` (one case: `"Mouse Inalámbrico Pro"`, a miss with
   the catalog);
4. `add_to_cart("wireless-mouse-pro")`, which lands as quantity 2.

Example `5f3ac6`, verbatim, is in the Appendix.
- **Seven more es conversations** opened with the same display-name hold and did not re-fire: 4
  `same-category-immediate` and 3 `distinct-immediate`. In 5 the model looked the mouse up and then
  stopped; in 2 it re-issued the earlier add and got a K-061 dedup hold.
- **en and pt-BR** never issued a second add in C2 turn 2.

**What the display-name call is.** It is the *vehicle* of the re-add, not its *cause*.
- **The cause is upstream:** the es model's intent to (re)add the turn-1 mouse when the customer
  says "Also add …". The reply in `5f3ac6` says the mouse "ya estaba en tu carrito con una unidad
  adicional".
- **The name is just the ID it fills in,** because the mouse's ID was surfaced in turn 1 and tool
  results are not replayed (plan §2).
- **The hold reason is then unevaluable.** "If they did ask for this one, look it up first" refers
  to `LSP-…`, which is not a product. The model resolves "this one" to the mouse from history and
  follows the lookup branch.

**Does removing the trigger remove the path? No, only this opener. Confidence: inferred, medium.**
- **The path is independent of the held value.** hold → conditional reason → lookup → add is the
  same path en takes from guessed slugs (`Wireless_Mouse_Pro`, …, 38/38 recoveries in C1) and every
  language takes from confirm-turn holds. It is R3's accepted one-decision depth.
- **Without the bait, the re-add intent has three possible outcomes:**
  - (a) a guessed or translated slug, which is held and takes the same path, so no change;
  - (b) lookup first, as es does on 74/75 add turns, giving a **grounded add with no hold at all**,
    which is worse;
  - (c) no attempt, which is better.
- The data cannot separate these. In 13/13 es cases the bait call came *before* any mouse lookup,
  which hints at an opportunistic add (favouring c), but that is not proof.

The re-measure is the test, and it must log the first mouse-directed call in turn 2 so the outcome
can be classified as (a), (b) or (c).

### 3.3 DEF-2 — pt-BR `filter_products(category=<product name>)`

**v9 introduced this route (confirmed by counts; the causal link is inferred).**
- **v8 (Q3):** `filter_products` was called 6 times, all with `category=""` (whole catalog). It was
  never called with a product name.
- **v9 (C1):** 34 such calls are logged by LM Studio, and 35 pt-BR add turns (proxy) open with
  `filter_products(category="Mouse sem fio Pro" | "Mouse sem fio", [minPrice 0–1, maxPrice
  100|999])`.
- **Turn-2 first calls in pt-BR:** `filter_products` 35, `lookup_product_fact` 32, `add_to_cart` 8.
- **Likely invitation:** v9 names `filter_products` as an ID source for adds, twice:
  - the prompt: "copied exactly from a `lookup_product_fact` or `filter_products` result";
  - the `add_to_cart` schema description (D1).

**The dead end (confirmed).**
- The `category` argument is an exact match. The catalog has two categories (`Peripherals`,
  `Storage`), and the schema doesn't name them ("Only products in this category.").
- The empty result is `{"items": [], "finding": "no matching products found"}`: 16 tokens, with no
  list and no hint (`tools.py` `FilterProductsTool.run`).
- All 35 alive filter-path turns end right there, with a text reply of "não encontrei". The turn has
  exactly one tool call.
- Plan D2 put recovery only on `lookup_product_fact`. R2 anticipated the model giving up after a
  miss, but not a miss on a different tool.

**Fix options and their token cost** (each must fit the DEF-1 budget, §5):

| Option | Tokens added on a DEF-2 turn | Iterations | Def bump | Side effects |
|---|---|---|---|---|
| **(a) Catalog on a category miss.** When `category` is given and matches no catalog category, return D2's `hint` + `catalog` | +397 (the same list as a lookup miss) | +1 (pick, then add) | **no** (tool result is code) | Grounds every product for the rest of the turn (D3/R3 hole, like a lookup miss), so measure M2 again. An empty result from a price range only stays as it is, so "nothing under $5" doesn't get a 397-token list |
| (b) Hint only (name the categories; "for one named product use `lookup_product_fact`") | +~45, then the lookup miss +397 | +2 | no | More iterations and more tokens than (a), because pt-BR still translates the name and misses the lookup (40/40) |
| (c) v10 prompt: steer named products to `lookup_product_fact` | +~25 on **every** call | 0 | **yes, `v10`** | Spends budget on every call to fix one path. A 3B adherence bet. Re-pins and re-gates |
| (d) Enumerate categories in the `category` schema | +~15 on every call | 0 | no | Couples a code schema to catalog data. Does not stop a translated *product* name being sent as a category |

**Recommendation: (a), scoped to an unknown category.**
- It mirrors D2, which already works for es (44/44 alive).
- It costs the same as the lookup path pt-BR already takes.
- It needs no `v10`.
- It is affordable only once DEF-1's concurrency fix is in. Under 3-way concurrency it would make
  pt-BR add turns as long as es's (~2.85–2.95k), i.e. three long sequences at once.

**Confidence:**
- the cause of the dead end: confirmed;
- that v9's wording invites the route: inferred, high (0 vs 35);
- that (a) recovers: inferred. es shows the model picks from the list, but pt-BR after a *filter*
  miss is untested (R2).

### 3.4 DEF-5 and DEF-6

**DEF-5 (`R$`).** 7 pt-BR replies write `R$ 29,99`.
- The amount is right but the unit is wrong: it states the price in reais.
- v9's one-sentence F3 rule stopped `€` but not the pt-BR locale habit.
- **Action: none for K-078.** Fold a sharper sentence into `v10` only if a `v10` is cut for another
  reason. Otherwise, keep it as an open K-079 F3 backlog item (`architect` when it is picked up).
  Cutting `v10` for this alone means re-pins and a re-gate for a non-gated measure.

**DEF-6 (FalkorDB `Query timed out`).**
- **What failed:** the three trial-0 turn-0 calls to `Repository.find_waiting_run_for_thread`
  (`repository.py:3093`) timed out together at 14:20:56, seconds after uvicorn started.
- **Why it's minor:** the query is index-anchored on `WorkflowRun.status`, and it never recurred.
  It is a cold start, not a query-plan problem (inferred; `ws:demo` holds 1,254 runs).
- **Cost:** 3 of the 1.2% dead-turn budget (13.5 turns).
- **Action: none in code.**
  - For the re-run, `qa-engineer` sends one throwaway warm-up turn (or one read-only call) before
    trial 0, and reports it apart.
  - If it recurs outside bring-up, send it to `graph-dba` (TIMEOUT posture, `QUERIES.md` §9.1).

## 4. Root cause

**DEF-1 (confirmed).**
- **Root cause:** the demo's capacity model allows three LLM sequences in flight
  (`TURN_WORKERS` 4 ≥ 3 customers) on a single 8,192-token unified KV pool. At v8's sizes that pool
  already had only ~3% headroom at the 3-way peak.
- **Trigger:** v9's longer sequences:
  - +227 tokens per call, everywhere;
  - +397 per miss list, carried in most es/pt-BR turns;
  - +~130 per held add.
- **Contributing factors:**
  - the plan's R4 counted only the miss list, at ~250 tokens, as "only on a miss";
  - no gate or report measured KV headroom: U3 reported 0 overflows, not the 245-token margin;
  - the TP-007 probe synchronizes three conversations, which is the worst case for a shared pool;
  - one overflow fails every in-flight call, and agent calls have no retry (`retry.py` covers
    background calls only).

**DEF-4 (confirmed).**
- **Root cause:** an ID-shaped string sits in every customer line, via the `displayName` speaker
  prefix.
- **Trigger:** the probe's slug-shaped names, together with v9's "copy an identifier" instruction.
- **Contributing factor:** IDs surfaced in earlier turns are not replayed. A model that wants an ID
  it doesn't have reaches for the nearest look-alike.

**DEF-3 (inferred).**
- **Root cause:** R3 as accepted. The guard is one decision deep, and es on this model ignores
  "only what the customer asked for in their current message" on half its `distinct-immediate`
  turns.
- **Trigger:** the DEF-4 held call. Its conditional reason points at a non-product, so the model
  resolves it to the mouse from history.

**DEF-2 (confirmed mechanism, inferred invitation).**
- **Root cause:** a recovery asymmetry. A failed `lookup_product_fact` lists the catalog, but a
  failed `filter_products` gives nothing back.
- **Trigger:** v9 names `filter_products` as an ID source for adds, and pt-BR starts using it with
  a translated product name as the category.

## 5. Suggested fix & prevention

### 5.1 DEF-1 fix options compared

"Demand" is the summed length of the in-flight sequences at the binding concurrency. S7's measured
3-way maximum is 8,863 (runtime log). The 2-way maximum, estimated from proxy prompts, is ~6.4k.

| Option | Effect on headroom | Cost | Verdict |
|---|---|---|---|
| **O1 `FALKORCHAT_STOREFRONT_TURN_WORKERS=2`** for the demo (`start_demo.sh`, overridable) | The binding concurrency becomes 2-way: ~6.4k of 8,192, **~1.8k headroom (22%)**. That is enough to absorb DEF-2's +397 | A third concurrent customer queues behind the first finished turn. U3 Q4 (v8, 2 workers) measured p95 14.1 s, the same as at 4 workers. Replaying S7's alive groups with a crude queue gives an upper bound of ~20 s pooled p95. No model reload and no VRAM. Does not cover `:8200` traffic | **Recommended (first)** |
| O2 LM Studio `--parallel 2` | Same cap, enforced globally (`:8200` too) | Needs an unload and reload of ministral (operator call). `ensure_demo_models` checks only context, so it would need a parallel check too | Optional hardening after O1 |
| O3 a larger context (`-c 12288`) | The 3-way pool becomes 12,288: S7's worst instant fits with ~3.4k to spare | **VRAM ≈ +416 MiB per +4,096 tokens** (inferred: ~104 KiB/token for 26 layers × 8 KV heads × 128 × K+V × f16; check with `lms load … --estimate-only`). ministral alone already uses 5,353 of 6,141 MiB, plus WDDM paging when the embedder runs. `ensure_demo_models.py` hard-codes `DEMO_CONTEXT_LENGTH = 8192`, so it needs script and doc changes and a live fit check | Only if the stakeholder requires true 3-way parallelism |
| O4 trim each call | Without a def bump: catalog as `[id, name]` pairs ≈ −100 per miss (IDs only ≈ −225); a storefront speaker label (the DEF-4 fix) ≈ −8 per customer line. With `v10`: removing prompt/schema duplication ≈ −100–170 per call | Getting back just to v8's **3%** margin needs ≈ −227 per sequence, plus whatever DEF-2 adds | **Insufficient alone.** Worth it as a latency/headroom bonus |
| O5 per-call `max_tokens` | None. Completions are 6–80 tokens, overflows trigger at prompt ingestion (72/77), and llama.cpp doesn't reserve `max_tokens` | — | Rejected |
| O6 executor retry on `Context size has been exceeded`, with jitter | Turns a collision into latency instead of a dead turn | The error is ambiguous (a genuinely oversized prompt gets the same 400), so attempts must be bounded. Unjittered retries of the three victims collide again. It is a code change in the agent loop | Optional, backlog (`architect`). Not needed for the re-run |

### 5.2 How the fixes share the token budget

With O1 in place, 2-way demand is the binding quantity:

| Change | Δ per sequence | 2-way worst (≈6.4k) | 3-way worst (8,863 measured) |
|---|---|---|---|
| as-is (S7) | — | ~6.4k | 8,863 (overflows) |
| + DEF-2 (a), on pt-BR filter turns only | +397 | ~6.8k (fits) | ~9.2k (overflows) |
| + DEF-4 storefront speaker label | −~8 per customer line (≤ −40 on turn 4) | ~6.75k | — |
| + optional catalog `[id, name]` pairs | −~100 per miss | ~6.65k | — |
| a future `v10` (e.g. DEF-5) | ± its prompt delta, on every call | must stay under ~8.0k | — |

**The ordering constraint:** DEF-2 (a) *spends* budget, so it must land after O1 (or O3).
**Prevention for every future def bump:** report the run's peak 2-way and 3-way in-flight KV
(`slots.py`'s method) next to the dead-turn rate, and fail review on a margin under ~10%.

### 5.3 Recommended fix set, in order

1. **`devops`: O1.**
   - `start_demo.sh` exports `FALKORCHAT_STOREFRONT_TURN_WORKERS=${…:-2}`, with a comment citing this
     RCA.
   - Update the script header and the `falkor-chat/docs/HISTORY.md` entry.
   - No model change. It doesn't block steps 2–3, but it must be in before step 4.
2. **`architect`: one short design note**, with no `v10`. The plan was executed against, so this is
   a successor carrying `Extends:` (slug by `teco`). It covers:
   - (i) **DEF-2 option (a):** the exact trigger is "`category` given, matching no catalog
     category". Results emptied only by a price range stay unchanged. Reuse D2's `hint` and
     `catalog`.
   - (ii) **DEF-4:** customer turns carry no `displayName` in the speaker label for the storefront
     deployment, e.g. a fixed `Customer` label. The switch must **not** live in step `config`:
     `config` is create-only, so that would force `v10`. Multi-human threads keep named speakers.
   - (iii) the §5.2 budget table.
   - (iv) whether to take the optional catalog compaction now. My recommendation is to defer it:
     it touches D2's K-079 rationale, and O1 makes it unnecessary.
3. **`tdd-engineer`: implement (2), with repro tests first:**
   - `filter_products` with an unknown category returns `hint` + `catalog`, and grounds those IDs
     for an add in the same turn;
   - an empty result from a price range alone is unchanged;
   - storefront-assembled messages contain no `displayName` substring, while the multi-human path
     is byte-identical. Run targeted tests only, per the storefront/`wf_repo` hazards.
4. **`qa-engineer`: the re-run**, with the proxy kept:
   - C1 n = 75, C2 for all three languages, and the confirm variant;
   - one warm-up turn first (DEF-6);
   - new reported measures: peak 2-way/3-way in-flight KV from `lms-runtime.log`; for C2 es turn 2,
     the first mouse-directed call classified (a)/(b)/(c) (§3.2); latency reported next to the
     dead-turn rate.
5. **The stakeholder: C2 es.** At ≥ 3/24 after the trigger fix, R3's guard candidates (plan §8)
   come back as a decision.
6. **Deferred:** DEF-5 (into any later `v10`); O2, O6 and the catalog compaction (backlog).

### 5.4 Re-run priors (my estimates, not promises)

| Gate | S7 | Prior after 1–3 | Main risk |
|---|---|---|---|
| Dead turns ≤ 1.2% | 258/1,125 | 0–4 (P(pass) ≈ 0.9) | `:8200` or another caller adding a third sequence; a DEF-6-style cold start without the warm-up |
| es M1 ≥ 60 | 59 | 68–75 (≈ 0.9) | The speaker-label change perturbs es's lookup-first habit (unlikely) |
| en M1 ≥ 65 | 70 | 68–74 (≈ 0.9) | — |
| pt-BR M1 ≥ 60 | 25 | 50–72 (≈ 0.55–0.65) | R2: pt-BR may report "not found" after a filter-miss list. Without fix (a), the prior is ~40, a sure fail |
| M2 ≤ 2/225 | 0 | 0–2 (≈ 0.9) | The filter-miss list grounds the whole catalog |
| M3 ≤ 2/225 | 13 | 0–3 (≈ 0.8) | All 13 followed dead turns. The residual is a C2-like re-add in turns 3/4 |
| Turn p95 ≤ 29 s | 12.0 s (flattered) | pooled 12–20 s, en turn-2 15–25 s (≈ 0.85) | Queueing at 2 workers on en's ~3.5-call add turns |
| Wrong language | 0 | 0 | — |
| C2 es re-fire (escalation ≥ 3/24) | 6/24 | 1–6/24. **About an even chance it still escalates** | §3.2 outcome (b): an un-held, lookup-first re-add |
| C2 en / pt-BR | 0 / 0 | 0–1 | — |

## 6. Checks on the report's own numbers

| Report claim | Check | Result |
|---|---|---|
| 255 failed C1 calls, all agent, none internal | proxy: 255 C1 + 12 C2c 400s, 0 internal | **holds** |
| "one overflow fails every sequence in flight" | 89 groups, each exactly 3, one per language; runtime `send_error` ×3 per group | **holds** |
| Miss list "~250 tokens" (DEF-1 cause; plan D2/R4) | Δ`prompt_tokens`, n = 121 | **wrong: 397** |
| DEF-1 "v9 makes each request bigger" | Per-call p50 +9%. The binding quantity is the 3-way sum against a 245-token margin | **true but incomplete** (§3.1) |
| "Deaths on turns 2–4, where the history is longest" | v9 history is *shorter* than v8's. Turns 2–4 are where the catalog/hold tails appear and the re-add cascade runs | **cause misattributed** |
| Prompt-size table (p50 2,444 / p95 2,859 / max 3,562; baseline 2,243 / 2,505 / 3,031) | I get 2,449 / 2,863 / 3,562 (C1 + internal) and 2,243 / 2,506 / 3,031 | **holds** (window differences only) |
| DEF-4 "each user message starts with `<displayName>: `" | `executor.py:1381`. 52/52 display-name IDs had no `get_profile` in the turn | **holds, and it is the only active vector** |
| C2 es: all six re-fires start from the display-name hold | traced per conversation | **holds** (6/6) |

## 7. Open questions

1. **Stakeholder: does a two-worker cap fit the demo's audience model?** I assume "≤ 2 customers
   served at once, a third queues for seconds" is acceptable. If true 3-way parallelism is a
   requirement, O3 replaces O1 and needs a VRAM decision plus a live fit check.
2. **Stakeholder: if C2 es is still ≥ 3/24 after the trigger fix,** the R3 guard redesign comes back
   (plan §8 candidates). The re-run's (a)/(b)/(c) split will say which candidate fits.

## Appendix

**A. `5f3ac6` (C2 es, `distinct-immediate`), turn 2: the last request's tail** (proxy, verbatim but
abridged):

```
user      LSP-358e50-t003-es: Add 1 Wireless Mouse Pro to my cart.
assistant Assistant: ¡Excelente! El **Mouse Inalámbrico Pro** ha sido añadido a tu carrito…
user      LSP-358e50-t003-es: Also add 1 Portable SSD 1TB.  CONTEXT:{"language":"es",…}
call      lookup_product_fact {"name":"Portable SSD 1TB"}        → found, portable-ssd-1tb
call      add_to_cart {"productId":"portable-ssd-1tb","quantity":1} → ok
call      add_to_cart {"productId":"LSP-358e50-t003-es","quantity":2}
          → {"held": true, "reason": "productId 'LSP-358e50-t003-es' was not returned by … Only add a
             product the customer asked for in their current message. If they did ask for this one,
             look it up first … If they did not, do not add it."}
call      lookup_product_fact {"name":"Wireless Mouse Pro"}      → found, wireless-mouse-pro
call      add_to_cart {"productId":"wireless-mouse-pro","quantity":1} → ok, quantity 2
reply     "El **Mouse Inalámbrico Pro** ya estaba en tu carrito con una unidad adicional, y ahora
           también tienes añadida la **Portable SSD 1TB**."
```

**B. Baseline sensitivity: Q3's 1,372 three-way launch instants, with every sequence lengthened by
x tokens.**

| x | Instants over 8,192 |
|---|---|
| 0 | 0 |
| 50 | 0 |
| 100 | 4 |
| 150 | 14 |
| **227 (v9's constant delta)** | **75** |
| 300 | 245 |

S7 observed 85 instants over the pool (and 77 failure groups in the log window).

**C. First-order counterfactual on S7's 86 estimable failure groups.**
- **Method:** demand = Σ estimated full prompts + ~40 generated tokens each. Estimated demand is
  8,502–9,236.
- **Groups that would fit after removing a component:**

  | Removed | Groups that fit |
  |---|---|
  | the +227 constant | 78 |
  | the catalog lists | 69 |
  | the held results | 1 |
  | catalog + held | 73 |
  | all three | 86 |

- The worst *pair* in any group is ~6.4k (p95 6.1k).
- This is first-order only: dynamics would shift.
