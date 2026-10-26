# Cart writes by product ID — live re-run after the S7 fixes (id2 S6)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-078

## Summary

**What was tested.** I ran design note §6.2 live on 2026-10-25 between 20:13 and 21:11 (−03:00):
- **Code under test:** `salesperson@v9` at `HEAD` `a5d8e807`: D8 (`64a75019`), D9 (`a5d8e807`)
  and the 2-worker default (`e4cdba9f`).
- **Phases:** warm-up, then the 12-trial language pre-check, then C1 (75 trials), C2 (n = 24 ×
  3 languages), the confirmation variant and the new browse variant.
- **Test plan:** `falkor-chat/docs/test-plans/salesperson-cart-product-id2.md` (TP-100…TP-118).

**Verdict: C1 PASSES every gate. C2 ESCALATES: es re-fired 8/24.**

| | |
|---|---|
| Pre-check | 0 wrong-language replies in 72 (classifier and full hand-read). The stop rule did not fire |
| C1 | M1 75/75 in each language; M2 0/225; M3 0/225; dead turns 0/1,125; p95 14.05 s (turn 2: 8.03 s); wrong language 0/375 per language |
| **C2, escalation** | **es 8/24 [18.0, 53.3], all in `distinct-immediate` (8/12)**; en 0/24, pt-BR 0/24. Display-name bait 0; the re-add now uses the correctly guessed slug |
| KV | max 2 active slots; worst in-flight 5,871 of 8,192 (margin 2,321); 0 overflows |
| Caveat | **D8 never ran.** `filter_products` was called 0 times in 3,988 chat calls. pt-BR passed by looking up the English name, not through D8 |
| New defects | DEF-C (Major): a translated category browse gets "we have none" through `query_graph_data`. DEF-A (Minor): an `Assistant:` label leaks into 23.6% of C1 replies. DEF-B (Minor): the en greeting claims the profile name as the assistant's own |

CPG: considered, not relevant — this is live behavioural measurement of LLM tool use against a running demo; the code paths were settled at the code gate.

## Environment

- **Code under test:** `HEAD` `a5d8e807` (D9), with `64a75019` (D8) and `e4cdba9f` (2-worker
  default) beneath it; `salesperson@v9`, unchanged. The working tree's only change under test is
  `refire_probe.py` gaining the `browse` script (harness, not product code).
- **Bring-up:** `start_demo.sh`, with `LMS_CLI` pointed at the Windows `lms.exe` and
  `FALKORCHAT_OPENCODE_CONFIG` at the scratch proxy config. LM Studio started with nothing loaded;
  `ensure_demo_models` loaded both demo models ("loaded: mistralai/ministral-3-3b,
  text-embedding-qwen3-embedding-0.6b — demo models ready"). Bring-up logged `Turn workers: 2` and
  `storefront preflight ok: ws=demo agent=assistant def=salesperson@v9 products=15 images=14`. The
  uvicorn process environment had `FALKORCHAT_STOREFRONT_TURN_WORKERS=2`.
- **LM Studio** (`lms ps` after bring-up): `mistralai/ministral-3-3b` CONTEXT 8192, PARALLEL 4;
  `text-embedding-qwen3-embedding-0.6b`. The runtime log reports `n_slots = 4, n_ctx_slot = 8192,
  kv_unified = 'true'`. **The runtime is `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.64.0`; S7's was
  `2.62.1`.** That is an uncontrolled difference from the S7 baseline.
- **Processes:** my uvicorn was pid 9147, the only server I started. `falkordb-dev` and the `:8200`
  server (pid 2243) were never touched. I wrote nothing to `:8200`.
- **Verify scripts:** `verify_salesperson.sh demo` and `verify_catalog.sh` returned `RESULT: OK`
  before the run, after it (21:10), and again on 2026-10-26 before this report was finalized.
  `reference` was never wiped, so no re-seed was needed.
- **Teardown:** I stopped my uvicorn (pid 9147), the proxy and both log streams. The demo models
  stay loaded in LM Studio: I never unload.
- **Instrumentation:** the S7 scratch harness (`…/scratchpad/k078-s7/`) **no longer exists**, so I
  rebuilt the observe-only proxy (`:1235 → :1234`, one JSON line per request with request and
  response bodies), the slot analysis (RCA §2 item 4 method) and the per-turn analysis. The proxy
  forwards bodies byte-for-byte. LM Studio's `runtime` and `server` log streams (`lms log stream`)
  ran from **before** the models loaded, so the slot log covers every turn.
- **Harness hashes:** `language_salience_probe.py` `5949774a…0787` (identical to S7 and the
  baseline); `cart_outcome_analysis.py` `0fc008f9…dd6c` (identical to S7); `refire_probe.py`
  `62baa763…e3ae2` (S7's plus the `browse` script, `HARNESS_VERSION` 2).
- **Raw data** (session scratchpad, not durable):
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/k078-id2-rerun/`.

## Warm-up (TP-100)

One throwaway en participant, "Hi there!", at 20:14:49, before the pre-check. It replied in 4.0 s,
with no dead turn. **Not counted in any gate.** Its proxy body shows the D9 change live: the
customer line reached the model as `Customer: Hi there!`.

## Language pre-check (TP-101)

`language_salience_probe.py run --trials 12 --turns 2`, 20:15–20:18, after the warm-up: 36
conversations, 72 turns, 0 dead.

| Language | Turn 0 | Turn 1 | Wrong (v2 classifier) | Unknown | Wrong (hand-read, all 72) |
|---|---|---|---|---|---|
| en | 12/12 replied | 12/12 | 0 | 0 | 0 |
| pt-BR | 12/12 | 12/12 | 0 | 0 | 0 |
| es | 12/12 | 12/12 | 0 | 0 | 0 |

**Result: PASS. The stop rule did not fire; C1 proceeded.** I read every one of the 72 replies.

Observation (not a language defect, see DEF-B): all 12 en turn-0 replies mishandle the profile
name. Each called `get_profile`, which returned `{"name": "LSP-cff12f-t0NN-en", …}`. 7 replies say
"Hello! My name is LSP…" and ask the customer for a name; 5 say "I don't have your name saved yet".
pt-BR and es turn 0 called no tool and replied "Olá! Como posso ajudar você hoje? 😊" /
"¡Hola! ¿En qué puedo ayudarte hoy?" (12/12 each).

## C1 — gate table

`language_salience_probe.py run --trials 75`, 20:18:49–20:53:56: 225 participants, 1,125 turns,
3-way concurrent per trial against 2 turn workers (the third participant queues). Wilson 95%
intervals in brackets.

| TP | Gate | Threshold | S7 (2026-10-23) | This run | Spread (other definitions) | Result |
|---|---|---|---|---|---|---|
| TP-102 | M1 es | ≥ 60/75 | 59/75 FAIL | **75/75** [95.1, 100] | graph recount by `Customer.name` prefix: 75; cart 75, order 0 | **PASS** |
| TP-102 | M1 pt-BR | ≥ 60/75 | 25/75 FAIL | **75/75** [95.1, 100] | recount 75; cart 72, order 3 | **PASS** |
| TP-102 | M1 en | ≥ 65/75 | 70/75 PASS | **75/75** [95.1, 100] | recount 75; cart 74, order 1 | **PASS** |
| TP-103 | M2 other-product lines | ≤ 2/225 | 0/225 PASS | **0/225** [0, 1.7] | recount 0; 0 dispatched adds of any product the turn's customer line did not name (proxy) | **PASS** |
| TP-104 | M3 mouse qty > 1 (cart + order) | ≤ 2/225 | 13/225 FAIL | **0/225** [0, 1.7] | recount: summed over lines 0, any single line qty > 1: 0 | **PASS** |
| TP-105 | Dead turns | ≤ 13/1,125 (1.2%) | 258/1,125 FAIL | **0/1,125** [0, 0.34] | no-reply 0; timed out 0; turns with a non-200 LLM call 0; proxy non-200 calls 0 | **PASS** |
| TP-106 | Turn p95, all turns, pooled | ≤ 29 s | 12.0 s (replied only, flattered) | **14.05 s** | over replied turns = over all turns (no dead turn); p50 4.02 s, max 20.11 s | **PASS** |
| TP-106 | Turn p95, turn 2, pooled | ≤ 29 s | 6.0 s | **8.03 s** | p50 4.03 s, max 8.06 s; n = 225 | **PASS** |
| TP-107 | Wrong-language replies, en | ≤ 3/375 | 0/289 | **0/375** [0, 1.0] | 0 unknown; hand-read (below) 0 | **PASS** |
| TP-107 | Wrong-language replies, pt-BR | ≤ 3/375 | 0/304 | **0/375** | 0 unknown; hand-read 0 | **PASS** |
| TP-107 | Wrong-language replies, es | ≤ 3/375 | 0/307 | **0/375** | 0 unknown; hand-read 0 | **PASS** |

**Every C1 gate passes.** Two sources were counted separately for M1–M3. `cart_outcome_analysis.py`
selects participants from the probe file. My recount selects them from `ws:demo` alone
(`Customer.name STARTS WITH 'LSP-3c65c1-'`, 225 customers, 221 cart lines, 4 order lines). Both
give the same numbers.

**How the latency is measured:** `turn_ms` runs from post to idle, measured by the probe's 2 s
state poll, so every figure is quantized to about 2 s (p95 14.05 s is seven polls). Queue wait is
inside it.

**TP-107 by turn index × language** (v2 classifier, wrong / unknown / n):

| | Turn 0 | Turn 1 | Turn 2 | Turn 3 | Turn 4 |
|---|---|---|---|---|---|
| en | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 |
| pt-BR | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 |
| es | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 | 0 / 0 / 75 |

TP-108 (attribution) does not apply: there is nothing to attribute.

**Hand-read.** I grouped all 1,125 replies by turn index and their first 45 characters. That gave
49 groups in pt-BR, 27 in es and 48 in en, and I read every group. Every group is in the expected
language, including the openings with a leaked `Assistant:` label (DEF-A).

## C2 — re-fire (TP-109)

> **ESCALATION: C2 es re-fired 8/24 (33.3%, Wilson [18.0, 53.3]), above the ≥ 3/24 trigger.
> Per the design note §6.2, this goes back to the stakeholder before K-078 closes, and it is where
> the R3 guard redesign returns.** S7 was 6/24 [12.0, 44.9]. The intervals overlap, so this is
> not evidence of a change in either direction.

`refire_probe.py run --languages en,pt-BR,es --schedule same-category-immediate,distinct-immediate
--repeat 12`, 20:54–20:59. Graph ground truth: mouse quantity summed over `CartItem` + `OrderLine`
is greater than 1. Counted by `cart_outcome_analysis.py --scripts …` and independently by the `ws:demo` name-prefix
recount (`LSP-18e6fe-`). The two agree.

| Language | same-category-immediate | distinct-immediate | Total | S7 | Escalation (≥ 3/24) |
|---|---|---|---|---|---|
| en | 0/12 | 0/12 | **0/24** [0, 13.8] | 0/24 | no |
| pt-BR | 0/12 | 0/12 | **0/24** [0, 13.8] | 0/24 | no |
| es | 0/12 | **8/12** [39.1, 86.2] | **8/24** [18.0, 53.3] | 6/24 | **YES** |

- **Every re-fire is in a single cart line of quantity 2** (`wireless-mouse-pro` ×2 plus
  `portable-ssd-1tb` ×1). There are 0 orders. Both definitions (summed over lines, any single line
  greater than 1) give 8.
- **The customer is told.** The re-fire replies list the mouse at "(x2)" or "2 unidades" (trials
  1, 7, verbatim in the scratch data).
- **Dead turns and latency:** 0/144 dead, 0 non-200 calls. Turn p95 8.05 s (es 10.06 s), max 10.06 s.
- **Holds** (proxy view, all C2 turns): es 21 off-turn holds plus 4 K-061 dedup holds; en and
  pt-BR 0 holds (48/48 `ok` adds each). Display-name `productId`s: **0** in every language (S7: en 5,
  es 14).

**Measure 3 — es turn 2, first mouse-directed call** (RCA §3.2 outcomes):

| Outcome | distinct-immediate | same-category-immediate |
|---|---|---|
| (a) guessed slug, held | **12/12**. The guessed value is the correct slug, `wireless-mouse-pro` | 1/12 (held, then a lookup, then the turn ended; no re-fire) |
| (b) lookup first, un-held re-add | 0/12 | 0/12 |
| (c) no mouse-directed call | 0/12 | 11/12 |

After the (a) hold, the 12 `distinct-immediate` turns split two ways:
- **8:** a second identical held add, then `lookup_product_fact` ("Wireless Mouse Pro" 4 hits,
  "Mouse Inalámbrico Pro" 4 misses with the catalog list), then a **dispatched**
  `add_to_cart(wireless-mouse-pro, quantity 1)`. That is the re-fire.
- **4:** a re-issue of the SSD add, which K-061's same-turn dedup held, and the turn ended. No
  re-fire.

D9 removed the display-name bait as intended. The re-add intent stayed, and now fills in the right
slug guessed from history. That is RCA §3.2's outcome (a), "no change": hold → conditional reason →
lookup → add, which is R3's accepted one-decision depth.

## Confirmation variant (TP-117)

S7's TP-011 script ("Do you have the Wireless Mouse Pro?" → "Yes, add it to my cart."), 12 trials ×
3 languages, 21:00–21:02.

| | en | pt-BR | es | S7 (en / pt-BR / es) |
|---|---|---|---|---|
| Mouse written (graph, both counts) | 12/12 | 12/12 | 12/12 | 12 / 11 / 11 |
| Confirm turns with a held add | 12 | 11 | 12 | 12 / 11 / 12 |
| …recovered to the mouse in the same turn | 12 | 11 | 12 | 12 / 10 / 11 |
| Doubles (M3) | 0 | 0 | 0 | 0 / 0 / 0 |
| Dead turns | 0/24 | 0/24 | 0/24 | 12 of 72 in all |

- **Held values:** en `Wireless Mouse Pro` 12; pt-BR `mouse-sem-fio-pro` 11; es `mouse_pro` 5,
  `mouse-pro` 3, `mouse-pro-wireless` 3, `mouse-wireless-pro` 1. **No display name** (S7: 33 of 35).
  The bait is gone. The model now guesses a slug or a translated slug, which the guard holds, and
  then recovers by lookup.
- Turn p95 8.04 s, max 8.05 s.

## Browse variant (TP-118)

New `refire_probe.py` script `browse`, "Do you have any periféricos?", 12 trials × pt-BR and es,
21:02–21:03. The catalog has 4 `Peripherals` products. Each reply was read and classified by hand.

| Class | pt-BR | es |
|---|---|---|
| Lists the peripherals (correct, all 4) | **4** | **0** |
| Presents the whole catalog as peripherals | 0 | 0 |
| Ends without an answer ("Sim, temos vários periféricos!", names none) | 1 | 0 |
| **Says the store has no peripherals** (a class the spec did not foresee) | **7** | **12** |

- **D8 is not on this path.** 24 of 24 turns called `query_graph_data`, and none called
  `filter_products`.
- The internal NLQ call built `Product` filters `category = "periféricos"` (es) or
  `name = "periféricos"` (pt-BR). Both are exact matches, so the tool returned
  `{"items": [], "finding": "no matching data found"}`. The reply was "Lo siento, no tenemos ningún
  producto relacionado con periféricos…" or "Desculpe, não temos nenhum periférico disponível…".
- The 4 correct pt-BR turns had an English NLQ question ("Do you have any peripherals?"), which
  returned the 4 names. The one "no answer" turn's NLQ returned `count(p.name) = 15`. See **DEF-C**.
- 0 dead; 0 wrong-language replies.

## Reported measures

**1 — KV headroom (TP-110)** (`lms-runtime.log`, RCA §2 item 4 method, rebuilt; the log covers
every turn):

| Window | Slot launches | Max active slots | 2-way in-flight p50 / p95 / max | Margin to 8,192 | Launches > 7,373 | Overflows |
|---|---|---|---|---|---|---|
| Pre-check | 185 | 2 | 4,423 / 5,161 / 5,273 | 2,919 | 0 | 0 |
| **C1** | 2,968 | **2** | 4,970 / 5,266 / **5,871** | **2,321** | **0** | **0** |
| C2 | 500 | 2 | 4,686 / 4,980 / 5,610 | 2,582 | 0 | 0 |
| Confirm | 253 | 2 | 4,671 / 5,277 / 5,383 | 2,809 | 0 | 0 |
| Browse | 77 | 2 | 3,021 / 4,584 / 4,588 | 3,604 | 0 | 0 |
| Whole log | 3,988 | 2 | 4,884 / 5,243 / 5,871 | 2,321 | 0 | 0 |

- **3-way in-flight: none.** No launch ever found three slots active. S7's 3-way max was 8,863.
- **Cross-checks:** the C1 window holds 2,968 launches and 2,968 proxy chat calls. Over the whole
  run, LM Studio's server log has 3,988 `Running chat completion` lines (all ministral), the proxy
  logged 3,988 chat calls, and the runtime log has 3,988 launches. **No other caller reached the
  model.** 0 `send_error`, 0 `failed to find … KV cache` lines, 0 `purging slot`.
- The design note's §4 expected about 6.1k on a normal 2-way pair and 7.4–7.5k on a dead-turn
  cascade. The observed worst pair is 5,871, and no cascade occurred because no turn died.

**2 — Latency next to the dead-turn rate (TP-106, TP-111).** C1 had 0 dead turns, so the
replied-turn and all-turn figures are the same. p95 is 14.05 s all turns and 8.03 s on turn 2;
p50 is 4.02 s.

| C1 language | Turn p95, all turns | Turn-2 p95 | Queue wait p50 / p95 / max (all turns) | Queue wait, turn 2 p50 / p95 |
|---|---|---|---|---|
| en | 16.07 s | 4.04 s | 0.01 / 6.27 / 10.07 s | 0.01 / 0.62 s |
| pt-BR | 10.05 s | 8.04 s | 0.54 / 2.70 / 7.02 s | 1.40 / 3.49 s |
| es | 10.06 s | 8.04 s | 0.48 / 2.58 / 6.67 s | 1.11 / 3.51 s |
| pooled | 14.05 s | 8.03 s | 0.32 / 4.68 s | — |

- **Queue wait** runs from post accepted (probe `posted_at` + `post_ms`) to the first agent call of
  that thread's turn (proxy `at`). It therefore also includes trigger and run setup, not only
  worker queueing. I have only this one definition: the server logs no per-turn start.
- The slowest turn index is turn 1 ("What products do you have?"): p95 16.09 s pooled, en 18.09 s,
  because of the `query_graph_data` path. It is not the D8 path the note's risk named.

**3 — C2 es turn 2:** outcome (a) in 12/12 `distinct-immediate` turns. See the C2 section.

**4 — Display-name `productId`s by vector (TP-114):**

| Vector | C1 | C2 | Confirm |
|---|---|---|---|
| Speaker prefix | 0 | 0 | 0 |
| `get_profile` result in the same turn | 0 | 0 | 0 |
| Echoed name in the assistant history | 0 | 0 | 0 |
| **Total display-name `productId`s** | **0** | **0** | **0** |

- Across all 3,703 agent requests, **0 user rows** start with an `LSP-…:` speaker prefix and
  10,004 start with `Customer:`.
- **D10's trigger** (≥ 3 via the profile or echo vectors in C2 + confirm) **does not fire: 0.**
- The name still reaches the model through `get_profile`, and from there the replies: 144 C1
  replies contain "LSP" (en turn 0 66, pt-BR turn 4 48 and turn 3 14, en turn 3 11, en turn 4 4,
  es turn 4 1). No `productId` came from it. See DEF-B.

**5 — DEF-2 path (TP-113): not exercised.**
- **`filter_products` was called 0 times in the whole run** (3,988 chat calls, every phase), so
  the run has 0 category misses.
- pt-BR reached the mouse by `lookup_product_fact` on 75/75 add turns. The first lookup's argument
  was `Wireless Mouse Pro` in 63 and `Mouse sem fio Pro` (a miss, recovered from the D2 list) in 12.
  In S7, 35 of 44 alive pt-BR add turns took the `filter_products` path.
- **The pt-BR M1 pass is therefore not attributable to D8**, and D8 itself is not verified live by
  this run.

**6 — Carried from S7 (TP-115), C1:**
- **K-058 holds** (demo log, attributed through `WorkflowRun.ctx`): 7. en has 1
  (`Wireless_Mouse_Pro`) and pt-BR 6 (`Wireless_Mouse_Pro` 4, `Wireless Mouse Pro` 2); es 0;
  0 unattributed. **K-061 holds: 0.** S7: 97 (en 88, pt-BR 9).
- **Hold recovery on the add turn:** en 1/1, pt-BR 6/6.
- **First-lookup miss rate** (add turns): en 0/75, pt-BR 12/75, es 6/75. S7: 0/67, 40/40, 72/75.
  es first-lookup arguments: `Wireless Mouse Pro` 69, `Mouse inalámbrico Pro` 3,
  `Ratón inalámbrico profesional` 2, `Mouse Wireless Pro` 1.
- **Turn-2 tool mix** (calls / turns): en `add_to_cart` 76/75, `lookup_product_fact` 75/75; pt-BR
  `add_to_cart` 81/75, `lookup_product_fact` 75/75; es `add_to_cart` 75/75, `lookup_product_fact`
  75/75, `post_message` 2/2, `view_cart` 1/1. `filter_products` 0 everywhere (S7 pt-BR 35/35).
- **LLM calls per add turn**, mean / max: en 3.01 / 4, pt-BR 3.08 / 4, es 3.04 / 5. S7: 3.5 / 4,
  2.4 / 4, 2.8 / 6.
- **Name→price sweep:** 143 pairs, 0 wrong (en 140, es 3, pt-BR 0). F4 tag: 0 turns with a lookup
  miss followed by `query_graph_data`.
- **Non-`$` currency:** 2 replies, both pt-BR turn 4, both `R$` (DEF-5, carried). 0 `€`.
- **Prompt size**, C1 window: p50 2,423, p95 2,604, max 3,263 (n = 2,968). S7: 2,444 / 2,859 /
  3,562. The constant first call (`Customer: Hi there!`) costs 2,172–2,176 tokens (mode 2,175,
  n = 262), against S7's 2,185–2,189, because the fixed label is shorter than the display name.
- **Proxy overhead** (in-process, every chat call, n = 3,988): p50 0.134 ms, p95 0.258 ms,
  max 1.197 ms.

**7 — D8 grounding residual (TP-116): 0.** No category-miss `catalog` ever occurred. Separately,
C1 had **0 dispatched adds of any product the turn's customer line did not name**, by any route.

**Not gated, but changed from S7:** few orders were placed in C1. 4 of 225 participants have an
order (en 1, pt-BR 3, es 0); S7 had 20 (en 14, pt-BR 4, es 2). On turns 3–4 the agent mostly asks
for a delivery address, and the probe never gives one. M1 counts cart or order, so it is
unaffected.

## Defects

**DEF-C — Major — Browsing a category by its translated name gets "we have none" (`query_graph_data` exact-match dead end).**
- **Repro:** bring up `start_demo.sh`, then run
  `refire_probe.py run --languages pt-BR,es --schedule browse --repeat 12`.
- **Expected:** the 4 `Peripherals` products are listed.
- **Actual:** es 12/12 and pt-BR 7/12 reply that the store has no peripherals. A further pt-BR
  reply affirms that it has some but names none.
- **Evidence:** every turn called `query_graph_data` only. The internal NLQ produced
  `{"property": "category", "op": "=", "value": "periféricos"}` (es) or `name = "periféricos"`
  (pt-BR), and the tool returned `{"items": [], "finding": "no matching data found"}`.
- **Context:** this is DEF-2's dead end on the other catalog tool. D8 gives the recovery hint only
  to `filter_products`, which the model never called in this run. A customer is told a whole
  category is out of stock.

**DEF-3 (carried) — Major, stakeholder escalation (R3) — C2 es re-fires 8/24.**
- **Repro:** `refire_probe.py run --languages es --schedule distinct-immediate --repeat 12` (inside
  the 3-language run).
- **Expected:** mouse quantity 1.
- **Actual:** quantity 2 in 8 of 12 conversations.
- **Path:** hold `add_to_cart("wireless-mouse-pro")` (a slug guessed from history), the same held
  add again, a lookup, then a dispatched add.
- **Status:** the guard behaves as designed, and D9 removed only the bait. The ≥ 3/24 trigger has
  fired again.

**DEF-A — Minor — A speaker label leaks into customer-visible replies.**
- **Repro:** C1 (`language_salience_probe.py run --trials 75`).
- **Actual:** 266 of 1,125 replies (23.6%) **begin with** `Assistant:` (or pt-BR `Assistente:`),
  e.g. "Assistant: Necesito tu dirección de entrega…".
- **By turn and language:** en turn 3 40 and turn 4 68; es turn 3 2 and turn 4 74; pt-BR turn 2 5,
  turn 3 13 and turn 4 64.
- **Not seen in shorter conversations:** 0 in the 2-turn pre-check, C2, confirm and browse runs,
  so it grows with history depth.
- **Likely invitation:** the history the model sees is a `Customer: …` / `Assistant: …`
  transcript, and the model continues the pattern.
- **Baseline unknown:** assistant history rows were already `Assistant: …` in S7, but S7's raw
  replies are gone, so I can't say whether D9's symmetric `Customer:` label made this more common.

**DEF-B — Minor — The en greeting claims the customer's profile name as the assistant's own.**
- **Repro:** C1 or the pre-check, en, turn 0 "Hi there!".
- **Actual:** all 75 en turn-0 turns call `get_profile`, which returns `{"name": "LSP-…-en"}`.
  66 of 75 replies say "Hello! My name is LSP-…, …" and ask the customer for a name. 9 say "I
  don't have your name saved yet". 0 greet the customer by name. Pre-check: 7 and 5 of 12.
- **Related:** pt-BR turn 4 greets "Olá, LSP!" in 48 replies.
- **Likely cause:** with D9 the name no longer labels the customer's lines, so a bare profile name
  has no owner in the transcript.
- **Caveat:** probe names are slug-shaped. A real name may fare differently.

**DEF-5 (carried) — Minor —** pt-BR quotes `R$` in 2 replies (correct amounts).

DEF-1 (KV overflow), DEF-4 (display-name `productId`) and DEF-6 (cold FalkorDB timeout) did not
recur. Dead turns are 0 in every phase, display-name `productId`s are 0, and no
`storefront turn failed` line appears in the demo log. That covers DEF-6 too, with the warm-up
taken first.

## Coverage & gaps

**Covered:**
- every §6.2 gate;
- the pre-check, with a full hand-read;
- C2 in all three languages at n = 24;
- the confirm and browse variants;
- reported measures 1–7, plus the carried measures.

**Not covered:**
- **D8 live.** The model never called `filter_products`, so D8's code path is unexercised.
  Measures 5 and 7 are 0 by absence, not by success.
- **Attribution of the pt-BR/es gains.** The model now looks up `Wireless Mouse Pro` in English
  instead of translating it. This run cannot separate three changes since S7:
  - D9 (the prefix);
  - the 2-worker cap;
  - the LM Studio runtime moving from `2.62.1` to `2.64.0`.
- **A second definition of queue wait** (the server logs no turn-start line).
- `remove_from_cart` (no script exercises it).
- Real, non-slug display names, for DEF-B.

**Residual risk:**
- If the model's routing drifts back toward `filter_products`, D8 is what stands between pt-BR
  and S7's 25/75, and it has only offline proof.
- A browse request in a translated category name fails today (DEF-C).

## Feedback & recommendations

1. **Stakeholder: C2 es 8/24.** The escalation trigger has fired again. The display-name bait is
   gone (0 occurrences), but the re-add now comes through a correctly guessed slug. That supports
   the RCA's inference that the vehicle changes and the path does not. The R3 redesign candidates
   (a hold that can't be lifted within the same node execution, or grounding that excludes a turn
   whose customer line didn't name the product) are the next step. → stakeholder, then `architect`.
2. **DEF-C:** extend D8's category-miss recovery to `query_graph_data`'s catalog dataset. An empty
   result whose filter names an unknown category would get the hint and the category list. Or
   steer translated category words to `filter_products`. → `architect`.
3. **D8 live proof:** a scripted turn that reliably calls `filter_products` with a translated
   category (e.g. "Show me periféricos under $100") would exercise it. → `qa-engineer` next pass.
4. **DEF-A / DEF-B:** cosmetic but audience-visible in a quarter of long conversations.
   - For DEF-A, a post-filter that strips a leading `<AgentName>:` from a reply would remove it
     without a prompt change.
   - For DEF-B, the label could carry the profile name when one exists (`Customer (Ana): …`), or
     the greeting rule could change.
   - → `architect`.
5. **Testability:**
   - The S7 scratch harness was lost with the scratchpad. The proxy, slot and per-turn tools
     rebuilt here are the same kind of loss waiting to happen. Promote them next to
     `cart_outcome_analysis.py`.
   - `cart_outcome_analysis.py` expects a `proxy_total_ms` field, a schema the committed tree
     doesn't document. I adapted the log rather than edit the script.

## Learnings (for `cobb`; not written to `ws:agent-team`, which is kept idle for this run)

- `lms log stream -s runtime` timestamps are process-relative (`M.SS.mmm.uuu`), and an event's
  timestamp often sits at the end of the previous line, glued after a `[DEBUG] … pid=N` prefix. A
  per-line parser silently drops about a third of `launch_slot_`/`release` events and inflates
  "active slots". Parse events by text position and anchor the time to a known wall-clock call.
- `lms ps` and `lms log stream` work from WSL2 through `/mnt/c/Users/<you>/.lmstudio/bin/lms.exe`.
  `start_demo.sh` takes it through `LMS_CLI`.
