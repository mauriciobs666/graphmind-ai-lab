# Cart writes keyed by product ID — live acceptance check (S7)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-078 (+ K-079 F3)

## Summary

**What was tested.** I ran plan §6.3 for K-078 live on 2026-10-23 between 14:20 and 15:05 (−03:00):
- **Code under test:** `salesperson@v9`, from `4dac501e` and `8a72f08d`. Repo `HEAD` was `7a63a866`,
  which is docs-only on top.
- **Gates:** the stakeholder's 2026-10-22 bar.
- **C1:** the TP-007 probe, 75 trials of en + pt-BR + es, 3-way concurrent, 5 turns each.
- **C2:** add-then-add re-fire, n = 24 per language. es was optional; I ran it.
- **Extra:** a 12-trial confirmation-turn variant.
- **Test plan:** `falkor-chat/docs/test-plans/salesperson-cart-product-id.md` (TP-001…TP-011).

**Verdict: FAIL. C1 misses four of eight gates. C2 needs escalation: es reached 6/24 re-fires.**

| | |
|---|---|
| C1 gates | en M1, M2, latency and wrong-language pass. **pt-BR M1 (25/75), es M1 (59/75), M3 (13/225) and dead turns (258/1,125) fail** |
| What works | The ID path for es: every es add turn that didn't die wrote the mouse (44/44), and the miss list is what makes that work. en recovers every held guessed slug in a turn that stayed alive (38/38). |
| Blocker 1 (DEF-1) | KV-cache overflow under 3-way concurrency kills 22.9% of turns (baseline 0). It drags es just under its gate, and it causes all 13 M3 doubles. |
| Blocker 2 (DEF-2) | In 35 of the 44 pt-BR add turns that stayed alive, the agent calls `filter_products(category="Mouse sem fio…")` instead of `lookup_product_fact`. That returns no catalog list, so the model reports "not found". |
| **C2, escalation** | **es: 6/24 re-fires (Wilson [12.0, 44.9]), all six in `distinct-immediate` (6/12).** en 0/24, pt-BR 0/24. |

CPG: considered, not relevant — this is live behavioural measurement of LLM tool use against a
running demo; the code paths were settled at the code gate, and `cpg_falkorchat` is stale per the plan.

## Environment

- **Bring-up:** `start_demo.sh`.
  - The first attempt stopped at step 3 because `qwen/qwen3-4b-2507` was loaded. The stakeholder
    unloaded it; I unloaded nothing.
  - The second attempt seeded `salesperson@v9` and logged
    `storefront preflight ok: ws=demo agent=assistant def=salesperson@v9 products=15 images=14`.
  - My uvicorn was pid 171072, the only process I started. I stopped it at 15:05:38.
  - `falkordb-dev` and the `:8200` server (pid 2112) were never touched.
- **Verify scripts:** `verify_salesperson.sh demo` and `verify_catalog.sh` returned `RESULT: OK`
  both before the run and after it.
- **LM Studio:**
  - Exactly `mistralai/ministral-3-3b` (8192, PARALLEL 4, the same as the baseline per `lms ps`) and
    the embedder were loaded.
  - The probe's `run_start`/`run_end` snapshots and a 10 s log-only watcher agree. The watcher had one
    gap, 14:52:01–14:55:53, which the next run's `run_start` snapshot covers.
  - No foreign model appeared.
- **Instrumentation (approved by `teco`):** an observe-only HTTP proxy, `:1235` → `:1234`.
  - The server reached it through a scratch copy of `opencode.local.json`, kept outside the repo, with
    the baseURL set to `:1235`. No code or prompt changed.
  - `ensure_demo_models`'s `GET /api/v0/models` went through it: "both demo models already loaded …
    nothing to do".
  - **Overhead:** time inside the proxy per chat call was p50 0.18 ms, p95 0.37 ms and max 2.17 ms
    (n = 2,851). On an idle `GET /v1/models` (n = 50, two rounds), the median was 3.16 / 3.24 ms
    direct and 4.13 / 3.51 ms through the proxy.
  - **No other caller:** LM Studio's own server log counted 2,386 `Running chat completion` lines for
    14:26–14:56, and the proxy logged 2,388 chat calls in the same window. Nothing else called the model.
- **Harnesses:**
  - `language_salience_probe.py` sha256 `5949774a…0787`, identical to the baseline's.
  - New, untracked, for `teco` to commit:
    - `salesperson/scripts/cart_outcome_analysis.py`, sha256 `0fc008f9…dd6c`. Read-only: `GRAPH.RO_QUERY` only.
    - `salesperson/scripts/refire_probe.py`, sha256 `c560141b…cdd1`.
  - **Calibration:** run on the baseline Q3 file, the analysis script reproduces plan §2 exactly:
    - M1 70 / 1 / 0 (en / pt-BR / es), M2 0, M3 0;
    - 146 holds: pt-BR 74 and es 72. The two holds §2 left unattributed resolve to es through
      `WorkflowRun.ctx`;
    - turn-2 p95 en 20.1 s, pt-BR 10.1 s, es 10.1 s.
- **Raw data** (session scratchpad, not durable):
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/k078-s7/`.
  - Probe runs: `k078.jsonl` (C1), `c2.jsonl`, `c2-confirm.jsonl`.
  - Logs: `proxy.jsonl` (every request and response), `demo.log`, `lms-server.log`, `lms-runtime.log`,
    `models-watch.log`.
  - Analyses: `c1-analysis.{json,txt}`, `c1-aggregate.txt`, `c1-extra.txt`, `c2-*.json`.

## C1 — gate table (n = 75 per language, against the plan §2 baseline)

Spread columns: "add turn alive" counts only participants whose add turn (index 2) didn't die. "No dead turn" counts only participants with no dead turn at all.

| TP | Measure | Baseline (`v8`) | This run (`v9`) | Spread | Gate | Result |
|---|---|---|---|---|---|---|
| TP-001 | M1 pt-BR | 1/75 [0.2, 7.2] | **25/75** (33.3%, [23.7, 44.6]); 21 in cart, 4 in an order | add turn alive 9/44; no dead turn 8/43 | ≥ 60/75 | **FAIL** |
| TP-001 | M1 es | 0/75 [0.0, 4.9] | **59/75** (78.7%, [68.1, 86.4]); 57 in cart, 2 in an order | add turn alive 44/44 ([92.0, 100]); no dead turn 43/43 | ≥ 60/75 | **FAIL** (one short; every miss is a dead turn, DEF-1) |
| TP-001 | M1 en | 70/75 [85.3, 97.1] | **70/75** (93.3%, [85.3, 97.1]); 56 in cart, 14 in an order | add turn alive 48/49; no dead turn 42/43 | ≥ 65/75 | **PASS** |
| TP-002 | M2, other-product lines | 0/225 | **0/225** | — | ≤ 2/225 | **PASS** |
| TP-003 | M3, mouse quantity > 1 (cart + order) | 0/225 | **13/225** (en 8, es 5; 5.8%, [3.4, 9.6]) | participants with no dead turn: 0. All 13 inspected by hand (below) | ≤ 2/225 | **FAIL** |
| TP-004 | Dead turns (`lastTurn == failed`) | 0/1,125 | **258/1,125** (22.9%, [20.6, 25.5]) | no-reply turns 225/1,125 (20.0%); 96/225 participants have ≥ 1 | ≤ 1.2% | **FAIL** |
| TP-005 | Turn p95, all / turn 2, pooled | 14.1 / 14.1 s | **12.0 / 6.0 s** | p50 4.0 s, max 20.1 s; n = 900 replied turns | ≤ 29 s each | **PASS**, but the replied-only definition flatters it, because 225 turns died early |
| TP-005 | Turn-2 p95 per language (reported) | en 20.1, pt-BR 10.1, es 10.1 s | **en 6.0, pt-BR 6.0, es 6.0 s** (n = 49 / 44 / 44) | max 6.0 / 8.0 / 6.0 s | reported | — |
| TP-006 | Wrong-language replies | 0/375 per language | **0/289 en, 0/304 pt-BR, 0/307 es** (v2 classifier, replied turns) | not a full hand-read; see below | ≤ 3/375 per language | **PASS** |

**Hand-read for TP-006.** This is lighter than the baseline's full hand-read. I read the opening
~100 characters of all 611 pt-BR and es replies, grouped by opening, and every third en reply.
None was in the wrong language.

**M3 inspection (TP-003): all 13 doubles have the same shape.**
1. The add-turn `add_to_cart` **was written**, but the turn then died on a later call (DEF-1), so the
   customer never saw a confirmation.
2. On turn 3 ("place my order") or turn 4 ("status"), the model added the mouse again. en did it by
   way of a held guessed slug, then `lookup_product_fact`, then the add; es did it by lookup, then add.
   That is R3's accepted one-decision depth.

Example `p-dd380512…` (en): turn 2 was held, then died. Turn 3 was held, looked up, added and died.
Turn 4 was held, looked up, added (qty 2), then `place_order` placed a ×2 order.

In C1, every add on a non-add turn followed a dead prior turn:
- 13 re-adds;
- 28 first writes (en 13, es 6, pt-BR 9);
- 28 held adds on non-add turns (en 27, pt-BR 1).

No re-add followed a turn that completed.

## C2 — re-fire (TP-010), graph ground truth: mouse quantity summed over CartItem + OrderLine > 1

| Language | same-category-immediate | distinct-immediate | Total | Pre-guard (ml §9.2) | Escalation (≥ 3/24) |
|---|---|---|---|---|---|
| en | 0/12 | 0/12 | **0/24** [0.0, 13.8] | 1/24 [0.7, 20.2] | no |
| pt-BR | 0/12 | 0/12 | **0/24** [0.0, 13.8] | 1/24 | no |
| es (optional) | 0/12 | **6/12** | **6/24** (25.0%, [12.0, 44.9]) | 1/24 | **YES: return to the stakeholder before close** |

- **Dead turns and latency:** C2 had 0 dead turns out of 144, with turn p95 8.0 s. The conversations
  are short, and DEF-1 didn't occur.
- **K-058 holds on turn 2** (demo log): en 5, es 14, pt-BR 0.
- **K-061 holds:** 3, all es. They held the model's re-issue of the SSD add it had just made, and so
  worked as designed.
- **Every es re-fire is the same sequence** in turn 2 ("Also add 1 Portable SSD 1TB."):
  1. look up the SSD and add it;
  2. `add_to_cart(productId="LSP-358e50-t001-es", quantity=2)`. The id is the customer's display
     name, and the guard held it;
  3. after the conditional hold reason, `lookup_product_fact("Mouse Inalámbrico Pro")` (a miss, with
     the catalog list) or `lookup_product_fact("Wireless Mouse Pro")`;
  4. `add_to_cart("wireless-mouse-pro")`, which lands as quantity 2.

  The model ignores "Only add a product the customer asked for in their current message", which is
  exactly R3. The hold fired every time; the guard is one decision deep, as accepted. C2 shows that
  for es on this model, that one decision is not enough.

## Confirmation-turn add (TP-011, review Pass 3; reported)

Script: "Do you have the Wireless Mouse Pro?" → "Yes, add it to my cart.", 12 trials × 3 languages.

| | en | pt-BR | es |
|---|---|---|---|
| Mouse written (graph, M1) | 12/12 | 11/12 | 11/12 |
| Confirm turns with a held add | 12 | 11 | 12 |
| …recovered to the mouse in the same turn | 12 | 10 | 11 |
| Doubles (M3) | 0 | 0 | 0 |

- A confirmation add **does happen**. It is **always held first**, and **is recovered** by a lookup
  in the same turn.
- The held value is nearly always the customer's display name (`LSP-8d3d38-t000-en`, …; 33 of 35 held
  values). Only two en holds used `Wireless Mouse Pro`.
- The model doesn't refuse the confirmation under "current message only". The cost is one extra
  iteration plus a lookup.
- 12 turns of this run died, all to DEF-1 (4 trials × 3 languages, on the confirm turn).

## Reported measures (C1 unless noted)

**K-058 holds** (demo log, attributed through `WorkflowRun.ctx`): **97** in all.
- **en 88:** `Wireless Mouse Pro` 34, `Wireless_Mouse_Pro` 30, `WirelessMousePro` 24.
- **pt-BR 9:** `Wireless Mouse Pro` 8, `WirelessMousePro` 1.
- **es 0.**
- 0 unattributed. Each hold was in a distinct run. **K-061 holds: 0.**

**Hold recovery on the add turn** (LLM log):
- **en:** 61 add turns held once. 38 of 38 alive turns recovered to `wireless-mouse-pro`. Of 23 that
  died, 9 had recovered first.
- **pt-BR:** 8 held, all in turns that died, none recovered.
- **es:** never held on the add turn, because it always looks up first.

**First-lookup miss rate per language** (impl review Pass 3), measured on add turns that called
`lookup_product_fact`:

| | Result `found: false` | Argument not an exact catalog name | First-lookup arguments |
|---|---|---|---|
| en | 0/67 | 0/67 | `Wireless Mouse Pro` 67 |
| pt-BR | **40/40** | 40/40 | `Mouse sem fio Pro` 39, `Mouse Wireless Pro` 1 |
| es | **72/75** | 72/75 | `Ratón inalámbrico profesional` 66, `Ratón inalámbrico Pro` 3, `Mouse inalámbrico Pro` 2, `Mouse Inalámbrico Pro` 1, `Wireless Mouse Pro` 3 |

**The model still translates the lookup name.** es recovers from the miss list every time. In pt-BR,
the lookup path wrote the mouse in 9 of the 9 alive turns where it was taken; DEF-2 is the other path.

**Turn-2 tool mix** (calls / turns containing the tool):

| Tool | en | pt-BR | es |
|---|---|---|---|
| `add_to_cart` | 118 / 71 | 24 / 24 | 53 / 53 |
| `lookup_product_fact` | 67 / 67 | 47 / 40 | 76 / 75 |
| `filter_products` | 1 / 1 | **35 / 35** | — |
| `get_profile`, `save_profile`, `post_message` | 1 / 1 each | — | 1 / 1 each |

- LLM calls per add turn, mean / max: en 3.5 / 4, pt-BR 2.4 / 4, es 2.8 / 6.
- The common en path is guessed slug, held, lookup, add: 38 of 49 alive turns.

**Name→price sweep (K-079, review F6) and the F4 tag:**

| Run | Pairs stated | Wrong | Source |
|---|---|---|---|
| Baseline Q3 | 79 | 31 | RCA §2 |
| Baseline Q3 | 106 | 32 | this run's sweep script, run on the baseline file |
| This run (C1) | **126** | **0** | sweep script (en 86, es 39, pt-BR 1) |

- The script pairs each English catalog name with the first price after it on the same line.
- **F4 tag:** no wrong pair to tag. In all of C1, **0 turns** had a `lookup_product_fact` miss
  followed by `query_graph_data`, so the miss list never sat next to an identity-free price list.
- **Limitation:** translated product names aren't matched. Replies like "Mouse sem fio Pro – R$ 29,99"
  are outside the sweep; every amount I hand-checked there was the correct 29.99.

**Non-`$` currency (K-079 F3):**
- **This run: 7 replies, all pt-BR, all `R$`** with the correct amount (e.g. "**Total:** R$ 29,99").
  There were 0 `€`.
- Baseline: RCA 7 (`€`); by this script's regex, 10 (7 `€`, 3 `R$`).
- The `€` case is gone. `R$` remains in pt-BR (DEF-5).

**Prompt size:** prompt tokens across all successful calls (agent and internal):

| | p50 | p95 | max | n |
|---|---|---|---|---|
| Baseline (from `lms-server-q3.log`) | 2,243 | 2,505 | 3,031 | 2,606 |
| This run | 2,444 | 2,859 | 3,562 | 2,596 |

## Defects

**DEF-1 — Blocker — Under the TP-007 3-way concurrency, LM Studio's shared KV cache overflows, and 22.9% of turns die.**
- **Repro:** `start_demo.sh` (ministral 8192, PARALLEL 4), then
  `language_salience_probe.py run --trials 75`.
- **Expected:** ≤ 1.2% dead (baseline 0/1,125).
- **Actual:** 258/1,125 dead, plus 12 of 72 turns in the confirm run.
- **Evidence:**
  - All 255 failed C1 chat calls are **agent** calls that returned HTTP 400
    `Context size has been exceeded`; none came from the internal `query_graph_data` call.
  - LM Studio's runtime log shows `decode: failed to find free space in the KV cache`.
  - The three participants of a trial die on the same turn. Dead counts are 86/86/86, and the
    per-trial patterns are identical. So one overflow fails every sequence in flight.
  - Deaths are on turns 2–4, where the history is longest: turn 2 88, turn 3 90, turn 4 77, turn 0 3
    (DEF-6).
  - C2's 2-turn conversations had 0 dead out of 144.
- **Cause, inferred from the evidence:** v9 makes each request bigger than v8 did. Its prompt is
  longer, en takes more iterations (guessed slug, held, lookup), and every es add turn carries the
  ~250-token miss list. That pushes the combined concurrent context past what the 8192-token shared
  KV cache holds, which is plan R4 materialized.
- **Impact:**
  - es misses its gate by exactly the dead add turns: 44/44 alive.
  - All 13 M3 doubles follow a write whose confirmation died.
  - A demo audience sees about 1 in 5 turns fail.

**DEF-2 — Major — pt-BR add turns route the translated name to `filter_products(category=…)`, which returns no catalog list.**
- **Repro:** C1 pt-BR, turn "Add the Wireless Mouse Pro to my cart."
- **Expected:** a lookup, then the catalog list, then a pick and an add.
- **Actual:** in 35 of 44 alive pt-BR add turns, the only tool call is `filter_products` with
  `category` set to `"Mouse sem fio Pro"` or `"Mouse sem fio"`, often with `minPrice` 0 and `maxPrice`
  100 or 999. It returns `{"items": [], "finding": "no matching products found"}`, and the reply is
  "Desculpe, não encontrei nenhum produto chamado **"Mouse sem fio Pro"** no catálogo…".
- **Impact:** pt-BR M1 is 9/44 even on alive turns, against 44/44 for es.
- **Context:** plan D2 puts the recovery list only on `lookup_product_fact`. This is R2's shape on a
  path that plan doesn't cover.

**DEF-3 — Major (stakeholder escalation, R3) — C2 es re-fires 6/24.**
- **Repro:** `refire_probe.py run --languages es --schedule distinct-immediate`.
- **Expected:** a mouse quantity of 1.
- **Actual:** quantity 2 in 6/12 conversations. The steps are in the C2 section above.
- **Status:** the guard behaves as designed. The escalation trigger in plan §6.3 has fired.

**DEF-4 — Minor — the model passes the customer's display name as `productId`.**
- **Actual:** values like `LSP-358e50-t001-es` and `LSP-8d3d38-t000-en` appear in 33 of 35 confirm-turn
  holds and in the C2 es re-fires.
- **Impact:** the guard holds every one, so nothing wrong is written, but each costs an iteration.
- **Likely invitation:** each user message in the model's context starts with `"<displayName>: "`.

**DEF-5 — Minor — K-079 F3 doesn't cover pt-BR: 7 replies quote `R$`.** The amounts are correct.
The v9 sentence "All prices are in US dollars; always write them with `$`" doesn't stop a pt-BR reply
from writing `R$`.

**DEF-6 — Minor — the first turns after bring-up time out on FalkorDB.**
- **Actual:** `storefront turn failed` with `redis.exceptions.ResponseError: Query timed out`, raised
  in `Repository.find_waiting_run_for_thread` (from `trigger.maybe_trigger`).
- **Scope:** all three participants of trial 0, turn 0, at 14:20:56, which gives 3 dead turns. It
  never recurred.
- **Likely cause:** a cold read on `ws:demo`, which holds 1,254 `WorkflowRun`s from earlier runs.

## Coverage & gaps

**Covered:**
- every gate row in §6.3;
- every "reported, not gated" row;
- all four review additions;
- C2 for all three languages;
- a confirmation-turn variant.

**Not covered:**
- A run with less concurrency (e.g. `TURN_WORKERS=2`, U3's Q4 lever), to separate the ID design from
  DEF-1. Without it, the es M1 figure of 59/75 is confounded.
- A full hand-read of every reply.
- `remove_from_cart`: not exercised by these scripts.

**Residual risk:**
- The C1 M1/M3 figures for es and en are partly measurements of DEF-1.
- The alive-turn spread (es 44/44, en 48/49) is the best available estimate of the ID path itself.

## Feedback & recommendations

1. **DEF-1 first.** It is environmental capacity, but v9 caused it.
   - One cheap probe that changes no code: re-run C1 with `FALKORCHAT_STOREFRONT_TURN_WORKERS=2`.
   - The levers: shrink the per-call context (the v9 prompt; the miss list's size); load ministral with
     a larger context or fewer parallel slots (a stakeholder call: VRAM); or treat
     `Context size has been exceeded` as retryable in the executor. → `architect` / `devops`.
2. **DEF-2:** give `filter_products`' empty result the same `catalog` list and hint, or steer the v10
   prompt toward lookups for named products. Either way, measure M2 again (plan R2). → `architect`.
3. **DEF-3:** the stakeholder decides whether es 6/24 is acceptable under R3. The plan lists two
   candidates if not: a hold that can't be lifted within the same node execution, or grounding that
   excludes whole-catalog results. → stakeholder, then `architect`.
4. **Testability:**
   - `Message.toolsUsed` is a sorted set, and runs have `trace: false`, so per-turn tool arguments and
     results can only be seen through a request-logging proxy. A storefront-safe per-turn tool trace
     would make this check repeatable without one.
   - The probe's latency gate counts replied turns only, so a run that dies fast looks fast. Report
     latency next to the dead-turn rate, never alone.
