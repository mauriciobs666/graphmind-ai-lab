# Salesperson demo on one LLM — live QA (U3)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-066

## Summary

**What was tested:** plan §7 (Q0–Q5, E1/E2) for K-066, live, on 2026-10-22 between 08:51 and 11:01
(−03:00), against `ad01aee7`. In that build, `query_graph_data`'s internal call uses the calling
agent step's model, so the salesperson storefront runs on **one chat LLM**
(`mistralai/ministral-3-3b`) plus the embedder. Per the stakeholder's ruling, both were loaded for
Q1–Q5.

**Verdict: the single-model demo works and beats the three-model baseline on every live measure.**
**One gated stakeholder item:** Q2's catalog regression on `nlq-12`.

| Step | Verdict | Headline |
|---|---|---|
| Pre-step | done | `reference` re-seeded; `verify_salesperson.sh demo` and `verify_catalog.sh` both exit 0 |
| Q0 | **pass** | fresh qwen comparator: catalog 21/21, KB 13/19; tree clean; no catalog drift vs. the tracked file |
| E1/E2 | **pass** | exactly ministral (`mistralai/ministral-3-3b`) + embedder; qwen absent at all three checks; `:8200` up and untouched |
| Q1 | **pass** (all 3 hard clauses) | no qwen anywhere; `query_graph_data` dispatched in en/pt-BR/es; the `nlq-12` turn names Fitness Tracker Band in all 3 languages |
| Q2 | **GATE: stakeholder item** | catalog 20/21 vs. 21/21. `nlq-12` regresses: ministral emits a duplicate `filters` key, giving a false "no matching data found" |
| Q3 | **pass** | 0/1,125 dead turns (Wilson [0.0%, 0.34%]) vs. 6/1,125; 0 context overflows; 0/1,125 wrong-language by full hand-read; not contaminated (no qwen) |
| Q4 | **trade-off reported** | `TURN_WORKERS=2`: 0 overflows (none to avoid), p95 14.1 s (same); one unexplained 66 s post-suspend outlier; host suspend contaminated trial 19 |
| Q5(a) | **pass** | p95 14.1 s vs. ≤ 29 s (baseline 24.1 s) |
| Q5(b) | TTFT **pass**; first-embed **borderline (stakeholder)** | TTFT penalty +0.087 / +0.075 / −0.217 s; first embed after chat 1.613 / 1.281 / 1.283 s (one above 1.5 s) |
| Q5(d) | **pass** | 0 embedding requests during Q3; no paging during Q3 |

**CPG:** considered, not relevant — this is live behavioral measurement (LLM routing, latency,
GPU memory, NLQ eval), with no code-level question beyond the few call sites the plan and review
already settled. `cpg_falkorchat` is stale per the plan and review.

## Environment

- Code under test: `ad01aee7` (`query_graph_data` inherits the calling step's model). Repo `HEAD`
  at run start: `d359ab73` (docs-only on top of `ad01aee7`).
- Harness `salesperson/scripts/language_salience_probe.py`, sha256
  `5949774a8aafefc8f4b6b3afe2cac381ad1bf8b61f0b3bf7abb5e5831bb10787`, same as the salience2 baseline.
- LM Studio runtime `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.62.1` (U0: 2.59.0; bench: 2.40.0).
  GPU: RTX 4050 Laptop, 6141 MiB, NVIDIA adapter LUID `0x00011540` (U0: `0x00011343`).
- Shared services: FalkorDB `falkordb-dev` and the `:8200` agent-team uvicorn (pid 2112) were up
  throughout. Neither was stopped or restarted.
- `FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json` for every falkor-chat command.
  No tracked file was edited. `models.json` was overridden for Q2 only, via a scratch copy.
- **Instruments, re-created and stored durably (review P3-1), all new and untracked, under
  `salesperson/scripts/`:**

  | File | Replaces | Use | sha256 |
  |---|---|---|---|
  | `gpu_process_memory.ps1` | U0's `gpumem.ps1` | per-process + adapter WDDM Dedicated/Shared Usage, CSV every N s | `f70374ce0f0953ab3cc5a492efbf3ba27c45deff348ff43a1b00024a8c642781` |
  | `swap_bench.py` | U0's `bench.py` | chat×N → embed×M → chat×K, TTFT/tok/s from `/api/v0` `stats` | `4bba57ebd5f0a3b6e3e585ad15778d66c72ef28a019ac6554d5c5ce373222d20` |
  | `single_model_analysis.py` | salience2's `attribute_failures.py` + the architect's p95 | nearest-rank latency, dead turns + Wilson, failure attribution by call site, bursts | `92d01d3b09c853b915096c142b76464f4748306cb005bd9a93f84c851f31df8e` |
  | `nlq_compare.py` | — | pairwise Layer 1 diff of two eval result files, with raw `toolCompletion` | `ebb95d3d1ff2cb15ea148fce752b93e2a900f072d22d4f007eb5d8c38a151b07` |
  | `routing_smoke.py` | — | Q1 driver (reuses the probe's plumbing) | `0d03e56c4b9bedab194a4defe568a66d15e3077ca2df0e1cc428255b71c03a42` |
  | `qwen_watch.sh` | — | 10 s poll of loaded models; logs and unloads any `qwen/*` | — |

  Hashes are as of this run. None of these scripts is committed; `teco` commits.
- Raw data (session scratchpad, not durable; this report is meant to stand alone):
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/01483502-06e5-42ad-9b05-36efaa8d1ff1/scratchpad/u3/`.
  - Probe runs: `single-model.jsonl` (Q3), `q4-workers2.jsonl` (Q4).
  - NLQ results: `nlq-qwen-current.json`, `nlq-ministral.json`.
  - Q1: `q1.jsonl`.
  - Bench: `swap-rep{1,2,3}.jsonl`.
  - Logs: `gpumem.csv`, `lms-runtime.log`, `lms-server.log`, `demo-q3.log`, `demo-q4.log`.
  - Hand-read: `q3-handread.jsonl`.
  - qwen watcher: `qwen-events.log` (+ `.poll`).

## Pre-step — re-seed `reference`

On arrival `reference` was empty: `verify_catalog.sh` reported `Product count : 0 (expected 15)`
and `RESULT: FAIL`, and `verify_salesperson.sh demo` reported both defs `reference def : MISSING`
while the `ws:demo` snapshots were present. `ws:nlq-eval` was intact (12 `Document`, 12 `Chunk`,
62 `Entity`). I re-seeded with `EMBEDDING_DIM=1024 bootstrap_schema.sh demo` → `seed_demo.sh demo`
→ `seed_catalog.sh demo` → `seed_salesperson.sh demo` (chain exit 0, `u3/reseed.log`). After that:
`verify_salesperson.sh demo` exited 0 (`RESULT: OK — 2 defs in sync between reference and ws:demo`),
and `verify_catalog.sh` exited 0 (`RESULT: OK — product catalog in sync (15 products)`).

## Q0 — fresh qwen NLQ comparator

**Result: ran; tree clean. Comparator established.**

- 08:52:17–08:52:23 (−03:00): `lms unload --all` ("No models to unload"), then
  `lms load qwen/qwen3-4b-2507 -c 8192 -y` (3.26 s, 2.33 GiB) and
  `lms load text-embedding-qwen3-embedding-0.6b -y` (1.73 s). `lms ps` listed exactly those two
  (`qwen/qwen3-4b-2507` 8192/PARALLEL 4, embedder 8192). The qwen load warned
  `2 models match the provided model key … Loading the first one`; the IDENTIFIER was still
  `qwen/qwen3-4b-2507`.
- Eval run with `FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json` and
  `NLQ_EVAL_RESULTS_PATH=u3/nlq-qwen-current.json` (parent directory created first). Printed
  `step model: lmstudio/qwen/qwen3-4b-2507`. `runAt` 2026-10-22T11:53:30Z. Each pair took 1.3–2.7 s.
- `git status --short falkor-chat/server/tests/eval/` afterwards printed nothing.

| Dataset | Q0 (fresh qwen, runtime 2.59.0) | Tracked file (qwen, 2026-08-30) |
|---|---|---|
| catalog Layer 1 | **21/21** | 20/20 |
| knowledge_base Layer 1 | 13/19 | 13/19 |
| Overall Layer 1 | 34/40 (85.0%, CI [70.9%, 92.9%]) | 33/39 |

Per shape (Q0): aggregation 8/8, compound-filter 4/4, filter-list 7/7, not-found 6/6, single-fact
9/9, relationship-traversal 0/4, conflicting-facts 0/2. Not-found false-answer rate 0/6.

`nlq_compare.py` (tracked → Q0) found one difference only: `nlq-40` (catalog compound filter), which
is absent from the tracked file and correct in Q0. **No catalog pair disagrees between Q0 and the
tracked file**, so there's no runtime-drift signal on catalog. Q2's catalog comparator is therefore
all 21 catalog pairs.

## E2 — single-model set loaded (run before E1 and Q2)

I loaded the set before bringing the demo up, so Q2 ran with no demo process at all. That's a
stricter form of "no demo traffic flowing".

- 08:54:33–08:54:53: `lms unload --all` (unloaded `qwen/qwen3-4b-2507` and the embedder), then
  `lms load mistralai/ministral-3-3b -c 8192 --gpu max -y` (12.57 s, 4.18 GiB), then
  `lms load text-embedding-qwen3-embedding-0.6b --gpu max -y` (3.99 s, 609.54 MiB). There's no
  `--ttl` on either.
- `lms ps` listed **exactly two** models: `mistralai/ministral-3-3b` (IDENTIFIER exactly that; 4.49 GB,
  8192, PARALLEL 4) and `text-embedding-qwen3-embedding-0.6b`. `GET /api/v0/models` (state `loaded`)
  agreed: `mistralai/ministral-3-3b vlm 8192` and `text-embedding-qwen3-embedding-0.6b embeddings 2048`.
- The runtime log for the ministral load shows the same lines U0 saw:
  `W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999999, abort`
  and `n_slots = 4, n_ctx_slot = 8192, kv_unified = 'true'`.
- **Runtime drift:** the log names `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.62.1`. U0 ran on 2.59.0,
  and the bench on 2.40.0.
- **Adapter LUID drift:** the NVIDIA adapter is now `0x00011540`, not U0's `0x00011343`. The
  sampler auto-detects it.
- qwen check #1 (E2): absent. `:8200` answered HTTP 200 and was left alone.

## E1 — demo up

`FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json falkor-chat/scripts/start_demo.sh`
(defaults otherwise: `--reload` off, `FALKORCHAT_STOREFRONT_TURN_WORKERS` unset, so 4) was started
at 08:57 from `ad01aee7`'s code. The log (`u3/demo-q3.log`) shows
`RESULT: OK — product catalog in sync (15 products)`,
`RESULT: OK — 2 defs in sync between reference and ws:demo`, and
`storefront preflight ok: ws=demo agent=assistant def=salesperson@v8 products=15 images=14` at
08:57:50. The uvicorn pid is 10746, the only process I started. `POST :8000/mcp` → **404**,
which confirms the storefront process mounts no `/mcp` (review P3-3).

## Q1 — routing + catalog smoke

**Result: PASS on all three hard clauses.** There are two aggregate/superlative-quality
observations, recorded verbatim below.

Driver: `salesperson/scripts/routing_smoke.py`. It makes one fresh session per language, run in
sequence; each session asks three turns. Data: `u3/q1.jsonl`, plus `toolsUsed` in
`u3/q1-toolsused.txt`.
Ground truth, read from `reference`: Wearables has 2 products (Smartwatch Series 5 $249.99, Fitness
Tracker Band $79.99). The cheapest product overall is Gaming Mouse Pad XL ($19.99).

| Lang | Turn | `toolsUsed` | Reply (verbatim, trimmed) | vs. catalog |
|---|---|---|---|---|
| en | aggregate | `[query_graph_data]` | "I currently have **2** products available in the Wearables category." | correct |
| en | superlative | `[query_graph_data]` | "The cheapest product I have available is priced at **$19.99**." | correct price; no name given |
| en | `nlq-12` | `[filter_products]` | "I currently have **one** Wearable product under $200: - **Fitness Tracker Band** – $79.99" | correct |
| pt-BR | aggregate | `[query_graph_data]` | "Tenemos dois produtos na categoria de Wearables." | correct count; the sentence opens with the Spanish verb *Tenemos* |
| pt-BR | superlative | `[query_graph_data]` | "O produto mais barato que temos é o **Gaming Mouse Pad XL**, mas ele não está na categoria de Wearables." | correct |
| pt-BR | `nlq-12` | `[filter_products]` | "Na categoria de Wearables, temos apenas um produto com preço abaixo de US$ 200: - **Fitness Tracker Band** (US$ 79,99)." | correct |
| es | aggregate | `[query_graph_data]` | "En nuestra tienda hay **2 productos** disponibles en la categoría de *Wearables*." | correct |
| es | superlative | `[lookup_product_fact, query_graph_data]` | "El producto más barato en la categoría **Wearables** es el **Fitness Tracker Band**, que cuesta **$79.99**." | **wrong for the question asked**: it carried "Wearables" over from the previous turn; the cheapest product overall is $19.99 |
| es | `nlq-12` | `[filter_products]` | "En la categoría **Wearables**, solo hay un producto que cuesta menos de $200: - **Fitness Tracker Band** – $79.99" | correct |

- **(a) PASS.** Both `/api/v0/models` snapshots (run start and run end) listed only
  `mistralai/ministral-3-3b` (vlm) and `text-embedding-qwen3-embedding-0.6b` (embeddings). The demo
  log's only `qwen` substring is the embedder's pin line
  (`ws:demo already pinned to 'lmstudio/text-embedding-qwen3-embedding-0.6b' — no-op`); the string
  `qwen/qwen3-4b-2507` appears nowhere in it. In LM Studio's server log since E2, every
  `Running chat completion` line is `[mistralai/ministral-3-3b]`. The `qwen/qwen3-4b-2507` mentions
  there are only inside downloaded-model listing JSON. Of those completions, 89 are two-message
  conversations, the shape of a `query_graph_data` (or eval render) call. Q2 accounts for 80 of the
  89, and the rest fall in the Q1 window.
- **(b) PASS.** `query_graph_data` dispatched in every language: en 2, pt-BR 2, es 2.
- **(c) PASS.** The `nlq-12` turn names **Fitness Tracker Band** in all three languages, without
  claiming nothing matches. The agent sent it to `filter_products` every time, so Q2's `nlq-12`
  miss didn't surface here. That's consistent with the plan's Pass 1 disposition (M2): this checks
  the shopper-visible answer, not which tool produced it.
- **Misses folded into the stakeholder surface:** es superlative (wrong scope), and the pt-BR reply
  opening with *Tenemos*. Q3's hand-read covers language.

## Q2 — NLQ correctness on ministral (gate)

**Result: GATE NOT PASSED. One catalog regression (`nlq-12`), which goes to the stakeholder.**

The run used `FALKORCHAT_MODEL_CONFIG=u3/models-step-ministral.json`, a scratch copy of
`config/models.json` with only `defaults.step` changed to `lmstudio/mistralai/ministral-3-3b`,
plus `NLQ_EVAL_RESULTS_PATH=u3/nlq-ministral.json`. It printed
`step model: lmstudio/mistralai/ministral-3-3b`. Pairs took 2.0–3.0 s.
`git status --short falkor-chat/server/tests/eval/` afterwards printed nothing.

| Dataset | Q0 qwen | Q2 ministral |
|---|---|---|
| catalog Layer 1 (gated) | 21/21 | **20/21** |
| knowledge_base Layer 1 (reported) | 13/19 | 12/19 |
| Overall Layer 1 | 34/40 | 32/40 (80.0%, CI [65.2%, 89.5%]) |
| catalog Layer 2 (reported) | 17/21 | 6/21 |
| knowledge_base Layer 2 (reported) | 9/19 | 3/19 |

Per shape (ministral): aggregation 8/8, compound-filter **3/4**, filter-list **6/7**, not-found 6/6,
single-fact 9/9, relationship-traversal 0/4, conflicting-facts 0/2. Not-found false-answer rate 0/6.

**The catalog regression, verbatim** (`nlq_compare.py`, `u3/q2-compare.json`):

- `nlq-12`, catalog/compound-filter: *"Show me Wearables products that cost under $200."*
  Expected `{'type': 'set', 'values': ['Fitness Tracker Band']}`. Got
  `expected ['fitness tracker band'], got []`. `toolResult`:
  `{'items': [], 'finding': 'no matching data found'}`. `toolCompletion`:

  ```
  ```json
  {
    "matches": [
      {
        "var": "p",
        "label": "Product",
        "filters": [{"property": "category", "op": "=", "value": "Wearables"}],
        "filters": [{"property": "price", "op": "<", "value": 200}]
      }
    ]
  }
  ```
  ```

  ministral emitted the `filters` key **twice** instead of one two-element list, and left out
  `returns`. The tool turned that into a false "no matching data found". This is the exact failure
  plan §2.3/R3 predicted from the bench. It is also the `nlq-12` turn that Q1(c) checks live.

**Other miss (knowledge_base, not gated):** `nlq-27` *"Can you list all the Product-type
entities?"* The completion targeted label `Product` with no filters, where the KB uses `Entity`
nodes. Result: `no matching data found` (expected `18 million dollars`, `Atlas-7`, `NovaGrid`).

**Unanswerable shapes (§2.3).** Both models gave identical outcomes:
- `nlq-34`/`35`/`36`/`39` abstained (`no matching data found`).
- `nlq-37` returned a row (`Marlowe Robotics`).
- `nlq-38` returned a count row (`count(e.entityId)` 12 for ministral; `count(e.name)` 12 for qwen).

All six are `layer1Correct: false` for both models. This is moot in `ws:demo`, which has no
`Entity` data.

**Layer 2 caveat (reported, not gated).** The eval's Layer 2 render call is the harness's own
prompt, resolved as `step` (`_render_answer` → `gateway.llm("step", …)`), so in Q2 ministral
renders it. ministral often answers *"Information was not found."* even when Layer 1 returned the
right rows. Examples: `nlq-16` (expected `Gaming Mouse Pad XL`) and `nlq-21` (expected
`Organization`). In production the shopper-facing rendering is done by the `assistant` agent, which
was ministral before this change too, so this isn't a regression introduced by K-066. But it does
mean the 6/21 doesn't describe the storefront. Q1 checks the storefront's rendering directly.

## Q3 — salience + dead turns

**Result: PASS.** 0 of 1,125 turns were dead (0.0%, Wilson [0.0%, 0.34%]), against a pass line
of ≤ 1.2%. The baseline was 6/1,125 (0.5%, [0.2%, 1.2%]). There were **0 wrong-language replies**
by the v2 classifier; the hand-read is below. Window: run_start 09:00:24.770 → run_end
09:49:07.644 (−03:00), 75 complete trials, no partial trial, no resume, 0 join errors.

| Measure | This run (ministral for both roles) | Baseline salience2 (three models) |
|---|---|---|
| Turns / replied | 1,125 / 1,125 | 1,125 / 1,119 |
| Dead turns | **0** (0.0%, Wilson [0.0%, 0.34%]) | 6 (0.5%, [0.2%, 1.2%]) |
| `Context size has been exceeded` events | **0** | 6, in 2 bursts of 3 |
| Wrong-language, v2 classifier (turn level, per language) | 0/375 each, Wilson [0.0%, 1.0%] | 0/373 each, [0.0%, 1.0%] |
| Wrong-language trials, per language | 0/75 each, [0.0%, 4.9%] | 0/75 each, [0.0%, 4.9%] |
| Unclassified (`unknown`) | 0 | 0 |
| Turn latency p50 / p95 / max (replied, nearest-rank) | **6.0 / 14.1 / 44.2 s** | 6.0 / 24.1 / 62.3 s |
| p95 by turn index 0/1/2/3/4 | 8.0 / 16.1 / 14.1 / 10.0 / 8.0 s | 4.0 / 36.2 / 24.1 / 12.1 / 12.1 s |

Source: `single_model_analysis.py` (`u3/q3-analysis.txt`/`.json`) and
`language_salience_probe.py aggregate` (`u3/q3-aggregate.txt`).

**F4-style abnormal-path triggers:** timed out 0; `POST /messages` non-200 0; `errors` non-empty 0;
no-reply 0. **Attribution:** the demo log for the Q3 window (`u3/demo-q3-window.log`, 6,862 lines)
has **0** `ERROR` lines, 0 `ProviderCallError` and 0 `Traceback`. LM Studio's `server` and `runtime`
logs for the window (`u3/lms-server-q3.log`, `u3/lms-runtime-q3.log`) have 0 lines matching
`context size` or `exceed`. With nothing to attribute, the split by call site is agent 0 / tool 0.

**Tool path exercised:** in the Q3 window, assistant messages' `toolsUsed` include
`query_graph_data` 228 times. The other tools: `get_profile` 527, `add_to_cart` 94, `save_profile` 93,
`view_cart` 72, `lookup_product_fact` 39, `place_order` 7, `filter_products` 6. LM Studio logged
2,607 `Running chat completion` lines in the window, **all** `[mistralai/ministral-3-3b]`.

**LM Studio snapshots** (probe `run_start`/`run_end`): exactly `mistralai/ministral-3-3b` and
`text-embedding-qwen3-embedding-0.6b` loaded at both points. Unlike salience2, the embedder did not
idle out, because it was loaded without a TTL. qwen check #3 at 09:49:29 (`lms ps`): absent.
`qwen-events.log` is empty: the 10 s watcher never saw a `qwen/*` model during Q3, so the run is
**not contaminated**.

**Hand-read (authoritative): 0 wrong-language replies out of 1,125**, every reply read in full
(`u3/q3-replies.txt`). One judgment per reply is in `u3/q3-handread.jsonl`: 1,125 lines, no
duplicates, all `adherent`. The v2 classifier agrees on all 1,125, with 0 disagreements and 0
`unknown`. Turn level: 0/375 per language, Wilson [0.0%, 1.0%]. Trial level: 0/75 per language,
[0.0%, 4.9%].

Same as salience2, the hand-read found product-name oddities that didn't change a judgment:
- pt-BR product names in Spanish- or Italian-looking forms (`Altoparlante`, `Padão`,
  `Tapa de mouse`), plus European-Portuguese verb forms (`salvaste`, `digita`).
- es replies quoting the English name (`(Wireless Mouse Pro)`).

Three borderline lines are noted in the file. All three were judged adherent because the sentence
language is unchanged:
- #885 pt-BR: "…ou algo else?", one English word.
- #1069 pt-BR: "poderé", a Spanish-like conjugation.
- #389 es: opens with "Asiento,".

**Content observations (not language; not attributed to K-066):**
- **Fabricated prices.** Two en replies to "What products do you have?" list scrambled or invented
  prices: #156 (t31k1) gives Wireless Mouse Pro as $349.99 and adds "unusually low" notes, and #286
  (t57k1) calls itself a "final corrected list" with wrong prices. A few replies quote € instead of $.
- **Translated product names block the cart.** es and pt-BR carts usually end up empty, because the
  agent renames "Wireless Mouse Pro" and then can't match it.
- **Off-turn guard.** The demo log holds 146 `WARNING … off-turn write held: … did not dispatch
  add_to_cart(productName=…)`. Every one carries a Portuguese or Spanish rendering of the name:
  `Mouse sem fio Pro` 70, `Ratón inalámbrico profesional (Wireless Mouse Pro)` 32,
  `Ratón inalámbrico profesional` 23, `Ratón inalámbrico Pro` 13, others 8. The K-058 off-turn guard
  holds an `add_to_cart` whose product name doesn't appear in the turn's English scripted text.

All of these are the `assistant` agent's own behaviour, and its model was ministral before this
change too. The salience2 demo log is gone, so this run can't compare the counts.

**Setup for the run.** qwen check #2 at 09:00:19 was absent (`lms ps`: ministral + embedder only).
From 09:00:16 a watcher (`qwen_watch.sh`) polled `/api/v0/models` every 10 s, set to log and unload
any `qwen/*` model. The GPU sampler ran from 08:58 and the LM Studio `runtime`/`server` log streams
from 08:52. Command: `language_salience_probe.py run --trials 75 --base-url http://127.0.0.1:8000
--lmstudio-url http://localhost:1234 --out u3/single-model.jsonl`, with the demo defaults
(`PARALLEL 4`, `TURN_WORKERS` 4).

## Q5 — paging cost

**(a) Turn latency: PASS.** Q3's p95 `turn_ms` over its 1,125 replied turns (nearest-rank, the
baseline's definition) is **14.1 s**, against a pass line of ≤ 29 s and a baseline of 24.1 s.
p50 is 6.0 s and max 44.2 s. Values are quantized by the probe's 2 s poll, as in the baseline.

**(b) Per-switch swap: TTFT PASS; first-embed-after-chat BORDERLINE (stakeholder item).**
Instrument: `salesperson/scripts/swap_bench.py`. It sends `POST /api/v0/chat/completions` with
temperature 0, `max_tokens` 256, and a nonce-prefixed synthetic catalog prompt of
**3,077–3,086 prompt tokens** (U0: 3,106–3,167). It reads `stats.time_to_first_token`/`tokens_per_second`;
embeddings go to `POST /api/v0/embeddings`, timed by wall clock. Sequence chat×4 → embed×5 →
chat×2, repeated three times back to back from 09:52:23, with the demo idle (`u3/swap-rep{1,2,3}.jsonl`).

| Rep | Steady TTFT, chats 2–4 (s) | First chat after embeds, TTFT (s) | Penalty vs steady mean | First embed after chat (s) | Embeds 2–5 (s) | tok/s range |
|---|---|---|---|---|---|---|
| 1 | 1.065 / 1.076 / 1.088 | 1.163 | +0.087 | **1.613** | 0.039, 0.027, 0.024, 0.023 | 35.86–37.77 |
| 2 | 1.078 / 1.077 / 1.080 | 1.153 | +0.075 | 1.281 | 0.037, 0.024, 0.022, 0.021 | 35.73–36.45 |
| 3 | 1.076 / 1.080 / 1.085 | 0.863 | −0.217 | 1.283 | 0.037, 0.025, 0.020, 0.020 | 35.79–36.44 |

- **TTFT penalty:** +0.087, +0.075, −0.217 s, all ≤ 0.3 s, so **pass**. U0 measured +0.07–0.1 s.
- **First embed after a chat:** 1.613 / 1.281 / 1.283 s, median 1.283 s. Two of three are inside the
  ≤ 1.5 s pass line, and one (1.613 s) is in the stakeholder zone (≤ 5 s). U0's single reading was
  1.385 s. Per the teco ruling on review Pass 3's nit, this is reported as a **borderline stakeholder
  item, not a fail**. The band sits just above a single prior reading, and n=3 can't separate
  "≈1.3 s typical" from "sometimes >1.5 s". (It's moot for the storefront path; see (d).)
- **Discarded calibration pass** (`u3/swap1.jsonl`, not used above): my first bench prompt came out at
  7,806–7,810 tokens, about 2.5× U0's, which made it non-comparable. It's kept on disk for
  transparency: TTFT 2.96–3.18 s, 32.0–32.6 tok/s, first embed after chat 1.798 s. I resized the
  prompt and confirmed it with one calibration call at 3,082 tokens (`u3/swap-calib.jsonl`).
- **Drift note:** steady-state TTFT here is ~1.07–1.09 s and tok/s 35.7–37.8, against U0's
  0.75–0.77 s and 38.5–40.5 tok/s at a similar prompt size. The runtime moved from 2.59.0 to 2.62.1
  between the two runs. This run can't separate that from the prompt-content difference
  (re-created generator), so it's reported, not interpreted.
- **Paging is visible in the sampler** (`u3/gpumem.csv`, NVIDIA LUID `0x00011540`). At 09:52:58,
  during rep 1's embeds, `llama-server` (ministral, pid 18620) dropped from **5297 → 3755 MiB**
  dedicated (−1,542 MiB) while LM Studio (embedder, pid 9256) rose from 599 → 2136 MiB dedicated /
  0 → 484 MiB shared. By 09:53:04 it had reversed: `llama-server` 5297 MiB, LM Studio 600 MiB. Rep 2
  showed the same pattern (09:53:53: LM Studio 1951, `llama-server` 3945). Rep 3's embed burst fell
  between 6 s samples. The adapter total stayed at 5887–5917 MiB dedicated throughout. That's the
  regime U0 described (~1.6 GiB of ministral paged out while the embedder runs).

**(d) Embedder traffic during Q3: 0 (as expected).** LM Studio logs every embedding request as
`[INFO] Received request to embed: <text>` in the `server` source, and the embedder's runtime lines
are tagged `text-embedding-qwen3-embedding-0.6b`. I confirmed both signatures with the bench's
own 15 requests. In Q3's window (server log 09:00:24 → 09:49:29) the counts are **0** and **0**.
The session total of 20 matches my two bench runs exactly (5 + 15). So nothing in the storefront
path embeds, consistent with §4.4 measure 3 and review Pass 3 (c), and there's no defect for `teco`.

**Memory over Q3 (report only).** 485 samples, 09:00:20–09:49:08, all from the same pids:

| Series | Dedicated MiB min / median / max | Shared MiB min / median / max |
|---|---|---|
| Adapter total | 5880 / 5886 / 5916 | 181 / 181 / 181 |
| `llama-server` 18620 (ministral) | 5291 / 5291 / 5297 | 180 / 180 / 180 |
| LM Studio 9256 (embedder) | 575 / 575 / 611 | 0 / 0 / 0 |

**No paging happened during Q3.** ministral stayed fully resident and the idle embedder stayed
paged down to ~0.6 GiB the whole time. The demo, in the shape Q3 drives it, never pays the swap that
(b) measures. That answers plan §8 open question 1 with live data: for the storefront path, loading
the embedder buys nothing, and leaving it loaded costs nothing measurable while it's idle. Per the
stakeholder's ruling it stays loaded.

## Q4 — concurrency lever (reported as a trade-off, per the P3-2 ruling)

**Lever (i):** `FALKORCHAT_STOREFRONT_TURN_WORKERS=2`, server-side only. I stopped my Q3 demo
(pid 10746, SIGTERM, clean `Finished server process`) and restarted it with the variable set. The
new uvicorn is pid 24916, and `/proc/24916/environ` shows `FALKORCHAT_STOREFRONT_TURN_WORKERS=2`.
Preflight was OK again at 09:56:19. LM Studio was untouched (`lms ps` before and after: ministral
8192/PARALLEL 4 + embedder). The run was `language_salience_probe.py run --trials 25 …
--out u3/q4-workers2.jsonl`, from 09:56:33 to 10:58:05. Lever (ii) (`--parallel 2`) wasn't run,
because (i) left nothing to fix (see below).

**Environment incident: the host was suspended mid-run.** Every instrument went silent from
**10:07:18 to 10:54:55** (2,857 s): the 10 s qwen watcher's polls, LM Studio's logs and the demo
log. The probe's monotonic clock shows the dead turn below took 4.1 s while its wall-clock window
spans 47 minutes. **Trial 19 straddles the suspend and is contaminated.** Trials 20–24 ran after
the resume. There was no such gap during Q3 (one poll-gap > 20 s in the whole session: this one).

| Measure | Q4, 2 workers, all 25 trials | Q4 excluding trial 19 | Q3, 4 workers, trials 0–24 (same n) | Q3, 4 workers, all 75 trials |
|---|---|---|---|---|
| Turns / replied | 375 / 374 | 360 / 360 | 375 / 375 | 1,125 / 1,125 |
| Dead turns | 1 (0.27%, Wilson [0.05%, 1.49%]) | 0 ([0.0%, 1.06%]) | 0 ([0.0%, 1.01%]) | 0 ([0.0%, 0.34%]) |
| `Context size has been exceeded` | 0 | 0 | 0 | 0 |
| p50 / p95 / max turn latency | 4.0 / 16.0 / 66.3 s | 4.0 / **14.1** / 66.3 s | 6.0 / **14.1** / 20.1 s | 6.0 / 14.1 / 44.2 s |
| Turn-1 p95 (heaviest index) | 20.1 s | 20.1 s | 18.1 s | 16.1 s |
| p95 by turn index 0–4 | 4.0 / 20.1 / 10.0 / 8.0 / 6.0 | same | 8.0 / 18.1 / 12.1 / 10.1 / 10.0 | 8.0 / 16.1 / 14.1 / 10.0 / 8.0 |
| Wrong-language (v2) | 0 (turn level 0/124–125 per language, [0.0%, 3.0%]) | — | — | 0 |
| Embedding requests | 0 | — | — | 0 |

- **The one dead turn** (trial 19, pt-BR, turn 4, posted 10:07:23.655) failed at 10:54:52.435 with
  `ProviderCallError: lmstudio/mistralai/ministral-3-3b @ http://localhost:1234/v1/chat/completions:
  HTTP 400 Bad Request: {"error":"Engine protocol predict stream timed out after 600000ms without
  receiving data."}`. The call site is agent (`_run_agent_node`). LM Studio's 600 s stream timeout
  fired across the suspend. It's an environment artifact, **not a context overflow and not
  attributable to the lever or to K-066**.
- **The 66.3 s max** is trial 22's en greeting ("Hi there!"), posted 10:55:56.982, about a minute
  after the resume. The es and pt-BR participants ran all five of their turns meanwhile
  (10:55:56–10:56:25). Then LM Studio received **no** chat request at all from 10:56:23 until
  10:57:02, when the en turn's first request (a 2-message conversation) arrived. The demo log has
  nothing in that gap except state polls: no error, no sweep line. So the turn sat in the server
  for about 37 s with both workers apparently free. That's **unexplained**. It happened once, right
  after a host suspend, and never in Q3 or in Q4's pre-suspend trials 0–18 (next-slowest Q4 turn:
  26.1 s, trial 7 en turn 1, and Q3 had one at the same index). I'm not raising it as a defect
  from one post-suspend sample. It's recorded so a later run can watch for it: a third participant's
  turn waiting with free workers would be a starvation bug in the 2-worker configuration.
- **Memory:** Q4 has the same resident pattern as Q3 (ministral `llama-server` 5255–5297 MiB
  dedicated). After the resume, the embedder's LM Studio process read 0 MiB dedicated and
  `llama-server` 5255/222 MiB.

**Trade-off for the stakeholder** (not gated against Q5(a), per the P3-2 ruling):
- **Overflows avoided: none to avoid in this run.** At the default 4 workers, Q3 saw 0 context
  overflows in 1,125 turns, so this run can't show a benefit for the lever. U0's synthetic result
  (three concurrent ~3.1k-token requests all fail in the shared 8192 pool) still stands as a
  mechanism. The scripted 5-turn conversations just never got long enough together to reach it here.
- **Latency cost: not measurable at p95.** On the uncontaminated data, Q4's p95 is 14.1 s, the
  same as Q3's first 25 trials. Its p50 is lower (4.0 vs 6.0 s). Turn-1 p95 is +2.0 s (20.1 vs
  18.1 s), one poll step. The queueing P3-2 predicted is visible only in the single 66.3 s outlier,
  whose cause is unexplained (above).
- **My recommendation:** keep the default **4 workers** for the demo. The lever bought nothing
  measurable in this run, and the one oddity seen under 2 workers (a turn idle for ~37 s with free
  workers) would hurt a live audience more than a rare overflow burst. If the stakeholder wants
  insurance against U0's three-long-requests failure, `TURN_WORKERS=2` costs nothing measurable at
  p95. Rerun Q4 clean (no suspend) before adopting it, to rule out the starvation shape.

## Stakeholder items

These go back via `teco`. None was silently passed.

1. **Q2 gate — the catalog regression `nlq-12` (needs an explicit accept or reject).** On the
   production tool path, ministral gets 20/21 catalog pairs, where Q0's fresh qwen gets 21/21.
   - *"Show me Wearables products that cost under $200."* gets the completion
     `"filters": [{…category = Wearables}], "filters": [{…price < 200}]` (a duplicate key, no
     `returns`). The result is `{'items': [], 'finding': 'no matching data found'}`, where the
     expected answer is `Fitness Tracker Band`. The full raw text is in Q2.
   - **Mitigating evidence from Q1(c):** on the live storefront, the agent sent that same question
     to `filter_products` in all three languages and answered correctly. So the regression is
     reachable only when the agent picks `query_graph_data` for a compound filter.
   - The fix path, if rejected, is the prompt work plan R3 names (a compound-filter example in
     `_QUERY_REQUEST_INSTRUCTIONS`, plus a `model-bench` pack bump). That's out of this scope.
2. **Q5(b) — first-embed-after-chat is borderline (report only; the stakeholder decides).** Three
   readings: 1.613 / 1.281 / 1.283 s. The pass line is ≤ 1.5 s and the stakeholder band ≤ 5 s.
   Q5(d) shows the storefront never embeds and Q3 saw no paging at all, so this cost lands only on a
   non-storefront caller that embeds while the demo runs.
3. **Q4 — the `TURN_WORKERS=2` trade-off.** No overflows to avoid at 4 workers in this run, and no
   p95 cost at 2. One unexplained 66.3 s post-suspend turn under 2 workers. **Recommendation:** keep
   4 workers for the demo; rerun Q4 cleanly if the stakeholder wants the cap as insurance.
4. **Plan §8 open question 1 — "does the demo need the embedder loaded?"** Answered from live data:
   0 embedding requests during Q3, and the idle embedder sat paged down at ~0.6 GiB while ministral
   stayed fully resident. For the storefront path, loading the embedder neither helps nor measurably
   hurts. The stakeholder's "load both" ruling stands unchanged.

Q3 dead turns did **not** land in the 1.2%–fail band (0/1,125), so there's no Q3 stakeholder item.

## Defects

None raised against K-066.
- The Q4 66.3 s turn is recorded as an unexplained, single, post-suspend observation, not a defect
  (see Q4).
- The Q2 `nlq-12` miss is the plan's predicted model limitation (R3), surfaced as gate item 1, not a
  code defect.

## Coverage & gaps

**Covered:**
- Q0–Q5 and E1/E2 as specified, with the teco rulings applied.
- Live NLQ eval on both models (40 pairs each).
- Storefront routing in 3 languages.
- n=75 salience/dead-turn rerun with a full hand-read.
- An n=25 lever run.
- A three-repetition swap bench.
- Per-process GPU memory across the whole window.
- LM Studio-side request accounting (model and embedding counts).

**Not covered, and residual risk:**
- **The context-overflow mechanism isn't disproven.** Q3 had 0 overflows, but U0's synthetic
  three-concurrent-~3.1k-token failure stands. Longer free-form conversations than the 5 scripted
  turns could still reach it. The scripted conversation is the only load shape tested.
- **Q4 is weaker than planned:** trial 19 is contaminated by a host suspend, trials 20–24 ran after
  a resume, and lever (ii) (`--parallel 2`) wasn't run.
- **Q5(b) is n=3 per measure.**
- **The `:8200` server stayed up.** No qwen load was observed in about 2 h of 10 s polling, but the
  R2 risk is operational and stays live on demo day.
- **Q1 is one session per language.** It proves the routing exists, not how often each tool is
  chosen.
- **The Layer 2 drop (6/21 catalog) is an eval-harness render** by ministral and wasn't separately
  assessed against storefront rendering beyond Q1/Q3.

## Feedback & recommendations

1. **Demo-day setting:** ministral `-c 8192 --gpu max` + embedder, default 4 storefront workers,
   `qwen` unloaded with `:8200` idle. This configuration produced 0 dead turns in 1,125 and a p95 of
   14.1 s.
2. **Decide the `nlq-12` regression (item 1)** before closing K-066. If it's rejected, route the
   prompt fix through R3's path.
3. **Commit the re-created instruments** in `salesperson/scripts/` so P3-1 can't recur. They're
   untracked as of this run.
4. **Optional follow-ups:**
   - a clean Q4 rerun, to settle the 37 s idle-with-free-workers observation;
   - a long-conversation stress variant of the probe, to probe U0's overflow mechanism in the real
     demo flow;
   - backlog the translated-product-name cart failures (es/pt-BR `add_to_cart` held by the off-turn
     guard) as an agent-quality item. It predates K-066.

## End state

- `lms ps` at 11:01:23: `mistralai/ministral-3-3b` (8192, PARALLEL 4, IDLE) and
  `text-embedding-qwen3-embedding-0.6b` (8192, IDLE). **qwen not loaded.** The watcher logged 0 qwen
  events across the session.
- Stopped only what I started:
  - demo uvicorn pids 10746 (Q3) and 24916 (Q4), both with `Finished server process`;
  - the GPU sampler;
  - the two `lms log stream` processes (Windows pids 4332/16840 and their WSL wrappers);
  - the qwen watcher.
- Still running and untouched: `:8200` uvicorn (pid 2112, HTTP 200) and FalkorDB (`PONG`).
- `reference` is seeded: `verify_salesperson.sh demo` and `verify_catalog.sh` both report
  `RESULT: OK` at the end.
- No git command that mutates the tree was run. Untracked new files: this report and the five
  instrument files in `salesperson/scripts/` (`gpu_process_memory.ps1`, `nlq_compare.py`,
  `routing_smoke.py`, `single_model_analysis.py`, `swap_bench.py`), plus `qwen_watch.sh`.
- `git status --short falkor-chat/server/tests/eval/` was empty after both Q0 and Q2.
- `claude/**` and `resume_*.sh` were not touched.
