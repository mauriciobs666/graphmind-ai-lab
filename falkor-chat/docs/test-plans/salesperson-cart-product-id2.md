# Cart writes by product ID — live re-run after the S7 fixes (id2 S6)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-078 · **Extends:** `falkor-chat/docs/test-plans/salesperson-cart-product-id.md`

## Scope & objective

The live re-run of K-078 on the already-published `salesperson@v9`, after three changes:
- `start_demo.sh` defaults to 2 storefront turn workers (`e4cdba9f`, RCA DEF-1 / O1);
- `filter_products` on an unknown category returns a hint plus the price-free catalog
  (`64a75019`, D8, RCA DEF-2);
- storefront customer lines reach the model as `Customer: …` (`a5d8e807`, D9, RCA DEF-4).

The spec is `falkor-chat/docs/plans/salesperson-cart-product-id2.md` v2 §6.2. C1 is gated at the
stakeholder's bar; C2, the confirmation variant and the browse variant are reported. A language
pre-check runs before C1, with a stop rule.

## References

- Design note: `falkor-chat/docs/plans/salesperson-cart-product-id2.md` §3 (D8–D11), §4 (budget),
  §6.2 (re-run spec), §8 (R-A…R-F).
- S7 plan and report (the baseline column): `falkor-chat/docs/test-plans/salesperson-cart-product-id.md`,
  `falkor-chat/docs/test-reports/salesperson-cart-product-id-report.md`.
- RCA: `falkor-chat/docs/reviews/salesperson-cart-product-id-rca.md` §2 (methods, incl. the
  in-flight KV method), §3.
- Code: `64a75019`, `a5d8e807`, `e4cdba9f`; `HEAD` `a5d8e807`.
- Harnesses: `salesperson/scripts/language_salience_probe.py`, `salesperson/scripts/refire_probe.py`
  (gains the `browse` script here), `salesperson/scripts/cart_outcome_analysis.py`.

## Risk assessment

| Risk | Why it matters | Covered by |
|---|---|---|
| R-B: removing the display-name prefix removes the probe's only in-line language cue | a K-065 regression would make every other number moot | TP-101 (stop rule), TP-108 |
| R-A: pt-BR does not pick from a `filter_products` miss list | pt-BR M1 is the gate most likely to fail | TP-102, TP-113 |
| 2-worker queueing pushes latency toward 29 s | p95 gate; a third customer now queues | TP-106, TP-111 |
| R-D: thin KV margin on a cascade | dead turns return | TP-105, TP-110 |
| R-F: the category-miss catalog grounds an un-asked add | wrong-product writes | TP-103, TP-116 |
| R3 one-decision guard, C2 es | doubled purchases; stakeholder escalation | TP-104, TP-109, TP-112 |
| R-C: the display name reaches the model via `get_profile` / greetings | D10 trigger | TP-114 |

## Test items

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| TP-100 | Warm-up turn (DEF-6) | one throwaway participant, "Hi there!", before the pre-check | reported on its own line, in no gate | P3 | e2e |
| TP-101 | Language pre-check | `language_salience_probe.py run --trials 12 --turns 2` (72 turns); classifier v2 plus a hand-read of all 72 replies | **0 wrong-language replies in any language; any one stops the run before C1** | P1 | functional |
| TP-102 | M1 per language | C1: `language_salience_probe.py run --trials 75`; `cart_outcome_analysis.py`, plus an independent graph recount on `ws:demo` | es ≥ 60/75, pt-BR ≥ 60/75, en ≥ 65/75 | P1 | e2e |
| TP-103 | M2 other-product lines | same; each inspected by hand | ≤ 2/225 | P1 | e2e |
| TP-104 | M3 mouse quantity > 1 (cart + order) | same; graph recount; each inspected by hand | ≤ 2/225 | P1 | e2e |
| TP-105 | Dead turns | probe `dead_turn`, plus no-reply turns, plus proxy 4xx/5xx; warm-up excluded | ≤ 1.2% (≤ 13/1,125) | P1 | non-functional |
| TP-106 | Turn p95, all turns and turn 2, pooled | probe `turn_ms` (post → idle, so queue wait counts), nearest rank; over replied turns and over all turns | ≤ 29 s each | P1 | non-functional |
| TP-107 | Wrong-language replies | `aggregate` (v2), every flagged/unknown reply hand-read; by turn index × language | ≤ 3/375 per language | P1 | functional |
| TP-108 | Wrong-language attribution | if TP-107 > 0, which turn indexes | a rise confined to turns 0–1 is D9's | P2 | functional |
| TP-109 | C2 re-fire | `refire_probe.py`, both schedules, en/pt-BR/es, n = 24 each; graph M3 | reported; **≥ 3/24 in any language escalates** | P1 | e2e |
| TP-110 | KV headroom (measure 1) | `lms log stream -s runtime` from before the first turn; RCA §2 item 4 method | reported: peak 2-/3-way in-flight, margin to 8,192, max active slots (expect ≤ 2), launches > 7,373 | P2 | non-functional |
| TP-111 | Queue wait per language (measure 2) | post accepted → first agent call of that thread's turn (proxy `at`) | p50/p95 per language, reported next to dead turns | P2 | non-functional |
| TP-112 | C2 es turn-2 first mouse-directed call (measure 3) | proxy, per RCA §3.2 (a)/(b)/(c) | reported | P2 | integration |
| TP-113 | DEF-2 path per language (measure 5) | proxy: `filter_products` category misses (hint + catalog) and how many of those turns dispatched a right-product add | reported | P2 | integration |
| TP-114 | Display-name `productId`s by vector (measure 4) | proxy: prefix / `get_profile` in turn / echoed name in history, across C1, C2, confirm | prefix 0 expected; D10 trigger ≥ 3 via the last two in C2 + confirm | P2 | integration |
| TP-115 | Carried measures (measure 6) | K-058/K-061 holds and recovery, turn-2 tool mix, non-`$` currency, name→price sweep + F4 tag, first-lookup miss rate | reported | P3 | integration |
| TP-116 | D8 grounding residual (measure 7) | C1 turns where a category-miss `catalog` was followed by an add of a product the turn's customer line did not name | reported | P2 | integration |
| TP-117 | Confirmation variant | `refire_probe.py --schedule confirm --repeat 12`, 3 languages | reported: add happened, held, recovered, doubles | P3 | exploratory |
| TP-118 | Browse variant | new `refire_probe.py` `browse` script, "Do you have any periféricos?", 12 × pt-BR, es | reported: lists peripherals / whole catalog as peripherals / no answer | P3 | exploratory |

## Environment & data

- **Bring-up:** `start_demo.sh` (2-worker default, effective value logged) with `LMS_CLI` set, so it
  loads ministral (`-c 8192`) and the embedder. LM Studio starts with nothing loaded (stakeholder OK
  for the run's duration). A foreign model at bring-up stops the run; nothing is ever unloaded.
- **Verify:** `verify_salesperson.sh demo` and `verify_catalog.sh` exit 0 before and after.
- **Instrumentation:** an observe-only proxy `:1235 → :1234` (rebuilt: the S7 scratch copy is gone),
  reached through a scratch copy of `opencode.local.json` whose baseURL is `:1235`; overhead
  measured. `lms log stream -s runtime` and `-s server` captured from before the warm-up.
- **Isolation:** `:8200` kept idle (no ingest, no `ws:agent-team` writes); `falkordb-dev` and `:8200`
  never stopped. Only the uvicorn, proxy and log streams I start are stopped.
- **Data:** participant IDs from the probe JSONLs; graph read with `GRAPH.RO_QUERY`, filtered by
  participant. No offline pytest suite (it wipes `reference`).

## Entry / exit criteria

- **Entry:** both verify scripts OK; only the demo models loaded; `:8200` idle; effective workers 2.
- **Stop:** any wrong-language reply in TP-101 → stop before C1 and return to `architect`.
- **Exit:** C1 75 complete trials; C2 n = 24 per language; confirm and browse runs complete; report written.

## Out of scope

- Offline unit tests (done at the code gate) and the full pytest suite.
- The R3 guard redesign, DEF-5 (`R$`), any prompt or code change.
- A run at other concurrency settings.
