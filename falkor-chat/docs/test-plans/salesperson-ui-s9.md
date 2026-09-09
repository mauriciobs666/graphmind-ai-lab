# S9 storefront concurrency core — acceptance test plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** salesperson-ui S9 (S9a + S9a-fix), F8 (M<n> TBD) · **Version:** 1.1

*2026-09-09 — v1.1: revised in place after execution (`docs/test-reports/salesperson-ui-s9-report.md`).
Three items gained sub-variants that the run showed were needed to discriminate anything: **TP-011**
splits into `generic` / `redis-timeout` / `TP-011c` (the last reading the status a **real** client
gets, since `TestClient` re-raises server exceptions by default and hides it); **TP-012** gains
**TP-012b** for the same reason; **TP-019** becomes five probes around the branch — `ValueError`,
`KeyError`, `TypeError`, a `redis ConnectionError` from the reset, and a `RuntimeError` from the
reset *itself* — because one negative case does not establish that a catch is narrow. **TP-014 as
first written was not decidable**: it used a single shared gate, so both turns completed before the
assertion and the case passed vacuously; it now releases exactly one turn, which is what makes
"cleared only by its own worker" observable. And **TP-028's criterion was untestable as written** —
S9's done-condition says "poll latency unaffected", which has no threshold and therefore cannot
fail; it is measured here against a falsifiable bound instead (see the report §5).*

## 1. Scope & objective

Verify, **by driving the running system**, the behaviour S9's delivered sub-units owe. Six static
review passes (`docs/reviews/salesperson-ui-impl.md` Passes 17, 20, 21, 22) have judged this code by
reading it; none has executed it at acceptance altitude. This plan targets exactly the gap: the
mechanisms whose correctness is a property of *ordering and state*, which a trace cannot settle.

**Under test** — everything through commit `fc2b43b`:

- `Storefront.reserve_turn` / `release_turn` / `enqueue_turn` / `_run_turn` / `shutdown_turns`
- `Storefront.turn_payload`'s derived queue position
- `POST /shop/api/messages`'s reserve → write → submit sequence and its `409 TurnInProgress`
- `Storefront._await_quiesce` and `reset_participant`, and F8's `504 reset_state_unknown`
- `Storefront._reset_state_unknown`'s widened `except (redis TimeoutError, RuntimeError)`
  (`falkorchat/storefront.py:1413`, hours old, one unit test)
- `FALKORCHAT_STOREFRONT_QUIESCE_S`'s real, observable behaviour

**Explicitly out of scope — not built, so not defects here.** S9 is one plan row split into S9a–S9f
for dispatch (`docs/plans/salesperson-ui-coordination.md` §S9 split table, lines 197–201). Only
**S9a** and **S9a-fix** (plus the P20-1, P21 and Pass-22/U55 repairs) have landed. The following
belong to queued sub-units and are **verified as absent**, not as broken:

| Not built | Owner |
|---|---|
| Cancellation of a queued turn in front of `_await_quiesce` | S9b |
| The dead-turn latch `turn.lastTurn` and its lifecycle | S9c |
| Removal of the per-participant record cache (`lookup`, `_records`, `cached_ids`) | S9d |
| The three `INHERITED_HANDLERS` reason strings + armed-fault measurements | S9e |
| `STOREFRONT_QUIESCE_S`'s documentation | S9f |

Also out of scope: re-reviewing the diff (explicitly excluded by the dispatch brief); P20-7, P20-8
and P17-9, recorded open and not mine to close; the client/SPA half (S12/S13, not built);
real-LLM turn quality (R1) and the §6.4 load harness (S15).

## 2. References

- `docs/requirements/salesperson-ui.md` — FR-4, FR-5, FR-7; AC-3, AC-5
- `docs/plans/salesperson-ui.md` v1.31 — §4.4 (measures 1, 1a, 2, 4), §4.8 (the two resets, **F8**),
  §5.1's **S9 row** (line 1083), §5.2 *The queue position* / *The dead-turn signal*
- `docs/reviews/salesperson-ui-impl.md` `## Pass 22` and the dispositions table — **risk input only**
- `falkor-chat/docs/SERVER.md` §1.3 — the `FALKORCHAT_STOREFRONT_*` env rows
- `falkor-chat/server/falkorchat/storefront.py`, `.../storefront_api.py`
- `falkor-chat/server/tests/test_storefront.py`, `.../test_storefront_api.py` (existing coverage —
  this plan extends past it, at the HTTP/route seam rather than the method seam)

## 3. Risk assessment

Prioritised by **risk × likelihood**, and this ordering is what decides where the effort goes.

| # | Risk | Why it is high | Items |
|---|---|---|---|
| R-a | The `409` is a *check-then-act*, not a reservation — a genuinely concurrent second post admits two turns and the map then reports `idle` under a live turn | P17-1's exact defect; a sleep-timed pair passes on **both** implementations, so only a gated concurrent pair discriminates. The whole invariant `_await_quiesce` rests on | TP-002, TP-004 |
| R-b | The message is written *before* the refusal — a transcript entry with no reply, forever | §4.4 measure 1a's stated reason for the ordering; observable only against the graph, not the response | TP-001, TP-003 |
| R-c | The queue position is stale or wrong at some `turn_workers` | P17-2 shipped exactly this, twice-reasoned. A count in a mutating system read once is a reading, not a state | TP-006–TP-010 |
| R-d | `_reset_state_unknown`'s new `RuntimeError` catch is hours old with one scripted test; its *narrowness* is also a contract (a bug must not hide behind "unknown") | Newest, least-exercised surface. A single passing case is not a state space | TP-015–TP-018 |
| R-e | The reservation leaks (`409`-locks a participant for the process's life) or is released where work **was** queued (an orphan live turn — P20-1) | The two failure directions are asymmetric and the code deliberately chooses one; both edges need execution | TP-011–TP-014 |
| R-f | Shutdown drops accepted turns, or an in-flight turn is killed mid-write | `wait=True`/no `cancel_futures` is the contract; the pre-`submit` flag deliberately loses one turn in a named window | TP-019–TP-021 |
| R-g | `QUIESCE_S` does nothing observable (as `SERVER.md` §1.3 still claims) | S9f open. A documented "changes nothing observable" that is now false is a live trap for an operator | TP-022–TP-023 |
| R-h | The turn runs on the request thread after all — `POST /messages` becomes the slowest route | The single decision the S9 row makes for the implementer | TP-024–TP-026 |

**Deliberately not tested, and why.** (i) Real-LLM turn latency/quality — R1's variable, needs LM
Studio and the §6.4 harness (S15); every item here uses a controlled stub so the assertion is on the
mechanism, not the model. (ii) ~50-participant load (AC-3/FR-5) — that is §6.4's Run A/B, a
separate instrument, and §4.4 already records that AC-3's literal wording is unmet by design.
(iii) The `{handlers} × {routes}` gate and the AST guards — S8's, already asserted in-suite, and
static by construction. (iv) Multi-process behaviour — `--workers > 1` is rejected in §4.4 measure 4.

## 4. Environment & data setup

- **Working directory** `falkor-chat/server`; runner `.venv/bin/python -m pytest`.
- **FalkorDB** live at `127.0.0.1:6379` (v4.18.11).
- **Workspaces.** Integration items run against the suite's own `ws:test` (wiped per test by
  `conftest.py`) and, for the graph-truth assertions, a **dedicated key `ws:qa_s9`** created by this
  pass. `ws:acme` is **stakeholder data**: read-only, verified at 871 nodes at the start and the end
  of the pass. `seed_workflows.sh` / `seed_salesperson.sh` are **never run**.
- **Instruments.** Three, chosen per item:
  1. **`TestClient` over the real `create_app(storefront=True)`** — real routes, real `Storefront`,
     real `ThreadPoolExecutor`, real FalkorDB, with a **gated stub trigger** in place of the LLM so
     turn duration is deterministic. This is the primary instrument: the mechanisms under test are
     in-process by design (the turn map is process-local), so this *is* the running system.
  2. **A real `uvicorn` process** (own launcher in the scratchpad, stub trigger injected before
     serve) for genuine cross-connection concurrency and real SIGTERM → lifespan → `shutdown_turns`.
  3. **`redis-cli GRAPH.RO_QUERY` / `mcp__cypher__query`** for graph-truth assertions — never
     `GRAPH.QUERY` against a graph that may not exist.
- **Known side effect, declared:** the `seeded` fixture wipes the shared `reference` graph on setup
  (`tests/test_storefront_api.py:643`). Any run of that file needs a re-seed afterwards.

## 5. Entry / exit criteria

**Entry** — suite baseline green at the stated 2642 passed / 14 deselected; FalkorDB reachable;
`ws:acme` reads 871 nodes.
**Exit** — every P1 item has an observed outcome (pass/fail/blocked, never inferred); each failure
carries a reproduction and the acceptance criterion it violates; `ws:acme` re-reads 871 nodes.

## 6. Test items

Priority: **P1** = risk-ranked core, must run. **P2** = should run. **P3** = if time allows.
Type: F functional · I integration · C contract · E2E · X exploratory · N non-functional.

| ID | Title | Pre / setup | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|---|
| TP-000 | Suite baseline | clean tree | `.venv/bin/python -m pytest` from `server` | 2642 passed, 14 deselected | P1 | I |
| TP-001 | Two posts 100 ms apart from one participant | gated stub turn, 1 worker | join A; POST /messages; sleep 0.1; POST /messages | 2nd = `409 turn_in_progress`; **exactly one** `Message` authored by A in the graph beyond the first | P1 | E2E |
| TP-002 | Two posts held **concurrently inside the write** | second request enters while the first is blocked inside `services.post_message` | two threads, one `TestClient`, gate released after both are in | 2nd = `409`, **no** `Message` written for it. A check-then-act answers `200` here | P1 | E2E |
| TP-003 | The refused message is genuinely absent from the graph | after TP-001/TP-002 | count A's `Message` nodes before and after the refused post | delta = **0** for the refused post | P1 | I |
| TP-004 | Turn map is not corrupted by the refused post | after TP-002 | read `turn_in_flight(A)` while the accepted turn runs, then after it ends | `True` throughout the turn, `False` after; never `False` under a live turn | P1 | I |
| TP-005 | Two **different** participants are independent (FR-4) | 2 workers | A and B post concurrently | both `200`; both messages written; each `GET /state` shows only its own | P1 | E2E |
| TP-006 | Queue position, `turn_workers=1`, three participants | gated stub | sample `GET /state` for all three once the first is observed `thinking` | `thinking/0`, `queued/0`, `queued/1` | P1 | E2E |
| TP-007 | The countdown | continue TP-006 | release the first turn; re-sample | the third falls to `0`; positions are **strictly non-increasing** across the sequence | P1 | E2E |
| TP-008 | Position at `turn_workers=4` | 4 running turns + a fifth arrival | sample the fifth | `queued/**0**`, not `4` | P1 | E2E |
| TP-009 | `SERVER.md` §1.3's own published measurement | 5 simultaneous arrivals | read the fifth's position at `turn_workers` 1, 2, 4 | `3`, `2`, `0` as documented — or a correction | P2 | C |
| TP-010 | `queuePosition` and `state` always present, `0` when idle/thinking | fresh join | `GET /state` idle; during `thinking` | both keys present; `queuePosition == 0` in both | P2 | C |
| TP-011 | `post_message` raising releases the reservation | stub repo raising from the write | POST /messages; then `GET /state`; then POST again | first raises through; `turn.state == "idle"`; second post is **accepted**, not `409` | P1 | I |
| TP-012 | Post after `shutdown_turns()` | shutdown called | POST /messages | `RuntimeError` propagates, **no** map entry left; observe what the HTTP client actually receives | P1 | E2E |
| TP-013 | A `submit` that raises *after* it was entered leaves the booking standing | `threading.Thread.start` patched to raise, warm pool | `enqueue_turn` | `RuntimeError` propagates; `turn_in_flight` **`True`** at that instant; the queued item still runs and clears its own booking | P1 | I |
| TP-014 | A booking is cleared only by its own worker | live turn | `clear_all_turns()`, then a fresh accepted post | the new entry survives the old worker's `finally`; `turn_in_flight` still `True`; the fresh ordinal is **strictly greater** than the wiped one's | P1 | I |
| TP-015 | F8 happy path — reset-mine end to end | A with transcript, cart, profile | `POST /shop/api/reset` | `200 {threadId, language}`; token still works; `profile == {name: <displayName>, deliveryAddress: null}`; old transcript gone; fresh `Thread` in the graph | P1 | E2E |
| TP-016 | F8 ordering 1 — reset times out, re-read succeeds | repo raises `redis TimeoutError` from the reset only | `POST /shop/api/reset` | `504 reset_state_unknown` **with** a state body; never `503`; never `500` | P1 | C |
| TP-017 | F8 ordering 2 — re-read also times out | reset **and** `get_state` raise `redis TimeoutError` | same | `504`, **no** state body, never `500` | P1 | C |
| TP-018 | F8 + the new `RuntimeError` catch | reset raises `TimeoutError`; the re-read raises `RuntimeError` | same | `504`, no state body, never `500` | P1 | C |
| TP-019 | The catch's **narrowness** is also the contract | re-read raises `ValueError` / `KeyError` | same | those must **not** be swallowed into `504` — a genuine bug must not hide behind "unknown" | P1 | X |
| TP-020 | Is a real `RuntimeError` reachable through `get_state`? | production wiring | enumerate `get_state`'s reach for a `RuntimeError` producer | either a named live path, or the finding that the catch is defensive-only | P2 | X |
| TP-021 | Graceful drain | queued turns behind a busy worker | call `shutdown_turns()` | every accepted turn completes; nothing dropped; second call returns immediately | P1 | I |
| TP-022 | Real process shutdown, in-flight turn | uvicorn + stub turn in flight | SIGTERM | lifespan shutdown drains the turn before exit; measure what the in-flight turn experiences | P2 | E2E |
| TP-023 | `QUIESCE_S` — the timeout branch is live | turn in flight longer than `quiesce_s` | `POST /shop/api/reset` | `503 quiesce_timeout` and **nothing reset** (graph unchanged) | P1 | E2E |
| TP-024 | `QUIESCE_S` — the wait branch is live | turn shorter than `quiesce_s` | same | `200`, after a wait bounded by the turn, not by `quiesce_s` | P1 | E2E |
| TP-025 | A turn arriving during quiesce | reset-mine waiting on A's turn | B posts; A posts | B accepted (participant-scoped wait); A `409` — reset-mine has **no** intake stop (S10's) | P2 | X |
| TP-026 | The turn runs off the request thread | stub turn of 2 s | time `POST /shop/api/messages` | response returns in ≪ 2 s; the turn completes later | P1 | N |
| TP-027 | Armed fault — a turn that dies is not a `5xx` | stub trigger raising | POST /messages, then poll `GET /state` | `200`; state returns to `idle`; **no** `lastTurn` field (S9c not built — recorded as the expected gap, not a defect) | P1 | E2E |
| TP-028 | Poll latency unaffected while the queue is full | all workers busy + a queue | time `GET /shop/api/state` | millisecond-scale; no coupling to turn depth (§4.4 measure 1) | P2 | N |
| TP-029 | The anyio limiter is raised inside the lifespan | app started | read `to_thread.current_default_thread_limiter().total_tokens` inside a request | `100`, not `40` (§4.4 measure 2) | P3 | C |
| TP-100 | `ws:acme` untouched | — | node count + label census at start and end | `871` both times; `Message` 52, `Entity` 544, `WorkflowRun` 21 | P1 | I |
</content>
</invoke>
