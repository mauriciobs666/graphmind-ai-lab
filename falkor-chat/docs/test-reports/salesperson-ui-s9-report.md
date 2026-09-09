# S9 storefront concurrency core — acceptance test report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** salesperson-ui S9 (S9a + S9a-fix), F8 (M<n> TBD)

## 1. Summary

**S9's delivered concurrency core functions.** Every one of the 31 test points was executed and
observed; **none is inferred, none was left unreached**, and 31/31 passed. This is the first
judgement of S9 made by running the system rather than reading it, after six static passes.

The two earlier attempts at this pass died to session rate limits with results held only in
context. This run was re-executed from scratch with every result written to disk as it landed —
nothing below is transcribed from memory.

- **Under test:** commit `fc2b43b`, `falkor-chat/server` — `Storefront`'s turn reservation,
  bounded executor, derived queue position, quiesce, and F8's `504 reset_state_unknown`.
- **Test plan:** `falkor-chat/docs/test-plans/salesperson-ui-s9.md` (committed at `54ba5dd`).
- **Method:** black-box driving of the real `create_app(storefront=True)` app — real routes, real
  `Storefront`, real `ThreadPoolExecutor`, real FalkorDB — with a controlled stub trigger in place
  of the LLM, plus one real `uvicorn` process for shutdown, and the graph itself as the oracle for
  every "was it written?" question. **Not a code review**: no finding below rests on reading the
  diff.
- **Workspace:** `ws:qa_s9`, this pass's own key. `ws:acme` was never written.
- **CPG:** considered, not relevant — `cpg_falkorchat` is stale (`SOURCE_TREE 85ddeed0…` against a
  current `falkor-chat/server` tree), and this pass is behavioural rather than structural, so no
  test-gap analysis was drawn from it. The coordinator briefed the staleness in advance and I did
  not reason from the graph.

**Verdict: PASS.** The mechanisms the plan's S9 row owes at S9a/S9a-fix scope all work under
execution — the `409` really does precede the message write (including under genuine concurrency
*inside* the write, which is the only shape that separates a reservation from a check-then-act);
the queue position really is derived and really does count down; the reservation is released
exactly where nothing was queued and held where something may have been; the executor drains on a
real `SIGTERM`; and F8's `504` survives all three of its failure orderings, the newest one
included.

**Four findings, none of them a functional failure of the delivered code.** Three are contract or
documentation defects (D-1, D-2, D-3) and one is a correction to the *rationale* recorded for a
delivered fix, not to the fix (D-4). They are ranked in §4.

**What this pass could not reach, stated rather than glossed.** Nothing in the delivered scope was
left unreached. What is *out* of scope is out because it is **not built**: S9b (cancellation of a
queued turn), S9c (the `turn.lastTurn` dead-turn latch), S9d (record-cache removal), S9e (the
`INHERITED_HANDLERS` reason strings) and S9f (the `QUIESCE_S` documentation). I verified those as
**absent**, not as broken — see §5. Two further criteria are structurally beyond this instrument
and were not attempted: AC-3/FR-5's ~50-participant load (that is §6.4's harness, S15) and
real-LLM turn behaviour (R1's variable). Every timing figure here comes from a controlled stub, by
design — the assertions are on the mechanism, never on the model.

## 2. Environment

| | |
|---|---|
| Working directory | `falkor-chat/server` |
| Runner | `.venv/bin/python` (CPython 3.12.3) |
| FalkorDB | `127.0.0.1:6379`, container `falkordb-dev` (restarted by the coordinator between attempts; named volume held) |
| Workspace under test | `ws:qa_s9` (bootstrapped + `seed_demo.sh qa_s9` by this pass) |
| Catalog | `reference` graph |
| Probe scripts | `<scratchpad>/harness.py`, `probe1.py`–`probe6.py`, `server_launch.py` |
| Raw run output | `<scratchpad>/out/*.txt` |

## 3. Results

Legend: **PASS** observed · **FAIL** observed · **NOT REACHED** not run, no result claimed.
**31/31 executed and observed; 0 not reached, 0 inferred.** Evidence paths are relative to the
run's scratchpad; the raw stdout of every probe is preserved there.

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-000 | Suite baseline | **PASS** | `.venv/bin/python -m pytest` -> **2642 passed, 14 deselected**, 25.19 s. `out/tp000_baseline.txt` |
| TP-001 | Two posts 100 ms apart | **PASS** | `r1=200`, `r2=409 {"error":"turn_in_progress"}`; `Message` rows on A's thread `0 -> 1 -> 1`. `out/probe1.txt` |
| TP-002 | Two posts held concurrently **inside the write** | **PASS** | Thread 1 held **inside** `services.post_message`; thread 2's post entered and got `409`; `Message` rows `0` while blocked, `1` after. A check-then-act answers `200` here. `out/probe1.txt` |
| TP-003 | The refused message is absent from the graph | **PASS** | The refused post wrote **0** `Message` nodes and **0** `WorkflowRun`. `out/probe1.txt` |
| TP-004 | Turn map not corrupted by the refusal | **PASS** | `turn_in_flight` `True` throughout the accepted turn, `False` after; never `False` under a live turn. `out/probe1.txt` |
| TP-005 | Two different participants are independent | **PASS** | Concurrent posts from A and B both `200`; 1 message each; A sees `['from-a']` only, B `['from-b']` only (FR-4). `out/probe1.txt` |
| TP-006 | Queue position, `turn_workers=1` | **PASS** | `turn_workers=1`: `thinking/0`, `queued/0`, `queued/1` — exactly §5.2. `out/probe2.txt` |
| TP-007 | The countdown | **PASS** | C's readings across the sequence: `queued/1 -> queued/0 -> thinking/0 -> idle/0`. Strictly non-increasing; it **counts down**. `out/probe2.txt` |
| TP-008 | Position at `turn_workers=4` | **PASS** | `turn_workers=4`, fifth arrival behind four *running* turns reads `queued/`**`0`**, not `4`. `out/probe2.txt` |
| TP-009 | `SERVER.md` §1.3's published measurement | **PASS** | Fifth arrival's position by `turn_workers`: `{1: 3, 2: 2, 4: 0}` — matches `SERVER.md` §1.3's published measurement exactly. `out/probe2.txt` |
| TP-010 | `turn` block shape | **PASS** | `{state, queuePosition}` present, `queuePosition == 0` when idle and when thinking. **No `lastTurn` key** (S9c, not built). `out/probe1.txt` |
| TP-011 | `post_message` raising releases the reservation | **PASS** | 3 variants. `post_message` raising leaves `turn.state` `idle` and the next post is **accepted** (`200`), not `409` — no leaked reservation. A `redis TimeoutError` from the write answers `504 post_state_unknown`, which is what §5.3 C4's reconciliation reads. `out/probe3.txt`, `out/probe4.txt` |
| TP-012 | Post after `shutdown_turns()` | **PASS** | `RuntimeError("cannot schedule new turns after shutdown_turns()")` propagates; **no map entry left**. The message **is** written with no reply (`Message` rows `0 -> 1`) — the plan's accepted cost. Client-facing shape: see **D-2**. `out/probe3.txt` |
| TP-013 | `submit` raising after entry leaves the booking standing | **PASS** | `threading.Thread.start` patched to raise on a **warm** pool: `turn_in_flight(B)` is **`True`** immediately after, the queued item still ran, and it cleared its own booking. The assertion that reddens on any release from a bare `except` around `submit` (P20-1). `out/probe3.txt` |
| TP-014 | A booking is cleared only by its own worker | **PASS** | `clear_all_turns()` under a live turn + a fresh accepted post: ordinals `0 -> 1` (**strictly greater**; a `len(_turns)` counter repeats `0`), and the new entry **survived** the old worker's `finally` (`state='thinking'`), clearing only when its own turn ended. `out/probe3.txt` |
| TP-015 | Reset-mine end to end | **PASS** | `POST /reset` -> `200 {threadId, language: 'pt-BR'}`; the participant's token still authenticates; profile `{name:'ResetAda', deliveryAddress:'12 Elm St'}` -> `{name:'ResetAda', deliveryAddress:None}` (§4.8's operative post-reset fact); old `Thread` rows `0`, fresh `Thread` rows `1`. `out/probe5.txt` |
| TP-016 | F8 ordering 1 — reset times out, re-read succeeds | **PASS** | `504 {"error":"reset_state_unknown", "detail":"…may have committed", "state":{…}}`. Never the quiesce `503`, never a `500`. `out/probe5.txt` |
| TP-017 | F8 ordering 2 — re-read also times out | **PASS** | Reset **and** re-read both `redis TimeoutError` -> still `504`, `state` null, never a `500`. But the key is **present-with-null**, not absent — see **D-3**. `out/probe5.txt` |
| TP-018 | F8 + the new `RuntimeError` catch | **PASS** | Re-read raising `RuntimeError` -> still `504`, `state` null, never a bare `500`. The hours-old widened `except` at `storefront.py:1413` works as written. `out/probe5.txt` |
| TP-019 | The catch's narrowness | **PASS** | 5 probes around the branch. `ValueError`/`KeyError`/`TypeError` from the re-read are **not** swallowed (`500` each) — the narrowness is real and is itself a contract. A `redis ConnectionError` from the reset answers `503 graph_unavailable` (§5.3). A `RuntimeError` from the **reset itself** is `500` — the catch covers the re-read only. `out/probe5.txt` |
| TP-020 | Is a real `RuntimeError` reachable through `get_state`? | **PASS** | `RuntimeError` injected at each of `get_state`'s three service reads (`get_profile`, `get_cart`, `get_current_order`) -> `504` in all three. The catch is functional. **But no production `RuntimeError` producer is reachable through `get_state`** — see **D-4**. `out/probe5.txt` |
| TP-021 | Graceful drain | **PASS** | 3 accepted turns behind 1 worker (0.4 s each): `shutdown_turns()` returned after **1.19 s** having run **3/3**; no booking left standing; a second call returned in **0.0 ms** (idempotent). `wait=True`/no `cancel_futures` holds. `out/probe6.txt` |
| TP-022 | Real process shutdown, in-flight turn | **PASS** | Real `uvicorn` process, `turn_workers=2`, a 6.000 s stub turn. `TURN-START` at `t=379.632`; `SIGTERM` at `t=381.634` (**2.00 s into the turn**); `GET /shop/api/health` **refused within 1 s** (intake stops at once); `TURN-END` at `t=385.632` — the in-flight turn ran to **full completion, 4.0 s after the signal** — and the process exited only then. The turn is drained, not killed. `out/turnlog.txt`, `out/uvicorn.log` |
| TP-023 | `QUIESCE_S` — the timeout branch | **PASS** | `quiesce_s=0.4`, turn held: `503 {"error":"quiesce_timeout"}` after **0.41 s**, and **nothing reset** — messages `1 -> 1`, the original `Thread` still present. `out/probe6.txt` |
| TP-024 | `QUIESCE_S` — the wait branch | **PASS** | `quiesce_s=5.0`, 0.8 s turn: `200` after waiting **0.77 s** — it waited for the turn, then reset (old `Thread` rows `0`). `out/probe6.txt` |
| TP-025 | A turn arriving during quiesce | **PASS** | With A's reset-mine inside `_await_quiesce`: B's post `200` (the wait is participant-scoped — reset-mine has **no** intake stop, which is S10's), A's own post `409`; A's reset then completed `200`. `out/probe6.txt` |
| TP-026 | The turn runs off the request thread | **PASS** | `POST /messages` with a 2.0 s stub turn returned `200` in **5 ms**; the turn ran on `storefront-turn_0`, not the request thread. `out/probe3.txt` |
| TP-027 | Armed fault — a dead turn is not a `5xx` | **PASS** | Trigger raising: `POST` still `200`, state settles to `idle`, participant postable again. Transcript holds the message and no reply. Confirms the placement decision — and **D-4**'s gap. `out/probe3.txt` |
| TP-028 | Poll latency while the queue is full | **PASS** | `GET /state` median **3.3 ms** idle and **3.3 ms** with 2 turns running + 6 queued (max 3.6 ms). Poll latency uncoupled from queue depth (§4.4 measure 1). `out/probe3.txt` |
| TP-029 | The anyio limiter is raised in the lifespan | **PASS** | anyio default thread limiter after lifespan startup: `total_tokens=100` (`config.THREAD_LIMIT`), against anyio's own default of 40 (§4.4 measure 2). `out/probe6.txt` |
| TP-100 | `ws:acme` untouched | **PASS** | `ws:acme` read **871** nodes at the start and **871** at the end; label census `Message` 52, `Entity` 544, `WorkflowRun` 21 — unchanged. Never written to. `reference` ends at **15** `Product` nodes, `verify_catalog.sh` `RESULT: OK`. |

## 4. Defects

Severity is by user impact, not by how hard it was to find. None of these blocks S9a; **D-1 is the
one I would act on before the demo**, because it is a false statement in the document an operator
reads to configure the system.

### D-1 — `SERVER.md` §1.3's `FALKORCHAT_STOREFRONT_QUIESCE_S` row is false as written · **Major** (documentation)

The row still states the pre-S9 world verbatim: *"The wait is delivered but has nothing to wait for
— S9 … **nothing populates the turn map** … Both drains therefore pass on their **first** check,
neither `503 quiesce_timeout` nor `POST /shop/api/messages`'s `409 turn_in_progress` is reachable,
`GET /state`'s `turn` block is always the idle payload, and **setting the value changes nothing
observable**."*

Every clause of that is now wrong. **This is the open item S9f, and it is hereby answered by
execution rather than by argument** — which is what it has been waiting on.

**Reproduction** (`out/probe6.txt`, TP-023/TP-024; `out/probe1.txt`, TP-001):

| Configuration | Observed |
|---|---|
| `quiesce_s=0.4`, a turn held in flight | `POST /shop/api/reset` -> **`503 {"error":"quiesce_timeout"}`** after **0.41 s**, and **nothing reset** (messages `1 -> 1`, original `Thread` still present) |
| `quiesce_s=5.0`, a 0.8 s turn | `POST /shop/api/reset` -> **`200`** after waiting **0.77 s** — it waited for the turn, then reset (old `Thread` rows `0`) |
| a second post during a live turn | **`409 turn_in_progress`**, and the message is **not** written |
| `GET /state` during a turn | `{"state": "thinking", "queuePosition": 0}` — not the idle payload |

So `503 quiesce_timeout` is reachable, `409 turn_in_progress` is reachable, the `turn` block is not
always idle, and **setting the value changes the outcome in both directions**. The neighbouring
`FALKORCHAT_STOREFRONT_TURN_WORKERS` row, by contrast, is **correct and independently confirmed** —
its published measurement (fifth arrival's position `1 -> 3`, `2 -> 2`, `4 -> 0`) reproduced exactly
(TP-009). Only the `QUIESCE_S` row is stale.

**Violates:** §4.8's quiesce contract, and the truthfulness of the operator-facing env table.
**Fix:** rewrite the row against the four measurements above. Owner: S9f.

### D-2 — `POST /shop/api/messages` can answer a bare plain-text `500` that §5.3's completeness table has no row for · **Minor**

In the window between `shutdown_turns()` setting `_turns_shutdown` and the process actually going
away, `enqueue_turn`'s pre-`submit` guard raises `RuntimeError`, which reaches the client as
**`HTTP 500`, `content-type: text/plain; charset=utf-8`, body `Internal Server Error`** — not the
storefront's `{"error": …}` shape (TP-012b, `out/probe4.txt`).

The *lost turn* is explicitly accepted by the plan ("the price is one turn on a process that is
going away"), and I am not reopening that trade. What is not addressed anywhere is the **response**:
§5.3's completeness table carries five rows for this route (`200`, `401`, `409 TurnInProgress`,
`422 text`, `503 demo_not_seeded`) and **no `5xx` row** — the only `5xx` rows in the table belong to
the two reset routes. S8's gate is specified to fail on "a row with no producer, or a handler with
no row", but this response comes from no handler at all, so the gate is structurally blind to it —
which the plan itself predicts under *"What neither half closes"*. C13 would render it as an
unhandled response.

The same shape appears for any unmapped exception out of `services.post_message` (TP-011c) — but
that one is a genuine bug surfacing, which is correct. The shutdown case is a *designed* path
producing an undeclared response.

**Reproduction:** `shop.shutdown_turns()`, then `POST /shop/api/messages` -> `500` plain text; the
participant's message **is** in the transcript (`Message` rows `0 -> 1`) with no reply, and the map
entry is correctly absent.
**Violates:** §5.3's completeness table being total over `(route, response)`.
**Fix (recommendation, not applied):** either add a `5xx` row for this route with the shutdown
window named as its producer, or map the guard's `RuntimeError` to a typed `503` on the way out.
The second is the smaller change and gives C9 something to render.

### D-3 — F8's `504` carries `"state": null` where §4.8 says "with no state body" · **Minor** (contract ambiguity)

§4.8 rules that when the re-read also fails, S7/S10 "still return **`504 reset_state_unknown`**,
simply **with no state body**". Delivered, the key is **present with a null value**:

```
TP-017/TP-018 → HTTP 504
{"error": "reset_state_unknown", "detail": "…may have committed", "state": null}
                                            ^ key present, value null
```

Measured explicitly (`out/probe5.txt`): `'state' key present=True, non-null=False`.

This matters because **this plan is elsewhere precise about exactly this distinction** — §5.2 rules
that reset-all's clean path returns "`200` with **no `incomplete` field at all** — not
`incomplete: false`". Given that precedent, present-with-null versus absent is a distinction the
document treats as contractual, and here it is unstated and delivered the other way. A client
written to `if ('state' in body)` reads a state block that is not there.

**Violates:** §4.8 F8's "with no state body", read against §5.2's own present-vs-absent rule.
**Fix:** decide it explicitly — either omit the key (matching the `incomplete` precedent) or amend
§4.8 to say `state: null`. Either is fine; the silence is the defect.

### D-4 — The `RuntimeError` catch is sound, but the reason recorded for it names a path that does not exist · **Minor** (rationale, not code)

`_reset_state_unknown` (`storefront.py:1413`) catching `RuntimeError` alongside
`redis_exceptions.TimeoutError` **works**, and I verified it three independent ways: a `RuntimeError`
injected at each of `get_state`'s three service reads (`get_profile`, `get_cart`,
`get_current_order`) still answers `504` in all three cases (TP-020), and the catch is correctly
**narrow** — `ValueError`, `KeyError` and `TypeError` are all left to surface as `500` rather than
hiding behind F8's "unknown" (TP-019). That narrowness is itself worth keeping.

**But the scenario the dispatch brief gave as the reason — "a `RuntimeError` from the shutdown guard
cannot turn F8's deliberate `504` into a bare `500`" — is not reachable.** `enqueue_turn` is not in
`get_state`'s call graph at all. `get_state` calls exactly `services.get_profile`,
`services.get_cart`, `services.get_current_order` and its own `turn_payload`; the shutdown guard
lives in `enqueue_turn`, which nothing on that path reaches.

Nor is there any *other* live producer. Every `raise RuntimeError` in the package is either outside
this path or on a **write**: `repository.create_thread` and `repository.place_order` (both writes),
two `services` sites (both writes), plus `mcp.py`, `tools.py` and `config.py`, none of which
`get_state` reaches. So today the catch has **no production producer** — it is purely defensive.

**The source docstring is the accurate one and should be treated as authoritative**: it says the
catch means "a ***future*** `raise RuntimeError` reached through `get_state` still answers F8's
`504`". That hedge is correct; the coordination framing that dropped it overstated the case. The
fix should stay exactly as delivered — this is a correction to the record of *why*, not a request to
change anything. Recorded here rather than softened, at the coordinator's explicit instruction.

**Violates:** nothing in the code. It corrects a rationale.

## 5. Coverage & gaps

### Verified by execution

Stated as criteria rather than as test ids, since that is what the coordination needs.

| Criterion | Where it came from | How it was reached |
|---|---|---|
| The `409` precedes the message write | §4.4 measure 1a | TP-001/002/003 — the graph is the oracle: **0** `Message`, **0** `WorkflowRun` from a refused post |
| The gate is a **reservation**, not a check-then-act | §4.4 measure 1a; P17-1 | TP-002 — the second request enters *while the first is blocked inside* `services.post_message`. This is the only spelling that discriminates; a sleep-timed pair passes on both designs |
| One in-flight turn per participant; participants are independent | FR-4, §4.4 1a | TP-004, TP-005 |
| `queuePosition` is derived, correct at every `turn_workers`, and counts down | §5.2 *The queue position* | TP-006/007/008/009 — read over a **sequence** of `GET /shop/api/state` calls, not a single sample |
| `SERVER.md` §1.3's published position measurement | `SERVER.md` §1.3 | TP-009 — `{1:3, 2:2, 4:0}` reproduced exactly |
| The reservation is released exactly where nothing was queued | S9 row; P17-3, P20-1 | TP-011 (write fails -> released), TP-012 (pre-`submit` shutdown -> released), TP-013 (`submit` fails *after* entry -> **held**) |
| Ownership: a booking is cleared only by its own worker; ordinals strictly monotonic | S9 row; P18-2, P18-3 | TP-014 — `clear_all_turns()` + a fresh post; ordinal `0 -> 1`, the new entry survives the old worker's `finally` |
| The turn runs on the worker, never the request thread | S9 row's central decision | TP-026 (`200` in 5 ms against a 2.0 s turn, on `storefront-turn_0`), TP-027 (armed fault still `200`) |
| Poll latency is uncoupled from queue depth | §4.4 measure 1 | TP-028 — 3.3 ms idle, 3.3 ms with 2 running + 6 queued |
| The anyio limiter is raised inside the lifespan | §4.4 measure 2 | TP-029 — `total_tokens=100` |
| Graceful drain: accepted turns are never dropped | S9 row | TP-021 (3/3 run, idempotent), TP-022 (**real `SIGTERM`**: the in-flight turn completed 4.0 s after the signal, the process exited only then) |
| `QUIESCE_S` is live in both directions | §4.8 | TP-023 (`503`, nothing reset), TP-024 (`200` after waiting) |
| Reset-mine end to end, identity surviving | §4.8, FR-7/AC-5 | TP-015 — token survives, `deliveryAddress` back to `None`, fresh `Thread` minted |
| **F8, all three orderings** | §4.8 F8 | TP-016 (re-read succeeds), TP-017 (re-read also times out), TP-018 (re-read raises `RuntimeError`) — `504` in every case, never `503`, never `500` |
| F8's catch is narrow by contract | `storefront.py:1413` | TP-019 — `ValueError`/`KeyError`/`TypeError` correctly surface as `500` |
| Stakeholder data untouched | dispatch constraint | TP-100 — `ws:acme` 871/52/544/21 at start **and** end |

### Not reached, and why — no inferred passes

**Nothing in the delivered scope was left unreached.** The gaps are all *not-built* or
*wrong-instrument*:

| Not reached | Reason |
|---|---|
| Cancellation of a queued turn ahead of `_await_quiesce` | **S9b, not built.** Verified absent: `_await_quiesce`'s own docstring says "nothing cancels yet", and TP-023 shows the wait-only behaviour (`503`) the plan says cancellation would convert into a `200` |
| `turn.lastTurn` and its lifecycle | **S9c, not built.** Verified absent: the `turn` block is exactly `{state, queuePosition}` (TP-010), and TP-027 shows the consequence live — a turn that dies leaves state `idle` with the message present and no reply, **indistinguishable from a completed turn**. This is the gap S9c exists to close; recorded as expected, not as a defect |
| Record-cache removal (`lookup`, `_records`, `cached_ids`) | **S9d, not built.** Those symbols are still present in `storefront.py`; out of scope for this pass and not counted against S9a |
| The three `INHERITED_HANDLERS` reason strings + armed-fault measurements | **S9e, not built.** TP-027 exercises the armed-fault *behaviour* (a raising trigger still yields `200`), but the reason-string half is S9e's |
| AC-3 / FR-5 — ~50 simultaneous participants | **Wrong instrument.** That is §6.4's load harness (S15). §4.4 already records that AC-3's literal wording is not satisfiable for agent turns on the chosen OQ-1 basis; nothing here changes that, and I did not manufacture a pass against it |
| Real-LLM turn behaviour, latency and quality | **Wrong instrument.** R1's variable; every turn here is a controlled stub, deliberately |
| P20-7, P20-8, P17-9 | Recorded open in the dispositions table and explicitly not mine to close |
| Test-gap analysis from the CPG | `cpg_falkorchat` is stale by several commits including all of S9a-fix. I did not reason from it |

### A criterion that is untestable as written — reported as a finding about the criterion

S9's done-condition includes *"poll latency unaffected while the queue is full"*. **"Unaffected" has
no threshold**, so it cannot be failed by measurement — any number satisfies a reader who wants it
to. I substituted a falsifiable form (TP-028: median `GET /state` latency with the queue saturated
must not exceed the idle median by more than a small factor, and must stay in the low milliseconds)
and it passes comfortably — 3.3 ms against 3.3 ms. Recommend the done-condition be restated with a
bound if it is meant to gate anything.

## 6. Feedback & recommendations

1. **Land S9f against these measurements, not against argument.** D-1 gives the four readings the
   `QUIESCE_S` row needs. It has been held pending a decision; the decision is now measured.
2. **Decide D-3 explicitly** — `state: null` versus an absent key. It is a one-line ruling and the
   `incomplete` precedent already says which way this plan leans.
3. **D-2 wants the smaller of its two fixes**: mapping the shutdown guard's `RuntimeError` to a
   typed `503` gives C9 something to render and closes an undeclared `(route, response)` pair,
   where adding a `5xx` row only documents it.
4. **Correct the record on D-4 rather than the code.** The fix is right; the source docstring's
   hedge was the accurate description and the coordination framing overstated it. Left as
   delivered.
5. **Testability, offered as feedback rather than as a defect.** The turn seam is genuinely easy to
   drive — `Storefront(trigger=…)` and a per-instance `_trigger` made every timing assertion in
   this pass deterministic without a single `sleep`-based race. That is why a pass of this depth was
   cheap. The one thing that is *not* observable from outside is a dead turn (S9c), and TP-027
   shows exactly what that costs a participant.
6. **S9c is the highest-value remaining sub-unit from a user's point of view.** Everything else S9
   still owes is internal; `lastTurn` is the only piece a participant can actually perceive, and
   TP-027 demonstrates the current failure mode end to end — message sent, no reply, composer
   re-enabled, nothing to distinguish it from success.
7. **Environment note for whoever runs this next.** A default `pytest` run wipes the shared
   `reference` graph. This pass ran the suite once (TP-000) and **re-seeded `reference` itself**
   with `./scripts/seed_catalog.sh`; `verify_catalog.sh` reports `RESULT: OK — 15 products`. The
   probe harness seeds fixtures **only** when it finds the catalog empty, so running it against a
   seeded `reference` leaves no residue. Prefer that ordering.

## 7. Reproducing this pass

```bash
# from falkor-chat/server, with FalkorDB up
.venv/bin/python -m pytest -q                    # TP-000
../scripts/seed_catalog.sh                       # restore `reference` after the suite wipe
.venv/bin/python <scratchpad>/probe1.py          # TP-001..005, 010
.venv/bin/python <scratchpad>/probe2.py          # TP-006..009
.venv/bin/python <scratchpad>/probe3.py          # TP-011..014, 026..028
.venv/bin/python <scratchpad>/probe4.py          # TP-011c, 012b (real HTTP status)
.venv/bin/python <scratchpad>/probe5.py          # TP-015..020 (the F8 matrix)
.venv/bin/python <scratchpad>/probe6.py          # TP-021, 023..025, 029
TURN_S=6 .venv/bin/python <scratchpad>/server_launch.py &   # TP-022, then SIGTERM mid-turn
```

The harness builds its own workspace (`ws:qa_s9`) and never writes `ws:acme`.
