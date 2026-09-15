# Salesperson UI — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** S15 (M<n> TBD)

## Summary

**What was tested:** the fully-assembled `salesperson/` storefront SPA against its `falkor-chat`
server surface (`/shop/api`), commit `993e8b2` (all of S0–S14, S17, S12d committed per
`docs/plans/salesperson-ui2-coordination.md`'s ledger). Covers every item in
`docs/test-plans/salesperson-ui.md`: baselines, the new load harness
(`salesperson/scripts/load_demo.py`), the live acceptance pass (§6.3), the load/concurrency
measurement (§6.4), and the first-ever live run of the two S12b/S12d Playwright specs.

**Overall verdict: CONDITIONAL PASS, with four genuine, reproducible product-level defects found**
(DEF-1, DEF-2, DEF-3, DEF-6) plus two low-severity test-only defects (DEF-4, DEF-5) — see Defects
below. AC-1, AC-2, AC-4, AC-6, AC-7, AC-8, AC-9a, AC-10, AC-11 are met.
AC-3 is met for read paths and **not met as literally worded** for agent-reply latency under
load, exactly as `docs/plans/salesperson-ui.md` §6.4/§10 anticipated — measured, not asserted.
AC-5 is functionally correct but **its primary access path (a direct link to `/shop/presenter`)
is broken** (DEF-1) — a bookmark, QR code, or browser refresh 404s; the presenter view itself
works once reached (verified via a client-side-navigation workaround). AC-9's chrome-switch half
(AC-9c) is confirmed live. A load-only defect (DEF-3) shows the platform's own dead-turn signal
does not fire for the failure class this pass's heavier concurrent runs actually triggered.

**CPG:** considered, not relevant — `cpg_salesperson` does not exist in this FalkorDB instance
(confirmed, per the coordination brief and independently by `GRAPH.LIST`); `cpg_falkorchat`
exists but this is a behavior-altitude acceptance/load pass driving the running system end to
end, not a structural/test-gap analysis of the server's Python source, so the CPG's call-graph
answers are not the applicable tool here.

## Environment

- FalkorDB `v4.18.x`-class instance, confirmed reachable throughout.
- Server: `falkor-chat/scripts/start_demo.sh` (never `start_server.sh` — `--reload` stays off,
  R7), `ws:demo`, built SPA at `salesperson/dist`.
- Live-LLM backend: LM Studio, reached at `http://localhost:1234` (the shared
  `~/.config/opencode/opencode.json`'s configured `192.168.0.69:1234` is unreachable from this
  box — "No route to host" — a scratch copy with `baseURL` corrected to `localhost` was used via
  `FALKORCHAT_OPENCODE_CONFIG`, never editing the shared file; consistent with the documented
  technique in `claude/qa-engineer/qa-testing-techniques.md`). Models actually loaded and used:
  `mistralai/ministral-3-3b` (agent), `qwen/qwen3-4b-2507` (seen once as a load target during a
  concurrent-model-swap failure — see DEF-3 evidence), `text-embedding-qwen3-embedding-0.6b`.
- Stub-LLM backend (Run A only): `salesperson/scripts/stub_llm_server.py` (new), a fixed-delay,
  plain-text, zero-dependency fake OpenAI-compatible endpoint.

## Baseline (TP-012, TP-013)

Both green, fresh-run, before any new work — per this agent's standing rule.

- **TP-012 — PASS.** `falkor-chat/server`, `.venv/bin/python -m pytest -q`: **2821 passed, 14
  deselected, 0 failed.** This run wiped `reference` exactly as `falkor-chat/docs/SERVER.md` §1.7
  documents (R8) — reseeded (`seed_demo.sh demo`, `seed_catalog.sh`, `seed_salesperson.sh demo`)
  and re-verified (`verify_catalog.sh` → OK 15/15; `verify_salesperson.sh demo` → both defs in
  sync) before any live/load work began.
- **TP-013 — PASS.** `salesperson`: `npm run typecheck` (`tsc -b`) clean; `npm test`
  (`vitest run`): **243 passed (26 files)**.

## Results table

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-001 | Name-only join, profile shows immediately | **PASS** | `POST /shop/api/session` → `GET /shop/api/state` (no message sent) already returns `profile.name` set; Playwright: profile sheet shows "Live QA Alice" immediately after join |
| TP-002 | Two-participant isolation + network-side host-root check | **PASS** | Load harness: 0/50 isolation violations (profile-name cross-check on every `/state` poll); Playwright: Bob's cart/profile show 0 mentions of Alice; `curl` confirms `/`, `/channels`, `/search`, `/mcp` all `404` (dev_surface off), `/openapi.json` `200` (documented R6 exemption) |
| TP-003 | Cart add/remove/quantity → correct running total | **PASS** (contract defect found, DEF-2) | Corrected Playwright run: 2 adds → "Cart total: $79.98" both in the chat transcript and the aggregate cart-panel total; **the per-line price renders `$NaN`** (DEF-2) |
| TP-004 | Order lifecycle (placed → fulfilled → delivered; separate cancel) | **PASS for placement/address**; lifecycle-advance not independently re-verified this pass — see Coverage & gaps | `POST /shop/api/messages` flow produced a real order (`Order ID: d2d5a9bbea1d49a1a8cee6914f597bf7`, correct 2-line total); compound "add + place + address in one message" reliably confused the 3B model (observation, not a UI defect — see Coverage & gaps) |
| TP-005 | Measured, n=10: order-time address adherence | **PASS — 5/5 (100%) asked for the address**, sequential/uncontended (reduced from n=10 to n=5 for this clean control — see Measured figures for why, and the n=10 concurrent variant alongside it) | see Measured figures below |
| TP-006 | Live 15-product catalog | **PASS** | `GET /shop/api/catalog` → 15 rows; Playwright catalog sheet: 15 priced entries, 14 `<img>` elements |
| TP-007 | Measured, n=10/locale: language adherence + chrome switch | **PASS for chrome (AC-9c); PASS for pt-BR/es content (100% both variants); a genuine, reproducible `en`-specific gap found at every concurrency tried — see DEF-6/Measured figures** | Playwright: pt-BR join screen shows "Bem-vindo à loja, Live QA Carlos" + composer placeholder "Digite uma mensagem…"; measured adherence rates below |
| TP-008 | Presenter reset-everyone + reset-mine; roster; from a phone | **PASS (functionally), reached only via a workaround — see DEF-1** | Roster showed all 4 live participants (name+language, no activity data) before reset; "Reset complete — 4 participants cleared" after; roster then empty; Alice's own screen reverted to the join screen (her token invalidated) — all at 390×844 |
| TP-009 | Image present/absent, no placeholder element | **PASS** | `GET /shop/api/catalog`: 14/15 have `imageUrl`, `smart-home-hub` is `null` (matches `salesperson/README.md`'s documented deliberate exception); `CatalogPanel.tsx` confirmed to render `<img>` only when non-null, no placeholder branch |
| TP-010 | Mobile viewport, no horizontal scroll | **PASS** | `mobile-shell.spec.ts` live run: the two viewport-integrity tests pass at both 360×740 and 390×844; one unrelated test in the same file is stale (DEF-5) |
| TP-011 | Presenter Playwright spec live | **FAIL as written (DEF-4), underlying view confirmed working via workaround** | `presenter.spec.ts`'s own `page.goto('/presenter')` never reaches the app (see DEF-1/DEF-4) |
| TP-014 | Run A — stub-LLM, 50 participants | **PASS** | see Measured figures |
| TP-015 | `reset_all`-under-load queue-depth headroom | **PASS** | see Measured figures |
| TP-016 | Run B — live-LLM concurrency sweep | **PASS (AC-3 met for reads, not met as literally worded for agent turns — per plan's own recording rule)** | see Measured figures |
| TP-017 | Embedding-overhead delta (measured proxy) | **PASS (informational)** | see Measured figures |
| TP-018 | AC-10 readiness gate status | **PASS** | `falkor-chat/docs/HISTORY.md` 2026-08-30 entry: K-056 resolved (model swap); `falkor-chat/docs/BACKLOG.md`: K-060 still 🟡 in-progress, its own track — AC-10's condition ("first real demo gated on K-056") is satisfied; K-060 remains an accepted, separately-tracked residual risk (R2) |

## Defects

### DEF-1 — HIGH — `GET /shop/presenter` (and any deep SPA route) 404s on direct navigation; the presenter's primary access path is broken

**Where:** `falkor-chat/server/falkorchat/app.py` (`/shop` mount: plain
`StaticFiles(directory=served_dir, html=True)`, no SPA-fallback route) vs.
`salesperson/src/routes.tsx` (`createBrowserRouter(..., { basename: '/shop' })` — HTML5-history
routing, which requires exactly such a fallback for any path other than the served directory
root).

**Steps to reproduce:**
1. Bring up the demo (`start_demo.sh`).
2. `curl -i http://<host>:8000/shop/presenter` (or open it in a browser — a fresh tab, a
   bookmark, a QR code, or a page refresh while already on that route).

**Expected:** the SPA loads and client-side-routes to the presenter key screen (this is literally
what `presenter.spec.ts`'s own `page.goto('/presenter')` assumes, and what a presenter following
a shared link/QR code needs).

**Actual:** `404 {"detail":"Not Found"}` (FastAPI's own not-found handler — the request never
reaches the `/shop` static mount's own file resolution, and there is no `404.html`/fallback
`index.html` behavior configured). Confirmed via `curl` directly and via Playwright (a literal
`page.goto('http://host:8000/shop/presenter')` never renders the app).

**Root cause confirmed live:** the SPA's rendering logic itself is fine — loading `/shop/` first
and then simulating in-app navigation (`history.pushState('/shop/presenter')` +
a `popstate` event, bypassing the second HTTP request) renders `PresenterKeyScreen` correctly.
The defect is purely the missing server-side fallback route for HTML5-history client-side
routing under `/shop`.

**Impact:** `docs/plans/salesperson-ui.md`'s own framing of AC-5 ("reachable from a phone... which
the key-based presenter control supports") assumes the presenter reaches `/shop/presenter`
directly, e.g. via a shared link or QR code — exactly the workflow this breaks. There is no
in-app UI element that navigates there client-side (`routePaths.ts`'s `presenter: '/presenter'`
is reached only by a full navigation in real use), so as delivered there is **no working way for
a presenter to open their own view** other than a browser/devtools trick.

**Recommendation (not implemented — routing a fix is outside this pass's remit):** register a
`GET` fallback route under the `/shop` mount (matched before or instead of the bare
`StaticFiles` mount) that serves `served_dir/index.html` for any request that isn't a real static
asset and isn't `/shop/api/*` — the standard SPA-deployment pattern. `StaticFiles(html=True)`
alone does not provide this.

### DEF-2 — HIGH — Cart panel renders `$NaN` per line item (contract drift between server and client on the cart-item price field)

**Where:** `salesperson/src/views/CartPanel.tsx:61` (`formatCurrency(item.unitPrice *
item.quantity, locale)`) vs. `falkor-chat/server/falkorchat/services.py`'s `_priced_cart_lines`/
`get_cart` (the actual `/shop/api/state` `cart.items[]` shape carries `price`, not `unitPrice` —
confirmed by reading the delivered server code directly). `salesperson/src/api/endpoints.ts`'s
`CartItem` interface also declares `unitPrice: number`, matching the client's wrong assumption,
not the server's actual field.

**Steps to reproduce:**
1. Join, ask the agent to add any product to the cart.
2. Open the cart sheet.

**Expected:** the line shows the item's price (e.g. `$29.99`).

**Actual:** the line shows `$NaN` (confirmed live, Playwright DOM read: `"1× Wireless Mouse
Pro\n$NaN\nTotal\n$29.99"`). The aggregate **Total** at the bottom is correct (it reads
`cart.total`, a separate, correctly-named field), so only the per-line prices are broken — but
every cart with any items shows this on first open, in a business-facing demo.

**Why unit tests didn't catch it:** `CartPanel.test.tsx`'s own fixtures hand-write
`unitPrice: 29.99` in their mock `cart.items` — matching the (wrong) TypeScript interface, not
the real server response shape — so the test is green while the real integration is broken. This
is exactly the class of defect a mocked unit/contract test cannot catch and only driving the real
server surfaces (this pass's whole premise).

**Recommendation (not implemented):** rename one side to match the other (`CartPanel.tsx`/
`CartItem` → `item.price`, or the server → emit `unitPrice` on `get_cart` for symmetry with
`place_order`'s `order_lines`, which already uses `unitPrice`); either way, fix the test fixture
to be sourced from (or checked against) the real response shape, not hand-typed to match the
interface under test.

### DEF-3 — HIGH — The dead-turn latch (`turn.lastTurn`) does not fire for the platform's own internally-handled provider failures — confirmed via 230+ silently-failed runs generated by this pass's own load

**Where:** `falkor-chat/server/falkorchat/storefront.py`'s `_run_turn` sets `lastTurn` (via
`_mark_turn_failed`) **only** in its own `except Exception` block — i.e., only when an exception
escapes all the way up through `trigger.maybe_trigger`. But
`falkor-chat/server/falkorchat/services.py`'s `_drive_or_fault` (the executor's own top-level
fault net) **catches** `ProviderCallError`/`ModelResolutionError`/`WorkflowConfigError`/
`NotImplementedError` internally, logs it ("workflow drive fault for run %s"), marks the
`WorkflowRun` `failed` in the graph, and **returns normally** rather than re-raising (confirmed
reading `services.py` directly) — so for this whole common failure class, `maybe_trigger` never
raises, `_run_turn`'s `except` never fires, and `lastTurn` is never set.

**Steps to reproduce (as it happened live in this pass):**
1. Run enough concurrent tool-calling conversations to make LM Studio genuinely fail some
   completions (this pass's heavier n=30/n=10-concurrent language/address-adherence checks did
   this; the lighter, single-turn-only Run B sweep at the same 50-concurrency did **not** — see
   the nuance below).
2. Poll `GET /shop/api/state` for an affected participant.

**Expected (per `docs/plans/salesperson-ui.md` §5.2 "The dead-turn signal"):** `turn.state:
'idle'`, `turn.lastTurn: 'failed'` — "without this field the participant sees their message, no
reply, and a composer that quietly re-enables, which is indistinguishable from a completed turn."

**Actual:** `turn.state: 'idle'`, `turn.lastTurn: null` — no reply ever posted, no failure signal.
Reproduced live: participant `LifecycleA` posted 3 consecutive messages, got **zero** assistant
replies across all 3, and `lastTurn` stayed `null` throughout.

**Root-caused and quantified via direct graph evidence** (`ws:demo`, read-only queries):
`MATCH (r:WorkflowRun) RETURN r.status, count(r)` → **230 `failed`, 131 `waiting`, 3 `running`**
at the time of this check. Sampling `r.ctx` on the failed rows: **100% carry a `ProviderCallError`**
(HTTP 500 from LM Studio, `{"error":"terminated"}` HTTP 400, and one `qwen/qwen3-4b-2507`
model-load contention failure — all genuine provider-side failures under this pass's own
concurrent load, not fabricated). None of these 230 would have set `lastTurn`, by the code path
traced above.

**Important nuance, stated precisely so the load-test numbers above aren't misread:** Run B
(TP-016, single-turn "Hi there!" conversations, no tool calls) reported **zero** dead turns and
**zero** errors at every concurrency level including 50 — and this is independently corroborated
at the graph level: every one of Run B's 113 `WorkflowRun`s (matching its own
`1+2+4+8+16+32+50=113` participant count exactly, by displayName) shows `status: 'waiting'`
(normal, successful single-step completion), **zero** `failed`. So TP-016's own published curve
is not contaminated by this gap — the gap is real and proven, but it only manifests under the
heavier, tool-calling-inclusive concurrent load this pass's AC-8/AC-9 measured runs happened to
also generate, not under Run B's own simpler scripted turns. **This is exactly why the harness's
own `dead_turns_seen` counter must never be read as proof of zero failures** — it is a measurement
of the platform's own signal, and that signal has a proven gap; it is accurate here only because
independently verified against the graph.

**Impact:** under real demo conditions with ~50 participants actually shopping (not just
chatting), a meaningful fraction of turns can die with zero visible feedback — the exact
"indistinguishable from a completed turn" failure mode this field was built to prevent, occurring
because this particular (and, under this pass's tool-calling load, dominant) failure path bypasses
it entirely. S9's own unit-level done-condition ("a stub trigger that raises leaves
turn:{state:'idle', lastTurn:'failed'}") only exercises the narrower, already-latched path — this
gap was invisible to that test by construction.

**Recommendation (not implemented):** either (a) widen `_run_turn`'s failure detection to also
inspect the `WorkflowRun`'s own terminal status after `maybe_trigger` returns (re-reading the
graph, mirroring what `_drive_or_fault` itself already does for its "fault escaped" branch), or
(b) have `_drive_or_fault` re-raise (or return a distinguishable failure signal) for the
chat-triggered path specifically, so the storefront's turn-isolation layer can observe it. Either
way, this needs a decision from whoever owns `executor.py`/`storefront.py`'s failure-isolation
contract, not a QA-authored patch.

### DEF-6 — MEDIUM (measured, attribution not fully established) — English-configured participants sometimes receive a fully-formed Spanish reply, reproduced at both low and high concurrency, never observed for pt-BR/es or under fully sequential (concurrency=1) use

**Steps to reproduce:** join an `en`-language participant concurrently with at least one other
(any-language) participant's conversation, and send a scripted 5-turn conversation. Reproduced
twice independently:
- **Heavy variant** (40-way concurrent, this pass's AC-8+AC-9 combined load): 5/10 `en`
  conversations affected (turn 1 onward).
- **Literal-plan variant** (10 trials, only 3-way concurrent per trial — "three participants...
  simultaneously," per §6.3 item 7's own wording): **2/10 trials affected** (`LitAdhere-7-en`,
  `LitAdhere-9-en` — turns 0-2 of 5 each came back in Spanish, e.g. *"¡Hola LitAdhere-7-en! ¿En
  qué puedo ayudarte hoy con tus compras?..."*, before self-correcting to English on turns 3-4 of
  the same conversation).
- **Never observed** in any fully sequential (concurrency=1) run this pass performed
  (`DiagPT`, `LifecycleB`, `QA Cart Check`, the 5-participant sequential AC-8 control all got
  correct-language replies throughout).

**Investigated, not conclusively root-caused:** the server code was read directly
(`falkorchat/executor.py`) — `run_ctx` is loaded fresh from the graph per `run_id`/`run` row, no
shared mutable state was found that could explain a cross-participant language mix-up at the
application layer. The leading hypothesis is a concurrent-request attribution issue at the LM
Studio serving layer (prompt/response mixing under concurrent multi-language chat-completion
calls) rather than a `falkor-chat`/`salesperson` code defect, but this pass did not have the
tooling to confirm that attribution against LM Studio's own internals — stated as a hypothesis,
not a finding, for that reason.

**Impact:** this is a real, reproducible, concurrency-triggered failure of AC-9b's core claim
("customer-facing copy appears in the language they chose") for the **default** language, at a
measured ~20% (low concurrency) to ~50% (high concurrency) conversation-level rate. Distinct from
DEF-3: the replies here are fully-formed and coherent, just in the wrong language — not empty,
not latched as failed.

**Recommendation (not implemented):** worth a `data-scientist`/`graph-dba`-informed follow-up
specifically aimed at LM Studio's concurrent-request handling (does it exhibit prompt-prefix
mixing under concurrent load with different system prompts/CONTEXT blocks?) before trusting this
demo with a live, multi-language concurrent audience.

### DEF-4 — LOW (test-only) — `presenter.spec.ts` navigates to the wrong URL, independent of DEF-1

**Where:** `salesperson/tests/e2e/presenter.spec.ts:40` (and its other `page.goto('/presenter')`
calls) vs. `salesperson/playwright.config.ts`'s `baseURL: 'http://127.0.0.1:8000/shop/'`.

**Root cause:** WHATWG URL resolution treats a leading `/` as **absolute**, discarding the
base's own path segment — `new URL('/presenter', 'http://host:8000/shop/')` resolves to
`http://host:8000/presenter`, not `http://host:8000/shop/presenter`. Confirmed directly
(`python3 -c "from urllib.parse import urljoin; print(urljoin(...))"`) and by the live Playwright
run's own failures ("element(s) not found" for `Presenter key`, `0` roster rows).

**Impact:** even once DEF-1 is fixed, this spec still hits the wrong path (the *root* mount,
which is genuinely unmounted — `dev_surface=False` — so it would keep failing regardless).
`mobile-shell.spec.ts`'s own navigation (`page.goto('')`, relative) does not have this bug.

**Recommendation (not implemented — it's not this pass's test file to fix):** use a relative path
(`page.goto('presenter')`) so it joins onto `baseURL` correctly.

### DEF-5 — LOW (test-only) — `mobile-shell.spec.ts`'s "seed placeholders" test is stale

**Where:** `salesperson/tests/e2e/mobile-shell.spec.ts:41-57` asserts `/catalog will appear
here/i`, `/cart will appear here/i`, `/order status will appear here/i`, `/profile will appear
here/i` — S12b-era seed-placeholder copy that **S14 replaced with real panel content**
(`CartPanel.tsx`/`OrderPanel.tsx`/`ProfilePanel.tsx`/`CatalogPanel.tsx`, all committed). The
equivalent defect in the *unit*-test tier (`Shell.test.tsx`) was already caught and fixed by
`S12b-testfix` (per the coordination ledger) — this e2e-tier sibling was not, because this is the
first time the e2e suite has ever actually run (mocked or live) since S14 landed.

**Impact:** none on the product — the real panels render correctly (confirmed elsewhere in this
report, TP-003/TP-006/TP-009). This is pure test debt.

**Recommendation (not implemented):** update the assertions to real S14 content (e.g., an empty
cart's actual copy) or delete the test if its "wired into their own sheets" claim is now covered
by the panel-specific unit tests.

## Coverage & gaps

- **AC-7's lifecycle-advance transitions (fulfilled → delivered, separate cancel) were not
  independently re-verified this pass via `POST /shop/api/order/advance`** — every attempt to get
  a clean order through the live agent hit either the compound-instruction confusion noted above
  or, once, DEF-3's silent-failure gap, and time/token budget went to root-causing DEF-3 instead
  of re-attempting a third time. `advance_order`'s contract is already covered by S8/S10's own
  delivered `TestClient` contract tests (§6.2) and S14's own component tests — this is a residual
  gap in *this pass's* live/black-box coverage specifically, not an unverified feature.
- **The compound-instruction observation** ("add this AND place my order AND here's my address,"
  all in one message, repeatedly failed to actually place the order) is noted but not filed as a
  defect — it is consistent with the already-tracked K-057/K-060 small-model reliability family
  (R2), not a `salesperson`/`falkor-chat` code defect; a business-facing demo script should avoid
  compound asks for this reason.
- **DEF-6's root cause (LM Studio's own concurrent-request handling vs. an application-level
  isolation bug) was not conclusively established** — the server-side `run_ctx` code was read and
  cleared of a shared-mutable-state explanation, but confirming the alternative (LM Studio
  internals) was beyond this pass's tooling. Recorded as measured evidence with a stated
  hypothesis, not a proven root cause.
- **S16 (docs closeout)** is out of scope, per the test plan.
- **A full security review** was not performed beyond TP-002's residual-risk check; out of scope
  per the test plan.

## Measured figures (load harness, `salesperson/scripts/load_demo.py`)

### Run A — stub-LLM, 50 participants, 5 scripted turns each (TP-014)

```
participants_joined: 50/50, turns_completed: 250/250, errors: 0, isolation_violations: 0
turn_in_progress_409_count: 0, dead_turns_seen: 0
state_poll p50/p95/p99/max (ms):        14.8 / 144.7 / 176.8 / 182.6
state_poll (incl. steady-poll) p95 (ms): 72.7
agent_turn p50/p95/p99/max (ms):        4030.5 / 4219.9 / 6057.0 / 6062.3
write p50/p95/p99/max (ms):             35.5 / 166.2 / 203.3 / 216.5
```

**Target ("p95 `GET /shop/api/state` < 300 ms, zero errors, zero isolation violations") — MET**:
p95 144.7 ms (72.7 ms including the post-conversation steady-poll phase), 0 errors, 0 violations.

### TP-015 — `reset_all`-under-load queue-depth headroom (during Run A, 61 participants incl.
pre-existing dev artifacts)

```
reset_status: 200, clearedParticipants: 61, reset_wall_ms: 68.9
peak_waiting_queries observed: 14   (cap, MAX_QUEUED_QUERIES: 25)
num_samples: 124 (≈28 ms sampling interval via `redis-cli GRAPH.INFO`, subprocess-shelled)
```

**Against the plan's own estimate ("~18 queued, under the cap, but not by much"): the observed
peak (14) is under both the plan's own estimate and the cap, with 11 queries of headroom.**
Sampling resolution caveat: at ~28 ms/sample against a 68.9 ms reset window, the true
instantaneous peak could exceed 14 by a small margin the sampling could miss — stated plainly
rather than overclaiming precision that a purely software polling loop cannot deliver.

### Run B — live-LLM concurrency sweep, 1 turn/participant (TP-016)

| Concurrency | n | turn p50 (ms) | turn p95 (ms) | turn max (ms) | dead turns | timeouts |
|---|---|---|---|---|---|---|
| 1 | 1 | 4025.9 | 4025.9 | 4025.9 | 0 | 0 |
| 2 | 2 | 3015.3 | 3922.2 | 4023.0 | 0 | 0 |
| 4 | 4 | 6028.4 | 8045.3 | 8046.5 | 0 | 0 |
| 8 | 8 | 3031.8 | 6050.0 | 6050.2 | 0 | 0 |
| 16 | 16 | 7293.3 | 13327.4 | 13330.5 | 0 | 0 |
| 32 | 32 | 12094.0 | 21124.4 | 22279.4 | 0 | 0 |
| 50 | 50 | 18115.0 | 30340.9 | 32355.3 | 0 | 0 |

`state_poll` p95 stayed ≤ 48.4 ms at every concurrency level, including 50.

**Dead-turn count published beside the curve, as required:** **0 at every level** — and, unlike a
bare self-report, this is independently corroborated against the graph (all 113 of this sweep's
`WorkflowRun`s show `status: 'waiting'`, zero `failed`; see DEF-3's nuance paragraph). The curve
is genuinely uncontaminated by DEF-3's gap.

**AC-3 recording, per §6.4/§10's own rule:** **met for all read paths at every concurrency up to
50** (state-poll p95 stays two orders of magnitude below the agent-turn latency); **not met as
literally worded ("for any participant") for agent-reply latency**, which grows from ~4 s
uncontended to a **p50 of 18.1 s / p95 of 30.3 s / max 32.4 s at 50 concurrent participants** —
consistent with R1's prediction that one local LM Studio instance on modest hardware is the
ceiling, not the UI or the storefront server.

### TP-017 — §4.4 measure 3's embedding-overhead delta (measured proxy)

The delivered storefront post path makes **no** embedding call at all (measure 3's whole point),
so there is nothing in-product to instrument directly. Measured instead: the embedding endpoint
LM Studio actually serves (`text-embedding-qwen3-embedding-0.6b`), directly, warm (post
first-call model-load):

```
concurrency=1  (sequential): ~22-52 ms/call, p50 ≈ 38 ms
concurrency=16 (concurrent): p50 ≈ 303 ms, p95 ≈ 457 ms
```

**Interpretation:** even the smaller, cheaper embedding call is not free under concurrency — at
16-way concurrent load it costs ~300-450 ms per call, contending for the same GPU every chat
completion also needs. This is a measured lower bound on what §4.4 measure 3 avoids per posted
message; it is small next to a full chat-completion turn's multi-second cost (Run B above), which
is consistent with the plan's own corrected sizing ("roughly 1 of 2-9 endpoint calls... a
considerably smaller fraction of GPU-seconds") while still confirming the removed call was
genuinely contended, not free.

### Measured adherence runs (TP-005, TP-007)

Two variants were run, disclosed as a deliberate test-design choice: the plan's literal wording
("three participants... simultaneously," implying low, realistic concurrency) as a clean
baseline, and a heavier all-at-once variant matching this demo's actual ~50-participant target —
because the heavier variant is what actually surfaced DEF-3 and the language anomaly below, and a
clean-baseline-only run would have hidden both.

**Heavy variant — n=10/locale, all 30 conversations + AC-8's 10 concurrently (40-way concurrent,
5 turns each for AC-9, 2 for AC-8):**

```
AC-9b adherence (heuristic language classifier, per-turn):
  en:    25/50 turns adherent (50%)  — a mix of DEF-6 (5/10 conversations got a fully-formed
                                        Spanish reply) and DEF-3 (silently-failed turns re-reading
                                        a stale prior reply)
  pt-BR: 50/50 turns adherent (100%)
  es:    50/50 turns adherent (100%)
AC-8: 0/10 "asked for address" — confounded by DEF-3 (see below), not a clean read
```

Cross-checked against DEF-3 and DEF-6 (both independently confirmed below): the heavy variant's
`en` shortfall and AC-8's 0/10 are **not** one fresh defect per failed conversation — they are
this same load window's DEF-3 (silently-failed turns, 230 graph-confirmed) and DEF-6 (genuine
language swap, confirmed to reproduce even at low concurrency) surfacing together, not a third,
separate mechanism.

**Literal-concurrency variant — n=10 trials of 3-simultaneous (en/pt-BR/es) for AC-9b, n=5
sequential (uncontended) for AC-8 — the plan's own literal wording ("three participants...
simultaneously"):**

```
AC-9b adherence (150 turns total, 10 trials × 3 locales × 5 turns):
  en:    39/50 turns adherent (78%)  — 2/10 trials affected: 6 turns lost to DEF-6 (language
                                        swap to Spanish, trials 7 & 9, turns 0-2 of 5 each,
                                        self-corrected to English by turn 3-4), 5 turns lost to
                                        DEF-3 (trial 8: all 5 turns came back completely empty,
                                        `lastTurn` never latched)
  pt-BR: 50/50 turns adherent (100%)
  es:    50/50 turns adherent (100%)

AC-8 (n=5, fully sequential/uncontended): 5/5 (100%) asked for the address before/at placement —
  a clean, unconfounded read of the literal AC-8 question, distinct from the heavy variant's
  0/10 figure above, which is now understood to be a DEF-3 measurement artifact, not a genuine
  prompt-adherence failure (see DEF-3's write-up).
```

**Reading these two variants together:** AC-8's literal design intent (does the assistant ask for
the address at order time) is **cleanly met** — the heavy variant's 0/10 was an artifact of
concurrent load triggering DEF-3, not a real adherence failure, and re-running at concurrency=1
proves it. AC-9b's `pt-BR`/`es` results are clean and identical across both variants (100%).
AC-9b's `en` result is the one that does **not** wash out at lower concurrency — DEF-6 reproduced
in 2 of 10 trials even at the plan's own literal 3-way-concurrent wording, so this is recorded as
a real, if partial, measured gap against AC-9b, not merely a load-testing artifact.

## Feedback & recommendations

1. **DEF-3 is the standout finding of this pass and worth prioritizing**: a load test with only
   simple, non-tool-calling scripted turns (as Run B's own design uses, matching the plan's own
   wording) will never surface it — it took the heavier, tool-calling-inclusive adherence runs to
   trigger real provider failures and expose the latch gap. Recommend adding a genuine
   provider-failure-injection unit test at the `services._drive_or_fault` → `Storefront._run_turn`
   seam (a stub that raises `ProviderCallError` *from inside* `_drive_or_fault`'s own catch,
   rather than from `trigger.maybe_trigger` directly) as a permanent regression guard — S9's
   existing "stub trigger that raises" test does not exercise this path.
2. **A CartPanel-shaped contract test would have caught DEF-2**: assert the client's `CartItem`/
   `OrderLine` TypeScript interfaces against the server's actual documented response shape (§5.2),
   not against hand-typed fixtures that can silently drift from a card the client-side interface
   also drifted from. The plan's own S12a contract-test bullet (§6.2) covers credential/session
   rules (C1-C14) but not payload *field-name* fidelity — worth a named rule if this pattern
   repeats.
3. **DEF-1 should route to whoever owns `app.py`'s `/shop` mount** — it is a one-route,
   well-understood fix (a standard SPA-fallback route), but it is production code and out of this
   pass's remit to patch.
4. **`presenter.spec.ts`/`mobile-shell.spec.ts`** (DEF-4/DEF-5) are both cheap, narrow test-only
   fixes; recommend a short follow-up unit rather than folding them into a future feature unit's
   scope.
5. **DEF-6 needs a decision before this demo runs live with a real multi-language audience** —
   the 3B model's English output is the one that degrades under concurrency, in the language
   most of a business audience is likely to pick. Recommend `data-scientist` and/or `graph-dba`
   look at LM Studio's concurrent-request behavior specifically (not a `salesperson`/`falkor-chat`
   code question) before the first live demo, alongside K-060's own track (R2).
6. **Testability note**: this was the *first* time either Playwright spec ran against anything
   other than its own mocks, and the first time cart/order/presenter flows were driven against a
   real, concurrently-loaded server — both DEF-1 and DEF-2 are exactly the class of defect that
   only manifests at that seam. Recommend this component's CI (however it eventually gets one)
   run the e2e suite against a real `start_demo.sh`-launched server, not only `npm test`'s mocked
   tier.
