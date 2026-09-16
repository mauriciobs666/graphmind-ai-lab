# Salesperson UI — Test Plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** S15 (M<n> TBD)

## 1. Scope & objective

First QA pass on the fully-assembled `salesperson/` storefront SPA + its `falkor-chat` server
surface, per `docs/plans/salesperson-ui.md` §5.1's **S15** row: every implementation unit
(S0–S14, S17, S12d) is committed (`docs/plans/salesperson-ui2-coordination.md`'s ledger). This
plan covers S15's whole deliverable set: the load/concurrency harness
(`salesperson/scripts/load_demo.py`), the live acceptance pass (§6.3), the load/concurrency
measurement (§6.4), and running the two Playwright specs S12b/S12d wrote against mocks for the
first time against a real running server.

**Objective:** gather recorded evidence for AC-1…AC-11 (`docs/requirements/salesperson-ui.md`),
per the mapping in `docs/plans/salesperson-ui.md` §10, with AC-3/AC-8/AC-9 carrying **measured**
numbers rather than assertions, and state plainly wherever AC-3's literal wording ("for any
participant") is not met — §6.4 already names the gap; this plan gathers the evidence for it,
not a fresh judgment call.

## 2. References

- `docs/requirements/salesperson-ui.md` — FR-1…FR-11, AC-1…AC-11.
- `docs/plans/salesperson-ui.md` — §4.4 (concurrency measures), §4.9 (no unauthenticated read
  path), §5.1's S15 row, §5.2 (the `/shop/api` contract), §6.3 (live acceptance), §6.4 (load/
  concurrency), §7 (risks), §10 (AC → step map).
- `docs/plans/salesperson-ui2-coordination.md` — ledger of every S15-dependency unit's acceptance
  (S11, S13, S14, S17, S12d), confirming S15's dependency set is closed.
- `falkor-chat/server/falkorchat/storefront_api.py`, `storefront.py` — the delivered `/shop/api`
  router and its `responses={}` declarations, read directly against §5.2 rather than taken on
  the plan's prose alone.
- `falkor-chat/docs/SERVER.md` §1.7 (testing hazards), §1.8 (model resolution).
- `salesperson/README.md`, `salesperson/AGENTS.md` — build/bring-up instructions, the Node
  toolchain trap.
- `falkor-chat/scripts/start_demo.sh` — the S11 bring-up script (from-cold-box demo deployment).
- `salesperson/tests/e2e/mobile-shell.spec.ts`, `salesperson/tests/e2e/presenter.spec.ts` — the
  two existing Playwright specs, written against `page.route()` mocks; this is their first run
  against a live server.
- `claude/qa-engineer/qa-testing-techniques.md` — the LM Studio `baseURL` reachability gotcha
  (re-confirmed live in this pass, §5 below).

## 3. Risk assessment (from `docs/plans/salesperson-ui.md` §7, read for this pass)

| Risk | Relevance to this pass |
|---|---|
| **R1** — the LLM endpoint (not the UI) is the FR-5/AC-3 ceiling | Central. §6.4's Run B exists to quantify this, not assert it away. This plan's TP-016 is the primary evidence-gathering item, and the report will not record a bare "pass" against AC-3's literal wording. |
| **R2** — K-060 (agent occasionally fabricates catalog facts) | Open, separate track, not a build gate for S15. Out of scope here beyond noting it if seen live. |
| **R3** — prompt adherence (language, address confirmation) | TP-005/TP-007 are the measured runs §6.3 requires for AC-8/AC-9; a low adherence rate is evidence for R3's reversal path, not a defect in this UI. |
| **R4** — reset is destructive and timing-sensitive | TP-008 (functional) and TP-015 (load) both exercise it; TP-015 runs against the **`demo`** workspace only, never `acme`/`reference`. |
| **R6** — no auth, one standing shared secret | Accepted per FR-1's scope; TP-002 verifies the *residual* claim (no unauthenticated read path into the workspace) rather than re-litigating the accepted risk. |
| **R7** — `--reload` kills in-flight work on any file write under `falkor-chat/` | Mitigated structurally by using `start_demo.sh` (non-empty `UVICORN_ARGS`, `--reload` off by default) for every live/load run in this pass. No file under `falkor-chat/` is written once the server is up. |
| **R8** — `reference` wiped by a default `pytest -q` run | Hit exactly as documented while establishing this pass's baseline (§5 below) — expected, not a defect; reseeded and re-verified before any live/load work began. |
| **R9** — publish/materialize drift | Checked via `verify_salesperson.sh demo` before this pass's live work (§5). |
| **R10** — poll load (50 × 2 routes / 2 s) | TP-014's Run A is exactly this scenario at 50 participants. |
| **R12** — join not idempotent | Accepted per the plan's own reversal-trigger note; not exercised deliberately (would require injecting a socket timeout, out of scope for this pass). |
| **R13** — `routes.tsx`'s `JoinScreen` chrome stays hardcoded English | Accepted per §4.13's own reversal trigger; TP-007's chrome check targets the namespaces S17 actually swept, not this named exception. |

## 4. Test items

Each item cites the plan section it verifies. Type: `e2e` (live, black-box, driven against the
running app), `load` (the harness), `integration` (existing suite, re-run as baseline), `manual`
(the S12b/S12d Playwright specs, now run live).

| ID | Title | Type | Priority | AC(s) |
|---|---|---|---|---|
| TP-001 | Name-only join, no login step, profile panel shows the name immediately | e2e | High | AC-1, AC-8 |
| TP-002 | Two-participant isolation (cart+transcript) + network-side host-root check | e2e | High | AC-2 |
| TP-003 | Cart add/remove/quantity-change updates lines + running total | e2e | High | AC-6 |
| TP-004 | Order lifecycle (placed → fulfilled → delivered; separate cancel) reflected in UI | e2e | High | AC-7 |
| TP-005 | Measured, n=10: order-time address confirmation adherence | e2e | High | AC-8 |
| TP-006 | Catalog shows the live 15-product electronics catalog | e2e | Medium | AC-9a |
| TP-007 | Measured, n=10/locale: per-turn language adherence (en/pt-BR/es) + SPA chrome switches too | e2e | High | AC-9b, AC-9c |
| TP-008 | Presenter reset-everyone + participant reset-mine; roster shows/clears correctly; driven from a phone viewport | e2e | High | AC-5 |
| TP-009 | Product image present/absent — both branches, no placeholder element | e2e | Medium | AC-11 |
| TP-010 | Mobile viewport (360×740, 390×844): no horizontal scroll, usable — Playwright `mobile-shell.spec.ts` live | manual | High | AC-4 |
| TP-011 | Presenter view Playwright spec (`presenter.spec.ts`) live | manual | Medium | AC-5 |
| TP-012 | Baseline: `falkor-chat/server` pytest suite green before any new work | integration | High | (gate) |
| TP-013 | Baseline: `salesperson` `tsc -b` + `vitest run` green before any new work | integration | High | (gate) |
| TP-014 | Run A — stub-LLM, 50 participants: p95 `GET /shop/api/state` < 300 ms, zero errors, zero isolation violations | load | High | AC-3 |
| TP-015 | `reset_all`-under-load queue-depth headroom check (observed peak vs. `MAX_QUEUED_QUERIES`=25) | load | High | AC-3 |
| TP-016 | Run B — live-LLM concurrency sweep (1,2,4,8,16,32,50): reply-latency curve **+ dead-turn count published beside it** | load | High | AC-3 |
| TP-017 | §4.4 measure 3's embedding-overhead delta — measured proxy (direct embedding-endpoint benchmark, since the delivered code path has no `_safe_embed` call to instrument) | load | Medium | AC-3 (context for R1) |
| TP-018 | AC-10 readiness-gate status recorded (not a build gate) | functional | Low | AC-10 |

## 5. Environment & data setup

- FalkorDB up (`redis-cli -p 6379 ping` → `PONG`), confirmed before starting.
- **Baseline established first** (TP-012/TP-013), per this agent's standing rule: never layer new
  results on an unconfirmed baseline. Running the server suite is expected to wipe `reference`
  (`falkor-chat/docs/SERVER.md` §1.7) — reseed (`seed_demo.sh demo`, `seed_catalog.sh`,
  `seed_salesperson.sh demo`) and re-verify (`verify_catalog.sh`, `verify_salesperson.sh demo`)
  before any live/load work, same as the coordination's own S10 precedent.
- SPA built via `./build.sh` (Node from `~/.local/node/current/bin`, prepended to `PATH`).
- Demo brought up via `falkor-chat/scripts/start_demo.sh` — never `start_server.sh` — so
  `UVICORN_ARGS` defaults to `--reload` **off** (R7). `FALKORCHAT_STOREFRONT_PRESENTER_KEY` is
  exported by this pass for TP-008/TP-015 (the script sets no default).
- **Live-LLM reachability, checked before trusting the shared config**
  (`claude/qa-engineer/qa-testing-techniques.md`): `~/.config/opencode/opencode.json`'s
  `provider.lmstudio.options.baseURL` is a LAN address (`192.168.0.69:1234`) that failed to
  connect (`curl`: "No route to host") from this box; `http://localhost:1234/v1/models`
  responded 200 with the exact models `falkor-chat/config/models.json` names
  (`qwen/qwen3-4b-2507`, `mistralai/ministral-3-3b`, `text-embedding-qwen3-embedding-0.6b`). Per
  the documented technique, the shared file is **not edited** — a scratch copy with `baseURL`
  corrected to `localhost` is used via `FALKORCHAT_OPENCODE_CONFIG` for every live-LLM run in
  this pass.
- **Stub-LLM mode** (TP-014/TP-015): `salesperson/scripts/stub_llm_server.py` (new, this pass),
  a zero-dependency fake OpenAI-compatible endpoint returning a fixed-delay, plain-text (no
  tool-call-shaped) completion — verified against `falkorchat/llm.py`'s parsing so the
  `salesperson` workflow's `assistant` step completes the turn as an ordinary reply rather than
  mis-parsing the canned text as an embedded tool call.
- Playwright's `chromium-headless-shell` already installed (`~/.cache/ms-playwright`) — no
  install step needed this pass.
- All load/live work targets `ws:demo` only; TP-015's `reset-all` is scoped to that workspace by
  `start_demo.sh`'s own `FALKORCHAT_WS_ID` pin (never `acme`/`reference`).

## 6. Entry / exit criteria

**Entry:** TP-012 and TP-013 both green (fresh run, not taken on report); FalkorDB reachable;
demo build succeeds; presenter key set.

**Exit:** every AC in §4's table has recorded evidence (a captured request/response, a measured
number, or a screenshot-equivalent DOM assertion from Playwright); AC-3/AC-8/AC-9 carry measured
figures; the report states plainly where AC-3's literal wording is not met; TP-015 reports the
**observed** peak queue depth against the cap; TP-016 publishes the dead-turn count beside the
latency curve, never the curve alone; any genuine defect is filed with severity, repro steps, and
evidence, not silently fixed.

## 7. Out of scope

- **S16** (docs closeout) — a separate unit, not this plan's deliverable.
- **K-060** (catalog-fact fabrication) — its own open track (R2); noted if observed live, not
  root-caused here.
- **R12** (join non-idempotency) — accepted per the plan's own reversal trigger; reproducing it
  needs an injected FalkorDB socket timeout, out of scope for a black-box pass.
- **R13** (`JoinScreen`'s hardcoded English chrome)** — accepted per §4.13's reversal trigger;
  TP-007's chrome check does not re-flag it.
- A full security review — routes to `security-expert` on request, not part of this QA pass
  beyond the accepted-risk residual check in TP-002.
- Performance tuning or code changes of any kind — this is a verification pass; any defect found
  is documented, not fixed, per this agent's standing guardrail.
