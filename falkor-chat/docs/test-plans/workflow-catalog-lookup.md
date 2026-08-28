# `workflow-catalog-lookup` — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-052 (M6)

## 1. Scope & objective

Live acceptance pass for K-052 (structured catalog/reference lookup for workflows) — the first
QA gate on newly-implemented code. Nothing in this feature has been driven against a running
server yet; both prior gates (`analyst`'s implementation review and the offline `pytest`/
`test_queries.sh` suites it re-ran) are static or fixture-level. This plan exercises the real
`salesperson@v1` demo agent, a real local LLM (LM Studio, `qwen/qwen3-4b-2507`), and the real
FalkorDB `reference`/`ws:<id>` graphs, against **AC-1..AC-5** of
`falkor-chat/docs/requirements/workflow-catalog-lookup.md`.

## 2. References

- Requirements: `falkor-chat/docs/requirements/workflow-catalog-lookup.md` — FR-1..FR-7,
  AC-1..AC-5 (the acceptance bar for this pass).
- Gated plan: `falkor-chat/docs/plans/workflow-catalog-lookup.md` v1 (`architect`, approved) —
  schema (§3.1), seed path (§3.2), `SALESPERSON_DEF` scaffold (§3.3), tool design (§3.5).
- Implementation review: `falkor-chat/docs/reviews/workflow-catalog-lookup-impl.md`
  (`analyst`, **approve with suggestions**, 2026-08-28) — re-ran the full offline suite (1811
  passed / 4 deselected) and `./scripts/test_queries.sh` (346/346) on this exact commit,
  restored `ws:acme`'s baseline afterward. 2 MINOR findings (one fixed by `teco` directly — the
  `QUERIES.md` header count; one open, non-blocking — `filter_products`'s `categoryNormalized`
  NULL-coalesce edge, see §3 below), 1 NIT. No blocker.
- Canonical Cypher: `falkor-chat/docs/QUERIES.md` §15 (`lookup_product`/`filter_products`).
- Code under test: `server/falkorchat/repository.py` (`lookup_product`/`filter_products`),
  `services.py` (same names), `tools.py` (`LookupProductFactTool`/`FilterProductsTool`),
  `proof_defs.py` (`SALESPERSON_DEF` v1), `scripts/{seed,verify}_catalog.sh`,
  `scripts/{seed,verify}_salesperson.sh`.
- Precedent for live-driving a `conversation`-kind def via REST + `@mention`:
  `falkor-chat/docs/test-plans/mention-reply-delivery.md` /
  `-report.md` (same request shapes, same graph-ground-truth technique).

**CPG:** considered, not relevant — per the coordination brief, `cpg_falkorchat` is stale
(built 2026-08-26T22:27:22Z, at least 2 commits behind) and this is acceptance testing of
new, not-yet-shipped code, not a structural-impact-analysis question. Read `tools.py`/
`services.py`/`repository.py`/`proof_defs.py` directly instead.

## 3. Risk assessment

**What's already covered, and not re-tested here:** unit/integration correctness of
`Repository.lookup_product`/`filter_products` against a fixture reference graph, `Tool.run`
dispatch against a fake `Services`, `_validate_def_spec` accepting `SALESPERSON_DEF`, the
republish-is-a-no-op property, and the `ctx.endConversation` guard-safety regression test —
all already exist in `server/tests/test_{repository,services,tools,salesperson_scaffold}.py`
and were independently re-run and mutation-tested by `analyst` (U16). Re-deriving those here
would be duplicate unit-layer coverage this agent's operating rules explicitly say to extend
past, not re-litigate.

**What genuinely hasn't been checked yet, and is this pass's real risk surface:**
- **Does the real LLM's argument extraction actually work against the real tool schemas?** The
  unit tests drive `Tool.run` directly with hand-built `arguments` dicts — they cannot prove a
  real model, given a customer's natural-language question, calls `lookup_product_fact`/
  `filter_products` with the right argument values at all. This is exactly the class of bug
  `analyst`'s own U15-finding caught (`filter_products`'s category case-sensitivity, only
  surfaced by a live model lowercasing "audio") — the fix for that is now in the code under
  test, but this is the **first** live confirmation the fix actually closes the gap it was
  built for, not a second static read of it.
- **AC-4 (phrasing tolerance) is inherently a live-model question** — the plan's own test
  strategy table (§5) says so explicitly: "this is inherently about the LLM's own argument
  extraction, not the DB layer." No unit test can substitute for this.
- **The scaffold's single safety-critical property** (`ctx.endConversation` never fires, so the
  demo keeps parking for the next customer turn instead of silently ending) was verified by
  `analyst` via **mutation testing against the unit test**, not by watching a real multi-turn
  conversation actually stay parked. A live multi-turn run is a different, complementary kind
  of evidence.
- **The known, disclosed MINOR gap** (`filter_products`'s `categoryNormalized` NULL-coalesce
  drops any `Product` missing that property, including from an unfiltered "list everything"
  call) — the review confirms this never manifests today because every seed/fixture row sets
  `categoryNormalized`. This pass's own seed data is written by `seed_catalog.sh`, the same
  script the review inspected, so this gap is **not** expected to reproduce here; not
  independently re-probed (would require injecting a malformed fixture row into the shared
  `reference` graph, an intentional escalation of the review's own finding, out of this pass's
  scope — the review's disposition already stands).

**Explicit, disclosed test-design substitution:** debug tool-call tracing
(`WorkflowRun.trace=true`, which persists `TraceEvent`s recording exact tool name/arguments) is
**off by default** on the `@mention` trigger path (`WorkflowTrigger.__init__`'s `trace: bool =
False`, never overridden in `app.py`'s trigger construction) — there is no low-risk way to turn
it on for this pass without either patching `app.py` (out of scope — QA does not modify the
system under test) or starting a second, parallel `WorkflowRun` against an already-triggered
message via `POST /workflow-runs` (would duplicate tool calls and customer-visible replies for
the same turn, polluting evidence). **Consequence for AC-4:** "two phrasings map onto the same
fixed query shape" is verified by the customer-visible outcome — both phrasings independently
produce the *correct* price for the *same* product — rather than by inspecting raw tool-call
arguments. This is real evidence (a wrong shape, or an untethered guess, could not reliably
reproduce the exact seeded price twice from two different phrasings), but it is indirect, not a
literal trace diff; recorded as a coverage gap, not silently substituted.

**Regression baseline — deliberately not re-run.** `analyst` (U16) already ran, on this exact
commit, the full offline `pytest` suite (1811 passed / 4 deselected) and
`./scripts/test_queries.sh` (346/346), both **today**, and restored `ws:acme`'s standard
baseline afterward. Both suites `GRAPH.DELETE` the shared `reference` graph at teardown
(`falkor-chat/AGENTS.md`) — re-running either here would needlessly duplicate work already done
today on the same code, and would force restoring `ws:acme`'s shared baseline a second time for
no incremental confidence. This pass treats U16's run as the regression baseline and does not
repeat it — a deliberate, disclosed choice per this agent's risk-based-prioritization principle,
not an oversight. The scripts this pass *does* run (`bootstrap_schema.sh`, `seed_demo.sh`,
`seed_catalog.sh`, `seed_salesperson.sh`, `verify_catalog.sh`, `verify_salesperson.sh`) are all
idempotent, additive, create-only writes against a **fresh throwaway workspace** — none of them
delete or destructively rewrite shared state.

## 4. Environment & data setup

Pre-flight only in TP-01; provisioning happens in TP-02/TP-04 as dedicated test items (not
silently done before testing starts), since "does the seed/materialize path work at all" is
itself part of AC-5.

- FalkorDB: `falkordb-dev` container, already up (confirmed `PONG` at 127.0.0.1:6379).
- LM Studio: already up at `http://localhost:1234`, `qwen/qwen3-4b-2507` listed among served
  models (same model U15 live-proved AC-1/AC-3/AC-4 against).
- Server venv: `falkor-chat/server/.venv` already present.
- **Fresh throwaway workspace: `ws:qa-catalog-lookup`** — per the coordinator's brief, not
  `ws:live-salesperson` (U15's own smoke-test workspace).
- Server started with `FALKORCHAT_WS_ID=qa-catalog-lookup`,
  `FALKORCHAT_TRIGGER_DEF_KEY=salesperson`, `FALKORCHAT_TRIGGER_DEF_VERSION=v1`, and
  `UVICORN_ARGS` overridden to a non-empty value that does **not** include `--reload` (the
  bash `:-` default substitutes `--reload` on an *empty* override too — `docs/SERVER.md` §1.7's
  warning that reload restarts the worker, killing in-flight background loops, mid-conversation
  during a live QA pass).

## 5. Test items

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-01 | Environment pre-flight (FalkorDB, LM Studio, venv) | P0 | environment |
| TP-02 | Fresh-workspace bootstrap + seed (schema, demo, catalog, `salesperson@v1`) | P0 | setup |
| TP-03 | `verify_catalog.sh` / `verify_salesperson.sh` green after first seed | P1 | integration |
| TP-04 | Start server wired to the `salesperson` trigger; health check | P0 | environment |
| TP-05 | Fresh thread creation in `demo-general` | P0 | setup |
| TP-06 | AC-1 — exact-name single-item fact lookup | P0 | e2e/acceptance |
| TP-07 | AC-2 — category filter ("what Wearables do you have") | P0 | e2e/acceptance |
| TP-08 | AC-2 — price-range filter ("under $30") | P0 | e2e/acceptance |
| TP-09 | AC-3 — not-found product name | P0 | e2e/acceptance |
| TP-10 | AC-3 — not-found category | P0 | e2e/acceptance |
| TP-11 | AC-4 — phrasing A ("how much is X") | P0 | e2e/acceptance |
| TP-12 | AC-4 — phrasing B ("what's the price of X"), cross-checked against TP-11 | P0 | e2e/acceptance |
| TP-13 | Graph-level ground truth for TP-06..TP-12 | P1 | integration |
| TP-14 | AC-5 — seed-script idempotence (re-run `seed_catalog.sh` + `seed_salesperson.sh`) | P0 | acceptance |
| TP-15 | Live scaffold-safety corroboration (run never reaches `ended`/`done`) | P1 | regression |

### TP-01 — Environment pre-flight
**Preconditions:** none.
**Steps:** `redis-cli PING`; `GET http://localhost:1234/v1/models`; confirm
`server/.venv/bin/python -c "import falkorchat"` succeeds.
**Expected:** FalkorDB responds `PONG`; `qwen/qwen3-4b-2507` listed; venv importable.
**Priority:** P0. **Type:** environment.

### TP-02 — Fresh-workspace bootstrap + seed
**Preconditions:** TP-01 passed.
**Steps:**
1. `./scripts/bootstrap_schema.sh qa-catalog-lookup` (EMBEDDING_DIM=1024).
2. `FALKORCHAT_WS_ID=qa-catalog-lookup ./scripts/seed_demo.sh qa-catalog-lookup`.
3. `./scripts/seed_catalog.sh` (global `reference`, `<wsId>` arg unused per FR-6).
4. `FALKORCHAT_WS_ID=qa-catalog-lookup ./scripts/seed_salesperson.sh qa-catalog-lookup`.
**Expected:** all four exit 0; `seed_catalog.sh` reports 15 `Product` rows processed, 15 total
in `reference`; `seed_salesperson.sh` reports `salesperson@v1` created in `reference` and
materialized into `ws:qa-catalog-lookup`.
**Priority:** P0. **Type:** setup.

### TP-03 — `verify_catalog.sh` / `verify_salesperson.sh` green
**Preconditions:** TP-02.
**Steps:** `./scripts/verify_catalog.sh`; `./scripts/verify_salesperson.sh qa-catalog-lookup`.
**Expected:** both exit 0, `RESULT: OK`; catalog count 15; def topology matches plan §3.3 (2
steps, 1 transition, start key `assistant`).
**Priority:** P1. **Type:** integration.

### TP-04 — Start server, health check
**Preconditions:** TP-02/TP-03.
**Steps:** start `./scripts/start_server.sh` in the background with
`FALKORCHAT_WS_ID=qa-catalog-lookup FALKORCHAT_TRIGGER_DEF_KEY=salesperson
FALKORCHAT_TRIGGER_DEF_VERSION=v1 UVICORN_ARGS="--port 8000"` (no `--reload`); poll
`GET http://localhost:8000/health`.
**Expected:** `{"status":"ok"}`; startup banner confirms `Workflow: ... (salesperson def
salesperson@v1)`.
**Priority:** P0. **Type:** environment.

### TP-05 — Fresh thread creation
**Preconditions:** TP-04.
**Steps:** `POST /channels/demo-general/threads {"title": "qa-workflow-catalog-lookup"}`.
**Expected:** `201`, new `threadId`.
**Priority:** P0. **Type:** setup.

### TP-06..TP-12 — Live conversation (one thread, sequential `@mention` turns)

One continuous conversation in TP-05's thread — the intended usage shape of a persistent,
`waitsForHuman` conversational agent, and the cheapest way to get seven independent live
tool-call turns without re-provisioning between each. Each turn:
`POST /threads/{tid}/messages {"text": "@assistant <question>", "mentions": ["assistant"]}`,
then poll `GET /threads/{tid}/workflow-runs` until the run's latest step-run settles, then
`GET /threads/{tid}/messages?since=<pre-post ts>` for the assistant's reply.

| ID | AC | Question | Expected answer (from `seed_catalog.sh`'s literal) |
|---|---|---|---|
| TP-06 | AC-1 | "How much does the Wireless Mouse Pro cost?" | $29.99 |
| TP-07 | AC-2 | "What Wearables products do you have?" | Fitness Tracker Band ($79.99), Smartwatch Series 5 ($249.99) — both, no others |
| TP-08 | AC-2 | "What products do you have under $30?" | Gaming Mouse Pad XL ($19.99), Wireless Charging Pad ($24.99), Wireless Mouse Pro ($29.99) — all three, no others |
| TP-09 | AC-3 | "How much does the Quantum Toaster 3000 cost?" | No fabricated price; plain statement nothing matched |
| TP-10 | AC-3 | "Do you have any Furniture products?" | No fabricated list; plain statement nothing matched |
| TP-11 | AC-4a | "How much is the Portable SSD 1TB?" | $109.99 |
| TP-12 | AC-4b | "What's the price of the Portable SSD 1TB?" | $109.99 — must match TP-11 exactly |

**Expected (all seven):** `WorkflowRun.status` returns to `"waiting"` after each turn (the
`assistant` step parks — never `"done"`, see TP-15); a new assistant-authored `Message` appears
in the thread read, containing the expected fact/list/abstention in natural language.
**Priority:** P0. **Type:** e2e/acceptance.
**Note:** any turn producing a fabricated answer (TP-09/TP-10), an incorrect fact/list (TP-06/
TP-07/TP-08), or a mismatched TP-11/TP-12 pair is a reportable defect, not assumed to be a
one-off model flub — each gets a rerun (once) to distinguish a reproducible defect from
nondeterministic model noise, per §6's verdict rule.

### TP-13 — Graph-level ground truth
**Preconditions:** TP-06..TP-12 executed.
**Steps:** direct read-only Cypher against `ws:qa-catalog-lookup`:
```cypher
MATCH (r:WorkflowRun {runId:$runId})-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (sr)-[:PRODUCED]->(m:Message)
RETURN sr.stepKey, sr.status, sr.output, m.msgId, m.text
ORDER BY sr.startedAt
```
**Expected:** one `PRODUCED` edge to a real `Message` per turn, text matching what the REST read
returned (defense against a read-path bug masking a write-path problem, mirroring
`mention-reply-delivery`'s TP-06 technique) — independent confirmation the reply is really
persisted, not just present in a possibly-cached REST response.
**Priority:** P1. **Type:** integration.

### TP-14 — AC-5: seed-script idempotence
**Preconditions:** TP-02 (first seed already ran).
**Steps:** re-run `./scripts/seed_catalog.sh` and
`FALKORCHAT_WS_ID=qa-catalog-lookup ./scripts/seed_salesperson.sh qa-catalog-lookup` a second
time; re-run `./scripts/verify_catalog.sh` / `./scripts/verify_salesperson.sh qa-catalog-lookup`.
**Expected:** both re-runs exit 0; `seed_catalog.sh` still reports exactly 15 `Product` nodes in
`reference` (no duplication — `productId` is a deterministic slug, per plan §3.2);
`seed_salesperson.sh` reports "already present — no-op" for both the `reference` def and the
`ws:qa-catalog-lookup` snapshot; both verify scripts still report `OK` afterward.
**Priority:** P0. **Type:** acceptance.

### TP-15 — Live scaffold-safety corroboration
**Preconditions:** TP-06..TP-12.
**Steps:** inspect each turn's `WorkflowRun.status` (already captured in TP-06..TP-12) and the
`ended`/`decision` step's `StepRun` presence for the run.
**Expected:** across all seven turns, the run's status is always `"waiting"` after the turn
completes, never `"done"`; no `StepRun` for the `ended` step ever appears — live corroboration
(not a substitute for) `analyst`'s mutation-tested regression guard on `ctx.endConversation`.
**Priority:** P1. **Type:** regression.

## 6. Entry / exit criteria

**Entry:** TP-01 passes.

**Exit / verdict rule:**
- **Pass** — TP-02..TP-14 all pass on first attempt (or after at most one reproducibility rerun
  per §5's note), TP-15 corroborates no premature-ended run. Verdict: "K-052 meets its
  acceptance criteria against the live running system."
- **Pass with defects** — one or more non-blocking issues found (e.g. a single flaky/incorrect
  model turn that does not reproduce on rerun, a minor UX wording issue) that don't invalidate
  the core AC — reported as defects with severity, verdict still ships.
- **Fail** — any of AC-1..AC-5 reproducibly fails to hold against the live system (wrong answer,
  fabricated answer, seed non-idempotence, or the run advancing to `ended`/`done` prematurely).

## 7. Out of scope

- Re-deriving unit/integration-level correctness already covered by
  `test_{repository,services,tools,salesperson_scaffold}.py` and independently re-verified by
  `analyst` (U16) — see §3.
- Re-running the destructive offline `pytest` / `test_queries.sh` suites — see §3's disclosed
  decision to rely on U16's same-day, same-commit run instead.
- Independently re-probing the open MINOR (`filter_products`'s `categoryNormalized`
  NULL-coalesce edge) beyond confirming it does not manifest with this pass's own seed data —
  the review's disposition (non-blocking follow-up) already stands; escalating it would require
  deliberately corrupting shared `reference` data, out of scope for an acceptance pass.
- Literal tool-call-argument tracing for AC-4 — `trace` is off by default on the `@mention`
  path and there is no low-risk way to turn it on without modifying the system under test; see
  §3's disclosed substitution (customer-visible-outcome evidence instead).
- Browser/UI automation — driven via direct REST calls using the exact shapes `web/app.js`
  sends, same substitution `mention-reply-delivery`/`web-api-coverage` already used and
  disclosed.
- K-053/K-054/K-055 (the three sibling capabilities layered onto later `salesperson` versions) —
  not yet implemented, not in scope for this unit.
