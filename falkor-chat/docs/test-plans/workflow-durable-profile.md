# `workflow-durable-profile` — Test Plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-054 (M6)

## 1. Scope & objective

First live acceptance pass for K-054 (durable, workspace-scoped customer name/delivery-address
capture) — both prior gates on this code (the implementer's own mutation-tested unit suite,
commits `663093d`/`36096d0`, and `analyst`'s plan-gate review that caught and closed the v1
unconditional-`SET` BLOCKER) are static or fixture-level. This plan exercises the real
`salesperson@v3` demo agent, the real local LLM (LM Studio, `mistralai/ministral-3-3b`), and the
real FalkorDB `reference`/`ws:<id>` graphs against **AC-1..AC-3** of
`docs/requirements/workflow-durable-profile.md`, plus a cheap regression spot-check of the two
already-shipped sibling capabilities (catalog lookup, cart/order) sharing the same def.

## 2. References

- Requirements: `docs/requirements/workflow-durable-profile.md` — FR-1..FR-4, AC-1..AC-3.
- Implementation plan: `docs/plans/workflow-durable-profile.md` (`architect`, §5's test-strategy
  table is this plan's starting point).
- Graph design: `docs/plans/workflow-durable-profile-graph.md` v2 (`graph-dba`) — the
  `coalesce()`-guarded `write_profile`/`read_profile` Cypher, live-verified twice at design time
  (§0) against a disposable probe graph. This pass is the first time that Cypher is exercised
  through the real tool-call path instead of a probe script.
- Plan-gate review: `docs/plans/workflow-salesperson-demo-coordination.md` — 1 BLOCKER (v1's
  unconditional `SET` would null an omitted field on a partial update, defeating AC-2), fixed in
  the graph note's v2 and mirrored in `repository.upsert_profile`'s code (`coalesce()` per field).
  This plan's AC-2 items are this BLOCKER's live-path regression check.
- Code under test: `server/falkorchat/repository.py` (`upsert_profile`/`get_profile`,
  `:3070-3122`), `services.py` (`get_profile`/`save_profile`, `:2839-2864`), `tools.py`
  (`GetProfileTool`/`SaveProfileTool`, `:705-782`), `proof_defs.py` (`SALESPERSON_DEF` v3 — two
  tools added, `systemPrompt` extended, `config.model` carried forward unchanged from v2.1),
  `scripts/{seed,verify}_salesperson.sh`.
- Precedent for live-driving `salesperson` via REST + `@mention` and the ground-truth-Cypher
  discipline: `docs/test-reports/workflow-cart-and-totals2-report.md`,
  `docs/test-reports/workflow-catalog-lookup2-report.md`.

**CPG:** considered, not relevant — same reasoning as both prior K-052/K-053/K-054 planning
documents in this coordination: two nullable scalar properties added to an already-existing node,
no new label/relationship/index, `repository.py`/`services.py`/`tools.py` read directly. No
structural-impact-analysis question this feature raises that a CPG would answer better than
reading the (small, already-read) diff.

## 3. Risk assessment

**What's already covered, not re-tested here:** `Repository.upsert_profile`/`get_profile`
correctness against a fixture graph (both AC-2 partial-update directions, `test_repository.py`
`:3327-3420`), `Services.get_profile`/`save_profile` against a fake repo (`test_services.py`),
`Tool.run` dispatch for both new tools against a fake services object (`test_tools.py`), and
`_validate_def_spec` accepting `SALESPERSON_DEF` v3 (`test_salesperson_scaffold.py`) — 15
profile-tagged tests, confirmed green as this plan's own pre-flight (§4). Re-driving those
scenarios through direct repository/service calls here would be duplicate unit-layer coverage;
this plan extends past it.

**What genuinely hasn't been checked yet, and is this pass's real risk surface:**
- **Does the real LLM actually call `get_profile` early and `save_profile` when given
  name/address, and does it correctly withhold re-asking once both are known?** The unit tests
  drive `Tool.run` directly with hand-built arguments — they cannot prove a real model, given a
  customer's free-text turn, dispatches these two tools at the right moments. The plan's own §6
  risk flags this as a prompt-discipline concern separate from persistence correctness; this pass
  is designed to distinguish the two failure modes if either appears.
- **Does the `coalesce()` fix actually round-trip correctly through the live tool-call path**,
  not just the graph note's own probe-script verification and the fixture-level repository tests?
  This is AC-2's core claim and the exact axis the BLOCKER was found on — proven here with
  ground-truth Cypher immediately after each live turn, never trusting reply text alone.
- **Cross-conversation persistence over the real `@mention`/thread mechanism** (AC-1) — the unit
  test for this (`test_profile_persists_across_repository_instances_ac1`) proves it at the
  repository layer; this pass proves it end-to-end (new `Thread`, same actor, same workspace).
- **Regression on the two already-shipped sibling tools** (catalog lookup, cart) sharing the same
  `assistant` step and the same, now-longer, `systemPrompt` — a spot-check, not a full re-run
  (already covered by `docs/test-reports/workflow-{catalog-lookup2,cart-and-totals2}-report.md`).

**Deliberately not tested here (out of scope, stated plainly):**
- Cross-workspace profile persistence, writing to `identity`, auto-attaching a profile to an
  order, an extensible profile schema — all explicitly out of scope per the requirements doc.
- Multi-tenant/distinct-customer profile isolation — not independently demonstrable until K-016
  (M2.5) lands (the single-hardcoded-actor caveat, `docs/SERVER.md` §1.3); this plan's "new
  conversation, same customer" claim for AC-1 is provable today, "different customer gets a
  different profile" is not.
- A full re-run of every K-052/K-053 acceptance criterion — one cheap spot-check per capability
  only, per the task brief.
- Rate/statistical measurement of Ministral's own already-characterized duplicate-instruction
  defect (`docs/reviews/salesperson-tool-reliability-ml.md`) — out of this capability's scope;
  noted only if it happens to appear opportunistically.

## 4. Environment & data setup

- FalkorDB: `falkordb-dev`, already up (confirmed `PONG`).
- LM Studio: up at `http://localhost:1234`; `mistralai/ministral-3-3b` confirmed listed
  (`GET /v1/models`).
- Server venv: `server/.venv`, already present. Pre-flight unit check: `pytest -k profile -q` →
  **15 passed / 1924 deselected** (all new K-054 tests green before any live driving).
- Fresh throwaway workspace: `ws:qa-durable-profile` — bootstrapped
  (`EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh qa-durable-profile`), demo-seeded, catalog
  present (global `reference`, 15 products), `salesperson@v3`/`order-fulfillment@v1` materialized
  (`seed_salesperson.sh qa-durable-profile`) — confirmed `OK`/in-sync via `verify_catalog.sh`/
  `verify_workflows.sh qa-durable-profile`/`verify_salesperson.sh qa-durable-profile` (topology:
  2 steps/1 transition, unchanged from v2.1) before any test item ran. Zero pre-existing
  `Customer` nodes confirmed (`MATCH (c:Customer) RETURN count(c)` → `0`).
- Server bound to this workspace: `FALKORCHAT_WS_ID=qa-durable-profile FALKORCHAT_USER_ID=u1
  FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1 FALKORCHAT_TRIGGER_DEF_KEY=salesperson
  FALKORCHAT_TRIGGER_DEF_VERSION=v3 FALKORCHAT_EMBEDDING_DIM=1024
  FALKORCHAT_OPENCODE_CONFIG=<scratch copy, baseURL corrected to http://localhost:1234>`, port
  `8021`, no `--reload` (a live-driving pass must not risk a mid-conversation worker restart).
  `GET /health` → `{"status":"ok"}` on first attempt; log confirms the baseURL correction rule
  fired (`model provider lmstudio: baseURL http://localhost:1234 -> http://localhost:1234/v1`).
- Single-hardcoded-actor caveat (`docs/SERVER.md` §1.3) applies as usual: every live turn in this
  pass is the same actor `u1` — "same customer, new conversation" is what AC-1 asks for and what
  this environment can prove; distinct-customer isolation is out of scope (§3).

## 5. Test items

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-01 | Environment pre-flight + fresh-workspace provisioning | P0 | environment/setup |
| TP-02 | AC-1 (write) — give name and delivery address in one conversation | P0 | e2e/acceptance |
| TP-03 | AC-1 (read) — fresh, separate conversation already knows the customer, does not ask again | P0 | e2e/acceptance |
| TP-04 | AC-2 — later turn gives only an updated delivery address; name preserved | P0 | e2e/acceptance |
| TP-05 | AC-2 (symmetric direction) — later turn gives only an updated name; address preserved | P1 | e2e/acceptance |
| TP-06 | AC-3 — same combined `salesperson` def, not a separate workflow | P0 | acceptance/structural |
| TP-07 | Regression spot-check — catalog lookup + cart add in the same conversation | P1 | regression |
| TP-08 | Seed/verify idempotence (second run) | P1 | acceptance |
| TP-09 | Final regression: offline suite + `test_queries.sh`; restore `ws:acme`/`reference` | P0 | regression |

### TP-01 — Environment pre-flight + fresh-workspace provisioning
**Preconditions:** none.
**Steps:** the checks in §4.
**Expected:** all green — FalkorDB up, LM Studio serving `mistralai/ministral-3-3b`, profile unit
tests green, workspace bootstrapped/seeded/verified in sync, zero pre-existing `Customer` nodes,
server up and healthy on the corrected LM Studio config.
**Priority:** P0. **Type:** environment/setup.

### TP-02 — AC-1 (write): give name and delivery address in one conversation
**Preconditions:** TP-01 passed.
**Steps:** in a fresh thread ("Thread A"), `@assistant` a turn giving both a name and a delivery
address (e.g. "Hi, my name is Jane Doe and my delivery address is 123 Main St, Springfield.").
Poll `GET /threads/{tid}/workflow-runs` for a terminal run status, then read the reply. Ground
truth immediately after: `MATCH (c:Customer {customerId:'u1'}) RETURN c.name,
c.deliveryAddress, c.profileUpdatedAt`.
**Expected:** reply confirms the name/address were saved (`toolsUsed` includes `save_profile`);
ground truth shows exactly one `Customer` node with `name='Jane Doe'`,
`deliveryAddress='123 Main St, Springfield'`, `profileUpdatedAt` set.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-03 — AC-1 (read): fresh, separate conversation already knows the customer
**Preconditions:** TP-02 passed.
**Steps:** in a **new**, independent thread ("Thread B" — fresh `threadId`, same actor `u1`, same
workspace), `@assistant` a turn that would trigger a profile check but does not itself supply a
name/address (e.g. "Hi, do you have any wireless mice in stock?"). Read the reply in full.
**Expected:** the reply does **not** ask for the customer's name or delivery address (both already
known); it answers the actual question asked. Ground truth unchanged from TP-02 (same `Customer`
row, `profileUpdatedAt` unchanged unless the model also called `save_profile` redundantly — record
either way, only a re-ask on missing data is the acceptance failure per AC-1/AC-2's own risk note
in the plan (§6): "does not ask again" is a prompt discipline, not an engine guarantee, and a
finding here is triaged accordingly).
**Priority:** P0. **Type:** e2e/acceptance.

### TP-04 — AC-2: later turn gives only an updated delivery address
**Preconditions:** TP-03 passed.
**Steps:** in Thread B (or a new Thread C — either satisfies "a later, separate conversation"),
`@assistant` a turn giving **only** an updated delivery address, not repeating the name (e.g. "My
delivery address has changed to 456 Oak Ave, Springfield."). Ground truth immediately after:
`MATCH (c:Customer {customerId:'u1'}) RETURN c.name, c.deliveryAddress, c.profileUpdatedAt`.
**Expected:** reply confirms the address update; ground truth shows `name` **unchanged**
(`'Jane Doe'`, not nulled — this is the exact BLOCKER axis) and `deliveryAddress` updated to
`'456 Oak Ave, Springfield'`, `profileUpdatedAt` bumped.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-05 — AC-2 (symmetric direction): later turn gives only an updated name
**Preconditions:** TP-04 passed.
**Steps:** in a new turn (same or a new thread), `@assistant` gives **only** an updated name, not
repeating the address (e.g. "Actually, please call me Jane Smith instead."). Ground truth
immediately after, same query as TP-04.
**Expected:** `name` updated to `'Jane Smith'`; `deliveryAddress` **unchanged** (`'456 Oak Ave,
Springfield'`, not nulled) — confirms the `coalesce()` fix holds in both field directions live,
not just the one direction the task brief names, at negligible extra cost.
**Priority:** P1 (extra confidence on the exact bug axis; not separately required by AC-2's
wording, cheap to run). **Type:** e2e/acceptance.

### TP-06 — AC-3: same combined `salesperson` def
**Preconditions:** none (structural, not conversational).
**Steps:** `./scripts/verify_salesperson.sh qa-durable-profile`; separately confirm
`SALESPERSON_DEF["key"] == "salesperson"` (not a new def key) by reading `proof_defs.py`.
**Expected:** one def, `key:"salesperson"`, `version:"v3"`, topology unchanged (2 steps/1
transition, same as v2.1) — the two new tools/prompt guidance are `config` additions to the same
existing def, not a second workflow.
**Priority:** P0. **Type:** acceptance/structural.

### TP-07 — Regression spot-check: catalog lookup + cart add in the same conversation
**Preconditions:** TP-01 passed; can run in any thread, interleaved with or after the profile
turns above (proves the longer v3 `systemPrompt` didn't regress the other two tool families).
**Steps:** in one thread, `@assistant` one catalog-lookup turn (e.g. "How much is the Wireless
Mouse Pro?") and one cart-add turn (e.g. "Add 1 Wireless Mouse Pro to my cart."). Ground truth
after the cart turn: `MATCH (:Customer {customerId:'u1'})-[:HAS_CART]->(:Cart)-[:HAS_ITEM]->(i)
RETURN i.productId, i.quantity` (exact relationship names per `docs/QUERIES.md` §16 — confirm
against the live schema if this differs).
**Expected:** catalog answer matches the known fixture price ($29.99); cart add reply confirms;
ground truth shows the `CartItem` created/updated correctly — no regression from the K-052/K-053
baseline.
**Priority:** P1. **Type:** regression.

### TP-08 — Seed/verify idempotence (second run)
**Preconditions:** TP-01 passed.
**Steps:** re-run `./scripts/seed_salesperson.sh qa-durable-profile`;
`./scripts/verify_salesperson.sh qa-durable-profile`.
**Expected:** both defs report "already present — no-op"; verify still `OK`, same topology.
**Priority:** P1. **Type:** acceptance.

### TP-09 — Final regression: offline suite + `test_queries.sh`
**Preconditions:** all prior items complete.
**Steps:** stop the QA server; `server/.venv/bin/python -m pytest -q`;
`./scripts/test_queries.sh` (this wipes `reference` at teardown, per
`falkor-chat/AGENTS.md`); restore `ws:acme`/`reference` via
`bootstrap_schema.sh acme` → `seed_demo.sh acme` → `seed_catalog.sh acme` →
`seed_workflows.sh acme` → `seed_salesperson.sh acme`; re-verify with `verify_workflows.sh acme` /
`verify_catalog.sh` / `verify_salesperson.sh acme`.
**Expected:** offline suite green, matching the last recorded baseline (1935 passed / 4
deselected per the K-054 cluster-2 commit message) or explained if it differs; `test_queries.sh`
green (408/408 baseline); `ws:acme`/`reference` fully restored and re-verified in sync.
**Priority:** P0. **Type:** regression. **Run last** — this is the one item that touches shared
state, per the task's own practicalities note.

## 6. Entry/exit criteria

**Entry:** TP-01 all green (environment, workspace, pre-flight unit tests).
**Exit:** AC-1..AC-3 (TP-02/03/04/05/06) demonstrated live with ground-truth Cypher backing every
persistence claim; TP-07/08 show no regression; TP-09's final regression suite green and shared
workspaces restored. A failure on TP-02..TP-06 is a reportable defect, triaged per §3's stated
distinction between a persistence-layer bug (BLOCKER-class, would reopen the closed BLOCKER) and
a prompt-discipline gap (the model re-asks despite correct data — non-blocking per the plan's own
§6 risk note, but still recorded).

## 7. What's explicitly out of scope

Restated from §3: cross-workspace persistence, `identity` writes, auto-attach-to-order, an
extensible profile schema, multi-tenant profile isolation, a full K-052/K-053 re-run, and any
statistical measurement of Ministral's own already-characterized duplicate-instruction defect.
