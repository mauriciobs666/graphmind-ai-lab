# `workflow-catalog-lookup` — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-052 (M6)

## Summary

Live acceptance pass for K-052 (structured catalog/reference lookup for workflows), executed
against the real running system — FalkorDB (`falkordb-dev`), the M1 server on
`http://localhost:8000`, and LM Studio on `http://localhost:1234` serving `qwen/qwen3-4b-2507` —
per `falkor-chat/docs/test-plans/workflow-catalog-lookup.md`. Workspace: a fresh throwaway
`ws:qa-catalog-lookup`, materialized from the current commit's `salesperson@v1` def
(`falkorchat.proof_defs.SALESPERSON_DEF`), untouched by any prior session's smoke testing
(`ws:live-salesperson`).

**Verdict: PASS WITH DEFECTS.** All five acceptance criteria (AC-1..AC-5) hold against the live
system: verified directly (fresh, short conversations, one AC per conversation) and cross-checked
against ground-truth Cypher run directly on `reference`. The seed/materialize/verify path (AC-5)
is genuinely idempotent — a second `seed_catalog.sh`/`seed_salesperson.sh` run is a clean no-op,
confirmed by both scripts' own reporting and by `verify_catalog.sh`/`verify_salesperson.sh`
staying green. The scaffold's safety-critical property (the `assistant` step never reaches the
`ended`/`done` terminal state) held across all 8 live conversation turns and 5 `WorkflowRun`s —
live corroboration of `analyst`'s mutation-tested regression guard.

**One real, reproducible defect was found and is reported below (D-1)**, discovered specifically
*because* this was a live pass rather than a repeat of the static/fixture-level gates: within one
continuous, multi-turn conversation, the live local model reproducibly (2/2 attempts) fabricated
catalog facts — invented products that don't exist, and an invented price for a real product —
on the fifth and later tool-calling turns, after having answered the first two turns correctly.
Root-caused via ground-truth Cypher (the repository/service/tool layer returns the *correct* data
every time) and via re-testing the identical questions in fresh, independent conversations (all
succeeded correctly) — this isolates the defect to live tool-call reliability degrading over an
extended conversation, not to K-052's own implementation. Recorded as a real, user-impacting
defect (the "salesperson" demo is explicitly designed as "one long-lived conversational loop, many
turns," so this failure mode is squarely in its intended usage pattern), but not a K-052 code
blocker — the repository/services/tools/def layer is confirmed correct in every case tested.

**CPG:** considered, not relevant — `cpg_falkorchat` is stale relative to `falkor-chat/server`
(per the coordination brief) and this is a live acceptance pass of new code driving the running
system, not a structural-impact-analysis question. Verified repository/tool behavior by reading
`repository.py`/`tools.py`/`proof_defs.py` directly and by querying the live graph.

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-01 | PASS | `redis-cli PING` → `PONG`; `GET :1234/v1/models` lists `qwen/qwen3-4b-2507`; `server/.venv` imports `falkorchat` cleanly. |
| TP-02 | PASS | `bootstrap_schema.sh qa-catalog-lookup` (dim 1024) exit 0; `seed_demo.sh qa-catalog-lookup` exit 0 (agent `assistant`, channel `demo-general`, thread `demo-welcome`); `seed_catalog.sh` → "15 product row(s) processed... 15 Product node(s) now in reference"; `seed_salesperson.sh qa-catalog-lookup` → `reference` def already present (published by an earlier session, correctly a no-op), `ws:qa-catalog-lookup` snapshot materialized fresh. |
| TP-03 | PASS | `verify_catalog.sh` → `RESULT: OK — product catalog in sync (15 products)`; `verify_salesperson.sh qa-catalog-lookup` → `RESULT: OK`, topology "2 steps (assistant/agent, ended/decision), 1 transition". |
| TP-04 | PASS (after one environment fix — see Feedback #1) | First launch: server up (`{"status":"ok"}`) but every live-model call failed (`ProviderCallError: connection failed [Errno 111]` to the shared opencode config's gateway-IP `baseURL`). Relaunched with `FALKORCHAT_OPENCODE_CONFIG` pointed at a scratch copy with `baseURL` corrected to `http://localhost:1234` (the shared, pristine config file itself was **not** modified) — banner then logs `baseURL http://localhost:1234 -> http://localhost:1234/v1 (rule)`, `GET /health` → `{"status":"ok"}`. |
| TP-05 | PASS | `POST /channels/demo-general/threads {"title":"qa-workflow-catalog-lookup"}` → `201`, `threadId 88cdf9ad52ab4e158659fe26e2b2f32b`. |
| TP-06 (AC-1) | PASS | "How much does the Wireless Mouse Pro cost?" → run `285d6eb2...` reached `waiting`; reply: "The Wireless Mouse Pro costs $29.99." — matches `seed_catalog.sh`'s literal exactly. (First attempt, before the TP-04 config fix, failed with `status:"failed"` due to the LM Studio connectivity issue — not a feature defect; not counted against this item.) |
| TP-07 (AC-2, category) | PASS | "What Wearables products do you have?" → reply: "Fitness Tracker Band: $79.99, Smartwatch Series 5: $249.99" — both seeded Wearables items, correct prices, no extras. |
| TP-08 (AC-2, price range) | PASS (isolated) — see **D-1** | In the long-running conversation: reply fabricated 2 non-existent products ("USB Flash Drive (16GB)", "Power Bank (10,000mAh)"), reproduced identically on a same-thread rerun. Ground-truth Cypher against `reference` confirms the correct set is `Gaming Mouse Pad XL ($19.99), Wireless Charging Pad ($24.99), Wireless Mouse Pro ($29.99)`. Re-tested in a **fresh** thread (`49a1860d...`): reply returned exactly that correct 3-item set, ordered by price ascending, no extras — PASS in isolation. |
| TP-09 (AC-3, product) | PASS | "How much does the Quantum Toaster 3000 cost?" → "The Quantum Toaster 3000 is not available in our catalog." — plain abstention, no fabricated price. |
| TP-10 (AC-3, category) | PASS | "Do you have any Furniture products?" → "We do not have any Furniture products in our catalog." — plain abstention. |
| TP-11 (AC-4a) | PASS (isolated) — see **D-1** | In the long-running conversation: "How much is the Portable SSD 1TB?" → "$149.99" (wrong; ground truth is $109.99), reproduced identically on a same-thread rerun. Re-tested in a **fresh** thread (`443e7566...`): "The Portable SSD 1TB is priced at $109.99 and falls under the Storage category." — correct. |
| TP-12 (AC-4b) | PASS (isolated) | In the long-running conversation, phrasing B repeated the same (wrong) $149.99 already established in that conversation's own history — a weak, contaminated signal, not independent evidence. Re-tested in a **fresh** thread (`72eb19e1...`): "The price of the Portable SSD 1TB is $109.99." — correct, matches TP-11's fresh-thread answer exactly. AC-4 ("two phrasings, same fixed shape, correct answer") holds when tested as intended — two independent conversations, not two turns of one contaminated conversation. |
| TP-13 | PASS | Direct Cypher against `ws:qa-catalog-lookup` for all 8 turns (main thread + 3 fresh threads) confirms every `StepRun`→`PRODUCED`→`Message` edge's text matches the REST-reported reply exactly, including the two fabricated-content turns (graph ground truth agrees with what the user actually saw — confirms the write path, not just the read path, and confirms the fabrication is genuine model output, not a display artifact). |
| TP-14 (AC-5) | PASS | Re-ran `seed_catalog.sh` → still exactly 15 `Product` nodes (no duplication — deterministic slug ids hold). Re-ran `seed_salesperson.sh qa-catalog-lookup` → both `reference` def and `ws:qa-catalog-lookup` snapshot report "already present — no-op". `verify_catalog.sh`/`verify_salesperson.sh qa-catalog-lookup` both still `RESULT: OK` afterward. |
| TP-15 | PASS | All 5 `WorkflowRun`s (1 failed on the pre-fix connectivity issue, 4 real conversations) show `status` either `waiting` or the one pre-fix `failed` — never `done`. Direct Cypher: `MATCH (sr:StepRun {stepKey:'ended'}) RETURN count(sr)` → `0` across the whole workspace. Live corroboration of the `ctx.endConversation` guard-safety property `analyst` mutation-tested statically. |

## Defects

### D-1 — MAJOR (not a K-052 implementation defect) — Live tool-call reliability degrades within an extended conversation, causing fabricated catalog facts

**Severity:** MAJOR by user impact (a "salesperson" assistant inventing product names and prices
is a serious trust failure for its intended use), but **root-caused to be outside K-052's own
code** — the repository/service/tool/def layer is confirmed correct in every isolated test.

**Steps to reproduce:**
1. Seed `ws:qa-catalog-lookup` per TP-02, start the server per TP-04.
2. In a single thread, `@mention` the assistant with a sequence of catalog questions, letting
   each turn's `WorkflowRun` resume the same run (this thread's own history, per the trigger's
   documented "resume on next message" behavior):
   - Turn 1: "How much does the Wireless Mouse Pro cost?" → correct ($29.99).
   - Turn 2: "What Wearables products do you have?" → correct (both items, right prices).
   - Turn 3: "What products do you have under $30?" → **fabricated**: "Wireless Mouse Pro: $29.99,
     USB Flash Drive (16GB): $24.99, Power Bank (10,000mAh): $29.99" — 2 of 3 listed products do
     not exist in the catalog; the 2 real matches (Gaming Mouse Pad XL, Wireless Charging Pad) are
     missing entirely.
   - Turn 4 (rerun of turn 3's question): identical fabricated reply.
   - Turns 5-6: "How much does the Quantum Toaster 3000 cost?" / "Do you have any Furniture
     products?" → both correct (plain abstention).
   - Turn 7: "How much is the Portable SSD 1TB?" → **fabricated**: "$149.99" (correct is $109.99).
   - Turn 8 (rerun): identical fabricated reply.
   - Turn 9: "What's the price of the Portable SSD 1TB?" → repeats the same wrong $149.99 already
     stated earlier in this conversation's own history.

**Expected:** every turn returns the correct fact/list or a plain abstention, per FR-4's "never
guess a price or category you have not retrieved from a tool" (the `SALESPERSON_DEF` system
prompt's own explicit instruction).

**Actual:** turns 3-4 and 7-9 (the model's *later* tool-calling turns in one continuous
conversation) fabricate catalog facts instead — inventing products, inventing a price for a real
product — reproducibly (2/2 reruns each), even though the *identical* questions, asked as the
*first* message of a fresh conversation, succeed correctly every time (confirmed for both the
price-range-filter shape and the exact-name-lookup shape, in dedicated fresh threads
`49a1860d...`/`443e7566...`/`72eb19e1...`).

**Root cause (isolated, not fully diagnosed — see Feedback):**
- **Not** the repository layer: `MATCH (p:Product) WHERE p.categoryNormalized = coalesce(...)
  AND p.price >= coalesce(...) AND p.price <= coalesce(30, ...) ...` run directly against
  `reference` returns exactly the correct 3-row set (`docs/QUERIES.md` §15.2's own Cypher).
- **Not** the service/tool/def layer: the identical question, asked in a fresh conversation
  against the identical seeded data, gets the correct answer every time.
- **Is** something about the live model's behavior once a conversation has accumulated several
  prior tool-calling turns — consistent with a small (4B-parameter) local model's tool-call
  reliability degrading as context grows, though this pass could not directly confirm whether the
  tool was even invoked on the failing turns (debug tracing is off by default on the `@mention`
  path — see the test plan's disclosed §3 substitution). The fact that turn 9 exactly repeats
  turn 7's wrong answer, rather than independently re-deriving a new (possibly different) wrong
  answer, is consistent with the model pattern-matching its own prior conversational turn instead
  of re-invoking the tool at all.

**Impact:** every one of K-052's own AC-1..AC-5 acceptance criteria is satisfied when tested as
literally worded (a single question against a freshly-seeded catalog) — this is why the overall
verdict is not FAIL. But the `salesperson` scaffold is explicitly designed for "one long-lived
conversational loop, many turns" (`SALESPERSON_MAX_STEPS = 40`), and three more sibling
capabilities (K-053/K-054/K-055) are about to layer more tools onto this exact same scaffold and
model — extending real conversations further and multiplying tool choices, which this evidence
suggests will make the failure mode more likely to surface, not less.

## Coverage & gaps

**Covered:**
- All five AC's, each independently verified against a fresh, short live conversation and
  cross-checked against ground-truth Cypher and/or graph-level `Message` content — the acceptance
  bar as literally worded.
- The seed/verify idempotence chain (AC-5), including a genuine second-run no-op confirmation.
- The scaffold's safety-critical guard property, corroborated live across 8 real conversation
  turns (in addition to `analyst`'s static mutation testing).
- Graph-level ground truth for every reply (write path, not just read path).

**Gaps, some deliberately scoped out, one newly surfaced by this pass:**
- **Not statistically characterized**: how often the D-1 degradation occurs, at what turn count,
  or whether it's specific to `qwen/qwen3-4b-2507` — this pass demonstrates the failure mode is
  real and reproducible on the exact turns tested, not its overall rate. That characterization is
  `docs/BACKLOG.md`'s already-flagged K-027 epic (`data-scientist`/`architect` territory),
  unaffected in scope by this finding but now has fresh, concrete live evidence supporting its
  priority.
- **No literal tool-call-argument tracing** for D-1's root cause (disclosed in the test plan
  before testing began) — `trace` is off by default on the `@mention` path; enabling it would
  require modifying the system under test, out of scope for QA. The evidence gathered (ground-
  truth Cypher + fresh-conversation isolation) is real but indirect on whether the tool was even
  called on the failing turns.
- **The known, disclosed MINOR** (`filter_products`'s `categoryNormalized` NULL-coalesce edge,
  `docs/reviews/workflow-catalog-lookup-impl.md`) did not manifest — expected, since this pass's
  seed data comes from the same `seed_catalog.sh` the review already confirmed always sets that
  property. Not independently re-probed (would require deliberately corrupting shared `reference`
  data), per the test plan's explicit scoping.
- Regression baseline (offline `pytest`, `test_queries.sh`) deliberately not re-run — relied on
  `analyst` (U16)'s same-day, same-commit run (1811 passed/4 deselected; 346/346), per the test
  plan's disclosed §3 decision to avoid redundant destructive shared-state mutation.
- No browser/UI automation — driven via direct REST calls matching `web/app.js`'s request shapes,
  same disclosed substitution prior QA passes in this component have used.

## Feedback & recommendations

1. **D-1 is worth a `data-scientist`/`architect` follow-up before K-053/K-054/K-055 ship**, given
   those three sibling capabilities add more tools to this exact scaffold and model. A cheap
   first diagnostic: turn on `trace=true` for a scripted repro of D-1's exact turn sequence (would
   need a small `app.py` change, or driving `services.start_workflow_run` directly with
   `trigger_msg_id` set to an already-posted message and `trace=True`) to see definitively whether
   the tool is being called with wrong/stale arguments, or not being called at all on the failing
   turns. Not attempted here — modifying the system under test is out of QA's remit, and the
   evidence already gathered (ground-truth correctness + fresh-conversation isolation) is
   sufficient to characterize the defect's shape without it.
2. **Environment gotcha, fixed for this session only, not for the shared config.** The shared
   `~/.config/opencode/opencode.json`'s `lmstudio` provider `baseURL` is hardcoded to a gateway IP
   (`http://192.168.0.69:1234`) that was unreachable from this WSL2 box during this entire
   session, while `http://localhost:1234` worked throughout (mirrored networking is evidently
   active) — every live model call failed until this pass pointed its own server process at a
   scratch copy with the corrected `baseURL`, via the documented `FALKORCHAT_OPENCODE_CONFIG`
   override seam (the shared file itself was left untouched, per this agent's "never mutate shared
   environment without disclosure" guardrail). Recommend `devops`/the demo owner either update the
   shared file's `baseURL` to `localhost` (matching the current WSL2 network state) or add a
   preflight check to `start_server.sh` that fails fast with a clear message when the configured
   `baseURL` isn't reachable, rather than surfacing as a mid-run `WorkflowRun.status:"failed"` with
   a buried `ProviderCallError` (as happened to this pass's very first live trigger attempt,
   TP-06's pre-fix run `254da7ba...`).
3. **Ship K-052 as-is.** No implementation-layer defect was found anywhere in the repository/
   service/tool/def code under test — every AC holds under direct, isolated live testing, and the
   one real defect found (D-1) is a live-model/conversation-length characteristic outside this
   feature's own code, already flagged as a known open epic (K-027) this pass gives fresh
   supporting evidence for.
4. **Test-plan process note.** This pass deviated from its own plan's literal "one continuous
   8-turn conversation" design once TP-08/TP-11 surfaced apparent failures — re-testing the same
   questions in fresh, independent threads was necessary to distinguish "K-052 is broken" from
   "this specific long test conversation contaminated the signal," and turned out to be the latter.
   Worth a standing note for any future live QA pass against this scaffold (or its 3 upcoming
   siblings): **a single long test conversation is not, on its own, reliable evidence for or
   against a specific tool-call shape** — a finding discovered mid-turn in a long conversation
   should be re-verified in a fresh thread before being reported as a defect in the mechanism
   itself, exactly as done here.

## Artifacts left in the live demo (disclosed, not cleaned up)

- Workspace `ws:qa-catalog-lookup` (fresh, this pass's own): `demo-general` channel/`demo-welcome`
  thread from `seed_demo.sh`, plus 4 test threads (`qa-workflow-catalog-lookup`,
  `qa-catalog-lookup-fresh-a/b/c`) carrying 5 `WorkflowRun`s and their messages, per the table
  above. Left in place for `teco`'s own disposition — this pass did not delete anything.
- `reference` graph: the `Product` catalog (15 rows, shared/global per FR-6) and `salesperson@v1`
  def were already present from an earlier session (confirmed idempotent, not re-created);
  nothing new was added there by this pass beyond what TP-02/TP-14 already show as a no-op.
- No shared files were modified — the corrected LM Studio config lives only in this session's
  scratchpad (`/tmp/.../scratchpad/opencode-qa.json`), referenced only via
  `FALKORCHAT_OPENCODE_CONFIG` for this pass's own server process, which has been stopped.
