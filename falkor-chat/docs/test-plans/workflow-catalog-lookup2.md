# `workflow-catalog-lookup` — Test Plan (Ministral re-verification)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-052, K-056 (M6) · **Extends:** `docs/test-plans/workflow-catalog-lookup.md`

## 1. Scope & objective

Regression re-verification of K-052's full acceptance-test suite against `salesperson@v2.1`,
now re-pointed from `qwen/qwen3-4b-2507` to `mistralai/ministral-3-3b` (`docs/HISTORY.md`/commit
`03a3c8c`, K-056). **This is not a redesign** — the underlying feature (`SALESPERSON_DEF`'s
catalog-lookup tools, unchanged since K-052) is identical; only the LLM driving the `assistant`
step has changed. The methodology, conversation shapes, and AC list are reused unchanged from
`docs/test-plans/workflow-catalog-lookup.md` (the "parent" plan); this document states only what
is different for this cycle and gives the full, self-contained test-item list so this report
stands on its own.

**Why re-run rather than trust the pilot eval:** `docs/reviews/salesperson-tool-reliability-ml.md`
§9's Ministral pilot (176 + 32 probe turns via a throwaway harness script, not the production
`@mention`/REST path) found zero instances of the qwen-era skip-and-fabricate mechanism, but that
pilot did not exercise K-052's actual tool surface, its actual seeded catalog data, or the real
`@mention`-trigger/REST path this feature ships through. This pass is the first live check of
K-052's own ACs, through its own real path, on the real new model.

## 2. References

- Parent plan (methodology, full risk assessment, conversation-shape rationale — not repeated
  here): `docs/test-plans/workflow-catalog-lookup.md`.
- Parent report (qwen-era baseline, D-1 defect this pass must confirm does or doesn't reproduce):
  `docs/test-reports/workflow-catalog-lookup-report.md`.
- Requirements (unchanged): `docs/requirements/workflow-catalog-lookup.md` — AC-1..AC-5.
- Re-point unit: `docs/plans/workflow-salesperson-demo-coordination.md` U43 (delivered, commit
  `03a3c8c`) — `SALESPERSON_DEF` `v2.1`, `config.model: "lmstudio/mistralai/ministral-3-3b"` on
  the `assistant` step only; `triage`/`access-request` untouched.
- Ministral evidence to date: `docs/reviews/salesperson-tool-reliability-ml.md` §8.4 (0/176
  skip-and-fabricate at pilot scale), §9 (duplicate-instruction follow-up, 1/32, honestly-grounded,
  self-disclosing, judged non-blocking) — informational context for this pass's watch item, not a
  substitute for a live re-run through the real path.
- Code under test (unchanged from K-052 except `proof_defs.py`'s `config.model` field):
  `server/falkorchat/repository.py`, `services.py`, `tools.py`, `proof_defs.py`
  (`SALESPERSON_DEF` v2.1), `scripts/{seed,verify}_catalog.sh`, `scripts/{seed,verify}_salesperson.sh`.

**CPG:** considered, not relevant — this is a live acceptance re-run against a changed model
dependency, not a structural-impact-analysis question; `cpg_falkorchat` freshness is immaterial
here since no code under test changed shape (only a config value did).

## 3. What's different from the parent plan

- **Model:** `mistralai/ministral-3-3b` (`lmstudio/mistralai/ministral-3-3b` ref), not
  `qwen/qwen3-4b-2507`. Reached the same way the coordination's own prior units did: a
  session-scratch `FALKORCHAT_OPENCODE_CONFIG` copy with `baseURL` corrected to
  `http://localhost:1234` (the shared `~/.config/opencode/opencode.json`'s gateway-IP baseURL,
  `http://192.168.0.69:1234`, is confirmed still unreachable from this WSL2 box as of this
  session — re-checked, not assumed).
- **Def version:** `salesperson@v2.1` (was `v1` at the original pass; `v2` and `v2.1` layered on
  by K-053 and K-056 respectively — topology unchanged, still 2 steps/1 transition, per
  `verify_salesperson.sh`'s own topology check).
- **Fresh workspace:** `ws:qa-catalog-lookup2` (not the original pass's `ws:qa-catalog-lookup`,
  and not `ws:acme`) — same disposable-workspace discipline the original pass and every sibling
  pass in this coordination have used.
- **New watch item (informational, not a gate):** TP-16, below — did Ministral's own
  duplicate-instruction defect (ml-note §9) surface anywhere in this pass. K-052's tool surface is
  entirely read-only (`lookup_product_fact`/`filter_products`), so there is no cart-line-quantity
  analogue to watch for directly; the closest possible manifestation here would be a duplicated
  *reply* (the same fact volunteered twice unprompted) rather than a duplicated write, which this
  item watches for opportunistically across TP-06..TP-12's turns without a dedicated new
  conversation script (K-053's re-run, `workflow-cart-and-totals2.md`, carries the substantive
  write-tool watch item, since only it has a tool surface the defect can actually manifest on).
- **AC-4's substitution reasoning is now doubly load-bearing.** Debug tracing is still off by
  default on the `@mention` path (unchanged); with a *different* model, "two phrasings match" is
  now also implicitly testing whether Ministral resolves both phrasings to the same tool-argument
  shape, not just whether qwen did — same customer-visible-outcome substitution as the parent
  plan, called out again because it now also stands in for AC-9-style model-independence
  evidence, not repeated as a new gap.

Everything else — the seven-turn shared-thread-then-fresh-thread technique for isolating a
long-conversation-only failure from a feature bug, the ground-truth Cypher cross-check discipline,
the seed-idempotence check, the scaffold-safety corroboration — is reused unchanged.

## 4. Environment & data setup

- FalkorDB: `falkordb-dev`, already up (`PONG` reconfirmed this session).
- LM Studio: up at `http://localhost:1234`; `mistralai/ministral-3-3b` confirmed listed via a
  direct `GET /v1/models` this session (also `qwen/qwen3-4b-2507` still listed and untouched).
- Server venv: `server/.venv`, already present, confirmed importable.
- Fresh throwaway workspace: `ws:qa-catalog-lookup2`.
- Server started with `FALKORCHAT_WS_ID=qa-catalog-lookup2`, `FALKORCHAT_ENABLE_AGENT=1`,
  `FALKORCHAT_WORKFLOW_ENABLED=1`, `FALKORCHAT_TRIGGER_DEF_KEY=salesperson`,
  `FALKORCHAT_TRIGGER_DEF_VERSION=v2.1`, `FALKORCHAT_EMBEDDING_DIM=1024`,
  `FALKORCHAT_OPENCODE_CONFIG=<scratch copy, corrected baseURL>`, `UVICORN_ARGS` set to a
  non-empty value without `--reload`.

## 5. Test items

Same ID scheme as the parent plan, one-to-one, plus TP-16 (new, informational).

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-01 | Environment pre-flight (FalkorDB, LM Studio incl. Ministral listed, venv) | P0 | environment |
| TP-02 | Fresh-workspace bootstrap + seed (schema, demo, catalog, `salesperson@v2.1`) | P0 | setup |
| TP-03 | `verify_catalog.sh`/`verify_salesperson.sh` green after first seed | P1 | integration |
| TP-04 | Start server wired to `salesperson@v2.1`/Ministral; health check | P0 | environment |
| TP-05 | Fresh thread creation in `demo-general` | P0 | setup |
| TP-06 | AC-1 — exact-name single-item fact lookup | P0 | e2e/acceptance |
| TP-07 | AC-2 — category filter ("what Wearables do you have") | P0 | e2e/acceptance |
| TP-08 | AC-2 — price-range filter ("under $30") — **also the D-1 repro slot (turn 3 of the qwen-era repro sequence)** | P0 | e2e/acceptance |
| TP-09 | AC-3 — not-found product name | P0 | e2e/acceptance |
| TP-10 | AC-3 — not-found category | P0 | e2e/acceptance |
| TP-11 | AC-4 — phrasing A ("how much is X") — **also the D-1 repro slot (turn 7)** | P0 | e2e/acceptance |
| TP-12 | AC-4 — phrasing B ("what's the price of X"), cross-checked against TP-11 | P0 | e2e/acceptance |
| TP-13 | Graph-level ground truth for TP-06..TP-12 | P1 | integration |
| TP-14 | AC-5 — seed-script idempotence | P0 | acceptance |
| TP-15 | Live scaffold-safety corroboration (run never reaches `ended`/`done`) | P1 | regression |
| TP-16 | Informational watch — Ministral duplicate-instruction/self-repetition pattern (ml-note §9) across TP-06..TP-12's turns | P2 | exploratory |

### TP-06..TP-12 — Live conversation (D-1 repro sequence reused verbatim)

One continuous conversation in TP-05's thread, **the exact same 9-question sequence the parent
report's D-1 reproduced on** (`docs/test-reports/workflow-catalog-lookup-report.md` D-1 repro
steps) — this is deliberate: re-running the identical sequence that broke on qwen is the direct,
apples-to-apples test of whether Ministral regresses the same way, not a fresh/easier sequence
that would only weakly speak to the question.

| Turn | ID | AC | Question | Expected answer |
|---|---|---|---|---|
| 1 | TP-06 | AC-1 | "How much does the Wireless Mouse Pro cost?" | $29.99 |
| 2 | TP-07 | AC-2 | "What Wearables products do you have?" | Fitness Tracker Band ($79.99), Smartwatch Series 5 ($249.99) |
| 3 | TP-08 | AC-2 | "What products do you have under $30?" | Gaming Mouse Pad XL ($19.99), Wireless Charging Pad ($24.99), Wireless Mouse Pro ($29.99) — the qwen-era D-1 fabrication slot |
| — | — | — | (rerun of turn 3, same question) | same correct 3-item set |
| 4 | TP-09 | AC-3 | "How much does the Quantum Toaster 3000 cost?" | plain abstention |
| 5 | TP-10 | AC-3 | "Do you have any Furniture products?" | plain abstention |
| 6 | TP-11 | AC-4a | "How much is the Portable SSD 1TB?" | $109.99 — the qwen-era D-1 fabrication slot |
| — | — | — | (rerun of turn 6, same question) | $109.99 |
| 7 | TP-12 | AC-4b | "What's the price of the Portable SSD 1TB?" | $109.99, must match TP-11 |

**Expected (all seven items):** every turn returns the correct fact/list/abstention on the *first*
attempt in this one continuous conversation — this is the specific regression this pass tests for
(the parent report's turns 3-4 and 7-9 fabricated on qwen; a clean run here is the direct
disconfirmation). If any turn fabricates or errs, rerun once in the same thread (reproducibility
check, mirroring the parent methodology) and then, regardless of outcome, re-verify the same
question in a fresh thread (isolates "long-conversation-specific" from "feature-broken," same
technique the parent report used) before concluding.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-13 — Graph-level ground truth
Same technique as the parent plan: direct Cypher against `ws:qa-catalog-lookup2` confirming every
`StepRun`→`PRODUCED`→`Message` edge's text matches the REST-reported reply.
**Priority:** P1. **Type:** integration.

### TP-14 — AC-5: seed-script idempotence
Re-run `seed_catalog.sh` + `seed_salesperson.sh qa-catalog-lookup2` a second time; re-run both
verify scripts. **Priority:** P0. **Type:** acceptance.

### TP-15 — Live scaffold-safety corroboration
Inspect every turn's `WorkflowRun.status` and confirm no `StepRun` for the `ended` step ever
appears. **Priority:** P1. **Type:** regression.

### TP-16 — Informational watch: Ministral duplicate-instruction pattern
**Preconditions:** TP-06..TP-12 executed.
**Steps:** review the raw reply text and (where available) `Message.toolsUsed` for TP-06..TP-12's
turns for any sign of an unprompted repeated fact/list (the read-path analogue of ml-note §9's
write-duplication pattern) — e.g. a later turn spontaneously re-stating an earlier turn's fact
unprompted, or invoking `lookup_product_fact`/`filter_products` for a product not asked about in
the current turn's own text.
**Expected:** not gated — report whatever is observed, including "not observed," as informational
context feeding K-053's own more substantive write-tool watch item.
**Priority:** P2. **Type:** exploratory.

## 6. Entry / exit criteria

**Entry:** TP-01 passes (including Ministral confirmed listed/reachable).

**Exit / verdict rule (unchanged from the parent plan):**
- **Pass** — TP-02..TP-14 all pass on first attempt (or after at most one same-thread
  reproducibility rerun), TP-15 corroborates no premature-ended run. Verdict: "K-052 continues to
  meet its acceptance criteria against the live running system on `mistralai/ministral-3-3b`."
- **Pass with defects** — one or more non-blocking issues found that don't invalidate the core
  ACs — reported as defects with severity, verdict still ships.
- **Fail** — any of AC-1..AC-5 reproducibly fails to hold, **or** the qwen-era D-1 mechanism
  reproduces on Ministral (a reportable regression in its own right, distinct from a fresh AC
  failure).

## 7. Out of scope

- Everything the parent plan already scoped out (§7 there): re-deriving unit/integration
  coverage, re-running the destructive offline suites mid-pass (deferred to a single final step
  shared with K-053's own re-run, per the coordinator's brief), independently re-probing the
  disclosed `categoryNormalized` MINOR, literal tool-call-argument tracing, browser/UI automation.
- A full statistical characterization of Ministral's own duplicate-instruction rate — that is
  `docs/reviews/salesperson-tool-reliability-ml.md` §9's job, already done at a meaningful sample
  size (n=32) via a dedicated harness; TP-16 here is an opportunistic watch during AC testing, not
  a fresh eval.
- K-054/K-055 — still not yet implemented, not in scope.
