# LLM Provider & Model Configuration — Landing 2 Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-042 (M4)

Executes `docs/test-plans/llm-provider-config2.md` (TP2-001..TP2-015). Against `main` at the
Landing-2-complete commit (`c4cf5ad` plus the coordination doc's bookkeeping commits on top; tree
clean at pass start), FalkorDB `falkordb-dev` (untouched, still up throughout), LM Studio at
`localhost:1234` (real, live, 18 models loaded including two embedding models and four distinct
chat-capable models). Run date 2026-08-11.

## Summary

**Verdict: PASS.** All seven in-scope acceptance criteria (AC-6, AC-7, AC-8, AC-9, AC-10 — all
four kinds, AC-11) plus AC-4's trace half hold when driven against the real, running system. This
is the first execution-based confirmation of Landing 2's design, closing the loop the two prior
static/diff-scoped `analyst` gates could not.

**One defect found, D-2 (Major):** the REST entry point for starting a workflow run
(`POST /workflow-runs`) returns a raw, unhandled `500 Internal Server Error` — a Python traceback,
not JSON — for exactly the class of fault AC-8 exists to exercise (a drive-time
`ModelResolutionError`), instead of the documented `{"status":"failed","error":...}` envelope. The
underlying run **is** correctly recorded as `failed` in the graph with a diagnostic message and
`AT_STEP` cleared — `GET /workflow-runs/{id}` reads it back cleanly — so AC-8's core behavioral
contract (no zombie run, no substituted model, the cause is recorded) holds; only the *initiating*
REST call's contract is broken. See Defects below.

No destructive action was taken against `reference`, `ws:acme`, or `ws:test`. All QA state lived
in two throwaway workspaces (`ws:qak042l2`, `ws:qak042l2dim4`), both deleted at teardown. One
pre-existing drift in shared state (`reference` missing both seeded workflow defs, found at the
very start of this pass, before any action of this pass's own) was detected and repaired — see
Environment notes.

## Results table

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP2-001 | AC-4 (trace half) | **PASS** | Run `316d963a…` (non-debug, `trace:false`, default). `GET .../step-runs`: `s1` → `resolvedModel: "lmstudio/qwen/qwen3-4b-2507"`, `s2` → `resolvedModel: "lmstudio/mistralai_ministral-3-3b-instruct-2512"` — two distinct concrete models on the durable trace of a single, ordinary run |
| TP2-002 | AC-7 (negative) | **PASS** | Publishing a step naming `nope/badmodel` → `400 WorkflowDefSpecError`: `"step 's1' names an unresolvable model 'nope/badmodel': unknown provider 'nope' (ref 'nope/badmodel', kind 'step') — declare it in ...opencode.json or ...models-v1.json"` — names both the step key and the identifier |
| TP2-003 | AC-7 (positive) | **PASS** | Six defs naming resolvable roles/concrete models (including a role, `roleA`) all published `201` |
| TP2-004 | AC-7 (M-4 ordering) | **PASS** | Republishing `qak042-ac4`@`v1` with a 3rd step (topology change) **and** an unresolvable model on that step → `409 WorkflowDefConflictError` naming the topology diff (`steps[s3], transitions[s2->s3@done#0]`), not the 400 the bad model alone would produce |
| TP2-005 | AC-6 (baseline) | **PASS** | Run `9e4d7126…` against overlay v1 (`roleA -> [qwen/qwen3-4b-2507]`): `resolvedModel == "lmstudio/qwen/qwen3-4b-2507"`, `modelSource: "step"` |
| TP2-006 | AC-6 (the point) | **PASS** | Overlay edited to `roleA -> [qwen/qwen3-4b-thinking-2507]` (v2); **server process restarted** against v2; **same def, no republish**. Run `7e1929bb…`: `resolvedModel == "lmstudio/qwen/qwen3-4b-thinking-2507"` — the new mapping, with zero republish |
| TP2-007 | AC-8 | **PASS, with D-2 found alongside** | Overlay v2 (post-restart) no longer declares `vanishRole`. Fresh run of a def naming it: graph state (`GET /workflow-runs/{id}`) shows `status: "failed"`, `ctx: {"error":"unexpected: ModelResolutionError(\"unknown role 'vanishRole' (kind='step') — declare it in .../models-v2.json's roles map, or use '<provider>/<model-id>'\")"}`, `atStepKey: null`; `step-runs` is `[]` (no model, fallback or otherwise, was ever used). The *initiating* `POST /workflow-runs` call itself returned a raw `500` — see D-2 |
| TP2-008 | AC-9 | **PASS** | Role `fallbackRole -> [lan/google/gemma-3n-e4b (unreachable), lmstudio/qwen/qwen3-4b-thinking-2507 (reachable)]`. Run `4e6bd8f5…`, total wall time 6.3s (fast failover, no hung timeout): `s1` → `resolvedModel: "lmstudio/qwen/qwen3-4b-thinking-2507"` (the **second** chain element), `modelFallback: true` |
| TP2-009 | AC-10, `step` | **PASS** | `write_model_overrides(ws, agent="lmstudio/mistralai_ministral-3-3b-instruct-2512")` (the `agentModelOverride` property governs the executor's `step` kind — the documented, non-1:1 crosswalk, `docs/QUERIES.md` §13). Def `qak042-ac10step`'s step explicitly names `qwen/qwen3-4b-2507`. Run `186738e0…`: `resolvedModel` on **both** steps is the override model, `modelSource: "workspace"` — beats the step's own explicit choice |
| TP2-010 | AC-10, `guard` | **PASS** | Guard override set to a reachable model; the `{"kind":"llm"}` transition's own declared `model` is the unreachable LAN ref. Run `e3375ff8…` completed (`status: "done"`, both steps executed) — the guard call reached the override model (had the unreachable declared model been used instead, the call would have failed fast exactly as AC-9's first chain element did, and the run would not have completed) |
| TP2-011 | AC-10, `embedding` | **PASS** | With `embeddingModelOverride` set to the reachable, dimension-matching model: a posted message got a real embedding (`Message.embedding IS NOT NULL` → `true`). Override cleared (falls back to `defaults.embedding`, pointed at the unreachable LAN provider for this test): the next message's embed failed with `ProviderCallError: lan/google/gemma-3n-e4b @ http://192.168.0.69:1234/v1/embeddings: connection failed: [Errno 111] Connection refused`, no vector written — both directions independently confirmed |
| TP2-012 | AC-10, `agent` | **PASS** | With `responderModelOverride` **and** `embeddingModelOverride` both set to reachable models (`defaults.agent`/`defaults.embedding` both point at the unreachable LAN provider in this overlay): `@assistant`-mentioning the seeded demo agent produced an actual posted reply in the thread. A first attempt with only the responder override set (embedding override not yet applied) failed cleanly with the same `ProviderCallError` as TP2-011's negative case — confirming `AgentResponder.maybe_respond`'s own query-embedding step is independently kind-gated too |
| TP2-013 | AC-11 | **PASS** | `ws:qak042l2dim4` (vector index dimension 4), configured embedding model declares `dim: 1024`. Posting a message raised `EmbeddingDimensionError: embedding dimension mismatch for workspace 'qak042l2dim4', label 'Message': the workspace's vector index is dimension 4, but the configured embedding model 'lmstudio/text-embedding-qwen3-embedding-0.6b' declares dimension 1024 ... refusing to embed before calling the model (no HTTP call made, no vector written)` — pre-flight (the message states no HTTP call was made), names both dimensions, the model, and the workspace; `Message.embedding` stayed null |
| TP2-014 | baseline | **PASS** | `.venv/bin/python -m pytest -q` from `server/` → `866 passed, 1 deselected in 9.57s` — reproduced independently, exact match to the coordination doc's own figure |
| TP2-015 | doc/script cross-check | **PASS** | Unfiltered `grep` for the four legacy var names, same result set as Landing 1's TP-010: only the tripwire's own list (`config.py`), test files' own tripwire tests, review/requirements documents discussing the topic historically, and `local-model-ram-budget-ml.md`'s already-amended pointer prose. No new operational reference |

## Defects

### D-2 — Major: `POST /workflow-runs` returns a raw `500` (traceback) instead of the documented failed-run envelope, for a drive-time `ModelResolutionError`

**Severity:** Major (a genuine break in the primary REST contract for exactly the scenario AC-8
exists to prove; not data-destructive — the run is still correctly recorded `failed` in the graph
and fully readable via a follow-up `GET`, so there is a workaround and no zombie/corrupt state).

**Steps to reproduce:**
1. Publish a `process`-kind def whose start step names a role that resolves at publish time.
2. Edit the overlay to remove that role, and restart the server process (config is read once at
   wiring time — FR-15, no reload path).
3. `POST /workflow-runs` to start a **fresh** run of that def (never run before under the old
   overlay).

**Expected** (per `services._drive_or_fault`'s own docstring, `services.py:1562-1600`, and the
REST route's documented error map in `api.py`'s §12 comment block: *"A fault during the drive is
NOT an error status: the run is already correctly terminal in the graph, so it comes back 201/200
carrying `{"status": "failed", "error": …}`, not a traceback"*): a `201` (or `200`) response body
`{"status": "failed", "error": "..."}`.

**Actual:** a raw `500 Internal Server Error`, `Content-Type: text/plain`, body is a bare Python
traceback ending in:
```
falkorchat.modelconfig.ModelResolutionError: unknown role 'vanishRole' (kind='step') —
declare it in .../models-v2.json's roles map, or use '<provider>/<model-id>'
```
The run **is** correctly written as `status: "failed"` with the diagnostic message
(`ctx: {"error": "unexpected: ModelResolutionError(...)"}`) and `atStepKey: null` — confirmed by a
follow-up `GET /workflow-runs/{id}`, which returns a clean `200` with exactly that state. No
`StepRun` was ever written for the unresolvable step (`GET .../step-runs` → `[]`), so "no other
model used in its place" holds regardless of the REST-layer defect.

**Root cause:** `executor._drive`'s own fault net (`executor.py:415-424`) is a bare
`except Exception as exc:` — it correctly catches `ModelResolutionError`, stamps `fail_run` with a
diagnostic note, and re-raises. But the layer above it, `services._drive_or_fault`
(`services.py:1601-1609`), which is what turns that re-raised exception into the clean JSON
envelope for the synchronous REST paths (`start_workflow_run`/`submit_workflow_input`), catches
**only** `(NotImplementedError, WorkflowConfigError)` — a pair that predates L2-5's introduction of
`ModelResolutionError` as a third drive-time fault class reaching this same net.
`_drive_or_fault`'s own docstring states its narrow-catch limit is deliberately for "faults raised
*before* anything is written" (budget exhaustion, an early-bail case) — but `ModelResolutionError`
is not that case: `fail_run` has already landed by the time it reaches this handler, exactly the
scenario the docstring says should get the clean envelope. The catch-list was simply never
extended to include the new exception type Landing 2 introduced.

**Isolation:** the run's own graph state is correct in every particular (confirmed above via
`GET`), isolating the defect to the REST layer's exception-to-envelope translation, not to the
executor's fault-containment logic itself (which — per the coordination log's own note on L2-5 —
was believed to already be correct and untouched; it is, for the *graph* half of the contract, but
the REST-envelope half of `_drive_or_fault` was not extended alongside it).

**Likely also affects** (by code inspection, not independently reproduced live in this pass, since
none of this pass's defs used a parking `human`/`wait` step): `POST /workflow-runs/{id}/input`
(`submit_workflow_input`), which routes through the same `_drive_or_fault` helper — a resume that
hits a drive-time `ModelResolutionError` (or any other exception type outside the two-item
allowlist) would very likely exhibit the identical `500`.

**Suggested fix:** add `ModelResolutionError` (imported from `modelconfig`) to
`_drive_or_fault`'s caught-exception tuple in `services.py`. Cheap, single-line, and the docstring's
own stated intent already covers this case — it is a gap in the *set*, not in the *design*.

## Environment notes

- **A misconfiguration in this pass's own setup, self-diagnosing, not a product defect.** The
  first attempt at Phase A started the server with `FALKORCHAT_WORKFLOW_ENABLED=1` but
  `FALKORCHAT_ENABLE_AGENT=0`, on the assumption the workflow engine alone would wire a
  `ModelGateway`. Reading `app.py::_build_default_app` directly (`app.py:264`: `if not
  config.ENABLE_AGENT: return create_app(services)` — an **early return**, before the
  `WORKFLOW_ENABLED` branch is ever reached) showed this was wrong: the gateway, and therefore the
  FR-9 publish-time check, is wired only when `ENABLE_AGENT` is on. Under the misconfigured run,
  publishing a def naming an unresolvable model (`nope/badmodel`) returned `201`, not `400` — at
  first glance alarming, but the server log carried the exact, correctly-worded WARNING L2-4's own
  m-7 finding promises (*"no ModelGateway wired — skipping the FR-9 model-resolvability check for
  step 's1': 'nope/badmodel'"*), which is what caught the misconfiguration immediately rather than
  producing a silent false pass. Corrected to `ENABLE_AGENT=1` for the remainder of the pass; this
  is recorded as a positive confirmation of m-7's design intent, not a defect.
- **Pre-existing drift found at the start of this pass.** Before running anything of this pass's
  own, `./scripts/verify_workflows.sh acme` reported both `triage@v1` and `access-request@v1`
  **missing from `reference`** (present in `ws:acme`'s snapshot, so no live run was at risk — a
  split-brain, not data loss). This was already the state at the very first check, before this
  pass had published, run, or torn down anything — consistent with the known `pytest`
  `wf_repo`-fixture hazard (`falkor-chat/AGENTS.md`) from a prior session's test run that was never
  followed by a reseed. Repaired immediately via `./scripts/seed_workflows.sh acme`, verified back
  in sync, **before** any of this pass's own def-publish/run activity, so it could not have
  contaminated any TP2 result. Re-triggered a second time by this pass's own TP2-014 (`pytest -q`,
  the same documented hazard) and repaired again at teardown.
- **`reference` gained six permanent `WorkflowDef` nodes** (`qak042-ac4`, `qak042-ac6`, `qak042-ac8`,
  `qak042-ac9`, `qak042-ac10step`, `qak042-ac10guard`) from this pass's own publishing — global-graph
  residue with no unpublish mechanism in the product, the same category of harmless-but-permanent
  artifact any `pytest`/dev session already leaves in `reference` today. Not cleaned up (nothing to
  clean up with); does not collide with `triage`/`access-request` or any other named def.

## Coverage & gaps

**Covered, with real execution evidence:** AC-4 (trace half), AC-6, AC-7 (including the M-4
ordering), AC-8, AC-9, AC-10 (all four kinds — `step`, `guard`, `agent`, `embedding`), AC-11 — all
driven against the real running server, real FalkorDB, and real LM Studio, with two genuine
process restarts (AC-6/AC-8) and one genuinely-unreachable network endpoint (AC-9, AC-10's
reachability-contrast technique).

**Deliberately not covered:** AC-2/AC-3 end-to-end (no cloud API key, unchanged Landing-1 carve-out,
not reopened); `compose.yaml`/`Dockerfile` container-build verification (no Docker in this
pipeline, a pre-existing, already-recorded gap); the U9 gate's non-gating minor (no REST/admin
write path for `write_model_overrides`) — used exactly as documented (a direct
`Repository.write_model_overrides` call) as the legitimate test-setup mechanism the QA brief
authorized, not re-filed as a new finding.

**Method note.** AC-10's `guard`/`agent`/`embedding`-kind proofs use a reachability contrast
(explicit/default choice → a genuinely unreachable endpoint; override → a genuinely reachable one)
rather than Landing 1's logging-reverse-proxy technique, since none of those three kinds carry a
persisted trace field to read directly. This is judged sufficient because the override precedence
algorithm itself is already unit-and-mutation-tested per kind (the U9 `analyst` gate); this pass's
job was confirming the override reaches the real, wired consumer at each of the four kinds'
resolution points end-to-end, which a clean success/fast-failure contrast proves without further
machinery. `submit_workflow_input` (the resume path) was not independently driven live — none of
this pass's defs used a parking `human`/`wait` step — so D-2's "likely also affects" claim there is
by code inspection, clearly labeled as such, not directly observed.

## Feedback & recommendations

1. **Fix D-2** — a one-line addition to `_drive_or_fault`'s caught-exception tuple
   (`services.py`), since the docstring's own stated intent already covers this exact case.
2. **The reachability-contrast technique (§5 of the test plan) is worth keeping** as a documented
   QA technique alongside Landing 1's logging-proxy one: for any kind with no persisted trace,
   pointing the "loses" arm at a genuinely unreachable endpoint and the "wins" arm at a genuinely
   reachable one turns "did the override reach this consumer" into a fast, unambiguous
   success/failure signal with zero extra infrastructure.
3. **AC-9's fallback is fast** (6.3s wall time for the full two-step run, including the failed
   first attempt) — the unreachable LAN host fails via `ConnectionRefusedError`, not a hung
   timeout, so the fallback chain's user-facing latency cost in this environment is negligible.
4. **`_drive_or_fault`'s catch-list is the one place in this feature where a new exception type can
   silently regress the REST contract without any test noticing** unless that test happens to drive
   the exact synchronous REST path (not the chat-triggered background path, and not a direct
   `executor`/`services` call) with the exact new exception type. Worth a standing note for any
   future unit that introduces a new drive-time exception class.

## Deviations from the QA brief

- AC-10's `agent`/`embedding`-kind proofs used the reachability-contrast technique instead of
  Landing 1's logging-reverse-proxy (§5 above) — a scope/tooling trade-off, not a blocker
  encountered; both kinds are still proven end-to-end through the real, wired production consumer.
- AC-8's setup combined the required overlay edit (removing `vanishRole`) with AC-6's required
  overlay edit (remapping `roleA`) into a single restart cycle, rather than two separate restarts —
  reduces process churn without weakening either AC's proof, since the two edits are
  independent and don't interact.

## Environment teardown

`ws:qak042l2` and `ws:qak042l2dim4` both deleted (`GRAPH.DELETE`) at the end of the run. `reference`,
`ws:acme`, `ws:test` confirmed present; `reference`/`ws:acme` reconfirmed in sync
(`./scripts/verify_workflows.sh acme` → `RESULT: OK`) after repairing both the pre-existing drift
found at pass start and the drift `pytest -q` (TP2-014) itself re-triggered. `falkordb-dev`
container never stopped/restarted. All server instances started for this pass (four, across
Phases A/B/C/D — differing `FALKORCHAT_WS_ID`/`ENABLE_AGENT`/`WORKFLOW_ENABLED`/overlay
combinations) were stopped; no process from this pass was left running. All per-kind workspace
overrides written for AC-10's setup were cleared (`write_model_overrides(..., agent=None,
guard=None, embedding=None, responder=None)`) before teardown.
