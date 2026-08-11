# LLM Provider & Model Configuration — Landing 2 Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-042 (M4) · **Extends:** `docs/test-plans/llm-provider-config.md`

## 1. Scope & objective

Acceptance pass for **Landing 2** of the LLM provider/model configuration feature (K-042),
executed against the **running system** (real FalkorDB, real LM Studio, real HTTP transport) —
the first black-box pass on Landing 2, mirroring Landing 1's own acceptance pass (unit U6,
`docs/test-plans/llm-provider-config.md`). This is the terminal gate for Landing 2: there is no
further `analyst` re-review after this pass.

Unit U7 of `docs/plans/llm-provider-config-coordination.md`. Landing 2 (L2-1..L2-7) is fully
implemented, independently code-reviewed (no blockers across five gates), and committed:
`17c20dc`, `0801b3c`, `eb1a60f`, `44494d5`, `c4cf5ad`.

**In scope:**

- **AC-6** — role re-map + restart, no republish.
- **AC-7** — publish-time rejection of an unresolvable model/role, naming the step and
  identifier; the M-4 ordering (a def that both breaks topology and names an unresolvable model
  returns 409, not 400).
- **AC-8** — a model that resolves at publish but is unresolvable at drive time (no fallback
  chain) fails the run loudly, naming what failed, with no other model substituted.
- **AC-9** — an ordered fallback chain whose first model is unreachable; the next model answers
  and the trace records it, with `modelFallback = true`.
- **AC-10** — a per-workspace model override is a hard cap over an explicit choice, for **all
  four consumer kinds** — `step`, `guard`, `agent`, `embedding` (the reason B-1 was scoped the
  way it was).
- **AC-11** — an embedding-dimension mismatch refuses loudly, pre-flight, before any vector is
  written.
- **AC-4, the trace half** — a normal (non-debug, `trace:false`) run with two steps naming
  different models shows two different `resolvedModel` values on `GET
  /workflow-runs/{id}/step-runs` (Landing 1's own pass, TP-005, only proved the Landing-1-buildable
  "two different consumer kinds" half; this is the "two steps in one run, on the durable trace"
  half FR-8/`-graph.md` exist to guarantee).

**Out of scope / not reopened:**

- **AC-2/AC-3** — remain deferred/model-gated (Landing 1's stakeholder decision; Landing 2 adds
  no new AC-2/AC-3 scope).
- **`compose.yaml`/`Dockerfile`** container-build verification — no Docker in this pipeline,
  already a recorded residual gap.
- **The M-9 non-gating minor from U9's gate** (no REST/admin write path for
  `write_model_overrides`) — known, recorded, not re-filed. AC-10's setup below uses the same
  direct-repository-call mechanism the U9 gate itself used, per the coordination brief.

## 2. References

- `falkor-chat/docs/requirements/llm-provider-config.md` — AC-6..AC-11, AC-4 (ground truth).
- `falkor-chat/docs/plans/llm-provider-config.md` (`Version: 4`) §7 (L2-1..L2-7), §8 (graph
  interface), §10 (AC → landing → verification map).
- `falkor-chat/docs/plans/llm-provider-config-graph.md` v4 — `WorkspaceConfig` schema, the
  `write_model_overrides`/`read_model_overrides`/`read_index_dimension` Cypher.
- `falkor-chat/docs/QUERIES.md` §13.1/§13.2 — the override write/read Cypher used for AC-10 setup.
- `falkor-chat/docs/plans/llm-provider-config-coordination.md` — ledger, ruled findings, the U9
  non-gating-minor writeup.
- `falkor-chat/docs/test-plans/llm-provider-config.md` / `-report.md` — Landing 1's own pass
  (methodology precedent: real infra, throwaway workspace, direct-call evidence for
  hard-to-orchestrate kinds).
- Code under test: `server/falkorchat/modelconfig.py`, `executor.py`, `services.py`,
  `repository.py`, `embedding.py`, `guards.py`, `responder.py`, `app.py`.

## 3. Risk assessment

| Risk | Why it matters | Coverage |
|---|---|---|
| The `guard` kind's workspace carrier (B-1) is the single most expensive finding in this feature's design phase — a regression here is invisible unless specifically driven | AC-10 explicitly names `guard`, not just `step`/`agent` | AC-10 guard-kind item, driven through a real llm-guard transition |
| AC-8's drive-time-unresolvable scenario is easy to mis-test as publish-time rejection (already covered, different AC) | The two ACs are adjacent and easy to conflate | AC-8 item's setup deliberately separates "resolves at publish" from "unresolvable at drive time" via a restart between the two states |
| FR-15's "no live reload" is asserted design, never proven live | AC-6/AC-8 both depend on it | Both items include a genuine process restart between writing the old and new overlay states |
| AC-4's trace half is the one property distinguishing Landing 2's design from a `TraceEvent`-based alternative that would silently vanish on non-debug runs | It's the core reason `-graph.md` exists | Driven with `trace:false` explicitly, not the debug/trace path |
| AC-9's fallback needs a genuinely unreachable endpoint, not a simulated one | A mocked "unreachable" proves nothing about the real B-2 exception ladder | Uses the real declared LAN endpoint, independently reconfirmed unreachable this session |
| AC-11 depends on a real dimension mismatch existing | A contrived mismatch might not match how the guard is actually gated (label-scoped, m-4) | Uses a throwaway workspace bootstrapped at a genuinely different dimension from the configured embedding model's declared `dim`, then a plain `Message` post (the actual write path) |

**Deliberately not tested / limited:**

- A live hosted-cloud-provider call (AC-2/AC-3): no API key in this environment, unchanged from
  Landing 1's carve-out.
- `compose.yaml`/`Dockerfile` container build: no Docker in this environment.

## 4. Environment & data setup

- FalkorDB: existing `falkordb-dev` container, not restarted.
- LM Studio: real instance at `localhost:1234`, confirmed reachable, with multiple distinct chat
  models loaded (`qwen/qwen3-4b-2507`, `qwen/qwen3-4b-thinking-2507`,
  `mistralai_ministral-3-3b-instruct-2512`) and two distinct embedding models
  (`text-embedding-qwen3-embedding-0.6b`, `text-embedding-nomic-embed-text-v1.5`) — enough
  concrete distinctness to prove "two different models answered" without inference.
- The stakeholder's declared LAN endpoint (`http://192.168.0.69:1234`) independently
  re-confirmed **unreachable** from this WSL box this session (`curl -m 5` → connection refused /
  timeout) — used as the "genuinely unreachable first element" for AC-9's fallback chain and as
  the "explicit/default choice that must lose" arm of AC-10's override tests (see §5's technique
  note).
- Two throwaway workspaces, bootstrapped and deleted at teardown, `reference`/`ws:acme`/`ws:test`
  never written by anything in this pass except the necessarily-global `reference` def-publish
  calls (see §6's residual-state note):
  - `ws:qak042l2` — `EMBEDDING_DIM=1024` (matches `config/models.json`'s shipped embedding
    model's declared `dim`) — used for AC-4, AC-6, AC-7, AC-8, AC-9, AC-10 (all four kinds).
  - `ws:qak042l2dim4` — `EMBEDDING_DIM=4` — dedicated to AC-11's mismatch, deliberately separate
    from `ws:test`'s own dim-4 fixture per the brief's caution against touching shared test state.
- QA-authored config fixtures under `/tmp/qa-k042-l2/` (not committed, not part of the
  deliverable): a shared `opencode.json` declaring two providers (`lmstudio` → real
  `localhost:1234/v1`; `lan` → the declared-unreachable LAN host) and three overlay variants
  (`models-v1.json`/`-v2.json`/`-v3.json`) covering the pre-/post-restart role states and the
  "defaults point at the unreachable provider" state used for AC-10's agent/embedding-kind proof.
- The real production server (`uvicorn falkorchat.app:app`), started fresh for each phase that
  needs a config change or a different `FALKORCHAT_WS_ID`/enablement combination — never the
  offline test client. `FALKORCHAT_ENABLE_AGENT=1` is required (not just `WORKFLOW_ENABLED=1`)
  for the `ModelGateway` to be wired at all (`app.py::_build_default_app`, confirmed by reading
  the code directly after an initial misconfigured run silently skipped the FR-9 check — see the
  test report's environment notes).

## 5. Technique note: proving an override/fallback "won" without a trace field

`StepRun.resolvedModel`/`modelSource`/`modelFallback` exist only for the `step` kind (FR-8). For
`guard`, `agent` and `embedding` — which AC-10 explicitly requires covering — there is no
persisted trace of which model answered. Rather than build Landing 1's logging-reverse-proxy
apparatus (viable, but heavier machinery than needed here), this pass uses a **reachability
contrast**: the *explicit/default* choice is pointed at the declared-unreachable LAN provider,
and the *override* is pointed at the real, reachable `localhost:1234` provider. If the override
is honored, the call **succeeds** (a message gets embedded, a guard transition fires, an `@mention`
gets an actual reply); if it is not honored, the call fails fast with a `ProviderCallError` naming
the unreachable URL — an unambiguous, pre-existing observable requiring no new infrastructure.
This is judged sufficient because FR-16/FR-17's precedence logic itself is already
unit-and-mutation-tested per kind (U9's gate); this pass's job is confirming the override reaches
the **real wired consumer** end-to-end, not re-proving the precedence algorithm in isolation.

## 6. Test items

| ID | AC | Preconditions | Steps | Expected result | Priority |
|---|---|---|---|---|---|
| TP2-001 | AC-4 (trace half) | `ws:qak042l2`; a 2-step process def, each step naming a distinct concrete model, published + materialized | `POST /workflow-runs` with `trace:false` (default); `GET .../step-runs` | Run `status:"done"`; step-runs shows two rows with two different `resolvedModel` values, each matching its step's declared model | high |
| TP2-002 | AC-7 (negative) | A def naming an undeclared provider (`nope/badmodel`) on a step | `POST /workflow-defs` | 400, `WorkflowDefSpecError` naming the step key and the offending identifier; nothing published | high |
| TP2-003 | AC-7 (positive) | A def naming a resolvable role | `POST /workflow-defs` | 201 | high |
| TP2-004 | AC-7 (M-4 ordering) | An already-published def (from TP2-001) | Republish the **same key/version** with different topology **and** an unresolvable model on the new step | 409 `WorkflowDefConflictError` (topology), not 400 — proves the ordering | high |
| TP2-005 | AC-6 | Overlay v1 declares role `roleA -> [modelX]`; a 1-step-plus-terminal def naming `roleA`, published + materialized | Run #1 against overlay v1; observe `resolvedModel` | `resolvedModel == modelX` | high |
| TP2-006 | AC-6 (the point of the AC) | Overlay edited to `roleA -> [modelY]` (v2); server **process restarted** against v2, same def, **not republished** | Run #2 against overlay v2; observe `resolvedModel` | `resolvedModel == modelY`, with no republish between runs | high |
| TP2-007 | AC-8 | Overlay v1 declares role `vanishRole -> [modelX]`; a def naming `vanishRole` published (resolves fine at publish) but **not yet run**; overlay v2 removes `vanishRole` entirely; server restarted against v2 | Start a fresh run of the def (first run ever, now against v2) | Run terminates `status:"failed"`, an error naming the unresolvable identifier is recorded on the run, `atStepKey` cleared, `step-runs` empty (no model — fallback or otherwise — was used) | high |
| TP2-008 | AC-9 | Overlay declares role `fallbackRole -> [lan/<model> (unreachable), lmstudio/<model> (reachable)]`; a def naming `fallbackRole` | Run the def; observe `resolvedModel`/`modelFallback` | `resolvedModel` is the **second** chain element; `modelFallback == true` | high |
| TP2-009 | AC-10, `step` kind | Workspace override `agent=<modelZ>` written via `Repository.write_model_overrides` (crosswalk: this property governs the executor's `step` kind); a def step explicitly names a different model | Run the def; observe `resolvedModel`/`modelSource` | `resolvedModel == modelZ` (the override), `modelSource == "workspace"`, beating the step's own explicit choice | high |
| TP2-010 | AC-10, `guard` kind | Workspace override `guard=<reachable model>`; a def with an `{"kind":"llm"}` guard transition whose own declared `model` is the unreachable LAN ref | Run the def | Run completes (`status:"done"`, both steps executed) — the guard call reached the reachable override model, not the unreachable declared one (§5 technique) | high |
| TP2-011 | AC-10, `embedding` kind | Overlay's `defaults.embedding` points at the unreachable LAN provider; workspace override `embedding=<reachable model>` (dimension matching the workspace's index) | Post a plain message (embeds regardless of `@mention`); check `Message.embedding IS NOT NULL` | With the override set: embedding succeeds. Cleared: embedding fails with a `ProviderCallError` naming the unreachable URL, no vector written (§5 technique, both directions) | high |
| TP2-012 | AC-10, `agent` kind | Overlay's `defaults.agent` points at the unreachable LAN provider; workspace overrides `responder=<reachable model>` **and** `embedding=<reachable model>` (the responder also embeds the query for retrieval); a demo agent/channel/thread seeded | `@mention` the agent | An actual assistant reply is posted in the thread, sourced from the override model, not the unreachable default | high |
| TP2-013 | AC-11 | `ws:qak042l2dim4` (index dimension 4); overlay's embedding model declares `dim:1024` | Post a plain message | Embedding refused **pre-flight** (`EmbeddingDimensionError`, before any HTTP call), naming both dimensions (4 vs 1024), the model ref and the workspace; `Message.embedding` stays null | high |
| TP2-014 | baseline | venv ready | `.venv/bin/python -m pytest -q` from `server/` | All tests pass; count independently reproduced, not cited from the coordination doc | high |
| TP2-015 | doc/script cross-check | none | Unfiltered `grep` for the four legacy var names across the repo (excluding `.git/`/`.venv/`/`__pycache__/`/`docs/archive/`) | No new operational reference beyond what Landing 1's TP-010 already found clean (tripwire's own list, historical/pointer prose, already-amended `local-model-ram-budget-ml.md`) | medium |

## 7. Entry / exit criteria

**Entry:** FalkorDB and LM Studio reachable; Landing 2 committed and gated per the coordination
ledger; `git status` clean.

**Exit:** every in-scope TP item has a recorded pass/fail outcome with evidence in the test
report; every defect found is filed with severity; no destructive action taken against
`reference`/`ws:acme`/`ws:test`; any transient drift to shared state caused by this pass's own
tooling (e.g. an offline `pytest` run's known `reference`-wiping side effect) is detected and
repaired before sign-off.

## 8. Out of scope (restated)

AC-2/AC-3 (deferred, no cloud key); `compose.yaml`/`Dockerfile` container-build verification (no
Docker); the U9 gate's already-recorded non-gating minor (no REST/admin write path for workspace
overrides).
