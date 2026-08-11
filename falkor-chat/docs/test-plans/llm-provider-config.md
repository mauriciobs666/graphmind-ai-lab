# LLM Provider & Model Configuration — Landing 1 Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-042 (M4) · **Extended by:** `docs/test-plans/llm-provider-config2.md`

## 1. Scope & objective

Acceptance pass for **Landing 1** of the LLM provider/model configuration feature (K-042),
executed against the **running system** (real FalkorDB, real LM Studio, real HTTP transport) —
the first black-box pass on this feature; the two prior gates
(`docs/reviews/llm-provider-config.md` Pass 1-3, `## Landing 1 code review`) were static (design
docs, then a diff review).

Unit U6 of `docs/plans/llm-provider-config-coordination.md`. Landing 1 is committed at `a2b8aa9`;
tree is clean at `595cc70` (HEAD at plan time). Full offline suite reported 791 passed/1
deselected — re-verified independently in TP-009 below, not trusted on report.

**In scope (per the coordination doc's U6 row and stakeholder decisions):**

- **AC-1** — an existing, unedited `opencode.json` works.
- **AC-4, partial** — the Landing-1-buildable slice: two different **consumer kinds** (not two
  steps within one workflow — that needs Landing 2's trace visibility) end up calling different
  concrete models when configured that way.
- **AC-5** — agent / llm-guard / embedding-worker each default per kind when no model is named,
  and the three defaults may differ.
- **AC-12** — a per-kind timeout override is honored.
- **AC-13** — a legacy env var with no config file present is a loud startup failure, not silent
  operation.
- **AC-2, AC-3 — structural only** (no cloud API key in this environment, stakeholder decision
  2026-08-10 #3): confirm the design supports secret substitution and three provider kinds by
  reading code/config and exercising the parse/resolve path offline; no live cloud call.

**Out of scope — Landing 2, not yet implemented, absence is expected, not a defect:** AC-6, AC-7,
AC-8, AC-9, AC-10, AC-11.

## 2. References

- `falkor-chat/docs/requirements/llm-provider-config.md` — FR-1..FR-20, AC-1..AC-13 (ground truth).
- `falkor-chat/docs/plans/llm-provider-config.md` v3 §6 (L1-1..L1-6) — Landing 1 implementation plan.
- `falkor-chat/docs/plans/llm-provider-config-graph.md` v3 — graph-side design (Landing 2, not built yet).
- `falkor-chat/docs/reviews/llm-provider-config.md` v4 — design + diff-scoped review history.
- `falkor-chat/docs/plans/llm-provider-config-coordination.md` — ledger, environment facts, U6 scope.
- Code under test: `server/falkorchat/modelconfig.py`, `transport.py`, `llm.py`, `embedding.py`,
  `app.py`, `executor.py`, `guards.py`, `responder.py`, `tools.py`, `config.py`.

## 3. Risk assessment

| Risk | Why it matters | Coverage |
|---|---|---|
| Static review said "sound" but the real LM Studio/network stack behaves differently than mocked tests assume | Two review gates were static/diff-only; this is the first execution against real infra | All TP items drive the real server + real LM Studio |
| AC-13's tripwire is the safety net for FR-20's "replaced, not fallback" — a gap here means silent misconfiguration in production | Explicitly named as high-value by the coordination doc and the code review (Major 1, since closed) | TP-002, TP-003 |
| Per-kind defaults/timeouts (AC-5/AC-12) are the entire value proposition of Landing 1 | It's the "bigger sting" driver from requirements | TP-004, TP-005, TP-006 |
| Secret hygiene (AC-2) — a leaked credential is a security defect, not a functional one | `Secret` wrapper is a hand-rolled invariant, not framework-enforced | TP-007 |
| Env-var cutover touches many files (scripts, docs, `.env.example`) — a stale doc/script could mislead an operator | FR-20 blast radius spans 10+ files | TP-010 (doc/script cross-check) |

**Deliberately not tested / limited:**
- AC-6..AC-11: no implementation exists (Landing 2). Probing for absence is not useful; confirmed
  by reading the plan/graph note only.
- A live hosted-cloud-provider call (AC-2/AC-3 full): no API key available in this environment
  (stakeholder decision). Structural verification only.
- Concurrency/load on the resolver: `ModelGateway.resolve()` is a pure offline lookup after
  construction (per plan §3); no shared mutable state across calls in Landing 1 (Landing 2's
  `FallbackClient` mutable-state hazard, M-5, does not exist yet). Not a Landing-1 risk.
- `compose.yaml`/`Dockerfile` container build: no Docker available in this environment either
  (same gap `teco` flagged at U4). Out of reach for this pass; flagged as a residual gap in §"gaps".

## 4. Test items

Evidence method note: TP-004/005/006 drive the real `ModelGateway`/`OpenAICompatibleLLM`/
`OpenAICompatibleEmbedder` production code against the real LM Studio through a small
**logging reverse proxy** (`localhost:18234 → localhost:1234`, transparent pass-through, logs each
request's `model` field) so the exact model id each call used is directly observable evidence
rather than inference. This is real network I/O against the real backend — nothing in the
resolution/transport layers is mocked. TP-004/005 additionally drive the two run-less consumers
(embedding worker, agent responder) through genuine REST calls against a live server process;
TP-006's guard/step calls and the timeout manipulation are driven by invoking the real,
running-server-equivalent `ModelGateway.from_env()` directly (same env vars, same files, same
transport) — full workflow-run orchestration for a guard call was assessed as disproportionate
machinery for the same signal and is noted as a deliberate scope trade-off, not a gap in what's
verified.

| ID | Title | Preconditions | Steps | Expected result | Priority | Type |
|---|---|---|---|---|---|---|
| TP-001 | AC-1: unedited real `opencode.json` works | Real FalkorDB up; LM Studio up; `opencode/agents/severino/opencode.json` present, unmodified | 1. Hash the file. 2. Start the real server (`uvicorn falkorchat.app:app`) with `FALKORCHAT_OPENCODE_CONFIG` pointed at that file, `FALKORCHAT_ENABLE_AGENT=1`, throwaway `FALKORCHAT_WS_ID`. 3. `GET /health`. 4. Drive a real call that resolves through the file's declared provider (post a message to trigger the embedding worker). 5. Re-hash the file. | Server starts; `/health` 200; the embedding call succeeds (provider `lmstudio` resolved from the file); file hash unchanged (OpenCode would still read it unmodified) | high | acceptance |
| TP-002 | AC-13a: legacy var set, no config file at all | Real FalkorDB up | Start the server with `FALKORCHAT_LLM_MODEL` set, `FALKORCHAT_ENABLE_AGENT=1`, no `FALKORCHAT_OPENCODE_CONFIG` | Startup fails immediately with an explicit error naming the legacy var(s) and the two replacement files — process does not bind a port, no silent fallback to legacy behavior | high | acceptance |
| TP-003 | AC-13b: legacy var set even with a *valid* config present | A valid `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` pair ready | Start the server with `FALKORCHAT_LLM_BASE_URL` set **and** a valid `FALKORCHAT_OPENCODE_CONFIG` | Startup still fails with the same tripwire — proves "replaced" (FR-20), not "used only if the new config is missing" | high | acceptance |
| TP-004 | AC-5: agent / guard / embedding-worker default per kind, and the three differ | QA overlay declares three distinct concrete models for `agent`/`guard`/`embedding` kinds; server running against it through the proxy | 1. Post a message with no `@mention` (embedding-worker only). 2. Resolve kind `agent` via `ModelGateway.llm("agent")` and call `.complete()`. 3. Resolve kind `guard` via `.llm("guard")` and call `.complete()`. None of the three calls names an explicit model. | Proxy log shows three distinct `model` values, one per kind, exactly matching the overlay's three configured defaults | high | acceptance |
| TP-005 | AC-4 (partial): two different consumer *kinds* land on different concrete models | Same QA overlay/proxy setup as TP-004; `step` kind default also configured, distinct from `embedding`'s | 1. Resolve kind `step` via `ModelGateway.llm("step")`, call `.complete()`. 2. Post a message (embedding-worker call, kind `embedding`). | Proxy log shows the `step` call and the `embedding` call used two different `model` values, each matching that kind's configured default | high | acceptance |
| TP-006 | AC-12: a per-kind timeout override is honored | QA overlay B: `timeouts.guard` set to an unrealistically small value (e.g. `0.01`s); QA overlay C: same but `timeouts.guard = 60` | 1. With overlay B loaded, resolve kind `guard`, call `.complete()`. 2. With overlay C loaded (same call, same model, same network path), repeat. | Call 1 fails with a `ProviderCallError` naming a timeout cause within roughly the configured window (not the 180s kind-timeout floor); call 2 succeeds normally — proving the override, not a fixed default, governs the call's allowed time | high | acceptance |
| TP-007 | AC-2 (structural): `{env:}`/`{file:}` secret substitution, no literal leak | none (offline) | 1. Set a throwaway env var to a fake secret value. 2. Load a provider whose `apiKey` is `{env:THAT_VAR}` through `ProviderCatalog`/`ModelGateway`. 3. Inspect the resulting `ProviderSpec`/`ResolvedModel` `repr()`, and the `Authorization` header the transport would send. 4. Grep the overlay/shared files and any startup log captured in TP-001/004 for the literal secret value. | The resolved header carries `Bearer <value>` (substitution worked); `repr()` of both dataclasses never contains the literal value; the literal never appears in any config file on disk or in captured log output | medium | structural |
| TP-008 | AC-3 (structural): three provider kinds parse and resolve uniformly | `config/opencode.example.json` (shipped, already declares local/LAN/cloud) | Load it with `ProviderCatalog`/`Overlay`/`ModelGateway` offline; resolve one ref from each of the three providers (`lmstudio`, `lan-host`, `openai`) | All three resolve to a `ResolvedModel` with `protocol="openai"` and a normalized `base_url`, with no live network call attempted | medium | structural |
| TP-009 | Baseline: full offline suite is green | venv ready | `.venv/bin/python -m pytest -q` from `server/` | All tests pass; deselected count matches what the diff-scoped gate reported (own independent run, not trusted from the coordination doc) | high | regression |
| TP-010 | Doc/script cross-check: no surviving reference to the replaced env vars as *the* way to configure a model | none | `grep -rn` (unfiltered, per the coordination doc's own method-note fix) for the four legacy var names across `README.md`, `AGENTS.md`, `scripts/start_server.sh`, `server/.env.example`, `docs/DESIGN.md` | Every hit is either the tripwire's own list (`config.py`) or an explicit historical/negative reference (e.g. "no longer an env var"); no doc instructs an operator to set one to configure a model | medium | structural |

## 5. Environment & data setup

- FalkorDB: existing `falkordb-dev` container, not restarted; a throwaway workspace
  (`FALKORCHAT_WS_ID=qa-k042`) bootstrapped and torn down (`GRAPH.DELETE ws:qa-k042`) at the end —
  `reference` and `ws:acme` are never written.
- LM Studio: real instance at `localhost:1234`, models already loaded verified reachable
  (`qwen/qwen3-4b-2507`, `qwen/qwen3-4b-thinking-2507`, `mistralai_ministral-3-3b-instruct-2512`,
  `text-embedding-qwen3-embedding-0.6b`, among others) — used to give the agent/step/guard/embedding
  kinds four genuinely distinct, genuinely loaded models.
- A throwaway logging reverse proxy (`localhost:18234`), plain Python `http.server`, forwards to
  `localhost:1234` unchanged and logs each POST body's `model` field to a file — used only to make
  "which model answered" observable evidence; it never fabricates or alters a response.
- QA-authored config fixtures under `/tmp/qa-k042/` (not committed, not part of the deliverable):
  a proxy-routed `opencode.json` and two/three `models.json` overlay variants (distinct per-kind
  defaults; the two timeout variants for TP-006).
- TP-001 uses the real, in-repo `opencode/agents/severino/opencode.json` directly (unmodified) —
  not a copy — to keep AC-1's "no edit to that file" claim genuine.

## 6. Entry / exit criteria

**Entry:** offline suite green (own re-run, TP-009), FalkorDB and LM Studio reachable, `git
status` clean at commit `a2b8aa9`/`595cc70`.

**Exit:** every in-scope TP item has a recorded pass/fail/blocked outcome with evidence in the
test report; every defect found is filed with severity; no destructive action taken against
`reference` or any pre-existing workspace.

## 7. Out of scope (restated)

AC-6, AC-7, AC-8, AC-9, AC-10, AC-11 (Landing 2 — roles, publish-time resolvability, no-fallback
suspend, fallback chains, workspace override, embedding-dimension guard); a live hosted-cloud
provider call; `docker build`/`docker compose` verification of `compose.yaml`/`Dockerfile`.
