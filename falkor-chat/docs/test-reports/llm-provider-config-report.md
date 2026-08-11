# LLM Provider & Model Configuration — Landing 1 Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-042 (M4)

Executes `docs/test-plans/llm-provider-config.md` (TP-001..TP-010). Against commit `595cc70`
(HEAD; Landing 1 implementation committed at `a2b8aa9`), FalkorDB `falkordb-dev` (untouched, still
"Up 13 hours" throughout), LM Studio at `localhost:1234` (real, live). Run date 2026-08-11.

## Summary

**Verdict: PASS, with one defect (Minor) and one residual gap (environment, not code) to note.**
All nine in-scope acceptance criteria hold when driven against the real, running system — the
first execution-based confirmation of this feature, distinct from the two prior static/diff-scoped
`analyst` gates. One defect found: the shipped cloud-provider example in
`config/opencode.example.json` cannot itself be resolved (missing `options.baseURL`) — self-
diagnosing, one-line fix, does not affect the resolver logic itself (confirmed by isolating the
fix and re-testing). No destructive action was taken against `reference`, `ws:acme`, or `ws:test`;
all QA state lived in a throwaway `ws:qa-k042`, deleted at teardown.

## Results table

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-001 | AC-1 | **PASS** | Real `opencode/agents/severino/opencode.json` (unedited) → server started, `/health` 200, a real post-message call resolved+embedded successfully with no error in the server log. `sha256sum` before/after identical: `fa6b6a94...459305a5` |
| TP-002 | AC-13a | **PASS** | `FALKORCHAT_LLM_MODEL` set, no `FALKORCHAT_OPENCODE_CONFIG` → `RuntimeError: legacy model configuration env var(s) FALKORCHAT_LLM_MODEL are set, but K-042 replaced them with two config files...` raised at import time (the same path `uvicorn falkorchat.app:app` takes) |
| TP-003 | AC-13b | **PASS** | Same tripwire fires with `FALKORCHAT_LLM_BASE_URL` set **and** a valid `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` pair present — proves replacement, not conditional fallback |
| TP-004 | AC-5 | **PASS** | Live calls through a logging proxy (`localhost:18234→1234`) resolved: `agent`→`qwen/qwen3-4b-2507`, `guard`→`mistralai_ministral-3-3b-instruct-2512`, `embedding`→`text-embedding-qwen3-embedding-0.6b` — three distinct concrete models, none of the three calls named an explicit model |
| TP-005 | AC-4 (partial) | **PASS** | `step`→`qwen/qwen3-4b-thinking-2507` vs `embedding`→`text-embedding-qwen3-embedding-0.6b` — two different consumer kinds, two different concrete models, both via real network calls |
| TP-006 | AC-12 | **PASS** | `timeouts.guard=0.01` → `ProviderCallError: ...TimeoutError: timed out` raised after 0.024s; identical call with `timeouts.guard=60` → succeeded after 1.054s. The override, not the 180s kind-timeout floor, governed both |
| TP-007 | AC-2 (structural) | **PASS** | `{env:QA_FAKE_SECRET}` resolved to the real value inside the `Authorization: Bearer ...` header; `repr(ResolvedModel)`/`repr(ProviderSpec)` never contain the literal; literal absent from both QA config files on disk |
| TP-008 | AC-3 (structural) | **PASS, with a defect (see Defects)** | Shipped `config/opencode.example.json` declares `lmstudio`, `lan-host`, `openai` — all three parse; `lmstudio`/`lan-host` resolve cleanly (`protocol="openai"`, correctly normalized `base_url`); `openai` raises `ModelConfigError` (missing `baseURL`) until patched, then resolves identically/uniformly to the other two |
| TP-009 | baseline | **PASS** | `.venv/bin/python -m pytest -q` → `791 passed, 1 deselected in 9.42s` — reproduced independently, matches the coordination doc's prior report exactly |
| TP-010 | doc/script cross-check | **PASS** | Unfiltered grep for all four legacy var names across `README.md`, `AGENTS.md`, `scripts/start_server.sh`, `server/.env.example`, `docs/DESIGN.md` → zero hits |

## Defects

### D-1 — Minor: shipped `config/opencode.example.json`'s `openai` provider entry has no `options.baseURL`, so it cannot resolve as shipped

**Severity:** Minor (self-diagnosing error, one-line fix, no data/security impact, does not affect
the other two documented providers or the resolver logic itself).

**Steps to reproduce:**
1. Load the shipped `falkor-chat/config/opencode.example.json` with `ProviderCatalog`/`Overlay`/`ModelGateway`.
2. Resolve any ref against provider `openai` (e.g. `openai/gpt-4o-mini`).

**Expected:** the shipped, documented "cloud provider using `{env:}`" example (plan §5's file
list explicitly names this as its purpose) resolves like the other two providers in the same file.

**Actual:**
```
falkorchat.modelconfig.ModelConfigError: provider 'openai' has no options.baseURL in
../config/opencode.example.json and no overlay override in <inline>
```
`lmstudio` and `lan-host` in the same file resolve without incident.

**Root cause:** OpenCode's real `@ai-sdk/openai` package has its own implicit default base URL,
so a genuine OpenCode file may legitimately omit `options.baseURL` for provider `openai`.
`modelconfig._build_provider_spec` has no equivalent implicit default for any `npm` value — every
provider requires `options.baseURL` (from the shared file) or a `providers.<id>.baseURL` overlay
override, unconditionally (§4.2, `modelconfig.py:386-391`). The shipped example was authored
without this, so it doesn't practice what it documents.

**Isolation:** adding `"baseURL": "https://api.openai.com/v1"` to the same file's `openai` entry
and re-resolving succeeds and matches the other two providers exactly:
```
lmstudio/qwen/qwen3-4b-2507         -> protocol='openai' base_url='http://localhost:1234/v1'
lan-host/google/gemma-3n-e4b        -> protocol='openai' base_url='http://192.168.0.69:1234/v1'
openai/gpt-4o-mini                  -> protocol='openai' base_url='https://api.openai.com/v1'
```
This confirms the resolver logic itself is correct and uniform across all three provider kinds —
the defect is narrowly in the shipped fixture, not in `modelconfig.py`'s resolution behavior. AC-3
is therefore judged **structurally verified** (the underlying claim — three provider kinds parse
and resolve uniformly — holds), with this fixture gap called out separately rather than folded
into a fail.

**Suggested fix:** add `options.baseURL: "https://api.openai.com/v1"` to
`config/opencode.example.json`'s `openai` provider entry (one line), or add a short comment
explaining that falkor-chat, unlike OpenCode's own `@ai-sdk/openai` package, requires an explicit
`baseURL` for every provider with no implicit default. Either closes the gap; the former is
cheaper and keeps the example self-consistent with what it demonstrates.

## Coverage & gaps

**Covered, with real execution evidence:** AC-1, AC-4 (Landing-1-buildable slice), AC-5, AC-12,
AC-13 end-to-end against the real running server/`ModelGateway` and real LM Studio; AC-2, AC-3
structurally, per the stakeholder's no-cloud-key decision.

**Deliberately not covered (Landing 2, no implementation exists yet):** AC-6 (role remap without
republish), AC-7 (publish-time unresolvable-model rejection), AC-8 (no-fallback suspend), AC-9
(fallback chain), AC-10 (workspace override), AC-11 (embedding-dimension guard). Not probed — an
absence here is expected, not a finding.

**Method note, stated plainly per the QA brief:** TP-004/TP-005's `step` and `guard` kind calls,
and TP-006's timeout manipulation, were driven by invoking the real, production
`ModelGateway.from_env()` directly (same env vars, same config files, same transport, same real
network path to LM Studio through a logging proxy) rather than orchestrating a full multi-turn
workflow run to reach the `guard`/`step` resolution points through REST. Full workflow
orchestration (the seeded `triage` def's `intake`→`research`→`answer` conversation, with an LLM
judging when to advance) was assessed as disproportionate machinery for the same signal — it adds
non-determinism (an LLM must decide to advance a guard) without adding evidentiary value over a
direct call through the identical production code path. `agent` (TP-004) and `embedding`
(TP-004/005) kinds *were* driven end-to-end through real REST calls against a running server with
a background-task-scheduled responder/embedder, so the REST/background-task seam is not
unexercised.

**Residual gaps, unrelated to Landing 1's own correctness:**
- `compose.yaml`/`Dockerfile` (the two new config-file paths + read-only bind mount) remain
  unverified against a real `docker build`/`docker compose` — no Docker in this environment
  either, the same gap `coder`/`teco` already flagged at U4. Still open for whichever step next
  has Docker access.
- AC-2/AC-3's live hosted-cloud-provider call remains untested end-to-end — no API key available
  in this environment (stakeholder decision, 2026-08-10 #3). The structural verification in this
  report (TP-007/TP-008) is the full extent of what's checkable here.
- OpenCode's own continued ability to read the unedited `opencode/agents/severino/opencode.json`
  file (the second half of AC-1's claim) was not independently re-verified with the actual OpenCode
  CLI — the file's byte-for-byte-unchanged hash is the evidence offered; OpenCode was not invoked
  in this pass.

## Feedback & recommendations

1. **Fix D-1** before or alongside any Landing-2 documentation pass that touches
   `config/opencode.example.json` — cheap, and it's the one artifact meant to make the cloud-
   provider case concrete for an administrator.
2. **The logging-proxy technique used for TP-004/005/006 is worth keeping** as a documented QA
   technique for this feature: a transparent pass-through reverse proxy that logs each request's
   `model` field turns "which concrete model actually answered" from an inference into direct
   evidence, without needing Landing 2's `resolvedModel` trace property to exist yet. Worth reusing
   for the Landing 2 acceptance pass (U7) to cross-check the trace's claims against what was
   actually sent over the wire.
3. **No testability blockers found.** `ModelGateway.from_env()` is cleanly invokable outside the
   full FastAPI app, which is what made the direct-call evidence path for `step`/`guard`/timeout
   practical; this is a positive testability property worth preserving in Landing 2.
4. **AC-13's tripwire message is good QA-facing text** — both TP-002 and TP-003 got a message
   naming the exact variable set and both replacement files/paths on the first try, with no need
   to read source to interpret the failure.

## Deviations from the QA brief

- TP-004/005/006 used direct `ModelGateway` calls for `step`/`guard` kinds rather than full
  workflow-run orchestration, as detailed in "Coverage & gaps" above — a scope trade-off made for
  signal-per-cost, not a blocker encountered.
- No LM Studio model mismatch was encountered — the models the shipped `config/models.json` and
  this QA pass's overlay name (`qwen/qwen3-4b-2507`, `qwen/qwen3-4b-thinking-2507`,
  `mistralai_ministral-3-3b-instruct-2512`, `text-embedding-qwen3-embedding-0.6b`) were all present
  and loaded on the real LM Studio instance throughout.
- The server started cleanly with the shipped example configs in every scenario tried except the
  isolated `openai` provider case (D-1), which used a QA-authored fixture, not a shipped one.

## Environment teardown

`ws:qa-k042` deleted (`GRAPH.DELETE`) at the end of the run. `reference`, `ws:acme`, `ws:test`
confirmed present and unmodified (`GRAPH.LIST` before/after). `falkordb-dev` container never
stopped/restarted. The QA logging proxy and all server instances started for this pass were
stopped; no process from this pass was left running.
