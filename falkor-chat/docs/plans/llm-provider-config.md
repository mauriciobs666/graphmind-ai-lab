# LLM Provider & Model Configuration — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** K-042 (M4) · **Version:** 1

Turns `docs/requirements/llm-provider-config.md` (FR-1..FR-20, AC-1..AC-13) into an ordered,
two-landing build. Coordination record: `docs/plans/llm-provider-config-coordination.md`.
Graph-side mechanics (FR-8 trace recording, FR-16/FR-17 storage, FR-19 index dimension) are
owned by `docs/plans/llm-provider-config-graph.md` — this plan states the **interface** it needs
from that document and never invents the Cypher.

---

## 1. Goal & scope

**Goal.** Replace falkor-chat's four independently-constructed, env-var-configured LLM clients
with **one internal model-resolution seam** (`ModelGateway`) fed by **two hand-edited config
files**: a pristine, unmodified OpenCode `opencode.json` (providers only) plus a falkor-chat
overlay (per-kind defaults, per-model settings, later roles). Every consumer — the `@mention`
responder, the llm-guard judge, the workflow executor's agent nodes, the embedding worker —
resolves its model through that one seam, and any of them may name its own model.

**In scope:** FR-1..FR-20 across two landings (§6, §7).

**Out of scope** (from the requirements, restated so an implementer does not drift): editing
config through a UI/API; live reload; extending the `opencode.json` schema; interpreting
OpenCode's `agent.*` / tool / prompt blocks; cost or token accounting; streaming; re-embedding or
migrating a workspace when the embedding model changes; per-user model choice; adding a fifth LLM
consumer.

**Also out of scope, decided here:** a **native Anthropic Messages client** (`/v1/messages`,
`x-api-key`, non-OpenAI payload). See §4.7 — the file format, resolver and provider `protocol`
field fully support it, and an unsupported protocol fails loudly by name at startup rather than
sending a wrong-shaped payload. Anthropic is reachable in this build through its documented
OpenAI-SDK-compatibility base URL (`https://api.anthropic.com/v1/`).

---

## 2. Context & findings (verified 2026-08-10)

### 2.1 Where the four consumers are built today

All four are constructed in `server/falkorchat/app.py::_build_default_app` (`:245-303`), each
reading module constants from `server/falkorchat/config.py`:

| # | Consumer | Constructed at | Client | Config read |
|---|---|---|---|---|
| 1 | `@mention` AI responder | `app.py:270-273` → `responder.AgentResponder` | `LMStudioLLM()` + `LMStudioEmbedder()` | `config.LLM_BASE_URL`, `config.LLM_MODEL` (default args on `llm.LMStudioLLM.__init__`, `llm.py:99-100`) |
| 2 | llm-kind guard judge | `app.py:288` → `app._build_llm_judge(LMStudioLLM())` | `LMStudioLLM()` | same |
| 3 | executor agent nodes | `app.py:290` → `executor.WorkflowExecutor(llm=LMStudioLLM())`, used at `executor.py:585` (`self._llm.chat(...)`) | `LMStudioLLM()` | same |
| 4 | embedding worker | `app.py:269-271` → `embedding.EmbeddingWorker(repo, LMStudioEmbedder())`; the same embedder instance is also handed to `tools.build_builtin_registry` (`app.py:287`) and to `AgentResponder` | `LMStudioEmbedder()` | `config.EMBEDDING_BASE_URL`, `config.EMBEDDING_MODEL` (`embedding.py:59-60`), `config.EMBEDDING_DIM` (`embedding.py:84`) |

Three distinct `LMStudioLLM()` instances exist today and every one of them is the *same* model.
That is exactly what FR-4 targets.

### 2.2 The HTTP layer is duplicated and has no timeout

`llm._urllib_transport` (`llm.py:78-90`) and `embedding._urllib_transport` (`embedding.py:39-51`)
are byte-identical copies, both calling `urllib.request.urlopen(req)` **with no `timeout=`**.
FR-14's per-model timeout has nowhere to land until this is unified. The injectable `Transport`
alias — `(url, payload) -> dict` — is the seam every offline test uses; keeping that signature
stable is a hard constraint (37 `llm=` injection sites across 8 test files, and ~15
`LMStudioLLM(...)` constructions in `test_llm.py`).

### 2.3 LM Studio's `/v1` prefix — live-verified, and the failure is silent

Probed against the running LM Studio on this box (2026-08-10):

| Request | Result |
|---|---|
| `POST http://localhost:1234/v1/chat/completions` `{}` | **400**, proper OpenAI error envelope (`"No models loaded"`) |
| `POST http://localhost:1234/chat/completions` `{}` | **HTTP 200**, body `{"error":"Unexpected endpoint or method. (POST /chat/completions)"}` |
| `GET /models` and `GET /v1/models` | both **200** with the real model list |
| `http://192.168.0.69:1234` (the stakeholder's declared `baseURL`), root **and** `/v1` | connection failure (`000`) — not reachable from WSL right now |

Two consequences that shape the design:

1. **A missing `/v1` is not an HTTP error.** It is a `200` carrying an error envelope, so today's
   `resp["choices"][0]["message"]["content"]` (`llm.py:110`) raises a bare
   `KeyError: 'choices'` — an opaque failure with no mention of the URL, provider or model. FR-10's
   "fails loudly" must therefore include *body-level* error detection, not just HTTP status.
2. **`GET /models` is not a discriminator** (root answers 200 too), so no "probe the endpoint"
   trick can auto-detect the prefix. The normalization must be a declared rule (§4.3).

The two real sample files disagree on the suffix: `opencode/agents/severino/opencode.json` has
`http://localhost:1234/v1`; `~/.config/opencode/opencode.json` has `http://192.168.0.69:1234`.
falkor-chat must accept **both**, unmodified (FR-1/AC-1).

### 2.4 The stakeholder's real shared file, read

One provider `lmstudio`, `npm: "@ai-sdk/openai-compatible"`, `options.baseURL:
"http://192.168.0.69:1234"`, **no `apiKey`**, one model entry `google/gemma-3n-e4b`. LM Studio on
`localhost:1234` currently serves seven models including `qwen/qwen3-4b-2507`,
`mistralai_ministral-3-3b-instruct-2512` and `prism-ml/bonsai-27b` — **none of which appear in the
file's `models` map**. This settles a design question (§4.4): the `models` map is *metadata*, not
an allow-list. Requiring a ref's model id to be listed there would make AC-1 fail against the
stakeholder's own file.

### 2.5 The model-reference grammar is already fixed by a real file

`opencode/agents/severino/opencode.json` names its agent's model
`"lmstudio/qwen/qwen3-4b-2507"` — provider `lmstudio`, model id `qwen/qwen3-4b-2507`. So a ref
**splits on the first `/` only**; the model id may contain further slashes. A ref with **no** `/`
must be reserved for Landing 2's roles (FR-7) and therefore rejected in Landing 1 with a
forward-looking message, not silently reinterpreted.

### 2.6 `TraceEvent`s are debug-only — FR-8 cannot live on them

`executor._drive_loop:388` selects the tracer: `tracer = self._tracer if run["trace"] else
_NULL_TRACER`. `NullTracer.record` (`executor.py:147-152`) writes nothing. A normal (non-debug)
run therefore writes **zero** `TraceEvent`s. AC-4/AC-6/AC-9/AC-10 all read "the trace shows the
concrete model" for ordinary runs, so **the resolved model must be a durable property of the
`StepRun`**, not a `TraceEvent`. This is the single most important interface constraint on
`docs/plans/llm-provider-config-graph.md` (§8).

The `StepRun` is written by `executor._record` (`:821-838`) → `repo.record_step_and_advance(...)`,
one atomic query per step (§12.2). That call is where a `model` field belongs.

### 2.7 Publish-time validation seam (FR-9)

`services._validate_def_spec` (`services.py:776-878`) is a `@staticmethod` called once, from
`publish_workflow_def` (`services.py:906`). Its docstring establishes a house rule: **new
invariants run LAST**, so an older check keeps failing for its own reason. It raises
`WorkflowDefSpecError` → 400 (`app.py:_handle_wf_spec_error`). Because it is static and has no
access to `self`, the model-resolvability check is added as a **separate instance-level pass in
`publish_workflow_def`, immediately after `_validate_def_spec` returns** — lower blast radius than
converting the static method, and it preserves the "last" rule by construction.

`_normalize_opaque` (`services.py:186-205`) is the existing helper for reading a step `config` /
transition `guard` that may arrive as a dict (service/MCP callers) or a JSON string (REST). The
new pass must use it, or REST-published defs escape the check — the exact M-7 defect the helper
was written for.

### 2.8 Where a consumer names its model

- **workflow step** → `Step.config.model` — `config` is an opaque serialized string parsed
  app-side (DESIGN §6.1, AGENTS.md rule 8). **No DDL, no schema change.**
- **llm guard** → `{"kind":"llm","text":…,"model":…}` in `TRANSITION.guard` — same opacity.
- **embedding** → the `embedding` kind default, or an explicit ref at the wiring site.
- **agent** (the chat participant) → §4.6.

### 2.9 FR-20 blast radius, re-verified

The four variables named by FR-20 are **only ever read as defaults in `config.py`** — no script,
compose file or CI sets them. What must change:

| Site | Change |
|---|---|
| `server/falkorchat/config.py:38-49` | delete `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `LLM_BASE_URL`, `LLM_MODEL`; add the two file-path vars; add the AC-13 tripwire |
| `server/falkorchat/llm.py:99-100` | default args referencing the deleted constants |
| `server/falkorchat/embedding.py:59-60` | same |
| `server/falkorchat/app.py:250-303` | docstring + the whole wiring block |
| `server/tests/test_llm.py:53-54` | asserts against `config.LLM_BASE_URL` / `config.LLM_MODEL` |
| `server/tests/test_workflow_live.py:11,137` | docstring + skip message referencing `config.EMBEDDING_BASE_URL` |
| `scripts/start_server.sh` (header `:6-35`, defaults `:75`, export block `:153-163`) | document/export the two file-path vars |
| `compose.yaml` (`services.server.environment`) | pass the two paths; bind-mount the shared file read-only; the container cannot reach `localhost:1234` |
| `README.md:133-138, 205, 256-266` · `falkor-chat/AGENTS.md:78` + key-scripts table | run instructions |
| `docs/DESIGN.md` §1.3, §14.2, new §14.8 | the M2 stack table and the config seam |

`FALKORCHAT_EMBEDDING_DIM` is **not** in FR-20's list and survives — with a narrowed role (§4.5).

---

## 3. Design: the one seam (FR-4)

```
                     ┌───────────────────────────────────────────────┐
  opencode.json ──▶  │ models.py                                     │
  (pristine,         │   ProviderCatalog  ← parse + {env:}/{file:}    │
   FR-1/FR-11)       │   Overlay          ← defaults · models · roles │
                     │   ModelGateway     ← resolve(kind, requested,  │
  models.json  ──▶   │                               ws) → Resolution │
  (falkor-chat       │                    ← .llm(...) / .embedder(...)│
   overlay)          └───────┬───────────────────────────────────────┘
                             │ ResolvedModel(base_url, model, key, timeout, params)
                             ▼
                     ┌───────────────────────────────────┐
                     │ transport.make_http_transport()   │  timeout + headers + loud errors
                     └───────┬───────────────────────────┘
                             ▼
        OpenAICompatibleLLM / OpenAICompatibleEmbedder
                             ▲
      ┌──────────────┬───────┴────────┬──────────────────┐
   responder      judge            executor         embedding worker
   (kind=agent)   (kind=guard)     (kind=step)      (kind=embedding)
```

**The FR-4 rule, in one sentence:** *every LLM consumer holds a `ModelGateway` and asks it for a
client; a directly-injected client is sugar that `__init__` wraps into a `StaticModelGateway`, so
there is exactly one internal path and zero consumers reading endpoint/model settings privately.*

That last clause is what keeps the 37 existing `llm=` / 23 `guard_judge=` test injections working
untouched: `WorkflowExecutor(llm=stub)` becomes, inside `__init__`,
`self._models = models or StaticModelGateway(llm=llm)`. The `llm=` keyword remains **dependency
injection for tests**, never a configuration route — it reads nothing from `config.py` or any
file, which is precisely what FR-4 forbids.

### 3.1 Kinds (FR-6)

Four, matching the four consumers. Fixed, closed set:

| kind | consumer | resolved at |
|---|---|---|
| `agent` | `AgentResponder` — the `@mention` reply | per call (`maybe_respond`) |
| `step` | workflow `type:'agent'` step nodes | per step (`_run_agent_node`) |
| `guard` | the llm-kind guard judge | per guard evaluation |
| `embedding` | `EmbeddingWorker`, `tools.GraphRAGRetrieve`, the responder's query embed | per call (`embed_message(ws, …)`) |

Resolving **per call** rather than at construction is a Landing-1 decision taken specifically so
Landing 2's workspace override (FR-16, which is a function of `ws`) needs no signature changes:
every resolution point already has `ctx.ws` or `ws` in hand.

---

## 4. Design decisions & rationale

### 4.1 Two files, two env vars

| Variable | Meaning | Default |
|---|---|---|
| `FALKORCHAT_OPENCODE_CONFIG` | the pristine, shared `opencode.json` (FR-1/FR-2/FR-11) | `~/.config/opencode/opencode.json` |
| `FALKORCHAT_MODEL_CONFIG` | falkor-chat's own overlay (FR-11) | `<falkor-chat>/config/models.json` |

Named after their *formats*, not their contents — `PROVIDERS_FILE`/`MODELS_FILE` were rejected as
mutually confusable. Both are read **once, at wiring time** (FR-15); no reload path exists.

**Absence rules.** Both files are required *only when an LLM consumer is actually wired* —
`config.ENABLE_AGENT` or `config.WORKFLOW_ENABLED`. With both off (the default, and the pytest
baseline) the gateway is never constructed, importing `falkorchat.app` stays network-free **and
now also file-free**, and DESIGN §14.7's guarantees are untouched. When a consumer *is* wired and
a file is missing or malformed, startup fails with a message naming the **variable**, the
**resolved path**, and the shipped example. This is the AC-13 posture: never a silent fallback.

**Malformed** = not valid JSON, or a known key with the wrong shape. **Unknown top-level keys are
accepted and logged**, not rejected — this is what keeps an OpenCode file with `agent.*`,
`$schema`, `mcp`, `permission` blocks acceptable unmodified (AC-1), and what stops a Landing-1
build from rejecting an overlay that already carries `roles`. `roles` specifically gets a named
"parsed but not honoured until Landing 2" log line in Landing 1.

### 4.2 What is read from the shared file

Only `provider.<id>`:

```
provider.<id>.options.baseURL   → the endpoint
provider.<id>.options.apiKey    → the credential (after {env:}/{file:} substitution)
provider.<id>.options.headers   → extra headers, if present
provider.<id>.npm               → hint for `protocol` (see §4.7)
provider.<id>.name              → display only
provider.<id>.models.<id>.*     → metadata only (see §4.4)
```

Everything else in the file — `agent`, `mcp`, `permission`, `$schema`, `theme` — is ignored
without comment. falkor-chat **writes nothing** to this file, ever.

### 4.3 The `/v1` normalization rule (AC-1, resolved)

> **Rule.** After `{env:}`/`{file:}` substitution, take the provider's `baseURL`. If its URL
> **path component is empty or `/`**, append `/v1`. If it has any non-empty path, use it verbatim.
> Strip exactly one trailing `/` before joining `chat/completions` / `embeddings`.

Verified correct against every endpoint form in play:

| declared `baseURL` | path | resolved API base |
|---|---|---|
| `http://localhost:1234/v1` (severino sample) | `/v1` | `http://localhost:1234/v1` |
| `http://192.168.0.69:1234` (stakeholder's file) | `` | `http://192.168.0.69:1234/v1` |
| `https://api.openai.com/v1` | `/v1` | unchanged |
| `https://api.anthropic.com` | `` | `https://api.anthropic.com/v1` ✔ (its real prefix) |
| `http://gw.lan/openai/v1` (a proxy) | `/openai/v1` | unchanged |

**Escape hatch, always available:** the overlay may set `providers.<id>.baseURL`, which wins
outright over the shared file and over the rule. That is how an admin overrules the heuristic
**without editing the shared file** (FR-11 intact). A heuristic with a documented, per-provider
override is strictly better here than either alternative — probing is impossible (§2.3), and
demanding a hand-written suffix would break AC-1 on the stakeholder's own file.

### 4.4 A ref resolves on its **provider**, not on the `models` map

`lmstudio/qwen/qwen3-4b-2507` resolves iff provider `lmstudio` is declared. The model id is
**not** checked against `provider.lmstudio.models`. Rationale in §2.4: LM Studio serves seven
models while the file lists one; an allow-list would fail AC-1 against the real file and would
force the admin to edit the shared file every time they load a model — the precise duplication
this feature exists to delete. An **unknown provider** is unresolvable and fails per FR-9/FR-10.

Consequence, stated plainly: a typo'd *model id* is caught at call time (FR-10, a loud provider
error naming provider + model + URL), not at publish time. A typo'd *provider* is caught at
publish time (FR-9). This asymmetry is deliberate and must be in the QA test plan.

### 4.5 `EMBEDDING_DIM` (FR-19's fault line)

Today `EmbeddingWorker._dim` comes from `config.EMBEDDING_DIM` (env, default 1536), and
`repository.set_embedding` re-validates with the same value (`repository.py:690`). Under this
plan:

1. The overlay may declare `models."<ref>".dim` for an embedding model — **authoritative when
   present**.
2. `FALKORCHAT_EMBEDDING_DIM` remains the fallback when it is not, so every existing script and
   the `ws:test` dim-4 estate keep working unchanged.
3. `scripts/bootstrap_schema.sh`'s own `EMBEDDING_DIM` is untouched — it is DDL-time input for
   vector-index creation and is a different thing from the model's output width.
4. Landing 2 (FR-19) adds the *third* number: the workspace's **frozen index dimension**, read
   from FalkorDB, compared against (1)/(2). That comparison and its query belong to
   `docs/plans/llm-provider-config-graph.md`.

### 4.6 How the chat **agent** names its model (FR-5, agent axis)

The overlay carries an optional `agents` map keyed by `agentId`:

```json
"agents": { "assistant": "lmstudio/qwen/qwen3-4b-2507" }
```

Resolution for kind `agent` is: `agents[<agentId>]` → `defaults.agent`. **Chosen over an
`Agent.model` graph property** because: it needs no DDL, no new query, no `reference`/workspace
sync, it is unit-testable offline, and "configuration is hand-edited files" is an explicit
requirement (Out of scope: "editing configuration through a UI or API"). The rejected alternative
is recorded in §9 as an open question for `graph-dba`, since the workspace override (FR-16)
*does* need graph storage and the two could plausibly share a home.

Landing-1 note: the `agents` map is Landing 2 (it is part of the FR-5 agent axis that only becomes
interesting once roles/overrides exist); Landing 1 resolves kind `agent` straight to
`defaults.agent`. The map key is reserved and parsed-but-ignored in Landing 1, exactly like
`roles`.

### 4.7 Provider protocol (FR-13, and what Landing 1 does *not* build)

`ResolvedModel` carries `protocol: "openai" | "anthropic"`, defaulted from the overlay's
`providers.<id>.protocol`, else inferred from the shared file's `npm`
(`@ai-sdk/anthropic` → `anthropic`; anything else → `openai`). **Only `openai` is implemented.**
A resolution that yields `protocol="anthropic"` raises `ModelConfigError` at startup naming the
provider and the reason — a declared seam, never a wrong-shaped payload.

This still satisfies FR-13's three provider kinds: LM Studio (`openai`), a second LAN
OpenAI-compatible host (`openai`), and hosted cloud — OpenAI natively, **and Anthropic via its
documented OpenAI-SDK compatibility base URL `https://api.anthropic.com/v1/`** (a beta layer
Anthropic explicitly documents; it does not support extended thinking, prompt caching, PDFs or
citations). A native Messages-API client is a follow-up item, not a Landing-2 obligation — file it
as **K-043** if the stakeholder wants it.

### 4.8 `{env:}` / `{file:}` substitution (FR-12)

Applied to **every string value inside the `provider.*` subtree** of the shared file and inside
the overlay's `providers.*` subtree. Occurrences are replaced in place (embedded occurrences too,
not just whole-value). `{file:~/...}` expands `~` and strips trailing whitespace. An **unresolved**
`{env:NAME}` (variable absent) or an unreadable `{file:...}` is a **startup error naming the
variable/path and the file it appeared in** — never an empty string, never a silent skip, because
an empty `Authorization: Bearer ` reaches a cloud provider as a 401 with no local diagnosis.

**Secret hygiene, enforced in code:** the credential is held in a `Secret` wrapper whose `__repr__`
/ `__str__` render `***`; `ResolvedModel.__repr__` never contains it; no log line, trace payload
or error message ever interpolates it. This is a done-condition, not a guideline (§7 L1-2, and
`devops`' U6 review).

### 4.9 Loud failure at call time (FR-10)

`transport.make_http_transport` classifies four failures and raises `ProviderCallError` naming
**provider · model · resolved URL · cause** for each:

1. transport/connection failure (`URLError`, timeout);
2. HTTP error status (`HTTPError`) — body included, truncated;
3. **HTTP 200 whose JSON body carries a top-level `error`** — the live-verified LM Studio
   wrong-prefix case (§2.3);
4. a 200 body that is not a JSON object.

The client layer adds a fifth: a well-formed body missing `choices` / `data`. Inside a workflow
run this reaches `executor._drive`'s M-1 fault net, which stamps `fail_run` with the message
(`_fail_with_note`, `executor.py:848`) — the run terminates `failed` with a readable cause and
`AT_STEP` cleared, never a `running` zombie and never another model.

> **Terminology reconciliation (assumption, flagged).** FR-10/AC-8 say the run "**suspends** with
> an error". The engine's existing, correct behaviour for a fault is `failed` with the cause in
> `ctx.error` — `waiting` means "a human can unblock this", which an unresolvable model is not.
> **Assumption: `failed`-with-cause satisfies the intent** ("never silently falls back", "an error
> stating what could not be resolved"). Flagged for `tico`/`analyst` in §9; no code change is
> planned to make a run `waiting` on a provider error.

---

## 5. New and changed files

| File | Landing | What |
|---|---|---|
| `server/falkorchat/transport.py` | 1 | **new** — `make_http_transport(*, timeout, headers, opener)`, `ProviderCallError`. The one HTTP transport. |
| `server/falkorchat/models.py` | 1 | **new** — `Secret`, `ProviderSpec`, `ResolvedModel`, `Resolution`, `ProviderCatalog`, `Overlay`, `ModelGateway`, `StaticModelGateway`, `ModelConfigError`, `ModelResolutionError`. Pure/offline. |
| `falkor-chat/config/models.json` | 1 | **new, committed** — the shipped overlay (defaults for the four kinds; no secrets). |
| `falkor-chat/config/opencode.example.json` | 1 | **new, committed** — a documented shared-file example (LM Studio + a LAN host + a cloud provider using `{env:}`). |
| `server/falkorchat/config.py` | 1 | remove the four FR-20 constants; add `OPENCODE_CONFIG_PATH`, `MODEL_CONFIG_PATH`, `assert_no_legacy_model_env()`. |
| `server/falkorchat/llm.py` | 1 | `LMStudioLLM` → `OpenAICompatibleLLM(base_url, model, *, transport=None, params=None)`; required args; drop `_urllib_transport`. |
| `server/falkorchat/embedding.py` | 1 | `LMStudioEmbedder` → `OpenAICompatibleEmbedder`; `EmbeddingWorker(repo, embedder=None, *, models=None, expected_dim=None)` resolving per call. |
| `server/falkorchat/app.py` | 1 | wiring through the gateway; `_build_llm_judge(models)`. |
| `server/falkorchat/executor.py` | 1 | `models=` param; `self._models = models or StaticModelGateway(llm=llm)`; per-step resolution in `_run_agent_node`. |
| `server/falkorchat/guards.py` | 1 | forward `model=` to the judge **only when the guard declares one**. |
| `server/falkorchat/responder.py` | 1 | hold the gateway; resolve per call. |
| `server/falkorchat/services.py` | 2 | `publish_workflow_def` gains the model-resolvability pass (FR-9). |
| `server/falkorchat/repository.py` | 2 | `record_step_and_advance` gains `model=` (FR-8) — **shape owned by `-graph.md`**. |

New tests: `server/tests/test_transport.py`, `server/tests/test_models.py`.
Extended: `test_llm.py`, `test_embedding.py`, `test_app.py`, `test_executor_agent.py`,
`test_guards.py`, `test_responder.py`, `test_services.py`, `test_workflow_live.py`.

---

## 6. Landing 1 — the seam, the files, the cutover

**Requirements:** FR-1..FR-6, FR-11..FR-15, FR-20. **Acceptance reachable:** AC-1, AC-4 (the
"each step calls its own model" half), AC-5, AC-12, AC-13; AC-2/AC-3 structurally (§10).
**Demonstrable on its own:** yes — two workflow steps on two different LM Studio models, a guard
on a third, all from two hand-edited files, with the old env vars gone.

| # | Unit | Files | Done when |
|---|---|---|---|
| **L1-1** | **One HTTP transport with timeout + loud errors.** `make_http_transport(*, timeout: float, headers: Mapping[str,str] \| None = None, opener=urllib.request.urlopen) -> Transport`; `ProviderCallError(RuntimeError)`. Bind timeout/headers at construction so the `Transport` alias `(url, payload) -> dict` is **unchanged**. Delete both `_urllib_transport` copies; import from here. | `transport.py` (new), `llm.py`, `embedding.py` | `test_transport.py` pins the four §4.9 failure classes offline via an injected `opener` (incl. the 200-with-`error`-envelope case, message naming url+model). `test_llm.py`/`test_embedding.py` green with no edits beyond the import. |
| **L1-2** | **`models.py` — parse, substitute, merge, resolve.** Shared-file reader (providers only); overlay reader (`providers`/`defaults`/`models`; `roles`/`agents` parsed + logged as reserved); `{env:}`/`{file:}` (§4.8) with `Secret`; `/v1` normalization (§4.3) + per-provider override; ref grammar split-on-first-`/` (§2.5) with a no-`/` ref rejected as "roles are not available until Landing 2"; per-kind defaults; per-model settings (`timeout`, `temperature`, `maxTokens`, `dim`); `protocol` gate (§4.7). `ModelGateway.from_env()`, `.resolve(kind, *, requested=None, ws=None) -> Resolution`, `.llm(...)`, `.embedder(...)`. `Resolution` carries `chain: tuple[ResolvedModel, ...]` with `primary = chain[0]` — length 1 in Landing 1 (**the FR-18 seam**). `resolve` accepts and ignores `ws` via a `WorkspaceOverrides` port defaulting to a null implementation (**the FR-16/FR-17 seam**). `StaticModelGateway(llm=None, embedder=None, dim=None)`. | `models.py` (new) | `test_models.py` green, entirely offline: both real sample files parse (fixtures copied into `tests/data/`); the `/v1` table in §4.3 is pinned row by row; `{env:}` resolves, is absent→error, and never appears in `repr()`; unknown top-level keys accepted; a no-`/` ref rejected with the Landing-2 message; per-kind defaults differ and each resolves; `dim` honoured. `python -c "import falkorchat.models"` with no env set does not read a file or touch the network. |
| **L1-3** | **Generalize the clients + per-model settings.** `OpenAICompatibleLLM(base_url, model, *, transport=None, params=None)` and `OpenAICompatibleEmbedder(base_url, model, *, transport=None)` — `base_url`/`model` **required** (no `config.*` defaults); `params` merged into the chat payload (`temperature`, `max_tokens`). No back-compat aliases: FR-13 makes the `LMStudio` name a lie. | `llm.py`, `embedding.py`, `test_llm.py`, `test_embedding.py` | The 3 default-arg constructions in `test_llm.py` (`:23,:43,:57`) and the `config.LLM_*` assertions (`:53-54`) are updated to explicit values; a new test pins that `params` reach the payload and that omitting them sends neither key. Suite green. |
| **L1-4** | **Rewire all four consumers onto the gateway (FR-4/FR-5/FR-6).** `WorkflowExecutor.__init__(..., models=None, llm=None)` → `self._models = models or StaticModelGateway(llm=llm)`; `_execute_step`'s "no LLM wired ⇒ empty stub" branch keys off `self._models.has_chat()` (**preserving the deliberate offline-stub affordance — never tidy it into a raise**); `_run_agent_node` resolves `self._models.llm("step", requested=config.get("model"), ws=ctx.ws)`. `guards.evaluate_guard` passes `model=` to the judge **only when the guard dict declares one** (zero churn for the 23 existing stub judges). `app._build_llm_judge(models)` resolves kind `guard` per evaluation. `AgentResponder` holds the gateway, resolving `agent` + `embedding` inside `maybe_respond`. `EmbeddingWorker(repo, embedder=None, *, models=None, expected_dim=None)` → `self._models = models or StaticModelGateway(embedder=embedder, dim=expected_dim)`, resolving inside `embed_message(ws, …)`. `app._build_default_app` builds **one** `ModelGateway.from_env()` and passes it everywhere. | `executor.py`, `guards.py`, `responder.py`, `embedding.py`, `app.py`, `tools.py` (embedder type hint only) | All 37 `llm=` and 23 `guard_judge=` injections still pass **unmodified**. New offline tests: two steps naming different models call different URLs/model ids (AC-4, fake transport recording calls); an llm-guard naming its own model hits that model; `agent`/`guard`/`embedding` defaults may differ and each is used (AC-5); a per-model `timeout` reaches `urlopen` (AC-12, asserted through the injected opener). |
| **L1-5** | **FR-20 cutover + AC-13 tripwire.** Delete the four constants. Add `config.assert_no_legacy_model_env()`: if any of `FALKORCHAT_LLM_BASE_URL`, `FALKORCHAT_LLM_MODEL`, `FALKORCHAT_EMBEDDING_BASE_URL`, `FALKORCHAT_EMBEDDING_MODEL` is **set**, raise naming them and pointing at the two files — called from `ModelGateway.from_env()`. Update `scripts/start_server.sh` (header comment, defaults, export block), `compose.yaml` (the two paths + a read-only bind mount of the shared file + a note that `localhost:1234` is not reachable from the container — use `host.docker.internal` or a LAN address), `README.md`, `falkor-chat/AGENTS.md`. Ship `config/models.json` + `config/opencode.example.json`. | `config.py`, `scripts/start_server.sh`, `compose.yaml`, `README.md`, `AGENTS.md`, `config/*.json`, `test_workflow_live.py:11,137` | `grep -rn "FALKORCHAT_LLM_\|FALKORCHAT_EMBEDDING_BASE_URL\|FALKORCHAT_EMBEDDING_MODEL"` over `server/`, `scripts/`, `compose.yaml`, `README.md`, `AGENTS.md` returns **only** the tripwire's own list. Setting one of them and starting the server fails with the tripwire message (AC-13). `./scripts/start_server.sh` brings up a working demo on a box whose `~/.config/opencode/opencode.json` exists. |
| **L1-6** | **Docs, in the same change.** `docs/DESIGN.md`: §1.3 — the stack table's model rows become *the shipped defaults in `config/models.json`*, not hardcoded constants; §14.2 — the layering box gains `models.py` + `transport.py`; **new §14.8 "The model-resolution seam"** (the §3 diagram, the four kinds, the two files, the `/v1` rule, the FR-4 rule sentence); §14.7 — add a hazard bullet: *a wired agent now also requires two config files; a missing one aborts startup by design*. `docs/HISTORY.md`: one dated Landing-1 entry with new suite counts. `docs/BACKLOG.md`: K-042 → 🟡. Flag to `tico`: `docs/manuals/llm-provider-config.md` (admin-facing "how to configure models") is now worth writing. | `docs/DESIGN.md`, `docs/HISTORY.md`, `docs/BACKLOG.md` | No document still describes model choice as an env var. §14.8 exists and the §1.3 table points at it. |

### 6.1 Seams Landing 1 must leave open (explicit)

1. `Resolution.chain` is a tuple, `primary = chain[0]` — FR-18's fallback chain becomes a change
   inside the client wrapper, not a signature change at 4 call sites.
2. `resolve(..., ws=...)` exists and is threaded from every call site from day one — FR-16/FR-17
   swaps a `NullWorkspaceOverrides` for the graph-backed one.
3. A ref **without** a `/` is rejected with a Landing-2-aware message — the role namespace stays
   unclaimed.
4. Overlay keys `roles` and `agents` are reserved, parsed, and logged as not-yet-honoured — an
   admin's Landing-2-ready file never fails on a Landing-1 build.
5. `ResolvedModel.label` (`"<provider>/<model>"`) is populated in Landing 1 and is exactly the
   string FR-8 records — Landing 2 adds the write, not the value.
6. Every resolution point already has `ctx.ws`/`ws` in scope. Nothing in Landing 2 needs a new
   parameter to travel through a function that does not already carry the workspace.

---

## 7. Landing 2 — roles, precedence, traceability, guards

**Requirements:** FR-7..FR-10, FR-16..FR-19. **Acceptance:** AC-6..AC-11 (+ AC-4's trace half).

| # | Unit | Files | Done when |
|---|---|---|---|
| **L2-1** | **Roles (FR-7) + ordered fallback chains (FR-18).** Overlay `roles.<name> = {"models": ["<ref>", …], "timeout": …, …}`. A consumer ref without `/` resolves to a role → a `Resolution` whose `chain` is the ordered, per-model-settings-applied list. New `FallbackClient` wrapper: on `ProviderCallError` from element *n*, try *n+1*; when all fail, raise a `ProviderCallError` naming **every** model tried. Exposes `.last_used: ResolvedModel` for FR-8. A **role name must not contain `/`**; a role that resolves to another role is rejected at load (no recursion). | `models.py`, `transport.py` | `test_models.py`: a role resolves to its first model; changing the mapping changes the resolution with **no republish** (AC-6, exercised by re-loading the overlay); a chain whose first element raises falls to the second and reports it as used (AC-9, offline via a failing fake transport). |
| **L2-2** | **Record the model that actually ran (FR-8).** `executor._record` passes the concrete label of the model that answered (`FallbackClient.last_used.label`, or the single resolved model) into `repo.record_step_and_advance(..., model=…)`. Must be a durable **`StepRun` property**, not a `TraceEvent` — see §2.6 and §8. Also emit a `model_resolved` trace event for debug runs (additive, not the mechanism). Surface it on `GET /workflow-runs/{id}/step-runs`. | `executor.py`, `repository.py`, `services.py`, `schemas.py`, `api.py` | Two steps on two models produce two different `StepRun.model` values on a **non-debug** run (AC-4); the role-swap run shows the new concrete model (AC-6); the fallback run shows the model that answered, not the one that failed (AC-9). |
| **L2-3** | **Workspace override + precedence (FR-16/FR-17).** Implement `WorkspaceOverrides` against the storage `-graph.md` specifies; read **once per drive / per responder call**, not per resolution, and pass it down (a per-step graph read would be a hot-path regression). Precedence, first-match-wins: **workspace → the consumer's own choice → the per-kind default.** The override is a **hard cap** — it beats an explicit step choice. | `models.py`, `executor.py`, `responder.py`, `repository.py` | With an override set, a step that explicitly names a different model runs on the **workspace's** model and `StepRun.model` shows it (AC-10). A unit test pins all three precedence rungs and the hard-cap direction. |
| **L2-4** | **Publish-time rejection (FR-9).** New instance-level pass in `publish_workflow_def`, **after** `_validate_def_spec` returns (§2.7), using `_normalize_opaque` so REST-published defs are covered: for each step, `config.model`; for each transition, an `{"kind":"llm"}` guard's `model`. Each is `gateway.validate_ref(...)`; failure → `WorkflowDefSpecError` naming **the step key (or the transition endpoints) and the offending identifier** → 400, nothing written. A `Services` built without a gateway skips the pass (the entire offline test estate). | `services.py`, `app.py` | AC-7: publishing a def whose step names `nope/thing` or an undeclared role fails 400 with the step key and the identifier in `detail`; a def naming a *valid* role publishes; `test_services.py`'s existing publish tests are unaffected (no gateway ⇒ no pass). |
| **L2-5** | **Loud use-time failure, run half (FR-10).** Confirm and pin: an unresolvable ref at drive time raises `ModelResolutionError`, reaches `_drive`'s M-1 net, and terminates the run `failed` with the identifier in `ctx.error`, `AT_STEP` cleared, **no other model used**. For the responder, the equivalent is an ERROR log naming the identifier and **no reply posted** (`background._safe_respond` isolation). | `executor.py`, `responder.py` (assertions only, likely no behaviour change) | AC-8 pinned by test at the drive level; a second test asserts no fallback model was called. |
| **L2-6** | **Embedding-dimension guard (FR-19).** Before the first write for a workspace, compare the resolved embedding model's declared `dim` (§4.5) against the workspace's **frozen vector-index dimension**, read via the query `-graph.md` specifies. Mismatch → raise `EmbeddingDimensionError` **before** calling the embedder, so **no vector is written** and no wasted inference happens. Cache the index dimension per `(ws, process)` — it cannot change while the index exists. Keep `repository.set_embedding`'s existing length check as the last line of defence. | `embedding.py`, `repository.py` | AC-11: a mismatch raises with a message naming the model, its dim, the workspace and the index dim; `MATCH (m:Message {msgId:…}) RETURN m.embedding` is null afterwards. `ws:test` (dim 4, DESIGN §14.7) keeps working via the declared-dim path. |
| **L2-7** | **Docs + close.** DESIGN §14.8 gains roles/precedence/trace; `docs/HISTORY.md` Landing-2 entry; BACKLOG K-042 → ✅ + M4 row; hand off to `qa-engineer` for `docs/test-plans/llm-provider-config.md`. | docs | Every AC in §10 has a row with a verdict or a recorded gate. |

---

## 8. Interface required from `docs/plans/llm-provider-config-graph.md`

That document did not exist when this plan was written. This plan **does not** design any of the
following; it states what the resolver needs and defers the Cypher, schema, index and RAM analysis:

1. **FR-8 — resolved model on the execution trace.** A **durable, always-written** field carrying
   the concrete `"<provider>/<model-id>"` string for each executed step, readable per run. It
   **cannot be a `TraceEvent`**: `executor._drive_loop:388` selects a `NullTracer` for every
   non-debug run, so a `TraceEvent`-only design would make AC-4/AC-6/AC-9/AC-10 hold for debug
   runs only. The natural home is a `StepRun` property written by the existing atomic
   `record_step_and_advance` (§12.2) — adding a property to that one query, not a second write.
   Needed: the query change, the read surface, and the RAM note (AGENTS.md rule 6).
2. **FR-16/FR-17 — where the workspace override lives and how it is read.** The resolver needs a
   read shaped `get_model_overrides(ws) -> {kind -> ref}` (or `{kind|"*" -> ref}`), cheap enough to
   call **once per run drive / per responder call**. Needed: the node/property choice, the write
   path (admin-set, hand-edited or API — the requirements put UI editing out of scope, so a seed
   script or a one-shot query is acceptable), and whether it lives in `ws:{id}` or `reference`.
3. **FR-19 — the workspace's frozen vector-index dimension.** A read returning the dimension of
   `ws:{id}`'s `Message.embedding` (and `Chunk.embedding`) vector index on the pinned FalkorDB
   build, plus confirmation of the exact procedure/field name (`db.indexes()` shape varies by
   build — must be live-verified, not assumed).
4. **Open question routed there:** should the chat agent's model choice (§4.6) live on the `Agent`
   node instead of the overlay file? This plan chose the file; if the workspace override lands on a
   node that would naturally also carry an agent-level setting, say so and this plan is amended.

---

## 9. Risks, open questions, and what changes for the developer

### 9.1 What Landing 1 changes about the day-to-day run

- **`./scripts/start_server.sh` now requires two files.** `~/.config/opencode/opencode.json` (or
  `FALKORCHAT_OPENCODE_CONFIG`) and `falkor-chat/config/models.json` (committed, so it is there).
  If the shared file is absent and the agent is enabled, the server **refuses to start** with a
  message naming the variable, the path and `config/opencode.example.json`. This is FR-20/AC-13
  working as specified, not a regression.
- **`pytest` is unchanged.** Both agent flags are off by default, so the gateway is never built:
  no file is read, no network is touched, DESIGN §14.7 stands.
- **Changing a model no longer means an env var.** Edit `config/models.json`, restart.
- **⚠️ The stakeholder's declared endpoint does not currently work.**
  `http://192.168.0.69:1234` is unreachable from WSL right now (verified §2.3), while
  `http://localhost:1234/v1` answers. The first live run of Landing 1 will therefore fail at call
  time with a `ProviderCallError` naming that URL — *correct behaviour*, but it needs one of:
  (a) LM Studio bound to the LAN interface, (b) editing `options.baseURL` in the shared file, or
  (c) `providers.lmstudio.baseURL` in falkor-chat's overlay (§4.3 escape hatch, keeps the shared
  file pristine). **This is a demo-readiness item for `devops`/`qa-engineer`, decided at run time,
  not a design change.**
- **Containers:** `compose.yaml`'s server cannot reach `localhost:1234`. The bind-mounted shared
  file must either use a LAN address or the compose service must add
  `host.docker.internal:host-gateway` (the pattern `cpg/mcp/docker-run.sh` already uses in this
  repo).

### 9.2 Risks

| Risk | Mitigation |
|---|---|
| The `/v1` heuristic is wrong for some future provider | Per-provider overlay override (§4.3), documented, always available; the rule is pinned row-by-row in `test_models.py`. |
| A missing `/v1` returns **HTTP 200** with an error body → today a bare `KeyError` | L1-1 detects body-level errors and names provider/model/URL (§4.9). Live-verified failure mode, pinned by test. |
| Test churn breaks the 60 existing injection sites | `StaticModelGateway` sugar in `__init__` (§3) — 0 changes to `llm=`/`guard_judge=` call sites; the conditional `model=` kwarg in `guards` (L1-4) keeps stub judges intact. |
| Secrets leaking into logs / trace payloads / `repr` | `Secret` wrapper with redacting `__repr__`/`__str__`; a `test_models.py` test asserts the literal secret does not appear in `repr(resolution)` or in a raised `ProviderCallError`; `devops` re-checks at U6. |
| Timeouts change behaviour: today there is **no** timeout, so a hung endpoint hangs forever; a default timeout will now cut long generations | Ship a deliberately generous default (**180 s**) in `config/models.json`, per-model overridable (FR-14/AC-12). Call out in HISTORY: slow local models on first load can exceed a small timeout. |
| Renaming `LMStudioLLM` breaks an out-of-tree caller | Only in-repo callers exist (`app.py`, `test_llm.py`, `test_workflow_live.py` — enumerated §2.9). No alias, deliberately. |
| FR-9 makes publish depend on config, so a def publishable on one box is rejected on another | Accepted and required by AC-7. The pass is **skipped entirely** when `Services` has no gateway, so the offline/test estate and the `reference`-graph seeding path are unaffected. |
| `EMBEDDING_DIM` now has three sources (env, overlay, index) | §4.5 fixes precedence; FR-19 (L2-6) turns the remaining mismatch into a loud refusal. |

### 9.3 Open questions (surfaced, assumed past — see §4/§7 for the assumption taken)

1. **FR-10 "suspends" vs. the engine's `failed`.** Assumed: `failed`-with-cause satisfies the
   intent (§4.9). If `tico`/the stakeholder means literally `waiting`, L2-5 changes and a
   human-unblockable path for provider errors must be designed. **Ask before implementing L2-5.**
2. **Agent-model home** — overlay `agents` map (chosen, §4.6) vs. an `Agent` node property.
   Routed to `-graph.md` (§8.4).
3. **Native Anthropic Messages API** — declared out of scope (§4.7). If required, file **K-043**;
   it is a second protocol client, not a config change.
4. **Manual** — `docs/manuals/llm-provider-config.md` is likely wanted at Landing 1 close
   (already flagged in the coordination doc). `tico` decides.

---

## 10. Test strategy

**Altitude and the network-free rule.** DESIGN §14.7 is binding: the default `pytest` run must
stay network-free and FalkorDB-optional. Everything in §6 and §7 except the live-marked tests is
exercised **offline** through the two existing seams — the injectable `Transport`
(`(url, payload) -> dict`) and, new, an injectable `opener` inside `make_http_transport`. The four
consumers are covered by asserting **which URL and which model id** a fake transport received;
no live model is needed to prove that step A and step B used different models.

**Ordered behaviours to drive (Landing 1)** — usable directly as a red→green list:

1. A `{env:VAR}` `apiKey` resolves; an unset `VAR` raises naming the variable and the file; the
   literal secret appears in **no** `repr`, log or error string.
2. Each row of the §4.3 `/v1` table maps declared → resolved; an overlay `providers.<id>.baseURL`
   beats both.
3. `lmstudio/qwen/qwen3-4b-2507` splits into (`lmstudio`, `qwen/qwen3-4b-2507`); a ref with no
   `/` is rejected with the Landing-2 role message.
4. Both real sample files (`opencode/agents/severino/opencode.json` and a redacted copy of the
   stakeholder's) parse unmodified, `agent`/`mcp`/`$schema` blocks and all (AC-1).
5. A model id absent from the provider's `models` map still resolves (§4.4).
6. Four per-kind defaults, all different, each reaching its own consumer (AC-5).
7. Two workflow steps naming different models produce two different `(url, model)` pairs (AC-4,
   Landing-1 half).
8. An llm-guard naming its own model is judged by that model; a guard **not** naming one passes no
   `model=` kwarg (the zero-churn contract).
9. A per-model `timeout` reaches the opener (AC-12); omitting it uses the shipped default.
10. Each of the four §4.9 failure classes raises `ProviderCallError` naming provider · model · URL
    — **including the 200-with-`error`-envelope case**.
11. Setting any legacy env var aborts startup with the tripwire message (AC-13).

**Landing 2** adds: role resolution and re-mapping without republish (AC-6); chain fallback and
which model answered (AC-9); `StepRun.model` on a **non-debug** run (AC-4/AC-8); the three
precedence rungs and the hard cap (AC-10); publish rejection naming step + identifier (AC-7);
drive-time unresolvable → `failed` with the cause and no other model called (AC-8); dimension
mismatch refuses before embedding, no vector written (AC-11).

**Live (`pytest -m live`)** — extend `test_workflow_live.py`: one run whose two steps genuinely
hit two different LM Studio models, asserting the two `StepRun.model` values. Gated on LM Studio
reachability exactly as today.

**AC → landing → verification map**

| AC | Landing | How verified |
|---|---|---|
| AC-1 | 1 | offline parse of both real files, unmodified |
| AC-2 | 1 | `{env:}` substitution unit-tested; **end-to-end against a hosted provider deferred — no API key** (structurally verified, per the coordination decision) |
| AC-3 | 1 | LM Studio exercised end-to-end; second LAN host + cloud **structurally verified only** (no second host exists, no key) |
| AC-4 | 1 (calls) + 2 (trace) | offline transport assertions; then `StepRun.model` |
| AC-5 | 1 | offline, four differing defaults |
| AC-6 | 2 | role re-map + restart, no republish |
| AC-7 | 2 | publish → 400 naming step + identifier |
| AC-8 | 2 | drive-time failure → run `failed` with cause (see §9.3 Q1) |
| AC-9 | 2 | chain fallback, reported model |
| AC-10 | 2 | override beats explicit step choice |
| AC-11 | 2 | refuse before embed, no vector written |
| AC-12 | 1 | timeout reaches the opener |
| AC-13 | 1 | legacy env tripwire |

**Suite discipline.** No Cypher changes in Landing 1 ⇒ `./scripts/test_queries.sh` untouched.
Landing 2 touches `record_step_and_advance` and adds two reads ⇒ **`QUERIES.md` + the query suite
must rise with enumerated assertions**, per the `graph-dba` gate convention (owned by
`-graph.md`).
