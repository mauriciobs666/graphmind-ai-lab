# LLM Provider & Model Configuration — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-042 (M4) · **Version:** 4 · **Reviews:** `docs/reviews/llm-provider-config.md`

Turns `docs/requirements/llm-provider-config.md` (FR-1..FR-20, AC-1..AC-13) into an ordered,
two-landing build. Coordination record: `docs/plans/llm-provider-config-coordination.md`.
Graph-side mechanics (FR-8 trace recording, FR-16/FR-17 storage, FR-19 index dimension) are
owned by `docs/plans/llm-provider-config-graph.md` — this plan states the **interface** it needs
from that document and never invents the Cypher.

> **Revision note — 2026-08-10 (v1 → v2).** Revised in place against the `analyst` gate
> (`docs/reviews/llm-provider-config.md`, verdict *needs changes*: 2 blockers, 6 majors,
> 9 minors). All 17 findings adopted except two partial rebuttals recorded with evidence in
> §11.2. Substantive changes: the `guard` kind's workspace carrier (**B-1**, new §4.10 — v1's
> §6.1 item 6 was **false**); a rewritten, ordered exception ladder for the transport
> (**B-2**, §4.9); `server/.env.example` folded into the FR-20 cutover after re-deriving the
> blast radius with a dotfile-catching method (**M-1**, §2.9); a test-fixture strategy and the
> removal of the `~/.config/...` product default (**M-2**, §4.1); `tools.GraphragRetrieveTool`
> re-booked as a real FR-4 consumer (**M-3**); the FR-9 pass moved after K-034's conflict check
> (**M-4**); `FallbackClient.last_used` replaced by a `ChatResult.model` carrier (**M-5**);
> `baseURL` scheme/netloc validation at load time (**M-6**); `models.py` renamed
> `modelconfig.py` (**m-8**); `StepRun.resolvedModel` / `resolved_model=` / `modelSource`
> adopted per adjudication A-1, and `ResolvedModel.label` renamed `.ref`. Full disposition
> table: §11.

> **Revision note — 2026-08-10 (v2 → v3).** Pass 2 of the `analyst` gate on Version 2
> (`docs/reviews/llm-provider-config.md`, `## Pass 2`) confirmed both original blockers and all
> majors/minors from Pass 1 genuinely resolved, and raised one new blocker: **P2-B**, a
> cross-document schema drift against `docs/plans/llm-provider-config-graph.md` v2. That document's
> response to Pass-1 finding m-5 was not the documentation-only note this plan's v2 still carried —
> it added a fully-specified fourth `StepRun` property, `modelFallback` (boolean, nullable, absent
> by default; schema, write query, both read projections, `StepResult` carrier, RAM note, and a
> binding resolver contract in its §6.2), with a stated rationale this plan had not caught up with
> or rebutted (AC-9 is a formal acceptance criterion, not a debugging concern, and `TraceEvent`s are
> debug-only by construction — the same reason `resolvedModel` itself doesn't live there). This
> revision **adopts `-graph.md`'s design in full** rather than re-deciding it: `ChatResult` gains a
> `.fallback` carrier alongside `.model` (added at L2-1, where `FallbackClient` first exists to set
> it); `record_step_and_advance` gains `model_fallback=` alongside `resolved_model=`/`model_source=`
> (L2-2); `StepResult` gains `modelFallback: bool | None = None`; the read surface
> (`GET /workflow-runs/{id}/step-runs`) surfaces it alongside `resolvedModel`/`modelSource`; §2.6,
> §5, §8.1 and §11.1 are brought into agreement with `-graph.md` §1.3/§1.4/§1.5/§1.7/§6.2. One
> stale restatement inside `-graph.md` §6.5 (the withdrawn A-2 framing) was also raised in Pass 2 —
> that finding is `graph-dba`'s, not this plan's, and is not touched here. Full Pass 2 disposition:
> §12.

> **Revision note — 2026-08-11 (v3 → v4).** Wording-only fix, no design change: three spots (§5's
> `llm.py` row, §7's L2-1 done-condition, §12.1's P2-B disposition) described the non-fallback
> value of `ChatResult.fallback`/`StepRun.modelFallback` as `` `None`/`False` ``, implying either
> is possible. It is `None` only, never `False` — `-graph.md` (v3) is unambiguous on this in
> §1.3 ("absent when no fallback occurred"), §1.4 ("`modelFallback`: absent means
> 'unknown/not applicable', never 'confirmed no fallback'") and its §6.2 schema comment ("omit
> (`None`) when it is [not a fallback]"), and this plan's own §7 L2-2 done-condition already
> stated it correctly. All three occurrences corrected to `` `None` ``. Flagged by `teco`'s
> coordination doc (`docs/plans/llm-provider-config-coordination.md`, "Design phase closed"
> section) as a residual minor from the Pass 3 gate, routed to `architect` ahead of Landing 2.

---

## 1. Goal & scope

**Goal.** Replace falkor-chat's four independently-constructed, env-var-configured LLM clients
with **one internal model-resolution seam** (`ModelGateway`) fed by **two hand-edited config
files**: a pristine, unmodified OpenCode `opencode.json` (providers only) plus a falkor-chat
overlay (per-kind defaults, per-model settings, later roles). Every consumer — the `@mention`
responder, the llm-guard judge, the workflow executor's agent nodes, the embedding worker (and
the agent-node retrieval tool) — resolves its model through that one seam, and any of them may
name its own model.

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

## 2. Context & findings (verified 2026-08-10; re-verified at v2)

### 2.1 Where the four consumers are built today

All are constructed in `server/falkorchat/app.py::_build_default_app` (`:244`), each reading
module constants from `server/falkorchat/config.py`:

| # | Consumer | Constructed at | Client | Config read |
|---|---|---|---|---|
| 1 | `@mention` AI responder | `app.py:270-273` → `responder.AgentResponder` | `LMStudioLLM()` + `LMStudioEmbedder()` | `config.LLM_BASE_URL`, `config.LLM_MODEL` (default args, `llm.py:99-100`) |
| 2 | llm-kind guard judge | `app.py:288` → `app._build_llm_judge(LMStudioLLM())` | `LMStudioLLM()` | same |
| 3 | executor agent nodes | `app.py:290` → `WorkflowExecutor(llm=LMStudioLLM())`, used at `executor.py:585` | `LMStudioLLM()` | same |
| 4 | embedding worker | `app.py:269-271` → `EmbeddingWorker(repo, LMStudioEmbedder())` | `LMStudioEmbedder()` | `config.EMBEDDING_BASE_URL`, `config.EMBEDDING_MODEL` (`embedding.py:59-60`), `config.EMBEDDING_DIM` (`:84`) |
| 4b | **agent-node retrieval** | `app.py:287` → `tools.build_builtin_registry(services, embedder, …)` (`tools.py:354`) → `GraphragRetrieveTool` (`tools.py:250`) | the **same** `LMStudioEmbedder` instance, bound at construction (`tools.py:258`), called at `tools.py:292` | inherits the above |

Three distinct `LMStudioLLM()` instances exist today and every one is the *same* model. Row **4b**
was mis-booked in v1 as a "type hint only" change; it is a **fifth binding site of a real LLM
consumer** and is corrected throughout this revision (M-3).

### 2.2 The HTTP layer is duplicated and has no timeout

`llm._urllib_transport` (`llm.py:78-90`) and `embedding._urllib_transport` (`embedding.py:39-51`)
are byte-identical copies, both calling `urllib.request.urlopen(req)` **with no `timeout=`**.
FR-14's per-model timeout has nowhere to land until this is unified. The injectable `Transport`
alias — `(url, payload) -> dict` — is the seam every offline test uses; keeping that signature
stable is a hard constraint (**38** `llm=` injection sites across 8 test files and **24**
`guard_judge=` sites, counted by the analyst; ~15 `LMStudioLLM(...)` constructions in
`test_llm.py`).

### 2.3 LM Studio's `/v1` prefix — live-verified, and the failure is silent on **both** paths

Probed against the running LM Studio on this box; the embeddings rows were added by the analyst
and re-confirmed here:

| Request | HTTP | Body |
|---|---|---|
| `POST /v1/chat/completions` `{}` | **400** | `{"error": {"message": "No models loaded…"}}` — `error` is an **object** |
| `POST /chat/completions` `{}` | **200** | `{"error":"Unexpected endpoint or method. (POST /chat/completions)"}` — `error` is a **string** |
| `POST /v1/embeddings` `{}` | **400** | `{"error":"No models loaded…"}` — a **string** here |
| `POST /embeddings` `{}` | **200** | `{"error":"Unexpected endpoint or method. (POST /embeddings)"}` |
| `GET /models` **and** `GET /v1/models` | both **200** | the real model list |
| `http://192.168.0.69:1234` (the stakeholder's declared `baseURL`), root **and** `/v1` | connection failure (`000`) | not reachable from WSL right now |

Three consequences that shape the design:

1. **A missing `/v1` is not an HTTP error.** It is a `200` carrying an error envelope, so today's
   `resp["choices"][0]["message"]["content"]` (`llm.py:110`) raises a bare `KeyError: 'choices'`.
   FR-10's "fails loudly" must include *body-level* error detection.
2. **The `error` value is a string on one path and an object on another, on the same server.** A
   classifier written as `body["error"]["message"]` raises `TypeError` in exactly the case the
   check exists to diagnose. Detect on **key presence**, then render (§4.9).
3. **`GET /models` is not a discriminator** (root answers 200 too), so no probe can auto-detect
   the prefix. Normalization must be a declared rule (§4.3).

The two real sample files disagree on the suffix: `opencode/agents/severino/opencode.json` has
`http://localhost:1234/v1`; `~/.config/opencode/opencode.json` has `http://192.168.0.69:1234`.
falkor-chat must accept **both**, unmodified (FR-1/AC-1).

### 2.4 The stakeholder's real shared file, read

One provider `lmstudio`, `npm: "@ai-sdk/openai-compatible"`, `options.baseURL:
"http://192.168.0.69:1234"`, **no `apiKey`**, one model entry `google/gemma-3n-e4b`. LM Studio on
`localhost:1234` currently serves seven models including `qwen/qwen3-4b-2507` and
`mistralai_ministral-3-3b-instruct-2512` — **none of which appear in the file's `models` map**.
This settles a design question (§4.4): the map is *metadata*, not an allow-list.

### 2.5 The model-reference grammar is already fixed by a real file

`opencode/agents/severino/opencode.json` names its agent's model
`"lmstudio/qwen/qwen3-4b-2507"` — provider `lmstudio`, model id `qwen/qwen3-4b-2507`. A ref
**splits on the first `/` only**. A ref with **no** `/` is reserved for Landing 2's roles (FR-7)
and is therefore rejected in Landing 1 with a forward-looking message.

### 2.6 `TraceEvent`s are debug-only — FR-8 cannot live on them

`executor._drive_loop:390` selects the tracer: `tracer = self._tracer if run["trace"] else
_NULL_TRACER`; `NullTracer.record` (`:147-152`) writes nothing. A non-debug run writes **zero**
`TraceEvent`s, while AC-4/AC-6/AC-9/AC-10 all read "the trace shows the concrete model" for
ordinary runs. So the resolved model must be a durable **`StepRun` property**. Written by
`executor._record` (`:821`) → `repo.record_step_and_advance(...)`, one atomic query per step
(§12.2). Per adjudication **A-1** the names are `StepRun.resolvedModel` (graph property),
`resolved_model=` (repository kwarg), `resolvedModel` (API field), plus `modelSource`. **v3:**
a fourth property, `modelFallback` (repository kwarg `model_fallback=`), rides the same query —
`-graph.md`'s v2 answer to m-5, adopted in full (P2-B; see §7 L2-1/L2-2, §8.1).

### 2.7 Publish-time validation seam (FR-9) — corrected placement

`services._validate_def_spec` (`services.py:776`) is a `@staticmethod` called once, from
`publish_workflow_def` (`:906`). Its docstring establishes a house rule: **new invariants run
LAST**, so an older check keeps failing for its own reason.

`publish_workflow_def` runs, verified in the tree: `_validate_def_spec` (`:906`) → serialize
(`:909-921`) → `read_def_structure` (`:922`) → **`_check_no_structural_conflict`** (`:923`) →
`repo.publish_def` (`:932`).

v1 placed the model-resolvability pass "immediately after `_validate_def_spec` returns" and
claimed that preserved the "last" rule. **It did not** — it would sit *ahead* of K-034's conflict
check, so re-publishing a def whose topology changed *and* whose step names an unresolvable model
would return a 400 about the model instead of K-034's 409 "topology is immutable", and the 409 is
the diagnostically important one (it tells the author to publish a new version, not to fix a
name). **Corrected: the pass goes immediately *before* `self._repo.publish_def(...)` (`:932`),
after `_check_no_structural_conflict`.** Nothing is written either way, so the move is free.

`_normalize_opaque` (`services.py:187`) is the existing helper for reading a step `config` /
transition `guard` that may arrive as a dict (service/MCP callers) or a JSON string (REST). The
new pass must use it, or REST-published defs escape the check — the exact M-7 defect it exists for.

### 2.8 Where a consumer names its model

- **workflow step** → `Step.config.model` — opaque serialized string parsed app-side (DESIGN §6.1,
  AGENTS.md rule 8). **No DDL.**
- **llm guard** → `{"kind":"llm","text":…,"model":…}` in `TRANSITION.guard` — same opacity.
- **embedding** → the `embedding` kind default, or an explicit ref at the wiring site.
- **agent** (the chat participant) → §4.6.

### 2.9 FR-20 blast radius, **re-derived** (M-1)

> **Method note — why v1 missed a site.** v1's scan was extension-filtered
> (`grep -r --include="*.py" --include="*.sh" --include="*.yaml" --include="*.md" …`) and
> therefore **structurally could not match a dotfile with no extension**. The re-derivation used
> no `--include` at all, excluding only `.git/`, `.venv/`, `__pycache__/` and `docs/archive/`.
> Any future blast-radius claim in this component must use the unfiltered form.

v1 asserted the four variables are *"only ever read as defaults in `config.py`"*. **That was
wrong.** `server/.env.example` sets all four, and its own header (`:5`) instructs *"copy this file
to `.env` (and `source` it) … if you run uvicorn by hand"* — so with the AC-13 tripwire in place
the **shipped example becomes a startup brick**: follow the documented instructions and the server
refuses to start.

| Site | Change |
|---|---|
| **`server/.env.example:20,21,30,31`** | **(M-1, new)** replace the four lines with the two file-path variables; keep `FALKORCHAT_EMBEDDING_DIM` (survives, §4.5); update the header comment |
| `server/falkorchat/config.py:39,41,48,49` | delete `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `LLM_BASE_URL`, `LLM_MODEL`; add the two file-path vars; add the AC-13 tripwire |
| `server/falkorchat/llm.py:99-100` | default args referencing the deleted constants |
| `server/falkorchat/embedding.py:59-60` | same |
| `server/falkorchat/app.py:244-303` | docstring + the whole wiring block |
| `server/tests/test_llm.py:53-54` | asserts against `config.LLM_BASE_URL` / `config.LLM_MODEL` |
| `server/tests/test_workflow_live.py:11,137` | docstring + skip message |
| `scripts/start_server.sh` (header `:6-35`, defaults `:75`, export block `:153-163`) | document/export the two file-path vars |
| `compose.yaml` (`services.server.environment`) | **does not set any of the four today** — it *gains* the two paths, a read-only bind mount of the shared file, and a `host.docker.internal` note. (The coordination record lists `compose.yaml` as a site that *sets* them; it does not. Relayed to `teco` for correction.) |
| `README.md:133-138, 205, 256-266` · `falkor-chat/AGENTS.md:78` + key-scripts table | run instructions. (`README.md:138,266` reference only `FALKORCHAT_EMBEDDING_DIM`, which survives.) |
| `docs/plans/local-model-ram-budget-ml.md` (**`Status: active`**, owner `data-scientist`) | 8 references incl. a literal `FALKORCHAT_LLM_MODEL=` env block (`:98,105,166,167,173,373,379`). **Not rewritten by this plan** — it is another agent's active method note. Routed to `teco`: it needs a dated amendment noting the env-var mechanism was replaced by K-042, applied by its owner. Named here so it is designed, not discovered. |
| `docs/DESIGN.md` §1.3, §14.2, new §14.8 | the M2 stack table and the config seam |

`FALKORCHAT_EMBEDDING_DIM` is **not** in FR-20's list and survives — with a narrowed role (§4.5).

---

## 3. Design: the one seam (FR-4)

```
                     ┌───────────────────────────────────────────────┐
  opencode.json ──▶  │ modelconfig.py                                │
  (pristine,         │   ProviderCatalog  ← parse + {env:}/{file:}    │
   FR-1/FR-11)       │   Overlay          ← defaults · models · roles │
                     │   ModelGateway     ← resolve(kind, requested,  │
  models.json  ──▶   │                          ws, overrides)        │
  (falkor-chat       │                    ← .llm(...) / .embedder(...)│
   overlay)          └───────┬───────────────────────────────────────┘
                             │ ResolvedModel(ref, base_url, model, key, timeout, params)
                             ▼
                     ┌───────────────────────────────────┐
                     │ transport.make_http_transport()   │  timeout + headers + loud errors
                     └───────┬───────────────────────────┘
                             ▼
        OpenAICompatibleLLM / OpenAICompatibleEmbedder
                             ▲
      ┌──────────────┬───────┴────────┬──────────────────┬────────────────┐
   responder      judge            executor         embedding worker   retrieval tool
   (kind=agent)   (kind=guard)     (kind=step)      (kind=embedding)   (kind=embedding)
```

**The FR-4 rule, in one sentence:** *every LLM consumer holds a `ModelGateway` and asks it for a
client; a directly-injected client is sugar that `__init__` wraps into a `StaticModelGateway`, so
there is exactly one internal path and zero consumers reading endpoint/model settings privately.*

That clause is what keeps the 38 `llm=` / 24 `guard_judge=` test injections working untouched:
`WorkflowExecutor(llm=stub)` becomes, inside `__init__`,
`self._models = models or StaticModelGateway(llm=llm)`. The `llm=` keyword remains **dependency
injection for tests**, never a configuration route — it reads nothing from `config.py` or any
file. The gate adjudicated this faithful to FR-4 (A-4), with **two closures now specified**:

- **`StaticModelGateway.resolve(kind, *, requested=None, ws=None, overrides=None)`** ignores
  `requested` (a static client cannot honour it) and logs a **WARNING once per `(kind, ref)`**
  naming the ref and the fact that a statically-injected client is in use. Without this, AC-4
  passes under a real gateway and silently regresses to one model under any `llm=` wiring.
- **An enforcement test** (L1-4): no module outside `falkorchat/modelconfig.py` and `tests/`
  constructs `OpenAICompatible{LLM,Embedder}`. This makes the FR-4 invariant executable rather
  than aspirational.

### 3.1 Kinds (FR-6) and how each resolution point reaches the workspace

Four kinds, fixed and closed. **Adding a fifth consumer kind means adding its override property**
or the new kind silently escapes FR-17's hard cap — recorded here and relayed to `tico` (§9.3).

| kind | consumer | resolution point | carries `ws`? |
|---|---|---|---|
| `agent` | `AgentResponder` | `maybe_respond(ctx, …)` (`responder.py:82`) | ✔ `ctx.ws` |
| `step` | executor `type:'agent'` nodes | `_run_agent_node(ctx, …)` (`executor.py:539`) | ✔ `ctx.ws` |
| `embedding` | `EmbeddingWorker` | `embed_message(ws, …)` (`embedding.py:86`) | ✔ `ws` |
| `embedding` | `GraphragRetrieveTool` (M-3) | `run(arguments, *, ctx, run)` (`tools.py:289`) | ✔ `ctx.ws` |
| `guard` | the llm-guard judge | `guards.evaluate_guard(…)` (`guards.py:181`) → `judge(...)` (`app.py:388`) | ✘ **none — see §4.10** |

Resolving **per call** rather than at construction is a Landing-1 decision taken so Landing 2's
workspace override (a function of `ws`) needs no signature changes.

---

## 4. Design decisions & rationale

### 4.1 Two files, two env vars — **no home-directory default** (M-2)

| Variable | Meaning | Default |
|---|---|---|
| `FALKORCHAT_OPENCODE_CONFIG` | the pristine, shared `opencode.json` (FR-1/FR-2/FR-11) | **none — required when a consumer is wired** |
| `FALKORCHAT_MODEL_CONFIG` | falkor-chat's own overlay (FR-11) | `<falkor-chat>/config/models.json` (in-repo, committed) |

Named after their *formats*. Both are read **once, at wiring time** (FR-15); no reload path.

**v1 defaulted the shared path to `~/.config/opencode/opencode.json`. That default is removed.** A
product default pointing into one specific user's home is the "works on my box" failure mode, and
AC-13's whole posture is explicitness. The convenience default belongs in the **dev script**, not
the product: `scripts/start_server.sh` sets
`FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"`,
so the stakeholder's day-to-day command is unchanged while a bare `uvicorn` run must be explicit.

**Absence rules.** Both files are required *only when an LLM consumer is actually wired* —
`config.ENABLE_AGENT` or `config.WORKFLOW_ENABLED`. With both off (the library default) the
gateway is never constructed and importing `falkorchat.app` stays network-free and file-free.
When a consumer *is* wired and a file is missing, unreadable or malformed, startup fails naming
the **variable**, the **resolved path**, and the shipped example.

> **This is not the same as "pytest is unchanged", which v1 claimed and which was false (M-2).**
> `server/tests/test_app.py:159-162` monkeypatches `config.ENABLE_AGENT = True` and calls
> `_build_default_app()`; `:181-195` does the same for `WORKFLOW_ENABLED`. Both genuinely build
> the clients (they stub `create_app`, not the construction), so under Landing 1 both would call
> `ModelGateway.from_env()`. The fixture strategy is a **done-condition on L1-4**, and the
> done-condition is stated as *"the full suite passes on a machine with no
> `~/.config/opencode/opencode.json`"*. The repair must **not** be "make a missing file non-fatal
> when it looks like a test" — that would quietly defeat AC-13.

**Malformed** = not valid JSON, or a known key with the wrong shape. **Unknown top-level keys are
accepted and logged**, not rejected — this is what keeps an OpenCode file with `agent.*`,
`$schema`, `mcp`, `permission` blocks acceptable unmodified (AC-1), and what stops a Landing-1
build from rejecting an overlay that already carries `roles`. `roles` and `agents` get a named
"parsed but not honoured until Landing 2" log line in Landing 1.

### 4.2 What is read from the shared file

Only `provider.<id>`:

```
provider.<id>.options.baseURL   → the endpoint
provider.<id>.options.apiKey    → the credential (after {env:}/{file:} substitution)
provider.<id>.options.headers   → extra headers, if present
provider.<id>.npm               → hint for `protocol` (§4.7)
provider.<id>.name              → display only
provider.<id>.models.<id>.*     → metadata only (§4.4)
```

Everything else — `agent`, `mcp`, `permission`, `$schema`, `theme` — is ignored without comment.
falkor-chat **writes nothing** to this file, ever.

### 4.3 The `/v1` normalization rule (AC-1) — with validation, ordering, and visibility (A-3, M-6)

> **Rule, in three ordered steps.**
> 1. **Validate.** After `{env:}`/`{file:}` substitution, require
>    `urlparse(base).scheme in {"http","https"}` **and** a non-empty `netloc`. Otherwise raise
>    `ModelConfigError` naming the provider, the file and the offending value. *(Load time — not
>    call time.)*
> 2. **Strip** any trailing `/` from the URL (all of them, not one).
> 3. **Normalize.** Re-parse; if the path component is now empty, append `/v1`. Otherwise use it
>    verbatim.

Stripping **before** normalizing is what makes step 3 total; v1 wrote the strip as a later step,
which a literal transcription turns into `http://host:1234//v1`. Every row below was re-derived
with `urllib.parse.urlparse` on this box:

| declared `baseURL` | scheme | netloc | path | resolved API base |
|---|---|---|---|---|
| `http://localhost:1234/v1` (severino sample) | `http` | `localhost:1234` | `/v1` | unchanged |
| `http://192.168.0.69:1234` (stakeholder's file) | `http` | `192.168.0.69:1234` | `` | `…:1234/v1` |
| `http://host:1234/` | `http` | `host:1234` | `/` | `http://host:1234/v1` |
| `https://api.openai.com/v1` | `https` | `api.openai.com` | `/v1` | unchanged |
| `https://api.anthropic.com` | `https` | `api.anthropic.com` | `` | `…/v1` ✔ its real prefix |
| `https://api.anthropic.com/v1/` | `https` | `api.anthropic.com` | `/v1/` | `…/v1` |
| `http://gw.lan/openai/v1` (a proxy) | `http` | `gw.lan` | `/openai/v1` | unchanged |
| `192.168.0.69:1234` (typo) | `''` | `''` | `192.168.0.69:1234` | **rejected at load** |
| `localhost:1234/v1` (typo) | `localhost` | `''` | `1234/v1` | **rejected at load** |

The last two rows are why step 1 exists: dropping `http://` is the commonest `baseURL` mistake,
`urlparse` gives both a **non-empty path**, and without validation both would sail through startup
and take the "verbatim" branch.

**Escape hatch, always available:** the overlay may set `providers.<id>.baseURL`, which wins
outright over the shared file and over the rule. That is how an admin overrules the heuristic
**without editing the shared file** (FR-11 intact).

**Visibility (A-3 gap 3).** This rule makes falkor-chat interpret the shared file *differently
from OpenCode*: `@ai-sdk/openai-compatible` appends `/chat/completions` to `baseURL` verbatim and
infers no `/v1`, so on the stakeholder's own file falkor-chat POSTs to `…:1234/v1/…` while
OpenCode POSTs to `…:1234/…`. That divergence is precisely what makes AC-1 pass — a literal
reading yields an unusable provider — so it is the right call, but it must not be invisible.
**Emit one INFO line per provider at startup** naming the declared `baseURL`, the resolved API
base, and which of {the `/v1` rule, an overlay override, verbatim} produced it.

### 4.4 A ref resolves on its **provider**, not on the `models` map

`lmstudio/qwen/qwen3-4b-2507` resolves iff provider `lmstudio` is declared. The model id is **not**
checked against `provider.lmstudio.models`. Rationale in §2.4: LM Studio serves seven models while
the file lists one; an allow-list would fail AC-1 against the real file and force the admin to
edit the shared file every time they load a model — the exact duplication this feature deletes. An
**unknown provider** is unresolvable and fails per FR-9/FR-10.

Consequence, stated plainly: a typo'd *model id* is caught at call time (FR-10, a loud provider
error naming provider + model + URL), not at publish time. A typo'd *provider* is caught at
publish time (FR-9). This asymmetry is deliberate and belongs in the QA test plan.

### 4.5 `EMBEDDING_DIM` (FR-19's fault line)

Today `EmbeddingWorker._dim` comes from `config.EMBEDDING_DIM` (env, default 1536), and
`repository.set_embedding` re-validates with the same value (`repository.py:690`). Under this plan:

1. The overlay may declare `models."<ref>".dim` for an embedding model — **authoritative when
   present**.
2. `FALKORCHAT_EMBEDDING_DIM` remains the fallback, so every existing script and the `ws:test`
   dim-4 estate keep working unchanged.
3. `scripts/bootstrap_schema.sh`'s own `EMBEDDING_DIM` is untouched — DDL-time input for
   vector-index creation, a different thing from the model's output width.
4. Landing 2 (FR-19) adds the third number: the workspace's **frozen index dimension**, read from
   FalkorDB. That query belongs to `-graph.md`. **Scope correction (m-4):** the per-write check
   gates on **the label being written**. Nothing in the app writes `Chunk.embedding` today —
   `grep -rn "Chunk" server/falkorchat/*.py` returns exactly one comment (`config.py:32`) — so
   checking `Chunk`'s index on the `Message` write path would turn a divergent-but-unused index
   into a refusal to embed messages the workspace would happily accept. `Chunk` belongs in the
   **startup assertion** (an operator warning), not the write gate.

### 4.6 How the chat **agent** names its model (FR-5, agent axis)

The overlay carries an optional `agents` map keyed by `agentId`:

```json
"agents": { "assistant": "lmstudio/qwen/qwen3-4b-2507" }
```

Resolution for kind `agent`: `agents[<agentId>]` → `defaults.agent`. **Chosen over an `Agent.model`
graph property** because it needs no DDL, no new query, no `reference`/workspace sync, it is
unit-testable offline, and "configuration is hand-edited files" is an explicit requirement. The
rejected alternative is recorded in §8.4 as an open question for `graph-dba`.

Landing-1 note: the `agents` map is Landing 2; Landing 1 resolves kind `agent` straight to
`defaults.agent`. The key is reserved and parsed-but-ignored in Landing 1, like `roles`.

### 4.7 Provider protocol (FR-13, and what Landing 1 does *not* build)

`ResolvedModel` carries `protocol: "openai" | "anthropic"`, from the overlay's
`providers.<id>.protocol`, else inferred from the shared file's `npm` (`@ai-sdk/anthropic` →
`anthropic`; anything else → `openai`). **Only `openai` is implemented.** A resolution yielding
`protocol="anthropic"` raises `ModelConfigError` at startup naming the provider and the reason — a
declared seam, never a wrong-shaped payload.

This still satisfies FR-13's three provider kinds: LM Studio (`openai`), a second LAN
OpenAI-compatible host (`openai`), and hosted cloud — OpenAI natively, **and Anthropic via its
documented OpenAI-SDK-compatibility base URL `https://api.anthropic.com/v1/`** (a beta layer that
does not support extended thinking, prompt caching, PDFs or citations). A native Messages-API
client is a follow-up (K-043 if wanted), not a Landing-2 obligation.

### 4.8 `{env:}` / `{file:}` substitution (FR-12)

Applied to **every string value inside the `provider.*` subtree** of the shared file and the
overlay's `providers.*` subtree — which covers every place a credential can appear today.
Occurrences are replaced in place (embedded, not just whole-value). `{file:~/...}` expands `~` and
strips trailing whitespace. An **unresolved** `{env:NAME}` or an unreadable `{file:...}` is a
**startup error naming the variable/path and the file it appeared in** — never an empty string,
because an empty `Authorization: Bearer ` reaches a cloud provider as a 401 with no local
diagnosis.

**Secret hygiene, enforced in code:** the credential is held in a `Secret` wrapper whose
`__repr__`/`__str__` render `***`; `ResolvedModel.__repr__` never contains it; no log line, trace
payload or error message interpolates it. A done-condition, not a guideline (L1-2, and `devops`'
U6 review).

### 4.9 Loud failure at call time (FR-10) — the ordered exception ladder (B-2)

v1 listed the failure classes as *"1. transport/connection failure (`URLError`, timeout);
2. HTTP error status (`HTTPError`)"*. **Both halves were defective.** Verified on this box's
interpreter (`server/.venv/bin/python`, 3.12):

```
HTTPError <: URLError : True          →  a `URLError`-first ladder makes the HTTPError branch DEAD CODE
TimeoutError <: URLError: False       →  a read timeout ESCAPES a URLError-only ladder
TimeoutError MRO      : TimeoutError → OSError → Exception
URLError <: OSError   : True
```

The timeout hole is not hypothetical — **this plan creates it**: §2.2 establishes there is no
timeout today and FR-14/L1-1 add one. And FR-18 inherits it: L2-1's chain advances on
`ProviderCallError`, so a **hung** endpoint — the single most likely reason to want a fallback —
would not trigger the chain at all, while AC-9's scripted "endpoint unreachable" case passed.

> **`make_http_transport` catches in exactly this order**, each raising `ProviderCallError` naming
> **provider · model · resolved URL · cause**:
>
> 1. `urllib.error.HTTPError` — **first**, because it is a subclass of `URLError`. Include the
>    response body (truncated); on a cloud 401 that body is the only thing that says *why*.
> 2. `urllib.error.URLError` — connection refused, DNS failure, unknown URL type.
> 3. `(TimeoutError, OSError)` — the bare read-phase timeout and any other socket error.
>    `TimeoutError` **must be named explicitly**; it is not a `URLError`.
> 4. `ValueError` — belt and braces for malformed-URL shapes (see §11.2 for what this does and
>    does not catch).
> 5. **Body-level:** a `200` whose decoded JSON object carries an `error` key. Detect on **key
>    presence**, then render: `msg = err.get("message", str(err)) if isinstance(err, Mapping)
>    else str(err)`. The same server returns a **string** on the wrong-prefix path and an
>    **object** on the right one (§2.3), so `body["error"]["message"]` would `TypeError` in
>    exactly the case this check exists for (m-1).
> 6. **Body-shape:** a `200` body that is not a JSON object.
>
> The client layer adds a seventh: a well-formed body missing `choices` / `data`.

Inside a workflow run this reaches `executor._drive`'s M-1 fault net, which stamps `fail_run` with
the message (`_fail_with_note`, `executor.py:848`) — the run terminates `failed` with a readable
cause and `AT_STEP` cleared, never a `running` zombie and never another model.

> **Settled by the stakeholder (2026-08-10):** FR-10/AC-8's "suspends" is implemented as
> **`failed`-with-cause**. Not reopened.

### 4.10 The `guard` kind's workspace carrier — the B-1 fix

**v1's §6.1 item 6 was false.** Three of four kinds carry the workspace to their resolution point
(§3.1); `guard` does not, and the naive fix lands inside the SHA-locked `_drive_loop`
(recomputed by the analyst as **`71055f756280`** — the lock is live on this tree). Verified:

- `guards.evaluate_guard(guard, *, ctx, run, step_output, thread, judge)` — its `ctx` is the
  **run ctx dict**, not a `CallContext`; the call site passes `ctx=run_ctx` (`executor.py:805-808`).
- `_select_transition(transitions, run, run_ctx, result)` (`executor.py:769`) has no `CallContext`.
- `repository.get_run`'s projection (`repository.py:1470-1495`) returns twelve fields and **`ws` is
  not among them** — it cannot be; the workspace *is* the graph key.

**The chosen carrier: the `run` dict, stamped in `_drive`.** I evaluated the gate's suggestion and
adopt it, because the tree makes it stronger than the review states — the dict *already* travels
the whole way with **no signature change at any level**:

| step | site | in the lock? | change |
|---|---|---|---|
| 1 | `_drive` (`executor.py:339`) stamps `run["ws"] = ctx.ws`, and in L2 `run["modelOverrides"] = <per-drive read>`, before `self._drive_loop(ctx, run)` | **outside** | **the only new code** |
| 2 | `_drive_loop` forwards `run` to `_select_transition` (`:405-407`) | inside | **none — already does** |
| 3 | `_select_transition` forwards `run` to `evaluate_guard` (`:805-808`) | body outside | **none — already does** |
| 4 | `evaluate_guard` passes `run=run` to the judge | outside | one conditional kwarg |
| 5 | the production judge resolves `models.llm("guard", requested=guard.get("model"), ws=run.get("ws"), overrides=run.get("modelOverrides"))` | outside | new |

**Zero edits inside `_drive_loop`, so no lock reopen and no SHA recompute.**

**Thread safety.** `run` is a fresh dict per drive — `run()` and `resume()` each build it from
`self._repo.get_run(...)` (`executor.py:304`, `:331`) immediately before calling `_drive`. It is a
per-drive local, never shared, so stamping it is safe under the anyio worker threadpool
(`api.py:105`, sync routes + sync `BackgroundTasks`) and under `mcp.py:71`'s daemon
thread-per-message. **This is also the answer to the review's "inbound carrier is unsolved" gap:
the `run` dict is the lock-free carrier for the per-drive `WorkspaceOverrides` read. Do not store
it on `self` — the executor is a process-wide singleton and that would be a cross-run data race.**

**Step 4's zero-churn tactic.** Adding an unconditional kwarg would break the 24 stub judges,
which take `(condition, *, understanding, recent_turns, ctx, step_output)`. So `evaluate_guard`
passes `run=run` **only when the judge advertises it**: `if getattr(judge, "accepts_run", False)`.
The production judge becomes a small callable **object** with `accepts_run = True` instead of a
closure; every stub lacks the attribute and is called exactly as today. This is an explicit
capability flag, not signature-sniffing, and it is testable in both directions.

---

## 5. New and changed files

| File | Landing | What |
|---|---|---|
| `server/falkorchat/transport.py` | 1 | **new** — `make_http_transport(*, timeout, headers, opener)`, `ProviderCallError`, the §4.9 ladder. |
| `server/falkorchat/modelconfig.py` | 1 | **new** — `Secret`, `ProviderSpec`, `ResolvedModel`, `Resolution`, `ProviderCatalog`, `Overlay`, `ModelGateway`, `StaticModelGateway`, `ModelConfigError`, `ModelResolutionError`. Pure/offline. **Named `modelconfig.py`, not `models.py`** (m-8): in a FastAPI codebase `models.py` reads as "pydantic/ORM models", which this repo already calls `schemas.py`, and "model" here means a third thing. |
| `falkor-chat/config/models.json` | 1 | **new, committed** — the shipped overlay (per-kind defaults + per-kind timeouts; no secrets). |
| `falkor-chat/config/opencode.example.json` | 1 | **new, committed** — a documented shared-file example (LM Studio + a LAN host + a cloud provider using `{env:}`). |
| `server/.env.example` | 1 | **(M-1)** replace the four FR-20 lines with the two file paths; update the header. |
| `server/falkorchat/config.py` | 1 | remove the four FR-20 constants; add `OPENCODE_CONFIG_PATH`, `MODEL_CONFIG_PATH`, `assert_no_legacy_model_env()`. |
| `server/falkorchat/llm.py` | 1 (+2, L2-1) | `LMStudioLLM` → `OpenAICompatibleLLM(base_url, model, *, transport=None, params=None)`; required args; drop `_urllib_transport`; **add `model: str \| None = None` to `ChatResult`** (M-5). **v3, landed at L2-1 alongside `FallbackClient`:** add `fallback: bool \| None = None` to `ChatResult` — `-graph.md`'s `modelFallback` carrier (P2-B), set `True` iff the answering chain element's index is `> 0`, `None` on a length-1 chain. |
| `server/falkorchat/embedding.py` | 1 | `LMStudioEmbedder` → `OpenAICompatibleEmbedder`; `EmbeddingWorker(repo, embedder=None, *, models=None, expected_dim=None)` resolving per call. |
| `server/falkorchat/tools.py` | 1 | **(M-3) a real change, not a type hint** — `GraphragRetrieveTool` takes the gateway (or an `embedder_for(ws)` callable) instead of a bound `Embedder`, resolving inside `run()`, which already has `ctx.ws`. `build_builtin_registry` signature follows. |
| `server/falkorchat/app.py` | 1 | wiring through one gateway; `_build_llm_judge(models)` returns an object with `accepts_run = True`. |
| `server/falkorchat/executor.py` | 1 | `models=` param; `self._models = models or StaticModelGateway(llm=llm)`; per-step resolution in `_run_agent_node`; `_drive` stamps `run["ws"]` (§4.10). |
| `server/falkorchat/guards.py` | 1 | forward `model=` to the judge only when the guard declares one; forward `run=` only when `judge.accepts_run`. |
| `server/falkorchat/responder.py` | 1 | hold the gateway; resolve per call. |
| `server/falkorchat/services.py` | 2 | model-resolvability pass **before `repo.publish_def`** (§2.7). |
| `server/falkorchat/repository.py` | 2 | `record_step_and_advance` gains `resolved_model=` / `model_source=` / **`model_fallback=`** (v3, P2-B) — **shape owned by `-graph.md`**. |

New tests: `server/tests/test_transport.py`, `server/tests/test_modelconfig.py`.
Extended: `test_llm.py`, `test_embedding.py`, `test_app.py`, `test_executor_agent.py`,
`test_guards.py`, `test_responder.py`, `test_tools.py`, `test_services.py`, `test_workflow_live.py`.

---

## 6. Landing 1 — the seam, the files, the cutover

**Requirements:** FR-1..FR-6, FR-11..FR-15, FR-20. **Acceptance reachable:** AC-1, AC-4 (the
"each step calls its own model" half), AC-5, AC-12, AC-13; AC-2/AC-3 structurally (§10).
**Demonstrable on its own:** yes.

| # | Unit | Files | Done when |
|---|---|---|---|
| **L1-1** | **One HTTP transport with timeout + the §4.9 ladder.** `make_http_transport(*, timeout: float, headers=None, opener=urllib.request.urlopen) -> Transport`; `ProviderCallError(RuntimeError)`. Bind timeout/headers at construction so the `Transport` alias `(url, payload) -> dict` is **unchanged**. Implement the ladder in the §4.9 order — `HTTPError` before `URLError`, `(TimeoutError, OSError)` named explicitly, `ValueError`, then the two body classes with the string-or-object `error` renderer. Delete both `_urllib_transport` copies. | `transport.py` (new), `llm.py`, `embedding.py` | `test_transport.py` has **one case per ladder branch**, driven through an injected `opener`: an opener raising `HTTPError` (asserting the branch is reached **and** the body reaches the message — the dead-code regression); one raising `URLError`; **one raising a bare `TimeoutError`**; one raising `ValueError`; a 200 with `error` as a **string**; a 200 with `error` as an **object**; a 200 that is not an object. Each asserts provider · model · URL in the message. `test_llm.py`/`test_embedding.py` green with only the import changed. |
| **L1-2** | **`modelconfig.py` — parse, validate, substitute, merge, resolve.** Shared-file reader (providers only); overlay reader (`providers`/`defaults`/`models`; `roles`/`agents` parsed + logged as reserved); `{env:}`/`{file:}` (§4.8) with `Secret`; **`baseURL` scheme/netloc validation at load time** then the strip-then-normalize rule (§4.3) + per-provider override + the per-provider INFO line; ref grammar split-on-first-`/` with a no-`/` ref rejected as "roles are not available until Landing 2"; per-kind defaults; per-model settings — **reserve `timeout`/`dim`/`protocol` and pass every other key through into the request payload** (camelCase → snake_case) so `top_p`, `max_completion_tokens`, `reasoning_effort` need no plan revision (m-3); the `protocol` gate (§4.7). `ModelGateway.from_env()`, `.resolve(kind, *, requested=None, ws=None, overrides=None) -> Resolution`, `.llm(...)`, `.embedder(...)`. `Resolution` carries `chain: tuple[ResolvedModel, ...]`, `primary = chain[0]` — length 1 in L1 (**FR-18 seam**). `ws`/`overrides` accepted and ignored via a `WorkspaceOverrides` port defaulting to null (**FR-16/FR-17 seam**). `StaticModelGateway` per §3. `ResolvedModel.ref` (**not** `.label` — A-1). | `modelconfig.py` (new) | `test_modelconfig.py` green, entirely offline: both real sample files parse (fixtures in `tests/data/`); **every row of the §4.3 table**, including the two rejected typo rows and both trailing-slash rows; `{env:}` resolves, is absent→error, and never appears in `repr()`; unknown top-level keys accepted; a no-`/` ref rejected with the Landing-2 message; four differing per-kind defaults each resolve; `dim` honoured; an unknown passthrough key reaches the payload. `python -c "import falkorchat.modelconfig"` with no env set reads no file and touches no network. |
| **L1-3** | **Generalize the clients + per-model settings.** `OpenAICompatibleLLM(base_url, model, *, transport=None, params=None)` and `OpenAICompatibleEmbedder(base_url, model, *, transport=None)` — `base_url`/`model` **required**; `params` merged into the chat payload. `ChatResult` gains `model: str \| None = None` (frozen dataclass, additive and default-safe) and the client populates it with the `ref` that answered — the FR-8 carrier (M-5). No back-compat aliases. | `llm.py`, `embedding.py`, `test_llm.py`, `test_embedding.py` | The 3 default-arg constructions (`test_llm.py:23,43,57`) and the `config.LLM_*` assertions (`:53-54`) updated to explicit values; new tests pin that `params` reach the payload, that omitting them sends neither key, and that `ChatResult.model` carries the answering ref. Suite green. |
| **L1-4** | **Rewire all five consumer bindings onto the gateway (FR-4/FR-5/FR-6).** `WorkflowExecutor.__init__(..., models=None, llm=None)` → `self._models = models or StaticModelGateway(llm=llm)`; the "no LLM wired ⇒ empty stub" branch keys off `self._models.has_chat()` (**preserve the deliberate offline-stub affordance — never tidy it into a raise**); `_run_agent_node` resolves `self._models.llm("step", requested=config.get("model"), ws=ctx.ws)`; **`_drive` stamps `run["ws"] = ctx.ws`** (§4.10 — the L2 seam, landed now). `guards.evaluate_guard` passes `model=` only when the guard declares one, and `run=` only when `judge.accepts_run`. `app._build_llm_judge(models)` returns an object with `accepts_run = True`, resolving kind `guard` per evaluation. `AgentResponder` holds the gateway (`agent` + `embedding` inside `maybe_respond`). `EmbeddingWorker(repo, embedder=None, *, models=None, expected_dim=None)` resolving inside `embed_message(ws, …)`. **`GraphragRetrieveTool` takes the gateway and resolves inside `run()`** (M-3). `app._build_default_app` builds **one** `ModelGateway.from_env()`. | `executor.py`, `guards.py`, `responder.py`, `embedding.py`, `tools.py`, `app.py` | All 38 `llm=` and 24 `guard_judge=` injections pass **unmodified**. **A `conftest.py` autouse fixture points `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` at the `tests/data/` fixtures**, and the done-condition is *"the full suite passes on a machine with no `~/.config/opencode/opencode.json`"* — `test_app.py:159,181` included (M-2). **FR-4 enforcement test:** no module outside `modelconfig.py`/`tests/` constructs `OpenAICompatible{LLM,Embedder}` (A-4). New offline tests: two steps naming different models call different URLs/model ids (AC-4); an llm-guard naming its own model hits that model; a guard **without** one passes no `model=`; the four defaults may differ and each is used (AC-5); the retrieval tool resolves through the gateway; a per-model `timeout` reaches the opener (AC-12); `StaticModelGateway.resolve(requested=…)` logs the WARNING once per `(kind, ref)`. |
| **L1-5** | **FR-20 cutover + AC-13 tripwire.** Delete the four constants. Add `config.assert_no_legacy_model_env()`: if any of the four is **set**, raise naming them and pointing at the two files — called from `ModelGateway.from_env()`. Update **`server/.env.example`** (M-1), `scripts/start_server.sh` (header, defaults incl. the `$HOME/.config/...` convenience default per §4.1, export block), `compose.yaml` (the two paths + a read-only bind mount of the shared file + `host.docker.internal:host-gateway`, since the container cannot reach `localhost:1234`), `README.md`, `falkor-chat/AGENTS.md`. Ship `config/models.json` + `config/opencode.example.json`. | `config.py`, `server/.env.example`, `scripts/start_server.sh`, `compose.yaml`, `README.md`, `AGENTS.md`, `config/*.json`, `test_workflow_live.py:11,137` | **The unfiltered scan** — `grep -rn -e FALKORCHAT_LLM_BASE_URL -e FALKORCHAT_LLM_MODEL -e FALKORCHAT_EMBEDDING_BASE_URL -e FALKORCHAT_EMBEDDING_MODEL .` excluding `.git/`, `.venv/`, `__pycache__/`, `docs/archive/` and the K-042 documents — returns **only** the tripwire's own list. **No `--include` filter** (§2.9 method note). Copying `.env.example` to `.env`, sourcing it and running uvicorn **starts successfully**. Setting a legacy var fails with the tripwire message (AC-13). |
| **L1-6** | **Docs, in the same change.** `docs/DESIGN.md`: §1.3 — the stack table's model rows become *the shipped defaults in `config/models.json`*; §14.2 — the layering box gains `modelconfig.py` + `transport.py`; **new §14.8 "The model-resolution seam"** (the §3 diagram, the four kinds + five bindings, the two files, the §4.3 rule, the §4.10 guard carrier, the FR-4 rule sentence); §14.7 — a hazard bullet: *a wired agent now requires two config files; the suite supplies them by fixture and must pass with no `~/.config/opencode/`*. `docs/HISTORY.md`: one dated Landing-1 entry with suite counts. `docs/BACKLOG.md`: K-042 → 🟡. Relay to `teco`: `docs/plans/local-model-ram-budget-ml.md` needs its owner's amendment (§2.9); `docs/manuals/llm-provider-config.md` is now worth writing (`tico`). | `docs/DESIGN.md`, `docs/HISTORY.md`, `docs/BACKLOG.md` | No document still describes model choice as an env var. §14.8 exists and §1.3 points at it. |

### 6.1 Seams Landing 1 must leave open (corrected — v1 item 6 was false)

1. `Resolution.chain` is a tuple, `primary = chain[0]` — FR-18 becomes a wrapper swap, not a
   signature change. **Sound.**
2. `resolve(..., ws=…, overrides=…)` exists and is threaded from every call site from day one —
   FR-16/FR-17 swaps a `NullWorkspaceOverrides` for the graph-backed one. **Sound for `agent`,
   `step` and both `embedding` bindings.**
3. A ref **without** a `/` is rejected with a Landing-2-aware message — the role namespace stays
   unclaimed. **Sound.**
4. Overlay keys `roles` and `agents` are reserved, parsed, and logged as not-yet-honoured — an
   admin's Landing-2-ready file never fails on a Landing-1 build. **Sound.**
5. `ResolvedModel.ref` (`"<provider>/<model>"`) is populated in Landing 1 and is exactly the string
   FR-8 persists as `StepRun.resolvedModel` — Landing 2 adds the write, not the value.
6. **~~Every resolution point already has `ctx.ws`/`ws` in scope.~~ FALSE — corrected.**
   **Three of four kinds carry the workspace; `guard` does not** (`evaluate_guard`'s `ctx` is the
   run ctx dict, `_select_transition` has no `CallContext`, `get_run` does not project `ws`). The
   `guard` kind is carried on **the `run` dict**, stamped by `_drive` outside the SHA lock —
   see §4.10. **Landing 1 lands the `run["ws"]` stamp and the `accepts_run` judge protocol** so
   Landing 2 adds only the override read.
7. **The inbound carrier for the per-drive `WorkspaceOverrides` read is the same `run` dict**
   (§4.10). Not `self` — the executor is a process-wide singleton and every drive runs on the
   anyio threadpool or an `mcp.py` daemon thread, so `self` would be a cross-run data race.

---

## 7. Landing 2 — roles, precedence, traceability, guards

**Requirements:** FR-7..FR-10, FR-16..FR-19. **Acceptance:** AC-6..AC-11 (+ AC-4's trace half).

| # | Unit | Files | Done when |
|---|---|---|---|
| **L2-1** | **Roles (FR-7) + ordered fallback chains (FR-18).** Overlay `roles.<name> = {"models": ["<ref>", …], "timeout": …}`. A ref without `/` resolves to a role → a `Resolution` whose `chain` is the ordered, settings-applied list. `FallbackClient`: on `ProviderCallError` from element *n*, try *n+1*; when all fail, raise naming **every** model tried. **The answering model travels on the return value — `ChatResult.model` (L1-3) — never on client state (M-5).** **v3 (P2-B):** `FallbackClient` also sets `ChatResult.fallback = (index of the successful element > 0)` on the same return value — `-graph.md`'s `modelFallback` carrier, computed exactly where the chain-walking already happens, so no new state is introduced to compute it. A role name must not contain `/`; a role resolving to another role is rejected at load. | `modelconfig.py`, `transport.py`, `llm.py` | `test_modelconfig.py`: a role resolves to its first model; re-mapping changes resolution with **no republish** (AC-6); a chain whose first element raises falls to the second and `ChatResult.model` reports the second **and `ChatResult.fallback` is `True`** (AC-9) — **with a case where the first raises `TimeoutError`**, the B-2 hole; a chain of one (no fallback) reports `ChatResult.fallback` as `None`. A test asserts no mutable per-client "last used" state exists. |
| **L2-2** | **Record the model that actually ran (FR-8).** `_run_agent_node` carries the answering ref out on `StepResult`; `executor._record` passes it to `repo.record_step_and_advance(..., resolved_model=…, model_source=…, model_fallback=…)` (**A-1 names**). Durable **`StepRun` property**, not a `TraceEvent` (§2.6). **v3, superseding the v2 text (P2-B):** `modelSource ∈ {workspace, step, default}` names the **precedence rung only**; it is `StepRun.modelFallback` (boolean, nullable, absent by default — same "nullable, absent by default" contract as the other two) that marks whether the winning rung's answer came from a fallback, per `-graph.md` §1.3/§6.2's fully-specified design, **adopted here rather than the documentation-only alternative this plan previously specified**. `StepResult` gains `modelFallback: bool | None = None` alongside `resolvedModel`/`modelSource`, all three defaulting to `None` for non-LLM step types. `modelFallback` is **orthogonal to `modelSource`** — a workspace override can itself resolve to a role with its own fallback chain, so `('workspace', True)` is valid and meaningful, not a contradiction — and is computed by comparing the answering `ChatResult` against the chain the winning rung named (`ChatResult.fallback`, L2-1), never inferred from `modelSource`. **Multi-call rule (m-6):** an agent node loops up to `maxIterations` and can answer on model A then model B; **the last answering model wins** for all three fields — `resolvedModel`, `modelSource` **and `modelFallback`** are overwritten together on each iteration and read once after the loop exits, so a "set once on first resolution" implementation does not silently record the wrong iteration's fallback state. Surface `resolvedModel`, `modelSource` **and `modelFallback`** on `GET /workflow-runs/{id}/step-runs` (matching `-graph.md` §1.7's read projection). | `executor.py`, `repository.py`, `services.py`, `schemas.py`, `api.py` | Two steps on two models produce two different `StepRun.resolvedModel` values on a **non-debug** run (AC-4); the role-swap run shows the new concrete model (AC-6); the fallback run shows the model that answered **and `modelFallback = true`** on that row (AC-9); a non-fallback run's `StepRun.modelFallback` is absent (not `false`); a two-model node records the last iteration's `resolvedModel`/`modelSource`/`modelFallback` together. |
| **L2-3** | **Workspace override + precedence (FR-16/FR-17).** Implement `WorkspaceOverrides` against the storage `-graph.md` specifies. **Read once per drive / per responder call**, in `_drive` (`executor.py:339`, outside the lock), stamped onto `run["modelOverrides"]` (§4.10) — never per resolution, never on `self`. Precedence, first-match-wins: **workspace → the consumer's own choice → the per-kind default**; the workspace is a **hard cap**. Overrides are per-kind (adjudication A-2). | `modelconfig.py`, `executor.py`, `responder.py`, `embedding.py`, `tools.py`, `repository.py` | With an override set, a step that explicitly names a different model runs on the **workspace's** model and `resolvedModel` shows it (AC-10). Unit tests pin all three rungs and the hard-cap direction **for all four kinds — `guard` included**, which is the finding B-1 existed to make possible. |
| **L2-4** | **Publish-time rejection (FR-9).** New instance-level pass in `publish_workflow_def`, **immediately before `self._repo.publish_def(...)` (`services.py:932`), after `_check_no_structural_conflict`** (§2.7 — M-4), using `_normalize_opaque` so REST-published defs are covered: each step's `config.model`; each `{"kind":"llm"}` guard's `model`. Failure → `WorkflowDefSpecError` naming **the step key (or the transition endpoints) and the offending identifier** → 400, nothing written. A `Services` built without a gateway **skips** the pass — **and when it skips while the def declares any model/role, logs a WARNING naming the def and the identifiers** (m-7), so "validation didn't run" is never invisible. | `services.py`, `app.py` | AC-7: publishing a def naming `nope/thing` or an undeclared role fails 400 with the step key and identifier in `detail`; a def naming a valid role publishes. **A test pins that a def failing *both* K-034 topology-immutability and model resolution returns the 409, not the 400** (the M-4 regression). Existing publish tests unaffected. |
| **L2-5** | **Loud use-time failure, run half (FR-10).** Pin: an unresolvable ref at drive time raises `ModelResolutionError`, reaches `_drive`'s M-1 net, terminates the run `failed` with the identifier in `ctx.error`, `AT_STEP` cleared, **no other model used**. For the responder: an ERROR log naming the identifier and **no reply posted**. | `executor.py`, `responder.py` (assertions; likely no behaviour change) | AC-8 pinned at drive level; a second test asserts no fallback model was called. |
| **L2-6** | **Embedding-dimension guard (FR-19).** Before the first write for a workspace, compare the resolved embedding model's declared `dim` (§4.5) against the workspace's frozen vector-index dimension, via the query `-graph.md` specifies. Mismatch → raise `EmbeddingDimensionError` **before** calling the embedder: no vector written, no wasted inference. Cache per `(ws, process)`. **Gate each write on the label being written; `Chunk` belongs in the startup assertion only** (m-4, §4.5). Keep `set_embedding`'s length check as the last line of defence. | `embedding.py`, `repository.py` | AC-11: a mismatch raises naming model, its dim, the workspace and the index dim; the message's `embedding` is null afterwards. `ws:test` (dim 4) keeps working via the declared-dim path. A test pins that a divergent **`Chunk`** index does **not** block a `Message` embed. |
| **L2-7** | **Docs + close.** DESIGN §14.8 gains roles/precedence/trace; `docs/HISTORY.md` Landing-2 entry; BACKLOG K-042 → ✅ + M4 row; hand off to `qa-engineer`. | docs | Every AC in §10 has a row with a verdict or a recorded gate. |

---

## 8. Interface required from `docs/plans/llm-provider-config-graph.md`

This plan does not design any of the following; it states what the resolver needs and defers the
Cypher, schema, index and RAM analysis. **Names below follow adjudication A-1** (`-graph.md`'s
names govern; this plan adopted them).

1. **FR-8 — `StepRun.resolvedModel` (+ `modelSource`, `modelFallback`).** A **durable,
   always-written** property carrying the concrete `"<provider>/<model-id>"` for each executed
   step, written by the existing atomic `record_step_and_advance` (repository kwarg
   `resolved_model=` / `model_source=` / **`model_fallback=`**, v3), plus its read surface and a
   RAM note (AGENTS.md rule 6). It **cannot be a `TraceEvent`** (`executor.py:390` — a non-debug
   run writes none). `modelSource` names the **precedence rung only**; whether that rung's answer
   came from an FR-18 fallback is the fourth, boolean property `modelFallback` (nullable, absent by
   default) — **`-graph.md`'s fully-specified v2 answer to m-5, adopted in full (P2-B)**, superseding
   this plan's earlier documentation-only proposal (logs + debug `TraceEvent`s), because AC-9 is a
   formal acceptance criterion and `TraceEvent`s are debug-only by construction — the identical
   argument that keeps `resolvedModel` itself off a `TraceEvent`. This plan carries the resolver-side
   half of the contract (§7 L2-1/L2-2): `ChatResult.fallback`, set by `FallbackClient` from the
   answering chain index, travels to `StepResult.modelFallback` and then to
   `record_step_and_advance(..., model_fallback=...)`, unchanged through the SHA-locked
   `_drive_loop` boundary exactly as `resolvedModel` already does (§4.10's carrier reasoning applies
   identically — nothing new crosses the lock).
2. **FR-16/FR-17 — workspace override storage and read.** A read shaped
   `get_model_overrides(ws) -> {kind -> ref}` (per-kind; the wildcard is correctly rejected —
   adjudication A-2), cheap enough to call **once per run drive / per responder call**.
   > **⚠️ Changed by B-1 — please relay.** `-graph.md` §2.6 recommends reading the overrides *"at
   > `Executor.run`/`resume` entry, alongside the snapshot read that already happens there"*. The
   > snapshot read is at `executor.py:376-378` — **inside `_drive_loop`, inside the SHA lock**;
   > `run`/`resume` do not read the snapshot. **The correct read site is `_drive`
   > (`executor.py:339`), outside the lock**, and the value is carried on the `run` dict
   > (§4.10), not on `self`. This plan owns that carrier; `-graph.md` need only name the query and
   > the read site.
3. **FR-19 — the workspace's frozen vector-index dimension.** A read returning the dimension of
   `ws:{id}`'s `Message.embedding` index on the pinned build (the analyst credits `-graph.md`
   §3.1/§3.2 for re-probing rather than trusting the quirks KB, which was wrong for this build).
   **Scope note (m-4):** `Chunk.embedding` belongs to the startup assertion, not the per-write
   gate — nothing in the app writes `Chunk` today (`grep -rn "Chunk" server/falkorchat/*.py` → one
   comment, `config.py:32`).
4. **Open question routed there:** should the chat agent's model choice (§4.6) live on the `Agent`
   node instead of the overlay file? This plan chose the file; if the workspace override lands on a
   node that would naturally carry an agent-level setting too, say so and this plan is amended.

---

## 9. Risks, open questions, and what changes for the developer

### 9.1 What Landing 1 changes about the day-to-day run

- **`./scripts/start_server.sh` now requires two files.** The script keeps a
  `$HOME/.config/opencode/opencode.json` convenience default (§4.1), so the stakeholder's command
  is unchanged; a bare `uvicorn` run must set `FALKORCHAT_OPENCODE_CONFIG` explicitly. If the file
  is absent and the agent is enabled, the server **refuses to start**, naming the variable, the
  path and `config/opencode.example.json`. That is FR-20/AC-13 working as specified.
- **`server/.env.example` changes shape.** Copy-to-`.env`-and-source still works; the four model
  variables are replaced by the two file paths (M-1). Before this fix, following the shipped
  instructions would have *bricked startup*.
- **`pytest` needs a fixture, not luck.** Two existing tests build the wired app (M-2); a
  `conftest.py` autouse fixture points both variables at `tests/data/` fixtures. The
  done-condition is that the suite passes on a machine with **no** `~/.config/opencode/`.
- **Changing a model no longer means an env var.** Edit `config/models.json`, restart.
- **⚠️ The stakeholder's declared endpoint does not currently work.**
  `http://192.168.0.69:1234` is unreachable from WSL right now (§2.3), while
  `http://localhost:1234/v1` answers. The first live run will fail at call time with a
  `ProviderCallError` naming that URL — *correct behaviour* — and needs one of: (a) LM Studio bound
  to the LAN interface, (b) editing `options.baseURL` in the shared file, or (c)
  `providers.lmstudio.baseURL` in falkor-chat's overlay (§4.3 escape hatch, shared file stays
  pristine). A demo-readiness item for `devops`/`qa-engineer`.
- **Containers:** `compose.yaml`'s server cannot reach `localhost:1234`; it needs
  `host.docker.internal:host-gateway` (the pattern `cpg/mcp/docker-run.sh` already uses here) or a
  LAN address.

### 9.2 Risks

| Risk | Mitigation |
|---|---|
| The `/v1` heuristic is wrong for some future provider | Per-provider overlay override (§4.3), always available; every row pinned in `test_modelconfig.py`; **and now an INFO line per provider at startup** so the divergence from OpenCode's own interpretation is visible, not silent (A-3). |
| A missing `/v1` returns **HTTP 200** with an error body → today a bare `KeyError` | §4.9 ladder classes 5–6, with the string-or-object renderer (m-1). Live-verified on both the chat and embeddings paths. |
| **A read timeout escapes classification and FR-18's chain never fires** | The explicit `(TimeoutError, OSError)` rung, with a dedicated `test_transport.py` case and an L2-1 fallback case driven by `TimeoutError` (B-2). |
| A malformed `baseURL` fails only at call time | Load-time scheme/netloc validation (§4.3 step 1, M-6). |
| Test churn breaks the 62 existing injection sites | `StaticModelGateway` sugar (§3); the conditional `model=` kwarg and the `accepts_run` flag (§4.10) keep stub judges intact. |
| Secrets leaking into logs / trace payloads / `repr` | `Secret` wrapper; a test asserts the literal secret appears in neither `repr(resolution)` nor a raised `ProviderCallError`; `devops` re-checks at U6. |
| **One global 180 s timeout can pin the anyio threadpool** — every posted message schedules `_safe_embed` (`api.py:97`) on a pool whose default capacity is 40, shared with all sync REST routes; `mcp.py:71` spawns an unbounded daemon thread per message | **Per-kind defaults instead of one global** (m-2): `embedding` ≈ 30 s (short, predictable, on the hot path), `agent`/`step`/`guard` 180 s. Still a strict improvement over today's unbounded wait. Shipped in `config/models.json`, per-model overridable (AC-12). |
| Renaming `LMStudioLLM` breaks a caller | Only in-repo callers exist (§2.9). No alias, deliberately. |
| FR-9 makes publish config-dependent, and silently does not run on an unwired server | Required by AC-7. The pass is skipped without a gateway **and logs a WARNING when it skips a def that declares a model/role** (m-7). |
| `EMBEDDING_DIM` now has three sources | §4.5 fixes precedence; FR-19 (L2-6) turns a mismatch into a loud refusal, gated on the label being written (m-4). |

### 9.3 Open questions

1. **For `tico` (one sentence in FR-16):** the workspace override is implemented as four **per-kind**
   overrides (adjudication A-2 — a faithful refinement, since FR-17's own chain is kind-indexed at
   two of three rungs and FR-5 declares four namers whose model families are not interchangeable).
   Should FR-16 record that "everything" is scoped to the **closed set of consumer kinds**, so a
   future fifth consumer must add its own override property rather than silently escaping the hard
   cap? *(§3.1 declares the set closed, which holds the line today.)*
2. **Agent-model home** — overlay `agents` map (chosen, §4.6) vs. an `Agent` node property. Routed
   to `-graph.md` (§8.4).
3. **Native Anthropic Messages API** — declared out of scope (§4.7). File **K-043** if required.
4. **Manual** — `docs/manuals/llm-provider-config.md` at Landing 1 close. `tico` decides.
5. **`docs/plans/local-model-ram-budget-ml.md`** (`Status: active`, owner `data-scientist`) carries
   8 now-obsolete `FALKORCHAT_LLM_MODEL` references. Routed to `teco` for its owner to amend;
   **not** rewritten by this plan (§2.9).

**Settled, not reopened:** FR-10's "suspends" → `failed`-with-cause (stakeholder, 2026-08-10);
AC-2/AC-3 deferred/model-gated; `resolvedModel`/`modelSource` naming (A-1); per-kind overrides
(A-2); FR-4 satisfied by `StaticModelGateway` (A-4); the `~/.config` product default removed (M-2).

---

## 10. Test strategy

**Altitude and the network-free rule.** DESIGN §14.7 is binding: the default `pytest` run must
stay network-free and FalkorDB-optional. Everything except the live-marked tests is exercised
**offline** through the two seams — the injectable `Transport` and, new, an injectable `opener`
inside `make_http_transport`. The five consumer bindings are covered by asserting **which URL and
which model id** a fake transport received; no live model is needed to prove step A and step B
used different models. **Config files come from `tests/data/` via an autouse fixture** (M-2), never
from the developer's home directory.

**Ordered behaviours to drive (Landing 1)** — usable directly as a red→green list:

1. A `{env:VAR}` `apiKey` resolves; an unset `VAR` raises naming the variable and the file; the
   literal secret appears in **no** `repr`, log or error string.
2. Every row of the §4.3 table maps declared → resolved — **including both trailing-slash rows and
   both rejected typo rows** (`192.168.0.69:1234`, `localhost:1234/v1`).
3. An overlay `providers.<id>.baseURL` beats both the file and the rule; the startup INFO line
   names which source won.
4. `lmstudio/qwen/qwen3-4b-2507` splits into (`lmstudio`, `qwen/qwen3-4b-2507`); a ref with no `/`
   is rejected with the Landing-2 role message.
5. Both real sample files parse unmodified, `agent`/`mcp`/`$schema` blocks and all (AC-1).
6. A model id absent from the provider's `models` map still resolves (§4.4).
7. Four per-kind defaults, all different, each reaching its own consumer (AC-5) — **including the
   retrieval tool** (M-3).
8. Two workflow steps naming different models produce two different `(url, model)` pairs (AC-4,
   Landing-1 half).
9. An llm-guard naming its own model is judged by that model; a guard **not** naming one passes no
   `model=` kwarg; a judge without `accepts_run` receives no `run=` kwarg (the zero-churn contract,
   both directions).
10. A per-kind/per-model `timeout` reaches the opener (AC-12); omitting it uses the kind default.
11. **One `test_transport.py` case per §4.9 ladder branch**, each asserting provider · model · URL:
    `HTTPError` (body preserved — the dead-code regression), `URLError`, **bare `TimeoutError`**,
    `ValueError`, 200 + `error` as a **string**, 200 + `error` as an **object**, 200 not an object.
12. The FR-4 enforcement test: no module outside `modelconfig.py`/`tests/` constructs
    `OpenAICompatible{LLM,Embedder}`.
13. `StaticModelGateway.resolve(requested=…)` ignores the ref and WARNs once per `(kind, ref)`.
14. Setting any legacy env var aborts startup with the tripwire message (AC-13); copying
    `.env.example` → `.env`, sourcing it and starting uvicorn **succeeds** (M-1).
15. The full suite passes with `HOME` pointed at an empty directory (M-2).

**Landing 2** adds: role resolution and re-mapping without republish (AC-6); chain fallback
including a `TimeoutError`-triggered one, with `ChatResult.model` reporting the answering model
**and `ChatResult.fallback` / `StepRun.modelFallback` = `true` on that row, `absent` on a
non-fallback row** (AC-9, v3/P2-B); `StepRun.resolvedModel` on a **non-debug** run (AC-4); a
two-model node recording the last iteration's `resolvedModel`/`modelSource`/`modelFallback`
together (m-6); the three precedence rungs and the hard cap **for all four kinds including
`guard`** (AC-10); publish rejection naming step + identifier, **and the 409-beats-400 ordering
test** (AC-7, M-4); the skipped-pass WARNING (m-7); drive-time unresolvable → `failed` with cause
and no other model called (AC-8); dimension mismatch refusing before embedding with no vector
written, and a divergent `Chunk` index not blocking a `Message` embed (AC-11, m-4).

**Live (`pytest -m live`)** — extend `test_workflow_live.py`: one run whose two steps genuinely hit
two different LM Studio models, asserting the two `StepRun.resolvedModel` values.

**AC → landing → verification map**

| AC | Landing | How verified |
|---|---|---|
| AC-1 | 1 | offline parse of both real files, unmodified |
| AC-2 | 1 | `{env:}` unit-tested; **end-to-end against a hosted provider deferred — no API key** (structurally verified) |
| AC-3 | 1 | LM Studio end-to-end; second LAN host + cloud **structurally verified only** |
| AC-4 | 1 (calls) + 2 (trace) | offline transport assertions; then `StepRun.resolvedModel` |
| AC-5 | 1 | offline, four differing defaults, five bindings |
| AC-6 | 2 | role re-map + restart, no republish |
| AC-7 | 2 | publish → 400 naming step + identifier; 409 still wins on a topology conflict |
| AC-8 | 2 | drive-time failure → run `failed` with cause (settled: `failed`, not `waiting`) |
| AC-9 | 2 | chain fallback incl. `TimeoutError`; `ChatResult.model` reports the answerer, `ChatResult.fallback`/`StepRun.modelFallback` marks the degradation (v3/P2-B) |
| AC-10 | 2 | override beats explicit step choice, all four kinds |
| AC-11 | 2 | refuse before embed, no vector written; `Chunk` not a blocker |
| AC-12 | 1 | timeout reaches the opener |
| AC-13 | 1 | legacy env tripwire; `.env.example` starts clean |

**Suite discipline.** No Cypher changes in Landing 1 ⇒ `./scripts/test_queries.sh` untouched.
Landing 2 touches `record_step_and_advance` and adds two reads ⇒ **`QUERIES.md` + the query suite
must rise with enumerated assertions** (owned by `-graph.md`).

---

## 11. Review disposition (gate Pass 1, `docs/reviews/llm-provider-config.md`)

### 11.1 Adopted

| Finding | Where addressed |
|---|---|
| **B-1** guard kind has no workspace; naive fix hits the SHA lock | §4.10 (new), §3.1 table, §6.1 items 6–7, L1-4, L2-3, §8.2 |
| **B-2** exception ladder: `HTTPError` dead code, `TimeoutError` escapes | §4.9 (rewritten, verified on this interpreter), L1-1, L2-1, §9.2, §10.11 |
| **M-1** `server/.env.example` sets all four vars | §2.9 (re-derived with a dotfile-catching method + method note), §5, L1-5 |
| **M-2** two tests drive `_build_default_app()` with flags on | §4.1 (default removed), L1-4 fixture strategy, §9.1, §10.15 |
| **M-3** `GraphragRetrieveTool` is a real consumer | §2.1 row 4b, §3.1, §5, L1-4, §10.7 |
| **M-4** FR-9 pass placed before K-034's conflict check | §2.7 (corrected + rationale), L2-4 + its regression test |
| **M-5** `FallbackClient.last_used` is shared mutable state | `ChatResult.model` carrier — L1-3, L2-1, L2-2 |
| **M-6** no scheme/netloc validation | §4.3 step 1 + two rejected table rows, L1-2 (see §11.2 for a correction) |
| **A-1** `resolvedModel`/`resolved_model=`/`modelSource`; `.label` → `.ref` | §2.6, §5, L1-2, L2-2, §8.1 |
| **A-2** per-kind override is faithful; closed-kind-set residual | §3.1, §9.3 Q1, L2-3 |
| **A-3** `/v1` rule gaps (validation, ordering, visibility) | §4.3, §2.3 |
| **A-4** FR-4 faithful; two closures | §3 (`StaticModelGateway.resolve` + enforcement test), L1-4 |
| **A-5** inbound carrier unsolved | §4.10, §6.1 item 7 |
| **m-1** `error` string *or* object | §2.3, §4.9 class 5, §10.11 |
| **m-2** one global timeout vs. the threadpool | Per-kind defaults, §9.2, `config/models.json` |
| **m-3** per-model settings key set too closed | L1-2 passthrough rule |
| **m-4** `Chunk` at embed time | §4.5, L2-6, §8.3 |
| **m-5** `modelSource` ≠ fallback marker | Superseded by **P2-B** (§12) — the documentation-only answer recorded here in v2 was replaced by `-graph.md`'s `modelFallback` property, adopted in v3: §2.6, L2-1, L2-2, §8.1 |
| **m-6** one `StepRun`, many calls | L2-2 "last answering model wins" |
| **m-7** FR-9 silently skipped without a gateway | L2-4 WARNING |
| **m-8** `models.py` collides with FastAPI/ORM convention | renamed `modelconfig.py` throughout |
| **m-9** citation drift | `app.py:244`, `executor.py:390`, `services.py:187`, counts 38/24 corrected |

### 11.2 Two partial rebuttals, with evidence

1. **M-6's stated failure mode is wrong in detail; its remedy is right and is adopted.** The
   review says a malformed `baseURL` *"fail[s] at first call with `ValueError: unknown url type`
   from `urlopen` — which is neither an `HTTPError`, a `URLError`, a timeout, nor a body-level
   error, so it escapes the §4.9 taxonomy entirely."* Executed on `server/.venv/bin/python` 3.12:

   ```
   urlopen('192.168.0.69:1234/v1/chat/completions') -> URLError: <urlopen error unknown url type: 192.168.0.69>
   urlopen('localhost:1234/v1/chat/completions')    -> URLError: <urlopen error unknown url type: localhost>
   ```

   It raises **`URLError`**, not `ValueError`, so it would in fact be caught by rung 2 and reported
   with provider · model · URL. The finding's *conclusion* stands on its own merits and is adopted
   in full — validating at **load** time is strictly better than diagnosing at call time, matches
   AC-13's explicitness posture, and the two `urlparse` rows the review supplies are correct
   (re-derived in §4.3). `ValueError` is kept as ladder rung 4 as belt-and-braces, but the plan no
   longer claims it is the observed failure mode.

2. **`compose.yaml` was already correct in v1 and remains unchanged in substance.** The review's
   M-1 parenthetical agrees (*"`compose.yaml` does not currently set any of the four — the
   coordination record's blast-radius list is wrong about that, the plan is right"*). Recorded here
   because the coordination record still carries the error; relayed to `teco` rather than fixed by
   this plan (§2.9).

### 11.3 Not addressed by this plan, by design

`-graph.md` §2.1/§7-Q1's overstated *"FR-19 and FR-16 contradict each other"* reasoning (A-2's
correction) belongs to that document's owner. This plan states only the conclusion it depends on:
**overrides are per-kind**. Likewise `-graph.md` §2.6's mis-located read site is flagged as an
interface change in §8.2 for `teco` to relay — **not edited here**.

---

## 12. Review disposition (gate Pass 2, `docs/reviews/llm-provider-config.md` `## Pass 2`)

Pass 2 re-gated Version 2 end to end (not diffed against Pass-1 memory) against both this plan and
`docs/plans/llm-provider-config-graph.md`. **Verdict: needs changes** — one new blocker
(**P2-B**, this plan's), one new minor (`-graph.md` §6.5's stale A-2 restatement, routed to
`graph-dba`, not touched here). Everything carried forward from Pass 1 — both original blockers,
all six majors, eight of nine minors — was independently re-verified against the live tree, the
running interpreter and the SHA-locked-region boundary, and confirmed genuinely resolved; nothing
there was reopened or needed rework.

### 12.1 P2-B — adopted in full

**Finding:** `-graph.md` v2's answer to Pass-1 m-5 was not the documentation-only note this plan's
v2 still stated in L2-2 — it added a fully-specified fourth `StepRun` property, `modelFallback`
(schema, write query, both read projections, `StepResult` carrier, RAM note, and a binding
resolver contract at `-graph.md` §6.2). This plan's v2 never adopted it: zero hits for
`modelFallback`/`model_fallback` anywhere in the document, confirmed by the reviewer's grep and
independently reproduced before this revision.

**Resolution: adopted** (the reviewer's recommended direction, §1 of its two suggested
resolutions) — the design was already sound and already gated on the graph side; nothing about it
needed re-deciding, only bringing this plan's text into agreement. Landed at:

- §2.6 — the FR-8 property list gains `modelFallback` / `model_fallback=`.
- §5 — the `llm.py` row gains `ChatResult.fallback`; the `repository.py` row gains
  `model_fallback=`.
- §7 L2-1 — `FallbackClient` sets `ChatResult.fallback = (index of the successful element > 0)` at
  the same point it already resolves `ChatResult.model`, so no new state is introduced to compute
  it; done-condition extended to assert `fallback` is `True` on the fallen-back case and
  `None` on a length-1 chain.
- §7 L2-2 — rewritten in place: `StepResult`/`record_step_and_advance` carry `modelFallback` as a
  fourth, nullable, absent-by-default field, orthogonal to `modelSource` per `-graph.md` §6.2's
  binding requirement 4; the m-6 "last answering model wins" rule is restated to cover all three
  fields overwritten together, not just `resolvedModel`/`modelSource`; the read surface
  (`GET /workflow-runs/{id}/step-runs`) gains the field alongside the other two.
- §8.1 — the interface-required section states the full four-property contract and the
  resolver-side carrier chain (`ChatResult.fallback` → `StepResult.modelFallback` →
  `record_step_and_advance(model_fallback=...)`), crossing the SHA lock exactly as `resolvedModel`
  already does — nothing new added at the locked boundary.
- §10 — the Landing-2 behaviour list and the AC-9 row of the AC→landing→verification map both
  name `modelFallback`/`ChatResult.fallback` explicitly.
- §11.1 — the m-5 disposition row now points to this section instead of restating the superseded
  v2 answer.

**Not touched:** `-graph.md` itself, and its §6.5 stale restatement (the paired Pass-2 minor) —
that finding's owner is `graph-dba`; this plan coordinates with it only through §8's interface
section, per the constraint that has held since Pass 1.
