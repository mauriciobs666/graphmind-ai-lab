# LLM Provider & Model Configuration — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** K-042 (M4) · **Last updated:** 2026-08-10

## Intent

As a falkor-chat administrator, be able to declare LLM providers and models **once**, in the
same file format OpenCode already uses (`opencode.json`), point falkor-chat at that file via an
environment variable, and then pick **which model** is used at **each place falkor-chat needs an
LLM** — workflow nodes, agents, and anything else — instead of every consumer sharing one
globally-configured model.

Two drivers, in the stakeholder's order of pain:

1. **Per-consumer model choice (the bigger sting).** Today every LLM consumer is handed the same
   single model. There is no way to say "this workflow node uses the big model, the guard judge
   uses the cheap fast one".
2. **One config, shared with OpenCode.** The same provider/model settings are maintained twice —
   once in `opencode.json`, once in falkor-chat's environment variables. The env var pointing at
   the config file exists so a single file can serve both tools.

## Problem & current state

*(grounded in `server/falkorchat/config.py`, `llm.py`, `embedding.py`, `app.py` — 2026-08-10)*

- Configuration is **environment variables only**, one endpoint + one model per kind:
  - chat: `FALKORCHAT_LLM_BASE_URL` (default `http://localhost:1234/v1`), `FALKORCHAT_LLM_MODEL`
    (default `qwen/qwen3-4b-2507`)
  - embeddings: `FALKORCHAT_EMBEDDING_BASE_URL`, `FALKORCHAT_EMBEDDING_MODEL`,
    plus `FALKORCHAT_EMBEDDING_DIM`
- **Four consumers**, all sharing the single chat model (or the single embedding model):
  1. the `@mention` **AI responder**
  2. the workflow **guard judge** (`llm`-kind guards)
  3. the workflow **executor's agent nodes** (tool-calling)
  4. the **embedding worker**
- There is no notion of a *named provider* and no way for a workflow definition, an agent, or any
  other consumer to name the model it wants.
- OpenCode, by contrast, already declares providers and models in `opencode.json`
  (`provider.<id>` with `options.baseURL` / `options.apiKey` and a `models` map) and lets each
  consumer reference a model as `"<provider>/<model-id>"`.

## User stories

- As an administrator, I want to declare providers and models once in an `opencode.json`-format
  file, so that falkor-chat and OpenCode read the same configuration and I never maintain it twice.
- As an administrator, I want to point falkor-chat at that file with an environment variable, so
  that I can share one file across tools (or keep separate ones) without editing code.
- As an administrator, I want each **workflow step** to name the model it runs on, so that an
  expensive reasoning step and a cheap routine step don't have to share one model.
- As an administrator, I want each **agent** to name its model, so that different agent
  participants in a workspace can run on different models.
- As an administrator, I want each **guard** (the llm-kind judge) to name its model, so that
  judging doesn't silently consume the same model as the agent doing the work.
- As an administrator, I want the **embedding** model declared through the same config, so there
  is one place to look for every model the system uses.
- As an administrator, I want a per-kind default, so that I only name a model where I actually
  want to deviate.

## Functional requirements

- **FR-1** — Providers and models are declared in a configuration file using the **same format as
  `opencode.json`** (`provider.<id>` carrying endpoint/credential options and a `models` map),
  such that a file authored for OpenCode is accepted by falkor-chat unchanged.
- **FR-2** — The path to that file is supplied by an **environment variable**, so one file can be
  shared with OpenCode.
- **FR-3** — A model is referenced by a **single uniform identifier** combining provider and model
  (the OpenCode `"<provider>/<model-id>"` convention), and the same identifier form works
  everywhere a model can be named.
- **FR-4** — **Every** place falkor-chat uses an LLM resolves its model through **one common
  internal mechanism** — no consumer reads endpoint/model settings by its own private route, and a
  future consumer gets the capability by using that mechanism rather than by adding new
  configuration. *(Stakeholder's words: "create an internal abstraction and use it everywhere".)*
- **FR-5** — Each of the following can name its own model, and any of them may omit it:
  - a **workflow step**
  - an **agent**
  - an **llm-kind guard**
  - the **embedding** consumer
- **FR-6** — When a consumer names no model, it resolves to the **default for its kind** (an agent
  default, a guard default, an embedding default, …), declared in the same configuration.
- **FR-7** — A consumer may name **either** a concrete model (`"<provider>/<model-id>"`) **or** a
  **role** — a stable name declared in the configuration and mapped there to a concrete model.
  Roles exist so that a model can be swapped by editing the shared configuration, without
  republishing a workflow definition (workflow defs are topology-immutable — K-034).
- **FR-8** — Given roles and concrete names coexist, it must be possible to determine **which
  concrete model actually ran** for any given execution, without re-deriving it by hand from the
  configuration. The resolved concrete model is recorded on the **workflow run's execution trace**,
  so it remains accurate even after the configuration is later changed.
- **FR-9** — A workflow definition that names a model or role which the configuration cannot
  resolve is **rejected at publish time**, with an error naming the offending step and identifier.
- **FR-10** — An unresolvable model encountered at **use time** fails loudly — the run suspends
  (or the reply fails) with an error stating what could not be resolved. It never silently falls
  back to another model.
- **FR-11** — The shared OpenCode file stays a **valid, unmodified `opencode.json`** — falkor-chat
  adds no keys to it. falkor-chat's own extras (roles, per-kind defaults) live in a **second,
  falkor-chat-specific file**, also located by environment variable, that layers on top of the
  providers declared in the shared file.
- **FR-12** — Credentials are never required to be written literally into the shared file:
  OpenCode's `{env:VAR}` and `{file:path}` substitution is honoured wherever a value can appear.
- **FR-13** — Three provider kinds must work: a **local OpenAI-compatible endpoint** (LM Studio),
  a **second OpenAI-compatible host** on the LAN, and **hosted cloud providers** (OpenAI /
  Anthropic) authenticated with an API key.
- **FR-14** — A model or role may carry **per-model settings** — at minimum request **timeout**,
  plus generation settings such as temperature and max tokens — so that a large slow model and a
  small fast one can be operated under different limits.
- **FR-15** — Configuration is read at **startup**; a change takes effect on server restart. No
  live reload is required.
- **FR-16** — A **workspace** may override model choice for everything running in it.
- **FR-17** — Resolution order is fixed and first-match-wins: **workspace override → the
  step/agent/guard's own choice → the per-kind default.** The workspace override is a **hard cap**:
  it beats an explicit per-step choice. FR-8 is what makes this safe — the trace shows the model
  that actually ran, so an overruled step choice is discoverable rather than invisible.
- **FR-18** — A role may declare an **ordered fallback chain** of models. When a call fails
  (endpoint down, error response), the next model in the chain is tried. The execution trace
  records **which model actually answered** (FR-8), so a silent downgrade is never invisible.
- **FR-19** — The embedding consumer **refuses to embed, loudly**, when the configured embedding
  model's vector dimension does not match the target workspace's vector index — rather than
  writing vectors that are silently accepted and then never retrieved.
- **FR-20** — The existing `FALKORCHAT_LLM_*` and `FALKORCHAT_EMBEDDING_BASE_URL`/`_MODEL`
  environment variables are **replaced**, not kept as a fallback. Every place that sets them today
  (`scripts/start_server.sh`, container/compose definitions, docs) is updated in the same change,
  so there is one source of truth.

## Out of scope

- **Editing configuration through a UI or API.** Both files are hand-edited by the administrator.
- **Live / hot reload.** A change requires a server restart (FR-15).
- **Modifying the `opencode.json` schema, or contributing anything back to OpenCode.**
  falkor-chat *reads* that format; it never extends or rewrites the shared file (FR-11).
- **Interpreting OpenCode's `agent.*` blocks, tool permissions, prompts or modes.** Only the
  `provider` declarations (and value substitution) are consumed from the shared file.
- **Cost, token accounting, budgets or rate limiting.**
- **Streaming responses, prompt caching, or any change to how prompts are built.**
- **Re-embedding or migrating existing workspaces** when the embedding model changes. FR-19 only
  refuses a mismatch; it does not repair one.
- **Per-user model choice.** Overrides go as far as the workspace (FR-16), not to individual users.
- **Changing which consumers exist.** This feature configures the four existing LLM consumers; it
  adds no new ones.

## Acceptance criteria

- **AC-1** — Given an existing `opencode.json` authored for OpenCode, when falkor-chat is pointed
  at it by environment variable, then it starts and its providers/models are usable — with **no
  edit to that file**, and OpenCode continues to read it unchanged.
- **AC-2** — Given a provider whose `apiKey` is written as `{env:SOME_KEY}` (or `{file:...}`),
  when the server starts with that variable set, then the provider authenticates and **no literal
  secret appears in either config file**.
- **AC-3** — Given three providers declared — LM Studio on localhost, a second OpenAI-compatible
  host, and one cloud provider — then a model from **each** can be exercised end-to-end.
- **AC-4** — Given a workflow whose step A names one model and step B names another, when the
  workflow runs, then each step's LLM call goes to **its own** model, and the run trace shows the
  two different concrete models.
- **AC-5** — Given an agent, an llm-guard and the embedding worker that name **no** model, when
  they run, then each uses **its kind's default** — and the three defaults may differ.
- **AC-6** — Given a step that names the **role** `reasoning`, when the role's mapping in the
  falkor-chat config file is changed to a different model and the server is restarted, then the
  next run uses the new model **with no workflow republish**, and the trace shows the new concrete
  model.
- **AC-7** — Given a workflow definition whose step names a model or role the configuration cannot
  resolve, when it is published, then publish **fails** with an error identifying the step and the
  unresolvable identifier.
- **AC-8** — Given a model that resolves at publish but fails at call time, when a run reaches it
  and no fallback chain applies, then the run **suspends with an error naming what failed** — and
  no other model is used in its place.
- **AC-9** — Given a role with an ordered fallback chain whose first model's endpoint is
  unreachable, when a step using that role runs, then the next model in the chain answers and the
  trace records **that** model as the one that ran.
- **AC-10** — Given a workspace override, when a step that explicitly names a different model runs
  in that workspace, then the **workspace's** model is used, and the trace shows it (not the
  step's declared choice).
- **AC-11** — Given an embedding model whose dimension differs from the target workspace's vector
  index, when embedding is attempted, then it **fails with a clear message** and **no vector is
  written**.
- **AC-12** — Given a role that declares a request timeout longer than the default, when a slow
  model is called through it, then the call is allowed the declared time rather than the default.
- **AC-13** — Given the old `FALKORCHAT_LLM_MODEL` / `FALKORCHAT_LLM_BASE_URL` /
  `FALKORCHAT_EMBEDDING_BASE_URL` / `FALKORCHAT_EMBEDDING_MODEL` variables are set and no config
  file is provided, then the server does **not** silently run on them — the replacement is
  complete and the failure is explicit.

## Open questions

*(none — stakeholder readback confirmed 2026-08-10)*

## Notes for design (context, not requirements)

- Today's four consumers are constructed directly in `server/falkorchat/app.py`
  (`LMStudioLLM()`, `LMStudioEmbedder()`, `_build_llm_judge(LMStudioLLM())`), each reading module
  constants from `config.py`. FR-4's "one internal mechanism" is aimed squarely at that.
- Workflow-def steps already carry an opaque `config` dict, and workflow runs already persist an
  execution trace in the workspace graph — relevant to FR-5 and FR-8.
- K-034 (topology-immutable defs, silent no-op on property-only edits) is the constraint that
  makes FR-7's roles worth having.
- Where the **workspace** override is stored, and how a workspace-level setting reaches the
  resolver, is a design question (FR-16/FR-17 state only the behaviour and the precedence).

## Decision log

2026-08-10 — What triggered this, and which driver stings more? → Both, but **per-node model
choice stings more** than the duplicated configuration.

2026-08-10 — Which places can name their own model? → **All four** — workflow step, agent, guard,
embedding worker — and additionally: *"we should create an internal abstraction and use it
everywhere"* (one uniform seam, no consumer bypassing it).

2026-08-10 — What happens when a consumer names no model? → **Default per kind** (an agent
default, a guard default, an embedding default), not a single global default and not a hard
failure.

2026-08-10 — Embedding-model dimension mismatch against a workspace's frozen vector index? →
**Refuse to embed, loudly.** No silent degradation.

2026-08-10 — Do the existing `FALKORCHAT_LLM_*` / `FALKORCHAT_EMBEDDING_*` env vars survive? →
**Replaced.** Not a fallback, not deprecated-with-warning — scripts/compose/docs are updated in
the same change.

2026-08-10 — Can a **workspace** override model choice? → **Yes, in scope.**

2026-08-10 — Should a failing model fall back to another? → **Yes — a declared, ordered fallback
chain per role.** (Distinct from FR-10: an *unresolvable* name still fails loudly; a *failing
call* may fall through a chain the admin declared.)

2026-08-10 — Workspace override vs. an explicit per-step model: who wins? → **The workspace
wins** — it is a hard cap. Accepted consequence: an explicit step choice can be overruled; the
trace (FR-8) is what keeps that visible.

2026-08-10 — Which providers must work on day one? → **All three**: LM Studio local, a second
OpenAI-compatible host, and hosted cloud (OpenAI/Anthropic). Secret handling is therefore in scope.

2026-08-10 — When must a config edit take effect? → **Restart is fine.** No live reload, no
on-demand reload endpoint.

2026-08-10 — Where do falkor-chat's extras (roles, per-kind defaults) live? → **A second,
falkor-chat-specific file.** The shared `opencode.json` stays pristine and unmodified; neither
tool's file can break the other.

2026-08-10 — Per-model settings (temperature, max tokens, timeout)? → **In scope**, not deferred.

2026-08-10 — Where do you look to see which concrete model actually ran? → **The workflow run's
execution trace** (not server logs, not a resolved-config dump).

2026-08-10 — What happens when a model name doesn't resolve? → **Reject at publish, fail loudly at
run.** Explicitly *not* a silent fallback to the kind default, and explicitly *not* refusing to
start the server.

2026-08-10 — What should a per-step model swap cost? Concrete model in the def (republish per
swap) vs. a role mapped in the shared config (config edit only)? → **Allow both.** A step may name
a concrete model or a role. Accepted consequence: two ways to express one thing, hence FR-8
(the resolved model must be discoverable per execution).

2026-08-10 — Readback confirmed; scope split into two landings (Landing 1: FR-1..FR-6, FR-11..FR-15, FR-20 — config files, resolver seam, per-kind defaults, per-model settings, env-var cutover. Landing 2: FR-7..FR-10, FR-16..FR-19 — roles, fallback chains, workspace override, resolved-model trace, embedding-dimension guard). No cloud API key available, so AC-2/AC-3 verification is deferred/model-gated; the design still supports them. Tracked as K-042 under milestone M4.
