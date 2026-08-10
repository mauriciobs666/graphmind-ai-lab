# LLM Provider & Model Configuration — Feature Requirements

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-10

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

## Out of scope

*(to be filled as the interview proceeds)*

## Acceptance criteria

*(to be filled as the interview proceeds)*

## Open questions

1. Where does a per-step model choice live from the admin's point of view, and what does changing
   it cost? (Workflow definitions are topology-immutable — see K-034.)
2. How soon must a config change take effect — restart, or live?
3. What should happen when the config file is missing, malformed, or names a model that the
   provider doesn't actually serve?
4. Do the existing `FALKORCHAT_LLM_*` / `FALKORCHAT_EMBEDDING_*` env vars keep working, or are
   they replaced?
5. Which parts of the OpenCode schema must be honoured, and which may be ignored (e.g. `npm`,
   `agent.*` blocks, tool permissions)?
6. Embedding **dimension** is frozen at vector-index creation. What should happen if the config
   names an embedding model whose dimension doesn't match the workspace's index?

## Decision log

2026-08-10 — What triggered this, and which driver stings more? → Both, but **per-node model
choice stings more** than the duplicated configuration.

2026-08-10 — Which places can name their own model? → **All four** — workflow step, agent, guard,
embedding worker — and additionally: *"we should create an internal abstraction and use it
everywhere"* (one uniform seam, no consumer bypassing it).

2026-08-10 — What happens when a consumer names no model? → **Default per kind** (an agent
default, a guard default, an embedding default), not a single global default and not a hard
failure.

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
