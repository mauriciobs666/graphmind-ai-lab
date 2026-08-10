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

*(to be filled as the interview proceeds)*

## Functional requirements

*(to be filled as the interview proceeds)*

## Out of scope

*(to be filled as the interview proceeds)*

## Acceptance criteria

*(to be filled as the interview proceeds)*

## Open questions

1. What is the smallest thing that can carry a model choice (per workflow step? per agent? per
   consumer kind?), and what happens when it doesn't declare one?
2. Are **embeddings** in scope, or is this chat/completion models only?
3. What should happen when the config file is missing, malformed, or names a model that the
   provider doesn't actually serve?
4. Do the existing `FALKORCHAT_LLM_*` env vars keep working, or are they replaced?
5. Which parts of the OpenCode schema must be honoured, and which may be ignored?

## Decision log

2026-08-10 — What triggered this, and which driver stings more: per-node model choice **and**
single shared config, but **per-node model choice is the bigger pain**.
